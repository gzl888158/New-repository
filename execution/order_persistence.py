"""
订单持久化与重启同步模块
========================

解决「程序重启后，内存挂单状态清空，交易所挂单还在，本地状态不同步」导致的
重复下单 / 风控失效问题。

模块组成：
1. PersistedOrderStatus + 固定状态机   : INIT → PENDING → PARTIAL_FILLED → FILLED/CANCELLED/REJECTED
2. OrderStore                          : SQLite WAL 持久化，订单每次状态变动立即落盘
3. TradingGate                         : 下单闸门（同步完成前 / 巡检发现不一致时关闭）
4. OrderStartupSynchronizer            : 启动强制双向比对（本地库 ↔ 交易所挂单）
5. OrderPatrolService                  : 后台巡检协程（3-5 秒轮询核对）

设计要点：
- 状态机是纯规则层（fail-closed），非法流转显式拒绝，绝不静默改状态。
- OrderStore 每次写入都立即 commit（WAL + synchronous=NORMAL），崩溃可恢复。
- 启动时先做 integrity_check + 自动备份，再拉交易所挂单双向比对，比对通过才开放闸门。
- 拉取挂单 API 失败 → 关闭闸门并重试，重试耗尽则硬锁（fail-closed）。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import threading
import time
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger


# ════════════════════════════════════════════════════════════════════
# 1. 订单状态机（固定状态流转，禁止随意改状态）
# ════════════════════════════════════════════════════════════════════

class PersistedOrderStatus(str, Enum):
    """持久化订单状态（与交易所语义对齐的固定状态集合）。"""
    INIT = "init"                        # 已创建，待提交交易所
    PENDING = "pending"                  # 已挂单（交易所 live）
    PARTIAL_FILLED = "partial_filled"    # 部分成交（剩余仍挂单）
    FILLED = "filled"                    # 完全成交（终态）
    CANCELLED = "cancelled"              # 已撤销（终态）
    REJECTED = "rejected"                # 失败/拒绝（终态）


class InvalidPersistedOrderTransition(Exception):
    """非法订单状态流转异常（fail-closed）。"""


# 固定状态流转表：源状态 → 允许到达的目标状态集合
_PERSISTED_TRANSITIONS: Dict[PersistedOrderStatus, Set[PersistedOrderStatus]] = {
    PersistedOrderStatus.INIT: {
        PersistedOrderStatus.PENDING,
        PersistedOrderStatus.REJECTED,
    },
    PersistedOrderStatus.PENDING: {
        PersistedOrderStatus.PARTIAL_FILLED,
        PersistedOrderStatus.FILLED,
        PersistedOrderStatus.CANCELLED,
        PersistedOrderStatus.REJECTED,
    },
    PersistedOrderStatus.PARTIAL_FILLED: {
        PersistedOrderStatus.FILLED,
        PersistedOrderStatus.CANCELLED,
        PersistedOrderStatus.REJECTED,
    },
    # 终态：禁止任何流转
    PersistedOrderStatus.FILLED: set(),
    PersistedOrderStatus.CANCELLED: set(),
    PersistedOrderStatus.REJECTED: set(),
}

_TERMINAL_STATUSES: Set[PersistedOrderStatus] = {
    PersistedOrderStatus.FILLED,
    PersistedOrderStatus.CANCELLED,
    PersistedOrderStatus.REJECTED,
}

# 非终态（未完结）状态集合
_OPEN_STATUSES: Set[PersistedOrderStatus] = {
    PersistedOrderStatus.INIT,
    PersistedOrderStatus.PENDING,
    PersistedOrderStatus.PARTIAL_FILLED,
}

# OKX 订单 state → 持久化状态映射
_OKX_STATE_MAP: Dict[str, PersistedOrderStatus] = {
    "live": PersistedOrderStatus.PENDING,
    "partially_filled": PersistedOrderStatus.PARTIAL_FILLED,
    "filled": PersistedOrderStatus.FILLED,
    "canceled": PersistedOrderStatus.CANCELLED,
    "cancelled": PersistedOrderStatus.CANCELLED,
    "mmp_canceled": PersistedOrderStatus.CANCELLED,
    "failed": PersistedOrderStatus.REJECTED,
}


def _safe_float(value: Any) -> float:
    """安全转换价格/数量为浮点数：None→0，NaN/Inf/非法值一律置 0（fail-closed，拒绝脏数值）。"""
    try:
        result = float(value if value is not None else 0.0)
    except (TypeError, ValueError):
        return 0.0
    if result != result or result in (float("inf"), float("-inf")):
        return 0.0
    return result


def coerce_status(status: Any) -> PersistedOrderStatus:
    """容错归一化：接受 PersistedOrderStatus / str。"""
    if isinstance(status, PersistedOrderStatus):
        return status
    if isinstance(status, str):
        return PersistedOrderStatus(status)
    raise TypeError(f"Expected PersistedOrderStatus or str, got {type(status).__name__}")


def map_okx_state(state_str: str) -> PersistedOrderStatus:
    """把 OKX 返回的 state 字符串映射到持久化状态（未知一律 REJECTED，fail-closed）。"""
    return _OKX_STATE_MAP.get((state_str or "").strip().lower(), PersistedOrderStatus.REJECTED)


def is_terminal_status(status: Any) -> bool:
    return coerce_status(status) in _TERMINAL_STATUSES


def is_open_status(status: Any) -> bool:
    return coerce_status(status) in _OPEN_STATUSES


def can_transition(old: Any, new: Any) -> bool:
    """校验状态流转是否合法（同状态视为幂等，合法）。"""
    old_s, new_s = coerce_status(old), coerce_status(new)
    if old_s == new_s:
        return True
    return new_s in _PERSISTED_TRANSITIONS.get(old_s, set())


def assert_transition(old: Any, new: Any) -> PersistedOrderStatus:
    """执行状态流转裁决，非法流转抛异常（fail-closed）。"""
    new_s = coerce_status(new)
    if not can_transition(old, new_s):
        raise InvalidPersistedOrderTransition(
            f"Illegal persisted order transition: {coerce_status(old).value} -> {new_s.value}"
        )
    return new_s


def fetch_open_orders(okx_client) -> List[Dict[str, Any]]:
    """拉取交易所全部活跃「普通挂单」，失败抛异常（fail-closed）。

    注意：
    - okx_client.get_orders() 内部吞掉异常并返回 []，无法区分「无挂单」与「API 失败」。
      这里改用底层 _make_request 直连原始响应，明确识别 code != "0" 或 None 并抛异常，
      确保 API 失败会触发重试/硬锁。
    - 算法/条件单（止盈止损等）由 conditional_order_manager 独立管理并在其自身启动时
      对账，本模块只负责普通挂单（orders-pending），避免重复追踪与状态语义冲突。
    """
    make_request = getattr(okx_client, "_make_request", None)

    if make_request is None:
        # 极端回退：无底层方法时退回公开接口（无法区分空失败，仅尽力而为）
        try:
            return list(okx_client.get_orders(inst_type="SWAP") or [])
        except Exception as e:
            raise RuntimeError(f"get_orders failed: {e}")

    resp = make_request("GET", "/api/v5/trade/orders-pending?instType=SWAP")
    if resp is None or str(resp.get("code", "")) != "0":
        raise RuntimeError(f"orders-pending API failed: {resp}")

    if isinstance(resp, dict):
        data = resp.get("data") or []
    else:
        data = resp
    return list(data or [])


# ════════════════════════════════════════════════════════════════════
# 2. OrderStore：SQLite WAL 持久化
# ════════════════════════════════════════════════════════════════════

_ORDER_COLUMNS = [
    "trace_id", "exchange_order_id", "cl_ord_id", "algo_id",
    "symbol", "side", "pos_side", "order_type", "price", "quantity",
    "filled_quantity", "status", "strategy", "create_time", "update_time",
    "raw_json",
]


class OrderStore:
    """订单持久化存储（SQLite WAL）。

    - 每次状态变动立即 commit（WAL 模式，synchronous=NORMAL）。
    - 支持启动 integrity_check 完整性校验 + 自动备份。
    - 线程安全（内部 RLock），可被 asyncio 单线程事件循环安全调用。
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        self._open()

    def _open(self) -> None:
        dir_path = os.path.dirname(self._db_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        self._conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            timeout=30,
        )
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_schema()

    def _create_schema(self) -> None:
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                trace_id         TEXT PRIMARY KEY,
                exchange_order_id TEXT,
                cl_ord_id        TEXT,
                algo_id          TEXT,
                symbol           TEXT NOT NULL,
                side             TEXT,
                pos_side         TEXT,
                order_type       TEXT,
                price            REAL,
                quantity         REAL,
                filled_quantity  REAL DEFAULT 0,
                status           TEXT NOT NULL,
                strategy         TEXT,
                create_time      REAL,
                update_time      REAL,
                raw_json         TEXT
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_status ON orders(status)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_exchange ON orders(exchange_order_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_client ON orders(cl_ord_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol)")
        self._conn.commit()

    # ── 写路径 ──────────────────────────────────────────────

    def _upsert(self, order: Dict[str, Any]) -> None:
        """创建 / 快照 / 交易所补录（不经过状态机裁决）。

        已存在同 trace_id 的行时仅更新非状态字段（symbol/side/price/quantity 等），
        绝不覆盖状态机裁决的 status，避免 REPLACE 绕过状态机。
        """
        now = time.time()
        vals = {
            "trace_id": order.get("trace_id") or self._gen_trace_id(),
            "exchange_order_id": order.get("exchange_order_id") or None,
            "cl_ord_id": order.get("cl_ord_id") or None,
            "algo_id": order.get("algo_id") or None,
            "symbol": order.get("symbol", ""),
            "side": order.get("side", ""),
            "pos_side": order.get("pos_side", ""),
            "order_type": order.get("order_type", ""),
            "price": _safe_float(order.get("price")),
            "quantity": _safe_float(order.get("quantity")),
            "filled_quantity": _safe_float(order.get("filled_quantity")),
            "status": coerce_status(order.get("status", PersistedOrderStatus.INIT)).value,
            "strategy": order.get("strategy", ""),
            "create_time": float(order.get("create_time") or now),
            "update_time": float(order.get("update_time") or now),
            "raw_json": json.dumps(order.get("raw_json") or {}, ensure_ascii=False, default=str),
        }
        with self._lock:
            existing = self._conn.execute(
                "SELECT trace_id FROM orders WHERE trace_id = :trace_id",
                {"trace_id": vals["trace_id"]},
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    f"INSERT INTO orders ({', '.join(_ORDER_COLUMNS)}) "
                    f"VALUES ({', '.join(':' + c for c in _ORDER_COLUMNS)})",
                    vals,
                )
            else:
                # 已存在：仅更新允许字段，不覆盖 status/create_time（受控更新，绕过状态机的风险）
                updatable = [c for c in _ORDER_COLUMNS
                             if c not in ("trace_id", "status", "create_time")]
                params = {c: vals[c] for c in updatable}
                params["trace_id"] = vals["trace_id"]
                self._conn.execute(
                    f"UPDATE orders SET {', '.join(c + ' = :' + c for c in updatable)} "
                    f"WHERE trace_id = :trace_id",
                    params,
                )
            self._conn.commit()

    def create_order(self, order: Dict[str, Any]) -> str:
        """创建订单（INIT），返回 trace_id。"""
        order = dict(order)
        order.setdefault("status", PersistedOrderStatus.INIT)
        order.setdefault("trace_id", self._gen_trace_id())
        order.setdefault("create_time", time.time())
        order.setdefault("update_time", time.time())
        self._upsert(order)
        logger.debug(f"[order_persistence] create INIT: {order['trace_id']} {order.get('symbol')}")
        return order["trace_id"]

    def transition_status(self, trace_id: str, new_status: Any,
                          exchange_order_id: Optional[str] = None,
                          filled_quantity: Optional[float] = None,
                          **extra_fields) -> bool:
        """状态流转（状态机裁决），立即落盘。非法流转返回 False 并告警。"""
        new_s = coerce_status(new_status)
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM orders WHERE trace_id = :tid", {"tid": trace_id}
            ).fetchone()
            if row is None:
                logger.warning(f"[order_persistence] transition on missing order: {trace_id}")
                return False
            old_s = row["status"]
            if not can_transition(old_s, new_s):
                logger.error(
                    f"[order_persistence] illegal transition rejected: {trace_id} "
                    f"{old_s} -> {new_s.value}"
                )
                return False

            fields = ["update_time = :update_time"]
            params: Dict[str, Any] = {"update_time": time.time(), "tid": trace_id}
            if exchange_order_id is not None:
                fields.append("exchange_order_id = :exchange_order_id")
                params["exchange_order_id"] = exchange_order_id
            if filled_quantity is not None:
                fields.append("filled_quantity = :filled_quantity")
                params["filled_quantity"] = _safe_float(filled_quantity)
            for key, value in extra_fields.items():
                if key in _ORDER_COLUMNS and key not in ("trace_id", "status", "update_time"):
                    fields.append(f"{key} = :{key}")
                    params[key] = value
            fields.append("status = :status")
            params["status"] = new_s.value

            self._conn.execute(
                f"UPDATE orders SET {', '.join(fields)} WHERE trace_id = :tid", params
            )
            self._conn.commit()
        logger.debug(f"[order_persistence] transition: {trace_id} {old_s} -> {new_s.value}")
        return True

    def set_exchange_order_id(self, trace_id: str, exchange_order_id: str,
                              algo_id: Optional[str] = None) -> bool:
        """下单受理后回填交易所 order_id（INIT → PENDING 时调用）。"""
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM orders WHERE trace_id = :tid", {"tid": trace_id}
            ).fetchone()
            if row is None:
                return False
            params: Dict[str, Any] = {
                "tid": trace_id,
                "exchange_order_id": exchange_order_id,
                "algo_id": algo_id,
                "update_time": time.time(),
            }
            self._conn.execute(
                "UPDATE orders SET exchange_order_id = :exchange_order_id, "
                "algo_id = :algo_id, update_time = :update_time WHERE trace_id = :tid",
                params,
            )
            self._conn.commit()
        return True

    # ── 读路径 ──────────────────────────────────────────────

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        try:
            d["raw"] = json.loads(d.get("raw_json") or "{}")
        except Exception:
            d["raw"] = {}
        return d

    def get(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE trace_id = :tid", {"tid": trace_id}
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_exchange_id(self, exchange_order_id: str) -> Optional[Dict[str, Any]]:
        if not exchange_order_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE exchange_order_id = :oid",
                {"oid": exchange_order_id},
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_by_client_id(self, cl_ord_id: str) -> Optional[Dict[str, Any]]:
        if not cl_ord_id:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM orders WHERE cl_ord_id = :cid", {"cid": cl_ord_id}
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_open_orders(self) -> List[Dict[str, Any]]:
        """查询全部未完结订单（INIT/PENDING/PARTIAL_FILLED）。"""
        placeholders = ", ".join(f"'{s.value}'" for s in _OPEN_STATUSES)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM orders WHERE status IN ({placeholders}) "
                f"ORDER BY create_time ASC"
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_all_orders(self) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM orders ORDER BY create_time ASC").fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_open(self) -> int:
        placeholders = ", ".join(f"'{s.value}'" for s in _OPEN_STATUSES)
        with self._lock:
            row = self._conn.execute(
                f"SELECT COUNT(*) AS c FROM orders WHERE status IN ({placeholders})"
            ).fetchone()
        return int(row["c"]) if row else 0

    # ── 防护：完整性校验 + 自动备份 ──────────────────────────

    def integrity_check(self) -> Tuple[bool, str]:
        """启动完整性校验：返回 (是否通过, 结果说明)。"""
        try:
            with self._lock:
                row = self._conn.execute("PRAGMA integrity_check").fetchone()
                # PRAGMA integrity_check 单行返回 'ok' 即通过，否则逐行错误
                first = row[0] if row else ""
                if str(first).lower() == "ok":
                    return True, "ok"
                rows = self._conn.execute("PRAGMA integrity_check").fetchall()
                detail = "; ".join(str(r[0]) for r in rows[:5])
                return False, detail or "integrity check failed"
        except Exception as e:
            return False, str(e)

    def backup(self, backup_dir: Optional[str] = None) -> Optional[str]:
        """自动备份：使用 sqlite3 backup API 生成一致性快照。返回备份路径或 None。"""
        try:
            backup_dir = backup_dir or os.path.join(os.path.dirname(self._db_path), "backups")
            os.makedirs(backup_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(backup_dir, f"order_store_{ts}.db")
            with self._lock:
                dst = sqlite3.connect(dest)
                try:
                    self._conn.backup(dst)
                finally:
                    dst.close()
            logger.info(f"[order_persistence] backup created: {dest}")
            return dest
        except Exception as e:
            logger.warning(f"[order_persistence] backup failed: {e}")
            return None

    def close(self) -> None:
        try:
            with self._lock:
                if self._conn:
                    self._conn.close()
                    self._conn = None
        except Exception as e:
            logger.warning(f"[order_persistence] close error: {e}")

    @staticmethod
    def _gen_trace_id() -> str:
        import uuid
        return f"ord_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


# ════════════════════════════════════════════════════════════════════
# 3. TradingGate：下单闸门
# ════════════════════════════════════════════════════════════════════

class TradingGate:
    """下单闸门（fail-closed：默认关闭，同步/巡检通过后才打开）。"""

    def __init__(self, initially_open: bool = False):
        self._open = initially_open
        self._reason = "not_synced"
        self._closed_at = time.time()

    def open(self) -> None:
        self._open = True
        self._reason = ""
        logger.info("[order_persistence] trading gate OPENED")

    def close(self, reason: str) -> None:
        self._open = False
        self._reason = reason or "unknown"
        self._closed_at = time.time()
        logger.warning(f"[order_persistence] trading gate CLOSED: {self._reason}")

    def is_open(self) -> bool:
        return self._open

    def is_blocked(self) -> bool:
        return not self._open

    def get_state(self) -> Dict[str, Any]:
        return {
            "open": self._open,
            "reason": self._reason,
            "closed_at": self._closed_at,
        }


# ════════════════════════════════════════════════════════════════════
# 4. 启动强制同步
# ════════════════════════════════════════════════════════════════════

class OrderStartupSynchronizer:
    """启动强制双向比对：

    1. 读取本地库未完结订单
    2. 拉取 OKX 全部活跃挂单（普通挂单 + 条件/算法单）
    3. 双向比对：交易所存在本地不存在 → 补入库；本地挂单交易所已成交/撤销 → 更新状态
    4. 无法对齐的脏订单 → 告警 + 关闭闸门；全部对齐后才打开闸门
    """

    def __init__(self, store: OrderStore, gate: TradingGate, okx_client,
                 alert_manager=None, config: Optional[Dict[str, Any]] = None):
        self.store = store
        self.gate = gate
        self.okx_client = okx_client
        self.alert_manager = alert_manager
        cfg = config or {}
        self._max_retries = int(cfg.get("api_retry_times", 3))
        self._retry_delay = float(cfg.get("api_retry_delay", 2.0))

    def _pull_exchange_open_orders(self) -> List[Dict[str, Any]]:
        """拉取交易所全部活跃挂单（普通 + 算法/条件单）。失败抛异常（fail-closed）。"""
        return fetch_open_orders(self.okx_client)

    def _pull_with_retry(self) -> List[Dict[str, Any]]:
        last_err: Optional[Exception] = None
        for attempt in range(1, self._max_retries + 1):
            try:
                return self._pull_exchange_open_orders()
            except Exception as e:
                last_err = e
                logger.warning(
                    f"[order_persistence] pull open orders failed ({attempt}/{self._max_retries}): {e}"
                )
                time.sleep(self._retry_delay * attempt)
        raise RuntimeError(f"pull open orders exhausted after {self._max_retries} retries: {last_err}")

    async def run(self) -> Dict[str, Any]:
        """执行启动同步。返回结构化结果（供上层决定是否开放闸门）。"""
        result: Dict[str, Any] = {
            "ok": False,
            "local_open": 0,
            "exchange_open": 0,
            "inserted": 0,
            "aligned": 0,
            "dirty": [],
        }

        # 0. DB 完整性校验 + 自动备份
        ok_integrity, detail = await asyncio.to_thread(self.store.integrity_check)
        if not ok_integrity:
            await self._alert("order_db_corrupt", f"订单库完整性校验失败: {detail}", "EMERGENCY")
            self.gate.close("db_integrity_failed")
            result["error"] = f"db integrity failed: {detail}"
            return result
        backup_path = await asyncio.to_thread(self.store.backup)
        if not backup_path:
            logger.warning("[order_persistence] startup backup skipped/failed")

        # 1. 读取本地库未完结订单
        local_open = await asyncio.to_thread(self.store.get_open_orders)
        result["local_open"] = len(local_open)

        # 2. 拉取交易所活跃挂单（失败重试，重试耗尽硬锁）
        try:
            exchange_orders = await asyncio.to_thread(self._pull_with_retry)
        except Exception as e:
            await self._alert("order_api_hardlock", f"拉取挂单API失败，硬锁交易: {e}", "EMERGENCY")
            self.gate.close("order_api_failed")
            result["error"] = str(e)
            return result
        result["exchange_open"] = len(exchange_orders)

        # 构建交易所索引：ordId / clOrdId / algoId
        exchange_by_ord = {}
        exchange_by_cl = {}
        exchange_by_algo = {}
        for od in exchange_orders:
            oid = od.get("ordId") or ""
            clid = od.get("clOrdId") or ""
            aid = od.get("algoId") or ""
            if oid:
                exchange_by_ord[oid] = od
            if clid:
                exchange_by_cl[clid] = od
            if aid:
                exchange_by_algo[aid] = od

        matched_exchange_ids: Set[str] = set()

        # 3a. 本地 → 交易所：更新状态 / 补状态
        for lo in local_open:
            trace_id = lo.get("trace_id", "")
            # INIT 且无交易所 order_id = 重启前已创建但从未受理的订单。
            # 重启后内存执行上下文已丢失，这类订单不可能再被提交交易所，
            # 直接判为 REJECTED 终态，避免被误判为「脏订单」而关闭闸门。
            if (lo.get("status") == PersistedOrderStatus.INIT.value
                    and not lo.get("exchange_order_id")):
                try:
                    await asyncio.to_thread(
                        self.store.transition_status, trace_id, PersistedOrderStatus.REJECTED
                    )
                    result["aligned"] += 1
                    logger.info(
                        f"[order_persistence] init residue -> rejected: {trace_id} {lo.get('symbol')}"
                    )
                except Exception as e:
                    logger.warning(f"[order_persistence] init residue transition failed: {e}")
                continue

            matched = self._find_match(lo, exchange_by_ord, exchange_by_cl, exchange_by_algo)

            if matched is not None:
                od = matched
                oid = od.get("ordId") or ""
                if oid:
                    matched_exchange_ids.add(oid)
                new_status = map_okx_state(od.get("state", ""))
                filled_qty = _safe_float(od.get("accFillSz") or od.get("fillSz"))
                # 直接以交易所为准对齐（状态机裁决；非法则记录 dirty）
                try:
                    assert_transition(lo.get("status", "init"), new_status)
                    await asyncio.to_thread(
                        self.store.transition_status,
                        trace_id, new_status,
                        exchange_order_id=oid or None,
                        filled_quantity=filled_qty,
                    )
                    # 若无交易所 order_id 但本次拿到，补回填
                    if oid and not lo.get("exchange_order_id"):
                        await asyncio.to_thread(self.store.set_exchange_order_id, trace_id, oid)
                    result["aligned"] += 1
                except InvalidPersistedOrderTransition as e:
                    result["dirty"].append({"trace_id": trace_id, "reason": str(e)})
            else:
                # 本地有、交易所无：尝试用 get_order 单点查询其真实终态
                resolved = await asyncio.to_thread(self._resolve_missing_local, lo)
                if resolved is not None:
                    new_status, filled_qty = resolved
                    try:
                        assert_transition(lo.get("status", "init"), new_status)
                        await asyncio.to_thread(
                            self.store.transition_status,
                            trace_id, new_status,
                            filled_quantity=filled_qty,
                        )
                        result["aligned"] += 1
                    except InvalidPersistedOrderTransition as e:
                        result["dirty"].append({"trace_id": trace_id, "reason": str(e)})
                else:
                    # 无法对齐的脏订单
                    result["dirty"].append({
                        "trace_id": trace_id,
                        "symbol": lo.get("symbol"),
                        "exchange_order_id": lo.get("exchange_order_id"),
                        "reason": "unresolved_open_order",
                    })

        # 3b. 交易所 → 本地：交易所存在本地不存在 → 补入库
        for od in exchange_orders:
            oid = od.get("ordId") or ""
            if oid and oid in matched_exchange_ids:
                continue
            # 跳过已被本地通过 clOrdId/algoId 匹配到的单
            if self._already_matched(od, local_open):
                continue
            status = map_okx_state(od.get("state", ""))
            if is_terminal_status(status):
                continue  # 交易所已终态且本地无记录，无需补录
            order = self._exchange_to_order(od, status)
            await asyncio.to_thread(self.store._upsert, order)
            result["inserted"] += 1

        # 4. 判定
        if result["dirty"]:
            reason = f"存在 {len(result['dirty'])} 个无法对齐的脏订单"
            await self._alert("order_dirty", reason, "CRITICAL",
                              metadata={"dirty": result["dirty"]})
            self.gate.close("dirty_orders")
            result["error"] = reason
        else:
            self.gate.open()
            result["ok"] = True

        logger.info(
            f"[order_persistence] startup sync: local={result['local_open']} "
            f"exchange={result['exchange_open']} aligned={result['aligned']} "
            f"inserted={result['inserted']} dirty={len(result['dirty'])} ok={result['ok']}"
        )
        return result

    def _find_match(self, lo: Dict[str, Any], by_ord, by_cl, by_algo):
        eid = lo.get("exchange_order_id")
        if eid and eid in by_ord:
            return by_ord[eid]
        clid = lo.get("cl_ord_id")
        if clid and clid in by_cl:
            return by_cl[clid]
        aid = lo.get("algo_id")
        if aid and aid in by_algo:
            return by_algo[aid]
        return None

    def _already_matched(self, od: Dict[str, Any], local_open: List[Dict[str, Any]]) -> bool:
        oid = od.get("ordId") or ""
        clid = od.get("clOrdId") or ""
        aid = od.get("algoId") or ""
        for lo in local_open:
            if oid and lo.get("exchange_order_id") == oid:
                return True
            if clid and lo.get("cl_ord_id") == clid:
                return True
            if aid and lo.get("algo_id") == aid:
                return True
        return False

    def _resolve_missing_local(self, lo: Dict[str, Any]) -> Optional[Tuple[PersistedOrderStatus, float]]:
        """本地 open 但交易所 pending 里没有：用 get_order 查真实终态。"""
        eid = lo.get("exchange_order_id")
        symbol = lo.get("symbol", "")
        if not eid or not symbol:
            return None
        try:
            detail = self.okx_client.get_order(symbol, eid)
            if not detail:
                return None
            status = map_okx_state(detail.get("state", ""))
            filled_qty = _safe_float(detail.get("accFillSz") or detail.get("fillSz"))
            return status, filled_qty
        except Exception as e:
            logger.warning(f"[order_persistence] get_order({eid}) failed: {e}")
            return None

    def _exchange_to_order(self, od: Dict[str, Any], status: PersistedOrderStatus) -> Dict[str, Any]:
        now = time.time()
        return {
            "trace_id": self._gen_synthetic_trace_id(od),
            "exchange_order_id": od.get("ordId") or "",
            "cl_ord_id": od.get("clOrdId") or "",
            "algo_id": od.get("algoId") or "",
            "symbol": od.get("instId") or od.get("instId", ""),
            "side": od.get("side", ""),
            "pos_side": od.get("posSide", ""),
            "order_type": od.get("ordType", ""),
            "price": _safe_float(od.get("px")),
            "quantity": _safe_float(od.get("sz")),
            "filled_quantity": _safe_float(od.get("accFillSz") or od.get("fillSz")),
            "status": status.value,
            "strategy": od.get("tag", "") or "exchange_orphan",
            "create_time": float(od.get("cTime", 0)) / 1000 if od.get("cTime") else now,
            "update_time": float(od.get("uTime", 0)) / 1000 if od.get("uTime") else now,
            "raw_json": od,
        }

    @staticmethod
    def _gen_synthetic_trace_id(od: Dict[str, Any]) -> str:
        oid = od.get("ordId") or od.get("algoId") or "unknown"
        return f"exch_{oid}"

    async def _alert(self, alert_type: str, message: str, severity: str,
                     metadata: Optional[Dict[str, Any]] = None):
        logger.log("CRITICAL" if severity == "EMERGENCY" else severity,
                   f"[order_persistence] ALERT [{alert_type}]: {message}")
        if self.alert_manager:
            try:
                await self.alert_manager.send_alert(
                    alert_type, message, severity=severity, metadata=metadata or {}
                )
            except Exception as e:
                logger.warning(f"[order_persistence] alert failed: {e}")


# ════════════════════════════════════════════════════════════════════
# 5. 后台巡检协程（3-5 秒轮询）
# ════════════════════════════════════════════════════════════════════

class OrderPatrolService:
    """后台巡检：周期性核对本地库与交易所订单，发现不一致 → 告警 + 暂停新委托。"""

    def __init__(self, store: OrderStore, gate: TradingGate, okx_client,
                 alert_manager=None, config: Optional[Dict[str, Any]] = None):
        self.store = store
        self.gate = gate
        self.okx_client = okx_client
        self.alert_manager = alert_manager
        cfg = config or {}
        self._interval = float(cfg.get("patrol_interval", 4.0))
        self._running = False

    async def run_loop(self):
        self._running = True
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self._patrol_once()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[order_persistence] patrol error: {e}")

    async def stop(self):
        self._running = False

    async def _patrol_once(self):
        try:
            exchange_orders = await asyncio.to_thread(fetch_open_orders, self.okx_client)
        except Exception as e:
            logger.error(f"[order_persistence] patrol pull failed: {e}")
            # API 失败 → 暂停新委托（fail-closed）
            await self._alert("order_api_patrol_failed", f"巡检拉取挂单失败: {e}", "ERROR")
            self.gate.close("patrol_api_failed")
            return

        exchange_by_ord = {od.get("ordId"): od for od in exchange_orders if od.get("ordId")}
        local_open = await asyncio.to_thread(self.store.get_open_orders)

        mismatches: List[Dict[str, Any]] = []
        for lo in local_open:
            status = lo.get("status")
            if status == PersistedOrderStatus.INIT.value:
                # 尚未受理（无 exchange_order_id），不参与交易所比对
                if not lo.get("exchange_order_id"):
                    continue
            eid = lo.get("exchange_order_id")
            if not eid:
                continue
            od = exchange_by_ord.get(eid)
            if od is None:
                # 本地 pending 但交易所已无此单 → 不一致
                resolved = await asyncio.to_thread(self._resolve_missing, lo)
                if resolved is None:
                    mismatches.append({
                        "trace_id": lo.get("trace_id"),
                        "exchange_order_id": eid,
                        "local_status": status,
                        "exchange_status": "missing",
                    })
                else:
                    new_status, filled_qty = resolved
                    if new_status != coerce_status(status):
                        await asyncio.to_thread(
                            self.store.transition_status, lo.get("trace_id"),
                            new_status, filled_quantity=filled_qty,
                        )
            else:
                new_status = map_okx_state(od.get("state", ""))
                if new_status != coerce_status(status):
                    mismatches.append({
                        "trace_id": lo.get("trace_id"),
                        "exchange_order_id": eid,
                        "local_status": status,
                        "exchange_status": new_status.value,
                    })

        if mismatches:
            logger.warning(f"[order_persistence] patrol mismatch: {mismatches}")
            await self._alert("order_state_mismatch",
                              f"巡检发现 {len(mismatches)} 个订单状态不一致",
                              "WARNING", metadata={"mismatches": mismatches})
            self.gate.close("patrol_mismatch")

    def _resolve_missing(self, lo: Dict[str, Any]):
        eid = lo.get("exchange_order_id")
        symbol = lo.get("symbol", "")
        if not eid or not symbol:
            return None
        try:
            detail = self.okx_client.get_order(symbol, eid)
            if not detail:
                return None
            status = map_okx_state(detail.get("state", ""))
            filled_qty = _safe_float(detail.get("accFillSz") or detail.get("fillSz"))
            return status, filled_qty
        except Exception:
            return None

    async def _alert(self, alert_type: str, message: str, severity: str,
                     metadata: Optional[Dict[str, Any]] = None):
        logger.log("WARNING" if severity == "WARNING" else "ERROR",
                   f"[order_persistence] ALERT [{alert_type}]: {message}")
        if self.alert_manager:
            try:
                await self.alert_manager.send_alert(
                    alert_type, message, severity=severity, metadata=metadata or {}
                )
            except Exception as e:
                logger.warning(f"[order_persistence] alert failed: {e}")


__all__ = [
    "PersistedOrderStatus",
    "InvalidPersistedOrderTransition",
    "map_okx_state",
    "is_terminal_status",
    "is_open_status",
    "can_transition",
    "assert_transition",
    "OrderStore",
    "TradingGate",
    "OrderStartupSynchronizer",
    "OrderPatrolService",
]
