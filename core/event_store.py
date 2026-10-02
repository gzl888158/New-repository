"""P1 企业级升级：持久化事件溯源存储（EventStore）—— 企业级强化版

对标 Jane Street / DE Shaw 的事件溯源范式：
- 所有关键动作（信号/下单/成交/平仓/风控拦截）以 append-only JSONL 落盘
- 事件为「真相来源(SSOT)」，进程崩溃/重启后可重放重建历史状态
- 按天滚动 + 保留期清理，兼顾可审计性与磁盘占用

企业级强化（相比基础版新增）：
1. 全局单调序列号 seq —— 每次 append 分配递增 seq，支持断点重放与乱序检测
2. 幂等去重 —— 相同 event_id 重复写入被拒绝，防止重试/重放导致重复事件
3. sha256 完整性校验 —— 每条事件追加确定性校验和，重放时可验证（防篡改/损坏）
4. 损坏行检测与告警 —— 损坏 JSON 行不再静默跳过，累计统计并首次告警
5. 可选 fsync 崩溃强一致 —— fsync_on_append=True 时每批写入刷盘（sacrifice 吞吐换强一致）
6. 可观测 stats() —— 暴露写入/去重/损坏/校验异常统计

设计约束：本模块零业务依赖（不 import unified_layer/risk/execution），可独立单测。
事件 ID 复用 core.event_id 的全局唯一生成器，保证与订单热路径 traceID 同源。

事件 schema（每行一条 JSONL）：
{
  "seq": 123,                         # 全局单调递增序列号（升级前旧事件无此字段）
  "event_id": "evt-xxx-00001",
  "event_type": "order_placed",
  "timestamp": "2026-09-09T19:00:00.123456",
  "source": "event_bus",
  "symbol": "ETH-USDT-SWAP",
  "version": 1,
  "checksum": "sha256-hex",           # 确定性校验和（enable_checksum=True 时写入）
  "data": {...}
}

向后兼容：旧事件（升级前写入，缺 seq/checksum）仍可被 replay/tail/count 正常读取；
seq 缺失时跳过断点过滤，checksum 缺失时跳过校验。
"""

import glob
import hashlib
import json
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from loguru import logger

from core.event_id import EventIDGenerator


class EventStore:
    """持久化 append-only 事件存储（按天滚动 + 保留期 + seq/幂等/校验和）。"""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        retention_days: int = 7,
        flush_every: int = 1,
        fsync_on_append: bool = False,
        enable_checksum: bool = True,
        seen_ids_capacity: int = 20000,
    ):
        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "events",
            )
        self._data_dir = data_dir
        self._retention_days = max(1, retention_days)
        self._flush_every = max(1, flush_every)
        self._fsync_on_append = fsync_on_append
        self._enable_checksum = enable_checksum
        self._seen_ids_capacity = max(1, seen_ids_capacity)

        self._lock = threading.RLock()
        self._id_gen = EventIDGenerator.get_instance()

        self._active_day: Optional[str] = None        # "2026-09-09"
        self._active_file: Optional[str] = None       # 绝对路径
        self._write_count_since_flush = 0

        # 企业级状态：全局序列号 + 幂等去重 seen set
        self._last_seq: int = 0
        self._seen_ids: Set[str] = set()
        self._seen_ids_order: List[str] = []          # FIFO 队列，配合有界淘汰

        # P3 哈希链：上一条事件的 checksum，用于将事件链成「不可篡改」的追加链。
        # 每条事件的 checksum = sha256(prev_checksum + 规范化载荷)，使篡改/删除/重排
        # 任一事件都会破坏后续链路，由 verify_chain 检出。
        self._prev_checksum: str = ""

        # 可观测统计（线程安全：均在 _lock 内更新）
        self._stats: Dict[str, int] = {
            "appended": 0,
            "duplicates_rejected": 0,
            "corrupt_lines": 0,
            "checksum_mismatches": 0,
            "replayed": 0,
        }

        os.makedirs(self._data_dir, exist_ok=True)
        self._init_seq_from_disk()
        self._prune()

    # ── 文件管理 ─────────────────────────────────────────
    def _file_for_day(self, day: str) -> str:
        return os.path.join(self._data_dir, f"events_{day}.jsonl")

    def _open_active_file(self) -> str:
        day = datetime.now().strftime("%Y-%m-%d")
        if day != self._active_day:
            self._active_day = day
            self._active_file = self._file_for_day(day)
            self._write_count_since_flush = 0
        return self._active_file

    def _existing_files(self) -> List[str]:
        return sorted(glob.glob(os.path.join(self._data_dir, "events_*.jsonl")))

    def _init_seq_from_disk(self) -> None:
        """启动时从磁盘恢复全局序列号（只读每文件最后一行，O(文件数)）。

        同时恢复哈希链尾（最后一条有效事件的 checksum），使重启后的新事件
        能无缝续接到既有链上。升级前的旧版（非链式）checksum 无法作为链头，
        长度复位为空串，由 _chain_head_for 识别。
        """
        max_seq = 0
        tail_rec = None
        for path in self._existing_files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    # 倒序找最后一条有效 JSON（跳过末尾空行）
                    lines = f.readlines()
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    seq = rec.get("seq")
                    if isinstance(seq, int) and seq > max_seq:
                        max_seq = seq
                        tail_rec = rec
                    break
            except OSError:
                continue
        self._last_seq = max_seq
        self._prev_checksum = self._chain_head_for(tail_rec) if tail_rec else ""
        if max_seq:
            logger.debug(f"EventStore restored last_seq={max_seq}")

    def _chain_head_for(self, rec: Dict[str, Any]) -> str:
        """计算续链链头：旧版（非链式）校验和无法接入哈希链，复位为空串；新版返回其 checksum。"""
        checksum = rec.get("checksum", "")
        if not checksum:
            return ""
        et = rec.get("event_type", "")
        ts = rec.get("timestamp", "")
        seq = rec.get("seq", 0)
        data = rec.get("data", {}) or {}
        # 旧版（非链式）校验和 → 链路在此复位（连续 append 以空串为新链首）
        if self._compute_checksum_legacy(et, ts, seq, data) == checksum:
            return ""
        return checksum

    # ── 校验和 ─────────────────────────────────────────
    @staticmethod
    def _compute_checksum(
        event_type: str,
        timestamp: str,
        seq: int,
        data: Dict[str, Any],
        prev_checksum: str = "",
    ) -> str:
        """确定性 sha256 校验和（sort_keys=True 保证跨进程一致）。

        哈希链：将上一条事件的 checksum 作为 prev_checksum 链入规范化载荷，
        checksum = sha256(prev_checksum + 规范化{event_type,timestamp,seq,data})。
        篡改/删除/重排任一事件都会使后续 checksum 与链头不一致，由 verify_chain 检出。
        prev_checksum 为空串表示链首（或无校验和旧事件的续接点）。
        """
        payload = json.dumps(
            {
                "prev_checksum": prev_checksum,
                "event_type": event_type,
                "timestamp": timestamp,
                "seq": seq,
                "data": data,
            },
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _compute_checksum_legacy(
        event_type: str,
        timestamp: str,
        seq: int,
        data: Dict[str, Any],
    ) -> str:
        """旧版（非链式）校验和算法，用于识别升级前写入的历史事件。

        升级前的事件 checksum 不含 prev_checksum 字段，无法接入哈希链；
        verify_chain / replay(verify_checksum) 通过本算法识别它们并视为
        「合法但链路复位点」，避免把历史数据误判为篡改。
        """
        payload = json.dumps(
            {"event_type": event_type, "timestamp": timestamp, "seq": seq, "data": data},
            sort_keys=True, ensure_ascii=False, default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # ── 写入 ─────────────────────────────────────────
    def append(
        self,
        event_type: str,
        data: Dict[str, Any],
        *,
        event_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        source: str = "",
        symbol: str = "",
        version: int = 1,
    ) -> str:
        """追加一条事件（线程安全、幂等），返回写入的 event_id。

        幂等语义：event_id 已在近期已写集合中时，拒绝重复写入并返回该 id，
        在 stats["duplicates_rejected"] 累计，不产生重复事件。
        """
        if not event_type:
            raise ValueError("event_type 不能为空")
        if not isinstance(data, dict):
            raise ValueError("data 必须是 dict")

        # 兜底：symbol 未显式传入时，从 data 提取（便于按标的过滤/重放）
        if not symbol:
            symbol = data.get("symbol") or data.get("instId") or ""

        eid = event_id or self._id_gen.generate()
        ts = (timestamp or datetime.now()).isoformat()

        with self._lock:
            # 幂等去重
            if eid in self._seen_ids:
                self._stats["duplicates_rejected"] += 1
                logger.debug(f"EventStore duplicate event_id ignored: {eid}")
                return eid

            self._last_seq += 1
            seq = self._last_seq

            record: Dict[str, Any] = {
                "seq": seq,
                "event_id": eid,
                "event_type": event_type,
                "timestamp": ts,
                "source": source,
                "symbol": symbol,
                "version": int(version),
                "data": data,
            }
            if self._enable_checksum:
                # 哈希链：将上一条 checksum 链入本事件，形成不可篡改的追加链
                record["checksum"] = self._compute_checksum(
                    event_type, ts, seq, data, self._prev_checksum
                )
                self._prev_checksum = record["checksum"]

            line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
            path = self._open_active_file()
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
                self._write_count_since_flush += 1
                if self._write_count_since_flush >= self._flush_every:
                    f.flush()
                    if self._fsync_on_append:
                        os.fsync(f.fileno())
                    self._write_count_since_flush = 0

            # 维护幂等去重 seen set（有界 FIFO 淘汰）
            self._seen_ids.add(eid)
            self._seen_ids_order.append(eid)
            if len(self._seen_ids_order) > self._seen_ids_capacity:
                old = self._seen_ids_order.pop(0)
                self._seen_ids.discard(old)

            self._stats["appended"] += 1
            return eid

    # ── 重放 ─────────────────────────────────────────
    def replay(
        self,
        from_ts: Optional[datetime] = None,
        to_ts: Optional[datetime] = None,
        event_type: Optional[str] = None,
        limit: Optional[int] = None,
        from_seq: Optional[int] = None,
        verify_checksum: bool = False,
    ) -> List[Dict[str, Any]]:
        """按时间升序重放事件，可过滤时间范围/类型/条数/序列号，可选校验和。

        from_seq: 仅返回 seq > from_seq 的事件（断点续放/增量对账）
        verify_checksum: 为 True 时校验每条事件 checksum（无 checksum 的旧事件跳过）
        """
        from_str = from_ts.isoformat() if from_ts else None
        to_str = to_ts.isoformat() if to_ts else None

        result: List[Dict[str, Any]] = []
        # 哈希链链头游标（verify_checksum=True 时跨文件续链）
        chain_prev: List[str] = [""]
        with self._lock:
            for path in self._existing_files():
                self._read_file_into(
                    path, result,
                    from_str=from_str, to_str=to_str,
                    event_type=event_type, limit=limit,
                    from_seq=from_seq, verify_checksum=verify_checksum,
                    chain_prev=chain_prev,
                )
                if limit is not None and len(result) >= limit:
                    break
        return result

    def tail(self, n: int = 50) -> List[Dict[str, Any]]:
        """返回最近 n 条事件（时间升序）。损坏行计入统计并告警。"""
        n = max(0, n)
        if n == 0:
            return []
        with self._lock:
            files = self._existing_files()
            if not files:
                return []
            # 从最新文件倒序读，直到凑满 n 条
            collected: List[Dict[str, Any]] = []
            for path in reversed(files):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except OSError:
                    continue
                for line in reversed(lines):
                    line = line.strip()
                    if not line:
                        continue
                    rec = self._parse_line(line)
                    if rec is None:
                        continue
                    collected.append(rec)
                    if len(collected) >= n:
                        collected.reverse()
                        return collected
            collected.reverse()
            return collected

    def count(self, event_type: Optional[str] = None) -> int:
        """统计事件总数（可按类型过滤）。损坏行计入统计并告警。"""
        total = 0
        with self._lock:
            for path in self._existing_files():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            if event_type is not None:
                                rec = self._parse_line(line)
                                if rec is None:
                                    continue
                                if rec.get("event_type") != event_type:
                                    continue
                            total += 1
                except OSError:
                    continue
        return total

    # ── 内部读取 / 解析 ─────────────────────────────────
    def _parse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """解析单行 JSON；损坏行统计并首次告警，返回 None 表示跳过。"""
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            self._stats["corrupt_lines"] += 1
            if self._stats["corrupt_lines"] == 1:
                logger.warning(
                    f"EventStore detected corrupt line in {self._data_dir} "
                    f"(corrupt_lines will be tracked in stats)"
                )
            return None

    def _verify_record_chain(self, rec: Dict[str, Any], prev_checksum: str):
        """校验单条记录是否接入以 prev_checksum 为链头的哈希链。

        返回 (valid, next_prev)。valid=True 时 next_prev 为该记录 checksum（供下一条续链）；
        无 checksum 的旧事件直接通过且 next_prev 复位为空串（链路在旧事件处重启）。
        """
        checksum = rec.get("checksum")
        if not checksum or not self._enable_checksum:
            return True, ""
        et = rec.get("event_type", "")
        ts = rec.get("timestamp", "")
        seq = rec.get("seq", 0)
        data = rec.get("data", {}) or {}
        expected = self._compute_checksum(et, ts, seq, data, prev_checksum)
        if expected == checksum:
            return True, checksum
        # 升级前的历史事件使用旧版（非链式）校验和：合法，但链路在此复位
        if self._compute_checksum_legacy(et, ts, seq, data) == checksum:
            return True, ""
        self._stats["checksum_mismatches"] += 1
        return False, ""

    def _read_file_into(self, path, result, *, from_str, to_str, event_type, limit,
                        from_seq, verify_checksum, chain_prev=None):
        prev_checksum = chain_prev[0] if chain_prev is not None else ""
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = self._parse_line(line)
                    if rec is None:
                        continue

                    # 断点过滤：seq 缺失（旧事件）时无法过滤，视为通过
                    if from_seq is not None:
                        seq = rec.get("seq")
                        if isinstance(seq, int) and seq <= from_seq:
                            continue

                    ts = rec.get("timestamp", "")
                    if from_str is not None and ts < from_str:
                        continue
                    if to_str is not None and ts > to_str:
                        continue
                    if event_type is not None and rec.get("event_type") != event_type:
                        continue
                    if verify_checksum:
                        ok, prev_checksum = self._verify_record_chain(rec, prev_checksum)
                        if not ok:
                            continue

                    result.append(rec)
                    if limit is not None and len(result) >= limit:
                        return
        except OSError:
            return
        finally:
            if chain_prev is not None:
                chain_prev[0] = prev_checksum

    # ── 可观测性 ─────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        """返回事件存储运行统计（写入/去重/损坏/校验异常/当前序列号）。"""
        with self._lock:
            s = dict(self._stats)
        s["last_seq"] = self._last_seq
        s["data_dir"] = self._data_dir
        return s

    @property
    def last_seq(self) -> int:
        return self._last_seq

    # ── 全链校验（不可篡改） ─────────────────────────
    def verify_chain(self) -> Dict[str, Any]:
        """全链校验：按 seq 顺序重算哈希链，检测篡改/删除/重排。

        从链首（prev=""）开始逐条校验 checksum == sha256(prev_checksum + 规范化载荷)。
        首个不匹配点的链头被污染，其后的记录将无法验证。

        返回 {valid, scanned, verified, mismatches, first_invalid_seq}。
        旧事件（无 checksum）跳过并使链路复位为空串。
        """
        prev_checksum = ""
        scanned = 0
        verified = 0
        mismatches = 0
        first_invalid_seq = None

        with self._lock:
            for path in self._existing_files():
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            rec = self._parse_line(line)
                            if rec is None:
                                continue
                            if not rec.get("checksum"):
                                # 旧事件无校验和，链在此复位
                                prev_checksum = ""
                                continue
                            scanned += 1
                            ok, next_prev = self._verify_record_chain(rec, prev_checksum)
                            if not ok:
                                mismatches += 1
                                if first_invalid_seq is None:
                                    first_invalid_seq = rec.get("seq")
                                prev_checksum = ""  # 断链后不复用污染链头
                                continue
                            verified += 1
                            prev_checksum = next_prev
                except OSError:
                    continue

        return {
            "valid": mismatches == 0,
            "scanned": scanned,
            "verified": verified,
            "mismatches": mismatches,
            "first_invalid_seq": first_invalid_seq,
        }

    # ── 清理 ─────────────────────────────────────────
    def _prune(self) -> int:
        """删除超过保留期的历史事件文件，返回删除文件数。"""
        cutoff = (datetime.now() - timedelta(days=self._retention_days)).strftime("%Y-%m-%d")
        removed = 0
        for path in self._existing_files():
            name = os.path.basename(path)
            day = name.removeprefix("events_").removesuffix(".jsonl")
            if day and day < cutoff:
                try:
                    os.remove(path)
                    removed += 1
                    logger.info(f"EventStore pruned expired file: {name}")
                except OSError as e:
                    logger.warning(f"EventStore failed to prune {name}: {e}")
        return removed

    def prune(self) -> int:
        """手动触发过期清理。"""
        with self._lock:
            return self._prune()

    @property
    def data_dir(self) -> str:
        return self._data_dir