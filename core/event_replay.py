"""P1 企业级升级：事件溯源重放 —— 从事件日志重建订单状态

用途：
1. 崩溃/重启后，重放 ORDER_PLACED / ORDER_FILLED / 平仓事件，重建订单生命周期
   （OrderLifecycleManager 内存状态为易失状态，事件日志为其持久化重建源）
2. 对账检测：识别孤立订单（已下单但未成交 = 可能因崩溃而丢失成交回执的悬单）
3. 为 Dashboard 事件查询 API 提供只读重建视图

设计约束：
- 纯只读，不修改任何业务状态；仅依赖 core.event_store.EventStore 读取
- 零业务依赖（不 import risk/execution/unified_layer），可独立单测
- 与 SQLite trade_records（实际成交 SSOT）互补：本模块回答「订单级生命周期」，
  而非「净持仓」——净持仓由 trade_records + OKX 仓位同步负责
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from core.event_store import EventStore

# 与订单生命周期相关的可重建事件类型（其余事件不参与订单状态重建）
_ORDER_EVENT_TYPES = {
    "order_placed",
    "order_filled",
    "order_cancelled",
    "position_closed",
    "stop_loss_triggered",
    "take_profit_triggered",
}

# 平仓/降风险/撤单事件 → 归类为「关闭」，并标注关闭原因
_CLOSE_EVENT_TYPES = {
    "position_closed": "close",
    "stop_loss_triggered": "stop_loss",
    "take_profit_triggered": "take_profit",
    "order_cancelled": "cancelled",
}


class EventReplayer:
    """基于 EventStore 的订单状态重放与对账器（只读）。"""

    def __init__(
        self,
        event_store: Optional[EventStore] = None,
        data_dir: Optional[str] = None,
    ):
        if event_store is not None:
            self._store = event_store
        else:
            self._store = EventStore(data_dir=data_dir)

    @property
    def store(self) -> EventStore:
        return self._store

    # ── 订单状态重建 ─────────────────────────────────────
    def replay_order_state(
        self,
        from_ts: Optional[datetime] = None,
        to_ts: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """重放订单相关事件，按 exchange_order_id 重建订单生命周期。

        返回 { exchange_order_id: order_view }，每项包含：
          symbol / direction / strategy_name / reduce_only / trace_id
          placed_at / filled_at / closed_at / closed_reason
        """
        events = self._store.replay(from_ts=from_ts, to_ts=to_ts, limit=limit)

        orders: Dict[str, Dict[str, Any]] = {}

        for evt in events:
            etype = evt.get("event_type", "")
            if etype not in _ORDER_EVENT_TYPES:
                continue
            data = evt.get("data", {}) or {}
            key = data.get("exchange_order_id") or data.get("order_id") or ""
            if not key:
                continue
            ts = evt.get("timestamp", "")

            order = orders.setdefault(key, {
                "exchange_order_id": key,
                "symbol": data.get("symbol", ""),
                "direction": data.get("direction", ""),
                "strategy_name": data.get("strategy_name", ""),
                "reduce_only": bool(data.get("reduce_only", False)),
                "trace_id": data.get("trace_id", ""),
                "quantity": data.get("quantity", 0),
                "placed_at": None,
                "filled_at": None,
                "closed_at": None,
                "closed_reason": None,
            })

            if etype == "order_placed":
                order["placed_at"] = order["placed_at"] or ts
                # 开仓单可能带完整方向信息，补全早期字段
                for f in ("symbol", "direction", "strategy_name", "trace_id"):
                    if not order.get(f) and data.get(f):
                        order[f] = data[f]
                if order.get("quantity") in (0, None) and data.get("quantity"):
                    order["quantity"] = data["quantity"]
            elif etype == "order_filled":
                order["filled_at"] = order["filled_at"] or ts
            elif etype in _CLOSE_EVENT_TYPES:
                order["closed_at"] = order["closed_at"] or ts
                order["closed_reason"] = order["closed_reason"] or _CLOSE_EVENT_TYPES[etype]

        return orders

    # ── 对账 ─────────────────────────────────────────────
    def reconcile(
        self,
        from_ts: Optional[datetime] = None,
        to_ts: Optional[datetime] = None,
        limit: Optional[int] = None,
        orphan_timeout_seconds: int = 600,
    ) -> Dict[str, Any]:
        """重建订单状态并对账，识别异常订单。

        归类：
        - in_flight      : 已下单未成交（非减仓单）
        - open_position  : 已成交未平仓（持仓周转中，非减仓单）
        - closed         : 已触发平仓事件（减仓单已闭环）
        - orphan_placed_no_fill : 已下单但长时间未成交（超过 orphan_timeout_seconds），
                                  可能是崩溃丢回执导致的悬单，需人工核查
        - orphan_closed_no_place: 平仓事件缺失对应开仓事件（多为重放窗口截断所致）
        """
        orders = self.replay_order_state(from_ts=from_ts, to_ts=to_ts, limit=limit)
        now = datetime.now()
        timeout = timedelta(seconds=max(0, orphan_timeout_seconds))

        placed = filled = closed = 0
        open_positions: List[Dict[str, Any]] = []
        in_flight: List[Dict[str, Any]] = []
        closed_orders: List[Dict[str, Any]] = []
        orphan_placed_no_fill: List[Dict[str, Any]] = []
        orphan_closed_no_place: List[Dict[str, Any]] = []

        for key, o in orders.items():
            if o["placed_at"]:
                placed += 1
            if o["filled_at"]:
                filled += 1
            if o["closed_at"]:
                closed += 1

            if o["closed_at"]:
                closed_orders.append(o)
                if not o["placed_at"]:
                    orphan_closed_no_place.append(o)
                continue

            if o["filled_at"]:
                if o["reduce_only"]:
                    # 减仓单已成交但未见平仓事件（窗口截断或漏发）——不判为持仓
                    closed_orders.append(o)
                else:
                    open_positions.append(o)
                continue

            if o["placed_at"]:
                # 已下单未成交：判断是否为悬单（超过超时窗口）
                in_flight.append(o)
                try:
                    p = datetime.fromisoformat(o["placed_at"])
                    if not o["reduce_only"] and (now - p) > timeout:
                        orphan_placed_no_fill.append(o)
                except (ValueError, TypeError):
                    pass

        return {
            "from_ts": from_ts.isoformat() if from_ts else None,
            "to_ts": to_ts.isoformat() if to_ts else None,
            "total_orders_reconstructed": len(orders),
            "summary": {
                "placed": placed,
                "filled": filled,
                "closed": closed,
                "open_positions": len(open_positions),
                "in_flight": len(in_flight),
                "orphan_placed_no_fill": len(orphan_placed_no_fill),
                "orphan_closed_no_place": len(orphan_closed_no_place),
            },
            "open_positions": open_positions,
            "in_flight": in_flight,
            "orphan_placed_no_fill": orphan_placed_no_fill,
            "orphan_closed_no_place": orphan_closed_no_place,
        }

    # ── 查询视图（供 API 复用） ──────────────────────────
    def tail(self, n: int = 50) -> List[Dict[str, Any]]:
        """最近 n 条事件（时间升序）。"""
        return self._store.tail(n=n)

    def count(self, event_type: Optional[str] = None) -> int:
        """事件总数统计（可按类型过滤）。"""
        return self._store.count(event_type=event_type)

    def replay(self, **kwargs) -> List[Dict[str, Any]]:
        """透传 EventStore.replay，供 API 按条件查询事件流。"""
        return self._store.replay(**kwargs)

    def verify_chain(self) -> Dict[str, Any]:
        """全链完整性校验（不可篡改）：返回哈希链校验结果。"""
        return self._store.verify_chain()