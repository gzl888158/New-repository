"""
统一止盈止损监控引擎 - TP/SL Monitor

职责：提供止盈止损状态的「单一口径」聚合视图，供 Dashboard 与主进程状态上报复用。

统一口径：
1. 保本缓冲 breakeven_buffer —— 按真实往返手续费（taker x2）折算，附带安全系数，
   取代散落各处的硬编码 0.2%/0.3%。
2. 持仓保护水位 —— 把交易所侧条件单（SL/TP/move_stop）与当前持仓按 (instId, posSide)
   对齐，判定每个持仓是否已挂止盈/止损，输出「未受保护持仓」清单。
3. 止损执行统计 —— 从 stop_loss_audit 表汇总触发类型、执行延迟、滑点。

设计为纯函数聚合 + 轻量无状态类，不参与下单路径，仅做读取与口径收敛，
避免影响主交易进程的实时下单。
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from loguru import logger


def _f(v) -> float:
    """安全转 float，空串/None/非法值返回 0.0"""
    try:
        if v is None or v == "":
            return 0.0
        return float(v)
    except (ValueError, TypeError):
        return 0.0


class TpSlMonitor:
    """统一止盈止损监控器（只读、无状态口径聚合）"""

    # 安全系数：保本缓冲至少覆盖往返手续费 + 一个最小滑点缓冲
    BREAKEVEN_SAFETY_MULT = 1.5
    BREAKEVEN_MIN_BUFFER = 0.001  # 最低 0.1%

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        trading_cfg = self.config.get("trading", {})
        self._taker_fee = _f(trading_cfg.get("taker_fee_rate", 0.0005))
        self._maker_fee = _f(trading_cfg.get("maker_fee_rate", 0.0002))

    def breakeven_buffer(self) -> float:
        """统一保本缓冲：往返 taker 手续费 * 安全系数，下限 0.1%"""
        round_trip = self._taker_fee * 2.0
        buffer = round_trip * self.BREAKEVEN_SAFETY_MULT
        return max(buffer, self.BREAKEVEN_MIN_BUFFER)

    @staticmethod
    def parse_algo_order(order: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """把一条交易所 pending 条件单归类为 SL / TP / move_stop。

        返回 {algo_id, inst_id, pos_side, side, type, trigger_px, callback_ratio, sz, state}
        """
        if not order:
            return None
        ord_type = str(order.get("ordType", "") or "").lower()
        inst_id = order.get("instId", "")
        algo_id = order.get("algoId", "")
        if not inst_id:
            return None

        sl_trigger = _f(order.get("slTriggerPx"))
        tp_trigger = _f(order.get("tpTriggerPx"))
        callback = _f(order.get("callbackRatio"))

        if ord_type == "move_stop" or callback > 0:
            kind = "move_stop"
            trigger_px = sl_trigger if sl_trigger > 0 else _f(order.get("triggerPx"))
        elif tp_trigger > 0 and sl_trigger <= 0:
            kind = "take_profit"
            trigger_px = tp_trigger
            callback = 0.0
        elif sl_trigger > 0 and tp_trigger <= 0:
            kind = "stop_loss"
            trigger_px = sl_trigger
            callback = 0.0
        else:
            # 两者同时存在或都缺失，按字段优先顺序兜底
            if tp_trigger > 0:
                kind, trigger_px, callback = "take_profit", tp_trigger, 0.0
            elif sl_trigger > 0:
                kind, trigger_px, callback = "stop_loss", sl_trigger, 0.0
            else:
                kind, trigger_px, callback = "unknown", 0.0, 0.0

        return {
            "algo_id": algo_id,
            "inst_id": inst_id,
            "pos_side": order.get("posSide", ""),
            "side": order.get("side", ""),
            "type": kind,
            "trigger_px": trigger_px,
            "callback_ratio": callback,
            "sz": _f(order.get("sz")),
            "state": order.get("state", ""),
        }

    def build_protection_view(
        self,
        positions: List[Dict[str, Any]],
        algo_orders: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """把持仓与条件单对齐，输出统一保护视图。

        返回：
        {
            "positions": [{symbol, side, mark_px, avg_px, upl, sl, tp[], move_stop,
                           has_sl, has_tp, unprotected}],
            "algo_orders": [{...parse_algo_order 结果}],
            "summary": {total_positions, protected, unprotected, sl_count, tp_count,
                        move_stop_count, breakeven_buffer}
        }
        """
        parsed_orders = [self.parse_algo_order(o) for o in (algo_orders or [])]
        parsed_orders = [p for p in parsed_orders if p]

        # 按 (instId, posSide) 聚合条件单，posSide 为空时用 'net' 兜底
        orders_by_key: Dict[str, List[Dict[str, Any]]] = {}
        for o in parsed_orders:
            key = (o["inst_id"], o["pos_side"] or "net")
            orders_by_key.setdefault(key, []).append(o)

        view_positions: List[Dict[str, Any]] = []
        for p in (positions or []):
            inst_id = p.get("instId", "")
            pos_side = p.get("posSide", "net")
            pos_qty = _f(p.get("pos"))
            if inst_id and pos_qty == 0:
                continue

            orders = orders_by_key.get((inst_id, pos_side), []) or orders_by_key.get((inst_id, "net"), [])
            sl = None
            tp_list: List[float] = []
            move_stop = None
            for o in orders:
                if o["type"] == "stop_loss" and o["trigger_px"] > 0:
                    sl = o["trigger_px"]
                elif o["type"] == "take_profit" and o["trigger_px"] > 0:
                    tp_list.append(o["trigger_px"])
                elif o["type"] == "move_stop":
                    move_stop = {
                        "trigger_px": o["trigger_px"],
                        "callback_ratio": o["callback_ratio"],
                    }

            has_sl = sl is not None
            has_move_stop = move_stop is not None
            has_tp = len(tp_list) > 0
            mark_px = _f(p.get("markPx"))

            view_positions.append({
                "symbol": inst_id,
                "side": pos_side,
                "mark_px": round(mark_px, 8),
                "avg_px": round(_f(p.get("avgPx")), 8),
                "upl": round(_f(p.get("upl")), 4),
                "sl": sl,
                "tp": sorted(tp_list),
                "move_stop": move_stop,
                "has_sl": has_sl,
                "has_move_stop": has_move_stop,
                "has_tp": has_tp,
                "unprotected": not has_sl and not has_move_stop,
            })

        total = len(view_positions)
        protected = sum(1 for v in view_positions if v["has_sl"] or v["has_move_stop"])
        summary = {
            "total_positions": total,
            "protected": protected,
            "unprotected": total - protected,
            "sl_count": sum(1 for v in view_positions if v["has_sl"]),
            "tp_count": sum(1 for v in view_positions if v["has_tp"]),
            "move_stop_count": sum(1 for v in view_positions if v["has_move_stop"]),
            "breakeven_buffer": round(self.breakeven_buffer(), 6),
        }

        return {
            "positions": view_positions,
            "algo_orders": parsed_orders,
            "summary": summary,
        }

    @staticmethod
    def compute_sl_execution_stats(rows) -> Dict[str, Any]:
        """从 stop_loss_audit 查询结果汇总止损执行统计。

        rows: sqlite3.Row 或 dict 列表，需含 trigger_type / execution_latency_ms /
              slippage_pct / pnl_percent 字段。
        """
        by_type: Dict[str, Dict[str, Any]] = {}
        total = 0
        latency_sum = 0.0
        slippage_sum = 0.0
        pnl_sum = 0.0

        for r in rows:
            if isinstance(r, dict):
                ttype = r.get("trigger_type", "unknown")
                lat = _f(r.get("execution_latency_ms"))
                slip = _f(r.get("slippage_pct"))
                pnl = _f(r.get("pnl_percent"))
            else:
                ttype = r["trigger_type"] if "trigger_type" in r.keys() else "unknown"
                lat = _f(r["execution_latency_ms"] if "execution_latency_ms" in r.keys() else 0)
                slip = _f(r["slippage_pct"] if "slippage_pct" in r.keys() else 0)
                pnl = _f(r["pnl_percent"] if "pnl_percent" in r.keys() else 0)

            b = by_type.setdefault(ttype, {"count": 0, "latency_sum": 0.0, "slippage_sum": 0.0, "pnl_sum": 0.0})
            b["count"] += 1
            b["latency_sum"] += lat
            b["slippage_sum"] += slip
            b["pnl_sum"] += pnl
            total += 1
            latency_sum += lat
            slippage_sum += slip
            pnl_sum += pnl

        trigger_types = {}
        for ttype, b in by_type.items():
            n = b["count"]
            trigger_types[ttype] = {
                "count": n,
                "avg_latency_ms": round(b["latency_sum"] / n, 2) if n else 0.0,
                "avg_slippage_pct": round(b["slippage_sum"] / n, 4) if n else 0.0,
                "avg_pnl_pct": round(b["pnl_sum"] / n, 4) if n else 0.0,
            }

        return {
            "total_count": total,
            "avg_latency_ms": round(latency_sum / total, 2) if total else 0.0,
            "avg_slippage_pct": round(slippage_sum / total, 4) if total else 0.0,
            "avg_pnl_pct": round(pnl_sum / total, 4) if total else 0.0,
            "trigger_types": trigger_types,
        }
