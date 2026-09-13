"""
算法订单执行监控器 (Execution Monitor)

实时追踪 TWAP / VWAP / Iceberg 的执行状态：
  - 切片级事件时间线 (submitted → filled → confirmed)
  - 进度 vs 计划（调度漂移检测）
  - 切片滑点分析（arrival / VWAP / benchmark）
  - 市场冲击估算（每片前后价差）
  - 成本节省（vs 市价单基准）
  - 执行预警（超滑点 / 低填充 / 进度滞后）

供 algo_execution_engine 在每次切片事件时回调，供 dashboard API 实时查询。
"""
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
from loguru import logger


class SliceEvent(Enum):
    """切片事件类型"""
    SCHEDULED = "scheduled"          # 计划排定
    SUBMITTED = "submitted"          # 已提交到交易所
    PARTIALLY_FILLED = "partial"     # 部分成交
    FILLED = "filled"                # 全部成交
    CANCELLED = "cancelled"          # 已取消
    REJECTED = "rejected"            # 被拒
    EXPIRED = "expired"              # 过期


class AlertLevel(Enum):
    """告警等级"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SliceExecutionRecord:
    """单个切片的完整执行记录"""
    record_id: str
    order_id: str
    algo_type: str
    symbol: str
    side: str
    sequence: int
    # 计划
    scheduled_at: Optional[datetime] = None
    scheduled_qty: float = 0.0
    # 提交
    submitted_at: Optional[datetime] = None
    submitted_qty: float = 0.0
    submitted_price: float = 0.0      # 限价
    order_type: str = "limit"
    # 成交
    filled_at: Optional[datetime] = None
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    # 上下文（切片时刻的市场状态）
    mid_price_at_slice: float = 0.0
    vwap_at_slice: float = 0.0        # 执行期间市场 VWAP
    bid_depth_at_slice: float = 0.0    # 买方深度
    ask_depth_at_slice: float = 0.0    # 卖方深度
    spread_bps_at_slice: float = 0.0
    # 质量指标
    fill_rate: float = 0.0             # 本片填充率
    arrival_slippage_bps: float = 0.0  # vs 提交时 mid
    vwap_slippage_bps: float = 0.0     # vs 市场 VWAP
    latency_ms: float = 0.0            # 提交→成交延迟
    status: str = "pending"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seq": self.sequence,
            "qty_planned": round(self.scheduled_qty, 4),
            "qty_filled": round(self.filled_qty, 4),
            "fill_rate": round(self.fill_rate, 3),
            "price": {
                "limit": round(self.submitted_price, 4) if self.submitted_price else 0,
                "avg_fill": round(self.avg_fill_price, 4),
                "mid_at_slice": round(self.mid_price_at_slice, 4),
                "vwap_at_slice": round(self.vwap_at_slice, 4),
            },
            "slippage": {
                "arrival_bps": round(self.arrival_slippage_bps, 2),
                "vwap_bps": round(self.vwap_slippage_bps, 2),
            },
            "market": {
                "spread_bps": round(self.spread_bps_at_slice, 1),
                "bid_depth": round(self.bid_depth_at_slice, 2),
                "ask_depth": round(self.ask_depth_at_slice, 2),
            },
            "latency_ms": round(self.latency_ms, 1),
            "timeline": {
                "scheduled": self.scheduled_at.isoformat() if self.scheduled_at else None,
                "submitted": self.submitted_at.isoformat() if self.submitted_at else None,
                "filled": self.filled_at.isoformat() if self.filled_at else None,
            },
        }


@dataclass
class AlgoExecutionTimeline:
    """单笔算法订单的完整执行时间线"""
    order_id: str
    algo_type: str
    symbol: str
    side: str
    total_quantity: float
    # 计划
    num_slices: int = 0
    start_time: Optional[datetime] = None
    scheduled_end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    # 实时进度
    filled_quantity: float = 0.0
    executed_slices: int = 0
    rejected_slices: int = 0
    failed_slices: int = 0
    fill_rate: float = 0.0
    progress_pct: float = 0.0          # 进度百分比
    # 时间进度
    elapsed_seconds: float = 0.0
    schedule_adherence: float = 100.0   # 进度/时间 比 (%)
    # 价格
    arrival_price: float = 0.0
    avg_execution_price: float = 0.0
    target_benchmark: float = 0.0      # TWAP/VWAP 基准
    # 成本
    total_cost: float = 0.0
    commission: float = 0.0
    estimated_market_impact: float = 0.0
    saved_vs_market_order: float = 0.0
    # 汇总滑点
    avg_arrival_slip_bps: float = 0.0
    avg_vwap_slip_bps: float = 0.0
    max_arrival_slip_bps: float = 0.0
    # 切片记录
    slice_records: List[SliceExecutionRecord] = field(default_factory=list)
    # 告警
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "created"

    def to_dict(self) -> Dict[str, Any]:
        last_slice = self.slice_records[-1] if self.slice_records else None
        return {
            "order_id": self.order_id,
            "algo_type": self.algo_type,
            "symbol": self.symbol,
            "side": self.side,
            "status": self.status,
            "progress": {
                "total_qty": round(self.total_quantity, 4),
                "filled_qty": round(self.filled_quantity, 4),
                "fill_rate": round(self.fill_rate, 3),
                "progress_pct": round(self.progress_pct, 1),
                "slices_done": self.executed_slices,
                "slices_total": self.num_slices,
                "slices_failed": self.failed_slices,
                "slices_rejected": self.rejected_slices,
            },
            "timing": {
                "elapsed_seconds": round(self.elapsed_seconds, 1),
                "total_duration": round(self.duration_seconds, 1),
                "schedule_adherence": round(self.schedule_adherence, 1),
                "time_remaining": round(max(0, self.duration_seconds - self.elapsed_seconds), 1),
            },
            "prices": {
                "arrival": round(self.arrival_price, 4),
                "avg_execution": round(self.avg_execution_price, 4),
                "benchmark": round(self.target_benchmark, 4),
                "last_mid": round(last_slice.mid_price_at_slice, 4) if last_slice else 0,
            },
            "costs": {
                "total_usd": round(self.total_cost, 4),
                "commission": round(self.commission, 4),
                "market_impact_est": round(self.estimated_market_impact, 4),
                "saved_vs_market": round(self.saved_vs_market_order, 4),
            },
            "slippage": {
                "avg_arrival_bps": round(self.avg_arrival_slip_bps, 2),
                "avg_vwap_bps": round(self.avg_vwap_slip_bps, 2),
                "max_arrival_bps": round(self.max_arrival_slip_bps, 2),
            },
            "alerts": [a["message"] for a in self.alerts[-5:]] if self.alerts else [],
            "recent_slices": [s.to_dict() for s in self.slice_records[-10:]],
        }


@dataclass
class CostSavingsSummary:
    """算法执行成本节省汇总"""
    total_notional: float = 0.0         # 总名义价值
    total_executed: float = 0.0         # 总已执行量
    # 实际执行成本
    actual_cost: float = 0.0
    actual_cost_bps: float = 0.0
    # 市价单基准成本（假设）
    market_order_cost_est: float = 0.0  # 滑点 + 价差 + 冲击
    market_order_cost_bps: float = 0.0
    # 节省
    saved_cost: float = 0.0
    saved_cost_bps: float = 0.0
    # 拆解
    spread_saved: float = 0.0           # 价差节省
    impact_saved: float = 0.0           # 冲击节省
    timing_saved: float = 0.0           # 择时节省
    # 订单数
    total_orders: int = 0
    total_slices: int = 0
    # 按算法类型
    by_algo_type: Dict[str, Dict] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_notional": round(self.total_notional, 2),
            "total_executed": round(self.total_executed, 4),
            "actual_cost": {
                "total_usd": round(self.actual_cost, 4),
                "total_bps": round(self.actual_cost_bps, 2),
            },
            "market_order_benchmark": {
                "estimated_usd": round(self.market_order_cost_est, 4),
                "estimated_bps": round(self.market_order_cost_bps, 2),
            },
            "savings": {
                "total_usd": round(self.saved_cost, 4),
                "total_bps": round(self.saved_cost_bps, 2),
                "spread_saved": round(self.spread_saved, 4),
                "impact_saved": round(self.impact_saved, 4),
                "timing_saved": round(self.timing_saved, 4),
            },
            "orders": {
                "total": self.total_orders,
                "total_slices": self.total_slices,
            },
            "by_type": {k: {"orders": v.get("orders", 0), "saved_bps": round(v.get("saved_bps", 0), 2)}
                       for k, v in self.by_algo_type.items()},
        }


# ═══════════════════════════════════════════════════════════════
# 执行监控器
# ═══════════════════════════════════════════════════════════════

class AlgoExecutionMonitor:
    """算法订单实时执行监控器（切片级 TWAP/VWAP/Iceberg 执行追踪）"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("execution_monitor", {}) if config else {}
        self._enabled = cfg.get("enabled", True)

        # 活跃订单时间线
        self._active_timelines: Dict[str, AlgoExecutionTimeline] = {}
        # 归档（已完成）
        self._completed_timelines: deque = deque(maxlen=cfg.get("max_history", 100))
        # 最新切片事件（用于实时面板）
        self._recent_events: deque = deque(maxlen=cfg.get("max_events", 500))
        # 市场基准快照
        self._market_snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        # 成本节省汇总
        self._savings_summary = CostSavingsSummary()

        # 告警阈值
        self._warning_thresholds = {
            "schedule_drift_pct": cfg.get("schedule_drift_warn", 20),   # 进度漂移>20%告警
            "slice_slippage_bps": cfg.get("slice_slip_warn", 15),        # 单切片滑点>15bps
            "cumulative_slippage_bps": cfg.get("cumul_slip_warn", 30),   # 累计滑点>30bps
            "low_fill_rate": cfg.get("fill_rate_warn", 0.5),             # 填充率<50%
            "max_rejections": cfg.get("max_rejections", 3),              # 连续拒绝>3次
        }

        logger.info(f"ExecutionMonitor initialized: max_history={cfg.get('max_history', 100)}")

    # ── 生命周期 ──────────────────────────────────────────────

    def on_order_created(self, order_id: str, algo_type: str, symbol: str,
                          side: str, total_quantity: float, num_slices: int,
                          duration_seconds: float, arrival_price: float = 0.0):
        """订单创建时注册"""
        if not self._enabled:
            return
        t = AlgoExecutionTimeline(
            order_id=order_id,
            algo_type=algo_type,
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            num_slices=num_slices,
            start_time=datetime.now(),
            duration_seconds=duration_seconds,
            arrival_price=arrival_price,
            status="running",
        )
        self._active_timelines[order_id] = t
        self._log_event(order_id, "order_created",
                       f"Algo {algo_type.upper()} started: {symbol} {side} qty={total_quantity} slices={num_slices}")

    def on_slice_scheduled(self, order_id: str, sequence: int,
                            quantity: float, scheduled_time: datetime):
        """切片已排定"""
        tl = self._active_timelines.get(order_id)
        if not tl:
            return
        record = SliceExecutionRecord(
            record_id=f"{order_id}_s{sequence}",
            order_id=order_id,
            algo_type=tl.algo_type,
            symbol=tl.symbol,
            side=tl.side,
            sequence=sequence,
            scheduled_at=scheduled_time,
            scheduled_qty=quantity,
            status="scheduled",
        )
        tl.slice_records.append(record)
        self._log_event(order_id, f"slice_{sequence}_scheduled",
                       f"Slice #{sequence} scheduled: qty={quantity:.4f}")

    def on_slice_submitted(self, order_id: str, sequence: int,
                            submitted_qty: float, limit_price: float,
                            order_type: str = "limit"):
        """切片已提交到交易所"""
        tl = self._active_timelines.get(order_id)
        if not tl:
            return
        record = self._find_or_create_slice_record(tl, order_id, sequence)
        record.submitted_at = datetime.now()
        record.submitted_qty = submitted_qty
        record.submitted_price = limit_price
        record.order_type = order_type
        record.status = "submitted"

        # 捕捉市场状态
        self._capture_market_context(record, tl.symbol)
        self._log_event(order_id, f"slice_{sequence}_submitted",
                       f"Slice #{sequence} submitted: qty={submitted_qty:.4f} @ {limit_price:.4f}")

    def on_slice_filled(self, order_id: str, sequence: int,
                         filled_qty: float, avg_fill_price: float):
        """切片已成交（全部或部分）"""
        tl = self._active_timelines.get(order_id)
        if not tl:
            return
        record = self._find_or_create_slice_record(tl, order_id, sequence)
        record.filled_at = datetime.now()
        record.filled_qty += filled_qty
        record.status = "filled"

        # 计算执行质量
        if record.avg_fill_price > 0 and filled_qty > 0:
            # 加权平均
            total = record.avg_fill_price * (record.filled_qty - filled_qty) + \
                    avg_fill_price * filled_qty
            record.avg_fill_price = total / max(record.filled_qty, 1e-10)
        else:
            record.avg_fill_price = avg_fill_price

        record.fill_rate = record.filled_qty / max(record.scheduled_qty, 1e-10)

        # 滑点计算
        self._compute_slice_slippage(record, tl.side)

        # 延迟
        if record.submitted_at:
            record.latency_ms = (record.filled_at - record.submitted_at).total_seconds() * 1000

        # 累计更新
        self._update_cumulative_metrics(tl)

        self._log_event(order_id, f"slice_{sequence}_filled",
                       f"Slice #{sequence} filled: {filled_qty:.4f} @ {avg_fill_price:.4f}")
        self._check_alerts(tl, record)

    def on_slice_failed(self, order_id: str, sequence: int, reason: str = ""):
        """切片失败"""
        tl = self._active_timelines.get(order_id)
        if not tl:
            return
        record = self._find_or_create_slice_record(tl, order_id, sequence)
        record.status = "rejected"
        record.filled_at = datetime.now()
        tl.rejected_slices += 1
        self._log_event(order_id, f"slice_{sequence}_failed",
                       f"Slice #{sequence} FAILED: {reason}")
        self._check_alerts(tl, record)

    def on_order_completed(self, order_id: str):
        """订单执行完成"""
        tl = self._active_timelines.pop(order_id, None)
        if tl:
            tl.status = "completed"
            self._log_event(order_id, "order_completed",
                           f"{tl.algo_type.upper()} completed: "
                           f"filled={tl.filled_quantity:.4f}/{tl.total_quantity:.4f} "
                           f"slip={tl.avg_arrival_slip_bps:.1f}bps")
            self._completed_timelines.append(tl)

    def on_order_cancelled(self, order_id: str, reason: str = ""):
        """订单被取消"""
        tl = self._active_timelines.pop(order_id, None)
        if tl:
            tl.status = "cancelled"
            self._log_event(order_id, "order_cancelled",
                           f"{tl.algo_type.upper()} cancelled: {reason or 'user cancelled'}")

    # ── 成本基准 ──────────────────────────────────────────────

    def record_cost_savings(self, order_id: str, notional: float,
                             actual_cost: float,
                             market_order_estimated_cost: float,
                             spread_saved: float = 0.0,
                             impact_saved: float = 0.0):
        """记录成本节省"""
        tl = self._active_timelines.get(order_id)
        if tl:
            tl.saved_vs_market_order += market_order_estimated_cost - actual_cost

        s = self._savings_summary
        s.total_notional += notional
        s.total_orders += 1
        s.actual_cost += actual_cost
        s.market_order_cost_est += market_order_estimated_cost
        s.saved_cost += market_order_estimated_cost - actual_cost
        s.spread_saved += spread_saved
        s.impact_saved += impact_saved

    # ── 内部计算 ──────────────────────────────────────────────

    def _find_or_create_slice_record(self, tl: AlgoExecutionTimeline,
                                      order_id: str, sequence: int) -> SliceExecutionRecord:
        for r in tl.slice_records:
            if r.sequence == sequence:
                return r
        record = SliceExecutionRecord(
            record_id=f"{order_id}_s{sequence}",
            order_id=order_id, algo_type=tl.algo_type,
            symbol=tl.symbol, side=tl.side,
            sequence=sequence, status="pending",
        )
        tl.slice_records.append(record)
        return record

    def _capture_market_context(self, record: SliceExecutionRecord, symbol: str):
        """捕获切片时的市场快照（从已缓存的行情）"""
        snapshots = self._market_snapshots.get(symbol)
        if snapshots:
            latest = snapshots[-1]
            record.mid_price_at_slice = latest.get("mid", 0)
            record.vwap_at_slice = latest.get("vwap", 0)
            record.bid_depth_at_slice = latest.get("bid_depth", 0)
            record.ask_depth_at_slice = latest.get("ask_depth", 0)
            record.spread_bps_at_slice = latest.get("spread_bps", 0)

    def _compute_slice_slippage(self, record: SliceExecutionRecord, side: str):
        arrival = record.mid_price_at_slice or record.submitted_price
        vwap_cmp = record.vwap_at_slice or arrival
        fill = record.avg_fill_price
        if arrival > 0 and fill > 0:
            slip_arr = (fill - arrival) / arrival * 10000
            record.arrival_slippage_bps = -slip_arr if side == "sell" else slip_arr
        if vwap_cmp > 0 and fill > 0:
            slip_vwap = (fill - vwap_cmp) / vwap_cmp * 10000
            record.vwap_slippage_bps = -slip_vwap if side == "sell" else slip_vwap

    def _update_cumulative_metrics(self, tl: AlgoExecutionTimeline):
        """基于所有切片记录更新累计指标"""
        filled_records = [r for r in tl.slice_records
                         if r.status in ("filled", "partial")]
        tl.executed_slices = len(filled_records)
        tl.failed_slices = sum(1 for r in tl.slice_records if r.status == "rejected")

        if filled_records:
            total_filled = sum(r.filled_qty for r in filled_records)
            total_cost = sum(r.filled_qty * r.avg_fill_price for r in filled_records)
            tl.filled_quantity = total_filled
            tl.fill_rate = total_filled / max(tl.total_quantity, 1e-10)
            tl.progress_pct = tl.fill_rate * 100
            tl.avg_execution_price = total_cost / max(total_filled, 1e-10)
            tl.total_cost = total_cost

            slips = [r.arrival_slippage_bps for r in filled_records if r.arrival_slippage_bps != 0]
            vwap_slips = [r.vwap_slippage_bps for r in filled_records if r.vwap_slippage_bps != 0]
            if slips:
                tl.avg_arrival_slip_bps = sum(slips) / len(slips)
                tl.max_arrival_slip_bps = max(abs(s) for s in slips)
            if vwap_slips:
                tl.avg_vwap_slip_bps = sum(vwap_slips) / len(vwap_slips)

        # 进度 vs 时间
        tl.elapsed_seconds = (datetime.now() - (tl.start_time or datetime.now())).total_seconds()
        if tl.duration_seconds > 0:
            expected_progress = min(100, tl.elapsed_seconds / tl.duration_seconds * 100)
            if expected_progress > 0:
                tl.schedule_adherence = tl.progress_pct / expected_progress * 100
            else:
                tl.schedule_adherence = 100.0

    def _check_alerts(self, tl: AlgoExecutionTimeline,
                       record: SliceExecutionRecord):
        """检查告警条件"""
        alerts = []

        # 进度漂移
        if tl.schedule_adherence < (100 - self._warning_thresholds["schedule_drift_pct"]):
            severity = AlertLevel.CRITICAL if tl.schedule_adherence < 50 else AlertLevel.WARNING
            alerts.append({
                "severity": severity.value,
                "type": "schedule_drift",
                "message": (f"[{tl.symbol}] {tl.algo_type.upper()}进度滞后: "
                           f"已完成{tl.progress_pct:.0f}%, 预期{min(100, tl.elapsed_seconds / max(tl.duration_seconds, 1) * 100):.0f}%"),
            })

        # 单切片滑点
        if abs(record.arrival_slippage_bps) > self._warning_thresholds["slice_slippage_bps"]:
            alerts.append({
                "severity": AlertLevel.WARNING.value,
                "type": "slice_slippage",
                "message": (f"[{tl.symbol}] 切片#{record.sequence} 滑点超标: "
                           f"{record.arrival_slippage_bps:.1f}bps"),
            })

        # 累计滑点
        if tl.avg_arrival_slip_bps > self._warning_thresholds["cumulative_slippage_bps"]:
            alerts.append({
                "severity": AlertLevel.CRITICAL.value,
                "type": "cumulative_slippage",
                "message": (f"[{tl.symbol}] {tl.algo_type.upper()}累计滑点超标: "
                           f"{tl.avg_arrival_slip_bps:.1f}bps > {self._warning_thresholds['cumulative_slippage_bps']}bps"),
            })

        # 连续拒绝
        recent_fails = sum(1 for r in tl.slice_records[-self._warning_thresholds["max_rejections"]:]
                          if r.status == "rejected")
        if recent_fails >= self._warning_thresholds["max_rejections"]:
            alerts.append({
                "severity": AlertLevel.CRITICAL.value,
                "type": "consecutive_rejections",
                "message": (f"[{tl.symbol}] {tl.algo_type.upper()}连续拒绝 {recent_fails} 个切片"),
            })

        # 低填充率
        if record.fill_rate < self._warning_thresholds["low_fill_rate"] and record.fill_rate > 0:
            alerts.append({
                "severity": AlertLevel.WARNING.value,
                "type": "low_fill_rate",
                "message": (f"[{tl.symbol}] 切片#{record.sequence} 填充率低: {record.fill_rate:.0%}"),
            })

        tl.alerts.extend(alerts)

    def _log_event(self, order_id: str, event_type: str, message: str):
        self._recent_events.append({
            "timestamp": datetime.now().isoformat(),
            "order_id": order_id,
            "event": event_type,
            "message": message,
        })

    # ── 市场快照 ──────────────────────────────────────────────

    def push_market_snapshot(self, symbol: str, mid: float, vwap: float = 0.0,
                              bid_depth: float = 0.0, ask_depth: float = 0.0,
                              spread_bps: float = 0.0):
        """推送市场快照（供外部行情服务调用）"""
        self._market_snapshots[symbol].append({
            "ts": datetime.now().isoformat(),
            "mid": mid, "vwap": vwap,
            "bid_depth": bid_depth, "ask_depth": ask_depth,
            "spread_bps": spread_bps,
        })

    # ── 查询接口 ──────────────────────────────────────────────

    def get_active_summary(self) -> List[Dict[str, Any]]:
        """获取所有活跃算法订单的摘要"""
        summaries = []
        for order_id, tl in self._active_timelines.items():
            self._update_cumulative_metrics(tl)
            summaries.append({
                "order_id": order_id,
                "algo_type": tl.algo_type,
                "symbol": tl.symbol,
                "side": tl.side,
                "status": tl.status,
                "total_qty": round(tl.total_quantity, 4),
                "filled_qty": round(tl.filled_quantity, 4),
                "progress_pct": round(tl.progress_pct, 1),
                "schedule_adherence": round(tl.schedule_adherence, 1),
                "avg_price": round(tl.avg_execution_price, 4),
                "avg_slip_bps": round(tl.avg_arrival_slip_bps, 2),
                "alerts": len(tl.alerts),
                "slices": f"{tl.executed_slices}/{tl.num_slices}",
            })
        return summaries

    def get_order_timeline(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取单笔订单的完整时间线"""
        tl = self._active_timelines.get(order_id)
        if not tl:
            return None
        self._update_cumulative_metrics(tl)
        return tl.to_dict()

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的执行事件"""
        return list(self._recent_events)[-limit:]

    def get_cost_savings(self) -> Dict[str, Any]:
        """获取成本节省汇总"""
        s = self._savings_summary
        s.actual_cost_bps = s.actual_cost / max(s.total_notional, 1.0) * 10000
        s.market_order_cost_bps = s.market_order_cost_est / max(s.total_notional, 1.0) * 10000
        s.saved_cost_bps = s.saved_cost / max(s.total_notional, 1.0) * 10000
        return s.to_dict()

    def get_alerts(self, min_severity: str = "warning") -> List[Dict[str, Any]]:
        """获取活跃告警"""
        all_alerts = []
        levels = {"info": 0, "warning": 1, "critical": 2}
        min_level = levels.get(min_severity, 0)
        for tl in self._active_timelines.values():
            for a in tl.alerts:
                if levels.get(a.get("severity", "info"), 0) >= min_level:
                    all_alerts.append({**a, "order_id": tl.order_id, "symbol": tl.symbol})
        return all_alerts

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "active_orders": len(self._active_timelines),
            "completed_orders": len(self._completed_timelines),
            "total_events": len(self._recent_events),
            "alerts_active": len(self.get_alerts()),
            "active_summary": self.get_active_summary(),
            "cost_savings": self.get_cost_savings(),
            "recent_events": self.get_recent_events(10),
        }
