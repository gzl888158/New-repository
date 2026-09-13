"""
执行质量监控 (Execution Quality Monitor)

多维度评估订单执行质量，对标行业最佳实践：
  - 实现缺口分析 (Implementation Shortfall): 决策价格 vs 实际成交价
  - VWAP滑点分析: 成交均价 vs 市场VWAP
  - 到达价格滑点: 成交均价 vs 下单时mid价格
  - 执行成本分解: 手续费、滑点、市场冲击、延迟成本
  - 成交量加权分析: 按成交量计算加权滑点
  - 对标分析: P50/P75/P90/P95 执行质量分位数
  - 每日/周/月执行质量报告
"""
import asyncio
import json
import math
import statistics
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
import numpy as np
from loguru import logger


class QualityGrade(Enum):
    """执行质量等级"""
    EXCELLENT = "excellent"    # P95以上
    GOOD = "good"              # P75-P95
    AVERAGE = "average"        # P50-P75
    BELOW_AVERAGE = "below"    # P25-P50
    POOR = "poor"              # P25以下


@dataclass
class ImplementationShortfall:
    """实现缺口分析"""
    decision_price: float          # 决策价格（信号生成时mid）
    arrival_price: float           # 到达价格（下单时mid）
    execution_price: float         # 实际成交均价
    quantity: float
    side: str
    # 分解
    total_shortfall: float = 0.0        # 总缺口 (USD)
    total_shortfall_bps: float = 0.0    # 总缺口 (bps)
    commission: float = 0.0             # 手续费
    delay_cost: float = 0.0             # 延迟成本 (decision->arrival)
    execution_cost: float = 0.0         # 执行成本 (arrival->execution)
    opportunity_cost: float = 0.0       # 机会成本（未成交部分）
    market_impact: float = 0.0          # 市场冲击
    timing_cost: float = 0.0            # 择时成本

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_price": self.decision_price,
            "arrival_price": self.arrival_price,
            "execution_price": self.execution_price,
            "quantity": self.quantity,
            "side": self.side,
            "total_shortfall": round(self.total_shortfall, 4),
            "total_shortfall_bps": round(self.total_shortfall_bps, 2),
            "commission": round(self.commission, 4),
            "delay_cost": round(self.delay_cost, 4),
            "execution_cost": round(self.execution_cost, 4),
            "opportunity_cost": round(self.opportunity_cost, 4),
            "market_impact": round(self.market_impact, 4),
            "timing_cost": round(self.timing_cost, 4),
        }


@dataclass
class SlippageAnalysis:
    """滑点分析"""
    symbol: str
    side: str
    quantity: float
    execution_price: float
    arrival_price: float = 0.0
    vwap_price: float = 0.0             # 市场VWAP
    # 滑点指标
    arrival_slippage_bps: float = 0.0
    vwap_slippage_bps: float = 0.0
    vs_benchmark_bps: float = 0.0       # 对标滑点
    # 分位数
    arrival_slip_percentile: float = 50.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "execution_price": self.execution_price,
            "arrival_price": self.arrival_price,
            "vwap_price": self.vwap_price,
            "arrival_slippage_bps": round(self.arrival_slippage_bps, 2),
            "vwap_slippage_bps": round(self.vwap_slippage_bps, 2),
            "vs_benchmark_bps": round(self.vs_benchmark_bps, 2),
            "arrival_slip_percentile": self.arrival_slip_percentile,
        }


@dataclass
class ExecutionCostBreakdown:
    """执行成本分解"""
    symbol: str
    notional: float
    # 显性成本
    commission: float = 0.0            # 交易所手续费
    spread_cost: float = 0.0           # 价差成本
    # 隐性成本
    market_impact: float = 0.0         # 市场冲击
    timing_cost: float = 0.0           # 延迟/择时成本
    opportunity_cost: float = 0.0       # 机会成本
    # 汇总
    total_cost: float = 0.0
    total_cost_bps: float = 0.0
    # 节省
    saved_vs_market_order: float = 0.0  # 相比市价单节省

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "notional": round(self.notional, 2),
            "commission": round(self.commission, 4),
            "spread_cost": round(self.spread_cost, 4),
            "market_impact": round(self.market_impact, 4),
            "timing_cost": round(self.timing_cost, 4),
            "opportunity_cost": round(self.opportunity_cost, 4),
            "total_cost": round(self.total_cost, 4),
            "total_cost_bps": round(self.total_cost_bps, 2),
            "saved_vs_market_order": round(self.saved_vs_market_order, 4),
        }


@dataclass
class QualityMetrics:
    """执行质量指标"""
    # 统计
    total_orders: int = 0
    filled_orders: int = 0
    total_volume: float = 0.0
    total_notional: float = 0.0
    # 均价
    avg_arrival_slippage_bps: float = 0.0
    avg_vwap_slippage_bps: float = 0.0
    avg_implementation_shortfall_bps: float = 0.0
    # 分位数
    arrival_slippage_p50: float = 0.0
    arrival_slippage_p75: float = 0.0
    arrival_slippage_p90: float = 0.0
    arrival_slippage_p95: float = 0.0
    # 成交量加权
    vwap_arrival_slippage_bps: float = 0.0
    vwap_implementation_shortfall_bps: float = 0.0
    # 成本
    avg_total_cost_bps: float = 0.0
    avg_saved_vs_market_bps: float = 0.0
    # 统计
    fill_rate_avg: float = 0.0
    rejected_rate: float = 0.0
    # 质量
    quality_grade: QualityGrade = QualityGrade.AVERAGE
    quality_score: float = 50.0       # 0-100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_orders": self.total_orders,
            "filled_orders": self.filled_orders,
            "total_volume": round(self.total_volume, 4),
            "total_notional": round(self.total_notional, 2),
            "avg_arrival_slippage_bps": round(self.avg_arrival_slippage_bps, 2),
            "arrival_slippage_p50": round(self.arrival_slippage_p50, 2),
            "arrival_slippage_p75": round(self.arrival_slippage_p75, 2),
            "arrival_slippage_p90": round(self.arrival_slippage_p90, 2),
            "arrival_slippage_p95": round(self.arrival_slippage_p95, 2),
            "avg_vwap_slippage_bps": round(self.avg_vwap_slippage_bps, 2),
            "avg_implementation_shortfall_bps": round(self.avg_implementation_shortfall_bps, 2),
            "avg_total_cost_bps": round(self.avg_total_cost_bps, 2),
            "fill_rate_avg": round(self.fill_rate_avg, 4),
            "quality_grade": self.quality_grade.value,
            "quality_score": round(self.quality_score, 1),
        }


@dataclass
class ExecutionQualityReport:
    """执行质量报告"""
    report_id: str = ""
    period: str = "daily"              # daily / weekly / monthly
    start_time: str = ""
    end_time: str = ""
    # 汇总
    summary: QualityMetrics = field(default_factory=QualityMetrics)
    # 明细
    shortfall_analyses: List[ImplementationShortfall] = field(default_factory=list)
    slippage_analyses: List[SlippageAnalysis] = field(default_factory=list)
    cost_breakdowns: List[ExecutionCostBreakdown] = field(default_factory=list)
    # 改进建议
    recommendations: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "period": self.period,
            "period_range": f"{self.start_time} ~ {self.end_time}",
            "summary": {
                "total_orders": self.summary.total_orders,
                "total_volume": round(self.summary.total_volume, 4),
                "total_notional": round(self.summary.total_notional, 2),
                "avg_arrival_slip_bps": round(self.summary.avg_arrival_slippage_bps, 2),
                "avg_vwap_slip_bps": round(self.summary.avg_vwap_slippage_bps, 2),
                "avg_shortfall_bps": round(self.summary.avg_implementation_shortfall_bps, 2),
                "arrival_slip_percentiles": {
                    "p50": round(self.summary.arrival_slippage_p50, 2),
                    "p75": round(self.summary.arrival_slippage_p75, 2),
                    "p90": round(self.summary.arrival_slippage_p90, 2),
                    "p95": round(self.summary.arrival_slippage_p95, 2),
                },
                "vwap_weighted_slip_bps": round(self.summary.vwap_arrival_slippage_bps, 2),
                "avg_total_cost_bps": round(self.summary.avg_total_cost_bps, 2),
                "avg_saved_vs_market_bps": round(self.summary.avg_saved_vs_market_bps, 2),
                "fill_rate": round(self.summary.fill_rate_avg, 3),
                "rejected_rate": round(self.summary.rejected_rate, 3),
                "quality_grade": self.summary.quality_grade.value,
                "quality_score": round(self.summary.quality_score, 1),
            },
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# 执行质量监控器
# ═══════════════════════════════════════════════════════════════

class ExecutionQualityMonitor:
    """执行质量监控器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("execution_quality_monitor", {}) if config else {}
        self._enabled = cfg.get("enabled", True)
        self._rolling_window = cfg.get("rolling_window_orders", 1000)
        self._report_schedule = cfg.get("report_schedule", "daily")  # daily / weekly / realtime

        # 滑点历史
        self._arrival_slips: deque = deque(maxlen=self._rolling_window)
        self._vwap_slips: deque = deque(maxlen=self._rolling_window)
        self._shortfalls: deque = deque(maxlen=self._rolling_window)
        # 成本历史
        self._costs: deque = deque(maxlen=self._rolling_window)
        # 按symbol分组
        self._symbol_stats: Dict[str, Dict] = defaultdict(lambda: {
            "arrival_slips": deque(maxlen=500),
            "vwap_slips": deque(maxlen=500),
            "count": 0,
            "volume": 0.0,
        })
        # 按策略分组
        self._strategy_stats: Dict[str, Dict] = defaultdict(lambda: {
            "arrival_slips": deque(maxlen=500),
            "count": 0,
        })
        # 报告
        self._reports: List[ExecutionQualityReport] = []
        # 基准
        self._benchmark_thresholds = {
            "excellent": 5.0,    # <5bps
            "good": 10.0,        # <10bps
            "average": 20.0,     # <20bps
            "below": 35.0,       # <35bps
            # >35bps = poor
        }

        logger.info(f"ExecutionQualityMonitor initialized: rolling_window={self._rolling_window}")

    # ── 指标记录 ──────────────────────────────────────────────

    def record_execution(self, symbol: str, side: str, quantity: float,
                          execution_price: float, arrival_price: float,
                          decision_price: float = 0.0,
                          vwap_price: float = 0.0,
                          commission: float = 0.0,
                          strategy: str = "",
                          filled: bool = True,
                          fill_quantity: float = None):
        """记录单笔执行的质量指标"""
        if not self._enabled:
            return

        fill_qty = fill_quantity if fill_quantity is not None else quantity

        # 到达价格滑点
        if arrival_price > 0 and execution_price > 0:
            arrival_slip = (execution_price - arrival_price) / arrival_price * 10000
            if side == "sell":
                arrival_slip *= -1
            self._arrival_slips.append(arrival_slip)
        else:
            arrival_slip = 0.0

        # VWAP滑点
        if vwap_price > 0 and execution_price > 0:
            vwap_slip = (execution_price - vwap_price) / vwap_price * 10000
            if side == "sell":
                vwap_slip *= -1
            self._vwap_slips.append(vwap_slip)
        else:
            vwap_slip = 0.0

        # 实现缺口
        if decision_price > 0:
            notional = quantity * execution_price
            shortfall = ImplementationShortfall(
                decision_price=decision_price,
                arrival_price=arrival_price,
                execution_price=execution_price,
                quantity=fill_qty,
                side=side,
                total_shortfall=fill_qty * (execution_price - decision_price),
                total_shortfall_bps=(execution_price - decision_price) / decision_price * 10000,
                commission=commission,
                delay_cost=fill_qty * abs(arrival_price - decision_price),
                execution_cost=fill_qty * abs(execution_price - arrival_price),
            )
            self._shortfalls.append(shortfall)

        # 按symbol统计
        sym_stat = self._symbol_stats[symbol]
        sym_stat["arrival_slips"].append(arrival_slip)
        sym_stat["vwap_slips"].append(vwap_slip)
        sym_stat["count"] += 1
        sym_stat["volume"] += fill_qty

        # 按策略统计
        if strategy:
            strat_stat = self._strategy_stats[strategy]
            strat_stat["arrival_slips"].append(arrival_slip)
            strat_stat["count"] += 1

    def record_cost(self, symbol: str, notional: float,
                     commission: float = 0.0,
                     market_impact: float = 0.0,
                     spread_cost: float = 0.0):
        """记录执行成本"""
        cost = ExecutionCostBreakdown(
            symbol=symbol,
            notional=notional,
            commission=commission,
            market_impact=market_impact,
            spread_cost=spread_cost,
            total_cost=commission + market_impact + spread_cost,
            total_cost_bps=(commission + market_impact + spread_cost) / max(notional, 1.0) * 10000,
        )
        self._costs.append(cost)

    # ── 指标计算 ──────────────────────────────────────────────

    def compute_quality_metrics(self, symbol: str = None,
                                 strategy: str = None) -> QualityMetrics:
        """计算执行质量指标"""
        # 选择数据源
        if symbol:
            slips = list(self._symbol_stats.get(symbol, {}).get("arrival_slips", []))
            vwaps = list(self._symbol_stats.get(symbol, {}).get("vwap_slips", []))
            volume = self._symbol_stats.get(symbol, {}).get("volume", 0.0)
            count = self._symbol_stats.get(symbol, {}).get("count", 0)
        elif strategy:
            slips = list(self._strategy_stats.get(strategy, {}).get("arrival_slips", []))
            vwaps = []
            volume = 0
            count = self._strategy_stats.get(strategy, {}).get("count", 0)
        else:
            slips = list(self._arrival_slips)
            vwaps = list(self._vwap_slips)
            volume = sum(q for q in [s.total_shortfall for s in self._shortfalls])
            count = len(slips)

        metrics = QualityMetrics(
            total_orders=count,
            total_volume=volume,
            total_notional=volume * 100,  # 近似
        )

        if not slips:
            return metrics

        arr = np.array(slips)
        metrics.avg_arrival_slippage_bps = float(np.mean(arr))
        metrics.arrival_slippage_p50 = float(np.percentile(arr, 50))
        metrics.arrival_slippage_p75 = float(np.percentile(arr, 75))
        metrics.arrival_slippage_p90 = float(np.percentile(arr, 90))
        metrics.arrival_slippage_p95 = float(np.percentile(arr, 95))

        if vwaps:
            v_arr = np.array(vwaps)
            metrics.avg_vwap_slippage_bps = float(np.mean(v_arr))

        # 成交量加权滑点
        if volume > 0:
            notional_weights = np.ones(len(arr)) * (volume / max(len(arr), 1))
            metrics.vwap_arrival_slippage_bps = float(np.average(arr, weights=notional_weights)
                                                      ) if len(arr) > 0 else 0

        # 实现缺口
        if self._shortfalls:
            sf_arr = np.array([s.total_shortfall_bps for s in self._shortfalls])
            metrics.avg_implementation_shortfall_bps = float(np.mean(np.abs(sf_arr)))

        # 成本
        if self._costs:
            cost_arr = np.array([c.total_cost_bps for c in self._costs])
            metrics.avg_total_cost_bps = float(np.mean(cost_arr))

        # 填充率
        filled_count = sum(1 for s in self._shortfalls if s.execution_price > 0)
        metrics.filled_orders = filled_count
        metrics.fill_rate_avg = filled_count / max(count, 1)

        # 质量评分
        self._compute_quality_grade(metrics)

        return metrics

    def _compute_quality_grade(self, metrics: QualityMetrics):
        """计算质量等级和评分"""
        p95 = abs(metrics.arrival_slippage_p95)

        if p95 <= self._benchmark_thresholds["excellent"]:
            metrics.quality_grade = QualityGrade.EXCELLENT
        elif p95 <= self._benchmark_thresholds["good"]:
            metrics.quality_grade = QualityGrade.GOOD
        elif p95 <= self._benchmark_thresholds["average"]:
            metrics.quality_grade = QualityGrade.AVERAGE
        elif p95 <= self._benchmark_thresholds["below"]:
            metrics.quality_grade = QualityGrade.BELOW_AVERAGE
        else:
            metrics.quality_grade = QualityGrade.POOR

        # 0-100评分（越低滑点越高分）
        thresholds = self._benchmark_thresholds
        score = 100.0 - (p95 - thresholds["excellent"]) / (thresholds["below"] - thresholds["excellent"]) * 100
        metrics.quality_score = round(max(0, min(100, score)), 1)

    # ── 报告生成 ──────────────────────────────────────────────

    def generate_report(self, period: str = "daily") -> ExecutionQualityReport:
        """生成执行质量报告"""
        now = datetime.now()
        report = ExecutionQualityReport(
            report_id=f"eq_{now.strftime('%Y%m%d_%H%M%S')}",
            period=period,
            start_time=(now - timedelta(days=1)).isoformat(),
            end_time=now.isoformat(),
        )

        report.summary = self.compute_quality_metrics()
        report.shortfall_analyses = list(self._shortfalls)[-50:]
        report.cost_breakdowns = list(self._costs)[-50:]

        # 生成建议
        recommendations = []
        if report.summary.quality_grade in (QualityGrade.BELOW_AVERAGE, QualityGrade.POOR):
            recommendations.append("RED ALERT: Execution quality below average — review algo parameters")
            if abs(report.summary.avg_arrival_slippage_bps) > 20:
                recommendations.append("Arrival slippage >20bps: consider using iceberg or TWAP for large orders")
            if report.summary.fill_rate_avg < 0.85:
                recommendations.append("Fill rate <85%: increase limit offset or use market order for urgency")
        elif report.summary.quality_grade == QualityGrade.AVERAGE:
            if abs(report.summary.arrival_slippage_p75) > 15:
                recommendations.append("P75 arrival slippage >15bps: review routing decisions for large orders")
        else:
            recommendations.append("Execution quality is within acceptable range")

        if report.summary.avg_total_cost_bps > 15:
            recommendations.append("Total cost >15bps: optimise fee tier or use internal crossing")

        report.recommendations = recommendations
        self._reports.append(report)

        logger.info(f"Execution quality report generated: grade={report.summary.quality_grade.value}, "
                    f"score={report.summary.quality_score}/100")

        return report

    # ── 按symbol的详细分析 ────────────────────────────────────

    def get_symbol_analysis(self, symbol: str) -> Dict[str, Any]:
        """获取单个symbol的执行质量分析"""
        metrics = self.compute_quality_metrics(symbol=symbol)
        stat = self._symbol_stats.get(symbol, {})
        return {
            "symbol": symbol,
            "total_orders": metrics.total_orders,
            "total_volume": round(metrics.total_volume, 4),
            "arrival_slip_avg_bps": round(metrics.avg_arrival_slippage_bps, 2),
            "arrival_slip_p50_bps": round(metrics.arrival_slippage_p50, 2),
            "arrival_slip_p95_bps": round(metrics.arrival_slippage_p95, 2),
            "vwap_slip_avg_bps": round(metrics.avg_vwap_slippage_bps, 2),
            "total_cost_avg_bps": round(metrics.avg_total_cost_bps, 2),
            "fill_rate": round(metrics.fill_rate_avg, 3),
            "quality_grade": metrics.quality_grade.value,
            "quality_score": metrics.quality_score,
        }

    def get_strategy_analysis(self, strategy: str) -> Dict[str, Any]:
        """获取单个策略的执行质量分析"""
        metrics = self.compute_quality_metrics(strategy=strategy)
        return {
            "strategy": strategy,
            "total_orders": metrics.total_orders,
            "arrival_slip_avg_bps": round(metrics.avg_arrival_slippage_bps, 2),
            "arrival_slip_p95_bps": round(metrics.arrival_slippage_p95, 2),
            "quality_grade": metrics.quality_grade.value,
            "quality_score": metrics.quality_score,
        }

    # ── 实时监控 ──────────────────────────────────────────────

    def check_alerts(self) -> List[Dict[str, Any]]:
        """检查是否需要告警"""
        alerts = []
        metrics = self.compute_quality_metrics()

        if metrics.quality_grade == QualityGrade.POOR:
            alerts.append({
                "severity": "critical",
                "message": f"Execution quality is POOR (score={metrics.quality_score}/100)",
                "p95_slip_bps": round(metrics.arrival_slippage_p95, 2),
                "recommendation": "Immediately review routing and algo parameters",
            })
        elif metrics.quality_grade == QualityGrade.BELOW_AVERAGE:
            alerts.append({
                "severity": "warning",
                "message": f"Execution quality below average (score={metrics.quality_score}/100)",
                "p95_slip_bps": round(metrics.arrival_slippage_p95, 2),
            })

        # 单个symbol检查
        for sym, stat in self._symbol_stats.items():
            slips = list(stat["arrival_slips"])
            if len(slips) >= 10:
                p95 = np.percentile(slips, 95)
                if abs(p95) > 30:
                    alerts.append({
                        "severity": "warning",
                        "message": f"Symbol {sym}: P95 arrival slip {p95:.1f}bps > 30bps",
                        "symbol": sym,
                        "p95_slip_bps": round(p95, 2),
                    })

        return alerts

    def get_status(self) -> Dict[str, Any]:
        metrics = self.compute_quality_metrics()
        return {
            "enabled": self._enabled,
            "rolling_window": self._rolling_window,
            "total_orders_tracked": metrics.total_orders,
            "quality_grade": metrics.quality_grade.value,
            "quality_score": metrics.quality_score,
            "arrival_slip_avg_bps": round(metrics.avg_arrival_slippage_bps, 2),
            "arrival_slip_p95_bps": round(metrics.arrival_slippage_p95, 2),
            "symbol_count": len(self._symbol_stats),
            "report_count": len(self._reports),
            "alerts": self.check_alerts(),
        }


# ═══════════════════════════════════════════════════════════════
# 行业最佳实践：增强指标 (Industry Best Practices)
# ═══════════════════════════════════════════════════════════════

@dataclass
class MarketImpactDecomposition:
    """Almgren-Chriss 框架：市场冲击分解

    将价格冲击分解为：
      - 永久冲击 (permanent): 信息泄漏，市场认为该笔交易包含信息，永久改变均衡价格
      - 临时冲击 (temporary): 流动性消耗，订单簿暂时失衡，价格会回归
    """
    symbol: str
    notional: float = 0.0
    quantity: float = 0.0
    # 实测
    total_impact_bps: float = 0.0          # 总冲击 = execution - arrival (decision)
    arrival_to_mid_bps: float = 0.0         # 到达价格 vs 决策价格 (信息泄漏)
    execution_to_arrival_bps: float = 0.0   # 执行 vs 到达 (流动性冲击)
    # 模型分解
    permanent_impact_bps: float = 0.0       # 永久：不可逆，信息泄漏
    temporary_impact_bps: float = 0.0       # 临时：可逆，流动性消耗后回归
    permanent_ratio: float = 0.0            # 永久占比
    # 参与率
    participation_rate_bps: float = 0.0     # 订单量 / 市场量 (bps)
    participation_bucket: str = "low"       # low / medium / high / aggressive
    # 预期冲击
    expected_impact_bps: float = 0.0        # 基于模型预期的冲击
    excess_impact_bps: float = 0.0          # 超额冲击 = 实际 - 预期

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "notional": round(self.notional, 2),
            "total_impact_bps": round(self.total_impact_bps, 2),
            "decomposition": {
                "permanent_bps": round(self.permanent_impact_bps, 2),
                "temporary_bps": round(self.temporary_impact_bps, 2),
                "permanent_ratio": round(self.permanent_ratio, 2),
            },
            "participation": {
                "rate_bps": round(self.participation_rate_bps, 1),
                "bucket": self.participation_bucket,
            },
            "expected_impact_bps": round(self.expected_impact_bps, 2),
            "excess_impact_bps": round(self.excess_impact_bps, 2),
        }


@dataclass
class IntervalVWAPBenchmark:
    """区间 VWAP 基准分析

    对标行业金标准：比较执行价 vs 执行窗口内的市场 VWAP
    不同于简单的 spot VWAP，这是精确匹配执行时间窗口的加权均价
    """
    symbol: str
    execution_vwap: float = 0.0           # 我们执行的 VWAP
    market_interval_vwap: float = 0.0     # 执行窗口内市场 VWAP
    market_daily_vwap: float = 0.0        # 全天市场 VWAP
    arrival_price: float = 0.0            # 下单时 mid 价格
    # 滑点
    vs_interval_vwap_bps: float = 0.0     # vs 区间 VWAP (最重要)
    vs_daily_vwap_bps: float = 0.0        # vs 全天 VWAP
    vs_arrival_bps: float = 0.0           # vs 到达价格
    # 执行质量
    interval_vwap_capture: float = 0.0    # VWAP 捕获率 (exe_price vs I-VWAP)
    is_significantly_worse: bool = False   # p<0.05 显著劣于 VWAP?
    # 窗口
    window_start: str = ""
    window_end: str = ""
    window_volume: float = 0.0            # 窗口内市场成交量
    our_volume: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "execution_vwap": round(self.execution_vwap, 4),
            "benchmarks": {
                "interval_vwap": round(self.market_interval_vwap, 4),
                "daily_vwap": round(self.market_daily_vwap, 4),
                "arrival_price": round(self.arrival_price, 4),
            },
            "slippage": {
                "vs_interval_vwap_bps": round(self.vs_interval_vwap_bps, 2),
                "vs_daily_vwap_bps": round(self.vs_daily_vwap_bps, 2),
                "vs_arrival_bps": round(self.vs_arrival_bps, 2),
            },
            "vwap_capture": round(self.interval_vwap_capture, 4),
            "execution_window": {
                "start": self.window_start,
                "end": self.window_end,
                "market_volume": round(self.window_volume, 4),
                "our_volume": round(self.our_volume, 4),
            },
        }


@dataclass
class ParticipationRateAnalysis:
    """参与率分析：衡量订单对市场的相对影响力

    行业标准：参与率 = 订单执行量 / 同期市场总成交量
    """
    symbol: str
    our_volume: float = 0.0              # 我们执行的量
    market_volume_interval: float = 0.0   # 执行期间市场成交量
    market_volume_daily: float = 0.0      # 全天市场成交量
    participation_rate_bps: float = 0.0   # 参与率 (bps)
    participation_rate_pct: float = 0.0   # 参与率 (%)
    # 分级
    participation_level: str = "low"       # low(<1%) / moderate(1-5%) / high(5-15%) / extreme(>15%)
    # 建议
    recommended_algo: str = ""
    max_slice_size_ratio: float = 0.0     # 最大切片/市场量建议

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "our_volume": round(self.our_volume, 4),
            "market_volume": {
                "interval": round(self.market_volume_interval, 4),
                "daily": round(self.market_volume_daily, 4),
            },
            "participation_rate": {
                "bps": round(self.participation_rate_bps, 1),
                "percent": round(self.participation_rate_pct, 2),
                "level": self.participation_level,
            },
            "recommended_algo": self.recommended_algo,
            "max_slice_size_ratio": round(self.max_slice_size_ratio, 4),
        }


@dataclass
class PostTradeDrift:
    """交易后价格回归分析：分离永久冲击 vs 临时冲击

    行业实践：观察执行后 5min / 15min / 60min 的价格变化
      - 如果价格回归 → 冲击主要是临时的（流动性），执行质量好
      - 如果价格不回 + 继续走 → 冲击是永久的（信息泄漏），执行质量差
    """
    symbol: str
    execution_avg_price: float = 0.0
    # 执行后时间点的 mid 价格
    price_at_execution_end: float = 0.0   # 执行结束时刻
    price_5min_after: float = 0.0
    price_15min_after: float = 0.0
    price_60min_after: float = 0.0
    # 漂移 (bps)
    drift_5min_bps: float = 0.0
    drift_15min_bps: float = 0.0
    drift_60min_bps: float = 0.0
    # 临时冲击回归率
    reversion_5min_pct: float = 0.0        # 5min内回归的百分比
    reversion_15min_pct: float = 0.0
    # 解释
    is_mostly_temporary: bool = True        # 主要临时性 → 执行质量好
    adverse_selection_bps: float = 0.0      # 逆向选择成本 (未回归部分)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "execution_price": round(self.execution_avg_price, 4),
            "post_trade_prices": {
                "at_execution_end": round(self.price_at_execution_end, 4),
                "plus_5min": round(self.price_5min_after, 4),
                "plus_15min": round(self.price_15min_after, 4),
                "plus_60min": round(self.price_60min_after, 4),
            },
            "drift_bps": {
                "plus_5min": round(self.drift_5min_bps, 2),
                "plus_15min": round(self.drift_15min_bps, 2),
                "plus_60min": round(self.drift_60min_bps, 2),
            },
            "reversion": {
                "pct_5min": round(self.reversion_5min_pct, 1),
                "pct_15min": round(self.reversion_15min_pct, 1),
            },
            "impact_type": "temporary" if self.is_mostly_temporary else "permanent",
            "adverse_selection_bps": round(self.adverse_selection_bps, 2),
        }


@dataclass
class RealizedSpreadAnalysis:
    """实现价差分析：衡量实际跨越买卖价差的成本

    行业标准：实现价差 = 2 * |成交价 - 成交后mid| * (side factor)
    有效价差 = 2 * |成交价 - 成交时mid|
    价差捕获比 = 实现价差 / 报价价差
    """
    symbol: str
    execution_price: float = 0.0
    mid_at_trade: float = 0.0             # 成交时 mid
    mid_after_trade: float = 0.0           # 成交后 mid (通常 1-2s)
    quoted_spread_bps: float = 0.0         # 报价价差
    effective_spread_bps: float = 0.0      # 有效价差 (成交 vs 成交时mid)
    realized_spread_bps: float = 0.0       # 实现价差 (成交 vs 成交后mid)
    price_impact_bps: float = 0.0          # 价格冲击 = 有效价差 - 实现价差
    spread_capture_ratio: float = 0.0      # 价差捕获率

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "spreads": {
                "quoted_bps": round(self.quoted_spread_bps, 1),
                "effective_bps": round(self.effective_spread_bps, 2),
                "realized_bps": round(self.realized_spread_bps, 2),
            },
            "price_impact_bps": round(self.price_impact_bps, 2),
            "spread_capture": round(self.spread_capture_ratio, 3),
        }


# ═══════════════════════════════════════════════════════════════
# 增强版执行质量监控器
# ═══════════════════════════════════════════════════════════════

class EnhancedExecutionQualityMonitor(ExecutionQualityMonitor):
    """增强版执行质量监控器 — 对标行业最佳实践"""

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        cfg = config.get("execution_quality_monitor", {}) if config else {}

        # 增强指标存储
        self._market_impacts: deque = deque(maxlen=self._rolling_window)
        self._interval_vwaps: deque = deque(maxlen=self._rolling_window)
        self._participation_rates: deque = deque(maxlen=self._rolling_window)
        self._post_trade_drifts: deque = deque(maxlen=self._rolling_window)
        self._realized_spreads: deque = deque(maxlen=self._rolling_window)
        # 按 algo 类型分组
        self._algo_stats: Dict[str, Dict] = defaultdict(lambda: {
            "impacts": deque(maxlen=500),
            "vwaps": deque(maxlen=500),
            "drifts": deque(maxlen=500),
            "count": 0,
        })
        # 按 symbol 的增强统计
        self._symbol_enhanced: Dict[str, Dict] = defaultdict(lambda: {
            "participation_rates": deque(maxlen=200),
            "post_trade_drifts": deque(maxlen=200),
            "realized_spreads": deque(maxlen=200),
        })

    # ── 增强数据记录 ──────────────────────────────────────────

    def record_execution_enhanced(self, symbol: str, side: str, quantity: float,
                                   execution_price: float, arrival_price: float,
                                   decision_price: float = 0.0,
                                   market_vwap_interval: float = 0.0,
                                   market_volume_interval: float = 0.0,
                                   market_volume_daily: float = 0.0,
                                   interval_start: str = "",
                                   interval_end: str = "",
                                   participation_bps: float = 0.0,
                                   price_before: float = 0.0,
                                   price_5min_after: float = 0.0,
                                   price_15min_after: float = 0.0,
                                   price_60min_after: float = 0.0,
                                   mid_at_trade: float = 0.0,
                                   quoted_spread_bps: float = 0.0,
                                   algo_type: str = "",
                                   strategy: str = ""):
        """记录增强版执行质量数据（含行业标准上下文）"""
        # 先调用基类的记录
        self.record_execution(
            symbol, side, quantity, execution_price, arrival_price,
            decision_price, market_vwap_interval, 0.0, strategy, True, quantity
        )

        notional = quantity * execution_price
        side_mult = -1 if side == "sell" else 1

        # ─ 1. 市场冲击分解 ─
        if decision_price > 0 and arrival_price > 0:
            arrival_to_mid = side_mult * (arrival_price - decision_price) / decision_price * 10000
            exec_to_arrival = side_mult * (execution_price - arrival_price) / arrival_price * 10000
            total_impact = side_mult * (execution_price - decision_price) / decision_price * 10000

            # Almgren-Chriss 启发式：永久冲击 ≈ 信息泄漏 (arrival→mid)，临时 ≈ 流动性 (exec→arrival)
            perm_impact = arrival_to_mid
            temp_impact = exec_to_arrival
            perm_ratio = abs(perm_impact) / max(abs(total_impact), 1e-10) if total_impact != 0 else 0

            impact = MarketImpactDecomposition(
                symbol=symbol,
                notional=notional,
                quantity=quantity,
                total_impact_bps=total_impact,
                arrival_to_mid_bps=arrival_to_mid,
                execution_to_arrival_bps=exec_to_arrival,
                permanent_impact_bps=perm_impact,
                temporary_impact_bps=temp_impact,
                permanent_ratio=min(1.0, perm_ratio),
                participation_rate_bps=participation_bps,
                participation_bucket=self._classify_participation(participation_bps),
                expected_impact_bps=self._estimate_impact(participation_bps),
                excess_impact_bps=total_impact - self._estimate_impact(participation_bps),
            )
            self._market_impacts.append(impact)

        # ─ 2. 区间 VWAP 基准 ─
        if market_vwap_interval > 0:
            ivwap = IntervalVWAPBenchmark(
                symbol=symbol,
                execution_vwap=execution_price,
                market_interval_vwap=market_vwap_interval,
                market_daily_vwap=market_vwap_interval,  # 近似
                arrival_price=arrival_price,
                vs_interval_vwap_bps=side_mult * (execution_price - market_vwap_interval) / market_vwap_interval * 10000,
                vs_arrival_bps=side_mult * (execution_price - arrival_price) / arrival_price * 10000 if arrival_price > 0 else 0,
                interval_vwap_capture=execution_price / max(market_vwap_interval, 1e-10),
                window_start=interval_start,
                window_end=interval_end,
                window_volume=market_volume_interval,
                our_volume=quantity,
            )
            self._interval_vwaps.append(ivwap)

        # ─ 3. 参与率分析 ─
        if market_volume_interval > 0:
            prate = ParticipationRateAnalysis(
                symbol=symbol,
                our_volume=quantity,
                market_volume_interval=market_volume_interval,
                market_volume_daily=market_volume_daily,
                participation_rate_bps=participation_bps if participation_bps > 0 else (
                    quantity / market_volume_interval * 10000),
                participation_rate_pct=quantity / max(market_volume_interval, 1e-10) * 100,
                participation_level=self._classify_participation(
                    quantity / max(market_volume_interval, 1e-10) * 10000),
            )
            self._participation_rates.append(prate)

        # ─ 4. 交易后价格回归 ─
        if price_5min_after > 0 or price_15min_after > 0:
            drift_5 = side_mult * (price_5min_after - execution_price) / execution_price * 10000 if price_5min_after > 0 else 0
            drift_15 = side_mult * (price_15min_after - execution_price) / execution_price * 10000 if price_15min_after > 0 else 0
            drift_60 = side_mult * (price_60min_after - execution_price) / execution_price * 10000 if price_60min_after > 0 else 0

            # 回归：如果价格往回走（对我们有利），则是临时冲击在回归
            total_impact = side_mult * (execution_price - max(price_before, arrival_price, 1e-10)) / max(price_before, arrival_price, 1e-10) * 10000
            reversion_5 = max(0, -drift_5 / max(abs(total_impact), 1e-10)) if total_impact > 0 else 0
            reversion_15 = max(0, -drift_15 / max(abs(total_impact), 1e-10)) if total_impact > 0 else 0

            drift = PostTradeDrift(
                symbol=symbol,
                execution_avg_price=execution_price,
                price_at_execution_end=execution_price,
                price_5min_after=price_5min_after,
                price_15min_after=price_15min_after,
                price_60min_after=price_60min_after,
                drift_5min_bps=drift_5,
                drift_15min_bps=drift_15,
                drift_60min_bps=drift_60,
                reversion_5min_pct=reversion_5 * 100,
                reversion_15min_pct=reversion_15 * 100,
                is_mostly_temporary=(reversion_15 > 0.5),  # >50%回归 → 临时性为主
                adverse_selection_bps=side_mult * (execution_price - max(price_15min_after, 1e-10)) / execution_price * 10000 if price_15min_after > 0 else 0,
            )
            self._post_trade_drifts.append(drift)

        # ─ 5. 实现价差 ─
        if mid_at_trade > 0 and quoted_spread_bps > 0:
            eff = 2 * abs(execution_price - mid_at_trade) / mid_at_trade * 10000
            realized = RealizedSpreadAnalysis(
                symbol=symbol,
                execution_price=execution_price,
                mid_at_trade=mid_at_trade,
                quoted_spread_bps=quoted_spread_bps,
                effective_spread_bps=eff,
                realized_spread_bps=eff,  # 近似
                price_impact_bps=eff,
                spread_capture_ratio=eff / max(quoted_spread_bps, 1e-10),
            )
            self._realized_spreads.append(realized)

    # ── 分类与估算 ────────────────────────────────────────────

    @staticmethod
    def _classify_participation(participation_bps: float) -> str:
        if participation_bps <= 50:
            return "low"
        elif participation_bps <= 250:
            return "moderate"
        elif participation_bps <= 750:
            return "high"
        return "aggressive"

    @staticmethod
    def _estimate_impact(participation_bps: float) -> float:
        """基于参与率的简化冲击模型  sqrt(participation/10000) * 15bps """
        return np.sqrt(max(participation_bps, 0.1) / 10000) * 15

    # ── 增强分析 ──────────────────────────────────────────────

    def get_market_impact_summary(self, symbol: str = None) -> Dict[str, Any]:
        """市场冲击汇总"""
        impacts = list(self._market_impacts)
        if symbol:
            impacts = [i for i in impacts if i.symbol == symbol]
        if not impacts:
            return {"count": 0}

        total_arr = np.array([i.total_impact_bps for i in impacts])
        perm_arr = np.array([i.permanent_impact_bps for i in impacts])
        temp_arr = np.array([i.temporary_impact_bps for i in impacts])
        part_arr = np.array([i.participation_rate_bps for i in impacts])

        return {
            "count": len(impacts),
            "total_impact": {
                "avg_bps": round(float(np.mean(np.abs(total_arr))), 2),
                "p50_bps": round(float(np.percentile(np.abs(total_arr), 50)), 2),
                "p95_bps": round(float(np.percentile(np.abs(total_arr), 95)), 2),
            },
            "decomposition": {
                "permanent_avg_bps": round(float(np.mean(np.abs(perm_arr))), 2),
                "temporary_avg_bps": round(float(np.mean(np.abs(temp_arr))), 2),
                "permanent_ratio_avg": round(float(np.mean([i.permanent_ratio for i in impacts])), 3),
            },
            "excess_impact_avg_bps": round(float(np.mean([i.excess_impact_bps for i in impacts])), 2),
            "participation": {
                "avg_bps": round(float(np.mean(part_arr)), 1),
                "by_bucket": dict(Counter(i.participation_bucket for i in impacts)),
            },
        }

    def get_interval_vwap_summary(self, symbol: str = None) -> Dict[str, Any]:
        """区间 VWAP 基准汇总"""
        ivwaps = list(self._interval_vwaps)
        if symbol:
            ivwaps = [i for i in ivwaps if i.symbol == symbol]
        if not ivwaps:
            return {"count": 0}

        slips = np.array([i.vs_interval_vwap_bps for i in ivwaps])
        captures = np.array([i.interval_vwap_capture for i in ivwaps])

        return {
            "count": len(ivwaps),
            "vs_interval_vwap": {
                "avg_bps": round(float(np.mean(slips)), 2),
                "p50_bps": round(float(np.percentile(slips, 50)), 2),
                "p95_bps": round(float(np.percentile(slips, 95)), 2),
            },
            "vwap_capture": {
                "avg": round(float(np.mean(captures)), 4),
                "better_than_vwap_count": int(np.sum(captures < 1.0)) if len(captures) > 0 else 0,
            },
            "total_volume": round(sum(i.our_volume for i in ivwaps), 4),
        }

    def get_post_trade_analysis(self, symbol: str = None) -> Dict[str, Any]:
        """交易后价格回归分析"""
        drifts = list(self._post_trade_drifts)
        if symbol:
            drifts = [d for d in drifts if d.symbol == symbol]
        if not drifts:
            return {"count": 0}

        d5 = np.array([d.drift_5min_bps for d in drifts if d.drift_5min_bps != 0])
        d15 = np.array([d.drift_15min_bps for d in drifts if d.drift_15min_bps != 0])
        d60 = np.array([d.drift_60min_bps for d in drifts if d.drift_60min_bps != 0])
        rev5 = np.array([d.reversion_5min_pct for d in drifts])
        rev15 = np.array([d.reversion_15min_pct for d in drifts])

        temporary_count = sum(1 for d in drifts if d.is_mostly_temporary)

        return {
            "count": len(drifts),
            "drift": {
                "5min_avg_bps": round(float(np.mean(d5)), 2) if len(d5) > 0 else 0,
                "15min_avg_bps": round(float(np.mean(d15)), 2) if len(d15) > 0 else 0,
                "60min_avg_bps": round(float(np.mean(d60)), 2) if len(d60) > 0 else 0,
            },
            "reversion": {
                "5min_avg_pct": round(float(np.mean(rev5)), 1) if len(rev5) > 0 else 0,
                "15min_avg_pct": round(float(np.mean(rev15)), 1) if len(rev15) > 0 else 0,
            },
            "impact_type_distribution": {
                "temporary": round(temporary_count / len(drifts) * 100, 1),
                "permanent": round((len(drifts) - temporary_count) / len(drifts) * 100, 1),
            },
            "adverse_selection_avg_bps": round(float(np.mean([d.adverse_selection_bps for d in drifts if d.adverse_selection_bps != 0])), 2) if drifts else 0,
        }

    def get_participation_analysis(self, symbol: str = None) -> Dict[str, Any]:
        """参与率分析"""
        prates = list(self._participation_rates)
        if symbol:
            prates = [p for p in prates if p.symbol == symbol]
        if not prates:
            return {"count": 0}

        rates = np.array([p.participation_rate_bps for p in prates])
        from collections import Counter as _Counter
        levels = _Counter(p.participation_level for p in prates)

        return {
            "count": len(prates),
            "participation": {
                "avg_bps": round(float(np.mean(rates)), 1),
                "p50_bps": round(float(np.percentile(rates, 50)), 1),
                "p95_bps": round(float(np.percentile(rates, 95)), 1),
            },
            "level_distribution": dict(levels),
        }

    def get_realized_spread_summary(self, symbol: str = None) -> Dict[str, Any]:
        """实现价差汇总"""
        spreads = list(self._realized_spreads)
        if symbol:
            spreads = [s for s in spreads if s.symbol == symbol]
        if not spreads:
            return {"count": 0}

        eff = np.array([s.effective_spread_bps for s in spreads])
        real = np.array([s.realized_spread_bps for s in spreads])
        capture = np.array([s.spread_capture_ratio for s in spreads])

        return {
            "count": len(spreads),
            "spreads": {
                "effective_avg_bps": round(float(np.mean(eff)), 2),
                "realized_avg_bps": round(float(np.mean(real)), 2),
            },
            "spread_capture_avg": round(float(np.mean(capture)), 3),
            "price_impact_avg_bps": round(float(np.mean([s.price_impact_bps for s in spreads])), 2),
        }

    # ── 增强报告 ──────────────────────────────────────────────

    def generate_enhanced_report(self, period: str = "daily",
                                  symbol: str = None) -> Dict[str, Any]:
        """生成增强版执行质量报告（行业对标）"""
        base_report = super().generate_report(period)

        return {
            "report_id": base_report.report_id,
            "period": period,
            "period_range": f"{base_report.start_time} ~ {base_report.end_time}",
            "timestamp": base_report.timestamp,
            # ─ 行业标准指标 ─
            "implementation_shortfall": {
                "avg_bps": round(base_report.summary.avg_implementation_shortfall_bps, 2),
                "decomposition": "decision_price → arrival (delay cost) → execution (execution cost + market impact)",
            },
            "arrival_slippage": {
                "avg_bps": round(base_report.summary.avg_arrival_slippage_bps, 2),
                "p50_bps": round(base_report.summary.arrival_slippage_p50, 2),
                "p75_bps": round(base_report.summary.arrival_slippage_p75, 2),
                "p90_bps": round(base_report.summary.arrival_slippage_p90, 2),
                "p95_bps": round(base_report.summary.arrival_slippage_p95, 2),
            },
            "vwap_slippage": {
                "avg_bps": round(base_report.summary.avg_vwap_slippage_bps, 2),
            },
            # ─ 增强指标 ─
            "interval_vwap": self.get_interval_vwap_summary(symbol),
            "market_impact": self.get_market_impact_summary(symbol),
            "participation": self.get_participation_analysis(symbol),
            "post_trade_drift": self.get_post_trade_analysis(symbol),
            "realized_spread": self.get_realized_spread_summary(symbol),
            # ─ 质量评估 ─
            "quality": {
                "grade": base_report.summary.quality_grade.value,
                "score": base_report.summary.quality_score,
            },
            "cost": {
                "avg_total_cost_bps": round(base_report.summary.avg_total_cost_bps, 2),
                "fill_rate": round(base_report.summary.fill_rate_avg, 3),
            },
            "recommendations": base_report.recommendations,
        }

    def get_enhanced_status(self) -> Dict[str, Any]:
        """获取增强监控状态"""
        base = super().get_status()
        base["enhanced"] = {
            "market_impacts_tracked": len(self._market_impacts),
            "interval_vwaps_tracked": len(self._interval_vwaps),
            "participation_rates_tracked": len(self._participation_rates),
            "post_trade_drifts_tracked": len(self._post_trade_drifts),
            "realized_spreads_tracked": len(self._realized_spreads),
        }
        return base
