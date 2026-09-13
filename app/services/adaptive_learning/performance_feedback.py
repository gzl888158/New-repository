"""
性能反馈系统 (Performance Feedback System)

提供多维度策略评分、归因分析、闭环反馈、策略比较和告警功能。
所有数学计算均内联实现，不依赖外部 ML/统计库。
"""

import asyncio
import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from app.services.enterprise import EnterpriseServiceMixin


# =============================================================================
# Enums
# =============================================================================

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ScoreDimension(Enum):
    RETURNS = "returns"
    RISK = "risk"
    CONSISTENCY = "consistency"
    EFFICIENCY = "efficiency"
    ADAPTABILITY = "adaptability"


# =============================================================================
# Performance Metrics Dataclass
# =============================================================================

@dataclass
class PerformanceMetrics:
    """策略性能指标数据类"""

    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win_loss_ratio: float = 0.0
    expectancy: float = 0.0
    kelly_fraction: float = 0.0
    daily_volatility: float = 0.0
    downside_deviation: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    trade_count: int = 0
    avg_holding_time: float = 0.0
    consistency_score: float = 0.0
    efficiency_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# Math Helpers (inline implementations)
# =============================================================================

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: List[float], ddof: int = 1) -> float:
    n = len(values)
    if n <= ddof:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (n - ddof)


def _std(values: List[float], ddof: int = 1) -> float:
    return math.sqrt(_variance(values, ddof))


def _covariance(x: List[float], y: List[float]) -> float:
    n = min(len(x), len(y))
    if n <= 1:
        return 0.0
    mx = _mean(x)
    my = _mean(y)
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / (n - 1)


def _correlation(x: List[float], y: List[float]) -> float:
    sx = _std(x)
    sy = _std(y)
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return _covariance(x, y) / (sx * sy)


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    k = (p / 100.0) * (n - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    d0 = sorted_vals[int(f)] * (c - k)
    d1 = sorted_vals[int(c)] * (k - f)
    return d0 + d1


def _sharpe(returns: List[float], risk_free: float = 0.0, periods_per_year: int = 365) -> float:
    if not returns or len(returns) < 2:
        return 0.0
    avg_ret = _mean(returns) - risk_free / periods_per_year
    s = _std(returns)
    if s == 0.0:
        return 0.0
    return (avg_ret / s) * math.sqrt(periods_per_year)


def _sortino(returns: List[float], risk_free: float = 0.0, periods_per_year: int = 365) -> float:
    if not returns:
        return 0.0
    mar = risk_free / periods_per_year
    downside = [r - mar for r in returns if r < mar]
    if len(downside) < 2:
        return 0.0
    avg_ret = _mean(returns) - mar
    dd = math.sqrt(sum(x ** 2 for x in downside) / len(downside))
    if dd == 0.0:
        return 0.0
    return (avg_ret / dd) * math.sqrt(periods_per_year)


def _max_drawdown(equity: List[float]) -> Tuple[float, int]:
    if not equity:
        return 0.0, 0
    peak = equity[0]
    max_dd = 0.0
    dd_start: Optional[int] = None
    max_dd_duration = 0
    for i, val in enumerate(equity):
        if val >= peak:
            peak = val
            dd_start = None
        else:
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
            if dd_start is None:
                dd_start = i
            duration = i - dd_start
            if duration > max_dd_duration:
                max_dd_duration = duration
    return max_dd, max_dd_duration


def _calmar(annualized_return: float, max_dd: float) -> float:
    if max_dd == 0.0:
        return 0.0
    return annualized_return / max_dd


# =============================================================================
# PerformanceMetrics 计算引擎
# =============================================================================

def compute_metrics(
    trades: List[Dict[str, Any]],
    equity_curve: List[Dict[str, Any]],
) -> PerformanceMetrics:
    """从交易记录和权益曲线计算所有性能指标"""

    if not trades and not equity_curve:
        return PerformanceMetrics()

    # 提取权益值序列
    equity_values = [e.get("equity", 0.0) if isinstance(e, dict) else float(e) for e in equity_curve]

    # 日收益率
    daily_returns: List[float] = []
    for i in range(1, len(equity_values)):
        if equity_values[i - 1] > 0:
            daily_returns.append((equity_values[i] - equity_values[i - 1]) / equity_values[i - 1])

    # 交易盈亏
    trade_pnls: List[float] = []
    trade_wins: List[float] = []
    trade_losses: List[float] = []
    holding_times: List[float] = []

    for t in trades:
        pnl = t.get("pnl", t.get("realized_pnl", 0.0))
        trade_pnls.append(pnl)
        if pnl > 0:
            trade_wins.append(pnl)
        elif pnl < 0:
            trade_losses.append(abs(pnl))
        holding = t.get("holding_time", t.get("duration", 0.0))
        if holding:
            holding_times.append(holding)

    # ---- 基础指标 ----
    total_return = (equity_values[-1] / equity_values[0] - 1.0) if equity_values and equity_values[0] > 0 else 0.0

    days = max(len(daily_returns), 1)
    annualized_return = ((1.0 + total_return) ** (365.0 / days) - 1.0) if total_return > -1.0 else 0.0

    sharpe_ratio = _sharpe(daily_returns) if daily_returns else 0.0
    sortino_ratio = _sortino(daily_returns) if daily_returns else 0.0
    max_dd, max_dd_duration = _max_drawdown(equity_values) if equity_values else (0.0, 0)
    calmar_ratio = _calmar(annualized_return, max_dd)

    # ---- 交易指标 ----
    trade_count = len(trade_pnls)
    if trade_count > 0:
        win_count = len(trade_wins)
        loss_count = len(trade_losses)
        win_rate = win_count / trade_count if trade_count > 0 else 0.0

        total_profit = sum(trade_wins)
        total_loss = sum(trade_losses)
        profit_factor = total_profit / total_loss if total_loss > 0 else (float("inf") if total_profit > 0 else 0.0)

        avg_win = _mean(trade_wins) if trade_wins else 0.0
        avg_loss = _mean(trade_losses) if trade_losses else 0.0
        avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (float("inf") if avg_win > 0 else 0.0)

        expectancy = (win_rate * avg_win - (1.0 - win_rate) * avg_loss) if avg_loss > 0 else avg_win * win_rate

        avg_pnl = _mean(trade_pnls)
        pnl_std = _std(trade_pnls)
        if pnl_std > 0 and avg_pnl > 0:
            kelly_fraction = avg_pnl / (pnl_std ** 2)
            kelly_fraction = max(0.0, min(kelly_fraction, 1.0))
        else:
            kelly_fraction = 0.0
    else:
        win_rate = 0.0
        profit_factor = 0.0
        avg_win_loss_ratio = 0.0
        expectancy = 0.0
        kelly_fraction = 0.0

    # ---- 风险指标 ----
    daily_volatility = _std(daily_returns) * math.sqrt(365) if daily_returns else 0.0

    downside_returns = [r for r in daily_returns if r < 0]
    downside_deviation = _std(downside_returns) * math.sqrt(365) if len(downside_returns) >= 2 else 0.0

    if daily_returns:
        var_95 = _percentile(daily_returns, 5.0)
        below_var = [r for r in daily_returns if r <= var_95]
        cvar_95 = _mean(below_var) if below_var else var_95
    else:
        var_95 = 0.0
        cvar_95 = 0.0

    avg_holding_time = _mean(holding_times) if holding_times else 0.0

    # ---- 一致性得分 ----
    consistency_score = _compute_consistency(daily_returns, trade_pnls)

    # ---- 效率得分 ----
    efficiency_score = _compute_efficiency(total_return, max_dd, daily_volatility, trade_count)

    # 安全截断无穷值
    def _safe(value: float) -> float:
        if math.isinf(value) or math.isnan(value):
            return 0.0
        return value

    return PerformanceMetrics(
        total_return=_safe(total_return),
        annualized_return=_safe(annualized_return),
        sharpe_ratio=_safe(sharpe_ratio),
        sortino_ratio=_safe(sortino_ratio),
        calmar_ratio=_safe(calmar_ratio),
        max_drawdown=_safe(max_dd),
        max_drawdown_duration=max_dd_duration,
        win_rate=_safe(win_rate),
        profit_factor=_safe(profit_factor),
        avg_win_loss_ratio=_safe(avg_win_loss_ratio),
        expectancy=_safe(expectancy),
        kelly_fraction=_safe(kelly_fraction),
        daily_volatility=_safe(daily_volatility),
        downside_deviation=_safe(downside_deviation),
        var_95=_safe(var_95),
        cvar_95=_safe(cvar_95),
        trade_count=trade_count,
        avg_holding_time=_safe(avg_holding_time),
        consistency_score=_safe(consistency_score),
        efficiency_score=_safe(efficiency_score),
    )


def _compute_consistency(daily_returns: List[float], trade_pnls: List[float]) -> float:
    """计算收益稳定性得分 (0-1)"""
    score = 0.0
    weights = 0.0

    # 因子1: 滚动 Sharpe 稳定性
    if len(daily_returns) >= 20:
        window = min(20, len(daily_returns) // 2)
        rolling_sharpes: List[float] = []
        for i in range(len(daily_returns) - window + 1):
            window_returns = daily_returns[i:i + window]
            if len(window_returns) >= 2:
                avg = _mean(window_returns)
                s = _std(window_returns)
                if s > 0:
                    rolling_sharpes.append(avg / s)
        if rolling_sharpes:
            sharpe_cv = _std(rolling_sharpes) / (abs(_mean(rolling_sharpes)) + 1e-9)
            score += max(0.0, 1.0 - min(sharpe_cv, 1.0)) * 0.40
            weights += 0.40

    # 因子2: 连续盈亏模式
    if len(trade_pnls) >= 10:
        streaks: List[int] = []
        current_streak = 0
        for pnl in trade_pnls:
            if pnl >= 0:
                current_streak = current_streak + 1 if current_streak >= 0 else 1
            else:
                current_streak = current_streak - 1 if current_streak <= 0 else -1
        # 理想情况：有规律地交替
        sign_changes = sum(1 for i in range(1, len(trade_pnls)) if trade_pnls[i] * trade_pnls[i - 1] < 0)
        expected_changes = (len(trade_pnls) - 1) * 0.5
        if expected_changes > 0:
            pattern_score = 1.0 - abs(sign_changes / expected_changes - 1.0)
            score += max(0.0, pattern_score) * 0.30
            weights += 0.30

    # 因子3: 收益率分布偏度（正偏度 = 好）
    if len(daily_returns) >= 5:
        avg = _mean(daily_returns)
        s = _std(daily_returns)
        if s > 0:
            skew = sum(((r - avg) / s) ** 3 for r in daily_returns) / len(daily_returns)
            # 正偏度更好；映射到 0-1
            skew_score = min(1.0, max(0.0, (skew + 3.0) / 6.0))
            score += skew_score * 0.30
            weights += 0.30

    return score / weights if weights > 0 else 0.5


def _compute_efficiency(
    total_return: float,
    max_drawdown: float,
    daily_volatility: float,
    trade_count: int,
) -> float:
    """计算效率得分 (0-1)：单位风险/资本的回报"""
    score = 0.0
    weights = 0.0

    # 回报/最大回撤比
    if max_drawdown > 0:
        return_to_dd = abs(total_return) / max_drawdown
        dd_score = min(1.0, return_to_dd / 2.0)
        score += dd_score * 0.40
        weights += 0.40
    elif total_return > 0:
        score += 0.40
        weights += 0.40

    # 回报/波动率比
    if daily_volatility > 0:
        return_to_vol = abs(annualized_return_without_call(total_return, 1) / daily_volatility) if daily_volatility > 0 else 0.0
        vol_score = min(1.0, return_to_vol / 1.5)
        score += vol_score * 0.35
        weights += 0.35

    # 每笔交易的平均收益贡献（避免过度交易惩罚好策略）
    if trade_count > 0:
        avg_contribution = abs(total_return) / max(trade_count, 1)
        contrib_score = min(1.0, avg_contribution * 100)
        score += contrib_score * 0.25
        weights += 0.25

    return score / weights if weights > 0 else 0.0


def annualized_return_without_call(total_return: float, years: float) -> float:
    if years <= 0 or total_return <= -1.0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


# =============================================================================
# Performance Scorer: 多维度评分系统
# =============================================================================

class PerformanceScorer:
    """多维度策略评分系统

    五个评分维度（每个 0-100）：
    - Returns:    收益（归一化到基准和时间）
    - Risk:       风险（回撤、VaR、波动率）
    - Consistency: 一致性（滚动 Sharpe 稳定性、连续盈亏）
    - Efficiency:  效率（单位资本/保证金/风险回报）
    - Adaptability: 适应性（策略对市场变化的适应程度）
    """

    DEFAULT_WEIGHTS: Dict[str, float] = {
        "returns": 0.25,
        "risk": 0.25,
        "consistency": 0.15,
        "efficiency": 0.20,
        "adaptability": 0.15,
    }

    def __init__(self, config: Dict[str, Any]):
        pf_config = config.get("performance_feedback", {})
        scoring_config = pf_config.get("scoring", {})
        self._weights = scoring_config.get("weights", dict(self.DEFAULT_WEIGHTS))
        self._rolling_windows = scoring_config.get("rolling_windows", ["24h", "7d", "30d", "90d"])
        self._degradation_threshold = scoring_config.get("degradation_threshold", 20.0)
        self._degradation_enabled = scoring_config.get("degradation_enabled", True)
        self._score_history: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._benchmark_returns: Dict[str, float] = {}
        self._lock = asyncio.Lock()

    def set_benchmark_return(self, benchmark: str, annualized_return: float):
        self._benchmark_returns[benchmark] = annualized_return

    async def score(
        self,
        metrics: PerformanceMetrics,
        strategy: str,
        performance_history: Optional[List[Dict[str, Any]]] = None,
        market_regimes: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """计算多维度评分"""
        async with self._lock:
            dimensions = {
                "returns": self._score_returns(metrics),
                "risk": self._score_risk(metrics),
                "consistency": self._score_consistency(metrics),
                "efficiency": self._score_efficiency(metrics),
                "adaptability": self._score_adaptability(metrics, performance_history or [], market_regimes or []),
            }

            composite = sum(
                dimensions[dim] * self._weights.get(dim, 0.20)
                for dim in dimensions
            )
            total_weight = sum(self._weights.get(dim, 0.0) for dim in dimensions)
            if total_weight > 0:
                composite /= total_weight

            timestamp = datetime.now().isoformat()
            record = {
                "timestamp": timestamp,
                "strategy": strategy,
                "composite": round(composite, 2),
                "dimensions": {k: round(v, 2) for k, v in dimensions.items()},
            }
            self._score_history[strategy]["all"].append(record)

            # 检查分数退化
            degradation_alerts = []
            if self._degradation_enabled:
                alerts = self._check_degradation(strategy, composite)
                degradation_alerts.extend(alerts)

            return {
                "strategy": strategy,
                "composite_score": round(composite, 2),
                "dimensions": {k: round(v, 2) for k, v in dimensions.items()},
                "weights": dict(self._weights),
                "degradation_alerts": degradation_alerts,
                "timestamp": timestamp,
            }

    async def get_rolling_scores(
        self, strategy: str, window: str = "30d"
    ) -> Dict[str, Any]:
        """获取滚动窗口评分"""
        async with self._lock:
            all_history = self._score_history.get(strategy, {}).get("all", [])
            now = datetime.now()
            window_seconds = self._parse_window_seconds(window)
            cutoff = now - timedelta(seconds=window_seconds)

            window_records = []
            for rec in all_history:
                try:
                    ts = datetime.fromisoformat(rec["timestamp"])
                    if ts >= cutoff:
                        window_records.append(rec)
                except (ValueError, KeyError):
                    continue

            if not window_records:
                return {"strategy": strategy, "window": window, "scores": [], "avg_composite": 0.0, "trend": "flat"}

            scores = [r["composite"] for r in window_records]
            avg = _mean(scores)

            if len(scores) >= 3:
                first_3_avg = _mean(scores[:min(3, len(scores))])
                last_3_avg = _mean(scores[-min(3, len(scores)):])
                diff = last_3_avg - first_3_avg
                if diff > 5:
                    trend = "improving"
                elif diff < -5:
                    trend = "declining"
                else:
                    trend = "stable"
            else:
                trend = "flat"

            return {
                "strategy": strategy,
                "window": window,
                "scores": window_records[-20:],
                "avg_composite": round(avg, 2),
                "min_composite": round(min(scores), 2),
                "max_composite": round(max(scores), 2),
                "trend": trend,
                "count": len(window_records),
            }

    def _score_returns(self, metrics: PerformanceMetrics) -> float:
        """收益维度评分 (0-100)"""
        ar = metrics.annualized_return
        # 无风险利率约 3-5%，超额收益
        excess = ar - 0.03
        if excess <= 0:
            return max(0.0, 30.0 + excess * 1000)  # 略低于无风险：0-30
        # 对数映射：10% 超额 ≈ 70, 30% 超额 ≈ 90, 100%+ ≈ 100
        if excess < 1.0:
            return min(100.0, 50.0 + math.log1p(excess) * 25.0)
        else:
            return min(100.0, 50.0 + math.log(excess) * 10.0 + 25.0)

    def _score_risk(self, metrics: PerformanceMetrics) -> float:
        """风险维度评分 (0-100)（越低风险越高分）"""
        score = 100.0
        # 最大回撤惩罚：每 1% 回撤扣 1.5 分
        score -= metrics.max_drawdown * 150.0
        # 波动率惩罚：每 1% 日波动扣 1 分
        score -= metrics.daily_volatility * 100.0
        # VaR 惩罚
        score -= abs(metrics.var_95) * 200.0
        # 下行偏差惩罚
        score -= metrics.downside_deviation * 80.0
        return max(0.0, min(100.0, score))

    def _score_consistency(self, metrics: PerformanceMetrics) -> float:
        """一致性维度评分 (0-100)"""
        return metrics.consistency_score * 100.0

    def _score_efficiency(self, metrics: PerformanceMetrics) -> float:
        """效率维度评分 (0-100)"""
        return metrics.efficiency_score * 100.0

    def _score_adaptability(
        self,
        metrics: PerformanceMetrics,
        history: List[Dict[str, Any]],
        regimes: List[Dict[str, Any]],
    ) -> float:
        """适应性维度评分 (0-100)"""
        score = 50.0  # 基础分

        # 因子1: 不同市场状态下的表现稳定性
        if regimes and len(regimes) >= 2:
            regime_pnls: Dict[str, List[float]] = defaultdict(list)
            for entry in regimes:
                regime = entry.get("regime", "unknown")
                pnl = entry.get("pnl", entry.get("return", 0.0))
                regime_pnls[regime].append(pnl)

            if len(regime_pnls) >= 2:
                regime_avgs = [_mean(v) for v in regime_pnls.values()]
                avg_of_avgs = _mean(regime_avgs) if regime_avgs else 0.0
                if abs(avg_of_avgs) > 1e-9:
                    regime_cv = _std(regime_avgs) / abs(avg_of_avgs)
                    # 不同状态的收益变异性越低越好
                    score += max(0.0, 30.0 - regime_cv * 30.0)
                else:
                    score += 10.0

        # 因子2: 近期 vs 历史表现对比
        if history and len(history) >= 10:
            recent = history[-5:]
            older = history[:-5]
            recent_avg = _mean([h.get("pnl", h.get("return", 0.0)) for h in recent])
            older_avg = _mean([h.get("pnl", h.get("return", 0.0)) for h in older])
            if older_avg > 0:
                ratio = recent_avg / older_avg
                # 如果近期表现不低于历史的 70%，说明适应性好
                if ratio >= 0.7:
                    score += 20.0
                elif ratio >= 0.3:
                    score += 10.0

        return max(0.0, min(100.0, score))

    def _check_degradation(self, strategy: str, current_score: float) -> List[Dict[str, Any]]:
        """检测分数退化"""
        alerts = []
        history = self._score_history.get(strategy, {}).get("all", [])
        if len(history) < 5:
            return alerts

        recent_scores = [r["composite"] for r in history[-5:]]
        peak = max(s[-1]["composite"] for s in [history[:-5]] if s) if len(history) > 5 else max(recent_scores)
        if peak == 0:
            peak = current_score

        drop = peak - current_score
        if drop >= self._degradation_threshold:
            alerts.append({
                "type": "score_degradation",
                "strategy": strategy,
                "severity": AlertSeverity.WARNING.value if drop < 2 * self._degradation_threshold else AlertSeverity.CRITICAL.value,
                "current_score": round(current_score, 2),
                "peak_score": round(peak, 2),
                "drop": round(drop, 2),
                "message": f"Score degraded by {drop:.1f} points from peak {peak:.1f}",
            })

        return alerts

    @staticmethod
    def _parse_window_seconds(window: str) -> int:
        mapping = {
            "1h": 3600, "6h": 21600, "12h": 43200, "24h": 86400,
            "3d": 259200, "7d": 604800, "14d": 1209600,
            "30d": 2592000, "60d": 5184000, "90d": 7776000,
        }
        return mapping.get(window, 2592000)


# =============================================================================
# Attribution Analyzer: 归因分析
# =============================================================================

class AttributionAnalyzer:
    """多维度归因分析器

    支持：
    - 因子归因：市场收益贡献、Alpha 贡献、选币效应、择时效应、成本归因
    - 市场状态归因：按市场状态分解 PnL
    - 时间归因：按小时/周几分解 PnL
    - 仓位规模归因：按仓位大小桶分解 PnL
    """

    def __init__(self, config: Dict[str, Any]):
        pf_config = config.get("performance_feedback", {})
        attr_config = pf_config.get("attribution", {})
        self._enable_factor = attr_config.get("factor_attribution", True)
        self._enable_regime = attr_config.get("regime_attribution", True)
        self._enable_time = attr_config.get("time_attribution", True)
        self._enable_size = attr_config.get("size_attribution", True)
        self._position_buckets = attr_config.get("position_buckets", [0.01, 0.05, 0.10, 0.25, 0.50])

    async def analyze(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[Dict[str, Any]],
        market_data: Optional[List[Dict[str, Any]]] = None,
        fees_data: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """执行完整的归因分析"""
        result: Dict[str, Any] = {}

        if self._enable_factor:
            result["factor_attribution"] = self._factor_attribution(trades, equity_curve, market_data or [], fees_data or [])

        if self._enable_regime:
            result["regime_attribution"] = self._regime_attribution(trades)

        if self._enable_time:
            result["time_attribution"] = self._time_attribution(trades)

        if self._enable_size:
            result["size_attribution"] = self._size_attribution(trades)

        result["total_pnl"] = round(sum(t.get("pnl", t.get("realized_pnl", 0.0)) for t in trades), 4)
        result["analysis_time"] = datetime.now().isoformat()

        return result

    def _factor_attribution(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[Dict[str, Any]],
        market_data: List[Dict[str, Any]],
        fees_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """因子归因分析"""
        total_pnl = sum(t.get("pnl", t.get("realized_pnl", 0.0)) for t in trades)

        # a) 市场收益贡献 (beta * market_return)
        market_contribution = 0.0
        if market_data and trades:
            trade_returns = [t.get("pnl_pct", t.get("return", 0.0)) for t in trades]
            market_returns = [m.get("return", 0.0) for m in market_data]
            if len(trade_returns) >= 3 and len(market_returns) >= 3:
                min_len = min(len(trade_returns), len(market_returns))
                beta = _covariance(trade_returns[:min_len], market_returns[:min_len]) / max(
                    _variance(market_returns[:min_len]), 1e-9
                )
                market_contribution = beta * sum(market_returns[:min_len])

        # b) Alpha 贡献（残差）
        alpha_contribution = total_pnl - market_contribution

        # c) 选币效应（每个标的的收益贡献）
        selection_effect: Dict[str, float] = {}
        for t in trades:
            symbol = t.get("symbol", t.get("instId", "unknown"))
            pnl = t.get("pnl", t.get("realized_pnl", 0.0))
            selection_effect[symbol] = selection_effect.get(symbol, 0.0) + pnl
        # 排序取 Top/Bottom
        sorted_selection = sorted(selection_effect.items(), key=lambda x: x[1], reverse=True)
        top_contributors = sorted_selection[:5]
        bottom_contributors = sorted_selection[-5:] if len(sorted_selection) >= 5 else []

        # d) 择时效应
        timing_trades = sorted(trades, key=lambda t: t.get("entry_time", t.get("timestamp", "")))
        timing_effect = 0.0
        if len(timing_trades) >= 4:
            # 将交易分成四等份，比较各份的 PnL
            quarter = max(1, len(timing_trades) // 4)
            quarter_pnls = []
            for q in range(4):
                start = q * quarter
                end = start + quarter if q < 3 else len(timing_trades)
                quarter_pnls.append(
                    sum(t.get("pnl", t.get("realized_pnl", 0.0)) for t in timing_trades[start:end])
                )
            avg_q = _mean(quarter_pnls)
            if avg_q > 0:
                timing_effect = _std(quarter_pnls) / avg_q  # 变异系数，越低择时越好

        # e) 成本归因
        total_fees = sum(t.get("fee", t.get("commission", 0.0)) for t in trades)
        total_slippage = sum(t.get("slippage", 0.0) for t in trades)
        total_funding = sum(t.get("funding", t.get("funding_fee", 0.0)) for t in trades)

        return {
            "total_pnl": round(total_pnl, 4),
            "market_contribution": round(market_contribution, 4),
            "market_contribution_pct": round(market_contribution / total_pnl * 100, 1) if total_pnl != 0 else 0.0,
            "alpha_contribution": round(alpha_contribution, 4),
            "alpha_contribution_pct": round(alpha_contribution / total_pnl * 100, 1) if total_pnl != 0 else 0.0,
            "selection_effect": {
                "top_contributors": [{"symbol": s, "pnl": round(p, 4)} for s, p in top_contributors],
                "bottom_contributors": [{"symbol": s, "pnl": round(p, 4)} for s, p in bottom_contributors],
                "concentration": round(
                    sum(abs(p) for _, p in top_contributors) / max(abs(total_pnl), 1e-9), 4
                ),
            },
            "timing_effect": round(timing_effect, 4),
            "cost_attribution": {
                "total_fees": round(total_fees, 4),
                "total_slippage": round(total_slippage, 4),
                "total_funding": round(total_funding, 4),
                "total_cost": round(total_fees + total_slippage + total_funding, 4),
                "cost_ratio": round(
                    (total_fees + total_slippage + total_funding) / max(abs(total_pnl), 1e-9), 4
                ),
            },
        }

    def _regime_attribution(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """市场状态归因：按 market regime 分解 PnL"""
        regime_pnl: Dict[str, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "count": 0, "win_count": 0})

        for t in trades:
            regime = t.get("regime", t.get("market_regime", "unknown"))
            pnl = t.get("pnl", t.get("realized_pnl", 0.0))
            entry = regime_pnl[regime]
            entry["pnl"] += pnl
            entry["count"] += 1
            if pnl > 0:
                entry["win_count"] += 1

        breakdown = {}
        total_pnl = sum(v["pnl"] for v in regime_pnl.values())
        for regime, data in sorted(regime_pnl.items(), key=lambda x: x[1]["pnl"], reverse=True):
            breakdown[regime] = {
                "pnl": round(data["pnl"], 4),
                "pnl_pct": round(data["pnl"] / total_pnl * 100, 1) if total_pnl != 0 else 0.0,
                "trade_count": data["count"],
                "win_rate": round(data["win_count"] / data["count"], 3) if data["count"] > 0 else 0.0,
            }

        return {
            "total_pnl": round(total_pnl, 4),
            "regime_breakdown": breakdown,
        }

    def _time_attribution(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """时间归因：按小时和星期几分解 PnL"""
        hour_pnl: Dict[int, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "count": 0})
        dow_pnl: Dict[int, Dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "count": 0})

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

        for t in trades:
            ts_str = t.get("entry_time", t.get("timestamp", t.get("time", "")))
            if not ts_str:
                continue
            try:
                if isinstance(ts_str, (int, float)):
                    ts = datetime.fromtimestamp(ts_str / 1000.0 if ts_str > 1e12 else ts_str)
                else:
                    ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00").split("+")[0].split("[")[0])
            except (ValueError, TypeError):
                continue

            pnl = t.get("pnl", t.get("realized_pnl", 0.0))
            hour = ts.hour
            dow = ts.weekday()

            hour_pnl[hour]["pnl"] += pnl
            hour_pnl[hour]["count"] += 1
            dow_pnl[dow]["pnl"] += pnl
            dow_pnl[dow]["count"] += 1

        hour_breakdown = {}
        for h in sorted(hour_pnl.keys()):
            data = hour_pnl[h]
            hour_breakdown[f"{h:02d}:00"] = {
                "pnl": round(data["pnl"], 4),
                "trade_count": data["count"],
            }

        dow_breakdown = {}
        for d in sorted(dow_pnl.keys()):
            data = dow_pnl[d]
            dow_breakdown[day_names[d]] = {
                "pnl": round(data["pnl"], 4),
                "trade_count": data["count"],
            }

        return {
            "hourly_breakdown": hour_breakdown,
            "day_of_week_breakdown": dow_breakdown,
        }

    def _size_attribution(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """仓位规模归因：按仓位大小桶分解 PnL"""
        buckets = self._position_buckets
        bucket_labels = []
        for i, threshold in enumerate(buckets):
            if i == 0:
                bucket_labels.append(f"<{threshold:.0%}")
            else:
                bucket_labels.append(f"{buckets[i - 1]:.0%}-{threshold:.0%}")
        bucket_labels.append(f">={buckets[-1]:.0%}")

        bucket_data: List[Dict[str, Any]] = [
            {"label": label, "pnl": 0.0, "count": 0, "win_count": 0}
            for label in bucket_labels
        ]

        for t in trades:
            position_pct = t.get("position_pct", t.get("position_size_pct", 0.0))
            if isinstance(position_pct, str):
                try:
                    position_pct = float(position_pct)
                except ValueError:
                    position_pct = 0.0
            pnl = t.get("pnl", t.get("realized_pnl", 0.0))

            bucket_idx = 0
            for threshold in buckets:
                if position_pct < threshold:
                    break
                bucket_idx += 1

            if bucket_idx >= len(bucket_data):
                bucket_idx = len(bucket_data) - 1

            bucket_data[bucket_idx]["pnl"] += pnl
            bucket_data[bucket_idx]["count"] += 1
            if pnl > 0:
                bucket_data[bucket_idx]["win_count"] += 1

        for b in bucket_data:
            b["pnl"] = round(b["pnl"], 4)
            b["win_rate"] = round(b["win_count"] / b["count"], 3) if b["count"] > 0 else 0.0

        return {"position_buckets": bucket_data}


# =============================================================================
# Feedback Controller: 闭环反馈
# =============================================================================

class FeedbackController:
    """闭环反馈控制器

    提供：
    - 性能差距分析
    - 调整建议及量化预期改善
    - 建议置信度（基于样本量）
    - 反馈应用跟踪
    - 学习率控制
    - 振荡检测
    """

    def __init__(self, config: Dict[str, Any]):
        pf_config = config.get("performance_feedback", {})
        fb_config = pf_config.get("feedback", {})
        self._learning_rate = fb_config.get("learning_rate", 0.1)
        self._min_samples_for_feedback = fb_config.get("min_samples", 20)
        self._max_adjustment_pct = fb_config.get("max_adjustment_pct", 0.30)
        self._oscillation_threshold = fb_config.get("oscillation_threshold", 3)
        self._feedback_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._applied_feedback: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def analyze_gap(
        self,
        strategy: str,
        metrics: PerformanceMetrics,
        expected_metrics: Optional[Dict[str, float]] = None,
        trade_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """性能差距分析"""
        async with self._lock:
            expected = expected_metrics or {}
            gaps = {}
            confidence = self._compute_confidence(trade_count or metrics.trade_count)

            # 各指标差距
            metric_pairs = [
                ("sharpe_ratio", "sharpe"),
                ("win_rate", "win_rate"),
                ("profit_factor", "profit_factor"),
                ("max_drawdown", "max_drawdown"),
                ("annualized_return", "annualized_return"),
            ]
            for attr_name, expected_key in metric_pairs:
                actual = getattr(metrics, attr_name, 0.0)
                target = expected.get(expected_key, actual * 1.1 if actual > 0 else 0.1)
                if target != 0:
                    gap_pct = (actual - target) / abs(target)
                else:
                    gap_pct = 0.0
                gaps[expected_key] = {
                    "actual": round(actual, 4),
                    "expected": round(target, 4),
                    "gap_pct": round(gap_pct * 100, 1),
                    "direction": "above" if gap_pct >= 0 else "below",
                }

            return {
                "strategy": strategy,
                "gaps": gaps,
                "confidence": round(confidence, 2),
                "sample_size": trade_count or metrics.trade_count,
                "timestamp": datetime.now().isoformat(),
            }

    async def generate_recommendations(
        self, strategy: str, gap_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """基于差距分析生成调整建议"""
        recommendations = []
        gaps = gap_analysis.get("gaps", {})
        confidence = gap_analysis.get("confidence", 0.5)

        for metric, data in gaps.items():
            gap_pct = data["gap_pct"]
            direction = data["direction"]

            if abs(gap_pct) < 2:
                continue

            rec = self._build_recommendation(strategy, metric, direction, gap_pct, confidence)
            if rec:
                recommendations.append(rec)

        # 按优先级排序
        recommendations.sort(key=lambda r: abs(r["expected_improvement_pct"]), reverse=True)
        return recommendations

    async def apply_feedback(
        self, strategy: str, recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """应用反馈建议"""
        async with self._lock:
            # 振荡检测
            oscillation_warning = self._detect_oscillation(strategy, recommendations)

            applied = []
            skipped = []

            for rec in recommendations:
                rec_id = rec.get("id", "")
                if oscillation_warning and oscillation_warning.get("risk") == "high":
                    skipped.append({**rec, "skip_reason": "oscillation_risk"})
                    continue

                applied_rec = {
                    **rec,
                    "applied_at": datetime.now().isoformat(),
                    "learning_rate_used": self._learning_rate,
                }
                applied.append(applied_rec)
                self._applied_feedback[strategy].append(applied_rec)

            # 记录反馈历史
            self._feedback_history[strategy].append({
                "timestamp": datetime.now().isoformat(),
                "recommendations": recommendations,
                "applied_count": len(applied),
                "skipped_count": len(skipped),
                "oscillation_warning": oscillation_warning,
            })

            return {
                "strategy": strategy,
                "applied": applied,
                "skipped": skipped,
                "oscillation_warning": oscillation_warning,
                "total_applied": len(applied),
                "total_skipped": len(skipped),
            }

    async def get_feedback_history(self, strategy: str) -> List[Dict[str, Any]]:
        """获取反馈历史"""
        return self._feedback_history.get(strategy, [])

    async def get_applied_feedback(self, strategy: str) -> List[Dict[str, Any]]:
        """获取已应用的反馈"""
        return self._applied_feedback.get(strategy, [])

    async def track_effect(
        self,
        strategy: str,
        feedback_id: str,
        before_metrics: PerformanceMetrics,
        after_metrics: PerformanceMetrics,
    ) -> Dict[str, Any]:
        """跟踪反馈应用效果"""
        effect = {
            "feedback_id": feedback_id,
            "strategy": strategy,
            "timestamp": datetime.now().isoformat(),
            "changes": {
                "sharpe_ratio": round(after_metrics.sharpe_ratio - before_metrics.sharpe_ratio, 4),
                "win_rate": round(after_metrics.win_rate - before_metrics.win_rate, 4),
                "profit_factor": round(after_metrics.profit_factor - before_metrics.profit_factor, 4),
                "max_drawdown": round(after_metrics.max_drawdown - before_metrics.max_drawdown, 4),
                "consistency_score": round(after_metrics.consistency_score - before_metrics.consistency_score, 4),
            },
        }

        # 判断效果
        improvements = sum(1 for v in effect["changes"].values() if v > 0)
        degradations = sum(1 for v in effect["changes"].values() if v < 0)
        if improvements > degradations:
            effect["net_effect"] = "positive"
        elif degradations > improvements:
            effect["net_effect"] = "negative"
        else:
            effect["net_effect"] = "neutral"

        # 更新对应 applied_feedback
        for entry in self._applied_feedback.get(strategy, []):
            if entry.get("id") == feedback_id:
                entry["effect"] = effect
                break

        return effect

    def _compute_confidence(self, sample_size: int) -> float:
        """基于样本量计算置信度 (0-1)"""
        if sample_size < 5:
            return 0.1
        if sample_size >= 500:
            return 0.95
        # 对数映射：20 → 0.4, 50 → 0.6, 100 → 0.75, 200 → 0.85
        return min(0.95, math.log(sample_size) / math.log(500) * 0.85 + 0.1)

    def _build_recommendation(
        self, strategy: str, metric: str, direction: str, gap_pct: float, confidence: float
    ) -> Optional[Dict[str, Any]]:
        """构建单条调整建议"""
        adjustment_pct = min(abs(gap_pct) / 100 * self._learning_rate, self._max_adjustment_pct)
        expected_improvement = abs(gap_pct) * confidence * self._learning_rate

        rec_id = f"{strategy}_{metric}_{int(datetime.now().timestamp())}"

        suggestion_map = {
            "sharpe": "Consider adjusting position sizing or adding filters to improve risk-adjusted returns",
            "win_rate": "Review entry criteria; tighten signal thresholds to improve win probability",
            "profit_factor": "Optimize risk-reward by adjusting take-profit and stop-loss levels",
            "max_drawdown": "Reduce position size or add hedging to limit drawdown exposure",
            "annualized_return": "Explore higher-conviction setups or increase position on high-probability trades",
        }

        return {
            "id": rec_id,
            "strategy": strategy,
            "metric": metric,
            "direction": direction,
            "gap_pct": round(gap_pct, 1),
            "adjustment_pct": round(adjustment_pct * 100, 1),
            "expected_improvement_pct": round(expected_improvement, 1),
            "confidence": round(confidence, 2),
            "suggestion": suggestion_map.get(metric, "Adjust related strategy parameter to close gap"),
        }

    def _detect_oscillation(
        self, strategy: str, recommendations: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """检测反馈振荡（yo-yo 效应）"""
        history = self._applied_feedback.get(strategy, [])
        if len(history) < self._oscillation_threshold * 2:
            return None

        # 检查最近的建议是否有方向反转
        recent = history[-self._oscillation_threshold * 2:]

        # 简化检测：查最近 N 条是否有交替方向
        # 检查 applied feedback 的方向模式
        metric_directions: Dict[str, List[str]] = defaultdict(list)
        for rec in recommendations:
            metric_directions[rec["metric"]].append(rec["direction"])

        oscillation_count = 0
        for metric, dirs in metric_directions.items():
            # 查看历史中该 metric 的最近方向
            hist_dirs = []
            for entry in history:
                for rec in entry.get("applied", entry.get("recommendations", [])):
                    if rec.get("metric") == metric:
                        hist_dirs.append(rec.get("direction", ""))
            hist_dirs = hist_dirs[-self._oscillation_threshold:]
            if len(hist_dirs) >= self._oscillation_threshold:
                changes = sum(1 for i in range(1, len(hist_dirs)) if hist_dirs[i] != hist_dirs[i - 1])
                if changes >= self._oscillation_threshold - 1:
                    oscillation_count += 1

        if oscillation_count > 0:
            risk = "high" if oscillation_count >= 2 else "medium"
            return {
                "detected": True,
                "oscillation_count": oscillation_count,
                "risk": risk,
                "message": f"Detected {oscillation_count} oscillating feedback patterns; consider reducing learning rate",
            }

        return None

    def set_learning_rate(self, rate: float):
        self._learning_rate = max(0.01, min(rate, 0.5))
        logger.info(f"Feedback learning rate set to {self._learning_rate}")


# =============================================================================
# Strategy Comparator: 策略比较
# =============================================================================

class StrategyComparator:
    """策略比较器

    支持：
    - 同类策略对比
    - 基准对比（市场指数、无风险利率）
    - 滚动超额收益跟踪
    - 排名百分位计算
    - Bootstrap 统计检验
    """

    def __init__(self, config: Dict[str, Any]):
        pf_config = config.get("performance_feedback", {})
        cmp_config = pf_config.get("comparison", {})
        self._bootstrap_samples = cmp_config.get("bootstrap_samples", 1000)
        self._bootstrap_confidence = cmp_config.get("bootstrap_confidence", 0.95)
        self._strategy_metrics: Dict[str, PerformanceMetrics] = {}
        self._rolling_outperformance: Dict[str, List[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    def register_strategy_metrics(self, strategy: str, metrics: PerformanceMetrics):
        """注册策略指标用于比较"""
        self._strategy_metrics[strategy] = metrics

    async def compare_with_peers(
        self, strategy: str, peer_strategies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """同类策略对比"""
        async with self._lock:
            if strategy not in self._strategy_metrics:
                return {"error": f"Strategy '{strategy}' not registered"}

            target = self._strategy_metrics[strategy]
            peers = peer_strategies or [s for s in self._strategy_metrics if s != strategy]

            if not peers:
                return {"strategy": strategy, "rank": 1, "total": 1, "message": "No peers to compare"}

            # 计算各策略的综合排名
            scores = self._compute_rank_scores()
            sorted_strategies = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            ranks = {s: i + 1 for i, (s, _) in enumerate(sorted_strategies)}

            rank = ranks.get(strategy, len(sorted_strategies))
            percentile = (1.0 - (rank - 1) / max(len(sorted_strategies) - 1, 1)) * 100

            comparisons = {}
            for peer in peers:
                if peer in self._strategy_metrics:
                    pm = self._strategy_metrics[peer]
                    comparisons[peer] = {
                        "sharpe_ratio": round(target.sharpe_ratio - pm.sharpe_ratio, 4),
                        "win_rate": round(target.win_rate - pm.win_rate, 4),
                        "profit_factor": round(target.profit_factor - pm.profit_factor, 4),
                        "max_drawdown": round(target.max_drawdown - pm.max_drawdown, 4),
                    }

            # Bootstrap 显著性检验
            bootstrap_result = self._bootstrap_test(strategy, [p for p in peers if p in self._strategy_metrics])

            return {
                "strategy": strategy,
                "rank": rank,
                "total_strategies": len(sorted_strategies),
                "percentile": round(percentile, 1),
                "peer_comparisons": comparisons,
                "bootstrap_test": bootstrap_result,
            }

    async def compare_with_benchmark(
        self, strategy: str, benchmark_return: float, risk_free_rate: float = 0.03
    ) -> Dict[str, Any]:
        """与基准对比"""
        metrics = self._strategy_metrics.get(strategy)
        if not metrics:
            return {"error": f"Strategy '{strategy}' not registered"}

        excess_return = metrics.annualized_return - benchmark_return
        excess_sharpe = metrics.sharpe_ratio - (benchmark_return - risk_free_rate) / max(metrics.daily_volatility, 1e-9)
        excess_sortino = metrics.sortino_ratio - (benchmark_return - risk_free_rate) / max(metrics.downside_deviation, 1e-9)

        info_ratio = excess_return / max(metrics.daily_volatility, 1e-9) if metrics.daily_volatility > 0 else 0.0

        return {
            "strategy": strategy,
            "benchmark_return": round(benchmark_return, 4),
            "strategy_return": round(metrics.annualized_return, 4),
            "excess_return": round(excess_return, 4),
            "excess_sharpe": round(excess_sharpe, 4),
            "excess_sortino": round(excess_sortino, 4),
            "information_ratio": round(info_ratio, 4),
            "risk_free_rate": risk_free_rate,
        }

    async def get_rolling_outperformance(
        self, strategy: str, benchmark_returns: List[float], window: int = 30
    ) -> Dict[str, Any]:
        """滚动超额收益跟踪"""
        metrics = self._strategy_metrics.get(strategy)
        if not metrics:
            return {"error": f"Strategy '{strategy}' not registered"}

        # 使用已有历史数据生成表现
        hist = self._rolling_outperformance.get(strategy, [])
        if not hist:
            return {"strategy": strategy, "rolling_outperformance": [], "window": window}

        rolling = []
        for i in range(len(hist) - window + 1):
            window_hist = hist[i:i + window]
            avg_outperformance = _mean(window_hist)
            rolling.append(round(avg_outperformance, 6))

        return {
            "strategy": strategy,
            "window": window,
            "rolling_outperformance": rolling,
            "current_outperformance": rolling[-1] if rolling else 0.0,
            "max_outperformance": max(rolling) if rolling else 0.0,
            "min_outperformance": min(rolling) if rolling else 0.0,
        }

    def track_outperformance(self, strategy: str, strategy_return: float, benchmark_return: float):
        """记录单期超额收益"""
        self._rolling_outperformance[strategy].append(strategy_return - benchmark_return)

    def _compute_rank_scores(self) -> Dict[str, float]:
        """计算各策略综合排名分"""
        scores = {}
        for name, m in self._strategy_metrics.items():
            score = (
                m.sharpe_ratio * 0.30
                + m.sortino_ratio * 0.15
                + min(m.calmar_ratio, 10.0) * 0.15
                + m.win_rate * 0.15
                + m.profit_factor * 0.10
                + m.consistency_score * 0.10
                + m.efficiency_score * 0.05
            )
            # 回撤惩罚
            score -= m.max_drawdown * 0.5
            scores[name] = score
        return scores

    def _bootstrap_test(
        self, strategy: str, peers: List[str]
    ) -> Dict[str, Any]:
        """Bootstrap 假设检验：策略 A 是否显著优于同行"""
        target = self._strategy_metrics.get(strategy)
        if not target:
            return {"error": "Target strategy not found"}

        # 模拟：从历史交易中重采样
        # 由于我们没有原始交易数据，使用指标构建伪分布
        n_bootstrap = self._bootstrap_samples

        # 生成目标策略的伪收益分布
        target_mean = target.annualized_return
        target_std = target.daily_volatility
        target_sample = [random.gauss(target_mean, target_std) for _ in range(n_bootstrap)]

        peer_means = []
        for peer_name in peers:
            pm = self._strategy_metrics.get(peer_name)
            if pm:
                peer_means.append(pm.annualized_return)

        if not peer_means:
            return {"error": "No peer data for bootstrap"}

        peer_avg = _mean(peer_means)
        peer_std = _std(peer_means) if len(peer_means) >= 2 else abs(peer_avg) * 0.5 if peer_avg != 0 else 0.1

        # Bootstrap: 重采样目标策略 vs 同行
        better_count = 0
        for _ in range(n_bootstrap):
            target_draw = _mean(random.choices(target_sample, k=min(100, len(target_sample))))
            peer_draw = random.gauss(peer_avg, max(peer_std, 1e-9))
            if target_draw > peer_draw:
                better_count += 1

        p_value = 1.0 - (better_count / n_bootstrap)
        is_significant = p_value < (1.0 - self._bootstrap_confidence)

        return {
            "bootstrap_samples": n_bootstrap,
            "confidence_level": self._bootstrap_confidence,
            "p_value": round(p_value, 4),
            "is_significant": is_significant,
            "target_better_pct": round(better_count / n_bootstrap * 100, 1),
            "message": f"Strategy {'significantly' if is_significant else 'not significantly'} outperforms peers at {self._bootstrap_confidence:.0%} confidence",
        }


# =============================================================================
# Performance Alerter: 告警系统
# =============================================================================

class PerformanceAlerter:
    """性能告警系统

    告警条件：
    - 回撤超过阈值
    - Sharpe 比率低于最低值
    - 胜率退化
    - 连续亏损超过预期
    - 评分低于关键水平
    """

    def __init__(self, config: Dict[str, Any]):
        pf_config = config.get("performance_feedback", {})
        alert_config = pf_config.get("alerts", {})
        self._thresholds = {
            "max_drawdown": alert_config.get("max_drawdown_threshold", 0.15),
            "min_sharpe": alert_config.get("min_sharpe_threshold", 0.5),
            "min_win_rate": alert_config.get("min_win_rate_threshold", 0.35),
            "max_consecutive_losses": alert_config.get("max_consecutive_losses", 5),
            "critical_score": alert_config.get("critical_score_threshold", 40.0),
            "warning_score": alert_config.get("warning_score_threshold", 60.0),
        }
        self._cooldown_seconds = alert_config.get("cooldown_seconds", 3600)
        self._alert_history: List[Dict[str, Any]] = []
        self._last_alert_times: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def check(
        self,
        strategy: str,
        metrics: PerformanceMetrics,
        score: Optional[Dict[str, Any]] = None,
        recent_trades: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """执行所有告警检查"""
        async with self._lock:
            alerts = []

            # 1. 回撤告警
            alert = self._check_drawdown(strategy, metrics)
            if alert:
                alerts.append(alert)

            # 2. Sharpe 比率告警
            alert = self._check_sharpe(strategy, metrics)
            if alert:
                alerts.append(alert)

            # 3. 胜率告警
            alert = self._check_win_rate(strategy, metrics)
            if alert:
                alerts.append(alert)

            # 4. 连续亏损告警
            alert = self._check_consecutive_losses(strategy, recent_trades or [])
            if alert:
                alerts.append(alert)

            # 5. 评分告警
            if score:
                alert = self._check_score(strategy, score)
                if alert:
                    alerts.append(alert)

            # 应用冷却时间过滤
            filtered_alerts = []
            now = datetime.now()
            for alert in alerts:
                alert_key = f"{strategy}_{alert['type']}_{alert['severity']}"
                last_time = self._last_alert_times.get(alert_key)
                if last_time and (now - last_time).total_seconds() < self._cooldown_seconds:
                    continue
                self._last_alert_times[alert_key] = now
                alert["timestamp"] = now.isoformat()
                filtered_alerts.append(alert)

            self._alert_history.extend(filtered_alerts)
            return filtered_alerts

    async def get_history(
        self, strategy: Optional[str] = None, severity: Optional[str] = None, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """获取告警历史"""
        results = self._alert_history
        if strategy:
            results = [a for a in results if a.get("strategy") == strategy]
        if severity:
            results = [a for a in results if a.get("severity") == severity]
        return results[-limit:]

    def _check_drawdown(self, strategy: str, metrics: PerformanceMetrics) -> Optional[Dict[str, Any]]:
        threshold = self._thresholds["max_drawdown"]
        if metrics.max_drawdown >= threshold:
            severity = AlertSeverity.CRITICAL if metrics.max_drawdown >= threshold * 1.5 else AlertSeverity.WARNING
            return {
                "type": "drawdown",
                "strategy": strategy,
                "severity": severity.value,
                "value": round(metrics.max_drawdown, 4),
                "threshold": threshold,
                "message": f"Max drawdown {metrics.max_drawdown:.1%} exceeds threshold {threshold:.1%}",
            }
        return None

    def _check_sharpe(self, strategy: str, metrics: PerformanceMetrics) -> Optional[Dict[str, Any]]:
        threshold = self._thresholds["min_sharpe"]
        if metrics.sharpe_ratio <= threshold and metrics.trade_count >= 10:
            severity = AlertSeverity.CRITICAL if metrics.sharpe_ratio <= 0 else AlertSeverity.WARNING
            return {
                "type": "sharpe_ratio",
                "strategy": strategy,
                "severity": severity.value,
                "value": round(metrics.sharpe_ratio, 4),
                "threshold": threshold,
                "message": f"Sharpe ratio {metrics.sharpe_ratio:.2f} below minimum {threshold}",
            }
        return None

    def _check_win_rate(self, strategy: str, metrics: PerformanceMetrics) -> Optional[Dict[str, Any]]:
        threshold = self._thresholds["min_win_rate"]
        if metrics.win_rate <= threshold and metrics.trade_count >= 10:
            return {
                "type": "win_rate",
                "strategy": strategy,
                "severity": AlertSeverity.WARNING.value,
                "value": round(metrics.win_rate, 4),
                "threshold": threshold,
                "message": f"Win rate {metrics.win_rate:.1%} below threshold {threshold:.1%}",
            }
        return None

    def _check_consecutive_losses(
        self, strategy: str, recent_trades: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        threshold = self._thresholds["max_consecutive_losses"]
        consecutive = 0
        for t in reversed(recent_trades):
            pnl = t.get("pnl", t.get("realized_pnl", 0.0))
            if pnl < 0:
                consecutive += 1
            else:
                break

        if consecutive >= threshold:
            severity = AlertSeverity.CRITICAL if consecutive >= threshold * 1.5 else AlertSeverity.WARNING
            return {
                "type": "consecutive_losses",
                "strategy": strategy,
                "severity": severity.value,
                "value": consecutive,
                "threshold": threshold,
                "message": f"{consecutive} consecutive losses (threshold: {threshold})",
            }
        return None

    def _check_score(
        self, strategy: str, score: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        composite = score.get("composite_score", 100.0)
        critical = self._thresholds["critical_score"]
        warning = self._thresholds["warning_score"]

        if composite <= critical:
            return {
                "type": "score_drop",
                "strategy": strategy,
                "severity": AlertSeverity.CRITICAL.value,
                "value": round(composite, 2),
                "threshold": critical,
                "message": f"Composite score {composite:.1f} below critical level {critical}",
            }
        elif composite <= warning:
            return {
                "type": "score_drop",
                "strategy": strategy,
                "severity": AlertSeverity.WARNING.value,
                "value": round(composite, 2),
                "threshold": warning,
                "message": f"Composite score {composite:.1f} below warning level {warning}",
            }
        return None


# =============================================================================
# Main Class: PerformanceFeedback
# =============================================================================

class PerformanceFeedback(EnterpriseServiceMixin):
    """性能反馈系统主类

    整合评分、归因、反馈、比较、告警五大模块，提供统一的性能反馈接口。
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        pf_config = config.get("performance_feedback", {})

        self._scorer = PerformanceScorer(config)
        self._attribution = AttributionAnalyzer(config)
        self._feedback = FeedbackController(config)
        self._comparator = StrategyComparator(config)
        self._alerter = PerformanceAlerter(config)

        self._strategy_metrics: Dict[str, PerformanceMetrics] = {}
        self._strategy_trades: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._strategy_equity: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._update_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._max_history = pf_config.get("max_history", 500)

        self._benchmark_returns: Dict[str, float] = pf_config.get("benchmark_returns", {})
        self._risk_free_rate = pf_config.get("risk_free_rate", 0.03)

        self._lock = asyncio.Lock()
        self._running = False

        # 注册基准
        for benchmark, ret in self._benchmark_returns.items():
            self._scorer.set_benchmark_return(benchmark, ret)

    async def start(self):
        """启动性能反馈系统"""
        self._running = True
        logger.info("PerformanceFeedback system started")

    async def stop(self):
        """停止性能反馈系统"""
        self._running = False
        logger.info("PerformanceFeedback system stopped")

    async def update(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[Dict[str, Any]],
        strategy: str,
    ) -> Dict[str, Any]:
        """更新策略数据并返回完整的性能分析结果"""
        async with self._lock:
            # 存储数据
            self._strategy_trades[strategy].extend(trades)
            if len(self._strategy_trades[strategy]) > self._max_history:
                self._strategy_trades[strategy] = self._strategy_trades[strategy][-self._max_history:]

            self._strategy_equity[strategy] = equity_curve

            # 计算指标
            metrics = compute_metrics(
                self._strategy_trades[strategy],
                equity_curve,
            )
            self._strategy_metrics[strategy] = metrics
            self._comparator.register_strategy_metrics(strategy, metrics)

            # 多维度评分
            all_trades = self._strategy_trades[strategy]
            score_result = await self._scorer.score(metrics, strategy, performance_history=all_trades)

            # 归因分析
            attribution = await self._attribution.analyze(
                self._strategy_trades[strategy],
                equity_curve,
            )

            # 差距分析与反馈
            gap_analysis = await self._feedback.analyze_gap(strategy, metrics)
            recommendations = await self._feedback.generate_recommendations(strategy, gap_analysis)

            # 告警检查
            recent_trades = all_trades[-20:] if all_trades else []
            alerts = await self._alerter.check(strategy, metrics, score_result, recent_trades)

            # 记录更新历史
            update_record = {
                "timestamp": datetime.now().isoformat(),
                "strategy": strategy,
                "metrics": metrics.to_dict(),
                "composite_score": score_result["composite_score"],
                "alert_count": len(alerts),
            }
            self._update_history[strategy].append(update_record)
            if len(self._update_history[strategy]) > self._max_history:
                self._update_history[strategy] = self._update_history[strategy][-self._max_history:]

            # 跟踪超额收益
            for b_name, b_ret in self._benchmark_returns.items():
                self._comparator.track_outperformance(strategy, metrics.annualized_return, b_ret)

            return {
                "strategy": strategy,
                "metrics": metrics.to_dict(),
                "score": score_result,
                "attribution": attribution,
                "gap_analysis": gap_analysis,
                "recommendations": recommendations,
                "alerts": alerts,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_score(self, strategy: str, window: str = "30d") -> Dict[str, Any]:
        """获取策略评分"""
        if strategy not in self._strategy_metrics:
            return {"error": f"Strategy '{strategy}' not found"}

        return await self._scorer.get_rolling_scores(strategy, window)

    async def get_attribution(self, strategy: str) -> Dict[str, Any]:
        """获取策略归因分析"""
        if strategy not in self._strategy_trades:
            return {"error": f"Strategy '{strategy}' not found"}

        return await self._attribution.analyze(
            self._strategy_trades[strategy],
            self._strategy_equity.get(strategy, []),
        )

    async def get_feedback(self, strategy: str) -> Dict[str, Any]:
        """获取策略反馈"""
        if strategy not in self._strategy_metrics:
            return {"error": f"Strategy '{strategy}' not found"}

        metrics = self._strategy_metrics[strategy]
        gap = await self._feedback.analyze_gap(strategy, metrics)
        recommendations = await self._feedback.generate_recommendations(strategy, gap)
        history = await self._feedback.get_feedback_history(strategy)

        return {
            "strategy": strategy,
            "gap_analysis": gap,
            "recommendations": recommendations,
            "history": history[-10:],
        }

    async def apply_feedback(
        self, strategy: str, recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """应用反馈建议"""
        if not recommendations:
            return {"strategy": strategy, "applied": [], "message": "No recommendations to apply"}

        # 记录应用前的指标（深拷贝，避免与 after 引用同一对象导致效果跟踪失效）
        before_metrics = copy.deepcopy(self._strategy_metrics.get(strategy))

        result = await self._feedback.apply_feedback(strategy, recommendations)

        # 如果应用后又有新数据，可以跟踪效果
        if before_metrics and strategy in self._strategy_metrics:
            after_metrics = copy.deepcopy(self._strategy_metrics[strategy])
            for applied in result.get("applied", []):
                await self._feedback.track_effect(
                    strategy, applied.get("id", ""), before_metrics, after_metrics
                )

        logger.info(f"Applied {result['total_applied']} feedback items for {strategy}")
        return result

    async def compare_strategies(self, strategies: List[str]) -> Dict[str, Any]:
        """比较多个策略"""
        if len(strategies) < 2:
            return {"error": "Need at least 2 strategies to compare"}

        results = {}
        for strategy in strategies:
            if strategy not in self._strategy_metrics:
                results[strategy] = {"error": f"Strategy '{strategy}' not found"}
                continue
            peer_result = await self._comparator.compare_with_peers(strategy)
            results[strategy] = peer_result

        # 基准对比
        benchmark_comparisons = {}
        for strategy in strategies:
            if strategy in self._strategy_metrics:
                for b_name, b_ret in self._benchmark_returns.items():
                    bc = await self._comparator.compare_with_benchmark(
                        strategy, b_ret, self._risk_free_rate
                    )
                    benchmark_comparisons[f"{strategy}_vs_{b_name}"] = bc

        return {
            "peer_comparisons": results,
            "benchmark_comparisons": benchmark_comparisons,
            "timestamp": datetime.now().isoformat(),
        }

    async def get_alerts(self) -> List[Dict[str, Any]]:
        """获取当前告警"""
        return await self._alerter.get_history()

    def get_summary(self) -> Dict[str, Any]:
        """获取系统摘要"""
        strategies = list(self._strategy_metrics.keys())
        strategy_summaries = {}
        for s in strategies:
            m = self._strategy_metrics[s]
            strategy_summaries[s] = {
                "total_return": round(m.total_return, 4),
                "sharpe_ratio": round(m.sharpe_ratio, 4),
                "max_drawdown": round(m.max_drawdown, 4),
                "win_rate": round(m.win_rate, 4),
                "trade_count": m.trade_count,
                "consistency_score": round(m.consistency_score, 4),
                "efficiency_score": round(m.efficiency_score, 4),
            }

        return {
            "system": "PerformanceFeedback",
            "running": self._running,
            "strategy_count": len(strategies),
            "strategies": strategy_summaries,
            "alert_count": len(self._alerter._alert_history),
            "feedback_applied": sum(
                len(v) for v in self._feedback._applied_feedback.values()
            ),
            "timestamp": datetime.now().isoformat(),
        }

    @property
    def scorer(self) -> PerformanceScorer:
        return self._scorer

    @property
    def attribution(self) -> AttributionAnalyzer:
        return self._attribution

    @property
    def feedback(self) -> FeedbackController:
        return self._feedback

    @property
    def comparator(self) -> StrategyComparator:
        return self._comparator

    @property
    def alerter(self) -> PerformanceAlerter:
        return self._alerter