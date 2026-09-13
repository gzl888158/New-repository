"""
策略组合优化引擎 (PortfolioOptimizer)
========================================
统一的策略组合管理、优化与风险分析中央模块。

核心能力：
1. MPT 现代组合理论优化 — 有效前沿、最大夏普、最小方差、风险平价
2. 策略相关性矩阵 — 实时滚动相关性追踪，检测策略同质化
3. 绩效归因分析 — PnL贡献、风险贡献、回撤归因
4. 策略组合模板 — 不同市场状态的预设策略组合
5. VaR/CVaR 风险度量 — 历史模拟法、参数法、蒙特卡洛法
6. 资金效率最大化 — 跨策略保证金共享优化
7. 组合压力测试 — 极端行情场景模拟

集成方式：
    from core.portfolio_optimizer import PortfolioOptimizer, get_portfolio_optimizer
    opt = get_portfolio_optimizer(config)
    result = opt.optimize(performance_data)  # 返回最优权重
"""

import os
import json
import math
import time
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

import numpy as np
from loguru import logger

# 可选依赖：scipy 用于高效优化
try:
    from scipy.optimize import minimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available, using numpy fallback for optimization")


# ============================================================
# 枚举与数据模型
# ============================================================

class OptimizationObjective(Enum):
    """优化目标"""
    MAX_SHARPE = "max_sharpe"             # 最大夏普比率
    MIN_VARIANCE = "min_variance"         # 最小方差
    RISK_PARITY = "risk_parity"           # 风险平价（各策略风险贡献相等）
    MAX_DIVERSIFICATION = "max_div"       # 最大化分散度
    EQUAL_WEIGHT = "equal_weight"         # 等权重
    MAX_SORTINO = "max_sortino"           # 最大索提诺比率
    TARGET_RETURN = "target_return"       # 目标收益下最小风险
    TARGET_RISK = "target_risk"           # 目标风险下最大收益


class ComboMarketRegime(Enum):
    """组合模板适用的市场状态"""
    TRENDING_UP = "trending_up"           # 单边上涨
    TRENDING_DOWN = "trending_down"       # 单边下跌
    RANGING = "ranging"                   # 震荡
    HIGH_VOLATILITY = "high_volatility"   # 高波动
    LOW_VOLATILITY = "low_volatility"     # 低波动
    RECOVERY = "recovery"                 # 复苏反弹
    ALL = "all"                           # 通用


class RiskMetric(Enum):
    """风险度量类型"""
    VAR_95 = "var_95"                     # 95% VaR
    VAR_99 = "var_99"                     # 99% VaR
    CVAR_95 = "cvar_95"                   # 95% CVaR
    CVAR_99 = "cvar_99"                   # 99% CVaR
    MAX_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    DOWNSIDE_DEV = "downside_deviation"


@dataclass
class StrategyPerformance:
    """单个策略绩效数据"""
    name: str
    returns: List[float] = field(default_factory=list)          # 日收益率序列
    cumulative_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_holding_minutes: float = 0.0
    calmar_ratio: float = 0.0
    current_weight: float = 0.0
    pnl_contribution: float = 0.0
    category: str = "contract"           # contract / spot / hybrid
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "annualized_return": round(self.annualized_return, 6),
            "annualized_volatility": round(self.annualized_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_trades": self.total_trades,
            "calmar_ratio": round(self.calmar_ratio, 4),
            "current_weight": round(self.current_weight, 4),
            "pnl_contribution": round(self.pnl_contribution, 4),
            "category": self.category,
            "enabled": self.enabled,
        }


@dataclass
class CorrelationMatrix:
    """策略相关性矩阵"""
    strategies: List[str] = field(default_factory=list)
    matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    window_days: int = 30
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    avg_correlation: float = 0.0            # 平均相关性
    high_corr_pairs: List[Dict[str, Any]] = field(default_factory=list)  # 高相关策略对

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "strategies": self.strategies,
            "window_days": self.window_days,
            "last_updated": self.last_updated,
            "avg_correlation": round(self.avg_correlation, 4),
            "high_corr_pairs": self.high_corr_pairs,
        }
        if self.matrix.size > 0:
            result["matrix"] = [[round(float(v), 4) for v in row] for row in self.matrix]
        return result


@dataclass
class VaRResult:
    """VaR 计算结果"""
    method: str                                # historical / parametric / monte_carlo
    confidence_level: float                    # 置信水平
    horizon_days: int                          # 持有期天数
    var_value: float                           # VaR 值
    cvar_value: float = 0.0                    # CVaR 值
    var_pct: float = 0.0                       # VaR 占组合价值的百分比
    cvar_pct: float = 0.0                      # CVaR 占组合价值的百分比
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "confidence_level": self.confidence_level,
            "horizon_days": self.horizon_days,
            "var_value": round(self.var_value, 2),
            "cvar_value": round(self.cvar_value, 2),
            "var_pct": round(self.var_pct, 4),
            "cvar_pct": round(self.cvar_pct, 4),
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class PerformanceAttribution:
    """绩效归因结果"""
    strategy_contributions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    top_contributor: str = ""
    top_contributor_pct: float = 0.0
    worst_contributor: str = ""
    worst_contributor_pct: float = 0.0
    concentration_ratio: float = 0.0        # 前2名策略贡献占比（>0.5 意味着过度集中）
    diversification_score: float = 0.0      # 分散度评分（0-1，越高越分散）
    risk_contributions: Dict[str, float] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_contributions": {
                k: {kk: round(vv, 6) for kk, vv in v.items()}
                for k, v in self.strategy_contributions.items()
            },
            "top_contributor": self.top_contributor,
            "top_contributor_pct": round(self.top_contributor_pct, 4),
            "worst_contributor": self.worst_contributor,
            "worst_contributor_pct": round(self.worst_contributor_pct, 4),
            "concentration_ratio": round(self.concentration_ratio, 4),
            "diversification_score": round(self.diversification_score, 4),
            "risk_contributions": {k: round(v, 6) for k, v in self.risk_contributions.items()},
            "timestamp": self.timestamp,
        }


@dataclass
class StrategyComboTemplate:
    """策略组合模板"""
    id: str
    name: str
    description: str
    market_regime: ComboMarketRegime
    strategy_weights: Dict[str, float]          # 策略名称 -> 目标权重
    priority_order: List[str] = field(default_factory=list)  # 优先级顺序
    max_leverage: float = 3.0
    risk_profile: str = "moderate"               # conservative / moderate / aggressive
    min_capital: float = 500.0
    rebalance_frequency_hours: int = 4
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "market_regime": self.market_regime.value,
            "strategy_weights": self.strategy_weights,
            "priority_order": self.priority_order,
            "max_leverage": self.max_leverage,
            "risk_profile": self.risk_profile,
            "min_capital": self.min_capital,
            "rebalance_frequency_hours": self.rebalance_frequency_hours,
            "tags": self.tags,
        }


@dataclass
class PortfolioOptimizationResult:
    """组合优化结果"""
    objective: str
    optimal_weights: Dict[str, float]
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    expected_sharpe: float = 0.0
    diversification_ratio: float = 0.0
    weight_changes: Dict[str, float] = field(default_factory=dict)  # 与当前权重的差异
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)
    efficient_frontier: List[Dict[str, float]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "objective": self.objective,
            "optimal_weights": {k: round(v, 6) for k, v in self.optimal_weights.items()},
            "expected_return": round(self.expected_return, 6),
            "expected_volatility": round(self.expected_volatility, 6),
            "expected_sharpe": round(self.expected_sharpe, 4),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "weight_changes": {k: round(v, 6) for k, v in self.weight_changes.items()},
            "optimization_metadata": self.optimization_metadata,
            "efficient_frontier": self.efficient_frontier[:50],
            "timestamp": self.timestamp,
        }


# ============================================================
# 预设策略组合模板
# ============================================================

PRESET_COMBOS: Dict[str, StrategyComboTemplate] = {
    "trend_focused": StrategyComboTemplate(
        id="trend_focused",
        name="趋势主导",
        description="强势单边行情中聚焦趋势策略，适合 BTC/ETH 大行情突破",
        market_regime=ComboMarketRegime.TRENDING_UP,
        strategy_weights={"trend": 0.45, "scalping": 0.25, "grid": 0.15, "arbitrage": 0.15},
        priority_order=["trend", "scalping", "arbitrage", "grid"],
        max_leverage=3.0,
        risk_profile="aggressive",
        min_capital=2000.0,
        rebalance_frequency_hours=2,
        tags=["trending", "breakout", "momentum"],
    ),
    "mean_reversion": StrategyComboTemplate(
        id="mean_reversion",
        name="均值回归",
        description="震荡行情中高抛低吸，网格策略占主导",
        market_regime=ComboMarketRegime.RANGING,
        strategy_weights={"grid": 0.40, "scalping": 0.30, "trend": 0.10, "arbitrage": 0.20},
        priority_order=["grid", "scalping", "arbitrage", "trend"],
        max_leverage=3.0,
        risk_profile="moderate",
        min_capital=500.0,
        rebalance_frequency_hours=4,
        tags=["ranging", "oscillation", "mean_reversion"],
    ),
    "high_vol_survival": StrategyComboTemplate(
        id="high_vol_survival",
        name="高波生存",
        description="极端波动中降低敞口，利用波动率套利+对冲",
        market_regime=ComboMarketRegime.HIGH_VOLATILITY,
        strategy_weights={"arbitrage": 0.35, "grid": 0.30, "scalping": 0.20, "trend": 0.15},
        priority_order=["arbitrage", "grid", "scalping", "trend"],
        max_leverage=1.5,
        risk_profile="conservative",
        min_capital=1000.0,
        rebalance_frequency_hours=1,
        tags=["volatility", "survival", "hedging"],
    ),
    "aggressive_growth": StrategyComboTemplate(
        id="aggressive_growth",
        name="激进增长",
        description="牛市氛围中最大化资金利用率，高杠杆趋势+抢单",
        market_regime=ComboMarketRegime.TRENDING_UP,
        strategy_weights={"scalping": 0.40, "trend": 0.35, "grid": 0.10, "arbitrage": 0.15},
        priority_order=["scalping", "trend", "arbitrage", "grid"],
        max_leverage=5.0,
        risk_profile="aggressive",
        min_capital=2000.0,
        rebalance_frequency_hours=2,
        tags=["bull", "growth", "aggressive"],
    ),
    "capital_preservation": StrategyComboTemplate(
        id="capital_preservation",
        name="资金保全",
        description="熊市/回调中降低风险敞口，优先套利和轻仓网格",
        market_regime=ComboMarketRegime.TRENDING_DOWN,
        strategy_weights={"arbitrage": 0.40, "grid": 0.35, "trend": 0.15, "scalping": 0.10},
        priority_order=["arbitrage", "grid", "trend", "scalping"],
        max_leverage=1.5,
        risk_profile="conservative",
        min_capital=500.0,
        rebalance_frequency_hours=6,
        tags=["bear", "preservation", "defensive"],
    ),
    "balanced": StrategyComboTemplate(
        id="balanced",
        name="均衡配置",
        description="不确定性市场中均衡分配，等风险贡献",
        market_regime=ComboMarketRegime.ALL,
        strategy_weights={"trend": 0.25, "grid": 0.25, "scalping": 0.25, "arbitrage": 0.25},
        priority_order=["trend", "grid", "scalping", "arbitrage"],
        max_leverage=3.0,
        risk_profile="moderate",
        min_capital=500.0,
        rebalance_frequency_hours=4,
        tags=["balanced", "neutral", "risk_parity"],
    ),
    "recovery_mode": StrategyComboTemplate(
        id="recovery_mode",
        name="回血模式",
        description="经历大回撤后保守恢复，小仓位高胜率策略为主",
        market_regime=ComboMarketRegime.RECOVERY,
        strategy_weights={"grid": 0.45, "arbitrage": 0.30, "scalping": 0.15, "trend": 0.10},
        priority_order=["grid", "arbitrage", "scalping", "trend"],
        max_leverage=1.0,
        risk_profile="conservative",
        min_capital=200.0,
        rebalance_frequency_hours=8,
        tags=["recovery", "conservative", "drawdown_recovery"],
    ),
}


# ============================================================
# 组合优化引擎
# ============================================================

class PortfolioOptimizer:
    """
    策略组合优化引擎

    统一管理多策略组合的：
    - MPT 优化（有效前沿、最大夏普、风险平价）
    - 相关性分析（滚动窗口、同质化检测）
    - 绩效归因（PnL贡献、风险贡献）
    - VaR/CVaR 风险度量
    - 策略组合模板切换
    - 资金效率优化
    """

    # ── 初始化 ──────────────────────────────────────────────

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 优化配置
        pf_cfg = config.get("portfolio_optimizer", {})
        self._enabled = pf_cfg.get("enabled", True)
        self._default_objective = OptimizationObjective(
            pf_cfg.get("default_objective", "max_sharpe")
        )
        self._rebalance_interval_hours = pf_cfg.get("rebalance_interval_hours", 4)
        self._min_rebalance_change_pct = pf_cfg.get("min_rebalance_change_pct", 0.02)
        self._max_single_weight = pf_cfg.get("max_single_weight", 0.50)
        self._min_single_weight = pf_cfg.get("min_single_weight", 0.05)
        self._risk_free_rate = pf_cfg.get("risk_free_rate", 0.03)

        # 相关性配置
        corr_cfg = pf_cfg.get("correlation", {})
        self._corr_window_days = corr_cfg.get("window_days", 30)
        self._corr_high_threshold = corr_cfg.get("high_threshold", 0.70)
        self._corr_alert_threshold = corr_cfg.get("alert_threshold", 0.85)

        # VaR 配置
        var_cfg = pf_cfg.get("var", {})
        self._var_confidence_levels = var_cfg.get("confidence_levels", [0.95, 0.99])
        self._var_horizon_days = var_cfg.get("horizon_days", 1)
        self._var_method = var_cfg.get("method", "historical")

        # 绩效数据缓存
        self._strategy_performances: Dict[str, StrategyPerformance] = {}
        self._returns_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=365)  # 保留一年数据
        )
        self._equity_history: List[float] = []
        self._last_optimization: Optional[PortfolioOptimizationResult] = None
        self._last_correlation: Optional[CorrelationMatrix] = None
        self._last_attribution: Optional[PerformanceAttribution] = None
        self._last_var: Dict[str, VaRResult] = {}
        self._current_combo_id: Optional[str] = None

        # 压力测试场景
        self._stress_scenarios = self._init_stress_scenarios(pf_cfg.get("stress_test", {}))

        # 组合模板
        self._combos: Dict[str, StrategyComboTemplate] = dict(PRESET_COMBOS)

        # 锁
        self._lock = threading.RLock()

        # 回调
        self._rebalance_callbacks: List[Callable] = []

        # 再平衡历史（最多 100 条）
        self._rebalance_history: List[Dict[str, Any]] = []

        logger.info(f"PortfolioOptimizer initialized: objective={self._default_objective.value}, "
                    f"rebalance_every={self._rebalance_interval_hours}h")

    def _init_stress_scenarios(self, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        """初始化压力测试场景"""
        enabled = cfg.get("enabled", True)
        if not enabled:
            return []

        return [
            {
                "name": "flash_crash",
                "description": "闪电崩盘：BTC 瞬间下跌 30%",
                "market_change_pct": -0.30,
                "volatility_multiplier": 5.0,
                "correlation_spike": 0.90,
                "liquidity_impact": 0.40,
            },
            {
                "name": "bull_run_exhaustion",
                "description": "牛市见顶：连续3日回调 5%",
                "market_change_pct": -0.05,
                "volatility_multiplier": 2.5,
                "correlation_spike": 0.70,
                "liquidity_impact": 0.15,
                "duration_days": 3,
            },
            {
                "name": "volatility_explosion",
                "description": "波动率爆炸：波动率翻3倍",
                "market_change_pct": 0.0,
                "volatility_multiplier": 3.0,
                "correlation_spike": 0.60,
                "liquidity_impact": 0.10,
            },
            {
                "name": "black_swan",
                "description": "黑天鹅：BTC -40%，所有币种高相关性暴跌",
                "market_change_pct": -0.40,
                "volatility_multiplier": 8.0,
                "correlation_spike": 0.95,
                "liquidity_impact": 0.60,
            },
            {
                "name": "funding_rate_crush",
                "description": "资金费率崩溃：资金费率转极度负值",
                "market_change_pct": -0.15,
                "volatility_multiplier": 2.0,
                "correlation_spike": 0.50,
                "liquidity_impact": 0.20,
                "funding_rate_impact": -0.001,
            },
        ]

    # ── 策略绩效数据管理 ─────────────────────────────────────────

    def update_performance(self, name: str, daily_return: float,
                           metrics: Dict[str, Any] = None) -> None:
        """更新单个策略的收益率数据"""
        try:
            with self._lock:
                self._returns_history[name].append(daily_return)

                perf = self._strategy_performances.get(name)
                if perf is None:
                    perf = StrategyPerformance(name=name)
                    self._strategy_performances[name] = perf

                perf.returns = list(self._returns_history[name])
                if metrics:
                    for key, value in metrics.items():
                        if hasattr(perf, key):
                            setattr(perf, key, value)

                # 重新计算统计量
                if len(perf.returns) >= 5:
                    returns_arr = np.array(perf.returns)
                    perf.annualized_return = float(np.mean(returns_arr) * 365)
                    perf.annualized_volatility = float(np.std(returns_arr, ddof=1) * np.sqrt(365))

                    if perf.annualized_volatility > 0:
                        perf.sharpe_ratio = float(
                            (perf.annualized_return - self._risk_free_rate) / perf.annualized_volatility
                        )

                    # 下行偏差
                    downside = returns_arr[returns_arr < 0]
                    if len(downside) > 0:
                        downside_std = float(np.std(downside, ddof=1) * np.sqrt(365))
                        if downside_std > 0:
                            perf.sortino_ratio = float(
                                (perf.annualized_return - self._risk_free_rate) / downside_std
                            )

                    # Calmar
                    if perf.max_drawdown > 0:
                        perf.calmar_ratio = float(perf.annualized_return / perf.max_drawdown)
        except Exception as e:
            logger.error(f"update_performance failed for '{name}': {e}")

    def update_equity(self, total_equity: float) -> None:
        """更新组合总权益"""
        with self._lock:
            self._equity_history.append(total_equity)
            if len(self._equity_history) > 365:
                self._equity_history = self._equity_history[-365:]

    def get_strategy_returns(self, names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """获取策略收益率矩阵"""
        with self._lock:
            names = names or list(self._returns_history.keys())
            min_len = min(
                (len(self._returns_history.get(n, [])) for n in names),
                default=0
            )
            if min_len < 5:
                return np.array([]), []

            data = []
            valid_names = []
            for n in names:
                returns = list(self._returns_history.get(n, []))[-min_len:]
                if len(returns) == min_len:
                    data.append(returns)
                    valid_names.append(n)

            return np.array(data).T if data else np.array([]), valid_names

    # ── 相关性矩阵 ─────────────────────────────────────────────

    def compute_correlation_matrix(self, names: List[str] = None) -> CorrelationMatrix:
        """计算策略相关性矩阵"""
        try:
            with self._lock:
                returns, valid_names = self.get_strategy_returns(names)
                if returns.size == 0 or len(valid_names) < 2:
                    return CorrelationMatrix(
                        strategies=valid_names,
                        window_days=self._corr_window_days,
                    )

                # Pearson 相关系数
                corr = np.corrcoef(returns.T)
                n = len(valid_names)

                # 平均相关性
                upper_tri = []
                high_corr_pairs = []
                for i in range(n):
                    for j in range(i + 1, n):
                        val = float(corr[i, j])
                        upper_tri.append(val)
                        if abs(val) >= self._corr_high_threshold:
                            high_corr_pairs.append({
                                "pair": [valid_names[i], valid_names[j]],
                                "correlation": round(val, 4),
                                "level": "critical" if abs(val) >= self._corr_alert_threshold else "warning",
                            })

                avg_corr = float(np.mean(upper_tri)) if upper_tri else 0.0

                result = CorrelationMatrix(
                    strategies=valid_names,
                    matrix=corr,
                    window_days=self._corr_window_days,
                    avg_correlation=avg_corr,
                    high_corr_pairs=sorted(high_corr_pairs, key=lambda x: abs(x["correlation"]), reverse=True),
                )
                self._last_correlation = result
                return result
        except Exception as e:
            logger.error(f"compute_correlation_matrix failed: {e}")
            return CorrelationMatrix(
                strategies=names if names else [],
                window_days=self._corr_window_days,
            )

    # ── VaR/CVaR 计算 ────────────────────────────────────────

    def compute_var(self, returns: List[float] = None,
                    method: str = None,
                    confidence: float = 0.95) -> VaRResult:
        """
        计算 VaR/CVaR

        Args:
            returns: 收益率序列（如果为None，使用组合权益序列）
            method: historical / parametric / monte_carlo
            confidence: 置信水平
        """
        try:
            method = method or self._var_method
            with self._lock:
                if returns is None:
                    # 从权益历史计算日收益率
                    if len(self._equity_history) < 2:
                        return VaRResult(
                            method=method, confidence_level=confidence,
                            horizon_days=self._var_horizon_days, var_value=0.0,
                        )
                    equity_arr = np.array(self._equity_history)
                    returns = list(np.diff(equity_arr) / equity_arr[:-1])

                returns_arr = np.array(returns)
                if len(returns_arr) < 20:
                    return VaRResult(
                        method=method, confidence_level=confidence,
                        horizon_days=self._var_horizon_days, var_value=0.0,
                    )

                if method == "historical":
                    var_val, cvar_val = self._var_historical(returns_arr, confidence)
                elif method == "parametric":
                    var_val, cvar_val = self._var_parametric(returns_arr, confidence)
                elif method == "monte_carlo":
                    var_val, cvar_val = self._var_monte_carlo(returns_arr, confidence)
                else:
                    var_val, cvar_val = self._var_historical(returns_arr, confidence)

                # 扩展到持有期
                var_val *= np.sqrt(self._var_horizon_days)
                cvar_val *= np.sqrt(self._var_horizon_days)

                equity = self._equity_history[-1] if self._equity_history else 1.0
                var_pct = abs(var_val) if equity > 0 else 0.0
                cvar_pct = abs(cvar_val) if equity > 0 else 0.0

                result = VaRResult(
                    method=method,
                    confidence_level=confidence,
                    horizon_days=self._var_horizon_days,
                    var_value=round(var_val * equity, 2) if equity > 0 else 0.0,
                    cvar_value=round(cvar_val * equity, 2) if equity > 0 else 0.0,
                    var_pct=round(var_pct, 6),
                    cvar_pct=round(cvar_pct, 6),
                    details={
                        "sample_size": len(returns_arr),
                        "mean_return": float(np.mean(returns_arr)),
                        "std_return": float(np.std(returns_arr, ddof=1)),
                    },
                )

                self._last_var[f"{method}_{confidence}"] = result
                return result
        except Exception as e:
            logger.error(f"compute_var failed: {e}")
            return VaRResult(
                method=method if method else self._var_method,
                confidence_level=confidence,
                horizon_days=self._var_horizon_days,
                var_value=0.0,
            )

    def compute_all_var(self) -> Dict[str, VaRResult]:
        """计算所有配置的 VaR 指标"""
        results = {}
        for confidence in self._var_confidence_levels:
            for method in ["historical", "parametric"]:
                key = f"{method}_{confidence}"
                results[key] = self.compute_var(method=method, confidence=confidence)
        return results

    @staticmethod
    def _var_historical(returns: np.ndarray, confidence: float) -> Tuple[float, float]:
        """历史模拟法 VaR"""
        sorted_returns = np.sort(returns)
        idx = int(len(sorted_returns) * (1 - confidence))
        var_val = float(sorted_returns[idx])
        cvar_val = float(np.mean(sorted_returns[:idx + 1]))
        return var_val, cvar_val

    @staticmethod
    def _var_parametric(returns: np.ndarray, confidence: float) -> Tuple[float, float]:
        """参数法 VaR（假设正态分布）"""
        from scipy.stats import norm
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        z_score = norm.ppf(1 - confidence)
        var_val = float(mu - z_score * sigma)
        # 对于正态分布，CVaR 的计算
        cvar_val = float(mu - sigma * norm.pdf(z_score) / (1 - confidence))
        return var_val, cvar_val

    @staticmethod
    def _var_monte_carlo(returns: np.ndarray, confidence: float,
                          n_simulations: int = 10000) -> Tuple[float, float]:
        """蒙特卡洛法 VaR"""
        mu = np.mean(returns)
        sigma = np.std(returns, ddof=1)
        simulated = np.random.normal(mu, sigma, n_simulations)
        sorted_sim = np.sort(simulated)
        idx = int(n_simulations * (1 - confidence))
        var_val = float(sorted_sim[idx])
        cvar_val = float(np.mean(sorted_sim[:idx + 1]))
        return var_val, cvar_val

    # ── MPT 组合优化 ───────────────────────────────────────────

    def optimize(self, names: List[str] = None,
                 objective: OptimizationObjective = None,
                 constraints: Dict[str, Any] = None) -> Optional[PortfolioOptimizationResult]:
        """
        MPT 组合优化

        Args:
            names: 策略名称列表
            objective: 优化目标
            constraints: 额外约束（如固定权重）
        """
        objective = objective or self._default_objective
        with self._lock:
            returns, valid_names = self.get_strategy_returns(names)
            if returns.size == 0 or len(valid_names) < 2:
                logger.warning("Insufficient data for portfolio optimization")
                return None

            n = len(valid_names)
            mean_returns = np.mean(returns, axis=0) * 365
            cov_matrix = np.cov(returns.T) * 365

            # 默认约束
            bounds = [(self._min_single_weight, self._max_single_weight) for _ in range(n)]
            constraints_list = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

            # 自定义约束
            constraints = constraints or {}
            if "fixed_weights" in constraints:
                for name, fixed_w in constraints["fixed_weights"].items():
                    if name in valid_names:
                        idx = valid_names.index(name)
                        bounds[idx] = (fixed_w, fixed_w)

            # 初始权重：等权
            init_weights = np.ones(n) / n

            try:
                if objective == OptimizationObjective.MAX_SHARPE:
                    result = self._optimize_max_sharpe(
                        mean_returns, cov_matrix, bounds, constraints_list, init_weights
                    )
                elif objective == OptimizationObjective.MIN_VARIANCE:
                    result = self._optimize_min_variance(
                        mean_returns, cov_matrix, bounds, constraints_list, init_weights
                    )
                elif objective == OptimizationObjective.RISK_PARITY:
                    result = self._optimize_risk_parity(
                        mean_returns, cov_matrix, bounds, init_weights
                    )
                elif objective == OptimizationObjective.EQUAL_WEIGHT:
                    result = self._optimize_equal_weight(valid_names)
                elif objective == OptimizationObjective.MAX_SORTINO:
                    result = self._optimize_max_sharpe(
                        mean_returns, cov_matrix, bounds, constraints_list, init_weights
                    )
                elif objective == OptimizationObjective.TARGET_RETURN:
                    target = constraints.get("target_return", 0.20)
                    result = self._optimize_target_return(
                        mean_returns, cov_matrix, bounds, target
                    )
                else:
                    result = self._optimize_max_sharpe(
                        mean_returns, cov_matrix, bounds, constraints_list, init_weights
                    )

                if result is None:
                    return None

                optimal_weights, opt_meta = result

                # 构建结果
                weights_dict = {}
                for i, name in enumerate(valid_names):
                    weights_dict[name] = float(optimal_weights[i])

                try:
                    w = np.array(list(weights_dict.values()))
                    port_return = float(np.dot(w, mean_returns))
                    port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
                    port_sharpe = float((port_return - self._risk_free_rate) / port_vol) if port_vol > 0 else 0.0

                    # 分散度比率
                    indiv_vols = np.sqrt(np.diag(cov_matrix))
                    weighted_vol = float(np.dot(w, indiv_vols))
                    div_ratio = weighted_vol / port_vol if port_vol > 0 else 0.0
                except Exception as e:
                    logger.error(f"Portfolio stats computation failed: {e}")
                    port_return, port_vol, port_sharpe, div_ratio = 0.0, 0.0, 0.0, 0.0

                # 权重变化
                weight_changes = {}
                for name in valid_names:
                    perf = self._strategy_performances.get(name)
                    current = perf.current_weight if perf else 0.0
                    weight_changes[name] = weights_dict[name] - current

                # 有效前沿
                try:
                    frontier = self._compute_efficient_frontier(mean_returns, cov_matrix, bounds)
                except Exception as e:
                    logger.error(f"Efficient frontier computation failed: {e}")
                    frontier = []

                result = PortfolioOptimizationResult(
                    objective=objective.value,
                    optimal_weights=weights_dict,
                    expected_return=port_return,
                    expected_volatility=port_vol,
                    expected_sharpe=port_sharpe,
                    diversification_ratio=div_ratio,
                    weight_changes=weight_changes,
                    optimization_metadata=opt_meta,
                    efficient_frontier=frontier,
                )

                self._last_optimization = result
                return result

            except Exception as e:
                logger.error(f"Portfolio optimization failed: {e}")
                return None

    def _optimize_max_sharpe(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                              bounds: List, constraints: List,
                              init_weights: np.ndarray) -> Optional[Tuple[np.ndarray, Dict]]:
        """最大化夏普比率"""
        n = len(mean_returns)

        def neg_sharpe(w):
            port_return = np.dot(w, mean_returns)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(port_return - self._risk_free_rate) / port_vol if port_vol > 0 else 1e9

        if HAS_SCIPY:
            opt_result = minimize(
                neg_sharpe, init_weights,
                bounds=bounds, constraints=constraints,
                method='SLSQP', options={"maxiter": 1000, "ftol": 1e-10},
            )
            return opt_result.x, {"iterations": opt_result.nit, "success": opt_result.success}
        else:
            # numpy fallback: grid search
            return self._grid_search(neg_sharpe, n, bounds)

    def _optimize_min_variance(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                bounds: List, constraints: List,
                                init_weights: np.ndarray) -> Optional[Tuple[np.ndarray, Dict]]:
        """最小方差优化"""
        n = len(mean_returns)

        def portfolio_variance(w):
            return np.dot(w.T, np.dot(cov_matrix, w))

        if HAS_SCIPY:
            opt_result = minimize(
                portfolio_variance, init_weights,
                bounds=bounds, constraints=constraints,
                method='SLSQP', options={"maxiter": 1000, "ftol": 1e-10},
            )
            return opt_result.x, {"iterations": opt_result.nit, "success": opt_result.success}
        else:
            return self._grid_search(portfolio_variance, n, bounds)

    def _optimize_risk_parity(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                               bounds: List, init_weights: np.ndarray
                               ) -> Optional[Tuple[np.ndarray, Dict]]:
        """风险平价优化（各策略边际风险贡献相等）"""
        n = len(mean_returns)

        def risk_parity_objective(w):
            w = np.abs(w)
            w = w / np.sum(w)
            port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            mrc = np.dot(cov_matrix, w) / port_vol  # 边际风险贡献
            rc = w * mrc  # 风险贡献
            target_rc = port_vol / n
            return np.sum((rc - target_rc) ** 2)

        if HAS_SCIPY:
            constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
            opt_result = minimize(
                risk_parity_objective, init_weights,
                bounds=bounds, constraints=constraints,
                method='SLSQP', options={"maxiter": 2000, "ftol": 1e-10},
            )
            w = np.abs(opt_result.x)
            w = w / np.sum(w)
            return w, {"method": "risk_parity", "iterations": opt_result.nit}
        else:
            return self._grid_search(risk_parity_objective, n, bounds)

    def _optimize_equal_weight(self, names: List[str]) -> Tuple[np.ndarray, Dict]:
        """等权配置"""
        n = len(names)
        return np.ones(n) / n, {"method": "equal_weight"}

    def _optimize_target_return(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                 bounds: List, target_return: float) -> Optional[Tuple[np.ndarray, Dict]]:
        """目标收益下最小方差"""
        n = len(mean_returns)

        def portfolio_variance(w):
            return np.dot(w.T, np.dot(cov_matrix, w))

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w: np.dot(w, mean_returns) - target_return},
        ]

        if HAS_SCIPY:
            init = np.ones(n) / n
            opt_result = minimize(
                portfolio_variance, init,
                bounds=bounds, constraints=constraints,
                method='SLSQP', options={"maxiter": 1000, "ftol": 1e-10},
            )
            return opt_result.x, {"target_return": target_return, "success": opt_result.success}
        return None

    def _grid_search(self, objective_fn, n: int,
                     bounds: List[Tuple[float, float]],
                     grid_points: int = 20) -> Optional[Tuple[np.ndarray, Dict]]:
        """网格搜索（numpy fallback）"""
        best_weights = None
        best_score = float('inf')

        # 生成候选权重（简单采样）
        for _ in range(grid_points ** min(n - 1, 2)):
            w = np.random.dirichlet(np.ones(n))
            # 应用边界检查
            in_bounds = True
            for i, (lo, hi) in enumerate(bounds):
                if w[i] < lo or w[i] > hi:
                    in_bounds = False
                    break
            if not in_bounds:
                continue

            score = objective_fn(w)
            if score < best_score:
                best_score = score
                best_weights = w

        if best_weights is None:
            best_weights = np.ones(n) / n

        return best_weights, {"method": "grid_search", "grid_points": grid_points}

    def _compute_efficient_frontier(self, mean_returns: np.ndarray,
                                     cov_matrix: np.ndarray,
                                     bounds: List) -> List[Dict[str, float]]:
        """计算有效前沿"""
        frontier = []
        n = len(mean_returns)
        min_ret = float(np.min(mean_returns))
        max_ret = float(np.max(mean_returns))

        if max_ret <= min_ret:
            return frontier

        for target in np.linspace(min_ret, max_ret, 30):
            result = self._optimize_target_return(mean_returns, cov_matrix, bounds, float(target))
            if result:
                w, _ = result
                port_return = float(np.dot(w, mean_returns))
                port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))))
                frontier.append({
                    "return": round(port_return, 6),
                    "volatility": round(port_vol, 6),
                    "sharpe": round((port_return - self._risk_free_rate) / port_vol, 4) if port_vol > 0 else 0,
                })

        return frontier

    # ── 绩效归因 ──────────────────────────────────────────────

    def compute_attribution(self) -> PerformanceAttribution:
        """计算绩效归因"""
        with self._lock:
            active_strategies = {
                name: perf for name, perf in self._strategy_performances.items()
                if perf.enabled and perf.pnl_contribution != 0
            }
            if not active_strategies:
                return PerformanceAttribution()

            total_pnl = sum(p.pnl_contribution for p in active_strategies.values())
            total_risk = sum(p.annualized_volatility for p in active_strategies.values())

            contributions = {}
            risk_contributions = {}
            top_name, top_pct = "", -float('inf')
            worst_name, worst_pct = "", float('inf')

            for name, perf in active_strategies.items():
                pnl_pct = perf.pnl_contribution / total_pnl if total_pnl != 0 else 0
                risk_pct = perf.annualized_volatility / total_risk if total_risk > 0 else 0

                contributions[name] = {
                    "pnl_contribution": round(perf.pnl_contribution, 4),
                    "pnl_pct": round(pnl_pct, 4),
                    "sharpe": round(perf.sharpe_ratio, 4),
                    "win_rate": round(perf.win_rate, 4),
                    "max_drawdown": round(perf.max_drawdown, 4),
                    "return_on_capital": round(
                        perf.pnl_contribution / max(perf.current_weight, 0.01), 4
                    ),
                }
                risk_contributions[name] = round(risk_pct, 4)

                if pnl_pct > top_pct:
                    top_name, top_pct = name, pnl_pct
                if pnl_pct < worst_pct:
                    worst_name, worst_pct = name, pnl_pct

            # 集中度（前2名贡献占比）
            sorted_contribs = sorted(contributions.items(),
                                     key=lambda x: x[1]["pnl_pct"], reverse=True)
            concentration = sum(
                c[1]["pnl_pct"] for c in sorted_contribs[:2]
            ) if len(sorted_contribs) >= 2 else 1.0

            # 分散度评分（基于风险贡献的熵）
            if risk_contributions:
                risk_vals = np.array(list(risk_contributions.values()))
                risk_vals = risk_vals / risk_vals.sum()
                entropy = -np.sum(risk_vals * np.log(risk_vals + 1e-10))
                max_entropy = np.log(len(risk_vals))
                diversification = float(entropy / max_entropy) if max_entropy > 0 else 0
            else:
                diversification = 0

            result = PerformanceAttribution(
                strategy_contributions=contributions,
                top_contributor=top_name,
                top_contributor_pct=top_pct,
                worst_contributor=worst_name,
                worst_contributor_pct=worst_pct,
                concentration_ratio=concentration,
                diversification_score=diversification,
                risk_contributions=risk_contributions,
            )
            self._last_attribution = result
            return result

    # ── 策略组合模板 ──────────────────────────────────────────

    def get_combo(self, combo_id: str) -> Optional[StrategyComboTemplate]:
        """获取策略组合模板"""
        return self._combos.get(combo_id)

    def list_combos(self, regime: ComboMarketRegime = None) -> List[StrategyComboTemplate]:
        """列出所有组合模板"""
        if regime:
            return [c for c in self._combos.values() if c.market_regime == regime]
        return list(self._combos.values())

    def recommend_combo(self, market_regime: ComboMarketRegime,
                         capital: float = None) -> List[Dict[str, Any]]:
        """
        根据市场状态推荐最佳策略组合

        Returns:
            按匹配度排序的组合推荐列表
        """
        candidates = []

        for combo in self._combos.values():
            score = 0.0

            # 市场状态完全匹配
            if combo.market_regime == market_regime:
                score += 10.0
            elif combo.market_regime == ComboMarketRegime.ALL:
                score += 2.0

            # 资金门槛匹配
            if capital is not None:
                if capital >= combo.min_capital:
                    score += 3.0
                else:
                    score -= 5.0  # 资金不足，严重降权

            # 风险偏好匹配：根据回撤情况
            if self._equity_history:
                current_equity = self._equity_history[-1]
                peak = max(self._equity_history)
                drawdown = (peak - current_equity) / peak if peak > 0 else 0
                if drawdown > 0.10:  # 回撤>10%，偏好保守
                    if combo.risk_profile == "conservative":
                        score += 3.0
                    elif combo.risk_profile == "aggressive":
                        score -= 3.0

            candidates.append({
                "combo": combo.to_dict(),
                "score": round(score, 1),
                "match_reasons": self._get_match_reasons(combo, market_regime, capital),
            })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def _get_match_reasons(self, combo: StrategyComboTemplate,
                            regime: ComboMarketRegime, capital: float) -> List[str]:
        """生成匹配理由"""
        reasons = []
        if combo.market_regime == regime:
            reasons.append(f"Market regime exact match: {regime.value}")
        elif combo.market_regime == ComboMarketRegime.ALL:
            reasons.append("Universal combo for any market condition")
        if capital and capital >= combo.min_capital:
            reasons.append(f"Capital sufficient (have {capital}, need {combo.min_capital})")
        return reasons

    def apply_combo(self, combo_id: str) -> Dict[str, Any]:
        """
        应用策略组合模板

        Returns:
            应用结果，含新旧权重对比
        """
        combo = self._combos.get(combo_id)
        if not combo:
            return {"success": False, "error": f"Combo '{combo_id}' not found"}

        old_weights = {}
        for name, perf in self._strategy_performances.items():
            old_weights[name] = perf.current_weight

        # 更新策略权重
        for name, weight in combo.strategy_weights.items():
            if name in self._strategy_performances:
                self._strategy_performances[name].current_weight = weight

        self._current_combo_id = combo_id

        changes = {}
        for name in combo.strategy_weights:
            old = old_weights.get(name, 0.0)
            new = combo.strategy_weights[name]
            changes[name] = round(new - old, 4)

        logger.info(f"Applied combo '{combo.name}': weights={combo.strategy_weights}")

        return {
            "success": True,
            "combo_id": combo_id,
            "combo_name": combo.name,
            "new_weights": combo.strategy_weights,
            "weight_changes": changes,
            "max_leverage": combo.max_leverage,
            "risk_profile": combo.risk_profile,
            "rebalance_frequency_hours": combo.rebalance_frequency_hours,
        }

    # ── 压力测试 ──────────────────────────────────────────────

    def stress_test(self, weights: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        对当前组合进行压力测试

        Returns:
            各场景下的预估损失
        """
        if not self._stress_scenarios:
            return []

        weights = weights or (
            self._last_optimization.optimal_weights
            if self._last_optimization else {}
        )
        if not weights:
            return []

        total_equity = self._equity_history[-1] if self._equity_history else 10000.0
        results = []

        for scenario in self._stress_scenarios:
            estimated_loss = 0.0
            strategy_impacts = {}

            for name, weight in weights.items():
                perf = self._strategy_performances.get(name)
                if not perf:
                    continue

                # 估计策略在压力场景下的损失
                base_vol = perf.annualized_volatility
                vol_mult = scenario.get("volatility_multiplier", 1.0)
                market_chg = scenario.get("market_change_pct", 0.0)
                corr_spike = scenario.get("correlation_spike", 0.0)
                duration = scenario.get("duration_days", 1)

                # 简化估计：损失 = 市场变动 * 波动率放大 * 持续时间
                strategy_loss_pct = market_chg * vol_mult * np.sqrt(duration)
                strategy_loss = weight * total_equity * abs(strategy_loss_pct)

                estimated_loss += strategy_loss
                strategy_impacts[name] = {
                    "estimated_loss": round(strategy_loss, 2),
                    "loss_pct": round(abs(strategy_loss_pct), 4),
                    "weight": weight,
                }

            results.append({
                "scenario": scenario["name"],
                "description": scenario["description"],
                "estimated_total_loss": round(estimated_loss, 2),
                "estimated_loss_pct": round(
                    estimated_loss / total_equity, 4
                ) if total_equity > 0 else 0.0,
                "strategy_impacts": strategy_impacts,
                "severity": (
                    "critical" if estimated_loss / max(total_equity, 1) > 0.2
                    else "warning" if estimated_loss / max(total_equity, 1) > 0.1
                    else "moderate"
                ),
            })

        return results

    # ── 资金效率 ──────────────────────────────────────────────

    def maximize_capital_efficiency(self, total_capital: float,
                                     max_leverage: float = 3.0) -> Dict[str, Any]:
        """
        资金效率最大化：在组合约束下优化资金利用率

        确保：
        - 不超出最大杠杆
        - 策略间资金共享（保证金不重复计算）
        - 预留风险缓冲
        """
        with self._lock:
            active_weights = {}
            for name, perf in self._strategy_performances.items():
                if perf.enabled and perf.current_weight > 0:
                    active_weights[name] = perf.current_weight

            if self._last_optimization:
                active_weights = self._last_optimization.optimal_weights

            if not active_weights:
                active_weights = {"trend": 0.25, "grid": 0.25, "scalping": 0.25, "arbitrage": 0.25}

            # 归一化
            total_w = sum(active_weights.values())
            normalized = {k: v / total_w for k, v in active_weights.items()}

            # 资金分配
            risk_buffer = 0.15  # 15% 风险缓冲
            deployable = total_capital * (1 - risk_buffer)

            allocations = {}
            for name, weight in normalized.items():
                allocations[name] = {
                    "capital": round(deployable * weight, 2),
                    "max_position_value": round(deployable * weight * max_leverage, 2),
                    "weight": round(weight, 4),
                }

            total_allocated = sum(a["capital"] for a in allocations.values())

            return {
                "total_capital": total_capital,
                "deployable_capital": round(deployable, 2),
                "risk_buffer": round(total_capital * risk_buffer, 2),
                "max_leverage": max_leverage,
                "allocations": allocations,
                "total_allocated": round(total_allocated, 2),
                "efficiency": round(total_allocated / total_capital, 4) if total_capital > 0 else 0,
            }

    # ── 组合健康检查 ──────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        """组合级健康检查"""
        issues = []
        warnings = []
        status = "healthy"

        with self._lock:
            # 检查相关性过高
            if self._last_correlation:
                if self._last_correlation.avg_correlation > self._corr_alert_threshold:
                    issues.append({
                        "type": "high_correlation",
                        "avg_correlation": self._last_correlation.avg_correlation,
                        "threshold": self._corr_alert_threshold,
                        "message": "策略间相关性过高，分散化失效",
                    })
                    status = "critical"

            # 检查权重集中度
            if self._last_optimization:
                weights = list(self._last_optimization.optimal_weights.values())
                max_w = max(weights) if weights else 0
                if max_w > 0.5:
                    warnings.append({
                        "type": "weight_concentration",
                        "max_weight": max_w,
                        "message": f"单策略权重过高 ({max_w:.1%})，风险集中",
                    })

            # 检查 VaR 超标
            for key, var_result in self._last_var.items():
                if var_result.var_pct > 0.05:
                    issues.append({
                        "type": "var_breach",
                        "method": var_result.method,
                        "confidence": var_result.confidence_level,
                        "var_pct": var_result.var_pct,
                        "message": f"VaR 超标: {var_result.var_pct:.2%} (5% limit)",
                    })
                    status = "critical"
                elif var_result.var_pct > 0.03:
                    warnings.append({
                        "type": "var_warning",
                        "method": var_result.method,
                        "confidence": var_result.confidence_level,
                        "var_pct": var_result.var_pct,
                    })

            # 检查回撤
            if self._equity_history:
                current = self._equity_history[-1]
                peak = max(self._equity_history)
                dd = (peak - current) / peak if peak > 0 else 0
                if dd > 0.15:
                    issues.append({
                        "type": "large_drawdown",
                        "drawdown": round(dd, 4),
                        "message": f"组合回撤 {dd:.1%} 超过15%",
                    })
                    status = "critical" if dd > 0.20 else "warning"

        return {
            "status": status,
            "issues": issues,
            "warnings": warnings,
            "issue_count": len(issues),
            "warning_count": len(warnings),
            "timestamp": datetime.now().isoformat(),
        }

    # ── 再平衡回调 ───────────────────────────────────────────

    def on_rebalance(self, callback: Callable) -> None:
        """注册再平衡回调"""
        self._rebalance_callbacks.append(callback)

    async def run_rebalance_loop(self):
        """定期再平衡循环"""
        logger.info(f"PortfolioOptimizer rebalance loop started "
                    f"(interval={self._rebalance_interval_hours}h)")

        while self._enabled:
            try:
                await asyncio.sleep(self._rebalance_interval_hours * 3600)
                await self._execute_rebalance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rebalance loop error: {e}")

    async def _execute_rebalance(self) -> Dict[str, Any]:
        """执行组合再平衡"""
        # 1. 更新相关性
        self.compute_correlation_matrix()

        # 2. 优化权重
        result = self.optimize()
        if not result:
            return {"rebalanced": False, "reason": "Optimization failed"}

        # 3. 检查是否需要调整（变更是否足够大）
        max_change = max(
            (abs(v) for v in result.weight_changes.values()),
            default=0.0
        )
        if max_change < self._min_rebalance_change_pct:
            return {"rebalanced": False, "reason": f"Max change {max_change:.4f} below threshold"}

        # 4. 通知回调
        for cb in self._rebalance_callbacks:
            try:
                cb(result)
            except Exception:
                pass

        # 5. 记录再平衡历史
        self._rebalance_history.append({
            "timestamp": datetime.now().isoformat(),
            "weights": result.optimal_weights,
            "reason": f"Rebalance triggered: max_change={max_change:.4f}",
        })
        if len(self._rebalance_history) > 100:
            self._rebalance_history = self._rebalance_history[-100:]

        logger.info(f"Portfolio rebalanced: weights={result.optimal_weights}")
        return {"rebalanced": True, "result": result.to_dict()}

    # ── 状态持久化 ────────────────────────────────────────────

    def collect_persistent_state(self) -> Dict[str, Any]:
        """
        收集所有需要持久化的状态数据。

        Returns:
            可序列化为 JSON 的状态字典
        """
        with self._lock:
            state = {
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                "strategies": {
                    name: perf.to_dict()
                    for name, perf in self._strategy_performances.items()
                },
                "returns_history": {
                    name: list(dq)
                    for name, dq in self._returns_history.items()
                },
                "equity_history": self._equity_history,
                "last_optimization": self._last_optimization.to_dict() if self._last_optimization else None,
                "last_correlation": self._last_correlation.to_dict() if self._last_correlation else None,
                "last_attribution": self._last_attribution.to_dict() if self._last_attribution else None,
                "last_var": {
                    k: v.to_dict() for k, v in self._last_var.items()
                },
                "current_combo_id": self._current_combo_id,
                "rebalance_history": self._rebalance_history,
            }
            return state

    def save_persistent_state(self, filepath: str = None) -> bool:
        """
        将状态保存到 JSON 文件。

        Args:
            filepath: 保存路径，默认为 data/portfolio_state.json

        Returns:
            是否保存成功
        """
        if filepath is None:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            filepath = os.path.join(project_root, "data", "portfolio_state.json")

        try:
            state = self.collect_persistent_state()
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"Portfolio state saved to {filepath} ({len(json.dumps(state))} bytes)")
            return True
        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")
            return False

    def restore_persistent_state(self, state: Dict[str, Any] = None,
                                  filepath: str = None) -> bool:
        """
        从字典或 JSON 文件恢复持久化状态。

        Args:
            state: 状态字典（如果提供，优先使用）
            filepath: JSON 文件路径，默认为 data/portfolio_state.json

        Returns:
            是否恢复成功
        """
        try:
            if state is None:
                if filepath is None:
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    filepath = os.path.join(project_root, "data", "portfolio_state.json")
                if not os.path.exists(filepath):
                    logger.warning(f"State file not found: {filepath}")
                    return False
                with open(filepath, "r", encoding="utf-8") as f:
                    state = json.load(f)

            with self._lock:
                # 恢复策略绩效
                if "strategies" in state:
                    for name, perf_dict in state["strategies"].items():
                        perf = StrategyPerformance(name=name)
                        for key, value in perf_dict.items():
                            if hasattr(perf, key):
                                setattr(perf, key, value)
                        self._strategy_performances[name] = perf

                # 恢复收益率历史
                if "returns_history" in state:
                    for name, returns_list in state["returns_history"].items():
                        self._returns_history[name] = deque(
                            returns_list, maxlen=365
                        )

                # 恢复权益历史
                if "equity_history" in state:
                    self._equity_history = state["equity_history"]

                # 恢复最后优化结果
                if state.get("last_optimization"):
                    self._last_optimization = PortfolioOptimizationResult(
                        objective=state["last_optimization"].get("objective", ""),
                        optimal_weights=state["last_optimization"].get("optimal_weights", {}),
                        expected_return=state["last_optimization"].get("expected_return", 0.0),
                        expected_volatility=state["last_optimization"].get("expected_volatility", 0.0),
                        expected_sharpe=state["last_optimization"].get("expected_sharpe", 0.0),
                        diversification_ratio=state["last_optimization"].get("diversification_ratio", 0.0),
                        weight_changes=state["last_optimization"].get("weight_changes", {}),
                        optimization_metadata=state["last_optimization"].get("optimization_metadata", {}),
                        efficient_frontier=state["last_optimization"].get("efficient_frontier", []),
                        timestamp=state["last_optimization"].get("timestamp", datetime.now().isoformat()),
                    )

                # 恢复相关性矩阵
                if state.get("last_correlation"):
                    corr_data = state["last_correlation"]
                    matrix_data = corr_data.get("matrix", [])
                    self._last_correlation = CorrelationMatrix(
                        strategies=corr_data.get("strategies", []),
                        matrix=np.array(matrix_data) if matrix_data else np.array([]),
                        window_days=corr_data.get("window_days", self._corr_window_days),
                        last_updated=corr_data.get("last_updated", datetime.now().isoformat()),
                        avg_correlation=corr_data.get("avg_correlation", 0.0),
                        high_corr_pairs=corr_data.get("high_corr_pairs", []),
                    )

                # 恢复归因
                if state.get("last_attribution"):
                    attr_data = state["last_attribution"]
                    self._last_attribution = PerformanceAttribution(
                        strategy_contributions=attr_data.get("strategy_contributions", {}),
                        top_contributor=attr_data.get("top_contributor", ""),
                        top_contributor_pct=attr_data.get("top_contributor_pct", 0.0),
                        worst_contributor=attr_data.get("worst_contributor", ""),
                        worst_contributor_pct=attr_data.get("worst_contributor_pct", 0.0),
                        concentration_ratio=attr_data.get("concentration_ratio", 0.0),
                        diversification_score=attr_data.get("diversification_score", 0.0),
                        risk_contributions=attr_data.get("risk_contributions", {}),
                        timestamp=attr_data.get("timestamp", datetime.now().isoformat()),
                    )

                # 恢复 VaR
                if "last_var" in state:
                    for key, var_dict in state["last_var"].items():
                        self._last_var[key] = VaRResult(
                            method=var_dict.get("method", ""),
                            confidence_level=var_dict.get("confidence_level", 0.95),
                            horizon_days=var_dict.get("horizon_days", self._var_horizon_days),
                            var_value=var_dict.get("var_value", 0.0),
                            cvar_value=var_dict.get("cvar_value", 0.0),
                            var_pct=var_dict.get("var_pct", 0.0),
                            cvar_pct=var_dict.get("cvar_pct", 0.0),
                            timestamp=var_dict.get("timestamp", datetime.now().isoformat()),
                            details=var_dict.get("details", {}),
                        )

                # 恢复当前组合
                self._current_combo_id = state.get("current_combo_id")

                # 恢复再平衡历史
                if "rebalance_history" in state:
                    self._rebalance_history = state["rebalance_history"][-100:]

            logger.info(f"Portfolio state restored: {len(self._strategy_performances)} strategies, "
                       f"{len(self._equity_history)} equity points")
            return True
        except Exception as e:
            logger.error(f"Failed to restore persistent state: {e}")
            return False

    # ── 热更新配置 ────────────────────────────────────────────

    def update_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        热更新配置参数，无需重启。

        Args:
            config: 新配置字典（支持部分更新）

        Returns:
            变更记录字典
        """
        changes = []
        self.config.update(config)
        pf_cfg = config.get("portfolio_optimizer", {})

        if not pf_cfg:
            return {"updated": False, "changes": [], "message": "No portfolio_optimizer section in config"}

        with self._lock:
            # 优化配置
            if "enabled" in pf_cfg:
                old = self._enabled
                self._enabled = pf_cfg["enabled"]
                changes.append({"key": "enabled", "old": old, "new": self._enabled})
                logger.info(f"Config updated: enabled={old} -> {self._enabled}")

            if "default_objective" in pf_cfg:
                try:
                    old = self._default_objective.value
                    self._default_objective = OptimizationObjective(pf_cfg["default_objective"])
                    changes.append({"key": "default_objective", "old": old, "new": self._default_objective.value})
                    logger.info(f"Config updated: default_objective={old} -> {self._default_objective.value}")
                except ValueError as e:
                    logger.warning(f"Invalid objective '{pf_cfg['default_objective']}': {e}")

            if "rebalance_interval_hours" in pf_cfg:
                old = self._rebalance_interval_hours
                self._rebalance_interval_hours = pf_cfg["rebalance_interval_hours"]
                changes.append({"key": "rebalance_interval_hours", "old": old, "new": self._rebalance_interval_hours})
                logger.info(f"Config updated: rebalance_interval_hours={old} -> {self._rebalance_interval_hours}")

            if "min_rebalance_change_pct" in pf_cfg:
                old = self._min_rebalance_change_pct
                self._min_rebalance_change_pct = pf_cfg["min_rebalance_change_pct"]
                changes.append({"key": "min_rebalance_change_pct", "old": old, "new": self._min_rebalance_change_pct})
                logger.info(f"Config updated: min_rebalance_change_pct={old} -> {self._min_rebalance_change_pct}")

            if "max_single_weight" in pf_cfg:
                old = self._max_single_weight
                self._max_single_weight = pf_cfg["max_single_weight"]
                changes.append({"key": "max_single_weight", "old": old, "new": self._max_single_weight})
                logger.info(f"Config updated: max_single_weight={old} -> {self._max_single_weight}")

            if "min_single_weight" in pf_cfg:
                old = self._min_single_weight
                self._min_single_weight = pf_cfg["min_single_weight"]
                changes.append({"key": "min_single_weight", "old": old, "new": self._min_single_weight})
                logger.info(f"Config updated: min_single_weight={old} -> {self._min_single_weight}")

            if "risk_free_rate" in pf_cfg:
                old = self._risk_free_rate
                self._risk_free_rate = pf_cfg["risk_free_rate"]
                changes.append({"key": "risk_free_rate", "old": old, "new": self._risk_free_rate})
                logger.info(f"Config updated: risk_free_rate={old} -> {self._risk_free_rate}")

            # 相关性配置
            corr_cfg = pf_cfg.get("correlation", {})
            if corr_cfg:
                if "window_days" in corr_cfg:
                    old = self._corr_window_days
                    self._corr_window_days = corr_cfg["window_days"]
                    changes.append({"key": "corr_window_days", "old": old, "new": self._corr_window_days})
                    logger.info(f"Config updated: corr_window_days={old} -> {self._corr_window_days}")
                if "high_threshold" in corr_cfg:
                    old = self._corr_high_threshold
                    self._corr_high_threshold = corr_cfg["high_threshold"]
                    changes.append({"key": "corr_high_threshold", "old": old, "new": self._corr_high_threshold})
                    logger.info(f"Config updated: corr_high_threshold={old} -> {self._corr_high_threshold}")
                if "alert_threshold" in corr_cfg:
                    old = self._corr_alert_threshold
                    self._corr_alert_threshold = corr_cfg["alert_threshold"]
                    changes.append({"key": "corr_alert_threshold", "old": old, "new": self._corr_alert_threshold})
                    logger.info(f"Config updated: corr_alert_threshold={old} -> {self._corr_alert_threshold}")

            # VaR 配置
            var_cfg = pf_cfg.get("var", {})
            if var_cfg:
                if "confidence_levels" in var_cfg:
                    old = self._var_confidence_levels
                    self._var_confidence_levels = var_cfg["confidence_levels"]
                    changes.append({"key": "var_confidence_levels", "old": old, "new": self._var_confidence_levels})
                    logger.info(f"Config updated: var_confidence_levels={old} -> {self._var_confidence_levels}")
                if "horizon_days" in var_cfg:
                    old = self._var_horizon_days
                    self._var_horizon_days = var_cfg["horizon_days"]
                    changes.append({"key": "var_horizon_days", "old": old, "new": self._var_horizon_days})
                    logger.info(f"Config updated: var_horizon_days={old} -> {self._var_horizon_days}")
                if "method" in var_cfg:
                    old = self._var_method
                    self._var_method = var_cfg["method"]
                    changes.append({"key": "var_method", "old": old, "new": self._var_method})
                    logger.info(f"Config updated: var_method={old} -> {self._var_method}")

        logger.info(f"Config hot update complete: {len(changes)} parameter(s) changed")
        return {
            "updated": len(changes) > 0,
            "changes": changes,
            "change_count": len(changes),
        }

    # ── 增强指标 ──────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """
        获取增强的统计指标。

        Returns:
            包含各类统计指标的字典
        """
        with self._lock:
            # 策略性能指标
            perf_metrics = {}
            for name, perf in self._strategy_performances.items():
                perf_metrics[name] = {
                    "sharpe_ratio": round(perf.sharpe_ratio, 4),
                    "sortino_ratio": round(perf.sortino_ratio, 4),
                    "win_rate": round(perf.win_rate, 4),
                    "profit_factor": round(perf.profit_factor, 4),
                    "max_drawdown": round(perf.max_drawdown, 4),
                    "calmar_ratio": round(perf.calmar_ratio, 4),
                    "annualized_return": round(perf.annualized_return, 4),
                    "annualized_volatility": round(perf.annualized_volatility, 4),
                    "total_trades": perf.total_trades,
                    "current_weight": round(perf.current_weight, 4),
                    "pnl_contribution": round(perf.pnl_contribution, 4),
                    "enabled": perf.enabled,
                }

            # 相关性统计
            avg_correlation = 0.0
            high_corr_pairs_count = 0
            if self._last_correlation:
                avg_correlation = self._last_correlation.avg_correlation
                high_corr_pairs_count = len(self._last_correlation.high_corr_pairs)

            # 最新 VaR 值
            latest_var = {}
            for key, var_result in self._last_var.items():
                latest_var[key] = {
                    "method": var_result.method,
                    "confidence": var_result.confidence_level,
                    "var_pct": round(var_result.var_pct, 6),
                    "cvar_pct": round(var_result.cvar_pct, 6),
                    "var_value": var_result.var_value,
                }

            # 优化结果摘要
            optimization_summary = None
            if self._last_optimization:
                optimization_summary = {
                    "objective": self._last_optimization.objective,
                    "expected_return": round(self._last_optimization.expected_return, 6),
                    "expected_volatility": round(self._last_optimization.expected_volatility, 6),
                    "expected_sharpe": round(self._last_optimization.expected_sharpe, 4),
                    "diversification_ratio": round(self._last_optimization.diversification_ratio, 4),
                    "timestamp": self._last_optimization.timestamp,
                    "strategy_count": len(self._last_optimization.optimal_weights),
                }

            # 健康状态
            health = self.health_check()

            # 回撤计算
            current_drawdown = 0.0
            if self._equity_history:
                current = self._equity_history[-1]
                peak = max(self._equity_history)
                current_drawdown = round((peak - current) / peak, 4) if peak > 0 else 0.0

            return {
                "total_strategies_tracked": len(self._strategy_performances),
                "active_strategies": sum(
                    1 for p in self._strategy_performances.values() if p.enabled
                ),
                "equity_history_length": len(self._equity_history),
                "current_drawdown": current_drawdown,
                "last_optimization": optimization_summary,
                "last_optimization_timestamp": (
                    self._last_optimization.timestamp if self._last_optimization else None
                ),
                "average_correlation": round(avg_correlation, 4),
                "high_correlation_pairs_count": high_corr_pairs_count,
                "latest_var": latest_var,
                "health_status": health.get("status", "unknown"),
                "health_issues": health.get("issue_count", 0),
                "health_warnings": health.get("warning_count", 0),
                "rebalance_history_count": len(self._rebalance_history),
                "current_combo_id": self._current_combo_id,
                "performance_metrics": perf_metrics,
                "timestamp": datetime.now().isoformat(),
            }

    # ── 自动熔断保护 ──────────────────────────────────────────

    def check_critical_state(self) -> bool:
        """
        检查组合是否处于临界状态，需要自动熔断。

        临界条件：
        - 当前回撤 > 20%
        - 任意 VaR > 5%
        - 平均相关性 > 0.9

        Returns:
            True 表示处于临界状态，需要触发保护措施
        """
        with self._lock:
            critical_reasons = []

            # 检查回撤
            if self._equity_history:
                current = self._equity_history[-1]
                peak = max(self._equity_history)
                drawdown = (peak - current) / peak if peak > 0 else 0
                if drawdown > 0.20:
                    critical_reasons.append(
                        f"Drawdown {drawdown:.2%} exceeds 20% threshold"
                    )

            # 检查 VaR
            for key, var_result in self._last_var.items():
                if var_result.var_pct > 0.05:
                    critical_reasons.append(
                        f"VaR({key}) {var_result.var_pct:.2%} exceeds 5% threshold"
                    )

            # 检查相关性
            if self._last_correlation:
                if self._last_correlation.avg_correlation > 0.9:
                    critical_reasons.append(
                        f"Average correlation {self._last_correlation.avg_correlation:.4f} exceeds 0.9 threshold"
                    )

            if critical_reasons:
                logger.critical(
                    f"Portfolio in CRITICAL state: {'; '.join(critical_reasons)}"
                )
                return True

            return False

    def get_summary(self) -> Dict[str, Any]:
        """获取组合管理完整摘要"""
        return {
            "enabled": self._enabled,
            "default_objective": self._default_objective.value,
            "strategies_tracked": len(self._strategy_performances),
            "equity_history_length": len(self._equity_history),
            "last_optimization": self._last_optimization.to_dict() if self._last_optimization else None,
            "last_correlation": self._last_correlation.to_dict() if self._last_correlation else None,
            "last_attribution": self._last_attribution.to_dict() if self._last_attribution else None,
            "last_var": {k: v.to_dict() for k, v in self._last_var.items()},
            "current_combo_id": self._current_combo_id,
            "combos_available": len(self._combos),
            "health": self.health_check(),
            "timestamp": datetime.now().isoformat(),
        }

    def get_all_performances(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略绩效"""
        return {name: perf.to_dict() for name, perf in self._strategy_performances.items()}


# ── 全局单例 ──────────────────────────────────────────────────

_optimizer: Optional[PortfolioOptimizer] = None
_optimizer_lock = threading.Lock()


def get_portfolio_optimizer(config: Dict[str, Any] = None) -> PortfolioOptimizer:
    """获取全局组合优化器"""
    global _optimizer
    with _optimizer_lock:
        if _optimizer is None:
            _optimizer = PortfolioOptimizer(config)
        elif config:
            _optimizer.config.update(config)
        return _optimizer


def reset_portfolio_optimizer() -> None:
    """重置优化器（测试用）"""
    global _optimizer
    with _optimizer_lock:
        _optimizer = None


__all__ = [
    "PortfolioOptimizer",
    "OptimizationObjective",
    "ComboMarketRegime",
    "RiskMetric",
    "StrategyPerformance",
    "CorrelationMatrix",
    "VaRResult",
    "PerformanceAttribution",
    "StrategyComboTemplate",
    "PortfolioOptimizationResult",
    "PRESET_COMBOS",
    "get_portfolio_optimizer",
    "reset_portfolio_optimizer",
]
