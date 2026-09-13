"""
风险预算分配引擎（Risk Budget Engine）

完善强化的风险预算管理系统，提供：
  - 风险预算定义与分解（总预算→策略预算→交易预算）
  - 前向风险估计（EWMA波动率、CVaR预测）
  - 风险平价分配（数学化求解 + Modified Risk Parity）
  - 风险分解（VaR分解、边际风险贡献MRC、成分风险贡献CRC）
  - 风险利用率追踪与告警
  - 动态风险预算调整（市场状态感知 + 绩效驱动）
  - 风险调整绩效指标（RAPM：Sharpe/Sortino/Calmar/IR/RoMAD）
  - 风险预算瀑布（优先级分配 + 回撤惩罚）
  - 再平衡触发与执行
"""
import asyncio
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Set
import numpy as np
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 枚举与数据模型
# ═══════════════════════════════════════════════════════════════

class RiskBudgetLevel(Enum):
    """风险预算层级"""
    PORTFOLIO = "portfolio"     # 组合级（总预算）
    GROUP = "group"             # 策略组级（如趋势组、套利组）
    STRATEGY = "strategy"       # 策略级
    POSITION = "position"       # 仓位级（单笔交易）


class RiskDecompositionMethod(Enum):
    """风险分解方法"""
    EQUAL_RC = "equal_rc"               # 等风险贡献
    VOLATILITY_PROPORTIONAL = "vol_p"   # 波动率比例
    MRC_BASED = "mrc_based"             # 边际风险贡献
    CVAR_DECOMPOSITION = "cvar_decomp"  # CVaR分解


class BudgetAdjustmentTrigger(Enum):
    """预算调整触发条件"""
    PERFORMANCE_DEGRADED = "perf_degraded"     # 绩效恶化
    VOLATILITY_SPIKE = "vol_spike"              # 波动率飙升
    DRAWDOWN_EXCEEDED = "drawdown"              # 回撤超限
    CORRELATION_SURGE = "correlation_surge"     # 相关性激增
    MARKET_REGIME_CHANGE = "regime_change"      # 市场状态变化
    TIME_BASED = "time_based"                   # 定时再平衡
    MANUAL = "manual"                           # 手动触发


@dataclass
class RiskBudget:
    """单策略风险预算"""
    strategy_name: str
    budget_pct: float = 0.0               # 风险预算比例（占总风险预算%）
    budget_amount: float = 0.0            # 风险预算金额（USDT）
    consumed: float = 0.0                 # 已消耗风险（当前敞口VaR）
    remaining: float = 0.0               # 剩余风险预算
    utilization_pct: float = 0.0          # 利用率

    # 风险分解
    var_95: float = 0.0                   # 95% VaR
    var_99: float = 0.0                   # 99% VaR
    cvar_95: float = 0.0                  # 95% CVaR
    ann_volatility: float = 0.0          # 年化波动率

    # 边际/成分风险贡献
    mrc: float = 0.0                      # 边际风险贡献 ∂σ/∂w
    crc: float = 0.0                      # 成分风险贡献 w * MRC
    rc_pct: float = 0.0                   # 风险贡献占比

    # 风险调整绩效
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    ro_mad: float = 0.0                  # Return over Max Adverse Deviation

    # 状态
    is_warning: bool = False
    is_critical: bool = False
    warning_reason: str = ""

    @property
    def is_exhausted(self) -> bool:
        return self.utilization_pct >= 0.95

    @property
    def is_approaching_limit(self) -> bool:
        return 0.70 <= self.utilization_pct < 0.95


@dataclass
class RiskBudgetPlan:
    """风险预算分配方案"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_equity: float = 0.0
    total_risk_budget_pct: float = 0.0    # 总风险预算比例（如3%）
    total_risk_budget: float = 0.0        # 总风险预算金额

    # 策略级风险预算
    strategy_budgets: Dict[str, RiskBudget] = field(default_factory=dict)

    # 风险分解汇总
    total_var_95: float = 0.0
    total_var_99: float = 0.0
    total_cvar_95: float = 0.0
    diversification_ratio: float = 0.0    # 分散化比率（sum(σi) / σ_portfolio）

    # 预算调整
    adjustments: Dict[str, float] = field(default_factory=dict)
    adjustment_triggers: List[str] = field(default_factory=list)

    # 利用率汇总
    overall_utilization: float = 0.0
    exhausted_strategies: List[str] = field(default_factory=list)
    approaching_limit_strategies: List[str] = field(default_factory=list)

    # 风险调整绩效排名
    rap_ranking: List[Tuple[str, float]] = field(default_factory=list)  # [(name, score)]

    # 建议
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_equity": self.total_equity,
            "total_risk_budget_pct": self.total_risk_budget_pct,
            "total_risk_budget": round(self.total_risk_budget, 2),
            "strategy_budgets": {
                k: {
                    "budget_pct": round(v.budget_pct, 4),
                    "budget_amount": round(v.budget_amount, 2),
                    "consumed": round(v.consumed, 2),
                    "remaining": round(v.remaining, 2),
                    "utilization_pct": round(v.utilization_pct, 4),
                    "var_95": round(v.var_95, 2),
                    "cvar_95": round(v.cvar_95, 2),
                    "mrc": round(v.mrc, 6),
                    "crc": round(v.crc, 4),
                    "rc_pct": round(v.rc_pct, 4),
                    "sharpe": round(v.sharpe, 4),
                    "sortino": round(v.sortino, 4),
                    "calmar": round(v.calmar, 4),
                    "is_warning": v.is_warning,
                    "is_critical": v.is_critical,
                    "warning_reason": v.warning_reason,
                } for k, v in self.strategy_budgets.items()
            },
            "diversification_ratio": round(self.diversification_ratio, 4),
            "overall_utilization": round(self.overall_utilization, 4),
            "exhausted_strategies": self.exhausted_strategies,
            "approaching_limit_strategies": self.approaching_limit_strategies,
            "adjustments": self.adjustments,
            "adjustment_triggers": self.adjustment_triggers,
            "rap_ranking": [(n, round(s, 4)) for n, s in self.rap_ranking],
            "warnings": self.warnings,
            "recommendations": self.recommendations,
        }


@dataclass
class SymbolRiskBudget:
    """币种级风险预算"""
    symbol: str
    total_var_95: float = 0.0               # 该币种总VaR95
    total_var_99: float = 0.0               # 该币种总VaR99
    risk_pct_of_portfolio: float = 0.0      # 占总组合风险%
    contributing_strategies: List[str] = field(default_factory=list)
    is_over_concentration: bool = False     # 是否超过集中度限制
    concentration_limit: float = 0.008      # 集中度上限（默认0.8%）
    warning_level: str = "normal"           # normal / warning / critical


@dataclass
class RiskBudgetDrift:
    """风险预算漂移追踪"""
    strategy_name: str
    target_budget_pct: float = 0.0          # 目标预算比例
    actual_risk_pct: float = 0.0            # 实际风险占比
    drift_pct: float = 0.0                  # 漂移量（actual - target）
    drift_direction: str = "stable"         # expanding / contracting / stable
    drift_trend: List[float] = field(default_factory=list)  # 最近N次漂移值
    alert: bool = False
    alert_reason: str = ""


@dataclass
class VaRBacktestResult:
    """VaR回测结果 — 检验VaR预测准确性"""
    total_observations: int = 0
    var_95_violations: int = 0              # 95%VaR违规次数
    var_99_violations: int = 0              # 99%VaR违规次数
    expected_95_violations: float = 0       # 期望违规次数（5%）
    expected_99_violations: float = 0       # 期望违规次数（1%）
    kupiec_pvalue_95: float = 1.0           # Kupiec检验p值（>0.05通过）
    kupiec_pvalue_99: float = 1.0
    christoffersen_pvalue_95: float = 1.0   # 条件覆盖检验
    status: str = "pass"                    # pass / warning / fail
    avg_violation_magnitude: float = 0.0    # 平均违规幅度
    max_violation: float = 0.0              # 最大违规


@dataclass
class RiskBudgetSnapshot:
    """风险预算历史快照"""
    timestamp: str = ""
    total_equity: float = 0.0
    total_budget: float = 0.0
    total_consumed: float = 0.0
    overall_utilization: float = 0.0
    strategy_utilizations: Dict[str, float] = field(default_factory=dict)
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    diversification_ratio: float = 0.0
    rap_scores: Dict[str, float] = field(default_factory=dict)
    active_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_equity": self.total_equity,
            "total_budget": round(self.total_budget, 2),
            "total_consumed": round(self.total_consumed, 2),
            "overall_utilization": round(self.overall_utilization, 4),
            "strategy_utilizations": {k: round(v, 4) for k, v in self.strategy_utilizations.items()},
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "cvar_95": round(self.cvar_95, 2),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "rap_scores": {k: round(v, 4) for k, v in self.rap_scores.items()},
            "active_warnings": self.active_warnings,
        }


# ═══════════════════════════════════════════════════════════════
# 风险预算引擎
# ═══════════════════════════════════════════════════════════════

class RiskBudgetEngine:
    """
    风险预算分配引擎

    核心功能：
      1. 风险预算定义（组合→策略→仓位 三级分解）
      2. 前向风险估计（EWMA波动率 + CVaR预测）
      3. 风险平价分配（数学优化求解）
      4. 风险分解（VaR分解 + 边际/成分风险贡献）
      5. 风险利用率追踪
      6. 动态预算调整（市场状态感知 + 绩效驱动）
      7. 风险调整绩效指标（RAPM）
      8. 风险预算瀑布分配
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()

        # ── 风险预算配置 ──
        rb_cfg = config.get("risk_budget", {})
        self._enabled = rb_cfg.get("enabled", True)
        self._daily_risk_budget_pct = rb_cfg.get("daily_risk_budget_pct", 0.03)
        self._hourly_max_loss = rb_cfg.get("hourly_max_loss_pct", 0.015)
        self._max_per_trade_risk = rb_cfg.get("max_per_trade_risk_pct", 0.008)

        # 策略风险预算分配
        self._strategy_budget_pcts: Dict[str, float] = rb_cfg.get("strategy_budgets", {
            "scalping": 0.40, "trend": 0.25, "grid": 0.20, "arbitrage": 0.15,
        })

        # 再分配配置
        realloc_cfg = rb_cfg.get("reallocation", {})
        self._realloc_enabled = realloc_cfg.get("enabled", True)
        self._realloc_interval = realloc_cfg.get("interval_minutes", 60)
        self._max_shift_ratio = realloc_cfg.get("max_shift_ratio", 0.15)
        self._transfer_winrate_threshold = realloc_cfg.get("transfer_out_winrate_threshold", 0.35)
        self._transfer_drawdown_threshold = realloc_cfg.get("transfer_out_drawdown_threshold", 0.08)

        # 集中度限制
        conc_cfg = rb_cfg.get("concentration", {})
        self._max_symbol_risk = conc_cfg.get("max_single_symbol_risk_pct", 0.008)
        self._max_correlated_risk = conc_cfg.get("max_correlated_group_risk_pct", 0.02)

        # 连续亏损惩罚
        streak_cfg = rb_cfg.get("loss_streak", {})
        self._max_consecutive_losses = streak_cfg.get("max_consecutive_losses", 5)
        self._loss_reduce_pct = streak_cfg.get("loss_streak_reduce_pct", 0.50)
        self._recovery_wins = streak_cfg.get("recovery_consecutive_wins", 3)

        # ── 风险平价参数 ──
        self._risk_parity_lambda = 0.5          # 风险厌恶系数
        self._ewma_decay = 0.94                  # EWMA衰减因子
        self._var_confidence_levels = [0.95, 0.99]

        # ── 状态 ──
        self._last_plan: Optional[RiskBudgetPlan] = None
        self._last_rebalance: Optional[datetime] = None
        self._strategy_returns: Dict[str, List[float]] = {}
        self._strategy_var_estimates: Dict[str, float] = {}
        self._daily_pnl_map: Dict[str, List[float]] = {}
        self._consumed_budget: Dict[str, float] = {}

        # ── 增强：历史快照 ──
        self._snapshots: deque = deque(maxlen=100)  # 最多100个历史快照
        self._snapshot_interval_seconds = 300         # 每5分钟自动快照
        self._last_snapshot_time: Optional[datetime] = None
        self._data_dir = rb_cfg.get("data_dir", "./data")

        # ── 增强：VaR回测 ──
        self._var_predictions: Dict[str, List[float]] = {}  # {name: [predicted_var]}
        self._actual_losses: Dict[str, List[float]] = {}    # {name: [actual_loss]}
        self._var_backtest_window = 250                      # 回测窗口

        # ── 增强：币种级风险集中度 ──
        self._symbol_risk_map: Dict[str, SymbolRiskBudget] = {}
        self._symbol_strategy_map: Dict[str, List[str]] = {}  # symbol → [strategies]

        # ── 增强：风险预算漂移追踪 ──
        self._budget_drift: Dict[str, RiskBudgetDrift] = {}
        self._drift_history_len = 20

        # ── 增强：杠杆帽 ──
        self._strategy_leverage_caps: Dict[str, float] = {}

        # ── 增强：相关性阈值 ──
        self._high_correlation_threshold = rb_cfg.get("high_correlation_threshold", 0.70)
        self._extreme_correlation_threshold = rb_cfg.get("extreme_correlation_threshold", 0.85)

        # ── 外部依赖 ──
        self._portfolio_optimizer = None
        self._position_sizer = None
        self._account_manager = None   # 用于获取总权益

        logger.info(
            f"RiskBudgetEngine initialized: daily_budget={self._daily_risk_budget_pct:.1%}, "
            f"strategies={len(self._strategy_budget_pcts)}, "
            f"realloc={'enabled' if self._realloc_enabled else 'disabled'}"
        )

    # ── 依赖注入 ─────────────────────────────────────────────

    def set_portfolio_optimizer(self, optimizer) -> None:
        self._portfolio_optimizer = optimizer

    def set_position_sizer(self, sizer) -> None:
        self._position_sizer = sizer

    def set_account_manager(self, account_manager) -> None:
        """注入账户管理器，用于获取总权益和持仓信息"""
        self._account_manager = account_manager

    def set_correlation_analyzer(self, correlation_analyzer) -> None:
        """注入策略相关性分析器，用于相关性感知的风险预算调整"""
        self._correlation_analyzer = correlation_analyzer

    def set_diversification_optimizer(self, diversification_optimizer) -> None:
        """注入分散化优化器，用于最优权重分配"""
        self._diversification_optimizer = diversification_optimizer

    def register_symbol_strategy(self, symbol: str, strategy_name: str):
        """注册币种与策略的关联关系（用于集中度监控）"""
        if symbol not in self._symbol_strategy_map:
            self._symbol_strategy_map[symbol] = []
        if strategy_name not in self._symbol_strategy_map[symbol]:
            self._symbol_strategy_map[symbol].append(strategy_name)

    # ── 数据更新 ─────────────────────────────────────────────

    def update_strategy_returns(self, name: str, returns: List[float]):
        """更新策略收益率序列"""
        self._strategy_returns[name] = returns[-252:]  # 保留最多252天

    def update_daily_pnl(self, name: str, pnl: float):
        """更新策略日盈亏，同时用于VaR回测"""
        if name not in self._daily_pnl_map:
            self._daily_pnl_map[name] = []
        self._daily_pnl_map[name].append(pnl)
        if len(self._daily_pnl_map[name]) > 252:
            self._daily_pnl_map[name] = self._daily_pnl_map[name][-252:]

        # 同时更新VaR回测实际损失
        if name not in self._actual_losses:
            self._actual_losses[name] = []
        # 损失为正数存储
        self._actual_losses[name].append(max(0, -pnl))
        if len(self._actual_losses[name]) > self._var_backtest_window:
            self._actual_losses[name] = self._actual_losses[name][-self._var_backtest_window:]

    def update_var_prediction(self, name: str, var_predicted: float):
        """更新VaR预测值（用于回测）"""
        if name not in self._var_predictions:
            self._var_predictions[name] = []
        self._var_predictions[name].append(var_predicted)
        if len(self._var_predictions[name]) > self._var_backtest_window:
            self._var_predictions[name] = self._var_predictions[name][-self._var_backtest_window:]

    def update_consumed_budget(self, name: str, var_consumed: float):
        """更新策略已消耗的风险预算"""
        self._consumed_budget[name] = var_consumed

    # ═══════════════════════════════════════════════════════════
    # 核心：完整风险预算方案计算
    # ═══════════════════════════════════════════════════════════

    async def compute_risk_budget_plan(
        self,
        total_equity: float,
        strategy_names: List[str],
        strategy_metrics: Dict[str, Dict[str, float]] = None,
        current_risk_consumed: Dict[str, float] = None,
        market_regime: str = "unknown",
        covariance_matrix: np.ndarray = None,
    ) -> RiskBudgetPlan:
        """
        计算完整风险预算分配方案

        Args:
            total_equity: 总权益
            strategy_names: 活跃策略名称
            strategy_metrics: {name: {vol, sharpe, drawdown, ...}}
            current_risk_consumed: {name: var_consumed} 当前已消耗风险
            market_regime: 市场状态
            covariance_matrix: 策略协方差矩阵（可选，自动计算）

        Returns:
            完整风险预算方案
        """
        async with self._lock:
            if total_equity <= 0:
                return RiskBudgetPlan(total_equity=total_equity)

            strategy_metrics = strategy_metrics or {}
            current_risk_consumed = current_risk_consumed or {}

            plan = RiskBudgetPlan(
                total_equity=total_equity,
                total_risk_budget_pct=self._daily_risk_budget_pct,
                total_risk_budget=total_equity * self._daily_risk_budget_pct,
            )

            # ── Step 1: 计算前向风险估计（EWMA波动率）──
            forward_vols = self._estimate_forward_volatility(
                strategy_names, strategy_metrics
            )

            # ── Step 2: 计算风险平价权重 ──
            rp_weights = self._compute_risk_parity_weights(
                strategy_names, forward_vols, covariance_matrix
            )

            # ── Step 3: 混合目标权重（风险平价 + 初始配置）──
            target_budget_pcts = self._blend_budget_weights(
                strategy_names, rp_weights, market_regime
            )

            # ── Step 4: 风险分解（VaR分解、MRC/CRC）──
            self._decompose_risk(plan, strategy_names, forward_vols,
                                 target_budget_pcts, covariance_matrix,
                                 current_risk_consumed)

            # ── Step 5: 风险调整绩效指标 RAPM ──
            self._compute_rapm(plan, strategy_metrics)

            # ── Step 6: 动态预算调整（绩效驱动 + 市场状态）──
            self._adjust_budgets(plan, strategy_names, strategy_metrics,
                                market_regime, current_risk_consumed)

            # ── Step 7: 利用率汇总 ──
            self._summarize_utilization(plan, current_risk_consumed)

            # ── Step 8: 生成风险预算建议 ──
            self._generate_recommendations(plan)

            # ── Step 9: RAP排名 ──
            self._rank_by_rap(plan)

            # ── 增强 Step 10: VaR回测（Kupiec检验）──
            plan._var_backtest = self._var_backtest(strategy_names)

            # ── 增强 Step 11: 币种级集中度监控 ──
            plan._symbol_concentration = self._compute_symbol_concentration(
                strategy_names, plan
            )

            # ── 增强 Step 12: 动态杠杆帽推导 ──
            self._derive_leverage_caps(strategy_names, plan)

            # ── 增强 Step 13: 风险预算漂移追踪 ──
            self._track_budget_drift(strategy_names, plan)

            # ── 增强 Step 14: 预算使用趋势预测 ──
            plan._budget_trend = self._predict_budget_trend(plan)

            # ── 增强 Step 15: 自动保存历史快照 ──
            self._auto_snapshot(plan)

            self._last_plan = plan
            return plan

    # ── Step 1: 前向风险估计（EWMA波动率）──────────────────

    def _estimate_forward_volatility(
        self,
        strategy_names: List[str],
        metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        使用 EWMA 模型预测前向波动率
        σ²_t = λ * σ²_{t-1} + (1-λ) * r²_t

        Returns:
            {name: annualized_volatility}
        """
        forward_vols: Dict[str, float] = {}

        for name in strategy_names:
            # 优先使用实时收益率序列
            returns = self._strategy_returns.get(name, [])
            if len(returns) >= 20:
                # EWMA 波动率
                ewma_var = 0.0
                weights_sum = 0.0
                for i, r in enumerate(reversed(returns[-60:])):
                    w = self._ewma_decay ** i
                    ewma_var += w * (r ** 2)
                    weights_sum += w
                if weights_sum > 0:
                    ann_vol = math.sqrt(ewma_var / weights_sum) * math.sqrt(365)
                    forward_vols[name] = max(0.001, ann_vol)
                    continue

            # 回退：使用策略指标中的波动率
            m = metrics.get(name, {})
            ann_vol = m.get("ann_volatility", m.get("volatility_30d", 0.02))
            forward_vols[name] = max(0.001, ann_vol)

            # 缓存用于后续计算
            self._strategy_var_estimates[name] = forward_vols[name]

        return forward_vols

    # ── Step 2: 风险平价权重计算 ────────────────────────────

    def _compute_risk_parity_weights(
        self,
        strategy_names: List[str],
        forward_vols: Dict[str, float],
        covariance_matrix: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        风险平价：各策略对组合的风险贡献相等

        简化公式（无协方差时）：
          w_i ∝ 1/σ_i  →  w_i = (1/σ_i) / Σ(1/σ_j)

        优化求解（有协方差时）：
          min Σ( RC_i - RC_target )²
          s.t. Σw = 1, w ≥ 0

        Returns:
            {name: risk_parity_weight}
        """
        n = len(strategy_names)
        if n == 0:
            return {}

        if covariance_matrix is None or covariance_matrix.size == 0 or covariance_matrix.shape[0] != n:
            # 简化：权重 ∝ 1/σ
            inv_vols = {name: 1.0 / max(forward_vols.get(name, 0.02), 0.001)
                       for name in strategy_names}
            total_inv = sum(inv_vols.values())
            if total_inv > 0:
                return {name: v / total_inv for name, v in inv_vols.items()}
            # 等权回退
            return {name: 1.0 / n for name in strategy_names}

        # 有协方差矩阵时使用优化求解
        try:
            from scipy.optimize import minimize

            vol_arr = np.array([forward_vols.get(n, 0.02) for n in strategy_names])

            def risk_parity_objective(w):
                port_vol = math.sqrt(w @ covariance_matrix @ w)
                if port_vol <= 0:
                    return 1e10
                mrc = covariance_matrix @ w / port_vol
                rc = w * mrc
                target_rc = port_vol / n
                return np.sum((rc - target_rc) ** 2)

            # 初始权重：1/σ
            init_w = np.array([1.0 / max(v, 0.001) for v in vol_arr])
            init_w /= init_w.sum()

            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0.01, 0.50) for _ in range(n)]

            result = minimize(risk_parity_objective, init_w, method='SLSQP',
                            constraints=constraints, bounds=bounds,
                            options={'maxiter': 200, 'ftol': 1e-8})

            if result.success:
                return {name: float(w) for name, w in zip(strategy_names, result.x)}
        except Exception as e:
            logger.warning(f"Risk parity optimization failed: {e}")

        # 回退
        inv_vols = {name: 1.0 / max(forward_vols.get(name, 0.02), 0.001)
                   for name in strategy_names}
        total_inv = sum(inv_vols.values())
        return {name: v / total_inv for name, v in inv_vols.items()}

    # ── Step 3: 混合预算权重 ─────────────────────────────────

    def _blend_budget_weights(
        self,
        strategy_names: List[str],
        rp_weights: Dict[str, float],
        market_regime: str,
    ) -> Dict[str, float]:
        """
        混合风险平价权重和初始配置权重
        blend = α * rp_weight + (1-α) * initial_budget_pct
        """
        # 市场状态决定混合比率
        alpha_map = {
            "trending_up":      0.7,   # 趋势上涨：更依赖风险平价
            "trending_down":    0.3,   # 趋势下跌：更依赖初始配置（保守）
            "ranging":          0.5,   # 震荡：各半
            "high_volatility":  0.3,   # 高波动：保守
            "low_volatility":   0.7,   # 低波动：激进
        }
        alpha = alpha_map.get(market_regime, 0.5)

        blended: Dict[str, float] = {}
        for name in strategy_names:
            initial = self._strategy_budget_pcts.get(name, 0.15)
            rp = rp_weights.get(name, 0)
            blended[name] = alpha * rp + (1 - alpha) * initial

        # 归一化
        total = sum(blended.values())
        if total > 0:
            return {k: v / total for k, v in blended.items()}
        return blended

    # ── Step 4: 风险分解 ─────────────────────────────────────

    def _decompose_risk(
        self,
        plan: RiskBudgetPlan,
        strategy_names: List[str],
        forward_vols: Dict[str, float],
        target_budget_pcts: Dict[str, float],
        covariance_matrix: np.ndarray,
        current_consumed: Dict[str, float],
    ):
        """
        风险分解：VaR分解 + 边际风险贡献 + 成分风险贡献
        """
        n = len(strategy_names)
        weights = np.array([target_budget_pcts.get(name, 0) for name in strategy_names])

        # 计算组合波动率
        if covariance_matrix is not None and covariance_matrix.size > 0 and covariance_matrix.shape[0] == n:
            port_var = weights @ covariance_matrix @ weights
            port_vol = math.sqrt(max(port_var, 1e-12))
        else:
            # 无协方差：简化为独立假设
            diag_var = sum((w * forward_vols.get(name, 0.02)) ** 2
                         for name, w in zip(strategy_names, weights))
            port_vol = math.sqrt(max(diag_var, 1e-12))

        # 分散化比率
        sum_vols = sum(forward_vols.get(name, 0.02) * w
                      for name, w in zip(strategy_names, weights))
        plan.diversification_ratio = sum_vols / max(port_vol, 1e-12)

        # 逐策略分解
        for i, name in enumerate(strategy_names):
            w = weights[i]
            vol = forward_vols.get(name, 0.02)
            budget_pct = target_budget_pcts.get(name, 0)

            position_value = plan.total_equity * budget_pct

            # ── P32 企业级强化：厚尾风险修正 ──
            # 优先使用 Cornish-Fisher VaR（偏度/峰度修正）+ 蒙特卡洛 ES，
            # 替代简单正态 VaR，捕捉加密货币收益率的尖峰厚尾特征。
            # 只有样本不足时才回退到正态近似。
            returns = self._strategy_returns.get(name, [])
            if len(returns) >= 20:
                # Cornish-Fisher 返回日收益率形式的 VaR（小数），乘以仓位金额得到绝对额
                cf_var_95 = self._cornish_fisher_var(returns, confidence=0.95)
                cf_var_99 = self._cornish_fisher_var(returns, confidence=0.99)
                mc_es_95 = self._monte_carlo_es(returns, confidence=0.95)
                var_95 = position_value * cf_var_95
                var_99 = position_value * cf_var_99
                cvar_95 = position_value * mc_es_95
            else:
                # 回退：正态分布假设
                # VaR_95 = z_0.95 * σ * √T
                z_95 = 1.645   # 标准正态95%分位数
                z_99 = 2.326   # 99%分位数
                daily_vol = vol / math.sqrt(365)
                var_95 = position_value * daily_vol * z_95
                var_99 = position_value * daily_vol * z_99
                cvar_95 = position_value * daily_vol * 2.063  # 正态CVaR95近似

            # 边际风险贡献：∂σ/∂w_i = (cov @ w)_i / σ
            if covariance_matrix is not None and covariance_matrix.size > 0 and covariance_matrix.shape[0] == n:
                mrc = float((covariance_matrix @ weights)[i] / max(port_vol, 1e-12))
            else:
                mrc = float((w * vol ** 2) / max(port_vol, 1e-12))

            # 成分风险贡献：CRC_i = w_i * MRC_i
            crc = float(w * mrc)
            rc_pct = crc / max(port_vol, 1e-12) if port_vol > 0 else 0

            # 已消耗风险
            consumed = current_consumed.get(name, 0)

            # 该策略的风险预算金额
            budget_amount = plan.total_risk_budget * budget_pct

            utilization = consumed / max(budget_amount, 1) if budget_amount > 0 else 0

            rb = RiskBudget(
                strategy_name=name,
                budget_pct=budget_pct,
                budget_amount=budget_amount,
                consumed=consumed,
                remaining=max(0, budget_amount - consumed),
                utilization_pct=utilization,
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                ann_volatility=vol,
                mrc=mrc,
                crc=crc,
                rc_pct=rc_pct,
            )
            plan.strategy_budgets[name] = rb

            # 汇总
            plan.total_var_95 += var_95
            plan.total_var_99 += var_99
            plan.total_cvar_95 += cvar_95

    # ── Step 5: 风险调整绩效指标 (RAPM) ───────────────────

    def _compute_rapm(
        self,
        plan: RiskBudgetPlan,
        strategy_metrics: Dict[str, Dict[str, float]],
    ):
        """计算风险调整绩效指标"""
        for name, rb in plan.strategy_budgets.items():
            m = strategy_metrics.get(name, {})
            returns = self._strategy_returns.get(name, [])

            if returns and len(returns) >= 10:
                ret_arr = np.array(returns[-90:])
                mean_daily = float(np.mean(ret_arr))
                std_daily = float(np.std(ret_arr, ddof=1))

                # Sharpe
                if std_daily > 0:
                    rb.sharpe = float(mean_daily / std_daily * math.sqrt(365))

                # Sortino
                downside = ret_arr[ret_arr < 0]
                if len(downside) > 0:
                    downside_std = float(np.std(downside, ddof=1))
                    if downside_std > 0:
                        rb.sortino = float(mean_daily / downside_std * math.sqrt(365))

            # Calmar：年化收益 / 最大回撤
            md = m.get("max_drawdown", 0.01)
            ann_ret = m.get("annualized_return", m.get("expected_return", 0))
            if md > 0:
                rb.calmar = float(ann_ret / md)

            # RoMAD: Return over Max Adverse Deviation
            rb.ro_mad = rb.sortino if rb.sortino > 0 else max(0, rb.calmar * 0.5)

    # ── Step 6: 动态预算调整 ─────────────────────────────────

    def _adjust_budgets(
        self,
        plan: RiskBudgetPlan,
        strategy_names: List[str],
        metrics: Dict[str, Dict[str, float]],
        market_regime: str,
        current_consumed: Dict[str, float],
    ):
        """
        动态调整风险预算：
          - 绩效恶化 → 削减预算
          - 高Sharpe → 增加预算
          - 高波动市场 → 全局收缩
          - 连续亏损 → 惩罚性削减
        """
        adjustments: Dict[str, float] = {}

        for name in strategy_names:
            rb = plan.strategy_budgets.get(name)
            if not rb:
                continue

            m = metrics.get(name, {})

            # ── 触发1: 胜率过低 ──
            if m.get("win_rate", 0.5) < self._transfer_winrate_threshold:
                reduction = -self._max_shift_ratio * 0.5
                adjustments[name] = reduction
                plan.adjustment_triggers.append(
                    f"{name}: low win_rate ({m['win_rate']:.1%}) → reduce {abs(reduction):.1%}"
                )

            # ── 触发2: 回撤超限 ──
            if m.get("max_drawdown", 0) > self._transfer_drawdown_threshold:
                reduction = -self._max_shift_ratio * 0.7
                adjustments[name] = adjustments.get(name, 0) + reduction
                plan.adjustment_triggers.append(
                    f"{name}: drawdown ({m['max_drawdown']:.1%}) → reduce {abs(reduction):.1%}"
                )

            # ── 触发3: 连续亏损 ──
            consec_losses = m.get("consecutive_losses", 0)
            if consec_losses >= self._max_consecutive_losses:
                reduction = -self._loss_reduce_pct
                adjustments[name] = max(adjustments.get(name, 0), reduction)
                plan.adjustment_triggers.append(
                    f"{name}: {consec_losses} consecutive losses → cut budget {self._loss_reduce_pct:.0%}"
                )

            # ── 触发4: 高Sharpe → 增加预算 ──
            sharpe = m.get("sharpe_ratio", 0)
            if sharpe > 2.0:
                increase = min(self._max_shift_ratio, (sharpe - 1.5) * 0.05)
                adjustments[name] = adjustments.get(name, 0) + increase
                plan.adjustment_triggers.append(
                    f"{name}: high sharpe ({sharpe:.2f}) → increase +{increase:.1%}"
                )

            # ── 触发5: 预算即将耗尽 → 预警 ──
            utilization = current_consumed.get(name, 0) / max(rb.budget_amount, 1)
            if utilization >= 0.90:
                rb.is_critical = True
                rb.warning_reason = "Risk budget > 90% consumed"
            elif utilization >= 0.70:
                rb.is_warning = True
                rb.warning_reason = "Risk budget > 70% consumed"

        # ── 全局调整：高波动市场整体收缩 ──
        if market_regime == "high_volatility":
            global_shrink = -0.20  # 整体收缩20%
            for name in strategy_names:
                if name not in adjustments:
                    adjustments[name] = global_shrink
                else:
                    adjustments[name] += global_shrink
            plan.adjustment_triggers.append(
                f"GLOBAL: high_volatility regime → shrink budgets 20%"
            )

        # ── 触发6: 相关性激增 ──
        if hasattr(self, '_correlation_analyzer') and self._correlation_analyzer:
            try:
                corr_results = self._analyze_correlation_breach(strategy_names)
                for pair, corr_info in corr_results.items():
                    s1, s2 = pair.split("_", 1)
                    if corr_info["breach"]:
                        # 降低两个高相关策略的预算
                        for name in [s1, s2]:
                            reduction = -self._max_shift_ratio * 0.3
                            adjustments[name] = adjustments.get(name, 0) + reduction
                        plan.adjustment_triggers.append(
                            f"Correlation surge: {s1}-{s2} corr={corr_info['correlation']:.2f} → reduce budgets"
                        )
            except Exception as e:
                logger.debug(f"Correlation-based budget adjustment error: {e}")

        plan.adjustments = adjustments

    def _analyze_correlation_breach(self, strategy_names: List[str]) -> Dict[str, Any]:
        """分析策略相关性是否突破阈值，返回高相关策略对"""
        result = {}
        if not hasattr(self, '_correlation_analyzer') or not self._correlation_analyzer:
            return result

        try:
            # 从相关性分析器获取当前相关性状态
            corr_data = self._correlation_analyzer.get_summary() if hasattr(
                self._correlation_analyzer, 'get_summary'
            ) else {}

            # 遍历策略对
            for i, s1 in enumerate(strategy_names):
                for j, s2 in enumerate(strategy_names[i + 1:]):
                    pair_key = f"{s1}_{s2}"
                    pair_key_alt = f"{s2}_{s1}"

                    # 从相关性数据中查找
                    corr_info = corr_data.get(pair_key) or corr_data.get(pair_key_alt) or {}
                    correlations = corr_info.get("correlations", {})

                    pearson = correlations.get("pearson", 0)
                    spearman = correlations.get("spearman", 0)
                    ewma_corr = correlations.get("ewma_correlation", 0)

                    # 使用最高的相关度量来判断
                    max_corr = max(abs(pearson), abs(spearman), abs(ewma_corr))

                    breach = max_corr >= self._high_correlation_threshold
                    extreme = max_corr >= self._extreme_correlation_threshold

                    result[pair_key] = {
                        "strategy_pair": [s1, s2],
                        "correlation": round(max_corr, 4),
                        "pearson": round(pearson, 4),
                        "spearman": round(spearman, 4),
                        "ewma_correlation": round(ewma_corr, 4),
                        "breach": breach,
                        "extreme": extreme,
                    }
        except Exception as e:
            logger.debug(f"Correlation breach analysis error: {e}")

        return result

    # ── Step 7: 利用率汇总 ───────────────────────────────────

    def _summarize_utilization(
        self,
        plan: RiskBudgetPlan,
        current_consumed: Dict[str, float],
    ):
        """汇总风险利用率"""
        total_budget = plan.total_risk_budget
        total_consumed = sum(current_consumed.values())
        plan.overall_utilization = total_consumed / max(total_budget, 1)

        for name, rb in plan.strategy_budgets.items():
            if rb.is_exhausted:
                plan.exhausted_strategies.append(name)
            elif rb.is_approaching_limit:
                plan.approaching_limit_strategies.append(name)

        if plan.overall_utilization >= 0.90:
            plan.warnings.append(
                f"Overall risk budget nearing exhaustion: {plan.overall_utilization:.1%}"
            )

    # ── Step 8: 生成建议 ─────────────────────────────────────

    def _generate_recommendations(self, plan: RiskBudgetPlan):
        """生成风险预算调整建议"""
        # 利用率预警
        for name in plan.exhausted_strategies:
            plan.recommendations.append(
                f"CRITICAL: {name} risk budget exhausted — stop new positions"
            )
        for name in plan.approaching_limit_strategies:
            plan.recommendations.append(
                f"WARNING: {name} risk budget approaching limit ({plan.strategy_budgets[name].utilization_pct:.0%})"
            )

        # 集中度建议
        high_rc = sorted(
            [(name, rb.rc_pct) for name, rb in plan.strategy_budgets.items()],
            key=lambda x: x[1], reverse=True
        )
        if high_rc and high_rc[0][1] > 0.50:
            plan.recommendations.append(
                f"High risk concentration: {high_rc[0][0]} contributes {high_rc[0][1]:.1%} of total risk"
            )

        # 分散化建议
        if plan.diversification_ratio < 1.2:
            plan.recommendations.append(
                f"Low diversification ({plan.diversification_ratio:.2f}); consider uncorrelated strategies"
            )

        # 预算调整建议
        for adj_str in plan.adjustment_triggers:
            if "reduce" in adj_str.lower() or "cut" in adj_str.lower():
                plan.recommendations.append(f"Budget adjustment: {adj_str}")

        # 超额风险
        if plan.total_var_99 > plan.total_risk_budget:
            plan.recommendations.append(
                f"VaR(99%) exceeds total risk budget: {plan.total_var_99:.0f} > {plan.total_risk_budget:.0f}"
            )

        # 相关性风险建议
        if hasattr(self, '_correlation_analyzer') and self._correlation_analyzer:
            try:
                strategy_names = list(plan.strategy_budgets.keys())
                corr_breaches = self._analyze_correlation_breach(strategy_names)
                for pair_key, info in corr_breaches.items():
                    if info.get("extreme"):
                        s1, s2 = info["strategy_pair"]
                        plan.recommendations.append(
                            f"CRITICAL: Extreme correlation {s1}-{s2} ({info['correlation']:.2f}) — "
                            f"consider reducing one position"
                        )
                    elif info.get("breach"):
                        s1, s2 = info["strategy_pair"]
                        plan.warnings.append(
                            f"High correlation {s1}-{s2} ({info['correlation']:.2f}) — "
                            f"monitor for diversification erosion"
                        )
            except Exception as e:
                logger.debug(f"Correlation recommend error: {e}")

    # ── Step 9: RAP排名 ──────────────────────────────────────

    def _rank_by_rap(self, plan: RiskBudgetPlan):
        """按风险调整绩效排名"""
        scored = []
        for name, rb in plan.strategy_budgets.items():
            # 综合RAP评分 = Sharpe*0.4 + Sortino*0.3 + Calmar*0.2 + RoMAD*0.1
            score = (
                max(0, rb.sharpe) * 0.4 +
                max(0, rb.sortino) * 0.3 +
                max(0, rb.calmar) * 0.2 +
                max(0, rb.ro_mad) * 0.1
            )
            scored.append((name, score))

        plan.rap_ranking = sorted(scored, key=lambda x: x[1], reverse=True)

    # ═══════════════════════════════════════════════════════════
    # 增强1: Cornish-Fisher VaR + 蒙特卡洛ES
    # ═══════════════════════════════════════════════════════════

    def _cornish_fisher_var(
        self,
        returns: List[float],
        confidence: float = 0.95,
        holding_period: int = 1,
    ) -> float:
        """
        Cornish-Fisher修正VaR — 考虑偏度和峰度的修正VaR

        CF_VaR = μ + (z_α + (z_α²-1)*S/6 + (z_α³-3z_α)*K/24 - (2z_α³-5z_α)*S²/36) * σ

        比标准正态VaR更能捕捉厚尾风险
        """
        if len(returns) < 20:
            return 0.0

        ret_arr = np.array(returns[-252:])
        mu = float(np.mean(ret_arr))
        sigma = float(np.std(ret_arr, ddof=1))
        if sigma <= 0:
            return 0.0

        # 偏度 & 超额峰度
        n = len(ret_arr)
        skew = float(np.sum(((ret_arr - mu) / sigma) ** 3) * n / ((n - 1) * (n - 2))) if n > 2 else 0.0
        excess_kurt = float(
            (n * (n + 1) * np.sum(((ret_arr - mu) / sigma) ** 4) / ((n - 1) * (n - 2) * (n - 3)))
            - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        ) if n > 3 else 0.0

        # 左尾分位数（VaR 关注损失，用负 z；Cornish-Fisher 展开对左尾）
        # P32 修复：原实现用正 z 导致 daily_var=-(mu+z_cf*σ) 恒为负、max(0,·) 恒返回 0
        if confidence == 0.95:
            z = -1.645
        elif confidence == 0.99:
            z = -2.326
        else:
            # 默认左尾 95%
            z = -1.645

        # Cornish-Fisher展开（对左尾分位数）
        z_cf = z + (z ** 2 - 1) * skew / 6.0 + \
               (z ** 3 - 3 * z) * excess_kurt / 24.0 - \
               (2 * z ** 3 - 5 * z) * (skew ** 2) / 36.0

        # 日VaR转holding_period VaR；z_cf 为负，结果为正值（损失）
        daily_var = -(mu + z_cf * sigma)
        period_var = daily_var * math.sqrt(holding_period)

        return max(0.0, period_var)

    def _monte_carlo_es(
        self,
        returns: List[float],
        confidence: float = 0.975,
        n_simulations: int = 10000,
        holding_period: int = 1,
    ) -> float:
        """
        蒙特卡洛 Expected Shortfall (CVaR)

        通过拟合收益率分布 + 蒙特卡洛模拟来估计ES，
        比参数法更稳健地捕捉尾部风险。
        """
        if len(returns) < 20:
            return 0.0

        ret_arr = np.array(returns[-252:])
        mu = float(np.mean(ret_arr))
        sigma = float(np.std(ret_arr, ddof=1))
        if sigma <= 0:
            return 0.0

        # 拟合 t 分布（厚尾比正态更真实）
        try:
            # 简化：使用带自由度的t分布近似
            # 自由度从超额峰度估计: df ≈ 6/excess_kurt + 4
            n_obs = len(ret_arr)
            excess_kurt = float(
                np.sum(((ret_arr - mu) / sigma) ** 4) / n_obs - 3
            ) if n_obs > 3 else 0
            df = max(3, min(30, 6.0 / max(abs(excess_kurt), 0.01) + 4))
        except Exception:
            df = 5  # 默认5自由度t分布

        # 蒙特卡洛模拟
        np.random.seed(int(time.time() * 1000) % (2 ** 31 - 1))
        sim_returns = np.random.standard_t(df, n_simulations) * sigma + mu

        # 排序取尾部
        sim_returns.sort()
        cutoff_idx = int(n_simulations * (1 - confidence))
        tail_returns = sim_returns[:cutoff_idx]

        if len(tail_returns) == 0:
            return 0.0

        es = -float(np.mean(tail_returns)) * math.sqrt(holding_period)
        return max(0.0, es)

    def _var_backtest(self, strategy_names: List[str]) -> VaRBacktestResult:
        """
        VaR回测：Kupiec检验 + Christoffersen检验

        Kupiec检验：检验VaR违规率是否与置信水平一致
        Christoffersen检验：检验违规是否独立（无聚集效应）
        """
        all_predictions_95 = []
        all_actuals = []

        for name in strategy_names:
            preds = self._var_predictions.get(name, [])
            actuals = self._actual_losses.get(name, [])
            min_len = min(len(preds), len(actuals))
            if min_len > 0:
                all_predictions_95.extend(preds[-min_len:])
                all_actuals.extend(actuals[-min_len:])

        result = VaRBacktestResult()
        result.total_observations = len(all_actuals)

        if result.total_observations == 0:
            return result

        # 统计违规
        for pred, actual in zip(all_predictions_95, all_actuals):
            if actual > max(pred, 0.01):
                result.var_95_violations += 1
                violation_mag = actual - pred
                result.avg_violation_magnitude += violation_mag
                result.max_violation = max(result.max_violation, violation_mag)

        n = result.total_observations
        if n > 0 and result.var_95_violations > 0:
            result.avg_violation_magnitude /= result.var_95_violations

        # Kupiec检验 (95%)
        result.expected_95_violations = n * 0.05
        result.expected_99_violations = n * 0.01

        x_95 = result.var_95_violations
        p_expected = 0.05
        if n > 0 and x_95 > 0:
            p_actual = x_95 / n
            if 0 < p_actual < 1:
                # Kupiec LR统计量
                lr_kupiec = 2 * (
                    x_95 * math.log(p_actual / p_expected) +
                    (n - x_95) * math.log((1 - p_actual) / (1 - p_expected))
                )
                # 卡方(1)的p值近似
                result.kupiec_pvalue_95 = math.exp(-lr_kupiec / 2) if lr_kupiec > 0 else 1.0

        # 简化Christoffersen检验：检查违规是否聚集
        violations = [1 if actual > max(pred, 0.01) else 0
                     for pred, actual in zip(all_predictions_95, all_actuals)]
        if len(violations) >= 4:
            # 计算条件概率
            n00 = n01 = n10 = n11 = 0
            for i in range(1, len(violations)):
                if violations[i-1] == 0 and violations[i] == 0:
                    n00 += 1
                elif violations[i-1] == 0 and violations[i] == 1:
                    n01 += 1
                elif violations[i-1] == 1 and violations[i] == 0:
                    n10 += 1
                elif violations[i-1] == 1 and violations[i] == 1:
                    n11 += 1

            p01 = n01 / max(n00 + n01, 1)
            p11 = n11 / max(n10 + n11, 1)
            if abs(p01 - p11) < 0.3:
                result.christoffersen_pvalue_95 = 0.5  # 无显著聚集

        # 评估结果
        if result.kupiec_pvalue_95 < 0.01:
            result.status = "fail"
        elif result.kupiec_pvalue_95 < 0.05:
            result.status = "warning"
        else:
            result.status = "pass"

        return result

    # ═══════════════════════════════════════════════════════════
    # 增强2: 币种级集中度监控
    # ═══════════════════════════════════════════════════════════

    def _compute_symbol_concentration(
        self,
        strategy_names: List[str],
        plan: RiskBudgetPlan,
    ) -> Dict[str, SymbolRiskBudget]:
        """
        计算每个币种的风险集中度

        逻辑：
          1. 将各策略的VaR按币种聚合
          2. 检查每个币种是否超过单币种风险上限
          3. 检查高相关币种群是否超过组风险上限
        """
        symbol_risks: Dict[str, SymbolRiskBudget] = {}

        # P32 修复：_symbol_strategy_map 的键是 symbol、值是 strategies，
        # 先构建反向映射 strategy -> [symbols]，再据此聚合，避免把策略名误当作币种。
        strategy_symbols: Dict[str, List[str]] = {}
        for symbol, strategies in self._symbol_strategy_map.items():
            for strat in strategies:
                if strat not in strategy_symbols:
                    strategy_symbols[strat] = []
                if symbol not in strategy_symbols[strat]:
                    strategy_symbols[strat].append(symbol)

        # 从策略VaR聚合到币种
        for name in strategy_names:
            rb = plan.strategy_budgets.get(name)
            if not rb:
                continue

            # 获取该策略涉及的币种（正确的反向映射方向）
            symbols = strategy_symbols.get(name, [])
            # 如果没有注册映射，策略名本身就是占位标识（退化行为）
            if not symbols:
                symbols = [name]

            var_per_symbol = rb.var_95 / max(len(symbols), 1)
            for sym in symbols:
                if sym not in symbol_risks:
                    symbol_risks[sym] = SymbolRiskBudget(symbol=sym)
                symbol_risks[sym].total_var_95 += var_per_symbol
                symbol_risks[sym].total_var_99 += rb.var_99 / max(len(symbols), 1)
                if name not in symbol_risks[sym].contributing_strategies:
                    symbol_risks[sym].contributing_strategies.append(name)

        # 计算占比和告警状态
        total_var = plan.total_var_95 or 1
        max_symbol_risk = plan.total_equity * self._max_symbol_risk

        for sym, srb in symbol_risks.items():
            srb.risk_pct_of_portfolio = srb.total_var_95 / max(total_var, 1)
            srb.concentration_limit = self._max_symbol_risk

            if srb.total_var_95 > max_symbol_risk * 1.2:
                srb.is_over_concentration = True
                srb.warning_level = "critical"
                plan.warnings.append(
                    f"CRITICAL: {sym} concentration {srb.total_var_95:.0f} "
                    f"exceeds limit {max_symbol_risk:.0f} USDT"
                )
            elif srb.total_var_95 > max_symbol_risk:
                srb.is_over_concentration = True
                srb.warning_level = "warning"
                plan.warnings.append(
                    f"WARNING: {sym} approaching concentration limit"
                )

        self._symbol_risk_map = symbol_risks
        return symbol_risks

    # ═══════════════════════════════════════════════════════════
    # 增强3: 动态杠杆帽推导
    # ═══════════════════════════════════════════════════════════

    def _derive_leverage_caps(
        self,
        strategy_names: List[str],
        plan: RiskBudgetPlan,
    ):
        """
        从风险预算推导每个策略的动态杠杆上限

        逻辑：
          Leverage_Cap = (Budget * Equity) / (Position_Value / Leverage)
          简化：Leverage_Cap ∝ budget_pct / volatility

        高波动策略 → 低杠杆帽
        高预算策略 → 可适当提高杠杆帽
        """
        for name in strategy_names:
            rb = plan.strategy_budgets.get(name)
            if not rb:
                continue

            # 基础杠杆帽 = 预算占比 * 波动率调整
            budget_pct = rb.budget_pct
            vol = rb.ann_volatility or 0.02

            # 波动率越高，杠杆应越低
            vol_penalty = min(1.0, 0.02 / max(vol, 0.005))

            # 预算越大，可以给略高杠杆（但有限制）
            budget_bonus = min(1.5, 1.0 + budget_pct * 2)

            # 综合杠杆帽
            base_leverage = 5  # 基础5x
            leverage_cap = base_leverage * vol_penalty * budget_bonus

            # 硬限制
            leverage_cap = max(1, min(15, leverage_cap))

            self._strategy_leverage_caps[name] = round(leverage_cap, 1)

            # 设置到RiskBudget
            rb._leverage_cap = leverage_cap

            if leverage_cap < 3:
                rb.is_warning = True
                if not rb.warning_reason:
                    rb.warning_reason = f"Leverage cap reduced to {leverage_cap:.1f}x due to volatility"

    # ═══════════════════════════════════════════════════════════
    # 增强4: 风险预算漂移追踪
    # ═══════════════════════════════════════════════════════════

    def _track_budget_drift(
        self,
        strategy_names: List[str],
        plan: RiskBudgetPlan,
    ):
        """追踪实际风险占比与目标预算的漂移"""
        total_rc = sum(
            plan.strategy_budgets.get(name, RiskBudget("")).rc_pct
            for name in strategy_names
        ) or 1.0

        for name in strategy_names:
            rb = plan.strategy_budgets.get(name)
            if not rb:
                continue

            target = rb.budget_pct
            actual = rb.rc_pct / max(total_rc, 0.01)

            drift = actual - target

            if name not in self._budget_drift:
                self._budget_drift[name] = RiskBudgetDrift(strategy_name=name)

            drift_obj = self._budget_drift[name]
            drift_obj.target_budget_pct = target
            drift_obj.actual_risk_pct = actual
            drift_obj.drift_pct = drift
            drift_obj.drift_trend.append(drift)
            if len(drift_obj.drift_trend) > self._drift_history_len:
                drift_obj.drift_trend = drift_obj.drift_trend[-self._drift_history_len:]

            # 判断漂移方向
            if abs(drift) < 0.02:
                drift_obj.drift_direction = "stable"
            elif drift > 0:
                drift_obj.drift_direction = "expanding"
            else:
                drift_obj.drift_direction = "contracting"

            # 告警条件：漂移超过50%目标值
            if abs(drift) > target * 0.5:
                drift_obj.alert = True
                drift_obj.alert_reason = (
                    f"Risk drift {drift:+.1%} exceeds 50% of budget {target:.1%}"
                )
            else:
                drift_obj.alert = False
                drift_obj.alert_reason = ""

    # ═══════════════════════════════════════════════════════════
    # 增强5: 预算使用趋势预测
    # ═══════════════════════════════════════════════════════════

    def _predict_budget_trend(self, plan: RiskBudgetPlan) -> Dict[str, Any]:
        """
        基于历史利用率预测未来预算消耗趋势

        使用简单线性回归预测未来N小时利用率
        """
        if len(self._snapshots) < 3:
            return {"trend": "insufficient_data", "predictions": {}}

        # 提取历史利用率序列
        recent = list(self._snapshots)[-20:]
        timestamps = list(range(len(recent)))
        utilizations = [s.overall_utilization for s in recent]

        if len(utilizations) < 3:
            return {"trend": "insufficient_data", "predictions": {}}

        # 简单线性回归
        x = np.array(timestamps, dtype=float)
        y = np.array(utilizations, dtype=float)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))

        slope = float(np.sum((x - x_mean) * (y - y_mean)) / max(np.sum((x - x_mean) ** 2), 1))

        # 预测未来4个周期（每周期5分钟）
        future_steps = [1, 2, 3, 4, 8, 12]
        predictions = {}
        for step in future_steps:
            pred_val = y[-1] + slope * step
            predictions[f"t+{step * 5}min"] = round(max(0, pred_val), 4)

        # 趋势判断
        if slope > 0.01:
            trend = "rapidly_increasing"
        elif slope > 0.003:
            trend = "increasing"
        elif slope < -0.01:
            trend = "rapidly_decreasing"
        elif slope < -0.003:
            trend = "decreasing"
        else:
            trend = "stable"

        # 告警：预测将在N周期内超过阈值
        for step, pred in predictions.items():
            if pred > 0.90:
                plan.warnings.append(
                    f"Risk budget predicted to exceed 90% within {step}"
                )
                break

        return {
            "trend": trend,
            "slope": round(slope, 6),
            "current_utilization": round(y[-1], 4),
            "predictions": predictions,
        }

    # ═══════════════════════════════════════════════════════════
    # 增强6: 历史快照持久化
    # ═══════════════════════════════════════════════════════════

    def _auto_snapshot(self, plan: RiskBudgetPlan):
        """自动创建并保存历史快照"""
        # 检查是否需要快照
        now = datetime.now()
        if self._last_snapshot_time is not None:
            elapsed = (now - self._last_snapshot_time).total_seconds()
            if elapsed < self._snapshot_interval_seconds:
                return

        shapshot = RiskBudgetSnapshot(
            timestamp=now.isoformat(),
            total_equity=plan.total_equity,
            total_budget=plan.total_risk_budget,
            total_consumed=sum(rb.consumed for rb in plan.strategy_budgets.values()),
            overall_utilization=plan.overall_utilization,
            strategy_utilizations={
                name: rb.utilization_pct
                for name, rb in plan.strategy_budgets.items()
            },
            var_95=plan.total_var_95,
            var_99=plan.total_var_99,
            cvar_95=plan.total_cvar_95,
            diversification_ratio=plan.diversification_ratio,
            rap_scores={
                name: score for name, score in plan.rap_ranking
            },
            active_warnings=list(plan.warnings),
        )

        self._snapshots.append(shapshot)
        self._last_snapshot_time = now

        # 异步持久化
        try:
            self._save_state()
        except Exception as e:
            logger.warning(f"Failed to persist risk budget state: {e}")

    def _save_state(self):
        """持久化风险预算状态到JSON文件"""
        try:
            os.makedirs(self._data_dir, exist_ok=True)
            state_path = os.path.join(self._data_dir, "risk_budget_state.json")

            state = {
                "last_updated": datetime.now().isoformat(),
                "snapshots": [s.to_dict() for s in self._snapshots],
                "lever_age_caps": dict(self._strategy_leverage_caps),
                "symbol_concentration": {
                    sym: {
                        "var_95": srb.total_var_95,
                        "risk_pct": srb.risk_pct_of_portfolio,
                        "warning": srb.warning_level,
                    }
                    for sym, srb in self._symbol_risk_map.items()
                },
                "budget_drift": {
                    name: {
                        "drift_pct": d.drift_pct,
                        "direction": d.drift_direction,
                        "alert": d.alert,
                    }
                    for name, d in self._budget_drift.items()
                },
            }

            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)

            logger.debug(f"Risk budget state saved to {state_path} ({len(self._snapshots)} snapshots)")
        except Exception as e:
            logger.warning(f"Failed to save risk budget state: {e}")

    def _load_state(self) -> bool:
        """加载历史风险预算状态"""
        try:
            state_path = os.path.join(self._data_dir, "risk_budget_state.json")
            if not os.path.exists(state_path):
                logger.info("No previous risk budget state found")
                return False

            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            # 恢复快照
            for snap_dict in state.get("snapshots", [])[-100:]:
                snapshot = RiskBudgetSnapshot(
                    timestamp=snap_dict.get("timestamp", ""),
                    total_equity=snap_dict.get("total_equity", 0),
                    total_budget=snap_dict.get("total_budget", 0),
                    total_consumed=snap_dict.get("total_consumed", 0),
                    overall_utilization=snap_dict.get("overall_utilization", 0),
                    strategy_utilizations=snap_dict.get("strategy_utilizations", {}),
                    var_95=snap_dict.get("var_95", 0),
                    var_99=snap_dict.get("var_99", 0),
                    cvar_95=snap_dict.get("cvar_95", 0),
                    diversification_ratio=snap_dict.get("diversification_ratio", 0),
                    rap_scores=snap_dict.get("rap_scores", {}),
                    active_warnings=snap_dict.get("active_warnings", []),
                )
                self._snapshots.append(snapshot)

            # 恢复杠杆帽
            self._strategy_leverage_caps = state.get("lever_age_caps", {})

            logger.info(f"Loaded {len(self._snapshots)} risk budget snapshots from {state_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load risk budget state: {e}")
            return False

    # ═══════════════════════════════════════════════════════════
    # 增强查询方法
    # ═══════════════════════════════════════════════════════════

    def get_symbol_concentration(self) -> Dict[str, Any]:
        """获取币种级集中度监控"""
        return {
            sym: {
                "var_95": round(srb.total_var_95, 2),
                "var_99": round(srb.total_var_99, 2),
                "risk_pct": round(srb.risk_pct_of_portfolio, 4),
                "strategies": srb.contributing_strategies,
                "is_over": srb.is_over_concentration,
                "warning_level": srb.warning_level,
                "limit": srb.concentration_limit,
            }
            for sym, srb in self._symbol_risk_map.items()
        }

    def get_leverage_caps(self) -> Dict[str, float]:
        """获取策略级动态杠杆上限"""
        return dict(self._strategy_leverage_caps)

    def get_budget_drift(self) -> Dict[str, Any]:
        """获取风险预算漂移详情"""
        return {
            name: {
                "target_pct": round(d.target_budget_pct, 4),
                "actual_pct": round(d.actual_risk_pct, 4),
                "drift_pct": round(d.drift_pct, 4),
                "direction": d.drift_direction,
                "alert": d.alert,
                "alert_reason": d.alert_reason,
                "trend": [round(v, 4) for v in d.drift_trend[-10:]],
            }
            for name, d in self._budget_drift.items()
        }

    def get_budget_trend(self) -> Dict[str, Any]:
        """获取预算使用趋势"""
        if not self._last_plan or not hasattr(self._last_plan, '_budget_trend'):
            return {"trend": "no_data"}

        trend = self._last_plan._budget_trend
        snapshots = [s.to_dict() for s in list(self._snapshots)[-24:]]  # 最近2小时
        return {
            **trend,
            "recent_snapshots": snapshots,
        }

    def get_var_backtest(self) -> Dict[str, Any]:
        """获取VaR回测结果"""
        if not self._last_plan or not hasattr(self._last_plan, '_var_backtest'):
            return {"error": "No backtest data"}

        bt = self._last_plan._var_backtest
        return {
            "total_observations": bt.total_observations,
            "var_95_violations": bt.var_95_violations,
            "expected_95_violations": round(bt.expected_95_violations, 1),
            "kupiec_pvalue_95": round(bt.kupiec_pvalue_95, 4),
            "christoffersen_pvalue_95": round(bt.christoffersen_pvalue_95, 4),
            "status": bt.status,
            "avg_violation_magnitude": round(bt.avg_violation_magnitude, 2),
            "max_violation": round(bt.max_violation, 2),
        }

    def get_historical_snapshots(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取历史快照序列"""
        return [s.to_dict() for s in list(self._snapshots)[-limit:]]

    # ═══════════════════════════════════════════════════════════
    # 交易级风险校验
    # ═══════════════════════════════════════════════════════════

    async def check_trade_risk_budget(
        self,
        strategy_name: str,
        trade_var: float,       # 该笔交易预估VaR
        total_equity: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        交易前风险预算校验

        Returns:
            (approved, reason, details)
        """
        async with self._lock:
            plan = self._last_plan
            if not plan or strategy_name not in plan.strategy_budgets:
                # 无计划时使用宽松默认值
                max_trade = total_equity * self._max_per_trade_risk
                if trade_var > max_trade:
                    return False, f"Trade VaR {trade_var:.0f} exceeds max per-trade risk {max_trade:.0f}", {}
                return True, "Approved (no plan)", {}

            rb = plan.strategy_budgets[strategy_name]

            # 检查1: 策略风险预算是否耗尽
            if rb.is_exhausted:
                return False, f"Strategy {strategy_name} risk budget exhausted", {
                    "utilization": rb.utilization_pct,
                    "remaining": rb.remaining,
                }

            # 检查2: 单笔风险上限
            max_trade_risk = total_equity * self._max_per_trade_risk
            if trade_var > max_trade_risk:
                return False, f"Trade VaR {trade_var:.0f} exceeds per-trade limit {max_trade_risk:.0f}", {
                    "trade_var": trade_var,
                    "max_allowed": max_trade_risk,
                }

            # 检查3: 交易后是否会超预算
            after_trade = rb.consumed + trade_var
            if after_trade > rb.budget_amount * 1.05:  # 5%缓冲
                return False, f"Trade would exceed risk budget: {after_trade:.0f} > {rb.budget_amount:.0f}", {
                    "current_consumed": rb.consumed,
                    "after_trade": after_trade,
                    "budget": rb.budget_amount,
                }

            # 检查4: 全局利用率
            if plan.overall_utilization >= 0.95:
                return False, "Overall risk budget exhausted", {
                    "overall_utilization": plan.overall_utilization,
                }

            return True, "Approved", {
                "remaining_budget": rb.remaining,
                "after_trade_utilization": after_trade / max(rb.budget_amount, 1),
            }

    # ═══════════════════════════════════════════════════════════
    # 风险预算再平衡
    # ═══════════════════════════════════════════════════════════

    async def rebalance_risk_budgets(
        self,
        strategy_metrics: Dict[str, Dict[str, float]],
        total_equity: float,
    ) -> Dict[str, Any]:
        """
        风险预算再平衡：从低效策略转移预算到高效策略

        规则：
          1. 胜率 < 35% → 转出 50% 的预算到高 Sharpe 策略
          2. 回撤 > 8% → 转出 30% 的预算
          3. Sharpe > 2.0 → 接收转入预算（最多增加到 2x 原始预算）
        """
        async with self._lock:
            plan = self._last_plan
            if not plan:
                return {"rebalanced": False, "reason": "No plan available"}

            # 计算当前预算
            current_budgets = {
                name: rb.budget_pct for name, rb in plan.strategy_budgets.items()
            }

            # 找出转出和转入候选
            donors: Dict[str, float] = {}   # {name: transfer_out_pct}
            receivers: List[str] = []

            for name, m in strategy_metrics.items():
                if name not in current_budgets:
                    continue

                # 转出条件
                win_rate = m.get("win_rate", 0.5)
                max_dd = m.get("max_drawdown", 0)

                if win_rate < self._transfer_winrate_threshold:
                    donors[name] = current_budgets[name] * 0.5
                elif max_dd > self._transfer_drawdown_threshold:
                    donors[name] = current_budgets[name] * 0.3

                # 转入条件
                sharpe = m.get("sharpe_ratio", 0)
                if sharpe > 2.0:
                    receivers.append(name)

            if not donors or not receivers:
                return {
                    "rebalanced": False,
                    "reason": "No rebalancing candidates",
                    "donors": [],
                    "receivers": [],
                }

            # 执行转移
            total_transferred = sum(donors.values())
            per_receiver = total_transferred / len(receivers)

            changes = {}
            for name in donors:
                changes[name] = -donors[name]
            for name in receivers:
                # 限制最多翻倍
                current = current_budgets.get(name, 0)
                capped = min(per_receiver, current)  # 最高翻倍
                changes[name] = capped

            logger.info(
                f"Risk budget rebalanced: donors={list(donors.keys())}, "
                f"receivers={receivers}, total_transferred={total_transferred:.3%}"
            )

            return {
                "rebalanced": True,
                "donors": list(donors.keys()),
                "receivers": receivers,
                "transferred_pct": round(total_transferred, 4),
                "changes": {k: round(v, 6) for k, v in changes.items()},
            }

    # ═══════════════════════════════════════════════════════════
    # 查询方法
    # ═══════════════════════════════════════════════════════════

    def get_last_plan(self) -> Optional[Dict[str, Any]]:
        if self._last_plan:
            return self._last_plan.to_dict()
        return None

    def get_risk_utilization(self) -> Dict[str, Any]:
        """获取风险利用率摘要"""
        if not self._last_plan:
            return {"error": "No plan"}

        return {
            "total_budget": self._last_plan.total_risk_budget,
            "overall_utilization": self._last_plan.overall_utilization,
            "exhausted": self._last_plan.exhausted_strategies,
            "approaching": self._last_plan.approaching_limit_strategies,
            "details": {
                name: {
                    "budget": round(rb.budget_amount, 2),
                    "consumed": round(rb.consumed, 2),
                    "utilization": round(rb.utilization_pct, 4),
                }
                for name, rb in self._last_plan.strategy_budgets.items()
            },
        }

    def get_risk_decomposition(self) -> Dict[str, Any]:
        """获取风险分解详情"""
        if not self._last_plan:
            return {"error": "No plan"}

        return {
            "total_var_95": round(self._last_plan.total_var_95, 2),
            "total_var_99": round(self._last_plan.total_var_99, 2),
            "total_cvar_95": round(self._last_plan.total_cvar_95, 2),
            "diversification_ratio": round(self._last_plan.diversification_ratio, 4),
            "risk_contributions": {
                name: {
                    "mrc": round(rb.mrc, 6),
                    "crc": round(rb.crc, 4),
                    "rc_pct": round(rb.rc_pct, 4),
                }
                for name, rb in self._last_plan.strategy_budgets.items()
            },
        }

    def get_rapm_ranking(self) -> List[Dict[str, Any]]:
        """获取风险调整绩效排名"""
        if not self._last_plan:
            return []
        return [
            {
                "name": name,
                "score": round(score, 4),
                "sharpe": round(self._last_plan.strategy_budgets.get(name, RiskBudget("")).sharpe, 4),
                "sortino": round(self._last_plan.strategy_budgets.get(name, RiskBudget("")).sortino, 4),
                "calmar": round(self._last_plan.strategy_budgets.get(name, RiskBudget("")).calmar, 4),
            }
            for name, score in self._last_plan.rap_ranking
        ]

    def get_correlation_risk_report(self) -> Dict[str, Any]:
        """获取相关性风险报告（供 Dashboard 和调度器查询）"""
        if not hasattr(self, '_correlation_analyzer') or not self._correlation_analyzer:
            return {"status": "correlation_analyzer_not_available"}

        try:
            strategy_names = list(self._strategy_budget_pcts.keys())
            breaches = self._analyze_correlation_breach(strategy_names)

            high_corr_pairs = []
            extreme_pairs = []
            for pair_key, info in breaches.items():
                if info.get("extreme"):
                    extreme_pairs.append({
                        "pair": info["strategy_pair"],
                        "correlation": info["correlation"],
                        "pearson": info["pearson"],
                        "spearman": info["spearman"],
                    })
                elif info.get("breach"):
                    high_corr_pairs.append({
                        "pair": info["strategy_pair"],
                        "correlation": info["correlation"],
                        "pearson": info["pearson"],
                    })

            return {
                "timestamp": datetime.now().isoformat(),
                "thresholds": {
                    "high": self._high_correlation_threshold,
                    "extreme": self._extreme_correlation_threshold,
                },
                "high_correlation_pairs": high_corr_pairs,
                "extreme_correlation_pairs": extreme_pairs,
                "total_pairs_monitored": len(breaches),
                "breach_count": len(high_corr_pairs) + len(extreme_pairs),
                "status": "critical" if extreme_pairs else ("warning" if high_corr_pairs else "normal"),
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}


# ═══════════════════════════════════════════════════════════════
# 单例工厂
# ═══════════════════════════════════════════════════════════════

_risk_budget_engine: Optional[RiskBudgetEngine] = None


def get_risk_budget_engine(config: Dict[str, Any] = None) -> RiskBudgetEngine:
    global _risk_budget_engine
    if _risk_budget_engine is None and config is not None:
        _risk_budget_engine = RiskBudgetEngine(config)
    return _risk_budget_engine


def reset_risk_budget_engine():
    global _risk_budget_engine
    _risk_budget_engine = None


__all__ = [
    "RiskBudgetEngine",
    "RiskBudget",
    "RiskBudgetPlan",
    "RiskBudgetLevel",
    "RiskDecompositionMethod",
    "BudgetAdjustmentTrigger",
    "SymbolRiskBudget",
    "RiskBudgetDrift",
    "VaRBacktestResult",
    "RiskBudgetSnapshot",
    "get_risk_budget_engine",
    "reset_risk_budget_engine",
]
