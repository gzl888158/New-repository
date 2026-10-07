"""
动态资金分配引擎（Dynamic Capital Allocator）

增强版资金分配算法，替代简单的 equal/performance/risk_adjusted 三种分配方式。
提供：
  - 多级资金池管理（底仓 / 加仓 / 风控隔离）
  - Kelly 公式最优仓位计算
  - 波动率自适应分配
  - 回撤响应式降仓
  - 分配优先级瀑布（高Sharpe优先）
  - 闲置资金检测与部署
  - 跨策略资金共享
  - 盈亏驱动的池间资金流动
  - 市场状态感知的动态权重
"""
import asyncio
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


def _is_finite(x) -> bool:
    """判断是否为有限数值（拒绝 None/bool/NaN/Inf）。"""
    if x is None or isinstance(x, bool):
        return False
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(v) or math.isinf(v))


def _safe_float(x, default: float = 0.0) -> float:
    """将输入安全转换为有限 float，None/NaN/Inf/非法值返回 default。"""
    if x is None or isinstance(x, bool):
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


# ═══════════════════════════════════════════════════════════════
# 枚举与数据模型
# ═══════════════════════════════════════════════════════════════

class PoolType(Enum):
    """资金池类型"""
    BASE = "base"           # 底仓池 — 核心策略运行资金
    ADDON = "addon"         # 加仓池 — 盈利增仓/追加资金
    RESERVE = "reserve"     # 风控隔离池 — 极端行情备用


class MarketRegime(Enum):
    """市场状态（轻量版，不依赖MarketRegimeEngine）"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"
    REVERSAL = "reversal"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNKNOWN = "unknown"


class AllocationPriority(Enum):
    """分配优先级"""
    CRITICAL = 0     # 最高：已有仓位保证金补足
    HIGH = 1         # 高：高Sharpe策略优先
    MEDIUM = 2       # 中：中等表现策略
    LOW = 3          # 低：低Sharpe/实验策略
    FROZEN = 4       # 冻结：已失效策略，不分配


@dataclass
class CapitalPool:
    """资金池状态"""
    pool_type: PoolType
    total_capital: float = 0.0         # 池内总资金
    allocated: float = 0.0             # 已分配资金
    used: float = 0.0                  # 已占用保证金
    available: float = 0.0             # 可用资金
    locked: float = 0.0                # 锁定额度（跨池借用）
    min_ratio: float = 0.0             # 最小占比
    max_ratio: float = 0.0             # 最大占比

    @property
    def free(self) -> float:
        return max(0, self.available - self.used - self.locked)

    @property
    def usage_pct(self) -> float:
        if self.total_capital <= 0:
            return 0.0
        return (self.used + self.locked) / self.total_capital


@dataclass
class StrategyAllocation:
    """单策略资金分配状态"""
    name: str
    priority: AllocationPriority = AllocationPriority.MEDIUM
    target_weight: float = 0.0         # 目标权重（占总资金%）
    current_weight: float = 0.0        # 当前权重
    allocated_capital: float = 0.0     # 已分配资金
    used_margin: float = 0.0           # 已用保证金
    available: float = 0.0             # 可用资金
    unrealized_pnl: float = 0.0        # 未实现盈亏
    realized_pnl: float = 0.0          # 已实现盈亏

    # 策略绩效快照
    win_rate: float = 0.5
    sharpe_ratio: float = 0.0
    profit_factor: float = 1.0
    max_drawdown: float = 0.0
    trade_count: int = 0
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    volatility_30d: float = 0.0        # 30日年化波动率

    # Kelly 指标
    kelly_fraction: float = 0.0        # Kelly 最优仓位比例
    half_kelly: float = 0.0            # 半Kelly（更保守）

    # 状态标记
    is_active: bool = True             # 策略是否活跃
    is_frozen: bool = False            # 是否被冻结
    is_open_blocked: bool = False      # 是否禁止新增仓位
    freeze_reason: str = ""            # 冻结原因
    is_probing: bool = False           # 是否处于缩量试探期（冻结观察期满后）

    # ── 企业级：资金利用率追踪 ──
    capital_utilization_pct: float = 0.0    # 实际资金利用率（已用保证金/已分配资金）
    idle_capital: float = 0.0               # 该策略闲置资金
    last_utilization_warning: str = ""      # 最近一次利用率警告时间

    @property
    def total_equity(self) -> float:
        """策略总权益 = 分配资金 + 未实现盈亏 + 已实现盈亏"""
        return self.allocated_capital + self.unrealized_pnl + self.realized_pnl

    @property
    def margin_usage_pct(self) -> float:
        if self.total_equity <= 0:
            return 0.0
        return self.used_margin / self.total_equity

    @property
    def is_underutilized(self) -> bool:
        """资金利用率 < 50% 视为低效占用"""
        return self.capital_utilization_pct < 0.50 and self.allocated_capital > 0


@dataclass
class AllocationPlan:
    """资金分配方案"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_capital: float = 0.0
    total_equity: float = 0.0

    # 三级资金池
    pools: Dict[str, CapitalPool] = field(default_factory=dict)

    # 策略分配
    strategy_allocations: Dict[str, StrategyAllocation] = field(default_factory=dict)

    # 分配结果
    allocation_changes: Dict[str, float] = field(default_factory=dict)  # 策略->权重变化
    pool_flows: Dict[str, float] = field(default_factory=dict)          # 池间资金流动

    # 资金效率
    idle_cash: float = 0.0               # 闲置资金
    idle_duration_minutes: float = 0.0   # 闲置资金持续时长（分钟）
    capital_efficiency: float = 0.0      # 资金使用效率 (0-1)
    deployment_ratio: float = 0.0        # 部署率
    strategy_utilization: Dict[str, float] = field(default_factory=dict)  # 每策略利用率
    risk_adjusted_efficiency: float = 0.0  # 风险调整资金效率（Sharpe加权）(0-1)
    opportunity_cost: float = 0.0          # 闲置资金机会成本（日化）

    # 风险指标
    total_leverage: float = 0.0
    concentration_ratio: float = 0.0     # 前2名集中度

    # 元信息
    method: str = "dynamic"
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    auto_actions: List[str] = field(default_factory=list)  # 企业级：自动执行的动作

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "total_capital": self.total_capital,
            "total_equity": self.total_equity,
            "pools": {
                k: {
                    "total": round(v.total_capital, 2),
                    "allocated": round(v.allocated, 2),
                    "used": round(v.used, 2),
                    "available": round(v.available, 2),
                    "free": round(v.free, 2),
                    "usage_pct": round(v.usage_pct, 4),
                } for k, v in self.pools.items()
            },
            "strategy_allocations": {
                k: {
                    "target_weight": round(v.target_weight, 4),
                    "allocated_capital": round(v.allocated_capital, 2),
                    "used_margin": round(v.used_margin, 2),
                    "available": round(v.available, 2),
                    "total_equity": round(v.total_equity, 2),
                    "kelly_fraction": round(v.kelly_fraction, 4),
                    "priority": v.priority.name,
                    "is_frozen": v.is_frozen,
                    "is_open_blocked": v.is_open_blocked,
                    "is_probing": v.is_probing,
                    "freeze_reason": v.freeze_reason,
                } for k, v in self.strategy_allocations.items()
            },
            "allocation_changes": {k: round(v, 4) for k, v in self.allocation_changes.items()},
            "pool_flows": {k: round(v, 2) for k, v in self.pool_flows.items()},
            "idle_cash": round(self.idle_cash, 2),
            "idle_duration_minutes": round(self.idle_duration_minutes, 1),
            "capital_efficiency": round(self.capital_efficiency, 4),
            "deployment_ratio": round(self.deployment_ratio, 4),
            "risk_adjusted_efficiency": round(self.risk_adjusted_efficiency, 4),
            "opportunity_cost": round(self.opportunity_cost, 4),
            "strategy_utilization": {k: round(v, 4) for k, v in self.strategy_utilization.items()},
            "total_leverage": round(self.total_leverage, 4),
            "warnings": self.warnings,
            "recommendations": self.recommendations,
            "auto_actions": self.auto_actions,
        }


# ═══════════════════════════════════════════════════════════════
# 动态资金分配引擎
# ═══════════════════════════════════════════════════════════════

class DynamicAllocator:
    """
    动态资金分配引擎

    核心算法：
      1. 三级资金池管理（底仓/加仓/风控隔离）
      2. Kelly 公式计算最优仓位
      3. 波动率自适应调整
      4. 分配优先级瀑布
      5. 闲置资金检测与跨策略共享
      6. 盈亏驱动池间流动
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()

        # ── 资金池配置 ──
        cap_cfg = config.get("capital_pool", {})
        self._base_ratio = _safe_float(cap_cfg.get("base_ratio"), 0.60)
        self._addon_ratio = _safe_float(cap_cfg.get("add_reserve_ratio"), 0.25)
        self._reserve_ratio = _safe_float(cap_cfg.get("risk_isolation_ratio"), 0.15)

        # ── 盈亏再分配配置 ──
        pnl_cfg = config.get("pnl_reallocation", {})
        self._profit_to_addon = _safe_float(pnl_cfg.get("profit_to_add_ratio"), 0.50)
        self._loss_shrink = _safe_float(pnl_cfg.get("loss_shrink_ratio"), 0.30)
        self._consecutive_profit_days = int(_safe_float(pnl_cfg.get("consecutive_profit_days"), 3))
        self._consecutive_loss_days = int(_safe_float(pnl_cfg.get("consecutive_loss_days"), 2))
        self._max_addon_ratio = _safe_float(pnl_cfg.get("max_add_pool_ratio"), 0.40)
        self._min_base_ratio = _safe_float(pnl_cfg.get("min_base_pool_ratio"), 0.50)

        # ── 分配控制参数 ──
        alloc_cfg = config.get("allocation_agent", {})
        self._max_weight_change = _safe_float(alloc_cfg.get("max_allocation_change"), 0.10)
        self._min_trade_count = int(_safe_float(alloc_cfg.get("min_trade_count"), 20))
        self._rebalance_interval = int(_safe_float(alloc_cfg.get("rebalance_interval"), 3600))

        # ── 风险参数 ──
        self._max_single_weight = _safe_float(config.get("symbol_allocation", {}).get("max_symbol_weight"), 0.15) * 3  # 策略级放宽
        self._min_single_weight = _safe_float(config.get("symbol_allocation", {}).get("min_symbol_weight"), 0.02)
        self._max_total_leverage = _safe_float(config.get("leverage_tiers", {}).get("absolute_max"), 10)

        # ── Kelly 参数 ──
        self._kelly_enabled = True
        self._kelly_max_fraction = 0.25       # Kelly 仓位上限
        self._use_half_kelly = True           # 默认使用半Kelly

        # ── 企业级：资金效率强化参数 ──
        eff_cfg = config.get("capital_efficiency", {})
        self._idle_sweep_threshold = _safe_float(eff_cfg.get("idle_sweep_threshold"), 0.20)     # 闲置>20%触发归集
        self._idle_sweep_min_duration = _safe_float(eff_cfg.get("idle_sweep_min_duration"), 30) # 闲置持续30分钟才归集
        self._utilization_reclaim_threshold = _safe_float(eff_cfg.get("utilization_reclaim_threshold"), 0.50)  # 利用率<50%回收
        self._min_capital_efficiency = _safe_float(eff_cfg.get("min_capital_efficiency"), 0.15)  # 最低资金效率
        self._emergency_reserve_threshold = _safe_float(eff_cfg.get("emergency_reserve_threshold"), 0.10)  # 风控池<10%触发补充
        self._max_concentration_ratio = _safe_float(eff_cfg.get("max_concentration_ratio"), 0.50)  # 最大集中度
        self._volatility_pool_adapt = eff_cfg.get("volatility_pool_adapt", True)       # 波动率自适应池比例
        self._opportunity_cost_daily = _safe_float(eff_cfg.get("opportunity_cost_daily"), 0.0005)   # 闲置资金机会成本日化率（0.05%）
        self._positive_return_protection = eff_cfg.get("positive_return_protection", True)  # 正收益策略保护

        # ── 状态 ──
        self._pools: Dict[str, CapitalPool] = {}
        self._strategy_allocations: Dict[str, StrategyAllocation] = {}
        self._last_plan: Optional[AllocationPlan] = None
        self._last_rebalance: Optional[datetime] = None
        self._idle_start_time: Optional[datetime] = None  # 闲置资金开始时间

        # ── 策略绩效缓存 ──
        self._strategy_returns: Dict[str, List[float]] = {}
        self._daily_pnl: Dict[str, List[float]] = {}

        # ── 冻结策略观察期自动解冻 + 缩量试探（企业级 fail-closed） ──
        freeze_cfg = config.get("freeze_policy", {})
        self._freeze_enabled = bool(freeze_cfg.get("enabled", True))
        self._freeze_observe_seconds = max(0.0, _safe_float(freeze_cfg.get("observe_minutes"), 60.0) * 60.0)
        self._probe_weight_ratio = max(0.05, min(1.0, _safe_float(freeze_cfg.get("probe_weight_ratio"), 0.5)))
        self._probe_max_attempts = max(0, int(_safe_float(freeze_cfg.get("probe_max_attempts"), 3)))
        self._consecutive_loss_freeze = max(1, int(_safe_float(freeze_cfg.get("consecutive_loss_freeze"), 5)))
        self._max_drawdown_freeze = max(0.0, _safe_float(freeze_cfg.get("max_drawdown_freeze"), 0.15))
        # P1: 永久冻结自动恢复——经过足够长的冷却期后，允许策略重新进入试探周期
        # 避免策略因短期市场不适被永久冻结，需人工介入解冻
        self._auto_recovery_seconds = max(0.0, _safe_float(freeze_cfg.get("auto_recovery_hours"), 24.0) * 3600.0)
        # 正收益高盈亏比策略（低胜率高盈亏比，回撤天然大）的回撤冻结阈值放宽
        self._positive_return_drawdown_freeze = max(
            self._max_drawdown_freeze,
            _safe_float(freeze_cfg.get("positive_return_drawdown_freeze"), 0.60),
        )
        self._positive_return_min_profit_factor = _safe_float(
            freeze_cfg.get("positive_return_min_profit_factor"), 2.0
        )
        # 正收益高盈亏比策略的最低成交数门槛（放宽，避免样本略不足即降为 LOW）
        self._positive_return_min_trade_count = max(
            0, int(_safe_float(freeze_cfg.get("positive_return_min_trade_count"), 10))
        )
        # 冻结状态机：{strategy: {frozen_at, reason, probing, probe_attempts, probe_wins_baseline}}
        self._freeze_state: Dict[str, Dict[str, Any]] = {}
        # 本轮处于「缩量试探」的策略集合（由 _evaluate_strategy_priorities 填充，供瀑布缩量使用）
        self._probing_strategies: set = set()

        # ── 外部依赖（延迟注入） ──
        self._portfolio_optimizer = None
        self._performance_provider: Optional[Callable] = None

        logger.info(
            f"DynamicAllocator initialized: "
            f"pools(base={self._base_ratio:.0%}, addon={self._addon_ratio:.0%}, reserve={self._reserve_ratio:.0%}), "
            f"kelly={'half' if self._use_half_kelly else 'full'}"
        )

    # ── 依赖注入 ─────────────────────────────────────────────

    def set_portfolio_optimizer(self, optimizer) -> None:
        self._portfolio_optimizer = optimizer

    def set_performance_provider(self, provider: Callable) -> None:
        """注入策略绩效数据提供者"""
        self._performance_provider = provider

    # ── 绩效数据更新 ─────────────────────────────────────────

    def update_strategy_returns(self, name: str, daily_return: float):
        """更新策略日收益率"""
        daily_return = _safe_float(daily_return, 0.0)
        if name not in self._strategy_returns:
            self._strategy_returns[name] = []
        self._strategy_returns[name].append(daily_return)
        if len(self._strategy_returns[name]) > 90:
            self._strategy_returns[name] = self._strategy_returns[name][-90:]

    def update_strategy_pnl(self, name: str, pnl: float):
        """更新策略每日盈亏"""
        pnl = _safe_float(pnl, 0.0)
        if name not in self._daily_pnl:
            self._daily_pnl[name] = []
        self._daily_pnl[name].append(pnl)
        if len(self._daily_pnl[name]) > 90:
            self._daily_pnl[name] = self._daily_pnl[name][-90:]

    # ── 核心：完整分配方案计算 ───────────────────────────────

    async def compute_allocation_plan(
        self,
        total_capital: float,
        total_equity: float,
        strategy_names: List[str],
        strategy_metrics: Dict[str, Dict[str, Any]] = None,
        market_regime: MarketRegime = MarketRegime.UNKNOWN,
        current_weights: Dict[str, float] = None,
        used_margin_by_strategy: Dict[str, float] = None,
        persist_last_plan: bool = True,
    ) -> AllocationPlan:
        """
        计算完整资金分配方案

        Args:
            total_capital: 总资金
            total_equity: 总权益（含浮动盈亏）
            strategy_names: 活跃策略名称列表
            strategy_metrics: 策略绩效及可选预测指标；预测指标包括
                projected_pnl_per_cycle、projection_confidence、
                bear_case_horizon、bear_case_fail_closed
            market_regime: 当前市场状态
            current_weights: 当前策略权重
            used_margin_by_strategy: 各策略已用保证金 {name: used_margin}（企业级修复数据断链）

        Returns:
            完整分配方案
        """
        async with self._lock:
            total_capital = _safe_float(total_capital, 0.0)
            total_equity = _safe_float(total_equity, 0.0)
            strategy_names = list(strategy_names or [])

            # fail-closed：无法确认权益时不做任何分配，返回空方案而非崩溃/放行
            if total_equity <= 0:
                plan = AllocationPlan(
                    total_capital=total_capital,
                    total_equity=total_equity,
                )
                plan.warnings.append(
                    "Invalid or non-positive total equity; allocation aborted (fail-closed)"
                )
                if persist_last_plan:
                    self._last_plan = plan
                return plan

            plan = AllocationPlan(
                total_capital=total_capital,
                total_equity=total_equity,
            )

            strategy_metrics = strategy_metrics or {}
            current_weights = current_weights or {}
            used_margin_by_strategy = used_margin_by_strategy or {}

            # ── Step 1: 计算三级资金池 ──
            self._compute_capital_pools(plan, total_equity)

            # ── Step 2: 市场状态调整池比例 ──
            self._adjust_pools_for_regime(plan, market_regime, total_equity)

            # ── Step 3: 评估策略优先级 ──
            priorities = self._evaluate_strategy_priorities(
                strategy_names, strategy_metrics, market_regime, total_equity
            )

            # ── Step 4: Kelly 公式计算最优仓位 ──
            kelly_weights = self._compute_kelly_weights(
                strategy_names, strategy_metrics, total_equity
            )

            # ── Step 5: 分配优先级瀑布 ──
            self._waterfall_allocation(plan, strategy_names, priorities, kelly_weights,
                                       current_weights, total_equity, market_regime,
                                       used_margin_by_strategy, strategy_metrics)

            # ── Step 6: 计算权重变化 ──
            self._compute_changes(plan, current_weights)

            # ── Step 7: 闲置资金检测 ──
            self._detect_idle_cash(plan)

            # ── Step 8: 资金效率评估 ──
            self._evaluate_efficiency(plan, strategy_metrics)

            # ── Step 9: 盈亏驱动池间流动 ──
            self._compute_pool_flows(plan, strategy_metrics)

            # ── Step 10: 生成建议 ──
            self._generate_recommendations(plan)

            # ── Step 11: 企业级 — 闲置资金自动归集 ──
            self._auto_sweep_idle(plan)

            # ── Step 12: 企业级 — 低利用率策略资金回收 ──
            self._reclaim_underutilized(plan, strategy_metrics)

            # ── Step 13: 企业级 — 应急储备检查 ──
            self._emergency_reserve_check(plan, strategy_metrics)

            if persist_last_plan:
                self._last_plan = plan
            return plan

    # ── Step 1: 三级资金池 ───────────────────────────────────

    def _compute_capital_pools(self, plan: AllocationPlan, total_equity: float):
        """初始化三级资金池"""
        plan.pools = {
            "base": CapitalPool(
                pool_type=PoolType.BASE,
                total_capital=total_equity * self._base_ratio,
                available=total_equity * self._base_ratio,
                min_ratio=self._min_base_ratio,
                max_ratio=0.70,
            ),
            "addon": CapitalPool(
                pool_type=PoolType.ADDON,
                total_capital=total_equity * self._addon_ratio,
                available=total_equity * self._addon_ratio,
                min_ratio=0.10,
                max_ratio=self._max_addon_ratio,
            ),
            "reserve": CapitalPool(
                pool_type=PoolType.RESERVE,
                total_capital=total_equity * self._reserve_ratio,
                available=total_equity * self._reserve_ratio,
                min_ratio=0.05,
                max_ratio=0.25,
            ),
        }

    # ── Step 2: 市场状态调整 ─────────────────────────────────

    def _adjust_pools_for_regime(self, plan: AllocationPlan, regime: MarketRegime,
                                  total_equity: float):
        """根据市场状态动态调整池比例"""
        # 调整系数：底仓 / 加仓 / 风控
        regime_adjustments = {
            MarketRegime.TRENDING_UP:    (1.05, 1.10, 0.85),   # 趋势上涨→增仓
            MarketRegime.TRENDING_DOWN:  (0.85, 0.60, 1.30),   # 趋势下跌→减仓+增风控
            MarketRegime.BREAKOUT:       (1.00, 1.15, 0.85),   # 突破确认→适度提高进攻池
            MarketRegime.BREAKDOWN:      (0.80, 0.45, 1.55),   # 破位→收缩加仓并提高隔离池
            MarketRegime.REVERSAL:       (0.70, 0.35, 1.65),   # 反转不确定性高→防守优先
            MarketRegime.RANGING:        (1.00, 0.95, 1.00),   # 震荡→略减加仓
            MarketRegime.HIGH_VOLATILITY:(0.80, 0.50, 1.50),   # 高波动→大幅减仓
            MarketRegime.LOW_VOLATILITY: (1.10, 1.15, 0.75),   # 低波动→增仓
            MarketRegime.UNKNOWN:        (1.00, 1.00, 1.00),
        }

        adj_base, adj_addon, adj_reserve = regime_adjustments.get(
            regime, (1.0, 1.0, 1.0)
        )

        # 调整
        for pool_name, adj in [("base", adj_base), ("addon", adj_addon), ("reserve", adj_reserve)]:
            pool = plan.pools[pool_name]
            pool.total_capital *= adj
            pool.available *= adj

            # 限制在 min/max 范围内
            pool_ratio = pool.total_capital / max(total_equity, 1)
            if pool_ratio > pool.max_ratio:
                pool.total_capital = total_equity * pool.max_ratio
                pool.available = pool.total_capital
            elif pool_ratio < pool.min_ratio:
                pool.total_capital = total_equity * pool.min_ratio
                pool.available = pool.total_capital

        # 资金守恒：regime 调整对三池比例做加减后，三池总额会偏离 total_equity
        # （如 TRENDING_DOWN 使三池和≈0.855），导致部分资金凭空消失。
        # 调整本质是「减仓→增风控」的池间再分配，不是总资金缩减，故重新归一化到权益总额。
        pool_total = sum(p.total_capital for p in plan.pools.values())
        if pool_total > 0 and abs(pool_total - total_equity) > 1e-9:
            scale = total_equity / pool_total
            for pool in plan.pools.values():
                pool.total_capital *= scale
                pool.available = pool.total_capital

    # ── Step 3: 策略优先级评估 ─────────────────────────────

    def _evaluate_strategy_priorities(
        self,
        strategy_names: List[str],
        metrics: Dict[str, Dict[str, Any]],
        regime: MarketRegime,
        total_equity: float = 0.0,
    ) -> Dict[str, AllocationPriority]:
        """综合评估策略优先级（含冻结策略观察期自动解冻 + 缩量试探）。"""
        priorities: Dict[str, AllocationPriority] = {}
        self._probing_strategies = set()

        for name in strategy_names:
            m = metrics.get(name, {})

            if bool(m.get("bear_case_fail_closed")):
                priorities[name] = AllocationPriority.FROZEN
                continue

            trade_count = _safe_float(m.get("trade_count"), 0)
            consecutive_losses = _safe_float(m.get("consecutive_losses"), 0)
            max_drawdown = _safe_float(m.get("max_drawdown"), 0)
            profit_factor = _safe_float(m.get("profit_factor"), 1.0)
            total_pnl = _safe_float(m.get("total_pnl"), 0.0)

            should_freeze, _reason = self._detect_freeze_condition(
                trade_count, consecutive_losses, max_drawdown,
                profit_factor=profit_factor, total_pnl=total_pnl,
            )

            if should_freeze:
                # 冻结状态机：观察期自动解冻 / 缩量试探
                probe_priority = self._resolve_frozen_strategy(name, m, reason=_reason)
                if probe_priority is not None:
                    priorities[name] = probe_priority
                    if probe_priority == AllocationPriority.LOW:
                        self._probing_strategies.add(name)
                    continue

            # 不满足冻结条件（或试探期已恢复）：清除冻结状态，走正常评估
            if name in self._freeze_state:
                logger.info(f"[FreezePolicy] {name} 已恢复（不再满足冻结条件），解除冻结")
                self._freeze_state.pop(name, None)

            # 交易数不足 → 最低优先级但不冻结
            # 正收益高盈亏比策略放宽门槛（已证明正期望，避免样本略不足即降为 LOW）
            min_trade = self._min_trade_count
            if total_pnl > 0 and profit_factor >= self._positive_return_min_profit_factor:
                min_trade = self._positive_return_min_trade_count
            if trade_count < min_trade:
                priorities[name] = AllocationPriority.LOW
                continue

            # 综合评分
            sharpe = _safe_float(m.get("sharpe_ratio"), 0)
            win_rate = _safe_float(m.get("win_rate"), 0.5)

            # 评分 = Sharpe*0.4 + WinRate*0.2 + ProfitFactor*0.2 + TradeCount*0.2
            tc_norm = min(1.0, trade_count / 50)
            score = sharpe * 0.4 + win_rate * 0.2 + profit_factor * 0.2 + tc_norm * 0.2

            # 前瞻推算维度（非破坏性微调，不改原权重）
            # projected_pnl_per_cycle 由 AGI orchestrator 注入，反映未来 horizon 期望盈亏
            projected_pnl = _safe_float(m.get("projected_pnl_per_cycle"), 0.0)
            proj_conf = max(0.0, min(1.0, _safe_float(m.get("projection_confidence"), 0.0)))
            # 归一化：以权益 5% 为满档（projected_pnl 多为小数，用 sigmoid 风格压缩到 [-0.15, 0.15])
            _eq_ref = max(_safe_float(total_equity, 0.0), 1.0)
            if total_equity <= 0:
                _eq_ref = max(abs(total_pnl), 1.0)
            _proj_norm = max(-1.0, min(1.0, projected_pnl / (0.05 * _eq_ref)))
            score += _proj_norm * 0.15 * proj_conf

            if score >= 0.6:
                priorities[name] = AllocationPriority.HIGH
            elif score >= 0.35:
                priorities[name] = AllocationPriority.MEDIUM
            else:
                priorities[name] = AllocationPriority.LOW

        return priorities

    def _detect_freeze_condition(
        self,
        trade_count: float,
        consecutive_losses: float,
        max_drawdown: float,
        profit_factor: float = 1.0,
        total_pnl: float = 0.0,
    ) -> Tuple[bool, str]:
        """判断策略是否满足冻结条件。返回 (should_freeze, reason)。

        正收益高盈亏比策略（低胜率高盈亏比，回撤天然大）放宽回撤冻结阈值，
        避免误杀盈利但回撤大的策略（如 sync 这种低胜率 / 高盈亏比）。
        """
        if trade_count > 0:
            if consecutive_losses >= self._consecutive_loss_freeze:
                return True, "consecutive_losses"
            dd_threshold = self._max_drawdown_freeze
            if total_pnl > 0 and profit_factor >= self._positive_return_min_profit_factor:
                dd_threshold = self._positive_return_drawdown_freeze
            if max_drawdown > dd_threshold:
                return True, "high_drawdown"
        return False, ""

    def _resolve_frozen_strategy(
        self,
        name: str,
        m: Dict[str, Any],
        reason: str = "",
    ) -> Optional[AllocationPriority]:
        """冻结状态机：观察期自动解冻 + 缩量试探。

        仅当 `_detect_freeze_condition` 判定应冻结时调用。

        返回：
          - FROZEN：观察期内 / 试探耗尽（永久冻结，需人工）
          - LOW：进入缩量试探
          - None：本轮判定应由正常评估路径接管（防御分支，正常情况下不会发生）
        """
        if not self._freeze_enabled:
            return AllocationPriority.FROZEN

        now = time.time()
        reason = reason or "high_drawdown"  # 兜底（调用方未传入时）

        state = self._freeze_state.get(name)
        if state is None:
            # 首次冻结：记录时间，进入观察期
            self._freeze_state[name] = {
                "frozen_at": now,
                "reason": reason,
                "probing": False,
                "probe_attempts": 0,
            }
            logger.warning(
                f"[FreezePolicy] {name} 触发冻结（{reason}），"
                f"进入观察期 {self._freeze_observe_seconds / 60:.0f}min"
            )
            return AllocationPriority.FROZEN

        # 冻结原因随指标变化时更新（如 high_drawdown → consecutive_losses）
        if state.get("reason") != reason:
            state["reason"] = reason

        if not state.get("probing"):
            # 观察期中：期满才进入缩量试探
            elapsed = now - _safe_float(state.get("frozen_at"), now)
            if self._freeze_observe_seconds <= 0 or elapsed < self._freeze_observe_seconds:
                return AllocationPriority.FROZEN
            state["probing"] = True
            state["probe_attempts"] = int(state.get("probe_attempts", 0)) + 1
            state["probe_started_at"] = now
            logger.info(
                f"[FreezePolicy] {name} 观察期满 → 缩量试探 "
                f"第 {state['probe_attempts']}/{self._probe_max_attempts} 次（权重 x{self._probe_weight_ratio:.0%}）"
            )
            return AllocationPriority.LOW

        # 试探窗口内：窗口时长 = 观察期时长；期满仍未恢复则计数并回观察期
        probe_elapsed = now - _safe_float(state.get("probe_started_at"), now)
        if self._freeze_observe_seconds > 0 and probe_elapsed >= self._freeze_observe_seconds:
            if int(state.get("probe_attempts", 0)) >= self._probe_max_attempts:
                # P1: 永久冻结自动恢复——冷却期足够长后，重置状态允许重新试探
                # 避免策略因短期市场不适被永久冻结，需人工介入解冻
                if self._auto_recovery_seconds > 0:
                    total_frozen_duration = now - _safe_float(state.get("frozen_at"), now)
                    if total_frozen_duration >= self._auto_recovery_seconds:
                        logger.info(
                            f"[FreezePolicy] {name} 永久冻结已超过 "
                            f"{total_frozen_duration / 3600:.1f}h（阈值 {self._auto_recovery_seconds / 3600:.1f}h），"
                            f"自动恢复，重新进入试探周期"
                        )
                        # 重置冻结状态，允许策略重新进入观察期→试探循环
                        self._freeze_state.pop(name, None)
                        # 递归调用重新进入首次冻结逻辑
                        return self._resolve_frozen_strategy(name, m, reason)
                logger.warning(
                    f"[FreezePolicy] {name} 试探 {self._probe_max_attempts} 次仍未恢复，"
                    f"永久冻结（需人工确认）"
                )
                return AllocationPriority.FROZEN
            # 重新进入观察期，等待下一轮试探
            state["probing"] = False
            state["frozen_at"] = now
            logger.info(f"[FreezePolicy] {name} 试探未恢复，重新进入观察期")
            return AllocationPriority.FROZEN

        # 试探窗口内，继续缩量试探
        return AllocationPriority.LOW

    # ── Step 4: Kelly 最优仓位 ────────────────────────────────

    def _compute_kelly_weights(
        self,
        strategy_names: List[str],
        metrics: Dict[str, Dict[str, Any]],
        total_equity: float = 0.0,
    ) -> Dict[str, float]:
        """
        Kelly 公式: f* = (p * b - q) / b
        p = 胜率, b = 盈亏比 (avg_win/avg_loss), q = 1-p

        使用半 Kelly 以降低波动
        """
        kelly_weights: Dict[str, float] = {}

        for name in strategy_names:
            m = metrics.get(name, {})
            win_rate = _safe_float(m.get("win_rate"), 0.5)
            profit_factor = _safe_float(m.get("profit_factor"), 1.0)

            if win_rate <= 0 or profit_factor <= 0:
                kelly_weights[name] = 0.0
                continue

            # Kelly: f* = p - (1-p)/b, 其中 b ≈ profit_factor 近似盈亏比
            b = max(0.1, profit_factor)
            p = max(0.01, min(0.99, win_rate))
            q = 1 - p
            kelly = p - q / b

            # 限制 Kelly 范围
            kelly = max(0.0, min(self._kelly_max_fraction, kelly))

            # 半Kelly
            if self._use_half_kelly:
                kelly *= 0.5

            # 波动率调整
            vol = _safe_float(m.get("volatility_30d"), 0.02)
            if vol > 0:
                vol_adj = min(1.0, 0.02 / vol)  # 高波动→降低仓位
                kelly *= vol_adj

            # Forecasts may reduce Kelly exposure, but never increase it.
            # Use the worse of expected and bear-case PnL when both are available.
            if total_equity > 0:
                projected_pnl = _safe_float(m.get("projected_pnl_per_cycle"), 0.0)
                bear_case_pnl = _safe_float(m.get("bear_case_horizon"), projected_pnl)
                downside = min(projected_pnl, bear_case_pnl, 0.0)
                confidence = max(
                    0.0, min(1.0, _safe_float(m.get("projection_confidence"), 0.0))
                )
                risk_reference = max(total_equity * 0.05, 1.0)
                downside_severity = min(1.0, abs(downside) / risk_reference)
                kelly *= 1.0 - (0.5 * confidence * downside_severity)

            kelly_weights[name] = round(kelly, 6)

        return kelly_weights

    # ── Step 5: 分配优先级瀑布 ──────────────────────────────

    def _waterfall_allocation(
        self,
        plan: AllocationPlan,
        strategy_names: List[str],
        priorities: Dict[str, AllocationPriority],
        kelly_weights: Dict[str, float],
        current_weights: Dict[str, float],
        total_equity: float,
        regime: MarketRegime,
        used_margin_by_strategy: Dict[str, float] = None,
        strategy_metrics: Dict[str, Dict[str, float]] = None,
    ):
        """
        分配瀑布策略：
        1. CRITICAL: 已持仓保证金→直接从底仓扣除
        2. HIGH: Kelly权重→优先从底仓+加仓池分配
        3. MEDIUM: 剩余Kelly权重→从加仓池分配
        4. LOW: 最小权重→从加仓池剩余分配
        5. FROZEN: 不分配，已有仓位释放
        """
        base_pool = plan.pools["base"]
        addon_pool = plan.pools["addon"]
        reserve_pool = plan.pools["reserve"]

        used_margin_by_strategy = used_margin_by_strategy or {}
        strategy_metrics = strategy_metrics or {}

        # 按优先级排序
        priority_order = sorted(
            priorities.items(),
            key=lambda x: (x[1].value, -kelly_weights.get(x[0], 0))
        )

        remaining_base = base_pool.available
        remaining_addon = addon_pool.available

        for name, priority in priority_order:
            current_w = current_weights.get(name, 0)
            kelly_w = kelly_weights.get(name, 0)
            m = strategy_metrics.get(name, {})

            alloc = StrategyAllocation(
                name=name,
                priority=priority,
                current_weight=_safe_float(current_w, 0.0),
                used_margin=_safe_float(used_margin_by_strategy.get(name), 0.0),
                # ── 企业级：盈亏拆账 ──
                realized_pnl=_safe_float(m.get("realized_pnl"), 0.0),
                unrealized_pnl=_safe_float(m.get("unrealized_pnl"), 0.0),
                # ── 绩效快照 ──
                win_rate=_safe_float(m.get("win_rate"), 0.5),
                sharpe_ratio=_safe_float(m.get("sharpe_ratio"), 0.0),
                profit_factor=_safe_float(m.get("profit_factor"), 1.0),
                max_drawdown=_safe_float(m.get("max_drawdown"), 0.0),
                trade_count=int(_safe_float(m.get("trade_count"), 0)),
                consecutive_wins=int(_safe_float(m.get("consecutive_wins"), 0)),
                consecutive_losses=int(_safe_float(m.get("consecutive_losses"), 0)),
                volatility_30d=_safe_float(m.get("volatility_30d"), 0.0),
            )

            if priority == AllocationPriority.FROZEN:
                # 冻结/开仓门控：权重归零，不将该策略纳入资金分配。
                alloc.target_weight = 0
                alloc.allocated_capital = 0
                alloc.is_frozen = True
                alloc.is_open_blocked = bool(m.get("bear_case_fail_closed"))
                # 投影门控独立于基于历史绩效的冻结状态机。
                fs = self._freeze_state.get(name)
                if alloc.is_open_blocked:
                    alloc.freeze_reason = "bear_case_projection"
                elif isinstance(fs, dict) and fs.get("reason"):
                    alloc.freeze_reason = fs.get("reason", "")
                else:
                    alloc.freeze_reason = (
                        "consecutive_losses" if kelly_w == 0 else "high_drawdown"
                    )
                plan.strategy_allocations[name] = alloc
                continue

            # 确定目标权重
            if priority == AllocationPriority.HIGH:
                target_w = max(self._min_single_weight, min(self._max_single_weight, kelly_w))
            elif priority == AllocationPriority.MEDIUM:
                target_w = max(self._min_single_weight, min(self._max_single_weight * 0.7, kelly_w))
            else:  # LOW
                target_w = self._min_single_weight
                # 冻结策略缩量试探：权重再打折，控制试探风险
                if name in self._probing_strategies:
                    target_w = self._min_single_weight * self._probe_weight_ratio

            # 市场状态微调
            target_w = self._regime_weight_adj(target_w, regime, priority)

            # 分配资金：优先底仓→加仓池→风控池
            needed = total_equity * target_w
            allocated = 0.0

            # 底仓分配（策略只要有交易历史就优先从底仓出）
            if remaining_base > 0 and priority != AllocationPriority.LOW:
                from_base = min(needed, remaining_base)
                allocated += from_base
                remaining_base -= from_base
                base_pool.allocated += from_base

            # 加仓池分配
            remaining_needed = needed - allocated
            if remaining_needed > 0 and remaining_addon > 0:
                from_addon = min(remaining_needed, remaining_addon)
                allocated += from_addon
                remaining_addon -= from_addon
                addon_pool.allocated += from_addon

            # 风控池仅在紧急情况使用（策略优先级 HIGH 且资金不足）
            remaining_needed = needed - allocated
            if remaining_needed > 0 and priority == AllocationPriority.HIGH:
                from_reserve = min(remaining_needed, reserve_pool.available * 0.3)
                allocated += from_reserve
                reserve_pool.allocated += from_reserve
                reserve_pool.available -= from_reserve

            alloc.target_weight = target_w
            alloc.allocated_capital = allocated
            alloc.available = max(0.0, allocated - alloc.used_margin)
            alloc.kelly_fraction = kelly_w
            alloc.half_kelly = kelly_w * 0.5
            alloc.is_probing = name in self._probing_strategies

            plan.strategy_allocations[name] = alloc

        # 更新池可用资金
        base_pool.available = remaining_base
        addon_pool.available = remaining_addon

        # 归一化权重
        total_w = sum(a.target_weight for a in plan.strategy_allocations.values())
        if total_w > 0:
            for alloc in plan.strategy_allocations.values():
                alloc.target_weight /= total_w

    def _regime_weight_adj(self, weight: float, regime: MarketRegime,
                           priority: AllocationPriority) -> float:
        """市场状态对权重的微调"""
        adjustments = {
            MarketRegime.TRENDING_UP:     1.05,
            MarketRegime.TRENDING_DOWN:   0.85,
            MarketRegime.RANGING:         0.95,
            MarketRegime.HIGH_VOLATILITY: 0.70,
            MarketRegime.LOW_VOLATILITY:  1.10,
            MarketRegime.UNKNOWN:         1.00,
        }
        adj = adjustments.get(regime, 1.0)
        # 高优先级策略在市场不利时减少调整幅度
        if priority == AllocationPriority.HIGH and adj < 1.0:
            adj = 1.0 - (1.0 - adj) * 0.5
        return weight * adj

    # ── Step 6: 权重变化 ─────────────────────────────────────

    def _compute_changes(self, plan: AllocationPlan, current_weights: Dict[str, float]):
        """计算需要调整的权重变化"""
        for name, alloc in plan.strategy_allocations.items():
            current_w = _safe_float(current_weights.get(name), 0.0)
            diff = alloc.target_weight - current_w

            # 限制单次变更幅度
            if abs(diff) > self._max_weight_change:
                diff = self._max_weight_change if diff > 0 else -self._max_weight_change

            if abs(diff) > 0.001:
                plan.allocation_changes[name] = round(diff, 4)

    # ── Step 7: 闲置资金检测（企业级：时间加权） ─────────────

    def _detect_idle_cash(self, plan: AllocationPlan):
        """检测各池闲置资金，带时间加权追踪"""
        total_idle = 0.0
        for pool in plan.pools.values():
            pool_free = pool.available - pool.allocated
            if pool_free > 0:
                total_idle += pool_free

        plan.idle_cash = max(0, total_idle)

        # 时间加权：追踪闲置持续时间
        idle_pct = plan.idle_cash / max(plan.total_equity, 1)
        if idle_pct > self._idle_sweep_threshold:
            if self._idle_start_time is None:
                self._idle_start_time = datetime.now()
            plan.idle_duration_minutes = (datetime.now() - self._idle_start_time).total_seconds() / 60
        else:
            self._idle_start_time = None
            plan.idle_duration_minutes = 0

        # 分级告警
        if idle_pct > 0.30:
            plan.warnings.append(
                f"CRITICAL: {plan.idle_cash:.0f} USDT idle ({idle_pct:.1%}) for {plan.idle_duration_minutes:.0f}min"
            )
        elif idle_pct > self._idle_sweep_threshold:
            plan.warnings.append(
                f"High idle cash: {plan.idle_cash:.0f} USDT ({idle_pct:.1%}) for {plan.idle_duration_minutes:.0f}min"
            )

    # ── Step 8: 资金效率评估（企业级：每策略利用率） ─────────

    def _evaluate_efficiency(self, plan: AllocationPlan,
                             strategy_metrics: Dict[str, Dict[str, float]] = None):
        """评估资金使用效率，含每策略利用率追踪、风险调整效率与机会成本"""
        strategy_metrics = strategy_metrics or {}
        total_allocated = sum(
            a.allocated_capital
            for a in plan.strategy_allocations.values()
        )
        total_used = sum(
            a.used_margin
            for a in plan.strategy_allocations.values()
        )

        if plan.total_equity > 0:
            plan.capital_efficiency = total_allocated / plan.total_equity
            plan.deployment_ratio = total_used / max(1, total_allocated)

        # 企业级：每策略利用率
        for name, alloc in plan.strategy_allocations.items():
            if alloc.allocated_capital > 0:
                alloc.capital_utilization_pct = alloc.used_margin / alloc.allocated_capital
                alloc.idle_capital = max(0, alloc.allocated_capital - alloc.used_margin)
            plan.strategy_utilization[name] = alloc.capital_utilization_pct

        # ── 企业级：风险调整资金效率（Sharpe加权） ──
        self._compute_risk_adjusted_efficiency(plan, strategy_metrics)

        # ── 企业级：闲置资金机会成本（日化 0.05%） ──
        plan.opportunity_cost = plan.idle_cash * self._opportunity_cost_daily

        # 集中度：前2名策略占比
        sorted_allocs = sorted(
            plan.strategy_allocations.values(),
            key=lambda a: a.allocated_capital, reverse=True
        )
        if len(sorted_allocs) >= 2 and plan.total_equity > 0:
            top2 = sorted_allocs[0].allocated_capital + sorted_allocs[1].allocated_capital
            plan.concentration_ratio = top2 / plan.total_equity

        if plan.concentration_ratio > self._max_concentration_ratio:
            plan.warnings.append(
                f"High concentration: top2={plan.concentration_ratio:.1%}"
            )

        # 企业级：最低资金效率硬约束（FORCE_REBALANCE 真正执行）
        if plan.capital_efficiency < self._min_capital_efficiency and plan.total_equity > 100:
            plan.warnings.append(
                f"Capital efficiency {plan.capital_efficiency:.1%} below floor {self._min_capital_efficiency:.1%}"
            )
            plan.auto_actions.append("FORCE_REBALANCE: efficiency below floor")
            # 真正执行：高优先级策略标记加仓，推动资金重新部署
            for name, alloc in plan.strategy_allocations.items():
                if alloc.priority == AllocationPriority.HIGH and not alloc.is_frozen:
                    plan.allocation_changes[name] = min(
                        self._max_weight_change, alloc.target_weight * 0.10
                    )

    def _compute_risk_adjusted_efficiency(self, plan: AllocationPlan,
                                          strategy_metrics: Dict[str, Dict[str, float]]):
        """风险调整资金效率 = 资金效率 × Sharpe加权因子。

        正Sharpe策略提升效率评分，负Sharpe策略拉低评分，
        使资金效率指标能区分「高效盈利」与「低效占用」。
        """
        total_allocated = sum(
            a.allocated_capital for a in plan.strategy_allocations.values()
        )
        if total_allocated <= 0:
            plan.risk_adjusted_efficiency = plan.capital_efficiency
            return

        # 按分配资金加权的 Sharpe
        weighted_sharpe = 0.0
        for name, alloc in plan.strategy_allocations.items():
            if alloc.allocated_capital <= 0:
                continue
            sharpe = _safe_float(strategy_metrics.get(name, {}).get("sharpe_ratio"), 0.0)
            weight = alloc.allocated_capital / total_allocated
            weighted_sharpe += sharpe * weight

        # Sharpe 映射到效率因子：Sharpe>1 → 1.2，0~1 → 1.0，<0 → 0.6~1.0
        if weighted_sharpe >= 1.0:
            sharpe_factor = 1.2
        elif weighted_sharpe >= 0:
            sharpe_factor = 1.0
        else:
            sharpe_factor = max(0.5, 1.0 + weighted_sharpe * 0.4)

        plan.risk_adjusted_efficiency = min(
            1.0, plan.capital_efficiency * sharpe_factor
        )

    # ── Step 9: 盈亏驱动池间流动 ─────────────────────────────

    def _compute_pool_flows(self, plan: AllocationPlan,
                             strategy_metrics: Dict[str, Dict[str, float]]):
        """
        盈利→流入加仓池；亏损→收缩底仓→流入风控池
        """
        total_profit = sum(
            _safe_float(m.get("total_pnl"), 0)
            for m in strategy_metrics.values()
        )

        if total_profit > 0:
            # 盈利：50% 流入加仓池
            flow = total_profit * self._profit_to_addon
            plan.pool_flows["profit_to_addon"] = flow

            # 限制加仓池不超过最大比例
            addon_pool = plan.pools.get("addon")
            if addon_pool:
                current_addon_ratio = addon_pool.total_capital / max(plan.total_equity, 1)
                if current_addon_ratio >= self._max_addon_ratio:
                    plan.pool_flows["profit_to_addon"] = 0
                    plan.recommendations.append("Addon pool at max ratio; directing profit to reserve")

        elif total_profit < 0:
            # 亏损：收缩底仓 30%
            shrink = abs(total_profit) * self._loss_shrink
            plan.pool_flows["loss_shrink_base"] = -shrink

            if abs(total_profit) / max(plan.total_equity, 1) > 0.05:
                plan.recommendations.append(
                    f"Significant loss detected ({total_profit:.0f} USDT); reducing base exposure"
                )

    # ── Step 10: 生成建议 ─────────────────────────────────────

    def _generate_recommendations(self, plan: AllocationPlan):
        """生成资金分配建议"""
        # 闲置资金建议
        if plan.idle_cash > plan.total_equity * 0.10:
            plan.recommendations.append(
                f"Deploy {plan.idle_cash:.0f} USDT idle cash to active strategies"
            )

        # 集中度建议
        if plan.concentration_ratio > 0.50:
            plan.recommendations.append(
                "Consider diversifying to reduce concentration risk"
            )

        # Kelly 建议
        high_kelly = [
            (name, alloc.kelly_fraction)
            for name, alloc in plan.strategy_allocations.items()
            if alloc.kelly_fraction > 0.15 and not alloc.is_frozen
        ]
        if high_kelly:
            top = sorted(high_kelly, key=lambda x: x[1], reverse=True)[:2]
            plan.recommendations.append(
                f"High Kelly strategies: {', '.join(f'{n}({k:.1%})' for n, k in top)} — consider increasing allocation"
            )

        # 冻结策略建议
        frozen = [
            name for name, alloc in plan.strategy_allocations.items()
            if alloc.is_frozen
        ]
        if frozen:
            plan.recommendations.append(
                f"Frozen strategies: {', '.join(frozen)} — review and possibly retire"
            )

    # ── 跨策略资金共享 ────────────────────────────────────────

    async def borrow_from_idle(
        self,
        plan: AllocationPlan,
        target_strategy: str,
        amount: float,
    ) -> Tuple[bool, str]:
        """
        从闲置策略借调资金

        当高优先级策略需要资金时，从低优先级闲置策略借用。
        """
        async with self._lock:
            amount = _safe_float(amount, 0.0)
            if target_strategy not in plan.strategy_allocations:
                return False, f"Strategy {target_strategy} not in plan"

            target_alloc = plan.strategy_allocations[target_strategy]
            if target_alloc.priority.value > AllocationPriority.MEDIUM.value:
                return False, "Target strategy priority too low"

            # 寻找可借用来源
            borrowed = 0.0
            for name, alloc in sorted(
                plan.strategy_allocations.items(),
                key=lambda x: (-x[1].priority.value, x[1].available),
            ):
                if name == target_strategy:
                    continue
                if alloc.is_frozen:
                    continue

                # 从低优先级/多余可用资金借用
                available_to_borrow = alloc.available - alloc.used_margin
                if available_to_borrow <= 0:
                    continue

                borrow_amount = min(available_to_borrow, amount - borrowed)
                if borrow_amount <= 0:
                    continue

                alloc.allocated_capital -= borrow_amount
                alloc.available -= borrow_amount
                target_alloc.allocated_capital += borrow_amount
                target_alloc.available += borrow_amount
                borrowed += borrow_amount

                if borrowed >= amount:
                    break

            return borrowed > 0, f"Borrowed {borrowed:.0f} USDT from idle strategies"

    # ── 动态最小资金验证 ──────────────────────────────────────

    def validate_minimum_capital(
        self,
        plan: AllocationPlan,
        min_capital_per_strategy: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        验证每策略是否满足最小资金要求

        不满足时标记警告并提出调整方案。
        """
        issues = {}
        for name, alloc in plan.strategy_allocations.items():
            if alloc.is_frozen:
                continue

            min_cap = _safe_float(min_capital_per_strategy.get(name), 200)
            if alloc.allocated_capital < min_cap:
                issues[name] = {
                    "current": alloc.allocated_capital,
                    "required": min_cap,
                    "shortfall": min_cap - alloc.allocated_capital,
                    "recommendation": (
                        "Increase allocation or merge with another strategy"
                    ),
                }

        return {
            "all_pass": len(issues) == 0,
            "issues": issues,
        }

    # ── 连续盈利/亏损检测 ────────────────────────────────────

    def check_consecutive_streaks(
        self,
        strategy_metrics: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """检测连续盈利/亏损天数，触发再分配建议"""
        streaks = {}
        for name, m in strategy_metrics.items():
            consecutive_wins = _safe_float(m.get("consecutive_wins"), 0)
            consecutive_losses = _safe_float(m.get("consecutive_losses"), 0)

            action = "hold"
            if consecutive_wins >= self._consecutive_profit_days:
                action = "increase"     # 连续盈利→增加分配
            elif consecutive_losses >= self._consecutive_loss_days:
                action = "decrease"     # 连续亏损→减少分配

            streaks[name] = {
                "consecutive_wins": consecutive_wins,
                "consecutive_losses": consecutive_losses,
                "action": action,
            }

        return streaks

    # ── 快照与查询 ────────────────────────────────────────────

    def get_last_plan(self) -> Optional[Dict[str, Any]]:
        """获取最近分配方案"""
        if self._last_plan:
            return self._last_plan.to_dict()
        return None

    def get_pool_status(self) -> Dict[str, Any]:
        """获取资金池状态"""
        if not self._last_plan:
            return {"error": "No allocation plan computed"}
        return {
            k: {
                "total": round(v.total_capital, 2),
                "used": round(v.used, 2),
                "available": round(v.available, 2),
                "free": round(v.free, 2),
                "usage_pct": round(v.usage_pct, 4),
            }
            for k, v in self._last_plan.pools.items()
        }

    def get_strategy_allocations(self) -> Dict[str, Dict[str, Any]]:
        """获取策略资金分配详情"""
        if not self._last_plan:
            return {}
        return {
            k: {
                "target_weight": round(v.target_weight, 4),
                "allocated_capital": round(v.allocated_capital, 2),
                "used_margin": round(v.used_margin, 2),
                "available": round(v.available, 2),
                "kelly_fraction": round(v.kelly_fraction, 4),
                "priority": v.priority.name,
                "is_frozen": v.is_frozen,
                "is_probing": v.is_probing,
                "freeze_reason": v.freeze_reason,
            }
            for k, v in self._last_plan.strategy_allocations.items()
        }

    def get_freeze_state(self) -> Dict[str, Dict[str, Any]]:
        """返回冻结状态机快照，供 AGI 编排器感知「观察期自动解冻 / 缩量试探」状态。"""
        now = time.time()
        result: Dict[str, Dict[str, Any]] = {}
        for name, state in self._freeze_state.items():
            frozen_at = _safe_float(state.get("frozen_at"), now)
            probing = bool(state.get("probing", False))
            probe_attempts = int(state.get("probe_attempts", 0))
            observe_seconds = self._freeze_observe_seconds

            if probing:
                started = _safe_float(state.get("probe_started_at"), now)
                remaining = max(0.0, observe_seconds - (now - started))
            else:
                remaining = max(0.0, observe_seconds - (now - frozen_at))

            result[name] = {
                "reason": state.get("reason", ""),
                "probing": probing,
                "probe_attempts": probe_attempts,
                "probe_max_attempts": self._probe_max_attempts,
                "observe_seconds": round(observe_seconds, 1),
                "remaining_seconds": round(remaining, 1),
                "permanently_frozen": (
                    probing and probe_attempts >= self._probe_max_attempts
                ),
            }
        return result

    # ── 企业级 Step 11: 闲置资金自动归集 ─────────────────────

    def _auto_sweep_idle(self, plan: AllocationPlan):
        """闲置资金自动归集：底仓→加仓→风控池三级瀑布

        规则：
        - 底仓池闲置 > 20% 且持续 > 30分钟 → 50% 流入加仓池
        - 加仓池闲置 > 20% 且持续 > 30分钟 → 50% 流入风控池
        - 归集后底仓/加仓不低于各自最低比例
        """
        idle_pct = plan.idle_cash / max(plan.total_equity, 1)
        if idle_pct <= self._idle_sweep_threshold:
            return
        if plan.idle_duration_minutes < self._idle_sweep_min_duration:
            return

        base_pool = plan.pools.get("base")
        addon_pool = plan.pools.get("addon")
        reserve_pool = plan.pools.get("reserve")

        # 底仓→加仓
        if base_pool and base_pool.free > base_pool.total_capital * self._idle_sweep_threshold:
            sweep = base_pool.free * 0.50
            min_base = plan.total_equity * self._min_base_ratio
            if base_pool.total_capital - sweep >= min_base:
                base_pool.total_capital -= sweep
                base_pool.available -= sweep
                if addon_pool:
                    addon_pool.total_capital += sweep
                    addon_pool.available += sweep
                plan.pool_flows["base_to_addon_sweep"] = sweep
                plan.auto_actions.append(
                    f"AUTO_SWEEP: {sweep:.0f} USDT base→addon (idle {plan.idle_duration_minutes:.0f}min)"
                )
                logger.info(f"Auto-sweep: {sweep:.0f} USDT from base to addon pool")

        # 加仓→风控
        if addon_pool and addon_pool.free > addon_pool.total_capital * self._idle_sweep_threshold:
            sweep = addon_pool.free * 0.50
            if addon_pool.total_capital - sweep > 0:
                addon_pool.total_capital -= sweep
                addon_pool.available -= sweep
                if reserve_pool:
                    reserve_pool.total_capital += sweep
                    reserve_pool.available += sweep
                plan.pool_flows["addon_to_reserve_sweep"] = sweep
                plan.auto_actions.append(
                    f"AUTO_SWEEP: {sweep:.0f} USDT addon→reserve (idle {plan.idle_duration_minutes:.0f}min)"
                )
                logger.info(f"Auto-sweep: {sweep:.0f} USDT from addon to reserve pool")

    # ── 企业级 Step 12: 低利用率策略资金回收 ─────────────────

    def _reclaim_underutilized(self, plan: AllocationPlan,
                               strategy_metrics: Dict[str, Dict[str, float]] = None):
        """回收长期低利用率策略的闲置资金

        规则：
        - 策略资金利用率 < 50% 且分配资金 > 0 → 回收 30% 分配资金
        - 利用率 < 20% 且非正收益策略 → 全额回收（保留最低额度）
        - 正收益策略保护：Sharpe > 0 或近期正收益，即使利用率低也只回收 30%
        - 机会成本回灌：闲置资金机会成本高的策略优先回收
        - 回收资金流入加仓池
        - 保护：每个策略至少保留 min_single_weight * total_equity
        """
        strategy_metrics = strategy_metrics or {}
        reclaimed_total = 0.0
        reclaim_list = []

        for name, alloc in plan.strategy_allocations.items():
            if alloc.is_frozen:
                continue
            if alloc.allocated_capital <= 0:
                continue

            util_pct = _safe_float(alloc.capital_utilization_pct, 0.0)
            min_keep = plan.total_equity * self._min_single_weight

            # 正收益策略保护：Sharpe > 0 或近期正收益
            m = strategy_metrics.get(name, {})
            is_positive_return = (
                _safe_float(m.get("sharpe_ratio"), 0) > 0 or _safe_float(m.get("total_pnl"), 0) > 0
            )

            if util_pct < 0.20 and alloc.allocated_capital > min_keep:
                if self._positive_return_protection and is_positive_return:
                    # 正收益策略：即使利用率低也只部分回收
                    reclaim = alloc.allocated_capital * 0.30
                    level = "PARTIAL_PROTECTED"
                else:
                    # 极低利用率：全额回收（保留最低额度）
                    reclaim = alloc.allocated_capital - min_keep
                    level = "FULL"
            elif util_pct < self._utilization_reclaim_threshold:
                # 低利用率：回收 30%
                reclaim = alloc.allocated_capital * 0.30
                level = "PARTIAL"
            else:
                continue

            reclaim = min(reclaim, alloc.available)  # 不能超过可用资金
            if reclaim <= 0:
                continue

            # 机会成本回灌：记录回收决策依据
            idle_oc = alloc.idle_capital * self._opportunity_cost_daily

            alloc.allocated_capital -= reclaim
            alloc.available -= reclaim
            reclaimed_total += reclaim
            reclaim_list.append((name, level, reclaim, util_pct, idle_oc))

        if reclaim_list:
            # 回收资金流入加仓池
            addon_pool = plan.pools.get("addon")
            if addon_pool:
                addon_pool.total_capital += reclaimed_total
                addon_pool.available += reclaimed_total

            for name, level, reclaim, util_pct, idle_oc in reclaim_list:
                plan.auto_actions.append(
                    f"RECLAIM_{level}: {name} reclaim {reclaim:.0f} USDT "
                    f"(util={util_pct:.1%}, idle_oc={idle_oc:.2f}/day)"
                )
                logger.info(
                    f"Reclaim {level}: {name} {reclaim:.0f} USDT "
                    f"(utilization={util_pct:.1%}, opportunity_cost={idle_oc:.2f}/day)"
                )

    # ── 企业级 Step 13: 应急储备检查 ─────────────────────────

    def _emergency_reserve_check(self, plan: AllocationPlan,
                                 strategy_metrics: Dict[str, Dict[str, float]] = None):
        """风控池应急补充检查

        规则：
        - 风控池 < 总权益 * emergency_reserve_threshold(10%) → 从底仓/加仓各补充 50% 缺口
        - 连续亏损（任意策略 consecutive_losses >= 2）→ 额外补充 50% 底仓资金到风控池
        - 风控池 > 25% → 超出部分回退到底仓
        """
        strategy_metrics = strategy_metrics or {}
        reserve_pool = plan.pools.get("reserve")
        if not reserve_pool:
            return

        # 检测连续亏损：任意策略连续亏损笔数达到阈值
        consecutive_loss = any(
            _safe_float(m.get("consecutive_losses"), 0) >= self._consecutive_loss_days
            for m in strategy_metrics.values()
        )

        reserve_ratio = reserve_pool.total_capital / max(plan.total_equity, 1)
        target_reserve = plan.total_equity * self._emergency_reserve_threshold

        if reserve_ratio < self._emergency_reserve_threshold:
            # 缺口：需补充到目标比例
            shortage = target_reserve - reserve_pool.total_capital
            from_base = shortage * 0.50
            from_addon = shortage * 0.50

            # 连续亏损：底仓额外补充 50%
            extra_from_base = 0.0
            if consecutive_loss:
                extra_from_base = from_base * 0.50

            base_pool = plan.pools.get("base")
            addon_pool = plan.pools.get("addon")

            actual_base = 0.0
            actual_addon = 0.0

            total_from_base = from_base + extra_from_base
            if base_pool and base_pool.free >= total_from_base:
                base_pool.total_capital -= total_from_base
                base_pool.available -= total_from_base
                actual_base = total_from_base
            elif base_pool and base_pool.free >= from_base:
                base_pool.total_capital -= from_base
                base_pool.available -= from_base
                actual_base = from_base

            if addon_pool and addon_pool.free >= from_addon:
                addon_pool.total_capital -= from_addon
                addon_pool.available -= from_addon
                actual_addon = from_addon

            total_replenished = actual_base + actual_addon
            if total_replenished > 0:
                reserve_pool.total_capital += total_replenished
                reserve_pool.available += total_replenished
                plan.pool_flows["emergency_reserve_replenish"] = total_replenished
                action = (
                    f"EMERGENCY_REPLENISH: {total_replenished:.0f} USDT to reserve "
                    f"(reserve={reserve_ratio:.1%}, target={self._emergency_reserve_threshold:.1%}"
                )
                if consecutive_loss:
                    action += ", consecutive_loss_boost"
                action += ")"
                plan.auto_actions.append(action)
                logger.warning(
                    f"Emergency reserve replenish: {total_replenished:.0f} USDT "
                    f"(from base={actual_base:.0f}, addon={actual_addon:.0f}, "
                    f"consecutive_loss={consecutive_loss})"
                )

        elif reserve_ratio > 0.25:
            # 风控池过大：超出部分回退到底仓
            excess = reserve_pool.total_capital - plan.total_equity * 0.25
            base_pool = plan.pools.get("base")
            if base_pool and excess > 0:
                reserve_pool.total_capital -= excess
                reserve_pool.available -= excess
                base_pool.total_capital += excess
                base_pool.available += excess
                plan.pool_flows["reserve_excess_return"] = excess
                plan.auto_actions.append(
                    f"RESERVE_EXCESS_RETURN: {excess:.0f} USDT back to base"
                )


# ═══════════════════════════════════════════════════════════════
# 单例工厂
# ═══════════════════════════════════════════════════════════════

_dynamic_allocator: Optional[DynamicAllocator] = None


def get_dynamic_allocator(config: Dict[str, Any] = None) -> DynamicAllocator:
    """获取全局单例 DynamicAllocator"""
    global _dynamic_allocator
    if _dynamic_allocator is None and config is not None:
        _dynamic_allocator = DynamicAllocator(config)
    return _dynamic_allocator


def reset_dynamic_allocator():
    """重置单例（测试用）"""
    global _dynamic_allocator
    _dynamic_allocator = None


# ═══════════════════════════════════════════════════════════════
# 波动率目标管理器
# ═══════════════════════════════════════════════════════════════

class VolatilityTargeter:
    """
    波动率目标管理器

    功能：
    - 目标波动率设定（例如年化20%）
    - 实时波动率估计（EWMA + GARCH简化近似）
    - 仓位缩放：current_vol / target_vol → leverage adjustment
    - 波动率上限：不允许超过max_vol
    - 波动率下限：不主动加杠杆追低波动
    - 波动率预测区间（上下界）
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()
        self._target_vol = _safe_float(config.get("target_volatility"), 0.20)        # 年化目标波动率
        self._max_vol = _safe_float(config.get("max_volatility"), 0.50)              # 波动率上限
        self._min_vol = _safe_float(config.get("min_volatility"), 0.05)              # 波动率下限
        self._ewma_lambda = _safe_float(config.get("ewma_lambda"), 0.94)             # EWMA 衰减因子
        self._garch_omega = _safe_float(config.get("garch_omega"), 0.00001)
        self._garch_alpha = _safe_float(config.get("garch_alpha"), 0.05)
        self._garch_beta = _safe_float(config.get("garch_beta"), 0.90)
        self._forecast_horizon = int(_safe_float(config.get("forecast_horizon"), 21))  # 预测天数
        self._confidence_z = _safe_float(config.get("confidence_z"), 1.96)            # 95% 置信区间
        self._current_vol = 0.0
        self._latest_estimate: Dict[str, float] = {}

        logger.info(
            f"VolatilityTargeter initialized: "
            f"target={self._target_vol:.0%}, max={self._max_vol:.0%}, lambda={self._ewma_lambda}"
        )

    def estimate_volatility(self, returns: List[float]) -> Dict:
        """
        估计当前波动率

        Returns:
            {current, ewma, predicted, upper, lower}
        """
        clean = [r for r in returns if _is_finite(r)] if returns else []
        if len(clean) < 2:
            return {
                "current": self._target_vol, "ewma": self._target_vol,
                "predicted": self._target_vol, "upper": self._max_vol, "lower": self._min_vol,
            }

        arr = np.array(clean, dtype=np.float64)
        # 年化系数（假设日收益）
        annual_factor = np.sqrt(365)

        # ── EWMA 波动率 ──
        squared = arr ** 2
        ewma_var = squared[0]
        for i in range(1, len(squared)):
            ewma_var = self._ewma_lambda * ewma_var + (1 - self._ewma_lambda) * squared[i]
        ewma_vol = np.sqrt(max(ewma_var, 1e-12)) * annual_factor

        # ── GARCH(1,1) 简化近似 ──
        long_run_var = np.var(arr) if len(arr) > 1 else 0.0001
        garch_var = self._garch_omega
        if self._garch_alpha + self._garch_beta < 1:
            garch_var = self._garch_omega / (1 - self._garch_alpha - self._garch_beta)
        garch_var = max(garch_var, long_run_var, 1e-12)
        garch_vol = np.sqrt(garch_var) * annual_factor

        # ── 混合估计 ──
        current_vol = 0.6 * ewma_vol + 0.4 * garch_vol

        # ── 预测区间 ──
        se = current_vol / np.sqrt(len(clean))
        upper = min(self._max_vol, current_vol + self._confidence_z * se)
        lower = max(self._min_vol, current_vol - self._confidence_z * se)

        self._current_vol = current_vol
        self._latest_estimate = {
            "current": round(current_vol, 6),
            "ewma": round(ewma_vol, 6),
            "predicted": round(garch_vol, 6),
            "upper": round(upper, 6),
            "lower": round(lower, 6),
        }
        return self._latest_estimate

    def compute_scale_factor(
        self, current_vol: float, target_vol: float,
        current_leverage: float, max_leverage: float,
    ) -> Dict:
        """
        计算仓位缩放因子

        仅在波动率高于目标时降低杠杆，
        不在波动率低于目标时主动加杠杆。

        Returns:
            {scale_factor, target_position, max_allowed_leverage, reason}
        """
        current_vol = _safe_float(current_vol, 0.0)
        target_vol = _safe_float(target_vol, 0.0)
        current_leverage = _safe_float(current_leverage, 1.0)
        max_leverage = _safe_float(max_leverage, 1.0)

        if current_vol <= 0 or target_vol <= 0:
            return {"scale_factor": 1.0, "target_position": current_leverage,
                    "max_allowed_leverage": max_leverage, "reason": "invalid_vol"}

        raw_scale = target_vol / max(current_vol, 1e-6)
        reason = "hold"

        if current_vol > self._max_vol:
            # 波动率突破上限：强制降仓
            scale_factor = min(0.5, raw_scale)
            reason = f"vol_breach({current_vol:.2%} > {self._max_vol:.2%})"
        elif current_vol > target_vol:
            # 波动率高于目标：按比例降仓
            scale_factor = min(raw_scale, 1.0)
            reason = f"vol_above_target({current_vol:.2%} > {target_vol:.2%})"
        else:
            # 波动率低于目标：不主动加杠杆
            scale_factor = min(1.0, raw_scale)
            reason = f"vol_below_target({current_vol:.2%} < {target_vol:.2%})"

        scale_factor = max(0.1, min(1.0, scale_factor))
        target_position = current_leverage * scale_factor
        max_allowed = min(max_leverage, target_position * 1.2)

        return {
            "scale_factor": round(scale_factor, 4),
            "target_position": round(target_position, 4),
            "max_allowed_leverage": round(max_allowed, 4),
            "reason": reason,
        }

    def check_volatility_breach(self, current_vol: float) -> Tuple[bool, str]:
        """检查波动率是否突破上限"""
        current_vol = _safe_float(current_vol, 0.0)
        if current_vol > self._max_vol:
            return True, f"Volatility {current_vol:.2%} exceeds max {self._max_vol:.2%}; reduce positions immediately"
        if current_vol > self._target_vol * 1.5:
            return True, f"Volatility {current_vol:.2%} exceeds 1.5x target {self._target_vol:.2%}; caution advised"
        return False, "volatility within acceptable range"

    def get_volatility_summary(self) -> Dict:
        """获取波动率摘要"""
        return {
            "target_volatility": self._target_vol,
            "max_volatility": self._max_vol,
            "current_volatility": round(self._current_vol, 6),
            "latest_estimate": self._latest_estimate,
            "ewma_lambda": self._ewma_lambda,
        }


# ═══════════════════════════════════════════════════════════════
# 自适应 Kelly 仓位优化
# ═══════════════════════════════════════════════════════════════

class AdaptiveKelly:
    """
    自适应Kelly仓位优化

    功能：
    - 基础Kelly: f* = (p*b - q) / b
    - 市场状态调整：不同regime下的Kelly系数
    - 回撤保护：回撤中自动降低Kelly分数
    - 连续盈亏调整：连赢→增加，连输→减少
    - 样本量调整：交易数不足时缩水Kelly
    - Modified Kelly: f* = (μ - r_f) / σ^2 (连续版本)
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()
        self._max_kelly = _safe_float(config.get("max_kelly_fraction"), 0.25)
        self._default_fraction = _safe_float(config.get("default_kelly_fraction"), 0.5)
        self._min_trade_count = int(_safe_float(config.get("min_trade_count_kelly"), 20))

        # 不同市场状态下的 Kelly 乘数
        self._regime_multipliers: Dict[str, float] = {
            "trending_up": _safe_float(config.get("kelly_trending_up"), 1.10),
            "trending_down": _safe_float(config.get("kelly_trending_down"), 0.60),
            "ranging": _safe_float(config.get("kelly_ranging"), 0.90),
            "high_volatility": _safe_float(config.get("kelly_high_vol"), 0.50),
            "low_volatility": _safe_float(config.get("kelly_low_vol"), 1.00),
            "unknown": _safe_float(config.get("kelly_unknown"), 0.80),
        }

        # 回撤惩罚曲线参数
        self._dd_penalty_start = _safe_float(config.get("kelly_dd_start"), 0.05)    # 回撤5%开始惩罚
        self._dd_penalty_max = _safe_float(config.get("kelly_dd_max"), 0.30)         # 回撤30%惩罚最大
        self._dd_min_multiplier = _safe_float(config.get("kelly_dd_min"), 0.30)      # 最低保留30% Kelly

        # 连续盈亏调整参数
        self._streak_win_step = _safe_float(config.get("kelly_streak_win_step"), 0.05)
        self._streak_loss_step = _safe_float(config.get("kelly_streak_loss_step"), 0.08)
        self._streak_max_adj = _safe_float(config.get("kelly_streak_max_adj"), 0.30)

        # P31: 企业级估计误差修正（凯利公式对胜率/赔率估计误差极度敏感，
        # 点估计会导致过度下注，企业级实现用贝叶斯收缩 + 赔率收缩替代点估计）
        self._use_wilson_lcb = config.get("kelly_use_wilson_lcb", False)      # 可选：Wilson置信下界（更保守）
        self._wilson_z = _safe_float(config.get("kelly_wilson_z"), 1.645)                  # Wilson置信度 z 值
        self._win_prior_strength = _safe_float(config.get("kelly_win_rate_prior_strength"), 40.0)  # 胜率先验强度（等效α+β样本量，先验50%）
        self._b_shrinkage = _safe_float(config.get("kelly_b_shrinkage_strength"), 20.0)    # 赔率收缩强度（等效先验样本量）
        self._use_semivariance = config.get("kelly_use_semivariance", True)   # 连续Kelly使用下行半方差
        self._skew_penalty = _safe_float(config.get("kelly_skew_penalty_strength"), 0.5)   # 负偏度惩罚强度

        self._latest_summary: Dict[str, Any] = {}

        logger.info(
            f"AdaptiveKelly initialized: max={self._max_kelly:.0%}, "
            f"default_fraction={self._default_fraction:.0%}"
        )

    def compute_kelly(
        self, win_rate: float, avg_win: float, avg_loss: float,
        regime: str = "unknown", drawdown: float = 0,
        consecutive_wins: int = 0, consecutive_losses: int = 0,
        trade_count: int = 0,
    ) -> Dict:
        """
        计算自适应 Kelly 仓位

        f* = (p*b - q) / b
        """
        win_rate = _safe_float(win_rate, 0.0)
        avg_win = _safe_float(avg_win, 0.0)
        avg_loss = _safe_float(avg_loss, 0.0)
        drawdown = _safe_float(drawdown, 0.0)
        consecutive_wins = int(_safe_float(consecutive_wins, 0))
        consecutive_losses = int(_safe_float(consecutive_losses, 0))
        trade_count = int(_safe_float(trade_count, 0))

        avg_loss_abs = abs(avg_loss) if avg_loss != 0 else 0.01

        # ── 基础 Kelly（P31: 置信下界胜率 + 赔率收缩）──
        if win_rate <= 0 or avg_win <= 0:
            base_kelly = 0.0
            wilson_win_rate = win_rate
            shrunk_b = 0.0
        else:
            b = avg_win / avg_loss_abs
            # 赔率收缩：向盈亏平衡先验(b=1)收缩，样本越少越保守
            shrunk_b = self._shrink_odds(b, trade_count, self._b_shrinkage)
            # 胜率估计误差修正：默认贝叶斯收缩（向50%先验），可选Wilson下界（更保守）
            if self._use_wilson_lcb and trade_count > 0:
                wins = win_rate * trade_count
                wilson_win_rate = self._wilson_lower_bound(wins, trade_count, self._wilson_z)
            else:
                wilson_win_rate = self._shrink_win_rate(win_rate, trade_count, self._win_prior_strength)
            p = max(0.01, min(0.99, wilson_win_rate))
            q = 1 - p
            base_kelly = (p * shrunk_b - q) / max(shrunk_b, 0.01)
            base_kelly = max(0.0, min(self._max_kelly * 2, base_kelly))

        # ── 市场状态调整 ──
        regime_mult = self.compute_regime_kelly_multiplier(regime)
        kelly_regime = base_kelly * regime_mult

        # ── 回撤惩罚 ──
        dd_penalty = self.compute_drawdown_penalty(drawdown)
        kelly_dd = kelly_regime * dd_penalty

        # ── 连续盈亏调整 ──
        streak_adj = 1.0
        if consecutive_wins >= 2:
            streak_bonus = min(self._streak_max_adj, consecutive_wins * self._streak_win_step)
            streak_adj += streak_bonus
        if consecutive_losses >= 2:
            streak_penalty = min(self._streak_max_adj, consecutive_losses * self._streak_loss_step)
            streak_adj -= streak_penalty
        streak_adj = max(0.5, min(1.5, streak_adj))
        kelly_streak = kelly_dd * streak_adj

        # ── 样本量折扣 ──
        if trade_count < self._min_trade_count and trade_count > 0:
            sample_discount = trade_count / self._min_trade_count
            sample_discount = max(0.25, sample_discount)
        else:
            sample_discount = 1.0
        kelly_final = kelly_streak * sample_discount
        kelly_final = max(0.0, min(self._max_kelly, kelly_final))

        # ── 分数 Kelly ──
        fractional = self.compute_fractional_kelly(kelly_final, self._default_fraction)

        return {
            "base_kelly": round(base_kelly, 6),
            "wilson_win_rate": round(wilson_win_rate, 6),
            "shrunk_odds": round(shrunk_b, 6),
            "regime_multiplier": round(regime_mult, 4),
            "drawdown_penalty": round(dd_penalty, 4),
            "streak_adjustment": round(streak_adj, 4),
            "sample_discount": round(sample_discount, 4),
            "final_kelly": round(kelly_final, 6),
            "fractional_kelly": round(fractional, 6),
            "regime": regime,
        }

    def compute_continuous_kelly(self, returns: List[float], risk_free: float = 0.02) -> float:
        """
        连续版本 Kelly: f* = (μ - r_f) / σ^2

        μ = 年化收益率, σ^2 = 年化方差

        P31 企业级增强：
        - 下行半方差（semi-variance）替代对称方差，只惩罚下行波动
        - 负偏度惩罚，对厚尾/崩盘风险更保守
        """
        if not returns or len(returns) < 2:
            return 0.0

        clean = [r for r in returns if _is_finite(r)]
        if len(clean) < 2:
            return 0.0
        arr = np.array(clean, dtype=np.float64)
        mu_daily = float(np.mean(arr))

        # P31: 下行半方差——只统计低于均值的波动，*2 保持与全方差的量级一致
        if self._use_semivariance:
            downside = arr[arr < mu_daily]
            if downside.size > 1:
                var_est = float(np.mean((downside - mu_daily) ** 2)) * 2.0
            else:
                var_est = float(np.var(arr, ddof=1)) if len(arr) > 1 else 1e-8
        else:
            var_est = float(np.var(arr, ddof=1)) if len(arr) > 1 else 1e-8

        if var_est <= 1e-12:
            return 0.0

        # P31: 负偏度惩罚——负偏度（左尾厚）代表崩盘风险，按比例降仓
        skew_adj = 1.0
        if self._skew_penalty > 0 and len(arr) >= 3:
            skew = self._safe_skewness(arr)
            if skew < 0:
                skew_adj = max(0.5, 1.0 + self._skew_penalty * skew)

        rf_daily = risk_free / 365
        excess_return = mu_daily - rf_daily
        kelly = excess_return / max(var_est, 1e-8)
        kelly *= skew_adj
        kelly = max(0.0, min(self._max_kelly, kelly))
        return round(kelly, 6)

    def compute_fractional_kelly(self, kelly: float, fraction: float = 0.5) -> float:
        """计算分数 Kelly"""
        kelly = _safe_float(kelly, 0.0)
        fraction = _safe_float(fraction, 0.5)
        fraction = max(0.1, min(1.0, fraction))
        return round(kelly * fraction, 6)

    def _wilson_lower_bound(self, wins: float, n: int, z: float = 1.645) -> float:
        """Wilson 置信区间下界（保守胜率估计）

        企业级凯利的关键修正：胜率点估计会因采样误差导致过度下注。
        用 Wilson 区间下界替代点估计，样本越少越保守，随样本增加收敛到真实值。
        """
        if n <= 0:
            return 0.5
        wins = _safe_float(wins, 0.0)
        z = _safe_float(z, 1.645)
        if not _is_finite(wins) or not _is_finite(z):
            return 0.5
        p = max(0.0, min(1.0, wins / n))
        z2 = z * z
        denom = 1.0 + z2 / n
        center = p + z2 / (2.0 * n)
        margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n)
        lcb = (center - margin) / denom
        return max(0.0, min(1.0, lcb))

    def _shrink_odds(self, b: float, n: int, strength: float = 20.0, prior_b: float = 1.0) -> float:
        """赔率收缩：向盈亏平衡先验(b=1)收缩，样本越少越保守

        防止小样本下赔率被极端盈利/亏损交易放大导致过度下注。
        """
        b = _safe_float(b, 0.0)
        strength = _safe_float(strength, 20.0)
        if strength <= 0 or n <= 0:
            return b
        weight = n / (n + strength)
        return prior_b * (1.0 - weight) + b * weight

    def _shrink_win_rate(self, p: float, n: int, strength: float = 40.0, prior: float = 0.5) -> float:
        """经验贝叶斯胜率收缩：向先验(0.5)收缩，样本越少越保守

        等效于 Beta(α, β) 后验均值，α+β=strength，先验胜率=prior。
        避免小样本胜率被少数交易主导导致过度下注。
        """
        p = _safe_float(p, 0.5)
        strength = _safe_float(strength, 40.0)
        if strength <= 0 or n <= 0:
            return p
        prior_wins = prior * strength
        post = (prior_wins + p * n) / (strength + n)
        return max(0.0, min(1.0, post))

    @staticmethod
    def _safe_skewness(arr: np.ndarray) -> float:
        """样本偏度（Fisher-Pearson），对空/常量数组安全"""
        try:
            arr = np.asarray(arr, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
        except (TypeError, ValueError):
            return 0.0
        n = arr.size
        if n < 3:
            return 0.0
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=0))
        if sigma <= 1e-12:
            return 0.0
        return float(np.mean(((arr - mu) / sigma) ** 3))

    def compute_regime_kelly_multiplier(self, regime: str) -> float:
        """获取市场状态对应的 Kelly 乘数"""
        return self._regime_multipliers.get(regime, self._regime_multipliers.get("unknown", 0.80))

    def compute_drawdown_penalty(self, drawdown: float) -> float:
        """
        回撤惩罚曲线

        drawdown 0~start: 无惩罚
        drawdown start~max: 线性下降至 dd_min_multiplier
        drawdown > max: 保持 dd_min_multiplier
        """
        drawdown = abs(_safe_float(drawdown, 0.0))
        if drawdown <= self._dd_penalty_start:
            return 1.0
        if drawdown >= self._dd_penalty_max:
            return self._dd_min_multiplier

        ratio = (drawdown - self._dd_penalty_start) / (self._dd_penalty_max - self._dd_penalty_start)
        penalty = 1.0 - ratio * (1.0 - self._dd_min_multiplier)
        return round(penalty, 4)

    def get_kelly_summary(self) -> Dict:
        """获取 Kelly 配置摘要"""
        return {
            "max_kelly_fraction": self._max_kelly,
            "default_fraction": self._default_fraction,
            "regime_multipliers": self._regime_multipliers,
            "dd_penalty_start": self._dd_penalty_start,
            "dd_penalty_max": self._dd_penalty_max,
            "dd_min_multiplier": self._dd_min_multiplier,
            "latest": self._latest_summary,
        }


# ═══════════════════════════════════════════════════════════════
# 资金效率监控器
# ═══════════════════════════════════════════════════════════════

class CapitalEfficiencyMonitor:
    """
    资金效率监控器

    功能：
    - 资金周转率：总交易额 / 平均资金占用
    - 保证金利用率：已用保证金 / 总权益
    - 闲置资金检测与提醒
    - 资本回报率（ROCE）：PnL / 分配资金
    - 效率趋势：改善/恶化
    - 跨策略效率排名
    - 资金再部署建议
    """

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()
        self._idle_threshold = _safe_float(config.get("idle_cash_threshold"), 0.15)     # 闲置超过15%触发提醒
        self._efficiency_min = _safe_float(config.get("efficiency_min"), 0.02)          # 最低ROC E阈值
        self._turnover_min = _safe_float(config.get("turnover_min"), 1.0)               # 最低年化周转率
        self._trend_window = int(_safe_float(config.get("trend_window"), 5))            # 趋势观察窗口
        self._deployment_min = _safe_float(config.get("deployment_min"), 0.30)          # 最低部署率

        # 策略指标存储
        self._strategy_metrics: Dict[str, Dict[str, Any]] = {}
        # 效率历史（用于趋势判断）
        self._efficiency_history: Dict[str, List[float]] = {}
        # 最新摘要
        self._latest_summary: Dict[str, Any] = {}

        logger.info(
            f"CapitalEfficiencyMonitor initialized: "
            f"idle_threshold={self._idle_threshold:.0%}, efficiency_min={self._efficiency_min:.0%}"
        )

    def update_metrics(
        self, strategy_name: str, allocated: float, used_margin: float,
        pnl: float, volume: float, holding_time: float,
    ) -> None:
        """
        更新单策略资金效率指标

        Args:
            strategy_name: 策略名称
            allocated: 分配资金
            used_margin: 已用保证金
            pnl: 累计盈亏
            volume: 交易额
            holding_time: 持仓时间（天）
        """
        self._strategy_metrics[strategy_name] = {
            "allocated": max(0, _safe_float(allocated, 0.0)),
            "used_margin": max(0, _safe_float(used_margin, 0.0)),
            "pnl": _safe_float(pnl, 0.0),
            "volume": max(0, _safe_float(volume, 0.0)),
            "holding_time": max(1, _safe_float(holding_time, 1.0)),
            "updated_at": time.time(),
        }

    async def compute_efficiency(self, strategy_name: str) -> Dict:
        """
        计算单策略资金效率

        Returns:
            {
                roc_e: 资本回报率,
                turnover: 年化资金周转率,
                margin_util: 保证金利用率,
                deployment_rate: 资金部署率,
                idle_pct: 闲置资金比例,
                efficiency_score: 综合效率评分,
                trend: 效率趋势,
            }
        """
        m = self._strategy_metrics.get(strategy_name)
        if not m:
            return {"error": f"Strategy {strategy_name} not found"}

        pnl = _safe_float(m.get("pnl"), 0.0)
        used_margin = _safe_float(m.get("used_margin"), 0.0)
        volume = _safe_float(m.get("volume"), 0.0)
        allocated = max(_safe_float(m.get("allocated"), 0.0), 1.0)
        holding_time = max(_safe_float(m.get("holding_time"), 1.0), 1.0)

        # ── 资本回报率（ROCE） ──
        roc_e = pnl / allocated

        # ── 年化资金周转率 ──
        turnover = (volume / allocated) * (365 / holding_time) if holding_time > 0 else 0

        # ── 保证金利用率 ──
        margin_util = used_margin / allocated if allocated > 0 else 0

        # ── 部署率 ──
        deployment_rate = used_margin / allocated if allocated > 0 else 0

        # ── 闲置比例 ──
        idle_pct = 1.0 - min(1.0, deployment_rate)

        # ── 综合效率评分 ──
        score_roc = min(1.0, max(0, roc_e / 0.20))           # 20% ROCE = 满分
        score_turnover = min(1.0, turnover / 10.0)            # 10x 周转率 = 满分
        score_margin = min(1.0, margin_util / 0.80)           # 80% 保证金利用率 = 满分
        efficiency_score = score_roc * 0.40 + score_turnover * 0.30 + score_margin * 0.30

        # ── 效率趋势 ──
        trend = self._compute_efficiency_trend(strategy_name, efficiency_score)

        result = {
            "strategy": strategy_name,
            "roc_e": round(roc_e, 6),
            "turnover": round(turnover, 4),
            "margin_utilization": round(margin_util, 4),
            "deployment_rate": round(deployment_rate, 4),
            "idle_pct": round(idle_pct, 4),
            "efficiency_score": round(efficiency_score, 4),
            "trend": trend,
        }
        return result

    def _compute_efficiency_trend(self, strategy_name: str, current_score: float) -> str:
        """判断效率趋势：improving / stable / declining"""
        if strategy_name not in self._efficiency_history:
            self._efficiency_history[strategy_name] = []

        history = self._efficiency_history[strategy_name]
        history.append(current_score)
        if len(history) > self._trend_window:
            history.pop(0)

        if len(history) < 2:
            return "stable"

        # 简单线性趋势
        recent_avg = np.mean(history[-min(3, len(history)):])
        older_avg = np.mean(history[:min(3, len(history))])

        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "declining"
        else:
            return "stable"

    async def rank_strategies(self) -> List[Dict]:
        """按效率从高到低排序所有策略"""
        ranked = []
        for name in list(self._strategy_metrics.keys()):
            eff = await self.compute_efficiency(name)
            if "error" not in eff:
                ranked.append(eff)

        ranked.sort(key=lambda x: x.get("efficiency_score", 0), reverse=True)
        return ranked

    async def detect_idle_capital(self) -> Dict:
        """
        检测各策略中闲置资金过高的策略

        Returns:
            {
                idle_strategies: [{strategy, idle_pct, idle_amount, allocated, recommendation}],
                total_idle_amount,
                idle_count,
            }
        """
        idle_list = []
        total_idle_amount = 0.0

        for name, m in self._strategy_metrics.items():
            allocated = max(_safe_float(m.get("allocated"), 0.0), 1.0)
            deployed = _safe_float(m.get("used_margin"), 0.0)
            idle_amount = max(0, allocated - deployed)
            idle_pct = idle_amount / allocated if allocated > 0 else 0

            if idle_pct > self._idle_threshold:
                recommendation = ""
                if idle_pct > 0.40:
                    recommendation = f"URGENT: {idle_amount:.0f} USDT idle ({idle_pct:.0%}); consider reallocating or reducing allocation"
                elif idle_pct > 0.25:
                    recommendation = f"Warning: {idle_amount:.0f} USDT idle ({idle_pct:.0%}); monitor closely"
                else:
                    recommendation = f"Notice: {idle_amount:.0f} USDT idle ({idle_pct:.0%})"

                idle_list.append({
                    "strategy": name,
                    "idle_pct": round(idle_pct, 4),
                    "idle_amount": round(idle_amount, 2),
                    "allocated": round(allocated, 2),
                    "recommendation": recommendation,
                })
                total_idle_amount += idle_amount

        return {
            "idle_strategies": idle_list,
            "total_idle_amount": round(total_idle_amount, 2),
            "idle_count": len(idle_list),
        }

    async def suggest_redeployment(self) -> List[Dict]:
        """
        资金再部署建议

        从闲置策略转移资金到高效策略
        """
        async with self._lock:
            idle_result = await self.detect_idle_capital()
            ranked = await self.rank_strategies()

            if not idle_result["idle_strategies"] or not ranked:
                return []

            suggestions = []
            # 高效策略：效率评分 > 0.4 且非闲置
            efficient = [s for s in ranked if s.get("efficiency_score", 0) > 0.4]

            for idle_strat in idle_result["idle_strategies"]:
                idle_name = idle_strat["strategy"]
                idle_amount = idle_strat["idle_amount"]

                if idle_amount <= 0:
                    continue

                # 找到最佳接收方
                for eff_strat in efficient:
                    if eff_strat["strategy"] == idle_name:
                        continue
                    if eff_strat.get("deployment_rate", 0) > 0.80:
                        # 接收方已经部署充分，不使用过多闲置资金接收
                        transfer_amount = min(idle_amount * 0.7, idle_amount)
                    else:
                        transfer_amount = idle_amount

                    if transfer_amount > 1:
                        suggestions.append({
                            "from_strategy": idle_name,
                            "to_strategy": eff_strat["strategy"],
                            "amount": round(transfer_amount, 2),
                            "reason": (
                                f"Move {transfer_amount:.0f} USDT from idle {idle_name} "
                                f"(idle={idle_strat['idle_pct']:.0%}) to efficient {eff_strat['strategy']} "
                                f"(score={eff_strat['efficiency_score']:.2f})"
                            ),
                        })

            # 去重并按金额排序
            seen = set()
            unique_suggestions = []
            for s in sorted(suggestions, key=lambda x: x["amount"], reverse=True):
                key = (s["from_strategy"], s["to_strategy"])
                if key not in seen:
                    seen.add(key)
                    unique_suggestions.append(s)

            return unique_suggestions[:5]  # 最多返回 5 条建议

    def get_efficiency_summary(self) -> Dict:
        """获取资金效率摘要"""
        if not self._strategy_metrics:
            return {"status": "no_data", "total_strategies": 0}

        total_allocated = sum(_safe_float(m.get("allocated"), 0.0) for m in self._strategy_metrics.values())
        total_used = sum(_safe_float(m.get("used_margin"), 0.0) for m in self._strategy_metrics.values())
        total_pnl = sum(_safe_float(m.get("pnl"), 0.0) for m in self._strategy_metrics.values())
        overall_roc = total_pnl / max(total_allocated, 1)
        overall_margin_util = total_used / max(total_allocated, 1)

        return {
            "status": "active",
            "total_strategies": len(self._strategy_metrics),
            "total_allocated": round(total_allocated, 2),
            "total_used_margin": round(total_used, 2),
            "total_pnl": round(total_pnl, 2),
            "overall_roc_e": round(overall_roc, 6),
            "overall_margin_utilization": round(overall_margin_util, 4),
            "idle_threshold": self._idle_threshold,
        }


# ═══════════════════════════════════════════════════════════════
# DynamicAllocator 扩展方法（Monkey-patch）
# ═══════════════════════════════════════════════════════════════

async def _da_link_volatility_targeter(self, targeter: 'VolatilityTargeter'):
    """链接波动率目标器"""
    self._vol_targeter = targeter
    logger.info("VolatilityTargeter linked to DynamicAllocator")


async def _da_link_adaptive_kelly(self, kelly: 'AdaptiveKelly'):
    """链接自适应Kelly"""
    self._adaptive_kelly = kelly
    logger.info("AdaptiveKelly linked to DynamicAllocator")


async def _da_link_efficiency_monitor(self, monitor: 'CapitalEfficiencyMonitor'):
    """链接资金效率监控器"""
    self._efficiency_monitor = monitor
    logger.info("CapitalEfficiencyMonitor linked to DynamicAllocator")


async def _da_compute_adjusted_allocation(
    self, strategy_name: str,
    base_weight: float, returns: List[float],
    current_leverage: float = 1.0,
) -> Dict:
    """综合考虑波动率目标和Kelly调整后的分配"""
    base_weight = _safe_float(base_weight, 0.0)
    result: Dict[str, Any] = {"base_weight": base_weight, "adjusted_weight": base_weight}

    # 波动率调整
    clean_returns = [r for r in returns if _is_finite(r)] if returns else []
    if hasattr(self, '_vol_targeter') and self._vol_targeter and clean_returns:
        vol_info = self._vol_targeter.estimate_volatility(clean_returns)
        scale = self._vol_targeter.compute_scale_factor(
            vol_info["ewma"], 0.20, current_leverage, 10,
        )
        vol_adjusted = base_weight * _safe_float(scale.get("scale_factor"), 1.0)
        result["vol_adjusted"] = round(vol_adjusted, 6)
        result["vol_scale"] = scale
        result["vol_info"] = vol_info
    else:
        result["vol_adjusted"] = base_weight

    # Kelly 调整
    if hasattr(self, '_adaptive_kelly') and self._adaptive_kelly:
        # 从内部绩效缓存拉取策略指标做 Kelly
        m = {}
        if hasattr(self, '_strategy_returns') and self._strategy_returns:
            s_returns = [r for r in self._strategy_returns.get(strategy_name, []) if _is_finite(r)]
            if s_returns:
                wins = [r for r in s_returns if r > 0]
                losses = [r for r in s_returns if r < 0]
                avg_win = np.mean(wins) if wins else 0.01
                avg_loss = abs(np.mean(losses)) if losses else 0.01
                win_rate = len(wins) / len(s_returns) if s_returns else 0.5
                m = {
                    "win_rate": win_rate,
                    "avg_win": avg_win,
                    "avg_loss": avg_loss,
                    "trade_count": len(s_returns),
                }

        if m:
            kelly_result = self._adaptive_kelly.compute_kelly(
                win_rate=m.get("win_rate", 0.5),
                avg_win=m.get("avg_win", 0.01),
                avg_loss=m.get("avg_loss", 0.01),
                trade_count=m.get("trade_count", 0),
            )
            kelly_adjusted = (result.get("vol_adjusted", base_weight)
                              * kelly_result.get("final_kelly", base_weight) * 2)
            result["kelly_adjusted"] = round(min(kelly_adjusted, 0.50), 6)
            result["kelly_detail"] = kelly_result

    result["adjusted_weight"] = result.get("kelly_adjusted", result.get("vol_adjusted", base_weight))
    result["adjusted_weight"] = max(0.0, min(0.50, result["adjusted_weight"]))
    return result


async def _da_get_capital_efficiency_report(self) -> Dict:
    """获取资金效率报告"""
    if hasattr(self, '_efficiency_monitor') and self._efficiency_monitor:
        summary = self._efficiency_monitor.get_efficiency_summary()
        idle = await self._efficiency_monitor.detect_idle_capital()
        ranked = await self._efficiency_monitor.rank_strategies()
        suggestions = await self._efficiency_monitor.suggest_redeployment()
        return {
            "summary": summary,
            "idle_capital": idle,
            "strategy_ranking": ranked,
            "redeployment_suggestions": suggestions,
        }
    return {"status": "no_monitor"}


# Monkey-patch DynamicAllocator
DynamicAllocator.link_volatility_targeter = _da_link_volatility_targeter
DynamicAllocator.link_adaptive_kelly = _da_link_adaptive_kelly
DynamicAllocator.link_efficiency_monitor = _da_link_efficiency_monitor
DynamicAllocator.compute_adjusted_allocation = _da_compute_adjusted_allocation
DynamicAllocator.get_capital_efficiency_report = _da_get_capital_efficiency_report


__all__ = [
    "DynamicAllocator",
    "PoolType",
    "MarketRegime",
    "AllocationPriority",
    "CapitalPool",
    "StrategyAllocation",
    "AllocationPlan",
    "VolatilityTargeter",
    "AdaptiveKelly",
    "CapitalEfficiencyMonitor",
    "get_dynamic_allocator",
    "reset_dynamic_allocator",
]
