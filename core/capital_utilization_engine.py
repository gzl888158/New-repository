"""企业级自适应资金利用率引擎。

= 功能概览 =
1. 动态目标利用率：根据市场波动率、权益模式、时段、回撤实时调整目标
2. 多维利用率评分：资本效率、部署率、策略级利用率、趋势分析
3. 自适应响应：连续严重度函数 + 迟滞防振荡 + 策略级差异化调节
4. 利用率热力图：按策略/时段/市场状态/权益层级追踪历史

= 设计原则 =
- 目标利用率随波动率反向变化（高波动→低利用率保安全）
- 响应采用迟滞机制避免频繁切换（上升/下降阈值不对称）
- 策略级差异：trend 策略目标利用率低于 grid/scalping
- 权益模式联动：紧急模式强制 0% 目标，衰退模式降低目标
- 时段感知：亚洲盘提高目标，欧美重叠盘降低目标
"""

import os
import json
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class UtilizationTier(Enum):
    """利用率等级"""
    CRITICAL_LOW = "critical_low"    # < 10%
    LOW = "low"                      # 10-30%
    MODERATE_LOW = "moderate_low"    # 30-50%
    OPTIMAL = "optimal"              # 50-85%
    HIGH = "high"                    # 85-95%
    CRITICAL_HIGH = "critical_high"  # > 95%


class UtilizationAction(Enum):
    """利用率调节动作"""
    NONE = "none"
    BOOST_AGGRESSIVE = "boost_aggressive"    # 激进加仓
    BOOST_MODERATE = "boost_moderate"        # 适度加仓
    BOOST_CONSERVATIVE = "boost_conservative" # 保守加仓
    HOLD = "hold"                             # 保持
    REDUCE_CONSERVATIVE = "reduce_conservative"
    REDUCE_MODERATE = "reduce_moderate"
    REDUCE_AGGRESSIVE = "reduce_aggressive"
    FORCE_REBALANCE = "force_rebalance"      # 强制再平衡（资本效率过低）
    EMERGENCY_FREEZE = "emergency_freeze"     # 紧急冻结


@dataclass
class UtilizationSnapshot:
    """利用率快照"""
    timestamp: datetime
    # 原始数据
    total_equity: float
    used_margin: float
    available: float
    utilization_rate: float
    # 派生指标
    target_utilization: float
    tier: UtilizationTier
    action: UtilizationAction
    # 策略级数据
    strategy_utilization: Dict[str, float] = field(default_factory=dict)
    # 调节参数
    position_boost: float = 1.0
    signal_relaxation: float = 0.0
    # 环境上下文
    equity_mode: str = "normal"
    volatility_regime: str = "normal"
    session: str = "unknown"


@dataclass
class UtilizationReport:
    """利用率综合报告"""
    timestamp: datetime
    # 核心指标
    current_utilization: float
    dynamic_target: float
    utilization_gap: float                     # 当前 vs 目标差距
    utilization_tier: UtilizationTier
    recommended_action: UtilizationAction
    # 效率指标
    capital_efficiency: float                  # PnL / used_margin
    deployment_ratio: float                    # used / (used + available)
    utilization_trend: float                   # 最近N期斜率
    utilization_volatility: float              # 利用率的波动率
    # 策略级
    strategy_targets: Dict[str, float]         # 各策略目标利用率
    strategy_actions: Dict[str, UtilizationAction]
    # 调节建议
    position_boost: float
    signal_relaxation: float
    allocation_shift: Dict[str, float]         # 策略分配偏移建议
    # 上下文
    equity_mode: str
    volatility_regime: str
    session: str
    account_tier: str


# ═══════════════════════════════════════════════════════════════
# 核心引擎
# ═══════════════════════════════════════════════════════════════

class CapitalUtilizationEngine:
    """企业级自适应资金利用率引擎。

    用法：
        engine = CapitalUtilizationEngine(config)
        engine.set_equity_monitor(equity_monitor)
        report = engine.analyze(
            total_equity=56.67,
            used_margin=25.0,
            available=31.0,
            strategy_usage={"grid": 10.0, "trend": 8.0, "scalping": 7.0},
            recent_pnl={"grid": 0.05, "trend": -0.02, "scalping": 0.03},
            atr_ratio=1.2,  # 当前ATR / 历史均值
        )
        # 应用建议
        engine.apply_recommendations(report)
    """

    # ── 策略级基础目标利用率 ──
    STRATEGY_BASE_TARGETS = {
        "grid": 0.80,        # 网格策略：高利用率（持续开仓）
        "scalping": 0.75,    # 剥头皮：较高利用率（快进快出）
        "arbitrage": 0.70,   # 套利：中等利用率（机会驱动）
        "trend": 0.60,       # 趋势：较低利用率（等待趋势确认）
        "spot_grid": 0.65,   # 现货网格
        "spot_martingale": 0.55,  # 马丁格尔：最低（风险控制）
    }

    # ── 时段调节系数 ──
    SESSION_ADJUSTMENTS = {
        "asian": 1.10,       # 亚洲盘 (UTC 0-8)：提高10%
        "european": 1.00,    # 欧洲盘 (UTC 8-12)：标准
        "overlap_eu_us": 0.85,  # 欧美重叠 (UTC 13-16)：降低15%
        "us": 0.90,          # 美国盘 (UTC 13-20)：降低10%
        "low_liquidity": 0.70,  # 低流动性 (UTC 20-24)：降低30%
        "weekend": 0.75,     # R123: 0.50→0.75 周末降低幅度减小
    }

    # ── 波动率调节系数 ──
    VOLATILITY_ADJUSTMENTS = {
        "very_low": 1.20,    # ATR < 0.5x 均值
        "low": 1.10,         # 0.5-0.8x
        "normal": 1.00,      # 0.8-1.2x
        "high": 0.80,        # 1.2-2.0x
        "very_high": 0.55,   # > 2.0x
    }

    # ── 权益模式调节系数 ──
    EQUITY_MODE_ADJUSTMENTS = {
        "normal": 1.00,
        "growth": 1.15,      # 增长模式：可提高利用率
        "decline": 0.60,     # 衰退模式：大幅降低
        "recovery": 0.80,    # 恢复模式：谨慎提高
        "emergency": 0.0,    # 紧急模式：归零
        "milestone": 1.05,
    }

    # ── 账户层级目标利用率上限 ──
    TIER_CAPS = {
        "nano": 0.75, "micro": 0.80, "small": 0.85,
        "medium": 0.90, "large": 0.95, "xlarge": 0.95,
    }

    # ── 账户层级仓位乘数上限 ──
    TIER_POSITION_BOOST_CAPS = {
        "nano": 3.0, "micro": 2.5, "small": 2.0,  # R122: 反转，小账户更高boost
        "medium": 1.8, "large": 1.5, "xlarge": 1.5,
    }

    # ── 强制再平衡触发阈值 ──
    # 近期资本效率（总PnL / 总已用保证金）低于此值，且总权益超过最小规模时，
    # 触发 FORCE_REBALANCE：资金未被有效利用，需向高效策略重新分配。
    FORCE_REBALANCE_EFFICIENCY_THRESHOLD = 0.15
    FORCE_REBALANCE_MIN_EQUITY = 5.0  # R100: 100→5 小账户也需要强制再平衡
    # 资金占用低于此值视为「闲置」而非「无效利用」，不触发 FORCE_REBALANCE
    FORCE_REBALANCE_MIN_UTILIZATION = 0.10
    # 已用保证金低于此值（USDT）时效率分母无意义，返回中性值避免被 PnL 放大成异常值
    MIN_EFFICIENCY_MARGIN = 1.0

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        trading_cfg = config.get("trading", {}) if config else {}

        # ── 基础配置 ──
        self._base_target = trading_cfg.get("target_utilization", 0.85)
        self._min_utilization = trading_cfg.get("min_utilization", 0.50)
        self._max_utilization = 0.95
        self._emergency_floor = 0.95  # 超过此值强制降仓

        # ── 动态目标参数 ──
        self._dynamic_target = self._base_target
        self._volatility_regime = "normal"
        self._current_session = "unknown"
        self._equity_mode = "normal"
        self._account_tier = "nano"

        # ── 迟滞（防振荡）参数 ──
        self._hysteresis_band = 0.05     # 5% 迟滞带
        self._last_action: Optional[UtilizationAction] = None
        self._last_action_time: Optional[datetime] = None
        self._action_cooldown = 120      # 动作冷却时间（秒）
        self._oscillation_counter = 0
        self._max_oscillations = 3       # 振荡上限，超过则锁定

        # ── 历史追踪 ──
        self._snapshots: Deque[UtilizationSnapshot] = deque(maxlen=500)
        self._reports: Deque[UtilizationReport] = deque(maxlen=100)
        self._utilization_history: Deque[float] = deque(maxlen=200)

        # ── 策略级追踪 ──
        self._strategy_utilization_history: Dict[str, Deque[float]] = {}
        self._strategy_pnl_history: Dict[str, Deque[float]] = {}

        # ── 外部依赖 ──
        self._equity_monitor = None
        self._regime_engine = None

        # ── 持久化 ──
        self._state_path = os.path.join("data", "utilization_engine_state.json")
        self._save_interval = 60
        self._save_counter = 0

        # ── 运行时 ──
        self._running = False
        self._monitor_task = None

        logger.info(f"CapitalUtilizationEngine initialized (base_target={self._base_target:.0%})")

    # ═══════════════════════════════════════════════════════════════
    # 依赖注入
    # ═══════════════════════════════════════════════════════════════

    def set_equity_monitor(self, monitor):
        self._equity_monitor = monitor

    def set_regime_engine(self, engine):
        self._regime_engine = engine

    # ═══════════════════════════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════════════════════════

    async def start(self):
        self._load_state()
        self._running = True
        self._monitor_task = asyncio.create_task(self._persist_loop())
        logger.info("CapitalUtilizationEngine started")

    async def stop(self):
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("CapitalUtilizationEngine stopped")

    async def _persist_loop(self):
        while self._running:
            self._save_counter += 1
            if self._save_counter >= self._save_interval // 5:
                self._save_state()
                self._save_counter = 0
            await asyncio.sleep(5)

    # ═══════════════════════════════════════════════════════════════
    # 核心：多维分析
    # ═══════════════════════════════════════════════════════════════

    def analyze(self, total_equity: float, used_margin: float, available: float,
                strategy_usage: Dict[str, float] = None,
                recent_pnl: Dict[str, float] = None,
                atr_ratio: float = 1.0,
                session: str = None,
                total_drawdown_pct: float = 0.0) -> UtilizationReport:
        """执行多维利用率分析，返回综合报告。

        Args:
            total_equity: 总权益
            used_margin: 已用保证金
            available: 可用余额
            strategy_usage: 各策略保证金占用 {strategy_name: margin}
            recent_pnl: 各策略近期盈亏 {strategy_name: pnl}
            atr_ratio: 当前ATR / 历史均值ATR
            session: 当前交易时段（可选，自动检测）
            total_drawdown_pct: 当前总回撤百分比
        """
        if total_equity <= 0:
            return self._empty_report()

        now = datetime.now()
        utilization = used_margin / total_equity if total_equity > 0 else 0

        # ── 1. 更新上下文 ──
        self._update_context(atr_ratio, session)

        # ── 2. 计算动态目标利用率 ──
        self._dynamic_target = self._compute_dynamic_target(
            utilization, atr_ratio, total_drawdown_pct
        )

        # ── 3. 利用率分级 ──
        tier = self._classify_utilization(utilization)

        # ── 4. 计算效率指标 ──
        capital_efficiency = self._calc_capital_efficiency(
            recent_pnl or {}, strategy_usage or {}
        )
        efficiency_is_valid = (
            sum((strategy_usage or {}).values()) > self.MIN_EFFICIENCY_MARGIN
            and bool(recent_pnl)
        )
        deployment_ratio = used_margin / (used_margin + available) if (used_margin + available) > 0 else 0
        util_trend = self._calc_utilization_trend(utilization)
        util_volatility = self._calc_utilization_volatility()

        # ── 5. 策略级目标 ──
        strategy_targets = self._compute_strategy_targets(atr_ratio, total_drawdown_pct)
        strategy_actions = self._compute_strategy_actions(
            strategy_usage or {}, strategy_targets, recent_pnl or {}, total_equity
        )

        # ── 6. 确定全局动作 ──
        action = self._determine_action(
            utilization, tier, util_trend, capital_efficiency, total_equity,
            efficiency_is_valid=efficiency_is_valid,
        )

        # ── 7. 计算调节参数 ──
        position_boost = self._compute_position_boost(utilization, tier, action)
        signal_relaxation = self._compute_signal_relaxation(utilization, tier, action)
        allocation_shift = self._compute_allocation_shift(
            strategy_usage or {}, strategy_targets, recent_pnl or {}, action
        )

        # ── 8. 构建报告 ──
        report = UtilizationReport(
            timestamp=now,
            current_utilization=utilization,
            dynamic_target=self._dynamic_target,
            utilization_gap=self._dynamic_target - utilization,
            utilization_tier=tier,
            recommended_action=action,
            capital_efficiency=capital_efficiency,
            deployment_ratio=deployment_ratio,
            utilization_trend=util_trend,
            utilization_volatility=util_volatility,
            strategy_targets=strategy_targets,
            strategy_actions=strategy_actions,
            position_boost=position_boost,
            signal_relaxation=signal_relaxation,
            allocation_shift=allocation_shift,
            equity_mode=self._equity_mode,
            volatility_regime=self._volatility_regime,
            session=self._current_session,
            account_tier=self._account_tier,
        )

        # ── 9. 保存快照 ──
        snapshot = UtilizationSnapshot(
            timestamp=now,
            total_equity=total_equity,
            used_margin=used_margin,
            available=available,
            utilization_rate=utilization,
            target_utilization=self._dynamic_target,
            tier=tier,
            action=action,
            strategy_utilization=strategy_usage or {},
            position_boost=position_boost,
            signal_relaxation=signal_relaxation,
            equity_mode=self._equity_mode,
            volatility_regime=self._volatility_regime,
            session=self._current_session,
        )
        self._snapshots.append(snapshot)
        self._reports.append(report)
        self._utilization_history.append(utilization)

        # 更新策略级历史
        for sname, usage in (strategy_usage or {}).items():
            if sname not in self._strategy_utilization_history:
                self._strategy_utilization_history[sname] = deque(maxlen=200)
            strat_util = usage / total_equity if total_equity > 0 else 0
            self._strategy_utilization_history[sname].append(strat_util)

        for sname, pnl in (recent_pnl or {}).items():
            if sname not in self._strategy_pnl_history:
                self._strategy_pnl_history[sname] = deque(maxlen=200)
            self._strategy_pnl_history[sname].append(pnl)

        return report

    # ═══════════════════════════════════════════════════════════════
    # 动态目标利用率计算
    # ═══════════════════════════════════════════════════════════════

    def _update_context(self, atr_ratio: float, session: str = None):
        """更新环境上下文"""
        # 波动率分级
        if atr_ratio < 0.5:
            self._volatility_regime = "very_low"
        elif atr_ratio < 0.8:
            self._volatility_regime = "low"
        elif atr_ratio < 1.2:
            self._volatility_regime = "normal"
        elif atr_ratio < 2.0:
            self._volatility_regime = "high"
        else:
            self._volatility_regime = "very_high"

        # 时段检测
        if session:
            self._current_session = session
        else:
            self._current_session = self._detect_session()

        # 权益模式
        if self._equity_monitor:
            status = self._equity_monitor.get_equity_status()
            self._equity_mode = status.get("mode", "normal")
            self._account_tier = status.get("account_tier", "nano")

    def _detect_session(self) -> str:
        """检测当前交易时段"""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()

        if weekday >= 5:  # 周六日
            return "weekend"
        if 0 <= hour < 8:
            return "asian"
        if 8 <= hour < 13:
            return "european"
        if 13 <= hour < 16:
            return "overlap_eu_us"
        if 16 <= hour < 20:
            return "us"
        return "low_liquidity"

    def _compute_dynamic_target(self, current_util: float, atr_ratio: float,
                                 drawdown_pct: float) -> float:
        """计算动态目标利用率。

        公式：
            target = base_target × vol_adj × session_adj × equity_adj × dd_adj
        """
        # 基础目标
        target = self._base_target

        # 波动率调节
        vol_adj = self.VOLATILITY_ADJUSTMENTS.get(self._volatility_regime, 1.0)
        target *= vol_adj

        # 时段调节
        session_adj = self.SESSION_ADJUSTMENTS.get(self._current_session, 1.0)
        target *= session_adj

        # 权益模式调节
        equity_adj = self.EQUITY_MODE_ADJUSTMENTS.get(self._equity_mode, 1.0)
        target *= equity_adj

        # 回撤调节：回撤越大，目标越低
        if drawdown_pct > 0:
            dd_adj = max(0.3, 1.0 - drawdown_pct * 3)  # 10%回撤→70%目标
            target *= dd_adj

        # 账户层级上限
        target = min(target, self.TIER_CAPS.get(self._account_tier, 0.85))

        # 限幅
        target = max(0.10, min(0.95, target))

        return target

    def _compute_strategy_targets(self, atr_ratio: float,
                                   drawdown_pct: float = 0.0) -> Dict[str, float]:
        """计算各策略的动态目标利用率。

        与全局动态目标保持一致的五因子：波动率 × 时段 × 权益模式 × 回撤 × 账户层级上限。
        """
        # 回撤调节：回撤越大，策略目标越低（与全局动态目标共用同一回撤因子）
        dd_adj = max(0.3, 1.0 - drawdown_pct * 3) if drawdown_pct > 0 else 1.0
        tier_cap = self.TIER_CAPS.get(self._account_tier, 0.85)

        targets = {}
        for sname, base in self.STRATEGY_BASE_TARGETS.items():
            # 波动率调节
            vol_adj = self.VOLATILITY_ADJUSTMENTS.get(self._volatility_regime, 1.0)
            # 权益模式调节
            equity_adj = self.EQUITY_MODE_ADJUSTMENTS.get(self._equity_mode, 1.0)
            # 时段调节
            session_adj = self.SESSION_ADJUSTMENTS.get(self._current_session, 1.0)

            target = base * vol_adj * equity_adj * session_adj * dd_adj
            target = min(target, tier_cap)
            targets[sname] = max(0.05, min(0.95, target))
        return targets

    # ═══════════════════════════════════════════════════════════════
    # 利用率分级
    # ═══════════════════════════════════════════════════════════════

    def _classify_utilization(self, utilization: float) -> UtilizationTier:
        """将利用率映射到等级"""
        if utilization < 0.10:
            return UtilizationTier.CRITICAL_LOW
        if utilization < 0.30:
            return UtilizationTier.LOW
        if utilization < self._min_utilization:
            return UtilizationTier.MODERATE_LOW
        if utilization <= self._max_utilization:
            return UtilizationTier.OPTIMAL
        if utilization <= 0.98:
            return UtilizationTier.HIGH
        return UtilizationTier.CRITICAL_HIGH

    # ═══════════════════════════════════════════════════════════════
    # 效率指标计算
    # ═══════════════════════════════════════════════════════════════

    def _calc_capital_efficiency(self, recent_pnl: Dict[str, float],
                                  strategy_usage: Dict[str, float]) -> float:
        """计算资本效率 = 总PnL / 总已用保证金"""
        total_pnl = sum(recent_pnl.values())
        total_used = sum(strategy_usage.values())
        # 分母过小（几乎无持仓）时比值无意义，返回中性值避免被 PnL 放大成异常值
        if total_used <= self.MIN_EFFICIENCY_MARGIN:
            return 0.0
        return total_pnl / total_used

    def _calc_utilization_trend(self, current_util: float) -> float:
        """计算利用率趋势（最近20期线性回归斜率）"""
        history = list(self._utilization_history)
        if len(history) < 5:
            return 0.0

        # 简单线性回归斜率
        n = min(20, len(history))
        recent = history[-n:]
        x_mean = (n - 1) / 2
        y_mean = sum(recent) / n

        numerator = sum((i - x_mean) * (recent[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _calc_utilization_volatility(self) -> float:
        """计算利用率波动率（标准差）"""
        history = list(self._utilization_history)
        if len(history) < 5:
            return 0.0
        recent = list(history)[-20:]
        mean = sum(recent) / len(recent)
        variance = sum((x - mean) ** 2 for x in recent) / len(recent)
        return variance ** 0.5

    # ═══════════════════════════════════════════════════════════════
    # 动作决策（迟滞防振荡）
    # ═══════════════════════════════════════════════════════════════

    def _determine_action(self, utilization: float, tier: UtilizationTier,
                           trend: float, efficiency: float,
                           total_equity: Optional[float] = None,
                           efficiency_is_valid: bool = True) -> UtilizationAction:
        """确定全局调节动作，含迟滞防振荡机制"""
        now = datetime.now()

        # 紧急模式优先
        if self._equity_mode == "emergency":
            return UtilizationAction.EMERGENCY_FREEZE

        # 资本效率硬约束：资金规模足够大、确实被占用、但效率过低时，强制再平衡。
        # utilization 极低说明资金处于「闲置」而非「无效利用」，
        # 应交由后续 BOOST 逻辑部署闲置资金，而非误触发 FORCE_REBALANCE。
        if (total_equity is not None
                and total_equity > self.FORCE_REBALANCE_MIN_EQUITY
                and utilization >= self.FORCE_REBALANCE_MIN_UTILIZATION
            and efficiency_is_valid
                and efficiency < self.FORCE_REBALANCE_EFFICIENCY_THRESHOLD):
            return UtilizationAction.FORCE_REBALANCE

        # ── 相对动态目标的硬约束（优先于迟滞）──
        # gap = 当前利用率 - 动态目标（正数 = 超目标）
        gap = utilization - self._dynamic_target
        if gap > 0.20:
            # 严重超目标：强制降仓（如周末/高波动导致目标大幅降低）
            return UtilizationAction.REDUCE_MODERATE
        if gap > 0.10 and utilization > 0.80:
            # 明显超目标且绝对水平高：保守降仓
            return UtilizationAction.REDUCE_CONSERVATIVE
        if gap < -0.35:
            # 严重低于目标：激进加仓
            return UtilizationAction.BOOST_AGGRESSIVE

        # 振荡检测
        if self._last_action_time:
            elapsed = (now - self._last_action_time).total_seconds()
            if elapsed < self._action_cooldown:
                return self._last_action or UtilizationAction.HOLD

        # 基于利用率等级确定基础动作
        raw_action = self._get_raw_action(tier, efficiency, trend)

        # 迟滞检查：如果新动作与上次动作方向相反，需要额外确认
        if self._last_action and raw_action != self._last_action:
            if self._is_oscillating(raw_action):
                self._oscillation_counter += 1
                if self._oscillation_counter >= self._max_oscillations:
                    logger.warning(
                        f"Utilization action oscillation detected ({self._oscillation_counter}x), "
                        f"locking to HOLD for {self._action_cooldown * 2}s"
                    )
                    self._action_cooldown = min(600, self._action_cooldown * 2)
                    return UtilizationAction.HOLD
                # 迟滞：维持上次动作
                return self._last_action
            else:
                self._oscillation_counter = max(0, self._oscillation_counter - 1)

        self._last_action = raw_action
        self._last_action_time = now
        return raw_action

    def _get_raw_action(self, tier: UtilizationTier, efficiency: float,
                         trend: float) -> UtilizationAction:
        """根据等级和效率确定基础动作"""
        if tier == UtilizationTier.CRITICAL_LOW:
            return UtilizationAction.BOOST_AGGRESSIVE
        if tier == UtilizationTier.LOW:
            return UtilizationAction.BOOST_MODERATE
        if tier == UtilizationTier.MODERATE_LOW:
            # 效率高则激进加仓，效率低则保守
            if efficiency > 0.01:
                return UtilizationAction.BOOST_MODERATE
            return UtilizationAction.BOOST_CONSERVATIVE
        if tier == UtilizationTier.OPTIMAL:
            return UtilizationAction.HOLD
        if tier == UtilizationTier.HIGH:
            if trend > 0.001:  # 利用率还在上升
                return UtilizationAction.REDUCE_CONSERVATIVE
            return UtilizationAction.HOLD
        if tier == UtilizationTier.CRITICAL_HIGH:
            return UtilizationAction.REDUCE_AGGRESSIVE
        return UtilizationAction.HOLD

    def _is_oscillating(self, new_action: UtilizationAction) -> bool:
        """检测是否在 BOOST 类和 REDUCE 类之间振荡"""
        boost_actions = {
            UtilizationAction.BOOST_AGGRESSIVE,
            UtilizationAction.BOOST_MODERATE,
            UtilizationAction.BOOST_CONSERVATIVE,
        }
        reduce_actions = {
            UtilizationAction.REDUCE_CONSERVATIVE,
            UtilizationAction.REDUCE_MODERATE,
            UtilizationAction.REDUCE_AGGRESSIVE,
        }
        last_is_boost = self._last_action in boost_actions
        last_is_reduce = self._last_action in reduce_actions
        new_is_boost = new_action in boost_actions
        new_is_reduce = new_action in reduce_actions

        return (last_is_boost and new_is_reduce) or (last_is_reduce and new_is_boost)

    # ═══════════════════════════════════════════════════════════════
    # 策略级动作
    # ═══════════════════════════════════════════════════════════════

    def _compute_strategy_actions(self, strategy_usage: Dict[str, float],
                                   strategy_targets: Dict[str, float],
                                   recent_pnl: Dict[str, float],
                                   total_equity: float) -> Dict[str, UtilizationAction]:
        """计算各策略的调节动作。

        策略级利用率 = 该策略已用保证金 / 总权益，与策略目标利用率（百分比）比较。
        """
        actions = {}
        if total_equity <= 0:
            return actions

        for sname, used in strategy_usage.items():
            target = strategy_targets.get(sname, 0.50)
            # 策略级当前利用率
            strat_util = used / total_equity
            gap = target - strat_util

            if gap > 0.20:
                actions[sname] = UtilizationAction.BOOST_MODERATE
            elif gap > 0.05:
                actions[sname] = UtilizationAction.BOOST_CONSERVATIVE
            elif gap < -0.10:
                pnl = recent_pnl.get(sname, 0)
                if pnl < 0:
                    actions[sname] = UtilizationAction.REDUCE_MODERATE
                else:
                    actions[sname] = UtilizationAction.REDUCE_CONSERVATIVE
            else:
                actions[sname] = UtilizationAction.HOLD

        return actions

    # ═══════════════════════════════════════════════════════════════
    # 调节参数计算
    # ═══════════════════════════════════════════════════════════════

    def _compute_position_boost(self, utilization: float, tier: UtilizationTier,
                                 action: UtilizationAction) -> float:
        """计算仓位乘数（连续函数，非离散档位）。

        - EMERGENCY_FREEZE → 0.0（冻结）
        - HOLD → 1.0（保持）
        - REDUCE_* → 0.5~1.0（降仓）
        - BOOST_* → 1.0~cap（加仓，连续 sigmoid）
        """
        if action == UtilizationAction.EMERGENCY_FREEZE:
            return 0.0

        if action == UtilizationAction.HOLD:
            return 1.0

        # 强制再平衡：不改变总仓位，仅重新分配资金，故仓位乘数保持中性
        if action == UtilizationAction.FORCE_REBALANCE:
            return 1.0

        # 降仓动作：返回小于等于 1.0 的乘数
        if action in (UtilizationAction.REDUCE_CONSERVATIVE,
                      UtilizationAction.REDUCE_MODERATE,
                      UtilizationAction.REDUCE_AGGRESSIVE):
            reduce_map = {
                UtilizationAction.REDUCE_CONSERVATIVE: 0.85,
                UtilizationAction.REDUCE_MODERATE: 0.70,
                UtilizationAction.REDUCE_AGGRESSIVE: 0.50,
            }
            return reduce_map[action]

        # 加仓动作：基于利用率差距的连续函数
        gap = max(0, self._dynamic_target - utilization)
        if gap <= 0:
            return 1.0

        # sigmoid-like 连续函数
        # gap=0.0 → boost=1.0, gap=0.3 → boost≈2.0, gap=0.5 → boost≈3.0
        import math
        boost = 1.0 + 4.0 / (1.0 + math.exp(-10.0 * (gap - 0.25)))

        # 账户层级上限
        cap = self.TIER_POSITION_BOOST_CAPS.get(self._account_tier, 2.0)

        # 动作级修正
        if action == UtilizationAction.BOOST_AGGRESSIVE:
            cap = min(cap * 1.2, 6.0)
        elif action == UtilizationAction.BOOST_CONSERVATIVE:
            cap = cap * 0.7

        return max(0.5, min(boost, cap))

    def _compute_signal_relaxation(self, utilization: float, tier: UtilizationTier,
                                    action: UtilizationAction) -> float:
        """计算信号质量阈值放松量（连续函数）。

        语义统一：正数 = 放松（降低阈值，让更多信号通过）；
                  负数 = 收紧（提高阈值，让更少信号通过）。
        """
        if action == UtilizationAction.EMERGENCY_FREEZE:
            return -0.30  # 紧急：大幅提高阈值（收紧）

        if action == UtilizationAction.FORCE_REBALANCE:
            return -0.10  # 强制再平衡：收紧信号，只放行高质量信号

        if tier in (UtilizationTier.HIGH, UtilizationTier.CRITICAL_HIGH):
            return -0.10  # 高利用率：提高阈值降低信号

        gap = max(0, self._dynamic_target - utilization)
        if gap <= 0.02:
            return 0.0

        # 连续映射：gap 0→0.0, gap 0.3→0.10, gap 0.5→0.15
        relaxation = min(0.15, gap * 0.30)
        return relaxation

    def _compute_allocation_shift(self, strategy_usage: Dict[str, float],
                                   strategy_targets: Dict[str, float],
                                   recent_pnl: Dict[str, float],
                                   action: UtilizationAction) -> Dict[str, float]:
        """计算策略分配偏移建议。

        普通动作：渐进式向目标靠近（30% 步长）。
        FORCE_REBALANCE：激进再平衡（60% 步长 + 更强效率修正 + 更宽边界），
        向盈利策略大幅倾斜、砍掉亏损策略。
        """
        shifts = {}
        if action == UtilizationAction.EMERGENCY_FREEZE:
            return {s: 0.0 for s in strategy_targets}

        is_force = action == UtilizationAction.FORCE_REBALANCE
        step = 0.6 if is_force else 0.3        # 渐进步长
        profit_amp = 1.5 if is_force else 1.2  # 盈利策略放大系数
        loss_damp = 0.2 if is_force else 0.5   # 亏损策略压减系数
        clip = 0.15 if is_force else 0.10      # 单步偏移上限

        total_used = sum(strategy_usage.values()) if strategy_usage else 1.0

        for sname, target in strategy_targets.items():
            current = strategy_usage.get(sname, 0) / total_used if total_used > 0 else 0
            pnl = recent_pnl.get(sname, 0)

            # 向目标靠近
            raw_shift = (target - current) * step

            # 效率修正：盈利策略多分配，亏损策略少分配
            if pnl > 0.001:
                raw_shift *= profit_amp
            elif pnl < -0.001:
                raw_shift *= loss_damp

            shifts[sname] = max(-clip, min(clip, raw_shift))

        return shifts

    # ═══════════════════════════════════════════════════════════════
    # 公共接口
    # ═══════════════════════════════════════════════════════════════

    def get_dynamic_target(self) -> float:
        return self._dynamic_target

    def get_latest_report(self) -> Optional[UtilizationReport]:
        return self._reports[-1] if self._reports else None

    def get_utilization_summary(self) -> Dict[str, Any]:
        """获取利用率摘要"""
        report = self.get_latest_report()
        if not report:
            return {"available": False}

        return {
            "current_utilization": report.current_utilization,
            "dynamic_target": report.dynamic_target,
            "utilization_gap": report.utilization_gap,
            "tier": report.utilization_tier.value,
            "action": report.recommended_action.value,
            "capital_efficiency": report.capital_efficiency,
            "deployment_ratio": report.deployment_ratio,
            "utilization_trend": report.utilization_trend,
            "utilization_volatility": report.utilization_volatility,
            "position_boost": report.position_boost,
            "signal_relaxation": report.signal_relaxation,
            "equity_mode": report.equity_mode,
            "volatility_regime": report.volatility_regime,
            "session": report.session,
            "account_tier": report.account_tier,
            "strategy_targets": report.strategy_targets,
            "strategy_actions": {k: v.value for k, v in report.strategy_actions.items()},
            "allocation_shift": report.allocation_shift,
            "oscillation_counter": self._oscillation_counter,
            "timestamp": report.timestamp.isoformat(),
        }

    def get_utilization_heatmap(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取利用率热力图数据"""
        snapshots = list(self._snapshots)[-limit:]
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "utilization": s.utilization_rate,
                "target": s.target_utilization,
                "tier": s.tier.value,
                "action": s.action.value,
                "session": s.session,
                "volatility": s.volatility_regime,
                "equity_mode": s.equity_mode,
                "strategy_utilization": s.strategy_utilization,
            }
            for s in snapshots
        ]

    def get_oscillation_status(self) -> Dict[str, Any]:
        """获取振荡状态"""
        return {
            "oscillation_counter": self._oscillation_counter,
            "max_oscillations": self._max_oscillations,
            "last_action": self._last_action.value if self._last_action else "none",
            "last_action_time": self._last_action_time.isoformat() if self._last_action_time else None,
            "action_cooldown": self._action_cooldown,
            "is_locked": self._oscillation_counter >= self._max_oscillations,
        }

    def reset_oscillation_state(self):
        """重置振荡计数器（外部触发）"""
        self._oscillation_counter = 0
        self._action_cooldown = 120
        self._last_action = None
        self._last_action_time = None
        logger.info("Utilization oscillation state reset")

    def reset_full_state(self) -> Dict[str, Any]:
        """完整重置引擎运行时状态（企业级资金效率重置按钮的底层实现）。

        重置范围：
        1. 迟滞防振荡状态（计数器 / 冷却 / 上次动作）
        2. 全部历史快照、报告、利用率历史、策略级历史
        3. 动态目标利用率恢复为基础值
        4. 环境上下文（波动率 / 时段 / 权益模式 / 账户层级）
        5. 持久化计数器，并立即落盘

        Returns:
            重置前的状态快照，便于上层记录与日志审计。
        """
        before = {
            "snapshots": len(self._snapshots),
            "reports": len(self._reports),
            "utilization_history": len(self._utilization_history),
            "strategy_utilization_history": {
                k: len(v) for k, v in self._strategy_utilization_history.items()
            },
            "strategy_pnl_history": {
                k: len(v) for k, v in self._strategy_pnl_history.items()
            },
            "dynamic_target": self._dynamic_target,
            "oscillation_counter": self._oscillation_counter,
            "action_cooldown": self._action_cooldown,
            "last_action": self._last_action.value if self._last_action else None,
            "volatility_regime": self._volatility_regime,
            "current_session": self._current_session,
            "equity_mode": self._equity_mode,
            "account_tier": self._account_tier,
        }

        # 1. 重置振荡状态
        self.reset_oscillation_state()

        # 2. 清空历史追踪
        self._snapshots.clear()
        self._reports.clear()
        self._utilization_history.clear()
        self._strategy_utilization_history.clear()
        self._strategy_pnl_history.clear()

        # 3. 恢复动态目标到基础值
        self._dynamic_target = self._base_target

        # 4. 重置环境上下文
        self._volatility_regime = "normal"
        self._current_session = "unknown"
        self._equity_mode = "normal"
        self._account_tier = "nano"

        # 5. 重置持久化计数器并立即落盘
        self._save_counter = 0
        self._save_state()

        logger.info(
            f"CapitalUtilizationEngine full reset: cleared {before['snapshots']} snapshots, "
            f"{before['reports']} reports, {before['utilization_history']} history entries; "
            f"dynamic_target {before['dynamic_target']:.0%} -> {self._dynamic_target:.0%}"
        )
        return before

    # ═══════════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════════

    def _save_state(self):
        try:
            state = {
                "dynamic_target": self._dynamic_target,
                "volatility_regime": self._volatility_regime,
                "current_session": self._current_session,
                "equity_mode": self._equity_mode,
                "account_tier": self._account_tier,
                "last_action": self._last_action.value if self._last_action else None,
                "last_action_time": self._last_action_time.isoformat() if self._last_action_time else None,
                "oscillation_counter": self._oscillation_counter,
                "action_cooldown": self._action_cooldown,
                "saved_at": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"UtilizationEngine save error: {e}")

    def _load_state(self):
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._dynamic_target = state.get("dynamic_target", self._base_target)
            self._volatility_regime = state.get("volatility_regime", "normal")
            self._current_session = state.get("current_session", "unknown")
            self._equity_mode = state.get("equity_mode", "normal")
            self._account_tier = state.get("account_tier", "nano")
            self._oscillation_counter = state.get("oscillation_counter", 0)
            self._action_cooldown = state.get("action_cooldown", 120)

            action_str = state.get("last_action")
            if action_str:
                try:
                    self._last_action = UtilizationAction(action_str)
                except ValueError:
                    pass
            action_time = state.get("last_action_time")
            if action_time:
                try:
                    self._last_action_time = datetime.fromisoformat(action_time)
                except ValueError:
                    pass

            logger.info(
                f"UtilizationEngine state loaded: target={self._dynamic_target:.0%}, "
                f"vol={self._volatility_regime}, session={self._current_session}"
            )
        except Exception as e:
            logger.debug(f"UtilizationEngine load error: {e}")

    def _empty_report(self) -> UtilizationReport:
        """返回空报告"""
        return UtilizationReport(
            timestamp=datetime.now(),
            current_utilization=0.0,
            dynamic_target=self._base_target,
            utilization_gap=0.0,
            utilization_tier=UtilizationTier.CRITICAL_LOW,
            recommended_action=UtilizationAction.HOLD,
            capital_efficiency=0.0,
            deployment_ratio=0.0,
            utilization_trend=0.0,
            utilization_volatility=0.0,
            strategy_targets={},
            strategy_actions={},
            position_boost=1.0,
            signal_relaxation=0.0,
            allocation_shift={},
            equity_mode="normal",
            volatility_regime="normal",
            session="unknown",
            account_tier="nano",
        )