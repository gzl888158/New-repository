"""
智能交易体 (Intelligent Trading Agent) v2 - 生产级核心决策引擎

功能：
1. 市场状态感知与自适应决策
2. 信号质量评估与多维度审核（7维度）
3. 成长性学习：从历史交易中学习优化参数
4. 分层协调：策略级->资产级->全局级
5. 错误报警与自动恢复
6. 状态持久化：重启后恢复所有决策状态
7. 健康自检：自动检测并修复衰退状态
8. 决策审计追踪：完整决策链路可追溯
"""

import json
import logging
import os
import sqlite3
import time
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from collections import deque
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# 负期望退出/黑名单/学习统计仅基于近 N 小时的成交，避免历史回撤期亏损
# 永久拖累判定（与 alert_evaluator._QUALITY_WINDOW_HOURS 口径一致）。
_LEARNING_WINDOW_HOURS = 24

# P30 最小利润防线：各策略最低止盈目标比例与手续费覆盖倍数。
# target_pct 对齐 config.yaml strategies.<name> 的最小止盈档位：
#   scalping 首档止盈 profit_taking_levels[0]=0.003（0.3%），grid min_grid_spacing=0.02，
#   trend tp1_pct=0.02，arbitrage take_profit_pct=0.02。
# 手续费按双边 taker 0.1% 估算；expected_profit = notional * target_pct，
# 要求 >= estimated_fee * multiplier（外加绝对利润下限），防止手续费蚕食与极小仓位磨损。
_MIN_PROFIT_TARGET_PCT = {
    "scalping": 0.003,
    "grid": 0.02,
    "trend": 0.02,
    "arbitrage": 0.02,
}
_MIN_PROFIT_FEE_MULTIPLIER = {
    "scalping": 2.0,
    "grid": 3.0,
    "trend": 4.0,
    "arbitrage": 5.0,
}

from core.strategy_audit import get_strategy_audit_logger


class MarketRegime(Enum):
    """市场状态分类"""
    STRONG_TREND_UP = "strong_trend_up"       # 强上升趋势
    WEAK_TREND_UP = "weak_trend_up"           # 弱上升趋势
    SIDEWAYS = "sideways"                      # 震荡
    WEAK_TREND_DOWN = "weak_trend_down"       # 弱下降趋势
    STRONG_TREND_DOWN = "strong_trend_down"   # 强下降趋势
    HIGH_VOLATILITY = "high_volatility"        # 高波动
    LOW_VOLATILITY = "low_volatility"          # 低波动
    UNKNOWN = "unknown"


class DecisionLevel(Enum):
    """决策层级"""
    STRATEGY = "strategy"      # 策略级
    SYMBOL = "symbol"          # 币种级
    PORTFOLIO = "portfolio"    # 组合级
    GLOBAL = "global"          # 全局级


class AlertSeverity(Enum):
    """报警严重程度"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AgentHealth(Enum):
    """智能体健康状态"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"       # 性能下降但可用
    STALE_DATA = "stale_data"   # 市场数据过期
    OVER_REJECTING = "over_rejecting"  # 拒绝率过高
    RECOVERING = "recovering"   # 自动恢复中


@dataclass
class DecisionStats:
    """决策统计"""
    total_audits: int = 0
    approved: int = 0
    rejected: int = 0
    reduced: int = 0
    delayed: int = 0
    rejection_reasons: Dict[str, int] = field(default_factory=dict)
    last_audit_time: Optional[datetime] = None
    last_approve_time: Optional[datetime] = None
    consecutive_rejections: int = 0
    rejection_rate_1h: float = 0.0
    _recent_actions: deque = field(default_factory=lambda: deque(maxlen=200))


@dataclass
class AgentDecision:
    """智能体决策结果"""
    decision_id: str
    timestamp: datetime
    level: DecisionLevel
    action: str  # approve, reject, reduce, delay, escalate
    reason: str
    confidence: float  # 0-1
    source: str  # 决策来源模块
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketSnapshot:
    """市场快照"""
    symbol: str
    price: float
    volume_24h: float
    change_24h: float
    volatility_24h: float
    regime: MarketRegime
    trend_strength: float  # 0-1, 1=最强趋势
    liquidity_score: float  # 0-1, 1=最流动性
    timestamp: datetime


class IntelligentTradingAgent:
    """智能交易体 v2 - 生产级核心决策引擎"""

    # 状态持久化版本
    STATE_VERSION = 2
    STATE_FILENAME = "intelligent_agent_state.json"

    def __init__(self, config: Optional[Dict] = None):
        self._config = config or {}

        # 状态持久化路径
        data_dir = self._config.get("data_dir", "./data")
        self._state_path = Path(data_dir) / self.STATE_FILENAME

        # 市场状态
        self._market_regimes: Dict[str, MarketRegime] = {}
        self._market_snapshots: Dict[str, deque] = {}  # symbol -> deque of snapshots

        # 统一市场状态引擎（可选注入，复用 MarketRegimeEngine 的多因子融合结果）
        # 注入后 get_market_regime / _select_active_strategies / audit_signal 优先使用真实 regime，
        # 未注入时回退到旧版 change_24h/volatility 分类，保证向后兼容。
        self._regime_engine = None

        # 企业级：多层信号前置过滤链（可选注入；注入后 audit_signal 优先走统一过滤器框架，
        # 未注入时回退到下方手写串行维度，保证向后兼容）
        self._signal_pre_filter_chain = None

        # 企业级：统一防抖动引擎（可选注入；注入后 AntiDebounceFilter 生效）
        self._debounce_engine = None

        # 指标流水线（可选注入；用于把决策动作/拒绝率/学习质量/自愈动作外发到统一观测面）
        self._metrics_pipeline = None

        # 决策历史
        self._decision_history: deque = deque(maxlen=1000)
        self._rejected_decisions: deque = deque(maxlen=500)

        # 性能追踪
        self._strategy_performance: Dict[str, Dict] = {}
        self._symbol_performance: Dict[str, Dict] = {}

        # P11-1: 符号级黑名单
        self._symbol_blacklist: Dict[str, Dict] = {}
        self._symbol_detailed_stats: Dict[str, Dict] = {}

        # P1: 正期望符号白名单（键为 symbol，值为 {reason, updated_at, stats, seeded}）
        self._symbol_whitelist: Dict[str, Dict] = {}

        # P11-2: 策略自动暂停
        self._strategy_pause: Dict[str, Dict] = {}

        # P2-12: 策略永久退出（不再自动恢复）
        self._strategy_exit: Dict[str, Dict] = {}  # {strategy_name: {reason, exited_at, cumulative_pnl, recovery_count}}
        self._strategy_recovery_count: Dict[str, int] = {}  # 自动恢复次数
        self._max_recovery_attempts = self._config.get("max_recovery_attempts", 3)  # 最多自动恢复3次
        self._exit_cumulative_pnl_threshold = self._config.get("exit_cumulative_pnl_threshold", -50)  # 累计亏损阈值
        self._exit_min_win_rate = self._config.get("exit_min_win_rate", 0.05)  # 退出胜率阈值
        self._exit_min_trades = self._config.get("exit_min_trades", 50)  # 退出最少交易数
        self._exit_min_profit_factor = self._config.get("exit_min_profit_factor", 0.5)  # 退出盈亏比阈值
        self._exit_negative_expectation_trades = self._config.get(
            "exit_negative_expectation_trades", 50
        )  # 负期望退出最少交易数

        # P11-3: 时段风险配置
        self._high_risk_hours: set = self._config.get("high_risk_hours", {4, 8, 11, 13, 17, 20, 22})
        self._medium_risk_hours: set = self._config.get("medium_risk_hours", {0, 1, 2, 15, 16, 21})
        self._hour_risk_multiplier: Dict[int, float] = {}

        # 学习参数
        self._learned_params: Dict[str, Any] = {}
        self._learning_epoch = 0
        self._last_learning_time = datetime.min

        # 企业级硬约束：各策略 min_signal_quality 不可跌破的硬下限（红线）
        self._signal_quality_floors: Dict[str, float] = {
            "grid": 0.35,      # Grid 不得低于 0.35（高风时段 0.50 另由时段风控处理）
            "scalping": 0.18,  # Scalping 不得低于 0.18
        }

        # 报警状态
        self._active_alerts: Dict[str, Dict] = {}
        self._alert_cooldowns: Dict[str, float] = {}

        # P12: 决策统计
        self._decision_stats = DecisionStats()

        # P12: 健康状态
        self._health = AgentHealth.HEALTHY
        self._last_health_check = datetime.now()
        self._health_check_interval = self._config.get("health_check_interval_seconds", 300)
        self._last_market_update = datetime.min
        self._stale_data_threshold = self._config.get("stale_data_threshold_seconds", 1800)
        self._max_rejection_rate = self._config.get("max_rejection_rate", 0.70)
        self._cumulative_reject_rate_threshold = self._config.get("cumulative_reject_rate_threshold", 0.70)
        self._min_audits_for_health_check = self._config.get("min_audits_for_health_check", 20)
        self._auto_reset_cooldown = self._config.get("auto_reset_cooldown_seconds", 3600)
        self._last_reset_time = datetime.min

        # P12: 决策审计追踪
        self._audit_trail: deque = deque(maxlen=self._config.get("audit_trail_max", 500))

        # 参数
        self.max_snapshots_per_symbol = 100
        self.learning_interval_hours = 6
        self.min_confidence_threshold = self._config.get("min_confidence_threshold", 0.45)
        # 企业级：自适应置信度阈值封顶，避免低胜率/无历史策略把阈值推高形成拒单死亡螺旋
        self._adaptive_conf_cap = self._config.get("adaptive_confidence_cap", 0.50)
        self.max_consecutive_rejections = 5
        self._auto_pause_consecutive_losses = self._config.get("auto_pause_consecutive_losses", 5)
        self._auto_pause_hours = self._config.get("auto_pause_hours", 2)
        self._symbol_blacklist_threshold = self._config.get("symbol_blacklist_pnl_threshold", -100)
        self._symbol_blacklist_days = self._config.get("symbol_blacklist_days", 1)

        # P20: 动态连续亏损阈值 - 基于账户规模和策略类型
        self._consecutive_loss_base = self._config.get("consecutive_loss_base", 3)  # 基础阈值
        self._consecutive_loss_small_account = self._config.get("consecutive_loss_small_account", 8)  # 小账户(<100 USDT)
        self._consecutive_loss_medium_account = self._config.get("consecutive_loss_medium_account", 6)  # 中账户(100-500 USDT)
        self._consecutive_loss_large_account = self._config.get("consecutive_loss_large_account", 5)  # 大账户(>500 USDT)
        self._consecutive_loss_grid_bonus = self._config.get("consecutive_loss_grid_bonus", 3)  # 网格策略额外容忍
        self._consecutive_loss_scalping_bonus = self._config.get("consecutive_loss_scalping_bonus", -1)  # 剥头皮更严格

        # P32: 策略权重 - 动态分配
        self._strategy_weights: Dict[str, float] = {
            "grid": 0.35, "trend": 0.30, "scalping": 0.20, "arbitrage": 0.15
        }
        self._strategy_active: Dict[str, bool] = {
            "grid": True, "trend": True, "scalping": True, "arbitrage": True
        }
        self._last_strategy_switch = datetime.min
        self._strategy_switch_cooldown = 300  # 5分钟冷却

        # P12: 尝试加载持久化状态
        self._load_state()

        # P0 止血：确定性硬黑名单（负期望网格符号），不依赖学习周期数据口径
        self._apply_p0_hard_blacklist()

        # P1: 正期望符号白名单种子（待 6h 学习重排后自动覆盖）
        self._apply_p1_initial_whitelist()

    # ========== 市场状态感知 ==========

    def update_market_state(self, symbol: str, price: float, volume_24h: float,
                           change_24h: float, volatility_24h: float) -> MarketSnapshot:
        """更新市场状态"""
        # 判断市场状态（优先融合引擎，回退旧版 change_24h/volatility 分类）
        fused = self._get_fused_regime_data(symbol)
        if fused is not None:
            regime = fused["local_regime"]
        else:
            regime = self._classify_market_regime(change_24h, volatility_24h)

        # 趋势强度
        if fused is not None:
            trend_strength = min(1.0, abs(fused["trend_strength"]))
        else:
            trend_strength = min(1.0, abs(change_24h) / 0.1)  # 10%变化=满强度

        # 流动性评分
        liquidity_score = min(1.0, volume_24h / 10000000)  # 10M USDT=满流动性

        snapshot = MarketSnapshot(
            symbol=symbol,
            price=price,
            volume_24h=volume_24h,
            change_24h=change_24h,
            volatility_24h=volatility_24h,
            regime=regime,
            trend_strength=trend_strength,
            liquidity_score=liquidity_score,
            timestamp=datetime.now(),
        )

        if symbol not in self._market_snapshots:
            self._market_snapshots[symbol] = deque(maxlen=self.max_snapshots_per_symbol)
        self._market_snapshots[symbol].append(snapshot)
        self._market_regimes[symbol] = regime
        self._last_market_update = datetime.now()

        return snapshot

    def _classify_market_regime(self, change_24h: float, volatility_24h: float) -> MarketRegime:
        """分类市场状态"""
        if volatility_24h > 0.15:
            return MarketRegime.HIGH_VOLATILITY
        if volatility_24h < 0.02:
            return MarketRegime.LOW_VOLATILITY

        if change_24h > 0.05:
            return MarketRegime.STRONG_TREND_UP
        if change_24h > 0.02:
            return MarketRegime.WEAK_TREND_UP
        if change_24h < -0.05:
            return MarketRegime.STRONG_TREND_DOWN
        if change_24h < -0.02:
            return MarketRegime.WEAK_TREND_DOWN

        return MarketRegime.SIDEWAYS

    # ========== 统一市场状态引擎注入（复用 MarketRegimeEngine 多因子融合） ==========

    def set_regime_engine(self, engine) -> None:
        """注入 MarketRegimeEngine，使智能体的市场状态感知复用已强化的多因子融合结果。"""
        self._regime_engine = engine
        logger.info("MarketRegimeEngine injected into IntelligentTradingAgent")

    def set_signal_pre_filter_chain(self, chain) -> None:
        """注入多层信号前置过滤链（SignalPreFilterChain），audit_signal 优先走统一过滤器框架。"""
        self._signal_pre_filter_chain = chain
        logger.info("SignalPreFilterChain injected into IntelligentTradingAgent")

    def set_debounce_engine(self, engine) -> None:
        """注入统一防抖动引擎（AntiDebounceEngine），供过滤链 AntiDebounceFilter 使用。"""
        self._debounce_engine = engine
        logger.info("AntiDebounceEngine injected into IntelligentTradingAgent")

    def set_metrics_pipeline(self, pipeline) -> None:
        """注入 MetricsPipeline，把决策动作/拒绝率/学习质量/自愈动作外发到统一指标流水线。

        未注入或 record 失败时静默降级，绝不阻塞决策主链路。
        """
        self._metrics_pipeline = pipeline
        logger.info("MetricsPipeline injected into IntelligentTradingAgent")

    def _record_metric(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """安全埋点：将单个指标写入 MetricsPipeline，异常时仅 debug 日志不抛出。"""
        pipeline = self._metrics_pipeline
        if pipeline is None:
            return
        try:
            pipeline.record(name, value, labels=labels)
        except Exception as e:  # noqa: BLE001 - 埋点失败不得影响决策
            logger.debug(f"MetricsPipeline.record({name}) failed: {e}")

    @staticmethod
    def _map_engine_regime_to_local(regime_str: str, trend_strength: float = 0.0) -> MarketRegime:
        """将 MarketRegimeEngine 的 regime 字符串映射到本地 MarketRegime 枚举。

        引擎 trend_strength 范围 -1..1（|值|>0.35 视为强趋势），据此区分强弱，
        避免把温和趋势一律判为 STRONG 而误杀网格策略。
        """
        if not regime_str:
            return MarketRegime.UNKNOWN
        if regime_str == "trend_bullish":
            return MarketRegime.STRONG_TREND_UP if abs(trend_strength) > 0.35 else MarketRegime.WEAK_TREND_UP
        if regime_str == "trend_bearish":
            return MarketRegime.STRONG_TREND_DOWN if abs(trend_strength) > 0.35 else MarketRegime.WEAK_TREND_DOWN
        if regime_str == "range_bound":
            return MarketRegime.SIDEWAYS
        if regime_str in ("extreme_volatility", "funding_crush", "liquidity_crisis"):
            return MarketRegime.HIGH_VOLATILITY
        return MarketRegime.UNKNOWN

    def _get_fused_regime_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取融合后的市场状态数据（优先 MarketRegimeEngine，不可用时返回 None）。"""
        engine = self._regime_engine
        if engine is None:
            return None
        try:
            data = engine.get_symbol_regime(symbol)
        except Exception as e:
            logger.debug(f"Fused regime data unavailable for {symbol}: {e}")
            return None
        if not data:
            return None
        regime_str = data.get("regime")
        if not regime_str or regime_str == "unknown":
            return None
        trend_strength = float(data.get("trend_strength", 0.0))
        return {
            "regime_str": regime_str,
            "local_regime": self._map_engine_regime_to_local(regime_str, trend_strength),
            "trend_strength": trend_strength,
            "volatility": float(data.get("volatility", 0.0)),
            "strength": float(data.get("strength", 0.0)),
            "confidence": float(data.get("confidence", 0.0)),
        }

    def get_market_regime(self, symbol: str) -> MarketRegime:
        """获取市场状态（优先融合引擎，回退旧版 change_24h/volatility 分类）"""
        fused = self._get_fused_regime_data(symbol)
        if fused is not None:
            return fused["local_regime"]
        return self._market_regimes.get(symbol, MarketRegime.UNKNOWN)

    # P32: 策略自动切换引擎
    def _select_active_strategies(self, symbol: str, atr: float = 0, adx: float = 0) -> Dict[str, float]:
        """基于市场状态动态选择策略权重

        优先使用 MarketRegimeEngine 的多因子融合结果（regime + trend_strength），
        回退到旧版 ADX/ATR 近似。Returns: 策略名称 -> 权重
        """
        now = datetime.now()
        if (now - self._last_strategy_switch).total_seconds() < self._strategy_switch_cooldown:
            return self._strategy_weights  # 冷却期内不切换

        # 默认权重
        weights = {"grid": 0.35, "trend": 0.30, "scalping": 0.20, "arbitrage": 0.15}
        active = {"grid": True, "trend": True, "scalping": True, "arbitrage": True}

        # 优先：真实多因子 regime 驱动（消除 ADX proxy 近似）
        fused = self._get_fused_regime_data(symbol)
        if fused is not None:
            regime_str = fused["regime_str"]
            abs_trend = abs(fused["trend_strength"])

            if regime_str in ("extreme_volatility", "funding_crush", "liquidity_crisis"):
                # 极端状态：仅剥头皮快进快出，暂停网格/趋势
                weights = {"grid": 0.10, "trend": 0.05, "scalping": 0.70, "arbitrage": 0.15}
                active["grid"] = False
                active["trend"] = False
                logger.info(f"P32: Strategy switch -> SCALPING ONLY for {symbol} ({regime_str})")
            elif abs_trend > 0.35:
                # 强趋势：趋势策略主导
                weights = {"grid": 0.15, "trend": 0.55, "scalping": 0.15, "arbitrage": 0.15}
                logger.info(f"P32: Strategy switch -> TREND mode for {symbol} (trend_strength={fused['trend_strength']:.2f})")
            elif abs_trend < 0.1:
                # 纯震荡：网格为主
                weights = {"grid": 0.50, "trend": 0.10, "scalping": 0.25, "arbitrage": 0.15}
                logger.info(f"P32: Strategy switch -> GRID mode for {symbol} (range, |trend|={abs_trend:.2f})")
            else:
                # 温和震荡：网格 + 剥头皮均衡
                weights = {"grid": 0.40, "trend": 0.10, "scalping": 0.35, "arbitrage": 0.15}
                logger.info(f"P32: Strategy switch -> GRID+SCALP for {symbol} (|trend|={abs_trend:.2f})")

            self._strategy_weights = weights
            self._strategy_active = active
            self._last_strategy_switch = now
            return weights

        # 回退：旧版 ADX/ATR 近似（regime_engine 未注入或无数据时）
        snapshots = self._market_snapshots.get(symbol)
        if not snapshots or len(snapshots) < 5:
            return weights

        prices = [s.price for s in list(snapshots)[-20:] if s.price > 0]
        if len(prices) < 5:
            return weights

        # 计算ATR比例（如果没有传入ATR）
        if atr <= 0:
            avg_price = sum(prices) / len(prices)
            price_range = max(prices) - min(prices)
            atr_ratio = price_range / avg_price if avg_price > 0 else 0
        else:
            avg_price = prices[-1]
            atr_ratio = atr / avg_price if avg_price > 0 else 0

        # 使用ADX判断趋势（如果没有传入ADX，使用价格方向变化率）
        if adx <= 0:
            if len(prices) >= 10:
                first_half = sum(prices[:len(prices)//2]) / (len(prices)//2)
                second_half = sum(prices[len(prices)//2:]) / (len(prices) - len(prices)//2)
                price_change = (second_half - first_half) / first_half if first_half > 0 else 0
                adx_proxy = min(100, abs(price_change) * 500)  # 近似ADX
            else:
                adx_proxy = 20
        else:
            adx_proxy = adx

        # 策略切换逻辑
        if adx_proxy > 25:
            # 趋势市场：趋势策略主导
            weights = {"grid": 0.15, "trend": 0.55, "scalping": 0.15, "arbitrage": 0.15}
            active["grid"] = True  # 网格仍可运行但权重降低
            logger.info(
                f"P32: Strategy switch -> TREND mode for {symbol} "
                f"(ADX={adx_proxy:.1f}, ATR_ratio={atr_ratio:.4f})"
            )
        elif adx_proxy < 20:
            # 震荡市场：网格+剥头皮主导
            if atr_ratio > 0.03:
                # 高波动震荡：剥头皮为主
                weights = {"grid": 0.25, "trend": 0.05, "scalping": 0.55, "arbitrage": 0.15}
                active["trend"] = False
                logger.info(
                    f"P32: Strategy switch -> SCALPING mode for {symbol} "
                    f"(high vol range, ADX={adx_proxy:.1f}, ATR_ratio={atr_ratio:.4f})"
                )
            elif atr_ratio < 0.008:
                # 低波动：仅剥头皮，网格暂停
                weights = {"grid": 0.05, "trend": 0.05, "scalping": 0.75, "arbitrage": 0.15}
                active["grid"] = False
                active["trend"] = False
                logger.info(
                    f"P32: Strategy switch -> SCALPING ONLY for {symbol} "
                    f"(low vol, ADX={adx_proxy:.1f}, ATR_ratio={atr_ratio:.4f})"
                )
            else:
                # 正常震荡：网格为主
                weights = {"grid": 0.50, "trend": 0.10, "scalping": 0.25, "arbitrage": 0.15}
                logger.info(
                    f"P32: Strategy switch -> GRID mode for {symbol} "
                    f"(range, ADX={adx_proxy:.1f}, ATR_ratio={atr_ratio:.4f})"
                )
        else:
            # 过渡状态：保持当前权重
            pass

        self._strategy_weights = weights
        self._strategy_active = active
        self._last_strategy_switch = now
        return weights

    def get_active_strategies(self) -> Dict[str, bool]:
        """获取当前活跃策略"""
        return dict(self._strategy_active)

    def get_strategy_weights(self) -> Dict[str, float]:
        """获取当前策略权重"""
        return dict(self._strategy_weights)

    def refresh_strategy_weights(self, symbol: str, atr: float = 0, adx: float = 0) -> Dict[str, float]:
        """供调度器周期性调用，驱动策略切换引擎（优先融合真实多因子 regime）。"""
        return self._select_active_strategies(symbol, atr=atr, adx=adx)

    def get_trend_strength(self, symbol: str) -> float:
        """获取趋势强度"""
        snapshots = self._market_snapshots.get(symbol)
        if not snapshots:
            return 0.0
        return snapshots[-1].trend_strength

    # ========== 信号审核 ==========

    def _get_dynamic_consecutive_loss_threshold(self, strategy_name: str) -> int:
        """P20: 根据账户规模和策略类型动态计算连续亏损暂停阈值

        小账户(<100 USDT)需要更高容忍度，避免正常波动被误判。
        网格策略天然依赖价差获利，单笔胜率不代表策略有效性，给予额外容忍度。
        剥头皮策略高频交易，对连续亏损更敏感，降低阈值。
        """
        equity = self._config.get("total_capital", 100.0)

        # 账户规模分级
        if equity < 100.0:
            base_threshold = self._consecutive_loss_small_account  # 8
        elif equity < 500.0:
            base_threshold = self._consecutive_loss_medium_account  # 6
        else:
            base_threshold = self._consecutive_loss_large_account  # 5

        # 策略类型调整
        if "grid" in strategy_name.lower():
            base_threshold += self._consecutive_loss_grid_bonus  # +3
        elif "scalping" in strategy_name.lower() or "scalp" in strategy_name.lower():
            base_threshold += self._consecutive_loss_scalping_bonus  # -1

        return max(3, base_threshold)  # 绝对下限3

    def audit_signal(
        self,
        symbol: str,
        strategy_name: str,
        signal_type: str,
        direction: str,
        price: float,
        quantity: float,
        confidence: float,
        is_close: bool = False,
    ) -> AgentDecision:
        """审核交易信号，多维度判断是否应该执行

        Returns:
            AgentDecision with action: approve/reject/reduce/delay
        """
        decision_id = f"audit_{int(time.time()*1000)}_{symbol}"

        # P12: 自动健康检查
        self._maybe_health_check()

        # 企业级：统一多层前置过滤链优先（注入后收敛各维度 + 短路 + 可解释报告），
        # 框架异常时回退到下方手写串行维度，保证下单链路不因框架故障中断。
        if self._signal_pre_filter_chain is not None:
            try:
                return self._audit_via_chain(
                    decision_id, symbol, strategy_name, signal_type,
                    direction, price, quantity, confidence, is_close,
                )
            except Exception as e:
                logger.warning(f"SignalPreFilterChain failed, falling back to legacy audit_signal: {e}")

        # P11-1: 维度0 - 币种×策略黑名单检查（最高优先级）
        # 先查复合键 symbol|strategy，再查全局 symbol 键（兼容旧格式与全局拉黑）
        for key in (self._make_blacklist_key(symbol, strategy_name), symbol):
            if key not in self._symbol_blacklist:
                continue
            blacklist_info = self._symbol_blacklist[key]
            if blacklist_info["until"] > datetime.now():
                _bl_symbol, _bl_strategy = self._parse_blacklist_key(key)
                strategy_label = _bl_strategy or "全局"
                decision = AgentDecision(
                    decision_id=decision_id,
                    timestamp=datetime.now(),
                    level=DecisionLevel.SYMBOL,
                    action="reject",
                    reason=f"Symbol {symbol}（策略 {strategy_label}）在黑名单中: {blacklist_info['reason']}，解禁时间 {blacklist_info['until'].strftime('%H:%M')}",
                    confidence=0.95,
                    source="audit_blacklist",
                    details={"blacklist_info": blacklist_info, "strategy": _bl_strategy},
                )
                self._record_decision(decision, symbol, strategy_name, confidence)
                return decision
            else:
                # 黑名单已过期，自动清除
                del self._symbol_blacklist[key]
                logger.info(f"P11-1: {key} blacklist expired, auto-cleared")
                self._save_state()

        # P11-2: 维度0.5 - 策略自动暂停检查
        if strategy_name in self._strategy_pause:
            pause_info = self._strategy_pause[strategy_name]
            if pause_info["until"] > datetime.now() and not is_close:
                decision = AgentDecision(
                    decision_id=decision_id,
                    timestamp=datetime.now(),
                    level=DecisionLevel.STRATEGY,
                    action="reject",
                    reason=f"策略 {strategy_name} 已暂停: {pause_info['reason']}，恢复时间 {pause_info['until'].strftime('%H:%M')}",
                    confidence=0.9,
                    source="audit_strategy_pause",
                    details={"pause_info": pause_info},
                )
                self._record_decision(decision, symbol, strategy_name, confidence)
                return decision
            elif pause_info["until"] <= datetime.now():
                del self._strategy_pause[strategy_name]
                logger.info(f"P11-2: {strategy_name} auto-pause expired, strategy resumed")
                self._save_state()

        # 维度1: 置信度检查（自适应阈值）
        adaptive_threshold = self._get_adaptive_confidence_threshold(strategy_name)
        if confidence < adaptive_threshold:
            decision = AgentDecision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                level=DecisionLevel.STRATEGY,
                action="reject",
                reason=f"信号置信度过低: {confidence:.2f} < {adaptive_threshold:.2f} (adaptive)",
                confidence=confidence,
                source="audit_confidence",
                details={"adaptive_threshold": adaptive_threshold},
            )
            self._record_decision(decision, symbol, strategy_name, confidence)
            return decision

        # 维度2: 市场状态适配
        regime = self.get_market_regime(symbol)
        if not is_close:
            regime_ok, regime_reason = self._check_regime_compatibility(strategy_name, regime, direction)
            if not regime_ok:
                decision = AgentDecision(
                    decision_id=decision_id,
                    timestamp=datetime.now(),
                    level=DecisionLevel.SYMBOL,
                    action="reject",
                    reason=regime_reason,
                    confidence=0.7,
                    source="audit_regime",
                )
                self._record_decision(decision, symbol, strategy_name, confidence)
                return decision

        # 维度2.5: 趋势方向一致性（regime 方向 + 整体 strength 交叉验证）
        # 独立于策略层方向检测，用 MarketRegimeEngine 的融合 regime 做兜底防御：
        # 中等强度逆势（0.35 < strength < 0.6，强趋势已在 RegimeGate 硬拒）软降仓而非硬拒，避免误杀。
        if not is_close and direction in ("long", "short"):
            fused = self._get_fused_regime_data(symbol)
            if fused is not None:
                regime_str = fused["regime_str"]
                strength = fused["strength"]
                opposing = (
                    (regime_str == "trend_bearish" and direction == "long")
                    or (regime_str == "trend_bullish" and direction == "short")
                )
                if opposing and strength > 0.35:
                    reduced_qty = quantity * 0.6
                    decision = AgentDecision(
                        decision_id=decision_id,
                        timestamp=datetime.now(),
                        level=DecisionLevel.SYMBOL,
                        action="reduce",
                        reason=f"趋势方向与信号相反 (regime={regime_str}, strength={strength:.2f}, direction={direction})，仓位降至60%: {reduced_qty:.4f}",
                        confidence=0.7,
                        source="audit_trend_alignment",
                        details={"reduced_quantity": reduced_qty, "regime": regime_str, "strength": strength},
                    )
                    self._record_decision(decision, symbol, strategy_name, confidence)
                    return decision

        # 维度3: 策略表现检查 - P20动态阈值
        perf = self._strategy_performance.get(strategy_name, {})
        consecutive_losses = perf.get("consecutive_losses", 0)
        dynamic_threshold = self._get_dynamic_consecutive_loss_threshold(strategy_name)
        if consecutive_losses >= dynamic_threshold and not is_close:
            decision = AgentDecision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                level=DecisionLevel.STRATEGY,
                action="delay",
                reason=f"策略连续亏损 {consecutive_losses} 次(阈值={dynamic_threshold})，暂停开仓",
                confidence=0.8,
                source="audit_performance",
                details={"consecutive_losses": consecutive_losses, "dynamic_threshold": dynamic_threshold},
            )
            self._record_decision(decision, symbol, strategy_name, confidence)
            return decision
        # P20: 中间预警阈值 = 动态阈值的60%，不低于3
        early_warning = max(3, int(dynamic_threshold * 0.6))
        if consecutive_losses >= early_warning and not is_close:
            logger.warning(
                f"P20 early warning: {strategy_name} consecutive_losses={consecutive_losses} "
                f"(threshold={dynamic_threshold}, early_warning={early_warning}), "
                f"monitoring but not blocking"
            )

        # 维度4: 交易频率检查
        symbol_perf = self._symbol_performance.get(symbol, {})
        recent_trades = symbol_perf.get("trades_last_hour", 0)
        if recent_trades > 10 and not is_close:
            decision = AgentDecision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                level=DecisionLevel.SYMBOL,
                action="delay",
                reason=f"{symbol} 1小时内交易 {recent_trades} 次，频率过高",
                confidence=0.65,
                source="audit_frequency",
            )
            self._record_decision(decision, symbol, strategy_name, confidence)
            return decision

        # P11-3: 维度5 - 时段风险调整
        if not is_close:
            hour_risk = self._get_hour_risk_level()
            if hour_risk == "high":
                reduced_qty = quantity * 0.4
                # 量纲对齐：高风险时段门槛 = 正常自适应阈值 + 0.1，硬地板 0.50。
                # 旧 0.60 地板与 IDE 0.43 全局阈值量纲错配（价差 +17pp），且因
                # adaptive_threshold 封顶 0.50，max(0.60, 0.6) 恒为 0.60，使
                # 「+0.1 时段溢价」完全失效。恢复「正常门槛+0.1」的时段溢价，
                # 地板降至与 mean_reversion 归一化「刚过线=0.5」一致，防止误杀。
                risk_confidence = max(adaptive_threshold + 0.1, 0.50)
                if confidence < risk_confidence:
                    decision = AgentDecision(
                        decision_id=decision_id,
                        timestamp=datetime.now(),
                        level=DecisionLevel.GLOBAL,
                        action="reject",
                        reason=f"高风险时段({datetime.now().hour}时)，置信度{confidence:.2f}<{risk_confidence:.2f}",
                        confidence=0.85,
                        source="audit_hour_risk",
                    )
                    self._record_decision(decision, symbol, strategy_name, confidence)
                    return decision
                decision = AgentDecision(
                    decision_id=decision_id,
                    timestamp=datetime.now(),
                    level=DecisionLevel.GLOBAL,
                    action="reduce",
                    reason=f"高风险时段({datetime.now().hour}时)，仓位降至40%: {reduced_qty:.4f}",
                    confidence=0.8,
                    source="audit_hour_risk",
                    details={"reduced_quantity": reduced_qty, "risk_level": "high"},
                )
                self._record_decision(decision, symbol, strategy_name, confidence)
                return decision
            elif hour_risk == "medium":
                reduced_qty = quantity * 0.7
                decision = AgentDecision(
                    decision_id=decision_id,
                    timestamp=datetime.now(),
                    level=DecisionLevel.GLOBAL,
                    action="reduce",
                    reason=f"中风险时段({datetime.now().hour}时)，仓位降至70%: {reduced_qty:.4f}",
                    confidence=0.7,
                    source="audit_hour_risk",
                    details={"reduced_quantity": reduced_qty, "risk_level": "medium"},
                )
                self._record_decision(decision, symbol, strategy_name, confidence)
                return decision

        # 维度6: 高波动市场减少仓位
        if regime == MarketRegime.HIGH_VOLATILITY and not is_close:
            reduced_qty = quantity * 0.5
            decision = AgentDecision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                level=DecisionLevel.SYMBOL,
                action="reduce",
                reason=f"高波动市场，建议减少仓位至 {reduced_qty:.4f}",
                confidence=0.75,
                source="audit_volatility",
                details={"reduced_quantity": reduced_qty},
            )
            self._record_decision(decision, symbol, strategy_name, confidence)
            return decision

        # P30: 维度7 - 最低盈利阈值检查（防止手续费蚕食利润）
        if not is_close and price > 0 and quantity > 0:
            min_profit = self._check_minimum_profit_threshold(
                symbol, strategy_name, price, quantity, direction
            )
            if not min_profit:
                decision = AgentDecision(
                    decision_id=decision_id,
                    timestamp=datetime.now(),
                    level=DecisionLevel.SYMBOL,
                    action="reject",
                    reason=f"预期利润不足以覆盖手续费，拒绝交易以保护资金",
                    confidence=0.85,
                    source="audit_min_profit",
                )
                self._record_decision(decision, symbol, strategy_name, confidence)
                return decision

        # P32: 维度8 - 磨损型交易检测(WTI)
        # 数据分析：62.1%交易在-0.01~0 USDT区间，持续磨损资金
        if not is_close and price > 0 and quantity > 0:
            is_wear_type, wear_reason = self._check_wear_type_trade(
                symbol, strategy_name, price, quantity
            )
            if is_wear_type:
                decision = AgentDecision(
                    decision_id=decision_id,
                    timestamp=datetime.now(),
                    level=DecisionLevel.SYMBOL,
                    action="reject",
                    reason=f"WTI: 检测到磨损型交易 - {wear_reason}",
                    confidence=0.80,
                    source="audit_wear_type",
                )
                self._record_decision(decision, symbol, strategy_name, confidence)
                return decision

        # P32: 维度9 - 多时间框架确认
        # 确保信号方向与更大时间框架的市场趋势一致
        if not is_close and direction in ("long", "short"):
            mtf_confirmed, mtf_reason = self._check_multi_timeframe(
                symbol, strategy_name, signal_type, direction, price
            )
            if not mtf_confirmed:
                decision = AgentDecision(
                    decision_id=decision_id,
                    timestamp=datetime.now(),
                    level=DecisionLevel.SYMBOL,
                    action="reject",
                    reason=f"MTF: 多时间框架未确认 - {mtf_reason}",
                    confidence=0.75,
                    source="audit_multi_timeframe",
                )
                self._record_decision(decision, symbol, strategy_name, confidence)
                return decision

        # 通过审核
        decision = AgentDecision(
            decision_id=decision_id,
            timestamp=datetime.now(),
            level=DecisionLevel.STRATEGY,
            action="approve",
            reason="审核通过",
            confidence=confidence,
            source="audit_pass",
        )
        self._record_decision(decision, symbol, strategy_name, confidence)
        return decision

    # ========== 企业级：统一多层前置过滤链桥接 ==========

    def _audit_via_chain(self, decision_id, symbol, strategy_name, signal_type,
                         direction, price, quantity, confidence, is_close) -> AgentDecision:
        """走统一过滤器框架，把 FilterChainResult 映射回 AgentDecision。"""
        from core.signal_pre_filter import SignalContext

        ctx = SignalContext(
            symbol=symbol, strategy_name=strategy_name, signal_type=signal_type,
            direction=direction, price=price, quantity=quantity,
            confidence=confidence, is_close=is_close,
        )
        state = self._build_pre_filter_state(symbol, strategy_name, signal_type, direction, price, quantity, is_close)
        result = self._signal_pre_filter_chain.evaluate(ctx, state)
        decision = self._chain_result_to_decision(decision_id, symbol, strategy_name, confidence, result)

        # 企业级：开仓信号审核通过后记录到防抖动引擎，冷却期内拦截同品种+方向重复开仓。
        # 平仓信号不记录（风控退路不受防抖动限制）；record 异常不阻断主下单链路。
        if decision.action == "approve" and not is_close and self._debounce_engine is not None:
            try:
                self._debounce_engine.record(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    direction=direction,
                    signal_type=signal_type,
                    is_close=False,
                    pnl_usdt=0.0,
                )
            except Exception as e:
                logger.debug(f"AntiDebounceEngine record failed: {e}")

        return decision

    def _build_pre_filter_state(self, symbol, strategy_name, signal_type, direction, price, quantity, is_close) -> Dict[str, Any]:
        """把智能体内部状态组装为过滤链所需 state（重字段惰性求值，保持短路性能）。"""
        state: Dict[str, Any] = {}

        # 维度0/0.5：黑名单 / 策略暂停（含过期清理副作用）
        hit, reason, until, label = self._resolve_blacklist_hit(symbol, strategy_name)
        state["blacklist"] = {"hit": hit, "reason": reason, "until": until, "strategy_label": label}
        hit, reason = self._resolve_strategy_pause_hit(strategy_name)
        state["strategy_paused"] = {"hit": hit, "reason": reason}

        # 维度1：自适应置信度阈值
        state["adaptive_threshold"] = self._get_adaptive_confidence_threshold(strategy_name)

        # 维度2：市场状态兼容性 + 高波动
        regime = self.get_market_regime(symbol)
        state["high_volatility"] = (regime == MarketRegime.HIGH_VOLATILITY)
        if is_close:
            state["regime_ok"] = True
            state["regime_reason"] = "OK"
        else:
            ok, reason = self._check_regime_compatibility(strategy_name, regime, direction)
            state["regime_ok"] = ok
            state["regime_reason"] = reason

        # 维度2.5：趋势方向一致性
        state["trend_alignment"] = self._resolve_trend_alignment(symbol, direction)

        # 维度3：连续亏损
        perf = self._strategy_performance.get(strategy_name, {})
        state["consecutive_losses"] = perf.get("consecutive_losses", 0)
        state["dynamic_threshold"] = self._get_dynamic_consecutive_loss_threshold(strategy_name)

        # 维度4：交易频率
        symbol_perf = self._symbol_performance.get(symbol, {})
        state["trades_last_hour"] = symbol_perf.get("trades_last_hour", 0)
        state["max_trades_per_hour"] = 10

        # 维度5：时段风险
        state["hour_risk_level"] = self._get_hour_risk_level()

        # 维度7：最低盈利阈值（防手续费蚕食利润）
        if is_close or price <= 0 or quantity <= 0:
            state["min_profit_ok"] = True
        else:
            state["min_profit_ok"] = self._check_minimum_profit_threshold(
                symbol, strategy_name, price, quantity, direction
            )

        # 维度8/9：磨损型交易 / 多时间框架（重逻辑，惰性求值以保持短路性能）
        state["wear_type"] = lambda: self._pack_wear_type(symbol, strategy_name, price, quantity)
        state["mtf"] = lambda: self._pack_mtf(symbol, strategy_name, signal_type, direction, price)

        # 企业级：统一防抖动引擎（注入后 AntiDebounceFilter 生效）
        state["debounce_engine"] = self._debounce_engine
        state["volatility"] = self._get_volatility_for_symbol(symbol)
        state["drawdown"] = self._get_current_drawdown()
        state["account_tier"] = self._get_account_tier_for_debounce()

        return state

    def _resolve_blacklist_hit(self, symbol, strategy_name):
        """查询黑名单（复合键优先，回退全局键），过期自动清理。"""
        for key in (self._make_blacklist_key(symbol, strategy_name), symbol):
            if key not in self._symbol_blacklist:
                continue
            info = self._symbol_blacklist[key]
            if info["until"] > datetime.now():
                _, bl_strategy = self._parse_blacklist_key(key)
                return True, info["reason"], info["until"].strftime("%H:%M"), bl_strategy or "全局"
            del self._symbol_blacklist[key]
            logger.info(f"P11-1: {key} blacklist expired, auto-cleared")
            self._save_state()
        return False, "", "", ""

    def _resolve_strategy_pause_hit(self, strategy_name):
        """查询策略暂停状态，过期自动清理。"""
        if strategy_name in self._strategy_pause:
            info = self._strategy_pause[strategy_name]
            if info["until"] > datetime.now():
                return True, info["reason"]
            del self._strategy_pause[strategy_name]
            logger.info(f"P11-2: {strategy_name} auto-pause expired, strategy resumed")
            self._save_state()
        return False, ""

    def _resolve_trend_alignment(self, symbol, direction):
        """趋势方向一致性（复用 MarketRegimeEngine 融合 regime）。"""
        if direction not in ("long", "short"):
            return {"opposing": False, "regime_str": None, "strength": 0.0}
        fused = self._get_fused_regime_data(symbol)
        if fused is None:
            return {"opposing": False, "regime_str": None, "strength": 0.0}
        regime_str = fused["regime_str"]
        strength = fused["strength"]
        opposing = (
            (regime_str == "trend_bearish" and direction == "long")
            or (regime_str == "trend_bullish" and direction == "short")
        )
        return {"opposing": opposing, "regime_str": regime_str, "strength": strength}

    def _pack_wear_type(self, symbol, strategy_name, price, quantity):
        is_wear, reason = self._check_wear_type_trade(symbol, strategy_name, price, quantity)
        return {"hit": is_wear, "reason": reason}

    def _pack_mtf(self, symbol, strategy_name, signal_type, direction, price):
        ok, reason = self._check_multi_timeframe(symbol, strategy_name, signal_type, direction, price)
        return {"ok": ok, "reason": reason}

    def _chain_result_to_decision(self, decision_id, symbol, strategy_name, confidence, result) -> AgentDecision:
        """把 FilterChainResult 映射为 AgentDecision（保持对外接口不变）。"""
        if result.passed:
            decision = AgentDecision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                level=DecisionLevel.STRATEGY,
                action="approve",
                reason="审核通过",
                confidence=confidence,
                source="audit_pass",
                details={"chain_breakdown": result.breakdown},
            )
        else:
            d = result.decision
            level_map = {
                "strategy": DecisionLevel.STRATEGY,
                "symbol": DecisionLevel.SYMBOL,
                "portfolio": DecisionLevel.PORTFOLIO,
                "global": DecisionLevel.GLOBAL,
            }
            decision = AgentDecision(
                decision_id=decision_id,
                timestamp=datetime.now(),
                level=level_map.get(d.level, DecisionLevel.SYMBOL),
                action=d.action,
                reason=d.reason,
                confidence=d.confidence,
                source=d.source,
                details=d.details,
            )
        self._record_decision(decision, symbol, strategy_name, confidence)
        return decision

    def _check_regime_compatibility(
        self, strategy_name: str, regime: MarketRegime, direction: str
    ) -> Tuple[bool, str]:
        """检查策略与市场状态的兼容性"""
        strategy_lower = strategy_name.lower()

        if "grid" in strategy_lower:
            if regime in (MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN):
                return False, f"网格策略不适用于强趋势市场 ({regime.value})"
            if regime == MarketRegime.HIGH_VOLATILITY:
                return False, f"网格策略不适用于高波动市场"

        if "trend" in strategy_lower:
            if regime == MarketRegime.SIDEWAYS:
                return False, f"趋势策略不适用于震荡市场"
            if regime == MarketRegime.LOW_VOLATILITY:
                return False, f"趋势策略在低波动市场信号不可靠"

        if "scalping" in strategy_lower:
            if regime == MarketRegime.LOW_VOLATILITY:
                return False, f"剥头皮策略在低波动市场利润不足"

        if "arbitrage" in strategy_lower:
            if regime == MarketRegime.HIGH_VOLATILITY:
                return False, f"套利策略在高波动市场风险过大"

        return True, "OK"

    # P30: 最低盈利阈值检查
    def _check_minimum_profit_threshold(
        self, symbol: str, strategy_name: str, price: float,
        quantity: float, direction: str
    ) -> bool:
        """P30: 检查交易预期利润是否足以覆盖手续费。

        规则：
        - 估算手续费 = price * quantity * 0.001 (双边0.1%，taker费率)
        - 预期利润 = price * quantity * 目标止盈比例（按策略最小止盈档位）
        - 若预期利润 < max(手续费 * 覆盖倍数, 绝对利润下限) 则拒绝，保护资金。

        Returns:
            True if expected profit is sufficient, False otherwise
        """
        try:
            if price <= 0 or quantity <= 0:
                return True  # 无效输入不拦截，交由下游 min_notional 处理

            # 估算交易名义价值与双边手续费 (taker: 0.05% * 2 = 0.1%)
            notional = price * quantity
            estimated_fee = notional * 0.001

            # 根据策略类型匹配最小止盈目标比例与手续费覆盖倍数
            strategy_lower = strategy_name.lower()
            matched = next(
                (k for k in _MIN_PROFIT_TARGET_PCT if k in strategy_lower),
                None,
            )
            target_pct = _MIN_PROFIT_TARGET_PCT.get(matched, 0.005)
            min_multiplier = _MIN_PROFIT_FEE_MULTIPLIER.get(matched, 3.0)

            min_expected_profit = estimated_fee * min_multiplier
            # 绝对利润下限：极小名义价值交易费用趋近 0 但仍无实际意义
            MIN_ABSOLUTE_PROFIT = 0.0001  # 最低绝对利润 0.0001 USDT
            min_expected_profit = max(min_expected_profit, MIN_ABSOLUTE_PROFIT)

            expected_profit = notional * target_pct
            ok = expected_profit >= min_expected_profit

            if not ok:
                logger.warning(
                    f"P30: Min profit REJECT {symbol} {strategy_name} "
                    f"notional={notional:.4f} expected_profit={expected_profit:.6f} "
                    f"min_required={min_expected_profit:.6f} "
                    f"(fee={estimated_fee:.6f} x{min_multiplier:.0f})"
                )
            else:
                logger.debug(
                    f"P30: Min profit ok {symbol} {strategy_name} "
                    f"notional={notional:.4f} expected_profit={expected_profit:.6f} "
                    f"min_required={min_expected_profit:.6f} "
                    f"(fee={estimated_fee:.6f} x{min_multiplier:.0f})"
                )
            return ok

        except Exception as e:
            logger.debug(f"P30: Min profit check error for {symbol}: {e}")
            return True  # 出错时放行，避免误杀

    # P32: 磨损型交易检测 (Wear-Type Indicator)
    def _check_wear_type_trade(
        self, symbol: str, strategy_name: str, price: float, quantity: float
    ) -> Tuple[bool, str]:
        """P32: 检测磨损型交易 - 无法盈利但持续消耗手续费的交易
        
        检测条件：
        1. 该币种该策略历史平均净PnL < 0 → 拒绝
        2. 该币种最近10笔交易全部亏损 → 拒绝
        3. 该币种该策略净PnL为负且交易>5笔 → 拒绝
        
        Returns:
            (is_wear_type: bool, reason: str)
        """
        try:
            import sqlite3
            db_path = os.path.join("data", "trading.db")
            if not os.path.exists(db_path):
                return False, ""
            
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # 检查1: 该币种该策略历史平均净PnL（trades.pnl_usdt 已为扣除手续费后的净盈亏）
            row = conn.execute(
                "SELECT AVG(pnl_usdt) as avg_net, COUNT(*) as cnt, "
                "SUM(pnl_usdt) as total_net "
                "FROM trades WHERE symbol = ? AND strategy_name = ? "
                "AND replace(exit_time, 'T', ' ') > datetime('now', 'localtime', '-7 days')",
                (symbol, strategy_name)
            ).fetchone()
            
            if row and row['cnt'] >= 5 and (row['avg_net'] or 0) < 0 and (row['total_net'] or 0) < 0:
                # P32 恢复感知：近7天整体亏损，但若最近几笔已转盈利（恢复中），
                # 降级放行，避免7天聚合窗口的滞后把刚恢复的策略继续误杀。
                recovery = conn.execute(
                    "SELECT SUM(pnl_usdt) as recent_net, COUNT(*) as recent_cnt "
                    "FROM trades WHERE symbol = ? AND strategy_name = ? "
                    "ORDER BY exit_time DESC LIMIT 5",
                    (symbol, strategy_name)
                ).fetchone()
                if recovery and recovery['recent_cnt'] >= 3 and (recovery['recent_net'] or 0) > 0:
                    conn.close()
                    logger.info(
                        f"P32: WTI 恢复感知放行 {symbol} {strategy_name} "
                        f"近7天总净PnL={row['total_net']:.4f} 但最近{recovery['recent_cnt']}笔净PnL={recovery['recent_net']:.4f}>0，降级放行"
                    )
                    return False, ""
                conn.close()
                return True, (
                    f"{symbol} {strategy_name} 近7天平均净PnL={row['avg_net']:.6f} "
                    f"(共{row['cnt']}笔，总净PnL={row['total_net']:.4f})，持续磨损资金"
                )
            
            # 检查2: 最近10笔交易全部亏损
            recent = conn.execute(
                "SELECT pnl_usdt FROM trades "
                "WHERE symbol = ? AND strategy_name = ? "
                "ORDER BY exit_time DESC LIMIT 10",
                (symbol, strategy_name)
            ).fetchall()
            
            if len(recent) >= 5:
                all_losing = all((r['pnl_usdt'] or 0) < 0 for r in recent)
                if all_losing:
                    conn.close()
                    return True, (
                        f"{symbol} {strategy_name} 最近{len(recent)}笔全部亏损，"
                        f"疑似磨损型交易模式"
                    )
            
            # 检查3: 该币种近期净PnL为负且交易量大
            recent_net = conn.execute(
                "SELECT SUM(pnl_usdt) as net, COUNT(*) as cnt "
                "FROM trades WHERE symbol = ? "
                "AND replace(exit_time, 'T', ' ') > datetime('now', 'localtime', '-24 hours')",
                (symbol,)
            ).fetchone()
            
            if recent_net and recent_net['cnt'] >= 8 and (recent_net['net'] or 0) < -0.02:
                conn.close()
                return True, (
                    f"{symbol} 近24h净PnL={recent_net['net']:.4f} "
                    f"(共{recent_net['cnt']}笔)，持续产生磨损亏损"
                )
            
            conn.close()
            return False, ""
            
        except Exception as e:
            logger.debug(f"P32: WTI check error for {symbol}: {e}")
            return False, ""

    # P32: 多时间框架确认
    def _check_multi_timeframe(
        self, symbol: str, strategy_name: str, signal_type: str, direction: str, price: float
    ) -> Tuple[bool, str]:
        """P32: 多时间框架确认 - 验证信号方向与更大时间框架趋势一致
        
        策略规则：
        - 网格：短周期(1M)信号 + 中周期(5M)趋势过滤
        - 趋势：长周期(1H)趋势确认 + 中周期(15M)入场
        - 剥头皮：短周期(1M)微观结构 + 中周期(5M)动量
        
        Returns:
            (confirmed: bool, reason: str)
        """
        try:
            snapshots = self._market_snapshots.get(symbol)
            if not snapshots or len(snapshots) < 3:
                return True, "insufficient data"  # 数据不足时放行
            
            snap_list = list(snapshots)
            strategy_lower = strategy_name.lower()
            
            if "grid" in strategy_lower:
                # 网格策略：检查短期价格趋势是否与信号方向一致
                recent_snaps = snap_list[-5:]
                if len(recent_snaps) < 3:
                    return True, "insufficient grid data"
                
                # 短期价格方向
                first_price = recent_snaps[0].price
                last_price = recent_snaps[-1].price
                if first_price > 0:
                    short_trend = (last_price - first_price) / first_price
                else:
                    short_trend = 0
                
                # 网格策略在强趋势中逆势开仓风险高
                if direction == "long" and short_trend < -0.005:
                    return False, f"网格做多信号与短期下跌趋势({short_trend:.3%})冲突"
                if direction == "short" and short_trend > 0.005:
                    return False, f"网格做空信号与短期上涨趋势({short_trend:.3%})冲突"
                
                return True, "grid confirmed"
            
            elif "trend" in strategy_lower:
                # 趋势策略：需要中长期趋势确认
                if len(snap_list) < 10:
                    return True, "insufficient trend data"
                
                mid_snaps = snap_list[-10:]
                mid_start = sum(s.price for s in mid_snaps[:5]) / 5
                mid_end = sum(s.price for s in mid_snaps[-5:]) / 5
                if mid_start > 0:
                    mid_trend = (mid_end - mid_start) / mid_start
                else:
                    mid_trend = 0
                
                if direction == "long" and mid_trend < -0.003:
                    return False, f"趋势做多信号与中期下跌({mid_trend:.3%})冲突"
                if direction == "short" and mid_trend > 0.003:
                    return False, f"趋势做空信号与中期上涨({mid_trend:.3%})冲突"
                
                return True, "trend confirmed"
            
            elif "scalping" in strategy_lower:
                # 剥头皮策略：检查微观动量
                # mean_reversion 是逆动量策略（超跌做多/超涨做空），其 long 信号
                # 往往出现在短期动量向下（超跌）时，顺动量方向检查会系统性误杀。
                # 豁免动量方向检查，交由信号层自身的均值回归判定。
                if signal_type and "mean_reversion" in str(signal_type).lower():
                    return True, "mean_reversion exempt (contrarian)"
                if len(snap_list) < 3:
                    return True, "insufficient scalping data"
                
                recent_3 = snap_list[-3:]
                momentum = sum(
                    (recent_3[i+1].price - recent_3[i].price) / recent_3[i].price 
                    for i in range(len(recent_3)-1) if recent_3[i].price > 0
                )
                
                if direction == "long" and momentum < -0.002:
                    return False, f"剥头皮做多信号与微观动量({momentum:.3%})冲突"
                if direction == "short" and momentum > 0.002:
                    return False, f"剥头皮做空信号与微观动量({momentum:.3%})冲突"
                
                return True, "scalping confirmed"
            
            return True, "unknown strategy"
            
        except Exception as e:
            logger.debug(f"P32: MTF check error for {symbol}: {e}")
            return True, f"error: {e}"  # 出错时放行

    # P11-2: 自适应置信度阈值
    def _get_adaptive_confidence_threshold(self, strategy_name: str) -> float:
        """根据策略历史表现动态调整置信度阈值

        胜率越低，阈值越高（更严格）
        胜率越高，阈值越低（更宽松）
        """
        perf = self._strategy_performance.get(strategy_name, {})
        total = perf.get("total_trades", 0)
        if total < 10:
            return self.min_confidence_threshold

        win_rate = perf.get("wins", 0) / total if total > 0 else 0

        # 胜率映射到阈值: 0%胜率→0.65, 50%胜率→0.45, 100%胜率→0.30
        if win_rate < 0.1:
            base_threshold = 0.65  # 极低胜率，严格过滤
        elif win_rate < 0.3:
            base_threshold = 0.55
        elif win_rate < 0.5:
            base_threshold = 0.45
        elif win_rate < 0.7:
            base_threshold = 0.38
        else:
            base_threshold = 0.30
        # 企业级：阈值封顶，防止低胜率策略把阈值推高到 0.65 形成拒单死亡螺旋
        base_threshold = min(base_threshold, self._adaptive_conf_cap)

        # 学习参数双向 clamp 到 [min_confidence_threshold, cap]，允许下调而非只升不降
        learned = self._learned_params.get(f"{strategy_name}_min_confidence_threshold")
        if learned is not None:
            return min(max(float(learned), self.min_confidence_threshold), self._adaptive_conf_cap)
        return base_threshold

    # P11-3: 时段风险等级
    def _get_hour_risk_level(self) -> str:
        """获取当前时段的风险等级

        基于历史数据分析：
        - 高风险时段 (UTC): 4, 8, 11, 13, 17, 20, 22
        - 中风险时段 (UTC): 0, 1, 2, 15, 16, 21
        - 低风险时段 (UTC): 其余
        """
        current_hour = datetime.now(timezone.utc).hour
        if current_hour in self._high_risk_hours:
            return "high"
        elif current_hour in self._medium_risk_hours:
            return "medium"
        return "low"

    def get_hour_risk_multiplier(self) -> float:
        """获取当前时段仓位乘数（用于外部调用）"""
        risk = self._get_hour_risk_level()
        if risk == "high":
            return 0.4
        elif risk == "medium":
            return 0.7
        return 1.0

    # ========== 防抖动引擎辅助方法 ==========

    def _get_volatility_for_symbol(self, symbol: str) -> float:
        """获取当前品种的波动率（用于 AntiDebounceEngine 自适应冷却）。"""
        try:
            if self._regime_engine is not None:
                regime = self._regime_engine.get_regime(symbol)
                if regime and hasattr(regime, 'factor_scores') and regime.factor_scores:
                    vol_score = regime.factor_scores.get('volatility', 0.0)
                    return max(0.0, min(float(vol_score), 1.0))
        except Exception:
            pass
        return 0.0

    def _get_current_drawdown(self) -> float:
        """获取当前总回撤（用于 AntiDebounceEngine 自适应冷却）。"""
        try:
            perf = self._strategy_performance
            if perf:
                drawdowns = [p.get("max_drawdown", 0.0) for p in perf.values()]
                if drawdowns:
                    return max(0.0, max(drawdowns))
        except Exception:
            pass
        return 0.0

    def _get_account_tier_for_debounce(self) -> str:
        """获取当前账户档位（用于 AntiDebounceEngine 自适应冷却）。"""
        try:
            eq = self._config.get("total_capital", 100.0)
            if eq < 50:
                return "nano"
            elif eq < 100:
                return "micro"
            elif eq < 500:
                return "small"
            elif eq < 2000:
                return "medium"
            elif eq < 10000:
                return "large"
            return "xlarge"
        except Exception:
            pass
        return "small"

    # ========== P11-1: 币种×策略黑名单管理 ==========

    @staticmethod
    def _make_blacklist_key(symbol: str, strategy_name: str = None) -> str:
        """构造黑名单键。

        - strategy_name 为空 → 全局币种键（symbol），对所有策略生效
        - strategy_name 非空 → 复合键 symbol|strategy，仅对该策略生效
        """
        if strategy_name:
            return f"{symbol}|{strategy_name}"
        return symbol

    @staticmethod
    def _parse_blacklist_key(key: str) -> Tuple[str, Optional[str]]:
        """解析黑名单键，返回 (symbol, strategy_name)。

        全局键无 | → strategy_name 为 None；复合键含 | → 拆分为 symbol 与 strategy。
        """
        if "|" in key:
            symbol, strategy = key.split("|", 1)
            return symbol, strategy
        return key, None

    def _apply_p0_hard_blacklist(self):
        """P0 止血：对已确认负期望的符号实施确定性硬黑名单。

        数据依据 data/reports/daily_report_2026-08-17.json 及 trades 表近 7 天统计：
        - UNI-USDT-SWAP 13 笔 -0.071 USDT（网格负期望）
        - AVAX-USDT-SWAP 22 笔 -0.3584 USDT，胜率 9.1%（网格负期望）

        P2（近7天全亏，全局拉黑，对所有策略生效）：
        - SUI-USDT-SWAP scalping 11 笔 -1.77 USDT（含手续费）
        - IOST-USDT-SWAP short -1.11 USDT
        - MINA-USDT-SWAP short -0.34 USDT

        自动学习黑名单（_learn_from_history）因历史口径未及时覆盖这些符号，
        故在此直接写入，保证止血不依赖 6h 学习周期。待 P1 正期望白名单落地后
        移除本硬编码。
        """
        now = datetime.now()
        hard_stops = {
            "UNI-USDT-SWAP|grid": ("P0止血: 网格负期望 (8/17 13笔 -0.071 USDT)", 2),
            "AVAX-USDT-SWAP|grid": ("P0止血: 网格负期望 (近7天 22笔 -0.3584 USDT, 胜率9.1%)", 2),
            "SUI-USDT-SWAP": ("P2拉黑: 7天全亏 (scalping 11笔 -1.77 USDT)", 7),
            "IOST-USDT-SWAP": ("P2拉黑: 7天全亏 (short -1.11 USDT)", 7),
            "MINA-USDT-SWAP": ("P2拉黑: 7天全亏 (short -0.34 USDT)", 7),
        }
        applied = False
        for key, (reason, days) in hard_stops.items():
            existing = self._symbol_blacklist.get(key)
            if existing and existing.get("until") and existing["until"] > now:
                continue
            self._symbol_blacklist[key] = {
                "reason": reason,
                "until": now + timedelta(days=days),
                "added_at": now,
            }
            logger.warning(f"P0止血: {key} 硬黑名单 {days}d: {reason}")
            applied = True
        if applied:
            self._save_state()

    def add_symbol_to_blacklist(self, symbol: str, reason: str, days: int = None,
                                strategy_name: str = None):
        """将币种（或 币种×策略）加入黑名单"""
        if days is None:
            days = self._symbol_blacklist_days
        key = self._make_blacklist_key(symbol, strategy_name)
        self._symbol_blacklist[key] = {
            "reason": reason,
            "until": datetime.now() + timedelta(days=days),
            "added_at": datetime.now(),
        }
        logger.warning(f"P11-1: {key} added to blacklist for {days}d: {reason}")
        self._save_state()

    def remove_symbol_from_blacklist(self, symbol: str, strategy_name: str = None):
        """手动移除黑名单"""
        key = self._make_blacklist_key(symbol, strategy_name)
        if key in self._symbol_blacklist:
            del self._symbol_blacklist[key]
            logger.info(f"P11-1: {key} manually removed from blacklist")
            self._save_state()

    def is_symbol_blacklisted(self, symbol: str, strategy_name: str = None) -> bool:
        """检查币种（或 币种×策略）是否在黑名单中"""
        if strategy_name:
            key = self._make_blacklist_key(symbol, strategy_name)
            if key in self._symbol_blacklist:
                return self._symbol_blacklist[key]["until"] > datetime.now()
        # 全局键兜底
        if symbol in self._symbol_blacklist:
            return self._symbol_blacklist[symbol]["until"] > datetime.now()
        return False

    def get_blacklist(self) -> Dict[str, Dict]:
        """获取完整黑名单（键为 symbol 或 symbol|strategy）"""
        result = {}
        for key, info in self._symbol_blacklist.items():
            if info["until"] > datetime.now():
                symbol, strategy = self._parse_blacklist_key(key)
                result[key] = {
                    "symbol": symbol,
                    "strategy": strategy,
                    "reason": info["reason"],
                    "until": info["until"].isoformat(),
                    "remaining_hours": (info["until"] - datetime.now()).total_seconds() / 3600,
                }
        return result

    def get_global_blacklisted_symbols(self) -> set:
        """返回全局拉黑（对所有策略生效）的币种集合，供平仓兜底使用"""
        symbols = set()
        for key, info in self._symbol_blacklist.items():
            if info["until"] <= datetime.now():
                continue
            symbol, strategy = self._parse_blacklist_key(key)
            if strategy is None:
                symbols.add(symbol)
        return symbols

    def get_blacklisted_strategy_pairs(self) -> set:
        """返回被拉黑的 (symbol, strategy) 复合对集合"""
        pairs = set()
        for key, info in self._symbol_blacklist.items():
            if info["until"] <= datetime.now():
                continue
            symbol, strategy = self._parse_blacklist_key(key)
            if strategy is not None:
                pairs.add((symbol, strategy))
        return pairs

    # ========== P1: 正期望符号白名单管理 ==========

    def _apply_p1_initial_whitelist(self):
        """P1: 初始正期望符号白名单种子。

        数据依据 docs/enterprise_profit_strategy.md Section 5.5 与实盘统计：
        OP-USDT-SWAP 46 笔 +0.107 USDT 为当前唯一明确达标的正期望符号。
        6h 学习周期 rebuild_whitelist() 落地后会自动覆盖本种子。
        """
        now = datetime.now()
        seeds = {
            "OP-USDT-SWAP": {
                "reason": "P1种子: 46笔 +0.107 USDT 正期望",
                "updated_at": now,
                "stats": {
                    "count": 46,
                    "win_rate": None,
                    "total_pnl": 0.107,
                    "profit_factor": None,
                },
                "seeded": True,
            },
        }
        applied = False
        for symbol, info in seeds.items():
            if symbol in self._symbol_whitelist:
                continue
            self._symbol_whitelist[symbol] = info
            logger.info(f"P1: {symbol} 加入初始白名单种子")
            applied = True
        if applied:
            self._save_state()

    def get_whitelisted_symbols(self) -> Set[str]:
        """返回当前正期望白名单符号集合（供 RegimeGate 查询）。"""
        return set(self._symbol_whitelist.keys())

    def is_symbol_whitelisted(self, symbol: str) -> bool:
        """检查符号是否在正期望白名单中。"""
        return symbol in self._symbol_whitelist

    def get_whitelist(self) -> Dict[str, Dict]:
        """获取完整白名单信息。"""
        result = {}
        for symbol, info in self._symbol_whitelist.items():
            updated_at = info.get("updated_at")
            result[symbol] = {
                "reason": info.get("reason", ""),
                "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else updated_at,
                "stats": info.get("stats", {}),
                "seeded": info.get("seeded", False),
            }
        return result

    def rebuild_whitelist(self, trade_records: List[Dict]) -> Dict[str, Any]:
        """P1: 基于近 7 天交易记录重建正期望符号白名单。

        入榜条件（Section 5.5）：
            count >= whitelist_min_trades 且 total_pnl > 0 且
            (win_rate > whitelist_min_win_rate 或 profit_factor > whitelist_min_profit_factor)
        摘牌条件：
            total_pnl < 0 或 max_consecutive_losses >= whitelist_max_consecutive_losses

        Args:
            trade_records: get_recent_trades(limit=1000) 返回的交易记录列表。
        Returns:
            {"added": [...], "removed": [...], "whitelist": {...}}
        """
        now = datetime.now()
        cutoff_7d = now - timedelta(days=7)

        min_trades = int(self._config.get("whitelist_min_trades", 5))
        min_win_rate = float(self._config.get("whitelist_min_win_rate", 0.40))
        min_profit_factor = float(self._config.get("whitelist_min_profit_factor", 1.2))
        max_consecutive_losses = int(self._config.get("whitelist_max_consecutive_losses", 5))

        # 按 symbol 聚合近 7 天交易统计
        symbol_stats: Dict[str, Dict] = {}
        for tr in trade_records or []:
            try:
                exit_time = tr.get("exit_time")
                if isinstance(exit_time, str):
                    exit_time = datetime.fromisoformat(exit_time)
                if exit_time is None or exit_time < cutoff_7d:
                    continue
                symbol = tr.get("symbol", "")
                if not symbol:
                    continue
                pnl = float(tr.get("pnl_usdt", 0) or 0)
                win = bool(tr.get("win", False))

                s = symbol_stats.setdefault(symbol, {
                    "count": 0, "wins": 0, "losses": 0,
                    "gross_profit": 0.0, "gross_loss": 0.0, "total_pnl": 0.0,
                    "consecutive_losses": 0, "max_consecutive_losses": 0,
                })
                s["count"] += 1
                s["total_pnl"] += pnl
                if pnl > 0:
                    s["wins"] += 1
                    s["gross_profit"] += pnl
                    s["consecutive_losses"] = 0
                else:
                    s["losses"] += 1
                    s["gross_loss"] += abs(pnl)
                    s["consecutive_losses"] += 1
                    s["max_consecutive_losses"] = max(
                        s["max_consecutive_losses"], s["consecutive_losses"]
                    )
            except (ValueError, TypeError):
                continue

        new_whitelist: Dict[str, Dict] = {}
        for symbol, s in symbol_stats.items():
            if s["count"] < min_trades:
                continue
            win_rate = s["wins"] / s["count"] if s["count"] > 0 else 0.0
            if s["gross_loss"] > 0:
                profit_factor = s["gross_profit"] / s["gross_loss"]
            elif s["gross_profit"] > 0:
                profit_factor = float("inf")
            else:
                profit_factor = 0.0

            qualifies = (
                s["total_pnl"] > 0
                and (win_rate > min_win_rate or profit_factor > min_profit_factor)
            )
            if qualifies and s["max_consecutive_losses"] < max_consecutive_losses:
                new_whitelist[symbol] = {
                    "reason": (
                        f"P1: 7d {s['count']}笔 pnl={s['total_pnl']:.4f} "
                        f"wr={win_rate:.2f} pf={profit_factor:.2f}"
                    ),
                    "updated_at": now,
                    "stats": {
                        "count": s["count"],
                        "win_rate": win_rate,
                        "total_pnl": s["total_pnl"],
                        "profit_factor": profit_factor,
                    },
                    "seeded": False,
                }

        # 摘牌：对已在白名单但近 7 天转负或连续亏损达标的符号移除
        added: List[str] = []
        removed: List[str] = []
        for symbol in list(self._symbol_whitelist.keys()):
            s = symbol_stats.get(symbol)
            if s is None:
                # 近 7 天无交易：保留，避免频繁抖动
                continue
            if s["total_pnl"] < 0 or s["max_consecutive_losses"] >= max_consecutive_losses:
                del self._symbol_whitelist[symbol]
                removed.append(symbol)

        # 入榜：合并新达标符号
        for symbol, info in new_whitelist.items():
            if symbol not in self._symbol_whitelist:
                self._symbol_whitelist[symbol] = info
                added.append(symbol)

        if added or removed:
            self._save_state()
            logger.info(
                f"P1: 白名单重排 +{len(added)} -{len(removed)}: "
                f"新增={added} 摘牌={removed}"
            )

        return {
            "added": added,
            "removed": removed,
            "whitelist": self.get_whitelist(),
        }

    # ========== P11-2: 策略暂停管理 ==========

    def pause_strategy(self, strategy_name: str, reason: str, hours: int = None):
        """暂停策略开仓"""
        if hours is None:
            hours = self._auto_pause_hours
        self._strategy_pause[strategy_name] = {
            "reason": reason,
            "until": datetime.now() + timedelta(hours=hours),
            "paused_at": datetime.now(),
        }
        logger.warning(f"P11-2: {strategy_name} auto-paused for {hours}h: {reason}")
        self._save_state()
        # P2-12: 审计留痕
        try:
            get_strategy_audit_logger().log("pause", strategy_name, reason=reason, hours=hours)
        except Exception as e:
            logger.debug(f"Strategy audit (pause) error: {e}")

    def resume_strategy(self, strategy_name: str):
        """手动恢复策略（P2-12: 已退出策略不可恢复，需先 reinstate）"""
        if strategy_name in self._strategy_exit:
            logger.warning(
                f"P2-12: Cannot resume '{strategy_name}' — strategy is PERMANENTLY EXITED. "
                f"Use reinstate_strategy() to manually re-enable."
            )
            return
        if strategy_name in self._strategy_pause:
            del self._strategy_pause[strategy_name]
            logger.info(f"P11-2: {strategy_name} manually resumed")
            self._save_state()
            # P2-12: 审计留痕
            try:
                get_strategy_audit_logger().log("resume", strategy_name)
            except Exception as e:
                logger.debug(f"Strategy audit (resume) error: {e}")

    def is_strategy_paused(self, strategy_name: str) -> bool:
        """检查策略是否被暂停"""
        if strategy_name not in self._strategy_pause:
            return False
        return self._strategy_pause[strategy_name]["until"] > datetime.now()

    # ── P2-12: 策略永久退出条件 ──

    def exit_strategy(self, strategy_name: str, reason: str, cumulative_pnl: float = 0.0):
        """永久退出策略（不再自动恢复，需人工确认后手动恢复）

        Args:
            strategy_name: 策略名称
            reason: 退出原因
            cumulative_pnl: 累计盈亏
        """
        self._strategy_exit[strategy_name] = {
            "reason": reason,
            "exited_at": datetime.now(),
            "cumulative_pnl": cumulative_pnl,
            "recovery_count": self._strategy_recovery_count.get(strategy_name, 0),
        }
        logger.error(
            f"P2-12: Strategy '{strategy_name}' PERMANENTLY EXITED: {reason} "
            f"(cumulative PnL={cumulative_pnl:.4f}, "
            f"recovery_attempts={self._strategy_recovery_count.get(strategy_name, 0)})"
        )
        self._save_state()
        # P2-12: 审计留痕
        try:
            get_strategy_audit_logger().log(
                "exit", strategy_name, reason=reason, cumulative_pnl=cumulative_pnl,
                recovery_count=self._strategy_recovery_count.get(strategy_name, 0),
            )
        except Exception as e:
            logger.debug(f"Strategy audit (exit) error: {e}")

    def reinstate_strategy(self, strategy_name: str) -> bool:
        """手动恢复已退出的策略（需人工确认）"""
        if strategy_name in self._strategy_exit:
            exited_info = self._strategy_exit.pop(strategy_name)
            self._strategy_recovery_count.pop(strategy_name, None)
            logger.warning(
                f"P2-12: Strategy '{strategy_name}' MANUALLY REINSTATED "
                f"(was exited since {exited_info['exited_at'].isoformat()}, "
                f"reason: {exited_info['reason']})"
            )
            self._save_state()
            # P2-12: 审计留痕
            try:
                get_strategy_audit_logger().log(
                    "reinstate", strategy_name, reason=exited_info.get("reason", "")
                )
            except Exception as e:
                logger.debug(f"Strategy audit (reinstate) error: {e}")
            return True
        return False

    def is_strategy_exited(self, strategy_name: str) -> bool:
        """检查策略是否已被永久退出"""
        return strategy_name in self._strategy_exit

    def get_exited_strategies(self) -> Dict[str, Dict]:
        """获取已退出策略列表"""
        result = {}
        for name, info in self._strategy_exit.items():
            result[name] = {
                "reason": info["reason"],
                "exited_at": info["exited_at"].isoformat(),
                "cumulative_pnl": info["cumulative_pnl"],
                "recovery_count": info["recovery_count"],
            }
        return result

    def record_recovery_attempt(self, strategy_name: str) -> int:
        """记录一次自动恢复尝试，返回当前恢复次数"""
        count = self._strategy_recovery_count.get(strategy_name, 0) + 1
        self._strategy_recovery_count[strategy_name] = count
        logger.warning(
            f"P2-12: Strategy '{strategy_name}' recovery attempt #{count} "
            f"(max={self._max_recovery_attempts})"
        )
        return count

    def check_exit_conditions(self, strategy_name: str, win_rate: float,
                              cumulative_pnl: float, trade_count: int,
                              profit_factor: Optional[float] = None) -> bool:
        """检查策略是否满足退出条件

        Args:
            strategy_name: 策略名称
            win_rate: 胜率
            cumulative_pnl: 累计盈亏
            trade_count: 交易笔数
            profit_factor: 盈亏比（毛利/毛损，None 表示未提供，跳过盈亏比条件）

        Returns:
            True 如果策略应被永久退出
        """
        recovery_count = self._strategy_recovery_count.get(strategy_name, 0)
        avg_pnl = cumulative_pnl / trade_count if trade_count > 0 else 0.0

        # 条件1: 自动恢复次数超限且仍亏损
        if recovery_count >= self._max_recovery_attempts and cumulative_pnl < 0:
            self.exit_strategy(
                strategy_name,
                f"自动恢复 {recovery_count} 次后仍亏损 (PnL={cumulative_pnl:.4f})",
                cumulative_pnl,
            )
            return True

        # 条件2: 累计亏损超过阈值 + 胜率过低 + 足够交易量
        if (cumulative_pnl <= self._exit_cumulative_pnl_threshold
                and win_rate < self._exit_min_win_rate
                and trade_count >= self._exit_min_trades):
            self.exit_strategy(
                strategy_name,
                f"累计亏损 {cumulative_pnl:.2f} USDT, 胜率 {win_rate:.1%}, "
                f"{trade_count} 笔交易 — 负期望策略永久退出",
                cumulative_pnl,
            )
            return True

        # 条件2b: 盈亏比过低（毛利远小于毛损）+ 足够交易量
        if (profit_factor is not None
                and profit_factor < self._exit_min_profit_factor
                and trade_count >= self._exit_min_trades
                and cumulative_pnl < 0):
            self.exit_strategy(
                strategy_name,
                f"盈亏比过低 {profit_factor:.2f} (< {self._exit_min_profit_factor:.2f}) "
                f"且累计亏损 {cumulative_pnl:.2f} USDT ({trade_count}笔) — 永久退出",
                cumulative_pnl,
            )
            return True

        # 条件2c: 负期望（平均每笔亏损）+ 足够交易量 + 胜率低于盈亏平衡
        if (avg_pnl < 0
                and win_rate < 0.5
                and trade_count >= self._exit_negative_expectation_trades
                and cumulative_pnl < 0):
            self.exit_strategy(
                strategy_name,
                f"负期望策略：平均每笔 {avg_pnl:.4f} USDT, 胜率 {win_rate:.1%}, "
                f"{trade_count} 笔 — 永久退出",
                cumulative_pnl,
            )
            return True

        # 条件3: 策略暂停/恢复循环 ≥ 5 次（反复失败）
        if recovery_count >= 5:
            self.exit_strategy(
                strategy_name,
                f"反复暂停/恢复 {recovery_count} 次，判定为失效策略",
                cumulative_pnl,
            )
            return True

        return False

    # ── P11-2: 策略暂停/恢复（增强：记录恢复次数） ──

    def get_paused_strategies(self) -> Dict[str, Dict]:
        """获取暂停策略列表"""
        result = {}
        for name, info in self._strategy_pause.items():
            if info["until"] > datetime.now():
                result[name] = {
                    "reason": info["reason"],
                    "until": info["until"].isoformat(),
                    "remaining_hours": (info["until"] - datetime.now()).total_seconds() / 3600,
                }
        return result

    # ========== 成长性学习 ==========

    def _clamp_learned_signal_quality(self) -> int:
        """纠正 learned_params 中各策略 min_signal_quality 跌破硬约束的坏值。

        历史事故：learned_params 曾残留 scalping=0.12 / grid=0.3，跌破硬约束
        （scalping≥0.18、grid≥0.35），导致信号质量阈值被错误压低。此方法在
        状态加载后与每次学习调整后调用，将跌破下界的值 clamp 回硬约束，避免回归。
        """
        corrected = 0
        for strategy, floor in self._signal_quality_floors.items():
            key = f"{strategy}_min_signal_quality"
            if key in self._learned_params:
                val = self._learned_params[key]
                if val is None or val < floor:
                    self._learned_params[key] = floor
                    corrected += 1
                    logger.warning(
                        f"P0: 修正学习参数 {key} 跌破硬约束 {val} -> {floor}"
                    )
        if corrected:
            self._record_metric("agent_param_clamp_total", float(corrected))
        return corrected

    def learn_from_history(self, trade_records: List[Dict]) -> Dict[str, Any]:
        """从历史交易中学习，调整参数

        P11增强：
        - P11-1: 符号级PnL统计与黑名单自动管理
        - P11-2: 策略自动暂停（连续亏损N次）
        - P11-3: 时段风险系数动态学习
        """
        self._learning_epoch += 1
        self._last_learning_time = datetime.now()

        if not trade_records:
            return {"epoch": self._learning_epoch, "learned": False, "reason": "无交易记录"}

        # 仅统计近 _LEARNING_WINDOW_HOURS 小时内的成交（按平仓时间 exit_time 过滤），
        # 避免全历史亏损永久触发负期望退出/黑名单误判。无 exit_time 的记录保留以向后兼容。
        _window_cutoff = datetime.now() - timedelta(hours=_LEARNING_WINDOW_HOURS)
        _windowed_records = []
        for _trade in trade_records:
            _exit_ts = _trade.get("exit_time")
            _keep = True
            if _exit_ts:
                try:
                    _exit_dt = (_exit_ts if isinstance(_exit_ts, datetime)
                                else datetime.fromisoformat(str(_exit_ts)))
                    _keep = _exit_dt >= _window_cutoff
                except (ValueError, TypeError):
                    _keep = True
            if _keep:
                _windowed_records.append(_trade)
        trade_records = _windowed_records
        if not trade_records:
            return {"epoch": self._learning_epoch, "learned": False, "reason": "近24h无交易记录"}

        # 按策略分组统计
        strategy_stats = {}
        # P11-1: 按币种分组统计
        symbol_stats: Dict[str, Dict] = {}
        # P11-1: 按 币种×策略 分组统计（复合维度黑名单）
        symbol_strategy_stats: Dict[str, Dict] = {}
        # P11-3: 按时段分组统计
        hour_stats: Dict[int, Dict] = {}

        for trade in trade_records:
            strategy = trade.get("strategy_name", "unknown")
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    "wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0,
                    "gross_profit": 0.0, "gross_loss": 0.0,
                }
            stats = strategy_stats[strategy]
            stats["count"] += 1
            pnl = trade.get("pnl_usdt", 0)
            stats["total_pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
                stats["gross_profit"] += pnl
            else:
                stats["losses"] += 1
                stats["gross_loss"] += abs(pnl)

            # P11-1: 符号级统计
            symbol = trade.get("symbol", "unknown")
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}
            sym_stats = symbol_stats[symbol]
            sym_stats["count"] += 1
            sym_stats["total_pnl"] += pnl
            if pnl > 0:
                sym_stats["wins"] += 1
            else:
                sym_stats["losses"] += 1

            # P11-1: 币种×策略复合统计
            if symbol != "unknown" and strategy != "unknown":
                skey = self._make_blacklist_key(symbol, strategy)
                if skey not in symbol_strategy_stats:
                    symbol_strategy_stats[skey] = {"wins": 0, "losses": 0, "total_pnl": 0.0, "count": 0}
                ss = symbol_strategy_stats[skey]
                ss["count"] += 1
                ss["total_pnl"] += pnl
                if pnl > 0:
                    ss["wins"] += 1
                else:
                    ss["losses"] += 1

            # P11-3: 时段统计
            try:
                ts = trade.get("timestamp", "")
                if ts:
                    if isinstance(ts, str):
                        hour = int(ts[11:13]) if len(ts) >= 13 else -1
                    else:
                        hour = ts.hour if hasattr(ts, "hour") else -1
                    if hour >= 0:
                        if hour not in hour_stats:
                            hour_stats[hour] = {"count": 0, "total_pnl": 0.0}
                        hour_stats[hour]["count"] += 1
                        hour_stats[hour]["total_pnl"] += pnl
            except (ValueError, IndexError):
                pass

        # P11-1: 币种×策略黑名单自动更新（复合维度）
        blacklist_updates = []
        
        # P29: 账户规模判断——用 total_capital 精确判定（<100 微账户 / <500 小账户）
        # P0: 原 total_abs_pnl<10 启发式会被单笔大额亏损（如 AVAX|scalping -12.88）误导，改用账户权益
        equity = self._config.get("total_capital", 100.0)
        if equity < 100.0:
            is_small_account = True
            pnl_threshold = -0.03
        elif equity < 500.0:
            is_small_account = True
            pnl_threshold = -0.30
        else:
            is_small_account = False
            pnl_threshold = self._symbol_blacklist_threshold
        
        # P31: 小账户降低最低交易笔数要求，从20降到5
        min_trade_count = 5 if is_small_account else 20
        
        for skey, stats in symbol_strategy_stats.items():
            symbol, strategy = self._parse_blacklist_key(skey)
            # 排除非交易类策略（如 sync 持仓同步）
            if strategy in ("sync", "unknown"):
                continue
            if stats["count"] >= min_trade_count:
                win_rate = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
                # P31: 账户规模分档阈值（微账户 -0.03 / 小账户 -0.30 / 其余默认 -100）
                effective_threshold = pnl_threshold
                
                # 条件1: 累计亏损超过阈值
                if stats["total_pnl"] < effective_threshold:
                    if skey not in self._symbol_blacklist:
                        days = self._symbol_blacklist_days
                        # 亏损越大，黑名单时间越长
                        if stats["total_pnl"] < -1000:
                            days = 3
                        elif stats["total_pnl"] < -500:
                            days = 2
                        self.add_symbol_to_blacklist(
                            symbol,
                            f"累计亏损 {stats['total_pnl']:.4f} USDT ({stats['count']}笔, 胜率{win_rate:.1%})",
                            days=days,
                            strategy_name=strategy,
                        )
                        blacklist_updates.append({
                            "symbol": symbol, "strategy": strategy, "action": "added",
                            "total_pnl": stats["total_pnl"],
                            "count": stats["count"], "days": days,
                        })
                # P31: 条件1b - 小账户低胜率+亏损（5+笔，胜率<20%，PnL<-0.02）
                elif is_small_account and win_rate < 0.20 and stats["count"] >= 5 and stats["total_pnl"] < -0.02:
                    if skey not in self._symbol_blacklist:
                        self.add_symbol_to_blacklist(
                            symbol,
                            f"小账户低胜率+亏损 {win_rate:.1%} ({stats['count']}笔, PnL={stats['total_pnl']:.4f})",
                            days=1,
                            strategy_name=strategy,
                        )
                        blacklist_updates.append({
                            "symbol": symbol, "strategy": strategy, "action": "added_low_wr_small",
                            "win_rate": win_rate, "count": stats["count"],
                            "total_pnl": stats["total_pnl"],
                        })
                # 条件2: 胜率低于10%且交易超过50笔
                elif win_rate < 0.1 and stats["count"] >= 50:
                    if skey not in self._symbol_blacklist:
                        self.add_symbol_to_blacklist(
                            symbol,
                            f"极低胜率 {win_rate:.1%} ({stats['count']}笔)",
                            days=2,
                            strategy_name=strategy,
                        )
                        blacklist_updates.append({
                            "symbol": symbol, "strategy": strategy, "action": "added_low_wr",
                            "win_rate": win_rate, "count": stats["count"],
                        })
                # P29: 条件2.5 - 小账户低胜率多笔交易：胜率<20%且30+笔且累计亏损，自动黑名单
                elif win_rate < 0.20 and stats["count"] >= 30 and stats["total_pnl"] < -0.01:
                    if skey not in self._symbol_blacklist:
                        self.add_symbol_to_blacklist(
                            symbol,
                            f"小账户低胜率 {win_rate:.1%} ({stats['count']}笔, PnL={stats['total_pnl']:.4f})",
                            days=1,
                            strategy_name=strategy,
                        )
                        blacklist_updates.append({
                            "symbol": symbol, "strategy": strategy, "action": "added_low_wr_small_acct",
                            "win_rate": win_rate, "count": stats["count"],
                            "total_pnl": stats["total_pnl"],
                        })
                # 条件3: 盈利恢复自动解除该币种×策略黑名单
                elif stats["total_pnl"] > 0 and skey in self._symbol_blacklist:
                    self.remove_symbol_from_blacklist(symbol, strategy_name=strategy)
                    blacklist_updates.append({
                        "symbol": symbol, "strategy": strategy, "action": "removed_profitable",
                        "total_pnl": stats["total_pnl"],
                    })

        # 更新符号级详细统计（保留原 symbol 维度展示）
        for symbol, stats in symbol_stats.items():
            self._symbol_detailed_stats[symbol] = stats

        # P11-2: 策略自动暂停
        pause_updates = []
        exit_updates = []
        for strategy, stats in strategy_stats.items():
            if stats["count"] < 10:
                continue

            win_rate = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
            avg_pnl = stats["total_pnl"] / stats["count"] if stats["count"] > 0 else 0

            # 连续亏损检查: 通过performance追踪
            perf = self._strategy_performance.get(strategy, {})
            consecutive_losses = perf.get("consecutive_losses", 0)

            # P20: 连续亏损动态阈值
            dynamic_threshold = self._get_dynamic_consecutive_loss_threshold(strategy)

            # 胜率低于5%且交易超过50笔 → 自动暂停24小时
            if win_rate < 0.05 and stats["count"] >= 50 and strategy not in self._strategy_pause:
                self.pause_strategy(
                    strategy,
                    f"胜率极低 {win_rate:.1%} ({stats['count']}笔, 总PnL={stats['total_pnl']:.2f})",
                    hours=24,
                )
                pause_updates.append({
                    "strategy": strategy, "action": "paused_24h",
                    "win_rate": win_rate, "total_pnl": stats["total_pnl"],
                })

            # P20: 连续亏损达到动态阈值 → 自动暂停
            elif consecutive_losses >= dynamic_threshold and strategy not in self._strategy_pause:
                # P20: 根据账户规模动态调整暂停时长
                equity = self._config.get("total_capital", 100.0)
                if equity < 100.0:
                    pause_hours = max(1, self._auto_pause_hours // 2)  # 小账户减半暂停时间
                else:
                    pause_hours = self._auto_pause_hours
                self.pause_strategy(
                    strategy,
                    f"连续亏损 {consecutive_losses} 次(阈值={dynamic_threshold})",
                    hours=pause_hours,
                )
                pause_updates.append({
                    "strategy": strategy, "action": "paused_consecutive",
                    "consecutive_losses": consecutive_losses,
                    "dynamic_threshold": dynamic_threshold,
                })

            # 胜率恢复 → 自动解除暂停
            elif win_rate > 0.4 and strategy in self._strategy_pause:
                self.resume_strategy(strategy)
                # P2-12: 记录恢复尝试
                self.record_recovery_attempt(strategy)
                pause_updates.append({
                    "strategy": strategy, "action": "resumed",
                    "win_rate": win_rate,
                })

            # P2-12: 检查策略退出条件
            if strategy not in self._strategy_exit:
                cumulative_pnl = stats.get("total_pnl", 0.0)
                trade_count = stats.get("count", 0)
                gross_profit = stats.get("gross_profit", 0.0)
                gross_loss = stats.get("gross_loss", 0.0)
                profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else None
                if self.check_exit_conditions(
                    strategy, win_rate, cumulative_pnl, trade_count,
                    profit_factor=profit_factor,
                ):
                    exit_updates.append({
                        "strategy": strategy, "action": "exited",
                        "win_rate": win_rate, "cumulative_pnl": cumulative_pnl,
                        "trade_count": trade_count,
                    })
                    # 跳过已退出策略的后续学习调整
                    continue

            # 原有学习调整
            learned = {}

            if win_rate < 0.4:
                learned["min_signal_quality"] = min(0.5, self._learned_params.get(
                    f"{strategy}_min_signal_quality", 0.3) + 0.05)
                learned["min_confidence_threshold"] = min(self._adaptive_conf_cap, self._learned_params.get(
                    f"{strategy}_min_confidence_threshold", 0.35) + 0.05)
                if "reason" not in learned:
                    learned["reason"] = f"胜率过低 {win_rate:.1%}，提高信号质量阈值"

            if avg_pnl < 0:
                learned["position_multiplier"] = max(0.3, self._learned_params.get(
                    f"{strategy}_position_multiplier", 1.0) - 0.1)
                if "reason" not in learned:
                    learned["reason"] = f"平均亏损 {avg_pnl:.4f}，减少仓位"

            if win_rate > 0.6 and avg_pnl < 0.01:
                learned["take_profit_multiplier"] = self._learned_params.get(
                    f"{strategy}_take_profit_multiplier", 1.0) + 0.1
                if "reason" not in learned:
                    learned["reason"] = f"高胜率低盈亏，放宽止盈"

            self._learned_params.update({f"{strategy}_{k}": v for k, v in learned.items()
                                        if k not in ("reason",)})

            logger.info(
                f"Agent learning [{self._learning_epoch}]: {strategy} "
                f"win_rate={win_rate:.1%} avg_pnl={avg_pnl:.4f} "
                f"learned={learned}"
            )

        # P11-3: 时段风险系数动态学习
        hour_learned = []
        for hour, stats in hour_stats.items():
            if stats["count"] >= 10:
                avg_hour_pnl = stats["total_pnl"] / stats["count"]
                # 按小时平均亏损幅度调整风险等级
                if avg_hour_pnl < -2.0:
                    if hour not in self._high_risk_hours:
                        self._high_risk_hours.add(hour)
                        hour_learned.append({"hour": hour, "level": "high", "avg_pnl": avg_hour_pnl})
                elif avg_hour_pnl < -0.5:
                    if hour not in self._medium_risk_hours and hour not in self._high_risk_hours:
                        self._medium_risk_hours.add(hour)
                        hour_learned.append({"hour": hour, "level": "medium", "avg_pnl": avg_hour_pnl})
                elif avg_hour_pnl > 0.5:
                    # 盈利时段，从风险集合中移除
                    if hour in self._high_risk_hours:
                        self._high_risk_hours.discard(hour)
                        hour_learned.append({"hour": hour, "level": "removed_high", "avg_pnl": avg_hour_pnl})
                    if hour in self._medium_risk_hours:
                        self._medium_risk_hours.discard(hour)
                        hour_learned.append({"hour": hour, "level": "removed_medium", "avg_pnl": avg_hour_pnl})

        if hour_learned:
            logger.info(f"P11-3: Hour risk updated: {hour_learned}")

        # P0: 学习调整后统一 clamp，防止 min_signal_quality 跌破硬约束
        self._clamp_learned_signal_quality()

        # P12: 学习完成后持久化状态
        self._save_state()

        # 埋点：学习质量外发到 MetricsPipeline
        self._record_metric("agent_learning_epoch", float(self._learning_epoch))
        self._record_metric("agent_learned_params_count", float(len(self._learned_params)))
        self._record_metric("agent_paused_strategies_count", float(len(self.get_paused_strategies())))
        self._record_metric("agent_blacklist_count", float(len(self.get_blacklist())))

        return {
            "epoch": self._learning_epoch,
            "learned": True,
            "strategy_stats": strategy_stats,
            "symbol_stats": symbol_stats,
            "learned_params": dict(self._learned_params),
            "blacklist_updates": blacklist_updates,
            "pause_updates": pause_updates,
            "exit_updates": exit_updates,
            "hour_risk_updates": hour_learned,
            "blacklist": self.get_blacklist(),
            "paused_strategies": self.get_paused_strategies(),
            "regime_distribution": self._get_regime_distribution(),
        }

    def _get_regime_distribution(self) -> Dict[str, int]:
        """从 MarketRegimeEngine 拉取当前各币种 regime 分布（用于学习循环输出市场结构）。"""
        engine = self._regime_engine
        if engine is None or not hasattr(engine, "get_all_symbol_regimes"):
            return {}
        try:
            all_regimes = engine.get_all_symbol_regimes()
        except Exception as e:
            logger.debug(f"Regime distribution unavailable: {e}")
            return {}
        dist: Dict[str, int] = {}
        for sym_data in all_regimes.values():
            if isinstance(sym_data, dict):
                regime_str = sym_data.get("regime", "unknown")
                dist[regime_str] = dist.get(regime_str, 0) + 1
        return dist

    def get_learned_param(self, strategy_name: str, param_name: str, default: Any = None) -> Any:
        """获取学习到的参数"""
        return self._learned_params.get(f"{strategy_name}_{param_name}", default)

    # ========== 分层协调 ==========

    def coordinate_strategies(
        self, active_signals: List[Dict]
    ) -> List[AgentDecision]:
        """协调多个策略的信号，避免冲突"""
        decisions = []

        # 按symbol分组
        by_symbol: Dict[str, List[Dict]] = {}
        for signal in active_signals:
            symbol = signal.get("symbol", "")
            if symbol not in by_symbol:
                by_symbol[symbol] = []
            by_symbol[symbol].append(signal)

        for symbol, signals in by_symbol.items():
            if len(signals) <= 1:
                continue

            # 检查是否有冲突方向
            directions = set(s.get("direction", "") for s in signals)
            if len(directions) > 1:
                # 同一symbol有冲突信号，按优先级选
                priorities = {s.get("strategy_name", ""): s.get("priority", 0) for s in signals}
                best = max(signals, key=lambda s: s.get("priority", 0))

                for s in signals:
                    if s != best:
                        decisions.append(AgentDecision(
                            decision_id=f"coord_{int(time.time()*1000)}_{symbol}",
                            timestamp=datetime.now(),
                            level=DecisionLevel.SYMBOL,
                            action="reject",
                            reason=f"方向冲突：{best['strategy_name']}优先",
                            confidence=0.9,
                            source="coordination",
                        ))

        return decisions

    # ========== 错误报警 ==========

    def raise_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        message: str,
        details: Optional[Dict] = None,
        cooldown_seconds: float = 300,
    ) -> bool:
        """触发分层报警

        Returns:
            True if alert was raised (not in cooldown)
        """
        now = time.time()
        if alert_id in self._alert_cooldowns:
            if now - self._alert_cooldowns[alert_id] < cooldown_seconds:
                return False

        self._alert_cooldowns[alert_id] = now
        self._active_alerts[alert_id] = {
            "severity": severity.value,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now(),
            "count": self._active_alerts.get(alert_id, {}).get("count", 0) + 1,
        }

        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(severity, logger.warning)

        log_func(f"[AGENT-{severity.value.upper()}] {message}")
        return True

    def clear_alert(self, alert_id: str):
        """清除报警"""
        self._active_alerts.pop(alert_id, None)
        self._alert_cooldowns.pop(alert_id, None)

    def get_active_alerts(self) -> Dict[str, Dict]:
        """获取活跃报警"""
        return dict(self._active_alerts)

    # ========== 性能追踪 ==========

    def record_trade_result(self, strategy_name: str, symbol: str, pnl: float, is_win: bool):
        """记录交易结果"""
        if strategy_name not in self._strategy_performance:
            self._strategy_performance[strategy_name] = {
                "total_trades": 0, "wins": 0, "losses": 0,
                "total_pnl": 0.0, "consecutive_losses": 0, "consecutive_wins": 0,
            }
        perf = self._strategy_performance[strategy_name]
        perf["total_trades"] += 1
        perf["total_pnl"] += pnl
        if is_win:
            perf["wins"] += 1
            perf["consecutive_wins"] += 1
            perf["consecutive_losses"] = 0
        else:
            perf["losses"] += 1
            perf["consecutive_losses"] += 1
            perf["consecutive_wins"] = 0

            # P20: 连续亏损达到动态阈值立即触发策略自动暂停
            dynamic_threshold = self._get_dynamic_consecutive_loss_threshold(strategy_name)
            if perf["consecutive_losses"] >= dynamic_threshold:
                if strategy_name not in self._strategy_pause:
                    # P20: 根据账户规模动态调整暂停时长
                    equity = self._config.get("total_capital", 100.0)
                    if equity < 100.0:
                        pause_hours = max(1, self._auto_pause_hours // 2)  # 小账户减半暂停时间
                    else:
                        pause_hours = self._auto_pause_hours
                    self.pause_strategy(
                        strategy_name,
                        f"连续亏损 {perf['consecutive_losses']} 次(阈值={dynamic_threshold}) (总PnL={perf['total_pnl']:.2f})",
                        hours=pause_hours,
                    )
                    self.raise_alert(
                        f"strategy_pause_{strategy_name}",
                        AlertSeverity.WARNING,
                        f"策略 {strategy_name} 连续亏损 {perf['consecutive_losses']} 次(阈值={dynamic_threshold})，自动暂停 {pause_hours} 小时",
                        details={"consecutive_losses": perf["consecutive_losses"], "total_pnl": perf["total_pnl"], "dynamic_threshold": dynamic_threshold},
                    )

        if symbol not in self._symbol_performance:
            self._symbol_performance[symbol] = {"trades_last_hour": 0, "last_trade_time": None}
        sym_perf = self._symbol_performance[symbol]
        now = datetime.now()
        if sym_perf["last_trade_time"] and (now - sym_perf["last_trade_time"]).seconds < 3600:
            sym_perf["trades_last_hour"] += 1
        else:
            sym_perf["trades_last_hour"] = 1
        sym_perf["last_trade_time"] = now

    def should_learn(self) -> bool:
        """判断是否应该执行学习"""
        if self._last_learning_time == datetime.min:
            return True
        return (datetime.now() - self._last_learning_time).total_seconds() >= self.learning_interval_hours * 3600

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要（含P11增强信息）"""
        return {
            "strategies": dict(self._strategy_performance),
            "learned_params": dict(self._learned_params),
            "learning_epoch": self._learning_epoch,
            "active_alerts": len(self._active_alerts),
            "market_regimes": {k: v.value for k, v in self._market_regimes.items()},
            # P11-1: 黑名单状态
            "blacklist": self.get_blacklist(),
            "blacklisted_symbols": [s for s in self._symbol_blacklist if self.is_symbol_blacklisted(s)],
            "symbol_detailed_stats": dict(self._symbol_detailed_stats),
            # P11-2: 策略暂停状态
            "paused_strategies": self.get_paused_strategies(),
            "paused_strategy_names": [s for s in self._strategy_pause if self.is_strategy_paused(s)],
            # P11-3: 时段风险
            "current_hour_risk": self._get_hour_risk_level(),
            "current_hour_multiplier": self.get_hour_risk_multiplier(),
            "high_risk_hours": sorted(list(self._high_risk_hours)),
            "medium_risk_hours": sorted(list(self._medium_risk_hours)),
        }

    # ========== P12: 状态持久化 ==========

    def _save_state(self) -> bool:
        """持久化智能体状态到磁盘

        保存内容：
        - 黑名单 (symbol_blacklist)
        - 策略暂停 (strategy_pause)
        - 学习参数 (learned_params)
        - 学习周期计数 (learning_epoch)
        - 符号详细统计 (symbol_detailed_stats)
        - 策略表现 (strategy_performance)
        - 时段风险配置 (high_risk_hours, medium_risk_hours)
        - 决策统计 (decision_stats)
        - 健康状态 (health)
        """
        try:
            state = {
                "version": self.STATE_VERSION,
                "saved_at": datetime.now().isoformat(),
                "blacklist": {
                    symbol: {
                        "reason": info["reason"],
                        "until": info["until"].isoformat(),
                        "added_at": info["added_at"].isoformat() if isinstance(info["added_at"], datetime) else info["added_at"],
                    }
                    for symbol, info in self._symbol_blacklist.items()
                    if info.get("until") and info["until"] > datetime.now()
                },
                "whitelist": {
                    symbol: {
                        "reason": info.get("reason", ""),
                        "updated_at": info["updated_at"].isoformat() if isinstance(info.get("updated_at"), datetime) else info.get("updated_at"),
                        "stats": info.get("stats", {}),
                        "seeded": info.get("seeded", False),
                    }
                    for symbol, info in self._symbol_whitelist.items()
                },
                "strategy_pause": {
                    name: {
                        "reason": info["reason"],
                        "until": info["until"].isoformat(),
                        "paused_at": info["paused_at"].isoformat() if isinstance(info["paused_at"], datetime) else info["paused_at"],
                    }
                    for name, info in self._strategy_pause.items()
                },
                "learned_params": dict(self._learned_params),
                "learning_epoch": self._learning_epoch,
                "last_learning_time": self._last_learning_time.isoformat() if self._last_learning_time != datetime.min else None,
                "high_risk_hours": sorted(list(self._high_risk_hours)),
                "medium_risk_hours": sorted(list(self._medium_risk_hours)),
                "decision_stats": {
                    "total_audits": self._decision_stats.total_audits,
                    "approved": self._decision_stats.approved,
                    "rejected": self._decision_stats.rejected,
                    "reduced": self._decision_stats.reduced,
                    "delayed": self._decision_stats.delayed,
                    "consecutive_rejections": self._decision_stats.consecutive_rejections,
                    "rejection_rate_1h": self._decision_stats.rejection_rate_1h,
                    "rejection_reasons": dict(self._decision_stats.rejection_reasons),
                },
                "health": self._health.value,
                "last_reset_time": self._last_reset_time.isoformat() if self._last_reset_time != datetime.min else None,
                "last_market_update": self._last_market_update.isoformat() if self._last_market_update != datetime.min else None,
                # P2-12: 策略退出状态
                "strategy_exit": {
                    name: {
                        "reason": info["reason"],
                        "exited_at": info["exited_at"].isoformat(),
                        "cumulative_pnl": info["cumulative_pnl"],
                        "recovery_count": info["recovery_count"],
                    }
                    for name, info in self._strategy_exit.items()
                },
                "strategy_recovery_count": dict(self._strategy_recovery_count),
            }

            # 确保目录存在
            self._state_path.parent.mkdir(parents=True, exist_ok=True)

            # 原子写入：先写临时文件，再重命名
            tmp_path = self._state_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False, default=str)
            tmp_path.replace(self._state_path)

            logger.debug(f"P12: Agent state saved ({len(self._symbol_blacklist)} blacklist, "
                         f"{len(self._strategy_pause)} paused, "
                         f"{len(self._strategy_exit)} exited, "
                         f"{len(self._learned_params)} learned params)")
            return True
        except Exception as e:
            logger.error(f"P12: Failed to save agent state: {e}")
            return False

    def _load_state(self) -> bool:
        """从磁盘加载持久化状态"""
        try:
            if not self._state_path.exists():
                logger.info("P12: No saved state found, starting fresh")
                return False

            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)

            saved_version = state.get("version", 0)
            if saved_version < self.STATE_VERSION:
                logger.info(f"P12: State version {saved_version} < {self.STATE_VERSION}, partial migration")

            saved_at = state.get("saved_at", "unknown")
            logger.info(f"P12: Loading agent state saved at {saved_at}")

            # 恢复黑名单
            for symbol, info in state.get("blacklist", {}).items():
                try:
                    until = datetime.fromisoformat(info["until"])
                    if until > datetime.now():
                        self._symbol_blacklist[symbol] = {
                            "reason": info.get("reason", "saved"),
                            "until": until,
                            "added_at": datetime.fromisoformat(info.get("added_at", saved_at)) if info.get("added_at") else datetime.now(),
                        }
                except (ValueError, KeyError):
                    pass

            # 恢复正期望白名单
            for symbol, info in state.get("whitelist", {}).items():
                try:
                    updated_at = info.get("updated_at")
                    if isinstance(updated_at, str):
                        updated_at = datetime.fromisoformat(updated_at)
                    self._symbol_whitelist[symbol] = {
                        "reason": info.get("reason", ""),
                        "updated_at": updated_at if isinstance(updated_at, datetime) else datetime.now(),
                        "stats": info.get("stats", {}),
                        "seeded": info.get("seeded", False),
                    }
                except (ValueError, KeyError):
                    pass

            # 恢复策略暂停
            for name, info in state.get("strategy_pause", {}).items():
                try:
                    until = datetime.fromisoformat(info["until"])
                    if until > datetime.now():
                        self._strategy_pause[name] = {
                            "reason": info.get("reason", "saved"),
                            "until": until,
                            "paused_at": datetime.fromisoformat(info.get("paused_at", saved_at)) if info.get("paused_at") else datetime.now(),
                        }
                except (ValueError, KeyError):
                    pass

            # 恢复学习参数
            self._learned_params = state.get("learned_params", {})
            self._learning_epoch = state.get("learning_epoch", 0)
            # P0: 加载后立即纠正跌破硬约束的历史坏值（如 scalping=0.12 / grid=0.3）
            self._clamp_learned_signal_quality()

            # 恢复学习时间
            last_lt = state.get("last_learning_time")
            if last_lt:
                try:
                    self._last_learning_time = datetime.fromisoformat(last_lt)
                except ValueError:
                    self._last_learning_time = datetime.min

            # 恢复时段风险
            high_risk = state.get("high_risk_hours", [])
            if high_risk:
                self._high_risk_hours = set(high_risk)
            medium_risk = state.get("medium_risk_hours", [])
            if medium_risk:
                self._medium_risk_hours = set(medium_risk)

            # 恢复决策统计
            ds = state.get("decision_stats", {})
            if ds:
                self._decision_stats.total_audits = ds.get("total_audits", 0)
                self._decision_stats.approved = ds.get("approved", 0)
                self._decision_stats.rejected = ds.get("rejected", 0)
                self._decision_stats.reduced = ds.get("reduced", 0)
                self._decision_stats.delayed = ds.get("delayed", 0)
                self._decision_stats.consecutive_rejections = ds.get("consecutive_rejections", 0)
                self._decision_stats.rejection_rate_1h = ds.get("rejection_rate_1h", 0.0)
                rr = ds.get("rejection_reasons", {})
                if isinstance(rr, dict):
                    self._decision_stats.rejection_reasons = rr

            # 恢复健康状态
            health_val = state.get("health", "healthy")
            try:
                self._health = AgentHealth(health_val)
            except ValueError:
                self._health = AgentHealth.HEALTHY

            # 恢复重置时间
            last_reset = state.get("last_reset_time")
            if last_reset:
                try:
                    self._last_reset_time = datetime.fromisoformat(last_reset)
                except ValueError:
                    pass

            # 恢复市场更新时间
            last_mu = state.get("last_market_update")
            if last_mu:
                try:
                    self._last_market_update = datetime.fromisoformat(last_mu)
                except ValueError:
                    pass

            # P2-12: 恢复策略退出状态
            for name, info in state.get("strategy_exit", {}).items():
                try:
                    self._strategy_exit[name] = {
                        "reason": info.get("reason", "saved"),
                        "exited_at": datetime.fromisoformat(info.get("exited_at", saved_at)),
                        "cumulative_pnl": info.get("cumulative_pnl", 0.0),
                        "recovery_count": info.get("recovery_count", 0),
                    }
                except (ValueError, KeyError):
                    pass

            # P2-12: 恢复自动恢复计数
            self._strategy_recovery_count = state.get("strategy_recovery_count", {})

            logger.info(f"P12: State loaded: {len(self._symbol_blacklist)} blacklist, "
                        f"{len(self._strategy_pause)} paused, "
                        f"{len(self._strategy_exit)} exited, "
                        f"{len(self._learned_params)} learned params, "
                        f"epoch={self._learning_epoch}, health={self._health.value}")
            
            # P30: 如果黑名单为空，从数据库初始化黑名单
            if not self._symbol_blacklist:
                self._init_blacklist_from_db()
            
            return True
        except json.JSONDecodeError as e:
            logger.warning(f"P12: Corrupted state file, starting fresh: {e}")
            # P30: 即使状态文件损坏，也从数据库初始化黑名单
            self._init_blacklist_from_db()
            return False
        except Exception as e:
            logger.error(f"P12: Failed to load agent state: {e}")
            # P30: 即使状态加载失败，也从数据库初始化黑名单
            self._init_blacklist_from_db()
            return False

    def _init_blacklist_from_db(self):
        """P30: 从SQLite数据库初始化黑名单
        
        在系统重启后，从持久化的交易记录中分析各币种表现，
        自动将证明亏损的币种加入黑名单。
        这解决了重启后内存清空导致黑名单丢失的问题。
        """
        db_path = os.path.join("data", "trading.db")
        if not os.path.exists(db_path):
            logger.debug("P30: No trading database found, skipping blacklist init")
            return
        
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            
            # 查询所有已平仓交易的 币种×策略 统计（trades.pnl_usdt 为扣除手续费后的净盈亏）
            # P0: 复合维度黑名单，排除持仓同步类非交易策略
            rows = conn.execute(
                "SELECT symbol, strategy_name, COUNT(*) as cnt, SUM(pnl_usdt) as total_pnl, "
                "SUM(fees) as total_fees, "
                "SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins "
                "FROM trades WHERE strategy_name IS NOT NULL "
                "AND strategy_name NOT IN ('sync', 'unknown', '') "
                "GROUP BY symbol, strategy_name ORDER BY total_pnl ASC"
            ).fetchall()
            
            # P31: 查询近24小时各 币种×策略 表现，用于豁免近期恢复的币种
            recent_rows = conn.execute(
                "SELECT symbol, strategy_name, SUM(pnl_usdt) as recent_pnl, COUNT(*) as recent_cnt, "
                "SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as recent_wins "
                "FROM trades "
                "WHERE replace(exit_time, 'T', ' ') > datetime('now', 'localtime', '-24 hours') "
                "AND strategy_name IS NOT NULL "
                "AND strategy_name NOT IN ('sync', 'unknown', '') "
                "GROUP BY symbol, strategy_name"
            ).fetchall()
            recent_stats = {}
            for r in recent_rows:
                rkey = self._make_blacklist_key(r['symbol'], r['strategy_name'])
                recent_stats[rkey] = {
                    'pnl': r['recent_pnl'] or 0,
                    'cnt': r['recent_cnt'] or 0,
                    'wins': r['recent_wins'] or 0
                }
            
            # 账户规模判断——用 total_capital 精确判定（<100 微账户 / <500 小账户）
            # P0: 原 total_abs_pnl<10 启发式会被单笔大额亏损（如 AVAX|scalping -12.88）误导，改用账户权益
            equity = self._config.get("total_capital", 100.0)
            if equity < 100.0:
                is_small_account = True
                pnl_threshold = -0.03
            elif equity < 500.0:
                is_small_account = True
                pnl_threshold = -0.30
            else:
                is_small_account = False
                pnl_threshold = self._symbol_blacklist_threshold
            
            # P31: 小账户降低最低交易笔数要求，从20降到5
            min_trade_count = 5 if is_small_account else 20
            
            blacklisted = []
            for row in rows:
                symbol = row['symbol']
                strategy = row['strategy_name'] or 'unknown'
                cnt = row['cnt'] or 0
                total_pnl = row['total_pnl'] or 0
                total_fees = row['total_fees'] or 0
                win_rate = row['wins'] / cnt if cnt > 0 else 0
                skey = self._make_blacklist_key(symbol, strategy)
                
                if cnt < min_trade_count:
                    continue
                
                # P31: 近期恢复豁免 - 近24h PnL>0且WR>50%的币种×策略暂不拉黑
                recent = recent_stats.get(skey, {})
                recent_pnl = recent.get('pnl', 0)
                recent_cnt = recent.get('cnt', 0)
                recent_wins = recent.get('wins', 0)
                recent_wr = recent_wins / recent_cnt if recent_cnt > 0 else 0
                if recent_pnl > 0 and recent_wr >= 0.50 and recent_cnt >= 2:
                    logger.info(
                        f"P31: {skey} exempted from blacklist - "
                        f"recent 24h PnL={recent_pnl:.4f}, WR={recent_wr:.1%} ({recent_cnt} trades) "
                        f"(all-time PnL={total_pnl:.4f}, WR={win_rate:.1%})"
                    )
                    continue
                
                # P31: 账户规模分档阈值（微账户 -0.03 / 小账户 -0.30 / 其余默认 -100）
                effective_threshold = pnl_threshold
                
                # 条件1: 累计亏损超过阈值
                if total_pnl < effective_threshold:
                    days = self._symbol_blacklist_days
                    if total_pnl < -1000:
                        days = 3
                    elif total_pnl < -500:
                        days = 2
                    self.add_symbol_to_blacklist(
                        symbol,
                        f"P31-DB: 累计亏损 {total_pnl:.4f} USDT ({cnt}笔, 胜率{win_rate:.1%})",
                        days=days,
                        strategy_name=strategy,
                    )
                    blacklisted.append(skey)
                    continue
                
                # P31: 条件1b - 手续费效率检查（手续费吞噬利润）
                # 如果总手续费 > 5 * abs(总PnL) 且 PnL为负，说明该币种交易成本远高于收益
                elif total_pnl < 0 and total_fees > 0 and total_fees > abs(total_pnl) * 5:
                    self.add_symbol_to_blacklist(
                        symbol,
                        f"P31-DB: 手续费蚕食利润 fees={total_fees:.4f} vs PnL={total_pnl:.4f} ({cnt}笔, 胜率{win_rate:.1%})",
                        days=1,
                        strategy_name=strategy,
                    )
                    blacklisted.append(skey)
                    continue
                
                # P31: 条件1c - 小账户低胜率+亏损（5+笔，胜率<20%，PnL<-0.02）
                elif is_small_account and win_rate < 0.20 and cnt >= 5 and total_pnl < -0.02:
                    self.add_symbol_to_blacklist(
                        symbol,
                        f"P31-DB: 小账户低胜率+亏损 {win_rate:.1%} ({cnt}笔, PnL={total_pnl:.4f})",
                        days=1,
                        strategy_name=strategy,
                    )
                    blacklisted.append(skey)
                    continue
                
                # 条件2: 胜率<10%且50+笔
                elif win_rate < 0.1 and cnt >= 50:
                    self.add_symbol_to_blacklist(
                        symbol,
                        f"P30-DB: 极低胜率 {win_rate:.1%} ({cnt}笔)",
                        days=2,
                        strategy_name=strategy,
                    )
                    blacklisted.append(skey)
                    continue
                
                # 条件3: 小账户胜率<20%且30+笔且亏损
                elif is_small_account and win_rate < 0.20 and cnt >= 30 and total_pnl < -0.01:
                    self.add_symbol_to_blacklist(
                        symbol,
                        f"P30-DB: 小账户低胜率 {win_rate:.1%} ({cnt}笔, PnL={total_pnl:.4f})",
                        days=1,
                        strategy_name=strategy,
                    )
                    blacklisted.append(skey)
                    continue
                
                # P30增强: 胜率<35%且50+笔且累计亏损>0.1 USDT
                elif win_rate < 0.35 and cnt >= 50 and total_pnl < -0.1:
                    self.add_symbol_to_blacklist(
                        symbol,
                        f"P30-DB: 持续低胜率 {win_rate:.1%} ({cnt}笔, PnL={total_pnl:.4f})",
                        days=2,
                        strategy_name=strategy,
                    )
                    blacklisted.append(skey)
                    continue
            
            conn.close()
            
            if blacklisted:
                logger.warning(
                    f"P30: Initialized blacklist from DB: {blacklisted} "
                    f"(small_account={is_small_account}, equity={equity:.2f})"
                )
                # 立即保存状态确保黑名单持久化
                self._save_state()
            else:
                logger.debug(f"P30: No symbol×strategy pairs qualify for blacklist from DB "
                            f"({len(rows)} pairs analyzed)")
                
        except Exception as e:
            logger.error(f"P30: Failed to init blacklist from DB: {e}")

    # ========== P12: 决策记录与审计追踪 ==========

    def _record_decision(
        self,
        decision: AgentDecision,
        symbol: str,
        strategy_name: str,
        confidence: float,
    ):
        """记录决策到统计和审计追踪"""
        stats = self._decision_stats
        stats.total_audits += 1
        stats.last_audit_time = datetime.now()

        if decision.action == "approve":
            stats.approved += 1
            stats.last_approve_time = datetime.now()
            stats.consecutive_rejections = 0
        elif decision.action == "reject":
            stats.rejected += 1
            stats.consecutive_rejections += 1
            reason = decision.reason[:50]
            stats.rejection_reasons[reason] = stats.rejection_reasons.get(reason, 0) + 1
        elif decision.action == "reduce":
            stats.reduced += 1
            stats.consecutive_rejections = 0
        elif decision.action == "delay":
            stats.delayed += 1
            stats.consecutive_rejections = 0

        # 更新1小时拒绝率
        self._update_rejection_rate_1h()

        # 埋点：决策动作与拒绝率外发到 MetricsPipeline
        self._record_metric("agent_decision_total", 1.0, {"action": decision.action})
        total = stats.total_audits
        self._record_metric(
            "agent_reject_rate",
            stats.rejected / total if total > 0 else 0.0,
        )
        self._record_metric("agent_rejection_rate_1h", stats.rejection_rate_1h)
        self._record_metric("agent_consecutive_rejections", float(stats.consecutive_rejections))
        if decision.action == "reject":
            self._record_metric("agent_reject_reason_total", 1.0, {"reason": decision.reason[:50]})

        # 审计追踪
        audit_entry = {
            "decision_id": decision.decision_id,
            "timestamp": decision.timestamp.isoformat(),
            "symbol": symbol,
            "strategy": strategy_name,
            "action": decision.action,
            "reason": decision.reason[:200],
            "confidence": confidence,
            "source": decision.source,
            "level": decision.level.value,
            "details": {k: str(v) if isinstance(v, datetime) else v for k, v in decision.details.items()},
        }
        self._audit_trail.append(audit_entry)

        # 决策历史
        self._decision_history.append(decision)
        if decision.action == "reject":
            self._rejected_decisions.append(decision)

        # 必要时持久化
        if stats.total_audits % 50 == 0:
            self._save_state()

    def _update_rejection_rate_1h(self):
        """更新1小时滑动拒绝率"""
        cutoff = datetime.now() - timedelta(hours=1)
        recent_actions = 0
        recent_rejections = 0
        for entry in self._audit_trail:
            try:
                ts = datetime.fromisoformat(entry["timestamp"])
                if ts >= cutoff:
                    recent_actions += 1
                    if entry["action"] == "reject":
                        recent_rejections += 1
            except (ValueError, KeyError):
                pass
        if recent_actions > 0:
            self._decision_stats.rejection_rate_1h = recent_rejections / recent_actions

    def get_audit_trail(self, limit: int = 50) -> List[Dict]:
        """获取审计追踪（最近N条）"""
        trail = list(self._audit_trail)
        return trail[-limit:]

    def get_decision_stats(self) -> Dict[str, Any]:
        """获取决策统计摘要"""
        stats = self._decision_stats
        total = stats.total_audits
        return {
            "total_audits": total,
            "approved": stats.approved,
            "rejected": stats.rejected,
            "reduced": stats.reduced,
            "delayed": stats.delayed,
            "approve_rate": stats.approved / total if total > 0 else 0.0,
            "reject_rate": stats.rejected / total if total > 0 else 0.0,
            "rejection_rate_1h": stats.rejection_rate_1h,
            "consecutive_rejections": stats.consecutive_rejections,
            "top_rejection_reasons": sorted(stats.rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:5],
            "last_audit_time": stats.last_audit_time.isoformat() if stats.last_audit_time else None,
            "last_approve_time": stats.last_approve_time.isoformat() if stats.last_approve_time else None,
        }

    # ========== P12: 健康监控与自愈 ==========

    def _maybe_health_check(self):
        """定期健康检查，自动检测并修复衰退状态"""
        now = datetime.now()
        if (now - self._last_health_check).total_seconds() < self._health_check_interval:
            return

        self._last_health_check = now
        previous_health = self._health

        # 检查1: 市场数据是否过期
        if self._last_market_update != datetime.min:
            stale_seconds = (now - self._last_market_update).total_seconds()
            if stale_seconds > self._stale_data_threshold:
                self._health = AgentHealth.STALE_DATA
                self.raise_alert(
                    "health_stale_data",
                    AlertSeverity.WARNING,
                    f"市场数据过期 {stale_seconds:.0f}s (阈值{self._stale_data_threshold}s)",
                    cooldown_seconds=600,
                )

        # 检查2: 拒绝率是否过高（累计 + 1小时滑动双口径）
        stats = self._decision_stats
        total = stats.total_audits
        cumulative_reject_rate = stats.rejected / total if total > 0 else 0.0
        over_rejecting = False
        if total >= self._min_audits_for_health_check:
            self._update_rejection_rate_1h()
            # 累计拒绝率检测长期停摆，弥补 1h 滑动在低活跃时段的盲区
            if cumulative_reject_rate > self._cumulative_reject_rate_threshold:
                over_rejecting = True
                alert_msg = (
                    f"累计拒绝率 {cumulative_reject_rate:.1%} ({stats.rejected}/{total}) "
                    f"超过阈值 {self._cumulative_reject_rate_threshold:.1%}，市场性停摆"
                )
            elif stats.rejection_rate_1h > self._max_rejection_rate:
                over_rejecting = True
                alert_msg = f"1小时拒绝率 {stats.rejection_rate_1h:.1%} 超过阈值 {self._max_rejection_rate:.1%}"
        if over_rejecting:
            self._health = AgentHealth.OVER_REJECTING
            self.raise_alert(
                "health_over_rejecting",
                AlertSeverity.CRITICAL,
                alert_msg,
                details={
                    "cumulative_reject_rate": cumulative_reject_rate,
                    "rejection_rate_1h": stats.rejection_rate_1h,
                },
                cooldown_seconds=600,
            )

        # 检查3: 连续拒绝检查
        if stats.consecutive_rejections > 50:
            self._health = AgentHealth.OVER_REJECTING
            self.raise_alert(
                "health_consecutive_rejections",
                AlertSeverity.WARNING,
                f"连续拒绝 {stats.consecutive_rejections} 次，可能阈值设置过严",
                cooldown_seconds=600,
            )

        # 检查4: 如果健康状态已恢复，标记为恢复中
        if previous_health != AgentHealth.HEALTHY and self._health == AgentHealth.HEALTHY:
            self._health = AgentHealth.RECOVERING
            logger.info(f"P12: Agent transitioning from {previous_health.value} to recovering")

        # 自动恢复
        if self._health != AgentHealth.HEALTHY and self._health != AgentHealth.RECOVERING:
            self._auto_reset()

        # 如果恢复中且满足条件，切回HEALTHY
        if self._health == AgentHealth.RECOVERING:
            self._update_rejection_rate_1h()
            market_ok = True
            if self._last_market_update != datetime.min:
                market_ok = (now - self._last_market_update).total_seconds() < self._stale_data_threshold
            if stats.rejection_rate_1h < 0.5 and market_ok:
                self._health = AgentHealth.HEALTHY
                logger.info("P12: Agent recovered to HEALTHY")

        # 健康状态变化时持久化
        if self._health != previous_health:
            self._save_state()

        # 埋点：健康状态与拒绝率治理外发到 MetricsPipeline（观测自愈/衰退迁移）
        self._record_metric("agent_health_state", 1.0, {"health": self._health.value})
        self._record_metric("agent_cumulative_reject_rate", cumulative_reject_rate)
        self._record_metric("agent_rejection_rate_1h", stats.rejection_rate_1h)

    def _auto_reset(self):
        """自动恢复机制

        当智能体处于异常状态时，尝试自动恢复：
        - STALE_DATA: 标记为等待数据恢复
        - OVER_REJECTING: 临时降低阈值，放宽审核
        - DEGRADED: 清除部分临时状态
        """
        now = datetime.now()
        cooldown = (now - self._last_reset_time).total_seconds()

        if cooldown < self._auto_reset_cooldown:
            return  # 冷却中

        self._last_reset_time = now

        # 埋点：自愈动作计数外发
        self._record_metric("agent_auto_reset_total", 1.0, {"health": self._health.value})

        if self._health == AgentHealth.OVER_REJECTING:
            # 降低自适应阈值以放宽审核
            old_threshold = self.min_confidence_threshold
            self.min_confidence_threshold = max(0.30, self.min_confidence_threshold - 0.05)
            self._record_metric("agent_min_confidence_threshold", self.min_confidence_threshold)
            logger.warning(
                f"P12: Auto-reset: lowered min_confidence_threshold "
                f"{old_threshold:.2f} -> {self.min_confidence_threshold:.2f}"
            )
            
            # P23: 同时降低学习参数中的阈值，打破过度拒绝的死循环
            for key in list(self._learned_params.keys()):
                if "_min_confidence_threshold" in key or "_min_signal_quality" in key:
                    old_val = self._learned_params[key]
                    if old_val > 0.40:
                        self._learned_params[key] = max(0.35, old_val - 0.05)
                        logger.warning(
                            f"P23: Auto-reset: lowered learned param {key} "
                            f"{old_val:.2f} -> {self._learned_params[key]:.2f}"
                        )
            
            # P24: 清除决策统计，重置拒绝计数器，避免持续被判定为over_rejecting
            self._decision_stats.consecutive_rejections = 0
            self._decision_stats.rejection_rate_1h = 0.0
            self._decision_stats.total_audits = 0
            self._decision_stats.approved = 0
            self._decision_stats.rejected = 0
            self._decision_stats.reduced = 0
            self._decision_stats.delayed = 0
            logger.info("P24: Auto-reset: cleared decision stats to prevent over_rejecting loop")
            
            self.raise_alert(
                "auto_reset_over_rejecting",
                AlertSeverity.WARNING,
                f"自动恢复: 降低置信度阈值至 {self.min_confidence_threshold:.2f}，已重置决策统计",
                cooldown_seconds=1800,
            )
            self._health = AgentHealth.HEALTHY

        elif self._health == AgentHealth.STALE_DATA:
            # 标记为恢复中，等待新数据
            self._health = AgentHealth.RECOVERING
            logger.info("P12: Auto-reset: waiting for fresh market data")

        elif self._health == AgentHealth.DEGRADED:
            # 清除过期数据，重置异常状态
            self._decision_stats.consecutive_rejections = 0
            self._health = AgentHealth.RECOVERING
            logger.info("P12: Auto-reset: cleared consecutive rejections, entering recovery")

        self._save_state()

    def get_adaptive_params_snapshot(self) -> Dict[str, Any]:
        """返回自适应智能体当前关键自适应参数快照（供仪表盘观测与参数回滚校验）。"""
        return {
            "min_confidence_threshold": self.min_confidence_threshold,
            "adaptive_confidence_cap": self._adaptive_conf_cap,
            "signal_quality_floors": dict(self._signal_quality_floors),
            "strategy_weights": dict(self._strategy_weights),
            "strategy_active": dict(self._strategy_active),
            "learned_params": dict(self._learned_params),
            "learning_epoch": self._learning_epoch,
            "hour_risk_multiplier": dict(self._hour_risk_multiplier),
            "high_risk_hours": sorted(self._high_risk_hours),
            "max_rejection_rate": self._max_rejection_rate,
            "cumulative_reject_rate_threshold": self._cumulative_reject_rate_threshold,
        }

    def get_health_status(self) -> Dict[str, Any]:
        """获取智能体健康状态完整快照"""
        now = datetime.now()
        stats = self._decision_stats

        # 市场数据新鲜度
        market_data_age = None
        if self._last_market_update != datetime.min:
            market_data_age = (now - self._last_market_update).total_seconds()

        self._update_rejection_rate_1h()

        return {
            "health": self._health.value,
            "is_healthy": self._health == AgentHealth.HEALTHY,
            "last_health_check": self._last_health_check.isoformat(),
            "market_data_age_seconds": market_data_age,
            "market_data_stale": market_data_age is not None and market_data_age > self._stale_data_threshold,
            "decision_stats": {
                "total_audits": stats.total_audits,
                "approve_rate": stats.approved / stats.total_audits if stats.total_audits > 0 else 0.0,
                "reject_rate": stats.rejected / stats.total_audits if stats.total_audits > 0 else 0.0,
                "rejection_rate_1h": stats.rejection_rate_1h,
                "consecutive_rejections": stats.consecutive_rejections,
                "reduce_rate": stats.reduced / stats.total_audits if stats.total_audits > 0 else 0.0,
                "delay_rate": stats.delayed / stats.total_audits if stats.total_audits > 0 else 0.0,
            },
            "blacklist_count": len([s for s in self._symbol_blacklist if self.is_symbol_blacklisted(s)]),
            "paused_strategies_count": len([s for s in self._strategy_pause if self.is_strategy_paused(s)]),
            "learning_epoch": self._learning_epoch,
            "learned_params_count": len(self._learned_params),
            "last_learning_time": self._last_learning_time.isoformat() if self._last_learning_time != datetime.min else None,
            "last_reset_time": self._last_reset_time.isoformat() if self._last_reset_time != datetime.min else None,
            "current_hour_risk": self._get_hour_risk_level(),
            "current_hour_multiplier": self.get_hour_risk_multiplier(),
            "audit_trail_size": len(self._audit_trail),
            "active_alerts": len(self._active_alerts),
            "health_check_interval": self._health_check_interval,
            "stale_data_threshold": self._stale_data_threshold,
            "max_rejection_rate": self._max_rejection_rate,
            "auto_reset_cooldown": self._auto_reset_cooldown,
            "metrics_pipeline_injected": self._metrics_pipeline is not None,
            "adaptive_params": self.get_adaptive_params_snapshot(),
        }