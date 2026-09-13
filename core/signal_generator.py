"""
信号生成器
===========
核心定位：开多/开空/加仓/减仓/止盈/止损/对冲信号分级输出

信号类型：
- OPEN_LONG: 开多
- OPEN_SHORT: 开空
- ADD_POSITION: 加仓
- REDUCE_POSITION: 减仓
- TAKE_PROFIT: 止盈
- STOP_LOSS: 止损
- HEDGE: 对冲
- CLOSE_ALL: 全部平仓

信号级别：
- STRONG (权重 > 0.8): 强信号，立即执行
- MEDIUM (权重 0.5-0.8): 中等信号，结合其他条件
- WEAK (权重 < 0.5): 弱信号，等待确认

自适应学习：
- 规则权重动态调整：基于历史胜率和盈亏比自动优化各规则权重
- 信号反馈闭环：每次交易结果反馈到规则评分，劣化规则自动降权
- 学习率衰减：逐步降低学习率避免过度拟合
"""

import asyncio
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import threading
import time
from loguru import logger

if TYPE_CHECKING:
    from core.signal_obfuscator import SignalObfuscator
    from core.anti_pattern_detector import AntiPatternDetector


class SignalType(Enum):
    """信号类型"""
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    ADD_POSITION = "add_position"
    REDUCE_POSITION = "reduce_position"
    TAKE_PROFIT = "take_profit"
    STOP_LOSS = "stop_loss"
    HEDGE = "hedge"
    CLOSE_ALL = "close_all"
    HOLD = "hold"


class SignalLevel(Enum):
    """信号级别"""
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


class SignalSource(Enum):
    """信号来源"""
    TREND_BREAKOUT = "trend_breakout"
    TREND_PULLBACK = "trend_pullback"
    GRID_REBOUND = "grid_rebound"
    VOLATILITY_ARB = "volatility_arb"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    VOLUME_SURGE = "volume_surge"
    RISK_CONTROL = "risk_control"
    MANUAL = "manual"


@dataclass
class TradingSignal:
    """交易信号"""
    symbol: str
    signal_type: SignalType
    source: SignalSource
    level: SignalLevel
    weight: float
    price: float
    quantity: float
    timestamp: datetime
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    strategy_id: str = ""
    position_id: str = ""
    expiry: datetime = None
    
    def is_expired(self) -> bool:
        if self.expiry is None:
            return False
        return datetime.now() > self.expiry
    
    def is_executable(self) -> bool:
        """判断信号是否可执行（强/中级别且未过期）"""
        return (self.level in [SignalLevel.STRONG, SignalLevel.MEDIUM] and
                not self.is_expired() and self.weight > 0.3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "source": self.source.value,
            "level": self.level.value,
            "weight": self.weight,
            "price": self.price,
            "quantity": self.quantity,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "metadata": self.metadata,
            "strategy_id": self.strategy_id,
            "position_id": self.position_id,
            "expiry": self.expiry.isoformat() if self.expiry else None
        }


@dataclass
class SignalContext:
    """信号上下文，包含市场状态和持仓信息"""
    symbol: str
    current_price: float
    position_side: Optional[str] = None
    position_size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    margin_used: float = 0.0
    leverage: float = 1.0
    
    indicators: Dict[str, float] = field(default_factory=dict)
    market_state: str = "neutral"
    volatility_level: str = "normal"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    recent_signals: List[TradingSignal] = field(default_factory=list)
    recent_trades: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RulePerformance:
    """规则表现追踪（用于自适应学习）"""
    rule_name: str
    total_signals: int = 0
    executed_signals: int = 0
    winning_signals: int = 0
    losing_signals: int = 0
    filtered_signals: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class SignalRule:
    """信号规则基类"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.weight = self.config.get("weight", 1.0)
    
    def evaluate(self, context: SignalContext) -> Optional[TradingSignal]:
        """
        评估规则，返回信号或None
        
        子类必须实现此方法
        """
        raise NotImplementedError
    
    def _create_signal(self, context: SignalContext, signal_type: SignalType,
                       source: SignalSource, weight: float, reason: str,
                       quantity: float = 0.0, **metadata) -> TradingSignal:
        """创建信号辅助方法"""
        
        level = SignalLevel.STRONG if weight >= 0.8 else (
            SignalLevel.MEDIUM if weight >= 0.5 else SignalLevel.WEAK
        )
        
        adjusted_weight = weight * self.weight
        
        return TradingSignal(
            symbol=context.symbol,
            signal_type=signal_type,
            source=source,
            level=level,
            weight=min(adjusted_weight, 1.0),
            price=context.current_price,
            quantity=quantity,
            timestamp=datetime.now(),
            reason=reason,
            metadata=metadata,
            strategy_id=self.name
        )


class TrendBreakoutRule(SignalRule):
    """趋势突破信号规则 — 增强版：动态ATR阈值"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("trend_breakout", config)
        self.breakout_threshold = self.config.get("breakout_threshold", 0.02)
        self.confirmation_periods = self.config.get("confirmation_periods", 2)
        self.volume_factor = self.config.get("volume_factor", 1.5)
        self.min_adx = self.config.get("min_adx", 25)
        self.dynamic_threshold = self.config.get("dynamic_threshold", True)
        self.atr_adjust_factor = self.config.get("atr_adjust_factor", 1.5)
    
    def _get_dynamic_adx_threshold(self, context: SignalContext) -> float:
        """根据波动率动态调整ADX阈值
        - 高波动：提高ADX阈值，需要更强趋势才开仓
        - 低波动：降低ADX阈值，更容易捕捉趋势启动
        """
        if not self.dynamic_threshold:
            return self.min_adx
        
        atr_pct = context.indicators.get("atr_percent", 0)
        if atr_pct <= 0:
            return self.min_adx
        
        # ATR占比 > 3% = 高波动 → ADX阈值提高到30
        if atr_pct > 0.03:
            return self.min_adx + 5
        # ATR占比 > 2% = 中等波动 → ADX阈值提高到28
        elif atr_pct > 0.02:
            return self.min_adx + 3
        # ATR占比 < 0.5% = 极低波动 → ADX阈值降低到20
        elif atr_pct < 0.005:
            return max(15, self.min_adx - 5)
        # ATR占比 < 1% = 低波动 → ADX阈值降低到22
        elif atr_pct < 0.01:
            return max(18, self.min_adx - 3)
        
        return self.min_adx
    
    def evaluate(self, context: SignalContext) -> Optional[TradingSignal]:
        if not self.enabled:
            return None
        
        indicators = context.indicators
        
        adx = indicators.get("adx", 0)
        plus_di = indicators.get("plus_di", 0)
        minus_di = indicators.get("minus_di", 0)
        rsi = indicators.get("rsi", 50)
        macd_hist = indicators.get("macd_histogram", 0)
        
        dynamic_adx = self._get_dynamic_adx_threshold(context)
        
        if adx < dynamic_adx:
            return None
        
        weight = 0.0
        reason = ""
        signal_type = SignalType.HOLD
        
        if plus_di > minus_di and adx > dynamic_adx:
            if macd_hist > 0 and rsi < 70:
                weight = 0.6
                reason = f"上升趋势确认 (ADX={adx:.1f}, +DI={plus_di:.1f}, 动态阈值={dynamic_adx})"
                signal_type = SignalType.OPEN_LONG
                
                if context.position_side == "long" and context.position_size > 0:
                    weight = 0.4
                    reason = "上升趋势延续，建议加仓"
                    signal_type = SignalType.ADD_POSITION
        
        elif minus_di > plus_di and adx > dynamic_adx:
            if macd_hist < 0 and rsi > 30:
                weight = 0.6
                reason = f"下降趋势确认 (ADX={adx:.1f}, -DI={minus_di:.1f}, 动态阈值={dynamic_adx})"
                signal_type = SignalType.OPEN_SHORT
                
                if context.position_side == "short" and context.position_size > 0:
                    weight = 0.4
                    reason = "下降趋势延续，建议加仓"
                    signal_type = SignalType.ADD_POSITION
        
        if weight > 0:
            return self._create_signal(
                context, signal_type, SignalSource.TREND_BREAKOUT,
                weight, reason,
                adx=adx, plus_di=plus_di, minus_di=minus_di,
                rsi=rsi, macd_hist=macd_hist, dynamic_adx=dynamic_adx
            )
        
        return None


class MeanReversionRule(SignalRule):
    """均值回归信号规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("mean_reversion", config)
        self.rsi_oversold = self.config.get("rsi_oversold", 30)
        self.rsi_overbought = self.config.get("rsi_overbought", 70)
        self.boll_upper_threshold = self.config.get("boll_upper_threshold", 0.95)
        self.boll_lower_threshold = self.config.get("boll_lower_threshold", 0.05)
    
    def evaluate(self, context: SignalContext) -> Optional[TradingSignal]:
        if not self.enabled:
            return None
        
        indicators = context.indicators
        
        rsi = indicators.get("rsi", 50)
        boll_pos = indicators.get("bollinger_position", 0.5)
        kdj_k = indicators.get("kdj_k", 50)
        kdj_j = indicators.get("kdj_j", 50)
        
        weight = 0.0
        reason = ""
        signal_type = SignalType.HOLD
        
        if rsi < self.rsi_oversold and boll_pos < self.boll_lower_threshold:
            weight = 0.5
            
            if kdj_j < 20:
                weight = 0.7
            
            reason = f"超卖回归信号 (RSI={rsi:.1f}, BOLL位置={boll_pos:.2f})"
            signal_type = SignalType.OPEN_LONG
            
            if context.position_side == "short":
                weight = 0.6
                reason = "超卖区域，建议平空"
                signal_type = SignalType.STOP_LOSS
        
        elif rsi > self.rsi_overbought and boll_pos > self.boll_upper_threshold:
            weight = 0.5
            
            if kdj_j > 80:
                weight = 0.7
            
            reason = f"超买回归信号 (RSI={rsi:.1f}, BOLL位置={boll_pos:.2f})"
            signal_type = SignalType.OPEN_SHORT
            
            if context.position_side == "long":
                weight = 0.6
                reason = "超买区域，建议平多"
                signal_type = SignalType.TAKE_PROFIT
        
        if weight > 0:
            return self._create_signal(
                context, signal_type, SignalSource.MEAN_REVERSION,
                weight, reason,
                rsi=rsi, boll_pos=boll_pos, kdj_k=kdj_k, kdj_j=kdj_j
            )
        
        return None


class VolatilityArbitrageRule(SignalRule):
    """波动率套利信号规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("volatility_arb", config)
        self.vol_high_threshold = self.config.get("vol_high_threshold", 0.8)
        self.vol_low_threshold = self.config.get("vol_low_threshold", 0.2)
        self.atr_multiplier = self.config.get("atr_multiplier", 2.0)
    
    def evaluate(self, context: SignalContext) -> Optional[TradingSignal]:
        if not self.enabled:
            return None
        
        indicators = context.indicators
        
        vol_pct = indicators.get("volatility_percentile", 0.5)
        atr_pct = indicators.get("atr_percent", 0)
        boll_width = indicators.get("bollinger_width", 0)
        
        weight = 0.0
        reason = ""
        signal_type = SignalType.HOLD
        
        if vol_pct > self.vol_high_threshold:
            weight = 0.5
            reason = f"高波动率环境 (波动率分位数={vol_pct:.2f})"
            signal_type = SignalType.HEDGE
            
            metadata = {
                "vol_pct": vol_pct,
                "atr_pct": atr_pct,
                "boll_width": boll_width,
                "suggested_hedge_ratio": 0.5
            }
            
            return self._create_signal(
                context, signal_type, SignalSource.VOLATILITY_ARB,
                weight, reason, **metadata
            )
        
        elif vol_pct < self.vol_low_threshold:
            weight = 0.3
            reason = f"低波动率环境，适合建仓 (波动率分位数={vol_pct:.2f})"
            
            if context.position_side is None:
                signal_type = SignalType.OPEN_LONG
            else:
                signal_type = SignalType.ADD_POSITION
            
            return self._create_signal(
                context, signal_type, SignalSource.VOLATILITY_ARB,
                weight, reason,
                vol_pct=vol_pct, atr_pct=atr_pct
            )
        
        return None


class GridReboundRule(SignalRule):
    """网格反弹信号规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("grid_rebound", config)
        self.grid_spacing = self.config.get("grid_spacing", 0.01)
        self.rebound_threshold = self.config.get("rebound_threshold", 0.005)
    
    def evaluate(self, context: SignalContext) -> Optional[TradingSignal]:
        if not self.enabled:
            return None
        
        indicators = context.indicators
        
        if context.position_side is None:
            return None
        
        pnl_pct = context.unrealized_pnl_pct
        
        weight = 0.0
        reason = ""
        signal_type = SignalType.HOLD
        
        if context.position_side == "long":
            if pnl_pct < -self.grid_spacing:
                weight = 0.4
                reason = f"多单亏损{pnl_pct*100:.2f}%，建议网格加仓"
                signal_type = SignalType.ADD_POSITION
            elif pnl_pct > self.rebound_threshold:
                weight = 0.5
                reason = f"多单盈利{pnl_pct*100:.2f}%，建议网格平仓"
                signal_type = SignalType.REDUCE_POSITION
        
        elif context.position_side == "short":
            if pnl_pct < -self.grid_spacing:
                weight = 0.4
                reason = f"空单亏损{pnl_pct*100:.2f}%，建议网格加仓"
                signal_type = SignalType.ADD_POSITION
            elif pnl_pct > self.rebound_threshold:
                weight = 0.5
                reason = f"空单盈利{pnl_pct*100:.2f}%，建议网格平仓"
                signal_type = SignalType.REDUCE_POSITION
        
        if weight > 0:
            return self._create_signal(
                context, signal_type, SignalSource.GRID_REBOUND,
                weight, reason,
                pnl_pct=pnl_pct, grid_spacing=self.grid_spacing
            )
        
        return None


class RiskControlRule(SignalRule):
    """风控信号规则"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("risk_control", config)
        self.max_loss_pct = self.config.get("max_loss_pct", 0.03)
        self.max_profit_pct = self.config.get("max_profit_pct", 0.1)
        self.max_hold_hours = self.config.get("max_hold_hours", 24)
    
    def evaluate(self, context: SignalContext) -> Optional[TradingSignal]:
        if not self.enabled:
            return None
        
        if context.position_side is None or context.position_size == 0:
            return None
        
        pnl_pct = context.unrealized_pnl_pct
        weight = 0.0
        reason = ""
        signal_type = SignalType.HOLD
        
        if pnl_pct <= -self.max_loss_pct:
            weight = 1.0
            reason = f"触发止损 (亏损{pnl_pct*100:.2f}%)"
            signal_type = SignalType.STOP_LOSS
        
        elif pnl_pct >= self.max_profit_pct:
            weight = 0.8
            reason = f"触发止盈 (盈利{pnl_pct*100:.2f}%)"
            signal_type = SignalType.TAKE_PROFIT
        
        if weight > 0:
            return self._create_signal(
                context, signal_type, SignalSource.RISK_CONTROL,
                weight, reason,
                pnl_pct=pnl_pct
            )
        
        return None


class DivergenceRule(SignalRule):
    """背离检测信号规则 — 识别RSI/MACD背离，预警趋势反转"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("divergence_detection", config)
        self.lookback_bars = self.config.get("lookback_bars", 20)
        self.min_divergence_strength = self.config.get("min_divergence_strength", 0.6)
    
    def evaluate(self, context: SignalContext) -> Optional[TradingSignal]:
        if not self.enabled:
            return None
        
        metadata = context.metadata if hasattr(context, 'metadata') else {}
        
        rsi_div = metadata.get("rsi_divergence", "")
        macd_div = metadata.get("macd_divergence", "")
        indicators = context.indicators
        
        weight = 0.0
        reason = ""
        signal_type = SignalType.HOLD
        
        # 综合判断背离方向
        divergence_direction = ""  # "bearish" or "bullish"
        div_strength = 0.0
        
        if rsi_div == "bearish":
            divergence_direction = "bearish"
            div_strength += 0.6
        elif rsi_div == "bullish":
            divergence_direction = "bullish"
            div_strength += 0.6
        
        if macd_div == "bearish":
            if divergence_direction == "bullish":
                return None  # RSI和MACD背离方向冲突，忽略
            divergence_direction = "bearish"
            div_strength += 0.8
        elif macd_div == "bullish":
            if divergence_direction == "bearish":
                return None  # 方向冲突
            divergence_direction = "bullish"
            div_strength += 0.8
        
        if not divergence_direction:
            # 从indicators做简化检测
            rsi = indicators.get("rsi", 50)
            adx = indicators.get("adx", 20)
            
            # 低价位+RSI超卖+ADX确认 = 潜在底背离
            if rsi < 35 and adx > 20:
                weight = 0.35
                reason = f"潜在底背离信号 (RSI={rsi:.1f}, ADX={adx:.1f})"
                signal_type = SignalType.OPEN_LONG
                
                if context.position_side == "short":
                    reason = "潜在底背离，建议平空"
                    signal_type = SignalType.STOP_LOSS
                    weight = 0.5
            
            # 高价位+RSI超买+ADX确认 = 潜在顶背离
            elif rsi > 65 and adx > 20:
                weight = 0.35
                reason = f"潜在顶背离信号 (RSI={rsi:.1f}, ADX={adx:.1f})"
                signal_type = SignalType.OPEN_SHORT
                
                if context.position_side == "long":
                    reason = "潜在顶背离，建议平多"
                    signal_type = SignalType.TAKE_PROFIT
                    weight = 0.5
        elif divergence_direction == "bullish":
            # 底背离 → 做多/平空信号
            weight = 0.55 + div_strength * 0.1
            reason = f"底背离确认 (RSI={rsi_div}, MACD={macd_div})"
            signal_type = SignalType.OPEN_LONG
            
            if context.position_side == "short":
                reason = "底背离信号，建议平空"
                signal_type = SignalType.STOP_LOSS
                weight = 0.6
        elif divergence_direction == "bearish":
            # 顶背离 → 做空/平多信号
            weight = 0.55 + div_strength * 0.1
            reason = f"顶背离确认 (RSI={rsi_div}, MACD={macd_div})"
            signal_type = SignalType.OPEN_SHORT
            
            if context.position_side == "long":
                reason = "顶背离信号，建议平多"
                signal_type = SignalType.TAKE_PROFIT
                weight = 0.6
        
        if weight > 0:
            return self._create_signal(
                context, signal_type, SignalSource.MEAN_REVERSION,
                weight, reason,
                rsi_divergence=rsi_div, macd_divergence=macd_div
            )
        
        return None


class MarketStructureRule(SignalRule):
    """市场结构信号规则 — 基于HH/HL/LH/LL的摆动点趋势识别"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("market_structure", config)
        self.swing_window = self.config.get("swing_window", 5)
        self.min_swing_points = self.config.get("min_swing_points", 3)
    
    def evaluate(self, context: SignalContext) -> Optional[TradingSignal]:
        if not self.enabled:
            return None
        
        indicators = context.indicators
        
        adx = indicators.get("adx", 20)
        ma20 = indicators.get("ma20", 0)
        ma50 = indicators.get("ma50", 0)
        recent_close = indicators.get("recent_close", context.current_price)
        
        weight = 0.0
        reason = ""
        signal_type = SignalType.HOLD
        
        # 简化版市场结构判断：基于均线排列和ADX
        if ma50 > 0 and ma20 > 0:
            ma_trend_strong = abs(ma20 - ma50) / ma50 > 0.03
            
            if ma20 > ma50 and recent_close > ma20 and adx > 25:
                if ma_trend_strong:
                    weight = 0.65
                    reason = f"强势上升结构(HH/HL)，均线多头排列"
                else:
                    weight = 0.5
                    reason = f"上升结构，均线多头排列"
                signal_type = SignalType.OPEN_LONG
                
                if context.position_side == "long" and context.position_size > 0:
                    weight = 0.45
                    reason = "上升结构延续，建议加仓"
                    signal_type = SignalType.ADD_POSITION
            
            elif ma20 < ma50 and recent_close < ma20 and adx > 25:
                if ma_trend_strong:
                    weight = 0.65
                    reason = f"强势下降结构(LH/LL)，均线空头排列"
                else:
                    weight = 0.5
                    reason = f"下降结构，均线空头排列"
                signal_type = SignalType.OPEN_SHORT
                
                if context.position_side == "short" and context.position_size > 0:
                    weight = 0.45
                    reason = "下降结构延续，建议加仓"
                    signal_type = SignalType.ADD_POSITION
        
        if weight > 0:
            return self._create_signal(
                context, signal_type, SignalSource.TREND_BREAKOUT,
                weight, reason,
                adx=adx, ma20=ma20, ma50=ma50
            )
        
        return None


class SignalGenerator:
    """
    生产级信号生成器
    
    整合多个信号规则，通过完整流水线输出综合信号：
    1. 规则评估 → 2. 频率限流 → 3. 冲突解决 → 4. 多框架确认 → 5. 市场过滤 → 6. 质量评分 → 7. 聚合 → 8. 分发
    
    自适应学习：
    - 记录每个规则的信号质量和实际交易结果
    - 周期性优化规则权重（胜率高→加权重，胜率低→降权重）
    - 支持手动启用/禁用学习，学习率衰减
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.rules: List[SignalRule] = []
        self._setup_default_rules()
        
        self._signal_history: Dict[str, deque] = {}
        self._history_size = self.config.get("signal_history_size", 100)
        
        self._signal_callbacks: List[Callable[[TradingSignal], None]] = []
        self._lock = threading.RLock()
        
        self._signal_count = 0
        self._signal_by_type: Dict[str, int] = {}
        self._signal_by_symbol: Dict[str, int] = {}
        
        # 自适应学习
        learning_config = self.config.get("adaptive_learning", {})
        self._adaptive_enabled = learning_config.get("enabled", False)
        self._learning_rate = learning_config.get("learning_rate", 0.1)
        self._learning_interval = learning_config.get("interval_seconds", 3600)
        self._min_samples_for_learning = learning_config.get("min_samples", 10)
        self._max_weight_change = learning_config.get("max_weight_change", 0.3)
        self._decay_factor = learning_config.get("decay_factor", 0.95)
        self._min_rule_weight = learning_config.get("min_rule_weight", 0.2)
        self._max_rule_weight = learning_config.get("max_rule_weight", 2.0)
        
        # 规则表现追踪
        self._rule_performance: Dict[str, 'RulePerformance'] = {}
        for rule in self.rules:
            self._rule_performance[rule.name] = RulePerformance(rule_name=rule.name)
        
        self._pending_signals: Dict[str, TradingSignal] = {}  # signal_id -> signal (待验证)
        self._learning_task: Optional[asyncio.Task] = None
        self._learning_generation = 0  # 学习迭代代数
        
        # ─── 生产级增强：信号频率限流 ───
        self._signal_rate_tracker: Dict[str, List[float]] = {}  # key -> [timestamps]
        self._max_signals_per_second = self.config.get("max_signals_per_second", 10)
        self._max_signals_per_symbol_second = self.config.get("max_signals_per_symbol_second", 3)
        self._signal_cooldown_seconds = self.config.get("signal_cooldown_seconds", 5)
        self._last_signal_time: Dict[str, float] = {}  # f"{symbol}:{direction}" -> timestamp
        
        # ─── 生产级增强：冲突解决 ───
        self._conflict_resolution = self.config.get("conflict_resolution", "majority_vote")
        self._max_conflicting_signals = self.config.get("max_conflicting_signals", 3)
        self._opposite_signal_delay = self.config.get("opposite_signal_delay_seconds", 30)
        self._last_opposite_signal_time: Dict[str, float] = {}  # symbol -> last opposite timestamp
        
        # ─── 生产级增强：多框架确认 ───
        self._confirmation_required = self.config.get("confirmation_required", True)
        self._min_confirmation_rules = self.config.get("min_confirmation_rules", 2)
        self._confirmation_window = self.config.get("confirmation_window_seconds", 10)
        self._pending_confirmations: Dict[str, Dict[str, Any]] = {}  # symbol -> pending
        
        # ─── 生产级增强：市场状态过滤 ───
        self._market_regime_filter = self.config.get("market_regime_filter", True)
        # 市场状态提供者（callable 或 dict），外部注入
        self._regime_provider = None
        self._regime_info_cache = None
        self._regime_blocked_types: Dict[str, List[str]] = {
            "extreme_volatility": ["OPEN_LONG", "OPEN_SHORT", "ADD_POSITION"],
            "liquidity_crisis": ["OPEN_LONG", "OPEN_SHORT"],
            "funding_crush": ["OPEN_LONG"],
        }
        
        # ─── 生产级增强：信号聚合 ───
        self._signal_aggregation = self.config.get("signal_aggregation", True)
        self._aggregation_window = self.config.get("aggregation_window_seconds", 5)
        self._max_aggregated = self.config.get("max_aggregated_signals", 5)
        self._aggregation_buffers: Dict[str, List[TradingSignal]] = {}  # symbol -> pending signals
        
        # ─── 生产级增强：信号质量过滤 ───
        self._min_signal_weight = self.config.get("min_signal_weight", 0.35)
        self._min_quality_score = self.config.get("min_quality_score", 0.35)
        self._quality_engine = None  # 外部注入
        
        # ─── 生产级增强：信号上下文增强 ───
        self._include_market_context = self.config.get("include_market_context", True)
        self._include_risk_metrics = self.config.get("include_risk_metrics", True)
        self._include_position_context = self.config.get("include_position_context", True)
        
        # ─── 生产级增强：统计 ───
        self._signals_filtered_by_rate = 0
        self._signals_filtered_by_conflict = 0
        self._signals_filtered_by_regime = 0
        self._signals_filtered_by_quality = 0
        self._aggregated_signals_count = 0
        
        # ─── 防针对量化数据：三层防护 ───
        self._signal_obfuscator: Optional["SignalObfuscator"] = None
        self._anti_pattern_detector: Optional["AntiPatternDetector"] = None
        self._anti_targeting_enabled = self.config.get("anti_targeting", {}).get("enabled", True)
        self._obfuscation_enabled = self.config.get("anti_targeting", {}).get("signal_obfuscator", {}).get("enabled", True)
        self._pattern_detection_enabled = self.config.get("anti_targeting", {}).get("anti_pattern_detector", {}).get("enabled", True)
        self._signals_obfuscated_count = 0
        self._dummy_signals_injected_count = 0
        self._pattern_risk_score: float = 0.0
        self._last_risk_check_time: float = 0.0
        
        self._init_anti_targeting_modules()
        
    def set_quality_engine(self, engine) -> None:
        """注入外部信号质量引擎"""
        self._quality_engine = engine
    
    def set_regime_info_provider(self, provider) -> None:
        """注入市场状态信息提供者（可为 callable 或 dict）"""
        self._regime_provider = provider
    
    def set_regime_info_cache(self, regime_info: Optional[Dict[str, Any]]) -> None:
        """设置市场状态缓存（静态快照）"""
        self._regime_info_cache = regime_info
    
    # ═══════════════════════════════════════════════════════════════
    # 防针对量化数据：三层防护集成
    # ═══════════════════════════════════════════════════════════════
    
    def _init_anti_targeting_modules(self) -> None:
        """初始化防针对量化数据三层防护模块"""
        if not self._anti_targeting_enabled:
            logger.info("Anti-targeting modules disabled globally")
            return
        
        try:
            from core.signal_obfuscator import SignalObfuscator
            self._signal_obfuscator = SignalObfuscator(self.config)
            logger.info("SignalObfuscator initialized (Layer 1)")
        except Exception as e:
            logger.warning(f"Failed to init SignalObfuscator: {e}")
            self._obfuscation_enabled = False
        
        try:
            from core.anti_pattern_detector import AntiPatternDetector
            self._anti_pattern_detector = AntiPatternDetector(self.config)
            # 注册告警回调：高风险时自动升级混淆
            self._anti_pattern_detector.register_alert_callback(self._on_pattern_alert)
            logger.info("AntiPatternDetector initialized (Layer 3)")
        except Exception as e:
            logger.warning(f"Failed to init AntiPatternDetector: {e}")
            self._pattern_detection_enabled = False
        
        # 建立风险反馈闭环：AntiPatternDetector → SignalObfuscator
        if self._signal_obfuscator and self._anti_pattern_detector:
            logger.info("Anti-targeting feedback loop: Detector → Obfuscator (active)")
    
    def _on_pattern_alert(self, alert) -> None:
        """模式告警回调 — 自动升级混淆强度"""
        from core.signal_obfuscator import ObfuscationMode
        from core.anti_pattern_detector import RiskLevel
        
        if not self._signal_obfuscator:
            return
        
        risk_level = alert.severity  # "warning", "danger", "critical"
        
        if risk_level == "critical":
            self._signal_obfuscator.set_mode(ObfuscationMode.AGGRESSIVE)
            logger.warning(f"ANTI-TARGETING: Upgraded obfuscation to AGGRESSIVE due to {alert.alert_type}")
        elif risk_level == "danger":
            if self._signal_obfuscator._mode != ObfuscationMode.AGGRESSIVE:
                self._signal_obfuscator.set_mode(ObfuscationMode.AGGRESSIVE)
        elif risk_level == "warning":
            current_mode = self._signal_obfuscator._mode
            if current_mode == ObfuscationMode.LIGHT:
                self._signal_obfuscator.set_mode(ObfuscationMode.STANDARD)
    
    def _run_anti_targeting_pipeline(self, signals: List[TradingSignal], 
                                      context: Optional[SignalContext] = None) -> List[TradingSignal]:
        """
        防针对量化数据流水线
        
        流程：
        1. 记录信号到 AntiPatternDetector 用于模式分析
        2. 评估模式风险评分
        3. 风险评分反馈到 SignalObfuscator
        4. 信号混淆处理
        5. 过滤虚拟信号（在执行层）
        """
        if not self._anti_targeting_enabled:
            return signals
        
        if not signals:
            return signals
        
        # Step 1: 记录信号到反模式检测器
        if self._pattern_detection_enabled and self._anti_pattern_detector:
            for signal in signals:
                try:
                    direction = "long" if signal.signal_type in (SignalType.OPEN_LONG, SignalType.ADD_POSITION) else \
                               "short" if signal.signal_type in (SignalType.OPEN_SHORT,) else "close"
                    self._anti_pattern_detector.record_signal(
                        symbol=signal.symbol,
                        signal_type=signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                        source=signal.source.value if hasattr(signal.source, 'value') else str(signal.source),
                        weight=signal.weight,
                        direction=direction,
                    )
                except Exception as e:
                    logger.debug(f"AntiPatternDetector record_signal error: {e}")
        
        # Step 2: 定期评估模式风险
        now = time.time()
        if self._pattern_detection_enabled and self._anti_pattern_detector and \
           now - self._last_risk_check_time > 60:
            try:
                self._pattern_risk_score = self._anti_pattern_detector.evaluate_risk()
                self._last_risk_check_time = now
                
                # 反馈风险评分到混淆器
                if self._signal_obfuscator:
                    self._signal_obfuscator.set_pattern_risk(self._pattern_risk_score)
            except Exception as e:
                logger.debug(f"AntiPatternDetector evaluate_risk error: {e}")
        
        # Step 3: 信号混淆
        if self._obfuscation_enabled and self._signal_obfuscator:
            try:
                context_dict = self._build_obfuscation_context(context) if context else {}
                obfuscated = self._signal_obfuscator.obfuscate(signals, context_dict)
                
                # 统计虚拟信号
                dummy_count = sum(1 for s in obfuscated if s.metadata.get("is_dummy", False))
                self._signals_obfuscated_count += len(signals)
                self._dummy_signals_injected_count += dummy_count
                
                return obfuscated
            except Exception as e:
                logger.error(f"SignalObfuscator error: {e}")
        
        return signals
    
    def _build_obfuscation_context(self, context: Optional[SignalContext]) -> Dict[str, Any]:
        """构建混淆器上下文"""
        if context is None:
            return {}
        
        return {
            "symbol": context.symbol,
            "current_price": context.current_price,
            "market_state": context.market_state,
            "volatility_level": context.volatility_level,
            "position_side": context.position_side,
            "position_size": context.position_size,
            "unrealized_pnl_pct": context.unrealized_pnl_pct,
            "indicators": dict(context.indicators) if context.indicators else {},
        }
    
    def record_order_to_detector(self, symbol: str, side: str, quantity: float, 
                                   price: float, is_split: bool = False) -> None:
        """记录订单到反模式检测器（由执行层调用）"""
        if self._pattern_detection_enabled and self._anti_pattern_detector:
            try:
                self._anti_pattern_detector.record_order(
                    symbol=symbol, side=side, quantity=quantity,
                    price=price, is_split=is_split,
                )
            except Exception as e:
                logger.debug(f"AntiPatternDetector record_order error: {e}")
    
    def record_slippage_to_detector(self, symbol: str, expected_price: float,
                                      actual_price: float, side: str) -> None:
        """记录滑点到反模式检测器（由执行层调用）"""
        if self._pattern_detection_enabled and self._anti_pattern_detector:
            try:
                self._anti_pattern_detector.record_slippage(
                    symbol=symbol, expected_price=expected_price,
                    actual_price=actual_price, side=side,
                )
            except Exception as e:
                logger.debug(f"AntiPatternDetector record_slippage error: {e}")
    
    def get_anti_targeting_stats(self) -> Dict[str, Any]:
        """获取防针对量化数据统计"""
        stats = {
            "enabled": self._anti_targeting_enabled,
            "obfuscation_enabled": self._obfuscation_enabled,
            "pattern_detection_enabled": self._pattern_detection_enabled,
            "signals_obfuscated": self._signals_obfuscated_count,
            "dummy_signals_injected": self._dummy_signals_injected_count,
            "pattern_risk_score": round(self._pattern_risk_score, 4),
        }
        
        if self._signal_obfuscator:
            stats["obfuscator"] = self._signal_obfuscator.get_stats()
        
        if self._anti_pattern_detector:
            stats["pattern_detector"] = self._anti_pattern_detector.get_stats()
        
        return stats
    
    def _should_dispatch(self, signal: TradingSignal) -> bool:
        """综合判断信号是否应分发（含防针对过滤）"""
        # 虚拟信号不执行
        if signal.metadata.get("is_dummy", False):
            return False
        
        if signal.weight < self._min_signal_weight:
            self._signals_filtered_by_quality += 1
            return False
        
        # 质量评分过滤
        quality_score = signal.metadata.get("quality_score", 0)
        if quality_score < self._min_quality_score:
            self._signals_filtered_by_quality += 1
            return False
        
        if not signal.is_executable():
            return False
        
        return True
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """生产级热参数更新"""
        with self._lock:
            self.config = {**self.config, **new_config}
            
            # 更新频率限流参数
            self._max_signals_per_second = self.config.get("max_signals_per_second", 10)
            self._max_signals_per_symbol_second = self.config.get("max_signals_per_symbol_second", 3)
            self._signal_cooldown_seconds = self.config.get("signal_cooldown_seconds", 5)
            
            # 更新冲突解决参数
            self._conflict_resolution = self.config.get("conflict_resolution", "majority_vote")
            self._max_conflicting_signals = self.config.get("max_conflicting_signals", 3)
            self._opposite_signal_delay = self.config.get("opposite_signal_delay_seconds", 30)
            
            # 更新确认参数
            self._confirmation_required = self.config.get("confirmation_required", True)
            self._min_confirmation_rules = self.config.get("min_confirmation_rules", 2)
            
            # 更新过滤参数
            self._market_regime_filter = self.config.get("market_regime_filter", True)
            self._min_signal_weight = self.config.get("min_signal_weight", 0.35)
            self._min_quality_score = self.config.get("min_quality_score", 0.35)
            
            # 更新聚合参数
            self._signal_aggregation = self.config.get("signal_aggregation", True)
            self._aggregation_window = self.config.get("aggregation_window_seconds", 5)
            self._max_aggregated = self.config.get("max_aggregated_signals", 5)
            
            # 更新上下文增强参数
            self._include_market_context = self.config.get("include_market_context", True)
            self._include_risk_metrics = self.config.get("include_risk_metrics", True)
            self._include_position_context = self.config.get("include_position_context", True)
            
            # 更新自适应学习参数
            lc = self.config.get("adaptive_learning", {})
            self._learning_rate = lc.get("learning_rate", self._learning_rate)
            self._decay_factor = lc.get("decay_factor", self._decay_factor)
            self._min_rule_weight = lc.get("min_rule_weight", self._min_rule_weight)
            self._max_rule_weight = lc.get("max_rule_weight", self._max_rule_weight)
            
            # 更新规则配置
            rule_configs = self.config.get("rules", {})
            for rule in self.rules:
                if rule.name in rule_configs:
                    rc = rule_configs[rule.name]
                    rule.enabled = rc.get("enabled", rule.enabled)
                    rule.weight = rc.get("weight", rule.weight)
                    rule.config = {**rule.config, **rc}
            
            # 更新历史大小
            new_size = self.config.get("signal_history_size", self._history_size)
            if new_size != self._history_size:
                self._history_size = new_size
                for symbol in self._signal_history:
                    self._signal_history[symbol] = deque(
                        list(self._signal_history[symbol])[-new_size:], maxlen=new_size
                    )
            
            # 更新防针对量化数据配置
            at_cfg = self.config.get("anti_targeting", {})
            self._anti_targeting_enabled = at_cfg.get("enabled", self._anti_targeting_enabled)
            
            if self._signal_obfuscator:
                try:
                    self._signal_obfuscator.update_config(self.config)
                    self._obfuscation_enabled = at_cfg.get("signal_obfuscator", {}).get("enabled", self._obfuscation_enabled)
                except Exception as e:
                    logger.debug(f"SignalObfuscator hot-update error: {e}")
            
            if self._anti_pattern_detector:
                try:
                    self._anti_pattern_detector.update_config(self.config)
                    self._pattern_detection_enabled = at_cfg.get("anti_pattern_detector", {}).get("enabled", self._pattern_detection_enabled)
                except Exception as e:
                    logger.debug(f"AntiPatternDetector hot-update error: {e}")
            
            logger.info(f"SignalGenerator config updated: {len(new_config)} params")
    
    # ═══════════════════════════════════════════════════════════════
    # 生产级信号流水线
    # ═══════════════════════════════════════════════════════════════
    
    def _check_rate_limit(self, symbol: str, direction: str) -> bool:
        """信号频率限流检查 — 防止信号洪水"""
        now = time.time()
        key = f"{symbol}:{direction}"
        
        # 同币种同方向冷却
        last_time = self._last_signal_time.get(key, 0)
        if now - last_time < self._signal_cooldown_seconds:
            self._signals_filtered_by_rate += 1
            return False
        
        # 全局速率限制
        global_key = "__global__"
        if global_key not in self._signal_rate_tracker:
            self._signal_rate_tracker[global_key] = []
        self._signal_rate_tracker[global_key] = [
            t for t in self._signal_rate_tracker[global_key] if now - t < 1.0
        ]
        if len(self._signal_rate_tracker[global_key]) >= self._max_signals_per_second:
            self._signals_filtered_by_rate += 1
            return False
        
        # 每币种速率限制
        if symbol not in self._signal_rate_tracker:
            self._signal_rate_tracker[symbol] = []
        self._signal_rate_tracker[symbol] = [
            t for t in self._signal_rate_tracker[symbol] if now - t < 1.0
        ]
        if len(self._signal_rate_tracker[symbol]) >= self._max_signals_per_symbol_second:
            self._signals_filtered_by_rate += 1
            return False
        
        # 记录本次信号
        self._signal_rate_tracker[global_key].append(now)
        self._signal_rate_tracker[symbol].append(now)
        self._last_signal_time[key] = now
        return True
    
    def _resolve_signal_conflicts(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """解决信号冲突 — 多规则产生相反信号时的智能仲裁"""
        if len(signals) <= 1:
            return signals
        
        # 按方向分组
        long_signals = [s for s in signals if s.signal_type == SignalType.OPEN_LONG]
        short_signals = [s for s in signals if s.signal_type == SignalType.OPEN_SHORT]
        close_signals = [s for s in signals if s.signal_type in (
            SignalType.TAKE_PROFIT, SignalType.STOP_LOSS, SignalType.CLOSE_ALL
        )]
        neutral_signals = [s for s in signals if s not in long_signals 
                          and s not in short_signals and s not in close_signals]
        
        # 平仓信号始终保留（安全优先）
        resolved = list(close_signals)
        
        if long_signals and short_signals:
            # 方向冲突
            if len(long_signals) + len(short_signals) > self._max_conflicting_signals:
                logger.warning(f"Too many conflicting signals ({len(long_signals)}L vs "
                             f"{len(short_signals)}S), dropping all open signals")
                self._signals_filtered_by_conflict += len(long_signals) + len(short_signals)
            else:
                if self._conflict_resolution == "strongest_weight":
                    # 最强权重策略
                    best_long = max(long_signals, key=lambda s: s.weight)
                    best_short = max(short_signals, key=lambda s: s.weight)
                    if best_long.weight > best_short.weight * 1.2:
                        resolved.append(best_long)
                        self._signals_filtered_by_conflict += len(short_signals)
                    elif best_short.weight > best_long.weight * 1.2:
                        resolved.append(best_short)
                        self._signals_filtered_by_conflict += len(long_signals)
                    else:
                        self._signals_filtered_by_conflict += len(long_signals) + len(short_signals)
                elif self._conflict_resolution == "risk_averse":
                    # 风险规避策略：都不开仓
                    self._signals_filtered_by_conflict += len(long_signals) + len(short_signals)
                else:
                    # majority_vote：多数派胜出
                    long_weight = sum(s.weight for s in long_signals)
                    short_weight = sum(s.weight for s in short_signals)
                    if long_weight > short_weight:
                        resolved.extend(long_signals)
                        self._signals_filtered_by_conflict += len(short_signals)
                    elif short_weight > long_weight:
                        resolved.extend(short_signals)
                        self._signals_filtered_by_conflict += len(long_signals)
                    else:
                        self._signals_filtered_by_conflict += len(long_signals) + len(short_signals)
        else:
            resolved.extend(long_signals)
            resolved.extend(short_signals)
        
        resolved.extend(neutral_signals)
        return resolved
    
    def _check_opposite_signal_delay(self, symbol: str, direction: str) -> bool:
        """检查反向信号延迟 — 防止频繁翻转"""
        if not self._opposite_signal_delay:
            return True
        
        opposite = "short" if direction == "long" else "long"
        last_opposite = self._last_opposite_signal_time.get(f"{symbol}:{opposite}", 0)
        
        if time.time() - last_opposite < self._opposite_signal_delay:
            return False
        
        return True
    
    def _check_multi_timeframe_confirmation(self, context: SignalContext, 
                                            signal: TradingSignal) -> bool:
        """多时间框架确认 — 在大周期上验证信号方向"""
        if not self._confirmation_required:
            return True
        
        indicators = context.indicators
        
        # 使用MA排列确认趋势方向
        ma20 = indicators.get("ma20", 0)
        ma50 = indicators.get("ma50", 0)
        adx = indicators.get("adx", 0)
        
        if ma20 <= 0 or ma50 <= 0:
            return True  # 数据不足时放行
        
        if signal.signal_type == SignalType.OPEN_LONG:
            # 做多需确认：大周期不处于下跌趋势
            if ma20 < ma50 and adx > 20:
                return False
        elif signal.signal_type == SignalType.OPEN_SHORT:
            # 做空需确认：大周期不处于上涨趋势
            if ma20 > ma50 and adx > 20:
                return False
        
        return True
    
    def _check_market_regime_filter(self, symbol: str, signal_type: SignalType) -> bool:
        """市场状态过滤 — 极端行情下拦截开仓信号"""
        if not self._market_regime_filter:
            return True
        
        try:
            regime_info = None
            provider = getattr(self, '_regime_provider', None)
            if callable(provider):
                regime_info = provider()
            elif isinstance(provider, dict):
                regime_info = provider
            elif self._regime_info_cache is not None:
                regime_info = self._regime_info_cache
            
            if regime_info is None:
                return True
            
            regime = regime_info.get("regime", "") if isinstance(regime_info, dict) else ""
            signal_type_str = (signal_type.value if hasattr(signal_type, 'value') else str(signal_type)).upper()
            
            blocked_types = self._regime_blocked_types.get(regime, [])
            if signal_type_str in blocked_types:
                self._signals_filtered_by_regime += 1
                logger.debug(f"Signal blocked by regime filter: {regime} -> {signal_type_str}")
                return False
        except Exception:
            pass
        
        return True
    
    def _apply_signal_strength_decay(self, signal: TradingSignal) -> TradingSignal:
        """信号强度衰减 — 信号产生后随时间衰减权重"""
        if signal.expiry is None:
            return signal
        
        now = datetime.now()
        if signal.timestamp:
            age_seconds = (now - signal.timestamp).total_seconds()
            if age_seconds > 30:
                decay = max(0.3, 1.0 - (age_seconds - 30) / 300)
                signal.weight = round(signal.weight * decay, 4)
                if signal.weight < 0.3:
                    signal.level = SignalLevel.WEAK
        
        return signal
    
    def _enrich_signal_context(self, signal: TradingSignal, context: SignalContext) -> TradingSignal:
        """增强信号上下文 — 附加市场、风险、持仓信息"""
        if self._include_market_context:
            signal.metadata["market_state"] = context.market_state
            signal.metadata["volatility_level"] = context.volatility_level
        
        if self._include_risk_metrics and context.indicators:
            signal.metadata["atr_percent"] = context.indicators.get("atr_percent", 0)
            signal.metadata["rsi"] = context.indicators.get("rsi", 50)
            signal.metadata["adx"] = context.indicators.get("adx", 0)
        
        if self._include_position_context:
            signal.metadata["position_side"] = context.position_side
            signal.metadata["position_size"] = context.position_size
            signal.metadata["unrealized_pnl_pct"] = context.unrealized_pnl_pct
        
        return signal
    
    def _aggregate_signals(self, symbol: str, signals: List[TradingSignal]) -> List[TradingSignal]:
        """信号聚合 — 在时间窗口内聚合同方向信号"""
        if not self._signal_aggregation or len(signals) <= 1:
            return signals
        
        now = datetime.now()
        
        # 按信号类型分组
        groups: Dict[str, List[TradingSignal]] = {}
        for s in signals:
            key = s.signal_type.value
            if key not in groups:
                groups[key] = []
            groups[key].append(s)
        
        aggregated = []
        for sig_type, group in groups.items():
            if len(group) <= 1:
                aggregated.extend(group)
                continue
            
            if len(group) > self._max_aggregated:
                group = sorted(group, key=lambda s: s.weight, reverse=True)[:self._max_aggregated]
            
            # 合并权重
            total_weight = sum(s.weight for s in group)
            avg_weight = total_weight / len(group)
            
            # 使用最高权重信号作为基础
            best = max(group, key=lambda s: s.weight)
            best.weight = min(avg_weight * 1.2, 1.0)
            best.reason = f"[聚合{len(group)}个] {best.reason}"
            best.metadata["aggregated_from"] = len(group)
            best.metadata["sources"] = [s.source.value for s in group]
            best.metadata["total_weight"] = round(total_weight, 4)
            
            aggregated.append(best)
            self._aggregated_signals_count += 1
        
        return aggregated
    
    def _evaluate_signal_quality(self, signal: TradingSignal) -> float:
        """评估信号质量 — 综合多因子评分
        
        优先使用外部 SignalQualityEngine（18因子评估），
        不可用时回退到内置简化评分。
        """
        # 尝试使用外部质量引擎
        if self._quality_engine is not None:
            try:
                signal_dict = self._trading_signal_to_dict(signal)
                result = self._quality_engine.evaluate_signal(signal_dict)
                if result and "overall_score" in result:
                    quality_score = float(result["overall_score"])
                    # 将详细因子评分写入metadata
                    signal.metadata["quality_factors"] = result.get("factor_scores", {})
                    signal.metadata["quality_breakdown"] = result.get("explanation", [])
                    signal.metadata["quality_grade"] = result.get("quality", "fair")
                    return min(1.0, max(0.0, quality_score))
            except Exception as e:
                logger.debug(f"External quality engine failed, using built-in: {e}")
        
        # 内置简化评分（回退方案）
        score = signal.weight * 0.6  # 基础权重贡献
        
        # 规则权重加成
        rule = next((r for r in self.rules if r.name == signal.strategy_id), None)
        if rule:
            score += rule.weight * 0.1
        
        # 信号级别加成
        if signal.level == SignalLevel.STRONG:
            score += 0.2
        elif signal.level == SignalLevel.MEDIUM:
            score += 0.1
        
        # 来源多样性加成（多来源确认的信号更可靠）
        sources = signal.metadata.get("sources", [])
        if isinstance(sources, list) and len(sources) >= 2:
            score += 0.05 * min(len(sources), 4)
        
        return min(1.0, max(0.0, score))
    
    def _trading_signal_to_dict(self, signal: TradingSignal) -> Dict[str, Any]:
        """将 TradingSignal 转换为 SignalQualityEngine 期望的 dict 格式"""
        direction = "long" if signal.signal_type in (SignalType.OPEN_LONG, SignalType.ADD_POSITION) else \
                   "short" if signal.signal_type in (SignalType.OPEN_SHORT,) else "close"
        side = "buy" if signal.signal_type in (SignalType.OPEN_LONG, SignalType.ADD_POSITION) else \
               "sell" if signal.signal_type in (SignalType.OPEN_SHORT,) else "close"
        
        return {
            "signal_id": signal.metadata.get("signal_id", f"{signal.symbol}:{signal.strategy_id}:{int(time.time())}"),
            "symbol": signal.symbol,
            "strategy_name": signal.strategy_id or signal.metadata.get("strategy_name", "signal_generator"),
            "signal_type": signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
            "direction": direction,
            "side": side,
            "price": signal.price,
            "quantity": signal.quantity,
            "leverage": signal.metadata.get("leverage", 1),
            "stop_loss": signal.metadata.get("stop_loss"),
            "take_profit": signal.metadata.get("take_profit"),
            "confidence": signal.weight,
            "timeframe": signal.metadata.get("timeframe", "1H"),
            "confirmed": signal.level in (SignalLevel.STRONG, SignalLevel.MEDIUM),
            "source": signal.source.value if hasattr(signal.source, 'value') else str(signal.source),
            "metadata": signal.metadata,
        }
    
    # ═══════════════════════════════════════════════════════════════
    # 生产级信号分发管道
    # ═══════════════════════════════════════════════════════════════
    
    def _dispatch_pipeline(self, signals: List[TradingSignal]) -> None:
        """生产级信号分发管道
        
        支持三种分发模式：
        - immediate: 立即逐条分发
        - batched: 收集后批量分发
        - aggregated: 聚合后分发
        
        支持三种优先级:
        - quality_first: 按质量评分排序
        - confidence_first: 按置信度排序
        - time_first: 按时间排序
        """
        if not signals:
            return
        
        dispatch_mode = self.config.get("dispatch_mode", "immediate")
        dispatch_priority = self.config.get("dispatch_priority", "quality_first")
        
        # 按优先级排序
        if dispatch_priority == "quality_first":
            signals = sorted(signals, key=lambda s: (
                s.metadata.get("quality_score", 0),
                s.weight
            ), reverse=True)
        elif dispatch_priority == "confidence_first":
            signals = sorted(signals, key=lambda s: s.weight, reverse=True)
        elif dispatch_priority == "time_first":
            signals = sorted(signals, key=lambda s: s.timestamp)
        
        if dispatch_mode == "batched":
            # 批量模式：收集完整批次后一次性分发
            self._dispatch_batch(signals)
        elif dispatch_mode == "aggregated":
            # 聚合模式已在 _aggregate_signals 中处理，这里直接分发
            self._dispatch_signals(signals)
        else:
            # immediate 模式：逐条分发
            self._dispatch_signals(signals)
    
    def _dispatch_batch(self, signals: List[TradingSignal]) -> None:
        """批量分发信号 — 打包为批次一次性发送"""
        batch_id = f"batch_{int(time.time() * 1000)}"
        batch = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "signal_count": len(signals),
            "signals": [s.to_dict() for s in signals],
            "summary": {
                "total_weight": round(sum(s.weight for s in signals), 4),
                "avg_quality": round(sum(s.metadata.get("quality_score", 0) for s in signals) / max(len(signals), 1), 4),
                "types": list(set(s.signal_type.value for s in signals)),
                "symbols": list(set(s.symbol for s in signals)),
            }
        }
        
        with self._lock:
            callbacks = self._signal_callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(batch)
            except Exception as e:
                logger.error(f"Error in batch signal callback: {e}")
        
        logger.info(f"Batch dispatched: {batch_id} ({len(signals)} signals, "
                    f"avg_quality={batch['summary']['avg_quality']:.3f})")
    
    # ═══════════════════════════════════════════════════════════════
    # 信号持久化
    # ═══════════════════════════════════════════════════════════════
    
    def _persist_signals(self, signals: List[TradingSignal]) -> None:
        """持久化信号到数据库"""
        if not self.config.get("persist_signals", True):
            return
        
        try:
            import sqlite3
            import json
            import os
            
            db_path = self.config.get("signal_db_path", "./data/trading.db")
            os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
            
            conn = sqlite3.connect(db_path, timeout=5)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=3000")
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signal_generator_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    signal_type TEXT,
                    source TEXT,
                    level TEXT,
                    weight REAL,
                    price REAL,
                    quantity REAL,
                    quality_score REAL,
                    reason TEXT,
                    strategy_id TEXT,
                    metadata TEXT,
                    timestamp TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sgh_symbol ON signal_generator_history(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_sgh_timestamp ON signal_generator_history(timestamp)')
            
            now_iso = datetime.now().isoformat()
            for signal in signals:
                try:
                    metadata_json = json.dumps(signal.metadata, ensure_ascii=False, default=str)
                except (TypeError, ValueError):
                    metadata_json = "{}"
                
                conn.execute('''
                    INSERT INTO signal_generator_history 
                    (symbol, signal_type, source, level, weight, price, quantity, 
                     quality_score, reason, strategy_id, metadata, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.symbol,
                    signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                    signal.source.value if hasattr(signal.source, 'value') else str(signal.source),
                    signal.level.value if hasattr(signal.level, 'value') else str(signal.level),
                    signal.weight,
                    signal.price,
                    signal.quantity,
                    signal.metadata.get("quality_score", 0),
                    signal.reason[:500] if signal.reason else "",
                    signal.strategy_id or "",
                    metadata_json,
                    signal.timestamp.isoformat() if signal.timestamp else now_iso,
                    now_iso,
                ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"Signal persistence failed (non-critical): {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 生产级信号生成流水线（主入口）
    # ═══════════════════════════════════════════════════════════════
    
    def generate_signals_pipeline(self, context: SignalContext) -> List[TradingSignal]:
        """
        生产级信号生成流水线
        
        完整流程：
        1. 规则评估 → 2. 频率限流 → 3. 冲突解决 → 4. 多框架确认
        → 5. 市场过滤 → 6. 质量评分 → 7. 上下文增强 → 8. 聚合 → 9. 分发
        """
        if not self.config.get("enabled", True):
            return []
        
        # Step 1: 规则评估
        raw_signals = []
        for rule in self.rules:
            try:
                signal = rule.evaluate(context)
                if signal:
                    raw_signals.append(signal)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        if not raw_signals:
            return []
        
        # Step 2: 频率限流
        filtered_signals = []
        for signal in raw_signals:
            direction = "long" if signal.signal_type in (SignalType.OPEN_LONG, SignalType.ADD_POSITION) else \
                       "short" if signal.signal_type in (SignalType.OPEN_SHORT,) else "close"
            if self._check_rate_limit(signal.symbol, direction):
                filtered_signals.append(signal)
        
        if not filtered_signals:
            return []
        
        # Step 3: 冲突解决
        filtered_signals = self._resolve_signal_conflicts(filtered_signals)
        if not filtered_signals:
            return []
        
        # Step 4: 多框架确认 + 市场过滤
        confirmed_signals = []
        for signal in filtered_signals:
            if not self._check_multi_timeframe_confirmation(context, signal):
                continue
            if not self._check_market_regime_filter(signal.symbol, signal.signal_type):
                continue
            
            # 反向信号延迟检查
            direction = "long" if signal.signal_type in (SignalType.OPEN_LONG, SignalType.ADD_POSITION) else \
                       "short" if signal.signal_type in (SignalType.OPEN_SHORT,) else "close"
            if signal.signal_type in (SignalType.OPEN_LONG, SignalType.OPEN_SHORT):
                if not self._check_opposite_signal_delay(signal.symbol, direction):
                    continue
            
            confirmed_signals.append(signal)
        
        if not confirmed_signals:
            return []
        
        # Step 5: 信号强度衰减
        confirmed_signals = [self._apply_signal_strength_decay(s) for s in confirmed_signals]
        
        # Step 6: 权重排序
        confirmed_signals.sort(key=lambda s: s.weight, reverse=True)
        
        # Step 7: 质量评分
        for signal in confirmed_signals:
            quality = self._evaluate_signal_quality(signal)
            signal.metadata["quality_score"] = round(quality, 4)
        
        # Step 8: 上下文增强
        confirmed_signals = [self._enrich_signal_context(s, context) for s in confirmed_signals]
        
        # Step 9: 信号聚合
        final_signals = self._aggregate_signals(context.symbol, confirmed_signals)
        
        # Step 9.5: 防针对量化数据 — 信号混淆 + 模式检测
        final_signals = self._run_anti_targeting_pipeline(final_signals, context)
        
        # Step 10: 记录 → 过滤不可分发信号 → 分发 → 持久化
        self._record_signals(context.symbol, final_signals)
        dispatchable = [s for s in final_signals if self._should_dispatch(s)]
        self._dispatch_pipeline(dispatchable)
        self._persist_signals(final_signals)
        
        return final_signals
    
    def should_dispatch_signal(self, signal: TradingSignal) -> bool:
        """判断信号是否应该分发（在流水线外独立调用）"""
        return self._should_dispatch(signal)
        
    def enable_adaptive_learning(self, learning_rate: float = None) -> None:
        """启用策略参数自适应学习"""
        self._adaptive_enabled = True
        if learning_rate is not None:
            self._learning_rate = learning_rate
        if self._learning_task is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop is not None:
                self._learning_task = loop.create_task(self._adaptive_learning_loop())
                logger.info(f"Adaptive learning enabled (lr={self._learning_rate})")
            else:
                # 无运行中的事件循环（同步上下文）：降级，不启动后台学习任务
                logger.warning("Adaptive learning enabled without running event loop; "
                               "background learning task skipped")
    
    def disable_adaptive_learning(self) -> None:
        """禁用自适应学习"""
        self._adaptive_enabled = False
        if self._learning_task:
            self._learning_task.cancel()
            self._learning_task = None
        logger.info("Adaptive learning disabled")
    
    def record_signal_outcome(self, signal_id: str, pnl: float, 
                               is_win: bool, executed: bool = True) -> None:
        """
        记录信号执行结果，用于自适应学习
        
        Args:
            signal_id: 信号ID (strategy_rule_name)
            pnl: 实际盈亏
            is_win: 是否盈利
            executed: 是否实际执行
        """
        if not self._adaptive_enabled:
            return
        
        with self._lock:
            if signal_id not in self._rule_performance:
                return
            
            perf = self._rule_performance[signal_id]
            perf.total_signals += 1
            if executed:
                perf.executed_signals += 1
                if is_win:
                    perf.winning_signals += 1
                    perf.total_profit += abs(pnl)
                else:
                    perf.losing_signals += 1
                    perf.total_loss += abs(pnl)
            else:
                perf.filtered_signals += 1
    
    async def _adaptive_learning_loop(self) -> None:
        """参数自适应学习循环"""
        logger.info(f"Adaptive learning loop started (interval={self._learning_interval}s)")
        while self._adaptive_enabled:
            try:
                await asyncio.sleep(self._learning_interval)
                self._optimize_rule_weights()
                self._learning_generation += 1
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Adaptive learning loop error: {e}")
                await asyncio.sleep(60)
    
    def _optimize_rule_weights(self) -> None:
        """分析历史信号表现，动态优化各规则权重"""
        if not self._adaptive_enabled:
            return
        
        with self._lock:
            weight_changes = []
            
            for rule_name, perf in self._rule_performance.items():
                if perf.total_signals < self._min_samples_for_learning:
                    continue
                
                # 找到对应的规则实例
                rule = None
                for r in self.rules:
                    if r.name == rule_name:
                        rule = r
                        break
                if rule is None:
                    continue
                
                # 计算执行率（信号被执行的比例）
                execution_rate = perf.executed_signals / max(perf.total_signals, 1)
                
                # 计算胜率
                total_executed = perf.winning_signals + perf.losing_signals
                win_rate = perf.winning_signals / max(total_executed, 1)
                
                # 计算盈亏比
                profit_factor = (perf.total_profit / max(perf.total_loss, 0.001))
                
                # 综合评分
                quality_score = (win_rate * 0.5 + min(profit_factor / 3.0, 1.0) * 0.3 + execution_rate * 0.2)
                
                old_weight = rule.weight
                
                # 质量评分→权重调整：0.5=基准，>0.5加权重，<0.5降权重
                if quality_score > 0.6:
                    # 表现好：增加权重
                    delta = self._learning_rate * (quality_score - 0.5)
                    new_weight = min(old_weight + delta, 2.0)
                elif quality_score < 0.4:
                    # 表现差：降低权重
                    delta = self._learning_rate * (0.5 - quality_score)
                    new_weight = max(old_weight - delta, 0.3)
                else:
                    # 表现一般：微调
                    delta = self._learning_rate * (quality_score - 0.5) * 0.5
                    new_weight = old_weight + delta
                
                # 限制单次变化幅度
                change = new_weight - old_weight
                if abs(change) > self._max_weight_change:
                    change = self._max_weight_change if change > 0 else -self._max_weight_change
                    new_weight = old_weight + change
                
                rule.weight = round(new_weight, 4)
                
                if abs(change) > 0.01:
                    weight_changes.append({
                        "rule": rule_name,
                        "old_weight": round(old_weight, 4),
                        "new_weight": round(new_weight, 4),
                        "change": round(change, 4),
                        "quality_score": round(quality_score, 4),
                        "win_rate": round(win_rate, 4),
                        "profit_factor": round(profit_factor, 4),
                        "samples": perf.total_signals
                    })
            
            if weight_changes:
                logger.info(f"Adaptive learning gen#{self._learning_generation}: "
                           f"{len(weight_changes)} rules adjusted")
                for change in weight_changes:
                    direction = "↑" if change["change"] > 0 else "↓"
                    logger.info(f"  {change['rule']}: {change['old_weight']}→{change['new_weight']} "
                               f"{direction} (score={change['quality_score']}, "
                               f"wr={change['win_rate']}, pf={change['profit_factor']})")
    
    def get_learning_status(self) -> Dict[str, Any]:
        """获取自适应学习状态"""
        with self._lock:
            performances = {}
            for name, perf in self._rule_performance.items():
                total_exec = perf.winning_signals + perf.losing_signals
                performances[name] = {
                    "total_signals": perf.total_signals,
                    "executed_signals": perf.executed_signals,
                    "winning_signals": perf.winning_signals,
                    "losing_signals": perf.losing_signals,
                    "win_rate": round(perf.winning_signals / max(total_exec, 1), 4),
                    "profit_factor": round(perf.total_profit / max(perf.total_loss, 0.001), 4),
                    "execution_rate": round(perf.executed_signals / max(perf.total_signals, 1), 4),
                    "current_weight": next((r.weight for r in self.rules if r.name == name), 0)
                }
            
            return {
                "enabled": self._adaptive_enabled,
                "learning_rate": self._learning_rate,
                "generation": self._learning_generation,
                "rule_performances": performances,
                "rule_weights": {r.name: r.weight for r in self.rules}
            }
    
    def _setup_default_rules(self) -> None:
        """设置默认信号规则"""
        rule_configs = self.config.get("rules", {})
        
        self.rules.append(TrendBreakoutRule(rule_configs.get("trend_breakout")))
        self.rules.append(MeanReversionRule(rule_configs.get("mean_reversion")))
        self.rules.append(VolatilityArbitrageRule(rule_configs.get("volatility_arb")))
        self.rules.append(GridReboundRule(rule_configs.get("grid_rebound")))
        self.rules.append(RiskControlRule(rule_configs.get("risk_control")))
        self.rules.append(DivergenceRule(rule_configs.get("divergence_detection")))
        self.rules.append(MarketStructureRule(rule_configs.get("market_structure")))
        
        logger.info(f"SignalGenerator initialized with {len(self.rules)} rules")
    
    def add_rule(self, rule: SignalRule) -> None:
        """添加信号规则"""
        with self._lock:
            self.rules.append(rule)
            if rule.name not in self._rule_performance:
                self._rule_performance[rule.name] = RulePerformance(rule_name=rule.name)
    
    def register_callback(self, callback: Callable[[TradingSignal], None]) -> None:
        """注册信号回调"""
        with self._lock:
            self._signal_callbacks.append(callback)
    
    def generate_signals(self, context: SignalContext) -> List[TradingSignal]:
        """
        生成信号列表
        
        按权重排序，高权重信号优先
        """
        signals = []
        
        for rule in self.rules:
            try:
                signal = rule.evaluate(context)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
        
        if signals:
            signals.sort(key=lambda s: s.weight, reverse=True)
            
            self._record_signals(context.symbol, signals)
            
            self._dispatch_signals(signals)
        
        return signals
    
    def generate_primary_signal(self, context: SignalContext) -> Optional[TradingSignal]:
        """生成主信号（权重最高的信号）"""
        signals = self.generate_signals(context)
        
        if signals:
            return signals[0]
        
        return None
    
    def generate_combined_signal(self, context: SignalContext) -> Optional[TradingSignal]:
        """
        生成组合信号（综合多个规则的判断）
        
        当多个规则指向相同方向时，增强信号强度
        """
        signals = self.generate_signals(context)
        
        if not signals:
            return None
        
        direction_weights = {
            SignalType.OPEN_LONG: 0.0,
            SignalType.OPEN_SHORT: 0.0,
            SignalType.ADD_POSITION: 0.0,
            SignalType.REDUCE_POSITION: 0.0,
            SignalType.TAKE_PROFIT: 0.0,
            SignalType.STOP_LOSS: 0.0,
            SignalType.HEDGE: 0.0,
        }
        
        reasons = []
        sources = []
        
        for signal in signals:
            if signal.signal_type in direction_weights:
                direction_weights[signal.signal_type] += signal.weight
                reasons.append(signal.reason)
                sources.append(signal.source.value)
        
        best_type = max(direction_weights, key=direction_weights.get)
        best_weight = direction_weights[best_type]
        
        if best_weight < 0.3:
            return None
        
        level = SignalLevel.STRONG if best_weight >= 0.8 else (
            SignalLevel.MEDIUM if best_weight >= 0.5 else SignalLevel.WEAK
        )
        
        combined_reason = "; ".join(reasons[:3])
        
        return TradingSignal(
            symbol=context.symbol,
            signal_type=best_type,
            source=SignalSource.MOMENTUM,
            level=level,
            weight=min(best_weight, 1.0),
            price=context.current_price,
            quantity=0.0,
            timestamp=datetime.now(),
            reason=combined_reason,
            metadata={"sources": sources, "rule_count": len(signals)}
        )
    
    def generate_open_signal(self, context: SignalContext, 
                             direction: str = "auto") -> Optional[TradingSignal]:
        """生成开仓信号"""
        signals = self.generate_signals(context)
        
        open_signals = [s for s in signals if s.signal_type in 
                       [SignalType.OPEN_LONG, SignalType.OPEN_SHORT]]
        
        if direction == "long":
            open_signals = [s for s in open_signals if s.signal_type == SignalType.OPEN_LONG]
        elif direction == "short":
            open_signals = [s for s in open_signals if s.signal_type == SignalType.OPEN_SHORT]
        
        if open_signals:
            return open_signals[0]
        
        return None
    
    def generate_close_signal(self, context: SignalContext,
                               close_type: str = "all") -> Optional[TradingSignal]:
        """生成平仓信号"""
        signals = self.generate_signals(context)
        
        close_signals = [s for s in signals if s.signal_type in
                        [SignalType.TAKE_PROFIT, SignalType.STOP_LOSS, 
                         SignalType.REDUCE_POSITION, SignalType.CLOSE_ALL]]
        
        if close_type == "profit":
            close_signals = [s for s in close_signals if s.signal_type == SignalType.TAKE_PROFIT]
        elif close_type == "loss":
            close_signals = [s for s in close_signals if s.signal_type == SignalType.STOP_LOSS]
        
        if close_signals:
            return close_signals[0]
        
        return None
    
    def should_add_position(self, context: SignalContext) -> Tuple[bool, float, str]:
        """
        判断是否应该加仓
        
        Returns:
            (should_add, suggested_ratio, reason)
        """
        signals = self.generate_signals(context)
        
        add_signals = [s for s in signals if s.signal_type == SignalType.ADD_POSITION]
        
        if not add_signals:
            return False, 0.0, ""
        
        best_signal = add_signals[0]
        
        ratio = min(best_signal.weight * 0.5, 0.3)
        
        return True, ratio, best_signal.reason
    
    def should_reduce_position(self, context: SignalContext) -> Tuple[bool, float, str]:
        """
        判断是否应该减仓
        
        Returns:
            (should_reduce, suggested_ratio, reason)
        """
        signals = self.generate_signals(context)
        
        reduce_signals = [s for s in signals if s.signal_type in
                         [SignalType.REDUCE_POSITION, SignalType.TAKE_PROFIT]]
        
        if not reduce_signals:
            return False, 0.0, ""
        
        best_signal = reduce_signals[0]
        
        ratio = min(best_signal.weight * 0.5, 0.5)
        
        return True, ratio, best_signal.reason
    
    def _record_signals(self, symbol: str, signals: List[TradingSignal]) -> None:
        """记录信号历史"""
        with self._lock:
            if symbol not in self._signal_history:
                self._signal_history[symbol] = deque(maxlen=self._history_size)
            
            for signal in signals:
                self._signal_history[symbol].append(signal)
            
            self._signal_count += len(signals)
            self._signal_by_symbol[symbol] = self._signal_by_symbol.get(symbol, 0) + len(signals)
            
            for signal in signals:
                type_key = signal.signal_type.value
                self._signal_by_type[type_key] = self._signal_by_type.get(type_key, 0) + 1
    
    def _dispatch_signals(self, signals: List[TradingSignal]) -> None:
        """分发信号到回调"""
        with self._lock:
            callbacks = self._signal_callbacks.copy()
        
        for signal in signals:
            for callback in callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    logger.error(f"Error in signal callback: {e}")
    
    def get_signal_history(self, symbol: str, limit: int = 20) -> List[TradingSignal]:
        """获取信号历史"""
        with self._lock:
            if symbol in self._signal_history:
                return list(self._signal_history[symbol])[-limit:]
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息（含流水线健康度）"""
        with self._lock:
            # 计算流水线健康度
            total_evaluated = self._signal_count
            passed_filter = total_evaluated - self._signals_filtered_by_rate - \
                           self._signals_filtered_by_conflict - self._signals_filtered_by_regime - \
                           self._signals_filtered_by_quality
            
            pipeline_stats = {
                "total_signals": self._signal_count,
                "signals_by_type": self._signal_by_type.copy(),
                "signals_by_symbol": self._signal_by_symbol.copy(),
                "symbols_tracked": len(self._signal_history),
                "rules_count": len(self.rules),
                # 流水线过滤统计
                "pipeline_health": {
                    "filtered_by_rate": self._signals_filtered_by_rate,
                    "filtered_by_conflict": self._signals_filtered_by_conflict,
                    "filtered_by_regime": self._signals_filtered_by_regime,
                    "filtered_by_quality": self._signals_filtered_by_quality,
                    "aggregated_signals": self._aggregated_signals_count,
                    "passed_filter": max(passed_filter, 0),
                    "pass_rate": round(max(passed_filter, 0) / max(total_evaluated, 1), 4),
                },
                # 流水线配置快照
                "pipeline_config": {
                    "dispatch_mode": self.config.get("dispatch_mode", "immediate"),
                    "dispatch_priority": self.config.get("dispatch_priority", "quality_first"),
                    "conflict_resolution": self._conflict_resolution,
                    "market_regime_filter": self._market_regime_filter,
                    "confirmation_required": self._confirmation_required,
                    "signal_aggregation": self._signal_aggregation,
                    "min_signal_weight": self._min_signal_weight,
                    "min_quality_score": self._min_quality_score,
                },
                # 规则权重快照
                "rule_weights": {r.name: r.weight for r in self.rules},
                "rule_enabled": {r.name: r.enabled for r in self.rules},
                # 防针对量化数据统计
                "anti_targeting": self.get_anti_targeting_stats(),
            }
            
            return pipeline_stats
    
    def clear_history(self, symbol: str = None) -> None:
        """清除信号历史"""
        with self._lock:
            if symbol:
                if symbol in self._signal_history:
                    self._signal_history[symbol].clear()
            else:
                self._signal_history.clear()


_generator_instance: Optional[SignalGenerator] = None
_generator_lock = threading.Lock()

def get_signal_generator(config: Dict[str, Any] = None) -> SignalGenerator:
    """获取信号生成器单例（线程安全）"""
    global _generator_instance
    if _generator_instance is None:
        with _generator_lock:
            if _generator_instance is None:
                _generator_instance = SignalGenerator(config)
    return _generator_instance