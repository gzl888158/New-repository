"""
多策略容器调度器
================
核心定位：高波动捕捉、多策略并行运算，独立算力隔离，策略互不干扰

特性：
- 单交易对独立策略实例：某币种策略异常不会影响其余币种运行
- 四类激进策略并行：趋势突破、网格震荡、波动率套利、波段反转
- 算力隔离：每个策略实例独立线程/协程
- 资源管控：CPU、内存、请求频率限制
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable, Type
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import traceback
from loguru import logger


class StrategyState(Enum):
    """策略状态"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    FROZEN = "frozen"
    ERROR = "error"
    STOPPED = "stopped"


class StrategyType(Enum):
    """策略类型"""
    TREND_BREAKOUT = "trend_breakout"
    GRID_OSCILLATION = "grid_oscillation"
    VOLATILITY_ARB = "volatility_arb"
    BAND_REVERSAL = "band_reversal"


@dataclass
class StrategyMetrics:
    """策略运行指标"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_profit: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    
    signals_generated: int = 0
    signals_executed: int = 0
    execution_rate: float = 0.0
    
    compute_time_ms: float = 0.0
    compute_count: int = 0
    
    def update_pnl(self, pnl: float) -> None:
        self.total_pnl += pnl
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.avg_profit = (self.avg_profit * (self.winning_trades - 1) + pnl) / self.winning_trades
        else:
            self.losing_trades += 1
            self.avg_loss = (self.avg_loss * (self.losing_trades - 1) + abs(pnl)) / self.losing_trades
        
        if self.winning_trades + self.losing_trades > 0:
            self.win_rate = self.winning_trades / (self.winning_trades + self.losing_trades)
        
        if self.avg_loss > 0:
            self.profit_factor = (self.avg_profit * self.winning_trades) / (self.avg_loss * self.losing_trades)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": round(self.total_pnl, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_profit": round(self.avg_profit, 4),
            "avg_loss": round(self.avg_loss, 4),
            "profit_factor": round(self.profit_factor, 4),
            "signals_generated": self.signals_generated,
            "signals_executed": self.signals_executed,
            "execution_rate": round(self.execution_rate, 4),
            "avg_compute_ms": round(self.compute_time_ms / max(self.compute_count, 1), 2)
        }


@dataclass
class StrategyInstance:
    """策略实例（单交易对）"""
    id: str
    strategy_type: StrategyType
    symbol: str
    state: StrategyState = StrategyState.INITIALIZING
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: StrategyMetrics = field(default_factory=StrategyMetrics)
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    last_signal_at: Optional[datetime] = None
    last_trade_at: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_at: Optional[datetime] = None
    
    task: Optional[asyncio.Task] = None
    thread: Optional[threading.Thread] = None
    
    position_side: Optional[str] = None
    position_size: float = 0.0
    entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    
    def is_active(self) -> bool:
        return self.state in [StrategyState.RUNNING, StrategyState.PAUSED]
    
    def is_tradable(self) -> bool:
        return self.state == StrategyState.RUNNING
    
    def freeze(self, reason: str) -> None:
        self.state = StrategyState.FROZEN
        self.last_error = reason
        self.last_error_at = datetime.now()
    
    def unfreeze(self) -> None:
        if self.state == StrategyState.FROZEN:
            self.state = StrategyState.RUNNING
    
    def pause(self) -> None:
        if self.state == StrategyState.RUNNING:
            self.state = StrategyState.PAUSED
    
    def resume(self) -> None:
        if self.state == StrategyState.PAUSED:
            self.state = StrategyState.RUNNING
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "strategy_type": self.strategy_type.value,
            "symbol": self.symbol,
            "state": self.state.value,
            "position_side": self.position_side,
            "position_size": self.position_size,
            "entry_price": self.entry_price,
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "metrics": self.metrics.to_dict(),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_signal_at": self.last_signal_at.isoformat() if self.last_signal_at else None,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "last_error": self.last_error,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None
        }


class BaseStrategy(ABC):
    """策略基类"""
    
    def __init__(self, instance: StrategyInstance, config: Dict[str, Any]):
        self.instance = instance
        self.config = config
        self._running = False
        self._paused = False
        self._indicator_engine = None  # 由 StrategyContainer 注入
        # P22: 状态无漂移 - 交易所仓位提供者，策略可通过此接口获取真实仓位
        self._position_provider: Optional[Callable[[], Optional[List[Dict[str, Any]]]]] = None
        # P22: 最近一次交易所仓位快照，每次 tick/bar 前由 StrategyContainer 更新
        self._exchange_positions_snapshot: Optional[List[Dict[str, Any]]] = None
        self._exchange_positions_ts: float = 0.0
        # P22: 回测与实盘隔离 - 标记当前运行模式，防止回测代码误入实盘
        self._is_backtest_mode: bool = False
        self._is_paper_trading: bool = False
    
    def set_indicator_engine(self, engine) -> None:
        """注入指标引擎，使策略能使用实时技术指标"""
        self._indicator_engine = engine
    
    def set_position_provider(self, provider: Callable[[], Optional[List[Dict[str, Any]]]]) -> None:
        """P22: 注入交易所仓位提供者，用于状态无漂移校验"""
        self._position_provider = provider
    
    def set_backtest_mode(self, enabled: bool = True) -> None:
        """P22: 设置回测模式，禁用实盘API调用"""
        self._is_backtest_mode = enabled
        if enabled:
            logger.info(f"Strategy {self.instance.id}: Backtest mode enabled - real API calls disabled")
    
    def set_paper_trading_mode(self, enabled: bool = True) -> None:
        """P22: 设置模拟盘模式"""
        self._is_paper_trading = enabled
    
    def assert_live_mode(self, operation: str = "trade") -> None:
        """P22: 实盘模式断言 - 回测代码禁止调用实盘API
        
        Raises:
            RuntimeError: 当策略处于回测模式时调用实盘操作
        """
        if self._is_backtest_mode:
            raise RuntimeError(
                f"P22: BACKTEST SAFETY BLOCK - '{operation}' attempted in backtest mode. "
                f"Backtest code must not call live trading APIs. "
                f"Use BacktestAdapter to wrap strategy for safe backtesting."
            )
    
    def is_backtest_mode(self) -> bool:
        """P22: 检查是否处于回测模式"""
        return self._is_backtest_mode
    
    def is_paper_trading(self) -> bool:
        """P22: 检查是否处于模拟盘模式"""
        return self._is_paper_trading
    
    def _get_indicators(self) -> Optional[Dict[str, float]]:
        """获取当前交易对的实时指标数据"""
        if self._indicator_engine is None:
            return None
        try:
            indicator_set = self._indicator_engine.calculate_all(self.instance.symbol)
            return {
                name: result.value
                for name, result in indicator_set.indicators.items()
            }
        except Exception:
            return None
    
    async def verify_exchange_positions(self) -> Dict[str, Any]:
        """P22: 状态无漂移 - 校验本地仓位与交易所仓位一致性
        子类可覆写此方法实现策略特定的仓位校验。
        返回: {"drifted": bool, "corrections": list, "exchange_positions": list}
        """
        return {"drifted": False, "corrections": [], "exchange_positions": []}
    
    @abstractmethod
    async def on_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理行情数据，返回信号"""
        pass
    
    @abstractmethod
    async def on_bar(self, bar_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """处理K线数据，返回信号"""
        pass
    
    @abstractmethod
    async def on_position_update(self, position: Dict[str, Any]) -> None:
        """处理仓位更新"""
        pass
    
    def start(self) -> None:
        self._running = True
        self._paused = False
        self.instance.state = StrategyState.RUNNING
        self.instance.started_at = datetime.now()
    
    def stop(self) -> None:
        self._running = False
        self.instance.state = StrategyState.STOPPED
    
    def pause(self) -> None:
        self._paused = True
        self.instance.pause()
    
    def resume(self) -> None:
        self._paused = False
        self.instance.resume()
    
    def is_running(self) -> bool:
        return self._running and not self._paused


class TrendBreakoutStrategy(BaseStrategy):
    """趋势突破策略 — 增强版：使用IndicatorEngine实时指标"""
    
    def __init__(self, instance: StrategyInstance, config: Dict[str, Any]):
        super().__init__(instance, config)
        self.breakout_threshold = config.get("breakout_threshold", 0.02)
        self.confirmation_periods = config.get("confirmation_periods", 3)
        self.trailing_stop_pct = config.get("trailing_stop_pct", 0.015)
        self.min_adx = config.get("min_adx", 20)  # ADX阈值
        
        self._price_history: deque = deque(maxlen=100)
        self._high_price = 0.0
        self._low_price = float('inf')
        self._breakout_level = 0.0
        self._last_bar_time = None
    
    async def on_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_running():
            return None
        
        price = tick_data.get("price", 0)
        if price <= 0:
            return None
        
        self._price_history.append(price)
        self._high_price = max(self._high_price, price)
        self._low_price = min(self._low_price, price)
        
        return None
    
    async def on_bar(self, bar_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_running():
            return None
        
        close = bar_data.get("close", 0)
        high = bar_data.get("high", 0)
        low = bar_data.get("low", 0)
        
        if close <= 0:
            return None
        
        # 使用指标引擎获取实时ADX、RSI、MACD
        indicators = self._get_indicators()
        
        adx = indicators.get("adx", 20) if indicators else 20
        plus_di = indicators.get("plus_di", 0) if indicators else 0
        minus_di = indicators.get("minus_di", 0) if indicators else 0
        rsi = indicators.get("rsi", 50) if indicators else 50
        macd_hist = indicators.get("macd_histogram", 0) if indicators else 0
        
        range_pct = (high - low) / close if close > 0 else 0
        
        signal = None
        
        if not self.instance.position_side:
            # 上升趋势确认：ADX>阈值 + DI金叉 + MACD>0 + RSI未超买
            if (adx > self.min_adx and plus_di > minus_di 
                    and macd_hist > 0 and rsi < 70
                    and close > self._high_price * (1 - self.breakout_threshold)):
                strength = min((adx - self.min_adx) / 30 + range_pct * 3, 1.0)
                signal = {
                    "type": "open_long",
                    "symbol": self.instance.symbol,
                    "price": close,
                    "reason": f"趋势突破做多 (ADX={adx:.1f} +DI={plus_di:.1f} RSI={rsi:.1f})",
                    "strength": strength,
                    "strategy_type": "trend_breakout"
                }
            # 下降趋势确认
            elif (adx > self.min_adx and minus_di > plus_di
                    and macd_hist < 0 and rsi > 30
                    and close < self._low_price * (1 + self.breakout_threshold)):
                strength = min((adx - self.min_adx) / 30 + range_pct * 3, 1.0)
                signal = {
                    "type": "open_short",
                    "symbol": self.instance.symbol,
                    "price": close,
                    "reason": f"趋势突破做空 (ADX={adx:.1f} -DI={minus_di:.1f} RSI={rsi:.1f})",
                    "strength": strength,
                    "strategy_type": "trend_breakout"
                }
        else:
            # 已有持仓：趋势反转检测
            if self.instance.position_side == "long" and minus_di > plus_di and adx > self.min_adx:
                signal = {
                    "type": "take_profit",
                    "symbol": self.instance.symbol,
                    "price": close,
                    "reason": f"上升趋势反转 (ADX={adx:.1f})",
                    "strength": 0.6,
                    "strategy_type": "trend_breakout"
                }
            elif self.instance.position_side == "short" and plus_di > minus_di and adx > self.min_adx:
                signal = {
                    "type": "take_profit",
                    "symbol": self.instance.symbol,
                    "price": close,
                    "reason": f"下降趋势反转 (ADX={adx:.1f})",
                    "strength": 0.6,
                    "strategy_type": "trend_breakout"
                }
        
        self.instance.metrics.signals_generated += 1
        
        return signal
    
    async def on_position_update(self, position: Dict[str, Any]) -> None:
        side = position.get("side")
        size = position.get("size", 0)
        entry_price = position.get("entry_price", 0)
        
        self.instance.position_side = side if abs(size) > 0 else None
        self.instance.position_size = abs(size)
        self.instance.entry_price = entry_price if entry_price > 0 else 0


class GridOscillationStrategy(BaseStrategy):
    """网格震荡策略"""
    
    def __init__(self, instance: StrategyInstance, config: Dict[str, Any]):
        super().__init__(instance, config)
        self.grid_count = config.get("grid_count", 10)
        self.grid_spacing_pct = config.get("grid_spacing_pct", 0.01)
        self.profit_target_pct = config.get("profit_target_pct", 0.005)
        
        self._grid_levels: List[float] = []
        self._base_price = 0.0
        self._filled_grids: Dict[int, bool] = {}
    
    async def on_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_running():
            return None
        
        price = tick_data.get("price", 0)
        if price <= 0:
            return None
        
        if self._base_price == 0:
            self._base_price = price
            self._init_grid_levels()
        
        return None
    
    async def on_bar(self, bar_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_running():
            return None
        
        close = bar_data.get("close", 0)
        if close <= 0:
            return None
        
        signal = None
        
        for i, level in enumerate(self._grid_levels):
            if not self._filled_grids.get(i, False):
                if close <= level:
                    signal = {
                        "type": "open_long",
                        "symbol": self.instance.symbol,
                        "price": close,
                        "reason": f"触及网格买点 {i+1}/{self.grid_count}",
                        "strength": 0.5,
                        "grid_level": i
                    }
                    self._filled_grids[i] = True
                    break
                elif close >= level:
                    signal = {
                        "type": "open_short",
                        "symbol": self.instance.symbol,
                        "price": close,
                        "reason": f"触及网格卖点 {i+1}/{self.grid_count}",
                        "strength": 0.5,
                        "grid_level": i
                    }
                    self._filled_grids[i] = True
                    break
        
        self.instance.metrics.signals_generated += 1
        
        return signal
    
    async def on_position_update(self, position: Dict[str, Any]) -> None:
        self.instance.position_side = position.get("side") if position.get("size", 0) != 0 else None
        self.instance.position_size = abs(position.get("size", 0))
    
    def _init_grid_levels(self) -> None:
        if self._base_price <= 0:
            return
        
        self._grid_levels = []
        for i in range(1, self.grid_count + 1):
            buy_level = self._base_price * (1 - i * self.grid_spacing_pct)
            self._grid_levels.append(round(buy_level, 6))


class VolatilityArbStrategy(BaseStrategy):
    """波动率套利策略 — 增强版：使用IndicatorEngine实时波动率分位数"""
    
    def __init__(self, instance: StrategyInstance, config: Dict[str, Any]):
        super().__init__(instance, config)
        self.vol_high_threshold = config.get("vol_high_threshold", 0.75)
        self.vol_low_threshold = config.get("vol_low_threshold", 0.25)
        self.hedge_ratio = config.get("hedge_ratio", 0.5)
    
    async def on_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None
    
    async def on_bar(self, bar_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_running():
            return None
        
        close = bar_data.get("close", 0)
        if close <= 0:
            return None
        
        # 使用指标引擎获取波动率分位数、ATR%
        indicators = self._get_indicators()
        
        vol_pct = indicators.get("volatility_percentile", 0.5) if indicators else 0.5
        atr_pct = indicators.get("atr_percent", 0) if indicators else 0
        boll_width = indicators.get("bollinger_width", 0) if indicators else 0
        
        signal = None
        
        # 高波动率 → 启动对冲
        if vol_pct > self.vol_high_threshold:
            strength = min((vol_pct - self.vol_high_threshold) / (1.0 - self.vol_high_threshold) * 0.8 + 0.3, 1.0)
            signal = {
                "type": "hedge",
                "symbol": self.instance.symbol,
                "price": close,
                "reason": f"高波动率对冲 (vol_pct={vol_pct:.2f} ATR={atr_pct*100:.2f}%)",
                "strength": strength,
                "hedge_ratio": self.hedge_ratio,
                "strategy_type": "volatility_arb"
            }
        # 低波动率 → 适合建仓
        elif vol_pct < self.vol_low_threshold:
            strength = min((self.vol_low_threshold - vol_pct) / self.vol_low_threshold * 0.5 + 0.3, 1.0)
            if not self.instance.position_side:
                signal = {
                    "type": "open_long",
                    "symbol": self.instance.symbol,
                    "price": close,
                    "reason": f"低波动率建仓 (vol_pct={vol_pct:.2f})",
                    "strength": strength,
                    "strategy_type": "volatility_arb"
                }
        
        self.instance.metrics.signals_generated += 1
        
        return signal
    
    async def on_position_update(self, position: Dict[str, Any]) -> None:
        pass


class BandReversalStrategy(BaseStrategy):
    """波段反转策略 — 增强版：使用IndicatorEngine实时RSI、布林带、KDJ"""
    
    def __init__(self, instance: StrategyInstance, config: Dict[str, Any]):
        super().__init__(instance, config)
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.boll_band_width = config.get("boll_band_width", 2.0)
        self.kdj_confirmation = config.get("kdj_confirmation", True)
    
    async def on_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None
    
    async def on_bar(self, bar_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.is_running():
            return None
        
        close = bar_data.get("close", 0)
        if close <= 0:
            return None
        
        # 使用指标引擎获取RSI、布林带位置、KDJ
        indicators = self._get_indicators()
        
        rsi = indicators.get("rsi", 50) if indicators else 50
        boll_pos = indicators.get("bollinger_position", 0.5) if indicators else 0.5
        kdj_k = indicators.get("kdj_k", 50) if indicators else 50
        kdj_j = indicators.get("kdj_j", 50) if indicators else 50
        
        signal = None
        
        # 超卖反弹：RSI超卖 + 布林下轨 + KDJ金叉确认
        if rsi < self.rsi_oversold and boll_pos < 0.15:
            strength = min((self.rsi_oversold - rsi) / self.rsi_oversold + (0.15 - boll_pos) * 2, 1.0)
            reason = f"RSI超卖反弹 (RSI={rsi:.1f} BOLL={boll_pos:.2f}"
            if kdj_j < 20 and self.kdj_confirmation:
                strength += 0.1
                reason += f" KDJ={kdj_j:.1f}"
            reason += ")"
            signal = {
                "type": "open_long",
                "symbol": self.instance.symbol,
                "price": close,
                "reason": reason,
                "strength": min(strength, 1.0),
                "strategy_type": "band_reversal"
            }
        # 超买反转：RSI超买 + 布林上轨 + KDJ死叉确认
        elif rsi > self.rsi_overbought and boll_pos > 0.85:
            strength = min((rsi - self.rsi_overbought) / (100 - self.rsi_overbought) + (boll_pos - 0.85) * 2, 1.0)
            reason = f"RSI超买反转 (RSI={rsi:.1f} BOLL={boll_pos:.2f}"
            if kdj_j > 80 and self.kdj_confirmation:
                strength += 0.1
                reason += f" KDJ={kdj_j:.1f}"
            reason += ")"
            signal = {
                "type": "open_short",
                "symbol": self.instance.symbol,
                "price": close,
                "reason": reason,
                "strength": min(strength, 1.0),
                "strategy_type": "band_reversal"
            }
        
        self.instance.metrics.signals_generated += 1
        
        return signal
    
    async def on_position_update(self, position: Dict[str, Any]) -> None:
        self.instance.position_side = position.get("side") if position.get("size", 0) != 0 else None


class StrategyContainer:
    """
    多策略容器调度器
    
    管理：
    - 多交易对独立策略实例
    - 策略生命周期（创建、启动、暂停、停止）
    - 资源隔离和监控
    """
    
    STRATEGY_CLASSES: Dict[StrategyType, Type[BaseStrategy]] = {
        StrategyType.TREND_BREAKOUT: TrendBreakoutStrategy,
        StrategyType.GRID_OSCILLATION: GridOscillationStrategy,
        StrategyType.VOLATILITY_ARB: VolatilityArbStrategy,
        StrategyType.BAND_REVERSAL: BandReversalStrategy,
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        self._instances: Dict[str, StrategyInstance] = {}
        self._strategies: Dict[str, BaseStrategy] = {}
        self._lock = threading.RLock()

        self._max_instances_per_symbol = self.config.get("max_instances_per_symbol", 4)
        self._compute_interval_ms = self.config.get("compute_interval_ms", 100)

        # P21: 单策略超时保护 - 防止单策略计算阻塞整个系统
        self._strategy_timeouts: Dict[str, float] = {
            "grid": self.config.get("strategy_timeout_grid", 5.0),
            "trend": self.config.get("strategy_timeout_trend", 5.0),
            "scalping": self.config.get("strategy_timeout_scalping", 2.0),
            "arbitrage": self.config.get("strategy_timeout_arbitrage", 3.0),
            "default": self.config.get("strategy_timeout_default", 5.0),
        }
        self._strategy_timeout_count: Dict[str, int] = {}  # 超时计数

        self._signal_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []

        self._running = False
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # 算力动态调度：per-symbol 处理时间戳，控制策略处理频率
        self._compute_scheduler = None  # ComputeScheduler 引用（可选）
        self._indicator_engine = None   # IndicatorEngine 引用（可选，set_indicator_engine 注入）
        self._last_process_ts: Dict[str, float] = {}  # per-symbol 上次处理时间戳

        # P22: 状态无漂移 - 交易所仓位提供者，每次 tick/bar 前拉取真实仓位
        self._position_provider: Optional[Callable[[], Optional[List[Dict[str, Any]]]]] = None
        self._position_verify_interval: float = 2.0  # 仓位校验间隔（秒），避免频繁API调用
        self._last_position_verify_ts: float = 0.0
        self._last_exchange_positions: Optional[List[Dict[str, Any]]] = None

    def set_compute_scheduler(self, scheduler) -> None:
        """注入算力调度器，激活 per-symbol 处理频率动态调整"""
        self._compute_scheduler = scheduler
        logger.info("ComputeScheduler injected into StrategyContainer for dynamic frequency control")

    def set_position_provider(self, provider: Callable[[], Optional[List[Dict[str, Any]]]]) -> None:
        """P22: 注入交易所仓位提供者，用于状态无漂移校验"""
        self._position_provider = provider
        logger.info("PositionProvider injected into StrategyContainer for state drift prevention")

    def _get_exchange_positions(self) -> Optional[List[Dict[str, Any]]]:
        """P22: 获取交易所真实仓位（带缓存，避免频繁API调用）"""
        if self._position_provider is None:
            return None
        now = time.time()
        # 缓存有效期内直接返回缓存的仓位
        if self._last_exchange_positions is not None and \
           (now - self._last_position_verify_ts) < self._position_verify_interval:
            return self._last_exchange_positions
        try:
            positions = self._position_provider()
            if positions is not None:
                self._last_exchange_positions = positions
                self._last_position_verify_ts = now
            return positions
        except Exception as e:
            logger.debug(f"P22: Failed to get exchange positions: {e}")
            return self._last_exchange_positions  # 返回缓存

    async def _verify_strategy_positions(self, strategy: BaseStrategy, inst_id: str) -> None:
        """P22: 校验策略本地仓位与交易所仓位一致性，漂移时自动修正"""
        try:
            result = await strategy.verify_exchange_positions()
            if result.get("drifted"):
                corrections = result.get("corrections", [])
                logger.warning(
                    f"P22: Position drift detected for {inst_id}: "
                    f"{len(corrections)} corrections applied - {corrections}"
                )
                # 记录漂移事件到实例指标
                with self._lock:
                    if inst_id in self._instances:
                        self._instances[inst_id].metrics.error_count += 1
                        self._instances[inst_id].last_error = f"position_drift_{len(corrections)}"
                        self._instances[inst_id].last_error_at = datetime.now()
        except Exception as e:
            logger.debug(f"P22: Position verification error for {inst_id}: {e}")

    def _get_strategy_timeout(self, strategy_type: str) -> float:
        """P21: 获取策略超时时间，按策略类型区分"""
        for key in self._strategy_timeouts:
            if key in strategy_type.lower():
                return self._strategy_timeouts[key]
        return self._strategy_timeouts["default"]

    def _record_strategy_timeout(self, inst_id: str) -> None:
        """P21: 记录策略超时，超时过多时自动冻结实例"""
        count = self._strategy_timeout_count.get(inst_id, 0) + 1
        self._strategy_timeout_count[inst_id] = count
        if count >= 5:
            logger.critical(
                f"P21: Strategy {inst_id} timed out {count} times, "
                f"auto-freezing instance to prevent system blocking"
            )
            self.freeze_instance(inst_id, f"timeout_{count}_times")

    def set_indicator_engine(self, engine) -> None:
        """注入指标引擎到所有已创建及未来策略实例"""
        self._indicator_engine = engine
        with self._lock:
            for strat in self._strategies.values():
                try:
                    strat.set_indicator_engine(engine)
                except Exception:
                    pass
        logger.info("IndicatorEngine injected into StrategyContainer strategies")
    
    def create_instance(self, strategy_type: StrategyType, symbol: str,
                        config: Dict[str, Any] = None) -> StrategyInstance:
        """
        创建策略实例
        
        Args:
            strategy_type: 策略类型
            symbol: 交易对
            config: 策略配置
        
        Returns:
            StrategyInstance: 策略实例
        """
        instance_id = f"{strategy_type.value}_{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        instance = StrategyInstance(
            id=instance_id,
            strategy_type=strategy_type,
            symbol=symbol,
            config=config or {}
        )
        
        strategy_class = self.STRATEGY_CLASSES.get(strategy_type)
        if not strategy_class:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        strategy = strategy_class(instance, instance.config)
        
        # 注入指标引擎（如果已设置）
        if hasattr(self, '_indicator_engine') and self._indicator_engine:
            strategy.set_indicator_engine(self._indicator_engine)
        
        with self._lock:
            self._instances[instance_id] = instance
            self._strategies[instance_id] = strategy
        
        logger.info(f"Created strategy instance: {instance_id}")
        
        return instance
    
    def start_instance(self, instance_id: str) -> bool:
        """启动策略实例"""
        with self._lock:
            if instance_id not in self._strategies:
                return False
            
            strategy = self._strategies[instance_id]
            instance = self._instances[instance_id]
            
            try:
                strategy.start()
                logger.info(f"Started strategy instance: {instance_id}")
                return True
            except Exception as e:
                instance.last_error = str(e)
                instance.last_error_at = datetime.now()
                instance.state = StrategyState.ERROR
                logger.error(f"Failed to start strategy {instance_id}: {e}")
                return False
    
    def stop_instance(self, instance_id: str) -> bool:
        """停止策略实例"""
        with self._lock:
            if instance_id not in self._strategies:
                return False
            
            strategy = self._strategies[instance_id]
            strategy.stop()
            
            logger.info(f"Stopped strategy instance: {instance_id}")
            return True
    
    def pause_instance(self, instance_id: str) -> bool:
        """暂停策略实例"""
        with self._lock:
            if instance_id not in self._strategies:
                return False
            
            self._strategies[instance_id].pause()
            return True
    
    def resume_instance(self, instance_id: str) -> bool:
        """恢复策略实例"""
        with self._lock:
            if instance_id not in self._strategies:
                return False
            
            self._strategies[instance_id].resume()
            return True
    
    def freeze_instance(self, instance_id: str, reason: str) -> bool:
        """冻结策略实例（失效自动冻结）"""
        with self._lock:
            if instance_id not in self._instances:
                return False
            
            self._instances[instance_id].freeze(reason)
            logger.warning(f"Frozen strategy instance {instance_id}: {reason}")
            return True
    
    def unfreeze_instance(self, instance_id: str) -> bool:
        """解冻策略实例"""
        with self._lock:
            if instance_id not in self._instances:
                return False
            
            self._instances[instance_id].unfreeze()
            return True
    
    async def process_tick(self, symbol: str, tick_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理tick数据，分发到所有相关策略实例

        算力调度：根据 ComputeScheduler 动态调整处理频率（高波动50ms，低波动500ms）
        距上次处理不足间隔时跳过，降低本地设备7×24小时负载

        Returns:
            信号列表
        """
        # 算力调度闸门：距上次处理不足间隔则跳过
        if self._compute_scheduler is not None:
            try:
                interval_ms = self._compute_scheduler.get_compute_interval_ms(symbol)
                now_ts = time.time()
                last_ts = self._last_process_ts.get(symbol, 0.0)
                if (now_ts - last_ts) * 1000 < interval_ms:
                    return []  # 未到调度间隔，跳过本次
                self._last_process_ts[symbol] = now_ts
            except Exception as e:
                logger.debug(f"ComputeScheduler interval check error for {symbol}: {e}")

        signals = []

        with self._lock:
            instances_to_process = [
                (inst_id, inst, self._strategies[inst_id])
                for inst_id, inst in self._instances.items()
                if inst.symbol == symbol and inst.is_tradable()
            ]
        
        # P22: 状态无漂移 - 每tick前校验策略本地仓位与交易所仓位一致性
        exchange_positions = self._get_exchange_positions()
        if exchange_positions is not None:
            for inst_id, instance, strategy in instances_to_process:
                await self._verify_strategy_positions(strategy, inst_id)
        
        for inst_id, instance, strategy in instances_to_process:
            try:
                start_time = time.time()
                
                # P21: 单策略超时保护 - 防止单策略计算阻塞整个系统
                strategy_type = (instance.strategy_type.value 
                                 if hasattr(instance, 'strategy_type') and instance.strategy_type 
                                 else "default")
                timeout = self._get_strategy_timeout(strategy_type)
                
                signal = await asyncio.wait_for(
                    strategy.on_tick(tick_data),
                    timeout=timeout
                )
                
                if signal:
                    signals.append(signal)
                    instance.last_signal_at = datetime.now()
                
                elapsed_ms = (time.time() - start_time) * 1000
                instance.metrics.compute_time_ms += elapsed_ms
                instance.metrics.compute_count += 1
                
            except asyncio.TimeoutError:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.warning(
                    f"P21: Strategy {inst_id} ({strategy_type}) on_tick timeout "
                    f"after {elapsed_ms:.0f}ms (limit={timeout}s), signal discarded"
                )
                self._record_strategy_timeout(inst_id)
                instance.last_error = f"timeout_{elapsed_ms:.0f}ms"
                instance.last_error_at = datetime.now()
            except Exception as e:
                logger.error(f"Error in strategy {inst_id} processing tick: {e}")
                instance.last_error = str(e)
                instance.last_error_at = datetime.now()
        
        if signals:
            self._dispatch_signals(symbol, signals)
        
        return signals
    
    async def process_bar(self, symbol: str, bar_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理K线数据，分发到所有相关策略实例

        算力调度：与 process_tick 共享 per-symbol 调度间隔

        Returns:
            信号列表
        """
        # 算力调度闸门：距上次处理不足间隔则跳过
        if self._compute_scheduler is not None:
            try:
                interval_ms = self._compute_scheduler.get_compute_interval_ms(symbol)
                now_ts = time.time()
                last_ts = self._last_process_ts.get(symbol, 0.0)
                if (now_ts - last_ts) * 1000 < interval_ms:
                    return []
                self._last_process_ts[symbol] = now_ts
            except Exception as e:
                logger.debug(f"ComputeScheduler interval check error for {symbol}: {e}")

        signals = []

        with self._lock:
            instances_to_process = [
                (inst_id, inst, self._strategies[inst_id])
                for inst_id, inst in self._instances.items()
                if inst.symbol == symbol and inst.is_tradable()
            ]
        
        # P22: 状态无漂移 - 每根K线前校验策略本地仓位与交易所仓位一致性
        exchange_positions = self._get_exchange_positions()
        if exchange_positions is not None:
            for inst_id, instance, strategy in instances_to_process:
                await self._verify_strategy_positions(strategy, inst_id)
        
        for inst_id, instance, strategy in instances_to_process:
            try:
                start_time = time.time()
                
                # P21: 单策略超时保护 - 防止单策略计算阻塞整个系统
                strategy_type = (instance.strategy_type.value 
                                 if hasattr(instance, 'strategy_type') and instance.strategy_type 
                                 else "default")
                timeout = self._get_strategy_timeout(strategy_type)
                
                signal = await asyncio.wait_for(
                    strategy.on_bar(bar_data),
                    timeout=timeout
                )
                
                if signal:
                    signals.append(signal)
                    instance.last_signal_at = datetime.now()
                
                elapsed_ms = (time.time() - start_time) * 1000
                instance.metrics.compute_time_ms += elapsed_ms
                instance.metrics.compute_count += 1
                
            except asyncio.TimeoutError:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.warning(
                    f"P21: Strategy {inst_id} ({strategy_type}) on_bar timeout "
                    f"after {elapsed_ms:.0f}ms (limit={timeout}s), signal discarded"
                )
                self._record_strategy_timeout(inst_id)
                instance.last_error = f"timeout_{elapsed_ms:.0f}ms"
                instance.last_error_at = datetime.now()
            except Exception as e:
                logger.error(f"Error in strategy {inst_id} processing bar: {e}")
                instance.last_error = str(e)
                instance.last_error_at = datetime.now()
        
        if signals:
            self._dispatch_signals(symbol, signals)
        
        return signals
    
    def update_position(self, symbol: str, position: Dict[str, Any]) -> None:
        """更新仓位信息到所有相关策略实例"""
        with self._lock:
            for inst_id, instance in self._instances.items():
                if instance.symbol == symbol and inst_id in self._strategies:
                    try:
                        task = asyncio.create_task(
                            self._strategies[inst_id].on_position_update(position)
                        )
                        task.add_done_callback(
                            lambda t, iid=inst_id: logger.debug(
                                f"Position update task for {iid} "
                                f"{'failed' if t.exception() else 'completed'}"
                            ) if t.exception() else None
                        )
                    except RuntimeError:
                        # 无事件循环（同步环境），同步执行
                        try:
                            import asyncio as _asyncio
                            loop = _asyncio.new_event_loop()
                            loop.run_until_complete(
                                self._strategies[inst_id].on_position_update(position)
                            )
                            loop.close()
                        except Exception:
                            pass
                    except Exception as e:
                        logger.error(f"Error updating position for {inst_id}: {e}")
    
    def register_signal_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """注册信号回调"""
        self._signal_callbacks.append(callback)
    
    def _dispatch_signals(self, symbol: str, signals: List[Dict[str, Any]]) -> None:
        """分发信号"""
        for signal in signals:
            for callback in self._signal_callbacks:
                try:
                    callback(symbol, signal)
                except Exception as e:
                    logger.error(f"Error in signal callback: {e}")
    
    def get_instance(self, instance_id: str) -> Optional[StrategyInstance]:
        """获取策略实例"""
        with self._lock:
            return self._instances.get(instance_id)
    
    def get_instances_by_symbol(self, symbol: str) -> List[StrategyInstance]:
        """获取指定交易对的所有策略实例"""
        with self._lock:
            return [inst for inst in self._instances.values() if inst.symbol == symbol]
    
    def get_instances_by_type(self, strategy_type: StrategyType) -> List[StrategyInstance]:
        """获取指定类型的所有策略实例"""
        with self._lock:
            return [inst for inst in self._instances.values() 
                    if inst.strategy_type == strategy_type]
    
    def get_all_instances(self) -> List[StrategyInstance]:
        """获取所有策略实例"""
        with self._lock:
            return list(self._instances.values())
    
    def get_active_instances(self) -> List[StrategyInstance]:
        """获取所有活跃的策略实例"""
        with self._lock:
            return [inst for inst in self._instances.values() if inst.is_active()]
    
    def get_frozen_instances(self) -> List[StrategyInstance]:
        """获取所有冻结的策略实例"""
        with self._lock:
            return [inst for inst in self._instances.values() 
                    if inst.state == StrategyState.FROZEN]
    
    def get_aggregate_metrics(self) -> Dict[str, Any]:
        """获取聚合指标"""
        total_metrics = StrategyMetrics()
        
        with self._lock:
            for instance in self._instances.values():
                m = instance.metrics
                total_metrics.total_trades += m.total_trades
                total_metrics.winning_trades += m.winning_trades
                total_metrics.losing_trades += m.losing_trades
                total_metrics.total_pnl += m.total_pnl
                total_metrics.signals_generated += m.signals_generated
                total_metrics.signals_executed += m.signals_executed
                total_metrics.compute_time_ms += m.compute_time_ms
                total_metrics.compute_count += m.compute_count
            
            if total_metrics.winning_trades + total_metrics.losing_trades > 0:
                total_metrics.win_rate = total_metrics.winning_trades / (
                    total_metrics.winning_trades + total_metrics.losing_trades)
            
            if total_metrics.signals_generated > 0:
                total_metrics.execution_rate = total_metrics.signals_executed / total_metrics.signals_generated
        
        return {
            "instances_count": len(self._instances),
            "active_count": len(self.get_active_instances()),
            "frozen_count": len(self.get_frozen_instances()),
            "metrics": total_metrics.to_dict()
        }
    
    def remove_instance(self, instance_id: str) -> bool:
        """移除策略实例"""
        with self._lock:
            if instance_id in self._strategies:
                self._strategies[instance_id].stop()
            
            if instance_id in self._instances:
                del self._instances[instance_id]
            
            if instance_id in self._strategies:
                del self._strategies[instance_id]
            
            logger.info(f"Removed strategy instance: {instance_id}")
            return True
    
    def clear_all(self) -> None:
        """清除所有策略实例"""
        with self._lock:
            for strategy in self._strategies.values():
                try:
                    strategy.stop()
                except Exception:
                    pass
            
            self._instances.clear()
            self._strategies.clear()
            
            logger.info("Cleared all strategy instances")


_container_instance: Optional[StrategyContainer] = None

def get_strategy_container(config: Dict[str, Any] = None) -> StrategyContainer:
    """获取策略容器单例"""
    global _container_instance
    if _container_instance is None:
        _container_instance = StrategyContainer(config)
    return _container_instance