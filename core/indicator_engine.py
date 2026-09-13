"""
指标高速计算内核
================
核心定位：高波动捕捉、多策略并行运算，独立算力隔离，策略互不干扰

支持的指标：
- RSI (相对强弱指标)
- MACD (异同移动平均线)
- 布林带 (Bollinger Bands)
- ATR (平均真实波幅)
- 波动率分位数
- 资金流指标 (MFI)
- 基差套利指标
- ADX (平均趋向指标)
- KDJ 随机指标
- 成交量加权平均价 (VWAP)
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
import threading
from loguru import logger
from functools import lru_cache
import time

if TYPE_CHECKING:
    from core.compute_scheduler import ComputeScheduler


@dataclass
class IndicatorResult:
    """指标计算结果"""
    name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class IndicatorSet:
    """指标集合，用于单次计算返回多个指标"""
    symbol: str
    timestamp: datetime
    indicators: Dict[str, IndicatorResult] = field(default_factory=dict)
    mtf_indicators: Dict[str, Dict[str, IndicatorResult]] = field(default_factory=dict)  # {"5m": {...}, "15m": {...}}
    
    def get(self, name: str, default: float = 0.0) -> float:
        if name in self.indicators:
            return self.indicators[name].value
        return default
    
    def get_result(self, name: str) -> Optional[IndicatorResult]:
        return self.indicators.get(name)
    
    def get_mtf(self, timeframe: str, name: str, default: float = 0.0) -> float:
        """获取多时间周期指标值"""
        tf_data = self.mtf_indicators.get(timeframe, {})
        result = tf_data.get(name)
        return result.value if result else default


class CircularBuffer:
    """环形缓冲区，用于高效存储历史数据"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = np.zeros(capacity, dtype=np.float64)
        self.index = 0
        self.size = 0
        # 使用 RLock：get_latest 在持锁时会调用 get_ordered（嵌套加锁），
        # 非重入 Lock 会导致死锁
        self._lock = threading.RLock()
    
    def push(self, value: float) -> None:
        with self._lock:
            self.buffer[self.index] = value
            self.index = (self.index + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1
    
    def push_batch(self, values: np.ndarray) -> None:
        with self._lock:
            n = len(values)
            if n >= self.capacity:
                self.buffer[:] = values[-self.capacity:]
                self.index = 0
                self.size = self.capacity
            else:
                remaining = self.capacity - self.index
                if n <= remaining:
                    self.buffer[self.index:self.index + n] = values
                else:
                    self.buffer[self.index:] = values[:remaining]
                    self.buffer[:n - remaining] = values[remaining:]
                self.index = (self.index + n) % self.capacity
                self.size = min(self.size + n, self.capacity)
    
    def get_ordered(self) -> np.ndarray:
        with self._lock:
            if self.size < self.capacity:
                return self.buffer[:self.size].copy()
            return np.concatenate([self.buffer[self.index:], self.buffer[:self.index]])
    
    def get_latest(self, n: int = 1) -> np.ndarray:
        with self._lock:
            if n >= self.size:
                return self.get_ordered()
            start = (self.index - n) % self.capacity
            if start < self.index:
                return self.buffer[start:self.index].copy()
            else:
                return np.concatenate([self.buffer[start:], self.buffer[:self.index]])
    
    @property
    def is_full(self) -> bool:
        return self.size >= self.capacity


class SymbolIndicatorCache:
    """单交易对指标缓存，存储历史价格和预计算数据"""
    
    def __init__(self, symbol: str, max_history: int = 500):
        self.symbol = symbol
        self.max_history = max_history
        
        self.prices = CircularBuffer(max_history)
        self.highs = CircularBuffer(max_history)
        self.lows = CircularBuffer(max_history)
        self.volumes = CircularBuffer(max_history)
        self.closes = CircularBuffer(max_history)
        
        # Multi-timeframe buffers: {"5m": {"closes": CircularBuffer, ...}, "15m": {...}}
        self.mtf_buffers: Dict[str, Dict[str, CircularBuffer]] = {}
        
        self.last_update = datetime.min
        self.indicator_cache: Dict[str, Any] = {}
        self._lock = threading.RLock()
    
    def register_timeframe(self, timeframe: str, max_history: int = None) -> None:
        """注册一个多时间周期数据缓冲区"""
        capacity = max_history or self.max_history
        with self._lock:
            if timeframe not in self.mtf_buffers:
                self.mtf_buffers[timeframe] = {
                    "prices": CircularBuffer(capacity),
                    "highs": CircularBuffer(capacity),
                    "lows": CircularBuffer(capacity),
                    "volumes": CircularBuffer(capacity),
                    "closes": CircularBuffer(capacity),
                }
    
    def update_mtf(self, timeframe: str, open_price: float, high: float,
                   low: float, close: float, volume: float) -> None:
        """更新指定时间周期的K线数据"""
        if timeframe not in self.mtf_buffers:
            self.register_timeframe(timeframe)
        buffers = self.mtf_buffers[timeframe]
        buffers["prices"].push(open_price)
        buffers["highs"].push(high)
        buffers["lows"].push(low)
        buffers["closes"].push(close)
        buffers["volumes"].push(volume)
    
    def get_mtf_closes(self, timeframe: str) -> np.ndarray:
        """获取指定时间周期的收盘价序列"""
        if timeframe not in self.mtf_buffers:
            return np.array([])
        return self.mtf_buffers[timeframe]["closes"].get_ordered()
    
    def get_mtf_highs(self, timeframe: str) -> np.ndarray:
        if timeframe not in self.mtf_buffers:
            return np.array([])
        return self.mtf_buffers[timeframe]["highs"].get_ordered()
    
    def get_mtf_lows(self, timeframe: str) -> np.ndarray:
        if timeframe not in self.mtf_buffers:
            return np.array([])
        return self.mtf_buffers[timeframe]["lows"].get_ordered()
    
    def get_mtf_volumes(self, timeframe: str) -> np.ndarray:
        if timeframe not in self.mtf_buffers:
            return np.array([])
        return self.mtf_buffers[timeframe]["volumes"].get_ordered()
    
    def update(self, open_price: float, high: float, low: float, 
               close: float, volume: float) -> None:
        with self._lock:
            self.prices.push(open_price)
            self.highs.push(high)
            self.lows.push(low)
            self.closes.push(close)
            self.volumes.push(volume)
            self.last_update = datetime.now()
            self.indicator_cache.clear()
    
    def update_batch(self, opens: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                     closes: np.ndarray, volumes: np.ndarray) -> None:
        with self._lock:
            for o, h, l, c, v in zip(opens, highs, lows, closes, volumes):
                self.prices.push(o)
                self.highs.push(h)
                self.lows.push(l)
                self.closes.push(c)
                self.volumes.push(v)
            self.last_update = datetime.now()
            self.indicator_cache.clear()
    
    def get_closes(self) -> np.ndarray:
        return self.closes.get_ordered()
    
    def get_highs(self) -> np.ndarray:
        return self.highs.get_ordered()
    
    def get_lows(self) -> np.ndarray:
        return self.lows.get_ordered()
    
    def get_volumes(self) -> np.ndarray:
        return self.volumes.get_ordered()
    
    def get_tr(self) -> np.ndarray:
        """计算真实波幅 (True Range)"""
        closes = self.get_closes()
        highs = self.get_highs()
        lows = self.get_lows()
        
        if len(closes) < 2:
            return np.array([highs[0] - lows[0]]) if len(highs) > 0 else np.array([0.0])
        
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )
        return np.concatenate([[highs[0] - lows[0]], tr])


class IndicatorEngine:
    """
    指标高速计算引擎
    
    特性：
    - 多交易对并行计算，每个交易对独立缓存
    - 增量更新，避免全量重算
    - 线程安全，支持并发访问
    - 支持指标依赖链（如MACD依赖EMA）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self._symbol_caches: Dict[str, SymbolIndicatorCache] = {}
        self._cache_lock = threading.RLock()
        
        self._default_period = self.config.get("indicator_period", 14)
        self._atr_period = self.config.get("atr_period", 14)
        self._rsi_period = self.config.get("rsi_period", 14)
        self._macd_fast = self.config.get("macd_fast_period", 12)
        self._macd_slow = self.config.get("macd_slow_period", 26)
        self._macd_signal = self.config.get("macd_signal_period", 9)
        self._bollinger_period = self.config.get("bollinger_period", 20)
        self._bollinger_std = self.config.get("bollinger_std", 2.0)
        
        self._max_history = self.config.get("max_indicator_history", 500)
        
        self._compute_times: Dict[str, float] = {}
        self._compute_count: Dict[str, int] = {}
        
        # 多时间周期配置
        mtf_config = self.config.get("multi_timeframe", {})
        self._mtf_enabled = mtf_config.get("enabled", True)
        self._mtf_timeframes = mtf_config.get("timeframes", ["5m", "15m", "1h"])

        # 算力动态调度：低波动/CPU过载时跳过计算，返回上次结果
        self._compute_scheduler: Optional["ComputeScheduler"] = None
        self._last_indicator_sets: Dict[str, "IndicatorSet"] = {}  # per-symbol 上次计算结果缓存

    def set_compute_scheduler(self, scheduler: "ComputeScheduler") -> None:
        """注入算力调度器（由 TradingScheduler 在初始化时注入）"""
        self._compute_scheduler = scheduler

    def register_symbol(self, symbol: str) -> None:
        """注册交易对，初始化缓存"""
        with self._cache_lock:
            if symbol not in self._symbol_caches:
                self._symbol_caches[symbol] = SymbolIndicatorCache(symbol, self._max_history)
                logger.debug(f"Registered symbol for indicator calculation: {symbol}")
        # 同步注册到算力调度器
        if self._compute_scheduler is not None:
            try:
                self._compute_scheduler.register_symbol(symbol)
            except Exception as e:
                logger.debug(f"ComputeScheduler register_symbol error for {symbol}: {e}")
    
    def update_market_data(self, symbol: str, open_price: float, high: float,
                           low: float, close: float, volume: float) -> None:
        """更新市场数据，推送新K线"""
        if symbol not in self._symbol_caches:
            self.register_symbol(symbol)
        
        self._symbol_caches[symbol].update(open_price, high, low, close, volume)
    
    def update_batch(self, symbol: str, bars: List[Dict[str, Any]]) -> None:
        """批量更新历史数据"""
        if not bars:
            return
        
        if symbol not in self._symbol_caches:
            self.register_symbol(symbol)
        
        opens = np.array([b.get("open", b.get("o", 0)) for b in bars])
        highs = np.array([b.get("high", b.get("h", 0)) for b in bars])
        lows = np.array([b.get("low", b.get("l", 0)) for b in bars])
        closes = np.array([b.get("close", b.get("c", 0)) for b in bars])
        volumes = np.array([b.get("volume", b.get("v", 0)) for b in bars])
        
        self._symbol_caches[symbol].update_batch(opens, highs, lows, closes, volumes)
    
    def calculate_all(self, symbol: str) -> IndicatorSet:
        """
        计算所有指标，返回指标集合
        包括：RSI, MACD, 布林带, ATR, ADX, 波动率分位数

        算力调度：低波动/CPU过载时跳过本次计算，返回上次缓存结果（降低7×24小时本地设备负载）
        """
        # 算力调度闸门：低波动symbol跳过本次计算
        if self._compute_scheduler is not None:
            try:
                if not self._compute_scheduler.should_compute(symbol):
                    # 跳过本次计算，返回上次缓存结果
                    cached = self._last_indicator_sets.get(symbol)
                    if cached is not None:
                        return cached
                    # 首次无缓存时仍执行一次计算以建立基线
            except Exception as e:
                logger.debug(f"ComputeScheduler should_compute error for {symbol}: {e}")

        start_time = time.time()

        if symbol not in self._symbol_caches:
            return IndicatorSet(symbol=symbol, timestamp=datetime.now())
        
        cache = self._symbol_caches[symbol]
        timestamp = datetime.now()
        
        indicators = {}
        
        try:
            rsi = self.calculate_rsi(symbol)
            if rsi is not None:
                indicators["rsi"] = IndicatorResult("rsi", rsi, timestamp)
                indicators["rsi_overbought"] = IndicatorResult("rsi_overbought", 1.0 if rsi > 70 else 0.0, timestamp)
                indicators["rsi_oversold"] = IndicatorResult("rsi_oversold", 1.0 if rsi < 30 else 0.0, timestamp)
        except Exception as e:
            logger.warning(f"RSI calculation error for {symbol}: {e}")
        
        try:
            macd, signal, hist = self.calculate_macd(symbol)
            if macd is not None:
                indicators["macd"] = IndicatorResult("macd", macd, timestamp, 
                                                     {"signal": signal, "histogram": hist})
                indicators["macd_signal"] = IndicatorResult("macd_signal", signal, timestamp)
                indicators["macd_histogram"] = IndicatorResult("macd_histogram", hist, timestamp)
                indicators["macd_cross_up"] = IndicatorResult("macd_cross_up", 
                    1.0 if hist > 0 and len(cache.get_closes()) > 1 else 0.0, timestamp)
                indicators["macd_cross_down"] = IndicatorResult("macd_cross_down",
                    1.0 if hist < 0 and len(cache.get_closes()) > 1 else 0.0, timestamp)
        except Exception as e:
            logger.warning(f"MACD calculation error for {symbol}: {e}")
        
        try:
            upper, middle, lower = self.calculate_bollinger(symbol)
            if upper is not None:
                closes = cache.get_closes()
                current_close = closes[-1] if len(closes) > 0 else 0
                indicators["bollinger_upper"] = IndicatorResult("bollinger_upper", upper, timestamp)
                indicators["bollinger_middle"] = IndicatorResult("bollinger_middle", middle, timestamp)
                indicators["bollinger_lower"] = IndicatorResult("bollinger_lower", lower, timestamp)
                indicators["bollinger_width"] = IndicatorResult("bollinger_width", 
                    (upper - lower) / middle if middle > 0 else 0, timestamp)
                indicators["bollinger_position"] = IndicatorResult("bollinger_position",
                    (current_close - lower) / (upper - lower) if upper > lower else 0.5, timestamp)
        except Exception as e:
            logger.warning(f"Bollinger calculation error for {symbol}: {e}")
        
        try:
            atr = self.calculate_atr(symbol)
            if atr is not None:
                indicators["atr"] = IndicatorResult("atr", atr, timestamp)
                closes = cache.get_closes()
                if len(closes) > 0:
                    atr_pct = atr / closes[-1] if closes[-1] > 0 else 0
                    indicators["atr_percent"] = IndicatorResult("atr_percent", atr_pct, timestamp)
        except Exception as e:
            logger.warning(f"ATR calculation error for {symbol}: {e}")
        
        try:
            adx, plus_di, minus_di = self.calculate_adx(symbol)
            if adx is not None:
                indicators["adx"] = IndicatorResult("adx", adx, timestamp,
                                                    {"plus_di": plus_di, "minus_di": minus_di})
                indicators["plus_di"] = IndicatorResult("plus_di", plus_di, timestamp)
                indicators["minus_di"] = IndicatorResult("minus_di", minus_di, timestamp)
                indicators["trend_strength"] = IndicatorResult("trend_strength",
                    1.0 if adx > 25 else 0.0, timestamp)
                indicators["strong_trend"] = IndicatorResult("strong_trend",
                    1.0 if adx > 50 else 0.0, timestamp)
        except Exception as e:
            logger.warning(f"ADX calculation error for {symbol}: {e}")
        
        try:
            vol_pct = self.calculate_volatility_percentile(symbol)
            if vol_pct is not None:
                indicators["volatility_percentile"] = IndicatorResult(
                    "volatility_percentile", vol_pct, timestamp)
                indicators["high_volatility"] = IndicatorResult("high_volatility",
                    1.0 if vol_pct > 0.8 else 0.0, timestamp)
                indicators["low_volatility"] = IndicatorResult("low_volatility",
                    1.0 if vol_pct < 0.2 else 0.0, timestamp)
        except Exception as e:
            logger.warning(f"Volatility percentile calculation error for {symbol}: {e}")
        
        try:
            mfi = self.calculate_mfi(symbol)
            if mfi is not None:
                indicators["mfi"] = IndicatorResult("mfi", mfi, timestamp)
                indicators["mfi_overbought"] = IndicatorResult("mfi_overbought",
                    1.0 if mfi > 80 else 0.0, timestamp)
                indicators["mfi_oversold"] = IndicatorResult("mfi_oversold",
                    1.0 if mfi < 20 else 0.0, timestamp)
        except Exception as e:
            logger.warning(f"MFI calculation error for {symbol}: {e}")
        
        try:
            kdj_k, kdj_d, kdj_j = self.calculate_kdj(symbol)
            if kdj_k is not None:
                indicators["kdj_k"] = IndicatorResult("kdj_k", kdj_k, timestamp)
                indicators["kdj_d"] = IndicatorResult("kdj_d", kdj_d, timestamp)
                indicators["kdj_j"] = IndicatorResult("kdj_j", kdj_j, timestamp)
        except Exception as e:
            logger.warning(f"KDJ calculation error for {symbol}: {e}")
        
        try:
            vwap = self.calculate_vwap(symbol)
            if vwap is not None:
                closes = cache.get_closes()
                current_close = closes[-1] if len(closes) > 0 else 0
                indicators["vwap"] = IndicatorResult("vwap", vwap, timestamp)
                indicators["vwap_deviation"] = IndicatorResult("vwap_deviation",
                    (current_close - vwap) / vwap if vwap > 0 else 0, timestamp)
        except Exception as e:
            logger.warning(f"VWAP calculation error for {symbol}: {e}")
        
        elapsed = time.time() - start_time
        self._compute_times[symbol] = self._compute_times.get(symbol, 0) + elapsed
        self._compute_count[symbol] = self._compute_count.get(symbol, 0) + 1

        result = IndicatorSet(symbol=symbol, timestamp=timestamp, indicators=indicators)

        # 缓存本次结果，供低波动跳过时返回
        self._last_indicator_sets[symbol] = result

        # 反馈波动率分位数给算力调度器，触发优先级动态调整
        if self._compute_scheduler is not None:
            try:
                vol_result = indicators.get("volatility_percentile")
                if vol_result is not None:
                    self._compute_scheduler.update_symbol_volatility(symbol, float(vol_result.value))
            except Exception as e:
                logger.debug(f"ComputeScheduler update_symbol_volatility error for {symbol}: {e}")

        return result
    
    def calculate_rsi(self, symbol: str, period: int = None) -> Optional[float]:
        """计算RSI (相对强弱指标)"""
        period = period or self._rsi_period
        
        if symbol not in self._symbol_caches:
            return None
        
        cache = self._symbol_caches[symbol]
        closes = cache.get_closes()
        
        if len(closes) < period + 1:
            return None
        
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_loss = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        if len(avg_gain) == 0 or len(avg_loss) == 0:
            return None
        
        avg_gain = avg_gain[-1]
        avg_loss = avg_loss[-1]
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return round(rsi, 4)
    
    def calculate_macd(self, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """计算MACD，返回 (macd_line, signal_line, histogram)"""
        if symbol not in self._symbol_caches:
            return None, None, None
        
        cache = self._symbol_caches[symbol]
        closes = cache.get_closes()
        
        if len(closes) < self._macd_slow + self._macd_signal:
            return None, None, None
        
        ema_fast = self._calculate_ema(closes, self._macd_fast)
        ema_slow = self._calculate_ema(closes, self._macd_slow)
        
        # 对齐两个 EMA 序列长度（快线周期更短 → 序列更长），
        # 否则 ema_fast - ema_slow 会因长度不一致触发 numpy 广播异常
        min_len = min(len(ema_fast), len(ema_slow))
        if min_len == 0:
            return None, None, None
        
        macd_line = ema_fast[-min_len:] - ema_slow[-min_len:]
        
        signal_line = self._calculate_ema_series(macd_line, self._macd_signal)
        
        if len(signal_line) == 0:
            return None, None, None
        
        signal_line = signal_line[-1]
        histogram = macd_line[-1] - signal_line
        
        return round(float(macd_line[-1]), 6), round(float(signal_line), 6), round(float(histogram), 6)
    
    def calculate_bollinger(self, symbol: str, period: int = None, 
                            std_dev: float = None) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """计算布林带，返回 (upper, middle, lower)"""
        period = period or self._bollinger_period
        std_dev = std_dev or self._bollinger_std
        
        if symbol not in self._symbol_caches:
            return None, None, None
        
        cache = self._symbol_caches[symbol]
        closes = cache.get_closes()
        
        if len(closes) < period:
            return None, None, None
        
        recent_closes = closes[-period:]
        middle = np.mean(recent_closes)
        std = np.std(recent_closes)
        
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        
        return round(upper, 6), round(middle, 6), round(lower, 6)
    
    def calculate_atr(self, symbol: str, period: int = None) -> Optional[float]:
        """计算ATR (平均真实波幅)"""
        period = period or self._atr_period
        
        if symbol not in self._symbol_caches:
            return None
        
        cache = self._symbol_caches[symbol]
        tr = cache.get_tr()
        
        if len(tr) < period:
            return None
        
        atr = np.mean(tr[-period:])
        return round(atr, 6)
    
    def calculate_adx(self, symbol: str, period: int = 14) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """计算ADX和DI指标"""
        if symbol not in self._symbol_caches:
            return None, None, None
        
        cache = self._symbol_caches[symbol]
        highs = cache.get_highs()
        lows = cache.get_lows()
        closes = cache.get_closes()
        
        if len(highs) < period + 1:
            return None, None, None
        
        plus_dm = np.zeros(len(highs) - 1)
        minus_dm = np.zeros(len(highs) - 1)
        tr = np.zeros(len(highs) - 1)
        
        for i in range(1, len(highs)):
            up_move = highs[i] - highs[i-1]
            down_move = lows[i-1] - lows[i]
            
            plus_dm[i-1] = up_move if up_move > down_move and up_move > 0 else 0
            minus_dm[i-1] = down_move if down_move > up_move and down_move > 0 else 0
            
            tr[i-1] = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i-1]),
                         abs(lows[i] - closes[i-1]))
        
        smooth_plus_dm = np.convolve(plus_dm, np.ones(period)/period, mode='valid')
        smooth_minus_dm = np.convolve(minus_dm, np.ones(period)/period, mode='valid')
        smooth_tr = np.convolve(tr, np.ones(period)/period, mode='valid')
        
        if len(smooth_plus_dm) == 0 or len(smooth_tr) == 0:
            return None, None, None
        
        # 逐点计算 +DI/-DI/DX（抑制除零警告），ADX 应为 DX 序列的移动平均
        # 而非最后一个点的 DX 值，否则 ADX 等价于 period=1，失去平滑意义
        with np.errstate(divide='ignore', invalid='ignore'):
            plus_di_series = 100 * smooth_plus_dm / smooth_tr
            minus_di_series = 100 * smooth_minus_dm / smooth_tr
            di_sum = plus_di_series + minus_di_series
            dx_series = 100 * np.abs(plus_di_series - minus_di_series) / di_sum
        
        plus_di_series = np.where(np.isfinite(plus_di_series), plus_di_series, 0.0)
        minus_di_series = np.where(np.isfinite(minus_di_series), minus_di_series, 0.0)
        dx_series = np.where(np.isfinite(dx_series), dx_series, 0.0)
        
        adx = float(np.mean(dx_series[-period:])) if len(dx_series) > 0 else 0.0
        plus_di = float(plus_di_series[-1])
        minus_di = float(minus_di_series[-1])
        
        return round(adx, 4), round(plus_di, 4), round(minus_di, 4)
    
    def calculate_volatility_percentile(self, symbol: str, lookback: int = 100) -> Optional[float]:
        """计算波动率分位数 (当前ATR在历史中的位置)"""
        if symbol not in self._symbol_caches:
            return None
        
        cache = self._symbol_caches[symbol]
        tr = cache.get_tr()
        
        if len(tr) < lookback:
            lookback = len(tr)
        
        if lookback < 5:
            return None
        
        recent_tr = tr[-lookback:]
        current_atr = np.mean(recent_tr[-14:]) if len(recent_tr) >= 14 else np.mean(recent_tr)
        
        historical_atrs = []
        for i in range(len(recent_tr) - 14):
            historical_atrs.append(np.mean(recent_tr[i:i+14]))
        
        if not historical_atrs:
            return None
        
        percentile = np.sum(np.array(historical_atrs) < current_atr) / len(historical_atrs)
        
        return round(percentile, 4)
    
    def calculate_mfi(self, symbol: str, period: int = 14) -> Optional[float]:
        """计算MFI (资金流量指标)"""
        if symbol not in self._symbol_caches:
            return None
        
        cache = self._symbol_caches[symbol]
        highs = cache.get_highs()
        lows = cache.get_lows()
        closes = cache.get_closes()
        volumes = cache.get_volumes()
        
        if len(closes) < period + 1:
            return None
        
        typical_prices = (highs + lows + closes) / 3
        money_flow = typical_prices * volumes
        
        positive_flow = np.zeros(len(closes) - 1)
        negative_flow = np.zeros(len(closes) - 1)
        
        for i in range(1, len(closes)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_flow[i-1] = money_flow[i]
            else:
                negative_flow[i-1] = money_flow[i]
        
        positive_mf = np.sum(positive_flow[-period:])
        negative_mf = np.sum(negative_flow[-period:])
        
        if negative_mf == 0:
            return 100.0
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        
        return round(mfi, 4)
    
    def calculate_kdj(self, symbol: str, n: int = 9, m1: int = 3, m2: int = 3) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """计算KDJ随机指标"""
        if symbol not in self._symbol_caches:
            return None, None, None
        
        cache = self._symbol_caches[symbol]
        highs = cache.get_highs()
        lows = cache.get_lows()
        closes = cache.get_closes()
        
        if len(closes) < n:
            return None, None, None
        
        rsv = np.zeros(len(closes) - n + 1)
        for i in range(n - 1, len(closes)):
            lowest = np.min(lows[i-n+1:i+1])
            highest = np.max(highs[i-n+1:i+1])
            if highest - lowest > 0:
                rsv[i-n+1] = (closes[i] - lowest) / (highest - lowest) * 100
            else:
                rsv[i-n+1] = 50
        
        k = np.zeros(len(rsv))
        d = np.zeros(len(rsv))
        
        k[0] = rsv[0]
        d[0] = rsv[0]
        
        for i in range(1, len(rsv)):
            k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
            d[i] = (2/3) * d[i-1] + (1/3) * k[i]
        
        j = 3 * k[-1] - 2 * d[-1]
        
        return round(k[-1], 4), round(d[-1], 4), round(j, 4)
    
    def calculate_vwap(self, symbol: str) -> Optional[float]:
        """计算VWAP (成交量加权平均价)"""
        if symbol not in self._symbol_caches:
            return None
        
        cache = self._symbol_caches[symbol]
        highs = cache.get_highs()
        lows = cache.get_lows()
        closes = cache.get_closes()
        volumes = cache.get_volumes()
        
        if len(closes) < 2:
            return None
        
        typical_prices = (highs + lows + closes) / 3
        cumulative_tp_volume = np.sum(typical_prices * volumes)
        cumulative_volume = np.sum(volumes)
        
        if cumulative_volume == 0:
            return None
        
        vwap = cumulative_tp_volume / cumulative_volume
        
        return round(vwap, 6)
    
    # ── 多时间周期指标方法 ──────────────────────────────────────────

    def update_mtf_bar(self, symbol: str, timeframe: str, bar: Dict[str, Any]) -> None:
        """更新指定时间周期的K线数据"""
        if not self._mtf_enabled:
            return
        if symbol not in self._symbol_caches:
            self.register_symbol(symbol)

        cache = self._symbol_caches[symbol]
        open_price = bar.get("open", bar.get("o", 0))
        high = bar.get("high", bar.get("h", 0))
        low = bar.get("low", bar.get("l", 0))
        close = bar.get("close", bar.get("c", 0))
        volume = bar.get("volume", bar.get("v", 0))

        cache.update_mtf(timeframe, open_price, high, low, close, volume)

    def calculate_all_mtf(self, symbol: str, timeframes: List[str] = None) -> IndicatorSet:
        """
        计算所有时间周期的指标，返回包含MTF数据的指标集合

        此方法在基础指标之上，额外计算各时间周期的RSI/ATR/布林带，
        为多周期共振策略提供数据支持。
        """
        # 先计算基础指标
        result = self.calculate_all(symbol)
        if not self._mtf_enabled or symbol not in self._symbol_caches:
            return result

        timeframes = timeframes or self._mtf_timeframes
        cache = self._symbol_caches[symbol]
        timestamp = datetime.now()

        mtf_results: Dict[str, Dict[str, IndicatorResult]] = {}

        for tf in timeframes:
            if tf not in cache.mtf_buffers:
                continue

            closes = cache.get_mtf_closes(tf)
            if len(closes) < self._rsi_period + 1:
                continue

            tf_indicators: Dict[str, IndicatorResult] = {}

            # RSI
            try:
                rsi_val = self._calc_rsi_on_data(closes, self._rsi_period)
                if rsi_val is not None:
                    tf_indicators[f"rsi_{tf}"] = IndicatorResult(f"rsi_{tf}", rsi_val, timestamp)
            except Exception:
                pass

            # ATR
            try:
                highs = cache.get_mtf_highs(tf)
                lows = cache.get_mtf_lows(tf)
                if len(highs) >= self._atr_period:
                    atr_val = self._calc_atr_on_data(highs, lows, closes, self._atr_period)
                    if atr_val is not None:
                        tf_indicators[f"atr_{tf}"] = IndicatorResult(f"atr_{tf}", atr_val, timestamp)
            except Exception:
                pass

            # Bollinger
            try:
                if len(closes) >= self._bollinger_period:
                    upper, middle, lower = self._calc_bollinger_on_data(
                        closes, self._bollinger_period, self._bollinger_std)
                    tf_indicators[f"boll_upper_{tf}"] = IndicatorResult(
                        f"boll_upper_{tf}", upper, timestamp)
                    tf_indicators[f"boll_lower_{tf}"] = IndicatorResult(
                        f"boll_lower_{tf}", lower, timestamp)
                    tf_indicators[f"boll_pos_{tf}"] = IndicatorResult(f"boll_pos_{tf}",
                        (closes[-1] - lower) / (upper - lower) if upper > lower else 0.5, timestamp)
            except Exception:
                pass

            if tf_indicators:
                mtf_results[tf] = tf_indicators

        result.mtf_indicators = mtf_results

        return result

    def get_mtf_confluence(self, symbol: str, indicator_name: str,
                           timeframes: List[str] = None) -> Dict[str, Any]:
        """
        多周期共振分析：检查某指标在多个时间周期上是否方向一致

        Returns:
            {confluence: bool, direction: str, agreement_ratio: float, details: {...}}
        """
        if not self._mtf_enabled:
            return {"confluence": False, "direction": "none", "agreement_ratio": 0.0}

        result = self.calculate_all_mtf(symbol, timeframes)
        timeframes = timeframes or self._mtf_timeframes

        bullish_count = 0
        bearish_count = 0
        details = {}

        for tf in timeframes:
            tf_data = result.mtf_indicators.get(tf, {})

            if indicator_name == "rsi":
                rsi_key = f"rsi_{tf}"
                if rsi_key in tf_data:
                    val = tf_data[rsi_key].value
                    details[tf] = val
                    if val < 40:
                        bullish_count += 1
                    elif val > 60:
                        bearish_count += 1

            elif indicator_name == "bollinger":
                pos_key = f"boll_pos_{tf}"
                if pos_key in tf_data:
                    val = tf_data[pos_key].value
                    details[tf] = val
                    if val < 0.3:
                        bullish_count += 1
                    elif val > 0.7:
                        bearish_count += 1

        total = bullish_count + bearish_count
        if total == 0:
            return {"confluence": False, "direction": "none", "agreement_ratio": 0.0}

        agreement_ratio = max(bullish_count, bearish_count) / max(total, 1)
        direction = "bullish" if bullish_count > bearish_count else "bearish"
        confluence = agreement_ratio >= 0.67  # 至少2/3时间周期一致

        return {
            "confluence": confluence,
            "direction": direction,
            "agreement_ratio": round(agreement_ratio, 3),
            "details": details
        }

    @staticmethod
    def _calc_rsi_on_data(closes: np.ndarray, period: int) -> Optional[float]:
        """在给定数据上计算RSI（不依赖缓存）"""
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100 - (100 / (1 + rs)), 4)

    @staticmethod
    def _calc_atr_on_data(highs: np.ndarray, lows: np.ndarray,
                           closes: np.ndarray, period: int) -> Optional[float]:
        """在给定数据上计算ATR（不依赖缓存）"""
        if len(closes) < 2:
            return None
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1]))
        )
        if len(tr) < period:
            period = len(tr)
        return round(np.mean(tr[-period:]), 6)

    @staticmethod
    def _calc_bollinger_on_data(closes: np.ndarray, period: int,
                                 std_dev: float) -> Tuple[float, float, float]:
        """在给定数据上计算布林带（不依赖缓存）"""
        recent = closes[-period:]
        middle = float(np.mean(recent))
        std = float(np.std(recent))
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return round(upper, 6), round(middle, 6), round(lower, 6)

    # ── 套利指标 ──────────────────────────────────────────────────

    def calculate_basis(self, spot_price: float, futures_price: float) -> Dict[str, float]:
        """
        计算基差套利指标
        返回: {basis, basis_rate, annualized_rate}
        """
        if spot_price <= 0 or futures_price <= 0:
            return {"basis": 0, "basis_rate": 0, "annualized_rate": 0}
        
        basis = futures_price - spot_price
        basis_rate = basis / spot_price
        annualized_rate = basis_rate * 365 * 24 * 60
        
        return {
            "basis": round(basis, 6),
            "basis_rate": round(basis_rate, 6),
            "annualized_rate": round(annualized_rate, 6)
        }
    
    def get_trend_signals(self, symbol: str) -> Dict[str, Any]:
        """获取趋势信号组合"""
        indicator_set = self.calculate_all(symbol)
        
        signals = {
            "bullish": 0,
            "bearish": 0,
            "strength": 0,
            "reasons": []
        }
        
        rsi = indicator_set.get("rsi")
        if rsi is not None:
            if rsi < 30:
                signals["bullish"] += 1
                signals["reasons"].append("RSI超卖")
            elif rsi > 70:
                signals["bearish"] += 1
                signals["reasons"].append("RSI超买")
        
        macd_hist = indicator_set.get("macd_histogram")
        if macd_hist is not None:
            if macd_hist > 0:
                signals["bullish"] += 1
                signals["reasons"].append("MACD金叉")
            else:
                signals["bearish"] += 1
                signals["reasons"].append("MACD死叉")
        
        boll_pos = indicator_set.get("bollinger_position")
        if boll_pos is not None:
            if boll_pos > 0.9:
                signals["bearish"] += 1
                signals["reasons"].append("接近布林上轨")
            elif boll_pos < 0.1:
                signals["bullish"] += 1
                signals["reasons"].append("接近布林下轨")
        
        adx = indicator_set.get("adx")
        plus_di = indicator_set.get("plus_di")
        minus_di = indicator_set.get("minus_di")
        if adx is not None and adx > 25:
            if plus_di > minus_di:
                signals["bullish"] += 1
                signals["strength"] = 1
                signals["reasons"].append("上升趋势")
            else:
                signals["bearish"] += 1
                signals["strength"] = -1
                signals["reasons"].append("下降趋势")
        
        return signals
    
    def _calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算EMA (指数移动平均)"""
        if len(data) < period:
            return np.array([])
        
        multiplier = 2 / (period + 1)
        ema = np.zeros(len(data) - period + 1)
        
        ema[0] = np.mean(data[:period])
        
        for i in range(1, len(ema)):
            ema[i] = (data[period + i - 1] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def _calculate_ema_series(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算EMA序列"""
        if len(data) < period:
            return np.array([])
        
        multiplier = 2 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        
        for i in range(1, len(data)):
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1]
        
        return ema
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        stats = {
            "symbols_registered": len(self._symbol_caches),
            "compute_times": self._compute_times.copy(),
            "compute_count": self._compute_count.copy(),
            "avg_compute_time": {}
        }
        
        for symbol, count in self._compute_count.items():
            total_time = self._compute_times.get(symbol, 0)
            stats["avg_compute_time"][symbol] = total_time / count if count > 0 else 0
        
        return stats
    
    def clear_cache(self, symbol: str = None) -> None:
        """清除缓存"""
        with self._cache_lock:
            if symbol:
                if symbol in self._symbol_caches:
                    self._symbol_caches[symbol].indicator_cache.clear()
            else:
                for cache in self._symbol_caches.values():
                    cache.indicator_cache.clear()


_engine_instance: Optional[IndicatorEngine] = None

def get_indicator_engine(config: Dict[str, Any] = None) -> IndicatorEngine:
    """获取指标引擎单例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = IndicatorEngine(config)
    return _engine_instance