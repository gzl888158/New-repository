"""
统一市场状态融合引擎
负责融合多维度市场状态，输出单一的最终状态和权重解释。
遵循经验回顾中的核心教训：建立统一的状态融合与可解释计算链。
"""
import asyncio
import numpy as np
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from services.trend_vote import compute_trend_vote


class MarketRegime(Enum):
    """市场主状态"""
    TREND_BULLISH = "trend_bullish"      # 牛市趋势
    TREND_BEARISH = "trend_bearish"      # 熊市趋势
    RANGE_BOUND = "range_bound"          # 震荡区间
    EXTREME_VOLATILITY = "extreme_volatility"  # 极端波动
    FUNDING_CRUSH = "funding_crush"      # 资金费率极端
    LIQUIDITY_CRISIS = "liquidity_crisis"  # 流动性危机


class MarketSubtype(Enum):
    """市场子状态"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    REVERSAL = "reversal"
    CONTINUATION = "continuation"


class RegimeStrength(Enum):
    """状态强度"""
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    STRONG = 0.9


class MarketRegimeEngine:
    """统一市场状态融合引擎"""

    def __init__(self, config: Dict[str, Any], okx_client=None):
        self.config = config
        self.okx_client = okx_client
        
        self._current_regime: Optional[MarketRegime] = None
        self._current_subtype: Optional[MarketSubtype] = None
        self._current_strength: float = 0.0
        self._regime_confidence: float = 0.0
        self._last_update: Optional[datetime] = None
        
        self._factor_scores: Dict[str, float] = {}
        self._factor_weights: Dict[str, float] = {
            "trend": 0.30,
            "volatility": 0.25,
            "funding_rate": 0.15,
            "liquidity": 0.15,
            "momentum": 0.10,
            "sentiment": 0.05,
        }
        
        self._kline_cache: Dict[str, List[Dict]] = {}
        self._funding_cache: Dict[str, Dict] = {}
        self._funding_cache_time: Dict[str, datetime] = {}  # 资金费率缓存时间戳
        self._funding_cache_ttl: int = 300                   # 资金费率缓存有效期(秒)
        self._orderbook_cache: Dict[str, Dict] = {}

        # 企业级趋势行情判断参数（可从 config market_regime 覆盖）
        self._trend_adx_period = config.get("market_regime", {}).get("trend_adx_period", 14)
        self._trend_adx_floor = config.get("market_regime", {}).get("trend_adx_floor", 15)
        self._trend_adx_saturation = config.get("market_regime", {}).get("trend_adx_saturation", 40)
        self._trend_mtf_weight = config.get("market_regime", {}).get("trend_mtf_weight", 0.4)
        
        # 多币种支持
        self._watched_symbols: List[str] = config.get("market_regime", {}).get(
            "watched_symbols", ["BTC-USDT-SWAP"]
        )
        self._base_symbol: str = "BTC-USDT-SWAP"  # 基准币种（用于整体市场状态判断）
        # 各币种的独立市场状态
        self._symbol_regimes: Dict[str, Dict[str, Any]] = {}
        self._symbol_factor_scores: Dict[str, Dict[str, float]] = {}
        
        self._update_interval = 60
        self._running = False
        
        # 企业级状态稳定性参数（可从 config market_regime 覆盖）
        self._smoothing_alpha = max(0.0, min(1.0, config.get("market_regime", {}).get("smoothing_alpha", 0.5)))
        self._regime_hysteresis_margin = max(0.0, config.get("market_regime", {}).get("regime_hysteresis_margin", 0.08))
        self._min_stable_updates = max(1, int(config.get("market_regime", {}).get("min_stable_updates", 3)))
        self._stability_confidence_discount = max(0.0, min(1.0, config.get("market_regime", {}).get("stability_confidence_discount", 0.7)))
        self._history_max_len = max(1, int(config.get("market_regime", {}).get("history_max_len", 100)))
        
        # 因子得分 EMA 平滑状态（跨更新周期持久，抑制瞬时噪声）
        self._smoothed_scores: Dict[str, float] = {}
        self._smoothed_symbol_scores: Dict[str, Dict[str, float]] = {}
        
        # 状态稳定性追踪（持续时间 / 连续周期 / 切换历史）
        self._regime_history: List[Dict[str, Any]] = []
        self._regime_switch_count: int = 0
        self._regime_consecutive_updates: int = 0
        self._regime_started_at: Optional[datetime] = None
        self._stable: bool = False
        
        logger.info(f"MarketRegimeEngine initialized, watching {len(self._watched_symbols)} symbols: {self._watched_symbols}")

    async def start(self):
        """启动状态监控循环"""
        if self._running:
            return
        self._running = True
        asyncio.create_task(self._regime_update_loop())
        logger.info("MarketRegimeEngine started")

    async def stop(self):
        """停止状态监控循环"""
        self._running = False
        logger.info("MarketRegimeEngine stopped")

    async def _regime_update_loop(self):
        """定期更新市场状态"""
        while self._running:
            try:
                await self._update_regime()
            except Exception as e:
                logger.error(f"Regime update error: {e}")
            await asyncio.sleep(self._update_interval)

    async def _update_regime(self):
        """融合所有因子，输出最终状态（支持多币种）"""
        await self._collect_factor_data()
        
        # 计算基准币种（BTC）的整体市场状态（因子得分先 EMA 平滑，再带迟滞判定）
        raw_scores = await self._calculate_factor_scores(self._base_symbol)
        factor_scores = self._apply_smoothing(self._base_symbol, raw_scores)
        self._factor_scores = factor_scores
        
        previous_regime = self._current_regime
        regime, subtype, strength, confidence, breakdown = self._resolve_regime(
            factor_scores, previous_regime=previous_regime
        )
        
        # 追踪状态稳定性；短命状态降级置信度
        self._track_regime_transition(regime)
        if not self._stable:
            confidence = max(0.0, confidence * self._stability_confidence_discount)
        
        self._current_regime = regime
        self._current_subtype = subtype
        self._current_strength = strength
        self._regime_confidence = confidence
        self._last_update = datetime.now()
        
        logger.info(
            f"[REGIME] {regime.value} | {subtype.value} | strength={strength:.2f} | "
            f"confidence={confidence:.2f} | stable={self._stable} | age={self._get_regime_age_seconds():.0f}s"
        )
        logger.debug(f"[REGIME] Breakdown: {breakdown}")
        
        # 计算每个监控币种的独立市场状态（同样做平滑与迟滞）
        for symbol in self._watched_symbols:
            if symbol == self._base_symbol:
                # BTC 直接复用已计算的数据
                self._symbol_factor_scores[symbol] = factor_scores
                self._symbol_regimes[symbol] = {
                    "regime": regime.value,
                    "subtype": subtype.value,
                    "strength": strength,
                    "confidence": confidence,
                    "factor_scores": factor_scores,
                }
            else:
                try:
                    raw_symbol_scores = await self._calculate_factor_scores(symbol)
                    symbol_scores = self._apply_smoothing(symbol, raw_symbol_scores)
                    self._symbol_factor_scores[symbol] = symbol_scores
                    prev_sym = self._symbol_regimes.get(symbol, {}).get("regime")
                    prev_sym_regime = MarketRegime(prev_sym) if prev_sym else None
                    sym_regime, sym_subtype, sym_strength, sym_conf, _ = self._resolve_regime(
                        symbol_scores, previous_regime=prev_sym_regime
                    )
                    self._symbol_regimes[symbol] = {
                        "regime": sym_regime.value,
                        "subtype": sym_subtype.value,
                        "strength": sym_strength,
                        "confidence": sym_conf,
                        "factor_scores": symbol_scores,
                    }
                except Exception as e:
                    logger.debug(f"Failed to compute regime for {symbol}: {e}")

    async def _collect_factor_data(self):
        """收集各因子数据（支持多币种）"""
        if not self.okx_client:
            return
        
        try:
            # 并行获取所有监控币种的数据
            for symbol in self._watched_symbols:
                await self._collect_symbol_data(symbol)
        except Exception as e:
            logger.debug(f"Failed to collect factor data: {e}")

    async def _collect_symbol_data(self, symbol: str):
        """收集单个币种的因子数据"""
        try:
            # 尝试使用异步方法，回退到同步方法
            if hasattr(self.okx_client, 'get_ticker_async'):
                ticker = await self.okx_client.get_ticker_async(symbol)
            else:
                ticker = self.okx_client.get_ticker(symbol)
            
            if ticker:
                self._orderbook_cache[symbol] = {
                    "last_price": float(ticker.get("last", 0)),
                    "mark_price": float(ticker.get("markPx", 0)),
                    "bid_price": float(ticker.get("bidPx", 0)),
                    "ask_price": float(ticker.get("askPx", 0)),
                    "volume_24h": float(ticker.get("vol24h", 0)),
                    "high_24h": float(ticker.get("high24h", 0)),
                    "low_24h": float(ticker.get("low24h", 0)),
                }
            
            # 获取资金费率
            if hasattr(self.okx_client, 'get_funding_rate_async'):
                funding_info = await self.okx_client.get_funding_rate_async(symbol)
            else:
                funding_info = self.okx_client.get_funding_rate(symbol)
            
            if funding_info:
                funding_rate_val = float(funding_info.get("fundingRate", 0))
                self._funding_cache[symbol] = {
                    "funding_rate": funding_rate_val,
                    "next_funding_time": funding_info.get("nextFundingTime"),
                    "funding_rate_24h_avg": float(funding_info.get("fundingRate24hAvg", 0)),
                }
                self._funding_cache_time[symbol] = datetime.now()
            else:
                last_time = self._funding_cache_time.get(symbol)
                if last_time and (datetime.now() - last_time).total_seconds() > self._funding_cache_ttl:
                    logger.warning(f"{symbol} funding rate data stale ({self._funding_cache_ttl}s+), using fallback")
            
            # 获取K线（1H和4H）
            if hasattr(self.okx_client, 'get_kline_async'):
                klines_1h = await self.okx_client.get_kline_async(symbol, "1H", limit=60)
            else:
                klines_1h = self.okx_client.get_kline(symbol, "1H", limit=60)
            if klines_1h:
                # OKX history-candles 倒序返回（最新在前），按时间戳正序排序，
                # 否则 compute_trend_vote 的 EMA/斜率/结构判断会整体反转（牛市判熊市）
                klines_1h = self._sort_klines_ascending(klines_1h)
                self._kline_cache[f"{symbol}_1H"] = klines_1h
            
            if hasattr(self.okx_client, 'get_kline_async'):
                klines_4h = await self.okx_client.get_kline_async(symbol, "4H", limit=20)
            else:
                klines_4h = self.okx_client.get_kline(symbol, "4H", limit=20)
            if klines_4h:
                klines_4h = self._sort_klines_ascending(klines_4h)
                self._kline_cache[f"{symbol}_4H"] = klines_4h
                
        except Exception as e:
            logger.debug(f"Failed to collect data for {symbol}: {e}")

    async def _calculate_factor_scores(self, symbol: str = None) -> Dict[str, float]:
        """计算各因子得分（-1 到 +1），支持指定币种"""
        if symbol is None:
            symbol = self._base_symbol
        
        scores = {}
        
        scores["trend"] = await self._calculate_trend_score(symbol)
        scores["volatility"] = await self._calculate_volatility_score(symbol)
        scores["funding_rate"] = await self._calculate_funding_score(symbol)
        scores["liquidity"] = await self._calculate_liquidity_score(symbol)
        scores["momentum"] = await self._calculate_momentum_score(symbol)
        scores["sentiment"] = await self._calculate_sentiment_score(symbol)
        
        return scores

    def _apply_smoothing(self, symbol: str, raw_scores: Dict[str, float]) -> Dict[str, float]:
        """对因子得分做 EMA 平滑，抑制瞬时噪声导致的 regime 抖动。

        每个因子得分跨更新周期做指数加权：smooth = alpha * raw + (1-alpha) * prev。
        首次无历史时直接采用原始值。BTC 与各币种分别维护平滑状态。
        """
        smoothed = self._smoothed_scores if symbol == self._base_symbol \
            else self._smoothed_symbol_scores.setdefault(symbol, {})
        alpha = self._smoothing_alpha

        result = {}
        for factor, raw in raw_scores.items():
            prev = smoothed.get(factor)
            if prev is None:
                result[factor] = raw
            else:
                result[factor] = alpha * raw + (1.0 - alpha) * prev
            smoothed[factor] = result[factor]
        return result

    async def _calculate_trend_score(self, symbol: str = None) -> float:
        """趋势得分：+1 强牛市，-1 强熊市

        企业级强化：复用共享的 ADX 趋势强度 + 价格结构 + EMA 排列 + DI 方向投票，
        再叠加多周期（1H+4H）一致性。弱趋势/震荡（低 ADX 或方向分歧）会自然趋近 0，
        便于 _resolve_regime 准确区分 TREND_BULLISH/BEARISH 与 RANGE_BOUND。
        """
        if symbol is None:
            symbol = self._base_symbol

        closes, highs, lows = self._extract_ohlc(symbol, "4H")
        if len(closes) < 20:
            return 0.0

        try:
            vote = compute_trend_vote(
                closes, highs, lows,
                adx_period=self._trend_adx_period,
                adx_floor=self._trend_adx_floor,
                adx_saturation=self._trend_adx_saturation,
            )

            # 多周期确认：1H 与 4H 方向一致性增强（0=相反，0.5=无数据，1=一致）
            mf_align = self._calc_timeframe_alignment(symbol, vote["direction"])

            score = vote["direction"] * vote["adx_strength"] * (
                1.0 - self._trend_mtf_weight + self._trend_mtf_weight * mf_align
            )
            return max(-1.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"Trend score calc failed: {e}")
            return 0.0

    @staticmethod
    def _sort_klines_ascending(klines):
        """按时间戳正序排序 K 线（兼容 list 与 dict 两种格式）。

        OKX history-candles 返回倒序（最新在前），趋势/动量等方向类计算依赖正序，
        此处统一排序，避免方向判断整体反转（牛市判熊市、熊市判牛市）。
        """
        if not klines:
            return klines
        try:
            if isinstance(klines[0], dict):
                return sorted(klines, key=lambda k: float(k.get("timestamp", k.get("ts", 0))))
            return sorted(klines, key=lambda k: int(k[0]))
        except (IndexError, TypeError, ValueError, KeyError):
            return klines

    def _extract_ohlc(self, symbol: str, timeframe: str) -> Tuple[List[float], List[float], List[float]]:
        """提取 OHLC 序列，兼容 dict 和 list 两种 K 线格式"""
        klines = self._kline_cache.get(f"{symbol}_{timeframe}", [])
        closes: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        for k in klines:
            try:
                if isinstance(k, dict):
                    closes.append(float(k.get("close", 0)))
                    highs.append(float(k.get("high", 0)))
                    lows.append(float(k.get("low", 0)))
                else:
                    closes.append(float(k[4]))
                    highs.append(float(k[2]))
                    lows.append(float(k[3]))
            except (IndexError, TypeError, ValueError):
                continue
        return closes, highs, lows

    def _calc_timeframe_alignment(self, symbol: str, direction: float) -> float:
        """1H 趋势方向与主周期方向一致性：1=一致，0=相反，0.5=无数据"""
        closes1h, _, _ = self._extract_ohlc(symbol, "1H")
        if len(closes1h) < 20:
            return 0.5
        try:
            ema5 = self._calculate_ema(closes1h, 5)
            ema20 = self._calculate_ema(closes1h, 20)
            s = ema5[-1] - ema20[-1]
            dir1h = 1.0 if s > 0 else (-1.0 if s < 0 else 0.0)
            if dir1h == 0.0:
                return 0.5
            return 1.0 if (dir1h > 0) == (direction > 0) else 0.0
        except Exception:
            return 0.5

    async def _calculate_volatility_score(self, symbol: str = None) -> float:
        """波动率得分：+1 高波动，-1 低波动"""
        if symbol is None:
            symbol = self._base_symbol
        klines = self._kline_cache.get(f"{symbol}_1H", [])
        if len(klines) < 20:
            return 0.0
        
        try:
            # K线数据兼容 list 和 dict 格式
            if isinstance(klines[0], dict):
                prices = [float(k.get("close", 0)) for k in klines]
            else:
                prices = [float(k[4]) for k in klines]
            
            returns = np.diff(prices) / prices[:-1]
            
            current_std = np.std(returns[-10:])
            avg_std = np.std(returns)
            
            if avg_std == 0:
                return 0.0
            
            ratio = current_std / avg_std
            
            if ratio > 2.0:
                return min(1.0, (ratio - 1) * 0.5)
            elif ratio < 0.5:
                return max(-1.0, -(1 - ratio) * 0.5)
            
            return 0.0
        except Exception as e:
            logger.debug(f"Volatility score calc failed: {e}")
            return 0.0

    async def _calculate_funding_score(self, symbol: str = None) -> float:
        """资金费率得分：+1 极端正（多头付空头），-1 极端负（空头付多头）"""
        if symbol is None:
            symbol = self._base_symbol
        funding = self._funding_cache.get(symbol, {})
        rate = funding.get("funding_rate", 0)
        
        if rate > 0.001:
            return min(1.0, rate * 500)
        elif rate < -0.001:
            return max(-1.0, rate * 500)
        
        return 0.0

    async def _calculate_liquidity_score(self, symbol: str = None) -> float:
        """流动性得分：+1 高流动性，-1 低流动性"""
        if symbol is None:
            symbol = self._base_symbol
        ob = self._orderbook_cache.get(symbol, {})
        last_price = ob.get("last_price", 0)
        bid = ob.get("bid_price", 0)
        ask = ob.get("ask_price", 0)
        
        if last_price == 0 or bid == 0 or ask == 0:
            return 0.0
        
        spread = (ask - bid) / last_price
        
        if spread < 0.0001:
            return 1.0
        elif spread > 0.001:
            return -1.0
        
        return max(-1.0, min(1.0, (0.0001 - spread) / 0.0009))

    async def _calculate_momentum_score(self, symbol: str = None) -> float:
        """动量得分：+1 强上涨动量，-1 强下跌动量"""
        if symbol is None:
            symbol = self._base_symbol
        klines = self._kline_cache.get(f"{symbol}_1H", [])
        if len(klines) < 10:
            return 0.0
        
        try:
            # K线数据兼容 list 和 dict 格式
            if isinstance(klines[0], dict):
                prices = [float(k.get("close", 0)) for k in klines]
            else:
                prices = [float(k[4]) for k in klines]
            
            if len(prices) < 10:
                return 0.0
            
            recent = prices[-5:]
            earlier = prices[-10:-5]
            
            recent_return = recent[-1] / recent[0] - 1
            earlier_return = earlier[-1] / earlier[0] - 1
            
            momentum = recent_return - earlier_return
            return max(-1.0, min(1.0, momentum * 20))
        except Exception as e:
            logger.debug(f"Momentum score calc failed: {e}")
            return 0.0

    async def _calculate_sentiment_score(self, symbol: str = None) -> float:
        """情绪得分（简化版）：基于价格位置"""
        if symbol is None:
            symbol = self._base_symbol
        ob = self._orderbook_cache.get(symbol, {})
        last = ob.get("last_price", 0)
        high = ob.get("high_24h", 0)
        low = ob.get("low_24h", 0)
        
        if high == low:
            return 0.0
        
        position = (last - low) / (high - low)
        
        if position > 0.8:
            return min(1.0, (position - 0.5) * 2)
        elif position < 0.2:
            return max(-1.0, -(0.5 - position) * 2)
        
        return 0.0

    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """计算 EMA"""
        if len(prices) < period:
            return prices
        
        ema = []
        multiplier = 2 / (period + 1)
        ema.append(prices[0])
        
        for i in range(1, len(prices)):
            ema.append(prices[i] * multiplier + ema[-1] * (1 - multiplier))
        
        return ema

    def _resolve_regime(
        self,
        factor_scores: Dict[str, float],
        previous_regime: Optional[MarketRegime] = None,
    ) -> Tuple[MarketRegime, MarketSubtype, float, float, Dict[str, float]]:
        """融合因子得分，输出最终状态（核心函数）

        P0: 使用原始因子得分（非加权得分）进行阈值判定。
        加权得分最大仅1.0*weight（如volatility为0.25），无法达到旧阈值0.4/0.6。

        企业级迟滞（hysteresis）：每个状态的进入阈值高于退出阈值（进入需更强信号、
        已处于该状态时用更宽松阈值维持），避免因子在阈值附近抖动导致 regime 频繁翻转。
        """
        weighted_scores = {}
        total_weight = sum(self._factor_weights.values())

        for factor, score in factor_scores.items():
            weight = self._factor_weights.get(factor, 0)
            weighted_scores[factor] = score * weight / total_weight

        raw_trend = factor_scores.get("trend", 0)
        raw_vol = factor_scores.get("volatility", 0)
        raw_funding = factor_scores.get("funding_rate", 0)
        raw_liquidity = factor_scores.get("liquidity", 0)

        breakdown = {k: round(v, 4) for k, v in weighted_scores.items()}

        hyst = max(0.0, self._regime_hysteresis_margin)
        prev = previous_regime

        # 高优先级危机状态（迟滞：进入阈值 = 基准 + hyst，退出阈值 = 基准 - hyst）
        if raw_vol > (0.6 - hyst if prev == MarketRegime.EXTREME_VOLATILITY else 0.6 + hyst):
            return self._build_result(
                MarketRegime.EXTREME_VOLATILITY,
                MarketSubtype.STRONG if raw_vol > 0.8 else MarketSubtype.MODERATE,
                min(1.0, raw_vol * 1.2),
                0.7 + raw_vol * 0.3,
                breakdown,
            )

        af = abs(raw_funding)
        if af > (0.5 - hyst if prev == MarketRegime.FUNDING_CRUSH else 0.5 + hyst):
            return self._build_result(
                MarketRegime.FUNDING_CRUSH,
                MarketSubtype.STRONG if af > 0.7 else MarketSubtype.MODERATE,
                min(1.0, af * 1.5),
                0.6 + af * 0.4,
                breakdown,
            )

        if raw_liquidity < (-0.5 + hyst if prev == MarketRegime.LIQUIDITY_CRISIS else -0.5 - hyst):
            return self._build_result(
                MarketRegime.LIQUIDITY_CRISIS,
                MarketSubtype.WEAK,
                min(1.0, abs(raw_liquidity) * 1.0),
                0.7 + abs(raw_liquidity) * 0.3,
                breakdown,
            )

        # 趋势 vs 震荡（迟滞：bullish/bearish 在退出阈值带内保持上一状态）
        was_bullish = prev == MarketRegime.TREND_BULLISH
        was_bearish = prev == MarketRegime.TREND_BEARISH

        is_bullish = raw_trend > (0.2 - hyst if was_bullish else 0.2 + hyst)
        is_bearish = raw_trend < (-0.2 + hyst if was_bearish else -0.2 - hyst)

        if is_bullish:
            if raw_trend > 0.5:
                subtype = MarketSubtype.STRONG
            elif raw_trend > 0.35:
                subtype = MarketSubtype.MODERATE
            else:
                subtype = MarketSubtype.WEAK
            return self._build_result(
                MarketRegime.TREND_BULLISH,
                subtype,
                min(1.0, raw_trend * 1.5),
                0.6 + raw_trend * 0.4,
                breakdown,
            )
        if is_bearish:
            if raw_trend < -0.5:
                subtype = MarketSubtype.STRONG
            elif raw_trend < -0.35:
                subtype = MarketSubtype.MODERATE
            else:
                subtype = MarketSubtype.WEAK
            return self._build_result(
                MarketRegime.TREND_BEARISH,
                subtype,
                min(1.0, abs(raw_trend) * 1.5),
                0.6 + abs(raw_trend) * 0.4,
                breakdown,
            )

        return self._build_result(
            MarketRegime.RANGE_BOUND,
            MarketSubtype.MODERATE,
            0.3 + abs(raw_trend) * 0.5,
            0.5 + abs(raw_trend) * 0.3,
            breakdown,
        )

    @staticmethod
    def _build_result(
        regime: MarketRegime,
        subtype: MarketSubtype,
        strength: float,
        confidence: float,
        breakdown: Dict[str, float],
    ) -> Tuple[MarketRegime, MarketSubtype, float, float, Dict[str, float]]:
        """统一构造状态返回，强度与置信度裁剪到 [0, 1]。"""
        return (
            regime,
            subtype,
            max(0.0, min(1.0, strength)),
            max(0.0, min(1.0, confidence)),
            breakdown,
        )

    def _track_regime_transition(self, new_regime: MarketRegime):
        """追踪 regime 稳定性：持续时间、连续更新周期数、切换历史。

        在 regime 切换时记录切换事件（含上一状态持续时间），并更新稳定性标志。
        状态连续满足 min_stable_updates 个周期才视为稳定（stable）。
        """
        now = datetime.now()

        if self._current_regime is None:
            self._regime_consecutive_updates = 1
            self._regime_started_at = now
        elif self._current_regime != new_regime:
            self._regime_switch_count += 1
            duration = (now - self._regime_started_at).total_seconds() if self._regime_started_at else 0.0
            self._regime_history.append({
                "from": self._current_regime.value,
                "to": new_regime.value,
                "at": now.isoformat(),
                "duration_seconds": round(duration, 3),
                "consecutive_updates": self._regime_consecutive_updates,
            })
            if len(self._regime_history) > self._history_max_len:
                self._regime_history = self._regime_history[-self._history_max_len:]
            self._regime_consecutive_updates = 1
            self._regime_started_at = now
        else:
            self._regime_consecutive_updates += 1

        self._stable = self._regime_consecutive_updates >= self._min_stable_updates

    def _get_regime_age_seconds(self) -> float:
        """当前 regime 已持续的秒数。"""
        if not self._regime_started_at:
            return 0.0
        return max(0.0, (datetime.now() - self._regime_started_at).total_seconds())

    def get_regime(self) -> Dict[str, Any]:
        """获取当前市场状态"""
        return {
            "regime": self._current_regime.value if self._current_regime else "unknown",
            "subtype": self._current_subtype.value if self._current_subtype else "unknown",
            "strength": self._current_strength,
            "confidence": self._regime_confidence,
            "factor_scores": self._factor_scores,
            "factor_weights": self._factor_weights,
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }

    def get_symbol_regime(self, symbol: str) -> Dict[str, Any]:
        """获取指定币种的独立市场状态数据
        
        如果该币种在监控列表中，返回其独立计算的市场状态。
        否则，基于BTC基准 + 从缓存中提取币种特定数据。
        """
        # 优先返回独立计算的币种状态
        if symbol in self._symbol_regimes:
            sym_data = self._symbol_regimes[symbol]
            result = {
                "regime": sym_data["regime"],
                "strength": sym_data["strength"],
                "confidence": sym_data["confidence"],
                "volatility": sym_data.get("factor_scores", {}).get("volatility", 0.02),
                "volatility_percentile": sym_data.get("factor_scores", {}).get("volatility", 0.0),
                "momentum": sym_data.get("factor_scores", {}).get("momentum", 0.0),
                "trend_strength": sym_data.get("factor_scores", {}).get("trend", 0.0),
                "liquidity_score": sym_data.get("factor_scores", {}).get("liquidity", 0.0),
                "funding_score": sym_data.get("factor_scores", {}).get("funding_rate", 0.0),
            }
            # 补充币种特定的波动率数据
            if symbol in self._orderbook_cache:
                cache = self._orderbook_cache[symbol]
                high24 = cache.get("high_24h", 0)
                low24 = cache.get("low_24h", 0)
                last = cache.get("last_price", 0)
                if high24 > 0 and low24 > 0 and last > 0:
                    vol_24h = (high24 - low24) / last
                    result["volatility"] = vol_24h
                    result["volatility_percentile"] = min(vol_24h / 0.05, 1.0)
            return result
        
        # 回退：基于BTC基准 + 币种特定缓存数据
        result = {
            "regime": self._current_regime.value if self._current_regime else "unknown",
            "strength": self._current_strength,
            "confidence": self._regime_confidence,
            "volatility": self._factor_scores.get("volatility", 0.02),
            "volatility_percentile": self._factor_scores.get("volatility", 0.0),
            "momentum": self._factor_scores.get("momentum", 0.0),
            "trend_strength": self._factor_scores.get("trend", 0.0),
            "liquidity_score": self._factor_scores.get("liquidity", 0.0),
        }
        # 如果缓存中有该币种的ticker数据，使用币种特定波动率
        if symbol in self._orderbook_cache:
            cache = self._orderbook_cache[symbol]
            high24 = cache.get("high_24h", 0)
            low24 = cache.get("low_24h", 0)
            last = cache.get("last_price", 0)
            if high24 > 0 and low24 > 0 and last > 0:
                vol_24h = (high24 - low24) / last
                result["volatility"] = vol_24h
                result["volatility_percentile"] = min(vol_24h / 0.05, 1.0)
        return result

    def get_all_symbol_regimes(self) -> Dict[str, Dict[str, Any]]:
        """获取所有监控币种的市场状态"""
        result = {}
        for symbol in self._watched_symbols:
            result[symbol] = self.get_symbol_regime(symbol)
        return result

    def add_watched_symbol(self, symbol: str):
        """动态添加监控币种"""
        if symbol not in self._watched_symbols:
            self._watched_symbols.append(symbol)
            logger.info(f"Added symbol to market regime watch: {symbol}")

    def remove_watched_symbol(self, symbol: str):
        """动态移除监控币种"""
        if symbol in self._watched_symbols and symbol != self._base_symbol:
            self._watched_symbols.remove(symbol)
            self._symbol_regimes.pop(symbol, None)
            self._symbol_factor_scores.pop(symbol, None)
            logger.info(f"Removed symbol from market regime watch: {symbol}")

    def get_position_adjustment(self) -> Dict[str, float]:
        """获取基于市场状态的仓位调整建议"""
        if not self._current_regime:
            return {"overall": 1.0, "trend": 1.0, "scalping": 1.0, "grid": 1.0, "arbitrage": 1.0}
        
        adjustment = {"overall": 1.0}
        
        if self._current_regime == MarketRegime.EXTREME_VOLATILITY:
            adjustment["overall"] = max(0.3, 1.0 - self._current_strength * 0.5)
            adjustment["trend"] = adjustment["overall"] * 0.5
            adjustment["scalping"] = adjustment["overall"] * 0.8
            adjustment["grid"] = adjustment["overall"] * 0.6
            adjustment["arbitrage"] = adjustment["overall"] * 0.9
        elif self._current_regime == MarketRegime.FUNDING_CRUSH:
            adjustment["overall"] = 0.7
            adjustment["trend"] = 0.5 if self._current_subtype == MarketSubtype.STRONG else 0.7
            adjustment["scalping"] = 0.9
            adjustment["grid"] = 0.6
            adjustment["arbitrage"] = 1.0
        elif self._current_regime == MarketRegime.LIQUIDITY_CRISIS:
            adjustment["overall"] = max(0.2, 1.0 - self._current_strength * 0.6)
            adjustment["trend"] = adjustment["overall"] * 0.4
            adjustment["scalping"] = adjustment["overall"] * 0.5
            adjustment["grid"] = adjustment["overall"] * 0.6
            adjustment["arbitrage"] = adjustment["overall"] * 0.3
        elif self._current_regime == MarketRegime.TREND_BULLISH:
            adjustment["overall"] = 1.0 + self._current_strength * 0.2
            adjustment["trend"] = 1.2 if self._current_subtype == MarketSubtype.STRONG else 1.0
            adjustment["scalping"] = 0.9
            adjustment["grid"] = 0.8
            adjustment["arbitrage"] = 0.9
        elif self._current_regime == MarketRegime.TREND_BEARISH:
            adjustment["overall"] = max(0.5, 1.0 - self._current_strength * 0.3)
            adjustment["trend"] = 0.8 if self._current_subtype == MarketSubtype.STRONG else 0.9
            adjustment["scalping"] = 0.8
            adjustment["grid"] = 0.7
            adjustment["arbitrage"] = 0.9
        elif self._current_regime == MarketRegime.RANGE_BOUND:
            adjustment["overall"] = 1.0
            adjustment["trend"] = 0.7
            adjustment["scalping"] = 1.1
            adjustment["grid"] = 1.2
            adjustment["arbitrage"] = 1.0
        
        return adjustment

    def get_strategy_recommendation(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略级别的操作建议"""
        adjustment = self.get_position_adjustment()
        strat_adjust = adjustment.get(strategy_name, adjustment["overall"])
        
        recommendation = {
            "strategy": strategy_name,
            "adjustment_factor": strat_adjust,
            "regime": self._current_regime.value if self._current_regime else "unknown",
            "confidence": self._regime_confidence,
            "action": "increase" if strat_adjust > 1.1 else "decrease" if strat_adjust < 0.9 else "maintain",
            "reason": self._get_recommendation_reason(strategy_name, strat_adjust),
        }
        
        return recommendation

    def _get_recommendation_reason(self, strategy_name: str, factor: float) -> str:
        """生成建议理由"""
        if not self._current_regime:
            return "Unknown market regime"
        
        regime = self._current_regime.value
        subtype = self._current_subtype.value
        
        reasons = {
            ("trend_bullish", "strong", "trend"): "Strong bullish trend favors trend following",
            ("trend_bullish", "strong", "grid"): "Trend may break grid bounds, reduce exposure",
            ("trend_bearish", "strong", "trend"): "Strong bearish trend, reduce long exposure",
            ("range_bound", "moderate", "grid"): "Range bound market is ideal for grid strategy",
            ("range_bound", "moderate", "scalping"): "Sideways movement creates scalping opportunities",
            ("extreme_volatility", "strong", "trend"): "Extreme volatility increases whipsaw risk",
            ("extreme_volatility", "strong", "arbitrage"): "Arbitrage spreads may widen during volatility",
            ("funding_crush", "strong", "grid"): "Extreme funding rates increase holding costs",
            ("liquidity_crisis", "weak", "scalping"): "Low liquidity increases slippage risk",
        }
        
        return reasons.get((regime, subtype, strategy_name), f"{regime} market affects {strategy_name}")
