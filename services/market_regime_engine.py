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
    """市场主状态。

    兼容历史口径，同时新增 breakout / breakdown / reversal，统一为“状态对象”而非
    仅趋势/震荡/高风险三分类，避免策略门控对关键状态缺失。
    """
    TREND_BULLISH = "trend_bullish"      # 牛市趋势
    TREND_BEARISH = "trend_bearish"      # 熊市趋势
    BREAKOUT = "breakout"                # 突破（多头强势突破）
    BREAKDOWN = "breakdown"              # 破位（空头强势突破）
    REVERSAL = "reversal"                # 反转（趋势衰竭/高概率反手）
    RANGE_BOUND = "range_bound"          # 震荡区间
    EXTREME_VOLATILITY = "extreme_volatility"  # 极端波动
    FUNDING_CRUSH = "funding_crush"      # 资金费率极端
    LIQUIDITY_CRISIS = "liquidity_crisis"  # 流动性危机

    # 兼容历史字段名/同义词：旧代码可能按 trending_up / trending_down / ranging
    # 传入字符串，统一映射到现有标准值。
    TRENDING_UP = "trend_bullish"
    TRENDING_DOWN = "trend_bearish"
    RANGING = "range_bound"
    HIGH_VOL = "extreme_volatility"
    LOW_VOL = "range_bound"


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
        self._alert_manager = None
        
        self._current_regime: Optional[MarketRegime] = None
        self._current_subtype: Optional[MarketSubtype] = None
        self._current_strength: float = 0.0
        self._regime_confidence: float = 0.0
        self._last_update: Optional[datetime] = None

        # 细粒度检测器/仲裁器：主引擎优先消费 HMM 检测器输出，避免 breakout/reversal
        # 被旧 trend/range 口径吞并。
        self._detector = None
        self._regime_arbiter = None
        
        self._factor_scores: Dict[str, float] = {}
        self._factor_weights: Dict[str, float] = {
            "trend": 0.30,
            "volatility": 0.25,
            "funding_rate": 0.15,
            "liquidity": 0.15,
            "momentum": 0.10,
            "sentiment": 0.05,
        }
        configured_weights = config.get("market_regime", {}).get("factor_weights", {})
        if isinstance(configured_weights, dict):
            for factor, value in configured_weights.items():
                if factor not in self._factor_weights:
                    continue
                try:
                    parsed_weight = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(parsed_weight) and parsed_weight >= 0.0:
                    self._factor_weights[factor] = parsed_weight
        if sum(self._factor_weights.values()) <= 0.0:
            self._factor_weights = {
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
        self._data_collection_ok: Dict[str, bool] = {}
        self._data_stale: bool = False

        # 企业级趋势行情判断参数（可从 config market_regime 覆盖）
        self._trend_adx_period = config.get("market_regime", {}).get("trend_adx_period", 14)
        self._trend_adx_floor = config.get("market_regime", {}).get("trend_adx_floor", 15)
        self._trend_adx_saturation = config.get("market_regime", {}).get("trend_adx_saturation", 40)
        self._trend_mtf_weight = config.get("market_regime", {}).get("trend_mtf_weight", 0.4)

        # 危机阈值（可从 config market_regime 覆盖）
        self._extreme_vol_threshold = float(config.get("market_regime", {}).get("extreme_vol_threshold", 0.6))
        self._funding_crush_threshold = float(config.get("market_regime", {}).get("funding_crush_threshold", 0.5))
        self._liquidity_crisis_threshold = float(config.get("market_regime", {}).get("liquidity_crisis_threshold", -0.5))

        # 多币种支持
        self._watched_symbols: List[str] = config.get("market_regime", {}).get(
            "watched_symbols", ["BTC-USDT-SWAP"]
        )
        self._base_symbol: str = "BTC-USDT-SWAP"  # 基准币种（用于整体市场状态判断）
        # 各币种的独立市场状态
        self._symbol_regimes: Dict[str, Dict[str, Any]] = {}
        self._symbol_factor_scores: Dict[str, Dict[str, float]] = {}
        # 各币种的 ADX/DI 趋势明细（单边检测/逆势确认的统一数据源，随趋势评分更新）
        self._symbol_trend_detail: Dict[str, Dict[str, Any]] = {}
        
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

        # P3: Regime 分类准确率验证
        self._regime_predictions: List[Dict[str, Any]] = []  # 历史预测记录
        self._regime_accuracy_window = int(config.get("market_regime", {}).get("regime_accuracy_window", 20))  # 验证窗口（预测后N个周期）
        self._regime_accuracy_enabled = config.get("market_regime", {}).get("regime_accuracy_validation", True)
        self._last_accuracy_check: Optional[datetime] = None
        
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

    def set_alert_manager(self, alert_manager):
        """设置告警管理器"""
        self._alert_manager = alert_manager

    async def _regime_update_loop(self):
        """定期更新市场状态"""
        while self._running:
            try:
                await self._update_regime()
            except Exception as e:
                logger.error(f"Regime update error: {e}")
                if self._alert_manager:
                    try:
                        await self._alert_manager.send_alert(
                            alert_type="regime_engine_error",
                            message=f"市场状态引擎更新循环异常（策略可能使用过时状态数据）: {e}",
                            severity="CRITICAL",
                            symbol=self._base_symbol,
                            metadata={"error": str(e)},
                        )
                    except Exception:
                        pass
            await asyncio.sleep(self._update_interval)

    async def _update_regime(self):
        """融合所有因子，输出最终状态（支持多币种）"""
        await self._collect_factor_data()

        if self.okx_client:
            self._data_stale = not self._data_collection_ok.get(self._base_symbol, False)
            if self._data_stale:
                logger.warning(f"[REGIME] {self._base_symbol} market data incomplete; keeping previous regime")
                return

        # P2: 并行计算所有币种（含 base symbol）的因子得分，消除串行瓶颈
        async def _compute_symbol_regime(symbol):
            if self.okx_client and not self._data_collection_ok.get(symbol, False):
                return None
            try:
                raw_symbol_scores = await self._calculate_factor_scores(symbol)
                symbol_scores = self._apply_smoothing(symbol, raw_symbol_scores)
                prev_sym = self._symbol_regimes.get(symbol, {}).get("regime")
                prev_sym_regime = MarketRegime(prev_sym) if prev_sym else None
                sym_regime, sym_subtype, sym_strength, sym_conf, _ = self._resolve_regime(
                    symbol_scores, previous_regime=prev_sym_regime
                )
                return {
                    "symbol": symbol,
                    "scores": symbol_scores,
                    "regime": sym_regime.value,
                    "subtype": sym_subtype.value,
                    "strength": sym_strength,
                    "confidence": sym_conf,
                }
            except Exception as e:
                logger.debug(f"Failed to compute regime for {symbol}: {e}")
                return None

        results = await asyncio.gather(
            *[_compute_symbol_regime(s) for s in self._watched_symbols],
            return_exceptions=True,
        )

        # 提取 base symbol 结果作为主市场状态
        base_result = None
        for result in results:
            if isinstance(result, Exception):
                logger.debug(f"Symbol regime computation failed: {result}")
            elif result is not None and result["symbol"] == self._base_symbol:
                base_result = result
                break

        if base_result is None:
            logger.warning(f"[REGIME] {self._base_symbol} regime computation failed; keeping previous regime")
            return

        factor_scores = base_result["scores"]
        regime = MarketRegime(base_result["regime"])
        subtype = RegimeSubtype(base_result["subtype"])
        strength = base_result["strength"]
        confidence = base_result["confidence"]

        self._factor_scores = factor_scores

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
        logger.debug(f"[REGIME] Breakdown: factor_scores={factor_scores}")

        # 存储所有币种的 regime 结果
        for result in results:
            if isinstance(result, Exception) or result is None:
                continue
            symbol = result["symbol"]
            self._symbol_factor_scores[symbol] = result["scores"]
            self._symbol_regimes[symbol] = {
                "regime": result["regime"],
                "subtype": result["subtype"],
                "strength": result["strength"],
                "confidence": result["confidence"],
                "factor_scores": result["scores"],
            }

        # P3: 记录预测并验证历史准确率
        if self._regime_accuracy_enabled:
            self._record_regime_prediction()
            await self._validate_regime_predictions()

    def _record_regime_prediction(self):
        """记录当前 regime 预测，用于后续准确率验证"""
        if not self._current_regime:
            return

        prediction = {
            "timestamp": datetime.now(),
            "regime": self._current_regime.value,
            "confidence": self._regime_confidence,
            "price": self._orderbook_cache.get(self._base_symbol, {}).get("last_price", 0),
            "validated": False,
            "correct": None,
        }
        self._regime_predictions.append(prediction)

        # 限制历史记录长度
        max_predictions = self._regime_accuracy_window * 3
        if len(self._regime_predictions) > max_predictions:
            self._regime_predictions = self._regime_predictions[-max_predictions:]

    async def _validate_regime_predictions(self):
        """验证历史 regime 预测的准确率"""
        now = datetime.now()

        # 每小时验证一次即可，避免频繁计算
        if self._last_accuracy_check and (now - self._last_accuracy_check).total_seconds() < 3600:
            return

        validated_count = 0
        correct_count = 0

        for pred in self._regime_predictions:
            if pred["validated"]:
                continue

            # 需要等待足够时间才能验证（至少 regime_accuracy_window 个周期）
            age_cycles = (now - pred["timestamp"]).total_seconds() / self._update_interval
            if age_cycles < self._regime_accuracy_window:
                continue

            # 获取预测时的价格和当前价格
            pred_price = pred["price"]
            if pred_price <= 0:
                pred["validated"] = True
                continue

            current_price = self._orderbook_cache.get(self._base_symbol, {}).get("last_price", 0)
            if current_price <= 0:
                continue

            price_change_pct = (current_price - pred_price) / pred_price
            pred_regime = pred["regime"]

            # 判断预测是否正确
            correct = False
            if pred_regime == "trend_bullish" and price_change_pct > 0.02:
                correct = True
            elif pred_regime == "trend_bearish" and price_change_pct < -0.02:
                correct = True
            elif pred_regime == "range_bound" and abs(price_change_pct) < 0.03:
                correct = True
            elif pred_regime == "breakout" and price_change_pct > 0.05:
                correct = True
            elif pred_regime == "breakdown" and price_change_pct < -0.05:
                correct = True
            elif pred_regime == "extreme_volatility":
                # 极端波动预测：检查后续是否有大幅波动
                correct = abs(price_change_pct) > 0.05
            else:
                # 其他状态：宽松判断
                correct = abs(price_change_pct) < 0.10

            pred["validated"] = True
            pred["correct"] = correct
            pred["actual_price_change"] = price_change_pct
            validated_count += 1
            if correct:
                correct_count += 1

        if validated_count > 0:
            accuracy = correct_count / validated_count
            logger.info(
                f"[REGIME_ACCURACY] Validated {validated_count} predictions, "
                f"accuracy={accuracy:.2%} ({correct_count}/{validated_count})"
            )

        self._last_accuracy_check = now

    def get_regime_accuracy_stats(self) -> Dict[str, Any]:
        """获取 regime 分类准确率统计"""
        validated = [p for p in self._regime_predictions if p["validated"]]
        if not validated:
            return {"total_predictions": len(self._regime_predictions), "validated": 0, "accuracy": None}

        correct = sum(1 for p in validated if p["correct"])
        accuracy = correct / len(validated) if validated else 0

        # 按 regime 类型分组统计
        by_regime = {}
        for pred in validated:
            regime = pred["regime"]
            if regime not in by_regime:
                by_regime[regime] = {"total": 0, "correct": 0}
            by_regime[regime]["total"] += 1
            if pred["correct"]:
                by_regime[regime]["correct"] += 1

        for regime, stats in by_regime.items():
            stats["accuracy"] = stats["correct"] / stats["total"] if stats["total"] > 0 else 0

        return {
            "total_predictions": len(self._regime_predictions),
            "validated": len(validated),
            "correct": correct,
            "accuracy": accuracy,
            "by_regime": by_regime,
        }

    async def _collect_factor_data(self):
        """收集各因子数据（支持多币种）"""
        if not self.okx_client:
            return

        self._data_collection_ok = {symbol: False for symbol in self._watched_symbols}
        try:
            # 并行获取所有监控币种的数据
            results = await asyncio.gather(
                *[self._collect_symbol_data(symbol) for symbol in self._watched_symbols],
                return_exceptions=True,
            )
            for symbol, result in zip(self._watched_symbols, results):
                if isinstance(result, Exception):
                    logger.debug(f"Failed to collect data for {symbol}: {result}")
                else:
                    self._data_collection_ok[symbol] = result
        except Exception as e:
            logger.debug(f"Failed to collect factor data: {e}")

    async def _collect_symbol_data(self, symbol: str) -> bool:
        """收集单个币种的因子数据"""
        try:
            # 并行获取所有 API 数据（带超时保护）
            async def _get_ticker():
                if hasattr(self.okx_client, 'get_ticker_async'):
                    return await self.okx_client.get_ticker_async(symbol)
                return self.okx_client.get_ticker(symbol)
            
            async def _get_funding():
                if hasattr(self.okx_client, 'get_funding_rate_async'):
                    return await self.okx_client.get_funding_rate_async(symbol)
                return self.okx_client.get_funding_rate(symbol)
            
            async def _get_kline_1h():
                if hasattr(self.okx_client, 'get_kline_async'):
                    return await self.okx_client.get_kline_async(symbol, "1H", limit=60)
                return self.okx_client.get_kline(symbol, "1H", limit=60)
            
            async def _get_kline_4h():
                if hasattr(self.okx_client, 'get_kline_async'):
                    return await self.okx_client.get_kline_async(symbol, "4H", limit=60)
                return self.okx_client.get_kline(symbol, "4H", limit=60)
            
            ticker, funding_info, klines_1h, klines_4h = await asyncio.wait_for(
                asyncio.gather(
                    _get_ticker(),
                    _get_funding(),
                    _get_kline_1h(),
                    _get_kline_4h(),
                ),
                timeout=10.0,
            )
            
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
            
            if funding_info:
                funding_rate_val = float(funding_info.get("fundingRate", 0))
                self._funding_cache[symbol] = {
                    "funding_rate": funding_rate_val,
                    "next_funding_time": funding_info.get("nextFundingTime"),
                    "funding_rate_24h_avg": float(funding_info.get("fundingRate24hAvg", 0)),
                }
                self._funding_cache_time[symbol] = datetime.now()
                funding_available = True
            else:
                last_time = self._funding_cache_time.get(symbol)
                funding_age = (datetime.now() - last_time).total_seconds() if last_time else None
                funding_available = bool(
                    symbol in self._funding_cache
                    and funding_age is not None
                    and funding_age <= self._funding_cache_ttl
                )
                if not funding_available:
                    logger.warning(f"{symbol} funding rate data stale ({self._funding_cache_ttl}s+), using fallback")
            
            if klines_1h:
                # OKX history-candles 倒序返回（最新在前），按时间戳正序排序，
                # 否则 compute_trend_vote 的 EMA/斜率/结构判断会整体反转（牛市判熊市）
                klines_1h = self._sort_klines_ascending(klines_1h)
                self._kline_cache[f"{symbol}_1H"] = klines_1h
            
            if klines_4h:
                klines_4h = self._sort_klines_ascending(klines_4h)
                self._kline_cache[f"{symbol}_4H"] = klines_4h

            if self._detector is not None and klines_1h:
                try:
                    await self._detector.detect_regime(
                        klines_1h,
                        symbol,
                        timeframe="medium",
                        timeframe_data={
                            **({"medium": klines_1h} if klines_1h else {}),
                            **({"long": klines_4h} if klines_4h else {}),
                        },
                    )
                except Exception as detector_error:
                    logger.debug(f"Detector update failed for {symbol}: {detector_error}")

            return bool(ticker and funding_available and klines_1h and klines_4h)
        except asyncio.TimeoutError:
            logger.warning(f"API timeout collecting data for {symbol} (>10s), skipping this cycle")
            return False
        except Exception as e:
            logger.debug(f"Failed to collect data for {symbol}: {e}")
            return False

    async def _calculate_factor_scores(self, symbol: str = None) -> Dict[str, float]:
        """计算各因子得分（-1 到 +1），支持指定币种"""
        if symbol is None:
            symbol = self._base_symbol
        
        scores = {}

        trend, vol, funding, liq, mom, sent = await asyncio.gather(
            self._calculate_trend_score(symbol),
            self._calculate_volatility_score(symbol),
            self._calculate_funding_score(symbol),
            self._calculate_liquidity_score(symbol),
            self._calculate_momentum_score(symbol),
            self._calculate_sentiment_score(symbol),
        )
        scores["trend"] = trend
        scores["volatility"] = vol
        scores["funding_rate"] = funding
        scores["liquidity"] = liq
        scores["momentum"] = mom
        scores["sentiment"] = sent

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

            # 缓存 ADX/DI 明细，供框架层趋势确认因子（单边检测/背离/逆势）统一取用，
            # 避免智能体/过滤链各自重复计算 ADX 导致口径漂移。
            votes = vote.get("votes") or {}
            self._symbol_trend_detail[symbol] = {
                "adx": float(vote.get("adx", 0.0)),
                "adx_strength": float(vote.get("adx_strength", 0.0)),
                "direction": float(vote.get("direction", 0.0)),
                "di_dir": float(votes.get("di", 0.0)),
                "ema_dir": float(votes.get("ema", 0.0)),
                "slope_dir": float(votes.get("slope", 0.0)),
                "structure_dir": float(votes.get("structure", 0.0)),
            }
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
        """资金费率得分：+1 极端正（多头付空头），-1 极端负（空头付多头）

        强化：瞬时费率与 24h 均值加权（0.6 瞬时 + 0.4 均值），抑制单次快照噪声，
        避免资金费率单点抖动导致 FUNDING_CRUSH 状态误触发。
        """
        if symbol is None:
            symbol = self._base_symbol
        funding = self._funding_cache.get(symbol, {})
        rate = funding.get("funding_rate", 0) or 0.0
        avg = funding.get("funding_rate_24h_avg", 0) or 0.0

        # 无均值数据（首次/降级）时退化为纯瞬时
        blended = rate if avg == 0.0 else rate * 0.6 + avg * 0.4

        if blended > 0.001:
            return min(1.0, blended * 500)
        elif blended < -0.001:
            return max(-1.0, blended * 500)

        return 0.0

    async def _calculate_liquidity_score(self, symbol: str = None) -> float:
        """流动性得分：+1 高流动性，-1 低流动性

        强化：价差为主（0.8）+ 24h 成交量微调（0.2），避免低成交时段价差窄但无量
        被误判为高流动性，减少 LIQUIDITY_CRISIS 误触发。
        """
        if symbol is None:
            symbol = self._base_symbol
        ob = self._orderbook_cache.get(symbol, {})
        last_price = ob.get("last_price", 0) or 0.0
        bid = ob.get("bid_price", 0) or 0.0
        ask = ob.get("ask_price", 0) or 0.0
        volume = ob.get("volume_24h", 0) or 0.0

        if last_price == 0 or bid == 0 or ask == 0:
            return 0.0

        spread = (ask - bid) / last_price

        # 价差得分（-1 到 +1）
        if spread < 0.0001:
            spread_score = 1.0
        elif spread > 0.001:
            spread_score = -1.0
        else:
            spread_score = max(-1.0, min(1.0, (0.0001 - spread) / 0.0009))

        # 成交量得分：log10 压缩，1e8 量级封顶为 1.0（无严格阈值，仅作软微调）
        volume_score = 0.0
        if volume > 0:
            volume_score = min(1.0, max(0.0, np.log10(volume + 1.0) / 8.0))

        return max(-1.0, min(1.0, spread_score * 0.8 + volume_score * 0.2))

    async def _calculate_momentum_score(self, symbol: str = None) -> float:
        """动量得分：+1 强上涨动量，-1 强下跌动量

        强化：多周期动量加权（短周期近5根 vs 前5根 0.6 + 长周期近10根 vs 前10根 0.4），
        相比单一 10 根窗口更稳健，减少单根 K 线异常对动量的扰动。
        """
        if symbol is None:
            symbol = self._base_symbol
        klines = self._kline_cache.get(f"{symbol}_1H", [])
        if len(klines) < 21:
            return 0.0

        try:
            # K线数据兼容 list 和 dict 格式
            if isinstance(klines[0], dict):
                prices = [float(k.get("close", 0)) for k in klines]
            else:
                prices = [float(k[4]) for k in klines]

            if len(prices) < 21:
                return 0.0

            # 短周期动量（近5根 vs 前5根，加速信号）
            short_recent = prices[-1] / prices[-6] - 1 if prices[-6] > 0 else 0.0
            short_earlier = prices[-6] / prices[-11] - 1 if prices[-11] > 0 else 0.0
            # 长周期动量（近10根 vs 前10根）
            long_recent = prices[-1] / prices[-11] - 1 if prices[-11] > 0 else 0.0
            long_earlier = prices[-11] / prices[-21] - 1 if prices[-21] > 0 else 0.0

            momentum = (short_recent - short_earlier) * 0.6 + (long_recent - long_earlier) * 0.4
            return max(-1.0, min(1.0, momentum * 20))
        except Exception as e:
            logger.debug(f"Momentum score calc failed: {e}")
            return 0.0

    async def _calculate_sentiment_score(self, symbol: str = None) -> float:
        """情绪得分：综合价格位置、成交量压力、资金费率

        多因子融合避免单一价格位置误判（如高位缩量被误判为乐观）。
        """
        if symbol is None:
            symbol = self._base_symbol
        ob = self._orderbook_cache.get(symbol, {})
        last = ob.get("last_price", 0)
        high = ob.get("high_24h", 0)
        low = ob.get("low_24h", 0)
        volume = ob.get("volume_24h", 0) or 0.0
        prev_volume = ob.get("volume_24h_prev", 0) or 0.0

        # 价格位置得分（-1 到 +1）
        price_score = 0.0
        if high > low:
            position = (last - low) / (high - low)
            if position > 0.8:
                price_score = min(1.0, (position - 0.5) * 2)
            elif position < 0.2:
                price_score = max(-1.0, -(0.5 - position) * 2)

        # 成交量压力得分（放量上涨为正，放量下跌为负）
        volume_score = 0.0
        if prev_volume > 0 and volume > 0:
            vol_ratio = volume / prev_volume
            mid = (high + low) / 2 if (high + low) > 0 else last
            direction = 1.0 if last > mid else (-1.0 if last < mid else 0.0)
            if vol_ratio > 1.2:
                volume_score = direction * min(1.0, (vol_ratio - 1.0))

        # 资金费率得分（正费率=多头付费=过热，负费率=空头付费=恐慌）
        funding_score = 0.0
        funding = self._funding_cache.get(symbol, {})
        rate = funding.get("funding_rate", 0) or 0.0
        if abs(rate) > 0.0001:
            funding_score = max(-1.0, min(1.0, rate * 1000))

        # 融合：价格位置 0.5 + 成交量压力 0.3 + 资金费率 0.2
        score = price_score * 0.5 + volume_score * 0.3 + funding_score * 0.2
        return max(-1.0, min(1.0, score))

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

        关键修正：
        1) 兼容并保留旧的 trend/range/high-risk 口径；
        2) 增强 breakout / breakdown / reversal 三种关键市场状态；
        3) 保持 hysteresis 迟滞，避免状态在边界前后抖动。
        """
        weighted_scores = {}
        total_weight = sum(self._factor_weights.values())

        for factor, score in factor_scores.items():
            weight = self._factor_weights.get(factor, 0)
            weighted_scores[factor] = score * weight / total_weight

        raw_trend = float(factor_scores.get("trend", 0.0))
        raw_vol = float(factor_scores.get("volatility", 0.0))
        raw_funding = float(factor_scores.get("funding_rate", 0.0))
        raw_liquidity = float(factor_scores.get("liquidity", 0.0))
        raw_momentum = float(factor_scores.get("momentum", 0.0))
        directional_factors = ("trend", "momentum", "sentiment")
        active_directional_factors = [
            factor for factor in directional_factors
            if self._factor_weights[factor] > 0.0 and float(factor_scores.get(factor, 0.0)) != 0.0
        ]
        directional_weight = sum(self._factor_weights[factor] for factor in active_directional_factors)
        directional_score = (
            sum(
                float(factor_scores.get(factor, 0.0)) * self._factor_weights[factor]
                for factor in active_directional_factors
            ) / directional_weight
            if directional_weight > 0.0 else raw_trend
        )

        breakdown = {k: round(v, 4) for k, v in weighted_scores.items()}

        hyst = max(0.0, self._regime_hysteresis_margin)
        prev = previous_regime

        # 高优先级危机状态（迟滞）
        ev_t = self._extreme_vol_threshold
        if raw_vol > (ev_t - hyst if prev == MarketRegime.EXTREME_VOLATILITY else ev_t + hyst):
            return self._build_result(
                MarketRegime.EXTREME_VOLATILITY,
                MarketSubtype.STRONG if raw_vol > 0.8 else MarketSubtype.MODERATE,
                min(1.0, raw_vol * 1.2),
                0.7 + raw_vol * 0.3,
                breakdown,
            )

        af = abs(raw_funding)
        fc_t = self._funding_crush_threshold
        if af > (fc_t - hyst if prev == MarketRegime.FUNDING_CRUSH else fc_t + hyst):
            return self._build_result(
                MarketRegime.FUNDING_CRUSH,
                MarketSubtype.STRONG if af > 0.7 else MarketSubtype.MODERATE,
                min(1.0, af * 1.5),
                0.6 + af * 0.4,
                breakdown,
            )

        lc_t = self._liquidity_crisis_threshold
        if raw_liquidity < (lc_t + hyst if prev == MarketRegime.LIQUIDITY_CRISIS else lc_t - hyst):
            return self._build_result(
                MarketRegime.LIQUIDITY_CRISIS,
                MarketSubtype.WEAK,
                min(1.0, abs(raw_liquidity) * 1.0),
                0.7 + abs(raw_liquidity) * 0.3,
                breakdown,
            )

        # breakouts / breakdowns / reversals：仅在非高危状态下启用，覆盖“趋势延续”和
        # “反转/突破”这两类关键市场状态。
        breakout_strength = max(directional_score, raw_momentum)
        breakdown_strength = max(-directional_score, -raw_momentum)
        reversal_strength = abs(raw_trend - raw_momentum)

        breakout_threshold = 0.68 + hyst
        breakdown_threshold = 0.68 + hyst
        reversal_threshold = 0.65 + hyst

        if breakout_strength > breakout_threshold and directional_score > 0.18:
            subtype = MarketSubtype.STRONG if breakout_strength > 0.7 else MarketSubtype.MODERATE
            strength = min(1.0, breakout_strength * 1.15)
            return self._build_result(
                MarketRegime.BREAKOUT,
                subtype,
                strength,
                0.7 + breakout_strength * 0.25,
                breakdown,
            )

        if breakdown_strength > breakdown_threshold and directional_score < -0.18:
            subtype = MarketSubtype.STRONG if breakdown_strength > 0.7 else MarketSubtype.MODERATE
            strength = min(1.0, breakdown_strength * 1.15)
            return self._build_result(
                MarketRegime.BREAKDOWN,
                subtype,
                strength,
                0.7 + breakdown_strength * 0.25,
                breakdown,
            )

        if reversal_strength > reversal_threshold and abs(raw_trend) < 0.35:
            subtype = MarketSubtype.REVERSAL
            strength = min(1.0, reversal_strength * 1.0)
            return self._build_result(
                MarketRegime.REVERSAL,
                subtype,
                strength,
                0.65 + reversal_strength * 0.25,
                breakdown,
            )

        # 趋势 vs 震荡（迟滞：bullish/bearish 在退出阈值带内保持上一状态）
        was_bullish = prev == MarketRegime.TREND_BULLISH
        was_bearish = prev == MarketRegime.TREND_BEARISH

        is_bullish = directional_score > (0.2 - hyst if was_bullish else 0.2 + hyst)
        is_bearish = directional_score < (-0.2 + hyst if was_bearish else -0.2 - hyst)

        # 动量作为趋势确认：同向动量增强强度，背离动量衰减强度，不改分类方向
        if is_bullish:
            if directional_score > 0.5:
                subtype = MarketSubtype.STRONG
            elif directional_score > 0.35:
                subtype = MarketSubtype.MODERATE
            else:
                subtype = MarketSubtype.WEAK
            mom_align = 1.0 if raw_momentum >= 0 else 0.5
            strength = min(1.0, directional_score * 1.5 * (0.8 + 0.2 * mom_align))
            return self._build_result(
                MarketRegime.TREND_BULLISH,
                subtype,
                strength,
                0.6 + directional_score * 0.4,
                breakdown,
            )
        if is_bearish:
            if directional_score < -0.5:
                subtype = MarketSubtype.STRONG
            elif directional_score < -0.35:
                subtype = MarketSubtype.MODERATE
            else:
                subtype = MarketSubtype.WEAK
            mom_align = 1.0 if raw_momentum <= 0 else 0.5
            strength = min(1.0, abs(directional_score) * 1.5 * (0.8 + 0.2 * mom_align))
            return self._build_result(
                MarketRegime.TREND_BEARISH,
                subtype,
                strength,
                0.6 + abs(directional_score) * 0.4,
                breakdown,
            )

        return self._build_result(
            MarketRegime.RANGE_BOUND,
            MarketSubtype.MODERATE,
            0.3 + abs(directional_score) * 0.5,
            0.5 + abs(directional_score) * 0.3,
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

    def set_detector(self, detector):
        """注入细粒度检测器，供主引擎优先使用其 breakout/reversal 结果。"""
        self._detector = detector

    def set_regime_arbiter(self, arbiter):
        """注入融合仲裁器，主引擎直接走统一 regime 输出。"""
        self._regime_arbiter = arbiter

    def _coerce_detector_regime(self, detector_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """将检测器的 regime 输出归一到主引擎 schema。"""
        if not isinstance(detector_result, dict) or not detector_result.get("regime"):
            return None

        raw_regime = str(detector_result.get("regime", "unknown")).lower()
        aliases = {
            "trending_up": "trend_bullish",
            "trending_down": "trend_bearish",
            "ranging": "range_bound",
            "high_volatility": "extreme_volatility",
            "low_volatility": "range_bound",
            "breakout": "breakout",
            "breakdown": "breakdown",
            "reversal": "reversal",
            "trend_bullish": "trend_bullish",
            "trend_bearish": "trend_bearish",
            "range_bound": "range_bound",
            "extreme_volatility": "extreme_volatility",
            "funding_crush": "funding_crush",
            "liquidity_crisis": "liquidity_crisis",
        }
        regime = aliases.get(raw_regime, raw_regime)

        probabilities = detector_result.get("probabilities") or {}
        detector_conf = 0.0
        if isinstance(probabilities, dict):
            detector_conf = max(
                [float(v) for v in probabilities.values() if isinstance(v, (int, float))],
                default=0.0,
            )
        else:
            detector_conf = float(detector_result.get("confidence", 0.0) or 0.0)

        if isinstance(probabilities, dict) and regime in probabilities:
            detector_conf = float(probabilities.get(regime, detector_conf))

        return {
            "regime": regime,
            "normalized_regime": regime,
            "state": regime,
            "subtype": detector_result.get("subtype") or ("reversal" if regime == "reversal" else "moderate"),
            "strength": float(detector_result.get("strength", detector_conf)),
            "confidence": float(detector_result.get("confidence", detector_conf)),
            "detector_raw": detector_result,
            "probabilities": probabilities if isinstance(probabilities, dict) else {},
            "detector_reversal_prob": (
                float(probabilities.get("reversal", 0.0))
                if isinstance(probabilities, dict)
                else 0.0
            ),
            "factor_scores": detector_result.get("probabilities", {}),
            "factor_weights": self._factor_weights,
            "last_update": detector_result.get("detected_at") or datetime.now().isoformat(),
        }

    def _get_fused_regime(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """优先消费 arbiter / detector 的细粒度输出；若不可用再回退主引擎。"""
        if self._regime_arbiter is not None:
            try:
                arbiter_result = self._regime_arbiter.arbitrate(symbol) if symbol else self._regime_arbiter.arbitrate()
                if isinstance(arbiter_result, dict) and arbiter_result.get("regime"):
                    return arbiter_result
            except Exception as exc:
                logger.debug(f"RegimeArbiter refused fused regime: {exc}")

        if self._detector is not None:
            try:
                get_regime = getattr(self._detector, "get_regime", None)
                if callable(get_regime):
                    detector_result = get_regime(symbol) if symbol else get_regime()
                    coerced = self._coerce_detector_regime(detector_result)
                    if coerced is not None:
                        return coerced
            except Exception as exc:
                logger.debug(f"Detector regime fusion failed: {exc}")

        return None

    def get_regime(self, include_fusion: bool = True) -> Dict[str, Any]:
        """获取当前市场状态。优先融合细粒度 detector/arbiter 结果。"""
        fused = self._get_fused_regime() if include_fusion else None
        if fused is not None:
            fused["data_stale"] = fused.get("data_stale", False) or self._data_stale
            return fused

        regime = self._current_regime.value if self._current_regime else "unknown"
        return {
            "regime": regime,
            "normalized_regime": regime,
            "state": regime,
            "subtype": self._current_subtype.value if self._current_subtype else "unknown",
            "strength": self._current_strength,
            "confidence": self._regime_confidence,
            "factor_scores": self._factor_scores,
            "factor_weights": self._factor_weights,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "data_stale": self._data_stale,
        }

    def get_symbol_regime(self, symbol: str, include_fusion: bool = True) -> Dict[str, Any]:
        """获取指定币种的独立市场状态数据。优先消费细粒度状态。"""
        fused = self._get_fused_regime(symbol) if include_fusion else None
        if fused is not None:
            fused["symbol_specific"] = True
            fused["data_stale"] = fused.get("data_stale", False) or self._data_collection_ok.get(symbol) is False
            return fused

        # 优先返回独立计算的币种状态
        if symbol in self._symbol_regimes:
            sym_data = self._symbol_regimes[symbol]
            result = {
                "regime": sym_data["regime"],
                "normalized_regime": sym_data["regime"],
                "state": sym_data["regime"],
                "strength": sym_data["strength"],
                "confidence": sym_data["confidence"],
                "volatility": sym_data.get("factor_scores", {}).get("volatility", 0.02),
                "volatility_percentile": sym_data.get("factor_scores", {}).get("volatility", 0.0),
                "momentum": sym_data.get("factor_scores", {}).get("momentum", 0.0),
                "trend_strength": sym_data.get("factor_scores", {}).get("trend", 0.0),
                "liquidity_score": sym_data.get("factor_scores", {}).get("liquidity", 0.0),
                "funding_score": sym_data.get("factor_scores", {}).get("funding_rate", 0.0),
                "data_stale": self._data_collection_ok.get(symbol) is False,
                "symbol_specific": True,
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
        regime = self._current_regime.value if self._current_regime else "unknown"
        result = {
            "regime": regime,
            "normalized_regime": regime,
            "state": regime,
            "strength": self._current_strength,
            "confidence": self._regime_confidence,
            "volatility": self._factor_scores.get("volatility", 0.02),
            "volatility_percentile": self._factor_scores.get("volatility", 0.0),
            "momentum": self._factor_scores.get("momentum", 0.0),
            "trend_strength": self._factor_scores.get("trend", 0.0),
            "liquidity_score": self._factor_scores.get("liquidity", 0.0),
            "data_stale": self._data_collection_ok.get(symbol) is False,
            "symbol_specific": False,
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

    def get_symbol_trend_detail(self, symbol: str) -> Optional[Dict[str, Any]]:
        """返回币种 ADX/DI 趋势明细（单边检测/背离/逆势确认的统一数据源）。

        由 _calculate_trend_score 在每次趋势评分时缓存；无缓存（币种未被监控或
        数据不足）返回 None。字段：adx/adx_strength/direction/di_dir/ema_dir/slope_dir/structure_dir。
        """
        return self._symbol_trend_detail.get(symbol)

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
        try:
            regime_info = self.get_regime()
        except Exception:
            regime_info = {}
        regime_name = str(
            regime_info.get(
                "regime",
                self._current_regime.value if self._current_regime else "unknown",
            )
        ).lower()
        try:
            current_strength = float(regime_info.get("strength", self._current_strength))
        except (TypeError, ValueError):
            current_strength = self._current_strength
        current_strength = max(0.0, min(1.0, current_strength))
        subtype = regime_info.get("subtype", self._current_subtype)
        if isinstance(subtype, Enum):
            subtype = subtype.value

        if regime_name in ("", "unknown"):
            return {
                "overall": 1.0, "trend": 1.0, "scalping": 1.0,
                "grid": 1.0, "arbitrage": 1.0, "spot_grid": 1.0,
                "spot_martingale": 1.0,
            }
        
        adjustment = {"overall": 1.0}
        
        if regime_name == MarketRegime.EXTREME_VOLATILITY.value:
            adjustment["overall"] = max(0.3, 1.0 - current_strength * 0.5)
            adjustment["trend"] = adjustment["overall"] * 0.5
            adjustment["scalping"] = adjustment["overall"] * 0.8
            adjustment["grid"] = adjustment["overall"] * 0.6
            adjustment["arbitrage"] = adjustment["overall"] * 0.9
        elif regime_name == MarketRegime.FUNDING_CRUSH.value:
            adjustment["overall"] = 0.7
            adjustment["trend"] = 0.5 if subtype == MarketSubtype.STRONG.value else 0.7
            adjustment["scalping"] = 0.9
            adjustment["grid"] = 0.6
            adjustment["arbitrage"] = 1.0
        elif regime_name == MarketRegime.LIQUIDITY_CRISIS.value:
            adjustment["overall"] = max(0.2, 1.0 - current_strength * 0.6)
            adjustment["trend"] = adjustment["overall"] * 0.4
            adjustment["scalping"] = adjustment["overall"] * 0.5
            adjustment["grid"] = adjustment["overall"] * 0.6
            adjustment["arbitrage"] = adjustment["overall"] * 0.3
        elif regime_name == MarketRegime.TREND_BULLISH.value:
            adjustment["overall"] = 1.0 + current_strength * 0.2
            adjustment["trend"] = 1.2 if subtype == MarketSubtype.STRONG.value else 1.0
            adjustment["scalping"] = 0.9
            adjustment["grid"] = 0.8
            adjustment["arbitrage"] = 0.9
        elif regime_name == MarketRegime.TREND_BEARISH.value:
            adjustment["overall"] = max(0.5, 1.0 - current_strength * 0.3)
            adjustment["trend"] = 0.8 if subtype == MarketSubtype.STRONG.value else 0.9
            adjustment["scalping"] = 0.8
            adjustment["grid"] = 0.7
            adjustment["arbitrage"] = 0.9
        elif regime_name == MarketRegime.RANGE_BOUND.value:
            adjustment["overall"] = 1.0
            adjustment["trend"] = 0.7
            adjustment["scalping"] = 1.1
            adjustment["grid"] = 1.2
            adjustment["arbitrage"] = 1.0
        elif regime_name == MarketRegime.BREAKOUT.value:
            adjustment.update({
                "overall": 0.9, "trend": 1.1, "scalping": 0.85,
                "grid": 0.6, "arbitrage": 1.0, "spot_grid": 0.6,
                "spot_martingale": 0.4,
            })
        elif regime_name == MarketRegime.BREAKDOWN.value:
            adjustment.update({
                "overall": 0.8, "trend": 1.0, "scalping": 0.7,
                "grid": 0.5, "arbitrage": 0.8, "spot_grid": 0.5,
                "spot_martingale": 0.3,
            })
        elif regime_name == MarketRegime.REVERSAL.value:
            adjustment.update({
                "overall": 0.6, "trend": 0.7, "scalping": 0.6,
                "grid": 0.5, "arbitrage": 0.7, "spot_grid": 0.5,
                "spot_martingale": 0.3,
            })
        
        return adjustment

    def get_strategy_recommendation(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略级别的操作建议"""
        adjustment = self.get_position_adjustment()
        strat_adjust = adjustment.get(strategy_name, adjustment["overall"])
        regime_info = self.get_regime()
        regime = regime_info.get(
            "regime",
            self._current_regime.value if self._current_regime else "unknown",
        )
        
        recommendation = {
            "strategy": strategy_name,
            "adjustment_factor": strat_adjust,
            "regime": regime,
            "confidence": regime_info.get("confidence", self._regime_confidence),
            "action": "increase" if strat_adjust > 1.1 else "decrease" if strat_adjust < 0.9 else "maintain",
            "reason": self._get_recommendation_reason(
                strategy_name,
                strat_adjust,
                regime=regime,
                subtype=regime_info.get("subtype"),
            ),
        }
        
        return recommendation

    def _get_recommendation_reason(
        self,
        strategy_name: str,
        factor: float,
        regime: Optional[str] = None,
        subtype: Optional[str] = None,
    ) -> str:
        """生成建议理由"""
        if regime is None and not self._current_regime:
            return "Unknown market regime"
        
        regime = regime or self._current_regime.value
        subtype = subtype or (self._current_subtype.value if self._current_subtype else "unknown")
        if isinstance(regime, Enum):
            regime = regime.value
        if isinstance(subtype, Enum):
            subtype = subtype.value
        
        reasons = {
            ("trend_bullish", "strong", "trend"): "Strong bullish trend favors trend following",
            ("trend_bullish", "strong", "grid"): "Trend may break grid bounds, reduce exposure",
            ("trend_bearish", "strong", "trend"): "Strong bearish trend, reduce long exposure",
            ("range_bound", "moderate", "grid"): "Range bound market is ideal for grid strategy",
            ("range_bound", "moderate", "scalping"): "Sideways movement creates scalping opportunities",
            ("breakout", "moderate", "trend"): "Bullish breakout favors confirmed long trend entries",
            ("breakdown", "moderate", "trend"): "Bearish breakdown favors confirmed short trend entries",
            ("reversal", "reversal", "trend"): "Reversal regime requires reduced trend exposure",
            ("reversal", "reversal", "grid"): "Reversal regime carries elevated mean-reversion risk",
            ("extreme_volatility", "strong", "trend"): "Extreme volatility increases whipsaw risk",
            ("extreme_volatility", "strong", "arbitrage"): "Arbitrage spreads may widen during volatility",
            ("funding_crush", "strong", "grid"): "Extreme funding rates increase holding costs",
            ("liquidity_crisis", "weak", "scalping"): "Low liquidity increases slippage risk",
        }
        
        return reasons.get((regime, subtype, strategy_name), f"{regime} market affects {strategy_name}")
