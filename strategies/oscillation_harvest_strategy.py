"""
企业级震荡收割策略（OscillationHarvestStrategy）

核心思想：
    仅在市场处于 RANGE_BOUND（震荡区间）状态下，识别品种的支撑/阻力区间，
    于支撑位做多、阻力位做空，回归区间中轨止盈，实现震荡行情的反复收割。

企业级能力：
    1. 双重状态过滤：品种级 regime == range_bound 且 全局(BTC)状态非极端风险。
    2. 支撑/阻力区间识别：滚动高低点 + 最小区间宽度（≥3x双边手续费）+ 触及次数验证过滤假区间。
    3. 高周期趋势确认：高周期 EMA20/EMA50 偏离超阈值视为强趋势，放弃震荡开仓避免区间突破。
    4. RSI 超卖/超买 + 布林带 %B 双确认：避免在转折前逆势接飞刀，边界自适应波动。
    5. ATR 止损：止损置于区间外侧，容忍区间内正常噪声。
    6. 手续费覆盖校验：预期盈利必须 > 1.5x 往返手续费。
    7. 品种冷却：单品种开仓后冷却，防止区间内过度交易。
    8. 全链路 traceID：信号源头生成、开仓/平仓复用，贯穿 信号→裁决→订单→记账。
    9. 状态持久化 / 指标埋点 / 异常分级：复用 TrendStrategyBase + EnterpriseStrategyMixin。

标的：tier1 + tier2 主流永续合约（BTC/ETH/SOL/XRP/DOGE/ADA/AVAX/NEAR/APT/SUI/ARB/OP）。
运行方式：默认禁用，先回测/纸面验证，验证通过后再手动开启。
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from strategies._trend_base import TrendStrategyBase
from strategies.trend_sub_strategies import _atr


class OscillationHarvestStrategy(TrendStrategyBase):
    STRATEGY_KEY = "oscillation_harvest"
    DISPLAY_NAME = "震荡收割"
    SYMBOL_SCOPE = "tier1_tier2"
    DEFAULT_BAR = "1H"
    ENTRY_SIGNAL_TYPE = "oscillation_harvest_entry"
    EXIT_SIGNAL_TYPE = "oscillation_harvest_exit"

    def __init__(self, config, okx_client, redis_cache):
        super().__init__(config, okx_client, redis_cache)

        # 支撑/阻力区间识别（回测：1H + lookback=96 为最优正期望组合）
        self._lookback_bars = self._safe_int(self._cfg.get("lookback_bars", 96), 96)
        self._support_band_pct = self._safe_float(self._cfg.get("support_band_pct", 0.01), 0.01)
        # 止盈目标：回归区间中轨的比例（1.0=中轨，0.5=半程）
        self._mid_tp_ratio = self._safe_float(self._cfg.get("mid_tp_ratio", 1.0), 1.0)
        # 区间最小宽度（相对中轨价格），低于此值视为无效震荡（磨损）
        self._min_band_width_pct = self._safe_float(self._cfg.get("min_band_width_pct", 0.003), 0.003)

        # RSI 确认
        self._rsi_period = self._safe_int(self._cfg.get("rsi_period", 14), 14)
        self._rsi_oversold = self._safe_float(self._cfg.get("rsi_oversold", 30.0), 30.0)
        self._rsi_overbought = self._safe_float(self._cfg.get("rsi_overbought", 70.0), 70.0)

        # ATR 止损
        self._atr_sl_mult = self._safe_float(self._cfg.get("atr_sl_mult", 1.5), 1.5)

        # 状态过滤与冷却
        self._min_regime_confidence = self._safe_float(self._cfg.get("min_regime_confidence", 0.4), 0.4)
        self._symbol_cooldown_seconds = self._safe_float(self._cfg.get("symbol_cooldown_seconds", 300.0), 300.0)

        # 高周期趋势确认：高周期若强趋势则放弃震荡开仓（避免区间即将突破）
        self._confirm_bar = str(self._cfg.get("confirm_bar", "4H"))
        self._confirm_trend_threshold = self._safe_float(self._cfg.get("confirm_trend_threshold", 0.02), 0.02)

        # 区间稳健性：支撑/阻力至少被触及次数（过滤单次极值造成的假区间）
        self._min_band_touch_count = self._safe_int(self._cfg.get("min_band_touch_count", 2), 2)
        self._touch_proximity_pct = self._safe_float(self._cfg.get("touch_proximity_pct", 0.0015), 0.0015)

        # 成交量分布(VP)确认：支撑/阻力位需有高成交量节点(HVN)佐证，过滤低量假区间
        self._vp_confirm_enabled = bool(self._cfg.get("vp_confirm_enabled", True))
        self._vp_bins = self._safe_int(self._cfg.get("vp_bins", 20), 20)
        self._vp_hvn_ratio = self._safe_float(self._cfg.get("vp_hvn_ratio", 1.2), 1.2)

        # 布林带辅助确认：RSI + 布林带 %B 双确认，自适应波动边界
        self._use_bollinger_confirm = bool(self._cfg.get("use_bollinger_confirm", True))
        self._boll_period = self._safe_int(self._cfg.get("boll_period", 20), 20)
        self._boll_std_mult = self._safe_float(self._cfg.get("boll_std_mult", 2.0), 2.0)

        # 分级止盈：TP1 半程减仓 + TP2 中轨减仓，剩余仓位跟随原有中轨回归逻辑
        self._staged_tp_enabled = bool(self._cfg.get("staged_tp_enabled", True))
        self._tp1_mid_ratio = self._safe_float(self._cfg.get("tp1_mid_ratio", 0.5), 0.5)
        self._tp1_close_ratio = self._safe_float(self._cfg.get("tp1_close_ratio", 0.4), 0.4)
        self._tp2_mid_ratio = self._safe_float(self._cfg.get("tp2_mid_ratio", 1.0), 1.0)
        self._tp2_close_ratio = self._safe_float(self._cfg.get("tp2_close_ratio", 0.3), 0.3)

        # 自适应调参：接入 AdaptiveController（空闲资金仓位 boost + 风险预算动态阈值）
        self._adaptive_enabled = bool(self._cfg.get("adaptive_enabled", True))
        # 连续亏损/风险熔断时，动态上浮信号质量阈值（收紧开仓）
        self._risk_lock_quality_boost = self._safe_float(self._cfg.get("risk_lock_quality_boost", 0.15), 0.15)

        # 可观测性：过滤理由本地计数（get_health 暴露 + MetricsPipeline 埋点）
        self._filter_stats: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 市场状态过滤
    # ------------------------------------------------------------------
    def _get_regime_engine(self):
        """获取 MarketRegimeEngine（优先直接注入，回退到 coordinator）。"""
        engine = getattr(self, "_regime_engine", None)
        if engine is not None:
            return engine
        coordinator = getattr(self, "_coordinator", None)
        if coordinator is not None:
            return getattr(coordinator, "_regime_engine", None)
        return None

    def _get_symbol_regime(self, symbol: str) -> Optional[Dict[str, Any]]:
        engine = self._get_regime_engine()
        if engine is None:
            return None
        try:
            data = engine.get_symbol_regime(symbol)
            return data if isinstance(data, dict) else None
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] get_symbol_regime failed for {symbol}: {e}")
            return None

    def _is_global_regime_safe(self) -> bool:
        """全局(BTC)状态不允许处于极端风险状态。"""
        engine = self._get_regime_engine()
        if engine is None:
            return False
        try:
            regime = engine.get_regime() or {}
            regime_type = regime.get("regime", "unknown")
            return regime_type not in ("extreme_volatility", "funding_crush", "liquidity_crisis")
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] get_regime failed: {e}")
            return False

    def _is_range_bound(self, symbol: str) -> bool:
        """品种级必须为 range_bound，且全局状态安全。"""
        if not self._is_global_regime_safe():
            return False
        sym_regime = self._get_symbol_regime(symbol)
        if not sym_regime:
            return False
        if sym_regime.get("regime") != "range_bound":
            return False
        if self._safe_float(sym_regime.get("confidence"), 0.0) < self._min_regime_confidence:
            return False
        return True

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------
    @staticmethod
    def _rsi(closes, period: int = 14) -> float:
        closes = np.asarray(closes, dtype=float)
        if len(closes) < period + 1:
            return 50.0
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains[-period:]))
        avg_loss = float(np.mean(losses[-period:]))
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    @staticmethod
    def _ema(data, period: int) -> float:
        """指数移动平均（数据不足取均值）。"""
        data = np.asarray(data, dtype=float)
        if len(data) == 0:
            return 0.0
        if len(data) < period:
            return float(np.mean(data))
        alpha = 2.0 / (period + 1.0)
        result = data[0]
        for val in data[1:]:
            result = alpha * val + (1.0 - alpha) * result
        return float(result)

    def _bollinger(self, closes, period: Optional[int] = None, std_mult: Optional[float] = None):
        """布林带：返回 {mid, upper, lower, pct_b}。数据不足返回 None。"""
        period = period or self._boll_period
        std_mult = std_mult if std_mult is not None else self._boll_std_mult
        closes = np.asarray(closes, dtype=float)
        if len(closes) < period:
            return None
        mid = float(np.mean(closes[-period:]))
        std = float(np.std(closes[-period:]))
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        if upper - lower <= 0:
            return None
        pct_b = (closes[-1] - lower) / (upper - lower)
        return {"mid": mid, "upper": upper, "lower": lower, "pct_b": float(pct_b)}

    async def _fetch_klines_bar(self, symbol: str, bar: str, limit: int) -> list:
        """获取指定周期的正序 K 线（升序）。"""
        try:
            klines = await self.okx_client.get_kline_async(symbol, bar, limit=limit)
            if not klines:
                return []
            return sorted(klines, key=lambda k: int(k[0]))
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] _fetch_klines_bar({bar}) failed for {symbol}: {e}")
            return []

    async def _higher_tf_trending(self, symbol: str) -> bool:
        """高周期趋势确认：高周期 EMA20/EMA50 偏离超阈值视为强趋势，放弃震荡开仓。"""
        if not self._confirm_bar or self._confirm_bar == self._bar:
            return False
        klines = await self._fetch_klines_bar(symbol, self._confirm_bar, limit=60)
        if len(klines) < 50:
            return False
        closes = np.array([float(k[4]) for k in klines])
        ema_fast = self._ema(closes, 20)
        ema_slow = self._ema(closes, 50)
        if ema_slow <= 0:
            return False
        return abs(ema_fast - ema_slow) / ema_slow > self._confirm_trend_threshold

    def _compute_support_resistance(self, highs, lows, volumes=None) -> Optional[Dict[str, float]]:
        """基于滚动高低点识别支撑/阻力（排除当前未完成 bar 以避免未来函数）。"""
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        if len(highs) < self._lookback_bars or len(lows) < self._lookback_bars:
            return None

        support = float(np.min(lows[-self._lookback_bars:-1]))
        resistance = float(np.max(highs[-self._lookback_bars:-1]))
        if resistance <= support:
            return None

        mid = (support + resistance) / 2.0
        range_pct = (resistance - support) / mid if mid > 0 else 0.0

        # 触及次数：统计窗口内价格触及支撑/阻力的次数（验证区间有效性，过滤单次极值假区间）
        prox = mid * self._touch_proximity_pct if mid > 0 else 0.0
        window_lows = lows[-self._lookback_bars:-1]
        window_highs = highs[-self._lookback_bars:-1]
        support_touches = int(np.sum(window_lows <= support + prox))
        resistance_touches = int(np.sum(window_highs >= resistance - prox))
        touch_count = min(support_touches, resistance_touches)

        # 成交量分布(VP)确认：支撑/阻力位是否处于高成交量节点(HVN)
        vp_confirmed = True
        if self._vp_confirm_enabled and volumes is not None:
            volumes = np.asarray(volumes, dtype=float)
            vol_window = volumes[-self._lookback_bars:-1]
            price_window_lows = lows[-self._lookback_bars:-1]
            price_window_highs = highs[-self._lookback_bars:-1]
            vp_confirmed = self._check_vp_confirmation(
                price_window_lows, price_window_highs, vol_window,
                support, resistance
            )

        return {"support": support, "resistance": resistance, "mid": mid,
                "range_pct": range_pct, "touch_count": touch_count,
                "vp_confirmed": vp_confirmed}

    def _check_vp_confirmation(self, bar_lows, bar_highs, volumes,
                               support: float, resistance: float) -> bool:
        """检查支撑/阻力位是否有高成交量节点(HVN)佐证。

        将价格区间分为 vp_bins 个 bin，统计每个 bin 的成交量。
        若支撑和阻力附近的 bin 成交量 >= 平均成交量 * hvn_ratio，则确认。
        """
        if resistance <= support or len(volumes) == 0:
            return False
        n_bins = self._vp_bins
        bin_size = (resistance - support) / n_bins
        if bin_size <= 0:
            return False

        vol_profile = np.zeros(n_bins)
        for i in range(len(volumes)):
            price_mid = (bar_lows[i] + bar_highs[i]) / 2.0
            bin_idx = int((price_mid - support) / bin_size)
            bin_idx = max(0, min(n_bins - 1, bin_idx))
            vol_profile[bin_idx] += volumes[i]

        avg_vol = np.mean(vol_profile) if np.sum(vol_profile) > 0 else 0.0
        if avg_vol <= 0:
            return False

        threshold = avg_vol * self._vp_hvn_ratio
        support_bin = 0
        resistance_bin = n_bins - 1
        support_vol = vol_profile[support_bin]
        resistance_vol = vol_profile[resistance_bin]
        return support_vol >= threshold and resistance_vol >= threshold

    # ------------------------------------------------------------------
    # 信号评估
    # ------------------------------------------------------------------
    def _evaluate(self, symbol: str, closes, highs, lows, volumes=None) -> Optional[Dict[str, Any]]:
        if not self._is_range_bound(symbol):
            return {"signal": None, "reason": "not_range_bound"}

        s_r = self._compute_support_resistance(highs, lows, volumes)
        if not s_r:
            return None

        support = s_r["support"]
        resistance = s_r["resistance"]
        mid = s_r["mid"]
        range_pct = s_r["range_pct"]
        touch_count = int(s_r.get("touch_count", 0))
        vp_confirmed = bool(s_r.get("vp_confirmed", True))

        # 区间过窄（磨损型）直接放弃
        if range_pct < self._min_band_width_pct:
            return {"signal": None, "reason": f"band_too_narrow({range_pct:.4f})"}

        price = float(np.asarray(closes, dtype=float)[-1])
        rsi = self._rsi(closes, self._rsi_period)
        atr = _atr(np.asarray(highs, dtype=float), np.asarray(lows, dtype=float),
                   np.asarray(closes, dtype=float), 14)
        bb = self._bollinger(closes) if self._use_bollinger_confirm else None

        # 边界触及：96-bar 支撑/阻力 或 布林带极值（自适应波动边界）
        near_support = price <= support * (1.0 + self._support_band_pct)
        near_resistance = price >= resistance * (1.0 - self._support_band_pct)
        bb_low = bb is not None and bb["pct_b"] <= 0.05
        bb_high = bb is not None and bb["pct_b"] >= 0.95
        oversold = rsi < self._rsi_oversold
        overbought = rsi > self._rsi_overbought

        # 区间是否获多次触及验证（min_band_touch_count=0 视为不要求验证）
        range_verified = self._min_band_touch_count <= 0 or touch_count >= self._min_band_touch_count

        # 做多：低点触及 + RSI 超卖；做空：高点触及 + RSI 超买
        direction = None
        band_depth = 0.0
        stop_loss = None
        take_profit = None
        boll_only = False  # 区间未验证时降级为布林带震荡兜底

        if range_verified:
            if (near_support or bb_low) and oversold:
                direction = "long"
                band_depth = (support - price) / support if support > 0 else 0.0
                sl_offset = atr * self._atr_sl_mult if atr > 0 else support * self._support_band_pct
                max_sl = price * self._max_stop_loss_pct
                sl_offset = min(sl_offset, max_sl) if max_sl > 0 else sl_offset
                stop_loss = price - sl_offset
                tp_dist = max(sl_offset * 1.5, (mid - price) * self._mid_tp_ratio)
                take_profit = price + tp_dist
            elif (near_resistance or bb_high) and overbought:
                direction = "short"
                band_depth = (price - resistance) / resistance if resistance > 0 else 0.0
                sl_offset = atr * self._atr_sl_mult if atr > 0 else resistance * self._support_band_pct
                max_sl = price * self._max_stop_loss_pct
                sl_offset = min(sl_offset, max_sl) if max_sl > 0 else sl_offset
                stop_loss = price + sl_offset
                tp_dist = max(sl_offset * 1.5, (price - mid) * self._mid_tp_ratio)
                take_profit = price - tp_dist
        elif bb is not None:
            # 区间未验证：降级为布林带震荡兜底（弱震荡市用动态波动边界替代固定支撑/阻力）。
            # 仅当布林带极值 + RSI 极端双共振时才开仓，且止损置于布林带外侧，止盈回归中轨。
            if bb_low and oversold:
                direction = "long"
                boll_only = True
                band_depth = (bb["lower"] - price) / bb["lower"] if bb["lower"] > 0 else 0.0
                sl_offset = atr * self._atr_sl_mult if atr > 0 else bb["lower"] * 0.01
                max_sl = price * self._max_stop_loss_pct
                sl_offset = min(sl_offset, max_sl) if max_sl > 0 else sl_offset
                stop_loss = price - sl_offset
                tp_dist = max(sl_offset * 1.5, bb["mid"] - price)
                take_profit = price + tp_dist
            elif bb_high and overbought:
                direction = "short"
                boll_only = True
                band_depth = (price - bb["upper"]) / bb["upper"] if bb["upper"] > 0 else 0.0
                sl_offset = atr * self._atr_sl_mult if atr > 0 else bb["upper"] * 0.01
                max_sl = price * self._max_stop_loss_pct
                sl_offset = min(sl_offset, max_sl) if max_sl > 0 else sl_offset
                stop_loss = price + sl_offset
                tp_dist = max(sl_offset * 1.5, price - bb["mid"])
                take_profit = price - tp_dist

        if direction is None:
            reason = "range_untested(touches={})".format(touch_count) if not range_verified else "no_entry"
            return {"signal": None, "reason": reason, "rsi": rsi,
                    "support": support, "resistance": resistance,
                    "boll_pct_b": bb["pct_b"] if bb else None}

        # 置信度：区间深度 + 区间宽度 + regime 置信度 + 布林带深度
        sym_regime = self._get_symbol_regime(symbol) or {}
        regime_conf = self._safe_float(sym_regime.get("confidence"), 0.0)
        range_score = min(1.0, max(0.0, (range_pct - self._min_band_width_pct) /
                                   max(self._min_band_width_pct * 3, 1e-9)))
        depth_score = min(1.0, band_depth * 20.0)
        boll_score = 0.0
        if bb is not None:
            if direction == "long":
                boll_score = min(1.0, max(0.0, (0.2 - bb["pct_b"]) * 5.0))
            else:
                boll_score = min(1.0, max(0.0, (bb["pct_b"] - 0.8) * 5.0))
        confidence = 0.40 + depth_score * 0.18 + range_score * 0.12 + regime_conf * 0.15 + boll_score * 0.10
        if boll_only:
            confidence -= 0.05
        if not vp_confirmed:
            confidence -= 0.06
        confidence = min(0.85, confidence)

        return {
            "signal": direction,
            "confidence": confidence,
            "rsi": rsi,
            "support": support,
            "resistance": resistance,
            "mid": mid,
            "range_pct": range_pct,
            "touch_count": touch_count,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "boll_pct_b": bb["pct_b"] if bb else None,
            "boll_only": boll_only,
        }

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    async def _check_signals(self):
        open_count = sum(1 for p in self._position_state.values() if p.get("status") == "open")
        if open_count >= self._max_concurrent_positions:
            self._record_filter("", "max_positions_reached")
            return

        # 自适应调参：连续亏损/风险熔断时动态上浮信号质量阈值
        min_quality = self._dynamic_min_quality()

        for symbol in self._symbols:
            if self._position_state.get(symbol, {}).get("status") == "open":
                self._record_filter(symbol, "position_exists")
                continue
            if not self._cooled_down(symbol):
                self._record_filter(symbol, "cooldown")
                continue

            # 高周期趋势确认：高周期强趋势时放弃震荡开仓，避免区间即将突破
            if await self._higher_tf_trending(symbol):
                self._record_filter(symbol, "higher_tf_trending")
                continue

            klines = await self._fetch_klines(symbol, limit=self._lookback_bars + 50)
            if len(klines) < self._lookback_bars + 1:
                self._record_filter(symbol, "insufficient_klines")
                continue
            _, highs, lows, closes, volumes = self._klines_to_arrays(klines)

            result = self._evaluate(symbol, closes, highs, lows, volumes)
            if not result:
                self._record_filter(symbol, "no_support_resistance")
                continue
            direction = result.get("signal")
            if direction not in ("long", "short"):
                reason = (result.get("reason") or "no_entry").split("(")[0]
                self._record_filter(symbol, reason)
                continue

            confidence = self._safe_float(result.get("confidence"), 0.0)
            if confidence < min_quality:
                self._record_filter(symbol, "low_confidence")
                continue

            confidence = await self._funding_enhancer.adjust_entry_confidence(symbol, direction, confidence)
            if confidence < min_quality:
                self._record_filter(symbol, "low_confidence_after_funding")
                continue

            price = float(np.asarray(closes, dtype=float)[-1])
            stop_loss = self._round_price(symbol, result.get("stop_loss"))
            take_profit = self._round_price(symbol, result.get("take_profit"))
            if not self._validate_tp_sl(price, direction, stop_loss, take_profit):
                self._record_filter(symbol, "invalid_tp_sl")
                continue

            quantity, _ = self._calculate_quantity(symbol, price)
            if quantity <= 0:
                self._record_filter(symbol, "zero_quantity")
                continue

            # 自适应调参：空闲资金放大仓位（低利用率时提升仓位，由 AdaptiveController 封顶）
            if self._adaptive_enabled and self._adaptive_controller:
                try:
                    boost = self._adaptive_controller.get_position_boost()
                    if boost > 1.0:
                        quantity = self.okx_client.round_quantity_to_lot(
                            symbol, quantity * boost, round_up=False)
                except Exception:
                    pass
            if quantity <= 0:
                self._record_filter(symbol, "zero_quantity_after_boost")
                continue

            if not self._profit_covers_fees(price, direction, quantity, take_profit):
                self._record_filter(symbol, "fees_not_covered")
                continue

            if self._check_cross_strategy_conflict(symbol, direction):
                self._record_filter(symbol, "cross_strategy_conflict")
                continue

            self._record_metric("oscillation_harvest_signal_confidence", confidence,
                                {"symbol": symbol, "direction": direction})
            self._publish_entry_signal(symbol, direction, price, quantity,
                                       stop_loss, take_profit, confidence)

    async def _manage_positions(self):
        """分级止盈 + 回归中轨止盈 / 状态退出 / 区间反向出场。"""
        for symbol, pos in list(self._position_state.items()):
            if pos.get("status") != "open":
                continue
            direction = pos.get("direction")
            klines = await self._fetch_klines(symbol, limit=self._lookback_bars + 50)
            if len(klines) < self._lookback_bars + 1:
                continue
            _, highs, lows, closes, volumes = self._klines_to_arrays(klines)
            result = self._evaluate(symbol, closes, highs, lows, volumes)
            if not result:
                continue

            price = float(np.asarray(closes, dtype=float)[-1])
            mid = self._safe_float(result.get("mid"), 0.0)
            signal = result.get("signal")
            quantity = self._safe_float(pos.get("current_quantity"), 0.0)
            entry_price = self._safe_float(pos.get("entry_price"), 0.0)
            if quantity <= 0 or entry_price <= 0:
                continue

            # 离开震荡区间 → 风控平仓
            if result.get("reason") == "not_range_bound":
                self._record_exit_metrics(symbol, direction, entry_price, price, quantity, "regime_exit")
                self._publish_exit_signal(symbol, direction, price, quantity, reason="regime_exit")
                continue

            reverse = (direction == "long" and signal == "short") or \
                      (direction == "short" and signal == "long")

            # ── 分级止盈 ──
            if self._staged_tp_enabled and mid > 0 and not reverse:
                dist_to_mid = abs(mid - entry_price)
                if direction == "long":
                    tp1_price = entry_price + dist_to_mid * self._tp1_mid_ratio
                    tp2_price = entry_price + dist_to_mid * self._tp2_mid_ratio
                else:
                    tp1_price = entry_price - dist_to_mid * self._tp1_mid_ratio
                    tp2_price = entry_price - dist_to_mid * self._tp2_mid_ratio

                tp1_hit = (direction == "long" and price >= tp1_price) or \
                          (direction == "short" and price <= tp1_price)
                tp2_hit = (direction == "long" and price >= tp2_price) or \
                          (direction == "short" and price <= tp2_price)

                if tp1_hit and not pos.get("tp1_done"):
                    close_qty = self.okx_client.round_quantity_to_lot(
                        symbol, quantity * self._tp1_close_ratio, round_up=False)
                    if close_qty > 0:
                        self._publish_exit_signal(symbol, direction, price, close_qty,
                                                  reason="take_profit_1")
                        pos["current_quantity"] = quantity - close_qty
                        quantity = quantity - close_qty
                    pos["tp1_done"] = True

                if tp2_hit and not pos.get("tp2_done") and pos.get("tp1_done"):
                    remaining = self._safe_float(pos.get("current_quantity"), 0.0)
                    close_qty = self.okx_client.round_quantity_to_lot(
                        symbol, remaining * self._tp2_close_ratio, round_up=False)
                    if close_qty > 0:
                        self._publish_exit_signal(symbol, direction, price, close_qty,
                                                  reason="take_profit_2")
                        pos["current_quantity"] = remaining - close_qty
                        quantity = remaining - close_qty
                    pos["tp2_done"] = True

            if reverse:
                reason = "exit"
                quantity = self._safe_float(pos.get("current_quantity"), 0.0)
                if quantity > 0:
                    self._record_exit_metrics(symbol, direction, entry_price, price, quantity, reason)
                    self._publish_exit_signal(symbol, direction, price, quantity, reason=reason)
                continue

            reverted = (direction == "long" and mid > 0 and price >= mid) or \
                       (direction == "short" and mid > 0 and price <= mid)

            if reverted:
                quantity = self._safe_float(pos.get("current_quantity"), 0.0)
                if quantity > 0:
                    self._record_exit_metrics(symbol, direction, entry_price, price, quantity, "take_profit")
                    self._publish_exit_signal(symbol, direction, price, quantity, reason="take_profit")

    # ------------------------------------------------------------------
    # 冷却与校验辅助
    # ------------------------------------------------------------------
    def _cooled_down(self, symbol: str) -> bool:
        last_ts = self._last_signal_time.get(symbol)
        if last_ts is None:
            return True
        if isinstance(last_ts, str):
            try:
                last_ts = datetime.fromisoformat(last_ts)
            except (ValueError, TypeError):
                return True
        return (datetime.now() - last_ts).total_seconds() >= self._symbol_cooldown_seconds

    def _validate_tp_sl(self, price: float, direction: str, stop_loss, take_profit) -> bool:
        if direction == "long":
            if stop_loss is not None and stop_loss >= price:
                return False
            if take_profit is not None and take_profit <= price:
                return False
        else:
            if stop_loss is not None and stop_loss <= price:
                return False
            if take_profit is not None and take_profit >= price:
                return False
        return True

    def _profit_covers_fees(self, price: float, direction: str,
                            quantity: float, take_profit) -> bool:
        if take_profit is None:
            return True
        round_trip_fee = self._taker_fee_rate * 2
        position_value = price * quantity
        fee_cost = position_value * round_trip_fee
        if direction == "long":
            expected_profit = (take_profit - price) * quantity
        else:
            expected_profit = (price - take_profit) * quantity
        return expected_profit > fee_cost * 1.5

    def _dynamic_min_quality(self) -> float:
        """自适应信号质量阈值：连续亏损/风险熔断时上浮，收紧开仓。"""
        quality = self._min_signal_quality
        if not self._adaptive_enabled or self._adaptive_controller is None:
            return quality
        try:
            status = self._adaptive_controller.get_risk_budget_status()
            if status.get("streak_lock_active"):
                quality += self._risk_lock_quality_boost
        except Exception as e:
            # fail-closed：风险锁状态查询失败时保守收紧阈值，避免在风控状态未知时放行
            logger.debug(f"[oscillation] risk budget status query failed, tightening min quality: {e}")
            quality += self._risk_lock_quality_boost
        return quality

    def _record_filter(self, symbol: str, reason: str):
        """记录一次过滤/拒绝原因（本地计数 + MetricsPipeline 埋点）。"""
        self._filter_stats[reason] = self._filter_stats.get(reason, 0) + 1
        try:
            self._increment_metric("oscillation_harvest_filter_total", 1.0,
                                   {"reason": reason, "symbol": symbol})
        except Exception:
            pass
        logger.debug(f"[{self._strategy_name}] {symbol} 过滤: {reason}")

    def _record_exit_metrics(self, symbol: str, direction: str, entry_price: float,
                             exit_price: float, quantity: float, reason: str):
        """出场埋点：记录平仓笔数与已实现盈亏（本地 + MetricsPipeline）。"""
        ret = 0.0
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price if direction == "long" \
                else (entry_price - exit_price) / entry_price
        pnl_usdt = ret * entry_price * quantity
        try:
            self._record_metric("oscillation_harvest_position_closed_total", 1.0,
                                {"reason": reason, "symbol": symbol, "direction": direction})
            self._record_metric("oscillation_harvest_position_pnl", pnl_usdt,
                                {"reason": reason, "symbol": symbol})
        except Exception:
            pass

    def get_health(self) -> Dict[str, Any]:
        health = super().get_health()
        health["regime_engine"] = self._get_regime_engine() is not None
        health["range_positions"] = sum(
            1 for p in self._position_state.values() if p.get("status") == "open")
        health["adaptive_enabled"] = self._adaptive_enabled
        health["min_signal_quality"] = self._min_signal_quality
        health["dynamic_min_quality"] = self._dynamic_min_quality()
        health["risk_lock_quality_boost"] = self._risk_lock_quality_boost
        health["adaptive_controller_injected"] = self._adaptive_controller is not None
        health["filter_stats"] = dict(self._filter_stats)
        return health
