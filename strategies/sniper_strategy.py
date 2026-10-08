"""
市场狙击策略（Sniper）

核心思想：不做频繁轮询交易，只在「高确信狙击点」精准开仓，一击即中。
按市场状态自动切换双模式：

  1. 趋势突破狙击（trending / 趋势市）：
     收盘价突破近 N 周期关键位 + ADX 转强 + MACD 同向 + 量能放大，四条件共振才开仓。

  2. 关键位反转狙击（range_bound / 震荡市）：
     触及支撑/阻力关键位 + RSI 超买超卖 + 反转 K 线确认，三条件共振才开仓。

标的：tier1 + tier2 主流永续。bar 默认 5m（精准狙击），ATR 止损/止盈。
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from loguru import logger

from strategies._trend_base import TrendStrategyBase


class SniperStrategy(TrendStrategyBase):
    STRATEGY_KEY = "sniper"
    DISPLAY_NAME = "市场狙击"
    SYMBOL_SCOPE = "tier1_tier2"
    DEFAULT_BAR = "5m"
    ENTRY_SIGNAL_TYPE = "sniper_entry"
    EXIT_SIGNAL_TYPE = "sniper_exit"

    def __init__(self, config, okx_client, redis_cache):
        super().__init__(config, okx_client, redis_cache)

        # ── 趋势突破狙击参数 ──
        self._adx_threshold = self._safe_float(self._cfg.get("adx_threshold", 20.0), 20.0)
        self._adx_period = self._safe_int(self._cfg.get("adx_period", 14), 14)
        self._breakout_lookback = self._safe_int(self._cfg.get("breakout_lookback", 20), 20)
        self._volume_multiplier = self._safe_float(self._cfg.get("volume_multiplier", 1.5), 1.5)

        # ── 关键位反转狙击参数 ──
        self._rsi_period = self._safe_int(self._cfg.get("rsi_period", 14), 14)
        self._rsi_oversold = self._safe_float(self._cfg.get("rsi_oversold", 30.0), 30.0)
        self._rsi_overbought = self._safe_float(self._cfg.get("rsi_overbought", 70.0), 70.0)

        # ── 止损止盈 ──
        self._atr_period = self._safe_int(self._cfg.get("atr_period", 14), 14)
        self._atr_sl_mult = self._safe_float(self._cfg.get("atr_sl_multiplier", 2.0), 2.0)
        self._atr_tp_mult = self._safe_float(self._cfg.get("atr_tp_multiplier", 3.0), 3.0)
        self._trailing_stop_enabled = bool(self._cfg.get("trailing_stop_enabled", True))
        self._trailing_atr_mult = self._safe_float(
            self._cfg.get("trailing_atr_multiplier", self._atr_sl_mult), self._atr_sl_mult
        )

        # ── 分级止盈 ──
        self._staged_tp_enabled = bool(self._cfg.get("staged_tp_enabled", True))
        self._tp1_atr_mult = self._safe_float(self._cfg.get("tp1_atr_mult", 1.5), 1.5)
        self._tp1_close_ratio = self._safe_float(self._cfg.get("tp1_close_ratio", 0.4), 0.4)
        self._tp2_atr_mult = self._safe_float(self._cfg.get("tp2_atr_mult", 2.5), 2.5)
        self._tp2_close_ratio = self._safe_float(self._cfg.get("tp2_close_ratio", 0.3), 0.3)

        self._filter_stats: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # 过滤埋点
    # ------------------------------------------------------------------
    def _record_filter(self, symbol: str, reason: str):
        self._filter_stats[reason] = self._filter_stats.get(reason, 0) + 1
        try:
            self._record_metric(f"{self._strategy_name}_filter_total", 1.0,
                                {"reason": reason, "symbol": symbol})
        except Exception:
            pass
        logger.debug(f"[{self._strategy_name}] filter: {symbol} {reason}")

    # ------------------------------------------------------------------
    # 市场状态
    # ------------------------------------------------------------------
    def _current_regime(self) -> str:
        """读取市场状态；无法获取时保守回退 trending（狙击默认要求趋势确认）。"""
        if self._regime_engine is None:
            return "trending"
        try:
            getter = getattr(self._regime_engine, "get_regime", None)
            if not callable(getter):
                return "trending"
            regime = getter()
            if isinstance(regime, dict):
                return str(regime.get("regime") or "trending").lower()
            return str(regime).lower()
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] regime query failed: {e}")
            return "trending"

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------
    @staticmethod
    def _wilder_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    period: int = 14) -> float:
        """标准 Wilder ADX 最新值。数据不足返回 -1。"""
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        if n <= 2 * period or np.any(np.isnan(highs)) or np.any(np.isnan(lows)) or np.any(np.isnan(closes)):
            return -1.0

        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            up = highs[i] - highs[i - 1]
            down = lows[i - 1] - lows[i]
            plus_dm[i] = up if (up > down and up > 0) else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0

        def _wilder(x: np.ndarray) -> np.ndarray:
            s = np.zeros(n)
            s[period] = float(np.sum(x[1:period + 1]))
            for i in range(period + 1, n):
                s[i] = s[i - 1] - s[i - 1] / period + x[i]
            return s

        atr = _wilder(tr)
        s_plus = _wilder(plus_dm)
        s_minus = _wilder(minus_dm)
        dx = np.zeros(n)
        for i in range(period, n):
            if atr[i] <= 0:
                continue
            pdi = 100.0 * s_plus[i] / atr[i]
            mdi = 100.0 * s_minus[i] / atr[i]
            denom = pdi + mdi
            dx[i] = 100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0

        start = 2 * period
        adx = float(np.mean(dx[period + 1:start + 1]))
        for i in range(start + 1, n):
            adx = (adx * (period - 1) + dx[i]) / period
        return float(adx)

    @staticmethod
    def _ema(series: np.ndarray, period: int) -> np.ndarray:
        series = np.asarray(series, dtype=float)
        out = np.empty_like(series)
        out[0] = series[0]
        k = 2.0 / (period + 1)
        for i in range(1, len(series)):
            out[i] = series[i] * k + out[i - 1] * (1 - k)
        return out

    @staticmethod
    def _macd_hist(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> float:
        """返回最新 MACD 柱（histogram）值。数据不足返回 0。"""
        closes = np.asarray(closes, dtype=float)
        if len(closes) < slow + signal:
            return 0.0
        ema_fast = SniperStrategy._ema(closes, fast)
        ema_slow = SniperStrategy._ema(closes, slow)
        macd = ema_fast - ema_slow
        signal_line = SniperStrategy._ema(macd, signal)
        return float(macd[-1] - signal_line[-1])

    @staticmethod
    def _rsi(closes: np.ndarray, period: int = 14) -> float:
        """返回最新 RSI 值。数据不足返回 50。"""
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
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """返回最新 ATR 值。数据不足返回 0。"""
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(closes) < period + 1:
            return 0.0
        tr = np.zeros(len(closes))
        for i in range(1, len(closes)):
            tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
        return float(np.mean(tr[-period:]))

    # ------------------------------------------------------------------
    # 信号扫描
    # ------------------------------------------------------------------
    async def _check_signals(self):
        open_count = sum(1 for p in self._position_state.values() if p.get("status") == "open")
        if open_count >= self._max_concurrent_positions:
            self._record_filter("", "max_positions_reached")
            return

        regime = self._current_regime()
        trending = regime not in ("range_bound", "ranging", "oscillation", "sideways")

        for symbol in self._symbols:
            if self._position_state.get(symbol, {}).get("status") == "open":
                self._record_filter(symbol, "position_exists")
                continue

            klines = await self._fetch_klines(symbol, limit=120)
            if len(klines) < 40:
                self._record_filter(symbol, "insufficient_klines")
                continue
            opens, highs, lows, closes, volumes = self._klines_to_arrays(klines)

            if trending:
                entry = self._evaluate_breakout(symbol, opens, highs, lows, closes, volumes)
            else:
                entry = self._evaluate_reversal(symbol, opens, highs, lows, closes, volumes)

            if entry is None:
                continue

            direction, price, stop_loss, take_profit, confidence = entry
            if not self._validate_tp_sl(price, direction, stop_loss, take_profit):
                self._record_filter(symbol, "tp_sl_validation_failed")
                continue

            quantity, _ = self._calculate_quantity(symbol, price)
            if quantity <= 0:
                self._record_filter(symbol, "quantity_below_threshold")
                continue

            if not self._profit_covers_fees(price, direction, quantity, take_profit):
                self._record_filter(symbol, "profit_covers_fees_failed")
                continue

            if self._check_cross_strategy_conflict(symbol, direction):
                self._record_filter(symbol, "cross_strategy_conflict")
                continue

            self._publish_entry_signal(symbol, direction, price, quantity,
                                       stop_loss, take_profit, confidence)

    def _evaluate_breakout(self, symbol: str, opens, highs, lows, closes, volumes):
        """趋势突破狙击：关键位突破 + ADX 转强 + MACD 同向 + 量能放大。"""
        lookback = self._breakout_lookback
        # 关键位：近 lookback 周期（不含最后一根）高低点
        resist = float(np.max(highs[-lookback - 1:-1])) if len(highs) > lookback + 1 else float(np.max(highs))
        support = float(np.min(lows[-lookback - 1:-1])) if len(lows) > lookback + 1 else float(np.min(lows))
        price = float(closes[-1])

        adx = self._wilder_adx(highs, lows, closes, self._adx_period)
        macd = self._macd_hist(closes)
        vol_avg = float(np.mean(volumes[-lookback - 1:-1])) if len(volumes) > lookback + 1 else float(np.mean(volumes))
        vol_cur = float(volumes[-1])
        vol_ok = vol_avg > 0 and vol_cur >= vol_avg * self._volume_multiplier

        # 做多：收盘突破阻力 + MACD>0 + 量能放大
        if price > resist and macd > 0 and vol_ok:
            adx_ok = adx < 0 or adx >= self._adx_threshold  # 无法计算 ADX 时放行（由其它条件兜底）
            if not adx_ok:
                self._record_filter(symbol, "breakout_adx_weak")
                return None
            atr = self._atr(highs, lows, closes, self._atr_period)
            stop = self._round_price(symbol, price - atr * self._atr_sl_mult) if atr > 0 else None
            take = self._round_price(symbol, price + atr * self._atr_tp_mult) if atr > 0 else None
            adx_term = adx / 100.0 if adx > 0 else 0.15
            vol_term = min(0.2, (vol_cur / vol_avg - 1.0) * 0.2)
            confidence = min(0.95, 0.55 + adx_term + vol_term)
            return "long", price, stop, take, round(confidence, 3)

        # 做空：收盘跌破支撑 + MACD<0 + 量能放大
        if price < support and macd < 0 and vol_ok:
            adx_ok = adx < 0 or adx >= self._adx_threshold
            if not adx_ok:
                self._record_filter(symbol, "breakout_adx_weak")
                return None
            atr = self._atr(highs, lows, closes, self._atr_period)
            stop = self._round_price(symbol, price + atr * self._atr_sl_mult) if atr > 0 else None
            take = self._round_price(symbol, price - atr * self._atr_tp_mult) if atr > 0 else None
            adx_term = adx / 100.0 if adx > 0 else 0.15
            vol_term = min(0.2, (vol_cur / vol_avg - 1.0) * 0.2)
            confidence = min(0.95, 0.55 + adx_term + vol_term)
            return "short", price, stop, take, round(confidence, 3)

        return None

    def _evaluate_reversal(self, symbol: str, opens, highs, lows, closes, volumes):
        """关键位反转狙击：触及支撑/阻力 + RSI 超买超卖 + 反转 K 线确认。"""
        lookback = self._breakout_lookback
        resist = float(np.max(highs[-lookback - 1:-1])) if len(highs) > lookback + 1 else float(np.max(highs))
        support = float(np.min(lows[-lookback - 1:-1])) if len(lows) > lookback + 1 else float(np.min(lows))
        price = float(closes[-1])
        rsi = self._rsi(closes, self._rsi_period)

        # 反转 K 线：做多需阳线（收盘 > 开盘），做空需阴线（收盘 < 开盘）
        bullish_candle = float(closes[-1]) > float(opens[-1])
        bearish_candle = float(closes[-1]) < float(opens[-1])

        # 做多：触及支撑 + RSI 超卖 + 阳线
        if float(lows[-1]) <= support * 1.002 and rsi < self._rsi_oversold and bullish_candle:
            atr = self._atr(highs, lows, closes, self._atr_period)
            sl_dist = atr * self._atr_sl_mult if atr > 0 else support * 0.01
            stop = self._round_price(symbol, support - sl_dist) if atr > 0 else self._round_price(symbol, support * 0.99)
            sr_tp = self._round_price(symbol, resist) if resist > price else self._round_price(symbol, price + atr * self._atr_tp_mult)
            atr_tp = self._round_price(symbol, price + atr * self._atr_tp_mult)
            sr_reward = abs(float(sr_tp) - price)
            atr_reward = abs(float(atr_tp) - price)
            min_reward = sl_dist * 1.5
            take = sr_tp if sr_reward >= min_reward else (atr_tp if atr_reward >= min_reward else self._round_price(symbol, price + min_reward))
            confidence = round(min(0.9, 0.6 + (30.0 - rsi) * 0.01), 3)
            return "long", price, stop, take, confidence

        # 做空：触及阻力 + RSI 超买 + 阴线
        if float(highs[-1]) >= resist * 0.998 and rsi > self._rsi_overbought and bearish_candle:
            atr = self._atr(highs, lows, closes, self._atr_period)
            sl_dist = atr * self._atr_sl_mult if atr > 0 else resist * 0.01
            stop = self._round_price(symbol, resist + sl_dist) if atr > 0 else self._round_price(symbol, resist * 1.01)
            sr_tp = self._round_price(symbol, support) if support < price else self._round_price(symbol, price - atr * self._atr_tp_mult)
            atr_tp = self._round_price(symbol, price - atr * self._atr_tp_mult)
            sr_reward = abs(price - float(sr_tp))
            atr_reward = abs(price - float(atr_tp))
            min_reward = sl_dist * 1.5
            take = sr_tp if sr_reward >= min_reward else (atr_tp if atr_reward >= min_reward else self._round_price(symbol, price - min_reward))
            confidence = round(min(0.9, 0.6 + (rsi - 70.0) * 0.01), 3)
            return "short", price, stop, take, confidence

        return None

    # ------------------------------------------------------------------
    # 持仓管理
    # ------------------------------------------------------------------
    async def _manage_positions(self):
        for symbol, pos in list(self._position_state.items()):
            if pos.get("status") != "open":
                continue
            direction = pos.get("direction")
            price = await self._get_current_price(symbol)
            if price <= 0:
                continue
            stop_loss = self._safe_float(pos.get("stop_loss"), 0.0)
            take_profit = self._safe_float(pos.get("take_profit"), 0.0)
            quantity = self._safe_float(pos.get("current_quantity"), 0.0)
            entry_price = self._safe_float(pos.get("entry_price"), 0.0)
            if quantity <= 0 or entry_price <= 0:
                continue

            klines = await self._fetch_klines(symbol, limit=self._atr_period + 10)
            atr = 0.0
            if len(klines) >= self._atr_period + 2:
                _, highs, lows, closes, _ = self._klines_to_arrays(klines)
                atr = self._atr(highs, lows, closes, self._atr_period)

            # ── 分级止盈 ──
            if self._staged_tp_enabled and atr > 0:
                if direction == "long":
                    tp1_price = entry_price + atr * self._tp1_atr_mult
                    tp2_price = entry_price + atr * self._tp2_atr_mult
                else:
                    tp1_price = entry_price - atr * self._tp1_atr_mult
                    tp2_price = entry_price - atr * self._tp2_atr_mult

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
                    if direction == "long":
                        pos["stop_loss"] = entry_price
                        stop_loss = entry_price
                    else:
                        pos["stop_loss"] = entry_price
                        stop_loss = entry_price

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

            # ── ATR 尾随止损 ──
            if self._trailing_stop_enabled and atr > 0:
                if direction == "long":
                    new_stop = price - atr * self._trailing_atr_mult
                    if stop_loss <= 0 or new_stop > stop_loss:
                        pos["stop_loss"] = new_stop
                        stop_loss = new_stop
                else:
                    new_stop = price + atr * self._trailing_atr_mult
                    if stop_loss <= 0 or new_stop < stop_loss:
                        pos["stop_loss"] = new_stop
                        stop_loss = new_stop

            quantity = self._safe_float(pos.get("current_quantity"), 0.0)
            if quantity <= 0:
                continue

            hit_sl = (direction == "long" and stop_loss > 0 and price <= stop_loss) or \
                     (direction == "short" and stop_loss > 0 and price >= stop_loss)
            hit_tp = (direction == "long" and take_profit > 0 and price >= take_profit) or \
                     (direction == "short" and take_profit > 0 and price <= take_profit)

            if hit_sl:
                self._publish_exit_signal(symbol, direction, price, quantity, reason="stop_loss")
            elif hit_tp:
                self._publish_exit_signal(symbol, direction, price, quantity, reason="take_profit")

    # ------------------------------------------------------------------
    # 校验辅助
    # ------------------------------------------------------------------
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

    def _profit_covers_fees(self, price: float, direction: str, quantity: float, take_profit) -> bool:
        if take_profit is None:
            return True
        round_trip_fee = self._taker_fee_rate * 2
        fee_cost = price * quantity * round_trip_fee
        if direction == "long":
            expected_profit = (take_profit - price) * quantity
        else:
            expected_profit = (price - take_profit) * quantity
        return expected_profit > fee_cost * 1.5

    def get_health(self) -> Dict[str, Any]:
        base = super().get_health()
        base.update({
            "adx_threshold": self._adx_threshold,
            "breakout_lookback": self._breakout_lookback,
            "rsi_oversold": self._rsi_oversold,
            "rsi_overbought": self._rsi_overbought,
            "filter_stats": dict(self._filter_stats),
        })
        return base
