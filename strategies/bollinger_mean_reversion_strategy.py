"""
带趋势过滤器的布林均值回归策略（备选策略 E，轻仓运行）

规则：仅大趋势背景下，回调至轨道边界逆势短线博弈。
强制限制：趋势强度不足时策略休眠；禁止单边逆势加仓（每品种单持仓）。

标的：仅 BTC/ETH 主流永续合约。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from loguru import logger

from strategies._trend_base import TrendStrategyBase
from strategies.trend_sub_strategies import _ema, _atr


class BollingerMeanReversionStrategy(TrendStrategyBase):
    STRATEGY_KEY = "bollinger_mean_reversion"
    DISPLAY_NAME = "布林均值回归"
    SYMBOL_SCOPE = "tier1"
    DEFAULT_BAR = "15m"
    ENTRY_SIGNAL_TYPE = "bollinger_mean_reversion_entry"
    EXIT_SIGNAL_TYPE = "bollinger_mean_reversion_exit"

    def __init__(self, config, okx_client, redis_cache):
        super().__init__(config, okx_client, redis_cache)
        self._boll_period = self._safe_int(self._cfg.get("boll_period", 20), 20)
        self._boll_std_mult = self._safe_float(self._cfg.get("boll_std_mult", 2.0), 2.0)
        self._trend_ema_period = self._safe_int(self._cfg.get("trend_ema_period", 50), 50)
        self._trend_lookback = self._safe_int(self._cfg.get("trend_lookback", 10), 10)
        # 趋势强度不足时休眠：ATR% 低于阈值不逆势开仓
        self._atr_pct_threshold = self._safe_float(self._cfg.get("atr_pct_threshold", 0.002), 0.002)
        self._atr_sl_mult = self._safe_float(self._cfg.get("atr_sl_mult", 2.0), 2.0)

    def _evaluate(self, closes, highs, lows) -> Optional[Dict[str, Any]]:
        if len(closes) < self._boll_period + 1:
            return None
        closes = np.asarray(closes, dtype=float)
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)

        mid = float(np.mean(closes[-self._boll_period:]))
        std = float(np.std(closes[-self._boll_period:]))
        if std <= 0:
            return None
        upper = mid + self._boll_std_mult * std
        lower = mid - self._boll_std_mult * std
        price = float(closes[-1])

        atr = _atr(highs, lows, closes, 14)
        atr_pct = (atr / price) if price > 0 and atr > 0 else 0.0

        ema_now = _ema(closes, self._trend_ema_period)
        ema_prev = None
        if len(closes) > self._trend_ema_period + self._trend_lookback:
            ema_prev = _ema(closes[:-self._trend_lookback], self._trend_ema_period)

        trend = "neutral"
        if ema_now is not None and ema_prev is not None and ema_prev > 0:
            slope = (ema_now - ema_prev) / ema_prev
            if slope > 0:
                trend = "up"
            elif slope < 0:
                trend = "down"

        direction = None
        band_depth = 0.0
        if trend == "up" and price <= lower:
            direction = "long"
            band_depth = (lower - price) / lower if lower > 0 else 0.0
        elif trend == "down" and price >= upper:
            direction = "short"
            band_depth = (price - upper) / upper if upper > 0 else 0.0

        if direction is None:
            return {"signal": None, "trend": trend, "atr_pct": atr_pct,
                    "mid": mid, "upper": upper, "lower": lower}

        confidence = min(0.9, 0.5 + min(0.3, band_depth * 5.0))
        sl_offset = atr * self._atr_sl_mult if atr > 0 else 0.0
        stop_loss = (price - sl_offset) if direction == "long" else (price + sl_offset)
        # 均值回归目标：回归中轨
        take_profit = mid

        return {
            "signal": direction,
            "confidence": confidence,
            "trend": trend,
            "atr_pct": atr_pct,
            "mid": mid,
            "upper": upper,
            "lower": lower,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    async def _check_signals(self):
        open_count = sum(1 for p in self._position_state.values() if p.get("status") == "open")
        if open_count >= self._max_concurrent_positions:
            return

        for symbol in self._symbols:
            if self._position_state.get(symbol, {}).get("status") == "open":
                continue

            klines = await self._fetch_klines(symbol, limit=150)
            if len(klines) < self._boll_period + 1:
                continue
            _, highs, lows, closes, volumes = self._klines_to_arrays(klines)

            result = self._evaluate(closes, highs, lows)
            if not result:
                continue
            direction = result.get("signal")
            if direction not in ("long", "short"):
                continue

            # 趋势强度不足 → 休眠
            atr_pct = self._safe_float(result.get("atr_pct"), 0.0)
            if atr_pct < self._atr_pct_threshold:
                logger.debug(f"[{self._strategy_name}] {symbol} 趋势强度不足 "
                             f"(ATR% {atr_pct:.4f}), 休眠")
                continue

            confidence = self._safe_float(result.get("confidence"), 0.0)
            if confidence < self._min_signal_quality:
                continue

            confidence = await self._funding_enhancer.adjust_entry_confidence(symbol, direction, confidence)
            if confidence < self._min_signal_quality:
                continue

            price = float(closes[-1])
            stop_loss = self._round_price(symbol, result.get("stop_loss"))
            take_profit = self._round_price(symbol, result.get("take_profit"))
            if not self._validate_tp_sl(price, direction, stop_loss, take_profit):
                continue

            quantity, _ = self._calculate_quantity(symbol, price)
            if quantity <= 0:
                continue

            if not self._profit_covers_fees(price, direction, quantity, take_profit):
                continue

            self._publish_entry_signal(symbol, direction, price, quantity,
                                       stop_loss, take_profit, confidence)

    async def _manage_positions(self):
        """回归中轨止盈 / 反向信号出场。"""
        for symbol, pos in list(self._position_state.items()):
            if pos.get("status") != "open":
                continue
            direction = pos.get("direction")
            klines = await self._fetch_klines(symbol, limit=150)
            if len(klines) < self._boll_period + 1:
                continue
            _, highs, lows, closes, _ = self._klines_to_arrays(klines)
            result = self._evaluate(closes, highs, lows)
            if not result:
                continue
            price = float(closes[-1])
            mid = self._safe_float(result.get("mid"), 0.0)
            signal = result.get("signal")
            quantity = self._safe_float(pos.get("current_quantity"), 0.0)

            reverse = (direction == "long" and signal == "short") or \
                      (direction == "short" and signal == "long")
            # 回归中轨止盈
            reverted = (direction == "long" and mid > 0 and price >= mid) or \
                       (direction == "short" and mid > 0 and price <= mid)

            if reverse or reverted:
                reason = "exit" if reverse else "take_profit"
                self._publish_exit_signal(symbol, direction, price, quantity, reason=reason)

    # ------------------------------------------------------------------
    # 校验辅助
    # ------------------------------------------------------------------
    def _validate_tp_sl(self, price: float, direction: str,
                        stop_loss, take_profit) -> bool:
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
