"""
多周期 EMA 趋势跟踪策略（主策略 A）

入场：短 EMA 上穿长 EMA 做多；短 EMA 下穿长 EMA 做空。
过滤：波动率阈值（ATR%）过滤震荡行情 + 波动率突破过滤器（D）增强。
出场：反向均线信号 / 移动止盈 / 固定止损。

标的：仅 BTC/ETH 主流永续合约。
"""
from __future__ import annotations

from typing import Any, Dict

from loguru import logger

from strategies._trend_base import TrendStrategyBase
from strategies.trend_sub_strategies import evaluate_ema_trend


class EmaTrendStrategy(TrendStrategyBase):
    STRATEGY_KEY = "ema_trend"
    DISPLAY_NAME = "EMA趋势跟踪"
    SYMBOL_SCOPE = "tier1"
    DEFAULT_BAR = "15m"
    ENTRY_SIGNAL_TYPE = "ema_trend_entry"
    EXIT_SIGNAL_TYPE = "ema_trend_exit"

    def __init__(self, config, okx_client, redis_cache):
        super().__init__(config, okx_client, redis_cache)
        self._ema_fast = self._safe_int(self._cfg.get("ema_fast", 9), 9)
        self._ema_slow = self._safe_int(self._cfg.get("ema_slow", 21), 21)
        self._atr_pct_threshold = self._safe_float(self._cfg.get("atr_pct_threshold", 0.003), 0.003)
        self._atr_sl_mult = self._safe_float(self._cfg.get("atr_sl_mult", 2.0), 2.0)
        self._atr_tp_mult = self._safe_float(self._cfg.get("atr_tp_mult", 3.0), 3.0)

    def _ema_params(self) -> Dict[str, Any]:
        return {
            "ema_fast": self._ema_fast,
            "ema_slow": self._ema_slow,
            "atr_pct_threshold": self._atr_pct_threshold,
            "atr_sl_mult": self._atr_sl_mult,
            "atr_tp_mult": self._atr_tp_mult,
        }

    async def _check_signals(self):
        open_count = sum(1 for p in self._position_state.values() if p.get("status") == "open")
        if open_count >= self._max_concurrent_positions:
            return

        for symbol in self._symbols:
            if self._position_state.get(symbol, {}).get("status") == "open":
                continue
            if symbol in self._last_signal_time:
                # 简单冷却，避免同一周期重复开仓
                from datetime import timedelta
                if (__import__("datetime").datetime.now() - self._last_signal_time[symbol]).total_seconds() < self._loop_interval:
                    continue

            # P33: 退出后冷却检查 — 防止止损后立即追入
            _last_exit = self._last_exit_time.get(symbol, 0)
            _cooldown_remaining = self._post_exit_cooldown - (__import__("time").time() - _last_exit)
            if _cooldown_remaining > 0:
                logger.debug(
                    f"P33: [{self._strategy_name}] {symbol} entry rejected — "
                    f"post-exit cooldown {_cooldown_remaining:.0f}s remaining"
                )
                continue

            klines = await self._fetch_klines(symbol, limit=120)
            if len(klines) < self._ema_slow + 2:
                continue
            _, highs, lows, closes, volumes = self._klines_to_arrays(klines)

            result = evaluate_ema_trend(closes, highs, lows, self._ema_params())
            direction = result.get("signal")
            if direction not in ("long", "short"):
                continue

            confidence = self._safe_float(result.get("confidence"), 0.0)
            if confidence < self._min_signal_quality:
                logger.debug(f"[{self._strategy_name}] {symbol} confidence {confidence:.2f} "
                             f"< {self._min_signal_quality}, 跳过")
                continue

            # 波动率突破过滤器（D）增强：启用且要求确认时，未确认则跳过
            filt = self._vol_breakout_filter.evaluate(closes, highs, lows, volumes, direction)
            if self._vol_breakout_filter.enabled and self._vol_breakout_filter.require_confirmation \
                    and not filt.get("confirmed"):
                logger.debug(f"[{self._strategy_name}] {symbol} 波动率突破未确认，跳过")
                continue

            # 资金费率增强：调整置信度
            confidence = await self._funding_enhancer.adjust_entry_confidence(symbol, direction, confidence)
            if confidence < self._min_signal_quality:
                continue

            price = float(closes[-1])
            stop_loss = self._round_price(symbol, result.get("stop_loss"))
            take_profit = self._round_price(symbol, result.get("take_profit"))
            if not self._validate_tp_sl(price, direction, stop_loss, take_profit):
                continue

            quantity, base_position = self._calculate_quantity(symbol, price)
            if quantity <= 0:
                logger.debug(f"[{self._strategy_name}] {symbol} 数量不足，跳过")
                continue

            # 预期收益覆盖手续费检查
            if not self._profit_covers_fees(price, direction, quantity, take_profit):
                logger.debug(f"[{self._strategy_name}] {symbol} 预期收益不足以覆盖手续费，跳过")
                continue

            self._publish_entry_signal(symbol, direction, price, quantity,
                                       stop_loss, take_profit, confidence)

    async def _manage_positions(self):
        """反向均线信号出场。"""
        for symbol, pos in list(self._position_state.items()):
            if pos.get("status") != "open":
                continue
            direction = pos.get("direction")
            klines = await self._fetch_klines(symbol, limit=120)
            if len(klines) < self._ema_slow + 2:
                continue
            _, highs, lows, closes, _ = self._klines_to_arrays(klines)
            result = evaluate_ema_trend(closes, highs, lows, self._ema_params())
            signal = result.get("signal")
            reverse = (direction == "long" and signal == "short") or \
                      (direction == "short" and signal == "long")
            if not reverse:
                continue
            price = float(closes[-1])
            self._publish_exit_signal(symbol, direction, price,
                                      self._safe_float(pos.get("current_quantity"), 0.0),
                                      reason="exit")

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
