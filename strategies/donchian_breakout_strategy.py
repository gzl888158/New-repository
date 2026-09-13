"""
唐奇安通道突破策略（主策略 B，海龟逻辑）

入场：N 周期新高做多，N 周期新低做空。
出场：ATR 尾随止盈止损 / 反向突破出场。
限制：仅在趋势判定开启（ATR% 不低于阈值），震荡区间暂停。

标的：仅 BTC/ETH 主流永续合约。
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from loguru import logger

from strategies._trend_base import TrendStrategyBase
from strategies.trend_sub_strategies import evaluate_donchian


class DonchianBreakoutStrategy(TrendStrategyBase):
    STRATEGY_KEY = "donchian_breakout"
    DISPLAY_NAME = "唐奇安通道突破"
    SYMBOL_SCOPE = "tier1"
    DEFAULT_BAR = "15m"
    ENTRY_SIGNAL_TYPE = "donchian_breakout_entry"
    EXIT_SIGNAL_TYPE = "donchian_breakout_exit"

    def __init__(self, config, okx_client, redis_cache):
        super().__init__(config, okx_client, redis_cache)
        self._donchian_period = self._safe_int(self._cfg.get("donchian_period", 20), 20)
        self._atr_sl_mult = self._safe_float(self._cfg.get("atr_sl_multiplier", 2.0), 2.0)
        self._atr_tp_mult = self._safe_float(self._cfg.get("atr_tp_multiplier", 3.0), 3.0)
        # 趋势判定：ATR% 低于阈值视为震荡，暂停开仓
        self._atr_pct_threshold = self._safe_float(self._cfg.get("atr_pct_threshold", 0.002), 0.002)

        # ADX 入场上限过滤（回测验证：ADX>=25 追高是 2025 亏损主因）
        # 默认关闭，避免改变现有策略行为；启用后仅允许 ADX < 阈值的突破入场。
        self._adx_max_filter_enabled = bool(self._cfg.get("adx_max_filter_enabled", False))
        self._adx_max_threshold = self._safe_float(self._cfg.get("adx_max_threshold", 25.0), 25.0)
        self._adx_period = self._safe_int(self._cfg.get("adx_period", 14), 14)

        # 单币种收益贡献 cap（回测验证：cap=15% 可降低集中度并提升稳健性）
        # 默认关闭；启用后当某币种累计收益占总收益比例 >= cap 时暂停该币种开仓。
        self._symbol_pnl_cap_enabled = bool(self._cfg.get("symbol_pnl_cap_enabled", False))
        self._symbol_pnl_cap_ratio = self._safe_float(self._cfg.get("symbol_pnl_cap_ratio", 0.15), 0.15)
        self._symbol_pnl: Dict[str, float] = {}
        self._total_pnl: float = 0.0

        # 企业级增强：风险预算感知动态阈值（连续亏损/熔断时上浮信号质量阈值收紧开仓）
        self._adaptive_enabled = bool(self._cfg.get("adaptive_enabled", True))
        self._risk_lock_quality_boost = self._safe_float(self._cfg.get("risk_lock_quality_boost", 0.10), 0.10)
        # 企业级增强：过滤理由本地计数（get_health 暴露 + MetricsPipeline 埋点）
        self._filter_stats: Dict[str, int] = {}

    def _dynamic_min_quality(self) -> float:
        """自适应信号质量阈值：连续亏损/风险熔断时上浮，收紧开仓。"""
        quality = self._min_signal_quality
        if not self._adaptive_enabled or self._adaptive_controller is None:
            return quality
        try:
            status = self._adaptive_controller.get_risk_budget_status()
            if status.get("streak_lock_active"):
                quality += self._risk_lock_quality_boost
        except Exception:
            pass
        return quality

    def _record_filter(self, symbol: str, reason: str):
        """记录一次过滤/拒绝原因（本地计数 + MetricsPipeline 埋点）。"""
        self._filter_stats[reason] = self._filter_stats.get(reason, 0) + 1
        try:
            self._increment_metric(f"{self._strategy_name}_filter_total", 1.0,
                                   {"reason": reason, "symbol": symbol})
        except Exception:
            pass
        logger.debug(f"[{self._strategy_name}] filter: {symbol} {reason}")

    def _record_exit_metrics(self, symbol: str, direction: str, entry_price: float,
                             exit_price: float, quantity: float, reason: str):
        """出场埋点：记录平仓笔数与已实现盈亏（本地 + MetricsPipeline）。"""
        ret = 0.0
        if entry_price > 0:
            ret = (exit_price - entry_price) / entry_price if direction == "long" \
                else (entry_price - exit_price) / entry_price
        pnl_usdt = ret * entry_price * quantity
        try:
            self._record_metric(f"{self._strategy_name}_position_closed_total", 1.0,
                                {"reason": reason, "symbol": symbol, "direction": direction})
            self._record_metric(f"{self._strategy_name}_position_pnl", pnl_usdt,
                                {"reason": reason, "symbol": symbol})
        except Exception:
            pass

    def _donchian_params(self) -> Dict[str, Any]:
        return {
            "donchian_period": self._donchian_period,
            "atr_sl_multiplier": self._atr_sl_mult,
            "atr_tp_multiplier": self._atr_tp_mult,
        }

    @staticmethod
    def _wilder_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                    period: int = 14) -> float:
        """标准 Wilder ADX 最新值（与回测 _adx_series 同源）。数据不足返回 -1。"""
        highs = np.asarray(highs, dtype=float)
        lows = np.asarray(lows, dtype=float)
        closes = np.asarray(closes, dtype=float)
        n = len(closes)
        if n <= 2 * period:
            return -1.0
        if np.any(np.isnan(highs)) or np.any(np.isnan(lows)) or np.any(np.isnan(closes)):
            return -1.0

        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(highs[i] - lows[i],
                        abs(highs[i] - closes[i - 1]),
                        abs(lows[i] - closes[i - 1]))
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

    def _adx_allows_entry(self, highs: np.ndarray, lows: np.ndarray,
                          closes: np.ndarray) -> bool:
        """ADX 上限过滤：仅在 ADX < 阈值时允许入场。未启用或无法计算时放行。"""
        if not self._adx_max_filter_enabled:
            return True
        adx = self._wilder_adx(highs, lows, closes, self._adx_period)
        if adx < 0:
            return True
        return adx < self._adx_max_threshold

    def _pnl_cap_allows_entry(self, symbol: str) -> bool:
        """单币种收益贡献 cap：累计收益占比超阈值时暂停该币种开仓。未启用时放行。"""
        if not self._symbol_pnl_cap_enabled:
            return True
        if self._total_pnl <= 0:
            return True
        return (self._symbol_pnl.get(symbol, 0.0) / self._total_pnl) < self._symbol_pnl_cap_ratio

    def _record_realized_pnl(self, symbol: str, direction: str,
                             entry_price: float, exit_price: float):
        """以价格收益率作为收益贡献度量，累加到币种与总账户。"""
        if not self._symbol_pnl_cap_enabled:
            return
        if entry_price <= 0:
            return
        ret = (exit_price - entry_price) / entry_price if direction == "long" \
            else (entry_price - exit_price) / entry_price
        self._symbol_pnl[symbol] = self._symbol_pnl.get(symbol, 0.0) + ret
        self._total_pnl += ret

    async def _check_signals(self):
        open_count = sum(1 for p in self._position_state.values() if p.get("status") == "open")
        if open_count >= self._max_concurrent_positions:
            self._record_filter("", "max_positions_reached")
            return

        # 企业级增强：动态信号质量阈值（连续亏损/熔断时收紧）
        dynamic_min_quality = self._dynamic_min_quality()

        for symbol in self._symbols:
            if self._position_state.get(symbol, {}).get("status") == "open":
                self._record_filter(symbol, "position_exists")
                continue

            klines = await self._fetch_klines(symbol, limit=120)
            if len(klines) < self._donchian_period + 2:
                self._record_filter(symbol, "insufficient_klines")
                continue
            _, highs, lows, closes, volumes = self._klines_to_arrays(klines)

            result = evaluate_donchian(highs, lows, closes, self._donchian_params())
            direction = result.get("signal")
            if direction not in ("long", "short"):
                self._record_filter(symbol, "no_signal")
                continue

            # ADX 入场上限过滤：仅在 ADX < 阈值时允许突破入场（默认关闭）
            if not self._adx_allows_entry(highs, lows, closes):
                logger.debug(f"[{self._strategy_name}] {symbol} ADX 过高，跳过入场")
                self._record_filter(symbol, "adx_too_high")
                continue

            # 趋势判定：震荡区间暂停（ATR% 过低）
            price = float(closes[-1])
            atr = self._safe_float(result.get("atr"), 0.0)
            atr_pct = (atr / price) if price > 0 and atr > 0 else 0.0
            if atr_pct < self._atr_pct_threshold:
                logger.debug(f"[{self._strategy_name}] {symbol} ATR% {atr_pct:.4f} "
                             f"< {self._atr_pct_threshold}, 震荡暂停")
                self._record_filter(symbol, "low_volatility_ranging")
                continue

            confidence = self._safe_float(result.get("confidence"), 0.0)
            if confidence < dynamic_min_quality:
                logger.debug(f"[{self._strategy_name}] {symbol} confidence {confidence:.2f} "
                             f"< {dynamic_min_quality}, 跳过")
                self._record_filter(symbol, "signal_quality_low")
                continue

            # 波动率突破过滤器（D）增强
            filt = self._vol_breakout_filter.evaluate(closes, highs, lows, volumes, direction)
            if self._vol_breakout_filter.enabled and self._vol_breakout_filter.require_confirmation \
                    and not filt.get("confirmed"):
                logger.debug(f"[{self._strategy_name}] {symbol} 波动率突破未确认，跳过")
                self._record_filter(symbol, "vol_breakout_unconfirmed")
                continue

            # 资金费率增强：调整置信度
            confidence = await self._funding_enhancer.adjust_entry_confidence(symbol, direction, confidence)
            if confidence < dynamic_min_quality:
                self._record_filter(symbol, "signal_quality_low")
                continue

            stop_loss = self._round_price(symbol, result.get("stop_loss"))
            take_profit = self._round_price(symbol, result.get("take_profit"))
            if not self._validate_tp_sl(price, direction, stop_loss, take_profit):
                self._record_filter(symbol, "tp_sl_validation_failed")
                continue

            quantity, base_position = self._calculate_quantity(symbol, price)
            if quantity <= 0:
                logger.debug(f"[{self._strategy_name}] {symbol} 数量不足，跳过")
                self._record_filter(symbol, "quantity_below_threshold")
                continue

            if not self._profit_covers_fees(price, direction, quantity, take_profit):
                logger.debug(f"[{self._strategy_name}] {symbol} 预期收益不足以覆盖手续费，跳过")
                self._record_filter(symbol, "profit_covers_fees_failed")
                continue

            # 单币种收益贡献 cap：累计收益占比超阈值时暂停该币种开仓（默认关闭）
            if not self._pnl_cap_allows_entry(symbol):
                logger.debug(f"[{self._strategy_name}] {symbol} 收益贡献占比超限，暂停开仓")
                self._record_filter(symbol, "symbol_pnl_cap")
                continue

            self._publish_entry_signal(symbol, direction, price, quantity,
                                       stop_loss, take_profit, confidence)

    async def _manage_positions(self):
        """ATR 尾随止损 + 反向突破出场。"""
        for symbol, pos in list(self._position_state.items()):
            if pos.get("status") != "open":
                continue
            direction = pos.get("direction")
            klines = await self._fetch_klines(symbol, limit=120)
            if len(klines) < self._donchian_period + 2:
                continue
            _, highs, lows, closes, _ = self._klines_to_arrays(klines)
            result = evaluate_donchian(highs, lows, closes, self._donchian_params())
            price = float(closes[-1])
            atr = self._safe_float(result.get("atr"), 0.0)
            quantity = self._safe_float(pos.get("current_quantity"), 0.0)
            entry_price = self._safe_float(pos.get("entry_price"), 0.0)

            # 反向突破出场
            signal = result.get("signal")
            reverse = (direction == "long" and signal == "short") or \
                      (direction == "short" and signal == "long")
            if reverse:
                self._record_realized_pnl(symbol, direction, entry_price, price)
                self._record_exit_metrics(symbol, direction, entry_price, price, quantity, "exit")
                self._publish_exit_signal(symbol, direction, price, quantity, reason="exit")
                continue

            # ATR 尾随止损
            if atr <= 0:
                continue
            cur_stop = self._safe_float(pos.get("stop_loss"), 0.0)
            if direction == "long":
                new_stop = price - atr * self._atr_sl_mult
                if cur_stop <= 0 or new_stop > cur_stop:
                    pos["stop_loss"] = new_stop
                if cur_stop > 0 and price <= cur_stop:
                    self._record_realized_pnl(symbol, direction, entry_price, price)
                    self._record_exit_metrics(symbol, direction, entry_price, price, quantity, "stop_loss")
                    self._publish_exit_signal(symbol, direction, price, quantity, reason="stop_loss")
            else:
                new_stop = price + atr * self._atr_sl_mult
                if cur_stop <= 0 or new_stop < cur_stop:
                    pos["stop_loss"] = new_stop
                if cur_stop > 0 and price >= cur_stop:
                    self._record_realized_pnl(symbol, direction, entry_price, price)
                    self._record_exit_metrics(symbol, direction, entry_price, price, quantity, "stop_loss")
                    self._publish_exit_signal(symbol, direction, price, quantity, reason="stop_loss")

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

    # ------------------------------------------------------------------
    # 状态持久化（在基类基础上追加收益贡献 cap 的累计状态）
    # ------------------------------------------------------------------
    def get_health(self) -> Dict[str, Any]:
        """企业级增强：暴露自适应调参状态与过滤埋点统计，供 Dashboard / 健康检查消费。"""
        base = super().get_health()
        base.update({
            "adaptive_enabled": self._adaptive_enabled,
            "min_signal_quality": self._min_signal_quality,
            "dynamic_min_quality": self._dynamic_min_quality(),
            "risk_lock_quality_boost": self._risk_lock_quality_boost,
            "adaptive_controller_injected": self._adaptive_controller is not None,
            "filter_stats": dict(self._filter_stats),
            "adx_max_filter_enabled": self._adx_max_filter_enabled,
            "adx_max_threshold": self._adx_max_threshold,
            "symbol_pnl_cap_enabled": self._symbol_pnl_cap_enabled,
        })
        return base

    def collect_persistent_state(self) -> Dict[str, Any]:
        state = super().collect_persistent_state()
        state["_symbol_pnl"] = dict(self._symbol_pnl)
        state["_total_pnl"] = self._total_pnl
        return state

    def restore_persistent_state(self, state: Dict[str, Any]):
        super().restore_persistent_state(state)
        raw = state.get("_symbol_pnl", {}) or {}
        self._symbol_pnl = {str(k): self._safe_float(v, 0.0) for k, v in raw.items()}
        self._total_pnl = self._safe_float(state.get("_total_pnl"), 0.0)
