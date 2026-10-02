"""Trend-based trading decisions with ATR risk exits and standard signal objects."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from core.signal_generator import SignalLevel, SignalSource, SignalType, TradingSignal


@dataclass(frozen=True)
class DecisionConfig:
    """Thresholds and risk limits for trend-following decisions."""

    fast_period: int = 12
    slow_period: int = 26
    rsi_period: int = 14
    atr_period: int = 14
    min_trend_spread_pct: float = 0.001
    long_rsi_min: float = 50.0
    long_rsi_max: float = 70.0
    short_rsi_min: float = 30.0
    short_rsi_max: float = 50.0
    stop_atr_multiple: float = 2.0
    take_profit_atr_multiple: float = 3.0
    risk_per_trade: float = 0.01
    max_position_notional_pct: float = 0.25
    max_leverage: float = 1.0
    min_bars: int = 27

    def __post_init__(self) -> None:
        if self.fast_period < 2 or self.slow_period <= self.fast_period:
            raise ValueError("slow_period must be greater than fast_period >= 2")
        if min(self.rsi_period, self.atr_period, self.min_bars) < 2:
            raise ValueError("indicator periods and min_bars must be >= 2")
        rsi_thresholds = (self.long_rsi_min, self.long_rsi_max, self.short_rsi_min, self.short_rsi_max)
        if any(not math.isfinite(value) or not 0 <= value <= 100 for value in rsi_thresholds):
            raise ValueError("RSI thresholds must be finite values in [0, 100]")
        if self.long_rsi_min > self.long_rsi_max or self.short_rsi_min > self.short_rsi_max:
            raise ValueError("each RSI minimum must be <= its maximum")
        if not math.isfinite(self.min_trend_spread_pct) or self.min_trend_spread_pct < 0:
            raise ValueError("min_trend_spread_pct must be finite and non-negative")
        if not 0.0 < self.risk_per_trade <= 0.05:
            raise ValueError("risk_per_trade must be in (0, 0.05]")
        if not 0.0 < self.max_position_notional_pct <= 1.0:
            raise ValueError("max_position_notional_pct must be in (0, 1]")
        if not 0.0 < self.max_leverage <= 20.0:
            raise ValueError("max_leverage must be in (0, 20]")
        if (
            not math.isfinite(self.stop_atr_multiple)
            or not math.isfinite(self.take_profit_atr_multiple)
            or self.stop_atr_multiple <= 0
            or self.take_profit_atr_multiple <= 0
        ):
            raise ValueError("ATR multiples must be positive")


@dataclass(frozen=True)
class TrendAssessment:
    """Normalized trend and indicator data for an evaluated candle series."""

    direction: str
    strength: float
    fast_ema: float
    slow_ema: float
    rsi: float
    atr: float
    close: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "direction": self.direction,
            "strength": self.strength,
            "fast_ema": self.fast_ema,
            "slow_ema": self.slow_ema,
            "rsi": self.rsi,
            "atr": self.atr,
            "close": self.close,
        }


@dataclass(frozen=True)
class DecisionResult:
    """Signal plus the evidence and explanation used to produce it."""

    signal: TradingSignal
    trend: TrendAssessment | None
    reason: str


class DecisionEngine:
    """Identify trend direction and produce entry, exit, or hold signals.

    Candles may be close prices or mappings containing ``close`` and optionally
    ``high``/``low``. New entries require a valid equity value so order quantity
    can be constrained by both stop distance and the configured notional cap.
    """

    def __init__(self, config: DecisionConfig | Mapping[str, Any] | None = None) -> None:
        if config is None:
            self.config = DecisionConfig()
        elif isinstance(config, DecisionConfig):
            self.config = config
        else:
            self.config = DecisionConfig(**config)

    def decide(
        self,
        symbol: str,
        candles: Sequence[float | Mapping[str, Any]],
        position: Mapping[str, Any] | None = None,
        equity: float | None = None,
    ) -> DecisionResult:
        """Return a standard ``TradingSignal`` for the latest market snapshot."""
        if not symbol or not isinstance(symbol, str):
            raise ValueError("symbol must be a non-empty string")
        parsed = self._parse_candles(candles)
        required_bars = max(
            self.config.min_bars,
            self.config.slow_period,
            self.config.rsi_period + 1,
            self.config.atr_period,
        )
        if len(parsed) < required_bars:
            return self._hold(symbol, 0.0, "Insufficient candles for trend analysis", None)

        closes = [bar[2] for bar in parsed]
        fast_ema = self._ema(closes, self.config.fast_period)
        slow_ema = self._ema(closes, self.config.slow_period)
        rsi = self._rsi(closes, self.config.rsi_period)
        atr = self._atr(parsed, self.config.atr_period)
        close = closes[-1]
        strength = abs(fast_ema - slow_ema) / close
        if strength < self.config.min_trend_spread_pct:
            direction = "sideways"
        else:
            direction = "long" if fast_ema > slow_ema else "short"
        trend = TrendAssessment(direction, strength, fast_ema, slow_ema, rsi, atr, close)

        position_side, position_size, entry_price, position_id = self._position_details(position)
        if position_size > 0:
            exit_signal = self._exit_signal(
                symbol, close, atr, trend, position_side, position_size, entry_price, position_id
            )
            if exit_signal is not None:
                return DecisionResult(exit_signal, trend, exit_signal.reason)
            return self._hold(symbol, close, "Position remains within exit limits", trend)

        if direction == "long" and self.config.long_rsi_min <= rsi <= self.config.long_rsi_max:
            side = "long"
        elif direction == "short" and self.config.short_rsi_min <= rsi <= self.config.short_rsi_max:
            side = "short"
        else:
            return self._hold(symbol, close, "Trend or RSI confirmation is insufficient", trend)

        account_equity = self._positive_finite(equity)
        if account_equity is None:
            return self._hold(symbol, close, "Positive finite equity is required to size an entry", trend)

        stop_distance = self.config.stop_atr_multiple * atr
        if stop_distance <= 0 or not math.isfinite(stop_distance):
            return self._hold(symbol, close, "ATR is not usable for risk-based sizing", trend)
        risk_quantity = account_equity * self.config.risk_per_trade / stop_distance
        notional_cap = account_equity * self.config.max_position_notional_pct * self.config.max_leverage
        quantity = min(risk_quantity, notional_cap / close)
        if quantity <= 0 or not math.isfinite(quantity):
            return self._hold(symbol, close, "Calculated entry size is not positive and finite", trend)

        signal_type = SignalType.OPEN_LONG if side == "long" else SignalType.OPEN_SHORT
        stop_price = close - stop_distance if side == "long" else close + stop_distance
        target_distance = self.config.take_profit_atr_multiple * atr
        target_price = close + target_distance if side == "long" else close - target_distance
        confidence = min(0.99, max(0.5, strength / max(self.config.min_trend_spread_pct * 4, 1e-9)))
        signal = self._signal(
            symbol,
            signal_type,
            close,
            quantity,
            confidence,
            f"Confirmed {side} trend with RSI confirmation",
            {
                "trend": trend.to_dict(),
                "stop_price": stop_price,
                "take_profit_price": target_price,
                "risk_amount": quantity * stop_distance,
                "notional_cap": notional_cap,
            },
        )
        return DecisionResult(signal, trend, signal.reason)

    def _exit_signal(
        self,
        symbol: str,
        close: float,
        atr: float,
        trend: TrendAssessment,
        side: str,
        quantity: float,
        entry_price: float,
        position_id: str,
    ) -> TradingSignal | None:
        if entry_price > 0:
            stop_distance = self.config.stop_atr_multiple * atr
            target_distance = self.config.take_profit_atr_multiple * atr
            if side == "long" and close <= entry_price - stop_distance:
                signal_type, reason = SignalType.STOP_LOSS, "Long position crossed its ATR stop"
            elif side == "long" and close >= entry_price + target_distance:
                signal_type, reason = SignalType.TAKE_PROFIT, "Long position reached its ATR target"
            elif side == "short" and close >= entry_price + stop_distance:
                signal_type, reason = SignalType.STOP_LOSS, "Short position crossed its ATR stop"
            elif side == "short" and close <= entry_price - target_distance:
                signal_type, reason = SignalType.TAKE_PROFIT, "Short position reached its ATR target"
            elif side == "long" and trend.direction == "short":
                signal_type, reason = SignalType.CLOSE_ALL, "Confirmed downtrend reverses the long position"
            elif side == "short" and trend.direction == "long":
                signal_type, reason = SignalType.CLOSE_ALL, "Confirmed uptrend reverses the short position"
            else:
                return None
        elif side == "long" and trend.direction == "short":
            signal_type, reason = SignalType.CLOSE_ALL, "Confirmed downtrend reverses the long position"
        elif side == "short" and trend.direction == "long":
            signal_type, reason = SignalType.CLOSE_ALL, "Confirmed uptrend reverses the short position"
        else:
            return None
        return self._signal(
            symbol,
            signal_type,
            close,
            quantity,
            1.0,
            reason,
            {"trend": trend.to_dict(), "entry_price": entry_price},
            position_id=position_id,
        )

    def _hold(
        self, symbol: str, price: float, reason: str, trend: TrendAssessment | None
    ) -> DecisionResult:
        signal = self._signal(
            symbol,
            SignalType.HOLD,
            price,
            0.0,
            0.0,
            reason,
            {"trend": trend.to_dict() if trend else None},
        )
        return DecisionResult(signal, trend, reason)

    @staticmethod
    def _signal(
        symbol: str,
        signal_type: SignalType,
        price: float,
        quantity: float,
        weight: float,
        reason: str,
        metadata: dict[str, Any],
        position_id: str = "",
    ) -> TradingSignal:
        level = (
            SignalLevel.STRONG if weight >= 0.8
            else SignalLevel.MEDIUM if weight >= 0.5
            else SignalLevel.WEAK
        )
        return TradingSignal(
            symbol=symbol,
            signal_type=signal_type,
            source=SignalSource.TREND_BREAKOUT,
            level=level,
            weight=weight,
            price=price,
            quantity=quantity,
            timestamp=datetime.now(),
            reason=reason,
            metadata=metadata,
            position_id=position_id,
        )

    @staticmethod
    def _parse_candles(
        candles: Sequence[float | Mapping[str, Any]],
    ) -> list[tuple[float, float, float]]:
        if isinstance(candles, (str, bytes)):
            raise ValueError("candles must be a sequence of prices or candle mappings")
        parsed = []
        for index, candle in enumerate(candles):
            if isinstance(candle, Mapping):
                close = DecisionEngine._positive_finite(candle.get("close"))
                high = DecisionEngine._positive_finite(candle.get("high", close))
                low = DecisionEngine._positive_finite(candle.get("low", close))
            else:
                close = DecisionEngine._positive_finite(candle)
                high = low = close
            if (
                close is None
                or high is None
                or low is None
                or high < low
                or close > high
                or close < low
            ):
                raise ValueError(f"invalid OHLC data at candle index {index}")
            parsed.append((high, low, close))
        return parsed

    @staticmethod
    def _positive_finite(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) and number > 0 else None

    @staticmethod
    def _ema(values: Sequence[float], period: int) -> float:
        alpha = 2.0 / (period + 1)
        result = values[0]
        for value in values[1:]:
            result += alpha * (value - result)
        return result

    @staticmethod
    def _rsi(closes: Sequence[float], period: int) -> float:
        changes = [current - previous for previous, current in zip(closes[:-1], closes[1:], strict=True)]
        sample = changes[-period:]
        gains = sum(max(change, 0.0) for change in sample) / period
        losses = sum(max(-change, 0.0) for change in sample) / period
        if losses == 0:
            return 100.0 if gains > 0 else 50.0
        return 100.0 - 100.0 / (1.0 + gains / losses)

    @staticmethod
    def _atr(candles: Sequence[tuple[float, float, float]], period: int) -> float:
        true_ranges = []
        for index, (high, low, close) in enumerate(candles):
            previous_close = candles[index - 1][2] if index else close
            true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
        return sum(true_ranges[-period:]) / period

    @staticmethod
    def _position_details(
        position: Mapping[str, Any] | None,
    ) -> tuple[str, float, float, str]:
        if not isinstance(position, Mapping):
            return "", 0.0, 0.0, ""
        raw_size = position.get("size", position.get("quantity", 0.0))
        try:
            signed_size = float(raw_size)
        except (TypeError, ValueError, OverflowError):
            signed_size = 0.0
        if not math.isfinite(signed_size):
            signed_size = 0.0
        side = str(position.get("side", "")).lower()
        if side not in {"long", "short"}:
            side = "long" if signed_size > 0 else "short" if signed_size < 0 else ""
        try:
            entry_price = float(position.get("entry_price", 0.0))
        except (TypeError, ValueError, OverflowError):
            entry_price = 0.0
        if not math.isfinite(entry_price) or entry_price < 0:
            entry_price = 0.0
        return side, abs(signed_size), entry_price, str(position.get("position_id", ""))


__all__ = ["DecisionConfig", "DecisionEngine", "DecisionResult", "TrendAssessment"]
