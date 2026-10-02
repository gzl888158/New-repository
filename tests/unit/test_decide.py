import math

import pytest

from core.decide import DecisionConfig, DecisionEngine
from core.signal_generator import SignalType


def _rising_confirmed_prices():
    return [100 + index * 0.08 + 1.3 * math.sin(index * 0.9) for index in range(60)]


def test_trend_entry_is_sized_by_risk_and_notional_caps():
    engine = DecisionEngine()
    result = engine.decide("BTC-USDT", _rising_confirmed_prices(), equity=10_000)

    assert result.signal.signal_type == SignalType.OPEN_LONG
    assert result.trend.direction == "long"
    assert 0 < result.signal.quantity
    assert result.signal.metadata["risk_amount"] <= 100.000001
    assert result.signal.metadata["notional_cap"] == 2_500


def test_missing_equity_fails_closed_for_new_entry():
    result = DecisionEngine().decide("BTC-USDT", _rising_confirmed_prices())

    assert result.signal.signal_type == SignalType.HOLD
    assert result.signal.quantity == 0
    assert "equity" in result.reason


def test_existing_position_uses_stop_loss_before_reversal():
    prices = [100 + index * 0.2 for index in range(60)]
    result = DecisionEngine().decide(
        "BTC-USDT",
        prices,
        position={"side": "long", "size": 0.5, "entry_price": prices[-1] + 100},
    )

    assert result.signal.signal_type == SignalType.STOP_LOSS
    assert result.signal.quantity == 0.5


def test_existing_position_closes_on_confirmed_trend_reversal_without_entry_price():
    prices = [200 - index * 0.5 for index in range(60)]
    result = DecisionEngine().decide(
        "BTC-USDT", prices, position={"side": "long", "size": 0.5}
    )

    assert result.signal.signal_type == SignalType.CLOSE_ALL


def test_insufficient_history_holds_and_invalid_ohlc_is_rejected():
    engine = DecisionEngine()
    result = engine.decide("BTC-USDT", [100.0] * 10, equity=1_000)
    assert result.signal.signal_type == SignalType.HOLD

    with pytest.raises(ValueError, match="invalid OHLC"):
        engine.decide("BTC-USDT", [{"high": 99, "low": 100, "close": 100}])


def test_config_rejects_invalid_rsi_thresholds():
    with pytest.raises(ValueError, match="RSI minimum"):
        DecisionConfig(long_rsi_min=75, long_rsi_max=70)
