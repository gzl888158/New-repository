import pytest

from core.models import Signal


def _signal(**overrides):
    values = {
        "symbol": "BTC-USDT-SWAP",
        "strategy_name": "test",
        "signal_type": "open",
        "direction": "long",
        "price": 100.0,
        "quantity": 1.0,
        "leverage": 1,
    }
    values.update(overrides)
    return Signal(**values)


@pytest.mark.parametrize("direction", ["long", "short", "buy", "sell"])
def test_signal_accepts_supported_directions(direction):
    assert _signal(direction=direction).direction == direction


@pytest.mark.parametrize("confidence", [0.0, 1.0])
def test_signal_accepts_confidence_boundaries(confidence):
    assert _signal(confidence=confidence).confidence == confidence


@pytest.mark.parametrize("direction", ["up", "", None])
def test_signal_rejects_unsupported_directions(direction):
    with pytest.raises(ValueError, match="direction"):
        _signal(direction=direction)


@pytest.mark.parametrize("field_name", ["price", "quantity", "confidence"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_signal_rejects_non_finite_numeric_fields(field_name, value):
    with pytest.raises(ValueError, match=field_name):
        _signal(**{field_name: value})


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_signal_rejects_confidence_outside_unit_interval(confidence):
    with pytest.raises(ValueError, match="confidence"):
        _signal(confidence=confidence)