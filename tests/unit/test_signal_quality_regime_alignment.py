from types import SimpleNamespace

import pytest

from services.signal_quality_engine import SignalQualityEngine
from services.signal_processor import SignalProcessor


@pytest.mark.parametrize(
    ("regime", "side", "direction_key", "expected"),
    [
        ("breakout", "long", "direction", 0.94),
        ("breakout", "short", "direction", 0.14),
        ("breakdown", "short", "direction", 0.94),
        ("breakdown", "long", "direction", 0.14),
        ("reversal", "long", "direction", 0.35),
        ("breakout", "long", "side", 0.94),
        ("breakout", "long", "both", 0.94),
    ],
)
def test_trend_alignment_scores_fused_special_regimes(regime, side, direction_key, expected):
    regime_engine = SimpleNamespace(
        get_regime=lambda: {"regime": regime, "strength": 0.8}
    )
    quality_engine = SignalQualityEngine({}, regime_engine=regime_engine)

    signal = {direction_key: side} if direction_key != "both" else {"direction": "", "side": side}
    score = quality_engine._score_trend_alignment(signal)

    assert score == pytest.approx(expected)


@pytest.mark.parametrize(
    ("regime", "strategy", "direction", "issue"),
    [
        ("breakout", "trend", "short", "short_against_breakout"),
        ("breakdown", "trend", "long", "long_against_breakdown"),
        ("breakout", "grid", "long", "mean_reversion_in_breakout"),
        ("breakdown", "spot_grid", "short", "mean_reversion_in_breakdown"),
        ("reversal", "trend", "long", "unconfirmed_reversal"),
    ],
)
def test_fallback_quality_penalizes_special_regime_mismatch(regime, strategy, direction, issue):
    processor = SignalProcessor(
        {
            "currencies": {"tier1_symbols": ["BTC"]},
            "strategies": {strategy: {"enabled": True}},
        },
        None, None, None, None, None, None, None, None,
    )
    processor.set_regime_engine(SimpleNamespace(
        get_regime=lambda: {"regime": regime, "strength": 0.8}
    ))

    accepted, breakdown = processor._evaluate_signal_quality({
        "symbol": "BTC-USDT-SWAP",
        "direction": direction,
        "price": 100.0,
        "quantity": 1.0,
        "confidence": 0.9,
        "strategy_name": strategy,
    })

    assert accepted is True
    assert issue in breakdown["issues"]