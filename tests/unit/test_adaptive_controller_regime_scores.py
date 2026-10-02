import pytest

from risk.adaptive_controller import AdaptiveController


def test_special_regime_position_factors_reach_allocation_scores():
    controller = AdaptiveController.__new__(AdaptiveController)
    controller._base_allocations = {
        "grid": 0.12,
        "trend": 0.25,
        "spot_grid": 0.12,
        "spot_martingale": 0.10,
    }
    controller._regime_engine = type(
        "RegimeEngine",
        (),
        {
            "get_position_adjustment": lambda self: {
                "overall": 0.6,
                "trend": 0.7,
                "grid": 0.5,
                "spot_grid": 0.5,
                "spot_martingale": 0.3,
            },
            "get_regime": lambda self: {"regime": "reversal"},
        },
    )()

    scores = controller._calculate_regime_scores()

    assert scores == {
        "grid": 0.5,
        "trend": 0.7,
        "spot_grid": 0.5,
        "spot_martingale": 0.3,
    }


@pytest.mark.parametrize(
    ("regime", "expected"),
    [
        ("breakout", "BREAKOUT"),
        ("breakdown", "BREAKDOWN"),
        ("reversal", "REVERSAL"),
    ],
)
def test_special_regimes_map_for_dynamic_allocator_sync(regime, expected):
    from risk.dynamic_allocator import MarketRegime

    controller = AdaptiveController.__new__(AdaptiveController)
    controller._config = {}
    controller._regime_engine = type(
        "RegimeEngine", (), {"get_regime": lambda self: {"regime": regime}}
    )()

    assert controller._map_market_regime() == getattr(MarketRegime, expected)