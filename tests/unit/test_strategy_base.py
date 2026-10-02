import inspect

from strategies.base import StrategyBase
from strategies._trend_base import TrendStrategyBase


def test_strategy_base_requires_all_strategy_hooks():
    assert StrategyBase.__abstractmethods__ == {
        "_check_signals",
        "_manage_positions",
        "_get_risk_params",
    }


def test_trend_strategy_base_keeps_signal_and_position_hooks_abstract():
    assert inspect.isabstract(TrendStrategyBase)
    assert TrendStrategyBase.__abstractmethods__ == {"_check_signals", "_manage_positions"}