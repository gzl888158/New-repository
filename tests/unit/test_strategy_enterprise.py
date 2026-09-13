"""
企业级策略接入层单元测试

覆盖：
1. EnterpriseStrategyMixin 的数值防御与参数校验（容错健壮性 / 风控合规）
2. TrendStrategy._analyze_period_with_indicators 的趋势判断安全访问（逻辑完善）
"""
import math
import pytest
from unittest.mock import MagicMock

from core.strategy_enterprise import EnterpriseStrategyMixin
from strategies.trend_strategy import TrendStrategy


class TestSafeConversion:
    """数值防御测试"""

    def test_safe_float_normal(self):
        m = EnterpriseStrategyMixin()
        assert m._safe_float("1.5") == 1.5
        assert m._safe_float(2.0) == 2.0
        assert m._safe_float(0) == 0.0

    def test_safe_float_nan_inf_none(self):
        m = EnterpriseStrategyMixin()
        assert m._safe_float(float("nan")) == 0.0
        assert m._safe_float(float("inf")) == 0.0
        assert m._safe_float(float("-inf")) == 0.0
        assert m._safe_float(None) == 0.0
        assert m._safe_float(None, default=-1.0) == -1.0

    def test_safe_float_invalid_string(self):
        m = EnterpriseStrategyMixin()
        assert m._safe_float("abc", default=7.0) == 7.0
        assert m._safe_float([1, 2], default=3.0) == 3.0

    def test_safe_int(self):
        m = EnterpriseStrategyMixin()
        assert m._safe_int("5") == 5
        assert m._safe_int(3.7) == 3
        assert m._safe_int(None, default=9) == 9
        assert m._safe_int("xyz", default=1) == 1
        assert m._safe_int("4.9", default=0) == 4


class TestValidation:
    """参数校验测试"""

    def test_validate_symbol_valid(self):
        m = EnterpriseStrategyMixin()
        assert m._validate_symbol("BTC-USDT-SWAP") is True
        assert m._validate_symbol("ETH-USDT") is True

    def test_validate_symbol_invalid(self):
        m = EnterpriseStrategyMixin()
        assert m._validate_symbol("") is False
        assert m._validate_symbol(None) is False
        assert m._validate_symbol("BTC-USD") is False
        assert m._validate_symbol(123) is False

    def test_validate_direction(self):
        m = EnterpriseStrategyMixin()
        for d in ("long", "short", "buy", "sell"):
            assert m._validate_direction(d) is True
        assert m._validate_direction("invalid") is False
        assert m._validate_direction(None) is False

    def test_validate_confidence(self):
        m = EnterpriseStrategyMixin()
        assert m._validate_confidence(0.5) is True
        assert m._validate_confidence(0.0) is True
        assert m._validate_confidence(1.0) is True
        assert m._validate_confidence(-0.1) is False
        assert m._validate_confidence(1.5) is False
        assert m._validate_confidence(None) is False
        assert m._validate_confidence(float("nan")) is False

    def test_validate_quantity(self):
        m = EnterpriseStrategyMixin()
        assert m._validate_quantity(1.0) is True
        assert m._validate_quantity(0.0) is False
        assert m._validate_quantity(-1.0) is False
        assert m._validate_quantity(None) is False
        assert m._validate_quantity(0.5, min_qty=0.1) is True
        assert m._validate_quantity(0.05, min_qty=0.1) is False

    def test_validate_price(self):
        m = EnterpriseStrategyMixin()
        assert m._validate_price(100.0) is True
        assert m._validate_price(0.0) is False
        assert m._validate_price(-5.0) is False
        assert m._validate_price(None) is False


def _make_trend():
    """构造轻量 TrendStrategy 实例（跳过 __init__，仅测试趋势判断逻辑）。"""
    s = TrendStrategy.__new__(TrendStrategy)
    s._rsi_oversold = 30
    s._rsi_overbought = 70
    s._adx_threshold = 25
    s._divergence_detection = False
    s._market_structure_enabled = False
    return s


class TestAnalyzePeriodWithIndicators:
    """趋势判断安全访问与打分逻辑测试"""

    def test_empty_or_none_indicators(self):
        s = _make_trend()
        assert s._analyze_period_with_indicators({}) == (None, 0)
        assert s._analyze_period_with_indicators(None) == (None, 0)

    def test_missing_core_indicators(self):
        s = _make_trend()
        # ma20 无效 -> 直接放弃
        assert s._analyze_period_with_indicators({"ma20": 0}) == (None, 0)
        # recent_close 缺失 -> 默认 0 -> 直接放弃
        assert s._analyze_period_with_indicators({"ma20": 100, "ma50": 95}) == (None, 0)

    def test_nan_defense(self):
        s = _make_trend()
        ind = {"ma20": float("nan"), "ma50": 95, "recent_close": 105}
        assert s._analyze_period_with_indicators(ind) == (None, 0)

    def test_bullish_signal(self):
        s = _make_trend()
        ind = {
            "ma20": 100.0, "ma50": 95.0,
            "vwma20": 101.0, "vwma50": 96.0,
            "rsi": 55.0,
            "macd": 0.5, "signal_line": 0.1, "histogram": 0.4,
            "adx": 30.0, "+di": 26.0, "-di": 14.0,
            "recent_close": 105.0, "recent_high": 106.0, "recent_low": 97.0,
            "recent_volume": 2000.0, "volume_ma20": 1000.0,
        }
        direction, confidence = s._analyze_period_with_indicators(ind)
        assert direction == "long"
        assert 0.4 <= confidence <= 0.95

    def test_bearish_signal(self):
        s = _make_trend()
        ind = {
            "ma20": 98.0, "ma50": 100.0,
            "vwma20": 96.0, "vwma50": 101.0,
            "rsi": 45.0,
            "macd": -0.5, "signal_line": -0.1, "histogram": -0.4,
            "adx": 30.0, "+di": 14.0, "-di": 26.0,
            "recent_close": 95.0, "recent_high": 103.0, "recent_low": 96.0,
            "recent_volume": 2000.0, "volume_ma20": 1000.0,
        }
        direction, confidence = s._analyze_period_with_indicators(ind)
        assert direction == "short"
        assert 0.4 <= confidence <= 0.95

    def test_no_clear_direction(self):
        s = _make_trend()
        ind = {
            "ma20": 100.0, "ma50": 100.0,
            "vwma20": 100.0, "vwma50": 100.0,
            "rsi": 50.0,
            "macd": 0.0, "signal_line": 0.0, "histogram": 0.0,
            "adx": 10.0, "+di": 10.0, "-di": 10.0,
            "recent_close": 100.0, "recent_high": 100.0, "recent_low": 100.0,
            "recent_volume": 1000.0, "volume_ma20": 1000.0,
        }
        direction, _ = s._analyze_period_with_indicators(ind)
        assert direction is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
