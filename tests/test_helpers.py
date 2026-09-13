"""
helpers.py 核心计算函数单元测试
"""
import math
import pytest
from utils.helpers import (
    calculate_position_size,
    calculate_margin,
    calculate_pnl,
    calculate_pnl_percent,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_take_profit,
    calculate_stop_loss,
    calculate_tp_sl_from_atr,
    validate_tp_sl_prices,
    calculate_risk_reward_ratio,
    calculate_trailing_stop,
    adjust_stop_loss_for_volatility,
    check_profit_protection,
    evaluate_signal_quality,
    clamp,
)


class TestPositionCalculations:
    """仓位计算相关测试"""

    def test_calculate_position_size_normal(self):
        qty = calculate_position_size(1000, 100, 5, 0.01)
        assert abs(qty - 0.5) < 1e-10

    def test_calculate_position_size_zero_price(self):
        qty = calculate_position_size(1000, 0, 5, 0.01)
        assert qty == 0.0

    def test_calculate_position_size_negative_price(self):
        qty = calculate_position_size(1000, -10, 5, 0.01)
        assert qty == 0.0

    def test_calculate_position_size_zero_leverage(self):
        qty = calculate_position_size(1000, 100, 0, 0.01)
        assert qty == 0.0

    def test_calculate_margin_normal(self):
        margin = calculate_margin(1, 100, 5)
        assert abs(margin - 20.0) < 1e-10

    def test_calculate_margin_zero_leverage(self):
        margin = calculate_margin(1, 100, 0)
        assert margin == 0.0


class TestPnLCalculations:
    """盈亏计算相关测试"""

    def test_calculate_pnl_long_profit(self):
        pnl = calculate_pnl(100, 110, 1, "long")
        assert abs(pnl - 10) < 1e-10

    def test_calculate_pnl_long_loss(self):
        pnl = calculate_pnl(100, 90, 1, "long")
        assert abs(pnl - (-10)) < 1e-10

    def test_calculate_pnl_short_profit(self):
        pnl = calculate_pnl(100, 90, 1, "short")
        assert abs(pnl - 10) < 1e-10

    def test_calculate_pnl_short_loss(self):
        pnl = calculate_pnl(100, 110, 1, "short")
        assert abs(pnl - (-10)) < 1e-10

    def test_calculate_pnl_invalid_direction(self):
        with pytest.raises(ValueError):
            calculate_pnl(100, 110, 1, "invalid")

    def test_calculate_pnl_percent_long(self):
        pct = calculate_pnl_percent(100, 110, "long")
        assert abs(pct - 0.10) < 1e-10

    def test_calculate_pnl_percent_short(self):
        pct = calculate_pnl_percent(100, 90, "short")
        assert abs(pct - 0.10) < 1e-10

    def test_calculate_pnl_percent_zero_entry(self):
        pct = calculate_pnl_percent(0, 100, "long")
        assert pct == 0.0


class TestRiskReward:
    """风险收益比测试"""

    def test_risk_reward_ratio_long(self):
        ratio = calculate_risk_reward_ratio(100, 120, 90, "long")
        assert abs(ratio - 2.0) < 1e-10

    def test_risk_reward_ratio_short(self):
        ratio = calculate_risk_reward_ratio(100, 80, 110, "short")
        assert abs(ratio - 2.0) < 1e-10

    def test_risk_reward_ratio_zero_risk(self):
        ratio = calculate_risk_reward_ratio(100, 120, 100, "long")
        assert ratio == float("inf")

    def test_risk_reward_ratio_invalid_direction(self):
        with pytest.raises(ValueError):
            calculate_risk_reward_ratio(100, 120, 90, "invalid")


class TestTakeProfitStopLoss:
    """止盈止损测试"""

    def test_take_profit_long(self):
        tp = calculate_take_profit(100, "long", 0.10, 0.0, 4)
        assert abs(tp - 110.0) < 1e-4

    def test_take_profit_short(self):
        tp = calculate_take_profit(100, "short", 0.10, 0.0, 4)
        assert abs(tp - 90.0) < 1e-4

    def test_stop_loss_long(self):
        sl = calculate_stop_loss(100, "long", 0.10, 0.0, 4)
        assert abs(sl - 90.0) < 1e-4

    def test_stop_loss_short(self):
        sl = calculate_stop_loss(100, "short", 0.10, 0.0, 4)
        assert abs(sl - 110.0) < 1e-4

    def test_tp_sl_invalid_direction(self):
        with pytest.raises(ValueError):
            calculate_take_profit(100, "invalid", 0.10)

    def test_tp_sl_zero_entry(self):
        tp = calculate_take_profit(0, "long", 0.10)
        assert tp == 0.0

    def test_tp_sl_from_atr(self):
        tp, sl = calculate_tp_sl_from_atr(100, "long", 2.0, 3.0, 1.0, 0.0, 4)
        assert tp > 100
        assert sl < 100

    def test_validate_tp_sl_valid_long(self):
        result = validate_tp_sl_prices(100, "long", 120, 90)
        assert result["valid"] is True
        assert len(result["errors"]) == 0

    def test_validate_tp_sl_invalid_tp_below_entry_long(self):
        result = validate_tp_sl_prices(100, "long", 90, 80)
        assert result["valid"] is False
        assert any("TP" in e and "above" in e for e in result["errors"])

    def test_validate_tp_sl_invalid_sl_above_entry_long(self):
        result = validate_tp_sl_prices(100, "long", 120, 110)
        assert result["valid"] is False
        assert any("SL" in e and "below" in e for e in result["errors"])

    def test_validate_tp_sl_valid_short(self):
        result = validate_tp_sl_prices(100, "short", 80, 120)
        assert result["valid"] is True


class TestTrailingStop:
    """移动止损测试"""

    def test_trailing_stop_long_no_profit(self):
        sl = calculate_trailing_stop(100, 95, "long", 0.02, 0.005)
        assert abs(sl - 98.0) < 1e-10

    def test_trailing_stop_long_with_profit(self):
        sl = calculate_trailing_stop(100, 120, "long", 0.02, 0.005)
        assert sl > 100
        assert sl < 120

    def test_trailing_stop_short_no_profit(self):
        sl = calculate_trailing_stop(100, 105, "short", 0.02, 0.005)
        assert abs(sl - 102.0) < 1e-10

    def test_trailing_stop_invalid_direction(self):
        with pytest.raises(ValueError):
            calculate_trailing_stop(100, 105, "invalid")


class TestVolatilityAdjustment:
    """波动率调整测试"""

    def test_adjust_sl_volatility_zero_atr(self):
        sl = adjust_stop_loss_for_volatility(100, "long", 0, 0.02, 1.0)
        base_sl = calculate_stop_loss(100, "long", 0.02)
        assert abs(sl - base_sl) < 1e-4

    def test_adjust_sl_volatility_high_vol(self):
        sl_low = adjust_stop_loss_for_volatility(100, "long", 0.5, 0.02, 1.0)
        sl_high = adjust_stop_loss_for_volatility(100, "long", 2.0, 0.02, 1.0)
        assert sl_high < sl_low


class TestStatistics:
    """统计指标测试"""

    def test_sharpe_ratio_empty(self):
        assert calculate_sharpe_ratio([]) == 0.0

    def test_sharpe_ratio_single(self):
        assert calculate_sharpe_ratio([0.01]) == 0.0

    def test_sharpe_ratio_constant(self):
        assert calculate_sharpe_ratio([0.01, 0.01, 0.01]) == 0.0

    def test_sharpe_ratio_positive(self):
        returns = [0.01, 0.02, 0.01, 0.03, 0.02]
        sr = calculate_sharpe_ratio(returns)
        assert sr > 0

    def test_max_drawdown_empty(self):
        assert calculate_max_drawdown([]) == 0.0

    def test_max_drawdown_no_drawdown(self):
        assert calculate_max_drawdown([100, 110, 120, 130]) == 0.0

    def test_max_drawdown_simple(self):
        dd = calculate_max_drawdown([100, 90, 95, 80, 100])
        assert abs(dd - 0.20) < 1e-10

    def test_max_drawdown_zero_peak(self):
        dd = calculate_max_drawdown([0, -10, -5])
        assert dd == 0.0


class TestUtilityFunctions:
    """工具函数测试"""

    def test_clamp_within_range(self):
        assert clamp(5, 0, 10) == 5

    def test_clamp_below_min(self):
        assert clamp(-5, 0, 10) == 0

    def test_clamp_above_max(self):
        assert clamp(15, 0, 10) == 10

    def test_check_profit_protection_long_safe(self):
        assert check_profit_protection(100, 140, "long", 0.5) is True

    def test_check_profit_protection_long_breach(self):
        assert check_profit_protection(100, 160, "long", 0.5) is False

    def test_check_profit_protection_short_safe(self):
        assert check_profit_protection(100, 60, "short", 0.5) is True

    def test_check_profit_protection_short_breach(self):
        assert check_profit_protection(100, 40, "short", 0.5) is False

    def test_check_profit_protection_invalid_direction(self):
        with pytest.raises(ValueError):
            check_profit_protection(100, 100, "invalid")


class TestSignalQuality:
    """信号质量评估测试"""

    def test_evaluate_signal_quality_long_good(self):
        indicators = {"rsi": 45, "volume_delta": 0.5, "vwap_distance": 0.01, "momentum": 0.005}
        market_state = {"state": "uptrend", "volatility": "normal", "volume_ratio": 1.0}
        score = evaluate_signal_quality(indicators, market_state, "long")
        assert 0 <= score <= 1
        assert score > 0.5

    def test_evaluate_signal_quality_short_good(self):
        indicators = {"rsi": 55, "volume_delta": -0.5, "vwap_distance": -0.01, "momentum": -0.005}
        market_state = {"state": "downtrend", "volatility": "normal", "volume_ratio": 1.0}
        score = evaluate_signal_quality(indicators, market_state, "short")
        assert 0 <= score <= 1
        assert score > 0.5

    def test_evaluate_signal_quality_invalid_direction(self):
        with pytest.raises(ValueError):
            evaluate_signal_quality({}, {}, "invalid")

    def test_evaluate_signal_quality_empty_indicators(self):
        market_state = {"state": "range", "volatility": "normal", "volume_ratio": 1.0}
        score = evaluate_signal_quality({}, market_state, "long")
        assert 0 <= score <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
