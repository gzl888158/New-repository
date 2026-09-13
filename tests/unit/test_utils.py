import pytest
import numpy as np
from utils.helpers import (
    calculate_position_size,
    calculate_position_margin,
    calculate_pnl,
    calculate_pnl_percent,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    round_to_tick_size,
    format_price,
    generate_order_id,
    clamp,
    calculate_risk_reward_ratio,
    calculate_take_profit,
    calculate_stop_loss,
    calculate_trailing_stop,
    detect_market_state,
    calculate_liquidation_price,
    calculate_funding_fee,
    estimate_net_profit,
    calculate_round_trip_cost,
    calculate_scaled_take_profit,
    adjust_stop_loss_for_volatility,
    check_profit_protection,
    calculate_partial_close_quantity,
    calculate_adaptive_position_size,
    adjust_signal_thresholds,
    evaluate_signal_quality,
    calculate_tp_sl_from_atr,
    validate_tp_sl_prices
)


class TestPositionCalculations:
    def test_calculate_position_size(self):
        size = calculate_position_size(1000, 60000, 6)
        assert size > 0
        assert size < 1

    def test_calculate_position_size_invalid_price(self):
        size = calculate_position_size(1000, 0, 6)
        assert size == 0.0

    def test_calculate_position_size_invalid_leverage(self):
        size = calculate_position_size(1000, 60000, 0)
        assert size == 0.0

    def test_calculate_position_margin(self):
        margin = calculate_position_margin(0.1, 60000, 6)
        assert margin > 0

    def test_calculate_pnl_long(self):
        pnl = calculate_pnl(60000, 62000, 0.1, "long")
        assert pnl > 0

    def test_calculate_pnl_short(self):
        pnl = calculate_pnl(60000, 58000, 0.1, "short")
        assert pnl > 0

    def test_calculate_pnl_loss(self):
        pnl = calculate_pnl(60000, 59000, 0.1, "long")
        assert pnl < 0

    def test_calculate_pnl_percent(self):
        pct = calculate_pnl_percent(60000, 66000, "long")
        assert pct == 0.1


class TestRiskMetrics:
    def test_sharpe_ratio_positive(self):
        returns = [0.01, 0.02, 0.015, 0.005, 0.03]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe > 0

    def test_sharpe_ratio_zero_volatility(self):
        returns = [0.01, 0.01, 0.01, 0.01, 0.01]
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0

    def test_sharpe_ratio_empty(self):
        returns = []
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0

    def test_max_drawdown(self):
        equity = [100, 110, 95, 105, 90, 120]
        dd = calculate_max_drawdown(equity)
        assert dd > 0
        assert dd <= 1

    def test_max_drawdown_empty(self):
        equity = []
        dd = calculate_max_drawdown(equity)
        assert dd == 0

    def test_risk_reward_ratio(self):
        ratio = calculate_risk_reward_ratio(60000, 62000, 58000, "long")
        assert ratio > 0


class TestFormatting:
    def test_format_price(self):
        assert format_price(60000.1234, "BTC-USDT-SWAP") == "60000.12"
        assert format_price(1.2345, "ETH-USDT-SWAP") == "1.23"
        assert format_price(0.0001234, "PEPE-USDT-SWAP") == "0.000123"

    def test_generate_order_id(self):
        order_id = generate_order_id()
        assert len(order_id) > 0
        assert order_id.startswith("ORD")

    def test_round_to_tick_size(self):
        assert round_to_tick_size(0.12345, 0.01) == 0.12

    def test_clamp(self):
        assert clamp(5, 0, 10) == 5
        assert clamp(-5, 0, 10) == 0
        assert clamp(15, 0, 10) == 10


class TestTP_Sl:
    def test_calculate_take_profit_long(self):
        tp = calculate_take_profit(60000, "long", 0.02)
        assert tp > 60000

    def test_calculate_take_profit_short(self):
        tp = calculate_take_profit(60000, "short", 0.02)
        assert tp < 60000

    def test_calculate_stop_loss_long(self):
        sl = calculate_stop_loss(60000, "long", 0.02)
        assert sl < 60000

    def test_calculate_stop_loss_short(self):
        sl = calculate_stop_loss(60000, "short", 0.02)
        assert sl > 60000

    def test_calculate_trailing_stop_long(self):
        ts = calculate_trailing_stop(60000, 61000, "long")
        assert ts > 60000

    def test_validate_tp_sl_prices(self):
        result = validate_tp_sl_prices(60000, "long", 62000, 58000)
        assert result["valid"] is True

    def test_validate_tp_sl_invalid(self):
        result = validate_tp_sl_prices(60000, "long", 58000, 62000)
        assert result["valid"] is False

    def test_calculate_tp_sl_from_atr(self):
        tp, sl = calculate_tp_sl_from_atr(60000, "long", 1200)
        assert tp > 60000
        assert sl < 60000


class TestLiquidation:
    def test_calculate_liquidation_price_long(self):
        liq = calculate_liquidation_price(60000, "long", 10)
        assert liq > 0
        assert liq < 60000

    def test_calculate_liquidation_price_short(self):
        liq = calculate_liquidation_price(60000, "short", 10)
        assert liq > 60000


class TestCostCalculations:
    def test_calculate_round_trip_cost(self):
        costs = calculate_round_trip_cost(0.1, 60000)
        assert costs["total_cost"] > 0
        assert "notional_value" in costs

    def test_calculate_funding_fee_long_pays(self):
        fee = calculate_funding_fee(6000, 0.001, True)
        assert fee < 0

    def test_calculate_funding_fee_short_receives(self):
        fee = calculate_funding_fee(6000, 0.001, False)
        assert fee > 0

    def test_estimate_net_profit(self):
        result = estimate_net_profit(60000, 61000, 0.1, "long")
        assert "net_pnl" in result


class TestMarketState:
    def test_detect_market_state_insufficient_data(self):
        state = detect_market_state([60000, 60100])
        assert state["state"] == "unknown"

    def test_detect_market_state_bullish(self):
        prices = [60000, 60200, 60500, 60800, 61000, 61200, 61500, 61800, 62000, 62200,
                  62500, 62800, 63000, 63200, 63500, 63800, 64000, 64200, 64500, 64800]
        state = detect_market_state(prices)
        assert state["state"] == "uptrend"

    def test_detect_market_state_bearish(self):
        prices = [65000, 64800, 64500, 64200, 64000, 63800, 63500, 63200, 63000, 62800,
                  62500, 62200, 62000, 61800, 61500, 61200, 61000, 60800, 60500, 60200]
        state = detect_market_state(prices)
        assert state["state"] == "downtrend"


class TestAdaptivePositioning:
    def test_calculate_adaptive_position_size(self):
        market_state = {
            "volatility": "normal",
            "state": "range",
            "volume_ratio": 1.0
        }
        size = calculate_adaptive_position_size(0.1, market_state, "scalping")
        assert size > 0

    def test_adjust_signal_thresholds(self):
        base = {"profit_target_min": 0.015, "stop_loss": 0.025}
        market_state = {"volatility": "normal", "state": "range", "volume_ratio": 1.0}
        adjusted = adjust_signal_thresholds(base, market_state)
        assert adjusted != base


class TestSignalQuality:
    def test_evaluate_signal_quality(self):
        indicators = {"rsi": 35, "momentum": 0.008}
        market_state = {"state": "uptrend", "volatility": "normal", "volume_ratio": 1.2}
        quality = evaluate_signal_quality(indicators, market_state, "long")
        assert 0 <= quality <= 1


class TestPartialClose:
    def test_calculate_partial_close_quantity(self):
        targets = [{"threshold": 0.01, "close_ratio": 0.5}, {"threshold": 0.02, "close_ratio": 0.5}]
        qty = calculate_partial_close_quantity(0.1, targets, 0.005)
        assert qty > 0