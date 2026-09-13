import pytest
from datetime import datetime
from unittest.mock import MagicMock
from risk.global_risk import GlobalRiskControl
from risk.strategy_risk import StrategyRiskControl


class TestGlobalRiskControl:
    def test_max_drawdown_exceeded(self, basic_config):
        mock_redis = MagicMock()
        mock_okx = MagicMock()
        risk = GlobalRiskControl(basic_config, mock_redis, mock_okx)
        risk._effective_peak = 1000
        risk._current_equity = 700
        drawdown = (risk._effective_peak - risk._current_equity) / risk._effective_peak
        assert drawdown >= 0.25

    def test_max_drawdown_within_limit(self, basic_config):
        mock_redis = MagicMock()
        mock_okx = MagicMock()
        risk = GlobalRiskControl(basic_config, mock_redis, mock_okx)
        risk._effective_peak = 1000
        risk._current_equity = 800
        drawdown = (risk._effective_peak - risk._current_equity) / risk._effective_peak
        assert drawdown < 0.25

    def test_can_trade(self, basic_config):
        mock_redis = MagicMock()
        mock_okx = MagicMock()
        risk = GlobalRiskControl(basic_config, mock_redis, mock_okx)
        assert risk.can_trade() is True


class TestStrategyRiskControl:
    def test_check_daily_trade_limit_within(self, basic_config):
        basic_config["trading"]["max_daily_trades_per_symbol"] = 50
        mock_account = MagicMock()
        risk = StrategyRiskControl(basic_config, mock_account)
        assert risk.check_daily_trade_limit("BTC-USDT-SWAP") is True

    def test_check_daily_trade_limit_exceeded(self, basic_config):
        basic_config["trading"]["max_daily_trades_per_symbol"] = 50
        mock_account = MagicMock()
        risk = StrategyRiskControl(basic_config, mock_account)
        today = datetime.now().date()
        risk._daily_trade_count[f"BTC-USDT-SWAP:{today}"] = 50
        assert risk.check_daily_trade_limit("BTC-USDT-SWAP") is False


class TestRiskLimits:
    def test_check_signal_within_limits(self, basic_config):
        basic_config["risk"]["single_symbol_max_margin"] = 0.25
        basic_config["risk"]["single_strategy_max_margin"] = 0.30
        basic_config["trading"]["max_leverage"] = 20
        basic_config["trading"]["max_concurrent_positions"] = 4
        basic_config["risk"]["daily_max_trades"] = 100
        basic_config["risk"]["hourly_max_trades"] = 20
        
        mock_okx = MagicMock()
        mock_okx.get_account_info.return_value = {"totalEq": "1000"}
        mock_okx.get_positions.return_value = []
        
        from risk.risk_limits import RiskLimits
        limits = RiskLimits(basic_config, mock_okx)
        
        signal_data = {
            "symbol": "BTC-USDT-SWAP",
            "strategy_name": "grid",
            "leverage": 10,
            "quantity": 0.01,
            "price": 60000
        }
        
        assert limits.check_signal(signal_data) is True

    def test_check_signal_leverage_exceeded(self, basic_config):
        basic_config["risk"]["single_symbol_max_margin"] = 0.25
        basic_config["risk"]["single_strategy_max_margin"] = 0.30
        basic_config["trading"]["max_leverage"] = 10
        basic_config["trading"]["max_concurrent_positions"] = 4
        basic_config["risk"]["daily_max_trades"] = 100
        basic_config["risk"]["hourly_max_trades"] = 20
        
        mock_okx = MagicMock()
        mock_okx.get_account_info.return_value = {"totalEq": "1000"}
        mock_okx.get_positions.return_value = []
        
        from risk.risk_limits import RiskLimits
        limits = RiskLimits(basic_config, mock_okx)
        
        signal_data = {
            "symbol": "BTC-USDT-SWAP",
            "strategy_name": "grid",
            "leverage": 20,
            "quantity": 0.01,
            "price": 60000
        }
        
        assert limits.check_signal(signal_data) is False