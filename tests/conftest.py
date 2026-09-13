import pytest
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_okx_client():
    client = AsyncMock()
    client.get_account_info = AsyncMock(return_value={
        "code": "0",
        "data": [{
            "totalEq": "1000",
            "availableBalance": "800",
            "usedMargin": "200"
        }]
    })
    client.get_positions = AsyncMock(return_value={
        "code": "0",
        "data": []
    })
    client.get_ticker = AsyncMock(return_value={
        "code": "0",
        "data": [{
            "instId": "BTC-USDT-SWAP",
            "last": "60000",
            "ask": "60001",
            "bid": "59999",
            "vol24h": "1000"
        }]
    })
    client.place_order = AsyncMock(return_value={
        "code": "0",
        "data": [{
            "ordId": "test_order_id",
            "sCode": "0"
        }]
    })
    client.cancel_order = AsyncMock(return_value={
        "code": "0",
        "data": [{
            "ordId": "test_order_id"
        }]
    })
    client.get_order = AsyncMock(return_value={
        "code": "0",
        "data": [{
            "ordId": "test_order_id",
            "state": "filled",
            "avgPx": "60000",
            "fillSz": "0.01"
        }]
    })
    client.get_bills = AsyncMock(return_value={
        "code": "0",
        "data": []
    })
    return client

@pytest.fixture
def mock_redis_cache():
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock(return_value=True)
    cache.delete = MagicMock(return_value=True)
    cache.keys = MagicMock(return_value=[])
    return cache

@pytest.fixture
def mock_sqlite_storage():
    storage = MagicMock()
    storage.get_trade_by_id = MagicMock(return_value=None)
    storage.save_trade = MagicMock(return_value=True)
    storage.get_open_positions = MagicMock(return_value=[])
    storage.get_account_history = MagicMock(return_value=[])
    storage.update_position = MagicMock(return_value=True)
    return storage

@pytest.fixture
def mock_trade_journal():
    journal = MagicMock()
    journal.record_trade = MagicMock(return_value=True)
    journal.update_trade_record = MagicMock(return_value=True)
    journal.get_trade_history = MagicMock(return_value=[])
    journal.get_open_trades = MagicMock(return_value=[])
    return journal

@pytest.fixture
def mock_account_manager():
    manager = MagicMock()
    manager.get_total_equity = MagicMock(return_value=1000.0)
    manager.get_available_balance = MagicMock(return_value=800.0)
    manager.get_used_margin = MagicMock(return_value=200.0)
    manager.allocate_margin = MagicMock(return_value=100.0)
    return manager

@pytest.fixture
def config():
    """Load full config from config.yaml for integration tests"""
    from configs.settings import load_config
    return load_config()

@pytest.fixture
def basic_config():
    return {
        "system": {
            "log_level": "INFO",
            "trading_enabled": True
        },
        "trading": {
            "risk_per_trade": 0.02,
            "max_concurrent_positions": 6,
            "default_leverage": 6,
            "min_margin": 0.5,
            "min_notional": 15.0,
            "min_profit_cost_ratio": 2.0,
            "min_hold_seconds": 300,
            "grid_allocation": 0.10,
            "spot_grid_allocation": 0.15,
            "spot_martingale_allocation": 0.10,
            "trend_allocation": 0.30,
            "scalping_allocation": 0.20,
            "arbitrage_allocation": 0.15,
            "max_total_leverage": 20,
            "max_drawdown": 0.25,
            "daily_max_loss": 0.10,
            "hourly_max_loss": 0.05,
            "max_consecutive_losses": 5,
            "max_daily_trades_per_symbol": 50,
            "total_capital": 1000,
            "trading_capital_ratio": 0.8
        },
        "strategies": {
            "grid": {
                "enabled": True,
                "min_signal_quality": 0.35,
                "martingale_layers": 5
            },
            "trend": {
                "enabled": True,
                "min_signal_quality": 0.35,
                "max_additions": 4
            },
            "scalping": {
                "enabled": True,
                "min_signal_quality": 0.20,
                "run_hours_start": 8,
                "run_hours_end": 24
            },
            "arbitrage": {
                "enabled": True
            },
            "spot_grid": {
                "enabled": True,
                "min_signal_quality": 0.25
            },
            "spot_martingale": {
                "enabled": True,
                "min_signal_quality": 0.25
            }
        },
        "risk": {
            "circuit_breaker": {
                "enabled": True,
                "max_daily_loss": 0.10,
                "max_hourly_loss": 0.05
            },
            "margin_call_threshold": 0.5,
            "margin_warning_threshold": 0.8,
            "position_loss_threshold": 0.5,
            "full_close_threshold": 0.8,
            "circuit_breakers": {
                "btc_movement_threshold": 0.05,
                "btc_movement_window": 3600,
                "liquidation_volume_threshold": 50,
                "liquidation_window": 300,
                "rapid_drawdown_threshold": 0.05,
                "rapid_drawdown_window": 3600,
                "drawdown_acceleration_threshold": 0.02,
                "volatility_surge_threshold": 0.03,
                "volatility_surge_window": 300,
                "order_rejection_rate": 0.3,
                "api_timeout": 30,
                "network_timeout": 60
            }
        },
        "currencies": {
            "tier1_symbols": ["BTC", "ETH"],
            "tier2_symbols": ["BNB", "SOL", "XRP", "ADA", "DOGE"],
            "tier3_symbols": ["DOT", "AVAX", "MATIC", "LINK", "ATOM"],
            "tier1_settings": {
                "slippage": 0.001,
                "position_limit": 0.30,
                "leverage_max": 20,
                "leverage_min": 1
            },
            "tier2_settings": {
                "slippage": 0.002,
                "position_limit": 0.40,
                "leverage_max": 10,
                "leverage_min": 1
            },
            "tier3_settings": {
                "slippage": 0.005,
                "position_limit": 0.50,
                "leverage_max": 5,
                "leverage_min": 1
            }
        },
        "sqlite": {
            "db_path": ":memory:"
        },
        "hardware": {
            "high_priority_cores": [0, 1]
        }
    }