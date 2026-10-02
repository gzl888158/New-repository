from datetime import datetime
import math
import threading
import asyncio
from types import SimpleNamespace

from core.models import TickData
from market_data.manager import MarketDataManager
from services.market_data_service import DataQualityChecker, MarketDataService


def _checker():
    return DataQualityChecker({"market_data": {"max_kline_gap_intervals": 1.5}})


def test_tick_with_non_finite_numeric_field_is_rejected():
    tick = TickData(
        symbol="BTC-USDT-SWAP",
        price=100.0,
        volume=1.0,
        bid_price=99.0,
        bid_volume=math.nan,
        ask_price=101.0,
        ask_volume=1.0,
        timestamp=datetime.now(),
    )

    report = _checker().check_tick(tick)

    assert report["quality_score"] == 0.0
    assert "non_finite:bid_volume" in report["issues"]


def test_kline_series_reports_gap_and_non_finite_values():
    candles = [
        {"timestamp": 1_700_000_000, "open": 100, "high": 101, "low": 99, "close": 100, "vol": 1},
        {"timestamp": 1_700_000_120, "open": 100, "high": 101, "low": 99, "close": 100, "vol": math.inf},
    ]

    report = _checker().check_kline_series("BTC-USDT-SWAP", candles, "1m")

    assert report["valid"] is False
    assert report["missing_bars"] == 1
    assert 1 in report["invalid_rows"]
    assert any(issue.startswith("kline_gap:") for issue in report["issues"])
    assert "non_finite_ohlcv:1" in report["issues"]


def test_kline_series_reports_out_of_order_timestamps():
    candles = [
        [1_700_000_060_000, "100", "101", "99", "100", "1"],
        [1_700_000_000_000, "100", "101", "99", "100", "1"],
    ]

    report = _checker().check_kline_series("BTC-USDT-SWAP", candles, "1m")

    assert report["valid"] is False
    assert any(issue.startswith("non_monotonic_timestamp:") for issue in report["issues"])


def test_rest_fallback_fetches_symbols_concurrently():
    barrier = threading.Barrier(3, timeout=2)

    class Client:
        def get_ticker(self, symbol):
            barrier.wait()
            return {"symbol": symbol}

    service = MarketDataService({}, Client(), redis_cache=None)
    service._subscribed_symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]

    import asyncio
    results = asyncio.run(service._fetch_rest_fallback_tickers())

    assert {symbol for symbol, _ in results} == set(service._subscribed_symbols)


def test_market_data_manager_does_not_return_gapped_cached_klines():
    manager = MarketDataManager.__new__(MarketDataManager)
    manager._data_pool = SimpleNamespace(get_klines=lambda *args: [
        {"timestamp": 1_700_000_000_000, "symbol": "BTC-USDT-SWAP", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"timestamp": 1_700_000_120_000, "symbol": "BTC-USDT-SWAP", "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
    ])
    manager._series_quality_checker = _checker()
    manager._okx_client = None

    result = asyncio.run(manager.get_klines("BTC-USDT-SWAP", "1m", limit=2))

    assert result is None