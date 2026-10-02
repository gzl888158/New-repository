import asyncio
import time

from core.okx_client import OKXClient


def _client():
    client = object.__new__(OKXClient)
    client._ticker_cache = {}
    client._ticker_cache_ttl_normal = 2.0
    client._ticker_cache_ttl_degraded = 30.0
    client._ticker_max_server_age_seconds = 5.0
    client._network_healthy = True
    return client


def _ticker(age_seconds=0):
    return {
        "ts": str(int((time.time() - age_seconds) * 1000)),
        "last": "100",
    }


def test_sync_ticker_skips_stale_server_timestamp_cache_and_fallback():
    client = _client()
    stale = _ticker(age_seconds=20)
    client._ticker_cache["BTC-USDT-SWAP"] = {"data": stale, "ts": time.time()}
    client._make_request = lambda *args: None

    assert client.get_ticker("BTC-USDT-SWAP") is None


def test_sync_ticker_rejects_stale_rest_response():
    client = _client()
    stale = _ticker(age_seconds=20)
    client._make_request = lambda *args: {"code": "0", "data": [stale]}

    assert client.get_ticker("BTC-USDT-SWAP") is None


def test_async_ticker_skips_stale_server_timestamp_cache():
    client = _client()
    stale = _ticker(age_seconds=20)
    client._ticker_cache["BTC-USDT-SWAP"] = {"data": stale, "ts": time.time()}
    calls = []

    async def request(*args):
        calls.append(args)
        return None

    client._async_make_request = request

    assert asyncio.run(client.get_ticker_async("BTC-USDT-SWAP")) is None
    assert calls


def test_fresh_server_timestamp_cache_is_reused():
    client = _client()
    fresh = _ticker()
    client._ticker_cache["BTC-USDT-SWAP"] = {"data": fresh, "ts": time.time()}
    client._make_request = lambda *args: (_ for _ in ()).throw(
        AssertionError("fresh cache should avoid REST")
    )

    assert client.get_ticker("BTC-USDT-SWAP") is fresh


def _orderbook_client(orderbook, min_depth=2):
    client = object.__new__(OKXClient)
    client._orderbook_min_depth = min_depth
    client.requested_paths = []
    client._make_request = lambda *args: (
        client.requested_paths.append(args[1]) or {"code": "0", "data": [orderbook]}
    )
    return client


def test_orderbook_rejects_empty_side_and_shallow_depth():
    empty_side = _orderbook_client({"bids": [["100", "1"]], "asks": []})
    shallow = _orderbook_client({"bids": [["100", "1"]], "asks": [["101", "1"]]})

    assert empty_side.get_order_book("BTC-USDT-SWAP") is None
    assert shallow.get_order_book("BTC-USDT-SWAP") is None


def test_orderbook_rejects_non_finite_level_values():
    client = _orderbook_client({
        "bids": [["NaN", "1"], ["99", "2"]],
        "asks": [["101", "1"], ["102", "2"]],
    })

    assert client.get_order_book("BTC-USDT-SWAP") is None


def test_orderbook_returns_valid_bilateral_depth():
    orderbook = {
        "bids": [["100", "1"], ["99", "2"]],
        "asks": [["101", "1"], ["102", "2"]],
    }
    client = _orderbook_client(orderbook)

    assert client.get_order_book("BTC-USDT-SWAP", depth=1) == orderbook
    assert "&sz=2" in client.requested_paths[0]


def _funding_client(monkeypatch, next_funding_seconds=3600):
    client = _client()
    client._funding_rate_cache = {}
    client._funding_rate_cache_ttl = 60.0
    client._funding_rate_settlement_window = 120.0
    client._funding_rate_settlement_ttl = 10.0
    client._funding_rate_stale_fallback_ttl = 300.0
    now = time.time()
    monkeypatch.setattr("core.okx_client.time.time", lambda: now)
    return client, now + next_funding_seconds


def test_funding_rate_is_cached_for_normal_ttl(monkeypatch):
    client, next_funding = _funding_client(monkeypatch)
    rate = {"fundingRate": "0.001", "nextFundingTime": str(int(next_funding * 1000))}
    calls = []
    client._make_request = lambda *args: (calls.append(args) or {"code": "0", "data": [rate]})

    assert client.get_funding_rate("BTC-USDT-SWAP") is rate
    assert client.get_funding_rate("BTC-USDT-SWAP") is rate
    assert len(calls) == 1


def test_funding_rate_uses_short_ttl_near_settlement(monkeypatch):
    client, next_funding = _funding_client(monkeypatch, next_funding_seconds=30)
    rate = {"fundingRate": "0.001", "nextFundingTime": str(int(next_funding * 1000))}
    client._funding_rate_cache["BTC-USDT-SWAP"] = {"data": rate, "ts": time.time() - 11}
    calls = []
    client._make_request = lambda *args: (calls.append(args) or {"code": "0", "data": [rate]})

    assert client.get_funding_rate("BTC-USDT-SWAP") is rate
    assert len(calls) == 1


def test_funding_rate_uses_bounded_cache_on_request_failure(monkeypatch):
    client, _ = _funding_client(monkeypatch)
    rate = {"fundingRate": "0.001", "nextFundingTime": "0"}
    client._funding_rate_cache["BTC-USDT-SWAP"] = {"data": rate, "ts": time.time() - 90}
    client._make_request = lambda *args: None

    assert client.get_funding_rate("BTC-USDT-SWAP") is rate


def test_async_funding_rate_uses_cache(monkeypatch):
    client, next_funding = _funding_client(monkeypatch)
    rate = {"fundingRate": "0.001", "nextFundingTime": str(int(next_funding * 1000))}
    calls = []

    async def request(*args):
        calls.append(args)
        return {"code": "0", "data": [rate]}

    client._async_make_request = request

    async def fetch_twice():
        first = await client.get_funding_rate_async("BTC-USDT-SWAP")
        second = await client.get_funding_rate_async("BTC-USDT-SWAP")
        return first, second

    first, second = asyncio.run(fetch_twice())

    assert first is rate
    assert second is rate
    assert len(calls) == 1