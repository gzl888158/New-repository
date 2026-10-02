"""OKXClient.get_kline_async 的 403/429 缓存兜底 — 单元测试。

背景：代理节点被 OKX 屏蔽时，history-candles 返回 403（code="403"），
此前的 get_kline_async 只有 429 才走缓存，403 直接返回空导致指标计算失败。

修复后：403（代理屏蔽）与 429（限流）均跳过同步回退、优先使用 K 线缓存；
403 使用更长的缓存窗口（10 分钟），因代理恢复较慢。
"""

import asyncio
import time

from core.okx_client import OKXClient


def _make_kline_client():
    c = object.__new__(OKXClient)
    c._network_healthy = True
    c._kline_cache = {}
    c._async_make_request = None  # 测试中覆盖
    c._make_request = None  # 403/429 路径不应触发同步回退
    return c


def test_kline_async_403_uses_cache_within_ttl():
    """403 且缓存有效（<10 分钟）时应返回缓存数据。"""
    c = _make_kline_client()
    symbol, bar, limit = "BTC-USDT-SWAP", "1H", 100
    key = f"{symbol}:{bar}:{limit}"
    cached_data = [{"last": "60000", "ts": "1"}]
    c._kline_cache[key] = {"data": cached_data, "ts": time.time()}

    async def _fake_async(method, path):
        return {"code": "403", "data": {}, "msg": "HTTP 403"}

    c._async_make_request = _fake_async

    result = asyncio.run(c.get_kline_async(symbol, bar, limit=limit))
    assert result == cached_data


def test_kline_async_403_no_cache_returns_empty():
    """403 且无缓存时应返回空列表而非抛异常。"""
    c = _make_kline_client()

    async def _fake_async(method, path):
        return {"code": "403", "data": {}, "msg": "HTTP 403"}

    c._async_make_request = _fake_async

    result = asyncio.run(c.get_kline_async("BTC-USDT-SWAP", "1H", limit=100))
    assert result == []


def test_kline_async_429_uses_cache_within_ttl():
    """429 且缓存有效（<5 分钟）时应返回缓存数据（保持原行为）。"""
    c = _make_kline_client()
    symbol, bar, limit = "ETH-USDT-SWAP", "4H", 60
    key = f"{symbol}:{bar}:{limit}"
    cached_data = [{"last": "3000", "ts": "2"}]
    c._kline_cache[key] = {"data": cached_data, "ts": time.time()}

    async def _fake_async(method, path):
        return {"code": "429", "data": {}, "msg": "HTTP 429"}

    c._async_make_request = _fake_async

    result = asyncio.run(c.get_kline_async(symbol, bar, limit=limit))
    assert result == cached_data


def test_kline_async_403_expired_cache_returns_stale_data():
    """403 且缓存已过期（>10 分钟）时，宽松策略仍返回过期缓存（长周期K线仍可用）。"""
    c = _make_kline_client()
    symbol, bar, limit = "SOL-USDT-SWAP", "1m", 120
    key = f"{symbol}:{bar}:{limit}"
    stale_data = [{"last": "97"}]
    c._kline_cache[key] = {"data": stale_data, "ts": time.time() - 700}  # 11.6 分钟前

    async def _fake_async(method, path):
        return {"code": "403", "data": {}, "msg": "HTTP 403"}

    c._async_make_request = _fake_async

    result = asyncio.run(c.get_kline_async(symbol, bar, limit=limit))
    assert result == stale_data
