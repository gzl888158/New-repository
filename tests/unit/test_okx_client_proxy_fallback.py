"""OKXClient 异步请求代理故障直连降级 — 单元测试。

背景：异步 REST 请求（行情 K线/ticker/orderbook 的主数据源）此前通过
aiohttp 的 trust_env=True 环境变量走代理，代理故障（ClientError/TimeoutError）
时只做退避重试、永不直连，也不累计 _proxy_fail_count 触发代理禁用，
导致代理不稳定时行情数据全部失败（日志 ServerTimeoutError）。

修复后：异步请求对齐同步 _request 的降级逻辑 ——
  1. 请求发送前按 _proxy_disabled_until 动态选择代理/直连 session；
  2. 代理故障时累计 _proxy_fail_count 并递进退避禁用代理；
  3. 代理故障时立即用 trust_env=False 的直连 session 重试一次；
  4. 直连成功重置 _proxy_fail_count。
"""

import asyncio
from unittest.mock import MagicMock

import aiohttp
import pytest

from core.okx_client import OKXClient


class _FakeResp:
    """模拟 aiohttp 响应（async context manager）。"""

    def __init__(self, status, data):
        self.status = status
        self._data = data

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def json(self):
        return self._data

    async def text(self):
        return ""


class _FailingProxySession:
    """代理 session：get/post 返回一个 __aenter__ 抛 ClientError 的上下文（模拟代理超时/挂死）。

    注意：aiohttp 的 session.get() 是普通方法（非 async），返回 async context manager；
    async with 会 await 其 __aenter__() 执行实际请求。故此处 __aenter__ 抛异常。
    """
    closed = False

    def get(self, *a, **k):
        return _FailingCtx()

    def post(self, *a, **k):
        return _FailingCtx()


class _FailingCtx:
    async def __aenter__(self):
        raise aiohttp.ClientError("proxy timeout")

    async def __aexit__(self, *exc):
        return False


class _DirectSession:
    """直连 session：返回成功响应。"""
    closed = False

    def __init__(self, resp):
        self._resp = resp

    def get(self, *a, **k):
        return self._resp

    def post(self, *a, **k):
        return self._resp


def _make_client(proxy="http://127.0.0.1:7897"):
    c = object.__new__(OKXClient)
    c.proxy = proxy
    c.rest_url = "https://www.okx.com"
    c._connect_timeout = 5

    # 请求限流 / 熔断 / 网络健康等最小依赖
    c._rest_request_times = []
    c._rest_max_requests_per_window = 10
    c._rest_rate_limit_window = 1.0
    c._api_pool = MagicMock()
    c._api_pool.count = 3
    c._network_healthy = True
    c._CRITICAL_API_PREFIXES = ()
    c._ws_public_connected = True
    c._ws_private_connected = True

    # 代理健康管理状态
    c._proxy_fail_count = 0
    c._proxy_disabled_until = 0.0
    c._proxy_disable_duration = 60.0
    c._proxy_disable_cycle = 0

    # 重试风暴 / 限流
    c._retry_storm_count = 0
    c._last_retry_storm_warning = 0.0
    c._retry_storm_threshold = 100
    c._async_request_semaphore = MagicMock()

    # 回调与工具方法
    c._on_latency = None
    c._on_api_call = None
    c._on_network_success = MagicMock()
    c._on_network_failure = MagicMock()
    c._get_headers = MagicMock(return_value={"OK-ACCESS-KEY": "x"})
    c._get_adaptive_timeout = MagicMock(return_value=20)
    c._is_latency_circuit_open = MagicMock(return_value=False)
    c._is_rate_limited = MagicMock(return_value=False)
    c._track_latency = MagicMock()
    c._handle_rate_limit = MagicMock()
    c._handle_429_rate_limit = MagicMock()
    c._async_rate_limit_backoff = MagicMock()
    c._rotate_key = MagicMock()

    # 代理 session（失败）与直连 session（成功）
    c._async_session = _FailingProxySession()
    c._async_session_direct = _DirectSession(_FakeResp(200, {"code": "0", "data": [{"last": "60000"}]}))
    return c


def test_async_proxy_failure_falls_back_to_direct():
    """代理故障时，异步请求应立即直连重试并成功返回。"""
    c = _make_client()
    result = asyncio.run(
        c._async_make_request_inner("GET", "/api/v5/market/ticker?instId=BTC-USDT-SWAP")
    )

    assert result is not None
    assert result.get("code") == "0"
    assert result["data"][0]["last"] == "60000"
    # 直连成功 → 代理失败计数重置
    assert c._proxy_fail_count == 0
    # 网络成功回调被触发
    c._on_network_success.assert_called()


def test_async_proxy_failure_increments_fail_count():
    """代理与直连均失败时，代理失败计数应累计。"""
    c = _make_client()
    # 直连也失败
    c._async_session_direct = _DirectSession(_FakeResp(500, {"code": "500", "data": {}}))

    asyncio.run(c._async_make_request_inner("GET", "/api/v5/market/ticker?instId=BTC-USDT-SWAP"))

    # 代理失败计数 >= 1（直连 500 不算连接成功，不重置计数）
    assert c._proxy_fail_count >= 1
