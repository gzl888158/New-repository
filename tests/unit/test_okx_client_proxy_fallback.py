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

    # 多代理源自动切换状态
    c._proxy_list = [proxy]
    c._proxy_states = {proxy: {"fail_count": 0, "disabled_until": 0.0, "disable_cycle": 0}}
    c._current_proxy_idx = 0
    c._proxy_fail_threshold = 3
    c._proxy_disable_duration = 60.0
    c._proxy_max_disable = 600.0
    c.allow_direct_fallback = True  # 默认沿用旧降级行为（代理失败回退直连），新行为见下方专项测试

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
    proxy = "http://127.0.0.1:7897"
    c = _make_client(proxy)
    result = asyncio.run(
        c._async_make_request_inner("GET", "/api/v5/market/ticker?instId=BTC-USDT-SWAP")
    )

    assert result is not None
    assert result.get("code") == "0"
    assert result["data"][0]["last"] == "60000"
    # 直连成功 → 代理状态不变（直连成功不代表代理恢复，不重置代理计数）
    # 网络成功回调被触发
    c._on_network_success.assert_called()


def test_async_proxy_failure_increments_fail_count():
    """代理与直连均失败时，代理失败计数应累计（按代理独立计数）。"""
    proxy = "http://127.0.0.1:7897"
    c = _make_client(proxy)
    # 直连也失败
    c._async_session_direct = _DirectSession(_FakeResp(500, {"code": "500", "data": {}}))

    asyncio.run(c._async_make_request_inner("GET", "/api/v5/market/ticker?instId=BTC-USDT-SWAP"))

    # 代理失败计数 >= 1（直连 500 不算连接成功，不重置计数）
    assert c._proxy_states[proxy]["fail_count"] >= 1


def test_no_direct_fallback_retries_proxy_only():
    """allow_direct_fallback=False 时，代理失败不回退直连，仅冷却单个代理。

    境内直连被墙（WinError 64 / 超时）时，直连回退是必败死路。
    修复后代理失败绝不切直连；单代理连续失败达阈值后冷却该代理（不是全局禁用），
    冷却到期自动恢复，不会触发死亡螺旋。
    """
    proxy = "http://127.0.0.1:7897"
    c = _make_client(proxy)
    c.allow_direct_fallback = False

    result = asyncio.run(
        c._async_make_request_inner("GET", "/api/v5/market/ticker?instId=BTC-USDT-SWAP")
    )

    # 代理 session 一直失败且未回退直连 → 重试耗尽后返回 None
    assert result is None
    # 代理失败计数累计，达阈值后该代理被冷却（disabled_until > 0）
    assert c._proxy_states[proxy]["fail_count"] >= c._proxy_fail_threshold
    assert c._proxy_states[proxy]["disabled_until"] > 0.0
    # 关键：allow_direct_fallback=False 时，即使代理被冷却，_get_active_proxy 仍返回该代理（强制走代理）
    assert c._get_active_proxy() == proxy


def test_multi_proxy_auto_switch_on_failure():
    """多代理源：当前代理失败达阈值后自动切换到下一个健康代理。"""
    proxy_a = "http://127.0.0.1:7897"
    proxy_b = "http://127.0.0.1:7898"
    c = _make_client(proxy_a)
    c._proxy_list = [proxy_a, proxy_b]
    c._proxy_states = {
        proxy_a: {"fail_count": 0, "disabled_until": 0.0, "disable_cycle": 0},
        proxy_b: {"fail_count": 0, "disabled_until": 0.0, "disable_cycle": 0},
    }
    c._current_proxy_idx = 0
    c.allow_direct_fallback = False

    # 模拟 proxy_a 连续失败达阈值
    for _ in range(c._proxy_fail_threshold):
        c._mark_proxy_failed(proxy_a)

    # proxy_a 被冷却，应自动切换到 proxy_b
    assert c._proxy_states[proxy_a]["disabled_until"] > 0.0
    assert c._get_active_proxy() == proxy_b
    assert c.proxy == proxy_b

    # proxy_b 成功后重置其状态
    c._mark_proxy_success(proxy_b)
    assert c._proxy_states[proxy_b]["fail_count"] == 0
    assert c._proxy_states[proxy_b]["disabled_until"] == 0.0


def test_proxy_list_empty_means_direct():
    """proxy_list 为空时，_get_active_proxy 返回 None（直连）。"""
    c = _make_client()
    c._proxy_list = []
    c._proxy_states = {}
    assert c._get_active_proxy() is None


def _make_refused_and_timeout_exc():
    """构造「连接被拒」与「超时」两类异常（urllib3 若可用则按其真实包装形态构造）。"""
    try:
        from urllib3.exceptions import ProxyError, NewConnectionError
        timeout_exc = ProxyError("Cannot connect to proxy.", TimeoutError("_ssl.c:983: The handshake operation timed out"))
        refused_exc = NewConnectionError(None, "Failed to establish a new connection: [WinError 10061] 由于目标计算机积极拒绝，无法连接。")
        return timeout_exc, refused_exc
    except ImportError:
        return TimeoutError("timed out"), ConnectionRefusedError(10061, "connection refused")


def test_is_connection_refused_distinguishes_refused_from_timeout():
    """_is_connection_refused 应区分「连接被拒(无监听)」与「超时(半死)」。

    urllib3 的 ProxyError 对两者 message 都是 "Cannot connect to proxy."，不能据此判断；
    必须下钻到底层 TimeoutError vs ConnectionRefusedError/10061。
    """
    timeout_exc, refused_exc = _make_refused_and_timeout_exc()
    assert OKXClient._is_connection_refused(timeout_exc) is False
    assert OKXClient._is_connection_refused(refused_exc) is True
    assert OKXClient._is_connection_refused(ConnectionRefusedError(111, "Connection refused")) is True
    assert OKXClient._is_connection_refused(None) is False


def test_mark_proxy_failed_hard_fail_uses_max_disable():
    """hard_fail=True（连接被拒/无监听）时，禁用冷却直接拉满 _proxy_max_disable。"""
    import time
    proxy = "http://127.0.0.1:7897"
    c = _make_client(proxy)
    for _ in range(c._proxy_fail_threshold):
        c._mark_proxy_failed(proxy, hard_fail=True)
    remaining = c._proxy_states[proxy]["disabled_until"] - time.time()
    # 拉满：remaining 应接近 _proxy_max_disable（600s），远大于软失败基础时长 60s
    assert 0 < remaining <= c._proxy_max_disable
    assert remaining > c._proxy_disable_duration * 2


def test_mark_proxy_failed_soft_fail_exponential_backoff():
    """hard_fail=False（超时/半死）时，首个禁用周期按基础时长指数退避（60s）。"""
    import time
    proxy = "http://127.0.0.1:7897"
    c = _make_client(proxy)
    for _ in range(c._proxy_fail_threshold):
        c._mark_proxy_failed(proxy, hard_fail=False)
    remaining = c._proxy_states[proxy]["disabled_until"] - time.time()
    # 首个 disable_cycle=1 → backoff = _proxy_disable_duration * 2^0 = 60s
    assert 0 < remaining <= c._proxy_disable_duration + 1.0

