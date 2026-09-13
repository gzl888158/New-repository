"""WebSocket 断连/半死导致交易停滞 — 修复验证测试。

背景：系统行情链路为 OKXClient 内建 WebSocket（books 高频推送）→ MarketDataService
质检 → redis tick。旧实现存在两个导致「交易停滞」的缺陷：

1. okx_client._monitor_public_connection 用 `max(last_pong, last_data)` 合并判断心跳健康，
   当「TCP 仍 OPEN 但服务器停止推送 books」（半死连接）且应用层 ping/pong 仍正常时，
   数据断流被新鲜的 pong 掩盖，监控永远不触发重连 → 行情永久停滞。

2. market_data_service._rest_fallback_loop 仅在 `not is_ws_public_connected()`（TCP 断开）
   时轮询 REST，半死状态下连接布尔值仍为 True，REST 兜底死等 → 数据断流超过
   risk_gate network_timeout_threshold(300s) 后触发 L5 熔断冻结全部交易。

修复：public 监控新增独立的数据断流检测（books 断流 > data_timeout 即强制重连）；
REST fallback 触发条件扩大到「连接断开 OR 数据断流（public_last_data_age 超阈值）」。
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.okx_client import OKXClient


def _make_ws_client(last_data_age: float, last_pong_age: float) -> OKXClient:
    """构造一个最小依赖的 OKXClient，用于驱动 _monitor_public_connection 单次迭代。"""
    c = object.__new__(OKXClient)
    c._ws_running = True
    c._ws_public_reconnecting = False
    c._ws_public = MagicMock()
    c._ws_public.close = AsyncMock()
    # 覆盖 staticmethod，模拟连接状态 OPEN
    c._is_ws_open = lambda ws: True
    now = time.time()
    c._ws_public_last_data = now - last_data_age
    c._ws_public_last_pong = now - last_pong_age
    c._ws_data_timeout = 120
    c._ws_heartbeat_interval = 30
    c._ws_pong_timeout = 30
    c._ws_public_connected = True
    c._subscribed_public_channels = [{"channel": "books", "instId": "BTC-USDT-SWAP"}]
    return c


def test_public_data_stall_triggers_reconnect():
    """数据断流(>data_timeout)但 pong 正常时，必须强制重连（半死连接检测）。"""
    c = _make_ws_client(last_data_age=300, last_pong_age=10)
    connect_called = {"n": 0}

    async def fake_connect():
        connect_called["n"] += 1
        c._ws_running = False  # 单次迭代后退出监控循环

    c._connect_public_ws = fake_connect

    asyncio.run(c._monitor_public_connection())

    assert connect_called["n"] == 1, "数据断流但 pong 正常时必须触发重连"
    c._ws_public.close.assert_called_once()
    assert c._ws_public_connected is False


def test_public_data_flowing_no_reconnect():
    """数据正常流动（books 持续推送）时，即使 pong 超时也不应重连。"""
    c = _make_ws_client(last_data_age=5, last_pong_age=120)
    connect_called = {"n": 0}

    async def fake_connect():
        connect_called["n"] += 1
        c._ws_running = False

    c._connect_public_ws = fake_connect

    # 只让监控循环跑两个周期（约 2s + 5s + 5s），数据正常时不应触发重连
    async def run_bounded():
        task = asyncio.ensure_future(c._monitor_public_connection())
        await asyncio.sleep(2.5)  # 越过初始 2s，进入首次 else 分支
        assert connect_called["n"] == 0
        c._ws_running = False
        await task

    asyncio.run(run_bounded())

    assert connect_called["n"] == 0, "数据正常流动时不应误触发重连"
    assert c._ws_public_connected is True


def test_rest_fallback_triggers_on_data_stale():
    """WS 连接正常但数据断流时，REST fallback 应兜底轮询（不再死等 TCP 断开）。"""
    from services.market_data_service import MarketDataService

    okx = MagicMock()
    okx.is_ws_public_connected.return_value = True  # 半死：连接 OPEN
    okx.get_ws_status.return_value = {"public_last_data_age": 300.0}  # 但 books 断流 300s
    ticker = {
        "last": "60000", "vol24h": "100",
        "bidPx": "59999", "bidSz": "1",
        "askPx": "60001", "askSz": "1",
        "ts": str(int(time.time() * 1000)),
    }
    okx.get_ticker.return_value = ticker

    redis = MagicMock()
    svc = MarketDataService({"market_data": {}}, okx, redis)
    svc._subscribed_symbols = ["BTC-USDT-SWAP"]
    svc._running = True
    svc._quality_checker = MagicMock()
    svc._quality_checker.check_tick.return_value = {"quality_score": 1.0}

    ticker_calls = {"n": 0}

    def fake_get_ticker(symbol):
        ticker_calls["n"] += 1
        svc._running = False  # 第一次轮询后停止循环
        return ticker

    okx.get_ticker.side_effect = fake_get_ticker

    asyncio.run(svc._rest_fallback_loop())

    assert ticker_calls["n"] >= 1, "数据断流时应通过 REST 轮询兜底"
    redis.set_tick.assert_called()
