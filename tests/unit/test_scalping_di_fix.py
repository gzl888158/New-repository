"""scalping_strategy._check_trend_ready_for_entry -DM 条件反转修复的单元测试。

修复前缺陷（strategies/scalping_strategy.py:1982）：
    down = lows[i-1] - lows[i] if lows[i] > lows[i-1] else 0   # 条件写反
导致 down 恒为 0 → minus_di 恒为 0 → 空头方向 short 100% 被
「-DI=0.0 <= +DI=x」拒绝，scalping 空头方向完全失效。

修复后：down = lows[i-1] - lows[i] if lows[i-1] > lows[i] else 0，
下行趋势中 minus_di > plus_di，空头方向可正常通过趋势门禁。
"""

import asyncio

import pytest

from strategies.scalping_strategy import ScalpingStrategy


class _FakeOkx:
    """仅提供 _check_trend_ready_for_entry 依赖的 get_kline。"""

    def __init__(self, klines):
        self._klines = klines

    def get_kline(self, symbol, timeframe, limit=50):
        return self._klines


def _make_target(klines):
    """绕过 ScalpingStrategy 重型 __init__，仅装配 okx_client 与 DX 历史缓存。"""
    s = object.__new__(ScalpingStrategy)
    s.okx_client = _FakeOkx(klines)
    # P33 趋势确认门禁依赖 _dx_history（Wilder 平滑），绕过 __init__ 时需手动补齐
    s._dx_history = {}
    return s


def _synth_downtrend_klines(n=50, start=100.0, step=0.5):
    """构造明显下行趋势 K 线，格式 [ts, open, high, low, close, vol]。"""
    klines = []
    price = start
    for i in range(n):
        close = price
        open_ = price + step * 0.4   # 高开低走
        high = open_ + 0.3
        low = close - 0.3
        klines.append([i, open_, high, low, close, 100.0])
        price -= step
    return klines


def test_short_direction_not_rejected_by_zero_minus_di():
    """下行趋势：short 方向应通过趋势门禁（修复前 -DI=0 恒被拒）。"""
    klines = _synth_downtrend_klines()
    target = _make_target(klines)

    # price 取近期区间中高位，满足做空「等反弹到高位」入场条件（price_position >= 0.35）
    # 50 根 K 线价格 100 → 75.5，近期 low≈75.2、high≈85.5，price=82 → position≈0.66
    result = asyncio.run(target._check_trend_ready_for_entry("TEST-USDT-SWAP", "short", 82.0))

    assert result is True


def test_downtrend_still_waits_for_rally_at_low():
    """下行趋势但 price 在低位：short 应等反弹（price_position < 0.35），这是等位入场的正确行为。"""
    klines = _synth_downtrend_klines()
    target = _make_target(klines)

    # price=76 贴近近期低位，price_position≈0.08 < 0.35，应被「too low, wait for rally」拒绝
    result = asyncio.run(target._check_trend_ready_for_entry("TEST-USDT-SWAP", "short", 76.0))

    assert result is False
