"""grid_strategy._calculate_adx TR 归一化修复的单元测试（P34）。

修复前缺陷：DI 未除以 True Range，导致：
- 高价币（ETH~3000）：DM×100 远超 100 → +DI/-DI 双双 clamp 到 100，方向判断恒 false
- 低价币（DOGE~0.1）：DM×100 接近 0 → +DI/-DI 双双为 0，无法判断方向

修复后 DI = (DM / TR) × 100，与价格刻度无关，方向判断恢复有效。
"""

import numpy as np
import pytest

from strategies.grid_strategy import GridStrategy


def _make_target():
    """绕过 GridStrategy 重型 __init__，仅装配 _calculate_adx 所需状态。"""
    g = object.__new__(GridStrategy)
    g._dx_history = {}
    g._trend_confirmation_mode = "regime_aware"
    return g


def _synth(closes, hi_pad, lo_pad):
    closes = np.asarray(closes, dtype=float)
    highs = closes + hi_pad
    lows = closes - lo_pad
    return highs, lows, closes


def test_high_price_uptrend_di_not_saturated():
    """高价币上行：+DI > -DI，且不饱和到 100（修复前 +DI 被 clamp 到 100）。"""
    base = 3000.0
    closes = base + np.arange(60) * 5.0
    highs, lows, closes = _synth(closes, 2.0, 2.0)
    _, plus_di, minus_di = _make_target()._calculate_adx(highs, lows, closes, "T")
    assert plus_di > minus_di
    assert plus_di < 100.0            # 核心：不再饱和
    assert minus_di == pytest.approx(0.0, abs=1e-6)


def test_high_price_choppy_di_not_both_saturated():
    """高价币震荡：+DI/-DI 均 < 100（修复前双双 clamp 到 100，+DI<=-DI 恒 true）。"""
    base = 3000.0
    moves = np.array([10.0 if i % 2 == 0 else -10.0 for i in range(60)])
    closes = base + np.cumsum(moves)
    highs, lows, closes = _synth(closes, 15.0, 15.0)
    _, plus_di, minus_di = _make_target()._calculate_adx(highs, lows, closes, "T")
    assert 0.0 < plus_di < 100.0
    assert 0.0 < minus_di < 100.0


def test_low_price_uptrend_di_not_degenerate():
    """低价币上行：+DI 不为 0（修复前退化为 0，导致 0<=0 恒 reject）。"""
    base = 0.1
    closes = base + np.arange(60) * 0.001
    highs, lows, closes = _synth(closes, 0.0005, 0.0005)
    _, plus_di, minus_di = _make_target()._calculate_adx(highs, lows, closes, "T")
    assert plus_di > 0.0
    assert plus_di > minus_di


def test_downtrend_minus_di_dominant():
    """下行：-DI > +DI。"""
    base = 3000.0
    closes = base - np.arange(60) * 5.0
    highs, lows, closes = _synth(closes, 2.0, 2.0)
    _, plus_di, minus_di = _make_target()._calculate_adx(highs, lows, closes, "T")
    assert minus_di > plus_di


# ── P33 门禁「趋势模式下才启用」修复 ─────────────────────────

def _klines(n=50, high=100.0, low=90.0, close=95.0):
    """构造 n 根 K 线：[ts, open, high, low, close, vol]。"""
    return [[i, close, high, low, close, 1.0] for i in range(n)]


def _target_with_adx(adx, plus_di, minus_di):
    from unittest.mock import MagicMock, AsyncMock
    g = _make_target()
    g.okx_client = MagicMock()
    g.okx_client.get_kline_async = AsyncMock(return_value=_klines())
    g._calculate_adx = MagicMock(return_value=(adx, plus_di, minus_di))
    return g


@pytest.mark.asyncio
async def test_p33_ranging_allows_entry():
    """震荡市（ADX<20）放行，不再被 ADX 门槛拒绝。"""
    g = _target_with_adx(adx=10.0, plus_di=20.0, minus_di=20.0)
    assert await g._check_trend_ready_for_entry("T", "buy", 95.0) is True
    assert await g._check_trend_ready_for_entry("T", "sell", 95.0) is True


@pytest.mark.asyncio
async def test_p33_trend_buy_against_di_rejected():
    """趋势市（ADX>=20）做多但 +DI<=-DI → 拒绝（逆势）。"""
    g = _target_with_adx(adx=30.0, plus_di=15.0, minus_di=25.0)
    assert await g._check_trend_ready_for_entry("T", "buy", 95.0) is False


@pytest.mark.asyncio
async def test_p33_trend_sell_against_di_rejected():
    """趋势市（ADX>=20）做空但 -DI<=+DI → 拒绝（逆势）。"""
    g = _target_with_adx(adx=30.0, plus_di=25.0, minus_di=15.0)
    assert await g._check_trend_ready_for_entry("T", "sell", 95.0) is False


@pytest.mark.asyncio
async def test_p33_trend_aligned_allows():
    """趋势市顺势（+DI>-DI 且价格位置合理）→ 放行。"""
    g = _target_with_adx(adx=30.0, plus_di=25.0, minus_di=15.0)
    assert await g._check_trend_ready_for_entry("T", "buy", 95.0) is True


@pytest.mark.asyncio
async def test_p33_trend_buy_price_too_high_rejected():
    """趋势市顺势但价格在近期高位（追高）→ 拒绝。"""
    g = _target_with_adx(adx=30.0, plus_di=25.0, minus_di=15.0)
    assert await g._check_trend_ready_for_entry("T", "buy", 99.0) is False


@pytest.mark.asyncio
async def test_p33_strict_ranging_rejected():
    """strict 模式：震荡市（ADX<20）直接拒绝（旧行为，可回退）。"""
    g = _target_with_adx(adx=10.0, plus_di=20.0, minus_di=20.0)
    g._trend_confirmation_mode = "strict"
    assert await g._check_trend_ready_for_entry("T", "buy", 95.0) is False
    assert await g._check_trend_ready_for_entry("T", "sell", 95.0) is False


@pytest.mark.asyncio
async def test_p33_strict_trend_aligned_allows():
    """strict 模式：趋势市（ADX>=20）顺势仍放行。"""
    g = _target_with_adx(adx=30.0, plus_di=25.0, minus_di=15.0)
    g._trend_confirmation_mode = "strict"
    assert await g._check_trend_ready_for_entry("T", "buy", 95.0) is True