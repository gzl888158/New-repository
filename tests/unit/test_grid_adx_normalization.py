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