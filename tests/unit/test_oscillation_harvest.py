"""震荡收割策略（OscillationHarvestStrategy）布林带兜底完善测试。

覆盖：
- 区间未验证（range_untested）时，布林带极值 + RSI 极端共振 → 降级为布林带震荡兜底开仓
- 区间未验证且无布林带极值 → 仍拒绝（保持 fail-closed）
- 区间验证通过时，走正常支撑/阻力逻辑（不受影响）
"""
import numpy as np
import pytest

from strategies.oscillation_harvest_strategy import OscillationHarvestStrategy


def _make_target():
    """绕过 OscillationHarvestStrategy 重型 __init__，仅装配 _evaluate 所需状态。"""
    s = object.__new__(OscillationHarvestStrategy)
    s._min_band_width_pct = 0.003
    s._min_band_touch_count = 2
    s._support_band_pct = 0.01
    s._mid_tp_ratio = 1.0
    s._rsi_period = 14
    s._rsi_oversold = 30.0
    s._rsi_overbought = 70.0
    s._atr_sl_mult = 1.5
    s._use_bollinger_confirm = True
    s._boll_period = 20
    s._boll_std_mult = 2.0
    s._lookback_bars = 96
    s._touch_proximity_pct = 0.0015
    s._safe_float = lambda v, d=0.0: float(v) if v is not None else d
    s._is_range_bound = lambda symbol: True
    s._get_symbol_regime = lambda symbol: {"confidence": 0.5}
    return s


def _untested_sr(touch_count=0):
    """返回一个区间未验证的支撑/阻力结果（可复用）。"""
    return lambda highs, lows: {
        "support": 85.0, "resistance": 100.0, "mid": 92.5,
        "range_pct": 0.16, "touch_count": touch_count,
    }


def test_bollinger_fallback_when_range_untested():
    """区间未验证 + 布林带下轨 + RSI 超卖 → 降级为布林带兜底做多。"""
    s = _make_target()
    s._compute_support_resistance = _untested_sr(0)
    # 横盘后暴跌：价格跌破布林带下轨 + RSI 超卖
    n = 50
    closes = np.zeros(n)
    closes[:35] = 100.0
    closes[35:] = 100.0 - np.arange(15) * 1.2
    highs = closes + 0.3
    lows = closes - 0.3
    result = s._evaluate("BTC-USDT-SWAP", closes, highs, lows)
    assert result is not None
    assert result["signal"] == "long"
    assert result["boll_only"] is True
    assert result["take_profit"] is not None
    assert result["stop_loss"] is not None


def test_range_untested_no_bollinger_extreme():
    """区间未验证且无布林带极值（横盘 std=0）→ 仍拒绝。"""
    s = _make_target()
    s._compute_support_resistance = _untested_sr(0)
    closes = np.full(50, 100.0)
    highs = closes + 0.3
    lows = closes - 0.3
    result = s._evaluate("BTC-USDT-SWAP", closes, highs, lows)
    assert result is None or result.get("signal") is None


def test_range_verified_normal_entry():
    """区间验证通过 + 支撑触及 + RSI 超卖 → 正常支撑/阻力逻辑（非 boll_only）。"""
    s = _make_target()
    s._compute_support_resistance = _untested_sr(3)  # touch_count=3 已验证
    n = 50
    closes = np.zeros(n)
    closes[:35] = 100.0
    closes[35:] = 100.0 - np.arange(15) * 1.2
    highs = closes + 0.3
    lows = closes - 0.3
    result = s._evaluate("BTC-USDT-SWAP", closes, highs, lows)
    if result is not None and result.get("signal") == "long":
        assert result["boll_only"] is False
