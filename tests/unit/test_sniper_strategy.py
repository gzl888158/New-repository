"""市场狙击策略（SniperStrategy）单元测试。

覆盖：
- 指标计算（ADX / RSI / MACD / ATR）的静态方法
- 趋势突破狙击（_evaluate_breakout）：关键位突破 + ADX + MACD + 量能
- 关键位反转狙击（_evaluate_reversal）：支撑/阻力 + RSI 超买超卖 + 反转 K 线
"""
import numpy as np
import pytest

from strategies.sniper_strategy import SniperStrategy


def _make_target():
    """绕过 SniperStrategy 重型 __init__，仅装配信号判断所需状态。"""
    s = object.__new__(SniperStrategy)
    s._strategy_name = "sniper"
    s._breakout_lookback = 20
    s._adx_threshold = 20.0
    s._adx_period = 14
    s._volume_multiplier = 1.5
    s._rsi_period = 14
    s._rsi_oversold = 30.0
    s._rsi_overbought = 70.0
    s._atr_period = 14
    s._atr_sl_mult = 2.0
    s._atr_tp_mult = 3.0
    s._filter_stats = {}
    s._record_metric = lambda *a, **k: None
    s._round_price = lambda symbol, price: price
    return s


# ------------------------------------------------------------------
# 指标计算
# ------------------------------------------------------------------
def test_adx_uptrend_positive():
    closes = 100.0 + np.arange(80) * 2.0
    highs = closes + 1.0
    lows = closes - 1.0
    adx = SniperStrategy._wilder_adx(highs, lows, closes, 14)
    assert adx > 0


def test_rsi_downtrend_oversold():
    closes = 100.0 - np.arange(30) * 3.0
    rsi = SniperStrategy._rsi(closes, 14)
    assert rsi < 30.0


def test_rsi_uptrend_overbought():
    closes = 100.0 + np.arange(30) * 3.0
    rsi = SniperStrategy._rsi(closes, 14)
    assert rsi > 70.0


def test_macd_uptrend_positive():
    closes = 100.0 + np.arange(60) * 1.0
    hist = SniperStrategy._macd_hist(closes)
    assert hist > 0


def test_atr_positive():
    closes = 100.0 + np.arange(40) * 0.5
    highs = closes + 1.5
    lows = closes - 1.5
    atr = SniperStrategy._atr(highs, lows, closes, 14)
    assert atr > 0


# ------------------------------------------------------------------
# 趋势突破狙击
# ------------------------------------------------------------------
def test_breakout_long_signal():
    s = _make_target()
    n = 60
    closes = 100.0 + np.arange(n) * 1.0
    closes[-1] = closes[-2] + 8.0  # 最后一根突破关键位
    opens = np.copy(closes)
    opens[-1] = closes[-2]  # 突破 K 线开盘在关键位下方
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    volumes = np.ones(n) * 100.0
    volumes[-1] = 300.0  # 量能放大 3x

    result = s._evaluate_breakout("BTC-USDT-SWAP", opens, highs, lows, closes, volumes)
    assert result is not None
    assert result[0] == "long"


def test_breakout_no_signal_flat():
    s = _make_target()
    n = 60
    closes = 100.0 + np.sin(np.arange(n) * 0.3) * 0.5  # 无趋势震荡
    opens = np.copy(closes)
    highs = closes + 0.5
    lows = closes - 0.5
    volumes = np.ones(n) * 100.0

    result = s._evaluate_breakout("BTC-USDT-SWAP", opens, highs, lows, closes, volumes)
    assert result is None


# ------------------------------------------------------------------
# 关键位反转狙击
# ------------------------------------------------------------------
def test_reversal_long_signal():
    s = _make_target()
    n = 60
    closes = 100.0 - np.arange(n) * 1.0  # 持续下跌 → RSI 超卖
    opens = np.copy(closes)
    # 最后一根：触及支撑后反转收阳
    opens[-1] = closes[-2] - 1.0
    closes[-1] = closes[-2] + 2.0
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    lows[-1] = closes[-2] - 1.5  # 触及前 20 周期低点支撑
    volumes = np.ones(n) * 100.0

    result = s._evaluate_reversal("BTC-USDT-SWAP", opens, highs, lows, closes, volumes)
    assert result is not None
    assert result[0] == "long"
