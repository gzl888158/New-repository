import pytest
import numpy as np
from src.main import calculate_macd, get_single_timeframe_signal

def test_calculate_macd():
    """测试MACD计算"""
    close_prices = np.random.rand(100) * 10000
    macd_line, signal_line = calculate_macd(close_prices)
    assert len(macd_line) == len(signal_line)
    assert len(macd_line) > 0

def test_get_single_timeframe_signal():
    """测试单周期信号"""
    candles = [{"high": i, "low": i-1, "close": i} for i in range(100, 200)]
    signal = get_single_timeframe_signal(candles)
    assert signal in ["buy", "sell", "hold"]