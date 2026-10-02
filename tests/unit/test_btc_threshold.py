"""
资金分级 BTC 阈值显式化（high_value_equity_threshold）单元测试
============================================================
覆盖：SignalProcessor 从 trading.high_value_equity_threshold 读取（默认 2000.0）。
"""
import pytest

from services.signal_processor import SignalProcessor


def _make_sp(config):
    return SignalProcessor(config, None, None, None, None, None, None, None, None)


def test_signal_processor_btc_threshold_from_config():
    sp = _make_sp({"trading": {"high_value_equity_threshold": 1500.0}})
    assert sp._high_value_equity_threshold == pytest.approx(1500.0)


def test_signal_processor_btc_threshold_default():
    sp = _make_sp({})
    assert sp._high_value_equity_threshold == pytest.approx(2000.0)
