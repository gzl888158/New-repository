"""
订单量/价格取整回归测试（模块 6 高风险用例）
============================================

覆盖 core.okx_client.OKXClient 的两个下单正确性纯函数：
- round_quantity_to_lot：按合约 lot size 取整数量
  * round_up=False（开仓）：向下取整，数量不足最小单位返回 0
  * round_up=True（减仓/平仓）：向上取整，至少返回 lot_sz（避免残留仓位）
- round_price_to_tick：按合约 tick size 取整价格

这些函数直接决定下单量/价是否符合交易所精度要求，取整错误会导致
51121（数量不符合 lot size）、PRICE_PRECISION 校验失败或资金不足。

通过 object.__new__ 绕过 OKXClient.__init__（避免网络/密钥/线程副作用），
仅注入 get_instrument_info 返回固定合约规格，纯函数确定性断言。
"""

import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.okx_client import OKXClient


def _make_client(instrument_info):
    """构造绕过 __init__ 的 OKXClient，仅覆盖 get_instrument_info。"""
    client = OKXClient.__new__(OKXClient)
    client.get_instrument_info = lambda symbol: instrument_info
    return client


# lotSz=0.001, tickSz=0.1 的通用合约规格，使取整断言直观
BTC_SPEC = {"lotSz": "0.001", "tickSz": "0.1", "ctVal": "0.01"}


class TestRoundQuantityToLot:
    def test_round_down_open_order(self):
        """开仓向下取整：1.2345 -> 1.234"""
        client = _make_client(BTC_SPEC)
        result = client.round_quantity_to_lot("BTC-USDT-SWAP", 1.2345, round_up=False)
        assert result == pytest.approx(1.234)

    def test_round_up_reduce_order(self):
        """减仓/平仓向上取整：1.2345 -> 1.235"""
        client = _make_client(BTC_SPEC)
        result = client.round_quantity_to_lot("BTC-USDT-SWAP", 1.2345, round_up=True)
        assert result == pytest.approx(1.235)

    def test_round_down_below_min_returns_zero(self):
        """开仓数量不足最小单位：向下取整返回 0（阻止无效下单）"""
        client = _make_client(BTC_SPEC)
        result = client.round_quantity_to_lot("BTC-USDT-SWAP", 0.0005, round_up=False)
        assert result == 0.0

    def test_round_up_below_min_returns_lot(self):
        """减仓数量不足最小单位：向上取整至少返回 lot_sz（避免残留仓位）"""
        client = _make_client(BTC_SPEC)
        result = client.round_quantity_to_lot("BTC-USDT-SWAP", 0.0005, round_up=True)
        assert result == pytest.approx(0.001)

    def test_exact_multiple_unchanged(self):
        """已是 lot 整数倍时不改变"""
        client = _make_client(BTC_SPEC)
        result = client.round_quantity_to_lot("BTC-USDT-SWAP", 2.0, round_up=False)
        assert result == pytest.approx(2.0)

    def test_no_instrument_info_returns_original(self):
        """无合约规格时保守返回原值（fail-open，交由交易所校验）"""
        client = _make_client(None)
        result = client.round_quantity_to_lot("UNKNOWN-SWAP", 1.2345, round_up=False)
        assert result == pytest.approx(1.2345)

    def test_invalid_lot_sz_returns_original(self):
        """lotSz <= 0 时返回原值，避免除零崩溃"""
        client = _make_client({"lotSz": "0", "tickSz": "0.1"})
        result = client.round_quantity_to_lot("BTC-USDT-SWAP", 1.2345, round_up=False)
        assert result == pytest.approx(1.2345)


class TestRoundPriceToTick:
    def test_round_half_up_down(self):
        """价格向下就近取整：65000.04 -> 65000.0"""
        client = _make_client(BTC_SPEC)
        result = client.round_price_to_tick("BTC-USDT-SWAP", 65000.04)
        assert result == pytest.approx(65000.0)

    def test_round_half_up_up(self):
        """价格向上就近取整（half up）：65000.05 -> 65000.1"""
        client = _make_client(BTC_SPEC)
        result = client.round_price_to_tick("BTC-USDT-SWAP", 65000.05)
        assert result == pytest.approx(65000.1)

    def test_no_instrument_info_returns_original(self):
        """无合约规格时返回原值"""
        client = _make_client(None)
        result = client.round_price_to_tick("UNKNOWN-SWAP", 65000.04)
        assert result == pytest.approx(65000.04)

    def test_invalid_tick_sz_returns_original(self):
        """tickSz <= 0 时返回原值"""
        client = _make_client({"lotSz": "0.001", "tickSz": "0"})
        result = client.round_price_to_tick("BTC-USDT-SWAP", 65000.04)
        assert result == pytest.approx(65000.04)
