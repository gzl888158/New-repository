"""
减仓/平仓 reduce_only 正确性回归测试（模块 6 高风险用例）
======================================================

覆盖 core.okx_client.OKXClient.place_order 的 reduce_only 下单参数构建：
- reduceOnly 标记是否正确写入
- posSide 映射：reduce_only 时 sell→long(平多)、buy→short(平空)；开仓时 buy→long、sell→short
- 现货单用 tdMode=cash，不写 posSide/lever
- 减仓/平仓单匹配持仓实际保证金模式（cross/isolated），防止 51169 幽灵平仓
- 减仓/平仓单数量向上取整（round_up=True），数量不足最小单位时跳过下单

通过 object.__new__ 绕过 __init__，注入 coin_to_contracts / round_quantity_to_lot /
_get_position_mgn_mode / _make_request 桩，拦截最终请求体做确定性断言。
"""

import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.okx_client import OKXClient


def _make_client(actual_mgn_mode=""):
    """构造带拦截桩的 OKXClient，返回 (client, captured) 用于断言请求体。"""
    client = OKXClient.__new__(OKXClient)

    captured = {"calls": 0, "body": None}
    round_calls = []

    client.coin_to_contracts = lambda symbol, qty: qty

    def fake_round(symbol, qty, round_up=False):
        round_calls.append(round_up)
        return qty

    client.round_quantity_to_lot = fake_round
    client._get_position_mgn_mode = lambda symbol, pos_side: actual_mgn_mode

    def fake_make_request(method, path, body=None):
        captured["calls"] += 1
        captured["body"] = json.loads(body) if body else {}
        return {"code": "0", "data": [{"ordId": "test_ord"}]}

    client._make_request = fake_make_request

    return client, captured, round_calls


class TestReduceOnlyPosSide:
    def test_close_long_sell_maps_long(self):
        """平多：reduce_only=True + sell → posSide=long"""
        client, captured, round_calls = _make_client()
        client.place_order(
            symbol="BTC-USDT-SWAP", side="sell", order_type="market",
            quantity=0.01, reduce_only=True,
        )
        body = captured["body"]
        assert body["reduceOnly"] is True
        assert body["posSide"] == "long"
        assert body["side"] == "sell"
        assert round_calls == [True], "reduce_only 单应向上取整"

    def test_close_short_buy_maps_short(self):
        """平空：reduce_only=True + buy → posSide=short"""
        client, captured, _ = _make_client()
        client.place_order(
            symbol="BTC-USDT-SWAP", side="buy", order_type="market",
            quantity=0.01, reduce_only=True,
        )
        body = captured["body"]
        assert body["reduceOnly"] is True
        assert body["posSide"] == "short"
        assert body["side"] == "buy"

    def test_open_long_buy_maps_long_no_reduce_only(self):
        """开多：无 reduceOnly，buy → posSide=long"""
        client, captured, _ = _make_client()
        client.place_order(
            symbol="BTC-USDT-SWAP", side="buy", order_type="limit",
            quantity=0.01, price=65000.0, reduce_only=False,
        )
        body = captured["body"]
        assert "reduceOnly" not in body
        assert body["posSide"] == "long"

    def test_open_short_sell_maps_short(self):
        """开空：sell → posSide=short"""
        client, captured, _ = _make_client()
        client.place_order(
            symbol="BTC-USDT-SWAP", side="sell", order_type="limit",
            quantity=0.01, price=65000.0, reduce_only=False,
        )
        assert "reduceOnly" not in captured["body"]
        assert captured["body"]["posSide"] == "short"

    def test_explicit_pos_side_overrides_inference(self):
        """显式 posSide 优先于自动推断"""
        client, captured, _ = _make_client()
        client.place_order(
            symbol="BTC-USDT-SWAP", side="sell", order_type="market",
            quantity=0.01, reduce_only=True, pos_side="short",
        )
        assert captured["body"]["posSide"] == "short"


class TestTdMode:
    def test_spot_uses_cash_no_pos_side_no_lever(self):
        """现货单：tdMode=cash，不写 posSide/lever"""
        client, captured, _ = _make_client()
        client.place_order(
            symbol="BTC-USDT", side="buy", order_type="limit",
            quantity=0.01, price=65000.0,
        )
        body = captured["body"]
        assert body["tdMode"] == "cash"
        assert "posSide" not in body
        assert "lever" not in body

    def test_swap_default_isolated(self):
        """合约单默认 tdMode=isolated"""
        client, captured, _ = _make_client()
        client.place_order(
            symbol="BTC-USDT-SWAP", side="buy", order_type="limit",
            quantity=0.01, price=65000.0,
        )
        assert captured["body"]["tdMode"] == "isolated"

    def test_reduce_only_matches_actual_cross_margin(self):
        """减仓单匹配持仓实际 cross 模式，防 51169"""
        client, captured, _ = _make_client(actual_mgn_mode="cross")
        client.place_order(
            symbol="BTC-USDT-SWAP", side="sell", order_type="market",
            quantity=0.01, reduce_only=True,
        )
        assert captured["body"]["tdMode"] == "cross"

    def test_reduce_only_matches_actual_isolated_margin(self):
        """减仓单匹配持仓实际 isolated 模式"""
        client, captured, _ = _make_client(actual_mgn_mode="isolated")
        client.place_order(
            symbol="BTC-USDT-SWAP", side="sell", order_type="market",
            quantity=0.01, reduce_only=True,
        )
        assert captured["body"]["tdMode"] == "isolated"


class TestQuantitySkip:
    def test_below_lot_skips_order(self):
        """数量不足最小单位时跳过下单，返回 None 且不发请求"""
        client = OKXClient.__new__(OKXClient)
        client.coin_to_contracts = lambda symbol, qty: qty
        client.round_quantity_to_lot = lambda symbol, qty, round_up=False: 0.0
        client._get_position_mgn_mode = lambda symbol, pos_side: ""

        calls = {"n": 0}
        client._make_request = lambda method, path, body=None: calls.__setitem__("n", calls["n"] + 1)

        result = client.place_order(
            symbol="BTC-USDT-SWAP", side="buy", order_type="limit",
            quantity=0.0001, price=65000.0,
        )
        assert result is None
        assert calls["n"] == 0, "数量不足时不应发下单请求"
