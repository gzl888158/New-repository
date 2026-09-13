"""
条件单参数校验回归测试（模块 6 高风险用例）
==========================================

覆盖条件单下单前的关键参数校验，防止被交易所以确定性错误拒绝：

validate_conditional_price（core.okx_client）：
- 方向校验：long 止盈须 > 最新价、止损须 < 最新价；short 相反（防 sCode 51277）
- 非正触发价拒绝；无行情/无方向时 fail-open 交由交易所判定

_calculate_stop_price（core.conditional_order_manager）：
- long 止损在 mark 下方、short 在 mark 上方
- 强平距离钳制到 [0.8%, 8%]，兼顾高杠杆防插针与低杠杆不过松

_check_immediate_trigger（防 -2021 "Order would immediately trigger"）：
- long 触发价 >= 现价、short 触发价 <= 现价判定为会立即触发

_gen_algo_cl_ord_id（幂等 algoClOrdId）：
- 同参数确定性、仅字母数字、长度 <= 32、不同价格不同 ID

_is_retryable_failure（失败分流）：
- 网络/限流可重试，参数类拒绝(51008/51169/51277/51010)不重试

通过 object.__new__ 注入 get_ticker / config 桩做确定性断言。
"""

import sys
import os
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.okx_client import OKXClient
from core.conditional_order_manager import ConditionalOrderManager


def _make_client(last="100"):
    client = OKXClient.__new__(OKXClient)
    client.get_ticker = lambda symbol: None if last is None else {"last": str(last)}
    return client


def _make_manager(last="100", margin_call_offset=0.30):
    manager = ConditionalOrderManager.__new__(ConditionalOrderManager)
    manager._okx_client = SimpleNamespace(
        get_ticker=lambda symbol: None if last is None else {"last": str(last)}
    )
    manager.config = {"conditional_order": {"stop_loss": {"margin_call_offset": margin_call_offset}}}
    manager._algo_id_salt = "abc123"
    return manager


class TestValidateConditionalPrice:
    @pytest.mark.parametrize("order_type,pos_side,price,expected", [
        ("take_profit", "long", 105.0, 105.0),   # long TP 须 > last
        ("take_profit", "long", 95.0, None),
        ("stop_loss", "long", 95.0, 95.0),       # long SL 须 < last
        ("stop_loss", "long", 105.0, None),
        ("take_profit", "short", 95.0, 95.0),    # short TP 须 < last
        ("take_profit", "short", 105.0, None),
        ("stop_loss", "short", 105.0, 105.0),    # short SL 须 > last
        ("stop_loss", "short", 95.0, None),
    ])
    def test_direction_matrix(self, order_type, pos_side, price, expected):
        result = _make_client("100").validate_conditional_price(
            "BTC-USDT-SWAP", order_type, price, pos_side=pos_side
        )
        if expected is None:
            assert result is None
        else:
            assert result == pytest.approx(expected)

    def test_side_sell_maps_long_position(self):
        """平仓 side=sell 推导为多头持仓"""
        assert _make_client("100").validate_conditional_price(
            "BTC-USDT-SWAP", "take_profit", 105.0, side="sell") == pytest.approx(105.0)
        assert _make_client("100").validate_conditional_price(
            "BTC-USDT-SWAP", "take_profit", 95.0, side="sell") is None

    def test_side_buy_maps_short_position(self):
        """平仓 side=buy 推导为空头持仓"""
        assert _make_client("100").validate_conditional_price(
            "BTC-USDT-SWAP", "take_profit", 95.0, side="buy") == pytest.approx(95.0)
        assert _make_client("100").validate_conditional_price(
            "BTC-USDT-SWAP", "take_profit", 105.0, side="buy") is None

    def test_nonpositive_price_rejected(self):
        assert _make_client("100").validate_conditional_price(
            "BTC-USDT-SWAP", "stop_loss", 0.0, pos_side="long") is None

    def test_no_ticker_fail_open(self):
        assert _make_client(None).validate_conditional_price(
            "BTC-USDT-SWAP", "stop_loss", 95.0, pos_side="long") == pytest.approx(95.0)

    def test_no_direction_fail_open(self):
        assert _make_client("100").validate_conditional_price(
            "BTC-USDT-SWAP", "stop_loss", 95.0) == pytest.approx(95.0)


class TestCalculateStopPrice:
    def test_long_stop_below_mark(self):
        """leverage=10 → offset 0.03 → 97.0"""
        assert _make_manager()._calculate_stop_price("long", 100.0, 10) == pytest.approx(97.0)

    def test_short_stop_above_mark(self):
        assert _make_manager()._calculate_stop_price("short", 100.0, 10) == pytest.approx(103.0)

    def test_high_leverage_clamped_to_min_0_8pct(self):
        """leverage=100 → liq 0.003 → 钳制到 0.008 → 99.2"""
        assert _make_manager()._calculate_stop_price("long", 100.0, 100) == pytest.approx(99.2)

    def test_low_leverage_clamped_to_max_8pct(self):
        """leverage=1 → liq 0.30 → 钳制到 0.08 → 92.0"""
        assert _make_manager()._calculate_stop_price("long", 100.0, 1) == pytest.approx(92.0)


class TestImmediateTrigger:
    def test_long_trigger_at_or_above_current(self):
        manager = _make_manager("100")
        assert manager._check_immediate_trigger("BTC-USDT-SWAP", "long", 100.0) is True
        assert manager._check_immediate_trigger("BTC-USDT-SWAP", "long", 105.0) is True
        assert manager._check_immediate_trigger("BTC-USDT-SWAP", "long", 95.0) is False

    def test_short_trigger_at_or_below_current(self):
        manager = _make_manager("100")
        assert manager._check_immediate_trigger("BTC-USDT-SWAP", "short", 100.0) is True
        assert manager._check_immediate_trigger("BTC-USDT-SWAP", "short", 95.0) is True
        assert manager._check_immediate_trigger("BTC-USDT-SWAP", "short", 105.0) is False

    def test_no_ticker_returns_false(self):
        assert _make_manager(None)._check_immediate_trigger("BTC-USDT-SWAP", "long", 95.0) is False


class TestGenAlgoClOrdId:
    def test_deterministic(self):
        manager = _make_manager()
        a = manager._gen_algo_cl_ord_id("BTC-USDT-SWAP", "long", "stop_loss", 95.0)
        b = manager._gen_algo_cl_ord_id("BTC-USDT-SWAP", "long", "stop_loss", 95.0)
        assert a == b

    def test_alnum_and_max_len(self):
        oid = _make_manager()._gen_algo_cl_ord_id("BTC-USDT-SWAP", "long", "take_profit", 105.5)
        assert oid.isalnum()
        assert len(oid) <= 32

    def test_different_price_differs(self):
        manager = _make_manager()
        a = manager._gen_algo_cl_ord_id("BTC-USDT-SWAP", "long", "take_profit", 105.0)
        b = manager._gen_algo_cl_ord_id("BTC-USDT-SWAP", "long", "take_profit", 106.0)
        assert a != b


class TestIsRetryableFailure:
    @pytest.mark.parametrize("s_code", ["0", "exception", "50004", "50011", "50013", "50014", "51103", "429"])
    def test_retryable_codes(self, s_code):
        assert _make_manager()._is_retryable_failure({"_failed": True, "sCode": s_code, "sMsg": ""}) is True

    @pytest.mark.parametrize("s_code", ["51008", "51169", "51277", "51010"])
    def test_non_retryable_param_rejection(self, s_code):
        assert _make_manager()._is_retryable_failure({"_failed": True, "sCode": s_code, "sMsg": ""}) is False

    def test_keyword_fallback(self):
        assert _make_manager()._is_retryable_failure(
            {"_failed": True, "sCode": "99999", "sMsg": "request timeout"}) is True
        assert _make_manager()._is_retryable_failure(
            {"_failed": True, "sCode": "99999", "sMsg": "order rejected: insufficient margin"}) is False

    def test_success_not_retryable(self):
        assert _make_manager()._is_retryable_failure({"_failed": False}) is False
        assert _make_manager()._is_retryable_failure(None) is False
