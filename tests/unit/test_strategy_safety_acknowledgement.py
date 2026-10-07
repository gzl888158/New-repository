from unittest.mock import AsyncMock, MagicMock

import pytest

from services.signal_processor import SignalProcessor
from execution.order_executor import OrderExecutor
from strategies.grid_strategy import GridStrategy
from strategies.spot_martingale_strategy import SpotMartingaleStrategy


@pytest.mark.asyncio
async def test_spot_martingale_preserves_position_when_close_signal_is_rejected():
    strategy = SpotMartingaleStrategy.__new__(SpotMartingaleStrategy)
    strategy._active_positions = {
        "BTC-USDT": {
            "total_quantity": 2.0,
            "avg_entry_price": 100.0,
            "current_layers": 2,
        }
    }
    strategy.okx_client = MagicMock()
    strategy.okx_client.get_spot_balance.return_value = {
        "available": 2.0,
        "total": 2.0,
    }
    strategy.okx_client.get_instrument_info.return_value = {"lotSz": "0.001"}
    strategy.okx_client.get_instrument_info_async = AsyncMock(return_value={"lotSz": "0.001"})
    strategy._get_quantity_precision = AsyncMock(return_value=3)
    strategy._safe_float = MagicMock(return_value=0.001)
    strategy._place_martingale_order = AsyncMock(return_value=False)

    await strategy._close_position("BTC-USDT", 95.0, "stop_loss")

    assert strategy._active_positions["BTC-USDT"]["total_quantity"] == 2.0
    strategy._place_martingale_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_spot_martingale_keeps_position_tracked_until_balance_confirms_close():
    strategy = SpotMartingaleStrategy.__new__(SpotMartingaleStrategy)
    strategy._active_positions = {
        "BTC-USDT": {
            "total_quantity": 2.0,
            "avg_entry_price": 100.0,
            "current_layers": 1,
            "status": "active",
        }
    }
    strategy.okx_client = MagicMock()
    strategy.okx_client.get_spot_balance.return_value = {
        "available": 2.0,
        "total": 2.0,
    }
    strategy.okx_client.get_instrument_info.return_value = {"lotSz": "0.001"}
    strategy.okx_client.get_instrument_info_async = AsyncMock(return_value={"lotSz": "0.001"})
    strategy._get_quantity_precision = AsyncMock(return_value=3)
    strategy._safe_float = MagicMock(return_value=0.001)
    strategy._place_martingale_order = AsyncMock(return_value=True)

    await strategy._close_position("BTC-USDT", 95.0, "stop_loss")

    assert strategy._active_positions["BTC-USDT"]["status"] == "closing"


@pytest.mark.asyncio
async def test_spot_martingale_partial_close_waits_for_balance_reconciliation():
    strategy = SpotMartingaleStrategy.__new__(SpotMartingaleStrategy)
    strategy._active_positions = {
        "BTC-USDT": {
            "total_quantity": 2.0,
            "avg_entry_price": 100.0,
            "current_layers": 3,
            "status": "active",
        }
    }
    strategy.okx_client = MagicMock()
    strategy.okx_client.get_spot_balance.return_value = {
        "available": 2.0,
        "total": 2.0,
    }
    strategy.okx_client.get_instrument_info.return_value = {"lotSz": "0.001"}
    strategy.okx_client.get_instrument_info_async = AsyncMock(return_value={"lotSz": "0.001"})
    strategy._get_quantity_precision = AsyncMock(return_value=3)
    strategy._safe_float = MagicMock(return_value=0.001)
    strategy._place_martingale_order = AsyncMock(return_value=True)

    await strategy._partial_close("BTC-USDT", 105.0, 1.0, 3)

    position = strategy._active_positions["BTC-USDT"]
    assert position["status"] == "closing"
    assert position["total_quantity"] == 2.0

    strategy._reconcile_closing_position("BTC-USDT", 1.0)

    assert position["status"] == "active"
    assert position["total_quantity"] == 1.0
    assert position["current_layers"] == 2


@pytest.mark.asyncio
async def test_spot_martingale_preserves_position_when_balance_is_unavailable():
    strategy = SpotMartingaleStrategy.__new__(SpotMartingaleStrategy)
    strategy._active_positions = {"BTC-USDT": {"total_quantity": 1.0}}
    strategy.okx_client = MagicMock()
    strategy.okx_client.get_spot_balance.return_value = None
    strategy._place_martingale_order = AsyncMock()

    await strategy._close_position("BTC-USDT", 95.0, "stop_loss")

    assert "BTC-USDT" in strategy._active_positions
    strategy._place_martingale_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_spot_martingale_sell_signal_is_marked_as_close():
    strategy = SpotMartingaleStrategy.__new__(SpotMartingaleStrategy)
    strategy._validate_symbol = MagicMock(return_value=True)
    strategy._validate_direction = MagicMock(return_value=True)
    strategy._validate_price = MagicMock(return_value=True)
    strategy._validate_quantity = MagicMock(return_value=True)
    strategy._get_quantity_precision = AsyncMock(return_value=3)
    strategy._signal_callback = AsyncMock(return_value=True)
    strategy._record_metric = MagicMock()

    dispatched = await strategy._place_martingale_order("BTC-USDT", "sell", 100.0, 0.1, 1)

    assert dispatched is True
    signal = strategy._signal_callback.await_args.args[0]
    assert signal.signal_type == "spot_martingale_close"


def test_grid_refuses_position_margin_when_exchange_query_is_unavailable():
    strategy = GridStrategy.__new__(GridStrategy)
    strategy.okx_client = MagicMock()
    strategy.okx_client.get_positions.return_value = None

    assert strategy._get_existing_symbol_margin("BTC-USDT-SWAP") is None


@pytest.mark.asyncio
async def test_order_executor_reports_queue_admission():
    executor = OrderExecutor.__new__(OrderExecutor)
    executor._order_queue = MagicMock()
    executor._order_queue.add_order = AsyncMock(return_value="order-1")

    admitted = await executor.handle_signal({
        "symbol": "BTC-USDT",
        "direction": "sell",
        "price": 100.0,
        "quantity": 0.1,
        "leverage": 1,
        "strategy_name": "spot_martingale",
        "signal_type": "spot_martingale_stop_loss",
        "timestamp": "2025-01-01T00:00:00",
    })

    assert admitted is True
    executor._order_queue.add_order.assert_awaited_once()


@pytest.mark.asyncio
async def test_signal_callback_reports_rejected_processing_as_not_dispatched():
    processor = SignalProcessor.__new__(SignalProcessor)
    processor.setup_signal_routing([])
    processor._process_signal = AsyncMock(return_value=None)

    accepted = await processor._signal_callback(
        {"symbol": "BTC-USDT", "strategy_name": "spot_martingale"}
    )

    assert accepted is False


@pytest.mark.asyncio
async def test_bear_case_gate_rejects_open_signal_before_order_dispatch():
    order_executor = MagicMock()
    order_executor.handle_signal = AsyncMock(return_value=True)
    processor = SignalProcessor(
        {}, None, None, order_executor, None, None, None, None, None
    )
    processor.set_bear_case_open_gate(
        lambda strategy: "bear_case_horizon=-60 below_threshold=-50"
        if strategy == "grid" else None
    )

    await processor._process_signal({
        "symbol": "ETH-USDT",
        "strategy_name": "grid",
        "direction": "buy",
        "signal_type": "open",
        "quantity": 1.0,
    })

    order_executor.handle_signal.assert_not_awaited()
    assert processor._dead_letters[-1]["reject_reason"].startswith(
        "bear_case_projection:"
    )


def test_bear_case_gate_exempts_close_and_reduce_only_signals():
    processor = SignalProcessor(
        {}, None, None, None, None, None, None, None, None
    )
    gate = MagicMock(return_value="bear-case threshold exceeded")
    processor.set_bear_case_open_gate(gate)

    for signal, is_close in (
        ({"strategy_name": "grid", "direction": "close"}, True),
        ({"strategy_name": "grid", "reduce_only": True}, True),
    ):
        assert processor._reject_bear_case_open_signal(signal, is_close) is False

    gate.assert_not_called()
