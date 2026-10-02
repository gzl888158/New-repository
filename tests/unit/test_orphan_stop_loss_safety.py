from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from risk.adaptive_controller import AdaptiveController


def _controller():
    controller = object.__new__(AdaptiveController)
    controller.config = {
        "execution": {
            "orphan_stop_loss_enabled": True,
            "orphan_stop_loss_pct": 0.03,
            "orphan_stop_loss_failure_auto_close": True,
        }
    }
    controller.okx_client = Mock()
    controller.okx_client.get_algo_order_history.return_value = []
    controller.sqlite_storage = Mock()
    controller._alert = AsyncMock()
    controller._orphan_close_orders = {}
    return controller


def _position():
    return SimpleNamespace(
        symbol="BTC-USDT-SWAP",
        side="long",
        quantity=1.0,
        leverage=5,
    )


@pytest.mark.asyncio
async def test_stop_order_lookup_failure_does_not_place_duplicate():
    controller = _controller()
    controller.okx_client.get_algo_orders.return_value = None

    await controller._place_orphan_stop_loss(_position())

    controller.okx_client.place_order.assert_not_called()
    controller._alert.assert_awaited_once()
    assert controller._alert.await_args.args[2] == "critical"


@pytest.mark.asyncio
async def test_existing_sync_position_rechecks_protective_stop():
    controller = _controller()
    position = SimpleNamespace(
        symbol="BTC-USDT-SWAP",
        side="long",
        quantity=1.0,
        avg_cost=100.0,
        mark_price=101.0,
        unrealized_pnl=1.0,
        margin=20.0,
        leverage=5,
        liquidation_price=50.0,
    )
    controller.sqlite_storage.get_trade_records_by_status.return_value = [
        {
            "id": 1,
            "symbol": position.symbol,
            "side": "long",
            "strategy_name": "sync",
        }
    ]
    controller.okx_client.get_positions_checked.return_value = [
        {"instId": position.symbol, "posSide": "long", "pos": "1"}
    ]
    controller.okx_client.get_positions.return_value = [
        {"instId": position.symbol, "posSide": "long", "pos": "1"}
    ]
    controller.okx_client._parse_position.return_value = position
    controller.okx_client.get_algo_orders.return_value = []
    controller.okx_client.get_ticker.return_value = {"last": "101"}
    controller.okx_client.contracts_to_coins.return_value = 1.0
    controller.okx_client.place_order.return_value = {
        "algoId": "algo-1",
        "sCode": "0",
    }

    await controller._check_position_consistency()

    controller.okx_client.place_order.assert_called_once()
    assert controller.okx_client.place_order.call_args.kwargs["reduce_only"] is True
    assert controller.okx_client.place_order.call_args.kwargs["clOrdId"]


@pytest.mark.asyncio
async def test_triggered_orphan_stop_with_remaining_position_market_closes_reduce_only():
    controller = _controller()
    position = _position()
    stop_client_id = controller._orphan_stop_client_id(
        position.symbol, position.side
    )
    controller.okx_client.get_algo_orders.return_value = []
    controller.okx_client.get_algo_order_history.return_value = [
        {
            "algoClOrdId": stop_client_id,
            "state": "effective",
            "posSide": "long",
            "slTriggerPx": "97",
            "ordId": "stop-child-1",
        }
    ]
    controller.okx_client.get_order.return_value = {"state": "canceled"}
    controller.okx_client.get_positions_checked.return_value = [
        {"instId": position.symbol, "posSide": "long", "pos": "1"}
    ]
    controller.okx_client._parse_position.return_value = position
    controller.okx_client.contracts_to_coins.return_value = 1.0
    controller.okx_client.place_order.return_value = {
        "ordId": "fallback-close-1",
        "sCode": "0",
    }

    await controller._place_orphan_stop_loss(position)

    controller.okx_client.get_order.assert_called_once_with(
        position.symbol, "stop-child-1"
    )
    controller.okx_client.place_order.assert_called_once()
    close_order = controller.okx_client.place_order.call_args.kwargs
    assert close_order["order_type"] == "market"
    assert close_order["side"] == "sell"
    assert close_order["quantity"] == 1.0
    assert close_order["reduce_only"] is True
    assert close_order["pos_side"] == "long"
    assert close_order["clOrdId"].startswith("ORPHANCLOSE")
    assert controller._orphan_close_orders[
        f"{position.symbol}:long:legacy"
    ] == "fallback-close-1"
    controller._alert.assert_awaited_once()
    assert controller._alert.await_args.args[2] == "critical"


@pytest.mark.asyncio
async def test_legacy_triggered_stop_is_matched_using_position_start_time():
    controller = _controller()
    position = _position()
    position_created_at = datetime.now() - timedelta(seconds=10)
    controller.okx_client.get_algo_orders.return_value = []
    controller.okx_client.get_algo_order_history.return_value = [
        {
            "state": "effective",
            "posSide": "long",
            "slTriggerPx": "97",
            "ordId": "legacy-stop-child",
            "cTime": str(int((position_created_at + timedelta(seconds=1)).timestamp() * 1000)),
        }
    ]
    controller.okx_client.get_order.return_value = {"state": "canceled"}
    controller.okx_client.get_positions_checked.return_value = [
        {"instId": position.symbol, "posSide": "long", "pos": "1"}
    ]
    controller.okx_client._parse_position.return_value = position
    controller.okx_client.contracts_to_coins.return_value = 1.0
    controller.okx_client.place_order.return_value = {
        "ordId": "legacy-fallback-close",
        "sCode": "0",
    }

    await controller._place_orphan_stop_loss(
        position,
        position_token="position-1",
        position_created_at=position_created_at,
    )

    controller.okx_client.place_order.assert_called_once()
    assert controller.okx_client.place_order.call_args.kwargs["order_type"] == "market"
    assert controller.okx_client.place_order.call_args.kwargs["reduce_only"] is True


@pytest.mark.asyncio
async def test_triggered_orphan_stop_uses_fresh_position_quantity_for_fallback():
    controller = _controller()
    position = _position()
    current_position = SimpleNamespace(
        symbol=position.symbol,
        side="long",
        quantity=0.4,
        leverage=5,
    )
    controller.okx_client.get_algo_orders.return_value = []
    controller.okx_client.get_algo_order_history.return_value = [
        {
            "algoClOrdId": controller._orphan_stop_client_id(
                position.symbol, position.side
            ),
            "state": "effective",
            "posSide": "long",
            "slTriggerPx": "97",
            "ordId": "stop-child-partial",
        }
    ]
    controller.okx_client.get_order.return_value = {"state": "canceled"}
    controller.okx_client.get_positions_checked.return_value = [
        {"instId": position.symbol, "posSide": "long", "pos": "0.4"}
    ]
    controller.okx_client._parse_position.return_value = current_position
    controller.okx_client.contracts_to_coins.return_value = 0.4
    controller.okx_client.place_order.return_value = {
        "ordId": "fallback-close-partial",
        "sCode": "0",
    }

    await controller._place_orphan_stop_loss(position)

    assert controller.okx_client.place_order.call_args.kwargs["quantity"] == 0.4


@pytest.mark.asyncio
async def test_triggered_orphan_stop_does_not_close_when_fresh_position_is_empty():
    controller = _controller()
    position = _position()
    controller.okx_client.get_algo_orders.return_value = []
    controller.okx_client.get_algo_order_history.return_value = [
        {
            "algoClOrdId": controller._orphan_stop_client_id(
                position.symbol, position.side
            ),
            "state": "effective",
            "posSide": "long",
            "slTriggerPx": "97",
            "ordId": "stop-child-filled",
        }
    ]
    controller.okx_client.get_order.return_value = {"state": "filled"}
    controller.okx_client.get_positions_checked.return_value = []

    await controller._place_orphan_stop_loss(position)

    controller.okx_client.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_triggered_orphan_stop_fails_closed_when_fresh_position_query_fails():
    controller = _controller()
    position = _position()
    controller.okx_client.get_algo_orders.return_value = []
    controller.okx_client.get_algo_order_history.return_value = [
        {
            "algoClOrdId": controller._orphan_stop_client_id(
                position.symbol, position.side
            ),
            "state": "effective",
            "posSide": "long",
            "slTriggerPx": "97",
            "ordId": "stop-child-unknown-position",
        }
    ]
    controller.okx_client.get_order.return_value = {"state": "canceled"}
    controller.okx_client.get_positions_checked.return_value = None

    await controller._place_orphan_stop_loss(position)

    controller.okx_client.place_order.assert_not_called()
    controller._alert.assert_awaited_once()
    assert controller._alert.await_args.args[2] == "critical"


@pytest.mark.asyncio
async def test_triggered_orphan_stop_waits_while_child_order_is_live():
    controller = _controller()
    position = _position()
    controller.okx_client.get_algo_orders.return_value = []
    controller.okx_client.get_algo_order_history.return_value = [
        {
            "algoClOrdId": controller._orphan_stop_client_id(
                position.symbol, position.side
            ),
            "state": "effective",
            "posSide": "long",
            "slTriggerPx": "97",
            "ordId": "stop-child-live",
        }
    ]
    controller.okx_client.get_order.return_value = {
        "state": "partially_filled"
    }

    await controller._place_orphan_stop_loss(position)

    controller.okx_client.place_order.assert_not_called()
    controller._alert.assert_not_awaited()


@pytest.mark.asyncio
async def test_live_fallback_close_is_not_submitted_twice():
    controller = _controller()
    position = _position()
    lifecycle_key = f"{position.symbol}:long:position-1"
    controller._orphan_close_orders[lifecycle_key] = "fallback-close-live"
    controller.okx_client.get_order.return_value = {
        "state": "partially_filled"
    }

    await controller._place_orphan_stop_loss(
        position,
        position_token="position-1",
        position_created_at=datetime.now(),
    )

    controller.okx_client.get_order.assert_called_once_with(
        position.symbol, "fallback-close-live"
    )
    controller.okx_client.get_algo_orders.assert_not_called()
    controller.okx_client.place_order.assert_not_called()


@pytest.mark.asyncio
async def test_triggered_orphan_stop_without_child_order_id_does_not_duplicate_close():
    controller = _controller()
    position = _position()
    controller.okx_client.get_algo_orders.return_value = []
    controller.okx_client.get_algo_order_history.return_value = [
        {
            "algoClOrdId": controller._orphan_stop_client_id(
                position.symbol, position.side
            ),
            "state": "effective",
            "posSide": "long",
            "slTriggerPx": "97",
        }
    ]

    await controller._place_orphan_stop_loss(position)

    controller.okx_client.get_order.assert_not_called()
    controller.okx_client.place_order.assert_not_called()
    controller._alert.assert_awaited_once()
