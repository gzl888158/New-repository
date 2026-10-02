from unittest.mock import Mock

import pytest

from core.okx_client import OKXClient
from core.risk_gate import RiskGate
from risk.adaptive_controller import AdaptiveController
from risk.pnl_reconciler import PnLReconciler
from risk.risk_limits import RiskLimits


def test_checked_position_query_distinguishes_empty_from_failure():
    client = object.__new__(OKXClient)
    client._make_request = Mock(return_value={"code": "0", "data": []})

    assert client.get_positions_checked() == []

    client._make_request.return_value = {"code": "500", "msg": "unavailable"}
    assert client.get_positions_checked() is None


def test_risk_gate_returns_unknown_capacity_when_exchange_query_fails():
    gate = object.__new__(RiskGate)
    gate._l1 = Mock()
    gate._l1.get_cached_positions.return_value = None
    gate._okx_client = Mock()
    gate._okx_client.get_positions_checked.return_value = None

    assert gate.get_active_position_count() is None


@pytest.mark.asyncio
async def test_pnl_reconciliation_does_not_close_positions_on_query_failure():
    client = Mock()
    client.get_positions_checked.return_value = None
    storage = Mock()
    reconciler = PnLReconciler({}, client, storage)

    result = await reconciler._reconcile_open_positions()

    assert result["error"] == "okx_positions_none"
    storage.get_all_open_records.assert_not_called()


@pytest.mark.asyncio
async def test_risk_limit_margin_snapshot_survives_query_failure():
    client = Mock()
    client.get_positions_checked.return_value = None
    limits = RiskLimits({"risk": {}, "trading": {}}, client)
    limits._symbol_margins = {"BTC-USDT-SWAP": 12.0}
    limits._strategy_margins = {"trend": 8.0}

    await limits._update_margins()

    assert limits._symbol_margins == {"BTC-USDT-SWAP": 12.0}
    assert limits._strategy_margins == {"trend": 8.0}


@pytest.mark.asyncio
async def test_adaptive_position_reconciliation_aborts_on_query_failure():
    controller = object.__new__(AdaptiveController)
    controller.okx_client = Mock()
    controller.okx_client.get_positions_checked.return_value = None
    controller.sqlite_storage = Mock()

    await controller._check_position_consistency()

    controller.sqlite_storage.get_trade_records_by_status.assert_not_called()
