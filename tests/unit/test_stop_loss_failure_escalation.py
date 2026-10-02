import sqlite3
from unittest.mock import AsyncMock, Mock

import pytest

from core.stop_loss_manager import StopLossManager


def _manager(db_path):
    manager = StopLossManager(
        {"sqlite": {"db_path": str(db_path)}, "trading": {}},
        None,
        None,
        None,
        Mock(),
    )
    manager._alert_callback = AsyncMock()
    manager._save_stop_loss_audit = Mock()
    manager._execute_stop_loss_order = AsyncMock(return_value=False)
    return manager


@pytest.mark.asyncio
async def test_repeated_stop_loss_failure_is_alerted_and_retained(tmp_path):
    manager = _manager(tmp_path / "repeated.sqlite")
    position = {
        "direction": "long",
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "quantity": 1.0,
    }

    assert not await manager.check_and_execute_stop_loss(
        "BTC-USDT-SWAP", "trend", 90.0, position
    )
    first_pending = manager.get_pending_stop_losses()[0]

    assert not await manager.check_and_execute_stop_loss(
        "BTC-USDT-SWAP", "trend", 89.0, position
    )
    second_pending = manager.get_pending_stop_losses()[0]

    assert second_pending["failure_count"] == 2
    assert second_pending["first_attempt_at"] == first_pending["first_attempt_at"]
    assert second_pending["last_attempt_at"]
    assert manager._alert_callback.await_count == 2
    assert manager._alert_callback.await_args.kwargs["severity"] == "EMERGENCY"
    assert manager._stop_loss_events == []
    manager._save_stop_loss_audit.assert_not_called()


@pytest.mark.asyncio
async def test_pending_stop_loss_state_survives_restart_and_clears_on_success(tmp_path):
    db_path = tmp_path / "stop-loss.sqlite"
    config = {"sqlite": {"db_path": str(db_path)}, "trading": {}}
    manager = StopLossManager(config, None, None, None, Mock())
    manager._execute_stop_loss_order = AsyncMock(return_value=False)
    manager.set_alert_callback(AsyncMock())
    position = {
        "direction": "long",
        "entry_price": 100.0,
        "stop_loss": 95.0,
        "quantity": 1.0,
    }

    assert not await manager.check_and_execute_stop_loss(
        "BTC-USDT-SWAP", "trend", 90.0, position
    )

    restored = StopLossManager(config, None, None, None, Mock())
    pending = restored.get_pending_stop_losses()
    assert len(pending) == 1
    assert pending[0]["failure_count"] == 1

    restored._execute_stop_loss_order = AsyncMock(return_value=True)
    restored.set_alert_callback(AsyncMock())
    assert await restored.check_and_execute_stop_loss(
        "BTC-USDT-SWAP", "trend", 90.0, position
    )
    assert restored.get_pending_stop_losses() == []

    with sqlite3.connect(db_path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM pending_stop_losses"
        ).fetchone()[0]
    assert count == 0
