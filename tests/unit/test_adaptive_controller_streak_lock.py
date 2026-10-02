from datetime import datetime, timedelta

import pytest

from risk.adaptive_controller import AdaptiveController


def _controller_for_budget_checks() -> AdaptiveController:
    controller = AdaptiveController.__new__(AdaptiveController)
    controller._risk_budget_enabled = True
    controller._streak_lock_active = True
    controller._streak_lock_started_at = datetime.now() - timedelta(hours=3)
    controller._streak_probe_after_seconds = 7200
    controller._streak_probe_risk_pct = 0.001
    controller._consecutive_loss_count = 5
    controller._max_per_trade_risk_pct = 0.008
    controller._daily_risk_budget_pct = 0.03
    controller._strategy_risk_limits = {"grid": 0.5}
    controller._daily_risk_consumed = {}
    controller._symbol_risk_exposure = {}
    controller._max_symbol_risk_pct = 0.008
    controller._hourly_pnl = 0.0
    controller._hourly_max_loss_pct = 0.015
    return controller


def test_loss_streak_budget_reduction_is_not_normalized_away():
    controller = AdaptiveController.__new__(AdaptiveController)
    controller._streak_reduce_pct = 0.5
    controller._strategy_risk_limits = {"grid": 0.6, "trend": 0.4}
    controller._risk_budget = dict(controller._strategy_risk_limits)
    controller._consecutive_loss_count = 5
    controller._risk_budget_log = []

    controller._apply_loss_streak_lock()

    assert controller._risk_budget == {"grid": 0.3, "trend": 0.2}
    assert sum(controller._risk_budget.values()) == pytest.approx(0.5)


def test_active_loss_streak_lock_allows_only_limited_probe_after_wait():
    controller = _controller_for_budget_checks()

    allowed, reason = controller.check_trade_risk_budget(
        "grid", "BTC-USDT-SWAP", trade_risk_usdt=0.5, equity=1000.0
    )

    assert allowed is True
    assert reason == "loss_streak_probe_approved"


def test_active_loss_streak_lock_rejects_probe_over_limit():
    controller = _controller_for_budget_checks()

    allowed, reason = controller.check_trade_risk_budget(
        "grid", "BTC-USDT-SWAP", trade_risk_usdt=1.1, equity=1000.0
    )

    assert allowed is False
    assert reason.startswith("loss_streak_probe_limit")


def test_active_loss_streak_lock_restores_after_restart(tmp_path):
    state_path = tmp_path / "risk_budget_state.json"
    controller = AdaptiveController.__new__(AdaptiveController)
    controller._risk_budget_state_path = str(state_path)
    controller._risk_budget_enabled = True
    controller._capital_utilization = {"total_equity": 1000.0}
    controller._daily_risk_budget_pct = 0.03
    controller._max_per_trade_risk_pct = 0.008
    controller._risk_budget = {"grid": 0.3}
    controller._strategy_risk_limits = {"grid": 0.6}
    controller._daily_risk_consumed = {}
    controller._streak_lock_active = True
    controller._streak_lock_started_at = datetime.now() - timedelta(minutes=10)
    controller._streak_probe_after_seconds = 7200
    controller._streak_probe_risk_pct = 0.001
    controller._consecutive_loss_count = 5
    controller._consecutive_win_count = 0
    controller._hourly_pnl = -10.0
    controller._symbol_risk_exposure = {}

    controller._persist_risk_budget_state()

    restored = AdaptiveController.__new__(AdaptiveController)
    restored._risk_budget_state_path = str(state_path)
    restored._risk_budget = {"grid": 0.6}
    restored._strategy_risk_limits = {"grid": 0.6}
    restored._streak_lock_active = False
    restored._streak_lock_started_at = None
    restored._consecutive_loss_count = 0
    restored._consecutive_win_count = 0

    restored._restore_risk_budget_state()

    assert restored._streak_lock_active is True
    assert restored._consecutive_loss_count == 5
    assert restored._risk_budget["grid"] == pytest.approx(0.3)
    assert restored._streak_lock_started_at is not None


def test_corrupt_streak_state_fails_closed(tmp_path):
    state_path = tmp_path / "risk_budget_state.json"
    state_path.write_text("{invalid", encoding="utf-8")
    controller = AdaptiveController.__new__(AdaptiveController)
    controller._risk_budget_state_path = str(state_path)
    controller._risk_budget = {"grid": 0.6}
    controller._strategy_risk_limits = {"grid": 0.6}
    controller._streak_lock_active = False
    controller._streak_lock_started_at = None
    controller._consecutive_loss_count = 0
    controller._consecutive_win_count = 0
    controller._max_consecutive_losses = 5
    controller._streak_reduce_pct = 0.5

    controller._restore_risk_budget_state()

    assert controller._streak_lock_active is True
    assert controller._consecutive_loss_count == 5
    assert controller._risk_budget["grid"] == pytest.approx(0.3)