"""ParameterRollbackGuard 参数变更验证 + 自动回滚守卫单元测试。"""
import pytest

from core.parameter_rollback_guard import ParameterRollbackGuard, RollbackDecision


def _guard(**kwargs) -> ParameterRollbackGuard:
    # 默认窗口 0 秒、最小样本 0 笔，便于在测试中立即得出明确决策
    defaults = {
        "validation_window_seconds": 0.0,
        "pnl_drop_threshold": 0.10,
        "win_rate_drop_threshold": 0.20,
        "min_trades": 0,
    }
    defaults.update(kwargs)
    return ParameterRollbackGuard(**defaults)


# ============================================================
# 生命周期
# ============================================================

def test_begin_validation_stores_pending():
    guard = _guard()
    guard.begin_validation("grid", baseline_metrics={"pnl": 0.0, "win_rate": 0.5})
    assert guard.pending_count() == 1
    assert "grid" in guard.get_pending()


def test_begin_validation_stores_rollback_target_version():
    guard = _guard()
    guard.begin_validation(
        "grid",
        baseline_metrics={"pnl": 0.0, "win_rate": 0.5},
        rollback_target_version=7,
    )
    assert guard.get_pending()["grid"]["rollback_target_version"] == 7


def test_cancel_validation():
    guard = _guard()
    guard.begin_validation("grid")
    assert guard.cancel_validation("grid") is True
    assert guard.pending_count() == 0
    assert guard.cancel_validation("grid") is False


# ============================================================
# 决策
# ============================================================

def test_check_no_pending_validation():
    guard = _guard()
    decision = guard.check("grid")
    assert decision.should_rollback is False
    assert decision.reason == "no_pending_validation"


def test_check_before_window_returns_in_progress():
    guard = _guard(validation_window_seconds=3600.0)
    guard.begin_validation("grid", baseline_metrics={"pnl": 0.0, "win_rate": 0.5})
    decision = guard.check("grid", current_metrics={"trades": 10, "pnl": -1.0, "win_rate": 0.0})
    assert decision.should_rollback is False
    assert decision.reason == "validation_in_progress"


def test_check_insufficient_sample():
    guard = _guard(min_trades=5)
    guard.begin_validation("grid", baseline_metrics={"pnl": 0.0, "win_rate": 0.5})
    decision = guard.check("grid", current_metrics={"trades": 3, "pnl": -1.0, "win_rate": 0.0})
    assert decision.should_rollback is False
    assert decision.reason == "insufficient_sample"


def test_check_pnl_drop_triggers_rollback():
    guard = _guard()
    guard.begin_validation("grid", baseline_metrics={"pnl": 10.0, "win_rate": 0.6})
    decision = guard.check("grid", current_metrics={"trades": 10, "pnl": 8.0, "win_rate": 0.6})
    assert decision.should_rollback is True
    assert "pnl_dropped_by" in decision.reason


def test_check_win_rate_drop_triggers_rollback():
    guard = _guard()
    guard.begin_validation("grid", baseline_metrics={"pnl": 10.0, "win_rate": 0.6})
    decision = guard.check("grid", current_metrics={"trades": 10, "pnl": 10.0, "win_rate": 0.3})
    assert decision.should_rollback is True
    assert "win_rate_dropped_by" in decision.reason


def test_check_no_degradation_keeps_config():
    guard = _guard()
    guard.begin_validation("grid", baseline_metrics={"pnl": 10.0, "win_rate": 0.6})
    decision = guard.check("grid", current_metrics={"trades": 10, "pnl": 11.0, "win_rate": 0.6})
    assert decision.should_rollback is False
    assert decision.reason == "no_degradation"


def test_check_consumes_pending_after_expiry():
    guard = _guard()
    guard.begin_validation("grid", baseline_metrics={"pnl": 10.0, "win_rate": 0.6})
    guard.check("grid", current_metrics={"trades": 10, "pnl": 10.0, "win_rate": 0.6})
    assert guard.pending_count() == 0


def test_check_rollback_decision_to_dict():
    guard = _guard()
    guard.begin_validation("grid", baseline_metrics={"pnl": 10.0, "win_rate": 0.6})
    decision = guard.check("grid", current_metrics={"trades": 10, "pnl": 8.0, "win_rate": 0.6})
    data = decision.to_dict()
    assert data["strategy_name"] == "grid"
    assert data["should_rollback"] is True
    assert "pnl_change_pct" in data


def test_pnl_baseline_zero_uses_absolute_change():
    guard = _guard()
    guard.begin_validation("grid", baseline_metrics={"pnl": 0.0, "win_rate": 0.5})
    decision = guard.check("grid", current_metrics={"trades": 10, "pnl": -0.15, "win_rate": 0.5})
    assert decision.should_rollback is True
    assert decision.pnl_change_pct == -0.15
