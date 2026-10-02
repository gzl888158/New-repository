"""
每小时盈利效率守卫（hourly_pnl_guard）单元测试
=============================================
覆盖：snapshot_to_dict 暴露 active_hours/pnl_per_hour、_diagnose 每小时持续失血告警、
_hourly_pnl_actions 收敛动作、禁用与阈值边界。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "min_active_hours": 24,
        "min_trades": 3,
        "hourly_loss_threshold": 0.5,
        "reduce_target": 0.1,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"hourly_pnl_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_hourly_fields():
    snap = _Contrib(strategies={
        "grid": _Contrib(total_pnl=1.0, active_hours=10.0, pnl_per_hour=-2.5),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["active_hours"] == pytest.approx(10.0)
    assert d["strategies"]["grid"]["pnl_per_hour"] == pytest.approx(-2.5)


def test_snapshot_hourly_fields_default_to_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["active_hours"] == 0.0
    assert d["strategies"]["grid"]["pnl_per_hour"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_hourly_pnl_negative():
    orch = _orch()
    perception = _perception({
        "grid": {"active_hours": 48.0, "total_trades": 10, "pnl_per_hour": -1.2},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "hourly_pnl_negative"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["pnl_per_hour"] == pytest.approx(-1.2)


def test_diagnose_no_alert_when_positive():
    orch = _orch()
    perception = _perception({
        "grid": {"active_hours": 48.0, "total_trades": 10, "pnl_per_hour": 0.8},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "hourly_pnl_negative"]


def test_diagnose_insufficient_active_hours():
    orch = _orch(min_active_hours=24)
    perception = _perception({
        "grid": {"active_hours": 5.0, "total_trades": 10, "pnl_per_hour": -1.2},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "hourly_pnl_negative"]


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=3)
    perception = _perception({
        "grid": {"active_hours": 48.0, "total_trades": 1, "pnl_per_hour": -1.2},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "hourly_pnl_negative"]


def test_diagnose_below_threshold_magnitude_no_alert():
    # pnl_per_hour = -0.3，|−0.3| < 0.5 → 不触发
    orch = _orch(hourly_loss_threshold=0.5)
    perception = _perception({
        "grid": {"active_hours": 48.0, "total_trades": 10, "pnl_per_hour": -0.3},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "hourly_pnl_negative"]


# ── 动作生成 ─────────────────────────────────────────────

def test_hourly_pnl_actions_generate_decrease():
    orch = _orch(reduce_target=0.1)
    alerts = [{"type": "hourly_pnl_negative", "strategy": "grid", "message": "持续失血"}]
    actions = orch._hourly_pnl_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "reallocate"
    assert a["strategy"] == "grid"
    assert a["action"] == "decrease"
    assert a["target_allocation"] == pytest.approx(0.1)


def test_hourly_pnl_actions_no_alert():
    orch = _orch()
    assert orch._hourly_pnl_actions([{"type": "spot_overweight"}]) == []


def test_hourly_pnl_actions_skip_missing_strategy():
    orch = _orch()
    actions = orch._hourly_pnl_actions([{"type": "hourly_pnl_negative", "strategy": ""}])
    assert actions == []


def test_hourly_pnl_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "hourly_pnl_negative", "strategy": "grid", "message": "x"}]
    assert orch._hourly_pnl_actions(alerts) == []
    perception = _perception({
        "grid": {"active_hours": 48.0, "total_trades": 10, "pnl_per_hour": -1.2},
    })
    assert not [a for a in orch._diagnose(perception) if a["type"] == "hourly_pnl_negative"]
