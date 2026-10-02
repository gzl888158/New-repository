"""
健康度骤降守卫（health_crash_guard）单元测试
===========================================
覆盖：snapshot_to_dict 暴露 delta_health、_diagnose 健康度骤降告警、
_health_crash_actions 降杠杆动作、阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "crash_threshold": 20,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"health_crash_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_delta_health():
    snap = _Contrib(strategies={
        "grid": _Contrib(delta_health=-25.0),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["delta_health"] == pytest.approx(-25.0)


def test_snapshot_delta_health_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["delta_health"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_health_crash():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 15, "delta_health": -30.0},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "health_crash"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["delta_health"] == pytest.approx(-30.0)


def test_diagnose_no_alert_when_improving():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 15, "delta_health": 5.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "health_crash"]


def test_diagnose_small_drop_no_alert():
    orch = _orch(crash_threshold=20)
    perception = _perception({
        "grid": {"total_trades": 15, "delta_health": -5.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "health_crash"]


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"total_trades": 3, "delta_health": -30.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "health_crash"]


def test_diagnose_boundary_at_threshold():
    orch = _orch(crash_threshold=20)
    perception = _perception({
        "grid": {"total_trades": 15, "delta_health": -20.0},
    })
    alerts = orch._diagnose(perception)
    assert [a for a in alerts if a["type"] == "health_crash"]  # ≤ -20 边界触发


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "health_crash", "strategy": "grid", "message": "骤降"}]
    actions = orch._health_crash_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["strategy"] == "grid"
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "health_crash", "strategy": "grid"},
        {"type": "health_crash", "strategy": "grid"},
    ]
    assert len(orch._health_crash_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._health_crash_actions([{"type": "spot_overweight"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "health_crash", "strategy": "grid", "message": "x"}]
    assert orch._health_crash_actions(alerts) == []
    perception = _perception({
        "grid": {"total_trades": 15, "delta_health": -30.0},
    })
    assert not [a for a in orch._diagnose(perception) if a["type"] == "health_crash"]
