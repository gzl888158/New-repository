"""
回撤持续时间绝对阈值守卫（drawdown_duration_guard）单元测试
=============================================================
覆盖：_diagnose 回撤持续过久告警、_drawdown_duration_guard_actions 降杠杆动作、
阈值边界（max_drawdown_hours）、样本不足、去重与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "max_drawdown_hours": 168.0,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"drawdown_duration_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_max_drawdown_duration_hours():
    snap = _Contrib(strategies={
        "grid": _Contrib(max_drawdown_duration_hours=200.0),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["max_drawdown_duration_hours"] == pytest.approx(200.0)


def test_snapshot_drawdown_duration_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["max_drawdown_duration_hours"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_long_drawdown_duration():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "max_drawdown_duration_hours": 200.0},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "long_drawdown_duration"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["max_drawdown_duration_hours"] == pytest.approx(200.0)


def test_diagnose_no_alert_when_short_duration():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "max_drawdown_duration_hours": 50.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "long_drawdown_duration"]


def test_diagnose_at_threshold_boundary():
    # ddh == max_drawdown_hours（168.0）触发（>= 条件）
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "max_drawdown_duration_hours": 168.0},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "long_drawdown_duration"]
    assert len(hits) == 1


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"total_trades": 5, "max_drawdown_duration_hours": 200.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "long_drawdown_duration"]


def test_diagnose_custom_max_hours():
    # max_drawdown_hours 降低后，仅在更短回撤持续时触发
    orch = _orch(max_drawdown_hours=72.0)
    perception = _perception({
        "grid": {"total_trades": 10, "max_drawdown_duration_hours": 100.0},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "long_drawdown_duration"]
    assert len(hits) == 1


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "long_drawdown_duration", "strategy": "grid", "message": "回撤过久"}]
    actions = orch._drawdown_duration_guard_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["strategy"] == "grid"
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "long_drawdown_duration", "strategy": "grid"},
        {"type": "long_drawdown_duration", "strategy": "grid"},
    ]
    assert len(orch._drawdown_duration_guard_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._drawdown_duration_guard_actions(
        [{"type": "drawdown_duration_deteriorating"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "long_drawdown_duration", "strategy": "grid", "message": "x"}]
    assert orch._drawdown_duration_guard_actions(alerts) == []
    perception = _perception({
        "grid": {"total_trades": 10, "max_drawdown_duration_hours": 200.0},
    })
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "long_drawdown_duration"]
