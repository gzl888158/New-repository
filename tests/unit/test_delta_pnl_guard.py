"""
单周期亏损急跌守卫（delta_pnl_guard）单元测试
=============================================================
覆盖：_diagnose 单周期急跌告警、_delta_pnl_guard_actions 降杠杆动作、
阈值边界（loss_threshold）、样本不足、去重与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "loss_threshold": 0.02,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"delta_pnl_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies, equity=1000.0):
    return {"equity": equity, "contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_delta_pnl():
    snap = _Contrib(strategies={
        "grid": _Contrib(delta_pnl=-30.0),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["delta_pnl"] == pytest.approx(-30.0)


def test_snapshot_delta_pnl_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["delta_pnl"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_delta_pnl_spike():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "delta_pnl": -30.0},
    }, equity=1000.0)
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "delta_pnl_spike"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["delta_pnl"] == pytest.approx(-30.0)


def test_diagnose_no_alert_when_small_loss():
    # -5/1000 = 0.5% < 2% 阈值，不触发
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "delta_pnl": -5.0},
    }, equity=1000.0)
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "delta_pnl_spike"]


def test_diagnose_no_alert_when_positive():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "delta_pnl": 10.0},
    }, equity=1000.0)
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "delta_pnl_spike"]


def test_diagnose_no_alert_when_zero_equity():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "delta_pnl": -30.0},
    }, equity=0.0)
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "delta_pnl_spike"]


def test_diagnose_at_threshold_boundary():
    # -20/1000 = 2% == loss_threshold，触发（>= 条件）
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "delta_pnl": -20.0},
    }, equity=1000.0)
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "delta_pnl_spike"]
    assert len(hits) == 1


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"total_trades": 5, "delta_pnl": -30.0},
    }, equity=1000.0)
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "delta_pnl_spike"]


def test_diagnose_custom_loss_threshold():
    # loss_threshold 提高到 5%，-30/1000=3% < 5% 不触发
    orch = _orch(loss_threshold=0.05)
    perception = _perception({
        "grid": {"total_trades": 10, "delta_pnl": -30.0},
    }, equity=1000.0)
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "delta_pnl_spike"]


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "delta_pnl_spike", "strategy": "grid", "message": "急跌"}]
    actions = orch._delta_pnl_guard_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["strategy"] == "grid"
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "delta_pnl_spike", "strategy": "grid"},
        {"type": "delta_pnl_spike", "strategy": "grid"},
    ]
    assert len(orch._delta_pnl_guard_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._delta_pnl_guard_actions([{"type": "delta_pnl_deteriorating"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "delta_pnl_spike", "strategy": "grid", "message": "x"}]
    assert orch._delta_pnl_guard_actions(alerts) == []
    perception = _perception({
        "grid": {"total_trades": 10, "delta_pnl": -30.0},
    }, equity=1000.0)
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "delta_pnl_spike"]
