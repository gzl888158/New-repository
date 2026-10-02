"""
风险调整后贡献守卫（risk_adjusted_contribution_guard）单元测试
=============================================================
覆盖：snapshot_to_dict 暴露 risk_adjusted_contribution、_diagnose 小赚大扛告警、
_risk_adjusted_contribution_actions 降杠杆动作、阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "min_trades": 10,
        "min_risk_adjusted_ratio": 1.0,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"risk_adjusted_contribution_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_risk_adjusted_contribution():
    snap = _Contrib(strategies={
        "grid": _Contrib(risk_adjusted_contribution=0.5),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["risk_adjusted_contribution"] == pytest.approx(0.5)


def test_snapshot_risk_adjusted_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["risk_adjusted_contribution"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_low_risk_adjusted_contribution():
    orch = _orch()
    perception = _perception({
        "grid": {"total_pnl": 0.5, "total_trades": 10,
                 "risk_adjusted_contribution": 0.4, "max_drawdown": 0.6},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "low_risk_adjusted_contribution"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["risk_adjusted_contribution"] == pytest.approx(0.4)


def test_diagnose_no_alert_when_good_ratio():
    orch = _orch()
    perception = _perception({
        "grid": {"total_pnl": 5.0, "total_trades": 10,
                 "risk_adjusted_contribution": 3.0, "max_drawdown": 0.3},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "low_risk_adjusted_contribution"]


def test_diagnose_no_alert_when_losing():
    orch = _orch()
    perception = _perception({
        "grid": {"total_pnl": -2.0, "total_trades": 10,
                 "risk_adjusted_contribution": -0.5, "max_drawdown": 0.4},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "low_risk_adjusted_contribution"]


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"total_pnl": 0.5, "total_trades": 5,
                 "risk_adjusted_contribution": 0.4, "max_drawdown": 0.6},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "low_risk_adjusted_contribution"]


def test_diagnose_zero_ratio_no_alert():
    # rac == 0（字段缺失/未计算）不应误触发（total_pnl>0 时 rac 理论恒>0）
    orch = _orch()
    perception = _perception({
        "grid": {"total_pnl": 0.5, "total_trades": 10, "risk_adjusted_contribution": 0.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "low_risk_adjusted_contribution"]


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "low_risk_adjusted_contribution", "strategy": "grid", "message": "小赚大扛"}]
    actions = orch._risk_adjusted_contribution_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["strategy"] == "grid"
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "low_risk_adjusted_contribution", "strategy": "grid"},
        {"type": "low_risk_adjusted_contribution", "strategy": "grid"},
    ]
    assert len(orch._risk_adjusted_contribution_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._risk_adjusted_contribution_actions([{"type": "spot_overweight"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "low_risk_adjusted_contribution", "strategy": "grid", "message": "x"}]
    assert orch._risk_adjusted_contribution_actions(alerts) == []
    perception = _perception({
        "grid": {"total_pnl": 0.5, "total_trades": 10,
                 "risk_adjusted_contribution": 0.4, "max_drawdown": 0.6},
    })
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "low_risk_adjusted_contribution"]
