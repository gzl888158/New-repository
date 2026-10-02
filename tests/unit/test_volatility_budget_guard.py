"""
组合波动率预算守卫（volatility_budget_guard）单元测试
====================================================
覆盖：_diagnose 波动率超预算告警、_volatility_budget_actions 降杠杆动作、
阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _config(**overrides):
    guard = {
        "enabled": True,
        "volatility_budget": 50.0,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"volatility_budget_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


def test_diagnose_volatility_over_budget():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 15, "volatility": 80.0},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "volatility_over_budget"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["volatility"] == pytest.approx(80.0)


def test_diagnose_no_alert_within_budget():
    orch = _orch(volatility_budget=50.0)
    perception = _perception({
        "grid": {"total_trades": 15, "volatility": 30.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "volatility_over_budget"]


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"total_trades": 3, "volatility": 80.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "volatility_over_budget"]


def test_diagnose_boundary_at_budget_no_alert():
    orch = _orch(volatility_budget=50.0)
    perception = _perception({
        "grid": {"total_trades": 15, "volatility": 50.0},  # == 预算，不触发（> 才触发）
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "volatility_over_budget"]


def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "volatility_over_budget", "strategy": "grid", "message": "波动过大"}]
    actions = orch._volatility_budget_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["strategy"] == "grid"
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "volatility_over_budget", "strategy": "grid"},
        {"type": "volatility_over_budget", "strategy": "grid"},
    ]
    assert len(orch._volatility_budget_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._volatility_budget_actions([{"type": "spot_overweight"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "volatility_over_budget", "strategy": "grid", "message": "x"}]
    assert orch._volatility_budget_actions(alerts) == []
    perception = _perception({"grid": {"total_trades": 15, "volatility": 80.0}})
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "volatility_over_budget"]
