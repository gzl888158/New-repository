"""
组合压力场景多样化（severe 尾部场景）单元测试
===========================================
覆盖：严重尾部场景分级告警（critical）、仅严重失败、动作响应严重告警。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _config(**overrides):
    guard = {
        "enabled": True,
        "stress_scenario_pct": 0.20,
        "stress_loss_budget": 0.50,
        "reduce_target": 0.15,
        "severe_scenario_pct": 0.40,
        "severe_loss_budget": 0.80,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"portfolio_stress_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def test_diagnose_both_scenarios_fail():
    orch = _orch()
    # gross=300, equity=100: base 60/100=0.6>=0.5；severe 120/100=1.2>=0.8 → 双触发
    perception = {"net_exposure": {"long": 150.0, "short": 150.0}, "equity": 100.0}
    alerts = orch._diagnose(perception)
    assert [a for a in alerts if a["type"] == "stress_test_failed"]
    severe = [a for a in alerts if a["type"] == "severe_stress_test_failed"]
    assert len(severe) == 1
    assert severe[0]["level"] == "critical"


def test_diagnose_only_severe_fails():
    orch = _orch()
    # gross=220, equity=100: base 44/100=0.44<0.5（不触发）；severe 88/100=0.88>=0.8（触发）
    perception = {"net_exposure": {"long": 110.0, "short": 110.0}, "equity": 100.0}
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "stress_test_failed"]
    assert [a for a in alerts if a["type"] == "severe_stress_test_failed"]


def test_stress_actions_respond_to_severe():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "severe_stress_test_failed"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    actions = orch._stress_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["action"] == "decrease"
