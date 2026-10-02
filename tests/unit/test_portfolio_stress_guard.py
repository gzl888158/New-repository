"""
组合压力测试守卫（portfolio_stress_guard）单元测试
=================================================
覆盖：_diagnose 尾部压力测试告警、_stress_actions 收敛最高权重策略、阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _config(**overrides):
    guard = {
        "enabled": True,
        "stress_scenario_pct": 0.20,
        "stress_loss_budget": 0.50,
        "reduce_target": 0.15,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"portfolio_stress_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def test_diagnose_stress_test_failed():
    orch = _orch()
    # gross=1000, stress_loss=200, 200/100=2.0 >= 0.5 → 失败
    perception = {"net_exposure": {"long": 500.0, "short": 500.0}, "equity": 100.0}
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "stress_test_failed"]
    assert len(hits) == 1
    assert hits[0]["stress_loss"] == pytest.approx(200.0)


def test_diagnose_no_stress_within_budget():
    orch = _orch()
    # gross=20, stress_loss=4, 4/100=0.04 < 0.5
    perception = {"net_exposure": {"long": 10.0, "short": 10.0}, "equity": 100.0}
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "stress_test_failed"]


def test_diagnose_skips_zero_gross():
    orch = _orch()
    perception = {"net_exposure": {"long": 0.0, "short": 0.0}, "equity": 100.0}
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "stress_test_failed"]


def test_stress_actions_reduce_top_weight():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "stress_test_failed"}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._stress_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_stress_actions_skip_when_top_weight_below_target():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "stress_test_failed"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.05}}}}
    assert orch._stress_actions(decision, alerts) == []


def test_stress_actions_no_alert():
    orch = _orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._stress_actions(decision, [{"type": "spot_overweight"}]) == []


def test_stress_guard_disabled():
    orch = _orch(enabled=False)
    perception = {"net_exposure": {"long": 500.0, "short": 500.0}, "equity": 100.0}
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "stress_test_failed"]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._stress_actions(decision, [{"type": "stress_test_failed"}]) == []
