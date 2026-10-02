"""
AGI 决策震荡抑制（decision_oscillation_guard）单元测试
====================================================
覆盖：_reconcile_actions 记录防守减仓周期、_offensive_allocation_actions
在冷却期内抑制「减了又加」的进攻、禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _config(**overrides):
    guard = {"enabled": True, "cooldown_cycles": 3}
    guard.update(overrides)
    return {"agi_orchestrator": {"decision_oscillation_guard": guard}}


def _offensive_cfg(**overrides):
    return {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "decision_oscillation_guard": dict({"enabled": True, "cooldown_cycles": 3}, **overrides),
    }}


def test_reconcile_records_defensive_reduce():
    orch = QuantAGIOrchestrator(config=_config())
    orch._reconciliation_enabled = True
    orch._cycle_count = 10
    orch._reconcile_actions([
        {"type": "reallocate", "strategy": "grid", "action": "decrease",
         "target_allocation": 0.1},
    ])
    assert orch._last_defensive_reduce_cycle["grid"] == 10


def test_oscillation_blocks_offensive_increase_within_cooldown():
    orch = QuantAGIOrchestrator(config=_offensive_cfg())
    # 模拟上一周期「grid」刚被防守减仓
    orch._last_defensive_reduce_cycle["grid"] = orch._cycle_count
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_oscillation_allows_offensive_after_cooldown_expiry():
    orch = QuantAGIOrchestrator(config=_offensive_cfg())
    # 减仓发生在 cooldown_cycles 之前 → 冷却已过期，允许进攻
    orch._cycle_count = 10
    orch._last_defensive_reduce_cycle["grid"] = 10 - 3  # 恰好等于 cooldown，已过期
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert any(a["strategy"] == "grid" and a["action"] == "increase" for a in actions)


def test_oscillation_not_block_other_strategy():
    orch = QuantAGIOrchestrator(config=_offensive_cfg())
    orch._last_defensive_reduce_cycle["grid"] = orch._cycle_count
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被震荡抑制，sync 仍进攻
    assert all(a["strategy"] != "grid" for a in actions)
    assert any(a["strategy"] == "sync" and a["action"] == "increase" for a in actions)


def test_oscillation_disabled_allows_increase():
    orch = QuantAGIOrchestrator(config=_offensive_cfg(enabled=False))
    orch._last_defensive_reduce_cycle["grid"] = orch._cycle_count
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert any(a["strategy"] == "grid" and a["action"] == "increase" for a in actions)
