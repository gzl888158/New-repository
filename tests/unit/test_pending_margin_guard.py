"""
挂单保证金守卫（pending_margin_guard）单元测试
===============================================
覆盖：_perceive_pending_margin 感知、_diagnose 高挂单保证金告警、
_pending_margin_actions 收敛最高权重策略、阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "ratio_threshold": 0.15,
        "reduce_target": 0.15,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"pending_margin_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


class FakeAccountManager:
    def __init__(self, pending=0.0):
        self._pending = pending

    def get_pending_margin(self):
        return self._pending


# ── 感知 ─────────────────────────────────────────────────

def test_perceive_pending_margin():
    orch = _orch()
    orch.account_manager = FakeAccountManager(pending=8.5)
    assert orch._perceive_pending_margin() == {"pending": 8.5}


def test_perceive_pending_margin_no_account_manager():
    orch = _orch()
    assert orch._perceive_pending_margin() == {}


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_high_pending_margin():
    orch = _orch(ratio_threshold=0.15)
    perception = {"equity": 100.0, "pending_margin": {"pending": 15.0}}  # ratio 0.15 == 阈值
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "high_pending_margin"]
    assert len(hits) == 1
    assert hits[0]["pending_margin"] == pytest.approx(15.0)
    assert hits[0]["pending_margin_ratio"] == pytest.approx(0.15)


def test_diagnose_no_alert_below_threshold():
    orch = _orch(ratio_threshold=0.15)
    perception = {"equity": 100.0, "pending_margin": {"pending": 10.0}}  # ratio 0.10 < 0.15
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_pending_margin"]


def test_diagnose_skips_zero_pending():
    orch = _orch()
    perception = {"equity": 100.0, "pending_margin": {"pending": 0.0}}
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_pending_margin"]


def test_diagnose_skips_missing_equity():
    orch = _orch()
    perception = {"pending_margin": {"pending": 15.0}}  # 无 equity，无法算占比
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_pending_margin"]


# ── 动作生成 ─────────────────────────────────────────────

def test_pending_margin_actions_reduce_top_weight():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "high_pending_margin"}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._pending_margin_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"  # 最高权重
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_pending_margin_actions_skip_when_top_weight_below_target():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "high_pending_margin"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.05}}}}
    assert orch._pending_margin_actions(decision, alerts) == []


def test_pending_margin_actions_no_alert():
    orch = _orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._pending_margin_actions(decision, [{"type": "spot_overweight"}]) == []


def test_pending_margin_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "high_pending_margin"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._pending_margin_actions(decision, alerts) == []
    perception = {"equity": 100.0, "pending_margin": {"pending": 15.0}}
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "high_pending_margin"]
