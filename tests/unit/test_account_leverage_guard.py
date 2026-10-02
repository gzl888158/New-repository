"""
账户级杠杆守卫（account_leverage_guard）单元测试
=================================================
覆盖：_perceive_account_leverage 感知、_diagnose 高杠杆告警、
_account_leverage_actions 收敛最高权重策略、阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "soft_threshold_ratio": 0.8,
        "reduce_target": 0.15,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"account_leverage_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


class FakeAccountManager:
    def __init__(self, current=0.0, max_leverage=0.0):
        self._current = current
        self._max = max_leverage

    def get_current_leverage(self):
        return self._current

    def get_account_summary(self):
        return {"max_leverage": self._max}


# ── 感知 ─────────────────────────────────────────────────

def test_perceive_account_leverage():
    orch = _orch()
    orch.account_manager = FakeAccountManager(current=4.0, max_leverage=5.0)
    assert orch._perceive_account_leverage() == {"current": 4.0, "max": 5.0}


def test_perceive_account_leverage_no_account_manager():
    orch = _orch()
    assert orch._perceive_account_leverage() == {}


def test_perceive_account_leverage_no_summary_max_zero():
    """无 get_account_summary（旧 account_manager）→ current 仍可取，max 回退 0。"""
    class NoSummaryAM(FakeAccountManager):
        def get_account_summary(self):
            return None

    orch = _orch()
    orch.account_manager = NoSummaryAM(current=4.0)
    assert orch._perceive_account_leverage() == {"current": 4.0, "max": 0.0}


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_high_account_leverage():
    orch = _orch(soft_threshold_ratio=0.8)
    perception = {"account_leverage": {"current": 4.0, "max": 5.0}}  # ratio 0.8 == 阈值
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "high_account_leverage"]
    assert len(hits) == 1
    assert hits[0]["current_leverage"] == pytest.approx(4.0)
    assert hits[0]["leverage_ratio"] == pytest.approx(0.8)


def test_diagnose_no_alert_below_threshold():
    orch = _orch(soft_threshold_ratio=0.8)
    perception = {"account_leverage": {"current": 3.0, "max": 5.0}}  # ratio 0.6 < 0.8
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_account_leverage"]


def test_diagnose_skips_zero_max():
    orch = _orch()
    perception = {"account_leverage": {"current": 4.0, "max": 0.0}}
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_account_leverage"]


def test_diagnose_skips_zero_current():
    orch = _orch()
    perception = {"account_leverage": {"current": 0.0, "max": 5.0}}
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_account_leverage"]


# ── 动作生成 ─────────────────────────────────────────────

def test_account_leverage_actions_reduce_top_weight():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "high_account_leverage"}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._account_leverage_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"  # 最高权重
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_account_leverage_actions_skip_when_top_weight_below_target():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "high_account_leverage"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.05}}}}
    assert orch._account_leverage_actions(decision, alerts) == []


def test_account_leverage_actions_no_alert():
    orch = _orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._account_leverage_actions(decision, [{"type": "spot_overweight"}]) == []


def test_account_leverage_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "high_account_leverage"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._account_leverage_actions(decision, alerts) == []
    perception = {"account_leverage": {"current": 4.0, "max": 5.0}}
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "high_account_leverage"]
