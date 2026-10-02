"""
组合多空净敞口监控（net_exposure_guard）单元测试
===============================================
覆盖：_perceive_net_exposure 感知、_diagnose 方向性失衡告警、
_net_exposure_actions 收敛最高权重策略、阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "imbalance_ratio_threshold": 0.6,
        "min_gross_exposure": 1.0,
        "reduce_target": 0.15,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"net_exposure_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


class FakeAccountManager:
    def __init__(self, long=0.0, short=0.0):
        self._long = long
        self._short = short

    def get_long_exposure(self):
        return self._long

    def get_short_exposure(self):
        return self._short


# ── 感知 ─────────────────────────────────────────────────

def test_perceive_net_exposure():
    orch = _orch()
    orch.account_manager = FakeAccountManager(long=800.0, short=200.0)
    assert orch._perceive_net_exposure() == {"long": 800.0, "short": 200.0}


def test_perceive_net_exposure_no_account_manager():
    orch = _orch()
    assert orch._perceive_net_exposure() == {}


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_directional_imbalance():
    orch = _orch()
    perception = {"net_exposure": {"long": 800.0, "short": 200.0}}  # imbalance 0.6
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "directional_imbalance"]
    assert len(hits) == 1
    assert hits[0]["imbalance_ratio"] == pytest.approx(0.6)


def test_diagnose_no_imbalance_when_balanced():
    orch = _orch()
    perception = {"net_exposure": {"long": 500.0, "short": 500.0}}
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "directional_imbalance"]


def test_diagnose_no_imbalance_below_threshold():
    orch = _orch(imbalance_ratio_threshold=0.6)
    perception = {"net_exposure": {"long": 600.0, "short": 400.0}}  # imbalance 0.2
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "directional_imbalance"]


def test_diagnose_skips_tiny_gross_exposure():
    orch = _orch(min_gross_exposure=1.0)
    perception = {"net_exposure": {"long": 0.5, "short": 0.0}}  # gross 0.5 < 1.0
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "directional_imbalance"]


# ── 动作生成 ─────────────────────────────────────────────

def test_net_exposure_actions_reduce_top_weight():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "directional_imbalance"}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._net_exposure_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"  # 最高权重
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_net_exposure_actions_skip_when_top_weight_below_target():
    orch = _orch(reduce_target=0.15)
    alerts = [{"type": "directional_imbalance"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.05}}}}
    assert orch._net_exposure_actions(decision, alerts) == []


def test_net_exposure_actions_no_alert():
    orch = _orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._net_exposure_actions(decision, [{"type": "spot_overweight"}]) == []


def test_net_exposure_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "directional_imbalance"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._net_exposure_actions(decision, alerts) == []
    perception = {"net_exposure": {"long": 800.0, "short": 200.0}}
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "directional_imbalance"]
