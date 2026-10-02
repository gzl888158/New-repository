"""
盈亏比绝对阈值守卫（profit_factor_guard）单元测试
=============================================================
覆盖：_diagnose 低盈亏比告警、_profit_factor_guard_actions 降杠杆动作、
阈值边界（min_profit_factor）、样本不足、去重与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "min_profit_factor": 1.0,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"profit_factor_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_profit_factor():
    snap = _Contrib(strategies={
        "grid": _Contrib(profit_factor=0.5),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["profit_factor"] == pytest.approx(0.5)


def test_snapshot_profit_factor_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["profit_factor"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_low_profit_factor():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "profit_factor": 0.5},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "low_profit_factor"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["profit_factor"] == pytest.approx(0.5)


def test_diagnose_no_alert_when_good_profit_factor():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "profit_factor": 1.5},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "low_profit_factor"]


def test_diagnose_no_alert_at_threshold_boundary():
    # profit_factor == min_profit_factor（1.0）不触发（严格小于才触发）
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "profit_factor": 1.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "low_profit_factor"]


def test_diagnose_no_alert_when_zero_missing():
    # profit_factor == 0（字段缺失/未计算）不应误触发（0 < pf 条件排除）
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "profit_factor": 0.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "low_profit_factor"]


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"total_trades": 5, "profit_factor": 0.5},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "low_profit_factor"]


def test_diagnose_custom_min_profit_factor():
    # min_profit_factor 提高后，仅在更低盈亏比时触发
    orch = _orch(min_profit_factor=0.5)
    perception = _perception({
        "grid": {"total_trades": 10, "profit_factor": 0.3},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "low_profit_factor"]
    assert len(hits) == 1


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "low_profit_factor", "strategy": "grid", "message": "低盈亏比"}]
    actions = orch._profit_factor_guard_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["strategy"] == "grid"
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "low_profit_factor", "strategy": "grid"},
        {"type": "low_profit_factor", "strategy": "grid"},
    ]
    assert len(orch._profit_factor_guard_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._profit_factor_guard_actions([{"type": "profit_factor_deteriorating"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "low_profit_factor", "strategy": "grid", "message": "x"}]
    assert orch._profit_factor_guard_actions(alerts) == []
    perception = _perception({
        "grid": {"total_trades": 10, "profit_factor": 0.5},
    })
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "low_profit_factor"]
