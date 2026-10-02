"""
夏普比率绝对阈值守卫（sharpe_ratio_guard）单元测试
=============================================================
覆盖：_diagnose 负夏普告警、_sharpe_ratio_actions 降杠杆动作、
阈值边界（min_sharpe_ratio）、样本不足、去重与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "min_sharpe_ratio": 0.0,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"sharpe_ratio_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_sharpe_ratio():
    snap = _Contrib(strategies={
        "grid": _Contrib(sharpe_ratio=-0.5),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["sharpe_ratio"] == pytest.approx(-0.5)


def test_snapshot_sharpe_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["sharpe_ratio"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_negative_sharpe():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "sharpe_ratio": -0.5},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "negative_sharpe"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["sharpe_ratio"] == pytest.approx(-0.5)


def test_diagnose_no_alert_when_positive_sharpe():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "sharpe_ratio": 1.5},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "negative_sharpe"]


def test_diagnose_no_alert_at_threshold_boundary():
    # sharpe == min_sharpe_ratio（0.0）不触发（严格小于才触发）
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "sharpe_ratio": 0.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "negative_sharpe"]


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"total_trades": 5, "sharpe_ratio": -0.5},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "negative_sharpe"]


def test_diagnose_custom_min_ratio():
    # min_sharpe_ratio 提高后，仅在更低负夏普时触发
    orch = _orch(min_sharpe_ratio=-1.0)
    perception = _perception({
        "grid": {"total_trades": 10, "sharpe_ratio": -1.5},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "negative_sharpe"]
    assert len(hits) == 1


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "negative_sharpe", "strategy": "grid", "message": "负夏普"}]
    actions = orch._sharpe_ratio_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["strategy"] == "grid"
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "negative_sharpe", "strategy": "grid"},
        {"type": "negative_sharpe", "strategy": "grid"},
    ]
    assert len(orch._sharpe_ratio_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._sharpe_ratio_actions([{"type": "sharpe_deteriorating"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "negative_sharpe", "strategy": "grid", "message": "x"}]
    assert orch._sharpe_ratio_actions(alerts) == []
    perception = _perception({
        "grid": {"total_trades": 10, "sharpe_ratio": -0.5},
    })
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "negative_sharpe"]
