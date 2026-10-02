"""
最大回撤绝对值守卫（max_drawdown_guard）单元测试
=============================================================
覆盖：_diagnose 回撤超阈值告警、_max_drawdown_guard_actions 降杠杆动作、
高/低/边界回撤、样本不足、自定义阈值、去重与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "max_drawdown_threshold": 0.2,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"max_drawdown_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies, equity=1000.0):
    return {"equity": equity, "contribution": {"strategies": strategies}}


def _high_mdd_alerts(alerts):
    return [a for a in alerts if a["type"] == "high_max_drawdown"]


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_max_drawdown():
    snap = _Contrib(strategies={"grid": _Contrib(max_drawdown=0.15)}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["max_drawdown"] == pytest.approx(0.15)


def test_snapshot_max_drawdown_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["max_drawdown"] == pytest.approx(0.0)


# ── 诊断 ─────────────────────────────────────────────────

def _run(orch, strategies):
    return orch._diagnose(_perception(strategies))


def test_diagnose_high_drawdown_triggers():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 10, "max_drawdown": 0.25}})
    hits = _high_mdd_alerts(alerts)
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["max_drawdown"] == pytest.approx(0.25)


def test_diagnose_low_drawdown_no_trigger():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 10, "max_drawdown": 0.1}})
    assert _high_mdd_alerts(alerts) == []


def test_diagnose_boundary_at_threshold_triggers():
    orch = _orch(max_drawdown_threshold=0.2)
    # 回撤 == 阈值触发（>=）
    alerts = _run(orch, {"grid": {"total_trades": 10, "max_drawdown": 0.2}})
    assert len(_high_mdd_alerts(alerts)) == 1


def test_diagnose_below_boundary_no_trigger():
    orch = _orch(max_drawdown_threshold=0.2)
    alerts = _run(orch, {"grid": {"total_trades": 10, "max_drawdown": 0.199}})
    assert _high_mdd_alerts(alerts) == []


def test_diagnose_insufficient_trades():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 9, "max_drawdown": 0.5}})
    assert _high_mdd_alerts(alerts) == []


def test_diagnose_custom_threshold():
    orch = _orch(max_drawdown_threshold=0.1)
    alerts = _run(orch, {"grid": {"total_trades": 10, "max_drawdown": 0.15}})
    assert len(_high_mdd_alerts(alerts)) == 1


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "high_max_drawdown", "strategy": "grid", "message": "回撤过深"}]
    actions = orch._max_drawdown_guard_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["param"] == "leverage"
    assert a["strategy"] == "grid"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "high_max_drawdown", "strategy": "grid"},
        {"type": "high_max_drawdown", "strategy": "grid"},
    ]
    assert len(orch._max_drawdown_guard_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._max_drawdown_guard_actions([{"type": "max_drawdown_deteriorating"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "high_max_drawdown", "strategy": "grid", "message": "x"}]
    assert orch._max_drawdown_guard_actions(alerts) == []
    assert _high_mdd_alerts(_run(orch, {"grid": {"total_trades": 10, "max_drawdown": 0.5}})) == []
