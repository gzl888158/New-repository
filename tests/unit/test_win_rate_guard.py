"""
胜率绝对值守卫（win_rate_guard）单元测试
=============================================================
覆盖：_diagnose 胜率低于阈值告警、_win_rate_guard_actions 降杠杆动作、
低/高/边界胜率、样本不足、自定义阈值、去重与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "min_win_rate": 0.35,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"win_rate_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies, equity=1000.0):
    return {"equity": equity, "contribution": {"strategies": strategies}}


def _low_wr_alerts(alerts):
    return [a for a in alerts if a["type"] == "low_win_rate"]


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_win_rate():
    snap = _Contrib(strategies={"grid": _Contrib(win_rate=0.4)}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["win_rate"] == pytest.approx(0.4)


def test_snapshot_win_rate_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    # snapshot_to_dict 对未设置的 win_rate 默认值为 0.0
    assert d["strategies"]["grid"]["win_rate"] == pytest.approx(0.0)


# ── 诊断 ─────────────────────────────────────────────────

def _run(orch, strategies):
    return orch._diagnose(_perception(strategies))


def test_diagnose_low_win_rate_triggers():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 10, "win_rate": 0.3}})
    hits = _low_wr_alerts(alerts)
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["win_rate"] == pytest.approx(0.3)


def test_diagnose_high_win_rate_no_trigger():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 10, "win_rate": 0.5}})
    assert _low_wr_alerts(alerts) == []


def test_diagnose_boundary_at_threshold_no_trigger():
    orch = _orch(min_win_rate=0.35)
    # 胜率 == 阈值不触发（严格小于）
    alerts = _run(orch, {"grid": {"total_trades": 10, "win_rate": 0.35}})
    assert _low_wr_alerts(alerts) == []


def test_diagnose_below_boundary_triggers():
    orch = _orch(min_win_rate=0.35)
    alerts = _run(orch, {"grid": {"total_trades": 10, "win_rate": 0.349}})
    assert len(_low_wr_alerts(alerts)) == 1


def test_diagnose_insufficient_trades():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 9, "win_rate": 0.1}})
    assert _low_wr_alerts(alerts) == []


def test_diagnose_custom_threshold():
    orch = _orch(min_win_rate=0.5)
    alerts = _run(orch, {"grid": {"total_trades": 10, "win_rate": 0.4}})
    assert len(_low_wr_alerts(alerts)) == 1


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "low_win_rate", "strategy": "grid", "message": "胜率过低"}]
    actions = orch._win_rate_guard_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["param"] == "leverage"
    assert a["strategy"] == "grid"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "low_win_rate", "strategy": "grid"},
        {"type": "low_win_rate", "strategy": "grid"},
    ]
    assert len(orch._win_rate_guard_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._win_rate_guard_actions([{"type": "win_rate_deteriorating"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "low_win_rate", "strategy": "grid", "message": "x"}]
    assert orch._win_rate_guard_actions(alerts) == []
    assert _low_wr_alerts(_run(orch, {"grid": {"total_trades": 10, "win_rate": 0.1}})) == []
