"""
单笔期望值绝对值守卫（pnl_per_trade_guard）单元测试
=============================================================
覆盖：_diagnose 单笔期望值低于阈值告警、_pnl_per_trade_guard_actions 降杠杆动作、
低/高/边界期望值、样本不足、自定义阈值、去重与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "min_pnl_per_trade": 0.0,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"pnl_per_trade_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies, equity=1000.0):
    return {"equity": equity, "contribution": {"strategies": strategies}}


def _low_ppt_alerts(alerts):
    return [a for a in alerts if a["type"] == "low_pnl_per_trade"]


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_pnl_per_trade():
    snap = _Contrib(strategies={"grid": _Contrib(pnl_per_trade=1.5)}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["pnl_per_trade"] == pytest.approx(1.5)


def test_snapshot_pnl_per_trade_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    # snapshot_to_dict 对未设置的 pnl_per_trade 默认值为 0.0
    assert d["strategies"]["grid"]["pnl_per_trade"] == pytest.approx(0.0)


# ── 诊断 ─────────────────────────────────────────────────

def _run(orch, strategies):
    return orch._diagnose(_perception(strategies))


def test_diagnose_low_pnl_per_trade_triggers():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 10, "pnl_per_trade": -0.5}})
    hits = _low_ppt_alerts(alerts)
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["pnl_per_trade"] == pytest.approx(-0.5)


def test_diagnose_positive_pnl_per_trade_no_trigger():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 10, "pnl_per_trade": 0.5}})
    assert _low_ppt_alerts(alerts) == []


def test_diagnose_boundary_at_threshold_no_trigger():
    orch = _orch(min_pnl_per_trade=0.0)
    # 单笔期望值 == 阈值不触发（严格小于）
    alerts = _run(orch, {"grid": {"total_trades": 10, "pnl_per_trade": 0.0}})
    assert _low_ppt_alerts(alerts) == []


def test_diagnose_below_boundary_triggers():
    orch = _orch(min_pnl_per_trade=0.0)
    alerts = _run(orch, {"grid": {"total_trades": 10, "pnl_per_trade": -0.001}})
    assert len(_low_ppt_alerts(alerts)) == 1


def test_diagnose_insufficient_trades():
    orch = _orch()
    alerts = _run(orch, {"grid": {"total_trades": 9, "pnl_per_trade": -1.0}})
    assert _low_ppt_alerts(alerts) == []


def test_diagnose_custom_threshold():
    orch = _orch(min_pnl_per_trade=0.5)
    alerts = _run(orch, {"grid": {"total_trades": 10, "pnl_per_trade": 0.1}})
    assert len(_low_ppt_alerts(alerts)) == 1


def test_diagnose_multiple_strategies():
    orch = _orch()
    alerts = _run(orch, {
        "grid": {"total_trades": 10, "pnl_per_trade": -0.3},
        "trend": {"total_trades": 10, "pnl_per_trade": 0.8},
        "martin": {"total_trades": 12, "pnl_per_trade": -0.1},
    })
    hits = _low_ppt_alerts(alerts)
    assert len(hits) == 2
    strategies = {h["strategy"] for h in hits}
    assert strategies == {"grid", "martin"}


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "low_pnl_per_trade", "strategy": "grid", "message": "每笔无正期望"}]
    actions = orch._pnl_per_trade_guard_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["param"] == "leverage"
    assert a["strategy"] == "grid"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "low_pnl_per_trade", "strategy": "grid"},
        {"type": "low_pnl_per_trade", "strategy": "grid"},
    ]
    assert len(orch._pnl_per_trade_guard_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._pnl_per_trade_guard_actions([{"type": "pnl_per_trade_deteriorating"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "low_pnl_per_trade", "strategy": "grid", "message": "x"}]
    assert orch._pnl_per_trade_guard_actions(alerts) == []
    assert _low_ppt_alerts(_run(orch, {"grid": {"total_trades": 10, "pnl_per_trade": -1.0}})) == []
