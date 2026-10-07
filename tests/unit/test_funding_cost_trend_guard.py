"""
资金费率趋势外推守卫（funding_cost_trend_guard）单元测试
=============================================================
覆盖：_diagnose 连续 window 周期资金费率上升告警、_funding_cost_trend_actions
收敛持仓动作、上升/平坦/下降轨迹、total_pnl<=0 与 funding<=0 不触发、去重与禁用。
"""
import os
import tempfile
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "window": 3,
        "reduce_target": 0.1,
    }
    guard.update(overrides)
    state_path = os.path.join(tempfile.gettempdir(), "test_funding_cost_trend_state.json")
    return {"agi_orchestrator": {"funding_cost_trend_guard": guard, "state_path": state_path}}


def _orch(**overrides):
    orch = QuantAGIOrchestrator(config=_config(**overrides))
    orch._strategy_funding_cost_history.clear()
    return orch


def _perception(strategies, equity=1000.0):
    return {"equity": equity, "contribution": {"strategies": strategies}}


def _funding_alerts(alerts):
    return [a for a in alerts if a["type"] == "funding_cost_rising"]


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_total_funding_cost():
    snap = _Contrib(strategies={
        "grid": _Contrib(total_funding_cost=15.0),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["total_funding_cost"] == pytest.approx(15.0)


def test_snapshot_total_funding_cost_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["total_funding_cost"] == 0.0


# ── 诊断：连续上升轨迹 ───────────────────────────────────

def _run_cycles(orch, fundings):
    """按给定资金费序列依次 _diagnose，返回最后一次的 funding_cost_rising 告警列表。"""
    last = []
    for f in fundings:
        perception = _perception({
            "grid": {"total_pnl": 100.0, "total_fees": 10.0, "total_funding_cost": f},
        })
        last = orch._diagnose(perception)
    return _funding_alerts(last)


def test_diagnose_rising_funding_cost_triggers():
    orch = _orch(window=3)
    # gross = 100 + 10 = 110；ratio = funding / 110 依次 0.1 → 0.2 → 0.3 严格上升
    hits = _run_cycles(orch, [11.0, 22.0, 33.0])
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["funding_ratio"] == pytest.approx(33.0 / 110.0)


def test_diagnose_not_trigger_until_window_filled():
    orch = _orch(window=3)
    # 前两个周期不足以填满 window，不应触发
    first = _run_cycles(orch, [11.0])
    assert first == []
    two = _run_cycles(orch, [11.0, 22.0])
    assert two == []


def test_diagnose_flat_funding_cost_no_trigger():
    orch = _orch(window=3)
    hits = _run_cycles(orch, [11.0, 11.0, 11.0])
    assert hits == []


def test_diagnose_declining_funding_cost_no_trigger():
    orch = _orch(window=3)
    hits = _run_cycles(orch, [33.0, 22.0, 11.0])
    assert hits == []


def test_diagnose_no_trigger_when_total_pnl_non_positive():
    orch = _orch(window=3)
    # total_pnl <= 0 时守卫不评估（funding_ratio 语义失效）
    last = []
    for f in [11.0, 22.0, 33.0]:
        perception = _perception({
            "grid": {"total_pnl": 0.0, "total_fees": 10.0, "total_funding_cost": f},
        })
        last = orch._diagnose(perception)
    assert _funding_alerts(last) == []


def test_diagnose_no_trigger_when_funding_zero():
    orch = _orch(window=3)
    # funding <= 0 时 ratio 记为 0，不触发
    hits = _run_cycles(orch, [0.0, 0.0, 0.0])
    assert hits == []


def test_diagnose_custom_window():
    orch = _orch(window=2)
    # window=2：前两个周期（0.1→0.2）即触发
    first = _run_cycles(orch, [11.0])
    assert first == []
    two = _run_cycles(orch, [11.0, 22.0])
    assert len(two) == 1


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_reallocate_decrease():
    orch = _orch(reduce_target=0.1)
    alerts = [{"type": "funding_cost_rising", "strategy": "grid", "message": "资金费率上升"}]
    actions = orch._funding_cost_trend_actions({}, alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "reallocate"
    assert a["action"] == "decrease"
    assert a["strategy"] == "grid"
    assert a["target_allocation"] == pytest.approx(0.1)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "funding_cost_rising", "strategy": "grid"},
        {"type": "funding_cost_rising", "strategy": "grid"},
    ]
    assert len(orch._funding_cost_trend_actions({}, alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._funding_cost_trend_actions({}, [{"type": "high_funding_cost"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "funding_cost_rising", "strategy": "grid", "message": "x"}]
    assert orch._funding_cost_trend_actions({}, alerts) == []
    # 诊断也不触发
    assert _run_cycles(orch, [11.0, 22.0, 33.0]) == []
