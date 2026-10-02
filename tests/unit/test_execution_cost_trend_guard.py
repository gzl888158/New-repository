"""
执行成本趋势外推守卫（execution_cost_trend_guard）单元测试
=============================================================
覆盖：_diagnose 连续 window 周期执行成本率上升告警、_execution_cost_trend_actions
收敛敞口动作、上升/平坦/下降轨迹、total_pnl<=0 与 exec_cost<=0 不触发、去重与禁用。
"""
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
    return {"agi_orchestrator": {"execution_cost_trend_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies, equity=1000.0):
    return {"equity": equity, "contribution": {"strategies": strategies}}


def _rising_alerts(alerts):
    return [a for a in alerts if a["type"] == "execution_cost_rising"]


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_execution_cost_fields():
    snap = _Contrib(strategies={
        "grid": _Contrib(total_slippage_cost=8.0, total_spread_cost=7.0),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["total_slippage_cost"] == pytest.approx(8.0)
    assert d["strategies"]["grid"]["total_spread_cost"] == pytest.approx(7.0)


def test_snapshot_execution_cost_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["total_slippage_cost"] == 0.0
    assert d["strategies"]["grid"]["total_spread_cost"] == 0.0


# ── 诊断：连续上升轨迹 ───────────────────────────────────

def _run_cycles(orch, slippages, spreads):
    """按给定滑点/点差序列依次 _diagnose，返回最后一次的 execution_cost_rising 告警列表。"""
    last = []
    for s, sp in zip(slippages, spreads):
        perception = _perception({
            "grid": {
                "total_pnl": 100.0,
                "total_fees": 10.0,
                "total_slippage_cost": s,
                "total_spread_cost": sp,
            },
        })
        last = orch._diagnose(perception)
    return _rising_alerts(last)


def test_diagnose_rising_execution_cost_triggers():
    orch = _orch(window=3)
    # gross = 100 + 10 = 110；ratio = (slippage+spread) / 110 依次 0.1 → 0.2 → 0.3 严格上升
    hits = _run_cycles(orch, [5.5, 11.0, 16.5], [5.5, 11.0, 16.5])
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["execution_cost_ratio"] == pytest.approx(33.0 / 110.0)


def test_diagnose_not_trigger_until_window_filled():
    orch = _orch(window=3)
    assert _run_cycles(orch, [5.5], [5.5]) == []
    assert _run_cycles(orch, [5.5, 11.0], [5.5, 11.0]) == []


def test_diagnose_flat_execution_cost_no_trigger():
    orch = _orch(window=3)
    hits = _run_cycles(orch, [11.0, 11.0, 11.0], [11.0, 11.0, 11.0])
    assert hits == []


def test_diagnose_declining_execution_cost_no_trigger():
    orch = _orch(window=3)
    hits = _run_cycles(orch, [16.5, 11.0, 5.5], [16.5, 11.0, 5.5])
    assert hits == []


def test_diagnose_no_trigger_when_total_pnl_non_positive():
    orch = _orch(window=3)
    last = []
    for s, sp in [(5.5, 5.5), (11.0, 11.0), (16.5, 16.5)]:
        perception = _perception({
            "grid": {
                "total_pnl": 0.0,
                "total_fees": 10.0,
                "total_slippage_cost": s,
                "total_spread_cost": sp,
            },
        })
        last = orch._diagnose(perception)
    assert _rising_alerts(last) == []


def test_diagnose_no_trigger_when_exec_cost_zero():
    orch = _orch(window=3)
    hits = _run_cycles(orch, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert hits == []


def test_diagnose_custom_window():
    orch = _orch(window=2)
    assert _run_cycles(orch, [5.5], [5.5]) == []
    two = _run_cycles(orch, [5.5, 11.0], [5.5, 11.0])
    assert len(two) == 1


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_reallocate_decrease():
    orch = _orch(reduce_target=0.1)
    alerts = [{"type": "execution_cost_rising", "strategy": "grid", "message": "执行成本上升"}]
    actions = orch._execution_cost_trend_actions({}, alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "reallocate"
    assert a["action"] == "decrease"
    assert a["strategy"] == "grid"
    assert a["target_allocation"] == pytest.approx(0.1)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "execution_cost_rising", "strategy": "grid"},
        {"type": "execution_cost_rising", "strategy": "grid"},
    ]
    assert len(orch._execution_cost_trend_actions({}, alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._execution_cost_trend_actions({}, [{"type": "high_execution_cost"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "execution_cost_rising", "strategy": "grid", "message": "x"}]
    assert orch._execution_cost_trend_actions({}, alerts) == []
    assert _run_cycles(orch, [5.5, 11.0, 16.5], [5.5, 11.0, 16.5]) == []
