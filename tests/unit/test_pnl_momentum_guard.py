"""
PnL 动量绝对阈值守卫（pnl_momentum_guard）单元测试
=============================================================
覆盖：_diagnose 动量衰竭告警、_pnl_momentum_guard_actions 降杠杆动作、
阈值边界（min_momentum_ratio）、样本不足、去重与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "min_momentum_ratio": 0.7,
        "min_trades": 10,
        "deterioration_leverage": 1.0,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"pnl_momentum_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_trend_pnl_7d_vs_30d():
    snap = _Contrib(strategies={
        "grid": _Contrib(trend_pnl_7d_vs_30d=0.5),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["trend_pnl_7d_vs_30d"] == pytest.approx(0.5)


def test_snapshot_momentum_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["trend_pnl_7d_vs_30d"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_momentum_faded():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "trend_pnl_7d_vs_30d": 0.5},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "pnl_momentum_faded"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["pnl_momentum"] == pytest.approx(0.5)


def test_diagnose_no_alert_when_good_momentum():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "trend_pnl_7d_vs_30d": 1.5},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "pnl_momentum_faded"]


def test_diagnose_no_alert_at_threshold_boundary():
    # mom == min_momentum_ratio（0.7）不触发（严格小于才触发）
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "trend_pnl_7d_vs_30d": 0.7},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "pnl_momentum_faded"]


def test_diagnose_no_alert_when_zero_missing():
    # mom == 0（字段缺失/无数据）不应误触发（0 < mom 条件排除）
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "trend_pnl_7d_vs_30d": 0.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "pnl_momentum_faded"]


def test_diagnose_no_alert_when_negative():
    # mom < 0（近期转亏，非「衰竭」语义，由其他守卫捕获）不应触发本守卫
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 10, "trend_pnl_7d_vs_30d": -0.5},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "pnl_momentum_faded"]


def test_diagnose_insufficient_trades():
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"total_trades": 5, "trend_pnl_7d_vs_30d": 0.5},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "pnl_momentum_faded"]


def test_diagnose_custom_min_ratio():
    # min_momentum_ratio 降低后，仅在更低动量时触发
    orch = _orch(min_momentum_ratio=0.4)
    perception = _perception({
        "grid": {"total_trades": 10, "trend_pnl_7d_vs_30d": 0.3},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "pnl_momentum_faded"]
    assert len(hits) == 1


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_param_adjust():
    orch = _orch(deterioration_leverage=1.0)
    alerts = [{"type": "pnl_momentum_faded", "strategy": "grid", "message": "动量衰竭"}]
    actions = orch._pnl_momentum_guard_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["strategy"] == "grid"
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)


def test_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "pnl_momentum_faded", "strategy": "grid"},
        {"type": "pnl_momentum_faded", "strategy": "grid"},
    ]
    assert len(orch._pnl_momentum_guard_actions(alerts)) == 1


def test_actions_no_alert():
    orch = _orch()
    assert orch._pnl_momentum_guard_actions([{"type": "pnl_momentum_deteriorating"}]) == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "pnl_momentum_faded", "strategy": "grid", "message": "x"}]
    assert orch._pnl_momentum_guard_actions(alerts) == []
    perception = _perception({
        "grid": {"total_trades": 10, "trend_pnl_7d_vs_30d": 0.5},
    })
    assert not [a for a in orch._diagnose(perception)
                if a["type"] == "pnl_momentum_faded"]
