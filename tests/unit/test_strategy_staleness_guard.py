"""
策略休眠资金回收守卫（strategy_staleness_guard）单元测试
=======================================================
覆盖：snapshot_to_dict 暴露 lifecycle_last_trade_age_hours、_diagnose 闲置告警、
_strategy_staleness_actions 回收资金动作、阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, snapshot_to_dict


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "reclaim_idle_hours": 48,
        "min_trades": 5,
        "reclaim_target": 0.05,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"strategy_staleness_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── snapshot_to_dict 字段暴露 ────────────────────────────

class _Contrib:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_snapshot_exposes_idle_hours():
    snap = _Contrib(strategies={
        "grid": _Contrib(lifecycle_last_trade_age_hours=60.0),
    }, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["lifecycle_last_trade_age_hours"] == pytest.approx(60.0)


def test_snapshot_idle_hours_default_zero():
    snap = _Contrib(strategies={"grid": _Contrib()}, available=True)
    d = snapshot_to_dict(snap)
    assert d["strategies"]["grid"]["lifecycle_last_trade_age_hours"] == 0.0


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_stale_strategy():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 8, "lifecycle_last_trade_age_hours": 60.0},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "strategy_stale"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["idle_hours"] == pytest.approx(60.0)


def test_diagnose_no_alert_when_active():
    orch = _orch()
    perception = _perception({
        "grid": {"total_trades": 8, "lifecycle_last_trade_age_hours": 2.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "strategy_stale"]


def test_diagnose_insufficient_trades_no_alert():
    # 无数据/新生策略（trades 不足）不回收，避免误回收从未活跃的策略
    orch = _orch(min_trades=5)
    perception = _perception({
        "grid": {"total_trades": 2, "lifecycle_last_trade_age_hours": 100.0},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "strategy_stale"]


def test_diagnose_boundary_at_threshold():
    orch = _orch(reclaim_idle_hours=48)
    perception = _perception({
        "grid": {"total_trades": 8, "lifecycle_last_trade_age_hours": 48.0},
    })
    alerts = orch._diagnose(perception)
    assert [a for a in alerts if a["type"] == "strategy_stale"]  # ≥ 边界触发


# ── 动作生成 ─────────────────────────────────────────────

def test_actions_generate_reclaim():
    orch = _orch(reclaim_target=0.05)
    alerts = [{"type": "strategy_stale", "strategy": "grid", "message": "闲置"}]
    actions = orch._strategy_staleness_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "reallocate"
    assert a["strategy"] == "grid"
    assert a["action"] == "decrease"
    assert a["target_allocation"] == pytest.approx(0.05)


def test_actions_no_alert():
    orch = _orch()
    assert orch._strategy_staleness_actions([{"type": "spot_overweight"}]) == []


def test_actions_skip_missing_strategy():
    orch = _orch()
    actions = orch._strategy_staleness_actions([{"type": "strategy_stale", "strategy": ""}])
    assert actions == []


def test_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "strategy_stale", "strategy": "grid", "message": "x"}]
    assert orch._strategy_staleness_actions(alerts) == []
    perception = _perception({
        "grid": {"total_trades": 8, "lifecycle_last_trade_age_hours": 60.0},
    })
    assert not [a for a in orch._diagnose(perception) if a["type"] == "strategy_stale"]
