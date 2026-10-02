"""
策略级已实现亏损守卫（realized_loss_guard）单元测试
===================================================
覆盖：_diagnose 已实现亏损告警、_realized_loss_actions 收敛动作、
阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "loss_threshold": 0.03,
        "reduce_target": 0.1,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"realized_loss_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies, equity=100.0):
    return {"equity": equity, "contribution": {"strategies": strategies}}


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_realized_loss():
    # realized_pnl=-5 / equity=100 → loss_pct 0.05 >= 0.03 → 触发
    orch = _orch()
    perception = _perception({
        "grid": {"realized_pnl": -5.0, "unrealized_pnl": 1.0, "total_trades": 10},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "realized_loss"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["loss_pct"] == pytest.approx(0.05)


def test_diagnose_no_alert_when_realized_positive():
    orch = _orch()
    perception = _perception({
        "grid": {"realized_pnl": 5.0, "unrealized_pnl": 0.0, "total_trades": 10},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "realized_loss"]


def test_diagnose_below_threshold():
    # realized_pnl=-2 / equity=100 → loss_pct 0.02 < 0.03 → 不触发
    orch = _orch(loss_threshold=0.03)
    perception = _perception({
        "grid": {"realized_pnl": -2.0, "unrealized_pnl": 0.0, "total_trades": 10},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "realized_loss"]


def test_diagnose_missing_equity():
    # 无 equity → 无法算占比，不触发
    orch = _orch()
    perception = _perception({"grid": {"realized_pnl": -5.0}}, equity=None)
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "realized_loss"]


# ── 动作生成 ─────────────────────────────────────────────

def test_realized_loss_actions_generate_decrease():
    orch = _orch(reduce_target=0.1)
    alerts = [{"type": "realized_loss", "strategy": "grid", "message": "已实现亏损"}]
    actions = orch._realized_loss_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "reallocate"
    assert a["strategy"] == "grid"
    assert a["action"] == "decrease"
    assert a["target_allocation"] == pytest.approx(0.1)


def test_realized_loss_actions_dedup():
    orch = _orch()
    alerts = [
        {"type": "realized_loss", "strategy": "grid"},
        {"type": "realized_loss", "strategy": "grid"},
    ]
    assert len(orch._realized_loss_actions(alerts)) == 1


def test_realized_loss_actions_no_alert():
    orch = _orch()
    assert orch._realized_loss_actions([{"type": "strategy_floating_loss", "strategy": "grid"}]) == []


def test_realized_loss_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "realized_loss", "strategy": "grid", "message": "x"}]
    assert orch._realized_loss_actions(alerts) == []
    perception = _perception({"grid": {"realized_pnl": -5.0}})
    assert not [a for a in orch._diagnose(perception) if a["type"] == "realized_loss"]
