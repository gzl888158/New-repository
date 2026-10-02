"""
交易频率守卫（trade_frequency_guard）单元测试
=============================================
覆盖：_diagnose 每小时交易笔数过高告警、_trade_frequency_actions 收敛动作、
阈值边界与禁用。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


# ── 工具 ─────────────────────────────────────────────────

def _config(**overrides):
    guard = {
        "enabled": True,
        "min_active_hours": 24,
        "min_trades": 10,
        "frequency_threshold": 2.0,
        "reduce_target": 0.1,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"trade_frequency_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _perception(strategies):
    return {"contribution": {"strategies": strategies}}


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_high_trade_frequency():
    # 96 笔 / 48h = 2.0 笔/小时 == 阈值 → 触发
    orch = _orch()
    perception = _perception({
        "grid": {"active_hours": 48.0, "total_trades": 96},
    })
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "high_trade_frequency"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["trades_per_hour"] == pytest.approx(2.0)


def test_diagnose_no_alert_below_threshold():
    # 40 笔 / 40h = 1.0 笔/小时 < 2.0 → 不触发
    orch = _orch()
    perception = _perception({
        "grid": {"active_hours": 40.0, "total_trades": 40},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_trade_frequency"]


def test_diagnose_insufficient_active_hours():
    # active_hours=5 < 24 → 不评估
    orch = _orch(min_active_hours=24)
    perception = _perception({
        "grid": {"active_hours": 5.0, "total_trades": 100},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_trade_frequency"]


def test_diagnose_insufficient_trades():
    # total_trades=5 < 10 → 不评估
    orch = _orch(min_trades=10)
    perception = _perception({
        "grid": {"active_hours": 48.0, "total_trades": 5},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_trade_frequency"]


def test_diagnose_zero_active_hours():
    # active_hours=0 → 不评估（避免除零）
    orch = _orch()
    perception = _perception({
        "grid": {"active_hours": 0.0, "total_trades": 100},
    })
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_trade_frequency"]


# ── 动作生成 ─────────────────────────────────────────────

def test_trade_frequency_actions_generate_decrease():
    orch = _orch(reduce_target=0.1)
    alerts = [{"type": "high_trade_frequency", "strategy": "grid", "message": "高频刷单"}]
    actions = orch._trade_frequency_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "reallocate"
    assert a["strategy"] == "grid"
    assert a["action"] == "decrease"
    assert a["target_allocation"] == pytest.approx(0.1)


def test_trade_frequency_actions_no_alert():
    orch = _orch()
    assert orch._trade_frequency_actions([{"type": "spot_overweight"}]) == []


def test_trade_frequency_actions_skip_missing_strategy():
    orch = _orch()
    actions = orch._trade_frequency_actions([{"type": "high_trade_frequency", "strategy": ""}])
    assert actions == []


def test_trade_frequency_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "high_trade_frequency", "strategy": "grid", "message": "x"}]
    assert orch._trade_frequency_actions(alerts) == []
    perception = _perception({
        "grid": {"active_hours": 48.0, "total_trades": 96},
    })
    assert not [a for a in orch._diagnose(perception) if a["type"] == "high_trade_frequency"]
