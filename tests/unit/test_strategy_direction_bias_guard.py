"""
策略方向偏好失衡守卫（strategy_direction_bias_guard）单元测试
=============================================================
覆盖：_diagnose 方向笔数失衡告警、_direction_bias_actions 降杠杆、
阈值边界与禁用、offensive gate 抑制。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


# ── 工具 ─────────────────────────────────────────────────

def _direction_bias_orch(**overrides):
    cfg = {"agi_orchestrator": {"strategy_direction_bias_guard": {
        "enabled": True, "min_trades": 10, "bias_ratio_threshold": 0.9,
        "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["strategy_direction_bias_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _direction_bias_perception(long_trades, short_trades):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": long_trades + short_trades,
            "total_pnl": 5.0,
            "long_pnl": 3.0,
            "short_pnl": 2.0,
            "long_trades": long_trades,
            "short_trades": short_trades,
        }}},
        "freeze_state": {},
    }


# ── 诊断 ─────────────────────────────────────────────────

def test_direction_bias_detects_long_bias():
    # 多头 19 / 空头 1，单边占比 19/20 = 0.95 ≥ 0.9 → direction_bias
    orch = _direction_bias_orch()
    alerts = orch._diagnose(_direction_bias_perception(long_trades=19, short_trades=1))
    hits = [a for a in alerts if a["type"] == "direction_bias"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["bias_ratio"] == pytest.approx(0.95)


def test_direction_bias_detects_short_bias():
    # 空头 18 / 多头 2，单边占比 18/20 = 0.9 == 阈值 → direction_bias
    orch = _direction_bias_orch()
    alerts = orch._diagnose(_direction_bias_perception(long_trades=2, short_trades=18))
    hits = [a for a in alerts if a["type"] == "direction_bias"]
    assert len(hits) == 1
    assert hits[0]["bias_ratio"] == pytest.approx(0.9)


def test_direction_bias_below_threshold():
    # 多头 15 / 空头 5，单边占比 15/20 = 0.75 < 0.9 → 不触发
    orch = _direction_bias_orch()
    alerts = orch._diagnose(_direction_bias_perception(long_trades=15, short_trades=5))
    assert not [a for a in alerts if a["type"] == "direction_bias"]


def test_direction_bias_insufficient_trades():
    # 总笔数 9 < min_trades 10 → 不评估
    orch = _direction_bias_orch()
    alerts = orch._diagnose(_direction_bias_perception(long_trades=9, short_trades=0))
    assert not [a for a in alerts if a["type"] == "direction_bias"]


def test_direction_bias_disabled():
    orch = _direction_bias_orch(enabled=False)
    alerts = orch._diagnose(_direction_bias_perception(long_trades=19, short_trades=1))
    assert not [a for a in alerts if a["type"] == "direction_bias"]


# ── 动作生成 ─────────────────────────────────────────────

def test_direction_bias_actions_generate():
    orch = _direction_bias_orch()
    alerts = [{"type": "direction_bias", "strategy": "grid", "message": "方向偏好失衡"}]
    actions = orch._direction_bias_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["value"] == pytest.approx(1.0)


def test_direction_bias_actions_dedup():
    orch = _direction_bias_orch()
    alerts = [
        {"type": "direction_bias", "strategy": "grid"},
        {"type": "direction_bias", "strategy": "grid"},
    ]
    assert len(orch._direction_bias_actions(alerts)) == 1


def test_direction_bias_actions_no_alert():
    orch = _direction_bias_orch()
    assert orch._direction_bias_actions([{"type": "long_side_losing", "strategy": "grid"}]) == []


def test_direction_bias_actions_disabled():
    orch = _direction_bias_orch(enabled=False)
    alerts = [{"type": "direction_bias", "strategy": "grid"}]
    assert orch._direction_bias_actions(alerts) == []
