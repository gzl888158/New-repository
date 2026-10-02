"""
信号质量门槛强化（offensive min_regime_confidence）单元测试
========================================================
覆盖：进攻开单需市场状态置信度达标（够确定才开单）、置信度不足阻断、阈值=0 向后兼容。
"""
from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _orch(min_regime_confidence=0.5):
    return QuantAGIOrchestrator(config={"agi_orchestrator": {"offensive_allocation": {
        "enabled": True, "min_regime_strength": 0.6, "min_regime_confidence": min_regime_confidence,
        "max_drawdown_pct": 0.05, "boost_step": 0.05, "max_target": 0.4,
    }}})


def _perception(confidence):
    return {
        "equity": 1000.0,
        "total_capital": 1000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 0.0,
        "market_regime": {"regime": "trend_bullish", "strength": 0.7, "confidence": confidence},
        "contribution": {"strategies": {"grid": {"health_grade": "A"}}},
        "freeze_state": {},
    }


def test_offensive_blocked_on_low_confidence():
    orch = _orch(min_regime_confidence=0.5)
    alerts = orch._diagnose(_perception(confidence=0.3))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_allowed_on_high_confidence():
    orch = _orch(min_regime_confidence=0.5)
    alerts = orch._diagnose(_perception(confidence=0.7))
    assert [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_no_gate_when_threshold_zero():
    # 阈值=0 → 无置信度门槛（向后兼容，即使 confidence 缺失/为0 也放行）
    orch = _orch(min_regime_confidence=0.0)
    alerts = orch._diagnose(_perception(confidence=0.0))
    assert [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_boundary_confidence_at_threshold():
    orch = _orch(min_regime_confidence=0.5)
    alerts = orch._diagnose(_perception(confidence=0.5))
    assert [a for a in alerts if a["type"] == "offensive_opportunity"]  # >= 边界触发
