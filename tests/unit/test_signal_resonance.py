"""
信号共振过滤（offensive signal_resonance）单元测试
=================================================
覆盖：进攻候选不仅健康度 A/B，还需趋势不衰退 + 累计盈利为正（多信号共振）。
"""
from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _orch(signal_resonance=True):
    return QuantAGIOrchestrator(config={"agi_orchestrator": {"offensive_allocation": {
        "enabled": True, "min_regime_strength": 0.6, "signal_resonance": signal_resonance,
        "max_drawdown_pct": 0.05, "boost_step": 0.05, "max_target": 0.4,
    }}})


def _perception(strategy):
    return {
        "equity": 1000.0,
        "total_capital": 1000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 0.0,
        "market_regime": {"regime": "trend_bullish", "strength": 0.7},
        "contribution": {"strategies": {"grid": strategy}},
        "freeze_state": {},
    }


def test_signal_resonance_filters_declining_trend():
    orch = _orch(signal_resonance=True)
    strategy = {"health_grade": "A", "trend": "declining", "total_pnl": 5.0}
    alerts = orch._diagnose(_perception(strategy))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_signal_resonance_filters_negative_pnl():
    orch = _orch(signal_resonance=True)
    strategy = {"health_grade": "A", "trend": "improving", "total_pnl": -5.0}
    alerts = orch._diagnose(_perception(strategy))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_signal_resonance_allows_resonant_signal():
    orch = _orch(signal_resonance=True)
    strategy = {"health_grade": "A", "trend": "improving", "total_pnl": 5.0}
    alerts = orch._diagnose(_perception(strategy))
    assert [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_signal_resonance_disabled_allows_declining():
    orch = _orch(signal_resonance=False)
    strategy = {"health_grade": "A", "trend": "declining", "total_pnl": 5.0}
    alerts = orch._diagnose(_perception(strategy))
    assert [a for a in alerts if a["type"] == "offensive_opportunity"]
