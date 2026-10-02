"""
开单时机校准（trend_confirmation_cycles）单元测试
================================================
覆盖：趋势未确认阻断进攻、趋势确认放行、阈值=0 向后兼容。
"""
from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _orch(trend_confirmation_cycles=2):
    return QuantAGIOrchestrator(config={"agi_orchestrator": {"offensive_allocation": {
        "enabled": True, "min_regime_strength": 0.6, "trend_confirmation_cycles": trend_confirmation_cycles,
        "max_drawdown_pct": 0.05, "boost_step": 0.05, "max_target": 0.4,
    }}})


def _perception(regime="trend_bullish"):
    return {
        "equity": 1000.0,
        "total_capital": 1000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 0.0,
        "market_regime": {"regime": regime, "strength": 0.7},
        "contribution": {"strategies": {"grid": {"health_grade": "A"}}},
        "freeze_state": {},
    }


def test_trend_not_confirmed_blocks_offensive():
    orch = _orch(trend_confirmation_cycles=2)
    orch._trend_confirmed_regime = "range_bound"  # 与当前 trend_bullish 不同 → confirmed_streak=1 < 2
    orch._trend_confirmed_streak = 1
    alerts = orch._diagnose(_perception("trend_bullish"))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_trend_confirmed_allows_offensive():
    orch = _orch(trend_confirmation_cycles=2)
    orch._trend_confirmed_regime = "trend_bullish"  # 延续 → confirmed_streak = 1+1 = 2 >= 2
    orch._trend_confirmed_streak = 1
    alerts = orch._diagnose(_perception("trend_bullish"))
    assert [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_trend_confirmation_disabled_allows_immediate():
    orch = _orch(trend_confirmation_cycles=0)
    orch._trend_confirmed_regime = "range_bound"
    orch._trend_confirmed_streak = 1
    alerts = orch._diagnose(_perception("trend_bullish"))
    assert [a for a in alerts if a["type"] == "offensive_opportunity"]
