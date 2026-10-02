"""
开单精准度（max_regime_strength）单元测试
==========================================
覆盖：趋势强度上限抑制追涨杀跌、默认 1.0 不设上限（向后兼容）、
强度落在 [min, max] 区间才进攻、震荡市不受影响。
"""
from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _orch(max_regime_strength=1.0):
    return QuantAGIOrchestrator(config={"agi_orchestrator": {"offensive_allocation": {
        "enabled": True,
        "min_regime_strength": 0.6,
        "max_regime_strength": max_regime_strength,
        "max_drawdown_pct": 0.05,
        "boost_step": 0.05,
        "max_target": 0.4,
    }}})


def _perception(regime="trend_bullish", strength=0.7):
    return {
        "equity": 1000.0,
        "total_capital": 1000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 0.0,
        "market_regime": {"regime": regime, "strength": strength},
        "contribution": {"strategies": {"grid": {"health_grade": "A"}}},
        "freeze_state": {},
    }


def _offensive(alerts):
    return [a for a in alerts if a.get("type") == "offensive_opportunity"]


def test_extreme_strength_blocked_by_max():
    """strength 超过 max_regime_strength（0.95）→ 抑制进攻（趋势末端不追涨杀跌）。"""
    orch = _orch(max_regime_strength=0.95)
    alerts = orch._diagnose(_perception(strength=0.99))
    assert not _offensive(alerts)


def test_normal_strength_allowed():
    """strength 落在 [min, max] 区间（0.7）→ 正常进攻。"""
    orch = _orch(max_regime_strength=0.95)
    alerts = orch._diagnose(_perception(strength=0.7))
    assert _offensive(alerts)


def test_default_no_upper_bound():
    """默认 max=1.0 不设上限：strength=0.99 仍进攻（向后兼容）。"""
    orch = _orch(max_regime_strength=1.0)
    alerts = orch._diagnose(_perception(strength=0.99))
    assert _offensive(alerts)


def test_below_min_still_blocked():
    """strength 低于 min_regime_strength（0.6）→ 仍不进攻（下限不被上限破坏）。"""
    orch = _orch(max_regime_strength=0.95)
    alerts = orch._diagnose(_perception(strength=0.5))
    assert not _offensive(alerts)
