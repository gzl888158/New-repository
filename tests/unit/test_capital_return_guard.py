"""
资本回报率阈值守卫（capital_return_guard）单元测试
==================================================
覆盖：单位资本产出低于绝对阈值触发 capital_return_low、动作降杠杆、
未达阈值不触发、交易笔数不足不触发（与趋势型 capital_return_trend_guard 互补）。
"""
from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _orch(enabled=True, threshold=-10.0, min_trades=10):
    return QuantAGIOrchestrator(config={"agi_orchestrator": {"capital_return_guard": {
        "enabled": enabled,
        "capital_return_threshold": threshold,
        "min_trades": min_trades,
        "deterioration_leverage": 1.0,
    }}})


def _perception(pnl_per_capital_pct=-15.0, total_trades=20):
    return {
        "equity": 1000.0,
        "total_capital": 1000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 0.0,
        "contribution": {"strategies": {"grid": {
            "total_trades": total_trades,
            "pnl_per_capital_pct": pnl_per_capital_pct,
        }}},
        "freeze_state": {},
    }


def _alerts_of_type(alerts, type_):
    return [a for a in alerts if a.get("type") == type_]


def test_capital_return_low_detected():
    """pnl_per_capital_pct=-15% < -10% 阈值且 trades≥10 → 触发 capital_return_low。"""
    orch = _orch()
    alerts = orch._diagnose(_perception(pnl_per_capital_pct=-15.0, total_trades=20))
    assert _alerts_of_type(alerts, "capital_return_low")


def test_capital_return_actions_converge():
    """capital_return_low → _capital_return_actions 生成降杠杆 param_adjust。"""
    orch = _orch()
    alerts = [{"type": "capital_return_low", "strategy": "grid",
               "pnl_per_capital_pct": -15.0, "message": "资本回报率过低"}]
    actions = orch._capital_return_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["value"] == 1.0


def test_not_triggered_above_threshold():
    """pnl_per_capital_pct=-5% > -10% 阈值 → 不触发。"""
    orch = _orch()
    alerts = orch._diagnose(_perception(pnl_per_capital_pct=-5.0, total_trades=20))
    assert not _alerts_of_type(alerts, "capital_return_low")


def test_not_triggered_when_few_trades():
    """trades=5 < min_trades=10 → 不触发（样本不足避免噪声误触发）。"""
    orch = _orch()
    alerts = orch._diagnose(_perception(pnl_per_capital_pct=-15.0, total_trades=5))
    assert not _alerts_of_type(alerts, "capital_return_low")


def test_disabled_no_alerts():
    """enabled=False → 不触发（向后兼容）。"""
    orch = _orch(enabled=False)
    alerts = orch._diagnose(_perception(pnl_per_capital_pct=-15.0, total_trades=20))
    assert not _alerts_of_type(alerts, "capital_return_low")
