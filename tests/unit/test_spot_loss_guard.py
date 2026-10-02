"""
现货亏损止损回收（spot_loss）单元测试
======================================
覆盖：现货策略累计亏损达阈值触发 spot_loss、动作收敛现货、未达阈值不触发、
spot_profit（盈利止盈）不受影响。
"""
from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _orch(loss_take_pct=0.02, profit_take_pct=0.02):
    return QuantAGIOrchestrator(config={"agi_orchestrator": {"spot_hold_guard": {
        "enabled": True,
        "max_spot_currencies": 3,
        "profit_take_pct": profit_take_pct,
        "loss_take_pct": loss_take_pct,
        "reduce_target": 0.1,
    }}})


def _perception(spot_grid_pnl=0.0, spot_martingale_pnl=0.0):
    return {
        "equity": 1000.0,
        "total_capital": 1000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 0.0,
        "spot_holdings": {"currencies": {"BTC": 0.1}, "count": 1},
        "contribution": {"strategies": {
            "spot_grid": {"total_pnl": spot_grid_pnl},
            "spot_martingale": {"total_pnl": spot_martingale_pnl},
        }},
        "freeze_state": {},
    }


def _alerts_of_type(alerts, type_):
    return [a for a in alerts if a.get("type") == type_]


def test_spot_loss_detected_when_below_threshold():
    """现货策略累计亏损 -30（equity=1000，-3% < -2% 阈值）→ 生成 spot_loss 告警。"""
    orch = _orch(loss_take_pct=0.02)
    alerts = orch._diagnose(_perception(spot_grid_pnl=-20.0, spot_martingale_pnl=-10.0))
    assert _alerts_of_type(alerts, "spot_loss"), "现货亏损达阈值应触发 spot_loss"


def test_spot_loss_actions_converge():
    """spot_loss → _spot_hold_actions 收敛 spot_grid/spot_martingale 到 reduce_target。"""
    orch = _orch(loss_take_pct=0.02)
    alerts = [{"type": "spot_loss", "spot_pnl": -30.0, "message": "现货累计亏损达标"}]
    actions = orch._spot_hold_actions(alerts)
    strategies = {a.get("strategy") for a in actions}
    assert "spot_grid" in strategies and "spot_martingale" in strategies
    assert all(a.get("action") == "decrease" for a in actions)
    assert all(a.get("target_allocation") == 0.1 for a in actions)


def test_spot_loss_not_triggered_when_small():
    """现货策略累计亏损 -5（-0.5% > -2% 阈值）→ 不触发 spot_loss。"""
    orch = _orch(loss_take_pct=0.02)
    alerts = orch._diagnose(_perception(spot_grid_pnl=-5.0, spot_martingale_pnl=0.0))
    assert not _alerts_of_type(alerts, "spot_loss")


def test_spot_profit_still_works():
    """现货累计盈利 +30（+3% ≥ 2% 阈值）→ 仍触发 spot_profit（盈利止盈不被破坏）。"""
    orch = _orch(loss_take_pct=0.02, profit_take_pct=0.02)
    alerts = orch._diagnose(_perception(spot_grid_pnl=20.0, spot_martingale_pnl=10.0))
    assert _alerts_of_type(alerts, "spot_profit")
