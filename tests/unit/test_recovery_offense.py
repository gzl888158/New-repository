"""
恢复期渐进进攻（recovery_offense）单元测试
==========================================
覆盖：默认 RECOVERY 禁进攻（向后兼容）、启用后渐进进攻、恢复期乘数过低仍禁、
DECLINE/EMERGENCY 冻结不受影响、NORMAL 模式 boost_step 不缩放。
"""
from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _orch(recovery_offense_enabled=False, recovery_offense_min_multiplier=0.3):
    return QuantAGIOrchestrator(config={"agi_orchestrator": {"offensive_allocation": {
        "enabled": True,
        "min_regime_strength": 0.6,
        "max_drawdown_pct": 0.05,
        "boost_step": 0.05,
        "max_target": 0.4,
        "recovery_offense_enabled": recovery_offense_enabled,
        "recovery_offense_min_multiplier": recovery_offense_min_multiplier,
    }}})


def _perception(mode="recovery", position_multiplier=0.5):
    return {
        "equity": 1000.0,
        "total_capital": 1000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 0.0,
        "equity_status": {"mode": mode, "position_multiplier": position_multiplier},
        "market_regime": {"regime": "trend_bullish", "strength": 0.7},
        "contribution": {"strategies": {"grid": {"health_grade": "A"}}},
        "freeze_state": {},
    }


def _offensive(alerts):
    return [a for a in alerts if a.get("type") == "offensive_opportunity"]


def test_default_recovery_blocks_offensive():
    """默认（recovery_offense_enabled=False）RECOVERY 模式禁进攻（向后兼容）。"""
    orch = _orch(recovery_offense_enabled=False)
    alerts = orch._diagnose(_perception("recovery", 0.5))
    assert not _offensive(alerts)


def test_recovery_offense_enabled_allows_progressive():
    """启用后 RECOVERY 模式 + 乘数达标 → 允许渐进进攻，且 boost_step 被乘数缩放。"""
    orch = _orch(recovery_offense_enabled=True)
    alerts = orch._diagnose(_perception("recovery", 0.5))
    opp = _offensive(alerts)
    assert opp, "恢复期乘数 0.5 达标，应产生 offensive_opportunity"
    # boost_step = raw_boost(0.05) * position_multiplier(0.5) = 0.025
    assert abs(opp[0]["boost_step"] - 0.025) < 1e-9, f"boost_step 未按乘数缩放: {opp[0]['boost_step']}"


def test_recovery_offense_low_multiplier_blocks():
    """恢复期乘数低于下限（0.1 < 0.3）→ 仍禁进攻（刚脱离 EMERGENCY 不宜进攻）。"""
    orch = _orch(recovery_offense_enabled=True, recovery_offense_min_multiplier=0.3)
    alerts = orch._diagnose(_perception("recovery", 0.1))
    assert not _offensive(alerts)


def test_decline_still_blocked_with_recovery_offense():
    """DECLINE 衰退模式即使启用 recovery_offense 仍禁进攻（增强不误伤衰退冻结）。"""
    orch = _orch(recovery_offense_enabled=True)
    alerts = orch._diagnose(_perception("decline", 0.5))
    assert not _offensive(alerts)


def test_emergency_still_blocked_with_recovery_offense():
    """EMERGENCY 紧急模式即使启用 recovery_offense 仍禁进攻（硬冻结不可突破）。"""
    orch = _orch(recovery_offense_enabled=True)
    alerts = orch._diagnose(_perception("emergency", 0.5))
    assert not _offensive(alerts)


def test_normal_boost_step_not_scaled():
    """NORMAL 模式不缩放 boost_step（恢复期缩放仅作用于 RECOVERY）。"""
    orch = _orch(recovery_offense_enabled=True)
    alerts = orch._diagnose(_perception("normal", 1.0))
    opp = _offensive(alerts)
    assert opp, "NORMAL 模式应正常进攻"
    assert abs(opp[0]["boost_step"] - 0.05) < 1e-9, f"NORMAL boost_step 不应被缩放: {opp[0]['boost_step']}"
