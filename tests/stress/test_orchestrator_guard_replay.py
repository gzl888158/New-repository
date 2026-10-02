"""编排层守卫参数回放验证（orchestrator guard replay validation）
================================================================
用感知序列回放 harness 验证 QuantAGIOrchestrator 编排层守卫的参数有效性。

与 test_guard_parameter_validation.py 的区别：
- 后者验证 BacktestEngine 已参数化的策略级守卫（信号质量/磨损/趋势过滤）
- 本文件验证编排层守卫（trend_confirmation/stress_loss_budget/drawdown_accel/
  downside_momentum/tail_risk/health_degradation），这些守卫作用于运行时
  感知数据、无对应回测路径，必须用感知序列回放验证。

验证方式：A/B 对比——守卫参数开 vs 关/严 vs 宽，回放同一 perception 序列，
对比 alerts 触发情况与进攻机会差异。
"""
import pytest

from tests.stress.replay_harness import (
    PerceptionReplayHarness,
    trend_confirmation_sequence,
    regime_switch_sequence,
    stress_escalation_sequence,
    drawdown_acceleration_sequence,
    downside_momentum_sequence,
    tail_risk_sequence,
    health_degradation_sequence,
)


# ---------------------------------------------------------------------------
# 进攻块基线配置：满足所有前置条件，使 trend_confirmation_cycles 成为唯一变量
# ---------------------------------------------------------------------------
OFFENSIVE_BASE = {
    "enabled": True,
    "min_regime_strength": 0.5,
    "min_regime_confidence": 0.5,
    "max_drawdown_pct": 0.15,
    "boost_step": 0.05,
}

HEALTH_ALERT_TYPES = {
    "portfolio_health_low",
    "portfolio_health_deteriorating",
    "health_deteriorating",
    "health_crash",
    "strategy_health_critical",
}


# ---------------------------------------------------------------------------
# 1. trend_confirmation_cycles（开单时机校准）
# ---------------------------------------------------------------------------

class TestTrendConfirmationReplay:
    """趋势确认周期：市场状态需持续 >= cycles 个周期才开单。

    验证 _trend_confirmed_streak 在 _reflect 里跨周期更新（第 7826-7830 行），
    以及 _diagnose 进攻块读取 streak 判断 regime_confirmed（第 3807-3815 行）。
    """

    def test_cycles_delay_offense(self):
        """cycles=3 时，前 2 周期无 offensive_opportunity，第 3 周期才出现。"""
        cfg = {"offensive_allocation": {**OFFENSIVE_BASE, "trend_confirmation_cycles": 3}}
        seq = trend_confirmation_sequence(5, regime="trend_bullish")
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        # streak 累积：1,2,3,4,5
        assert result["streak_log"] == [1, 2, 3, 4, 5]
        # offensive_opportunity 只在第 3 周期及之后出现
        assert result["offensive_opportunity_cycles"] == [3, 4, 5]

    def test_no_cycles_immediate_offense(self):
        """cycles=0 时，从第 1 周期就有 offensive_opportunity。"""
        cfg = {"offensive_allocation": {**OFFENSIVE_BASE, "trend_confirmation_cycles": 0}}
        seq = trend_confirmation_sequence(5, regime="trend_bullish")
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert result["offensive_opportunity_cycles"] == [1, 2, 3, 4, 5]

    def test_ab_comparison(self):
        """A/B：cycles=3 vs cycles=0，进攻次数 3 vs 5。"""
        seq = trend_confirmation_sequence(5, regime="trend_bullish")
        ab = PerceptionReplayHarness().run_ab(
            seq,
            guard_cfg_on={"offensive_allocation": {**OFFENSIVE_BASE, "trend_confirmation_cycles": 3}},
            guard_cfg_off={"offensive_allocation": {**OFFENSIVE_BASE, "trend_confirmation_cycles": 0}},
        )
        assert ab["on"]["offensive_opportunity_count"] == 3
        assert ab["off"]["offensive_opportunity_count"] == 5
        assert ab["delta"]["offensive_delta"] == -2

    def test_regime_switch_resets_streak(self):
        """regime 切换后 streak 重置。"""
        cfg = {"offensive_allocation": {**OFFENSIVE_BASE, "trend_confirmation_cycles": 3}}
        # 前 3 周期 trend_bullish，后 3 周期 range_bound（无 range_bound_enabled 不进攻）
        seq = regime_switch_sequence(switch_at=3, n=6)
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        # streak：1,2,3（bullish）→ 1,2,3（range_bound，重置）
        assert result["streak_log"] == [1, 2, 3, 1, 2, 3]
        # 仅第 3 周期进攻（bullish 确认后；range_bound 未开 range_bound_enabled）
        assert result["offensive_opportunity_cycles"] == [3]


# ---------------------------------------------------------------------------
# 2. portfolio_stress_guard（组合压力测试守卫）
# ---------------------------------------------------------------------------

class TestStressGuardReplay:
    """组合压力测试：毛敞口压力损失占权益比例超 budget 时收敛。

    触发逻辑（_diagnose 第 4011-4029 行）：
      stress_loss = gross × scenario_pct
      stress_loss / equity >= stress_loss_budget → stress_test_failed (warning)
      severe_loss = gross × severe_scenario_pct
      severe_loss / equity >= severe_loss_budget → severe_stress_test_failed (critical)
    """

    def test_low_budget_triggers_at_threshold(self):
        """budget=0.10, scenario_pct=0.20：gross>=5000 时触发。"""
        cfg = {"portfolio_stress_guard": {
            "enabled": True,
            "stress_scenario_pct": 0.20,
            "stress_loss_budget": 0.10,
        }}
        # stress_loss = gross * 0.20; 触发当 gross*0.20/10000 >= 0.10 → gross >= 5000
        seq = stress_escalation_sequence(10000.0, [2000, 4000, 5000, 6000, 8000])
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "stress_test_failed" in result["alerts_by_type"]
        stress_cycles = [
            i + 1 for i, alerts in enumerate(result["alerts_log"])
            if any(a.get("type") == "stress_test_failed" for a in alerts)
        ]
        assert stress_cycles == [3, 4, 5]

    def test_high_budget_no_trigger(self):
        """budget=0.50：gross=8000 时 stress_loss/equity=16% < 50% 不触发。"""
        cfg = {"portfolio_stress_guard": {
            "enabled": True,
            "stress_scenario_pct": 0.20,
            "stress_loss_budget": 0.50,
        }}
        seq = stress_escalation_sequence(10000.0, [2000, 4000, 6000, 8000])
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "stress_test_failed" not in result["alerts_by_type"]

    def test_severe_stress_triggers_critical(self):
        """severe_scenario_pct=0.40, severe_budget=0.30：gross>=7500 触发 critical。"""
        cfg = {"portfolio_stress_guard": {
            "enabled": True,
            "stress_scenario_pct": 0.20,
            "stress_loss_budget": 0.50,
            "severe_scenario_pct": 0.40,
            "severe_loss_budget": 0.30,
        }}
        # severe_loss = gross * 0.40; 触发当 gross*0.40/10000 >= 0.30 → gross >= 7500
        seq = stress_escalation_sequence(10000.0, [6000, 7500, 10000])
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "severe_stress_test_failed" in result["alerts_by_type"]
        severe_alerts = [
            a for alerts in result["alerts_log"] for a in alerts
            if a.get("type") == "severe_stress_test_failed"
        ]
        assert all(a.get("level") == "critical" for a in severe_alerts)

    def test_ab_comparison(self):
        """A/B：budget=0.10 vs budget=0.50。"""
        seq = stress_escalation_sequence(10000.0, [2000, 4000, 6000, 8000, 10000])
        ab = PerceptionReplayHarness().run_ab(
            seq,
            guard_cfg_on={"portfolio_stress_guard": {
                "enabled": True, "stress_scenario_pct": 0.20, "stress_loss_budget": 0.10,
            }},
            guard_cfg_off={"portfolio_stress_guard": {
                "enabled": True, "stress_scenario_pct": 0.20, "stress_loss_budget": 0.50,
            }},
        )
        assert "stress_test_failed" in ab["on"]["alerts_by_type"]
        assert "stress_test_failed" not in ab["off"]["alerts_by_type"]
        assert ab["delta"]["alerts_delta"] > 0


# ---------------------------------------------------------------------------
# 3. drawdown_accelerating（回撤加速预警）
# ---------------------------------------------------------------------------

class TestDrawdownAccelReplay:
    """回撤加速：max_drawdown_pct 连续 window 周期严格加深时触发。

    _drawdown_history 是 deque(maxlen=window)，window 满后持续检查最近 window 个值
    是否严格递增（_diagnose 第 2364-2384 行）。
    """

    def test_accel_triggers_on_window_full(self):
        """window=3，dd 序列 0.02→0.04→0.06→0.08→0.10：第 3 周期首次触发。"""
        cfg = {"drawdown_acceleration_guard": {"enabled": True, "window": 3}}
        seq = drawdown_acceleration_sequence(5, start_dd=0.02)
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "drawdown_accelerating" in result["alerts_by_type"]
        accel_cycles = [
            i + 1 for i, alerts in enumerate(result["alerts_log"])
            if any(a.get("type") == "drawdown_accelerating" for a in alerts)
        ]
        # 第 3/4/5 周期触发（deque 满后持续严格递增）
        assert 3 in accel_cycles
        assert accel_cycles == [3, 4, 5]

    def test_disabled_no_trigger(self):
        """disabled 时不触发。"""
        cfg = {"drawdown_acceleration_guard": {"enabled": False, "window": 3}}
        seq = drawdown_acceleration_sequence(5, start_dd=0.02)
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "drawdown_accelerating" not in result["alerts_by_type"]

    def test_ab_comparison(self):
        """A/B：enabled vs disabled。"""
        seq = drawdown_acceleration_sequence(5, start_dd=0.03)
        ab = PerceptionReplayHarness().run_ab(
            seq,
            guard_cfg_on={"drawdown_acceleration_guard": {"enabled": True, "window": 3}},
            guard_cfg_off={"drawdown_acceleration_guard": {"enabled": False, "window": 3}},
        )
        assert "drawdown_accelerating" in ab["on"]["alerts_by_type"]
        assert "drawdown_accelerating" not in ab["off"]["alerts_by_type"]


# ---------------------------------------------------------------------------
# 4. downside_momentum_guard（连续下跌收敛）
# ---------------------------------------------------------------------------

class TestDownsideMomentumReplay:
    """连续下跌：consecutive_down >= max_consecutive_down 时触发 market_panicking。

    触发逻辑（_diagnose 第 2403-2415 行）。
    """

    def test_consecutive_down_triggers(self):
        """max_consecutive_down=3，consecutive_down: 1→5：第 3/4/5 周期触发。"""
        cfg = {"downside_momentum_guard": {"enabled": True, "max_consecutive_down": 3}}
        seq = downside_momentum_sequence(5)
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "market_panicking" in result["alerts_by_type"]
        panic_cycles = [
            i + 1 for i, alerts in enumerate(result["alerts_log"])
            if any(a.get("type") == "market_panicking" for a in alerts)
        ]
        assert panic_cycles == [3, 4, 5]

    def test_disabled_no_trigger(self):
        """disabled 时不触发。"""
        cfg = {"downside_momentum_guard": {"enabled": False, "max_consecutive_down": 3}}
        seq = downside_momentum_sequence(5)
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "market_panicking" not in result["alerts_by_type"]

    def test_ab_comparison(self):
        """A/B：enabled vs disabled。"""
        seq = downside_momentum_sequence(5)
        ab = PerceptionReplayHarness().run_ab(
            seq,
            guard_cfg_on={"downside_momentum_guard": {"enabled": True, "max_consecutive_down": 3}},
            guard_cfg_off={"downside_momentum_guard": {"enabled": False, "max_consecutive_down": 3}},
        )
        assert "market_panicking" in ab["on"]["alerts_by_type"]
        assert "market_panicking" not in ab["off"]["alerts_by_type"]


# ---------------------------------------------------------------------------
# 5. tail_risk_guard（尾部风险收敛）
# ---------------------------------------------------------------------------

class TestTailRiskReplay:
    """尾部风险：组合中最深策略回撤 max(max_drawdown_i) 超阈值时触发 high_tail_risk。

    触发逻辑（_diagnose 第 3327-3348 行）：遍历 contribution.strategies 取
    各策略 max_drawdown 字段的最大值，超过 tail_risk_threshold 时触发。
    """

    def test_tail_risk_triggers(self):
        """threshold=0.15，策略 max_drawdown=0.20：每周期都触发。"""
        cfg = {"tail_risk_guard": {"enabled": True, "tail_risk_threshold": 0.15}}
        seq = tail_risk_sequence(max_dd=0.20, n=3)
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "high_tail_risk" in result["alerts_by_type"]
        assert result["alerts_by_type"]["high_tail_risk"] == 3

    def test_high_threshold_no_trigger(self):
        """threshold=0.50，max_drawdown=0.20 不触发。"""
        cfg = {"tail_risk_guard": {"enabled": True, "tail_risk_threshold": 0.50}}
        seq = tail_risk_sequence(max_dd=0.20, n=3)
        result = PerceptionReplayHarness(agi_cfg=cfg).replay(seq)

        assert "high_tail_risk" not in result["alerts_by_type"]

    def test_ab_comparison(self):
        """A/B：threshold=0.15 vs 0.50。"""
        seq = tail_risk_sequence(max_dd=0.20, n=3)
        ab = PerceptionReplayHarness().run_ab(
            seq,
            guard_cfg_on={"tail_risk_guard": {"enabled": True, "tail_risk_threshold": 0.15}},
            guard_cfg_off={"tail_risk_guard": {"enabled": True, "tail_risk_threshold": 0.50}},
        )
        assert "high_tail_risk" in ab["on"]["alerts_by_type"]
        assert "high_tail_risk" not in ab["off"]["alerts_by_type"]


# ---------------------------------------------------------------------------
# 6. health_degradation（策略健康度劣化收敛）
# ---------------------------------------------------------------------------

class TestHealthDegradationReplay:
    """策略健康度劣化：A→F 序列触发组合健康度告警。

    编排层健康度守卫默认启用（无 enabled 开关），按 overall_health_score
    和 health_grade 自动触发 portfolio_health_low / deteriorating 等告警。
    """

    def test_degradation_triggers_health_alerts(self):
        """A→B→C→D→F 序列应在 D/F 阶段触发健康度告警。"""
        seq = health_degradation_sequence(["A", "B", "C", "D", "F"])
        result = PerceptionReplayHarness(agi_cfg={}).replay(seq)

        triggered = set(result["alerts_by_type"].keys()) & HEALTH_ALERT_TYPES
        assert len(triggered) > 0, (
            f"未触发健康度告警，实际 alerts: {result['alerts_by_type']}"
        )

    def test_healthy_no_trigger(self):
        """全 A 级时不触发健康度告警。"""
        seq = health_degradation_sequence(["A", "A", "A", "A", "A"])
        result = PerceptionReplayHarness(agi_cfg={}).replay(seq)

        triggered = set(result["alerts_by_type"].keys()) & HEALTH_ALERT_TYPES
        assert len(triggered) == 0, f"健康度 A 级不应触发告警，实际: {triggered}"

    def test_ab_comparison(self):
        """A/B：F 级劣化 vs 全 A 级健康。"""
        seq_degraded = health_degradation_sequence(["A", "B", "C", "D", "F"])
        seq_healthy = health_degradation_sequence(["A", "A", "A", "A", "A"])
        result_degraded = PerceptionReplayHarness(agi_cfg={}).replay(seq_degraded)
        result_healthy = PerceptionReplayHarness(agi_cfg={}).replay(seq_healthy)

        degraded_health = set(result_degraded["alerts_by_type"].keys()) & HEALTH_ALERT_TYPES
        healthy_health = set(result_healthy["alerts_by_type"].keys()) & HEALTH_ALERT_TYPES
        assert len(degraded_health) > 0
        assert len(healthy_health) == 0
        assert result_degraded["total_alerts"] > result_healthy["total_alerts"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
