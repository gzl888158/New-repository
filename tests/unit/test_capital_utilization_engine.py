"""CapitalUtilizationEngine 企业级资金利用率引擎 — 正式单元测试。

覆盖矩阵：
  P0  初始化 & 配置解析
  P0  利用率分级 (UtilizationTier)
  P0  时段检测 (_detect_session)
  P0  波动率分级 (_update_context)
  P0  五因子动态目标 (_compute_dynamic_target)
  P0  策略级目标 (_compute_strategy_targets)
  P0  资本效率 / 趋势 / 波动率 (_calc_*)
  P0  动作决策 (_determine_action / _get_raw_action)
  P0  迟滞防振荡 (_is_oscillating / 振荡计数)
  P0  策略级动作 (_compute_strategy_actions)
  P0  仓位乘数 (_compute_position_boost)
  P0  信号放松 (_compute_signal_relaxation)
  P0  分配偏移 (_compute_allocation_shift)
  P1  analyze() 集成管线
  P1  持久化 (_save_state / _load_state)
  P1  公共 API (get_* / reset_oscillation_state)
  P2  边界/异常 (零权益、负值、空输入)
"""

import json
import math
import os
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from core.capital_utilization_engine import (
    CapitalUtilizationEngine,
    UtilizationAction,
    UtilizationReport,
    UtilizationSnapshot,
    UtilizationTier,
)


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _make_engine(account_tier: str = "nano", **kwargs) -> CapitalUtilizationEngine:
    cfg = {"trading": {
        "target_utilization": kwargs.pop("target_utilization", 0.85),
        "min_utilization": kwargs.pop("min_utilization", 0.50),
    }}
    cfg["trading"].update(kwargs)
    eng = CapitalUtilizationEngine(cfg)
    eng._account_tier = account_tier
    return eng


def _make_mock_equity_monitor(mode: str = "normal", tier: str = "nano"):
    m = MagicMock()
    m.get_equity_status = MagicMock(return_value={"mode": mode, "account_tier": tier})
    return m


# ═══════════════════════════════════════════════════════════════
# P0: 初始化 & 配置
# ═══════════════════════════════════════════════════════════════

class TestInit:
    def test_default_config(self):
        eng = CapitalUtilizationEngine()
        assert eng._base_target == 0.85
        assert eng._min_utilization == 0.50
        assert eng._max_utilization == 0.95
        assert eng._hysteresis_band == 0.05
        assert eng._action_cooldown == 120
        assert eng._max_oscillations == 3
        assert len(eng._snapshots) == 0

    def test_custom_config(self):
        eng = _make_engine("micro", target_utilization=0.80, min_utilization=0.40)
        assert eng._base_target == 0.80
        assert eng._min_utilization == 0.40
        assert eng._account_tier == "micro"

    def test_config_none(self):
        eng = CapitalUtilizationEngine(None)
        assert eng._base_target == 0.85

    def test_config_empty(self):
        eng = CapitalUtilizationEngine({})
        assert eng._base_target == 0.85

    def test_config_no_trading_key(self):
        eng = CapitalUtilizationEngine({"other": {}})
        assert eng._base_target == 0.85

    def test_dependency_injection(self):
        eng = _make_engine()
        assert eng._equity_monitor is None
        assert eng._regime_engine is None

        mock_em = _make_mock_equity_monitor()
        eng.set_equity_monitor(mock_em)
        assert eng._equity_monitor is mock_em

        mock_re = MagicMock()
        eng.set_regime_engine(mock_re)
        assert eng._regime_engine is mock_re


# ═══════════════════════════════════════════════════════════════
# P0: UtilizationTier 分级
# ═══════════════════════════════════════════════════════════════

class TestUtilizationTier:
    @pytest.mark.parametrize("util,expected", [
        (0.00, UtilizationTier.CRITICAL_LOW),
        (0.05, UtilizationTier.CRITICAL_LOW),
        (0.099, UtilizationTier.CRITICAL_LOW),
        (0.10, UtilizationTier.LOW),
        (0.20, UtilizationTier.LOW),
        (0.299, UtilizationTier.LOW),
        (0.30, UtilizationTier.MODERATE_LOW),
        (0.40, UtilizationTier.MODERATE_LOW),
        (0.499, UtilizationTier.MODERATE_LOW),
        (0.50, UtilizationTier.OPTIMAL),
        (0.70, UtilizationTier.OPTIMAL),
        (0.95, UtilizationTier.OPTIMAL),
        (0.96, UtilizationTier.HIGH),
        (0.98, UtilizationTier.HIGH),
        (0.99, UtilizationTier.CRITICAL_HIGH),
        (1.00, UtilizationTier.CRITICAL_HIGH),
    ])
    def test_classify_boundaries(self, util, expected):
        eng = _make_engine()
        assert eng._classify_utilization(util) == expected

    def test_respects_min_utilization_config(self):
        eng = _make_engine(min_utilization=0.60)
        assert eng._classify_utilization(0.55) == UtilizationTier.MODERATE_LOW
        assert eng._classify_utilization(0.60) == UtilizationTier.OPTIMAL
        # 0.30 >= 0.30 → MODERATE_LOW bound (original min would be 0.50→OPTIMAL)
        assert eng._classify_utilization(0.29) == UtilizationTier.LOW

    def test_all_tiers_have_distinct_values(self):
        values = set(t.value for t in UtilizationTier)
        assert len(values) == len(UtilizationTier)


# ═══════════════════════════════════════════════════════════════
# P0: 时段检测
# ═══════════════════════════════════════════════════════════════

class TestSessionDetection:
    @pytest.mark.parametrize("hour,weekday,expected", [
        (0, 0, "asian"), (3, 2, "asian"), (7, 4, "asian"),
        (8, 1, "european"), (10, 3, "european"), (12, 0, "european"),
        (13, 0, "overlap_eu_us"), (14, 2, "overlap_eu_us"), (15, 4, "overlap_eu_us"),
        (16, 1, "us"), (18, 3, "us"), (19, 0, "us"),
        (20, 0, "low_liquidity"), (22, 2, "low_liquidity"), (23, 4, "low_liquidity"),
    ])
    def test_weekday_sessions(self, hour, weekday, expected):
        eng = _make_engine()
        mock_dt = datetime(2026, 8, 17 + weekday, hour, 0, 0, tzinfo=timezone.utc)
        with patch("core.capital_utilization_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.timezone = timezone
            result = eng._detect_session()
        assert result == expected

    @pytest.mark.parametrize("hour", [0, 8, 13, 16, 20])
    def test_weekend(self, hour):
        eng = _make_engine()
        # 2026-08-22 is Saturday
        mock_dt = datetime(2026, 8, 22, hour, 0, 0, tzinfo=timezone.utc)
        with patch("core.capital_utilization_engine.datetime") as mock_datetime:
            mock_datetime.now.return_value = mock_dt
            mock_datetime.timezone = timezone
            result = eng._detect_session()
        assert result == "weekend"

    def test_explicit_session_overrides_auto(self):
        eng = _make_engine()
        eng._update_context(1.0, session="weekend")
        assert eng._current_session == "weekend"


# ═══════════════════════════════════════════════════════════════
# P0: 波动率分级
# ═══════════════════════════════════════════════════════════════

class TestVolatilityRegime:
    @pytest.mark.parametrize("atr_ratio,expected", [
        (0.2, "very_low"),
        (0.49, "very_low"),
        (0.50, "low"),
        (0.70, "low"),
        (0.79, "low"),
        (0.80, "normal"),
        (1.00, "normal"),
        (1.19, "normal"),
        (1.20, "high"),
        (1.50, "high"),
        (1.99, "high"),
        (2.00, "very_high"),
        (3.00, "very_high"),
    ])
    def test_regime_boundaries(self, atr_ratio, expected):
        eng = _make_engine()
        eng._update_context(atr_ratio)
        assert eng._volatility_regime == expected

    def test_equity_monitor_integration(self):
        eng = _make_engine()
        eng.set_equity_monitor(_make_mock_equity_monitor("growth", "small"))
        eng._update_context(1.0)
        assert eng._equity_mode == "growth"
        assert eng._account_tier == "small"

    def test_no_equity_monitor_no_crash(self):
        eng = _make_engine()
        eng._update_context(1.0)
        assert eng._equity_mode == "normal"
        assert eng._account_tier == "nano"


# ═══════════════════════════════════════════════════════════════
# P0: 五因子动态目标
# ═══════════════════════════════════════════════════════════════

class TestDynamicTarget:
    def test_normal_conditions(self):
        eng = _make_engine()
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        target = eng._compute_dynamic_target(0.5, 1.0, 0.0)
        # 0.85 * 1.0 * 1.0 * 1.0 * 1.0 = 0.85 → nano cap 0.75
        assert target == pytest.approx(0.75)

    def test_high_volatility_reduces(self):
        eng = _make_engine("small")
        eng._volatility_regime = "very_high"
        eng._current_session = "european"
        target = eng._compute_dynamic_target(0.5, 2.5, 0.0)
        # 0.85 * 0.55 * 1.0 * 1.0 * 1.0 = 0.4675 → small cap 0.85 (no constraint)
        assert target == pytest.approx(0.4675)

    def test_weekend_reduces(self):
        eng = _make_engine("small")
        eng._volatility_regime = "normal"
        eng._current_session = "weekend"
        target = eng._compute_dynamic_target(0.5, 1.0, 0.0)
        # 0.85 * 1.0 * 0.50 * 1.0 * 1.0 = 0.425
        assert target == pytest.approx(0.425)

    def test_drawdown_reduces(self):
        eng = _make_engine("small")
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        target = eng._compute_dynamic_target(0.5, 1.0, 0.10)
        # 0.85 * 1.0 * 1.0 * 1.0 * (1-0.3) = 0.595
        assert target == pytest.approx(0.595)

    def test_drawdown_floor(self):
        eng = _make_engine("small")
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        target = eng._compute_dynamic_target(0.5, 1.0, 0.30)
        # 0.85 * 1.0 * 1.0 * 1.0 * max(0.3, 1-0.9)=0.3 = 0.255
        assert target == pytest.approx(0.255)

    def test_emergency_zero(self):
        eng = _make_engine()
        eng._equity_mode = "emergency"
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        target = eng._compute_dynamic_target(0.5, 1.0, 0.0)
        # 0.85 * 0.0 (emergency) = 0.0 → floor max(0.10, 0.0) = 0.10
        assert target == pytest.approx(0.10)

    def test_target_floor(self):
        eng = _make_engine()
        eng._volatility_regime = "very_high"
        eng._current_session = "weekend"
        eng._equity_mode = "decline"
        target = eng._compute_dynamic_target(0.5, 3.0, 0.25)
        # 0.85 * 0.55 * 0.50 * 0.60 * 0.25 = 0.035 → floor 0.10
        assert target == pytest.approx(0.10)

    def test_target_ceiling(self):
        eng = _make_engine("xlarge")
        eng._volatility_regime = "very_low"
        eng._current_session = "asian"
        eng._equity_mode = "growth"
        target = eng._compute_dynamic_target(0.5, 0.3, 0.0)
        # 0.85 * 1.20 * 1.10 * 1.15 = 1.2903 → xlarge cap 0.95
        assert target == pytest.approx(0.95)

    def test_tier_caps_are_applied(self):
        eng = _make_engine("nano")
        eng._volatility_regime = "very_low"
        eng._current_session = "asian"
        target = eng._compute_dynamic_target(0.5, 0.3, 0.0)
        # 0.85 * 1.20 * 1.10 = 1.122 → nano cap 0.75
        assert target == pytest.approx(0.75)

    def test_growth_mode_boosts(self):
        eng = _make_engine("xlarge")
        eng._equity_mode = "growth"
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        target = eng._compute_dynamic_target(0.5, 1.0, 0.0)
        # 0.85 * 1.0 * 1.0 * 1.15 = 0.9775 → xlarge cap 0.95
        assert target == pytest.approx(0.95)


# ═══════════════════════════════════════════════════════════════
# P0: 策略级目标
# ═══════════════════════════════════════════════════════════════

class TestStrategyTargets:
    def test_respond_to_drawdown(self):
        eng = _make_engine("nano")
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        eng._equity_mode = "normal"

        t0 = eng._compute_strategy_targets(1.0, 0.0)
        t20 = eng._compute_strategy_targets(1.0, 0.20)

        assert t0["trend"] == pytest.approx(0.60)
        assert t20["trend"] == pytest.approx(0.24)

    def test_respect_tier_cap(self):
        eng = _make_engine("nano")
        eng._volatility_regime = "very_low"
        eng._current_session = "asian"
        eng._equity_mode = "growth"

        t = eng._compute_strategy_targets(0.3, 0.0)
        assert t["grid"] == pytest.approx(0.75)
        assert all(v <= 0.75 for v in t.values())

    def test_all_six_strategies_present(self):
        eng = _make_engine("nano")
        t = eng._compute_strategy_targets(1.0, 0.0)
        for sname in ("grid", "scalping", "arbitrage", "trend", "spot_grid", "spot_martingale"):
            assert sname in t

    def test_base_targets_ordering(self):
        """spot_martingale < trend < spot_grid < arbitrage < scalping < grid"""
        eng = _make_engine("small")
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        eng._equity_mode = "normal"
        t = eng._compute_strategy_targets(1.0, 0.0)
        assert t["spot_martingale"] < t["trend"] < t["spot_grid"] < t["arbitrage"] < t["scalping"] < t["grid"]

    def test_volatility_affects_all_strategies(self):
        eng = _make_engine("small")
        eng._current_session = "european"
        eng._equity_mode = "normal"

        eng._volatility_regime = "normal"
        t_normal = eng._compute_strategy_targets(1.0, 0.0)

        eng._volatility_regime = "very_high"
        t_high = eng._compute_strategy_targets(2.5, 0.0)

        # All strategies should be lower in high volatility
        for sname in t_normal:
            assert t_high[sname] < t_normal[sname]

    def test_session_affects_all_strategies(self):
        eng = _make_engine("small")
        eng._volatility_regime = "normal"
        eng._equity_mode = "normal"

        eng._current_session = "asian"
        t_asian = eng._compute_strategy_targets(1.0, 0.0)

        eng._current_session = "weekend"
        t_weekend = eng._compute_strategy_targets(1.0, 0.0)

        for sname in t_asian:
            assert t_weekend[sname] < t_asian[sname]

    def test_equity_mode_affects_all_strategies(self):
        eng = _make_engine("small")
        eng._volatility_regime = "normal"
        eng._current_session = "european"

        eng._equity_mode = "growth"
        t_growth = eng._compute_strategy_targets(1.0, 0.0)

        eng._equity_mode = "decline"
        t_decline = eng._compute_strategy_targets(1.0, 0.0)

        for sname in t_growth:
            assert t_decline[sname] < t_growth[sname]

    def test_strategy_target_floor(self):
        eng = _make_engine("small")
        eng._volatility_regime = "very_high"
        eng._current_session = "weekend"
        eng._equity_mode = "decline"
        t = eng._compute_strategy_targets(3.0, 0.25)
        for sname, v in t.items():
            assert v >= 0.05, f"{sname} target {v} below floor 0.05"


# ═══════════════════════════════════════════════════════════════
# P0: 资本效率 / 趋势 / 波动率
# ═══════════════════════════════════════════════════════════════

class TestCapitalEfficiency:
    def test_positive_efficiency(self):
        eng = _make_engine()
        result = eng._calc_capital_efficiency(
            {"grid": 5.0, "trend": -2.0}, {"grid": 50.0, "trend": 30.0}
        )
        assert result == pytest.approx(3.0 / 80.0)

    def test_zero_usage(self):
        eng = _make_engine()
        result = eng._calc_capital_efficiency({"grid": 5.0}, {})
        assert result == 0.0

    def test_all_zero_usage(self):
        eng = _make_engine()
        result = eng._calc_capital_efficiency({"grid": 5.0}, {"grid": 0.0, "trend": 0.0})
        assert result == 0.0

    def test_negative_efficiency(self):
        eng = _make_engine()
        result = eng._calc_capital_efficiency(
            {"grid": -10.0}, {"grid": 50.0}
        )
        assert result == pytest.approx(-0.2)

    def test_empty_inputs(self):
        eng = _make_engine()
        assert eng._calc_capital_efficiency({}, {}) == 0.0


class TestUtilizationTrend:
    def test_empty_history(self):
        eng = _make_engine()
        assert eng._calc_utilization_trend(0.5) == 0.0

    def test_insufficient_history(self):
        eng = _make_engine()
        for u in [0.5, 0.5, 0.5, 0.5]:
            eng._utilization_history.append(u)
        assert eng._calc_utilization_trend(0.5) == 0.0

    def test_flat_trend(self):
        eng = _make_engine()
        for _ in range(10):
            eng._utilization_history.append(0.5)
        assert eng._calc_utilization_trend(0.5) == pytest.approx(0.0, abs=1e-10)

    def test_positive_trend(self):
        eng = _make_engine()
        for i in range(20):
            eng._utilization_history.append(0.3 + i * 0.01)
        trend = eng._calc_utilization_trend(0.5)
        assert trend > 0.001

    def test_negative_trend(self):
        eng = _make_engine()
        for i in range(20):
            eng._utilization_history.append(0.7 - i * 0.01)
        trend = eng._calc_utilization_trend(0.3)
        assert trend < -0.001

    def test_uses_max_20_points(self):
        eng = _make_engine()
        for i in range(50):
            eng._utilization_history.append(0.3 + i * 0.01)
        trend = eng._calc_utilization_trend(0.8)
        assert trend > 0.001  # still positive from last 20


class TestUtilizationVolatility:
    def test_empty_history(self):
        eng = _make_engine()
        assert eng._calc_utilization_volatility() == 0.0

    def test_flat_no_volatility(self):
        eng = _make_engine()
        for _ in range(10):
            eng._utilization_history.append(0.5)
        assert eng._calc_utilization_volatility() == pytest.approx(0.0, abs=1e-10)

    def test_high_volatility(self):
        eng = _make_engine()
        for v in [0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]:
            eng._utilization_history.append(v)
        vol = eng._calc_utilization_volatility()
        assert vol > 0.15


# ═══════════════════════════════════════════════════════════════
# P0: 基础动作决策
# ═══════════════════════════════════════════════════════════════

class TestGetRawAction:
    @pytest.mark.parametrize("tier,eff,trend,expected", [
        (UtilizationTier.CRITICAL_LOW, 0.0, 0.0, UtilizationAction.BOOST_AGGRESSIVE),
        (UtilizationTier.LOW, 0.0, 0.0, UtilizationAction.BOOST_MODERATE),
        (UtilizationTier.MODERATE_LOW, 0.05, 0.0, UtilizationAction.BOOST_MODERATE),
        (UtilizationTier.MODERATE_LOW, -0.01, 0.0, UtilizationAction.BOOST_CONSERVATIVE),
        (UtilizationTier.OPTIMAL, 0.0, 0.0, UtilizationAction.HOLD),
        (UtilizationTier.HIGH, 0.0, 0.0, UtilizationAction.HOLD),
        (UtilizationTier.HIGH, 0.0, 0.01, UtilizationAction.REDUCE_CONSERVATIVE),
        (UtilizationTier.CRITICAL_HIGH, 0.0, 0.0, UtilizationAction.REDUCE_AGGRESSIVE),
    ])
    def test_all_tiers(self, tier, eff, trend, expected):
        eng = _make_engine()
        assert eng._get_raw_action(tier, eff, trend) == expected

    def test_moderate_low_efficiency_boundary(self):
        eng = _make_engine()
        # efficiency > 0.01 → BOOST_MODERATE; efficiency == 0.01 → BOOST_CONSERVATIVE
        assert eng._get_raw_action(UtilizationTier.MODERATE_LOW, 0.01, 0.0) == UtilizationAction.BOOST_CONSERVATIVE
        assert eng._get_raw_action(UtilizationTier.MODERATE_LOW, 0.02, 0.0) == UtilizationAction.BOOST_MODERATE

    def test_high_trend_boundary(self):
        eng = _make_engine()
        # trend > 0.001 → REDUCE_CONSERVATIVE; trend == 0.001 → HOLD
        assert eng._get_raw_action(UtilizationTier.HIGH, 0.0, 0.001) == UtilizationAction.HOLD
        assert eng._get_raw_action(UtilizationTier.HIGH, 0.0, 0.002) == UtilizationAction.REDUCE_CONSERVATIVE


# ═══════════════════════════════════════════════════════════════
# P0: 振荡检测
# ═══════════════════════════════════════════════════════════════

class TestOscillation:
    def _set_last(self, eng, action):
        eng._last_action = action
        eng._last_action_time = datetime.now()

    def test_boost_to_reduce_oscillates(self):
        eng = _make_engine()
        self._set_last(eng, UtilizationAction.BOOST_AGGRESSIVE)
        assert eng._is_oscillating(UtilizationAction.REDUCE_CONSERVATIVE) is True

    def test_reduce_to_boost_oscillates(self):
        eng = _make_engine()
        self._set_last(eng, UtilizationAction.REDUCE_MODERATE)
        assert eng._is_oscillating(UtilizationAction.BOOST_MODERATE) is True

    def test_hold_not_oscillating(self):
        eng = _make_engine()
        self._set_last(eng, UtilizationAction.BOOST_MODERATE)
        assert eng._is_oscillating(UtilizationAction.HOLD) is False

    def test_same_family_not_oscillating(self):
        eng = _make_engine()
        self._set_last(eng, UtilizationAction.BOOST_CONSERVATIVE)
        assert eng._is_oscillating(UtilizationAction.BOOST_AGGRESSIVE) is False

    def test_freeze_not_oscillating(self):
        eng = _make_engine()
        self._set_last(eng, UtilizationAction.BOOST_MODERATE)
        assert eng._is_oscillating(UtilizationAction.EMERGENCY_FREEZE) is False

    def test_oscillation_counter_increments(self):
        eng = _make_engine()
        eng._dynamic_target = 0.50
        eng._volatility_regime = "normal"
        eng._current_session = "european"

        # First: CRITICAL_LOW → BOOST_AGGRESSIVE (gap=-0.30, not < -0.35 so no hard constraint intercept)
        eng._determine_action(0.20, UtilizationTier.CRITICAL_LOW, 0.0, 0.0)
        assert eng._last_action == UtilizationAction.BOOST_AGGRESSIVE
        # Clear cooldown by adjusting time
        eng._last_action_time = datetime(2020, 1, 1)

        # Then: CRITICAL_HIGH → REDUCE_AGGRESSIVE (opposite direction → oscillation)
        # gap=0.99-0.50=0.49 > 0.20 → hard constraint REDUCE_MODERATE
        # Need to avoid hard constraint: use utilization=0.99, dynamic_target=0.95
        eng._dynamic_target = 0.95
        eng._determine_action(0.99, UtilizationTier.CRITICAL_HIGH, 0.0, 0.0)
        assert eng._oscillation_counter == 1

    def test_oscillation_lock(self):
        eng = _make_engine()
        eng._oscillation_counter = 3
        eng._max_oscillations = 3
        eng._last_action = UtilizationAction.BOOST_AGGRESSIVE
        eng._last_action_time = datetime(2020, 1, 1)
        eng._dynamic_target = 0.95  # gap=0.04, not > 0.10 → no hard constraint intercept

        # CRITICAL_HIGH → REDUCE_AGGRESSIVE raw, oscillating → locked to HOLD
        action = eng._determine_action(0.99, UtilizationTier.CRITICAL_HIGH, 0.0, 0.0)
        assert action == UtilizationAction.HOLD
        assert eng._action_cooldown == 240  # doubled from 120


# ═══════════════════════════════════════════════════════════════
# P0: 动作决策（迟滞 + 相对目标 + 紧急）
# ═══════════════════════════════════════════════════════════════

class TestDetermineAction:
    def test_emergency_always_freeze(self):
        eng = _make_engine()
        eng._equity_mode = "emergency"
        action = eng._determine_action(0.50, UtilizationTier.OPTIMAL, 0.0, 0.0)
        assert action == UtilizationAction.EMERGENCY_FREEZE

    def test_relative_target_hard_constraint_over_20(self):
        eng = _make_engine()
        eng._dynamic_target = 0.50
        action = eng._determine_action(0.80, UtilizationTier.OPTIMAL, 0.0, 0.0)
        assert action == UtilizationAction.REDUCE_MODERATE

    def test_relative_target_hard_constraint_over_10_high_abs(self):
        eng = _make_engine()
        eng._dynamic_target = 0.70
        action = eng._determine_action(0.82, UtilizationTier.OPTIMAL, 0.0, 0.0)
        # gap=0.12 > 0.10 and utilization=0.82 > 0.80
        assert action == UtilizationAction.REDUCE_CONSERVATIVE

    def test_relative_target_hard_constraint_under_35(self):
        eng = _make_engine()
        eng._dynamic_target = 0.75
        action = eng._determine_action(0.30, UtilizationTier.LOW, 0.0, 0.0)
        # gap = -0.45 < -0.35
        assert action == UtilizationAction.BOOST_AGGRESSIVE

    def test_cooldown_returns_last_action(self):
        eng = _make_engine()
        eng._dynamic_target = 0.75
        eng._last_action = UtilizationAction.BOOST_MODERATE
        eng._last_action_time = datetime.now()  # just now → within cooldown
        eng._equity_mode = "normal"

        action = eng._determine_action(0.50, UtilizationTier.MODERATE_LOW, 0.0, 0.0)
        assert action == UtilizationAction.BOOST_MODERATE

    def test_no_previous_action_returns_raw(self):
        eng = _make_engine()
        eng._dynamic_target = 0.75
        eng._last_action = None
        eng._last_action_time = None

        action = eng._determine_action(0.05, UtilizationTier.CRITICAL_LOW, 0.0, 0.0)
        assert action == UtilizationAction.BOOST_AGGRESSIVE


# ═══════════════════════════════════════════════════════════════
# P0: 策略级动作
# ═══════════════════════════════════════════════════════════════

class TestStrategyActions:
    def test_gap_above_20_boosts_moderate(self):
        eng = _make_engine()
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        targets = eng._compute_strategy_targets(1.0, 0.0)
        # grid target = 0.80, usage = 10 on equity 100 → strat_util = 0.10
        actions = eng._compute_strategy_actions(
            {"grid": 10.0}, targets, {"grid": 0.5}, 100.0
        )
        assert actions["grid"] == UtilizationAction.BOOST_MODERATE

    def test_gap_5_to_20_boosts_conservative(self):
        eng = _make_engine()
        targets = {"grid": 0.80}
        actions = eng._compute_strategy_actions(
            {"grid": 70.0}, targets, {"grid": 0.5}, 100.0
        )
        # strat_util = 0.70, target = 0.80, gap = 0.10
        assert actions["grid"] == UtilizationAction.BOOST_CONSERVATIVE

    def test_over_target_negative_pnl_reduces_moderate(self):
        eng = _make_engine()
        targets = {"grid": 0.10}
        actions = eng._compute_strategy_actions(
            {"grid": 30.0}, targets, {"grid": -0.5}, 100.0
        )
        # strat_util = 0.30, target = 0.10, gap = -0.20 < -0.10
        assert actions["grid"] == UtilizationAction.REDUCE_MODERATE

    def test_over_target_positive_pnl_reduces_conservative(self):
        eng = _make_engine()
        targets = {"grid": 0.10}
        actions = eng._compute_strategy_actions(
            {"grid": 30.0}, targets, {"grid": 0.5}, 100.0
        )
        assert actions["grid"] == UtilizationAction.REDUCE_CONSERVATIVE

    def test_within_range_holds(self):
        eng = _make_engine()
        targets = {"grid": 0.80}
        # usage=76 on equity=100 → strat_util=0.76, target=0.80 → gap=0.04 (not > 0.05, not < -0.10)
        actions = eng._compute_strategy_actions(
            {"grid": 76.0}, targets, {"grid": 0.5}, 100.0
        )
        assert actions["grid"] == UtilizationAction.HOLD

    def test_zero_equity_returns_empty(self):
        eng = _make_engine()
        actions = eng._compute_strategy_actions(
            {"grid": 10.0}, {"grid": 0.8}, {}, 0.0
        )
        assert actions == {}

    def test_missing_target_defaults(self):
        eng = _make_engine()
        actions = eng._compute_strategy_actions(
            {"unknown": 10.0}, {}, {"unknown": 0.5}, 100.0
        )
        # default target = 0.50, gap = 0.40 > 0.20 → BOOST_MODERATE
        assert actions["unknown"] == UtilizationAction.BOOST_MODERATE


# ═══════════════════════════════════════════════════════════════
# P0: 仓位乘数
# ═══════════════════════════════════════════════════════════════

class TestPositionBoost:
    def test_emergency_freeze_zero(self):
        eng = _make_engine()
        boost = eng._compute_position_boost(0.5, UtilizationTier.OPTIMAL, UtilizationAction.EMERGENCY_FREEZE)
        assert boost == 0.0

    def test_hold_is_one(self):
        eng = _make_engine()
        boost = eng._compute_position_boost(0.5, UtilizationTier.OPTIMAL, UtilizationAction.HOLD)
        assert boost == 1.0

    def test_reduce_actions(self):
        eng = _make_engine()
        assert eng._compute_position_boost(0.5, UtilizationTier.HIGH, UtilizationAction.REDUCE_CONSERVATIVE) == 0.85
        assert eng._compute_position_boost(0.5, UtilizationTier.HIGH, UtilizationAction.REDUCE_MODERATE) == 0.70
        assert eng._compute_position_boost(0.5, UtilizationTier.HIGH, UtilizationAction.REDUCE_AGGRESSIVE) == 0.50

    def test_boost_no_gap_returns_one(self):
        eng = _make_engine()
        eng._dynamic_target = 0.50
        boost = eng._compute_position_boost(0.70, UtilizationTier.OPTIMAL, UtilizationAction.BOOST_CONSERVATIVE)
        # gap = max(0, 0.50 - 0.70) = 0 → boost = 1.0
        assert boost == 1.0

    def test_boost_sigmoid_shape(self):
        eng = _make_engine("small")
        eng._dynamic_target = 0.75
        # gap = 0.75 - 0.20 = 0.55 → sigmoid should give significant boost
        boost = eng._compute_position_boost(0.20, UtilizationTier.LOW, UtilizationAction.BOOST_AGGRESSIVE)
        assert 2.0 < boost <= 6.0

    def test_boost_respects_tier_cap(self):
        eng = _make_engine("nano")
        eng._dynamic_target = 0.75
        boost = eng._compute_position_boost(0.05, UtilizationTier.CRITICAL_LOW, UtilizationAction.BOOST_AGGRESSIVE)
        # nano cap = 1.5 * 1.2 = 1.8
        assert boost <= 1.8

    def test_boost_conservative_caps_lower(self):
        eng = _make_engine("small")
        eng._dynamic_target = 0.75
        boost = eng._compute_position_boost(0.10, UtilizationTier.LOW, UtilizationAction.BOOST_CONSERVATIVE)
        # small cap = 3.0 * 0.7 = 2.1
        assert boost <= 2.1

    def test_boost_floor(self):
        eng = _make_engine()
        eng._dynamic_target = 0.75
        boost = eng._compute_position_boost(0.50, UtilizationTier.OPTIMAL, UtilizationAction.BOOST_CONSERVATIVE)
        # gap = 0.25 → sigmoid ~1.5, but conservative cap may clip. floor = 0.5
        assert boost >= 0.5


# ═══════════════════════════════════════════════════════════════
# P0: 信号放松
# ═══════════════════════════════════════════════════════════════

class TestSignalRelaxation:
    def test_emergency_tightens(self):
        eng = _make_engine()
        rel = eng._compute_signal_relaxation(0.5, UtilizationTier.OPTIMAL, UtilizationAction.EMERGENCY_FREEZE)
        assert rel == -0.30

    def test_high_utilization_tightens(self):
        eng = _make_engine()
        rel = eng._compute_signal_relaxation(0.97, UtilizationTier.HIGH, UtilizationAction.REDUCE_CONSERVATIVE)
        assert rel == -0.10

    def test_critical_high_tightens(self):
        eng = _make_engine()
        rel = eng._compute_signal_relaxation(0.99, UtilizationTier.CRITICAL_HIGH, UtilizationAction.REDUCE_AGGRESSIVE)
        assert rel == -0.10

    def test_small_gap_no_relaxation(self):
        eng = _make_engine()
        eng._dynamic_target = 0.50
        rel = eng._compute_signal_relaxation(0.49, UtilizationTier.MODERATE_LOW, UtilizationAction.BOOST_CONSERVATIVE)
        # gap = 0.01 <= 0.02 → 0.0
        assert rel == 0.0

    def test_large_gap_relaxes(self):
        eng = _make_engine()
        eng._dynamic_target = 0.75
        rel = eng._compute_signal_relaxation(0.15, UtilizationTier.LOW, UtilizationAction.BOOST_AGGRESSIVE)
        # gap = 0.60 → 0.60 * 0.30 = 0.18 → capped 0.15
        assert rel == pytest.approx(0.15)

    def test_moderate_gap_relaxes(self):
        eng = _make_engine()
        eng._dynamic_target = 0.75
        rel = eng._compute_signal_relaxation(0.45, UtilizationTier.MODERATE_LOW, UtilizationAction.BOOST_MODERATE)
        # gap = 0.30 → 0.30 * 0.30 = 0.09
        assert rel == pytest.approx(0.09)


# ═══════════════════════════════════════════════════════════════
# P0: 分配偏移
# ═══════════════════════════════════════════════════════════════

class TestAllocationShift:
    def test_emergency_all_zeros(self):
        eng = _make_engine()
        shifts = eng._compute_allocation_shift(
            {"grid": 10}, {"grid": 0.8, "trend": 0.6}, {"grid": 0.1}, UtilizationAction.EMERGENCY_FREEZE
        )
        assert shifts == {"grid": 0.0, "trend": 0.0}

    def test_positive_pnl_amplifies(self):
        eng = _make_engine()
        eng._volatility_regime = "normal"
        eng._current_session = "european"
        shifts = eng._compute_allocation_shift(
            {"grid": 30}, {"grid": 0.80}, {"grid": 0.5}, UtilizationAction.HOLD
        )
        # 30/30=1.0 current, target=0.80 → raw_shift=(0.80-1.0)*0.3=-0.06 * 1.2 = -0.072
        assert shifts["grid"] == pytest.approx(-0.072)

    def test_negative_pnl_dampens(self):
        eng = _make_engine()
        shifts = eng._compute_allocation_shift(
            {"grid": 30}, {"grid": 0.80}, {"grid": -0.5}, UtilizationAction.HOLD
        )
        # raw_shift = -0.06 * 0.5 = -0.03
        assert shifts["grid"] == pytest.approx(-0.03)

    def test_shift_bounded(self):
        eng = _make_engine()
        shifts = eng._compute_allocation_shift(
            {"grid": 10}, {"grid": 0.95}, {"grid": 0.5}, UtilizationAction.HOLD
        )
        # 10/10=1.0 current, target=0.95 → raw=(0.95-1.0)*0.3*1.2 = -0.018
        assert -0.10 <= shifts["grid"] <= 0.10

    def test_empty_usage_handles_gracefully(self):
        eng = _make_engine()
        # _compute_allocation_shift iterates over strategy_targets, not strategy_usage.
        # With empty usage, each target gets a shift.
        shifts = eng._compute_allocation_shift(
            {}, {"grid": 0.8}, {}, UtilizationAction.HOLD
        )
        # grid not in usage → current=0, target=0.8 → raw_shift=(0.8-0)*0.3=0.24 → bounded ±0.10
        assert "grid" in shifts
        assert shifts["grid"] == pytest.approx(0.10)


# ═══════════════════════════════════════════════════════════════
# P1: analyze() 集成管线
# ═══════════════════════════════════════════════════════════════

class TestAnalyze:
    def test_low_utilization_boosts(self):
        eng = _make_engine("nano")
        rep = eng.analyze(
            total_equity=100.0, used_margin=15.0, available=85.0,
            strategy_usage={"grid": 15.0}, recent_pnl={"grid": 0.5},
            atr_ratio=1.0, total_drawdown_pct=0.0, session="european",
        )
        assert rep.current_utilization == pytest.approx(0.15)
        assert rep.utilization_tier == UtilizationTier.LOW
        assert rep.recommended_action == UtilizationAction.BOOST_AGGRESSIVE
        assert rep.position_boost > 1.0

    def test_emergency_freezes_position_boost(self):
        eng = _make_engine("nano")
        eng._equity_mode = "emergency"
        rep = eng.analyze(
            total_equity=100.0, used_margin=50.0, available=50.0,
            atr_ratio=1.0, total_drawdown_pct=0.0,
        )
        assert rep.recommended_action == UtilizationAction.EMERGENCY_FREEZE
        assert rep.position_boost == 0.0

    def test_report_fields_populated(self):
        eng = _make_engine("nano")
        rep = eng.analyze(
            total_equity=56.67, used_margin=26.67, available=30.0,
            strategy_usage={"grid": 10, "trend": 8, "scalping": 5},
            recent_pnl={"grid": 0.2, "trend": -0.1, "scalping": 0.3},
            atr_ratio=1.2, total_drawdown_pct=0.02,
        )
        assert rep.strategy_targets
        assert rep.strategy_actions
        assert 0.0 <= rep.position_boost <= 6.0
        assert rep.account_tier == "nano"

    def test_summary_and_reset(self):
        eng = _make_engine("nano")
        eng.analyze(total_equity=100.0, used_margin=15.0, available=85.0, atr_ratio=1.0)
        summary = eng.get_utilization_summary()
        assert summary.get("available", True) is not False
        assert "current_utilization" in summary
        eng.reset_oscillation_state()
        assert eng.get_oscillation_status()["oscillation_counter"] == 0

    def test_optimal_utilization(self):
        eng = _make_engine("small")
        rep = eng.analyze(
            total_equity=100.0, used_margin=60.0, available=40.0,
            atr_ratio=1.0, total_drawdown_pct=0.0, session="european",
        )
        # 0.60 should be in OPTIMAL range (0.50-0.95)
        assert rep.utilization_tier == UtilizationTier.OPTIMAL
        assert rep.recommended_action == UtilizationAction.HOLD
        assert rep.position_boost == 1.0

    def test_high_utilization_reduces(self):
        eng = _make_engine("small")
        eng._volatility_regime = "very_high"
        eng._current_session = "weekend"
        rep = eng.analyze(
            total_equity=100.0, used_margin=97.0, available=3.0,
            atr_ratio=2.5, total_drawdown_pct=0.0,
        )
        assert rep.utilization_tier == UtilizationTier.HIGH
        assert rep.recommended_action in (
            UtilizationAction.REDUCE_CONSERVATIVE,
            UtilizationAction.REDUCE_MODERATE,
            UtilizationAction.REDUCE_AGGRESSIVE,
        )

    def test_snapshots_accumulated(self):
        eng = _make_engine()
        for i in range(5):
            eng.analyze(total_equity=100.0, used_margin=50.0, available=50.0, atr_ratio=1.0)
        assert len(eng._snapshots) == 5
        assert len(list(eng._reports)) == 5

    def test_strategy_history_accumulated(self):
        eng = _make_engine()
        eng.analyze(
            total_equity=100.0, used_margin=30.0, available=70.0,
            strategy_usage={"grid": 20, "trend": 10},
            recent_pnl={"grid": 0.5, "trend": -0.2},
            atr_ratio=1.0,
        )
        assert "grid" in eng._strategy_utilization_history
        assert "trend" in eng._strategy_utilization_history
        assert "grid" in eng._strategy_pnl_history
        assert "trend" in eng._strategy_pnl_history

    def test_high_volatility_reduces_target(self):
        eng = _make_engine("small")
        rep_normal = eng.analyze(
            total_equity=100.0, used_margin=50.0, available=50.0,
            atr_ratio=1.0, total_drawdown_pct=0.0, session="european",
        )
        rep_high = eng.analyze(
            total_equity=100.0, used_margin=50.0, available=50.0,
            atr_ratio=2.5, total_drawdown_pct=0.0, session="european",
        )
        assert rep_high.dynamic_target < rep_normal.dynamic_target

    def test_drawdown_reduces_target(self):
        eng = _make_engine("small")
        rep_no_dd = eng.analyze(
            total_equity=100.0, used_margin=50.0, available=50.0,
            atr_ratio=1.0, total_drawdown_pct=0.0, session="european",
        )
        rep_dd = eng.analyze(
            total_equity=100.0, used_margin=50.0, available=50.0,
            atr_ratio=1.0, total_drawdown_pct=0.15, session="european",
        )
        assert rep_dd.dynamic_target < rep_no_dd.dynamic_target

    def test_allocation_shift_populated(self):
        eng = _make_engine("small")
        rep = eng.analyze(
            total_equity=100.0, used_margin=50.0, available=50.0,
            strategy_usage={"grid": 30, "trend": 20},
            recent_pnl={"grid": 0.5, "trend": -0.2},
            atr_ratio=1.0, session="european",
        )
        assert "grid" in rep.allocation_shift
        assert "trend" in rep.allocation_shift


# ═══════════════════════════════════════════════════════════════
# P1: 持久化
# ═══════════════════════════════════════════════════════════════

class TestPersistence:
    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eng = _make_engine("small")
            eng._state_path = os.path.join(tmpdir, "state.json")
            eng._dynamic_target = 0.65
            eng._volatility_regime = "high"
            eng._current_session = "us"
            eng._equity_mode = "decline"
            eng._account_tier = "small"
            eng._oscillation_counter = 2
            eng._action_cooldown = 240
            eng._last_action = UtilizationAction.REDUCE_MODERATE
            eng._last_action_time = datetime(2026, 1, 1, 12, 0, 0)

            eng._save_state()
            assert os.path.exists(eng._state_path)

            # Load into new engine
            eng2 = _make_engine()
            eng2._state_path = eng._state_path
            eng2._load_state()

            assert eng2._dynamic_target == 0.65
            assert eng2._volatility_regime == "high"
            assert eng2._current_session == "us"
            assert eng2._equity_mode == "decline"
            assert eng2._account_tier == "small"
            assert eng2._oscillation_counter == 2
            assert eng2._action_cooldown == 240
            assert eng2._last_action == UtilizationAction.REDUCE_MODERATE
            assert eng2._last_action_time == datetime(2026, 1, 1, 12, 0, 0)

    def test_load_missing_file_no_error(self):
        eng = _make_engine()
        eng._state_path = os.path.join(tempfile.gettempdir(), "nonexistent_util_state.json")
        eng._load_state()
        assert eng._dynamic_target == 0.85  # default

    def test_load_corrupted_file_no_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.json")
            with open(path, "w") as f:
                f.write("{not valid json")

            eng = _make_engine()
            eng._state_path = path
            eng._load_state()
            assert eng._dynamic_target == 0.85  # default preserved

    def test_load_partial_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "partial.json")
            with open(path, "w") as f:
                json.dump({"dynamic_target": 0.55}, f)

            eng = _make_engine()
            eng._state_path = path
            eng._load_state()
            assert eng._dynamic_target == 0.55
            assert eng._volatility_regime == "normal"  # default
            assert eng._oscillation_counter == 0         # default

    def test_load_invalid_action_enum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad_action.json")
            with open(path, "w") as f:
                json.dump({"last_action": "not_a_valid_action"}, f)

            eng = _make_engine()
            eng._state_path = path
            eng._load_state()
            assert eng._last_action is None  # gracefully ignored

    def test_save_creates_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            eng = _make_engine()
            eng._state_path = os.path.join(tmpdir, "sub", "deep", "state.json")
            eng._save_state()
            assert os.path.exists(eng._state_path)


# ═══════════════════════════════════════════════════════════════
# P1: 公共 API
# ═══════════════════════════════════════════════════════════════

class TestPublicAPI:
    def test_get_dynamic_target(self):
        eng = _make_engine()
        assert eng.get_dynamic_target() == 0.85

    def test_get_latest_report_none(self):
        eng = _make_engine()
        assert eng.get_latest_report() is None

    def test_get_latest_report_after_analyze(self):
        eng = _make_engine()
        rep = eng.analyze(total_equity=100.0, used_margin=50.0, available=50.0, atr_ratio=1.0)
        assert eng.get_latest_report() is rep

    def test_get_utilization_summary_no_report(self):
        eng = _make_engine()
        assert eng.get_utilization_summary() == {"available": False}

    def test_get_utilization_summary_full(self):
        eng = _make_engine()
        eng.analyze(
            total_equity=100.0, used_margin=50.0, available=50.0,
            strategy_usage={"grid": 30, "trend": 20},
            recent_pnl={"grid": 0.5, "trend": -0.2},
            atr_ratio=1.0, session="european",
        )
        summary = eng.get_utilization_summary()
        assert summary["current_utilization"] == 0.5
        assert "dynamic_target" in summary
        assert "utilization_gap" in summary
        assert "tier" in summary
        assert "action" in summary
        assert "capital_efficiency" in summary
        assert "strategy_targets" in summary
        assert "strategy_actions" in summary
        assert "allocation_shift" in summary
        assert "oscillation_counter" in summary
        assert "timestamp" in summary

    def test_get_utilization_heatmap(self):
        eng = _make_engine()
        for i in range(5):
            eng.analyze(total_equity=100.0, used_margin=50.0, available=50.0, atr_ratio=1.0)
        heatmap = eng.get_utilization_heatmap(limit=3)
        assert len(heatmap) == 3
        for entry in heatmap:
            assert "utilization" in entry
            assert "tier" in entry
            assert "action" in entry
            assert "session" in entry

    def test_get_oscillation_status(self):
        eng = _make_engine()
        status = eng.get_oscillation_status()
        assert status["oscillation_counter"] == 0
        assert status["max_oscillations"] == 3
        assert status["is_locked"] is False
        assert status["last_action"] == "none"

    def test_reset_oscillation_state(self):
        eng = _make_engine()
        eng._oscillation_counter = 5
        eng._action_cooldown = 600
        eng._last_action = UtilizationAction.HOLD
        eng._last_action_time = datetime.now()

        eng.reset_oscillation_state()
        assert eng._oscillation_counter == 0
        assert eng._action_cooldown == 120
        assert eng._last_action is None
        assert eng._last_action_time is None


# ═══════════════════════════════════════════════════════════════
# P2: 边界 & 异常
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_zero_equity_returns_empty_report(self):
        eng = _make_engine()
        rep = eng.analyze(total_equity=0.0, used_margin=0.0, available=0.0, atr_ratio=1.0)
        assert rep.current_utilization == 0.0
        assert rep.utilization_tier == UtilizationTier.CRITICAL_LOW
        assert rep.strategy_targets == {}
        assert rep.strategy_actions == {}

    def test_negative_equity_returns_empty(self):
        eng = _make_engine()
        rep = eng.analyze(total_equity=-100.0, used_margin=50.0, available=50.0, atr_ratio=1.0)
        assert rep.current_utilization == 0.0

    def test_utilization_exceeds_100_percent(self):
        eng = _make_engine()
        rep = eng.analyze(total_equity=100.0, used_margin=150.0, available=0.0, atr_ratio=1.0)
        assert rep.current_utilization == 1.50
        assert rep.utilization_tier == UtilizationTier.CRITICAL_HIGH

    def test_empty_strategy_usage(self):
        eng = _make_engine()
        rep = eng.analyze(total_equity=100.0, used_margin=50.0, available=50.0,
                          strategy_usage=None, recent_pnl=None, atr_ratio=1.0)
        # strategy_targets always computed (based on context, not usage)
        assert len(rep.strategy_targets) == 6
        # strategy_actions is empty when no usage provided
        assert rep.strategy_actions == {}
        # allocation_shift iterates over strategy_targets, so all 6 get shifts
        assert len(rep.allocation_shift) == 6

    def test_used_margin_zero(self):
        eng = _make_engine()
        rep = eng.analyze(total_equity=100.0, used_margin=0.0, available=100.0, atr_ratio=1.0)
        assert rep.current_utilization == 0.0
        assert rep.deployment_ratio == 0.0
        assert rep.capital_efficiency == 0.0

    def test_available_zero(self):
        eng = _make_engine()
        rep = eng.analyze(total_equity=100.0, used_margin=100.0, available=0.0, atr_ratio=1.0)
        assert rep.deployment_ratio == 1.0

    def test_deployment_ratio_division_by_zero(self):
        """used_margin + available = 0 → deployment_ratio = 0"""
        eng = _make_engine()
        rep = eng.analyze(total_equity=100.0, used_margin=0.0, available=0.0, atr_ratio=1.0)
        assert rep.deployment_ratio == 0.0

    def test_snapshot_maxlen(self):
        eng = _make_engine()
        for i in range(600):
            eng.analyze(total_equity=100.0, used_margin=50.0, available=50.0, atr_ratio=1.0)
        assert len(eng._snapshots) == 500
        assert len(list(eng._reports)) == 100

    def test_many_strategies(self):
        eng = _make_engine("small")
        rep = eng.analyze(
            total_equity=1000.0, used_margin=500.0, available=500.0,
            strategy_usage={
                "grid": 100, "trend": 80, "scalping": 70, "arbitrage": 60,
                "spot_grid": 50, "spot_martingale": 40,
            },
            recent_pnl={
                "grid": 5, "trend": -2, "scalping": 3, "arbitrage": 1,
                "spot_grid": 2, "spot_martingale": -1,
            },
            atr_ratio=1.0, session="european",
        )
        assert len(rep.strategy_targets) == 6
        assert len(rep.strategy_actions) == 6
        assert len(rep.allocation_shift) == 6


# ═══════════════════════════════════════════════════════════════
# P0: 账户层级常量
# ═══════════════════════════════════════════════════════════════

class TestTierConstants:
    def test_constants_defined(self):
        assert set(CapitalUtilizationEngine.TIER_CAPS) == {
            "nano", "micro", "small", "medium", "large", "xlarge"
        }
        assert set(CapitalUtilizationEngine.TIER_POSITION_BOOST_CAPS) == {
            "nano", "micro", "small", "medium", "large", "xlarge"
        }

    def test_caps_monotonic(self):
        tiers = ["nano", "micro", "small", "medium", "large", "xlarge"]
        caps = [CapitalUtilizationEngine.TIER_CAPS[t] for t in tiers]
        boost_caps = [CapitalUtilizationEngine.TIER_POSITION_BOOST_CAPS[t] for t in tiers]
        assert caps == sorted(caps)
        assert boost_caps == sorted(boost_caps)

    def test_session_adjustments_all_present(self):
        expected = {"asian", "european", "overlap_eu_us", "us", "low_liquidity", "weekend"}
        assert set(CapitalUtilizationEngine.SESSION_ADJUSTMENTS) == expected

    def test_volatility_adjustments_all_present(self):
        expected = {"very_low", "low", "normal", "high", "very_high"}
        assert set(CapitalUtilizationEngine.VOLATILITY_ADJUSTMENTS) == expected

    def test_equity_mode_adjustments_all_present(self):
        expected = {"normal", "growth", "decline", "recovery", "emergency", "milestone"}
        assert set(CapitalUtilizationEngine.EQUITY_MODE_ADJUSTMENTS) == expected

    def test_strategy_base_targets_all_present(self):
        expected = {"grid", "scalping", "arbitrage", "trend", "spot_grid", "spot_martingale"}
        assert set(CapitalUtilizationEngine.STRATEGY_BASE_TARGETS) == expected

    def test_all_utilization_actions_distinct(self):
        values = set(a.value for a in UtilizationAction)
        assert len(values) == len(UtilizationAction)


# ═══════════════════════════════════════════════════════════════
# P1: 相对目标硬约束（强化项）
# ═══════════════════════════════════════════════════════════════

class TestRelativeTargetConstraint:
    def test_hard_constraint_reduces(self):
        eng = _make_engine("nano")
        eng._dynamic_target = 0.50
        action = eng._determine_action(
            utilization=0.80, tier=UtilizationTier.OPTIMAL, trend=0.0, efficiency=0.0
        )
        assert action == UtilizationAction.REDUCE_MODERATE

    def test_gap_between_10_and_20_no_constraint(self):
        """gap=0.15 > 0.10 but utilization=0.75 < 0.80 → no hard constraint"""
        eng = _make_engine("nano")
        eng._dynamic_target = 0.60
        eng._last_action = None
        eng._last_action_time = None
        action = eng._determine_action(
            utilization=0.75, tier=UtilizationTier.OPTIMAL, trend=0.0, efficiency=0.0
        )
        # Not triggering hard constraint, falls through to raw_action
        assert action == UtilizationAction.HOLD

    def test_gap_under_35_boosts_aggressive(self):
        eng = _make_engine("nano")
        eng._dynamic_target = 0.75
        eng._last_action = None
        eng._last_action_time = None
        action = eng._determine_action(
            utilization=0.25, tier=UtilizationTier.LOW, trend=0.0, efficiency=0.0
        )
        assert action == UtilizationAction.BOOST_AGGRESSIVE


# ═══════════════════════════════════════════════════════════════
# P0: 强制再平衡（资本效率硬约束）
# ═══════════════════════════════════════════════════════════════

class TestForceRebalance:
    def test_triggers_low_efficiency_large_equity(self):
        eng = _make_engine("small")
        eng._equity_mode = "normal"
        action = eng._determine_action(0.5, UtilizationTier.OPTIMAL, 0.0, 0.10, 150.0)
        assert action == UtilizationAction.FORCE_REBALANCE

    def test_no_trigger_high_efficiency(self):
        eng = _make_engine("small")
        eng._equity_mode = "normal"
        action = eng._determine_action(0.5, UtilizationTier.OPTIMAL, 0.0, 0.20, 150.0)
        assert action == UtilizationAction.HOLD

    def test_no_trigger_small_equity(self):
        eng = _make_engine("small")
        eng._equity_mode = "normal"
        action = eng._determine_action(0.5, UtilizationTier.OPTIMAL, 0.0, 0.10, 90.0)
        assert action == UtilizationAction.HOLD

    def test_no_trigger_missing_equity(self):
        """total_equity 缺省（None）时不得触发，兼容旧调用签名"""
        eng = _make_engine("small")
        eng._equity_mode = "normal"
        action = eng._determine_action(0.5, UtilizationTier.OPTIMAL, 0.0, 0.10)
        assert action == UtilizationAction.HOLD

    def test_emergency_overrides_force_rebalance(self):
        eng = _make_engine("small")
        eng._equity_mode = "emergency"
        action = eng._determine_action(0.5, UtilizationTier.OPTIMAL, 0.0, 0.10, 150.0)
        assert action == UtilizationAction.EMERGENCY_FREEZE

    def test_position_boost_neutral(self):
        eng = _make_engine("small")
        boost = eng._compute_position_boost(
            0.5, UtilizationTier.OPTIMAL, UtilizationAction.FORCE_REBALANCE
        )
        assert boost == 1.0

    def test_signal_relaxation_tightens(self):
        eng = _make_engine("small")
        rel = eng._compute_signal_relaxation(
            0.5, UtilizationTier.OPTIMAL, UtilizationAction.FORCE_REBALANCE
        )
        assert rel == -0.10

    def test_allocation_shift_more_aggressive(self):
        eng = _make_engine("small")
        normal = eng._compute_allocation_shift(
            {"grid": 10}, {"grid": 0.8}, {"grid": 0.5}, UtilizationAction.HOLD
        )
        forced = eng._compute_allocation_shift(
            {"grid": 10}, {"grid": 0.8}, {"grid": 0.5}, UtilizationAction.FORCE_REBALANCE
        )
        # 盈利策略在 FORCE_REBALANCE 下向目标移动幅度更大
        assert abs(forced["grid"]) >= abs(normal["grid"])

    def test_allocation_shift_clip_wider(self):
        eng = _make_engine("small")
        shifts = eng._compute_allocation_shift(
            {}, {"grid": 0.8}, {}, UtilizationAction.FORCE_REBALANCE
        )
        # current=0, target=0.8 → raw=0.48 → clip 0.15（普通为 0.10）
        assert shifts["grid"] == pytest.approx(0.15)

    def test_analyze_triggers_force_rebalance(self):
        eng = _make_engine("small")
        rep = eng.analyze(
            total_equity=200.0, used_margin=100.0, available=100.0,
            strategy_usage={"grid": 100.0},
            recent_pnl={"grid": 10.0},  # efficiency = 10/100 = 0.10 < 0.15
            atr_ratio=1.0, session="european",
        )
        assert rep.capital_efficiency == pytest.approx(0.10)
        assert rep.recommended_action == UtilizationAction.FORCE_REBALANCE
        assert rep.position_boost == 1.0
        assert rep.signal_relaxation == -0.10

    def test_analyze_no_trigger_efficient(self):
        eng = _make_engine("small")
        rep = eng.analyze(
            total_equity=200.0, used_margin=100.0, available=100.0,
            strategy_usage={"grid": 100.0},
            recent_pnl={"grid": 30.0},  # efficiency = 30/100 = 0.30 >= 0.15
            atr_ratio=1.0, session="european",
        )
        assert rep.recommended_action != UtilizationAction.FORCE_REBALANCE