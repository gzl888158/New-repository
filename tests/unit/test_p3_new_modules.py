"""P3 新增模块单元测试：归因分析、场景模拟、参数优化、多时间框架规划。"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from core.capital_attribution_analyzer import CapitalAttributionAnalyzer
from core.capital_scenario_simulator import CapitalScenarioSimulator
from core.capital_parameter_optimizer import CapitalParameterOptimizer
from core.multi_timeframe_capital_planner import MultiTimeframeCapitalPlanner


# ═══════════════════════════════════════════════════════════════
# P3-1: CapitalAttributionAnalyzer
# ═══════════════════════════════════════════════════════════════

class TestAttributionAnalyzer:

    def test_init_default(self):
        analyzer = CapitalAttributionAnalyzer()
        assert analyzer is not None

    def test_init_with_config(self):
        analyzer = CapitalAttributionAnalyzer({"attribution": {"enabled": True}})
        assert analyzer is not None

    def test_record_period(self):
        analyzer = CapitalAttributionAnalyzer()
        analyzer.record_period(
            period_key="2026-10-01",
            strategy_weights={"grid": 0.5, "trend": 0.3, "scalping": 0.2},
            strategy_returns={"grid": 0.02, "trend": 0.01, "scalping": 0.03},
            strategy_allocated={"grid": 50.0, "trend": 30.0, "scalping": 20.0},
            strategy_pnl={"grid": 1.0, "trend": 0.3, "scalping": 0.6},
        )
        assert len(analyzer._period_data) == 1

    def test_record_multiple_periods(self):
        analyzer = CapitalAttributionAnalyzer()
        for i in range(5):
            analyzer.record_period(
                period_key=f"2026-10-{i+1:02d}",
                strategy_weights={"grid": 0.5, "trend": 0.5},
                strategy_returns={"grid": 0.01 * i, "trend": 0.02 * i},
                strategy_allocated={"grid": 50.0, "trend": 50.0},
                strategy_pnl={"grid": 0.5 * i, "trend": 1.0 * i},
            )
        assert len(analyzer._period_data) == 5

    def test_compute_attribution_basic(self):
        analyzer = CapitalAttributionAnalyzer()
        analyzer.record_period(
            period_key="2026-10-01",
            strategy_weights={"grid": 0.6, "trend": 0.4},
            strategy_returns={"grid": 0.02, "trend": 0.01},
            strategy_allocated={"grid": 60.0, "trend": 40.0},
            strategy_pnl={"grid": 1.2, "trend": 0.4},
        )
        result = analyzer.compute_attribution()
        assert result is not None
        assert hasattr(result, "portfolio_return")
        assert hasattr(result, "allocation_effect")
        assert hasattr(result, "selection_effect")
        assert hasattr(result, "interaction_effect")

    def test_compute_attribution_empty(self):
        analyzer = CapitalAttributionAnalyzer()
        result = analyzer.compute_attribution()
        assert result is not None
        assert result.portfolio_return == 0.0

    def test_compute_attribution_with_date_range(self):
        analyzer = CapitalAttributionAnalyzer()
        for i in range(7):
            analyzer.record_period(
                period_key=f"2026-10-{i+1:02d}",
                strategy_weights={"grid": 0.5, "trend": 0.5},
                strategy_returns={"grid": 0.01, "trend": 0.02},
                strategy_allocated={"grid": 50.0, "trend": 50.0},
                strategy_pnl={"grid": 0.5, "trend": 1.0},
            )
        start = datetime(2026, 10, 2)
        end = datetime(2026, 10, 5)
        result = analyzer.compute_attribution(period_start=start, period_end=end)
        assert result is not None

    def test_get_top_contributors(self):
        analyzer = CapitalAttributionAnalyzer()
        analyzer.record_period(
            period_key="2026-10-01",
            strategy_weights={"grid": 0.4, "trend": 0.3, "scalping": 0.3},
            strategy_returns={"grid": 0.05, "trend": -0.01, "scalping": 0.03},
            strategy_allocated={"grid": 40.0, "trend": 30.0, "scalping": 30.0},
            strategy_pnl={"grid": 2.0, "trend": -0.3, "scalping": 0.9},
        )
        analyzer.compute_attribution()
        top = analyzer.get_top_contributors(limit=2)
        assert len(top) <= 2

    def test_get_top_contributors_empty(self):
        analyzer = CapitalAttributionAnalyzer()
        top = analyzer.get_top_contributors(limit=3)
        assert top == []

    def test_strategy_attribution_has_all_strategies(self):
        analyzer = CapitalAttributionAnalyzer()
        strategies = {"grid": 0.5, "trend": 0.3, "scalping": 0.2}
        analyzer.record_period(
            period_key="2026-10-01",
            strategy_weights=strategies,
            strategy_returns={"grid": 0.02, "trend": 0.01, "scalping": 0.03},
            strategy_allocated={"grid": 50.0, "trend": 30.0, "scalping": 20.0},
            strategy_pnl={"grid": 1.0, "trend": 0.3, "scalping": 0.6},
        )
        result = analyzer.compute_attribution()
        for s in strategies:
            assert s in result.strategy_attribution


# ═══════════════════════════════════════════════════════════════
# P3-2: CapitalScenarioSimulator
# ═══════════════════════════════════════════════════════════════

class TestScenarioSimulator:

    def test_init_default(self):
        sim = CapitalScenarioSimulator()
        assert sim is not None

    def test_init_with_config(self):
        sim = CapitalScenarioSimulator({"n_simulations": 500, "horizon_days": 14})
        assert sim._n_simulations == 500
        assert sim._horizon_days == 14

    def test_monte_carlo_basic(self):
        np.random.seed(42)
        sim = CapitalScenarioSimulator({"n_simulations": 100, "horizon_days": 7})
        allocation = {"grid": 0.5, "trend": 0.3, "scalping": 0.2}
        historical = {
            "grid": [0.001, -0.002, 0.003, 0.001, -0.001, 0.002, 0.0, 0.001] * 4,
            "trend": [0.002, -0.003, 0.001, 0.004, -0.002, 0.001, 0.003, -0.001] * 4,
            "scalping": [0.001, 0.001, -0.001, 0.002, 0.0, -0.001, 0.001, 0.002] * 4,
        }
        result = sim.run_monte_carlo(allocation, historical, n_simulations=100, horizon_days=7)
        assert "var_95" in result
        assert "var_99" in result
        assert "cvar_95" in result
        assert "cvar_99" in result
        assert "max_drawdown_p50" in result
        assert "max_drawdown_p95" in result
        assert "expected_return" in result
        assert "sharpe_ratio" in result
        assert result["n_simulations"] == 100
        assert result["horizon_days"] == 7

    def test_monte_carlo_var_ordering(self):
        np.random.seed(42)
        sim = CapitalScenarioSimulator()
        allocation = {"grid": 0.5, "trend": 0.5}
        historical = {
            "grid": list(np.random.normal(0.001, 0.02, 60)),
            "trend": list(np.random.normal(0.001, 0.02, 60)),
        }
        result = sim.run_monte_carlo(allocation, historical, n_simulations=500)
        assert result["var_95"] >= result["var_99"]

    def test_monte_carlo_short_history(self):
        sim = CapitalScenarioSimulator()
        allocation = {"grid": 1.0}
        historical = {"grid": [0.01]}
        result = sim.run_monte_carlo(allocation, historical, n_simulations=50, horizon_days=5)
        assert "expected_return" in result

    def test_monte_carlo_empty_history(self):
        sim = CapitalScenarioSimulator()
        allocation = {"grid": 1.0}
        historical = {"grid": []}
        result = sim.run_monte_carlo(allocation, historical, n_simulations=50)
        assert "expected_return" in result

    def test_stress_test_basic(self):
        sim = CapitalScenarioSimulator()
        allocation = {"grid": 0.5, "trend": 0.3, "scalping": 0.2}
        scenarios = {
            "market_crash": {"grid": -0.10, "trend": -0.20, "scalping": -0.15},
            "mild_correction": {"grid": -0.03, "trend": -0.05, "scalping": -0.04},
        }
        result = sim.stress_test(allocation, scenarios)
        assert "market_crash" in result
        assert "mild_correction" in result
        assert result["market_crash"]["portfolio_return"] < result["mild_correction"]["portfolio_return"]

    def test_stress_test_worst_strategy(self):
        sim = CapitalScenarioSimulator()
        allocation = {"grid": 0.5, "trend": 0.5}
        scenarios = {
            "crash": {"grid": -0.05, "trend": -0.20},
        }
        result = sim.stress_test(allocation, scenarios)
        assert result["crash"]["worst_strategy"] == "trend"
        assert result["crash"]["worst_return"] == -0.20

    def test_stress_test_positive_scenario(self):
        sim = CapitalScenarioSimulator()
        allocation = {"grid": 0.6, "trend": 0.4}
        scenarios = {
            "bull_run": {"grid": 0.10, "trend": 0.15},
        }
        result = sim.stress_test(allocation, scenarios)
        assert result["bull_run"]["portfolio_return"] > 0


# ═══════════════════════════════════════════════════════════════
# P3-3: CapitalParameterOptimizer
# ═══════════════════════════════════════════════════════════════

class TestParameterOptimizer:

    def test_init_default(self):
        opt = CapitalParameterOptimizer()
        assert opt is not None

    def test_init_with_config(self):
        opt = CapitalParameterOptimizer({"optimizer": {"enabled": True}})
        assert opt is not None

    def test_optimize_basic(self):
        opt = CapitalParameterOptimizer()
        historical = {
            "grid": list(np.random.normal(0.002, 0.01, 30)),
            "trend": list(np.random.normal(0.001, 0.02, 30)),
            "scalping": list(np.random.normal(0.003, 0.008, 30)),
        }
        current_params = {"kelly_fraction": 0.25, "max_weight": 0.50}
        result = opt.optimize(historical, current_params)
        assert "suggested_params" in result
        assert "expected_sharpe" in result
        assert "expected_return" in result
        assert "expected_max_drawdown" in result
        assert "confidence" in result
        assert "reasoning" in result

    def test_optimize_weights_sum_to_one(self):
        opt = CapitalParameterOptimizer()
        historical = {
            "grid": list(np.random.normal(0.002, 0.01, 30)),
            "trend": list(np.random.normal(0.001, 0.02, 30)),
            "scalping": list(np.random.normal(0.003, 0.008, 30)),
        }
        result = opt.optimize(historical, {})
        weights = result["suggested_params"]["strategy_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_optimize_favors_high_sharpe(self):
        np.random.seed(42)
        opt = CapitalParameterOptimizer()
        historical = {
            "good": list(np.random.normal(0.01, 0.005, 60)),
            "bad": list(np.random.normal(-0.005, 0.03, 60)),
        }
        result = opt.optimize(historical, {})
        weights = result["suggested_params"]["strategy_weights"]
        assert weights["good"] > weights["bad"]

    def test_optimize_all_negative_returns(self):
        opt = CapitalParameterOptimizer()
        historical = {
            "grid": list(np.random.normal(-0.01, 0.02, 30)),
            "trend": list(np.random.normal(-0.02, 0.03, 30)),
        }
        result = opt.optimize(historical, {})
        weights = result["suggested_params"]["strategy_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01

    def test_optimize_with_constraints(self):
        opt = CapitalParameterOptimizer()
        historical = {
            "grid": list(np.random.normal(0.002, 0.01, 30)),
            "trend": list(np.random.normal(0.001, 0.02, 30)),
        }
        constraints = {"max_drawdown": 0.05, "min_sharpe": 1.5}
        result = opt.optimize(historical, {}, constraints=constraints)
        assert "suggested_params" in result

    def test_optimize_single_strategy(self):
        opt = CapitalParameterOptimizer()
        historical = {"grid": list(np.random.normal(0.001, 0.01, 30))}
        result = opt.optimize(historical, {})
        weights = result["suggested_params"]["strategy_weights"]
        assert abs(weights["grid"] - 1.0) < 0.01

    def test_optimize_has_timestamp(self):
        opt = CapitalParameterOptimizer()
        historical = {"grid": [0.01, 0.02, -0.01, 0.005]}
        result = opt.optimize(historical, {})
        assert "timestamp" in result


# ═══════════════════════════════════════════════════════════════
# P3-4: MultiTimeframeCapitalPlanner
# ═══════════════════════════════════════════════════════════════

class TestMultiTimeframePlanner:

    def test_init_default(self):
        planner = MultiTimeframeCapitalPlanner()
        assert planner is not None

    def test_init_with_config(self):
        planner = MultiTimeframeCapitalPlanner({"short_term_min_buffer": 10.0})
        assert planner._short_term["min_buffer"] == 10.0

    def test_update_short_term(self):
        planner = MultiTimeframeCapitalPlanner()
        planner.update_short_term(available_liquidity=30.0, reserved_for_orders=5.0)
        assert planner._short_term["available_liquidity"] == 30.0
        assert planner._short_term["reserved_for_orders"] == 5.0
        assert len(planner._short_term["history"]) == 1

    def test_update_mid_term(self):
        planner = MultiTimeframeCapitalPlanner()
        planner.update_mid_term(
            target_utilization=0.75,
            current_positions={"BTC": 10.0, "ETH": 5.0},
        )
        assert planner._mid_term["target_utilization"] == 0.75
        assert planner._mid_term["current_positions"] == {"BTC": 10.0, "ETH": 5.0}
        assert len(planner._mid_term["history"]) == 1

    def test_update_long_term(self):
        planner = MultiTimeframeCapitalPlanner()
        planner.update_long_term(weekly_target=1.05, current_equity=100.0)
        assert planner._long_term["weekly_target"] == 1.05
        assert planner._long_term["current_equity"] == 100.0
        assert planner._long_term["target_equity"] == 105.0
        assert len(planner._long_term["history"]) == 1

    def test_get_plan_all_tiers(self):
        planner = MultiTimeframeCapitalPlanner()
        planner.update_short_term(available_liquidity=30.0, reserved_for_orders=5.0)
        planner.update_mid_term(target_utilization=0.75, current_positions={"BTC": 10.0})
        planner.update_long_term(weekly_target=1.05, current_equity=100.0)
        plan = planner.get_plan()
        assert "short_term" in plan
        assert "mid_term" in plan
        assert "long_term" in plan
        assert "timestamp" in plan

    def test_get_plan_short_term_insufficient(self):
        planner = MultiTimeframeCapitalPlanner({"short_term_min_buffer": 10.0})
        planner.update_short_term(available_liquidity=3.0, reserved_for_orders=1.0)
        plan = planner.get_plan()
        assert plan["short_term"]["status"] == "insufficient"
        assert plan["short_term"]["action"] == "reduce_positions"

    def test_get_plan_short_term_ok(self):
        planner = MultiTimeframeCapitalPlanner({"short_term_min_buffer": 5.0})
        planner.update_short_term(available_liquidity=30.0, reserved_for_orders=5.0)
        plan = planner.get_plan()
        assert plan["short_term"]["status"] == "ok"
        assert plan["short_term"]["action"] == "hold"

    def test_get_plan_mid_term_increase(self):
        planner = MultiTimeframeCapitalPlanner()
        planner.update_mid_term(target_utilization=0.80, current_positions={"BTC": 0.50})
        plan = planner.get_plan()
        assert plan["mid_term"]["action"] == "increase"

    def test_get_plan_mid_term_reduce(self):
        planner = MultiTimeframeCapitalPlanner()
        planner.update_mid_term(target_utilization=0.30, current_positions={"BTC": 20.0})
        plan = planner.get_plan()
        assert plan["mid_term"]["action"] == "reduce"

    def test_get_plan_mid_term_hold(self):
        planner = MultiTimeframeCapitalPlanner()
        planner.update_mid_term(target_utilization=0.50, current_positions={"BTC": 0.50})
        plan = planner.get_plan()
        assert plan["mid_term"]["action"] == "hold"

    def test_check_constraints_all_pass(self):
        planner = MultiTimeframeCapitalPlanner({"short_term_min_buffer": 5.0})
        planner.update_short_term(available_liquidity=30.0, reserved_for_orders=5.0)
        planner.update_mid_term(target_utilization=0.50, current_positions={"BTC": 0.50})
        planner.update_long_term(weekly_target=1.05, current_equity=100.0)
        constraints = planner.check_constraints()
        assert constraints["short_term"] is True
        assert constraints["all"] is True

    def test_check_constraints_short_term_fail(self):
        planner = MultiTimeframeCapitalPlanner({"short_term_min_buffer": 50.0})
        planner.update_short_term(available_liquidity=3.0, reserved_for_orders=1.0)
        constraints = planner.check_constraints()
        assert constraints["short_term"] is False
        assert constraints["all"] is False

    def test_history_maxlen(self):
        planner = MultiTimeframeCapitalPlanner()
        for i in range(30):
            planner.update_short_term(available_liquidity=float(i), reserved_for_orders=0.0)
        assert len(planner._short_term["history"]) == 24  # maxlen=24

    def test_empty_plan(self):
        planner = MultiTimeframeCapitalPlanner()
        plan = planner.get_plan()
        assert "short_term" in plan
        assert "mid_term" in plan
        assert "long_term" in plan
