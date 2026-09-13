"""RiskBudgetEngine 企业级风险预算引擎 — 正式单元测试。

覆盖矩阵（聚焦本次「强化企业级风险预算分配」改动）：
  P0  厚尾 VaR 修正（Cornish-Fisher VaR）
  P0  蒙特卡洛 Expected Shortfall (MC ES)
  P0  风险分解 _decompose_risk 的厚尾接入 + 正态回退
  P0  币种集中度 _compute_symbol_concentration 反向映射方向
  P0  risk_budget 配置段解析（含默认值回退）
  P1  compute_risk_budget_plan 集成管线（厚尾路径产出正 VaR）
  P2  边界（样本不足、零波动、空策略）
"""

import math

import numpy as np
import pytest

from risk.risk_budget_engine import (
    RiskBudget,
    RiskBudgetEngine,
    RiskBudgetPlan,
    SymbolRiskBudget,
)


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _make_engine(**risk_budget_overrides) -> RiskBudgetEngine:
    """构造带完整 risk_budget 配置段的引擎，允许覆盖部分字段。"""
    cfg = {
        "risk_budget": {
            "enabled": True,
            "daily_risk_budget_pct": 0.03,
            "hourly_max_loss_pct": 0.015,
            "max_per_trade_risk_pct": 0.008,
            "strategy_budgets": {
                "scalping": 0.40,
                "trend": 0.25,
                "grid": 0.20,
                "arbitrage": 0.15,
            },
            "reallocation": {
                "enabled": True,
                "interval_minutes": 5,
                "max_shift_ratio": 0.15,
                "transfer_out_winrate_threshold": 0.35,
                "transfer_out_drawdown_threshold": 0.08,
            },
            "concentration": {
                "max_single_symbol_risk_pct": 0.008,
                "max_correlated_group_risk_pct": 0.02,
            },
            "loss_streak": {
                "max_consecutive_losses": 5,
                "loss_streak_reduce_pct": 0.50,
                "recovery_consecutive_wins": 3,
            },
            "high_correlation_threshold": 0.70,
            "extreme_correlation_threshold": 0.85,
            "data_dir": "./data",
        }
    }
    for key, value in risk_budget_overrides.items():
        cfg["risk_budget"][key] = value
    return RiskBudgetEngine(cfg)


def _left_skewed_returns(n: int = 60) -> list:
    """构造带负偏度（厚左尾）的确定性收益率序列。"""
    # 大部分为小额正收益，尾部有若干大额亏损 → 负偏度
    body = [0.002] * (n - 6)
    tail = [-0.04, -0.06, -0.09, -0.05, -0.12, -0.03]
    return body + tail


def _symmetric_returns(n: int = 60) -> list:
    """构造近似对称、近零均值/近零偏度的收益率序列。"""
    rng = np.random.default_rng(42)
    return list(rng.normal(0.0, 0.01, size=n))


# ═══════════════════════════════════════════════════════════════
# P0: Cornish-Fisher VaR
# ═══════════════════════════════════════════════════════════════

class TestCornishFisherVaR:
    def test_returns_positive_on_left_skewed_sample(self):
        """左偏厚尾样本下必须返回正损失（修复前恒为 0）。"""
        eng = _make_engine()
        returns = _left_skewed_returns()
        var = eng._cornish_fisher_var(returns, confidence=0.95)
        assert var > 0

    def test_var99_ge_var95(self):
        """置信水平越高，VaR 越大（单调性）。"""
        eng = _make_engine()
        returns = _left_skewed_returns()
        var95 = eng._cornish_fisher_var(returns, confidence=0.95)
        var99 = eng._cornish_fisher_var(returns, confidence=0.99)
        assert var99 >= var95

    def test_insufficient_sample_returns_zero(self):
        """样本 < 20 时返回 0.0。"""
        eng = _make_engine()
        assert eng._cornish_fisher_var([0.01] * 10) == 0.0
        assert eng._cornish_fisher_var([]) == 0.0

    def test_zero_volatility_returns_zero(self):
        """零波动（常数序列）返回 0.0。"""
        eng = _make_engine()
        assert eng._cornish_fisher_var([0.005] * 30) == 0.0

    def test_holding_period_scaling(self):
        """holding_period 增加时 VaR 按 √T 放大。"""
        eng = _make_engine()
        returns = _left_skewed_returns()
        var1 = eng._cornish_fisher_var(returns, confidence=0.95, holding_period=1)
        var4 = eng._cornish_fisher_var(returns, confidence=0.95, holding_period=4)
        assert var4 == pytest.approx(var1 * 2.0, rel=1e-9)


# ═══════════════════════════════════════════════════════════════
# P0: 蒙特卡洛 Expected Shortfall
# ═══════════════════════════════════════════════════════════════

class TestMonteCarloES:
    def test_returns_positive_on_left_skewed_sample(self):
        eng = _make_engine()
        es = eng._monte_carlo_es(_left_skewed_returns(), confidence=0.95)
        assert es > 0

    def test_es_ge_var95(self):
        """ES（尾部均值）应不低于 95% VaR。"""
        eng = _make_engine()
        returns = _left_skewed_returns()
        var95 = eng._cornish_fisher_var(returns, confidence=0.95)
        es = eng._monte_carlo_es(returns, confidence=0.95)
        # ES 为尾部期望，理论上 ≥ VaR；给少量蒙特卡洛容差
        assert es >= var95 * 0.8

    def test_insufficient_sample_returns_zero(self):
        eng = _make_engine()
        assert eng._monte_carlo_es([0.01] * 5) == 0.0

    def test_zero_volatility_returns_zero(self):
        eng = _make_engine()
        assert eng._monte_carlo_es([0.01] * 30) == 0.0


# ═══════════════════════════════════════════════════════════════
# P0: 风险分解（厚尾接入 + 正态回退）
# ═══════════════════════════════════════════════════════════════

class TestDecomposeRisk:
    def _make_plan(self, equity=10000.0):
        plan = RiskBudgetPlan(
            total_equity=equity,
            total_risk_budget_pct=0.03,
            total_risk_budget=equity * 0.03,
        )
        return plan

    def test_normal_fallback_without_returns(self):
        """未注册收益率序列 → 回退正态近似，VaR 为正值。"""
        eng = _make_engine()
        plan = self._make_plan()
        eng._decompose_risk(
            plan,
            strategy_names=["trend"],
            forward_vols={"trend": 0.5},
            target_budget_pcts={"trend": 0.6},
            covariance_matrix=None,
            current_consumed={},
        )
        rb = plan.strategy_budgets["trend"]
        assert rb.var_95 > 0
        assert rb.var_99 > rb.var_95

    def test_thick_tail_path_produces_positive_var(self):
        """注册 ≥20 个收益率后走 Cornish-Fisher + MC ES 路径，VaR/CVaR 为正值。"""
        eng = _make_engine()
        eng.update_strategy_returns("trend", _left_skewed_returns())
        plan = self._make_plan()
        eng._decompose_risk(
            plan,
            strategy_names=["trend"],
            forward_vols={"trend": 0.5},
            target_budget_pcts={"trend": 0.6},
            covariance_matrix=None,
            current_consumed={},
        )
        rb = plan.strategy_budgets["trend"]
        assert rb.var_95 > 0
        assert rb.var_99 >= rb.var_95
        assert rb.cvar_95 > 0

    def test_thick_tail_var_exceeds_normal_var_for_left_skew(self):
        """左偏厚尾样本的 CF VaR 应高于同波动率下的正态近似 VaR。"""
        eng = _make_engine()
        returns = _left_skewed_returns()
        eng.update_strategy_returns("trend", returns)

        # 用真实收益率估算年化波动率，构造 forward_vols
        daily_std = float(np.std(returns, ddof=1))
        ann_vol = daily_std * math.sqrt(365)

        plan = self._make_plan()
        eng._decompose_risk(
            plan,
            strategy_names=["trend"],
            forward_vols={"trend": ann_vol},
            target_budget_pcts={"trend": 1.0},
            covariance_matrix=None,
            current_consumed={},
        )
        cf_var = plan.strategy_budgets["trend"].var_95

        # 独立计算正态近似 VaR 作参照
        position_value = plan.total_equity * 1.0
        normal_var = position_value * (ann_vol / math.sqrt(365)) * 1.645
        assert cf_var >= normal_var

    def test_multi_strategy_decomposition_aggregates_totals(self):
        """多策略分解时汇总 total_var 应等于各策略之和。"""
        eng = _make_engine()
        eng.update_strategy_returns("trend", _left_skewed_returns())
        eng.update_strategy_returns("scalping", _symmetric_returns())
        plan = self._make_plan()
        eng._decompose_risk(
            plan,
            strategy_names=["trend", "scalping"],
            forward_vols={"trend": 0.5, "scalping": 0.8},
            target_budget_pcts={"trend": 0.6, "scalping": 0.4},
            covariance_matrix=None,
            current_consumed={},
        )
        assert plan.strategy_budgets["trend"].var_95 > 0
        assert plan.strategy_budgets["scalping"].var_95 > 0
        assert plan.total_var_95 == pytest.approx(
            plan.strategy_budgets["trend"].var_95 + plan.strategy_budgets["scalping"].var_95
        )


# ═══════════════════════════════════════════════════════════════
# P0: 币种集中度反向映射
# ═══════════════════════════════════════════════════════════════

class TestSymbolConcentration:
    def _make_plan(self, equity=10000.0):
        plan = RiskBudgetPlan(
            total_equity=equity,
            total_risk_budget_pct=0.03,
            total_risk_budget=equity * 0.03,
        )
        plan.strategy_budgets["trend"] = RiskBudget(strategy_name="trend", var_95=200.0, var_99=300.0)
        plan.strategy_budgets["grid"] = RiskBudget(strategy_name="grid", var_95=100.0, var_99=150.0)
        plan.total_var_95 = 300.0
        return plan

    def test_reverse_mapping_aggregates_to_symbol(self):
        """修复前会把策略名当 symbol；修复后应聚合到真实币种。"""
        eng = _make_engine()
        eng.register_symbol_strategy("BTC-USDT-SWAP", "trend")
        eng.register_symbol_strategy("BTC-USDT-SWAP", "grid")

        result = eng._compute_symbol_concentration(["trend", "grid"], self._make_plan())

        # 正确方向：键为币种而非策略名
        assert "BTC-USDT-SWAP" in result
        assert "trend" not in result
        assert "grid" not in result

        # 两个策略都映射到同一币种，VaR 聚合正确
        srb = result["BTC-USDT-SWAP"]
        assert srb.total_var_95 == pytest.approx(300.0)
        assert srb.total_var_99 == pytest.approx(450.0)
        assert set(srb.contributing_strategies) == {"trend", "grid"}

    def test_multi_symbol_strategy_splits_var(self):
        """单策略涉及多个币种时，VaR 均分到各币种。"""
        eng = _make_engine()
        eng.register_symbol_strategy("BTC-USDT-SWAP", "trend")
        eng.register_symbol_strategy("ETH-USDT-SWAP", "trend")

        plan = self._make_plan()
        plan.strategy_budgets = {
            "trend": RiskBudget(strategy_name="trend", var_95=200.0, var_99=300.0),
        }
        plan.total_var_95 = 200.0

        result = eng._compute_symbol_concentration(["trend"], plan)
        assert result["BTC-USDT-SWAP"].total_var_95 == pytest.approx(100.0)
        assert result["ETH-USDT-SWAP"].total_var_95 == pytest.approx(100.0)

    def test_unregistered_strategy_falls_back_to_name(self):
        """未注册币种映射时，退化用策略名本身作占位。"""
        eng = _make_engine()
        result = eng._compute_symbol_concentration(["trend"], self._make_plan())
        assert "trend" in result

    def test_over_concentration_flagged(self):
        """集中度超限时打上告警标记。"""
        eng = _make_engine()
        eng.register_symbol_strategy("BTC-USDT-SWAP", "trend")
        # 大权益使单币种绝对风险上限远小于聚合 VaR
        result = eng._compute_symbol_concentration(["trend", "grid"], self._make_plan(equity=1000.0))
        srb = result["BTC-USDT-SWAP"]
        assert srb.is_over_concentration is True
        assert srb.warning_level in ("warning", "critical")


# ═══════════════════════════════════════════════════════════════
# P0: risk_budget 配置段解析
# ═══════════════════════════════════════════════════════════════

class TestConfigParsing:
    def test_full_config_parsed(self):
        eng = _make_engine()
        assert eng._enabled is True
        assert eng._daily_risk_budget_pct == 0.03
        assert eng._hourly_max_loss == 0.015
        assert eng._max_per_trade_risk == 0.008
        assert eng._strategy_budget_pcts == {
            "scalping": 0.40, "trend": 0.25, "grid": 0.20, "arbitrage": 0.15,
        }
        assert eng._realloc_enabled is True
        assert eng._realloc_interval == 5
        assert eng._max_shift_ratio == 0.15
        assert eng._max_symbol_risk == 0.008
        assert eng._max_correlated_risk == 0.02
        assert eng._max_consecutive_losses == 5
        assert eng._loss_reduce_pct == 0.50
        assert eng._recovery_wins == 3
        assert eng._high_correlation_threshold == 0.70
        assert eng._extreme_correlation_threshold == 0.85

    def test_custom_overrides(self):
        eng = _make_engine(
            daily_risk_budget_pct=0.05,
            max_per_trade_risk_pct=0.01,
            strategy_budgets={"scalping": 0.5, "trend": 0.3, "grid": 0.2},
            reallocation={"enabled": False, "interval_minutes": 30},
            concentration={"max_single_symbol_risk_pct": 0.012, "max_correlated_group_risk_pct": 0.03},
            high_correlation_threshold=0.80,
        )
        assert eng._daily_risk_budget_pct == 0.05
        assert eng._max_per_trade_risk == 0.01
        assert eng._strategy_budget_pcts == {"scalping": 0.5, "trend": 0.3, "grid": 0.2}
        assert eng._realloc_enabled is False
        assert eng._realloc_interval == 30
        assert eng._max_symbol_risk == 0.012
        assert eng._max_correlated_risk == 0.03
        assert eng._high_correlation_threshold == 0.80

    def test_missing_config_uses_defaults(self):
        """无 risk_budget 段时走引擎默认值，不抛异常。"""
        eng = RiskBudgetEngine({})
        assert eng._enabled is True
        assert eng._daily_risk_budget_pct == 0.03
        assert eng._max_symbol_risk == 0.008
        assert eng._realloc_interval == 60
        assert set(eng._strategy_budget_pcts) == {"scalping", "trend", "grid", "arbitrage"}


# ═══════════════════════════════════════════════════════════════
# P1: 集成管线 compute_risk_budget_plan
# ═══════════════════════════════════════════════════════════════

class TestComputeRiskBudgetPlan:
    async def test_full_plan_with_thick_tail_returns(self):
        """完整管线在厚尾样本下产出正 VaR 与币种集中度。"""
        eng = _make_engine()
        eng.update_strategy_returns("trend", _left_skewed_returns())
        eng.update_strategy_returns("scalping", _symmetric_returns())
        eng.register_symbol_strategy("BTC-USDT-SWAP", "trend")
        eng.register_symbol_strategy("ETH-USDT-SWAP", "scalping")

        plan = await eng.compute_risk_budget_plan(
            total_equity=10000.0,
            strategy_names=["trend", "scalping"],
            strategy_metrics={"trend": {}, "scalping": {}},
            current_risk_consumed={},
        )

        assert plan.total_risk_budget == pytest.approx(300.0)
        assert "trend" in plan.strategy_budgets
        assert "scalping" in plan.strategy_budgets
        assert plan.strategy_budgets["trend"].var_95 > 0
        assert plan.strategy_budgets["trend"].cvar_95 > 0
        assert plan.total_var_95 > 0

        # 币种集中度产物存在且键为真实币种
        conc = plan._symbol_concentration
        assert "BTC-USDT-SWAP" in conc
        assert "trend" not in conc

    async def test_zero_equity_returns_empty_plan(self):
        eng = _make_engine()
        plan = await eng.compute_risk_budget_plan(
            total_equity=0.0,
            strategy_names=["trend"],
        )
        assert plan.total_equity == 0.0
        assert plan.strategy_budgets == {}
