"""企业级 What-if 场景模拟与压力测试。

蒙特卡洛模拟：基于历史波动率/相关性生成 1000 条路径，计算各分配方案的 VaR/CVaR/最大回撤分布。

= 使用场景 =
- 评估"如果现在这样调整会怎样"
- 压力测试极端市场条件下的资金分配稳健性
- 输出风险调整后收益最优方案
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from loguru import logger


class CapitalScenarioSimulator:
    """What-if 场景模拟器。

    用法：
        simulator = CapitalScenarioSimulator()
        result = simulator.run_monte_carlo(
            current_allocation={"grid": 0.4, "trend": 0.3, "scalping": 0.3},
            historical_returns={...},  # 各策略历史日收益率
            n_simulations=1000,
            horizon_days=30,
        )
        # result 包含 VaR, CVaR, max_drawdown 分布
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        self._n_simulations = config.get("n_simulations", 1000) if config else 1000
        self._horizon_days = config.get("horizon_days", 30) if config else 30
        logger.info("CapitalScenarioSimulator initialized")

    def run_monte_carlo(self, current_allocation: Dict[str, float],
                        historical_returns: Dict[str, List[float]],
                        n_simulations: int = None,
                        horizon_days: int = None,
                        confidence_level: float = 0.95) -> Dict[str, Any]:
        """蒙特卡洛模拟。

        Args:
            current_allocation: 当前各策略权重
            historical_returns: 各策略历史日收益率列表
            n_simulations: 模拟次数
            horizon_days: 模拟 horizon（天）
            confidence_level: VaR 置信度

        Returns:
            {
                var_95, var_99, cvar_95, cvar_99,
                max_drawdown_p50, max_drawdown_p95,
                expected_return, sharpe_ratio,
                simulation_paths: List[float]  # 最终权益分布
            }
        """
        n_sim = n_simulations or self._n_simulations
        horizon = horizon_days or self._horizon_days

        strategies = list(current_allocation.keys())
        weights = np.array([current_allocation[s] for s in strategies])

        # 构建收益率矩阵（策略 x 天数）
        returns_matrix = []
        for s in strategies:
            hist = historical_returns.get(s, [])
            if len(hist) < 2:
                # 数据不足，使用默认低波动率
                hist = [0.0] * 30
            returns_matrix.append(hist)
        returns_matrix = np.array(returns_matrix)  # shape: (n_strategies, n_days)

        # 计算均值和协方差
        mean_returns = np.mean(returns_matrix, axis=1)  # shape: (n_strategies,)
        if returns_matrix.shape[1] > 1:
            cov_matrix = np.atleast_2d(np.cov(returns_matrix))
        else:
            cov_matrix = np.diag(np.ones(len(strategies)) * 0.0001)

        # 蒙特卡洛模拟
        final_equities = []
        max_drawdowns = []

        for _ in range(n_sim):
            # 生成 horizon 天的随机收益率
            daily_returns = np.random.multivariate_normal(
                mean_returns, cov_matrix, size=horizon
            )  # shape: (horizon, n_strategies)

            # 计算组合日收益率
            portfolio_returns = daily_returns @ weights  # shape: (horizon,)

            # 计算权益曲线
            equity_curve = np.cumprod(1 + portfolio_returns)

            # 最终权益
            final_equities.append(equity_curve[-1])

            # 最大回撤
            running_max = np.maximum.accumulate(equity_curve)
            drawdowns = (running_max - equity_curve) / running_max
            max_drawdowns.append(np.max(drawdowns))

        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)

        # 计算风险指标
        var_95 = np.percentile(final_equities, (1 - confidence_level) * 100)
        var_99 = np.percentile(final_equities, 1)
        cvar_95 = np.mean(final_equities[final_equities <= var_95])
        cvar_99 = np.mean(final_equities[final_equities <= var_99])

        expected_return = np.mean(final_equities) - 1.0
        sharpe_ratio = expected_return / np.std(final_equities) if np.std(final_equities) > 0 else 0

        return {
            "var_95": round(float(var_95), 4),
            "var_99": round(float(var_99), 4),
            "cvar_95": round(float(cvar_95), 4),
            "cvar_99": round(float(cvar_99), 4),
            "max_drawdown_p50": round(float(np.percentile(max_drawdowns, 50)), 4),
            "max_drawdown_p95": round(float(np.percentile(max_drawdowns, 95)), 4),
            "expected_return": round(float(expected_return), 4),
            "sharpe_ratio": round(float(sharpe_ratio), 4),
            "n_simulations": n_sim,
            "horizon_days": horizon,
            "confidence_level": confidence_level,
        }

    def stress_test(self, current_allocation: Dict[str, float],
                    stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """压力测试：极端场景下的资金分配稳健性。

        Args:
            current_allocation: 当前各策略权重
            stress_scenarios: 压力场景 {scenario_name: {strategy: return}}
                例如：{"market_crash": {"grid": -0.10, "trend": -0.20, "scalping": -0.15}}

        Returns:
            {scenario_name: {portfolio_return, max_loss_strategy, ...}}
        """
        results = {}
        strategies = list(current_allocation.keys())
        weights = np.array([current_allocation[s] for s in strategies])

        for scenario_name, scenario_returns in stress_scenarios.items():
            returns_array = np.array([scenario_returns.get(s, 0.0) for s in strategies])
            portfolio_return = float(weights @ returns_array)

            # 找出最大亏损策略
            worst_strategy = strategies[np.argmin(returns_array)]
            worst_return = float(np.min(returns_array))

            results[scenario_name] = {
                "portfolio_return": round(portfolio_return, 4),
                "worst_strategy": worst_strategy,
                "worst_return": round(worst_return, 4),
                "total_loss": round(portfolio_return * sum(weights), 4),
            }

        return results
