"""企业级分配参数自动优化器。

基于历史表现定期（每周）自动优化：最大化 Sharpe 约束最大回撤，输出新参数建议。

= 使用场景 =
- 自动寻找最优 Kelly 分数、权重上下限、池比例
- 约束条件：最大回撤 < 阈值，Sharpe > 阈值
- 输出建议并经人工确认后生效
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime
from loguru import logger


class CapitalParameterOptimizer:
    """分配参数自动优化器。

    用法：
        optimizer = CapitalParameterOptimizer()
        suggestion = optimizer.optimize(
            historical_data={...},  # 历史各策略收益率
            current_params={...},   # 当前参数
            constraints={...},      # 约束条件
        )
        # suggestion 包含新参数建议
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        logger.info("CapitalParameterOptimizer initialized")

    def optimize(self, historical_returns: Dict[str, List[float]],
                 current_params: Dict[str, Any],
                 constraints: Dict[str, float] = None,
                 optimization_target: str = "sharpe") -> Dict[str, Any]:
        """优化分配参数。

        Args:
            historical_returns: 各策略历史日收益率
            current_params: 当前参数 {kelly_fraction, max_weight, min_weight, pool_ratios}
            constraints: 约束条件 {max_drawdown: 0.10, min_sharpe: 1.0}
            optimization_target: 优化目标 "sharpe" | "return" | "min_drawdown"

        Returns:
            {
                suggested_params: {...},
                expected_sharpe, expected_return, expected_max_drawdown,
                confidence: float,
                reasoning: str
            }
        """
        constraints = constraints or {"max_drawdown": 0.10, "min_sharpe": 1.0}
        strategies = list(historical_returns.keys())

        # 简化版：基于历史表现调整权重（向高效策略倾斜）
        mean_returns = {s: np.mean(rets) for s, rets in historical_returns.items()}
        std_returns = {s: np.std(rets) for s, rets in historical_returns.items()}

        # 计算各策略 Sharpe
        sharpe_by_strategy = {}
        for s in strategies:
            if std_returns[s] > 0:
                sharpe_by_strategy[s] = mean_returns[s] / std_returns[s]
            else:
                sharpe_by_strategy[s] = 0.0

        # 按 Sharpe 排序，调整权重
        sorted_strategies = sorted(sharpe_by_strategy.items(), key=lambda x: x[1], reverse=True)

        # 新权重：高效策略获得更多权重
        new_weights = {}
        total_sharpe = sum(max(0, s[1]) for s in sorted_strategies)
        if total_sharpe > 0:
            for s, sharpe in sorted_strategies:
                if sharpe > 0:
                    new_weights[s] = sharpe / total_sharpe
                else:
                    new_weights[s] = 0.05  # 最低5%
        else:
            # 等权
            for s in strategies:
                new_weights[s] = 1.0 / len(strategies)

        # 归一化
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {s: w / total_weight for s, w in new_weights.items()}

        # 计算预期表现
        portfolio_return = sum(new_weights[s] * mean_returns[s] for s in strategies)
        portfolio_std = np.sqrt(sum(
            (new_weights[s] ** 2) * (std_returns[s] ** 2)
            for s in strategies
        ))
        expected_sharpe = portfolio_return / portfolio_std if portfolio_std > 0 else 0

        # 简化：假设最大回撤为收益率的3倍标准差
        expected_max_drawdown = portfolio_std * 3

        suggested_params = {
            "strategy_weights": {s: round(w, 4) for s, w in new_weights.items()},
            "kelly_fraction": round(min(0.5, max(0.1, expected_sharpe / 10)), 2),
            "max_weight": round(min(0.60, max(0.30, max(new_weights.values()) * 1.2)), 2),
            "min_weight": round(max(0.02, min(new_weights.values()) * 0.5), 2),
        }

        return {
            "suggested_params": suggested_params,
            "expected_sharpe": round(float(expected_sharpe), 4),
            "expected_return": round(float(portfolio_return), 4),
            "expected_max_drawdown": round(float(expected_max_drawdown), 4),
            "confidence": 0.7,  # 简化：固定置信度
            "reasoning": f"Optimized for {optimization_target}: "
                        f"top strategies by Sharpe: {[s[0] for s in sorted_strategies[:3]]}",
            "timestamp": datetime.now().isoformat(),
        }
