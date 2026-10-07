"""
回测偏差防范与样本外测试 (Backtest Bias Prevention & Out-of-Sample Testing)

实现：
1. 未来函数检测 (point-in-time数据)
2. 幸存者偏差处理 (包含退市股票)
3. 数据窥探检测 (多重检验校正)
4. 样本外测试 (walk-forward分析)
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from loguru import logger


class BiasPrevention:
    """
    回测偏差防范工具

    用法：
        bp = BiasPrevention()
        bp.check_lookahead_bias(strategy_code)
        bp.adjust_for_survivorship_bias(stock_pool)
    """

    def __init__(self):
        pass

    def check_lookahead_bias(
        self,
        data_timestamps: pd.Series,
        trading_dates: pd.Series,
    ) -> Dict[str, Any]:
        """
        检测未来函数

        Args:
            data_timestamps: 数据实际发布时间
            trading_dates: 交易决策日期

        Returns:
            Dict with violation report
        """
        logger.info("[BiasPrevention] Checking look-ahead bias")

        violations = []
        for i, (data_ts, trade_date) in enumerate(zip(data_timestamps, trading_dates)):
            if data_ts > trade_date:
                violations.append({
                    "index": i,
                    "data_timestamp": data_ts,
                    "trading_date": trade_date,
                    "days_ahead": (data_ts - trade_date).days,
                })

        report = {
            "total_checks": len(data_timestamps),
            "violations": len(violations),
            "violation_rate": len(violations) / len(data_timestamps) if len(data_timestamps) > 0 else 0,
            "violations_detail": violations,
        }

        if violations:
            logger.warning(f"[BiasPrevention] Found {len(violations)} look-ahead violations")
        else:
            logger.info("[BiasPrevention] No look-ahead bias detected")

        return report

    def adjust_for_survivorship_bias(
        self,
        current_stock_pool: List[str],
        historical_delisted: Optional[List[str]] = None,
    ) -> List[str]:
        """
        调整幸存者偏差

        在回测开始时，应包含后来退市的股票

        Args:
            current_stock_pool: 当前在市的股票
            historical_delisted: 历史退市股票列表

        Returns:
            Adjusted stock pool (包括退市股票)
        """
        logger.info("[BiasPrevention] Adjusting for survivorship bias")

        if historical_delisted is None:
            logger.warning("[BiasPrevention] No delisted stocks provided")
            return current_stock_pool

        adjusted_pool = list(set(current_stock_pool + historical_delisted))
        logger.info(f"[BiasPrevention] Adjusted pool: {len(current_stock_pool)} -> {len(adjusted_pool)}")

        return adjusted_pool

    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Bonferroni校正 (多重检验校正)

        防止数据窥探 (data snooping)

        Args:
            p_values: 原始p值列表
            alpha: 显著性水平

        Returns:
            Dict with adjusted p-values
        """
        logger.info(f"[BiasPrevention] Applying Bonferroni correction for {len(p_values)} tests")

        n_tests = len(p_values)
        adjusted_alpha = alpha / n_tests

        adjusted_p = [min(p * n_tests, 1.0) for p in p_values]
        significant = [p < adjusted_alpha for p in adjusted_p]

        return {
            "n_tests": n_tests,
            "original_alpha": alpha,
            "adjusted_alpha": adjusted_alpha,
            "adjusted_p_values": adjusted_p,
            "significant_after_correction": sum(significant),
            "significant_rate": sum(significant) / n_tests if n_tests > 0 else 0,
        }

    def fdr_correction(
        self,
        p_values: List[float],
        q: float = 0.05,
    ) -> Dict[str, Any]:
        """
        FDR (False Discovery Rate) 校正

        比Bonferroni更宽松，适用于探索性分析

        Args:
            p_values: 原始p值列表
            q: FDR阈值

        Returns:
            Dict with adjusted results
        """
        logger.info(f"[BiasPrevention] Applying FDR correction for {len(p_values)} tests")

        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_indices]

        # Benjamini-Hochberg procedure
        thresholds = [q * (i + 1) / n_tests for i in range(n_tests)]
        significant = sorted_p <= thresholds

        # 找到最大的k使得p_k <= threshold_k
        max_k = -1
        for i in range(n_tests - 1, -1, -1):
            if significant[i]:
                max_k = i
                break

        # 所有i <= max_k的检验显著
        final_significant = np.zeros(n_tests, dtype=bool)
        if max_k >= 0:
            final_significant[sorted_indices[:max_k + 1]] = True

        return {
            "n_tests": n_tests,
            "q": q,
            "significant_count": sum(final_significant),
            "significant_rate": sum(final_significant) / n_tests if n_tests > 0 else 0,
            "significant_indices": np.where(final_significant)[0].tolist(),
        }


class WalkForwardAnalyzer:
    """
    Walk-Forward分析器 (样本外测试)

    实现滚动样本外测试，防止过拟合
    """

    def __init__(
        self,
        train_window: int = 252 * 3,  # 3年训练窗口
        test_window: int = 252,        # 1年测试窗口
        step_size: int = 252,          # 步进1年
    ):
        """
        Args:
            train_window: 训练窗口 (交易日)
            test_window: 测试窗口 (交易日)
            step_size: 步进大小
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        metric_func: Callable,
    ) -> pd.DataFrame:
        """
        运行Walk-Forward分析

        Args:
            data: 时间序列数据 (index=date)
            strategy_func: 策略函数 (train_data) -> model
            metric_func: 评估函数 (model, test_data) -> metric

        Returns:
            DataFrame with out-of-sample metrics for each fold
        """
        logger.info("[WalkForward] Running walk-forward analysis")

        dates = data.index.sort_values()
        if len(dates) < self.train_window + self.test_window:
            logger.error("[WalkForward] Insufficient data")
            return pd.DataFrame()

        results = []
        start_idx = 0

        while start_idx + self.train_window + self.test_window <= len(dates):
            # 划分训练集和测试集
            train_end_idx = start_idx + self.train_window
            test_end_idx = train_end_idx + self.test_window

            train_dates = dates[start_idx:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]

            train_data = data.loc[train_dates]
            test_data = data.loc[test_dates]

            # 训练策略
            model = strategy_func(train_data)

            # 测试评估
            metric = metric_func(model, test_data)

            results.append({
                "fold": len(results),
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                **metric,
            })

            # 步进
            start_idx += self.step_size

        logger.info(f"[WalkForward] Completed {len(results)} folds")
        return pd.DataFrame(results)

    def compute_robustness_stats(self, wf_results: pd.DataFrame) -> Dict[str, Any]:
        """
        计算Walk-Forward稳健性统计

        Args:
            wf_results: Walk-Forward结果

        Returns:
            Dict with robustness metrics
        """
        if wf_results.empty:
            return {}

        # 假设结果中有"return"列
        if "return" not in wf_results.columns:
            logger.warning("[WalkForward] No 'return' column found")
            return {}

        returns = wf_results["return"].dropna()

        stats = {
            "n_folds": len(returns),
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "positive_rate": (returns > 0).mean(),
            "min_return": returns.min(),
            "max_return": returns.max(),
            "sharpe": returns.mean() / returns.std() if returns.std() > 0 else 0.0,
        }

        return stats


class DataSnoopingDetector:
    """
    数据窥探检测器

    检测是否过度拟合历史数据
    """

    def __init__(self):
        pass

    def deflated_sharpe_ratio(
        self,
        sharpe: float,
        n_trials: int,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Deflated Sharpe Ratio (DSR)

        调整多重检验后的夏普比率

        Args:
            sharpe: 原始夏普比率
            n_trials: 试验次数 (测试了多少策略)
            skewness: 收益偏度
            kurtosis: 收益峰度

        Returns:
            Deflated Sharpe Ratio
        """
        logger.info(f"[DataSnooping] Computing DSR with n_trials={n_trials}")

        # 期望最大夏普比率 (假设n_trials次独立试验)
        expected_max_sharpe = stats.norm.ppf(1 - 1 / n_trials)

        # 调整后的夏普比率
        dsr = sharpe / expected_max_sharpe if expected_max_sharpe > 0 else 0.0

        logger.info(f"[DataSnooping] DSR: {dsr:.3f} (original: {sharpe:.3f})")
        return dsr

    def probability_of_backtest_overfitting(
        self,
        sharpe: float,
        n_trials: int,
        n_years: int,
    ) -> float:
        """
        计算回测过拟合概率

        基于Bailey和López de Prado (2014)

        Args:
            sharpe: 回测夏普比率
            n_trials: 试验次数
            n_years: 回测年数

        Returns:
            过拟合概率
        """
        logger.info(f"[DataSnooping] Computing PBO with n_trials={n_trials}, n_years={n_years}")

        # 简化版本：基于多重检验校正
        # 实际实现应更复杂

        # 假设真实夏普为0，计算观测到sharpe的概率
        p_value = 1 - stats.norm.cdf(sharpe * np.sqrt(n_years * 252))

        # 调整多重检验
        adjusted_p = min(p_value * n_trials, 1.0)

        logger.info(f"[DataSnooping] PBO: {adjusted_p:.3f}")
        return adjusted_p
