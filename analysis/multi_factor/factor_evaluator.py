"""
因子评估器 (Factor Evaluator)

评估因子的经济逻辑和统计显著性：
1. 经济逻辑评估 (单调性、经济显著性)
2. 因子分层回测 (分组收益)
3. 因子衰减分析
4. 因子交互效应
5. 因子稳健性检验
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from loguru import logger


class FactorEvaluator:
    """
    因子评估器

    评估因子的经济逻辑和统计质量

    用法：
        evaluator = FactorEvaluator()
        report = evaluator.evaluate_factor(factor_values, forward_returns, factor_name="ep")
    """

    def __init__(self, n_groups: int = 5):
        """
        Args:
            n_groups: 分层回测组数 (默认5组)
        """
        self.n_groups = n_groups

    def evaluate_factor(
        self,
        factor_values: pd.Series,
        forward_returns: pd.Series,
        factor_name: str = "factor",
        date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        综合评估因子

        Args:
            factor_values: 因子值 (index=stock)
            forward_returns: 未来收益率 (index=stock)
            factor_name: 因子名称
            date: 日期

        Returns:
            因子评估报告
        """
        logger.info(f"[FactorEval] Evaluating factor: {factor_name}")

        # 对齐数据
        common_idx = factor_values.index.intersection(forward_returns.index)
        factor_values = factor_values.loc[common_idx].dropna()
        forward_returns = forward_returns.loc[common_idx]
        common_idx = factor_values.index.intersection(forward_returns.index)
        factor_values = factor_values.loc[common_idx]
        forward_returns = forward_returns.loc[common_idx]

        if len(common_idx) < 30:
            logger.warning(f"[FactorEval] Insufficient data: {len(common_idx)} stocks")
            return {"error": "insufficient_data", "n_stocks": len(common_idx)}

        report = {
            "factor_name": factor_name,
            "date": date,
            "n_stocks": len(common_idx),
        }

        # 1. 统计显著性
        report["statistical_significance"] = self._assess_statistical_significance(
            factor_values, forward_returns
        )

        # 2. 经济显著性 (分层回测)
        report["economic_significance"] = self._assess_economic_significance(
            factor_values, forward_returns
        )

        # 3. 单调性检验
        report["monotonicity"] = self._assess_monotonicity(
            factor_values, forward_returns
        )

        # 4. 因子衰减
        report["decay_profile"] = self._assess_decay_profile(
            factor_values, forward_returns
        )

        # 5. 稳健性 (不同时间段)
        report["robustness"] = self._assess_robustness(
            factor_values, forward_returns
        )

        # 6. 综合评分
        report["composite_score"] = self._compute_composite_score(report)

        logger.info(
            f"[FactorEval] {factor_name}: composite_score={report['composite_score']:.3f}"
        )
        return report

    def _assess_statistical_significance(
        self, factor_values: pd.Series, forward_returns: pd.Series
    ) -> Dict[str, float]:
        """评估统计显著性"""
        # Rank IC
        if SCIPY_AVAILABLE:
            ic, p_value = stats.spearmanr(factor_values, forward_returns)
        else:
            rx = factor_values.rank().values
            ry = forward_returns.rank().values
            ic = np.corrcoef(rx, ry)[0, 1]
            p_value = 0.0

        # t-statistic
        n = len(factor_values)
        t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2 + 1e-10)

        return {
            "rank_ic": float(ic),
            "p_value": float(p_value),
            "t_statistic": float(t_stat),
            "significant_5pct": abs(t_stat) > 1.96,
            "significant_1pct": abs(t_stat) > 2.58,
        }

    def _assess_economic_significance(
        self, factor_values: pd.Series, forward_returns: pd.Series
    ) -> Dict[str, Any]:
        """评估经济显著性 (分层回测)"""
        # 分层
        groups = pd.qcut(
            factor_values, q=self.n_groups, labels=False, duplicates="drop"
        )
        group_returns = forward_returns.groupby(groups).mean()

        # 多空收益 (top - bottom)
        long_short_return = group_returns.iloc[-1] - group_returns.iloc[0]

        # 单调性 (各组收益是否单调递增/递减)
        diffs = group_returns.diff().dropna()
        monotonic = (diffs > 0).all() or (diffs < 0).all()

        return {
            "group_returns": group_returns.to_dict(),
            "long_short_return": float(long_short_return),
            "long_short_annualized": float(long_short_return * 12),  # 月度调仓
            "monotonic": monotonic,
            "economic_meaningful": abs(long_short_return) > 0.01,  # >1%月度
        }

    def _assess_monotonicity(
        self, factor_values: pd.Series, forward_returns: pd.Series
    ) -> Dict[str, Any]:
        """评估单调性 (Spearman相关 + 分层单调性)"""
        groups = pd.qcut(
            factor_values, q=self.n_groups, labels=False, duplicates="drop"
        )
        group_returns = forward_returns.groupby(groups).mean()

        # Spearman相关 (组号 vs 收益)
        if SCIPY_AVAILABLE:
            rank_corr, _ = stats.spearmanr(group_returns.index, group_returns.values)
        else:
            rank_corr = np.corrcoef(group_returns.index, group_returns.values)[0, 1]

        # Kendall tau
        if SCIPY_AVAILABLE:
            kendall_tau, _ = stats.kendalltau(factor_values, forward_returns)
        else:
            kendall_tau = rank_corr  # Fallback

        return {
            "rank_correlation": float(rank_corr),
            "kendall_tau": float(kendall_tau),
            "strictly_monotonic": (
                (group_returns.diff().dropna() > 0).all()
                or (group_returns.diff().dropna() < 0).all()
            ),
            "monotonicity_score": abs(rank_corr),
        }

    def _assess_decay_profile(
        self, factor_values: pd.Series, forward_returns: pd.Series
    ) -> Dict[str, float]:
        """评估因子衰减 (简化版，实际应使用多期收益)"""
        # 这里只计算单期IC，实际应传入多期收益数据
        if SCIPY_AVAILABLE:
            ic, _ = stats.spearmanr(factor_values, forward_returns)
        else:
            rx = factor_values.rank().values
            ry = forward_returns.rank().values
            ic = np.corrcoef(rx, ry)[0, 1]

        return {
            "ic_1period": float(ic),
            "note": "Full decay profile requires multi-period returns data",
        }

    def _assess_robustness(
        self, factor_values: pd.Series, forward_returns: pd.Series
    ) -> Dict[str, Any]:
        """评估因子稳健性 (子样本测试)"""
        n = len(factor_values)
        if n < 100:
            return {"robust": False, "reason": "insufficient_data"}

        # 随机抽样子样本
        np.random.seed(42)
        sub_ics = []
        for _ in range(10):
            idx = np.random.choice(n, size=int(0.7 * n), replace=False)
            sub_factor = factor_values.iloc[idx]
            sub_return = forward_returns.iloc[idx]

            if SCIPY_AVAILABLE:
                ic, _ = stats.spearmanr(sub_factor, sub_return)
            else:
                rx = sub_factor.rank().values
                ry = sub_return.rank().values
                ic = np.corrcoef(rx, ry)[0, 1]
            sub_ics.append(ic)

        sub_ics = np.array(sub_ics)
        ic_std = np.std(sub_ics)
        ic_mean = np.mean(sub_ics)

        return {
            "subsample_ic_mean": float(ic_mean),
            "subsample_ic_std": float(ic_std),
            "ic_cv": float(ic_std / (abs(ic_mean) + 1e-10)),  # 变异系数
            "robust": ic_std < 0.1 and abs(ic_mean) > 0.03,
        }

    def _compute_composite_score(self, report: Dict[str, Any]) -> float:
        """计算综合评分 (0-100)"""
        score = 0.0

        # 统计显著性 (30分)
        stat_sig = report.get("statistical_significance", {})
        if stat_sig.get("significant_5pct"):
            score += 20
        if stat_sig.get("significant_1pct"):
            score += 10

        # 经济显著性 (30分)
        econ_sig = report.get("economic_significance", {})
        if econ_sig.get("economic_meaningful"):
            score += 20
        if econ_sig.get("monotonic"):
            score += 10

        # 单调性 (20分)
        monotonicity = report.get("monotonicity", {})
        score += min(20, monotonicity.get("monotonicity_score", 0) * 100)

        # 稳健性 (20分)
        robustness = report.get("robustness", {})
        if robustness.get("robust"):
            score += 20

        return min(100, score)

    def generate_factor_report(
        self, evaluation_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        生成因子评估报告

        Args:
            evaluation_results: 多个因子的评估结果列表

        Returns:
            因子评估报告DataFrame
        """
        rows = []
        for result in evaluation_results:
            if "error" in result:
                continue

            row = {
                "factor_name": result["factor_name"],
                "n_stocks": result["n_stocks"],
                "rank_ic": result["statistical_significance"]["rank_ic"],
                "t_stat": result["statistical_significance"]["t_statistic"],
                "significant": result["statistical_significance"]["significant_5pct"],
                "long_short_return": result["economic_significance"][
                    "long_short_return"
                ],
                "monotonic": result["economic_significance"]["monotonic"],
                "monotonicity_score": result["monotonicity"]["monotonicity_score"],
                "robust": result["robustness"].get("robust", False),
                "composite_score": result["composite_score"],
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("composite_score", ascending=False)
        return df
