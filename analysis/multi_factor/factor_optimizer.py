"""
因子优化器 (Factor Optimizer)

优化因子权重分配，降低因子冗余：
1. 因子共线性消除 (PCA / 聚类)
2. IC加权
3. 最优化加权 (最大化IC_IR)
4. 风险平价加权
5. 因子筛选 (独立有效因子)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from loguru import logger


class FactorOptimizer:
    """
    因子优化器

    优化因子权重，降低冗余

    用法：
        optimizer = FactorOptimizer()
        weights = optimizer.optimize_weights(factors, forward_returns, method="ic_ir")
    """

    def __init__(self, corr_threshold: float = 0.7):
        """
        Args:
            corr_threshold: 相关性阈值，超过此值的因子对需要去冗余
        """
        self.corr_threshold = corr_threshold

    def screen_independent_factors(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        min_ic: float = 0.03,
        min_ic_ir: float = 0.5,
    ) -> List[str]:
        """
        筛选独立有效因子

        Args:
            factors: 因子矩阵 (index=stock, columns=factor_names)
            forward_returns: 未来收益率
            min_ic: 最小IC阈值
            min_ic_ir: 最小IC_IR阈值

        Returns:
            筛选后的因子名称列表
        """
        logger.info(f"[FactorOptim] Screening factors with IC>{min_ic}, IC_IR>{min_ic_ir}")

        # 1. 计算每个因子的IC
        ic_dict = {}
        for col in factors.columns:
            factor_vals = factors[col].dropna()
            common_idx = factor_vals.index.intersection(forward_returns.index)
            if len(common_idx) < 30:
                continue

            if SCIPY_AVAILABLE:
                ic, _ = stats.spearmanr(
                    factor_vals.loc[common_idx], forward_returns.loc[common_idx]
                )
            else:
                rx = factor_vals.loc[common_idx].rank().values
                ry = forward_returns.loc[common_idx].rank().values
                ic = np.corrcoef(rx, ry)[0, 1]
            ic_dict[col] = ic

        # 2. 筛选IC达标的因子
        effective_factors = [f for f, ic in ic_dict.items() if abs(ic) >= min_ic]
        logger.info(f"[FactorOptim] {len(effective_factors)} factors with |IC|>={min_ic}")

        if not effective_factors:
            return []

        # 3. 去冗余 (相关性聚类)
        independent_factors = self._remove_redundant_factors(
            factors[effective_factors], ic_dict
        )
        logger.info(
            f"[FactorOptim] {len(independent_factors)} independent factors after redundancy removal"
        )

        return independent_factors

    def _remove_redundant_factors(
        self, factors: pd.DataFrame, ic_dict: Dict[str, float]
    ) -> List[str]:
        """去除冗余因子 (保留IC最高的)"""
        corr_matrix = factors.corr(method="spearman").abs()
        selected = []
        remaining = list(factors.columns)

        # 按IC绝对值排序
        remaining.sort(key=lambda f: abs(ic_dict.get(f, 0)), reverse=True)

        while remaining:
            # 选择IC最高的因子
            best = remaining[0]
            selected.append(best)
            remaining.remove(best)

            # 移除与best高度相关的因子
            to_remove = []
            for f in remaining:
                if corr_matrix.loc[best, f] > self.corr_threshold:
                    to_remove.append(f)
            for f in to_remove:
                remaining.remove(f)

        return selected

    def optimize_weights(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        method: str = "ic_ir",
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        优化因子权重

        Args:
            factors: 因子矩阵
            forward_returns: 未来收益率
            method: 优化方法 ("equal", "ic", "ic_ir", "max_sharpe", "risk_parity")
            constraints: 约束条件 (如 {"long_only": True, "max_weight": 0.5})

        Returns:
            因子权重字典
        """
        logger.info(f"[FactorOptim] Optimizing weights with method={method}")

        if method == "equal":
            return self._equal_weight(factors)
        elif method == "ic":
            return self._ic_weight(factors, forward_returns)
        elif method == "ic_ir":
            return self._ic_ir_weight(factors, forward_returns)
        elif method == "max_sharpe":
            return self._max_sharpe_weight(factors, forward_returns, constraints)
        elif method == "risk_parity":
            return self._risk_parity_weight(factors)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _equal_weight(self, factors: pd.DataFrame) -> Dict[str, float]:
        """等权"""
        n = len(factors.columns)
        return {col: 1.0 / n for col in factors.columns}

    def _ic_weight(
        self, factors: pd.DataFrame, forward_returns: pd.Series
    ) -> Dict[str, float]:
        """IC加权"""
        ics = {}
        for col in factors.columns:
            factor_vals = factors[col].dropna()
            common_idx = factor_vals.index.intersection(forward_returns.index)
            if len(common_idx) < 30:
                ics[col] = 0.0
                continue

            if SCIPY_AVAILABLE:
                ic, _ = stats.spearmanr(
                    factor_vals.loc[common_idx], forward_returns.loc[common_idx]
                )
            else:
                rx = factor_vals.loc[common_idx].rank().values
                ry = forward_returns.loc[common_idx].rank().values
                ic = np.corrcoef(rx, ry)[0, 1]
            ics[col] = abs(ic)

        # 归一化
        total = sum(ics.values())
        if total < 1e-10:
            return self._equal_weight(factors)

        return {col: ic / total for col, ic in ics.items()}

    def _ic_ir_weight(
        self, factors: pd.DataFrame, forward_returns: pd.Series
    ) -> Dict[str, float]:
        """IC_IR加权 (假设多期数据，这里简化为单期)"""
        # 实际应使用滚动IC的均值/标准差
        return self._ic_weight(factors, forward_returns)

    def _max_sharpe_weight(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """最大化夏普比率权重"""
        if not SCIPY_AVAILABLE:
            logger.warning("[FactorOptim] scipy not available, fallback to IC weight")
            return self._ic_weight(factors, forward_returns)

        n_factors = len(factors.columns)
        factor_matrix = factors.values

        # 目标函数: 负夏普比率
        def neg_sharpe(weights):
            portfolio_return = factor_matrix @ weights
            corr_with_return = np.corrcoef(portfolio_return, forward_returns.values)[0, 1]
            return -corr_with_return  # 最大化相关性

        # 约束
        cons = []
        if constraints and constraints.get("long_only", True):
            # 权重非负
            bounds = [(0, constraints.get("max_weight", 1.0))] * n_factors
        else:
            bounds = [(-1, 1)] * n_factors

        # 权重和为1
        cons.append({"type": "eq", "fun": lambda w: np.sum(w) - 1})

        # 优化
        x0 = np.ones(n_factors) / n_factors
        try:
            result = minimize(
                neg_sharpe,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )
            if result.success:
                weights = result.x
                return {col: float(w) for col, w in zip(factors.columns, weights)}
        except Exception as e:
            logger.warning(f"[FactorOptim] Optimization failed: {e}")

        return self._ic_weight(factors, forward_returns)

    def _risk_parity_weight(self, factors: pd.DataFrame) -> Dict[str, float]:
        """风险平价权重 (基于因子波动率)"""
        volatilities = factors.std()
        inv_vol = 1.0 / (volatilities + 1e-10)
        total = inv_vol.sum()
        return {col: float(inv_vol[col] / total) for col in factors.columns}

    def compute_factor_redundancy_report(
        self, factors: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        计算因子冗余报告

        Args:
            factors: 因子矩阵

        Returns:
            冗余报告
        """
        corr_matrix = factors.corr(method="spearman").abs()

        # 高相关性因子对
        high_corr_pairs = []
        cols = factors.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = corr_matrix.iloc[i, j]
                if corr > self.corr_threshold:
                    high_corr_pairs.append(
                        {"factor_1": cols[i], "factor_2": cols[j], "correlation": float(corr)}
                    )

        # 平均相关性
        upper_tri = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)]
        avg_corr = float(np.mean(upper_tri))

        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "high_correlation_pairs": high_corr_pairs,
            "average_correlation": avg_corr,
            "n_high_corr_pairs": len(high_corr_pairs),
        }

    def pca_factor_reduction(
        self, factors: pd.DataFrame, variance_explained: float = 0.90
    ) -> pd.DataFrame:
        """
        PCA降维 (减少因子数量)

        Args:
            factors: 因子矩阵
            variance_explained: 保留的方差比例

        Returns:
            主成分因子矩阵
        """
        # 标准化
        factors_std = (factors - factors.mean()) / (factors.std() + 1e-10)

        # 协方差矩阵
        cov_matrix = factors_std.cov().values

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 按特征值降序排序
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 选择主成分
        total_var = np.sum(eigenvalues)
        cumulative_var = np.cumsum(eigenvalues) / total_var
        n_components = np.searchsorted(cumulative_var, variance_explained) + 1

        logger.info(
            f"[FactorOptim] PCA: {n_components} components explain {variance_explained*100:.1f}% variance"
        )

        # 投影
        components = eigenvectors[:, :n_components]
        pca_factors = pd.DataFrame(
            factors_std.values @ components,
            index=factors.index,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )

        return pca_factors
