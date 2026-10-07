"""
选股器 (Stock Selector)

实现因子打分与选股逻辑：
1. 因子标准化 (z-score, rank)
2. 因子加权 (等权, IC加权, 优化加权)
3. 选股过滤 (Top-N, 阈值)
4. 行业中性化 (可选)
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from loguru import logger

from .factor_engine import FactorEngine, FactorCategory


class StockSelector:
    """
    选股器

    用法：
        selector = StockSelector(factor_engine)
        selected = selector.select_stocks(factors, method="top_n", top_n=50)
    """

    def __init__(
        self,
        factor_engine: FactorEngine,
        weighting_method: str = "equal",
    ):
        """
        Args:
            factor_engine: 因子计算引擎
            weighting_method: 加权方法 ("equal", "ic_weighted", "optimized")
        """
        self.factor_engine = factor_engine
        self.weighting_method = weighting_method

        # 因子权重 (可通过set_weights手动设置)
        self.weights: Dict[str, float] = {}

    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        手动设置因子权重

        Args:
            weights: {factor_name: weight}
        """
        total = sum(weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in weights.items()}
        else:
            self.weights = weights
        logger.info(f"[StockSelector] Weights set: {len(self.weights)} factors")

    def compute_composite_score(
        self,
        factors: pd.DataFrame,
        ic_weights: Optional[Dict[str, float]] = None,
    ) -> pd.Series:
        """
        计算综合因子得分

        Args:
            factors: 因子矩阵 (已标准化)
            ic_weights: IC加权权重 (可选)

        Returns:
            Series with composite score (index=stock)
        """
        logger.info(f"[StockSelector] Computing composite score with method={self.weighting_method}")

        # 调整因子方向 (某些因子越小越好)
        adjusted_factors = self._adjust_factor_direction(factors)

        # 确定权重
        if self.weighting_method == "ic_weighted" and ic_weights:
            weights = ic_weights
        elif self.weighting_method == "optimized" and self.weights:
            weights = self.weights
        else:
            # 等权
            weights = {col: 1.0 / len(adjusted_factors.columns) for col in adjusted_factors.columns}

        # 加权求和
        composite = pd.Series(0.0, index=adjusted_factors.index)
        for factor_name, weight in weights.items():
            if factor_name in adjusted_factors.columns:
                composite += adjusted_factors[factor_name].fillna(0.0) * weight

        return composite

    def _adjust_factor_direction(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        调整因子方向 (统一为越大越好)

        对于方向为-1的因子，取负值
        """
        adjusted = factors.copy()

        for col in adjusted.columns:
            direction = self.factor_engine.get_factor_direction(col)
            if direction == -1:
                adjusted[col] = -adjusted[col]

        return adjusted

    def select_stocks(
        self,
        factors: pd.DataFrame,
        method: str = "top_n",
        top_n: int = 50,
        threshold: float = 0.0,
        ic_weights: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        选股

        Args:
            factors: 因子矩阵
            method: 选股方法 ("top_n", "threshold")
            top_n: 选前N只 (method="top_n")
            threshold: 得分阈值 (method="threshold")
            ic_weights: IC加权权重

        Returns:
            List of selected stock codes
        """
        logger.info(f"[StockSelector] Selecting stocks with method={method}")

        # 计算综合得分
        composite = self.compute_composite_score(factors, ic_weights)

        if method == "top_n":
            selected = composite.nlargest(top_n).index.tolist()
        elif method == "threshold":
            selected = composite[composite >= threshold].index.tolist()
        else:
            raise ValueError(f"Unknown method: {method}")

        logger.info(f"[StockSelector] Selected {len(selected)} stocks")
        return selected

    def compute_factor_contribution(
        self,
        factors: pd.DataFrame,
        ic_weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        计算各因子对综合得分的贡献

        Returns:
            DataFrame with factor contributions
        """
        adjusted_factors = self._adjust_factor_direction(factors)

        # 确定权重
        if self.weighting_method == "ic_weighted" and ic_weights:
            weights = ic_weights
        elif self.weighting_method == "optimized" and self.weights:
            weights = self.weights
        else:
            weights = {col: 1.0 / len(adjusted_factors.columns) for col in adjusted_factors.columns}

        # 计算贡献
        contributions = {}
        for factor_name, weight in weights.items():
            if factor_name in adjusted_factors.columns:
                contributions[factor_name] = adjusted_factors[factor_name].fillna(0.0) * weight

        return pd.DataFrame(contributions)

    def apply_industry_neutral(
        self,
        factors: pd.DataFrame,
        industry_map: Dict[str, str],
    ) -> pd.DataFrame:
        """
        行业中性化 (因子在行业内标准化)

        Args:
            factors: 因子矩阵
            industry_map: {stock: industry}

        Returns:
            Industry-neutralized factors
        """
        logger.info("[StockSelector] Applying industry neutralization")

        neutral_factors = factors.copy()

        # 按行业分组标准化
        industry_df = pd.DataFrame({
            "industry": industry_map,
        })

        for col in neutral_factors.columns:
            for industry in industry_df["industry"].unique():
                mask = industry_df["industry"] == industry
                series = neutral_factors.loc[mask, col]

                # 行业内z-score
                mean = series.mean()
                std = series.std()
                if std > 0:
                    neutral_factors.loc[mask, col] = (series - mean) / std
                else:
                    neutral_factors.loc[mask, col] = 0.0

        return neutral_factors

    def compute_stock_stats(
        self,
        factors: pd.DataFrame,
        selected_stocks: List[str],
    ) -> pd.DataFrame:
        """
        计算选中股票的因子统计

        Returns:
            DataFrame with factor statistics for selected stocks
        """
        selected_factors = factors.loc[factors.index.isin(selected_stocks)]

        stats = {
            "mean": selected_factors.mean(),
            "median": selected_factors.median(),
            "std": selected_factors.std(),
            "min": selected_factors.min(),
            "max": selected_factors.max(),
        }

        return pd.DataFrame(stats)

    def generate_selection_report(
        self,
        factors: pd.DataFrame,
        selected_stocks: List[str],
        composite_scores: pd.Series,
    ) -> Dict[str, Any]:
        """
        生成选股报告

        Returns:
            Dict with selection summary
        """
        report = {
            "total_stocks": len(factors),
            "selected_count": len(selected_stocks),
            "selection_rate": len(selected_stocks) / len(factors) if len(factors) > 0 else 0,
            "composite_score_stats": {
                "mean": composite_scores.mean(),
                "median": composite_scores.median(),
                "std": composite_scores.std(),
            },
            "selected_stocks": selected_stocks,
            "top_10_scores": composite_scores.nlargest(10).to_dict(),
        }

        return report
