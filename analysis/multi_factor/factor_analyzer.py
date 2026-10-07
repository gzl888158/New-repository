"""
因子分析器 (Factor Analyzer)

实现因子检验：
1. IC (Information Coefficient) 分析
2. 因子稳定性评估 (IC IR, 换手率)
3. 共线性检测 (相关矩阵, VIF)
4. 因子衰减测试
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
    warnings.warn("scipy not installed — IC analysis will use numpy fallback (slower)")

from loguru import logger


class FactorAnalyzer:
    """
    因子分析器

    用法：
        analyzer = FactorAnalyzer()
        ic_report = analyzer.compute_ic_report(factors, forward_returns)
        collinearity = analyzer.check_multicollinearity(factors)
    """

    def __init__(self, ic_method: str = "spearman"):
        """
        Args:
            ic_method: IC计算方法 ("spearman" 或 "pearson")
        """
        self.ic_method = ic_method

    @staticmethod
    def _spearmanr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Spearman rank correlation with numpy fallback."""
        if SCIPY_AVAILABLE:
            return stats.spearmanr(x, y)
        # Numpy fallback: rank-based Pearson correlation
        rx = pd.Series(x).rank().values
        ry = pd.Series(y).rank().values
        corr = np.corrcoef(rx, ry)[0, 1]
        return corr, 0.0  # p-value not computed in fallback

    @staticmethod
    def _pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        """Pearson correlation with numpy fallback."""
        if SCIPY_AVAILABLE:
            return stats.pearsonr(x, y)
        corr = np.corrcoef(x, y)[0, 1]
        return corr, 0.0  # p-value not computed in fallback

    def compute_ic_report(
        self,
        factors: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> pd.DataFrame:
        """
        计算IC分析报告

        Args:
            factors: 因子矩阵 (index=stock, columns=factor_names)
            forward_returns: 未来收益率 (index=stock)

        Returns:
            DataFrame with IC statistics for each factor
        """
        logger.info("[FactorAnalyzer] Computing IC report")

        results = []
        for factor_name in factors.columns:
            factor_values = factors[factor_name].dropna()
            returns = forward_returns.reindex(factor_values.index).dropna()

            # 对齐
            common_idx = factor_values.index.intersection(returns.index)
            if len(common_idx) < 30:
                logger.warning(f"[FactorAnalyzer] {factor_name}: insufficient data")
                continue

            factor_vals = factor_values.loc[common_idx]
            ret_vals = returns.loc[common_idx]

            # 计算IC
            if self.ic_method == "spearman":
                ic, p_value = self._spearmanr(factor_vals, ret_vals)
            else:
                ic, p_value = self._pearsonr(factor_vals, ret_vals)

            results.append({
                "factor": factor_name,
                "ic": ic,
                "p_value": p_value,
                "ic_abs": abs(ic),
                "significant": p_value < 0.05,
            })

        report = pd.DataFrame(results).set_index("factor")
        report = report.sort_values("ic_abs", ascending=False)

        logger.info(f"[FactorAnalyzer] IC report computed for {len(report)} factors")
        return report

    def compute_rolling_ic(
        self,
        factors_ts: pd.DataFrame,
        returns_ts: pd.Series,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        计算滚动IC (时变IC)

        Args:
            factors_ts: 因子时间序列 (MultiIndex: date, stock)
            returns_ts: 未来收益时间序列 (MultiIndex: date, stock)
            window: 滚动窗口

        Returns:
            DataFrame with rolling IC for each factor
        """
        logger.info(f"[FactorAnalyzer] Computing rolling IC with window={window}")

        dates = factors_ts.index.get_level_values(0).unique().sort_values()
        if len(dates) < window:
            logger.warning("[FactorAnalyzer] Insufficient dates for rolling IC")
            return pd.DataFrame()

        rolling_ic = []
        for i in range(window, len(dates)):
            window_dates = dates[i - window:i]
            window_factors = factors_ts.loc[window_days]
            window_returns = returns_ts.loc[window_days]

            # 计算该窗口内的IC
            ic_values = {}
            for factor_name in window_factors.columns:
                factor_vals = window_factors[factor_name].dropna()
                ret_vals = window_returns.reindex(factor_vals.index).dropna()
                common_idx = factor_vals.index.intersection(ret_vals.index)

                if len(common_idx) < 30:
                    continue

                if self.ic_method == "spearman":
                    ic, _ = self._spearmanr(factor_vals.loc[common_idx], ret_vals.loc[common_idx])
                else:
                    ic, _ = self._pearsonr(factor_vals.loc[common_idx], ret_vals.loc[common_idx])

                ic_values[factor_name] = ic

            rolling_ic.append({
                "date": dates[i],
                **ic_values,
            })

        return pd.DataFrame(rolling_ic).set_index("date")

    def compute_ic_stability(self, rolling_ic: pd.DataFrame) -> pd.DataFrame:
        """
        计算IC稳定性指标

        Args:
            rolling_ic: 滚动IC DataFrame (index=date, columns=factor_names)

        Returns:
            DataFrame with IC IR, IC turnover, etc.
        """
        logger.info("[FactorAnalyzer] Computing IC stability metrics")

        results = []
        for factor_name in rolling_ic.columns:
            ic_series = rolling_ic[factor_name].dropna()
            if len(ic_series) < 10:
                continue

            # IC均值
            ic_mean = ic_series.mean()

            # IC标准差
            ic_std = ic_series.std()

            # IC IR (Information Ratio of IC)
            ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0

            # IC > 0 比例
            ic_positive_ratio = (ic_series > 0).mean()

            # IC绝对值均值
            ic_abs_mean = ic_series.abs().mean()

            results.append({
                "factor": factor_name,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ic_ir": ic_ir,
                "ic_positive_ratio": ic_positive_ratio,
                "ic_abs_mean": ic_abs_mean,
            })

        report = pd.DataFrame(results).set_index("factor")
        return report.sort_values("ic_ir", ascending=False)

    def check_multicollinearity(
        self,
        factors: pd.DataFrame,
        threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        检测因子共线性

        Args:
            factors: 因子矩阵
            threshold: 相关性阈值

        Returns:
            Dict with correlation matrix, VIF, and high-correlation pairs
        """
        logger.info("[FactorAnalyzer] Checking multicollinearity")

        # 1. 相关矩阵
        corr_matrix = factors.corr(method="spearman")

        # 2. 高相关性因子对
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    high_corr_pairs.append({
                        "factor_1": corr_matrix.columns[i],
                        "factor_2": corr_matrix.columns[j],
                        "correlation": corr_val,
                    })

        # 3. VIF (Variance Inflation Factor)
        vif = self._compute_vif(factors)

        return {
            "correlation_matrix": corr_matrix,
            "high_correlation_pairs": high_corr_pairs,
            "vif": vif,
            "threshold": threshold,
        }

    def _compute_vif(self, factors: pd.DataFrame) -> pd.Series:
        """
        计算VIF (Variance Inflation Factor)

        VIF > 10 表示严重共线性
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        # 去除缺失值
        clean_factors = factors.dropna()
        if clean_factors.empty:
            return pd.Series(dtype=float)

        # 添加常数项
        X = clean_factors.copy()
        X["const"] = 1.0

        try:
            vif_values = []
            for i, col in enumerate(clean_factors.columns):
                col_idx = list(X.columns).index(col)
                vif = variance_inflation_factor(X.values, col_idx)
                vif_values.append({"factor": col, "vif": vif})

            return pd.DataFrame(vif_values).set_index("factor")["vif"]
        except Exception as e:
            logger.warning(f"[FactorAnalyzer] VIF computation failed: {e}")
            return pd.Series(dtype=float)

    def compute_factor_decay(
        self,
        factors_ts: pd.DataFrame,
        price_ts: pd.DataFrame,
        horizons: List[int] = [1, 5, 10, 20],
    ) -> pd.DataFrame:
        """
        计算因子衰减 (不同持有期的IC)

        Args:
            factors_ts: 因子时间序列
            price_ts: 价格时间序列
            horizons: 持有期列表

        Returns:
            DataFrame with IC for each horizon
        """
        logger.info(f"[FactorAnalyzer] Computing factor decay for horizons={horizons}")

        decay_results = []
        for horizon in horizons:
            # 计算未来horizon期收益
            forward_returns = price_ts.groupby(level=0)["close"].pct_change(horizon).shift(-horizon)

            # 计算该horizon下的IC
            ic_values = {}
            for factor_name in factors_ts.columns:
                factor_vals = factors_ts[factor_name].dropna()
                ret_vals = forward_returns.reindex(factor_vals.index).dropna()
                common_idx = factor_vals.index.intersection(ret_vals.index)

                if len(common_idx) < 30:
                    continue

                if self.ic_method == "spearman":
                    ic, _ = self._spearmanr(factor_vals.loc[common_idx], ret_vals.loc[common_idx])
                else:
                    ic, _ = self._pearsonr(factor_vals.loc[common_idx], ret_vals.loc[common_idx])

                ic_values[factor_name] = ic

            decay_results.append({
                "horizon": horizon,
                **ic_values,
            })

        return pd.DataFrame(decay_results).set_index("horizon")

    def compute_factor_turnover(
        self,
        factors_ts: pd.DataFrame,
        top_n: int = 50,
    ) -> pd.Series:
        """
        计算因子换手率 (选股稳定性)

        Args:
            factors_ts: 因子时间序列
            top_n: 选股数量

        Returns:
            Series with turnover for each factor
        """
        logger.info(f"[FactorAnalyzer] Computing factor turnover (top_n={top_n})")

        dates = factors_ts.index.get_level_values(0).unique().sort_values()
        if len(dates) < 2:
            return pd.Series(dtype=float)

        turnover_results = []
        for factor_name in factors_ts.columns:
            prev_stocks = None
            turnovers = []

            for date in dates:
                factor_vals = factors_ts.loc[date, factor_name].dropna()
                if len(factor_vals) < top_n:
                    continue

                # 选top_n (假设因子方向已调整)
                top_stocks = set(factor_vals.nlargest(top_n).index)

                if prev_stocks is not None:
                    # 换手率 = 新选入股票数 / top_n
                    new_stocks = top_stocks - prev_stocks
                    turnover = len(new_stocks) / top_n
                    turnovers.append(turnover)

                prev_stocks = top_stocks

            if turnovers:
                turnover_results.append({
                    "factor": factor_name,
                    "avg_turnover": np.mean(turnovers),
                })

        if not turnover_results:
            return pd.Series(dtype=float)

        return pd.DataFrame(turnover_results).set_index("factor")["avg_turnover"]
