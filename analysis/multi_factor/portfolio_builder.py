"""
组合构建器 (Portfolio Builder)

基于因子得分构建投资组合：
1. 等权组合
2. 因子得分加权
3. 均值-方差优化
4. 风险平价组合
5. 行业中性组合
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


class PortfolioBuilder:
    """
    组合构建器

    基于因子得分构建投资组合

    用法：
        builder = PortfolioBuilder()
        portfolio = builder.build_portfolio(stock_scores, method="top_n", top_n=50)
    """

    def __init__(self, max_weight: float = 0.1):
        """
        Args:
            max_weight: 单只股票最大权重 (默认10%)
        """
        self.max_weight = max_weight

    def build_portfolio(
        self,
        stock_scores: pd.Series,
        method: str = "top_n",
        top_n: int = 50,
        threshold: Optional[float] = None,
        industry: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """
        构建投资组合

        Args:
            stock_scores: 股票得分 (index=stock, value=score)
            method: 构建方法 ("top_n", "threshold", "equal", "score_weight", "optimize")
            top_n: 选股数量 (for top_n method)
            threshold: 得分阈值 (for threshold method)
            industry: 行业分类 (index=stock, value=industry)

        Returns:
            投资组合权重字典 {stock: weight}
        """
        logger.info(f"[PortfolioBuilder] Building portfolio with method={method}")

        # 排序
        stock_scores = stock_scores.dropna().sort_values(ascending=False)

        if method == "top_n":
            selected = stock_scores.head(top_n)
            return self._equal_weight(selected)
        elif method == "threshold":
            if threshold is None:
                threshold = stock_scores.median()
            selected = stock_scores[stock_scores >= threshold]
            return self._equal_weight(selected)
        elif method == "equal":
            return self._equal_weight(stock_scores)
        elif method == "score_weight":
            return self._score_weight(stock_scores)
        elif method == "optimize":
            return self._optimize_weight(stock_scores)
        elif method == "industry_neutral":
            if industry is None:
                logger.warning("[PortfolioBuilder] No industry data, fallback to equal weight")
                return self._equal_weight(stock_scores.head(top_n))
            return self._industry_neutral_weight(stock_scores, industry, top_n)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _equal_weight(self, stocks: pd.Series) -> Dict[str, float]:
        """等权"""
        n = len(stocks)
        weight = 1.0 / n
        return {stock: weight for stock in stocks.index}

    def _score_weight(self, stock_scores: pd.Series) -> Dict[str, float]:
        """得分加权"""
        # 确保得分非负
        min_score = stock_scores.min()
        if min_score < 0:
            stock_scores = stock_scores - min_score

        total = stock_scores.sum()
        if total < 1e-10:
            return self._equal_weight(stock_scores)

        weights = stock_scores / total

        # 应用最大权重限制
        weights = self._apply_max_weight(weights)

        return weights.to_dict()

    def _optimize_weight(self, stock_scores: pd.Series) -> Dict[str, float]:
        """优化权重 (最大化得分，约束最大权重)"""
        if not SCIPY_AVAILABLE:
            logger.warning("[PortfolioBuilder] scipy not available, fallback to score weight")
            return self._score_weight(stock_scores)

        n = len(stock_scores)
        scores = stock_scores.values

        # 目标函数: 负加权得分
        def neg_score(weights):
            return -np.dot(weights, scores)

        # 约束
        bounds = [(0, self.max_weight)] * n
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

        # 优化
        x0 = np.ones(n) / n
        try:
            result = minimize(
                neg_score,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=cons,
            )
            if result.success:
                weights = result.x
                return {stock: float(w) for stock, w in zip(stock_scores.index, weights)}
        except Exception as e:
            logger.warning(f"[PortfolioBuilder] Optimization failed: {e}")

        return self._score_weight(stock_scores)

    def _industry_neutral_weight(
        self,
        stock_scores: pd.Series,
        industry: pd.Series,
        top_n: int,
    ) -> Dict[str, float]:
        """行业中性加权 (每个行业选Top-N/行业数)"""
        # 合并数据
        df = pd.DataFrame({"score": stock_scores, "industry": industry})
        df = df.dropna().sort_values("score", ascending=False)

        # 每个行业选的股票数
        n_industries = df["industry"].nunique()
        per_industry = max(1, top_n // n_industries)

        # 每个行业选Top
        selected = df.groupby("industry").head(per_industry)

        # 行业内等权
        weights = selected.groupby("industry")["score"].apply(
            lambda x: pd.Series(1.0 / len(x), index=x.index)
        )

        # 行业间等权
        industry_weights = {ind: 1.0 / n_industries for ind in selected["industry"].unique()}
        final_weights = {}
        for ind, ind_weight in industry_weights.items():
            ind_stocks = weights.loc[ind] if ind in weights.index.get_level_values(0) else weights.xs(ind)
            for stock, stock_weight in ind_stocks.items():
                final_weights[stock] = ind_weight * stock_weight

        return final_weights

    def _apply_max_weight(self, weights: pd.Series) -> pd.Series:
        """应用最大权重限制"""
        clipped = weights.clip(upper=self.max_weight)
        total = clipped.sum()
        if total < 1e-10:
            return weights
        return clipped / total

    def compute_portfolio_stats(
        self,
        weights: Dict[str, float],
        returns: pd.DataFrame,
        risk_free_rate: float = 0.03,
    ) -> Dict[str, float]:
        """
        计算组合绩效指标

        Args:
            weights: 组合权重
            returns: 历史收益率矩阵 (index=date, columns=stock)
            risk_free_rate: 无风险利率

        Returns:
            绩效指标字典
        """
        # 对齐数据
        stocks = list(weights.keys())
        common_cols = [s for s in stocks if s in returns.columns]
        if not common_cols:
            return {"error": "no_common_stocks"}

        weights_arr = np.array([weights[s] for s in common_cols])
        returns_matrix = returns[common_cols].values

        # 组合收益
        portfolio_returns = returns_matrix @ weights_arr

        # 绩效指标
        n_periods = len(portfolio_returns)
        if n_periods < 2:
            return {"error": "insufficient_data"}

        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns, ddof=1)

        # 年化 (假设月度数据)
        annualized_return = mean_return * 12
        annualized_vol = std_return * np.sqrt(12)

        # 夏普比率
        sharpe = (annualized_return - risk_free_rate) / (annualized_vol + 1e-10)

        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # Calmar比率
        calmar = annualized_return / (abs(max_drawdown) + 1e-10)

        # 胜率
        win_rate = np.mean(portfolio_returns > 0)

        return {
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "calmar_ratio": float(calmar),
            "win_rate": float(win_rate),
            "n_periods": n_periods,
        }

    def compute_risk_contribution(
        self,
        weights: Dict[str, float],
        cov_matrix: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        计算风险贡献

        Args:
            weights: 组合权重
            cov_matrix: 协方差矩阵

        Returns:
            风险贡献字典
        """
        stocks = list(weights.keys())
        common_stocks = [s for s in stocks if s in cov_matrix.index and s in cov_matrix.columns]

        if not common_stocks:
            return {}

        weights_arr = np.array([weights[s] for s in common_stocks])
        cov_arr = cov_matrix.loc[common_stocks, common_stocks].values

        # 组合方差
        portfolio_var = weights_arr @ cov_arr @ weights_arr

        # 边际风险贡献
        marginal_risk = cov_arr @ weights_arr

        # 风险贡献
        risk_contrib = weights_arr * marginal_risk

        # 归一化
        total_risk = risk_contrib.sum()
        if total_risk < 1e-10:
            return {s: 0.0 for s in common_stocks}

        risk_contrib_pct = risk_contrib / total_risk

        return {stock: float(rc) for stock, rc in zip(common_stocks, risk_contrib_pct)}

    def rebalance_portfolio(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        threshold: float = 0.01,
    ) -> Dict[str, float]:
        """
        计算调仓交易

        Args:
            current_weights: 当前权重
            target_weights: 目标权重
            threshold: 调仓阈值 (权重变化超过此值才交易)

        Returns:
            调仓字典 {stock: trade_weight}
        """
        all_stocks = set(current_weights.keys()) | set(target_weights.keys())
        trades = {}

        for stock in all_stocks:
            current = current_weights.get(stock, 0.0)
            target = target_weights.get(stock, 0.0)
            diff = target - current

            if abs(diff) > threshold:
                trades[stock] = diff

        return trades
