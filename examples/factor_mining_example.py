"""
因子挖掘与组合构建完整示例

演示：
1. 因子计算与评估
2. 因子筛选 (独立有效因子)
3. 因子权重优化
4. 组合构建
5. 绩效评估
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from analysis.multi_factor import (
    FactorEngine,
    FactorAnalyzer,
    FactorEvaluator,
    FactorOptimizer,
    StockSelector,
    PortfolioBuilder,
)


def generate_mock_data(n_stocks: int = 100, n_days: int = 252):
    """生成模拟数据用于演示"""
    np.random.seed(42)
    dates = pd.date_range(end="2024-01-15", periods=n_days, freq="B")
    stocks = [f"{str(i).zfill(6)}" for i in range(1, n_stocks + 1)]

    # 模拟价格数据
    returns = np.random.randn(n_days, n_stocks) * 0.02
    prices = 100 * np.exp(np.cumsum(returns, axis=0))
    price_df = pd.DataFrame(prices, index=dates, columns=stocks)

    # 模拟财务数据
    fundamentals = pd.DataFrame(
        {
            "code": stocks,
            "pe_ratio": np.random.uniform(5, 50, n_stocks),
            "pb_ratio": np.random.uniform(0.5, 5, n_stocks),
            "roe": np.random.uniform(0.05, 0.3, n_stocks),
            "debt_ratio": np.random.uniform(0.2, 0.8, n_stocks),
            "total_market_cap": np.random.uniform(1e9, 1e12, n_stocks),
            "turnover_rate": np.random.uniform(0.01, 0.1, n_stocks),
        }
    )

    # 模拟行业分类
    industries = ["银行", "地产", "科技", "消费", "医药", "制造"]
    industry_map = pd.Series(
        np.random.choice(industries, n_stocks), index=stocks, name="industry"
    )

    return price_df, fundamentals, industry_map


def example_1_factor_evaluation():
    """示例1：因子评估"""
    print("\n" + "=" * 60)
    print("示例1：因子评估")
    print("=" * 60)

    # 生成模拟因子数据 (实际应从FactorEngine获取)
    np.random.seed(42)
    n_stocks = 100
    stocks = [f"{str(i).zfill(6)}" for i in range(1, n_stocks + 1)]

    # 模拟因子矩阵
    factors = pd.DataFrame(
        {
            "ep": np.random.uniform(0.02, 0.15, n_stocks),
            "bp": np.random.uniform(0.3, 3.0, n_stocks),
            "roe": np.random.uniform(0.05, 0.3, n_stocks),
            "debt_ratio": np.random.uniform(0.2, 0.8, n_stocks),
            "momentum_20d": np.random.uniform(-0.1, 0.2, n_stocks),
            "ln_market_cap": np.random.uniform(20, 28, n_stocks),
        },
        index=stocks,
    )

    # 模拟未来收益 (与因子有一定相关性)
    forward_returns = pd.Series(
        0.05 * factors["ep"] + 0.03 * factors["roe"] - 0.02 * factors["debt_ratio"] + np.random.randn(n_stocks) * 0.03,
        index=stocks,
    )

    # 评估每个因子
    evaluator = FactorEvaluator()
    evaluation_results = []

    for factor_name in factors.columns:
        result = evaluator.evaluate_factor(
            factors[factor_name],
            forward_returns,
            factor_name=factor_name,
            date="2024-01-15",
        )
        evaluation_results.append(result)

    # 生成评估报告
    report_df = evaluator.generate_factor_report(evaluation_results)
    print("\n因子评估报告:")
    print(report_df.to_string(index=False))

    return factors, forward_returns, report_df


def example_2_factor_screening(factors: pd.DataFrame, forward_returns: pd.Series):
    """示例2：因子筛选"""
    print("\n" + "=" * 60)
    print("示例2：因子筛选 (独立有效因子)")
    print("=" * 60)

    optimizer = FactorOptimizer(corr_threshold=0.7)

    # 筛选独立有效因子
    selected_factors = optimizer.screen_independent_factors(
        factors, forward_returns, min_ic=0.03, min_ic_ir=0.5
    )

    print(f"\n原始因子数: {len(factors.columns)}")
    print(f"筛选后因子数: {len(selected_factors)}")
    print(f"筛选后的因子: {selected_factors}")

    # 冗余报告
    redundancy_report = optimizer.compute_factor_redundancy_report(factors)
    print(f"\n平均因子相关性: {redundancy_report['average_correlation']:.3f}")
    print(f"高相关性因子对数: {redundancy_report['n_high_corr_pairs']}")

    if redundancy_report["high_correlation_pairs"]:
        print("\n高相关性因子对:")
        for pair in redundancy_report["high_correlation_pairs"][:5]:
            print(
                f"  {pair['factor_1']} <-> {pair['factor_2']}: {pair['correlation']:.3f}"
            )

    return selected_factors


def example_3_factor_optimization(
    factors: pd.DataFrame,
    forward_returns: pd.Series,
    selected_factors: list,
):
    """示例3：因子权重优化"""
    print("\n" + "=" * 60)
    print("示例3：因子权重优化")
    print("=" * 60)

    optimizer = FactorOptimizer()
    selected_factors_df = factors[selected_factors]

    # 不同优化方法
    methods = ["equal", "ic", "ic_ir", "risk_parity"]
    weight_results = {}

    for method in methods:
        weights = optimizer.optimize_weights(
            selected_factors_df, forward_returns, method=method
        )
        weight_results[method] = weights
        print(f"\n{method} 权重:")
        for factor, weight in sorted(weights.items(), key=lambda x: -x[1]):
            print(f"  {factor}: {weight:.3f}")

    return weight_results


def example_4_portfolio_construction(
    factors: pd.DataFrame,
    selected_factors: list,
    weights: dict,
    industry_map: pd.Series,
):
    """示例4：组合构建"""
    print("\n" + "=" * 60)
    print("示例4：组合构建")
    print("=" * 60)

    # 计算综合得分
    selected_factors_df = factors[selected_factors]
    composite_score = pd.Series(0.0, index=factors.index)

    for factor_name, weight in weights.items():
        if factor_name in selected_factors_df.columns:
            factor_vals = selected_factors_df[factor_name]
            # 标准化
            factor_std = (factor_vals - factor_vals.mean()) / (factor_vals.std() + 1e-10)
            composite_score += weight * factor_std

    # 构建组合
    builder = PortfolioBuilder(max_weight=0.05)

    methods = ["top_n", "score_weight", "industry_neutral"]
    portfolios = {}

    for method in methods:
        if method == "industry_neutral":
            portfolio = builder.build_portfolio(
                composite_score,
                method=method,
                top_n=30,
                industry=industry_map,
            )
        else:
            portfolio = builder.build_portfolio(
                composite_score,
                method=method,
                top_n=30,
            )

        portfolios[method] = portfolio
        print(f"\n{method} 组合:")
        print(f"  股票数: {len(portfolio)}")
        print(f"  前5大持仓:")
        sorted_portfolio = sorted(portfolio.items(), key=lambda x: -x[1])[:5]
        for stock, weight in sorted_portfolio:
            print(f"    {stock}: {weight:.3f}")

    return portfolios, composite_score


def example_5_performance_evaluation(
    portfolios: dict,
    price_df: pd.DataFrame,
):
    """示例5：绩效评估"""
    print("\n" + "=" * 60)
    print("示例5：绩效评估")
    print("=" * 60)

    builder = PortfolioBuilder()

    # 计算历史收益
    returns = price_df.pct_change().dropna()

    for method, portfolio in portfolios.items():
        stats = builder.compute_portfolio_stats(portfolio, returns.tail(60))
        print(f"\n{method} 组合绩效:")
        print(f"  年化收益: {stats['annualized_return']:.2%}")
        print(f"  年化波动: {stats['annualized_volatility']:.2%}")
        print(f"  夏普比率: {stats['sharpe_ratio']:.3f}")
        print(f"  最大回撤: {stats['max_drawdown']:.2%}")
        print(f"  胜率: {stats['win_rate']:.2%}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("因子挖掘与组合构建完整示例")
    print("=" * 60)

    # 示例1：因子评估
    factors, forward_returns, eval_report = example_1_factor_evaluation()

    # 示例2：因子筛选
    selected_factors = example_2_factor_screening(factors, forward_returns)

    if not selected_factors:
        print("\n无有效因子，退出")
        return

    # 示例3：因子权重优化
    weight_results = example_3_factor_optimization(
        factors, forward_returns, selected_factors
    )

    # 示例4：组合构建
    price_df, fundamentals, industry_map = generate_mock_data()
    portfolios, composite_score = example_4_portfolio_construction(
        factors,
        selected_factors,
        weight_results["ic"],  # 使用IC加权
        industry_map,
    )

    # 示例5：绩效评估
    example_5_performance_evaluation(portfolios, price_df)

    print("\n" + "=" * 60)
    print("示例完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
