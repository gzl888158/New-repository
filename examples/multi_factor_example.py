"""
多因子选股策略示例 (Multi-Factor Stock Selection Example)

完整示例：
1. 因子计算与检验
2. 因子打分选股
3. 回测验证
4. 偏差防范

运行：python examples/multi_factor_example.py
"""

import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from loguru import logger

from analysis.multi_factor import FactorEngine, FactorAnalyzer, StockSelector
from backtest.stock_backtest_engine import StockBacktestEngine
from backtest.bias_prevention import BiasPrevention, WalkForwardAnalyzer


def example_factor_computation():
    """示例1: 因子计算"""
    logger.info("=" * 60)
    logger.info("示例1: 因子计算")
    logger.info("=" * 60)

    # 初始化因子引擎
    engine = FactorEngine(
        momentum_window=20,
        volatility_window=20,
        min_data_days=60,
    )

    # 股票池 (示例)
    stock_list = ["000001", "000002", "600000", "600036"]

    # 计算因子
    date = "2024-01-15"
    factors = engine.compute_all_factors(stock_list, date)

    logger.info(f"计算了 {len(factors.columns)} 个因子")
    logger.info(f"因子列表: {list(factors.columns)}")
    logger.info(f"\n因子矩阵:\n{factors}")

    return factors


def example_factor_analysis(factors: pd.DataFrame):
    """示例2: 因子分析"""
    logger.info("\n" + "=" * 60)
    logger.info("示例2: 因子分析")
    logger.info("=" * 60)

    analyzer = FactorAnalyzer()

    # 模拟未来收益 (实际应从价格数据计算)
    forward_returns = pd.Series(
        np.random.randn(len(factors)) * 0.05,
        index=factors.index,
    )

    # IC分析
    ic_report = analyzer.compute_ic_report(factors, forward_returns)
    logger.info(f"\nIC分析报告:\n{ic_report}")

    # 共线性检测
    collinearity = analyzer.check_multicollinearity(factors, threshold=0.7)
    logger.info(f"\n高相关性因子对: {len(collinearity['high_correlation_pairs'])}")
    if collinearity["high_correlation_pairs"]:
        for pair in collinearity["high_correlation_pairs"][:3]:
            logger.info(f"  {pair['factor_1']} <-> {pair['factor_2']}: {pair['correlation']:.3f}")

    return ic_report


def example_stock_selection(factors: pd.DataFrame):
    """示例3: 选股"""
    logger.info("\n" + "=" * 60)
    logger.info("示例3: 因子打分选股")
    logger.info("=" * 60)

    engine = FactorEngine()
    selector = StockSelector(engine, weighting_method="equal")

    # 选股
    selected = selector.select_stocks(
        factors,
        method="top_n",
        top_n=2,
    )

    logger.info(f"选中股票: {selected}")

    # 综合得分
    composite = selector.compute_composite_score(factors)
    logger.info(f"\n综合得分:\n{composite.sort_values(ascending=False)}")

    return selected


def example_backtest():
    """示例4: 回测"""
    logger.info("\n" + "=" * 60)
    logger.info("示例4: 策略回测")
    logger.info("=" * 60)

    # 初始化回测引擎
    engine = StockBacktestEngine(
        initial_capital=1_000_000,
        commission_rate=0.001,
        slippage_rate=0.001,
    )

    # 定义选股函数
    def stock_selection_func(date):
        # 简化示例：随机选股
        all_stocks = ["000001", "000002", "600000", "600036"]
        return np.random.choice(all_stocks, size=2, replace=False).tolist()

    # 运行回测
    result = engine.run_backtest(
        stock_selection_func=stock_selection_func,
        start_date="2023-01-01",
        end_date="2024-01-01",
        rebalance_freq="monthly",
    )

    if result:
        perf = result["performance"]
        logger.info(f"\n回测绩效:")
        logger.info(f"  年化收益: {perf.get('annual_return', 0):.2%}")
        logger.info(f"  夏普比率: {perf.get('sharpe_ratio', 0):.2f}")
        logger.info(f"  最大回撤: {perf.get('max_drawdown', 0):.2%}")
        logger.info(f"  超额收益: {perf.get('excess_return', 0):.2%}")

    return result


def example_bias_prevention():
    """示例5: 偏差防范"""
    logger.info("\n" + "=" * 60)
    logger.info("示例5: 偏差防范")
    logger.info("=" * 60)

    bp = BiasPrevention()

    # 1. 未来函数检测
    data_timestamps = pd.to_datetime(["2024-01-15", "2024-01-16", "2024-01-17"])
    trading_dates = pd.to_datetime(["2024-01-14", "2024-01-16", "2024-01-16"])

    lookahead_report = bp.check_lookahead_bias(data_timestamps, trading_dates)
    logger.info(f"\n未来函数检测: {lookahead_report['violations']} 个违规")

    # 2. 多重检验校正
    p_values = [0.01, 0.03, 0.04, 0.5]
    bonferroni = bp.bonferroni_correction(p_values, alpha=0.05)
    logger.info(f"\nBonferroni校正: {bonferroni['significant_after_correction']}/{len(p_values)} 显著")

    fdr = bp.fdr_correction(p_values, q=0.05)
    logger.info(f"FDR校正: {fdr['significant_count']}/{len(p_values)} 显著")

    return lookahead_report


def example_walk_forward():
    """示例6: Walk-Forward分析"""
    logger.info("\n" + "=" * 60)
    logger.info("示例6: Walk-Forward样本外测试")
    logger.info("=" * 60)

    # 生成模拟数据
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="B")
    data = pd.DataFrame({
        "feature1": np.random.randn(len(dates)),
        "feature2": np.random.randn(len(dates)),
        "return": np.random.randn(len(dates)) * 0.02,
    }, index=dates)

    # Walk-Forward分析器
    wfa = WalkForwardAnalyzer(
        train_window=252 * 2,  # 2年训练
        test_window=252,        # 1年测试
        step_size=252,          # 步进1年
    )

    # 定义策略和评估函数
    def strategy_func(train_data):
        # 简化：返回均值
        return {"mean_return": train_data["return"].mean()}

    def metric_func(model, test_data):
        # 简化：计算测试集收益
        return {"return": model["mean_return"] * len(test_data)}

    # 运行Walk-Forward
    wf_results = wfa.run_walk_forward(data, strategy_func, metric_func)
    logger.info(f"\nWalk-Forward结果:\n{wf_results}")

    # 稳健性统计
    robustness = wfa.compute_robustness_stats(wf_results)
    logger.info(f"\n稳健性统计:")
    logger.info(f"  折数: {robustness.get('n_folds', 0)}")
    logger.info(f"  平均收益: {robustness.get('mean_return', 0):.4f}")
    logger.info(f"  正收益比例: {robustness.get('positive_rate', 0):.2%}")

    return wf_results


def main():
    """主函数"""
    logger.info("多因子选股策略完整示例")
    logger.info(f"运行时间: {datetime.now()}")

    try:
        # 1. 因子计算
        factors = example_factor_computation()

        # 2. 因子分析
        ic_report = example_factor_analysis(factors)

        # 3. 选股
        selected = example_stock_selection(factors)

        # 4. 回测
        backtest_result = example_backtest()

        # 5. 偏差防范
        bias_report = example_bias_prevention()

        # 6. Walk-Forward
        wf_results = example_walk_forward()

        logger.info("\n" + "=" * 60)
        logger.info("示例运行完成")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
