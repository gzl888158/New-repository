"""
股票回测引擎 (Stock Backtesting Engine)

实现A股多因子策略回测：
1. 股票池过滤 (剔除ST、新股、停牌)
2. 调仓调度 (月度/季度)
3. 交易成本 (手续费、滑点)
4. 绩效评估 (净值、年化、夏普、最大回撤)
5. 基准对比 (沪深300/中证500)
"""

import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None
    warnings.warn("AKShare not installed. Run: pip install akshare")

from loguru import logger


class StockBacktestEngine:
    """
    股票回测引擎

    用法：
        engine = StockBacktestEngine()
        result = engine.run_backtest(
            stock_selection_func=select_func,
            start_date="2020-01-01",
            end_date="2024-01-01",
            rebalance_freq="monthly",
        )
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        commission_rate: float = 0.001,  # 0.1% (单边)
        slippage_rate: float = 0.001,    # 0.1%
        stamp_tax_rate: float = 0.001,   # 0.1% (仅卖出)
        min_commission: float = 5.0,     # 最低佣金5元
    ):
        """
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金率 (双边)
            slippage_rate: 滑点率
            stamp_tax_rate: 印花税率 (仅卖出)
            min_commission: 最低佣金
        """
        if ak is None:
            raise ImportError("AKShare required: pip install akshare")

        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.stamp_tax_rate = stamp_tax_rate
        self.min_commission = min_commission

    def run_backtest(
        self,
        stock_selection_func: callable,
        start_date: str,
        end_date: str,
        rebalance_freq: str = "monthly",
        benchmark: str = "000300",  # 沪深300
    ) -> Dict[str, Any]:
        """
        运行回测

        Args:
            stock_selection_func: 选股函数 (date) -> List[str]
            start_date: 回测开始日期
            end_date: 回测结束日期
            rebalance_freq: 调仓频率 ("monthly", "quarterly")
            benchmark: 基准代码

        Returns:
            Dict with backtest results
        """
        logger.info(f"[Backtest] Running from {start_date} to {end_date}")

        # 1. 生成调仓日期
        rebalance_dates = self._generate_rebalance_dates(start_date, end_date, rebalance_freq)
        logger.info(f"[Backtest] {len(rebalance_dates)} rebalance dates")

        # 2. 获取价格数据
        price_data = self._get_price_data(start_date, end_date)
        if price_data.empty:
            logger.error("[Backtest] No price data")
            return {}

        # 3. 获取基准数据
        benchmark_data = self._get_benchmark_data(benchmark, start_date, end_date)

        # 4. 模拟交易
        portfolio = self._simulate_trading(
            stock_selection_func,
            rebalance_dates,
            price_data,
        )

        # 5. 计算绩效
        performance = self._compute_performance(portfolio, benchmark_data)

        return {
            "portfolio": portfolio,
            "performance": performance,
            "rebalance_dates": rebalance_dates,
        }

    def _generate_rebalance_dates(
        self,
        start_date: str,
        end_date: str,
        freq: str,
    ) -> List[str]:
        """生成调仓日期"""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        dates = []
        current = start

        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))

            if freq == "monthly":
                # 下个月同一天
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
            elif freq == "quarterly":
                # 下季度同一天
                if current.month >= 10:
                    current = current.replace(year=current.year + 1, month=current.month - 9)
                else:
                    current = current.replace(month=current.month + 3)
            else:
                raise ValueError(f"Unknown freq: {freq}")

        return dates

    def _get_price_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取所有股票价格数据"""
        # TODO: 实际实现中，应获取全市场股票数据
        # 这里简化处理，假设stock_selection_func返回的股票列表已知
        logger.warning("[Backtest] Price data loading not implemented. Use actual data source.")
        return pd.DataFrame()

    def _get_benchmark_data(
        self,
        benchmark: str,
        start_date: str,
        end_date: str,
    ) -> pd.Series:
        """获取基准指数数据"""
        try:
            df = ak.stock_zh_index_daily(symbol=f"sh{benchmark}")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.loc[start_date:end_date]
            return df["close"]
        except Exception as e:
            logger.warning(f"[Backtest] Failed to get benchmark data: {e}")
            return pd.Series(dtype=float)

    def _simulate_trading(
        self,
        stock_selection_func: callable,
        rebalance_dates: List[str],
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        模拟交易

        Returns:
            DataFrame with daily portfolio value
        """
        cash = self.initial_capital
        positions = {}  # {stock: shares}
        portfolio_values = []

        for i, date in enumerate(rebalance_dates):
            logger.info(f"[Backtest] Rebalance on {date}")

            # 1. 选股
            selected_stocks = stock_selection_func(date)
            if not selected_stocks:
                logger.warning(f"[Backtest] No stocks selected on {date}")
                continue

            # 2. 计算当前持仓市值
            portfolio_value = cash
            for stock, shares in positions.items():
                # TODO: 获取该股票的当日价格
                price = self._get_price_on_date(stock, date, price_data)
                if price > 0:
                    portfolio_value += shares * price

            # 3. 卖出不在新选股列表中的股票
            stocks_to_sell = [s for s in positions if s not in selected_stocks]
            for stock in stocks_to_sell:
                shares = positions[stock]
                price = self._get_price_on_date(stock, date, price_data)
                if price > 0:
                    # 计算卖出成本
                    proceeds = shares * price
                    cost = self._compute_sell_cost(proceeds)
                    cash += proceeds - cost
                    del positions[stock]

            # 4. 等权买入新选中的股票
            stocks_to_buy = [s for s in selected_stocks if s not in positions]
            if stocks_to_buy:
                # 计算每只股票的目标金额
                target_value_per_stock = portfolio_value / len(selected_stocks)

                for stock in stocks_to_buy:
                    price = self._get_price_on_date(stock, date, price_data)
                    if price > 0:
                        # 计算可买股数 (A股100股整数倍)
                        shares = int(target_value_per_stock / price / 100) * 100
                        if shares > 0:
                            cost = self._compute_buy_cost(shares * price)
                            cash -= shares * price + cost
                            positions[stock] = positions.get(stock, 0) + shares

            # 5. 记录当日组合价值
            total_value = cash
            for stock, shares in positions.items():
                price = self._get_price_on_date(stock, date, price_data)
                total_value += shares * price

            portfolio_values.append({
                "date": date,
                "cash": cash,
                "positions_value": total_value - cash,
                "total_value": total_value,
            })

        return pd.DataFrame(portfolio_values).set_index("date")

    def _get_price_on_date(
        self,
        stock: str,
        date: str,
        price_data: pd.DataFrame,
    ) -> float:
        """获取股票在指定日期的价格"""
        # TODO: 实际实现
        return 0.0

    def _compute_buy_cost(self, amount: float) -> float:
        """计算买入成本 (佣金 + 滑点)"""
        commission = max(amount * self.commission_rate, self.min_commission)
        slippage = amount * self.slippage_rate
        return commission + slippage

    def _compute_sell_cost(self, amount: float) -> float:
        """计算卖出成本 (佣金 + 滑点 + 印花税)"""
        commission = max(amount * self.commission_rate, self.min_commission)
        slippage = amount * self.slippage_rate
        stamp_tax = amount * self.stamp_tax_rate
        return commission + slippage + stamp_tax

    def _compute_performance(
        self,
        portfolio: pd.DataFrame,
        benchmark: pd.Series,
    ) -> Dict[str, Any]:
        """计算绩效指标"""
        if portfolio.empty:
            return {}

        # 组合净值
        nav = portfolio["total_value"] / self.initial_capital

        # 日收益率
        returns = nav.pct_change().dropna()

        # 年化收益率
        days = (nav.index[-1] - nav.index[0]).days
        annual_return = (nav.iloc[-1] / nav.iloc[0]) ** (365.0 / days) - 1 if days > 0 else 0.0

        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)

        # 夏普比率 (假设无风险利率3%)
        risk_free_rate = 0.03
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility > 0 else 0.0

        # 最大回撤
        cummax = nav.cummax()
        drawdown = (nav - cummax) / cummax
        max_drawdown = drawdown.min()

        # Calmar比率
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # 基准对比
        if not benchmark.empty:
            benchmark_nav = benchmark / benchmark.iloc[0]
            benchmark_return = (benchmark_nav.iloc[-1] / benchmark_nav.iloc[0]) ** (365.0 / days) - 1
            excess_return = annual_return - benchmark_return
        else:
            benchmark_return = 0.0
            excess_return = 0.0

        return {
            "initial_capital": self.initial_capital,
            "final_value": nav.iloc[-1] * self.initial_capital,
            "total_return": nav.iloc[-1] / nav.iloc[0] - 1,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar,
            "benchmark_return": benchmark_return,
            "excess_return": excess_return,
            "nav": nav,
            "returns": returns,
        }

    def filter_stock_pool(
        self,
        stock_list: List[str],
        date: str,
    ) -> List[str]:
        """
        过滤股票池

        剔除：
        - ST/*ST股票
        - 上市不足60天的新股
        - 停牌股票
        """
        filtered = []
        for stock in stock_list:
            try:
                # 1. 检查ST
                if self._is_st_stock(stock, date):
                    continue

                # 2. 检查上市日期
                if self._is_new_stock(stock, date, min_days=60):
                    continue

                # 3. 检查停牌
                if self._is_suspended(stock, date):
                    continue

                filtered.append(stock)
            except Exception as e:
                logger.warning(f"[Backtest] Failed to filter {stock}: {e}")
                continue

        logger.info(f"[Backtest] Filtered {len(stock_list)} -> {len(filtered)} stocks")
        return filtered

    def _is_st_stock(self, stock: str, date: str) -> bool:
        """检查是否为ST股票"""
        # TODO: 实际实现
        return False

    def _is_new_stock(self, stock: str, date: str, min_days: int = 60) -> bool:
        """检查是否为新股"""
        # TODO: 实际实现
        return False

    def _is_suspended(self, stock: str, date: str) -> bool:
        """检查是否停牌"""
        # TODO: 实际实现
        return False
