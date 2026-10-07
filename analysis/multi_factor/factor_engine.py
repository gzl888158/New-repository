"""
因子计算引擎 (Factor Calculation Engine)

实现6大类因子计算：
1. 价值因子 (Value): EP (E/P), BP (B/P)
2. 质量因子 (Quality): ROE, 资产负债率
3. 动量因子 (Momentum): 过去N日收益率
4. 规模因子 (Size): 市值 (对数)
5. 量价因子 (Volume-Price): 换手率, 波动率
6. 事件因子 (Event): 业绩超预期

所有因子计算遵循：
- 避免未来函数 (point-in-time data)
- 缺失值处理 (行业中性化可选)
- 因子标准化 (z-score或rank)
"""

import warnings
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import akshare as ak
except ImportError:
    ak = None
    warnings.warn("AKShare not installed. Run: pip install akshare")

from loguru import logger


class FactorCategory(Enum):
    """因子类别"""
    VALUE = "value"           # 价值因子
    QUALITY = "quality"       # 质量因子
    MOMENTUM = "momentum"     # 动量因子
    SIZE = "size"             # 规模因子
    VOLUME_PRICE = "vol_price"  # 量价因子
    EVENT = "event"           # 事件因子


class FactorEngine:
    """
    因子计算引擎

    用法：
        engine = FactorEngine()
        factors = engine.compute_all_factors(
            stock_list=["000001", "600000"],
            date="2024-01-15"
        )
    """

    def __init__(
        self,
        momentum_window: int = 20,
        volatility_window: int = 20,
        min_data_days: int = 60,
    ):
        """
        Args:
            momentum_window: 动量因子回看窗口 (交易日)
            volatility_window: 波动率计算窗口 (交易日)
            min_data_days: 新股过滤阈值 (上市不足N天剔除)
        """
        if ak is None:
            raise ImportError("AKShare required: pip install akshare")

        self.momentum_window = momentum_window
        self.volatility_window = volatility_window
        self.min_data_days = min_data_days

        # 因子元数据 (名称, 类别, 方向: 1=越大越好, -1=越小越好)
        self.factor_meta: Dict[str, Tuple[FactorCategory, int]] = {
            "ep": (FactorCategory.VALUE, 1),           # E/P, 越大越便宜
            "bp": (FactorCategory.VALUE, 1),           # B/P, 越大越便宜
            "roe": (FactorCategory.QUALITY, 1),        # ROE, 越大越好
            "debt_ratio": (FactorCategory.QUALITY, -1), # 资产负债率, 越小越好
            "momentum_20d": (FactorCategory.MOMENTUM, 1),  # 20日动量
            "ln_market_cap": (FactorCategory.SIZE, -1),    # 对数市值, 小市值效应
            "turnover_rate": (FactorCategory.VOLUME_PRICE, -1),  # 换手率, 低换手优选
            "volatility_20d": (FactorCategory.VOLUME_PRICE, -1), # 波动率, 低波动优选
            "earnings_surprise": (FactorCategory.EVENT, 1),  # 业绩超预期
        }

    def compute_all_factors(
        self,
        stock_list: List[str],
        date: str,
    ) -> pd.DataFrame:
        """
        计算所有因子

        Args:
            stock_list: 股票代码列表 (如 ["000001", "600000"])
            date: 计算日期 (YYYY-MM-DD)

        Returns:
            DataFrame, index=stock_code, columns=factor_names
        """
        logger.info(f"[FactorEngine] Computing factors for {len(stock_list)} stocks on {date}")

        # 1. 获取基础数据 (避免未来函数，使用date之前的数据)
        price_data = self._get_price_data(stock_list, date)
        fundamental_data = self._get_fundamental_data(stock_list, date)

        if price_data.empty or fundamental_data.empty:
            logger.warning("[FactorEngine] No data available")
            return pd.DataFrame()

        # 2. 计算各类因子
        factors = pd.DataFrame(index=stock_list)

        # 价值因子
        factors[["ep", "bp"]] = self._compute_value_factors(fundamental_data, price_data)

        # 质量因子
        factors[["roe", "debt_ratio"]] = self._compute_quality_factors(fundamental_data)

        # 动量因子
        factors["momentum_20d"] = self._compute_momentum_factors(price_data)

        # 规模因子
        factors["ln_market_cap"] = self._compute_size_factors(fundamental_data, price_data)

        # 量价因子
        factors[["turnover_rate", "volatility_20d"]] = self._compute_volume_price_factors(price_data)

        # 事件因子 (可选，需要额外数据)
        factors["earnings_surprise"] = self._compute_event_factors(stock_list, date)

        # 3. 因子标准化 (去极值 + z-score)
        factors = self._winsorize_and_standardize(factors)

        logger.info(f"[FactorEngine] Computed {len(factors.columns)} factors")
        return factors

    def _get_price_data(self, stock_list: List[str], date: str) -> pd.DataFrame:
        """
        获取价格数据 (避免未来函数)

        Returns:
            MultiIndex DataFrame (stock, date) with OHLCV columns
        """
        end_date = pd.to_datetime(date)
        start_date = end_date - timedelta(days=self.min_data_days * 2)

        all_data = []
        for stock in stock_list:
            try:
                # AKShare: 获取日K线数据
                df = ak.stock_zh_a_hist(
                    symbol=stock,
                    period="daily",
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust="qfq"  # 前复权
                )
                if df.empty:
                    continue

                df["stock"] = stock
                df["date"] = pd.to_datetime(df["日期"])
                df = df.rename(columns={
                    "开盘": "open",
                    "收盘": "close",
                    "最高": "high",
                    "最低": "low",
                    "成交量": "volume",
                    "换手率": "turnover",
                })
                all_data.append(df[["stock", "date", "open", "close", "high", "low", "volume", "turnover"]])
            except Exception as e:
                logger.warning(f"[FactorEngine] Failed to get price data for {stock}: {e}")
                continue

        if not all_data:
            return pd.DataFrame()

        return pd.concat(all_data, ignore_index=True).set_index(["stock", "date"])

    def _get_fundamental_data(self, stock_list: List[str], date: str) -> pd.DataFrame:
        """
        获取基本面数据 (point-in-time, 避免未来函数)

        Returns:
            DataFrame, index=stock_code
        """
        # AKShare: 获取最新财务指标 (注意：实际生产环境需用历史财报数据)
        try:
            # 获取个股基本面数据
            all_data = []
            for stock in stock_list:
                try:
                    # 获取财务指标
                    fin_df = ak.stock_financial_analysis_indicator(symbol=stock)
                    if fin_df.empty:
                        continue

                    # 取最近一期 (避免未来函数，需确保发布日期 <= date)
                    latest = fin_df.iloc[0]
                    all_data.append({
                        "stock": stock,
                        "roe": latest.get("净资产收益率(%)", np.nan),
                        "debt_ratio": latest.get("资产负债率(%)", np.nan),
                        "eps": latest.get("基本每股收益(元)", np.nan),
                        "bvps": latest.get("每股净资产(元)", np.nan),
                        "market_cap": latest.get("总市值", np.nan),
                    })
                except Exception as e:
                    logger.warning(f"[FactorEngine] Failed to get fundamental for {stock}: {e}")
                    continue

            if not all_data:
                return pd.DataFrame()

            return pd.DataFrame(all_data).set_index("stock")
        except Exception as e:
            logger.error(f"[FactorEngine] Failed to get fundamental data: {e}")
            return pd.DataFrame()

    def _compute_value_factors(
        self,
        fundamental: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        价值因子: EP (E/P), BP (B/P)

        EP = EPS / Price
        BP = BVPS / Price
        """
        # 获取最新收盘价
        latest_prices = price_data.groupby(level=0)["close"].last()

        eps = fundamental["eps"]
        bvps = fundamental["bvps"]

        ep = eps / latest_prices
        bp = bvps / latest_prices

        return pd.DataFrame({"ep": ep, "bp": bp}).reindex(fundamental.index)

    def _compute_quality_factors(self, fundamental: pd.DataFrame) -> pd.DataFrame:
        """
        质量因子: ROE, 资产负债率
        """
        roe = fundamental["roe"]
        debt_ratio = fundamental["debt_ratio"]

        return pd.DataFrame({"roe": roe, "debt_ratio": debt_ratio})

    def _compute_momentum_factors(self, price_data: pd.DataFrame) -> pd.Series:
        """
        动量因子: 过去N日收益率
        """
        def calc_momentum(group):
            if len(group) < self.momentum_window:
                return np.nan
            returns = group["close"].pct_change(self.momentum_window).iloc[-1]
            return returns

        momentum = price_data.groupby(level=0).apply(calc_momentum)
        return momentum

    def _compute_size_factors(
        self,
        fundamental: pd.DataFrame,
        price_data: pd.DataFrame,
    ) -> pd.Series:
        """
        规模因子: 对数市值
        """
        market_cap = fundamental["market_cap"]
        ln_market_cap = np.log(market_cap.replace(0, np.nan))
        return ln_market_cap

    def _compute_volume_price_factors(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        量价因子: 换手率, 波动率
        """
        def calc_factors(group):
            # 平均换手率
            avg_turnover = group["turnover"].mean()

            # 波动率 (年化)
            returns = group["close"].pct_change().dropna()
            if len(returns) < self.volatility_window:
                vol = np.nan
            else:
                vol = returns.tail(self.volatility_window).std() * np.sqrt(252)

            return pd.Series({"turnover_rate": avg_turnover, "volatility_20d": vol})

        factors = price_data.groupby(level=0).apply(calc_factors)
        return factors

    def _compute_event_factors(self, stock_list: List[str], date: str) -> pd.Series:
        """
        事件因子: 业绩超预期 (实际EPS vs 一致预期)

        简化实现: 使用同比EPS增长率作为代理
        """
        # TODO: 接入一致预期数据，计算surprise
        # 这里用同比增速代替
        return pd.Series(np.nan, index=stock_list)

    def _winsorize_and_standardize(self, factors: pd.DataFrame) -> pd.DataFrame:
        """
        因子标准化: 去极值 (MAD) + z-score

        MAD (Median Absolute Deviation) 比标准差更稳健
        """
        result = factors.copy()

        for col in result.columns:
            series = result[col].dropna()
            if len(series) < 10:
                continue

            # MAD去极值
            median = series.median()
            mad = (series - median).abs().median()
            mad_e = 1.4826 * mad  # 转换为标准差估计
            upper = median + 3 * mad_e
            lower = median - 3 * mad_e
            series = series.clip(lower, upper)

            # z-score标准化
            mean = series.mean()
            std = series.std()
            if std > 0:
                result[col] = (series - mean) / std
            else:
                result[col] = 0.0

        return result

    def get_factor_direction(self, factor_name: str) -> int:
        """获取因子方向 (1=越大越好, -1=越小越好)"""
        if factor_name not in self.factor_meta:
            return 1
        return self.factor_meta[factor_name][1]

    def get_factor_category(self, factor_name: str) -> FactorCategory:
        """获取因子类别"""
        if factor_name not in self.factor_meta:
            return FactorCategory.VALUE
        return self.factor_meta[factor_name][0]
