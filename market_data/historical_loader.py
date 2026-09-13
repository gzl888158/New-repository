"""
历史行情离线拉取模块
====================
核心定位：批量拉取历史K线、深度、成交、资金费率、基差等数据，
支持断点续传、增量更新、数据完整性校验。

特性：
- 支持多周期K线拉取（1m/5m/15m/30m/1h/4h/1d）
- 资金费率、基差数据定时拉取
- 断点续传：记录上次拉取时间点，避免重复请求
- 数据完整性校验：检查时间连续性、价格合理性
- 支持并发拉取多个交易对，提升效率
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


class DataRange:
    """数据时间范围"""

    def __init__(self, start_ts: int, end_ts: int):
        self.start_ts = start_ts
        self.end_ts = end_ts

    def to_dict(self) -> Dict[str, int]:
        return {"start_ts": self.start_ts, "end_ts": self.end_ts}

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> "DataRange":
        return cls(data.get("start_ts", 0), data.get("end_ts", 0))


class HistoricalLoader:
    """历史行情拉取器"""

    def __init__(self, okx_client=None, config: Dict[str, Any] = None):
        self.config = config or {}
        self._okx_client = okx_client
        self._lock = threading.RLock()

        # 拉取状态：{symbol: {timeframe: last_ts}}
        self._fetch_state: Dict[str, Dict[str, int]] = defaultdict(dict)

        # 并发控制
        self._semaphore = asyncio.Semaphore(config.get("max_concurrent_fetches", 5))

        # 拉取间隔（秒），避免触发API限流
        self._fetch_interval = config.get("fetch_interval", 0.5)

        # 最大单次拉取数量
        self._max_limit = config.get("max_limit", 100)

        # 数据完整性检查
        self._validate_enabled = config.get("validate_data", True)

        logger.info("HistoricalLoader initialized")

    def set_okx_client(self, okx_client):
        """设置 OKX 客户端"""
        self._okx_client = okx_client

    async def fetch_klines(self, symbol: str, timeframe: str = "1m",
                          start_ts: Optional[int] = None, end_ts: Optional[int] = None,
                          limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        拉取 K 线数据

        Args:
            symbol: 交易对
            timeframe: 时间周期（1m/5m/15m/30m/1h/4h/1d）
            start_ts: 开始时间戳（毫秒），None 表示从上次位置继续
            end_ts: 结束时间戳（毫秒），None 表示拉到最新
            limit: 单次拉取数量

        Returns:
            K线数据列表，按时间升序排列
        """
        if not self._okx_client:
            logger.error("OKX client not set")
            return None

        async with self._semaphore:
            try:
                # 确定起始位置
                if start_ts is None:
                    with self._lock:
                        start_ts = self._fetch_state.get(symbol, {}).get(timeframe, 0)

                klines_raw = await asyncio.to_thread(
                    self._okx_client.get_klines,
                    symbol, timeframe, min(limit, self._max_limit), start_ts
                )

                if not klines_raw:
                    return None

                # 标准化数据
                klines = self._normalize_klines(klines_raw, symbol, timeframe)

                # 完整性校验
                if self._validate_enabled:
                    klines = self._validate_klines(klines, symbol, timeframe)

                # 更新拉取状态
                if klines:
                    last_ts = klines[-1]["timestamp"]
                    with self._lock:
                        self._fetch_state[symbol][timeframe] = last_ts

                await asyncio.sleep(self._fetch_interval)
                return klines

            except Exception as e:
                logger.error(f"Failed to fetch klines for {symbol} {timeframe}: {e}")
                return None

    async def fetch_historical_klines_full(self, symbol: str, timeframe: str = "1m",
                                          days: int = 30) -> Optional[List[Dict[str, Any]]]:
        """
        拉取完整历史K线（自动分批）

        Args:
            symbol: 交易对
            timeframe: 时间周期
            days: 拉取天数

        Returns:
            完整的K线数据列表
        """
        end_ts = int(time.time() * 1000)
        start_ts = end_ts - days * 24 * 60 * 60 * 1000

        all_klines = []
        current_start = start_ts

        while current_start < end_ts:
            klines = await self.fetch_klines(symbol, timeframe, current_start, end_ts)
            if not klines:
                break

            all_klines.extend(klines)

            if len(klines) < self._max_limit:
                break

            current_start = klines[-1]["timestamp"] + self._timeframe_to_ms(timeframe)

        return all_klines if all_klines else None

    async def fetch_multiple_symbols_klines(self, symbols: List[str],
                                           timeframe: str = "1m",
                                           days: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """
        并发拉取多个交易对的K线

        Args:
            symbols: 交易对列表
            timeframe: 时间周期
            days: 拉取天数

        Returns:
            {symbol: klines_list}
        """
        tasks = [
            self.fetch_historical_klines_full(symbol, timeframe, days)
            for symbol in symbols
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug(f"Error fetching {symbol} klines: {result}")
                continue
            if result is not None:
                result_dict[symbol] = result

        return result_dict

    async def fetch_funding_rate(self, symbol: str, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        拉取资金费率历史

        Args:
            symbol: 交易对（合约）
            limit: 拉取数量

        Returns:
            资金费率列表
        """
        if not self._okx_client:
            return None

        async with self._semaphore:
            try:
                # P0: 使用正确的历史费率端点，而非当前费率端点
                funding_raw = await asyncio.to_thread(
                    self._okx_client._make_request,
                    "GET", f"/api/v5/public/funding-rate-history?instId={symbol}&limit={limit}"
                )

                if not funding_raw:
                    return None

                return self._normalize_funding_rate(funding_raw, symbol)

            except Exception as e:
                logger.error(f"Failed to fetch funding rate for {symbol}: {e}")
                return None

    async def fetch_basis(self, symbol: str, timeframe: str = "1m",
                         limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        拉取基差数据（期货价格 - 现货价格）

        Args:
            symbol: 交易对
            timeframe: 时间周期
            limit: 拉取数量

        Returns:
            基差数据列表
        """
        if not self._okx_client:
            return None

        async with self._semaphore:
            try:
                # 获取期货K线
                futures_klines = await asyncio.to_thread(
                    self._okx_client.get_klines,
                    symbol, timeframe, limit
                )

                # 获取现货K线（去掉-SWAP后缀）
                spot_symbol = symbol.replace("-SWAP", "")
                spot_klines = await asyncio.to_thread(
                    self._okx_client.get_klines,
                    spot_symbol, timeframe, limit
                )

                if not futures_klines or not spot_klines:
                    return None

                return self._calc_basis(futures_klines, spot_klines, symbol, timeframe)

            except Exception as e:
                logger.error(f"Failed to fetch basis for {symbol}: {e}")
                return None

    async def fetch_recent_trades(self, symbol: str, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """
        拉取最近成交记录

        Args:
            symbol: 交易对
            limit: 拉取数量

        Returns:
            成交记录列表
        """
        if not self._okx_client:
            return None

        async with self._semaphore:
            try:
                trades_raw = await asyncio.to_thread(
                    self._okx_client.get_trades,
                    symbol, limit
                )

                if not trades_raw:
                    return None

                return self._normalize_trades(trades_raw, symbol)

            except Exception as e:
                logger.error(f"Failed to fetch trades for {symbol}: {e}")
                return None

    def _timeframe_to_bar(self, timeframe: str) -> str:
        """时间周期转 OKX bar 参数"""
        mapping = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }
        return mapping.get(timeframe, "1m")

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """时间周期转毫秒"""
        mapping = {
            "1m": 60 * 1000,
            "5m": 5 * 60 * 1000,
            "15m": 15 * 60 * 1000,
            "30m": 30 * 60 * 1000,
            "1h": 60 * 60 * 1000,
            "4h": 4 * 60 * 60 * 1000,
            "1d": 24 * 60 * 60 * 1000,
        }
        return mapping.get(timeframe, 60 * 1000)

    def _normalize_klines(self, klines_raw: List[Any], symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """标准化K线数据"""
        result = []
        for k in klines_raw:
            if isinstance(k, (list, tuple)) and len(k) >= 6:
                result.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": int(k[0]),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                })
        # 按时间升序排列
        return sorted(result, key=lambda x: x["timestamp"])

    def _normalize_funding_rate(self, funding_raw: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
        """标准化资金费率数据"""
        result = []
        for f in funding_raw:
            result.append({
                "symbol": symbol,
                "timestamp": int(f.get("fundingTime", 0)),
                "funding_rate": float(f.get("fundingRate", 0)),
                "predicted_rate": float(f.get("predictedRate", 0)),
                "funding_time_str": f.get("fundingTimeStr", ""),
            })
        return sorted(result, key=lambda x: x["timestamp"])

    def _normalize_trades(self, trades_raw: List[Dict[str, Any]], symbol: str) -> List[Dict[str, Any]]:
        """标准化成交数据"""
        result = []
        for t in trades_raw:
            result.append({
                "symbol": symbol,
                "timestamp": int(t.get("ts", 0)),
                "price": float(t.get("px", 0)),
                "volume": float(t.get("sz", 0)),
                "side": t.get("side", ""),
                "trade_id": t.get("tradeId", ""),
            })
        return sorted(result, key=lambda x: x["timestamp"])

    def _calc_basis(self, futures_klines: List[Any], spot_klines: List[Any],
                   symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """计算基差"""
        futures_norm = self._normalize_klines(futures_klines, symbol, timeframe)
        spot_norm = self._normalize_klines(spot_klines, symbol.replace("-SWAP", ""), timeframe)

        # 按时间戳对齐
        futures_map = {k["timestamp"]: k for k in futures_norm}
        spot_map = {k["timestamp"]: k for k in spot_norm}

        result = []
        for ts in sorted(set(futures_map.keys()) & set(spot_map.keys())):
            futures = futures_map[ts]
            spot = spot_map[ts]
            basis = futures["close"] - spot["close"]
            basis_pct = basis / spot["close"] * 100 if spot["close"] > 0 else 0

            result.append({
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": ts,
                "futures_close": futures["close"],
                "spot_close": spot["close"],
                "basis": basis,
                "basis_pct": basis_pct,
            })

        return result

    def _validate_klines(self, klines: List[Dict[str, Any]],
                        symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """验证K线数据完整性"""
        if not klines:
            return klines

        valid = []
        interval_ms = self._timeframe_to_ms(timeframe)
        prev_ts = klines[0]["timestamp"]
        valid.append(klines[0])

        for k in klines[1:]:
            current_ts = k["timestamp"]

            # 检查时间连续性（允许最多2个周期的缺失）
            gap = current_ts - prev_ts
            if gap > interval_ms * 2:
                logger.warning(f"Kline gap detected for {symbol} {timeframe}: "
                             f"{prev_ts} -> {current_ts} ({gap}ms)")

            # 检查价格合理性
            if not self._validate_kline_price(k):
                logger.warning(f"Invalid kline price for {symbol} at {current_ts}: {k}")
                continue

            valid.append(k)
            prev_ts = current_ts

        return valid

    def _validate_kline_price(self, kline: Dict[str, Any]) -> bool:
        """验证单根K线价格"""
        try:
            o = kline["open"]
            h = kline["high"]
            l = kline["low"]
            c = kline["close"]

            if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                return False

            if h < l:
                return False

            if c > h or c < l:
                return False

            if o > h or o < l:
                return False

            return True
        except Exception:
            return False

    def get_fetch_state(self) -> Dict[str, Dict[str, int]]:
        """获取拉取状态"""
        with self._lock:
            return dict(self._fetch_state)

    def reset_fetch_state(self, symbol: str = "", timeframe: str = "") -> None:
        """重置拉取状态"""
        with self._lock:
            if symbol and timeframe:
                if symbol in self._fetch_state and timeframe in self._fetch_state[symbol]:
                    del self._fetch_state[symbol][timeframe]
            elif symbol:
                if symbol in self._fetch_state:
                    del self._fetch_state[symbol]
            else:
                self._fetch_state.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            total_symbols = len(self._fetch_state)
            total_timeframes = sum(len(v) for v in self._fetch_state.values())

        return {
            "okx_client_available": self._okx_client is not None,
            "max_concurrent_fetches": self._semaphore._value,
            "fetch_interval": self._fetch_interval,
            "max_limit": self._max_limit,
            "validate_enabled": self._validate_enabled,
            "tracked_symbols": total_symbols,
            "tracked_timeframes": total_timeframes,
            "fetch_state": self.get_fetch_state(),
        }
