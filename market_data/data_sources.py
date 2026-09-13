"""
市场数据源抽象层
定义统一的数据源接口，支持多种数据源接入。
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from loguru import logger


class DataSource(ABC):
    """数据源抽象基类"""

    SOURCE_TYPE = "abstract"
    PRIORITY = 100  # 优先级，数字越小优先级越高

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._healthy = True
        self._last_success = 0.0
        self._last_failure = 0.0
        self._consecutive_failures = 0
        self._total_requests = 0
        self._total_failures = 0

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取行情数据"""
        pass

    @abstractmethod
    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
        """获取订单簿"""
        pass

    @abstractmethod
    async def fetch_klines(self, symbol: str, timeframe: str = "1m", 
                          limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """获取K线数据"""
        pass

    def is_healthy(self) -> bool:
        """数据源是否健康"""
        return self._healthy and self._consecutive_failures < 5

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._total_requests
        return {
            "source_type": self.SOURCE_TYPE,
            "healthy": self.is_healthy(),
            "total_requests": total,
            "total_failures": self._total_failures,
            "failure_rate": self._total_failures / max(total, 1),
            "consecutive_failures": self._consecutive_failures,
            "last_success": self._last_success,
            "last_failure": self._last_failure,
            "priority": self.PRIORITY,
        }

    def _record_success(self):
        """记录成功请求"""
        self._last_success = time.time()
        self._consecutive_failures = 0
        self._total_requests += 1
        self._healthy = True

    def _record_failure(self):
        """记录失败请求"""
        self._last_failure = time.time()
        self._consecutive_failures += 1
        self._total_requests += 1
        self._total_failures += 1
        if self._consecutive_failures >= 5:
            self._healthy = False


class OKXDataSource(DataSource):
    """OKX 数据源（主要）"""

    SOURCE_TYPE = "okx"
    PRIORITY = 1

    def __init__(self, okx_client=None, config: Dict[str, Any] = None):
        super().__init__(config)
        self._okx_client = okx_client

    def set_client(self, okx_client):
        self._okx_client = okx_client

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        if not self._okx_client:
            self._record_failure()
            return None
        try:
            ticker = await asyncio.to_thread(self._okx_client.get_ticker, symbol)
            if ticker:
                self._record_success()
                return self._normalize_ticker(ticker, symbol)
            self._record_failure()
            return None
        except Exception as e:
            logger.debug(f"OKX fetch_ticker failed for {symbol}: {e}")
            self._record_failure()
            return None

    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
        if not self._okx_client:
            self._record_failure()
            return None
        try:
            orderbook = await asyncio.to_thread(self._okx_client.get_orderbook, symbol, depth)
            if orderbook:
                self._record_success()
                return self._normalize_orderbook(orderbook, symbol)
            self._record_failure()
            return None
        except Exception as e:
            logger.debug(f"OKX fetch_orderbook failed for {symbol}: {e}")
            self._record_failure()
            return None

    async def fetch_klines(self, symbol: str, timeframe: str = "1m", 
                          limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        if not self._okx_client:
            self._record_failure()
            return None
        try:
            klines = await asyncio.to_thread(self._okx_client.get_klines, symbol, timeframe, limit)
            if klines:
                self._record_success()
                return self._normalize_klines(klines, symbol, timeframe)
            self._record_failure()
            return None
        except Exception as e:
            logger.debug(f"OKX fetch_klines failed for {symbol}: {e}")
            self._record_failure()
            return None

    def _normalize_ticker(self, ticker: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """标准化行情数据"""
        return {
            "symbol": symbol,
            "source": self.SOURCE_TYPE,
            "timestamp": ticker.get("ts", int(time.time() * 1000)),
            "last": float(ticker.get("last", 0)),
            "bid": float(ticker.get("bidPx", 0)),
            "ask": float(ticker.get("askPx", 0)),
            "volume_24h": float(ticker.get("vol24h", 0)),
            "change_24h": float(ticker.get("chg24h", 0)),
            "high_24h": float(ticker.get("high24h", 0)),
            "low_24h": float(ticker.get("low24h", 0)),
        }

    def _normalize_orderbook(self, orderbook: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """标准化订单簿数据"""
        return {
            "symbol": symbol,
            "source": self.SOURCE_TYPE,
            "timestamp": orderbook.get("ts", int(time.time() * 1000)),
            "bids": orderbook.get("bids", [])[:20],
            "asks": orderbook.get("asks", [])[:20],
        }

    def _normalize_klines(self, klines: List[Any], symbol: str, timeframe: str) -> List[Dict[str, Any]]:
        """标准化K线数据"""
        result = []
        for k in klines:
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
        return result


class SimulatedDataSource(DataSource):
    """模拟数据源（备用）"""

    SOURCE_TYPE = "simulated"
    PRIORITY = 99

    def __init__(self, simulated_data=None, config: Dict[str, Any] = None):
        super().__init__(config)
        self._simulated_data = simulated_data
        self._base_prices: Dict[str, float] = {
            "BTC-USDT-SWAP": 60000.0,
            "ETH-USDT-SWAP": 3000.0,
            "SOL-USDT-SWAP": 150.0,
            "PEPE-USDT-SWAP": 0.00001,
        }

    async def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        try:
            base = self._base_prices.get(symbol, 100.0)
            now = int(time.time() * 1000)
            import random
            random.seed(now // 1000)
            last = base * (1 + random.uniform(-0.001, 0.001))
            
            self._record_success()
            return {
                "symbol": symbol,
                "source": self.SOURCE_TYPE,
                "timestamp": now,
                "last": round(last, 4),
                "bid": round(last * 0.999, 4),
                "ask": round(last * 1.001, 4),
                "volume_24h": 1000000.0,
                "change_24h": 0.0,
                "high_24h": round(last * 1.02, 4),
                "low_24h": round(last * 0.98, 4),
            }
        except Exception as e:
            logger.debug(f"Simulated fetch_ticker failed: {e}")
            self._record_failure()
            return None

    async def fetch_orderbook(self, symbol: str, depth: int = 20) -> Optional[Dict[str, Any]]:
        try:
            ticker = await self.fetch_ticker(symbol)
            if not ticker:
                return None
            
            last = ticker["last"]
            bids = []
            asks = []
            for i in range(depth):
                bids.append([str(last * (1 - 0.0001 * (i + 1))), str(10 + i)])
                asks.append([str(last * (1 + 0.0001 * (i + 1))), str(10 + i)])
            
            return {
                "symbol": symbol,
                "source": self.SOURCE_TYPE,
                "timestamp": int(time.time() * 1000),
                "bids": bids,
                "asks": asks,
            }
        except Exception:
            self._record_failure()
            return None

    async def fetch_klines(self, symbol: str, timeframe: str = "1m", 
                          limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        try:
            import random
            base = self._base_prices.get(symbol, 100.0)
            now = int(time.time() * 1000)
            interval_ms = self._timeframe_to_ms(timeframe)
            
            klines = []
            for i in range(limit):
                ts = now - i * interval_ms
                random.seed(ts)
                open_p = base * (1 + random.uniform(-0.005, 0.005))
                close_p = open_p * (1 + random.uniform(-0.003, 0.003))
                high_p = max(open_p, close_p) * (1 + random.uniform(0, 0.002))
                low_p = min(open_p, close_p) * (1 - random.uniform(0, 0.002))
                volume = random.uniform(100, 1000)
                
                klines.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": ts,
                    "open": round(open_p, 4),
                    "high": round(high_p, 4),
                    "low": round(low_p, 4),
                    "close": round(close_p, 4),
                    "volume": round(volume, 4),
                })
            
            self._record_success()
            return list(reversed(klines))
        except Exception:
            self._record_failure()
            return None

    def _timeframe_to_ms(self, timeframe: str) -> int:
        """时间框架转毫秒"""
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
