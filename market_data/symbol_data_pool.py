"""
per-symbol 独立数据缓存池
========================
核心定位：为每个交易对提供独立的数据缓存空间，隔离币种数据，
避免单币种数据处理卡顿阻塞全系统。

特性：
- 支持最多20个交易对独立缓存池
- 每个缓存池包含：最新tick、K线窗口、订单簿、成交记录
- LRU淘汰策略，自动清理过期数据
- 读写分离，支持并发访问
- 支持缓存统计和监控
"""

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, OrderedDict
from enum import Enum
from loguru import logger


class CacheItemType(Enum):
    """缓存项类型"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADE = "trade"
    KLINE = "kline"
    FUNDING = "funding"
    BASIS = "basis"


class SymbolCachePool:
    """单个交易对的数据缓存池"""

    def __init__(self, symbol: str, config: Dict[str, Any] = None):
        self.symbol = symbol
        self.config = config or {}

        # 缓存容量配置
        self._max_ticks = config.get("max_ticks", 10000)
        self._max_trades = config.get("max_trades", 5000)
        self._max_orderbook_snapshots = config.get("max_orderbook_snapshots", 100)
        self._max_klines = config.get("max_klines", 1000)
        self._max_funding = config.get("max_funding", 100)

        # 最新数据
        self._latest_ticker = None
        self._latest_orderbook = None
        self._latest_funding = None

        # 时间序列数据
        self._ticks = deque(maxlen=self._max_ticks)
        self._trades = deque(maxlen=self._max_trades)
        self._orderbook_snapshots = deque(maxlen=self._max_orderbook_snapshots)
        self._funding_rates = deque(maxlen=self._max_funding)

        # K线数据（按时间周期存储）
        self._klines: Dict[str, deque] = {}

        # 锁
        self._lock = threading.RLock()

        # 统计信息
        self._stats = {
            "ticker_count": 0,
            "orderbook_count": 0,
            "trade_count": 0,
            "kline_count": 0,
            "funding_count": 0,
            "hit_count": 0,
            "miss_count": 0,
        }

        # 时间戳
        self._last_update_ts = 0.0
        self._last_access_ts = 0.0

        logger.debug(f"SymbolCachePool created for {symbol}")

    def update_ticker(self, ticker: Dict[str, Any]) -> None:
        """更新行情数据"""
        with self._lock:
            self._latest_ticker = ticker.copy()
            self._ticks.append(ticker.copy())
            self._stats["ticker_count"] += 1
            self._last_update_ts = time.time()

    def update_orderbook(self, orderbook: Dict[str, Any]) -> None:
        """更新订单簿"""
        with self._lock:
            self._latest_orderbook = orderbook.copy()
            self._orderbook_snapshots.append(orderbook.copy())
            self._stats["orderbook_count"] += 1
            self._last_update_ts = time.time()

    def update_trade(self, trade: Dict[str, Any]) -> None:
        """更新成交记录"""
        with self._lock:
            self._trades.append(trade.copy())
            self._stats["trade_count"] += 1

    def update_kline(self, kline: Dict[str, Any], timeframe: str = "1m") -> None:
        """更新K线数据"""
        with self._lock:
            if timeframe not in self._klines:
                self._klines[timeframe] = deque(maxlen=self._max_klines)
            self._klines[timeframe].append(kline.copy())
            self._stats["kline_count"] += 1

    def update_funding_rate(self, funding: Dict[str, Any]) -> None:
        """更新资金费率"""
        with self._lock:
            self._latest_funding = funding.copy()
            self._funding_rates.append(funding.copy())
            self._stats["funding_count"] += 1

    def get_latest_ticker(self) -> Optional[Dict[str, Any]]:
        """获取最新行情"""
        with self._lock:
            self._last_access_ts = time.time()
            if self._latest_ticker:
                self._stats["hit_count"] += 1
                return self._latest_ticker.copy()
            self._stats["miss_count"] += 1
            return None

    def get_latest_orderbook(self) -> Optional[Dict[str, Any]]:
        """获取最新订单簿"""
        with self._lock:
            self._last_access_ts = time.time()
            if self._latest_orderbook:
                self._stats["hit_count"] += 1
                return self._latest_orderbook.copy()
            self._stats["miss_count"] += 1
            return None

    def get_latest_funding(self) -> Optional[Dict[str, Any]]:
        """获取最新资金费率"""
        with self._lock:
            self._last_access_ts = time.time()
            if self._latest_funding:
                self._stats["hit_count"] += 1
                return self._latest_funding.copy()
            self._stats["miss_count"] += 1
            return None

    def get_recent_ticks(self, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近tick数据"""
        with self._lock:
            self._last_access_ts = time.time()
            ticks = list(self._ticks)[-count:]
            if ticks:
                self._stats["hit_count"] += 1
            return [t.copy() for t in ticks]

    def get_recent_trades(self, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近成交记录"""
        with self._lock:
            self._last_access_ts = time.time()
            trades = list(self._trades)[-count:]
            return [t.copy() for t in trades]

    def get_klines(self, timeframe: str = "1m", count: int = 100) -> List[Dict[str, Any]]:
        """获取K线数据"""
        with self._lock:
            self._last_access_ts = time.time()
            if timeframe not in self._klines:
                self._stats["miss_count"] += 1
                return []

            klines = list(self._klines[timeframe])[-count:]
            if klines:
                self._stats["hit_count"] += 1
            return [k.copy() for k in klines]

    def get_funding_rates(self, count: int = 100) -> List[Dict[str, Any]]:
        """获取资金费率历史"""
        with self._lock:
            self._last_access_ts = time.time()
            rates = list(self._funding_rates)[-count:]
            return [r.copy() for r in rates]

    def get_orderbook_snapshots(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取订单簿快照"""
        with self._lock:
            self._last_access_ts = time.time()
            snapshots = list(self._orderbook_snapshots)[-count:]
            return [s.copy() for s in snapshots]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "symbol": self.symbol,
                "ticks_count": len(self._ticks),
                "trades_count": len(self._trades),
                "orderbook_snapshots_count": len(self._orderbook_snapshots),
                "funding_rates_count": len(self._funding_rates),
                "klines_count": {k: len(v) for k, v in self._klines.items()},
                "latest_ticker_ts": self._latest_ticker.get("timestamp", 0) if self._latest_ticker else 0,
                "latest_orderbook_ts": self._latest_orderbook.get("timestamp", 0) if self._latest_orderbook else 0,
                "last_update_ts": round(self._last_update_ts, 2),
                "last_access_ts": round(self._last_access_ts, 2),
                **self._stats,
            }

    def is_stale(self, stale_threshold: float = 30.0) -> bool:
        """判断缓存是否过期"""
        return time.time() - self._last_update_ts > stale_threshold

    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._latest_ticker = None
            self._latest_orderbook = None
            self._latest_funding = None
            self._ticks.clear()
            self._trades.clear()
            self._orderbook_snapshots.clear()
            self._funding_rates.clear()
            self._klines.clear()
            self._last_update_ts = 0.0
            logger.debug(f"SymbolCachePool cleared for {self.symbol}")


class SymbolDataPoolManager:
    """多交易对数据缓存池管理器"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        # 最大交易对数量
        self._max_symbols = self.config.get("max_symbols", 20)

        # 缓存池字典
        self._pools: Dict[str, SymbolCachePool] = {}

        # LRU访问顺序（用于淘汰策略）
        self._lru_order = OrderedDict()

        # 锁
        self._lock = threading.RLock()

        # 缓存池配置
        self._pool_config = self.config.get("pool_config", {})

        logger.info(f"SymbolDataPoolManager initialized (max {self._max_symbols} symbols)")

    def get_pool(self, symbol: str) -> SymbolCachePool:
        """获取或创建交易对缓存池"""
        with self._lock:
            if symbol in self._pools:
                self._lru_order.move_to_end(symbol)
                return self._pools[symbol]

            if len(self._pools) >= self._max_symbols:
                oldest = next(iter(self._lru_order))
                logger.info(f"LRU eviction: removing {oldest}")
                self._evict(oldest)

            pool = SymbolCachePool(symbol, self._pool_config)
            self._pools[symbol] = pool
            self._lru_order[symbol] = time.time()
            logger.info(f"Created SymbolCachePool for {symbol}")

            return pool

    def _evict(self, symbol: str) -> None:
        """淘汰交易对缓存池"""
        if symbol in self._pools:
            self._pools[symbol].clear()
            del self._pools[symbol]
            del self._lru_order[symbol]

    def get_symbols(self) -> List[str]:
        """获取所有交易对列表"""
        with self._lock:
            return list(self._pools.keys())

    def has_symbol(self, symbol: str) -> bool:
        """检查是否存在交易对"""
        with self._lock:
            return symbol in self._pools

    def remove_symbol(self, symbol: str) -> bool:
        """移除交易对缓存池"""
        with self._lock:
            if symbol in self._pools:
                self._evict(symbol)
                return True
            return False

    def update_ticker(self, ticker: Dict[str, Any]) -> None:
        """更新行情数据"""
        symbol = ticker.get("symbol", "")
        if symbol:
            pool = self.get_pool(symbol)
            pool.update_ticker(ticker)

    def update_orderbook(self, orderbook: Dict[str, Any]) -> None:
        """更新订单簿"""
        symbol = orderbook.get("symbol", "")
        if symbol:
            pool = self.get_pool(symbol)
            pool.update_orderbook(orderbook)

    def update_trade(self, trade: Dict[str, Any]) -> None:
        """更新成交记录"""
        symbol = trade.get("symbol", "")
        if symbol:
            pool = self.get_pool(symbol)
            pool.update_trade(trade)

    def update_kline(self, kline: Dict[str, Any], timeframe: str = "1m") -> None:
        """更新K线数据"""
        symbol = kline.get("symbol", "")
        if symbol:
            pool = self.get_pool(symbol)
            pool.update_kline(kline, timeframe)

    def update_funding_rate(self, funding: Dict[str, Any]) -> None:
        """更新资金费率"""
        symbol = funding.get("symbol", "")
        if symbol:
            pool = self.get_pool(symbol)
            pool.update_funding_rate(funding)

    def get_latest_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新行情"""
        with self._lock:
            if symbol in self._pools:
                self._lru_order.move_to_end(symbol)
                return self._pools[symbol].get_latest_ticker()
            return None

    def get_latest_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新订单簿"""
        with self._lock:
            if symbol in self._pools:
                self._lru_order.move_to_end(symbol)
                return self._pools[symbol].get_latest_orderbook()
            return None

    def get_latest_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新资金费率"""
        with self._lock:
            if symbol in self._pools:
                self._lru_order.move_to_end(symbol)
                return self._pools[symbol].get_latest_funding()
            return None

    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近tick数据"""
        with self._lock:
            if symbol in self._pools:
                self._lru_order.move_to_end(symbol)
                return self._pools[symbol].get_recent_ticks(count)
            return []

    def get_recent_trades(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近成交记录"""
        with self._lock:
            if symbol in self._pools:
                self._lru_order.move_to_end(symbol)
                return self._pools[symbol].get_recent_trades(count)
            return []

    def get_klines(self, symbol: str, timeframe: str = "1m", count: int = 100) -> List[Dict[str, Any]]:
        """获取K线数据"""
        with self._lock:
            if symbol in self._pools:
                self._lru_order.move_to_end(symbol)
                return self._pools[symbol].get_klines(timeframe, count)
            return []

    def get_funding_rates(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取资金费率历史"""
        with self._lock:
            if symbol in self._pools:
                self._lru_order.move_to_end(symbol)
                return self._pools[symbol].get_funding_rates(count)
            return []

    def get_stale_symbols(self, stale_threshold: float = 30.0) -> List[str]:
        """获取过期的交易对"""
        with self._lock:
            return [s for s, p in self._pools.items() if p.is_stale(stale_threshold)]

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有缓存池统计信息"""
        with self._lock:
            return {s: p.get_stats() for s, p in self._pools.items()}

    def get_overview(self) -> Dict[str, Any]:
        """获取概览信息"""
        with self._lock:
            total_ticks = sum(p.get_stats()["ticks_count"] for p in self._pools.values())
            total_trades = sum(p.get_stats()["trades_count"] for p in self._pools.values())
            total_klines = sum(sum(len(v) for v in p._klines.values()) for p in self._pools.values())

            return {
                "max_symbols": self._max_symbols,
                "current_symbols": len(self._pools),
                "total_ticks": total_ticks,
                "total_trades": total_trades,
                "total_klines": total_klines,
                "symbols": list(self._pools.keys()),
                "lru_order": list(self._lru_order.keys()),
            }

    def clear_all(self) -> None:
        """清空所有缓存池"""
        with self._lock:
            for symbol in list(self._pools.keys()):
                self._evict(symbol)
            logger.info("All SymbolCachePools cleared")
