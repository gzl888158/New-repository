"""
市场数据融合管理器
==================
统一协调多数据源、WebSocket实时推送、历史数据拉取、
缓存池、质量监控、持久化存储，提供标准化的数据访问接口。

核心特性：
- 多源冗余备份：WebSocket + REST 轮询双链路兜底
- per-symbol 独立缓存池，隔离币种数据
- 数据清洗校验引擎，确保数据质量
- Tick 数据本地持久化，供实时回测使用
- 自动故障切换和恢复机制
"""

import asyncio
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
from loguru import logger
from collections import defaultdict

from .websocket_feed import OKXWebSocketFeed, FeedType, FeedState
from .historical_loader import HistoricalLoader
from .symbol_data_pool import SymbolDataPoolManager
from .quality_monitor import DataCleaningEngine
from .tick_persistence import TickPersistence


class DataMode(Enum):
    """数据模式"""
    REALTIME = "realtime"
    HISTORICAL = "historical"
    HYBRID = "hybrid"


class MarketDataManager:
    """市场数据融合管理器"""

    def __init__(self, config: Dict[str, Any] = None, okx_client=None):
        self.config = config or {}

        # 核心模块
        self._ws_feed = OKXWebSocketFeed(config=self.config.get("websocket", {}))
        self._historical_loader = HistoricalLoader(okx_client=okx_client, config=self.config.get("historical", {}))
        self._data_pool = SymbolDataPoolManager(config=self.config.get("data_pool", {}))
        self._quality_engine = DataCleaningEngine(config=self.config.get("quality", {}))
        self._tick_persistence = TickPersistence(config=self.config.get("persistence", {}))

        # OKX 客户端
        self._okx_client = okx_client

        # 配置参数
        self._rest_poll_interval = self.config.get("rest_poll_interval", 1.0)
        self._failover_threshold = self.config.get("failover_threshold", 5.0)
        self._recovery_threshold = self.config.get("recovery_threshold", 10.0)
        self._max_symbols = self.config.get("max_symbols", 20)

        # 状态管理
        self._running = False
        self._data_mode = DataMode.REALTIME
        self._active_symbols: List[str] = []

        # 故障切换状态
        self._ws_healthy = False
        self._rest_backup_active = False
        self._last_ws_message_ts: Dict[str, float] = {}
        self._failover_lock = threading.RLock()

        # 回调管理
        self._callbacks: Dict[str, List[Callable[[Dict[str, Any]], None]]] = defaultdict(list)

        # 定时任务
        self._poll_tasks: Dict[str, asyncio.Task] = {}
        self._health_check_task = None

        # 统计信息
        self._stats = {
            "ticker_updates": 0,
            "orderbook_updates": 0,
            "trade_updates": 0,
            "kline_updates": 0,
            "failover_count": 0,
            "recovery_count": 0,
        }

        # 注册 WebSocket 回调
        self._ws_feed.add_callback(self._on_ws_message)

        logger.info("MarketDataManager initialized with all modules")

    def set_okx_client(self, okx_client) -> None:
        """设置 OKX 客户端"""
        self._okx_client = okx_client
        self._historical_loader.set_okx_client(okx_client)

    async def start(self) -> None:
        """启动市场数据管理器"""
        self._running = True

        await self._ws_feed.start()
        self._tick_persistence.start()

        asyncio.create_task(self._health_check_loop())
        asyncio.create_task(self._persistence_monitor_loop())

        logger.info("MarketDataManager started")

    async def stop(self) -> None:
        """停止市场数据管理器"""
        self._running = False

        await self._ws_feed.stop()
        self._tick_persistence.stop()

        for task in self._poll_tasks.values():
            task.cancel()
        self._poll_tasks.clear()

        if self._health_check_task:
            self._health_check_task.cancel()

        logger.info("MarketDataManager stopped")

    def subscribe_symbols(self, symbols: List[str], feed_types: List[FeedType] = None) -> None:
        """订阅交易对"""
        feed_types = feed_types or [FeedType.TICKER, FeedType.ORDERBOOK, FeedType.TRADES]

        with self._failover_lock:
            new_symbols = [s for s in symbols if s not in self._active_symbols]
            self._active_symbols.extend(new_symbols)

            # 限制最大订阅数量
            if len(self._active_symbols) > self._max_symbols:
                logger.warning(f"Max symbols ({self._max_symbols}) exceeded, truncating")
                self._active_symbols = self._active_symbols[:self._max_symbols]

            for feed_type in feed_types:
                self._ws_feed.subscribe(feed_type, new_symbols)

        logger.info(f"Subscribed to {len(new_symbols)} symbols: {new_symbols}")

    def unsubscribe_symbols(self, symbols: List[str]) -> None:
        """取消订阅交易对"""
        with self._failover_lock:
            for s in symbols:
                if s in self._active_symbols:
                    self._active_symbols.remove(s)

            for feed_type in [FeedType.TICKER, FeedType.ORDERBOOK, FeedType.TRADES]:
                self._ws_feed.unsubscribe(feed_type, symbols)

            # 停止 REST 轮询
            for s in symbols:
                if s in self._poll_tasks:
                    self._poll_tasks[s].cancel()
                    del self._poll_tasks[s]

        logger.info(f"Unsubscribed from {len(symbols)} symbols")

    def add_callback(self, callback: Callable[[Dict[str, Any]], None],
                     data_type: str = "all") -> None:
        """添加数据回调"""
        self._callbacks[data_type].append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None],
                        data_type: str = "all") -> None:
        """移除数据回调"""
        if callback in self._callbacks[data_type]:
            self._callbacks[data_type].remove(callback)

    def _on_ws_message(self, message: Dict[str, Any]) -> None:
        """WebSocket 消息处理"""
        feed_type = message.get("feed_type", "")
        symbol = message.get("symbol", "")

        if not symbol:
            return

        # 更新最后消息时间
        self._last_ws_message_ts[symbol] = time.time()
        self._ws_healthy = True

        # 数据清洗校验
        cleaned = None
        if feed_type == FeedType.TICKER.value:
            cleaned = self._quality_engine.clean_and_validate_ticker(message)
        elif feed_type == FeedType.ORDERBOOK.value:
            cleaned = self._quality_engine.clean_and_validate_orderbook(message)
        elif feed_type == FeedType.TRADES.value:
            cleaned = self._quality_engine.clean_and_validate_trade(message)
        elif feed_type == FeedType.KLINE.value:
            cleaned = self._quality_engine.clean_and_validate_kline(message)

        if cleaned:
            # 更新数据缓存池
            if feed_type == FeedType.TICKER.value:
                self._data_pool.update_ticker(cleaned)
                self._tick_persistence.persist_ticker(cleaned)
                self._stats["ticker_updates"] += 1
            elif feed_type == FeedType.ORDERBOOK.value:
                self._data_pool.update_orderbook(cleaned)
                self._tick_persistence.persist_orderbook(cleaned)
                self._stats["orderbook_updates"] += 1
            elif feed_type == FeedType.TRADES.value:
                self._data_pool.update_trade(cleaned)
                self._tick_persistence.persist_trade(cleaned)
                self._stats["trade_updates"] += 1
            elif feed_type == FeedType.KLINE.value:
                timeframe = message.get("timeframe", "1m")
                self._data_pool.update_kline(cleaned, timeframe)
                self._stats["kline_updates"] += 1

            # 触发回调
            self._notify_callbacks(cleaned, feed_type)

    def _notify_callbacks(self, data: Dict[str, Any], data_type: str) -> None:
        """通知所有回调"""
        for callback in self._callbacks.get("all", []):
            try:
                callback(data)
            except Exception as e:
                logger.debug(f"Callback error: {e}")

        for callback in self._callbacks.get(data_type, []):
            try:
                callback(data)
            except Exception as e:
                logger.debug(f"Callback error for {data_type}: {e}")

    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            try:
                await asyncio.sleep(1.0)

                # 检查 WebSocket 连接状态
                ws_state = self._ws_feed.get_state()
                if ws_state == FeedState.CONNECTED.value:
                    self._ws_healthy = True
                else:
                    self._ws_healthy = False

                # 检查各交易对的数据更新情况
                now = time.time()
                for symbol in self._active_symbols:
                    last_ts = self._last_ws_message_ts.get(symbol, 0)
                    if now - last_ts > self._failover_threshold:
                        self._trigger_rest_backup(symbol)

                    if now - last_ts < self._recovery_threshold and self._rest_backup_active:
                        self._stop_rest_backup(symbol)

            except Exception as e:
                logger.debug(f"Health check error: {e}")

    def _trigger_rest_backup(self, symbol: str) -> None:
        """触发 REST 轮询备份"""
        with self._failover_lock:
            if symbol in self._poll_tasks:
                return

            if not self._rest_backup_active:
                self._rest_backup_active = True
                self._stats["failover_count"] += 1
                logger.warning(f"Triggering REST backup for {symbol} (WS disconnected)")

            self._poll_tasks[symbol] = asyncio.create_task(self._rest_poll_loop(symbol))

    def _stop_rest_backup(self, symbol: str) -> None:
        """停止 REST 轮询备份"""
        with self._failover_lock:
            if symbol in self._poll_tasks:
                self._poll_tasks[symbol].cancel()
                del self._poll_tasks[symbol]

            # 检查是否所有备份都已停止
            if not self._poll_tasks:
                self._rest_backup_active = False
                self._stats["recovery_count"] += 1
                logger.info("WebSocket recovered, REST backup stopped")

    async def _rest_poll_loop(self, symbol: str) -> None:
        """REST 轮询循环"""
        while self._running:
            try:
                if not self._okx_client:
                    await asyncio.sleep(1.0)
                    continue

                # 通过 REST API 获取数据
                ticker_raw = await asyncio.to_thread(self._okx_client.get_ticker, symbol)
                if ticker_raw:
                    ticker = {
                        "symbol": symbol,
                        "feed_type": FeedType.TICKER.value,
                        "timestamp": int(time.time() * 1000),
                        "price": float(ticker_raw.get("last", 0)),
                        "bid_price": float(ticker_raw.get("bid", 0)),
                        "ask_price": float(ticker_raw.get("ask", 0)),
                        "bid_volume": float(ticker_raw.get("bidSize", 0)),
                        "ask_volume": float(ticker_raw.get("askSize", 0)),
                        "volume_24h": float(ticker_raw.get("vol24h", 0)),
                        "change_24h": float(ticker_raw.get("chg24h", 0)),
                        "high_24h": float(ticker_raw.get("high24h", 0)),
                        "low_24h": float(ticker_raw.get("low24h", 0)),
                        "funding_rate": float(ticker_raw.get("fundingRate", 0)),
                    }

                    cleaned = self._quality_engine.clean_and_validate_ticker(ticker)
                    if cleaned:
                        self._data_pool.update_ticker(cleaned)
                        self._tick_persistence.persist_ticker(cleaned)
                        self._notify_callbacks(cleaned, FeedType.TICKER.value)

                await asyncio.sleep(self._rest_poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"REST poll error for {symbol}: {e}")
                await asyncio.sleep(1.0)

    async def _persistence_monitor_loop(self) -> None:
        """持久化监控循环"""
        while self._running:
            try:
                await asyncio.sleep(60)
                stats = self._tick_persistence.get_stats()
                db_size = self._tick_persistence.get_database_size()
                logger.debug(f"Persistence stats: {stats}, DB size: {db_size}MB")
            except Exception as e:
                logger.debug(f"Persistence monitor error: {e}")

    def get_latest_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新行情"""
        return self._data_pool.get_latest_ticker(symbol)

    def get_latest_orderbook(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新订单簿"""
        return self._data_pool.get_latest_orderbook(symbol)

    def get_latest_funding(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取最新资金费率"""
        return self._data_pool.get_latest_funding(symbol)

    def get_recent_ticks(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近 tick 数据"""
        return self._data_pool.get_recent_ticks(symbol, count)

    def get_recent_trades(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近成交记录"""
        return self._data_pool.get_recent_trades(symbol, count)

    def get_klines(self, symbol: str, timeframe: str = "1m", count: int = 100) -> List[Dict[str, Any]]:
        """获取 K 线数据"""
        return self._data_pool.get_klines(symbol, timeframe, count)

    def get_funding_rates(self, symbol: str, count: int = 100) -> List[Dict[str, Any]]:
        """获取资金费率历史"""
        return self._data_pool.get_funding_rates(symbol, count)

    async def fetch_historical_klines(self, symbol: str, timeframe: str = "1m",
                                      days: int = 30) -> Optional[List[Dict[str, Any]]]:
        """拉取历史 K 线"""
        return await self._historical_loader.fetch_historical_klines_full(symbol, timeframe, days)

    async def fetch_multiple_historical_klines(self, symbols: List[str],
                                               timeframe: str = "1m",
                                               days: int = 1) -> Dict[str, List[Dict[str, Any]]]:
        """批量拉取历史 K 线"""
        return await self._historical_loader.fetch_multiple_symbols_klines(symbols, timeframe, days)

    async def fetch_funding_rate(self, symbol: str, limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """拉取资金费率"""
        return await self._historical_loader.fetch_funding_rate(symbol, limit)

    async def fetch_basis(self, symbol: str, timeframe: str = "1m",
                          limit: int = 100) -> Optional[List[Dict[str, Any]]]:
        """拉取基差数据"""
        return await self._historical_loader.fetch_basis(symbol, timeframe, limit)

    def query_historical_tickers(self, symbol: str, start_ts: int, end_ts: int,
                                 limit: int = 10000) -> List[Dict[str, Any]]:
        """查询历史 ticker 数据"""
        return self._tick_persistence.query_tickers(symbol, start_ts, end_ts, limit)

    def query_historical_trades(self, symbol: str, start_ts: int, end_ts: int,
                                limit: int = 10000) -> List[Dict[str, Any]]:
        """查询历史 trade 数据"""
        return self._tick_persistence.query_trades(symbol, start_ts, end_ts, limit)

    def query_historical_orderbooks(self, symbol: str, start_ts: int, end_ts: int,
                                    limit: int = 1000) -> List[Dict[str, Any]]:
        """查询历史 orderbook 数据"""
        return self._tick_persistence.query_orderbooks(symbol, start_ts, end_ts, limit)

    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        ws_stats = self._ws_feed.get_stats()
        pool_overview = self._data_pool.get_overview()
        quality_stats = self._quality_engine.get_stats()
        persistence_stats = self._tick_persistence.get_stats()
        historical_stats = self._historical_loader.get_stats()
        wal_status = self._tick_persistence.get_wal_status()

        return {
            "running": self._running,
            "data_mode": self._data_mode.value,
            "active_symbols": self._active_symbols,
            "ws_healthy": self._ws_healthy,
            "rest_backup_active": self._rest_backup_active,
            "websocket": ws_stats,
            "data_pool": pool_overview,
            "quality": quality_stats,
            "persistence": persistence_stats,
            "wal": wal_status,
            "historical": historical_stats,
            "stats": self._stats,
        }

    def get_quality_score(self, symbol: str = "") -> float:
        """获取数据质量评分"""
        return self._quality_engine.get_quality_score(symbol)

    def get_recent_issues(self, limit: int = 50, severity: str = None) -> List[Dict[str, Any]]:
        """获取最近质量问题"""
        return self._quality_engine.get_recent_issues(limit, severity)

    def get_symbol_list(self) -> List[str]:
        """获取所有交易对列表"""
        return self._data_pool.get_symbols()

    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取缓存池统计信息"""
        return self._data_pool.get_all_stats()

    def export_to_csv(self, symbol: str, start_ts: int, end_ts: int,
                      output_path: str, data_type: str = "ticker") -> bool:
        """导出数据为 CSV"""
        return self._tick_persistence.export_to_csv(symbol, start_ts, end_ts, output_path, data_type)

    def get_time_range(self, symbol: str) -> Optional[tuple]:
        """获取指定交易对的数据时间范围"""
        return self._tick_persistence.get_time_range(symbol)

    # ========== 兼容适配层（供旧调用方使用）==========

    async def get_ticker(self, symbol: str, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """兼容旧接口：异步获取行情（优先从缓存池读取，回退到REST拉取）"""
        ticker = self.get_latest_ticker(symbol)
        if ticker:
            return ticker

        # 缓存未命中，尝试 REST 拉取
        if self._okx_client:
            try:
                ticker_raw = await asyncio.to_thread(self._okx_client.get_ticker, symbol)
                if ticker_raw:
                    ticker = {
                        "symbol": symbol,
                        "feed_type": FeedType.TICKER.value,
                        "timestamp": int(time.time() * 1000),
                        "price": float(ticker_raw.get("last", 0)),
                        "bid_price": float(ticker_raw.get("bid", 0)),
                        "ask_price": float(ticker_raw.get("ask", 0)),
                        "bid_volume": float(ticker_raw.get("bidSize", 0)),
                        "ask_volume": float(ticker_raw.get("askSize", 0)),
                        "volume_24h": float(ticker_raw.get("vol24h", 0)),
                        "change_24h": float(ticker_raw.get("chg24h", 0)),
                        "high_24h": float(ticker_raw.get("high24h", 0)),
                        "low_24h": float(ticker_raw.get("low24h", 0)),
                        "funding_rate": float(ticker_raw.get("fundingRate", 0)),
                    }
                    cleaned = self._quality_engine.clean_and_validate_ticker(ticker)
                    if cleaned:
                        self._data_pool.update_ticker(cleaned)
                        return cleaned
            except Exception as e:
                logger.debug(f"get_ticker fallback failed for {symbol}: {e}")

        return None

    async def get_orderbook(self, symbol: str, depth: int = 20,
                            use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """兼容旧接口：异步获取订单簿"""
        orderbook = self.get_latest_orderbook(symbol)
        if orderbook:
            return orderbook

        if self._okx_client:
            try:
                ob_raw = await asyncio.to_thread(
                    self._okx_client.get_orderbook, symbol, depth
                )
                if ob_raw:
                    orderbook = {
                        "symbol": symbol,
                        "feed_type": FeedType.ORDERBOOK.value,
                        "timestamp": int(time.time() * 1000),
                        "bids": [[float(b[0]), float(b[1])] for b in ob_raw.get("bids", [])],
                        "asks": [[float(a[0]), float(a[1])] for a in ob_raw.get("asks", [])],
                    }
                    cleaned = self._quality_engine.clean_and_validate_orderbook(orderbook)
                    if cleaned:
                        self._data_pool.update_orderbook(cleaned)
                        return cleaned
            except Exception as e:
                logger.debug(f"get_orderbook fallback failed for {symbol}: {e}")

        return None

    async def get_klines(self, symbol: str, timeframe: str = "1m",
                         limit: int = 100, use_cache: bool = True) -> Optional[List[Dict[str, Any]]]:
        """兼容旧接口：异步获取K线数据"""
        # 直接调用底层数据池同步方法，避免与 async 版本同名导致无限递归
        klines = self._data_pool.get_klines(symbol, timeframe, limit)
        if klines:
            return klines

        if self._okx_client:
            try:
                klines_raw = await asyncio.to_thread(
                    self._okx_client.get_klines, symbol, timeframe, limit
                )
                if klines_raw:
                    from .historical_loader import HistoricalLoader
                    normalized = HistoricalLoader._normalize_klines(
                        HistoricalLoader.__new__(HistoricalLoader),
                        klines_raw, symbol, timeframe
                    )
                    for k in normalized:
                        cleaned = self._quality_engine.clean_and_validate_kline(k)
                        if cleaned:
                            self._data_pool.update_kline(cleaned, timeframe)
                    return normalized if normalized else None
            except Exception as e:
                logger.debug(f"get_klines fallback failed for {symbol}: {e}")

        return None

    async def get_multiple_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """兼容旧接口：批量获取行情"""
        tasks = [self.get_ticker(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        result_dict = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.debug(f"Error fetching {symbol}: {result}")
                continue
            if result is not None:
                result_dict[symbol] = result

        return result_dict

    def cleanup_cache(self) -> int:
        """兼容旧接口：清理过期缓存（返回清理数量）"""
        stale_symbols = self._data_pool.get_stale_symbols()
        count = 0
        for symbol in stale_symbols:
            pool = self._data_pool.get_pool(symbol)
            pool.clear()
            count += 1
        return count
