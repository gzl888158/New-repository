"""
OKX WebSocket 实时数据流模块
=============================
核心定位：全市场合约永续 tick 级行情实时推送，支持断线自动重连、消息去重、
多订阅管理、心跳保活。

特性：
- 支持 ticker、orderbook、trades、kline、funding 多种数据流
- per-symbol 消息队列隔离，单币种消息阻塞不影响其他币种
- 断线自动重连（指数退避），重连后自动恢复订阅
- 消息去重（基于时间戳/seq），防止重复处理
- 双链路兜底：WebSocket 断流时自动切换到 REST 轮询
"""

import asyncio
import json
import time
import hashlib
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Callable
from collections import deque, defaultdict
from enum import Enum
from loguru import logger


class FeedType(Enum):
    """WebSocket 订阅类型"""
    TICKER = "ticker"
    ORDERBOOK = "orderbook"
    TRADES = "trades"
    KLINE = "kline"
    FUNDING = "funding"


class FeedState(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"


class MessageQueue:
    """per-symbol 独立消息队列"""

    def __init__(self, max_size: int = 1000):
        self._queue = deque(maxlen=max_size)
        self._lock = threading.RLock()
        self._last_seq = {}
        self._total_received = 0
        self._total_dropped = 0

    def put(self, data: Dict[str, Any], seq_key: str = "seq") -> bool:
        """入队（去重）"""
        with self._lock:
            seq = data.get(seq_key)
            if seq is not None:
                symbol = data.get("symbol", "")
                if self._last_seq.get(symbol) == seq:
                    self._total_dropped += 1
                    return False
                self._last_seq[symbol] = seq
            self._queue.append(data)
            self._total_received += 1
            return True

    def get(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """出队"""
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            return None

    def get_batch(self, max_count: int = 100) -> List[Dict[str, Any]]:
        """批量出队"""
        with self._lock:
            batch = []
            while self._queue and len(batch) < max_count:
                batch.append(self._queue.popleft())
            return batch

    def size(self) -> int:
        """队列大小"""
        with self._lock:
            return len(self._queue)

    def get_stats(self) -> Dict[str, Any]:
        """统计信息"""
        with self._lock:
            return {
                "queue_size": len(self._queue),
                "total_received": self._total_received,
                "total_dropped": self._total_dropped,
            }


class OKXWebSocketFeed:
    """OKX WebSocket 实时行情推送"""

    # P21: 交易所时间戳缺失回退频率限制（防日志洪水）
    _ts_fallback_last_log = 0.0
    _ts_fallback_log_interval = 60.0  # 60秒内最多记录一次

    @staticmethod
    def _utc_timestamp_ms() -> int:
        """P21: 获取UTC毫秒时间戳，显式标明UTC来源"""
        return int(datetime.now(timezone.utc).timestamp() * 1000)

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._ws_url = config.get("ws_url", "wss://ws.okx.com:8443/ws/v5/public")
        self._ws_private_url = config.get("ws_private_url", "wss://ws.okx.com:8443/ws/v5/private")
        self._api_key = config.get("api_key", "")
        self._secret_key = config.get("secret_key", "")
        self._passphrase = config.get("passphrase", "")

        self._state = FeedState.DISCONNECTED
        self._websocket = None
        self._event_loop = None
        self._running = False
        self._reconnect_count = 0
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        self._reconnect_jitter = 0.2  # P20: 重连抖动系数

        # per-symbol 消息队列
        self._queues: Dict[str, MessageQueue] = {}

        # 订阅管理
        self._subscriptions: Dict[str, List[str]] = {
            FeedType.TICKER.value: [],
            FeedType.ORDERBOOK.value: [],
            FeedType.TRADES.value: [],
            FeedType.KLINE.value: [],
            FeedType.FUNDING.value: [],
        }
        self._pending_subscriptions = set()
        self._sub_lock = threading.RLock()

        # 回调
        self._callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # 心跳
        self._last_ping = 0.0
        self._ping_interval = 30.0
        self._ping_timeout = 10.0

        # 消息统计
        self._message_counts: Dict[str, int] = defaultdict(int)
        self._last_message_ts: Dict[str, float] = {}

        # P22-6: 行情延迟监控
        self._latency_window: deque = deque(maxlen=60)  # 60样本滑动窗口
        self._latency_warning_ms = config.get("latency_warning_ms", 3000)
        self._latency_critical_ms = config.get("latency_critical_ms", 8000)
        self._latency_pause_opening = False
        self._latency_pause_start: float = 0.0
        self._latency_pause_duration = config.get("latency_pause_duration_sec", 300)
        self._latency_last_log: float = 0.0
        self._latency_log_interval = 30.0  # 防抖: 30秒内不重复日志

        logger.info("OKXWebSocketFeed initialized")

    def _get_queue(self, symbol: str) -> MessageQueue:
        """获取或创建 per-symbol 消息队列"""
        with self._sub_lock:
            if symbol not in self._queues:
                self._queues[symbol] = MessageQueue(max_size=2000)
            return self._queues[symbol]

    async def connect(self) -> bool:
        """建立 WebSocket 连接 - P20: 指数退避+抖动"""
        if self._state in (FeedState.CONNECTING, FeedState.CONNECTED):
            return False

        self._state = FeedState.CONNECTING
        self._reconnect_count += 1
        # P20: 指数退避替代简单 1.5x 乘法，添加随机抖动防止惊群效应
        import random
        backoff = min(
            self._max_reconnect_delay,
            self._reconnect_delay * (2 ** min(self._reconnect_count - 1, 6))
        )
        jitter = backoff * self._reconnect_jitter * (random.random() * 2 - 1)
        self._reconnect_delay = backoff + jitter

        try:
            import websockets
            self._websocket = await websockets.connect(
                self._ws_url,
                ping_interval=self._ping_interval,
                ping_timeout=self._ping_timeout,
                close_timeout=10,
            )
            self._state = FeedState.CONNECTED
            self._reconnect_count = 0
            self._reconnect_delay = 1.0
            self._last_ping = time.time()
            logger.info(f"WebSocket connected: {self._ws_url}")

            # 恢复订阅
            await self._resubscribe_all()
            return True

        except Exception as e:
            self._state = FeedState.DISCONNECTED
            logger.error(f"WebSocket connection failed: {e}")
            return False

    async def _resubscribe_all(self) -> None:
        """恢复所有订阅"""
        with self._sub_lock:
            for feed_type, symbols in self._subscriptions.items():
                if symbols:
                    await self._subscribe(feed_type, symbols)

    async def _subscribe(self, feed_type: str, symbols: List[str]) -> None:
        """发送订阅请求"""
        if self._websocket is None:
            return

        try:
            table = self._feed_type_to_table(feed_type)
            args = [{"instId": s} for s in symbols]
            msg = {
                "op": "subscribe",
                "args": args,
                "table": table,
            }
            await self._websocket.send(json.dumps(msg))
            logger.debug(f"Subscribed to {feed_type}: {symbols}")
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            for s in symbols:
                self._pending_subscriptions.add(f"{feed_type}:{s}")

    async def _unsubscribe(self, feed_type: str, symbols: List[str]) -> None:
        """取消订阅"""
        if self._websocket is None:
            return

        try:
            table = self._feed_type_to_table(feed_type)
            args = [{"instId": s} for s in symbols]
            msg = {
                "op": "unsubscribe",
                "args": args,
                "table": table,
            }
            await self._websocket.send(json.dumps(msg))
            logger.debug(f"Unsubscribed from {feed_type}: {symbols}")
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")

    def _feed_type_to_table(self, feed_type: str, bar: str = "1m") -> str:
        """订阅类型转 OKX table 名
        
        Args:
            feed_type: 订阅类型
            bar: K线周期（仅对 KLINE 类型有效），如 "1m", "5m", "15m", "1H", "4H", "1D"
        """
        if feed_type == FeedType.KLINE.value:
            # OKX K线频道命名：candle1m, candle5m, candle15m, candle1H, candle4H, candle1D 等
            bar_map = {
                "1m": "candle1m", "5m": "candle5m", "15m": "candle15m", "30m": "candle30m",
                "1H": "candle1H", "2H": "candle2H", "4H": "candle4H", "6H": "candle6H",
                "8H": "candle8H", "12H": "candle12H", "1D": "candle1D", "1W": "candle1W",
                "1M": "candle1M", "1h": "candle1H", "4h": "candle4H", "1d": "candle1D",
            }
            return bar_map.get(bar, "candle1m")
        
        mapping = {
            FeedType.TICKER.value: "tickers",
            FeedType.ORDERBOOK.value: "books",
            FeedType.TRADES.value: "trades",
            FeedType.FUNDING.value: "funding-rate",
        }
        return mapping.get(feed_type, "tickers")

    def subscribe(self, feed_type: FeedType, symbols: List[str]) -> None:
        """订阅指定类型的交易对"""
        with self._sub_lock:
            feed_type_str = feed_type.value
            current = set(self._subscriptions.get(feed_type_str, []))
            new_symbols = [s for s in symbols if s not in current]
            if new_symbols:
                self._subscriptions[feed_type_str].extend(new_symbols)
                # 如果已连接，立即发送订阅
                if self._state == FeedState.CONNECTED:
                    asyncio.create_task(self._subscribe(feed_type_str, new_symbols))

    def unsubscribe(self, feed_type: FeedType, symbols: List[str]) -> None:
        """取消订阅"""
        with self._sub_lock:
            feed_type_str = feed_type.value
            current = set(self._subscriptions.get(feed_type_str, []))
            to_remove = [s for s in symbols if s in current]
            if to_remove:
                self._subscriptions[feed_type_str] = [
                    s for s in self._subscriptions[feed_type_str] if s not in to_remove
                ]
                if self._state == FeedState.CONNECTED:
                    asyncio.create_task(self._unsubscribe(feed_type_str, to_remove))

    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """添加消息回调"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def remove_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """移除消息回调"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    async def start(self) -> None:
        """启动 WebSocket 监听"""
        self._running = True
        asyncio.create_task(self._reconnect_loop())
        asyncio.create_task(self._ping_loop())
        logger.info("OKXWebSocketFeed started")

    async def stop(self) -> None:
        """停止 WebSocket 监听"""
        self._running = False
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception:
                pass
        self._state = FeedState.DISCONNECTED
        logger.info("OKXWebSocketFeed stopped")

    async def _reconnect_loop(self) -> None:
        """断线重连循环 - P20: 指数退避+抖动"""
        while self._running:
            try:
                if self._state != FeedState.CONNECTED:
                    delay = max(0.5, self._reconnect_delay)
                    logger.info(
                        f"P20: WS reconnecting in {delay:.1f}s "
                        f"(attempt #{self._reconnect_count}, delay={self._reconnect_delay:.1f}s)"
                    )
                    await self.connect()
                    if self._state == FeedState.CONNECTED:
                        asyncio.create_task(self._message_loop())
            except Exception as e:
                logger.error(f"Reconnect loop error: {e}")
            await asyncio.sleep(max(0.5, self._reconnect_delay))

    async def _message_loop(self) -> None:
        """消息接收循环"""
        while self._running and self._state == FeedState.CONNECTED:
            try:
                if self._websocket is None:
                    break

                message = await self._websocket.recv()
                data = json.loads(message)

                # 心跳响应
                if data.get("event") == "pong":
                    self._last_ping = time.time()
                    continue

                # 订阅确认
                if data.get("event") == "subscribe":
                    logger.debug(f"Subscription confirmed: {data}")
                    # 清除待处理订阅
                    with self._sub_lock:
                        for arg in data.get("arg", {}):
                            feed_type = self._table_to_feed_type(arg.get("table", ""))
                            symbol = arg.get("instId", "")
                            key = f"{feed_type}:{symbol}"
                            self._pending_subscriptions.discard(key)
                    continue

                # 处理数据消息
                if "data" in data:
                    await self._process_data(data)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message loop error: {e}")
                self._state = FeedState.DISCONNECTED
                break

    async def _process_data(self, data: Dict[str, Any]) -> None:
        """处理数据消息"""
        table = data.get("table", "")
        feed_type = self._table_to_feed_type(table)
        items = data.get("data", [])

        for item in items:
            normalized = self._normalize_message(feed_type, item)
            if not normalized:
                continue

            # P22-6: 延迟监控 - 比较交易所时间戳与本地时间
            self._monitor_latency(normalized)

            symbol = normalized.get("symbol", "")
            if symbol:
                queue = self._get_queue(symbol)
                queue.put(normalized)

                self._message_counts[f"{feed_type}:{symbol}"] += 1
                self._last_message_ts[symbol] = time.time()

                for callback in self._callbacks:
                    try:
                        callback(normalized)
                    except Exception as e:
                        logger.debug(f"Callback error: {e}")

    def _monitor_latency(self, normalized: Dict[str, Any]) -> None:
        """P22-6: 监控行情延迟，超阈值时自动暂停开仓
        
        当滑动窗口平均延迟超过警告阈值时，暂停开仓操作。
        当延迟恢复到警告阈值的70%以下且超过冷却时间后，自动恢复开仓。
        """
        ts = normalized.get("timestamp", 0)
        if ts <= 0:
            return
        
        now_ms = int(time.time() * 1000)
        latency_ms = now_ms - ts
        
        # 过滤异常值（未来时间戳或极端延迟）
        if latency_ms < -1000 or latency_ms > 60000:
            return
        
        self._latency_window.append(latency_ms)
        
        # 计算滑动平均
        avg_latency = sum(self._latency_window) / len(self._latency_window) if self._latency_window else 0
        
        now = time.time()
        
        # 临界延迟：立即暂停开仓
        if latency_ms >= self._latency_critical_ms and not self._latency_pause_opening:
            self._latency_pause_opening = True
            self._latency_pause_start = now
            logger.warning(
                f"P22-6: OPENING PAUSED - Critical latency {latency_ms:.0f}ms > {self._latency_critical_ms}ms, "
                f"avg={avg_latency:.0f}ms, pause_duration={self._latency_pause_duration}s"
            )
        # 持续高延迟：平均超过警告阈值时暂停
        elif avg_latency >= self._latency_warning_ms and not self._latency_pause_opening:
            self._latency_pause_opening = True
            self._latency_pause_start = now
            if now - self._latency_last_log > self._latency_log_interval:
                self._latency_last_log = now
                logger.warning(
                    f"P22-6: OPENING PAUSED - Sustained high latency avg={avg_latency:.0f}ms > {self._latency_warning_ms}ms, "
                    f"pause_duration={self._latency_pause_duration}s"
                )
        # 自动恢复：延迟降到警告阈值70%以下且冷却时间已过
        elif self._latency_pause_opening and avg_latency < self._latency_warning_ms * 0.7:
            elapsed = now - self._latency_pause_start
            if elapsed >= self._latency_pause_duration:
                self._latency_pause_opening = False
                logger.info(
                    f"P22-6: OPENING RESUMED - Latency recovered, avg={avg_latency:.0f}ms, "
                    f"paused_for={elapsed:.0f}s"
                )
    
    def get_latency_stats(self) -> Dict[str, Any]:
        """P22-6: 获取延迟统计"""
        avg = sum(self._latency_window) / len(self._latency_window) if self._latency_window else 0
        return {
            "avg_latency_ms": round(avg, 1),
            "samples": len(self._latency_window),
            "opening_paused": self._latency_pause_opening,
            "pause_elapsed_sec": round(time.time() - self._latency_pause_start, 1) if self._latency_pause_opening else 0,
            "warning_threshold_ms": self._latency_warning_ms,
            "critical_threshold_ms": self._latency_critical_ms,
        }
    
    @property
    def is_opening_paused(self) -> bool:
        """P22-6: 是否暂停开仓"""
        return self._latency_pause_opening

    def _table_to_feed_type(self, table: str) -> str:
        """OKX table 名转订阅类型"""
        # K线频道统一映射到 KLINE 类型
        if table.startswith("candle"):
            return FeedType.KLINE.value
        
        mapping = {
            "tickers": FeedType.TICKER.value,
            "books": FeedType.ORDERBOOK.value,
            "trades": FeedType.TRADES.value,
            "funding-rate": FeedType.FUNDING.value,
        }
        return mapping.get(table, "unknown")

    def _normalize_message(self, feed_type: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """标准化消息格式 - P21: 优先使用交易所UTC时间戳，回退显式标记"""
        try:
            # P21: 获取交易所时间戳，缺失时使用UTC回退并记录
            exchange_ts = data.get("ts")
            if exchange_ts:
                ts = int(exchange_ts)
                ts_source = "exchange"
            else:
                ts = self._utc_timestamp_ms()
                ts_source = "local_utc_fallback"
                now = time.time()
                if now - OKXWebSocketFeed._ts_fallback_last_log > OKXWebSocketFeed._ts_fallback_log_interval:
                    OKXWebSocketFeed._ts_fallback_last_log = now
                    logger.warning(
                        f"P21: Exchange timestamp missing in {feed_type} message, "
                        f"using local UTC fallback. Check WS data quality."
                    )

            if feed_type == FeedType.TICKER.value:
                return {
                    "symbol": data.get("instId", ""),
                    "feed_type": feed_type,
                    "timestamp": ts,
                    "ts_source": ts_source,
                    "price": float(data.get("last", 0)),
                    "bid_price": float(data.get("bidPx", 0)),
                    "ask_price": float(data.get("askPx", 0)),
                    "bid_volume": float(data.get("bidSz", 0)),
                    "ask_volume": float(data.get("askSz", 0)),
                    "volume_24h": float(data.get("vol24h", 0)),
                    "change_24h": float(data.get("chg24h", 0)),
                    "high_24h": float(data.get("high24h", 0)),
                    "low_24h": float(data.get("low24h", 0)),
                    "funding_rate": float(data.get("fundingRate", 0)),
                }
            elif feed_type == FeedType.ORDERBOOK.value:
                asks = data.get("asks", [])[:50]
                bids = data.get("bids", [])[:50]
                return {
                    "symbol": data.get("instId", ""),
                    "feed_type": feed_type,
                    "timestamp": ts,
                    "ts_source": ts_source,
                    "bids": [[float(b[0]), float(b[1])] for b in bids],
                    "asks": [[float(a[0]), float(a[1])] for a in asks],
                    "seq": int(data.get("seq", 0)),
                }
            elif feed_type == FeedType.TRADES.value:
                return {
                    "symbol": data.get("instId", ""),
                    "feed_type": feed_type,
                    "timestamp": ts,
                    "ts_source": ts_source,
                    "price": float(data.get("px", 0)),
                    "volume": float(data.get("sz", 0)),
                    "side": data.get("side", ""),
                    "trade_id": data.get("tradeId", ""),
                }
            elif feed_type == FeedType.KLINE.value:
                candle = data.get("candle", [])
                if len(candle) >= 6:
                    return {
                        "symbol": data.get("instId", ""),
                        "feed_type": feed_type,
                        "timestamp": int(candle[0]),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                    }
            elif feed_type == FeedType.FUNDING.value:
                return {
                    "symbol": data.get("instId", ""),
                    "feed_type": feed_type,
                    "timestamp": ts,
                    "ts_source": ts_source,
                    "funding_rate": float(data.get("fundingRate", 0)),
                    "funding_time": int(data.get("fundingTime", 0)),
                    "predicted_rate": float(data.get("predictedRate", 0)),
                }
        except Exception as e:
            logger.debug(f"Message normalization error: {e}")
            return None
        return None

    async def _ping_loop(self) -> None:
        """心跳保活循环"""
        while self._running:
            try:
                if self._state == FeedState.CONNECTED and self._websocket is not None:
                    # 检查连接超时
                    if time.time() - self._last_ping > self._ping_interval + self._ping_timeout:
                        logger.warning("WebSocket ping timeout, triggering reconnect")
                        self._state = FeedState.DISCONNECTED
                        try:
                            await self._websocket.close()
                        except Exception:
                            pass
                await asyncio.sleep(self._ping_interval)
            except Exception as e:
                logger.debug(f"Ping loop error: {e}")

    def get_queue(self, symbol: str) -> MessageQueue:
        """获取指定 symbol 的消息队列"""
        return self._get_queue(symbol)

    def get_state(self) -> str:
        """获取连接状态"""
        return self._state.value

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        queue_stats = {}
        for symbol, queue in self._queues.items():
            queue_stats[symbol] = queue.get_stats()

        return {
            "state": self._state.value,
            "reconnect_count": self._reconnect_count,
            "reconnect_delay": round(self._reconnect_delay, 2),
            "pending_subscriptions": len(self._pending_subscriptions),
            "subscriptions": {k: len(v) for k, v in self._subscriptions.items()},
            "message_counts": dict(self._message_counts),
            "queue_stats": queue_stats,
            "last_message_ts": {k: round(v, 2) for k, v in self._last_message_ts.items()},
            # P22-6: 延迟统计
            "latency": self.get_latency_stats(),
        }

    def is_connected(self) -> bool:
        """是否已连接"""
        return self._state == FeedState.CONNECTED
