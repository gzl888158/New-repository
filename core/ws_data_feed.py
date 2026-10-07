"""
[DEPRECATED] 此模块已被 market_data/websocket_feed.py 替代，仅供向后兼容。
生产代码请使用 market_data.OKXWebSocketFeed。

生产级 WebSocket 实时数据推送管理器
========================================
核心定位：统一管理OKX WebSocket连接，提供生产级实时数据流

功能：
- 双通道管理（公共频道 + 私有频道）
- 多频道订阅（tickers / orderbooks / trades / positions / orders / account）
- 自动重连（指数退避 + 随机抖动）
- 心跳保活（Ping/Pong + 超时检测）
- 连接健康监控（延迟/丢包/重连次数）
- 线程安全数据缓冲（生产者-消费者模式）
- 数据回调分发（tick / orderbook / trade / position / order / account）
- 优雅关闭（等待数据处理完成）

架构：
  WebSocketManager
  ├── PublicChannel (tickers, books, trades, mark-price)
  │   ├── 指数退避重连
  │   ├── 心跳检测
  │   └── 数据缓冲队列
  └── PrivateChannel (positions, orders, account, balance)
      ├── 登录认证
      ├── 指数退避重连
      ├── 心跳检测
      └── 数据缓冲队列
"""

import asyncio
import json
import time
import hmac
import hashlib
import base64
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from loguru import logger

try:
    import websockets
    from websockets.exceptions import ConnectionClosed, WebSocketException
except ImportError:
    websockets = None

# websockets v13+ 兼容
try:
    from websockets.protocol import State as _ws_State
    _WS_OPEN_STATE = _ws_State.OPEN
except ImportError:
    _WS_OPEN_STATE = None


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class ChannelType(Enum):
    """频道类型"""
    TICKERS = "tickers"
    BOOKS = "books"
    TRADES = "trades"
    MARK_PRICE = "mark-price"
    POSITIONS = "positions"
    ORDERS = "orders"
    ACCOUNT = "account"
    BALANCE = "balance"


class ConnectionState(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class WSConnectionStats:
    """WebSocket连接统计"""
    connected_at: float = 0.0
    last_ping_at: float = 0.0
    last_pong_at: float = 0.0
    last_data_at: float = 0.0
    messages_received: int = 0
    messages_sent: int = 0
    reconnect_count: int = 0
    total_disconnects: int = 0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    current_state: ConnectionState = ConnectionState.DISCONNECTED


@dataclass
class DataBuffer:
    """线程安全数据缓冲器"""
    max_size: int = 10000
    _deque: deque = field(default_factory=deque)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, item: Any) -> bool:
        """推入数据，超限时丢弃最旧数据"""
        with self._lock:
            if len(self._deque) >= self.max_size:
                self._deque.popleft()
            self._deque.append(item)
            return True

    def pop_all(self) -> List[Any]:
        """取出所有数据"""
        with self._lock:
            items = list(self._deque)
            self._deque.clear()
            return items

    def pop_batch(self, max_count: int) -> List[Any]:
        """取出指定数量的数据"""
        with self._lock:
            count = min(max_count, len(self._deque))
            items = [self._deque.popleft() for _ in range(count)]
            return items

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._deque)

    def clear(self):
        with self._lock:
            self._deque.clear()


# ═══════════════════════════════════════════════════════════════
# 回调类型定义
# ═══════════════════════════════════════════════════════════════

TickCallback = Callable[[Dict[str, Any]], None]
OrderbookCallback = Callable[[Dict[str, Any]], None]
TradeCallback = Callable[[Dict[str, Any]], None]
PositionCallback = Callable[[List[Dict[str, Any]]], None]
OrderCallback = Callable[[Dict[str, Any]], None]
AccountCallback = Callable[[Dict[str, Any]], None]
ConnectionCallback = Callable[[str, ConnectionState, ConnectionState], None]  # channel, old, new


# ═══════════════════════════════════════════════════════════════
# 生产级 WebSocket 管理器
# ═══════════════════════════════════════════════════════════════

class WebSocketManager:
    """
    生产级 WebSocket 实时数据推送管理器

    使用示例:
        mgr = WebSocketManager(config, okx_client)
        mgr.on_tick = lambda data: print(f"Tick: {data}")
        mgr.on_position = lambda data: print(f"Position: {data}")
        await mgr.subscribe_tickers(["BTC-USDT", "ETH-USDT"])
        await mgr.subscribe_books(["BTC-USDT"], depth=5)
        await mgr.start()
        # ... 运行中 ...
        await mgr.stop()
    """

    # ─── 默认配置 ───
    DEFAULT_PING_INTERVAL = 25          # OKX要求30秒内发送ping
    DEFAULT_PONG_TIMEOUT = 15           # 15秒无pong触发重连
    DEFAULT_DATA_TIMEOUT = 60           # 60秒无数据触发重连
    DEFAULT_RECONNECT_BASE = 1.0        # 重连基础等待(秒)
    DEFAULT_RECONNECT_MAX = 60.0        # 重连最大等待(秒)
    DEFAULT_RECONNECT_JITTER = 0.3      # 重连随机抖动比例
    DEFAULT_BUFFER_SIZE = 10000         # 数据缓冲大小

    def __init__(self, config: Dict[str, Any], okx_client=None):
        self.config = config
        self._okx_client = okx_client

        okx_cfg = config.get("okx", {})
        self._ws_public_url = okx_cfg.get("websocket_url", "wss://ws.okx.com:8443/ws/v5/public")
        self._ws_private_url = okx_cfg.get("websocket_private_url", "wss://ws.okx.com:8443/ws/v5/private")
        self._is_testnet = okx_cfg.get("is_testnet", False)

        if self._is_testnet:
            self._ws_public_url = "wss://wspap.okx.com:8443/ws/v5/public?brokerId=9999"
            self._ws_private_url = "wss://wspap.okx.com:8443/ws/v5/private?brokerId=9999"

        # ─── 连接状态 ───
        self._public_state = ConnectionState.DISCONNECTED
        self._private_state = ConnectionState.DISCONNECTED
        self._public_stats = WSConnectionStats()
        self._private_stats = WSConnectionStats()
        self._public_ws = None
        self._private_ws = None
        self._public_logged_in = False

        # ─── 订阅管理 ───
        self._public_subscriptions: Dict[str, Set[str]] = {
            "tickers": set(),
            "books5": set(),
            "books": set(),
            "trades": set(),
            "mark-price": set(),
        }
        self._private_subscriptions: Set[str] = set()
        self._subscription_lock = threading.Lock()

        # ─── 数据缓冲 ───
        self._tick_buffer = DataBuffer(max_size=self.DEFAULT_BUFFER_SIZE)
        self._book_buffer = DataBuffer(max_size=self.DEFAULT_BUFFER_SIZE)
        self._book_sequence_ids: Dict[str, int] = {}
        self._book_resync_pending: Set[str] = set()
        self._trade_buffer = DataBuffer(max_size=self.DEFAULT_BUFFER_SIZE)
        self._position_buffer = DataBuffer(max_size=1000)
        self._order_buffer = DataBuffer(max_size=5000)
        self._account_buffer = DataBuffer(max_size=1000)

        # ─── 回调注册 ───
        self._tick_callbacks: List[TickCallback] = []
        self._book_callbacks: List[OrderbookCallback] = []
        self._trade_callbacks: List[TradeCallback] = []
        self._position_callbacks: List[PositionCallback] = []
        self._order_callbacks: List[OrderCallback] = []
        self._account_callbacks: List[AccountCallback] = []
        self._connection_callbacks: List[ConnectionCallback] = []
        self._callback_lock = threading.Lock()

        # ─── 运行控制 ───
        self._running = False
        self._public_task: Optional[asyncio.Task] = None
        self._private_task: Optional[asyncio.Task] = None
        self._process_task: Optional[asyncio.Task] = None
        self._ping_tasks: List[asyncio.Task] = []

        # ─── 重连控制 ───
        self._reconnect_lock_public = asyncio.Lock()
        self._reconnect_lock_private = asyncio.Lock()

        # ─── 认证信息 ───
        if okx_client:
            self._api_key = getattr(okx_client, 'api_key', '')
            self._secret_key = getattr(okx_client, 'secret_key', '')
            self._passphrase = getattr(okx_client, 'passphrase', '')
        else:
            self._api_key = okx_cfg.get("api_key", "")
            self._secret_key = okx_cfg.get("secret_key", "")
            self._passphrase = okx_cfg.get("passphrase", "")

        logger.info(
            f"WebSocketManager initialized: public={self._ws_public_url}, "
            f"private={self._ws_private_url}, testnet={self._is_testnet}"
        )

    # ═══════════════════════════════════════════════════════════════
    # 回调注册
    # ═══════════════════════════════════════════════════════════════

    def on_tick(self, callback: TickCallback):
        """注册Tick数据回调"""
        with self._callback_lock:
            self._tick_callbacks.append(callback)

    def on_orderbook(self, callback: OrderbookCallback):
        """注册订单簿数据回调"""
        with self._callback_lock:
            self._book_callbacks.append(callback)

    def on_trade(self, callback: TradeCallback):
        """注册成交数据回调"""
        with self._callback_lock:
            self._trade_callbacks.append(callback)

    def on_position(self, callback: PositionCallback):
        """注册持仓数据回调"""
        with self._callback_lock:
            self._position_callbacks.append(callback)

    def on_order(self, callback: OrderCallback):
        """注册订单数据回调"""
        with self._callback_lock:
            self._order_callbacks.append(callback)

    def on_account(self, callback: AccountCallback):
        """注册账户数据回调"""
        with self._callback_lock:
            self._account_callbacks.append(callback)

    def on_connection_change(self, callback: ConnectionCallback):
        """注册连接状态变化回调 (channel, old_state, new_state)"""
        with self._callback_lock:
            self._connection_callbacks.append(callback)

    def remove_callback(self, callback: Callable):
        """移除回调"""
        with self._callback_lock:
            for lst in [self._tick_callbacks, self._book_callbacks,
                        self._trade_callbacks, self._position_callbacks,
                        self._order_callbacks, self._account_callbacks,
                        self._connection_callbacks]:
                if callback in lst:
                    lst.remove(callback)

    # ═══════════════════════════════════════════════════════════════
    # 订阅管理
    # ═══════════════════════════════════════════════════════════════

    def subscribe_tickers(self, symbols: List[str]) -> None:
        """订阅Ticker行情"""
        with self._subscription_lock:
            self._public_subscriptions["tickers"].update(symbols)
        logger.info(f"Subscribed tickers: {symbols}")

    def subscribe_books(self, symbols: List[str], depth: int = 5) -> None:
        """订阅订单簿深度"""
        key = f"books{depth}" if depth > 5 else "books5"
        if key not in self._public_subscriptions:
            self._public_subscriptions[key] = set()
        with self._subscription_lock:
            self._public_subscriptions[key].update(symbols)
        logger.info(f"Subscribed books (depth={depth}): {symbols}")

    def subscribe_trades(self, symbols: List[str]) -> None:
        """订阅逐笔成交"""
        with self._subscription_lock:
            self._public_subscriptions["trades"].update(symbols)
        logger.info(f"Subscribed trades: {symbols}")

    def subscribe_mark_price(self, symbols: List[str]) -> None:
        """订阅标记价格"""
        with self._subscription_lock:
            self._public_subscriptions["mark-price"].update(symbols)
        logger.info(f"Subscribed mark-price: {symbols}")

    def subscribe_private_channels(self, channels: List[str]) -> None:
        """订阅私有频道 (positions, orders, account, balance)"""
        with self._subscription_lock:
            self._private_subscriptions.update(channels)
        logger.info(f"Subscribed private channels: {channels}")

    def unsubscribe_tickers(self, symbols: List[str]) -> None:
        """取消订阅Ticker"""
        with self._subscription_lock:
            self._public_subscriptions["tickers"].difference_update(symbols)

    def unsubscribe_all(self) -> None:
        """取消所有订阅"""
        with self._subscription_lock:
            for key in self._public_subscriptions:
                self._public_subscriptions[key].clear()
            self._private_subscriptions.clear()

    # ═══════════════════════════════════════════════════════════════
    # 启动/停止
    # ═══════════════════════════════════════════════════════════════

    async def start(self) -> None:
        """启动WebSocket管理器"""
        if self._running:
            logger.warning("WebSocketManager already running")
            return

        self._running = True
        logger.info("WebSocketManager starting...")

        # 启动公共频道
        self._public_task = asyncio.create_task(self._run_public_channel())
        self._public_task.set_name("ws_public")

        # 启动私有频道（如果有认证信息）
        if self._api_key and self._secret_key:
            self._private_task = asyncio.create_task(self._run_private_channel())
            self._private_task.set_name("ws_private")

        # 启动数据处理协程
        self._process_task = asyncio.create_task(self._process_data_loop())
        self._process_task.set_name("ws_data_processor")

        logger.info("WebSocketManager started")

    async def stop(self) -> None:
        """优雅关闭WebSocket管理器"""
        logger.info("WebSocketManager stopping...")
        self._running = False

        # 关闭连接
        await self._close_connection("public")
        await self._close_connection("private")

        # 取消所有任务
        for task in [self._public_task, self._private_task, self._process_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 清空缓冲区
        self._tick_buffer.clear()
        self._book_buffer.clear()
        self._trade_buffer.clear()

        logger.info("WebSocketManager stopped")

    # ═══════════════════════════════════════════════════════════════
    # 公共频道主循环
    # ═══════════════════════════════════════════════════════════════

    async def _run_public_channel(self) -> None:
        """公共频道主循环：连接 → 订阅 → 接收 → 心跳"""
        while self._running:
            try:
                await self._connect_public()
                await self._subscribe_public_channels()
                await self._receive_public_loop()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Public channel error: {e}")
                await self._reconnect_delay("public")
            finally:
                await self._close_connection("public")

    async def _connect_public(self) -> None:
        """连接公共频道WebSocket"""
        self._update_connection_state("public", ConnectionState.CONNECTING)

        try:
            self._public_ws = await websockets.connect(
                self._ws_public_url,
                ping_interval=None,  # 手动管理心跳
                ping_timeout=None,
                close_timeout=5,
                max_size=2 ** 23,  # 8MB
            )
            self._public_stats.connected_at = time.time()
            self._public_stats.last_data_at = time.time()
            self._update_connection_state("public", ConnectionState.CONNECTED)
            logger.info("Public channel connected")
        except Exception as e:
            self._update_connection_state("public", ConnectionState.DISCONNECTED)
            logger.error(f"Public channel connection failed: {e}")
            raise

    async def _subscribe_public_channels(self) -> None:
        """发送公共频道订阅请求"""
        if not self._public_ws:
            return

        subscribe_args = []

        # P1-1: 异步上下文移除 threading.Lock 避免阻塞事件循环（仅读取快照，GIL 保证安全）
        # Tickers
        tickers = self._public_subscriptions.get("tickers", set())
        for symbol in tickers:
            subscribe_args.append({
                "channel": "tickers",
                "instId": symbol,
            })

        # Orderbooks
        for depth_key in ["books5", "books"]:
            symbols = self._public_subscriptions.get(depth_key, set())
            if symbols:
                depth = 5 if depth_key == "books5" else 50
                channel = "books5" if depth_key == "books5" else "books"
                for symbol in symbols:
                    subscribe_args.append({
                        "channel": channel,
                        "instId": symbol,
                    })

            # Trades
            trades = self._public_subscriptions.get("trades", set())
            for symbol in trades:
                subscribe_args.append({
                    "channel": "trades",
                    "instId": symbol,
                })

            # Mark price
            mark_prices = self._public_subscriptions.get("mark-price", set())
            for symbol in mark_prices:
                subscribe_args.append({
                    "channel": "mark-price",
                    "instId": symbol,
                })

        if subscribe_args:
            request = {
                "op": "subscribe",
                "args": subscribe_args,
            }
            await self._public_ws.send(json.dumps(request))
            self._public_stats.messages_sent += 1
            logger.info(f"Public channels subscribed: {len(subscribe_args)} channels")

    async def _receive_public_loop(self) -> None:
        """公共频道接收循环 + 心跳"""
        ping_task = asyncio.create_task(self._ping_loop("public"))
        self._ping_tasks.append(ping_task)

        try:
            async for message in self._public_ws:
                self._public_stats.messages_received += 1
                self._public_stats.last_data_at = time.time()

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                # 处理pong响应
                if "event" in data and data.get("event") == "pong":
                    self._public_stats.last_pong_at = time.time()
                    continue

                # 处理订阅确认
                if "event" in data and data.get("event") == "subscribe":
                    logger.debug(f"Public subscribe confirmed: {data.get('arg', {})}")
                    continue

                # 路由数据到缓冲区
                await self._route_public_data(data)

        except (ConnectionClosed, WebSocketException) as e:
            logger.warning(f"Public channel disconnected: {e}")
            self._public_stats.total_disconnects += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Public channel receive error: {e}")
        finally:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

    async def _route_public_data(self, data: Dict[str, Any]) -> None:
        """路由公共频道数据到对应缓冲区"""
        arg = data.get("arg", {})
        channel = arg.get("channel", "")
        data_list = data.get("data", [])

        if not data_list:
            return

        if channel == "tickers":
            for item in data_list:
                self._tick_buffer.push(item)
        elif channel == "books5":
            for item in data_list:
                self._book_buffer.push({**item, "_channel": channel, "_sequence_valid": None})
        elif channel == "books":
            for item in data_list:
                await self._route_sequenced_book(data, item)
        elif channel == "trades":
            for item in data_list:
                self._trade_buffer.push(item)
        elif channel == "mark-price":
            for item in data_list:
                self._tick_buffer.push({**item, "_channel": "mark-price"})

    async def _route_sequenced_book(self, envelope: Dict[str, Any], item: Dict[str, Any]) -> None:
        symbol = str(item.get("instId") or envelope.get("arg", {}).get("instId") or "")
        if not symbol:
            return

        action = envelope.get("action")
        if action == "snapshot":
            try:
                sequence_id = int(item["seqId"])
            except (KeyError, TypeError, ValueError):
                await self._resync_orderbook(symbol, "snapshot_missing_seq_id")
                return
            self._book_sequence_ids[symbol] = sequence_id
            self._book_resync_pending.discard(symbol)
            self._book_buffer.push({
                **item,
                "_channel": "books",
                "_sequence_valid": True,
                "_source": "websocket",
            })
            return

        if action != "update" or symbol in self._book_resync_pending:
            return

        try:
            previous_sequence_id = int(item["prevSeqId"])
            sequence_id = int(item["seqId"])
        except (KeyError, TypeError, ValueError):
            await self._resync_orderbook(symbol, "update_missing_sequence")
            return

        last_sequence_id = self._book_sequence_ids.get(symbol)
        if last_sequence_id is None or previous_sequence_id != last_sequence_id or sequence_id <= last_sequence_id:
            await self._resync_orderbook(symbol, "sequence_gap")
            return

        self._book_sequence_ids[symbol] = sequence_id
        self._book_buffer.push({
            **item,
            "_channel": "books",
            "_sequence_valid": True,
            "_source": "websocket",
        })

    async def _resync_orderbook(self, symbol: str, reason: str) -> None:
        """Invalidate a gapped stream, publish a marked REST fallback, and await a new WS snapshot."""
        self._book_sequence_ids.pop(symbol, None)
        already_pending = symbol in self._book_resync_pending
        self._book_resync_pending.add(symbol)
        if already_pending:
            return

        rest_book = None
        get_order_book = getattr(self._okx_client, "get_order_book", None)
        if callable(get_order_book):
            try:
                rest_book = await asyncio.to_thread(get_order_book, symbol, 50)
            except Exception as exc:
                logger.warning(f"Orderbook REST resync failed for {symbol}: {exc}")

        if isinstance(rest_book, dict) and rest_book.get("bids") and rest_book.get("asks"):
            self._book_buffer.push({
                **rest_book,
                "instId": symbol,
                "action": "snapshot",
                "_channel": "books",
                "_sequence_valid": False,
                "_source": "rest_fallback",
                "_resync_reason": reason,
            })
        else:
            self._book_buffer.push({
                "instId": symbol,
                "action": "invalidated",
                "_channel": "books",
                "_sequence_valid": False,
                "_source": "websocket",
                "_resync_reason": reason,
            })

        if self._public_ws is not None:
            subscribe_arg = {"channel": "books", "instId": symbol}
            try:
                await self._public_ws.send(json.dumps({"op": "unsubscribe", "args": [subscribe_arg]}))
                await self._public_ws.send(json.dumps({"op": "subscribe", "args": [subscribe_arg]}))
            except Exception as exc:
                logger.warning(f"Orderbook resubscribe failed for {symbol}: {exc}")

    # ═══════════════════════════════════════════════════════════════
    # 私有频道主循环
    # ═══════════════════════════════════════════════════════════════

    async def _run_private_channel(self) -> None:
        """私有频道主循环：连接 → 登录 → 订阅 → 接收 → 心跳"""
        while self._running:
            try:
                await self._connect_private()
                await self._login_private()
                await self._subscribe_private_channels()
                await self._receive_private_loop()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Private channel error: {e}")
                await self._reconnect_delay("private")
            finally:
                await self._close_connection("private")

    async def _connect_private(self) -> None:
        """连接私有频道WebSocket"""
        self._update_connection_state("private", ConnectionState.CONNECTING)

        try:
            self._private_ws = await websockets.connect(
                self._ws_private_url,
                ping_interval=None,
                ping_timeout=None,
                close_timeout=5,
                max_size=2 ** 23,
            )
            self._private_stats.connected_at = time.time()
            self._private_stats.last_data_at = time.time()
            self._update_connection_state("private", ConnectionState.CONNECTED)
            logger.info("Private channel connected")
        except Exception as e:
            self._update_connection_state("private", ConnectionState.DISCONNECTED)
            logger.error(f"Private channel connection failed: {e}")
            raise

    async def _login_private(self) -> None:
        """私有频道登录认证"""
        if not self._private_ws:
            return

        timestamp = str(int(time.time()))
        sign_str = f"{timestamp}GET/users/self/verify"
        signature = base64.b64encode(
            hmac.new(
                self._secret_key.encode(),
                sign_str.encode(),
                hashlib.sha256,
            ).digest()
        ).decode()

        login_msg = {
            "op": "login",
            "args": [{
                "apiKey": self._api_key,
                "passphrase": self._passphrase,
                "timestamp": timestamp,
                "sign": signature,
            }]
        }
        await self._private_ws.send(json.dumps(login_msg))
        self._private_stats.messages_sent += 1

        # 等待登录响应
        try:
            response = await asyncio.wait_for(self._private_ws.recv(), timeout=10)
            data = json.loads(response)
            if data.get("event") == "login" and data.get("code") == "0":
                self._public_logged_in = True
                logger.info("Private channel logged in")
            else:
                logger.error(f"Private login failed: {data}")
                raise ConnectionError(f"Login failed: {data.get('msg', 'Unknown')}")
        except asyncio.TimeoutError:
            logger.error("Private login timeout")
            raise ConnectionError("Login timeout")

    async def _subscribe_private_channels(self) -> None:
        """发送私有频道订阅请求"""
        if not self._private_ws:
            return

        subscribe_args = []
        # P1-1: 异步上下文移除 threading.Lock 避免阻塞事件循环（仅读取快照，GIL 保证安全）
        for channel in self._private_subscriptions:
            subscribe_args.append({"channel": channel, "instType": "ANY"})

        if subscribe_args:
            request = {"op": "subscribe", "args": subscribe_args}
            await self._private_ws.send(json.dumps(request))
            self._private_stats.messages_sent += 1
            logger.info(f"Private channels subscribed: {subscribe_args}")

    async def _receive_private_loop(self) -> None:
        """私有频道接收循环 + 心跳"""
        ping_task = asyncio.create_task(self._ping_loop("private"))
        self._ping_tasks.append(ping_task)

        try:
            async for message in self._private_ws:
                self._private_stats.messages_received += 1
                self._private_stats.last_data_at = time.time()

                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                if "event" in data and data.get("event") == "pong":
                    self._private_stats.last_pong_at = time.time()
                    continue

                if "event" in data:
                    continue

                await self._route_private_data(data)

        except (ConnectionClosed, WebSocketException) as e:
            logger.warning(f"Private channel disconnected: {e}")
            self._private_stats.total_disconnects += 1
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Private channel receive error: {e}")
        finally:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass

    async def _route_private_data(self, data: Dict[str, Any]) -> None:
        """路由私有频道数据到对应缓冲区"""
        arg = data.get("arg", {})
        channel = arg.get("channel", "")
        data_list = data.get("data", [])

        if not data_list:
            return

        if channel == "positions":
            self._position_buffer.push(data_list)
        elif channel == "orders":
            for item in data_list:
                self._order_buffer.push(item)
        elif channel == "account":
            for item in data_list:
                self._account_buffer.push(item)
        elif channel == "balance_and_position":
            for item in data_list:
                self._account_buffer.push({**item, "_channel": "balance"})

    # ═══════════════════════════════════════════════════════════════
    # 数据处理循环
    # ═══════════════════════════════════════════════════════════════

    async def _process_data_loop(self) -> None:
        """数据处理协程：从缓冲区读取并分发到回调"""
        while self._running:
            try:
                # 处理Tick数据
                ticks = self._tick_buffer.pop_batch(100)
                if ticks:
                    # P1-1: 异步上下文移除 threading.Lock 避免阻塞事件循环（仅读取快照，GIL 保证安全）
                    callbacks = list(self._tick_callbacks)
                    for tick in ticks:
                        for cb in callbacks:
                            try:
                                cb(tick)
                            except Exception as e:
                                logger.warning(f"Tick callback {cb.__name__} failed: {e}")

                # 处理订单簿数据
                books = self._book_buffer.pop_batch(50)
                if books:
                    callbacks = list(self._book_callbacks)
                    for book in books:
                        for cb in callbacks:
                            try:
                                cb(book)
                            except Exception as e:
                                logger.warning(f"OrderBook callback {cb.__name__} failed: {e}")

                # 处理成交数据
                trades = self._trade_buffer.pop_batch(50)
                if trades:
                    callbacks = list(self._trade_callbacks)
                    for trade in trades:
                        for cb in callbacks:
                            try:
                                cb(trade)
                            except Exception as e:
                                logger.warning(f"Trade callback {cb.__name__} failed: {e}")

                # 处理持仓数据
                positions = self._position_buffer.pop_all()
                if positions:
                    callbacks = list(self._position_callbacks)
                    for cb in callbacks:
                        try:
                            cb(positions)
                        except Exception as e:
                            logger.warning(f"Position callback {cb.__name__} failed: {e}")

                # 处理订单数据
                orders = self._order_buffer.pop_batch(50)
                if orders:
                    callbacks = list(self._order_callbacks)
                    for order in orders:
                        for cb in callbacks:
                            try:
                                cb(order)
                            except Exception as e:
                                logger.warning(f"Order callback {cb.__name__} failed: {e}")

                # 处理账户数据
                accounts = self._account_buffer.pop_all()
                if accounts:
                    callbacks = list(self._account_callbacks)
                    for cb in callbacks:
                        try:
                            cb(accounts)
                        except Exception as e:
                            logger.warning(f"Account callback {cb.__name__} failed: {e}")

                await asyncio.sleep(0.01)  # 10ms处理间隔，避免CPU空转

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Data processor error: {e}")
                await asyncio.sleep(0.1)

    # ═══════════════════════════════════════════════════════════════
    # 心跳管理
    # ═══════════════════════════════════════════════════════════════

    async def _ping_loop(self, channel: str) -> None:
        """心跳循环：定期发送ping，检测pong超时"""
        ws_attr = "_public_ws" if channel == "public" else "_private_ws"
        stats_attr = "_public_stats" if channel == "public" else "_private_stats"

        while self._running:
            try:
                await asyncio.sleep(self.DEFAULT_PING_INTERVAL)

                ws = getattr(self, ws_attr)
                if ws is None:
                    continue

                stats = getattr(self, stats_attr)

                # 检查连接是否存活
                try:
                    open_check = ws.state == _WS_OPEN_STATE if _WS_OPEN_STATE else ws.open
                except Exception:
                    open_check = False

                if not open_check:
                    logger.warning(f"{channel} channel not open, triggering reconnect")
                    break

                # 发送ping
                ping_msg = json.dumps({"op": "ping"})
                await ws.send(ping_msg)
                stats.messages_sent += 1
                stats.last_ping_at = time.time()

                # 检查pong超时
                if stats.last_pong_at > 0:
                    elapsed = time.time() - stats.last_pong_at
                    if elapsed > self.DEFAULT_PONG_TIMEOUT + self.DEFAULT_PING_INTERVAL:
                        logger.warning(f"{channel} pong timeout ({elapsed:.1f}s), reconnecting")
                        break

                # 检查数据超时
                if stats.last_data_at > 0:
                    elapsed = time.time() - stats.last_data_at
                    if elapsed > self.DEFAULT_DATA_TIMEOUT:
                        logger.warning(f"{channel} data timeout ({elapsed:.1f}s), reconnecting")
                        break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{channel} ping error: {e}")
                break

    # ═══════════════════════════════════════════════════════════════
    # 重连管理
    # ═══════════════════════════════════════════════════════════════

    async def _reconnect_delay(self, channel: str) -> None:
        """指数退避重连延迟"""
        stats = self._public_stats if channel == "public" else self._private_stats
        stats.reconnect_count += 1

        base = self.DEFAULT_RECONNECT_BASE
        backoff = min(
            self.DEFAULT_RECONNECT_MAX,
            base * (2 ** min(stats.reconnect_count - 1, 6))
        )
        # 随机抖动
        import random
        jitter = backoff * self.DEFAULT_RECONNECT_JITTER * (random.random() * 2 - 1)
        delay = backoff + jitter

        self._update_connection_state(channel, ConnectionState.RECONNECTING)

        logger.info(
            f"{channel} channel reconnecting in {delay:.1f}s "
            f"(attempt #{stats.reconnect_count})"
        )

        await asyncio.sleep(max(0.5, delay))

    async def _close_connection(self, channel: str) -> None:
        """关闭WebSocket连接"""
        ws_attr = "_public_ws" if channel == "public" else "_private_ws"
        ws = getattr(self, ws_attr)

        if ws:
            try:
                open_check = ws.state == _WS_OPEN_STATE if _WS_OPEN_STATE else ws.open
            except Exception:
                open_check = False

            if open_check:
                try:
                    await ws.close()
                except Exception:
                    pass

            setattr(self, ws_attr, None)

        self._update_connection_state(channel, ConnectionState.DISCONNECTED)

    def _update_connection_state(self, channel: str, new_state: ConnectionState) -> None:
        """更新连接状态并通知回调"""
        old_state = self._public_state if channel == "public" else self._private_state

        if channel == "public":
            self._public_state = new_state
            self._public_stats.current_state = new_state
        else:
            self._private_state = new_state
            self._private_stats.current_state = new_state

        if old_state != new_state:
            logger.debug(f"{channel} channel: {old_state.value} → {new_state.value}")
            with self._callback_lock:
                for cb in self._connection_callbacks:
                    try:
                        cb(channel, old_state, new_state)
                    except Exception:
                        pass

    # ═══════════════════════════════════════════════════════════════
    # 状态查询
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """获取连接统计"""
        return {
            "public": {
                "state": self._public_state.value,
                "connected_at": datetime.fromtimestamp(self._public_stats.connected_at).isoformat() if self._public_stats.connected_at > 0 else "",
                "messages_received": self._public_stats.messages_received,
                "messages_sent": self._public_stats.messages_sent,
                "reconnect_count": self._public_stats.reconnect_count,
                "total_disconnects": self._public_stats.total_disconnects,
                "last_pong_ago": time.time() - self._public_stats.last_pong_at if self._public_stats.last_pong_at > 0 else -1,
                "last_data_ago": time.time() - self._public_stats.last_data_at if self._public_stats.last_data_at > 0 else -1,
                "buffer_sizes": {
                    "tick": self._tick_buffer.size,
                    "book": self._book_buffer.size,
                    "trade": self._trade_buffer.size,
                },
            },
            "private": {
                "state": self._private_state.value,
                "connected_at": datetime.fromtimestamp(self._private_stats.connected_at).isoformat() if self._private_stats.connected_at > 0 else "",
                "messages_received": self._private_stats.messages_received,
                "messages_sent": self._private_stats.messages_sent,
                "reconnect_count": self._private_stats.reconnect_count,
                "total_disconnects": self._private_stats.total_disconnects,
                "last_pong_ago": time.time() - self._private_stats.last_pong_at if self._private_stats.last_pong_at > 0 else -1,
                "last_data_ago": time.time() - self._private_stats.last_data_at if self._private_stats.last_data_at > 0 else -1,
                "buffer_sizes": {
                    "position": self._position_buffer.size,
                    "order": self._order_buffer.size,
                    "account": self._account_buffer.size,
                },
            },
            "timestamp": datetime.now().isoformat(),
        }

    def is_healthy(self) -> bool:
        """检查连接健康状态"""
        public_healthy = (
            self._public_state == ConnectionState.CONNECTED
            and (time.time() - self._public_stats.last_data_at) < self.DEFAULT_DATA_TIMEOUT
        )
        private_healthy = (
            self._private_state == ConnectionState.CONNECTED
            and (time.time() - self._private_stats.last_data_at) < self.DEFAULT_DATA_TIMEOUT
        ) if self._api_key else True  # 无认证时跳过私有频道检查

        return public_healthy and private_healthy