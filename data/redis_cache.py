"""
Redis 缓存服务，缓存行情、持仓与账户数据并在 Redis 不可用时降级为内存缓存。
"""
import json
import time
import threading
import redis
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger

from core.models import TickData, BarData, Position, AccountInfo
from utils.helpers import safe_float, safe_int

class RedisCache:
    def __init__(self, config: Dict[str, Any]):
        self._tick_prefix = "tick:"
        self._bar_prefix = "bar:"
        self._position_prefix = "position:"
        self._account_key = "account:info"
        self._signal_prefix = "signal:"
        self._risk_prefix = "risk:"

        self._memory_cache = {}
        self._cache_lock = threading.RLock()
        self._redis_available = False
        self._signal_callback = None
        self._reconnect_interval = 30
        self._reconnect_task = None
        self._stop_event = threading.Event()
        self._reconnect_failures = 0  # 连续失败计数
        self._max_reconnect_failures = 10  # 连续失败上限后停止重试
        self._reconnect_suppressed = False  # 是否已停止重试

        self._redis_config = config["redis"]
        self._redis_enabled = self._redis_config.get("enabled", True)

        if self._redis_enabled:
            try:
                self._redis = redis.Redis(
                    host=self._redis_config["host"],
                    port=self._redis_config["port"],
                    db=self._redis_config["db"],
                    password=self._redis_config.get("password"),
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    protocol=2  # RESP2协议，兼容Redis 5.x (Windows)
                )
                self._redis.ping()
                self._redis_available = True
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Redis not available, using memory cache: {e}")
                self._redis = None
        else:
            self._redis = None
            logger.info("Redis disabled in config, using memory cache only")

        # Start auto-reconnect daemon thread (only if Redis enabled)
        if self._redis_enabled:
            self._reconnect_task = threading.Thread(
                target=self._reconnect_loop, daemon=True
            )
            self._reconnect_task.start()
    
    def is_available(self) -> bool:
        """检查Redis是否可用"""
        return self._redis_available

    def set_signal_callback(self, callback):
        self._signal_callback = callback

    def _mark_redis_unavailable(self, operation: str, error: Exception = None) -> None:
        """统一处理 Redis 操作失败：置可用标志为 False，仅在「可用→不可用」状态切换时
        记录一次 warning，避免连续失败刷屏。fail-closed：Redis 降级必须可观测，而非静默吞掉。
        """
        was_available = self._redis_available
        self._redis_available = False
        if was_available:
            logger.warning(f"Redis {operation} failed, degrading to memory cache: {error}")
        else:
            logger.debug(f"Redis {operation} failed (already degraded): {error}")

    def _invoke_signal_callback(self, signal_data: Dict[str, Any]) -> bool:
        """以一致方式调用信号回调，兼容事件循环运行中/未运行两种场景。

        返回 True 表示回调已成功调度/执行，False 表示无回调或执行失败。
        """
        if not self._signal_callback:
            return False
        import asyncio
        try:
            signal_payload = signal_data.get("data", signal_data)
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self._signal_callback(signal_payload))
            else:
                loop.run_until_complete(self._signal_callback(signal_payload))
            return True
        except Exception as e:
            logger.error(f"Signal callback error: {e}")
            return False

    def set_tick(self, tick: TickData, ttl_seconds: int = 120):
        key = f"{self._tick_prefix}{tick.symbol}"
        now = time.time()
        data = {
            "price": tick.price,
            "volume": tick.volume,
            "bid_price": tick.bid_price,
            "bid_volume": tick.bid_volume,
            "ask_price": tick.ask_price,
            "ask_volume": tick.ask_volume,
            "timestamp": tick.timestamp.isoformat(),
            "received_at": now,
        }

        if self._redis_available:
            try:
                self._redis.hset(key, mapping=data)
                self._redis.expire(key, ttl_seconds)
            except Exception as e:
                self._mark_redis_unavailable("set_tick", e)

        with self._cache_lock:
            self._memory_cache[key] = data

    def set_ticks(self, ticks: List[TickData], ttl_seconds: int = 120):
        """批量写入多个tick，减少Redis往返"""
        if not ticks:
            return

        now = time.time()
        pipe = None
        if self._redis_available:
            try:
                pipe = self._redis.pipeline()
            except Exception as e:
                self._mark_redis_unavailable("set_ticks.pipeline", e)

        with self._cache_lock:
            for tick in ticks:
                key = f"{self._tick_prefix}{tick.symbol}"
                data = {
                    "price": tick.price,
                    "volume": tick.volume,
                    "bid_price": tick.bid_price,
                    "bid_volume": tick.bid_volume,
                    "ask_price": tick.ask_price,
                    "ask_volume": tick.ask_volume,
                    "timestamp": tick.timestamp.isoformat(),
                    "received_at": now,
                }
                self._memory_cache[key] = data
                if pipe is not None:
                    pipe.hset(key, mapping=data)
                    pipe.expire(key, ttl_seconds)

        if pipe is not None:
            try:
                pipe.execute()
            except Exception as e:
                self._mark_redis_unavailable("set_ticks.execute", e)

    def get_tick(self, symbol: str, max_age_seconds: float = 120.0) -> Optional[TickData]:
        key = f"{self._tick_prefix}{symbol}"

        data = None
        if self._redis_available:
            try:
                data = self._redis.hgetall(key)
            except Exception as e:
                self._mark_redis_unavailable("get_tick", e)

        if not data:
            with self._cache_lock:
                data = self._memory_cache.get(key)

        if not data:
            return None

        try:
            received_at = safe_float(data.get("received_at", 0))
            if max_age_seconds > 0 and time.time() - received_at > max_age_seconds:
                logger.debug(f"Tick data for {symbol} is stale (age={time.time()-received_at:.1f}s)")
                return None

            return TickData(
                symbol=symbol,
                price=safe_float(data.get("price")),
                volume=safe_float(data.get("volume")),
                bid_price=safe_float(data.get("bid_price")),
                bid_volume=safe_float(data.get("bid_volume")),
                ask_price=safe_float(data.get("ask_price")),
                ask_volume=safe_float(data.get("ask_volume")),
                timestamp=datetime.fromisoformat(data["timestamp"])
            )
        except Exception as e:
            logger.error(f"Error parsing cached tick for {symbol}: {e}")
            return None

    def is_tick_fresh(self, symbol: str, max_age_seconds: float = 30.0) -> bool:
        key = f"{self._tick_prefix}{symbol}"
        data = None
        if self._redis_available:
            try:
                data = self._redis.hgetall(key)
            except Exception as e:
                self._mark_redis_unavailable("is_tick_fresh", e)
        if not data:
            with self._cache_lock:
                data = self._memory_cache.get(key)
        if not data:
            return False
        received_at = safe_float(data.get("received_at", 0))
        return time.time() - received_at <= max_age_seconds

    def set_bar(self, bar: BarData):
        key = f"{self._bar_prefix}{bar.symbol}:{bar.interval}"
        data = {
            "timestamp": bar.timestamp.isoformat(),
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        }

        if self._redis_available:
            try:
                self._redis.hset(key, mapping=data)
                self._redis.expire(key, 86400)
            except Exception as e:
                self._mark_redis_unavailable("set_bar", e)

        with self._cache_lock:
            self._memory_cache[key] = data

    def get_bar(self, symbol: str, interval: str) -> Optional[BarData]:
        key = f"{self._bar_prefix}{symbol}:{interval}"

        if self._redis_available:
            try:
                data = self._redis.hgetall(key)
                if data:
                    return BarData(
                        symbol=symbol,
                        interval=interval,
                        timestamp=datetime.fromisoformat(data["timestamp"]),
                        open=safe_float(data.get("open")),
                        high=safe_float(data.get("high")),
                        low=safe_float(data.get("low")),
                        close=safe_float(data.get("close")),
                        volume=safe_float(data.get("volume"))
                    )
            except Exception as e:
                self._mark_redis_unavailable("get_bar", e)

        with self._cache_lock:
            data = self._memory_cache.get(key)
        if not data:
            return None

        return BarData(
            symbol=symbol,
            interval=interval,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=safe_float(data.get("open")),
            high=safe_float(data.get("high")),
            low=safe_float(data.get("low")),
            close=safe_float(data.get("close")),
            volume=safe_float(data.get("volume"))
        )

    def set_position(self, position: Position):
        key = f"{self._position_prefix}{position.symbol}"
        data = {
            "symbol": position.symbol,
            "side": position.side,
            "quantity": position.quantity,
            "avg_cost": position.avg_cost,
            "mark_price": position.mark_price,
            "unrealized_pnl": position.unrealized_pnl,
            "margin": position.margin,
            "leverage": position.leverage,
            "maintenance_margin_rate": position.maintenance_margin_rate,
            "notional_usd": position.notional_usd,
            "liquidation_price": position.liquidation_price,
            "timestamp": position.timestamp.isoformat()
        }

        if self._redis_available:
            try:
                self._redis.hset(key, mapping=data)
                self._redis.expire(key, 30)
            except Exception as e:
                self._mark_redis_unavailable("set_position", e)

        with self._cache_lock:
            self._memory_cache[key] = data

    def get_position(self, symbol: str) -> Optional[Position]:
        key = f"{self._position_prefix}{symbol}"

        if self._redis_available:
            try:
                data = self._redis.hgetall(key)
                if data:
                    return Position(
                        symbol=data["symbol"],
                        side=data["side"],
                        quantity=safe_float(data.get("quantity")),
                        avg_cost=safe_float(data.get("avg_cost")),
                        mark_price=safe_float(data.get("mark_price")),
                        unrealized_pnl=safe_float(data.get("unrealized_pnl")),
                        margin=safe_float(data.get("margin")),
                        leverage=safe_int(data.get("leverage")),
                        maintenance_margin_rate=safe_float(data.get("maintenance_margin_rate")),
                        notional_usd=safe_float(data.get("notional_usd")),
                        liquidation_price=safe_float(data.get("liquidation_price")),
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
            except Exception as e:
                self._mark_redis_unavailable("get_position", e)

        with self._cache_lock:
            data = self._memory_cache.get(key)
        if not data:
            return None

        return Position(
            symbol=data["symbol"],
            side=data["side"],
            quantity=safe_float(data.get("quantity")),
            avg_cost=safe_float(data.get("avg_cost")),
            mark_price=safe_float(data.get("mark_price")),
            unrealized_pnl=safe_float(data.get("unrealized_pnl")),
            margin=safe_float(data.get("margin")),
            leverage=safe_int(data.get("leverage")),
            maintenance_margin_rate=safe_float(data.get("maintenance_margin_rate")),
            notional_usd=safe_float(data.get("notional_usd")),
            liquidation_price=safe_float(data.get("liquidation_price")),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

    def set_account_info(self, account: AccountInfo):
        data = {
            "total_equity": str(account.total_equity),
            "available_balance": str(account.available_balance),
            "used_margin": str(account.used_margin),
            "unrealized_pnl": str(account.unrealized_pnl),
            "margin_rate": str(account.margin_rate),
            "timestamp": account.timestamp.isoformat()
        }

        if self._redis_available:
            try:
                self._redis.hset(self._account_key, mapping=data)
                self._redis.expire(self._account_key, 60)
            except Exception as e:
                self._mark_redis_unavailable("set_account_info", e)

        with self._cache_lock:
            self._memory_cache[self._account_key] = data

    def get_account_info(self) -> Optional[AccountInfo]:
        if self._redis_available:
            try:
                data = self._redis.hgetall(self._account_key)
                if data:
                    return AccountInfo(
                        total_equity=safe_float(data.get("total_equity")),
                        available_balance=safe_float(data.get("available_balance")),
                        used_margin=safe_float(data.get("used_margin")),
                        unrealized_pnl=safe_float(data.get("unrealized_pnl")),
                        margin_rate=safe_float(data.get("margin_rate")),
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
            except Exception as e:
                self._mark_redis_unavailable("get_account_info", e)

        with self._cache_lock:
            data = self._memory_cache.get(self._account_key)
        if not data:
            return None

        return AccountInfo(
            total_equity=safe_float(data.get("total_equity")),
            available_balance=safe_float(data.get("available_balance")),
            used_margin=safe_float(data.get("used_margin")),
            unrealized_pnl=safe_float(data.get("unrealized_pnl")),
            margin_rate=safe_float(data.get("margin_rate")),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

    def publish_signal(self, signal_data: Dict[str, Any]) -> bool:
        """发布信号：优先 Redis pub/sub，不可用时降级为直接回调。

        fail-closed：返回 True 表示信号已成功发布/回调，False 表示信号被丢弃
        （Redis 不可用且无回调），调用方据此可观测到信号丢失而非静默吞掉。
        """
        strategy = str(signal_data.get("strategy_name", signal_data.get("strategy", "")) or "")
        from core.signal_flow_stats import record_signal_flow_event
        record_signal_flow_event("candidate", strategy=strategy)

        # Redis 不可用时直接走回调（无 pub/sub 通道）
        if not self._redis_available:
            published = self._invoke_signal_callback(signal_data)
            record_signal_flow_event(
                "published" if published else "publish_failed",
                strategy=strategy,
                reason="callback" if published else "redis_unavailable_no_callback",
            )
            return published

        channel = f"{self._signal_prefix}trading"
        try:
            message = json.dumps(signal_data)
        except Exception as e:
            record_signal_flow_event(
                "publish_failed",
                strategy=strategy,
                reason="serialization_error",
            )
            logger.error(f"Signal serialization failed: {e}")
            return False

        try:
            self._redis.publish(channel, message)
            record_signal_flow_event("published", strategy=strategy, reason="redis")
            return True
        except Exception as e:
            self._mark_redis_unavailable("publish_signal", e)
            # 发布失败，尝试直接回调兜底
            published = self._invoke_signal_callback(signal_data)
            record_signal_flow_event(
                "published" if published else "publish_failed",
                strategy=strategy,
                reason="callback_fallback" if published else "redis_and_callback_failed",
            )
            return published

    def subscribe_signals(self):
        if self._redis_available:
            try:
                pubsub = self._redis.pubsub()
                pubsub.subscribe(f"{self._signal_prefix}trading")
                return pubsub
            except Exception as e:
                self._mark_redis_unavailable("subscribe_signals", e)
        
        return None

    def cache_signal(self, symbol: str, signal_data: Dict[str, Any]):
        """缓存信号（供信号处理器去重/查询使用）。

        P0 修复：scheduler._on_strategy_signal 曾调用不存在的 cache_signal 导致
        信号记录链路 AttributeError 中断。现提供内存 + 可选 Redis 双层缓存。
        """
        try:
            key = f"{self._signal_prefix}cache:{symbol}"
            with self._cache_lock:
                self._memory_cache[key] = {
                    "symbol": symbol,
                    "signal": signal_data,
                    "timestamp": datetime.now().isoformat(),
                }
            if self._redis_available:
                try:
                    self._redis.set(key, json.dumps(signal_data), ex=300)
                except Exception as e:
                    self._mark_redis_unavailable("cache_signal", e)
        except Exception as e:
            logger.warning(f"cache_signal error: {e}")

    def set_risk_limit(self, key: str, value: float):
        risk_key = f"{self._risk_prefix}{key}"

        if self._redis_available:
            try:
                self._redis.set(risk_key, value)
                self._redis.expire(risk_key, 3600)
            except Exception as e:
                self._mark_redis_unavailable("set_risk_limit", e)

        with self._cache_lock:
            self._memory_cache[risk_key] = value

    def get_risk_limit(self, key: str, default: float = 0.0) -> float:
        risk_key = f"{self._risk_prefix}{key}"

        if self._redis_available:
            try:
                value = self._redis.get(risk_key)
                if value is not None:
                    return safe_float(value)
            except Exception as e:
                self._mark_redis_unavailable("get_risk_limit", e)

        with self._cache_lock:
            return safe_float(self._memory_cache.get(risk_key, default), default)

    def incr_risk_counter(self, key: str) -> int:
        counter_key = f"{self._risk_prefix}counter:{key}"

        if self._redis_available:
            try:
                return int(self._redis.incr(counter_key))
            except Exception as e:
                self._mark_redis_unavailable("incr_risk_counter", e)

        with self._cache_lock:
            current = self._memory_cache.get(counter_key, 0)
            current += 1
            self._memory_cache[counter_key] = current
            return current

    def reset_risk_counter(self, key: str):
        counter_key = f"{self._risk_prefix}counter:{key}"

        if self._redis_available:
            try:
                self._redis.delete(counter_key)
            except Exception as e:
                self._mark_redis_unavailable("reset_risk_counter", e)

        with self._cache_lock:
            self._memory_cache.pop(counter_key, None)

    def set_strategy_state(self, strategy_name: str, state: Dict[str, Any]):
        key = f"strategy:{strategy_name}:state"
        data = json.dumps(state)

        if self._redis_available:
            try:
                self._redis.set(key, data)
                self._redis.expire(key, 3600)
            except Exception as e:
                self._mark_redis_unavailable("set_strategy_state", e)

        with self._cache_lock:
            self._memory_cache[key] = data

    def get_strategy_state(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        key = f"strategy:{strategy_name}:state"

        if self._redis_available:
            try:
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception as e:
                self._mark_redis_unavailable("get_strategy_state", e)

        with self._cache_lock:
            data = self._memory_cache.get(key)
        if not data:
            return None

        return json.loads(data)

    def health_check(self) -> bool:
        """健康检查：Redis 禁用时始终返回 True"""
        if not self._redis_enabled:
            return True
        if self._redis_available:
            try:
                self._redis.ping()
                return True
            except Exception as e:
                self._mark_redis_unavailable("health_check", e)

        # 即使 _redis_available=False 也主动尝试重连
        if self._try_reconnect():
            return True

        return False

    def _try_reconnect(self) -> bool:
        """尝试重新连接 Redis，成功返回 True。使用指数退避避免日志轰炸。"""
        if self._reconnect_suppressed:
            return False
        
        if self._redis is None:
            try:
                self._redis = redis.Redis(
                    host=self._redis_config["host"],
                    port=self._redis_config["port"],
                    db=self._redis_config["db"],
                    password=self._redis_config.get("password"),
                    decode_responses=True,
                    socket_timeout=5,
                    socket_connect_timeout=5,
                    protocol=2
                )
                self._redis.ping()
                self._redis_available = True
                self._reconnect_failures = 0
                self._reconnect_suppressed = False
                logger.info("Redis initial connection recovered")
                return True
            except Exception as e:
                self._reconnect_failures += 1
                if self._reconnect_failures >= self._max_reconnect_failures:
                    if not self._reconnect_suppressed:
                        self._reconnect_suppressed = True
                        logger.warning(f"Redis unavailable after {self._reconnect_failures} attempts, "
                                     f"suppressing further reconnect logs. Caching disabled.")
                    return False
                # 指数退避：每N次失败只打印一次日志
                if self._reconnect_failures <= 3 or self._reconnect_failures % 5 == 0:
                    logger.debug(f"Redis initial reconnect failed (attempt {self._reconnect_failures}): {e}")
                return False
        try:
            self._redis.ping()
            if not self._redis_available:
                self._redis_available = True
                self._reconnect_failures = 0
                self._reconnect_suppressed = False
                logger.info("Redis reconnected successfully")
            return True
        except Exception as e:
            self._reconnect_failures += 1
            self._redis_available = False
            if self._reconnect_failures >= self._max_reconnect_failures:
                if not self._reconnect_suppressed:
                    self._reconnect_suppressed = True
                    logger.warning(f"Redis unavailable after {self._reconnect_failures} attempts, "
                                 f"suppressing further reconnect logs. Caching disabled.")
                return False
            # 指数退避：每N次失败只打印一次日志
            if self._reconnect_failures <= 3 or self._reconnect_failures % 5 == 0:
                logger.debug(f"Redis reconnect failed (attempt {self._reconnect_failures}): {e}")
            return False

    def _reconnect_loop(self):
        """后台重连循环，使用指数退避调整重试间隔。
        
        P5: 长时间抑制后自动重置，定期尝试恢复Redis连接
        """
        self._last_suppressed_reset = 0.0  # P5: 上次重置抑制的时间戳
        
        while not self._stop_event.wait(self._reconnect_interval):
            try:
                if self._reconnect_suppressed:
                    # P5: 已停止重试，但每10分钟自动重置抑制状态，给Redis一次恢复机会
                    now = time.time()
                    if now - self._last_suppressed_reset > 600:  # 10分钟
                        self._reconnect_suppressed = False
                        self._reconnect_failures = 0
                        self._last_suppressed_reset = now
                        logger.info("Redis reconnect suppression reset after 10min, attempting recovery")
                        self._try_reconnect()
                    else:
                        self._stop_event.wait(300)
                    continue
                if not self.health_check():
                    self._try_reconnect()
                    # 指数退避：失败越多，等待越久（上限5分钟）
                    if self._reconnect_failures > 0:
                        backoff = min(300, self._reconnect_interval * (2 ** min(self._reconnect_failures, 5)))
                        self._stop_event.wait(backoff)
            except Exception as e:
                logger.error(f"Reconnect loop error: {e}")

    def start_reconnect_task(self):
        """启动后台重连任务（幂等，重复调用安全）。"""
        if self._reconnect_task is not None and self._reconnect_task.is_alive():
            return
        self._stop_event.clear()
        self._reconnect_task = threading.Thread(
            target=self._reconnect_loop, daemon=True
        )
        self._reconnect_task.start()
        logger.info("Redis reconnect task started")

    def stop_reconnect_task(self):
        """停止后台重连任务。"""
        self._stop_event.set()
        if self._reconnect_task is not None:
            self._reconnect_task.join(timeout=5)
        logger.info("Redis reconnect task stopped")