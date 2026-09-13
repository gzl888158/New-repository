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
            except Exception:
                self._redis_available = False

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
            except Exception:
                self._redis_available = False

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
            except Exception:
                self._redis_available = False

    def get_tick(self, symbol: str, max_age_seconds: float = 120.0) -> Optional[TickData]:
        key = f"{self._tick_prefix}{symbol}"

        data = None
        if self._redis_available:
            try:
                data = self._redis.hgetall(key)
            except Exception:
                self._redis_available = False

        if not data:
            with self._cache_lock:
                data = self._memory_cache.get(key)

        if not data:
            return None

        try:
            received_at = float(data.get("received_at", 0))
            if max_age_seconds > 0 and time.time() - received_at > max_age_seconds:
                logger.debug(f"Tick data for {symbol} is stale (age={time.time()-received_at:.1f}s)")
                return None

            return TickData(
                symbol=symbol,
                price=float(data["price"]),
                volume=float(data["volume"]),
                bid_price=float(data["bid_price"]),
                bid_volume=float(data["bid_volume"]),
                ask_price=float(data["ask_price"]),
                ask_volume=float(data["ask_volume"]),
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
            except Exception:
                self._redis_available = False
        if not data:
            with self._cache_lock:
                data = self._memory_cache.get(key)
        if not data:
            return False
        received_at = float(data.get("received_at", 0))
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
            except Exception:
                self._redis_available = False

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
                        open=float(data["open"]),
                        high=float(data["high"]),
                        low=float(data["low"]),
                        close=float(data["close"]),
                        volume=float(data["volume"])
                    )
            except Exception:
                self._redis_available = False

        with self._cache_lock:
            data = self._memory_cache.get(key)
        if not data:
            return None

        return BarData(
            symbol=symbol,
            interval=interval,
            timestamp=datetime.fromisoformat(data["timestamp"]),
            open=float(data["open"]),
            high=float(data["high"]),
            low=float(data["low"]),
            close=float(data["close"]),
            volume=float(data["volume"])
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
            "timestamp": position.timestamp.isoformat()
        }

        if self._redis_available:
            try:
                self._redis.hset(key, mapping=data)
                self._redis.expire(key, 30)
            except Exception:
                self._redis_available = False

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
                        quantity=float(data["quantity"]),
                        avg_cost=float(data["avg_cost"]),
                        mark_price=float(data["mark_price"]),
                        unrealized_pnl=float(data["unrealized_pnl"]),
                        margin=float(data["margin"]),
                        leverage=int(data["leverage"]),
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
            except Exception:
                self._redis_available = False

        with self._cache_lock:
            data = self._memory_cache.get(key)
        if not data:
            return None

        return Position(
            symbol=data["symbol"],
            side=data["side"],
            quantity=float(data["quantity"]),
            avg_cost=float(data["avg_cost"]),
            mark_price=float(data["mark_price"]),
            unrealized_pnl=float(data["unrealized_pnl"]),
            margin=float(data["margin"]),
            leverage=int(data["leverage"]),
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
            except Exception:
                self._redis_available = False

        with self._cache_lock:
            self._memory_cache[self._account_key] = data

    def get_account_info(self) -> Optional[AccountInfo]:
        if self._redis_available:
            try:
                data = self._redis.hgetall(self._account_key)
                if data:
                    return AccountInfo(
                        total_equity=float(data["total_equity"]),
                        available_balance=float(data["available_balance"]),
                        used_margin=float(data["used_margin"]),
                        unrealized_pnl=float(data["unrealized_pnl"]),
                        margin_rate=float(data["margin_rate"]),
                        timestamp=datetime.fromisoformat(data["timestamp"])
                    )
            except Exception:
                self._redis_available = False

        with self._cache_lock:
            data = self._memory_cache.get(self._account_key)
        if not data:
            return None

        return AccountInfo(
            total_equity=float(data["total_equity"]),
            available_balance=float(data["available_balance"]),
            used_margin=float(data["used_margin"]),
            unrealized_pnl=float(data["unrealized_pnl"]),
            margin_rate=float(data["margin_rate"]),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

    def publish_signal(self, signal_data: Dict[str, Any]):
        if self._signal_callback and not self._redis_available:
            import asyncio
            try:
                signal_payload = signal_data.get("data", signal_data)
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self._signal_callback(signal_payload))
                else:
                    loop.run_until_complete(self._signal_callback(signal_payload))
            except Exception as e:
                logger.error(f"Direct signal callback error: {e}")
            return
        
        channel = f"{self._signal_prefix}trading"
        message = json.dumps(signal_data)
        
        if self._redis_available:
            try:
                self._redis.publish(channel, message)
            except Exception:
                self._redis_available = False
                if self._signal_callback:
                    import asyncio
                    try:
                        signal_payload = signal_data.get("data", signal_data)
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(self._signal_callback(signal_payload))
                    except Exception as e:
                        logger.error(f"Fallback signal callback error: {e}")

    def subscribe_signals(self):
        if self._redis_available:
            try:
                pubsub = self._redis.pubsub()
                pubsub.subscribe(f"{self._signal_prefix}trading")
                return pubsub
            except Exception:
                self._redis_available = False
        
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
                except Exception:
                    self._redis_available = False
        except Exception as e:
            logger.debug(f"cache_signal error: {e}")

    def set_risk_limit(self, key: str, value: float):
        risk_key = f"{self._risk_prefix}{key}"

        if self._redis_available:
            try:
                self._redis.set(risk_key, value)
                self._redis.expire(risk_key, 3600)
            except Exception:
                self._redis_available = False

        with self._cache_lock:
            self._memory_cache[risk_key] = value

    def get_risk_limit(self, key: str, default: float = 0.0) -> float:
        risk_key = f"{self._risk_prefix}{key}"

        if self._redis_available:
            try:
                value = self._redis.get(risk_key)
                if value is not None:
                    return float(value)
            except Exception:
                self._redis_available = False

        with self._cache_lock:
            return self._memory_cache.get(risk_key, default)

    def incr_risk_counter(self, key: str) -> int:
        counter_key = f"{self._risk_prefix}counter:{key}"

        if self._redis_available:
            try:
                return int(self._redis.incr(counter_key))
            except Exception:
                self._redis_available = False

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
            except Exception:
                self._redis_available = False

        with self._cache_lock:
            self._memory_cache.pop(counter_key, None)

    def set_strategy_state(self, strategy_name: str, state: Dict[str, Any]):
        key = f"strategy:{strategy_name}:state"
        data = json.dumps(state)

        if self._redis_available:
            try:
                self._redis.set(key, data)
                self._redis.expire(key, 3600)
            except Exception:
                self._redis_available = False

        with self._cache_lock:
            self._memory_cache[key] = data

    def get_strategy_state(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        key = f"strategy:{strategy_name}:state"

        if self._redis_available:
            try:
                data = self._redis.get(key)
                if data:
                    return json.loads(data)
            except Exception:
                self._redis_available = False

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
            except Exception:
                self._redis_available = False

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