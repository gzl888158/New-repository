"""
分层数据缓存系统
实现 L1（内存）+ L2（Redis）+ L3（持久化）三级缓存架构。
"""
import time
import json
import threading
from typing import Dict, Any, Optional, Tuple
from loguru import logger


class CacheEntry:
    """缓存条目"""

    def __init__(self, value: Any, ttl: float, priority: int = 5):
        self.value = value
        self.ttl = ttl
        self.created_at = time.time()
        self.last_access = time.time()
        self.access_count = 0
        self.priority = priority  # 1-10, 越高越重要

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl

    def access(self) -> Any:
        self.last_access = time.time()
        self.access_count += 1
        return self.value

    def age(self) -> float:
        return time.time() - self.created_at

    def idle_time(self) -> float:
        return time.time() - self.last_access


class L1MemoryCache:
    """L1 内存缓存（最快）"""

    def __init__(self, max_size: int = 1000, default_ttl: float = 5.0):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.access()

    def set(self, key: str, value: Any, ttl: Optional[float] = None, priority: int = 5) -> bool:
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            if key in self._cache:
                self._cache[key] = CacheEntry(value, ttl, priority)
                return True
            
            if len(self._cache) >= self.max_size:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(value, ttl, priority)
            return True

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._cache.pop(key, None) is not None

    def clear(self):
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self) -> int:
        """清理过期项"""
        with self._lock:
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            for k in expired:
                del self._cache[k]
            return len(expired)

    def _evict_lru(self):
        """淘汰最少使用项（优先级低优先，同优先级下淘汰最久未访问项）"""
        if not self._cache:
            return
        
        # 优先淘汰低优先级；同优先级内淘汰 idle_time 最大（最久未访问）的项
        lru_key = min(
            self._cache.keys(),
            key=lambda k: (
                self._cache[k].priority,
                -self._cache[k].idle_time(),
            )
        )
        del self._cache[lru_key]
        self._evictions += 1

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "level": "L1_memory",
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / max(total, 1),
                "evictions": self._evictions,
            }


class L2RedisCache:
    """L2 Redis 缓存（跨进程共享）"""

    def __init__(self, redis_client=None, default_ttl: float = 30.0, key_prefix: str = "md:"):
        self._redis = redis_client
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self._available = redis_client is not None
        self._hits = 0
        self._misses = 0
        self._errors = 0

    def _key(self, key: str) -> str:
        return f"{self.key_prefix}{key}"

    def get(self, key: str) -> Optional[Any]:
        if not self._available:
            self._misses += 1
            return None
        try:
            data = self._redis.get(self._key(key))
            if data is None:
                self._misses += 1
                return None
            self._hits += 1
            if isinstance(data, bytes):
                data = data.decode("utf-8")
            return json.loads(data)
        except Exception as e:
            self._errors += 1
            logger.debug(f"L2 cache get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        if not self._available:
            return False
        try:
            ttl = self.default_ttl if ttl is None else ttl
            data = json.dumps(value, default=str)
            # Redis SETEX 要求 TTL > 0，钳制到至少 1 秒避免报错
            self._redis.setex(self._key(key), max(1, int(ttl)), data)
            return True
        except Exception as e:
            self._errors += 1
            logger.debug(f"L2 cache set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        if not self._available:
            return False
        try:
            self._redis.delete(self._key(key))
            return True
        except Exception as e:
            self._errors += 1
            return False

    def is_available(self) -> bool:
        return self._available

    def get_stats(self) -> Dict[str, Any]:
        total = self._hits + self._misses
        return {
            "level": "L2_redis",
            "available": self._available,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(total, 1),
            "errors": self._errors,
        }


class TieredCache:
    """分层缓存管理器"""

    def __init__(self, redis_client=None, l1_max_size: int = 1000):
        self.l1 = L1MemoryCache(max_size=l1_max_size)
        self.l2 = L2RedisCache(redis_client=redis_client)
        logger.info("TieredCache initialized (L1 memory + L2 redis)")

    def get(self, key: str) -> Optional[Any]:
        """按 L1 -> L2 顺序获取"""
        value = self.l1.get(key)
        if value is not None:
            return value
        
        value = self.l2.get(key)
        if value is not None:
            self.l1.set(key, value, ttl=self.l1.default_ttl * 2)
            return value
        
        return None

    def set(self, key: str, value: Any, ttl: float = 5.0, 
            l2_ttl: float = 30.0, priority: int = 5) -> bool:
        """同时写入 L1 和 L2"""
        l1_ok = self.l1.set(key, value, ttl=ttl, priority=priority)
        l2_ok = self.l2.set(key, value, ttl=l2_ttl)
        return l1_ok or l2_ok

    def delete(self, key: str) -> bool:
        l1_ok = self.l1.delete(key)
        l2_ok = self.l2.delete(key)
        return l1_ok or l2_ok

    def cleanup(self) -> int:
        return self.l1.cleanup_expired()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "l1": self.l1.get_stats(),
            "l2": self.l2.get_stats(),
        }


# 各类数据的默认 TTL（秒）
class CacheTTL:
    TICKER_L1 = 2.0
    TICKER_L2 = 10.0
    ORDERBOOK_L1 = 1.0
    ORDERBOOK_L2 = 5.0
    KLINES_L1 = 5.0
    KLINES_L2 = 60.0
    INSTRUMENT_L1 = 3600.0
    INSTRUMENT_L2 = 86400.0
    FUNDING_L1 = 30.0
    FUNDING_L2 = 300.0
