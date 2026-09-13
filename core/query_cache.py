"""
数据库查询缓存层 - Database Query Cache Layer

功能：
1. 减少重复数据库查询
2. 智能缓存失效
3. 查询性能优化
4. 缓存命中率统计
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from loguru import logger
from functools import wraps
import hashlib
import json


class CacheEntry:
    """缓存条目"""
    
    def __init__(self, value: Any, ttl_seconds: int, created_at: float = None):
        self.value = value
        self.ttl_seconds = ttl_seconds
        self.created_at = created_at or time.time()
        self.access_count = 0
        self.last_access = self.created_at
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        return time.time() - self.created_at > self.ttl_seconds
    
    def access(self):
        """记录访问"""
        self.access_count += 1
        self.last_access = time.time()


class QueryCache:
    """查询缓存"""
    
    def __init__(self, default_ttl: int = 300, max_size: int = 1000):
        """
        参数：
        - default_ttl: 默认缓存时间（秒）
        - max_size: 最大缓存条目数
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._lock = threading.RLock()
        
        # 统计信息
        self._hits = 0
        self._misses = 0
        
        logger.info(f"QueryCache initialized: default_ttl={default_ttl}s, max_size={max_size}")
    
    def _generate_key(self, query: str, params: tuple = None) -> str:
        """生成缓存键"""
        key_data = f"{query}:{params}" if params else query
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, params: tuple = None) -> Optional[Any]:
        """获取缓存"""
        key = self._generate_key(query, params)
        
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            entry.access()
            self._hits += 1
            return entry.value
    
    def set(self, query: str, value: Any, params: tuple = None, ttl: int = None):
        """设置缓存"""
        key = self._generate_key(query, params)
        ttl = ttl or self._default_ttl
        
        with self._lock:
            # 检查是否需要清理
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            
            self._cache[key] = CacheEntry(value, ttl)
    
    def _evict_lru(self):
        """清理最少使用的缓存"""
        if not self._cache:
            return
        
        # 找到最少使用的条目
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_access)
        del self._cache[lru_key]
        logger.debug(f"Evicted LRU cache entry: {lru_key}")
    
    def invalidate(self, query: str = None, pattern: str = None):
        """
        失效缓存
        
        参数：
        - query: 精确匹配的查询
        - pattern: 模糊匹配模式（如 "SELECT * FROM trade_records"）
        """
        with self._lock:
            if query:
                key = self._generate_key(query)
                if key in self._cache:
                    del self._cache[key]
                    logger.debug(f"Invalidated cache for query: {query[:50]}...")
            elif pattern:
                # 删除匹配模式的所有缓存
                keys_to_delete = [
                    k for k, v in self._cache.items()
                    if pattern in str(v.value)
                ]
                for key in keys_to_delete:
                    del self._cache[key]
                if keys_to_delete:
                    logger.debug(f"Invalidated {len(keys_to_delete)} cache entries matching pattern: {pattern}")
    
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "max_size": self._max_size
            }


class CachedDatabase:
    """带缓存的数据库访问层"""
    
    def __init__(self, db_connection, cache: QueryCache = None):
        self._db = db_connection
        self._cache = cache or QueryCache()
        
        # 需要缓存的表及其TTL
        self._table_ttl = {
            "trade_records": 60,       # 交易记录缓存1分钟
            "position_history": 30,    # 持仓历史缓存30秒
            "account_history": 10,     # 账户历史缓存10秒
            "equity_curve": 120,       # 权益曲线缓存2分钟
            "strategy_performance": 300 # 策略性能缓存5分钟
        }
        
        logger.info("CachedDatabase initialized")
    
    def execute_query(self, query: str, params: tuple = None, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        执行查询（带缓存）
        
        参数：
        - query: SQL查询语句
        - params: 查询参数
        - use_cache: 是否使用缓存
        """
        # 检查缓存
        if use_cache:
            cached_result = self._cache.get(query, params)
            if cached_result is not None:
                return cached_result
        
        # 执行数据库查询
        try:
            cursor = self._db.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # 转换为字典列表
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            
            # 存入缓存
            if use_cache:
                # 根据表名确定TTL
                ttl = self._get_ttl_for_query(query)
                self._cache.set(query, results, params, ttl)
            
            return results
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    def execute_update(self, query: str, params: tuple = None):
        """执行更新（自动失效相关缓存）"""
        try:
            cursor = self._db.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            self._db.commit()
            
            # 失效相关表的缓存
            affected_table = self._extract_table_from_query(query)
            if affected_table:
                self._cache.invalidate(pattern=affected_table)
            
            return cursor.rowcount
        except Exception as e:
            self._db.rollback()
            logger.error(f"Database update failed: {e}")
            raise
    
    def _get_ttl_for_query(self, query: str) -> int:
        """根据查询确定TTL"""
        query_lower = query.lower()
        
        for table, ttl in self._table_ttl.items():
            if table in query_lower:
                return ttl
        
        return self._cache._default_ttl
    
    def _extract_table_from_query(self, query: str) -> Optional[str]:
        """从查询中提取表名"""
        query_lower = query.lower()
        
        # 简单的模式匹配
        if "insert into" in query_lower:
            parts = query_lower.split("insert into")[1].split("(")
            return parts[0].strip()
        elif "update" in query_lower:
            parts = query_lower.split("update")[1].split("set")
            return parts[0].strip()
        elif "delete from" in query_lower:
            parts = query_lower.split("delete from")[1].split("where")
            return parts[0].strip()
        
        return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self._cache.get_stats()


def cached_query(ttl: int = None, key_params: List[str] = None):
    """
    查询缓存装饰器
    
    用法：
    @cached_query(ttl=60)
    def get_trade_records(self, symbol: str, limit: int = 100):
        # 数据库查询
        ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 获取缓存实例
            cache = getattr(self, '_query_cache', None)
            if cache is None:
                return func(self, *args, **kwargs)
            
            # 生成缓存键
            if key_params:
                key_values = [kwargs.get(p, args[i] if i < len(args) else None) for i, p in enumerate(key_params)]
                cache_key = f"{func.__name__}:{':'.join(str(v) for v in key_values)}"
            else:
                cache_key = f"{func.__name__}:{args}:{kwargs}"
            
            # 检查缓存
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # 执行函数
            result = func(self, *args, **kwargs)
            
            # 存入缓存
            cache.set(cache_key, result, ttl=ttl)
            
            return result
        return wrapper
    return decorator


class QueryOptimizer:
    """查询优化器"""
    
    @staticmethod
    def optimize_select_query(query: str) -> str:
        """优化SELECT查询"""
        # 添加LIMIT子句（如果没有）
        if "limit" not in query.lower() and "select" in query.lower():
            query = query.rstrip(';') + " LIMIT 1000"
        
        return query
    
    @staticmethod
    def batch_queries(queries: List[str], batch_size: int = 10) -> List[List[str]]:
        """批量查询分组"""
        return [queries[i:i + batch_size] for i in range(0, len(queries), batch_size)]
    
    @staticmethod
    def add_index_hints(query: str, indexes: Dict[str, str]) -> str:
        """添加索引提示"""
        # 简化版本：仅记录建议
        for table, index in indexes.items():
            if table in query.lower() and index not in query.lower():
                logger.debug(f"Consider using index {index} for table {table}")
        
        return query