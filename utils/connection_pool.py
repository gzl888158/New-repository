"""
连接池管理器
管理HTTP连接池、WebSocket连接池、数据库连接池
提供连接复用、健康检查、自动重连机制
"""
import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from loguru import logger
import threading
from contextlib import asynccontextmanager


class PoolState(Enum):
    """连接池状态"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    SHUTDOWN = "shutdown"


class ConnectionState(Enum):
    """连接状态"""
    IDLE = "idle"
    IN_USE = "in_use"
    ERROR = "error"
    CLOSED = "closed"


@dataclass
class ConnectionInfo:
    """连接信息"""
    conn_id: str
    state: ConnectionState
    created_at: datetime
    last_used: datetime
    use_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


@dataclass
class PoolStats:
    """连接池统计"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    pending_requests: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    peak_connections: int = 0
    last_updated: datetime = field(default_factory=datetime.now)


class ConnectionPool:
    """连接池基类"""
    
    def __init__(
        self,
        name: str,
        max_connections: int = 10,
        min_connections: int = 2,
        max_idle_time_seconds: float = 300,
        connection_timeout_seconds: float = 30,
        health_check_interval_seconds: float = 60,
    ):
        self.name = name
        # 安全转换配置值，防止非法输入导致后续崩溃
        self.max_connections = max(1, int(max_connections) if max_connections and int(max_connections) > 0 else 10)
        self.min_connections = max(0, min(self.max_connections,
                                          int(min_connections) if min_connections is not None and int(min_connections) >= 0 else 2))
        self.max_idle_time_seconds = float(max_idle_time_seconds) if max_idle_time_seconds and float(max_idle_time_seconds) > 0 else 300.0
        self.connection_timeout_seconds = float(connection_timeout_seconds) if connection_timeout_seconds and float(connection_timeout_seconds) > 0 else 30.0
        self.health_check_interval_seconds = float(health_check_interval_seconds) if health_check_interval_seconds and float(health_check_interval_seconds) > 0 else 60.0
        
        self._connections: Dict[str, ConnectionInfo] = {}
        self._available: asyncio.Queue = None
        self._lock = asyncio.Lock()
        self._state = PoolState.INITIALIZING
        self._stats = PoolStats()
        self._health_task: Optional[asyncio.Task] = None
        self._pending_requests = 0
        
        self._created_count = 0
    
    async def initialize(self) -> bool:
        """初始化连接池"""
        try:
            self._available = asyncio.Queue(maxsize=self.max_connections)
            
            for _ in range(self.min_connections):
                conn_id = await self._create_connection()
                if conn_id:
                    self._connections[conn_id] = ConnectionInfo(
                        conn_id=conn_id,
                        state=ConnectionState.IDLE,
                        created_at=datetime.now(),
                        last_used=datetime.now(),
                    )
                    await self._available.put(conn_id)
            
            self._state = PoolState.HEALTHY
            self._update_stats()
            
            self._health_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Connection pool '{self.name}' initialized with {len(self._connections)} connections")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize connection pool '{self.name}': {e}")
            self._state = PoolState.UNHEALTHY
            return False
    
    async def _create_connection(self) -> Optional[str]:
        """创建新连接（子类实现）"""
        self._created_count += 1
        return f"{self.name}_conn_{self._created_count}"
    
    async def _close_connection(self, conn_id: str) -> bool:
        """关闭连接（子类实现）"""
        if conn_id in self._connections:
            self._connections[conn_id].state = ConnectionState.CLOSED
            del self._connections[conn_id]
            return True
        return False
    
    async def _check_connection_health(self, conn_id: str) -> bool:
        """检查连接健康状态（子类实现）"""
        return conn_id in self._connections
    
    @asynccontextmanager
    async def acquire(self, timeout_seconds: float = 10):
        """获取连接"""
        conn_id = None
        start_time = time.time()
        
        try:
            self._pending_requests += 1
            
            try:
                conn_id = await asyncio.wait_for(
                    self._available.get(),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"Connection pool '{self.name}' acquire timeout")
                if len(self._connections) < self.max_connections:
                    conn_id = await self._create_connection()
                    if conn_id:
                        self._connections[conn_id] = ConnectionInfo(
                            conn_id=conn_id,
                            state=ConnectionState.IN_USE,
                            created_at=datetime.now(),
                            last_used=datetime.now(),
                        )
                
                if not conn_id:
                    raise RuntimeError(f"Connection pool '{self.name}' exhausted")
            
            if conn_id and conn_id in self._connections:
                self._connections[conn_id].state = ConnectionState.IN_USE
                self._connections[conn_id].last_used = datetime.now()
                self._connections[conn_id].use_count += 1
            
            yield conn_id
            
        finally:
            self._pending_requests -= 1
            
            if conn_id:
                if conn_id in self._connections:
                    self._connections[conn_id].state = ConnectionState.IDLE
                
                try:
                    self._available.put_nowait(conn_id)
                except asyncio.QueueFull:
                    pass
                
                self._stats.total_requests += 1
            
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats.avg_latency_ms = (
                (self._stats.avg_latency_ms * (self._stats.total_requests - 1) + elapsed_ms)
                / self._stats.total_requests
                if self._stats.total_requests > 0 else elapsed_ms
            )
    
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._state != PoolState.SHUTDOWN:
            try:
                await asyncio.sleep(self.health_check_interval_seconds)
                
                healthy_count = 0
                error_count = 0
                
                for conn_id, info in list(self._connections.items()):
                    is_healthy = await self._check_connection_health(conn_id)
                    
                    if is_healthy:
                        healthy_count += 1
                        
                        if info.state == ConnectionState.ERROR:
                            info.state = ConnectionState.IDLE
                            info.error_count = 0
                    else:
                        error_count += 1
                        info.state = ConnectionState.ERROR
                        info.error_count += 1
                        
                        if info.error_count >= 3:
                            await self._close_connection(conn_id)
                
                if healthy_count >= self.min_connections:
                    self._state = PoolState.HEALTHY
                elif healthy_count > 0:
                    self._state = PoolState.DEGRADED
                else:
                    self._state = PoolState.UNHEALTHY
                
                self._update_stats()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error in pool '{self.name}': {e}")
    
    def _update_stats(self):
        """更新统计信息"""
        self._stats.total_connections = len(self._connections)
        self._stats.active_connections = sum(
            1 for info in self._connections.values()
            if info.state == ConnectionState.IN_USE
        )
        self._stats.idle_connections = sum(
            1 for info in self._connections.values()
            if info.state == ConnectionState.IDLE
        )
        self._stats.pending_requests = self._pending_requests
        self._stats.peak_connections = max(
            self._stats.peak_connections,
            self._stats.total_connections
        )
        self._stats.last_updated = datetime.now()
    
    def get_stats(self) -> PoolStats:
        """获取连接池统计"""
        self._update_stats()
        return self._stats
    
    def get_state(self) -> PoolState:
        """获取连接池状态"""
        return self._state
    
    async def shutdown(self):
        """关闭连接池"""
        self._state = PoolState.SHUTDOWN
        
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        for conn_id in list(self._connections.keys()):
            await self._close_connection(conn_id)
        
        self._connections.clear()
        
        while not self._available.empty():
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        logger.info(f"Connection pool '{self.name}' shutdown complete")


class HTTPConnectionPool(ConnectionPool):
    """HTTP连接池"""
    
    def __init__(self, name: str, base_url: str, **kwargs):
        super().__init__(name, **kwargs)
        self.base_url = base_url
        self._session = None
    
    async def _create_connection(self) -> Optional[str]:
        """创建HTTP会话"""
        import aiohttp
        
        if not self._session:
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout_seconds)
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
            )
        
        return await super()._create_connection()
    
    async def _close_connection(self, conn_id: str) -> bool:
        """关闭HTTP连接"""
        return await super()._close_connection(conn_id)
    
    async def _check_connection_health(self, conn_id: str) -> bool:
        """检查HTTP连接健康状态"""
        if self._session and not self._session.closed:
            return True
        return False
    
    async def shutdown(self):
        """关闭HTTP连接池"""
        if self._session:
            await self._session.close()
            self._session = None
        
        await super().shutdown()


class ConnectionPoolManager:
    """连接池管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._pools: Dict[str, ConnectionPool] = {}
            self._initialized = True
    
    async def create_pool(self, name: str, pool_type: str, config: Dict[str, Any]) -> Optional[ConnectionPool]:
        """创建连接池"""
        if name in self._pools:
            logger.warning(f"Pool '{name}' already exists")
            return self._pools[name]
        
        pool = None
        
        if pool_type == "http":
            pool = HTTPConnectionPool(
                name=name,
                base_url=config.get("base_url", ""),
                max_connections=config.get("max_connections", 10),
                min_connections=config.get("min_connections", 2),
            )
        
        if pool:
            if await pool.initialize():
                self._pools[name] = pool
                logger.info(f"Created connection pool '{name}'")
                return pool
            else:
                logger.error(f"Failed to create connection pool '{name}'")
                return None
        
        return None
    
    def get_pool(self, name: str) -> Optional[ConnectionPool]:
        """获取连接池"""
        return self._pools.get(name)
    
    def list_pools(self) -> List[str]:
        """列出所有连接池"""
        return list(self._pools.keys())
    
    def get_all_stats(self) -> Dict[str, PoolStats]:
        """获取所有连接池统计"""
        return {name: pool.get_stats() for name, pool in self._pools.items()}
    
    async def shutdown_all(self):
        """关闭所有连接池"""
        for name, pool in list(self._pools.items()):
            await pool.shutdown()
            del self._pools[name]
        
        logger.info("All connection pools shutdown")