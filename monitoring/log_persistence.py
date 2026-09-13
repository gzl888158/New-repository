"""
全量日志持久化模块
行情日志、信号日志、下单日志、风控拦截日志、报错日志分类存储

核心功能：
1. 多类型日志分类存储
2. 异步写入，不阻塞主流程
3. SQLite持久化 + 文件备份
4. 日志查询与检索API
5. 自动轮转与清理
6. 日志压缩与归档
"""
import asyncio
import os
import json
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from enum import Enum
from loguru import logger
import threading


class LogType(Enum):
    """日志类型"""
    MARKET = "market"           # 行情日志
    SIGNAL = "signal"           # 信号日志
    ORDER = "order"             # 下单日志
    RISK = "risk"               # 风控拦截日志
    ERROR = "error"             # 报错日志
    SYSTEM = "system"           # 系统日志
    POSITION = "position"       # 持仓变更日志
    PNL = "pnl"                 # 盈亏日志


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogPersistenceManager:
    """
    全量日志持久化管理器
    
    核心设计：
    - 内存缓冲 + 异步批量写入
    - 分类存储，便于查询
    - SQLite主存储 + 文件备份
    - 自动过期清理
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        log_config = config.get("monitoring", {}).get("log_persistence", {})
        self._enabled = log_config.get("enabled", True)
        self._db_path = log_config.get("db_path", "./data/trading_logs.db")
        self._log_dir = log_config.get("log_dir", "./logs")
        self._retention_days = log_config.get("retention_days", 30)
        self._batch_size = log_config.get("batch_size", 50)
        self._flush_interval = log_config.get("flush_interval_seconds", 5)
        
        # 内存缓冲 {log_type: deque of log_entries}
        self._buffers: Dict[str, deque] = {lt.value: deque() for lt in LogType}
        
        # 运行状态
        self._running = False
        self._flush_task = None
        
        # 锁
        self._buffer_lock = threading.Lock()
        
        # 统计信息
        self._stats = {
            "total_logged": 0,
            "total_persisted": 0,
            "by_type": {lt.value: 0 for lt in LogType},
            "last_flush": 0.0,
            "buffer_size": 0,
        }
        
        # 确保目录存在
        os.makedirs(self._log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        
        # 初始化数据库
        self._init_database()
        
        logger.info(f"LogPersistenceManager initialized: "
                   f"db_path={self._db_path}, retention={self._retention_days}days")

    def _init_database(self):
        """初始化数据库表结构"""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            # 行情日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    event_type TEXT,
                    price REAL,
                    volume REAL,
                    data TEXT,
                    created_at REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_ts ON market_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_symbol ON market_logs(symbol)")
            
            # 信号日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    strategy TEXT,
                    signal_type TEXT,
                    direction TEXT,
                    price REAL,
                    quality REAL,
                    data TEXT,
                    created_at REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_ts ON signal_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_symbol ON signal_logs(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signal_strategy ON signal_logs(strategy)")
            
            # 下单日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS order_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    order_id TEXT,
                    client_order_id TEXT,
                    side TEXT,
                    order_type TEXT,
                    price REAL,
                    quantity REAL,
                    status TEXT,
                    strategy TEXT,
                    data TEXT,
                    created_at REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_ts ON order_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_symbol ON order_logs(symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_id ON order_logs(order_id)")
            
            # 风控拦截日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS risk_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT,
                    risk_type TEXT,
                    risk_level TEXT,
                    action TEXT,
                    reason TEXT,
                    data TEXT,
                    created_at REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_ts ON risk_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_risk_type ON risk_logs(risk_type)")
            
            # 报错日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS error_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    module TEXT,
                    error_type TEXT,
                    message TEXT,
                    traceback TEXT,
                    severity TEXT,
                    data TEXT,
                    created_at REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_ts ON error_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_severity ON error_logs(severity)")
            
            # 系统日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    module TEXT,
                    event TEXT,
                    level TEXT,
                    message TEXT,
                    data TEXT,
                    created_at REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_system_ts ON system_logs(timestamp)")
            
            # 持仓变更日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS position_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT NOT NULL,
                    pos_side TEXT,
                    old_qty REAL,
                    new_qty REAL,
                    old_avg_price REAL,
                    new_avg_price REAL,
                    pnl REAL,
                    reason TEXT,
                    data TEXT,
                    created_at REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_position_ts ON position_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_position_symbol ON position_logs(symbol)")
            
            # 盈亏日志表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pnl_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    symbol TEXT,
                    strategy TEXT,
                    realized_pnl REAL,
                    unrealized_pnl REAL,
                    total_pnl REAL,
                    data TEXT,
                    created_at REAL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnl_ts ON pnl_logs(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pnl_symbol ON pnl_logs(symbol)")
            
            conn.commit()
            conn.close()
            
            logger.info("Log database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize log database: {e}")

    async def start(self):
        """启动日志持久化服务"""
        if self._running:
            return
        
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        
        logger.info("LogPersistenceManager started")

    async def stop(self):
        """停止日志持久化服务"""
        self._running = False
        
        # 最后一次flush
        await self._flush_all()
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        logger.info("LogPersistenceManager stopped")

    def log_market(self, symbol: str, event_type: str, price: float = 0, 
                   volume: float = 0, data: Dict[str, Any] = None):
        """记录行情日志"""
        self._add_log(LogType.MARKET, {
            "symbol": symbol,
            "event_type": event_type,
            "price": price,
            "volume": volume,
            "data": json.dumps(data) if data else None,
        })

    def log_signal(self, symbol: str, strategy: str, signal_type: str, 
                   direction: str = "", price: float = 0, quality: float = 0,
                   data: Dict[str, Any] = None):
        """记录信号日志"""
        self._add_log(LogType.SIGNAL, {
            "symbol": symbol,
            "strategy": strategy,
            "signal_type": signal_type,
            "direction": direction,
            "price": price,
            "quality": quality,
            "data": json.dumps(data) if data else None,
        })

    def log_order(self, symbol: str, order_id: str = "", client_order_id: str = "",
                  side: str = "", order_type: str = "", price: float = 0,
                  quantity: float = 0, status: str = "", strategy: str = "",
                  data: Dict[str, Any] = None):
        """记录下单日志"""
        self._add_log(LogType.ORDER, {
            "symbol": symbol,
            "order_id": order_id,
            "client_order_id": client_order_id,
            "side": side,
            "order_type": order_type,
            "price": price,
            "quantity": quantity,
            "status": status,
            "strategy": strategy,
            "data": json.dumps(data) if data else None,
        })

    def log_risk(self, risk_type: str, risk_level: str, action: str, 
                 reason: str = "", symbol: str = "", data: Dict[str, Any] = None):
        """记录风控拦截日志"""
        self._add_log(LogType.RISK, {
            "symbol": symbol,
            "risk_type": risk_type,
            "risk_level": risk_level,
            "action": action,
            "reason": reason,
            "data": json.dumps(data) if data else None,
        })

    def log_error(self, module: str, error_type: str, message: str,
                  traceback: str = "", severity: str = "error",
                  data: Dict[str, Any] = None):
        """记录报错日志"""
        self._add_log(LogType.ERROR, {
            "module": module,
            "error_type": error_type,
            "message": message,
            "traceback": traceback,
            "severity": severity,
            "data": json.dumps(data) if data else None,
        })

    def log_system(self, module: str, event: str, message: str = "",
                   level: str = "info", data: Dict[str, Any] = None):
        """记录系统日志"""
        self._add_log(LogType.SYSTEM, {
            "module": module,
            "event": event,
            "message": message,
            "level": level,
            "data": json.dumps(data) if data else None,
        })

    def log_position(self, symbol: str, pos_side: str, old_qty: float, new_qty: float,
                     old_avg_price: float = 0, new_avg_price: float = 0,
                     pnl: float = 0, reason: str = "", data: Dict[str, Any] = None):
        """记录持仓变更日志"""
        self._add_log(LogType.POSITION, {
            "symbol": symbol,
            "pos_side": pos_side,
            "old_qty": old_qty,
            "new_qty": new_qty,
            "old_avg_price": old_avg_price,
            "new_avg_price": new_avg_price,
            "pnl": pnl,
            "reason": reason,
            "data": json.dumps(data) if data else None,
        })

    def log_pnl(self, symbol: str = "", strategy: str = "",
                realized_pnl: float = 0, unrealized_pnl: float = 0,
                total_pnl: float = 0, data: Dict[str, Any] = None):
        """记录盈亏日志"""
        self._add_log(LogType.PNL, {
            "symbol": symbol,
            "strategy": strategy,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "data": json.dumps(data) if data else None,
        })

    def _add_log(self, log_type: LogType, entry: Dict[str, Any]):
        """添加日志到缓冲"""
        if not self._enabled:
            return
        
        now = time.time()
        entry["timestamp"] = now
        entry["created_at"] = now
        
        with self._buffer_lock:
            self._buffers[log_type.value].append(entry)
            self._stats["total_logged"] += 1
            self._stats["by_type"][log_type.value] += 1
            
            # 计算当前缓冲总大小
            self._stats["buffer_size"] = sum(
                len(buf) for buf in self._buffers.values()
            )

    async def _flush_loop(self):
        """定时刷新循环"""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_all()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in log flush loop: {e}")

    async def _flush_all(self):
        """刷新所有缓冲到数据库"""
        # 收集所有需要写入的数据
        to_flush = {}
        with self._buffer_lock:
            for log_type, buffer in self._buffers.items():
                if buffer:
                    to_flush[log_type] = list(buffer)
                    buffer.clear()
            
            self._stats["buffer_size"] = 0
        
        if not to_flush:
            return
        
        # 异步写入数据库
        await asyncio.to_thread(self._write_to_db, to_flush)
        
        self._stats["last_flush"] = time.time()

    def _write_to_db(self, to_flush: Dict[str, List[Dict[str, Any]]]):
        """写入数据库"""
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            table_map = {
                "market": ("market_logs", ["timestamp", "symbol", "event_type", "price", "volume", "data", "created_at"]),
                "signal": ("signal_logs", ["timestamp", "symbol", "strategy", "signal_type", "direction", "price", "quality", "data", "created_at"]),
                "order": ("order_logs", ["timestamp", "symbol", "order_id", "client_order_id", "side", "order_type", "price", "quantity", "status", "strategy", "data", "created_at"]),
                "risk": ("risk_logs", ["timestamp", "symbol", "risk_type", "risk_level", "action", "reason", "data", "created_at"]),
                "error": ("error_logs", ["timestamp", "module", "error_type", "message", "traceback", "severity", "data", "created_at"]),
                "system": ("system_logs", ["timestamp", "module", "event", "level", "message", "data", "created_at"]),
                "position": ("position_logs", ["timestamp", "symbol", "pos_side", "old_qty", "new_qty", "old_avg_price", "new_avg_price", "pnl", "reason", "data", "created_at"]),
                "pnl": ("pnl_logs", ["timestamp", "symbol", "strategy", "realized_pnl", "unrealized_pnl", "total_pnl", "data", "created_at"]),
            }
            
            total_written = 0
            
            for log_type, entries in to_flush.items():
                if log_type not in table_map:
                    continue
                
                table_name, columns = table_map[log_type]
                placeholders = ",".join(["?"] * len(columns))
                col_names = ",".join(columns)
                
                rows = []
                for entry in entries:
                    row = [entry.get(col) for col in columns]
                    rows.append(row)
                
                if rows:
                    cursor.executemany(
                        f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})",
                        rows
                    )
                    total_written += len(rows)
            
            conn.commit()
            conn.close()
            
            self._stats["total_persisted"] += total_written
            
            # 同时写入文件备份（按天）
            self._write_file_backup(to_flush)
            
        except Exception as e:
            logger.error(f"Error writing logs to DB: {e}")

    def _write_file_backup(self, to_flush: Dict[str, List[Dict[str, Any]]]):
        """写入文件备份"""
        try:
            today = datetime.now().strftime("%Y%m%d")
            
            for log_type, entries in to_flush.items():
                if not entries:
                    continue
                
                filename = f"{log_type}_{today}.log"
                filepath = os.path.join(self._log_dir, filename)
                
                with open(filepath, "a", encoding="utf-8") as f:
                    for entry in entries:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        
        except Exception as e:
            logger.error(f"Error writing file backup: {e}")

    def query_logs(self, log_type: LogType, symbol: str = "", 
                   start_time: float = None, end_time: float = None,
                   limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """查询日志"""
        table_map = {
            LogType.MARKET: "market_logs",
            LogType.SIGNAL: "signal_logs",
            LogType.ORDER: "order_logs",
            LogType.RISK: "risk_logs",
            LogType.ERROR: "error_logs",
            LogType.SYSTEM: "system_logs",
            LogType.POSITION: "position_logs",
            LogType.PNL: "pnl_logs",
        }
        
        table_name = table_map.get(log_type)
        if not table_name:
            return []
        
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = f"SELECT * FROM {table_name} WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
            
            conn.close()
            return rows
            
        except Exception as e:
            logger.error(f"Error querying logs: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "enabled": self._enabled,
            "retention_days": self._retention_days,
            "db_path": self._db_path,
            "log_dir": self._log_dir,
        }

    async def cleanup_old_logs(self):
        """清理过期日志"""
        cutoff = time.time() - self._retention_days * 86400
        
        tables = [
            "market_logs", "signal_logs", "order_logs", "risk_logs",
            "error_logs", "system_logs", "position_logs", "pnl_logs"
        ]
        
        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.cursor()
            
            total_deleted = 0
            for table in tables:
                cursor.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,))
                total_deleted += cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logger.info(f"Cleaned up {total_deleted} old log entries")
            
        except Exception as e:
            logger.error(f"Error cleaning up old logs: {e}")
