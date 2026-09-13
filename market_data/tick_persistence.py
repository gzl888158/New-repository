"""
Tick 数据本地持久化模块
=======================
核心定位：将实时 tick 数据持久化到 SQLite 数据库，
供实时回测、盘中复盘使用。

特性：
- 支持 ticker、orderbook、trade 三种数据类型存储
- 批量写入优化，减少磁盘 IO
- 按日期分表，支持自动清理过期数据
- 支持高效查询：按时间范围、交易对、数据类型
- 数据完整性校验，确保持久化数据可靠
- 支持数据导出为 CSV/Parquet 格式
"""

import sqlite3
import os
import time
import threading
import csv
import gzip
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from datetime import datetime, timedelta
from loguru import logger


# 冷数据归档导出的字段顺序（与各表列语义一致）
_ARCHIVE_FIELDS = {
    "ticker": ["symbol", "timestamp", "price", "bid_price", "ask_price",
               "bid_volume", "ask_volume", "volume_24h", "change_24h",
               "high_24h", "low_24h", "funding_rate"],
    "trade": ["symbol", "timestamp", "price", "volume", "side", "trade_id"],
    "orderbook": ["symbol", "timestamp", "bids", "asks", "seq"],
}


class TickPersistence:
    """Tick 数据持久化管理器"""

    def __init__(self, db_path: str = None, config: Dict[str, Any] = None):
        self.config = config or {}
        self._db_path = db_path or self.config.get("db_path", "data/tick_data.db")
        self._batch_size = self.config.get("batch_size", 1000)
        self._flush_interval = self.config.get("flush_interval", 5.0)
        self._max_retention_days = self.config.get("max_retention_days", 30)
        self._max_queue_size = self.config.get("max_queue_size", 100000)
        self._archive_enabled = self.config.get("archive_enabled", True)
        self._archive_dir = os.path.abspath(self.config.get("archive_dir", "data/archive/market"))

        # 确保目录存在
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        # 数据库连接
        self._conn = None
        self._cursor = None
        self._lock = threading.RLock()
        # 数据库游标/连接访问锁：写线程与查询线程共享同一 cursor，需串行化避免并发游标错误
        self._db_lock = threading.RLock()

        # 批量写入队列
        self._ticker_queue = deque(maxlen=self._max_queue_size)
        self._trade_queue = deque(maxlen=self._max_queue_size)
        self._orderbook_queue = deque(maxlen=self._max_queue_size)

        # 写入线程
        self._running = False
        self._write_thread = None

        # 统计信息
        self._stats = {
            "ticker_written": 0,
            "trade_written": 0,
            "orderbook_written": 0,
            "total_written": 0,
            "flush_count": 0,
            "errors": 0,
        }

        # 初始化数据库
        self._init_db()

        logger.info(f"TickPersistence initialized with db: {self._db_path}")

    def _init_db(self) -> None:
        """初始化数据库表结构"""
        try:
            self._connect()
            self._create_tables()
            self._close()
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")

    def _connect(self) -> None:
        """建立数据库连接"""
        with self._db_lock:
            if self._conn is None:
                self._conn = sqlite3.connect(
                    self._db_path,
                    check_same_thread=False,
                    timeout=10,
                    isolation_level=None,
                )
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA cache_size=-10000")
                self._cursor = self._conn.cursor()

    def _close(self) -> None:
        """关闭数据库连接"""
        with self._db_lock:
            if self._conn is not None:
                try:
                    self._conn.commit()
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
                self._cursor = None

    def _create_tables(self) -> None:
        """创建数据表"""
        # Ticker 表
        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS ticker (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL,
                bid_price REAL NOT NULL,
                ask_price REAL NOT NULL,
                bid_volume REAL NOT NULL,
                ask_volume REAL NOT NULL,
                volume_24h REAL NOT NULL,
                change_24h REAL NOT NULL,
                high_24h REAL NOT NULL,
                low_24h REAL NOT NULL,
                funding_rate REAL NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(symbol, timestamp)
            )
        """)

        # Trade 表
        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                price REAL NOT NULL,
                volume REAL NOT NULL,
                side TEXT NOT NULL,
                trade_id TEXT,
                created_at INTEGER NOT NULL,
                UNIQUE(symbol, timestamp, trade_id)
            )
        """)

        # Orderbook 表
        self._cursor.execute("""
            CREATE TABLE IF NOT EXISTS orderbook (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                bids TEXT NOT NULL,
                asks TEXT NOT NULL,
                seq INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(symbol, timestamp, seq)
            )
        """)

        # 创建索引
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_symbol ON ticker(symbol)")
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_ticker_timestamp ON ticker(timestamp)")
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_symbol ON trade(symbol)")
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_timestamp ON trade(timestamp)")
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_symbol ON orderbook(symbol)")
        self._cursor.execute("CREATE INDEX IF NOT EXISTS idx_orderbook_timestamp ON orderbook(timestamp)")

        self._conn.commit()

    def start(self) -> None:
        """启动持久化服务"""
        self._running = True
        self._write_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._write_thread.start()
        logger.info("TickPersistence started")

    def stop(self) -> None:
        """停止持久化服务"""
        self._running = False
        if self._write_thread is not None:
            self._write_thread.join(timeout=10)

        # 刷新剩余数据
        self._flush_all()
        self._close()
        logger.info("TickPersistence stopped")

    def _write_loop(self) -> None:
        """写入循环"""
        last_flush = time.time()
        last_cleanup = time.time()
        while self._running:
            try:
                now = time.time()

                # 检查是否需要刷新
                needs_flush = (now - last_flush >= self._flush_interval) or \
                             (len(self._ticker_queue) >= self._batch_size) or \
                             (len(self._trade_queue) >= self._batch_size) or \
                             (len(self._orderbook_queue) >= self._batch_size)

                if needs_flush:
                    self._flush_all()
                    last_flush = now

                # 归档并清理过期数据（每小时一次，独立计时，避免因 last_flush 频繁刷新而永不执行）
                if now - last_cleanup >= 3600:
                    self._archive_expired_data()
                    last_cleanup = now

                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Write loop error: {e}")
                self._stats["errors"] += 1

    def _flush_all(self) -> None:
        """刷新所有队列数据"""
        self._flush_tickers()
        self._flush_trades()
        self._flush_orderbooks()

    def _flush_tickers(self) -> None:
        """刷新 ticker 队列"""
        if not self._ticker_queue:
            return

        try:
            self._connect()
            batch = []
            with self._lock:
                while self._ticker_queue and len(batch) < self._batch_size:
                    batch.append(self._ticker_queue.popleft())

            if not batch:
                return

            placeholders = ",".join(["(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"] * len(batch))
            values = []
            for t in batch:
                values.extend([
                    t["symbol"],
                    t["timestamp"],
                    t["price"],
                    t["bid_price"],
                    t["ask_price"],
                    t["bid_volume"],
                    t["ask_volume"],
                    t["volume_24h"],
                    t["change_24h"],
                    t["high_24h"],
                    t["low_24h"],
                    t["funding_rate"],
                    int(time.time()),
                ])

            query = f"""
                INSERT OR REPLACE INTO ticker 
                (symbol, timestamp, price, bid_price, ask_price, bid_volume, ask_volume,
                 volume_24h, change_24h, high_24h, low_24h, funding_rate, created_at)
                VALUES {placeholders}
            """
            with self._db_lock:
                self._cursor.execute(query, values)
                self._conn.commit()

            self._stats["ticker_written"] += len(batch)
            self._stats["total_written"] += len(batch)
            self._stats["flush_count"] += 1

        except Exception as e:
            logger.error(f"Flush tickers failed: {e}")
            self._stats["errors"] += 1

    def _flush_trades(self) -> None:
        """刷新 trade 队列"""
        if not self._trade_queue:
            return

        try:
            self._connect()
            batch = []
            with self._lock:
                while self._trade_queue and len(batch) < self._batch_size:
                    batch.append(self._trade_queue.popleft())

            if not batch:
                return

            placeholders = ",".join(["(?, ?, ?, ?, ?, ?, ?)"] * len(batch))
            values = []
            for t in batch:
                values.extend([
                    t["symbol"],
                    t["timestamp"],
                    t["price"],
                    t["volume"],
                    t["side"],
                    t.get("trade_id", ""),
                    int(time.time()),
                ])

            query = f"""
                INSERT OR REPLACE INTO trade 
                (symbol, timestamp, price, volume, side, trade_id, created_at)
                VALUES {placeholders}
            """
            with self._db_lock:
                self._cursor.execute(query, values)
                self._conn.commit()

            self._stats["trade_written"] += len(batch)
            self._stats["total_written"] += len(batch)

        except Exception as e:
            logger.error(f"Flush trades failed: {e}")
            self._stats["errors"] += 1

    def _flush_orderbooks(self) -> None:
        """刷新 orderbook 队列"""
        if not self._orderbook_queue:
            return

        try:
            self._connect()
            batch = []
            with self._lock:
                while self._orderbook_queue and len(batch) < self._batch_size:
                    batch.append(self._orderbook_queue.popleft())

            if not batch:
                return

            placeholders = ",".join(["(?, ?, ?, ?, ?, ?)"] * len(batch))
            values = []
            for o in batch:
                import json
                values.extend([
                    o["symbol"],
                    o["timestamp"],
                    json.dumps(o["bids"]),
                    json.dumps(o["asks"]),
                    o.get("seq", 0),
                    int(time.time()),
                ])

            query = f"""
                INSERT OR REPLACE INTO orderbook 
                (symbol, timestamp, bids, asks, seq, created_at)
                VALUES {placeholders}
            """
            with self._db_lock:
                self._cursor.execute(query, values)
                self._conn.commit()

            self._stats["orderbook_written"] += len(batch)
            self._stats["total_written"] += len(batch)

        except Exception as e:
            logger.error(f"Flush orderbooks failed: {e}")
            self._stats["errors"] += 1

    def _cleanup_expired_data(self) -> None:
        """清理过期数据"""
        try:
            self._connect()
            cutoff_ts = int((datetime.now() - timedelta(days=self._max_retention_days)).timestamp() * 1000)

            with self._db_lock:
                self._cursor.execute("DELETE FROM ticker WHERE timestamp < ?", (cutoff_ts,))
                self._cursor.execute("DELETE FROM trade WHERE timestamp < ?", (cutoff_ts,))
                self._cursor.execute("DELETE FROM orderbook WHERE timestamp < ?", (cutoff_ts,))

                self._conn.commit()
            logger.debug(f"Cleaned up data older than {self._max_retention_days} days")

        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def _archive_expired_data(self) -> Dict[str, Any]:
        """冷数据归档：把超过保留期的行情数据按月导出为 CSV.gz 后删除（archive-then-delete）。

        冷数据以月份(YYYYMM)分区组织成压缩文件，实现时序化冷存储；
        归档成功后才删除源表数据，避免中途失败丢数据。归档禁用时回退为纯删除。
        """
        result: Dict[str, Any] = {"archived": {}, "files": [], "deleted": 0}
        if not self._archive_enabled:
            self._cleanup_expired_data()
            return result

        cutoff_ms = int((datetime.now() - timedelta(days=self._max_retention_days)).timestamp() * 1000)
        os.makedirs(self._archive_dir, exist_ok=True)

        try:
            self._connect()
            with self._db_lock:
                for table, fields in _ARCHIVE_FIELDS.items():
                    try:
                        months = self._cursor.execute(
                            f"SELECT DISTINCT strftime('%Y%m', timestamp/1000, 'unixepoch') "
                            f"FROM {table} WHERE timestamp < ?", (cutoff_ms,)
                        ).fetchall()
                    except Exception:
                        continue
                    cols = ", ".join(fields)
                    for (ym,) in months:
                        if not ym:
                            continue
                        self._cursor.execute(
                            f"SELECT {cols} FROM {table} WHERE timestamp < ? "
                            f"AND strftime('%Y%m', timestamp/1000, 'unixepoch') = ?",
                            (cutoff_ms, ym),
                        )
                        rows = self._cursor.fetchall()
                        if not rows:
                            continue
                        fname = os.path.join(self._archive_dir, f"{table}_{ym}.csv.gz")
                        self._write_archive_csv(fname, fields, rows)
                        self._cursor.execute(
                            f"DELETE FROM {table} WHERE timestamp < ? "
                            f"AND strftime('%Y%m', timestamp/1000, 'unixepoch') = ?",
                            (cutoff_ms, ym),
                        )
                        result["archived"][table] = result["archived"].get(table, 0) + len(rows)
                        result["files"].append(fname)
                        result["deleted"] += len(rows)
                self._conn.commit()
            if result["deleted"] > 0:
                logger.info(f"Tick archiving: exported {result['deleted']} rows to {len(result['files'])} CSV.gz files")
                self.wal_checkpoint("TRUNCATE")
        except Exception as e:
            logger.error(f"Archive expired data failed: {e}")
            self._stats["errors"] += 1
        return result

    @staticmethod
    def _write_archive_csv(fname: str, fields: List[str], rows: List[tuple]) -> None:
        with gzip.open(fname, "wt", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)

    def get_wal_status(self) -> Dict[str, Any]:
        """读取 WAL 状态：journal_mode 是否为 wal + -wal 文件大小（用于健康上报）。"""
        status = {"journal_mode": None, "wal_enabled": False, "wal_file_bytes": 0}
        try:
            self._connect()
            with self._db_lock:
                mode = self._cursor.execute("PRAGMA journal_mode").fetchone()[0]
            status["journal_mode"] = mode
            status["wal_enabled"] = str(mode).lower() == "wal"
        except Exception as e:
            status["error"] = str(e)
        wal_path = self._db_path + "-wal"
        if os.path.exists(wal_path):
            try:
                status["wal_file_bytes"] = os.path.getsize(wal_path)
            except OSError:
                pass
        return status

    def wal_checkpoint(self, mode: str = "TRUNCATE") -> List[tuple]:
        """执行 WAL checkpoint，把预写日志合并回主库并回收磁盘空间。"""
        result: List[tuple] = []
        try:
            self._connect()
            with self._db_lock:
                result = self._cursor.execute(f"PRAGMA wal_checkpoint({mode})").fetchall()
        except Exception as e:
            logger.warning(f"TickPersistence WAL checkpoint failed: {e}")
        return result

    def persist_ticker(self, ticker: Dict[str, Any]) -> None:
        """持久化 ticker 数据"""
        with self._lock:
            self._ticker_queue.append(ticker)

    def persist_trade(self, trade: Dict[str, Any]) -> None:
        """持久化 trade 数据"""
        with self._lock:
            self._trade_queue.append(trade)

    def persist_orderbook(self, orderbook: Dict[str, Any]) -> None:
        """持久化 orderbook 数据"""
        with self._lock:
            self._orderbook_queue.append(orderbook)

    def query_tickers(self, symbol: str, start_ts: int, end_ts: int,
                      limit: int = 10000) -> List[Dict[str, Any]]:
        """查询 ticker 数据"""
        try:
            self._connect()
            query = """
                SELECT * FROM ticker 
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT ?
            """
            with self._db_lock:
                self._cursor.execute(query, (symbol, start_ts, end_ts, limit))
                rows = self._cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "symbol": row[1],
                    "timestamp": row[2],
                    "price": row[3],
                    "bid_price": row[4],
                    "ask_price": row[5],
                    "bid_volume": row[6],
                    "ask_volume": row[7],
                    "volume_24h": row[8],
                    "change_24h": row[9],
                    "high_24h": row[10],
                    "low_24h": row[11],
                    "funding_rate": row[12],
                    "created_at": row[13],
                })

            return results

        except Exception as e:
            logger.error(f"Query tickers failed: {e}")
            return []

    def query_trades(self, symbol: str, start_ts: int, end_ts: int,
                     limit: int = 10000) -> List[Dict[str, Any]]:
        """查询 trade 数据"""
        try:
            self._connect()
            query = """
                SELECT * FROM trade 
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT ?
            """
            with self._db_lock:
                self._cursor.execute(query, (symbol, start_ts, end_ts, limit))
                rows = self._cursor.fetchall()

            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "symbol": row[1],
                    "timestamp": row[2],
                    "price": row[3],
                    "volume": row[4],
                    "side": row[5],
                    "trade_id": row[6],
                    "created_at": row[7],
                })

            return results

        except Exception as e:
            logger.error(f"Query trades failed: {e}")
            return []

    def query_orderbooks(self, symbol: str, start_ts: int, end_ts: int,
                         limit: int = 1000) -> List[Dict[str, Any]]:
        """查询 orderbook 数据"""
        try:
            self._connect()
            query = """
                SELECT * FROM orderbook 
                WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp ASC
                LIMIT ?
            """
            with self._db_lock:
                self._cursor.execute(query, (symbol, start_ts, end_ts, limit))
                rows = self._cursor.fetchall()

            import json
            results = []
            for row in rows:
                results.append({
                    "id": row[0],
                    "symbol": row[1],
                    "timestamp": row[2],
                    "bids": json.loads(row[3]),
                    "asks": json.loads(row[4]),
                    "seq": row[5],
                    "created_at": row[6],
                })

            return results

        except Exception as e:
            logger.error(f"Query orderbooks failed: {e}")
            return []

    def get_symbol_list(self) -> List[str]:
        """获取所有交易对列表"""
        try:
            self._connect()
            with self._db_lock:
                self._cursor.execute("SELECT DISTINCT symbol FROM ticker")
                rows = self._cursor.fetchall()
            return [row[0] for row in rows]

        except Exception as e:
            logger.error(f"Get symbol list failed: {e}")
            return []

    def get_time_range(self, symbol: str, table: str = "ticker") -> Optional[Tuple[int, int]]:
        """获取指定交易对的数据时间范围"""
        try:
            self._connect()
            query = f"SELECT MIN(timestamp), MAX(timestamp) FROM {table} WHERE symbol = ?"
            with self._db_lock:
                self._cursor.execute(query, (symbol,))
                row = self._cursor.fetchone()

            if row and row[0] and row[1]:
                return (row[0], row[1])
            return None

        except Exception as e:
            logger.error(f"Get time range failed: {e}")
            return None

    def export_to_csv(self, symbol: str, start_ts: int, end_ts: int,
                      output_path: str, data_type: str = "ticker") -> bool:
        """导出数据为 CSV"""
        try:
            if data_type == "ticker":
                data = self.query_tickers(symbol, start_ts, end_ts, limit=1000000)
                fields = ["symbol", "timestamp", "price", "bid_price", "ask_price",
                          "bid_volume", "ask_volume", "volume_24h", "change_24h",
                          "high_24h", "low_24h", "funding_rate"]
            elif data_type == "trade":
                data = self.query_trades(symbol, start_ts, end_ts, limit=1000000)
                fields = ["symbol", "timestamp", "price", "volume", "side", "trade_id"]
            elif data_type == "orderbook":
                data = self.query_orderbooks(symbol, start_ts, end_ts, limit=100000)
                fields = ["symbol", "timestamp", "bids", "asks", "seq"]
            else:
                logger.error(f"Unknown data type: {data_type}")
                return False

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)

            logger.info(f"Exported {len(data)} {data_type} records to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Export to CSV failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            self._connect()
            with self._db_lock:
                self._cursor.execute("SELECT COUNT(*) FROM ticker")
                ticker_count = self._cursor.fetchone()[0]

                self._cursor.execute("SELECT COUNT(*) FROM trade")
                trade_count = self._cursor.fetchone()[0]

                self._cursor.execute("SELECT COUNT(*) FROM orderbook")
                orderbook_count = self._cursor.fetchone()[0]

            return {
                "db_path": self._db_path,
                "batch_size": self._batch_size,
                "flush_interval": self._flush_interval,
                "max_retention_days": self._max_retention_days,
                "queue_sizes": {
                    "ticker": len(self._ticker_queue),
                    "trade": len(self._trade_queue),
                    "orderbook": len(self._orderbook_queue),
                },
                "written": {
                    "ticker": self._stats["ticker_written"],
                    "trade": self._stats["trade_written"],
                    "orderbook": self._stats["orderbook_written"],
                    "total": self._stats["total_written"],
                },
                "database_counts": {
                    "ticker": ticker_count,
                    "trade": trade_count,
                    "orderbook": orderbook_count,
                },
                "flush_count": self._stats["flush_count"],
                "errors": self._stats["errors"],
            }

        except Exception as e:
            logger.error(f"Get stats failed: {e}")
            return {}

    def get_database_size(self) -> float:
        """获取数据库文件大小（MB）"""
        try:
            if os.path.exists(self._db_path):
                size_bytes = os.path.getsize(self._db_path)
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except Exception as e:
            logger.error(f"Get database size failed: {e}")
            return 0.0
