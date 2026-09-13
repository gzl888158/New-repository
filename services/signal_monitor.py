"""
信号监控模块，追踪信号从接收到成交的完整生命周期与最终结果。
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List
from loguru import logger
import numpy as np
import sqlite3
import json
import os
import time
import threading


class SignalOutcome(Enum):
    """信号结果类型"""
    EXECUTED = "executed"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"


class SignalLifecycle:
    """信号生命周期追踪"""
    def __init__(self, signal_id: str, fingerprint: str = ""):
        self.signal_id = signal_id
        self.fingerprint = fingerprint
        self.created_at = datetime.now()
        self.received_at = None
        self.validated_at = None
        self.quality_checked_at = None
        self.executed_at = None
        self.filled_at = None
        self.completed_at = None

        self.quality_score = 0.0
        self.quality_breakdown: Dict[str, Any] = {}
        self.confidence = 0.5
        self.expected_pnl = 0.0
        self.actual_pnl = 0.0
        self.actual_fee = 0.0
        self.duration_ms = 0

        self.outcome = None
        self.outcome_reason = ""

        self.symbol = ""
        self.strategy_name = ""
        self.direction = ""
        self.quantity = 0.0
        self.price = 0.0
        self.leverage = 1
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None

        # 信号溯源追踪：完整链路步骤
        self.trace_steps: List[Dict[str, Any]] = []
        # 信号来源上下文（策略/规则/触发条件）
        self.source_context: Dict[str, Any] = {}

    def mark_step(self, step: str, status: str = "ok", detail: Any = None):
        """记录任意链路步骤，用于完整溯源"""
        self.trace_steps.append({
            "step": step,
            "status": status,
            "detail": detail,
            "ts": datetime.now().isoformat(),
        })

    def mark_received(self):
        self.received_at = datetime.now()
        self.mark_step("received", "ok")

    def mark_validated(self, signal_data: Dict[str, Any]):
        self.validated_at = datetime.now()
        self.symbol = signal_data.get("symbol", "")
        self.strategy_name = signal_data.get("strategy_name", "")
        self.direction = signal_data.get("direction", "")
        self.quantity = signal_data.get("quantity", 0.0)
        self.price = signal_data.get("price", 0.0)
        self.leverage = signal_data.get("leverage", 1)
        self.confidence = signal_data.get("confidence", 0.5)
        sl = signal_data.get("stop_loss")
        tp = signal_data.get("take_profit")
        self.stop_loss = float(sl) if sl else None
        self.take_profit = float(tp) if tp else None
        self.source_context = {
            "source": signal_data.get("source", ""),
            "signal_type": signal_data.get("signal_type", ""),
            "timeframe": signal_data.get("timeframe", ""),
        }
        self.mark_step("validated", "ok", {
            "symbol": self.symbol,
            "strategy": self.strategy_name,
            "direction": self.direction,
        })

    def mark_quality_checked(self, quality_score: float, breakdown: Dict[str, Any] = None):
        self.quality_checked_at = datetime.now()
        self.quality_score = quality_score
        if breakdown:
            self.quality_breakdown = breakdown
        self.mark_step("quality_checked", "ok", {"score": quality_score})

    def mark_executed(self, order_id: str = ""):
        self.executed_at = datetime.now()
        self.mark_step("executed", "ok", {"order_id": order_id} if order_id else None)

    def mark_filled(self, pnl: float = 0.0, fee: float = 0.0):
        self.filled_at = datetime.now()
        self.actual_pnl = pnl
        self.actual_fee = fee
        self.outcome = SignalOutcome.FILLED
        self.mark_step("filled", "ok", {"pnl": pnl, "fee": fee})

    def mark_completed(self, outcome: SignalOutcome, reason: str = ""):
        self.completed_at = datetime.now()
        self.outcome = outcome
        self.outcome_reason = reason
        if self.created_at and self.completed_at:
            self.duration_ms = (self.completed_at - self.created_at).total_seconds() * 1000
        self.mark_step("completed", outcome.value, reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "fingerprint": self.fingerprint,
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "direction": self.direction,
            "quantity": self.quantity,
            "price": self.price,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "quality_breakdown": self.quality_breakdown,
            "expected_pnl": self.expected_pnl,
            "actual_pnl": self.actual_pnl,
            "actual_fee": self.actual_fee,
            "outcome": self.outcome.value if self.outcome else None,
            "outcome_reason": self.outcome_reason,
            "duration_ms": self.duration_ms,
            "source_context": self.source_context,
            "trace_steps": self.trace_steps,
            "timestamps": {
                "created": self.created_at.isoformat() if self.created_at else None,
                "received": self.received_at.isoformat() if self.received_at else None,
                "validated": self.validated_at.isoformat() if self.validated_at else None,
                "quality_checked": self.quality_checked_at.isoformat() if self.quality_checked_at else None,
                "executed": self.executed_at.isoformat() if self.executed_at else None,
                "filled": self.filled_at.isoformat() if self.filled_at else None,
                "completed": self.completed_at.isoformat() if self.completed_at else None,
            },
        }


class SignalMonitor:
    """信号监控与反馈系统"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._signal_lifecycles: Dict[str, SignalLifecycle] = {}
        self._signal_history: List[Dict[str, Any]] = []
        self._max_history_size = 500
        self._stats_cache: Dict[str, Any] = {}
        self._stats_cache_time = None

        # 信号去重：fingerprint -> [{"signal_id", "ts"}]
        self._fingerprint_cache: Dict[str, List[Dict[str, Any]]] = {}
        dedup_cfg = (config or {}).get("signal_dedup", {}) if isinstance(config, dict) else {}
        self._dedup_window_sec = int(dedup_cfg.get("window_sec", 60))
        self._dedup_max_per_window = int(dedup_cfg.get("max_per_window", 3))

        # 信号持久化
        self._db_path = (config or {}).get("signal_db_path", "./data/trading.db") if isinstance(config, dict) else "./data/trading.db"
        self._db_enabled = bool((config or {}).get("signal_persist_enabled", True)) if isinstance(config, dict) else True
        self._init_db()

        logger.info("SignalMonitor initialized (dedup_window={}s, db={})".format(self._dedup_window_sec, self._db_path if self._db_enabled else "off"))

    def track_signal(self, signal_id: str, fingerprint: str = "") -> SignalLifecycle:
        """开始追踪一个信号"""
        lifecycle = SignalLifecycle(signal_id, fingerprint=fingerprint)
        self._signal_lifecycles[signal_id] = lifecycle

        # 注册指纹用于去重
        if fingerprint:
            now = datetime.now()
            entries = self._fingerprint_cache.setdefault(fingerprint, [])
            entries.append({"signal_id": signal_id, "ts": now})
            # 清理过期条目
            cutoff = now.timestamp() - self._dedup_window_sec
            self._fingerprint_cache[fingerprint] = [
                e for e in entries if e["ts"].timestamp() >= cutoff
            ]

        return lifecycle

    def check_duplicate(self, fingerprint: str) -> Dict[str, Any]:
        """检查信号是否重复
        返回:
          - is_duplicate: 是否判定为重复
          - recent_count: 窗口内同指纹数量
          - window_sec: 窗口秒数
          - first_ts: 窗口内最早出现时间
        """
        if not fingerprint:
            return {"is_duplicate": False, "recent_count": 0, "window_sec": self._dedup_window_sec, "first_ts": None}

        now = datetime.now()
        entries = self._fingerprint_cache.get(fingerprint, [])
        cutoff = now.timestamp() - self._dedup_window_sec
        recent = [e for e in entries if e["ts"].timestamp() >= cutoff]

        # 更新缓存为清理后的列表
        self._fingerprint_cache[fingerprint] = recent

        is_dup = len(recent) >= self._dedup_max_per_window
        return {
            "is_duplicate": is_dup,
            "recent_count": len(recent),
            "window_sec": self._dedup_window_sec,
            "max_per_window": self._dedup_max_per_window,
            "first_ts": recent[0]["ts"].isoformat() if recent else None,
        }
    
    def get_lifecycle(self, signal_id: str) -> Optional[SignalLifecycle]:
        """获取信号生命周期"""
        return self._signal_lifecycles.get(signal_id)

    def complete_signal(self, signal_id: str, outcome: SignalOutcome, reason: str = ""):
        """标记信号完成"""
        lifecycle = self._signal_lifecycles.get(signal_id)
        if lifecycle:
            lifecycle.mark_completed(outcome, reason)
            self._record_history(lifecycle)
            # 更新数据库中的结果
            self._update_signal_outcome_db(lifecycle)
            self._signal_lifecycles.pop(signal_id, None)

    def _record_history(self, lifecycle: SignalLifecycle):
        """记录信号历史"""
        record = lifecycle.to_dict()
        self._signal_history.append(record)

        if len(self._signal_history) > self._max_history_size:
            self._signal_history = self._signal_history[-self._max_history_size:]

        # 持久化到数据库
        self._persist_signal_db(lifecycle)

    # ---------------- 数据库持久化 ----------------
    def _init_db(self):
        """初始化 trading_signals 表"""
        if not self._db_enabled:
            return
        try:
            os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        except Exception:
            pass
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    signal_id TEXT PRIMARY KEY,
                    fingerprint TEXT,
                    timestamp TEXT,
                    strategy TEXT,
                    strategy_name TEXT,
                    symbol TEXT,
                    side TEXT,
                    direction TEXT,
                    signal_type TEXT,
                    source TEXT,
                    price REAL,
                    quantity REAL,
                    leverage INTEGER,
                    stop_loss REAL,
                    take_profit REAL,
                    confidence REAL,
                    quality_score REAL,
                    quality_breakdown TEXT,
                    status TEXT,
                    outcome TEXT,
                    outcome_reason TEXT,
                    executed INTEGER DEFAULT 0,
                    actual_pnl REAL DEFAULT 0,
                    actual_fee REAL DEFAULT 0,
                    duration_ms REAL DEFAULT 0,
                    metadata TEXT,
                    trace_steps TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON trading_signals(timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON trading_signals(symbol)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_strategy ON trading_signals(strategy_name)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_signals_fingerprint ON trading_signals(fingerprint)')
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"SignalMonitor _init_db failed: {e}")
            self._db_enabled = False

    # ---------------- DB 连接管理 ----------------
    _DB_LOCK_RETRY_COUNT = 3
    _DB_LOCK_RETRY_DELAY = 0.05  # 50ms
    _DB_CONNECTION_TIMEOUT = 5.0  # 5秒连接超时
    _db_write_lock = threading.Lock()

    def _get_db_connection(self) -> Optional[sqlite3.Connection]:
        """获取带超时和WAL模式的数据库连接"""
        if not self._db_enabled:
            return None
        try:
            conn = sqlite3.connect(
                self._db_path,
                timeout=self._DB_CONNECTION_TIMEOUT,
                isolation_level=None  # 自动提交模式
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=3000")
            conn.execute("PRAGMA synchronous=NORMAL")
            return conn
        except sqlite3.OperationalError as e:
            logger.warning(f"SignalMonitor DB connection failed (operational): {e}")
            return None
        except Exception as e:
            logger.warning(f"SignalMonitor DB connection failed: {e}")
            return None

    def _safe_close_db(self, conn: Optional[sqlite3.Connection]) -> None:
        """安全关闭数据库连接"""
        if conn is None:
            return
        try:
            conn.close()
        except Exception:
            pass

    def _safe_db_execute(self, operation: str, params: tuple = (),
                         write_op: bool = False) -> bool:
        """
        安全执行数据库操作，支持自动重试

        Args:
            operation: SQL语句
            params: 参数元组
            write_op: 是否为写操作（需要锁保护）

        Returns:
            是否执行成功
        """
        if not self._db_enabled:
            return False

        lock = self._db_write_lock if write_op else None
        if lock:
            lock.acquire()

        conn = None
        last_error = None
        try:
            for attempt in range(self._DB_LOCK_RETRY_COUNT):
                conn = self._get_db_connection()
                if conn is None:
                    return False
                try:
                    conn.execute(operation, params)
                    return True
                except sqlite3.OperationalError as e:
                    last_error = e
                    error_msg = str(e).lower()
                    if "locked" in error_msg or "busy" in error_msg:
                        if attempt < self._DB_LOCK_RETRY_COUNT - 1:
                            time.sleep(self._DB_LOCK_RETRY_DELAY * (attempt + 1))
                            continue
                    logger.debug(f"SignalMonitor DB execute error (attempt {attempt + 1}): {e}")
                    return False
                except sqlite3.DatabaseError as e:
                    error_msg = str(e).lower()
                    if "corrupt" in error_msg or "malformed" in error_msg:
                        logger.warning(f"SignalMonitor DB corruption detected: {e}")
                        self._handle_db_corruption()
                        return False
                    logger.debug(f"SignalMonitor DB error: {e}")
                    return False
            logger.debug(f"SignalMonitor DB execute failed after {self._DB_LOCK_RETRY_COUNT} retries: {last_error}")
            return False
        finally:
            self._safe_close_db(conn)
            if lock:
                lock.release()

    def _handle_db_corruption(self) -> None:
        """处理数据库损坏：备份并重建"""
        try:
            corrupt_path = self._db_path + ".corrupted." + datetime.now().strftime("%Y%m%d_%H%M%S")
            if os.path.exists(self._db_path):
                os.rename(self._db_path, corrupt_path)
                logger.warning(f"Corrupted DB moved to {corrupt_path}")
            self._db_enabled = True
            self._init_db()
        except Exception as e:
            logger.error(f"SignalMonitor DB corruption recovery failed: {e}")
            self._db_enabled = False

    def _validate_signal_data(self, lifecycle: SignalLifecycle) -> bool:
        """验证信号数据完整性，防止无效数据写入DB"""
        if not lifecycle or not lifecycle.signal_id:
            return False
        if lifecycle.quantity < 0:
            logger.debug(f"SignalMonitor rejecting negative quantity: {lifecycle.signal_id}")
            return False
        if lifecycle.price < 0:
            logger.debug(f"SignalMonitor rejecting negative price: {lifecycle.signal_id}")
            return False
        return True

    def _persist_signal_db(self, lifecycle: SignalLifecycle):
        """保存/更新信号到数据库（带重试和数据验证）"""
        if not self._db_enabled:
            return
        if not self._validate_signal_data(lifecycle):
            logger.warning(f"SignalMonitor rejecting invalid signal data: {lifecycle.signal_id}")
            return

        now_iso = datetime.now().isoformat()
        outcome_str = lifecycle.outcome.value if lifecycle.outcome else ""
        executed = 1 if lifecycle.outcome == SignalOutcome.FILLED else (
            1 if lifecycle.executed_at is not None else 0
        )

        # 安全序列化 JSON 字段
        try:
            quality_json = json.dumps(lifecycle.quality_breakdown, ensure_ascii=False) if lifecycle.quality_breakdown else "{}"
            metadata_json = json.dumps(lifecycle.source_context, ensure_ascii=False) if lifecycle.source_context else "{}"
            trace_json = json.dumps(lifecycle.trace_steps, ensure_ascii=False) if lifecycle.trace_steps else "[]"
        except (TypeError, ValueError) as e:
            logger.debug(f"SignalMonitor JSON serialization failed for {lifecycle.signal_id}: {e}")
            quality_json = "{}"
            metadata_json = "{}"
            trace_json = "[]"

        sql = '''
            INSERT OR REPLACE INTO trading_signals
            (signal_id, fingerprint, timestamp, strategy, strategy_name, symbol, side,
             direction, signal_type, source, price, quantity, leverage, stop_loss,
             take_profit, confidence, quality_score, quality_breakdown, status, outcome,
             outcome_reason, executed, actual_pnl, actual_fee, duration_ms, metadata,
             trace_steps, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        params = (
            lifecycle.signal_id or "",
            lifecycle.fingerprint or "",
            lifecycle.created_at.isoformat() if lifecycle.created_at else now_iso,
            lifecycle.strategy_name or "",
            lifecycle.strategy_name or "",
            lifecycle.symbol or "",
            lifecycle.direction or "",
            lifecycle.direction or "",
            str(lifecycle.source_context.get("signal_type", ""))[:50] if lifecycle.source_context else "",
            str(lifecycle.source_context.get("source", ""))[:50] if lifecycle.source_context else "",
            max(lifecycle.price, 0.0),
            max(lifecycle.quantity, 0.0),
            max(lifecycle.leverage, 1),
            lifecycle.stop_loss,
            lifecycle.take_profit,
            lifecycle.confidence,
            lifecycle.quality_score,
            quality_json,
            outcome_str or "tracked",
            outcome_str,
            lifecycle.outcome_reason or "",
            executed,
            lifecycle.actual_pnl,
            lifecycle.actual_fee,
            lifecycle.duration_ms,
            metadata_json,
            trace_json,
            lifecycle.created_at.isoformat() if lifecycle.created_at else now_iso,
            now_iso,
        )

        if not self._safe_db_execute(sql, params, write_op=True):
            logger.debug(f"SignalMonitor _persist_signal_db failed for {lifecycle.signal_id}")

    def _update_signal_outcome_db(self, lifecycle: SignalLifecycle):
        """更新信号结果（成交/失败/拒绝等），带重试保护"""
        if not self._db_enabled or not lifecycle.signal_id:
            return

        outcome_str = lifecycle.outcome.value if lifecycle.outcome else ""
        executed = 1 if lifecycle.outcome == SignalOutcome.FILLED else 0

        try:
            quality_json = json.dumps(lifecycle.quality_breakdown, ensure_ascii=False) if lifecycle.quality_breakdown else "{}"
        except (TypeError, ValueError):
            quality_json = "{}"

        sql = '''
            UPDATE trading_signals
            SET status=?, outcome=?, outcome_reason=?, executed=?,
                actual_pnl=?, actual_fee=?, duration_ms=?,
                quality_breakdown=?, updated_at=?
            WHERE signal_id=?
        '''
        params = (
            outcome_str or "completed",
            outcome_str,
            lifecycle.outcome_reason or "",
            executed,
            lifecycle.actual_pnl,
            lifecycle.actual_fee,
            lifecycle.duration_ms,
            quality_json,
            datetime.now().isoformat(),
            lifecycle.signal_id,
        )

        if not self._safe_db_execute(sql, params, write_op=True):
            logger.debug(f"SignalMonitor _update_signal_outcome_db failed for {lifecycle.signal_id}")

    def load_recent_signals(self, limit: int = 100, hours: int = 0,
                            symbol: str = "", strategy_name: str = "") -> List[Dict[str, Any]]:
        """从数据库加载历史信号，合并到内存历史中（带异常保护）"""
        if not self._db_enabled:
            return []

        conn = None
        try:
            conn = self._get_db_connection()
            if conn is None:
                return []

            conn.row_factory = sqlite3.Row
            sql = "SELECT * FROM trading_signals"
            conditions = []
            params: List[Any] = []
            if hours > 0:
                cutoff = (datetime.now().timestamp() - hours * 3600)
                cutoff_iso = datetime.fromtimestamp(cutoff).isoformat()
                conditions.append("timestamp >= ?")
                params.append(cutoff_iso)
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if strategy_name:
                conditions.append("strategy_name = ?")
                params.append(strategy_name)
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(sql, params).fetchall()

            results = []
            for row in rows:
                try:
                    qb = row["quality_breakdown"]
                    md = row["metadata"]
                    ts = row["trace_steps"]
                    results.append({
                        "signal_id": row["signal_id"],
                        "fingerprint": row["fingerprint"] or "",
                        "symbol": row["symbol"] or "",
                        "strategy_name": row["strategy_name"] or "",
                        "direction": row["direction"] or "",
                        "quantity": row["quantity"] or 0.0,
                        "price": row["price"] or 0.0,
                        "leverage": row["leverage"] or 1,
                        "confidence": row["confidence"] or 0.0,
                        "quality_score": row["quality_score"] or 0.0,
                        "quality_breakdown": json.loads(qb) if qb else {},
                        "actual_pnl": row["actual_pnl"] or 0.0,
                        "actual_fee": row["actual_fee"] or 0.0,
                        "outcome": row["outcome"] or "",
                        "outcome_reason": row["outcome_reason"] or "",
                        "duration_ms": row["duration_ms"] or 0.0,
                        "source_context": json.loads(md) if md else {},
                        "trace_steps": json.loads(ts) if ts else [],
                        "timestamps": {
                            "created": row["timestamp"],
                            "completed": row["updated_at"],
                        },
                    })
                except Exception:
                    continue
            return results
        except sqlite3.OperationalError as e:
            logger.debug(f"SignalMonitor load_recent_signals DB operational error: {e}")
            return []
        except Exception as e:
            logger.debug(f"SignalMonitor load_recent_signals failed: {e}")
            return []
        finally:
            self._safe_close_db(conn)

    def query_signals_db(self, limit: int = 50, offset: int = 0,
                         symbol: str = "", strategy_name: str = "",
                         outcome: str = "", min_quality: float = 0.0) -> Dict[str, Any]:
        """查询数据库中的信号（供 Dashboard 使用，带异常保护）"""
        if not self._db_enabled:
            return {"signals": [], "total": 0}

        conn = None
        try:
            conn = self._get_db_connection()
            if conn is None:
                return {"signals": [], "total": 0}

            conn.row_factory = sqlite3.Row
            conditions = []
            params: List[Any] = []
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            if strategy_name:
                conditions.append("strategy_name = ?")
                params.append(strategy_name)
            if outcome:
                conditions.append("outcome = ?")
                params.append(outcome)
            if min_quality > 0:
                conditions.append("quality_score >= ?")
                params.append(min_quality)
            where_clause = (" WHERE " + " AND ".join(conditions)) if conditions else ""

            total = conn.execute(f"SELECT COUNT(*) FROM trading_signals{where_clause}", params).fetchone()[0]
            sql = f"SELECT * FROM trading_signals{where_clause} ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            rows = conn.execute(sql, params + [limit, offset]).fetchall()

            signals = []
            for row in rows:
                signals.append({
                    "signal_id": row["signal_id"],
                    "fingerprint": row["fingerprint"] or "",
                    "timestamp": row["timestamp"],
                    "strategy": row["strategy_name"] or "",
                    "symbol": row["symbol"] or "",
                    "side": row["direction"] or "",
                    "direction": row["direction"] or "",
                    "price": row["price"] or 0.0,
                    "quantity": row["quantity"] or 0.0,
                    "leverage": row["leverage"] or 1,
                    "confidence": row["confidence"] or 0.0,
                    "quality_score": row["quality_score"] or 0.0,
                    "status": row["status"] or "",
                    "outcome": row["outcome"] or "",
                    "executed": bool(row["executed"]),
                    "actual_pnl": row["actual_pnl"] or 0.0,
                    "duration_ms": row["duration_ms"] or 0.0,
                })
            return {"signals": signals, "total": total}
        except sqlite3.OperationalError as e:
            logger.debug(f"SignalMonitor query_signals_db operational error: {e}")
            return {"signals": [], "total": 0, "error": str(e)}
        except Exception as e:
            logger.debug(f"SignalMonitor query_signals_db failed: {e}")
            return {"signals": [], "total": 0, "error": str(e)}
        finally:
            self._safe_close_db(conn)
    
    def get_signal_stats(self, strategy_name: str = "", symbol: str = "", limit: int = 100) -> Dict[str, Any]:
        """获取信号统计"""
        now = datetime.now()
        if self._stats_cache and (now - self._stats_cache_time).total_seconds() < 60:
            return self._stats_cache
        
        filtered = self._signal_history
        if strategy_name:
            filtered = [s for s in filtered if s["strategy_name"] == strategy_name]
        if symbol:
            filtered = [s for s in filtered if s["symbol"] == symbol]
        
        recent = filtered[-limit:]
        
        total = len(recent)
        if total == 0:
            return {
                "total_signals": 0,
                "executed": 0,
                "filled": 0,
                "failed": 0,
                "rejected": 0,
                "avg_pnl": 0.0,
                "avg_confidence": 0.0,
                "avg_quality": 0.0,
                "avg_duration_ms": 0.0,
                "win_rate": 0.0,
            }
        
        outcomes = {}
        for s in recent:
            outcome = s.get("outcome", "")
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
        
        pnls = [s["actual_pnl"] for s in recent if s["outcome"] == "filled"]
        avg_pnl = np.mean(pnls) if pnls else 0.0
        
        confidences = [s["confidence"] for s in recent]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        qualities = [s["quality_score"] for s in recent]
        avg_quality = np.mean(qualities) if qualities else 0.0
        
        durations = [s["duration_ms"] for s in recent if s["duration_ms"] > 0]
        avg_duration = np.mean(durations) if durations else 0.0
        
        wins = sum(1 for p in pnls if p > 0)
        win_rate = wins / len(pnls) if pnls else 0.0
        
        stats = {
            "total_signals": total,
            "outcomes": outcomes,
            "executed": outcomes.get("executed", 0),
            "filled": outcomes.get("filled", 0),
            "failed": outcomes.get("failed", 0),
            "rejected": outcomes.get("rejected", 0),
            "avg_pnl": round(avg_pnl, 4),
            "avg_confidence": round(avg_confidence, 4),
            "avg_quality": round(avg_quality, 4),
            "avg_duration_ms": round(avg_duration, 2),
            "win_rate": round(win_rate, 4),
            "sample_size": len(pnls),
        }
        
        self._stats_cache = stats
        self._stats_cache_time = now
        
        return stats
    
    def get_signal_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取信号历史"""
        return self._signal_history[-limit:]
    
    def get_strategy_performance(self, strategy_name: str) -> Dict[str, Any]:
        """获取策略的信号表现"""
        strategy_signals = [s for s in self._signal_history if s["strategy_name"] == strategy_name]
        
        if not strategy_signals:
            return {
                "strategy_name": strategy_name,
                "total_signals": 0,
                "performance": {},
            }
        
        filled_signals = [s for s in strategy_signals if s["outcome"] == "filled"]
        
        pnls = [s["actual_pnl"] for s in filled_signals]
        total_pnl = sum(pnls)
        avg_pnl = np.mean(pnls) if pnls else 0.0
        
        qualities = [s["quality_score"] for s in strategy_signals]
        avg_quality = np.mean(qualities) if qualities else 0.0
        
        confidences = [s["confidence"] for s in strategy_signals]
        avg_confidence = np.mean(confidences) if confidences else 0.0
        
        high_quality_signals = [s for s in filled_signals if s["quality_score"] >= 0.7]
        low_quality_signals = [s for s in filled_signals if s["quality_score"] < 0.5]
        
        high_q_pnl = [s["actual_pnl"] for s in high_quality_signals]
        low_q_pnl = [s["actual_pnl"] for s in low_quality_signals]
        
        return {
            "strategy_name": strategy_name,
            "total_signals": len(strategy_signals),
            "filled_signals": len(filled_signals),
            "total_pnl": round(total_pnl, 4),
            "avg_pnl": round(avg_pnl, 4),
            "avg_quality": round(avg_quality, 4),
            "avg_confidence": round(avg_confidence, 4),
            "high_quality": {
                "count": len(high_quality_signals),
                "avg_pnl": round(np.mean(high_q_pnl), 4) if high_q_pnl else 0.0,
            },
            "low_quality": {
                "count": len(low_quality_signals),
                "avg_pnl": round(np.mean(low_q_pnl), 4) if low_q_pnl else 0.0,
            },
        }
    
    def get_quality_threshold_analysis(self) -> Dict[str, Any]:
        """分析不同质量阈值下的信号表现"""
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        results = {}
        
        filled_signals = [s for s in self._signal_history if s["outcome"] == "filled"]
        
        for threshold in thresholds:
            above = [s for s in filled_signals if s["quality_score"] >= threshold]
            below = [s for s in filled_signals if s["quality_score"] < threshold]
            
            above_pnls = [s["actual_pnl"] for s in above]
            below_pnls = [s["actual_pnl"] for s in below]
            
            results[threshold] = {
                "above_threshold": {
                    "count": len(above),
                    "avg_pnl": round(np.mean(above_pnls), 4) if above_pnls else 0.0,
                    "win_rate": round(sum(1 for p in above_pnls if p > 0) / len(above_pnls), 4) if above_pnls else 0.0,
                },
                "below_threshold": {
                    "count": len(below),
                    "avg_pnl": round(np.mean(below_pnls), 4) if below_pnls else 0.0,
                    "win_rate": round(sum(1 for p in below_pnls if p > 0) / len(below_pnls), 4) if below_pnls else 0.0,
                },
            }
        
        return results
    
    def get_feedback_for_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """为策略提供优化反馈"""
        performance = self.get_strategy_performance(strategy_name)
        
        feedback = {
            "strategy_name": strategy_name,
            "suggestions": [],
            "warnings": [],
            "opportunities": [],
        }
        
        avg_pnl = performance["avg_pnl"]
        avg_quality = performance["avg_quality"]
        filled_count = performance["filled_signals"]
        
        if filled_count < 5:
            feedback["warnings"].append("样本量不足，建议收集更多交易数据")
        
        if avg_pnl < -0.1:
            feedback["suggestions"].append("平均盈亏为负，建议检查策略参数")
        
        if avg_quality < 0.5:
            feedback["suggestions"].append("信号质量偏低，建议优化信号生成逻辑")
        
        high_q_pnl = performance["high_quality"]["avg_pnl"]
        low_q_pnl = performance["low_quality"]["avg_pnl"]
        
        if high_q_pnl > 0 and low_q_pnl < 0:
            feedback["opportunities"].append("高信号质量表现优于低质量，建议提高质量阈值")

        if high_q_pnl > low_q_pnl * 2:
            feedback["opportunities"].append("高质量信号表现显著更好，建议过滤低质量信号")

        return feedback

    # ---------------- 转化率漏斗分析 ----------------
    def get_conversion_funnel(self, strategy_name: str = "", symbol: str = "",
                              hours: int = 24) -> Dict[str, Any]:
        """信号转化漏斗分析：生成 → 接收 → 质检通过 → 执行 → 成交 → 盈利
        统计最近 hours 小时内的信号在每一阶段的数量与转化率。
        """
        now = datetime.now()
        cutoff = now.timestamp() - hours * 3600

        records = []
        for s in self._signal_history:
            try:
                ts_str = s.get("timestamps", {}).get("created")
                if not ts_str:
                    continue
                ts = datetime.fromisoformat(ts_str).timestamp()
                if ts < cutoff:
                    continue
                if strategy_name and s.get("strategy_name") != strategy_name:
                    continue
                if symbol and s.get("symbol") != symbol:
                    continue
                records.append(s)
            except Exception:
                continue

        total = len(records)
        received = sum(1 for s in records if s.get("timestamps", {}).get("received"))
        quality_passed = sum(1 for s in records if s.get("quality_score", 0) >= 0.35)
        executed = sum(1 for s in records if s.get("timestamps", {}).get("executed"))
        filled = sum(1 for s in records if s.get("outcome") == "filled")
        profitable = sum(1 for s in records if s.get("outcome") == "filled" and s.get("actual_pnl", 0) > 0)

        def rate(a, b):
            return round(a / b, 4) if b else 0.0

        funnel = [
            {"stage": "generated", "count": total, "conv_rate": 1.0},
            {"stage": "received", "count": received, "conv_rate": rate(received, total)},
            {"stage": "quality_passed", "count": quality_passed, "conv_rate": rate(quality_passed, received)},
            {"stage": "executed", "count": executed, "conv_rate": rate(executed, quality_passed)},
            {"stage": "filled", "count": filled, "conv_rate": rate(filled, executed)},
            {"stage": "profitable", "count": profitable, "conv_rate": rate(profitable, filled)},
        ]

        # 识别漏斗中最大的流失环节
        biggest_drop = None
        biggest_drop_rate = 0.0
        for i in range(1, len(funnel)):
            prev = funnel[i - 1]
            cur = funnel[i]
            if prev["count"] > 0:
                drop_rate = (prev["count"] - cur["count"]) / prev["count"]
                if drop_rate > biggest_drop_rate:
                    biggest_drop_rate = drop_rate
                    biggest_drop = {"from": prev["stage"], "to": cur["stage"],
                                    "drop_rate": round(drop_rate, 4)}

        return {
            "window_hours": hours,
            "filter": {"strategy_name": strategy_name, "symbol": symbol},
            "funnel": funnel,
            "overall_conv_rate": rate(profitable, total),
            "biggest_drop": biggest_drop,
            "summary": {
                "total": total,
                "filled": filled,
                "profitable": profitable,
                "win_rate": rate(profitable, filled),
            },
        }

    # ---------------- 信号溯源追踪 ----------------
    def get_signal_trace(self, signal_id: str) -> Dict[str, Any]:
        """获取信号完整链路追踪"""
        # 优先从活跃 lifecycle 取（含 trace_steps）
        lifecycle = self._signal_lifecycles.get(signal_id)
        if lifecycle:
            data = lifecycle.to_dict()
            data["is_active"] = True
            return data

        # 从历史记录查找
        for s in self._signal_history:
            if s.get("signal_id") == signal_id:
                return {**s, "is_active": False}

        return {"signal_id": signal_id, "found": False}

    def search_traces(self, symbol: str = "", strategy_name: str = "",
                      direction: str = "", outcome: str = "",
                      limit: int = 20) -> List[Dict[str, Any]]:
        """按条件搜索信号溯源记录"""
        results = []
        for s in reversed(self._signal_history):
            if symbol and s.get("symbol") != symbol:
                continue
            if strategy_name and s.get("strategy_name") != strategy_name:
                continue
            if direction and s.get("direction") != direction:
                continue
            if outcome and s.get("outcome") != outcome:
                continue
            results.append(s)
            if len(results) >= limit:
                break
        return results

    # ---------------- 信号归因分析 ----------------
    def get_attribution_analysis(self, strategy_name: str = "",
                                 min_samples: int = 5) -> Dict[str, Any]:
        """分析各质量因子对盈利的贡献度
        基于 quality_breakdown 中的因子得分与 actual_pnl 的相关性。
        """
        filled = [s for s in self._signal_history
                  if s.get("outcome") == "filled"
                  and s.get("quality_breakdown")
                  and s.get("actual_pnl") is not None]
        if strategy_name:
            filled = [s for s in filled if s.get("strategy_name") == strategy_name]

        if len(filled) < min_samples:
            return {
                "strategy_name": strategy_name or "all",
                "sample_size": len(filled),
                "min_samples_required": min_samples,
                "message": "样本不足，无法进行归因分析",
                "factor_attribution": {},
            }

        # 收集所有因子名
        factor_names = set()
        for s in filled:
            fb = s.get("quality_breakdown", {})
            factor_scores = fb.get("factor_scores", {}) if isinstance(fb, dict) else {}
            factor_names.update(factor_scores.keys())

        pnls = np.array([s.get("actual_pnl", 0) for s in filled])
        wins = (pnls > 0).astype(float)

        attribution = {}
        for fn in factor_names:
            scores = []
            for s in filled:
                fb = s.get("quality_breakdown", {})
                factor_scores = fb.get("factor_scores", {}) if isinstance(fb, dict) else {}
                scores.append(factor_scores.get(fn, 0.5))
            scores_arr = np.array(scores)

            try:
                if len(scores_arr) > 1 and np.std(scores_arr) > 0 and np.std(pnls) > 0:
                    corr_pnl = float(np.corrcoef(scores_arr, pnls)[0, 1])
                else:
                    corr_pnl = 0.0
            except Exception:
                corr_pnl = 0.0

            # 高分组的胜率 vs 低分组的胜率
            median = float(np.median(scores_arr))
            high_mask = scores_arr >= median
            low_mask = ~high_mask
            high_win = float(wins[high_mask].mean()) if high_mask.any() else 0.0
            low_win = float(wins[low_mask].mean()) if low_mask.any() else 0.0

            attribution[fn] = {
                "corr_with_pnl": round(corr_pnl, 4),
                "high_score_win_rate": round(high_win, 4),
                "low_score_win_rate": round(low_win, 4),
                "win_rate_lift": round(high_win - low_win, 4),
                "avg_score": round(float(scores_arr.mean()), 4),
                "samples": int(len(scores_arr)),
            }

        # 按胜率提升排序
        ranked = sorted(attribution.items(), key=lambda x: x[1]["win_rate_lift"], reverse=True)
        top_factors = [{"factor": k, **v} for k, v in ranked[:5]]

        return {
            "strategy_name": strategy_name or "all",
            "sample_size": len(filled),
            "factor_attribution": attribution,
            "top_factors_by_win_lift": top_factors,
            "summary": {
                "best_factor": ranked[0][0] if ranked else None,
                "best_win_rate_lift": ranked[0][1]["win_rate_lift"] if ranked else 0.0,
            },
        }

    # ---------------- 去重统计 ----------------
    def get_dedup_stats(self) -> Dict[str, Any]:
        """获取信号去重统计"""
        now = datetime.now()
        cutoff = now.timestamp() - self._dedup_window_sec

        active_fingerprints = 0
        total_entries = 0
        top_fingerprints = []

        for fp, entries in self._fingerprint_cache.items():
            recent = [e for e in entries if e["ts"].timestamp() >= cutoff]
            if recent:
                active_fingerprints += 1
                total_entries += len(recent)
                top_fingerprints.append({"fingerprint": fp, "count": len(recent)})

        top_fingerprints.sort(key=lambda x: x["count"], reverse=True)
        return {
            "window_sec": self._dedup_window_sec,
            "max_per_window": self._dedup_max_per_window,
            "active_fingerprints": active_fingerprints,
            "total_recent_signals": total_entries,
            "top_fingerprints": top_fingerprints[:10],
        }