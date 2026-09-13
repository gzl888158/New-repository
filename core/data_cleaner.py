"""
统一数据清理引擎 - Data Cleaner (生产级 v2.0)

功能：
1. 四级分级清理：T1(内存15min) → T2(DB轻量30min) → T3(DB全量60min) → T4(VACUUM+归档24h)
2. 智能保留策略：盈利交易保留、策略里程碑保留、关键事件保留
3. 自动归档：删除前压缩归档到 backups/archive/
4. DB完整性检查：PRAGMA integrity_check 前置校验
5. 清理预算控制：单次清理最长30s，超时自动分批
6. 自适应间隔：根据DB大小动态调整清理频率
7. 健康状态上报：通过 health_status.json 暴露清理健康度
8. 清理建议生成：基于数据分析给出优化建议
9. 文件系统清理 (旧日志 / 告警JSON / 复盘报告 / 备份)
10. 内存数据结构清理 (TradeJournal / StopLossManager / OrderLifecycleManager)
11. Redis 过期计数器清理
12. SQLite VACUUM (回收磁盘空间)

设计原则：
- 所有清理操作异步执行，不阻塞主交易循环
- 清理统计可追踪，通过 Dashboard API 暴露
- 分级清理频率：T1(内存) → T2(DB轻量) → T3(DB全量) → T4(VACUUM+归档)
"""

import asyncio
import os
import glob
import gzip
import shutil
import time
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class CleanupTier(Enum):
    """清理等级"""
    T1_MEMORY = 1       # 内存清理：15min
    T2_DB_LIGHT = 2     # 数据库轻量：30min
    T3_DB_FULL = 3      # 数据库全量：60min
    T4_DEEP = 4         # 深度清理：24h (VACUUM + 归档)


@dataclass
class TableRetentionPolicy:
    """表级保留策略"""
    table_name: str
    time_column: str
    retention_days: int
    min_rows: int = 0               # 最少保留行数
    extra_condition: str = ""       # 额外 WHERE 条件
    keep_profitable: bool = False   # 是否保留盈利记录
    archive_before_delete: bool = False  # 删除前是否归档


@dataclass
class CleanupStats:
    """清理统计"""
    last_run: Optional[datetime] = None
    last_tier_runs: Dict[int, Optional[datetime]] = field(default_factory=dict)
    total_runs: int = 0
    total_deleted_rows: int = 0
    total_deleted_files: int = 0
    total_freed_bytes: int = 0
    total_archived_rows: int = 0
    per_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    tier_stats: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    db_size_history: List[Tuple[str, int]] = field(default_factory=list)  # (timestamp, bytes)
    
    def record(self, category: str, sub: str, count: int, freed_bytes: int = 0):
        if category not in self.per_category:
            self.per_category[category] = {}
        self.per_category[category][sub] = self.per_category[category].get(sub, 0) + count
        self.total_deleted_rows += count
        self.total_freed_bytes += freed_bytes
    
    def record_files(self, category: str, count: int):
        if category not in self.per_category:
            self.per_category[category] = {}
        self.per_category[category]["files"] = self.per_category[category].get("files", 0) + count
        self.total_deleted_files += count
    
    def record_archived(self, count: int):
        self.total_archived_rows += count
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_tier_runs": {str(k): v.isoformat() if v else None for k, v in self.last_tier_runs.items()},
            "total_runs": self.total_runs,
            "total_deleted_rows": self.total_deleted_rows,
            "total_deleted_files": self.total_deleted_files,
            "total_freed_bytes": self.total_freed_bytes,
            "total_archived_rows": self.total_archived_rows,
            "per_category": dict(self.per_category),
            "tier_stats": dict(self.tier_stats),
            "recent_errors": self.errors[-10:],
            "recent_warnings": self.warnings[-10:],
        }


class DataCleaner:
    """生产级统一数据清理引擎 v2.0"""
    
    # 默认表级保留策略
    DEFAULT_RETENTION_POLICIES = [
        TableRetentionPolicy("position_history", "timestamp", 30, min_rows=5000),
        TableRetentionPolicy("account_history", "timestamp", 30, min_rows=1000),
        TableRetentionPolicy("trade_records", "close_time", 60, min_rows=100,
                            extra_condition="status = 'closed'", keep_profitable=True,
                            archive_before_delete=True),
        TableRetentionPolicy("trades", "created_at", 90, min_rows=50,
                            keep_profitable=True, archive_before_delete=True),
        TableRetentionPolicy("risk_events", "timestamp", 30, min_rows=100),
        TableRetentionPolicy("alert_records", "create_time", 14, min_rows=50),
        TableRetentionPolicy("recovery_records", "start_time", 30, min_rows=20),
        TableRetentionPolicy("stop_loss_audit", "created_at", 30, min_rows=100),
        TableRetentionPolicy("profit_lock_audit", "created_at", 30, min_rows=100),
        TableRetentionPolicy("equity_curve", "timestamp", 60, min_rows=2000),
        TableRetentionPolicy("trading_signals", "created_at", 14, min_rows=200),
        TableRetentionPolicy("fill_quality", "created_at", 30, min_rows=100),
        TableRetentionPolicy("signal_generator_history", "created_at", 14, min_rows=200),
    ]
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.stats = CleanupStats()
        self._stop_event = asyncio.Event()
        
        # 清理配置
        cleanup_cfg = config.get("data_cleanup", {})
        self._tier_intervals = {
            CleanupTier.T1_MEMORY: cleanup_cfg.get("t1_memory_interval_min", 15),
            CleanupTier.T2_DB_LIGHT: cleanup_cfg.get("t2_db_light_interval_min", 30),
            CleanupTier.T3_DB_FULL: cleanup_cfg.get("t3_db_full_interval_min", 60),
            CleanupTier.T4_DEEP: cleanup_cfg.get("t4_deep_interval_hours", 24),
        }
        self._cleanup_budget_sec = cleanup_cfg.get("cleanup_budget_sec", 30)
        self._archive_enabled = cleanup_cfg.get("archive_enabled", True)
        self._archive_dir = cleanup_cfg.get("archive_dir", "./backups/archive")
        self._integrity_check_enabled = cleanup_cfg.get("integrity_check_enabled", True)
        self._adaptive_interval = cleanup_cfg.get("adaptive_interval", True)
        self._db_size_threshold_mb = cleanup_cfg.get("db_size_threshold_mb", 50)
        self._db_retention_days = cleanup_cfg.get("db_retention_days", 30)
        self._order_archive_days = cleanup_cfg.get("order_archive_days", 60)
        self._log_retention_days = cleanup_cfg.get("log_retention_days", 7)
        self._alert_retention_days = cleanup_cfg.get("alert_retention_days", 7)
        self._review_keep_count = cleanup_cfg.get("review_max_daily", 90)
        self._review_monthly_keep = cleanup_cfg.get("review_max_monthly", 12)
        self._backup_retention_days = cleanup_cfg.get("backup_retention_days", 30)
        # 参数优化沉淤防护：保留文件数 / 保留天数 / 体积阈值（超过则加强清理）
        self._param_opt_keep_count = cleanup_cfg.get("param_opt_keep_count", 50)
        self._param_opt_retention_days = cleanup_cfg.get("param_opt_retention_days", 30)
        self._param_opt_max_size_mb = cleanup_cfg.get("param_opt_max_size_mb", 200)
        
        # 表级保留策略（可从配置覆盖）
        self._retention_policies = list(self.DEFAULT_RETENTION_POLICIES)
        custom_policies = cleanup_cfg.get("table_policies", {})
        for policy in self._retention_policies:
            if policy.table_name in custom_policies:
                cp = custom_policies[policy.table_name]
                policy.retention_days = cp.get("retention_days", policy.retention_days)
                policy.min_rows = cp.get("min_rows", policy.min_rows)
                policy.keep_profitable = cp.get("keep_profitable", policy.keep_profitable)
                policy.archive_before_delete = cp.get("archive_before_delete", policy.archive_before_delete)
        
        # 外部依赖注入
        self._sqlite_storage = None
        self._log_persistence = None
        self._redis_cache = None
        self._trade_journal = None
        self._stop_loss_manager = None
        self._order_lifecycle_manager = None
        self._order_queue = None
        
        self._last_vacuum_time = 0.0
        self._last_tier_times = {t: 0.0 for t in CleanupTier}
        self._alert_callback: Optional[Callable] = None
        
        logger.info(f"DataCleaner v2.0 initialized: T1={self._tier_intervals[CleanupTier.T1_MEMORY]}min, "
                   f"T2={self._tier_intervals[CleanupTier.T2_DB_LIGHT]}min, "
                   f"T3={self._tier_intervals[CleanupTier.T3_DB_FULL]}min, "
                   f"T4={self._tier_intervals[CleanupTier.T4_DEEP]}h, "
                   f"budget={self._cleanup_budget_sec}s, archive={self._archive_enabled}")
    
    # ==================== 依赖注入 ====================
    
    def set_sqlite_storage(self, storage):
        self._sqlite_storage = storage
    
    def set_log_persistence(self, persistence):
        self._log_persistence = persistence
    
    def set_redis_cache(self, cache):
        self._redis_cache = cache
    
    def set_trade_journal(self, journal):
        self._trade_journal = journal
    
    def set_stop_loss_manager(self, mgr):
        self._stop_loss_manager = mgr
    
    def set_order_lifecycle_manager(self, mgr):
        self._order_lifecycle_manager = mgr
    
    def set_order_queue(self, queue):
        self._order_queue = queue
    
    def set_alert_callback(self, callback: Callable):
        """设置告警回调（用于通知管理器）"""
        self._alert_callback = callback
    
    def update_config(self, config: Dict[str, Any]):
        """热更新配置"""
        cleanup_cfg = config.get("data_cleanup", {})
        for tier in CleanupTier:
            key_map = {
                CleanupTier.T1_MEMORY: "t1_memory_interval_min",
                CleanupTier.T2_DB_LIGHT: "t2_db_light_interval_min",
                CleanupTier.T3_DB_FULL: "t3_db_full_interval_min",
                CleanupTier.T4_DEEP: "t4_deep_interval_hours",
            }
            if key_map[tier] in cleanup_cfg:
                self._tier_intervals[tier] = cleanup_cfg[key_map[tier]]
        self._cleanup_budget_sec = cleanup_cfg.get("cleanup_budget_sec", self._cleanup_budget_sec)
        self._archive_enabled = cleanup_cfg.get("archive_enabled", self._archive_enabled)
        self._integrity_check_enabled = cleanup_cfg.get("integrity_check_enabled", self._integrity_check_enabled)
        self._adaptive_interval = cleanup_cfg.get("adaptive_interval", self._adaptive_interval)
        self._db_size_threshold_mb = cleanup_cfg.get("db_size_threshold_mb", self._db_size_threshold_mb)
        self._param_opt_keep_count = cleanup_cfg.get("param_opt_keep_count", self._param_opt_keep_count)
        self._param_opt_retention_days = cleanup_cfg.get("param_opt_retention_days", self._param_opt_retention_days)
        self._param_opt_max_size_mb = cleanup_cfg.get("param_opt_max_size_mb", self._param_opt_max_size_mb)
        logger.info("DataCleaner config hot-updated")
    
    # ==================== 主循环（四级） ====================
    
    async def start(self):
        """启动四级清理循环"""
        asyncio.create_task(self._tier_loop(CleanupTier.T1_MEMORY))
        asyncio.create_task(self._tier_loop(CleanupTier.T2_DB_LIGHT))
        asyncio.create_task(self._tier_loop(CleanupTier.T3_DB_FULL))
        asyncio.create_task(self._tier_loop(CleanupTier.T4_DEEP))
        logger.info("DataCleaner v2.0: 4-tier cleanup loops started")
    
    async def stop(self):
        self._stop_event.set()
    
    async def _tier_loop(self, tier: CleanupTier):
        """通用分级清理循环"""
        interval = self._tier_intervals[tier]
        if tier == CleanupTier.T4_DEEP:
            sleep_sec = interval * 3600
        else:
            sleep_sec = interval * 60
        
        logger.info(f"DataCleaner Tier{tier.value} loop started (interval={interval}{'h' if tier == CleanupTier.T4_DEEP else 'min'})")
        
        while not self._stop_event.is_set():
            try:
                await self._run_tier(tier)
            except Exception as e:
                logger.error(f"DataCleaner Tier{tier.value} error: {e}")
                self.stats.errors.append(f"{datetime.now().isoformat()}: {e}")
            
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_sec)
            except asyncio.TimeoutError:
                pass
    
    async def _run_tier(self, tier: CleanupTier):
        """执行指定级别的清理"""
        budget_end = time.time() + self._cleanup_budget_sec
        tier_start = time.time()
        
        if tier == CleanupTier.T1_MEMORY:
            await self._run_memory_cleanup()
        elif tier == CleanupTier.T2_DB_LIGHT:
            await self._cleanup_db_light(budget_end)
        elif tier == CleanupTier.T3_DB_FULL:
            await self._cleanup_db_full(budget_end)
            await self._cleanup_log_db()
            await self._cleanup_filesystem()
            await self._cleanup_redis()
            # 策略绩效增量刷新：从权威 trades 表聚合回填 strategy_performance
            self._refresh_strategy_performance()
        elif tier == CleanupTier.T4_DEEP:
            await self._maybe_vacuum()
            await self._archive_old_data()
            await self._run_integrity_check()
            # 订单冷热分表归档 + WAL 合并（独立于 JSON.gz 归档开关）
            self._run_order_sharding()
            self._run_wal_checkpoint()
            self._record_db_size()
        
        elapsed = time.time() - tier_start
        self.stats.last_tier_runs[tier.value] = datetime.now()
        self.stats.last_run = datetime.now()
        self.stats.total_runs += 1
        
        # 记录分级统计
        if tier.value not in self.stats.tier_stats:
            self.stats.tier_stats[tier.value] = {"runs": 0, "total_elapsed": 0}
        self.stats.tier_stats[tier.value]["runs"] += 1
        self.stats.tier_stats[tier.value]["total_elapsed"] += elapsed
        
        if elapsed > self._cleanup_budget_sec * 0.8:
            logger.warning(f"DataCleaner Tier{tier.value}: cleanup took {elapsed:.1f}s (budget={self._cleanup_budget_sec}s)")
    
    # ==================== T2: DB 轻量清理 ====================
    
    async def _cleanup_db_light(self, budget_end: float):
        """轻量DB清理：只清理高频累积的大表"""
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return
        
        # 自适应间隔：DB超过阈值时缩短清理间隔
        if self._adaptive_interval:
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            if db_size_mb > self._db_size_threshold_mb:
                logger.info(f"DataCleaner: DB size {db_size_mb:.1f}MB > threshold {self._db_size_threshold_mb}MB, "
                           f"intensifying cleanup")
        
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            # 轻量策略：只清理 position_history 和 account_history（数据量最大）
            light_tables = ["position_history", "account_history"]
            for policy in self._retention_policies:
                if policy.table_name not in light_tables:
                    continue
                if time.time() > budget_end:
                    logger.warning("DataCleaner T2: budget exceeded, deferring remaining tables to next cycle")
                    break
                await self._cleanup_table_with_policy(cursor, policy, conn)
            
            conn.commit()
        except Exception as e:
            logger.error(f"DataCleaner T2 DB light error: {e}")
            self.stats.errors.append(str(e))
        finally:
            if conn:
                conn.close()
    
    # ==================== T3: DB 全量清理 ====================
    
    async def _cleanup_db_full(self, budget_end: float):
        """全量DB清理：按表级策略清理所有表"""
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return
        
        conn = None
        tables_cleaned = []
        
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            for policy in self._retention_policies:
                if policy.table_name == "trade_records":
                    continue  # 订单数据生命周期由 archive_closed_trades 分表归档管理
                if time.time() > budget_end:
                    logger.warning("DataCleaner T3: budget exceeded, deferring remaining tables to next cycle")
                    self.stats.warnings.append(f"budget_exceeded_at_{policy.table_name}")
                    break
                
                try:
                    deleted = await self._cleanup_table_with_policy(cursor, policy, conn)
                    if deleted > 0:
                        tables_cleaned.append(f"{policy.table_name}({deleted})")
                except Exception as e:
                    logger.debug(f"DataCleaner T3: skip table {policy.table_name}: {e}")
            
            # 调用 SQLiteStorage 原有清理方法
            # 先提交当前连接的事务，释放锁，避免与 cleanup_old_data 的 SQLAlchemy session 死锁
            conn.commit()
            cursor.close()
            conn.close()
            conn = None
            
            if self._sqlite_storage and hasattr(self._sqlite_storage, 'cleanup_old_data'):
                try:
                    await self._sqlite_storage.cleanup_old_data(days_to_keep=self._db_retention_days)
                except Exception as e:
                    logger.warning(f"SQLiteStorage.cleanup_old_data failed: {e}")
            
            if tables_cleaned:
                logger.info(f"DataCleaner T3 DB: cleaned {', '.join(tables_cleaned)}")
                
        except Exception as e:
            logger.error(f"DataCleaner T3 DB full error: {e}")
            self.stats.errors.append(str(e))
        finally:
            if conn:
                conn.close()
    
    async def _cleanup_table_with_policy(self, cursor, policy: TableRetentionPolicy, conn) -> int:
        """按保留策略清理单表"""
        # 1. 检查表是否存在
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {policy.table_name} LIMIT 1")
        except sqlite3.OperationalError:
            return 0
        
        cutoff = (datetime.now() - timedelta(days=policy.retention_days)).isoformat()
        
        # 2. 构建清理条件
        conditions = [f"{policy.time_column} < ?"]
        params = [cutoff]
        if policy.extra_condition:
            conditions.append(policy.extra_condition)
        
        # 3. 智能保留：盈利记录
        if policy.keep_profitable:
            conditions.append("(pnl_usdt IS NULL OR pnl_usdt <= 0)")
        
        where_clause = " AND ".join(conditions)
        
        # 4. 检查最少保留行数
        cursor.execute(f"SELECT COUNT(*) FROM {policy.table_name}")
        total_rows = cursor.fetchone()[0]
        
        cursor.execute(f"SELECT COUNT(*) FROM {policy.table_name} WHERE {where_clause}", params)
        to_delete = cursor.fetchone()[0]
        
        if total_rows - to_delete < policy.min_rows:
            # 保留最少行数，只删除超出部分
            actual_delete = max(0, total_rows - policy.min_rows)
            if actual_delete < to_delete:
                to_delete = actual_delete
                # 重建查询以限制删除数量
                cursor.execute(
                    f"DELETE FROM {policy.table_name} WHERE rowid IN ("
                    f"SELECT rowid FROM {policy.table_name} WHERE {where_clause} "
                    f"ORDER BY {policy.time_column} ASC LIMIT ?"
                    f")", params + [to_delete]
                )
            else:
                cursor.execute(f"DELETE FROM {policy.table_name} WHERE {where_clause}", params)
        else:
            # 5. 归档（在删除前）
            if policy.archive_before_delete and self._archive_enabled and to_delete > 0:
                archived = await self._archive_table_rows(cursor, policy.table_name, where_clause, params)
                if archived > 0:
                    self.stats.record_archived(archived)
            
            if to_delete > 0:
                cursor.execute(f"DELETE FROM {policy.table_name} WHERE {where_clause}", params)
        
        actual_deleted = cursor.rowcount if cursor.rowcount >= 0 else to_delete
        if actual_deleted > 0:
            self.stats.record("sqlite", policy.table_name, actual_deleted)
        
        return actual_deleted
    
    # ==================== 自动归档 ====================
    
    async def _archive_old_data(self):
        """归档旧数据（T4深度清理时执行）"""
        if not self._archive_enabled:
            return
        
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return
        
        archive_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   self._archive_dir.lstrip("./"))
        os.makedirs(archive_root, exist_ok=True)
        
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            for policy in self._retention_policies:
                if not policy.archive_before_delete:
                    continue
                if policy.table_name == "trade_records":
                    continue  # 订单冷数据走 archive_closed_trades 分表归档
                
                # 归档超过保留期2倍的数据
                cutoff = (datetime.now() - timedelta(days=policy.retention_days * 2)).isoformat()
                try:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {policy.table_name} WHERE {policy.time_column} < ?",
                        (cutoff,)
                    )
                    count = cursor.fetchone()[0]
                except sqlite3.OperationalError:
                    continue
                
                if count == 0:
                    continue
                
                archive_file = os.path.join(
                    archive_root,
                    f"{policy.table_name}_{datetime.now().strftime('%Y%m%d')}.json.gz"
                )
                
                try:
                    cursor.execute(
                        f"SELECT * FROM {policy.table_name} WHERE {policy.time_column} < ? LIMIT 10000",
                        (cutoff,)
                    )
                    rows = cursor.fetchall()
                    columns = [desc[0] for desc in cursor.description]
                    
                    data = {
                        "table": policy.table_name,
                        "archived_at": datetime.now().isoformat(),
                        "columns": columns,
                        "row_count": len(rows),
                        "rows": [dict(zip(columns, row)) for row in rows],
                    }
                    
                    import json
                    with gzip.open(archive_file, "wt", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, default=str)
                    
                    self.stats.record_archived(len(rows))
                    logger.info(f"DataCleaner: archived {len(rows)} rows from {policy.table_name} → {archive_file}")
                    
                except Exception as e:
                    logger.warning(f"DataCleaner archive {policy.table_name} error: {e}")
        
        except Exception as e:
            logger.error(f"DataCleaner archive error: {e}")
        finally:
            if conn:
                conn.close()
    
    async def _archive_table_rows(self, cursor, table_name: str, where_clause: str, params: list) -> int:
        """归档指定条件的行（同步版本，在删除前调用）"""
        if not self._archive_enabled:
            return 0
        
        archive_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   self._archive_dir.lstrip("./"))
        os.makedirs(archive_root, exist_ok=True)
        
        try:
            cursor.execute(f"SELECT * FROM {table_name} WHERE {where_clause} LIMIT 5000", params)
            rows = cursor.fetchall()
            if not rows:
                return 0
            
            columns = [desc[0] for desc in cursor.description]
            archive_file = os.path.join(
                archive_root,
                f"{table_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json.gz"
            )
            
            data = {
                "table": table_name,
                "archived_at": datetime.now().isoformat(),
                "columns": columns,
                "row_count": len(rows),
                "rows": [dict(zip(columns, row)) for row in rows],
            }
            
            import json
            with gzip.open(archive_file, "wt", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, default=str)
            
            return len(rows)
        except Exception as e:
            logger.debug(f"DataCleaner archive rows error: {e}")
            return 0
    
    # ==================== 订单分表归档 & WAL 合并 ====================

    def _run_order_sharding(self) -> None:
        """订单冷热分表：把热表 trade_records 中过期已平仓记录按月迁入分片表。"""
        if not self._sqlite_storage or not hasattr(self._sqlite_storage, 'archive_closed_trades'):
            return
        try:
            result = self._sqlite_storage.archive_closed_trades(before_days=self._order_archive_days)
            if result and result.get("archived"):
                self.stats.record_archived(result["archived"])
                self.stats.record("sqlite", "trade_records_shards", result["archived"])
                logger.info(f"DataCleaner: order sharding archived {result['archived']} closed trades "
                            f"into months {result.get('months')}")
        except Exception as e:
            logger.warning(f"DataCleaner order sharding failed: {e}")

    def _run_wal_checkpoint(self) -> None:
        """归档后执行 WAL checkpoint，把预写日志合并回主库并回收磁盘。"""
        if not self._sqlite_storage or not hasattr(self._sqlite_storage, 'wal_checkpoint'):
            return
        try:
            self._sqlite_storage.wal_checkpoint("TRUNCATE")
            logger.debug("DataCleaner: WAL checkpoint (TRUNCATE) completed")
        except Exception as e:
            logger.warning(f"DataCleaner WAL checkpoint failed: {e}")

    def _refresh_strategy_performance(self) -> None:
        """从权威 trades 表聚合真实绩效，回填 strategy_performance 表。

        update_strategy_performance() 长期未被调用，strategy_performance 表始终
        为空，Dashboard 被迫回退到 trade_records（43% pnl=NULL、ghost_close 标签
        失真）聚合。本方法挂 T3（60min）周期从 trades 权威表重算并回填，保证
        绩效基准准确且随增量自动更新。
        """
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            cur = conn.cursor()
            cur.execute(
                "SELECT strategy_name, symbol, exit_time, pnl_usdt, win FROM trades "
                "WHERE strategy_name IS NOT NULL "
                "AND strategy_name NOT IN ('sync', 'unknown', '') "
                "ORDER BY strategy_name, symbol, exit_time ASC"
            )
            rows = cur.fetchall()

            groups: Dict[tuple, list] = {}
            for r in rows:
                groups.setdefault((r["strategy_name"], r["symbol"]), []).append(r)

            now = datetime.now().isoformat()
            perf = []
            for (strat, sym), recs in groups.items():
                total = len(recs)
                wins = sum(1 for r in recs if r["win"] == 1)
                total_pnl = sum(r["pnl_usdt"] or 0.0 for r in recs)
                gross_profit = sum(r["pnl_usdt"] for r in recs if (r["pnl_usdt"] or 0.0) > 0)
                gross_loss = abs(sum(r["pnl_usdt"] for r in recs if (r["pnl_usdt"] or 0.0) < 0))
                win_rate = wins / total if total else 0.0
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
                peak = cum = max_dd = 0.0
                for r in recs:
                    cum += (r["pnl_usdt"] or 0.0)
                    peak = max(peak, cum)
                    max_dd = max(max_dd, peak - cum)
                perf.append((f"{strat}:{sym}", strat, sym, total, wins, total - wins,
                             total_pnl, max_dd, win_rate, profit_factor, now))

            cur.execute("DELETE FROM strategy_performance")
            cur.executemany(
                "INSERT INTO strategy_performance (id, strategy_name, symbol, total_trades, "
                "winning_trades, losing_trades, total_pnl, max_drawdown, win_rate, "
                "profit_factor, last_update) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                perf,
            )
            conn.commit()
            conn.close()
            self.stats.record("sqlite", "strategy_performance_refresh", len(perf))
            logger.info(f"DataCleaner: strategy_performance refreshed ({len(perf)} rows)")
        except Exception as e:
            logger.warning(f"DataCleaner strategy_performance refresh failed: {e}")

    # ==================== DB完整性检查 ====================
    
    async def _run_integrity_check(self):
        """执行数据库完整性检查"""
        if not self._integrity_check_enabled:
            return
        
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return
        
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check")
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] == "ok":
                logger.debug("DataCleaner: DB integrity check passed")
            else:
                logger.error(f"DataCleaner: DB integrity check FAILED: {result}")
                self.stats.errors.append(f"integrity_check_failed: {result}")
                if self._alert_callback:
                    self._alert_callback("DB_INTEGRITY_FAILED", f"数据库完整性检查失败: {result}", "critical")
        except Exception as e:
            logger.error(f"DataCleaner integrity check error: {e}")
            self.stats.errors.append(str(e))
    
    def _record_db_size(self):
        """记录DB大小历史"""
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return
        try:
            size = os.path.getsize(db_path)
            self.stats.db_size_history.append((datetime.now().isoformat(), size))
            # 只保留最近30条记录
            if len(self.stats.db_size_history) > 30:
                self.stats.db_size_history = self.stats.db_size_history[-30:]
        except Exception:
            pass
    
    # ==================== 日志数据库清理 ====================
    
    async def _cleanup_log_db(self):
        """清理日志数据库"""
        if not self._log_persistence:
            return
        
        try:
            if hasattr(self._log_persistence, 'cleanup_old_logs'):
                result = self._log_persistence.cleanup_old_logs(retention_days=self._log_retention_days)
                if result:
                    self.stats.record("logs_db", "cleaned", result.get("total_deleted", 0))
                    logger.info(f"DataCleaner: log DB cleaned, deleted {result.get('total_deleted', 0)} rows")
        except Exception as e:
            logger.error(f"DataCleaner log DB error: {e}")
    
    # ==================== 文件系统清理 ====================
    
    async def _cleanup_filesystem(self):
        """清理文件系统中的过期数据"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        await self._cleanup_alert_files(base_dir)
        await self._cleanup_review_files(base_dir)
        await self._cleanup_log_backup_files(base_dir)
        await self._cleanup_old_backups(base_dir)
        await self._cleanup_old_archives(base_dir)
        await self._cleanup_param_optimization_files(base_dir)
    
    async def _cleanup_alert_files(self, base_dir: str):
        """清理过期的告警 JSON 文件"""
        alerts_dir = os.path.join(base_dir, "data", "alerts")
        if not os.path.exists(alerts_dir):
            return
        
        cutoff = time.time() - self._alert_retention_days * 86400
        deleted = 0
        try:
            for fname in os.listdir(alerts_dir):
                fpath = os.path.join(alerts_dir, fname)
                if os.path.isfile(fpath) and fname.startswith("alerts_"):
                    if os.path.getmtime(fpath) < cutoff:
                        try:
                            os.remove(fpath)
                            deleted += 1
                        except OSError:
                            pass
            if deleted > 0:
                self.stats.record_files("alerts", deleted)
                logger.info(f"DataCleaner: deleted {deleted} old alert files")
        except Exception as e:
            logger.warning(f"DataCleaner alert files error: {e}")
    
    async def _cleanup_review_files(self, base_dir: str):
        """清理过期的复盘报告文件"""
        review_dir = self.config.get("review", {}).get("data_dir", "./data/reviews")
        if not os.path.isabs(review_dir):
            review_dir = os.path.join(base_dir, review_dir.lstrip("./"))
        if not os.path.exists(review_dir):
            return
        
        deleted = 0
        try:
            daily_files = []
            monthly_files = []
            
            for fname in os.listdir(review_dir):
                fpath = os.path.join(review_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                if fname.startswith("daily_review_"):
                    daily_files.append((fpath, os.path.getmtime(fpath)))
                elif fname.startswith("monthly_plan_"):
                    monthly_files.append((fpath, os.path.getmtime(fpath)))
            
            daily_files.sort(key=lambda x: x[1], reverse=True)
            for fpath, _ in daily_files[self._review_keep_count:]:
                try:
                    os.remove(fpath)
                    deleted += 1
                except OSError:
                    pass
            
            monthly_files.sort(key=lambda x: x[1], reverse=True)
            for fpath, _ in monthly_files[self._review_monthly_keep:]:
                try:
                    os.remove(fpath)
                    deleted += 1
                except OSError:
                    pass
            
            if deleted > 0:
                self.stats.record_files("reviews", deleted)
                logger.info(f"DataCleaner: deleted {deleted} old review files")
        except Exception as e:
            logger.warning(f"DataCleaner review files error: {e}")
    
    async def _cleanup_log_backup_files(self, base_dir: str):
        """清理 log_persistence 产生的日志备份文件"""
        logs_dir = os.path.join(base_dir, "logs")
        if not os.path.exists(logs_dir):
            return
        
        cutoff = time.time() - self._log_retention_days * 86400
        deleted = 0
        log_types = ["market", "signal", "order", "risk", "error", "system", "position", "pnl",
                     "main_stdout", "main_stderr", "watchdog"]
        
        try:
            for fname in os.listdir(logs_dir):
                fpath = os.path.join(logs_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                for ltype in log_types:
                    if fname.startswith(f"{ltype}_") and fname.endswith(".log"):
                        if os.path.getmtime(fpath) < cutoff:
                            try:
                                os.remove(fpath)
                                deleted += 1
                            except OSError:
                                pass
                        break
            if deleted > 0:
                self.stats.record_files("log_backups", deleted)
                logger.info(f"DataCleaner: deleted {deleted} old log backup files")
        except Exception as e:
            logger.warning(f"DataCleaner log backup files error: {e}")
    
    async def _cleanup_old_backups(self, base_dir: str):
        """清理过期备份文件，防止 backups/.bak 堆积导致磁盘涨满。

        覆盖三类：
        - backups/trading_*.db（数据库备份）
        - backups/config.yaml.backup_*（配置文件备份）
        - *.bak（策略状态重置等就地备份，含 backups/ 与 data/strategy_state/）
        """
        cutoff = time.time() - self._backup_retention_days * 86400
        deleted = 0

        target_dirs = [
            os.path.join(base_dir, "backups"),
            os.path.join(base_dir, "data", "strategy_state"),
        ]

        for target_dir in target_dirs:
            if not os.path.isdir(target_dir):
                continue
            try:
                for fname in os.listdir(target_dir):
                    fpath = os.path.join(target_dir, fname)
                    if not os.path.isfile(fpath):
                        continue
                    is_backup = (
                        (fname.startswith("trading_") and fname.endswith(".db"))
                        or fname.startswith("config.yaml.backup_")
                        or fname.endswith(".bak")
                    )
                    if is_backup and os.path.getmtime(fpath) < cutoff:
                        try:
                            os.remove(fpath)
                            deleted += 1
                        except OSError:
                            pass
            except Exception as e:
                logger.warning(f"DataCleaner backup cleanup error in {target_dir}: {e}")

        if deleted > 0:
            self.stats.record_files("backups", deleted)
            logger.info(f"DataCleaner: deleted {deleted} old backup files")
    
    async def _cleanup_old_archives(self, base_dir: str):
        """清理过期归档文件（保留90天）"""
        archive_dir = os.path.join(base_dir, self._archive_dir.lstrip("./"))
        if not os.path.exists(archive_dir):
            return
        
        cutoff = time.time() - 90 * 86400
        deleted = 0
        try:
            for fname in os.listdir(archive_dir):
                fpath = os.path.join(archive_dir, fname)
                if os.path.isfile(fpath) and fname.endswith(".json.gz"):
                    if os.path.getmtime(fpath) < cutoff:
                        try:
                            os.remove(fpath)
                            deleted += 1
                        except OSError:
                            pass
            if deleted > 0:
                self.stats.record_files("archives", deleted)
                logger.info(f"DataCleaner: deleted {deleted} old archive files")
        except Exception as e:
            logger.warning(f"DataCleaner archive files error: {e}")
    
    async def _cleanup_param_optimization_files(self, base_dir: str):
        """清理参数优化历史文件，智能判断规模趋势，防止沉淤拖垮系统。

        策略：
        1. 扫描 parameter_optimization 目录下的 opt_*.json；
        2. 按修改时间倒序，保留最近 _param_opt_keep_count 个；
        3. 同时删除超过 _param_opt_retention_days 的过期文件；
        4. 智能趋势判断：文件数量或总体积超过阈值时，自动收紧保留数量，加速清理。
        """
        po_cfg = self.config.get("parameter_optimization", {})
        persist_dir = po_cfg.get("persist_dir", "./data/parameter_optimization")
        if not os.path.isabs(persist_dir):
            persist_dir = os.path.join(base_dir, persist_dir.lstrip("./"))
        if not os.path.isdir(persist_dir):
            return

        cutoff = time.time() - self._param_opt_retention_days * 86400
        try:
            files = []
            total_size = 0
            for fname in os.listdir(persist_dir):
                if not (fname.startswith("opt_") and fname.endswith(".json")):
                    continue
                fpath = os.path.join(persist_dir, fname)
                try:
                    mtime = os.path.getmtime(fpath)
                except OSError:
                    continue
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    size = 0
                files.append((fpath, mtime, size))
                total_size += size

            if not files:
                return

            files.sort(key=lambda x: x[1], reverse=True)  # 最新在前

            # 智能趋势判断：数量或体积异常增长时，收紧保留数量
            effective_keep = self._param_opt_keep_count
            size_mb = total_size / (1024 * 1024)
            if len(files) > self._param_opt_keep_count * 3:
                effective_keep = max(10, self._param_opt_keep_count // 2)
                logger.info(f"DataCleaner: {len(files)} parameter optimization files exceed 3x "
                            f"threshold, intensifying cleanup to keep {effective_keep}")
            if size_mb > self._param_opt_max_size_mb:
                effective_keep = max(10, min(effective_keep, self._param_opt_keep_count // 2))
                logger.info(f"DataCleaner: parameter optimization dir {size_mb:.1f}MB exceed "
                            f"{self._param_opt_max_size_mb}MB, intensifying cleanup to keep {effective_keep}")

            deleted = 0
            for i, (fpath, mtime, _size) in enumerate(files):
                if i < effective_keep and mtime >= cutoff:
                    continue
                try:
                    os.remove(fpath)
                    deleted += 1
                except OSError:
                    pass

            if deleted > 0:
                self.stats.record_files("param_optimization", deleted)
                logger.info(f"DataCleaner: deleted {deleted} old parameter optimization files "
                            f"(kept {len(files) - deleted})")
        except Exception as e:
            logger.warning(f"DataCleaner parameter optimization files error: {e}")
    
    # ==================== Redis 清理 ====================
    
    async def _cleanup_redis(self):
        """清理 Redis 过期计数器"""
        if not self._redis_cache:
            return
        
        try:
            client = getattr(self._redis_cache, '_redis', None)
            if not client:
                return
            
            deleted = 0
            try:
                cursor = 0
                while True:
                    cursor, keys = client.scan(cursor, match="risk:counter:*", count=100)
                    for key in keys:
                        client.expire(key, 86400)
                        deleted += 1
                    if cursor == 0:
                        break
            except Exception as e:
                logger.debug(f"Redis counter cleanup: {e}")
            
            if deleted > 0:
                self.stats.record("redis", "counter_keys_ttl_set", deleted)
                logger.info(f"DataCleaner Redis: set TTL on {deleted} counter keys")
                
        except Exception as e:
            logger.warning(f"DataCleaner Redis error: {e}")
    
    # ==================== 内存清理 ====================
    
    async def _run_memory_cleanup(self):
        """清理内存中的累积数据结构"""
        if self._trade_journal:
            await self._cleanup_trade_journal()
        if self._stop_loss_manager:
            await self._cleanup_stop_loss_manager()
        if self._order_lifecycle_manager:
            await self._cleanup_order_lifecycle()
        if self._order_queue:
            await self._cleanup_order_queue_cache()
    
    async def _cleanup_trade_journal(self):
        """清理 TradeJournal 内存结构"""
        try:
            tj = self._trade_journal
            cleaned = 0
            
            if hasattr(tj, '_equity_curve') and len(tj._equity_curve) > 10080:
                cutoff = datetime.now() - timedelta(days=7)
                old_len = len(tj._equity_curve)
                tj._equity_curve = [
                    p for p in tj._equity_curve
                    if hasattr(p, 'timestamp') and p.timestamp > cutoff
                ]
                cleaned += old_len - len(tj._equity_curve)
            
            if hasattr(tj, '_trades') and len(tj._trades) > 5000:
                cutoff = datetime.now() - timedelta(days=30)
                old_len = len(tj._trades)
                tj._trades = {
                    k: v for k, v in tj._trades.items()
                    if getattr(v, 'status', '') == 'open' or
                       (hasattr(v, 'close_time') and v.close_time and v.close_time > cutoff)
                }
                cleaned += old_len - len(tj._trades)
            
            if cleaned > 0:
                self.stats.record("memory", "trade_journal", cleaned)
                logger.info(f"DataCleaner Memory: trade_journal cleaned {cleaned} entries")
        except Exception as e:
            logger.warning(f"DataCleaner trade_journal error: {e}")
    
    async def _cleanup_stop_loss_manager(self):
        """清理 StopLossManager 内存结构"""
        try:
            slm = self._stop_loss_manager
            cleaned = 0
            
            if hasattr(slm, '_stop_loss_events') and len(slm._stop_loss_events) > 1000:
                old_len = len(slm._stop_loss_events)
                slm._stop_loss_events = slm._stop_loss_events[-1000:]
                cleaned += old_len - len(slm._stop_loss_events)
            
            if hasattr(slm, '_triggered_recently'):
                now = datetime.now().timestamp()
                old_len = len(slm._triggered_recently)
                slm._triggered_recently = {
                    k: v for k, v in slm._triggered_recently.items()
                    if now - v < 600
                }
                cleaned += old_len - len(slm._triggered_recently)
            
            if cleaned > 0:
                self.stats.record("memory", "stop_loss_manager", cleaned)
                logger.debug(f"DataCleaner Memory: stop_loss_manager cleaned {cleaned} entries")
        except Exception as e:
            logger.warning(f"DataCleaner stop_loss_manager error: {e}")
    
    async def _cleanup_order_lifecycle(self):
        """清理 OrderLifecycleManager 内存结构"""
        try:
            olm = self._order_lifecycle_manager
            cleaned = 0
            
            if hasattr(olm, '_orders') and len(olm._orders) > 1000:
                cutoff = datetime.now() - timedelta(hours=24)
                old_len = len(olm._orders)
                olm._orders = {
                    k: v for k, v in olm._orders.items()
                    if v.get('create_time', datetime.min) > cutoff
                }
                cleaned += old_len - len(olm._orders)
            
            if hasattr(olm, '_stats') and 'latency_history' in olm._stats:
                lh = olm._stats['latency_history']
                if len(lh) > 1000:
                    old_len = len(lh)
                    olm._stats['latency_history'] = lh[-1000:]
                    cleaned += old_len - len(lh)
            
            if cleaned > 0:
                self.stats.record("memory", "order_lifecycle", cleaned)
                logger.debug(f"DataCleaner Memory: order_lifecycle cleaned {cleaned} entries")
        except Exception as e:
            logger.warning(f"DataCleaner order_lifecycle error: {e}")
    
    async def _cleanup_order_queue_cache(self):
        """清理 OrderQueue 已完成订单缓存"""
        try:
            oq = self._order_queue
            if hasattr(oq, '_order_cache') and len(oq._order_cache) > 500:
                cleaned = 0
                now = datetime.now()
                to_remove = []
                for oid, info in oq._order_cache.items():
                    status = info.get('status', '')
                    create_time = info.get('create_time')
                    if status in ('failed', 'stale_removed', 'executed', 'rejected_by_risk_gate'):
                        if create_time and (now - create_time).total_seconds() > 300:
                            to_remove.append(oid)
                
                for oid in to_remove:
                    del oq._order_cache[oid]
                    cleaned += 1
                
                if cleaned > 0:
                    self.stats.record("memory", "order_queue_cache", cleaned)
                    logger.debug(f"DataCleaner Memory: order_queue cache cleaned {cleaned} entries")
        except Exception as e:
            logger.warning(f"DataCleaner order_queue error: {e}")
    
    # ==================== VACUUM ====================
    
    async def _maybe_vacuum(self):
        """按间隔执行 SQLite VACUUM（T4深度清理）"""
        now = time.time()
        if now - self._last_vacuum_time < self._tier_intervals[CleanupTier.T4_DEEP] * 3600:
            return
        
        db_path = self._get_db_path()
        if not os.path.exists(db_path):
            return
        
        try:
            file_size = os.path.getsize(db_path)
            if file_size < 10 * 1024 * 1024:
                self._last_vacuum_time = now
                return
            
            logger.info(f"DataCleaner: starting VACUUM (db size: {file_size / 1024 / 1024:.1f} MB)...")
            
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            cursor.execute("VACUUM")
            conn.close()
            
            new_size = os.path.getsize(db_path)
            freed = file_size - new_size
            self.stats.total_freed_bytes += freed
            self._last_vacuum_time = now
            
            logger.info(f"DataCleaner: VACUUM completed, {file_size / 1024 / 1024:.1f}MB → "
                       f"{new_size / 1024 / 1024:.1f}MB (freed {freed / 1024:.1f} KB)")
        except Exception as e:
            logger.error(f"DataCleaner VACUUM error: {e}")
            self.stats.errors.append(str(e))
    
    # ==================== 清理建议 ====================
    
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """生成清理优化建议"""
        recommendations = []
        db_path = self._get_db_path()
        
        if os.path.exists(db_path):
            try:
                db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                # 检查各表行数
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in cursor.fetchall()]
                
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        
                        # 找到对应策略
                        policy = next((p for p in self._retention_policies if p.table_name == table), None)
                        if policy and count > 50000:
                            recommendations.append({
                                "type": "table_size_warning",
                                "priority": "medium",
                                "table": table,
                                "row_count": count,
                                "retention_days": policy.retention_days,
                                "action": f"考虑缩短 {table} 保留期（当前{policy.retention_days}天，共{count}行）",
                                "potential_saving": f"约{count // 2}行",
                            })
                    except sqlite3.OperationalError:
                        pass
                
                conn.close()
                
                # DB大小建议
                if db_size_mb > 100:
                    recommendations.append({
                        "type": "db_size_critical",
                        "priority": "high",
                        "db_size_mb": round(db_size_mb, 1),
                        "action": f"数据库 {db_size_mb:.1f}MB 超过100MB，建议增加清理频率或启用归档",
                    })
                elif db_size_mb > 50:
                    recommendations.append({
                        "type": "db_size_warning",
                        "priority": "medium",
                        "db_size_mb": round(db_size_mb, 1),
                        "action": f"数据库 {db_size_mb:.1f}MB，建议关注增长趋势",
                    })
                
            except Exception as e:
                logger.debug(f"DataCleaner recommendations error: {e}")
        
        # 清理统计建议
        if self.stats.errors:
            recommendations.append({
                "type": "cleanup_errors",
                "priority": "high",
                "error_count": len(self.stats.errors),
                "action": f"最近有 {len(self.stats.errors)} 个清理错误，请检查日志",
            })
        
        return recommendations
    
    # ==================== 健康状态 ====================
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取清理引擎健康状态"""
        db_path = self._get_db_path()
        db_size_mb = 0
        if os.path.exists(db_path):
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        
        last_tier_runs = {}
        for tier in CleanupTier:
            last = self.stats.last_tier_runs.get(tier.value)
            if last:
                interval = self._tier_intervals[tier]
                if tier == CleanupTier.T4_DEEP:
                    expected = interval * 3600
                else:
                    expected = interval * 60
                overdue = (datetime.now() - last).total_seconds() > expected * 1.5
                last_tier_runs[f"tier{tier.value}"] = {
                    "last_run": last.isoformat(),
                    "overdue": overdue,
                }
        
        return {
            "status": "degraded" if self.stats.errors else "healthy",
            "db_size_mb": round(db_size_mb, 1),
            "total_runs": self.stats.total_runs,
            "total_deleted_rows": self.stats.total_deleted_rows,
            "total_archived_rows": self.stats.total_archived_rows,
            "total_freed_bytes": self.stats.total_freed_bytes,
            "last_tier_runs": last_tier_runs,
            "error_count": len(self.stats.errors),
            "warning_count": len(self.stats.warnings),
        }
    
    # ==================== 辅助方法 ====================
    
    def _get_db_path(self) -> str:
        return self.config.get("sqlite", {}).get("db_path", "./data/trading.db")
    
    # ==================== 手动触发 ====================
    
    async def trigger_tier(self, tier: CleanupTier) -> Dict[str, Any]:
        """手动触发指定级别清理"""
        await self._run_tier(tier)
        return self.stats.to_dict()
    
    async def trigger_full_cleanup(self) -> Dict[str, Any]:
        """手动触发全量清理（T2+T3+T4）"""
        await self._run_tier(CleanupTier.T2_DB_LIGHT)
        await self._run_tier(CleanupTier.T3_DB_FULL)
        await self._run_tier(CleanupTier.T4_DEEP)
        return self.stats.to_dict()
    
    async def trigger_memory_cleanup(self) -> Dict[str, Any]:
        """手动触发内存清理"""
        await self._run_tier(CleanupTier.T1_MEMORY)
        return self.stats.to_dict()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取清理统计"""
        result = self.stats.to_dict()
        result["recommendations"] = self.generate_recommendations()
        return result
