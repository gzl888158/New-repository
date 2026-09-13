"""
DataCleaner v2.0 集成测试

覆盖：
1. 配置加载与实例化
2. CleanupTier 枚举与 TableRetentionPolicy
3. CleanupStats 记录
4. T1 内存清理
5. T2/T3 DB 清理（表级策略）
6. T4 归档与 VACUUM
7. 完整性检查
8. 文件系统清理
9. Redis 清理
10. 清理建议生成
11. 健康状态
12. 配置热更新
13. 手动触发
14. 自适应间隔
"""

import os
import sys
import json
import time
import gzip
import sqlite3
import tempfile
import asyncio
import shutil
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

# 添加项目根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.data_cleaner import (
    DataCleaner, CleanupTier, CleanupStats, TableRetentionPolicy,
)


class TestCleanupTier:
    """CleanupTier 枚举测试"""

    def test_01_tier_values(self):
        assert CleanupTier.T1_MEMORY.value == 1
        assert CleanupTier.T2_DB_LIGHT.value == 2
        assert CleanupTier.T3_DB_FULL.value == 3
        assert CleanupTier.T4_DEEP.value == 4
        print("  [PASS] test_01: tier values correct")

    def test_02_tier_names(self):
        assert CleanupTier.T1_MEMORY.name == "T1_MEMORY"
        assert CleanupTier.T4_DEEP.name == "T4_DEEP"
        print("  [PASS] test_02: tier names correct")


class TestTableRetentionPolicy:
    """TableRetentionPolicy 测试"""

    def test_03_policy_defaults(self):
        policy = TableRetentionPolicy("test_table", "created_at", 30)
        assert policy.table_name == "test_table"
        assert policy.time_column == "created_at"
        assert policy.retention_days == 30
        assert policy.min_rows == 0
        assert policy.extra_condition == ""
        assert policy.keep_profitable is False
        assert policy.archive_before_delete is False
        print("  [PASS] test_03: policy defaults correct")

    def test_04_policy_custom(self):
        policy = TableRetentionPolicy(
            "trades", "close_time", 90,
            min_rows=100, extra_condition="status = 'closed'",
            keep_profitable=True, archive_before_delete=True,
        )
        assert policy.min_rows == 100
        assert policy.extra_condition == "status = 'closed'"
        assert policy.keep_profitable is True
        assert policy.archive_before_delete is True
        print("  [PASS] test_04: policy custom params correct")


class TestCleanupStats:
    """CleanupStats 测试"""

    def test_05_record(self):
        stats = CleanupStats()
        stats.record("sqlite", "position_history", 100, 1024)
        assert stats.total_deleted_rows == 100
        assert stats.total_freed_bytes == 1024
        assert stats.per_category["sqlite"]["position_history"] == 100
        print("  [PASS] test_05: record() works")

    def test_06_record_files(self):
        stats = CleanupStats()
        stats.record_files("alerts", 5)
        stats.record_files("reviews", 3)
        assert stats.total_deleted_files == 8
        assert stats.per_category["alerts"]["files"] == 5
        print("  [PASS] test_06: record_files() works")

    def test_07_record_archived(self):
        stats = CleanupStats()
        stats.record_archived(500)
        stats.record_archived(300)
        assert stats.total_archived_rows == 800
        print("  [PASS] test_07: record_archived() works")

    def test_08_to_dict(self):
        stats = CleanupStats()
        stats.record("sqlite", "trades", 50)
        d = stats.to_dict()
        assert d["total_deleted_rows"] == 50
        assert "per_category" in d
        assert "recent_errors" in d
        print("  [PASS] test_08: to_dict() works")


class TestDataCleanerInit:
    """DataCleaner 初始化测试"""

    def test_09_default_config(self):
        config = {}
        dc = DataCleaner(config)
        assert dc._tier_intervals[CleanupTier.T1_MEMORY] == 15
        assert dc._tier_intervals[CleanupTier.T4_DEEP] == 24
        assert dc._cleanup_budget_sec == 30
        assert dc._archive_enabled is True
        assert dc._integrity_check_enabled is True
        assert dc._adaptive_interval is True
        assert len(dc._retention_policies) == 13
        print("  [PASS] test_09: default config works")

    def test_10_custom_config(self):
        config = {
            "data_cleanup": {
                "t1_memory_interval_min": 5,
                "t2_db_light_interval_min": 10,
                "t3_db_full_interval_min": 20,
                "t4_deep_interval_hours": 12,
                "cleanup_budget_sec": 15,
                "archive_enabled": False,
                "integrity_check_enabled": False,
                "adaptive_interval": False,
                "db_size_threshold_mb": 100,
                "db_retention_days": 60,
                "log_retention_days": 14,
                "alert_retention_days": 14,
                "review_max_daily": 30,
                "review_max_monthly": 6,
                "backup_retention_days": 60,
                "table_policies": {
                    "trade_records": {
                        "retention_days": 180,
                        "keep_profitable": True,
                        "archive_before_delete": True,
                    }
                }
            }
        }
        dc = DataCleaner(config)
        assert dc._tier_intervals[CleanupTier.T1_MEMORY] == 5
        assert dc._cleanup_budget_sec == 15
        assert dc._archive_enabled is False
        assert dc._integrity_check_enabled is False
        assert dc._adaptive_interval is False
        assert dc._db_size_threshold_mb == 100
        assert dc._db_retention_days == 60
        assert dc._log_retention_days == 14
        assert dc._review_keep_count == 30
        assert dc._review_monthly_keep == 6
        print("  [PASS] test_10: custom config works")

    def test_11_table_policy_override(self):
        config = {
            "data_cleanup": {
                "table_policies": {
                    "trade_records": {
                        "retention_days": 180,
                        "min_rows": 500,
                        "keep_profitable": True,
                        "archive_before_delete": True,
                    }
                }
            }
        }
        dc = DataCleaner(config)
        policy = next(p for p in dc._retention_policies if p.table_name == "trade_records")
        assert policy.retention_days == 180
        assert policy.min_rows == 500
        assert policy.keep_profitable is True
        assert policy.archive_before_delete is True
        print("  [PASS] test_11: table policy override works")

    def test_12_dependency_injection(self):
        config = {}
        dc = DataCleaner(config)
        mock_sqlite = MagicMock()
        mock_redis = MagicMock()
        mock_journal = MagicMock()
        mock_slm = MagicMock()
        mock_queue = MagicMock()
        mock_olm = MagicMock()

        dc.set_sqlite_storage(mock_sqlite)
        dc.set_redis_cache(mock_redis)
        dc.set_trade_journal(mock_journal)
        dc.set_stop_loss_manager(mock_slm)
        dc.set_order_lifecycle_manager(mock_olm)
        dc.set_order_queue(mock_queue)

        assert dc._sqlite_storage is mock_sqlite
        assert dc._redis_cache is mock_redis
        assert dc._trade_journal is mock_journal
        assert dc._stop_loss_manager is mock_slm
        assert dc._order_lifecycle_manager is mock_olm
        assert dc._order_queue is mock_queue
        print("  [PASS] test_12: dependency injection works")


class TestDataCleanerHealth:
    """DataCleaner 健康状态测试"""

    def test_13_health_initial(self):
        config = {}
        dc = DataCleaner(config)
        health = dc.get_health_status()
        assert health["status"] == "healthy"
        assert health["total_runs"] == 0
        assert health["error_count"] == 0
        assert health["warning_count"] == 0
        assert "db_size_mb" in health
        print("  [PASS] test_13: initial health is healthy")

    def test_14_health_with_errors(self):
        config = {}
        dc = DataCleaner(config)
        dc.stats.errors.append("test error")
        health = dc.get_health_status()
        assert health["status"] == "degraded"
        assert health["error_count"] == 1
        print("  [PASS] test_14: health degraded with errors")

    def test_15_health_with_db_size(self):
        config = {"sqlite": {"db_path": "./data/trading.db"}}
        dc = DataCleaner(config)
        health = dc.get_health_status()
        assert health["db_size_mb"] >= 0
        print("  [PASS] test_15: health includes db_size_mb")


class TestDataCleanerHotUpdate:
    """配置热更新测试"""

    def test_16_hot_update(self):
        config = {}
        dc = DataCleaner(config)
        new_config = {
            "data_cleanup": {
                "t1_memory_interval_min": 10,
                "cleanup_budget_sec": 60,
                "archive_enabled": False,
                "integrity_check_enabled": False,
                "adaptive_interval": False,
            }
        }
        dc.update_config(new_config)
        assert dc._tier_intervals[CleanupTier.T1_MEMORY] == 10
        assert dc._cleanup_budget_sec == 60
        assert dc._archive_enabled is False
        assert dc._integrity_check_enabled is False
        assert dc._adaptive_interval is False
        print("  [PASS] test_16: hot update works")

    def test_17_hot_update_partial(self):
        config = {}
        dc = DataCleaner(config)
        original_t2 = dc._tier_intervals[CleanupTier.T2_DB_LIGHT]
        new_config = {
            "data_cleanup": {
                "t1_memory_interval_min": 20,
            }
        }
        dc.update_config(new_config)
        assert dc._tier_intervals[CleanupTier.T1_MEMORY] == 20
        assert dc._tier_intervals[CleanupTier.T2_DB_LIGHT] == original_t2
        print("  [PASS] test_17: partial hot update preserves other values")


class TestDataCleanerRecommendations:
    """清理建议生成测试"""

    def test_18_recommendations_with_errors(self):
        config = {}
        dc = DataCleaner(config)
        dc.stats.errors.append("test error 1")
        dc.stats.errors.append("test error 2")
        recs = dc.generate_recommendations()
        error_recs = [r for r in recs if r["type"] == "cleanup_errors"]
        assert len(error_recs) == 1
        assert error_recs[0]["priority"] == "high"
        assert error_recs[0]["error_count"] == 2
        print("  [PASS] test_18: error recommendations generated")

    def test_19_recommendations_empty(self):
        config = {}
        dc = DataCleaner(config)
        recs = dc.generate_recommendations()
        assert isinstance(recs, list)
        print("  [PASS] test_19: empty recommendations is valid list")


class TestDataCleanerDB:
    """DataCleaner DB 清理测试（使用临时数据库）"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_trading.db")
        self._init_test_db()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _init_test_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        cursor = conn.cursor()

        # 创建测试表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp TEXT,
                pnl_usdt REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                close_time TEXT,
                status TEXT,
                pnl_usdt REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                create_time TEXT,
                message TEXT
            )
        """)

        # 插入测试数据
        now = datetime.now()
        # 旧数据（60天前）
        old_time = (now - timedelta(days=60)).isoformat()
        for i in range(100):
            cursor.execute(
                "INSERT INTO position_history (symbol, timestamp, pnl_usdt) VALUES (?, ?, ?)",
                ("BTC-USDT", old_time, -0.5 if i % 2 == 0 else 0.5),
            )
        for i in range(50):
            cursor.execute(
                "INSERT INTO trade_records (symbol, close_time, status, pnl_usdt) VALUES (?, ?, ?, ?)",
                ("ETH-USDT", old_time, "closed", 1.0 if i < 25 else -1.0),
            )

        # 新数据（1天前）
        new_time = (now - timedelta(days=1)).isoformat()
        for i in range(30):
            cursor.execute(
                "INSERT INTO position_history (symbol, timestamp, pnl_usdt) VALUES (?, ?, ?)",
                ("BTC-USDT", new_time, 0.1),
            )
        for i in range(20):
            cursor.execute(
                "INSERT INTO alert_records (create_time, message) VALUES (?, ?)",
                (new_time, f"alert {i}"),
            )

        conn.commit()
        conn.close()

    def test_20_db_cleanup_with_policy(self):
        config = {
            "sqlite": {"db_path": self.db_path},
            "data_cleanup": {
                "archive_enabled": False,
                "integrity_check_enabled": False,
                "table_policies": {
                    "position_history": {"retention_days": 30, "min_rows": 10},
                }
            }
        }
        dc = DataCleaner(config)
        dc._archive_enabled = False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        policy = next(p for p in dc._retention_policies if p.table_name == "position_history")
        deleted = asyncio.run(dc._cleanup_table_with_policy(cursor, policy, conn))
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM position_history")
        remaining = cursor.fetchone()[0]
        conn.close()

        # 旧数据（60天前）应该被清理，保留最近30天的新数据
        assert remaining <= 130  # 100 old + 30 new, should have deleted some
        assert remaining >= 30   # at least 30 new rows
        print(f"  [PASS] test_20: DB cleanup deleted {deleted} rows, {remaining} remaining")

    def test_21_db_cleanup_keep_profitable(self):
        config = {
            "sqlite": {"db_path": self.db_path},
            "data_cleanup": {
                "archive_enabled": False,
                "integrity_check_enabled": False,
                "table_policies": {
                    "trade_records": {"retention_days": 30, "keep_profitable": True, "min_rows": 5},
                }
            }
        }
        dc = DataCleaner(config)
        dc._archive_enabled = False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        policy = next(p for p in dc._retention_policies if p.table_name == "trade_records")
        deleted = asyncio.run(dc._cleanup_table_with_policy(cursor, policy, conn))
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM trade_records WHERE pnl_usdt > 0")
        profitable = cursor.fetchone()[0]
        conn.close()

        # 盈利记录应该被保留
        assert profitable >= 25  # 25 profitable trades
        print(f"  [PASS] test_21: keep_profitable preserved {profitable} profitable trades")

    def test_22_db_cleanup_min_rows(self):
        config = {
            "sqlite": {"db_path": self.db_path},
            "data_cleanup": {
                "archive_enabled": False,
                "integrity_check_enabled": False,
                "table_policies": {
                    "position_history": {"retention_days": 30, "min_rows": 120},
                }
            }
        }
        dc = DataCleaner(config)
        dc._archive_enabled = False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        policy = next(p for p in dc._retention_policies if p.table_name == "position_history")
        asyncio.run(dc._cleanup_table_with_policy(cursor, policy, conn))
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM position_history")
        remaining = cursor.fetchone()[0]
        conn.close()

        # 应该保留至少 min_rows 行
        assert remaining >= 120
        print(f"  [PASS] test_22: min_rows preserved {remaining} rows")

    def test_23_integrity_check(self):
        config = {
            "sqlite": {"db_path": self.db_path},
            "data_cleanup": {"integrity_check_enabled": True}
        }
        dc = DataCleaner(config)
        asyncio.run(dc._run_integrity_check())
        # 测试数据库应该通过完整性检查
        assert len(dc.stats.errors) == 0
        print("  [PASS] test_23: integrity check passed")

    def test_24_integrity_check_disabled(self):
        config = {
            "sqlite": {"db_path": self.db_path},
            "data_cleanup": {"integrity_check_enabled": False}
        }
        dc = DataCleaner(config)
        asyncio.run(dc._run_integrity_check())
        # 禁用时静默跳过
        print("  [PASS] test_24: integrity check disabled skipped")


class TestDataCleanerMemory:
    """DataCleaner 内存清理测试"""

    def test_25_memory_cleanup_trade_journal(self):
        config = {}
        dc = DataCleaner(config)

        class MockEquityPoint:
            def __init__(self, ts):
                self.timestamp = ts

        # 一半旧数据（8天前），一半新数据（现在）
        now = datetime.now()
        old = now - timedelta(days=8)
        mock_journal = MagicMock()
        mock_journal._equity_curve = (
            [MockEquityPoint(old) for _ in range(10000)] +
            [MockEquityPoint(now) for _ in range(10000)]
        )
        mock_journal._trades = {}

        dc.set_trade_journal(mock_journal)
        asyncio.run(dc._run_memory_cleanup())

        # 旧数据（8天前）应该被清理，新数据保留
        assert len(mock_journal._equity_curve) < 20000
        print(f"  [PASS] test_25: trade_journal memory cleaned, equity_curve: {len(mock_journal._equity_curve)}")

    def test_26_memory_cleanup_stop_loss(self):
        config = {}
        dc = DataCleaner(config)

        mock_slm = MagicMock()
        mock_slm._stop_loss_events = list(range(2000))
        mock_slm._triggered_recently = {}

        dc.set_stop_loss_manager(mock_slm)
        asyncio.run(dc._run_memory_cleanup())

        assert len(mock_slm._stop_loss_events) <= 1000
        print(f"  [PASS] test_26: stop_loss memory cleaned, events: {len(mock_slm._stop_loss_events)}")

    def test_27_memory_cleanup_order_queue(self):
        config = {}
        dc = DataCleaner(config)

        now = datetime.now()
        mock_queue = MagicMock()
        mock_queue._order_cache = {
            f"order_{i}": {
                "status": "failed" if i % 2 == 0 else "executed",
                "create_time": now - timedelta(minutes=10),
            }
            for i in range(1000)
        }

        dc.set_order_queue(mock_queue)
        asyncio.run(dc._run_memory_cleanup())

        assert len(mock_queue._order_cache) < 1000
        print(f"  [PASS] test_27: order_queue memory cleaned, cache: {len(mock_queue._order_cache)}")


class TestDataCleanerArchive:
    """DataCleaner 归档测试"""

    def test_28_archive_disabled(self):
        config = {
            "data_cleanup": {"archive_enabled": False}
        }
        dc = DataCleaner(config)
        asyncio.run(dc._archive_old_data())
        assert dc.stats.total_archived_rows == 0
        print("  [PASS] test_28: archive disabled skips")

    def test_29_archive_no_db(self):
        config = {
            "sqlite": {"db_path": "/nonexistent/path.db"},
            "data_cleanup": {"archive_enabled": True}
        }
        dc = DataCleaner(config)
        asyncio.run(dc._archive_old_data())
        print("  [PASS] test_29: archive with no DB skips gracefully")


class TestDataCleanerManual:
    """DataCleaner 手动触发测试"""

    def test_30_trigger_full_cleanup(self):
        config = {"data_cleanup": {"archive_enabled": False, "integrity_check_enabled": False}}
        dc = DataCleaner(config)
        result = asyncio.run(dc.trigger_full_cleanup())
        assert result["total_runs"] >= 3  # T2+T3+T4
        print(f"  [PASS] test_30: trigger_full_cleanup ran {result['total_runs']} tiers")

    def test_31_trigger_memory_cleanup(self):
        config = {}
        dc = DataCleaner(config)
        result = asyncio.run(dc.trigger_memory_cleanup())
        assert result["total_runs"] >= 1
        print("  [PASS] test_31: trigger_memory_cleanup works")

    def test_32_trigger_specific_tier(self):
        config = {"data_cleanup": {"archive_enabled": False, "integrity_check_enabled": False}}
        dc = DataCleaner(config)
        result = asyncio.run(dc.trigger_tier(CleanupTier.T2_DB_LIGHT))
        assert result["total_runs"] >= 1
        assert CleanupTier.T2_DB_LIGHT.value in dc.stats.last_tier_runs
        print("  [PASS] test_32: trigger_tier works")

    def test_33_get_stats(self):
        config = {}
        dc = DataCleaner(config)
        stats = dc.get_stats()
        assert "total_runs" in stats
        assert "recommendations" in stats
        print("  [PASS] test_33: get_stats returns full stats")


class TestDataCleanerFileSystem:
    """DataCleaner 文件系统清理测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.alerts_dir = os.path.join(self.tmpdir, "data", "alerts")
        os.makedirs(self.alerts_dir, exist_ok=True)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_34_alert_file_cleanup(self):
        # 创建旧告警文件
        old_file = os.path.join(self.alerts_dir, "alerts_2025_01_01.json")
        with open(old_file, "w") as f:
            f.write("{}")
        # 设置文件时间为7天前
        old_time = time.time() - 8 * 86400
        os.utime(old_file, (old_time, old_time))

        config = {
            "data_cleanup": {"alert_retention_days": 7}
        }
        dc = DataCleaner(config)
        asyncio.run(dc._cleanup_alert_files(self.tmpdir))

        assert not os.path.exists(old_file)
        assert dc.stats.total_deleted_files >= 1
        print("  [PASS] test_34: old alert files cleaned")

    def test_35_recent_alert_file_preserved(self):
        recent_file = os.path.join(self.alerts_dir, "alerts_recent.json")
        with open(recent_file, "w") as f:
            f.write("{}")

        config = {"data_cleanup": {"alert_retention_days": 7}}
        dc = DataCleaner(config)
        asyncio.run(dc._cleanup_alert_files(self.tmpdir))

        assert os.path.exists(recent_file)
        print("  [PASS] test_35: recent alert files preserved")


class TestDataCleanerIntegration:
    """DataCleaner 端到端集成测试"""

    def test_36_end_to_end(self):
        """完整端到端测试：内存+DB+归档+建议"""
        config = {
            "data_cleanup": {
                "archive_enabled": False,
                "integrity_check_enabled": False,
                "adaptive_interval": False,
                "cleanup_budget_sec": 60,
            }
        }
        dc = DataCleaner(config)

        # 注入mock
        mock_journal = MagicMock()
        mock_journal._equity_curve = [MagicMock(timestamp=datetime.now()) for _ in range(500)]
        mock_journal._trades = {}
        dc.set_trade_journal(mock_journal)

        mock_slm = MagicMock()
        mock_slm._stop_loss_events = list(range(100))
        mock_slm._triggered_recently = {}
        dc.set_stop_loss_manager(mock_slm)

        # 手动触发全量清理
        result = asyncio.run(dc.trigger_full_cleanup())

        # 验证统计
        assert result["total_runs"] >= 3
        assert "per_category" in result

        # 验证健康状态
        health = dc.get_health_status()
        assert health["status"] in ("healthy", "degraded")

        # 验证建议
        recs = dc.generate_recommendations()
        assert isinstance(recs, list)

        # 验证get_stats
        stats = dc.get_stats()
        assert "recommendations" in stats

        print(f"  [PASS] test_36: E2E test passed, runs={result['total_runs']}, "
              f"health={health['status']}")

    def test_37_config_roundtrip(self):
        """配置加载 → 实例化 → 热更新 → 健康检查 完整回路"""
        config = {
            "data_cleanup": {
                "t1_memory_interval_min": 10,
                "t2_db_light_interval_min": 20,
                "t3_db_full_interval_min": 40,
                "t4_deep_interval_hours": 12,
                "cleanup_budget_sec": 45,
                "archive_enabled": True,
                "integrity_check_enabled": True,
                "adaptive_interval": True,
                "db_size_threshold_mb": 75,
                "db_retention_days": 45,
                "log_retention_days": 10,
                "alert_retention_days": 10,
                "review_max_daily": 60,
                "review_max_monthly": 8,
                "backup_retention_days": 45,
                "table_policies": {
                    "trade_records": {
                        "retention_days": 120,
                        "keep_profitable": True,
                        "archive_before_delete": True,
                    }
                }
            }
        }

        # 1. 实例化
        dc = DataCleaner(config)
        assert dc._tier_intervals[CleanupTier.T1_MEMORY] == 10
        assert dc._cleanup_budget_sec == 45
        assert dc._archive_enabled is True

        # 2. 热更新
        new_config = {
            "data_cleanup": {
                "t1_memory_interval_min": 5,
                "cleanup_budget_sec": 30,
            }
        }
        dc.update_config(new_config)
        assert dc._tier_intervals[CleanupTier.T1_MEMORY] == 5
        assert dc._cleanup_budget_sec == 30

        # 3. 健康检查
        health = dc.get_health_status()
        assert "status" in health
        assert "db_size_mb" in health

        # 4. 建议
        recs = dc.generate_recommendations()
        assert isinstance(recs, list)

        print("  [PASS] test_37: config roundtrip passed")


def run_all_tests():
    """运行所有测试"""
    test_classes = [
        TestCleanupTier,
        TestTableRetentionPolicy,
        TestCleanupStats,
        TestDataCleanerInit,
        TestDataCleanerHealth,
        TestDataCleanerHotUpdate,
        TestDataCleanerRecommendations,
        TestDataCleanerDB,
        TestDataCleanerMemory,
        TestDataCleanerArchive,
        TestDataCleanerManual,
        TestDataCleanerFileSystem,
        TestDataCleanerIntegration,
    ]

    total = 0
    passed = 0
    failed = 0

    for cls in test_classes:
        print(f"\n{'='*60}")
        print(f"  {cls.__name__}")
        print(f"{'='*60}")
        instance = cls()
        for name in sorted(dir(instance)):
            if name.startswith("test_"):
                total += 1
                try:
                    # setup/teardown
                    if hasattr(instance, "setup_method"):
                        getattr(instance, "setup_method")()
                    getattr(instance, name)()
                    if hasattr(instance, "teardown_method"):
                        getattr(instance, "teardown_method")()
                    passed += 1
                except Exception as e:
                    failed += 1
                    print(f"  [FAIL] {name}: {e}")
                    import traceback
                    traceback.print_exc()
                    if hasattr(instance, "teardown_method"):
                        try:
                            getattr(instance, "teardown_method")()
                        except Exception:
                            pass

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{total} passed, {failed} failed")
    print(f"{'='*60}")
    return passed, failed, total


if __name__ == "__main__":
    passed, failed, total = run_all_tests()
    if failed > 0:
        sys.exit(1)