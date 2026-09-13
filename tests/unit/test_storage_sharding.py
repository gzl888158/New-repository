"""
存储层企业级强化测试：分库分表 + WAL + 时序化冷归档
===================================================

覆盖：
1. ShardRouter：表名校验 / 分片命名 / 分片发现 / 惰性建表
2. SQLiteStorage：订单按月分片归档(archive_closed_trades) + 跨分片查询
   (get_db_realized_pnl / get_closed_records_missing_pnl / update_trade_in_shard）
3. SQLiteStorage：WAL 状态校验 + checkpoint
4. TickPersistence：行情时序化冷归档（CSV.gz archive-then-delete）+ WAL
"""

import os
import sys
import gzip
import csv
import sqlite3
import tempfile
import shutil
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.sharding import ShardRouter, is_valid_identifier, month_suffix
from data.sqlite_storage import SQLiteStorage, TRADE_COLS
from market_data.tick_persistence import TickPersistence


class TestShardRouter:
    """ShardRouter 纯逻辑测试"""

    def test_01_valid_identifier(self):
        assert is_valid_identifier("trade_records") is True
        assert is_valid_identifier("trade_records_202609") is True
        assert is_valid_identifier("Trade_10") is True
        assert is_valid_identifier("") is False
        assert is_valid_identifier("t; DROP TABLE x") is False
        assert is_valid_identifier("2leading_digit") is False
        assert is_valid_identifier("trade-records") is False
        print("  [PASS] test_01: identifier validation blocks injection")

    def test_02_shard_name_format(self):
        router = ShardRouter("trade_records")
        dt = datetime(2026, 9, 10, 15, 30)
        assert router.shard_name(dt) == "trade_records_202609"
        assert router.shard_name(datetime(2025, 1, 2)) == "trade_records_202501"
        print("  [PASS] test_02: shard name format {base}_YYYYMM")

    def test_03_is_shard_matching(self):
        router = ShardRouter("trade_records")
        assert router.is_shard("trade_records_202609") is True
        assert router.is_shard("trade_records_202501") is True
        assert router.is_shard("trade_records") is False          # 热表非分片
        assert router.is_shard("trade_records_2099") is False     # 非6位月份
        assert router.is_shard("ticker_202609") is False          # 不同 base
        assert router.is_shard("trade_records_x202609") is False
        print("  [PASS] test_03: shard regex matching correct")

    def test_04_illegal_base_rejected(self):
        try:
            ShardRouter("t; DROP TABLE x")
            assert False, "应该拒绝非法 base 表名"
        except ValueError:
            pass
        print("  [PASS] test_04: illegal base rejected")

    def test_05_month_suffix(self):
        assert month_suffix(datetime(2026, 9, 1)) == "202609"
        assert month_suffix(datetime(2026, 12, 31)) == "202612"
        print("  [PASS] test_05: month_suffix correct")


class TestSQLiteOrderSharding:
    """订单分库分表 + 跨分片查询 + WAL 测试（临时文件库）"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "trading.db")
        self.storage = SQLiteStorage({"sqlite": {"db_path": self.db_path}})

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _insert_trade(self, trade_id, symbol, status, close_time, pnl=0.0, margin=100.0):
        self.storage.save_trade_record({
            "id": trade_id,
            "symbol": symbol,
            "strategy_name": "test",
            "side": "long",
            "order_type": "limit",
            "signal_type": "trend_entry",
            "quantity": 1.0,
            "price": 100.0,
            "filled_price": 100.0,
            "leverage": 1,
            "margin": margin,
            "pnl": pnl,
            "fees": 0.0,
            "status": status,
            "exit_reason": "manual",
            "create_time": close_time - timedelta(hours=1),
            "close_time": close_time,
        })

    def test_10_archive_closed_trades(self):
        # 旧已平仓订单（3个月前的月份）
        old = datetime.now() - timedelta(days=100)
        for i in range(5):
            self._insert_trade(f"old_{i}", "BTC-USDT", "closed", old, pnl=10.0)
        # 新已平仓订单（1天前，不应归档）
        recent = datetime.now() - timedelta(days=1)
        self._insert_trade("recent_1", "BTC-USDT", "closed", recent, pnl=5.0)

        result = self.storage.archive_closed_trades(before_days=30)
        assert result["archived"] == 5, f"应归档5条，实际 {result['archived']}"

        # 热表只剩近期记录
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        hot = cur.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
        assert hot == 1
        # 分片表存在且含5条
        ym = old.strftime("%Y%m")
        shard_count = cur.execute(f"SELECT COUNT(*) FROM trade_records_{ym}").fetchone()[0]
        conn.close()
        assert shard_count == 5
        print(f"  [PASS] test_10: archived {result['archived']} trades into shard {ym}")

    def test_11_get_db_realized_pnl_cross_shard(self):
        old = datetime.now() - timedelta(days=100)
        for i in range(3):
            self._insert_trade(f"old_pnl_{i}", "ETH-USDT", "closed", old, pnl=10.0)
        self._insert_trade("recent_pnl", "ETH-USDT", "closed",
                           datetime.now() - timedelta(days=1), pnl=7.0)

        self.storage.archive_closed_trades(before_days=30)
        total = self.storage.get_db_realized_pnl()
        assert abs(total - 37.0) < 1e-6, f"跨分片 pnl 应为37，实际 {total}"
        print(f"  [PASS] test_11: cross-shard realized pnl = {total}")

    def test_12_get_closed_records_missing_pnl_and_update(self):
        old = datetime.now() - timedelta(days=100)
        self._insert_trade("old_miss", "BTC-USDT", "closed", old, pnl=0.0)
        self.storage.archive_closed_trades(before_days=30)

        records = self.storage.get_closed_records_missing_pnl(limit=10)
        assert any(r["id"] == "old_miss" for r in records)
        rec = next(r for r in records if r["id"] == "old_miss")
        assert rec["_shard"].startswith("trade_records_"), f"应携带分片标识，实际 {rec['_shard']}"

        # 原地回写对应分片
        ok = self.storage.update_trade_in_shard(rec["_shard"], "old_miss", {"pnl": 12.5})
        assert ok is True
        total = self.storage.get_db_realized_pnl()
        assert abs(total - 12.5) < 1e-6
        print(f"  [PASS] test_12: missing-pnl located in shard + written back ({rec['_shard']})")

    def test_13_wal_status_enabled(self):
        status = self.storage.get_wal_status()
        assert status["wal_enabled"] is True, f"WAL 应已启用，实际 {status}"
        assert status["journal_mode"] == "wal"
        print(f"  [PASS] test_13: WAL enabled, -wal bytes={status.get('wal_file_bytes')}")

    def test_14_wal_checkpoint_runs(self):
        self._insert_trade("wal_1", "BTC-USDT", "closed", datetime.now(), pnl=1.0)
        result = self.storage.wal_checkpoint("TRUNCATE")
        # checkpoint 返回 3 元组列表 (busy, log, checkpointed)
        assert isinstance(result, list)
        assert len(result) >= 1
        assert len(result[0]) == 3
        print(f"  [PASS] test_14: WAL checkpoint returned {result}")


class TestTickPersistenceArchive:
    """行情时序化冷归档 + WAL 测试（临时文件库）"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "tick.db")
        self.archive_dir = os.path.join(self.tmpdir, "archive")
        self.tp = TickPersistence(db_path=self.db_path, config={
            "max_retention_days": 30,
            "archive_enabled": True,
            "archive_dir": self.archive_dir,
        })

    def teardown_method(self):
        self.tp._close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _insert_old_ticker(self, symbol, ts_ms):
        self.tp._connect()
        with self.tp._db_lock:
            self.tp._cursor.execute(
                "INSERT INTO ticker (symbol, timestamp, price, bid_price, ask_price, "
                "bid_volume, ask_volume, volume_24h, change_24h, high_24h, low_24h, "
                "funding_rate, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, ts_ms, 100.0, 99.0, 101.0, 1.0, 1.0, 1000.0, 0.0, 110.0, 90.0, 0.0, ts_ms // 1000),
            )
            self.tp._conn.commit()

    def test_20_archive_expired_data(self):
        # 90天前（超过30天保留期）旧数据
        old_ms = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
        for i in range(3):
            self._insert_old_ticker("BTC-USDT", old_ms + i)
        # 10天前（保留期内）数据不应归档
        recent_ms = int((datetime.now() - timedelta(days=10)).timestamp() * 1000)
        self._insert_old_ticker("BTC-USDT", recent_ms)

        result = self.tp._archive_expired_data()
        assert result["deleted"] == 3, f"应删除3条，实际 {result['deleted']}"
        assert len(result["files"]) == 1

        # 生成 CSV.gz 文件且包含表头 + 3 行数据
        fname = result["files"][0]
        assert os.path.exists(fname)
        assert fname.endswith(".csv.gz")
        with gzip.open(fname, "rt", encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4, f"CSV 应含表头+3行，实际 {len(rows)}"

        # 源表只剩近期数据
        self.tp._connect()
        with self.tp._db_lock:
            count = self.tp._cursor.execute("SELECT COUNT(*) FROM ticker").fetchone()[0]
        assert count == 1
        print(f"  [PASS] test_20: archived {result['deleted']} rows → {os.path.basename(fname)}")

    def test_21_archive_disabled_falls_back_to_delete(self):
        tp2 = TickPersistence(db_path=os.path.join(self.tmpdir, "tick2.db"), config={
            "max_retention_days": 30,
            "archive_enabled": False,
        })
        old_ms = int((datetime.now() - timedelta(days=90)).timestamp() * 1000)
        tp2._connect()
        with tp2._db_lock:
            tp2._cursor.execute(
                "INSERT INTO ticker (symbol, timestamp, price, bid_price, ask_price, "
                "bid_volume, ask_volume, volume_24h, change_24h, high_24h, low_24h, "
                "funding_rate, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                ("BTC-USDT", old_ms, 100.0, 99.0, 101.0, 1.0, 1.0, 1000.0, 0.0, 110.0, 90.0, 0.0, old_ms // 1000),
            )
            tp2._conn.commit()
        result = tp2._archive_expired_data()
        assert result["deleted"] == 0 and result["files"] == []
        tp2._connect()
        with tp2._db_lock:
            count = tp2._cursor.execute("SELECT COUNT(*) FROM ticker").fetchone()[0]
        tp2._close()
        assert count == 0  # 禁用归档时仍清理
        print("  [PASS] test_21: archive disabled falls back to plain delete")

    def test_22_wal_status_and_checkpoint(self):
        status = self.tp.get_wal_status()
        assert status["wal_enabled"] is True
        assert status["journal_mode"] == "wal"
        result = self.tp.wal_checkpoint("TRUNCATE")
        assert isinstance(result, list)
        assert len(result) >= 1
        print(f"  [PASS] test_22: tick WAL enabled + checkpoint returned {result}")


def run_all_tests():
    test_classes = [
        TestShardRouter,
        TestSQLiteOrderSharding,
        TestTickPersistenceArchive,
    ]
    total = passed = failed = 0
    for cls in test_classes:
        print(f"\n{'=' * 60}\n  {cls.__name__}\n{'=' * 60}")
        instance = cls()
        for name in sorted(dir(instance)):
            if not name.startswith("test_"):
                continue
            total += 1
            try:
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                getattr(instance, name)()
                if hasattr(instance, "teardown_method"):
                    instance.teardown_method()
                passed += 1
            except Exception as e:
                failed += 1
                print(f"  [FAIL] {name}: {e}")
                import traceback
                traceback.print_exc()
                if hasattr(instance, "teardown_method"):
                    try:
                        instance.teardown_method()
                    except Exception:
                        pass
    print(f"\n{'=' * 60}\n  Results: {passed}/{total} passed, {failed} failed\n{'=' * 60}")
    return passed, failed, total


if __name__ == "__main__":
    _, failed, _ = run_all_tests()
    if failed > 0:
        sys.exit(1)