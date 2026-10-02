"""ExpiredResidueCleaner 过期残留自动清理器 — 单元测试。

覆盖矩阵：
  P0  配置解析（默认值 + 覆盖）
  P0  cron 表达式匹配
  P0  DryRun：只输出清单，不真实删除
  P0  软归档：先移动归档，软保留到期后物理删除
  P0  白名单保护：订单/成交/仓位/风控审计/WAL 绝不删除
  P0  人工保留标记（.keep / KEEP）跳过
  P0  实盘运行中跳过
  P0  总开关 / 紧急暂停开关
  P0  阈值触发（磁盘/内存）
  P0  时钟回拨跳过（防误删）
  P0  文件 mtime 超前（未来漂移）跳过
  P1  内存资源 scanner/purge 回收
"""

import os
import sys
import time
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.expired_residue_cleaner import (
    ExpiredResidueCleaner,
    ResidueItem,
    ResidueCategory,
)


def _cfg(tmp_path, **overrides):
    """构造指向 tmp_path 的清理配置。"""
    cfg = {
        "expired_residue_cleaner": {
            "enable_cleaner": True,
            "dry_run": False,
            "emergency_pause": False,
            "cron_expression": "0 3 * * *",
            "disk_threshold_pct": 95,
            "memory_threshold_pct": 95,
            "min_interval_sec": 0,
            "archive_dir": "./archive",
            "soft_retention_sec": 0,
            "state_file": "./state.json",
            "audit_file": "./audit.jsonl",
            "whitelist_patterns": ["*trading.db*", "*audit*"],
            "manual_keep_markers": [".keep", "KEEP"],
            "categories": {
                "lock_file": {
                    "enabled": True,
                    "ttl_sec": 3600,
                    "dirs": ["./data/locks"],
                    "patterns": ["*.lock"],
                },
                "trace_log": {
                    "enabled": True,
                    "ttl_sec": 86400,
                    "dirs": ["./logs/trace"],
                    "patterns": ["*.log"],
                },
            },
        }
    }
    for k, v in overrides.items():
        cfg["expired_residue_cleaner"][k] = v
    return cfg


def _make(tmp_path, **overrides):
    """构造 base_dir 指向 tmp_path 的清理器。"""
    return ExpiredResidueCleaner(_cfg(tmp_path, **overrides), base_dir=str(tmp_path))


def _old_file(path, age_sec):
    """创建 mtime 为过去 age_sec 秒的文件。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("x" * 16)
    old = time.time() - age_sec
    os.utime(path, (old, old))
    return path


# ═══════════════════════════════════════════════════════════════
# P0: 配置解析
# ═══════════════════════════════════════════════════════════════

class TestConfig:
    def test_defaults(self):
        c = ExpiredResidueCleaner({})
        assert c.enable_cleaner is True
        assert c.dry_run is True
        assert c.emergency_pause is False
        assert c.cron_expression == "0 3 * * *"
        assert c.soft_retention_sec == 7 * 86400

    def test_overrides(self, tmp_path):
        c = _make(tmp_path)
        assert c.dry_run is False
        assert c.soft_retention_sec == 0
        assert c._categories["lock_file"].ttl_sec == 3600
        assert c._categories["lock_file"].dirs == ["./data/locks"]

    def test_all_six_categories_present(self, tmp_path):
        c = _make(tmp_path)
        for cat in ResidueCategory:
            assert cat.value in c._categories


# ═══════════════════════════════════════════════════════════════
# P0: cron 匹配
# ═══════════════════════════════════════════════════════════════

class TestCron:
    def test_daily_at_3am(self):
        assert ExpiredResidueCleaner.cron_matches("0 3 * * *", datetime(2026, 9, 14, 3, 0))
        assert not ExpiredResidueCleaner.cron_matches("0 3 * * *", datetime(2026, 9, 14, 4, 0))

    def test_every_15_min(self):
        assert ExpiredResidueCleaner.cron_matches("*/15 * * * *", datetime(2026, 9, 14, 10, 30))
        assert not ExpiredResidueCleaner.cron_matches("*/15 * * * *", datetime(2026, 9, 14, 10, 7))

    def test_weekday_only(self):
        # dow: 0=周日 1=周一…6=周六；1-5 = 周一~周五
        assert ExpiredResidueCleaner.cron_matches("0 9 * * 1-5", datetime(2026, 9, 14, 9, 0))  # 周一
        assert not ExpiredResidueCleaner.cron_matches("0 9 * * 1-5", datetime(2026, 9, 13, 9, 0))  # 周日

    def test_invalid(self):
        assert not ExpiredResidueCleaner.cron_matches("bad", datetime.now())


# ═══════════════════════════════════════════════════════════════
# P0: DryRun
# ═══════════════════════════════════════════════════════════════

class TestDryRun:
    def test_dry_run_does_not_delete(self, tmp_path):
        c = _make(tmp_path, dry_run=True)
        f = _old_file(os.path.join(tmp_path, "data", "locks", "a.lock"), age_sec=7200)
        rep = c.run()
        assert rep.scanned_total >= 1
        assert os.path.exists(f), "DryRun 不应删除文件"
        assert rep.archived_count == 0
        assert rep.deleted_count == 0

    def test_real_run_archives(self, tmp_path):
        c = _make(tmp_path, dry_run=False)
        f = _old_file(os.path.join(tmp_path, "data", "locks", "a.lock"), age_sec=7200)
        rep = c.run()
        assert rep.archived_count == 1
        assert not os.path.exists(f), "过期文件应被移入归档目录"
        archive = os.path.join(tmp_path, "archive", "lock_file")
        assert len(os.listdir(archive)) == 1


# ═══════════════════════════════════════════════════════════════
# P0: 软删除（归档 → 软保留 → 物理删除）
# ═══════════════════════════════════════════════════════════════

class TestSoftDelete:
    def test_purge_after_soft_retention(self, tmp_path):
        c = _make(tmp_path, dry_run=False, soft_retention_sec=10)
        f = _old_file(os.path.join(tmp_path, "data", "locks", "a.lock"), age_sec=7200)
        c.run()
        assert not os.path.exists(f)

        archive_dir = os.path.join(tmp_path, "archive", "lock_file")
        archived_files = os.listdir(archive_dir)
        assert len(archived_files) == 1
        archived_path = os.path.join(archive_dir, archived_files[0])

        # 归档文件 mtime 被标记为归档时刻；模拟软保留期已过
        old = time.time() - 100
        os.utime(archived_path, (old, old))
        rep2 = c.run()
        assert rep2.deleted_count == 1
        assert not os.path.exists(archived_path)


# ═══════════════════════════════════════════════════════════════
# P0: 白名单 / 人工保留标记
# ═══════════════════════════════════════════════════════════════

class TestProtection:
    def test_whitelist_never_deleted(self, tmp_path):
        c = _make(tmp_path, dry_run=False)
        # 白名单文件（风控审计）放在被扫描目录且匹配 *.lock 模式，应被保护
        f = _old_file(os.path.join(tmp_path, "data", "locks", "risk_audit.lock"), age_sec=7200)
        rep = c.run()
        assert os.path.exists(f), "白名单文件绝不删除"
        assert rep.protected_skipped >= 1

    def test_manual_keep_marker(self, tmp_path):
        c = _make(tmp_path, dry_run=False)
        f = _old_file(os.path.join(tmp_path, "data", "locks", "keepme.keep"), age_sec=7200)
        c.run()
        assert os.path.exists(f), "人工保留标记文件绝不删除"


# ═══════════════════════════════════════════════════════════════
# P0: 实盘运行 / 开关
# ═══════════════════════════════════════════════════════════════

class TestSkipConditions:
    def test_live_strategy_skip(self, tmp_path):
        c = _make(tmp_path, dry_run=False)
        c.set_live_probe(lambda: True)
        f = _old_file(os.path.join(tmp_path, "data", "locks", "a.lock"), age_sec=7200)
        rep = c.run()
        assert rep.skipped and rep.skipped_reason == "live_strategy_running"
        assert os.path.exists(f), "实盘运行中禁止删除"

    def test_live_probe_exception_conservative_skip(self, tmp_path):
        c = _make(tmp_path, dry_run=False)
        c.set_live_probe(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        rep = c.run()
        assert rep.skipped

    def test_disabled(self, tmp_path):
        c = _make(tmp_path, enable_cleaner=False)
        rep = c.run()
        assert rep.skipped and rep.skipped_reason == "disabled"

    def test_emergency_pause(self, tmp_path):
        c = _make(tmp_path, emergency_pause=True)
        rep = c.run()
        assert rep.skipped and rep.skipped_reason == "emergency_pause"

    def test_min_interval_debounce(self, tmp_path):
        c = _make(tmp_path, dry_run=False, min_interval_sec=600)
        _old_file(os.path.join(tmp_path, "data", "locks", "a.lock"), age_sec=7200)
        c.run()  # 第一次运行成功
        rep = c.run()  # 间隔过短，应防抖跳过
        assert rep.skipped and rep.skipped_reason == "min_interval"


# ═══════════════════════════════════════════════════════════════
# P0: 阈值触发
# ═══════════════════════════════════════════════════════════════

class TestThreshold:
    def test_threshold_not_triggered(self, tmp_path):
        c = _make(tmp_path, disk_threshold_pct=100, memory_threshold_pct=100)
        assert c.threshold_triggered() is False

    def test_disk_threshold_triggered(self, tmp_path):
        c = _make(tmp_path, disk_threshold_pct=0)
        assert c.threshold_triggered() is True

    def test_auto_trigger_by_threshold(self, tmp_path):
        c = _make(tmp_path, disk_threshold_pct=0, memory_threshold_pct=100)
        assert c._should_trigger_auto() is True

    def test_auto_trigger_by_schedule(self, tmp_path):
        now = datetime.now()
        expr = f"{now.minute} {now.hour} * * *"
        c = _make(tmp_path, cron_expression=expr, disk_threshold_pct=100, memory_threshold_pct=100)
        assert c._should_trigger_auto() is True


# ═══════════════════════════════════════════════════════════════
# P0: 时钟漂移防护
# ═══════════════════════════════════════════════════════════════

class TestClockDrift:
    def test_backward_clock_skip(self, tmp_path):
        c = _make(tmp_path, dry_run=False)
        f = _old_file(os.path.join(tmp_path, "data", "locks", "a.lock"), age_sec=7200)

        # 第一次运行：now = T
        t0 = 1_700_000_000.0
        c.set_clock(lambda: t0)
        c.run()

        # 时钟回拨：now = T - 100
        c.set_clock(lambda: t0 - 100)
        rep = c.run()
        assert rep.skipped and rep.skipped_reason == "clock_drift"

    def test_future_mtime_skipped(self, tmp_path):
        c = _make(tmp_path, dry_run=False)
        f = os.path.join(tmp_path, "data", "locks", "future.lock")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("x" * 16)
        future = time.time() + 1000  # 超出 300s 容差
        os.utime(f, (future, future))
        rep = c.run()
        assert rep.drift_suspected >= 1
        assert os.path.exists(f), "未来时间戳文件不应删除"


# ═══════════════════════════════════════════════════════════════
# P1: 内存资源回收
# ═══════════════════════════════════════════════════════════════

class TestMemoryScanner:
    def test_memory_purge(self, tmp_path):
        c = _make(tmp_path, dry_run=False)
        purged = []
        c.register_memory_scanner(
            "kline_cache",
            lambda: [ResidueItem(
                category="kline_cache", key="kline:BTC", mtime=time.time() - 99999, size_bytes=0
            )],
            purge=lambda keys: purged.extend(keys),
        )
        rep = c.run()
        assert "kline:BTC" in purged
        assert rep.per_category.get("kline_cache", {}).get("purged", 0) == 1

    def test_memory_dry_run_no_purge(self, tmp_path):
        c = _make(tmp_path, dry_run=True)
        purged = []
        c.register_memory_scanner(
            "kline_cache",
            lambda: [ResidueItem(
                category="kline_cache", key="kline:BTC", mtime=time.time() - 99999, size_bytes=0
            )],
            purge=lambda keys: purged.extend(keys),
        )
        c.run()
        assert purged == []


# ═══════════════════════════════════════════════════════════════
# P0: 锁文件活性探测（避免误删当前进程持有的单实例锁）
# ═══════════════════════════════════════════════════════════════

class TestLockLiveness:
    def test_active_lock_protected(self, tmp_path):
        """锁内 PID 存活（当前进程）→ 保护，不清理。"""
        c = _make(tmp_path, dry_run=False)
        f = os.path.join(tmp_path, "data", "locks", "active.lock")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(str(os.getpid()))
        old = time.time() - 7200
        os.utime(f, (old, old))
        rep = c.run()
        assert os.path.exists(f), "活跃锁（PID 存活）不应被清理"
        assert rep.protected_skipped >= 1

    def test_dead_lock_cleaned(self, tmp_path):
        """锁内 PID 不存在（失效残留锁）→ 过期后正常清理。"""
        c = _make(tmp_path, dry_run=False)
        f = os.path.join(tmp_path, "data", "locks", "dead.lock")
        os.makedirs(os.path.dirname(f), exist_ok=True)
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(str(2**30))  # 远超 Windows PID 上限，必不存在
        old = time.time() - 7200
        os.utime(f, (old, old))
        rep = c.run()
        assert not os.path.exists(f), "失效残留锁（PID 不存在）应被清理"
        assert rep.archived_count >= 1

    def test_unparseable_lock_not_active(self, tmp_path):
        """锁文件内容无法解析为 PID → 视为非活跃锁，保持既有清理行为。"""
        c = _make(tmp_path, dry_run=False)
        f = _old_file(os.path.join(tmp_path, "data", "locks", "corrupt.lock"), age_sec=7200)
        rep = c.run()
        assert not os.path.exists(f), "无法解析 PID 的锁文件不应被当作活跃锁"
        assert rep.archived_count >= 1

    def test_injected_probe_active(self, tmp_path):
        """注入探测返回活跃 → 保护。"""
        c = _make(tmp_path, dry_run=False)
        c.set_lock_liveness_probe(lambda path: True)
        f = _old_file(os.path.join(tmp_path, "data", "locks", "a.lock"), age_sec=7200)
        rep = c.run()
        assert os.path.exists(f), "注入探测返回活跃时应保护"
        assert rep.protected_skipped >= 1

    def test_probe_exception_conservative_protect(self, tmp_path):
        """注入探测异常 → 保守保护。"""
        c = _make(tmp_path, dry_run=False)
        c.set_lock_liveness_probe(lambda path: (_ for _ in ()).throw(RuntimeError("boom")))
        f = _old_file(os.path.join(tmp_path, "data", "locks", "a.lock"), age_sec=7200)
        rep = c.run()
        assert os.path.exists(f), "探测异常应保守保护"
        assert rep.protected_skipped >= 1
