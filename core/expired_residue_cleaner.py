"""
过期残留自动清理器 - ExpiredResidueCleaner

核心目标：
    定时 + 阈值触发，识别并清理过期临时残留；防止内存泄漏、磁盘膨胀；
    交易/成交/仓位/风控审计日志（白名单）只归档绝不删除。

清理候选对象（黑名单，可独立配置 TTL）：
    1. 超 TTL 临时 K 线缓存        (kline_cache)
    2. 完结/取消订单超期内存临时对象 (order_temp)
    3. 过期 trace 请求链路日志      (trace_log)
    4. 回测中间临时文件/锁文件/崩溃dump (backtest_temp)
    5. 宕机遗留失效分布式锁/文件锁    (lock_file)
    6. 已处理完成告警缓存            (alert_cache)

强制业务规则：
    - 黑白名单：白名单(订单事件/成交/仓位快照/风控审计/WAL)禁止删除，仅归档；
      黑名单为上述清理候选。
    - 多资源独立可配置 TTL；cron 定时 + 磁盘/内存占用阈值双触发。
    - 实盘策略运行中直接跳过本次清理，禁止删除。
    - DryRun 试运行：只输出待清理清单，不真实删除。
    - 软删除：先迁移归档目录，软保留期到期后物理删除。
    - 全程审计日志 + 异常告警。
    - 全局/分项/紧急暂停开关；低 IO/CPU 权重。
    - 时间戳漂移校验，规避误删。

设计原则：
    - 与 core.data_cleaner.DataCleaner 分工明确：DataCleaner 负责 DB 表级保留与
      业务数据归档；本模块只负责「临时残留」文件的软删除与内存临时对象回收。
    - 绝不触碰白名单路径；人工保留标记(.keep / KEEP)一律跳过。
    - 文件资源：扫描 → 过期判定 → 软归档(移动+归档时间戳) → 软保留期到期物理删除。
    - 内存资源：通过注入的 scanner/purge 回调回收，只审计、不落盘归档（临时对象）。
"""

import os
import re
import sys
import time
import json
import shutil
import asyncio
import fnmatch
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover - 环境未装 psutil 时优雅降级
    psutil = None
    _HAS_PSUTIL = False


# ──────────────────────────────────────────────────────────────────────────
# 枚举 / 数据类
# ──────────────────────────────────────────────────────────────────────────

class ResidueCategory(str, Enum):
    """清理候选资源类型（黑名单）。"""
    KLINE_CACHE = "kline_cache"
    ORDER_TEMP = "order_temp"
    TRACE_LOG = "trace_log"
    BACKTEST_TEMP = "backtest_temp"
    LOCK_FILE = "lock_file"
    ALERT_CACHE = "alert_cache"


@dataclass
class ResidueItem:
    """单个待清理候选对象。"""
    category: str
    key: str                          # 唯一标识（文件用绝对路径，内存对象用自定义 key）
    path: Optional[str] = None        # 文件绝对路径（内存对象为 None）
    mtime: float = 0.0                # epoch 秒；文件取 st_mtime，内存对象取创建/更新时间
    size_bytes: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)
    protected: bool = False           # 命中白名单 / 人工保留标记


@dataclass
class CategoryConfig:
    """单个资源类型的清理配置。"""
    category: str
    enabled: bool = True
    ttl_sec: float = 86400
    dirs: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)  # 文件名 glob 匹配，如 *.json


@dataclass
class CleanupReport:
    """单次清理审计报告。"""
    run_ts: float = 0.0
    trigger_reason: str = "manual"
    skipped: bool = False
    skipped_reason: str = ""
    scanned_total: int = 0
    expired_count: int = 0
    protected_skipped: int = 0
    drift_suspected: int = 0
    archived_count: int = 0
    deleted_count: int = 0
    freed_bytes: int = 0
    per_category: Dict[str, Dict[str, int]] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_ts": datetime.fromtimestamp(self.run_ts).isoformat() if self.run_ts else None,
            "trigger_reason": self.trigger_reason,
            "skipped": self.skipped,
            "skipped_reason": self.skipped_reason,
            "scanned_total": self.scanned_total,
            "expired_count": self.expired_count,
            "protected_skipped": self.protected_skipped,
            "drift_suspected": self.drift_suspected,
            "archived_count": self.archived_count,
            "deleted_count": self.deleted_count,
            "freed_bytes": self.freed_bytes,
            "per_category": dict(self.per_category),
            "errors": list(self.errors),
        }


# ──────────────────────────────────────────────────────────────────────────
# 清理器
# ──────────────────────────────────────────────────────────────────────────

class ExpiredResidueCleaner:
    """过期残留自动清理器。

    依赖注入点（均可选，未注入时优雅降级）：
        set_live_probe(callable -> bool)   实盘策略是否运行的探针
        set_alert_callback(callable)       异常告警回调
        set_audit_writer(callable)         自定义审计落盘（默认写 JSONL）
        register_memory_scanner(...)       内存资源扫描/回收回调
        set_clock(callable -> float)       时钟注入（测试用）
    """

    # 白名单默认模式：订单事件/成交/仓位快照/风控审计/WAL 等持久化数据，禁止删除
    DEFAULT_WHITELIST_PATTERNS = [
        "*trading.db*",           # SQLite 主库
        "*trading.db-wal",        # WAL 预写日志
        "*trading.db-shm",
        "*event_store*",          # 事件存储（订单/成交/仓位/风控审计）
        "*audit*",                # 风控审计日志
        "*position_snapshot*",    # 仓位快照
        "*trade_records*",
        "*order_events*",
    ]

    # 人工保留标记：文件名包含以下片段时一律跳过
    DEFAULT_KEEP_MARKERS = [".keep", "KEEP", "manual_override"]

    # 时间漂移容差（秒）：文件 mtime 超前当前时间超过该值视为漂移，跳过删除
    DEFAULT_CLOCK_DRIFT_TOLERANCE_SEC = 300

    def __init__(self, config: Dict[str, Any], base_dir: Optional[str] = None):
        cfg = config.get("expired_residue_cleaner", {}) if isinstance(config, dict) else {}

        self.base_dir = base_dir or os.getcwd()
        self.cfg = cfg

        # ── 全局 / 开关 ──
        self.enable_cleaner = bool(cfg.get("enable_cleaner", True))
        self.dry_run = bool(cfg.get("dry_run", True))
        self.emergency_pause = bool(cfg.get("emergency_pause", False))
        self.cron_expression = str(cfg.get("cron_expression", "0 3 * * *"))
        self.disk_threshold_pct = float(cfg.get("disk_threshold_pct", 85))
        self.memory_threshold_pct = float(cfg.get("memory_threshold_pct", 80))
        self.min_interval_sec = float(cfg.get("min_interval_sec", 600))
        self.check_interval_sec = float(cfg.get("check_interval_sec", 60))
        self.low_io_priority = bool(cfg.get("low_io_priority", True))
        self.io_budget_bytes = int(cfg.get("io_budget_bytes", 64 * 1024 * 1024))
        self.clock_drift_tolerance_sec = float(
            cfg.get("clock_drift_tolerance_sec", self.DEFAULT_CLOCK_DRIFT_TOLERANCE_SEC)
        )

        # ── 软删除 ──
        self.archive_dir = self._resolve(cfg.get("archive_dir", "./backups/residue_archive"))
        self.soft_retention_sec = float(cfg.get("soft_retention_sec", 7 * 86400))

        # ── 持久化 / 审计路径 ──
        self.state_file = self._resolve(cfg.get("state_file", "./data/expired_residue_cleaner_state.json"))
        self.audit_file = self._resolve(cfg.get("audit_file", "./data/expired_residue_cleaner_audit.jsonl"))

        # ── 黑白名单 ──
        self.whitelist_patterns = list(
            cfg.get("whitelist_patterns", self.DEFAULT_WHITELIST_PATTERNS)
        )
        self.keep_markers = list(cfg.get("manual_keep_markers", self.DEFAULT_KEEP_MARKERS))

        # ── 分项资源 TTL 配置 ──
        self._categories: Dict[str, CategoryConfig] = {}
        cat_cfg = cfg.get("categories", {}) or {}
        for cat in ResidueCategory:
            c = cat_cfg.get(cat.value, {})
            self._categories[cat.value] = CategoryConfig(
                category=cat.value,
                enabled=bool(c.get("enabled", True)),
                ttl_sec=float(c.get("ttl_sec", 86400)),
                dirs=list(c.get("dirs", [])),
                patterns=list(c.get("patterns", ["*"])),
            )

        # ── 依赖注入 ──
        self._live_probe: Optional[Callable[[], bool]] = None
        self._lock_liveness_probe: Optional[Callable[[str], bool]] = None
        self._alert_callback: Optional[Callable] = None
        self._audit_writer: Optional[Callable[[Dict[str, Any]], None]] = None
        self._clock: Callable[[], float] = time.time
        # category -> (scanner_callable, purge_callable)
        self._memory_scanners: Dict[str, Any] = {}

        # ── 运行时状态 ──
        self._last_run_ts: float = 0.0
        self._load_state()

    # ──────────────────────────────────────────────────────────────
    # 依赖注入
    # ──────────────────────────────────────────────────────────────
    def set_live_probe(self, fn: Callable[[], bool]) -> None:
        self._live_probe = fn

    def set_alert_callback(self, fn: Callable) -> None:
        self._alert_callback = fn

    def set_audit_writer(self, fn: Callable[[Dict[str, Any]], None]) -> None:
        self._audit_writer = fn

    def set_clock(self, fn: Callable[[], float]) -> None:
        self._clock = fn

    def set_lock_liveness_probe(self, fn: Callable[[str], bool]) -> None:
        """注入锁文件活性探测。fn(path) 返回 True 表示该锁仍被活跃进程持有（应保护）。"""
        self._lock_liveness_probe = fn

    def register_memory_scanner(
        self,
        category: str,
        scanner: Callable[[], List[ResidueItem]],
        purge: Optional[Callable[[List[str]], None]] = None,
    ) -> None:
        """注册内存资源回收器。scanner 返回待清理的 ResidueItem 列表，
        purge 接收过期 item 的 key 列表执行真实回收（None 则只审计不回收）。"""
        self._memory_scanners[category] = (scanner, purge)

    # ──────────────────────────────────────────────────────────────
    # 实盘探测 / 触发判定
    # ──────────────────────────────────────────────────────────────
    def is_live_strategy_running(self) -> bool:
        if self._live_probe is None:
            return False
        try:
            return bool(self._live_probe())
        except Exception as e:
            logger.warning(f"ExpiredResidueCleaner live probe error: {e}")
            # 探针异常时保守返回 True，宁可跳过清理也不误删
            return True

    def _disk_usage_pct(self) -> float:
        try:
            usage = shutil.disk_usage(self.base_dir)
            return (usage.used / usage.total) * 100.0
        except Exception:
            return 0.0

    def _memory_usage_pct(self) -> float:
        if not _HAS_PSUTIL:
            return 0.0
        try:
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0

    def threshold_triggered(self) -> bool:
        """磁盘或内存占用超过阈值则触发。"""
        if self._disk_usage_pct() >= self.disk_threshold_pct:
            return True
        if self.memory_threshold_pct > 0 and self._memory_usage_pct() >= self.memory_threshold_pct:
            return True
        return False

    @staticmethod
    def _cron_field_match(field: str, value: int, domain_min: int, domain_max: int) -> bool:
        if field == "*":
            return True
        for part in field.split(","):
            part = part.strip()
            if not part:
                continue
            step = 1
            if "/" in part:
                base, _, step_s = part.partition("/")
                step = int(step_s)
            else:
                base = part
            if base == "*":
                start, end = domain_min, domain_max
            elif "-" in base:
                s, _, e = base.partition("-")
                start, end = int(s), int(e)
            else:
                start = end = int(base)
            if start <= value <= end and (value - start) % step == 0:
                return True
        return False

    @staticmethod
    def cron_matches(expr: str, now: datetime) -> bool:
        """5 字段 cron 匹配（minute hour dom month dow），dow: 0=周日 1=周一…6=周六。"""
        try:
            fields = expr.split()
            if len(fields) != 5:
                return False
            minute, hour, dom, month, dow = fields
            # Python weekday(): 周一=0 … 周日=6 → 转成 cron dow
            dow_value = (now.weekday() + 1) % 7
            return (
                ExpiredResidueCleaner._cron_field_match(minute, now.minute, 0, 59)
                and ExpiredResidueCleaner._cron_field_match(hour, now.hour, 0, 23)
                and ExpiredResidueCleaner._cron_field_match(dom, now.day, 1, 31)
                and ExpiredResidueCleaner._cron_field_match(month, now.month, 1, 12)
                and ExpiredResidueCleaner._cron_field_match(dow, dow_value, 0, 6)
            )
        except Exception:
            return False

    def schedule_triggered(self, now: datetime) -> bool:
        return self.cron_matches(self.cron_expression, now)

    def should_run(self) -> bool:
        """是否允许本次清理执行（含全部前置门槛）。"""
        return self._skip_reason() == ""

    def _should_trigger_auto(self) -> bool:
        """自动模式触发判定：cron 定时或磁盘/内存阈值。"""
        try:
            if self.schedule_triggered(datetime.now()):
                return True
        except Exception:
            pass
        return self.threshold_triggered()

    async def run_loop(self) -> None:
        """自驱动后台调度循环（由 scheduler 以 _register_task 注册，可优雅取消）。

        低 IO/CPU 权重：仅按 check_interval_sec 周期轮询，实际清理在独立线程执行，
        不阻塞事件循环、不抢占策略/行情算力。
        """
        logger.info(
            f"ExpiredResidueCleaner run_loop started "
            f"(interval={self.check_interval_sec:.0f}s, cron='{self.cron_expression}')"
        )
        while True:
            try:
                if self._should_trigger_auto():
                    # 线程池中执行同步清理，规避文件 IO 阻塞事件循环
                    await asyncio.to_thread(self.run, None, "auto")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"ExpiredResidueCleaner auto run error: {e}")
                self._alert("EXPIRED_RESIDUE_CLEANER_LOOP_ERROR", f"调度循环异常: {e}")
            await asyncio.sleep(self.check_interval_sec)

    def _skip_reason(self) -> str:
        if not self.enable_cleaner:
            return "disabled"
        if self.emergency_pause:
            return "emergency_pause"
        if self.is_live_strategy_running():
            return "live_strategy_running"
        return ""

    # ──────────────────────────────────────────────────────────────
    # 时钟漂移校验
    # ──────────────────────────────────────────────────────────────
    def _check_clock_drift(self, now: float) -> bool:
        """校验时钟是否回拨。返回 True 表示可信，False 表示检测到回拨应跳过。"""
        if self._last_run_ts > 0 and now < self._last_run_ts:
            # 时间回拨（如 NTP 校时/手动改时间），规避误删
            logger.warning(
                f"ExpiredResidueCleaner: clock moved backward "
                f"(last={self._last_run_ts:.0f} now={now:.0f}), skip cleanup"
            )
            return False
        return True

    def _is_future_drift(self, mtime: float, now: float) -> bool:
        return mtime > now + self.clock_drift_tolerance_sec

    # ──────────────────────────────────────────────────────────────
    # 主流程
    # ──────────────────────────────────────────────────────────────
    def run(self, dry_run: Optional[bool] = None, trigger_reason: str = "manual") -> CleanupReport:
        """执行一次完整清理。返回 CleanupReport。"""
        now = self._clock()
        report = CleanupReport(run_ts=now, trigger_reason=trigger_reason)

        skip_reason = self._skip_reason()
        if skip_reason:
            report.skipped = True
            report.skipped_reason = skip_reason
            self._write_audit(report)
            return report

        if not self._check_clock_drift(now):
            report.skipped = True
            report.skipped_reason = "clock_drift"
            self._write_audit(report)
            return report

        # 防抖：距上次清理不足 min_interval_sec 时跳过（避免频繁抢占 IO/CPU）
        if (
            self.min_interval_sec > 0
            and self._last_run_ts > 0
            and (now - self._last_run_ts) < self.min_interval_sec
        ):
            report.skipped = True
            report.skipped_reason = "min_interval"
            self._write_audit(report)
            return report

        effective_dry_run = self.dry_run if dry_run is None else dry_run

        try:
            # 1. 扫描文件资源
            file_items = self._scan_file_categories(now)
            # 2. 扫描内存资源
            memory_items = self._scan_memory_categories(now)

            all_items = file_items + memory_items
            report.scanned_total = len(all_items)

            # 3. 分类：受保护 / 漂移 / 过期
            expired: List[ResidueItem] = []
            for item in all_items:
                self._bump(report, item.category, "scanned", 1)
                if item.protected:
                    report.protected_skipped += 1
                    self._bump(report, item.category, "protected", 1)
                    continue
                if item.path and self._is_future_drift(item.mtime, now):
                    report.drift_suspected += 1
                    self._bump(report, item.category, "drift_suspected", 1)
                    continue
                if self._is_expired(item, now):
                    expired.append(item)

            report.expired_count = len(expired)

            # 4. DryRun：只输出清单
            if effective_dry_run:
                self._bump(report, "_dry_run", "candidates", len(expired))
            else:
                file_expired = [it for it in expired if it.path]
                mem_expired = [it for it in expired if not it.path]
                report.archived_count = self._soft_archive(file_expired, now, report)
                self._purge_memory(mem_expired, report)
                # 5. 软保留期到期 → 物理删除
                report.deleted_count, report.freed_bytes = self._purge_expired_archives(now, report)

            self._last_run_ts = now
            self._save_state(now)
        except Exception as e:
            logger.error(f"ExpiredResidueCleaner run error: {e}")
            report.errors.append(str(e))
            self._alert("EXPIRED_RESIDUE_CLEANER_ERROR", f"清理异常: {e}")

        self._write_audit(report)
        return report

    # ──────────────────────────────────────────────────────────────
    # 扫描
    # ──────────────────────────────────────────────────────────────
    def _scan_file_categories(self, now: float) -> List[ResidueItem]:
        items: List[ResidueItem] = []
        for cat in self._categories.values():
            if not cat.enabled or not cat.dirs:
                continue
            for d in cat.dirs:
                scan_dir = self._resolve(d)
                if not os.path.isdir(scan_dir):
                    continue
                try:
                    for name in os.listdir(scan_dir):
                        if not self._match_patterns(name, cat.patterns):
                            continue
                        fpath = os.path.join(scan_dir, name)
                        if not os.path.isfile(fpath):
                            continue
                        try:
                            st = os.stat(fpath)
                        except OSError:
                            continue
                        items.append(ResidueItem(
                            category=cat.category,
                            key=fpath,
                            path=fpath,
                            mtime=st.st_mtime,
                            size_bytes=st.st_size,
                            protected=self._is_protected(fpath, name)
                            or self._is_active_lock(cat.category, fpath),
                        ))
                except OSError as e:
                    logger.debug(f"ExpiredResidueCleaner scan {scan_dir} error: {e}")
        return items

    def _scan_memory_categories(self, now: float) -> List[ResidueItem]:
        items: List[ResidueItem] = []
        for category, (scanner, _purge) in self._memory_scanners.items():
            cat_cfg = self._categories.get(category)
            if cat_cfg and not cat_cfg.enabled:
                continue
            try:
                scanned = scanner() or []
                for it in scanned:
                    it.category = category
                    items.append(it)
            except Exception as e:
                logger.warning(f"ExpiredResidueCleaner memory scanner {category} error: {e}")
        return items

    # ──────────────────────────────────────────────────────────────
    # 判定
    # ──────────────────────────────────────────────────────────────
    def _is_expired(self, item: ResidueItem, now: float) -> bool:
        ttl = self._categories.get(item.category, CategoryConfig(item.category)).ttl_sec
        return (now - item.mtime) > ttl

    def _is_protected(self, path: str, name: str) -> bool:
        # 人工保留标记
        for marker in self.keep_markers:
            if marker in name:
                return True
        # 白名单（订单事件/成交/仓位快照/风控审计/WAL）
        for pat in self.whitelist_patterns:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(path, pat):
                return True
        return False

    # ──────────────────────────────────────────────────────────────
    # 锁文件活性探测（避免误删当前进程持有的单实例锁）
    # ──────────────────────────────────────────────────────────────
    def _is_active_lock(self, category: str, path: str) -> bool:
        """锁文件活性探测：仅 lock_file 类别生效。

        检查锁内 PID 是否存活，存活则保护（跳过清理）。用于规避
        data/app.lock 这类「当前进程仍持有」的单实例锁被误归档。
        """
        if category != ResidueCategory.LOCK_FILE.value:
            return False
        if self._lock_liveness_probe is not None:
            try:
                return bool(self._lock_liveness_probe(path))
            except Exception as e:
                logger.warning(f"ExpiredResidueCleaner lock liveness probe error: {e}")
                return True  # 探测异常保守保护，避免误删活跃锁
        return self._default_lock_liveness(path)

    def _default_lock_liveness(self, path: str) -> bool:
        """默认探测：读取锁文件内 PID，检查进程是否存活。

        真实锁文件由 single_instance.py 写入纯 PID；无法解析为 PID 时
        视为非活跃锁（保持向后兼容，不改变既有文件清理行为）。
        """
        pid = self._read_lock_pid(path)
        if pid is None:
            return False
        return self._pid_alive(pid)

    @staticmethod
    def _read_lock_pid(path: str) -> Optional[int]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if not content:
                return None
            return int(content)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """判定 PID 对应的进程是否存活（跨平台）。"""
        if pid <= 0:
            return False
        if _HAS_PSUTIL:
            try:
                return psutil.pid_exists(pid)
            except Exception:
                return True
        # 无 psutil：平台原生判定
        if sys.platform == "win32":
            try:
                import ctypes
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = ctypes.windll.kernel32.OpenProcess(
                    PROCESS_QUERY_LIMITED_INFORMATION, False, pid
                )
                if not handle:
                    return False
                ctypes.windll.kernel32.CloseHandle(handle)
                return True
            except Exception:
                return True
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
        except Exception:
            return True

    @staticmethod
    def _match_patterns(name: str, patterns: List[str]) -> bool:
        if not patterns or patterns == ["*"]:
            return True
        return any(fnmatch.fnmatch(name, p) for p in patterns)

    # ──────────────────────────────────────────────────────────────
    # 软归档 / 物理删除
    # ──────────────────────────────────────────────────────────────
    def _soft_archive(self, items: List[ResidueItem], now: float, report: CleanupReport) -> int:
        archived = 0
        for item in items:
            if report.freed_bytes >= self.io_budget_bytes:
                break
            src = item.path
            if not src or not os.path.isfile(src):
                continue
            dst_dir = os.path.join(self.archive_dir, item.category)
            try:
                os.makedirs(dst_dir, exist_ok=True)
                base = os.path.basename(src)
                dst = os.path.join(dst_dir, f"{base}.archived_{int(now)}")
                shutil.move(src, dst)
                # 记录归档时间戳，软保留期从归档时刻起算（不受原始 mtime 影响）
                os.utime(dst, (now, now))
                archived += 1
                self._bump(report, item.category, "archived", 1)
            except OSError as e:
                logger.warning(f"ExpiredResidueCleaner archive {src} failed: {e}")
                report.errors.append(f"archive_failed:{src}:{e}")
        return archived

    def _purge_expired_archives(self, now: float, report: CleanupReport):
        """软保留期到期的归档文件 → 物理删除。返回 (删除数, 释放字节数)。"""
        deleted = 0
        freed = 0
        if not os.path.isdir(self.archive_dir):
            return deleted, freed
        cutoff = now - self.soft_retention_sec
        for root, _dirs, files in os.walk(self.archive_dir):
            for name in files:
                fpath = os.path.join(root, name)
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                if st.st_mtime < cutoff and not self._is_protected(fpath, name):
                    try:
                        os.remove(fpath)
                        deleted += 1
                        freed += st.st_size
                        self._bump(report, "_archive", "deleted", 1)
                    except OSError as e:
                        report.errors.append(f"purge_failed:{fpath}:{e}")
        return deleted, freed

    def _purge_memory(self, items: List[ResidueItem], report: CleanupReport) -> None:
        by_cat: Dict[str, List[str]] = {}
        for it in items:
            by_cat.setdefault(it.category, []).append(it.key)
        for category, keys in by_cat.items():
            entry = self._memory_scanners.get(category)
            purge = entry[1] if entry else None
            if purge is None:
                self._bump(report, category, "skipped_no_purge", len(keys))
                continue
            try:
                purge(keys)
                self._bump(report, category, "purged", len(keys))
            except Exception as e:
                logger.warning(f"ExpiredResidueCleaner purge memory {category} failed: {e}")
                report.errors.append(f"purge_memory_failed:{category}:{e}")

    # ──────────────────────────────────────────────────────────────
    # 审计 / 状态持久化
    # ──────────────────────────────────────────────────────────────
    def _bump(self, report: CleanupReport, category: str, key: str, n: int) -> None:
        if category not in report.per_category:
            report.per_category[category] = {}
        report.per_category[category][key] = report.per_category[category].get(key, 0) + n

    def _write_audit(self, report: CleanupReport) -> None:
        record = report.to_dict()
        if self._audit_writer is not None:
            try:
                self._audit_writer(record)
                return
            except Exception as e:
                logger.warning(f"ExpiredResidueCleaner audit writer error: {e}")
        try:
            os.makedirs(os.path.dirname(self.audit_file), exist_ok=True)
            with open(self.audit_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as e:
            logger.warning(f"ExpiredResidueCleaner audit write error: {e}")

    def _alert(self, event: str, message: str) -> None:
        if self._alert_callback is not None:
            try:
                self._alert_callback(event, message)
            except Exception as e:
                logger.warning(f"ExpiredResidueCleaner alert callback error: {e}")

    def _save_state(self, now: float) -> None:
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({"last_run_ts": now}, f)
        except Exception as e:
            logger.debug(f"ExpiredResidueCleaner save state error: {e}")

    def _load_state(self) -> None:
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._last_run_ts = float(data.get("last_run_ts", 0.0))
        except Exception:
            self._last_run_ts = 0.0

    # ──────────────────────────────────────────────────────────────
    # 辅助
    # ──────────────────────────────────────────────────────────────
    def _resolve(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.abspath(os.path.join(self.base_dir, p))
