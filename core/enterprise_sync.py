"""
企业级同步引擎 (Enterprise Sync Engine)
========================================

统一管理所有数据通道的同步健康、版本控制、过期检测与自动恢复。

= 核心组件 =
1. SyncVersionManager — 版本化状态追踪，乐观并发控制
2. SyncHealthMonitor  — 统一同步健康监控（延迟/遗漏/漂移）
3. StaleDataDetector   — 自动过期数据检测与失效
4. RecoveryManager     — 同步间隙检测与自动恢复
5. SyncCoordinator     — 跨组件同步协调（position/order/state/dashboard）

= 设计原则 =
- 版本化：每次状态变更分配单调递增版本号，冲突时以交易所为准
- 多通道仲裁：REST 全量 > WebSocket 增量，但 WS 时间戳更新时以 WS 为准
- 自适应频率：同步健康度下降时自动提高同步频率
- 熔断保护：连续同步失败达阈值暂停自动修复，转为告警
- 间隙补偿：WS 断连后自动检测间隙并触发增量补偿同步

= 用法 =
    sync_engine = EnterpriseSyncEngine(config)
    sync_engine.register_channel("position_rest", position_manager)
    sync_engine.register_channel("position_ws", ws_feed)
    sync_engine.register_channel("order_rest", order_synchronizer)
    sync_engine.start()
"""

import os
import json
import time
import threading
import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class SyncChannel(Enum):
    """同步通道"""
    POSITION_REST = "position_rest"
    POSITION_WS = "position_ws"
    ORDER_REST = "order_rest"
    ORDER_WS = "order_ws"
    ACCOUNT_REST = "account_rest"
    ACCOUNT_WS = "account_ws"
    STATE_PERSIST = "state_persist"
    DASHBOARD_SNAPSHOT = "dashboard_snapshot"


class SyncHealth(Enum):
    """同步健康状态"""
    HEALTHY = "healthy"           # 正常：所有通道及时同步
    DEGRADED = "degraded"         # 降级：部分通道延迟但可接受
    STALE = "stale"               # 过期：数据超过 TTL 未更新
    DISCONNECTED = "disconnected" # 断开：通道完全失联
    RECOVERING = "recovering"     # 恢复中：正在补偿同步


class ConflictResolution(Enum):
    """冲突解决策略"""
    EXCHANGE_WINS = "exchange_wins"       # 以交易所为准
    LATEST_TIMESTAMP = "latest_timestamp" # 以最新时间戳为准
    MERGE_SAFE = "merge_safe"             # 安全合并
    MANUAL = "manual"                     # 人工介入


@dataclass
class SyncVersion:
    """同步版本信息"""
    entity: str                # 数据实体标识 (e.g., "position:ETH-USDT-SWAP:long")
    version: int = 0           # 单调递增版本号
    source: str = ""           # 数据来源 (rest/ws)
    checksum: str = ""         # 数据校验和
    timestamp: float = 0.0     # 更新时间戳
    exchange_timestamp: float = 0.0  # 交易所时间戳
    ttl_seconds: float = 30.0  # 有效期限


@dataclass
class SyncChannelStatus:
    """单个通道的同步状态"""
    channel: SyncChannel
    health: SyncHealth = SyncHealth.HEALTHY
    last_sync_time: float = 0.0
    last_success_time: float = 0.0
    last_error_time: float = 0.0
    last_error: str = ""
    sync_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0
    avg_latency_ms: float = 0.0
    latency_samples: deque = field(default_factory=lambda: deque(maxlen=20))
    data_age_seconds: float = 0.0  # 数据新鲜度


@dataclass
class SyncGap:
    """同步间隙"""
    channel: SyncChannel
    gap_start: float           # 间隙开始时间
    gap_end: float = 0.0       # 间隙结束时间（0=持续中）
    gap_duration_seconds: float = 0.0
    missed_updates: int = 0    # 估计遗漏更新数
    recovered: bool = False


@dataclass
class SyncReport:
    """同步综合报告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    overall_health: SyncHealth = SyncHealth.HEALTHY
    channels: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    version_count: int = 0
    stale_entities: List[str] = field(default_factory=list)
    active_gaps: List[Dict[str, Any]] = field(default_factory=list)
    recovery_actions: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 1. 同步版本管理器
# ═══════════════════════════════════════════════════════════════

class SyncVersionManager:
    """版本化状态追踪，提供乐观并发控制。

    每个数据实体（仓位/订单/账户）有独立版本号，
    更新时检查版本冲突，以交易所数据为最终仲裁。
    """

    def __init__(self, max_versions: int = 5000):
        self._lock = threading.Lock()
        self._versions: Dict[str, SyncVersion] = {}
        self._max_versions = max_versions
        self._version_counter: int = 0
        self._conflicts: deque = deque(maxlen=100)  # 冲突记录

    def get_version(self, entity: str) -> SyncVersion:
        """获取实体当前版本"""
        with self._lock:
            return self._versions.get(entity, SyncVersion(entity=entity))

    def update_version(self, entity: str, source: str, data_checksum: str = "",
                       exchange_ts: float = 0.0) -> SyncVersion:
        """更新实体版本，返回新版本号"""
        with self._lock:
            if entity not in self._versions:
                self._versions[entity] = SyncVersion(entity=entity)

            v = self._versions[entity]
            v.version += 1
            v.source = source
            v.checksum = data_checksum
            v.timestamp = time.time()
            if exchange_ts > 0:
                v.exchange_timestamp = exchange_ts
            v.ttl_seconds = 30.0  # 默认30秒TTL

            # 清理过期版本
            if len(self._versions) > self._max_versions:
                self._prune_stale()

            return v

    def check_conflict(self, entity: str, expected_version: int,
                       new_data: Any, source: str) -> Tuple[bool, ConflictResolution]:
        """检查乐观并发冲突。

        Returns:
            (is_conflict, resolution) — 是否有冲突及建议解决策略
        """
        with self._lock:
            current = self._versions.get(entity)
            if current is None:
                return False, ConflictResolution.EXCHANGE_WINS

            if current.version != expected_version:
                # 版本冲突：以交易所为准
                self._conflicts.append({
                    "entity": entity,
                    "expected": expected_version,
                    "current": current.version,
                    "source": source,
                    "time": time.time(),
                })
                return True, ConflictResolution.EXCHANGE_WINS

            return False, ConflictResolution.EXCHANGE_WINS

    def is_stale(self, entity: str) -> bool:
        """检查实体数据是否过期"""
        with self._lock:
            v = self._versions.get(entity)
            if v is None:
                return True
            return (time.time() - v.timestamp) > v.ttl_seconds

    def get_stale_entities(self) -> List[str]:
        """获取所有过期实体"""
        with self._lock:
            return [e for e, v in self._versions.items()
                    if (time.time() - v.timestamp) > v.ttl_seconds]

    def get_version_map(self) -> Dict[str, int]:
        """获取版本映射（用于快照）"""
        with self._lock:
            return {e: v.version for e, v in self._versions.items()}

    def _prune_stale(self):
        """清理过期版本（超过 5x TTL）"""
        threshold = time.time() - 150  # 5 * 30s
        stale = [e for e, v in self._versions.items() if v.timestamp < threshold]
        for e in stale:
            del self._versions[e]

    def reset(self):
        with self._lock:
            self._versions.clear()
            self._conflicts.clear()


# ═══════════════════════════════════════════════════════════════
# 2. 同步健康监控器
# ═══════════════════════════════════════════════════════════════

class SyncHealthMonitor:
    """统一监控所有同步通道的健康状态。

    追踪指标：
      - 同步延迟（上次同步距今时间）
      - 数据新鲜度（数据距今时间）
      - 错误率 / 连续错误数
      - 通道间漂移（不同通道数据不一致）
    """

    DEFAULTS = {
        "max_data_age_healthy_sec": 10.0,    # 数据新鲜度健康阈值
        "max_data_age_degraded_sec": 30.0,   # 降级阈值
        "max_data_age_stale_sec": 60.0,      # 过期阈值
        "max_consecutive_errors": 5,         # 连续错误熔断
        "max_channel_drift_sec": 5.0,        # 通道间最大漂移
        "latency_warning_ms": 1000,          # 延迟告警阈值
        "latency_critical_ms": 3000,         # 延迟严重阈值
    }

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        for key, default in self.DEFAULTS.items():
            setattr(self, f"_{key}", cfg.get(key, default))

        self._lock = threading.RLock()
        self._channels: Dict[SyncChannel, SyncChannelStatus] = {}
        self._start_time = time.time()
        self._last_health_check = 0.0

    def register_channel(self, channel: SyncChannel) -> SyncChannelStatus:
        """注册同步通道"""
        with self._lock:
            status = SyncChannelStatus(channel=channel)
            self._channels[channel] = status
            return status

    def record_sync(self, channel: SyncChannel, success: bool = True,
                    latency_ms: float = 0.0, data_age_seconds: float = 0.0,
                    error: str = ""):
        """记录一次同步事件"""
        with self._lock:
            if channel not in self._channels:
                self._channels[channel] = SyncChannelStatus(channel=channel)

            s = self._channels[channel]
            s.last_sync_time = time.time()
            s.sync_count += 1
            s.data_age_seconds = data_age_seconds

            if success:
                s.last_success_time = time.time()
                s.consecutive_errors = 0
                s.health = self._evaluate_health(s)
            else:
                s.error_count += 1
                s.consecutive_errors += 1
                s.last_error_time = time.time()
                s.last_error = error
                s.health = self._evaluate_health(s)

            if latency_ms > 0:
                s.latency_samples.append(latency_ms)
                if len(s.latency_samples) > 0:
                    s.avg_latency_ms = sum(s.latency_samples) / len(s.latency_samples)

    def _evaluate_health(self, s: SyncChannelStatus) -> SyncHealth:
        """评估通道健康状态"""
        if s.consecutive_errors >= self._max_consecutive_errors:
            return SyncHealth.DISCONNECTED
        if s.data_age_seconds > self._max_data_age_stale_sec:
            return SyncHealth.STALE
        if s.data_age_seconds > self._max_data_age_degraded_sec:
            return SyncHealth.DEGRADED
        if s.consecutive_errors > 0:
            return SyncHealth.DEGRADED
        return SyncHealth.HEALTHY

    def get_channel_health(self, channel: SyncChannel) -> SyncHealth:
        with self._lock:
            s = self._channels.get(channel)
            return s.health if s else SyncHealth.DISCONNECTED

    def get_overall_health(self) -> SyncHealth:
        """获取整体健康状态（取最差通道）"""
        with self._lock:
            if not self._channels:
                return SyncHealth.DISCONNECTED
            priorities = {
                SyncHealth.DISCONNECTED: 4,
                SyncHealth.STALE: 3,
                SyncHealth.DEGRADED: 2,
                SyncHealth.RECOVERING: 1,
                SyncHealth.HEALTHY: 0,
            }
            worst = SyncHealth.HEALTHY
            worst_pri = 0
            for s in self._channels.values():
                pri = priorities.get(s.health, 0)
                if pri > worst_pri:
                    worst = s.health
                    worst_pri = pri
            return worst

    def get_channel_status(self, channel: SyncChannel) -> Optional[Dict[str, Any]]:
        with self._lock:
            s = self._channels.get(channel)
            if not s:
                return None
            return {
                "channel": s.channel.value,
                "health": s.health.value,
                "last_sync": s.last_sync_time,
                "last_success": s.last_success_time,
                "data_age_sec": round(s.data_age_seconds, 2),
                "avg_latency_ms": round(s.avg_latency_ms, 1),
                "sync_count": s.sync_count,
                "error_count": s.error_count,
                "consecutive_errors": s.consecutive_errors,
                "last_error": s.last_error[-120:] if s.last_error else "",
            }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {ch.value: self.get_channel_status(ch) for ch in self._channels}

    def check_channel_drift(self) -> List[str]:
        """检查通道间数据漂移"""
        alerts = []
        with self._lock:
            rest_ts = {}
            ws_ts = {}
            for ch, s in self._channels.items():
                if "rest" in ch.value:
                    rest_ts[ch.value.replace("_rest", "")] = s.last_success_time
                elif "ws" in ch.value:
                    ws_ts[ch.value.replace("_ws", "")] = s.last_success_time

            for key in set(rest_ts.keys()) & set(ws_ts.keys()):
                # 防御性：任一通道从未成功同步（last_success_time=0.0）时跳过 drift 计算，
                # 否则会得到 ~56 年的假告警（启动期或 WS 断线时常见）
                if rest_ts[key] == 0.0 or ws_ts[key] == 0.0:
                    continue
                drift = abs(rest_ts[key] - ws_ts[key])
                if drift > self._max_channel_drift_sec:
                    alerts.append(f"Channel drift: {key} REST/WS gap={drift:.1f}s")
        return alerts


# ═══════════════════════════════════════════════════════════════
# 3. 过期数据检测器
# ═══════════════════════════════════════════════════════════════

class StaleDataDetector:
    """自动检测并失效过期数据。

    检测维度：
      - 时间过期：数据超过 TTL 未更新
      - 版本过期：版本号落后于最新已知版本
      - 来源过期：数据来源通道已断开
    """

    DEFAULTS = {
        "position_ttl_sec": 15.0,      # 持仓数据 TTL
        "order_ttl_sec": 10.0,         # 订单数据 TTL
        "account_ttl_sec": 30.0,       # 账户数据 TTL
        "dashboard_ttl_sec": 5.0,      # 仪表板数据 TTL
        "check_interval_sec": 3.0,     # 检查间隔
        "auto_invalidate": True,        # 自动失效过期数据
        "stale_alert_threshold": 3,     # 连续过期告警阈值
    }

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        for key, default in self.DEFAULTS.items():
            setattr(self, f"_{key}", cfg.get(key, default))

        self._lock = threading.Lock()
        self._timestamps: Dict[str, float] = {}  # entity → last_update_ts
        self._ttl_map: Dict[str, float] = {}     # entity → ttl
        self._stale_counts: Dict[str, int] = {}  # entity → consecutive_stale_count
        self._invalidated: Set[str] = set()

    def register_entity(self, entity: str, ttl_seconds: float = None,
                        category: str = "position"):
        """注册实体及其 TTL"""
        with self._lock:
            self._timestamps[entity] = time.time()
            if ttl_seconds is not None:
                self._ttl_map[entity] = ttl_seconds
            else:
                # 根据类别推断 TTL
                ttl_map = {
                    "position": self._position_ttl_sec,
                    "order": self._order_ttl_sec,
                    "account": self._account_ttl_sec,
                    "dashboard": self._dashboard_ttl_sec,
                }
                self._ttl_map[entity] = ttl_map.get(category, 30.0)

    def touch(self, entity: str):
        """更新实体时间戳"""
        with self._lock:
            self._timestamps[entity] = time.time()
            self._stale_counts.pop(entity, None)
            self._invalidated.discard(entity)

    def check_stale(self) -> List[str]:
        """检查并返回所有过期实体"""
        stale = []
        now = time.time()
        with self._lock:
            for entity, ts in list(self._timestamps.items()):
                ttl = self._ttl_map.get(entity, 30.0)
                if (now - ts) > ttl:
                    self._stale_counts[entity] = self._stale_counts.get(entity, 0) + 1
                    if self._auto_invalidate:
                        self._invalidated.add(entity)
                    stale.append(entity)
                else:
                    self._stale_counts.pop(entity, None)
        return stale

    def get_alert_entities(self) -> List[str]:
        """获取连续过期超过阈值的实体"""
        with self._lock:
            return [e for e, c in self._stale_counts.items()
                    if c >= self._stale_alert_threshold]

    def is_invalidated(self, entity: str) -> bool:
        with self._lock:
            return entity in self._invalidated

    def clear_invalidated(self):
        with self._lock:
            self._invalidated.clear()
            self._stale_counts.clear()

    def get_data_age(self, entity: str) -> float:
        with self._lock:
            ts = self._timestamps.get(entity, 0)
            return time.time() - ts if ts > 0 else float('inf')


# ═══════════════════════════════════════════════════════════════
# 4. 恢复管理器
# ═══════════════════════════════════════════════════════════════

class RecoveryManager:
    """同步间隙检测与自动恢复。

    恢复策略：
      1. 检测间隙：对比预期同步间隔与实际间隔
      2. 分级恢复：小间隙→增量补偿，大间隙→全量重同步
      3. 熔断保护：连续恢复失败达阈值暂停自动恢复
      4. 优先级：position > order > account > state
    """

    DEFAULTS = {
        "max_gap_small_sec": 30.0,        # 小间隙阈值（增量补偿）
        "max_gap_large_sec": 120.0,       # 大间隙阈值（全量重同步）
        "max_consecutive_failures": 3,    # 连续恢复失败熔断
        "recovery_cooldown_sec": 30.0,    # 恢复冷却时间
        "gap_check_interval_sec": 5.0,    # 间隙检查间隔
        "recovery_priority": [            # 恢复优先级
            "position", "order", "account", "state"
        ],
    }

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config or {}
        for key, default in self.DEFAULTS.items():
            setattr(self, f"_{key}", cfg.get(key, default))

        self._lock = threading.Lock()
        self._gaps: List[SyncGap] = []
        self._recovery_attempts: Dict[str, int] = {}
        self._recovery_failures: Dict[str, int] = {}
        self._last_recovery_time: Dict[str, float] = {}
        self._circuit_open: Set[str] = set()
        self._recovery_callbacks: Dict[str, Callable] = {}

    def register_recovery_callback(self, channel: str, callback: Callable):
        """注册恢复回调（全量重同步函数）"""
        self._recovery_callbacks[channel] = callback

    def detect_gap(self, channel: str, expected_interval: float,
                   last_sync_time: float) -> Optional[SyncGap]:
        """检测同步间隙"""
        now = time.time()
        gap_duration = now - last_sync_time

        if gap_duration > expected_interval * 1.5:
            with self._lock:
                gap = SyncGap(
                    channel=SyncChannel(channel) if channel in [c.value for c in SyncChannel] else None,
                    gap_start=last_sync_time,
                    gap_end=now,
                    gap_duration_seconds=gap_duration,
                    missed_updates=int(gap_duration / expected_interval),
                )
                self._gaps.append(gap)
                if len(self._gaps) > 50:
                    self._gaps = self._gaps[-50:]
                return gap
        return None

    def get_recovery_action(self, channel: str, gap: SyncGap) -> str:
        """获取恢复动作"""
        if channel in self._circuit_open:
            return "skip_circuit_open"

        if gap.gap_duration_seconds < self._max_gap_small_sec:
            return "incremental_sync"
        elif gap.gap_duration_seconds < self._max_gap_large_sec:
            return "full_sync_channel"
        else:
            return "full_sync_all"

    def attempt_recovery(self, channel: str) -> bool:
        """尝试恢复同步"""
        with self._lock:
            if channel in self._circuit_open:
                return False

            now = time.time()
            last = self._last_recovery_time.get(channel, 0)
            if now - last < self._recovery_cooldown_sec:
                return False

            self._recovery_attempts[channel] = self._recovery_attempts.get(channel, 0) + 1
            self._last_recovery_time[channel] = now

            # 执行恢复回调
            callback = self._recovery_callbacks.get(channel)
            if callback:
                try:
                    callback()
                    self._recovery_failures[channel] = 0
                    # 标记相应间隙为已恢复
                    for g in self._gaps:
                        if hasattr(g, 'channel') and g.channel and g.channel.value == channel:
                            g.recovered = True
                    return True
                except Exception as e:
                    self._recovery_failures[channel] = self._recovery_failures.get(channel, 0) + 1
                    if self._recovery_failures[channel] >= self._max_consecutive_failures:
                        self._circuit_open.add(channel)
                        logger.error(
                            f"Recovery circuit open for {channel}: "
                            f"{self._recovery_failures[channel]} consecutive failures"
                        )
                    return False

            return False

    def reset_circuit(self, channel: str):
        """重置熔断"""
        with self._lock:
            self._circuit_open.discard(channel)
            self._recovery_failures[channel] = 0

    def get_active_gaps(self) -> List[Dict[str, Any]]:
        """获取活跃间隙"""
        with self._lock:
            return [
                {
                    "channel": g.channel.value if g.channel else "unknown",
                    "start": g.gap_start,
                    "duration_sec": round(g.gap_duration_seconds, 1),
                    "missed_updates": g.missed_updates,
                    "recovered": g.recovered,
                }
                for g in self._gaps[-20:]
                if not g.recovered
            ]


# ═══════════════════════════════════════════════════════════════
# 5. 同步协调器（主引擎）
# ═══════════════════════════════════════════════════════════════

class EnterpriseSyncEngine:
    """企业级同步引擎 — 统一协调所有同步通道。

    整合 VersionManager、HealthMonitor、StaleDetector、RecoveryManager，
    提供一站式同步健康管理。
    """

    DEFAULTS = {
        "sync_check_interval_sec": 3.0,
        "report_interval_sec": 60.0,
        "persist_path": "data/enterprise_sync_state.json",
        "persist_interval_sec": 300,
        "auto_recovery_enabled": True,
        "alert_on_stale": True,
        "alert_on_drift": True,
        "alert_on_disconnect": True,
    }

    def __init__(self, config: Dict[str, Any] = None):
        cfg = (config or {}).get("enterprise_sync", {})
        for key, default in self.DEFAULTS.items():
            setattr(self, f"_{key}", cfg.get(key, default))

        # 子组件
        self.version_manager = SyncVersionManager()
        self.health_monitor = SyncHealthMonitor(cfg.get("health", {}))
        self.stale_detector = StaleDataDetector(cfg.get("stale", {}))
        self.recovery_manager = RecoveryManager(cfg.get("recovery", {}))

        # 运行时状态
        self._lock = threading.Lock()
        self._running = False
        self._channels: Dict[str, Any] = {}  # channel_name → handler
        self._last_report_time = 0.0
        self._last_save_time = 0.0
        self._reports: deque = deque(maxlen=50)
        self._stats = {
            "total_syncs": 0,
            "total_gaps_detected": 0,
            "total_recoveries": 0,
            "total_stale_invalidations": 0,
            "start_time": time.time(),
        }

        self._load_state()
        logger.info(
            f"EnterpriseSyncEngine initialized: "
            f"auto_recovery={self._auto_recovery_enabled}, "
            f"check_interval={self._sync_check_interval_sec}s"
        )

    # ═══════════════════════════════════════════════════════════════
    # 通道注册
    # ═══════════════════════════════════════════════════════════════

    def register_channel(self, name: str, handler: Any,
                         expected_interval: float = 5.0,
                         recovery_callback: Callable = None):
        """注册同步通道。

        Args:
            name: 通道名称 (e.g., 'position_rest', 'order_ws')
            handler: 通道处理器对象
            expected_interval: 预期同步间隔（用于间隙检测）
            recovery_callback: 恢复回调（全量重同步）
        """
        with self._lock:
            self._channels[name] = {
                "handler": handler,
                "expected_interval": expected_interval,
                "last_sync": time.time(),
            }
            try:
                channel_enum = SyncChannel(name)
            except ValueError:
                pass  # 自定义通道
            else:
                self.health_monitor.register_channel(channel_enum)

            if recovery_callback:
                self.recovery_manager.register_recovery_callback(name, recovery_callback)

            logger.info(f"EnterpriseSync: channel '{name}' registered (interval={expected_interval}s)")

    # ═══════════════════════════════════════════════════════════════
    # 同步事件记录
    # ═══════════════════════════════════════════════════════════════

    def record_sync(self, channel: str, success: bool = True,
                    latency_ms: float = 0.0, data_age_seconds: float = 0.0,
                    entities: List[str] = None, error: str = ""):
        """记录一次同步事件"""
        with self._lock:
            if channel in self._channels:
                self._channels[channel]["last_sync"] = time.time()

            self._stats["total_syncs"] += 1

        try:
            channel_enum = SyncChannel(channel)
        except ValueError:
            return

        self.health_monitor.record_sync(
            channel_enum, success=success, latency_ms=latency_ms,
            data_age_seconds=data_age_seconds, error=error
        )

        # 更新实体时间戳
        if entities and success:
            for entity in entities:
                self.stale_detector.touch(entity)

    def update_entity_version(self, entity: str, source: str,
                              data_checksum: str = "", exchange_ts: float = 0.0):
        """更新实体版本号"""
        v = self.version_manager.update_version(
            entity, source, data_checksum, exchange_ts
        )
        self.stale_detector.register_entity(entity)
        self.stale_detector.touch(entity)
        return v

    # ═══════════════════════════════════════════════════════════════
    # 主循环
    # ═══════════════════════════════════════════════════════════════

    def start(self):
        """启动同步引擎（在后台线程中运行）"""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="EnterpriseSync")
        self._thread.start()
        logger.info("EnterpriseSyncEngine started")

    def stop(self):
        """停止同步引擎"""
        self._running = False
        self._save_state()
        logger.info("EnterpriseSyncEngine stopped")

    def _run_loop(self):
        """主循环：周期性检查同步健康、检测间隙、触发恢复"""
        while self._running:
            try:
                self._check_cycle()
            except Exception as e:
                logger.error(f"EnterpriseSync cycle error: {e}")
            time.sleep(self._sync_check_interval_sec)

    def _check_cycle(self):
        """单次检查周期"""
        now = time.time()

        # 1. 检测各通道间隙
        self._detect_gaps()

        # 2. 检查过期数据
        stale = self.stale_detector.check_stale()
        if stale:
            self._stats["total_stale_invalidations"] += len(stale)
            if self._alert_on_stale:
                alert_entities = self.stale_detector.get_alert_entities()
                if alert_entities:
                    logger.warning(
                        f"EnterpriseSync: {len(alert_entities)} entities stale for >={self.stale_detector._stale_alert_threshold} cycles: "
                        f"{alert_entities[:5]}{'...' if len(alert_entities) > 5 else ''}"
                    )

        # 3. 检查通道漂移
        if self._alert_on_drift:
            drift_alerts = self.health_monitor.check_channel_drift()
            for alert in drift_alerts:
                logger.warning(f"EnterpriseSync drift: {alert}")

        # 4. 自动恢复
        if self._auto_recovery_enabled:
            self._auto_recover()

        # 5. 定期生成报告
        if now - self._last_report_time > self._report_interval_sec:
            self._last_report_time = now
            report = self.generate_report()
            self._reports.append(report)

        # 6. 持久化
        if now - self._last_save_time > self._persist_interval_sec:
            self._save_state()
            self._last_save_time = now

    def _detect_gaps(self):
        """检测所有通道的同步间隙"""
        with self._lock:
            for name, info in list(self._channels.items()):
                gap = self.recovery_manager.detect_gap(
                    name, info["expected_interval"], info["last_sync"]
                )
                if gap:
                    self._stats["total_gaps_detected"] += 1
                    logger.debug(
                        f"EnterpriseSync gap: {name} {gap.gap_duration_seconds:.1f}s "
                        f"(~{gap.missed_updates} missed)"
                    )

    def _auto_recover(self):
        """自动恢复：按优先级处理间隙"""
        active_gaps = self.recovery_manager.get_active_gaps()
        if not active_gaps:
            return

        # 按优先级排序
        priority_order = {p: i for i, p in enumerate(self.recovery_manager._recovery_priority)}
        active_gaps.sort(key=lambda g: (
            priority_order.get(g.get("channel", ""), 99),
            -g.get("duration_sec", 0)
        ))

        for gap in active_gaps[:3]:  # 每次最多处理3个
            channel = gap.get("channel", "")
            if channel in self._channels:
                success = self.recovery_manager.attempt_recovery(channel)
                if success:
                    self._stats["total_recoveries"] += 1
                    logger.info(f"EnterpriseSync recovery: {channel} recovered ({gap['duration_sec']:.1f}s gap)")

    # ═══════════════════════════════════════════════════════════════
    # 报告与查询
    # ═══════════════════════════════════════════════════════════════

    def generate_report(self) -> SyncReport:
        """生成同步综合报告"""
        report = SyncReport()
        report.overall_health = self.health_monitor.get_overall_health()
        report.channels = self.health_monitor.get_all_status()
        report.version_count = len(self.version_manager.get_version_map())
        report.stale_entities = self.stale_detector.get_alert_entities()
        report.active_gaps = self.recovery_manager.get_active_gaps()

        # 生成告警
        if report.overall_health == SyncHealth.DISCONNECTED:
            report.alerts.append("SYNC_DISCONNECTED: One or more channels fully disconnected")
        if report.overall_health == SyncHealth.STALE:
            report.alerts.append("SYNC_STALE: Data significantly out of date")
        if report.stale_entities:
            report.alerts.append(f"STALE_ENTITIES: {len(report.stale_entities)} entities stale")
        if report.active_gaps:
            report.alerts.append(f"ACTIVE_GAPS: {len(report.active_gaps)} sync gaps detected")

        return report

    def get_last_report(self) -> Optional[SyncReport]:
        return self._reports[-1] if self._reports else None

    def get_stats(self) -> Dict[str, Any]:
        return dict(self._stats)

    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要（供 Dashboard 使用）"""
        report = self.generate_report()
        return {
            "overall_health": report.overall_health.value,
            "channel_count": len(report.channels),
            "healthy_channels": sum(
                1 for c in report.channels.values()
                if c and c.get("health") == "healthy"
            ),
            "degraded_channels": sum(
                1 for c in report.channels.values()
                if c and c.get("health") in ("degraded", "stale")
            ),
            "disconnected_channels": sum(
                1 for c in report.channels.values()
                if c and c.get("health") == "disconnected"
            ),
            "stale_entities": len(report.stale_entities),
            "active_gaps": len(report.active_gaps),
            "version_count": report.version_count,
            "total_syncs": self._stats["total_syncs"],
            "total_recoveries": self._stats["total_recoveries"],
            "timestamp": datetime.now().isoformat(),
        }

    # ═══════════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════════

    def _save_state(self):
        try:
            state = {
                "version_map": self.version_manager.get_version_map(),
                "channels": {
                    name: {"last_sync": info["last_sync"]}
                    for name, info in self._channels.items()
                },
                "stats": self._stats,
                "saved_at": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"EnterpriseSync save failed: {e}")

    def _load_state(self):
        try:
            if not os.path.exists(self._persist_path):
                return
            with open(self._persist_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            for name, info in state.get("channels", {}).items():
                if name in self._channels:
                    self._channels[name]["last_sync"] = info.get("last_sync", 0)
            logger.info(f"EnterpriseSync state loaded: {len(state.get('version_map', {}))} versions")
        except Exception as e:
            logger.debug(f"EnterpriseSync load failed: {e}")


# ═══════════════════════════════════════════════════════════════
# 单例工厂
# ═══════════════════════════════════════════════════════════════

_sync_engine: Optional[EnterpriseSyncEngine] = None


def get_sync_engine(config: Dict[str, Any] = None) -> EnterpriseSyncEngine:
    """获取全局同步引擎单例"""
    global _sync_engine
    if _sync_engine is None:
        _sync_engine = EnterpriseSyncEngine(config)
    return _sync_engine


def reset_sync_engine():
    """重置单例（测试用）"""
    global _sync_engine
    _sync_engine = None