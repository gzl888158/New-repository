"""
算力动态调度架构
================
核心定位：高波动时段自动提升指标计算优先级，震荡低波动时段降低算力占用，
适配本地设备持续7×24小时运行。

三层闭环：
1. regime → 指标优先级（数据驱动）：高波动symbol每次必算，低波动symbol跳过返回缓存
2. regime → 策略容器调度频率：高波动50ms，正常100ms，低波动500ms
3. CPU/内存 → 自适应降频（过载保护）：CPU超阈值时全局降频，暂停非实时循环
"""

import json
import os
import time
import threading
from typing import Dict, Any, Optional, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class ComputePriority(Enum):
    """算力优先级"""
    HIGH = "high"        # 高波动：每次必算
    NORMAL = "normal"    # 正常波动：按默认频率算
    LOW = "low"          # 低波动：降频，跳过部分计算
    THROTTLED = "throttled"  # CPU过载降频：最低算力


@dataclass
class SymbolComputeState:
    """per-symbol 算力状态"""
    symbol: str
    priority: ComputePriority = ComputePriority.NORMAL
    volatility_percentile: float = 0.5
    last_compute_ts: float = 0.0
    compute_interval_ms: float = 100.0  # 动态调整的计算间隔
    skip_count: int = 0  # 被跳过的次数
    total_count: int = 0  # 总调度次数


class ComputeScheduler:
    """
    算力动态调度器

    职责：
    1. 接收 MarketRegimeEngine 的 regime 更新，动态调整 per-symbol 计算优先级
    2. 接收 CPU/内存监控数据，CPU过载时全局降频
    3. 提供 should_compute(symbol) 接口供 IndicatorEngine 调用（节流判断）
    4. 提供 get_compute_interval_ms(symbol) 接口供 StrategyContainer 调用（频率调整）
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        hardware = self.config.get("hardware", {})
        risk = self.config.get("risk", {})

        self._max_cpu_usage = hardware.get("max_cpu_usage", 80)
        self._max_memory_usage = hardware.get("max_memory_usage", 85)
        self._cpu_throttle_threshold = self._max_cpu_usage * 0.9  # 90% of max = 开始降频
        self._cpu_critical_threshold = self._max_cpu_usage  # 达到max = 激进降频

        # per-symbol 算力状态
        self._symbol_states: Dict[str, SymbolComputeState] = {}
        self._lock = threading.RLock()

        # 全局CPU/内存状态
        self._current_cpu = 0.0
        self._current_memory = 0.0
        self._global_throttle_active = False
        self._throttle_start_time: Optional[float] = None

        # 优先级 → 计算间隔映射（毫秒）
        self._priority_intervals = {
            ComputePriority.HIGH: 50,       # 高波动：50ms（提升2倍）
            ComputePriority.NORMAL: 100,    # 正常：100ms
            ComputePriority.LOW: 500,       # 低波动：500ms（降低5倍）
            ComputePriority.THROTTLED: 1000,  # 过载降频：1000ms（最低算力）
        }

        # 波动率阈值（来自 IndicatorEngine 的 volatility_percentile）
        self._high_vol_threshold = risk.get("high_volatility_percentile", 0.8)
        self._low_vol_threshold = risk.get("low_volatility_percentile", 0.2)

        # 低波动跳过比例：低波动symbol每N次调度只算1次
        self._low_vol_skip_ratio = 3  # 1/3 频率

        # ========== 生产级特性 ==========

        # CPU history for adaptive throttling (trend tracking)
        self._cpu_history: deque = deque(maxlen=10)

        # Throttle statistics
        self._throttle_activation_count: int = 0
        self._throttle_duration_total: float = 0.0

        # Compute budget tracking (max computes per second per priority)
        self._compute_budget: Dict[str, int] = {
            ComputePriority.HIGH.value: 20,
            ComputePriority.NORMAL.value: 10,
            ComputePriority.LOW.value: 3,
            ComputePriority.THROTTLED.value: 1,
        }
        self._budget_window_start: float = time.time()
        self._budget_counts: Dict[str, int] = {
            ComputePriority.HIGH.value: 0,
            ComputePriority.NORMAL.value: 0,
            ComputePriority.LOW.value: 0,
            ComputePriority.THROTTLED.value: 0,
        }

        # State persistence file path
        self._state_file: str = self.config.get(
            "state_file",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "compute_scheduler_state.json")
        )

        logger.info(
            f"ComputeScheduler initialized: "
            f"cpu_throttle={self._cpu_throttle_threshold:.0f}%, "
            f"high_vol>{self._high_vol_threshold}, low_vol<{self._low_vol_threshold}"
        )

    def register_symbol(self, symbol: str) -> None:
        """注册需要算力调度的symbol"""
        with self._lock:
            if symbol not in self._symbol_states:
                self._symbol_states[symbol] = SymbolComputeState(symbol=symbol)

    def update_symbol_volatility(self, symbol: str, volatility_percentile: float) -> None:
        """
        更新symbol波动率分位数（由 IndicatorEngine 或 MarketRegimeEngine 调用）

        Args:
            symbol: 交易对
            volatility_percentile: 波动率分位数 [0, 1]
        """
        with self._lock:
            state = self._symbol_states.get(symbol)
            if state is None:
                state = SymbolComputeState(symbol=symbol)
                self._symbol_states[symbol] = state

            state.volatility_percentile = volatility_percentile
            self._update_symbol_priority(state)

    def _update_symbol_priority(self, state: SymbolComputeState) -> None:
        """根据波动率和CPU负载更新symbol优先级"""
        if self._global_throttle_active:
            # CPU过载：所有symbol降为最低优先级
            state.priority = ComputePriority.THROTTLED
            state.compute_interval_ms = self._priority_intervals[ComputePriority.THROTTLED]
        elif state.volatility_percentile >= self._high_vol_threshold:
            # 高波动：提升优先级
            state.priority = ComputePriority.HIGH
            state.compute_interval_ms = self._priority_intervals[ComputePriority.HIGH]
        elif state.volatility_percentile <= self._low_vol_threshold:
            # 低波动：降低优先级
            state.priority = ComputePriority.LOW
            state.compute_interval_ms = self._priority_intervals[ComputePriority.LOW]
        else:
            # 正常波动
            state.priority = ComputePriority.NORMAL
            state.compute_interval_ms = self._priority_intervals[ComputePriority.NORMAL]

    def update_cpu_memory(self, cpu_percent: float, memory_percent: float) -> None:
        """
        更新CPU/内存使用率（由 PerformanceMonitor 或 APM 调用）

        CPU过载时触发全局降频：
        - cpu > critical_threshold: 所有symbol降为THROTTLED
        - cpu > throttle_threshold: 低波动symbol额外降频
        """
        with self._lock:
            self._current_cpu = cpu_percent
            self._current_memory = memory_percent

            # 记录CPU历史用于趋势分析
            self._cpu_history.append(cpu_percent)

            if cpu_percent >= self._cpu_critical_threshold:
                # CPU达到临界值：全局降频
                if not self._global_throttle_active:
                    self._global_throttle_active = True
                    self._throttle_start_time = time.time()
                    self._throttle_activation_count += 1
                    logger.warning(
                        f"⛔ CPU overload {cpu_percent:.1f}% >= {self._cpu_critical_threshold:.0f}%, "
                        f"global compute throttle activated (activation #{self._throttle_activation_count})"
                    )
                # 更新所有symbol优先级为THROTTLED
                for state in self._symbol_states.values():
                    state.priority = ComputePriority.THROTTLED
                    state.compute_interval_ms = self._priority_intervals[ComputePriority.THROTTLED]

            elif cpu_percent < self._cpu_throttle_threshold * 0.85:
                # CPU恢复到安全水平（85%的throttle阈值以下）：解除降频
                if self._global_throttle_active:
                    self._global_throttle_active = False
                    elapsed = time.time() - self._throttle_start_time if self._throttle_start_time else 0
                    self._throttle_duration_total += elapsed
                    logger.info(
                        f"CPU recovered to {cpu_percent:.1f}%, global throttle released "
                        f"(was active for {elapsed:.0f}s, total throttle duration: {self._throttle_duration_total:.0f}s)"
                    )
                    self._throttle_start_time = None
                    # 恢复所有symbol的波动率优先级
                    for state in self._symbol_states.values():
                        self._update_symbol_priority(state)

            # 自适应降频：根据CPU趋势预判性调整跳过比例
            self._adaptive_throttle_adjust()

    def should_compute(self, symbol: str) -> bool:
        """
        判断该symbol是否应该执行本次计算（节流判断）

        供 IndicatorEngine.calculate_all(symbol) 调用：
        - HIGH优先级：每次都算
        - NORMAL优先级：每次都算
        - LOW优先级：每 _low_vol_skip_ratio 次算1次（跳过2/3）
        - THROTTLED优先级：每 _low_vol_skip_ratio * 2 次算1次（跳过5/6）

        Returns:
            True: 执行计算 / False: 跳过本次（使用缓存）
        """
        with self._lock:
            state = self._symbol_states.get(symbol)
            if state is None:
                return True  # 未注册的symbol默认每次都算

            state.total_count += 1

            if state.priority == ComputePriority.HIGH:
                state.last_compute_ts = time.time()
                return True
            elif state.priority == ComputePriority.NORMAL:
                state.last_compute_ts = time.time()
                return True
            elif state.priority == ComputePriority.LOW:
                # 每3次算1次
                if state.total_count % self._low_vol_skip_ratio == 0:
                    state.last_compute_ts = time.time()
                    return True
                state.skip_count += 1
                return False
            elif state.priority == ComputePriority.THROTTLED:
                # 每6次算1次
                if state.total_count % (self._low_vol_skip_ratio * 2) == 0:
                    state.last_compute_ts = time.time()
                    return True
                state.skip_count += 1
                return False
            state.last_compute_ts = time.time()
            return True

    def get_compute_interval_ms(self, symbol: str) -> float:
        """
        获取该symbol的当前计算间隔（毫秒）

        供 StrategyContainer.process_tick 调用：控制策略处理频率
        """
        with self._lock:
            state = self._symbol_states.get(symbol)
            if state is None:
                return self._priority_intervals[ComputePriority.NORMAL]
            return state.compute_interval_ms

    def is_global_throttled(self) -> bool:
        """是否处于全局降频状态"""
        with self._lock:
            return self._global_throttle_active

    def get_high_priority_symbols(self) -> List[str]:
        """获取当前高优先级（高波动）的symbol列表"""
        with self._lock:
            return [s for s, state in self._symbol_states.items()
                    if state.priority == ComputePriority.HIGH]

    def get_status(self) -> Dict[str, Any]:
        """获取算力调度状态摘要"""
        with self._lock:
            priority_counts = defaultdict(int)
            total_skips = 0
            total_computes = 0
            for state in self._symbol_states.values():
                priority_counts[state.priority.value] += 1
                total_skips += state.skip_count
                total_computes += state.total_count

            return {
                "cpu_percent": round(self._current_cpu, 1),
                "memory_percent": round(self._current_memory, 1),
                "global_throttle_active": self._global_throttle_active,
                "throttle_elapsed_seconds": round(
                    time.time() - self._throttle_start_time, 0
                ) if self._throttle_start_time else 0,
                "priority_distribution": dict(priority_counts),
                "total_symbols": len(self._symbol_states),
                "total_skips": total_skips,
                "total_computes": total_computes,
                "high_priority_symbols": self.get_high_priority_symbols(),
            }

    # ========================================================================
    #  生产级特性 1: State Persistence（状态持久化）
    # ========================================================================

    def collect_persistent_state(self) -> Dict[str, Any]:
        """
        收集当前调度器的完整状态，用于持久化保存。

        Returns:
            包含所有可恢复状态的字典
        """
        with self._lock:
            symbol_states = {}
            for s, state in self._symbol_states.items():
                symbol_states[s] = {
                    "priority": state.priority.value,
                    "volatility_percentile": state.volatility_percentile,
                    "compute_interval_ms": state.compute_interval_ms,
                    "skip_count": state.skip_count,
                    "total_count": state.total_count,
                    "last_compute_ts": state.last_compute_ts,
                }

            return {
                "version": 1,
                "timestamp": time.time(),
                "global_throttle_active": self._global_throttle_active,
                "throttle_start_time": self._throttle_start_time,
                "throttle_activation_count": self._throttle_activation_count,
                "throttle_duration_total": self._throttle_duration_total,
                "symbol_states": symbol_states,
                "low_vol_skip_ratio": self._low_vol_skip_ratio,
            }

    def save_state(self) -> bool:
        """保存状态到 JSON 文件"""
        try:
            state = self.collect_persistent_state()
            os.makedirs(os.path.dirname(self._state_file), exist_ok=True)
            with open(self._state_file, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            logger.info(f"ComputeScheduler state saved to {self._state_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save ComputeScheduler state: {e}")
            return False

    def restore_persistent_state(self) -> bool:
        """
        从 JSON 文件恢复调度器状态。

        Returns:
            True 如果恢复成功，False 如果文件不存在或格式错误
        """
        if not os.path.exists(self._state_file):
            logger.info(f"No state file found at {self._state_file}, skipping restore")
            return False

        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            with self._lock:
                self._global_throttle_active = data.get("global_throttle_active", False)
                self._throttle_start_time = data.get("throttle_start_time")
                self._throttle_activation_count = data.get("throttle_activation_count", 0)
                self._throttle_duration_total = data.get("throttle_duration_total", 0.0)
                self._low_vol_skip_ratio = data.get("low_vol_skip_ratio", 3)

                symbol_states_data = data.get("symbol_states", {})
                for s, sdata in symbol_states_data.items():
                    state = self._symbol_states.get(s)
                    if state is None:
                        state = SymbolComputeState(symbol=s)
                        self._symbol_states[s] = state
                    state.volatility_percentile = sdata.get("volatility_percentile", 0.5)
                    state.compute_interval_ms = sdata.get("compute_interval_ms", 100.0)
                    state.skip_count = sdata.get("skip_count", 0)
                    state.total_count = sdata.get("total_count", 0)
                    state.last_compute_ts = sdata.get("last_compute_ts", 0.0)
                    # 恢复优先级
                    priority_str = sdata.get("priority", "normal")
                    try:
                        state.priority = ComputePriority(priority_str)
                    except ValueError:
                        state.priority = ComputePriority.NORMAL

            logger.info(
                f"ComputeScheduler state restored from {self._state_file}: "
                f"{len(symbol_states_data)} symbols, "
                f"throttle_active={self._global_throttle_active}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to restore ComputeScheduler state: {e}")
            return False

    # ========================================================================
    #  生产级特性 2: Hot Update Support（配置热更新）
    # ========================================================================

    def update_config(self, config: Dict[str, Any]) -> None:
        """
        热更新配置参数，无需重启调度器。

        支持更新的参数：
        - hardware.max_cpu_usage → _max_cpu_usage, _cpu_throttle_threshold, _cpu_critical_threshold
        - risk.high_volatility_percentile → _high_vol_threshold
        - risk.low_volatility_percentile → _low_vol_threshold
        - low_vol_skip_ratio → _low_vol_skip_ratio
        - priority_intervals → _priority_intervals
        - compute_budget → _compute_budget

        Args:
            config: 包含要更新参数的配置字典
        """
        hardware = config.get("hardware", {})
        risk = config.get("risk", {})
        changes = []

        with self._lock:
            # 更新 CPU 相关阈值
            if "max_cpu_usage" in hardware:
                old_val = self._max_cpu_usage
                self._max_cpu_usage = hardware["max_cpu_usage"]
                self._cpu_throttle_threshold = self._max_cpu_usage * 0.9
                self._cpu_critical_threshold = self._max_cpu_usage
                changes.append(
                    f"max_cpu_usage: {old_val} → {self._max_cpu_usage} "
                    f"(throttle={self._cpu_throttle_threshold:.0f}%, critical={self._cpu_critical_threshold:.0f}%)"
                )

            # 更新波动率阈值
            if "high_volatility_percentile" in risk:
                old_val = self._high_vol_threshold
                self._high_vol_threshold = risk["high_volatility_percentile"]
                changes.append(f"high_vol_threshold: {old_val} → {self._high_vol_threshold}")

            if "low_volatility_percentile" in risk:
                old_val = self._low_vol_threshold
                self._low_vol_threshold = risk["low_volatility_percentile"]
                changes.append(f"low_vol_threshold: {old_val} → {self._low_vol_threshold}")

            # 更新跳过比例
            if "low_vol_skip_ratio" in config:
                old_val = self._low_vol_skip_ratio
                self._low_vol_skip_ratio = config["low_vol_skip_ratio"]
                changes.append(f"low_vol_skip_ratio: {old_val} → {self._low_vol_skip_ratio}")

            # 更新优先级计算间隔
            if "priority_intervals" in config:
                new_intervals = config["priority_intervals"]
                for pri_key, interval_ms in new_intervals.items():
                    try:
                        pri = ComputePriority(pri_key)
                        old_val = self._priority_intervals[pri]
                        self._priority_intervals[pri] = interval_ms
                        changes.append(f"priority_intervals[{pri_key}]: {old_val}ms → {interval_ms}ms")
                    except ValueError:
                        logger.warning(f"Invalid priority key in update_config: {pri_key}")

            # 更新计算预算
            if "compute_budget" in config:
                new_budget = config["compute_budget"]
                for pri_key, budget in new_budget.items():
                    if pri_key in self._compute_budget:
                        old_val = self._compute_budget[pri_key]
                        self._compute_budget[pri_key] = budget
                        changes.append(f"compute_budget[{pri_key}]: {old_val} → {budget}/s")
                    else:
                        logger.warning(f"Invalid budget key in update_config: {pri_key}")

            # 更新状态文件路径
            if "state_file" in config:
                self._state_file = config["state_file"]
                changes.append(f"state_file: {self._state_file}")

            # 重新评估所有 symbol 优先级
            for state in self._symbol_states.values():
                self._update_symbol_priority(state)

        if changes:
            logger.info(f"ComputeScheduler config updated ({len(changes)} changes):")
            for change in changes:
                logger.info(f"  - {change}")
        else:
            logger.info("ComputeScheduler config update: no changes detected")

    # ========================================================================
    #  生产级特性 3: Enhanced Metrics（增强指标）
    # ========================================================================

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """
        获取增强的算力调度统计指标。

        Returns:
            包含详细统计信息的字典：
            - compute_savings: 总节省计算次数与总计算次数的对比
            - skip_rate_by_priority: 每个优先级级别的跳过率
            - cpu_time_saved_estimate: 估算节省的CPU时间
            - throttle_duration_total: 总降频持续时间
            - throttle_activation_count: 降频激活次数
            - per_symbol_stats: 每个 symbol 的详细统计
            - avg_compute_interval_by_priority: 每个优先级的平均计算间隔
        """
        with self._lock:
            total_skips = 0
            total_computes = 0
            priority_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"skips": 0, "computes": 0, "count": 0})
            per_symbol_stats = []

            for state in self._symbol_states.values():
                total_skips += state.skip_count
                total_computes += state.total_count
                pri = state.priority.value
                priority_stats[pri]["skips"] += state.skip_count
                priority_stats[pri]["computes"] += state.total_count
                priority_stats[pri]["count"] += 1

                actual_computes = state.total_count - state.skip_count
                skip_rate = state.skip_count / state.total_count if state.total_count > 0 else 0.0
                per_symbol_stats.append({
                    "symbol": state.symbol,
                    "priority": state.priority.value,
                    "volatility_percentile": round(state.volatility_percentile, 3),
                    "skips": state.skip_count,
                    "computes": state.total_count,
                    "actual_computes": actual_computes,
                    "skip_rate": round(skip_rate, 3),
                    "compute_interval_ms": state.compute_interval_ms,
                    "last_compute_ts": state.last_compute_ts,
                    "last_compute_ago_seconds": round(
                        time.time() - state.last_compute_ts, 1
                    ) if state.last_compute_ts > 0 else None,
                })

            # 计算各优先级的跳过率
            skip_rate_by_priority = {}
            for pri, stats in priority_stats.items():
                if stats["computes"] > 0:
                    skip_rate_by_priority[pri] = round(stats["skips"] / stats["computes"], 3)
                else:
                    skip_rate_by_priority[pri] = 0.0

            # 计算各优先级的平均计算间隔
            avg_interval_by_priority = {}
            for pri, stats in priority_stats.items():
                if stats["count"] > 0:
                    avg_interval_by_priority[pri] = round(
                        sum(
                            s.compute_interval_ms
                            for s in self._symbol_states.values()
                            if s.priority.value == pri
                        ) / stats["count"], 1
                    )
                else:
                    avg_interval_by_priority[pri] = 0.0

            # 估算节省的 CPU 时间（假设每次计算平均耗时 5ms）
            avg_compute_time_ms = 5.0
            cpu_time_saved_seconds = round(total_skips * avg_compute_time_ms / 1000.0, 2)

            return {
                "compute_savings": {
                    "total_skips": total_skips,
                    "total_computes_attempted": total_computes,
                    "total_actual_computes": total_computes - total_skips,
                    "overall_skip_rate": round(
                        total_skips / total_computes, 3
                    ) if total_computes > 0 else 0.0,
                },
                "skip_rate_by_priority": skip_rate_by_priority,
                "cpu_time_saved_estimate_seconds": cpu_time_saved_seconds,
                "throttle_duration_total_seconds": round(self._throttle_duration_total, 1),
                "throttle_activation_count": self._throttle_activation_count,
                "per_symbol_stats": per_symbol_stats,
                "avg_compute_interval_by_priority_ms": avg_interval_by_priority,
                "total_symbols": len(self._symbol_states),
                "priority_distribution": {
                    pri: stats["count"] for pri, stats in priority_stats.items()
                },
            }

    # ========================================================================
    #  生产级特性 4: Health Check（健康检查）
    # ========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        对调度器进行健康检查，返回状态和问题列表。

        Returns:
            {
                "status": "healthy" | "warning" | "critical",
                "issues": [...],
                "cpu": {"current": float, "trend": str},
                "memory": {"current": float, "trend": str},
                "throttle": {...},
                "timestamp": float
            }
        """
        with self._lock:
            issues = []
            status = "healthy"

            # CPU 趋势分析
            cpu_trend = "stable"
            if len(self._cpu_history) >= 3:
                recent = list(self._cpu_history)[-3:]
                if all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                    cpu_trend = "rising"
                elif all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                    cpu_trend = "falling"

            # Memory 趋势
            memory_trend = "stable"  # 暂无内存历史，使用当前值

            # CPU 检查
            current_cpu = self._current_cpu
            if current_cpu >= self._cpu_critical_threshold:
                issues.append(
                    f"CPU at {current_cpu:.1f}% (critical threshold: {self._cpu_critical_threshold:.0f}%)"
                )
                status = "critical"
            elif current_cpu >= self._cpu_throttle_threshold:
                issues.append(
                    f"CPU at {current_cpu:.1f}% (throttle threshold: {self._cpu_throttle_threshold:.0f}%)"
                )
                if status != "critical":
                    status = "warning"
            elif cpu_trend == "rising" and current_cpu >= self._cpu_throttle_threshold * 0.8:
                issues.append(
                    f"CPU trending up ({current_cpu:.1f}%), approaching throttle threshold"
                )
                if status == "healthy":
                    status = "warning"

            # 内存检查
            if self._current_memory >= self._max_memory_usage:
                issues.append(
                    f"Memory at {self._current_memory:.1f}% (max: {self._max_memory_usage}%)"
                )
                if status != "critical":
                    status = "critical"
            elif self._current_memory >= self._max_memory_usage * 0.85:
                issues.append(
                    f"Memory at {self._current_memory:.1f}% (warning: {self._max_memory_usage * 0.85:.0f}%)"
                )
                if status == "healthy":
                    status = "warning"

            # 降频持续时间检查
            if self._global_throttle_active:
                throttle_duration = time.time() - self._throttle_start_time if self._throttle_start_time else 0
                if throttle_duration > 300:  # 超过5分钟
                    issues.append(
                        f"Global throttle active for {throttle_duration:.0f}s (>{300}s), may indicate sustained overload"
                    )
                    if status != "critical":
                        status = "warning"
                elif throttle_duration > 60:
                    issues.append(f"Global throttle active for {throttle_duration:.0f}s")

            # 未注册 symbol 检查
            if len(self._symbol_states) == 0:
                issues.append("No symbols registered for compute scheduling")
                if status == "healthy":
                    status = "warning"

            return {
                "status": status,
                "issues": issues,
                "cpu": {
                    "current": round(current_cpu, 1),
                    "trend": cpu_trend,
                    "throttle_threshold": self._cpu_throttle_threshold,
                    "critical_threshold": self._cpu_critical_threshold,
                },
                "memory": {
                    "current": round(self._current_memory, 1),
                    "trend": memory_trend,
                    "max_threshold": self._max_memory_usage,
                },
                "throttle": {
                    "active": self._global_throttle_active,
                    "activation_count": self._throttle_activation_count,
                    "total_duration_seconds": round(self._throttle_duration_total, 1),
                    "current_duration_seconds": round(
                        time.time() - self._throttle_start_time, 1
                    ) if self._throttle_start_time else 0,
                },
                "symbols_registered": len(self._symbol_states),
                "timestamp": time.time(),
            }

    # ========================================================================
    #  生产级特性 5: Adaptive Throttling（自适应降频）
    # ========================================================================

    def _adaptive_throttle_adjust(self) -> None:
        """
        根据 CPU 使用率趋势自适应调整降频策略。

        逻辑：
        - 追踪最近 10 次 CPU 采样
        - CPU 持续上升 → 预判性增加跳过比例
        - CPU 持续下降 → 逐步恢复跳过比例
        - 确保 skip_ratio 在合理范围 [1, 10] 内
        - P13: 日志防抖 - 60秒内不重复输出相同趋势的调整日志
        """
        if len(self._cpu_history) < 5:
            return  # 数据不足，不做调整

        history = list(self._cpu_history)
        recent_5 = history[-5:]
        older_5 = history[-10:-5] if len(history) >= 10 else history[:5]

        avg_recent = sum(recent_5) / len(recent_5)
        avg_older = sum(older_5) / len(older_5)
        delta = avg_recent - avg_older

        old_ratio = self._low_vol_skip_ratio

        # P13: 日志防抖 - 60秒内不重复输出
        now = time.time()
        if not hasattr(self, '_last_throttle_log_ts'):
            self._last_throttle_log_ts = 0.0
            self._last_throttle_log_direction = None

        if delta > 5.0:
            # CPU 明显上升：预判性增加跳过比例
            new_ratio = min(self._low_vol_skip_ratio + 1, 10)
            if new_ratio != self._low_vol_skip_ratio:
                self._low_vol_skip_ratio = new_ratio
                if (now - self._last_throttle_log_ts > 60 or
                        self._last_throttle_log_direction != 'up'):
                    logger.info(
                        f"Adaptive throttle: CPU trending up ({delta:+.1f}%), "
                        f"skip_ratio increased {old_ratio} → {new_ratio}"
                    )
                    self._last_throttle_log_ts = now
                    self._last_throttle_log_direction = 'up'
        elif delta < -5.0:
            # CPU 明显下降：逐步恢复
            new_ratio = max(self._low_vol_skip_ratio - 1, 1)
            if new_ratio != self._low_vol_skip_ratio:
                self._low_vol_skip_ratio = new_ratio
                if (now - self._last_throttle_log_ts > 60 or
                        self._last_throttle_log_direction != 'down'):
                    logger.info(
                        f"Adaptive throttle: CPU trending down ({delta:+.1f}%), "
                        f"skip_ratio decreased {old_ratio} → {new_ratio}"
                    )
                    self._last_throttle_log_ts = now
                    self._last_throttle_log_direction = 'down'

    # ========================================================================
    #  生产级特性 6: Compute Budget（计算预算）
    # ========================================================================

    def check_compute_budget(self, symbol: str) -> bool:
        """
        检查 symbol 是否在每秒计算预算内。

        每个优先级有独立的每秒最大计算次数预算。
        预算窗口每秒重置一次。

        Args:
            symbol: 交易对

        Returns:
            True: 预算充足，可以计算
            False: 预算已耗尽，建议跳过
        """
        with self._lock:
            state = self._symbol_states.get(symbol)
            if state is None:
                return True

            pri = state.priority.value
            budget = self._compute_budget.get(pri, 10)

            # 每秒重置预算窗口
            now = time.time()
            if now - self._budget_window_start >= 1.0:
                self._budget_window_start = now
                for key in self._budget_counts:
                    self._budget_counts[key] = 0

            current_count = self._budget_counts.get(pri, 0)
            if current_count < budget:
                self._budget_counts[pri] = current_count + 1
                return True
            else:
                logger.warning(
                    f"Compute budget exhausted for priority '{pri}': "
                    f"{current_count}/{budget}/s, symbol={symbol}"
                )
                return False

# 全局单例
_compute_scheduler_instance: Optional[ComputeScheduler] = None


def get_compute_scheduler(config: Dict[str, Any] = None) -> ComputeScheduler:
    """获取算力调度器单例"""
    global _compute_scheduler_instance
    if _compute_scheduler_instance is None:
        _compute_scheduler_instance = ComputeScheduler(config)
    return _compute_scheduler_instance
