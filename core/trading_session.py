"""
生产级交易会话管理器
====================
核心定位：统一编排交易全生命周期，确保系统安全、稳定、高效运行。

功能：
- 会话状态机（PRE_MARKET → TRADING → POST_MARKET → CLOSED）
- 盘前检查（余额、连接、风控限额、持仓同步）
- 策略调度与优先级管理
- 优雅关闭与仓位排空
- 紧急暂停/恢复
- 会话级指标采集与报告
- 自动会话调度（定时启动/关闭）

架构：
  TradingSessionManager
  ├── SessionStateMachine（状态机）
  ├── PreMarketChecker（盘前检查）
  ├── StrategyScheduler（策略调度器）
  ├── EmergencyHandler（紧急处理）
  └── SessionMetrics（会话指标）
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class SessionState(Enum):
    """会话状态"""
    IDLE = "idle"                      # 空闲
    PRE_MARKET = "pre_market"          # 盘前检查
    STARTING = "starting"              # 启动中
    TRADING = "trading"                # 交易中
    PAUSED = "paused"                  # 已暂停
    RESUMING = "resuming"              # 恢复中
    DRAINING = "draining"              # 排空中（逐步清仓）
    POST_MARKET = "post_market"        # 盘后结算
    CLOSED = "closed"                  # 已关闭
    EMERGENCY = "emergency"            # 紧急状态
    ERROR = "error"                    # 错误


class SessionEvent(Enum):
    """会话事件"""
    # 生命周期事件
    INIT_COMPLETE = "init_complete"
    PRE_CHECKS_PASSED = "pre_checks_passed"
    PRE_CHECKS_FAILED = "pre_checks_failed"
    START_COMPLETE = "start_complete"
    PAUSE_REQUESTED = "pause_requested"
    PAUSE_COMPLETE = "pause_complete"
    RESUME_REQUESTED = "resume_requested"
    RESUME_COMPLETE = "resume_complete"
    DRAIN_REQUESTED = "drain_requested"
    DRAIN_COMPLETE = "drain_complete"
    CLOSE_REQUESTED = "close_requested"
    CLOSE_COMPLETE = "close_complete"
    # 紧急事件
    EMERGENCY_TRIGGERED = "emergency_triggered"
    EMERGENCY_RESOLVED = "emergency_resolved"
    # 错误事件
    CRITICAL_ERROR = "critical_error"
    RECOVERABLE_ERROR = "recoverable_error"
    # 定时事件
    SCHEDULED_START = "scheduled_start"
    SCHEDULED_CLOSE = "scheduled_close"
    # 外部事件
    MANUAL_START = "manual_start"
    MANUAL_STOP = "manual_stop"


# 状态转换表
_TRANSITIONS: Dict[SessionState, Dict[SessionEvent, SessionState]] = {
    SessionState.IDLE: {
        SessionEvent.INIT_COMPLETE: SessionState.PRE_MARKET,
        SessionEvent.MANUAL_START: SessionState.STARTING,
    },
    SessionState.PRE_MARKET: {
        SessionEvent.PRE_CHECKS_PASSED: SessionState.STARTING,
        SessionEvent.PRE_CHECKS_FAILED: SessionState.IDLE,
        SessionEvent.CRITICAL_ERROR: SessionState.ERROR,
    },
    SessionState.STARTING: {
        SessionEvent.START_COMPLETE: SessionState.TRADING,
        SessionEvent.CRITICAL_ERROR: SessionState.ERROR,
    },
    SessionState.TRADING: {
        SessionEvent.PAUSE_REQUESTED: SessionState.PAUSED,
        SessionEvent.DRAIN_REQUESTED: SessionState.DRAINING,
        SessionEvent.CLOSE_REQUESTED: SessionState.POST_MARKET,
        SessionEvent.EMERGENCY_TRIGGERED: SessionState.EMERGENCY,
        SessionEvent.RECOVERABLE_ERROR: SessionState.TRADING,
        SessionEvent.CRITICAL_ERROR: SessionState.ERROR,
    },
    SessionState.PAUSED: {
        SessionEvent.RESUME_REQUESTED: SessionState.RESUMING,
        SessionEvent.DRAIN_REQUESTED: SessionState.DRAINING,
        SessionEvent.CLOSE_REQUESTED: SessionState.POST_MARKET,
        SessionEvent.EMERGENCY_TRIGGERED: SessionState.EMERGENCY,
    },
    SessionState.RESUMING: {
        SessionEvent.RESUME_COMPLETE: SessionState.TRADING,
        SessionEvent.CRITICAL_ERROR: SessionState.ERROR,
    },
    SessionState.DRAINING: {
        SessionEvent.DRAIN_COMPLETE: SessionState.POST_MARKET,
        SessionEvent.EMERGENCY_TRIGGERED: SessionState.EMERGENCY,
        SessionEvent.CRITICAL_ERROR: SessionState.ERROR,
    },
    SessionState.POST_MARKET: {
        SessionEvent.CLOSE_COMPLETE: SessionState.CLOSED,
        SessionEvent.CRITICAL_ERROR: SessionState.ERROR,
    },
    SessionState.CLOSED: {
        SessionEvent.INIT_COMPLETE: SessionState.PRE_MARKET,
        SessionEvent.SCHEDULED_START: SessionState.STARTING,
    },
    SessionState.EMERGENCY: {
        SessionEvent.EMERGENCY_RESOLVED: SessionState.PAUSED,
        SessionEvent.DRAIN_REQUESTED: SessionState.DRAINING,
        SessionEvent.CLOSE_REQUESTED: SessionState.POST_MARKET,
    },
    SessionState.ERROR: {
        SessionEvent.INIT_COMPLETE: SessionState.PRE_MARKET,
        SessionEvent.MANUAL_START: SessionState.STARTING,
    },
}


@dataclass
class PreMarketCheckResult:
    """盘前检查结果"""
    passed: bool = False
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class SessionMetrics:
    """会话指标"""
    session_id: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_seconds: float = 0.0
    total_signals: int = 0
    total_orders: int = 0
    total_fills: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    peak_equity: float = 0.0
    final_equity: float = 0.0
    emergency_count: int = 0
    error_count: int = 0
    pause_count: int = 0
    state_history: List[Dict[str, Any]] = field(default_factory=list)
    strategy_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class StrategySchedule:
    """策略调度配置"""
    name: str
    priority: int = 0          # 0=最高, 越大越低
    start_delay_sec: float = 0  # 启动延迟
    depends_on: List[str] = field(default_factory=list)  # 依赖策略
    auto_restart: bool = True
    max_restarts: int = 3
    restart_cooldown_sec: float = 60.0


# ═══════════════════════════════════════════════════════════════
# TradingSessionManager
# ═══════════════════════════════════════════════════════════════

class TradingSessionManager:
    """
    生产级交易会话管理器

    使用示例:
        mgr = TradingSessionManager(config)
        mgr.register_strategy("grid", grid_strategy, StrategySchedule(name="grid", priority=1))
        mgr.register_strategy("trend", trend_strategy, StrategySchedule(name="trend", priority=0))
        await mgr.start_session()
        # ... 运行中 ...
        await mgr.close_session()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        session_cfg = config.get("trading_session", {})

        # ── 会话配置 ──
        self._session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._state = SessionState.IDLE
        self._previous_state = SessionState.IDLE
        self._state_lock = asyncio.Lock()

        # ── 调度配置 ──
        self._auto_start_enabled = session_cfg.get("auto_start_enabled", False)
        self._auto_start_time = session_cfg.get("auto_start_time", "00:00")
        self._auto_close_enabled = session_cfg.get("auto_close_enabled", False)
        self._auto_close_time = session_cfg.get("auto_close_time", "23:59")
        self._schedule_check_interval = session_cfg.get("schedule_check_interval", 30)

        # ── 盘前检查配置 ──
        self._pre_check_balance_min = session_cfg.get("pre_check_balance_min", 10.0)
        self._pre_check_connectivity = session_cfg.get("pre_check_connectivity", True)
        self._pre_check_risk_limits = session_cfg.get("pre_check_risk_limits", True)
        self._pre_check_position_sync = session_cfg.get("pre_check_position_sync", True)
        self._pre_check_timeout = session_cfg.get("pre_check_timeout", 30)

        # ── 紧急处理配置 ──
        self._emergency_drawdown_limit = session_cfg.get("emergency_drawdown_limit", 0.30)
        self._emergency_margin_limit = session_cfg.get("emergency_margin_limit", 0.85)
        self._emergency_auto_drain = session_cfg.get("emergency_auto_drain", True)

        # ── 排空配置 ──
        self._drain_timeout = session_cfg.get("drain_timeout", 300)
        self._drain_check_interval = session_cfg.get("drain_check_interval", 5)

        # ── 注册的策略 ──
        self._strategies: Dict[str, Any] = {}
        self._strategy_schedules: Dict[str, StrategySchedule] = {}
        self._strategy_status: Dict[str, Dict[str, Any]] = {}

        # ── 外部依赖 ──
        self._okx_client = None
        self._account_manager = None
        self._position_manager = None
        self._risk_gate = None
        self._circuit_breaker = None
        self._order_executor = None
        self._capital_manager = None

        # ── 回调 ──
        self._state_change_callbacks: List[Callable] = []
        self._emergency_callbacks: List[Callable] = []
        self._alert_callbacks: List[Callable] = []

        # ── 指标 ──
        self._metrics = SessionMetrics(session_id=self._session_id)
        self._event_log: List[Dict[str, Any]] = []
        self._max_event_log = session_cfg.get("max_event_log", 1000)

        # ── 运行控制 ──
        self._running = False
        self._schedule_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._drain_task: Optional[asyncio.Task] = None

        # ── 持久化 ──
        self._persist_dir = config.get("system", {}).get("data_dir", "data")
        self._session_log_file = os.path.join(self._persist_dir, "session_log.jsonl")

        logger.info(
            f"TradingSessionManager initialized: session_id={self._session_id}, "
            f"auto_start={self._auto_start_enabled}, auto_close={self._auto_close_enabled}"
        )

    # ═══════════════════════════════════════════════════════════════
    # 依赖注入
    # ═══════════════════════════════════════════════════════════════

    def inject_dependencies(self, okx_client=None, account_manager=None,
                            position_manager=None, risk_gate=None,
                            circuit_breaker=None, order_executor=None,
                            capital_manager=None):
        """注入外部依赖"""
        self._okx_client = okx_client
        self._account_manager = account_manager
        self._position_manager = position_manager
        self._risk_gate = risk_gate
        self._circuit_breaker = circuit_breaker
        self._order_executor = order_executor
        self._capital_manager = capital_manager
        logger.info("TradingSessionManager dependencies injected")

    # ═══════════════════════════════════════════════════════════════
    # 策略注册
    # ═══════════════════════════════════════════════════════════════

    def register_strategy(self, name: str, strategy_instance: Any,
                          schedule: StrategySchedule = None):
        """注册策略到会话管理器"""
        if schedule is None:
            schedule = StrategySchedule(name=name)

        self._strategies[name] = strategy_instance
        self._strategy_schedules[name] = schedule
        self._strategy_status[name] = {
            "active": False,
            "start_time": None,
            "restart_count": 0,
            "last_restart": None,
            "error_count": 0,
            "last_error": None,
        }
        logger.info(f"Strategy registered: {name} (priority={schedule.priority})")

    def unregister_strategy(self, name: str):
        """注销策略"""
        self._strategies.pop(name, None)
        self._strategy_schedules.pop(name, None)
        self._strategy_status.pop(name, None)
        logger.info(f"Strategy unregistered: {name}")

    # ═══════════════════════════════════════════════════════════════
    # 状态机
    # ═══════════════════════════════════════════════════════════════

    @property
    def state(self) -> SessionState:
        return self._state

    def can_transition(self, event: SessionEvent) -> bool:
        """检查是否可以转换状态"""
        valid_states = _TRANSITIONS.get(self._state, {})
        return event in valid_states

    async def _transition(self, event: SessionEvent) -> bool:
        """执行状态转换"""
        async with self._state_lock:
            valid_states = _TRANSITIONS.get(self._state, {})
            if event not in valid_states:
                logger.warning(
                    f"Invalid transition: {self._state.value} -> {event.value} "
                    f"(valid: {[e.value for e in valid_states]})"
                )
                return False

            new_state = valid_states[event]
            self._previous_state = self._state
            self._state = new_state

            # 记录状态变更
            self._log_event({
                "type": "state_change",
                "from": self._previous_state.value,
                "to": new_state.value,
                "event": event.value,
                "timestamp": time.time(),
            })

            self._metrics.state_history.append({
                "state": new_state.value,
                "timestamp": datetime.now().isoformat(),
            })

            logger.info(
                f"Session state: {self._previous_state.value} -> {new_state.value} "
                f"(event: {event.value})"
            )

            # 通知回调
            for cb in self._state_change_callbacks:
                try:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(self._previous_state, new_state, event)
                    else:
                        cb(self._previous_state, new_state, event)
                except Exception as e:
                    logger.error(f"State change callback error: {e}")

            return True

    # ═══════════════════════════════════════════════════════════════
    # 盘前检查
    # ═══════════════════════════════════════════════════════════════

    async def _run_pre_market_checks(self) -> PreMarketCheckResult:
        """执行盘前检查"""
        result = PreMarketCheckResult(timestamp=time.time())
        logger.info("Running pre-market checks...")

        # 1. 连接检查
        if self._pre_check_connectivity and self._okx_client:
            try:
                ticker = self._okx_client.get_ticker("BTC-USDT")
                if ticker and ticker.get("last"):
                    result.checks["connectivity"] = True
                    result.details["connectivity"] = f"BTC-USDT last={ticker['last']}"
                else:
                    result.checks["connectivity"] = False
                    result.errors.append("无法获取BTC-USDT行情")
            except Exception as e:
                result.checks["connectivity"] = False
                result.errors.append(f"连接检查失败: {e}")
        else:
            result.checks["connectivity"] = True

        # 2. 余额检查
        if self._account_manager:
            try:
                account_info = self._account_manager.get_account_info()
                if account_info:
                    total_eq = float(account_info.get("totalEq", 0))
                    avail_bal = float(account_info.get("availBal", 0))
                    result.details["total_equity"] = f"{total_eq:.2f} USDT"
                    result.details["available_balance"] = f"{avail_bal:.2f} USDT"

                    if total_eq >= self._pre_check_balance_min:
                        result.checks["balance"] = True
                    else:
                        result.checks["balance"] = False
                        result.errors.append(
                            f"余额不足: {total_eq:.2f} < {self._pre_check_balance_min} USDT"
                        )
                else:
                    result.checks["balance"] = False
                    result.errors.append("无法获取账户信息")
            except Exception as e:
                result.checks["balance"] = False
                result.errors.append(f"余额检查失败: {e}")
        else:
            result.checks["balance"] = True
            result.warnings.append("未注入AccountManager，跳过余额检查")

        # 3. 风控限额检查
        if self._pre_check_risk_limits and self._risk_gate:
            try:
                risk_status = self._risk_gate.get_status()
                if risk_status:
                    result.checks["risk_limits"] = not risk_status.get("blocked", False)
                    result.details["risk_status"] = str(risk_status)
                    if risk_status.get("blocked"):
                        result.errors.append("风控门禁已封锁")
                else:
                    result.checks["risk_limits"] = True
            except Exception as e:
                result.checks["risk_limits"] = False
                result.errors.append(f"风控检查失败: {e}")
        else:
            result.checks["risk_limits"] = True

        # 4. 持仓同步检查
        if self._pre_check_position_sync and self._position_manager:
            try:
                positions = self._position_manager.get_all_positions()
                result.checks["position_sync"] = True
                result.details["position_count"] = str(len(positions))
                if len(positions) > 0:
                    result.warnings.append(f"检测到 {len(positions)} 个未平仓持仓")
            except Exception as e:
                result.checks["position_sync"] = False
                result.errors.append(f"持仓同步检查失败: {e}")
        else:
            result.checks["position_sync"] = True

        # 5. 熔断器检查
        if self._circuit_breaker:
            try:
                cb_status = self._circuit_breaker.get_status()
                if cb_status:
                    is_tripped = cb_status.get("tripped", False)
                    result.checks["circuit_breaker"] = not is_tripped
                    if is_tripped:
                        result.errors.append("熔断器已触发")
                else:
                    result.checks["circuit_breaker"] = True
            except Exception as e:
                result.checks["circuit_breaker"] = False
                result.errors.append(f"熔断器检查失败: {e}")
        else:
            result.checks["circuit_breaker"] = True

        result.passed = all(result.checks.values()) if result.checks else True

        logger.info(
            f"Pre-market checks complete: passed={result.passed}, "
            f"checks={result.checks}, errors={len(result.errors)}"
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # 策略调度
    # ═══════════════════════════════════════════════════════════════

    async def _start_strategies(self):
        """按优先级和依赖顺序启动策略"""
        # 按优先级排序
        sorted_strategies = sorted(
            self._strategy_schedules.items(),
            key=lambda x: x[1].priority
        )

        started = set()
        failed = []

        for name, schedule in sorted_strategies:
            if name not in self._strategies:
                continue

            # 检查依赖
            deps_ready = all(dep in started for dep in schedule.depends_on)
            if not deps_ready:
                missing = [d for d in schedule.depends_on if d not in started]
                logger.warning(f"Strategy {name} dependencies not ready: {missing}, deferring")
                continue

            strategy = self._strategies[name]

            try:
                # 延迟启动
                if schedule.start_delay_sec > 0:
                    await asyncio.sleep(schedule.start_delay_sec)

                # 启动策略
                if hasattr(strategy, "start") and callable(strategy.start):
                    result = strategy.start()
                    if asyncio.iscoroutine(result):
                        await result

                self._strategy_status[name].update({
                    "active": True,
                    "start_time": time.time(),
                })
                started.add(name)
                logger.info(f"Strategy started: {name} (priority={schedule.priority})")

            except Exception as e:
                logger.error(f"Failed to start strategy {name}: {e}")
                self._strategy_status[name]["error_count"] += 1
                self._strategy_status[name]["last_error"] = str(e)
                failed.append(name)

        if failed:
            logger.warning(f"Failed to start {len(failed)} strategies: {failed}")

    async def _stop_strategies(self, drain: bool = False):
        """停止所有策略"""
        for name, strategy in reversed(list(self._strategies.items())):
            try:
                if drain and hasattr(strategy, "drain"):
                    result = strategy.drain()
                    if asyncio.iscoroutine(result):
                        await result

                if hasattr(strategy, "stop") and callable(strategy.stop):
                    result = strategy.stop()
                    if asyncio.iscoroutine(result):
                        await result

                self._strategy_status[name]["active"] = False
                logger.info(f"Strategy stopped: {name}")

            except Exception as e:
                logger.error(f"Failed to stop strategy {name}: {e}")

    async def _pause_strategies(self):
        """暂停所有策略"""
        for name, strategy in self._strategies.items():
            try:
                if hasattr(strategy, "pause") and callable(strategy.pause):
                    result = strategy.pause()
                    if asyncio.iscoroutine(result):
                        await result
                self._strategy_status[name]["active"] = False
                logger.info(f"Strategy paused: {name}")
            except Exception as e:
                logger.error(f"Failed to pause strategy {name}: {e}")

    async def _resume_strategies(self):
        """恢复所有策略"""
        for name, strategy in self._strategies.items():
            try:
                if hasattr(strategy, "resume") and callable(strategy.resume):
                    result = strategy.resume()
                    if asyncio.iscoroutine(result):
                        await result
                self._strategy_status[name]["active"] = True
                logger.info(f"Strategy resumed: {name}")
            except Exception as e:
                logger.error(f"Failed to resume strategy {name}: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 会话生命周期
    # ═══════════════════════════════════════════════════════════════

    async def start_session(self, skip_checks: bool = False) -> bool:
        """启动交易会话"""
        if self._state not in (SessionState.IDLE, SessionState.CLOSED):
            logger.warning(f"Cannot start session in state: {self._state.value}")
            return False

        logger.info("Starting trading session...")
        self._metrics.start_time = time.time()

        # 1. 盘前检查
        if not skip_checks:
            await self._transition(SessionEvent.INIT_COMPLETE)
            checks = await self._run_pre_market_checks()
            if not checks.passed:
                logger.error(f"Pre-market checks failed: {checks.errors}")
                await self._transition(SessionEvent.PRE_CHECKS_FAILED)
                return False
            await self._transition(SessionEvent.PRE_CHECKS_PASSED)
        else:
            await self._transition(SessionEvent.INIT_COMPLETE)
            await self._transition(SessionEvent.PRE_CHECKS_PASSED)

        # 2. 启动策略
        await self._transition(SessionEvent.START_COMPLETE)
        await self._start_strategies()

        # 3. 启动监控
        self._running = True
        if self._auto_start_enabled or self._auto_close_enabled:
            self._schedule_task = asyncio.create_task(self._schedule_loop())
        self._monitor_task = asyncio.create_task(self._emergency_monitor_loop())

        logger.info(f"Trading session started: {self._session_id}")
        return True

    async def pause_session(self, reason: str = "") -> bool:
        """暂停交易会话"""
        if not await self._transition(SessionEvent.PAUSE_REQUESTED):
            return False

        logger.info(f"Pausing trading session: {reason}")
        await self._pause_strategies()
        self._metrics.pause_count += 1

        self._log_event({
            "type": "session_paused",
            "reason": reason,
            "timestamp": time.time(),
        })

        return True

    async def resume_session(self) -> bool:
        """恢复交易会话"""
        if not await self._transition(SessionEvent.RESUME_REQUESTED):
            return False

        logger.info("Resuming trading session...")
        await self._resume_strategies()
        await self._transition(SessionEvent.RESUME_COMPLETE)

        return True

    async def close_session(self, drain_first: bool = True) -> bool:
        """关闭交易会话"""
        if drain_first and self._state in (SessionState.TRADING, SessionState.PAUSED):
            await self._drain_positions()

        if not await self._transition(SessionEvent.CLOSE_REQUESTED if not drain_first
                                       else SessionEvent.DRAIN_COMPLETE):
            return False

        logger.info("Closing trading session...")

        # 停止策略
        await self._stop_strategies()

        # 停止监控
        self._running = False
        if self._schedule_task:
            self._schedule_task.cancel()
        if self._monitor_task:
            self._monitor_task.cancel()

        # 计算指标
        self._metrics.end_time = time.time()
        self._metrics.duration_seconds = self._metrics.end_time - self._metrics.start_time

        # 持久化
        self._save_session_log()

        await self._transition(SessionEvent.CLOSE_COMPLETE)
        logger.info(f"Trading session closed: {self._session_id}, "
                     f"duration={self._metrics.duration_seconds:.0f}s")

        return True

    # ═══════════════════════════════════════════════════════════════
    # 紧急处理
    # ═══════════════════════════════════════════════════════════════

    async def trigger_emergency(self, reason: str) -> bool:
        """触发紧急状态"""
        if not await self._transition(SessionEvent.EMERGENCY_TRIGGERED):
            return False

        logger.error(f"EMERGENCY TRIGGERED: {reason}")
        self._metrics.emergency_count += 1

        # 暂停所有策略
        await self._pause_strategies()

        # 通知回调
        for cb in self._emergency_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(reason)
                else:
                    cb(reason)
            except Exception as e:
                logger.error(f"Emergency callback error: {e}")

        # 自动排空
        if self._emergency_auto_drain:
            asyncio.create_task(self._drain_positions())

        self._log_event({
            "type": "emergency",
            "reason": reason,
            "timestamp": time.time(),
        })

        return True

    async def resolve_emergency(self) -> bool:
        """解决紧急状态"""
        if not await self._transition(SessionEvent.EMERGENCY_RESOLVED):
            return False

        logger.info("Emergency resolved, session paused for review")
        return True

    async def _emergency_monitor_loop(self):
        """紧急状态监控循环"""
        while self._running:
            try:
                await asyncio.sleep(5)

                if self._state not in (SessionState.TRADING, SessionState.PAUSED):
                    continue

                # 检查回撤
                if self._account_manager:
                    account_info = self._account_manager.get_account_info()
                    if account_info:
                        total_eq = float(account_info.get("totalEq", 0))
                        if total_eq > 0 and self._metrics.peak_equity > 0:
                            drawdown = (self._metrics.peak_equity - total_eq) / self._metrics.peak_equity
                            if drawdown >= self._emergency_drawdown_limit:
                                await self.trigger_emergency(
                                    f"Drawdown {drawdown:.2%} exceeded limit {self._emergency_drawdown_limit:.2%}"
                                )
                        if total_eq > self._metrics.peak_equity:
                            self._metrics.peak_equity = total_eq

                # 检查保证金率
                if self._account_manager:
                    account_info = self._account_manager.get_account_info()
                    if account_info:
                        mgn_ratio = float(account_info.get("mgnRatio", 0))
                        if mgn_ratio >= self._emergency_margin_limit:
                            await self.trigger_emergency(
                                f"Margin ratio {mgn_ratio:.2%} exceeded limit {self._emergency_margin_limit:.2%}"
                            )

                # 检查熔断器
                if self._circuit_breaker:
                    cb_status = self._circuit_breaker.get_status()
                    if cb_status and cb_status.get("tripped"):
                        await self.trigger_emergency(
                            f"Circuit breaker tripped: {cb_status.get('reason', 'unknown')}"
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Emergency monitor error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 仓位排空
    # ═══════════════════════════════════════════════════════════════

    async def _drain_positions(self):
        """排空所有持仓"""
        if not await self._transition(SessionEvent.DRAIN_REQUESTED):
            return

        logger.info("Starting position drain...")
        start_time = time.time()

        try:
            while time.time() - start_time < self._drain_timeout:
                if self._position_manager:
                    positions = self._position_manager.get_all_positions()
                    if not positions:
                        logger.info("All positions drained")
                        break

                    logger.info(f"Draining {len(positions)} positions...")
                    # 通过订单执行器平仓
                    if self._order_executor:
                        for pos in positions:
                            try:
                                close_signal = {
                                    "symbol": pos.symbol,
                                    "direction": "sell" if pos.side.value == "long" else "buy",
                                    "quantity": pos.quantity,
                                    "strategy_name": "session_drain",
                                    "signal_type": "emergency_close",
                                    "reduce_only": True,
                                }
                                await self._order_executor._execute_order(close_signal)
                            except Exception as e:
                                logger.error(f"Failed to drain position {pos.symbol}: {e}")

                await asyncio.sleep(self._drain_check_interval)

            else:
                logger.warning(f"Drain timeout ({self._drain_timeout}s), forcing close")
                remaining = self._position_manager.get_all_positions() if self._position_manager else []
                if remaining:
                    logger.warning(f"Positions still open after drain: {len(remaining)}")

            await self._transition(SessionEvent.DRAIN_COMPLETE)

        except Exception as e:
            logger.error(f"Drain error: {e}")
            await self._transition(SessionEvent.CRITICAL_ERROR)

    # ═══════════════════════════════════════════════════════════════
    # 自动调度
    # ═══════════════════════════════════════════════════════════════

    async def _schedule_loop(self):
        """自动调度循环"""
        while self._running:
            try:
                now = datetime.now()
                current_time = now.strftime("%H:%M")

                # 自动启动
                if (self._auto_start_enabled and
                        self._state == SessionState.CLOSED and
                        current_time == self._auto_start_time):
                    logger.info(f"Auto-start triggered at {current_time}")
                    await self.start_session()

                # 自动关闭
                if (self._auto_close_enabled and
                        self._state == SessionState.TRADING and
                        current_time == self._auto_close_time):
                    logger.info(f"Auto-close triggered at {current_time}")
                    await self.close_session(drain_first=True)

                await asyncio.sleep(self._schedule_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Schedule loop error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════════════

    def get_session_id(self) -> str:
        return self._session_id

    def get_state(self) -> str:
        return self._state.value

    def get_strategy_status(self, name: str = None) -> Dict[str, Any]:
        """获取策略状态"""
        if name:
            return self._strategy_status.get(name, {})
        return dict(self._strategy_status)

    def get_active_strategies(self) -> List[str]:
        """获取活跃策略列表"""
        return [
            name for name, status in self._strategy_status.items()
            if status.get("active")
        ]

    def get_metrics(self) -> Dict[str, Any]:
        """获取会话指标"""
        return {
            "session_id": self._metrics.session_id,
            "state": self._state.value,
            "duration_seconds": (time.time() - self._metrics.start_time
                                 if self._metrics.start_time > 0 and self._running else
                                 self._metrics.duration_seconds),
            "total_signals": self._metrics.total_signals,
            "total_orders": self._metrics.total_orders,
            "total_fills": self._metrics.total_fills,
            "total_pnl": self._metrics.total_pnl,
            "total_fees": self._metrics.total_fees,
            "peak_equity": self._metrics.peak_equity,
            "emergency_count": self._metrics.emergency_count,
            "pause_count": self._metrics.pause_count,
            "active_strategies": len(self.get_active_strategies()),
            "event_count": len(self._event_log),
        }

    def get_event_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取事件日志"""
        return self._event_log[-limit:]

    # ═══════════════════════════════════════════════════════════════
    # 回调注册
    # ═══════════════════════════════════════════════════════════════

    def on_state_change(self, callback: Callable):
        """注册状态变更回调"""
        self._state_change_callbacks.append(callback)

    def on_emergency(self, callback: Callable):
        """注册紧急事件回调"""
        self._emergency_callbacks.append(callback)

    def on_alert(self, callback: Callable):
        """注册告警回调"""
        self._alert_callbacks.append(callback)

    # ═══════════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════════

    def _log_event(self, event: Dict[str, Any]):
        """记录事件"""
        self._event_log.append(event)
        if len(self._event_log) > self._max_event_log:
            self._event_log = self._event_log[-self._max_event_log:]

    def _save_session_log(self):
        """保存会话日志"""
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            with open(self._session_log_file, "a", encoding="utf-8") as f:
                summary = {
                    "session_id": self._session_id,
                    "start_time": datetime.fromtimestamp(self._metrics.start_time).isoformat(),
                    "end_time": datetime.fromtimestamp(self._metrics.end_time).isoformat(),
                    "duration_seconds": self._metrics.duration_seconds,
                    "state_history": self._metrics.state_history,
                    "metrics": self.get_metrics(),
                }
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
            logger.info(f"Session log saved: {self._session_log_file}")
        except Exception as e:
            logger.error(f"Failed to save session log: {e}")

    def record_signal(self, signal_count: int = 1):
        """记录信号"""
        self._metrics.total_signals += signal_count

    def record_order(self, order_count: int = 1):
        """记录订单"""
        self._metrics.total_orders += order_count

    def record_fill(self, fill_count: int = 1, pnl: float = 0.0, fee: float = 0.0):
        """记录成交"""
        self._metrics.total_fills += fill_count
        self._metrics.total_pnl += pnl
        self._metrics.total_fees += fee