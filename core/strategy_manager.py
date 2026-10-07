"""
统一策略管理器 (StrategyManager)
=================================
策略注册、生命周期、配置热重载、版本控制、健康监控、依赖管理的中央枢纽。

核心职责:
1. 策略注册与发现 — 集中管理所有策略实例
2. 生命周期管理 — 统一的 init/prepare/start/pause/resume/stop/restart 流程
3. 配置热重载 — 运行时动态更新策略参数，不重启交易系统
4. 版本控制 — 配置变更自动快照，支持回滚
5. 健康监控 — 心跳检测、状态诊断、自动告警
6. 依赖管理 — 策略间启动顺序依赖 (如套利依赖趋势活跃)
7. 调度集成 — 统一注入 AdaptiveController/StopLossManager/Coordinator
"""

import os
import json
import copy
import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Type, Callable
from dataclasses import dataclass, field
from loguru import logger

from core.parameter_rollback_guard import ParameterRollbackGuard
from core.strategy_audit import get_strategy_audit_logger


# ============================================================
# 枚举与数据模型
# ============================================================

class StrategyLifecycle(Enum):
    """策略生命周期状态"""
    UNREGISTERED = "unregistered"     # 未注册
    REGISTERED = "registered"         # 已注册，等待初始化
    INITIALIZING = "initializing"     # 初始化中
    PREPARED = "prepared"             # 已准备，等待启动
    STARTING = "starting"             # 启动中
    RUNNING = "running"               # 运行中
    PAUSING = "pausing"               # 暂停中
    PAUSED = "paused"                 # 已暂停
    RESUMING = "resuming"             # 恢复中
    STOPPING = "stopping"             # 停止中
    STOPPED = "stopped"               # 已停止
    RESTARTING = "restarting"         # 重启中
    ERROR = "error"                   # 错误
    DEGRADED = "degraded"             # 降级运行（部分功能受限）
    DRAINING = "draining"             # 排空中（逐步清仓后停止）


class StrategyHealth(Enum):
    """策略健康等级"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class RunMode(Enum):
    """策略管理器运行模式"""
    NORMAL = "normal"           # 标准模式：所有策略正常运行
    CONSERVATIVE = "conservative"  # 保守模式：仅核心策略运行，降低杠杆
    AGGRESSIVE = "aggressive"   # 激进模式：最大化资金利用率，放宽风控
    EMERGENCY = "emergency"     # 紧急模式：全部暂停，仅允许平仓


# 允许的生命周期转换
_ALLOWED_TRANSITIONS = {
    StrategyLifecycle.UNREGISTERED: {StrategyLifecycle.REGISTERED},
    StrategyLifecycle.REGISTERED: {StrategyLifecycle.INITIALIZING},
    StrategyLifecycle.INITIALIZING: {StrategyLifecycle.PREPARED, StrategyLifecycle.ERROR},
    StrategyLifecycle.PREPARED: {StrategyLifecycle.STARTING},
    StrategyLifecycle.STARTING: {StrategyLifecycle.RUNNING, StrategyLifecycle.ERROR},
    StrategyLifecycle.RUNNING: {StrategyLifecycle.PAUSING, StrategyLifecycle.STOPPING,
                                 StrategyLifecycle.DRAINING, StrategyLifecycle.DEGRADED,
                                 StrategyLifecycle.ERROR},
    StrategyLifecycle.PAUSING: {StrategyLifecycle.PAUSED, StrategyLifecycle.ERROR},
    StrategyLifecycle.PAUSED: {StrategyLifecycle.RESUMING, StrategyLifecycle.STOPPING},
    StrategyLifecycle.RESUMING: {StrategyLifecycle.RUNNING, StrategyLifecycle.ERROR},
    StrategyLifecycle.STOPPING: {StrategyLifecycle.STOPPED, StrategyLifecycle.ERROR},
    StrategyLifecycle.STOPPED: {StrategyLifecycle.RESTARTING},
    StrategyLifecycle.RESTARTING: {StrategyLifecycle.STARTING, StrategyLifecycle.ERROR},
    StrategyLifecycle.ERROR: {StrategyLifecycle.RESTARTING, StrategyLifecycle.STOPPING},
    StrategyLifecycle.DEGRADED: {StrategyLifecycle.RUNNING, StrategyLifecycle.STOPPING,
                                  StrategyLifecycle.RESTARTING},
    StrategyLifecycle.DRAINING: {StrategyLifecycle.STOPPING, StrategyLifecycle.RUNNING},
}


@dataclass
class StrategyConfigSnapshot:
    """策略配置快照"""
    strategy_name: str
    version: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    applied: bool = False


@dataclass
class StrategyMetrics:
    """策略运行时指标"""
    uptime_seconds: float = 0.0
    trades_total: int = 0
    trades_last_hour: int = 0
    signals_generated: int = 0
    signals_last_hour: int = 0
    pnl_total: float = 0.0
    pnl_last_hour: float = 0.0
    last_signal_time: Optional[str] = None
    last_trade_time: Optional[str] = None
    health_score: float = 0.0
    health_level: str = "unknown"


@dataclass
class StrategyDescriptor:
    """策略描述符"""
    name: str                           # 策略名称 (grid/trend/scalping/arbitrage/...)
    display_name: str = ""              # 显示名称
    description: str = ""               # 策略描述
    category: str = "contract"          # 类别: contract/spot/hybrid
    dependencies: List[str] = field(default_factory=list)  # 依赖策略列表
    requires_market_data: bool = True
    requires_position_sync: bool = True
    class_path: str = ""                # 类路径 (strategies.grid_strategy.GridStrategy)
    enabled_by_default: bool = True
    config_schema_keys: List[str] = field(default_factory=list)  # 关键配置键


# ============================================================
# 核心管理器
# ============================================================

class StrategyManager:
    """统一策略管理器"""

    # 策略注册表: 名称 -> 描述符
    _descriptor_registry: Dict[str, StrategyDescriptor] = {}

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._strategy_instances: Dict[str, Any] = {}          # 名称 -> 实例
        self._lifecycle_states: Dict[str, StrategyLifecycle] = {}  # 名称 -> 生命周期
        self._health_statuses: Dict[str, StrategyHealth] = {}     # 名称 -> 健康状态
        self._config_versions: Dict[str, List[StrategyConfigSnapshot]] = {}  # 名称 -> 版本历史
        self._max_versions = 20                                 # 每策略最多保留版本数
        self._start_order: List[str] = []                       # 启动顺序
        self._running = False
        self._lock = asyncio.Lock()
        self._batch_lock = asyncio.Lock()                       # 批量操作安全锁
        self._batch_operations: Dict[str, bool] = {}            # 批量操作进行中标记

        # 运行模式
        mgr_cfg = self.config.get("strategy_manager", {})
        self._run_mode = RunMode(mgr_cfg.get("default_run_mode", "normal"))
        self._mode_frozen_strategies: set = set()  # 模式切换时被冻结的策略

        # 策略状态追踪
        self._consecutive_errors: Dict[str, int] = {}     # 连续错误计数
        self._last_restart_time: Dict[str, datetime] = {}  # 上次重启时间
        self._degraded_count: Dict[str, int] = {}          # 进入DEGRADED次数

        # 参数变更验证 + 自动回滚守卫（P2-12 策略工程化）
        pv_cfg = self.config.get("strategy_manager", {}).get("parameter_validation", {})
        self._param_guard = ParameterRollbackGuard(
            validation_window_seconds=pv_cfg.get("validation_window_seconds", 1800.0),
            pnl_drop_threshold=pv_cfg.get("pnl_drop_threshold", 0.10),
            win_rate_drop_threshold=pv_cfg.get("win_rate_drop_threshold", 0.20),
            min_trades=pv_cfg.get("min_trades", 5),
        )

        # 策略加载器（动态加载，替代硬编码）
        self._loader = None               # StrategyLoader
        self._loader_initialized = False

        # 外部服务引用（由 scheduler 注入）
        self._coordinator = None          # StrategyCoordinator
        self._adaptive_controller = None  # AdaptiveController
        self._stop_loss_manager = None    # StopLossManager
        self._order_executor = None       # OrderExecutor
        self._okx_client = None           # OKXClient
        self._market_regime_engine = None  # MarketRegimeEngine
        self._intelligent_agent = None    # IntelligentTradingAgent（退出/暂停状态）
        self._mini_backtester = None      # MiniBacktester（滚动回测健康评估）
        self._stress_test_engine = None   # StressTestEngine（压测门禁）
        self._account_manager = None      # AccountManager（压测门禁权益来源）

        # 压测门禁配置（P2-12 策略工程化）
        stress_cfg = self.config.get("strategy_manager", {}).get("stress_gate", {})
        self._stress_gate_enabled = stress_cfg.get("enabled", True)
        self._stress_gate_min_score = stress_cfg.get("min_score", 0.6)
        self._stress_gate_block_ratings = stress_cfg.get("block_ratings", ["critical", "poor"])
        self._stress_gate_cache_ttl = stress_cfg.get("cache_ttl_seconds", 300)
        self._stress_gate_cache: Dict[str, Any] = {"ts": 0.0, "result": None}

        # 事件回调
        self._lifecycle_callbacks: Dict[str, List[Callable]] = {
            "on_start": [],
            "on_pause": [],
            "on_resume": [],
            "on_stop": [],
            "on_error": [],
            "on_config_change": [],
            "on_health_change": [],
        }

        # 初始化注册表
        self._init_descriptor_registry()
        logger.info(f"StrategyManager initialized: {len(self._descriptor_registry)} strategies registered")

    # ============================================================
    # 策略描述符注册
    # ============================================================

    def _init_descriptor_registry(self):
        """初始化策略描述符注册表（优先从 Loader 自动发现，失败时使用硬编码兜底）"""
        # 尝试从 StrategyLoader 自动发现
        try:
            from core.strategy_loader import StrategyDiscovery, StrategyDiscoveryInfo
            discoveries = StrategyDiscovery.discover_from_package("strategies")
            if discoveries:
                for info in discoveries:
                    self._descriptor_registry[info.name] = StrategyDescriptor(
                        name=info.name,
                        display_name=info.display_name or info.name,
                        category=info.category,
                        description=info.description,
                        class_path=info.class_path,
                        dependencies=info.dependencies,
                        enabled_by_default=info.enabled_by_default,
                        config_schema_keys=info.config_schema_keys,
                    )
                logger.info(f"Descriptor registry populated from auto-discovery: "
                           f"{list(self._descriptor_registry.keys())}")
                return
        except Exception as e:
            logger.debug(f"Auto-discovery for descriptors failed, using fallback: {e}")

        # 兜底：从已知映射创建
        from core.strategy_loader import StrategyDiscovery
        known = StrategyDiscovery.KNOWN_STRATEGIES
        for name, info in known.items():
            self._descriptor_registry[name] = StrategyDescriptor(
                name=name,
                display_name=info.get("display_name", name),
                category=info.get("category", "contract"),
                description=info.get("description", ""),
                class_path=info.get("class_path", ""),
                dependencies=info.get("dependencies", []),
                enabled_by_default=info.get("enabled_by_default", True),
                config_schema_keys=info.get("config_schema_keys", []),
            )
        logger.info(f"Descriptor registry populated from known mapping: "
                   f"{list(self._descriptor_registry.keys())}")

    def get_descriptor(self, name: str) -> Optional[StrategyDescriptor]:
        """获取策略描述符"""
        return self._descriptor_registry.get(name)

    def get_all_descriptors(self) -> Dict[str, StrategyDescriptor]:
        """获取所有策略描述符"""
        return dict(self._descriptor_registry)

    def get_enabled_strategy_names(self) -> List[str]:
        """获取当前启用的策略名称列表"""
        strategies_cfg = self.config.get("strategies", {})
        return [
            name for name, desc in self._descriptor_registry.items()
            if strategies_cfg.get(name, {}).get("enabled", desc.enabled_by_default)
        ]

    # ============================================================
    # 外部服务注入
    # ============================================================

    def inject_services(self, **services):
        """注入外部服务引用"""
        self._coordinator = services.get("coordinator", self._coordinator)
        self._adaptive_controller = services.get("adaptive_controller", self._adaptive_controller)
        self._stop_loss_manager = services.get("stop_loss_manager", self._stop_loss_manager)
        self._order_executor = services.get("order_executor", self._order_executor)
        self._okx_client = services.get("okx_client", self._okx_client)
        self._market_regime_engine = services.get("market_regime_engine", self._market_regime_engine)
        self._intelligent_agent = services.get("intelligent_agent", self._intelligent_agent)
        self._mini_backtester = services.get("mini_backtester", self._mini_backtester)
        self._stress_test_engine = services.get("stress_test_engine", self._stress_test_engine)
        self._conditional_order_manager = services.get("conditional_order_manager", getattr(self, "_conditional_order_manager", None))
        self._account_manager = services.get("account_manager", self._account_manager)
        logger.debug(f"StrategyManager: services injected (coordinator={self._coordinator is not None}, "
                     f"adaptive={self._adaptive_controller is not None}, "
                     f"stoploss={self._stop_loss_manager is not None})")

    # ============================================================
    # 上线前门禁（P2-12 策略工程化）
    # ============================================================

    def check_prelaunch_gate(self, name: str) -> Dict[str, Any]:
        """策略上线/恢复前的三重门禁检查。

        门禁顺序（任一拦截即拒绝，不再继续）：
        1. 永久退出检查 — 已退出策略必须人工 reinstate，禁止自动恢复
        2. 自动暂停检查 — 处于冷却期内的策略禁止启动
        3. 迷你回测健康检查 — 滚动回测判定 CRITICAL/FROZEN 的策略禁止启动

        返回：
            {"allowed": bool, "reason": str, "checks": {...}}
        """
        result: Dict[str, Any] = {"allowed": True, "reason": "", "checks": {}}

        # 门禁1：永久退出
        if self._intelligent_agent is not None:
            try:
                if hasattr(self._intelligent_agent, "is_strategy_exited") and \
                        self._intelligent_agent.is_strategy_exited(name):
                    result["allowed"] = False
                    result["reason"] = f"策略 '{name}' 已永久退出，需人工 reinstate_strategy() 确认后恢复"
                    result["checks"]["exited"] = True
                    self._audit_deny(name, "exited")
                    return result
            except Exception as e:
                logger.debug(f"Prelaunch gate: exited check error for {name}: {e}")
            result["checks"]["exited"] = False

        # 门禁2：自动暂停（冷却期）
        if self._intelligent_agent is not None:
            try:
                if hasattr(self._intelligent_agent, "is_strategy_paused") and \
                        self._intelligent_agent.is_strategy_paused(name):
                    result["allowed"] = False
                    result["reason"] = f"策略 '{name}' 处于自动暂停冷却期"
                    result["checks"]["paused"] = True
                    self._audit_deny(name, "paused")
                    return result
            except Exception as e:
                logger.debug(f"Prelaunch gate: paused check error for {name}: {e}")
            result["checks"]["paused"] = False

        # 门禁3：迷你回测健康
        if self._mini_backtester is not None:
            try:
                backtest_result = self._mini_backtester.get_backtest_result(name)
                if backtest_result is not None:
                    health = getattr(backtest_result, "health", None)
                    health_val = getattr(health, "value", str(health))
                    result["checks"]["backtest_health"] = health_val
                    if health_val in ("critical", "frozen"):
                        result["allowed"] = False
                        result["reason"] = (
                            f"策略 '{name}' 迷你回测健康状态为 {health_val}: "
                            f"{getattr(backtest_result, 'freeze_reason', '')}"
                        )
                        self._audit_deny(name, f"backtest_{health_val}")
                        return result
            except Exception as e:
                logger.debug(f"Prelaunch gate: backtest check error for {name}: {e}")

        # 门禁4：压力测试（账户级极端行情回撤评估，带缓存）
        stress_check = self._run_stress_gate()
        if stress_check["blocked"]:
            result["allowed"] = False
            result["reason"] = f"压力测试门禁未通过: {stress_check['reason']}"
            result["checks"]["stress_test"] = stress_check
            self._audit_deny(name, f"stress_{stress_check.get('rating', 'unknown')}")
            return result
        result["checks"]["stress_test"] = stress_check

        if result["allowed"]:
            get_strategy_audit_logger().log(
                "prelaunch_allow", name,
                reason=f"门禁通过 {result.get('checks', {})}",
            )
        return result

    def _run_stress_gate(self) -> Dict[str, Any]:
        """执行压力测试门禁（账户级极端行情回撤评估）。

        设计原则：
        - 压测失败（poor/critical 或评分低于阈值）表示当前账户杠杆/持仓暴露在
          极端行情下回撤超标，应阻止新增开仓（新策略上线/恢复）。
        - 数据获取失败（网络/API 异常）时 fail-open 跳过，不阻塞 —— 与熔断器
          一致，只针对真实风险事件，不针对网络问题。
        - 结果缓存 TTL 内复用，避免 start_all 时每个策略各跑一次全量压测。

        返回 {"blocked": bool, "reason": str, "rating": str, "score": float, "skipped": bool}
        """
        result: Dict[str, Any] = {
            "blocked": False, "reason": "", "rating": "skipped",
            "score": None, "skipped": True,
        }

        if not self._stress_gate_enabled:
            result["reason"] = "压测门禁未启用"
            return result

        if self._stress_test_engine is None:
            result["reason"] = "压力测试引擎未注入"
            return result

        # 缓存命中（TTL 内复用）
        import time as _time
        cache = self._stress_gate_cache
        if cache["result"] is not None and (_time.time() - cache["ts"]) < self._stress_gate_cache_ttl:
            return cache["result"]

        # 获取权益 + 持仓（任一失败则 fail-open 跳过）
        initial_capital = self._get_equity_for_stress_test()
        positions = self._get_positions_for_stress_test()
        if initial_capital <= 0 or positions is None:
            result["reason"] = "无法获取账户权益/持仓，压测门禁跳过（fail-open）"
            logger.warning(f"StrategyManager: stress gate skipped: capital={initial_capital}, positions={positions}")
            return result

        try:
            report = self._stress_test_engine.run_all_tests(initial_capital, positions)
        except Exception as e:
            result["reason"] = f"压力测试执行异常，跳过: {e}"
            logger.error(f"StrategyManager: stress test execution error: {e}")
            return result

        summary = report.get("summary", {})
        score = summary.get("overall_score", 0.0)
        rating = summary.get("overall_rating", "unknown")
        failed = summary.get("failed", 0)
        max_dd = summary.get("max_drawdown_all_scenarios", 0.0)

        result["skipped"] = False
        result["score"] = score
        result["rating"] = rating
        result["failed_scenarios"] = failed
        result["max_drawdown"] = max_dd
        result["scenarios"] = report.get("results", [])
        result["recommendations"] = report.get("recommendations", [])

        blocked = (rating in self._stress_gate_block_ratings) or (score < self._stress_gate_min_score)
        if blocked:
            result["blocked"] = True
            result["reason"] = (
                f"评级={rating}, 评分={score:.2f}, 失败场景={failed}, "
                f"最大回撤={max_dd:.1%}"
            )
            logger.warning(f"StrategyManager: stress gate BLOCKED ({result['reason']})")
        else:
            result["reason"] = f"评级={rating}, 评分={score:.2f}, 最大回撤={max_dd:.1%}"

        cache["ts"] = _time.time()
        cache["result"] = result
        self._persist_stress_gate_result(result, cache["ts"])
        return result

    def _persist_stress_gate_result(self, result: Dict[str, Any], ts: float) -> None:
        """将最新压测门禁结果落盘，供独立 dashboard 进程跨进程读取。"""
        try:
            os.makedirs("./data/strategy_manager", exist_ok=True)
            with open("./data/strategy_manager/stress_gate_result.json", "w", encoding="utf-8") as f:
                json.dump({"ts": ts, "result": result}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"StrategyManager: persist stress gate result error: {e}")

    def _get_equity_for_stress_test(self) -> float:
        """获取当前账户权益（压测初始资金）。"""
        if self._account_manager is not None:
            try:
                if hasattr(self._account_manager, "get_total_equity"):
                    equity = self._account_manager.get_total_equity()
                    if equity and equity > 0:
                        return float(equity)
            except Exception as e:
                logger.debug(f"Stress gate: account_manager equity error: {e}")
        # 回退到 config total_capital
        try:
            tc = self.config.get("total_capital") or self.config.get("trading", {}).get("total_capital")
            if tc:
                return float(tc)
        except (ValueError, TypeError):
            pass
        return 0.0

    def _get_positions_for_stress_test(self):
        """获取当前持仓并转换为压测所需格式（失败返回 None 触发 fail-open）。"""
        if self._okx_client is None:
            return None
        try:
            raw = self._okx_client.get_positions()
        except Exception as e:
            logger.debug(f"Stress gate: get_positions error: {e}")
            return None

        if raw is None:
            return None

        positions = []
        try:
            for p in raw:
                inst_id = p.get("instId", "")
                pos = abs(float(p.get("pos", 0) or 0))
                if pos <= 0 or not inst_id:
                    continue
                side = "long" if (p.get("posSide") or "").lower() in ("long", "net") else "short"
                entry_price = float(p.get("avgPx", 0) or 0)
                leverage = int(float(p.get("lever", 1) or 1)) or 1
                notional = pos * entry_price
                margin = notional / leverage if leverage > 0 else notional
                positions.append({
                    "symbol": inst_id,
                    "side": side,
                    "quantity": pos,
                    "entry_price": entry_price,
                    "leverage": leverage,
                    "margin": round(margin, 2),
                })
        except Exception as e:
            logger.debug(f"Stress gate: position transform error: {e}")
            return None

        return positions

    def _audit_deny(self, name: str, deny_kind: str) -> None:
        """记录门禁拦截审计。"""
        try:
            get_strategy_audit_logger().log(
                "prelaunch_deny", name,
                reason=f"上线门禁拦截: {deny_kind}",
            )
        except Exception as e:
            logger.debug(f"StrategyManager: audit deny log error: {e}")

    # ============================================================
    # 策略加载器集成 — 动态创建策略实例
    # ============================================================

    def set_loader(self, loader) -> None:
        """注入 StrategyLoader（用于动态创建策略实例）"""
        self._loader = loader
        logger.info("StrategyManager: loader injected")

    def load_and_register_all(self, **dependencies) -> "LoadReport":
        """
        使用 StrategyLoader 动态加载并注册所有策略。

        替代 scheduler.py 中硬编码的:
          - 6行 from strategies.xxx import XxxStrategy
          - 6行 XxxStrategy(config, okx_client, redis_cache)
          - 6行 .set_adaptive_controller()
          - 6行 self.strategy_manager.register_instance()

        Args:
            **dependencies: 容器中的依赖项
                - okx_client, redis_cache (必需)
                - adaptive_controller, stop_loss_manager, coordinator (可选)

        Returns:
            LoadReport: 加载结果报告
        """
        from core.strategy_loader import get_strategy_loader, LoadReport
        from core.strategy_loader import LoadStatus

        # 获取或创建加载器
        if self._loader is None:
            self._loader = get_strategy_loader(self.config)
        self._loader_initialized = True

        # 确保服务已注入到加载器依赖中
        all_deps = {
            "okx_client": self._okx_client,
            "redis_cache": dependencies.get("redis_cache"),
            "adaptive_controller": self._adaptive_controller,
            "stop_loss_manager": self._stop_loss_manager,
            "coordinator": self._coordinator,
            "market_regime_engine": self._market_regime_engine,
            "conditional_order_manager": getattr(self, "_conditional_order_manager", None),
        }
        all_deps.update(dependencies)

        # 执行加载
        instances, report = self._loader.load_all(**all_deps)

        # 将加载成功的实例注册到管理器
        registered = 0
        for strategy_name, instance in instances.items():
            if instance is not None:
                self.register_instance(strategy_name, instance)
                registered += 1

        logger.info(f"StrategyManager: load_and_register_all complete — "
                    f"{registered} instances registered, "
                    f"{report.loaded} loaded, {report.skipped} skipped, "
                    f"{report.failed} failed")

        return report

    def load_deferred(self, strategy_name: str, **dependencies) -> bool:
        """
        延迟加载单个策略（系统启动后按需加载）

        用于 lazy_load 模式，在系统运行后按需加载非关键策略。
        """
        if not self._loader:
            logger.error("StrategyManager.load_deferred: no loader available")
            return False

        all_deps = {
            "okx_client": self._okx_client,
            "adaptive_controller": self._adaptive_controller,
            "stop_loss_manager": self._stop_loss_manager,
            "coordinator": self._coordinator,
            "conditional_order_manager": getattr(self, "_conditional_order_manager", None),
        }
        all_deps.update(dependencies)

        instance, result = self._loader.load_single(strategy_name, **all_deps)
        if instance is not None:
            self.register_instance(strategy_name, instance)
            logger.info(f"Deferred strategy '{strategy_name}' loaded and registered")
            return True

        logger.error(f"Deferred load failed for '{strategy_name}': {result.error}")
        return False

    def get_loader_stats(self) -> Dict[str, Any]:
        """获取加载器统计信息"""
        if self._loader:
            return self._loader.get_load_stats()
        return {"error": "Loader not initialized"}

    # ============================================================
    # 策略注册
    # ============================================================

    def register_instance(self, name: str, strategy_instance: Any) -> bool:
        """注册策略实例"""
        if name not in self._descriptor_registry:
            logger.warning(f"Strategy '{name}' not in descriptor registry, auto-registering")
            self._descriptor_registry[name] = StrategyDescriptor(
                name=name, display_name=name, category="contract"
            )

        self._strategy_instances[name] = strategy_instance
        self._lifecycle_states[name] = StrategyLifecycle.REGISTERED
        self._health_statuses[name] = StrategyHealth.UNKNOWN

        # 注入依赖服务
        self._inject_to_instance(name, strategy_instance)

        # 注册到协调器
        if self._coordinator:
            try:
                self._coordinator.register_strategy(name, strategy_instance)
            except Exception as e:
                logger.debug(f"Coordinator registration for {name}: {e}")

        self._capture_config_snapshot(name, "initial_registration")
        logger.info(f"Strategy '{name}' registered")
        return True

    def _inject_to_instance(self, name: str, instance: Any):
        """向策略实例注入依赖服务"""
        if self._adaptive_controller and hasattr(instance, "set_adaptive_controller"):
            instance.set_adaptive_controller(self._adaptive_controller)
        if self._stop_loss_manager and hasattr(instance, "set_stop_loss_manager"):
            instance.set_stop_loss_manager(self._stop_loss_manager)
        if self._coordinator and hasattr(instance, "set_coordinator"):
            instance.set_coordinator(self._coordinator)
        if hasattr(self, "_conditional_order_manager") and self._conditional_order_manager and hasattr(instance, "set_conditional_order_manager"):
            instance.set_conditional_order_manager(self._conditional_order_manager)

    def get_instance(self, name: str) -> Optional[Any]:
        """获取策略实例"""
        return self._strategy_instances.get(name)

    def get_all_instances(self) -> Dict[str, Any]:
        """获取所有策略实例"""
        return dict(self._strategy_instances)

    # ============================================================
    # 生命周期管理
    # ============================================================

    async def init_strategy(self, name: str) -> bool:
        """初始化策略"""
        instance = self._strategy_instances.get(name)
        if not instance:
            logger.error(f"Strategy '{name}' not registered")
            return False

        async with self._lock:
            self._lifecycle_states[name] = StrategyLifecycle.INITIALIZING
            try:
                if hasattr(instance, "init_state_persistence"):
                    instance.init_state_persistence(name)
                if hasattr(instance, "load_state_async"):
                    await instance.load_state_async()
                self._lifecycle_states[name] = StrategyLifecycle.PREPARED
                logger.info(f"Strategy '{name}' initialized -> PREPARED")
                return True
            except Exception as e:
                self._lifecycle_states[name] = StrategyLifecycle.ERROR
                logger.error(f"Strategy '{name}' init failed: {e}")
                self._trigger_callbacks("on_error", name, str(e))
                return False

    async def start_strategy(self, name: str) -> bool:
        """启动策略"""
        instance = self._strategy_instances.get(name)
        if not instance:
            logger.error(f"Strategy '{name}' not registered")
            return False

        desc = self._descriptor_registry.get(name)
        if desc and not self.config.get("strategies", {}).get(name, {}).get("enabled", desc.enabled_by_default):
            logger.info(f"Strategy '{name}' disabled in config, skipping start")
            return False

        async with self._lock:
            current = self._lifecycle_states.get(name, StrategyLifecycle.UNREGISTERED)
            if current == StrategyLifecycle.RUNNING:
                return True  # 已经在运行

            # P2-12: 上线前门禁（退出/暂停/迷你回测健康）
            gate = self.check_prelaunch_gate(name)
            if not gate["allowed"]:
                logger.warning(f"P2-12: Strategy '{name}' start blocked by prelaunch gate: {gate['reason']}")
                return False

            self._lifecycle_states[name] = StrategyLifecycle.STARTING
            try:
                # 检查依赖
                if desc and desc.dependencies:
                    for dep in desc.dependencies:
                        dep_state = self._lifecycle_states.get(dep)
                        if dep not in ("market_regime",) and dep_state != StrategyLifecycle.RUNNING:
                            logger.warning(f"Strategy '{name}' depends on '{dep}' which is not RUNNING ({dep_state})")

                if hasattr(instance, "start"):
                    result = instance.start()
                    if asyncio.iscoroutine(result):
                        await result

                self._lifecycle_states[name] = StrategyLifecycle.RUNNING
                self._health_statuses[name] = StrategyHealth.HEALTHY
                self._trigger_callbacks("on_start", name)
                logger.info(f"Strategy '{name}' started -> RUNNING")
                return True
            except Exception as e:
                self._lifecycle_states[name] = StrategyLifecycle.ERROR
                self._health_statuses[name] = StrategyHealth.CRITICAL
                logger.error(f"Strategy '{name}' start failed: {e}")
                self._trigger_callbacks("on_error", name, str(e))
                return False

    async def pause_strategy(self, name: str) -> bool:
        """暂停策略"""
        instance = self._strategy_instances.get(name)
        if not instance:
            logger.error(f"Strategy '{name}' not registered")
            return False

        async with self._lock:
            self._lifecycle_states[name] = StrategyLifecycle.PAUSING
            try:
                if hasattr(instance, "pause"):
                    result = instance.pause()
                    if asyncio.iscoroutine(result):
                        await result
                elif hasattr(instance, "pause_opening"):
                    instance.pause_opening()
                # 通知协调器
                if self._coordinator:
                    from services.strategy_coordinator import StrategyState
                    self._coordinator.set_strategy_state(name, StrategyState.PAUSED)

                self._lifecycle_states[name] = StrategyLifecycle.PAUSED
                self._trigger_callbacks("on_pause", name)
                logger.info(f"Strategy '{name}' paused")
                return True
            except Exception as e:
                self._lifecycle_states[name] = StrategyLifecycle.ERROR
                logger.error(f"Strategy '{name}' pause failed: {e}")
                return False

    async def resume_strategy(self, name: str) -> bool:
        """恢复策略"""
        instance = self._strategy_instances.get(name)
        if not instance:
            logger.error(f"Strategy '{name}' not registered")
            return False

        async with self._lock:
            # P2-12: 上线前门禁（退出/暂停/迷你回测健康）
            gate = self.check_prelaunch_gate(name)
            if not gate["allowed"]:
                logger.warning(f"P2-12: Strategy '{name}' resume blocked by prelaunch gate: {gate['reason']}")
                return False

            self._lifecycle_states[name] = StrategyLifecycle.RESUMING
            try:
                if hasattr(instance, "resume"):
                    result = instance.resume()
                    if asyncio.iscoroutine(result):
                        await result
                elif hasattr(instance, "resume_opening"):
                    instance.resume_opening()
                # 通知协调器
                if self._coordinator:
                    from services.strategy_coordinator import StrategyState
                    self._coordinator.set_strategy_state(name, StrategyState.RUNNING)

                self._lifecycle_states[name] = StrategyLifecycle.RUNNING
                self._trigger_callbacks("on_resume", name)
                logger.info(f"Strategy '{name}' resumed -> RUNNING")
                return True
            except Exception as e:
                self._lifecycle_states[name] = StrategyLifecycle.ERROR
                logger.error(f"Strategy '{name}' resume failed: {e}")
                return False

    async def stop_strategy(self, name: str, drain_positions: bool = False) -> bool:
        """停止策略

        Args:
            name: 策略名称
            drain_positions: True=先排空仓位再停止(DRAINING)，False=立即停止
        """
        instance = self._strategy_instances.get(name)
        if not instance:
            logger.error(f"Strategy '{name}' not registered")
            return False

        async with self._lock:
            if drain_positions:
                self._lifecycle_states[name] = StrategyLifecycle.DRAINING
                await self._drain_positions(name)

            self._lifecycle_states[name] = StrategyLifecycle.STOPPING
            try:
                if hasattr(instance, "stop"):
                    result = instance.stop()
                    if asyncio.iscoroutine(result):
                        await result
                # 通知协调器
                if self._coordinator:
                    from services.strategy_coordinator import StrategyState
                    self._coordinator.set_strategy_state(name, StrategyState.STOPPED)

                self._lifecycle_states[name] = StrategyLifecycle.STOPPED
                self._trigger_callbacks("on_stop", name)
                logger.info(f"Strategy '{name}' stopped")
                return True
            except Exception as e:
                self._lifecycle_states[name] = StrategyLifecycle.ERROR
                logger.error(f"Strategy '{name}' stop failed: {e}")
                return False

    async def restart_strategy(self, name: str) -> bool:
        """重启策略"""
        async with self._lock:
            self._lifecycle_states[name] = StrategyLifecycle.RESTARTING
            logger.info(f"Strategy '{name}' restarting...")
            await self.stop_strategy(name)
            await asyncio.sleep(1)
            await self.init_strategy(name)
            return await self.start_strategy(name)

    def _validate_transition(self, name: str, from_state: StrategyLifecycle,
                             to_state: StrategyLifecycle) -> bool:
        """验证生命周期转换是否合法"""
        allowed = _ALLOWED_TRANSITIONS.get(from_state, set())
        if to_state not in allowed:
            logger.warning(f"Strategy '{name}': invalid transition {from_state.value} -> {to_state.value}, "
                          f"allowed: {[s.value for s in allowed]}")
            return False
        return True

    def mark_degraded(self, name: str, reason: str) -> bool:
        """将策略标记为降级运行（部分功能受限）

        适用于策略仍能运行但某些功能异常的场景（如数据源失效但仍有缓存可用）
        """
        current = self._lifecycle_states.get(name, StrategyLifecycle.UNREGISTERED)
        if not self._validate_transition(name, current, StrategyLifecycle.DEGRADED):
            return False

        self._lifecycle_states[name] = StrategyLifecycle.DEGRADED
        self._health_statuses[name] = StrategyHealth.WARNING
        self._degraded_count[name] = self._degraded_count.get(name, 0) + 1
        self._trigger_callbacks("on_health_change", name, {"reason": reason, "state": "degraded"})
        logger.warning(f"Strategy '{name}' marked DEGRADED: {reason} (total: {self._degraded_count[name]})")
        return True

    async def recover_degraded(self, name: str) -> bool:
        """尝试恢复降级策略到正常运行"""
        current = self._lifecycle_states.get(name)
        if current != StrategyLifecycle.DEGRADED:
            return False

        instance = self._strategy_instances.get(name)
        if not instance:
            return False

        try:
            # 尝试恢复策略
            if hasattr(instance, "recover_from_degraded"):
                result = instance.recover_from_degraded()
                if asyncio.iscoroutine(result):
                    await result

            self._lifecycle_states[name] = StrategyLifecycle.RUNNING
            self._health_statuses[name] = StrategyHealth.HEALTHY
            logger.info(f"Strategy '{name}' recovered from DEGRADED -> RUNNING")
            return True
        except Exception as e:
            logger.error(f"Strategy '{name}' degraded recovery failed: {e}")
            return False

    async def _auto_recover_degraded_loop(self):
        """自动恢复降级策略的后台循环"""
        mgr_cfg = self.config.get("strategy_manager", {})
        interval = mgr_cfg.get("health_check_interval_seconds", 60) * 2
        while True:
            try:
                await asyncio.sleep(interval)
                for name, state in list(self._lifecycle_states.items()):
                    if state == StrategyLifecycle.DEGRADED:
                        degraded_count = self._degraded_count.get(name, 0)
                        if degraded_count > 5:
                            logger.warning(f"Strategy '{name}' degraded {degraded_count} times, stopping auto-recovery")
                            continue
                        # P2-12: 检查策略是否已被永久退出
                        if hasattr(self, '_intelligent_agent') and self._intelligent_agent:
                            if hasattr(self._intelligent_agent, 'is_strategy_exited') and \
                               self._intelligent_agent.is_strategy_exited(name):
                                logger.info(f"P2-12: Skipping auto-recovery for '{name}' — strategy is permanently exited")
                                continue
                        logger.info(f"Auto-recovering degraded strategy '{name}'...")
                        await self.recover_degraded(name)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-recover-degraded loop error: {e}")

    # ============================================================
    # 运行模式管控
    # ============================================================

    def get_run_mode(self) -> RunMode:
        """获取当前运行模式"""
        return self._run_mode

    async def set_run_mode(self, mode: RunMode, reason: str = "") -> Dict[str, Any]:
        """切换运行模式

        模式切换规则：
        - NORMAL: 恢复所有被模式冻结的策略
        - CONSERVATIVE: 暂停套利、现货马丁等高风险策略，降低合约杠杆
        - AGGRESSIVE: 放宽部分风控限制，最大化资金利用率
        - EMERGENCY: 暂停全部策略，仅允许平仓
        """
        if mode == self._run_mode:
            return {"mode": mode.value, "changed": False, "message": "Already in this mode"}

        old_mode = self._run_mode
        self._run_mode = mode
        logger.warning(f"RunMode changed: {old_mode.value} -> {mode.value} (reason: {reason})")

        changes = {"mode": mode.value, "old_mode": old_mode.value, "actions": []}

        async with self._lock:
            if mode == RunMode.EMERGENCY:
                # 全部暂停
                for name in list(self._strategy_instances.keys()):
                    state = self._lifecycle_states.get(name)
                    if state == StrategyLifecycle.RUNNING:
                        await self.pause_strategy(name)
                        self._mode_frozen_strategies.add(name)
                        changes["actions"].append(f"paused:{name}")

            elif mode == RunMode.CONSERVATIVE:
                # 暂停高风险策略：arbitrage, spot_martingale, scalping
                conservative_pause = {"arbitrage", "spot_martingale"}
                conservative_aggressive = {"scalping"}
                for name in conservative_pause:
                    state = self._lifecycle_states.get(name)
                    if state == StrategyLifecycle.RUNNING:
                        await self.pause_strategy(name)
                        self._mode_frozen_strategies.add(name)
                        changes["actions"].append(f"paused:{name}")
                for name in conservative_aggressive:
                    state = self._lifecycle_states.get(name)
                    if state == StrategyLifecycle.RUNNING:
                        # 不暂停但标记为降级，降低杠杆
                        self.mark_degraded(name, f"mode_switch:{mode.value}")
                        changes["actions"].append(f"degraded:{name}")

            elif mode == RunMode.NORMAL:
                # 恢复之前被模式冻结的策略
                for name in list(self._mode_frozen_strategies):
                    state = self._lifecycle_states.get(name)
                    if state in (StrategyLifecycle.PAUSED, StrategyLifecycle.DEGRADED):
                        if await self.resume_strategy(name):
                            changes["actions"].append(f"resumed:{name}")
                        # 如果仍在 DEGRADED，尝试恢复
                        if self._lifecycle_states.get(name) == StrategyLifecycle.DEGRADED:
                            if await self.recover_degraded(name):
                                changes["actions"].append(f"recovered:{name}")
                self._mode_frozen_strategies.clear()

            elif mode == RunMode.AGGRESSIVE:
                # 恢复所有暂停策略，降低质量阈值
                for name in list(self._mode_frozen_strategies):
                    state = self._lifecycle_states.get(name)
                    if state == StrategyLifecycle.PAUSED:
                        if await self.resume_strategy(name):
                            changes["actions"].append(f"resumed:{name}")
                self._mode_frozen_strategies.clear()

        return changes

    # ============================================================
    # 批量操作安全锁
    # ============================================================

    async def _acquire_batch_lock(self, operation: str) -> bool:
        """获取批量操作锁（防并发冲突）"""
        async with self._batch_lock:
            if any(self._batch_operations.values()):
                logger.warning(f"Batch operation '{operation}' blocked: another batch operation in progress")
                return False
            self._batch_operations[operation] = True
            return True

    async def _release_batch_lock(self, operation: str):
        """释放批量操作锁"""
        async with self._batch_lock:
            self._batch_operations.pop(operation, None)

    async def start_all(self, ordered: bool = True) -> Dict[str, bool]:
        """启动所有已启用的策略

        Args:
            ordered: True=按依赖顺序启动
        """
        if not await self._acquire_batch_lock("start_all"):
            return {"_error": False, "_blocked": True}

        try:
            results = {}
            names = self.get_enabled_strategy_names()

            if ordered:
                # 按依赖排序：无依赖的先启动
                sorted_names = self._resolve_start_order(names)
            else:
                sorted_names = names

            for name in sorted_names:
                if name not in self._strategy_instances:
                    results[name] = False
                    continue

                await self.init_strategy(name)
                results[name] = await self.start_strategy(name)
                if results[name]:
                    self._start_order.append(name)

            running = sum(1 for v in results.values() if v)
            logger.info(f"StrategyManager: started {running}/{len(results)} strategies")
            return results
        finally:
            await self._release_batch_lock("start_all")

    async def stop_all(self, drain: bool = False) -> Dict[str, bool]:
        """停止所有策略"""
        if not await self._acquire_batch_lock("stop_all"):
            return {"_error": False, "_blocked": True}

        try:
            results = {}
            # 逆序停止（后启动的先停）
            for name in reversed(self._start_order):
                if name in self._strategy_instances:
                    results[name] = await self.stop_strategy(name, drain_positions=drain)
            self._start_order.clear()
            return results
        finally:
            await self._release_batch_lock("stop_all")

    async def pause_all(self) -> Dict[str, bool]:
        """暂停所有策略"""
        if not await self._acquire_batch_lock("pause_all"):
            return {"_error": False, "_blocked": True}

        try:
            results = {}
            for name in self._start_order:
                if self._lifecycle_states.get(name) == StrategyLifecycle.RUNNING:
                    results[name] = await self.pause_strategy(name)
            return results
        finally:
            await self._release_batch_lock("pause_all")

    async def resume_all(self) -> Dict[str, bool]:
        """恢复所有策略"""
        if not await self._acquire_batch_lock("resume_all"):
            return {"_error": False, "_blocked": True}

        try:
            results = {}
            for name in self._start_order:
                if self._lifecycle_states.get(name) == StrategyLifecycle.PAUSED:
                    results[name] = await self.resume_strategy(name)
            return results
        finally:
            await self._release_batch_lock("resume_all")

    def _resolve_start_order(self, names: List[str]) -> List[str]:
        """基于依赖关系解析启动顺序（拓扑排序）"""
        sorted_names = []
        visited = set()
        temp_visited = set()

        def visit(n: str):
            if n in temp_visited:
                return  # 循环依赖，跳过
            if n in visited:
                return
            temp_visited.add(n)
            desc = self._descriptor_registry.get(n)
            if desc:
                for dep in desc.dependencies:
                    if dep in names:
                        visit(dep)
            temp_visited.discard(n)
            visited.add(n)
            sorted_names.append(n)

        for name in names:
            if name not in visited:
                visit(name)

        return sorted_names

    async def _drain_positions(self, name: str):
        """排空策略持仓"""
        instance = self._strategy_instances.get(name)
        if not instance:
            return
        try:
            if hasattr(instance, "close_all_positions"):
                result = instance.close_all_positions()
                if asyncio.iscoroutine(result):
                    await result
            logger.info(f"Strategy '{name}' positions drained")
        except Exception as e:
            logger.error(f"Strategy '{name}' drain positions error: {e}")

    # ============================================================
    # 配置管理
    # ============================================================

    def get_strategy_config(self, name: str) -> Dict[str, Any]:
        """获取策略当前生效配置"""
        return self.config.get("strategies", {}).get(name, {})

    def get_all_strategy_configs(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略配置"""
        return self.config.get("strategies", {})

    async def hot_reload_config(self, name: str, updates: Dict[str, Any],
                                 reason: str = "manual_update",
                                 baseline_metrics: Optional[Dict[str, Any]] = None,
                                 trigger_validation: bool = True) -> bool:
        """运行时热重载策略配置

        不重启策略，直接动态更新运行中策略的参数。参数变更后默认开启 30 分钟
        验证（由 ParameterRollbackGuard 决策是否回滚）。

        Args:
            name: 策略名称
            updates: 要更新的配置键值对
            reason: 变更原因
            baseline_metrics: 变更前基线指标（含 pnl / win_rate），用于验证回滚
            trigger_validation: 是否开启参数验证（回滚动作本身不应再触发验证）
        """
        instance = self._strategy_instances.get(name)
        if not instance:
            logger.error(f"Strategy '{name}' not registered for hot-reload")
            return False

        async with self._lock:
            try:
                old_config = copy.deepcopy(self.get_strategy_config(name))

                # 先捕获快照（记录变更前配置，供回滚使用）
                snapshot = self._capture_config_snapshot(name, reason)

                # 更新内存配置
                if "strategies" not in self.config:
                    self.config["strategies"] = {}
                if name not in self.config["strategies"]:
                    self.config["strategies"][name] = {}
                self.config["strategies"][name].update(updates)

                # 通知策略实例
                if hasattr(instance, "update_config"):
                    await instance.update_config(updates)
                elif hasattr(instance, "apply_config"):
                    instance.apply_config(updates)

                # 标记快照已应用
                if self._config_versions.get(name):
                    self._config_versions[name][-1].applied = True

                self._trigger_callbacks("on_config_change", name, {
                    "old": old_config,
                    "new": self.get_strategy_config(name),
                    "reason": reason,
                })

                logger.info(f"Strategy '{name}' config hot-reloaded: {list(updates.keys())} (reason={reason})")

                # 开启参数变更验证（回滚动作不重复开启）
                if trigger_validation:
                    self._param_guard.begin_validation(
                        name, baseline_metrics=baseline_metrics,
                        reason=reason, new_config=updates,
                        rollback_target_version=snapshot.version,
                    )
                return True
            except Exception as e:
                logger.error(f"Strategy '{name}' hot-reload failed: {e}")
                return False

    def _capture_config_snapshot(self, name: str, reason: str = "") -> StrategyConfigSnapshot:
        """捕获配置快照"""
        if name not in self._config_versions:
            self._config_versions[name] = []

        versions = self._config_versions[name]
        version = (versions[-1].version + 1) if versions else 1

        snapshot = StrategyConfigSnapshot(
            strategy_name=name,
            version=version,
            config=copy.deepcopy(self.get_strategy_config(name)),
            reason=reason,
        )
        versions.append(snapshot)

        # 限制版本数
        if len(versions) > self._max_versions:
            self._config_versions[name] = versions[-self._max_versions:]

        return snapshot

    def get_config_versions(self, name: str) -> List[StrategyConfigSnapshot]:
        """获取策略配置版本历史"""
        return self._config_versions.get(name, [])

    async def rollback_config(self, name: str, target_version: int) -> bool:
        """回滚策略配置到指定版本"""
        versions = self._config_versions.get(name, [])
        target = None
        for v in versions:
            if v.version == target_version:
                target = v
                break

        if not target:
            logger.error(f"Version {target_version} not found for strategy '{name}'")
            return False

        return await self.hot_reload_config(
            name, target.config, reason=f"rollback_to_v{target_version}",
            trigger_validation=False,
        )

    async def process_pending_parameter_validations(
        self,
        current_metrics_by_strategy: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """处理到期的参数变更验证并执行自动回滚（P2-12 策略工程化）。

        遍历 ``ParameterRollbackGuard`` 中所有进行中的验证，对已到期者调用
        ``check`` 产出决策；当 ``should_rollback`` 为真时，自动回滚到变更前
        的配置快照版本。

        Args:
            current_metrics_by_strategy: 策略名 -> 当前指标（含 ``trades`` /
                ``pnl`` / ``win_rate``）。缺省时仅消费到期验证但不判定退化，
                由调用方（scheduler 健康循环）传入实盘指标。

        Returns:
            每个已到期策略的决策字典（含 ``should_rollback``、``reason``、
            ``rollback_applied`` 等），未到期/无验证的策略不会出现。
        """
        results: Dict[str, Any] = {}
        pending = self._param_guard.get_pending()
        if not pending:
            return results

        metrics_map = current_metrics_by_strategy or {}
        for name in list(pending.keys()):
            ctx = pending.get(name, {})
            decision = self._param_guard.check(name, current_metrics=metrics_map.get(name))

            # 未到期 / 已被其他调用消费，跳过
            if decision.reason in ("validation_in_progress", "no_pending_validation"):
                continue

            entry = decision.to_dict()
            entry["rollback_applied"] = False
            results[name] = entry

            if not decision.should_rollback:
                continue

            # 定位回滚目标版本：优先使用变更时记录的变更前快照版本
            target_version = ctx.get("rollback_target_version")
            if target_version is None:
                versions = self._config_versions.get(name, [])
                if len(versions) >= 2:
                    target_version = versions[-2].version

            if target_version is None:
                logger.error(
                    f"ParameterRollbackGuard: '{name}' 需要回滚但未找到目标版本，跳过"
                )
                continue

            ok = await self.rollback_config(name, target_version=int(target_version))
            entry["rollback_applied"] = bool(ok)
            entry["rollback_target_version"] = int(target_version)
            if ok:
                logger.warning(
                    f"StrategyManager: '{name}' 参数验证失败，已自动回滚到 v{target_version} "
                    f"(reason={decision.reason})"
                )
                # P2-12: 回滚审计留痕
                try:
                    get_strategy_audit_logger().log(
                        "rollback", name,
                        reason=decision.reason,
                        target_version=int(target_version),
                    )
                except Exception as e:
                    logger.debug(f"StrategyManager: rollback audit log error: {e}")
            else:
                logger.error(f"StrategyManager: '{name}' 自动回滚失败 (v{target_version})")

        return results

    def persist_config_yaml(self, path: str = "config.yaml") -> bool:
        """将内存配置持久化到 YAML 文件（只更新strategies段，保留其他配置）"""
        try:
            import yaml
            # 读取原始文件
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            # 只更新 strategies 段中的优化参数，不覆盖整个 strategies
            mem_strategies = self.config.get("strategies", {})
            raw_strategies = raw.get("strategies", {})
            for strategy_name, strategy_config in mem_strategies.items():
                if not isinstance(strategy_config, dict):
                    raw_strategies[strategy_name] = strategy_config
                else:
                    raw_strategies.setdefault(strategy_name, {})
                    for key, value in strategy_config.items():
                        raw_strategies[strategy_name][key] = value
            raw["strategies"] = raw_strategies

            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(raw, f, allow_unicode=True, default_flow_style=False)

            logger.info(f"Strategy configs persisted to {path}")
            return True
        except Exception as e:
            logger.error(f"Persist config YAML error: {e}")
            return False

    # ============================================================
    # 健康监控
    # ============================================================

    async def check_health(self, name: str) -> Tuple[StrategyHealth, Dict[str, Any]]:
        """检查单策略健康状态"""
        instance = self._strategy_instances.get(name)
        state = self._lifecycle_states.get(name, StrategyLifecycle.UNREGISTERED)
        details = {
            "strategy": name,
            "lifecycle": state.value,
            "timestamp": datetime.now().isoformat(),
            "issues": [],
        }

        if state == StrategyLifecycle.ERROR:
            details["issues"].append("lifecycle_error")
            details["consecutive_errors"] = self._consecutive_errors.get(name, 0)
            return StrategyHealth.CRITICAL, details

        if state == StrategyLifecycle.DEGRADED:
            details["issues"].append("degraded_state")
            details["degraded_count"] = self._degraded_count.get(name, 0)
            return StrategyHealth.WARNING, details

        if state == StrategyLifecycle.UNREGISTERED:
            return StrategyHealth.UNKNOWN, details

        if state not in (StrategyLifecycle.RUNNING, StrategyLifecycle.PAUSED):
            return StrategyHealth.WARNING, details

        # 检查策略自身健康报告
        if instance and hasattr(instance, "get_health"):
            try:
                health = instance.get_health()
                if health:
                    details["strategy_health"] = health
                    if health.get("status") == "critical":
                        details["issues"].append("strategy_reports_critical")
                        return StrategyHealth.CRITICAL, details
                    elif health.get("status") == "warning":
                        details["issues"].append("strategy_reports_warning")
            except Exception:
                pass

        # 检查心跳（通过 coordinator）
        if self._coordinator:
            try:
                coordinator_state = self._coordinator.get_strategy_state(name)
                if coordinator_state and str(coordinator_state) == "error":
                    details["issues"].append("coordinator_reports_error")
                    return StrategyHealth.CRITICAL, details
            except Exception:
                pass

        if details["issues"]:
            return StrategyHealth.WARNING, details
        return StrategyHealth.HEALTHY, details

    async def check_all_health(self) -> Dict[str, Any]:
        """检查所有策略健康状态"""
        results = {}
        overall = StrategyHealth.HEALTHY

        for name in self._strategy_instances:
            health, details = await self.check_health(name)
            self._health_statuses[name] = health
            results[name] = {"health": health.value, "details": details}

            if health == StrategyHealth.CRITICAL:
                overall = StrategyHealth.CRITICAL
            elif health == StrategyHealth.WARNING and overall != StrategyHealth.CRITICAL:
                overall = StrategyHealth.WARNING

        return {
            "overall_health": overall.value,
            "strategies": results,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "healthy": sum(1 for v in results.values() if v["health"] == "healthy"),
                "warning": sum(1 for v in results.values() if v["health"] == "warning"),
                "critical": sum(1 for v in results.values() if v["health"] == "critical"),
                "unknown": sum(1 for v in results.values() if v["health"] == "unknown"),
                "total": len(results),
            },
        }

    # ============================================================
    # 运行时指标采集
    # ============================================================

    def collect_metrics(self, name: str) -> StrategyMetrics:
        """采集策略运行时指标"""
        instance = self._strategy_instances.get(name)
        metrics = StrategyMetrics()
        if not instance:
            return metrics

        state = self._lifecycle_states.get(name, StrategyLifecycle.UNREGISTERED)
        metrics.health_level = state.value

        # 从实例获取指标
        for attr in ("total_trades", "winning_trades", "losing_trades"):
            if hasattr(instance, attr):
                setattr(metrics, attr, getattr(instance, attr, 0))

        # 从 coordinator 获取信号统计
        if self._coordinator:
            try:
                coordinator_signals = self._coordinator._strategy_signals.get(name, [])
                metrics.signals_generated = len(coordinator_signals)
                if coordinator_signals:
                    metrics.last_signal_time = coordinator_signals[-1].get("timestamp")
            except Exception:
                pass

        return metrics

    def collect_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """采集所有策略运行时指标"""
        return {name: self.collect_metrics(name) for name in self._strategy_instances}

    # ============================================================
    # 事件回调
    # ============================================================

    def on(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self._lifecycle_callbacks:
            self._lifecycle_callbacks[event].append(callback)

    def off(self, event: str, callback: Callable):
        """移除事件回调"""
        if event in self._lifecycle_callbacks and callback in self._lifecycle_callbacks[event]:
            self._lifecycle_callbacks[event].remove(callback)

    def _trigger_callbacks(self, event: str, name: str, data: Any = None):
        """触发事件回调"""
        for cb in self._lifecycle_callbacks.get(event, []):
            try:
                cb(name, data)
            except Exception as e:
                logger.debug(f"Callback error ({event}): {e}")

    # ============================================================
    # 状态查询
    # ============================================================

    def get_lifecycle_state(self, name: str) -> StrategyLifecycle:
        """获取策略生命周期状态"""
        return self._lifecycle_states.get(name, StrategyLifecycle.UNREGISTERED)

    def get_all_lifecycle_states(self) -> Dict[str, str]:
        """获取所有策略生命周期状态"""
        return {k: v.value for k, v in self._lifecycle_states.items()}

    def get_status(self) -> Dict[str, Any]:
        """获取管理器完整状态"""
        strategies_detail = []
        for name in self._strategy_instances:
            desc = self._descriptor_registry.get(name)
            strategies_detail.append({
                "name": name,
                "display_name": desc.display_name if desc else name,
                "category": desc.category if desc else "unknown",
                "lifecycle": self._lifecycle_states.get(name, StrategyLifecycle.UNREGISTERED).value,
                "health": self._health_statuses.get(name, StrategyHealth.UNKNOWN).value,
                "enabled_in_config": self.config.get("strategies", {}).get(name, {}).get(
                    "enabled", desc.enabled_by_default if desc else True),
                "config_version": (self._config_versions[name][-1].version
                                   if self._config_versions.get(name) else 0),
            })

        return {
            "registry_size": len(self._descriptor_registry),
            "instances_registered": len(self._strategy_instances),
            "start_order": self._start_order,
            "run_mode": self._run_mode.value,
            "batch_operations": list(self._batch_operations.keys()),
            "mode_frozen_strategies": list(self._mode_frozen_strategies),
            "lifecycle_summary": {
                s.value: sum(1 for v in self._lifecycle_states.values() if v == s)
                for s in StrategyLifecycle
            },
            "strategies": strategies_detail,
            "timestamp": datetime.now().isoformat(),
        }

    # ============================================================
    # 压测门禁状态（P2-12 供 dashboard 展示）
    # ============================================================

    def get_stress_gate_status(self, refresh: bool = False) -> Dict[str, Any]:
        """获取压测门禁状态（含最新压测结果与门禁配置）。

        Args:
            refresh: True 时主动执行一次压测刷新缓存，False 只读缓存（无副作用）。

        返回：
            {
                "enabled", "engine_injected", "min_score", "block_ratings",
                "cache_ttl_seconds", "latest_result", "cache_age_seconds",
                "timestamp",
            }
        """
        import time as _time

        result = self._stress_gate_cache.get("result")
        if refresh:
            result = self._run_stress_gate()

        cache_age = None
        if result is not None and self._stress_gate_cache.get("ts"):
            cache_age = round(_time.time() - self._stress_gate_cache["ts"], 1)

        return {
            "enabled": self._stress_gate_enabled,
            "engine_injected": self._stress_test_engine is not None,
            "min_score": self._stress_gate_min_score,
            "block_ratings": list(self._stress_gate_block_ratings),
            "cache_ttl_seconds": self._stress_gate_cache_ttl,
            "latest_result": result,
            "cache_age_seconds": cache_age,
            "timestamp": datetime.now().isoformat(),
        }

    # ============================================================
    # 持久化
    # ============================================================

    def persist_state(self) -> bool:
        """持久化管理器状态到文件"""
        try:
            os.makedirs("./data/strategy_manager", exist_ok=True)

            state = {
                "timestamp": datetime.now().isoformat(),
                "lifecycle_states": self.get_all_lifecycle_states(),
                "start_order": self._start_order,
                "config_versions": {
                    name: [
                        {
                            "version": v.version,
                            "timestamp": v.timestamp,
                            "reason": v.reason,
                            "applied": v.applied,
                        }
                        for v in versions
                    ]
                    for name, versions in self._config_versions.items()
                },
            }

            with open("./data/strategy_manager/state.json", "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)

            logger.debug("StrategyManager state persisted")
            return True
        except Exception as e:
            logger.error(f"StrategyManager persist state error: {e}")
            return False


# ============================================================
# 单例
# ============================================================

_manager_instance: Optional[StrategyManager] = None


def get_strategy_manager(config: Dict[str, Any] = None) -> StrategyManager:
    """获取策略管理器单例"""
    global _manager_instance
    if _manager_instance is None and config is not None:
        _manager_instance = StrategyManager(config)
    return _manager_instance


def reset_strategy_manager():
    """重置单例（用于重启）"""
    global _manager_instance
    _manager_instance = None


__all__ = [
    "StrategyManager",
    "StrategyLifecycle",
    "StrategyHealth",
    "RunMode",
    "StrategyDescriptor",
    "StrategyConfigSnapshot",
    "StrategyMetrics",
    "get_strategy_manager",
    "reset_strategy_manager",
]
