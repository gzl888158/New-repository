"""交易系统核心调度器，负责初始化并编排各交易组件、风控模块与主运行循环。"""
import asyncio
import os
import json
import time
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from loguru import logger
import numpy as np

from core.okx_client import OKXClient
from core.account_manager import AccountManager
from core.equity_monitor import EquityMonitor
from core.trade_journal import TradeJournal
from core.unified_layer import UnifiedAbstractionLayer, get_unified_layer
from core.event_id import EventIDGenerator
from core.state_manager import GlobalStateManager, get_global_state, StateKey
from core.apm_monitor import APMMonitor, get_apm_monitor, monitor_performance, trace_function
from data.redis_cache import RedisCache
from data.sqlite_storage import SQLiteStorage

from risk.strategy_risk import StrategyRiskControl
from risk.global_risk import GlobalRiskControl
from risk.notebook_fallback import NotebookFallbackControl
from risk.profit_optimizer import ProfitOptimizer
from risk.adaptive_controller import AdaptiveController
from risk.pnl_reconciler import PnLReconciler
from risk.correlation_risk import CorrelationRiskControl
from risk.black_swan_protection import BlackSwanProtection
from risk.allocation_agent import AllocationAgent
from risk.contract_risk_analyzer import ContractRiskAnalyzer, RiskLevel
from review.review_engine import ReviewEngine
from verification.verification_runner import VerificationRunner
from execution.order_queue import OrderQueue
from execution.order_executor import OrderExecutor
from execution.slippage_optimizer import SlippageOptimizer
from execution.fill_quality_tracker import FillQualityTracker
from execution.stale_order_manager import StaleOrderManager
from execution.order_lifecycle_manager import OrderLifecycleManager
from execution.execution_monitor import ExecutionMonitor, LatencyType
from execution.order_synchronizer import OrderStateSynchronizer
from execution.order_persistence import (
    OrderStore,
    TradingGate,
    OrderStartupSynchronizer,
    OrderPatrolService,
)
from execution.algo_orders import (
    SmartOrderRouter, AlgoExecutionEngine, AlgoOrderType, AlgoOrderConfig,
    TWAPExecutor, VWAPExecutor, IcebergOrderExecutor, DarkPoolRouter,
    ExecutionQualityMonitor, EnhancedExecutionQualityMonitor,
)
from core.stop_loss_manager import StopLossManager
from core.reversal_take_profit_engine import ReversalTakeProfitEngine
from core.profit_lock_engine import ProfitLockEngine
from core.tp_sl_monitor import TpSlMonitor
from core.conditional_order_manager import ConditionalOrderManager
from core.adaptive_tp_sl_engine import AdaptiveTpSlEngine
from core.adaptive_position_sizer import AdaptivePositionSizer
from core.atomic_writer import atomic_write_json
from core.order_fingerprint_masker import OrderFingerprintMasker
from monitoring.alert_manager import AlertManager
from monitoring.performance_monitor import PerformanceMonitor
from monitoring.alert_engine import AlertRuleEngine, AlertActionType, AlertState, DynamicThreshold
from monitoring.health_scorer import HealthScorer, HealthLevel, HealthComponent, ComponentScore
from monitoring.auto_recovery import AutoRecovery, RecoveryState, FailureType, RecoveryAction, RecoveryPolicy
from monitoring.notification_channels import NotificationManager
from monitoring.metrics_collector import MetricsCollector, get_metrics_collector
from analysis.historical_analyzer import HistoricalAnalyzer
from analysis.strategy_optimizer import StrategyOptimizer
from analysis.contribution_analyzer import ContributionAnalyzer, get_contribution_analyzer
from analysis.intelligent_analysis_agent import IntelligentAnalysisAgent

# P0: 策略组合优化引擎（MPT优化、相关性矩阵、VaR、绩效归因、策略组合模板）
from core.portfolio_optimizer import PortfolioOptimizer, get_portfolio_optimizer
from risk.dynamic_allocator import DynamicAllocator, get_dynamic_allocator
from risk.risk_budget_engine import RiskBudgetEngine, get_risk_budget_engine
from risk.strategy_correlation import StrategyCorrelationAnalyzer
from risk.diversification_optimizer import DiversificationOptimizer
from risk.portfolio_rebalancer import PortfolioRebalancer
from configs.settings import get_all_symbols

# P0: 统一参数优化系统（GA + BO + WF + MC）
from analysis.parameter_optimization import (
    ParameterOptimizationOrchestrator, OptimizationStrategy, OptimizationPhase,
)
from backtest.backtest_engine import BacktestEngine

from services.signal_processor import SignalProcessor
from services.signal_normalizer import SignalNormalizer
from services.signal_monitor import SignalMonitor
from services.market_regime_engine import MarketRegimeEngine
from services.signal_quality_engine import SignalQualityEngine
from services.strategy_coordinator import (
    StrategyCoordinator, SignalAggregationHub, PriorityScheduler,
    CircuitBreakerLinker, StrategyHealthMonitor,
)
from services.trading_recovery import TradingRecoveryService
from services.market_data_service import MarketDataService

# P0: 策略计算引擎（指标计算、信号生成、仓位规划、策略容器、迷你回测）
from core.strategy_engine import StrategyEngine, get_strategy_engine

# P0: 资金与仓位管理（资金池分区、币种权重、杠杆分级、盈亏再分配、对冲调度）
from core.capital_manager import CapitalManager, get_capital_manager, CapitalPoolType
from core.capital_adaptive_allocator import CapitalAdaptiveAllocator
from core.position_manager import PositionManager

# P0: 多层级风控拦截（五层串行校验，任意一层拦截直接驳回下单）
from core.risk_gate import RiskGate, get_risk_gate, RiskAction

# P2: 独立风控裁决器（traceID 贯穿 + 裁决事件溯源 + 唯一下单裁决入口）
from core.risk_adjudicator import RiskAdjudicator

# 实时风险监控引擎（保证金率/回撤 fail-closed，R1/R2 线上生效）
from core.risk_monitor import RealTimeRiskMonitor

# P22-8: 生产级红线合规检查器
from core.compliance_checker import ComplianceChecker, get_compliance_checker

# P0: 算力动态调度（高波动提升优先级，低波动降频，CPU过载全局降频）
from core.compute_scheduler import ComputeScheduler, get_compute_scheduler

# P0: 统一策略管理器（生命周期/热重载/健康监控/依赖管理）
from core.strategy_manager import StrategyManager, get_strategy_manager, StrategyLifecycle, StrategyHealth
from core.mini_backtester import get_mini_backtester

from decision.decision_coordinator import DecisionCoordinator
from decision.rule_based_engine import RuleBasedEngine
from decision.ensemble_decision_maker import EnsembleDecisionMaker, EnsembleMethod
from decision.decision_validator import DecisionValidator
from decision.decision_quality_evaluator import DecisionQualityEvaluator
from decision.confidence_calibrator import ConfidenceCalibrator
from decision.intelligent_decision_engine import (
    IntelligentDecisionEngine, get_intelligent_decision_engine,
    DecisionAuditEntry, MTFFusionResult, DecisionContext, TimeFrameSignal,
    DecisionUrgency, MetaDecisionVerdict,
)
from decision.ml_decision_engine import (
    MLDecisionEngine, DriftLevel, ModelStatus,
)
from decision.rl_agent import (
    TradingRLAgent, get_rl_agent, StateEncoding,
)

from app.services.trading_pipeline import (
    PipelineOrchestrator,
    PipelineStage,
    SignalProcessingPipeline,
    DecisionExecutor,
    AnomalyDetector,
    RecoveryHandler,
    PriceSpikeDetector,
    VolumeSurgeDetector,
    OrderFailureRateDetector,
    LatencySpikeDetector,
    SignalFrequencyDetector,
    PNLDropDetector,
)

from app.services.adaptive_learning import (
    OnlineLearner,
    ParameterAdaptor,
    StrategyEvolver,
    MarketRegimeDetector,
    PerformanceFeedback,
    KnowledgeBase,
    MetaLearner,
)

class TradingScheduler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.position_manager = None
        
        # P0: 后台任务注册表，支持优雅关闭和崩溃检测
        self._tasks: Dict[str, asyncio.Task] = {}
        
        logger.info("Initializing TradingScheduler - step 1: UnifiedAbstractionLayer")
        # P2: 初始化统一抽象层
        self.unified_layer = get_unified_layer(config)
        logger.info("UnifiedAbstractionLayer initialized")

        # P1: 事件溯源接线——创建 EventStore 并注入事件总线（所有 publish 自动落盘）
        # 失败不阻断启动：无 EventStore 时事件总线照常工作（仅不持久化）。
        self.event_store = None
        try:
            from core.event_store import EventStore
            self.event_store = EventStore()
            self.unified_layer.event_bus.set_event_store(self.event_store)
            logger.info("EventStore wired into event bus (event sourcing enabled)")
        except Exception as e:
            logger.warning(f"EventStore wiring failed (event sourcing disabled): {e}")

        self.agent_learning_memory = None
        try:
            from core.agent_learning_memory import AgentLearningMemory
            memory_cfg = config.get("agent_learning_memory", {})
            self.agent_learning_memory = AgentLearningMemory(
                event_store=self.event_store,
                max_entries=memory_cfg.get("max_entries", 5000),
                enabled=memory_cfg.get("enabled", True),
            )
            logger.info("Shared AgentLearningMemory initialized")
        except Exception as e:
            logger.warning(f"Shared AgentLearningMemory unavailable (degraded): {e}")

        # P1: 事件溯源重放器——从事件日志重建订单状态（只读对账，不修改业务状态）
        self.event_replayer = None
        try:
            from core.event_replay import EventReplayer
            self.event_replayer = EventReplayer(event_store=self.event_store)
            logger.info("EventReplayer initialized (order state replay/reconcile enabled)")
        except Exception as e:
            logger.warning(f"EventReplayer init failed (replay disabled): {e}")
        
        logger.info("Initializing TradingScheduler - step 2: GlobalStateManager")
        # P2: 初始化全局状态管理器
        self.state_manager = get_global_state(config)
        self.state_manager.load_state()
        logger.info("GlobalStateManager initialized and loaded")
        
        logger.info("Initializing TradingScheduler - step 3: APMMonitor")
        # P2: 初始化APM监控器
        self.apm_monitor = get_apm_monitor(config)
        logger.info("APMMonitor initialized")
        
        logger.info("Initializing TradingScheduler - step 4: OKXClient")
        self.okx_client = OKXClient(config)
        logger.info("OKXClient initialized")
        
        logger.info("Initializing TradingScheduler - step 5: RedisCache")
        self.redis_cache = RedisCache(config)
        logger.info("RedisCache initialized")
        
        logger.info("Initializing TradingScheduler - step 6: SQLiteStorage")
        self.sqlite_storage = SQLiteStorage(config)
        logger.info("SQLiteStorage initialized")

        # 健康检查失败抑制：连续失败 N 次后不再重复 ERROR 日志
        self._health_fail_count: Dict[str, int] = {}
        self._health_suppressed: Dict[str, bool] = {}
        # P6: reduce_size冷却期，防止同一币种频繁缩仓
        self._last_reduce_size_ts: Dict[str, float] = {}
        self._reduce_size_cooldown: int = 300  # 5分钟冷却
        # P8: 黑名单持仓平仓冷却期，防止平仓失败后每5秒重复下单
        self._blacklist_close_ts: Dict[str, float] = {}
        self._blacklist_close_cooldown: int = 60  # 60秒冷却

        # 统一利润锁定梯度引擎（手工仓 + 策略仓）：保本位移 → 部分落袋 → 紧追踪 → 反转落袋。
        # 替换原「浮盈 3% 才激活」的移动止盈，解决盈利不落袋、回落被平仓导致的资金磨损。
        self._tp_lock_cfg = self.config.get("trailing_take_profit", {})
        self._tp_lock_check_interval = int(self._tp_lock_cfg.get("check_interval", 5))
        self._tp_lock_min_notional = float(self._tp_lock_cfg.get("min_notional_usd", 1.0))
        # 利润锁定梯度引擎（纯计算 + 状态管理，README 见 core/profit_lock_engine.py）
        self.profit_lock_engine = ProfitLockEngine(self.config)
        self._tp_lock_enabled = self.profit_lock_engine.enabled
        # 统一止盈止损口径（fee-aware 保本缓冲 + 保护视图），供利润锁定引擎复用
        self.tp_sl_monitor = TpSlMonitor(self.config)
        # 利润锁定审计表（落库锁利动作，供复盘「是否盈利时落袋」）
        self._profit_lock_db_path = self.config.get("sqlite", {}).get("db_path", "./data/trading.db")
        self._init_profit_lock_audit_table()
        # 利润锁定状态持久化：watchdog 重启后恢复梯度进度（保本/部分落袋/紧追踪），避免锁利进度回退
        self._init_profit_lock_state_table()
        self._load_profit_lock_states()

        # 同步实际账户权益到配置：解决策略仍按初始小资金计算仓位导致资金利用率低的问题
        # 带超时保护，避免OKX API延迟导致启动卡住
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.okx_client.get_account_info)
                account_info = future.result(timeout=10)  # 10秒超时
            if account_info:
                parsed = self.okx_client._parse_account_info(account_info)
                equity = float(parsed.total_equity) if parsed else 0.0
                if equity > 0:
                    config["trading"]["total_capital"] = equity
                    logger.info(f"Updated trading.total_capital to actual USDT equity: {equity:.2f} USDT")
        except concurrent.futures.TimeoutError:
            logger.warning("Timeout fetching account equity for capital config, using configured value")
        except Exception as e:
            logger.warning(f"Could not fetch account equity for capital config: {e}")

        self.alert_manager = AlertManager(config)
        self.performance_monitor = PerformanceMonitor(config, self.alert_manager)
        # 注入redis和okx_client引用，用于监控其状态
        self.performance_monitor.set_dependencies(redis_cache=self.redis_cache, okx_client=self.okx_client)
        
        # ── 企业级资金变动自适应检测器 ──
        self.equity_monitor = EquityMonitor(config)
        
        self.account_manager = AccountManager(config, self.okx_client, self.redis_cache, equity_monitor=self.equity_monitor)
        self.capital_adaptive_allocator = CapitalAdaptiveAllocator(config, account_manager=self.account_manager)
        self.trade_journal = TradeJournal(config, self.sqlite_storage, self.redis_cache, self.okx_client)
        # P0-2 记账事件化：注入事件总线到 TradeJournal（开仓/平仓落账发布 POSITION_OPENED/TRADE_RECORDED）
        self.trade_journal.set_event_bus(self.unified_layer.event_bus)
        
        self.strategy_risk = StrategyRiskControl(config, self.account_manager)
        self.global_risk = GlobalRiskControl(config, self.redis_cache, self.okx_client, alert_manager=self.alert_manager)
        self.global_risk.set_account_manager(self.account_manager)  # P1-⑤：单一权益基准源
        self.strategy_risk.set_global_risk(self.global_risk)
        self.notebook_fallback = NotebookFallbackControl(config, self.okx_client, self.redis_cache)
        self.profit_optimizer = ProfitOptimizer(config)
        self.adaptive_controller = AdaptiveController(config, self.okx_client, self.sqlite_storage, self.trade_journal, self.profit_optimizer, self.account_manager, equity_monitor=self.equity_monitor)
        self.pnl_reconciler = PnLReconciler(config, self.okx_client, self.sqlite_storage)
        self.correlation_risk = CorrelationRiskControl(config, self.okx_client, self.redis_cache)

        # 初始化权益同步
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                equity = float(account_info.get("totalEq", 0))
                if equity > 0:
                    self.profit_optimizer.update_equity(equity)
                    logger.info(f"ProfitOptimizer initialized with equity: {equity:.2f}")
        except Exception as e:
            logger.warning(f"Failed to initialize ProfitOptimizer equity: {e}")
        
        self.order_queue = OrderQueue(config)
        self.order_executor = OrderExecutor(config, self.okx_client, self.redis_cache, self.sqlite_storage, self.account_manager, self.trade_journal)
        self.order_executor.set_order_queue(self.order_queue)
        self.order_executor.set_profit_optimizer(self.profit_optimizer)
        self.order_executor.set_alert_manager(self.alert_manager)
        # P1 埋点：注入事件总线到 OrderExecutor（下单/成交/平仓发布事件）
        self.order_executor.set_event_bus(self.unified_layer.event_bus)

        # 黑天鹅保护：必须在 self.order_executor 创建之后构造（依赖 self.order_executor）
        self.black_swan_protection = BlackSwanProtection(
            config, self.okx_client,
            global_risk=self.global_risk,
            order_executor=self.order_executor,
        )

        # 订单状态同步器（实时同步挂单/成交 + 定期全量对账 + 差异主动修复）
        self.order_state_synchronizer = OrderStateSynchronizer(
            config, self.okx_client, self.alert_manager
        )
        logger.info("OrderStateSynchronizer initialized")

        # ── 重启挂单丢失修复：订单持久化 + 下单闸门 + 启动同步 + 后台巡检 ──
        order_persistence_cfg = config.get("order_persistence", {})
        if order_persistence_cfg.get("enabled", True):
            order_db_path = order_persistence_cfg.get("db_path", "data/orders.db")
            self.order_store = OrderStore(order_db_path)
            self.order_gate = TradingGate(initially_open=False)
            # 注入到 OrderExecutor：下单前检查闸门 + INIT/PENDING/终态落盘
            self.order_executor.set_order_store(self.order_store)
            self.order_executor.set_trading_gate(self.order_gate)
            self.order_startup_synchronizer = OrderStartupSynchronizer(
                self.order_store, self.order_gate, self.okx_client,
                alert_manager=self.alert_manager, config=order_persistence_cfg,
            )
            self.order_patrol = OrderPatrolService(
                self.order_store, self.order_gate, self.okx_client,
                alert_manager=self.alert_manager, config=order_persistence_cfg,
            )
            logger.info(
                f"OrderPersistence initialized: db={order_db_path}, gate=closed (fail-closed)"
            )
        else:
            self.order_store = None
            self.order_gate = None
            self.order_startup_synchronizer = None
            self.order_patrol = None
            logger.info("OrderPersistence disabled, skip init")

        # 注入成交质量追踪器
        self.fill_quality_tracker = FillQualityTracker(config, self.sqlite_storage)
        self.order_executor.set_fill_quality_tracker(self.fill_quality_tracker)

        # 注入滑点优化器（动态限价偏移+自适应学习）
        self.slippage_optimizer = SlippageOptimizer(config)
        self.order_executor.set_slippage_optimizer(self.slippage_optimizer)
        logger.info("SlippageOptimizer injected into OrderExecutor")

        # P1: 注入订单指纹混淆器（防针对量化第二层防护，默认关闭）
        # 通过 config.execution.fingerprint_masking_enabled 控制开关
        self.fingerprint_masker = OrderFingerprintMasker(config)
        self.order_executor.set_fingerprint_masker(self.fingerprint_masker)
        logger.info(f"OrderFingerprintMasker injected into OrderExecutor (enabled={self.fingerprint_masker.enabled})")

        # 风控前置：注入 order_executor 到 global_risk（平仓/减仓信号走五层风控校验）
        self.global_risk.set_order_executor(self.order_executor)
        # 注入 order_executor 到 correlation_risk（相关性减仓也走五层风控）
        self.correlation_risk.order_executor = self.order_executor
        self.correlation_risk.set_alert_manager(self.alert_manager)

        # P0: 统一止损管理器（确保止损信号直达OrderExecutor）
        self.stop_loss_manager = StopLossManager(config, self.okx_client, self.redis_cache, self.trade_journal, self.order_executor)
        self.stop_loss_manager.set_alert_callback(self.alert_manager.send_alert)

        # P0: 条件单管理器（补挂失败处理、存量仓位风险控制、交易所同步）
        self.conditional_order_manager = ConditionalOrderManager(config, self.okx_client, self.redis_cache)
        self.conditional_order_manager.set_fill_quality_tracker(self.fill_quality_tracker)
        self.conditional_order_manager.set_trade_journal(self.trade_journal)
        self.order_executor.set_conditional_manager(self.conditional_order_manager)
        self.stop_loss_manager.set_conditional_manager(self.conditional_order_manager)

        # 行情反转智能计算落袋自适应引擎（HMM 为主 + 指标兜底，动态止盈 + 反转落袋）
        self.reversal_take_profit_engine = ReversalTakeProfitEngine(config)
        self.stop_loss_manager.set_reversal_take_profit_engine(self.reversal_take_profit_engine)
        logger.info("ReversalTakeProfitEngine initialized and injected into StopLossManager")

        # 挂单时效管理器（检测长时间未成交挂单、智能重定价/撤销）
        self.stale_order_manager = StaleOrderManager(config)
        self.stale_order_manager.set_dependencies(self.okx_client, self.order_executor)
        logger.info("StaleOrderManager initialized and injected into OrderExecutor")

        # P5: 注入WS状态检查器到订单生命周期管理器，WS断开时延长订单超时
        if hasattr(self, 'order_lifecycle') and self.order_lifecycle:
            self.order_lifecycle.set_ws_status_checker(
                lambda: self.okx_client.is_ws_public_connected() if self.okx_client else True
            )
            logger.info("WS status checker injected into OrderLifecycleManager")

        # 统一数据清理引擎（定期清理DB/文件/内存/Redis）
        from core.data_cleaner import DataCleaner
        self.data_cleaner = DataCleaner(config)
        self.data_cleaner.set_sqlite_storage(self.sqlite_storage)
        self.data_cleaner.set_redis_cache(self.redis_cache)
        self.data_cleaner.set_trade_journal(self.trade_journal)
        self.data_cleaner.set_stop_loss_manager(self.stop_loss_manager)
        self.data_cleaner.set_order_queue(self.order_executor._order_queue if hasattr(self.order_executor, '_order_queue') else None)
        logger.info("DataCleaner initialized")

        # 过期残留自动清理器（定时+阈值触发清理过期临时残留；白名单审计数据只归档绝不删除）
        from core.expired_residue_cleaner import ExpiredResidueCleaner
        self.residue_cleaner = ExpiredResidueCleaner(config)
        self.residue_cleaner.set_live_probe(self._has_live_strategy)
        logger.info("ExpiredResidueCleaner initialized")

        # P0: 精准交易成本分析器 - 杜绝磨损型交易
        from core.trade_cost_analyzer import TradeCostAnalyzer
        self.trade_cost_analyzer = TradeCostAnalyzer(config.get("trade_cost", {}))
        self.trade_cost_analyzer.set_okx_client(self.okx_client)
        self.trade_cost_analyzer.prefetch_fee()
        self.order_executor.set_trade_cost_analyzer(self.trade_cost_analyzer)
        self.trade_journal.set_trade_cost_analyzer(self.trade_cost_analyzer)
        logger.info("TradeCostAnalyzer initialized and injected into OrderExecutor and TradeJournal")

        # P0: 交易审核器 - 全链路审核
        from core.trade_auditor import TradeAuditor
        self.trade_auditor = TradeAuditor(config.get("trade_auditor", {}), data_dir=config.get("data_dir", "./data"), sqlite_storage=self.sqlite_storage)
        self.trade_auditor.load_state()
        self.order_executor.set_trade_auditor(self.trade_auditor)
        logger.info("TradeAuditor initialized and injected into OrderExecutor")

        # P0: 智能交易体 - 自适应判断 + 成长性学习 + 分层协调
        from core.intelligent_agent import IntelligentTradingAgent
        agent_cfg = dict(config.get("intelligent_agent", {}))
        # P0: 注入真实账户权益，供智能体黑名单小账户判定使用。
        # config.yaml 无 intelligent_agent 顶层键，真实 total_capital 位于 trading.total_capital
        # （若仍用默认 100.0，会导致小账户判定 is_small_account 恒为 False，
        #   从而漏掉 AVAX|scalping 等大额亏损币种的拉黑）
        if "total_capital" not in agent_cfg:
            agent_cfg["total_capital"] = config.get("trading", {}).get("total_capital", 100.0)
        self.intelligent_agent = IntelligentTradingAgent(agent_cfg)
        if self.agent_learning_memory is not None:
            self.intelligent_agent.set_learning_memory(self.agent_learning_memory)
        self.intelligent_agent.set_alert_manager(self.alert_manager)
        self.order_executor.set_intelligent_agent(self.intelligent_agent)
        # 企业级资金管理：注入智能体到 AdaptiveController，驱动可部署性感知分配
        self.adaptive_controller.set_intelligent_agent(self.intelligent_agent)
        logger.info("IntelligentTradingAgent initialized and injected into OrderExecutor")

        # 提前初始化analyzer和optimizer，供后续策略注册使用
        self.analyzer = HistoricalAnalyzer(self.trade_journal)
        self.optimizer = StrategyOptimizer(self.trade_journal, config)

        # 提前初始化 market_regime_engine（供 StrategyCoordinator 和 StrategyManager 使用）
        self.market_regime_engine = MarketRegimeEngine(config, self.okx_client)
        self.market_regime_engine.set_alert_manager(self.alert_manager)
        logger.info("MarketRegimeEngine initialized for unified market state fusion")

        # P0: 注入 MarketRegimeEngine 到智能体，让智能体的市场状态感知复用多因子融合结果，
        # 消除 get_market_regime 永远返回 UNKNOWN 的双套逻辑漂移（智能体旧版从未被喂数据）。
        if hasattr(self, 'intelligent_agent') and self.intelligent_agent:
            self.intelligent_agent.set_regime_engine(self.market_regime_engine)
            logger.info("MarketRegimeEngine injected into IntelligentTradingAgent")

        # 企业级：多层信号前置过滤链（收敛 audit_signal 9 维度 + RegimeGate 为统一口径，
        # 黑名单最高优先短路 + 可解释 breakdown；未注入时智能体回退 legacy 手写串行维度）
        from core.signal_pre_filter import SignalPreFilterChain
        self.signal_pre_filter_chain = SignalPreFilterChain(config=config)
        self.intelligent_agent.set_signal_pre_filter_chain(self.signal_pre_filter_chain)
        logger.info("SignalPreFilterChain initialized and injected into IntelligentTradingAgent")

        # 企业级：统一防抖动/防频繁交易引擎（收敛品种冷却+信号去重+策略冷却+全局冷却+亏损冷却+刷单检测+自适应冷却）
        from core.anti_debounce_engine import AntiDebounceEngine
        self.debounce_engine = AntiDebounceEngine(config=config)
        self.intelligent_agent.set_debounce_engine(self.debounce_engine)
        logger.info("AntiDebounceEngine initialized and injected into IntelligentTradingAgent")

        # 强化自适应智能体：注入 MetricsPipeline，把决策动作/拒绝率/学习质量/自愈动作外发到统一指标流水线
        from core.metrics_pipeline import get_metrics_pipeline
        self.metrics_pipeline = get_metrics_pipeline(config)
        self.intelligent_agent.set_metrics_pipeline(self.metrics_pipeline)
        logger.info("MetricsPipeline initialized and injected into IntelligentTradingAgent")

        # 提前初始化 strategy_coordinator（供 StrategyManager.inject_services 使用）
        self.strategy_coordinator = StrategyCoordinator(config, self.trade_journal, self.market_regime_engine, self.adaptive_controller)
        logger.info("StrategyCoordinator initialized for strategy collaboration")

        # P0: 初始化统一策略管理器（生命周期/热重载/健康监控）
        mgr_cfg = config.get("strategy_manager", {})
        if mgr_cfg.get("enabled", True):
            self.strategy_manager = get_strategy_manager(config)
            self.strategy_manager.inject_services(
                coordinator=self.strategy_coordinator,
                adaptive_controller=self.adaptive_controller,
                stop_loss_manager=self.stop_loss_manager,
                order_executor=self.order_executor,
                okx_client=self.okx_client,
                market_regime_engine=self.market_regime_engine,
                intelligent_agent=self.intelligent_agent,
                mini_backtester=get_mini_backtester(config),
                stress_test_engine=self._build_stress_test_engine(config),
                account_manager=getattr(self, "account_manager", None),
            )

            # P0: 使用 StrategyLoader 动态加载所有策略（替代硬编码）
            # load_and_register_all 会：
            #   1. 自动发现 strategies/ 包中的所有策略类
            #   2. 验证策略类和配置
            #   3. 解析依赖关系确定启动顺序
            #   4. 动态创建策略实例并注入所有依赖
            #   5. 自动注册到 StrategyManager 和 StrategyCoordinator
            load_report = self.strategy_manager.load_and_register_all(
                redis_cache=self.redis_cache,
            )
            logger.info(f"StrategyManager: dynamic loading complete — "
                       f"{load_report.loaded} loaded, {load_report.skipped} skipped")

            # 设置便捷引用（向后兼容）
            all_instances = self.strategy_manager.get_all_instances()
            self.grid_strategy = all_instances.get("grid")
            self.trend_strategy = all_instances.get("trend")
            self.scalping_strategy = all_instances.get("scalping")
            self.arbitrage_strategy = all_instances.get("arbitrage")
            self.spot_grid_strategy = all_instances.get("spot_grid")
            self.spot_martingale_strategy = all_instances.get("spot_martingale")

            # 注册 symbol -> strategy 精确映射，修正 _infer_strategy_from_symbol 误判
            # 现货策略使用 {base}-USDT 现货符号，与合约 -USDT-SWAP 可精确区分
            if self.account_manager is not None:
                if self.spot_grid_strategy is not None and hasattr(self.spot_grid_strategy, "_all_symbols"):
                    for sym in self.spot_grid_strategy._all_symbols:
                        self.account_manager.register_symbol_strategy(sym, "spot_grid")
                if self.spot_martingale_strategy is not None and hasattr(self.spot_martingale_strategy, "_all_symbols"):
                    for sym in self.spot_martingale_strategy._all_symbols:
                        self.account_manager.register_symbol_strategy(sym, "spot_martingale")
                # 注意：arbitrage 与 grid/trend/scalping 共用 -USDT-SWAP 合约符号，
                # 符号级推断无法区分，故不注册 arbitrage，其仓位由策略自身追踪。

            # 注册生命周期回调
            self.strategy_manager.on("on_error", self._on_strategy_error)
            self.strategy_manager.on("on_health_change", self._on_strategy_health_change)
        else:
            self.strategy_manager = None
            logger.info("StrategyManager disabled")

        # 注入 strategy_manager 到 adaptive_controller（策略名称单一事实来源）
        if self.strategy_manager is not None:
            self.adaptive_controller.set_strategy_manager(self.strategy_manager)

        # 注册策略实例到优化器，支持参数热更新
        if self.strategy_manager:
            all_instances = self.strategy_manager.get_all_instances()
            self.optimizer.register_all_strategies(**{
                k: v for k, v in all_instances.items() if v is not None
            })
            logger.info(f"Strategy instances registered to optimizer for hot-update "
                       f"({len(all_instances)} strategies)")
        else:
            logger.warning("StrategyManager not available, optimizer registration skipped")

        # 注入StrategyOptimizer到AdaptiveController，支持空闲资金优化时热更新参数
        self.adaptive_controller.set_strategy_optimizer(self.optimizer)

        # P0: 注入trade_journal到grid_strategy，用于网格对账确认实际成交
        if self.grid_strategy and hasattr(self.grid_strategy, 'set_trade_journal'):
            self.grid_strategy.set_trade_journal(self.trade_journal)

        # 注入 grid 主动管理持仓 symbol 提供器到 OrderExecutor：
        # 重启对账时避免把 grid 自身持仓（DB open 记录被 P7-4 sync 污染）误判为框架外 orphan。
        if self.grid_strategy and hasattr(self.grid_strategy, 'get_active_position_symbols'):
            self.order_executor.set_managed_position_symbols_provider(
                self.grid_strategy.get_active_position_symbols
            )

        # P0: 注入sqlite_storage到scalping_strategy，用于孤儿持仓恢复与对账
        if self.scalping_strategy and hasattr(self.scalping_strategy, 'set_sqlite_storage'):
            self.scalping_strategy.set_sqlite_storage(self.sqlite_storage)

        self.allocation_agent = AllocationAgent(config, self.trade_journal, self.profit_optimizer, self.account_manager)
        if self.agent_learning_memory is not None:
            self.allocation_agent.set_learning_memory(self.agent_learning_memory)
        if self.strategy_manager is not None:
            self.allocation_agent.set_strategy_manager(self.strategy_manager)
        self.allocation_agent.set_adaptive_controller(self.adaptive_controller)
        logger.info("AllocationAgent initialized for dynamic capital allocation")

        # P0: 初始化策略组合优化引擎（MPT优化、相关性矩阵、VaR、绩效归因、策略组合模板）
        self.portfolio_optimizer = get_portfolio_optimizer(config)
        # 将 PortfolioOptimizer 与 AllocationAgent 关联
        self.allocation_agent.set_portfolio_optimizer(self.portfolio_optimizer)

        # P0: 初始化动态资金分配引擎（三级资金池、Kelly公式、优先级瀑布）
        self.dynamic_allocator = get_dynamic_allocator(config)
        self.allocation_agent.set_dynamic_allocator(self.dynamic_allocator)
        logger.info("DynamicAllocator initialized and injected into AllocationAgent")

        # ── 策略协同系统：相关性分析、分散化优化、组合再平衡（需在 RiskBudgetEngine 之前初始化）──
        self.strategy_correlation = StrategyCorrelationAnalyzer(config)
        logger.info("StrategyCorrelationAnalyzer initialized for strategy correlation analysis")

        self.diversification_optimizer = DiversificationOptimizer(config)
        logger.info("DiversificationOptimizer initialized for portfolio diversification optimization")

        self.portfolio_rebalancer = PortfolioRebalancer(config)
        logger.info("PortfolioRebalancer initialized for portfolio rebalancing management")

        # P0: 初始化风险预算引擎（风险分解、风险平价、RAPM、动态预算调整）
        self.risk_budget_engine = get_risk_budget_engine(config)
        self.risk_budget_engine.set_portfolio_optimizer(self.portfolio_optimizer)
        self.risk_budget_engine.set_correlation_analyzer(self.strategy_correlation)
        self.risk_budget_engine.set_diversification_optimizer(self.diversification_optimizer)
        logger.info("RiskBudgetEngine initialized for risk budget allocation")

        logger.info("PortfolioOptimizer initialized for portfolio-level optimization")

        self.market_data_service = MarketDataService(config, self.okx_client, self.redis_cache, alert_manager=self.alert_manager)
        logger.info("MarketDataService initialized for robust market data ingestion")

        # P0: 初始化策略计算引擎（核心收益引擎，高波动捕捉、多策略并行）
        self.strategy_engine = get_strategy_engine(config)
        self.strategy_engine.initialize()
        logger.info("StrategyEngine initialized for high-frequency strategy computation")

        # P0: 注入真实仓位源到 StrategyContainer，恢复 P22 状态无漂移校验
        # 修复：set_position_provider 此前从未被调用，_get_exchange_positions 恒返回 None，
        # 导致 _verify_strategy_positions 不执行 → scalping 幽灵挂单 TTL 清理失效，
        # pending_entry_exists 持续拦截新开单（账户实际空仓）。
        if self.okx_client and hasattr(self.okx_client, "get_positions"):
            self.strategy_engine._strategy_container.set_position_provider(self.okx_client.get_positions)
            logger.info("PositionProvider injected into StrategyContainer (P22 drift verification restored)")

        # 注册策略引擎信号回调（将策略信号传递给信号处理器）
        self.strategy_engine.register_signal_callback(self._on_strategy_signal)
        logger.info("StrategyEngine signal callback registered")

        # P0: 初始化资金与仓位管理器（资金池分区、币种权重、杠杆分级、盈亏再分配、对冲调度）
        self.capital_manager = get_capital_manager(config)
        all_symbols = get_all_symbols(config)
        trading_capital = float(config.get("trading", {}).get("total_capital", 100.0))
        self.capital_manager.initialize(all_symbols, trading_capital)
        # 连接 AdaptiveController 实现风险预算协作
        self.capital_manager.set_adaptive_controller(self.adaptive_controller)
        logger.info(f"CapitalManager initialized: {len(all_symbols)} symbols, "
                    f"capital={trading_capital:.2f} USDT")

        # P0: 初始化五层风控拦截器（事前/事中/持仓实时/单日/紧急熔断）
        self.risk_gate = get_risk_gate(config, okx_client=self.okx_client)
        logger.info("RiskGate initialized: 5-layer serial risk control")

        # 持仓同步失败达到阈值时由 RiskGate 冻结新开仓。
        self.position_manager = PositionManager(
            config,
            self.okx_client,
            self.sqlite_storage,
            self.redis_cache,
        )
        self.position_manager.set_alert_manager(self.alert_manager)
        self.risk_gate.set_position_manager(self.position_manager)
        # P0-手动平仓清理链：注册持仓移除回调，触发 OrderExecutor 清理止损/策略/状态
        self.position_manager.on_position_removal(self.order_executor.handle_position_removal)
        logger.info("PositionManager initialized and connected to RiskGate + OrderExecutor removal callback")

        # P1: 注入SQLite存储，持久化风控拦截事件到 risk_events 表（修复审计缺口）
        self.risk_gate.set_sqlite_storage(self.sqlite_storage)
        logger.info("RiskGate SQLite storage injected for risk event persistence")

        # P2: 独立风控裁决器——复用六层 RiskGate + traceID 贯穿 + 裁决事件溯源（发布到 EventStore）
        self.risk_adjudicator = RiskAdjudicator(
            risk_gate=self.risk_gate,
            event_bus=self.unified_layer.event_bus,
        )
        logger.info("RiskAdjudicator initialized (independent risk adjudication choke point)")

        # P2: 实时风险监控引擎（保证金率/回撤 fail-closed，R1/R2 线上生效）
        self.risk_monitor = RealTimeRiskMonitor(config)
        self.risk_monitor.inject_dependencies(
            okx_client=self.okx_client,
            account_manager=self.account_manager,
            capital_manager=self.capital_manager,
            position_manager=self.position_manager,
            risk_gate=self.risk_gate,
            equity_monitor=self.equity_monitor,
        )
        logger.info("RealTimeRiskMonitor initialized and dependencies injected")

        # P2: 将 RealTimeRiskMonitor 注入 StrategyEngine，使 R1/R2 风险门控在信号执行前真正生效
        self.strategy_engine.inject_dependencies(risk_monitor=self.risk_monitor)
        logger.info("StrategyEngine: RealTimeRiskMonitor injected for pre-trade risk gate")

        # 启动时立即同步账户权益到 RiskGate，避免 equity=0 导致首次下单误拦截
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                equity = float(account_info.get("totalEq", 0))
                if equity > 0:
                    account = self.okx_client._parse_account_info(account_info)
                    self.risk_gate.update_account_status(
                        equity=equity,
                        available_margin=float(account.available_balance),
                        daily_pnl=float(account.unrealized_pnl),
                        daily_start_equity=equity - float(account.unrealized_pnl)
                    )
                    logger.info(f"RiskGate equity initialized: {equity:.2f}")
        except Exception as e:
            logger.warning(f"Failed to initialize RiskGate equity: {e}")

        # P0: 注入五层风控到订单执行器（通过路径：记录开仓次数/平仓盈亏给L4）
        self.order_executor.set_risk_gate(self.risk_gate)
        logger.info("RiskGate injected into OrderExecutor for L4 trade tracking")

        # P2: 注入独立风控裁决器到订单执行器（下单前最终裁决 + traceID 全链路串联）
        self.order_executor.set_risk_adjudicator(self.risk_adjudicator)
        logger.info("RiskAdjudicator injected into OrderExecutor for final adjudication")

        # P0: 注入风控埋点到OKX客户端（L2事中风控：API频率限流+网络延迟监控）
        self.okx_client.set_risk_callbacks(
            on_api_call=self.risk_gate.record_api_call,
            on_latency=self.risk_gate.record_latency
        )
        logger.info("RiskGate L2 callbacks injected into OKXClient for API rate/latency monitoring")

        # 复盘迭代指导框架引擎（每日复盘+月度迭代）— 必须在 risk_gate 赋值之后
        self.review_engine = ReviewEngine(
            config, okx_client=self.okx_client, risk_gate=self.risk_gate,
            trade_journal=self.trade_journal
        )
        self.review_engine.set_callbacks(
            on_daily_review=self._on_daily_review_completed,
            on_monthly_plan=self._on_monthly_plan_generated,
        )
        logger.info("ReviewEngine initialized: daily review + monthly iteration")

        # 激进合约专属风险分析引擎（市场行情+程序技术+策略逻辑三维扫描）
        self.contract_risk_analyzer = ContractRiskAnalyzer(config, self.okx_client, self.risk_gate)
        self.contract_risk_analyzer.set_callbacks(
            on_pause_new=self._on_risk_pause_new,
            on_pause_strategy=self._on_risk_pause_strategy,
            on_emergency_close=self._on_risk_emergency_close,
            on_reduce_size=self._on_risk_reduce_size,
            on_adjust_leverage=self._on_risk_adjust_leverage,
        )
        logger.info("ContractRiskAnalyzer initialized: 3-dimension risk scan (market/technical/strategy)")

        # 策略贡献度分析引擎（多窗口PnL拆解+健康度+生命周期+资金重分配）
        contrib_cfg = config.get("contribution", {})
        if contrib_cfg.get("enabled", True):
            self.contribution_analyzer = get_contribution_analyzer(
                sqlite_storage=self.sqlite_storage,
                trade_journal=self.trade_journal,
                config=config,
                okx_client=self.okx_client,
                account_manager=self.account_manager,
            )
            # 注入动态分配权重
            allocations = self.adaptive_controller.get_allocations() if hasattr(self.adaptive_controller, 'get_allocations') else {}
            self.contribution_analyzer.set_dynamic_allocations(allocations)
            logger.info("ContributionAnalyzer initialized: multi-window PnL + lifecycle + health scoring")
            self._contribution_task = None
        else:
            self.contribution_analyzer = None
            logger.info("ContributionAnalyzer disabled")

        # 验证运行器（周度策略验证）
        self.verification_runner = VerificationRunner(config)
        logger.info("VerificationRunner initialized")

        self.signal_quality_engine = SignalQualityEngine(config, self.market_regime_engine, self.trade_journal)
        logger.info("SignalQualityEngine initialized for multi-factor signal scoring")

        # 注册所有策略到协调器（动态方式，替代硬编码）
        if self.strategy_manager:
            for name, instance in self.strategy_manager.get_all_instances().items():
                if instance is not None:
                    self.strategy_coordinator.register_strategy(name, instance)
            logger.info(f"All strategies registered to StrategyCoordinator from StrategyManager")
        else:
            # 兜底：直接使用调度器属性
            self.strategy_coordinator.register_strategy("grid", self.grid_strategy)
            self.strategy_coordinator.register_strategy("trend", self.trend_strategy)
            self.strategy_coordinator.register_strategy("scalping", self.scalping_strategy)
            self.strategy_coordinator.register_strategy("arbitrage", self.arbitrage_strategy)
            self.strategy_coordinator.register_strategy("spot_grid", self.spot_grid_strategy)
            self.strategy_coordinator.register_strategy("spot_martingale", self.spot_martingale_strategy)
            logger.info("All strategies registered to StrategyCoordinator (fallback)")

        # ── 策略协同增强组件 ──
        # 信号聚合中心
        self.signal_hub = SignalAggregationHub(config, self.strategy_coordinator)
        logger.info("SignalAggregationHub initialized for multi-strategy signal consolidation")

        # 优先级调度器
        self.priority_scheduler = PriorityScheduler(config)
        logger.info("PriorityScheduler initialized for strategy execution ordering")

        # 熔断联动器
        self.breaker_linker = CircuitBreakerLinker(config, self.strategy_coordinator)
        logger.info("CircuitBreakerLinker initialized for strategy-circuit-breaker coordination")

        # 策略健康监控器
        self.health_monitor = StrategyHealthMonitor(config)
        logger.info("StrategyHealthMonitor initialized for real-time strategy health tracking")

        # 链接到协调器
        self.strategy_coordinator.link_signal_hub(self.signal_hub)
        self.strategy_coordinator.link_priority_scheduler(self.priority_scheduler)
        self.strategy_coordinator.link_breaker(self.breaker_linker)
        self.strategy_coordinator.link_health_monitor(self.health_monitor)
        logger.info("Strategy coordinator enhanced components linked successfully")

        # P1-⑥：熔断统一 —— 将策略协调层 CircuitBreakerLinker 注入 GlobalRiskControl，
        # 使风控熔断触发/恢复能同步驱动策略级阻断（消除两套熔断写路径不统一）。
        self.global_risk.set_breaker_linker(self.breaker_linker)

        self.decision_coordinator = DecisionCoordinator(config)
        self.decision_coordinator.set_account_manager(self.account_manager)
        logger.info("DecisionCoordinator initialized for decision orchestration")

        self.rule_engine = RuleBasedEngine(config)
        # 加载预置规则模板
        rule_cfg = config.get("rule_engine", {})
        if rule_cfg.get("load_default_templates", True):
            templates = self.rule_engine.load_default_templates()
            logger.info(f"RuleBasedEngine loaded {len(templates)} default rule templates")
        logger.info("RuleBasedEngine initialized for rule-based decision making")

        # 动态注册集成决策源（替代硬编码6行）
        self.ensemble_maker = EnsembleDecisionMaker()
        enabled_names = self.strategy_manager.get_enabled_strategy_names() if self.strategy_manager else [
            "grid", "trend", "scalping", "arbitrage", "spot_grid", "spot_martingale"
        ]
        for name in enabled_names:
            self.ensemble_maker.register_source(name, weight=1.0, reliability=0.5)
        logger.info(f"EnsembleDecisionMaker initialized with {len(enabled_names)} signal sources")

        self.decision_validator = DecisionValidator(okx_client=self.okx_client)
        self.decision_validator.set_risk_limits({
            "max_position_size": float(config.get("risk", {}).get("max_position_size", 1000)),
            "max_leverage": int(config.get("risk", {}).get("max_leverage", 20)),
            "max_exposure": float(config.get("risk", {}).get("max_exposure", 1.0)),
            "min_confidence": float(config.get("risk", {}).get("min_confidence", 0.15)),
        })
        logger.info("DecisionValidator initialized with risk limits")

        self.decision_evaluator = DecisionQualityEvaluator()
        logger.info("DecisionQualityEvaluator initialized for performance tracking")

        # 企业级智能交易记录分析智能体（市场状态/策略方向/信号质量/ADX确认/分析与优化方向报告）
        self.analysis_agent = IntelligentAnalysisAgent(
            config=config,
            trade_journal=self.trade_journal,
            sqlite_storage=self.sqlite_storage,
            market_regime_engine=self.market_regime_engine,
            decision_quality_evaluator=self.decision_evaluator,
            okx_client=self.okx_client,
            historical_analyzer=self.analyzer,
            contribution_analyzer=self.contribution_analyzer,
            strategy_optimizer=self.optimizer,
        )
        logger.info("IntelligentAnalysisAgent initialized for trading record analysis")

        self.confidence_calibrator = ConfidenceCalibrator()
        self.trade_journal.set_confidence_calibrator(self.confidence_calibrator)
        logger.info("ConfidenceCalibrator initialized for probability calibration")

        # ── P0: 智能决策核心引擎 ──
        self.intelligent_decision_engine = get_intelligent_decision_engine(config)
        self.intelligent_decision_engine.set_ensemble_maker(self.ensemble_maker)
        if self.risk_gate:
            self.intelligent_decision_engine.set_risk_gate(self.risk_gate)
        if self.portfolio_optimizer:
            self.intelligent_decision_engine.set_portfolio_optimizer(self.portfolio_optimizer)
        # 设置审计持久化路径
        audit_path = os.path.join(
            config.get("data_dir", "./data"),
            config.get("decision", {}).get("audit_file", "decision_audit_chain.json")
        )
        self.intelligent_decision_engine.set_audit_persist_path(audit_path)
        # 加载历史审计链
        self.intelligent_decision_engine.load_audit_chain()
        logger.info("IntelligentDecisionEngine initialized with audit chain support")

        # ── P0: 机器学习决策引擎 ──
        ml_cfg = config.get("ml_decision", {})
        self.ml_decision_engine = MLDecisionEngine(config) if ml_cfg.get("enabled", True) else None
        if self.ml_decision_engine:
            logger.info(
                f"MLDecisionEngine initialized: model={self.ml_decision_engine._model_id}, "
                f"enabled={self.ml_decision_engine._enabled}"
            )
        else:
            logger.info("MLDecisionEngine disabled per config")

        # ── P0: 强化学习智能体 ──
        rl_cfg = config.get("rl_agent", {})
        self.rl_agent = get_rl_agent(config) if rl_cfg.get("enabled", True) else None
        if self.rl_agent:
            if self.agent_learning_memory is not None:
                self.rl_agent.set_learning_memory(self.agent_learning_memory)
            logger.info(
                f"TradingRLAgent '{self.rl_agent.name}' initialized: "
                f"mode={self.rl_agent.mode.value}, epsilon={self.rl_agent._epsilon:.3f}"
            )
            # P0: 注入 MarketRegimeEngine 到 RL 智能体，让 StateEncoding 状态编码
            # 融合真实多因子市场状态，消除调用方 naive 状态导致的状态漂移。
            if hasattr(self, 'market_regime_engine') and self.market_regime_engine:
                self.rl_agent.set_regime_engine(self.market_regime_engine)
                logger.info("MarketRegimeEngine injected into TradingRLAgent")
            # 注入 RL Agent 到 OrderExecutor，平仓时回写 reward + MAB 评分
            if hasattr(self, 'order_executor') and self.order_executor:
                self.order_executor.set_rl_agent(self.rl_agent)
        else:
            logger.info("TradingRLAgent disabled per config")

        self.pipeline_orchestrator = PipelineOrchestrator(config)
        logger.info("PipelineOrchestrator initialized for trading pipeline management")

        self.signal_pipeline = SignalProcessingPipeline(
            config,
            quality_engine=self.signal_quality_engine,
            regime_engine=self.market_regime_engine,
            strategy_coordinator=self.strategy_coordinator,
        )
        logger.info("SignalProcessingPipeline initialized for signal processing")

        self.decision_executor = DecisionExecutor(
            config,
            order_executor=self.order_executor,
            decision_validator=self.decision_validator,
            confidence_calibrator=self.confidence_calibrator,
        )
        logger.info("DecisionExecutor initialized for decision execution")

        self.anomaly_detector = AnomalyDetector(config)
        self.anomaly_detector.register_detector(PriceSpikeDetector(config))
        self.anomaly_detector.register_detector(VolumeSurgeDetector(config))
        self.anomaly_detector.register_detector(OrderFailureRateDetector(config))
        self.anomaly_detector.register_detector(LatencySpikeDetector(config))
        self.anomaly_detector.register_detector(SignalFrequencyDetector(config))
        self.anomaly_detector.register_detector(PNLDropDetector(config))
        logger.info("AnomalyDetector initialized with 6 anomaly detectors")

        self.recovery_handler = RecoveryHandler(config)
        logger.info("RecoveryHandler initialized for automated recovery")

        # ── 运维自愈闭环协调器（监控告警 → 根因分析 → 自动恢复 → 反馈）──
        # 新增自动恢复行为，通过 config["ops_self_heal"]["enabled"]=true 显式开启（默认关闭）。
        ops_self_heal_cfg = config.get("ops_self_heal") or {}
        self.ops_self_heal = None
        if ops_self_heal_cfg.get("enabled", False):
            from core.ops_self_heal import OpsSelfHealCoordinator
            self.ops_self_heal = OpsSelfHealCoordinator(
                anomaly_detector=self.anomaly_detector,
                recovery_handler=self.recovery_handler,
                alert_manager=self.alert_manager,
                config=ops_self_heal_cfg,
            )
            logger.info("OpsSelfHealCoordinator initialized (ops_self_heal.enabled=true)")
        else:
            logger.debug("OpsSelfHealCoordinator disabled (ops_self_heal.enabled=false)")

        # ── P0: 订单生命周期管理器 ──
        self.order_lifecycle = OrderLifecycleManager(config)
        self.order_lifecycle.set_timeout_order_handler(
            self.order_executor._confirm_lifecycle_timeout
        )
        self.order_lifecycle.set_alert_manager(self.alert_manager)
        logger.info("OrderLifecycleManager initialized for order lifecycle tracking")

        # ── P0: 执行监控器 ──
        self.execution_monitor = ExecutionMonitor(config)
        logger.info("ExecutionMonitor initialized for real-time latency/throughput/SLA tracking")

        # ── P0: 算法订单执行系统 ──
        algo_cfg = config.get("algo_orders", {})
        if algo_cfg.get("enabled", True):
            # 智能订单路由器（注册默认场所 + 注入OKX客户端用于实时行情）
            self.smart_order_router = SmartOrderRouter(config)
            self.smart_order_router.setup_default_venues(okx_client=self.okx_client)
            logger.info("SmartOrderRouter initialized with default venues and real-time market data capability")

            # 算法执行器
            self.twap_executor = TWAPExecutor(config)
            self.vwap_executor = VWAPExecutor(config)
            self.vwap_executor.set_okx_client(self.okx_client)  # 注入OKX客户端用于拉K线构建成交量分布
            self.iceberg_executor = IcebergOrderExecutor(config)
            self.dark_pool_router = DarkPoolRouter(config)
            logger.info("Algo executors initialized: TWAP, VWAP, Iceberg, DarkPool")

            # 算法执行引擎（注册执行器）
            self.algo_execution_engine = AlgoExecutionEngine(config)
            self.algo_execution_engine.register_executor(AlgoOrderType.TWAP, self.twap_executor)
            self.algo_execution_engine.register_executor(AlgoOrderType.VWAP, self.vwap_executor)
            self.algo_execution_engine.register_executor(AlgoOrderType.ICEBERG, self.iceberg_executor)
            # 注入外部依赖
            self.algo_execution_engine.set_order_executor(self.order_executor)
            self.algo_execution_engine.set_okx_client(self.okx_client)
            logger.info("AlgoExecutionEngine initialized with 3 executors registered")

            # 执行质量监控器
            self.execution_quality_monitor = EnhancedExecutionQualityMonitor(config)
            # 注入到订单执行器，实现自动化执行质量记录
            self.order_executor.set_execution_quality_monitor(self.execution_quality_monitor)
            logger.info("ExecutionQualityMonitor initialized and hooked into OrderExecutor")
        else:
            self.smart_order_router = None
            self.algo_execution_engine = None
            self.execution_quality_monitor = None
            logger.info("Algo orders system disabled")

        # ── 关联恢复处理器与熔断器 ──
        self.recovery_handler.set_circuit_breaker(self.emergency_cb if hasattr(self, 'emergency_cb') else None)
        # 注入告警管理与 OKX 客户端依赖，使自愈动作（重连/重试/熔断）可真实执行
        self.recovery_handler.set_alert_manager(self.alert_manager)
        self.recovery_handler.set_okx_client(self.okx_client)

        self.pipeline_orchestrator.register_stage_handlers({
            PipelineStage.SIGNAL_RECEIVE: [self._handle_signal_receive],
            PipelineStage.SIGNAL_VALIDATION: [self._handle_signal_validation],
            PipelineStage.DECISION_MAKING: [self._handle_decision_making],
            PipelineStage.ORDER_GENERATION: [self._handle_order_generation],
            PipelineStage.ORDER_PLACEMENT: [self._handle_order_placement],
            PipelineStage.EXECUTION_MONITORING: [self._handle_execution_monitoring],
            PipelineStage.SETTLEMENT: [self._handle_settlement],
        })
        logger.info("Pipeline stage handlers registered")

        # 注入事件总线到流水线编排器，实现生命周期事件溯源（started/completed/failed/timeout）
        self.pipeline_orchestrator.set_event_bus(self.unified_layer.event_bus)
        logger.info("EventBus injected into PipelineOrchestrator for lifecycle event sourcing")

        self.online_learner = OnlineLearner(config)
        logger.info("OnlineLearner initialized for real-time learning")

        self.parameter_adaptor = ParameterAdaptor(config)
        logger.info("ParameterAdaptor initialized for adaptive parameter adjustment")

        self.strategy_evolver = StrategyEvolver(config)
        logger.info("StrategyEvolver initialized for strategy evolution")

        self.market_regime_detector = MarketRegimeDetector(config)
        logger.info("MarketRegimeDetector initialized for adaptive market state detection")
        self.stop_loss_manager.set_market_regime_detector(self.market_regime_detector)

        # RegimeArbiter：融合主引擎+检测器输出，统一注入所有下游（含止损）
        try:
            from services.regime_arbiter import RegimeArbiter
            self.regime_arbiter = RegimeArbiter(
                main_engine=self.market_regime_engine,
                detector=self.market_regime_detector,
                config=config.get("regime_arbiter", {}),
            )
            self.market_regime_engine.set_regime_arbiter(self.regime_arbiter)
            self.market_regime_engine.set_detector(self.market_regime_detector)
            self.stop_loss_manager.set_regime_arbiter(self.regime_arbiter)
            logger.info("RegimeArbiter initialized and injected into main engine and StopLossManager")
        except Exception as e:
            self.regime_arbiter = None
            self.market_regime_engine.set_detector(self.market_regime_detector)
            logger.warning(f"RegimeArbiter assembly failed (degraded): {e}")

        self.knowledge_base = KnowledgeBase(config)
        logger.info("KnowledgeBase initialized for trading experience storage")

        self.meta_learner = MetaLearner(config)
        logger.info("MetaLearner initialized for cross-strategy learning transfer")

        self.performance_feedback = PerformanceFeedback(config)
        logger.info("PerformanceFeedback initialized for strategy performance evaluation")

        # ── P0: 统一参数优化编排器 ──
        param_opt_cfg = config.get("parameter_optimization", {})
        if param_opt_cfg.get("enabled", True):
            self.param_optimizer = ParameterOptimizationOrchestrator(config)
            # 默认占位适应度函数，实际优化时由 _run_param_optimization 注入真实回测评估函数
            self.param_optimizer.set_fitness_fn(
                lambda params: 0.0  # placeholder, replaced at runtime
            )
            self.backtest_engine = BacktestEngine(config)
            logger.info(
                f"ParameterOptimizationOrchestrator initialized: "
                f"strategy={self.param_optimizer._default_strategy.value}"
            )
        else:
            self.param_optimizer = None
            self.backtest_engine = None
            logger.info("ParameterOptimizationOrchestrator disabled per config")

        # ── 自动寻优闭环协调器（参数优化 → 回测 → 上线 → 反馈调参）──
        self.auto_optimization = None
        try:
            from core.auto_optimization_loop import AutoOptimizationCoordinator
            self.auto_optimization = AutoOptimizationCoordinator(
                param_optimizer=self.param_optimizer,
                optimizer=self.optimizer,
                performance_feedback=self.performance_feedback,
                recommendation_builder=self._build_param_opt_recommendation,
                config=config.get("auto_optimization", {}),
            )
            logger.info("AutoOptimizationCoordinator initialized for auto optimization closed loop")
        except Exception as e:
            self.auto_optimization = None
            logger.warning(f"AutoOptimizationCoordinator assembly failed (degraded): {e}")

        # P0: 将知识库链接到在线学习器，实现经验闭环
        self.online_learner.set_knowledge_base(self.knowledge_base)
        logger.info("KnowledgeBase linked to OnlineLearner")

        self.signal_processor = SignalProcessor(
            config,
            global_risk=self.global_risk,
            strategy_risk=self.strategy_risk,
            order_executor=self.order_executor,
            alert_manager=self.alert_manager,
            trade_journal=self.trade_journal,
            adaptive_controller=self.adaptive_controller,
            profit_optimizer=self.profit_optimizer,
            account_manager=self.account_manager,
        )
        # 开仓全链路拒单溯源：注入事件总线，信号层丢弃时发布 SIGNAL_REJECTED
        self.signal_processor.set_event_bus(self.unified_layer.event_bus)
        logger.info("SignalProcessor initialized with service layer")

        self.signal_normalizer = SignalNormalizer(config)
        logger.info("SignalNormalizer initialized for signal standardization")

        self.signal_monitor = SignalMonitor(config)
        logger.info("SignalMonitor initialized for signal lifecycle tracking")

        self.signal_processor.set_regime_engine(self.market_regime_engine)
        self.signal_processor.set_quality_engine(self.signal_quality_engine)
        self.signal_processor.set_coordinator(self.strategy_coordinator)
        self.signal_processor.set_normalizer(self.signal_normalizer)

        # P1: L0 市场状态总门控 - 在信号进入审计前按 regime 硬过滤不匹配开仓信号
        from services.regime_gate import RegimeGate
        self.regime_gate = RegimeGate(
            self.market_regime_engine,
            whitelist_provider=lambda: self.intelligent_agent.get_whitelisted_symbols(),
            config=config.get("regime_gate", {}),
        )
        self.signal_processor.set_regime_gate(self.regime_gate)
        logger.info("RegimeGate (L0) initialized and injected into SignalProcessor")

        # 感知侧闭环：Regime识别门控 + 信号质量 统一感知决策（串成信号级感知闭环）
        self.signal_perception_loop = None
        try:
            from core.signal_perception_loop import SignalPerceptionCoordinator
            self.signal_perception_loop = SignalPerceptionCoordinator(
                regime_engine=self.market_regime_engine,
                regime_gate=self.regime_gate,
                quality_engine=self.signal_quality_engine,
                config=config,
            )
            self.signal_processor.set_perception_loop(self.signal_perception_loop)
            logger.info("SignalPerceptionCoordinator (perception loop) initialized and injected into SignalProcessor")
        except Exception as e:
            self.signal_perception_loop = None
            logger.warning(f"SignalPerceptionCoordinator assembly failed (degraded): {e}")

        self.signal_processor.set_decision_components(
            decision_coordinator=self.decision_coordinator,
            rule_engine=self.rule_engine,
            ensemble_maker=self.ensemble_maker,
            decision_validator=self.decision_validator,
            decision_evaluator=self.decision_evaluator,
            confidence_calibrator=self.confidence_calibrator,
        )
        self.adaptive_controller.set_regime_engine(self.market_regime_engine)

        # P0: 统一自适应止损止盈引擎（融合波动率+市场状态+策略表现+盈亏，收敛分散 TP/SL 逻辑）
        self.adaptive_tp_sl_engine = AdaptiveTpSlEngine(config, regime_engine=self.market_regime_engine)
        self.adaptive_controller.set_adaptive_tp_sl_engine(self.adaptive_tp_sl_engine)
        self.order_executor.set_adaptive_tp_sl_engine(self.adaptive_tp_sl_engine)
        logger.info("AdaptiveTpSlEngine initialized and injected into AdaptiveController + OrderExecutor")

        # P0: 统一自适应仓位引擎（融合 Kelly + 权益模式 + 信号 + 波动率 + 策略资金占比 + tier + 杠杆）
        self.adaptive_position_sizer = AdaptivePositionSizer(config)
        self.adaptive_controller.set_adaptive_position_sizer(self.adaptive_position_sizer)
        logger.info("AdaptivePositionSizer initialized and injected into AdaptiveController")

        # P0: 把统一仓位引擎接入下单链路（开仓信号统一计算仓位，收敛分散口径）
        self.signal_processor.set_position_sizing_provider(self._compute_unified_position_size)
        logger.info("AdaptivePositionSizer wired into SignalProcessor order chain (unified sizing)")

        self.signal_processor.set_capital_allocator(self.capital_adaptive_allocator)
        logger.info("CapitalAdaptiveAllocator wired into SignalProcessor (focused allocation gate)")

        logger.info("Smart decision engines and decision system components injected into SignalProcessor and AdaptiveController")

        self.signal_processor.set_pipeline_components(
            pipeline_orchestrator=self.pipeline_orchestrator,
            signal_pipeline=self.signal_pipeline,
            anomaly_detector=self.anomaly_detector,
            recovery_handler=self.recovery_handler,
        )
        logger.info("Trading pipeline components injected into SignalProcessor")

        # P0: 注入执行监控器到信号处理器，实现端到端延迟/SLA 追踪埋点
        self.signal_processor.set_execution_monitor(self.execution_monitor)
        logger.info("ExecutionMonitor injected into SignalProcessor")

        # P0: 注入智能决策引擎到信号处理器（决策上下文丰富化、审计记录、异常检测）
        self.signal_processor.set_intelligent_decision_engine(self.intelligent_decision_engine)
        logger.info("IntelligentDecisionEngine injected into SignalProcessor")

        # P0: 注入 ML 决策引擎到信号处理器（实时置信度 soft 修正，fail-open）
        self.signal_processor.set_ml_decision_engine(self.ml_decision_engine)
        logger.info("MLDecisionEngine injected into SignalProcessor (real-time confidence correction)")

        # P0: 注入五层风控拦截器到信号链路（L5→L4→L1→L2→L3 串行校验）
        self.signal_processor.set_risk_gate(self.risk_gate)
        logger.info("RiskGate (5-layer serial risk control) injected into SignalProcessor")

        # P2: 注入独立风控裁决器到信号链路（信号级裁决 + traceID 生成贯穿全链路）
        self.signal_processor.set_risk_adjudicator(self.risk_adjudicator)
        logger.info("RiskAdjudicator injected into SignalProcessor for signal-level adjudication")

        # P1: 注入相关性风控到信号链路（开仓前检查同向高相关集中度）
        self.signal_processor.set_correlation_risk(self.correlation_risk)
        logger.info("CorrelationRiskControl injected into SignalProcessor for pre-trade correlation check")

        # P1: 注入组合再平衡器到信号链路（总敞口硬限制门控）
        self.signal_processor.set_portfolio_rebalancer(self.portfolio_rebalancer)
        logger.info("PortfolioRebalancer injected into SignalProcessor for exposure limit check")

        self.trading_recovery = TradingRecoveryService(config, okx_client=self.okx_client)
        logger.info("TradingRecoveryService initialized for automatic trading recovery")

        # 初始化P2级监控告警组件
        self.notification_manager = NotificationManager(config.get("notifications", {}))
        self.alert_rule_engine = AlertRuleEngine(config, self.alert_manager)
        self.health_scorer = HealthScorer(config)
        self.auto_recovery = AutoRecovery(config, self.alert_manager)
        # 注入依赖：strategy_coordinator 使 RESET_STATE 动作能真正恢复策略状态，
        # okx_client 供持仓一致性检查使用；缺失会导致这些恢复动作静默失效
        self.auto_recovery.set_dependencies(
            strategy_coordinator=self.strategy_coordinator,
            okx_client=self.okx_client,
            redis_cache=self.redis_cache,
        )
        self.metrics_collector = get_metrics_collector()
        logger.info("P2 monitoring components initialized: AlertRuleEngine, HealthScorer, AutoRecovery, MetricsCollector")

        # P0: 算力动态调度器（高波动提升优先级，低波动/CPU过载降频，适配本地7×24小时运行）
        self.compute_scheduler = get_compute_scheduler(config)
        # 注入到 StrategyEngine（统一分发到 IndicatorEngine + StrategyContainer）
        try:
            self.strategy_engine.set_compute_scheduler(self.compute_scheduler)
        except Exception as e:
            logger.warning(f"Failed to inject ComputeScheduler into StrategyEngine: {e}")
        # 为所有交易对注册到调度器
        try:
            for sym in get_all_symbols(config):
                self.compute_scheduler.register_symbol(sym)
        except Exception as e:
            logger.debug(f"ComputeScheduler symbol registration error: {e}")
        logger.info("ComputeScheduler initialized and injected into StrategyEngine")

        # P0: 循环周期可调（CPU过载降频时通过 set_loop_intervals 动态延长，保护本地设备）
        loop_cfg = config.get("scheduler_intervals", {})
        self._loop_intervals = {
            "position_risk_monitor": loop_cfg.get("position_risk_monitor", 5),       # 持仓风控（实时性要求高）
            "health_scoring": loop_cfg.get("health_scoring", 30),
            "auto_recovery": loop_cfg.get("auto_recovery", 60),
            "monitoring": loop_cfg.get("monitoring", 300),
            "analysis": loop_cfg.get("analysis", 3600),
            "optimization": loop_cfg.get("optimization", 86400),
            "verification": loop_cfg.get("verification", 86400 * 7),
            "learning": loop_cfg.get("learning", 300),
            "compute_feedback": loop_cfg.get("compute_feedback", 10),  # CPU/算力反馈循环
            "contract_risk_scan": loop_cfg.get("contract_risk_scan", 30),  # 合约风险扫描循环
            "decision_coordinator": loop_cfg.get("decision_coordinator", 5),  # 决策协调器处理循环（5秒足够，降低CPU）
            "algo_orders_maintenance": loop_cfg.get("algo_orders_maintenance", 60),  # 算法订单维护循环（场所行情刷新/成交量分布更新）
            "compliance_check": loop_cfg.get("compliance_check", 3600),  # P22-8: 合规红线检查循环
            "intelligent_analysis": loop_cfg.get("intelligent_analysis", 21600),  # 智能交易记录分析（6小时）
            "param_optimization": loop_cfg.get("param_optimization", 86400 * 7),  # 参数优化循环（周）
            "param_optimization_first_delay": loop_cfg.get("param_optimization_first_delay", 3600),  # 参数优化首跑延迟（重启后1小时）
            "agi_orchestrator": loop_cfg.get("agi_orchestrator", 300),  # 量化AGI自治协调器循环
            "ops_self_heal": loop_cfg.get("ops_self_heal", 60),  # 运维自愈闭环循环
            "top_level_agi": loop_cfg.get("top_level_agi", 60),  # 顶层AGI跨闭环联动循环
        }
        logger.info(f"Scheduler loop intervals configured: {self._loop_intervals}")

        # P22-8: 初始化生产级红线合规检查器
        self.compliance_checker = ComplianceChecker(
            config,
            okx_client=self.okx_client,
            trade_journal=self.trade_journal,
            account_manager=self.account_manager,
            redis_cache=self.redis_cache,
            scheduler=self,
        )
        logger.info("P22-8: ComplianceChecker initialized in scheduler")

        # P3-3: 自动恢复时间戳（防止重复触发）
        self._last_auto_recovery_ts = datetime.min

        self._bind_cpu_cores()

        # ── 统一自治协调器（可选装配，默认关闭）──
        # 编排已有的 regime/contribution/capital/dynamic 组件形成资金侧自治闭环。
        # 通过 config["agi_orchestrator"]["enabled"]=true 显式开启，不影响现有稳定运行。
        agi_cfg = config.get("agi_orchestrator") or {}
        self.agi_orchestrator = None
        self.restricted_execution_channel = None
        if agi_cfg.get("enabled", False):
            try:
                from core.quant_agi_orchestrator import QuantAGIOrchestrator
                from core.restricted_execution_channel import RestrictedExecutionChannel
                self.agi_orchestrator = QuantAGIOrchestrator(
                    config=config,
                    regime_engine=getattr(self, "market_regime_engine", None),
                    contribution_analyzer=getattr(self, "contribution_analyzer", None),
                    capital_allocator=getattr(self, "capital_adaptive_allocator", None),
                    dynamic_allocator=getattr(self, "dynamic_allocator", None),
                    account_manager=getattr(self, "account_manager", None),
                    equity_monitor=getattr(self, "equity_monitor", None),
                    strategy_correlation=getattr(self, "strategy_correlation", None),
                    regime_arbiter=getattr(self, "regime_arbiter", None),
                    rl_agent=getattr(self, "rl_agent", None),
                )
                self.restricted_execution_channel = RestrictedExecutionChannel(
                    event_store=getattr(self, "event_store", None),
                    alert_manager=getattr(self, "alert_manager", None),
                    pending_path=agi_cfg.get("pending_path", "data/agi_pending_actions.json"),
                    idle_cash_deployer=self._idle_cash_deployer,
                    autonomous=bool(agi_cfg.get("autonomous", False)),
                    reallocate_deployer=self._reallocate_deployer,
                    kill_switch_check=self._is_kill_switch_active,
                    close_position_deployer=self._close_position_deployer,
                    param_adjust_deployer=self._param_adjust_deployer,
                    strategy_pause_deployer=self._strategy_pause_deployer,
                    strategy_resume_deployer=self._strategy_resume_deployer,
                    always_require_confirmation=agi_cfg.get("always_require_confirmation", ["reallocate"]),
                )
                logger.info("QuantAGIOrchestrator assembled (agi_orchestrator.enabled=true)")
            except Exception as e:
                self.agi_orchestrator = None
                self.restricted_execution_channel = None
                logger.warning(f"QuantAGIOrchestrator assembly failed (degraded): {e}")
        else:
            logger.debug("QuantAGIOrchestrator disabled (agi_orchestrator.enabled=false)")

        if self.agi_orchestrator is not None:
            self.signal_processor.set_bear_case_open_gate(
                self.agi_orchestrator.get_bear_case_open_block_reason
            )
            logger.info("AGI bear-case opening gate injected into SignalProcessor")
            try:
                from dashboard_api import inject_dashboard_dependencies
                inject_dashboard_dependencies(
                    okx_client=self.okx_client,
                    capital_manager=self.capital_manager,
                    state_manager=self.state_manager,
                    account_manager=self.account_manager,
                    strategy_manager=self.strategy_manager,
                    agi_orchestrator=self.agi_orchestrator,
                )
            except Exception as e:
                logger.warning(f"DashboardEngine dependency injection failed (degraded): {e}")

        # ── 顶层 AGI 编排器（跨闭环联动：聚合四闭环状态 → 联动诊断）──
        # 通过 config["top_level_agi"]["enabled"]=true 显式开启（默认关闭）。
        top_agi_cfg = config.get("top_level_agi") or {}
        self.top_level_agi = None
        if top_agi_cfg.get("enabled", False):
            try:
                from core.top_level_agi import TopLevelAGICoordinator
                self.top_level_agi = TopLevelAGICoordinator(
                    quant_agi=getattr(self, "agi_orchestrator", None),
                    perception=getattr(self, "signal_perception_loop", None),
                    ops_self_heal=getattr(self, "ops_self_heal", None),
                    auto_optimization=getattr(self, "auto_optimization", None),
                    learning_memory=self.agent_learning_memory,
                    config=top_agi_cfg,
                )
                logger.info("TopLevelAGICoordinator assembled (top_level_agi.enabled=true)")
            except Exception as e:
                self.top_level_agi = None
                logger.warning(f"TopLevelAGICoordinator assembly failed (degraded): {e}")
        else:
            logger.debug("TopLevelAGICoordinator disabled (top_level_agi.enabled=false)")

    def _has_live_strategy(self) -> bool:
        """探测是否有实盘策略正在运行（StrategyContainer 的 RUNNING/PAUSED 实例）。"""
        container = getattr(getattr(self, "strategy_engine", None), "_strategy_container", None)
        if container is None:
            return False
        try:
            return bool(container.get_active_instances())
        except Exception:
            # 探测异常时保守返回 True，宁可跳过清理也不误删
            return True
    
    def _setup_signal_routing(self):
        strategies = [
            self.grid_strategy, self.trend_strategy,
            self.scalping_strategy, self.arbitrage_strategy,
            self.spot_grid_strategy, self.spot_martingale_strategy
        ]
        self.signal_processor.setup_signal_routing(strategies, self.redis_cache)
        logger.info("Signal routing configured via SignalProcessor service")
    
    def _register_task(self, name: str, coro) -> asyncio.Task:
        """
        注册并创建后台任务，纳入任务注册表以便优雅关闭时统一取消。
        
        Args:
            name: 任务名称（需唯一）
            coro: 协程对象
        
        Returns:
            创建的 asyncio.Task 对象
        """
        task = asyncio.create_task(coro)
        self._tasks[name] = task

        def _on_task_done(t: asyncio.Task):
            self._tasks.pop(name, None)
            # P0 修复：检查任务是否因未处理异常而结束，避免协程异常被静默吞掉。
            # 原实现仅弹字典条目，从不调用 t.exception()，异常既不告警也不恢复。
            if t.cancelled():
                return
            exc = t.exception()
            if exc is not None:
                logger.error(
                    f"Background task '{name}' raised unhandled exception: "
                    f"{type(exc).__name__}: {exc}"
                )

        task.add_done_callback(_on_task_done)
        return task

    async def _cancel_all_tasks(self) -> None:
        """取消所有已注册的后台任务"""
        if not self._tasks:
            return
        logger.info(f"Canceling {len(self._tasks)} registered background tasks...")
        for name, task in list(self._tasks.items()):
            if not task.done():
                task.cancel()
        # 等待所有任务被取消
        results = await asyncio.gather(*[t for t in self._tasks.values()], return_exceptions=True)
        cancelled = sum(1 for r in results if isinstance(r, asyncio.CancelledError))
        remaining = sum(1 for r in results if not isinstance(r, asyncio.CancelledError))
        logger.info(f"Background tasks: {cancelled} cancelled, {remaining} completed normally")

    def _bind_cpu_cores(self):
        try:
            process = psutil.Process()
            high_priority_cores = self.config["hardware"]["high_priority_cores"]
            
            process.cpu_affinity(high_priority_cores)
            logger.info(f"Process bound to CPU cores: {high_priority_cores}")
        except Exception as e:
            logger.warning(f"Failed to bind CPU cores: {e}")
    
    def _emergency_close_all(self, reason: str) -> None:
        """紧急熔断：一键全仓平仓（逐币种遍历，避免传递"ALL"到OKX API）"""
        logger.critical(f"!!! EMERGENCY CLOSE ALL triggered: {reason} !!!")
        try:
            # 获取所有持仓
            positions = self.okx_client.get_positions() if hasattr(self.okx_client, 'get_positions') else []
            if not positions:
                logger.warning("No positions to close in emergency")
                return

            # 逐币种平仓，不传递"ALL"符号
            for pos_data in positions:
                try:
                    position = self.okx_client._parse_position(pos_data)
                    if not position or float(position.quantity) == 0:
                        continue

                    side = "sell" if position.side == "long" else "buy"
                    close_qty = self.okx_client.contracts_to_coins(position.symbol, abs(float(position.quantity)))

                    # 通过订单执行器平仓（走五层风控校验）
                    if hasattr(self.order_executor, 'handle_signal'):
                        close_signal = {
                            "symbol": position.symbol,
                            "strategy_name": "system",
                            "signal_type": "emergency_close",
                            "direction": side,
                            "price": float(position.mark_price),
                            "quantity": close_qty,
                            "leverage": position.leverage,
                            "confidence": 1.0,
                            "reduce_only": True,
                            "pos_side": position.side,
                            "reason": f"emergency: {reason}",
                            "timestamp": datetime.now().isoformat(),
                            "priority": 999
                        }
                        self._register_task(
                            f"emergency_close_{position.symbol}",
                            self.order_executor.handle_signal(close_signal)
                        )

                    logger.critical(f"Emergency close dispatched: {position.symbol} qty={close_qty:.4f}")
                except Exception as e:
                    logger.error(f"Error closing {pos_data.get('instId', 'unknown')} in emergency: {e}")

            logger.critical(f"Emergency close_all dispatched: {len(positions)} positions")
        except Exception as e:
            logger.error(f"Error in emergency close_all: {e}")
    
    def _emergency_stop_strategies(self, reason: str) -> None:
        """紧急熔断：停止所有策略"""
        logger.critical(f"!!! EMERGENCY STOP ALL STRATEGIES: {reason} !!!")
        try:
            # 冻结策略引擎中所有策略实例
            if hasattr(self, 'strategy_engine'):
                container = self.strategy_engine._strategy_container
                for instance in container.get_all_instances():
                    container.freeze_instance(instance.id, f"emergency: {reason}")

            logger.critical("All strategies frozen due to emergency")
        except Exception as e:
            logger.error(f"Error in emergency stop_strategies: {e}")

    def set_loop_intervals(self, intervals: Dict[str, int]) -> None:
        """
        运行时动态调整循环周期（CPU过载降频 / 高波动加速）

        Args:
            intervals: {"position_risk_monitor": 5, "health_scoring": 30, ...}
                       未提供的键保持原值
        """
        if not hasattr(self, '_loop_intervals'):
            self._loop_intervals = {}
        for k, v in intervals.items():
            if v and v > 0:
                self._loop_intervals[k] = v
        logger.debug(f"Loop intervals updated: {self._loop_intervals}")

    @staticmethod
    def _build_stress_test_engine(config: Dict[str, Any]):
        """惰性构建压力测试引擎（供策略上线门禁使用）。"""
        try:
            from backtest.stress_test import StressTestEngine
            return StressTestEngine(config)
        except Exception as e:
            logger.warning(f"Scheduler: failed to build StressTestEngine: {e}")
            return None

    def _get_loop_interval(self, name: str, default: int) -> int:
        """安全读取循环周期（成员变量不存在时回退到默认值）
        
        P7: CPU过载时自动延长非实时循环周期，减轻CPU压力
        """
        base = default
        if hasattr(self, '_loop_intervals'):
            base = self._loop_intervals.get(name, default)
        
        # P7: CPU过载时延长非实时循环
        cpu_overloaded = getattr(self.compute_scheduler, '_global_throttle_active', False) if hasattr(self, 'compute_scheduler') else False
        if cpu_overloaded:
            # 实时循环（position_risk_monitor, decision_coordinator）不受影响
            non_realtime = {'health_scoring', 'auto_recovery', 'monitoring', 'analysis', 
                          'optimization', 'verification', 'learning', 'algo_orders_maintenance',
                          'intelligent_analysis', 'param_optimization'}
            if name in non_realtime:
                return base * 2  # 非实时循环间隔翻倍
        
        return base

    def _get_current_drawdown(self) -> float:
        """P5: 获取当前回撤比例，用于自适应阈值调整"""
        try:
            if hasattr(self, 'profit_optimizer') and self.profit_optimizer:
                return self.profit_optimizer.get_current_drawdown()
            if hasattr(self, 'trade_journal') and self.trade_journal:
                return self.trade_journal.get_current_drawdown()
        except Exception:
            pass
        return 0.0

    def _compute_unified_position_size(self, signal_dict: Dict[str, Any]):
        """统一自适应仓位计算（下单链路仓位口径收敛，供 SignalProcessor 注入调用）。

        把分散在各策略的仓位计算收敛为 AdaptivePositionSizer 单一口径，
        仅针对开仓信号（平仓/减仓由 SignalProcessor 的 is_close_sig 分流跳过）。

        数据源（均为同步读缓存/内存状态）：
          - 权益       account_manager.get_total_equity()
          - 策略表现   trade_journal.get_strategy_stats()
          - 权益乘数   equity_monitor.get_position_multiplier()
          - 市场状态   market_regime_engine.get_regime()
          - 回撤       _get_current_drawdown()

        返回 quantity（>0）或 None；None 表示 fail-open，调用方保持原 quantity。
        """
        try:
            cfg = self.config.get("adaptive_position_sizing", {})
            if not cfg.get("force_order_chain", True):
                return None

            symbol = signal_dict.get("symbol", "")
            strategy_name = signal_dict.get("strategy_name", "") or "grid"
            price = float(signal_dict.get("price", 0) or 0)
            direction = signal_dict.get("direction", "long") or "long"
            if not symbol or price <= 0:
                return None

            # 账户权益（回退 config total_capital）
            account_balance = 0.0
            try:
                account_balance = float(getattr(self.account_manager, 'get_total_equity', lambda: 0)() or 0)
            except Exception:
                account_balance = 0.0
            if account_balance <= 0:
                account_balance = float(self.config.get("trading", {}).get("total_capital", 0) or 0)
            if account_balance <= 0:
                return None

            # 策略历史表现
            stats: Dict[str, Any] = {}
            try:
                stats = self.trade_journal.get_strategy_stats(strategy_name) or {}
            except Exception:
                stats = {}

            # 权益模式乘数
            equity_multiplier = None
            try:
                em = getattr(self, "equity_monitor", None)
                if em is not None and hasattr(em, "get_position_multiplier"):
                    equity_multiplier = em.get_position_multiplier()
            except Exception:
                equity_multiplier = None

            # 市场状态 regime + 波动率
            regime = "unknown"
            volatility = 0.0
            try:
                me = getattr(self, "market_regime_engine", None)
                if me is not None:
                    ro = me.get_regime()
                    if isinstance(ro, dict):
                        regime = ro.get("regime", "unknown") or "unknown"
                        volatility = float(ro.get("volatility", 0) or 0)
            except Exception:
                pass

            # 信号强度（confidence 优先，其次 signal_quality.overall_score）
            signal_strength = float(signal_dict.get("confidence", 0.5) or 0.5)
            sq = signal_dict.get("signal_quality")
            if isinstance(sq, dict) and sq.get("overall_score"):
                try:
                    signal_strength = float(sq["overall_score"])
                except (ValueError, TypeError):
                    pass

            # 回撤
            drawdown = float(self._get_current_drawdown() or 0)

            # 止损距离（从 signal 的 stop_loss_price 推导）
            stop_distance = 0.0
            try:
                sl_price = signal_dict.get("stop_loss_price")
                if sl_price:
                    sl = float(sl_price)
                    if direction in ("short", "sell"):
                        stop_distance = sl - price if sl > price else 0.0
                    else:
                        stop_distance = price - sl if price > sl else 0.0
            except (ValueError, TypeError):
                stop_distance = 0.0

            # 动态策略资金占比（优先 AdaptiveController._dynamic_allocations，
            # 让 DynamicAllocator 三级池 / allocation_shift 的结果真正驱动下单仓位）
            dynamic_allocation = None
            try:
                ac = getattr(self, "adaptive_controller", None)
                if ac is not None and hasattr(ac, "get_allocation"):
                    dynamic_allocation = ac.get_allocation(strategy_name)
            except Exception:
                dynamic_allocation = None

            # 专一分配：极小额账户把策略资金占比收敛到单一策略（聚焦=1.0，其余=0.0）
            try:
                ca = getattr(self, "capital_adaptive_allocator", None)
                if ca is not None and ca.is_focused_mode():
                    dynamic_allocation = ca.get_strategy_allocation(strategy_name)
            except Exception:
                pass

            # 资金利用率引擎仓位放大（boost_aggressive 时 > 1.0），
            # 解决小账户占用不足问题——把 idle-cash boost 传导进统一仓位口径。
            position_boost = 1.0
            try:
                ac = getattr(self, "adaptive_controller", None)
                if ac is not None and hasattr(ac, "get_position_boost"):
                    position_boost = float(ac.get_position_boost() or 1.0)
            except Exception:
                position_boost = 1.0

            result = self.adaptive_position_sizer.compute(
                symbol=symbol,
                price=price,
                account_balance=account_balance,
                strategy_name=strategy_name,
                win_rate=float(stats.get("win_rate", 0.5) or 0.5),
                avg_win=float(stats.get("avg_win", 0) or 0),
                avg_loss=float(stats.get("avg_loss", 0) or 0),
                trade_count=int(stats.get("total_trades", 0) or 0),
                regime=regime,
                drawdown=drawdown,
                signal_strength=signal_strength,
                volatility=volatility,
                equity_multiplier=equity_multiplier,
                stop_distance=stop_distance,
                allocation=dynamic_allocation,
                direction=direction,
                position_boost=position_boost,
            )

            if result.allowed and result.quantity > 0:
                logger.info(
                    f"Unified sizing OK: {symbol} {strategy_name} boost={position_boost:.2f} "
                    f"qty={result.quantity:.6f} notional={result.notional:.2f} margin={result.margin:.2f}"
                )
                return result.quantity
            logger.debug(f"Unified sizing rejected for {symbol}: {result.reject_reason}")
            return None
        except Exception as e:
            logger.debug(f"Unified position sizing provider error (fail-open): {e}")
            return None

    # ---- 复盘迭代回调 ----

    async def _on_daily_review_completed(self, report) -> None:
        """每日复盘完成回调：发送通知"""
        try:
            summary = report.summary
            recs = report.recommendations
            msg = (
                f"📊 每日复盘 [{report.date}]\n"
                f"交易: {summary.get('total_trades', 0)}笔 | "
                f"净盈亏: {summary.get('total_net_pnl', 0):.2f}U | "
                f"有效币种: {summary.get('effective_symbols', 0)}/{summary.get('symbols_reviewed', 0)}\n"
            )
            if recs:
                msg += "建议:\n" + "\n".join(f"  • {r}" for r in recs[:3])
            await self.notification_manager.broadcast(msg, "Daily Review", "INFO")
        except Exception as e:
            logger.debug(f"Daily review notification error: {e}")

    async def _on_monthly_plan_generated(self, plan) -> None:
        """月度迭代计划生成回调：发送通知"""
        try:
            summary = plan.summary
            msg = (
                f"📈 月度迭代计划 [{plan.month}]\n"
                f"月收益: {summary.get('monthly_return_pct', 0):.2%} | "
                f"淘汰币种: {len(plan.symbols_to_remove)} | "
                f"参数调整: {len(plan.strategy_param_adjustments)}项 | "
                f"系统修复: {len(plan.system_fixes)}项\n"
            )
            if plan.symbols_to_remove:
                msg += f"淘汰: {', '.join(plan.symbols_to_remove[:5])}\n"
            await self.notification_manager.broadcast(msg, "Monthly Iteration Plan", "INFO")
        except Exception as e:
            logger.debug(f"Monthly plan notification error: {e}")

    async def _daily_review_loop(self) -> None:
        """
        每日复盘循环

        每天凌晨02:00执行昨日复盘，生成DailyReviewReport并持久化
        """
        logger.info("Daily review loop started")
        while True:
            try:
                now = datetime.now()
                # 计算到下一个02:00的等待时间
                target = now.replace(hour=2, minute=0, second=0, microsecond=0)
                if now >= target:
                    target = target + timedelta(days=1)
                wait_seconds = (target - now).total_seconds()
                logger.info(f"Next daily review at {target.isoformat()} (in {wait_seconds/3600:.1f}h)")
                await asyncio.sleep(wait_seconds)

                # 执行复盘（复盘昨日）
                yesterday = datetime.now() - timedelta(days=1)
                await self.review_engine.run_daily_review(yesterday)

                # 检查是否月初（1-3号），触发月度迭代（幂等：每天最多触发一次）
                if 1 <= datetime.now().day <= 3:
                    month_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m")
                    plan = self.review_engine.get_monthly_plan(month_str)
                    if plan is None:
                        await self.review_engine.run_monthly_iteration()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daily review loop error: {e}")
                await asyncio.sleep(3600)

    async def _daily_report_loop(self) -> None:
        """每日账单生成与历史分析循环
        
        每天凌晨 03:00 生成前一日交易账单，保存到 data/reports/ 目录。
        每周日凌晨 04:00 生成历史分析报告（含优化建议）。
        同时保存 TradeAuditor 状态。
        """
        logger.info("Daily report loop started (bill generation + historical analysis)")
        while True:
            try:
                now = datetime.now()
                # 计算到下一个 03:00 的等待时间
                target = now.replace(hour=3, minute=0, second=0, microsecond=0)
                if now >= target:
                    target = target + timedelta(days=1)
                wait_seconds = (target - now).total_seconds()
                logger.info(f"Next daily report at {target.isoformat()} (in {wait_seconds/3600:.1f}h)")
                await asyncio.sleep(wait_seconds)

                # ── 1. 生成日账单 ──
                if hasattr(self, 'trade_auditor') and self.trade_auditor:
                    try:
                        daily_report = self.trade_auditor.generate_daily_report()
                        if daily_report:
                            # 保存日账单到文件
                            report_dir = os.path.join(
                                self.config.get("data_dir", "./data"), "reports"
                            )
                            os.makedirs(report_dir, exist_ok=True)
                            report_date = daily_report.get("date", datetime.now().date().isoformat())
                            report_path = os.path.join(report_dir, f"daily_report_{report_date}.json")
                            atomic_write_json(report_path, daily_report)
                            
                            summary = daily_report.get("summary", {})
                            logger.info(
                                f"Daily report generated: {report_date} | "
                                f"trades={summary.get('total_trades', 0)} | "
                                f"win_rate={summary.get('win_rate', 0):.1%} | "
                                f"net_pnl={summary.get('net_pnl', 0):.4f} | "
                                f"fee={summary.get('total_fee', 0):.4f} | "
                                f"saved to {report_path}"
                            )
                    except Exception as e:
                        logger.error(f"Daily report generation error: {e}")

                # ── 2. 保存审核状态 ──
                if hasattr(self, 'trade_auditor') and self.trade_auditor:
                    try:
                        self.trade_auditor.save_state()
                        logger.debug("TradeAuditor state saved")
                    except Exception as e:
                        logger.error(f"TradeAuditor save state error: {e}")

                # ── 3. 每周日生成历史分析报告 ──
                if now.weekday() == 6:  # Sunday
                    if hasattr(self, 'trade_auditor') and self.trade_auditor:
                        try:
                            history = self.trade_auditor.generate_history_analysis(days=7)
                            if history and "error" not in history:
                                report_dir = os.path.join(
                                    self.config.get("data_dir", "./data"), "reports"
                                )
                                os.makedirs(report_dir, exist_ok=True)
                                week_str = now.strftime("%Y-W%W")
                                report_path = os.path.join(report_dir, f"weekly_analysis_{week_str}.json")
                                atomic_write_json(report_path, history)
                                
                                summary = history.get("summary", {})
                                suggestions = history.get("suggestions", [])
                                logger.info(
                                    f"Weekly analysis generated: {week_str} | "
                                    f"trades={summary.get('total_trades', 0)} | "
                                    f"win_rate={summary.get('win_rate', 0):.1%} | "
                                    f"total_pnl={summary.get('total_pnl', 0):.4f} | "
                                    f"max_dd={summary.get('max_drawdown', 0):.4f} | "
                                    f"suggestions={len(suggestions)} | "
                                    f"saved to {report_path}"
                                )
                                # 打印优化建议
                                for s in suggestions:
                                    logger.info(f"  [Suggestion] {s}")
                        except Exception as e:
                            logger.error(f"Weekly analysis generation error: {e}")

                # ── 4. 每月1日生成月度分析报告 ──
                if now.day == 1:
                    if hasattr(self, 'trade_auditor') and self.trade_auditor:
                        try:
                            history = self.trade_auditor.generate_history_analysis(days=30)
                            if history and "error" not in history:
                                report_dir = os.path.join(
                                    self.config.get("data_dir", "./data"), "reports"
                                )
                                os.makedirs(report_dir, exist_ok=True)
                                month_str = now.strftime("%Y-%m")
                                report_path = os.path.join(report_dir, f"monthly_analysis_{month_str}.json")
                                atomic_write_json(report_path, history)
                                logger.info(f"Monthly analysis generated: {month_str} | saved to {report_path}")
                        except Exception as e:
                            logger.error(f"Monthly analysis generation error: {e}")

                # ── 5. 智能体性能摘要 + P12健康状态 ──
                if hasattr(self, 'intelligent_agent') and self.intelligent_agent:
                    try:
                        perf_summary = self.intelligent_agent.get_performance_summary()
                        logger.info(
                            f"IntelligentAgent status: epoch={perf_summary.get('learning_epoch', 0)} | "
                            f"alerts={perf_summary.get('active_alerts', 0)} | "
                            f"strategies={len(perf_summary.get('strategies', {}))}"
                        )

                        # P12: 健康状态报告
                        health = self.intelligent_agent.get_health_status()
                        logger.info(
                            f"P12 AgentHealth: {health['health']} | "
                            f"audits={health['decision_stats']['total_audits']} | "
                            f"approve={health['decision_stats']['approve_rate']:.1%} | "
                            f"reject={health['decision_stats']['reject_rate']:.1%} | "
                            f"reject_1h={health['decision_stats']['rejection_rate_1h']:.1%} | "
                            f"bl={health['blacklist_count']} paused={health['paused_strategies_count']} | "
                            f"hr_risk={health['current_hour_risk']}"
                        )

                        # P12: 决策统计报告
                        d_stats = self.intelligent_agent.get_decision_stats()
                        if d_stats["total_audits"] > 0:
                            logger.info(
                                f"P12 DecisionStats: {d_stats['total_audits']} audits | "
                                f"approve={d_stats['approve_rate']:.1%} reject={d_stats['reject_rate']:.1%} "
                                f"reduce={d_stats.get('reduce_rate', 0):.1%} delay={d_stats.get('delay_rate', 0):.1%} | "
                                f"reject_1h={d_stats['rejection_rate_1h']:.1%} | "
                                f"consec_rej={d_stats['consecutive_rejections']}"
                            )
                            if d_stats["top_rejection_reasons"]:
                                top_reason = d_stats["top_rejection_reasons"][0]
                                logger.info(f"P12 Top rejection: {top_reason[0][:60]} ({top_reason[1]}次)")

                        # P12: 健康异常告警
                        if not health["is_healthy"]:
                            logger.warning(
                                f"P12 Agent UNHEALTHY: {health['health']} | "
                                f"stale_data={health['market_data_stale']} | "
                                f"reject_1h={health['decision_stats']['rejection_rate_1h']:.1%}"
                            )
                    except Exception as e:
                        logger.error(f"IntelligentAgent summary error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Daily report loop error: {e}")
                await asyncio.sleep(3600)

    # ---- 激进合约风险分析回调 ----

    async def _on_risk_pause_new(self, alert) -> None:
        """风险响应：暂停开新仓"""
        logger.warning(f"[RiskAction] 暂停开新仓: {alert.message}")
        # 通过全局风控标记暂停
        if hasattr(self.global_risk, 'set_new_order_paused'):
            self.global_risk.set_new_order_paused(True, alert.message)

    async def _on_risk_pause_strategy(self, alert) -> None:
        """风险响应：暂停策略"""
        logger.warning(f"[RiskAction] 暂停策略: {alert.message}")
        self._emergency_stop_strategies(alert.message)

    async def _on_risk_emergency_close(self, alert) -> None:
        """风险响应：紧急全平"""
        logger.critical(f"[RiskAction] 紧急全平: {alert.message}")
        self._emergency_close_all(alert.message)

    async def _on_risk_reduce_size(self, alert) -> None:
        """风险响应：缩仓30%"""
        symbol = alert.symbol
        if not symbol:
            return
        
        # P6: 冷却期检查，防止同一币种频繁缩仓
        now = time.time()
        last_ts = self._last_reduce_size_ts.get(symbol, 0)
        if now - last_ts < self._reduce_size_cooldown:
            logger.debug(f"P6: reduce_size cooldown for {symbol}, {now - last_ts:.0f}s remaining")
            return
        
        logger.warning(f"[RiskAction] 缩仓30%: {symbol} - {alert.message}")
        if self.okx_client:
            try:
                positions = self.okx_client.get_positions()
                for pos_data in (positions or []):
                    inst_id = pos_data.get("instId", "")
                    if inst_id == symbol:
                        pos_qty = float(pos_data.get("pos", 0))
                        if pos_qty != 0:
                            pos_side = pos_data.get("posSide", "long")
                            reduce_signal = {
                                "symbol": symbol,
                                "strategy_name": "system",
                                "signal_type": "reduce_position",
                                "direction": "sell" if pos_side == "long" else "buy",
                                "price": 0,
                                "quantity": self.okx_client.contracts_to_coins(symbol, abs(pos_qty) * 0.3),
                                "leverage": 5,
                                "reduce_ratio": 0.3,
                                "confidence": 1.0,
                                "reduce_only": True,
                                "reason": f"contract_risk: {alert.message}",
                                "priority": 500,
                                "timestamp": datetime.now().isoformat()
                            }
                            await self.order_executor.handle_signal(reduce_signal)
                            self._last_reduce_size_ts[symbol] = now
                            break
            except Exception as e:
                logger.error(f"Risk reduce_size error: {e}")

    async def _on_risk_adjust_leverage(self, alert) -> None:
        """风险响应：降低杠杆"""
        symbol = alert.symbol
        logger.warning(f"[RiskAction] 降低杠杆: {symbol} - {alert.message}")
        # 安全检查：过滤无效的交易对名称（如"ALL"、空字符串等）
        if not symbol or symbol == "ALL" or "-" not in symbol:
            logger.warning(f"[RiskAction] 跳过无效交易对: {symbol}")
            return
        if self.okx_client:
            try:
                # 将杠杆降到5x
                if hasattr(self.okx_client, 'set_leverage'):
                    self.okx_client.set_leverage(symbol, leverage=5)
            except Exception as e:
                logger.error(f"Risk adjust_leverage error: {e}")

    async def _contract_risk_scan_loop(self) -> None:
        """
        激进合约风险扫描循环

        周期性执行三维风险扫描（市场行情+程序技术+策略逻辑），
        自动执行分级响应，联动五层风控
        """
        interval = self._get_loop_interval("contract_risk_scan", 30)
        logger.info(f"Contract risk scan loop started (interval={interval}s)")
        while True:
            try:
                await asyncio.sleep(interval)

                # 获取持仓数据
                positions = []
                try:
                    raw_positions = self.okx_client.get_positions()
                    positions = raw_positions if raw_positions else []
                except Exception:
                    pass

                # 获取策略统计（从策略引擎收集）
                strategy_stats = {}
                try:
                    if hasattr(self, 'strategy_engine'):
                        container = self.strategy_engine._strategy_container
                        for instance in container.get_all_instances():
                            strat_name = instance.strategy_type.value if hasattr(instance.strategy_type, 'value') else str(instance.strategy_type)
                            metrics = instance.metrics if hasattr(instance, 'metrics') else None
                            key = f"{strat_name}:{instance.symbol}"
                            strategy_stats[key] = {
                                "strategy_name": strat_name,
                                "symbol": instance.symbol,
                                "consecutive_losses": getattr(metrics, 'losing_trades', 0) if metrics else 0,
                                "drawdown_pct": getattr(metrics, 'max_drawdown', 0) if metrics else 0,
                                "live_win_rate": getattr(metrics, 'win_rate', 0.5) if metrics else 0.5,
                                "backtest_win_rate": 0.5,
                                "live_trades": (getattr(metrics, 'winning_trades', 0) + getattr(metrics, 'losing_trades', 0)) if metrics else 0,
                                "leverage": instance.config.get('leverage', 5) if hasattr(instance, 'config') else 5,
                                "position_ratio": abs(instance.position_size) / max(float(config.get("trading", {}).get("total_capital", 509.0)), 1) if hasattr(instance, 'position_size') else 0.0,
                            }
                except Exception as e:
                    logger.debug(f"Strategy stats collection error: {e}")

                # 记录API调用结果到技术风险检测器（每次扫描记录一次汇总状态）
                try:
                    if hasattr(self.risk_gate, '_api_call_times'):
                        error_count = getattr(self.risk_gate, '_api_error_count', 0)
                        total_count = len(self.risk_gate._api_call_times)
                        # 记录本次扫描周期内的错误增量
                        if total_count > 0 and error_count > 0:
                            self.contract_risk_analyzer._technical.record_api_result(
                                is_error=True,
                                error_code="api_error_summary"
                            )
                        else:
                            self.contract_risk_analyzer._technical.record_api_result(
                                is_error=False,
                                error_code=""
                            )
                except Exception:
                    pass

                # 执行全维度扫描
                alerts = await self.contract_risk_analyzer.full_scan(
                    positions=positions,
                    strategy_stats=strategy_stats
                )

                # 记录EMERGENCY/CRITICAL级别告警到通知
                for alert in alerts:
                    if alert.level in (RiskLevel.EMERGENCY, RiskLevel.CRITICAL):
                        try:
                            await self.notification_manager.broadcast(
                                f"[{alert.level.value.upper()}] {alert.category.value}: {alert.message}",
                                "Contract Risk Alert",
                                alert.level.value.upper()
                            )
                        except Exception:
                            pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Contract risk scan loop error: {e}")
                await asyncio.sleep(interval * 2)

    async def _algo_orders_maintenance_loop(self) -> None:
        """算法订单维护循环：刷新场所行情 + 更新成交量分布"""
        interval = self._get_loop_interval("algo_orders_maintenance", 60)
        if not self.smart_order_router:
            return
        logger.info(f"Algo orders maintenance loop started (interval={interval}s)")

        # 使用配置中启用的所有交易对
        from configs.settings import get_all_symbols
        try:
            symbols = get_all_symbols(self.config)
        except Exception:
            symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]

        while True:
            try:
                await asyncio.sleep(interval)

                # 1. 刷新场所实时行情（价差、深度）
                if self.smart_order_router and self.okx_client:
                    await self.smart_order_router.refresh_venue_market_data(symbols[:5])

                # 2. 刷新 VWAP 成交量分布（每5个循环刷新一次，节省API调用）
                if self.vwap_executor and getattr(self, '_algo_maintenance_count', 0) % 5 == 0:
                    for sym in symbols[:3]:
                        try:
                            await self.vwap_executor.refresh_volume_profile(sym, bar="1H", lookback_bars=168)
                        except Exception:
                            pass

                self._algo_maintenance_count = getattr(self, '_algo_maintenance_count', 0) + 1

                # 3. 持久化执行监控数据到文件（供 Dashboard 读取）
                if self.algo_execution_engine:
                    self.algo_execution_engine.persist_monitor_state()
                if self.execution_quality_monitor:
                    self._persist_quality_monitor_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Algo orders maintenance loop error: {e}")
                await asyncio.sleep(interval * 2)

    def _persist_quality_monitor_state(self):
        """持久化执行质量监控状态到文件"""
        import json
        import os
        try:
            path = os.path.join(os.path.dirname(__file__), "..", "data", "algo_orders",
                               "quality_monitor.json")
            path = os.path.normpath(path)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            data = {
                "timestamp": datetime.now().isoformat(),
                "status": self.execution_quality_monitor.get_status(),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Failed to persist quality monitor state: {e}")

    def _freeze_symbol(self, symbol: str, reason: str) -> None:
        """
        币种隔离：冻结单个币种的所有策略（不影响其他币种运行）

        1. 对支持 _extreme_vol_until 的策略：设置该 symbol 的暂停时间戳
        2. 冻结策略容器中该 symbol 的所有实例
        3. 该 symbol 的开仓信号在 signal_processor 层被拒绝（通过 risk_gate.is_symbol_frozen）
        4. 平仓信号仍然允许通过（冻结的币种仍可平仓止损）
        """
        logger.warning(f"⛔ Freezing symbol {symbol}: {reason}")
        try:
            import time as _time
            freeze_until = _time.time() + 600  # 冻结10分钟

            # 对所有策略设置 per-symbol 冻结
            strategies = [
                self.grid_strategy, self.trend_strategy,
                self.scalping_strategy, self.arbitrage_strategy,
                self.spot_grid_strategy, self.spot_martingale_strategy
            ]
            for strategy in strategies:
                if strategy and hasattr(strategy, '_extreme_vol_until'):
                    strategy._extreme_vol_until[symbol] = freeze_until
                    logger.debug(f"Froze {symbol} in {type(strategy).__name__}")

            # 冻结策略容器中该 symbol 的实例
            if hasattr(self, 'strategy_engine'):
                container = self.strategy_engine._strategy_container
                for instance in container.get_instances_by_symbol(symbol):
                    container.freeze_instance(instance.id, f"symbol_freeze: {reason}")

            logger.info(f"Symbol {symbol} frozen across all strategies (reason: {reason})")
        except Exception as e:
            logger.error(f"Error freezing symbol {symbol}: {e}")
    
    async def _position_risk_monitor_loop(self) -> None:
        """
        持仓风控监控循环（每5秒执行）

        统一职责：
        1. 同步账户/持仓/价格状态到五层风控拦截器（L1/L3/L4/L5 依赖这些状态）
        2. 执行 L3 持仓实时风控检查（爆仓预警、阶梯减仓、资金费率减仓）
        3. 执行 L5 紧急熔断检查（极端行情、网络断连）
        4. 触发动态减仓 / 止损全平 / 熔断平仓，循环持续运行
        """
        logger.info("Position risk monitor loop started (5s interval, unified RiskGate state sync)")
        while True:
            try:
                await asyncio.sleep(self._get_loop_interval("position_risk_monitor", 5))

                # ============ 1. 同步账户状态到 RiskGate（L1事前风控 + L4单日风控依赖）============
                await self._sync_risk_gate_account_status()

                # ============ 2. 同步持仓状态到 RiskGate（L3持仓实时风控 + L1单币种仓位上限依赖）============
                await self._sync_risk_gate_positions()

                # ============ 2.1 同步总敞口到 PortfolioRebalancer（P1: 总敞口硬限制）============
                try:
                    positions = self.okx_client.get_positions() or []
                    gross_notional = 0.0
                    for pos_data in positions:
                        pos = self.okx_client._parse_position(pos_data)
                        if pos and float(pos.quantity) != 0:
                            gross_notional += abs(float(pos.notional_usd) or (abs(float(pos.quantity)) * float(pos.mark_price)))
                    self.portfolio_rebalancer.update_gross_exposure(gross_notional)
                except Exception as e:
                    logger.debug(f"Sync gross exposure failed (non-blocking): {e}")

                # ============ 2.5 平仓黑名单币种持仓（释放被锁保证金）============
                await self._close_blacklisted_positions()

                # ============ 2.6 同步手动开单白名单到 L3，避免手动持仓被自动减仓 ============
                self._sync_manual_override_to_risk_gate()

                # ============ 3. 执行 L3 持仓实时风控检查 ============
                results = self.risk_gate.check_positions()

                for result in results:
                    if not result.passed:
                        logger.warning(f"L3 Position Risk: {result.reason}")

                        # 触发动态减仓 / 止损全平 / 熔断平仓
                        if result.action == RiskAction.CLOSE_ALL:
                            symbol = result.details.get("symbol", "")
                            logger.critical(f"L3 critical: closing {symbol}")

                            # 从 risk_gate 获取持仓详情用于构建平仓信号
                            pos_state = self.risk_gate._l3._position_states.get(symbol, {})
                            pos_side = pos_state.get("side", "long")
                            close_direction = "sell" if pos_side == "long" else "buy"

                            # 51169 防护：平仓前确认交易所实际存在对应方向持仓
                            if not self._has_exchange_position(symbol, pos_side):
                                logger.warning(
                                    f"L3 skip closing {symbol} ({pos_side}): no matching exchange position"
                                )
                                self.risk_gate.update_position(
                                    symbol=symbol, entry_price=0, current_price=0,
                                    size=0, leverage=1, side=pos_side
                                )
                                continue

                            close_signal = {
                                "signal_type": "stop_loss",
                                "symbol": symbol,
                                "direction": close_direction,
                                "pos_side": pos_side,
                                "reduce_only": True,
                                "close_position": True,
                                "price": pos_state.get("current_price", 0),
                                "quantity": self.okx_client.contracts_to_coins(symbol, pos_state.get("size", 0)),
                                "leverage": pos_state.get("leverage", 1),
                                "strategy_name": "risk_l3",
                                "reason": result.reason,
                                "priority": 999,
                                "timestamp": datetime.now().isoformat()
                            }
                            try:
                                await self.order_executor.handle_signal(close_signal)
                            except Exception as e:
                                logger.error(f"Error executing L3 close: {e}")

                        elif result.action == RiskAction.REDUCE:
                            symbol = result.details.get("symbol", "")
                            action = result.details.get("action", "")
                            reduce_pct = 0.5
                            if "reduce_30" in action:
                                reduce_pct = 0.3
                            elif "reduce_50" in action:
                                reduce_pct = 0.5
                            elif "reduce_100" in action:
                                reduce_pct = 1.0

                            # 从 risk_gate 获取持仓状态以构建完整的 reduce 信号
                            pos_state = self.risk_gate._l3._position_states.get(symbol, {})
                            pos_side = pos_state.get("side", "long")
                            current_price = pos_state.get("current_price", 0)
                            pos_size = pos_state.get("size", 0)
                            pos_leverage = pos_state.get("leverage", 1)

                            # 51169 防护：减仓前确认交易所实际存在对应方向持仓
                            if not self._has_exchange_position(symbol, pos_side):
                                logger.warning(
                                    f"L3 skip reducing {symbol} ({pos_side}): no matching exchange position"
                                )
                                self.risk_gate.update_position(
                                    symbol=symbol, entry_price=0, current_price=0,
                                    size=0, leverage=1, side=pos_side
                                )
                                continue

                            reduce_qty = self.okx_client.contracts_to_coins(symbol, pos_size * reduce_pct)
                            close_direction = "sell" if pos_side == "long" else "buy"

                            reduce_signal = {
                                "signal_type": "reduce_position",
                                "symbol": symbol,
                                "direction": close_direction,
                                "pos_side": pos_side,
                                "reduce_only": True,
                                "reduce_ratio": reduce_pct,
                                "price": current_price,
                                "quantity": reduce_qty,
                                "leverage": pos_leverage,
                                "strategy_name": "risk_l3",
                                "reason": result.reason,
                                "priority": 500,
                                "timestamp": datetime.now().isoformat()
                            }
                            try:
                                await self.order_executor.handle_signal(reduce_signal)
                            except Exception as e:
                                logger.error(f"Error executing L3 reduce: {e}")

                # ============ 4. 执行 L5 紧急熔断检查 ============
                l5_result = self.risk_gate._l5.check()
                if not l5_result.passed and l5_result.action == RiskAction.CLOSE_ALL:
                    logger.critical(f"L5 Emergency: {l5_result.reason}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position risk monitor: {e}")
                await asyncio.sleep(self._get_loop_interval("position_risk_monitor", 5) * 2)

    async def _trailing_tp_monitor_loop(self) -> None:
        """统一利润锁定梯度监控循环。

        覆盖手工仓（OKX 网页/App 开的仓，无 strategy_name / 无开仓记录）与策略仓：
        对每个非零持仓按 symbol+posSide 逐 tick 输入梯度引擎，按「保本位移 →
        部分落袋 → 紧追踪」三级梯度逐步锁利，防止盈利回吐导致资金磨损。
        反转落袋统一由 ReversalTakeProfitEngine 处理；manual_override 持仓在本循环跳过。
        """
        if not self._tp_lock_enabled:
            logger.info("Profit lock engine disabled, loop not started")
            return

        interval = self._get_loop_interval("trailing_tp_monitor", self._tp_lock_check_interval)
        logger.info(f"Profit lock monitor started (interval={interval}s)")
        while True:
            try:
                await asyncio.sleep(interval)
                await self._check_trailing_take_profit()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in profit lock monitor: {e}")
                await asyncio.sleep(interval * 2)

    def _init_profit_lock_audit_table(self) -> None:
        """初始化利润锁定审计表（记录每次锁利的动作、阶段、价格、pnl 等，供复盘）。"""
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(self._profit_lock_db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profit_lock_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol VARCHAR(50),
                    pos_side VARCHAR(10),
                    action VARCHAR(10),
                    exit_reason VARCHAR(50),
                    phase VARCHAR(20),
                    entry_price FLOAT,
                    close_price FLOAT,
                    pnl_pct FLOAT,
                    peak_price FLOAT,
                    retrace_pct FLOAT,
                    quantity FLOAT,
                    notional FLOAT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to init profit_lock_audit table: {e}")
        finally:
            if conn:
                conn.close()

    def _save_profit_lock_audit(
        self,
        symbol: str,
        pos_side: str,
        action: str,
        exit_reason: str,
        phase: str,
        entry_price: float,
        close_price: float,
        pnl_pct: float,
        peak_price: float,
        retrace_pct: float,
        quantity: float,
        notional: float,
    ) -> None:
        """落库单次利润锁定动作（失败不影响主下单流程）。"""
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(self._profit_lock_db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                INSERT INTO profit_lock_audit
                (symbol, pos_side, action, exit_reason, phase, entry_price, close_price,
                 pnl_pct, peak_price, retrace_pct, quantity, notional)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, pos_side, action, exit_reason, phase, entry_price, close_price,
                pnl_pct, peak_price, retrace_pct, quantity, notional,
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save profit_lock_audit: {e}")
        finally:
            if conn:
                conn.close()

    def _init_profit_lock_state_table(self) -> None:
        """初始化利润锁定状态持久化表（symbol+pos_side 为唯一键）。"""
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(self._profit_lock_db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS profit_lock_state (
                    symbol VARCHAR(50),
                    pos_side VARCHAR(10),
                    state_json TEXT,
                    updated_at DATETIME,
                    PRIMARY KEY (symbol, pos_side)
                )
            """)
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to init profit_lock_state table: {e}")
        finally:
            if conn:
                conn.close()

    def _load_profit_lock_states(self) -> None:
        """启动时从 SQLite 恢复利润锁定梯度状态（跨重启续用）。"""
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(self._profit_lock_db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT symbol, pos_side, state_json FROM profit_lock_state"
            ).fetchall()
        except Exception as e:
            logger.debug(f"No persisted profit_lock state: {e}")
            return
        finally:
            if conn:
                conn.close()

        state_map: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            try:
                import json
                data = json.loads(row["state_json"])
                state_map[f"{row['symbol']}:{row['pos_side']}"] = data
            except Exception as e:
                logger.warning(f"Failed to parse profit_lock state for {row['symbol']}: {e}")
        if state_map:
            self.profit_lock_engine.restore_state(state_map)

    def _sync_profit_lock_state(self, active_keys: set) -> None:
        """把当前活跃持仓的锁利状态 upsert 到 SQLite，并清理已消失持仓的残留状态。

        active_keys 为 `symbol:pos_side` 集合；仅活跃仓位才落库，DB 中不再活跃的 key
        一并删除，避免重启后恢复幽灵锁利状态（例如已平仓仓位的保本/追踪进度）。
        """
        import sqlite3, json
        conn = None
        try:
            conn = sqlite3.connect(self._profit_lock_db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            states = self.profit_lock_engine.dump_state()
            now = datetime.now().isoformat()
            # 1. upsert 活跃仓位状态
            for key, st in states.items():
                if key not in active_keys:
                    continue
                symbol, _, pos_side = key.partition(":")
                conn.execute(
                    "INSERT OR REPLACE INTO profit_lock_state "
                    "(symbol, pos_side, state_json, updated_at) VALUES (?, ?, ?, ?)",
                    (symbol, pos_side, json.dumps(st, ensure_ascii=False), now),
                )
            # 2. 删除 DB 中已无活跃持仓的残留状态
            try:
                rows = conn.execute(
                    "SELECT symbol, pos_side FROM profit_lock_state"
                ).fetchall()
            except Exception:
                rows = []
            for symbol, pos_side in rows:
                if f"{symbol}:{pos_side}" not in active_keys:
                    conn.execute(
                        "DELETE FROM profit_lock_state WHERE symbol = ? AND pos_side = ?",
                        (symbol, pos_side),
                    )
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to sync profit_lock state: {e}")
        finally:
            if conn:
                conn.close()

    async def _check_trailing_take_profit(self) -> None:
        """单次利润锁定梯度检查：遍历所有非零持仓，按梯度引擎判定锁利动作并执行。"""
        try:
            positions = self.okx_client.get_positions()
            if not positions:
                positions = []

            active_keys = set()
            manual_symbols = self._manual_override_symbols()

            for pos_data in positions:
                try:
                    position = self.okx_client._parse_position(pos_data)
                except Exception:
                    continue
                if not position or float(position.quantity) == 0:
                    continue

                symbol = position.symbol
                pos_side = position.side  # "long" / "short"
                if pos_side not in ("long", "short"):
                    continue

                # S6: 手动开单（manual_override）由用户自行管理，跳过自动锁利平仓
                if symbol in manual_symbols:
                    continue

                size_contracts = abs(float(position.quantity))
                entry = float(position.avg_cost)
                mark = float(position.mark_price)
                leverage = int(position.leverage) if position.leverage > 0 else 1
                notional = float(position.notional_usd)

                # 忽略名义价值过小的零碎仓位（避免对无意义的小仓反复平仓）
                if 0 < notional < self._tp_lock_min_notional:
                    continue

                if entry <= 0 or mark <= 0:
                    continue

                key = f"{symbol}:{pos_side}"
                active_keys.add(key)

                # 反转评分不再由本循环计算（S2 收敛：反转落袋统一走 ReversalTakeProfitEngine）
                # S4: 止损优先仲裁 — 查询止损侧（保命移动止损）是否已进入 trailing 保护态，
                #     为 True 时利润锁紧追踪全平让位，避免双信号平仓。
                sl_trailing_active = False
                try:
                    if self.order_executor is not None and hasattr(self.order_executor, "is_stop_loss_trailing_active"):
                        sl_trailing_active = bool(self.order_executor.is_stop_loss_trailing_active(symbol))
                except Exception as e:
                    logger.debug(f"Failed to query SL trailing state for {symbol}: {e}")

                decision = self.profit_lock_engine.compute(
                    symbol=symbol,
                    pos_side=pos_side,
                    entry_price=entry,
                    current_price=mark,
                    breakeven_buffer=self.tp_sl_monitor.breakeven_buffer(),
                    sl_trailing_active=sl_trailing_active,
                )

                if decision.action == "none":
                    continue

                # 51169 防护：确认交易所实际仍持有该方向仓位
                if not self._has_exchange_position(symbol, pos_side):
                    self.profit_lock_engine.reset_position(symbol, pos_side)
                    continue

                size = self.okx_client.contracts_to_coins(symbol, size_contracts)
                close_direction = "sell" if pos_side == "long" else "buy"
                is_full = decision.action == "full"
                close_qty = size if is_full else size * decision.partial_ratio
                if close_qty <= 0:
                    continue

                close_signal = {
                    "signal_type": "take_profit",
                    "symbol": symbol,
                    "direction": close_direction,
                    "pos_side": pos_side,
                    "reduce_only": True,
                    "close_position": is_full,
                    "price": mark,
                    "quantity": close_qty,
                    "leverage": leverage,
                    "strategy_name": "profit_lock",
                    "exit_reason": decision.exit_reason,
                    "reason": (
                        f"{decision.exit_reason}: pnl={decision.pnl_pct:.2%}, "
                        f"peak={decision.peak_price}, action={decision.action}"
                    ),
                    "priority": 700,
                    "timestamp": datetime.now().isoformat(),
                }

                logger.warning(
                    f"Profit lock {decision.action} {symbol} {pos_side}: "
                    f"pnl={decision.pnl_pct:.2%}, reason={decision.exit_reason}, "
                    f"qty={close_qty:.6f}"
                )
                self._save_profit_lock_audit(
                    symbol=symbol,
                    pos_side=pos_side,
                    action=decision.action,
                    exit_reason=decision.exit_reason,
                    phase=decision.phase,
                    entry_price=entry,
                    close_price=mark,
                    pnl_pct=decision.pnl_pct,
                    peak_price=decision.peak_price,
                    retrace_pct=decision.retrace_pct,
                    quantity=close_qty,
                    notional=notional,
                )
                await self.order_executor.handle_signal(close_signal)

            # 清理已平仓/消失仓位的梯度状态（防止残留导致误判）
            self.profit_lock_engine.prune(active_keys)
            # 同步持久化锁利状态（跨重启恢复 + 清理幽灵状态）
            self._sync_profit_lock_state(active_keys)
        except Exception as e:
            logger.error(f"Error in profit lock check: {e}")

    async def _sync_risk_gate_account_status(self) -> None:
        """同步账户状态到五层风控拦截器（equity / available_margin / daily_pnl / daily_start_equity）"""
        try:
            account = None
            # 优先使用 account_manager 缓存的最近有效账户数据
            if hasattr(self.account_manager, '_last_valid_account') and self.account_manager._last_valid_account:
                account = self.account_manager._last_valid_account

            if account is None:
                # 兜底：直接查询OKX
                account_info = self.okx_client.get_account_info()
                if account_info:
                    account = self.okx_client._parse_account_info(account_info)

            if account is None or account.total_equity <= 0:
                return

            equity = float(account.total_equity)
            available_margin = float(account.available_balance)
            # 日内盈亏：用浮动手盈近似（更精确的日初权益由 global_risk 跟踪）
            daily_pnl = float(account.unrealized_pnl)
            daily_start_equity = equity - daily_pnl  # 近似日初权益

            self.risk_gate.update_account_status(
                equity=equity,
                available_margin=available_margin,
                daily_pnl=daily_pnl,
                daily_start_equity=daily_start_equity
            )
            # 更新 L5 心跳，防止无持仓时误触发网络断开熔断
            self.risk_gate._l5.update_heartbeat()
        except Exception as e:
            logger.debug(f"RiskGate account status sync error: {e}")

    def _manual_override_symbols(self) -> set:
        """返回手动开单（manual_override）白名单涉及的 symbol 集合。

        供全局利润锁定循环（ProfitLock）与 RiskGate L3 复用，确保手动单不会被
        自动减仓/平仓（历史事故：手动单被误判自动清仓）。
        """
        try:
            executor = getattr(self, "order_executor", None)
            if executor is None or not hasattr(executor, "get_manual_override_symbols"):
                return set()
            return executor.get_manual_override_symbols() or set()
        except Exception as e:
            logger.debug(f"Failed to fetch manual_override symbols: {e}")
            return set()

    def _sync_manual_override_to_risk_gate(self) -> None:
        """把手动开单白名单同步到 RiskGate L3，避免手动持仓被自动减仓/平仓。"""
        try:
            symbols = self._manual_override_symbols()
            if hasattr(self.risk_gate, "set_manual_override_symbols"):
                self.risk_gate.set_manual_override_symbols(symbols)
        except Exception as e:
            logger.debug(f"Failed to sync manual_override symbols to risk gate: {e}")

    async def _sync_risk_gate_positions(self) -> None:
        """同步持仓状态到五层风控拦截器（L3持仓实时风控 + L1单币种仓位上限）"""
        try:
            positions = self.okx_client.get_positions()
            if not positions:
                positions = []

            active_symbols = set()
            # 构建全量持仓映射（用于替换而非增量更新，防止残留数据）
            positions_map: Dict[str, float] = {}
            
            for pos_data in positions:
                try:
                    position = self.okx_client._parse_position(pos_data)
                    if not position or float(position.quantity) == 0:
                        continue

                    symbol = position.symbol
                    size = abs(float(position.quantity))
                    entry_price = float(position.avg_cost)
                    current_price = float(position.mark_price)
                    leverage = int(position.leverage) if position.leverage > 0 else 1
                    side = position.side  # "long" / "short"
                    # L1仓位限制使用保证金（margin），而非名义价值（notionalUsd）
                    # 因为L1限制 equit * max_symbol_position_ratio 是保证金层面的约束
                    margin_val = float(position.margin)
                    if margin_val <= 0 and position.notional_usd > 0:
                        margin_val = float(position.notional_usd) / leverage
                    elif margin_val <= 0:
                        margin_val = (size * current_price) / leverage

                    active_symbols.add(symbol)

                    # L1: 单币种持仓保证金（累积同一symbol的多方向持仓）
                    positions_map[symbol] = positions_map.get(symbol, 0) + margin_val

                    # L3: 持仓明细（爆仓价预警、阶梯减仓、资金费率）
                    self.risk_gate.update_position(
                        symbol=symbol,
                        entry_price=entry_price,
                        current_price=current_price,
                        size=size,
                        leverage=leverage,
                        side=side,
                        liquidation_price=float(getattr(position, "liquidation_price", 0) or 0),
                        funding_rate=0
                    )

                    # L5: 价格更新（极端行情检测）
                    self.risk_gate.update_price(symbol, current_price)
                except Exception as e:
                    logger.debug(f"RiskGate position sync error for one position: {e}")

            # L1: 全量替换持仓价值（替换而非增量更新，防止已平仓币种数据残留）
            self.risk_gate.sync_positions_from_exchange(positions_map)

            # 清理已平仓的残留持仓（size=0 触发 risk_gate 移除）
            stale = self.risk_gate._l3._position_states.keys() - active_symbols
            for symbol in stale:
                self.risk_gate.update_position(
                    symbol=symbol, entry_price=0, current_price=0,
                    size=0, leverage=1, side="long"
                )
                logger.info(f"RiskGate: removed stale position {symbol} (no longer on OKX)")
        except Exception as e:
            logger.debug(f"RiskGate positions sync error: {e}")

    def _has_exchange_position(self, symbol: str, pos_side: str) -> bool:
        """确认交易所在指定方向存在非零持仓（防止 51169 幽灵平仓）。

        L3 风控循环基于 risk_gate 本地缓存 _position_states 判断，可能残留已平仓
        或方向已变化的仓位。平仓/减仓前用交易所实时数据二次确认，避免反复对无持仓
        合约发 reduce-only 平仓单。
        """
        try:
            positions = self.okx_client.get_positions() or []
            for p in positions:
                if p.get("instId") != symbol:
                    continue
                if pos_side and p.get("posSide") != pos_side:
                    continue
                try:
                    if abs(float(p.get("pos", 0) or 0)) > 0:
                        return True
                except (ValueError, TypeError):
                    continue
            return False
        except Exception as e:
            logger.debug(f"_has_exchange_position error for {symbol}: {e}")
            return False

    def _resolve_position_strategy(self, symbol: str, pos_side: str) -> Optional[str]:
        """根据持仓 symbol+side 推断其归属策略（用于复合黑名单平仓）。

        OKX 仓位对象不含 strategy_name，需回查 SQLite 未平仓记录按 side 匹配；
        无法确定时返回 None（保守不处理，避免误平其他策略的盈利仓位）。
        """
        try:
            if not hasattr(self, 'sqlite_storage') or not self.sqlite_storage:
                return None
            records = self.sqlite_storage.get_all_open_records(symbol)
            if not records:
                return None
            want = "long" if pos_side in ("long", "buy") else "short"
            for rec in records:
                rside = rec.get("side", "")
                if rside in ("long", "buy"):
                    rside = "long"
                elif rside in ("short", "sell"):
                    rside = "short"
                if rside == want:
                    return rec.get("strategy_name") or None
            # 无 side 匹配时，回退到最新未平仓记录的策略
            return records[0].get("strategy_name") or None
        except Exception as e:
            logger.debug(f"_resolve_position_strategy error for {symbol}: {e}")
            return None

    async def _close_blacklisted_positions(self) -> None:
        """平仓黑名单币种持仓，释放被锁死的保证金。

        黑名单币种不应持有任何仓位：智能体将其拉黑后，现存仓位应被强制平仓，
        否则会持续锁死保证金并可能扩大亏损（如 AVAX/POL 已拉黑但仍有浮亏持仓）。
        在持仓风控循环中每 5 秒调用，带 60 秒冷却避免平仓失败后重复下单。
        """
        try:
            if not hasattr(self, 'intelligent_agent') or not self.intelligent_agent:
                return

            try:
                blacklist = self.intelligent_agent.get_blacklist()
                # P0: 复合维度黑名单拆分——全局币种键（无条件平仓）与 币种×策略 键（按持仓策略匹配）
                global_symbols = self.intelligent_agent.get_global_blacklisted_symbols()
                blacklisted_pairs = self.intelligent_agent.get_blacklisted_strategy_pairs()
            except Exception:
                blacklist = {}
                global_symbols = set()
                blacklisted_pairs = set()
            if not blacklist:
                return

            positions = self.okx_client.get_positions()
            if not positions:
                return

            now = time.time()
            for pos_data in positions:
                try:
                    position = self.okx_client._parse_position(pos_data)
                except Exception:
                    continue
                if not position or float(position.quantity) == 0:
                    continue

                symbol = position.symbol
                pos_side = position.side  # "long" / "short"

                # P0: 复合维度黑名单匹配
                # 1) 全局币种键 → 无条件平仓
                # 2) 币种×策略键 → 仅当持仓归属策略命中时才平仓
                if symbol in global_symbols:
                    blacklist_key = symbol
                elif blacklisted_pairs:
                    position_strategy = self._resolve_position_strategy(symbol, pos_side)
                    if position_strategy and (symbol, position_strategy) in blacklisted_pairs:
                        blacklist_key = self.intelligent_agent._make_blacklist_key(
                            symbol, position_strategy
                        )
                    else:
                        continue
                else:
                    continue

                # 冷却检查：避免平仓信号发出后仓位尚未成交而重复下单
                if now - self._blacklist_close_ts.get(symbol, 0) < self._blacklist_close_cooldown:
                    continue

                size_contracts = abs(float(position.quantity))
                # position.quantity 是合约张数，place_order 期望币数（会再 coin_to_contracts），需先转换
                size = self.okx_client.contracts_to_coins(symbol, size_contracts)
                close_direction = "sell" if pos_side == "long" else "buy"
                current_price = float(position.mark_price)
                leverage = int(position.leverage) if position.leverage > 0 else 1
                reason = blacklist.get(blacklist_key, {}).get("reason", "blacklisted")

                close_signal = {
                    "signal_type": "stop_loss",
                    "symbol": symbol,
                    "direction": close_direction,
                    "pos_side": pos_side,
                    "reduce_only": True,
                    "close_position": True,
                    "price": current_price,
                    "quantity": size,
                    "leverage": leverage,
                    "strategy_name": "blacklist",
                    "reason": f"blacklist position close: {reason}",
                    "priority": 998,
                    "timestamp": datetime.now().isoformat(),
                }

                self._blacklist_close_ts[symbol] = now
                logger.warning(
                    f"Closing blacklisted position {symbol} ({pos_side}, size={size}) "
                    f"to release locked margin: {reason}"
                )
                await self.order_executor.handle_signal(close_signal)
        except Exception as e:
            logger.error(f"Error closing blacklisted positions: {e}")

    
    def _on_strategy_signal(self, signal) -> None:
        """
        策略引擎信号回调
        
        将策略计算引擎产生的高质量信号传递给信号处理器
        """
        try:
            symbol = signal.symbol
            signal_type = signal.signal_type.value
            weight = signal.weight
            reason = signal.reason
            
            logger.info(f"StrategyEngine signal: {symbol} {signal_type} (weight={weight:.2f}) - {reason}")
            
            # 将信号写入Redis，供信号处理器消费
            # P0: 信号生成阶段即生成全链路 traceID，贯穿 信号→裁决→订单→记账
            trace_id = EventIDGenerator.get_instance().generate()
            signal_data = {
                "trace_id": trace_id,
                "symbol": symbol,
                "signal_type": signal_type,
                "price": signal.price,
                "quantity": signal.quantity,
                "weight": weight,
                "level": signal.level.value,
                "source": signal.source.value,
                "reason": reason,
                "timestamp": signal.timestamp.isoformat(),
                "strategy_id": signal.strategy_id,
                "metadata": signal.metadata
            }
            
            self.redis_cache.publish_signal(signal_data)
            self.redis_cache.cache_signal(symbol, signal_data)

            # P3 埋点：策略信号统一经事件总线投递（SIGNAL_GENERATED 落盘，供重放/审计）
            self.unified_layer.event_bus.publish_signal(signal_data)
            
            # 更新策略指标
            signal_record = {
                "timestamp": signal.timestamp.isoformat(),
                "symbol": symbol,
                "type": signal_type,
                "weight": weight,
                "processed": False
            }
            self.trade_journal.record_signal(signal_record)
            
        except Exception as e:
            logger.error(f"Error processing strategy signal: {e}")
    
    async def start(self):
        logger.info("Initializing Trading Scheduler...")
        
        await self._health_check()
        
        # P2: 启动统一抽象层
        await self.unified_layer.start()
        
        # P2: 启动状态持久化
        await self.state_manager.start_persistence()

        # 强化自适应智能体：启动指标流水线（聚合/快照/健康检查后台循环）
        if getattr(self, 'metrics_pipeline', None) is not None:
            await self.metrics_pipeline.start()
            logger.info("MetricsPipeline started")
        
        # P2: 启动APM监控
        await self.apm_monitor.start_monitoring()
        
        await self.global_risk.start()
        await self.notebook_fallback.start()
        await self.performance_monitor.start()
        await self.account_manager.start()
        await self.equity_monitor.start()
        await self.trade_journal.start()
        if self.position_manager is not None:
            await self.position_manager.start()
        await self.adaptive_controller.start()
        await self.allocation_agent.start()
        await self.risk_monitor.start()
        
        await self.market_regime_engine.start()
        logger.info("MarketRegimeEngine started for real-time market state monitoring")
        
        await self.strategy_coordinator.start()
        logger.info("StrategyCoordinator started for strategy collaboration")

        # 启动协调循环（问题检测与自动修复）
        await self.strategy_coordinator.start_coordination_loop()
        logger.info("StrategyCoordinator coordination loop started for automatic issue detection and resolution")

        self._register_task("pnl_reconciler", self.pnl_reconciler.start())
        self._register_task("correlation_risk", self.correlation_risk.start())

        await self.black_swan_protection.start()
        logger.info("BlackSwanProtection started for extreme market monitoring")
        
        await self._subscribe_market_data()
        
        await self.order_executor.start()

        # ── 重启挂单丢失修复：启动强制同步（双向比对，通过后才开放下单闸门） ──
        if getattr(self, "order_startup_synchronizer", None) is not None:
            sync_result = await self.order_startup_synchronizer.run()
            if not sync_result.get("ok"):
                logger.critical(
                    f"[order_persistence] 启动同步未通过，暂停新委托: {sync_result.get('error')}"
                )
            else:
                logger.info("[order_persistence] 启动同步通过，下单闸门已开放")
                # 后台巡检协程：3-5 秒核对本地与交易所订单，发现不一致暂停新委托
                if getattr(self, "order_patrol", None) is not None:
                    self._register_task("order_patrol", self.order_patrol.run_loop())
                    logger.info("[order_persistence] OrderPatrolService started")

        # 启动条件单管理器（补挂失败处理、交易所同步）
        await self.conditional_order_manager.start()
        logger.info("ConditionalOrderManager started")

        # 启动挂单时效管理器（检测超时挂单、智能重定价）
        await self.stale_order_manager.start()
        logger.info("StaleOrderManager started")

        # 启动统一数据清理引擎（定期清理DB/文件/内存/Redis）
        await self.data_cleaner.start()
        logger.info("DataCleaner started")

        # 启动过期残留自动清理器（定时+阈值触发；白名单审计数据只归档绝不删除）
        if getattr(self.residue_cleaner, "enable_cleaner", False):
            self._register_task("expired_residue_cleaner", self.residue_cleaner.run_loop())
            logger.info("ExpiredResidueCleaner started")
        else:
            logger.info("ExpiredResidueCleaner disabled, skip start")

        # 通过策略管理器按依赖顺序统一启动所有已启用策略
        if self.strategy_manager:
            start_results = await self.strategy_manager.start_all(ordered=True)
            started = sum(1 for v in start_results.values() if v)
            logger.info(f"StrategyManager: {started}/{len(start_results)} strategies started")
        else:
            if self.grid_strategy: await self.grid_strategy.start()
            if self.trend_strategy: await self.trend_strategy.start()
            if self.scalping_strategy: await self.scalping_strategy.start()
            if self.arbitrage_strategy: await self.arbitrage_strategy.start()
            if self.spot_grid_strategy: await self.spot_grid_strategy.start()
            if self.spot_martingale_strategy: await self.spot_martingale_strategy.start()
        
        # 配置信号路由（必须在策略启动后执行）
        self._setup_signal_routing()

        # 注册持仓清理回调（处理51169无持仓错误时自动清理策略状态）
        if self.scalping_strategy is not None:
            self.order_executor.register_position_cleanup_callback(self.scalping_strategy.cleanup_position)
            self.order_executor.register_position_cleanup_callback(self.grid_strategy.cleanup_position)  # P6: 网格幽灵仓位清理
            self.order_executor.register_fill_callback(self.scalping_strategy.on_order_filled)  # ghost_close 专项 Phase 1: 成交回执桥
            logger.info("Position cleanup callbacks registered for scalping strategy")
        else:
            logger.info("Scalping strategy disabled, skipping cleanup callback registration")

        # ghost_close 专项 Phase 4: 注册 grid / trend 成交回执回调
        if self.grid_strategy is not None:
            self.order_executor.register_fill_callback(self.grid_strategy.on_order_filled)
            self.order_executor.register_strategy_pnl_callback(self.grid_strategy.on_close_pnl)  # P1: 平仓盈亏回传（日内亏损熔断）
            logger.info("Fill callback registered for grid strategy")
        if self.trend_strategy is not None:
            self.order_executor.register_fill_callback(self.trend_strategy.on_order_filled)
            logger.info("Fill callback registered for trend strategy")
        
        await self.optimizer.start()
        
        await self.pipeline_orchestrator.start()
        logger.info("PipelineOrchestrator started")
        
        await self.anomaly_detector.start()
        logger.info("AnomalyDetector started")
        
        await self.recovery_handler.start()
        logger.info("RecoveryHandler started")

        await self.order_lifecycle.start()
        logger.info("OrderLifecycleManager started")

        await self.execution_monitor.start()
        logger.info("ExecutionMonitor started")

        await self.order_state_synchronizer.start()
        logger.info("OrderStateSynchronizer started")

        await self.online_learner.start()
        logger.info("OnlineLearner started")
        
        await self.parameter_adaptor.start()
        logger.info("ParameterAdaptor started")
        
        await self.strategy_evolver.start()
        logger.info("StrategyEvolver started")

        await self.knowledge_base.start()
        logger.info("KnowledgeBase started")

        await self.market_regime_detector.start()
        logger.info("MarketRegimeDetector started")

        await self.performance_feedback.start()
        logger.info("PerformanceFeedback started")

        await self.meta_learner.start()
        logger.info("MetaLearner started")
        
        await self.trading_recovery.start()
        logger.info("TradingRecoveryService started for automatic trading recovery")

        # P0: 启动策略引擎定期回测（盘中滚动回测，失效自动冻结策略）
        self.strategy_engine.start_periodic_backtest(interval_seconds=60)
        logger.info("StrategyEngine periodic backtest started for strategy health monitoring")

        # P0: 启动资金管理定期任务（权重再平衡、资金池再平衡）
        await self.capital_manager.start_periodic_tasks()
        logger.info("CapitalManager periodic tasks started for capital rebalancing")

        # P0: 设置五层风控拦截器紧急熔断回调
        self.risk_gate.set_emergency_callbacks(
            close_all_cb=self._emergency_close_all,
            stop_strategies_cb=self._emergency_stop_strategies
        )
        # 币种隔离：注册单币种冻结回调（单币种极端行情只冻结该币种，不全盘停盘）
        self.risk_gate.set_freeze_symbol_callback(self._freeze_symbol)
        # 启动持仓风控监控循环
        self._register_task("position_risk_monitor", self._position_risk_monitor_loop())
        logger.info("RiskGate emergency callbacks + per-symbol freeze callback set, position risk monitor started")
        # 启动统一移动止盈锁利监控循环（手工仓 + 策略仓）
        self._register_task("trailing_tp_monitor", self._trailing_tp_monitor_loop())
        logger.info("Trailing take-profit lock monitor registered")

        # 启动P2级监控告警组件
        await self._start_monitoring_components()
        logger.info("P2 monitoring components started")

        await self._start_signal_listener()

        self._register_task("analysis_loop", self._analysis_loop())
        self._register_task("optimization_loop", self._optimization_loop())
        self._register_task("verification_loop", self._verification_loop())
        self._register_task("learning_loop", self._learning_loop())
        self._register_task("monitoring_loop", self._monitoring_loop())
        # P0: 统一参数优化编排循环（真实回测评估 + 每周触发）
        if self.param_optimizer:
            self._register_task("param_optimization", self._param_optimization_loop())
        # 策略贡献度分析循环
        if self.contribution_analyzer:
            self._contribution_task = self._register_task(
                "contribution_analysis", self._contribution_analysis_loop())
        # 算力调度反馈循环：周期性推送 CPU/内存数据 + MarketRegime波动率到 ComputeScheduler
        self._register_task("compute_feedback", self._compute_feedback_loop())
        # 激进合约风险扫描循环：三维风险扫描（市场行情+程序技术+策略逻辑）
        self._register_task("contract_risk_scan", self._contract_risk_scan_loop())
        # 复盘迭代循环：每日02:00复盘+月初月度迭代
        self._register_task("daily_review", self._daily_review_loop())
        # 策略管理器健康检查循环
        if self.strategy_manager:
            self._register_task("strategy_manager_health", self._strategy_manager_health_loop())
        # P0: 每日账单生成与历史分析循环
        self._register_task("daily_report", self._daily_report_loop())
        # 企业级智能交易记录分析循环（6小时：市场状态/策略方向/信号质量/ADX确认/优化方向报告）
        if self.analysis_agent:
            self._register_task("intelligent_analysis", self._intelligent_analysis_loop())

        # P0: 风险预算刷新循环（周期性风险预算计算 + 阈值突破告警）
        if self.risk_budget_engine:
            self._register_task("risk_budget_refresh", self._risk_budget_refresh_loop())
            # 启动时加载历史状态
            try:
                self.risk_budget_engine._load_state()
                logger.info("RiskBudgetEngine historical state loaded")
            except Exception as e:
                logger.warning(f"Failed to load risk budget state: {e}")

        # P0: 智能决策引擎循环（自适应阈值更新、审计链持久化、决策质量监控）
        if self.intelligent_decision_engine:
            self._register_task("decision_engine", self._decision_engine_loop())
            logger.info("IntelligentDecisionEngine loop registered")

        # P0: 决策协调器处理循环（过期清理、优先级升级、依赖拓扑排序执行）
        self._register_task("decision_coordinator", self._decision_coordinator_loop())
        logger.info("DecisionCoordinator processing loop registered")

        # P0: 规则引擎维护循环（冲突检测、缓存清理、性能统计、规则效果评估）
        if self.rule_engine:
            self._register_task("rule_engine_maintenance", self._rule_engine_maintenance_loop())
            logger.info("RuleBasedEngine maintenance loop registered")

        # P0: ML决策引擎维护循环（重训练检查、特征漂移检测、模型健康报告）
        if self.ml_decision_engine:
            self._register_task("ml_decision_maintenance", self._ml_decision_maintenance_loop())
            logger.info("MLDecisionEngine maintenance loop registered")

        # P0: RL智能体维护循环（训练步、探索衰减、模型持久化）
        if self.rl_agent:
            self._register_task("rl_agent_maintenance", self._rl_agent_maintenance_loop())
            logger.info("TradingRLAgent maintenance loop registered")

        # P0: 智能决策引擎审计链持久化循环
        if self.intelligent_decision_engine:
            self._register_task("audit_chain_persist", self._audit_chain_persist_loop())
            logger.info("Audit chain persistence loop registered")

        # P0: 算法订单维护循环（场所行情刷新 + 成交量分布更新）
        if self.smart_order_router:
            self._register_task("algo_orders_maintenance", self._algo_orders_maintenance_loop())
            logger.info("AlgoOrders maintenance loop registered")

        # P22-8: 合规红线检查循环
        if self.compliance_checker.enabled:
            self._register_task("compliance_check", self._compliance_check_loop())
            logger.info("P22-8: Compliance check loop registered")

        # 企业级：防抖动统计导出循环（跨进程供 dashboard 展示拦截命中情况）
        if getattr(self, "debounce_engine", None) is not None:
            self._register_task("debounce_stats_export", self._debounce_stats_export_loop())
            logger.info("AntiDebounce stats export loop registered")

        # 量化AGI自治协调器循环（感知→诊断→决策→执行→反馈闭环）
        if getattr(self, "agi_orchestrator", None) is not None:
            self._register_task("agi_orchestrator", self._agi_orchestrator_loop())
            logger.info("QuantAGIOrchestrator loop registered")

        # 运维自愈闭环循环（监控告警→根因分析→自动恢复→反馈）
        if getattr(self, "ops_self_heal", None) is not None:
            self._register_task("ops_self_heal", self._ops_self_heal_loop())
            logger.info("OpsSelfHeal loop registered")

        # 顶层 AGI 编排循环（跨闭环联动：聚合四闭环状态 → 联动诊断）
        if getattr(self, "top_level_agi", None) is not None:
            self._register_task("top_level_agi", self._top_level_agi_loop())
            logger.info("TopLevelAGI loop registered")

        logger.info("Trading Scheduler started successfully")

    async def _debounce_stats_export_loop(self) -> None:
        """防抖动统计导出循环：周期性把 AntiDebounceEngine 状态快照写入 JSON，供 dashboard 独立进程读取。"""
        interval = self._get_loop_interval("debounce_stats_export", 5)
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "debounce_status.json")
        logger.info(f"AntiDebounce stats export loop started ({interval}s interval -> {filepath})")
        while True:
            try:
                await asyncio.sleep(interval)
                engine = getattr(self, "debounce_engine", None)
                if engine is not None:
                    engine.export_status(filepath)
            except Exception as e:
                logger.warning(f"AntiDebounce stats export failed: {e}")

    def _record_agent_action_results(self, route_result: Dict[str, Any]) -> None:
        """Publish restricted execution receipts into shared cross-agent memory."""
        memory = getattr(self, "agent_learning_memory", None)
        if memory is None:
            return
        for item in route_result.get("action_results") or []:
            if not isinstance(item, dict):
                continue
            try:
                memory.record(
                    agent="restricted_execution_channel",
                    kind="action_result",
                    trace_id=item.get("trace_id"),
                    decision=item.get("type"),
                    outcome={
                        "status": item.get("status"),
                        "reason": item.get("reason", ""),
                    },
                    veto=item.get("status") == "rejected",
                )
            except Exception as e:
                logger.debug(f"Shared execution memory write failed: {e}")

    async def _idle_cash_deployer(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """低风险闲置资金归集落地：小幅提升 A/B 级核心策略的资金权重。

        - 仅接受 RestrictedExecutionChannel 判定为低风险的 idle_cash_deploy 动作。
        - 通过 AdaptiveController._apply_allocation_shift 做有界正偏移 + 归一化，
          不直接下单；真实开仓仍由策略信号 + RiskGate 六层决定。
        - fail-closed：目标缺失 / 控制器缺失 / 异常时返回未部署，绝不静默放行。
        """
        strategy = str(action.get("strategy") or "").strip()
        if not strategy:
            return {"deployed": False, "reason": "no_strategy"}
        ac = getattr(self, "adaptive_controller", None)
        if ac is None or not hasattr(ac, "_apply_allocation_shift"):
            return {"deployed": False, "reason": "no_adaptive_controller"}
        try:
            # 有界小幅正偏移，_apply_allocation_shift 内部 clamp + 归一化，避免权重漂移
            try:
                shift = float((self.config.get("agi_orchestrator") or {}).get("idle_deploy_shift", 0.05))
            except (TypeError, ValueError):
                shift = 0.05
            shift = max(0.01, min(0.15, shift))
            ac._apply_allocation_shift({strategy: shift})
            logger.info(f"[AGI] idle cash deploy: boost {strategy} +{shift:.2f}")
            return {"deployed": True, "strategy": strategy, "shift": shift}
        except Exception as e:
            logger.warning(f"[AGI] idle cash deploy failed (fail-closed): {e}")
            return {"deployed": False, "reason": str(e)[:200]}

    def _is_kill_switch_active(self) -> bool:
        """全局 Kill Switch 熔断检查（供 RestrictedExecutionChannel 完全自主模式查询）。

        fail-closed：无法确认时返回 True（视为熔断），绝不在无法确认安全时放行资金动作。
        """
        rg = getattr(self, "risk_gate", None)
        if rg is None:
            return False
        try:
            return bool(rg.is_kill_switch_enabled())
        except Exception as e:
            logger.warning(f"[AGI] kill switch check failed (fail-closed): {e}")
            return True

    async def _reallocate_deployer(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """完全自主：落地 reallocate 动作（把策略权重调整到 target_allocation）。

        - 通过 AdaptiveController._apply_allocation_shift 做有界相对偏移（单步幅度 clamp），
          不直接下单；真实开仓仍由策略信号 + RiskGate 六层决定。
        - 仅接受资金池内策略（_dynamic_allocations 中已配置的策略）；
          sync/manual_override 等占位标签不在池内，返回未部署（降级为通知）。
        - fail-closed：目标缺失 / 控制器缺失 / 异常时返回未部署，绝不静默放行。
        """
        strategy = str(action.get("strategy") or "").strip()
        target = action.get("target_allocation")
        if not strategy or target is None:
            return {"deployed": False, "reason": "missing_strategy_or_target"}
        ac = getattr(self, "adaptive_controller", None)
        if ac is None or not hasattr(ac, "_apply_allocation_shift"):
            return {"deployed": False, "reason": "no_adaptive_controller"}
        try:
            target_f = float(target)
            if strategy not in getattr(ac, "_dynamic_allocations", {}):
                return {"deployed": False, "reason": f"not_in_pool:{strategy}"}
            current = float(ac._dynamic_allocations.get(strategy, 0.0))
            shift = target_f - current
            ac._apply_allocation_shift({strategy: shift})
            logger.info(
                f"[AGI] reallocate deploy: {strategy} -> target {target_f:.4f} "
                f"(shift {shift:+.4f})"
            )
            return {"deployed": True, "strategy": strategy, "target_allocation": target_f}
        except Exception as e:
            logger.warning(f"[AGI] reallocate deploy failed (fail-closed): {e}")
            return {"deployed": False, "reason": str(e)[:200]}

    async def _close_position_deployer(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """账户级收益落袋平仓落地：按浮盈降序平掉 close_ratio 比例的浮盈持仓。

        - 遍历当前非零持仓，计算每仓浮盈比例，仅挑有浮盈（pnl > 0）的仓位。
        - 按浮盈比例降序，平掉「累计名义价值达到目标」的仓位（reduce_only 全平单仓），
          实现账户级利润落袋（优先落袋最肥的仓位）。
        - 手动单（manual_override）跳过，不触碰。
        - fail-closed：无 order_executor / 无浮盈持仓 / 异常时返回未部署，绝不静默放行。
        """
        try:
            close_ratio = float(action.get("close_ratio") or 0.0)
        except (TypeError, ValueError):
            close_ratio = 0.0
        if close_ratio <= 0:
            return {"deployed": False, "reason": "invalid_close_ratio"}
        executor = getattr(self, "order_executor", None)
        if executor is None or not hasattr(executor, "handle_signal"):
            return {"deployed": False, "reason": "no_order_executor"}
        try:
            positions = self.okx_client.get_positions()
            if not positions:
                return {"deployed": False, "reason": "no_positions"}
            manual_symbols = self._manual_override_symbols()
            profitable = []
            total_notional = 0.0
            for pos_data in positions:
                try:
                    position = self.okx_client._parse_position(pos_data)
                except Exception:
                    continue
                if not position or float(position.quantity) == 0:
                    continue
                symbol = position.symbol
                side = position.side
                if side not in ("long", "short"):
                    continue
                if symbol in manual_symbols:
                    continue
                entry = float(position.avg_cost)
                mark = float(position.mark_price)
                if entry <= 0 or mark <= 0:
                    continue
                pnl_pct = (mark - entry) / entry if side == "long" else (entry - mark) / entry
                if pnl_pct <= 0:
                    continue  # 只平有浮盈的仓位
                notional = float(position.notional_usd)
                profitable.append({
                    "symbol": symbol,
                    "side": side,
                    "notional": notional,
                    "pnl_pct": pnl_pct,
                    "quantity": abs(float(position.quantity)),
                    "leverage": int(position.leverage) if position.leverage > 0 else 1,
                })
                total_notional += notional
            if not profitable:
                return {"deployed": False, "reason": "no_profitable_positions"}
            profitable.sort(key=lambda x: -x["pnl_pct"])
            target_notional = total_notional * close_ratio
            closed_notional = 0.0
            closed = 0
            for p in profitable:
                if closed_notional >= target_notional:
                    break
                size = self.okx_client.contracts_to_coins(p["symbol"], p["quantity"])
                if size <= 0:
                    continue
                close_signal = {
                    "signal_type": "take_profit",
                    "symbol": p["symbol"],
                    "direction": "sell" if p["side"] == "long" else "buy",
                    "pos_side": p["side"],
                    "reduce_only": True,
                    "close_position": True,
                    "price": 0.0,
                    "quantity": size,
                    "leverage": p["leverage"],
                    "strategy_name": "agi_profit_take",
                    "exit_reason": "agi_profit_take",
                    "reason": (
                        f"AGI profit take: close profitable {p['symbol']} "
                        f"({p['pnl_pct']:.2%})"
                    ),
                    "priority": 700,
                    "timestamp": datetime.now().isoformat(),
                }
                await executor.handle_signal(close_signal)
                closed_notional += p["notional"]
                closed += 1
            logger.info(
                f"[AGI] profit take deploy: close_ratio={close_ratio:.2f}, "
                f"closed={closed} positions, notional~{closed_notional:.2f}"
            )
            return {"deployed": True, "closed_positions": closed, "close_ratio": close_ratio}
        except Exception as e:
            logger.warning(f"[AGI] profit take close failed (fail-closed): {e}")
            return {"deployed": False, "reason": str(e)[:200]}

    async def _param_adjust_deployer(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """策略参数自适应落地：调 strategy_manager.hot_reload_config 下调策略杠杆。

        - 仅接受 RestrictedExecutionChannel 判定为低风险的 param_adjust 动作。
        - 支持 symbol 维度：action 含 symbol 时落地到 currencies.symbol_overrides，
          实现 ARB 等币种在 tier 默认参数之上单独定制更精细的杠杆/间距。
        - 通过策略管理器热重载指定参数（默认 leverage），策略实例自身的 update_config
          负责生效；参数变更后由 ParameterRollbackGuard 做 30 分钟验证，可回滚。
        - fail-closed：目标缺失 / 策略未注册 / 热重载失败时返回未部署，绝不静默放行。
        """
        strategy = str(action.get("strategy") or "").strip()
        symbol = str(action.get("symbol") or "").strip()
        param = str(action.get("param") or "leverage").strip()
        value = action.get("value")
        if not strategy and not symbol:
            return {"deployed": False, "reason": "missing_strategy_or_symbol"}
        if value is None:
            return {"deployed": False, "reason": "missing_value"}
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return {"deployed": False, "reason": "invalid_value"}

        # 逐币种维度：落到 currencies.symbol_overrides（grid 读取 live，无需策略重载）
        if symbol:
            return self._deploy_symbol_param_adjust(symbol, param, value_f)

        mgr = getattr(self, "strategy_manager", None)
        if mgr is None or not hasattr(mgr, "hot_reload_config"):
            return {"deployed": False, "reason": "no_strategy_manager"}
        try:
            ok = await mgr.hot_reload_config(
                strategy, {param: value_f}, reason="agi_param_adaptation"
            )
            if ok:
                logger.info(f"[AGI] param adjust deploy: {strategy}.{param} -> {value_f}")
                return {"deployed": True, "strategy": strategy, "param": param, "value": value_f}
            return {"deployed": False, "reason": f"hot_reload_rejected:{strategy}"}
        except Exception as e:
            logger.warning(f"[AGI] param adjust deploy failed (fail-closed): {e}")
            return {"deployed": False, "reason": str(e)[:200]}

    def _deploy_symbol_param_adjust(self, symbol: str, param: str, value: float) -> Dict[str, Any]:
        """逐币种参数调整落地：写入 currencies.symbol_overrides[base][param]。

        - 共享 config 对象：grid 策略通过 get_symbol_config(symbol, self.config) 读取
          同一份 config，写入后即时生效（无需策略重启）。
        - fail-closed：config 缺失 / base 提取失败时返回未部署。
        """
        base = symbol.replace("-USDT", "").replace("USDT-", "").replace("-SWAP", "").strip()
        if not base:
            return {"deployed": False, "reason": "invalid_symbol"}
        try:
            currencies = self.config.setdefault("currencies", {})
            overrides = currencies.setdefault("symbol_overrides", {})
            entry = overrides.setdefault(base, {})
            entry[param] = value
            logger.info(f"[AGI] symbol param deploy: {base}.{param} -> {value}")
            return {"deployed": True, "symbol": symbol, "param": param, "value": value}
        except Exception as e:
            logger.warning(f"[AGI] symbol param deploy failed (fail-closed): {e}")
            return {"deployed": False, "reason": str(e)[:200]}

    async def _strategy_pause_deployer(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """策略生命周期落地：调 strategy_manager.pause_strategy 停开新仓。

        - 仅接受 RestrictedExecutionChannel 判定为低风险的 strategy_pause 动作。
        - 暂停策略实例（停止产生新开仓信号），不触碰已有持仓（孤儿仓仍走人工确认）。
        - fail-closed：策略未注册 / 暂停失败时返回未部署，绝不静默放行。
        """
        strategy = str(action.get("strategy") or "").strip()
        if not strategy:
            return {"deployed": False, "reason": "missing_strategy"}
        mgr = getattr(self, "strategy_manager", None)
        if mgr is None or not hasattr(mgr, "pause_strategy"):
            return {"deployed": False, "reason": "no_strategy_manager"}
        try:
            ok = await mgr.pause_strategy(strategy)
            if ok:
                logger.info(f"[AGI] strategy pause deploy: {strategy}")
                return {"deployed": True, "strategy": strategy}
            return {"deployed": False, "reason": f"pause_rejected:{strategy}"}
        except Exception as e:
            logger.warning(f"[AGI] strategy pause deploy failed (fail-closed): {e}")
            return {"deployed": False, "reason": str(e)[:200]}

    async def _strategy_resume_deployer(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """策略恢复落地：调 strategy_manager.resume_strategy 恢复开仓。

        - 仅接受 RestrictedExecutionChannel 判定为低风险的 strategy_resume 动作。
        - 恢复策略实例（重新产生开仓信号），resume_strategy 内部有 P2-12 上线门禁
          （退出/暂停/迷你回测健康）兜底，不直接下单。
        - fail-closed：策略未注册 / 恢复失败时返回未部署，绝不静默放行。
        """
        strategy = str(action.get("strategy") or "").strip()
        if not strategy:
            return {"deployed": False, "reason": "missing_strategy"}
        mgr = getattr(self, "strategy_manager", None)
        if mgr is None or not hasattr(mgr, "resume_strategy"):
            return {"deployed": False, "reason": "no_strategy_manager"}
        try:
            ok = await mgr.resume_strategy(strategy)
            if ok:
                logger.info(f"[AGI] strategy resume deploy: {strategy}")
                return {"deployed": True, "strategy": strategy}
            return {"deployed": False, "reason": f"resume_rejected:{strategy}"}
        except Exception as e:
            logger.warning(f"[AGI] strategy resume deploy failed (fail-closed): {e}")
            return {"deployed": False, "reason": str(e)[:200]}

    async def _agi_orchestrator_loop(self) -> None:
        """量化AGI自治协调器循环：周期性触发 run_cycle 形成感知→诊断→决策→执行→反馈闭环。

        - 由 config["agi_orchestrator"]["enabled"] 控制是否装配；未装配则本循环立即退出。
        - run_cycle 内部带 cooldown 幂等，循环间隔 >= cooldown 时每次真正执行一次闭环。
        - 动作指令经 RestrictedExecutionChannel 路由：高风险动作只排队待人工确认，
          绝不自动下单；低风险降风险动作通过受限 deployer 落地，并把逐动作结果反馈给 AGI。
        """
        orchestrator = getattr(self, "agi_orchestrator", None)
        if orchestrator is None:
            logger.debug("QuantAGIOrchestrator disabled, _agi_orchestrator_loop exits")
            return
        interval = self._get_loop_interval("agi_orchestrator", 300)
        logger.info(f"QuantAGIOrchestrator loop started (interval={interval}s)")
        while True:
            try:
                await asyncio.sleep(interval)
                report = await orchestrator.run_cycle()
                alerts = (report.get("diagnosis") or {}).get("alerts") or []
                critical = [a for a in alerts if a.get("level") == "critical"]
                health = (report.get("reflection") or {}).get("health_grade", "N/A")

                # 受限执行通道：路由动作指令（高风险排队待人工确认，低风险告警）
                channel = getattr(self, "restricted_execution_channel", None)
                if channel is not None:
                    try:
                        route_result = await channel.route(report)
                        self._record_agent_action_results(route_result)
                        # 决策→执行→反馈闭环：把执行结果回传 orchestrator，供下一周期诊断感知
                        if (
                            route_result.get("status") not in ("cached_report", "no_actions", "no_report")
                            and hasattr(orchestrator, "report_execution_result")
                        ):
                            orchestrator.report_execution_result(route_result)
                        if route_result.get("queued"):
                            logger.warning(
                                f"[AGI] cycle={report.get('cycle')} "
                                f"queued={route_result['queued']} manual-confirmation-required"
                            )
                    except Exception as e:
                        logger.warning(f"[AGI] restricted execution route failed: {e}")

                if critical:
                    logger.warning(
                        f"[AGI] cycle={report.get('cycle')} status={report.get('status')} "
                        f"critical_alerts={len(critical)}"
                    )
                else:
                    logger.info(
                        f"[AGI] cycle={report.get('cycle')} status={report.get('status')} health={health}"
                    )
            except Exception as e:
                logger.warning(f"[AGI] orchestrator loop failed (degraded): {e}")

    async def _ops_self_heal_loop(self) -> None:
        """运维自愈闭环循环：周期性拉取 AnomalyDetector 异常，做根因分析 + 自动恢复。

        - 监控环节：从 anomaly_detector 拉取最近 5 分钟内的异常
        - 根因分析 + 决策 + 恢复：交由 OpsSelfHealCoordinator.handle_event 完成
        - 冷却幂等：同一故障类型在冷却期内跳过，避免重复恢复
        - fail-closed：高风险根因（行情/风险）不自动恢复，仅告警
        """
        orchestrator = getattr(self, "ops_self_heal", None)
        if orchestrator is None:
            logger.debug("OpsSelfHealCoordinator disabled, _ops_self_heal_loop exits")
            return
        interval = self._get_loop_interval("ops_self_heal", 60)
        logger.info(f"OpsSelfHeal loop started (interval={interval}s)")
        while True:
            try:
                await asyncio.sleep(interval)
                detector = self.anomaly_detector
                if detector is None:
                    continue
                anomalies = detector.get_anomalies(limit=30)
                now = datetime.now()
                recent = [a for a in anomalies if (now - a.timestamp).total_seconds() < 300]
                for a in recent:
                    await orchestrator.handle_event(
                        a.type.value,
                        {"message": a.message, "severity": a.severity.value, "details": a.details},
                    )
            except Exception as e:
                logger.warning(f"[OpsSelfHeal] loop failed (degraded): {e}")

    async def _top_level_agi_loop(self) -> None:
        """顶层 AGI 编排循环：周期性聚合四闭环状态并运行跨闭环联动规则。

        - 由 config["top_level_agi"]["enabled"] 控制是否装配；未装配则本循环立即退出。
        - run_cycle 内部带 cooldown 幂等，且仅产出「低风险标记 + 高风险建议」，
          不自动执行改参/暂停等危险动作。
        """
        orchestrator = getattr(self, "top_level_agi", None)
        if orchestrator is None:
            logger.debug("TopLevelAGICoordinator disabled, _top_level_agi_loop exits")
            return
        interval = self._get_loop_interval("top_level_agi", 60)
        logger.info(f"TopLevelAGI loop started (interval={interval}s)")
        while True:
            try:
                await asyncio.sleep(interval)
                report = orchestrator.run_cycle()
                actions = report.get("actions") or []
                if actions:
                    logger.info(
                        f"[TopAGI] cycle={report.get('cycle')} linkage_actions={len(actions)}"
                    )
                else:
                    logger.debug(f"[TopAGI] cycle={report.get('cycle')} no linkage actions")

                decision_plan = report.get("decision_plan") or {}
                if decision_plan.get("auto_execute") and not report.get("cooldown"):
                    channel = getattr(self, "restricted_execution_channel", None)
                    if channel is None:
                        logger.warning(
                            "[TopAGI] auto-executable action withheld: RestrictedExecutionChannel unavailable"
                        )
                    else:
                        report["decision_id"] = f"top-agi-{report.get('cycle', 0)}"
                        route_result = await channel.route(report)
                        self._record_agent_action_results(route_result)
                        logger.info(
                            f"[TopAGI] routed cycle={report.get('cycle')} "
                            f"deployed={route_result.get('deployed', 0)} "
                            f"rejected={route_result.get('rejected', 0)}"
                        )
            except Exception as e:
                logger.warning(f"[TopAGI] loop failed (degraded): {e}")

    async def _compute_feedback_loop(self) -> None:
        """
        算力调度反馈循环

        职责：
        1. 周期性采集 CPU/内存使用率，推送给 ComputeScheduler（过载降频闭环）
        2. 从 MarketRegimeEngine 拉取 per-symbol 波动率分位数，推送给 ComputeScheduler（优先级动态调整）
        3. CPU过载时自动延长非实时循环周期（_health_scoring_loop 已内置，此处作为独立保底）
        """
        logger.info(f"Compute feedback loop started (interval={self._get_loop_interval('compute_feedback', 10)}s)")
        while True:
            try:
                await asyncio.sleep(self._get_loop_interval("compute_feedback", 10))

                # 1. 采集 CPU/内存 → ComputeScheduler
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.5)
                    memory = psutil.virtual_memory()
                    self.compute_scheduler.update_cpu_memory(cpu_percent, memory.percent)
                except Exception as e:
                    logger.debug(f"CPU/memory采集失败: {e}")

                # 2. 从 MarketRegimeEngine 拉取 per-symbol 波动率 → ComputeScheduler + CapitalManager
                try:
                    if hasattr(self.market_regime_engine, 'get_symbol_regime'):
                        for symbol in get_all_symbols(self.config):
                            regime = self.market_regime_engine.get_symbol_regime(symbol)
                            if regime and isinstance(regime, dict):
                                vol_pct = regime.get("volatility_percentile")
                                if vol_pct is not None:
                                    self.compute_scheduler.update_symbol_volatility(symbol, float(vol_pct))
                                    # 同步推送给资金管理器（波动率感知仓位分配）
                                    try:
                                        vol_value = regime.get("volatility", 0.02)
                                        momentum = regime.get("momentum", 0.0)
                                        self.capital_manager.update_symbol_metrics(
                                            symbol, float(vol_value), float(momentum), 0.5
                                        )
                                    except Exception:
                                        pass
                except Exception as e:
                    logger.debug(f"MarketRegime波动率反馈失败: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Compute feedback loop error: {e}")
                await asyncio.sleep(self._get_loop_interval("compute_feedback", 10))

    async def _start_monitoring_components(self):
        """启动P2级监控告警组件"""
        try:
            # 启动指标收集
            self.metrics_collector.start_collection(interval_seconds=60)
            logger.info("MetricsCollector started")
        except Exception as e:
            logger.warning(f"Failed to start MetricsCollector: {e}")

        try:
            # 启动健康度评分
            self._register_task("health_scoring", self._health_scoring_loop())
            logger.info("HealthScorer loop started")
        except Exception as e:
            logger.warning(f"Failed to start HealthScorer: {e}")

        try:
            # 启动自动恢复
            self._register_task("auto_recovery", self._auto_recovery_loop())
            logger.info("AutoRecovery loop started")
        except Exception as e:
            logger.warning(f"Failed to start AutoRecovery: {e}")

        # P1 重放恢复：启动后立即对账一次（回溯最近 24h 事件，发现崩溃窗口悬单）
        try:
            from datetime import timedelta as _td
            self._reconcile_event_order_state(
                from_ts=datetime.now() - _td(hours=24),
            )
            logger.info("Event order state reconcile completed at startup")
        except Exception as e:
            logger.warning(f"Startup event replay reconcile failed: {e}")

        # 启动对账：按交易所真实持仓清理 EnhancedStopLoss 幽灵止损/止盈状态
        try:
            self._reconcile_stop_loss_states()
        except Exception as e:
            logger.warning(f"Startup stop-loss state reconcile failed: {e}")

    def _reconcile_stop_loss_states(self) -> None:
        """启动对账：按交易所真实持仓清理 EnhancedStopLoss 的幽灵止损/止盈状态。

        安全保护（fail-safe）：get_positions 返回空列表时无法区分「真空仓」与「API 失败」，
        此时跳过对账，避免把 API 失败误判为空仓而误删全部止损状态。
        """
        if not self.okx_client or not hasattr(self.okx_client, "get_positions"):
            return
        if not self.order_executor or not hasattr(self.order_executor, "reconcile_stop_loss_states"):
            return
        try:
            positions = self.okx_client.get_positions() or []
        except Exception as e:
            logger.warning(f"Startup stop-loss reconcile: get_positions failed: {e}")
            return
        if not positions:
            logger.info("Startup stop-loss reconcile skipped: no positions returned")
            return
        active_symbols: set = set()
        for p in positions:
            if not isinstance(p, dict):
                continue
            inst = p.get("instId") or p.get("symbol")
            if not inst:
                continue
            try:
                pos = float(p.get("pos", 0) or 0)
            except (ValueError, TypeError):
                pos = 0.0
            if pos != 0:
                active_symbols.add(inst)
        if not active_symbols:
            logger.warning("Startup stop-loss reconcile skipped: positions present but no non-zero size")
            return
        cleaned = self.order_executor.reconcile_stop_loss_states(active_symbols)
        logger.info(f"Startup stop-loss reconcile: active={sorted(active_symbols)}, cleaned={cleaned}")

    async def _health_scoring_loop(self):
        """健康度评分循环"""
        while True:
            try:
                await asyncio.sleep(self._get_loop_interval("health_scoring", 30))

                # 收集各组件健康状态
                component_scores = {}
                cpu_percent = 50.0
                memory = None

                # CPU健康度
                try:
                    cpu_percent = await asyncio.to_thread(psutil.cpu_percent, interval=1)
                    component_scores["cpu"] = max(0, 100 - cpu_percent * 2)
                except Exception:
                    component_scores["cpu"] = 50

                # 内存健康度
                try:
                    memory = psutil.virtual_memory()
                    component_scores["memory"] = max(0, 100 - memory.percent)
                except Exception:
                    component_scores["memory"] = 50

                # 算力调度反馈：将CPU/内存数据推送给 ComputeScheduler（过载降频闭环）
                try:
                    if hasattr(self, 'compute_scheduler') and self.compute_scheduler is not None and memory is not None:
                        self.compute_scheduler.update_cpu_memory(cpu_percent, memory.percent)
                        # CPU过载时自动延长非实时循环周期（降频保护）
                        if cpu_percent >= self.compute_scheduler._cpu_critical_threshold:
                            self.set_loop_intervals({
                                "health_scoring": 60,
                                "auto_recovery": 120,
                                "monitoring": 600,
                                "analysis": 7200,
                                "learning": 600,
                            })
                        elif cpu_percent < self.compute_scheduler._cpu_throttle_threshold * 0.85:
                            # CPU恢复：恢复默认周期
                            self.set_loop_intervals({
                                "health_scoring": 30,
                                "auto_recovery": 60,
                                "monitoring": 300,
                                "analysis": 3600,
                                "learning": 300,
                            })
                except Exception as e:
                    logger.debug(f"ComputeScheduler CPU feedback error: {e}")

                # API健康度
                try:
                    api_latency = self.performance_monitor.get_api_latency() if hasattr(self.performance_monitor, 'get_api_latency') else 0
                    component_scores["api"] = max(0, 100 - api_latency / 50)
                except Exception:
                    component_scores["api"] = 50

                # Redis健康度
                try:
                    # 如果Redis在配置中禁用，给予满分（不拖累整体健康度）
                    redis_config = self.config.get("redis", {})
                    if redis_config.get("enabled", True) is False:
                        component_scores["redis"] = 100
                    else:
                        redis_available = self.redis_cache.is_available() if hasattr(self.redis_cache, 'is_available') else False
                        component_scores["redis"] = 100 if redis_available else 30
                except Exception:
                    component_scores["redis"] = 30

                # DataCleaner健康度
                try:
                    if hasattr(self, 'data_cleaner') and self.data_cleaner is not None:
                        dc_health = self.data_cleaner.get_health_status()
                        dc_score = 100.0
                        if dc_health.get("status") == "degraded":
                            dc_score = 60.0
                        if dc_health.get("error_count", 0) > 0:
                            dc_score = max(30.0, dc_score - dc_health["error_count"] * 5)
                        # 检查是否有分级清理逾期
                        overdue_tiers = sum(
                            1 for t in dc_health.get("last_tier_runs", {}).values()
                            if t.get("overdue", False)
                        )
                        dc_score = max(20.0, dc_score - overdue_tiers * 10)
                        component_scores["data_cleaner"] = dc_score
                except Exception:
                    component_scores["data_cleaner"] = 50

                # 计算综合健康度
                overall_score = sum(component_scores.values()) / len(component_scores)

                # 保存健康状态
                health_data = {
                    "overall_score": round(overall_score, 1),
                    "overall_level": "good" if overall_score >= 75 else "warning" if overall_score >= 50 else "critical",
                    "timestamp": datetime.now().isoformat(),
                    "components": {k: {"score": round(v, 1)} for k, v in component_scores.items()},
                    "data_cleaner": dc_health if 'dc_health' in dir() else None,
                }

                import json as _json
                os.makedirs("./data", exist_ok=True)
                atomic_write_json("./data/health_status.json", health_data)

                # 记录指标
                self.metrics_collector.record("system_cpu_usage", component_scores.get("cpu", 0))
                self.metrics_collector.record("system_memory_usage", component_scores.get("memory", 0))
                self.metrics_collector.record("system_api_health", component_scores.get("api", 0))

                # 打通「指标→阈值→告警→通知」闭环：将已采集的系统/性能指标送入规则引擎。
                # 仅评估系统资源类指标（CPU/内存/API 延迟），触发动作仅为通知/日志清理，
                # 不涉及任何交易动作（pause/close 等 action 未注册 handler，天然为空操作）。
                try:
                    alert_metrics = {
                        "cpu_usage_pct": cpu_percent,
                        "memory_usage_pct": memory.percent if memory is not None else 0,
                        "api_latency_ms": locals().get("api_latency", 0),
                    }
                    await self.alert_rule_engine.evaluate_metrics(alert_metrics)
                except Exception as e:
                    logger.debug(f"AlertRuleEngine evaluate error: {e}")

                if overall_score < 50:
                    logger.warning(f"System health score critical: {overall_score:.1f}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health scoring error: {e}")
                await asyncio.sleep(self._get_loop_interval("health_scoring", 30))

    def _export_risk_gate_status(self) -> None:
        """导出 L4/L5 五层风控状态到文件，供 dashboard（独立进程）读取真实暂停状态。

        手动平仓后 L4 连续亏损暂停会阻止开仓，但该状态此前不导出，前端无法感知，
        导致「长时间无交易」却看不到原因。此方法每轮导出 L4/L5 状态。
        """
        try:
            import os as _os
            import json as _json
            if self.risk_gate is None:
                return
            status = self.risk_gate.get_status() if hasattr(self.risk_gate, "get_status") else {}
            export = {
                "l4": status.get("L4_daily", {}) or {},
                "l5": status.get("L5_emergency", {}) or {},
                "trading_paused": bool(status.get("trading_paused", False)),
                "last_update": datetime.now().isoformat(),
            }
            path = "./data/risk_gate_status.json"
            _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(export, f, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Risk gate status export error: {e}")

    def _export_observability_stats(self) -> None:
        """导出「进程内实时」可观测统计到文件，供独立 dashboard 进程读取。

        dashboard 与主交易进程分离，无法通过 _get_scheduler_attr 访问主进程的
        signal_processor / ops_self_heal / auto_optimization / top_level_agi 等
        进程内组件（这些组件只在主进程存在）。此方法每轮把这些组件的统计汇总到
        data/observability_stats.json，dashboard 端点读文件即可展示真实数据，
        避免「拒单分析」子面板因后端 503 长期卡在「加载中」。
        """
        try:
            import os as _os
            import json as _json
            export = {"last_update": datetime.now().isoformat()}

            try:
                from core.signal_flow_stats import get_signal_flow_stats
                export["signal_flow"] = get_signal_flow_stats()
            except Exception as e:
                logger.debug(f"Signal flow stats export error: {e}")

            sp = getattr(self, "signal_processor", None)
            if sp is not None and hasattr(sp, "get_regime_gate_stats"):
                try:
                    export["regime_gate"] = sp.get_regime_gate_stats()
                except Exception:
                    pass
            if sp is not None and hasattr(sp, "get_perception_stats"):
                try:
                    export["perception"] = sp.get_perception_stats()
                except Exception:
                    pass

            orch = getattr(self, "ops_self_heal", None)
            if orch is not None and hasattr(orch, "get_self_heal_summary"):
                try:
                    export["ops_self_heal"] = orch.get_self_heal_summary()
                except Exception:
                    pass

            orch = getattr(self, "auto_optimization", None)
            if orch is not None and hasattr(orch, "get_summary"):
                try:
                    export["auto_optimization"] = orch.get_summary()
                except Exception:
                    pass

            orch = getattr(self, "top_level_agi", None)
            if orch is not None and hasattr(orch, "get_status"):
                try:
                    export["top_level_agi"] = orch.get_status()
                except Exception:
                    pass

            orch = getattr(self, "agi_orchestrator", None)
            if orch is not None and hasattr(orch, "get_last_report"):
                try:
                    export["agi_orchestrator"] = orch.get_last_report()
                except Exception:
                    pass
            if orch is not None and hasattr(orch, "get_decision_history"):
                try:
                    export["agi_decision_history"] = orch.get_decision_history(limit=50)
                except Exception:
                    pass

            path = "./data/observability_stats.json"
            _os.makedirs(_os.path.dirname(path) or ".", exist_ok=True)
            # default=str 兜底：组件统计里若混入 numpy/枚举等非 JSON 类型，降级为字符串
            # 而非整体导出失败，避免影响主循环。
            with open(path, "w", encoding="utf-8") as f:
                _json.dump(export, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.debug(f"Observability stats export error: {e}")

    def _reconcile_event_order_state(self, from_ts=None, orphan_timeout_seconds: int = 600) -> None:
        """P1 重放恢复：从事件日志重建订单生命周期并对账（只读，不修改状态）。

        - 启动时调用一次：发现崩溃窗口内的悬单（已下单未成交）并告警
        - 自动恢复循环周期性调用：持续兜底，但仅在发现悬单时才输出告警，避免日志刷屏
        """
        if self.event_replayer is None:
            return
        try:
            report = self.event_replayer.reconcile(
                from_ts=from_ts, orphan_timeout_seconds=orphan_timeout_seconds,
            )
            s = report.get("summary", {})
            orphans = report.get("orphan_placed_no_fill", []) or []
            if orphans:
                # 悬单 = 风险信号，必须可见
                logger.warning(
                    f"Event replay reconcile: {s.get('open_positions', 0)} open / "
                    f"{s.get('in_flight', 0)} in-flight / "
                    f"{s.get('orphan_placed_no_fill', 0)} orphan_placed_no_fill"
                )
                for o in orphans[:10]:
                    logger.warning(
                        f"  ⚠ orphan order (placed but unfilled): {o.get('symbol')} "
                        f"order={o.get('exchange_order_id')} placed_at={o.get('placed_at')}"
                    )
            else:
                logger.debug(
                    f"Event replay reconcile OK: {s.get('placed', 0)} placed / "
                    f"{s.get('filled', 0)} filled / {s.get('closed', 0)} closed"
                )
        except Exception as e:
            logger.debug(f"Event replay reconcile error: {e}")

    def _consume_risk_gate_reset_signal(self) -> None:
        """消费跨进程风控重置信号（由 dashboard /api/risk/reset_pause 写入）。

        dashboard 与主进程分离，无法直接调用内存中的 risk_gate.reset_daily()；
        通过 data/risk_gate_reset.json 信号文件触发 L4/L5 重置，恢复开仓能力。
        解决「手动平仓后连续亏损暂停导致长时间无交易且重置按钮无效」的问题。
        """
        try:
            import os as _os
            import json as _json
            path = "./data/risk_gate_reset.json"
            if not _os.path.exists(path):
                return
            reason = "manual reset"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = _json.load(f) or {}
                reason = payload.get("reason", reason)
            except Exception:
                pass
            if self.risk_gate is not None:
                if hasattr(self.risk_gate, "reset_daily"):
                    self.risk_gate.reset_daily()
                if hasattr(self.risk_gate, "reset_emergency"):
                    self.risk_gate.reset_emergency()
                logger.warning(f"Risk gate reset applied via cross-process signal: {reason}")
            try:
                _os.remove(path)
            except OSError:
                pass
        except Exception as e:
            logger.debug(f"Risk gate reset signal error: {e}")

    def _consume_kill_switch_signal(self) -> None:
        """消费跨进程 Kill Switch 信号（由 dashboard /api/kill_switch/toggle 写入）。

        dashboard 与主进程分离，无法直接调用内存中的 risk_gate.enable/disable_kill_switch()；
        通过 data/kill_switch_toggle.json 信号文件触发，实现 Dashboard 一键开关。
        """
        try:
            import os as _os
            import json as _json
            path = "./data/kill_switch_toggle.json"
            if not _os.path.exists(path):
                return
            payload = {}
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = _json.load(f) or {}
            except Exception:
                pass
            enable = bool(payload.get("enable", False))
            reason = payload.get("reason", "dashboard_manual")
            by = payload.get("by", "dashboard")
            if self.risk_gate is not None:
                if enable:
                    self.risk_gate.enable_kill_switch(reason=reason, by=by)
                else:
                    self.risk_gate.disable_kill_switch(reason=reason, by=by)
                logger.warning(
                    f"Kill Switch {'ENABLED' if enable else 'DISABLED'} via cross-process signal: {reason}"
                )
            try:
                _os.remove(path)
            except OSError:
                pass
        except Exception as e:
            logger.debug(f"Kill switch signal error: {e}")

    def _kill_switch_auto_trip_check(self) -> None:
        """企业级自适应：严重风险信号自动 fail-closed Kill Switch（人工解除，不自动解禁）。

        联动两类「需要人工介入」的紧急风险信号，当任一出现时自动 enable：
          1. EquityMonitor 进入 EMERGENCY（权益大幅回撤/紧急冻结）
          2. RiskGate L5 EmergencyCircuitBreaker 已触发（极端行情瞬间熔断）
        触发后仅禁止新开仓（平仓穿透），且保持 fail-closed——本方法永不自动 disable，
        需运维通过 Dashboard /api/kill_switch/toggle 显式解除，避免风险期间自动恢复。
        """
        try:
            if self.risk_gate is None:
                return
            if self.risk_gate.is_kill_switch_enabled():
                return  # 已启用，无需重复触发

            reasons = []
            em = getattr(self, "equity_monitor", None)
            if em is not None:
                try:
                    if not em.is_new_position_allowed():
                        reasons.append("equity_monitor_emergency")
                except Exception:
                    pass
            try:
                if self.risk_gate.is_emergency_triggered():
                    reasons.append("l5_emergency_circuit_breaker")
            except Exception:
                pass

            if reasons:
                reason = "auto_trip: " + ",".join(reasons)
                self.risk_gate.enable_kill_switch(reason=reason, by="auto")
                logger.warning(f"[KILL_SWITCH] auto fail-closed triggered: {reason}")
        except Exception as e:
            logger.debug(f"Kill switch auto trip check error: {e}")

    async def _auto_recovery_loop(self):
        """自动恢复检测循环（带防抖：避免每分钟重复日志轰炸）"""
        _last_recovery_log_time = None   # 防抖：上次日志时间
        _last_recovery_log_minutes = 0   # 防抖：上次日志时的无交易分钟数

        while True:
            try:
                await asyncio.sleep(self._get_loop_interval("auto_recovery", 60))

                # P5: 跨进程风控重置 + L4/L5 状态导出（手动平仓后连续亏损暂停的恢复入口）
                try:
                    self._consume_risk_gate_reset_signal()
                    self._export_risk_gate_status()
                except Exception as e:
                    logger.debug(f"Risk gate reset/status sync error: {e}")

                # P5: 可观测统计跨进程导出（拒单分析子面板：regime_gate/perception/
                # ops_self_heal/auto_optimization/top_level_agi）
                try:
                    self._export_observability_stats()
                except Exception as e:
                    logger.debug(f"Observability stats sync error: {e}")

                # P0: 跨进程 Kill Switch 信号（Dashboard 一键开关）
                try:
                    self._consume_kill_switch_signal()
                except Exception as e:
                    logger.debug(f"Kill switch signal sync error: {e}")

                # P0: 自适应风控联动自动触发（L5 熔断 / 权益 EMERGENCY → auto fail-closed）
                try:
                    self._kill_switch_auto_trip_check()
                except Exception as e:
                    logger.debug(f"Kill switch auto trip error: {e}")

                # P1: 定时事件溯源对账（每 15 分钟，仅发现悬单时告警）
                if not hasattr(self, "_last_event_reconcile_ts"):
                    self._last_event_reconcile_ts = datetime.min
                try:
                    from datetime import timedelta as _td
                    if (datetime.now() - self._last_event_reconcile_ts).total_seconds() >= 900:
                        self._reconcile_event_order_state(
                            from_ts=datetime.now() - _td(hours=24),
                        )
                        self._last_event_reconcile_ts = datetime.now()
                except Exception as e:
                    logger.debug(f"Periodic event reconcile error: {e}")

                # 检测长时间无交易（防抖：仅变化超过 10% 或间隔超过 30 分钟才重新日志）
                try:
                    import sqlite3
                    conn = sqlite3.connect(self.config.get("sqlite", {}).get("db_path", "./data/trading.db"))
                    c = conn.cursor()
                    c.execute("SELECT MAX(create_time) FROM trade_records")
                    row = c.fetchone()
                    conn.close()

                    if row and row[0]:
                        last_trade_time = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                        minutes_since = (datetime.now() - last_trade_time).total_seconds() / 60

                        if minutes_since > 60:
                            # 防抖：仅在分钟数变化 >10% 或距上次日志 >30 分钟时才记录
                            should_log = (
                                _last_recovery_log_time is None or
                                (datetime.now() - _last_recovery_log_time).total_seconds() > 1800 or
                                abs(minutes_since - _last_recovery_log_minutes) / max(_last_recovery_log_minutes, 1) > 0.1
                            )
                            if should_log:
                                logger.warning(f"No trade activity for {minutes_since:.0f} minutes, triggering recovery check")
                                _last_recovery_log_time = datetime.now()
                                _last_recovery_log_minutes = minutes_since
                            else:
                                logger.debug(f"No trade activity: {minutes_since:.0f} min (debounced)")

                            await self.trading_recovery.manual_recovery()
                except Exception as e:
                    logger.debug(f"Trade activity check error: {e}")

                # 检测API延迟异常
                try:
                    if hasattr(self.performance_monitor, 'get_api_latency'):
                        latency = self.performance_monitor.get_api_latency()
                        if latency > 5000:
                            logger.warning(f"API latency critical: {latency:.0f}ms, triggering reconnect")
                            # 触发WebSocket重连
                            if hasattr(self.okx_client, '_reconnect'):
                                await self.okx_client._reconnect()
                except Exception as e:
                    logger.debug(f"API latency check error: {e}")

                # P5: L4风控自动恢复 - 连续亏损暂停超过2小时后自动重置
                try:
                    if self.risk_gate and self.risk_gate.is_trading_paused():
                        l4_state = self.risk_gate.get_l4_state()
                        if l4_state and l4_state.get("trading_paused"):
                            pause_reason = l4_state.get("pause_reason", "")
                            # 仅对连续亏损暂停做自动恢复（硬亏损限制不自动恢复）
                            if "连续亏损" in pause_reason:
                                # 检查暂停持续时间
                                if hasattr(self.risk_gate._l4, '_pause_started_at'):
                                    pause_duration = time.time() - self.risk_gate._l4._pause_started_at
                                    # 超过2小时自动恢复
                                    if pause_duration > 7200:
                                        logger.warning(
                                            f"L4 consecutive loss pause exceeds 2h ({pause_duration/3600:.1f}h), "
                                            f"auto-resetting risk control"
                                        )
                                        self.risk_gate._l4.reset()
                                        logger.info("L4 risk control auto-reset: trading resumed")
                except Exception as e:
                    logger.debug(f"L4 auto-recovery check error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto recovery loop error: {e}")
                await asyncio.sleep(self._get_loop_interval("auto_recovery", 60) // 2)

    async def _monitoring_loop(self):
        """综合监控循环"""
        while True:
            try:
                await asyncio.sleep(self._get_loop_interval("monitoring", 300))

                # 保存监控指标
                try:
                    self.metrics_collector.save_to_file("./data/metrics_latest.json")
                except Exception as e:
                    logger.debug(f"Metrics save error: {e}")

                # 检查系统整体状态
                try:
                    with open("./data/health_status.json", "r", encoding="utf-8") as f:
                        import json as _json
                        health = _json.load(f)

                    score = health.get("overall_score", 0)
                    if score < 50:
                        logger.warning(f"System health degraded: {score:.1f}/100")
                        # 发送通知
                        await self.notification_manager.broadcast(
                            f"System health degraded: {score:.1f}/100. Components: {health.get('components', {})}",
                            "System Health Alert",
                            "WARNING"
                        )
                except Exception as e:
                    logger.debug(f"Health check error: {e}")

                # 心跳检测：确保所有持仓都有止损条件单
                try:
                    restored = await self.conditional_order_manager.heartbeat_check()
                    if restored > 0:
                        logger.warning(f"Conditional order heartbeat: restored {restored} missing SL orders")
                except Exception as e:
                    logger.debug(f"Heartbeat check error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self._get_loop_interval("monitoring", 300) // 5)

    async def _compliance_check_loop(self):
        """P22-8: 生产级红线合规检查循环"""
        # 首次检查延迟30秒，等待系统完成初始化
        await asyncio.sleep(30)
        
        while True:
            try:
                interval = self._get_loop_interval("compliance_check", 3600)
                await asyncio.sleep(interval)
                
                if not self.compliance_checker.enabled:
                    continue
                
                result = await self.compliance_checker.run_full_check()
                
                if not result["passed"]:
                    # 合规检查失败 - 发送告警
                    issue_names = [i["name"] for i in result["issues"]]
                    logger.error(
                        f"P22-8: COMPLIANCE CHECK FAILED - {len(result['issues'])} issues: "
                        f"{', '.join(issue_names[:5])}"
                    )
                    
                    # 自动修复
                    if result.get("auto_fix_actions"):
                        logger.info(f"P22-8: Auto-fix applied: {result['auto_fix_actions']}")
                    
                    # 发送通知
                    try:
                        await self.notification_manager.broadcast(
                            f"Compliance Check FAILED: {len(result['issues'])} issues\n"
                            f"Issues: {', '.join(issue_names[:5])}\n"
                            f"Auto-fix: {result.get('auto_fix_actions', [])}",
                            "P22-8: Compliance Alert",
                            "CRITICAL"
                        )
                    except Exception:
                        pass
                    
                    # 连续失败3次以上，触发紧急处理
                    # P29: 仅严重问题（资金不足、熔断、负权益）触发紧急暂停，集中度等问题不暂停
                    CRITICAL_ISSUE_KEYWORDS = ["资金不足", "可用余额不足", "保证金不足", "熔断", "circuit", "负权益", "negative equity"]
                    has_critical_issue = any(
                        any(kw in issue.get("name", "") or kw in issue.get("message", "") 
                            for kw in CRITICAL_ISSUE_KEYWORDS)
                        for issue in result.get("issues", [])
                    )
                    
                    if result["consecutive_failures"] >= 3 and has_critical_issue:
                        logger.critical(
                            f"P29: Compliance check failed {result['consecutive_failures']} times consecutively - "
                            f"CRITICAL issues detected, triggering emergency protocol"
                        )
                        # 暂停所有策略开仓
                        for strategy_attr in [
                            "grid_strategy", "trend_strategy", "scalping_strategy",
                            "arbitrage_strategy", "spot_grid_strategy", "spot_martingale_strategy",
                        ]:
                            strategy = getattr(self, strategy_attr, None)
                            if strategy and hasattr(strategy, "pause_opening"):
                                strategy.pause_opening = True
                                logger.warning(f"P29: Emergency pause opening for {strategy_attr}")
                    elif result["consecutive_failures"] >= 3:
                        logger.warning(
                            f"P29: Compliance check failed {result['consecutive_failures']} times - "
                            f"non-critical issues only, skipping emergency protocol"
                        )
                
                else:
                    # P29: 合规检查通过时，自动恢复策略开仓
                    for strategy_attr in [
                        "grid_strategy", "trend_strategy", "scalping_strategy",
                        "arbitrage_strategy", "spot_grid_strategy", "spot_martingale_strategy",
                    ]:
                        strategy = getattr(self, strategy_attr, None)
                        if strategy and hasattr(strategy, "pause_opening") and getattr(strategy, "pause_opening", False):
                            strategy.pause_opening = False
                            logger.info(f"P29: Auto-resumed opening for {strategy_attr} (compliance check passed)")
                    
                    if result.get("warnings"):
                        # 有警告
                        warning_names = [w["name"] for w in result["warnings"]]
                        logger.warning(
                            f"P22-8: Compliance check WARN - {len(result['warnings'])} warnings: "
                            f"{', '.join(warning_names[:3])}"
                        )
                    else:
                        logger.debug(
                            f"P22-8: Compliance check PASS - "
                            f"{result['passed_checks']}/{result['total_checks']} checks passed"
                        )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"P22-8: Compliance check loop error: {e}")
                await asyncio.sleep(300)

    async def _health_check(self):
        checks = [
            ("Redis", self.redis_cache.health_check()),
            ("SQLite", self.sqlite_storage.health_check())
        ]

        for name, result in checks:
            if result:
                if self._health_suppressed.get(name):
                    logger.info(f"{name} health check recovered")
                    self._health_suppressed[name] = False
                    self._health_fail_count[name] = 0
                else:
                    logger.debug(f"{name} health check passed")
            else:
                count = self._health_fail_count.get(name, 0) + 1
                self._health_fail_count[name] = count
                if count <= 3:
                    logger.error(f"{name} health check failed (x{count})")
                elif count == 10:
                    logger.warning(f"{name} health check consistently failed ({count}x), suppressing further logs")
                    self._health_suppressed[name] = True
                else:
                    logger.debug(f"{name} health check failed (x{count}, suppressed)")
    
    async def _subscribe_market_data(self):
        symbols = get_all_symbols(self.config)

        async def tick_handler(tick):
            self.redis_cache.set_tick(tick)

        self.market_data_service.register_tick_callback(tick_handler)

        await self.market_data_service.subscribe_market_data(symbols)

        logger.info(f"Subscribed to market data for {len(symbols)} symbols")
    
    async def _start_signal_listener(self):
        await self.signal_processor.start_signal_listener(self.redis_cache)
    
    async def _process_signal(self, signal_data: Dict[str, Any]):
        await self.signal_processor._process_signal(signal_data)
    
    async def _analysis_loop(self):
        while True:
            try:
                analysis = self.analyzer.analyze_all_strategies()
                report = self.analyzer.generate_analysis_report()

                if report["summary"]["performance_rating"] in ["差", "一般"]:
                    logger.warning(f"Performance alert: {report['summary']['performance_rating']}, PnL: {report['summary']['total_pnl']:.2f}")

                if analysis["shortcomings"]["total_shortcomings"] > 5:
                    logger.warning(f"High number of shortcomings: {analysis['shortcomings']['total_shortcomings']}")
            except Exception as e:
                logger.error(f"Analysis loop error: {e}")

            await asyncio.sleep(self._get_loop_interval("analysis", 3600))

    async def _contribution_analysis_loop(self):
        """策略贡献度分析循环：定期执行多维度贡献度评估和快照持久化"""
        contrib_cfg = self.config.get("contribution", {})
        analysis_interval = contrib_cfg.get("analysis_interval_minutes", 120) * 60
        persist_interval = contrib_cfg.get("snapshot_persist_interval_minutes", 60) * 60
        last_persist = datetime.min

        while True:
            try:
                if not self.contribution_analyzer:
                    await asyncio.sleep(600)
                    continue

                # 同步动态分配权重
                try:
                    if hasattr(self.adaptive_controller, 'get_allocations'):
                        allocations = self.adaptive_controller.get_allocations()
                        if allocations:
                            self.contribution_analyzer.set_dynamic_allocations(allocations)
                except Exception as ae:
                    logger.debug(f"Contribution sync allocations error: {ae}")

                # 执行7天窗口完整分析
                snapshot = self.contribution_analyzer.analyze(window="7d")
                overall_health = snapshot.overall_health_score

                # P2: 贡献度分析结果回传调整策略权重（闭环反馈）
                try:
                    suggestions = self.contribution_analyzer.get_capital_reallocation_suggestions(snapshot=snapshot)
                    if suggestions and hasattr(self.adaptive_controller, 'apply_contribution_feedback'):
                        self.adaptive_controller.apply_contribution_feedback(suggestions)
                except Exception as fb_err:
                    logger.debug(f"Contribution feedback error (non-blocking): {fb_err}")

                # 健康度告警
                alert_threshold = contrib_cfg.get("alerts", {}).get("health_danger_threshold", 35)
                if overall_health < alert_threshold:
                    logger.warning(
                        f"[CONTRIBUTION] System health LOW: {overall_health:.1f}/100 "
                        f"(threshold={alert_threshold})"
                    )

                # 衰退策略告警
                declining_count = sum(
                    1 for c in snapshot.strategies.values()
                    if c.trend == "declining"
                )
                total_active = sum(
                    1 for c in snapshot.strategies.values()
                    if c.total_trades > 0
                )
                declining_ratio = declining_count / max(total_active, 1)
                declining_threshold = contrib_cfg.get("alerts", {}).get("declining_ratio_threshold", 0.5)
                if declining_ratio > declining_threshold and total_active >= 2:
                    logger.warning(
                        f"[CONTRIBUTION] {declining_count}/{total_active} strategies declining "
                        f"({declining_ratio:.0%}), consider review"
                    )
                    
                    # P3-3: 全策略衰退时自动恢复
                    if declining_ratio >= 1.0 and overall_health < 35:
                        await self._auto_recover_declining_strategies(
                            snapshot, declining_count, overall_health
                        )

                # 协同效应告警
                synergy_low = contrib_cfg.get("alerts", {}).get("synergy_low_threshold", 0.3)
                if snapshot.synergy_score < synergy_low and len(snapshot.strategies) >= 2:
                    logger.warning(
                        f"[CONTRIBUTION] Strategy synergy LOW: {snapshot.synergy_score:.2f}"
                    )

                # 持久化快照（按独立间隔）
                now = datetime.now()
                if (now - last_persist).total_seconds() >= persist_interval:
                    self.contribution_analyzer.persist_snapshot(snapshot)
                    last_persist = now

                logger.debug(
                    f"[CONTRIBUTION] Analysis complete: health={overall_health:.1f}, "
                    f"synergy={snapshot.synergy_score:.2f}, "
                    f"lifecycle={snapshot.lifecycle_summary}"
                )

            except Exception as e:
                logger.error(f"Contribution analysis loop error: {e}")

            await asyncio.sleep(analysis_interval)

    async def _auto_recover_declining_strategies(self, snapshot, declining_count: int, health: float):
        """P3-3: 全策略衰退自动恢复
        
        当所有活跃策略都处于衰退趋势且系统健康度低于阈值时，自动执行恢复操作：
        1. 重置智能体过拒状态
        2. 降低信号质量阈值
        3. 清理策略性能计数器
        4. 记录恢复事件
        """
        try:
            # 冷却保护：6小时内不重复触发
            if hasattr(self, '_last_auto_recovery_ts'):
                cooldown = timedelta(hours=6)
                if datetime.now() - self._last_auto_recovery_ts < cooldown:
                    logger.debug(
                        f"P3-3: Auto-recovery cooldown active "
                        f"(last={self._last_auto_recovery_ts.strftime('%H:%M')}, "
                        f"next allowed after {(self._last_auto_recovery_ts + cooldown).strftime('%H:%M')})"
                    )
                    return
            
            logger.warning(
                f"P3-3: Auto-recovery triggered - {declining_count} strategies declining, "
                f"health={health:.1f}/100"
            )
            
            recovery_actions = []
            
            # 1. 重置智能体过拒状态
            if hasattr(self, 'intelligent_agent') and self.intelligent_agent:
                try:
                    # P2-12: 检查已退出策略，跳过恢复
                    for strategy_name in list(snapshot.strategies.keys()):
                        if hasattr(self.intelligent_agent, 'is_strategy_exited') and \
                           self.intelligent_agent.is_strategy_exited(strategy_name):
                            logger.info(
                                f"P2-12: Skipping auto-recovery for '{strategy_name}' "
                                f"— strategy is permanently exited"
                            )
                            recovery_actions.append(f"{strategy_name} skipped (exited)")
                    
                    if hasattr(self.intelligent_agent, '_auto_reset'):
                        self.intelligent_agent._auto_reset()
                        recovery_actions.append("intelligent_agent reset")
                except Exception as e:
                    logger.debug(f"P3-3: Agent reset failed: {e}")
            
            # 2. 降低信号质量阈值（临时放宽以激活交易）
            if hasattr(self, 'adaptive_controller') and self.adaptive_controller:
                try:
                    strategies_config = self.adaptive_controller.config.get("strategies", {})
                    for strategy_name in snapshot.strategies:
                        if snapshot.strategies[strategy_name].trend == "declining":
                            current_quality = strategies_config.get(strategy_name, {}).get(
                                "min_signal_quality"
                            )
                            if current_quality is not None:
                                # 降低10%但不超过下限，且只降不升（避免与空闲优化已放宽到0.10的阈值打架）
                                new_quality = min(current_quality, max(0.10, current_quality * 0.9))
                                strategies_config[strategy_name]["min_signal_quality"] = new_quality
                                # 同步热更新运行中的策略实例
                                optimizer = getattr(self.adaptive_controller, '_strategy_optimizer', None)
                                if optimizer:
                                    instance = optimizer._strategy_instances.get(strategy_name)
                                    if instance and hasattr(instance, "apply_param_update"):
                                        try:
                                            instance.apply_param_update({"min_signal_quality": new_quality})
                                        except Exception:
                                            pass
                                recovery_actions.append(
                                    f"{strategy_name} quality {current_quality:.3f}->{new_quality:.3f}"
                                )
                except Exception as e:
                    logger.debug(f"P3-3: Quality threshold adjust failed: {e}")
            
            # 3. 清理智能体性能计数器（性能追踪存于 intelligent_agent._strategy_performance）
            if hasattr(self, 'intelligent_agent') and self.intelligent_agent:
                try:
                    perf = getattr(self.intelligent_agent, '_strategy_performance', None)
                    if isinstance(perf, dict):
                        for strategy_name in snapshot.strategies:
                            if perf.pop(strategy_name, None) is not None:
                                recovery_actions.append(f"{strategy_name} performance cleared")
                except Exception as e:
                    logger.debug(f"P3-3: Performance clear failed: {e}")
            
            # 4. 记录恢复事件
            logger.info(
                f"P3-3: Auto-recovery complete: {len(recovery_actions)} actions: "
                f"{', '.join(recovery_actions[:5])}"
            )
            
            # 防止短时间内重复触发
            self._last_auto_recovery_ts = datetime.now()
            
        except Exception as e:
            logger.error(f"P3-3: Auto-recovery error: {e}")

    async def _optimization_loop(self):
        while True:
            try:
                recommendations = await self.optimizer.optimize_all_strategies()

                optimized_count = sum(1 for s in recommendations.values() if isinstance(s, dict) and s.get("optimized"))
                if optimized_count > 0:
                    logger.info(f"Optimization completed: {optimized_count} strategies optimized")

                    applied = await self.optimizer.apply_optimizations(recommendations)
                    if applied["total_applied"] > 0:
                        await self.optimizer.persist_config()
                        logger.info(f"Applied {applied['total_applied']} optimizations and persisted config")

                progress = self.optimizer.get_learning_progress()
                if progress["improvement_rate"] > 0.5:
                    logger.info(f"Learning improvement rate: {progress['improvement_rate']:.1%}")
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")

            await asyncio.sleep(self._get_loop_interval("optimization", 86400))

    def _map_params_to_backtest(self, strategy_name: str, params: Dict[str, Any]):
        """将策略参数映射到回测引擎的均线交叉参数。"""
        if strategy_name == "trend":
            fast = int(round(params.get("ema_fast", params.get("fast_period", 5))))
            slow = int(round(params.get("ema_slow", params.get("slow_period", 20))))
        else:
            fast = int(round(params.get("ema_fast", params.get("fast_period", 5))))
            slow = int(round(params.get("ema_slow", params.get("slow_period", 20))))
        if fast >= slow:
            fast = max(1, slow - 1)
        return fast, slow

    def _build_param_fitness_fn(self, strategy_name: str, symbol: str,
                                candles: list) -> Any:
        """构建真实适应度函数：基于已加载K线的回测评估，返回夏普比率。

        修复：scalping 参数(RSI/止盈/止损/持仓时长)此前被 _map_params_to_backtest
        压平成不存在的 ema_fast/ema_slow 键 → 恒 MA(5,20)，导致所有候选参数回测
        结果相同、fitness 恒等退化为固定值。现按策略分支：scalping 走 RSI 均值回归
        回测(参数真实驱动)，其余策略走 MA 交叉回测。
        """
        def fitness(params: Dict[str, float], data=None) -> float:
            try:
                if strategy_name == "scalping":
                    res = self.backtest_engine.run_scalping_with_candles(
                        candles, symbol, strategy_name,
                        initial_capital=100.0, leverage=6,
                        rsi_period=int(round(params.get("rsi_period", 4))),
                        rsi_oversold=float(params.get("rsi_oversold", 30.0)),
                        rsi_overbought=float(params.get("rsi_overbought", 70.0)),
                        profit_target_min=float(params.get("profit_target_min", 0.008)),
                        stop_loss=float(params.get("stop_loss", 0.005)),
                        max_hold_minutes=int(round(params.get("max_hold_minutes", 10))),
                    )
                else:
                    fast, slow = self._map_params_to_backtest(strategy_name, params)
                    res = self.backtest_engine.run_with_candles(
                        candles, symbol, strategy_name,
                        initial_capital=100.0, leverage=6,
                        fast_period=fast, slow_period=slow,
                    )
                summary = res.summary()
                if "error" in summary:
                    return 0.0
                sharpe = float(summary.get("sharpe_ratio", 0.0))
                net_pnl = float(summary.get("net_pnl", 0.0))
                # 无交易或零方差时退化为净收益
                return sharpe if abs(sharpe) >= 1e-9 else net_pnl
            except Exception as e:
                logger.debug(f"Backtest fitness eval error: {e}")
                return float('-inf')
        return fitness

    def _build_param_opt_recommendation(self, strategy_name: str, result) -> Optional[Dict[str, Any]]:
        """将编排器最优参数转换为 StrategyOptimizer 可应用的建议（仅精确匹配键）。

        这是三层优化器的关键握手点：编排器(慢速全搜索) → StrategyOptimizer
        (应用+持久化+版本回滚)，ParameterAdaptor(在线微调) 随后以新配置为基线。
        """
        if not result or not result.best_params:
            return None
        strategies_cfg = self.config.get("strategies", {}) or {}
        strategy_cfg = strategies_cfg.get(strategy_name, {}) if isinstance(strategies_cfg, dict) else {}
        changes = {}
        for k, v in result.best_params.items():
            if k in strategy_cfg:
                changes[k] = v
        if not changes:
            logger.info(f"Param optimization [{strategy_name}]: no exact-match config keys to apply "
                        f"(best_params={list(result.best_params.keys())})")
            return None
        return {
            strategy_name: {
                "strategy": strategy_name,
                "optimized": True,
                "changes": changes,
                "recommendations": [],
            }
        }

    async def _run_param_optimization(self):
        """执行统一参数优化编排（真实回测评估 + 调度触发闭环）。"""
        if not self.param_optimizer or not self.backtest_engine:
            logger.debug("ParameterOptimizationOrchestrator not available, skipping param optimization")
            return

        supported = {"trend", "grid", "scalping", "arbitrage"}
        try:
            strategy_names = [n for n in self.strategy_manager.get_enabled_strategy_names()
                              if n in supported] if self.strategy_manager else ["trend"]
        except Exception:
            strategy_names = ["trend"]
        if not strategy_names:
            logger.debug("No supported strategies for param optimization")
            return

        try:
            symbols = get_all_symbols(self.config)
            symbol = symbols[0] if symbols else "BTC-USDT-SWAP"
        except Exception:
            symbol = "BTC-USDT-SWAP"

        opt_cfg = self.config.get("parameter_optimization", {})
        bar = opt_cfg.get("backtest_bar", "1H")
        days = int(opt_cfg.get("backtest_days", 30))

        for strategy_name in strategy_names:
            try:
                candles = await asyncio.to_thread(
                    self.backtest_engine.fetch_historical_klines, symbol, bar, days
                )
                if len(candles) < 20:
                    logger.warning(f"Param optimization: insufficient candles for {symbol}, skip {strategy_name}")
                    continue

                fitness_fn = self._build_param_fitness_fn(strategy_name, symbol, candles)
                close_prices = np.array([float(c["close"]) for c in candles], dtype=np.float64)
                param_defs = self.param_optimizer.define_strategy_params(strategy_name)

                # 自动寻优闭环：优化 → 回测 → 上线 → 反馈（协调器 fail-closed 门控上线）
                report = await self.auto_optimization.run_cycle(
                    strategy_name,
                    fitness_fn=fitness_fn,
                    param_defs=param_defs,
                    price_data=close_prices,
                )
                logger.info(
                    f"Auto optimization [{strategy_name}] status={report.status} "
                    f"decision={report.deploy_decision} fitness={report.best_fitness:.4f} "
                    f"feedback_score={report.feedback_score:.3f} applied={report.deploy_applied}"
                )
            except Exception as e:
                logger.error(f"Param optimization failed for {strategy_name}: {e}")

    async def _param_optimization_loop(self):
        """统一参数优化调度循环（默认每周触发一次）。

        启动后先延迟首跑：避免每次重启立即触发「4 策略 × 千次回测」的完整优化，
        与启动阶段初始化和交易信号处理竞争 CPU/事件循环资源。
        """
        first_delay = self._get_loop_interval("param_optimization_first_delay", 3600)
        logger.info(f"Param optimization first run scheduled in {first_delay}s (avoid startup contention)")
        await asyncio.sleep(first_delay)
        while True:
            try:
                await self._run_param_optimization()
            except Exception as e:
                logger.error(f"Param optimization loop error: {e}")
            await asyncio.sleep(self._get_loop_interval("param_optimization", 86400 * 7))

    async def _verification_loop(self):
        while True:
            try:
                await asyncio.sleep(self._get_loop_interval("verification", 86400 * 7))
                
                results = await self.verification_runner.run_full_verification()
                report = self.verification_runner.generate_verification_report()
                
                if report["summary"]["overall_status"] == "FAIL":
                    logger.error("Weekly verification failed!")
                    await self.alert_manager.send_alert("VERIFICATION_FAILURE", report)
                else:
                    logger.info(f"Weekly verification passed. Total return: {report['summary']['key_metrics']['total_return']:.2%}")
            except Exception as e:
                logger.error(f"Verification loop error: {e}")
    
    async def _learning_loop(self):
        while True:
            try:
                await self.online_learner.process_batch()
                
                adaptions = await self.parameter_adaptor.adapt_parameters()
                if adaptions:
                    logger.info(f"Parameter adaptions applied: {len(adaptions)}")
                
                evolution = await self.strategy_evolver.evolve_strategies()
                if len(evolution) > 0:
                    logger.info(f"Strategy evolution completed: {len(evolution)} strategies evolved")
                
                # P0: 绩效反馈评估
                feedback_summary = await self.performance_feedback.update(
                    trades=[], equity_curve=[], strategy="all"
                )
                
                # P0: 元学习器跨策略知识迁移
                meta_result = await self.meta_learner.meta_update()
                if meta_result:
                    logger.debug(f"MetaLearner updated: {meta_result.get('status', 'unknown')}")
                
                # P0: 定期衰减知识库
                if hasattr(self.knowledge_base, '_apply_decay'):
                    self.knowledge_base._apply_decay()
                
                # P0: 智能交易体成长性学习 - 从历史交易中学习优化参数
                if hasattr(self, 'intelligent_agent') and self.intelligent_agent:
                    try:
                        if self.intelligent_agent.should_learn():
                            # 从trade_journal获取历史交易记录
                            trade_records = []
                            try:
                                trade_records = self.trade_journal.get_recent_trades(limit=1000)
                            except Exception as e:
                                logger.debug(f"Failed to get trade records for agent learning: {e}")
                            
                            if trade_records:
                                learning_result = self.intelligent_agent.learn_from_history(trade_records)

                                # P1: 正期望符号白名单重排（基于近 7 天交易记录）
                                try:
                                    wl_result = self.intelligent_agent.rebuild_whitelist(trade_records)
                                    if wl_result.get("added") or wl_result.get("removed"):
                                        logger.info(
                                            f"P1: whitelist rebuilt +{len(wl_result.get('added', []))} "
                                            f"-{len(wl_result.get('removed', []))}"
                                        )
                                except Exception as e:
                                    logger.error(f"P1: whitelist rebuild error: {e}")

                                if learning_result.get("learned"):
                                    learned_params = learning_result.get("learned_params", {})
                                    if learned_params:
                                        logger.info(f"IntelligentAgent learned {len(learned_params)} params: "
                                                   f"{ {k: round(v, 4) if isinstance(v, float) else v for k, v in list(learned_params.items())[:5]} }")
                                        
                                        # 将学习到的参数同步到策略
                                        await self._apply_learned_params(learned_params)
                                    
                                    # P11: 输出增强学习结果
                                    blacklist = learning_result.get("blacklist", {})
                                    if blacklist:
                                        logger.warning(f"P11-1: Agent blacklist active: {list(blacklist.keys())}")
                                    paused = learning_result.get("paused_strategies", {})
                                    if paused:
                                        logger.warning(f"P11-2: Agent paused strategies: {list(paused.keys())}")
                                    hour_updates = learning_result.get("hour_risk_updates", [])
                                    if hour_updates:
                                        logger.info(f"P11-3: Agent hour risk updated: {hour_updates}")
                                    bl_updates = learning_result.get("blacklist_updates", [])
                                    if bl_updates:
                                        for bu in bl_updates:
                                            logger.warning(f"P11-1: Blacklist {bu['action']}: {bu.get('symbol', '')}")
                                    pause_updates = learning_result.get("pause_updates", [])
                                    if pause_updates:
                                        for pu in pause_updates:
                                            logger.warning(f"P11-2: Strategy {pu['action']}: {pu.get('strategy', '')}")
                            else:
                                logger.debug("IntelligentAgent: no trade records available for learning")
                    except Exception as e:
                        logger.error(f"IntelligentAgent learning error: {e}")
                    
            except Exception as e:
                logger.error(f"Learning loop error: {e}")

            await asyncio.sleep(self._get_loop_interval("learning", 300))
    
    async def _intelligent_analysis_loop(self):
        """企业级智能交易记录分析循环（默认 6 小时）：
        生成综合分析报告 + 优化方向报告，将关键结论记录到日志。
        """
        while True:
            try:
                if not hasattr(self, 'analysis_agent') or not self.analysis_agent:
                    return

                analysis = await self.analysis_agent.generate_analysis_report(include_adx=True)
                optimization = await self.analysis_agent.generate_optimization_report()

                summary = analysis.get("summary", {})
                opt_summary = optimization.get("summary", {})
                adx_summary = summary.get("adx_summary", {})
                logger.info(
                    f"[智能分析] 市场={summary.get('market_regime')} "
                    f"强度={summary.get('market_strength')} "
                    f"健康度={summary.get('overall_health_grade')} "
                    f"信号质量={summary.get('signal_quality_grade')} "
                    f"ADX确认={adx_summary.get('confirmed_symbols', 0)}币种"
                )
                logger.info(
                    f"[优化方向] 增仓={opt_summary.get('increase_count')} "
                    f"减仓={opt_summary.get('reduce_count')} "
                    f"优先级行动={opt_summary.get('priority_action_count')}"
                )
                for action in optimization.get("priority_actions", [])[:5]:
                    logger.info(
                        f"  - [{action.get('priority')}] {action.get('type')}: "
                        f"{action.get('reason', '')}"
                    )
            except Exception as e:
                logger.error(f"Intelligent analysis loop error: {e}")

            await asyncio.sleep(self._get_loop_interval("intelligent_analysis", 21600))

    async def _apply_learned_params(self, learned_params: dict):
        """将智能体学习到的参数应用到各策略 + P12深度集成"""
        for param_key, param_value in learned_params.items():
            try:
                # 解析参数名格式: {strategy_name}_{param_name}
                parts = param_key.split("_", 1)
                if len(parts) < 2:
                    continue
                strategy_name = parts[0]
                param_name = parts[1]

                # 更新对应策略的参数
                if strategy_name == "grid" and hasattr(self, 'grid_strategy'):
                    if hasattr(self.grid_strategy, param_name):
                        setattr(self.grid_strategy, param_name, param_value)
                        logger.debug(f"Applied learned param: grid.{param_name} = {param_value}")
                elif strategy_name == "trend" and hasattr(self, 'trend_strategy'):
                    if hasattr(self.trend_strategy, param_name):
                        setattr(self.trend_strategy, param_name, param_value)
                        logger.debug(f"Applied learned param: trend.{param_name} = {param_value}")
                elif strategy_name == "scalping" and hasattr(self, 'scalping_strategy'):
                    if hasattr(self.scalping_strategy, param_name):
                        setattr(self.scalping_strategy, param_name, param_value)
                        logger.debug(f"Applied learned param: scalping.{param_name} = {param_value}")
                elif strategy_name == "arbitrage" and hasattr(self, 'arbitrage_strategy'):
                    if hasattr(self.arbitrage_strategy, param_name):
                        setattr(self.arbitrage_strategy, param_name, param_value)
                        logger.debug(f"Applied learned param: arbitrage.{param_name} = {param_value}")
                # P12: 通过策略管理器注入参数（支持更多策略类型）
                elif hasattr(self, 'strategy_manager'):
                    try:
                        strategy = self.strategy_manager.get_strategy(strategy_name)
                        if strategy and hasattr(strategy, param_name):
                            old_val = getattr(strategy, param_name, None)
                            setattr(strategy, param_name, param_value)
                            logger.info(f"P12 Injected {strategy_name}.{param_name}: {old_val} -> {param_value}")
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Failed to apply learned param {param_key}: {e}")

        # P12: 注入时段风险乘数到策略
        if hasattr(self, 'intelligent_agent') and self.intelligent_agent:
            try:
                hour_multiplier = self.intelligent_agent.get_hour_risk_multiplier()
                if hour_multiplier < 1.0:
                    for strategy_attr in ['grid_strategy', 'trend_strategy', 'scalping_strategy', 'arbitrage_strategy']:
                        strategy = getattr(self, strategy_attr, None)
                        if strategy and hasattr(strategy, 'position_multiplier'):
                            base_mult = getattr(strategy, 'position_multiplier', 1.0)
                            adjusted = base_mult * hour_multiplier
                            setattr(strategy, 'position_multiplier', adjusted)
                            logger.debug(f"P12 Hour risk: {strategy_attr} position_multiplier={adjusted:.2f}")
            except Exception as e:
                logger.debug(f"P12 Hour risk injection error: {e}")
    
    async def _handle_signal_receive(self, context):
        context.set_stage_start(PipelineStage.SIGNAL_RECEIVE)
        context.validated_signal = context.signal
        context.set_stage_end(PipelineStage.SIGNAL_RECEIVE)
        return True
    
    async def _handle_signal_validation(self, context):
        context.set_stage_start(PipelineStage.SIGNAL_VALIDATION)
        result = await self.signal_pipeline.process(context.validated_signal)
        if result["valid"]:
            context.validated_signal = result["signal"]
            context.set_stage_end(PipelineStage.SIGNAL_VALIDATION)
            return True
        context.set_stage_end(PipelineStage.SIGNAL_VALIDATION)
        return False
    
    async def _handle_decision_making(self, context):
        context.set_stage_start(PipelineStage.DECISION_MAKING)
        decision = await self.decision_coordinator.submit_decision(
            context.validated_signal
        )
        if decision:
            context.decision = decision
            context.set_stage_end(PipelineStage.DECISION_MAKING)
            return True
        context.set_stage_end(PipelineStage.DECISION_MAKING)
        return False
    
    async def _handle_order_generation(self, context):
        context.set_stage_start(PipelineStage.ORDER_GENERATION)
        order = await self.decision_executor.generate_order(context.decision)
        if order:
            context.order = order
            context.set_stage_end(PipelineStage.ORDER_GENERATION)
            return True
        context.set_stage_end(PipelineStage.ORDER_GENERATION)
        return False
    
    async def _handle_order_placement(self, context):
        context.set_stage_start(PipelineStage.ORDER_PLACEMENT)
        result = await self.order_executor.execute_order(context.order)
        if result:
            context.execution_result = result
            context.set_stage_end(PipelineStage.ORDER_PLACEMENT)
            return True
        context.set_stage_end(PipelineStage.ORDER_PLACEMENT)
        return False
    
    async def _handle_execution_monitoring(self, context):
        context.set_stage_start(PipelineStage.EXECUTION_MONITORING)
        try:
            # 真实成交确认/订单状态跟踪：从执行结果与订单中提取状态
            exec_result = context.execution_result or {}
            order = context.order or {}
            symbol = order.get("symbol", "") if isinstance(order, dict) else ""
            strategy = order.get("strategy_name", "") if isinstance(order, dict) else ""

            if isinstance(exec_result, dict):
                status = exec_result.get("status", "")
                order_id = exec_result.get("order_id") or exec_result.get("ordId") or ""
            else:
                status = "unknown"
                order_id = ""

            # 记录订单事件到执行监控器（吞吐量/成功率统计）
            if hasattr(self, 'execution_monitor') and self.execution_monitor:
                try:
                    quantity = float(order.get("quantity", 0)) if isinstance(order, dict) else 0
                    price = float(order.get("price", 0)) if isinstance(order, dict) else 0
                    await self.execution_monitor.record_order_event(
                        event_type="order_execution",
                        order_id=str(order_id),
                        symbol=symbol,
                        strategy=strategy,
                        quantity=quantity,
                        price=price,
                        status=status,
                        metadata={"stage": "execution_monitoring"},
                    )
                except Exception as e:
                    logger.debug(f"ExecutionMonitor record_order_event error: {e}")

            # 失败/超时检测：触发企业级自愈（RecoveryHandler）
            if status in ("failed", "error", "rejected", "timeout"):
                logger.warning(f"Order execution status '{status}' for {symbol} {strategy}, triggering recovery")
                if hasattr(self, 'recovery_handler') and self.recovery_handler:
                    try:
                        failure_type = "order_failed" if status != "timeout" else "performance_degradation"
                        await self.recovery_handler.handle_failure(failure_type, {
                            "symbol": symbol,
                            "strategy": strategy,
                            "order_data": order,
                            "error": f"Execution status: {status}",
                        })
                    except Exception as e:
                        logger.error(f"RecoveryHandler error in execution_monitoring: {e}")

            context.set_stage_end(PipelineStage.EXECUTION_MONITORING)
            return True
        except Exception as e:
            logger.error(f"Execution monitoring error: {e}")
            context.set_stage_end(PipelineStage.EXECUTION_MONITORING)
            return True
    
    async def _handle_settlement(self, context):
        context.set_stage_start(PipelineStage.SETTLEMENT)
        # P2 修复：TradeJournal 不存在 record_trade 方法，且真实记账已在
        # order_executor.execute_order 的成交回调中完成（record_fill）。
        # 此处仅落结算结果到上下文，避免调用不存在方法导致 AttributeError。
        if context.execution_result:
            context.settlement_result = context.execution_result
        context.set_stage_end(PipelineStage.SETTLEMENT)
        return True

    # ---- 策略管理器回调 ----

    async def _on_strategy_error(self, name: str, error_msg: str):
        """策略错误回调：记录日志并触发通知"""
        mgr_cfg = self.config.get("strategy_manager", {})
        if mgr_cfg.get("auto_restart_on_error", True):
            instance = self.strategy_manager.get_instance(name)
            restart_attempts = getattr(instance, '_restart_attempts', 0) if instance else 0
            max_attempts = mgr_cfg.get("max_restart_attempts", 3)
            if restart_attempts < max_attempts:
                logger.warning(f"Strategy '{name}' error, auto-restarting ({restart_attempts + 1}/{max_attempts})")
                if instance:
                    instance._restart_attempts = restart_attempts + 1
                await self.strategy_manager.restart_strategy(name)
            else:
                logger.error(f"Strategy '{name}' exceeded max restart attempts, leaving in ERROR state")
        try:
            await self.notification_manager.broadcast(
                f"Strategy '{name}' ERROR: {error_msg[:200]}",
                "Strategy Error", "CRITICAL"
            )
        except Exception:
            pass

    async def _on_strategy_health_change(self, name: str, data: Any = None):
        """策略健康状态变化回调"""
        health = self.strategy_manager._health_statuses.get(name)
        if health and health == StrategyHealth.CRITICAL:
            logger.warning(f"Strategy '{name}' health CRITICAL: {data}")
            try:
                await self.notification_manager.broadcast(
                    f"Strategy '{name}' health CRITICAL: {data}",
                    "Strategy Health Alert", "WARNING"
                )
            except Exception:
                pass

    async def _strategy_manager_health_loop(self):
        """策略管理器健康检查循环"""
        mgr_cfg = self.config.get("strategy_manager", {})
        interval = mgr_cfg.get("health_check_interval_seconds", 60)
        heartbeat_timeout = mgr_cfg.get("heartbeat_timeout_seconds", 120)
        logger.info(f"StrategyManager health loop started (interval={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)
                if not self.strategy_manager:
                    continue

                # 全量健康检查
                health_report = await self.strategy_manager.check_all_health()
                summary = health_report.get("summary", {})
                if summary.get("critical", 0) > 0:
                    logger.warning(
                        f"StrategyManager health: {summary.get('healthy', 0)}H/"
                        f"{summary.get('warning', 0)}W/{summary.get('critical', 0)}C"
                    )

                # 心跳超时检测
                now = datetime.now()
                for name in self.strategy_manager.get_enabled_strategy_names():
                    state = self.strategy_manager.get_lifecycle_state(name)
                    if state != StrategyLifecycle.RUNNING:
                        continue
                    instance = self.strategy_manager.get_instance(name)
                    if not instance:
                        continue
                    # 检查最后信号时间
                    last_signal = getattr(instance, '_last_signal_time', None)
                    if last_signal:
                        if isinstance(last_signal, str):
                            last_signal = datetime.fromisoformat(last_signal.replace('Z', '+00:00'))
                        elif isinstance(last_signal, datetime):
                            pass  # already a datetime
                        elif isinstance(last_signal, dict):
                            # 网格等策略按 symbol 存储多时间戳，取最近的时间
                            recent_values = [v for v in last_signal.values() if isinstance(v, (int, float))]
                            if recent_values:
                                last_signal = datetime.fromtimestamp(max(recent_values))
                            else:
                                logger.debug(f"Strategy '{name}' _last_signal_time dict is empty, skipping heartbeat check")
                                continue
                        else:
                            logger.debug(f"Strategy '{name}' _last_signal_time is unexpected type: {type(last_signal)}, skipping heartbeat check")
                            continue
                        elapsed = (now - last_signal.replace(tzinfo=None)).total_seconds()
                        if elapsed > heartbeat_timeout:
                            logger.warning(f"Strategy '{name}' heartbeat timeout: {elapsed:.0f}s since last signal")

                # 持久化状态
                self.strategy_manager.persist_state()

                # P2-12: 处理到期的参数变更验证，自动回滚退化配置
                try:
                    metrics = self._collect_strategy_metrics()
                    rollbacks = await self.strategy_manager.process_pending_parameter_validations(
                        current_metrics_by_strategy=metrics
                    )
                    if rollbacks:
                        logger.info(f"StrategyManager parameter validation processed: {list(rollbacks.keys())}")
                except Exception as e:
                    logger.error(f"Parameter validation processing error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"StrategyManager health loop error: {e}")
                await asyncio.sleep(interval * 2)

    def _collect_strategy_metrics(self) -> Dict[str, Dict[str, Any]]:
        """采集各策略当前 PnL/胜率指标，供参数变更验证回滚判断使用。"""
        metrics: Dict[str, Dict[str, Any]] = {}
        if not self.strategy_manager:
            return metrics
        try:
            for name in self.strategy_manager.get_enabled_strategy_names():
                stats = self.trade_journal.get_strategy_stats(name)
                metrics[name] = {
                    "trades": stats.get("total_trades", 0),
                    "pnl": stats.get("total_pnl", 0.0),
                    "win_rate": stats.get("win_rate", 0.0),
                }
        except Exception as e:
            logger.debug(f"Collect strategy metrics for validation error: {e}")
        return metrics

    async def _decision_engine_loop(self):
        """
        智能决策引擎循环

        职责：
          1. 每60秒更新自适应决策阈值（基于市场状态）
          2. 每300秒持久化审计链到磁盘
          3. 监控决策延迟并记录性能指标
          4. 检测决策异常并触发告警
        """
        interval = self._get_loop_interval("decision_engine", 60)
        persist_counter = 0
        logger.info(f"DecisionEngine loop started (interval={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)
                ide = self.intelligent_decision_engine
                if not ide:
                    continue

                # ── 1. 自适应阈值更新 ──
                try:
                    market_regime = "unknown"
                    volatility_pct = 0.5
                    if hasattr(self, 'market_regime_engine'):
                        regime_output = self.market_regime_engine.get_regime()
                        if regime_output and isinstance(regime_output, dict):
                            market_regime = regime_output.get("regime", "unknown")
                            volatility_pct = regime_output.get("volatility_percentile", 0.5)

                    # 计算近期胜率
                    recent_win_rate = 0.5
                    if ide._decision_history:
                        last_50 = list(ide._decision_history)[-50:]
                        wins = sum(1 for d in last_50 if d.get("success", False))
                        recent_win_rate = wins / max(len(last_50), 1)

                    new_threshold = ide.adapt_threshold(
                        market_regime=market_regime,
                        recent_win_rate=recent_win_rate,
                        volatility_percentile=volatility_pct,
                        equity_drawdown=self._get_current_drawdown(),
                    )
                    logger.debug(f"Decision threshold adapted: {new_threshold:.2%} "
                                f"(regime={market_regime}, wr={recent_win_rate:.1%})")
                except Exception as e:
                    logger.debug(f"Decision threshold adaptation error: {e}")

                # ── 2. 审计链持久化（每5次循环=5分钟）──
                persist_counter += 1
                if persist_counter >= 5:
                    try:
                        ide.persist_audit_chain()
                        persist_counter = 0
                    except Exception as e:
                        logger.debug(f"Audit chain persist error: {e}")

                # ── 3. 延迟监控 ──
                try:
                    avg_latency = ide.get_average_latency()
                    p95_latency = ide.get_latency_percentile(95)
                    if p95_latency > 1000:
                        logger.warning(
                            f"Decision latency high: avg={avg_latency:.0f}ms, p95={p95_latency:.0f}ms"
                        )
                except Exception:
                    pass

                # ── 4. 决策质量趋势告警 ──
                try:
                    if ide._consecutive_losses >= 5:
                        logger.warning(
                            f"Consecutive decision losses: {ide._consecutive_losses} — "
                            f"consider reducing exposure or switching to EMERGENCY_ONLY mode"
                        )
                except Exception:
                    pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DecisionEngine loop error: {e}")
                await asyncio.sleep(30)

        logger.info("DecisionEngine loop stopped")

    async def _decision_coordinator_loop(self):
        """
        决策协调器处理循环

        职责：
          1. 周期性调用 process_decisions() 处理队列中的决策
          2. 自动过期清理、优先级升级、依赖拓扑排序执行
          3. 记录协调器统计指标
          4. 检测异常（如审批率过低、冲突率过高）并发出告警
        """
        interval = self._get_loop_interval("decision_coordinator", 1)
        logger.info(f"DecisionCoordinator loop started (interval={interval}s)")
        alert_cooldown = 0  # 告警冷却计数器

        while True:
            try:
                await asyncio.sleep(interval)
                dc = self.decision_coordinator
                if not dc:
                    continue

                # ── 1. 处理决策队列（含过期清理、优先级升级、拓扑排序）──
                try:
                    processed = await dc.process_decisions()
                    if processed:
                        approved = sum(1 for d in processed if d.status.value == "approved")
                        rejected = sum(1 for d in processed if d.status.value == "rejected")
                        expired = sum(1 for d in processed if d.status.value == "expired")
                        logger.debug(
                            f"DecisionCoordinator processed: {len(processed)} decisions "
                            f"(approved={approved}, rejected={rejected}, expired={expired})"
                        )
                except Exception as e:
                    logger.error(f"DecisionCoordinator process error: {e}")

                # ── 2. 统计指标检查 ──
                try:
                    stats = dc.get_stats()
                    total = stats.get("decisions", {}).get("total", 0)
                    approved = stats.get("decisions", {}).get("approved", 0)
                    conflict_rate = stats.get("conflict_rate", 0)

                    # 审批率异常告警（每60次循环=60秒冷却一次）
                    if total > 50 and alert_cooldown <= 0:
                        approval_rate = approved / max(total, 1)
                        if approval_rate < 0.1:
                            logger.warning(
                                f"DecisionCoordinator approval rate critically low: "
                                f"{approval_rate:.1%} ({approved}/{total})"
                            )
                            alert_cooldown = 60

                        if conflict_rate > 0.3:
                            logger.warning(
                                f"DecisionCoordinator conflict rate high: "
                                f"{conflict_rate:.1%}"
                            )
                            alert_cooldown = 60

                    if alert_cooldown > 0:
                        alert_cooldown -= 1

                except Exception as e:
                    logger.debug(f"DecisionCoordinator stats error: {e}")

                # ── 3. 批处理状态更新 ──
                try:
                    for batch_id, batch in list(dc._active_batches.items()):
                        if batch.status.value in ("completed", "rolled_back"):
                            dc._active_batches.pop(batch_id, None)
                except Exception:
                    pass

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"DecisionCoordinator loop error: {e}")
                await asyncio.sleep(5)

        logger.info("DecisionCoordinator loop stopped")

    async def _rule_engine_maintenance_loop(self):
        """
        规则引擎维护循环

        职责：
          1. 周期性冲突检测并报告
          2. 清理过期条件缓存
          3. 收集规则性能统计数据
          4. 检测规则效果退化（过高误报率）并发出告警
          5. 标记低效规则建议优化
        """
        rule_cfg = self.config.get("rule_engine", {})
        interval = rule_cfg.get("maintenance_interval_seconds", 300)
        logger.info(f"RuleEngine maintenance loop started (interval={interval}s)")
        alert_cooldown = 0

        while True:
            try:
                await asyncio.sleep(interval)
                re = self.rule_engine
                if not re:
                    continue

                # ── 1. 冲突检测 ──
                try:
                    conflicts = re.detect_conflicts()
                    if conflicts:
                        conflict_report = re.get_conflict_report()
                        logger.warning(
                            f"RuleEngine: {conflict_report['conflict_count']} rule conflicts detected"
                        )
                        for c in conflicts[:5]:
                            logger.debug(f"  Conflict: {c[0]} ↔ {c[1]} ({c[2]})")
                except Exception as e:
                    logger.debug(f"RuleEngine conflict detection error: {e}")

                # ── 2. 缓存统计与清理 ──
                try:
                    cache_stats = re.get_cache_stats()
                    if cache_stats.get("entries", 0) > 1000:
                        re.invalidate_cache()
                        logger.info("RuleEngine: condition cache invalidated (size > 1000)")
                except Exception as e:
                    logger.debug(f"RuleEngine cache stats error: {e}")

                # ── 3. 规则性能分析 ──
                try:
                    stats = re.get_rule_stats()
                    low_precision_rules = []
                    for rid, s in stats.items():
                        # 高误报率告警：匹配次数>50且精度<30%
                        if s["matches"] > 50 and s["precision"] < 0.3:
                            low_precision_rules.append((rid, s["precision"], s["matches"]))
                        # 高延迟告警：评估延迟>5000us
                        if s["avg_eval_latency_us"] > 5000:
                            logger.debug(
                                f"RuleEngine: rule '{rid}' high eval latency: "
                                f"{s['avg_eval_latency_us']:.0f}us"
                            )

                    if low_precision_rules and alert_cooldown <= 0:
                        logger.warning(
                            f"RuleEngine: {len(low_precision_rules)} rules with low precision "
                            f"(<30%), consider review"
                        )
                        alert_cooldown = 3  # 3个周期冷却
                except Exception as e:
                    logger.debug(f"RuleEngine stats error: {e}")

                # ── 4. 规则组统计 ──
                try:
                    group_summary = re.get_group_summary()
                    engine_stats = re.get_engine_stats()
                    logger.debug(
                        f"RuleEngine: {engine_stats['total_rules']} rules, "
                        f"{engine_stats['enabled_rules']} enabled, "
                        f"{engine_stats['total_matches']} matches, "
                        f"groups={dict(group_summary)}"
                    )
                except Exception as e:
                    logger.debug(f"RuleEngine group summary error: {e}")

                if alert_cooldown > 0:
                    alert_cooldown -= 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RuleEngine maintenance loop error: {e}")
                await asyncio.sleep(30)

        logger.info("RuleEngine maintenance loop stopped")

    async def _ml_decision_maintenance_loop(self):
        """
        ML决策引擎维护循环

        职责：
          1. 周期性检查是否需要增量重训练（基于漂移检测/时间间隔）
          2. 特征漂移检测（PSI计算 + 漂移级别判定）
          3. 预测质量退化监控
          4. A/B模型对比与自动回滚
          5. 生成模型健康报告日志
        """
        ml_cfg = self.config.get("ml_decision", {})
        interval = ml_cfg.get("maintenance_interval_seconds", 300)
        logger.info(f"MLDecisionEngine maintenance loop started (interval={interval}s)")
        drift_alert_cooldown = 0

        while True:
            try:
                await asyncio.sleep(interval)
                ml = self.ml_decision_engine
                if not ml or not ml._enabled:
                    continue

                # ── 1. 增量重训练检查 ──
                try:
                    retrain_result = ml.check_and_retrain()
                    if retrain_result:
                        logger.info(
                            f"MLDecisionEngine: auto-retrained → v{retrain_result.version}, "
                            f"status={retrain_result.status.value}"
                        )
                except Exception as e:
                    logger.debug(f"MLDecisionEngine retrain check error: {e}")

                # ── 2. 漂移检测 & 健康报告 ──
                try:
                    online_status = ml.online_trainer.get_drift_status()
                    drift_level = online_status.get("drift_level", "none")

                    if drift_level != "none" and drift_alert_cooldown <= 0:
                        logger.warning(
                            f"MLDecisionEngine: drift detected — "
                            f"level={drift_level}, "
                            f"mean_err={online_status.get('mean_error', 0):.4f}, "
                            f"buffer={online_status.get('buffer_size', 0)}"
                        )
                        drift_alert_cooldown = 2
                except Exception as e:
                    logger.debug(f"MLDecisionEngine drift check error: {e}")

                # ── 3. 预测质量退化检查 ──
                try:
                    degraded, degradation = ml.health_monitor.check_quality_degradation()
                    if degraded:
                        logger.warning(
                            f"MLDecisionEngine: prediction quality degraded "
                            f"({degradation:.1%}), consider retrain"
                        )
                except Exception as e:
                    logger.debug(f"MLDecisionEngine quality check error: {e}")

                # ── 4. A/B模型对比 & 自动回滚 ──
                try:
                    rolled_back = ml.model_registry.compare_and_rollback(ml._model_id)
                    if rolled_back:
                        logger.warning(
                            f"MLDecisionEngine: auto-rolled back to version {rolled_back}"
                        )
                except Exception as e:
                    logger.debug(f"MLDecisionEngine rollback check error: {e}")

                # ── 5. 模型版本摘要日志 ──
                try:
                    active_v = ml.model_registry.get_active_model(ml._model_id)
                    version_history = ml.model_registry.get_version_history(ml._model_id)
                    prediction_count = ml._prediction_counter
                    n_versions = len(version_history)
                    logger.debug(
                        f"MLDecisionEngine health: active=v{active_v}, "
                        f"versions={n_versions}, predictions={prediction_count}, "
                        f"fitted={ml.model_ensemble.is_fitted()}"
                    )
                except Exception as e:
                    logger.debug(f"MLDecisionEngine summary error: {e}")

                if drift_alert_cooldown > 0:
                    drift_alert_cooldown -= 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"MLDecisionEngine maintenance loop error: {e}")
                await asyncio.sleep(30)

        logger.info("MLDecisionEngine maintenance loop stopped")

    async def _rl_agent_maintenance_loop(self):
        """
        RL智能体维护循环

        职责：
          1. 周期性执行训练步（从经验回放中采样学习）
          2. Epsilon衰减监控（含时间衰减：无交易时也逐步降低探索率）
          3. 定期持久化模型
          4. 探索率与性能统计日志
        """
        rl_cfg = self.config.get("rl_agent", {})
        interval = rl_cfg.get("maintenance_interval_seconds", 120)
        logger.info(f"TradingRLAgent maintenance loop started (interval={interval}s)")
        persist_counter = 0
        _last_epsilon_decay = time.time()

        while True:
            try:
                await asyncio.sleep(interval)
                rl = self.rl_agent
                if not rl or not rl.is_enabled():
                    continue

                # 实盘在线训练显式 opt-in；默认只维护和观测模型，不让权重随运行漂移。
                if rl._online_training_enabled and not rl._training_frozen:
                    replay_size = len(rl._replay_buffer) if hasattr(rl._replay_buffer, '__len__') else 0
                    if replay_size >= rl._batch_size:
                        for _ in range(min(5, max(1, replay_size // rl._batch_size))):
                            loss = rl.train_step()
                            if loss is not None:
                                break  # 完成一批次训练
                    else:
                        # 无经验数据时进行时间衰减：避免冷启动时 epsilon=1.0 永久不变
                        now = time.time()
                        if now - _last_epsilon_decay >= 600:  # 每10分钟衰减一次
                            rl._epsilon = max(rl._epsilon_min, rl._epsilon * 0.98)
                            _last_epsilon_decay = now

                # ── 2. 状态日志 ──
                stats = rl.get_stats()
                logger.debug(
                    f"TradingRLAgent: steps={stats['total_steps']}, "
                    f"episodes={stats['episodes_completed']}, "
                    f"epsilon={stats['epsilon']:.4f}, "
                    f"avg_reward={stats['avg_reward']:.4f}, "
                    f"replay={stats['replay_size']}"
                )

                # ── 3. 定期持久化 ──
                persist_counter += 1
                if persist_counter >= 5:  # 每5个周期保存一次
                    rl._save()
                    persist_counter = 0

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"TradingRLAgent maintenance loop error: {e}")
                await asyncio.sleep(30)

        logger.info("TradingRLAgent maintenance loop stopped")

    async def _audit_chain_persist_loop(self):
        """
        智能决策引擎审计链持久化循环

        职责：
          1. 周期性将内存中的审计链持久化到磁盘
          2. 清理超过上限的旧审计条目
          3. 确保数据不丢失
        """
        interval = self.config.get("decision", {}).get("audit_persist_interval_seconds", 300)
        logger.info(f"Audit chain persistence loop started (interval={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)
                if not self.intelligent_decision_engine:
                    continue

                # 持久化审计链
                self.intelligent_decision_engine.persist_audit_chain()
                logger.debug("Audit chain persisted to disk")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audit chain persistence loop error: {e}")
                await asyncio.sleep(30)

        logger.info("Audit chain persistence loop stopped")

    async def _risk_budget_refresh_loop(self):
        """
        风险预算周期性刷新循环

        职责：
          1. 每N分钟重新计算风险预算方案
          2. 更新策略收益率和已消耗风险数据
          3. 自动触发阈值突破告警
          4. 持久化快照状态
        """
        rb_cfg = self.config.get("risk_budget", {})
        realloc_cfg = rb_cfg.get("reallocation", {})
        interval = realloc_cfg.get("interval_minutes", 5) * 60  # 默认5分钟
        logger.info(f"RiskBudget refresh loop started (interval={interval}s)")

        while True:
            try:
                await asyncio.sleep(interval)

                rbe = self.risk_budget_engine
                if not rbe or not rbe._enabled:
                    continue

                # 获取总权益
                total_equity = 5000.0
                try:
                    if hasattr(self, 'account_manager') and self.account_manager:
                        total_equity = self.account_manager.get_total_equity()
                    elif rbe._account_manager:
                        total_equity = rbe._account_manager.get_total_equity()
                except Exception:
                    total_equity = float(
                        self.config.get("trading", {}).get("total_capital", 5000.0)
                    )

                # 获取活跃策略名称
                strategy_names = []
                if self.strategy_manager:
                    strategy_names = self.strategy_manager.get_enabled_strategy_names()
                elif hasattr(self, 'strategy_engine'):
                    try:
                        container = self.strategy_engine._strategy_container
                        strategy_names = [
                            inst.strategy_type.value if hasattr(inst.strategy_type, 'value')
                            else str(inst.strategy_type)
                            for inst in container.get_all_instances()
                        ]
                    except Exception:
                        strategy_names = ["scalping", "trend", "grid", "arbitrage"]

                if not strategy_names:
                    strategy_names = list(rbe._strategy_budget_pcts.keys())

                # 收集策略指标
                strategy_metrics: Dict[str, Dict[str, float]] = {}
                current_risk_consumed: Dict[str, float] = {}
                try:
                    agent = self.allocation_agent if hasattr(self, 'allocation_agent') else None
                    for name in strategy_names:
                        m = {}
                        if agent and hasattr(agent, '_get_strategy_metrics'):
                            m = agent._get_strategy_metrics(name)
                            m["consecutive_losses"] = agent._get_consecutive_count(name, "loss") if hasattr(agent, '_get_consecutive_count') else 0
                        strategy_metrics[name] = m
                        # 估算已消耗风险（当前持仓的VaR估计）
                        mtd = m or {}
                        current_risk_consumed[name] = mtd.get("position_value", 0) * 0.02  # 简化

                    # 更新收益率序列
                    for name in strategy_names:
                        if agent and hasattr(agent, '_get_strategy_returns'):
                            returns = agent._get_strategy_returns(name)
                            if returns:
                                rbe.update_strategy_returns(name, returns)
                except Exception as e:
                    logger.debug(f"Failed to collect strategy metrics: {e}")

                # 获取市场状态
                market_regime = "unknown"
                try:
                    if hasattr(self, 'market_regime_engine'):
                        regime_output = self.market_regime_engine.get_regime()
                        if regime_output and isinstance(regime_output, dict):
                            market_regime = regime_output.get("regime", "unknown")
                except Exception:
                    pass

                # 计算风险预算方案
                plan = await rbe.compute_risk_budget_plan(
                    total_equity=total_equity,
                    strategy_names=strategy_names,
                    strategy_metrics=strategy_metrics,
                    current_risk_consumed=current_risk_consumed,
                    market_regime=market_regime,
                )

                # ── 阈值突破告警 ──
                await self._check_risk_budget_alerts(plan)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"RiskBudget refresh loop error: {e}", exc_info=True)
                await asyncio.sleep(30)

        logger.info("RiskBudget refresh loop stopped")

    async def _check_risk_budget_alerts(self, plan):
        """
        检查风险预算阈值并触发告警

        告警条件：
          - 总体利用率 > 90% → CRITICAL
          - 总体利用率 > 75% → WARNING
          - 任何策略预算耗尽 → CRITICAL
          - VaR回测失败（Kupiec p < 0.01）→ WARNING
          - 币种集中度超限 → WARNING
          - 预算趋势预测将超过90% → WARNING
        """
        try:
            # 总体利用率告警
            if plan.overall_utilization >= 0.90:
                msg = (f"[RISK CRITICAL] 总体风险预算利用率 {plan.overall_utilization:.1%}，"
                       f"剩余预算 {plan.total_risk_budget * (1 - plan.overall_utilization):.0f} USDT")
                logger.warning(msg)
                if hasattr(self, 'notification_manager'):
                    await self.notification_manager.broadcast(msg, "Risk Budget Alert", "CRITICAL")
            elif plan.overall_utilization >= 0.75:
                logger.warning(f"[RISK WARNING] 总体风险预算利用率 {plan.overall_utilization:.1%}")

            # 策略耗尽告警
            for name in plan.exhausted_strategies:
                msg = f"[RISK CRITICAL] 策略 '{name}' 风险预算已耗尽，停止新开仓"
                logger.warning(msg)
                if hasattr(self, 'notification_manager'):
                    await self.notification_manager.broadcast(msg, "Strategy Risk Exhausted", "WARNING")

            # 币种集中度告警
            if hasattr(plan, '_symbol_concentration'):
                for sym, srb in plan._symbol_concentration.items():
                    if srb.warning_level == "critical":
                        msg = (f"[RISK CRITICAL] 币种集中度超限: {sym} VaR95={srb.total_var_95:.0f}USDT, "
                               f"涉及策略: {srb.contributing_strategies}")
                        logger.warning(msg)

            # VaR回测告警
            if hasattr(plan, '_var_backtest') and plan._var_backtest.status == "fail":
                msg = f"[RISK WARNING] VaR模型回测失败 (Kupiec p={plan._var_backtest.kupiec_pvalue_95:.4f})，风险估计可能不准确"
                logger.warning(msg)

            # 预算趋势告警
            if hasattr(plan, '_budget_trend'):
                trend = plan._budget_trend.get("trend", "")
                if trend == "rapidly_increasing":
                    msg = f"[RISK WARNING] 风险预算消耗趋势急剧上升，预测将很快超过90%"
                    logger.warning(msg)

        except Exception as e:
            logger.error(f"Risk budget alert check error: {e}")

    async def shutdown(self):
        logger.info("Initiating graceful shutdown...")

        # 取消所有注册的后台任务（防止无限循环阻止关闭）
        await self._cancel_all_tasks()

        if self.position_manager is not None:
            try:
                await self.position_manager.stop()
                logger.info("PositionManager stopped")
            except Exception as e:
                logger.error(f"Error stopping PositionManager: {e}")

        # P0: 持久化智能决策审计链
        if self.intelligent_decision_engine:
            try:
                self.intelligent_decision_engine.persist_audit_chain()
                logger.info("IntelligentDecisionEngine audit chain persisted")
            except Exception as e:
                logger.warning(f"Failed to persist audit chain: {e}")

        # P0: 持久化ML决策引擎模型注册表
        if self.ml_decision_engine:
            try:
                self.ml_decision_engine.model_registry.persist(self.ml_decision_engine._model_id)
                logger.info("MLDecisionEngine model registry persisted")
            except Exception as e:
                logger.warning(f"Failed to persist ML model registry: {e}")

        # P0: 推荐策略管理器停止所有策略（按逆序优雅停止）
        if self.strategy_manager:
            try:
                stop_results = await self.strategy_manager.stop_all()
                stopped = sum(1 for v in stop_results.values() if v)
                logger.info(f"StrategyManager: {stopped}/{len(stop_results)} strategies stopped")
                self.strategy_manager.persist_state()
            except Exception as e:
                logger.error(f"Error stopping StrategyManager: {e}")

        # P0: 停止条件单管理器（保存状态、取消后台循环）
        try:
            await self.conditional_order_manager.stop()
            logger.info("ConditionalOrderManager stopped")
        except Exception as e:
            logger.error(f"Error stopping ConditionalOrderManager: {e}")
        
        # P0: 停止市场状态引擎（避免后台协程在清理阶段继续访问OKX客户端）
        try:
            if hasattr(self, 'market_regime_engine') and self.market_regime_engine:
                await self.market_regime_engine.stop()
                logger.info("MarketRegimeEngine stopped")
        except Exception as e:
            logger.error(f"Error stopping MarketRegimeEngine: {e}")
        
        # P0: 停止相关性风控监控循环
        try:
            if hasattr(self, 'correlation_risk') and self.correlation_risk:
                self.correlation_risk._is_running = False
                logger.info("CorrelationRiskControl stopped")
        except Exception as e:
            logger.error(f"Error stopping CorrelationRiskControl: {e}")

        # P0: 停止黑天鹅保护监控
        try:
            if hasattr(self, 'black_swan_protection') and self.black_swan_protection:
                await self.black_swan_protection.stop()
                logger.info("BlackSwanProtection stopped")
        except Exception as e:
            logger.error(f"Error stopping BlackSwanProtection: {e}")
        
        # P0: 停止自适应控制器
        try:
            if hasattr(self, 'adaptive_controller') and self.adaptive_controller:
                if hasattr(self.adaptive_controller, 'stop'):
                    await self.adaptive_controller.stop()
                logger.info("AdaptiveController stopped")
        except Exception as e:
            logger.error(f"Error stopping AdaptiveController: {e}")

        # P0: 停止资金变动监控器（持久化状态）
        try:
            if hasattr(self, 'equity_monitor') and self.equity_monitor:
                await self.equity_monitor.stop()
                logger.info("EquityMonitor stopped")
        except Exception as e:
            logger.error(f"Error stopping EquityMonitor: {e}")
        
        # P0: 停止自适应学习系统各组件
        try:
            await self.meta_learner.stop()
            logger.info("MetaLearner stopped")
            await self.performance_feedback.stop()
            logger.info("PerformanceFeedback stopped")
            await self.market_regime_detector.stop()
            logger.info("MarketRegimeDetector stopped")
            await self.knowledge_base.stop()
            logger.info("KnowledgeBase stopped")
            await self.strategy_evolver.stop()
            logger.info("StrategyEvolver stopped")
            await self.parameter_adaptor.stop()
            logger.info("ParameterAdaptor stopped")
            await self.online_learner.stop()
            logger.info("OnlineLearner stopped")
        except Exception as e:
            logger.error(f"Error stopping adaptive learning components: {e}")

        # P0: 停止执行监控器和订单生命周期管理器
        try:
            await self.execution_monitor.stop()
            logger.info("ExecutionMonitor stopped")
            await self.order_state_synchronizer.stop()
            logger.info("OrderStateSynchronizer stopped")
            await self.order_lifecycle.stop()
            logger.info("OrderLifecycleManager stopped")
            # ── 重启挂单丢失修复：优雅关闭订单巡检与持久化存储 ──
            if getattr(self, "order_patrol", None) is not None:
                await self.order_patrol.stop()
                logger.info("OrderPatrolService stopped")
            if getattr(self, "order_store", None) is not None:
                self.order_store.close()
                logger.info("OrderStore closed")
            await self.pipeline_orchestrator.stop()
            logger.info("PipelineOrchestrator stopped")
            if getattr(self, 'metrics_pipeline', None) is not None:
                await self.metrics_pipeline.stop()
                logger.info("MetricsPipeline stopped")
            await self.anomaly_detector.stop()
            logger.info("AnomalyDetector stopped")
            await self.recovery_handler.stop()
            logger.info("RecoveryHandler stopped")
        except Exception as e:
            logger.error(f"Error stopping pipeline/monitor components: {e}")

        # P0: 关闭策略引擎（停止回测、清理策略实例）
        try:
            logger.info("Step 0.5: Shutting down StrategyEngine and CapitalManager...")
            self.strategy_engine.shutdown()
            await self.capital_manager.stop_periodic_tasks()
            await self.risk_monitor.stop()
            logger.info("StrategyEngine and CapitalManager shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down StrategyEngine/CapitalManager: {e}")
        
        # P0: 关闭风控拦截器监控
        try:
            logger.info("Step 0.6: RiskGate position monitor will be cancelled by event loop shutdown")
        except Exception as e:
            logger.error(f"Error in risk gate shutdown: {e}")
        
        try:
            logger.info("Step 0: Stopping P2 monitoring components...")
            await self.apm_monitor.stop_monitoring()
            await self.state_manager.stop_persistence()
            await self.unified_layer.shutdown()
            self.state_manager.save_state()
            logger.info("P2 monitoring components stopped")
        except Exception as e:
            logger.error(f"Error stopping P2 components: {e}")
        
        try:
            logger.info("Step 1: Canceling all pending orders...")
            await self.order_queue.clear_queue()
            self.conditional_order_manager.cancel_all_conditional_orders()
            logger.info("Pending orders canceled")
        except Exception as e:
            logger.error(f"Error canceling orders: {e}")
        
        try:
            logger.info("Step 2: Closing all positions...")
            positions = self.okx_client.get_positions()
            closed_count = 0
            for pos_data in positions:
                position = self.okx_client._parse_position(pos_data)
                if position and float(position.quantity) > 0:
                    side = "sell" if position.side == "long" else "buy"
                    self.okx_client.place_order(
                        symbol=position.symbol,
                        side=side,
                        order_type="market",
                        quantity=self.okx_client.contracts_to_coins(position.symbol, abs(float(position.quantity))),
                        leverage=position.leverage
                    )
                    closed_count += 1
            logger.info(f"Closed {closed_count} positions")
        except Exception as e:
            logger.error(f"Error closing positions: {e}")
        
        try:
            logger.info("Step 3: Closing WebSocket connections...")
            await self.okx_client.close_websocket()
            logger.info("WebSocket connections closed")
        except Exception as e:
            logger.error(f"Error closing WebSocket: {e}")
        
        try:
            logger.info("Step 4: Flushing SQLite storage...")
            self.sqlite_storage._Session().commit()
            logger.info("SQLite flushed")
        except Exception as e:
            logger.error(f"Error flushing SQLite: {e}")
        
        try:
            logger.info("Step 5: Generating final report...")
            summary = self.performance_monitor.get_trade_summary()
            await self.alert_manager.send_daily_summary(summary)
            logger.info("Final report generated")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
        
        try:
            logger.info("Step 6: Saving learning state...")
            await self.optimizer._save_learning_state()
            logger.info("Learning state saved")
        except Exception as e:
            logger.error(f"Error saving learning state: {e}")
        
        try:
            logger.info("Step 7: Generating analysis report...")
            analysis_report = self.analyzer.generate_analysis_report()
            await self.alert_manager.send_alert("SHUTDOWN_ANALYSIS", analysis_report)
            logger.info("Analysis report generated")
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
        
        logger.info("Graceful shutdown completed")