"""
交易信号处理器，负责信号的优先级调度、幂等去重及风控与执行流程编排。
"""
import asyncio
from datetime import datetime
from typing import Dict, Any, Callable, Optional, List, Set, Tuple
from loguru import logger
import uuid
import time

from decision.intelligent_decision_engine import DecisionUrgency, MetaDecisionVerdict
from core.unified_layer import Event, EventType
from core.event_id import EventIDGenerator
from utils.helpers import safe_float, safe_finite, safe_div


class SignalProcessor:
    def __init__(self, config: Dict[str, Any], global_risk, strategy_risk, 
                 order_executor, alert_manager, trade_journal, 
                 adaptive_controller, profit_optimizer, account_manager):
        self.config = config
        self._global_risk = global_risk
        self._strategy_risk = strategy_risk
        self._order_executor = order_executor
        self._alert_manager = alert_manager
        self._trade_journal = trade_journal
        self._adaptive_controller = adaptive_controller
        self._profit_optimizer = profit_optimizer
        self._account_manager = account_manager
        
        self._regime_engine = None
        self._quality_engine = None
        self._coordinator = None
        self._normalizer = None
        
        self._decision_coordinator = None
        self._rule_engine = None
        self._ensemble_maker = None
        self._decision_validator = None
        self._decision_evaluator = None
        self._confidence_calibrator = None
        
        self._pipeline_orchestrator = None
        self._signal_pipeline = None
        self._anomaly_detector = None
        self._recovery_handler = None
        self._execution_monitor = None

        # 智能决策核心引擎
        self._intelligent_decision_engine = None

        # 机器学习决策引擎（实时置信度修正），由 scheduler 注入，未注入时不启用
        self._ml_decision_engine = None

        # 五层风控拦截器
        self._risk_gate = None

        # P2: 独立风控裁决器（traceID 贯穿 + 裁决事件溯源），由 scheduler 注入
        self._risk_adjudicator = None

        # 相关性风控（开仓前检查同向高相关集中度），由 scheduler 注入
        self._correlation_risk = None

        # 事件总线（开仓全链路拒单溯源：发布 SIGNAL_REJECTED），由 scheduler 注入
        self._event_bus = None

        # L0 市场状态总门控（RegimeGate）：由 set_regime_gate 注入，默认 None 表示不启用
        self._regime_gate = None

        # 感知侧闭环协调器（Regime识别门控 + 信号质量 统一感知决策）：由 set_perception_loop 注入
        self._perception_loop = None

        # 统一自适应仓位提供者（callable(signal_dict)->quantity|None），由 scheduler 注入
        self._position_sizing_provider = None

        # 资金自适应分配引擎（专一分配硬门控），由 scheduler 注入
        self._capital_allocator = None

        # 组合再平衡器（总敞口硬限制），由 scheduler 注入
        self._portfolio_rebalancer = None
        # AGI bear-case projection gate; return a rejection reason for blocked strategies.
        self._bear_case_open_gate: Optional[Callable[[str], Optional[str]]] = None

        self._signal_callback = None
        self._last_signal_time: Dict[str, datetime] = {}
        self._signal_cooldown = 30
        # 信号幂等去重：基于signal_id防止重复下单
        self._processed_signal_ids: set = set()
        self._max_signal_id_cache = 10000  # 防止内存无限增长
        self._signal_id_order: List[str] = []  # 有序队列辅助FIFO清理
        
        # P2: 重新平衡策略优先级 - 防止高频策略(如scalping)过度阻塞网格策略
        # 网格策略从1提升到2，剥头皮从3降低到2，使各策略在同一竞争层级
        self._priority_map = {
            "arbitrage": 4,
            "scalping": 2,   # P2: 3→2，防止过度阻塞grid
            "trend": 2,
            "grid": 2,       # P2: 1→2，提升到与trend/scalping同级
            "spot_grid": 2,
            "spot_martingale": 2,
        }
        
        # P2: 防饥饿计数器 - 记录每个策略被连续阻塞的次数
        self._consecutive_blocks: Dict[str, int] = {}
        self._max_consecutive_blocks = 5  # 连续阻塞5次后强制放行
        
        self._cooldown_map = {
            "scalping": 10,
            "grid": 20,
            "spot_grid": 20,
            "spot_martingale": 15,
            "arbitrage": 60,
            "trend": 30,
        }
        
        self._signal_history: Dict[str, Any] = {}
        self._signal_queue: List[Dict[str, Any]] = []
        self._max_queue_size = 100
        self._signal_aggregation_window = 5
        self._pending_signals: Dict[str, List[Dict[str, Any]]] = {}
        self._processing_lock = asyncio.Lock()

        # 死信队列：保存被拒绝的信号，支持后续排查和重放
        self._dead_letters: List[Dict[str, Any]] = []
        self._max_dead_letters = 100
        
        self._high_value_symbols = {"BTC-USDT-SWAP", "BTC-USDT"}
        # 资金分级 BTC 阈值显式化：从 trading.high_value_equity_threshold 读取（默认 2000.0）
        self._high_value_equity_threshold = float(
            self.config.get("trading", {}).get("high_value_equity_threshold", 2000.0) or 2000.0)

        # 低胜率币种黑名单：历史统计胜率<20%或持续亏损的币种，禁止开仓（平仓不受限）
        # 黑名单已清零 - 6U资金场景下仅4个币种，需全部放开
        self._blacklist_symbols = set()

        # L0 RegimeGate 门控统计（拒绝/异常按 regime/策略分类，供 Dashboard 查询门控效果）
        self._regime_gate_stats: Dict[str, Any] = {
            "rejected": 0,
            "errors": 0,
            "by_regime": {},
            "by_strategy": {},
        }

        # P3: 端到端管线延迟追踪
        from core.apm_monitor import get_pipeline_latency_tracker
        self._pipeline_latency_tracker = get_pipeline_latency_tracker(config)

    def set_regime_engine(self, engine):
        """注入MarketRegimeEngine"""
        self._regime_engine = engine

    def set_quality_engine(self, engine):
        """注入SignalQualityEngine"""
        self._quality_engine = engine

    def set_coordinator(self, coordinator):
        """注入StrategyCoordinator"""
        self._coordinator = coordinator

    def set_regime_gate(self, gate):
        """注入L0市场状态总门控（RegimeGate）"""
        self._regime_gate = gate

    def set_perception_loop(self, loop):
        """注入感知侧闭环协调器（SignalPerceptionCoordinator）。

        注入后，_process_signal 会用 loop.perceive() 统一执行「Regime识别门控 + 信号质量」
        感知决策，替代分散的 RegimeGate 硬门控与 _evaluate_signal_quality 两处调用。
        未注入时保持原有分散逻辑不变。
        """
        self._perception_loop = loop
        logger.info("SignalPerceptionCoordinator injected into SignalProcessor (perception loop)")

    def _record_regime_gate_rejection(self, strategy_name: str, gate_result):
        """统计 RegimeGate 拒绝（按 regime/策略分类），供 Dashboard 观察门控效果。"""
        try:
            stats = self._regime_gate_stats
            stats["rejected"] = int(stats.get("rejected", 0)) + 1
            regime = getattr(gate_result, "regime", "unknown") or "unknown"
            stats["by_regime"][regime] = int(stats["by_regime"].get(regime, 0)) + 1
            stats["by_strategy"][strategy_name] = int(stats["by_strategy"].get(strategy_name, 0)) + 1
        except Exception as e:
            logger.debug(f"RegimeGate rejection stats error: {e}")

    def _record_regime_gate_error(self):
        """统计 RegimeGate 评估异常（fail-open 次数，用于发现引擎故障）。"""
        try:
            self._regime_gate_stats["errors"] = int(self._regime_gate_stats.get("errors", 0)) + 1
        except Exception:
            pass

    def get_regime_gate_stats(self) -> Dict[str, Any]:
        """返回 L0 RegimeGate 门控统计（拒绝数/异常数/按 regime/策略分布）。"""
        stats = dict(self._regime_gate_stats)
        stats["by_regime"] = dict(stats.get("by_regime", {}))
        stats["by_strategy"] = dict(stats.get("by_strategy", {}))
        return stats

    def get_perception_stats(self) -> Dict[str, Any]:
        """返回感知侧闭环统计（Regime门控 + 信号质量 关联分布），供 Dashboard 展示。"""
        if self._perception_loop is None:
            return {"enabled": False}
        try:
            stats = self._perception_loop.get_stats()
            stats["enabled"] = True
            return stats
        except Exception as e:
            logger.debug(f"get_perception_stats error: {e}")
            return {"enabled": True, "error": str(e)}

    def set_normalizer(self, normalizer):
        """注入SignalNormalizer"""
        self._normalizer = normalizer

    def set_decision_components(self, decision_coordinator, rule_engine, ensemble_maker, 
                                decision_validator, decision_evaluator, confidence_calibrator):
        """注入智能决策系统组件"""
        self._decision_coordinator = decision_coordinator
        self._rule_engine = rule_engine
        self._ensemble_maker = ensemble_maker
        self._decision_validator = decision_validator
        self._decision_evaluator = decision_evaluator
        self._confidence_calibrator = confidence_calibrator
        logger.info("Smart decision system components injected into SignalProcessor")

    def set_pipeline_components(self, pipeline_orchestrator, signal_pipeline, anomaly_detector, recovery_handler):
        """注入交易流水线系统组件"""
        self._pipeline_orchestrator = pipeline_orchestrator
        self._signal_pipeline = signal_pipeline
        self._anomaly_detector = anomaly_detector
        self._recovery_handler = recovery_handler
        logger.info("Trading pipeline components injected into SignalProcessor")

    def set_execution_monitor(self, execution_monitor):
        """注入执行监控器（端到端延迟/SLA 追踪）"""
        self._execution_monitor = execution_monitor
        logger.info("ExecutionMonitor injected into SignalProcessor")

    def set_risk_gate(self, risk_gate):
        """注入五层风控拦截器"""
        self._risk_gate = risk_gate
        logger.info("RiskGate (5-layer serial risk control) injected into SignalProcessor")

    def set_risk_adjudicator(self, adjudicator):
        """注入独立风控裁决器（traceID 贯穿 + 裁决事件溯源）。"""
        self._risk_adjudicator = adjudicator
        logger.info("RiskAdjudicator injected into SignalProcessor")

    def set_correlation_risk(self, correlation_risk):
        """注入相关性风控（开仓前检查同向高相关集中度）。"""
        self._correlation_risk = correlation_risk
        logger.info("CorrelationRiskControl injected into SignalProcessor")

    def set_event_bus(self, event_bus):
        """注入事件总线（开仓全链路拒单溯源：信号层丢弃时发布 SIGNAL_REJECTED）。"""
        self._event_bus = event_bus

    def set_position_sizing_provider(self, provider):
        """注入统一自适应仓位提供者（callable(signal_dict) -> quantity | None）。

        由 scheduler 组装并注入，用于把分散在各策略的仓位计算收敛为
        AdaptivePositionSizer 单一口径。provider 返回 None 或抛异常时
        fail-open 保持原 quantity。
        """
        self._position_sizing_provider = provider
        logger.info("Unified position sizing provider injected into SignalProcessor")

    def set_capital_allocator(self, allocator):
        """注入资金自适应分配引擎（专一分配硬门控）。"""
        self._capital_allocator = allocator
        logger.info("CapitalAdaptiveAllocator injected into SignalProcessor")

    def set_portfolio_rebalancer(self, rebalancer):
        """注入组合再平衡器（总敞口硬限制门控）。"""
        self._portfolio_rebalancer = rebalancer
        logger.info("PortfolioRebalancer injected into SignalProcessor for exposure limit check")

    def set_bear_case_open_gate(self, gate: Optional[Callable[[str], Optional[str]]]) -> None:
        """Inject a strategy-level gate that rejects new opens under bear-case limits."""
        self._bear_case_open_gate = gate

    def set_intelligent_decision_engine(self, engine):
        """注入智能决策核心引擎"""
        self._intelligent_decision_engine = engine
        logger.info("IntelligentDecisionEngine injected into SignalProcessor")

    def set_ml_decision_engine(self, engine):
        """注入机器学习决策引擎（实时决策置信度修正，fail-open）。"""
        self._ml_decision_engine = engine
        logger.info("MLDecisionEngine injected into SignalProcessor")

    def setup_signal_routing(self, strategies: list, redis_cache=None):
        async def direct_signal_handler(signal_data):
            # 兼容 Signal dataclass、dict、JSON字符串 三种格式
            if hasattr(signal_data, '__dataclass_fields__'):
                signal_dict = {
                    "symbol": getattr(signal_data, 'symbol', ''),
                    "strategy_name": getattr(signal_data, 'strategy_name', ''),
                    "signal_type": getattr(signal_data, 'signal_type', ''),
                    "direction": getattr(signal_data, 'direction', ''),
                    "price": getattr(signal_data, 'price', 0.0),
                    "quantity": getattr(signal_data, 'quantity', 0.0),
                    "leverage": getattr(signal_data, 'leverage', 1),
                    "stop_loss": getattr(signal_data, 'stop_loss', None),
                    "take_profit": getattr(signal_data, 'take_profit', None),
                    "confidence": getattr(signal_data, 'confidence', 0.5),
                    "timestamp": getattr(signal_data, 'timestamp', datetime.now()),
                }
                return (await self._process_signal(signal_dict)) is True
            elif isinstance(signal_data, str):
                # JSON字符串：解析后处理
                try:
                    import json as _json
                    parsed = _json.loads(signal_data)
                    if isinstance(parsed, dict):
                        # 兼容 {"type":"signal","data":{...}} 包装格式
                        if parsed.get("type") == "signal" and "data" in parsed:
                            return (await self._process_signal(parsed["data"])) is True
                        else:
                            return (await self._process_signal(parsed)) is True
                    else:
                        logger.warning(f"Signal handler received unparseable string: {signal_data[:100]}")
                        return False
                except Exception as e:
                    logger.warning(f"Signal handler JSON parse failed: {e}")
                    return False
            elif isinstance(signal_data, dict):
                # 兼容 {"type":"signal","data":{...}} 包装格式
                if signal_data.get("type") == "signal" and "data" in signal_data:
                    return (await self._process_signal(signal_data["data"])) is True
                else:
                    return (await self._process_signal(signal_data)) is True
            else:
                logger.warning(f"Signal handler received unsupported type: {type(signal_data)}")
                return False
        
        self._signal_callback = direct_signal_handler
        
        for strategy in strategies:
            if hasattr(strategy, 'set_signal_callback'):
                strategy.set_signal_callback(direct_signal_handler)
        
        if redis_cache:
            redis_cache.set_signal_callback(direct_signal_handler)
        
        logger.info("Direct signal routing setup completed")
    
    def _check_high_value_symbol(self, symbol: str, signal_type: str = "") -> bool:
        if symbol not in self._high_value_symbols:
            return True
        signal_type_lower = signal_type.lower() if signal_type else ""
        is_close_signal = ("close" in signal_type_lower or 
                          "take_profit" in signal_type_lower or 
                          "stop_loss" in signal_type_lower or 
                          "liquidation" in signal_type_lower or
                          "exit" in signal_type_lower)
        if is_close_signal:
            return True
        try:
            current_equity = 0.0
            if hasattr(self._global_risk, '_current_equity') and self._global_risk._current_equity > 0:
                current_equity = self._global_risk._current_equity
            elif self._account_manager and hasattr(self._account_manager, '_current_total_leverage'):
                try:
                    account_info = self._account_manager.redis_cache.get_account_info()
                    if account_info:
                        current_equity = account_info.total_equity
                except Exception:
                    pass
            if current_equity <= 0:
                return True
            if current_equity < self._high_value_equity_threshold:
                return False
            return True
        except Exception as e:
            logger.debug(f"High value symbol check error: {e}")
            return True

    def _check_blacklist_symbol(self, symbol: str, signal_type: str = "") -> bool:
        """检查币种是否在黑名单中。黑名单币种禁止开仓，但允许平仓/止损/止盈信号通过。"""
        if symbol not in self._blacklist_symbols:
            return True
        signal_type_lower = signal_type.lower() if signal_type else ""
        is_close_signal = ("close" in signal_type_lower or
                          "take_profit" in signal_type_lower or
                          "stop_loss" in signal_type_lower or
                          "liquidation" in signal_type_lower or
                          "exit" in signal_type_lower)
        if is_close_signal:
            return True
        return False

    def _push_dead_letter(self, signal_dict: Dict[str, Any], reject_reason: str,
                          layer: Optional[str] = None, reason_code: Optional[str] = None):
        """将拒绝的信号推入死信队列，并发布 SIGNAL_REJECTED 事件到 EventStore（可溯源）。

        纯增量改造：在原有内存 dead letter 逻辑之外，补齐 traceID 贯穿 + 拒单事件落盘，
        不改变任何 return 路径与判定行为。持久化失败静默降级，不阻断主流程。
        """
        # 1) traceID 贯穿：复用 signal 既有 trace_id，缺失则生成并回写
        try:
            trace_id = signal_dict.get("trace_id")
            if not trace_id:
                trace_id = EventIDGenerator.get_instance().generate()
                signal_dict["trace_id"] = trace_id
        except Exception:
            trace_id = None

        # 2) 保留原有死信内存队列逻辑
        try:
            dead_entry = {
                "signal_id": signal_dict.get("signal_id", str(uuid.uuid4())),
                "original_signal": dict(signal_dict),
                "reject_reason": reject_reason,
                "trace_id": trace_id,
                "rejected_at": datetime.now().isoformat(),
            }
            self._dead_letters.append(dead_entry)
            if len(self._dead_letters) > self._max_dead_letters:
                self._dead_letters = self._dead_letters[-self._max_dead_letters:]
        except Exception as e:
            logger.debug(f"Failed to push dead letter: {e}")

        # 3) 发布 SIGNAL_REJECTED 事件（含 layer/reason_code 归一化）
        if layer is None or reason_code is None:
            _layer, _reason_code = self._classify_reject_reason(reject_reason)
            layer = layer or _layer
            reason_code = reason_code or _reason_code
        try:
            from core.signal_flow_stats import record_signal_flow_event
            record_signal_flow_event(
                "signal_rejected",
                strategy=str(signal_dict.get("strategy_name", signal_dict.get("strategy", "")) or ""),
                layer=layer or "signal_processor",
                reason=reason_code or reject_reason,
            )
        except Exception as e:
            logger.debug(f"Signal flow rejection metric failed: {e}")

        if self._event_bus is not None:
            try:
                signal_type = str(signal_dict.get("signal_type", "") or "")
                is_close = ("close" in signal_type.lower()
                            or "liquidat" in signal_type.lower()
                            or "exit" in signal_type.lower()
                            or "stop" in signal_type.lower())
                self._event_bus.publish_sync(Event(EventType.SIGNAL_REJECTED, {
                    "trace_id": trace_id,
                    "symbol": signal_dict.get("symbol", ""),
                    "strategy": signal_dict.get("strategy_name", signal_dict.get("strategy", "")),
                    "signal_type": signal_type,
                    "direction": signal_dict.get("direction", ""),
                    "layer": layer or "signal_processor",
                    "reason_code": reason_code or reject_reason,
                    "reason": reject_reason,
                    "is_close": is_close,
                }))
            except Exception as e:
                logger.debug(f"Failed to publish SIGNAL_REJECTED event: {e}")

    @staticmethod
    def _classify_reject_reason(reject_reason: str):
        """将拒单原因字符串归一化为 (layer, reason_code) 二元组，枚举值从前缀映射推导。"""
        reason = reject_reason or ""
        # 前缀包含匹配（顺序敏感），未命中回退 signal_processor
        mapping = [
            ("regime_gate:", "regime_gate"),
            ("signal_quality", "signal_quality"),
            ("strategy_priority", "strategy_priority"),
            ("signal_conflict", "signal_conflict"),
            ("conflict_resolution", "signal_conflict"),
            ("collaborative_trigger", "coordinator"),
            ("capital_focus", "capital_allocation"),
            ("decision_validation", "decision_validation"),
            ("meta_", "meta_decision"),
            ("cba_rejected", "cost_benefit"),
            ("below_threshold", "threshold"),
            ("risk_gate_exception", "risk_gate"),
            ("risk_gate_blocked", "risk_gate"),
            ("trading_paused", "risk_gate"),
            ("strategy_risk_validation", "risk_gate"),
            ("frozen", "risk_gate"),
        ]
        for prefix, layer in mapping:
            if reason.startswith(prefix) or prefix in reason:
                return layer, reason
        return "signal_processor", reason

    def _publish_decision_event(self, signal_dict: Dict[str, Any], event_type: EventType,
                                reason: str = ""):
        """发布决策层正向事件（DECISION_VALIDATED / DECISION_APPROVED / DECISION_REJECTED）。

        企业级决策溯源：与既有 SIGNAL_REJECTED（负向）互补，补齐「信号→决策→下单」的
        正向决策链路（验证通过 / 最终批准），使决策层裁决可被 EventStore 重放审计。
        traceID 贯穿：复用 signal 既有 trace_id，缺失则生成并回写。持久化失败静默降级。
        """
        if self._event_bus is None:
            return
        try:
            trace_id = signal_dict.get("trace_id")
            if not trace_id:
                trace_id = EventIDGenerator.get_instance().generate()
                signal_dict["trace_id"] = trace_id
            self._event_bus.publish_sync(Event(event_type, {
                "trace_id": trace_id,
                "symbol": signal_dict.get("symbol", ""),
                "strategy": signal_dict.get("strategy_name", signal_dict.get("strategy", "")),
                "signal_type": str(signal_dict.get("signal_type", "") or ""),
                "direction": signal_dict.get("direction", ""),
                "confidence": signal_dict.get("confidence", 0.0),
                "reason": reason,
            }))
        except Exception as e:
            logger.debug(f"Failed to publish {event_type.value} event: {e}")

    @staticmethod
    def _collect_condition_fields(condition) -> Set[str]:
        """递归收集规则条件树中引用的所有字段名（用于 fail-open 守卫）。"""
        fields: Set[str] = set()
        if not isinstance(condition, dict):
            return fields
        if "field" in condition:
            fields.add(condition["field"])
        for key in ("and", "or", "not"):
            node = condition.get(key)
            if isinstance(node, list):
                for sub in node:
                    fields |= SignalProcessor._collect_condition_fields(sub)
            elif isinstance(node, dict):
                fields |= SignalProcessor._collect_condition_fields(node)
        return fields

    def _run_rule_gate(self, signal_dict: Dict[str, Any], confidence: float) -> str:
        """规则引擎门控：评估规则，REJECT 命中则返回拒绝原因，否则返回空串。

        fail-open 守卫：仅当规则条件引用的字段全部存在（非 None）时才采纳 REJECT，
        避免字段缺失被规则引擎的 `None→0` 默认值误判（如 risk_min_margin 的
        margin_after<50 会因缺字段把 0<50 误判为真而拒掉所有信号）。
        """
        rule_engine = getattr(self, "_rule_engine", None)
        if rule_engine is None:
            return ""
        try:
            enriched = dict(signal_dict)
            enriched.setdefault("confidence", confidence)
            enriched.setdefault("leverage", signal_dict.get("leverage", 5))
            enriched.setdefault("direction", self._normalize_direction(signal_dict.get("direction", "")))
            enriched.setdefault("quantity", signal_dict.get("quantity", 0.0))
            # 注入真实持仓数（若可获取），使 position_max_count 规则生效
            try:
                if getattr(self, "_risk_gate", None) is not None:
                    cnt = self._risk_gate.get_active_position_count()
                    if cnt is not None:
                        enriched["current_position_count"] = int(cnt)
            except Exception:
                pass

            for r in rule_engine.evaluate(enriched):
                if r.get("action") != "reject":
                    continue
                rule = rule_engine.get_rule(r.get("rule_id", ""))
                if rule is None:
                    continue
                fields = self._collect_condition_fields(getattr(rule, "condition", {}))
                if not fields:
                    continue
                if any(rule._get_field_value(enriched, f) is None for f in fields):
                    logger.debug(f"Rule {rule.rule_id} skipped (field missing, fail-open)")
                    continue
                return f"rule_{rule.rule_id}"
        except Exception as e:
            logger.debug(f"Rule engine gate fail-open: {e}")
        return ""

    def _decision_ensemble_gate(self, signal_dict: Dict[str, Any], confidence: float) -> Tuple[bool, float, str]:
        """决策集成仲裁：规则引擎门控 + ML 决策置信度修正（均 fail-open）。

        企业级强化：把此前仅初始化+维护循环、未在 _process_signal 实时决策路径
        被调用的 RuleBasedEngine / MLDecisionEngine 接入实时决策，形成
        「规则门控 → ML 置信度修正 → 集成事件溯源」的增量裁决层。

        返回 (pass, adjusted_confidence, reason)；任何引擎未就绪/异常均放行。
        """
        symbol = signal_dict.get("symbol", "")
        direction = self._normalize_direction(signal_dict.get("direction", ""))

        # ── 1. 规则引擎门控 ──
        reject_reason = self._run_rule_gate(signal_dict, confidence)
        if reject_reason:
            self._publish_decision_event(signal_dict, EventType.DECISION_REJECTED, reject_reason)
            return False, confidence, reject_reason

        # ── 2. ML 决策置信度修正（方向一致性 soft 修正）──
        ml_direction = ""
        ml_confidence = 0.0
        if self._ml_decision_engine is not None:
            try:
                pred = self._ml_decision_engine.predict(signal_dict, symbol=symbol)
                ml_direction = str(getattr(pred, "direction", "hold") or "hold")
                ml_confidence = float(getattr(pred, "confidence", 0.0) or 0.0)
                if ml_direction not in ("", "hold") and direction in ("long", "short"):
                    agree = (ml_direction == "buy" and direction == "long") or \
                            (ml_direction == "sell" and direction == "short")
                    confidence = max(0.0, min(1.0, confidence * (1.05 if agree else 0.95)))
            except Exception as e:
                logger.debug(f"ML decision engine fail-open: {e}")

        # ── 3. 发布集成仲裁事件（正向溯源，携 ML 方向/置信度）──
        self._publish_decision_event(
            signal_dict, EventType.DECISION_ENSEMBLE,
            f"ml_dir={ml_direction or 'n/a'},ml_conf={ml_confidence:.3f}",
        )
        return True, confidence, ""

    def get_dead_letters(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近N条死信"""
        return list(self._dead_letters[-limit:])

    def replay_dead_letter(self, signal_id: str) -> bool:
        """重放指定signal_id的死信，将其重新提交到_process_signal"""
        for i, entry in enumerate(self._dead_letters):
            if entry.get("signal_id") == signal_id:
                signal = entry.get("original_signal", {})
                logger.info(f"Replaying dead letter: {signal_id} (reason={entry.get('reject_reason')})")
                asyncio.create_task(self._process_signal(signal))
                self._dead_letters.pop(i)
                return True
        logger.warning(f"Dead letter not found for replay: {signal_id}")
        return False

    async def start_signal_listener(self, redis_cache):
        self._redis_cache = redis_cache  # P0: 保存引用用于重连
        
        async def listen():
            pubsub = redis_cache.subscribe_signals()
            if pubsub is None:
                logger.warning("Signal listener skipped: Redis not available")
                return
            
            reconnect_count = 0
            _MAX_RECONNECT_ATTEMPTS = 50
            while True:
                try:
                    # P0-3: 使用 to_thread 避免阻塞事件循环
                    message = await asyncio.to_thread(pubsub.get_message, timeout=1)
                    if message and message["type"] == "message":
                        signal_data = None
                        try:
                            data = message["data"]
                            if isinstance(data, bytes):
                                data = data.decode()

                            signal_data = __import__('json').loads(data)

                            if signal_data.get("type") == "signal":
                                await self._process_signal(signal_data["data"])
                        except Exception as e:
                            import traceback
                            sd_keys = list(signal_data.keys()) if isinstance(signal_data, dict) else type(signal_data).__name__
                            logger.error(f"Error processing signal: {e}")
                            logger.error(f"Signal data type: {sd_keys}")
                            logger.error(f"Traceback:\n{traceback.format_exc()}")
                            if self._alert_manager:
                                try:
                                    sym = ""
                                    if isinstance(signal_data, dict):
                                        sym = signal_data.get("symbol", "")
                                    await self._alert_manager.send_alert(
                                        "signal_process_error",
                                        f"Signal processing exception: {e}",
                                        severity="WARNING",
                                        symbol=sym,
                                    )
                                except Exception:
                                    pass
                except (ConnectionError, OSError) as e:
                    # P0: Redis断线时自动重连，避免信号监听静默失效
                    reconnect_count += 1
                    if reconnect_count > _MAX_RECONNECT_ATTEMPTS:
                        logger.critical(
                            f"Redis signal listener exhausted {_MAX_RECONNECT_ATTEMPTS} reconnect attempts, "
                            f"exiting listener. Manual restart required."
                        )
                        if self._alert_manager:
                            try:
                                await self._alert_manager.send_alert(
                                    "redis_listener_dead",
                                    f"Redis signal listener gave up after {_MAX_RECONNECT_ATTEMPTS} attempts",
                                    severity="CRITICAL",
                                )
                            except Exception:
                                pass
                        return
                    backoff = min(30, 2 ** min(reconnect_count, 6))
                    logger.warning(f"Redis connection lost in signal listener, reconnecting in {backoff}s (attempt {reconnect_count}/{_MAX_RECONNECT_ATTEMPTS})")
                    await asyncio.sleep(backoff)
                    # P0: 关闭旧的pubsub连接，避免Redis服务端残留订阅耗尽连接数
                    if pubsub is not None:
                        try:
                            pubsub.close()
                        except Exception:
                            pass
                    pubsub = redis_cache.subscribe_signals()
                    if pubsub is None:
                        logger.error("Redis reconnection failed, retrying...")
                        await asyncio.sleep(5)
                        continue
                    reconnect_count = 0
                # pubsub.get_message(timeout=1) 已经阻塞等待了1秒，无需额外sleep
                if not message:
                    await asyncio.sleep(0)  # yield事件循环，避免CPU空转
        
        asyncio.create_task(listen())

    async def _process_signal(self, signal_data: Dict[str, Any]):
        _t0 = time.perf_counter()  # 端到端延迟追踪起点：信号到达
        _pipe_ctx = self._pipeline_latency_tracker.start_cycle(
            symbol=str(signal_data.get("symbol", "")),
            strategy=str(signal_data.get("strategy_name", "")),
        )
        from core.signal_flow_stats import record_signal_flow_event
        raw_strategy = str(
            signal_data.get("strategy_name", signal_data.get("strategy", "")) or ""
        )
        record_signal_flow_event("received", strategy=raw_strategy)
        _pipe_ctx.begin_stage("validation")
        if self._normalizer:
            normalized = self._normalizer.normalize(signal_data)
            is_valid, errors, warnings = self._normalizer.validate(normalized)
            if not is_valid:
                logger.warning(f"Signal validation failed: {errors}")
                self._push_dead_letter(signal_data, "signal_validation_failed")
                return
            if warnings:
                logger.info(f"Signal warnings: {warnings}")
            signal_dict = self._normalizer.to_dict(normalized)
        else:
            signal_dict = signal_data

        _pipe_ctx.end_stage("validation")

        symbol = signal_dict.get("symbol", "")
        strategy_name = signal_dict.get("strategy_name", "")
        direction = signal_dict.get("direction", "")
        confidence = safe_float(signal_dict.get("confidence"), 0.5)
        
        if not symbol or not strategy_name:
            self._push_dead_letter(signal_dict, "missing_symbol_or_strategy")
            return

        # traceID 贯穿：复用上游 trace_id，缺失则生成并回写（决策层正向溯源 + 全链路追踪）。
        # 前置生成确保后续 RiskAdjudicator / 决策层事件 / 拒单事件复用同一 trace_id，
        # 消除正向路径（未拒单）下裁决器另起新 trace_id 导致的链路断裂。
        if not signal_dict.get("trace_id"):
            try:
                signal_dict["trace_id"] = EventIDGenerator.get_instance().generate()
            except Exception as e:
                logger.debug(f"trace_id generation failed (fail-open): {e}")

        # 接入企业级异常检测（AnomalyDetector）：信号频率/价格/成交量异常追踪
        _pipe_ctx.begin_stage("anomaly_detection")
        if self._anomaly_detector is not None:
            try:
                tasks = [self._anomaly_detector.detect("signal", {
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "direction": direction,
                    "signal_type": signal_dict.get("signal_type", ""),
                })]
                if signal_dict.get("price"):
                    tasks.append(self._anomaly_detector.detect("price", {
                        "symbol": symbol,
                        "price": signal_dict.get("price"),
                    }))
                if signal_dict.get("volume"):
                    tasks.append(self._anomaly_detector.detect("volume", {
                        "symbol": symbol,
                        "volume": signal_dict.get("volume"),
                    }))
                await asyncio.gather(*tasks)
            except Exception as e:
                logger.debug(f"AnomalyDetector detection error: {e}")
        _pipe_ctx.end_stage("anomaly_detection")

        # 信号幂等去重：基于signal_id防止Redis重连/网络重试导致的重复下单
        _pipe_ctx.begin_stage("dedup_check")
        signal_id = signal_dict.get("signal_id", "")
        if not signal_id:
            # 自动生成signal_id（symbol+strategy+direction+timestamp窗口）
            time_bucket = int(datetime.now().timestamp() // 5)  # 5秒时间桶
            signal_id = f"{symbol}:{strategy_name}:{direction}:{time_bucket}"
            signal_dict["signal_id"] = signal_id

        if signal_id in self._processed_signal_ids:
            logger.info(f"Signal deduplicated (signal_id={signal_id}): {symbol} {strategy_name} {direction}")
            self._push_dead_letter(signal_dict, "duplicate_signal")
            return

        if not self._check_high_value_symbol(symbol, signal_dict.get("signal_type", "")):
            logger.info(f"Signal rejected: {symbol} is high-value symbol, current equity below threshold")
            self._push_dead_letter(signal_dict, "high_value_symbol_low_equity")
            return

        if not self._check_blacklist_symbol(symbol, signal_dict.get("signal_type", "")):
            logger.info(f"Signal rejected: {symbol} is in blacklist (low win-rate symbol)")
            self._push_dead_letter(signal_dict, "blacklist_symbol")
            return
        _pipe_ctx.end_stage("dedup_check")

        # 感知侧闭环：Regime识别门控 + 信号质量 统一感知决策（注入时替代下方分散逻辑）
        _pipe_ctx.begin_stage("quality_assessment")
        perception_result = None
        if self._perception_loop is not None:
            try:
                perception_result = self._perception_loop.perceive(signal_dict)
                if perception_result.decision == "reject_gate":
                    self._record_regime_gate_rejection(strategy_name, perception_result)
                    logger.info(
                        f"Perception RegimeGate rejected: {symbol} {strategy_name} "
                        f"[{perception_result.regime}] {perception_result.reason}"
                    )
                    self._push_dead_letter(
                        signal_dict,
                        f"regime_gate:{perception_result.reason}",
                        layer="regime_gate",
                        reason_code=perception_result.reason,
                    )
                    return
                if perception_result.decision == "reject_quality":
                    logger.info(
                        f"Perception quality rejected: {strategy_name} {symbol} {direction} "
                        f"(score={perception_result.quality_score:.2f})"
                    )
                    self._push_dead_letter(signal_dict, "signal_quality_rejected")
                    return
            except Exception as e:
                # 感知闭环异常：fail-open 回退到原有分散逻辑（记录告警，避免误杀全部信号）
                self._record_regime_gate_error()
                logger.warning(f"SignalPerceptionCoordinator perceive error (fail-open): {e}")
                if self._alert_manager:
                    try:
                        await self._alert_manager.send_alert(
                            "regime_gate_degraded",
                            f"Perception loop error (fail-open): {e}",
                            severity="WARNING",
                            symbol=signal_dict.get("symbol", ""),
                        )
                    except Exception:
                        pass
                perception_result = None

        # P1: L0 市场状态总门控 - regime 不匹配的开仓信号直接丢弃（不进入审计）
        # 感知闭环已注入且成功执行时跳过此处（门控已在 perceive 内完成），避免重复评估。
        if perception_result is None and self._regime_gate is not None:
            try:
                gate_result = self._regime_gate.evaluate(
                    symbol,
                    strategy_name,
                    signal_type=signal_dict.get("signal_type", ""),
                    direction=direction,
                    confidence=confidence,
                )
                if not gate_result.allowed:
                    self._record_regime_gate_rejection(strategy_name, gate_result)
                    logger.info(
                        f"P1 RegimeGate rejected: {symbol} {strategy_name} "
                        f"[{gate_result.regime}] {gate_result.reason}"
                    )
                    self._push_dead_letter(
                        signal_dict,
                        f"regime_gate:{gate_result.reason}",
                        layer="regime_gate",
                        reason_code=gate_result.reason,
                    )
                    return
            except Exception as e:
                # 门控异常：fail-open 放行但可追溯（引擎故障降级，避免误杀全部信号），
                # 与静默 debug 不同，这里记 warning + 异常计数，便于发现 regime 引擎故障。
                self._record_regime_gate_error()
                logger.warning(f"RegimeGate evaluate error (fail-open): {e}")
                if self._alert_manager:
                    try:
                        await self._alert_manager.send_alert(
                            "regime_gate_degraded",
                            f"RegimeGate evaluate error (fail-open): {e}",
                            severity="WARNING",
                            symbol=signal_dict.get("symbol", ""),
                        )
                    except Exception:
                        pass

        # 币种隔离：检查币种是否被冻结（单币种熔断，冻结的币种禁止开仓但允许平仓）
        sig_type_lower = str(signal_dict.get("signal_type", "")).lower()
        direction_lower = str(signal_dict.get("direction", "")).lower()
        is_close_sig = (
            any(kw in sig_type_lower for kw in [
                "close", "stop_loss", "take_profit", "reduce", "exit", "liquidation"
            ])
            or direction_lower in {"close", "reduce", "reduce_only"}
            or bool(signal_dict.get("reduce_only"))
        )
        if self._reject_bear_case_open_signal(signal_dict, is_close_sig):
            return

        if not is_close_sig and self._risk_gate is not None:
            try:
                if self._risk_gate.is_symbol_frozen(symbol):
                    logger.info(f"Signal rejected: {symbol} is frozen (per-symbol circuit breaker)")
                    self._push_dead_letter(signal_dict, "symbol_frozen")
                    return
            except Exception as e:
                logger.debug(f"Symbol freeze check error: {e}")
        
        # P5: 仓位容量预检查 - 开仓信号在满载时前置拦截，避免执行层拒绝
        if not is_close_sig and self._risk_gate is not None:
            try:
                current_positions = self._risk_gate.get_active_position_count()
                if current_positions is None:
                    # fail-closed：容量查询失败（未知），按保守满载拒绝开仓，避免满载时误判为未满载
                    logger.warning(
                        f"Signal rejected: position capacity unknown (query failed), "
                        f"skipping {symbol} {strategy_name} open signal"
                    )
                    self._push_dead_letter(signal_dict, "position_capacity_unknown")
                    return
                # P29: 小账户提升默认并发上限至12，与order_executor保持一致
                max_positions = self.config.get("trading", {}).get("max_concurrent_positions", 12)
                if current_positions >= max_positions:
                    logger.info(
                        f"Signal rejected: position capacity full "
                        f"({current_positions}/{max_positions}), "
                        f"skipping {symbol} {strategy_name} open signal"
                    )
                    self._push_dead_letter(signal_dict, "position_capacity_full")
                    return
            except Exception as e:
                logger.debug(f"Position capacity check error: {e}")

        # 相关性风控预检查：开仓前检查是否会超过同向高相关集中度上限
        if not is_close_sig and self._correlation_risk is not None:
            try:
                side = "long" if direction.lower() in ("long", "buy") else "short"
                price = safe_float(signal_dict.get("price"), 0.0)
                qty = safe_float(signal_dict.get("quantity"), 0.0)
                leverage = safe_float(signal_dict.get("leverage"), 1.0)
                estimated_margin = (price * qty / leverage) if leverage > 0 else 0.0
                if estimated_margin > 0:
                    allowed, reason = self._correlation_risk.can_open_position(
                        symbol, side, estimated_margin
                    )
                    if not allowed:
                        logger.info(f"Signal rejected: correlation risk - {reason}")
                        self._push_dead_letter(signal_dict, f"correlation_risk:{reason}")
                        return
            except Exception as e:
                logger.debug(f"Correlation risk check error (fail-open): {e}")

        signal_key = f"{symbol}:{strategy_name}:{direction}"
        now = datetime.now()
        
        cooldown = self._cooldown_map.get(strategy_name, 30)
        last_time = self._last_signal_time.get(signal_key)
        if last_time and (now - last_time).total_seconds() < cooldown:
            self._push_dead_letter(signal_dict, "cooldown")
            return
        
        # ── 统一自适应仓位计算（开仓信号，收敛仓位口径）──
        # 必须在 RiskGate 之前执行：让风控基于收敛后的仓位判断，
        # 否则 trend 等原始大 qty 信号会被「单币种仓位超限/可用保证金不足」误拦。
        # 仅开仓信号参与；平仓/减仓/止损止盈保持策略原 quantity 不变。
        # fail-open：provider 未注入/计算失败/返回 None 时保持原 quantity。
        if not is_close_sig and self._position_sizing_provider is not None:
            try:
                unified_qty = self._position_sizing_provider(signal_dict)
                if unified_qty is not None and safe_float(unified_qty, 0.0) > 0:
                    old_qty = safe_float(signal_dict.get("quantity"), 0.0)
                    signal_dict["quantity"] = safe_float(unified_qty, 0.0)
                    signal_dict["position_sizing"] = "unified"
                    logger.debug(
                        f"Unified position sizing: {symbol} {strategy_name} "
                        f"qty {old_qty:.6f} -> {safe_float(unified_qty, 0.0):.6f}"
                    )
            except Exception as e:
                logger.debug(f"Unified position sizing error (fail-open, keep original): {e}")
        _pipe_ctx.end_stage("quality_assessment")

        # ============ 五层风控串行校验（核心拦截层）============
        # L5紧急熔断→L4单日风控→L1事前风控→L2事中风控→L3持仓风控
        # 校验通过：信号下发执行器
        # 校验拦截：直接丢弃信号，记录拦截原因并推送风险告警，不发起任何下单请求
        _pipe_ctx.begin_stage("risk_check")
        if self._risk_adjudicator is not None or self._risk_gate is not None:
            try:
                # P2: 独立风控裁决器优先（traceID 贯穿回写 signal + 裁决事件溯源），未注入回退 RiskGate
                # P0-A: 风控校验为同步方法，使用 to_thread 避免阻塞事件循环
                # P0-B: 传递 is_close 使平仓信号跳过 L1 保证金/仓位上限检查（平仓释放保证金），
                #       与 order_executor 第二次调用保持一致，避免平仓被误拦
                if self._risk_adjudicator is not None:
                    risk_result = await asyncio.to_thread(
                        self._risk_adjudicator.adjudicate, signal_dict, None, is_close_sig
                    )
                else:
                    risk_result = await asyncio.to_thread(
                        self._risk_gate.validate, signal_dict, None, is_close_sig
                    )
                if not risk_result.passed:
                    # 拦截路径：丢弃信号 + 记录拦截原因 + 推送风险告警 + 不下单
                    await self._handle_risk_block(signal_dict, risk_result)
                    return
                # 通过路径：risk_result.passed == True，继续后续流程
            except Exception as e:
                logger.error(f"RiskGate validation error, falling back to legacy checks: {e}")
                if self._alert_manager:
                    try:
                        await self._alert_manager.send_alert(
                            "risk_gate_degraded",
                            f"RiskGate exception, degraded to legacy checks: {e}",
                            severity="WARNING",
                            symbol=signal_dict.get("symbol", ""),
                        )
                    except Exception:
                        pass
                # P0: RiskGate异常时必须走legacy降级，不能让信号绕过所有风控
                if not self._global_risk.can_trade():
                    logger.warning("Legacy check: trading is paused, signal rejected")
                    self._push_dead_letter(signal_dict, "risk_gate_exception_trading_paused")
                    return
                if not self._strategy_risk.validate_signal(signal_dict):
                    logger.warning("Legacy check: strategy risk validation failed")
                    self._push_dead_letter(signal_dict, "risk_gate_exception_strategy_risk")
                    return
        else:
            # RiskGate 未注入时，保留原有轻量级风控作为兜底
            if not self._global_risk.can_trade():
                logger.warning("Trading is paused, signal rejected")
                self._push_dead_letter(signal_dict, "trading_paused")
                return

            if not self._strategy_risk.validate_signal(signal_dict):
                logger.warning("Strategy risk validation failed")
                self._push_dead_letter(signal_dict, "strategy_risk_validation_failed")
                return

        if not await self._check_signal_conflict(signal_dict):
            self._push_dead_letter(signal_dict, "signal_conflict")
            return
        
        # 策略协同协调器：冲突检测
        if self._coordinator:
            conflict_pass, conflicts = self._coordinator.check_signal_conflicts(strategy_name, signal_dict)
            if not conflict_pass:
                logger.warning(f"Strategy conflict detected for {strategy_name} {symbol}: {[c['type'] for c in conflicts]}")
                # 尝试解决冲突
                resolved, adjusted = self._coordinator.resolve_conflict(strategy_name, signal_dict, conflicts)
                if not resolved:
                    logger.info(f"Signal rejected after conflict resolution: {strategy_name} {symbol}")
                    self._push_dead_letter(signal_dict, "conflict_resolution_failed")
                    return
                signal_dict = adjusted
            
            # 协同触发检查
            trigger_ok, trigger_reason = self._coordinator.check_collaborative_trigger(strategy_name, signal_dict)
            if not trigger_ok:
                logger.info(f"Collaborative trigger blocked: {strategy_name} {symbol} - {trigger_reason}")
                self._push_dead_letter(signal_dict, "collaborative_trigger_blocked")
                return
        
        if perception_result is not None:
            quality_pass = perception_result.quality_acceptable
            quality_breakdown = perception_result.quality_breakdown
        else:
            quality_pass, quality_breakdown = self._evaluate_signal_quality(signal_dict)
        
        if not quality_pass:
            logger.info(f"Signal quality rejected: {strategy_name} {symbol} {direction} "
                       f"(score={quality_breakdown.get('overall_score', 0):.2f})")
            self._push_dead_letter(signal_dict, "signal_quality_rejected")
            return
        
        if not self._check_strategy_priority(symbol, strategy_name, direction, confidence, quality_breakdown):
            self._push_dead_letter(signal_dict, "strategy_priority_blocked")
            return
        
        # ── 资金自适应分配引擎硬门控（专一分配：单一标的+单一策略+单持仓）──
        # 仅拦截开仓；平仓/减仓/止损止盈不受限。
        if not is_close_sig and self._capital_allocator is not None:
            try:
                open_count = int(self._account_manager.get_open_position_count() or 0)
            except Exception:
                open_count = 0
            allowed, reason = self._capital_allocator.evaluate_open(symbol, strategy_name, open_count)
            if not allowed:
                logger.info(f"Focused allocation blocked: {strategy_name} {symbol} ({reason})")
                self._push_dead_letter(signal_dict, f"capital_focus_{reason}")
                return

        # ── P1: 组合总敞口硬限制（开仓前检查）──
        if not is_close_sig and self._portfolio_rebalancer is not None:
            try:
                qty = abs(float(signal_dict.get("quantity", 0) or 0))
                price = abs(float(signal_dict.get("price", 0) or 0))
                additional_notional = qty * price if price > 0 else 0.0
                eq_result = self._portfolio_rebalancer.check_exposure_limit(additional_notional)
                if not eq_result.get("allowed", True):
                    logger.warning(
                        f"Exposure limit blocked: {strategy_name} {symbol} "
                        f"({eq_result.get('reason', '')})"
                    )
                    self._push_dead_letter(signal_dict, "exposure_limit_exceeded")
                    return
            except Exception as e:
                logger.warning(f"Exposure limit check failed (fail-open): {e}")
        
        self._last_signal_time[signal_key] = now
        
        self._record_signal(signal_dict, quality_breakdown)
        
        # 记录到策略协同协调器
        if self._coordinator:
            self._coordinator.record_strategy_signal(strategy_name, signal_dict)

        # 智能决策系统：决策验证
        if self._decision_validator:
            from decision.decision_coordinator import Decision, DecisionType
            decision = Decision(
                decision_id=str(uuid.uuid4()),
                decision_type=DecisionType.SIGNAL,
                data=signal_dict,
                confidence=confidence,
                source=strategy_name,
                trace_id=signal_dict.get("trace_id", ""),
            )
            validation_result, validation_errors = await self._decision_validator.validate(decision)
            
            if validation_result.value == "invalid":
                logger.warning(f"Decision validation failed for {strategy_name} {symbol}: {[e.code for e in validation_errors]}")
                self._publish_decision_event(signal_dict, EventType.DECISION_REJECTED, "decision_validation_invalid")
                self._push_dead_letter(signal_dict, "decision_validation_invalid")
                return
            elif validation_result.value == "warning":
                logger.info(f"Decision validation warnings for {strategy_name} {symbol}: {[e.code for e in validation_errors]}")
            # 决策验证通过 → 发布 DECISION_VALIDATED（决策层正向溯源）
            self._publish_decision_event(signal_dict, EventType.DECISION_VALIDATED, "validation_passed")

        # 智能决策系统：置信度校准
        if self._confidence_calibrator:
            calibrated_confidence, _ = self._confidence_calibrator.calibrate(confidence)
            if calibrated_confidence != confidence:
                logger.debug(f"Confidence calibrated: {confidence:.2f} -> {calibrated_confidence:.2f}")
                signal_dict["confidence"] = calibrated_confidence
                confidence = calibrated_confidence

        # ── P0: 智能决策引擎增强 ──
        _pipe_ctx.end_stage("risk_check")
        _pipe_ctx.begin_stage("decision_engine")
        if self._intelligent_decision_engine:
            ide = self._intelligent_decision_engine
            try:
                cba = None  # 初始化成本收益分析，供下方因果链记录使用
                # 1. 丰富决策上下文（订单簿分析、市场状态）
                order_book = signal_dict.get("order_book", None)
                regime_info = "unknown"
                if self._regime_engine:
                    try:
                        regime_result = self._regime_engine.get_regime()
                        if isinstance(regime_result, dict):
                            regime_info = regime_result.get("regime", "unknown")
                    except Exception:
                        pass
                market_data = {
                    "volatility": quality_breakdown.get("volatility", 0.02),
                    "regime": regime_info,
                }
                context = ide.enrich_context(symbol, order_book=order_book, market_data=market_data)

                # 2. 元决策：判断当前是否应该做决策
                verdict, reason, meta_details = await ide.meta_decide(
                    symbol=symbol,
                    decision_urgency=DecisionUrgency.NORMAL,
                    market_volatility=quality_breakdown.get("volatility", 0.02),
                    current_exposure_pct=quality_breakdown.get("exposure_pct", 0.0),
                )

                if verdict == MetaDecisionVerdict.ABSTAIN:
                    logger.info(f"Meta-decision ABSTAIN for {symbol}: {reason}")
                    self._push_dead_letter(signal_dict, f"meta_abstain:{reason}")
                    return
                elif verdict == MetaDecisionVerdict.DEFER:
                    logger.info(f"Meta-decision DEFER for {symbol}: {reason}")
                    self._push_dead_letter(signal_dict, f"meta_defer:{reason}")
                    return
                elif verdict == MetaDecisionVerdict.REDUCE_SIZE:
                    logger.info(f"Meta-decision REDUCE_SIZE for {symbol}: {reason}")
                    # 减仓：将仓位降低50%
                    if signal_dict.get("quantity"):
                        signal_dict["quantity"] = safe_float(signal_dict["quantity"], 0.0) * 0.5
                elif verdict == MetaDecisionVerdict.EMERGENCY_ONLY:
                    if signal_dict.get("signal_type") not in ("stop_loss", "take_profit", "liquidation"):
                        logger.info(f"Meta-decision EMERGENCY_ONLY for {symbol}: {reason}")
                        self._push_dead_letter(signal_dict, f"meta_emergency_only:{reason}")
                        return

                # 3. 决策成本收益分析（如果有止盈止损价格）
                tp_price = signal_dict.get("take_profit_price") or signal_dict.get("target_price")
                sl_price = signal_dict.get("stop_loss_price")
                entry_price = safe_float(signal_dict.get("price"), 0.0)
                qty = safe_float(signal_dict.get("quantity"), 0.0)
                if tp_price and sl_price and entry_price > 0 and qty > 0:
                    cba = ide.analyze_cost_benefit(
                        decision_id=signal_id,
                        symbol=symbol,
                        direction=direction,
                        quantity=qty,
                        entry_price=entry_price,
                        target_price=safe_float(tp_price, 0.0),
                        stop_price=safe_float(sl_price, 0.0),
                        leverage=safe_float(signal_dict.get("leverage"), 5.0),
                        expected_hold_hours=4.0,
                        win_probability=confidence,
                    )
                    if not cba.is_profitable:
                        logger.info(f"Decision CBA rejected for {symbol}: {cba.recommendation}")
                        self._push_dead_letter(signal_dict, f"cba_rejected:{cba.recommendation}")
                        return
                    # 将CBA结果注入信号上下文
                    signal_dict["cost_benefit_analysis"] = {
                        "expected_profit": cba.expected_profit,
                        "total_cost": cba.total_cost,
                        "net_expected_value": cba.net_expected_value,
                        "breakeven_move_pct": cba.breakeven_move_pct,
                        "is_profitable": cba.is_profitable,
                    }

                # 4. 决策异常检测
                recent = list(ide._decision_history)[-20:]
                recent_dicts = [{
                    "quantity": d.get("quantity", 0),
                    "direction": d.get("direction", ""),
                    "symbol": d.get("symbol", ""),
                    "confidence": d.get("confidence", 0.5),
                } for d in recent]
                is_anomaly, anomaly_reason = ide.detect_anomaly(
                    signal_dict, recent_decisions=recent_dicts
                )
                if is_anomaly:
                    logger.warning(f"Decision anomaly detected for {symbol}: {anomaly_reason}")
                    signal_dict["anomaly_warning"] = anomaly_reason
                    # 异常决策不直接拒绝，但标记并降低置信度
                    signal_dict["confidence"] = max(0.15, confidence * 0.7)

                # 5. 自适应阈值检查
                if confidence < ide.get_current_threshold():
                    logger.info(f"Signal confidence {confidence:.2%} below adaptive threshold "
                               f"{ide.get_current_threshold():.2%} for {symbol}")
                    self._push_dead_letter(signal_dict,
                        f"below_threshold:{confidence:.2%}<{ide.get_current_threshold():.2%}")
                    return

                # 6. 记录决策到企业级因果链（不可变审计链 + 因果归因）
                try:
                    ide.build_and_record_audit_entry(
                        decision_id=signal_id,
                        symbol=symbol,
                        strategy_name=strategy_name,
                        direction=direction,
                        decision_type=signal_dict.get("signal_type", strategy_name),
                        confidence=confidence,
                        source_signals=[{
                            "source": strategy_name,
                            "direction": direction,
                            "strength": confidence,
                            "confidence": confidence,
                            "signal_type": signal_dict.get("signal_type", ""),
                        }],
                        context=context,
                        cost_benefit=cba,
                        meta_verdict=verdict.value,
                        is_anomaly=is_anomaly,
                        anomaly_reason=anomaly_reason,
                    )
                except Exception as _causal_err:
                    logger.warning(f"Failed to record causal chain entry for {symbol}: {_causal_err}")

            except Exception as e:
                logger.error(f"IntelligentDecisionEngine processing error for {symbol}: {e}")
                # P0-3: dead letter + alert，不再静默丢弃信号
                self._push_dead_letter(signal_dict, f"IntelligentDecisionEngine exception: {e}")
                if self._alert_manager:
                    try:
                        await self._alert_manager.send_alert(
                            alert_type="signal_processing_error",
                            message=f"{symbol} IntelligentDecisionEngine 异常，信号已入 dead letter: {e}",
                            severity="WARNING",
                            symbol=symbol,
                            metadata={"signal_id": signal_id, "strategy": strategy_name},
                        )
                    except Exception:
                        pass
                # 引擎异常时不下发信号，保证安全
                return

        # ── 决策集成仲裁：规则引擎门控 + ML 置信度修正（均 fail-open）──
        ensemble_pass, confidence, ensemble_reason = self._decision_ensemble_gate(signal_dict, confidence)
        if not ensemble_pass:
            self._push_dead_letter(signal_dict, ensemble_reason)
            return
        signal_dict["confidence"] = confidence
        _pipe_ctx.end_stage("decision_engine")

        _pipe_ctx.begin_stage("sizing")
        adjusted_signal = self._apply_dynamic_sizing(signal_dict)
        adjusted_signal["signal_quality"] = quality_breakdown
        _pipe_ctx.end_stage("sizing")

        _pipe_ctx.begin_stage("execution")

        # 决策层正向溯源：决策通过全部校验（验证+校准+智能决策），最终批准准备下单。
        self._publish_decision_event(signal_dict, EventType.DECISION_APPROVED, "approved")

        # 信号通过所有校验，记录signal_id到已处理集合（幂等去重）
        self._processed_signal_ids.add(signal_id)
        if len(self._processed_signal_ids) > self._max_signal_id_cache:
            # P0: FIFO清理 —— 从有序队列中删除最旧的一半
            self._signal_id_order.append(signal_id)
            if len(self._signal_id_order) > self._max_signal_id_cache * 2:
                # 清理：删除前一半最旧的ID，保留后一半
                keep_count = self._max_signal_id_cache // 2
                to_remove = set(self._signal_id_order[:keep_count])
                self._processed_signal_ids -= to_remove
                self._signal_id_order = self._signal_id_order[keep_count:]

        _pipe_ctx.end_stage("execution")
        _pipe_ctx.finish()

        # 全链路延迟追踪埋点：信号到达 → 风控 → 下单 端到端延迟与 SLA 监控
        total_latency_ms = (time.perf_counter() - _t0) * 1000
        tasks = []
        if self._execution_monitor is not None:
            async def _record_latency():
                try:
                    from execution.execution_monitor import LatencyType
                    await self._execution_monitor.record_latency(
                        LatencyType.E2E, total_latency_ms,
                        order_id=signal_id, symbol=symbol, strategy=strategy_name,
                    )
                except Exception as e:
                    logger.debug(f"ExecutionMonitor latency record error: {e}")
            tasks.append(_record_latency())
        if self._anomaly_detector is not None:
            async def _detect_latency():
                try:
                    await self._anomaly_detector.detect("latency", {
                        "latency_ms": total_latency_ms,
                        "symbol": symbol,
                    })
                except Exception as e:
                    logger.debug(f"AnomalyDetector latency detection error: {e}")
            tasks.append(_detect_latency())
        if tasks:
            await asyncio.gather(*tasks)
        # SLA：>500ms 记录告警（非关键校验降级提示）
        if total_latency_ms > 500:
            logger.warning(
                f"E2E latency {total_latency_ms:.0f}ms exceeds 500ms SLA for {symbol} {strategy_name}"
            )

        routed = await self._route_signal(adjusted_signal)

        if self._alert_manager:
            try:
                await self._alert_manager.send_trade_signal_alert(adjusted_signal)
            except Exception as e:
                logger.error(f"Failed to send trade signal alert for {symbol}: {e}")
        return routed

    def _reject_bear_case_open_signal(
        self, signal_dict: Dict[str, Any], is_close_sig: bool
    ) -> bool:
        """Reject a new opening signal when the AGI projection crosses its bear-case limit."""
        gate = self._bear_case_open_gate
        if is_close_sig or gate is None:
            return False

        strategy_name = str(signal_dict.get("strategy_name", "") or "")
        symbol = str(signal_dict.get("symbol", "") or "")
        try:
            block_reason = gate(strategy_name)
        except Exception as exc:
            logger.exception(
                f"Bear-case opening gate failed for {strategy_name}; rejecting open signal"
            )
            self._push_dead_letter(
                signal_dict,
                f"bear_case_gate_unavailable:{type(exc).__name__}",
                layer="pnl_projection",
                reason_code="bear_case_gate_unavailable",
            )
            return True
        if not block_reason:
            return False

        logger.warning(
            f"Bear-case projection rejected open signal: "
            f"strategy={strategy_name} symbol={symbol} reason={block_reason}"
        )
        self._push_dead_letter(
            signal_dict,
            f"bear_case_projection:{block_reason}",
            layer="pnl_projection",
            reason_code="bear_case_threshold",
        )
        return True

    async def _handle_risk_block(self, signal_data: Dict[str, Any], risk_result) -> None:
        """
        五层风控拦截处理：丢弃信号 + 记录拦截原因 + 推送风险告警 + 不发起任何下单请求

        Args:
            signal_data: 原始信号
            risk_result: RiskGate.validate() 返回的 RiskGateResult
        """
        symbol = signal_data.get("symbol", "")
        strategy_name = signal_data.get("strategy_name", "")
        signal_type = signal_data.get("signal_type", "")
        direction = signal_data.get("direction", "")
        blocked_layer = risk_result.blocked_layer.value if risk_result.blocked_layer else "unknown"
        action = risk_result.action.value if hasattr(risk_result.action, 'value') else str(risk_result.action)
        reason = risk_result.summary or "unknown"

        # 1. 记录拦截原因（结构化日志）
        logger.warning(
            f"⛔ RiskGate BLOCKED signal | "
            f"layer={blocked_layer} | action={action} | "
            f"strategy={strategy_name} | symbol={symbol} | "
            f"signal_type={signal_type} | direction={direction} | "
            f"reason={reason}"
        )

        # 1.5 推入死信队列
        self._push_dead_letter(signal_data, f"risk_gate_blocked:{blocked_layer}")

        # 2. 记录到信号历史（便于 Dashboard 查询拦截统计）
        try:
            interception_record = {
                "symbol": symbol,
                "strategy_name": strategy_name,
                "signal_type": signal_type,
                "direction": direction,
                "blocked_layer": blocked_layer,
                "action": action,
                "reason": reason,
                "results": [r.to_dict() for r in risk_result.results] if risk_result.results else [],
                "timestamp": datetime.now().isoformat(),
            }
            self._signal_history.setdefault("_risk_interceptions", []).append(interception_record)
            # 限制历史长度
            if len(self._signal_history["_risk_interceptions"]) > 200:
                self._signal_history["_risk_interceptions"] = \
                    self._signal_history["_risk_interceptions"][-200:]
        except Exception as e:
            logger.debug(f"Failed to record interception history: {e}")

        # 3. 推送风险告警（不阻塞主流程）
        if self._alert_manager:
            try:
                metadata = {
                    "blocked_layer": blocked_layer,
                    "action": action,
                    "strategy_name": strategy_name,
                    "signal_type": signal_type,
                    "direction": direction,
                    "full_results": [r.to_dict() for r in risk_result.results] if risk_result.results else [],
                }
                await self._alert_manager.send_risk_alert(
                    risk_type=blocked_layer,
                    message=f"风控拦截: {strategy_name} {symbol} {signal_type} - {reason}",
                    symbol=symbol,
                    metadata=metadata
                )
            except Exception as e:
                logger.error(f"Failed to push risk interception alert: {e}")

        # 4. 不发起任何下单请求（直接 return，不下发至 order_executor）

    async def _route_signal(self, signal_data: Dict[str, Any]) -> bool:
        """智能路由：根据信号特性选择最优处理路径"""
        strategy_name = signal_data.get("strategy_name", "")
        signal_type = signal_data.get("signal_type", "")
        symbol = signal_data.get("symbol", "")
        confidence = signal_data.get("confidence", 0.5)
        
        priority = self._get_signal_priority(signal_data)
        
        # P0: 统一路由所有信号到order_executor（之前三个分支执行相同代码，已简化）
        logger.debug(f"Routing signal: {signal_type} {symbol} priority={priority}")
        try:
            routed = await self._order_executor.handle_signal(signal_data)
            if routed is not False:
                from core.signal_flow_stats import record_signal_flow_event
                record_signal_flow_event(
                    "executor_queued",
                    strategy=str(signal_data.get("strategy_name", "") or ""),
                )
            return routed is not False
        except Exception as e:
            logger.error(f"Order execution failed for {symbol} {strategy_name}: {e}")
            # 接入企业级自愈（RecoveryHandler）：订单执行失败触发恢复处理
            if self._recovery_handler is not None:
                try:
                    await self._recovery_handler.handle_failure("order_failed", {
                        "symbol": symbol,
                        "strategy": strategy_name,
                        "order_data": signal_data,
                        "error": str(e),
                        "retry_count": 0,
                    })
                except Exception as re:
                    logger.error(f"RecoveryHandler failed for {symbol}: {re}")
            return False

    def _get_signal_priority(self, signal_data: Dict[str, Any]) -> int:
        """计算信号优先级"""
        strategy_name = signal_data.get("strategy_name", "")
        signal_type = signal_data.get("signal_type", "")
        confidence = signal_data.get("confidence", 0.5)
        quality_score = signal_data.get("signal_quality", {}).get("overall_score", 0.5)
        
        base_priority = self._priority_map.get(strategy_name, 1)
        
        if signal_type in ("stop_loss", "take_profit", "close", "liquidation"):
            return 5
        
        if confidence > 0.8:
            base_priority += 1
        elif confidence < 0.4:
            base_priority = max(1, base_priority - 1)
        
        if quality_score > 0.8:
            base_priority += 1
        elif quality_score < 0.4:
            base_priority = max(1, base_priority - 1)
        
        return min(5, max(1, base_priority))

    async def _aggregate_signals(self):
        """信号聚合：在时间窗口内聚合相同方向的信号"""
        while True:
            await asyncio.sleep(self._signal_aggregation_window)
            
            async with self._processing_lock:
                if not self._pending_signals:
                    continue
                
                for agg_key, signals in list(self._pending_signals.items()):
                    if len(signals) >= 2:
                        aggregated = self._merge_signals(signals)
                        logger.info(f"Aggregated {len(signals)} signals for {agg_key}")
                        await self._process_signal(aggregated)
                
                self._pending_signals.clear()

    def _merge_signals(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """合并多个信号为一个综合信号"""
        if len(signals) == 1:
            return signals[0]
        
        merged = signals[0].copy()
        
        quantities = [safe_float(s.get("quantity"), 0.0) for s in signals]
        merged["quantity"] = sum(quantities)
        
        confidences = [safe_float(s.get("confidence"), 0.5) for s in signals]
        merged["confidence"] = safe_finite(sum(confidences) / max(len(confidences), 1), 0.5)
        
        prices = [safe_float(s.get("price"), 0.0) for s in signals if safe_float(s.get("price"), 0.0) > 0]
        if prices:
            merged["price"] = safe_finite(sum(prices) / len(prices), 0.0)
        
        merged["source"] = "+".join(set(s.get("source", s.get("strategy_name", "")) for s in signals))
        merged["signal_count"] = len(signals)
        
        return merged

    def _evaluate_signal_quality(self, signal_data: Dict[str, Any]) -> tuple:
        """评估信号质量
        
        优先使用 QualityEngine，无引擎时使用轻量级内置校验。
        """
        if self._quality_engine:
            is_acceptable, breakdown = self._quality_engine.is_signal_acceptable(signal_data)
            return is_acceptable, breakdown
        
        # 轻量级内置校验：无 QualityEngine 时的降级评估
        conf = signal_data.get("confidence", 0.5)
        symbol = signal_data.get("symbol", "")
        direction = signal_data.get("direction", "")
        price = signal_data.get("price", 0)
        quantity = signal_data.get("quantity", 0)
        strategy = signal_data.get("strategy_name", "")
        
        issues = []
        quality_score = 0.70  # 基础分
        
        # 基本字段校验
        if not symbol or not direction or price <= 0 or quantity <= 0:
            issues.append("missing_fields")
            quality_score = 0.0
        
        # 置信度评估
        if conf >= 0.9:
            quality_score += 0.15
        elif conf >= 0.7:
            quality_score += 0.10
        elif conf < 0.5:
            quality_score -= 0.20
        
        # 币种白名单检查
        supported = self.config.get("currencies", {})
        tier_symbols = set()
        for tier in ["tier1", "tier2", "tier3"]:
            for base in supported.get(f"{tier}_symbols", []):
                tier_symbols.add(f"{base}-USDT-SWAP")
        if symbol not in tier_symbols:
            quality_score -= 0.15
            issues.append("unsupported_symbol")
        
        # 策略启用检查
        strategy_cfg = self.config.get("strategies", {}).get(strategy, {})
        if isinstance(strategy_cfg, dict) and not strategy_cfg.get("enabled", True):
            quality_score -= 0.3
            issues.append("strategy_disabled")
        
        # 市场状态匹配检查
        if self._regime_engine:
            try:
                regime_info = self._regime_engine.get_regime()
                regime = str(regime_info.get("regime", "unknown")).lower()
                regime_strength = safe_float(regime_info.get("strength"), 0.0)
                direction_key = str(direction).lower()
                is_long = direction_key in ("long", "buy")
                is_short = direction_key in ("short", "sell")
                
                # range_bound 市场 + 网格策略 = 良好
                # trend 市场 + 网格策略 = 降分
                if strategy == "grid" and regime in ("trend_bullish", "trend_bearish"):
                    quality_score -= 0.15
                    issues.append("grid_in_trend")
                elif strategy == "trend" and regime == "range_bound":
                    quality_score -= 0.20
                    issues.append("trend_in_range")
                elif strategy == "scalping" and regime == "extreme_volatility":
                    # 抢单策略在极端波动中容易被扫损
                    quality_score -= 0.25
                    issues.append("scalping_in_extreme_vol")
                elif strategy == "scalping" and regime == "liquidity_crisis":
                    quality_score -= 0.30
                    issues.append("scalping_in_low_liquidity")
                elif strategy == "arbitrage" and regime in ("extreme_volatility", "liquidity_crisis"):
                    quality_score -= 0.20
                    issues.append("arbitrage_in_high_risk")
                elif regime == "funding_crush" and regime_strength > 0.7:
                    # 资金费率极端时所有开仓降分
                    if direction == "long":
                        quality_score -= 0.15
                        issues.append("long_in_funding_crush")
                elif regime == "breakout":
                    if is_short:
                        quality_score -= 0.25
                        issues.append("short_against_breakout")
                    elif strategy in ("grid", "spot_grid", "spot_martingale"):
                        quality_score -= 0.20
                        issues.append("mean_reversion_in_breakout")
                elif regime == "breakdown":
                    if is_long:
                        quality_score -= 0.25
                        issues.append("long_against_breakdown")
                    elif strategy in ("grid", "spot_grid", "spot_martingale"):
                        quality_score -= 0.20
                        issues.append("mean_reversion_in_breakdown")
                elif regime == "reversal":
                    reversal_direction = str(signal_data.get("reversal_direction", "")).lower()
                    confirmed = bool(signal_data.get("reversal_confirmed"))
                    direction_matches = (
                        (is_long and reversal_direction in ("long", "buy", "bullish"))
                        or (is_short and reversal_direction in ("short", "sell", "bearish"))
                    )
                    if not confirmed or not direction_matches:
                        quality_score -= 0.25
                        issues.append("unconfirmed_reversal")
            except Exception:
                pass
        
        # 止损比例合理性检查：止损过近（<0.3%）或过远（>10%）降分
        sl_price = signal_data.get("stop_loss", signal_data.get("stop_loss_price"))
        if sl_price and price > 0 and direction:
            try:
                sl_pct = safe_div(abs(safe_float(sl_price, 0.0) - price), price, 0.0)
                if sl_pct < 0.003:
                    quality_score -= 0.15
                    issues.append("stop_loss_too_tight")
                elif sl_pct > 0.10:
                    quality_score -= 0.10
                    issues.append("stop_loss_too_wide")
            except (TypeError, ValueError):
                pass
        
        # 当前时间是否在交易窗口内
        trade_hours = self.config.get("trading", {}).get("trading_hours", {})
        if trade_hours:
            try:
                now_hour = datetime.now().hour
                start = trade_hours.get("start", 0)
                end = trade_hours.get("end", 24)
                if not (start <= now_hour < end):
                    quality_score -= 0.5
                    issues.append("outside_trading_hours")
            except Exception:
                pass
        
        quality_score = max(0.0, min(1.0, quality_score))
        
        quality_label = "excellent" if quality_score >= 0.8 else \
                       "good" if quality_score >= 0.6 else \
                       "poor" if quality_score >= 0.3 else "rejected"
        
        is_acceptable = quality_score >= 0.35  # 降低门槛到 0.35
        
        breakdown = {
            "overall_score": round(quality_score, 3),
            "quality": quality_label,
            "issues": issues,
            "confidence": conf,
            "source": "fallback_evaluator",
        }
        
        return is_acceptable, breakdown

    def _record_signal(self, signal_data: Dict[str, Any], quality_breakdown: Dict[str, Any]):
        """记录信号历史"""
        strategy = signal_data.get("strategy_name", "")
        if strategy not in self._signal_history:
            self._signal_history[strategy] = []
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": signal_data.get("symbol", ""),
            "direction": signal_data.get("direction", ""),
            "confidence": signal_data.get("confidence", 0.5),
            "quality_score": quality_breakdown.get("overall_score", 0),
            "quality": quality_breakdown.get("quality", "unknown"),
        }
        
        self._signal_history[strategy].append(entry)
        
        if len(self._signal_history[strategy]) > 100:
            self._signal_history[strategy] = self._signal_history[strategy][-100:]

    def _check_strategy_priority(self, symbol: str, strategy_name: str, 
                                  direction: str, confidence: float, 
                                  quality_breakdown: Dict[str, Any] = None) -> bool:
        current_priority = self._priority_map.get(strategy_name, 0)
        quality_score = quality_breakdown.get("overall_score", 0.5) if quality_breakdown else 0.5
        
        adjusted_priority = current_priority * (0.7 + quality_score * 0.3)
        
        # P2: 防饥饿 - 连续阻塞超过阈值时强制放行
        block_key = f"{symbol}:{strategy_name}"
        consecutive = self._consecutive_blocks.get(block_key, 0)
        if consecutive >= self._max_consecutive_blocks:
            logger.info(f"Anti-starvation: {strategy_name} {symbol} force-released after {consecutive} consecutive blocks")
            self._consecutive_blocks[block_key] = 0
            self._last_signal_time[f"{symbol}:{strategy_name}:{direction}"] = datetime.now()
            return True
        
        # P2: 提高阻塞阈值从1.1到1.5，减少不必要的信号延迟
        priority_threshold = 1.5
        
        for s_name, s_priority in self._priority_map.items():
            if s_name == strategy_name:
                continue
            
            other_key = f"{symbol}:{s_name}:{direction}"
            other_time = self._last_signal_time.get(other_key)
            
            if other_time:
                age_seconds = (datetime.now() - other_time).total_seconds()
                
                if age_seconds < 60:
                    other_adjusted = s_priority * 1.1
                elif age_seconds < 120:
                    other_adjusted = s_priority * 0.9
                else:
                    continue
                
                if other_adjusted > adjusted_priority * priority_threshold:
                    self._consecutive_blocks[block_key] = consecutive + 1
                    logger.info(f"Signal deferred: {strategy_name} {symbol} "
                               f"(priority={adjusted_priority:.2f}) blocked by {s_name} "
                               f"(priority={other_adjusted:.2f}) [block #{consecutive + 1}]")
                    return False
        
        # 放行：重置阻塞计数
        self._consecutive_blocks[block_key] = 0
        
        if self._regime_engine:
            recommendation = self._regime_engine.get_strategy_recommendation(strategy_name)
            adjustment_factor = recommendation.get("adjustment_factor", 1.0)
            
            if adjustment_factor < 0.7:
                logger.info(f"Signal deferred: {strategy_name} incompatible with current market regime "
                           f"(adjustment={adjustment_factor:.2f})")
                return False
        
        return True

    def _apply_dynamic_sizing(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        # 去重保护：统一仓位引擎已收敛仓位口径时，跳过旧 sizing，避免重复放大。
        if signal_data.get("position_sizing") == "unified":
            return dict(signal_data)

        confidence = safe_float(signal_data.get("confidence"), 0.5)
        strategy_name = signal_data.get("strategy_name", "")
        quality_score = safe_float(signal_data.get("signal_quality", {}).get("overall_score"), 0.5)

        open_count = len(getattr(self._trade_journal, '_open_positions', {})) if self._trade_journal else 0
        max_positions = self.config["trading"].get("max_concurrent_positions", 4)

        position_ratio = open_count / max_positions if max_positions > 0 else 0

        if position_ratio >= 0.75:
            size_multiplier = 0.4
        elif position_ratio >= 0.5:
            size_multiplier = 0.6
        elif position_ratio >= 0.25:
            size_multiplier = 0.85
        else:
            size_multiplier = 1.0

        if confidence > 0.8:
            size_multiplier *= 1.15
        elif confidence < 0.55:
            size_multiplier *= 0.75

        if quality_score > 0.8:
            size_multiplier *= 1.10
        elif quality_score < 0.5:
            size_multiplier *= 0.85

        if self._regime_engine:
            adjustment = self._regime_engine.get_position_adjustment()
            regime_factor = adjustment.get(strategy_name, adjustment.get("overall", 1.0))
            size_multiplier *= regime_factor
            logger.debug(f"Regime adjustment: {strategy_name} x{regime_factor:.2f}")

        actual_equity = self._get_actual_equity()
        configured_capital = self.config["trading"].get("total_capital", 100)
        if actual_equity > 0 and configured_capital > 0:
            equity_ratio = actual_equity / configured_capital
            if equity_ratio < 1.0:
                size_multiplier *= equity_ratio
                logger.debug(f"Equity adjustment: actual={actual_equity:.2f}, configured={configured_capital}, ratio={equity_ratio:.4f}")

        if actual_equity > 0:
            if self._profit_optimizer:
                self._profit_optimizer.update_equity(actual_equity)

        if self._adaptive_controller:
            alloc_ratio = self._adaptive_controller.get_allocation(strategy_name)
            base_alloc = self.config["trading"].get(f"{strategy_name}_allocation", 0.20)
            if base_alloc > 0:
                alloc_multiplier = alloc_ratio / base_alloc
                size_multiplier *= alloc_multiplier
                logger.debug(f"Adaptive allocation: {strategy_name} ratio={alloc_ratio:.3f} (base={base_alloc:.3f}, x{alloc_multiplier:.2f})")

        if self._profit_optimizer:
            compound_factor = self._profit_optimizer._compound_factor
            kelly_factor = self._profit_optimizer.get_kelly_fraction()
            drawdown_factor = self._profit_optimizer._drawdown_factor
            size_multiplier *= compound_factor * kelly_factor * drawdown_factor

        # 小资金保护阈值提升至1500 USDT，让中等账户（500-1500）也能享受保护
        # 之前阈值500导致 500-1500 USDT 账户被 kelly_factor + drawdown_factor 过度压缩
        if actual_equity > 0 and actual_equity < 1500:
            if actual_equity >= 800:
                min_multiplier = 0.7
            elif actual_equity >= 500:
                min_multiplier = 0.8
            elif actual_equity >= 200:
                min_multiplier = 0.9
            else:
                min_multiplier = 1.0
            if size_multiplier < min_multiplier:
                size_multiplier = min_multiplier
                logger.debug(f"Small cap adjustment: size multiplier raised to {min_multiplier:.2f} (equity={actual_equity:.2f})")

        idle_boost = 1.0
        if self._adaptive_controller:
            idle_boost = self._adaptive_controller.get_position_boost()
        if idle_boost > 1.0:
            size_multiplier *= idle_boost
            logger.debug(f"Idle cash boost: x{idle_boost:.2f} applied to {strategy_name}")

        # P14: 下限0.5，上限提升至5.0以充分利用闲置资金
        size_multiplier = max(0.5, min(5.0, size_multiplier))

        adjusted = dict(signal_data)
        original_qty = adjusted.get("quantity", 0)
        if original_qty and isinstance(original_qty, (int, float)):
            adjusted["quantity"] = original_qty * size_multiplier

        logger.debug(f"Dynamic sizing: {strategy_name} qty x{size_multiplier:.2f} "
                     f"(positions={open_count}/{max_positions}, conf={confidence:.2f}, "
                     f"quality={quality_score:.2f}, "
                     f"idle_boost={idle_boost:.2f})")

        return adjusted

    def _get_actual_equity(self) -> float:
        try:
            account_info = self._account_manager.get_account_info()
            if account_info:
                return safe_float(account_info.get("totalEq"), 0.0)
        except Exception as e:
            logger.debug(f"Failed to get actual equity: {e}")
        return 0.0

    def _normalize_direction(self, direction: str) -> str:
        direction_map = {
            "buy": "long",
            "sell": "short",
            "long": "long",
            "short": "short"
        }
        return direction_map.get(direction.lower(), "long")

    async def _check_signal_conflict(self, signal_data: Dict[str, Any]) -> bool:
        symbol = signal_data.get("symbol", "")
        direction = self._normalize_direction(signal_data.get("direction", ""))
        strategy_name = signal_data.get("strategy_name", "")
        signal_type = signal_data.get("signal_type", "").lower()
        is_close_signal = "close" in signal_type or "take_profit" in signal_type or "stop_loss" in signal_type
        
        if not symbol or not direction:
            return True

        if is_close_signal:
            return True
        
        # 网格策略豁免方向冲突检查：网格策略天然需要双向持仓（buy层+ sell层）
        if "grid" in strategy_name.lower():
            return True
        
        open_positions = self._trade_journal._open_positions
        
        if symbol in open_positions:
            existing_position = open_positions[symbol]
            existing_direction = self._normalize_direction(existing_position.direction)
            opposite_direction = "short" if direction == "long" else "long"
            
            if existing_direction == opposite_direction:
                logger.warning(f"Signal conflict: {strategy_name} {direction} {symbol} conflicts with existing {existing_direction} position")
                return False
        
        return True