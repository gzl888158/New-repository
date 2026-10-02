"""
统一抽象层 - Unified Abstraction Layer

功能：
1. 统一日志抽象（支持多输出、结构化日志）
2. 统一异常处理抽象（分类、分级、自动恢复）
3. 统一配置验证抽象（Schema验证、依赖检查）
4. 统一事件总线（解耦组件通信）
"""

import asyncio
import json
import threading
import traceback
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
from enum import Enum
from abc import ABC, abstractmethod
from loguru import logger


# ==================== 统一日志抽象 ====================

class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogField:
    """结构化日志字段"""
    
    def __init__(self, **kwargs):
        self.fields = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        return self.fields


class AbstractLogger(ABC):
    """日志抽象接口"""
    
    @abstractmethod
    def debug(self, message: str, **kwargs):
        pass
    
    @abstractmethod
    def info(self, message: str, **kwargs):
        pass
    
    @abstractmethod
    def warning(self, message: str, **kwargs):
        pass
    
    @abstractmethod
    def error(self, message: str, **kwargs):
        pass
    
    @abstractmethod
    def critical(self, message: str, **kwargs):
        pass
    
    @abstractmethod
    def log(self, level: LogLevel, message: str, **kwargs):
        pass


class StructuredLogger(AbstractLogger):
    """结构化日志实现"""
    
    # 已注册的文件 sink 路径（类级别，防止多次实例化重复添加 sink 导致日志重复输出）
    _registered_sinks: set = set()
    _sinks_lock = threading.Lock()
    
    def __init__(self, service_name: str = "trading_system"):
        self._service_name = service_name
        self._context: Dict[str, Any] = {}
        
        self._register_file_sink()
        logger.info(f"StructuredLogger initialized for {service_name}")
    
    @classmethod
    def _register_file_sink(cls):
        """注册文件 sink（幂等：同一路径只添加一次，避免日志重复输出）。"""
        sink_path = "data/logs/system_{time:YYYY-MM-DD}.log"
        with cls._sinks_lock:
            if sink_path in cls._registered_sinks:
                return
            logger.add(
                sink_path,
                format="{time} | {level} | {message}",
                rotation="1 day",
                retention="30 days",
                compression="zip"
            )
            cls._registered_sinks.add(sink_path)
    
    def set_context(self, **kwargs):
        """设置日志上下文（全局）"""
        self._context.update(kwargs)
    
    def _format_message(self, message: str, **kwargs) -> str:
        """格式化结构化日志消息"""
        context = {**self._context, **kwargs}
        
        if context:
            structured = json.dumps(context, ensure_ascii=False, default=str)
            return f"{message} | {structured}"
        
        return message
    
    def debug(self, message: str, **kwargs):
        logger.debug(self._format_message(message, **kwargs))
    
    def info(self, message: str, **kwargs):
        logger.info(self._format_message(message, **kwargs))
    
    def warning(self, message: str, **kwargs):
        logger.warning(self._format_message(message, **kwargs))
    
    def error(self, message: str, **kwargs):
        logger.error(self._format_message(message, **kwargs))
    
    def critical(self, message: str, **kwargs):
        logger.critical(self._format_message(message, **kwargs))
    
    def log(self, level: LogLevel, message: str, **kwargs):
        getattr(logger, level.value.lower())(self._format_message(message, **kwargs))
    
    def log_trade(self, trade_data: Dict[str, Any]):
        """记录交易日志"""
        self.info("Trade executed", **trade_data)
    
    def log_signal(self, signal_data: Dict[str, Any]):
        """记录信号日志"""
        self.info("Signal generated", **signal_data)
    
    def log_risk_event(self, risk_event: Dict[str, Any]):
        """记录风控事件"""
        self.warning("Risk event", **risk_event)


# ==================== 统一异常处理抽象 ====================

class ExceptionSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExceptionCategory(Enum):
    NETWORK = "network"
    API = "api"
    DATABASE = "database"
    VALIDATION = "validation"
    BUSINESS = "business"
    SYSTEM = "system"


class RecoveryResult(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class AbstractExceptionHandler(ABC):
    """异常处理抽象接口"""
    
    @abstractmethod
    def handle(self, exception: Exception, context: Dict[str, Any] = None) -> RecoveryResult:
        pass
    
    @abstractmethod
    def register_recovery_handler(self, exception_type: type, handler: Callable):
        pass
    
    @abstractmethod
    def get_exception_stats(self) -> Dict[str, Any]:
        pass


class UnifiedExceptionHandler(AbstractExceptionHandler):
    """统一异常处理器"""
    
    def __init__(self, logger: AbstractLogger = None):
        self._logger = logger or StructuredLogger()
        self._recovery_handlers: Dict[type, Callable] = {}
        self._exception_stats: Dict[str, int] = {}
        self._recovery_attempts: Dict[str, int] = {}
        
        # 注册默认恢复处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认恢复处理器"""
        from requests.exceptions import RequestException
        
        def network_recovery(exception: RequestException):
            self._logger.warning(f"Network error recovery attempt: {exception}")
            return RecoveryResult.PARTIAL
        
        self.register_recovery_handler(RequestException, network_recovery)
    
    def handle(self, exception: Exception, context: Dict[str, Any] = None) -> RecoveryResult:
        """处理异常"""
        exception_type = type(exception)
        context = context or {}
        
        # 记录异常
        self._logger.error(
            f"Exception: {exception_type.__name__}: {str(exception)}",
            category=ExceptionCategory.SYSTEM.value,
            severity=ExceptionSeverity.MEDIUM.value,
            **context
        )
        
        # 更新统计
        key = f"{exception_type.__name__}"
        self._exception_stats[key] = self._exception_stats.get(key, 0) + 1
        
        # 尝试恢复
        recovery_result = RecoveryResult.FAILED
        if exception_type in self._recovery_handlers:
            try:
                recovery_result = self._recovery_handlers[exception_type](exception)
                self._recovery_attempts[key] = self._recovery_attempts.get(key, 0) + 1
                
                if recovery_result == RecoveryResult.SUCCESS:
                    self._logger.info(f"Recovery successful for {exception_type.__name__}")
                elif recovery_result == RecoveryResult.PARTIAL:
                    self._logger.warning(f"Partial recovery for {exception_type.__name__}")
            except Exception as e:
                self._logger.error(f"Recovery failed: {e}")
        
        return recovery_result
    
    def register_recovery_handler(self, exception_type: type, handler: Callable):
        """注册恢复处理器"""
        self._recovery_handlers[exception_type] = handler
        self._logger.info(f"Registered recovery handler for {exception_type.__name__}")
    
    def get_exception_stats(self) -> Dict[str, Any]:
        """获取异常统计"""
        return {
            "exception_counts": self._exception_stats,
            "recovery_attempts": self._recovery_attempts
        }


# ==================== 统一配置验证抽象 ====================

class ValidationResult:
    """验证结果"""
    
    def __init__(self, valid: bool, errors: List[str] = None, warnings: List[str] = None):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
    
    def __bool__(self):
        return self.valid
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.valid = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)


class AbstractConfigValidator(ABC):
    """配置验证抽象接口"""
    
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        pass
    
    @abstractmethod
    def validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        pass
    
    @abstractmethod
    def validate_dependencies(self, config: Dict[str, Any]) -> ValidationResult:
        pass


class UnifiedConfigValidator(AbstractConfigValidator):
    """统一配置验证器"""
    
    def __init__(self, logger: AbstractLogger = None):
        self._logger = logger or StructuredLogger()
        self._schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_schema(self, name: str, schema: Dict[str, Any]):
        """注册验证Schema"""
        self._schemas[name] = schema
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """完整配置验证"""
        result = ValidationResult(True)
        
        # Schema验证
        for schema_name, schema in self._schemas.items():
            schema_result = self.validate_schema(config, schema)
            if not schema_result:
                result.errors.extend(schema_result.errors)
                result.valid = False
            result.warnings.extend(schema_result.warnings)
        
        # 依赖验证
        dep_result = self.validate_dependencies(config)
        if not dep_result:
            result.errors.extend(dep_result.errors)
            result.valid = False
        result.warnings.extend(dep_result.warnings)
        
        return result
    
    def validate_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """Schema验证"""
        result = ValidationResult(True)
        
        def validate_field(path: str, field_schema: Dict[str, Any], value: Any):
            field_name = path.split('.')[-1]
            
            # 必填检查
            if field_schema.get("required") and value is None:
                result.add_error(f"Missing required field: {path}")
                return
            
            if value is None:
                return
            
            # 类型检查
            expected_type = field_schema.get("type")
            if expected_type and not isinstance(value, expected_type):
                result.add_error(
                    f"Type mismatch for {path}: expected {expected_type.__name__}, got {type(value).__name__}"
                )
                return
            
            # 范围检查（仅对数值类型比较，避免 str/int 比较抛 TypeError）
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "min" in field_schema and value < field_schema["min"]:
                    result.add_error(f"{path} {value} is less than minimum {field_schema['min']}")
                
                if "max" in field_schema and value > field_schema["max"]:
                    result.add_error(f"{path} {value} is greater than maximum {field_schema['max']}")
            
            # 枚举检查
            if "enum" in field_schema and value not in field_schema["enum"]:
                result.add_error(f"{path} {value} not in allowed values: {field_schema['enum']}")
            
            # 子对象检查
            if isinstance(value, dict) and "properties" in field_schema:
                for sub_field, sub_schema in field_schema["properties"].items():
                    validate_field(f"{path}.{sub_field}", sub_schema, value.get(sub_field))
            
            # 列表检查
            if isinstance(value, list) and "items" in field_schema:
                for i, item in enumerate(value):
                    validate_field(f"{path}[{i}]", field_schema["items"], item)
        
        # 遍历Schema验证
        for field_name, field_schema in schema.items():
            validate_field(field_name, field_schema, config.get(field_name))
        
        return result
    
    def validate_dependencies(self, config: Dict[str, Any]) -> ValidationResult:
        """依赖验证"""
        result = ValidationResult(True)
        
        # 检查交易配置依赖
        trading = config.get("trading", {})
        
        # 资金分配总和检查
        allocation_fields = [
            "scalping_allocation", "trend_allocation", "grid_allocation",
            "arbitrage_allocation", "spot_grid_allocation", "spot_martingale_allocation"
        ]
        total_allocation = sum(trading.get(f, 0) for f in allocation_fields)
        if abs(total_allocation - 1.0) > 0.01:
            result.add_error(f"Strategy allocation sum {total_allocation:.4f} != 1.0")
        
        # 资本比例总和检查
        trading_ratio = trading.get("trading_capital_ratio", 0)
        risk_reserve = trading.get("risk_reserve_ratio", 0)
        profit_reserve = trading.get("profit_reserve_ratio", 0)
        capital_sum = trading_ratio + risk_reserve + profit_reserve
        if abs(capital_sum - 1.0) > 0.01:
            result.add_error(f"Capital ratio sum {capital_sum:.4f} != 1.0")
        
        # 检查策略启用依赖
        strategies = config.get("strategies", {})
        for strategy_name, strategy_config in strategies.items():
            if strategy_config.get("enabled"):
                allocation_key = f"{strategy_name}_allocation"
                if trading.get(allocation_key, 0) == 0:
                    result.add_warning(
                        f"Strategy {strategy_name} is enabled but has 0 allocation"
                    )
        
        # 检查Redis配置依赖
        redis = config.get("redis", {})
        if redis.get("password") and not redis.get("password").startswith("${"):
            result.add_warning("Redis password is hardcoded, consider using environment variable")
        
        return result


# ==================== 统一事件总线 ====================

class EventType(Enum):
    """事件类型"""
    SIGNAL_GENERATED = "signal_generated"
    ORDER_PLACED = "order_placed"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELLED = "order_cancelled"
    STOP_LOSS_TRIGGERED = "stop_loss_triggered"
    TAKE_PROFIT_TRIGGERED = "take_profit_triggered"
    POSITION_CLOSED = "position_closed"
    RISK_EVENT = "risk_event"
    RISK_ADJUDICATED = "risk_adjudicated"
    DECISION_VALIDATED = "decision_validated"  # 决策层：决策验证通过
    DECISION_APPROVED = "decision_approved"    # 决策层：决策最终批准（准备下单）
    DECISION_REJECTED = "decision_rejected"    # 决策层：决策验证拒绝
    DECISION_ENSEMBLE = "decision_ensemble"    # 决策层：集成仲裁（规则引擎+ML+共识）
    CONFIG_CHANGED = "config_changed"
    SYSTEM_HEALTH = "system_health"
    EXCEPTION_OCCURRED = "exception_occurred"
    SIGNAL_REJECTED = "signal_rejected"  # 信号层（signal_processor）丢弃
    ORDER_REJECTED = "order_rejected"    # 执行层（order_executor）拒绝
    POSITION_OPENED = "position_opened"  # 记账层：开仓落账
    TRADE_RECORDED = "trade_recorded"    # 记账层：平仓落账（成交记账）
    PIPELINE_STARTED = "pipeline_started"      # 流水线：开始执行
    PIPELINE_COMPLETED = "pipeline_completed"  # 流水线：成功完成
    PIPELINE_FAILED = "pipeline_failed"        # 流水线：阶段失败
    PIPELINE_TIMEOUT = "pipeline_timeout"      # 流水线：整体/阶段超时


# 事件 ID 单调计数器（保证同类型事件在同一时刻仍唯一）
_event_id_counter = 0
_event_id_counter_lock = threading.Lock()


class Event:
    """事件对象"""
    
    def __init__(self, event_type: EventType, data: Dict[str, Any], timestamp: datetime = None):
        self.event_type = event_type
        self.data = data
        self.timestamp = timestamp or datetime.now()
        self.event_id = self._generate_event_id(event_type, self.timestamp)
    
    @staticmethod
    def _generate_event_id(event_type: EventType, timestamp: datetime) -> str:
        """生成唯一事件 ID：微秒时间戳 + 单调计数器，避免同秒事件 ID 冲突。"""
        global _event_id_counter
        with _event_id_counter_lock:
            _event_id_counter += 1
            seq = _event_id_counter
        return f"{event_type.value}_{int(timestamp.timestamp() * 1_000_000)}_{seq:06d}"


class AbstractEventBus(ABC):
    """事件总线抽象接口"""
    
    @abstractmethod
    def subscribe(self, event_type: EventType, handler: Callable):
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: EventType, handler: Callable):
        pass
    
    @abstractmethod
    async def publish(self, event: Event):
        pass
    
    @abstractmethod
    def publish_sync(self, event: Event):
        pass


class LocalEventBus(AbstractEventBus):
    """本地事件总线实现"""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {}
        self._lock = threading.RLock()
        self._logger = StructuredLogger("event_bus")
        # P1: 可选持久化事件存储（默认 None = 不持久化，保持向后兼容）
        self._event_store = None

        logger.info("LocalEventBus initialized")
    
    def subscribe(self, event_type: EventType, handler: Callable):
        """订阅事件（线程安全）"""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                self._logger.info(f"Subscribed to {event_type.value}")
    
    def unsubscribe(self, event_type: EventType, handler: Callable):
        """取消订阅（线程安全）"""
        with self._lock:
            if event_type in self._subscribers and handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                self._logger.info(f"Unsubscribed from {event_type.value}")
    
    async def publish(self, event: Event):
        """异步发布事件"""
        self._persist(event)
        # 在锁内取订阅者快照，避免遍历期间并发修改订阅列表导致 RuntimeError
        with self._lock:
            handlers = list(self._subscribers.get(event.event_type, []))
        if not handlers:
            return
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                self._logger.error(f"Event handler error: {e}\n{traceback.format_exc()}")
    
    def publish_sync(self, event: Event):
        """同步发布事件"""
        self._persist(event)
        with self._lock:
            handlers = list(self._subscribers.get(event.event_type, []))
        if not handlers:
            return
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    # 同步发布无法执行协程 handler，直接调用只会产生未等待的协程对象
                    self._logger.error(
                        f"Coroutine handler {handler} cannot run in publish_sync; "
                        f"use publish() or a sync handler"
                    )
                    continue
                handler(event)
            except Exception as e:
                self._logger.error(f"Event handler error: {e}\n{traceback.format_exc()}")

    def set_event_store(self, event_store) -> None:
        """P1: 注入持久化事件存储（可选）。注入后，所有 publish 的事件将追加落盘。"""
        self._event_store = event_store

    def _persist(self, event: Event) -> None:
        """P1: 将事件持久化到 EventStore（若已注入）。持久化失败不阻断发布主流程。"""
        if self._event_store is None:
            return
        try:
            symbol = event.data.get("symbol", "") if isinstance(event.data, dict) else ""
            self._event_store.append(
                event_type=event.event_type.value,
                data=event.data,
                event_id=event.event_id,
                timestamp=event.timestamp,
                source="event_bus",
                symbol=symbol,
            )
        except Exception as e:
            self._logger.error(f"Event persistence failed: {e}")

    def _dispatch_async(self, event: Event):
        """调度异步发布；无运行中的事件循环时降级为同步发布，避免 create_task 抛 RuntimeError。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.publish_sync(event)
            return
        
        task = loop.create_task(self.publish(event))
        # 捕获 publish 内部未处理的意外异常，避免 "Task exception was never retrieved"
        def _on_done(t):
            try:
                t.result()
            except Exception as e:
                self._logger.error(f"Event publish failed: {e}\n{traceback.format_exc()}")
        task.add_done_callback(_on_done)
    
    def publish_signal(self, signal_data: Dict[str, Any]):
        """便捷方法：发布信号事件"""
        event = Event(EventType.SIGNAL_GENERATED, signal_data)
        self._dispatch_async(event)
    
    def publish_order(self, order_data: Dict[str, Any]):
        """便捷方法：发布订单事件"""
        event_type = EventType.ORDER_PLACED if order_data.get("status") == "pending" else EventType.ORDER_FILLED
        event = Event(event_type, order_data)
        self._dispatch_async(event)

    def publish_cancel(self, order_data: Dict[str, Any]):
        """便捷方法：发布撤单事件（P3：撤单事件统一经事件总线投递并落盘）。"""
        event = Event(EventType.ORDER_CANCELLED, order_data)
        self._dispatch_async(event)
    
    def publish_risk_event(self, risk_data: Dict[str, Any]):
        """便捷方法：发布风控事件"""
        event = Event(EventType.RISK_EVENT, risk_data)
        self._dispatch_async(event)


# ==================== 统一抽象层入口 ====================

class UnifiedAbstractionLayer:
    """统一抽象层入口"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self._logger = StructuredLogger("unified_layer")
        self._exception_handler = UnifiedExceptionHandler(self._logger)
        self._config_validator = UnifiedConfigValidator(self._logger)
        self._event_bus = LocalEventBus()
        
        # 注册默认配置Schema
        if config:
            self._register_default_schemas()
        
        logger.info("UnifiedAbstractionLayer initialized")
    
    def _register_default_schemas(self):
        """注册默认配置Schema"""
        trading_schema = {
            "total_capital": {
                "type": float,
                "required": True,
                "min": 10,
                "max": 1000000
            },
            "risk_per_trade": {
                "type": float,
                "required": True,
                "min": 0.001,
                "max": 0.1
            },
            "max_total_leverage": {
                "type": int,
                "required": True,
                "min": 1,
                "max": 125
            },
            "max_drawdown": {
                "type": float,
                "required": True,
                "min": 0.01,
                "max": 0.5
            }
        }
        
        self._config_validator.register_schema("trading", trading_schema)
    
    @property
    def logger(self) -> AbstractLogger:
        return self._logger
    
    @property
    def exception_handler(self) -> AbstractExceptionHandler:
        return self._exception_handler
    
    @property
    def config_validator(self) -> AbstractConfigValidator:
        return self._config_validator
    
    @property
    def event_bus(self) -> AbstractEventBus:
        return self._event_bus
    
    def validate_and_load_config(self, config: Dict[str, Any]) -> ValidationResult:
        """验证并加载配置"""
        result = self._config_validator.validate(config)
        
        if result:
            self._logger.info("Config validation passed", warnings=result.warnings)
        else:
            self._logger.error("Config validation failed", errors=result.errors)
        
        return result
    
    async def start(self):
        """启动统一抽象层"""
        self._logger.info("UnifiedAbstractionLayer starting")
    
    async def shutdown(self):
        """关闭统一抽象层"""
        self._logger.info("UnifiedAbstractionLayer shutting down")


# 全局单例
_unified_layer: Optional[UnifiedAbstractionLayer] = None
_unified_layer_lock = threading.Lock()


def get_unified_layer(config: Dict[str, Any] = None) -> UnifiedAbstractionLayer:
    """获取统一抽象层单例（线程安全）"""
    global _unified_layer
    
    if _unified_layer is None:
        with _unified_layer_lock:
            if _unified_layer is None:
                _unified_layer = UnifiedAbstractionLayer(config)
    
    return _unified_layer