"""
P22: 全链路事件ID体系 - Event ID System
========================================
生产级要求：每条日志带唯一事件ID，事后可完整复现事故全过程。

特性：
- 线程安全的事件ID生成器
- loguru 集成：自动注入事件ID到每条日志
- 事件上下文管理器：关联同一业务操作的所有日志
- 事件ID格式：evt-{timestamp}-{counter}，如 evt-17230001-00042
- 支持嵌套事件上下文（父子事件关联）
"""

import threading
import time
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any
from loguru import logger


# ── ContextVar: 协程安全的当前事件ID ──
_current_event_id: ContextVar[Optional[str]] = ContextVar("current_event_id", default=None)
_current_parent_event_id: ContextVar[Optional[str]] = ContextVar("current_parent_event_id", default=None)


class EventIDGenerator:
    """P22: 线程安全的事件ID生成器
    
    格式: evt-{base36_timestamp}-{counter:05d}
    示例: evt-1a2b3c-00001
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __init__(self):
        self._counter = 0
        self._counter_lock = threading.Lock()
    
    @classmethod
    def get_instance(cls) -> "EventIDGenerator":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def generate(self) -> str:
        """生成唯一事件ID"""
        with self._counter_lock:
            ts = int(time.time() * 1000)
            # 使用 base36 压缩时间戳，减少ID长度
            ts_compact = _base36_encode(ts)
            self._counter += 1
            if self._counter > 99999:
                self._counter = 1
            return f"evt-{ts_compact}-{self._counter:05d}"
    
    def generate_batch(self, count: int) -> list:
        """批量生成事件ID"""
        return [self.generate() for _ in range(count)]


def _base36_encode(num: int) -> str:
    """Base36编码（0-9a-z）"""
    chars = "0123456789abcdefghijklmnopqrstuvwxyz"
    if num == 0:
        return "0"
    result = []
    while num > 0:
        result.append(chars[num % 36])
        num //= 36
    return "".join(reversed(result))


def get_current_event_id() -> Optional[str]:
    """获取当前协程/线程的事件ID"""
    return _current_event_id.get()


def set_current_event_id(event_id: str) -> None:
    """设置当前事件ID"""
    _current_event_id.set(event_id)


def clear_current_event_id() -> None:
    """清除当前事件ID"""
    _current_event_id.set(None)


def event_id_filter(record: dict) -> bool:
    """P22: loguru filter - 自动注入事件ID到日志记录
    
    在 main.py 中配置：
        logger.configure(patcher=inject_event_id)
    """
    inject_event_id(record)
    return True


def inject_event_id(record: dict) -> None:
    """P22: loguru patcher - 在每条日志中注入事件ID"""
    eid = _current_event_id.get()
    # 防御：loguru 的 record["extra"] 可能缺失，使用 setdefault 避免 KeyError
    record.setdefault("extra", {})["event_id"] = eid if eid else "-"


class EventContext:
    """P22: 事件上下文管理器 - 关联同一业务操作的所有日志
    
    使用示例：
        with EventContext("place_order_BTC") as evt:
            logger.info("Starting order placement")
            # 此上下文内所有日志都带相同事件ID
            await place_order()
            logger.info("Order placed successfully")
            # 可通过 evt.id 获取事件ID
    
    嵌套事件支持：
        with EventContext("parent_event") as parent:
            logger.info("Parent log")
            with EventContext("child_event", parent_id=parent.id):
                logger.info("Child log with parent reference")
    """
    
    def __init__(self, description: str = "", parent_id: str = None, 
                 metadata: Dict[str, Any] = None, event_id: str = None):
        self._generator = EventIDGenerator.get_instance()
        # P2: 支持复用上游已有事件ID（全链路 traceID 串联），缺省时生成新ID（向后兼容）
        self.id = event_id or self._generator.generate()
        self.description = description
        self.parent_id = parent_id
        self.metadata = metadata or {}
        self._previous_id: Optional[str] = None
        self._previous_parent: Optional[str] = None
        self._start_time: float = 0.0
    
    def __enter__(self) -> "EventContext":
        self._previous_id = _current_event_id.get()
        self._previous_parent = _current_parent_event_id.get()
        _current_event_id.set(self.id)
        if self.parent_id:
            _current_parent_event_id.set(self.parent_id)
        elif self._previous_id:
            _current_parent_event_id.set(self._previous_id)
        self._start_time = time.time()
        
        if self.description:
            parent_info = f" (parent={self.parent_id})" if self.parent_id else ""
            logger.debug(f"EVENT_START: [{self.id}] {self.description}{parent_info}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            elapsed_ms = (time.time() - self._start_time) * 1000
            if exc_type:
                logger.error(
                    f"EVENT_ERROR: [{self.id}] {self.description} "
                    f"failed after {elapsed_ms:.0f}ms: {exc_val}"
                )
            elif self.description:
                logger.debug(
                    f"EVENT_END: [{self.id}] {self.description} "
                    f"completed in {elapsed_ms:.0f}ms"
                )
        finally:
            # 始终恢复上下文，避免日志 sink 异常导致 ContextVar 泄漏、污染后续日志
            _current_event_id.set(self._previous_id)
            _current_parent_event_id.set(self._previous_parent)
    
    async def __aenter__(self) -> "EventContext":
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


def event_id() -> str:
    """P22: 快速生成事件ID（不创建上下文）
    
    使用: evt_id = event_id()
    """
    return EventIDGenerator.get_instance().generate()


def configure_event_id_logging():
    """P22: 配置 loguru 单一日志门面：注入事件ID + 全局脱敏（模块7/9）。

    统一 patcher 同时完成：
      1. inject_event_id      —— 每条日志注入当前事件ID
      2. redact_log_record    —— 密钥/授权/clOrdId 精确与正则脱敏（唯一脱敏源）

    在 main.py / scripts 中调用此函数即可启用全链路事件ID + 密钥零泄漏脱敏。
    loguru 的 logger.configure(patcher=...) 会覆盖上一次配置，因此这里必须把
    所有 patcher 逻辑合成一个函数，避免后调覆盖先调导致脱敏失效。
    """
    from core.log_redactor import redact_log_record

    def _composed_patcher(record):
        inject_event_id(record)
        redact_log_record(record)

    logger.configure(patcher=_composed_patcher)
    logger.info("P22: Event ID logging configured - all logs will carry unique event IDs (with sensitive masking)")


__all__ = [
    "EventIDGenerator", "EventContext", "event_id",
    "get_current_event_id", "set_current_event_id", "clear_current_event_id",
    "configure_event_id_logging", "inject_event_id",
]