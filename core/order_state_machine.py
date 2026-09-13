"""P3 企业级升级：订单状态机（OrderStateMachine）

对标华尔街交易系统的「订单生命周期状态机」范式：
- 所有订单状态流转（创建→校验→提交→挂单→部分成交→成交/撤销/失败/超时）由唯一状态机裁决
- 状态机是纯规则层：封装「状态定义 + 事件校验 + 迁移合法性 + 终态判定 + 副作用声明」，零业务副作用，
  可独立单元测试，覆盖全部分支
- 事件驱动：每个合法迁移由「当前状态 + 触发事件 → 目标状态」唯一确定，非法的「状态×事件」组合被硬拒绝
- 副作用声明：将分散在 OrderLifecycleManager 里的计数器增减/事件钩子逻辑收敛为 TransitionEffect，
  由状态机根据「源状态→目标状态」推导，管理器只负责执行

关系：OrderLifecycleManager 保存订单状态存储 + 执行副作用；OrderStateMachine 裁决所有流转合法性。
本模块零执行层依赖（不依赖 okx/execution），可直接单测。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple


class OrderStatus(Enum):
    """订单状态（订单生命周期状态机状态集合）"""
    QUEUED = "queued"                        # 已入队，待校验
    VALIDATION = "validation"                # 校验中
    EXECUTING = "executing"                  # 执行中（下单请求已发出，待受理）
    PENDING = "pending"                      # 已挂单（交易所已受理，待成交）
    PARTIALLY_FILLED = "partially_filled"    # 部分成交（剩余仍在挂单）
    FILLED = "filled"                        # 完全成交（终态）
    CANCELLED = "cancelled"                  # 已撤销（终态）
    FAILED = "failed"                        # 失败（终态）
    TIMEOUT = "timeout"                      # 超时（终态）


class OrderPhase(Enum):
    """订单执行阶段（与状态机状态正交的流程阶段，用于阶段耗时统计）"""
    CREATION = "creation"
    VALIDATION = "validation"
    PLACEMENT = "placement"
    TRACKING = "tracking"
    SETTLEMENT = "settlement"


class OrderEvent(Enum):
    """订单事件（驱动状态迁移的触发条件，收敛事件校验）"""
    VALIDATE = "validate"                    # 进入校验
    SUBMIT = "submit"                        # 提交下单请求
    ACCEPT = "accept"                        # 交易所受理（生成挂单）
    PARTIAL_FILL = "partial_fill"            # 部分成交
    FILL = "fill"                            # 完全成交
    CANCEL = "cancel"                        # 撤销
    FAIL = "fail"                            # 失败
    TIMEOUT = "timeout"                      # 超时


class InvalidOrderTransition(Exception):
    """非法状态迁移异常（fail-closed：非法流转必须显式抛出，而非静默忽略）"""


@dataclass(frozen=True)
class TransitionEffect:
    """状态迁移副作用声明（收敛分散的计数器增减 / 事件钩子逻辑）。

    由状态机根据「源状态 → 目标状态」推导，OrderLifecycleManager 只负责执行：
      - decrement_active    : 进入终态时 active_orders -= 1
      - decrement_pending   : 从 PENDING 迁到终态时 pending_orders -= 1
      - decrement_executing : 从 EXECUTING 迁到终态时 executing_orders -= 1
      - increment_pending   : 进入 PENDING 时 pending_orders += 1
      - increment_executing : 进入 EXECUTING 时 executing_orders += 1
      - stat_counter        : 进入终态时累计统计计数键（如 "total_filled"）
      - hook                : 进入终态时触发的事件钩子键（如 "on_fill"）
      - record_latency      : 成功成交（FILLED）时记录端到端延迟
    """
    decrement_active: bool = False
    decrement_pending: bool = False
    decrement_executing: bool = False
    increment_pending: bool = False
    increment_executing: bool = False
    stat_counter: Optional[str] = None
    hook: Optional[str] = None
    record_latency: bool = False

    @property
    def is_empty(self) -> bool:
        return not (
            self.decrement_active or self.decrement_pending
            or self.decrement_executing or self.increment_pending
            or self.increment_executing or self.stat_counter or self.hook
            or self.record_latency
        )


class OrderStateMachine:
    """订单状态机：唯一的状态流转裁决入口（choke point）。

    用法：
        machine = OrderStateMachine()
        if not machine.can_apply(current, OrderEvent.FILL):
            return  # 非法流转，拒绝
        new_state = machine.apply(current, OrderEvent.FILL)
    """

    # 迁移表：状态 × 事件 → 目标状态（唯一权威流转规则来源）
    TRANSITIONS: Dict[OrderStatus, Dict[OrderEvent, OrderStatus]] = {
        OrderStatus.QUEUED: {
            OrderEvent.VALIDATE: OrderStatus.VALIDATION,
            OrderEvent.CANCEL: OrderStatus.CANCELLED,
            OrderEvent.FAIL: OrderStatus.FAILED,
        },
        OrderStatus.VALIDATION: {
            OrderEvent.SUBMIT: OrderStatus.EXECUTING,
            OrderEvent.CANCEL: OrderStatus.CANCELLED,
            OrderEvent.FAIL: OrderStatus.FAILED,
        },
        OrderStatus.EXECUTING: {
            OrderEvent.ACCEPT: OrderStatus.PENDING,
            OrderEvent.CANCEL: OrderStatus.CANCELLED,
            OrderEvent.FAIL: OrderStatus.FAILED,
            OrderEvent.TIMEOUT: OrderStatus.TIMEOUT,
        },
        OrderStatus.PENDING: {
            OrderEvent.PARTIAL_FILL: OrderStatus.PARTIALLY_FILLED,
            OrderEvent.FILL: OrderStatus.FILLED,
            OrderEvent.CANCEL: OrderStatus.CANCELLED,
            OrderEvent.FAIL: OrderStatus.FAILED,
            OrderEvent.TIMEOUT: OrderStatus.TIMEOUT,
        },
        OrderStatus.PARTIALLY_FILLED: {
            OrderEvent.FILL: OrderStatus.FILLED,
            OrderEvent.CANCEL: OrderStatus.CANCELLED,
            OrderEvent.FAIL: OrderStatus.FAILED,
            OrderEvent.TIMEOUT: OrderStatus.TIMEOUT,
        },
        # 终态：禁止任何迁移
        OrderStatus.FILLED: {},
        OrderStatus.CANCELLED: {},
        OrderStatus.FAILED: {},
        OrderStatus.TIMEOUT: {},
    }

    # 终态集合：进入后不可再迁移
    TERMINAL_STATES: FrozenSet[OrderStatus] = frozenset({
        OrderStatus.FILLED,
        OrderStatus.CANCELLED,
        OrderStatus.FAILED,
        OrderStatus.TIMEOUT,
    })

    # 活跃（非终态）状态集合
    ACTIVE_STATES: FrozenSet[OrderStatus] = frozenset({
        OrderStatus.QUEUED,
        OrderStatus.VALIDATION,
        OrderStatus.EXECUTING,
        OrderStatus.PENDING,
        OrderStatus.PARTIALLY_FILLED,
    })

    # 终态 → 统计计数键
    _STAT_COUNTER: Dict[OrderStatus, str] = {
        OrderStatus.FILLED: "total_filled",
        OrderStatus.CANCELLED: "total_cancelled",
        OrderStatus.FAILED: "total_failed",
        OrderStatus.TIMEOUT: "total_timeout",
    }

    # 终态 → 事件钩子键
    _STAT_HOOK: Dict[OrderStatus, str] = {
        OrderStatus.FILLED: "on_fill",
        OrderStatus.CANCELLED: "on_cancel",
        OrderStatus.FAILED: "on_fail",
        OrderStatus.TIMEOUT: "on_timeout",
    }

    @classmethod
    def initial_state(cls) -> OrderStatus:
        """订单初始状态。"""
        return OrderStatus.QUEUED

    @classmethod
    def terminal_states(cls) -> FrozenSet[OrderStatus]:
        return cls.TERMINAL_STATES

    @classmethod
    def active_states(cls) -> FrozenSet[OrderStatus]:
        return cls.ACTIVE_STATES

    @classmethod
    def is_terminal(cls, status: OrderStatus) -> bool:
        """是否为终态（不可再迁移）。"""
        return _coerce(status) in cls.TERMINAL_STATES

    @classmethod
    def is_active(cls, status: OrderStatus) -> bool:
        """是否为活跃（非终态）状态。"""
        return _coerce(status) in cls.ACTIVE_STATES

    @classmethod
    def allowed_events(cls, current: OrderStatus) -> FrozenSet[OrderEvent]:
        """当前状态允许触发的事件集合（收敛事件校验）。"""
        current = _coerce(current)
        return frozenset(cls.TRANSITIONS.get(current, {}).keys())

    @classmethod
    def can_apply(cls, current: OrderStatus, event: OrderEvent) -> bool:
        """校验「当前状态 × 事件」组合是否合法。"""
        current = _coerce(current)
        event = _coerce_event(event)
        return event in cls.TRANSITIONS.get(current, {})

    @classmethod
    def next_state(cls, current: OrderStatus, event: OrderEvent) -> Optional[OrderStatus]:
        """返回迁移后的目标状态；非法组合返回 None（不抛异常）。"""
        current = _coerce(current)
        event = _coerce_event(event)
        return cls.TRANSITIONS.get(current, {}).get(event)

    @classmethod
    def apply(cls, current: OrderStatus, event: OrderEvent) -> OrderStatus:
        """执行迁移并返回目标状态；非法组合抛 InvalidOrderTransition（fail-closed）。"""
        target = cls.next_state(current, event)
        if target is None:
            raise InvalidOrderTransition(
                f"Illegal order transition: {_coerce(current).value} "
                f"-({_coerce_event(event).value})-> (rejected)"
            )
        return target

    @classmethod
    def can_reach(cls, current: OrderStatus, target: OrderStatus) -> bool:
        """校验「源状态 → 目标状态」直接迁移是否合法（兼容目标态 API）。"""
        current = _coerce(current)
        target = _coerce(target)
        if current == target:
            return True  # 幂等：同状态迁移视为合法
        return target in cls.TRANSITIONS.get(current, {}).values()

    @classmethod
    def reachable_targets(cls, current: OrderStatus) -> FrozenSet[OrderStatus]:
        """当前状态一步可达的目标状态集合。"""
        current = _coerce(current)
        return frozenset(cls.TRANSITIONS.get(current, {}).values())

    @classmethod
    def get_effect(cls, source: OrderStatus, target: OrderStatus) -> TransitionEffect:
        """推导「源状态 → 目标状态」迁移的副作用声明（收敛计数器/钩子逻辑）。"""
        source = _coerce(source)
        target = _coerce(target)

        # 幂等：源状态 == 目标状态（无实际迁移）不产生任何副作用
        if source == target:
            return TransitionEffect()

        is_terminal = target in cls.TERMINAL_STATES
        # 递减语义（收敛原分散计数逻辑，避免计数泄漏）：
        #   - 离开 pending-like（PENDING/PARTIALLY_FILLED）时递减 pending 计数
        #   - 离开 executing（EXECUTING）时递减 executing 计数（含 EXECUTING→PENDING/PARTIALLY_FILLED 等非终态迁移）
        leaving_pending = source in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED) and (
            target not in (OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED)
        )
        leaving_executing = source == OrderStatus.EXECUTING

        return TransitionEffect(
            decrement_active=is_terminal,
            decrement_pending=leaving_pending,
            decrement_executing=leaving_executing,
            increment_pending=(target == OrderStatus.PENDING),
            increment_executing=(target == OrderStatus.EXECUTING),
            stat_counter=cls._STAT_COUNTER.get(target),
            hook=cls._STAT_HOOK.get(target),
            record_latency=(target == OrderStatus.FILLED),
        )


def _coerce(status: OrderStatus) -> OrderStatus:
    """容错：接受字符串（.value）并归一化为 OrderStatus。"""
    if isinstance(status, OrderStatus):
        return status
    if isinstance(status, str):
        return OrderStatus(status)
    raise TypeError(f"Expected OrderStatus or str, got {type(status).__name__}")


def _coerce_event(event: OrderEvent) -> OrderEvent:
    """容错：接受字符串（.value）并归一化为 OrderEvent。"""
    if isinstance(event, OrderEvent):
        return event
    if isinstance(event, str):
        return OrderEvent(event)
    raise TypeError(f"Expected OrderEvent or str, got {type(event).__name__}")


__all__ = [
    "OrderStatus",
    "OrderPhase",
    "OrderEvent",
    "OrderStateMachine",
    "TransitionEffect",
    "InvalidOrderTransition",
]