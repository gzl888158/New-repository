"""P3 企业级升级：订单状态机（OrderStateMachine）全量单元测试。

覆盖：
1. 枚举值与关键常量（初始态/终态/活跃态）
2. 迁移表结构完整性（与权威期望表逐项比对）
3. allowed_events / can_apply / next_state / apply 的全部分支
4. 非法迁移 fail-closed（apply 抛 InvalidOrderTransition）
5. 全组合穷举：每个「状态 × 事件」组合的合法性、目标态、异常行为完全一致
6. can_reach / reachable_targets / is_terminal / is_active（含字符串容错）
7. get_effect：19 条合法迁移的副作用声明逐项断言 + 幂等空副作用
8. TransitionEffect.is_empty
9. _coerce / _coerce_event 字符串容错与类型错误
10. OrderLifecycleManager 集成：完整成交生命周期计数器收敛 + 非法迁移拒绝 + 超时收敛
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.order_state_machine import (
    InvalidOrderTransition,
    OrderEvent,
    OrderPhase,
    OrderStateMachine,
    OrderStatus,
    TransitionEffect,
)


# ═══════════════════════════════════════════════════════════════
# 权威期望值（独立于被测实现，防止迁移表被意外改动）
# ═══════════════════════════════════════════════════════════════

EXPECTED_TRANSITIONS = {
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
    OrderStatus.FILLED: {},
    OrderStatus.CANCELLED: {},
    OrderStatus.FAILED: {},
    OrderStatus.TIMEOUT: {},
}

TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.FAILED, OrderStatus.TIMEOUT}
ACTIVE = {OrderStatus.QUEUED, OrderStatus.VALIDATION, OrderStatus.EXECUTING,
          OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED}


def _effect(**kwargs) -> TransitionEffect:
    """构造期望的 TransitionEffect，缺省字段按 dataclass 默认值。"""
    return TransitionEffect(**kwargs)


# 每条合法迁移的期望副作用声明（源, 事件, 目标, effect）
EXPECTED_EFFECTS = [
    # QUEUED
    (OrderStatus.QUEUED, OrderEvent.VALIDATE, _effect()),
    (OrderStatus.QUEUED, OrderEvent.CANCEL,
     _effect(decrement_active=True, stat_counter="total_cancelled", hook="on_cancel")),
    (OrderStatus.QUEUED, OrderEvent.FAIL,
     _effect(decrement_active=True, stat_counter="total_failed", hook="on_fail")),
    # VALIDATION
    (OrderStatus.VALIDATION, OrderEvent.SUBMIT, _effect(increment_executing=True)),
    (OrderStatus.VALIDATION, OrderEvent.CANCEL,
     _effect(decrement_active=True, stat_counter="total_cancelled", hook="on_cancel")),
    (OrderStatus.VALIDATION, OrderEvent.FAIL,
     _effect(decrement_active=True, stat_counter="total_failed", hook="on_fail")),
    # EXECUTING
    (OrderStatus.EXECUTING, OrderEvent.ACCEPT,
     _effect(decrement_executing=True, increment_pending=True)),
    (OrderStatus.EXECUTING, OrderEvent.CANCEL,
     _effect(decrement_active=True, decrement_executing=True, stat_counter="total_cancelled", hook="on_cancel")),
    (OrderStatus.EXECUTING, OrderEvent.FAIL,
     _effect(decrement_active=True, decrement_executing=True, stat_counter="total_failed", hook="on_fail")),
    (OrderStatus.EXECUTING, OrderEvent.TIMEOUT,
     _effect(decrement_active=True, decrement_executing=True, stat_counter="total_timeout", hook="on_timeout")),
    # PENDING
    (OrderStatus.PENDING, OrderEvent.PARTIAL_FILL, _effect()),
    (OrderStatus.PENDING, OrderEvent.FILL,
     _effect(decrement_active=True, decrement_pending=True, stat_counter="total_filled",
             hook="on_fill", record_latency=True)),
    (OrderStatus.PENDING, OrderEvent.CANCEL,
     _effect(decrement_active=True, decrement_pending=True, stat_counter="total_cancelled", hook="on_cancel")),
    (OrderStatus.PENDING, OrderEvent.FAIL,
     _effect(decrement_active=True, decrement_pending=True, stat_counter="total_failed", hook="on_fail")),
    (OrderStatus.PENDING, OrderEvent.TIMEOUT,
     _effect(decrement_active=True, decrement_pending=True, stat_counter="total_timeout", hook="on_timeout")),
    # PARTIALLY_FILLED
    (OrderStatus.PARTIALLY_FILLED, OrderEvent.FILL,
     _effect(decrement_active=True, decrement_pending=True, stat_counter="total_filled",
             hook="on_fill", record_latency=True)),
    (OrderStatus.PARTIALLY_FILLED, OrderEvent.CANCEL,
     _effect(decrement_active=True, decrement_pending=True, stat_counter="total_cancelled", hook="on_cancel")),
    (OrderStatus.PARTIALLY_FILLED, OrderEvent.FAIL,
     _effect(decrement_active=True, decrement_pending=True, stat_counter="total_failed", hook="on_fail")),
    (OrderStatus.PARTIALLY_FILLED, OrderEvent.TIMEOUT,
     _effect(decrement_active=True, decrement_pending=True, stat_counter="total_timeout", hook="on_timeout")),
]


# ═══════════════════════════════════════════════════════════════
# 测试
# ═══════════════════════════════════════════════════════════════

class TestEnumsAndConstants:
    def test_enum_values(self):
        assert OrderStatus.QUEUED.value == "queued"
        assert OrderStatus.PARTIALLY_FILLED.value == "partially_filled"
        assert OrderStatus.TIMEOUT.value == "timeout"
        assert OrderEvent.VALIDATE.value == "validate"
        assert OrderEvent.TIMEOUT.value == "timeout"
        assert OrderPhase.CREATION.value == "creation"

    def test_initial_state(self):
        assert OrderStateMachine.initial_state() == OrderStatus.QUEUED

    def test_terminal_and_active_states(self):
        assert set(OrderStateMachine.terminal_states()) == TERMINAL
        assert set(OrderStateMachine.active_states()) == ACTIVE

    def test_is_terminal_and_is_active(self):
        for s in OrderStatus:
            assert OrderStateMachine.is_terminal(s) == (s in TERMINAL)
            assert OrderStateMachine.is_active(s) == (s in ACTIVE)

    def test_is_terminal_and_is_active_string_coercion(self):
        assert OrderStateMachine.is_terminal("filled") is True
        assert OrderStateMachine.is_terminal("pending") is False
        assert OrderStateMachine.is_active("queued") is True
        assert OrderStateMachine.is_active("timeout") is False


class TestTransitionTable:
    def test_transition_table_matches_authoritative(self):
        assert OrderStateMachine.TRANSITIONS == EXPECTED_TRANSITIONS

    def test_terminal_states_have_no_out_transitions(self):
        for s in TERMINAL:
            assert OrderStateMachine.TRANSITIONS[s] == {}
            assert OrderStateMachine.allowed_events(s) == frozenset()

    def test_allowed_events(self):
        for state, table in EXPECTED_TRANSITIONS.items():
            assert OrderStateMachine.allowed_events(state) == frozenset(table.keys())

    def test_allowed_events_string_coercion(self):
        assert OrderStateMachine.allowed_events("pending") == frozenset({
            OrderEvent.PARTIAL_FILL, OrderEvent.FILL, OrderEvent.CANCEL,
            OrderEvent.FAIL, OrderEvent.TIMEOUT,
        })


class TestValidTransitions:
    def test_next_state_all_valid(self):
        for source, table in EXPECTED_TRANSITIONS.items():
            for event, target in table.items():
                assert OrderStateMachine.next_state(source, event) == target

    def test_apply_all_valid(self):
        for source, table in EXPECTED_TRANSITIONS.items():
            for event, target in table.items():
                assert OrderStateMachine.apply(source, event) == target

    def test_can_apply_all_valid(self):
        for source, table in EXPECTED_TRANSITIONS.items():
            for event in table:
                assert OrderStateMachine.can_apply(source, event) is True

    def test_apply_string_coercion(self):
        assert OrderStateMachine.apply("queued", "validate") == OrderStatus.VALIDATION
        assert OrderStateMachine.next_state("pending", "fill") == OrderStatus.FILLED
        assert OrderStateMachine.can_apply("executing", "accept") is True


class TestInvalidTransitions:
    def test_next_state_returns_none_for_invalid(self):
        # 终态不可迁移
        for terminal in TERMINAL:
            for event in OrderEvent:
                assert OrderStateMachine.next_state(terminal, event) is None
        # 非法「状态 × 事件」组合
        assert OrderStateMachine.next_state(OrderStatus.QUEUED, OrderEvent.SUBMIT) is None
        assert OrderStateMachine.next_state(OrderStatus.QUEUED, OrderEvent.ACCEPT) is None
        assert OrderStateMachine.next_state(OrderStatus.VALIDATION, OrderEvent.PARTIAL_FILL) is None
        assert OrderStateMachine.next_state(OrderStatus.PENDING, OrderEvent.SUBMIT) is None

    def test_can_apply_false_for_invalid(self):
        assert OrderStateMachine.can_apply(OrderStatus.FILLED, OrderEvent.CANCEL) is False
        assert OrderStateMachine.can_apply(OrderStatus.QUEUED, OrderEvent.FILL) is False
        assert OrderStateMachine.can_apply(OrderStatus.VALIDATION, OrderEvent.ACCEPT) is False

    def test_apply_raises_on_invalid(self):
        with pytest.raises(InvalidOrderTransition):
            OrderStateMachine.apply(OrderStatus.FILLED, OrderEvent.CANCEL)
        with pytest.raises(InvalidOrderTransition):
            OrderStateMachine.apply(OrderStatus.QUEUED, OrderEvent.FILL)

    def test_apply_raises_contains_context(self):
        with pytest.raises(InvalidOrderTransition) as exc:
            OrderStateMachine.apply(OrderStatus.QUEUED, OrderEvent.FILL)
        msg = str(exc.value)
        assert "queued" in msg and "fill" in msg


class TestExhaustiveConsistency:
    """穷举所有「状态 × 事件」组合，验证三组 API 行为完全一致。"""

    def test_exhaustive(self):
        for source in OrderStatus:
            for event in OrderEvent:
                expected = EXPECTED_TRANSITIONS[source].get(event)
                assert OrderStateMachine.can_apply(source, event) == (expected is not None)
                assert OrderStateMachine.next_state(source, event) == expected
                if expected is not None:
                    assert OrderStateMachine.apply(source, event) == expected
                else:
                    with pytest.raises(InvalidOrderTransition):
                        OrderStateMachine.apply(source, event)


class TestCanReachAndReachableTargets:
    def test_can_reach_direct(self):
        assert OrderStateMachine.can_reach(OrderStatus.QUEUED, OrderStatus.VALIDATION) is True
        assert OrderStateMachine.can_reach(OrderStatus.PENDING, OrderStatus.FILLED) is True
        assert OrderStateMachine.can_reach(OrderStatus.EXECUTING, OrderStatus.PENDING) is True

    def test_can_reach_idempotent(self):
        for s in OrderStatus:
            assert OrderStateMachine.can_reach(s, s) is True

    def test_can_reach_invalid(self):
        assert OrderStateMachine.can_reach(OrderStatus.QUEUED, OrderStatus.EXECUTING) is False
        assert OrderStateMachine.can_reach(OrderStatus.FILLED, OrderStatus.CANCELLED) is False
        assert OrderStateMachine.can_reach(OrderStatus.PENDING, OrderStatus.VALIDATION) is False

    def test_can_reach_string_coercion(self):
        assert OrderStateMachine.can_reach("queued", "validation") is True
        assert OrderStateMachine.can_reach("filled", "cancelled") is False

    def test_reachable_targets(self):
        assert OrderStateMachine.reachable_targets(OrderStatus.QUEUED) == frozenset({
            OrderStatus.VALIDATION, OrderStatus.CANCELLED, OrderStatus.FAILED,
        })
        assert OrderStateMachine.reachable_targets(OrderStatus.FILLED) == frozenset()


class TestGetEffect:
    def test_effects_for_all_valid_transitions(self):
        for source, event, expected in EXPECTED_EFFECTS:
            target = EXPECTED_TRANSITIONS[source][event]
            got = OrderStateMachine.get_effect(source, target)
            assert got == expected, f"{source.value} -({event.value})-> {target.value}"

    def test_effect_string_coercion(self):
        got = OrderStateMachine.get_effect("pending", "filled")
        assert got == _effect(decrement_active=True, decrement_pending=True,
                              stat_counter="total_filled", hook="on_fill", record_latency=True)

    def test_effect_idempotent_is_empty(self):
        for s in OrderStatus:
            effect = OrderStateMachine.get_effect(s, s)
            assert effect.is_empty is True
            assert effect == TransitionEffect()

    def test_is_empty_property(self):
        assert TransitionEffect().is_empty is True
        assert TransitionEffect(decrement_active=True).is_empty is False
        assert TransitionEffect(stat_counter="total_filled").is_empty is False
        assert TransitionEffect(hook="on_fill").is_empty is False
        assert TransitionEffect(record_latency=True).is_empty is False


class TestCoerceErrors:
    def test_coerce_raises_type_error_on_int(self):
        with pytest.raises(TypeError):
            OrderStateMachine.can_apply(123, OrderEvent.FILL)
        with pytest.raises(TypeError):
            OrderStateMachine.next_state(OrderStatus.PENDING, 123)

    def test_coerce_raises_on_unknown_string(self):
        with pytest.raises(ValueError):
            OrderStateMachine.can_apply("nonexistent", OrderEvent.FILL)


# ═══════════════════════════════════════════════════════════════
# OrderLifecycleManager 集成测试（状态机作为唯一流转裁决入口）
# ═══════════════════════════════════════════════════════════════

def _make_manager():
    from execution.order_lifecycle_manager import OrderLifecycleManager
    return OrderLifecycleManager({"execution": {"timeout": {"order_timeout": 60, "tracking_timeout": 300}}})


class TestLifecycleIntegration:
    def test_full_filled_lifecycle_converges_counters(self):
        async def run():
            mgr = _make_manager()
            order_id = await mgr.create_order({"order_id": "ord_1"})
            assert mgr.get_execution_summary()["active"] == 1

            await mgr.update_status(order_id, OrderStatus.VALIDATION, OrderPhase.VALIDATION)
            await mgr.update_status(order_id, OrderStatus.EXECUTING, OrderPhase.PLACEMENT)
            assert mgr.get_monitor_status()["executing_orders"] == 1

            await mgr.update_status(order_id, OrderStatus.PENDING, OrderPhase.TRACKING)
            assert mgr.get_monitor_status()["executing_orders"] == 0
            assert mgr.get_monitor_status()["pending_orders"] == 1

            await mgr.update_status(order_id, OrderStatus.FILLED, OrderPhase.SETTLEMENT)
            summary = mgr.get_execution_summary()
            assert summary["active"] == 0
            assert summary["pending"] == 0
            assert summary["executing"] == 0
            assert summary["filled"] == 1
            assert mgr.get_order_status(order_id)["status"] == "filled"

            history = mgr.get_order_state_history(order_id)
            assert [h["to"] for h in history] == ["validation", "executing", "pending", "filled"]

        asyncio.run(run())

    def test_apply_event_drives_transition(self):
        async def run():
            mgr = _make_manager()
            order_id = await mgr.create_order({"order_id": "ord_2"})
            await mgr.apply_event(order_id, OrderEvent.VALIDATE)
            await mgr.apply_event(order_id, OrderEvent.SUBMIT)
            assert mgr.get_order_status(order_id)["status"] == "executing"
            await mgr.apply_event(order_id, OrderEvent.ACCEPT)
            assert mgr.get_order_status(order_id)["status"] == "pending"
            await mgr.apply_event(order_id, OrderEvent.FILL)
            assert mgr.get_execution_summary()["filled"] == 1

        asyncio.run(run())

    def test_illegal_transition_rejected_and_counted(self):
        async def run():
            mgr = _make_manager()
            order_id = await mgr.create_order({"order_id": "ord_3"})
            # QUEUED 不可直接 FILL
            await mgr.update_status(order_id, OrderStatus.FILLED)
            assert mgr.get_order_status(order_id)["status"] == "queued"
            assert mgr.get_execution_stats()["illegal_transitions"] == 1

        asyncio.run(run())

    def test_illegal_event_rejected_and_counted(self):
        async def run():
            mgr = _make_manager()
            order_id = await mgr.create_order({"order_id": "ord_4"})
            # QUEUED + FILL 非法
            await mgr.apply_event(order_id, OrderEvent.FILL)
            assert mgr.get_order_status(order_id)["status"] == "queued"
            assert mgr.get_execution_stats()["illegal_transitions"] == 1

        asyncio.run(run())

    def test_timeout_converged_through_state_machine(self):
        async def run():
            mgr = _make_manager()
            order_id = await mgr.create_order({"order_id": "ord_5"})
            await mgr.update_status(order_id, OrderStatus.VALIDATION)
            await mgr.update_status(order_id, OrderStatus.EXECUTING)
            assert mgr.get_monitor_status()["executing_orders"] == 1

            # 将下单时间回拨，触发超时
            mgr._orders[order_id]["timestamps"]["created"] = datetime.now() - timedelta(seconds=120)
            await mgr._check_timeouts()

            # 状态机裁决为 TIMEOUT，副作用收敛：active/executing 归零、total_timeout +1
            assert mgr.get_execution_summary()["active"] == 0
            assert mgr.get_execution_summary()["executing"] == 0
            assert mgr.get_execution_summary()["pending"] == 0
            assert mgr.get_execution_summary()["timeout"] == 1

        asyncio.run(run())

    def test_self_transition_idempotent_no_double_count(self):
        async def run():
            mgr = _make_manager()
            order_id = await mgr.create_order({"order_id": "ord_6"})
            await mgr.update_status(order_id, OrderStatus.VALIDATION)
            await mgr.update_status(order_id, OrderStatus.EXECUTING)
            # 重复置为 EXECUTING（幂等），executing 计数不得二次递增
            await mgr.update_status(order_id, OrderStatus.EXECUTING)
            assert mgr.get_monitor_status()["executing_orders"] == 1

        asyncio.run(run())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])