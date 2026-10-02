from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from risk.allocation_agent import AllocationAgent


def _agent(method="dynamic", rebalance_enabled=False):
    return AllocationAgent(
        config={"allocation_agent": {
            "method": method,
            "rebalance_enabled": rebalance_enabled,
        }},
        trade_journal=Mock(),
        profit_optimizer=Mock(),
        account_manager=Mock(),
    )


@pytest.mark.asyncio
async def test_dynamic_failure_returns_empty_recommendation_fail_closed():
    agent = _agent()
    agent._strategy_weights = {name: 0.0 for name in agent._strategy_names}
    agent._strategy_weights.update({"grid": 0.7, "trend": 0.3})
    agent._dynamic_allocator = None

    weights = await agent._allocate_dynamic()

    # fail-closed: 异常时不返回陈旧权重，返回空建议
    assert weights == {}


@pytest.mark.asyncio
async def test_dynamic_allocator_exception_returns_empty_recommendation_fail_closed():
    agent = _agent()
    agent.trade_journal.get_trades_by_strategy.return_value = []
    agent.account_manager.get_total_equity.return_value = 1000.0
    agent.account_manager.get_total_capital.return_value = 1000.0
    agent._strategy_weights = {name: 0.0 for name in agent._strategy_names}
    agent._strategy_weights.update({"grid": 0.7, "trend": 0.3})

    class BrokenAllocator:
        async def compute_allocation_plan(self, **kwargs):
            raise RuntimeError("allocator unavailable")

    agent._dynamic_allocator = BrokenAllocator()
    weights = await agent._allocate_dynamic()

    # fail-closed: 分配器异常时不返回陈旧权重，返回空建议
    assert weights == {}


@pytest.mark.asyncio
async def test_legacy_allocation_modes_use_portfolio_optimizer():
    agent = _agent(method="risk_adjusted")
    agent._strategy_weights = {name: 0.0 for name in agent._strategy_names}
    agent._strategy_weights.update({"grid": 0.7, "trend": 0.3})
    agent.set_portfolio_optimizer(SimpleNamespace(
        optimize=lambda: SimpleNamespace(optimal_weights={"grid": 0.2, "trend": 0.8})
    ))

    weights = await agent._calculate_optimal_allocation()

    assert weights["grid"] == pytest.approx(0.2)
    assert weights["trend"] == pytest.approx(0.8)


@pytest.mark.parametrize("legacy_flag", [False, True])
@pytest.mark.asyncio
async def test_rebalance_loop_is_disabled(legacy_flag):
    agent = _agent(rebalance_enabled=legacy_flag)
    await agent.start()

    try:
        assert agent._rebalance_enabled is False
        assert len(agent._tasks) == 1  # performance observation only
    finally:
        await agent.stop()


@pytest.mark.asyncio
async def test_rebalance_is_noop_in_observer_mode():
    agent = _agent()
    agent._strategy_weights = {"grid": 0.7, "trend": 0.3}
    before = dict(agent._strategy_weights)

    await agent._rebalance()

    # observer 模式下 rebalance 不修改权重
    assert agent._strategy_weights == before


@pytest.mark.asyncio
async def test_apply_allocation_changes_is_noop_in_observer_mode():
    agent = _agent()
    agent._strategy_weights = {"grid": 0.7, "trend": 0.3}

    changes = await agent._apply_allocation_changes({"grid": 0.2, "trend": 0.8})

    # observer 模式下不应用任何变更
    assert changes == {}
    assert agent._strategy_weights == {"grid": 0.7, "trend": 0.3}


@pytest.mark.asyncio
async def test_manual_rebalance_is_noop_in_observer_mode():
    agent = _agent()
    agent._strategy_weights = {"grid": 0.7, "trend": 0.3}
    before = dict(agent._strategy_weights)

    await agent.manual_rebalance()

    # observer 模式下手动再平衡被禁用，权重不变
    assert agent._strategy_weights == before


@pytest.mark.asyncio
async def test_recommendations_use_live_adaptive_controller_weights():
    """建议的 current 基准必须来自 AdaptiveController 实时权重，而非陈旧 config 快照。"""
    agent = _agent(method="dynamic")
    # 陈旧快照（不应被用于 diff）
    agent._strategy_weights = {"grid": 0.7, "trend": 0.3}
    # 注入一个返回实时权重的 adaptive_controller
    live_ac = Mock()
    live_ac.get_allocations.return_value = {"grid": 0.2, "trend": 0.8}
    agent.set_adaptive_controller(live_ac)
    # 模拟动态分配器返回建议权重
    class FakeAllocator:
        async def compute_allocation_plan(self, **kwargs):
            from risk.dynamic_allocator import AllocationPlan, StrategyAllocation
            return AllocationPlan(
                strategy_allocations={
                    "grid": StrategyAllocation(name="grid", target_weight=0.5),
                    "trend": StrategyAllocation(name="trend", target_weight=0.5),
                },
                capital_efficiency=1.0, idle_cash=0.0, concentration_ratio=0.5,
                warnings=[], recommendations=[], pools={}, pool_flows={},
            )
        def update_strategy_pnl(self, name, pnl):
            pass
        def set_portfolio_optimizer(self, opt):
            pass
    agent.trade_journal.get_trades_by_strategy.return_value = []
    agent.account_manager.get_total_equity.return_value = 1000.0
    agent.account_manager.get_total_capital.return_value = 1000.0
    agent.set_dynamic_allocator(FakeAllocator())

    recs = await agent.get_recommendations()

    # current 必须等于 AdaptiveController 的实时权重，而非陈旧快照
    assert recs["grid"]["current"] == pytest.approx(0.2)
    assert recs["trend"]["current"] == pytest.approx(0.8)


def test_get_current_allocation_uses_live_weights():
    """get_current_allocation 必须返回 AdaptiveController 实时权重。"""
    agent = _agent()
    agent._strategy_weights = {"grid": 0.9, "trend": 0.1}  # 陈旧快照
    live_ac = Mock()
    live_ac.get_allocations.return_value = {"grid": 0.3, "trend": 0.7}
    agent.set_adaptive_controller(live_ac)

    alloc = agent.get_current_allocation()

    assert alloc == {"grid": 0.3, "trend": 0.7}