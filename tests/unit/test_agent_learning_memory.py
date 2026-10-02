from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.agent_learning_memory import AgentLearningMemory
from core.event_store import EventStore
from core.intelligent_agent import DecisionLevel, IntelligentTradingAgent
from core.top_level_agi import TopLevelAGICoordinator
from decision.rl_agent import AgentMode, StateEncoding, TradingRLAgent
from risk.allocation_agent import AllocationAgent


def test_memory_persists_deduplicates_and_filters_context(tmp_path):
    store = EventStore(data_dir=str(tmp_path / "events"))
    memory = AgentLearningMemory(event_store=store)
    entry_id = memory.record(
        "intelligent_agent",
        "signal_decision",
        trace_id="decision-1",
        context={
            "symbol": "BTC-USDT-SWAP",
            "strategy": "trend",
            "regime": "trend_bullish",
            "raw_signal": "must not persist",
        },
        decision="reject",
        outcome={"reason": "regime mismatch"},
        veto=True,
    )
    assert memory.record(
        "intelligent_agent",
        "signal_decision",
        trace_id="decision-1",
        context={"symbol": "BTC-USDT-SWAP", "strategy": "trend"},
        decision="reject",
        veto=True,
    ) == entry_id

    restored = AgentLearningMemory(event_store=store)
    memories = restored.recall(
        {"symbol": "btc-usdt-swap", "strategy": "trend", "regime": "trending_up"}
    )

    assert len(memories) == 1
    assert memories[0]["context"] == {
        "symbol": "btc-usdt-swap",
        "strategy": "trend",
        "regime": "trending_up",
    }
    assert restored.has_veto({
        "symbol": "BTC-USDT-SWAP", "strategy": "trend", "regime": "trending_up"
    })
    assert not restored.has_veto({
        "symbol": "BTC-USDT-SWAP", "strategy": "grid", "regime": "trending_up"
    })


def test_rl_respects_recent_intelligent_agent_veto(tmp_path):
    memory = AgentLearningMemory()
    memory.record(
        "intelligent_agent",
        "signal_decision",
        trace_id="audit-1",
        context={"symbol": "BTC-USDT-SWAP", "strategy": "trend", "regime": "trend_bullish"},
        decision="reject",
        veto=True,
    )
    agent = TradingRLAgent({
        "rl_agent": {
            "name": "memory_test",
            "enabled": True,
            "decision_enabled": True,
            "persist_dir": str(tmp_path),
        }
    })
    agent.set_learning_memory(memory)
    agent.select_action = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("RL must not produce a conflicting action after a shared veto")
    )

    adjustment = agent.get_parameter_adjustment(
        "leverage",
        5.0,
        StateEncoding(
            symbol="BTC-USDT-SWAP",
            strategy_id="trend",
            market_regime="trending_up",
        ),
    )

    assert adjustment == 5.0
    assert memory.get_summary()["by_agent"]["rl_agent"] == 1


def test_rl_episode_outcome_is_shared_with_decision_context(tmp_path):
    memory = AgentLearningMemory()
    agent = TradingRLAgent({
        "rl_agent": {
            "name": "outcome_test",
            "enabled": True,
            "decision_enabled": True,
            "persist_dir": str(tmp_path),
        }
    })
    agent.set_learning_memory(memory)
    state = StateEncoding(
        symbol="ETH-USDT-SWAP",
        strategy_id="grid",
        market_regime="ranging",
        decision_id="trace-episode-1",
    )
    agent.get_parameter_adjustment("leverage", 3.0, state)
    agent.end_episode(0.75)

    outcomes = memory.recall(
        {"symbol": "ETH-USDT-SWAP", "strategy": "grid", "regime": "ranging"},
        agent="rl_agent",
    )
    episode = next(item for item in outcomes if item["kind"] == "episode_outcome")

    assert episode["trace_id"] == "trace-episode-1"
    assert episode["outcome"]["reward"] == 0.75


@pytest.mark.asyncio
async def test_allocation_and_top_level_share_memory(tmp_path):
    memory = AgentLearningMemory()
    agent = AllocationAgent(
        config={}, trade_journal=Mock(), profit_optimizer=Mock(), account_manager=Mock()
    )
    agent.set_learning_memory(memory)
    agent._get_strategy_metrics = lambda strategy: {
        "win_rate": 0.6,
        "profit_factor": 1.2,
        "sharpe_ratio": 0.8,
        "max_drawdown": 0.1,
        "total_pnl": 12.0,
        "trade_count": 8,
    }

    await agent._record_performance()
    top_level = TopLevelAGICoordinator(
        learning_memory=memory,
        config={"cooldown_seconds": 0.0},
    )
    report = top_level.run_cycle()

    assert len(memory.recall({"strategy": "grid"}, agent="allocation_agent")) == 1
    assert report["summary"]["shared_learning_memory"]["by_agent"]["allocation_agent"] == 6


def test_intelligent_agent_publishes_veto_to_shared_memory(tmp_path):
    memory = AgentLearningMemory()
    agent = IntelligentTradingAgent({"data_dir": str(tmp_path)})
    agent.set_learning_memory(memory)
    decision = SimpleNamespace(
        decision_id="audit-veto-1",
        timestamp=datetime.now(),
        action="reject",
        reason="blocked by test gate",
        confidence=0.9,
        source="test_gate",
        level=DecisionLevel.SYMBOL,
        details={"regime": "trend_bullish"},
    )

    agent._record_decision(decision, "BTC-USDT-SWAP", "trend", 0.9)

    assert memory.has_veto({
        "symbol": "BTC-USDT-SWAP", "strategy": "trend", "regime": "trending_up"
    })
