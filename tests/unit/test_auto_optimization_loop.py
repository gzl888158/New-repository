"""
AutoOptimizationCoordinator 单元测试
======================================
覆盖：依赖缺失降级、上线成功、各门控拒绝、冷却幂等、统计、JSON 安全。
"""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from core.auto_optimization_loop import AutoOptimizationCoordinator, OptimizationCycleResult


def _opt_result(phase="complete", fitness=1.0, errors=None, best_params=None):
    return SimpleNamespace(
        phase=SimpleNamespace(value=phase),
        final_fitness=fitness,
        best_params=best_params or {"atr_period": 14},
        errors=errors or [],
        total_evaluations=100,
    )


class FakeParamOptimizer:
    def __init__(self, result):
        self._result = result
        self._history = [1, 2]

    def set_fitness_fn(self, fn):
        pass

    def set_param_defs(self, p):
        pass

    def set_price_data(self, d):
        pass

    async def optimize(self, strategy_name):
        return self._result


class FakeOptimizer:
    def __init__(self, applied=3):
        self._applied = applied

    async def apply_optimizations(self, recommendation):
        return {"total_applied": self._applied}

    async def persist_config(self):
        return True


class FakePerformanceFeedback:
    async def get_score(self, strategy, window="30d"):
        return {"score": 0.85}

    async def get_feedback(self, strategy):
        return {"recommendations": [{"id": "r1", "action": "tighten_sl"}]}

    def get_summary(self):
        return {"system": "PerformanceFeedback", "strategy_count": 1}


def _builder(strategy, result):
    return {"strategy": strategy, "params": result.best_params}


def _coordinator(param_optimizer=None, optimizer=None, performance_feedback=None,
                 recommendation_builder=_builder, cooldown=3600.0):
    return AutoOptimizationCoordinator(
        param_optimizer=param_optimizer,
        optimizer=optimizer,
        performance_feedback=performance_feedback,
        recommendation_builder=recommendation_builder,
        config={"cooldown_seconds": cooldown},
    )


def _assert_json_safe(obj):
    text = json.dumps(obj, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text
    return text


async def test_all_dependencies_none_degraded():
    orch = _coordinator()
    result = await orch.run_cycle("grid")
    assert result.status == "degraded"
    assert result.message == "no param optimizer"
    _assert_json_safe(asdict(result))


async def test_deploy_success():
    orch = _coordinator(
        param_optimizer=FakeParamOptimizer(_opt_result(fitness=2.5)),
        optimizer=FakeOptimizer(applied=3),
        performance_feedback=FakePerformanceFeedback(),
        recommendation_builder=_builder,
    )
    result = await orch.run_cycle("grid")
    assert result.status == "deployed"
    assert result.deploy_decision == "approved"
    assert result.deploy_applied == 3
    assert result.best_fitness == 2.5
    assert result.feedback_score == 0.85
    _assert_json_safe(asdict(result))


async def test_reject_non_complete_phase():
    orch = _coordinator(
        param_optimizer=FakeParamOptimizer(_opt_result(phase="failed")),
        optimizer=FakeOptimizer(),
    )
    result = await orch.run_cycle("grid")
    assert result.status == "deploy_rejected"
    assert result.deploy_decision == "rejected"
    assert "not complete" in result.deploy_reason


async def test_reject_errors():
    orch = _coordinator(
        param_optimizer=FakeParamOptimizer(_opt_result(errors=["boom"])),
        optimizer=FakeOptimizer(),
    )
    result = await orch.run_cycle("grid")
    assert result.deploy_decision == "rejected"
    assert "errors" in result.deploy_reason


async def test_reject_non_positive_fitness():
    orch = _coordinator(
        param_optimizer=FakeParamOptimizer(_opt_result(fitness=0.0)),
        optimizer=FakeOptimizer(),
    )
    result = await orch.run_cycle("grid")
    assert result.deploy_decision == "rejected"
    assert "non-positive" in result.deploy_reason


async def test_reject_no_recommendation():
    orch = _coordinator(
        param_optimizer=FakeParamOptimizer(_opt_result(fitness=1.0)),
        optimizer=FakeOptimizer(),
        recommendation_builder=lambda s, r: None,
    )
    result = await orch.run_cycle("grid")
    assert result.deploy_decision == "rejected"
    assert "no deployable" in result.deploy_reason


async def test_cooldown_skip():
    orch = _coordinator(
        param_optimizer=FakeParamOptimizer(_opt_result(fitness=2.0)),
        optimizer=FakeOptimizer(),
        performance_feedback=FakePerformanceFeedback(),
        cooldown=3600.0,
    )
    first = await orch.run_cycle("grid")
    assert first.status == "deployed"

    second = await orch.run_cycle("grid")
    assert second.status == "skipped"
    assert second.message == "cooldown active"


async def test_stats_and_summary_json_safe():
    orch = _coordinator(
        param_optimizer=FakeParamOptimizer(_opt_result(fitness=2.0)),
        optimizer=FakeOptimizer(applied=2),
        performance_feedback=FakePerformanceFeedback(),
    )
    await orch.run_cycle("grid")
    stats = orch.get_stats()
    assert stats["total_cycles"] == 1
    assert stats["deployed"] == 1
    assert stats["by_strategy"]["grid"]["deployed"] == 1
    _assert_json_safe(stats)

    summary = orch.get_summary()
    assert summary["stats"]["deployed"] == 1
    assert "performance_feedback" in summary
    _assert_json_safe(summary)
