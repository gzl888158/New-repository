"""冻结策略「观察期自动解冻 + 缩量试探」机制测试（P2-8 / AGI 能力增强）。"""
import pytest

from risk.dynamic_allocator import DynamicAllocator, AllocationPriority, MarketRegime


def _metrics(consecutive_losses=0, max_drawdown=0.0, trade_count=30,
             win_rate=0.5, sharpe=0.1, profit_factor=1.0, total_pnl=0.0):
    return {
        "consecutive_losses": consecutive_losses,
        "max_drawdown": max_drawdown,
        "trade_count": trade_count,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "profit_factor": profit_factor,
        "total_pnl": total_pnl,
    }


@pytest.fixture
def allocator(config):
    return DynamicAllocator(config)


def test_freeze_on_consecutive_losses(allocator):
    metrics = {"scalping": _metrics(consecutive_losses=8)}
    priorities = allocator._evaluate_strategy_priorities(
        ["scalping"], metrics, MarketRegime.RANGING
    )
    assert priorities["scalping"] == AllocationPriority.FROZEN
    assert "scalping" in allocator._freeze_state
    assert allocator._freeze_state["scalping"]["reason"] == "consecutive_losses"


def test_freeze_on_high_drawdown(allocator):
    metrics = {"sync": _metrics(consecutive_losses=3, max_drawdown=0.48)}
    priorities = allocator._evaluate_strategy_priorities(
        ["sync"], metrics, MarketRegime.RANGING
    )
    assert priorities["sync"] == AllocationPriority.FROZEN
    assert allocator._freeze_state["sync"]["reason"] == "high_drawdown"


def test_observe_period_then_probe(allocator, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr("risk.dynamic_allocator.time.time", lambda: t[0])

    metrics = {"scalping": _metrics(consecutive_losses=8)}

    # 首次冻结
    p = allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)
    assert p["scalping"] == AllocationPriority.FROZEN

    # 观察期内仍冻结
    t[0] += allocator._freeze_observe_seconds * 0.5
    p = allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)
    assert p["scalping"] == AllocationPriority.FROZEN

    # 观察期满 → 缩量试探
    t[0] += allocator._freeze_observe_seconds
    p = allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)
    assert p["scalping"] == AllocationPriority.LOW
    assert "scalping" in allocator._probing_strategies


def test_probe_recovery_auto_unfreeze(allocator, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr("risk.dynamic_allocator.time.time", lambda: t[0])

    metrics_frozen = {"scalping": _metrics(consecutive_losses=8)}
    allocator._evaluate_strategy_priorities(["scalping"], metrics_frozen, MarketRegime.RANGING)

    # 观察期满进入试探
    t[0] += allocator._freeze_observe_seconds + 1
    p = allocator._evaluate_strategy_priorities(["scalping"], metrics_frozen, MarketRegime.RANGING)
    assert p["scalping"] == AllocationPriority.LOW

    # 试探期出现盈利，连续亏损被打断（8 → 2 < 5）→ 自动解冻
    metrics_recovered = {"scalping": _metrics(consecutive_losses=2)}
    p = allocator._evaluate_strategy_priorities(
        ["scalping"], metrics_recovered, MarketRegime.RANGING
    )
    assert p["scalping"] != AllocationPriority.FROZEN
    assert "scalping" not in allocator._freeze_state


def test_probe_exhausted_permanent_freeze(allocator, monkeypatch):
    # 简化：观察期=60s，试探窗口=60s，最大试探 2 次
    allocator._freeze_observe_seconds = 60
    allocator._probe_max_attempts = 2

    t = [1000.0]
    monkeypatch.setattr("risk.dynamic_allocator.time.time", lambda: t[0])

    metrics = {"scalping": _metrics(consecutive_losses=8)}

    # 首次冻结
    allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)

    # 观察期满 → 第 1 次试探
    t[0] += 61
    p = allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)
    assert p["scalping"] == AllocationPriority.LOW

    # 试探窗口满，未恢复 → 回观察期
    t[0] += 61
    p = allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)
    assert p["scalping"] == AllocationPriority.FROZEN

    # 观察期满 → 第 2 次试探
    t[0] += 61
    p = allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)
    assert p["scalping"] == AllocationPriority.LOW

    # 第 2 次试探窗口满，仍未恢复，达到 max_attempts → 永久冻结
    t[0] += 61
    p = allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)
    assert p["scalping"] == AllocationPriority.FROZEN


@pytest.mark.asyncio
async def test_probe_weight_reduced(allocator, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr("risk.dynamic_allocator.time.time", lambda: t[0])

    metrics = {
        "scalping": _metrics(consecutive_losses=8),                 # 冻结 → 试探
        "grid": _metrics(consecutive_losses=4, trade_count=9),      # 普通 LOW（样本不足）
    }
    allocator._evaluate_strategy_priorities(
        ["scalping", "grid"], metrics, MarketRegime.RANGING
    )
    # 观察期满进入试探
    t[0] += allocator._freeze_observe_seconds + 1

    plan = await allocator.compute_allocation_plan(
        total_capital=1000.0,
        total_equity=1000.0,
        strategy_names=["scalping", "grid"],
        strategy_metrics=metrics,
        market_regime=MarketRegime.RANGING,
        current_weights={},
        used_margin_by_strategy={},
        persist_last_plan=False,
    )

    s = plan.strategy_allocations["scalping"]
    g = plan.strategy_allocations["grid"]
    assert s.is_probing is True
    assert g.is_probing is False
    # 缩量试探：probing 策略权重 = 普通 LOW 的 probe_weight_ratio 倍
    assert abs(s.target_weight - g.target_weight * allocator._probe_weight_ratio) < 1e-6


def test_no_freeze_when_below_threshold(allocator):
    metrics = {"grid": _metrics(consecutive_losses=4, trade_count=9)}
    priorities = allocator._evaluate_strategy_priorities(
        ["grid"], metrics, MarketRegime.RANGING
    )
    # 连续亏损 4 < 5 且回撤 0，不冻结（trade_count 9 < 20 → LOW，但非冻结）
    assert priorities["grid"] != AllocationPriority.FROZEN
    assert "grid" not in allocator._freeze_state


def test_get_freeze_state_snapshot(allocator, monkeypatch):
    t = [1000.0]
    monkeypatch.setattr("risk.dynamic_allocator.time.time", lambda: t[0])

    metrics = {"scalping": _metrics(consecutive_losses=8)}
    allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)

    fs = allocator.get_freeze_state()
    assert "scalping" in fs
    assert fs["scalping"]["probing"] is False
    assert fs["scalping"]["reason"] == "consecutive_losses"
    assert fs["scalping"]["remaining_seconds"] > 0

    # 观察期满 → 试探
    t[0] += allocator._freeze_observe_seconds + 1
    allocator._evaluate_strategy_priorities(["scalping"], metrics, MarketRegime.RANGING)
    fs = allocator.get_freeze_state()
    assert fs["scalping"]["probing"] is True
    assert fs["scalping"]["probe_attempts"] == 1
    assert fs["scalping"]["permanently_frozen"] is False


def test_positive_return_strategy_relaxed_drawdown_freeze(allocator):
    """正收益高盈亏比策略：回撤 48% 不冻结，且样本 14 笔不降为 LOW（放宽到 HIGH）。"""
    metrics = {"sync": _metrics(
        consecutive_losses=3, max_drawdown=0.48, trade_count=14,
        win_rate=0.14, profit_factor=6.3, total_pnl=1.57,
    )}
    priorities = allocator._evaluate_strategy_priorities(
        ["sync"], metrics, MarketRegime.RANGING
    )
    assert priorities["sync"] == AllocationPriority.HIGH
    assert "sync" not in allocator._freeze_state


def test_negative_return_strategy_drawdown_freeze(allocator):
    """负收益策略：回撤 48% 仍触发冻结（不放宽）。"""
    metrics = {"grid": _metrics(
        consecutive_losses=4, max_drawdown=0.48, trade_count=14,
        win_rate=0.0, profit_factor=0.5, total_pnl=-1.13,
    )}
    priorities = allocator._evaluate_strategy_priorities(
        ["grid"], metrics, MarketRegime.RANGING
    )
    assert priorities["grid"] == AllocationPriority.FROZEN
    assert allocator._freeze_state["grid"]["reason"] == "high_drawdown"
