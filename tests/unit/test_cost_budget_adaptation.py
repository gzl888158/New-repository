"""
滑点/资金费预算动态分配（cost_budget_adaptation）单元测试
=======================================================
覆盖：权益恶化收紧成本预算、权益健康放宽、禁用保持基础阈值。
"""
from collections import deque

import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _config(**overrides):
    guard = {"enabled": True, "adapt_span": 0.15}
    guard.update(overrides)
    return {"agi_orchestrator": {"cost_budget_adaptation": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def test_adaptive_cost_threshold_tightens_when_declining():
    orch = _orch()
    orch._adaptive_risk_enabled = True
    orch._equity_window = deque([1000.0, 900.0], maxlen=5)  # appetite 0
    # adjusted = 0.3 + (0 - 0.5)*0.15 = 0.225
    assert orch._adaptive_cost_threshold(0.3) == pytest.approx(0.225)


def test_adaptive_cost_threshold_loosens_when_healthy():
    orch = _orch()
    orch._adaptive_risk_enabled = True
    orch._equity_window = deque([900.0, 1000.0], maxlen=5)  # appetite 1
    # adjusted = 0.3 + (1 - 0.5)*0.15 = 0.375
    assert orch._adaptive_cost_threshold(0.3) == pytest.approx(0.375)


def test_adaptive_cost_threshold_disabled_uses_base():
    orch = _orch(enabled=False)
    orch._adaptive_risk_enabled = True
    orch._equity_window = deque([1000.0, 900.0], maxlen=5)
    assert orch._adaptive_cost_threshold(0.3) == pytest.approx(0.3)


def test_adaptive_cost_threshold_clamped():
    orch = _orch()
    orch._adaptive_risk_enabled = True
    orch._equity_window = deque([1000.0, 100.0], maxlen=5)  # 大幅下跌 → appetite 0
    # adjusted 会低于下界 → 被 clamp 到 0.05
    assert orch._adaptive_cost_threshold(0.1) == pytest.approx(0.05)
