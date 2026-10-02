"""
AGI 决策置信度自适应（confidence_adaptation）单元测试
===================================================
覆盖：权益恶化提高置信门槛、权益健康保持基础门槛、禁用保持基础门槛、
低置信进攻动作被自适应门槛丢弃。
"""
from collections import deque

import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _config(**overrides):
    guard = {"enabled": True, "adapt_span": 0.15}
    guard.update(overrides)
    return {"agi_orchestrator": {"confidence_adaptation": guard}}


def _declining_orch():
    orch = QuantAGIOrchestrator(config=_config())
    orch._reconciliation_enabled = True
    orch._adaptive_risk_enabled = True
    orch._equity_window = deque([1000.0, 900.0], maxlen=5)  # 下跌10% → appetite 0
    return orch


def test_confidence_adaptation_raises_threshold_when_declining():
    orch = _declining_orch()
    orch._apply_decision_quality([], {})
    # effective = 0.5 + (1 - 0) * 0.15 = 0.65
    assert orch._last_confidence_gate["min_confidence"] == pytest.approx(0.65)


def test_confidence_adaptation_base_when_healthy():
    orch = QuantAGIOrchestrator(config=_config())
    orch._reconciliation_enabled = True
    orch._adaptive_risk_enabled = True
    orch._equity_window = deque([900.0, 1000.0], maxlen=5)  # 上涨 → appetite 1
    orch._apply_decision_quality([], {})
    # effective = 0.5 + (1 - 1) * 0.15 = 0.5
    assert orch._last_confidence_gate["min_confidence"] == pytest.approx(0.5)


def test_confidence_adaptation_disabled_uses_base():
    orch = QuantAGIOrchestrator(config=_config(enabled=False))
    orch._reconciliation_enabled = True
    orch._adaptive_risk_enabled = True
    orch._equity_window = deque([1000.0, 900.0], maxlen=5)
    orch._apply_decision_quality([], {})
    assert orch._last_confidence_gate["min_confidence"] == pytest.approx(0.5)


def test_confidence_adaptation_drops_low_confidence_offensive_when_declining():
    orch = _declining_orch()
    action = {"type": "reallocate", "strategy": "grid", "action": "increase",
              "target_allocation": 0.3}
    decision = {"market_regime_strength": 0.8,
                "strategy_metrics": {"grid": {"profit_factor": 1.0, "sharpe_ratio": 0.0}}}
    # confidence = 0.3 + 0.3*0.8 = 0.54 < 0.65（自适应门槛）→ 丢弃
    kept = orch._apply_decision_quality([action], decision)
    assert kept == []


def test_confidence_adaptation_keeps_offensive_when_healthy():
    orch = QuantAGIOrchestrator(config=_config())
    orch._reconciliation_enabled = True
    orch._adaptive_risk_enabled = True
    orch._equity_window = deque([900.0, 1000.0], maxlen=5)  # appetite 1 → 门槛 0.5
    action = {"type": "reallocate", "strategy": "grid", "action": "increase",
              "target_allocation": 0.3}
    decision = {"market_regime_strength": 0.8,
                "strategy_metrics": {"grid": {"profit_factor": 1.0, "sharpe_ratio": 0.0}}}
    kept = orch._apply_decision_quality([action], decision)
    assert len(kept) == 1  # confidence 0.54 >= 0.5 → 保留
