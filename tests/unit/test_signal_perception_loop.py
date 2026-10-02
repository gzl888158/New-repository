"""
SignalPerceptionCoordinator 单元测试
======================================
覆盖：全依赖缺失、fake 注入、门控拒绝、质量拒绝、异常 fail-closed、统计累计、JSON 安全。
"""
import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from core.signal_perception_loop import SignalPerceptionCoordinator, PerceptionResult


class FakeRegimeEngine:
    def __init__(self, regime="range_bound"):
        self._regime = regime

    def get_regime(self):
        return {"regime": self._regime, "strength": 0.4}


class FakeRegimeGate:
    def __init__(self, allowed=True, reason="ok", regime="range_bound"):
        self._allowed = allowed
        self._reason = reason
        self._regime = regime

    def evaluate(self, symbol, strategy, signal_type="", direction="", confidence=0.0):
        return SimpleNamespace(allowed=self._allowed, reason=self._reason, regime=self._regime)


class BoomRegimeGate:
    def evaluate(self, *args, **kwargs):
        raise RuntimeError("gate boom")


class FakeQualityEngine:
    def __init__(self, acceptable=True, score=0.8, grade="good"):
        self._acceptable = acceptable
        self._score = score
        self._grade = grade

    def is_signal_acceptable(self, signal_data):
        return self._acceptable, {
            "overall_score": self._score,
            "quality": self._grade,
            "factor_scores": {},
            "signal_id": signal_data.get("signal_id", ""),
        }


class BoomQualityEngine:
    def is_signal_acceptable(self, signal_data):
        raise RuntimeError("quality boom")


def _sig():
    return {
        "symbol": "SOL-USDT-SWAP",
        "strategy_name": "grid",
        "signal_type": "open",
        "direction": "long",
        "confidence": 0.6,
    }


def _coordinator(regime_engine=None, regime_gate=None, quality_engine=None):
    return SignalPerceptionCoordinator(
        regime_engine=regime_engine,
        regime_gate=regime_gate,
        quality_engine=quality_engine,
        config={},
    )


def _assert_json_safe(obj):
    text = json.dumps(obj, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text
    return text


def test_all_dependencies_none_fail_closed():
    orch = _coordinator()
    result = orch.perceive(_sig())
    assert result.decision == "reject_gate"
    assert result.gate_allowed is False
    _assert_json_safe(asdict(result))


def test_pass_when_gate_and_quality_pass():
    orch = _coordinator(
        regime_engine=FakeRegimeEngine("range_bound"),
        regime_gate=FakeRegimeGate(allowed=True),
        quality_engine=FakeQualityEngine(acceptable=True, score=0.8),
    )
    result = orch.perceive(_sig())
    assert result.decision == "pass"
    assert result.regime == "range_bound"
    assert result.quality_score == 0.8
    assert result.quality_acceptable is True
    _assert_json_safe(asdict(result))


def test_reject_gate():
    orch = _coordinator(
        regime_engine=FakeRegimeEngine("trend_bullish"),
        regime_gate=FakeRegimeGate(allowed=False, reason="trend blocks grid", regime="trend_bullish"),
        quality_engine=FakeQualityEngine(),
    )
    result = orch.perceive(_sig())
    assert result.decision == "reject_gate"
    assert result.gate_allowed is False
    assert "trend blocks grid" in result.reason
    _assert_json_safe(asdict(result))


def test_reject_quality():
    orch = _coordinator(
        regime_engine=FakeRegimeEngine("range_bound"),
        regime_gate=FakeRegimeGate(allowed=True),
        quality_engine=FakeQualityEngine(acceptable=False, score=0.2, grade="poor"),
    )
    result = orch.perceive(_sig())
    assert result.decision == "reject_quality"
    assert result.quality_acceptable is False
    _assert_json_safe(asdict(result))


def test_quality_exception_fail_closed():
    orch = _coordinator(
        regime_engine=FakeRegimeEngine("range_bound"),
        regime_gate=FakeRegimeGate(allowed=True),
        quality_engine=BoomQualityEngine(),
    )
    result = orch.perceive(_sig())
    assert result.decision == "reject_quality"
    assert result.quality_acceptable is False
    _assert_json_safe(asdict(result))


def test_gate_exception_fail_closed():
    orch = _coordinator(
        regime_engine=FakeRegimeEngine("range_bound"),
        regime_gate=BoomRegimeGate(),
        quality_engine=FakeQualityEngine(),
    )
    result = orch.perceive(_sig())
    assert result.decision == "reject_gate"
    assert result.gate_allowed is False
    _assert_json_safe(asdict(result))


def test_stats_accumulate_by_decision():
    orch = _coordinator(
        regime_engine=FakeRegimeEngine("range_bound"),
        regime_gate=FakeRegimeGate(allowed=True),
        quality_engine=FakeQualityEngine(acceptable=True, score=0.8),
    )
    orch.perceive(_sig())
    orch.perceive(_sig())
    stats = orch.get_stats()
    assert stats["total_perceived"] == 2
    assert stats["passed"] == 2
    assert stats["by_regime"]["range_bound"]["passed"] == 2
    assert stats["by_strategy"]["grid"]["passed"] == 2
    assert stats["quality_distribution"]["good"] == 2
    _assert_json_safe(stats)


def test_stats_reject_gate_classified():
    orch = _coordinator(
        regime_engine=FakeRegimeEngine("trend_bearish"),
        regime_gate=FakeRegimeGate(allowed=False, reason="trend blocks grid", regime="trend_bearish"),
        quality_engine=FakeQualityEngine(),
    )
    orch.perceive(_sig())
    stats = orch.get_stats()
    assert stats["reject_gate"] == 1
    assert stats["by_regime"]["trend_bearish"]["reject_gate"] == 1
    _assert_json_safe(stats)
