"""
AGI 决策可解释审计增强（_guard_from_reason / _append_decision_lineage）单元测试
==============================================================================
覆盖：从 reason 提取守卫名、溯源历史动作结构化字段增强。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def test_guard_from_reason_extracts_guard():
    assert QuantAGIOrchestrator._guard_from_reason("hourly_pnl_guard: 失血") == "hourly_pnl_guard"
    assert QuantAGIOrchestrator._guard_from_reason("net_exposure_guard: 失衡") == "net_exposure_guard"


def test_guard_from_reason_empty_cases():
    assert QuantAGIOrchestrator._guard_from_reason(None) is None
    assert QuantAGIOrchestrator._guard_from_reason("") is None
    assert QuantAGIOrchestrator._guard_from_reason("   ") is None


def test_guard_from_reason_no_colon():
    assert QuantAGIOrchestrator._guard_from_reason("plain_reason") == "plain_reason"


def test_append_decision_lineage_enriched(tmp_path):
    orch = QuantAGIOrchestrator(config={"agi_orchestrator": {"decision_lineage": {
        "enabled": True, "max_entries": 10, "path": str(tmp_path / "lineage.json"),
    }}})
    report = {
        "decision_id": "agi-dec-1", "cycle": 1, "timestamp": "t", "status": "ok",
        "decision": {"rationale": ["r1"]},
        "reflection": {"health_grade": "A", "decision_quality": 0.5},
        "perception": {"equity": 100.0, "market_regime": {"regime": "trend_bullish"}},
        "actions": [{
            "type": "param_adjust", "strategy": "grid", "action": None, "param": "leverage",
            "value": 1.0, "target_allocation": None, "reason": "health_crash_guard: 骤降",
            "confidence": 0.9, "priority": 5, "rationale": "降杠杆",
        }],
        "alerts": [{"type": "x"}],
    }
    orch._append_decision_lineage(report)
    entry = orch._decision_lineage[-1]
    assert entry["summary"]["alerts"] == 1
    assert entry["summary"]["actions"] == 1
    assert entry["summary"]["decision_quality"] == pytest.approx(0.5)
    a = entry["actions"][0]
    assert a["guard"] == "health_crash_guard"
    assert a["reason"] == "health_crash_guard: 骤降"
    assert a["confidence"] == pytest.approx(0.9)
    assert a["priority"] == 5
    assert a["param"] == "leverage"
    assert a["value"] == pytest.approx(1.0)
    assert a["rationale"] == "降杠杆"
