"""
TopLevelAGICoordinator 单元测试
==================================
覆盖：全依赖缺失、首次不触发、三条联动规则、冷却、统计、JSON 安全。
"""
import json

import pytest

from core.top_level_agi import TopLevelAGICoordinator


class FakeQuantAGI:
    def __init__(self, grade="B", concentration=0.0, idle_cash=0.0, total_equity=1000.0):
        self.grade = grade
        self.concentration = concentration
        self.idle_cash = idle_cash
        self.total_equity = total_equity
        self.actions = []
        self.timestamp = None

    def get_last_report(self):
        return {
            "timestamp": self.timestamp,
            "cycle": 7,
            "actions": self.actions,
            "status": "ok",
            "reflection": {"health_grade": self.grade, "health_score": 70.0, "alerts_count": 0},
            "perception": {"contribution": {"concentration": self.concentration}},
            "decision": {"allocation_plan": {
                "idle_cash": self.idle_cash, "total_equity": self.total_equity,
            }},
        }


class FakePerception:
    def get_stats(self):
        return {"total_perceived": 10, "passed": 8, "reject_gate": 1, "reject_quality": 1}


class FakeOpsSelfHeal:
    def __init__(self, strategy_alerts=0):
        self.strategy_alerts = strategy_alerts

    def get_stats(self):
        return {
            "total_events": 5,
            "alert_only": 1,
            "by_root_cause": {"strategy": {"alert_only": self.strategy_alerts}},
        }


class FakeAutoOptimization:
    def __init__(self, deployed=0):
        self.deployed = deployed

    def get_stats(self):
        return {"total_cycles": 1, "deployed": self.deployed, "optimized": 0}


def _coordinator(quant_agi=None, perception=None, ops_self_heal=None,
                 auto_optimization=None, cooldown=0.0):
    return TopLevelAGICoordinator(
        quant_agi=quant_agi,
        perception=perception,
        ops_self_heal=ops_self_heal,
        auto_optimization=auto_optimization,
        config={"cooldown_seconds": cooldown},
    )


def _assert_json_safe(obj):
    text = json.dumps(obj, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text
    return text


def test_all_none_empty_report():
    orch = _coordinator()
    report = orch.run_cycle()
    assert report["actions"] == []
    assert report["summary"]["quant_agi"] is None
    assert report["analysis"][0]["status"] == "partial"
    assert report["decision_plan"]["status"] == "no_action"
    _assert_json_safe(report)


def test_first_cycle_no_linkage():
    orch = _coordinator(
        quant_agi=FakeQuantAGI(),
        perception=FakePerception(),
        ops_self_heal=FakeOpsSelfHeal(),
        auto_optimization=FakeAutoOptimization(deployed=5),
    )
    report = orch.run_cycle()
    # 首次运行仅初始化基线，不触发联动（即使 deployed=5 也不触发）
    assert report["actions"] == []
    _assert_json_safe(report)


def test_funding_rebalance_on_deploy_increase():
    ao = FakeAutoOptimization(deployed=0)
    orch = _coordinator(auto_optimization=ao, cooldown=0.0)
    orch.run_cycle()  # 初始化基线 deployed=0

    ao.deployed = 3
    report = orch.run_cycle()
    types = [a["type"] for a in report["actions"]]
    assert "funding_rebalance" in types
    assert report["linkage_state"]["funding_rebalance_needed"] is True
    _assert_json_safe(report)


def test_optimization_pause_on_strategy_alert():
    ops = FakeOpsSelfHeal(strategy_alerts=0)
    orch = _coordinator(ops_self_heal=ops, cooldown=0.0)
    orch.run_cycle()

    ops.strategy_alerts = 2
    report = orch.run_cycle()
    types = [a["type"] for a in report["actions"]]
    assert "optimization_pause_advised" in types
    assert report["linkage_state"]["optimization_pause_advised"] is True


def test_health_degraded():
    orch = _coordinator(quant_agi=FakeQuantAGI(grade="F"), cooldown=0.0)
    orch.run_cycle()  # 初始化
    report = orch.run_cycle()
    types = [a["type"] for a in report["actions"]]
    assert "health_degraded" in types
    assert report["linkage_state"]["health_grade"] == "F"


def test_cooldown_returns_cached():
    orch = _coordinator(
        quant_agi=FakeQuantAGI(),
        auto_optimization=FakeAutoOptimization(deployed=0),
        cooldown=3600.0,
    )
    first = orch.run_cycle()
    assert first["cooldown"] is False

    second = orch.run_cycle()
    assert second["cooldown"] is True


def test_stats_json_safe():
    ao = FakeAutoOptimization(deployed=0)
    orch = _coordinator(auto_optimization=ao, cooldown=0.0)
    orch.run_cycle()
    ao.deployed = 1
    orch.run_cycle()

    stats = orch.get_stats()
    assert stats["total_cycles"] == 2
    assert stats["linkage_actions"] == 1
    assert stats["by_type"]["funding_rebalance"] == 1
    _assert_json_safe(stats)

    status = orch.get_status()
    assert "linkage_state" in status
    assert "stats" in status
    _assert_json_safe(status)


# ── 企业级强化：fail-closed + JSON 安全 ────────────────────

def test_run_cycle_fail_closed_on_gather_exception():
    class BoomPerception:
        def get_stats(self):
            raise RuntimeError("boom")
    orch = _coordinator(perception=BoomPerception())
    report = orch.run_cycle()
    # 聚合异常 → 降级为空计划，不崩溃
    assert report["actions"] == []
    assert report["summary"]["perception"] is None
    _assert_json_safe(report)


def test_summarize_quant_agi_sanitizes_nan():
    report = {
        "status": "ok",
        "reflection": {"health_grade": "F", "health_score": float("nan"), "alerts_count": None},
    }
    summary = TopLevelAGICoordinator._summarize_quant_agi(report)
    assert summary["health_score"] == 0.0  # NaN → 0.0
    assert summary["alerts_count"] == 0    # None → 0


def test_sanitize_removes_nan_inf_recursively():
    obj = {"a": float("nan"), "b": [float("inf"), {"c": float("-inf")}], "d": "ok"}
    cleaned = TopLevelAGICoordinator._sanitize(obj)
    _assert_json_safe(cleaned)
    assert cleaned["a"] == 0.0
    assert cleaned["b"][0] == 0.0
    assert cleaned["b"][1]["c"] == 0.0
    assert cleaned["d"] == "ok"


# ── 跨闭环联动扩展：集中度 / 资金利用率 ────────────────────

def test_funding_rebalance_on_concentration_cross():
    qa = FakeQuantAGI(concentration=0.0)
    orch = _coordinator(quant_agi=qa, cooldown=0.0)
    orch.run_cycle()  # 基线 concentration=0
    qa.concentration = 0.7  # 跨过 0.6 阈值
    report = orch.run_cycle()
    types = [a["type"] for a in report["actions"]]
    assert "funding_rebalance" in types
    assert report["linkage_state"]["funding_rebalance_needed"] is True
    _assert_json_safe(report)


def test_optimization_advised_on_idle_cross():
    qa = FakeQuantAGI(idle_cash=0.0, total_equity=1000.0)
    orch = _coordinator(quant_agi=qa, cooldown=0.0)
    orch.run_cycle()  # 基线 idle 0
    qa.idle_cash = 400.0  # idle_pct 0.4 > 0.3
    report = orch.run_cycle()
    types = [a["type"] for a in report["actions"]]
    assert "optimization_advised" in types
    assert report["linkage_state"]["optimization_advised"] is True
    _assert_json_safe(report)


def test_idle_deploy_auto_authorized_only_from_fresh_quant_agi_action():
    from datetime import datetime

    qa = FakeQuantAGI(idle_cash=0.0)
    qa.timestamp = datetime.now().isoformat()
    qa.actions = [{"type": "idle_cash_deploy", "strategy": "grid"}]
    orch = TopLevelAGICoordinator(
        quant_agi=qa,
        config={"cooldown_seconds": 0.0, "auto_execute_low_risk": True},
    )
    orch.run_cycle()

    qa.idle_cash = 400.0
    qa.timestamp = datetime.now().isoformat()
    report = orch.run_cycle()

    assert report["actions"][0]["type"] == "idle_cash_deploy"
    assert report["actions"][0]["trace_id"] == "agi-cycle-7-0"
    assert report["decision_plan"]["auto_execute"] is True
    assert report["decision_plan"]["steps"][0]["requires_human_approval"] is False


@pytest.mark.parametrize("grade, age_hours", [("B", 1), ("F", 0)])
def test_idle_deploy_does_not_auto_authorize_stale_or_degraded_report(grade, age_hours):
    from datetime import datetime, timedelta

    qa = FakeQuantAGI(grade=grade, idle_cash=0.0)
    qa.timestamp = (datetime.now() - timedelta(hours=age_hours)).isoformat()
    qa.actions = [{"type": "idle_cash_deploy", "strategy": "grid"}]
    orch = TopLevelAGICoordinator(
        quant_agi=qa,
        config={"cooldown_seconds": 0.0, "auto_execute_low_risk": True},
    )
    orch.run_cycle()

    qa.idle_cash = 400.0
    report = orch.run_cycle()

    assert all(action["type"] != "idle_cash_deploy" for action in report["actions"])
    assert report["decision_plan"]["auto_execute"] is False


def test_idle_above_threshold_at_startup_is_rechecked_after_baseline():
    qa = FakeQuantAGI(idle_cash=400.0, total_equity=1000.0)
    orch = _coordinator(quant_agi=qa, cooldown=0.0)
    orch.run_cycle()  # baseline only

    report = orch.run_cycle()

    assert any(action["type"] == "optimization_advised" for action in report["actions"])


def test_no_linkage_below_threshold():
    qa = FakeQuantAGI(concentration=0.5, idle_cash=100.0, total_equity=1000.0)  # 均低于阈值
    orch = _coordinator(quant_agi=qa, cooldown=0.0)
    orch.run_cycle()
    report = orch.run_cycle()
    types = [a["type"] for a in report["actions"]]
    assert "funding_rebalance" not in types
    assert "optimization_advised" not in types


def test_funding_rebalance_auto_executes_when_low_risk_enabled():
    """AG-3: funding_rebalance 属于低风险动作，auto_execute_low_risk 开启时自动执行。"""
    ao = FakeAutoOptimization(deployed=0)
    orch = TopLevelAGICoordinator(
        auto_optimization=ao,
        config={"cooldown_seconds": 0.0, "auto_execute_low_risk": True},
    )
    orch.run_cycle()  # baseline

    ao.deployed = 3
    report = orch.run_cycle()

    steps = report["decision_plan"]["steps"]
    funding_steps = [s for s in steps if s["action"] == "funding_rebalance"]
    assert funding_steps
    assert all(s["auto_execute"] is True for s in funding_steps)
    assert all(s["requires_human_approval"] is False for s in funding_steps)
    assert report["decision_plan"]["auto_execute"] is True


def test_funding_rebalance_requires_approval_when_auto_execute_disabled():
    """AG-3: auto_execute_low_risk 关闭时，funding_rebalance 仍需人工审批。"""
    ao = FakeAutoOptimization(deployed=0)
    orch = TopLevelAGICoordinator(
        auto_optimization=ao,
        config={"cooldown_seconds": 0.0, "auto_execute_low_risk": False},
    )
    orch.run_cycle()

    ao.deployed = 3
    report = orch.run_cycle()

    steps = report["decision_plan"]["steps"]
    funding_steps = [s for s in steps if s["action"] == "funding_rebalance"]
    assert funding_steps
    assert all(s["auto_execute"] is False for s in funding_steps)
    assert all(s["requires_human_approval"] is True for s in funding_steps)


def test_high_risk_actions_never_auto_execute():
    """AG-3: optimization_pause_advised / health_degraded 始终需人工审批。"""
    ops = FakeOpsSelfHeal(strategy_alerts=0)
    orch = TopLevelAGICoordinator(
        ops_self_heal=ops,
        config={"cooldown_seconds": 0.0, "auto_execute_low_risk": True},
    )
    orch.run_cycle()

    ops.strategy_alerts = 2
    report = orch.run_cycle()

    steps = report["decision_plan"]["steps"]
    pause_steps = [s for s in steps if s["action"] == "optimization_pause_advised"]
    assert pause_steps
    assert all(s["auto_execute"] is False for s in pause_steps)
    assert all(s["requires_human_approval"] is True for s in pause_steps)


def test_risk_findings_defer_conflicting_optimization_and_order_plan():
    qa = FakeQuantAGI(grade="B", idle_cash=0.0, total_equity=1000.0)
    ops = FakeOpsSelfHeal(strategy_alerts=0)
    orch = _coordinator(quant_agi=qa, ops_self_heal=ops, cooldown=0.0)
    orch.run_cycle()

    qa.grade = "F"
    qa.idle_cash = 400.0
    ops.strategy_alerts = 2
    report = orch.run_cycle()

    action_types = [action["type"] for action in report["actions"]]
    assert action_types == ["optimization_pause_advised", "health_degraded"]
    assert all(step["auto_execute"] is False for step in report["decision_plan"]["steps"])
    assert [step["action"] for step in report["decision_plan"]["steps"]] == [
        "health_degraded",
        "optimization_pause_advised",
    ]
    assert report["decision_plan"]["requires_human_approval"] is True
    deferred = report["decision_plan"]["deferred_actions"]
    assert len(deferred) == 1
    assert deferred[0]["type"] == "optimization_advised"
    assert set(deferred[0]["blocked_by"]) == {
        "health_degraded",
        "optimization_pause_advised",
    }
    assert report["analysis"][2]["status"] == "deferred"
    assert orch.get_stats()["by_type"].get("optimization_advised", 0) == 0
    _assert_json_safe(report)
