"""
RestrictedExecutionChannel 单元测试
====================================
覆盖：路由分流（高风险排队 / 低风险告警 / 低风险自动归集）、fail-closed、
人工确认/拒绝、有界 FIFO、统计、JSON 安全、事件审计。
"""
import json
import os

import pytest

from core.restricted_execution_channel import RestrictedExecutionChannel


class FakeEventStore:
    def __init__(self):
        self.events = []

    def append(self, event_type, data, *, event_id=None, timestamp=None,
               source="", symbol="", version=1):
        self.events.append({
            "event_type": event_type,
            "event_id": event_id,
            "data": data,
            "source": source,
            "symbol": symbol,
        })
        return event_id


class FakeAlertManager:
    def __init__(self):
        self.alerts = []

    async def send_alert(self, alert_type, message, severity="INFO",
                         symbol="", metadata=None):
        self.alerts.append({
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "symbol": symbol,
            "metadata": metadata,
        })


class FakeDeployer:
    def __init__(self, result=None, raise_error=False):
        self.calls = []
        self._result = result if result is not None else {"deployed": True}
        self._raise_error = raise_error

    async def __call__(self, action):
        self.calls.append(action)
        if self._raise_error:
            raise RuntimeError("deploy boom")
        return self._result


def _channel(tmp_path, event_store=None, alert_manager=None, max_pending=200,
             idle_cash_deployer=None, autonomous=False, reallocate_deployer=None,
             kill_switch_check=None, close_position_deployer=None,
             param_adjust_deployer=None, strategy_pause_deployer=None,
             strategy_resume_deployer=None, always_require_confirmation=None):
    if always_require_confirmation is None and autonomous:
        always_require_confirmation = []
    return RestrictedExecutionChannel(
        event_store=event_store,
        alert_manager=alert_manager,
        pending_path=str(tmp_path / "agi_pending_actions.json"),
        max_pending=max_pending,
        idle_cash_deployer=idle_cash_deployer,
        autonomous=autonomous,
        reallocate_deployer=reallocate_deployer,
        kill_switch_check=kill_switch_check,
        close_position_deployer=close_position_deployer,
        param_adjust_deployer=param_adjust_deployer,
        strategy_pause_deployer=strategy_pause_deployer,
        strategy_resume_deployer=strategy_resume_deployer,
        always_require_confirmation=always_require_confirmation,
    )


def _assert_json_safe(obj):
    text = json.dumps(obj, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text
    return text


# ── 路由分流 ──────────────────────────────────────────────

async def test_route_no_report(tmp_path):
    ch = _channel(tmp_path)
    result = await ch.route(None)
    assert result["status"] == "no_report"
    assert result["queued"] == 0


async def test_route_no_actions(tmp_path):
    ch = _channel(tmp_path)
    result = await ch.route({"actions": []})
    assert result["status"] == "no_actions"

async def test_fail_closed_report_cannot_dispatch_actions(tmp_path):
    deployer = FakeDeployer()
    ch = _channel(tmp_path, idle_cash_deployer=deployer)
    result = await ch.route({
        "status": "fail_closed",
        "cycle": 2,
        "actions": [{"type": "idle_cash_deploy", "strategy": "grid"}],
    })

    assert result["status"] == "fail_closed"
    assert result["rejected"] == 0
    assert deployer.calls == []


async def test_non_list_actions_are_rejected(tmp_path):
    ch = _channel(tmp_path)
    result = await ch.route({"actions": {"type": "strategy_pause", "strategy": "grid"}})

    assert result["status"] == "invalid_report"
    assert result["rejected"] == 1


async def test_route_high_risk_queued(tmp_path):
    es = FakeEventStore()
    ch = _channel(tmp_path, event_store=es)
    report = {
        "cycle": 3,
        "actions": [{"type": "reallocate", "trace_id": "agi-3-0", "strategy": "grid",
                     "target_allocation": 0.2, "reason": "rebalance"}],
    }
    result = await ch.route(report)
    assert result["status"] == "ok"
    assert result["queued"] == 1
    assert result["notified"] == 0
    assert result["rejected"] == 0

    # 已落盘为 pending
    pending = ch.list_pending("pending")
    assert len(pending) == 1
    assert pending[0]["trace_id"] == "agi-3-0"
    assert pending[0]["status"] == "pending"

    # 事件审计
    queued_events = [e for e in es.events if e["event_type"] == "AGI_ACTION_QUEUED"]
    assert len(queued_events) == 1
    assert queued_events[0]["event_id"] == "agi-3-0"


async def test_route_low_risk_notified(tmp_path):
    am = FakeAlertManager()
    ch = _channel(tmp_path, alert_manager=am)
    report = {
        "cycle": 1,
        "actions": [{"type": "alert_action", "level": "critical",
                     "detail": "drawdown warning"}],
    }
    result = await ch.route(report)
    assert result["notified"] == 1
    assert result["queued"] == 0
    assert len(am.alerts) == 1
    assert am.alerts[0]["alert_type"] == "agi_alert_action"
    assert am.alerts[0]["severity"] == "CRITICAL"

async def test_repeated_route_is_idempotent(tmp_path):
    deployer = FakeDeployer()
    ch = _channel(tmp_path, idle_cash_deployer=deployer)
    report = {
        "cycle": 7,
        "decision_id": "agi-dec-7",
        "actions": [{"type": "idle_cash_deploy", "strategy": "grid"}],
    }

    first = await ch.route(report)
    second = await ch.route(report)

    assert first["deployed"] == 1
    assert second["skipped"] == 1
    assert second["action_results"][0]["status"] == "duplicate"
    assert len(deployer.calls) == 1
    assert ch.get_action_history()[0]["status"] == "deployed"
    assert second["action_results"][0]["result"]["deployed"] is True


async def test_reused_trace_id_with_different_payload_is_rejected(tmp_path):
    deployer = FakeDeployer()
    ch = _channel(tmp_path, idle_cash_deployer=deployer)
    await ch.route({
        "cycle": 11,
        "actions": [{
            "type": "idle_cash_deploy",
            "trace_id": "stable-id",
            "strategy": "grid",
        }],
    })

    result = await ch.route({
        "cycle": 11,
        "actions": [{
            "type": "idle_cash_deploy",
            "trace_id": "stable-id",
            "strategy": "trend",
        }],
    })

    assert result["rejected"] == 1
    assert result["action_results"][0]["reason"] == "trace_id_conflict"
    assert len(deployer.calls) == 1


async def test_executable_action_arguments_are_validated(tmp_path):
    deployer = FakeDeployer()
    ch = _channel(tmp_path, close_position_deployer=deployer)
    result = await ch.route({
        "cycle": 12,
        "actions": [{"type": "profit_take_close", "close_ratio": 1.5}],
    })

    assert result["rejected"] == 1
    assert result["action_results"][0]["reason"] == "invalid_close_ratio"
    assert deployer.calls == []


def test_tool_catalog_reflects_registered_execution_capabilities(tmp_path):
    deployer = FakeDeployer()
    ch = _channel(tmp_path, idle_cash_deployer=deployer)
    catalog = ch.get_tool_catalog()

    assert catalog["tools"]["idle_cash_deploy"] == {
        "risk": "low",
        "available": True,
        "execution": "deployer",
    }
    assert catalog["tools"]["profit_take_close"]["available"] is False



async def test_action_idempotency_survives_restart(tmp_path):
    deployer = FakeDeployer()
    report = {
        "cycle": 8,
        "decision_id": "agi-dec-8",
        "actions": [{"type": "idle_cash_deploy", "strategy": "grid"}],
    }
    first = _channel(tmp_path, idle_cash_deployer=deployer)
    await first.route(report)

    restarted = _channel(tmp_path, idle_cash_deployer=deployer)
    result = await restarted.route(report)

    assert result["skipped"] == 1
    assert len(deployer.calls) == 1


async def test_cached_report_is_never_routed_again(tmp_path):
    deployer = FakeDeployer()
    ch = _channel(tmp_path, idle_cash_deployer=deployer)
    result = await ch.route({
        "status": "cooldown",
        "cycle": 9,
        "decision_id": "agi-dec-9",
        "actions": [{"type": "idle_cash_deploy", "strategy": "grid"}],
    })

    assert result["status"] == "cached_report"
    assert result["skipped"] == 0
    assert deployer.calls == []

async def test_corrupt_action_history_fails_closed(tmp_path):
    history_path = tmp_path / "agi_pending_actions.json.history.json"
    history_path.write_text("{invalid", encoding="utf-8")
    deployer = FakeDeployer()
    ch = _channel(tmp_path, idle_cash_deployer=deployer)

    result = await ch.route({
        "cycle": 10,
        "decision_id": "agi-dec-10",
        "actions": [{"type": "idle_cash_deploy", "strategy": "grid"}],
    })

    assert result["rejected"] == 1
    assert result["action_results"][0]["reason"] == "action_history_unavailable"
    assert deployer.calls == []


async def test_route_malformed_action_not_crash(tmp_path):
    ch = _channel(tmp_path)
    report = {"cycle": 1, "actions": [None, "garbage", {"type": "alert_action",
                                                         "level": "info",
                                                         "detail": "ok"}]}
    result = await ch.route(report)
    assert result["status"] == "ok"
    # 非 dict 动作被静默跳过，不进入拒绝计数
    assert result["notified"] == 1


async def test_route_bounded_fifo(tmp_path):
    ch = _channel(tmp_path, max_pending=2)
    for i in range(3):
        await ch.route({"cycle": i, "actions": [
            {"type": "reallocate", "trace_id": f"agi-{i}", "strategy": "grid"}]})
    all_items = ch.list_pending("")
    assert len(all_items) == 2
    # FIFO：最旧的被挤出，保留最新两条
    assert all_items[0]["trace_id"] == "agi-1"
    assert all_items[1]["trace_id"] == "agi-2"


# ── 人工确认 / 拒绝 ───────────────────────────────────────

async def test_confirm_marks_confirmed(tmp_path):
    es = FakeEventStore()
    ch = _channel(tmp_path, event_store=es)
    await ch.route({"cycle": 1, "actions": [
        {"type": "reallocate", "trace_id": "agi-x", "strategy": "grid"}]})

    result = ch.confirm("agi-x")
    assert result["success"] is True
    assert result["status"] == "confirmed"

    assert ch.list_pending("pending") == []
    confirmed = ch.list_pending("confirmed")
    assert len(confirmed) == 1

    assert any(e["event_type"] == "AGI_ACTION_CONFIRMED" and
               e["event_id"] == "agi-x-confirmed" for e in es.events)


async def test_reject_marks_rejected(tmp_path):
    es = FakeEventStore()
    ch = _channel(tmp_path, event_store=es)
    await ch.route({"cycle": 1, "actions": [
        {"type": "reallocate", "trace_id": "agi-y", "strategy": "grid"}]})

    result = ch.reject("agi-y", reason="manual veto")
    assert result["success"] is True
    assert result["status"] == "rejected"

    rejected = ch.list_pending("rejected")
    assert len(rejected) == 1
    assert rejected[0]["rejected_reason"] == "manual veto"

    assert any(e["event_type"] == "AGI_ACTION_REJECTED" and
               e["event_id"] == "agi-y-rejected" for e in es.events)


def test_confirm_missing_trace_id(tmp_path):
    ch = _channel(tmp_path)
    result = ch.confirm("")
    assert result["success"] is False
    assert result["error"] == "trace_id required"


def test_confirm_not_found(tmp_path):
    ch = _channel(tmp_path)
    result = ch.confirm("does-not-exist")
    assert result["success"] is False
    assert result["error"] == "trace_id not found"


# ── 统计与 JSON 安全 ──────────────────────────────────────

async def test_stats_json_safe(tmp_path):
    es = FakeEventStore()
    ch = _channel(tmp_path, event_store=es)
    await ch.route({"cycle": 1, "actions": [
        {"type": "reallocate", "trace_id": "agi-a", "strategy": "grid"},
        {"type": "alert_action", "level": "info", "detail": "note"}]})
    ch.confirm("agi-a")

    stats = ch.get_stats()
    assert stats["routed"] == 2
    assert stats["queued"] == 1
    assert stats["notified"] == 1
    assert stats["confirmed"] == 1
    _assert_json_safe(stats)


def test_list_pending_empty_when_no_file(tmp_path):
    ch = _channel(tmp_path)
    assert ch.list_pending() == []


# ── 低风险闲置资金自动归集 ────────────────────────────────

async def test_idle_cash_deploy_routes_to_deployer(tmp_path):
    es = FakeEventStore()
    deployer = FakeDeployer()
    ch = _channel(tmp_path, event_store=es, idle_cash_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "idle_cash_deploy", "strategy": "grid", "health_grade": "B",
         "detail": "deploy idle cash"}]}
    result = await ch.route(report)
    assert result["deployed"] == 1
    assert result["queued"] == 0
    assert len(deployer.calls) == 1
    assert deployer.calls[0]["strategy"] == "grid"
    assert any(e["event_type"] == "AGI_ACTION_DEPLOYED" for e in es.events)


async def test_idle_cash_deploy_no_deployer_notifies(tmp_path):
    ch = _channel(tmp_path)  # 未注入 deployer
    report = {"cycle": 1, "actions": [
        {"type": "idle_cash_deploy", "strategy": "grid"}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["queued"] == 0
    assert result["notified"] == 1


async def test_idle_cash_deploy_fail_closed(tmp_path):
    deployer = FakeDeployer(raise_error=True)
    ch = _channel(tmp_path, idle_cash_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "idle_cash_deploy", "strategy": "grid"}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["rejected"] == 1


async def test_idle_cash_deploy_rejected_when_kill_switch_active(tmp_path):
    deployer = FakeDeployer()
    ch = _channel(
        tmp_path,
        idle_cash_deployer=deployer,
        kill_switch_check=lambda: True,
    )
    report = {"cycle": 1, "actions": [
        {"type": "idle_cash_deploy", "strategy": "grid"}
    ]}

    result = await ch.route(report)

    assert result["deployed"] == 0
    assert result["rejected"] == 1
    assert deployer.calls == []


# ── 完全自主模式（autonomous）────────────────────────────

async def test_autonomous_reallocate_deploys(tmp_path):
    """autonomous=True：reallocate 自动落地，不再排队人工确认。"""
    es = FakeEventStore()
    deployer = FakeDeployer()
    ch = _channel(tmp_path, event_store=es, autonomous=True,
                  reallocate_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "reallocate", "trace_id": "agi-auto-0", "strategy": "grid",
         "target_allocation": 0.3, "reason": "rebalance"}]}
    result = await ch.route(report)
    assert result["queued"] == 0
    assert result["deployed"] == 1
    assert len(deployer.calls) == 1
    assert deployer.calls[0]["strategy"] == "grid"
    assert any(e["event_type"] == "AGI_ACTION_DEPLOYED" for e in es.events)
    # 完全自主下不应写 pending
    assert ch.list_pending("pending") == []


async def test_autonomous_kill_switch_rejects(tmp_path):
    """autonomous=True 但 Kill Switch 熔断时：reallocate 被拒绝，绝不落地。"""
    deployer = FakeDeployer()
    ch = _channel(tmp_path, autonomous=True, reallocate_deployer=deployer,
                  kill_switch_check=lambda: True)
    report = {"cycle": 1, "actions": [
        {"type": "reallocate", "trace_id": "agi-auto-1", "strategy": "grid",
         "target_allocation": 0.3}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["rejected"] == 1
    assert len(deployer.calls) == 0


async def test_autonomous_kill_switch_fail_closed(tmp_path):
    """kill_switch 检查本身异常时 fail-closed 视为熔断，拒绝资金动作。"""
    def _boom():
        raise RuntimeError("rg down")
    deployer = FakeDeployer()
    ch = _channel(tmp_path, autonomous=True, reallocate_deployer=deployer,
                  kill_switch_check=_boom)
    report = {"cycle": 1, "actions": [
        {"type": "reallocate", "trace_id": "agi-auto-2", "strategy": "grid",
         "target_allocation": 0.3}]}
    result = await ch.route(report)
    assert result["rejected"] == 1
    assert len(deployer.calls) == 0


async def test_autonomous_recommendation_notifies(tmp_path):
    """autonomous=True：allocation_recommendation 自动通知（不排队、不落地）。"""
    am = FakeAlertManager()
    ch = _channel(tmp_path, alert_manager=am, autonomous=True)
    report = {"cycle": 1, "actions": [
        {"type": "allocation_recommendation", "detail": "Deploy idle cash"}]}
    result = await ch.route(report)
    assert result["queued"] == 0
    assert result["notified"] == 1
    assert len(am.alerts) == 1
    assert am.alerts[0]["alert_type"] == "agi_allocation_recommendation"


async def test_non_autonomous_keeps_queuing(tmp_path):
    """autonomous=False：保持原 fail-closed 排队逻辑（默认行为）。"""
    ch = _channel(tmp_path)  # autonomous 默认 False
    report = {"cycle": 1, "actions": [
        {"type": "reallocate", "trace_id": "agi-q", "strategy": "grid"}]}
    result = await ch.route(report)
    assert result["queued"] == 1
    assert result["deployed"] == 0


# ── 账户级收益落袋平仓（profit_take_close）────────────────

async def test_profit_take_close_routes_to_close_deployer(tmp_path):
    """profit_take_close 为低风险自动动作，路由到 close_position_deployer，不排队。"""
    es = FakeEventStore()
    deployer = FakeDeployer()
    ch = _channel(tmp_path, event_store=es, close_position_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "profit_take_close", "close_ratio": 0.4, "unrealized_pnl_pct": 0.05,
         "reason": "账户整体浮盈 5%，落袋 40% 仓位"}]}
    result = await ch.route(report)
    assert result["deployed"] == 1
    assert result["queued"] == 0
    assert len(deployer.calls) == 1
    assert deployer.calls[0]["close_ratio"] == 0.4
    assert any(e["event_type"] == "AGI_ACTION_DEPLOYED" for e in es.events)


async def test_profit_take_close_no_deployer_notifies(tmp_path):
    ch = _channel(tmp_path)  # 未注入 close deployer
    report = {"cycle": 1, "actions": [
        {"type": "profit_take_close", "close_ratio": 0.4}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["queued"] == 0
    assert result["notified"] == 1


async def test_profit_take_close_fail_closed(tmp_path):
    deployer = FakeDeployer(raise_error=True)
    ch = _channel(tmp_path, close_position_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "profit_take_close", "close_ratio": 0.4}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["rejected"] == 1


# ── 策略参数自适应（param_adjust）─────────────────────────

async def test_param_adjust_routes_to_deployer(tmp_path):
    """param_adjust 为低风险自动动作，路由到 param_adjust_deployer，不排队。"""
    es = FakeEventStore()
    deployer = FakeDeployer()
    ch = _channel(tmp_path, event_store=es, param_adjust_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "param_adjust", "strategy": "grid", "param": "leverage",
         "value": 1.0, "reason": "健康度F降杠杆"}]}
    result = await ch.route(report)
    assert result["deployed"] == 1
    assert result["queued"] == 0
    assert len(deployer.calls) == 1
    assert deployer.calls[0]["strategy"] == "grid"
    assert deployer.calls[0]["value"] == 1.0
    assert any(e["event_type"] == "AGI_ACTION_DEPLOYED" for e in es.events)


async def test_param_adjust_no_deployer_notifies(tmp_path):
    ch = _channel(tmp_path)  # 未注入 param_adjust deployer
    report = {"cycle": 1, "actions": [
        {"type": "param_adjust", "strategy": "grid", "value": 1.0}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["queued"] == 0
    assert result["notified"] == 1


async def test_param_adjust_fail_closed(tmp_path):
    deployer = FakeDeployer(raise_error=True)
    ch = _channel(tmp_path, param_adjust_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "param_adjust", "strategy": "grid", "value": 1.0}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["rejected"] == 1


# ── 自主策略生命周期（strategy_pause）────────────────────

async def test_strategy_pause_routes_to_deployer(tmp_path):
    """strategy_pause 为低风险自动动作，路由到 strategy_pause_deployer，不排队。"""
    es = FakeEventStore()
    deployer = FakeDeployer()
    ch = _channel(tmp_path, event_store=es, strategy_pause_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "strategy_pause", "strategy": "grid", "reason": "永久冻结停开新仓"}]}
    result = await ch.route(report)
    assert result["deployed"] == 1
    assert result["queued"] == 0
    assert len(deployer.calls) == 1
    assert deployer.calls[0]["strategy"] == "grid"
    assert any(e["event_type"] == "AGI_ACTION_DEPLOYED" for e in es.events)


async def test_strategy_pause_no_deployer_notifies(tmp_path):
    ch = _channel(tmp_path)  # 未注入 strategy_pause deployer
    report = {"cycle": 1, "actions": [
        {"type": "strategy_pause", "strategy": "grid"}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["queued"] == 0
    assert result["notified"] == 1


async def test_strategy_pause_fail_closed(tmp_path):
    deployer = FakeDeployer(raise_error=True)
    ch = _channel(tmp_path, strategy_pause_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "strategy_pause", "strategy": "grid"}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["rejected"] == 1


# ── 自主策略恢复（strategy_resume）────────────────────────

async def test_strategy_resume_routes_to_deployer(tmp_path):
    """strategy_resume 为低风险自动动作，路由到 strategy_resume_deployer，不排队。"""
    es = FakeEventStore()
    deployer = FakeDeployer()
    ch = _channel(tmp_path, event_store=es, strategy_resume_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "strategy_resume", "strategy": "grid", "reason": "健康度改善恢复开仓"}]}
    result = await ch.route(report)
    assert result["deployed"] == 1
    assert result["queued"] == 0
    assert len(deployer.calls) == 1
    assert deployer.calls[0]["strategy"] == "grid"
    assert any(e["event_type"] == "AGI_ACTION_DEPLOYED" for e in es.events)


async def test_strategy_resume_no_deployer_notifies(tmp_path):
    ch = _channel(tmp_path)  # 未注入 strategy_resume deployer
    report = {"cycle": 1, "actions": [
        {"type": "strategy_resume", "strategy": "grid"}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["queued"] == 0
    assert result["notified"] == 1


async def test_strategy_resume_fail_closed(tmp_path):
    deployer = FakeDeployer(raise_error=True)
    ch = _channel(tmp_path, strategy_resume_deployer=deployer)
    report = {"cycle": 1, "actions": [
        {"type": "strategy_resume", "strategy": "grid"}]}
    result = await ch.route(report)
    assert result["deployed"] == 0
    assert result["rejected"] == 1
