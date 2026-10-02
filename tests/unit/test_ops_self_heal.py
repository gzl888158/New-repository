"""
OpsSelfHealCoordinator 单元测试
==================================
覆盖：根因映射、自动恢复、高风险仅告警、无处理器降级、冷却幂等、统计、JSON 安全。
"""
import json
from dataclasses import asdict

import pytest

from core.ops_self_heal import OpsSelfHealCoordinator, SelfHealResult


class FakeRecoveryHandler:
    def __init__(self, succeed=True):
        self._succeed = succeed
        self.calls = []

    async def handle_failure(self, failure_type, context):
        self.calls.append(failure_type)
        if self._succeed:
            return {"status": "success", "action": "retry"}
        return {"status": "failed", "error": "boom"}

    def get_recovery_summary(self):
        return {"total_tasks": 0, "recent_24h": 0, "success_rate": 1.0}


class FakeAlertManager:
    async def send_system_alert(self, system_type, message, metadata=None):
        pass


def _coordinator(recovery_handler=None, alert_manager=None, anomaly_detector=None):
    return OpsSelfHealCoordinator(
        anomaly_detector=anomaly_detector,
        recovery_handler=recovery_handler,
        alert_manager=alert_manager,
        config={"cooldown_seconds": 60.0},
    )


def _assert_json_safe(obj):
    text = json.dumps(obj, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text
    return text


def test_diagnose_root_cause_mapping():
    orch = _coordinator()
    assert orch.diagnose_root_cause("connection_lost") == "network"
    assert orch.diagnose_root_cause("latency_spike") == "network"
    assert orch.diagnose_root_cause("order_failed") == "order"
    assert orch.diagnose_root_cause("strategy_error") == "strategy"
    assert orch.diagnose_root_cause("drawdown_exceeded") == "risk"
    assert orch.diagnose_root_cause("pnl_drop") == "risk"
    assert orch.diagnose_root_cause("state_corruption") == "state"
    assert orch.diagnose_root_cause("performance_degradation") == "performance"
    assert orch.diagnose_root_cause("totally_unknown") == "unknown"


async def test_network_auto_recover_success():
    rh = FakeRecoveryHandler(succeed=True)
    orch = _coordinator(recovery_handler=rh)
    result = await orch.handle_event("connection_lost", {"symbol": "SOL-USDT-SWAP"})
    assert result.decision == "auto_recover"
    assert result.root_cause == "network"
    assert result.recovered is True
    assert result.recovery_status == "success"
    assert rh.calls == ["connection_lost"]
    _assert_json_safe(asdict(result))


async def test_risk_alert_only_no_recovery():
    rh = FakeRecoveryHandler(succeed=True)
    orch = _coordinator(recovery_handler=rh)
    result = await orch.handle_event("drawdown_exceeded", {})
    assert result.decision == "alert_only"
    assert result.root_cause == "risk"
    assert result.recovered is False
    assert rh.calls == []  # 高风险根因不触发自动恢复
    _assert_json_safe(asdict(result))


async def test_unknown_alert_only():
    orch = _coordinator(recovery_handler=FakeRecoveryHandler())
    result = await orch.handle_event("totally_unknown", {})
    assert result.decision == "alert_only"
    assert result.root_cause == "unknown"
    assert result.recovered is False


async def test_no_recovery_handler_failed():
    orch = _coordinator()  # 无 recovery_handler
    result = await orch.handle_event("order_failed", {})
    assert result.decision == "auto_recover"
    assert result.recovered is False
    assert result.recovery_status == "failed"
    _assert_json_safe(asdict(result))


async def test_recovery_failure_propagates():
    rh = FakeRecoveryHandler(succeed=False)
    orch = _coordinator(recovery_handler=rh)
    result = await orch.handle_event("api_error", {})
    assert result.decision == "auto_recover"
    assert result.recovered is False
    assert result.recovery_status == "failed"


async def test_cooldown_skip():
    rh = FakeRecoveryHandler(succeed=True)
    orch = _coordinator(recovery_handler=rh)
    first = await orch.handle_event("connection_lost", {})
    assert first.decision == "auto_recover"

    second = await orch.handle_event("connection_lost", {})
    assert second.decision == "cooldown_skip"
    assert second.recovery_status == "skipped"
    assert rh.calls == ["connection_lost"]  # 第二次不触发恢复


async def test_stats_accumulate():
    rh = FakeRecoveryHandler(succeed=True)
    orch = _coordinator(recovery_handler=rh)
    await orch.handle_event("connection_lost", {})
    await orch.handle_event("drawdown_exceeded", {})
    await orch.handle_event("order_failed", {})

    stats = orch.get_stats()
    assert stats["total_events"] == 3
    assert stats["auto_recovered"] == 2
    assert stats["recovered_success"] == 2
    assert stats["alert_only"] == 1
    assert stats["by_root_cause"]["network"]["auto_recovered"] == 1
    assert stats["by_root_cause"]["risk"]["alert_only"] == 1
    _assert_json_safe(stats)


def test_get_self_heal_summary_json_safe():
    rh = FakeRecoveryHandler(succeed=True)
    orch = _coordinator(recovery_handler=rh)
    summary = orch.get_self_heal_summary()
    assert summary["stats"]["total_events"] == 0
    assert summary["success_rate"] == 0.0
    assert "recovery" in summary
    _assert_json_safe(summary)
