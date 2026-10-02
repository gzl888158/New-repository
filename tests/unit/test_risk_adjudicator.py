"""P2 企业级升级：独立风控裁决器（RiskAdjudicator）单元测试。

覆盖：
1. 构造器依赖校验（未注入 RiskGate 抛 ValueError）
2. traceID 生成与回写（缺失时生成，已有时复用，全链路同源）
3. 通过路径：passed=True 返回带 traceID 的 AdjudicationResult
4. 拦截路径：真实 RiskGate L0 Kill Switch 拦截开仓，blocked_layer 正确
5. fail-closed：risk_gate.validate 抛异常 → passed=False, action=REJECT
6. 裁决事件发布：publish_sync 收到 RISK_ADJUDICATED，payload 带 trace_id
7. 未注入 event_bus 时发布静默跳过（不崩溃）
8. AdjudicationResult.to_dict
"""

import pytest

from core.kill_switch import KillSwitch
from core.risk_adjudicator import AdjudicationResult, RiskAdjudicator
from core.risk_gate import (
    RiskAction,
    RiskCheckResult,
    RiskGate,
    RiskGateResult,
    RiskLayer,
)
from core.unified_layer import Event, EventType


class RecordingEventBus:
    """记录发布事件的最小事件总线（仅实现 publish_sync）。"""

    def __init__(self):
        self.events = []

    def publish_sync(self, event):
        self.events.append(event)


class FakeRiskGate:
    """可控 RiskGate 桩：validate 返回确定性结果或抛异常（测 fail-closed）。"""

    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc
        self.validate_calls = []

    def validate(self, signal, market_data=None, is_close=False):
        self.validate_calls.append((signal, market_data, is_close))
        if self._exc is not None:
            raise self._exc
        return self._result


def _pass_result() -> RiskGateResult:
    return RiskGateResult(
        passed=True, action=RiskAction.PASS, blocked_layer=None, results=[], summary="ok"
    )


def _reject_result() -> RiskGateResult:
    return RiskGateResult(
        passed=False,
        action=RiskAction.REJECT,
        blocked_layer=RiskLayer.L1_PRE_TRADE,
        results=[
            RiskCheckResult(
                RiskLayer.L1_PRE_TRADE, False, RiskAction.REJECT, "test reject"
            )
        ],
        summary="L1 拦截",
    )


class TestRiskAdjudicatorConstructor:
    def test_requires_risk_gate(self):
        with pytest.raises(ValueError):
            RiskAdjudicator(risk_gate=None)


class TestRiskAdjudicatorTraceID:
    def test_traceid_generated_and_written_back(self):
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(result=_pass_result()))
        signal = {"symbol": "ETH-USDT-SWAP"}

        result = adj.adjudicate(signal)

        assert result.trace_id
        assert result.trace_id.startswith("evt-")
        # 全链路 traceID 回写到 signal，供信号→裁决→订单同源追踪
        assert signal["trace_id"] == result.trace_id

    def test_traceid_reused_when_upstream_exists(self):
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(result=_pass_result()))
        upstream = "evt-upstream-00042"
        signal = {"symbol": "BTC-USDT-SWAP", "trace_id": upstream}

        result = adj.adjudicate(signal)

        # 复用上游 traceID，不重新生成
        assert result.trace_id == upstream
        assert signal["trace_id"] == upstream

    def test_non_dict_signal_coerced(self):
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(result=_pass_result()))
        result = adj.adjudicate(None)
        assert result.trace_id


class TestRiskAdjudicatorDecision:
    def test_pass_returns_trace_id_and_elapsed(self):
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(result=_pass_result()))
        result = adj.adjudicate({"symbol": "ETH-USDT-SWAP", "strategy_name": "grid"})

        assert result.passed is True
        assert result.action == RiskAction.PASS
        assert result.blocked_layer is None
        assert result.symbol == "ETH-USDT-SWAP"
        assert result.elapsed_ms >= 0.0

    def test_reject_propagates_blocked_layer(self):
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(result=_reject_result()))
        result = adj.adjudicate({"symbol": "ETH-USDT-SWAP"})

        assert result.passed is False
        assert result.action == RiskAction.REJECT
        assert result.blocked_layer == RiskLayer.L1_PRE_TRADE

    def test_fail_closed_on_validate_exception(self):
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(exc=RuntimeError("boom")))
        result = adj.adjudicate({"symbol": "ETH-USDT-SWAP"})

        # fail-closed：裁决器自身异常一律拒绝，绝不静默放行
        assert result.passed is False
        assert result.action == RiskAction.REJECT

    def test_validate_receives_is_close_flag(self):
        gate = FakeRiskGate(result=_pass_result())
        adj = RiskAdjudicator(risk_gate=gate)

        adj.adjudicate({"symbol": "ETH-USDT-SWAP"}, is_close=True)

        assert gate.validate_calls[0][2] is True


class TestRiskAdjudicatorEventPublish:
    def test_publishes_adjudication_event_with_trace_id(self):
        bus = RecordingEventBus()
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(result=_pass_result()), event_bus=bus)
        upstream = "evt-trace-12345"
        signal = {"symbol": "ETH-USDT-SWAP", "strategy_name": "grid", "trace_id": upstream}

        result = adj.adjudicate(signal)

        assert len(bus.events) == 1
        evt = bus.events[0]
        assert isinstance(evt, Event)
        assert evt.event_type == EventType.RISK_ADJUDICATED
        assert evt.data["trace_id"] == upstream
        assert evt.data["passed"] is True
        assert evt.data["symbol"] == "ETH-USDT-SWAP"
        assert evt.data["strategy"] == "grid"

    def test_publishes_event_on_reject_too(self):
        bus = RecordingEventBus()
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(result=_reject_result()), event_bus=bus)

        adj.adjudicate({"symbol": "ETH-USDT-SWAP"})

        assert len(bus.events) == 1
        assert bus.events[0].event_type == EventType.RISK_ADJUDICATED
        assert bus.events[0].data["passed"] is False

    def test_no_event_bus_does_not_crash(self):
        adj = RiskAdjudicator(risk_gate=FakeRiskGate(result=_pass_result()))
        result = adj.adjudicate({"symbol": "ETH-USDT-SWAP"})
        assert result.passed is True


class TestRiskAdjudicatorRiskGateIntegration:
    @staticmethod
    def _make_gate(tmp_path):
        # 注入临时 KillSwitch（tmp_path），避免污染生产 data/kill_switch_state.json
        rg = RiskGate(config={})
        rg.set_kill_switch(KillSwitch(state_file=str(tmp_path / "ks.json")))
        return rg

    def test_l0_kill_switch_blocks_open_via_adjudicator(self, tmp_path):
        rg = self._make_gate(tmp_path)
        adj = RiskAdjudicator(risk_gate=rg)
        rg.enable_kill_switch(reason="测试暂停")

        signal = {"symbol": "ETH-USDT-SWAP", "signal_type": "grid_open", "direction": "long"}
        result = adj.adjudicate(signal)

        assert result.passed is False
        assert result.blocked_layer == RiskLayer.L0_KILL_SWITCH
        # 即便被拦截，traceID 也已回写，可全程审计
        assert signal["trace_id"] == result.trace_id

    def test_close_signal_not_blocked_by_l0_via_adjudicator(self, tmp_path):
        rg = self._make_gate(tmp_path)
        adj = RiskAdjudicator(risk_gate=rg)
        rg.enable_kill_switch(reason="测试暂停")

        close_signal = {
            "symbol": "ETH-USDT-SWAP",
            "signal_type": "stop_loss",
            "direction": "close",
            "reduce_only": True,
        }
        result = adj.adjudicate(close_signal, is_close=True)

        assert result.blocked_layer != RiskLayer.L0_KILL_SWITCH

    def test_degraded_position_sync_blocks_open_but_allows_close(self):
        rg = RiskGate(config={})
        rg._kill_switch = type(
            "DisabledKillSwitch", (), {"is_enabled": lambda self: False}
        )()
        passing_result = RiskCheckResult(
            RiskLayer.L5_EMERGENCY, True, RiskAction.PASS, "ok"
        )
        rg._l5.check = lambda signal: passing_result
        rg._l4.check = lambda signal: RiskCheckResult(
            RiskLayer.L4_DAILY, True, RiskAction.PASS, "ok"
        )
        rg._l1.check = lambda signal: RiskCheckResult(
            RiskLayer.L1_PRE_TRADE, True, RiskAction.PASS, "ok"
        )
        rg._l2.check = lambda signal, market_data: RiskCheckResult(
            RiskLayer.L2_IN_TRADE, True, RiskAction.PASS, "ok"
        )
        rg._l3.check = lambda signal: []
        rg.set_position_manager(
            type("DegradedPositionManager", (), {
                "sync_degraded": True,
                "sync_fail_streak": 3,
            })()
        )

        open_result = rg.validate(
            {"symbol": "BTC-USDT-SWAP", "signal_type": "open_long"}
        )
        close_result = rg.validate(
            {
                "symbol": "BTC-USDT-SWAP",
                "signal_type": "stop_loss",
                "reduce_only": True,
            }
        )

        assert open_result.passed is False
        assert open_result.blocked_layer == RiskLayer.L1_PRE_TRADE
        assert open_result.results[0].details["position_sync_degraded"] is True
        assert close_result.passed is True


class TestAdjudicationResult:
    def test_to_dict(self):
        r = AdjudicationResult(
            trace_id="evt-1",
            passed=True,
            action=RiskAction.PASS,
            blocked_layer=None,
            symbol="ETH-USDT-SWAP",
        )
        d = r.to_dict()

        assert d["trace_id"] == "evt-1"
        assert d["passed"] is True
        assert d["action"] == "pass"
        assert d["blocked_layer"] is None
        assert d["symbol"] == "ETH-USDT-SWAP"
        assert "elapsed_ms" in d
        assert "adjudicated_at" in d

    def test_to_dict_with_blocked_layer(self):
        r = AdjudicationResult(
            trace_id="evt-2",
            passed=False,
            action=RiskAction.REJECT,
            blocked_layer=RiskLayer.L4_DAILY,
        )
        d = r.to_dict()
        assert d["action"] == "reject"
        assert d["blocked_layer"] == "L4_daily"