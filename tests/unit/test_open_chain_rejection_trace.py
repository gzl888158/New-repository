"""开仓全链路企业级优化 — 拒单可溯源 单元测试。

覆盖：
1. EventType 新增 SIGNAL_REJECTED / ORDER_REJECTED
2. SignalProcessor._classify_reject_reason 原因归一化
3. SignalProcessor._push_dead_letter 发布 SIGNAL_REJECTED（trace_id 复用 + 缺失生成回写）
4. LocalEventBus + EventStore 端到端持久化与重放
5. OrderExecutor._publish_event 发布 ORDER_REJECTED（执行层四拒单点共享埋点）
"""

import pytest

from core.unified_layer import Event, EventType, LocalEventBus
from core.event_store import EventStore
from core.event_id import EventIDGenerator


class RecordingBus:
    """内存事件总线假件：记录 publish_sync 发布的事件，避免落盘副作用。"""

    def __init__(self):
        self.events = []

    def publish_sync(self, event):
        self.events.append(event)

    async def publish(self, event):
        self.events.append(event)


def _make_signal_processor(bus):
    """用 object.__new__ 绕过 SignalProcessor 重型 __init__，仅装配拒单溯源所需属性。"""
    from services.signal_processor import SignalProcessor
    sp = object.__new__(SignalProcessor)
    sp._dead_letters = []
    sp._max_dead_letters = 100
    sp._event_bus = bus
    return sp


def _make_order_executor(bus):
    """用 object.__new__ 绕过 OrderExecutor 重型 __init__，仅装配事件发布所需属性。"""
    from execution.order_executor import OrderExecutor
    oe = object.__new__(OrderExecutor)
    oe._event_bus = bus
    return oe


# ── EventType 枚举 ─────────────────────────────────────────

def test_new_event_types_exist():
    assert EventType.SIGNAL_REJECTED.value == "signal_rejected"
    assert EventType.ORDER_REJECTED.value == "order_rejected"


# ── 原因归一化 ─────────────────────────────────────────

@pytest.mark.parametrize(
    "reason,expected_layer",
    [
        ("signal_quality_rejected", "signal_quality"),
        ("strategy_priority_blocked", "strategy_priority"),
        ("signal_conflict", "signal_conflict"),
        ("conflict_resolution_failed", "signal_conflict"),
        ("collaborative_trigger_blocked", "coordinator"),
        ("capital_focus_insufficient", "capital_allocation"),
        ("decision_validation_invalid", "decision_validation"),
        ("meta_abstain:foo", "meta_decision"),
        ("cba_rejected:profit_too_low", "cost_benefit"),
        ("below_threshold", "threshold"),
        ("risk_gate_exception_trading_paused", "risk_gate"),
        ("trading_paused", "risk_gate"),
        ("strategy_risk_validation_failed", "risk_gate"),
        ("regime_gate:strong_trend_down", "regime_gate"),
        ("something_unmapped", "signal_processor"),
    ],
)
def test_classify_reject_reason(reason, expected_layer):
    from services.signal_processor import SignalProcessor
    layer, code = SignalProcessor._classify_reject_reason(reason)
    assert layer == expected_layer
    assert code == reason  # reason_code 默认用全量兜底


# ── SignalProcessor 拒单发布 ─────────────────────────────────────────

def test_push_dead_letter_reuses_trace_id():
    bus = RecordingBus()
    sp = _make_signal_processor(bus)
    signal = {
        "symbol": "BTC-USDT-SWAP",
        "strategy_name": "grid",
        "signal_type": "entry",
        "direction": "long",
        "trace_id": "evt-upstream-001",
    }
    sp._push_dead_letter(signal, "signal_quality_rejected")

    assert len(bus.events) == 1
    evt = bus.events[0]
    assert evt.event_type == EventType.SIGNAL_REJECTED
    assert evt.data["trace_id"] == "evt-upstream-001"
    assert evt.data["layer"] == "signal_quality"
    assert evt.data["reason_code"] == "signal_quality_rejected"
    assert evt.data["symbol"] == "BTC-USDT-SWAP"
    assert evt.data["is_close"] is False


def test_push_dead_letter_generates_and_writes_back_trace_id():
    bus = RecordingBus()
    sp = _make_signal_processor(bus)
    signal = {"symbol": "ETH-USDT-SWAP", "strategy_name": "trend", "signal_type": "entry", "direction": "short"}
    sp._push_dead_letter(signal, "below_threshold")

    assert signal.get("trace_id") is not None  # 回写到 signal
    assert signal["trace_id"].startswith("evt-")
    assert bus.events[0].data["trace_id"] == signal["trace_id"]


def test_push_dead_letter_marks_close_signal():
    bus = RecordingBus()
    sp = _make_signal_processor(bus)
    sp._push_dead_letter({"symbol": "DOGE-USDT-SWAP", "signal_type": "close", "direction": "long"}, "signal_conflict")
    assert bus.events[0].data["is_close"] is True


def test_push_dead_letter_silent_when_no_bus():
    sp = _make_signal_processor(None)  # 未注入事件总线
    sp._push_dead_letter({"symbol": "X", "signal_type": "entry"}, "xyz")  # 不应抛异常
    assert len(sp._dead_letters) == 1  # 原有死信逻辑仍工作


# ── 端到端持久化与重放 ─────────────────────────────────────────

def test_signal_rejected_persisted_and_replayable(tmp_path):
    bus = LocalEventBus()
    store = EventStore(data_dir=str(tmp_path / "events"))
    bus.set_event_store(store)

    sp = _make_signal_processor(bus)
    sp._push_dead_letter(
        {"symbol": "SOL-USDT-SWAP", "strategy_name": "scalping", "signal_type": "entry", "direction": "long"},
        "signal_quality_rejected",
    )

    records = store.replay()
    rejected = [r for r in records if r["event_type"] == "signal_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["data"]["layer"] == "signal_quality"
    assert rejected[0]["data"]["trace_id"].startswith("evt-")


# ── OrderExecutor 执行层拒单发布 ─────────────────────────────────────────

def test_order_executor_publishes_order_rejected():
    bus = RecordingBus()
    oe = _make_order_executor(bus)
    oe._publish_event(EventType.ORDER_REJECTED, {
        "trace_id": "evt-exec-002",
        "symbol": "ADA-USDT-SWAP",
        "strategy": "trend",
        "signal_type": "entry",
        "direction": "long",
        "layer": "trade_auditor",
        "reason_code": "audit_blocked",
        "reason": "signal quality too low",
        "is_close": False,
    })

    assert len(bus.events) == 1
    assert bus.events[0].event_type == EventType.ORDER_REJECTED
    assert bus.events[0].data["layer"] == "trade_auditor"
    assert bus.events[0].data["trace_id"] == "evt-exec-002"


def test_order_executor_publish_skips_when_no_bus():
    oe = _make_order_executor(None)
    oe._publish_event(EventType.ORDER_REJECTED, {"symbol": "X"})  # 不应抛异常