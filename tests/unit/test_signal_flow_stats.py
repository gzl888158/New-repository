from core.signal_flow_stats import SignalFlowStats, get_signal_flow_stats


def test_signal_flow_stats_tracks_stages_and_rejections_by_dimension():
    stats = SignalFlowStats()
    stats.record("candidate", strategy="trend")
    stats.record("published", strategy="trend")
    stats.record(
        "signal_rejected",
        strategy="trend",
        layer="regime_gate",
        reason="range_regime",
    )
    stats.record("exchange_open_accepted", strategy="trend")

    snapshot = stats.snapshot()

    assert snapshot["totals"] == {
        "candidate": 1,
        "published": 1,
        "signal_rejected": 1,
        "exchange_open_accepted": 1,
    }
    assert snapshot["by_strategy"]["trend"]["published"] == 1
    assert snapshot["rejections"]["by_layer"]["regime_gate"] == 1
    assert snapshot["rejections"]["by_reason"]["range_regime"] == 1
    assert snapshot["rejections"]["by_strategy"]["trend"]["regime_gate"] == 1


def test_redis_publish_records_candidate_and_published():
    from data.redis_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache._redis_available = True
    cache._signal_prefix = "signal:"

    class _Publisher:
        @staticmethod
        def publish(channel, message):
            return 1

    cache._redis = _Publisher()
    before = get_signal_flow_stats()["totals"].copy()

    assert cache.publish_signal({"strategy_name": "trend"}) is True

    after = get_signal_flow_stats()["totals"]
    assert after["candidate"] - before.get("candidate", 0) == 1
    assert after["published"] - before.get("published", 0) == 1


def test_redis_publish_records_serialization_failure():
    from data.redis_cache import RedisCache

    cache = RedisCache.__new__(RedisCache)
    cache._redis_available = True
    cache._signal_prefix = "signal:"
    before = get_signal_flow_stats()["totals"].copy()

    assert cache.publish_signal({"strategy_name": "trend", "bad": object()}) is False

    after = get_signal_flow_stats()["totals"]
    assert after["candidate"] - before.get("candidate", 0) == 1
    assert after["publish_failed"] - before.get("publish_failed", 0) == 1


def test_signal_processor_dead_letter_records_layer_and_reason():
    from services.signal_processor import SignalProcessor

    processor = SignalProcessor.__new__(SignalProcessor)
    processor._dead_letters = []
    processor._max_dead_letters = 10
    processor._event_bus = None
    before = get_signal_flow_stats()["totals"].get("signal_rejected", 0)

    processor._push_dead_letter(
        {"trace_id": "signal-1", "strategy_name": "trend"},
        "regime_gate:range regime blocks trend",
        layer="regime_gate",
        reason_code="range_bound",
    )

    snapshot = get_signal_flow_stats()
    assert snapshot["totals"]["signal_rejected"] - before == 1
    assert snapshot["rejections"]["by_strategy"]["trend"]["regime_gate"] >= 1


def test_strategy_filter_metric_records_source_rejection():
    from core.strategy_enterprise import EnterpriseStrategyMixin

    before = get_signal_flow_stats()["totals"].get("source_rejected", 0)

    EnterpriseStrategyMixin._record_signal_filter_metric(
        "trend_gate_rejected_total", {"reason": "min_confirmation"}
    )

    snapshot = get_signal_flow_stats()
    assert snapshot["totals"]["source_rejected"] - before == 1
    assert snapshot["rejections"]["by_layer"]["strategy_gate"] >= 1


def test_order_rejection_event_records_execution_layer():
    from execution.order_executor import OrderExecutor
    from core.unified_layer import EventType

    executor = OrderExecutor.__new__(OrderExecutor)
    executor._event_bus = None
    before = get_signal_flow_stats()["totals"].get("order_rejected", 0)

    executor._publish_event(EventType.ORDER_REJECTED, {
        "strategy_name": "grid",
        "layer": "trade_cost",
        "reason_code": "insufficient_volatility",
    })

    snapshot = get_signal_flow_stats()
    assert snapshot["totals"]["order_rejected"] - before == 1
    assert snapshot["rejections"]["by_layer"]["trade_cost"] >= 1