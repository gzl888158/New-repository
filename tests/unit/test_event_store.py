"""P1 企业级升级：持久化事件溯源存储单元测试。

覆盖：
1. append + replay 基本读写
2. event_type / 时间范围过滤
3. tail 最近 n 条
4. count 统计
5. 跨实例持久化（重启后重放）
6. 保留期清理 prune
7. LocalEventBus 持久化集成（注入 EventStore 后 publish 落盘）
"""

from datetime import datetime, timedelta

import pytest

from core.event_store import EventStore
from core.unified_layer import Event, EventType, LocalEventBus


class TestEventStore:
    def _make_store(self, tmp_path):
        return EventStore(data_dir=str(tmp_path / "events"))

    def test_append_and_replay(self, tmp_path):
        store = self._make_store(tmp_path)
        store.append("order_placed", {"symbol": "ETH-USDT-SWAP", "qty": 1})
        store.append("order_filled", {"symbol": "ETH-USDT-SWAP", "price": 3000.0})

        records = store.replay()
        assert len(records) == 2
        assert records[0]["event_type"] == "order_placed"
        assert records[1]["event_type"] == "order_filled"
        assert all(r["event_id"].startswith("evt-") for r in records)

    def test_event_type_filter(self, tmp_path):
        store = self._make_store(tmp_path)
        store.append("order_placed", {"symbol": "ETH"})
        store.append("risk_event", {"symbol": "ETH"})

        only_orders = store.replay(event_type="order_placed")
        assert len(only_orders) == 1
        assert only_orders[0]["event_type"] == "order_placed"

    def test_time_range_filter(self, tmp_path):
        store = self._make_store(tmp_path)
        old = datetime.now() - timedelta(hours=2)
        new = datetime.now()
        store.append("order_placed", {"v": 1}, timestamp=old)
        store.append("order_filled", {"v": 2}, timestamp=new)

        cutoff = datetime.now() - timedelta(hours=1)
        recent = store.replay(from_ts=cutoff)
        assert len(recent) == 1
        assert recent[0]["event_type"] == "order_filled"

    def test_tail(self, tmp_path):
        store = self._make_store(tmp_path)
        for i in range(5):
            store.append("tick", {"i": i})

        tail = store.tail(2)
        assert len(tail) == 2
        assert tail[0]["data"]["i"] == 3
        assert tail[1]["data"]["i"] == 4

    def test_count(self, tmp_path):
        store = self._make_store(tmp_path)
        store.append("a", {})
        store.append("a", {})
        store.append("b", {})

        assert store.count() == 3
        assert store.count("a") == 2
        assert store.count("b") == 1

    def test_persistence_across_instances(self, tmp_path):
        path = str(tmp_path / "events")
        store1 = EventStore(data_dir=path)
        store1.append("order_placed", {"symbol": "ETH"})

        # 模拟进程重启：新实例从同一目录重放
        store2 = EventStore(data_dir=path)
        assert store2.count() == 1
        records = store2.replay()
        assert records[0]["event_type"] == "order_placed"
        assert records[0]["symbol"] == "ETH"

    def test_prune_retention(self, tmp_path):
        path = str(tmp_path / "events")
        store = EventStore(data_dir=path, retention_days=7)

        # 构造 10 天前的过期文件
        old_day = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        old_file = store._file_for_day(old_day)
        with open(old_file, "w", encoding="utf-8") as f:
            f.write('{"event_id": "evt-old", "event_type": "x", "timestamp": "2020-01-01T00:00:00", "data": {}}\n')

        removed = store.prune()
        assert removed == 1
        import os
        assert not os.path.exists(old_file)


class TestLocalEventBusPersistence:
    def test_persist_on_publish(self, tmp_path):
        store = EventStore(data_dir=str(tmp_path / "events"))
        bus = LocalEventBus()
        bus.set_event_store(store)

        bus.publish_sync(Event(EventType.ORDER_PLACED, {"symbol": "ETH-USDT-SWAP"}))
        bus.publish_sync(Event(EventType.ORDER_FILLED, {"symbol": "ETH-USDT-SWAP"}))

        assert store.count() == 2
        records = store.replay()
        assert records[0]["event_type"] == "order_placed"
        assert records[0]["symbol"] == "ETH-USDT-SWAP"

    def test_default_no_store_no_error(self):
        bus = LocalEventBus()
        # 未注入 store 时 publish 不应抛异常（向后兼容）
        bus.publish_sync(Event(EventType.SYSTEM_HEALTH, {"status": "ok"}))
        assert bus._event_store is None