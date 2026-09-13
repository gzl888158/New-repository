"""P1 企业级升级：事件溯源重放单元测试。

覆盖：
1. 订单生命周期重建（placed → filled → closed）
2. 对账归类（开仓持仓 / 在途 / 已平）
3. 悬单检测（已下单长时间未成交）
4. 非订单事件被忽略
5. 平仓事件缺失开仓事件（窗口截断）不计入持仓
"""

from datetime import datetime, timedelta

import pytest

from core.event_store import EventStore
from core.event_replay import EventReplayer


class TestEventReplayer:
    def _make(self, tmp_path):
        return EventReplayer(event_store=EventStore(data_dir=str(tmp_path / "events")))

    def _append(self, store, etype, t, **data):
        store.store.append(etype, data, timestamp=t)

    def test_lifecycle_reconstruction(self, tmp_path):
        rp = self._make(tmp_path)
        t0 = datetime.now() - timedelta(minutes=10)
        t1 = t0 + timedelta(seconds=5)
        t2 = t1 + timedelta(seconds=5)

        self._append(rp, "order_placed", t0, exchange_order_id="o1",
                     symbol="ETH-USDT-SWAP", direction="long", strategy_name="grid",
                     reduce_only=False, trace_id="tr-1")
        self._append(rp, "order_filled", t1, exchange_order_id="o1",
                     symbol="ETH-USDT-SWAP", reduce_only=False, trace_id="tr-1")
        self._append(rp, "position_closed", t2, exchange_order_id="o1",
                     symbol="ETH-USDT-SWAP", reduce_only=False, trace_id="tr-1")

        orders = rp.replay_order_state()
        assert set(orders.keys()) == {"o1"}
        o = orders["o1"]
        assert o["placed_at"] is not None
        assert o["filled_at"] is not None
        assert o["closed_at"] is not None
        assert o["closed_reason"] == "close"
        assert o["strategy_name"] == "grid"

    def test_open_position_and_in_flight(self, tmp_path):
        rp = self._make(tmp_path)
        now = datetime.now()

        # 持仓：已下单已成交，未平仓
        self._append(rp, "order_placed", now - timedelta(minutes=5), exchange_order_id="op",
                     symbol="BTC-USDT-SWAP", direction="short", strategy_name="scalp",
                     reduce_only=False)
        self._append(rp, "order_filled", now - timedelta(minutes=4), exchange_order_id="op",
                     symbol="BTC-USDT-SWAP", reduce_only=False)

        # 在途：已下单未成交
        self._append(rp, "order_placed", now - timedelta(seconds=30), exchange_order_id="fl",
                     symbol="SOL-USDT-SWAP", direction="long", strategy_name="trend",
                     reduce_only=False)

        report = rp.reconcile(orphan_timeout_seconds=600)
        assert report["summary"]["open_positions"] == 1
        assert report["summary"]["in_flight"] == 1
        assert report["summary"]["orphan_placed_no_fill"] == 0

    def test_orphan_detection(self, tmp_path):
        rp = self._make(tmp_path)
        # 1 小时前下单，至今未成交 → 悬单
        self._append(rp, "order_placed", datetime.now() - timedelta(hours=1),
                     exchange_order_id="orphan", symbol="ETH-USDT-SWAP",
                     direction="long", strategy_name="grid", reduce_only=False)

        report = rp.reconcile(orphan_timeout_seconds=300)
        assert report["summary"]["orphan_placed_no_fill"] == 1
        assert report["orphan_placed_no_fill"][0]["exchange_order_id"] == "orphan"

    def test_non_order_events_ignored(self, tmp_path):
        rp = self._make(tmp_path)
        self._append(rp, "signal_generated", datetime.now(), symbol="ETH-USDT-SWAP")
        self._append(rp, "risk_event", datetime.now(), symbol="ETH-USDT-SWAP")
        self._append(rp, "config_changed", datetime.now())

        report = rp.reconcile()
        assert report["total_orders_reconstructed"] == 0
        assert report["summary"]["placed"] == 0

    def test_close_without_place(self, tmp_path):
        rp = self._make(tmp_path)
        # 仅有平仓事件，缺少开仓事件（重放窗口截断）：不应判为持仓
        self._append(rp, "position_closed", datetime.now(), exchange_order_id="c1",
                     symbol="ETH-USDT-SWAP", reduce_only=True)

        report = rp.reconcile()
        assert report["summary"]["open_positions"] == 0
        assert report["summary"]["orphan_closed_no_place"] == 1

    def test_cancelled_order_terminated(self, tmp_path):
        """下单后撤单（order_cancelled）→ 订单终结，不再计入在途/悬单。"""
        rp = self._make(tmp_path)
        now = datetime.now()
        self._append(rp, "order_placed", now - timedelta(minutes=30), exchange_order_id="c1",
                     symbol="ETH-USDT-SWAP", direction="long", strategy_name="grid",
                     reduce_only=False)
        self._append(rp, "order_cancelled", now - timedelta(minutes=29), exchange_order_id="c1",
                     symbol="ETH-USDT-SWAP", reason="pending_tier_timeout")

        report = rp.reconcile(orphan_timeout_seconds=600)
        # 已撤单订单不应被误判为超时悬单或仍在途
        assert report["summary"]["orphan_placed_no_fill"] == 0
        assert report["summary"]["in_flight"] == 0
        # 撤单订单归类为已关闭，原因=cancelled
        orders = rp.replay_order_state()
        assert orders["c1"]["closed_at"] is not None
        assert orders["c1"]["closed_reason"] == "cancelled"