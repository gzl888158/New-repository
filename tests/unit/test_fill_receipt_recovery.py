"""
方案A（活跃订单落盘 + 全量对账补回执）回归测试
============================================
覆盖 fill 回执丢失根因修复的四个核心单元：
1. _persist_active_order / _remove_active_order：落盘与删除
2. _restore_active_orders：启动恢复 + 陈旧记录清理
3. _reconcile_fill_receipts：REST 全量对账补发回执 + fill_reconciled 幂等标记
4. get_execution_stats：观测指标暴露
"""

import os
import sys
import asyncio
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from execution.order_executor import OrderExecutor


def load_config():
    import yaml
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("execution", {})["persist_active_orders"] = True
    return cfg


def make_executor(config=None, sqlite_storage=None, okx_client=None):
    cfg = config or load_config()
    return OrderExecutor(
        cfg,
        okx_client or MagicMock(),
        MagicMock(),
        sqlite_storage if sqlite_storage is not None else MagicMock(),
    )


def test_persist_active_order_enabled():
    sqlite = MagicMock()
    ex = make_executor(sqlite_storage=sqlite)
    ex._persist_active_order("ord_1", {"symbol": "BTC-USDT", "status": "pending"})
    sqlite.save_active_order.assert_called_once_with(
        "ord_1", {"symbol": "BTC-USDT", "status": "pending"}
    )


def test_persist_active_order_disabled_noop():
    cfg = load_config()
    cfg["execution"]["persist_active_orders"] = False
    sqlite = MagicMock()
    ex = make_executor(config=cfg, sqlite_storage=sqlite)
    ex._persist_active_order("ord_1", {"symbol": "BTC-USDT"})
    sqlite.save_active_order.assert_not_called()


def test_remove_active_order():
    sqlite = MagicMock()
    ex = make_executor(sqlite_storage=sqlite)
    ex._remove_active_order("ord_1")
    sqlite.delete_active_order.assert_called_once_with("ord_1")


def test_restore_active_orders_skips_stale():
    sqlite = MagicMock()
    sqlite.load_active_orders.return_value = {
        "ord_pending": {"symbol": "ETH-USDT", "status": "pending"},
        "ord_filled": {"symbol": "ETH-USDT", "status": "filled"},
    }
    ex = make_executor(sqlite_storage=sqlite)
    # pending 订单被恢复到内存
    assert "ord_pending" in ex._active_orders
    # filled 陈旧记录被跳过并从落盘清理
    assert "ord_filled" not in ex._active_orders
    sqlite.delete_active_order.assert_called_with("ord_filled")


def test_reconcile_fill_receipts_recovers_lost_fill():
    okx = MagicMock()
    okx.get_order_history.return_value = [
        {"ordId": "ord_lost", "state": "filled", "avgPx": "100.5", "fillSz": "1"}
    ]
    ex = make_executor(okx_client=okx)
    ex._active_orders["ord_lost"] = {
        "symbol": "BTC-USDT",
        "direction": "long",
        "pos_side": "long",
        "signal_type": "open",
        "strategy_name": "grid",
        "quantity": 1.0,
        "clOrdId": "cl_1",
        "reduce_only": False,
    }
    received = []
    ex.register_fill_callback(lambda payload: received.append(payload))

    asyncio.run(ex._reconcile_fill_receipts())

    assert ex._fill_receipt_recovered == 1
    assert received, "应补发一条 fill 回执"
    assert received[0]["exchange_order_id"] == "ord_lost"
    # 幂等：再次对账不应重复补发
    asyncio.run(ex._reconcile_fill_receipts())
    assert ex._fill_receipt_recovered == 1


def test_reconcile_skips_without_get_order_history():
    okx = MagicMock(spec=[])  # 无 get_order_history 方法
    ex = make_executor(okx_client=okx)
    ex._active_orders["ord_x"] = {"symbol": "BTC-USDT"}
    asyncio.run(ex._reconcile_fill_receipts())
    assert ex._fill_receipt_recovered == 0


def test_execution_stats_exposes_metrics():
    ex = make_executor()
    stats = ex.get_execution_stats()
    assert stats["persist_active_orders"] is True
    assert stats["fill_receipt_recovered"] == 0