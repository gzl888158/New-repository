"""
51169 幽灵仓位清理回归测试（模块 6 高风险用例）
============================================

覆盖 execution.order_executor.OrderExecutor._reconcile_positions 的幽灵仓清理逻辑：

历史 bug 点（务必防止回归）：
- P2 修复：仍在 _active_orders 中的 open 记录（限价单尚未成交）必须跳过，不能误判为幽灵仓
- 交易所实际存在持仓的记录不得被标记关闭
- 幽灵清理一律不写 pnl（不估算），交由 PnLReconciler 用 OKX 平仓账单精确对账，避免用 mark_price 估算出假数据

通过 object.__new__ 绕过 __init__，注入 okx_client / redis_cache / sqlite_storage 桩，
用 asyncio.run 直接调用异步方法，断言 update_trade_record 的调用结果。
"""

import asyncio
import sys
import os
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from execution.order_executor import OrderExecutor


class FakeStorage:
    """拦截 get_trade_records_by_status / update_trade_record / 持仓历史快照。"""

    def __init__(self, open_records, mark_price=None):
        self.open_records = open_records
        self.mark_price = mark_price
        self.updates = []

    def get_trade_records_by_status(self, status):
        return self.open_records

    def get_latest_position_history(self, symbol, side):
        if self.mark_price is None:
            return None
        return {"mark_price": self.mark_price}

    def update_trade_record(self, rec_id, update):
        self.updates.append((rec_id, update))

    def save_position_history(self, data):
        pass

    def save_account_history(self, data):
        pass

    def get_latest_open_record(self, symbol):
        return None


class FakeRedis:
    def set_position(self, p):
        pass

    def set_account_info(self, a):
        pass


def _make_position(symbol="BTC-USDT-SWAP", side="long", qty=0.01):
    return SimpleNamespace(
        symbol=symbol, side=side, quantity=qty, avg_cost=65000.0,
        mark_price=65000.0, unrealized_pnl=0.0, margin=650.0,
        leverage=1.0, timestamp="2026-08-21T00:00:00",
    )


def _build_executor(okx_positions, parse_position, open_records, active_orders, mark_price=None):
    executor = OrderExecutor.__new__(OrderExecutor)
    executor.okx_client = SimpleNamespace(
        get_positions=lambda: okx_positions,
        _parse_position=parse_position,
        get_account_info=lambda: None,
    )
    storage = FakeStorage(open_records, mark_price)
    executor.redis_cache = FakeRedis()
    executor.sqlite_storage = storage
    executor._active_orders = set(active_orders)
    # 使 _recover_stop_losses_after_restart 对已存在持仓直接跳过
    executor._position_strategy_map = {"BTC-USDT-SWAP": "grid"}
    executor._stop_managers = {}
    executor._position_entry_time = {}
    return executor, storage


def _run_reconcile(executor):
    asyncio.run(executor._reconcile_positions())


class TestGhostPositionCleanup:
    def test_active_order_is_skipped_not_ghost_closed(self):
        """P2：仍在挂单中的 open 记录不得误判为幽灵仓"""
        executor, storage = _build_executor(
            okx_positions=[],
            parse_position=lambda pd: None,
            open_records=[
                {"id": "ord_active", "symbol": "BTC-USDT-SWAP", "side": "buy", "quantity": "0.01"},
            ],
            active_orders={"ord_active"},
            mark_price=100.0,
        )
        _run_reconcile(executor)
        assert storage.updates == [], "挂单中的 open 记录不应被标记 ghost_close"

    def test_ghost_position_is_closed(self):
        """交易所已无持仓的 open 记录应被标记 ghost_close"""
        executor, storage = _build_executor(
            okx_positions=[],
            parse_position=lambda pd: None,
            open_records=[
                {"id": "ord_ghost", "symbol": "BTC-USDT-SWAP", "side": "sell", "quantity": "0.01", "filled_price": "65000"},
            ],
            active_orders=set(),
            mark_price=55000,
        )
        _run_reconcile(executor)
        assert len(storage.updates) == 1
        rec_id, update = storage.updates[0]
        assert rec_id == "ord_ghost"
        assert update["status"] == "closed"
        assert update["exit_reason"] == "ghost_close"
        assert "pnl" not in update, "幽灵清理不得估算并写入 pnl，应交由 PnLReconciler 用 OKX 平仓账单对账"

    def test_existing_exchange_position_not_closed(self):
        """交易所实际存在的持仓不得被标记关闭"""
        position = _make_position(side="long", qty=0.01)
        executor, storage = _build_executor(
            okx_positions=[{"instId": "BTC-USDT-SWAP"}],
            parse_position=lambda pd: position,
            open_records=[
                {"id": "ord_keep", "symbol": "BTC-USDT-SWAP", "side": "buy", "quantity": "0.01"},
            ],
            active_orders=set(),
            mark_price=100.0,
        )
        _run_reconcile(executor)
        assert storage.updates == [], "交易所仍有持仓的记录不应被 ghost_close"

    def test_proportional_pnl_allocation(self):
        """同 symbol:side 多条幽灵记录均关闭，且一律不估算 pnl（交 PnLReconciler 对账）"""
        executor, storage = _build_executor(
            okx_positions=[],
            parse_position=lambda pd: None,
            open_records=[
                {"id": "g1", "symbol": "BTC-USDT-SWAP", "side": "buy", "quantity": "1", "filled_price": "65000"},
                {"id": "g2", "symbol": "BTC-USDT-SWAP", "side": "buy", "quantity": "2", "filled_price": "65000"},
            ],
            active_orders=set(),
            mark_price=65100,
        )
        _run_reconcile(executor)
        assert len(storage.updates) == 2
        for rec_id, update in storage.updates:
            assert update["status"] == "closed"
            assert update["exit_reason"] == "ghost_close"
            assert "pnl" not in update, "幽灵清理不得估算并写入 pnl，应交由 PnLReconciler 用 OKX 平仓账单对账"

    def test_no_pnl_history_keeps_none(self):
        """无法取得持仓盈亏快照时 pnl 保持 None，不写 0 假数据"""
        executor, storage = _build_executor(
            okx_positions=[],
            parse_position=lambda pd: None,
            open_records=[
                {"id": "ord_nopnl", "symbol": "BTC-USDT-SWAP", "side": "buy", "quantity": "0.01"},
            ],
            active_orders=set(),
            mark_price=None,
        )
        _run_reconcile(executor)
        assert len(storage.updates) == 1
        rec_id, update = storage.updates[0]
        assert update["status"] == "closed"
        assert update.get("pnl") is None
