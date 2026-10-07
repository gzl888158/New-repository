"""
P2 防御性/健壮性修复的针对性回归测试。

覆盖清单（对应 P2 企业级强化项）：
  1. capital_manager.rebalance_pools：每日结算保留 used_amount，仅清空 locked_amount
  2. pnl_reconciler._reconcile_account_pnl：动态基准 + 出入金检测，不再用静态 total_capital
  3. trade_journal._try_close_from_trade_records：平仓后关闭 trade_records 的 open 记录
  4. trade_journal._add_to_position：加仓占用策略已用资金
  5. notification_dispatcher：主通道失败后 fallback 降级
  6. auto_recovery._action_reset_state：跳过人工暂停策略，非暂停正确回到 IDLE
"""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.capital_manager import CapitalPoolController, CapitalPoolType
from core.notification_dispatcher import (
    NotificationDispatcher,
    Notification,
    NotificationChannel,
    NotificationPriority,
    ChannelConfig,
    DeliveryStatus,
    DeliveryRecord,
)
from core.trade_journal import TradeJournal, PositionSnapshot, TradeRecord
from core.risk_monitor import RealTimeRiskMonitor, RiskAlertLevel
from risk.pnl_reconciler import PnLReconciler
from monitoring.auto_recovery import AutoRecovery, FailureType
from services.strategy_coordinator import StrategyState


# ═══════════════════════════════════════════════════════════════
# 1. CapitalPoolController.rebalance_pools
# ═══════════════════════════════════════════════════════════════

class TestRebalancePoolsPreservesUsedAmount:
    def test_used_amount_preserved_locked_cleared(self):
        ctrl = CapitalPoolController({"trading": {"total_capital": 100.0}})
        base = ctrl.get_pool(CapitalPoolType.BASE)
        base.used_amount = 30.0   # 在仓保证金
        base.locked_amount = 5.0  # 挂单锁定

        ctrl.rebalance_pools()

        # total_amount 重置为目标比例（100 * 0.6 = 60）
        assert base.total_amount == pytest.approx(60.0)
        # 在仓保证金必须保留
        assert base.used_amount == pytest.approx(30.0)
        # P1-1: locked_amount 由 reconcile_locked_capital() 对账修正，rebalance 不清零
        assert base.locked_amount == pytest.approx(5.0)

    def test_used_amount_clamped_to_target(self):
        ctrl = CapitalPoolController({"trading": {"total_capital": 100.0}})
        base = ctrl.get_pool(CapitalPoolType.BASE)
        base.used_amount = 80.0  # 超过目标 60

        ctrl.rebalance_pools()

        # 保留但不得超过池总额
        assert base.used_amount == pytest.approx(60.0)
        assert base.locked_amount == 0.0

    def test_all_pools_locked_reset(self):
        ctrl = CapitalPoolController({"trading": {"total_capital": 100.0}})
        add = ctrl.get_pool(CapitalPoolType.ADD_POSITION_RESERVE)
        add.used_amount = 10.0
        add.locked_amount = 3.0

        ctrl.rebalance_pools()

        assert add.total_amount == pytest.approx(25.0)  # 100 * 0.25
        assert add.used_amount == pytest.approx(10.0)
        # P1-1: locked_amount 由 reconcile_locked_capital() 对账修正，rebalance 不清零
        assert add.locked_amount == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════
# 2. PnLReconciler._reconcile_account_pnl（动态基准 + 出入金）
# ═══════════════════════════════════════════════════════════════

def _make_reconciler(db_total_pnl, prev=None, account=None, total_capital=1000):
    storage = MagicMock()
    # 已实现盈亏口径：数据完整性修复后以 TradeJournal trades.pnl_usdt 权威账本为准
    storage.get_authoritative_realized_pnl.return_value = float(db_total_pnl)
    storage.get_db_realized_pnl.return_value = float(db_total_pnl)
    storage.get_latest_pnl_reconciliation.return_value = prev
    storage.save_pnl_reconciliation = MagicMock()

    okx_client = MagicMock()
    okx_client.get_account_info.return_value = account or {"totalEq": "1100", "upl": "10"}

    reconciler = PnLReconciler(
        {"trading": {"total_capital": total_capital}},
        okx_client,
        storage,
    )
    return reconciler, storage


class TestAccountPnlReconciliation:
    async def test_first_run_uses_static_baseline(self):
        reconciler, storage = _make_reconciler(db_total_pnl=150.0)
        result = await reconciler._reconcile_account_pnl()

        # 首次对账：基准 = 配置 total_capital = 1000
        assert result["initial_capital"] == pytest.approx(1000.0)
        assert result["external_flow"] == pytest.approx(0.0)
        assert result["okx_total_pnl"] == pytest.approx(100.0)  # 1100 - 1000
        assert result["discrepancy"] == pytest.approx(-50.0)     # 100 - 150
        assert result["unattributed"] == pytest.approx(-60.0)    # -50 - 10
        # 快照已持久化
        storage.save_pnl_reconciliation.assert_called_once()

    async def test_external_flow_adjusts_baseline(self):
        prev = {
            "initial_capital": 1000.0,
            "okx_equity": 1050.0,
            "db_realized_pnl": 40.0,
            "unrealized_pnl": 5.0,
        }
        reconciler, storage = _make_reconciler(
            db_total_pnl=50.0,
            prev=prev,
            account={"totalEq": "1200", "upl": "10"},
        )
        result = await reconciler._reconcile_account_pnl()

        # equity_delta=150, realized_delta=10, unrealized_delta=5 → external_flow=135
        assert result["external_flow"] == pytest.approx(135.0)
        # 出入金超过阈值(6.0)，基准同步上移 → 1000 + 135 = 1135
        assert result["initial_capital"] == pytest.approx(1135.0)
        assert result["okx_total_pnl"] == pytest.approx(65.0)  # 1200 - 1135
        assert result["discrepancy"] == pytest.approx(15.0)     # 65 - 50
        assert result["unattributed"] == pytest.approx(5.0)     # 15 - 10

    async def test_small_delta_below_threshold_keeps_baseline(self):
        prev = {
            "initial_capital": 1000.0,
            "okx_equity": 1200.0,
            "db_realized_pnl": 50.0,
            "unrealized_pnl": 10.0,
        }
        reconciler, storage = _make_reconciler(
            db_total_pnl=50.0,
            prev=prev,
            account={"totalEq": "1200", "upl": "10"},
        )
        result = await reconciler._reconcile_account_pnl()

        # 无变化 → external_flow=0，基准不变
        assert result["external_flow"] == pytest.approx(0.0)
        assert result["initial_capital"] == pytest.approx(1000.0)

    async def test_unattributed_decomposed_into_funding_fee(self):
        reconciler, storage = _make_reconciler(db_total_pnl=150.0)
        # 资金费账单：两笔净额 -2.5 + 0.5 = -2.0（负=支付成本，正=收取收益）
        reconciler.okx_client.get_all_bills_paginated.return_value = [
            {"fee": "-2.5"},
            {"fee": "0.5"},
        ]
        result = await reconciler._reconcile_account_pnl()

        # 同 test_first_run：unattributed = -60.0；资金费入账 -2.0
        assert result["funding_fee"] == pytest.approx(-2.0)
        assert result["slippage_spread"] == pytest.approx(-58.0)  # -60 - (-2.0)
        # 分解保持 additive：unattributed = funding_fee + slippage_spread
        assert result["unattributed"] == pytest.approx(
            result["funding_fee"] + result["slippage_spread"]
        )

    async def test_funding_fee_fetch_graceful_on_nonlist(self):
        # MagicMock 默认返回非 list，应 fail-open 回退 0.0，不抛异常
        reconciler, storage = _make_reconciler(db_total_pnl=150.0)
        reconciler.okx_client.get_all_bills_paginated.return_value = None
        result = await reconciler._reconcile_account_pnl()
        assert result["funding_fee"] == pytest.approx(0.0)
        assert result["slippage_spread"] == pytest.approx(result["unattributed"])


# ═══════════════════════════════════════════════════════════════
# 3. TradeJournal._try_close_from_trade_records（关闭 open 记录）
# ═══════════════════════════════════════════════════════════════

def _make_minimal_journal(storage):
    journal = object.__new__(TradeJournal)
    journal.sqlite_storage = storage
    journal._trades = {}
    journal._wins = 0
    journal._losses = 0
    journal._realized_pnl = 0.0
    journal._current_equity = 1000.0
    journal._total_fees_paid = 0.0
    return journal


class TestCloseFromTradeRecords:
    def _storage(self, open_records):
        storage = MagicMock()
        storage.get_all_open_records.return_value = open_records
        storage.update_trade_record = MagicMock()
        conn = MagicMock()
        storage.get_connection.return_value = conn
        return storage

    async def test_close_closes_open_record(self):
        open_rec = {
            "id": "open-123",
            "symbol": "BTC-USDT-SWAP",
            "side": "long",
            "price": "60000",
            "quantity": "0.01",
            "leverage": 6,
            "create_time": "2026-09-08T10:00:00",
            "strategy_name": "trend",
            "order_type": "open_long",
        }
        storage = self._storage([open_rec])
        journal = _make_minimal_journal(storage)

        result = await journal._try_close_from_trade_records(
            trade_id="trade-1",
            symbol="BTC-USDT-SWAP",
            direction="short",  # 与 open 记录方向相反 → 平仓
            price=61000.0,
            quantity=0.01,
            leverage=6,
            fees=1.0,
            signal_type="close",
            exit_reason="take_profit",
        )

        assert result is True
        assert "trade-1" in journal._trades
        # 关键：open 记录被关闭
        storage.update_trade_record.assert_called_once()
        call_args = storage.update_trade_record.call_args
        assert call_args[0][0] == "open-123"
        assert call_args[0][1]["status"] == "closed"
        assert call_args[0][1]["exit_reason"] == "take_profit"

    async def test_same_direction_not_close(self):
        open_rec = {
            "id": "open-123",
            "symbol": "BTC-USDT-SWAP",
            "side": "long",
            "price": "60000",
            "quantity": "0.01",
            "leverage": 6,
            "create_time": "2026-09-08T10:00:00",
            "strategy_name": "trend",
            "order_type": "open_long",
        }
        storage = self._storage([open_rec])
        journal = _make_minimal_journal(storage)

        result = await journal._try_close_from_trade_records(
            trade_id="trade-1",
            symbol="BTC-USDT-SWAP",
            direction="long",  # 方向相同 → 不是平仓
            price=61000.0,
            quantity=0.01,
            leverage=6,
            fees=1.0,
            signal_type="close",
            exit_reason="close",
        )

        assert result is False
        assert "trade-1" not in journal._trades
        storage.update_trade_record.assert_not_called()

    async def test_no_matching_open_record(self):
        storage = self._storage([])
        journal = _make_minimal_journal(storage)

        result = await journal._try_close_from_trade_records(
            trade_id="trade-1",
            symbol="BTC-USDT-SWAP",
            direction="short",
            price=61000.0,
            quantity=0.01,
            leverage=6,
            fees=1.0,
            signal_type="close",
            exit_reason="close",
        )

        assert result is False
        storage.update_trade_record.assert_not_called()


# ═══════════════════════════════════════════════════════════════
# 4. TradeJournal._add_to_position（加仓占用策略资金）
# ═══════════════════════════════════════════════════════════════

class TestAddToPositionReservesCapital:
    def _snapshot(self, margin=100.0):
        return PositionSnapshot(
            timestamp=None,
            symbol="BTC-USDT-SWAP",
            strategy_name="trend",
            direction="long",
            quantity=0.01,
            avg_cost=60000.0,
            mark_price=60000.0,
            unrealized_pnl=0.0,
            margin=margin,
            leverage=6,
            signal_type="open",
        )

    def _journal(self, snapshot, allocated=1000.0, used=50.0):
        journal = object.__new__(TradeJournal)
        journal._open_positions = {"BTC-USDT-SWAP": snapshot}
        journal._strategy_allocations = {"trend": allocated}
        journal._strategy_used_capital = {"trend": used}
        journal._current_equity = 1000.0
        return journal

    async def test_add_reserves_added_margin(self):
        snapshot = self._snapshot(margin=100.0)
        journal = self._journal(snapshot, allocated=1000.0, used=50.0)

        await journal._add_to_position(
            symbol="BTC-USDT-SWAP",
            price=61000.0,
            quantity=0.01,
            leverage=6,
            fees=1.0,
            signal_type="add",
        )

        # 新保证金 = 0.02 * 60500 / 6 ≈ 201.667，加仓占用资金已增加
        assert snapshot.quantity == pytest.approx(0.02)
        assert snapshot.avg_cost == pytest.approx(60500.0)
        assert journal._strategy_used_capital["trend"] > 50.0

    async def test_insufficient_capital_skips_add(self):
        snapshot = self._snapshot(margin=100.0)
        journal = self._journal(snapshot, allocated=150.0, used=149.0)

        await journal._add_to_position(
            symbol="BTC-USDT-SWAP",
            price=61000.0,
            quantity=0.01,
            leverage=6,
            fees=1.0,
            signal_type="add",
        )

        # 加仓所需保证金(≈101.667)超过可用(1.0) → 加仓被拒绝，持仓不变
        assert snapshot.quantity == pytest.approx(0.01)
        assert journal._strategy_used_capital["trend"] == pytest.approx(149.0)


# ═══════════════════════════════════════════════════════════════
# 5. NotificationDispatcher 主通道失败 fallback 降级
# ═══════════════════════════════════════════════════════════════

class _FakeFailingChannel:
    config = ChannelConfig(channel=NotificationChannel.TELEGRAM, enabled=True)

    async def send(self, notification):
        return DeliveryRecord(
            message_id=notification.message_id,
            channel=notification.channel,
            status=DeliveryStatus.FAILED,
            error="boom",
        )


class _FakeSucceedingChannel:
    config = ChannelConfig(channel=NotificationChannel.CONSOLE, enabled=True)

    async def send(self, notification):
        return DeliveryRecord(
            message_id=notification.message_id,
            channel=notification.channel,
            status=DeliveryStatus.DELIVERED,
        )


class TestNotificationFallback:
    def _notification(self):
        return Notification(
            priority=NotificationPriority.CRITICAL.value,
            channel=NotificationChannel.TELEGRAM.value,
            title="test",
            message="hello",
            max_retries=0,
        )

    async def test_fallback_succeeds_via_console(self):
        dispatcher = NotificationDispatcher({})
        dispatcher._channels[NotificationChannel.TELEGRAM] = _FakeFailingChannel()
        dispatcher._channels[NotificationChannel.CONSOLE] = _FakeSucceedingChannel()
        dispatcher._channel_configs[NotificationChannel.TELEGRAM] = ChannelConfig(
            channel=NotificationChannel.TELEGRAM, enabled=True
        )
        dispatcher._channel_configs[NotificationChannel.CONSOLE] = ChannelConfig(
            channel=NotificationChannel.CONSOLE, enabled=True
        )

        records = []
        dispatcher.on_delivery(lambda rec: records.append(rec))

        await dispatcher._deliver(self._notification())

        # 最终投递记录应为 console 且 DELIVERED
        assert records[-1].status == DeliveryStatus.DELIVERED
        assert records[-1].channel == "console"

    async def test_fallback_all_fail(self):
        dispatcher = NotificationDispatcher({})
        dispatcher._channels[NotificationChannel.TELEGRAM] = _FakeFailingChannel()
        dispatcher._channels[NotificationChannel.CONSOLE] = _FakeFailingChannel()
        dispatcher._channel_configs[NotificationChannel.TELEGRAM] = ChannelConfig(
            channel=NotificationChannel.TELEGRAM, enabled=True
        )
        dispatcher._channel_configs[NotificationChannel.CONSOLE] = ChannelConfig(
            channel=NotificationChannel.CONSOLE, enabled=True
        )

        records = []
        dispatcher.on_delivery(lambda rec: records.append(rec))

        await dispatcher._deliver(self._notification())

        assert records[-1].status == DeliveryStatus.FAILED

    def test_fallback_channels_ordering(self):
        dispatcher = NotificationDispatcher({})
        dispatcher.configure_channel(
            NotificationChannel.WEBHOOK,
            ChannelConfig(channel=NotificationChannel.WEBHOOK, enabled=True),
        )
        dispatcher.configure_channel(
            NotificationChannel.EMAIL,
            ChannelConfig(channel=NotificationChannel.EMAIL, enabled=True),
        )

        result = dispatcher._fallback_channels(NotificationChannel.TELEGRAM)

        # Console 优先作为本地兜底，随后 webhook / email；主通道 telegram 被排除
        assert result == [
            NotificationChannel.CONSOLE,
            NotificationChannel.WEBHOOK,
            NotificationChannel.EMAIL,
        ]


# ═══════════════════════════════════════════════════════════════
# 6. AutoRecovery._action_reset_state（跳过人工暂停）
# ═══════════════════════════════════════════════════════════════

class TestAutoRecoveryResetState:
    def _recovery(self, coordinator):
        recovery = AutoRecovery({}, MagicMock())
        recovery.set_dependencies(strategy_coordinator=coordinator)
        return recovery

    async def test_paused_strategy_not_reset(self):
        coordinator = MagicMock()
        coordinator.get_strategy_state.return_value = StrategyState.PAUSED
        coordinator.set_strategy_state = MagicMock()

        recovery = self._recovery(coordinator)
        result = await recovery._action_reset_state(
            FailureType.STRATEGY, {"strategy_name": "grid"}
        )

        assert result is False
        coordinator.set_strategy_state.assert_not_called()

    async def test_non_paused_strategy_reset_to_idle(self):
        coordinator = MagicMock()
        coordinator.get_strategy_state.return_value = StrategyState.ERROR
        coordinator.set_strategy_state = MagicMock()

        recovery = self._recovery(coordinator)
        result = await recovery._action_reset_state(
            FailureType.STRATEGY, {"strategy_name": "grid"}
        )

        assert result is True
        coordinator.set_strategy_state.assert_called_once_with("grid", StrategyState.IDLE)

    async def test_batch_reset_skips_paused(self):
        coordinator = MagicMock()
        coordinator.get_strategy_state.side_effect = lambda name: (
            StrategyState.PAUSED if name == "grid" else StrategyState.ERROR
        )
        coordinator.get_all_strategy_states.return_value = {
            "grid": StrategyState.PAUSED,
            "trend": StrategyState.ERROR,
        }
        coordinator.set_strategy_state = MagicMock()

        recovery = self._recovery(coordinator)
        result = await recovery._action_reset_state(FailureType.STRATEGY, {})

        assert result is True
        # 只重置非暂停的 trend，跳过 grid
        coordinator.set_strategy_state.assert_called_once_with("trend", StrategyState.IDLE)


# ═══════════════════════════════════════════════════════════════
# 7. RealTimeRiskMonitor 保证金率/回撤查询失败 fail-closed
# ═══════════════════════════════════════════════════════════════

class TestRiskMonitorFailClosed:
    """保证金率/回撤查询失败时应区分『未知』并按保守高评级处理，而非误判为无风险"""

    def _monitor(self, account_manager=None, position_manager=None):
        monitor = RealTimeRiskMonitor({})
        monitor.inject_dependencies(
            account_manager=account_manager,
            position_manager=position_manager,
        )
        return monitor

    def test_margin_ratio_unknown_returns_emergency(self):
        am = MagicMock()
        am.get_account_info.return_value = None  # 查询失败 → 未知
        score = self._monitor(account_manager=am)._check_margin_risk()

        assert score.score == 1.0
        assert score.level == RiskAlertLevel.EMERGENCY
        assert score.details["unknown"] is True

    def test_margin_ratio_query_exception_returns_emergency(self):
        am = MagicMock()
        am.get_account_info.side_effect = Exception("boom")
        score = self._monitor(account_manager=am)._check_margin_risk()

        assert score.score == 1.0
        assert score.level == RiskAlertLevel.EMERGENCY

    def test_drawdown_unknown_returns_emergency(self):
        pm = MagicMock()
        pm.get_account_risk.return_value = None  # 查询失败 → 未知
        score = self._monitor(position_manager=pm)._check_drawdown_risk()

        assert score.score == 1.0
        assert score.level == RiskAlertLevel.EMERGENCY
        assert score.details["unknown"] is True

    def test_margin_ratio_computed_from_details_when_mgnratio_empty(self):
        """mgnRatio 为空（单币种 USDT 账户）时改用 details.frozenBal/eq 计算占用率，避免误报 EMERGENCY"""
        am = MagicMock()
        am.get_account_info.return_value = {
            "totalEq": "180.6",
            "mgnRatio": "",  # 单币种账户该字段为空
            "details": [
                {"ccy": "USDT", "eq": "180.6", "frozenBal": "18.06"},
            ],
        }
        score = self._monitor(account_manager=am)._check_margin_risk()

        assert score.score == pytest.approx(0.1)  # 18.06 / 180.6
        assert score.level == RiskAlertLevel.INFO
        assert score.details.get("unknown") is not True

    def test_margin_ratio_high_utilization_returns_emergency(self):
        """占用率超过 emergency 阈值时正确判定为 EMERGENCY"""
        am = MagicMock()
        am.get_account_info.return_value = {
            "totalEq": "100.0",
            "mgnRatio": "",
            "details": [
                {"ccy": "USDT", "eq": "100.0", "frozenBal": "90.0"},
            ],
        }
        score = self._monitor(account_manager=am)._check_margin_risk()

        assert score.score == pytest.approx(0.9)
        assert score.level == RiskAlertLevel.EMERGENCY

    def test_margin_ratio_uses_avail_eq_primary(self):
        """availEq 存在时用 eq-availEq（可用保证金扣除法）计算占用率，与 _parse_account_info 同口径"""
        am = MagicMock()
        am.get_account_info.return_value = {
            "totalEq": "215.0",
            "mgnRatio": "",
            "details": [
                {"ccy": "USDT", "eq": "215.0", "availEq": "170.0", "frozenBal": "0.0"},
            ],
        }
        score = self._monitor(account_manager=am)._check_margin_risk()

        assert score.score == pytest.approx(45.0 / 215.0)  # (eq - availEq) / eq
        assert score.level == RiskAlertLevel.INFO
        assert score.details.get("unknown") is not True


# ═══════════════════════════════════════════════════════════════
# 7. PnLReconciler baseline 漂移护栏（防 discrepancy 假象）
# ═══════════════════════════════════════════════════════════════

class TestBaselineDriftGuard:
    async def test_negative_baseline_reset_to_config(self):
        """prev 的 baseline 漂移到负数（历史失真）时，重置为配置 total_capital。"""
        prev = {
            "initial_capital": -672.0,
            "okx_equity": 173.0,
            "db_realized_pnl": -2.5,
            "unrealized_pnl": 0.0,
        }
        reconciler, storage = _make_reconciler(
            db_total_pnl=-2.5,
            prev=prev,
            account={"totalEq": "173.0", "upl": "0"},
            total_capital=231.93,
        )
        result = await reconciler._reconcile_account_pnl()

        # baseline 从负数 -672 重置回 231.93，okx_total_pnl 反映真实亏损而非假盈利
        assert result["initial_capital"] == pytest.approx(231.93)
        assert result["okx_total_pnl"] == pytest.approx(173.0 - 231.93)

    async def test_drifted_baseline_reset_to_config(self):
        """prev 的 baseline 与配置偏差超 50%（正数漂移）时，同样重置。"""
        prev = {
            "initial_capital": 2000.0,  # 相对 total_capital=1000 漂移 +100%
            "okx_equity": 2000.0,
            "db_realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
        }
        reconciler, storage = _make_reconciler(
            db_total_pnl=0.0,
            prev=prev,
            account={"totalEq": "2000", "upl": "0"},
            total_capital=1000,
        )
        result = await reconciler._reconcile_account_pnl()
        assert result["initial_capital"] == pytest.approx(1000.0)

    async def test_oversized_external_flow_rejected(self):
        """单次 external_flow 超过权益 50% 上限时，拒绝调整 baseline（db 失真误判保护）。"""
        prev = {
            "initial_capital": 1000.0,
            "okx_equity": 1000.0,
            "db_realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
        }
        reconciler, storage = _make_reconciler(
            db_total_pnl=0.0,
            prev=prev,
            account={"totalEq": "500", "upl": "0"},  # 权益暴跌 500，疑似出金
            total_capital=1000,
        )
        result = await reconciler._reconcile_account_pnl()

        # external_flow=-500 超过 max(500*0.5,100)=250，拒绝调整
        assert result["external_flow"] == pytest.approx(-500.0)
        assert result["initial_capital"] == pytest.approx(1000.0)


class TestFundingFeeBillType:
    def test_funding_fee_uses_type_8(self):
        """资金费账单类型应为 type=8（资金费），而非 type=7（扣息）。"""
        reconciler, storage = _make_reconciler(db_total_pnl=0.0)
        reconciler.okx_client.get_all_bills_paginated.return_value = []
        reconciler._fetch_funding_fee()

        reconciler.okx_client.get_all_bills_paginated.assert_called_once_with(
            bill_type="8", max_pages=3
        )


class TestCloseSubtypeFilter:
    def _rec(self, symbol="BTC-USDT-SWAP", side="long"):
        now = datetime.now()
        return {
            "id": "rec1",
            "symbol": symbol,
            "side": side,
            "create_time": now - timedelta(minutes=10),
            "close_time": now - timedelta(minutes=5),
            "exit_reason": "ghost_close",
            "_shard": "trade_records",
        }

    def _reconciler_for_closed(self):
        storage = MagicMock()
        storage.get_closed_records_missing_pnl.return_value = [self._rec()]
        storage.update_trade_in_shard.return_value = True
        okx_client = MagicMock()
        okx_client.get_all_bills_paginated.return_value = []
        return PnLReconciler({"trading": {"total_capital": 1000}}, okx_client, storage), storage

    async def test_strong_liquidation_close_long_matched(self):
        """强减平多（subType=100）应被识别为平多账单，回填 pnl。"""
        reconciler, storage = self._reconciler_for_closed()
        now = datetime.now()
        reconciler.okx_client.get_all_bills_paginated.return_value = [{
            "instId": "BTC-USDT-SWAP",
            "subType": "100",
            "ts": str(int((now - timedelta(minutes=5)).timestamp() * 1000)),
            "pnl": "-10.5",
            "fee": "-0.3",
            "fillPx": "50000",
            "sz": "0.1",
        }]

        result = await reconciler._reconcile_closed_records()

        assert result["corrected"] == 1
        assert result["failed"] == 0
        # net = pnl + fee = -10.8
        updates = storage.update_trade_in_shard.call_args[0][2]
        assert updates["pnl"] == pytest.approx(-10.8)

    async def test_non_close_subtype_not_matched(self):
        """自动减仓（subType=9）不是平仓账单，不应匹配回填。"""
        reconciler, storage = self._reconciler_for_closed()
        now = datetime.now()
        reconciler.okx_client.get_all_bills_paginated.return_value = [{
            "instId": "BTC-USDT-SWAP",
            "subType": "9",
            "ts": str(int((now - timedelta(minutes=5)).timestamp() * 1000)),
            "pnl": "-10.5",
            "fee": "-0.3",
            "fillPx": "50000",
            "sz": "0.1",
        }]

        result = await reconciler._reconcile_closed_records()

        assert result["corrected"] == 0
        assert result["failed"] == 1
        storage.update_trade_in_shard.assert_not_called()

