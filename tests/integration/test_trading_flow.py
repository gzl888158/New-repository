"""端到端集成测试：验证完整交易流程

测试信号 → 风控 → 订单队列 → 执行 → 记录的完整链路
"""
import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from execution.order_queue import OrderQueue


class TestSignalToExecutionFlow:
    """信号到执行的完整流程"""

    @pytest.mark.asyncio
    async def test_full_entry_signal_flow(self, basic_config):
        """完整入场信号流程：信号 → 风控 → 队列 → 执行"""
        from services.signal_processor import SignalProcessor

        # 初始化各组件
        global_risk = MagicMock()
        global_risk.can_trade.return_value = True

        strategy_risk = MagicMock()
        strategy_risk.validate_signal.return_value = True

        # 真实订单队列
        order_queue = OrderQueue(basic_config)

        # Mock 执行器：从队列取单并执行
        executed_orders = []

        async def mock_handle_signal(signal):
            order_id = await order_queue.add_order(signal)
            executed_orders.append(order_id)

        order_executor = AsyncMock()
        order_executor.handle_signal = mock_handle_signal

        alert_manager = AsyncMock()
        trade_journal = MagicMock()
        trade_journal._open_positions = {}

        adaptive_controller = MagicMock()
        adaptive_controller.get_allocation.return_value = 0.20
        adaptive_controller.get_position_boost.return_value = 1.0

        profit_optimizer = MagicMock()
        profit_optimizer._compound_factor = 1.0
        profit_optimizer._drawdown_factor = 1.0
        profit_optimizer.get_kelly_fraction.return_value = 0.1

        account_manager = MagicMock()
        account_manager.get_account_info.return_value = {"totalEq": "1000"}

        processor = SignalProcessor(
            basic_config, global_risk, strategy_risk, order_executor,
            alert_manager, trade_journal, adaptive_controller,
            profit_optimizer, account_manager,
        )

        # 发送入场信号
        signal = {
            "symbol": "BTC-USDT-SWAP",
            "strategy_name": "trend",
            "direction": "long",
            "confidence": 0.8,
            "signal_type": "entry",
            "quantity": 0.01,
            "price": 60000,
        }

        await processor._process_signal(signal)

        # 验证信号被处理
        assert len(executed_orders) == 1, "信号未进入执行队列"
        queue_size = await order_queue.get_queue_size()
        assert queue_size == 1, f"队列大小异常: {queue_size}"

        # 验证订单优先级正确（trend = 2）
        order = await order_queue.get_next_order()
        assert order is not None, "未取到订单"
        assert order["symbol"] == "BTC-USDT-SWAP"
        assert order["strategy_name"] == "trend"

    @pytest.mark.asyncio
    async def test_signal_cooldown_blocks_duplicate(self, basic_config):
        """信号冷却应阻止重复信号"""
        from services.signal_processor import SignalProcessor

        global_risk = MagicMock()
        global_risk.can_trade.return_value = True
        strategy_risk = MagicMock()
        strategy_risk.validate_signal.return_value = True

        execute_count = 0

        async def mock_handle(signal):
            nonlocal execute_count
            execute_count += 1

        order_executor = AsyncMock()
        order_executor.handle_signal = mock_handle

        alert_manager = AsyncMock()
        trade_journal = MagicMock()
        trade_journal._open_positions = {}

        adaptive_controller = MagicMock()
        adaptive_controller.get_allocation.return_value = 0.20
        adaptive_controller.get_position_boost.return_value = 1.0

        profit_optimizer = MagicMock()
        profit_optimizer._compound_factor = 1.0
        profit_optimizer._drawdown_factor = 1.0
        profit_optimizer.get_kelly_fraction.return_value = 0.1

        account_manager = MagicMock()
        account_manager.get_account_info.return_value = {"totalEq": "1000"}

        processor = SignalProcessor(
            basic_config, global_risk, strategy_risk, order_executor,
            alert_manager, trade_journal, adaptive_controller,
            profit_optimizer, account_manager,
        )

        signal = {
            "symbol": "ETH-USDT-SWAP",
            "strategy_name": "grid",
            "direction": "long",
            "confidence": 0.7,
            "signal_type": "entry",
            "quantity": 0.1,
            "price": 3000,
        }

        # 连续发送两次相同信号
        await processor._process_signal(signal)
        await processor._process_signal(signal)

        # 第二次应被冷却拦截
        assert execute_count == 1, f"冷却未生效，执行了 {execute_count} 次"

    @pytest.mark.asyncio
    async def test_risk_pause_blocks_all_signals(self, basic_config):
        """全局风控暂停应阻止所有信号"""
        from services.signal_processor import SignalProcessor

        global_risk = MagicMock()
        global_risk.can_trade.return_value = False  # 风控暂停

        strategy_risk = MagicMock()
        order_executor = AsyncMock()
        alert_manager = AsyncMock()
        trade_journal = MagicMock()
        trade_journal._open_positions = {}

        adaptive_controller = MagicMock()
        profit_optimizer = MagicMock()
        profit_optimizer._compound_factor = 1.0
        profit_optimizer._drawdown_factor = 1.0
        profit_optimizer.get_kelly_fraction.return_value = 0.1

        account_manager = MagicMock()
        account_manager.get_account_info.return_value = {"totalEq": "1000"}

        processor = SignalProcessor(
            basic_config, global_risk, strategy_risk, order_executor,
            alert_manager, trade_journal, adaptive_controller,
            profit_optimizer, account_manager,
        )

        signal = {
            "symbol": "BTC-USDT-SWAP",
            "strategy_name": "grid",
            "direction": "long",
            "confidence": 0.9,
            "signal_type": "entry",
            "quantity": 0.01,
            "price": 60000,
        }

        await processor._process_signal(signal)

        order_executor.handle_signal.assert_not_called()


class TestOrderQueueExecutionFlow:
    """订单队列执行流程"""

    @pytest.mark.asyncio
    async def test_priority_ordering(self, basic_config):
        """止损订单应优先于其他订单"""
        queue = OrderQueue(basic_config)

        # 按 grid / trend / scalping / stop_loss 顺序入队
        await queue.add_order({"strategy_name": "grid", "symbol": "A", "signal_type": "entry"})
        await queue.add_order({"strategy_name": "trend", "symbol": "B", "signal_type": "entry"})
        await queue.add_order({"strategy_name": "scalping", "symbol": "C", "signal_type": "entry"})
        await queue.add_order({"strategy_name": "grid", "symbol": "D", "signal_type": "stop_loss"})

        # 取出顺序应为：stop_loss(D) > scalping(C) > trend(B) > grid(A)
        order1 = await queue.get_next_order()
        order2 = await queue.get_next_order()
        order3 = await queue.get_next_order()
        order4 = await queue.get_next_order()

        assert order1["symbol"] == "D", f"止损未优先: {order1['symbol']}"
        assert order2["symbol"] == "C", f"scalping 未第二: {order2['symbol']}"
        assert order3["symbol"] == "B", f"trend 未第三: {order3['symbol']}"
        assert order4["symbol"] == "A", f"grid 未最后: {order4['symbol']}"

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self, basic_config):
        """队列满时应拒绝新订单"""
        config = {"trading": {"max_queue_size": 3}}
        queue = OrderQueue(config)

        for i in range(3):
            await queue.add_order({"strategy_name": "grid", "symbol": f"S{i}", "signal_type": "entry"})

        with pytest.raises(RuntimeError, match="Order queue full"):
            await queue.add_order({"strategy_name": "grid", "symbol": "S4", "signal_type": "entry"})

    @pytest.mark.asyncio
    async def test_cancel_order_flow(self, basic_config):
        """订单取消流程"""
        queue = OrderQueue(basic_config)
        order_id = await queue.add_order({"strategy_name": "grid", "symbol": "X", "signal_type": "entry"})

        cancelled = await queue.cancel_order(order_id)
        assert cancelled is True

        # 取消后队列应为空
        order = await queue.get_next_order()
        assert order is None

    @pytest.mark.asyncio
    async def test_order_status_lifecycle(self, basic_config):
        """订单状态生命周期：queued → executing → filled"""
        queue = OrderQueue(basic_config)
        order_id = await queue.add_order({"strategy_name": "grid", "symbol": "Y", "signal_type": "entry"})

        assert await queue.get_order_status(order_id) == "queued"

        await queue.get_next_order()
        assert await queue.get_order_status(order_id) == "executing"

        await queue.update_order_status(order_id, "filled")
        assert await queue.get_order_status(order_id) == "filled"


class TestDatabaseIntegrationFlow:
    """数据库集成流程"""

    def test_trade_record_crud(self, basic_config):
        """交易记录 CRUD 流程"""
        from data.sqlite_storage import SQLiteStorage, TradeRecord

        storage = SQLiteStorage(basic_config)

        # 创建
        trade = {
            "id": "crud_test_1",
            "symbol": "BTC-USDT-SWAP",
            "strategy_name": "trend",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.01,
            "price": 60000.0,
            "leverage": 5,
            "margin": 120.0,
            "status": "filled",
            "create_time": datetime.now(),
        }
        storage.save_trade_records_batch([trade])

        # 读取
        session = storage._Session()
        record = session.query(TradeRecord).filter_by(id="crud_test_1").first()
        assert record is not None
        assert record.symbol == "BTC-USDT-SWAP"
        assert record.strategy_name == "trend"
        session.close()

    def test_batch_rollback_on_error(self, basic_config):
        """批量插入失败时应回滚（现有 batch 方法吞掉异常但保证回滚）"""
        from data.sqlite_storage import SQLiteStorage, TradeRecord

        storage = SQLiteStorage(basic_config)

        # 第一批正常
        good_trades = [
            {
                "id": f"good_{i}",
                "symbol": "BTC-USDT",
                "strategy_name": "grid",
                "side": "buy",
                "quantity": 0.01,
                "price": 100,
                "leverage": 5,
                "margin": 2,
                "status": "filled",
                "create_time": datetime.now(),
            }
            for i in range(5)
        ]
        storage.save_trade_records_batch(good_trades)

        # 第二批包含重复主键应失败回滚（方法吞掉异常，但事务回滚）
        bad_trades = good_trades + [
            {
                "id": "new_unique",
                "symbol": "ETH-USDT",
                "strategy_name": "grid",
                "side": "buy",
                "quantity": 0.1,
                "price": 100,
                "leverage": 5,
                "margin": 2,
                "status": "filled",
                "create_time": datetime.now(),
            }
        ]

        # 现有 batch 方法吞掉异常，调用方不感知失败，但数据应回滚
        storage.save_trade_records_batch(bad_trades)

        # new_unique 不应存在（回滚）
        session = storage._Session()
        assert session.query(TradeRecord).filter_by(id="new_unique").first() is None
        # good_0 等仍存在（之前已提交）
        assert session.query(TradeRecord).filter_by(id="good_0").first() is not None
        session.close()


class TestRiskControlIntegrationFlow:
    """风控集成流程"""

    def test_circuit_breaker_blocks_trading(self, basic_config):
        """熔断触发后应阻止交易"""
        from risk.global_risk import GlobalRiskControl

        mock_redis = MagicMock()
        mock_okx = MagicMock()
        mock_okx.get_account_info.return_value = {"totalEq": "800"}  # 权益下降

        risk = GlobalRiskControl(basic_config, mock_redis, mock_okx)
        risk._peak_equity = 1000.0
        risk._current_equity = 700.0  # 回撤 30% 超阈值
        risk._is_paused = True
        risk._pause_reason = "最大回撤超限"

        assert risk.can_trade() is False

    def test_risk_limits_integration(self, basic_config):
        """RiskLimits 与风控流程集成"""
        from risk.risk_limits import RiskLimits

        mock_okx = MagicMock()
        mock_okx.get_account_info.return_value = {"totalEq": "10000"}
        limits = RiskLimits(basic_config, mock_okx)

        # 正常信号应通过
        normal = {
            "symbol": "BTC-USDT-SWAP",
            "strategy_name": "grid",
            "leverage": 5,
            "quantity": 0.01,
            "price": 60000,
        }
        assert limits.check_signal(normal) is True

        # 超杠杆信号应被拒
        over_leveraged = dict(normal, leverage=100)
        assert limits.check_signal(over_leveraged) is False
