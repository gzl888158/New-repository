"""性能压测：验证高频交易场景下的系统吞吐量

测试核心热路径：
- 订单队列并发读写
- 信号处理吞吐量
- 数据库批量写入
- 风控检查延迟
"""
import asyncio
import time
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from execution.order_queue import OrderQueue, TokenBucket


class TestOrderQueuePerformance:
    """订单队列性能测试"""

    @pytest.fixture
    def order_queue(self):
        config = {"trading": {"max_queue_size": 10000}}
        return OrderQueue(config)

    @pytest.mark.asyncio
    async def test_add_order_throughput(self, order_queue):
        """测试订单入队吞吐量：基准 1000 单/秒"""
        n = 2000
        start = time.perf_counter()
        for i in range(n):
            await order_queue.add_order({
                "strategy_name": "grid",
                "symbol": "BTC-USDT-SWAP",
                "signal_type": "entry",
                "quantity": 0.01,
                "price": 60000,
            })
        elapsed = time.perf_counter() - start
        throughput = n / elapsed
        assert throughput >= 1000, f"入队吞吐量 {throughput:.0f}/s 低于基准 1000/s"
        print(f"\n[入队吞吐量] {n} 单耗时 {elapsed*1000:.1f}ms, 吞吐 {throughput:.0f}/s")

    @pytest.mark.asyncio
    async def test_get_next_order_throughput(self, order_queue):
        """测试订单出队吞吐量：基准 1000 单/秒"""
        n = 2000
        for i in range(n):
            await order_queue.add_order({
                "strategy_name": "scalping",
                "symbol": "ETH-USDT-SWAP",
                "quantity": 0.1,
                "price": 3000,
            })

        start = time.perf_counter()
        for _ in range(n):
            await order_queue.get_next_order()
        elapsed = time.perf_counter() - start
        throughput = n / elapsed
        assert throughput >= 1000, f"出队吞吐量 {throughput:.0f}/s 低于基准 1000/s"
        print(f"\n[出队吞吐量] {n} 单耗时 {elapsed*1000:.1f}ms, 吞吐 {throughput:.0f}/s")

    @pytest.mark.asyncio
    async def test_concurrent_add_get(self, order_queue):
        """并发读写测试：10 生产者 + 10 消费者"""
        n_per_producer = 200
        producers = 10
        consumers = 10
        total = n_per_producer * producers

        async def produce(pid):
            for i in range(n_per_producer):
                await order_queue.add_order({
                    "strategy_name": "trend",
                    "symbol": f"SYM{pid}-USDT",
                    "quantity": 0.01,
                    "price": 100,
                })

        async def consume():
            got = 0
            while got < n_per_producer:
                order = await order_queue.get_next_order()
                if order:
                    got += 1
                else:
                    await asyncio.sleep(0.001)

        start = time.perf_counter()
        await asyncio.gather(
            *[produce(i) for i in range(producers)],
            *[consume() for _ in range(consumers)],
        )
        elapsed = time.perf_counter() - start
        throughput = total / elapsed
        assert throughput >= 500, f"并发吞吐量 {throughput:.0f}/s 过低"
        print(f"\n[并发吞吐量] {total} 单 {producers}P+{consumers}C 耗时 {elapsed*1000:.1f}ms, 吞吐 {throughput:.0f}/s")


class TestTokenBucketPerformance:
    """令牌桶限流性能测试"""

    @pytest.mark.asyncio
    async def test_token_bucket_high_rate(self):
        """高速限流场景：1000 请求/秒速率下应能快速处理"""
        bucket = TokenBucket(rate=1000, capacity=1000)
        n = 500
        start = time.perf_counter()
        for _ in range(n):
            await bucket.acquire()
        elapsed = time.perf_counter() - start
        # 容量内的请求应立即完成
        assert elapsed < 1.0, f"令牌桶处理 {n} 请求耗时 {elapsed*1000:.1f}ms 过长"
        print(f"\n[令牌桶] {n} 请求耗时 {elapsed*1000:.1f}ms")


class TestSignalProcessorPerformance:
    """信号处理性能测试"""

    @pytest.mark.asyncio
    async def test_signal_throughput(self, basic_config):
        """信号处理吞吐量测试：1000 信号应在 2 秒内处理完成"""
        from services.signal_processor import SignalProcessor

        # Mock 依赖
        global_risk = MagicMock()
        global_risk.can_trade.return_value = True

        strategy_risk = MagicMock()
        strategy_risk.validate_signal.return_value = True

        order_executor = AsyncMock()
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
        profit_optimizer.update_equity = MagicMock()

        account_manager = MagicMock()
        account_manager.get_account_info.return_value = {"totalEq": "1000"}

        processor = SignalProcessor(
            basic_config, global_risk, strategy_risk, order_executor,
            alert_manager, trade_journal, adaptive_controller,
            profit_optimizer, account_manager,
        )

        n = 1000
        signals = [
            {
                "symbol": f"SYM{i % 10}-USDT-SWAP",
                "strategy_name": "grid",
                "direction": "long",
                "confidence": 0.7,
                "signal_type": "entry",
                "quantity": 0.01,
                "price": 100,
            }
            for i in range(n)
        ]

        start = time.perf_counter()
        for signal in signals:
            await processor._process_signal(signal)
        elapsed = time.perf_counter() - start
        throughput = n / elapsed

        assert elapsed < 2.0, f"处理 {n} 信号耗时 {elapsed:.2f}s 过长"
        print(f"\n[信号处理] {n} 信号耗时 {elapsed*1000:.1f}ms, 吞吐 {throughput:.0f}/s")


class TestDatabaseBatchPerformance:
    """数据库批量写入性能测试"""

    def test_batch_insert_throughput(self, basic_config):
        """批量插入吞吐量：1000 条记录应在 1 秒内完成"""
        from data.sqlite_storage import SQLiteStorage, TradeRecord

        storage = SQLiteStorage(basic_config)

        n = 1000
        trades = [
            {
                "id": f"test_perf_{i}",
                "symbol": f"SYM{i % 20}-USDT",
                "strategy_name": "grid",
                "side": "buy",
                "order_type": "limit",
                "quantity": 0.01,
                "price": 100.0 + i * 0.01,
                "leverage": 5,
                "margin": 2.0,
                "status": "filled",
                "create_time": datetime.now(),
            }
            for i in range(n)
        ]

        start = time.perf_counter()
        storage.save_trade_records_batch(trades)
        elapsed = time.perf_counter() - start
        throughput = n / elapsed

        assert elapsed < 1.0, f"批量插入 {n} 条耗时 {elapsed*1000:.1f}ms 过长"
        print(f"\n[批量插入] {n} 条耗时 {elapsed*1000:.1f}ms, 吞吐 {throughput:.0f}/s")

        # 验证写入正确
        session = storage._Session()
        count = session.query(TradeRecord).count()
        session.close()
        assert count == n, f"实际写入 {count} != 预期 {n}"


class TestRiskCheckPerformance:
    """风控检查延迟测试"""

    def test_risk_limits_check_latency(self, basic_config):
        """RiskLimits 单次检查应在 1ms 内完成"""
        from risk.risk_limits import RiskLimits

        mock_okx = MagicMock()
        mock_okx.get_account_info.return_value = {"totalEq": "10000"}
        limits = RiskLimits(basic_config, mock_okx)

        signal = {
            "symbol": "BTC-USDT-SWAP",
            "strategy_name": "grid",
            "leverage": 5,
            "quantity": 0.01,
            "price": 60000,
        }

        # 预热
        for _ in range(100):
            limits.check_signal(signal)

        n = 1000
        start = time.perf_counter()
        for _ in range(n):
            limits.check_signal(signal)
        elapsed = time.perf_counter() - start
        avg_latency_us = (elapsed / n) * 1_000_000

        assert avg_latency_us < 1000, f"平均延迟 {avg_latency_us:.1f}μs 超过 1000μs"
        print(f"\n[风控检查] {n} 次平均延迟 {avg_latency_us:.1f}μs")

    def test_strategy_risk_validate_latency(self, basic_config):
        """策略风控验证延迟应在 500μs 内"""
        from risk.strategy_risk import StrategyRiskControl

        risk = StrategyRiskControl(basic_config)
        signal = {
            "symbol": "BTC-USDT-SWAP",
            "strategy_name": "grid",
            "direction": "long",
            "leverage": 5,
            "quantity": 0.01,
            "price": 60000,
            "confidence": 0.7,
        }

        # 预热
        for _ in range(100):
            risk.validate_signal(signal)

        n = 1000
        start = time.perf_counter()
        for _ in range(n):
            risk.validate_signal(signal)
        elapsed = time.perf_counter() - start
        avg_latency_us = (elapsed / n) * 1_000_000

        assert avg_latency_us < 500, f"策略风控平均延迟 {avg_latency_us:.1f}μs 超过 500μs"
        print(f"\n[策略风控] {n} 次平均延迟 {avg_latency_us:.1f}μs")
