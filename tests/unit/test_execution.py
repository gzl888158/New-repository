import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from execution.order_queue import TokenBucket, OrderQueue


class TestTokenBucket:
    @pytest.mark.asyncio
    async def test_acquire_success(self):
        bucket = TokenBucket(rate=10, capacity=5)
        await bucket.acquire()

    @pytest.mark.asyncio
    async def test_acquire_multiple(self):
        bucket = TokenBucket(rate=100, capacity=10)
        for _ in range(10):
            await bucket.acquire()


class TestOrderQueue:
    @pytest.mark.asyncio
    async def test_add_order(self):
        config = {"trading": {"max_queue_size": 100}}
        queue = OrderQueue(config)
        order_id = await queue.add_order({"strategy_name": "grid", "symbol": "BTC-USDT-SWAP"})
        assert order_id is not None
        assert len(order_id) > 0

    @pytest.mark.asyncio
    async def test_add_order_full(self):
        config = {"trading": {"max_queue_size": 1000}}
        queue = OrderQueue(config)
        queue._max_queue_size = 1
        await queue.add_order({"strategy_name": "grid", "symbol": "BTC-USDT-SWAP"})
        with pytest.raises(RuntimeError):
            await queue.add_order({"strategy_name": "grid", "symbol": "ETH-USDT-SWAP"})

    @pytest.mark.asyncio
    async def test_get_next_order(self):
        config = {"trading": {"max_queue_size": 100}}
        queue = OrderQueue(config)
        await queue.add_order({"strategy_name": "grid", "symbol": "BTC-USDT-SWAP"})
        await queue.add_order({"strategy_name": "scalping", "symbol": "ETH-USDT-SWAP"})
        order = await queue.get_next_order()
        assert order is not None
        assert "order_id" in order

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        config = {"trading": {"max_queue_size": 100}}
        queue = OrderQueue(config)
        order_id = await queue.add_order({"strategy_name": "grid", "symbol": "BTC-USDT-SWAP"})
        result = await queue.cancel_order(order_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_queue(self):
        config = {"trading": {"max_queue_size": 100}}
        queue = OrderQueue(config)
        await queue.add_order({"strategy_name": "grid", "symbol": "BTC-USDT-SWAP"})
        await queue.add_order({"strategy_name": "grid", "symbol": "ETH-USDT-SWAP"})
        await queue.clear_queue()
        size = await queue.get_queue_size()
        assert size == 0