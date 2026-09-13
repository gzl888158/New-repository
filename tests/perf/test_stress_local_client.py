"""压力测试：用 LocalOKXClient 回放历史高峰流量
=============================================
验证系统在高频信号下的稳定性：
- 不丢单
- 不超限
- 无内存持续上涨
- 关键接口 P99 延迟有基线
"""

import asyncio
import time
import sys
import os
import pytest
import psutil
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.local_client import LocalOKXClient


class LatencyTracker:
    """延迟追踪器"""
    def __init__(self, window_size: int = 1000):
        self._samples: deque = deque(maxlen=window_size)
        self._count = 0
    
    def record(self, latency_ms: float) -> None:
        self._samples.append(latency_ms)
        self._count += 1
    
    @property
    def p50(self) -> float:
        if not self._samples:
            return 0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * 0.5)
        return sorted_samples[idx]
    
    @property
    def p90(self) -> float:
        if not self._samples:
            return 0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * 0.9)
        return sorted_samples[idx]
    
    @property
    def p99(self) -> float:
        if not self._samples:
            return 0
        sorted_samples = sorted(self._samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[idx]
    
    @property
    def avg(self) -> float:
        if not self._samples:
            return 0
        return sum(self._samples) / len(self._samples)
    
    def stats(self) -> dict:
        return {
            "count": self._count,
            "avg_ms": round(self.avg, 2),
            "p50_ms": round(self.p50, 2),
            "p90_ms": round(self.p90, 2),
            "p99_ms": round(self.p99, 2),
        }


class TestLocalClientStress:
    """LocalOKXClient 压力测试"""

    @pytest.fixture
    def client(self):
        config = {
            "symbols": ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"],
            # 余额足够大，避免订单吞吐/生命周期压测因余额不足返回 None
            "initial_balance": 10_000_000.0,
            "tick_interval_ms": 10,
        }
        return LocalOKXClient(config)

    def test_01_ticker_throughput(self, client):
        """测试 ticker 查询吞吐量：基准 5000 次/秒"""
        n = 10000
        symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
        tracker = LatencyTracker()

        start = time.perf_counter()
        for i in range(n):
            sym = symbols[i % 3]
            t0 = time.perf_counter()
            result = client.get_ticker(sym)
            tracker.record((time.perf_counter() - t0) * 1000)
            assert result is not None
        elapsed = time.perf_counter() - start
        throughput = n / elapsed

        stats = tracker.stats()
        print(f"\n[Ticker吞吐量] {n} 次耗时 {elapsed*1000:.0f}ms, 吞吐 {throughput:.0f}/s")
        print(f"  P50={stats['p50_ms']}ms P90={stats['p90_ms']}ms P99={stats['p99_ms']}ms")

        assert throughput >= 5000, f"Ticker吞吐量 {throughput:.0f}/s 低于基准 5000/s"
        assert stats["p99_ms"] < 5, f"P99延迟 {stats['p99_ms']}ms 超过 5ms"

    def test_02_order_throughput(self, client):
        """测试订单吞吐量：基准 1000 单/秒"""
        n = 5000
        tracker = LatencyTracker()

        start = time.perf_counter()
        for i in range(n):
            t0 = time.perf_counter()
            order = client.place_order(
                symbol="BTC-USDT-SWAP",
                side="buy" if i % 2 == 0 else "sell",
                order_type="limit",
                price=60000.0,
                quantity=0.01,
            )
            tracker.record((time.perf_counter() - t0) * 1000)
            assert order is not None
        elapsed = time.perf_counter() - start
        throughput = n / elapsed

        stats = tracker.stats()
        print(f"\n[订单吞吐量] {n} 单耗时 {elapsed*1000:.0f}ms, 吞吐 {throughput:.0f}/s")
        print(f"  P50={stats['p50_ms']}ms P90={stats['p90_ms']}ms P99={stats['p99_ms']}ms")

        assert throughput >= 1000, f"订单吞吐量 {throughput:.0f}/s 低于基准 1000/s"
        assert stats["p99_ms"] < 10, f"P99延迟 {stats['p99_ms']}ms 超过 10ms"

    def test_03_concurrent_ticker_queries(self, client):
        """并发 ticker 查询：100 并发不丢数据"""
        n_queries = 5000
        symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
        results = []

        async def _run():
            async def query(sym):
                return client.get_ticker(sym)

            tasks = []
            for i in range(n_queries):
                sym = symbols[i % 3]
                tasks.append(asyncio.create_task(query(sym)))
            
            return await asyncio.gather(*tasks)

        start = time.perf_counter()
        results = asyncio.run(_run())
        elapsed = time.perf_counter() - start

        success_count = sum(1 for r in results if r is not None)
        throughput = n_queries / elapsed

        print(f"\n[并发Ticker] {n_queries} 查询耗时 {elapsed*1000:.0f}ms, "
              f"成功 {success_count}/{n_queries}, 吞吐 {throughput:.0f}/s")

        assert success_count == n_queries, f"丢数据: {n_queries - success_count} 个查询失败"
        assert throughput >= 5000, f"并发吞吐量 {throughput:.0f}/s 低于基准 5000/s"

    def test_04_position_lifecycle_stress(self, client):
        """持仓生命周期压力测试：开仓→查询→平仓 循环"""
        n_cycles = 500
        tracker = LatencyTracker()

        for i in range(n_cycles):
            t0 = time.perf_counter()
            # 开仓
            client.place_order(
                symbol="BTC-USDT-SWAP",
                side="buy",
                order_type="market",
                quantity=0.01,
            )
            # 查询持仓
            pos = client.get_positions_dict().get("BTC-USDT-SWAP")
            # 平仓
            if pos:
                client.reduce_position("BTC-USDT-SWAP", pos.quantity)
            tracker.record((time.perf_counter() - t0) * 1000)

        stats = tracker.stats()
        print(f"\n[持仓生命周期] {n_cycles} 循环")
        print(f"  P50={stats['p50_ms']}ms P90={stats['p90_ms']}ms P99={stats['p99_ms']}ms")

        assert stats["p99_ms"] < 20, f"P99延迟 {stats['p99_ms']}ms 超过 20ms"

    def test_05_account_query_stress(self, client):
        """账户查询压力测试：连续 10000 次查询"""
        n = 10000
        tracker = LatencyTracker()

        start = time.perf_counter()
        for _ in range(n):
            t0 = time.perf_counter()
            account = client.get_account_info()
            tracker.record((time.perf_counter() - t0) * 1000)
            assert account is not None
        elapsed = time.perf_counter() - start

        stats = tracker.stats()
        throughput = n / elapsed
        print(f"\n[账户查询] {n} 次耗时 {elapsed*1000:.0f}ms, 吞吐 {throughput:.0f}/s")
        print(f"  P50={stats['p50_ms']}ms P99={stats['p99_ms']}ms")

        assert stats["p99_ms"] < 2, f"P99延迟 {stats['p99_ms']}ms 超过 2ms"

    def test_06_memory_no_leak(self, client):
        """内存泄漏检测：10 万次操作后内存增长 < 50MB"""
        process = psutil.Process(os.getpid())
        initial_mem = process.memory_info().rss / 1024 / 1024  # MB

        symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
        for i in range(100000):
            sym = symbols[i % 3]
            client.get_ticker(sym)
            if i % 100 == 0:
                client.place_order(
                    symbol=sym, side="buy", order_type="limit",
                    price=50000.0, quantity=0.01,
                )
            if i % 500 == 0:
                client.get_account_info()
                client.get_positions_dict()

        final_mem = process.memory_info().rss / 1024 / 1024
        growth = final_mem - initial_mem

        print(f"\n[内存泄漏] 10万次操作: {initial_mem:.1f}MB -> {final_mem:.1f}MB (增长 {growth:.1f}MB)")

        assert growth < 50, f"内存增长 {growth:.1f}MB 超过 50MB 阈值"

    def test_07_high_frequency_tick_replay(self, client):
        """高频 tick 回放：模拟 1000 tick/s 持续 10 秒，验证不丢 tick"""
        duration_sec = 10
        tick_rate = 1000  # tick/s
        total_ticks = duration_sec * tick_rate
        processed = 0

        start = time.perf_counter()
        deadline = start + duration_sec

        while time.perf_counter() < deadline:
            batch_start = time.perf_counter()
            batch_size = 0
            while time.perf_counter() - batch_start < 0.1:  # 100ms batch
                client._update_prices()
                ticker = client.get_ticker("BTC-USDT-SWAP")
                if ticker:
                    processed += 1
                    batch_size += 1
                if processed >= total_ticks:
                    break

        elapsed = time.perf_counter() - start
        actual_rate = processed / elapsed

        print(f"\n[高频Tick回放] 目标 {tick_rate}/s x {duration_sec}s, "
              f"实际处理 {processed} tick, 速率 {actual_rate:.0f}/s")

        assert processed >= total_ticks * 0.95, f"丢 tick: {total_ticks - processed} 个未处理"
        assert actual_rate >= tick_rate * 0.9, f"处理速率 {actual_rate:.0f}/s 低于目标 90%"

    def test_08_latency_baseline_report(self):
        """P99 延迟基线报告：汇总所有关键接口"""
        config = {
            "symbols": ["BTC-USDT-SWAP"],
            "initial_balance": 1000.0,
            "tick_interval_ms": 10,
        }
        client = LocalOKXClient(config)

        baselines = {}

        # Ticker
        tracker = LatencyTracker()
        for _ in range(1000):
            t0 = time.perf_counter()
            client.get_ticker("BTC-USDT-SWAP")
            tracker.record((time.perf_counter() - t0) * 1000)
        baselines["ticker"] = tracker.stats()

        # Place order
        tracker = LatencyTracker()
        for i in range(500):
            t0 = time.perf_counter()
            client.place_order(
                symbol="BTC-USDT-SWAP", side="buy" if i % 2 == 0 else "sell",
                order_type="limit", price=60000.0, quantity=0.01,
            )
            tracker.record((time.perf_counter() - t0) * 1000)
        baselines["place_order"] = tracker.stats()

        # Account
        tracker = LatencyTracker()
        for _ in range(1000):
            t0 = time.perf_counter()
            client.get_account_info()
            tracker.record((time.perf_counter() - t0) * 1000)
        baselines["account"] = tracker.stats()

        print("\n[P99 延迟基线]")
        for name, stats in baselines.items():
            print(f"  {name:15s}: P50={stats['p50_ms']:6.2f}ms  P90={stats['p90_ms']:6.2f}ms  "
                  f"P99={stats['p99_ms']:6.2f}ms  avg={stats['avg_ms']:6.2f}ms")

        # 关键接口 P99 必须 < 10ms
        for name in ["ticker", "account"]:
            assert baselines[name]["p99_ms"] < 10, \
                f"{name} P99={baselines[name]['p99_ms']}ms 超过 10ms"
        assert baselines["place_order"]["p99_ms"] < 20, \
            f"place_order P99={baselines['place_order']['p99_ms']}ms 超过 20ms"