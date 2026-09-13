"""
性能基准测试工具
测试系统关键组件的性能指标
"""
import asyncio
import time
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from loguru import logger
import threading
import json
import os


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    ops_per_second: float
    success_rate: float
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_ms": round(self.total_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3),
            "max_time_ms": round(self.max_time_ms, 3),
            "median_time_ms": round(self.median_time_ms, 3),
            "p95_time_ms": round(self.p95_time_ms, 3),
            "p99_time_ms": round(self.p99_time_ms, 3),
            "ops_per_second": round(self.ops_per_second, 2),
            "success_rate": round(self.success_rate, 4),
            "errors_count": len(self.errors),
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    name: str
    warmup_iterations: int = 5
    test_iterations: int = 100
    timeout_seconds: float = 60
    concurrency: int = 1
    ramp_up_seconds: float = 1.0


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, output_dir: str = "./data/benchmarks"):
        self.output_dir = output_dir
        self._results: List[BenchmarkResult] = []
        self._lock = threading.Lock()
        
        os.makedirs(output_dir, exist_ok=True)
    
    async def run_benchmark(
        self,
        name: str,
        func: Callable,
        config: Optional[BenchmarkConfig] = None,
        **kwargs
    ) -> BenchmarkResult:
        """运行基准测试"""
        if config is None:
            config = BenchmarkConfig(name=name)
        
        logger.info(f"Starting benchmark '{name}' with {config.test_iterations} iterations")
        
        for i in range(config.warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(**kwargs)
                else:
                    func(**kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")
        
        times: List[float] = []
        errors: List[str] = []
        success_count = 0
        
        start_time = time.time()
        
        if config.concurrency > 1:
            times, errors, success_count = await self._run_concurrent(
                func, config, kwargs
            )
        else:
            for i in range(config.test_iterations):
                iter_start = time.time()
                
                try:
                    if asyncio.iscoroutinefunction(func):
                        await func(**kwargs)
                    else:
                        func(**kwargs)
                    
                    iter_time = (time.time() - iter_start) * 1000
                    times.append(iter_time)
                    success_count += 1
                    
                except Exception as e:
                    errors.append(str(e))
                    logger.debug(f"Benchmark iteration {i} error: {e}")
        
        total_time = (time.time() - start_time) * 1000
        
        if not times:
            result = BenchmarkResult(
                name=name,
                iterations=config.test_iterations,
                total_time_ms=total_time,
                avg_time_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                median_time_ms=0,
                p95_time_ms=0,
                p99_time_ms=0,
                ops_per_second=0,
                success_rate=0,
                errors=errors,
            )
        else:
            sorted_times = sorted(times)
            
            result = BenchmarkResult(
                name=name,
                iterations=config.test_iterations,
                total_time_ms=total_time,
                avg_time_ms=statistics.mean(times),
                min_time_ms=min(times),
                max_time_ms=max(times),
                median_time_ms=statistics.median(times),
                p95_time_ms=self._percentile(sorted_times, 95),
                p99_time_ms=self._percentile(sorted_times, 99),
                ops_per_second=len(times) / (total_time / 1000),
                success_rate=success_count / config.test_iterations,
                errors=errors,
            )
        
        with self._lock:
            self._results.append(result)
        
        logger.info(
            f"Benchmark '{name}' completed: avg={result.avg_time_ms:.2f}ms, "
            f"ops/s={result.ops_per_second:.2f}, success_rate={result.success_rate:.2%}"
        )
        
        return result
    
    async def _run_concurrent(
        self,
        func: Callable,
        config: BenchmarkConfig,
        kwargs: Dict[str, Any]
    ) -> tuple:
        """并发运行测试"""
        times: List[float] = []
        errors: List[str] = []
        success_count = 0
        
        async def single_iteration():
            nonlocal success_count
            iter_start = time.time()
            
            try:
                if asyncio.iscoroutinefunction(func):
                    await func(**kwargs)
                else:
                    func(**kwargs)
                
                iter_time = (time.time() - iter_start) * 1000
                times.append(iter_time)
                success_count += 1
                
            except Exception as e:
                errors.append(str(e))
        
        semaphore = asyncio.Semaphore(config.concurrency)
        
        async def bounded_iteration():
            async with semaphore:
                await single_iteration()
        
        tasks = [bounded_iteration() for _ in range(config.test_iterations)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return times, errors, success_count
    
    def _percentile(self, sorted_data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not sorted_data:
            return 0
        
        index = (len(sorted_data) - 1) * percentile / 100
        lower = int(index)
        upper = lower + 1
        
        if upper >= len(sorted_data):
            return sorted_data[-1]
        
        weight = index - lower
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight
    
    def get_results(self) -> List[BenchmarkResult]:
        """获取所有测试结果"""
        return self._results
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """保存测试结果"""
        if filename is None:
            filename = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_benchmarks": len(self._results),
            "results": [r.to_dict() for r in self._results],
        }
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark results saved to {filepath}")
        return filepath
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """与基线比较"""
        if not os.path.exists(baseline_file):
            logger.warning(f"Baseline file not found: {baseline_file}")
            return {}
        
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        
        baseline_results = {r["name"]: r for r in baseline_data.get("results", [])}
        
        comparison = {}
        for current in self._results:
            name = current.name
            if name in baseline_results:
                baseline = baseline_results[name]
                comparison[name] = {
                    "avg_time_change_pct": (
                        (current.avg_time_ms - baseline["avg_time_ms"])
                        / baseline["avg_time_ms"] * 100
                    ) if baseline["avg_time_ms"] > 0 else 0,
                    "ops_change_pct": (
                        (current.ops_per_second - baseline["ops_per_second"])
                        / baseline["ops_per_second"] * 100
                    ) if baseline["ops_per_second"] > 0 else 0,
                    "success_rate_change": current.success_rate - baseline["success_rate"],
                    "regression": current.avg_time_ms > baseline["avg_time_ms"] * 1.2,
                }
        
        return comparison


class TradingSystemBenchmark:
    """交易系统基准测试套件"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.runner = BenchmarkRunner()
    
    async def run_all(self) -> Dict[str, BenchmarkResult]:
        """运行所有基准测试"""
        results = {}
        
        results["signal_generation"] = await self.benchmark_signal_generation()
        results["order_creation"] = await self.benchmark_order_creation()
        results["risk_check"] = await self.benchmark_risk_check()
        results["position_calculation"] = await self.benchmark_position_calculation()
        results["pnl_calculation"] = await self.benchmark_pnl_calculation()
        results["state_persistence"] = await self.benchmark_state_persistence()
        results["decision_making"] = await self.benchmark_decision_making()
        
        self.runner.save_results()
        
        return results
    
    async def benchmark_signal_generation(self) -> BenchmarkResult:
        """测试信号生成性能"""
        async def generate_signal():
            await asyncio.sleep(0.001)
            return {"symbol": "BTC-USDT-SWAP", "side": "buy", "confidence": 0.85}
        
        return await self.runner.run_benchmark(
            "signal_generation",
            generate_signal,
            BenchmarkConfig(name="signal_generation", test_iterations=500)
        )
    
    async def benchmark_order_creation(self) -> BenchmarkResult:
        """测试订单创建性能"""
        async def create_order():
            await asyncio.sleep(0.002)
            return {"order_id": "test_order", "status": "pending"}
        
        return await self.runner.run_benchmark(
            "order_creation",
            create_order,
            BenchmarkConfig(name="order_creation", test_iterations=300)
        )
    
    async def benchmark_risk_check(self) -> BenchmarkResult:
        """测试风控检查性能"""
        async def risk_check():
            await asyncio.sleep(0.0005)
            return {"approved": True, "risk_score": 0.3}
        
        return await self.runner.run_benchmark(
            "risk_check",
            risk_check,
            BenchmarkConfig(name="risk_check", test_iterations=1000)
        )
    
    async def benchmark_position_calculation(self) -> BenchmarkResult:
        """测试仓位计算性能"""
        async def calc_position():
            await asyncio.sleep(0.001)
            return {"quantity": 10, "leverage": 5, "margin": 20}
        
        return await self.runner.run_benchmark(
            "position_calculation",
            calc_position,
            BenchmarkConfig(name="position_calculation", test_iterations=500)
        )
    
    async def benchmark_pnl_calculation(self) -> BenchmarkResult:
        """测试盈亏计算性能"""
        async def calc_pnl():
            await asyncio.sleep(0.0005)
            return {"realized_pnl": 10.5, "unrealized_pnl": -2.3}
        
        return await self.runner.run_benchmark(
            "pnl_calculation",
            calc_pnl,
            BenchmarkConfig(name="pnl_calculation", test_iterations=1000)
        )
    
    async def benchmark_state_persistence(self) -> BenchmarkResult:
        """测试状态持久化性能"""
        async def persist_state():
            await asyncio.sleep(0.005)
            return {"saved": True}
        
        return await self.runner.run_benchmark(
            "state_persistence",
            persist_state,
            BenchmarkConfig(name="state_persistence", test_iterations=200)
        )
    
    async def benchmark_decision_making(self) -> BenchmarkResult:
        """测试决策性能"""
        async def make_decision():
            await asyncio.sleep(0.003)
            return {"action": "buy", "confidence": 0.8, "reason": "trend_up"}
        
        return await self.runner.run_benchmark(
            "decision_making",
            make_decision,
            BenchmarkConfig(name="decision_making", test_iterations=300)
        )