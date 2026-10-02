"""
执行监控器 - 实时追踪订单执行的延迟、吞吐量、SLA合规性

核心能力:
  - 多维度延迟追踪 (端到端、API往返、交易所确认)
  - 实时吞吐量统计 (订单/秒、成交量/秒)
  - SLA合规监控 (P50/P95/P99延迟阈值)
  - 自适应告警阈值 (基于滚动窗口统计)
  - 健康评分 (综合延迟、成功率、吞吐)
"""
import asyncio
import time
import statistics
from collections import deque
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


class LatencyType(Enum):
    """延迟类型"""
    E2E = "e2e"                    # 端到端: 信号生成 -> 订单确认
    API_ROUNDTRIP = "api_rtt"      # API往返: 请求发送 -> 响应接收
    EXCHANGE_ACK = "exchange_ack"  # 交易所确认: 下单 -> 成交
    VALIDATION = "validation"      # 验证耗时
    QUEUE_WAIT = "queue_wait"      # 队列等待


class SLACompliance(Enum):
    """SLA合规状态"""
    EXCELLENT = "excellent"    # P95 < 阈值
    GOOD = "good"              # P95 接近阈值
    DEGRADED = "degraded"      # P95 超过阈值
    CRITICAL = "critical"      # P99 超过阈值


class ExecutionEvent:
    """执行事件"""
    def __init__(self, event_type: str, order_id: str = "", symbol: str = "",
                 strategy: str = "", latency_ms: float = 0.0, metadata: Dict = None,
                 trace_id: str = ""):
        self.event_type = event_type
        self.order_id = order_id
        self.symbol = symbol
        self.strategy = strategy
        self.latency_ms = latency_ms
        self.timestamp = datetime.now()
        self.metadata = metadata or {}
        self.trace_id = trace_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp.isoformat(),
            "trace_id": self.trace_id,
            "metadata": self.metadata,
        }


class SLAWindow:
    """SLA滑动窗口"""
    def __init__(self, window_seconds: int = 300):
        self._window_seconds = window_seconds
        self._latencies: deque = deque()
        self._timestamps: deque = deque()

    def add(self, latency_ms: float):
        now = time.time()
        self._latencies.append(latency_ms)
        self._timestamps.append(now)
        self._prune(now)

    def _prune(self, now: float):
        cutoff = now - self._window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()
            self._latencies.popleft()

    def get_stats(self) -> Dict[str, Any]:
        if not self._latencies:
            return {"count": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0, "min": 0}
        sorted_lat = sorted(self._latencies)
        n = len(sorted_lat)
        return {
            "count": n,
            "avg": round(sum(sorted_lat) / n, 2),
            "p50": round(sorted_lat[n // 2], 2),
            "p95": round(sorted_lat[int(n * 0.95)], 2),
            "p99": round(sorted_lat[int(n * 0.99)], 2),
            "max": round(max(sorted_lat), 2),
            "min": round(min(sorted_lat), 2),
        }

    @property
    def count(self) -> int:
        return len(self._latencies)


class ExecutionMonitor:
    """执行监控器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        mon_cfg = self._config.get("execution_monitor", {})

        # ── 延迟窗口 ──
        self._latency_windows: Dict[str, SLAWindow] = {
            "1m": SLAWindow(60),
            "5m": SLAWindow(300),
            "15m": SLAWindow(900),
            "1h": SLAWindow(3600),
        }

        # ── 按延迟类型分窗口 ──
        self._type_windows: Dict[str, SLAWindow] = {}

        # ── 吞吐量 ──
        self._order_timestamps: deque = deque(maxlen=10000)
        self._volume_timestamps: deque = deque(maxlen=10000)
        self._throughput_window = mon_cfg.get("throughput_window_seconds", 60)

        # ── SLA阈值 ──
        self._sla_thresholds = mon_cfg.get("sla_thresholds", {
            "e2e_p95_ms": 500,
            "e2e_p99_ms": 1000,
            "api_rtt_p95_ms": 200,
            "api_rtt_p99_ms": 500,
        })

        # ── 健康评分 ──
        self._health_score = 100.0
        self._health_history: deque = deque(maxlen=100)

        # ── 事件历史 ──
        self._events: deque = deque(maxlen=mon_cfg.get("max_events", 1000))
        self._alert_thresholds = mon_cfg.get("alert_thresholds", {
            "latency_spike_factor": 2.0,
            "success_rate_min": 0.90,
            "throughput_drop_factor": 0.3,
        })

        # ── 统计 ──
        self._total_orders = 0
        self._success_orders = 0
        self._failed_orders = 0

        # ── 自适应阈值 ──
        self._adaptive_baseline: Dict[str, float] = {}
        self._adaptive_initialized = False

        self._lock = asyncio.Lock()
        self._running = False

        logger.info("ExecutionMonitor initialized: windows=1m/5m/15m/1h")

    async def start(self):
        self._running = True
        logger.info("ExecutionMonitor started")

    async def stop(self):
        self._running = False
        logger.info("ExecutionMonitor stopped")

    # ===================== 事件记录 =====================

    async def record_latency(self, latency_type: LatencyType, latency_ms: float,
                              order_id: str = "", symbol: str = "", strategy: str = "",
                              trace_id: str = ""):
        """记录延迟事件"""
        async with self._lock:
            event = ExecutionEvent(
                event_type=f"latency_{latency_type.value}",
                order_id=order_id, symbol=symbol, strategy=strategy,
                latency_ms=latency_ms,
                metadata={"latency_type": latency_type.value},
                trace_id=trace_id,
            )
            self._events.append(event)

            # 更新类型窗口
            if latency_type.value not in self._type_windows:
                self._type_windows[latency_type.value] = SLAWindow(300)
            self._type_windows[latency_type.value].add(latency_ms)

            # 更新全局窗口
            for window in self._latency_windows.values():
                window.add(latency_ms)

            # 更新自适应基线
            self._update_adaptive_baseline(latency_type, latency_ms)

            # 健康检查
            await self._check_latency_health(latency_type, latency_ms)

    async def record_order_event(self, event_type: str, order_id: str = "",
                                  symbol: str = "", strategy: str = "",
                                  quantity: float = 0, price: float = 0,
                                  status: str = "", metadata: Dict = None,
                                  trace_id: str = ""):
        """记录订单事件"""
        async with self._lock:
            event = ExecutionEvent(
                event_type=event_type, order_id=order_id,
                symbol=symbol, strategy=strategy,
                metadata=metadata or {},
                trace_id=trace_id,
            )
            event.metadata["quantity"] = quantity
            event.metadata["price"] = price
            event.metadata["status"] = status
            self._events.append(event)

            # 吞吐量
            now = time.time()
            self._order_timestamps.append(now)
            if quantity > 0:
                self._volume_timestamps.append((now, quantity * (price or 1)))

            # 成功/失败计数
            self._total_orders += 1
            if status == "success":
                self._success_orders += 1
            elif status in ("failed", "error"):
                self._failed_orders += 1

    # ===================== 自适应阈值 =====================

    def _update_adaptive_baseline(self, lat_type: LatencyType, latency_ms: float):
        """更新自适应基线"""
        key = lat_type.value
        if key not in self._type_windows:
            return

        window = self._type_windows[key]
        if window.count >= 50:
            stats = window.get_stats()
            if key not in self._adaptive_baseline:
                self._adaptive_baseline[key] = stats["p95"]
            else:
                # EMA更新基线
                alpha = 0.01
                self._adaptive_baseline[key] = (
                    alpha * stats["p95"] + (1 - alpha) * self._adaptive_baseline[key]
                )
            self._adaptive_initialized = True

    async def _check_latency_health(self, lat_type: LatencyType, latency_ms: float):
        """检查延迟健康"""
        if not self._adaptive_initialized:
            return

        key = lat_type.value
        baseline = self._adaptive_baseline.get(key, 0)
        spike_factor = self._alert_thresholds["latency_spike_factor"]

        if baseline > 0 and latency_ms > baseline * spike_factor:
            # 扣分
            penalty = min(30, (latency_ms / max(baseline, 1) - 1) * 10)
            self._health_score = max(0, self._health_score - penalty)
            self._health_history.append({"score": self._health_score, "reason": f"{key}_spike"})
        else:
            # 缓慢恢复
            self._health_score = min(100, self._health_score + 0.1)
            self._health_history.append({"score": self._health_score, "reason": "normal"})

    # ===================== 查询接口 =====================

    def get_latency_stats(self, window: str = "5m") -> Dict[str, Any]:
        """获取延迟统计"""
        w = self._latency_windows.get(window)
        if not w:
            return {}
        result = w.get_stats()

        # 按类型
        by_type = {}
        for t_name, t_window in self._type_windows.items():
            by_type[t_name] = t_window.get_stats()

        result["by_type"] = by_type
        result["adaptive_baseline"] = self._adaptive_baseline
        return result

    def get_throughput_stats(self) -> Dict[str, Any]:
        """获取吞吐量统计"""
        now = time.time()
        cutoff = now - self._throughput_window
        recent_orders = sum(1 for ts in self._order_timestamps if ts >= cutoff)
        recent_vol = sum(vol for ts, vol in self._volume_timestamps if ts >= cutoff)
        elapsed = min(self._throughput_window, now - (self._order_timestamps[0] if self._order_timestamps else now))

        return {
            "orders_per_second": round(recent_orders / max(elapsed, 1), 2),
            "volume_per_second": round(recent_vol / max(elapsed, 1), 2),
            "total_orders": self._total_orders,
            "success_rate": self._success_orders / max(self._total_orders, 1),
        }

    def get_sla_status(self) -> Dict[str, Any]:
        """获取SLA合规状态"""
        e2e_stats = self._type_windows.get("e2e", SLAWindow(300)).get_stats()
        api_stats = self._type_windows.get("api_rtt", SLAWindow(300)).get_stats()

        compliance = {}
        for name, stats, thresholds in [
            ("e2e", e2e_stats, ("e2e_p95_ms", "e2e_p99_ms")),
            ("api_rtt", api_stats, ("api_rtt_p95_ms", "api_rtt_p99_ms")),
        ]:
            if stats["count"] == 0:
                compliance[name] = SLACompliance.EXCELLENT.value
                continue

            p95_limit = self._sla_thresholds.get(thresholds[0], 500)
            p99_limit = self._sla_thresholds.get(thresholds[1], 1000)

            if stats["p99"] > p99_limit:
                compliance[name] = SLACompliance.CRITICAL.value
            elif stats["p95"] > p95_limit:
                compliance[name] = SLACompliance.DEGRADED.value
            elif stats["p95"] > p95_limit * 0.8:
                compliance[name] = SLACompliance.GOOD.value
            else:
                compliance[name] = SLACompliance.EXCELLENT.value

        return {
            "overall": compliance,
            "e2e_stats": e2e_stats,
            "api_rtt_stats": api_stats,
        }

    def get_health_score(self) -> float:
        """获取综合健康评分"""
        # 综合延迟、成功率、吞吐
        e2e_stats = self._type_windows.get("e2e", SLAWindow(300)).get_stats()
        success_rate = self._success_orders / max(self._total_orders, 1)

        # 延迟得分 (P95越接近阈值，分越低)
        p95 = e2e_stats.get("p95", 0)
        threshold = self._sla_thresholds.get("e2e_p95_ms", 500)
        latency_score = max(0, min(40, 40 * (1 - p95 / max(threshold, 1))))

        # 成功率得分
        success_score = max(0, min(40, 40 * success_rate))

        # 吞吐得分 (基于历史比较)
        tp = self.get_throughput_stats()
        base_throughput = 0.5
        throughput_score = min(20, 20 * (tp["orders_per_second"] / max(base_throughput, 0.01)))

        return round(latency_score + success_score + throughput_score, 1)

    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取当前活跃告警"""
        alerts = []
        e2e_stats = self._type_windows.get("e2e", SLAWindow(300)).get_stats()

        if e2e_stats.get("p99", 0) > self._sla_thresholds.get("e2e_p99_ms", 1000):
            alerts.append({
                "level": "critical",
                "type": "sla_breach",
                "message": f"E2E P99 latency {e2e_stats['p99']}ms exceeds SLA {self._sla_thresholds['e2e_p99_ms']}ms",
                "timestamp": datetime.now().isoformat(),
            })

        if self._adaptive_baseline:
            for key, baseline in self._adaptive_baseline.items():
                type_stats = self._type_windows.get(key, SLAWindow(300)).get_stats()
                if type_stats["p95"] > baseline * self._alert_thresholds["latency_spike_factor"]:
                    alerts.append({
                        "level": "warning",
                        "type": "latency_spike",
                        "message": f"{key} P95 {type_stats['p95']}ms > baseline {baseline}ms * {self._alert_thresholds['latency_spike_factor']}",
                        "timestamp": datetime.now().isoformat(),
                        "latency_type": key,
                    })

        success_rate = self._success_orders / max(self._total_orders, 1)
        if self._total_orders > 10 and success_rate < self._alert_thresholds["success_rate_min"]:
            alerts.append({
                "level": "critical",
                "type": "low_success_rate",
                "message": f"Order success rate {success_rate:.1%} below {self._alert_thresholds['success_rate_min']:.0%}",
                "timestamp": datetime.now().isoformat(),
            })

        return alerts

    def get_recent_events(self, limit: int = 50, event_type: str = None) -> List[Dict]:
        """获取最近事件"""
        events = list(self._events)
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return [e.to_dict() for e in events[-limit:]]

    def get_full_status(self) -> Dict[str, Any]:
        """获取完整监控状态"""
        return {
            "latency": {
                "1m": self._latency_windows["1m"].get_stats(),
                "5m": self._latency_windows["5m"].get_stats(),
                "15m": self._latency_windows["15m"].get_stats(),
                "1h": self._latency_windows["1h"].get_stats(),
                "by_type": {k: v.get_stats() for k, v in self._type_windows.items()},
            },
            "throughput": self.get_throughput_stats(),
            "sla": self.get_sla_status(),
            "health": {
                "score": self.get_health_score(),
                "health_history": list(self._health_history)[-20:],
            },
            "alerts": self.get_alerts(),
            "adaptive_baseline": self._adaptive_baseline,
        }
