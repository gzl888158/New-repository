"""
APM工具集成 - Application Performance Monitoring

功能：
1. 性能指标收集
2. 调用链追踪
3. 错误自动上报
4. 实时监控仪表盘
"""

import asyncio
import contextvars
import time
import threading
import psutil
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from loguru import logger
from collections import defaultdict


# ==================== 性能指标 ====================

class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"       # 计数器（累加）
    GAUGE = "gauge"           # 仪表（瞬时值）
    TIMER = "timer"           # 计时器（耗时）
    HISTOGRAM = "histogram"   # 直方图（分布）


class Metric:
    """性能指标"""
    
    def __init__(self, name: str, metric_type: MetricType, description: str = ""):
        self.name = name
        self.type = metric_type
        self.description = description
        self.value = 0
        self.count = 0
        self.sum = 0
        self.min = float('inf')
        self.max = float('-inf')
        self._lock = threading.Lock()
    
    def update(self, value: float):
        """更新指标值"""
        with self._lock:
            if self.type == MetricType.COUNTER:
                self.value += value
            elif self.type == MetricType.GAUGE:
                self.value = value
            elif self.type == MetricType.TIMER:
                self.count += 1
                self.sum += value
                self.min = min(self.min, value)
                self.max = max(self.max, value)
                self.value = self.sum / self.count if self.count > 0 else 0
            elif self.type == MetricType.HISTOGRAM:
                self.count += 1
                self.sum += value
                self.min = min(self.min, value)
                self.max = max(self.max, value)
                self.value = value
    
    def reset(self):
        """重置指标"""
        with self._lock:
            self.value = 0
            self.count = 0
            self.sum = 0
            self.min = float('inf')
            self.max = float('-inf')
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "value": self.value,
            "count": self.count,
            "sum": self.sum,
            "min": self.min if self.min != float('inf') else 0,
            "max": self.max if self.max != float('-inf') else 0
        }


# ==================== 调用链追踪 ====================

class SpanStatus(Enum):
    """Span状态"""
    OK = "ok"
    ERROR = "error"
    UNKNOWN = "unknown"


class Span:
    """调用链Span"""
    
    def __init__(self, name: str, parent_span: "Span" = None):
        self.name = name
        self.parent_span = parent_span
        self.trace_id = parent_span.trace_id if parent_span else self._generate_trace_id()
        self.span_id = self._generate_span_id()
        self.start_time = time.time()
        self.end_time = None
        self.duration = 0
        self.status = SpanStatus.OK
        self.tags: Dict[str, Any] = {}
        self.events: List[Dict[str, Any]] = []
        self._context_token = None
    
    def _generate_trace_id(self) -> str:
        """生成追踪ID"""
        import uuid
        return str(uuid.uuid4())[:16]
    
    def _generate_span_id(self) -> str:
        """生成Span ID"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def set_tag(self, key: str, value: Any):
        """设置标签"""
        self.tags[key] = value
    
    def add_event(self, name: str, attributes: Dict[str, Any] = None):
        """添加事件"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        })
    
    def finish(self, status: SpanStatus = SpanStatus.OK):
        """结束Span"""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.status = status

    def record_exception(self, exception: Exception):
        """记录异常标签与事件（不结束 span，结束交由 Tracer.finish_span 统一处理）。"""
        self.set_tag("error", True)
        self.set_tag("error_type", type(exception).__name__)
        self.set_tag("error_message", str(exception))
        self.add_event("exception", {
            "type": type(exception).__name__,
            "message": str(exception)
        })
        self.status = SpanStatus.ERROR

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span.span_id if self.parent_span else None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": round(self.duration * 1000, 2),
            "status": self.status.value,
            "tags": self.tags,
            "events": self.events
        }


# 当前上下文中的活动 span（跨 asyncio 任务/线程隔离，用于自动父级传播）
_current_span: "contextvars.ContextVar[Optional[Span]]" = contextvars.ContextVar(
    "apm_current_span", default=None
)


class Tracer:
    """调用链追踪器"""
    
    def __init__(self, service_name: str = "trading_system", slow_threshold_ms: float = 500.0):
        self._service_name = service_name
        self._current_spans: Dict[str, Span] = {}
        self._completed_spans: List[Span] = []
        self._max_spans = 1000
        self._slow_threshold_ms = slow_threshold_ms
        self._lock = threading.Lock()
        
        logger.info(f"Tracer initialized for {service_name} (slow_threshold={slow_threshold_ms}ms)")
    
    def start_span(self, name: str, parent_span: Span = None) -> Span:
        """开始Span。

        若未显式传入 parent_span，则自动继承当前上下文中的活动 span 作为父级，
        使嵌套调用（价格到达→计算→信号→风控→下单）串成同一条端到端 trace。
        """
        if parent_span is None:
            parent_span = _current_span.get()
        span = Span(name, parent_span)
        span._context_token = _current_span.set(span)
        
        with self._lock:
            self._current_spans[span.span_id] = span
        
        return span
    
    def finish_span(self, span: Span):
        """结束Span并恢复父级上下文"""
        span.finish(span.status)
        
        token = getattr(span, "_context_token", None)
        if token is not None:
            _current_span.reset(token)
            span._context_token = None
        
        with self._lock:
            if span.span_id in self._current_spans:
                del self._current_spans[span.span_id]
            
            self._completed_spans.append(span)
            
            if len(self._completed_spans) > self._max_spans:
                self._completed_spans = self._completed_spans[-self._max_spans:]
    
    def record_exception(self, span: Span, exception: Exception):
        """记录异常并结束 span"""
        span.record_exception(exception)
        self.finish_span(span)
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的追踪记录"""
        # 按trace_id分组
        traces: Dict[str, List[Span]] = defaultdict(list)
        
        with self._lock:
            for span in self._completed_spans[-100:]:
                traces[span.trace_id].append(span)
        
        # 转换为追踪记录
        result = []
        for trace_id, spans in traces.items():
            total_duration = sum(s.duration for s in spans)
            total_duration_ms = round(total_duration * 1000, 2)
            error_count = sum(1 for s in spans if s.status == SpanStatus.ERROR)
            
            result.append({
                "trace_id": trace_id,
                "span_count": len(spans),
                "total_duration_ms": total_duration_ms,
                "error_count": error_count,
                "is_slow": total_duration_ms > self._slow_threshold_ms,
                "slow_threshold_ms": self._slow_threshold_ms,
                "spans": [s.to_dict() for s in spans]
            })
        
        return sorted(result, key=lambda x: x["total_duration_ms"], reverse=True)[:limit]


# ==================== 性能监控器 ====================

class APMMonitor:
    """APM监控器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        
        # 指标存储
        self._metrics: Dict[str, Metric] = {}
        
        # 调用链追踪器
        self._tracer = Tracer(slow_threshold_ms=self._config.get("slow_trace_threshold_ms", 500.0))
        
        # 资源监控
        self._resource_metrics = {
            "cpu_percent": Metric("cpu.percent", MetricType.GAUGE, "CPU使用率"),
            "memory_percent": Metric("memory.percent", MetricType.GAUGE, "内存使用率"),
            "process_memory_mb": Metric("process.memory.mb", MetricType.GAUGE, "进程内存"),
            "network_io_bytes": Metric("network.io.bytes", MetricType.COUNTER, "网络IO字节")
        }
        
        # 交易指标
        self._trading_metrics = {
            "orders_placed": Metric("trading.orders.placed", MetricType.COUNTER, "下单数量"),
            "orders_filled": Metric("trading.orders.filled", MetricType.COUNTER, "成交数量"),
            "orders_cancelled": Metric("trading.orders.cancelled", MetricType.COUNTER, "取消数量"),
            "signals_generated": Metric("trading.signals.generated", MetricType.COUNTER, "信号数量"),
            "signals_rejected": Metric("trading.signals.rejected", MetricType.COUNTER, "拒绝信号"),
            "pnl_total": Metric("trading.pnl.total", MetricType.GAUGE, "总盈亏"),
            "pnl_daily": Metric("trading.pnl.daily", MetricType.GAUGE, "当日盈亏"),
            "win_rate": Metric("trading.win.rate", MetricType.GAUGE, "胜率")
        }
        
        # API指标
        self._api_metrics = {
            "api_requests_total": Metric("api.requests.total", MetricType.COUNTER, "API请求总数"),
            "api_requests_success": Metric("api.requests.success", MetricType.COUNTER, "成功请求"),
            "api_requests_failed": Metric("api.requests.failed", MetricType.COUNTER, "失败请求"),
            "api_latency": Metric("api.latency", MetricType.TIMER, "API延迟(ms)"),
            "api_rate_limit_hits": Metric("api.rate_limit.hits", MetricType.COUNTER, "限流次数")
        }
        
        # 合并所有指标
        self._metrics.update(self._resource_metrics)
        self._metrics.update(self._trading_metrics)
        self._metrics.update(self._api_metrics)
        
        # 监控任务
        self._monitor_task = None
        self._monitor_interval = 5  # 秒
        
        # 错误上报
        self._error_reports: List[Dict[str, Any]] = []
        self._max_error_reports = 100
        
        logger.info("APMMonitor initialized")
    
    def get_metric(self, name: str) -> Metric:
        """获取指标"""
        if name not in self._metrics:
            self._metrics[name] = Metric(name, MetricType.GAUGE)
        
        return self._metrics[name]
    
    def record_order(self, status: str):
        """记录订单状态"""
        if status == "placed":
            self._trading_metrics["orders_placed"].update(1)
        elif status == "filled":
            self._trading_metrics["orders_filled"].update(1)
        elif status == "cancelled":
            self._trading_metrics["orders_cancelled"].update(1)
    
    def record_signal(self, accepted: bool):
        """记录信号"""
        if accepted:
            self._trading_metrics["signals_generated"].update(1)
        else:
            self._trading_metrics["signals_rejected"].update(1)
    
    def record_api_call(self, success: bool, latency_ms: float):
        """记录API调用"""
        self._api_metrics["api_requests_total"].update(1)
        
        if success:
            self._api_metrics["api_requests_success"].update(1)
        else:
            self._api_metrics["api_requests_failed"].update(1)
        
        self._api_metrics["api_latency"].update(latency_ms)
    
    def record_pnl(self, total_pnl: float, daily_pnl: float):
        """记录盈亏"""
        self._trading_metrics["pnl_total"].update(total_pnl)
        self._trading_metrics["pnl_daily"].update(daily_pnl)
    
    def report_error(self, error: Dict[str, Any]):
        """上报错误"""
        self._error_reports.append(error)
        
        if len(self._error_reports) > self._max_error_reports:
            self._error_reports = self._error_reports[-self._max_error_reports:]
    
    def _update_resource_metrics(self):
        """更新资源指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self._resource_metrics["cpu_percent"].update(cpu_percent)
            
            # 内存使用率
            memory_info = psutil.virtual_memory()
            self._resource_metrics["memory_percent"].update(memory_info.percent)
            
            # 进程内存
            process = psutil.Process()
            process_memory_mb = process.memory_info().rss / 1024 / 1024
            self._resource_metrics["process_memory_mb"].update(process_memory_mb)
            
            # 网络IO
            try:
                net_io = psutil.net_io_counters()
                self._resource_metrics["network_io_bytes"].update(net_io.bytes_sent + net_io.bytes_recv)
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Failed to update resource metrics: {e}")
    
    async def start_monitoring(self):
        """启动监控"""
        if self._monitor_task is not None:
            return
        
        async def monitor_loop():
            while True:
                self._update_resource_metrics()
                await asyncio.sleep(self._monitor_interval)
        
        self._monitor_task = asyncio.create_task(monitor_loop())
        logger.info("APM monitoring started")
    
    async def stop_monitoring(self):
        """停止监控"""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
            logger.info("APM monitoring stopped")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """获取所有指标"""
        return {name: metric.to_dict() for name, metric in self._metrics.items()}
    
    def get_metrics_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """按分类获取指标"""
        prefix_map = {
            "cpu": "cpu.",
            "memory": "memory.",
            "process": "process.",
            "trading": "trading.",
            "api": "api."
        }
        
        prefix = prefix_map.get(category)
        if not prefix:
            return {}
        
        return {name: metric.to_dict() for name, metric in self._metrics.items() if name.startswith(prefix)}
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的追踪记录"""
        return self._tracer.get_recent_traces(limit)
    
    def get_error_reports(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取错误报告"""
        return self._error_reports[-limit:]
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康摘要"""
        metrics = self.get_all_metrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "resource": {
                "cpu_percent": metrics.get("cpu.percent", {}).get("value", 0),
                "memory_percent": metrics.get("memory.percent", {}).get("value", 0),
                "process_memory_mb": metrics.get("process.memory.mb", {}).get("value", 0)
            },
            "trading": {
                "orders_placed": metrics.get("trading.orders.placed", {}).get("value", 0),
                "orders_filled": metrics.get("trading.orders.filled", {}).get("value", 0),
                "signals_generated": metrics.get("trading.signals.generated", {}).get("value", 0),
                "pnl_total": metrics.get("trading.pnl.total", {}).get("value", 0),
                "pnl_daily": metrics.get("trading.pnl.daily", {}).get("value", 0)
            },
            "api": {
                "requests_total": metrics.get("api.requests.total", {}).get("value", 0),
                "success_rate": self._calculate_api_success_rate(),
                "avg_latency_ms": metrics.get("api.latency", {}).get("value", 0)
            }
        }
    
    def _calculate_api_success_rate(self) -> float:
        """计算API成功率"""
        total = self._api_metrics["api_requests_total"].value
        success = self._api_metrics["api_requests_success"].value
        
        if total == 0:
            return 100.0
        
        return round(success / total * 100, 2)


# ==================== 性能监控装饰器 ====================

def monitor_performance(metric_name: str = None):
    """
    性能监控装饰器
    
    用法：
    @monitor_performance("api.place_order")
    async def place_order(self, ...):
        ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                return await func(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                try:
                    apm = get_apm_monitor()
                    apm.record_api_call(success, duration_ms)
                except Exception:
                    pass
        
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            success = True
            
            try:
                return func(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                
                try:
                    apm = get_apm_monitor()
                    apm.record_api_call(success, duration_ms)
                except Exception:
                    pass
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def trace_function(span_name: str = None):
    """
    调用链追踪装饰器
    
    用法：
    @trace_function("place_order")
    async def place_order(self, ...):
        ...
    """
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            name = span_name or func.__name__
            tracer = get_apm_monitor()._tracer
            span = tracer.start_span(name)
            
            try:
                result = await func(*args, **kwargs)
                span.set_tag("result", "success")
                return result
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                tracer.finish_span(span)
        
        def sync_wrapper(*args, **kwargs):
            name = span_name or func.__name__
            tracer = get_apm_monitor()._tracer
            span = tracer.start_span(name)
            
            try:
                result = func(*args, **kwargs)
                span.set_tag("result", "success")
                return result
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                tracer.finish_span(span)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ==================== 端到端交易链路追踪 ====================

# 端到端交易阶段顺序：价格到达 → 计算 → 信号 → 风控 → 下单
TRADE_FLOW_STAGES = (
    "market_data_arrival",
    "signal_calculation",
    "signal_generation",
    "risk_check",
    "order_placement",
)


class TradeFlowTrace:
    """端到端交易链路追踪器。

    将「价格到达→计算→信号→风控→下单」串成一条 trace（共享 trace_id），
    阶段通过 :meth:`start_stage` / :meth:`finish_stage` 自动作为根 span 的子 span。
    根 span 总耗时 > slow_threshold_ms 时在 ``get_recent_traces()`` 中被标记 ``is_slow``。

    用法::

        flow = TradeFlowTrace(symbol="BTC-USDT-SWAP")
        s = flow.start_stage("market_data_arrival")
        ...
        flow.finish_stage(s)
        flow.finish()
    """

    def __init__(self, symbol: str = "", tracer: "Tracer" = None):
        self._tracer = tracer or get_apm_monitor()._tracer
        self._root = self._tracer.start_span("trade_flow")
        if symbol:
            self._root.set_tag("symbol", symbol)
        self._finished = False

    @property
    def trace_id(self) -> str:
        return self._root.trace_id

    def start_stage(self, name: str) -> Span:
        """开始一个阶段 span（自动继承根 span 为父级）。"""
        return self._tracer.start_span(name)

    def finish_stage(self, span: Span, status: SpanStatus = SpanStatus.OK):
        """结束一个阶段 span。"""
        span.status = status
        self._tracer.finish_span(span)

    def finish(self, status: SpanStatus = SpanStatus.OK):
        """结束整条链路。"""
        if self._finished:
            return
        self._root.status = status
        self._tracer.finish_span(self._root)
        self._finished = True

    def __enter__(self) -> "TradeFlowTrace":
        return self

    def __exit__(self, exc_type, exc, tb):
        self.finish(SpanStatus.ERROR if exc_type else SpanStatus.OK)
        return False


# ==================== 全局单例 ====================

_apm_monitor: Optional[APMMonitor] = None


def get_apm_monitor(config: Dict[str, Any] = None) -> APMMonitor:
    """获取APM监控器单例"""
    global _apm_monitor
    
    if _apm_monitor is None:
        _apm_monitor = APMMonitor(config)
    
    return _apm_monitor