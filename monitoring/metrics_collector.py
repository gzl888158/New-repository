"""
监控指标收集器
收集业务指标、性能指标、系统指标
"""
import asyncio
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from loguru import logger
import json
import os
from core.atomic_writer import atomic_write_json


@dataclass
class MetricValue:
    """指标值"""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
        }


@dataclass
class MetricSeries:
    """指标序列"""
    name: str
    description: str = ""
    unit: str = ""
    metric_type: str = "gauge"  # gauge / counter / histogram
    values: List[MetricValue] = field(default_factory=list)
    
    def add_value(self, value: float, labels: Optional[Dict[str, str]] = None):
        """添加值"""
        self.values.append(MetricValue(
            name=self.name,
            value=value,
            labels=labels or {},
        ))
    
    def get_latest(self) -> Optional[MetricValue]:
        """获取最新值"""
        if self.values:
            return self.values[-1]
        return None
    
    def get_values(self, limit: int = 100) -> List[MetricValue]:
        """获取最近N个值"""
        return self.values[-limit:]
    
    def get_statistics(self, window: int = 100) -> Dict[str, float]:
        """获取统计信息"""
        if not self.values:
            return {"min": 0, "max": 0, "avg": 0, "count": 0}
        
        recent_values = [v.value for v in self.values[-window:]]
        
        return {
            "min": min(recent_values),
            "max": max(recent_values),
            "avg": sum(recent_values) / len(recent_values),
            "count": len(recent_values),
            "latest": recent_values[-1] if recent_values else 0,
        }


class MetricsCollector:
    """指标收集器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, max_series_length: int = 1000):
        if not hasattr(self, '_initialized'):
            self.max_series_length = max_series_length
            self._metrics: Dict[str, MetricSeries] = {}
            self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
            self._collection_task: Optional[asyncio.Task] = None
            self._running = False
            self._initialized = True
            
            self._register_default_metrics()
    
    def _register_default_metrics(self):
        """注册默认指标"""
        self.register_metric("trading_pnl_total", "累计盈亏", "USDT", "gauge")
        self.register_metric("trading_pnl_daily", "每日盈亏", "USDT", "gauge")
        self.register_metric("trading_trades_total", "总交易次数", "count", "counter")
        self.register_metric("trading_trades_winning", "盈利交易次数", "count", "counter")
        self.register_metric("trading_trades_losing", "亏损交易次数", "count", "counter")
        self.register_metric("trading_win_rate", "胜率", "%", "gauge")
        self.register_metric("trading_drawdown", "回撤", "%", "gauge")
        
        self.register_metric("system_cpu_usage", "CPU使用率", "%", "gauge")
        self.register_metric("system_memory_usage", "内存使用", "MB", "gauge")
        self.register_metric("system_latency_api", "API延迟", "ms", "gauge")
        self.register_metric("system_latency_order", "订单延迟", "ms", "gauge")
        
        self.register_metric("strategy_pnl", "策略盈亏", "USDT", "gauge")
        self.register_metric("strategy_trades", "策略交易次数", "count", "counter")
        self.register_metric("strategy_win_rate", "策略胜率", "%", "gauge")
        
        self.register_metric("position_count", "持仓数量", "count", "gauge")
        self.register_metric("position_unrealized_pnl", "未实现盈亏", "USDT", "gauge")
        self.register_metric("position_total_leverage", "总杠杆", "x", "gauge")
        
        self.register_metric("risk_score", "风险评分", "score", "gauge")
        self.register_metric("risk_exposure", "风险敞口", "USDT", "gauge")
    
    def register_metric(self, name: str, description: str = "", unit: str = "", metric_type: str = "gauge"):
        """注册指标"""
        if name not in self._metrics:
            self._metrics[name] = MetricSeries(
                name=name,
                description=description,
                unit=unit,
                metric_type=metric_type,
            )
    
    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """记录指标值"""
        if name not in self._metrics:
            self.register_metric(name)
        
        self._metrics[name].add_value(value, labels)
        
        if len(self._metrics[name].values) > self.max_series_length:
            self._metrics[name].values = self._metrics[name].values[-self.max_series_length:]
        
        self._notify_callbacks(name, value, labels)
    
    def increment(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """增加计数器"""
        current = self._metrics[name].get_latest()
        new_value = (current.value if current else 0) + value
        self.record(name, new_value, labels)
    
    def get_metric(self, name: str) -> Optional[MetricSeries]:
        """获取指标"""
        return self._metrics.get(name)
    
    def get_value(self, name: str) -> Optional[float]:
        """获取指标最新值"""
        metric = self._metrics.get(name)
        if metric and metric.values:
            return metric.values[-1].value
        return None
    
    def get_statistics(self, name: str, window: int = 100) -> Dict[str, float]:
        """获取指标统计"""
        metric = self._metrics.get(name)
        if metric:
            return metric.get_statistics(window)
        return {}
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
        result = {}
        for name, series in self._metrics.items():
            latest = series.get_latest()
            stats = series.get_statistics()
            
            result[name] = {
                "description": series.description,
                "unit": series.unit,
                "type": series.metric_type,
                "latest": latest.value if latest else None,
                "statistics": stats,
            }
        
        return result
    
    def get_business_metrics(self) -> Dict[str, Any]:
        """获取业务指标"""
        prefix = "trading_"
        return {k: v for k, v in self.get_all_metrics().items() if k.startswith(prefix)}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        prefix = "system_"
        return {k: v for k, v in self.get_all_metrics().items() if k.startswith(prefix)}
    
    def get_strategy_metrics(self, strategy_name: Optional[str] = None) -> Dict[str, Any]:
        """获取策略指标"""
        prefix = "strategy_"
        metrics = {}
        
        for name, series in self._metrics.items():
            if name.startswith(prefix):
                if strategy_name:
                    for value in series.values[-10:]:
                        if value.labels.get("strategy") == strategy_name:
                            if name not in metrics:
                                metrics[name] = {"values": []}
                            metrics[name]["values"].append(value.to_dict())
                else:
                    latest = series.get_latest()
                    metrics[name] = {
                        "description": series.description,
                        "latest": latest.value if latest else None,
                    }
        
        return metrics
    
    def register_callback(self, metric_pattern: str, callback: Callable[[str, float, Dict], None]):
        """注册指标变更回调"""
        self._callbacks[metric_pattern].append(callback)
    
    def _notify_callbacks(self, name: str, value: float, labels: Optional[Dict[str, str]]):
        """通知回调"""
        for pattern, callbacks in self._callbacks.items():
            if pattern == "*" or pattern == name or name.startswith(pattern):
                for callback in callbacks:
                    try:
                        callback(name, value, labels or {})
                    except Exception as e:
                        logger.error(f"Metric callback error: {e}")
    
    def start_collection(self, interval_seconds: float = 60):
        """开始定时收集"""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop(interval_seconds))
        logger.info(f"Metrics collection started with interval {interval_seconds}s")
    
    def stop_collection(self):
        """停止定时收集"""
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
        logger.info("Metrics collection stopped")
    
    async def _collection_loop(self, interval: float):
        """收集循环"""
        while self._running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record("system_cpu_usage", cpu_percent)
            
            memory = psutil.virtual_memory()
            self.record("system_memory_usage", memory.used / (1024 * 1024))
            
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"System metrics collection error: {e}")
    
    def export_prometheus(self) -> str:
        """导出为Prometheus格式"""
        lines = []
        
        for name, series in self._metrics.items():
            lines.append(f"# HELP {name} {series.description}")
            lines.append(f"# TYPE {name} {series.metric_type}")
            
            latest = series.get_latest()
            if latest:
                labels_str = ""
                if latest.labels:
                    labels_str = "{" + ",".join(f'{k}="{v}"' for k, v in latest.labels.items()) + "}"
                
                lines.append(f"{name}{labels_str} {latest.value}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def export_json(self) -> str:
        """导出为JSON格式"""
        return json.dumps(self.get_all_metrics(), indent=2, default=str)
    
    def save_to_file(self, filepath: str):
        """保存到文件"""
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
        }
        
        for name, series in self._metrics.items():
            if series.values:
                data["metrics"][name] = {
                    "description": series.description,
                    "unit": series.unit,
                    "type": series.metric_type,
                    "values": [v.to_dict() for v in series.values[-100:]],
                }
        
        atomic_write_json(filepath, data)
    
    def load_from_file(self, filepath: str):
        """从文件加载"""
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for name, metric_data in data.get("metrics", {}).items():
                self.register_metric(
                    name,
                    metric_data.get("description", ""),
                    metric_data.get("unit", ""),
                    metric_data.get("type", "gauge"),
                )
                
                for value_data in metric_data.get("values", []):
                    self._metrics[name].add_value(
                        value_data["value"],
                        value_data.get("labels"),
                    )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load metrics from file: {e}")
            return False
    
    def clear(self):
        """清空所有指标"""
        self._metrics.clear()
        self._register_default_metrics()


def get_metrics_collector() -> MetricsCollector:
    """获取指标收集器单例"""
    return MetricsCollector()