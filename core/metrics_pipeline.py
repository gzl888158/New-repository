"""
生产级性能指标流水线
====================
核心定位：统一的生产级指标采集、聚合、导出流水线，支持多维度性能监控与分析。

功能：
- 多源指标采集（MetricsCollector / PerformanceMonitor / 策略指标 / 交易指标）
- 指标聚合流水线（预聚合 → 窗口聚合 → 派生指标）
- 多格式导出（Prometheus / JSON / InfluxDB Line Protocol / CSV）
- 实时指标推送（WebSocket / 回调）
- 指标快照持久化与回放
- 指标健康检查与异常检测
- 时间序列数据库集成
- 指标标签体系与维度下钻

架构：
  MetricsPipeline
  ├── MetricRegistry（指标注册中心）
  ├── MetricCollector（多源采集器）
  ├── AggregationEngine（聚合引擎）
  │   ├── PreAggregator（预聚合）
  │   ├── WindowAggregator（窗口聚合）
  │   └── DerivedMetrics（派生指标）
  ├── MetricExporter（指标导出器）
  │   ├── PrometheusExporter
  │   ├── JSONExporter
  │   ├── InfluxDBExporter
  │   └── CSVExporter
  ├── MetricHealthChecker（指标健康检查）
  └── MetricSnapshot（指标快照）
"""

import asyncio
import json
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from loguru import logger

try:
    from monitoring.metrics_collector import MetricsCollector, get_metrics_collector
except ImportError:
    MetricsCollector = None
    get_metrics_collector = None


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class MetricType(Enum):
    """指标类型"""
    GAUGE = "gauge"          # 瞬时值
    COUNTER = "counter"      # 累加计数器
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"      # 摘要


class MetricCategory(Enum):
    """指标类别"""
    SYSTEM = "system"            # 系统指标
    TRADING = "trading"          # 交易指标
    STRATEGY = "strategy"        # 策略指标
    RISK = "risk"                # 风险指标
    CAPITAL = "capital"          # 资金指标
    PERFORMANCE = "performance"  # 性能指标
    CUSTOM = "custom"            # 自定义


@dataclass
class MetricDefinition:
    """指标定义"""
    name: str
    description: str = ""
    unit: str = ""
    type: MetricType = MetricType.GAUGE
    category: MetricCategory = MetricCategory.CUSTOM
    labels: List[str] = field(default_factory=list)
    buckets: List[float] = field(default_factory=list)  # histogram 桶
    quantiles: List[float] = field(default_factory=list)  # summary 分位数
    help_text: str = ""


@dataclass
class MetricSample:
    """指标样本"""
    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "labels": self.labels,
            "timestamp": self.timestamp,
        }


@dataclass
class AggregatedMetric:
    """聚合指标"""
    name: str
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    avg: float = 0.0
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    stddev: float = 0.0
    latest: float = 0.0
    window_start: float = 0.0
    window_end: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "sum": self.sum,
            "min": self.min if self.min != float('inf') else 0,
            "max": self.max if self.max != float('-inf') else 0,
            "avg": self.avg,
            "p50": self.p50,
            "p95": self.p95,
            "p99": self.p99,
            "stddev": self.stddev,
            "latest": self.latest,
            "window_start": self.window_start,
            "window_end": self.window_end,
        }


# ═══════════════════════════════════════════════════════════════
# MetricRegistry
# ═══════════════════════════════════════════════════════════════

class MetricRegistry:
    """指标注册中心"""

    def __init__(self):
        self._metrics: Dict[str, MetricDefinition] = {}
        self._labels: Dict[str, Dict[str, str]] = defaultdict(dict)  # name -> {label_key: label_value}
        self._lock = threading.RLock()

        # 注册默认指标
        self._register_defaults()

    def _register_defaults(self):
        """注册默认系统指标"""
        defaults = [
            # 系统指标
            MetricDefinition("system_cpu_usage_pct", "CPU使用率", "%", MetricType.GAUGE, MetricCategory.SYSTEM),
            MetricDefinition("system_memory_usage_mb", "内存使用量", "MB", MetricType.GAUGE, MetricCategory.SYSTEM),
            MetricDefinition("system_memory_usage_pct", "内存使用率", "%", MetricType.GAUGE, MetricCategory.SYSTEM),
            MetricDefinition("system_disk_free_gb", "磁盘可用空间", "GB", MetricType.GAUGE, MetricCategory.SYSTEM),
            MetricDefinition("system_uptime_sec", "运行时间", "s", MetricType.COUNTER, MetricCategory.SYSTEM),
            # 交易指标
            MetricDefinition("trading_pnl_total", "累计盈亏", "USDT", MetricType.GAUGE, MetricCategory.TRADING),
            MetricDefinition("trading_pnl_daily", "每日盈亏", "USDT", MetricType.GAUGE, MetricCategory.TRADING),
            MetricDefinition("trading_trades_total", "总交易次数", "count", MetricType.COUNTER, MetricCategory.TRADING),
            MetricDefinition("trading_win_rate", "胜率", "%", MetricType.GAUGE, MetricCategory.TRADING),
            MetricDefinition("trading_fees_total", "累计手续费", "USDT", MetricType.COUNTER, MetricCategory.TRADING),
            MetricDefinition("trading_slippage_total", "累计滑点", "USDT", MetricType.COUNTER, MetricCategory.TRADING),
            MetricDefinition("trading_order_latency_ms", "订单延迟", "ms", MetricType.HISTOGRAM, MetricCategory.TRADING,
                           buckets=[10, 50, 100, 200, 500, 1000, 2000, 5000]),
            MetricDefinition("trading_api_latency_ms", "API延迟", "ms", MetricType.HISTOGRAM, MetricCategory.TRADING,
                           buckets=[50, 100, 200, 500, 1000, 2000, 5000]),
            MetricDefinition("trading_order_count", "活跃订单数", "count", MetricType.GAUGE, MetricCategory.TRADING),
            # 策略指标
            MetricDefinition("strategy_pnl", "策略盈亏", "USDT", MetricType.GAUGE, MetricCategory.STRATEGY,
                           labels=["strategy"]),
            MetricDefinition("strategy_trades", "策略交易次数", "count", MetricType.COUNTER, MetricCategory.STRATEGY,
                           labels=["strategy"]),
            MetricDefinition("strategy_win_rate", "策略胜率", "%", MetricType.GAUGE, MetricCategory.STRATEGY,
                           labels=["strategy"]),
            MetricDefinition("strategy_sharpe", "策略夏普比率", "", MetricType.GAUGE, MetricCategory.STRATEGY,
                           labels=["strategy"]),
            MetricDefinition("strategy_drawdown", "策略回撤", "%", MetricType.GAUGE, MetricCategory.STRATEGY,
                           labels=["strategy"]),
            MetricDefinition("strategy_trades_per_hour", "策略每小时交易", "count", MetricType.GAUGE, MetricCategory.STRATEGY,
                           labels=["strategy"]),
            # 风险指标
            MetricDefinition("risk_score", "综合风险评分", "score", MetricType.GAUGE, MetricCategory.RISK),
            MetricDefinition("risk_drawdown_pct", "回撤百分比", "%", MetricType.GAUGE, MetricCategory.RISK),
            MetricDefinition("risk_margin_ratio", "保证金率", "%", MetricType.GAUGE, MetricCategory.RISK),
            MetricDefinition("risk_total_exposure", "总风险敞口", "USDT", MetricType.GAUGE, MetricCategory.RISK),
            MetricDefinition("risk_leverage_max", "最大杠杆", "x", MetricType.GAUGE, MetricCategory.RISK),
            MetricDefinition("risk_direction_imbalance", "方向失衡", "", MetricType.GAUGE, MetricCategory.RISK),
            MetricDefinition("risk_concentration_max", "最大集中度", "", MetricType.GAUGE, MetricCategory.RISK),
            # 资金指标
            MetricDefinition("capital_total", "总资金", "USDT", MetricType.GAUGE, MetricCategory.CAPITAL),
            MetricDefinition("capital_available", "可用资金", "USDT", MetricType.GAUGE, MetricCategory.CAPITAL),
            MetricDefinition("capital_allocated", "已分配资金", "USDT", MetricType.GAUGE, MetricCategory.CAPITAL),
            MetricDefinition("capital_efficiency", "资金效率", "%", MetricType.GAUGE, MetricCategory.CAPITAL),
            MetricDefinition("capital_idle_pct", "闲置资金比例", "%", MetricType.GAUGE, MetricCategory.CAPITAL),
            MetricDefinition("capital_attrition_daily", "每日磨损", "USDT", MetricType.GAUGE, MetricCategory.CAPITAL),
            # 性能指标
            MetricDefinition("perf_signal_process_ms", "信号处理延迟", "ms", MetricType.HISTOGRAM, MetricCategory.PERFORMANCE,
                           buckets=[5, 10, 20, 50, 100, 200, 500]),
            MetricDefinition("perf_order_place_ms", "下单延迟", "ms", MetricType.HISTOGRAM, MetricCategory.PERFORMANCE,
                           buckets=[10, 20, 50, 100, 200, 500, 1000]),
            MetricDefinition("perf_ws_latency_ms", "WebSocket延迟", "ms", MetricType.GAUGE, MetricCategory.PERFORMANCE),
            MetricDefinition("perf_queue_size", "队列大小", "count", MetricType.GAUGE, MetricCategory.PERFORMANCE),
            MetricDefinition("perf_goroutine_count", "协程数", "count", MetricType.GAUGE, MetricCategory.PERFORMANCE),
        ]
        for m in defaults:
            self.register(m)

    def register(self, definition: MetricDefinition):
        """注册指标"""
        with self._lock:
            self._metrics[definition.name] = definition

    def unregister(self, name: str):
        """注销指标"""
        with self._lock:
            self._metrics.pop(name, None)

    def get(self, name: str) -> Optional[MetricDefinition]:
        """获取指标定义"""
        return self._metrics.get(name)

    def get_all(self) -> Dict[str, MetricDefinition]:
        """获取所有指标定义"""
        with self._lock:
            return dict(self._metrics)

    def get_by_category(self, category: MetricCategory) -> List[MetricDefinition]:
        """按类别获取指标"""
        with self._lock:
            return [m for m in self._metrics.values() if m.category == category]

    def get_categories(self) -> List[Dict[str, Any]]:
        """获取所有类别"""
        return [
            {"value": c.value, "label": c.value}
            for c in MetricCategory
        ]

    def set_label(self, name: str, key: str, value: str):
        """设置全局标签"""
        with self._lock:
            self._labels[name][key] = value

    def get_labels(self, name: str) -> Dict[str, str]:
        """获取全局标签"""
        with self._lock:
            return dict(self._labels.get(name, {}))


# ═══════════════════════════════════════════════════════════════
# AggregationEngine
# ═══════════════════════════════════════════════════════════════

class AggregationEngine:
    """聚合引擎"""

    def __init__(self, window_sec: float = 60.0, max_samples: int = 1000):
        self._window_sec = window_sec
        self._max_samples = max_samples
        self._samples: Dict[str, List[MetricSample]] = defaultdict(list)
        self._lock = threading.RLock()

    def add_sample(self, sample: MetricSample):
        """添加样本"""
        with self._lock:
            key = sample.name
            self._samples[key].append(sample)

            # 清理过期样本
            cutoff = time.time() - self._window_sec
            self._samples[key] = [
                s for s in self._samples[key]
                if s.timestamp >= cutoff
            ]

            # 限制最大样本数
            if len(self._samples[key]) > self._max_samples:
                self._samples[key] = self._samples[key][-self._max_samples:]

    def aggregate(self, name: str, window_sec: Optional[float] = None) -> Optional[AggregatedMetric]:
        """聚合指定指标"""
        with self._lock:
            samples = self._samples.get(name, [])
            if not samples:
                return None

            window = window_sec or self._window_sec
            cutoff = time.time() - window
            recent = [s for s in samples if s.timestamp >= cutoff]

            if not recent:
                return None

            return self._compute_aggregation(name, recent)

    def aggregate_all(self, window_sec: Optional[float] = None) -> Dict[str, AggregatedMetric]:
        """聚合所有指标"""
        result = {}
        with self._lock:
            for name in list(self._samples.keys()):
                agg = self.aggregate(name, window_sec)
                if agg:
                    result[name] = agg
        return result

    def aggregate_by_labels(self, name: str, label_key: str,
                           window_sec: Optional[float] = None) -> Dict[str, AggregatedMetric]:
        """按标签聚合"""
        with self._lock:
            samples = self._samples.get(name, [])
            if not samples:
                return {}

            window = window_sec or self._window_sec
            cutoff = time.time() - window
            recent = [s for s in samples if s.timestamp >= cutoff]

            grouped = defaultdict(list)
            for s in recent:
                label_val = s.labels.get(label_key, "__unknown__")
                grouped[label_val].append(s)

            result = {}
            for label_val, group_samples in grouped.items():
                agg = self._compute_aggregation(f"{name}:{label_key}={label_val}", group_samples)
                if agg:
                    result[label_val] = agg

            return result

    def _compute_aggregation(self, name: str, samples: List[MetricSample]) -> AggregatedMetric:
        """计算聚合指标"""
        values = [s.value for s in samples]
        count = len(values)
        total = sum(values)
        avg = total / count if count > 0 else 0.0

        sorted_vals = sorted(values)
        p50 = self._percentile(sorted_vals, 0.50)
        p95 = self._percentile(sorted_vals, 0.95)
        p99 = self._percentile(sorted_vals, 0.99)

        if count > 1:
            variance = sum((v - avg) ** 2 for v in values) / count
            stddev = variance ** 0.5
        else:
            stddev = 0.0

        return AggregatedMetric(
            name=name,
            count=count,
            sum=total,
            min=min(values) if values else 0,
            max=max(values) if values else 0,
            avg=avg,
            p50=p50,
            p95=p95,
            p99=p99,
            stddev=stddev,
            latest=values[-1] if values else 0,
            window_start=samples[0].timestamp if samples else 0,
            window_end=samples[-1].timestamp if samples else 0,
        )

    @staticmethod
    def _percentile(sorted_values: List[float], p: float) -> float:
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_values):
            return sorted_values[f] + c * (sorted_values[f + 1] - sorted_values[f])
        return sorted_values[f]

    def get_sample_count(self, name: str) -> int:
        """获取指标样本数"""
        with self._lock:
            return len(self._samples.get(name, []))

    def clear(self, name: Optional[str] = None):
        """清除指标数据"""
        with self._lock:
            if name:
                self._samples.pop(name, None)
            else:
                self._samples.clear()


# ═══════════════════════════════════════════════════════════════
# MetricExporter
# ═══════════════════════════════════════════════════════════════

class MetricExporter:
    """指标导出器"""

    def __init__(self, registry: MetricRegistry):
        self._registry = registry
        self._exporters: Dict[str, Any] = {}

    def export_prometheus(self, metrics: Dict[str, AggregatedMetric]) -> str:
        """导出为Prometheus格式"""
        lines = []

        for name, agg in metrics.items():
            definition = self._registry.get(name)
            if definition:
                lines.append(f"# HELP {name} {definition.description}")
                lines.append(f"# TYPE {name} {definition.type.value}")

            # 聚合指标作为标签
            labels = {}
            if definition and definition.labels:
                for lk in definition.labels:
                    global_label = self._registry.get_labels(name).get(lk, "")
                    if global_label:
                        labels[lk] = global_label

            labels_str = ""
            if labels:
                labels_str = "{" + ",".join(f'{k}="{v}"' for k, v in labels.items()) + "}"

            lines.append(f"{name}{labels_str} {agg.latest}")

            # 附加统计
            lines.append(f"# {name}_avg {agg.avg}")
            lines.append(f"# {name}_max {agg.max}")
            lines.append(f"# {name}_min {agg.min}")
            lines.append(f"# {name}_count {agg.count}")

            lines.append("")

        return "\n".join(lines)

    def export_json(self, metrics: Dict[str, AggregatedMetric],
                   include_definitions: bool = True) -> str:
        """导出为JSON格式"""
        output = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
        }

        if include_definitions:
            output["definitions"] = {
                name: {
                    "description": d.description,
                    "unit": d.unit,
                    "type": d.type.value,
                    "category": d.category.value,
                }
                for name, d in self._registry.get_all().items()
            }

        for name, agg in metrics.items():
            output["metrics"][name] = agg.to_dict()

        return json.dumps(output, indent=2, ensure_ascii=False)

    def export_influxdb(self, metrics: Dict[str, AggregatedMetric],
                       measurement: str = "trading_metrics",
                       tags: Dict[str, str] = None) -> str:
        """导出为InfluxDB Line Protocol格式"""
        lines = []
        tags = tags or {}
        tags_str = "," + ",".join(f"{k}={v}" for k, v in tags.items()) if tags else ""

        for name, agg in metrics.items():
            # InfluxDB line protocol:
            # measurement,tag1=val1,tag2=val2 field1=val1,field2=val2 timestamp
            ts_ns = int(agg.window_end * 1e9)
            fields = (
                f"latest={agg.latest},"
                f"avg={agg.avg},"
                f"max={agg.max},"
                f"min={agg.min},"
                f"count={agg.count}i,"
                f"p50={agg.p50},"
                f"p95={agg.p95},"
                f"p99={agg.p99},"
                f"stddev={agg.stddev}"
            )
            line = f"{measurement},metric={name}{tags_str} {fields} {ts_ns}"
            lines.append(line)

        return "\n".join(lines)

    def export_csv(self, metrics: Dict[str, AggregatedMetric]) -> str:
        """导出为CSV格式"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # 表头
        writer.writerow([
            "metric", "count", "sum", "min", "max", "avg",
            "p50", "p95", "p99", "stddev", "latest", "timestamp"
        ])

        for name, agg in sorted(metrics.items()):
            writer.writerow([
                name, agg.count, agg.sum, agg.min, agg.max, agg.avg,
                agg.p50, agg.p95, agg.p99, agg.stddev, agg.latest,
                datetime.fromtimestamp(agg.window_end).isoformat(),
            ])

        return output.getvalue()


# ═══════════════════════════════════════════════════════════════
# DerivedMetricsEngine
# ═══════════════════════════════════════════════════════════════

class DerivedMetricsEngine:
    """派生指标计算引擎"""

    @staticmethod
    def compute_sharpe_ratio(daily_returns: List[float], risk_free_rate: float = 0.03) -> float:
        """计算夏普比率"""
        if not daily_returns or len(daily_returns) < 2:
            return 0.0
        avg_return = sum(daily_returns) / len(daily_returns)
        if avg_return == 0:
            return 0.0
        variance = sum((r - avg_return) ** 2 for r in daily_returns) / (len(daily_returns) - 1)
        stddev = variance ** 0.5
        if stddev == 0:
            return 0.0
        daily_rf = risk_free_rate / 365
        return (avg_return - daily_rf) / stddev * (252 ** 0.5)

    @staticmethod
    def compute_sortino_ratio(daily_returns: List[float], risk_free_rate: float = 0.03) -> float:
        """计算索提诺比率"""
        if not daily_returns or len(daily_returns) < 2:
            return 0.0
        avg_return = sum(daily_returns) / len(daily_returns)
        daily_rf = risk_free_rate / 365
        downside = [min(0, r - daily_rf) for r in daily_returns]
        downside_variance = sum(d ** 2 for d in downside) / (len(downside) - 1) if len(downside) > 1 else 0
        downside_std = downside_variance ** 0.5
        if downside_std == 0:
            return 0.0
        return (avg_return - daily_rf) / downside_std * (252 ** 0.5)

    @staticmethod
    def compute_max_drawdown(equity_curve: List[float]) -> float:
        """计算最大回撤"""
        if not equity_curve:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def compute_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
        """计算卡尔玛比率"""
        if max_drawdown == 0:
            return 0.0
        return annual_return / max_drawdown

    @staticmethod
    def compute_win_rate(wins: int, total: int) -> float:
        """计算胜率"""
        return wins / total if total > 0 else 0.0

    @staticmethod
    def compute_profit_factor(gross_profit: float, gross_loss: float) -> Optional[float]:
        """计算盈亏比；无亏损时返回 None（JSON null），避免 Infinity 污染输出。"""
        return abs(gross_profit / gross_loss) if gross_loss != 0 else None

    @staticmethod
    def compute_expectancy(avg_win: float, avg_loss: float, win_rate: float) -> float:
        """计算期望值"""
        return win_rate * avg_win - (1 - win_rate) * abs(avg_loss)

    @staticmethod
    def compute_volatility(returns: List[float], annualize: bool = True) -> float:
        """计算波动率"""
        if not returns or len(returns) < 2:
            return 0.0
        avg = sum(returns) / len(returns)
        variance = sum((r - avg) ** 2 for r in returns) / (len(returns) - 1)
        daily_vol = variance ** 0.5
        return daily_vol * (252 ** 0.5) if annualize else daily_vol


# ═══════════════════════════════════════════════════════════════
# MetricHealthChecker
# ═══════════════════════════════════════════════════════════════

class MetricHealthChecker:
    """指标健康检查器"""

    def __init__(self):
        self._thresholds: Dict[str, Dict[str, float]] = {}
        self._anomalies: List[Dict[str, Any]] = []
        self._max_anomalies = 1000

    def set_threshold(self, metric_name: str, warning: float = None,
                     critical: float = None, min: float = None,
                     max: float = None):
        """设置指标阈值"""
        self._thresholds[metric_name] = {
            "warning": warning,
            "critical": critical,
            "min": min,
            "max": max,
        }

    def check(self, metrics: Dict[str, AggregatedMetric]) -> List[Dict[str, Any]]:
        """检查指标健康状态"""
        anomalies = []

        for name, agg in metrics.items():
            thresholds = self._thresholds.get(name, {})
            if not thresholds:
                continue

            value = agg.latest

            # 检查上下限
            if thresholds.get("min") is not None and value < thresholds["min"]:
                anomalies.append({
                    "metric": name,
                    "type": "below_min",
                    "value": value,
                    "threshold": thresholds["min"],
                    "severity": "critical",
                    "message": f"{name} = {value:.2f} < min({thresholds['min']:.2f})",
                    "timestamp": time.time(),
                })

            if thresholds.get("max") is not None and value > thresholds["max"]:
                anomalies.append({
                    "metric": name,
                    "type": "above_max",
                    "value": value,
                    "threshold": thresholds["max"],
                    "severity": "critical",
                    "message": f"{name} = {value:.2f} > max({thresholds['max']:.2f})",
                    "timestamp": time.time(),
                })

            # 检查告警级别
            if thresholds.get("critical") is not None and value >= thresholds["critical"]:
                anomalies.append({
                    "metric": name,
                    "type": "critical_threshold",
                    "value": value,
                    "threshold": thresholds["critical"],
                    "severity": "critical",
                    "message": f"{name} = {value:.2f} >= critical({thresholds['critical']:.2f})",
                    "timestamp": time.time(),
                })
            elif thresholds.get("warning") is not None and value >= thresholds["warning"]:
                anomalies.append({
                    "metric": name,
                    "type": "warning_threshold",
                    "value": value,
                    "threshold": thresholds["warning"],
                    "severity": "warning",
                    "message": f"{name} = {value:.2f} >= warning({thresholds['warning']:.2f})",
                    "timestamp": time.time(),
                })

        self._anomalies.extend(anomalies)
        if len(self._anomalies) > self._max_anomalies:
            self._anomalies = self._anomalies[-self._max_anomalies:]

        return anomalies

    def get_anomalies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取异常记录"""
        return self._anomalies[-limit:]

    def get_anomaly_summary(self) -> Dict[str, Any]:
        """获取异常摘要"""
        by_severity = defaultdict(int)
        by_metric = defaultdict(int)
        for a in self._anomalies[-100:]:
            by_severity[a["severity"]] += 1
            by_metric[a["metric"]] += 1

        return {
            "total_anomalies": len(self._anomalies),
            "recent_count": len(self._anomalies[-100:]),
            "by_severity": dict(by_severity),
            "by_metric": dict(by_metric),
        }


# ═══════════════════════════════════════════════════════════════
# MetricSnapshot
# ═══════════════════════════════════════════════════════════════

class MetricSnapshot:
    """指标快照管理"""

    def __init__(self, persist_dir: str = "data"):
        self._persist_dir = persist_dir
        self._snapshots: deque = deque(maxlen=100)
        self._snapshot_dir = os.path.join(persist_dir, "metric_snapshots")
        os.makedirs(self._snapshot_dir, exist_ok=True)

    def take_snapshot(self, metrics: Dict[str, AggregatedMetric],
                     metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建快照"""
        snapshot = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "metadata": metadata or {},
            "metrics": {
                name: agg.to_dict()
                for name, agg in metrics.items()
            },
        }
        self._snapshots.append(snapshot)
        return snapshot

    def save_snapshot(self, snapshot: Dict[str, Any]):
        """保存快照到文件"""
        try:
            filename = f"snapshot_{int(snapshot['timestamp'])}.json"
            filepath = os.path.join(self._snapshot_dir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save metric snapshot: {e}")

    def load_snapshot(self, timestamp: int) -> Optional[Dict[str, Any]]:
        """加载快照"""
        try:
            filename = f"snapshot_{timestamp}.json"
            filepath = os.path.join(self._snapshot_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load metric snapshot: {e}")
        return None

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """获取最新快照"""
        return self._snapshots[-1] if self._snapshots else None

    def get_snapshots(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取快照列表"""
        return list(self._snapshots)[-limit:]

    def cleanup_old_snapshots(self, max_age_days: int = 7):
        """清理旧快照"""
        try:
            cutoff = time.time() - max_age_days * 86400
            for filename in os.listdir(self._snapshot_dir):
                if filename.startswith("snapshot_") and filename.endswith(".json"):
                    filepath = os.path.join(self._snapshot_dir, filename)
                    if os.path.getmtime(filepath) < cutoff:
                        os.remove(filepath)
                        logger.debug(f"Removed old metric snapshot: {filename}")
        except Exception as e:
            logger.error(f"Failed to cleanup old snapshots: {e}")


# ═══════════════════════════════════════════════════════════════
# MetricsPipeline
# ═══════════════════════════════════════════════════════════════

class MetricsPipeline:
    """
    生产级性能指标流水线

    使用示例:
        pipeline = MetricsPipeline(config)

        # 注册自定义指标
        pipeline.register_metric(MetricDefinition(
            "my_custom_metric", "自定义指标", "units", MetricType.GAUGE))

        # 记录指标
        pipeline.record("trading_pnl_total", 150.5, {"strategy": "grid"})

        # 启动流水线
        await pipeline.start()

        # 获取聚合指标
        metrics = pipeline.get_aggregated_metrics()

        # 导出
        prometheus = pipeline.export("prometheus")
        json_data = pipeline.export("json")

        # 停止
        await pipeline.stop()
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        pipeline_cfg = config.get("metrics_pipeline", {})

        # ── 指标注册中心 ──
        self._registry = MetricRegistry()

        # ── 聚合引擎 ──
        window_sec = pipeline_cfg.get("aggregation_window_sec", 60.0)
        max_samples = pipeline_cfg.get("max_samples", 1000)
        self._aggregator = AggregationEngine(window_sec, max_samples)

        # ── 派生指标引擎 ──
        self._derived = DerivedMetricsEngine()

        # ── 指标导出器 ──
        self._exporter = MetricExporter(self._registry)

        # ── 健康检查器 ──
        self._health_checker = MetricHealthChecker()

        # 配置默认阈值
        self._configure_default_thresholds()

        # ── 快照管理 ──
        persist_dir = config.get("system", {}).get("data_dir", "data")
        self._snapshot = MetricSnapshot(persist_dir)

        # ── 外部采集器集成 ──
        self._metrics_collector = None
        if get_metrics_collector:
            try:
                self._metrics_collector = get_metrics_collector()
            except Exception:
                pass

        # ── 回调 ──
        self._record_callbacks: List[Callable] = []
        self._aggregate_callbacks: List[Callable] = []
        self._anomaly_callbacks: List[Callable] = []

        # ── 运行控制 ──
        self._running = False
        self._aggregate_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._aggregate_interval = pipeline_cfg.get("aggregate_interval_sec", 10)
        self._snapshot_interval = pipeline_cfg.get("snapshot_interval_sec", 300)
        self._health_interval = pipeline_cfg.get("health_check_interval_sec", 30)

        # ── 持久化 ──
        self._persist_dir = persist_dir
        self._metrics_log_file = os.path.join(self._persist_dir, "metrics_pipeline.jsonl")

        logger.info(
            f"MetricsPipeline initialized: "
            f"window={window_sec}s, "
            f"aggregate_interval={self._aggregate_interval}s, "
            f"snapshot_interval={self._snapshot_interval}s"
        )

    def _configure_default_thresholds(self):
        """配置默认健康阈值"""
        defaults = {
            "system_cpu_usage_pct": {"warning": 70, "critical": 90},
            "system_memory_usage_pct": {"warning": 80, "critical": 95},
            "system_disk_free_gb": {"min": 1.0},
            "risk_drawdown_pct": {"warning": 0.10, "critical": 0.20},
            "risk_margin_ratio": {"warning": 0.5, "critical": 0.7},
            "risk_score": {"warning": 0.5, "critical": 0.7},
            "capital_idle_pct": {"warning": 0.30, "critical": 0.50},
            "trading_win_rate": {"min": 0.30},
            "perf_signal_process_ms": {"warning": 100, "critical": 200},
            "perf_order_place_ms": {"warning": 200, "critical": 500},
            "perf_ws_latency_ms": {"warning": 1000, "critical": 3000},
        }
        for name, thresholds in defaults.items():
            self._health_checker.set_threshold(name, **thresholds)

    # ═══════════════════════════════════════════════════════════════
    # 指标注册
    # ═══════════════════════════════════════════════════════════════

    def register_metric(self, definition: MetricDefinition):
        """注册指标"""
        self._registry.register(definition)

    def unregister_metric(self, name: str):
        """注销指标"""
        self._registry.unregister(name)

    def get_metric_definition(self, name: str) -> Optional[MetricDefinition]:
        """获取指标定义"""
        return self._registry.get(name)

    def get_all_definitions(self) -> Dict[str, MetricDefinition]:
        """获取所有指标定义"""
        return self._registry.get_all()

    # ═══════════════════════════════════════════════════════════════
    # 指标记录
    # ═══════════════════════════════════════════════════════════════

    def record(self, name: str, value: float, labels: Dict[str, str] = None):
        """记录指标值"""
        sample = MetricSample(
            name=name,
            value=value,
            labels=labels or {},
        )

        # 添加到聚合器
        self._aggregator.add_sample(sample)

        # 同步到 MetricsCollector
        if self._metrics_collector:
            try:
                self._metrics_collector.record(name, value, labels)
            except Exception:
                pass

        # 通知回调
        for cb in self._record_callbacks:
            try:
                cb(sample)
            except Exception as e:
                logger.error(f"Record callback error: {e}")

    def record_batch(self, samples: List[MetricSample]):
        """批量记录指标"""
        for sample in samples:
            self.record(sample.name, sample.value, sample.labels)

    def increment(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """递增计数器"""
        agg = self._aggregator.aggregate(name)
        current = agg.latest if agg else 0
        self.record(name, current + value, labels)

    def record_latency(self, name: str, latency_ms: float, labels: Dict[str, str] = None):
        """记录延迟（自动使用histogram类型）"""
        self.record(name, latency_ms, labels)

    # ═══════════════════════════════════════════════════════════════
    # 指标查询
    # ═══════════════════════════════════════════════════════════════

    def get_aggregated_metrics(self, window_sec: Optional[float] = None) -> Dict[str, AggregatedMetric]:
        """获取聚合指标"""
        return self._aggregator.aggregate_all(window_sec)

    def get_metric(self, name: str, window_sec: Optional[float] = None) -> Optional[AggregatedMetric]:
        """获取单个指标"""
        return self._aggregator.aggregate(name, window_sec)

    def get_metrics_by_category(self, category: MetricCategory,
                               window_sec: Optional[float] = None) -> Dict[str, AggregatedMetric]:
        """按类别获取指标"""
        all_metrics = self._aggregator.aggregate_all(window_sec)
        definitions = self._registry.get_by_category(category)
        category_names = {d.name for d in definitions}
        return {k: v for k, v in all_metrics.items() if k in category_names}

    def get_metrics_by_labels(self, name: str, label_key: str,
                             window_sec: Optional[float] = None) -> Dict[str, AggregatedMetric]:
        """按标签获取指标"""
        return self._aggregator.aggregate_by_labels(name, label_key, window_sec)

    def get_latest_value(self, name: str) -> Optional[float]:
        """获取最新值"""
        agg = self._aggregator.aggregate(name)
        return agg.latest if agg else None

    def get_derived_metrics(self, daily_returns: List[float] = None,
                           equity_curve: List[float] = None,
                           wins: int = 0, total_trades: int = 0,
                           gross_profit: float = 0, gross_loss: float = 0) -> Dict[str, float]:
        """计算派生指标"""
        result = {}

        if daily_returns:
            result["sharpe_ratio"] = self._derived.compute_sharpe_ratio(daily_returns)
            result["sortino_ratio"] = self._derived.compute_sortino_ratio(daily_returns)
            result["volatility"] = self._derived.compute_volatility(daily_returns)

        if equity_curve:
            result["max_drawdown"] = self._derived.compute_max_drawdown(equity_curve)

        if total_trades > 0:
            result["win_rate"] = self._derived.compute_win_rate(wins, total_trades)

        if gross_loss != 0:
            result["profit_factor"] = self._derived.compute_profit_factor(gross_profit, gross_loss)

        return result

    # ═══════════════════════════════════════════════════════════════
    # 指标导出
    # ═══════════════════════════════════════════════════════════════

    def export(self, format: str = "prometheus", window_sec: Optional[float] = None) -> str:
        """导出指标"""
        metrics = self._aggregator.aggregate_all(window_sec)

        if format == "prometheus":
            return self._exporter.export_prometheus(metrics)
        elif format == "json":
            return self._exporter.export_json(metrics)
        elif format == "influxdb":
            return self._exporter.export_influxdb(metrics)
        elif format == "csv":
            return self._exporter.export_csv(metrics)
        else:
            raise ValueError(f"Unknown export format: {format}")

    def export_to_file(self, filepath: str, format: str = "json",
                      window_sec: Optional[float] = None):
        """导出到文件"""
        content = self.export(format, window_sec)
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"Metrics exported to {filepath} ({format})")

    # ═══════════════════════════════════════════════════════════════
    # 健康检查
    # ═══════════════════════════════════════════════════════════════

    def check_health(self) -> Dict[str, Any]:
        """检查指标健康状态"""
        metrics = self._aggregator.aggregate_all()
        anomalies = self._health_checker.check(metrics)

        # 通知异常回调
        for anomaly in anomalies:
            if anomaly["severity"] == "critical":
                for cb in self._anomaly_callbacks:
                    try:
                        cb(anomaly)
                    except Exception as e:
                        logger.error(f"Anomaly callback error: {e}")

        summary = self._health_checker.get_anomaly_summary()

        return {
            "healthy": len(anomalies) == 0,
            "anomalies": anomalies,
            "summary": summary,
            "timestamp": time.time(),
        }

    def get_anomalies(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取异常记录"""
        return self._health_checker.get_anomalies(limit)

    def set_threshold(self, metric_name: str, **thresholds):
        """设置指标阈值"""
        self._health_checker.set_threshold(metric_name, **thresholds)

    # ═══════════════════════════════════════════════════════════════
    # 快照管理
    # ═══════════════════════════════════════════════════════════════

    def take_snapshot(self, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """创建指标快照"""
        metrics = self._aggregator.aggregate_all()
        snapshot = self._snapshot.take_snapshot(metrics, metadata)
        self._snapshot.save_snapshot(snapshot)
        return snapshot

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """获取最新快照"""
        return self._snapshot.get_latest_snapshot()

    def get_snapshots(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取快照列表"""
        return self._snapshot.get_snapshots(limit)

    # ═══════════════════════════════════════════════════════════════
    # 回调注册
    # ═══════════════════════════════════════════════════════════════

    def on_record(self, callback: Callable):
        """注册记录回调"""
        self._record_callbacks.append(callback)

    def on_aggregate(self, callback: Callable):
        """注册聚合回调"""
        self._aggregate_callbacks.append(callback)

    def on_anomaly(self, callback: Callable):
        """注册异常回调"""
        self._anomaly_callbacks.append(callback)

    # ═══════════════════════════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════════════════════════

    async def start(self):
        """启动流水线"""
        self._running = True
        self._aggregate_task = asyncio.create_task(self._aggregate_loop())
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("MetricsPipeline started")

    async def stop(self):
        """停止流水线"""
        self._running = False

        for task in [self._aggregate_task, self._snapshot_task, self._health_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # 保存最终快照
        try:
            self.take_snapshot({"event": "pipeline_stop"})
        except Exception:
            pass

        logger.info("MetricsPipeline stopped")

    async def _aggregate_loop(self):
        """聚合循环"""
        while self._running:
            try:
                await asyncio.sleep(self._aggregate_interval)
                metrics = self._aggregator.aggregate_all()

                # 通知聚合回调
                for cb in self._aggregate_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(cb):
                            await cb(metrics)
                        else:
                            cb(metrics)
                    except Exception as e:
                        logger.error(f"Aggregate callback error: {e}")

                # 持久化
                self._save_metrics(metrics)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Aggregate loop error: {e}")

    async def _snapshot_loop(self):
        """快照循环"""
        while self._running:
            try:
                await asyncio.sleep(self._snapshot_interval)
                self.take_snapshot({"event": "periodic_snapshot"})
                self._snapshot.cleanup_old_snapshots()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot loop error: {e}")

    async def _health_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await asyncio.sleep(self._health_interval)
                health = self.check_health()
                if not health["healthy"]:
                    logger.warning(
                        f"Metrics health check: {len(health['anomalies'])} anomalies detected"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")

    def _save_metrics(self, metrics: Dict[str, AggregatedMetric]):
        """保存指标到文件"""
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            with open(self._metrics_log_file, "a", encoding="utf-8") as f:
                entry = {
                    "timestamp": time.time(),
                    "datetime": datetime.now().isoformat(),
                    "metrics": {
                        name: agg.to_dict()
                        for name, agg in list(metrics.items())[:50]
                    },
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 统计查询
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict[str, Any]:
        """获取流水线统计"""
        metrics = self._aggregator.aggregate_all()
        return {
            "registry_size": len(self._registry.get_all()),
            "aggregated_metrics": len(metrics),
            "total_samples": sum(self._aggregator.get_sample_count(name) for name in metrics),
            "snapshots_count": len(self._snapshot.get_snapshots()),
            "anomalies": self._health_checker.get_anomaly_summary(),
            "uptime": self._get_uptime_info(),
        }

    def _get_uptime_info(self) -> Dict[str, Any]:
        """获取运行时间信息"""
        agg = self._aggregator.aggregate("system_uptime_sec")
        uptime = agg.latest if agg else 0
        return {
            "seconds": uptime,
            "formatted": str(timedelta(seconds=int(uptime))),
        }

    def reset(self):
        """重置流水线"""
        self._aggregator.clear()
        self._health_checker._anomalies.clear()
        logger.info("MetricsPipeline reset")


# ═══════════════════════════════════════════════════════════════
# 便捷工厂函数
# ═══════════════════════════════════════════════════════════════

def create_pipeline_from_config(config: Dict[str, Any]) -> MetricsPipeline:
    """从配置创建指标流水线"""
    pipeline = MetricsPipeline(config)

    # 加载自定义指标定义
    pipeline_cfg = config.get("metrics_pipeline", {})
    custom_metrics = pipeline_cfg.get("custom_metrics", [])
    for m in custom_metrics:
        try:
            pipeline.register_metric(MetricDefinition(
                name=m["name"],
                description=m.get("description", ""),
                unit=m.get("unit", ""),
                type=MetricType(m.get("type", "gauge")),
                category=MetricCategory(m.get("category", "custom")),
                labels=m.get("labels", []),
                buckets=m.get("buckets", []),
                quantiles=m.get("quantiles", []),
                help_text=m.get("help_text", ""),
            ))
        except Exception as e:
            logger.warning(f"Failed to register custom metric: {m.get('name')} - {e}")

    # 加载自定义阈值
    custom_thresholds = pipeline_cfg.get("custom_thresholds", {})
    for name, thresholds in custom_thresholds.items():
        pipeline.set_threshold(name, **thresholds)

    return pipeline


# 全局实例
_pipeline_instance: Optional[MetricsPipeline] = None


def get_metrics_pipeline(config: Dict[str, Any] = None) -> MetricsPipeline:
    """获取全局指标流水线"""
    global _pipeline_instance
    if _pipeline_instance is None and config:
        _pipeline_instance = create_pipeline_from_config(config)
    return _pipeline_instance


__all__ = [
    "MetricsPipeline",
    "MetricRegistry",
    "AggregationEngine",
    "MetricExporter",
    "DerivedMetricsEngine",
    "MetricHealthChecker",
    "MetricSnapshot",
    "MetricType",
    "MetricCategory",
    "MetricDefinition",
    "MetricSample",
    "AggregatedMetric",
    "create_pipeline_from_config",
    "get_metrics_pipeline",
]