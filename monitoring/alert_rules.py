"""DEPRECATED（模块 2 架构收敛）：告警规则定义已收敛到 core/alert_registry.py（权威实现，规则更全且含 metric_aliases/suggest_actions）。

本模块仅被 monitoring/alert_engine.py（同样待废弃）引用，禁止在此新增或修改规则。
"""
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertCategory(Enum):
    RISK = "risk"
    TRADING = "trading"
    SYSTEM = "system"
    PERFORMANCE = "performance"


@dataclass
class AlertRule:
    """告警规则定义"""
    name: str
    category: AlertCategory
    severity: AlertSeverity
    metric: str
    threshold: float
    comparison: str  # gt / lt / eq / gte / lte
    description: str
    cooldown_seconds: int = 300  # 告警冷却，避免重复告警
    auto_recover: bool = False  # 是否自动恢复
    actions: List[str] = field(default_factory=list)  # 触发后动作


# ============================================================
# 风控类告警规则
# ============================================================
RISK_ALERT_RULES: List[AlertRule] = [
    AlertRule(
        name="max_drawdown_breach",
        category=AlertCategory.RISK,
        severity=AlertSeverity.EMERGENCY,
        metric="drawdown_pct",
        threshold=0.25,
        comparison="gte",
        description="最大回撤超过 25%，触发紧急熔断",
        cooldown_seconds=60,
        actions=["pause_trading", "close_all_positions", "notify_admin"],
    ),
    AlertRule(
        name="daily_loss_limit",
        category=AlertCategory.RISK,
        severity=AlertSeverity.CRITICAL,
        metric="daily_loss_pct",
        threshold=0.10,
        comparison="gte",
        description="日内亏损超过 10%，暂停开新仓",
        cooldown_seconds=300,
        actions=["pause_new_entries", "notify_admin"],
    ),
    AlertRule(
        name="hourly_loss_limit",
        category=AlertCategory.RISK,
        severity=AlertSeverity.CRITICAL,
        metric="hourly_loss_pct",
        threshold=0.05,
        comparison="gte",
        description="小时亏损超过 5%，触发短期熔断",
        cooldown_seconds=1800,
        actions=["pause_trading_1h", "notify_admin"],
    ),
    AlertRule(
        name="consecutive_losses",
        category=AlertCategory.RISK,
        severity=AlertSeverity.WARNING,
        metric="consecutive_losses",
        threshold=5,
        comparison="gte",
        description="连续亏损超过 5 笔，建议暂停策略",
        cooldown_seconds=600,
        actions=["reduce_position_size", "notify_admin"],
    ),
    AlertRule(
        name="margin_call_warning",
        category=AlertCategory.RISK,
        severity=AlertSeverity.CRITICAL,
        metric="margin_rate",
        threshold=0.5,
        comparison="lte",
        description="保证金率低于 50%，强平风险",
        cooldown_seconds=120,
        actions=["notify_admin", "reduce_positions"],
    ),
    AlertRule(
        name="position_loss_warning",
        category=AlertCategory.RISK,
        severity=AlertSeverity.WARNING,
        metric="position_loss_pct",
        threshold=0.5,
        comparison="gte",
        description="单仓位亏损超过 50%",
        cooldown_seconds=300,
        actions=["notify_admin"],
    ),
    AlertRule(
        name="circuit_breaker_triggered",
        category=AlertCategory.RISK,
        severity=AlertSeverity.EMERGENCY,
        metric="circuit_breaker_active",
        threshold=1,
        comparison="eq",
        description="熔断器触发，交易已暂停",
        cooldown_seconds=60,
        actions=["notify_admin"],
    ),
]

# ============================================================
# 交易类告警规则
# ============================================================
TRADING_ALERT_RULES: List[AlertRule] = [
    AlertRule(
        name="order_rejection_rate_high",
        category=AlertCategory.TRADING,
        severity=AlertSeverity.WARNING,
        metric="order_rejection_rate",
        threshold=0.30,
        comparison="gte",
        description="订单拒绝率超过 30%",
        cooldown_seconds=600,
        actions=["notify_admin"],
    ),
    AlertRule(
        name="high_slippage",
        category=AlertCategory.TRADING,
        severity=AlertSeverity.WARNING,
        metric="avg_slippage",
        threshold=0.005,
        comparison="gte",
        description="平均滑点超过 0.5%",
        cooldown_seconds=600,
        actions=["notify_admin"],
    ),
    AlertRule(
        name="order_queue_full",
        category=AlertCategory.TRADING,
        severity=AlertSeverity.CRITICAL,
        metric="order_queue_size",
        threshold=900,
        comparison="gte",
        description="订单队列接近上限（900/1000）",
        cooldown_seconds=60,
        actions=["notify_admin"],
    ),
    AlertRule(
        name="no_trades_long_time",
        category=AlertCategory.TRADING,
        severity=AlertSeverity.INFO,
        metric="hours_since_last_trade",
        threshold=6,
        comparison="gte",
        description="超过 6 小时无成交",
        cooldown_seconds=3600,
    ),
    AlertRule(
        name="api_error_rate_high",
        category=AlertCategory.TRADING,
        severity=AlertSeverity.CRITICAL,
        metric="api_error_rate",
        threshold=0.10,
        comparison="gte",
        description="API 错误率超过 10%",
        cooldown_seconds=300,
        actions=["notify_admin"],
    ),
]

# ============================================================
# 系统类告警规则
# ============================================================
SYSTEM_ALERT_RULES: List[AlertRule] = [
    AlertRule(
        name="high_memory_usage",
        category=AlertCategory.SYSTEM,
        severity=AlertSeverity.WARNING,
        metric="memory_usage_pct",
        threshold=80,
        comparison="gte",
        description="内存使用率超过 80%",
        cooldown_seconds=600,
    ),
    AlertRule(
        name="critical_memory_usage",
        category=AlertCategory.SYSTEM,
        severity=AlertSeverity.CRITICAL,
        metric="memory_usage_pct",
        threshold=90,
        comparison="gte",
        description="内存使用率超过 90%",
        cooldown_seconds=300,
        actions=["notify_admin"],
    ),
    AlertRule(
        name="high_cpu_usage",
        category=AlertCategory.SYSTEM,
        severity=AlertSeverity.WARNING,
        metric="cpu_usage_pct",
        threshold=80,
        comparison="gte",
        description="CPU 使用率超过 80%",
        cooldown_seconds=600,
    ),
    AlertRule(
        name="disk_space_low",
        category=AlertCategory.SYSTEM,
        severity=AlertSeverity.CRITICAL,
        metric="disk_free_gb",
        threshold=1,
        comparison="lte",
        description="磁盘可用空间低于 1GB",
        cooldown_seconds=1800,
        actions=["notify_admin", "cleanup_old_logs"],
    ),
    AlertRule(
        name="redis_connection_lost",
        category=AlertCategory.SYSTEM,
        severity=AlertSeverity.CRITICAL,
        metric="redis_connected",
        threshold=0,
        comparison="eq",
        description="Redis 连接断开，已降级为内存缓存",
        cooldown_seconds=60,
        actions=["notify_admin"],
    ),
    AlertRule(
        name="db_connection_lost",
        category=AlertCategory.SYSTEM,
        severity=AlertSeverity.EMERGENCY,
        metric="db_connected",
        threshold=0,
        comparison="eq",
        description="数据库连接断开",
        cooldown_seconds=30,
        actions=["notify_admin", "pause_trading"],
    ),
]

# ============================================================
# 性能类告警规则
# ============================================================
PERFORMANCE_ALERT_RULES: List[AlertRule] = [
    AlertRule(
        name="high_api_latency",
        category=AlertCategory.PERFORMANCE,
        severity=AlertSeverity.WARNING,
        metric="api_latency_ms",
        threshold=1000,
        comparison="gte",
        description="API 平均延迟超过 1000ms",
        cooldown_seconds=300,
    ),
    AlertRule(
        name="critical_api_latency",
        category=AlertCategory.PERFORMANCE,
        severity=AlertSeverity.CRITICAL,
        metric="api_latency_ms",
        threshold=3000,
        comparison="gte",
        description="API 平均延迟超过 3000ms",
        cooldown_seconds=120,
        actions=["notify_admin"],
    ),
    AlertRule(
        name="websocket_disconnect",
        category=AlertCategory.PERFORMANCE,
        severity=AlertSeverity.CRITICAL,
        metric="ws_connected",
        threshold=0,
        comparison="eq",
        description="WebSocket 断开，实时数据中断",
        cooldown_seconds=60,
        actions=["notify_admin", "reconnect_ws"],
    ),
    AlertRule(
        name="signal_processing_slow",
        category=AlertCategory.PERFORMANCE,
        severity=AlertSeverity.WARNING,
        metric="signal_process_ms",
        threshold=100,
        comparison="gte",
        description="信号处理延迟超过 100ms",
        cooldown_seconds=600,
    ),
]

ALL_RULES: List[AlertRule] = (
    RISK_ALERT_RULES + TRADING_ALERT_RULES +
    SYSTEM_ALERT_RULES + PERFORMANCE_ALERT_RULES
)


def get_rules_by_category(category: AlertCategory) -> List[AlertRule]:
    """按类别获取告警规则"""
    return [r for r in ALL_RULES if r.category == category]


def get_rules_by_severity(severity: AlertSeverity) -> List[AlertRule]:
    """按严重级别获取告警规则"""
    return [r for r in ALL_RULES if r.severity == severity]


def evaluate_metric(metric: str, value: float) -> List[AlertRule]:
    """评估指标值，返回触发的告警规则"""
    triggered = []
    for rule in ALL_RULES:
        if rule.metric != metric:
            continue
        if _compare(value, rule.threshold, rule.comparison):
            triggered.append(rule)
    return triggered


def _compare(value: float, threshold: float, op: str) -> bool:
    ops = {
        "gt": lambda v, t: v > t,
        "lt": lambda v, t: v < t,
        "gte": lambda v, t: v >= t,
        "lte": lambda v, t: v <= t,
        "eq": lambda v, t: v == t,
    }
    return ops.get(op, lambda v, t: False)(value, threshold)


def get_alert_summary() -> Dict[str, Any]:
    """获取告警规则摘要"""
    return {
        "total_rules": len(ALL_RULES),
        "by_category": {
            cat.value: len(get_rules_by_category(cat))
            for cat in AlertCategory
        },
        "by_severity": {
            sev.value: len(get_rules_by_severity(sev))
            for sev in AlertSeverity
        },
        "emergency_rules": [r.name for r in get_rules_by_severity(AlertSeverity.EMERGENCY)],
    }
