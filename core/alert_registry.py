"""
集中式告警规则注册中心
Centralized Alert Rule Registry

覆盖四大类别：风控(RISK) · 交易(TRADING) · 系统(SYSTEM) · 性能(PERFORMANCE)
提供规则注册、指标评估、实时触发、建议动作等功能。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


# ============================================================
# 枚举定义
# ============================================================

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

    @property
    def weight(self) -> int:
        return {self.INFO: 1, self.WARNING: 2, self.CRITICAL: 3, self.EMERGENCY: 4}[self]

    @property
    def label_cn(self) -> str:
        return {self.INFO: "信息", self.WARNING: "警告", self.CRITICAL: "严重", self.EMERGENCY: "紧急"}[self]


class AlertCategory(Enum):
    RISK = "risk"
    TRADING = "trading"
    SYSTEM = "system"
    PERFORMANCE = "performance"

    @property
    def label_cn(self) -> str:
        return {self.RISK: "风控", self.TRADING: "交易", self.SYSTEM: "系统", self.PERFORMANCE: "性能"}[self]

    @property
    def icon(self) -> str:
        return {self.RISK: "🛡", self.TRADING: "📊", self.SYSTEM: "⚙", self.PERFORMANCE: "⚡"}[self]


class AlertState(Enum):
    NORMAL = "normal"
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


# 告警升级顺序：同一规则在升级窗口内反复触发时，严重级别按此顺序逐级上调
_ESCALATION_ORDER = (
    AlertSeverity.INFO,
    AlertSeverity.WARNING,
    AlertSeverity.CRITICAL,
    AlertSeverity.EMERGENCY,
)


# ============================================================
# 告警规则数据类
# ============================================================

@dataclass
class AlertRule:
    """告警规则定义"""
    name: str                          # 规则唯一标识
    category: AlertCategory            # 告警类别
    severity: AlertSeverity            # 严重级别
    metric: str                        # 关键指标名
    threshold: float                   # 阈值
    comparison: str                    # 比较运算符: gt/lt/gte/lte/eq
    description: str                   # 规则描述
    cooldown_seconds: int = 300        # 告警冷却时间(秒)
    auto_recover: bool = False         # 是否自动恢复
    actions: List[str] = field(default_factory=list)  # 建议动作
    metric_aliases: List[str] = field(default_factory=list)  # 指标别名(前端→后端映射)
    suggest_actions: List[str] = field(default_factory=list)  # 建议操作说明


@dataclass
class TriggeredAlert:
    """已触发的告警实例"""
    rule_name: str
    category: str
    severity: str
    metric: str
    value: float
    threshold: float
    comparison: str
    description: str
    actions: List[str]
    suggest_actions: List[str]
    trigger_time: str = ""
    state: str = "triggered"
    escalated: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "category": self.category,
            "severity": self.severity,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "comparison": self.comparison,
            "description": self.description,
            "actions": self.actions,
            "suggest_actions": self.suggest_actions,
            "trigger_time": self.trigger_time,
            "state": self.state,
            "escalated": self.escalated,
        }


# ============================================================
# 集中式告警规则注册中心
# ============================================================

class AlertRegistry:
    """集中式告警规则注册中心

    单例模式，管理所有告警规则的注册、查询和评估。
    """

    _instance: Optional["AlertRegistry"] = None

    def __new__(cls) -> "AlertRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._rules: Dict[str, AlertRule] = {}          # name → rule
        self._rules_by_category: Dict[AlertCategory, List[AlertRule]] = {c: [] for c in AlertCategory}
        self._metric_index: Dict[str, List[AlertRule]] = {}  # metric → [rules]
        self._triggered_alerts: List[TriggeredAlert] = []
        self._trigger_history: List[TriggeredAlert] = []     # 历史触发记录
        self._last_evaluate_time: Optional[datetime] = None
        self._cooldowns: Dict[str, float] = {}               # rule_name → cooldown_end_time

        # 告警升级策略：同一规则在升级窗口内反复触发（跨冷却周期）达阈值即升级一级
        self._escalation_window: float = 3600.0              # 升级统计窗口（秒）
        self._escalation_threshold: int = 3                  # 窗口内触发次数阈值
        self._trigger_times: Dict[str, List[float]] = {}     # rule_name → [触发时间戳]

        self._register_all_rules()

    # ============================================================
    # 规则注册
    # ============================================================

    def _register_all_rules(self):
        """注册所有告警规则"""
        self._register_risk_rules()
        self._register_trading_rules()
        self._register_system_rules()
        self._register_performance_rules()
        logger.info(f"AlertRegistry: 已注册 {len(self._rules)} 条告警规则 "
                     f"({len(self._rules_by_category[AlertCategory.RISK])} 风控, "
                     f"{len(self._rules_by_category[AlertCategory.TRADING])} 交易, "
                     f"{len(self._rules_by_category[AlertCategory.SYSTEM])} 系统, "
                     f"{len(self._rules_by_category[AlertCategory.PERFORMANCE])} 性能)")

    def _register_rule(self, rule: AlertRule):
        """注册单条规则"""
        self._rules[rule.name] = rule
        self._rules_by_category[rule.category].append(rule)

        # 建立指标索引
        for alias in [rule.metric] + rule.metric_aliases:
            if alias not in self._metric_index:
                self._metric_index[alias] = []
            self._metric_index[alias].append(rule)

    # ============================================================
    # 风控类规则 (RISK)
    # ============================================================

    def _register_risk_rules(self):
        rules = [
            AlertRule(
                name="max_drawdown_breach",
                category=AlertCategory.RISK,
                severity=AlertSeverity.EMERGENCY,
                metric="drawdown_pct",
                threshold=0.25,
                comparison="gte",
                description="最大回撤超过 25%，触发紧急熔断",
                cooldown_seconds=60,
                auto_recover=True,
                actions=["pause_trading", "close_all_positions", "notify_admin"],
                metric_aliases=["account.drawdown_pct", "risk.drawdown_pct"],
                suggest_actions=["立即暂停所有交易策略", "平仓所有持仓头寸", "通知管理员排查原因"],
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
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["account.daily_pnl_pct", "risk.daily_loss_pct"],
                suggest_actions=["暂停新开仓操作", "检查当前持仓风险敞口", "通知风控管理员"],
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
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["risk.hourly_loss_pct"],
                suggest_actions=["暂停交易1小时", "复盘近期交易记录", "检查策略是否异常"],
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
                auto_recover=True,
                actions=["reduce_position_size", "notify_admin"],
                metric_aliases=["risk.consecutive_losses", "trading.consecutive_losses"],
                suggest_actions=["降低单笔仓位规模至50%", "暂停高风险策略", "等待市场信号明确后再入场"],
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
                auto_recover=True,
                actions=["notify_admin", "reduce_positions"],
                metric_aliases=["account.margin_rate", "risk.margin_rate"],
                suggest_actions=["立即补充保证金", "减仓降低保证金占用", "优先平仓亏损头寸"],
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
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["risk.position_loss_pct"],
                suggest_actions=["评估该仓位是否止损", "检查市场是否出现黑天鹅", "考虑对冲操作"],
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
                auto_recover=False,
                actions=["notify_admin"],
                metric_aliases=["risk.circuit_breaker_active", "system.circuit_breaker"],
                suggest_actions=["检查熔断原因（API错误率/延迟/资金不足）", "等待熔断冷却后手动恢复", "评估是否需要调整熔断阈值"],
            ),
            AlertRule(
                name="risk_paused",
                category=AlertCategory.RISK,
                severity=AlertSeverity.WARNING,
                metric="risk_paused",
                threshold=1,
                comparison="eq",
                description="风控已暂停交易",
                cooldown_seconds=300,
                auto_recover=False,
                actions=["notify_admin"],
                metric_aliases=["risk.is_paused"],
                suggest_actions=["检查风控暂停原因", "确认后可手动恢复交易", "复盘近期风险事件"],
            ),
            AlertRule(
                name="high_leverage_warning",
                category=AlertCategory.RISK,
                severity=AlertSeverity.WARNING,
                metric="leverage",
                threshold=10,
                comparison="gte",
                description="杠杆倍数过高（≥10x），爆仓风险增大",
                cooldown_seconds=600,
                auto_recover=True,
                actions=["reduce_position_size"],
                metric_aliases=["account.leverage", "risk.leverage"],
                suggest_actions=["降低杠杆倍数至5x以下", "缩小仓位规模", "设置更紧的止损"],
            ),
        ]
        for r in rules:
            self._register_rule(r)

    # ============================================================
    # 交易类规则 (TRADING)
    # ============================================================

    def _register_trading_rules(self):
        rules = [
            AlertRule(
                name="order_rejection_rate_high",
                category=AlertCategory.TRADING,
                severity=AlertSeverity.WARNING,
                metric="order_rejection_rate",
                threshold=0.30,
                comparison="gte",
                description="订单拒绝率超过 30%",
                cooldown_seconds=600,
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.order_rejection_rate"],
                suggest_actions=["检查API密钥权限", "验证订单参数是否正确", "降低下单频率"],
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
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.avg_slippage"],
                suggest_actions=["使用限价单替代市价单", "降低单笔订单规模", "避开高波动时段交易"],
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
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.order_queue_size"],
                suggest_actions=["暂停新订单提交", "清理已完成/已取消的订单", "检查是否有死循环下单"],
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
                auto_recover=True,
                actions=[],
                metric_aliases=["trading.hours_since_last_trade"],
                suggest_actions=["检查策略是否正常运行", "确认市场是否有交易机会", "检查API连接状态"],
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
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.api_error_rate", "system.api_error_rate"],
                suggest_actions=["检查网络连接稳定性", "确认OKX API服务状态", "降低API请求频率"],
            ),
            AlertRule(
                name="high_funding_rate",
                category=AlertCategory.TRADING,
                severity=AlertSeverity.WARNING,
                metric="funding_rate",
                threshold=0.001,
                comparison="gte",
                description="资金费率过高（≥0.1%），持仓成本增加",
                cooldown_seconds=600,
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.funding_rate"],
                suggest_actions=["考虑减少持仓时间", "评估资金费率套利机会", "监控费率变化趋势"],
            ),
            AlertRule(
                name="position_imbalance",
                category=AlertCategory.TRADING,
                severity=AlertSeverity.WARNING,
                metric="long_short_ratio",
                threshold=3,
                comparison="gte",
                description="多空持仓比例失衡（≥3:1）",
                cooldown_seconds=600,
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.long_short_ratio"],
                suggest_actions=["评估市场方向性风险", "考虑对冲操作", "调整多空仓位配比"],
            ),
            AlertRule(
                name="negative_expectancy",
                category=AlertCategory.TRADING,
                severity=AlertSeverity.CRITICAL,
                metric="expectancy_usdt",
                threshold=0,
                comparison="lt",
                description="策略期望值为负（平均每笔净亏损），负期望策略在空转",
                cooldown_seconds=1800,
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.expectancy_usdt", "strategy.expectancy_usdt"],
                suggest_actions=["评估策略是否长期负期望", "暂停该策略开新仓", "重新回测或下调信号质量阈值"],
            ),
            AlertRule(
                name="low_profit_factor",
                category=AlertCategory.TRADING,
                severity=AlertSeverity.WARNING,
                metric="profit_factor",
                threshold=1.0,
                comparison="lt",
                description="盈亏比（利润因子）< 1，风报比不佳",
                cooldown_seconds=1800,
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.profit_factor", "strategy.profit_factor"],
                suggest_actions=["检查止损/止盈配比是否合理", "收紧止损或放宽止盈", "评估策略盈亏结构"],
            ),
            AlertRule(
                name="low_win_rate",
                category=AlertCategory.TRADING,
                severity=AlertSeverity.WARNING,
                metric="win_rate",
                threshold=0.35,
                comparison="lt",
                description="整体胜率低于 35%（样本充足时）",
                cooldown_seconds=1800,
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["trading.win_rate", "strategy.win_rate"],
                suggest_actions=["复盘近期亏损原因", "检查入场信号质量", "考虑降低交易频率"],
            ),
        ]
        for r in rules:
            self._register_rule(r)

    # ============================================================
    # 系统类规则 (SYSTEM)
    # ============================================================

    def _register_system_rules(self):
        rules = [
            AlertRule(
                name="high_memory_usage",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.WARNING,
                metric="memory_usage_pct",
                threshold=80,
                comparison="gte",
                description="内存使用率超过 80%",
                cooldown_seconds=600,
                auto_recover=True,
                actions=[],
                metric_aliases=["system.memory_usage_pct"],
                suggest_actions=["检查内存泄漏", "清理不必要的缓存", "考虑增加系统内存"],
            ),
            AlertRule(
                name="critical_memory_usage",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.CRITICAL,
                metric="memory_usage_pct",
                threshold=90,
                comparison="gte",
                description="内存使用率超过 90%，系统濒临崩溃",
                cooldown_seconds=300,
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["system.memory_usage_pct"],
                suggest_actions=["立即释放缓存", "停止非核心服务", "准备重启系统"],
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
                auto_recover=True,
                actions=[],
                metric_aliases=["system.cpu_usage_pct"],
                suggest_actions=["检查是否有死循环策略", "降低策略计算频率", "优化数据处理逻辑"],
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
                auto_recover=True,
                actions=["notify_admin", "cleanup_old_logs"],
                metric_aliases=["system.disk_free_gb"],
                suggest_actions=["清理过期日志文件", "删除旧的数据备份", "压缩历史交易数据"],
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
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["system.redis_connected"],
                suggest_actions=["检查Redis服务状态", "验证网络连接", "重启Redis服务"],
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
                auto_recover=True,
                actions=["notify_admin", "pause_trading"],
                metric_aliases=["system.db_connected"],
                suggest_actions=["检查数据库服务状态", "验证数据库连接配置", "暂停交易等待恢复"],
            ),
            AlertRule(
                name="trading_process_down",
                category=AlertCategory.SYSTEM,
                severity=AlertSeverity.EMERGENCY,
                metric="trading_process",
                threshold=0,
                comparison="eq",
                description="交易进程已停止运行",
                cooldown_seconds=30,
                auto_recover=False,
                actions=["notify_admin"],
                metric_aliases=["system.trading_process"],
                suggest_actions=["检查watchdog守护进程", "查看系统日志排查崩溃原因", "手动重启交易进程"],
            ),
        ]
        for r in rules:
            self._register_rule(r)

    # ============================================================
    # 性能类规则 (PERFORMANCE)
    # ============================================================

    def _register_performance_rules(self):
        rules = [
            AlertRule(
                name="high_api_latency",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric="api_latency_ms",
                threshold=1000,
                comparison="gte",
                description="API 平均延迟超过 1000ms",
                cooldown_seconds=300,
                auto_recover=True,
                actions=[],
                metric_aliases=["system.api_latency_ms", "performance.api_latency_ms"],
                suggest_actions=["检查网络延迟", "考虑切换API节点", "降低并发请求数"],
            ),
            AlertRule(
                name="critical_api_latency",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.CRITICAL,
                metric="api_latency_ms",
                threshold=3000,
                comparison="gte",
                description="API 平均延迟超过 3000ms，严重影响交易",
                cooldown_seconds=120,
                auto_recover=True,
                actions=["notify_admin"],
                metric_aliases=["system.api_latency_ms", "performance.api_latency_ms"],
                suggest_actions=["立即切换API节点", "暂停高频交易策略", "检查OKX服务状态页面"],
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
                auto_recover=True,
                actions=["notify_admin", "reconnect_ws"],
                metric_aliases=["system.ws_connected", "performance.ws_connected"],
                suggest_actions=["触发WebSocket重连机制", "检查网络防火墙设置", "切换到REST API轮询模式备用"],
            ),
            AlertRule(
                name="signal_processing_slow",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric="signal_process_ms",
                threshold=100,
                comparison="gte",
                description="信号处理延迟超过 100ms，可能错过交易机会",
                cooldown_seconds=600,
                auto_recover=True,
                actions=[],
                metric_aliases=["performance.signal_process_ms"],
                suggest_actions=["优化信号处理逻辑", "减少不必要的计算", "检查数据库查询性能"],
            ),
            AlertRule(
                name="high_correlation_count",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.INFO,
                metric="high_correlation_count",
                threshold=3,
                comparison="gte",
                description="高相关性交易对数量过多（≥3），风险集中",
                cooldown_seconds=600,
                auto_recover=True,
                actions=[],
                metric_aliases=["correlation.high_count"],
                suggest_actions=["分散交易对选择", "降低高相关性品种的仓位", "检查市场是否出现系统性风险"],
            ),
            AlertRule(
                name="signal_execution_rate_low",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric="signal_execution_rate",
                threshold=0.3,
                comparison="lte",
                description="信号执行率过低（≤30%），策略效率下降",
                cooldown_seconds=600,
                auto_recover=True,
                actions=[],
                metric_aliases=["signals.executed_count", "performance.signal_execution_rate"],
                suggest_actions=["检查信号过滤条件是否过严", "优化订单执行逻辑", "排查滑点/手续费影响"],
            ),
            AlertRule(
                name="kelly_fraction_high",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.WARNING,
                metric="kelly_fraction",
                threshold=0.5,
                comparison="gte",
                description="凯利分数过高（≥0.5），仓位风险过大",
                cooldown_seconds=600,
                auto_recover=True,
                actions=["reduce_position_size"],
                metric_aliases=["adaptive.kelly_fraction", "performance.kelly_fraction"],
                suggest_actions=["降低凯利仓位系数", "使用半凯利或四分之一凯利", "重新评估胜率和赔率"],
            ),
            AlertRule(
                name="total_factor_low",
                category=AlertCategory.PERFORMANCE,
                severity=AlertSeverity.INFO,
                metric="total_factor",
                threshold=0.3,
                comparison="lte",
                description="自适应因子过低（≤0.3），市场环境不利",
                cooldown_seconds=600,
                auto_recover=True,
                actions=[],
                metric_aliases=["adaptive.total_factor", "performance.total_factor"],
                suggest_actions=["降低交易频率", "收紧止损幅度", "等待市场环境改善"],
            ),
        ]
        for r in rules:
            self._register_rule(r)

    # ============================================================
    # 指标评估
    # ============================================================

    def evaluate(self, metrics: Dict[str, float]) -> List[TriggeredAlert]:
        """基于当前指标实时评估所有规则

        Args:
            metrics: 指标名→值的字典，支持别名匹配

        Returns:
            触发的告警列表
        """
        triggered: List[TriggeredAlert] = []
        now = time.time()

        for metric_key, value in metrics.items():
            if not isinstance(value, (int, float)):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    continue

            # 查找匹配的规则
            matching_rules = self._metric_index.get(metric_key, [])
            for rule in matching_rules:
                # 检查冷却期
                if self._is_on_cooldown(rule.name, now):
                    continue

                # 执行比较
                if self._compare(value, rule.threshold, rule.comparison):
                    severity, escalated = self._record_trigger_and_escalate(rule, now)
                    alert = TriggeredAlert(
                        rule_name=rule.name,
                        category=rule.category.value,
                        severity=severity.value,
                        metric=rule.metric,
                        value=value,
                        threshold=rule.threshold,
                        comparison=rule.comparison,
                        description=rule.description,
                        actions=rule.actions,
                        suggest_actions=rule.suggest_actions,
                        trigger_time=datetime.now().isoformat(),
                        state="triggered",
                        escalated=escalated,
                    )
                    triggered.append(alert)
                    self._set_cooldown(rule.name, now)

        # 按严重度排序
        triggered.sort(key=lambda a: AlertSeverity(a.severity).weight, reverse=True)

        # 更新状态
        self._triggered_alerts = [a for a in triggered if AlertSeverity(a.severity) in
                                   (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY)]
        self._trigger_history.extend(triggered)
        if len(self._trigger_history) > 500:
            self._trigger_history = self._trigger_history[-500:]

        self._last_evaluate_time = datetime.now()

        if triggered:
            logger.info(f"AlertRegistry: 评估完成，触发 {len(triggered)} 条告警 "
                        f"(EMERGENCY: {sum(1 for a in triggered if a.severity == 'emergency')}, "
                        f"CRITICAL: {sum(1 for a in triggered if a.severity == 'critical')}, "
                        f"WARNING: {sum(1 for a in triggered if a.severity == 'warning')})")

        return triggered

    def _compare(self, value: float, threshold: float, op: str) -> bool:
        ops: Dict[str, Callable[[float, float], bool]] = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
        }
        return ops.get(op, lambda v, t: False)(value, threshold)

    def _is_on_cooldown(self, rule_name: str, now: float) -> bool:
        return now < self._cooldowns.get(rule_name, 0)

    def _set_cooldown(self, rule_name: str, now: float):
        rule = self._rules.get(rule_name)
        if rule:
            self._cooldowns[rule_name] = now + rule.cooldown_seconds

    # ============================================================
    # 告警升级策略（去重 + 升级）
    # ============================================================

    def configure_escalation(self, window_seconds: float = 3600.0, threshold: int = 3):
        """配置告警升级策略。

        Args:
            window_seconds: 升级统计窗口（秒）。窗口内累计触发次数达到 threshold
                即触发一次级别上调。
            threshold: 升级触发阈值（窗口内触发次数）。
        """
        self._escalation_window = float(window_seconds)
        self._escalation_threshold = int(threshold)

    def _record_trigger_and_escalate(self, rule: AlertRule, now: float) -> Tuple[AlertSeverity, bool]:
        """记录一次触发并计算升级后的严重级别。

        去重由 evaluate() 的冷却期保证（同一规则在冷却期内不重复触发）；
        此处负责跨冷却周期的升级：同一规则在升级窗口内反复触发达到阈值时，
        严重级别上调一级（上限 EMERGENCY）。

        Returns:
            (最终严重级别, 是否发生升级)
        """
        times = self._trigger_times.setdefault(rule.name, [])
        times.append(now)
        # 修剪窗口外的旧触发记录，仅保留升级窗口内的样本
        times[:] = [t for t in times if now - t <= self._escalation_window]

        base = rule.severity
        if len(times) >= self._escalation_threshold:
            idx = _ESCALATION_ORDER.index(base)
            escalated = _ESCALATION_ORDER[min(idx + 1, len(_ESCALATION_ORDER) - 1)]
            return escalated, escalated != base
        return base, False

    # ============================================================
    # 查询接口
    # ============================================================

    def get_all_rules(self) -> List[Dict[str, Any]]:
        """获取所有注册的规则"""
        result = []
        for rule in self._rules.values():
            result.append({
                "name": rule.name,
                "category": rule.category.value,
                "category_label": rule.category.label_cn,
                "category_icon": rule.category.icon,
                "severity": rule.severity.value,
                "severity_label": rule.severity.label_cn,
                "severity_weight": rule.severity.weight,
                "metric": rule.metric,
                "metric_aliases": rule.metric_aliases,
                "threshold": rule.threshold,
                "comparison": rule.comparison,
                "description": rule.description,
                "cooldown_seconds": rule.cooldown_seconds,
                "auto_recover": rule.auto_recover,
                "actions": rule.actions,
                "suggest_actions": rule.suggest_actions,
            })
        return result

    def get_rules_by_category(self, category: str) -> List[Dict[str, Any]]:
        """按类别获取规则"""
        cat = AlertCategory(category) if category in [c.value for c in AlertCategory] else None
        if cat is None:
            return []
        return [self._rule_to_dict(r) for r in self._rules_by_category.get(cat, [])]

    def get_rule(self, name: str) -> Optional[Dict[str, Any]]:
        """获取单条规则"""
        rule = self._rules.get(name)
        return self._rule_to_dict(rule) if rule else None

    def _rule_to_dict(self, rule: AlertRule) -> Dict[str, Any]:
        return {
            "name": rule.name,
            "category": rule.category.value,
            "category_label": rule.category.label_cn,
            "category_icon": rule.category.icon,
            "severity": rule.severity.value,
            "severity_label": rule.severity.label_cn,
            "severity_weight": rule.severity.weight,
            "metric": rule.metric,
            "metric_aliases": rule.metric_aliases,
            "threshold": rule.threshold,
            "comparison": rule.comparison,
            "description": rule.description,
            "cooldown_seconds": rule.cooldown_seconds,
            "auto_recover": rule.auto_recover,
            "actions": rule.actions,
            "suggest_actions": rule.suggest_actions,
        }

    def get_summary(self) -> Dict[str, Any]:
        """获取规则摘要统计"""
        by_cat = {}
        for cat in AlertCategory:
            rules = self._rules_by_category.get(cat, [])
            by_cat[cat.value] = {
                "count": len(rules),
                "label": cat.label_cn,
                "icon": cat.icon,
                "rules": [r.name for r in rules],
            }

        by_sev = {s.value: 0 for s in AlertSeverity}
        for rule in self._rules.values():
            by_sev[rule.severity.value] += 1

        return {
            "total_rules": len(self._rules),
            "by_category": {k: v["count"] for k, v in by_cat.items()},
            "by_category_detail": by_cat,
            "by_severity": by_sev,
            "emergency_rules": [r.name for r in self._rules.values()
                                if r.severity == AlertSeverity.EMERGENCY],
            "critical_rules": [r.name for r in self._rules.values()
                               if r.severity == AlertSeverity.CRITICAL],
            "escalation": {
                "window_seconds": self._escalation_window,
                "threshold": self._escalation_threshold,
            },
            "last_evaluate_time": self._last_evaluate_time.isoformat()
                if self._last_evaluate_time else None,
        }

    def get_triggered_alerts(self) -> List[Dict[str, Any]]:
        """获取当前触发的告警"""
        return [a.to_dict() for a in self._triggered_alerts]

    def get_trigger_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取告警触发历史"""
        return [a.to_dict() for a in self._trigger_history[-limit:]]

    def get_categories(self) -> List[Dict[str, Any]]:
        """获取所有类别"""
        return [{
            "value": c.value,
            "label": c.label_cn,
            "icon": c.icon,
            "rule_count": len(self._rules_by_category.get(c, [])),
        } for c in AlertCategory]

    def clear_cooldowns(self):
        """清除所有冷却状态"""
        self._cooldowns.clear()

    def reset(self):
        """重置注册中心状态"""
        self._triggered_alerts.clear()
        self._trigger_history.clear()
        self._cooldowns.clear()
        self._trigger_times.clear()
        self._last_evaluate_time = None


# ============================================================
# 全局单例
# ============================================================

_alert_registry: Optional[AlertRegistry] = None


def get_alert_registry() -> AlertRegistry:
    """获取告警注册中心全局单例"""
    global _alert_registry
    if _alert_registry is None:
        _alert_registry = AlertRegistry()
    return _alert_registry