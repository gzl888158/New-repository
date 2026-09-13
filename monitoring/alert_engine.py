"""DEPRECATED（模块 2 架构收敛）：告警规则引擎已收敛到 core/alert_registry.py（AlertRegistry.evaluate）与 core/alert_evaluator.py（notify_triggered_alerts / run_alert_evaluator）。

本模块仅被 core/scheduler.py 的历史健康评分循环引用，禁止新增逻辑。
"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Callable
from loguru import logger
from enum import Enum

from .alert_rules import AlertRule, AlertCategory, AlertSeverity, ALL_RULES, get_rules_by_category


class AlertActionType(Enum):
    PAUSE_TRADING = "pause_trading"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    NOTIFY_ADMIN = "notify_admin"
    PAUSE_NEW_ENTRIES = "pause_new_entries"
    PAUSE_TRADING_1H = "pause_trading_1h"
    REDUCE_POSITION_SIZE = "reduce_position_size"
    REDUCE_POSITIONS = "reduce_positions"
    CLEANUP_OLD_LOGS = "cleanup_old_logs"
    RECONNECT_WS = "reconnect_ws"


class AlertState(Enum):
    NORMAL = "normal"
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class DynamicThreshold:
    def __init__(self, base_threshold: float, min_threshold: float = None, 
                 max_threshold: float = None, volatility_factor: float = 0.1):
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold if min_threshold is not None else base_threshold * 0.5
        self.max_threshold = max_threshold if max_threshold is not None else base_threshold * 2.0
        self.volatility_factor = volatility_factor
        self.current_threshold = base_threshold
        self._history = deque(maxlen=100)
        self._last_adjust_time = datetime.now()

    def adjust(self, metric_value: float, volatility: float = 0.0):
        self._history.append(metric_value)
        
        if len(self._history) < 10:
            return
        
        avg_value = sum(self._history) / len(self._history)
        std_dev = (sum((v - avg_value) ** 2 for v in self._history) / len(self._history)) ** 0.5
        
        adjustment = volatility * self.volatility_factor
        adjusted_threshold = self.base_threshold * (1 + adjustment)
        
        self.current_threshold = max(
            self.min_threshold,
            min(self.max_threshold, adjusted_threshold)
        )
        
        self._last_adjust_time = datetime.now()

    def get_threshold(self) -> float:
        return self.current_threshold


class AlertInstance:
    def __init__(self, rule: AlertRule, value: float, metadata: Dict[str, Any] = None):
        self.rule = rule
        self.value = value
        self.metadata = metadata or {}
        self.state = AlertState.TRIGGERED
        self.trigger_time = datetime.now()
        self.ack_time = None
        self.resolve_time = None
        self.duration_seconds = 0
        self.executed_actions = []

    def acknowledge(self):
        self.state = AlertState.ACKNOWLEDGED
        self.ack_time = datetime.now()

    def resolve(self):
        self.state = AlertState.RESOLVED
        self.resolve_time = datetime.now()
        self.duration_seconds = (self.resolve_time - self.trigger_time).total_seconds()


class AlertRuleEngine:
    def __init__(self, config: Dict[str, Any], alert_manager):
        self.config = config
        self.alert_manager = alert_manager
        self._active_alerts: Dict[str, AlertInstance] = {}
        self._alert_history = deque(maxlen=5000)
        self._dynamic_thresholds: Dict[str, DynamicThreshold] = {}
        self._action_handlers: Dict[str, Callable] = {}
        self._trigger_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._cooldown_timers: Dict[str, float] = {}
        
        self._initialize_dynamic_thresholds()
        self._register_default_action_handlers()

    def _initialize_dynamic_thresholds(self):
        for rule in ALL_RULES:
            if rule.category in (AlertCategory.TRADING, AlertCategory.PERFORMANCE):
                self._dynamic_thresholds[rule.name] = DynamicThreshold(
                    base_threshold=rule.threshold,
                    min_threshold=rule.threshold * 0.5,
                    max_threshold=rule.threshold * 2.0,
                    volatility_factor=0.15
                )

    def _register_default_action_handlers(self):
        self._action_handlers[AlertActionType.NOTIFY_ADMIN.value] = self._action_notify_admin
        self._action_handlers[AlertActionType.CLEANUP_OLD_LOGS.value] = self._action_cleanup_logs

    def register_action_handler(self, action_type: str, handler: Callable):
        self._action_handlers[action_type] = handler

    async def evaluate_metrics(self, metrics: Dict[str, float]) -> List[AlertInstance]:
        triggered_alerts = []
        
        for metric, value in metrics.items():
            rules = self._get_rules_for_metric(metric)
            
            for rule in rules:
                await self._evaluate_rule(rule, metric, value, triggered_alerts)
        
        return triggered_alerts

    def _get_rules_for_metric(self, metric: str) -> List[AlertRule]:
        return [r for r in ALL_RULES if r.metric == metric]

    async def _evaluate_rule(self, rule: AlertRule, metric: str, value: float, 
                            triggered_alerts: List[AlertInstance]):
        if self._is_on_cooldown(rule):
            return
        
        threshold = self._get_effective_threshold(rule)
        
        if self._compare(value, threshold, rule.comparison):
            alert_key = f"{rule.name}:{rule.metric}"
            
            if alert_key in self._active_alerts:
                self._active_alerts[alert_key].value = value
                self._active_alerts[alert_key].metadata["last_value"] = value
            else:
                metadata = {
                    "metric": metric,
                    "threshold": threshold,
                    "comparison": rule.comparison,
                    "base_threshold": rule.threshold,
                    "is_dynamic": alert_key in self._dynamic_thresholds
                }
                
                alert = AlertInstance(rule, value, metadata)
                self._active_alerts[alert_key] = alert
                self._alert_history.append(alert)
                triggered_alerts.append(alert)
                
                await self._trigger_alert(alert)
                
                await self._execute_actions(alert)

            self._record_trigger(rule.name, value)

        else:
            self._check_resolve(rule, metric, value)
            
        self._update_dynamic_threshold(rule, value)

    def _is_on_cooldown(self, rule: AlertRule) -> bool:
        alert_key = f"{rule.name}:cooldown"
        now = datetime.now().timestamp()
        
        if alert_key in self._cooldown_timers:
            if now < self._cooldown_timers[alert_key]:
                return True
        
        return False

    def _set_cooldown(self, rule: AlertRule):
        alert_key = f"{rule.name}:cooldown"
        self._cooldown_timers[alert_key] = datetime.now().timestamp() + rule.cooldown_seconds

    def _get_effective_threshold(self, rule: AlertRule) -> float:
        if rule.name in self._dynamic_thresholds:
            return self._dynamic_thresholds[rule.name].get_threshold()
        return rule.threshold

    def _compare(self, value: float, threshold: float, op: str) -> bool:
        ops = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
        }
        return ops.get(op, lambda v, t: False)(value, threshold)

    def _check_resolve(self, rule: AlertRule, metric: str, value: float):
        alert_key = f"{rule.name}:{metric}"
        
        if alert_key in self._active_alerts and rule.auto_recover:
            threshold = self._get_effective_threshold(rule)
            
            opposite_ops = {
                "gt": "lt",
                "lt": "gt",
                "gte": "lte",
                "lte": "gte",
                "eq": "ne"
            }
            
            opposite_op = opposite_ops.get(rule.comparison, "ne")
            
            if opposite_op == "ne":
                resolved = value != threshold
            else:
                resolved = self._compare(value, threshold, opposite_op)
            
            if resolved:
                self._active_alerts[alert_key].resolve()
                logger.info(f"Alert resolved: {rule.name}")
                del self._active_alerts[alert_key]

    def _update_dynamic_threshold(self, rule: AlertRule, value: float):
        if rule.name in self._dynamic_thresholds:
            trigger_history = self._trigger_history[rule.name]
            volatility = 0.0
            
            if len(trigger_history) >= 5:
                avg = sum(trigger_history) / len(trigger_history)
                if avg > 0:
                    volatility = sum((v - avg) ** 2 for v in trigger_history) / len(trigger_history)
            
            self._dynamic_thresholds[rule.name].adjust(value, volatility)

    def _record_trigger(self, rule_name: str, value: float):
        self._trigger_history[rule_name].append(value)

    async def _trigger_alert(self, alert: AlertInstance):
        message = (
            f"🚨 {alert.rule.description}\n"
            f"当前值: {alert.value}\n"
            f"阈值: {alert.metadata.get('threshold', alert.rule.threshold)}"
        )
        
        await self.alert_manager.send_alert(
            alert_type=f"RULE_{alert.rule.name.upper()}",
            message=message,
            severity=alert.rule.severity.value.upper(),
            metadata={
                "rule_name": alert.rule.name,
                "metric": alert.rule.metric,
                "value": alert.value,
                "threshold": alert.metadata.get("threshold", alert.rule.threshold),
                "actions": alert.rule.actions
            }
        )
        
        self._set_cooldown(alert.rule)

    async def _execute_actions(self, alert: AlertInstance):
        for action_name in alert.rule.actions:
            if action_name in self._action_handlers:
                try:
                    await self._action_handlers[action_name](alert)
                    alert.executed_actions.append(action_name)
                except Exception as e:
                    logger.error(f"Failed to execute action {action_name}: {e}")

    async def _action_notify_admin(self, alert: AlertInstance):
        message = f"⚠️ 管理员通知：{alert.rule.name}\n{alert.rule.description}"
        await self.alert_manager.send_critical_alert(message)

    async def _action_cleanup_logs(self, alert: AlertInstance):
        try:
            import glob
            log_dir = "./logs"
            cutoff_date = datetime.now() - timedelta(days=7)
            
            for log_file in glob.glob(os.path.join(log_dir, "*.log")):
                if os.path.isfile(log_file):
                    file_time = datetime.fromtimestamp(os.path.getmtime(log_file))
                    if file_time < cutoff_date:
                        os.remove(log_file)
                        logger.info(f"Cleaned old log file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to cleanup logs: {e}")

    def get_active_alerts(self) -> List[AlertInstance]:
        return list(self._active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> List[AlertInstance]:
        return list(self._alert_history)[-limit:]

    def get_dynamic_thresholds(self) -> Dict[str, float]:
        return {name: dt.get_threshold() for name, dt in self._dynamic_thresholds.items()}

    def get_rules_by_category(self, category: str) -> List[AlertRule]:
        cat_enum = AlertCategory(category)
        return get_rules_by_category(cat_enum)

    async def acknowledge_alert(self, alert_key: str) -> bool:
        if alert_key in self._active_alerts:
            self._active_alerts[alert_key].acknowledge()
            return True
        return False

    async def resolve_alert(self, alert_key: str) -> bool:
        if alert_key in self._active_alerts:
            self._active_alerts[alert_key].resolve()
            del self._active_alerts[alert_key]
            return True
        return False
