"""
告警单一入口的升级/去重策略测试（模块 7 (2)）
============================================

验证 AlertRegistry（核心告警唯一入口）具备：
- 告警级别：info/warning/critical/emergency 权重与升级顺序
- 去重：同一规则在冷却期内不重复触发（evaluate 内冷却保证）
- 升级：同一规则在升级窗口内反复触发达阈值后严重级别上调一级（上限 EMERGENCY）

注意：升级统计跨冷却周期，测试用 clear_cooldowns() 模拟冷却到期。
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.alert_registry import (
    AlertRule,
    AlertCategory,
    AlertSeverity,
    get_alert_registry,
)


def _warning_rule(name="test_rule"):
    return AlertRule(
        name=name,
        category=AlertCategory.RISK,
        severity=AlertSeverity.WARNING,
        metric="m",
        threshold=1.0,
        comparison="gte",
        description="测试规则",
    )


def test_severity_weights_and_escalation_order():
    assert AlertSeverity.INFO.weight < AlertSeverity.WARNING.weight
    assert AlertSeverity.WARNING.weight < AlertSeverity.CRITICAL.weight
    assert AlertSeverity.CRITICAL.weight < AlertSeverity.EMERGENCY.weight


def test_no_escalation_below_threshold():
    registry = get_alert_registry()
    registry.reset()
    registry.configure_escalation(window_seconds=3600, threshold=3)

    rule = _warning_rule()
    base = 1000.0
    sev, esc = registry._record_trigger_and_escalate(rule, base)
    assert sev == AlertSeverity.WARNING and esc is False
    sev, esc = registry._record_trigger_and_escalate(rule, base + 1)
    assert sev == AlertSeverity.WARNING and esc is False


def test_escalation_warning_to_critical_at_threshold():
    registry = get_alert_registry()
    registry.reset()
    registry.configure_escalation(window_seconds=3600, threshold=3)

    rule = _warning_rule()
    base = 1000.0
    registry._record_trigger_and_escalate(rule, base)
    registry._record_trigger_and_escalate(rule, base + 1)
    sev, esc = registry._record_trigger_and_escalate(rule, base + 2)
    assert sev == AlertSeverity.CRITICAL
    assert esc is True


def test_escalation_caps_at_emergency():
    registry = get_alert_registry()
    registry.reset()
    registry.configure_escalation(window_seconds=3600, threshold=3)

    rule = AlertRule(
        name="emergency_rule",
        category=AlertCategory.RISK,
        severity=AlertSeverity.EMERGENCY,
        metric="m",
        threshold=1.0,
        comparison="gte",
        description="测试规则",
    )
    base = 1000.0
    for i in range(5):
        sev, esc = registry._record_trigger_and_escalate(rule, base + i)
    assert sev == AlertSeverity.EMERGENCY
    assert esc is False  # 已到上限，不再「升级」


def test_escalation_window_pruning():
    registry = get_alert_registry()
    registry.reset()
    registry.configure_escalation(window_seconds=10, threshold=3)

    rule = _warning_rule()
    registry._record_trigger_and_escalate(rule, 1000.0)
    registry._record_trigger_and_escalate(rule, 1001.0)
    # 时间跃迁到窗口之外，旧触发被修剪，仅剩当前一次 → 不升级
    sev, esc = registry._record_trigger_and_escalate(rule, 2000.0)
    assert sev == AlertSeverity.WARNING
    assert esc is False


def test_evaluate_end_to_end_escalation():
    """走公开 evaluate 路径，验证 escalated 标记与升级后 severity 正确回传。"""
    registry = get_alert_registry()
    registry.reset()
    registry.clear_cooldowns()
    registry.configure_escalation(window_seconds=3600, threshold=3)

    # consecutive_losses 规则：threshold 5, gte, WARNING, cooldown 600s
    r1 = registry.evaluate({"consecutive_losses": 6})
    assert len(r1) == 1
    assert r1[0].severity == "warning" and r1[0].escalated is False

    registry.clear_cooldowns()
    r2 = registry.evaluate({"consecutive_losses": 6})
    assert r2[0].severity == "warning" and r2[0].escalated is False

    registry.clear_cooldowns()
    r3 = registry.evaluate({"consecutive_losses": 6})
    assert r3[0].severity == "critical"
    assert r3[0].escalated is True
    # 升级信息应体现在 to_dict 中
    assert r3[0].to_dict()["escalated"] is True


def test_evaluate_dedup_within_cooldown():
    """冷却期内重复 evaluate 不重复触发（去重）。"""
    registry = get_alert_registry()
    registry.reset()
    registry.clear_cooldowns()
    registry.configure_escalation(window_seconds=3600, threshold=3)

    first = registry.evaluate({"consecutive_losses": 6})
    assert len(first) == 1
    # 未清冷却，再次评估应被去重
    second = registry.evaluate({"consecutive_losses": 6})
    assert len(second) == 0
