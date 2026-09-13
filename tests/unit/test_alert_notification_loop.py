"""
告警 → 通知渠道（Webhook/Telegram）闭环测试（模块 7 (3)）
=====================================================

验证触发告警经 notifier 逐条投递，打通「指标采集 → 阈值告警 → 通知渠道」闭环：
- 严重级别映射（info/warning/critical/emergency → INFO/WARNING/CRITICAL）
- 通知文本格式化（含升级标记）
- 空列表 / 无 notifier 静默跳过
- 逐条投递与参数正确性
- notifier 投递失败静默降级（不抛异常）
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import MagicMock, AsyncMock

from core.alert_evaluator import (
    notify_triggered_alerts,
    _format_alert_message,
    _SEVERITY_MAP,
)
from core.alert_registry import TriggeredAlert


def _alert(rule_name="drawdown_breach", severity="critical", value=0.3, escalated=False):
    return TriggeredAlert(
        rule_name=rule_name,
        category="risk",
        severity=severity,
        metric="drawdown_pct",
        value=value,
        threshold=0.25,
        comparison="gte",
        description="账户回撤超阈值",
        actions=["pause_trading"],
        suggest_actions=["降低仓位"],
        escalated=escalated,
    )


def test_severity_map():
    assert _SEVERITY_MAP["info"] == "INFO"
    assert _SEVERITY_MAP["warning"] == "WARNING"
    assert _SEVERITY_MAP["critical"] == "CRITICAL"
    assert _SEVERITY_MAP["emergency"] == "EMERGENCY"


def test_format_alert_message():
    msg = _format_alert_message(_alert())
    assert "drawdown_breach" in msg
    assert "critical" in msg
    assert "drawdown_pct" in msg
    assert "[已升级]" not in msg

    msg2 = _format_alert_message(_alert(escalated=True))
    assert "[已升级]" in msg2


def test_notify_empty_or_none_notifier():
    notifier = MagicMock()
    notifier.send_alert = AsyncMock(return_value=None)
    assert notify_triggered_alerts([], notifier) == 0
    assert notify_triggered_alerts([_alert()], None) == 0
    notifier.send_alert.assert_not_called()


def test_notify_dispatches_each_alert():
    notifier = MagicMock()
    notifier.send_alert = AsyncMock(return_value=None)

    alerts = [_alert("risk_rule_a", "warning"), _alert("risk_rule_b", "critical")]
    n = notify_triggered_alerts(alerts, notifier)

    assert n == 2
    assert notifier.send_alert.await_count == 2

    first = notifier.send_alert.await_args_list[0].kwargs
    assert first["alert_type"] == "ALERT_RISK_RULE_A"
    assert first["severity"] == "WARNING"
    assert first["symbol"] == ""
    assert first["metadata"]["rule_name"] == "risk_rule_a"

    second = notifier.send_alert.await_args_list[1].kwargs
    assert second["alert_type"] == "ALERT_RISK_RULE_B"
    assert second["severity"] == "CRITICAL"


def test_notify_survives_notifier_failure():
    notifier = MagicMock()

    async def _fail(**kwargs):
        raise RuntimeError("webhook down")

    notifier.send_alert = _fail

    alerts = [_alert("a"), _alert("b")]
    # 每条都尝试投递，失败静默降级 → 返回 0 且不抛异常
    assert notify_triggered_alerts(alerts, notifier) == 0
