"""向后兼容 stub：实现已迁移至 eval.alert_evaluator。

本文件仅做 re-export，保持 core.alert_evaluator 导入路径不变。
"""
from eval.alert_evaluator import *  # noqa: F401,F403
from eval.alert_evaluator import (  # noqa: F401
    _SEVERITY_MAP,
    _dispatch_alert_actions,
    _format_alert_message,
    collect_alert_metrics,
    get_latest_triggered,
    notify_triggered_alerts,
    run_alert_evaluator,
)
