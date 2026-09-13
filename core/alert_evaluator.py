"""
企业级告警自动评估引擎
========================
周期性采集核心风险/交易/系统指标，自动调用 AlertRegistry.evaluate()，
实现告警的自动触发（不再依赖手动评估）。

设计原则：
- 只采集 dashboard 进程能可靠获取的指标（账户/风险/交易统计/系统资源/DB），
  需要主进程内存态内部状态的指标（滑点/信号执行率/凯利等）不在此采集，
  仍可通过 /api/alert_evaluate 手动评估。
- 采集失败不抛异常，静默降级为不传该指标。
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from loguru import logger

try:
    import psutil
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    _HAS_PSUTIL = False


# 最近一次自动评估结果（供 SSE/API 读取，线程安全由调用方保证）
_latest_triggered: list = []
_latest_evaluate_time: Optional[str] = None
_state_lock = None

# 策略质量指标（期望值/盈亏比/胜率）上报所需的最小样本数，避免小样本误报
_MIN_SAMPLE = 5
# 策略质量指标统计窗口（小时）：仅统计近 N 小时成交，避免历史回撤期亏损永久拖累期望值导致误暂停
_QUALITY_WINDOW_HOURS = 24


def _get_state_lock():
    global _state_lock
    if _state_lock is None:
        import threading
        _state_lock = threading.Lock()
    return _state_lock


def get_latest_triggered() -> Dict[str, Any]:
    """获取最近一次自动评估触发的告警"""
    with _get_state_lock():
        return {
            "triggered_alerts": [a for a in _latest_triggered],
            "count": len(_latest_triggered),
            "evaluate_time": _latest_evaluate_time,
            "auto": True,
        }


def _get_db_path(engine) -> Optional[str]:
    """从 DashboardEngine 解析 DB 绝对路径"""
    p = getattr(engine, "_db_path", None)
    if p:
        return p
    return None


def _query_db(db_path: str, sql: str, params=()):
    conn = sqlite3.connect(db_path, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


def _collect_trade_metrics(db_path: str, equity: float) -> Dict[str, float]:
    """从 trade_records 采集交易统计指标（连亏、无成交时长、小时亏损）"""
    metrics: Dict[str, float] = {}
    try:
        rows = _query_db(
            db_path,
            "SELECT pnl, close_time FROM trade_records "
            "WHERE status='closed' AND pnl IS NOT NULL "
            "ORDER BY close_time DESC LIMIT 200",
        )

        # 连续亏损笔数
        consec = 0
        for r in rows:
            if float(r["pnl"] or 0) < 0:
                consec += 1
            else:
                break
        metrics["consecutive_losses"] = float(consec)

        # 距最近一次成交的小时数
        if rows and rows[0]["close_time"]:
            try:
                dt = datetime.fromisoformat(str(rows[0]["close_time"]))
                metrics["hours_since_last_trade"] = max(
                    0.0, (datetime.now() - dt).total_seconds() / 3600.0
                )
            except Exception:
                pass

        # 近 1 小时净亏损幅度（正值）
        cutoff = (datetime.now() - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        row = _query_db(
            db_path,
            "SELECT SUM(pnl) AS s FROM trade_records "
            "WHERE status='closed' AND close_time >= ? AND pnl IS NOT NULL",
            (cutoff,),
        )
        if row and row[0]["s"] is not None:
            hour_pnl = float(row[0]["s"])
            if equity > 0 and hour_pnl < 0:
                metrics["hourly_loss_pct"] = abs(hour_pnl) / equity
    except Exception as e:
        logger.debug(f"alert_evaluator: trade metrics error: {e}")
    return metrics


def _collect_strategy_quality_metrics(db_path: str) -> Dict[str, float]:
    """采集策略质量指标（期望值 / 盈亏比 / 胜率）。

    仅统计近 _QUALITY_WINDOW_HOURS 小时的成交，并在样本充足（>= _MIN_SAMPLE）
    时上报，避免历史回撤期亏损永久拖累期望值导致误暂停、也避免小样本误报。
    """
    metrics: Dict[str, float] = {}
    try:
        cutoff = (datetime.now() - timedelta(hours=_QUALITY_WINDOW_HOURS)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        rows = _query_db(
            db_path,
            "SELECT pnl FROM trade_records "
            "WHERE status='closed' AND pnl IS NOT NULL AND close_time >= ?",
            (cutoff,),
        )
        pnls = [float(r["pnl"]) for r in rows]
        total = len(pnls)
        if total < _MIN_SAMPLE:
            return metrics

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        total_pnl = sum(pnls)

        metrics["win_rate"] = len(wins) / total
        # 期望值：平均每笔净盈亏（USDT）
        metrics["expectancy_usdt"] = total_pnl / total

        avg_win = (sum(wins) / len(wins)) if wins else 0.0
        avg_loss = (sum(abs(p) for p in losses) / len(losses)) if losses else 0.0
        # 仅在有亏损时产出利润因子（有盈无亏不触发 low_profit_factor）
        if avg_loss > 0:
            metrics["profit_factor"] = avg_win / avg_loss
    except Exception as e:
        logger.debug(f"alert_evaluator: strategy quality metrics error: {e}")
    return metrics


def _collect_system_metrics() -> Dict[str, float]:
    """采集系统资源指标（psutil，可选）"""
    metrics: Dict[str, float] = {}
    if not _HAS_PSUTIL:
        return metrics
    try:
        metrics["memory_usage_pct"] = float(psutil.virtual_memory().percent)
        metrics["cpu_usage_pct"] = float(psutil.cpu_percent(interval=None))
        disk_path = os.path.splitdrive(os.path.abspath("."))[0] + "\\" or "C:\\"
        metrics["disk_free_gb"] = float(psutil.disk_usage(disk_path).free) / (1024 ** 3)

        # 主交易进程存活检测：是否存在命令行含 main.py/start.py 的 python 进程
        trading_alive = 0
        for proc in psutil.process_iter(["name", "cmdline"]):
            try:
                name = (proc.info.get("name") or "").lower()
                cmdline = " ".join(proc.info.get("cmdline") or [])
                if ("python" in name) and ("main.py" in cmdline or "start.py" in cmdline):
                    trading_alive = 1
                    break
            except Exception:
                continue
        metrics["trading_process"] = float(trading_alive)
    except Exception as e:
        logger.debug(f"alert_evaluator: system metrics error: {e}")
    return metrics


def _collect_account_metrics(engine) -> Dict[str, float]:
    """从 DashboardEngine 采集账户/风险指标"""
    metrics: Dict[str, float] = {}
    try:
        snapshot = engine.get_account_snapshot()
        equity = float(getattr(snapshot, "total_equity", 0) or 0)
        if equity <= 0:
            return metrics

        # 杠杆（持仓最大杠杆）
        try:
            pos_dist = engine.get_position_distribution()
            positions = pos_dist.get("positions", []) if isinstance(pos_dist, dict) else []
            max_lev = 1
            long_cnt = 0
            short_cnt = 0
            max_pos_loss = 0.0
            for p in positions:
                lev = int(p.get("leverage", 1) or 1)
                max_lev = max(max_lev, lev)
                side = str(p.get("side", "long")).lower()
                if side == "short":
                    short_cnt += 1
                else:
                    long_cnt += 1
                # 单仓位亏损比例（正值）
                pnl_pct = float(p.get("pnl_pct", 0) or 0)
                if pnl_pct < 0:
                    max_pos_loss = max(max_pos_loss, abs(pnl_pct))
            metrics["leverage"] = float(max_lev)
            if long_cnt + short_cnt > 0:
                metrics["long_short_ratio"] = float(long_cnt) / max(float(short_cnt), 1.0)
            if max_pos_loss > 0:
                metrics["position_loss_pct"] = max_pos_loss
        except Exception:
            pass

        # 保证金可用占比（越高越安全，低于 50% 触发 margin_call_warning）
        metrics["margin_rate"] = float(getattr(snapshot, "margin_ratio", 0) or 0)

        # 日内亏损幅度（正值）
        realized_today = float(getattr(snapshot, "realized_pnl_today", 0) or 0)
        if realized_today < 0:
            metrics["daily_loss_pct"] = abs(realized_today) / equity

        # 当前回撤
        try:
            risk = engine.get_risk_dashboard()
            rm = risk.get("risk_metrics", {}) if isinstance(risk, dict) else {}
            dd = float(rm.get("current_drawdown", 0) or 0)
            if dd > 0:
                metrics["drawdown_pct"] = dd
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"alert_evaluator: account metrics error: {e}")
    return metrics


def collect_alert_metrics(engine) -> Dict[str, float]:
    """汇总采集所有可用的告警指标"""
    metrics: Dict[str, float] = {}

    account = _collect_account_metrics(engine)
    metrics.update(account)

    equity = 0.0
    try:
        snap = engine.get_account_snapshot()
        equity = float(getattr(snap, "total_equity", 0) or 0)
    except Exception:
        pass

    db_path = _get_db_path(engine)
    if db_path:
        metrics.update(_collect_trade_metrics(db_path, equity))
        metrics.update(_collect_strategy_quality_metrics(db_path))
        try:
            _query_db(db_path, "SELECT 1")
            metrics["db_connected"] = 1.0
        except Exception:
            metrics["db_connected"] = 0.0

    metrics.update(_collect_system_metrics())
    return metrics


# 触发全局暂停的动作标识：仅当规则自身 actions 声明了「暂停交易/暂停新开仓/暂停1小时」
# 才下发 global_pause，避免把 margin_call_warning(减仓)、trading_process_down(仅通知)
# 等规则误判为暂停，导致小账户被长期暂停或重启即暂停。
# 注：pause_trading_1h（hourly_loss_limit）同样映射为 global_pause，宁可暂停不可漏风控。
_PAUSE_ACTIONS = ("pause_trading", "pause_new_entries", "pause_trading_1h")
# 同一规则在 N 秒内不重复下发干预信号
_DISPATCH_INTERVAL = 600
_dispatched_rules: Dict[str, float] = {}


def _dispatch_alert_actions(triggered) -> int:
    """将「明确要求暂停交易」的告警下发为干预信号（告警→交易动作闭环）。

    仅当规则自身 actions 包含 pause_trading / pause_new_entries / pause_trading_1h
    时下发 global_pause；其他动作（reduce_positions、notify_admin、close_all_positions
    等）不自动执行，其中不可逆动作（close_all_positions）仍需人工确认。
    """
    if not triggered:
        return 0

    actionable = [
        a for a in triggered
        if any(act in _PAUSE_ACTIONS for act in (a.actions or []))
    ]
    if not actionable:
        return 0

    now = time.time()
    dispatched = 0
    for alert in actionable:
        action = "global_pause"
        if now - _dispatched_rules.get(alert.rule_name, 0.0) < _DISPATCH_INTERVAL:
            continue

        reason = (f"[自动告警] {alert.description} "
                  f"(metric={alert.metric}, value={alert.value:.4f}, "
                  f"threshold={alert.threshold})")

        # 写入干预信号文件
        try:
            sig_path = "./data/intervention_signal.json"
            os.makedirs(os.path.dirname(sig_path), exist_ok=True)
            with open(sig_path, "w", encoding="utf-8") as f:
                json.dump({
                    "action": action,
                    "timestamp": now,
                    "requested_at": datetime.now().isoformat(),
                    "source": "alert_engine",
                    "reason": reason,
                    "confirmed": False,
                }, f, ensure_ascii=False)
        except Exception as e:
            logger.error(f"告警干预信号写入失败: {e}")
            continue

        # 追加干预历史
        try:
            hist_path = "./data/intervention_history.json"
            history = []
            if os.path.exists(hist_path):
                try:
                    with open(hist_path, "r", encoding="utf-8") as f:
                        history = json.load(f)
                except Exception:
                    history = []
            history.append({
                "timestamp": datetime.now().isoformat(),
                "action": action,
                "reason": reason,
            })
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(history[-100:], f, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"告警干预历史追加失败: {e}")

        _dispatched_rules[alert.rule_name] = now
        dispatched += 1
        logger.warning(f"[告警闭环] {alert.rule_name}({alert.severity}) → {action}")

    return dispatched


# ============================================================
# 告警 → 通知渠道（Webhook/Telegram）闭环（模块 7 (3)）
# ============================================================

# 注册中心严重级别 → AlertManager 严重级别映射
# emergency 保留为 EMERGENCY（AlertManager 已支持，见 _should_filter 的 levels 与
# _send_telegram 的 emoji），不再降级为 CRITICAL，避免紧急告警被误过滤漏报。
_SEVERITY_MAP = {
    "info": "INFO",
    "warning": "WARNING",
    "critical": "CRITICAL",
    "emergency": "EMERGENCY",
}


def _format_alert_message(alert) -> str:
    """将 TriggeredAlert 格式化为人类可读的通知文本"""
    esc = " [已升级]" if getattr(alert, "escalated", False) else ""
    return (
        f"告警: {alert.rule_name}{esc} ({alert.severity})\n"
        f"{alert.description}\n"
        f"metric={alert.metric} value={alert.value:.4f} "
        f"threshold={alert.threshold} {alert.comparison}"
    )


def notify_triggered_alerts(triggered, notifier) -> int:
    """将触发的告警推送到通知渠道（Webhook/Telegram），打通闭环。

    去重由 notifier 内部（AlertManager 类型级 + 内容级限流）与注册中心冷却期共同保证，
    此处只负责「格式化 + 逐条投递」。

    Args:
        triggered: AlertRegistry.evaluate() 返回的 TriggeredAlert 列表
        notifier: 提供 `send_alert(alert_type, message, severity, symbol, metadata)` 的
            异步通知器（AlertManager 兼容）。为 None 时静默跳过。

    Returns:
        实际投递的告警条数（失败静默降级，不计入）。
    """
    if not triggered or notifier is None:
        return 0

    async def _send():
        sent = 0
        for alert in triggered:
            severity = _SEVERITY_MAP.get(alert.severity, "WARNING")
            try:
                await notifier.send_alert(
                    alert_type=f"ALERT_{alert.rule_name.upper()}",
                    message=_format_alert_message(alert),
                    severity=severity,
                    symbol="",
                    metadata=alert.to_dict(),
                )
                sent += 1
            except Exception as e:
                logger.warning(f"告警通知投递失败 {alert.rule_name}: {e}")
        return sent

    try:
        # 评估线程中无运行中的事件循环，用短生命周期 loop 同步执行异步投递
        return asyncio.run(_send())
    except Exception as e:
        logger.warning(f"告警通知分发异常: {e}")
        return 0


def run_alert_evaluator(engine, interval: int = 60, stop_event=None, notifier=None):
    """后台循环：周期采集指标 → 自动评估 → 记录触发 → 干预信号 → 通知渠道

    Args:
        engine: DashboardEngine 实例
        interval: 评估周期（秒），默认 60s
        stop_event: threading.Event，置位后退出循环
        notifier: 可选，AlertManager 兼容的通知器；提供后触发告警将推送到
            Webhook/Telegram（指标采集→阈值告警→通知渠道闭环）。
    """
    from core.alert_registry import get_alert_registry

    registry = get_alert_registry()
    logger.info(f"告警自动评估引擎已启动（周期 {interval}s）")

    while stop_event is None or not stop_event.is_set():
        try:
            metrics = collect_alert_metrics(engine)
            triggered = registry.evaluate(metrics)
            _dispatch_alert_actions(triggered)
            if notifier is not None:
                notify_triggered_alerts(triggered, notifier)

            global _latest_triggered, _latest_evaluate_time
            with _get_state_lock():
                _latest_triggered = [a.to_dict() for a in triggered]
                _latest_evaluate_time = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"告警自动评估异常: {e}")

        # 分段休眠，便于快速响应 stop_event
        slept = 0
        while slept < interval:
            if stop_event is not None and stop_event.is_set():
                return
            time.sleep(1)
            slept += 1
