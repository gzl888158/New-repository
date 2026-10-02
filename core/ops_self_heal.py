"""
运维自愈闭环协调器 OpsSelfHealCoordinator
============================================

把分散在系统各处的「监控告警（AnomalyDetector / AlertManager）」与「自动恢复
（RecoveryHandler）」通过「根因分析」串成一条带反馈的运维自愈闭环，而不重写它们
内部的检测与恢复算法。

闭环流程（一个运维事件 = 一次 handle_event）：
    监控 Monitor  → 根因 Diagnose  → 决策 Decide  → 恢复 Recover  → 反馈 Reflect

设计约束：
  - 纯编排：只调用 anomaly_detector / recovery_handler / alert_manager 既有接口，
    不复制异常检测器与恢复动作映射。
  - fail-closed：高风险根因（risk/strategy/state）不自动恢复，仅告警 + 建议，
    避免自动执行 PAUSE/REBOOT/ROLLBACK 等危险动作；未知根因同样保守告警。
  - 幂等 + 冷却：同一故障类型在冷却期内跳过，避免重复恢复与日志轰炸。
  - JSON 安全：所有统计经 utils.helpers 的 safe_* 清洗，json.dumps 无 NaN/Inf。
  - 可观测：维护根因分布 + 恢复结果 + 自愈成功率，供 Dashboard / AGI 协调器观察
    「监控→根因→恢复」闭环效果。
"""
import copy
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from utils.helpers import safe_float, safe_int


# 故障类型（含 anomaly type 与 recovery failure_type）→ 根因类别
_ROOT_CAUSE_MAP: Dict[str, str] = {
    # 网络 / 连接
    "connection_lost": "network",
    "api_error": "network",
    "latency_spike": "network",
    "network_timeout": "network",
    # 订单 / 执行
    "order_failed": "order",
    "order_failure_rate": "order",
    "order_failure": "order",
    "order_timeout": "order",
    # 策略
    "strategy_error": "strategy",
    "signal_frequency": "strategy",
    "confidence_anomaly": "strategy",
    # 风险（需人工介入）
    "drawdown_exceeded": "risk",
    "pnl_drop": "risk",
    "price_spike": "risk",
    "volume_surge": "risk",
    "critical_error": "risk",
    # 状态 / 实例（需人工介入）
    "state_corruption": "state",
    "instance_failure": "state",
    # 性能（可降级）
    "performance_degradation": "performance",
}

# 可自动恢复的根因（低风险：重试/取消/降级，不会造成资金损失）
_AUTO_RECOVER_ROOT_CAUSES = {"network", "order", "performance"}
# 仅告警的根因（高风险：暂停/回滚/重启/切换，需人工确认）
_ALERT_ONLY_ROOT_CAUSES = {"risk", "strategy", "state"}


@dataclass
class SelfHealResult:
    """一次运维自愈闭环的统一结果（JSON 安全）。"""
    decision: str = "skip"               # auto_recover / alert_only / cooldown_skip / skip
    root_cause: str = "unknown"
    failure_type: str = ""
    recovered: bool = False
    recovery_status: str = "skipped"     # success / failed / skipped / none
    message: str = ""
    feedback: Dict[str, Any] = field(default_factory=dict)


class OpsSelfHealCoordinator:
    """统一运维自愈协调器：编排 anomaly_detector + recovery_handler + alert_manager。"""

    def __init__(
        self,
        anomaly_detector=None,
        recovery_handler=None,
        alert_manager=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._anomaly_detector = anomaly_detector
        self._recovery_handler = recovery_handler
        self._alert_manager = alert_manager
        self._config = dict(config or {})

        self._cooldown_seconds = safe_float(
            self._config.get("cooldown_seconds"), 60.0
        )
        if self._cooldown_seconds < 0:
            self._cooldown_seconds = 60.0

        self._stats: Dict[str, Any] = {
            "total_events": 0,
            "auto_recovered": 0,
            "recovered_success": 0,
            "recovered_failed": 0,
            "alert_only": 0,
            "cooldown_skip": 0,
            "by_root_cause": {},
        }
        self._cooldown: Dict[str, float] = {}

        logger.info(
            f"OpsSelfHealCoordinator initialized: cooldown={self._cooldown_seconds:.0f}s, "
            f"anomaly_detector={'on' if anomaly_detector else 'off'}, "
            f"recovery_handler={'on' if recovery_handler else 'off'}, "
            f"alert_manager={'on' if alert_manager else 'off'}"
        )

    # ─────────────────────────────────────────────────────────────
    # 运维自愈闭环入口
    # ─────────────────────────────────────────────────────────────

    async def handle_event(self, failure_type: str, context: Optional[Dict[str, Any]] = None) -> SelfHealResult:
        """对单个运维事件执行一次自愈闭环。

        Args:
            failure_type: 故障类型（可为 anomaly type 或 recovery failure_type）
            context: 事件上下文（symbol / retry_count / order_data 等）
        """
        ctx = dict(context or {})
        ftype = str(failure_type or "")

        # 1. 监控 Monitor：入口即监控事件
        # 2. 根因 Diagnose
        root_cause = self.diagnose_root_cause(ftype)

        # 3. 决策 Decide（含冷却）
        decision = self._decide(ftype, ctx, root_cause)

        # 4. 恢复 Recover
        recovered, recovery_status, message = await self._recover(ftype, ctx, root_cause, decision)

        # 5. 反馈 Reflect
        feedback = self._reflect(root_cause, decision, recovered)

        return SelfHealResult(
            decision=decision,
            root_cause=root_cause,
            failure_type=ftype,
            recovered=recovered,
            recovery_status=recovery_status,
            message=message,
            feedback=feedback,
        )

    def diagnose_root_cause(self, failure_type: str) -> str:
        """把故障类型归类到根因类别（纯映射，不重算）。未知回退 unknown。"""
        return _ROOT_CAUSE_MAP.get(str(failure_type or ""), "unknown")

    # ─────────────────────────────────────────────────────────────
    # 决策 Decide
    # ─────────────────────────────────────────────────────────────

    def _decide(self, failure_type: str, context: Dict[str, Any], root_cause: str) -> str:
        # 冷却期内：跳过（幂等）
        key = self._cooldown_key(failure_type, context)
        now = time.monotonic()
        last = self._cooldown.get(key)
        if last is not None and (now - last) < self._cooldown_seconds:
            return "cooldown_skip"

        # 非冷却跳过：立即更新冷却时间戳（与 _decide 用同一 key）
        self._cooldown[key] = now

        # fail-closed：未知根因保守告警，不自动恢复
        if root_cause in ("unknown",) or root_cause in _ALERT_ONLY_ROOT_CAUSES:
            return "alert_only"

        if root_cause in _AUTO_RECOVER_ROOT_CAUSES:
            return "auto_recover"

        # 兜底：未知类别保守告警
        return "alert_only"

    @staticmethod
    def _cooldown_key(failure_type: str, context: Dict[str, Any]) -> str:
        symbol = str(context.get("symbol", "") or "")
        return f"{failure_type}:{symbol}" if symbol else failure_type

    # ─────────────────────────────────────────────────────────────
    # 恢复 Recover
    # ─────────────────────────────────────────────────────────────

    async def _recover(self, failure_type, context, root_cause, decision) -> tuple:
        """执行恢复：auto_recover 调 recovery_handler；alert_only 发告警；其余跳过。"""
        if decision == "cooldown_skip":
            return False, "skipped", "cooldown active"

        if decision == "alert_only":
            await self._send_alert(failure_type, root_cause, context, "auto recovery blocked (high-risk root cause)")
            return False, "skipped", f"alert only ({root_cause})"

        if decision == "auto_recover":
            if self._recovery_handler is None:
                await self._send_alert(failure_type, root_cause, context, "no recovery handler available")
                return False, "failed", "no recovery handler"
            try:
                result = await self._recovery_handler.handle_failure(failure_type, context)
                status = str(result.get("status", "unknown"))
                if status == "success":
                    return True, "success", str(result.get("action", ""))
                return False, "failed", str(result.get("error", result.get("reason", status)))
            except Exception as e:
                logger.warning(f"[OpsSelfHeal] recovery failed for {failure_type}: {e}")
                await self._send_alert(failure_type, root_cause, context, f"recovery exception: {e}")
                return False, "failed", str(e)

        return False, "skipped", "no action"

    async def _send_alert(self, failure_type, root_cause, context, reason):
        """发送运维告警（告警管理器不可用或发送失败静默降级）。"""
        if self._alert_manager is None:
            return
        try:
            send = getattr(self._alert_manager, "send_system_alert", None)
            if callable(send):
                await send(
                    f"self_heal_{root_cause}",
                    f"自愈闭环: {failure_type} [{root_cause}] {reason}",
                    metadata={"failure_type": failure_type, "root_cause": root_cause,
                              "context": context, "reason": reason},
                )
        except Exception as e:
            logger.debug(f"[OpsSelfHeal] alert failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # 反馈 Reflect
    # ─────────────────────────────────────────────────────────────

    def _reflect(self, root_cause, decision, recovered) -> Dict[str, Any]:
        try:
            s = self._stats
            s["total_events"] = safe_int(s.get("total_events"), 0) + 1

            rc = s["by_root_cause"].setdefault(
                root_cause, {"total": 0, "auto_recovered": 0, "recovered_success": 0,
                             "recovered_failed": 0, "alert_only": 0, "cooldown_skip": 0}
            )
            rc["total"] = safe_int(rc.get("total"), 0) + 1

            if decision == "auto_recover":
                s["auto_recovered"] = safe_int(s.get("auto_recovered"), 0) + 1
                rc["auto_recovered"] = safe_int(rc.get("auto_recovered"), 0) + 1
                if recovered:
                    s["recovered_success"] = safe_int(s.get("recovered_success"), 0) + 1
                    rc["recovered_success"] = safe_int(rc.get("recovered_success"), 0) + 1
                else:
                    s["recovered_failed"] = safe_int(s.get("recovered_failed"), 0) + 1
                    rc["recovered_failed"] = safe_int(rc.get("recovered_failed"), 0) + 1
            elif decision == "alert_only":
                s["alert_only"] = safe_int(s.get("alert_only"), 0) + 1
                rc["alert_only"] = safe_int(rc.get("alert_only"), 0) + 1
            elif decision == "cooldown_skip":
                s["cooldown_skip"] = safe_int(s.get("cooldown_skip"), 0) + 1
                rc["cooldown_skip"] = safe_int(rc.get("cooldown_skip"), 0) + 1
        except Exception as e:
            logger.debug(f"[OpsSelfHeal] reflect stats failed: {e}")

        return {"root_cause": root_cause, "decision": decision, "recovered": recovered}

    # ─────────────────────────────────────────────────────────────
    # 可观测
    # ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """返回运维自愈闭环统计（深拷贝，不暴露内部引用）。"""
        return copy.deepcopy(self._stats)

    def get_self_heal_summary(self) -> Dict[str, Any]:
        """聚合根因统计 + 恢复引擎 summary + 异常摘要，供 Dashboard / AGI 观察闭环效果。"""
        summary = {
            "stats": copy.deepcopy(self._stats),
            "success_rate": self._success_rate(),
        }
        if self._recovery_handler is not None:
            try:
                summary["recovery"] = self._recovery_handler.get_recovery_summary()
            except Exception as e:
                logger.debug(f"[OpsSelfHeal] recovery summary failed: {e}")
        if self._anomaly_detector is not None:
            try:
                summary["anomaly"] = self._anomaly_detector.get_anomaly_summary()
            except Exception as e:
                logger.debug(f"[OpsSelfHeal] anomaly summary failed: {e}")
        return summary

    def _success_rate(self) -> float:
        total = safe_int(self._stats.get("auto_recovered"), 0)
        if total <= 0:
            return 0.0
        return safe_float(self._stats.get("recovered_success"), 0.0) / total


__all__ = ["OpsSelfHealCoordinator", "SelfHealResult"]
