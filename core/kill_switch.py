"""P0 企业级升级：持久化全局 Kill Switch（软暂停开关）

对标 DE Shaw / Jump 的独立 Kill Switch 范式：
- 一键禁止所有「开仓」，平仓/减仓/止损/止盈等降风险信号照常放行（永不阻断降风险路径）
- fail-closed：状态持久化到磁盘，进程崩溃/重启后仍保持「关闭」，不自动恢复开仓
- 独立于 EmergencyCircuitBreaker 的「冷却自动恢复」，本开关不自动解除，需显式 disable

与 EmergencyCircuitBreaker 的区别（互补而非重复）：
- EmergencyCircuitBreaker：触发即全平 + 冷却期结束后自动恢复（应对瞬间极端行情）
- KillSwitch：软暂停（只禁开仓、允许平仓），手动解除才恢复 + 重启保持（应对需要人工介入的持续风险）

设计约束：本模块零外部业务依赖，仅依赖 core.atomic_writer，可独立单测。
"""

import os
import json
import threading
from datetime import datetime
from typing import Optional

from loguru import logger

from core.atomic_writer import atomic_write_json

# 触发历史最大保留条数（FIFO，防状态文件无限膨胀）
_MAX_HISTORY = 50


class KillSwitch:
    """持久化全局开仓暂停开关（fail-closed / 重启保持 / 平仓穿透 / 触发历史审计）。"""

    def __init__(self, state_file: Optional[str] = None):
        if state_file is None:
            state_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "kill_switch_state.json",
            )
        self._state_file = state_file
        self._lock = threading.RLock()

        self._enabled = False
        self._reason = ""
        self._enabled_by = ""
        self._triggered_at: Optional[datetime] = None
        self._history: list = []

        # P0: 状态变更通知回调（可选注入，不破坏零依赖设计）
        # 由 scheduler 注入 alert_manager 发送告警，运营者能感知 KillSwitch 被开启/关闭
        self._on_change_callbacks: list = []

        # 启动时从磁盘恢复（fail-closed：磁盘上若为 enabled，则保持 enabled）
        self._load()

    # ── 持久化 ─────────────────────────────────────────────
    def _load(self) -> None:
        try:
            if not os.path.exists(self._state_file):
                logger.debug("KillSwitch state file not found, default disabled")
                return
            with open(self._state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._enabled = bool(data.get("enabled", False))
            self._reason = data.get("reason", "") or ""
            self._enabled_by = data.get("enabled_by", "") or ""
            self._history = data.get("history", []) or []
            if not isinstance(self._history, list):
                self._history = []
            ts = data.get("triggered_at")
            if ts:
                try:
                    self._triggered_at = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    self._triggered_at = None
            if self._enabled:
                logger.warning(
                    f"KillSwitch restored as ENABLED from disk (fail-closed): {self._reason}"
                )
        except Exception as e:
            # 读取失败时保守兜底（fail-closed）：宁可禁开仓，也不因状态未知而误放行
            logger.error(f"Failed to load KillSwitch state, defaulting to ENABLED (fail-closed): {e}")
            self._enabled = True
            self._reason = f"状态加载失败，保守禁开仓: {e}"

    def register_change_callback(self, callback) -> None:
        """注册状态变更回调：KillSwitch enable/disable 时通知（用于发送告警）。
        回调签名：callback(action: str, reason: str, by: str)
        """
        with self._lock:
            self._on_change_callbacks.append(callback)

    def _notify_change(self, action: str, reason: str, by: str) -> None:
        """触发所有注册的状态变更回调（异常不影响主流程）。"""
        for cb in self._on_change_callbacks:
            try:
                cb(action, reason, by)
            except Exception as e:
                logger.error(f"KillSwitch change callback error: {e}")

    def _append_history(self, action: str, reason: str, by: str) -> None:
        """追加一次触发/解除历史（FIFO 截断，防无限膨胀）。"""
        self._history.append({
            "action": action,
            "reason": reason,
            "by": by,
            "at": datetime.now().isoformat(),
        })
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

    def _save(self) -> None:
        data = {
            "enabled": self._enabled,
            "reason": self._reason,
            "enabled_by": self._enabled_by,
            "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            "history": self._history,
            "saved_at": datetime.now().isoformat(),
        }
        if not atomic_write_json(self._state_file, data):
            logger.error(f"Failed to persist KillSwitch state to {self._state_file}")

    # ── 操作 ─────────────────────────────────────────────
    def enable(self, reason: str = "", by: str = "") -> None:
        """启用全局开关：禁止新开仓（平仓放行），并持久化。"""
        with self._lock:
            self._enabled = True
            self._reason = reason
            self._enabled_by = by
            self._triggered_at = datetime.now()
            self._append_history("enable", reason, by)
            self._save()
            callbacks = list(self._on_change_callbacks)
        logger.critical(
            f"⛔ Global KillSwitch ENABLED (新开仓已禁止，平仓照常): {reason or '(no reason)'}"
        )
        self._notify_change_with(callbacks, "enable", reason, by)

    def disable(self, reason: str = "", by: str = "") -> None:
        """解除全局开关：恢复新开仓，并持久化。"""
        with self._lock:
            self._enabled = False
            self._reason = reason or "手动解除"
            self._enabled_by = by
            self._triggered_at = None
            self._append_history("disable", self._reason, by)
            self._save()
            callbacks = list(self._on_change_callbacks)
        logger.info(f"✅ Global KillSwitch DISABLED: {self._reason}")
        self._notify_change_with(callbacks, "disable", self._reason, by)

    @staticmethod
    def _notify_change_with(callbacks, action, reason, by):
        """在锁外触发回调，避免回调内部再调用 KillSwitch 方法导致死锁。"""
        for cb in callbacks:
            try:
                cb(action, reason, by)
            except Exception as e:
                logger.error(f"KillSwitch change callback error: {e}")

    def get_history(self) -> list:
        """返回触发/解除历史（最近 _MAX_HISTORY 条，供 Dashboard 复盘）。"""
        with self._lock:
            return list(self._history)

    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    def get_reason(self) -> str:
        with self._lock:
            return self._reason

    def get_triggered_at(self) -> Optional[datetime]:
        with self._lock:
            return self._triggered_at

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "enabled": self._enabled,
                "reason": self._reason,
                "enabled_by": self._enabled_by,
                "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
                "history": list(self._history),
            }