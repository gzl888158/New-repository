"""
参数变更验证 + 自动回滚守卫
============================
策略参数热更新后，记录基线指标（PnL / 胜率），在验证窗口结束后对比当前指标，
若出现显著退化（PnL 下降超阈值 或 胜率下降超阈值）则给出回滚建议，供
StrategyManager 执行 `rollback_config` 恢复上一版本配置。

设计原则：
- 纯内存状态 + 线程安全（RLock），不依赖数据库，便于独立单元测试。
- 与 StrategyManager 的配置快照/回滚解耦：本守卫只做"是否回滚"的决策，
  实际的配置回滚动作由调用方（StrategyManager）基于决策执行。
- 样本不足时不判定退化（避免小样本噪声误回滚）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger


# 默认阈值：PnL 下降 10% 或 胜率下降 20 个百分点即判定退化（对齐项目既有约定）
DEFAULT_VALIDATION_WINDOW_SECONDS = 1800.0  # 30 分钟
DEFAULT_PNL_DROP_THRESHOLD = 0.10
DEFAULT_WIN_RATE_DROP_THRESHOLD = 0.20
DEFAULT_MIN_TRADES = 5


@dataclass
class RollbackDecision:
    """参数验证的回滚决策"""
    strategy_name: str
    should_rollback: bool
    reason: str = ""
    pnl_change_pct: float = 0.0
    win_rate_change: float = 0.0
    baseline: Dict[str, Any] = field(default_factory=dict)
    current: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "should_rollback": self.should_rollback,
            "reason": self.reason,
            "pnl_change_pct": self.pnl_change_pct,
            "win_rate_change": self.win_rate_change,
            "baseline": self.baseline,
            "current": self.current,
        }


class ParameterRollbackGuard:
    """参数变更验证与自动回滚守卫（线程安全）"""

    def __init__(
        self,
        validation_window_seconds: float = DEFAULT_VALIDATION_WINDOW_SECONDS,
        pnl_drop_threshold: float = DEFAULT_PNL_DROP_THRESHOLD,
        win_rate_drop_threshold: float = DEFAULT_WIN_RATE_DROP_THRESHOLD,
        min_trades: int = DEFAULT_MIN_TRADES,
    ):
        self._validation_window_seconds = validation_window_seconds
        self._pnl_drop_threshold = pnl_drop_threshold
        self._win_rate_drop_threshold = win_rate_drop_threshold
        self._min_trades = min_trades
        # strategy_name -> 验证上下文
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    # ============================================================
    # 生命周期
    # ============================================================

    def begin_validation(
        self,
        strategy_name: str,
        baseline_metrics: Optional[Dict[str, Any]] = None,
        reason: str = "",
        new_config: Optional[Dict[str, Any]] = None,
        rollback_target_version: Optional[int] = None,
    ) -> None:
        """参数变更后开启一次验证，记录基线指标。

        Args:
            strategy_name: 策略名称
            baseline_metrics: 变更前基线指标，至少应含 ``pnl`` 与 ``win_rate``
            reason: 变更原因
            new_config: 变更后的配置（用于审计留痕）
            rollback_target_version: 变更前的配置快照版本号（供回滚定位），
                由 StrategyManager 传入，本守卫只做透明存储不做语义解释。
        """
        with self._lock:
            self._pending[strategy_name] = {
                "started_at": time.time(),
                "reason": reason,
                "new_config": dict(new_config or {}),
                "baseline": dict(baseline_metrics or {}),
                "rollback_target_version": rollback_target_version,
            }
        logger.info(
            f"ParameterRollbackGuard: 开始验证 '{strategy_name}' "
            f"(window={self._validation_window_seconds:.0f}s, reason={reason})"
        )

    def cancel_validation(self, strategy_name: str) -> bool:
        """取消未完成的验证（如策略被停止/退出）。"""
        with self._lock:
            existed = strategy_name in self._pending
            self._pending.pop(strategy_name, None)
        if existed:
            logger.info(f"ParameterRollbackGuard: 取消 '{strategy_name}' 的验证")
        return existed

    # ============================================================
    # 决策
    # ============================================================

    def check(
        self,
        strategy_name: str,
        current_metrics: Optional[Dict[str, Any]] = None,
    ) -> RollbackDecision:
        """检查某策略的验证是否到期并应回滚。

        Args:
            strategy_name: 策略名称
            current_metrics: 当前指标，至少应含 ``pnl`` 与 ``win_rate``

        Returns:
            RollbackDecision：若验证未到期返回 ``reason="validation_in_progress"``；
            若样本不足返回 ``reason="insufficient_sample"``；否则返回明确回滚/保留决策。
        """
        current = dict(current_metrics or {})
        with self._lock:
            ctx = self._pending.get(strategy_name)
            if ctx is None:
                return RollbackDecision(strategy_name, False, reason="no_pending_validation")

            elapsed = time.time() - ctx["started_at"]
            if elapsed < self._validation_window_seconds:
                return RollbackDecision(strategy_name, False, reason="validation_in_progress")

            baseline = ctx["baseline"]
            # 到期后即消费该验证（无论结论如何都移除 pending）
            self._pending.pop(strategy_name, None)

        decision = self._evaluate(strategy_name, baseline, current)
        if decision.should_rollback:
            logger.warning(
                f"ParameterRollbackGuard: '{strategy_name}' 验证失败 → 建议回滚 "
                f"({decision.reason}; pnl_change={decision.pnl_change_pct:.4f}, "
                f"win_rate_change={decision.win_rate_change:.4f})"
            )
        else:
            logger.info(
                f"ParameterRollbackGuard: '{strategy_name}' 验证通过 → 保留新参数 "
                f"(pnl_change={decision.pnl_change_pct:.4f}, win_rate_change={decision.win_rate_change:.4f})"
            )
        return decision

    def _evaluate(
        self,
        strategy_name: str,
        baseline: Dict[str, Any],
        current: Dict[str, Any],
    ) -> RollbackDecision:
        """根据基线/当前指标计算回滚决策（不含锁）。"""
        current_trades = _as_int(current.get("trades"))
        if current_trades < self._min_trades:
            return RollbackDecision(
                strategy_name, False,
                reason="insufficient_sample",
                baseline=baseline, current=current,
            )

        base_pnl = _as_float(baseline.get("pnl"))
        curr_pnl = _as_float(current.get("pnl"))
        base_wr = _as_float(baseline.get("win_rate"))
        curr_wr = _as_float(current.get("win_rate"))

        # PnL 变化：相对基线（基线为 0 时退化为绝对变化，避免除零）
        pnl_change = _safe_change_pct(curr_pnl, base_pnl)
        win_rate_change = curr_wr - base_wr

        reasons = []
        if pnl_change <= -self._pnl_drop_threshold:
            reasons.append(f"pnl_dropped_by_{abs(pnl_change):.0%}")
        if win_rate_change <= -self._win_rate_drop_threshold:
            reasons.append(f"win_rate_dropped_by_{abs(win_rate_change):.0%}")

        should_rollback = bool(reasons)
        return RollbackDecision(
            strategy_name=strategy_name,
            should_rollback=should_rollback,
            reason=";".join(reasons) if reasons else "no_degradation",
            pnl_change_pct=pnl_change,
            win_rate_change=win_rate_change,
            baseline=baseline,
            current=current,
        )

    # ============================================================
    # 查询
    # ============================================================

    def get_pending(self) -> Dict[str, Any]:
        """获取所有进行中的验证上下文（只读快照）。"""
        with self._lock:
            return {k: dict(v) for k, v in self._pending.items()}

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


# ============================================================
# 工具函数
# ============================================================

def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_change_pct(current: float, baseline: float) -> float:
    """计算 current 相对 baseline 的变化比例（基线为 0 时退化为绝对差）。"""
    if baseline == 0:
        return current - baseline
    return (current - baseline) / abs(baseline)
