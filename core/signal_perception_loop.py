"""
感知侧闭环协调器 SignalPerceptionCoordinator
==============================================

把分散在 SignalProcessor._process_signal 中的「市场识别硬门控（RegimeGate）」与
「信号质量软评分（SignalQualityEngine）」串成一条带反馈的感知闭环，而不重写它们
内部的算法。

闭环流程（一个信号 = 一次 perceive）：
    感知 Perceive  → 门控 Gate  → 质量 Quality  → 决策 Decide  → 反馈 Reflect

设计约束：
  - 纯编排：只调用 regime_gate.evaluate / quality_engine.is_signal_acceptable 既有接口，
    不复制门控规则与质量因子算法。
  - fail-closed：门控/质量任一环节未注入或异常时拒绝信号（degraded → reject），
    宁可误杀也不放行未经充分感知的信号；异常与拒绝次数可观测。
  - JSON 安全：所有统计/结果经 utils.helpers 的 safe_* 清洗，json.dumps 无 NaN/Inf。
  - 可观测：维护 regime × strategy × quality 关联统计，供 Dashboard / AGI 协调器
    观察「门控拒绝率 / 质量分布」的感知侧闭环效果。
"""
import copy
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from utils.helpers import safe_float, safe_finite, safe_int


@dataclass
class PerceptionResult:
    """一次感知闭环的统一决策结果（JSON 安全）。"""
    decision: str = "degraded"          # pass / reject_gate / reject_quality / degraded
    regime: str = "unknown"
    gate_allowed: bool = True
    gate_reason: str = ""
    quality_score: float = 0.5
    quality_grade: str = "degraded"
    quality_acceptable: bool = True
    quality_breakdown: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    feedback: Dict[str, Any] = field(default_factory=dict)


class SignalPerceptionCoordinator:
    """统一感知侧协调器：编排 regime_gate + quality_engine 形成信号级感知闭环。"""

    def __init__(
        self,
        regime_engine=None,
        regime_gate=None,
        quality_engine=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._regime_engine = regime_engine
        self._regime_gate = regime_gate
        self._quality_engine = quality_engine
        self._config = dict(config or {})

        self._stats: Dict[str, Any] = {
            "total_perceived": 0,
            "passed": 0,
            "reject_gate": 0,
            "reject_quality": 0,
            "degraded": 0,
            "errors": 0,
            "by_regime": {},
            "by_strategy": {},
            "quality_distribution": {},
        }

        logger.info(
            f"SignalPerceptionCoordinator initialized: "
            f"regime_engine={'on' if regime_engine else 'off'}, "
            f"regime_gate={'on' if regime_gate else 'off'}, "
            f"quality_engine={'on' if quality_engine else 'off'}"
        )

    # ─────────────────────────────────────────────────────────────
    # 感知闭环入口
    # ─────────────────────────────────────────────────────────────

    def perceive(self, signal_dict: Dict[str, Any]) -> PerceptionResult:
        """对单个信号执行一次感知闭环，返回统一决策结果。

        流程：感知 regime → 门控 → 质量 → 决策 → 反馈（统计）。
        不修改传入的 signal_dict。
        """
        sig = dict(signal_dict or {})
        symbol = str(sig.get("symbol", ""))
        strategy = str(sig.get("strategy_name", ""))
        signal_type = str(sig.get("signal_type", ""))
        direction = str(sig.get("direction", ""))
        confidence = safe_float(sig.get("confidence"), 0.5)

        # 1. 感知 Perceive：获取当前 regime
        regime = self._perceive_regime()

        # 2. 门控 Gate：RegimeGate 硬过滤（未注入放行；异常 fail-closed 拒绝）
        gate_allowed, gate_reason, gate_error = self._evaluate_gate(
            symbol, strategy, signal_type, direction, confidence
        )

        # 3. 质量 Quality：SignalQualityEngine 软评分（未注入/异常降级）
        quality = self._evaluate_quality(sig)

        # 4. 决策 Decide：融合门控 + 质量
        decision, reason = self._decide(gate_allowed, gate_reason, gate_error, quality)

        # 5. 反馈 Reflect：累加关联统计（regime × strategy × quality）
        feedback = self._reflect(regime, symbol, strategy, decision, quality)

        result = PerceptionResult(
            decision=decision,
            regime=regime,
            gate_allowed=gate_allowed,
            gate_reason=gate_reason,
            quality_score=safe_finite(quality.get("overall_score"), 0.5),
            quality_grade=str(quality.get("quality", "degraded")),
            quality_acceptable=bool(quality.get("acceptable", True)),
            quality_breakdown=copy.deepcopy(quality),
            reason=reason,
            feedback=feedback,
        )
        return result

    # ─────────────────────────────────────────────────────────────
    # 1. 感知 Perceive
    # ─────────────────────────────────────────────────────────────

    def _perceive_regime(self) -> str:
        if self._regime_engine is None:
            return "unknown"
        try:
            info = self._regime_engine.get_regime()
            return str((info or {}).get("regime", "unknown"))
        except Exception as e:
            logger.debug(f"[Perception] regime query failed: {e}")
            return "unknown"

    # ─────────────────────────────────────────────────────────────
    # 2. 门控 Gate
    # ─────────────────────────────────────────────────────────────

    def _evaluate_gate(self, symbol, strategy, signal_type, direction, confidence):
        """调用 RegimeGate.evaluate，返回 (allowed, reason, error)。

        未注入或异常时均 fail-closed 拒绝。
        """
        if self._regime_gate is None:
            return False, "no regime gate injected", False
        try:
            result = self._regime_gate.evaluate(
                symbol,
                strategy,
                signal_type=signal_type,
                direction=direction,
                confidence=confidence,
            )
            allowed = bool(getattr(result, "allowed", True))
            reason = str(getattr(result, "reason", ""))
            return allowed, reason, False
        except Exception as e:
            self._stats["errors"] = safe_int(self._stats.get("errors"), 0) + 1
            logger.warning(f"[Perception] RegimeGate evaluate failed (fail-closed): {e}")
            return False, f"gate_error:{e}", True

    # ─────────────────────────────────────────────────────────────
    # 3. 质量 Quality
    # ─────────────────────────────────────────────────────────────

    def _evaluate_quality(self, sig: Dict[str, Any]) -> Dict[str, Any]:
        """调用 SignalQualityEngine 评估，返回安全 dict（含 acceptable/overall_score/quality）。

        quality_engine 未注入或异常时返回 fail-closed dict（acceptable=False）。
        """
        if self._quality_engine is None:
            return {
                "acceptable": False,
                "overall_score": 0.0,
                "quality": "degraded",
                "available": False,
                "reason": "no quality engine",
            }
        try:
            acceptable, breakdown = self._quality_engine.is_signal_acceptable(sig)
            if not isinstance(breakdown, dict):
                breakdown = {}
            out = copy.deepcopy(breakdown)
            out["acceptable"] = bool(acceptable)
            out["available"] = True
            return out
        except Exception as e:
            self._stats["errors"] = safe_int(self._stats.get("errors"), 0) + 1
            logger.warning(f"[Perception] SignalQualityEngine evaluate failed (fail-closed): {e}")
            return {
                "acceptable": False,
                "overall_score": 0.0,
                "quality": "degraded",
                "available": False,
                "reason": f"quality_error:{e}",
            }

    # ─────────────────────────────────────────────────────────────
    # 4. 决策 Decide
    # ─────────────────────────────────────────────────────────────

    def _decide(self, gate_allowed, gate_reason, gate_error, quality) -> tuple:
        """融合门控与质量，产出统一决策。硬门控优先于软评分。"""
        # 门控硬拒绝（源头丢废信号）
        if not gate_allowed:
            return "reject_gate", gate_reason or "regime gate rejected"

        # 质量软拒绝（评分低于阈值）
        if quality.get("available") and not quality.get("acceptable"):
            score = safe_float(quality.get("overall_score"), 0.0)
            return "reject_quality", f"signal quality below threshold (score={score:.2f})"

        # 门控或质量其一不可用（未注入/异常）→ fail-closed 拒绝
        gate_unavailable = (self._regime_gate is None) or gate_error
        quality_unavailable = (self._quality_engine is None) or not quality.get("available")
        if gate_unavailable or quality_unavailable:
            reasons = []
            if gate_unavailable:
                reasons.append("gate_unavailable" if self._regime_gate is None else "gate_error")
            if quality_unavailable:
                reasons.append("quality_unavailable" if self._quality_engine is None else "quality_error")
            return "reject_quality", f"perception incomplete: {','.join(reasons)}"

        return "pass", "perception passed"

    # ─────────────────────────────────────────────────────────────
    # 5. 反馈 Reflect
    # ─────────────────────────────────────────────────────────────

    def _reflect(self, regime, symbol, strategy, decision, quality) -> Dict[str, Any]:
        """累加关联统计，供 Dashboard / AGI 观察感知侧闭环效果。"""
        try:
            s = self._stats
            s["total_perceived"] = safe_int(s.get("total_perceived"), 0) + 1

            if decision == "pass":
                s["passed"] = safe_int(s.get("passed"), 0) + 1
            elif decision in ("reject_gate", "reject_quality", "degraded"):
                s[decision] = safe_int(s.get(decision), 0) + 1

            # by_regime
            rb = s["by_regime"].setdefault(regime, {"total": 0, "passed": 0, "reject_gate": 0,
                                                    "reject_quality": 0, "degraded": 0})
            rb["total"] = safe_int(rb.get("total"), 0) + 1
            if decision in ("passed", "pass", "reject_gate", "reject_quality", "degraded"):
                key = "passed" if decision == "pass" else decision
                rb[key] = safe_int(rb.get(key), 0) + 1

            # by_strategy
            sb = s["by_strategy"].setdefault(strategy, {"total": 0, "passed": 0, "reject_gate": 0,
                                                        "reject_quality": 0, "degraded": 0})
            sb["total"] = safe_int(sb.get("total"), 0) + 1
            if decision in ("pass", "reject_gate", "reject_quality", "degraded"):
                key = "passed" if decision == "pass" else decision
                sb[key] = safe_int(sb.get(key), 0) + 1

            # quality_distribution
            grade = str(quality.get("quality", "degraded"))
            qd = s["quality_distribution"]
            qd[grade] = safe_int(qd.get(grade), 0) + 1
        except Exception as e:
            logger.debug(f"[Perception] reflect stats failed: {e}")

        return {
            "regime": regime,
            "symbol": symbol,
            "strategy": strategy,
            "decision": decision,
        }

    # ─────────────────────────────────────────────────────────────
    # 可观测
    # ─────────────────────────────────────────────────────────────

    def get_stats(self) -> Dict[str, Any]:
        """返回感知侧闭环统计（深拷贝，不暴露内部引用）。"""
        return copy.deepcopy(self._stats)


__all__ = ["SignalPerceptionCoordinator", "PerceptionResult"]
