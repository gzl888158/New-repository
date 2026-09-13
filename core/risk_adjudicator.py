"""P2 企业级升级：独立风控裁决器（RiskAdjudicator）

对标 Jane Street / DE Shaw / Jump 的「独立风控裁决层」范式：
- 所有下单动作（开仓/平仓/减仓/止损/止盈/条件单）必须经此裁决器裁决，是唯一 choke point
- 每次裁决生成（或复用）全链路 traceID，回写到 signal，实现「信号→裁决→订单→成交→平仓」同源追踪
- 裁决结果（PASS/REJECT/REDUCE/CLOSE_ALL/FREEZE）发布到事件总线 → EventStore 持久化，可重放审计
- fail-closed：裁决器自身异常时一律拒绝，绝不静默放行

与 RiskGate 的关系：RiskGate 是「六层检查引擎」，RiskAdjudicator 是其「独立裁决外壳」——
在复用 RiskGate 六层逻辑（L0 Kill Switch / L5 紧急熔断 / L4 单日 / L1 事前 / L2 事中 / L3 持仓）
的基础上，补齐 traceID 贯穿 + 裁决事件溯源 + 统一裁决入口三大企业级能力。

设计约束：本模块依赖 core.risk_gate / core.event_id / core.unified_layer，零执行层依赖，可独立单测。
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from core.event_id import EventIDGenerator
from core.risk_gate import (
    RiskAction,
    RiskCheckResult,
    RiskGate,
    RiskGateResult,
    RiskLayer,
)
from core.unified_layer import Event, EventType


@dataclass
class AdjudicationResult:
    """独立风控裁决结果（带全链路 traceID）。"""
    trace_id: str
    passed: bool
    action: RiskAction
    blocked_layer: Optional[RiskLayer]
    results: List[RiskCheckResult] = field(default_factory=list)
    summary: str = ""
    is_close: bool = False
    symbol: str = ""
    adjudicated_at: datetime = field(default_factory=datetime.now)
    elapsed_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "passed": self.passed,
            "action": self.action.value,
            "blocked_layer": self.blocked_layer.value if self.blocked_layer else None,
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "is_close": self.is_close,
            "symbol": self.symbol,
            "adjudicated_at": self.adjudicated_at.isoformat(),
            "elapsed_ms": round(self.elapsed_ms, 3),
        }


class RiskAdjudicator:
    """独立风控裁决器：唯一下单裁决入口（choke point）。

    用法：
        decision = adjudicator.adjudicate(signal)
        if not decision.passed:
            return  # 裁决拒绝，丢弃信号
        # signal["trace_id"] 已被回写，继续下单（全链路 traceID 同源）
    """

    def __init__(self, risk_gate: RiskGate, event_bus=None):
        if risk_gate is None:
            raise ValueError("RiskAdjudicator 必须注入 RiskGate")
        self._risk_gate = risk_gate
        self._event_bus = event_bus
        self._id_gen = EventIDGenerator.get_instance()
        logger.info("RiskAdjudicator initialized (independent risk adjudication choke point)")

    def set_event_bus(self, event_bus) -> None:
        """注入事件总线：注入后裁决事件将发布到 EventStore 持久化（可重放审计）。"""
        self._event_bus = event_bus

    def adjudicate(
        self,
        signal: Dict[str, Any],
        market_data: Dict[str, Any] = None,
        is_close: bool = False,
    ) -> AdjudicationResult:
        """执行独立裁决：traceID 贯穿 + 六层风控 + 裁决事件发布 + fail-closed。

        Args:
            signal: 交易信号 dict（会回写 trace_id，若缺失则生成并回写）
            market_data: 市场数据（L2 事中风控用）
            is_close: 是否为平仓/减仓信号

        Returns:
            AdjudicationResult 带全链路 traceID 的裁决结果
        """
        # 1. traceID 贯穿：复用已有 trace_id（信号→裁决→订单全链路同源），否则生成并回写
        if not isinstance(signal, dict):
            signal = dict(signal or {})
        trace_id = signal.get("trace_id") or self._id_gen.generate()
        signal["trace_id"] = trace_id

        symbol = signal.get("symbol", "")
        start = time.perf_counter()

        # 2. 六层风控裁决（复用 RiskGate 六层引擎）
        try:
            gate_result = self._risk_gate.validate(signal, market_data, is_close)
        except Exception as e:
            # fail-closed：裁决器异常一律拒绝，绝不静默放行
            logger.error(f"RiskAdjudicator exception (fail-closed REJECT): {e}")
            gate_result = RiskGateResult(
                passed=False,
                action=RiskAction.REJECT,
                blocked_layer=None,
                results=[],
                summary=f"裁决器异常(fail-closed): {e}",
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # 3. 发布裁决事件（带 traceID，无论通过与否，均落盘供重放审计）
        self._publish_adjudication(trace_id, gate_result, signal, is_close, elapsed_ms)

        # 4. 返回带 traceID 的裁决结果
        return AdjudicationResult(
            trace_id=trace_id,
            passed=gate_result.passed,
            action=gate_result.action,
            blocked_layer=gate_result.blocked_layer,
            results=gate_result.results,
            summary=gate_result.summary,
            is_close=is_close,
            symbol=symbol,
            elapsed_ms=elapsed_ms,
        )

    def _publish_adjudication(
        self,
        trace_id: str,
        gate_result: RiskGateResult,
        signal: Dict[str, Any],
        is_close: bool,
        elapsed_ms: float,
    ) -> None:
        if self._event_bus is None:
            return
        try:
            payload = {
                "trace_id": trace_id,
                "passed": gate_result.passed,
                "action": gate_result.action.value,
                "blocked_layer": gate_result.blocked_layer.value if gate_result.blocked_layer else None,
                "is_close": is_close,
                "symbol": signal.get("symbol", ""),
                "strategy": signal.get("strategy_name", ""),
                "signal_type": signal.get("signal_type", ""),
                "reason": gate_result.summary,
                "elapsed_ms": round(elapsed_ms, 3),
            }
            self._event_bus.publish_sync(Event(EventType.RISK_ADJUDICATED, payload))
        except Exception as e:
            logger.debug(f"RiskAdjudicator event publish failed (non-blocking): {e}")