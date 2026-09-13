"""
决策执行器，按优先级与批量模式执行交易决策并记录审计日志。
"""
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger
import asyncio
import time


class DecisionExecutor:
    def __init__(self, config: Dict[str, Any], order_executor=None, decision_validator=None, confidence_calibrator=None):
        self._config = config
        self._order_executor = order_executor
        self._decision_validator = decision_validator
        self._confidence_calibrator = confidence_calibrator
        self._executed_decisions = []
        self._max_history_size = config.get("max_history_size", 1000)
        self._priority_map = {"stop_loss": 0, "scalping": 1, "trend": 2, "grid": 3, "arbitrage": 4}
        self._pre_execution_hooks: list = []
        self._post_execution_hooks: list = []
        self._execution_semaphore = asyncio.Semaphore(config.get("max_concurrent_executions", 10))
        self._execution_timeout = config.get("execution_timeout", 30)
        self._batch_mode = config.get("batch_mode", False)
        self._batch_queue: list = []
        self._batch_max_size = config.get("batch_max_size", 10)
        self._batch_flush_interval = config.get("batch_flush_interval", 0.5)
        self._execution_stats = {"total": 0, "success": 0, "failed": 0, "batched": 0}
        self._audit_log: deque = deque(maxlen=200)

    async def execute(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        """执行决策（支持批量和优先级）"""
        start_time = time.time()
        
        # 批处理模式
        if self._batch_mode:
            self._batch_queue.append(decision)
            if len(self._batch_queue) >= self._batch_max_size:
                return await self._flush_batch()
            return {"status": "batched", "batch_size": len(self._batch_queue)}
        
        return await self._execute_single(decision, start_time)

    async def _execute_single(self, decision: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """单个决策执行"""
        decision_id = decision.get("data", {}).get("order_id", str(int(time.time()*1000)))
        
        async with self._execution_semaphore:
            try:
                # 预执行钩子
                for hook in self._pre_execution_hooks:
                    try:
                        modified = hook(decision)
                        if modified is not None:
                            decision = modified
                    except Exception as e:
                        logger.warning(f"Pre-execution hook failed: {e}")
                
                # 验证
                if self._decision_validator:
                    from decision.decision_coordinator import Decision, DecisionType
                    import uuid
                    decision_obj = Decision(
                        decision_id=decision.get("decision_id") or str(uuid.uuid4()),
                        decision_type=DecisionType(decision.get("decision_type", "signal")),
                        data=decision.get("data", {}),
                        confidence=decision.get("confidence", 0.5),
                        source=decision.get("source", "unknown"),
                    )
                    validation_result, validation_errors = await asyncio.wait_for(
                        self._decision_validator.validate(decision_obj),
                        timeout=self._execution_timeout
                    )
                    
                    if validation_result.value == "invalid":
                        self._record_decision(decision, {"status": "failed", "reason": "validation"})
                        self._audit_decision(decision, "rejected", time.time() - start_time)
                        return {"status": "failed", "decision_id": decision_id, "reason": "Validation failed"}
                    elif validation_result.value == "warning":
                        logger.info(f"Decision validation warnings: {[e.code for e in validation_errors]}")
                
                # 置信度校准
                if self._confidence_calibrator:
                    calibrated_confidence, _ = self._confidence_calibrator.calibrate(decision.get("confidence", 0.5))
                    decision["confidence"] = calibrated_confidence
                
                # 执行订单
                if self._order_executor:
                    result = await asyncio.wait_for(
                        self._order_executor.handle_signal(decision.get("data", decision)),
                        timeout=self._execution_timeout
                    )
                    elapsed = time.time() - start_time
                    self._execution_stats["total"] += 1
                    self._execution_stats["success"] += 1
                    self._record_decision(decision, result)
                    self._audit_decision(decision, "executed", elapsed, result)
                    self._record_latency("decision_execute", elapsed * 1000, {"status": "success"})
                    return {"status": "success", "decision_id": decision_id, "order_result": result, "elapsed_ms": elapsed * 1000}
                
                return {"status": "success", "decision_id": decision_id, "reason": "No order executor configured"}
                
            except asyncio.TimeoutError:
                self._execution_stats["total"] += 1
                self._execution_stats["failed"] += 1
                self._record_decision(decision, {"status": "timeout"})
                self._audit_decision(decision, "timeout", time.time() - start_time)
                return {"status": "timeout", "decision_id": decision_id, "reason": f"Execution timeout after {self._execution_timeout}s"}
            except Exception as e:
                self._execution_stats["total"] += 1
                self._execution_stats["failed"] += 1
                self._record_decision(decision, {"status": "failed", "error": str(e)})
                self._audit_decision(decision, "error", time.time() - start_time, str(e))
                return {"status": "failed", "decision_id": decision_id, "reason": str(e)}

    async def execute_batch(self, decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量执行决策（按优先级排序）"""
        sorted_decisions = sorted(
            decisions,
            key=lambda d: self._priority_map.get(
                d.get("data", {}).get("strategy_name", "grid"), 3
            )
        )
        results = []
        for decision in sorted_decisions:
            result = await self._execute_single(decision, time.time())
            results.append(result)
        self._execution_stats["batched"] += 1
        return results

    async def _flush_batch(self) -> Dict[str, Any]:
        """刷新批量队列"""
        if not self._batch_queue:
            return {"status": "empty", "reason": "No pending decisions"}
        batch = self._batch_queue.copy()
        self._batch_queue.clear()
        results = await self.execute_batch(batch)
        return {"status": "batch_executed", "count": len(results), "results": results}

    def _audit_decision(self, decision: Dict[str, Any], outcome: str, elapsed: float, detail: Any = None):
        """审计日志"""
        self._audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "strategy": decision.get("data", {}).get("strategy_name", ""),
            "symbol": decision.get("data", {}).get("symbol", ""),
            "confidence": decision.get("confidence", 0),
            "outcome": outcome,
            "elapsed_ms": elapsed * 1000,
            "detail": str(detail)[:200] if detail else None,
        })

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        stats = dict(self._execution_stats)
        if stats["total"] > 0:
            stats["success_rate"] = stats["success"] / stats["total"]
        stats["batch_queue_size"] = len(self._batch_queue)
        return stats

    def get_audit_log(self, limit: int = 50) -> list:
        """获取审计日志"""
        return list(self._audit_log)[-limit:]

    def _record_decision(self, decision: Dict[str, Any], result: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "decision": decision,
            "result": result,
        }
        self._executed_decisions.append(entry)
        if len(self._executed_decisions) > self._max_history_size:
            self._executed_decisions = self._executed_decisions[-self._max_history_size:]

    def get_executed_decisions(self, limit: int = 100) -> list:
        return self._executed_decisions[-limit:]

    def set_order_executor(self, order_executor):
        self._order_executor = order_executor
        logger.info("Order executor set for decision execution")

    def set_decision_validator(self, validator):
        self._decision_validator = validator
        logger.info("Decision validator set")

    def set_confidence_calibrator(self, calibrator):
        self._confidence_calibrator = calibrator
        logger.info("Confidence calibrator set")

    def add_pre_execution_hook(self, hook: callable):
        """添加预执行钩子，返回修改后的decision或None"""
        self._pre_execution_hooks.append(hook)

    def add_post_execution_hook(self, hook: callable):
        """添加后执行钩子"""
        self._post_execution_hooks.append(hook)
