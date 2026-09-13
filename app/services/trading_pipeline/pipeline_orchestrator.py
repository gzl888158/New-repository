"""
交易流水线编排器，按阶段串联信号接收、决策、下单与执行监控。
"""
from enum import Enum
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from loguru import logger
import asyncio
import heapq
import time


class PipelineStage(Enum):
    SIGNAL_RECEIVE = "signal_receive"
    SIGNAL_VALIDATION = "signal_validation"
    DECISION_MAKING = "decision_making"
    ORDER_GENERATION = "order_generation"
    ORDER_PLACEMENT = "order_placement"
    EXECUTION_MONITORING = "execution_monitoring"
    SETTLEMENT = "settlement"
    TIMEOUT_HANDLING = "timeout_handling"  # 超时处理


class PipelineStatus(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class PipelineContext:
    def __init__(self):
        self.signal = None
        self.validated_signal = None
        self.decision = None
        self.order = None
        self.execution_result = None
        self.settlement_result = None
        self.metrics = {}
        self.start_time = datetime.now()
        self.stage_times = {}
        self.priority = 0  # 优先级(0=最高)
        self.stage_timeouts = {}  # 每阶段超时秒数
        self.created_at = datetime.now()

    def set_stage_start(self, stage: PipelineStage):
        self.stage_times[stage.value] = {"start": datetime.now()}

    def set_stage_end(self, stage: PipelineStage):
        if stage.value in self.stage_times:
            self.stage_times[stage.value]["end"] = datetime.now()
            self.stage_times[stage.value]["duration_ms"] = (
                self.stage_times[stage.value]["end"] - 
                self.stage_times[stage.value]["start"]
            ).total_seconds() * 1000

    def get_stage_duration(self, stage: PipelineStage) -> float:
        return self.stage_times.get(stage.value, {}).get("duration_ms", 0.0)


class PipelineOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._status = PipelineStatus.STOPPED
        self._stages: Dict[str, List[Callable]] = {}
        self._contexts: Dict[str, PipelineContext] = {}
        self._active_contexts = 0
        self._max_concurrent = config.get("max_concurrent_pipelines", 10)
        self._lock = asyncio.Lock()
        self._priority_queue = []  # 优先级队列
        self._stage_timeouts = config.get("pipeline", {}).get("stage_timeouts", {
            PipelineStage.SIGNAL_RECEIVE.value: 5,
            PipelineStage.SIGNAL_VALIDATION.value: 10,
            PipelineStage.DECISION_MAKING.value: 15,
            PipelineStage.ORDER_GENERATION.value: 10,
            PipelineStage.ORDER_PLACEMENT.value: 30,
            PipelineStage.EXECUTION_MONITORING.value: 60,
            PipelineStage.SETTLEMENT.value: 30,
            PipelineStage.TIMEOUT_HANDLING.value: 15,
        })
        self._pipeline_timeout = config.get("pipeline", {}).get("total_timeout", 120)
        self._metrics = {
            "total_pipelines": 0,
            "success_pipelines": 0,
            "failed_pipelines": 0,
            "stage_latencies": {},
            "avg_latency_ms": 0.0,
            "total_rejected": 0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
            "all_latencies": [],
            "pipeline_latencies": [],
            "priority_queue_depth": 0,
            "timeout_count": 0,
        }

        for stage in PipelineStage:
            self._stages[stage.value] = []

    def register_stage_handler(self, stage: PipelineStage, handler: Callable):
        self._stages[stage.value].append(handler)
        logger.info(f"Registered handler for stage: {stage.value}")

    def register_stage_handlers(self, stage_handlers: Dict[PipelineStage, List[Callable]]):
        for stage, handlers in stage_handlers.items():
            for handler in handlers:
                self.register_stage_handler(stage, handler)

    async def start(self):
        self._status = PipelineStatus.RUNNING
        logger.info("Pipeline orchestrator started")

    async def stop(self):
        self._status = PipelineStatus.STOPPED
        logger.info("Pipeline orchestrator stopped")

    async def pause(self):
        self._status = PipelineStatus.PAUSED
        logger.info("Pipeline orchestrator paused")

    async def resume(self):
        self._status = PipelineStatus.RUNNING
        logger.info("Pipeline orchestrator resumed")

    def get_status(self) -> str:
        return self._status.value

    async def execute_pipeline_priority(self, signal_data: Dict[str, Any], priority: int = 5) -> Dict[str, Any]:
        """优先级管道执行"""
        if self._status != PipelineStatus.RUNNING:
            return {"status": "rejected", "reason": f"Pipeline is {self._status.value}"}
        
        if self._active_contexts >= self._max_concurrent:
            # 加入优先级队列
            heapq.heappush(self._priority_queue, (priority, time.time(), id(signal_data), signal_data))
            self._metrics["priority_queue_depth"] = len(self._priority_queue)
            return {"status": "queued", "priority": priority, "queue_position": len(self._priority_queue)}
        
        return await self.execute_pipeline(signal_data, priority=priority)

    async def execute_pipeline(self, signal_data: Dict[str, Any], priority: int = 5) -> Dict[str, Any]:
        if self._status != PipelineStatus.RUNNING:
            self._metrics["total_rejected"] += 1
            return {"status": "rejected", "reason": f"Pipeline is {self._status.value}"}

        async with self._lock:
            if self._active_contexts >= self._max_concurrent:
                self._metrics["total_rejected"] += 1
                return {"status": "rejected", "reason": "Max concurrent pipelines reached"}
            self._active_contexts += 1

        try:
            return await self._run_pipeline_context(signal_data, priority)
        finally:
            await self._release_slot()

    async def _run_pipeline_context(self, signal_data: Dict[str, Any], priority: int) -> Dict[str, Any]:
        context_id = f"pipeline_{datetime.now().timestamp():.0f}_{id(signal_data)}"
        context = PipelineContext()
        context.priority = priority
        context.stage_timeouts = self._stage_timeouts
        self._contexts[context_id] = context

        try:
            return await self._run_pipeline(context_id, context, signal_data)
        finally:
            if context_id in self._contexts:
                del self._contexts[context_id]

    async def _release_slot(self):
        async with self._lock:
            self._active_contexts -= 1
        # 有排队信号时，后台消费优先级队列
        if self._priority_queue:
            asyncio.create_task(self._process_priority_queue())

    async def _run_pipeline(self, context_id: str, context: PipelineContext, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Starting pipeline execution: {context_id}")
        try:
            pipeline_coro = self._run_pipeline_inner(context_id, context, signal_data)
            result = await asyncio.wait_for(pipeline_coro, timeout=self._pipeline_timeout)
            return result
        except asyncio.TimeoutError:
            self._metrics["failed_pipelines"] += 1
            self._metrics["timeout_count"] += 1
            logger.error(f"Pipeline {context_id} timed out after {self._pipeline_timeout}s")
            # 触发 TIMEOUT_HANDLING 阶段处理器（真实超时处理而非仅返回 status）
            await self._run_timeout_handlers(context, signal_data)
            return {"status": "timeout", "context_id": context_id, "error": f"Pipeline timeout after {self._pipeline_timeout}s"}

    async def _run_pipeline_inner(self, context_id: str, context: PipelineContext, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        self._metrics["total_pipelines"] += 1

        try:
            context.signal = signal_data

            for stage in PipelineStage:
                # TIMEOUT_HANDLING 为特殊阶段，仅在整体超时时由 _run_pipeline 触发，不参与正常串行循环
                if stage == PipelineStage.TIMEOUT_HANDLING:
                    continue
                context.set_stage_start(stage)
                
                if not self._stages.get(stage.value):
                    logger.debug(f"No handlers for stage: {stage.value}")
                    context.set_stage_end(stage)
                    continue

                for handler in self._stages[stage.value]:
                    try:
                        result = await handler(context)
                        if result is not None:
                            if isinstance(result, dict):
                                for key, value in result.items():
                                    setattr(context, key, value)
                    except Exception as e:
                        logger.error(f"Handler failed at stage {stage.value}: {e}")
                        self._metrics["failed_pipelines"] += 1
                        self._update_stage_latency(stage, context.get_stage_duration(stage))
                        return {
                            "status": "failed",
                            "stage": stage.value,
                            "error": str(e),
                            "latencies": context.stage_times,
                        }

                context.set_stage_end(stage)
                self._update_stage_latency(stage, context.get_stage_duration(stage))

            self._metrics["success_pipelines"] += 1
            total_latency = (datetime.now() - context.start_time).total_seconds() * 1000
            self._record_latency("pipeline_total", total_latency, {"status": "success"})
            self._metrics["avg_latency_ms"] = (
                self._metrics["avg_latency_ms"] * (self._metrics["total_pipelines"] - 1) + total_latency
            ) / self._metrics["total_pipelines"]

            # p95/p99 基于整条流水线的端到端总延迟（而非各阶段耗时混合）
            pipeline_lat = self._metrics.setdefault("pipeline_latencies", [])
            pipeline_lat.append(total_latency)
            if len(pipeline_lat) > 1000:
                pipeline_lat[:] = pipeline_lat[-1000:]
            sorted_lat = sorted(pipeline_lat)
            n = len(sorted_lat)
            self._metrics["p95_latency_ms"] = sorted_lat[int(n * 0.95)]
            self._metrics["p99_latency_ms"] = sorted_lat[int(n * 0.99)] if n > 100 else sorted_lat[-1]

            return {
                "status": "success",
                "context_id": context_id,
                "decision": context.decision,
                "order": context.order,
                "execution_result": context.execution_result,
                "latencies": context.stage_times,
                "total_latency_ms": total_latency,
            }

        except Exception as e:
            self._handle_exception(e, module="PipelineOrchestrator", function="_run_pipeline_inner", severity="high", category="pipeline")
            self._metrics["failed_pipelines"] += 1
            return {"status": "error", "error": str(e)}

    def _update_stage_latency(self, stage: PipelineStage, duration_ms: float):
        key = stage.value
        if key not in self._metrics["stage_latencies"]:
            self._metrics["stage_latencies"][key] = []
        self._metrics["stage_latencies"][key].append(duration_ms)
        if len(self._metrics["stage_latencies"][key]) > 100:
            self._metrics["stage_latencies"][key] = self._metrics["stage_latencies"][key][-100:]

    def get_metrics(self) -> Dict[str, Any]:
        avg_stage_latencies = {}
        for stage, latencies in self._metrics["stage_latencies"].items():
            if latencies:
                avg_stage_latencies[stage] = sum(latencies) / len(latencies)

        return {
            "status": self._status.value,
            "active_pipelines": self._active_contexts,
            "max_concurrent": self._max_concurrent,
            "total_pipelines": self._metrics["total_pipelines"],
            "success_rate": self._metrics["success_pipelines"] / max(self._metrics["total_pipelines"], 1),
            "total_rejected": self._metrics["total_rejected"],
            "p95_latency_ms": self._metrics.get("p95_latency_ms", 0),
            "p99_latency_ms": self._metrics.get("p99_latency_ms", 0),
            "timeout_count": self._metrics.get("timeout_count", 0),
            "priority_queue_depth": self._metrics.get("priority_queue_depth", 0),
            "avg_latency_ms": self._metrics["avg_latency_ms"],
            "stage_avg_latencies_ms": avg_stage_latencies,
        }

    def get_context(self, context_id: str) -> Optional[PipelineContext]:
        return self._contexts.get(context_id)

    async def _process_priority_queue(self):
        """处理优先级队列（后台消费任务，预留槽位避免超额订阅）"""
        while True:
            async with self._lock:
                if not self._priority_queue or self._active_contexts >= self._max_concurrent:
                    break
                priority, ts, obj_id, signal_data = heapq.heappop(self._priority_queue)
                self._metrics["priority_queue_depth"] = len(self._priority_queue)
                self._active_contexts += 1  # 预留槽位

            # 检查队列中的信号是否过期(超过60秒丢弃)
            if time.time() - ts > 60:
                await self._release_slot()
                self._metrics["total_rejected"] += 1
                logger.warning(f"Priority queue signal expired (age={time.time()-ts:.0f}s)")
                continue

            asyncio.create_task(self._execute_reserved_pipeline(signal_data, priority))

    async def _execute_reserved_pipeline(self, signal_data: Dict[str, Any], priority: int):
        """执行已预留槽位的排队流水线，结束后释放槽位并继续消费队列"""
        try:
            await self._run_pipeline_context(signal_data, priority)
        except Exception as e:
            logger.error(f"Reserved pipeline execution error: {e}")
        finally:
            await self._release_slot()

    async def _run_timeout_handlers(self, context: PipelineContext, signal_data: Dict[str, Any]):
        """整体超时时触发 TIMEOUT_HANDLING 阶段处理器"""
        handlers = self._stages.get(PipelineStage.TIMEOUT_HANDLING.value, [])
        if not handlers:
            logger.debug("No timeout_handling handlers registered")
            return
        context.signal = signal_data
        context.set_stage_start(PipelineStage.TIMEOUT_HANDLING)
        for handler in handlers:
            try:
                await handler(context)
            except Exception as e:
                logger.error(f"Timeout handler failed: {e}")
        context.set_stage_end(PipelineStage.TIMEOUT_HANDLING)
        self._update_stage_latency(
            PipelineStage.TIMEOUT_HANDLING,
            context.get_stage_duration(PipelineStage.TIMEOUT_HANDLING),
        )
