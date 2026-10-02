"""
算法执行引擎 (Algorithmic Execution Engine)

.. deprecated:: 实验性模块，未接入生产交易链路。

统一管理所有算法订单的生命周期：
  - 订单注册与状态机管理
  - 执行切片调度（时间/事件驱动）
  - 进度跟踪与自适应调整
  - 暂停/恢复/取消控制
  - 执行结果与审计追踪
"""
import asyncio
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Set
from loguru import logger

from .execution_monitor import AlgoExecutionMonitor


class AlgoOrderType(Enum):
    """算法订单类型"""
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    POV = "pov"                      # 参与率算法
    IMPLEMENTATION_SHORTFALL = "is"  # 实现缺口算法


class AlgoOrderStatus(Enum):
    """算法订单状态"""
    CREATED = "created"          # 已创建，未启动
    RUNNING = "running"          # 执行中
    PAUSED = "paused"            # 暂停
    COMPLETED = "completed"      # 完成
    PARTIALLY_COMPLETED = "partially_completed"  # 部分完成
    CANCELLED = "cancelled"      # 已取消
    FAILED = "failed"            # 执行失败
    EXPIRED = "expired"          # 已过期


class AlgoOrderState(Enum):
    """内部执行状态（更细粒度）"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    SCHEDULING = "scheduling"           # 调度下一个切片
    WAITING_INTERVAL = "waiting_interval"  # 等待切片间隔
    SUBMITTING = "submitting"           # 提交切片订单
    AWAITING_FILL = "awaiting_fill"     # 等待成交
    ADJUSTING = "adjusting"             # 自适应调整
    FINALIZING = "finalizing"           # 收尾处理


@dataclass
class AlgoOrderConfig:
    """算法订单配置"""
    order_id: str = ""
    symbol: str = ""
    algo_type: AlgoOrderType = AlgoOrderType.TWAP
    side: str = "buy"
    total_quantity: float = 0.0
    limit_price: Optional[float] = None      # 限价上限
    # 时间配置
    start_time: Optional[datetime] = None    # 开始时间
    end_time: Optional[datetime] = None      # 结束时间
    duration_seconds: float = 3600.0         # 执行时长(秒)
    # 切片配置
    num_slices: int = 10
    min_slice_interval: float = 10.0         # 最小切片间隔(秒)
    max_slice_interval: float = 60.0         # 最大切片间隔(秒)
    min_slice_qty: float = 0.001             # 最小切片量
    # 自适应
    adaptive: bool = True                    # 启用自适应调整
    aggressive_multiplier: float = 1.2       # 激进乘数（加速执行）
    passive_multiplier: float = 0.7          # 保守乘数（减速执行）
    max_participation_rate: float = 0.05     # 最大市场参与率
    # 风控
    max_slippage_pct: float = 0.02           # 最大滑点
    max_cost_pct: float = 0.01               # 最大执行成本
    auto_cancel_on_breach: bool = True       # 超限自动取消
    # 外部依赖
    executor_fn: Optional[Callable] = None   # 执行函数(order_params) -> fill_result
    market_data_fn: Optional[Callable] = None  # 获取行情(symbol) -> market_data
    # 切片事件回调（由执行引擎绑定到 AlgoExecutionMonitor）
    on_slice_scheduled_fn: Optional[Callable] = None  # (sequence, quantity, scheduled_time)
    on_slice_submitted_fn: Optional[Callable] = None  # (sequence, quantity, price, order_type)


@dataclass
class ExecutionSlice:
    """执行切片"""
    slice_id: str
    sequence: int
    quantity: float
    price_limit: Optional[float] = None
    order_type: str = "limit"
    # 执行结果
    status: str = "pending"           # pending / submitted / filled / partial / cancelled / rejected
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fill_cost: float = 0.0
    slippage_bps: float = 0.0
    # 审计
    order_response: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / max(self.quantity, 1e-10)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "sequence": self.sequence,
            "quantity": round(self.quantity, 4),
            "filled_quantity": round(self.filled_quantity, 4),
            "avg_fill_price": round(self.avg_fill_price, 4),
            "slippage_bps": round(self.slippage_bps, 2),
            "fill_rate": round(self.fill_rate, 3),
            "status": self.status,
        }


@dataclass
class AlgoExecutionResult:
    """算法执行结果"""
    order_id: str
    algo_type: str
    symbol: str
    side: str
    # 量
    total_quantity: float
    filled_quantity: float
    fill_rate: float
    # 价格
    target_price: float = 0.0         # 基准价格 (TWAP/VWAP)
    avg_execution_price: float = 0.0
    arrival_price: float = 0.0        # 到达价格（订单创建时mid）
    # 执行质量
    execution_slippage_bps: float = 0.0     # vs target
    arrival_slippage_bps: float = 0.0       # vs arrival
    total_cost: float = 0.0                 # 总成本 (USD)
    implementation_shortfall: float = 0.0    # 实现缺口 (USD)
    # 详情
    num_slices: int = 0
    filled_slices: int = 0
    rejected_slices: int = 0
    duration_seconds: float = 0.0
    # 时间线
    status: AlgoOrderStatus = AlgoOrderStatus.COMPLETED
    slices: List[ExecutionSlice] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "algo_type": self.algo_type,
            "symbol": self.symbol,
            "side": self.side,
            "quantities": {
                "total": round(self.total_quantity, 4),
                "filled": round(self.filled_quantity, 4),
                "fill_rate": round(self.fill_rate, 3),
            },
            "prices": {
                "target": round(self.target_price, 4),
                "avg_execution": round(self.avg_execution_price, 4),
                "arrival": round(self.arrival_price, 4),
            },
            "quality": {
                "execution_slippage_bps": round(self.execution_slippage_bps, 2),
                "arrival_slippage_bps": round(self.arrival_slippage_bps, 2),
                "total_cost": round(self.total_cost, 4),
                "implementation_shortfall": round(self.implementation_shortfall, 4),
            },
            "slices": {
                "total": self.num_slices,
                "filled": self.filled_slices,
                "rejected": self.rejected_slices,
            },
            "duration_seconds": round(self.duration_seconds, 2),
            "status": self.status.value,
            "timeline": {
                "start": self.start_time.isoformat() if self.start_time else None,
                "end": self.end_time.isoformat() if self.end_time else None,
            },
        }


# ═══════════════════════════════════════════════════════════════
# 算法执行引擎
# ═══════════════════════════════════════════════════════════════

class AlgoExecutionEngine:
    """统一算法执行引擎"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("algo_execution_engine", {}) if config else {}
        self._enabled = cfg.get("enabled", True)
        self._max_concurrent_algos = cfg.get("max_concurrent_algos", 5)
        self._slice_polling_interval = cfg.get("slice_polling_interval", 1.0)  # 秒
        self._order_timeout = cfg.get("order_timeout", 300)  # 订单超时(秒)
        self._persist_dir = cfg.get("persist_dir", "./data/algo_orders")

        # 注册的算法执行器
        self._executors: Dict[AlgoOrderType, Any] = {}  # type -> executor instance
        # 活跃的算法订单
        self._active_orders: Dict[str, AlgoOrderConfig] = {}
        self._order_states: Dict[str, AlgoOrderState] = {}
        self._order_statuses: Dict[str, AlgoOrderStatus] = {}
        self._order_slices: Dict[str, List[ExecutionSlice]] = defaultdict(list)
        self._order_results: Dict[str, AlgoExecutionResult] = {}
        # 调度任务
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        # 并发背压信号量：超出最大并发时等待槽位，而非直接抛错
        self._concurrency_semaphore = asyncio.Semaphore(self._max_concurrent_algos)

        # 引用外部执行器
        self._order_executor = None
        self._okx_client = None

        # 实时执行监控器
        self._execution_monitor = AlgoExecutionMonitor(config)

        import os
        os.makedirs(self._persist_dir, exist_ok=True)
        logger.info(f"AlgoExecutionEngine initialized: max_concurrent={self._max_concurrent_algos}")

    # ── 执行器注册 ────────────────────────────────────────────

    def register_executor(self, algo_type: AlgoOrderType, executor):
        """注册算法执行器实例"""
        self._executors[algo_type] = executor
        logger.info(f"Algorithm executor registered: {algo_type.value}")

    def set_order_executor(self, order_executor):
        """注入订单执行器（用于实际下单）"""
        self._order_executor = order_executor

    def set_okx_client(self, okx_client):
        """注入OKX客户端（用于行情数据）"""
        self._okx_client = okx_client

    # ── 订单管理 ──────────────────────────────────────────────

    async def submit_algo_order(self, config: AlgoOrderConfig) -> AlgoExecutionResult:
        """提交算法订单并等待完成（并发背压：超出上限时等待槽位）"""
        if config.algo_type not in self._executors:
            raise ValueError(f"No executor registered for {config.algo_type.value}")

        # 并发背压：等待可用槽位，而非直接抛错
        await self._concurrency_semaphore.acquire()
        try:
            config.order_id = config.order_id or f"algo_{uuid.uuid4().hex[:12]}"

            async with self._lock:
                # 幂等保护：重复 order_id 直接拒绝，避免覆盖活跃订单状态/任务
                if config.order_id in self._active_orders or config.order_id in self._tasks:
                    raise ValueError(f"duplicate algo order_id: {config.order_id}")
                self._active_orders[config.order_id] = config
                self._order_states[config.order_id] = AlgoOrderState.INITIALIZING
                self._order_statuses[config.order_id] = AlgoOrderStatus.CREATED

            # 通知监控器
            self._execution_monitor.on_order_created(
                order_id=config.order_id,
                algo_type=config.algo_type.value,
                symbol=config.symbol,
                side=config.side,
                total_quantity=config.total_quantity,
                num_slices=config.num_slices,
                duration_seconds=config.duration_seconds,
            )

            # 绑定切片调度/提交回调到执行监控器（打通延迟与成交质量统计链路）
            order_id = config.order_id
            config.on_slice_scheduled_fn = (
                lambda seq, qty, st, _oid=order_id: self._execution_monitor.on_slice_scheduled(_oid, seq, qty, st)
            )
            config.on_slice_submitted_fn = (
                lambda seq, qty, price, ot="limit", _oid=order_id: self._execution_monitor.on_slice_submitted(_oid, seq, qty, price, ot)
            )

            executor = self._executors[config.algo_type]
            task = asyncio.create_task(self._run_algo_order(config, executor))
            self._tasks[config.order_id] = task

            try:
                result = await task
                return result
            except asyncio.CancelledError:
                await self.cancel_algo_order(config.order_id)
                return self._order_results.get(config.order_id, AlgoExecutionResult(
                    order_id=config.order_id, algo_type=config.algo_type.value,
                    symbol=config.symbol, side=config.side,
                    total_quantity=config.total_quantity, filled_quantity=0, fill_rate=0,
                    status=AlgoOrderStatus.CANCELLED,
                ))
        finally:
            # 释放并发槽位（背压闭环）
            self._concurrency_semaphore.release()

    async def _run_algo_order(self, config: AlgoOrderConfig,
                               executor) -> AlgoExecutionResult:
        """运行算法订单主循环"""
        start_time = datetime.now()
        result = AlgoExecutionResult(
            order_id=config.order_id,
            algo_type=config.algo_type.value,
            symbol=config.symbol,
            side=config.side,
            total_quantity=config.total_quantity,
            filled_quantity=0.0,
            fill_rate=0.0,
            start_time=start_time,
        )

        try:
            self._order_statuses[config.order_id] = AlgoOrderStatus.RUNNING

            # 委托给具体算法执行器
            result = await executor.execute(config, self._on_slice_filled)

            result.status = AlgoOrderStatus.COMPLETED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()

            logger.info(f"Algo order completed: {config.order_id} "
                       f"filled={result.filled_quantity}/{result.total_quantity} "
                       f"avg_px={result.avg_execution_price:.4f} "
                       f"slip={result.execution_slippage_bps:.1f}bps")

        except asyncio.CancelledError:
            result.status = AlgoOrderStatus.CANCELLED
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()
        except Exception as e:
            result.status = AlgoOrderStatus.FAILED
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.duration_seconds = (result.end_time - start_time).total_seconds()
            logger.error(f"Algo order failed: {config.order_id} - {e}")

        finally:
            # 清理
            result.num_slices = len(self._order_slices.get(config.order_id, []))
            result.filled_slices = sum(1 for s in self._order_slices.get(config.order_id, [])
                                      if s.status == "filled")
            result.rejected_slices = sum(1 for s in self._order_slices.get(config.order_id, [])
                                        if s.status == "rejected")
            result.slices = self._order_slices.get(config.order_id, [])

            # 计算执行价格
            filled_slices = [s for s in result.slices if s.filled_quantity > 0]
            if filled_slices:
                total_filled = sum(s.filled_quantity for s in filled_slices)
                total_cost = sum(s.filled_quantity * s.avg_fill_price for s in filled_slices)
                result.avg_execution_price = total_cost / max(total_filled, 1e-10)
                result.filled_quantity = total_filled
                result.fill_rate = result.filled_quantity / max(result.total_quantity, 1e-10)

            self._order_results[config.order_id] = result

            # 通知监控器
            self._execution_monitor.on_order_completed(config.order_id)

            # 释放活跃订单/任务跟踪（防止内存泄漏，配合并发背压闭环）
            self._active_orders.pop(config.order_id, None)
            self._tasks.pop(config.order_id, None)

    async def _on_slice_filled(self, order_id: str, slice_data: ExecutionSlice):
        """切片成交回调"""
        async with self._lock:
            self._order_slices[order_id].append(slice_data)

        # 通知监控器
        if slice_data.status == "filled":
            self._execution_monitor.on_slice_filled(
                order_id=order_id,
                sequence=slice_data.sequence,
                filled_qty=slice_data.filled_quantity,
                avg_fill_price=slice_data.avg_fill_price,
            )
        elif slice_data.status == "rejected":
            self._execution_monitor.on_slice_failed(
                order_id=order_id,
                sequence=slice_data.sequence,
                reason=slice_data.error_message,
            )

    # ── 控制接口 ──────────────────────────────────────────────

    async def pause_algo_order(self, order_id: str):
        """暂停算法订单"""
        if order_id in self._order_statuses:
            self._order_statuses[order_id] = AlgoOrderStatus.PAUSED
            self._order_states[order_id] = AlgoOrderState.IDLE
            logger.info(f"Algo order paused: {order_id}")

    async def resume_algo_order(self, order_id: str):
        """恢复算法订单"""
        if order_id in self._order_statuses:
            self._order_statuses[order_id] = AlgoOrderStatus.RUNNING
            self._order_states[order_id] = AlgoOrderState.SCHEDULING
            logger.info(f"Algo order resumed: {order_id}")

    async def cancel_algo_order(self, order_id: str):
        """取消算法订单"""
        task = self._tasks.pop(order_id, None)
        if task is not None and not task.done():
            task.cancel()
            try:
                # 等待任务完成清理（释放锁/槽位），CancelledError 属预期结果
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        if order_id in self._order_statuses:
            self._order_statuses[order_id] = AlgoOrderStatus.CANCELLED
        self._execution_monitor.on_order_cancelled(order_id, "cancelled by user")
        logger.info(f"Algo order cancelled: {order_id}")

    async def cancel_all(self):
        """取消所有算法订单"""
        for order_id in list(self._tasks.keys()):
            await self.cancel_algo_order(order_id)

    # ── 查询接口 ──────────────────────────────────────────────

    def get_active_orders(self) -> List[Dict[str, Any]]:
        """获取所有活跃算法订单"""
        orders = []
        for order_id, config in self._active_orders.items():
            status = self._order_statuses.get(order_id, AlgoOrderStatus.CREATED)
            slices = self._order_slices.get(order_id, [])
            filled = sum(s.filled_quantity for s in slices)
            orders.append({
                "order_id": order_id,
                "symbol": config.symbol,
                "algo_type": config.algo_type.value,
                "side": config.side,
                "total_qty": config.total_quantity,
                "filled_qty": filled,
                "progress": round(filled / max(config.total_quantity, 1e-10) * 100, 1),
                "status": status.value,
                "slices_executed": len(slices),
                "slices_total": config.num_slices,
            })
        return orders

    def get_order_result(self, order_id: str) -> Optional[AlgoExecutionResult]:
        return self._order_results.get(order_id)

    def get_order_slices(self, order_id: str) -> List[ExecutionSlice]:
        return self._order_slices.get(order_id, [])

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "active_algos": len(self._active_orders),
            "max_concurrent": self._max_concurrent_algos,
            "registered_executors": [t.value for t in self._executors],
            "active_orders": self.get_active_orders(),
            "completed_results": len(self._order_results),
            "execution_monitor": self._execution_monitor.get_status(),
        }

    def get_execution_monitor(self) -> AlgoExecutionMonitor:
        """获取执行监控器实例"""
        return self._execution_monitor

    def persist_monitor_state(self):
        """将监控状态写入 JSON 文件，供 Dashboard 进程读取"""
        import json
        import os
        try:
            path = os.path.join(self._persist_dir, "algo_monitor.json")
            data = {
                "timestamp": datetime.now().isoformat(),
                "status": self._execution_monitor.get_status(),
                "engine": {
                    "enabled": self._enabled,
                    "active_algos": len(self._active_orders),
                    "max_concurrent": self._max_concurrent_algos,
                    "registered_executors": [t.value for t in self._executors],
                    "active_orders": self.get_active_orders(),
                    "completed_results": len(self._order_results),
                },
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist monitor state: {e}")

    def load_monitor_state(self) -> Optional[Dict[str, Any]]:
        """从 JSON 文件加载持久化的监控状态（供重启恢复/审计读取）"""
        import json
        import os
        try:
            path = os.path.join(self._persist_dir, "algo_monitor.json")
            if not os.path.exists(path):
                return None
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # 恢复 ISO 时间戳为 datetime 对象
            ts = data.get("timestamp")
            if isinstance(ts, str):
                try:
                    data["timestamp"] = datetime.fromisoformat(ts)
                except ValueError:
                    pass
            return data
        except Exception as e:
            logger.warning(f"Failed to load monitor state: {e}")
            return None
