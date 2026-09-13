"""
订单生命周期管理器
负责订单从创建到完成的全生命周期管理，包括：
- 状态追踪
- 异常处理
- 自动恢复
- 执行监控
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

# P3: 订单状态流转规则收敛到独立状态机（OrderStateMachine），本模块仅负责状态存储与副作用执行
from core.order_state_machine import (
    InvalidOrderTransition,
    OrderEvent,
    OrderPhase,
    OrderStateMachine,
    OrderStatus,
)


class OrderLifecycleManager:
    """订单生命周期管理器（P3: 状态流转规则已收敛到 core.order_state_machine.OrderStateMachine）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # P3: 独立订单状态机——所有订单流转/事件校验收敛到此处裁决
        self._state_machine = OrderStateMachine()

        # 订单状态存储 {order_id: lifecycle_info}
        self._orders: Dict[str, Dict[str, Any]] = {}
        
        # 执行统计
        self._stats: Dict[str, Any] = {
            "total_created": 0,
            "total_filled": 0,
            "total_cancelled": 0,
            "total_failed": 0,
            "total_timeout": 0,
            "avg_latency_ms": 0,
            "latency_history": [],
            "phase_timings": {},
            "error_counts": {},
            "illegal_transitions": 0,
        }
        
        # 执行监控
        self._monitor: Dict[str, Any] = {
            "active_orders": 0,
            "pending_orders": 0,
            "executing_orders": 0,
            "queue_depth": 0,
            "recent_errors": [],
            "throughput": [],
        }
        
        # 超时配置
        self._timeout_settings = config.get("execution", {}).get("timeout", {})
        self._order_timeout = self._timeout_settings.get("order_timeout", 60)  # P5: 30s->60s，匹配WS重连窗口
        self._tracking_timeout = self._timeout_settings.get("tracking_timeout", 300)
        self._ws_disconnected_timeout_extension = 30  # P5: WS断开时额外延长30s
        
        # 自动恢复配置
        self._recovery_enabled = config.get("execution", {}).get("auto_recovery", True)
        self._max_retries = config.get("execution", {}).get("max_retries", 3)
        
        # 监控周期
        self._monitor_interval = 5
        self._stats_interval = 60
        
        # 运行状态
        self._event_hooks: Dict[str, list] = {
            "on_create": [],
            "on_fill": [],
            "on_cancel": [],
            "on_fail": [],
            "on_timeout": [],
        }
        self._state_history: Dict[str, list] = {}  # 每个订单的状态变更历史
        self._running = False
        
        # 锁
        self._lock = asyncio.Lock()
        
        logger.info("OrderLifecycleManager initialized")

    def set_ws_status_checker(self, checker: callable):
        """P5: 注入WebSocket状态检查器，用于WS断开时延长订单超时"""
        self._ws_status_checker = checker

    def _is_ws_connected(self) -> bool:
        """P5: 检查WebSocket是否连接"""
        if hasattr(self, '_ws_status_checker') and self._ws_status_checker:
            try:
                return self._ws_status_checker()
            except Exception:
                pass
        return True  # 默认假设连接正常，不确定时不延长超时

    def register_event_hook(self, event_name: str, hook: callable):
        """注册事件钩子"""
        if event_name in self._event_hooks:
            self._event_hooks[event_name].append(hook)
            logger.info(f"Registered {event_name} hook: {hook.__name__}")

    def _trigger_hooks(self, event_name: str, order_id: str, data: Dict[str, Any]):
        """触发事件钩子"""
        for hook in self._event_hooks.get(event_name, []):
            try:
                hook(order_id, data)
            except Exception as e:
                logger.error(f"Hook {event_name} failed for {order_id}: {e}")

    async def start(self):
        """启动生命周期管理器"""
        if self._running:
            return
        self._running = True
        asyncio.create_task(self._monitor_loop())
        asyncio.create_task(self._stats_aggregation_loop())
        asyncio.create_task(self._timeout_check_loop())
        logger.info("OrderLifecycleManager started")

    async def stop(self):
        """停止生命周期管理器"""
        self._running = False
        logger.info("OrderLifecycleManager stopped")

    # ===================== 订单状态管理 =====================

    async def create_order(self, order_data: Dict[str, Any]) -> str:
        """创建订单记录"""
        async with self._lock:
            order_id = order_data.get("order_id", f"ORD{int(time.time()*1000)}")
            
            self._orders[order_id] = {
                **order_data,
                "status": OrderStatus.QUEUED.value,
                "phase": OrderPhase.CREATION.value,
                "timestamps": {
                    "created": datetime.now(),
                    OrderPhase.CREATION.value: datetime.now(),
                },
                "attempts": 0,
                "errors": [],
                "latencies": {},
                "exchange_order_id": None,
                "filled_price": None,
                "filled_quantity": 0,
            }
            
            self._stats["total_created"] += 1
            self._monitor["active_orders"] += 1
            
            logger.info(f"Order created: {order_id}")
            self._trigger_hooks("on_create", order_id, self._orders[order_id])
            return order_id

    def validate_transition(self, old_status: str, new_status: str) -> bool:
        """校验状态迁移是否合法（同状态视为幂等，合法）。委托给独立状态机裁决。"""
        try:
            return self._state_machine.can_reach(old_status, new_status)
        except (TypeError, ValueError):
            # 未知历史状态：保守放行并告警，避免阻断未知历史状态
            logger.warning(f"Unknown order status in transition check: '{old_status}' -> '{new_status}'")
            return True

    async def update_status(self, order_id: str, status: OrderStatus, phase: OrderPhase = None):
        """更新订单状态（状态机裁决：非法迁移直接拒绝）。"""
        async with self._lock:
            if order_id not in self._orders:
                return

            order = self._orders[order_id]
            old_status = order["status"]
            new_status = status.value

            # 状态机裁决：非法跳转直接拒绝（fail-closed）
            if not self._state_machine.can_reach(old_status, new_status):
                self._stats["illegal_transitions"] = self._stats.get("illegal_transitions", 0) + 1
                logger.warning(
                    f"Illegal order state transition rejected: {order_id} "
                    f"{old_status} -> {new_status}"
                )
                return

            order["status"] = new_status

            # 阶段耗时（与状态机正交的流程阶段统计）
            self._apply_phase_timing(order_id, phase)

            # 副作用（计数器/钩子/延迟）由状态机声明的 effect 驱动执行
            self._apply_transition_effect(order_id, old_status, status)

            # 状态历史
            self._append_state_history(order_id, old_status, new_status)

            logger.debug(f"Order {order_id} status: {old_status} -> {new_status}")

    async def apply_event(self, order_id: str, event: OrderEvent, phase: OrderPhase = None):
        """事件驱动迁移：由状态机裁决「当前状态 × 事件」并落到目标状态。

        与 update_status（目标态 API）等价，但以事件建模，供超时等触发场景使用。
        """
        async with self._lock:
            if order_id not in self._orders:
                return

            order = self._orders[order_id]
            old_status = order["status"]

            # 状态机裁决：非法「状态 × 事件」组合直接拒绝（fail-closed）
            target = self._state_machine.next_state(old_status, event)
            if target is None:
                self._stats["illegal_transitions"] = self._stats.get("illegal_transitions", 0) + 1
                logger.warning(
                    f"Illegal order event rejected: {order_id} "
                    f"{old_status} -({event.value})-> (rejected)"
                )
                return

            order["status"] = target.value
            self._apply_phase_timing(order_id, phase)
            self._apply_transition_effect(order_id, old_status, target)
            self._append_state_history(order_id, old_status, target.value)
            logger.debug(f"Order {order_id} status: {old_status} -({event.value})-> {target.value}")

    # ===================== 状态机副作用执行（收敛） =====================

    def _apply_phase_timing(self, order_id: str, phase: Optional[OrderPhase]):
        """记录阶段时间戳与阶段耗时（与状态流转正交）。"""
        if not phase:
            return
        order = self._orders[order_id]
        order["phase"] = phase.value
        order["timestamps"][phase.value] = datetime.now()

        prev_phase_time = None
        for p in list(OrderPhase):
            if p.value == phase.value:
                break
            pt = order["timestamps"].get(p.value)
            if pt:
                prev_phase_time = pt

        if prev_phase_time:
            duration_ms = (datetime.now() - prev_phase_time).total_seconds() * 1000
            order["latencies"][phase.value] = duration_ms
            self._stats["phase_timings"][phase.value] = (
                self._stats["phase_timings"].get(phase.value, []) + [duration_ms]
            )

    def _apply_transition_effect(self, order_id: str, source: str, target: OrderStatus):
        """执行状态机推导的迁移副作用（计数器增减/统计计数/事件钩子/延迟记录）。

        所有「进入某状态应做什么」的分散 if/elif 已收敛到状态机的 TransitionEffect，
        此处仅基于 effect 声明执行，不再出现针对具体状态的散落判断。
        """
        effect = self._state_machine.get_effect(source, target)
        if effect.is_empty:
            return

        if effect.decrement_active:
            self._monitor["active_orders"] = max(0, self._monitor.get("active_orders", 0) - 1)
        if effect.decrement_pending:
            self._monitor["pending_orders"] = max(0, self._monitor.get("pending_orders", 0) - 1)
        if effect.decrement_executing:
            self._monitor["executing_orders"] = max(0, self._monitor.get("executing_orders", 0) - 1)
        if effect.increment_pending:
            self._monitor["pending_orders"] = self._monitor.get("pending_orders", 0) + 1
        if effect.increment_executing:
            self._monitor["executing_orders"] = self._monitor.get("executing_orders", 0) + 1
        if effect.stat_counter:
            self._stats[effect.stat_counter] = self._stats.get(effect.stat_counter, 0) + 1
        if effect.hook:
            self._trigger_hooks(effect.hook, order_id, self._orders[order_id])
        if effect.record_latency:
            total_latency = (
                (datetime.now() - self._orders[order_id]["timestamps"]["created"]).total_seconds() * 1000
            )
            self._record_latency(total_latency)

    def _append_state_history(self, order_id: str, old_status: str, new_status: str):
        """追加状态变更历史（限制长度防内存增长）。"""
        if order_id not in self._state_history:
            self._state_history[order_id] = []
        self._state_history[order_id].append({
            "from": old_status,
            "to": new_status,
            "timestamp": datetime.now().isoformat(),
        })
        if len(self._state_history[order_id]) > 50:
            self._state_history[order_id] = self._state_history[order_id][-50:]

    async def record_error(self, order_id: str, error_code: str, error_message: str):
        """记录订单错误"""
        async with self._lock:
            if order_id not in self._orders:
                return
            
            error_info = {
                "error_code": error_code,
                "error_message": error_message,
                "timestamp": datetime.now(),
                "attempt": self._orders[order_id].get("attempts", 0),
            }
            
            self._orders[order_id]["errors"].append(error_info)
            self._orders[order_id]["attempts"] = self._orders[order_id].get("attempts", 0) + 1
            
            self._stats["error_counts"][error_code] = self._stats["error_counts"].get(error_code, 0) + 1
            
            if len(self._monitor["recent_errors"]) >= 50:
                self._monitor["recent_errors"].pop(0)
            self._monitor["recent_errors"].append({
                "order_id": order_id,
                "error_code": error_code,
                "error_message": error_message,
                "timestamp": datetime.now().isoformat(),
            })

            # 错误率超过阈值触发恢复检查
            if len(recent_orders := list(self._orders.values())) >= 20:
                recent_failed = sum(1 for o in recent_orders if o.get("status") == "failed")
                if recent_failed / len(recent_orders) > 0.3:
                    logger.warning(f"High failure rate detected: {recent_failed}/{len(recent_orders)}")

    async def set_exchange_order_id(self, order_id: str, exchange_id: str):
        """设置交易所订单ID"""
        async with self._lock:
            if order_id in self._orders:
                self._orders[order_id]["exchange_order_id"] = exchange_id

    async def record_fill(self, order_id: str, filled_price: float, filled_quantity: float):
        """记录成交信息"""
        async with self._lock:
            if order_id in self._orders:
                self._orders[order_id]["filled_price"] = filled_price
                self._orders[order_id]["filled_quantity"] = filled_quantity

    # ===================== 自动恢复 =====================

    async def check_recovery(self, order_id: str) -> Tuple[bool, Dict[str, Any]]:
        """检查是否需要恢复并重试"""
        async with self._lock:
            if order_id not in self._orders:
                return False, {}
            
            order = self._orders[order_id]
            attempts = order.get("attempts", 0)
            errors = order.get("errors", [])
            
            if not self._recovery_enabled:
                return False, {}
            
            if attempts >= self._max_retries:
                return False, {}
            
            # 检查最后一个错误是否可重试
            if errors:
                last_error = errors[-1]
                error_code = last_error.get("error_code", "")
                
                non_retryable = {"51008", "51121", "51100", "51101", "51102", "51103", "51104", "51105", "51106", "51169"}
                
                if error_code in non_retryable:
                    return False, {}
            
            return True, {
                "order_id": order_id,
                "attempts": attempts,
                "max_retries": self._max_retries,
                "retry_delay": 2 ** attempts,
                **{k: v for k, v in order.items() if k not in ["timestamps", "errors", "latencies"]},
            }

    # ===================== 超时检查 =====================

    async def _timeout_check_loop(self):
        """超时检查循环"""
        while self._running:
            await asyncio.sleep(self._monitor_interval)
            
            try:
                await self._check_timeouts()
            except Exception as e:
                logger.debug(f"Timeout check error: {e}")

    async def _check_timeouts(self):
        """检查超时订单（内联更新避免死锁，update_status也会获取同一把锁）
        
        P5: WS断开时自动延长订单超时，防止WS断连导致订单误判超时
        """
        async with self._lock:
            now = datetime.now()
            ws_connected = self._is_ws_connected()
            
            for order_id, order in list(self._orders.items()):
                status = order.get("status", "")
                created = order.get("timestamps", {}).get("created")
                if not created:
                    continue
                
                is_timeout = False
                if status == OrderStatus.EXECUTING.value:
                    # P5: WS断开时延长超时时间
                    effective_timeout = self._order_timeout
                    if not ws_connected:
                        effective_timeout += self._ws_disconnected_timeout_extension
                    is_timeout = (now - created).total_seconds() > effective_timeout
                elif status == OrderStatus.PENDING.value:
                    is_timeout = (now - created).total_seconds() > self._tracking_timeout
                
                if is_timeout:
                    # P3: 收敛到状态机裁决（TIMEOUT 事件 → TIMEOUT 终态）；避免 update_status 死锁，内联执行副作用
                    old_status = self._orders[order_id]["status"]
                    target = self._state_machine.next_state(old_status, OrderEvent.TIMEOUT)
                    if target is None:
                        self._stats["illegal_transitions"] = self._stats.get("illegal_transitions", 0) + 1
                        logger.warning(
                            f"Illegal timeout transition: {order_id} {old_status} "
                            f"-({OrderEvent.TIMEOUT.value})-> (rejected)"
                        )
                        continue
                    self._orders[order_id]["status"] = target.value
                    self._apply_transition_effect(order_id, old_status, target)
                    self._append_state_history(order_id, old_status, target.value)
                    logger.warning(f"Order {order_id} timed out (status={old_status} -> {target.value}, ws_connected={ws_connected})")

            # P0: 清理终态订单防止内存泄漏（保持在锁内，避免竞态）
            for order_id, order in list(self._orders.items()):
                if self._state_machine.is_terminal(order.get("status")):
                    self._orders.pop(order_id, None)
                    self._state_history.pop(order_id, None)

            # P0: 限制 _orders 最多保留 500 条（在锁内操作，确保不会误删活跃订单）
            if len(self._orders) > 500:
                terminal_keys = [k for k, v in self._orders.items()
                                 if self._state_machine.is_terminal(v.get("status"))]
                excess = min(len(terminal_keys), len(self._orders) - 500)
                for key in terminal_keys[:excess]:
                    self._orders.pop(key, None)
                    self._state_history.pop(key, None)
                if excess > 0:
                    logger.warning(f"_orders exceeded 500 limit, pruned {excess} terminal entries")

    # ===================== 监控 =====================

    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            await asyncio.sleep(self._monitor_interval)
            await self._update_throughput()

    async def _update_throughput(self):
        """更新吞吐量统计"""
        async with self._lock:
            now = datetime.now().timestamp()
            self._monitor["throughput"].append({
                "timestamp": now,
                "active": self._monitor["active_orders"],
                "pending": self._monitor["pending_orders"],
                "executing": self._monitor["executing_orders"],
            })
            
            if len(self._monitor["throughput"]) > 60:
                self._monitor["throughput"] = self._monitor["throughput"][-60:]

    async def _stats_aggregation_loop(self):
        """统计聚合循环"""
        while self._running:
            await asyncio.sleep(self._stats_interval)
            await self._aggregate_stats()

    async def _aggregate_stats(self):
        """聚合统计数据"""
        async with self._lock:
            latencies = self._stats["latency_history"]
            if latencies:
                self._stats["avg_latency_ms"] = sum(latencies) / len(latencies)
                
                if len(latencies) > 100:
                    self._stats["latency_history"] = latencies[-100:]

    def _record_latency(self, latency_ms: float):
        """记录延迟"""
        self._stats["latency_history"].append(latency_ms)

    # ===================== 状态查询 =====================

    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单状态"""
        return self._orders.get(order_id)

    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        latencies = self._stats.get("latency_history", [])
        
        return {
            "total_created": self._stats["total_created"],
            "total_filled": self._stats["total_filled"],
            "total_cancelled": self._stats["total_cancelled"],
            "total_failed": self._stats["total_failed"],
            "total_timeout": self._stats["total_timeout"],
            "fill_rate": self._stats["total_filled"] / max(self._stats["total_created"], 1),
            "avg_latency_ms": self._stats["avg_latency_ms"],
            "max_latency_ms": max(latencies) if latencies else 0,
            "min_latency_ms": min(latencies) if latencies else 0,
            "phase_timings": {
                phase: {
                    "avg_ms": sum(times) / len(times) if times else 0,
                    "count": len(times),
                } for phase, times in self._stats.get("phase_timings", {}).items()
            },
            "error_counts": self._stats.get("error_counts", {}),
            "illegal_transitions": self._stats.get("illegal_transitions", 0),
        }

    def get_monitor_status(self) -> Dict[str, Any]:
        """获取监控状态"""
        return {
            "active_orders": self._monitor["active_orders"],
            "pending_orders": self._monitor["pending_orders"],
            "executing_orders": self._monitor["executing_orders"],
            "recent_errors": self._monitor["recent_errors"][-20:],
            "throughput": self._monitor["throughput"],
        }

    def get_recent_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取最近订单"""
        orders = list(self._orders.values())
        orders.sort(key=lambda x: x.get("timestamps", {}).get("created", datetime.min), reverse=True)
        return orders[:limit]

    def get_order_state_history(self, order_id: str) -> list:
        """获取订单状态变更历史"""
        return self._state_history.get(order_id, [])

    def get_orders_by_status(self, status: OrderStatus) -> List[str]:
        """按状态获取订单ID列表"""
        return [oid for oid, o in self._orders.items() if o.get("status") == status.value]

    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        return {
            "active": self._monitor["active_orders"],
            "pending": self._monitor["pending_orders"],
            "executing": self._monitor["executing_orders"],
            "filled": self._stats["total_filled"],
            "failed": self._stats["total_failed"],
            "cancelled": self._stats["total_cancelled"],
            "timeout": self._stats["total_timeout"],
            "fill_rate": self._stats["total_filled"] / max(self._stats["total_created"], 1),
            "avg_latency_ms": self._stats["avg_latency_ms"],
            "registered_hooks": {k: len(v) for k, v in self._event_hooks.items()},
        }
