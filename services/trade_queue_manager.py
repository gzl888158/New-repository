"""
强化版交易队管理器
统一管理信号队列、订单队列、紧急队列，支持批量处理和动态优先级。
"""
import asyncio
import heapq
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from loguru import logger


class SignalQueue:
    """信号队列"""

    def __init__(self, max_size: int = 500):
        self._max_size = max_size
        self._queue: deque = deque(maxlen=max_size)
        self._lock = asyncio.Lock()
        self._processed = 0
        self._rejected = 0
        self._dedup_cache: Dict[str, float] = {}
        self._dedup_window = 30.0

    async def push(self, signal: Dict[str, Any]) -> bool:
        """推入信号"""
        async with self._lock:
            key = self._make_dedup_key(signal)
            now = time.time()
            
            if key in self._dedup_cache and now - self._dedup_cache[key] < self._dedup_window:
                self._rejected += 1
                return False
            
            self._dedup_cache[key] = now
            self._cleanup_dedup_cache(now)
            
            self._queue.append(signal)
            self._processed += 1
            return True

    async def pop(self) -> Optional[Dict[str, Any]]:
        """取出信号"""
        async with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    async def pop_batch(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """批量取出"""
        async with self._lock:
            batch = []
            for _ in range(min(batch_size, len(self._queue))):
                if self._queue:
                    batch.append(self._queue.popleft())
            return batch

    def _make_dedup_key(self, signal: Dict[str, Any]) -> str:
        return f"{signal.get('strategy', '')}:{signal.get('symbol', '')}:{signal.get('direction', '')}"

    def _cleanup_dedup_cache(self, now: float):
        """清理过期的去重缓存"""
        expired = [k for k, v in self._dedup_cache.items() if now - v > self._dedup_window * 2]
        for k in expired:
            del self._dedup_cache[k]

    def size(self) -> int:
        return len(self._queue)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "size": len(self._queue),
            "max_size": self._max_size,
            "processed": self._processed,
            "rejected": self._rejected,
            "dedup_cache_size": len(self._dedup_cache),
        }


class EnhancedOrderQueue:
    """增强版订单队列（支持紧急队列、动态优先级、批量处理）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._base_priorities = {
            "stop_loss": 0,
            "liquidation": 1,
            "scalping": 2,
            "trend": 3,
            "grid": 4,
            "arbitrage": 5,
        }
        
        self._main_queue: List[Tuple] = []
        self._urgent_queue: List[Tuple] = []
        self._order_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
        self._order_id_counter = 0
        
        self._max_queue_size = config.get("trading", {}).get("max_queue_size", 1000)
        self._max_urgent_size = config.get("trading", {}).get("max_urgent_size", 100)
        
        self._total_added = 0
        self._total_executed = 0
        self._total_cancelled = 0
        self._total_expired = 0

    def _compute_priority(self, order_data: Dict[str, Any]) -> int:
        """计算动态优先级"""
        strategy = order_data.get("strategy_name", "grid")
        signal_type = order_data.get("signal_type", "").lower()
        
        if "stop" in signal_type or "loss" in signal_type or "liquidation" in signal_type:
            return self._base_priorities["stop_loss"]
        
        base = self._base_priorities.get(strategy, 4)
        
        confidence = order_data.get("confidence", 0.5)
        if confidence > 0.8:
            base = max(0, base - 1)
        
        return base

    def _is_urgent(self, order_data: Dict[str, Any]) -> bool:
        """是否为紧急订单"""
        signal_type = order_data.get("signal_type", "").lower()
        if "stop" in signal_type or "loss" in signal_type:
            return True
        if "liquidation" in signal_type or "risk" in signal_type:
            return True
        if order_data.get("urgent", False):
            return True
        return False

    async def add_order(self, order_data: Dict[str, Any]) -> str:
        """添加订单"""
        async with self._lock:
            if len(self._main_queue) + len(self._urgent_queue) >= self._max_queue_size + self._max_urgent_size:
                logger.error("Order queue full, rejecting order")
                raise RuntimeError("Order queue full")
            
            order_id = f"ORD{int(time.time() * 1000)}{self._order_id_counter:04d}"
            self._order_id_counter += 1
            self._total_added += 1
            
            priority = self._compute_priority(order_data)
            is_urgent = self._is_urgent(order_data)
            seq = self._total_added
            
            entry = (priority, seq, order_id, order_data)
            
            if is_urgent:
                heapq.heappush(self._urgent_queue, entry)
            else:
                heapq.heappush(self._main_queue, entry)
            
            self._order_cache[order_id] = {
                **order_data,
                "order_id": order_id,
                "status": "queued",
                "priority": priority,
                "urgent": is_urgent,
                "create_time": datetime.now().isoformat(),
                "attempts": 0,
            }
            
            logger.info(f"Order {order_id} added, strategy={order_data.get('strategy_name')}, priority={priority}, urgent={is_urgent}")
            return order_id

    async def get_next_order(self) -> Optional[Dict[str, Any]]:
        """获取下一个订单（优先处理紧急队列）"""
        async with self._lock:
            if self._urgent_queue:
                priority, _, order_id, order_data = heapq.heappop(self._urgent_queue)
            elif self._main_queue:
                priority, _, order_id, order_data = heapq.heappop(self._main_queue)
            else:
                return None
            
            if order_id in self._order_cache:
                self._order_cache[order_id]["status"] = "executing"
                self._order_cache[order_id]["start_time"] = datetime.now().isoformat()
                self._order_cache[order_id]["attempts"] += 1
                self._total_executed += 1
            
            order_data["order_id"] = order_id
            order_data["priority"] = priority
            return order_data

    async def get_batch(self, batch_size: int = 5) -> List[Dict[str, Any]]:
        """批量获取订单"""
        async with self._lock:
            batch = []
            for _ in range(batch_size):
                if self._urgent_queue:
                    priority, _, order_id, order_data = heapq.heappop(self._urgent_queue)
                elif self._main_queue:
                    priority, _, order_id, order_data = heapq.heappop(self._main_queue)
                else:
                    break
                
                if order_id in self._order_cache:
                    self._order_cache[order_id]["status"] = "executing"
                
                order_data["order_id"] = order_id
                batch.append(order_data)
            return batch

    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        async with self._lock:
            self._main_queue = [(p, s, oid, od) for p, s, oid, od in self._main_queue if oid != order_id]
            self._urgent_queue = [(p, s, oid, od) for p, s, oid, od in self._urgent_queue if oid != order_id]
            heapq.heapify(self._main_queue)
            heapq.heapify(self._urgent_queue)
            
            if order_id in self._order_cache:
                self._order_cache[order_id]["status"] = "cancelled"
                self._order_cache[order_id]["cancel_time"] = datetime.now().isoformat()
                self._total_cancelled += 1
                return True
            return False

    async def update_order_status(self, order_id: str, status: str, **kwargs):
        """更新订单状态"""
        async with self._lock:
            if order_id in self._order_cache:
                self._order_cache[order_id]["status"] = status
                self._order_cache[order_id]["update_time"] = datetime.now().isoformat()
                for k, v in kwargs.items():
                    self._order_cache[order_id][k] = v

    async def cleanup_expired(self, max_age_seconds: int = 3600) -> int:
        """清理过期订单"""
        async with self._lock:
            now = datetime.now()
            expired_ids = []
            for order_id, info in self._order_cache.items():
                if info.get("status") not in ("queued", "executing"):
                    continue
                create_time_str = info.get("create_time", "")
                try:
                    create_time = datetime.fromisoformat(create_time_str)
                    if (now - create_time).total_seconds() > max_age_seconds:
                        expired_ids.append(order_id)
                except Exception:
                    continue
            
            for order_id in expired_ids:
                await self.cancel_order(order_id)
                self._total_expired += 1
            
            return len(expired_ids)

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """获取订单信息"""
        return self._order_cache.get(order_id)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "main_queue_size": len(self._main_queue),
            "urgent_queue_size": len(self._urgent_queue),
            "total_orders": len(self._order_cache),
            "total_added": self._total_added,
            "total_executed": self._total_executed,
            "total_cancelled": self._total_cancelled,
            "total_expired": self._total_expired,
        }


class TradeQueueManager:
    """交易队管理器（统一入口）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.signal_queue = SignalQueue(
            max_size=config.get("trading", {}).get("signal_queue_size", 500)
        )
        self.order_queue = EnhancedOrderQueue(config)
        self._running = False
        self._processor_task: Optional[asyncio.Task] = None
        logger.info("TradeQueueManager initialized")

    async def submit_signal(self, signal: Dict[str, Any]) -> bool:
        """提交信号"""
        return await self.signal_queue.push(signal)

    async def submit_order(self, order: Dict[str, Any]) -> str:
        """提交订单"""
        return await self.order_queue.add_order(order)

    async def get_pending_signals(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """获取待处理信号"""
        return await self.signal_queue.pop_batch(batch_size)

    async def get_pending_orders(self, batch_size: int = 1) -> List[Dict[str, Any]]:
        """获取待执行订单"""
        if batch_size == 1:
            order = await self.order_queue.get_next_order()
            return [order] if order else []
        return await self.order_queue.get_batch(batch_size)

    async def update_order_status(self, order_id: str, status: str, **kwargs):
        """更新订单状态"""
        await self.order_queue.update_order_status(order_id, status, **kwargs)

    async def cancel_order(self, order_id: str) -> bool:
        return await self.order_queue.cancel_order(order_id)

    async def start_signal_processor(self, callback):
        """启动信号处理器"""
        self._running = True
        self._processor_task = asyncio.create_task(self._process_loop(callback))

    async def stop(self):
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()

    async def _process_loop(self, callback):
        while self._running:
            try:
                signals = await self.get_pending_signals(batch_size=5)
                if signals:
                    for signal in signals:
                        try:
                            await callback(signal)
                        except Exception as e:
                            logger.error(f"Signal processing error: {e}")
                else:
                    await asyncio.sleep(0.5)  # 队列空闲时降低轮询频率
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Process loop error: {e}")
                await asyncio.sleep(1)

    def get_status(self) -> Dict[str, Any]:
        return {
            "signal_queue": self.signal_queue.get_stats(),
            "order_queue": self.order_queue.get_stats(),
            "running": self._running,
        }
