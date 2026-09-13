"""负责订单优先级队列管理与令牌桶限流，并做止损订单去重。"""
import asyncio
import heapq
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger


class TokenBucket:
    def __init__(self, rate: float, capacity: int = 10):
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_refill = datetime.now().timestamp()
        self._lock = asyncio.Lock()

    async def acquire(self):
        while True:
            async with self._lock:
                now = datetime.now().timestamp()
                time_passed = now - self._last_refill
                self._tokens = min(self._capacity, self._tokens + time_passed * self._rate)
                self._last_refill = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return

                wait_time = (1 - self._tokens) / self._rate
            # 释放锁后再sleep，不阻塞其他acquire
            await asyncio.sleep(wait_time)


class OrderQueue:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._priority_order = {
            "stop_loss": 0,
            "scalping": 1,
            "trend": 2,
            "grid": 3,
            "arbitrage": 4
        }
        
        self._queue = []
        self._order_id_counter = 0
        self._lock = asyncio.Lock()
        self._order_cache: Dict[str, Dict[str, Any]] = {}
        self._max_queue_size = config.get("trading", {}).get("max_queue_size", 1000)

        # 死信队列：失败订单超过最大重试次数后移入，避免静默丢失，支持审计与重放
        self._dead_letter: List[Dict[str, Any]] = []
        self._max_retries = config.get("execution", {}).get("retry_attempts", 3)
        self._dead_letter_max = config.get("execution", {}).get("dead_letter_max", 200)
        # 死信队列持久化：重启后保留失败订单
        dl_persist = config.get("execution", {}).get("dead_letter_persistence", {})
        self._dl_persistence_enabled = dl_persist.get("enabled", True)
        self._dl_persistence_file = dl_persist.get("file", "data/dead_letter_orders.json")
        self._load_dead_letter()

    async def add_order(self, order_data: Dict[str, Any]) -> str:
        async with self._lock:
            signal_type = order_data.get("signal_type", "")
            strategy_name = order_data.get("strategy_name", "grid")
            symbol = order_data.get("symbol", "")
            
            # 止损/平仓订单去重：同一symbol+strategy已有pending止损的，跳过
            is_stop_order = "stop" in signal_type.lower() or "loss" in signal_type.lower()
            if is_stop_order:
                for _, _, oid, cached_data in self._queue:
                    cached_sig = cached_data.get("signal_type", "")
                    if (cached_data.get("symbol") == symbol and
                        cached_data.get("strategy_name") == strategy_name and
                        ("stop" in cached_sig.lower() or "loss" in cached_sig.lower())):
                        logger.debug(f"Stop loss dedup: {symbol} {strategy_name} already has pending stop loss")
                        return oid
            
            # 队列接近满载时，清理超过60秒的旧订单腾出空间
            if len(self._queue) >= self._max_queue_size * 0.9:
                self._cleanup_stale_orders_locked(max_age_seconds=60)
            
            if len(self._queue) >= self._max_queue_size:
                # 尝试清理更激进的30秒旧订单
                self._cleanup_stale_orders_locked(max_age_seconds=30)
            
            if len(self._queue) >= self._max_queue_size:
                logger.error(f"Order queue full ({self._max_queue_size}), rejecting order")
                raise RuntimeError(f"Order queue full ({self._max_queue_size})")
            
            order_id = f"ORD{self._order_id_counter:08d}"
            self._order_id_counter += 1
            
            priority_key = self._priority_order.get(strategy_name, 3)
            
            if is_stop_order:
                priority_key = 0
            
            heapq.heappush(self._queue, (priority_key, self._order_id_counter, order_id, order_data))
            
            self._order_cache[order_id] = {
                **order_data,
                "status": "queued",
                "create_time": datetime.now(),
                "attempts": 0
            }
            
            logger.info(f"Order added: {order_id}, strategy: {strategy_name}, priority: {priority_key}")
            return order_id
    
    def _cleanup_stale_orders_locked(self, max_age_seconds: int = 60):
        """清理超过指定秒数的旧订单（需在锁内调用）"""
        now = datetime.now()
        new_queue = []
        removed_count = 0
        for item in self._queue:
            priority, counter, oid, cached_data = item
            # 从 _order_cache 中读取 create_time（队列项不包含此字段）
            cache = self._order_cache.get(oid, {}) if isinstance(oid, str) else {}
            create_time = cache.get("create_time")
            if create_time and (now - create_time).total_seconds() > max_age_seconds:
                removed_count += 1
                if oid in self._order_cache:
                    self._order_cache[oid]["status"] = "stale_removed"
            else:
                new_queue.append(item)
        if removed_count > 0:
            self._queue = new_queue
            heapq.heapify(self._queue)
            logger.warning(f"Cleaned up {removed_count} stale orders (> {max_age_seconds}s) from queue")
    
    async def get_next_order(self) -> Optional[Dict[str, Any]]:
        async with self._lock:
            if not self._queue:
                return None
            
            priority, _, order_id, order_data = heapq.heappop(self._queue)
            order_data["order_id"] = order_id
            
            if order_id in self._order_cache:
                self._order_cache[order_id]["status"] = "executing"
            
            logger.info(f"Order retrieved: {order_id}, priority: {priority}")
            return order_data
    
    async def get_queue_size(self) -> int:
        async with self._lock:
            return len(self._queue)
    
    async def cancel_order(self, order_id: str) -> bool:
        async with self._lock:
            for i, (priority, _, oid, order_data) in enumerate(self._queue):
                if oid == order_id:
                    del self._queue[i]
                    heapq.heapify(self._queue)
                    
                    if oid in self._order_cache:
                        self._order_cache[oid]["status"] = "cancelled"
                    
                    logger.info(f"Order cancelled: {order_id}")
                    return True
            return False
    
    async def update_order_status(self, order_id: str, status: str):
        async with self._lock:
            if order_id in self._order_cache:
                self._order_cache[order_id]["status"] = status
                self._order_cache[order_id]["update_time"] = datetime.now()
                # 失败订单：超过最大重试次数自动进入死信队列，避免静默丢失
                if status == "failed":
                    attempts = self._order_cache[order_id].get("attempts", 0)
                    if attempts >= self._max_retries:
                        self._move_to_dead_letter_locked(order_id, f"max retries exceeded ({attempts})")
                        self._order_cache.pop(order_id, None)
                        return
                # 终态订单清理：executed/failed/cancelled 状态从缓存移除，防止内存泄漏
                if status in ("executed", "failed", "cancelled", "stale_removed"):
                    self._order_cache.pop(order_id, None)
    
    async def get_order_status(self, order_id: str) -> Optional[str]:
        async with self._lock:
            return self._order_cache.get(order_id, {}).get("status")
    
    async def increment_order_attempts(self, order_id: str):
        async with self._lock:
            if order_id in self._order_cache:
                self._order_cache[order_id]["attempts"] = self._order_cache[order_id].get("attempts", 0) + 1
    
    async def get_order_attempts(self, order_id: str) -> int:
        async with self._lock:
            return self._order_cache.get(order_id, {}).get("attempts", 0)
    
    async def clear_queue(self):
        async with self._lock:
            self._queue.clear()
            self._order_cache.clear()
            logger.info("Order queue cleared")
    
    async def get_pending_orders(self) -> List[Dict[str, Any]]:
        async with self._lock:
            return [order for order in self._order_cache.values() if order["status"] in ("queued", "executing")]

    # ===================== 死信队列 =====================

    def _move_to_dead_letter_locked(self, order_id: str, reason: str):
        """将订单移入死信队列（需在锁内调用）"""
        entry = self._order_cache.get(order_id)
        if not entry:
            return
        dl_entry = {
            **entry,
            "order_id": order_id,
            "dead_letter_reason": reason,
            "dead_letter_time": datetime.now().isoformat(),
        }
        self._dead_letter.append(dl_entry)
        if len(self._dead_letter) > self._dead_letter_max:
            self._dead_letter = self._dead_letter[-self._dead_letter_max:]
        self._save_dead_letter()
        logger.warning(f"Order moved to dead-letter queue: {order_id} ({reason})")

    async def mark_dead_letter(self, order_id: str, reason: str) -> bool:
        """显式将订单移入死信队列"""
        async with self._lock:
            if order_id not in self._order_cache:
                return False
            self._move_to_dead_letter_locked(order_id, reason)
            self._order_cache.pop(order_id, None)
            return True

    async def get_dead_letter_orders(self, limit: int = 100) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._dead_letter[-limit:])

    async def get_dead_letter_size(self) -> int:
        async with self._lock:
            return len(self._dead_letter)

    async def requeue_dead_letter(self, order_id: str) -> bool:
        """将死信订单重新放回队列（人工确认后重放）"""
        async with self._lock:
            for i, entry in enumerate(self._dead_letter):
                if entry.get("order_id") == order_id:
                    dl_entry = self._dead_letter.pop(i)
                    self._save_dead_letter()
                    priority = self._priority_order.get(dl_entry.get("strategy_name", "grid"), 3)
                    self._order_id_counter += 1
                    heapq.heappush(self._queue, (priority, self._order_id_counter, order_id, dl_entry))
                    self._order_cache[order_id] = {
                        **dl_entry,
                        "status": "queued",
                        "create_time": datetime.now(),
                        "attempts": dl_entry.get("attempts", 0),
                    }
                    logger.info(f"Dead-letter order requeued: {order_id}")
                    return True
            return False

    # ===================== 死信队列持久化 =====================

    @staticmethod
    def _to_json_safe(obj):
        """递归转换为可 JSON 序列化的类型（datetime→isoformat，未知类型→str）"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {str(k): OrderQueue._to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [OrderQueue._to_json_safe(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        return str(obj)

    def _save_dead_letter(self):
        if not self._dl_persistence_enabled:
            return
        try:
            path = Path(self._dl_persistence_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"dead_letter": self._to_json_safe(self._dead_letter)}
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            logger.debug(f"Failed to persist dead-letter queue: {e}")

    def _load_dead_letter(self):
        if not self._dl_persistence_enabled:
            return
        try:
            path = Path(self._dl_persistence_file)
            if not path.exists():
                return
            payload = json.loads(path.read_text(encoding="utf-8"))
            items = payload.get("dead_letter", []) if isinstance(payload, dict) else []
            if isinstance(items, list):
                self._dead_letter = items[:self._dead_letter_max]
                logger.info(f"Loaded {len(self._dead_letter)} dead-letter orders from {path}")
        except Exception as e:
            logger.warning(f"Failed to load dead-letter queue: {e}")