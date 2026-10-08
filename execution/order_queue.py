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
        if self._rate <= 0:
            # fail-closed：无效限流配置（rate<=0）拒绝放行，避免除零导致无限等待/崩溃
            raise RuntimeError(f"TokenBucket rate must be > 0, got {self._rate}")
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
            signal_type = (order_data.get("signal_type") or "")
            strategy_name = order_data.get("strategy_name", "grid")
            symbol = order_data.get("symbol", "")
            
            # R31: 止损/平仓订单去重 — 同一symbol+strategy已有pending止损时，替换旧单而非拒绝新单
            is_stop_order = "stop" in signal_type.lower() or "loss" in signal_type.lower()
            if is_stop_order:
                # 1) 队列中仍排队的止损单 — 移除旧单，允许新单替换
                stale_indices = []
                for idx, (_, _, oid, cached_data) in enumerate(self._queue):
                    if not isinstance(cached_data, dict):
                        continue
                    cached_sig = (cached_data.get("signal_type") or "")
                    if (cached_data.get("symbol") == symbol and
                        cached_data.get("strategy_name") == strategy_name and
                        ("stop" in cached_sig.lower() or "loss" in cached_sig.lower())):
                        stale_indices.append((idx, oid))
                if stale_indices:
                    for _, stale_oid in stale_indices:
                        if stale_oid in self._order_cache:
                            self._order_cache[stale_oid]["status"] = "replaced"
                    # 从后往前删除，避免索引偏移
                    for idx, _ in sorted(stale_indices, reverse=True):
                        del self._queue[idx]
                    heapq.heapify(self._queue)
                    logger.debug(f"Stop loss replace: {symbol} {strategy_name} replaced {len(stale_indices)} pending stop(s)")
                # 2) 已从队列取出、正在执行的止损单 — 拒绝新单（执行中不可替换）
                for oid, cached in self._order_cache.items():
                    if not isinstance(cached, dict):
                        continue
                    if cached.get("status") != "executing":
                        continue
                    cached_sig = (cached.get("signal_type") or "")
                    if (cached.get("symbol") == symbol and
                        cached.get("strategy_name") == strategy_name and
                        ("stop" in cached_sig.lower() or "loss" in cached_sig.lower())):
                        logger.debug(f"Stop loss dedup: {symbol} {strategy_name} stop loss executing, skip")
                        return oid
            
            # R32: 队列接近满载时，清理超过180秒的旧订单腾出空间（原60秒过短，grid挂单易被误清）
            if len(self._queue) >= self._max_queue_size * 0.9:
                self._cleanup_stale_orders_locked(max_age_seconds=180)
            
            if len(self._queue) >= self._max_queue_size:
                # R32: 激进清理阈值从30秒提升到120秒
                self._cleanup_stale_orders_locked(max_age_seconds=120)
            
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
        moved_to_dead_letter = False
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
                        moved_to_dead_letter = True
                # 终态订单清理：executed/failed/cancelled 状态从缓存移除，防止内存泄漏
                if status in ("executed", "failed", "cancelled", "stale_removed"):
                    self._order_cache.pop(order_id, None)
        # 落盘在锁外执行（同步 I/O 不阻塞其它协程；单线程事件循环内读 _dead_letter 是原子的）
        if moved_to_dead_letter:
            self._save_dead_letter()
    
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

    def _move_to_dead_letter_locked(self, order_id: str, reason: str) -> bool:
        """将订单移入死信队列（需在锁内调用）。返回是否成功移入；落盘由调用方在锁外执行。"""
        entry = self._order_cache.get(order_id)
        if not entry:
            return False
        dl_entry = {
            **entry,
            "order_id": order_id,
            "dead_letter_reason": reason,
            "dead_letter_time": datetime.now().isoformat(),
        }
        self._dead_letter.append(dl_entry)
        if len(self._dead_letter) > self._dead_letter_max:
            self._dead_letter = self._dead_letter[-self._dead_letter_max:]
        logger.warning(f"Order moved to dead-letter queue: {order_id} ({reason})")
        return True

    async def mark_dead_letter(self, order_id: str, reason: str) -> bool:
        """显式将订单移入死信队列"""
        async with self._lock:
            if order_id not in self._order_cache:
                return False
            self._move_to_dead_letter_locked(order_id, reason)
            self._order_cache.pop(order_id, None)
        self._save_dead_letter()
        return True

    async def get_dead_letter_orders(self, limit: int = 100) -> List[Dict[str, Any]]:
        async with self._lock:
            return list(self._dead_letter[-limit:])

    async def get_dead_letter_size(self) -> int:
        async with self._lock:
            return len(self._dead_letter)

    async def requeue_dead_letter(self, order_id: str) -> bool:
        """将死信订单重新放回队列（人工确认后重放）"""
        found = False
        async with self._lock:
            for i, entry in enumerate(self._dead_letter):
                if entry.get("order_id") == order_id:
                    dl_entry = self._dead_letter.pop(i)
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
                    found = True
                    break
        if found:
            self._save_dead_letter()
        return found

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

    @staticmethod
    def _from_json_safe(obj):
        """递归恢复 JSON 加载后的类型（isoformat 字符串 → datetime，与 _to_json_safe 对称）。"""
        if isinstance(obj, dict):
            return {k: OrderQueue._from_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [OrderQueue._from_json_safe(v) for v in obj]
        if isinstance(obj, str):
            try:
                return datetime.fromisoformat(obj)
            except ValueError:
                return obj
        return obj

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
                self._dead_letter = self._from_json_safe(items)[:self._dead_letter_max]
                logger.info(f"Loaded {len(self._dead_letter)} dead-letter orders from {path}")
        except Exception as e:
            logger.warning(f"Failed to load dead-letter queue: {e}")