"""
条件单管理器 - Conditional Order Manager

功能：
1. 条件单补挂失败处理（重试机制）
2. 存量仓位补挂风险控制（避免"-2021 Order would immediately trigger"）
3. 条件单同步机制（系统重启后与交易所同步）
4. 强化异常处理和日志记录
5. 止盈设置逻辑优化
"""

import asyncio
import json
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from core.tp_sl_monitor import TpSlMonitor


class ConditionalOrderManager:
    """条件单管理器"""

    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        self.config = config
        self._okx_client = okx_client
        self._redis_cache = redis_cache

        # 统一止盈止损口径：保本缓冲、持仓保护视图等只读聚合
        self._tp_sl_monitor = TpSlMonitor(config)

        self._orders_file = "data/conditional_orders.json"
        self._active_orders: Dict[str, Dict[str, Any]] = {}
        self._pending_orders: Dict[str, Dict[str, Any]] = {}  # 等待重试的订单
        self._failed_orders: Dict[str, Dict[str, Any]] = {}   # 失败订单记录

        # 企业级强化：algoClOrdId 会话盐，避免跨进程/跨重启后历史成交或取消订单
        # 残留的确定性 ID 与同价位重挂单冲突；同会话内盐不变，重试幂等不受影响。
        self._algo_id_salt = uuid.uuid4().hex[:6]

        self._retry_interval = 5  # 重试间隔（秒）
        self._max_retries = 3     # 最大重试次数
        self._retry_lock = asyncio.Lock()
        self._save_lock = threading.Lock()  # 文件写入锁，防止并发写竞态（同步方法兼容）
        self._running = False     # 控制后台循环的运行状态
        self._heartbeat_enabled = True  # 心跳检测开关
        self._sync_enabled = True       # 同步循环开关

        # 审计日志
        self._audit_log: List[Dict[str, Any]] = []  # 最多 500 条
        self._audit_log_max = 500

        # 错误分类计数
        self._error_counts: Dict[str, int] = {
            "network": 0,
            "auth": 0,
            "rate_limit": 0,
            "order_rejected": 0,
            "unknown": 0,
        }

        # 增强指标
        self._uptime_start = time.time()
        self._placed_count = 0
        self._failed_placement_count = 0
        self._retry_success_count = 0
        self._retry_total_count = 0
        self._heartbeat_restored_count = 0
        self._sync_operations_count = 0

        # 企业级强化：下单失败熔断器（连续失败达阈值暂停下单，冷却后自动恢复）
        self._circuit_open = False          # 熔断器是否打开（打开时拒绝新挂单）
        self._circuit_opened_at = 0.0       # 熔断打开时间戳
        self._consecutive_failures = 0      # 连续下单失败计数
        self._circuit_threshold = 10        # 连续失败 N 次触发熔断
        self._circuit_cooldown = 300        # 熔断冷却时间（秒）

        self._load_active_orders()

        logger.info(f"ConditionalOrderManager initialized: {len(self._active_orders)} active orders loaded")

    def _save_active_orders(self):
        """原子写入条件单状态到文件（使用线程锁防止并发写竞态）
        
        包含版本号、时间戳，并同时保存 pending 和 failed 订单状态。
        """
        try:
            with self._save_lock:
                dir_path = os.path.dirname(self._orders_file)
                if dir_path:
                    os.makedirs(dir_path, exist_ok=True)
                tmp_file = self._orders_file + ".tmp"
                state = self.collect_persistent_state()
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(state, f, ensure_ascii=False, indent=2)
                os.replace(tmp_file, self._orders_file)
        except Exception as e:
            logger.error(f"Failed to save active orders: {e}")

    def _load_active_orders(self):
        try:
            if os.path.exists(self._orders_file):
                with open(self._orders_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    # 新版格式：包含 version 字段的是持久化状态
                    if "version" in data:
                        self.restore_persistent_state(data)
                        logger.info(
                            f"Loaded {len(self._active_orders)} active, "
                            f"{len(self._pending_orders)} pending, "
                            f"{len(self._failed_orders)} failed orders from {self._orders_file}"
                        )
                    else:
                        # 旧版格式：仅 active_orders
                        self._active_orders = data
                        logger.info(f"Loaded {len(self._active_orders)} active conditional orders from {self._orders_file} (legacy format)")
        except Exception as e:
            logger.error(f"Failed to load active orders: {e}")
            self._active_orders = {}

    async def start(self):
        """启动条件单管理器"""
        self._running = True
        asyncio.create_task(self._retry_loop())
        asyncio.create_task(self._sync_loop())
        asyncio.create_task(self._orphan_cleanup_loop())  # P2: 孤儿条件单清理
        logger.info("ConditionalOrderManager started")

    async def stop(self):
        """停止条件单管理器：取消后台循环，持久化当前状态"""
        logger.info("Stopping ConditionalOrderManager...")
        self._running = False
        self._save_active_orders()
        logger.info("ConditionalOrderManager stopped")

    async def _retry_loop(self):
        """重试失败的条件单"""
        while self._running:
            async with self._retry_lock:
                orders_to_retry = []
                for order_id, order_info in self._pending_orders.items():
                    if order_info.get("retry_count", 0) < self._max_retries:
                        orders_to_retry.append((order_id, order_info))

                for order_id, order_info in orders_to_retry:
                    try:
                        self._retry_total_count += 1
                        result = await self._retry_order(order_id, order_info)
                        if result:
                            logger.info(f"Retry succeeded for {order_info['symbol']}: {order_id}")
                            self._retry_success_count += 1
                            self._pending_orders.pop(order_id, None)
                        else:
                            order_info["retry_count"] = order_info.get("retry_count", 0) + 1
                            logger.warning(f"Retry {order_info['retry_count']}/{self._max_retries} for {order_info['symbol']}: {order_id}")
                            if order_info["retry_count"] >= self._max_retries:
                                order_info["failed_at"] = time.time()
                                self._failed_orders[order_id] = order_info
                                self._pending_orders.pop(order_id, None)
                                logger.error(f"Order failed after {self._max_retries} retries: {order_info['symbol']}")
                    except Exception as e:
                        self._categorize_error(e)
                        logger.error(f"Retry error for {order_id}: {e}")

            await asyncio.sleep(self._retry_interval)

    async def _sync_loop(self):
        """定期同步交易所条件单状态"""
        while self._running:
            if self._sync_enabled:
                await self._sync_with_exchange()
                self._sync_operations_count += 1
            await asyncio.sleep(30)

    async def _sync_with_exchange(self):
        """与交易所同步条件单状态"""
        try:
            exchange_orders = self._okx_client.get_algo_orders()
            if not exchange_orders:
                return

            exchange_order_ids = set()
            for order in exchange_orders:
                algo_id = order.get("algoId", "")
                if algo_id:
                    exchange_order_ids.add(algo_id)

            # 清理已取消/已成交的订单
            local_ids_to_remove = []
            for order_id in self._active_orders:
                if order_id not in exchange_order_ids:
                    local_ids_to_remove.append(order_id)

            for order_id in local_ids_to_remove:
                self._active_orders.pop(order_id, None)
                logger.debug(f"Removed stale order from local cache: {order_id}")

            # 同步新的交易所订单
            for order in exchange_orders:
                algo_id = order.get("algoId", "")
                if algo_id and algo_id not in self._active_orders:
                    # 安全价格转换：OKX可能返回空字符串
                    tp_px = order.get("tpTriggerPx", "") or "0"
                    sl_px = order.get("slTriggerPx", "") or "0"
                    try:
                        price = float(tp_px) if float(tp_px) > 0 else float(sl_px)
                    except (ValueError, TypeError):
                        price = 0.0
                    
                    try:
                        sz = float(order.get("sz", "0") or "0")
                    except (ValueError, TypeError):
                        sz = 0.0
                    
                    try:
                        lever = int(order.get("lever", "1") or "1")
                    except (ValueError, TypeError):
                        lever = 1
                    
                    order_info = {
                        "symbol": order.get("instId", ""),
                        "side": order.get("side", ""),
                        "type": "take_profit" if "tp" in order.get("ordType", "").lower() else "stop_loss",
                        "price": price,
                        "quantity": sz,
                        "leverage": lever,
                        "is_algo": True,
                        "status": order.get("state", "")
                    }
                    self._active_orders[algo_id] = order_info
                    logger.info(f"Synchronized order from exchange: {algo_id} - {order_info['symbol']}")

            self._save_active_orders()

        except Exception as e:
            logger.error(f"Failed to sync with exchange: {e}")

    async def _retry_order(self, order_id: str, order_info: Dict[str, Any]) -> bool:
        """重试放置条件单"""
        symbol = order_info["symbol"]
        side = order_info["side"]
        quantity = order_info["quantity"]
        leverage = order_info["leverage"]
        price = order_info["price"]
        order_type = order_info["type"]

        # 重试前重新校验方向，避免反复提交已知无效价格（sCode 51277）
        close_side = "sell" if side == "long" else "buy"
        if self._okx_client.validate_conditional_price(
            symbol, order_type, price, pos_side=side, side=close_side
        ) is None:
            logger.warning(f"Skip retrying {order_id}: invalid {order_type} price {price} for {symbol}")
            return False

        try:
            # 复用首次挂单时的 algoClOrdId，保证重试时交易所侧幂等，避免重复挂单
            clordid = order_info.get("clOrdId", "")
            result = self._okx_client.place_order(
                symbol=symbol,
                side="sell" if side == "long" else "buy",
                order_type="conditional",
                quantity=quantity,
                leverage=leverage,
                stop_price=price,
                reduce_only=True,
                pos_side=side,
                conditional_type=order_type,
                clOrdId=clordid
            )

            if result and not result.get("_failed"):
                new_order_id = result.get("algoId", "") or result.get("ordId", "")
                if new_order_id:
                    self._active_orders[new_order_id] = order_info
                    self._save_active_orders()
                    return True
            return False
        except Exception as e:
            self._categorize_error(e)
            logger.error(f"Retry order failed: {e}")
            return False

    def _check_immediate_trigger(self, symbol: str, side: str, trigger_price: float) -> bool:
        """检查是否会立即触发（避免"-2021 Order would immediately trigger"错误）"""
        try:
            ticker = self._okx_client.get_ticker(symbol)
            if not ticker:
                return False
            current_price = float(ticker["last"])

            if side == "long":
                if trigger_price >= current_price:
                    logger.warning(f"Stop loss would immediately trigger for {symbol}: current={current_price:.4f}, sl={trigger_price:.4f}")
                    return True
            else:
                if trigger_price <= current_price:
                    logger.warning(f"Stop loss would immediately trigger for {symbol}: current={current_price:.4f}, sl={trigger_price:.4f}")
                    return True

            return False
        except Exception as e:
            logger.error(f"Failed to check immediate trigger: {e}")
            return False

    async def place_stop_loss(self, symbol: str, side: str, quantity: float,
                              trigger_price: float, leverage: int, is_new_position: bool = True) -> Optional[str]:
        """
        放置止损条件单

        参数：
        - is_new_position: 是否是新开仓位（用于风险控制）
        """
        try:
            close_side = "sell" if side == "long" else "buy"
            clordid = self._gen_algo_cl_ord_id(symbol, side, "stop_loss", trigger_price)

            # 企业级强化：熔断器（连续失败暂停下单）与防重（同一持仓方向仅一张止损）
            if not self._circuit_breaker_allows():
                logger.warning(f"Circuit breaker open, skip placing SL for {symbol}")
                return None
            if self._has_active_order(symbol, side, "stop_loss"):
                logger.info(f"Skip placing SL for {symbol}: active stop_loss already exists")
                return None

            # 校验止损价方向，避免反复提交无效止损价
            if self._okx_client.validate_conditional_price(
                symbol, "stop_loss", trigger_price, pos_side=side, side=close_side
            ) is None:
                logger.warning(f"Skip placing SL for {symbol}: invalid price {trigger_price} vs market")
                return None

            # 风险控制：存量仓位不自动重置止损（避免瞬间平仓）
            if not is_new_position:
                if self._check_immediate_trigger(symbol, side, trigger_price):
                    logger.warning(f"Skip placing SL for existing position {symbol}: would immediately trigger")
                    return None

            result = self._okx_client.place_order(
                symbol=symbol,
                side=close_side,
                order_type="conditional",
                quantity=quantity,
                leverage=leverage,
                stop_price=trigger_price,
                reduce_only=True,
                pos_side=side,
                conditional_type="stop_loss",
                clOrdId=clordid
            )

            if result and not result.get("_failed"):
                order_id = result.get("algoId", "") or result.get("ordId", "")
                if order_id:
                    self._active_orders[order_id] = {
                        "symbol": symbol,
                        "side": side,
                        "type": "stop_loss",
                        "price": trigger_price,
                        "quantity": quantity,
                        "leverage": leverage,
                        "is_algo": True,
                        "is_new_position": is_new_position
                    }
                    self._save_active_orders()
                    self._placed_count += 1
                    self._record_placement_success()
                    self._add_audit_entry("place_stop_loss", symbol, order_id,
                                          {"price": trigger_price, "quantity": quantity, "side": side})
                    logger.info(f"Stop loss placed for {symbol}: {trigger_price:.4f} (algoId={order_id})")
                    return order_id
                # 成功但无 order_id（API 返回格式变化），保守按失败处理
                self._failed_placement_count += 1
                self._record_placement_failure()
                logger.warning(f"Stop loss placed but no algoId returned: {result}")
            else:
                # 失败：先检查特殊错误码
                s_code = str(result.get("sCode", "") or "")
                
                # 51068: 重复订单 — 已存在于交易所，同步即可
                if s_code == "51068":
                    return self._handle_51068_duplicate(symbol, side, result)
                
                # 51261: 订单超限 — 清理后重试一次
                if s_code == "51261":
                    logger.warning(f"51261 overflow for {symbol}, cleaning stale orders...")
                    cleaned = await self._handle_51261_overflow()
                    if cleaned > 0:
                        logger.info(f"51261 retry after cleaning {cleaned} orders for {symbol}")
                        # 重试一次
                        retry_result = self._okx_client.place_order(
                            symbol=symbol,
                            side=close_side,
                            order_type="conditional",
                            quantity=quantity,
                            leverage=leverage,
                            stop_price=trigger_price,
                            reduce_only=True,
                            pos_side=side,
                            conditional_type="stop_loss",
                            clOrdId=clordid
                        )
                        if retry_result and not retry_result.get("_failed"):
                            order_id = retry_result.get("algoId", "") or retry_result.get("ordId", "")
                            if order_id:
                                self._active_orders[order_id] = {
                                    "symbol": symbol, "side": side, "type": "stop_loss",
                                    "price": trigger_price, "quantity": quantity,
                                    "leverage": leverage, "is_algo": True,
                                    "is_new_position": is_new_position
                                }
                                self._save_active_orders()
                                self._placed_count += 1
                                self._record_placement_success()
                                self._add_audit_entry("place_stop_loss", symbol, order_id,
                                                      {"price": trigger_price, "quantity": quantity, "side": side})
                                logger.info(f"Stop loss placed after 51261 cleanup for {symbol}: {trigger_price:.4f}")
                                return order_id
                
                # 网络/限流类进重试队列，参数错误直接记 failed_orders
                self._failed_placement_count += 1
                self._record_placement_failure()
                self._queue_failed_order("pending_sl", symbol, side, "stop_loss",
                                         trigger_price, quantity, leverage, clordid, result)

            return None
        except Exception as e:
            self._categorize_error(e)
            self._failed_placement_count += 1
            self._record_placement_failure()
            logger.error(f"Failed to place stop loss: {e}")
            self._queue_failed_order("pending_sl", symbol, side, "stop_loss",
                                     trigger_price, quantity, leverage, clordid,
                                     {"_failed": True, "sCode": "exception", "sMsg": str(e)})
            return None

    async def place_take_profit(self, symbol: str, side: str, quantity: float,
                                trigger_price: float, leverage: int) -> Optional[str]:
        """放置止盈条件单"""
        try:
            close_side = "sell" if side == "long" else "buy"
            clordid = self._gen_algo_cl_ord_id(symbol, side, "take_profit", trigger_price)

            # 企业级强化：熔断器 + 防重（分段止盈按价位去重，仅拦截完全相同的重复单）
            if not self._circuit_breaker_allows():
                logger.warning(f"Circuit breaker open, skip placing TP for {symbol}")
                return None
            if self._has_active_order(symbol, side, "take_profit", trigger_price):
                logger.info(f"Skip placing TP for {symbol}: active take_profit at {trigger_price} already exists")
                return None

            # 校验止盈价方向，避免反复提交被交易所以 sCode 51277 拒绝
            if self._okx_client.validate_conditional_price(
                symbol, "take_profit", trigger_price, pos_side=side, side=close_side
            ) is None:
                logger.warning(f"Skip placing TP for {symbol}: invalid price {trigger_price} vs market")
                return None

            result = self._okx_client.place_order(
                symbol=symbol,
                side=close_side,
                order_type="conditional",
                quantity=quantity,
                leverage=leverage,
                stop_price=trigger_price,
                reduce_only=True,
                pos_side=side,
                conditional_type="take_profit",
                clOrdId=clordid
            )

            if result and not result.get("_failed"):
                order_id = result.get("algoId", "") or result.get("ordId", "")
                if order_id:
                    self._active_orders[order_id] = {
                        "symbol": symbol,
                        "side": side,
                        "type": "take_profit",
                        "price": trigger_price,
                        "quantity": quantity,
                        "leverage": leverage,
                        "is_algo": True
                    }
                    self._save_active_orders()
                    self._placed_count += 1
                    self._record_placement_success()
                    self._add_audit_entry("place_take_profit", symbol, order_id,
                                          {"price": trigger_price, "quantity": quantity, "side": side})
                    logger.info(f"Take profit placed for {symbol}: {trigger_price:.4f} (algoId={order_id})")
                    return order_id
                # 成功但无 order_id（API 返回格式变化），保守按失败处理
                self._failed_placement_count += 1
                self._record_placement_failure()
                logger.warning(f"Take profit placed but no algoId returned: {result}")
            else:
                # 失败：先检查特殊错误码
                s_code = str(result.get("sCode", "") or "")
                
                # 51068: 重复订单 — 已存在于交易所，同步即可
                if s_code == "51068":
                    return self._handle_51068_duplicate(symbol, side, result)
                
                # 51261: 订单超限 — 清理后重试一次
                if s_code == "51261":
                    logger.warning(f"51261 overflow for {symbol}, cleaning stale orders...")
                    cleaned = await self._handle_51261_overflow()
                    if cleaned > 0:
                        logger.info(f"51261 retry after cleaning {cleaned} orders for {symbol}")
                        retry_result = self._okx_client.place_order(
                            symbol=symbol,
                            side=close_side,
                            order_type="conditional",
                            quantity=quantity,
                            leverage=leverage,
                            stop_price=trigger_price,
                            reduce_only=True,
                            pos_side=side,
                            conditional_type="take_profit",
                            clOrdId=clordid
                        )
                        if retry_result and not retry_result.get("_failed"):
                            order_id = retry_result.get("algoId", "") or retry_result.get("ordId", "")
                            if order_id:
                                self._active_orders[order_id] = {
                                    "symbol": symbol, "side": side, "type": "take_profit",
                                    "price": trigger_price, "quantity": quantity,
                                    "leverage": leverage, "is_algo": True
                                }
                                self._save_active_orders()
                                self._placed_count += 1
                                self._record_placement_success()
                                self._add_audit_entry("place_take_profit", symbol, order_id,
                                                      {"price": trigger_price, "quantity": quantity, "side": side})
                                logger.info(f"Take profit placed after 51261 cleanup for {symbol}: {trigger_price:.4f}")
                                return order_id
                
                # 网络/限流类进重试队列，参数错误直接记 failed_orders
                self._failed_placement_count += 1
                self._record_placement_failure()
                self._queue_failed_order("pending_tp", symbol, side, "take_profit",
                                         trigger_price, quantity, leverage, clordid, result)

            return None
        except Exception as e:
            self._categorize_error(e)
            self._failed_placement_count += 1
            self._record_placement_failure()
            logger.error(f"Failed to place take profit: {e}")
            self._queue_failed_order("pending_tp", symbol, side, "take_profit",
                                     trigger_price, quantity, leverage, clordid,
                                     {"_failed": True, "sCode": "exception", "sMsg": str(e)})
            return None

    async def place_staged_take_profit(self, symbol: str, side: str, total_quantity: float,
                                       base_tp_price: float, entry_price: float, leverage: int) -> List[str]:
        """
        分段止盈：优化版本
        - TP1: 40% 仓位在 60% 目标距离处平仓
        - TP2: 50% 仓位在目标价处平仓
        - TP3: 10% 仓位在 150% 目标距离处平仓（激进部分）
        """
        close_side = "sell" if side == "long" else "buy"
        placed_order_ids = []

        try:
            if side == "long":
                tp_range = base_tp_price - entry_price
                tp1_price = entry_price + tp_range * 0.6
                tp2_price = base_tp_price
                tp3_price = entry_price + tp_range * 1.5
            else:
                tp_range = entry_price - base_tp_price
                tp1_price = entry_price - tp_range * 0.6
                tp2_price = base_tp_price
                tp3_price = entry_price - tp_range * 1.5

            stages = [
                (tp1_price, 0.40, "near"),
                (tp2_price, 0.50, "mid"),
                (tp3_price, 0.10, "far"),
            ]

            for tp_price, ratio, stage_name in stages:
                # 账本一致性（修复4）：stage_qty 保持币数交 place_take_profit；仅用张数取整结果做最小手数判断
                stage_qty = total_quantity * ratio
                contracts_check = self._okx_client.round_quantity_to_lot(
                    symbol, self._okx_client.coin_to_contracts(symbol, stage_qty)
                )
                if contracts_check <= 0:
                    logger.warning(f"Staged TP {stage_name} qty too small for {symbol}, skipping")
                    continue

                # 验证止盈价格合理性
                if side == "long" and tp_price <= entry_price:
                    logger.warning(f"Staged TP {stage_name} price {tp_price:.4f} <= entry {entry_price:.4f}, skipping")
                    continue
                elif side == "short" and tp_price >= entry_price:
                    logger.warning(f"Staged TP {stage_name} price {tp_price:.4f} >= entry {entry_price:.4f}, skipping")
                    continue

                order_id = await self.place_take_profit(
                    symbol=symbol,
                    side=side,
                    quantity=stage_qty,
                    trigger_price=round(tp_price, 4),
                    leverage=leverage
                )

                if order_id:
                    placed_order_ids.append(order_id)
                    logger.info(f"Staged TP {stage_name} placed for {symbol}: {tp_price:.4f} ({ratio*100:.0f}%)")
                else:
                    logger.warning(f"Staged TP {stage_name} failed for {symbol}: {tp_price:.4f}")

            return placed_order_ids
        except Exception as e:
            logger.error(f"Failed to place staged take profit: {e}")
            return placed_order_ids

    def cancel_conditional_order(self, symbol: str, order_id: str):
        """取消条件单"""
        try:
            order_info = self._active_orders.get(order_id)
            if not order_info:
                logger.warning(f"Order {order_id} not found in active orders")
                return False

            if order_info.get("is_algo"):
                result = self._okx_client.cancel_algo_order(symbol, order_id)
            else:
                result = self._okx_client.cancel_order(symbol, order_id)

            if result:
                self._active_orders.pop(order_id, None)
                self._save_active_orders()
                self._add_audit_entry("cancel", symbol, order_id, {"status": "success"})
                logger.info(f"Canceled conditional order: {order_id} for {symbol}")
                return True
            else:
                self._add_audit_entry("cancel", symbol, order_id, {"status": "failed"})
                logger.warning(f"Failed to cancel order: {order_id}")
                return False
        except Exception as e:
            logger.error(f"Failed to cancel conditional order: {e}")
            return False

    def cancel_all_conditional_orders(self, symbol: str = None):
        """取消所有条件单"""
        try:
            if symbol:
                orders_to_cancel = [
                    (order_id, order_info) for order_id, order_info in self._active_orders.items()
                    if order_info["symbol"] == symbol
                ]
            else:
                orders_to_cancel = list(self._active_orders.items())

            success_count = 0
            for order_id, order_info in orders_to_cancel:
                try:
                    if order_info.get("is_algo"):
                        self._okx_client.cancel_algo_order(order_info["symbol"], order_id)
                    else:
                        self._okx_client.cancel_order(order_info["symbol"], order_id)
                    success_count += 1
                except Exception as e:
                    logger.error(f"Failed to cancel order {order_id}: {e}")

            for order_id, _ in orders_to_cancel:
                self._active_orders.pop(order_id, None)
            self._save_active_orders()

            logger.info(f"Canceled {success_count}/{len(orders_to_cancel)} conditional orders" + (f" for {symbol}" if symbol else ""))
        except Exception as e:
            logger.error(f"Failed to cancel conditional orders: {e}")

    async def update_stop_loss(self, symbol: str, old_order_id: str, new_stop_price: float, is_new_position: bool = True) -> Optional[str]:
        """
        更新止损条件单（取消旧单 + 挂新单）

        参数：
        - is_new_position: 是否是新开仓位（存量仓位谨慎更新）
        """
        try:
            order_info = self._active_orders.get(old_order_id)
            if not order_info:
                logger.warning(f"Order {old_order_id} not found in active orders")
                return None

            if not is_new_position:
                if self._check_immediate_trigger(symbol, order_info["side"], new_stop_price):
                    logger.warning(f"Skip updating SL for existing position {symbol}: would immediately trigger")
                    return None

            self.cancel_conditional_order(symbol, old_order_id)

            new_order_id = await self.place_stop_loss(
                symbol=symbol,
                side=order_info["side"],
                quantity=order_info["quantity"],
                trigger_price=new_stop_price,
                leverage=order_info["leverage"],
                is_new_position=is_new_position
            )

            if new_order_id:
                logger.info(f"Stop loss updated for {symbol}: {new_stop_price:.4f}")
                return new_order_id
            else:
                logger.warning(f"Failed to update stop loss for {symbol}")
                return None
        except Exception as e:
            logger.error(f"Failed to update stop loss: {e}")
            return None

    async def ensure_all_positions_have_stops(self, skip_existing: bool = True):
        """确保所有持仓都有止损单"""
        try:
            positions = self._okx_client.get_positions()
            for pos_data in positions:
                try:
                    position = self._okx_client._parse_position(pos_data)
                    if not position:
                        continue

                    symbol = position.symbol
                    # 账本一致性（修复4）：position.quantity 是 OKX 合约张数，转币数再交 place_stop_loss
                    quantity = self._okx_client.contracts_to_coins(symbol, abs(float(position.quantity)))
                    if quantity <= 0:
                        continue

                    side = position.side
                    mark_price = float(position.mark_price)
                    leverage = int(position.leverage)

                    has_stop = any(
                        order_info["symbol"] == symbol and order_info["type"] == "stop_loss"
                        for order_info in self._active_orders.values()
                    )

                    if has_stop and skip_existing:
                        continue

                    if not has_stop:
                        stop_price = self._calculate_stop_price(side, mark_price, leverage)
                        if stop_price:
                            await self.place_stop_loss(
                                symbol=symbol,
                                side=side,
                                quantity=quantity,
                                trigger_price=stop_price,
                                leverage=leverage,
                                is_new_position=False
                            )
                except Exception as e:
                    logger.error(f"Error processing position {pos_data}: {e}")
        except Exception as e:
            logger.error(f"Failed to ensure all positions have stops: {e}")

    def _calculate_stop_price(self, side: str, mark_price: float, leverage: int) -> float:
        """计算止损价格（企业级：配置化强平距离 + 最小/最大止损比例钳制）

        距强平安全距离 = 1/杠杆 * margin_call_offset，并钳制到 [0.8%, 8%]，
        避免高杠杆下止损过紧（被插针扫出）或低杠杆下止损过松。
        """
        cfg = self.config.get("conditional_order", {}).get("stop_loss", {})
        try:
            margin_call_offset = float(cfg.get("margin_call_offset", 0.30))
        except (TypeError, ValueError):
            margin_call_offset = 0.30

        try:
            leverage = max(int(leverage), 1)
            liq_distance = 1.0 / leverage * margin_call_offset
            # 钳制到 [0.008, 0.08]，兼顾小仓位保护与防插针
            offset = min(max(liq_distance, 0.008), 0.08)
            if side == "long":
                stop_price = mark_price * (1 - offset)
            else:
                stop_price = mark_price * (1 + offset)
            return round(stop_price, 4)
        except Exception as e:
            logger.error(f"Failed to calculate stop price: {e}")
            return 0

    # ==================== 企业级强化：防重 / 幂等 / 熔断 ====================

    def _has_active_order(self, symbol: str, side: str, order_type: str,
                          price: Optional[float] = None) -> bool:
        """判断是否已存在同 (symbol, side, type[, price]) 的活跃条件单。

        - 止损/移动止损：同一持仓方向只允许一张 → 按 (symbol, side, type) 去重。
        - 止盈：分段止盈可能挂多张不同价位 → 额外按 price 去重，仅拦截完全相同的重复单。
        """
        for o in self._active_orders.values():
            if o.get("symbol") != symbol or o.get("side") != side or o.get("type") != order_type:
                continue
            if price is not None:
                if abs(float(o.get("price", 0) or 0) - float(price)) < 1e-9:
                    return True
            else:
                return True
        return False

    def _gen_algo_cl_ord_id(self, symbol: str, side: str, order_type: str,
                            price: Optional[float] = None) -> str:
        """生成确定性、幂等的 algoClOrdId（过滤为字母数字，最长 32 字符）。

        同一 (symbol, side, type[, price]) 生成相同 ID，使交易所侧具备幂等性，
        重试时不会重复挂单。仅保留字母数字以规避 sCode 51000（非法 clOrdId）。

        企业级强化：末尾加入会话盐 _algo_id_salt，使 ID 在同一会话内仍确定性
        （重试幂等），但跨会话/跨重启后不复用历史 ID，避免已成交/已取消订单残留
        的相同 ID 与同价位重挂单发生幂等冲突。
        """
        raw = f"{order_type}_{symbol}_{side}_{price if price is not None else ''}_{self._algo_id_salt}"
        cleaned = "".join(c for c in raw if c.isalnum())
        return cleaned[:32]

    def _is_retryable_failure(self, result: Optional[Dict[str, Any]]) -> bool:
        """判断条件单失败是否值得重试。

        只有网络 / 限流 / 系统繁忙类错误才重试；方向(51277)、资金不足(51008)、
        无持仓(51169)、超出限额(51010)、重复订单(51068)、订单超限(51261)
        等参数性拒绝属确定性失败，重试无用，应直接记为失败，避免无意义地反复提交。

        注意：51068 和 51261 在 place_stop_loss/place_take_profit 中有特殊处理逻辑，
        不经过此方法。
        """
        if not result or not result.get("_failed"):
            return False
        s_code = str(result.get("sCode", "") or "")
        s_msg = str(result.get("sMsg", "") or "").lower()

        # 底层网络错误（_make_request 返回 None）或捕获异常
        if s_code in ("0", "exception"):
            return True
        # OKX 限流 / 系统繁忙类错误码（可退避后重试）
        if s_code in ("50004", "50011", "50013", "50014", "51103", "429"):
            return True
        # 按错误信息关键词兜底
        retry_kw = ("timeout", "connection", "network", "rate limit", "too many",
                    "throttle", "temporarily", "try again", "busy", "dns", "reset",
                    "socket", "eof", "internal server error", "service unavailable")
        return any(k in s_msg for k in retry_kw)

    def _handle_51068_duplicate(self, symbol: str, side: str, result: Dict[str, Any]) -> Optional[str]:
        """处理 51068 重复订单：订单已存在于交易所，触发同步后返回成功。

        51068 = "already exists within algoClOrdId" — 说明该止损/止盈已经挂在交易所，
        只需同步本地状态即可，不计为失败。
        """
        s_msg = str(result.get("sMsg", "") or "")
        logger.info(f"51068 duplicate order for {symbol}/{side}: {s_msg}, syncing from exchange")
        # 异步触发同步（不阻塞当前流程）
        asyncio.ensure_future(self._sync_with_exchange())
        # 返回 None 表示已有订单，但不算失败
        return "duplicate_handled"

    async def _handle_51261_overflow(self) -> int:
        """处理 51261 条件单超限：清理过期/无持仓对应的挂单。

        每次触发时取消最多 20 个过期订单，释放配额后返回清理数量。
        优先取消处于 'canceled' 状态的订单，其次取消无活跃持仓对应的订单。
        """
        try:
            exchange_orders = self._okx_client.get_algo_orders()
            if not exchange_orders:
                return 0

            # 获取当前持仓的symbol列表
            try:
                positions = self._okx_client.get_positions()
                active_symbols = set()
                for p in positions:
                    inst_id = p.get("instId", "")
                    pos_qty = abs(float(p.get("pos", 0) or 0))
                    if inst_id and pos_qty > 0:
                        active_symbols.add(inst_id)
            except Exception:
                active_symbols = set()

            clean_count = 0
            # 第一优先级：取消已取消/已成交状态的订单
            for order in exchange_orders:
                state = order.get("state", "")
                algo_id = order.get("algoId", "")
                if state in ("canceled", "filled") and algo_id:
                    try:
                        self._okx_client.cancel_algo_order(
                            order.get("instId", ""), algo_id
                        )
                        self._active_orders.pop(algo_id, None)
                        clean_count += 1
                        logger.info(f"51261 cleanup: removed {state} algo order {algo_id}")
                    except Exception:
                        pass

            # 第二优先级：取消无持仓对应的订单
            if clean_count < 20:
                for order in exchange_orders:
                    if clean_count >= 20:
                        break
                    algo_id = order.get("algoId", "")
                    inst_id = order.get("instId", "")
                    state = order.get("state", "")
                    if inst_id not in active_symbols and algo_id and state not in ("canceled", "filled"):
                        try:
                            self._okx_client.cancel_algo_order(inst_id, algo_id)
                            self._active_orders.pop(algo_id, None)
                            clean_count += 1
                            logger.info(f"51261 cleanup: removed orphan algo order {algo_id} for {inst_id}")
                        except Exception:
                            pass

            if clean_count > 0:
                self._save_active_orders()
                logger.warning(f"51261 cleanup: removed {clean_count} stale algo orders to free quota")

            return clean_count
        except Exception as e:
            logger.error(f"51261 cleanup failed: {e}")
            return 0

    def _queue_failed_order(self, prefix: str, symbol: str, side: str, order_type: str,
                            price: float, quantity: float, leverage: int,
                            clordid: str, result: Optional[Dict[str, Any]]):
        """将失败的条件单按错误类型分流。

        - 网络/限流类：进 _pending_orders 等待重试（复用 clordid 保持幂等）。
        - 参数类拒绝：直接记 _failed_orders，不再无意义重试。
        """
        entry = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "price": price,
            "quantity": quantity,
            "leverage": leverage,
            "is_algo": True,
            "retry_count": 0,
            "clOrdId": clordid,
        }
        s_code = (result or {}).get("sCode")
        s_msg = (result or {}).get("sMsg")
        ts = int(time.time() * 1000)
        if self._is_retryable_failure(result):
            pending_id = f"{prefix}_{symbol}_{side}_{ts}"
            self._pending_orders[pending_id] = entry
            logger.warning(f"{order_type} placement failed (retryable), queued: {symbol} sCode={s_code}")
        else:
            failed_id = f"failed_{prefix}_{symbol}_{side}_{ts}"
            entry["sCode"] = s_code
            entry["sMsg"] = s_msg
            self._failed_orders[failed_id] = entry
            logger.warning(f"{order_type} placement rejected (non-retryable): {symbol} sCode={s_code} sMsg={s_msg}")

    def _circuit_breaker_allows(self) -> bool:
        """熔断器检查：打开且未过冷却期则拒绝下单。"""
        if not self._circuit_open:
            return True
        if time.time() - self._circuit_opened_at >= self._circuit_cooldown:
            # 冷却期结束，自动恢复
            self._circuit_open = False
            self._consecutive_failures = 0
            logger.info("Conditional order circuit breaker recovered (cooldown elapsed)")
            return True
        return False

    def _record_placement_failure(self):
        """记录一次下单失败，达到阈值则打开熔断器。"""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._circuit_threshold and not self._circuit_open:
            self._circuit_open = True
            self._circuit_opened_at = time.time()
            logger.error(
                f"Conditional order circuit breaker OPEN after {self._consecutive_failures} "
                f"consecutive failures (cooldown {self._circuit_cooldown}s)"
            )

    def _record_placement_success(self):
        """下单成功时重置连续失败计数并关闭熔断器。"""
        self._consecutive_failures = 0
        if self._circuit_open:
            self._circuit_open = False
            logger.info("Conditional order circuit breaker closed after successful placement")

    async def place_move_stop(self, symbol: str, side: str, quantity: float,
                              trigger_price: float, callback_rate: float,
                              leverage: int) -> Optional[str]:
        """
        放置移动止损条件单（OKX 的 move_stop 类型）
        
        参数：
        - callback_rate: 回调幅度（小数，如 0.015 表示 1.5%）
        - trigger_price: 触发价，价格达到此价后开始追踪
        """
        try:
            close_side = "sell" if side == "long" else "buy"

            # 企业级强化：熔断器 + 防重（同一持仓方向仅一张移动止损）
            if not self._circuit_breaker_allows():
                logger.warning(f"Circuit breaker open, skip placing move_stop for {symbol}")
                return None
            if self._has_active_order(symbol, side, "move_stop"):
                logger.info(f"Skip placing move_stop for {symbol}: active move_stop already exists")
                return None

            result = self._okx_client.place_order(
                symbol=symbol,
                side=close_side,
                order_type="move_stop",
                quantity=quantity,
                leverage=leverage,
                stop_price=trigger_price,
                reduce_only=True,
                pos_side=side,
                conditional_type="move_stop",
                clOrdId=self._gen_algo_cl_ord_id(symbol, side, "move_stop", trigger_price),
                extra_params={"callbackRatio": str(callback_rate)}
            )

            if result and not result.get("_failed"):
                order_id = result.get("algoId", "") or result.get("ordId", "")
                if order_id:
                    self._active_orders[order_id] = {
                        "symbol": symbol,
                        "side": side,
                        "type": "move_stop",
                        "trigger_price": trigger_price,
                        "callback_rate": callback_rate,
                        "quantity": quantity,
                        "leverage": leverage,
                        "is_algo": True
                    }
                    self._save_active_orders()
                    self._placed_count += 1
                    self._record_placement_success()
                    self._add_audit_entry("place_move_stop", symbol, order_id,
                                          {"trigger_price": trigger_price, "callback_rate": callback_rate,
                                           "quantity": quantity, "side": side})
                    logger.info(f"Move stop placed for {symbol}: trigger={trigger_price:.4f}, callback={callback_rate:.2%} (algoId={order_id})")
                    return order_id
                # 成功但无 order_id（API 返回格式变化），保守按失败处理
                self._failed_placement_count += 1
                self._record_placement_failure()
                logger.warning(f"Move stop placed but no algoId returned: {result}")
            else:
                self._failed_placement_count += 1
                self._record_placement_failure()
                logger.warning(f"Move stop placement failed: {symbol} "
                               f"sCode={(result or {}).get('sCode')} sMsg={(result or {}).get('sMsg')}")
            return None
        except Exception as e:
            self._categorize_error(e)
            self._failed_placement_count += 1
            self._record_placement_failure()
            logger.error(f"Failed to place move stop: {e}")
            return None

    async def update_stop_loss_after_tp(self, symbol: str, tp_type: str = "tp1"):
        """
        止盈触发后自动上移止损：
        - TP1 触发：止损移到保本价（入场价 + 手续费缓冲）
        - TP2 触发：止损移到 TP1 价格（锁定大部分利润）
        
        会取消旧的 SL 条件单并挂新的
        """
        try:
            # 找到该币种当前的 SL 单
            sl_orders = [
                (oid, info) for oid, info in self._active_orders.items()
                if info["symbol"] == symbol and info["type"] == "stop_loss"
            ]
            if not sl_orders:
                logger.debug(f"No SL order found for {symbol} to update after TP")
                return None

            # 找到对应的持仓获取入场价
            positions = self._okx_client.get_positions()
            target_pos = None
            for p in positions or []:
                if p.get("instId") == symbol:
                    target_pos = p
                    break

            if not target_pos:
                logger.warning(f"No position found for {symbol} when updating SL after TP")
                return None

            side = target_pos.get("posSide", "net")
            avg_price = float(target_pos.get("avgPx", 0))
            mark_price = float(target_pos.get("markPx", 0))
            leverage = int(float(target_pos.get("lever", 1)))
            # 账本一致性（修复4）：pos 是 OKX 合约张数，转币数再交 place_stop_loss
            quantity = self._okx_client.contracts_to_coins(symbol, abs(float(target_pos.get("pos", 0))))

            if avg_price <= 0 or quantity <= 0:
                return None

            # 计算新的止损价
            # 保本缓冲按真实往返 taker 手续费折算（含安全系数），取代硬编码 0.2%
            breakeven_buffer = self._tp_sl_monitor.breakeven_buffer()
            if tp_type == "tp1":
                # TP1后止损移到保本（含手续费缓冲）
                if side == "long":
                    new_sl_price = avg_price * (1 + breakeven_buffer)
                else:
                    new_sl_price = avg_price * (1 - breakeven_buffer)
            elif tp_type == "tp2":
                # TP2后止损移到TP1价格（假设TP1约为60%目标），并保证不低于保本缓冲
                if side == "long":
                    tp1_estimate = avg_price + (mark_price - avg_price) * 0.6
                    new_sl_price = max(tp1_estimate, avg_price * (1 + breakeven_buffer))
                else:
                    tp1_estimate = avg_price - (avg_price - mark_price) * 0.6
                    new_sl_price = min(tp1_estimate, avg_price * (1 - breakeven_buffer))
            else:
                return None

            # 检查新止损价是否有效（不会立即触发）
            if self._check_immediate_trigger(symbol, side, new_sl_price):
                logger.warning(f"New SL would immediately trigger for {symbol}, skipping update")
                return None

            # 取消旧SL单，挂新SL单
            for old_id, _ in sl_orders:
                self.cancel_conditional_order(symbol, old_id)

            new_order_id = await self.place_stop_loss(
                symbol=symbol,
                side=side,
                quantity=quantity,
                trigger_price=round(new_sl_price, 4),
                leverage=leverage,
                is_new_position=False
            )

            if new_order_id:
                logger.info(f"SL uplifted after {tp_type} for {symbol}: new_sl={new_sl_price:.4f}")
            return new_order_id

        except Exception as e:
            logger.error(f"Failed to update SL after TP for {symbol}: {e}")
            return None

    async def heartbeat_check(self):
        """
        条件单心跳检测：确保所有持仓都有对应的止损条件单，
        丢失的自动补挂（交易所API有时会静默丢失条件单）
        """
        if not self._heartbeat_enabled:
            return 0
        try:
            positions = self._okx_client.get_positions() or []
            restored_count = 0

            for pos_data in positions:
                try:
                    inst_id = pos_data.get("instId", "")
                    pos_side = pos_data.get("posSide", "net")
                    # 账本一致性（修复4）：pos 是 OKX 合约张数，转币数再交 place_stop_loss
                    pos_qty = self._okx_client.contracts_to_coins(inst_id, abs(float(pos_data.get("pos", 0))))
                    if pos_qty <= 0:
                        continue

                    # 检查是否有活跃的SL单
                    has_sl = any(
                        info["symbol"] == inst_id and info["type"] in ("stop_loss", "move_stop")
                        for info in self._active_orders.values()
                    )

                    if not has_sl:
                        mark_price = float(pos_data.get("markPx", 0))
                        leverage = int(float(pos_data.get("lever", 1)))
                        if mark_price > 0:
                            stop_price = self._calculate_stop_price(pos_side, mark_price, leverage)
                            if stop_price and not self._check_immediate_trigger(inst_id, pos_side, stop_price):
                                result = await self.place_stop_loss(
                                    symbol=inst_id,
                                    side=pos_side,
                                    quantity=pos_qty,
                                    trigger_price=stop_price,
                                    leverage=leverage,
                                    is_new_position=False
                                )
                                if result:
                                    restored_count += 1
                                    logger.warning(f"Restored missing SL for {inst_id}: {stop_price:.4f}")
                except Exception as inner_e:
                    logger.error(f"Heartbeat check error for position: {inner_e}")

            if restored_count > 0:
                logger.info(f"Heartbeat check: restored {restored_count} missing stop loss orders")
            self._heartbeat_restored_count += restored_count
            return restored_count

        except Exception as e:
            logger.error(f"Heartbeat check failed: {e}")
            return 0

    # ==================== P2: 孤儿条件单清理 ====================

    async def _orphan_cleanup_loop(self):
        """定期清理孤儿条件单（持仓已不存在但条件单仍在）"""
        while self._running:
            try:
                await self.cleanup_orphaned_orders()
            except Exception as e:
                logger.error(f"Orphan cleanup loop error: {e}")
            await asyncio.sleep(120)  # 每2分钟检查一次

    async def cleanup_orphaned_orders(self) -> int:
        """
        清理孤儿条件单：取消所有持仓已不存在的条件单
        
        返回清理的订单数量
        """
        try:
            positions = self._okx_client.get_positions() or []
            active_symbols = set()
            for p in positions:
                inst_id = p.get("instId", "")
                pos_qty = abs(float(p.get("pos", 0)))
                if inst_id and pos_qty > 0:
                    active_symbols.add(inst_id)

            cleaned_count = 0
            orphan_ids = []

            # 检查活跃条件单
            for order_id, order_info in list(self._active_orders.items()):
                symbol = order_info.get("symbol", "")
                if symbol and symbol not in active_symbols:
                    # 该持仓已不存在，取消条件单
                    try:
                        if order_info.get("is_algo"):
                            self._okx_client.cancel_algo_order(symbol, order_id)
                        else:
                            self._okx_client.cancel_order(symbol, order_id)
                    except Exception:
                        pass  # 交易所可能已经取消了
                    orphan_ids.append(order_id)
                    cleaned_count += 1
                    logger.info(f"Cleaned orphaned conditional order: {order_id} for {symbol} (position gone)")

            # 清理失败订单列表中超过24小时的旧记录
            stale_failed_ids = []
            now = time.time()
            for order_id, order_info in list(self._failed_orders.items()):
                failed_at = order_info.get("failed_at", 0)
                if isinstance(failed_at, str):
                    try:
                        failed_at = datetime.fromisoformat(failed_at).timestamp()
                    except (ValueError, TypeError):
                        failed_at = 0
                if now - failed_at > 86400:  # 24小时
                    stale_failed_ids.append(order_id)

            for order_id in stale_failed_ids:
                self._failed_orders.pop(order_id, None)
                cleaned_count += 1

            # 清理待重试列表中超过1小时且重试次数已用完的
            stale_pending_ids = []
            for order_id, order_info in list(self._pending_orders.items()):
                retry_count = order_info.get("retry_count", 0)
                created_at = order_info.get("created_at", 0)
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at).timestamp()
                    except (ValueError, TypeError):
                        created_at = 0
                if retry_count >= self._max_retries and now - created_at > 3600:
                    stale_pending_ids.append(order_id)

            for order_id in stale_pending_ids:
                self._failed_orders[order_id] = self._pending_orders.pop(order_id, {})
                self._failed_orders[order_id]["failed_at"] = now
                cleaned_count += 1

            # 从活跃订单中移除孤儿
            for order_id in orphan_ids:
                self._active_orders.pop(order_id, None)

            if cleaned_count > 0:
                self._save_active_orders()
                logger.info(f"Orphan cleanup: removed {cleaned_count} stale/failed/orphaned orders "
                           f"(orphans={len(orphan_ids)}, stale_failed={len(stale_failed_ids)}, "
                           f"stale_pending={len(stale_pending_ids)})")

            return cleaned_count
        except Exception as e:
            logger.error(f"Failed to cleanup orphaned orders: {e}")
            return 0

    def get_sl_orders_for_symbol(self, symbol: str) -> List[Tuple[str, Dict[str, Any]]]:
        """获取指定币种的所有止损单"""
        return [
            (oid, info) for oid, info in self._active_orders.items()
            if info["symbol"] == symbol and info["type"] in ("stop_loss", "move_stop")
        ]

    def get_tp_orders_for_symbol(self, symbol: str) -> List[Tuple[str, Dict[str, Any]]]:
        """获取指定币种的所有止盈单"""
        return [
            (oid, info) for oid, info in self._active_orders.items()
            if info["symbol"] == symbol and info["type"] == "take_profit"
        ]

    def get_active_orders(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._active_orders)

    def get_protection_view(self) -> Dict[str, Any]:
        """统一止盈止损保护视图（单一口径，供 scheduler / dashboard 复用）。

        把交易所侧条件单（SL/TP/move_stop）与当前持仓按 (instId, posSide) 对齐，
        输出每个持仓是否已挂止损/止盈，以及「未受保护持仓」清单与汇总统计。
        纯只读聚合，不参与下单路径。
        """
        try:
            positions = self._okx_client.get_positions() or []
            algo_orders = self._okx_client.get_algo_orders() or []
        except Exception as e:
            logger.warning(f"get_protection_view failed to fetch positions/orders: {e}")
            positions, algo_orders = [], []
        return self._tp_sl_monitor.build_protection_view(positions, algo_orders)

    def get_pending_orders(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._pending_orders)

    def get_failed_orders(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._failed_orders)

    def get_detailed_stats(self) -> Dict[str, Any]:
        """获取详细的条件单统计（按类型分组）"""
        sl_count = 0
        tp_count = 0
        move_stop_count = 0
        by_symbol: Dict[str, Dict[str, int]] = {}

        for order_info in self._active_orders.values():
            symbol = order_info["symbol"]
            otype = order_info["type"]
            if symbol not in by_symbol:
                by_symbol[symbol] = {"sl": 0, "tp": 0, "move_stop": 0}

            if otype == "stop_loss":
                sl_count += 1
                by_symbol[symbol]["sl"] += 1
            elif otype == "take_profit":
                tp_count += 1
                by_symbol[symbol]["tp"] += 1
            elif otype == "move_stop":
                move_stop_count += 1
                by_symbol[symbol]["move_stop"] += 1

        return {
            "active_sl_count": sl_count,
            "active_tp_count": tp_count,
            "active_move_stop_count": move_stop_count,
            "total_active": len(self._active_orders),
            "pending_count": len(self._pending_orders),
            "failed_count": len(self._failed_orders),
            "by_symbol": by_symbol,
        }

    def get_stats(self) -> Dict[str, Any]:
        """获取条件单统计信息"""
        return self.get_detailed_stats()

    # ==================== 错误分类 ====================

    def _categorize_error(self, error: Exception) -> str:
        """
        将错误分类为：network, auth, rate_limit, order_rejected, unknown
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()

        # 网络错误
        if any(kw in error_str for kw in ["timeout", "connection", "network", "dns", "reset", "refused",
                                            "broken pipe", "eof", "httperror", "socket"]):
            category = "network"
        elif "connection" in error_type:
            category = "network"
        # 认证错误
        elif any(kw in error_str for kw in ["auth", "unauthorized", "signature", "api key", "apikey",
                                              "invalid sign", "login", "credential", "forbidden"]):
            category = "auth"
        # 频率限制
        elif any(kw in error_str for kw in ["rate limit", "too many requests", "throttle", "429",
                                              "exceed", "frequency", "request limit"]):
            category = "rate_limit"
        # 订单被拒
        elif any(kw in error_str for kw in ["order would", "immediately trigger", "insufficient",
                                              "balance", "margin", "position", "order rejected",
                                              "-2021", "-2022", "invalid order", "not allowed",
                                              "cancel", "filled", "already"]):
            category = "order_rejected"
        else:
            category = "unknown"

        self._error_counts[category] = self._error_counts.get(category, 0) + 1
        return category

    # ==================== 审计日志 ====================

    def _add_audit_entry(self, action: str, symbol: str, order_id: str = "",
                         details: Dict[str, Any] = None):
        """添加审计日志条目，最多保留 _audit_log_max 条"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "symbol": symbol,
            "order_id": order_id,
            "details": details or {},
        }
        self._audit_log.append(entry)
        if len(self._audit_log) > self._audit_log_max:
            self._audit_log = self._audit_log[-self._audit_log_max:]

    def get_audit_trail(self, symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取审计日志

        参数：
        - symbol: 筛选指定币种，None 返回全部
        - limit: 返回条数上限
        """
        entries = self._audit_log
        if symbol:
            entries = [e for e in entries if e["symbol"] == symbol]
        return entries[-limit:]

    # ==================== 热更新 ====================

    def update_config(self, config: Dict[str, Any]):
        """
        热更新配置，支持更新：
        - retry_interval: 重试间隔
        - max_retries: 最大重试次数
        - heartbeat_enabled: 心跳检测开关
        - sync_enabled: 同步循环开关

        所有变更会被记录到日志。
        """
        changes = []

        if "retry_interval" in config:
            new_val = int(config["retry_interval"])
            if new_val != self._retry_interval:
                changes.append(f"retry_interval: {self._retry_interval} -> {new_val}")
                self._retry_interval = new_val

        if "max_retries" in config:
            new_val = int(config["max_retries"])
            if new_val != self._max_retries:
                changes.append(f"max_retries: {self._max_retries} -> {new_val}")
                self._max_retries = new_val

        if "heartbeat_enabled" in config:
            new_val = bool(config["heartbeat_enabled"])
            if new_val != self._heartbeat_enabled:
                changes.append(f"heartbeat_enabled: {self._heartbeat_enabled} -> {new_val}")
                self._heartbeat_enabled = new_val

        if "sync_enabled" in config:
            new_val = bool(config["sync_enabled"])
            if new_val != self._sync_enabled:
                changes.append(f"sync_enabled: {self._sync_enabled} -> {new_val}")
                self._sync_enabled = new_val

        if changes:
            logger.info(f"ConditionalOrderManager config updated: {', '.join(changes)}")
        else:
            logger.debug("ConditionalOrderManager config update: no changes detected")

    # ==================== 状态持久化（带版本号） ====================

    def collect_persistent_state(self) -> Dict[str, Any]:
        """收集所有需要持久化的状态，包含版本号和时间戳"""
        return {
            "version": 2,
            "saved_at": datetime.now().isoformat(),
            "active_orders": dict(self._active_orders),
            "pending_orders": dict(self._pending_orders),
            "failed_orders": dict(self._failed_orders),
        }

    def restore_persistent_state(self, state: Dict[str, Any]):
        """从持久化数据恢复状态"""
        if not isinstance(state, dict):
            logger.warning("restore_persistent_state: invalid state format")
            return

        self._active_orders = state.get("active_orders", {})
        self._pending_orders = state.get("pending_orders", {})
        self._failed_orders = state.get("failed_orders", {})

        saved_version = state.get("version", 1)
        saved_at = state.get("saved_at", "unknown")
        logger.info(
            f"Restored persistent state v{saved_version} from {saved_at}: "
            f"{len(self._active_orders)} active, {len(self._pending_orders)} pending, "
            f"{len(self._failed_orders)} failed"
        )

    # ==================== 批量操作 ====================

    async def batch_cancel_orders(self, symbols: List[str]) -> Dict[str, Any]:
        """
        批量取消指定币种的所有条件单

        返回：
        {
            "success": 成功取消数,
            "failed": 失败取消数,
            "by_symbol": {symbol: {"success": int, "failed": int}}
        }
        """
        results = {"success": 0, "failed": 0, "by_symbol": {}}
        symbols_set = set(symbols)

        # 收集需要取消的订单
        orders_to_cancel: List[Tuple[str, Dict[str, Any]]] = []
        for order_id, order_info in self._active_orders.items():
            if order_info["symbol"] in symbols_set:
                orders_to_cancel.append((order_id, order_info))

        for order_id, order_info in orders_to_cancel:
            symbol = order_info["symbol"]
            if symbol not in results["by_symbol"]:
                results["by_symbol"][symbol] = {"success": 0, "failed": 0}

            try:
                if order_info.get("is_algo"):
                    result = self._okx_client.cancel_algo_order(symbol, order_id)
                else:
                    result = self._okx_client.cancel_order(symbol, order_id)

                if result:
                    self._active_orders.pop(order_id, None)
                    results["success"] += 1
                    results["by_symbol"][symbol]["success"] += 1
                    self._add_audit_entry("batch_cancel", symbol, order_id, {"status": "success"})
                else:
                    results["failed"] += 1
                    results["by_symbol"][symbol]["failed"] += 1
                    self._add_audit_entry("batch_cancel", symbol, order_id, {"status": "failed"})
            except Exception as e:
                self._categorize_error(e)
                results["failed"] += 1
                results["by_symbol"][symbol]["failed"] += 1
                logger.error(f"Batch cancel failed for {symbol} order {order_id}: {e}")

        self._save_active_orders()
        logger.info(
            f"Batch cancel completed: {results['success']} success, {results['failed']} failed "
            f"across {len(symbols)} symbols"
        )
        return results

    async def batch_place_stop_losses(self, positions: List[Dict]) -> Dict[str, Any]:
        """
        批量放置止损单

        参数 positions 中每个元素应包含：
        - symbol, side, quantity, trigger_price, leverage, is_new_position(可选)

        返回：
        {
            "success": 成功数,
            "failed": 失败数,
            "order_ids": [成功创建的 order_id 列表],
            "by_symbol": {symbol: {"success": bool, "order_id": str}}
        }
        """
        results = {"success": 0, "failed": 0, "order_ids": [], "by_symbol": {}}

        for pos in positions:
            symbol = pos.get("symbol", "")
            side = pos.get("side", "long")
            quantity = float(pos.get("quantity", 0))
            trigger_price = float(pos.get("trigger_price", 0))
            leverage = int(pos.get("leverage", 1))
            is_new_position = pos.get("is_new_position", True)

            results["by_symbol"][symbol] = {"success": False, "order_id": None}

            if quantity <= 0 or trigger_price <= 0:
                logger.warning(f"Batch SL: skipping {symbol} due to invalid params (qty={quantity}, tp={trigger_price})")
                results["failed"] += 1
                continue

            try:
                order_id = await self.place_stop_loss(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    trigger_price=trigger_price,
                    leverage=leverage,
                    is_new_position=is_new_position,
                )
                if order_id:
                    results["success"] += 1
                    results["order_ids"].append(order_id)
                    results["by_symbol"][symbol]["success"] = True
                    results["by_symbol"][symbol]["order_id"] = order_id
                else:
                    results["failed"] += 1
            except Exception as e:
                self._categorize_error(e)
                results["failed"] += 1
                logger.error(f"Batch SL failed for {symbol}: {e}")

        logger.info(
            f"Batch stop loss completed: {results['success']} success, {results['failed']} failed "
            f"across {len(positions)} positions"
        )
        return results

    async def batch_place_take_profits(self, positions: List[Dict]) -> Dict[str, Any]:
        """
        批量放置止盈单

        参数 positions 中每个元素应包含：
        - symbol, side, quantity, trigger_price, leverage

        返回：
        {
            "success": 成功数,
            "failed": 失败数,
            "order_ids": [成功创建的 order_id 列表],
            "by_symbol": {symbol: {"success": bool, "order_id": str}}
        }
        """
        results = {"success": 0, "failed": 0, "order_ids": [], "by_symbol": {}}

        for pos in positions:
            symbol = pos.get("symbol", "")
            side = pos.get("side", "long")
            quantity = float(pos.get("quantity", 0))
            trigger_price = float(pos.get("trigger_price", 0))
            leverage = int(pos.get("leverage", 1))

            results["by_symbol"][symbol] = {"success": False, "order_id": None}

            if quantity <= 0 or trigger_price <= 0:
                logger.warning(f"Batch TP: skipping {symbol} due to invalid params (qty={quantity}, tp={trigger_price})")
                results["failed"] += 1
                continue

            try:
                order_id = await self.place_take_profit(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    trigger_price=trigger_price,
                    leverage=leverage,
                )
                if order_id:
                    results["success"] += 1
                    results["order_ids"].append(order_id)
                    results["by_symbol"][symbol]["success"] = True
                    results["by_symbol"][symbol]["order_id"] = order_id
                else:
                    results["failed"] += 1
            except Exception as e:
                self._categorize_error(e)
                results["failed"] += 1
                logger.error(f"Batch TP failed for {symbol}: {e}")

        logger.info(
            f"Batch take profit completed: {results['success']} success, {results['failed']} failed "
            f"across {len(positions)} positions"
        )
        return results

    # ==================== 增强指标 ====================

    def get_enhanced_stats(self) -> Dict[str, Any]:
        """获取增强统计信息，包含成功率、重试统计、心跳恢复、同步次数、每币种订单数、运行时长等"""
        total_placed = self._placed_count + self._failed_placement_count
        success_rate = (self._placed_count / total_placed * 100) if total_placed > 0 else 0.0

        retry_success_rate = (
            (self._retry_success_count / self._retry_total_count * 100)
            if self._retry_total_count > 0
            else 0.0
        )

        avg_retry_count = (
            (self._retry_total_count / self._failed_placement_count)
            if self._failed_placement_count > 0
            else 0.0
        )

        # 每币种订单数
        per_symbol: Dict[str, int] = {}
        for info in self._active_orders.values():
            sym = info["symbol"]
            per_symbol[sym] = per_symbol.get(sym, 0) + 1

        uptime_seconds = time.time() - self._uptime_start

        return {
            # 基础统计
            "active_orders": len(self._active_orders),
            "pending_orders": len(self._pending_orders),
            "failed_orders": len(self._failed_orders),
            # 成功率
            "placed_count": self._placed_count,
            "failed_placement_count": self._failed_placement_count,
            "success_rate": round(success_rate, 2),
            # 重试统计
            "retry_success_count": self._retry_success_count,
            "retry_total_count": self._retry_total_count,
            "retry_success_rate": round(retry_success_rate, 2),
            "avg_retry_count": round(avg_retry_count, 2),
            # 心跳和同步
            "heartbeat_restored_count": self._heartbeat_restored_count,
            "sync_operations_count": self._sync_operations_count,
            # 每币种
            "per_symbol": per_symbol,
            # 运行时长
            "uptime_seconds": round(uptime_seconds, 1),
            "uptime_human": self._format_uptime(uptime_seconds),
            # 错误分类
            "error_counts": dict(self._error_counts),
            # 企业级强化：熔断器状态
            "circuit_breaker": {
                "open": self._circuit_open,
                "opened_at": self._circuit_opened_at,
                "consecutive_failures": self._consecutive_failures,
                "threshold": self._circuit_threshold,
                "cooldown_seconds": self._circuit_cooldown,
            },
            # 审计日志条数
            "audit_entries": len(self._audit_log),
            # 配置状态
            "config": {
                "retry_interval": self._retry_interval,
                "max_retries": self._max_retries,
                "heartbeat_enabled": self._heartbeat_enabled,
                "sync_enabled": self._sync_enabled,
            },
        }

    def _format_uptime(self, seconds: float) -> str:
        """将秒数格式化为人类可读的运行时长"""
        days, rem = divmod(int(seconds), 86400)
        hours, rem = divmod(rem, 3600)
        minutes, secs = divmod(rem, 60)
        parts = []
        if days:
            parts.append(f"{days}d")
        if hours:
            parts.append(f"{hours}h")
        if minutes:
            parts.append(f"{minutes}m")
        parts.append(f"{secs}s")
        return " ".join(parts)