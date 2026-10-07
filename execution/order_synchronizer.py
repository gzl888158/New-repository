"""
订单状态同步器
实时同步已成交、挂单、撤单，本地与交易所仓位实时对账

核心功能：
1. WebSocket实时订单推送处理
2. 定期全量订单同步（防遗漏）
3. 本地-交易所仓位实时对账
4. 订单状态机管理
5. 异常订单检测与自动修复
6. 对账差异告警
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class OrderState(Enum):
    """订单状态"""
    UNKNOWN = "unknown"
    LIVE = "live"              # 挂单中
    PARTIALLY_FILLED = "partially_filled"  # 部分成交
    FILLED = "filled"          # 完全成交
    CANCELLED = "cancelled"    # 已撤销
    FAILED = "failed"          # 失败


class PositionSide(Enum):
    """持仓方向"""
    LONG = "long"
    SHORT = "short"
    NET = "net"


@dataclass
class OrderInfo:
    """订单信息"""
    order_id: str
    symbol: str
    side: str               # buy/sell
    pos_side: str           # long/short/net
    order_type: str         # limit/market/...
    price: float
    quantity: float
    filled_qty: float = 0.0
    avg_price: float = 0.0
    state: OrderState = OrderState.UNKNOWN
    create_time: float = 0.0
    update_time: float = 0.0
    strategy: str = ""
    client_order_id: str = ""
    fee: float = 0.0
    fee_ccy: str = ""
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PositionInfo:
    """持仓信息"""
    symbol: str
    pos_side: str           # long/short
    quantity: float = 0.0
    avg_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    leverage: int = 1
    margin: float = 0.0
    margin_ratio: float = 0.0
    liq_price: float = 0.0
    last_update: float = 0.0


@dataclass
class ReconciliationResult:
    """对账结果"""
    timestamp: float
    orders_checked: int = 0
    positions_checked: int = 0
    order_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    position_mismatches: List[Dict[str, Any]] = field(default_factory=list)
    missing_local_orders: List[str] = field(default_factory=list)
    missing_exchange_orders: List[str] = field(default_factory=list)
    is_consistent: bool = True


class OrderStateSynchronizer:
    """
    订单状态同步器
    
    核心设计：
    - WebSocket实时推送 + 定期全量同步双保险
    - 本地状态与交易所状态实时对账
    - 差异自动检测与告警
    - 支持状态回调通知上层模块
    """

    def __init__(self, config: Dict[str, Any], okx_client=None, alert_manager=None):
        self.config = config
        self.okx_client = okx_client
        self.alert_manager = alert_manager
        self._position_manager = None  # 用于检查 WS 同步健康度
        
        sync_config = config.get("execution", {}).get("order_synchronizer") or {}
        if not isinstance(sync_config, dict):
            sync_config = {}
        self._enabled = sync_config.get("enabled", True)
        self._full_sync_interval = sync_config.get("full_sync_interval", 60)       # 全量同步间隔60秒
        self._reconcile_interval = sync_config.get("reconcile_interval", 30)       # 对账间隔30秒
        self._max_order_age = sync_config.get("max_order_age_days", 7) * 86400     # 订单历史保留7天
        self._auto_repair = sync_config.get("auto_repair", True)                   # 对账差异自动修复（以交易所为准）
        self._sync_grace_period = sync_config.get("sync_grace_period", 5.0)        # WS 同步宽限期（秒）
        
        # 本地订单缓存 {order_id: OrderInfo}
        self._orders: Dict[str, OrderInfo] = {}
        
        # 本地持仓缓存 {symbol: {pos_side: PositionInfo}}
        self._positions: Dict[str, Dict[str, PositionInfo]] = {}
        
        # 挂单订单ID集合（加速查询）
        self._live_orders: Set[str] = set()
        
        # 订单更新回调列表
        self._order_callbacks: List[callable] = []
        
        # 持仓更新回调列表
        self._position_callbacks: List[callable] = []
        
        # 对账历史（最近100次）
        self._reconciliation_history: deque = deque(maxlen=100)
        
        # 运行状态
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # 统计信息
        self._stats = {
            "total_order_updates": 0,
            "total_position_updates": 0,
            "total_reconciliations": 0,
            "total_mismatches": 0,
            "last_full_sync": 0.0,
            "last_reconcile": 0.0,
            "ws_order_updates": 0,
            "rest_order_updates": 0,
        }
        
        # 锁
        self._order_lock = asyncio.Lock()
        self._position_lock = asyncio.Lock()

        # ── 企业级同步引擎 ──
        self._sync_engine = None
        try:
            from core.enterprise_sync import get_sync_engine
            self._sync_engine = get_sync_engine(config)
            self._sync_engine.register_channel(
                "order_rest", self,
                expected_interval=self._full_sync_interval,
                recovery_callback=lambda: asyncio.ensure_future(self._full_sync_orders())
            )
            logger.info("OrderStateSynchronizer: EnterpriseSyncEngine integrated")
        except Exception as e:
            logger.debug(f"OrderStateSynchronizer: EnterpriseSyncEngine skipped: {e}")

        logger.info(f"OrderStateSynchronizer initialized: "
                   f"full_sync_interval={self._full_sync_interval}s, "
                   f"reconcile_interval={self._reconcile_interval}s")

    def set_position_manager(self, position_manager):
        """注入 PositionManager 用于检查 WS 同步健康度"""
        self._position_manager = position_manager

    async def start(self):
        """启动同步器"""
        if self._running:
            return
        
        self._running = True
        
        # 启动全量同步任务
        self._tasks.append(asyncio.create_task(self._full_sync_loop()))
        
        # 启动对账任务
        self._tasks.append(asyncio.create_task(self._reconciliation_loop()))
        
        # 启动旧数据清理任务
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))
        
        logger.info("OrderStateSynchronizer started")

    async def stop(self):
        """停止同步器"""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
        
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("OrderStateSynchronizer stopped")

    def register_order_callback(self, callback: callable):
        """注册订单更新回调"""
        self._order_callbacks.append(callback)

    def register_position_callback(self, callback: callable):
        """注册持仓更新回调"""
        self._position_callbacks.append(callback)

    async def handle_ws_order_update(self, order_data: Dict[str, Any]):
        """
        处理WebSocket订单推送
        
        Args:
            order_data: OKX WebSocket订单数据
        """
        if not self._enabled:
            return
        
        try:
            order_id = order_data.get("ordId")
            if not order_id:
                return
            
            async with self._order_lock:
                order = self._parse_order_data(order_data)
                self._orders[order_id] = order
                
                # 更新挂单集合
                if order.state in (OrderState.LIVE, OrderState.PARTIALLY_FILLED):
                    self._live_orders.add(order_id)
                else:
                    self._live_orders.discard(order_id)
            
            self._stats["total_order_updates"] += 1
            self._stats["ws_order_updates"] += 1
            
            # 触发回调
            await self._notify_order_callbacks(order)
            
            logger.debug(f"WS order update: {order_id} {order.state.value}")
            
        except Exception as e:
            logger.error(f"Error handling WS order update: {e}")

    async def handle_ws_position_update(self, position_data: Dict[str, Any]):
        """
        处理WebSocket持仓推送
        
        Args:
            position_data: OKX WebSocket持仓数据
        """
        if not self._enabled:
            return
        
        try:
            symbol = position_data.get("instId")
            pos_side = position_data.get("posSide", "net")
            
            if not symbol:
                return
            
            async with self._position_lock:
                pos = self._parse_position_data(position_data)
                
                if symbol not in self._positions:
                    self._positions[symbol] = {}
                
                self._positions[symbol][pos_side] = pos
            
            self._stats["total_position_updates"] += 1
            
            # 触发回调
            await self._notify_position_callbacks(symbol, pos_side)
            
            logger.debug(f"WS position update: {symbol} {pos_side} qty={pos.quantity}")
            
        except Exception as e:
            logger.error(f"Error handling WS position update: {e}")

    async def _full_sync_loop(self):
        """全量同步循环"""
        while self._running:
            try:
                await asyncio.sleep(self._full_sync_interval)
                
                if not self.okx_client:
                    continue
                
                await self._full_sync_orders()
                await self._full_sync_positions()
                
                self._stats["last_full_sync"] = time.time()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in full sync loop: {e}")
                await asyncio.sleep(10)

    async def _full_sync_orders(self):
        """全量同步订单（挂单+最近成交）"""
        try:
            # 同步挂单
            pending_orders = await asyncio.to_thread(
                self.okx_client.get_orders
            )
            
            if pending_orders and isinstance(pending_orders, list):
                async with self._order_lock:
                    new_live = set()
                    for od in pending_orders:
                        order_id = od.get("ordId")
                        if order_id:
                            order = self._parse_order_data(od)
                            self._orders[order_id] = order
                            new_live.add(order_id)
                    
                    # 更新挂单集合
                    self._live_orders = new_live
                
                self._stats["rest_order_updates"] += len(pending_orders)
                logger.info(f"Full sync: {len(pending_orders)} pending orders")
            
            # 同步最近成交订单（最近100条）
            recent_orders = await asyncio.to_thread(
                self.okx_client.get_order_history,
                100
            )
            
            if recent_orders and isinstance(recent_orders, list):
                async with self._order_lock:
                    for od in recent_orders:
                        order_id = od.get("ordId")
                        if order_id and order_id not in self._orders:
                            order = self._parse_order_data(od)
                            self._orders[order_id] = order
                
                logger.info(f"Full sync: {len(recent_orders)} recent orders")

            # ── 企业级同步：记录订单 REST 同步事件 ──
            self._record_sync_event("order_rest", success=True,
                                    entities=list(self._orders.keys()))
                
        except Exception as e:
            logger.error(f"Error in full sync orders: {e}")
            self._record_sync_event("order_rest", success=False, error=str(e))

    async def _full_sync_positions(self):
        """全量同步持仓"""
        try:
            positions = await asyncio.to_thread(
                self.okx_client.get_positions
            )
            
            if positions and isinstance(positions, list):
                async with self._position_lock:
                    new_positions = {}
                    for pd in positions:
                        symbol = pd.get("instId")
                        pos_side = pd.get("posSide", "net")
                        
                        if not symbol:
                            continue
                        
                        if symbol not in new_positions:
                            new_positions[symbol] = {}
                        
                        pos = self._parse_position_data(pd)
                        new_positions[symbol][pos_side] = pos
                    
                    self._positions = new_positions
                
                logger.info(f"Full sync: {len(positions)} positions")

            # ── 企业级同步：记录持仓 REST 同步事件 ──
            entity_keys = [f"{s}:{ps}" for s, psd in self._positions.items() for ps in psd]
            self._record_sync_event("order_rest", success=True, entities=entity_keys)
                
        except Exception as e:
            logger.error(f"Error in full sync positions: {e}")
            self._record_sync_event("order_rest", success=False, error=str(e))

    def _record_sync_event(self, channel: str, success: bool = True,
                           entities: list = None, error: str = ""):
        """记录同步事件到企业级同步引擎"""
        if self._sync_engine:
            try:
                self._sync_engine.record_sync(
                    channel, success=success, entities=entities, error=error
                )
                if entities and success:
                    for entity in entities:
                        self._sync_engine.update_entity_version(
                            entity, source=channel, exchange_ts=time.time()
                        )
            except Exception:
                pass

    async def _reconciliation_loop(self):
        """对账循环"""
        while self._running:
            try:
                await asyncio.sleep(self._reconcile_interval)
                
                if not self.okx_client:
                    continue
                
                result = await self._perform_reconciliation()
                self._reconciliation_history.append(result)
                self._stats["total_reconciliations"] += 1
                self._stats["last_reconcile"] = time.time()
                
                if not result.is_consistent:
                    self._stats["total_mismatches"] += 1
                    logger.warning(f"Reconciliation mismatch found: "
                                  f"orders={len(result.order_mismatches)}, "
                                  f"positions={len(result.position_mismatches)}")
                    
                    # 发送告警（但检查 WS 同步健康度，避免误报）
                    if self.alert_manager:
                        # 如果持仓有差异但 position_ws 通道最近成功同步，可能是 REST 轮询时序差异，降级为 DEBUG
                        should_alert = True
                        if result.position_mismatches and self._position_manager and self._sync_engine:
                            try:
                                ws_health = self._sync_engine.get_channel_health("position_ws")
                                if ws_health and ws_health.is_healthy:
                                    # WS 通道健康，差异可能是时序问题，检查是否在宽限期内
                                    time_since_sync = time.time() - (ws_health.last_success_time or 0)
                                    if time_since_sync < self._sync_grace_period:
                                        should_alert = False
                                        logger.debug(
                                            f"Reconciliation position mismatch suppressed: "
                                            f"position_ws healthy (last sync {time_since_sync:.1f}s ago)"
                                        )
                            except Exception as e:
                                logger.debug(f"Failed to check position_ws health: {e}")
                        
                        if should_alert:
                            await self.alert_manager.send_alert(
                                "reconciliation_mismatch",
                                f"对账发现差异: 订单{len(result.order_mismatches)}个, "
                                f"持仓{len(result.position_mismatches)}个",
                                severity="WARNING",
                                metadata={
                                    "order_mismatches": len(result.order_mismatches),
                                    "position_mismatches": len(result.position_mismatches),
                                }
                            )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(10)

    async def _perform_reconciliation(self) -> ReconciliationResult:
        """执行对账（订单 + 持仓），并在 auto_repair 开启时以交易所为准主动修复差异"""
        result = ReconciliationResult(timestamp=time.time())
        repaired_orders = 0
        repaired_positions = 0

        try:
            # 获取交易所挂单（fail-closed：查询失败显式标记不一致，绝不当作一致）
            try:
                exchange_orders = await asyncio.to_thread(
                    self.okx_client.get_orders
                )
            except Exception as e:
                logger.error(f"Reconciliation: failed to fetch exchange orders: {e}")
                result.is_consistent = False
                result.order_mismatches.append({"type": "query_failed", "error": str(e)})
                exchange_orders = None

            if exchange_orders is not None and isinstance(exchange_orders, list):
                result.orders_checked = len(exchange_orders)

                exchange_order_ids = set()

                for od in exchange_orders:
                    order_id = od.get("ordId")
                    if not order_id:
                        continue

                    exchange_order_ids.add(order_id)

                    # 检查本地是否有此订单
                    local_order = self._orders.get(order_id)
                    if not local_order:
                        result.missing_local_orders.append(order_id)
                        result.is_consistent = False
                        # 主动修复：补录交易所存在但本地缺失的挂单
                        if self._auto_repair:
                            async with self._order_lock:
                                self._orders[order_id] = self._parse_order_data(od)
                                self._live_orders.add(order_id)
                            repaired_orders += 1
                        continue

                    # 检查状态是否一致
                    exchange_state = self._map_order_state(od.get("state", ""))
                    if local_order.state != exchange_state:
                        result.order_mismatches.append({
                            "order_id": order_id,
                            "symbol": od.get("instId"),
                            "local_state": local_order.state.value,
                            "exchange_state": exchange_state.value,
                            "type": "state_mismatch"
                        })
                        result.is_consistent = False
                        # 主动修复：以交易所为准更新本地订单状态
                        if self._auto_repair:
                            async with self._order_lock:
                                local_order.state = exchange_state
                                if exchange_state in (OrderState.LIVE, OrderState.PARTIALLY_FILLED):
                                    self._live_orders.add(order_id)
                                else:
                                    self._live_orders.discard(order_id)
                            repaired_orders += 1

                # 检查本地有但交易所没有的挂单
                for local_id in list(self._live_orders):
                    if local_id not in exchange_order_ids:
                        result.missing_exchange_orders.append(local_id)
                        result.is_consistent = False
                        # 主动修复：清理本地幽灵挂单（交易所已无此单）
                        if self._auto_repair:
                            async with self._order_lock:
                                self._live_orders.discard(local_id)
                                if local_id in self._orders and self._orders[local_id].state in (
                                        OrderState.LIVE, OrderState.PARTIALLY_FILLED):
                                    self._orders[local_id].state = OrderState.CANCELLED
                            repaired_orders += 1
            elif exchange_orders is not None:
                logger.error("Reconciliation: get_orders returned non-list response")
                result.is_consistent = False
                result.order_mismatches.append(
                    {"type": "invalid_response", "error": "get_orders returned non-list"}
                )

            # 持仓对账（fail-closed：查询失败显式标记不一致）
            try:
                exchange_positions = await asyncio.to_thread(
                    self.okx_client.get_positions
                )
            except Exception as e:
                logger.error(f"Reconciliation: failed to fetch exchange positions: {e}")
                result.is_consistent = False
                result.position_mismatches.append({"type": "query_failed", "error": str(e)})
                exchange_positions = None

            if exchange_positions is not None and isinstance(exchange_positions, list):
                result.positions_checked = len(exchange_positions)

                for pd in exchange_positions:
                    symbol = pd.get("instId")
                    pos_side = pd.get("posSide", "net")
                    exchange_qty = abs(float(pd.get("pos") or 0))

                    if not symbol or exchange_qty <= 0:
                        continue

                    # 检查本地持仓
                    local_pos = None
                    if symbol in self._positions and pos_side in self._positions[symbol]:
                        local_pos = self._positions[symbol][pos_side]

                    if not local_pos:
                        result.position_mismatches.append({
                            "symbol": symbol,
                            "pos_side": pos_side,
                            "local_qty": 0,
                            "exchange_qty": exchange_qty,
                            "type": "missing_local_position"
                        })
                        result.is_consistent = False
                        # 主动修复：补录交易所存在但本地缺失的持仓
                        if self._auto_repair:
                            async with self._position_lock:
                                self._positions.setdefault(symbol, {})[pos_side] = self._parse_position_data(pd)
                            repaired_positions += 1
                        continue

                    # 检查数量是否一致（允许微小差异）
                    local_qty = abs(local_pos.quantity)
                    if abs(local_qty - exchange_qty) > 0.0001:
                        result.position_mismatches.append({
                            "symbol": symbol,
                            "pos_side": pos_side,
                            "local_qty": local_qty,
                            "exchange_qty": exchange_qty,
                            "diff": local_qty - exchange_qty,
                            "type": "quantity_mismatch"
                        })
                        result.is_consistent = False
                        # 主动修复：以交易所为准更新本地持仓数量
                        if self._auto_repair:
                            async with self._position_lock:
                                local_pos.quantity = float(pd.get("pos") or 0)
                                local_pos.last_update = time.time()
                            repaired_positions += 1
            elif exchange_positions is not None:
                logger.error("Reconciliation: get_positions returned non-list response")
                result.is_consistent = False
                result.position_mismatches.append(
                    {"type": "invalid_response", "error": "get_positions returned non-list"}
                )

        except Exception as e:
            logger.error(f"Error performing reconciliation: {e}")
            result.is_consistent = False
            result.order_mismatches.append({"type": "reconciliation_error", "error": str(e)})

        if self._auto_repair and (repaired_orders or repaired_positions):
            logger.info(f"Reconciliation auto-repaired {repaired_orders} orders, "
                        f"{repaired_positions} positions")

        return result

    async def _cleanup_loop(self):
        """旧数据清理循环"""
        while self._running:
            try:
                # 每小时清理一次
                await asyncio.sleep(3600)
                
                cutoff = time.time() - self._max_order_age
                
                async with self._order_lock:
                    old_ids = [
                        oid for oid, order in self._orders.items()
                        if order.update_time > 0 and order.update_time < cutoff
                        and order.state not in (OrderState.LIVE, OrderState.PARTIALLY_FILLED)
                    ]
                    
                    for oid in old_ids:
                        del self._orders[oid]
                        self._live_orders.discard(oid)
                    
                    if old_ids:
                        logger.info(f"Cleaned up {len(old_ids)} old orders")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def _parse_order_data(self, data: Dict[str, Any]) -> OrderInfo:
        """解析订单数据"""
        state = self._map_order_state(data.get("state", ""))
        
        return OrderInfo(
            order_id=data.get("ordId", ""),
            symbol=data.get("instId", ""),
            side=data.get("side", ""),
            pos_side=data.get("posSide", "net"),
            order_type=data.get("ordType", ""),
            price=float(data.get("px", 0) or 0),
            quantity=float(data.get("sz", 0) or 0),
            filled_qty=float(data.get("accFillSz", 0) or 0),
            avg_price=float(data.get("avgPx", 0) or 0),
            state=state,
            create_time=float(data.get("cTime", 0)) / 1000 if data.get("cTime") else 0,
            update_time=float(data.get("uTime", 0)) / 1000 if data.get("uTime") else time.time(),
            strategy=data.get("tag", ""),
            client_order_id=data.get("clOrdId", ""),
            fee=float(data.get("fee", 0) or 0),
            fee_ccy=data.get("feeCcy", ""),
            raw_data=data,
        )

    def _parse_position_data(self, data: Dict[str, Any]) -> PositionInfo:
        """解析持仓数据"""
        pos_str = data.get("pos", "0")
        pos = float(pos_str) if pos_str else 0.0
        
        return PositionInfo(
            symbol=data.get("instId", ""),
            pos_side=data.get("posSide", "net"),
            quantity=pos,
            avg_price=float(data.get("avgPx", 0) or 0),
            unrealized_pnl=float(data.get("upl", 0) or 0),
            realized_pnl=float(data.get("realizedPnl", 0) or 0),
            leverage=int(float(data.get("lever", 1) or 1)),
            margin=float(data.get("margin", 0) or 0),
            margin_ratio=float(data.get("mgnRatio", 0) or 0),
            liq_price=float(data.get("liqPx", 0) or 0),
            last_update=time.time(),
        )

    def _map_order_state(self, state_str: str) -> OrderState:
        """映射OKX订单状态到内部状态"""
        state_map = {
            "live": OrderState.LIVE,
            "partially_filled": OrderState.PARTIALLY_FILLED,
            "filled": OrderState.FILLED,
            "canceled": OrderState.CANCELLED,
            "cancelled": OrderState.CANCELLED,
            "mmp_canceled": OrderState.CANCELLED,
            "failed": OrderState.FAILED,
        }
        return state_map.get(state_str.lower(), OrderState.UNKNOWN)

    async def _notify_order_callbacks(self, order: OrderInfo):
        """通知订单更新回调"""
        for callback in self._order_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(order)
                else:
                    callback(order)
            except Exception as e:
                logger.error(f"Error in order callback: {e}")

    async def _notify_position_callbacks(self, symbol: str, pos_side: str):
        """通知持仓更新回调"""
        pos = None
        if symbol in self._positions and pos_side in self._positions[symbol]:
            pos = self._positions[symbol][pos_side]
        
        for callback in self._position_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(symbol, pos_side, pos)
                else:
                    callback(symbol, pos_side, pos)
            except Exception as e:
                logger.error(f"Error in position callback: {e}")

    def get_order(self, order_id: str) -> Optional[OrderInfo]:
        """获取订单信息"""
        return self._orders.get(order_id)

    def get_live_orders(self, symbol: str = None) -> List[OrderInfo]:
        """获取挂单列表"""
        orders = [self._orders[oid] for oid in self._live_orders if oid in self._orders]
        
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        
        return orders

    def get_position(self, symbol: str, pos_side: str = "net") -> Optional[PositionInfo]:
        """获取持仓信息"""
        if symbol not in self._positions:
            return None
        return self._positions[symbol].get(pos_side)

    def get_all_positions(self) -> Dict[str, Dict[str, PositionInfo]]:
        """获取所有持仓"""
        return self._positions.copy()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self._stats,
            "cached_orders": len(self._orders),
            "live_orders": len(self._live_orders),
            "tracked_symbols": len(self._positions),
            "reconciliation_consistency_rate": (
                1.0 - self._stats["total_mismatches"] / max(self._stats["total_reconciliations"], 1)
            ),
        }

    def get_recent_reconciliations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的对账记录"""
        results = []
        for r in list(self._reconciliation_history)[-limit:]:
            results.append({
                "timestamp": r.timestamp,
                "orders_checked": r.orders_checked,
                "positions_checked": r.positions_checked,
                "order_mismatches": len(r.order_mismatches),
                "position_mismatches": len(r.position_mismatches),
                "is_consistent": r.is_consistent,
            })
        return results
