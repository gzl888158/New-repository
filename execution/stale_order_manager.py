"""
挂单时效管理器 - StaleOrderManager
解决长时间挂单不成交问题：
- 4级时效梯度检测 (5min/15min/30min/60min)
- 不成交原因诊断 (价格偏离/流动不足/方向错误/波动变化)
- 智能决策 (调价重挂/撤销等待/放弃)
- 统计追踪
"""
import asyncio
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger


class StaleLevel(Enum):
    """挂单时效等级"""
    NORMAL = 0       # < 5分钟，正常
    MILD = 1         # 5-15分钟，轻度超时
    MODERATE = 2    # 15-30分钟，中度超时
    SEVERE = 3      # 30-60分钟，重度超时
    CRITICAL = 4    # > 60分钟，严重超时


class StaleReason(Enum):
    """不成交原因"""
    PRICE_DEVIATION = "price_deviation"       # 挂单价偏离市场价太远
    LOW_LIQUIDITY = "low_liquidity"           # 盘口流动性不足
    WRONG_DIRECTION = "wrong_direction"       # 价格朝反方向运行
    VOLATILITY_SHIFT = "volatility_shift"     # 波动率大幅变化
    SPREAD_TOO_WIDE = "spread_too_wide"       # 买卖价差过大
    UNKNOWN = "unknown"                       # 未知原因


class StaleAction(Enum):
    """处理动作"""
    KEEP = "keep"                       # 保持挂单
    REPRICE = "reprice"                 # 撤销后重新定价挂单
    REPRICE_CLOSER = "reprice_closer"   # 撤销后以更接近市价重挂
    CANCEL_WAIT = "cancel_wait"         # 撤销并等待更好时机
    CANCEL_ABANDON = "cancel_abandon"   # 撤销并放弃该方向
    LOG_ONLY = "log_only"               # 仅记录


@dataclass
class StaleOrderInfo:
    """挂单时效信息"""
    order_id: str
    symbol: str
    side: str                       # buy/sell
    pos_side: str                   # long/short
    order_type: str                 # limit/market/conditional
    price: float
    quantity: float
    filled_qty: float = 0.0
    create_time: float = 0.0        # 挂单创建时间 (unix timestamp)
    strategy: str = ""
    first_detected_at: float = 0.0  # 首次检测到超时的时间
    stale_level: StaleLevel = StaleLevel.NORMAL
    stale_reason: StaleReason = StaleReason.UNKNOWN
    last_action: StaleAction = StaleAction.KEEP
    action_history: List[Dict] = field(default_factory=list)
    market_price_at_create: float = 0.0  # 挂单时的市场价
    repriced_count: int = 0          # 已重新定价次数
    new_order_id: str = ""           # 最近一次重挂后的新订单ID（闭环追踪）
    cancelled: bool = False          # 是否已被撤销
    leverage: int = 5                # 杠杆倍数（重挂时沿用原订单杠杆）

    @property
    def age_seconds(self) -> float:
        """挂单已存在秒数"""
        return time.time() - self.create_time

    @property
    def age_minutes(self) -> float:
        """挂单已存在分钟数"""
        return self.age_seconds / 60.0


@dataclass
class StaleStats:
    """挂单时效统计"""
    total_pending_checked: int = 0       # 总检查次数
    total_stale_detected: int = 0        # 总超时检测次数
    total_repriced: int = 0              # 总重新定价次数
    total_cancelled: int = 0             # 总撤销次数
    total_abandoned: int = 0             # 总放弃次数
    by_symbol: Dict[str, Dict[str, int]] = field(default_factory=dict)  # {symbol: {reason: count}}
    by_level: Dict[str, int] = field(default_factory=lambda: {
        "mild": 0, "moderate": 0, "severe": 0, "critical": 0
    })
    actions_by_type: Dict[str, int] = field(default_factory=lambda: {
        "reprice": 0, "reprice_closer": 0, "cancel_wait": 0, "cancel_abandon": 0
    })


class StaleOrderManager:
    """
    挂单时效管理器

    功能：
    1. 周期性扫描所有活跃挂单，检测超时订单
    2. 诊断不成交原因
    3. 执行智能决策（调价重挂 / 撤销等待 / 放弃）
    4. 记录统计信息供复盘使用
    """

    # 时效梯度阈值 (秒)
    TIER_THRESHOLDS = {
        StaleLevel.MILD: 5 * 60,       # 5分钟
        StaleLevel.MODERATE: 15 * 60,   # 15分钟
        StaleLevel.SEVERE: 30 * 60,     # 30分钟
        StaleLevel.CRITICAL: 60 * 60,   # 60分钟
    }

    # 价格偏离阈值 (挂单价相对市场价的偏差)
    PRICE_DEVIATION_MILD = 0.005      # 0.5% 轻度偏离
    PRICE_DEVIATION_MODERATE = 0.01   # 1.0% 中度偏离
    PRICE_DEVIATION_SEVERE = 0.02     # 2.0% 严重偏离
    PRICE_DEVIATION_MAX_REPRICE = 0.03  # 3.0% 超过此值放弃重挂

    # 重定价参数
    MAX_REPRICE_COUNT = 3              # 最多重挂次数
    REPRICE_CLOSER_RATIO = 0.5         # 调价时向市价靠近的比例 (50%)
    REPRICE_MAX_SLIPPAGE = 0.003       # 重定价时最大滑点容忍

    def __init__(self, config: Dict[str, Any]):
        cfg = config.get("execution", {}).get("stale_order") or {}
        if not isinstance(cfg, dict):
            cfg = {}

        # 时效阈值 (可从配置覆盖)
        self._tier_mild = cfg.get("tier_mild_seconds", self.TIER_THRESHOLDS[StaleLevel.MILD])
        self._tier_moderate = cfg.get("tier_moderate_seconds", self.TIER_THRESHOLDS[StaleLevel.MODERATE])
        self._tier_severe = cfg.get("tier_severe_seconds", self.TIER_THRESHOLDS[StaleLevel.SEVERE])
        self._tier_critical = cfg.get("tier_critical_seconds", self.TIER_THRESHOLDS[StaleLevel.CRITICAL])

        # 价格偏离阈值
        self._deviation_mild = cfg.get("price_deviation_mild", self.PRICE_DEVIATION_MILD)
        self._deviation_moderate = cfg.get("price_deviation_moderate", self.PRICE_DEVIATION_MODERATE)
        self._deviation_severe = cfg.get("price_deviation_severe", self.PRICE_DEVIATION_SEVERE)
        self._deviation_max_reprice = cfg.get("price_deviation_max_reprice", self.PRICE_DEVIATION_MAX_REPRICE)

        # 重定价参数
        self._max_reprice_count = cfg.get("max_reprice_count", self.MAX_REPRICE_COUNT)
        self._reprice_closer_ratio = cfg.get("reprice_closer_ratio", self.REPRICE_CLOSER_RATIO)
        self._reprice_max_slippage = cfg.get("reprice_max_slippage", self.REPRICE_MAX_SLIPPAGE)

        # 扫描间隔
        self._scan_interval = cfg.get("scan_interval_seconds", 30)

        # 内部状态
        self._stale_orders: Dict[str, StaleOrderInfo] = {}  # order_id -> StaleOrderInfo
        self._stats = StaleStats()
        self._okx_client = None
        self._order_executor = None
        self._running = False
        self._stop_event = asyncio.Event()

        # 方向暂停列表 (某币种某方向被放弃后暂停新开)
        self._direction_paused: Dict[str, float] = {}  # "SYMBOL:side:posSide" -> pause_until_ts

        # reprice 闭环：新订单ID -> 继承的 repriced_count（跨扫描周期延续重挂次数上限）
        self._reprice_chain: Dict[str, int] = {}
        self._reprice_verify_timeout = cfg.get("reprice_verify_timeout_seconds", 3.0)

        logger.info(
            f"StaleOrderManager initialized: "
            f"tiers={self._tier_mild}/{self._tier_moderate}/{self._tier_severe}/{self._tier_critical}s, "
            f"scan_interval={self._scan_interval}s, "
            f"max_reprice={self._max_reprice_count}"
        )

    def set_dependencies(self, okx_client, order_executor):
        """注入依赖"""
        self._okx_client = okx_client
        self._order_executor = order_executor

    # ==================== 时效等级判定 ====================

    def _get_stale_level(self, age_seconds: float) -> StaleLevel:
        """根据挂单年龄判定时效等级"""
        if age_seconds >= self._tier_critical:
            return StaleLevel.CRITICAL
        if age_seconds >= self._tier_severe:
            return StaleLevel.SEVERE
        if age_seconds >= self._tier_moderate:
            return StaleLevel.MODERATE
        if age_seconds >= self._tier_mild:
            return StaleLevel.MILD
        return StaleLevel.NORMAL

    # ==================== 不成交原因诊断 ====================

    def _diagnose_stale_reason(
        self,
        order: StaleOrderInfo,
        current_bid: float,
        current_ask: float,
        current_mid: float,
        bid_size: float = 0,
        ask_size: float = 0,
        volatility: float = 0,
    ) -> StaleReason:
        """
        诊断挂单不成交原因

        检查顺序：
        1. 价格偏离度
        2. 流动性不足
        3. 方向错误
        4. 买卖价差
        5. 波动率变化
        """
        # 计算价格偏离度
        if order.side == "buy":
            target_price = current_bid if current_bid > 0 else current_mid
        else:
            target_price = current_ask if current_ask > 0 else current_mid

        if target_price <= 0:
            return StaleReason.UNKNOWN

        deviation = abs(order.price - target_price) / target_price

        # 1. 价格偏离检测
        if deviation >= self._deviation_severe:
            return StaleReason.PRICE_DEVIATION
        if deviation >= self._deviation_moderate:
            # 进一步检查是否有其他因素
            pass

        # 2. 流动性不足 (盘口量不足以吃掉挂单)
        if order.side == "buy" and ask_size > 0:
            if ask_size < order.quantity * 0.5:
                return StaleReason.LOW_LIQUIDITY
        elif order.side == "sell" and bid_size > 0:
            if bid_size < order.quantity * 0.5:
                return StaleReason.LOW_LIQUIDITY

        # 3. 方向错误 (价格向反方向运行)
        if order.market_price_at_create > 0:
            price_change = (target_price - order.market_price_at_create) / order.market_price_at_create
            if order.side == "buy" and price_change > self._deviation_moderate:
                # 买入挂单但价格上涨了 -> 挂单价太低追不上
                return StaleReason.WRONG_DIRECTION
            if order.side == "sell" and price_change < -self._deviation_moderate:
                # 卖出挂单但价格下跌了 -> 挂单价太高卖不出
                return StaleReason.WRONG_DIRECTION

        # 4. 买卖价差过大
        if current_bid > 0 and current_ask > 0:
            spread = (current_ask - current_bid) / current_mid
            if spread > 0.005:  # 0.5%+ 价差
                return StaleReason.SPREAD_TOO_WIDE

        # 5. 有偏离但不太严重
        if deviation >= self._deviation_mild:
            return StaleReason.PRICE_DEVIATION

        return StaleReason.UNKNOWN

    # ==================== 决策引擎 ====================

    def _decide_action(
        self, order: StaleOrderInfo, reason: StaleReason, deviation: float
    ) -> StaleAction:
        """
        根据时效等级和不成交原因决定处理动作

        决策矩阵：
        ┌──────────────┬──────────┬──────────┬──────────┬──────────┐
        │ 原因\等级     │ MILD     │ MODERATE │ SEVERE   │ CRITICAL │
        ├──────────────┼──────────┼──────────┼──────────┼──────────┤
        │ PRICE_DEV    │ LOG_ONLY │ REPRICE  │ REPRICE  │ CANCEL   │
        │ LOW_LIQ      │ KEEP     │ CANCEL_W │ CANCEL_W │ CANCEL_A │
        │ WRONG_DIR    │ KEEP     │ CANCEL_W │ CANCEL_A │ CANCEL_A │
        │ VOL_SHIFT    │ KEEP     │ REPRICE  │ CANCEL_W │ CANCEL_A │
        │ SPREAD_WIDE  │ KEEP     │ CANCEL_W │ CANCEL_A │ CANCEL_A │
        │ UNKNOWN      │ LOG_ONLY │ LOG_ONLY │ CANCEL_W │ CANCEL_A │
        └──────────────┴──────────┴──────────┴──────────┴──────────┘
        """
        # 严重超时：强制撤销
        if order.stale_level == StaleLevel.CRITICAL:
            if reason == StaleReason.PRICE_DEVIATION and deviation < self._deviation_max_reprice:
                if order.repriced_count < self._max_reprice_count:
                    return StaleAction.REPRICE_CLOSER
            return StaleAction.CANCEL_ABANDON

        # 重度超时
        if order.stale_level == StaleLevel.SEVERE:
            if reason == StaleReason.PRICE_DEVIATION:
                if deviation < self._deviation_max_reprice and order.repriced_count < self._max_reprice_count:
                    return StaleAction.REPRICE_CLOSER
                return StaleAction.CANCEL_ABANDON
            if reason in (StaleReason.LOW_LIQUIDITY, StaleReason.VOLATILITY_SHIFT):
                return StaleAction.CANCEL_WAIT
            if reason in (StaleReason.WRONG_DIRECTION, StaleReason.SPREAD_TOO_WIDE):
                return StaleAction.CANCEL_ABANDON
            return StaleAction.CANCEL_WAIT

        # 中度超时
        if order.stale_level == StaleLevel.MODERATE:
            if reason == StaleReason.PRICE_DEVIATION:
                if deviation < self._deviation_max_reprice and order.repriced_count < self._max_reprice_count:
                    return StaleAction.REPRICE
                return StaleAction.CANCEL_WAIT
            if reason == StaleReason.WRONG_DIRECTION:
                return StaleAction.CANCEL_WAIT
            if reason == StaleReason.LOW_LIQUIDITY:
                return StaleAction.CANCEL_WAIT
            if reason == StaleReason.VOLATILITY_SHIFT:
                return StaleAction.REPRICE
            if reason == StaleReason.SPREAD_TOO_WIDE:
                return StaleAction.CANCEL_WAIT
            return StaleAction.LOG_ONLY

        # 轻度超时
        if order.stale_level == StaleLevel.MILD:
            if reason == StaleReason.PRICE_DEVIATION and deviation >= self._deviation_moderate:
                return StaleAction.REPRICE
            return StaleAction.LOG_ONLY

        return StaleAction.KEEP

    # ==================== 价格计算 ====================

    def _calculate_reprice(
        self,
        order: StaleOrderInfo,
        current_bid: float,
        current_ask: float,
        current_mid: float,
        closer: bool = False,
    ) -> float:
        """
        计算重新定价

        策略：
        - 买单：新价格 = max(当前买一价, 原价 * (1 - 靠近比例))
        - 卖单：新价格 = min(当前卖一价, 原价 * (1 + 靠近比例))
        - closer模式：更激进地靠近市价
        """
        ratio = self._reprice_closer_ratio if closer else self._reprice_closer_ratio * 0.6

        if order.side == "buy":
            # 买入单：价格应该向买一价靠近（降低挂单价）
            target = current_bid if current_bid > 0 else current_mid
            if target <= 0:
                return order.price
            new_price = order.price - (order.price - target) * ratio
            # 不低于最低合理价（市价的97%）
            new_price = max(new_price, current_mid * (1 - self._deviation_max_reprice))
        else:
            # 卖出单：价格应该向卖一价靠近（提高挂单价）
            target = current_ask if current_ask > 0 else current_mid
            if target <= 0:
                return order.price
            new_price = order.price + (target - order.price) * ratio
            # 不高于最高合理价（市价的103%）
            new_price = min(new_price, current_mid * (1 + self._deviation_max_reprice))

        # 确保价格变化有意义（至少0.1%）
        if abs(new_price - order.price) / order.price < 0.001:
            if order.side == "buy":
                new_price = current_mid * 0.998
            else:
                new_price = current_mid * 1.002

        return round(new_price, 6)

    # ==================== 执行动作 ====================

    async def _execute_action(
        self, order: StaleOrderInfo, action: StaleAction,
        current_bid: float, current_ask: float, current_mid: float
    ) -> bool:
        """执行决策动作"""
        timestamp = datetime.now().isoformat()
        record = {
            "time": timestamp,
            "action": action.value,
            "level": order.stale_level.name,
            "reason": order.stale_reason.value,
            "price": order.price,
            "current_mid": current_mid,
        }

        success = True

        if action == StaleAction.KEEP:
            pass

        elif action == StaleAction.LOG_ONLY:
            logger.info(
                f"Stale order detected: {order.symbol} {order.side} "
                f"@{order.price} age={order.age_minutes:.0f}min "
                f"level={order.stale_level.name} reason={order.stale_reason.value}"
            )

        elif action in (StaleAction.REPRICE, StaleAction.REPRICE_CLOSER):
            closer = (action == StaleAction.REPRICE_CLOSER)
            new_price = self._calculate_reprice(order, current_bid, current_ask, current_mid, closer)

            # 撤销原单
            cancel_ok = await self._cancel_stale_order(order)
            if not cancel_ok:
                logger.warning(f"Failed to cancel stale order {order.order_id} for reprice")
                success = False
            else:
                # 重挂新单（闭环确认新订单ID）
                new_order_id = await self._replace_order(order, new_price)
                if new_order_id:
                    order.new_order_id = new_order_id
                    order.price = new_price
                    order.repriced_count += 1
                    order.create_time = time.time()  # 重置时效计时
                    order.stale_level = StaleLevel.NORMAL
                    # 记录重挂链：新订单继承重挂次数，防止跨订单绕过 MAX_REPRICE_COUNT
                    self._reprice_chain[new_order_id] = order.repriced_count
                    self._stats.total_repriced += 1
                    self._stats.actions_by_type[action.value] = (
                        self._stats.actions_by_type.get(action.value, 0) + 1
                    )
                    record["new_price"] = new_price
                    record["new_order_id"] = new_order_id
                    logger.info(
                        f"Stale order repriced: {order.symbol} {order.side} "
                        f"{order.price:.4f} -> {new_price:.4f} "
                        f"(repriced #{order.repriced_count}, new_id={new_order_id})"
                    )
                else:
                    logger.warning(f"Failed to replace stale order {order.order_id}")
                    success = False

        elif action == StaleAction.CANCEL_WAIT:
            cancel_ok = await self._cancel_stale_order(order)
            if cancel_ok:
                self._stats.total_cancelled += 1
                logger.info(f"Stale order cancelled (wait): {order.symbol} {order.side} @{order.price}")
            else:
                success = False

        elif action == StaleAction.CANCEL_ABANDON:
            cancel_ok = await self._cancel_stale_order(order)
            if cancel_ok:
                self._stats.total_abandoned += 1
                # 暂停该方向
                direction_key = f"{order.symbol}:{order.side}:{order.pos_side}"
                self._direction_paused[direction_key] = time.time() + 1800  # 暂停30分钟
                logger.warning(
                    f"Stale order abandoned: {order.symbol} {order.side} @{order.price}, "
                    f"direction paused for 30min"
                )
            else:
                success = False

        order.action_history.append(record)
        return success

    async def _cancel_stale_order(self, order: StaleOrderInfo) -> bool:
        """撤销超时挂单（使用线程池执行同步API调用）"""
        if self._okx_client is None:
            logger.error("OKXClient not injected into StaleOrderManager")
            return False

        try:
            loop = asyncio.get_event_loop()
            # 判断是否为条件单/算法单
            if order.order_type in ("conditional", "move_stop"):
                result = await loop.run_in_executor(
                    None, self._okx_client.cancel_algo_order, order.symbol, order.order_id
                )
            else:
                result = await loop.run_in_executor(
                    None, self._okx_client.cancel_order, order.symbol, order.order_id
                )

            if result:
                order.cancelled = True
                logger.debug(f"Order {order.order_id} cancelled successfully")
                return True
            else:
                logger.warning(f"Order {order.order_id} cancel returned False")
                return False
        except Exception as e:
            logger.error(f"Failed to cancel order {order.order_id}: {e}")
            return False

    async def _replace_order(self, order: StaleOrderInfo, new_price: float) -> Optional[str]:
        """撤销后以新价格重挂，确认新订单上链后返回新订单ID（reprice 闭环）"""
        if self._order_executor is None:
            logger.error("OrderExecutor not injected into StaleOrderManager")
            return None

        try:
            # 构建重挂信号字典（OrderExecutor.handle_signal 期望 Dict[str, Any]）
            signal_data = {
                "symbol": order.symbol,
                "signal_type": "reprice",
                "direction": order.side,
                "price": new_price,
                "quantity": order.quantity,
                "strategy_name": order.strategy or "stale_reprice",
                "confidence": 0.6,  # 重挂信号置信度降低
                "leverage": order.leverage,
                "timestamp": datetime.now().isoformat(),
                "order_type": order.order_type,
                "pos_side": order.pos_side,
            }

            # 委托给 OrderExecutor 处理（异步入队，需后续确认是否真正上链）
            await self._order_executor.handle_signal(signal_data)

            # 闭环确认：等待新订单出现在交易所挂单中
            new_order_id = await self._verify_replaced_order(order, new_price)
            if new_order_id:
                logger.info(f"Reprice confirmed: {order.symbol} {order.side} {new_price} -> {new_order_id}")
                return new_order_id
            else:
                logger.warning(f"Reprice not confirmed for {order.symbol} {order.side} @{new_price}")
                return None
        except Exception as e:
            logger.error(f"Failed to reprice order {order.order_id}: {e}")
            return None

    async def _verify_replaced_order(self, order: StaleOrderInfo, new_price: float) -> Optional[str]:
        """等待并确认重挂新订单已出现在交易所挂单中，返回新订单ID"""
        deadline = time.time() + self._reprice_verify_timeout
        while time.time() < deadline:
            try:
                pending = await self._fetch_pending_orders()
                for od in pending:
                    oid = od.get("ordId") or od.get("order_id", "")
                    if not oid or oid == order.order_id:
                        continue
                    sym = od.get("instId") or od.get("symbol", "")
                    side = od.get("side", "")
                    px = float(od.get("px") or od.get("price", 0) or 0)
                    if sym == order.symbol and side == order.side and abs(px - new_price) < 1e-6:
                        return oid
            except Exception as e:
                logger.debug(f"Verify reprice failed: {e}")
            await asyncio.sleep(0.5)
        return None

    # ==================== 主扫描循环 ====================

    async def _scan_loop(self) -> None:
        """周期性扫描活跃挂单"""
        logger.info(f"StaleOrderManager scan loop started (interval={self._scan_interval}s)")
        while not self._stop_event.is_set():
            try:
                await self._scan_pending_orders()
            except Exception as e:
                logger.error(f"StaleOrderManager scan error: {e}")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._scan_interval)
            except asyncio.TimeoutError:
                pass

    async def _scan_pending_orders(self) -> None:
        """扫描所有活跃挂单并处理超时"""
        if self._okx_client is None:
            return

        try:
            # 从交易所获取所有活跃挂单
            pending_orders = await self._fetch_pending_orders()
            if not pending_orders:
                return

            self._stats.total_pending_checked += len(pending_orders)

            for order_data in pending_orders:
                order_id = order_data.get("order_id", order_data.get("ordId", ""))
                if not order_id:
                    continue

                # 获取或创建时效信息
                if order_id not in self._stale_orders:
                    info = StaleOrderInfo(
                        order_id=order_id,
                        symbol=order_data.get("symbol", order_data.get("instId", "")),
                        side=order_data.get("side", ""),
                        pos_side=order_data.get("posSide", ""),
                        order_type=order_data.get("order_type", order_data.get("ordType", "limit")),
                        price=float(order_data.get("price", order_data.get("px", 0))),
                        quantity=float(order_data.get("quantity", order_data.get("sz", 0))),
                        filled_qty=float(order_data.get("filled_qty", order_data.get("accFillSz", 0))),
                        create_time=float(order_data.get("create_time", order_data.get("cTime", 0))) / 1000.0
                            if float(order_data.get("create_time", order_data.get("cTime", 0))) > 1e10
                            else float(order_data.get("create_time", order_data.get("cTime", 0))),
                        strategy=order_data.get("strategy", order_data.get("tag", "")),
                        market_price_at_create=float(order_data.get("market_price", 0)),
                    )
                    # 继承 reprice 重挂链：新订单延续旧订单的重挂次数，防止绕过 MAX_REPRICE_COUNT
                    info.repriced_count = self._reprice_chain.pop(order_id, 0)
                    self._stale_orders[order_id] = info
                else:
                    info = self._stale_orders[order_id]
                    # 更新成交量和价格
                    info.filled_qty = float(order_data.get("filled_qty", order_data.get("accFillSz", 0)))
                    info.price = float(order_data.get("price", order_data.get("px", info.price)))

                # 已撤销的订单跳过
                if info.cancelled:
                    continue

                # 判定时效等级
                info.stale_level = self._get_stale_level(info.age_seconds)
                if info.stale_level == StaleLevel.NORMAL:
                    continue

                # 获取当前市场行情
                current_bid, current_ask, current_mid = await self._get_current_prices(info.symbol)

                # 诊断不成交原因
                deviation = abs(info.price - current_mid) / current_mid if current_mid > 0 else 0
                info.stale_reason = self._diagnose_stale_reason(
                    info, current_bid, current_ask, current_mid
                )

                # 决策
                action = self._decide_action(info, info.stale_reason, deviation)

                if action == StaleAction.KEEP:
                    continue

                # 记录统计
                self._stats.total_stale_detected += 1
                level_key = info.stale_level.name.lower()
                self._stats.by_level[level_key] = self._stats.by_level.get(level_key, 0) + 1

                # 按币种统计
                if info.symbol not in self._stats.by_symbol:
                    self._stats.by_symbol[info.symbol] = {}
                reason_key = info.stale_reason.value
                self._stats.by_symbol[info.symbol][reason_key] = (
                    self._stats.by_symbol[info.symbol].get(reason_key, 0) + 1
                )

                # 执行动作
                info.last_action = action
                await self._execute_action(info, action, current_bid, current_ask, current_mid)

            # 清理已撤销或已成交的订单
            self._cleanup_stale_cache()

        except Exception as e:
            logger.error(f"StaleOrderManager scan error: {e}")

    async def _fetch_pending_orders(self) -> List[Dict]:
        """从交易所获取所有活跃挂单（使用线程池执行同步API调用）"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._okx_client.get_orders, "SWAP"
            )
            return result if isinstance(result, list) else []
        except Exception as e:
            logger.error(f"Failed to fetch pending orders: {e}")
            return []

    async def _get_current_prices(self, symbol: str) -> Tuple[float, float, float]:
        """获取当前买卖价和中间价（使用线程池执行同步API调用）"""
        try:
            loop = asyncio.get_event_loop()
            ticker = await loop.run_in_executor(
                None, self._okx_client.get_ticker, symbol
            )
            if ticker and isinstance(ticker, dict):
                bid = float(ticker.get("bidPx", 0))
                ask = float(ticker.get("askPx", 0))
                last = float(ticker.get("last", 0))
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else last
                return bid, ask, mid
        except Exception as e:
            logger.debug(f"Failed to get prices for {symbol}: {e}")
        return 0, 0, 0

    def _cleanup_stale_cache(self) -> None:
        """清理缓存中的已撤销订单"""
        to_remove = []
        for order_id, info in self._stale_orders.items():
            if info.cancelled and info.age_seconds > 300:  # 撤销超过5分钟后清理
                to_remove.append(order_id)
        for order_id in to_remove:
            del self._stale_orders[order_id]

    @staticmethod
    def _parse_leverage(value) -> int:
        """解析杠杆倍数，非法值回退到 5x（符合最低杠杆约束，避免 leverage=1 被风控拒绝）"""
        try:
            lev = int(float(value))
        except (TypeError, ValueError):
            return 5
        return lev if lev >= 1 else 5

    # ==================== 公开接口 ====================

    async def start(self) -> None:
        """启动管理器"""
        self._running = True
        self._stop_event.clear()
        asyncio.create_task(self._scan_loop())
        logger.info("StaleOrderManager started")

    async def stop(self) -> None:
        """停止管理器"""
        self._running = False
        self._stop_event.set()
        logger.info("StaleOrderManager stopped")

    def is_direction_paused(self, symbol: str, side: str, pos_side: str) -> bool:
        """检查某方向是否被暂停"""
        key = f"{symbol}:{side}:{pos_side}"
        pause_until = self._direction_paused.get(key, 0)
        if pause_until > time.time():
            return True
        if pause_until > 0:
            del self._direction_paused[key]
        return False

    def reset_direction_pause(self, symbol: str, side: str, pos_side: str) -> None:
        """手动重置方向暂停"""
        key = f"{symbol}:{side}:{pos_side}"
        self._direction_paused.pop(key, None)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_pending_checked": self._stats.total_pending_checked,
            "total_stale_detected": self._stats.total_stale_detected,
            "total_repriced": self._stats.total_repriced,
            "total_cancelled": self._stats.total_cancelled,
            "total_abandoned": self._stats.total_abandoned,
            "by_level": dict(self._stats.by_level),
            "by_symbol": dict(self._stats.by_symbol),
            "actions_by_type": dict(self._stats.actions_by_type),
            "direction_paused": {
                k: {"paused_until": v, "remaining_seconds": max(0, v - time.time())}
                for k, v in self._direction_paused.items()
                if v > time.time()
            },
            "active_stale_count": len([
                o for o in self._stale_orders.values()
                if o.stale_level != StaleLevel.NORMAL and not o.cancelled
            ]),
        }

    def get_stale_orders(self) -> List[Dict[str, Any]]:
        """获取当前所有超时挂单详情"""
        result = []
        for oid, info in self._stale_orders.items():
            if info.cancelled:
                continue
            result.append({
                "order_id": oid,
                "symbol": info.symbol,
                "side": info.side,
                "pos_side": info.pos_side,
                "order_type": info.order_type,
                "price": info.price,
                "quantity": info.quantity,
                "filled_qty": info.filled_qty,
                "age_minutes": round(info.age_minutes, 1),
                "stale_level": info.stale_level.name,
                "stale_reason": info.stale_reason.value,
                "last_action": info.last_action.value,
                "repriced_count": info.repriced_count,
                "strategy": info.strategy,
                "action_history": info.action_history[-5:],  # 最近5条
            })
        result.sort(key=lambda x: x["age_minutes"], reverse=True)
        return result
