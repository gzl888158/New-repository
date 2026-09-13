"""
虚拟订单执行引擎
================
负责模拟订单执行：滑点模拟、手续费扣除、成交判断

功能：
- 订单生命周期管理（创建、取消、成交、过期）
- 滑点模拟（基于波动率和订单大小的动态滑点）
- Taker/Maker 费率区分
- 限价单等待触发、市价单立即成交
- 部分成交支持
- 订单历史追踪
"""

import threading
import uuid
import time
import random
from typing import Dict, Any, Optional, List, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass
class VirtualOrder:
    """虚拟订单"""
    order_id: str
    sandbox_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float  # 限价单价格，市价单为0
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    fee_paid: float = 0.0
    is_maker: bool = False
    leverage: int = 1
    pos_side: str = "long"  # long/short
    strategy_name: str = ""
    signal_source: str = ""
    create_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    fill_time: Optional[datetime] = None
    cancel_time: Optional[datetime] = None
    expire_time: Optional[datetime] = None
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)

    def is_alive(self) -> bool:
        return self.status in (OrderStatus.PENDING, OrderStatus.OPEN, OrderStatus.PARTIALLY_FILLED)

    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity

    def fill_rate(self) -> float:
        return self.filled_quantity / self.quantity if self.quantity > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "sandbox_id": self.sandbox_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "type": self.order_type.value,
            "quantity": round(self.quantity, 8),
            "price": round(self.price, 6),
            "status": self.status.value,
            "filled_quantity": round(self.filled_quantity, 8),
            "filled_price": round(self.filled_price, 6),
            "fee_paid": round(self.fee_paid, 6),
            "is_maker": self.is_maker,
            "leverage": self.leverage,
            "pos_side": self.pos_side,
            "strategy_name": self.strategy_name,
            "slippage": round(self.slippage, 6),
            "create_time": self.create_time.isoformat(),
            "fill_time": self.fill_time.isoformat() if self.fill_time else None,
        }


@dataclass
class FillResult:
    """成交结果"""
    order_id: str
    symbol: str
    side: OrderSide
    filled_quantity: float
    filled_price: float
    fee: float
    is_maker: bool
    slippage: float
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "filled_quantity": round(self.filled_quantity, 8),
            "filled_price": round(self.filled_price, 6),
            "fee": round(self.fee, 6),
            "is_maker": self.is_maker,
            "slippage": round(self.slippage, 6),
            "timestamp": self.timestamp.isoformat(),
        }


class SlippageModel:
    """
    滑点模拟模型
    
    基于当前波动率和订单大小动态计算滑点。
    市价单始终有一定滑点，限价单在价格吻合时无滑点。
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self._base_slippage = self.config.get("base_slippage_pct", 0.0001)  # 0.01%
        self._volatility_factor = self.config.get("volatility_factor", 0.5)
        self._size_factor = self.config.get("size_factor", 0.0001)  # 每单位交易的滑点增量
        self._max_slippage = self.config.get("max_slippage_pct", 0.005)  # 0.5%

    def calculate(self, order_quantity: float, order_price: float,
                  volatility_pct: float, is_market: bool = True) -> float:
        """
        计算滑点
        
        Args:
            order_quantity: 订单数量
            order_price: 订单价格
            volatility_pct: 当前波动率（ATR%）
            is_market: 是否市价单
        
        Returns:
            滑点百分比（正数表示不利方向）
        """
        if not is_market:
            return 0.0  # 限价单无滑点

        vol_slippage = volatility_pct * self._volatility_factor
        size_slippage = order_quantity * order_price * self._size_factor
        random_noise = random.uniform(-0.00005, 0.00005)

        slippage = self._base_slippage + vol_slippage + size_slippage + random_noise
        return max(0.0, min(slippage, self._max_slippage))


class VirtualOrderExecutor:
    """
    虚拟订单执行引擎

    模拟真实交易所的订单匹配逻辑：
    - 市价单：按当前价+滑点立即成交
    - 限价单：挂单后按K线价格判断是否触发
    - 止损/止盈单：触发条件满足后转为市价单成交
    - 部分成交：大单可能分批成交
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}

        self._maker_fee_rate = self.config.get("maker_fee_rate", 0.0002)
        self._taker_fee_rate = self.config.get("taker_fee_rate", 0.0005)

        self._slippage_model = SlippageModel(self.config.get("slippage", {}))

        # 订单过期时间
        self._limit_order_ttl_seconds = self.config.get("limit_order_ttl", 3600)  # 1小时

        # 订单存储
        self._orders: Dict[str, VirtualOrder] = {}
        self._order_lock = threading.RLock()

        # 成交历史
        self._fill_history: List[FillResult] = []
        self._max_fill_history = self.config.get("max_fill_history", 1000)

        # 订单计数器
        self._order_counter = 0
        self._total_filled = 0
        self._total_cancelled = 0
        self._total_expired = 0
        self._total_rejected = 0

    # ── 订单生成 ─────────────────────────────────────────────────

    def _generate_order_id(self) -> str:
        self._order_counter += 1
        return f"sandbox_{int(time.time()*1000)}_{self._order_counter:06d}"

    # ── 下单 ─────────────────────────────────────────────────────

    def place_market_order(self, sandbox_id: str, symbol: str, side: OrderSide,
                           quantity: float, leverage: int = 1, pos_side: str = "long",
                           strategy_name: str = "", current_price: float = 0,
                           volatility_pct: float = 0) -> FillResult:
        """
        下市价单（立即成交）

        Args:
            current_price: 当前市场价格
            volatility_pct: 当前波动率百分比
        """
        order_id = self._generate_order_id()
        slippage_pct = self._slippage_model.calculate(
            quantity, current_price, volatility_pct, is_market=True
        )

        # 滑点方向：买多则价格略高，卖空则价格略低
        direction = 1 if side == OrderSide.BUY else -1
        filled_price = current_price * (1 + direction * slippage_pct)

        fee_rate = self._taker_fee_rate
        fee = quantity * filled_price * fee_rate

        order = VirtualOrder(
            order_id=order_id,
            sandbox_id=sandbox_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=current_price,
            status=OrderStatus.FILLED,
            filled_quantity=quantity,
            filled_price=filled_price,
            fee_paid=fee,
            is_maker=False,
            leverage=leverage,
            pos_side=pos_side,
            strategy_name=strategy_name,
            fill_time=datetime.now(),
            slippage=slippage_pct,
        )

        with self._order_lock:
            self._orders[order_id] = order
            self._total_filled += 1

        fill = FillResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            filled_quantity=quantity,
            filled_price=filled_price,
            fee=fee,
            is_maker=False,
            slippage=slippage_pct,
        )
        self._record_fill(fill)

        logger.debug(f"[{sandbox_id}] 市价成交: {side.value} {quantity} {symbol} "
                     f"@ {filled_price:.4f} (滑点{slippage_pct*100:.3f}%)")
        return fill

    def place_limit_order(self, sandbox_id: str, symbol: str, side: OrderSide,
                          quantity: float, limit_price: float, leverage: int = 1,
                          pos_side: str = "long", strategy_name: str = "",
                          ttl_seconds: int = None) -> VirtualOrder:
        """下限价单（挂单，等待价格触发）"""
        order_id = self._generate_order_id()
        ttl = ttl_seconds or self._limit_order_ttl_seconds

        order = VirtualOrder(
            order_id=order_id,
            sandbox_id=sandbox_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=limit_price,
            status=OrderStatus.OPEN,
            leverage=leverage,
            pos_side=pos_side,
            strategy_name=strategy_name,
            expire_time=datetime.now() + timedelta(seconds=ttl),
        )

        with self._order_lock:
            self._orders[order_id] = order

        logger.debug(f"[{sandbox_id}] 挂限价单: {side.value} {quantity} {symbol} @ {limit_price}")
        return order

    def place_conditional_order(self, sandbox_id: str, symbol: str, side: OrderSide,
                                order_type: OrderType, quantity: float,
                                trigger_price: float, leverage: int = 1,
                                pos_side: str = "long", strategy_name: str = "") -> VirtualOrder:
        """
        下条件单（止损/止盈）
        当价格触及 trigger_price 时触发
        """
        order_id = self._generate_order_id()

        order = VirtualOrder(
            order_id=order_id,
            sandbox_id=sandbox_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=trigger_price,
            status=OrderStatus.OPEN,
            leverage=leverage,
            pos_side=pos_side,
            strategy_name=strategy_name,
            metadata={"trigger_price": trigger_price},
        )

        with self._order_lock:
            self._orders[order_id] = order

        logger.debug(f"[{sandbox_id}] 挂条件单: {order_type.value} {side.value} "
                     f"{quantity} {symbol} trigger={trigger_price}")
        return order

    # ── 订单匹配 ─────────────────────────────────────────────────

    def process_orders(self, prices: Dict[str, float],
                       volatility_pcts: Dict[str, float] = None) -> List[FillResult]:
        """
        处理所有挂单，匹配成交

        对每个symbol的最新价格，检查：
        1. 限价单是否触及
        2. 止损/止盈单是否触发
        3. 过期订单自动取消

        Returns:
            本次产生的成交列表
        """
        volatility_pcts = volatility_pcts or {}
        fills = []

        with self._order_lock:
            # 筛选活跃订单
            active_orders = [
                o for o in self._orders.values()
                if o.is_alive() and o.symbol in prices
            ]

        for order in active_orders:
            current_price = prices[order.symbol]
            vol_pct = volatility_pcts.get(order.symbol, 0.01)

            fill = None

            if order.order_type == OrderType.LIMIT:
                fill = self._check_limit_order(order, current_price, vol_pct)
            elif order.order_type in (OrderType.STOP_LOSS, OrderType.TAKE_PROFIT):
                fill = self._check_conditional_order(order, current_price, vol_pct)

            if fill:
                fills.append(fill)

            # 检查过期
            if order.expire_time and datetime.now() > order.expire_time:
                with self._order_lock:
                    if order.is_alive():
                        order.status = OrderStatus.EXPIRED
                        order.cancel_time = datetime.now()
                        self._total_expired += 1

        for fill in fills:
            self._record_fill(fill)

        return fills

    def _check_limit_order(self, order: VirtualOrder, current_price: float,
                           vol_pct: float) -> Optional[FillResult]:
        """检查限价单是否成交"""
        triggered = False

        if order.side == OrderSide.BUY:
            # 买入限价：价格 <= 挂单价时成交
            if current_price <= order.price:
                triggered = True
        else:
            # 卖出限价：价格 >= 挂单价时成交
            if current_price >= order.price:
                triggered = True

        if not triggered:
            return None

        # 限价单按挂单价成交（Maker费率）
        fee_rate = self._maker_fee_rate
        fee = order.quantity * order.price * fee_rate

        with self._order_lock:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = order.price
            order.fee_paid = fee
            order.is_maker = True
            order.fill_time = datetime.now()
            order.slippage = 0.0
            self._total_filled += 1

        logger.debug(f"[{order.sandbox_id}] 限价单成交: {order.side.value} "
                     f"{order.quantity} {order.symbol} @ {order.price}")

        return FillResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_quantity=order.quantity,
            filled_price=order.price,
            fee=fee,
            is_maker=True,
            slippage=0.0,
        )

    def _check_conditional_order(self, order: VirtualOrder, current_price: float,
                                  vol_pct: float) -> Optional[FillResult]:
        """检查条件单（止损/止盈）是否触发"""
        trigger_price = order.price

        triggered = False
        if order.order_type == OrderType.STOP_LOSS:
            # 止损：多头价格跌破触发价 / 空头价格涨破触发价
            if order.pos_side == "long" and current_price <= trigger_price:
                triggered = True
            elif order.pos_side == "short" and current_price >= trigger_price:
                triggered = True
        elif order.order_type == OrderType.TAKE_PROFIT:
            # 止盈：多头价格涨破触发价 / 空头价格跌破触发价
            if order.pos_side == "long" and current_price >= trigger_price:
                triggered = True
            elif order.pos_side == "short" and current_price <= trigger_price:
                triggered = True

        if not triggered:
            return None

        # 条件触发后转为市价单执行
        slippage_pct = self._slippage_model.calculate(
            order.quantity, current_price, vol_pct, is_market=True
        )
        direction = 1 if order.side == OrderSide.BUY else -1
        filled_price = current_price * (1 + direction * slippage_pct)

        fee_rate = self._taker_fee_rate
        fee = order.quantity * filled_price * fee_rate

        with self._order_lock:
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = filled_price
            order.fee_paid = fee
            order.is_maker = False
            order.fill_time = datetime.now()
            order.slippage = slippage_pct
            self._total_filled += 1

        logger.info(f"[{order.sandbox_id}] {order.order_type.value}触发: "
                    f"{order.side.value} {order.quantity} {order.symbol} "
                    f"trigger={trigger_price} fill={filled_price:.4f}")

        return FillResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_quantity=order.quantity,
            filled_price=filled_price,
            fee=fee,
            is_maker=False,
            slippage=slippage_pct,
        )

    def _record_fill(self, fill: FillResult) -> None:
        """记录成交"""
        self._fill_history.append(fill)
        if len(self._fill_history) > self._max_fill_history:
            self._fill_history = self._fill_history[-self._max_fill_history:]

    # ── 订单管理 ─────────────────────────────────────────────────

    def cancel_order(self, order_id: str) -> Tuple[bool, str]:
        """取消订单"""
        with self._order_lock:
            if order_id not in self._orders:
                return False, "订单不存在"
            order = self._orders[order_id]
            if not order.is_alive():
                return False, f"订单已{order.status.value}"
            order.status = OrderStatus.CANCELLED
            order.cancel_time = datetime.now()
            self._total_cancelled += 1
        return True, "取消成功"

    def cancel_all_orders(self, sandbox_id: str = None,
                          symbol: str = None) -> int:
        """批量取消订单"""
        cancelled = 0
        with self._order_lock:
            for order in list(self._orders.values()):
                if not order.is_alive():
                    continue
                if sandbox_id and order.sandbox_id != sandbox_id:
                    continue
                if symbol and order.symbol != symbol:
                    continue
                order.status = OrderStatus.CANCELLED
                order.cancel_time = datetime.now()
                self._total_cancelled += 1
                cancelled += 1
        return cancelled

    # ── 查询 ─────────────────────────────────────────────────────

    def get_order(self, order_id: str) -> Optional[VirtualOrder]:
        with self._order_lock:
            return self._orders.get(order_id)

    def get_orders(self, sandbox_id: str = None, symbol: str = None,
                   status: OrderStatus = None, limit: int = 50) -> List[VirtualOrder]:
        """查询订单"""
        with self._order_lock:
            orders = list(self._orders.values())

        if sandbox_id:
            orders = [o for o in orders if o.sandbox_id == sandbox_id]
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        if status:
            orders = [o for o in orders if o.status == status]

        orders.sort(key=lambda o: o.create_time, reverse=True)
        return orders[:limit]

    def get_alive_orders(self, sandbox_id: str = None) -> List[VirtualOrder]:
        """获取所有活跃订单"""
        with self._order_lock:
            orders = [o for o in self._orders.values() if o.is_alive()]
        if sandbox_id:
            orders = [o for o in orders if o.sandbox_id == sandbox_id]
        return orders

    def get_fill_history(self, symbol: str = None,
                         limit: int = 100) -> List[FillResult]:
        """获取成交历史"""
        fills = self._fill_history
        if symbol:
            fills = [f for f in fills if f.symbol == symbol]
        return fills[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._order_lock:
            total_orders = len(self._orders)
            alive = len([o for o in self._orders.values() if o.is_alive()])
            by_status = {}
            for o in self._orders.values():
                s = o.status.value
                by_status[s] = by_status.get(s, 0) + 1

        return {
            "total_orders": total_orders,
            "alive_orders": alive,
            "total_filled": self._total_filled,
            "total_cancelled": self._total_cancelled,
            "total_expired": self._total_expired,
            "total_rejected": self._total_rejected,
            "by_status": by_status,
            "fill_history_count": len(self._fill_history),
        }

    def clear_history(self) -> None:
        """清除历史"""
        with self._order_lock:
            self._orders.clear()
            self._fill_history.clear()
            self._order_counter = 0
            self._total_filled = 0
            self._total_cancelled = 0
            self._total_expired = 0
            self._total_rejected = 0
