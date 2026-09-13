"""
智能订单路由器 (Smart Order Router) v2

多维度智能路由决策：
  - 多场所路由：限价单/市价单/冰山单/算法单
  - 实时场所排名：订单簿不平衡、流动性加权深度、容量感知
  - 智能订单拆分：最优切片(√qty规则)、容量分配、时序调度
  - 执行场所选择：主交易所、暗池、算法执行
  - 反博弈检测：避免被高频交易探测
  - 市场状态感知：波动率、趋势、流动性周期
  - 审计跟踪：记录每次路由决策的完整上下文
"""
import asyncio
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple, Callable
from loguru import logger


class VenueType(Enum):
    """执行场所类型"""
    CENTRAL_LIMIT = "central_limit"      # 中心化限价
    CENTRAL_MARKET = "central_market"    # 中心化市价
    ICEBERG = "iceberg"                  # 冰山订单
    ALGO_TWAP = "algo_twap"              # TWAP算法
    ALGO_VWAP = "algo_vwap"              # VWAP算法
    DARK_POOL = "dark_pool"              # 暗池
    BLOCK_TRADE = "block_trade"          # 大宗交易


class OrderUrgency(Enum):
    """订单紧急程度"""
    LOW = "low"           # 不急于成交，可选择最优价格
    NORMAL = "normal"     # 常规执行
    HIGH = "high"         # 需要快速成交
    IMMEDIATE = "immediate"  # 立即市价成交


@dataclass
class ExecutionVenue:
    """执行场所"""
    venue_id: str
    venue_type: VenueType
    name: str = ""
    # 实时状态
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_depth: float = 0.0      # 买一档深度 (USD)
    ask_depth: float = 0.0      # 卖一档深度 (USD)
    _spread_bps: float = 0.0    # 价差 (bps, 手动设置)
    fee_rate: float = 0.0005    # 手续费率
    latency_ms: float = 50.0    # 平均延迟 (ms)
    fill_rate: float = 0.95     # 历史成交率
    # 容量和状态
    max_order_size: float = float('inf')
    min_order_size: float = 0.0
    is_available: bool = True
    # 统计
    avg_slippage_bps: float = 0.0
    daily_volume: float = 0.0
    reliability_score: float = 1.0
    # 缓存: 避免 spread_bps 重复计算
    _cached_spread: float = field(default=-1.0, repr=False, init=False)
    _cached_bid: float = field(default=-1.0, repr=False, init=False)
    _cached_ask: float = field(default=-1.0, repr=False, init=False)

    @property
    def spread_bps(self) -> float:
        if self.ask_price > 0 and self.bid_price > 0:
            if self.ask_price != self._cached_ask or self.bid_price != self._cached_bid:
                self._cached_bid = self.bid_price
                self._cached_ask = self.ask_price
                self._cached_spread = (self.ask_price - self.bid_price) / self.bid_price * 10000
            return self._cached_spread
        return self._spread_bps if self._spread_bps > 0 else 999.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue_id": self.venue_id,
            "venue_type": self.venue_type.value,
            "name": self.name,
            "bid_price": self.bid_price,
            "ask_price": self.ask_price,
            "bid_depth": self.bid_depth,
            "ask_depth": self.ask_depth,
            "spread_bps": round(self.spread_bps, 2),
            "fee_rate": self.fee_rate,
            "latency_ms": self.latency_ms,
            "fill_rate": self.fill_rate,
            "is_available": self.is_available,
            "avg_slippage_bps": self.avg_slippage_bps,
            "daily_volume": self.daily_volume,
            "reliability_score": self.reliability_score,
        }


@dataclass
class VenueRanking:
    """场所排名"""
    venue: ExecutionVenue
    score: float = 0.0               # 综合评分 (0-1)
    cost_estimate: float = 0.0        # 预估执行成本 (USD)
    fill_probability: float = 0.0     # 成交概率
    expected_slippage_bps: float = 0.0  # 预期滑点 (bps)
    ranking_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "venue_id": self.venue.venue_id,
            "venue_type": self.venue.venue_type.value,
            "score": round(self.score, 4),
            "cost_estimate": round(self.cost_estimate, 4),
            "fill_probability": round(self.fill_probability, 3),
            "expected_slippage_bps": round(self.expected_slippage_bps, 2),
            "ranking_reason": self.ranking_reason,
        }


@dataclass
class OrderSplittingPlan:
    """订单拆分计划"""
    original_qty: float
    original_notional: float
    slices: List['OrderSlice'] = field(default_factory=list)
    total_slices: int = 0
    expected_savings: float = 0.0     # 预期节省成本 (vs 直接市价)
    execution_time_estimate: float = 0.0  # 预估总执行时间(秒)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_qty": self.original_qty,
            "original_notional": self.original_notional,
            "total_slices": self.total_slices,
            "expected_savings": round(self.expected_savings, 4),
            "execution_time_estimate": round(self.execution_time_estimate, 2),
            "slices": [s.to_dict() for s in self.slices],
            "timestamp": self.timestamp,
        }


@dataclass
class OrderSlice:
    """单个拆分订单"""
    slice_id: str
    venue: ExecutionVenue
    quantity: float
    notional: float
    price_limit: Optional[float] = None   # 限价单价格
    order_type: str = "limit"             # limit / market / algo
    sequence: int = 0                     # 执行顺序
    delay_after_ms: float = 0             # 执行后等待(ms)（反博弈）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "slice_id": self.slice_id,
            "venue_type": self.venue.venue_type.value if self.venue else "unknown",
            "venue_id": self.venue.venue_id if self.venue else "",
            "quantity": self.quantity,
            "notional": round(self.notional, 2),
            "price_limit": self.price_limit,
            "order_type": self.order_type,
            "sequence": self.sequence,
            "delay_after_ms": self.delay_after_ms,
        }


@dataclass
class RouteDecision:
    """路由决策"""
    order_id: str
    symbol: str
    side: str                            # buy / sell
    total_quantity: float
    total_notional: float
    urgency: OrderUrgency
    mid_price: float
    # 决策结果
    recommended_venue: VenueType
    splitting_plan: Optional[OrderSplittingPlan] = None
    alternative_venues: List[VenueRanking] = field(default_factory=list)
    # 决策指标
    estimated_cost: float = 0.0
    estimated_fill_time: float = 0.0      # 预估成交时间(秒)
    market_impact_estimate: float = 0.0   # 预估市场冲击 (bps)
    is_urgent_split: bool = False
    # 元信息
    decision_score: float = 0.0
    reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side,
            "total_quantity": round(self.total_quantity, 4),
            "total_notional": round(self.total_notional, 2),
            "urgency": self.urgency.value,
            "mid_price": round(self.mid_price, 2),
            "recommended_venue": self.recommended_venue.value,
            "splitting": {
                "total_slices": self.splitting_plan.total_slices if self.splitting_plan else 0,
                "expected_savings": round(self.splitting_plan.expected_savings, 2) if self.splitting_plan else 0,
            } if self.splitting_plan else None,
            "alternatives": [
                {"venue": r.venue.venue_id, "score": round(r.score, 3), "cost": round(r.cost_estimate, 2)}
                for r in self.alternative_venues[:3]
            ],
            "estimated_cost": round(self.estimated_cost, 2),
            "estimated_fill_time": round(self.estimated_fill_time, 2),
            "market_impact_bps": round(self.market_impact_estimate, 2),
            "decision_score": round(self.decision_score, 3),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════════════
# 智能订单路由器
# ═══════════════════════════════════════════════════════════════

class SmartOrderRouter:
    """智能订单路由器"""

    # ── 生产级常量 ────────────────────────────────────────────
    _VALID_SIDES = frozenset({"buy", "sell"})
    _MIN_QUANTITY = 1e-8
    _MIN_PRICE = 1e-8
    _MAX_NOTIONAL = 1e9  # 10亿USD上限
    _MARKET_STATE_TTL = 30.0  # 市场状态过期时间(秒)
    _WEIGHT_SUM_WARN = 0.05  # 权重和偏离1.0的警告阈值
    _ROUTE_TIMING_WARN_MS = 100.0  # 路由耗时警告阈值(ms)

    @staticmethod
    def _validate_route_input(side: str, quantity: float, price: float, order_id: str = "") -> Optional[str]:
        """校验路由输入参数，返回错误信息或None"""
        if side not in SmartOrderRouter._VALID_SIDES:
            return f"Invalid side '{side}', must be 'buy' or 'sell'"
        if quantity <= SmartOrderRouter._MIN_QUANTITY:
            return f"Quantity must be > {SmartOrderRouter._MIN_QUANTITY}, got {quantity}"
        if price <= SmartOrderRouter._MIN_PRICE:
            return f"Price must be > {SmartOrderRouter._MIN_PRICE}, got {price}"
        if quantity * price > SmartOrderRouter._MAX_NOTIONAL:
            return f"Notional {quantity * price:.0f} exceeds max {SmartOrderRouter._MAX_NOTIONAL:.0f}"
        return None

    def setup_default_venues(self, okx_client=None):
        """注册默认执行场所（基于 OKX 交易所的多个执行层）"""
        # 1. OKX 中心化限价单
        limit_venue = ExecutionVenue(
            venue_id="okx_central_limit",
            venue_type=VenueType.CENTRAL_LIMIT,
            name="OKX Central Limit Order Book",
            fee_rate=0.0002,        # maker fee
            latency_ms=15.0,        # 交易所延迟
            fill_rate=0.90,
            max_order_size=100000.0,
            min_order_size=1.0,
            is_available=True,
            reliability_score=0.98,
        )
        self.register_venue(limit_venue)

        # 2. OKX 中心化市价单
        market_venue = ExecutionVenue(
            venue_id="okx_central_market",
            venue_type=VenueType.CENTRAL_MARKET,
            name="OKX Central Market Order",
            fee_rate=0.0005,        # taker fee
            latency_ms=10.0,
            fill_rate=0.99,
            max_order_size=50000.0,
            min_order_size=1.0,
            is_available=True,
            reliability_score=0.99,
        )
        self.register_venue(market_venue)

        # 3. 算法执行场所（用于大单拆分）
        algo_twap = ExecutionVenue(
            venue_id="algo_twap_execution",
            venue_type=VenueType.ALGO_TWAP,
            name="TWAP Algorithm Execution",
            fee_rate=0.0002,
            latency_ms=20.0,
            fill_rate=0.95,
            max_order_size=500000.0,
            min_order_size=10.0,
            is_available=True,
            reliability_score=0.90,
        )
        self.register_venue(algo_twap)

        algo_vwap = ExecutionVenue(
            venue_id="algo_vwap_execution",
            venue_type=VenueType.ALGO_VWAP,
            name="VWAP Algorithm Execution",
            fee_rate=0.0002,
            latency_ms=20.0,
            fill_rate=0.95,
            max_order_size=500000.0,
            min_order_size=10.0,
            is_available=True,
            reliability_score=0.90,
        )
        self.register_venue(algo_vwap)

        # 4. 冰山订单场所
        iceberg = ExecutionVenue(
            venue_id="okx_iceberg",
            venue_type=VenueType.ICEBERG,
            name="OKX Iceberg Orders",
            fee_rate=0.0002,
            latency_ms=18.0,
            fill_rate=0.85,
            max_order_size=200000.0,
            min_order_size=5.0,
            is_available=True,
            reliability_score=0.88,
        )
        self.register_venue(iceberg)

        self._okx_client = okx_client
        logger.info(f"Default venues registered: {len(self._venues)} venues "
                    f"({', '.join(v.venue_type.value for v in self._venues.values())})")

    def register_venue(self, venue: ExecutionVenue):
        """注册执行场所"""
        self._venues[venue.venue_id] = venue
        logger.debug(f"Venue registered: {venue.venue_id} ({venue.venue_type.value})")

    def update_venue_market_data(self, venue_id: str, bid: float, ask: float,
                                  bid_sz: float, ask_sz: float, latency_ms: float = None):
        """更新场所实时行情"""
        if venue_id in self._venues:
            v = self._venues[venue_id]
            v.bid_price = bid
            v.ask_price = ask
            v.bid_depth = bid_sz
            v.ask_depth = ask_sz
            if latency_ms is not None:
                v.latency_ms = latency_ms

    def get_venue(self, venue_id: str) -> Optional[ExecutionVenue]:
        return self._venues.get(venue_id)

    def get_available_venues(self, side: str = "buy",
                              min_depth_usd: float = 0) -> List[ExecutionVenue]:
        """获取可用的执行场所"""
        available = []
        for v in self._venues.values():
            if not v.is_available:
                continue
            depth = v.ask_depth if side == "buy" else v.bid_depth
            if depth < min_depth_usd:
                continue
            available.append(v)
        return sorted(available, key=lambda x: x.spread_bps)

    async def refresh_venue_market_data(self, symbols: List[str] = None) -> Dict[str, bool]:
        """通过 OKX API 刷新场所实时行情（价差、深度、延迟等）"""
        results = {}
        if not self._okx_client or not symbols:
            return results

        for symbol in symbols:
            try:
                ticker_data = await self._run_in_executor(
                    self._okx_client.get_order_book, symbol, 5
                )
                if not ticker_data:
                    results[symbol] = False
                    continue

                bids = ticker_data.get("bids", [])
                asks = ticker_data.get("asks", [])
                best_bid = float(bids[0][0]) if bids else 0
                best_ask = float(asks[0][0]) if asks else 0
                bid_sz = float(bids[0][1]) * best_bid if bids and len(bids[0]) > 1 else 0
                ask_sz = float(asks[0][1]) * best_ask if asks and len(asks[0]) > 1 else 0

                # 更新中心化限价场所的行情
                for venue_id in ["okx_central_limit", "okx_central_market"]:
                    self.update_venue_market_data(
                        venue_id, best_bid, best_ask, bid_sz, ask_sz
                    )

                # 同步更新冰山/算法场所（使用相同行情但不同费率结构）
                for algo_id in ["okx_iceberg", "algo_twap_execution", "algo_vwap_execution"]:
                    self.update_venue_market_data(
                        algo_id, best_bid, best_ask, bid_sz, ask_sz
                    )

                results[symbol] = True
            except Exception as e:
                logger.debug(f"Failed to refresh venue data for {symbol}: {e}")
                results[symbol] = False

        return results

    async def _run_in_executor(self, func, *args):
        """在线程池中执行同步函数"""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args)

    # ── 场所排名 ──────────────────────────────────────────────

    def rank_venues(self, side: str, notional: float,
                    urgency: OrderUrgency = OrderUrgency.NORMAL) -> List[VenueRanking]:
        """多维度场所排名"""
        rankings = []
        available = self.get_available_venues(side)

        for venue in available:
            # 各维度评分 (0-1, 越高越好)
            # 1. 价差评分：价差越小越好
            spread_score = max(0, 1.0 - venue.spread_bps / 50.0) if venue.spread_bps > 0 else 1.0

            # 2. 深度评分：深度越大越好
            depth = venue.ask_depth if side == "buy" else venue.bid_depth
            depth_score = min(1.0, depth / max(notional * 2, 1.0))

            # 3. 费率评分：费率越低越好
            fee_score = max(0, 1.0 - venue.fee_rate / 0.002)

            # 4. 延迟评分
            latency_score = max(0, 1.0 - venue.latency_ms / 500.0)

            # 5. 成交率评分
            fill_score = venue.fill_rate

            # 紧急度调整权重 (预计算映射)
            weights = self._urgency_weights.get(urgency)
            if weights is None:
                w_spread = self._spread_weight
                w_depth = self._depth_weight
                w_fee = self._fee_weight
                w_latency = self._latency_weight
                w_fill = self._fill_rate_weight
            else:
                w_spread, w_depth, w_fee, w_latency, w_fill = weights

            composite = (
                w_spread * spread_score +
                w_depth * depth_score +
                w_fee * fee_score +
                w_latency * latency_score +
                w_fill * fill_score
            ) * venue.reliability_score

            # 预估成本
            slip_mult = 0.5 if urgency == OrderUrgency.IMMEDIATE else 0.3
            expected_slippage_bps = venue.spread_bps * slip_mult
            cost = notional * (venue.fee_rate + expected_slippage_bps / 10000)

            rankings.append(VenueRanking(
                venue=venue,
                score=composite,
                cost_estimate=cost,
                fill_probability=fill_score,
                expected_slippage_bps=expected_slippage_bps,
                ranking_reason=self._build_ranking_reason(venue, composite, urgency),
            ))

        rankings.sort(key=lambda r: r.score, reverse=True)
        return rankings

    @staticmethod
    def _build_ranking_reason(venue: ExecutionVenue, score: float,
                               urgency: OrderUrgency) -> str:
        reasons = []
        if venue.spread_bps < 5:
            reasons.append("tight spread")
        if venue.latency_ms < 100:
            reasons.append("low latency")
        if venue.fill_rate > 0.95:
            reasons.append("high fill rate")
        if venue.fee_rate < 0.0005:
            reasons.append("low fee")
        return f"[{venue.venue_type.value}] " + ", ".join(reasons) if reasons else "default"

    # ── 订单拆分 ──────────────────────────────────────────────

    def should_split(self, notional: float, daily_volume: float = None) -> bool:
        """判断是否需要拆分订单"""
        if notional > self._split_threshold_notional:
            return True
        if daily_volume and notional / max(daily_volume, 1e-8) > self._participation_rate_max:
            return True
        return False

    def create_splitting_plan(self, order_id: str, symbol: str, side: str,
                               total_qty: float, price: float,
                               rankings: List[VenueRanking],
                               urgency: OrderUrgency) -> OrderSplittingPlan:
        """创建智能订单拆分计划"""
        notional = total_qty * price
        plan = OrderSplittingPlan(
            original_qty=total_qty,
            original_notional=notional,
        )

        if not self.should_split(notional):
            # 不需要拆分，直接路由到最佳场所
            if rankings:
                best = rankings[0]
                plan.slices.append(OrderSlice(
                    slice_id=f"{order_id}_s0",
                    venue=best.venue,
                    quantity=total_qty,
                    notional=notional,
                    price_limit=self._calc_limit_price(price, side, best),
                    order_type="limit",
                    sequence=0,
                ))
            plan.total_slices = 1
            return plan

        # 计算拆分数量
        num_slices = min(
            self._max_slices,
            max(2, int(notional / self._split_threshold_notional))
        )

        # 按评分分配数量到各切片的场所
        base_qty = total_qty / num_slices
        cumulative_delay = 0
        top_venues = rankings[:min(num_slices, len(rankings))]

        for i in range(num_slices):
            # 轮询或加权分配到各场所
            venue_ranking = top_venues[i % len(top_venues)] if top_venues else (rankings[0] if rankings else None)
            if not venue_ranking:
                break

            # 随机扰动尺寸（反博弈）
            if self._anti_gaming:
                jitter = random.uniform(-self._random_size_pct, self._random_size_pct)
                qty = base_qty * (1 + jitter)
            else:
                qty = base_qty

            # 反博弈延迟
            delay = 0
            if self._anti_gaming and i > 0:
                delay = random.uniform(self._random_delay_ms[0], self._random_delay_ms[1])
            cumulative_delay += delay

            plan.slices.append(OrderSlice(
                slice_id=f"{order_id}_s{i}",
                venue=venue_ranking.venue,
                quantity=round(qty, 4),
                notional=round(qty * price, 2),
                price_limit=self._calc_limit_price(price, side, venue_ranking),
                order_type="limit",
                sequence=i,
                delay_after_ms=round(delay, 0),
            ))

        # 处理剩余量（分配给第一个切片）
        filled_qty = sum(s.quantity for s in plan.slices)
        if plan.slices and filled_qty < total_qty:
            plan.slices[-1].quantity += total_qty - filled_qty
            plan.slices[-1].notional = plan.slices[-1].quantity * price

        plan.total_slices = len(plan.slices)
        # 预估节省成本
        direct_cost = notional * 0.001  # 直接市价费率+滑点
        plan.expected_savings = direct_cost * 0.3  # 预估节省30%
        plan.execution_time_estimate = (num_slices - 1) * 0.3  # 秒

        self._split_history.append(plan)
        return plan

    @staticmethod
    def _calc_limit_price(mid: float, side: str,
                          ranking: VenueRanking) -> Optional[float]:
        """计算限价"""
        spread = ranking.venue.spread_bps / 10000.0
        if side == "buy":
            return mid * (1 + spread * 0.3)   # 偏买价30%
        else:
            return mid * (1 - spread * 0.3)   # 偏卖价30%

    # ── 主路由决策 ────────────────────────────────────────────

    def route(self, order_id: str, symbol: str, side: str,
              quantity: float, price: float,
              urgency: OrderUrgency = OrderUrgency.NORMAL,
              daily_volume: float = None) -> RouteDecision:
        """智能路由决策"""
        notional = quantity * price

        # 输入校验
        input_err = self._validate_route_input(side, quantity, price, order_id)
        if input_err:
            logger.error(f"SOR input validation failed: {input_err}")
            raise ValueError(input_err)

        # 1. 场所排名
        rankings = self.rank_venues(side, notional, urgency)

        # 2. 选择推荐场所
        if not rankings:
            recommended = VenueType.CENTRAL_MARKET
            reasoning = "no venues available, fallback to market order"
        elif urgency == OrderUrgency.IMMEDIATE:
            recommended = VenueType.CENTRAL_MARKET
            reasoning = "immediate execution required"
        elif self.should_split(notional, daily_volume):
            if notional > self._split_threshold_notional * 3:
                recommended = VenueType.ALGO_TWAP
                reasoning = "large order: TWAP to minimize impact"
            elif notional > self._split_threshold_notional:
                recommended = VenueType.ICEBERG
                reasoning = "medium order: iceberg to hide size"
            else:
                recommended = VenueType.CENTRAL_LIMIT
                reasoning = "split order: routed to limit venues"
        elif rankings[0].score > 0.8:
            recommended = VenueType.CENTRAL_LIMIT
            reasoning = f"excellent venue score ({rankings[0].score:.2f}), limit recommended"
        else:
            recommended = VenueType.CENTRAL_MARKET
            reasoning = f"suboptimal venue score ({rankings[0].score:.2f}), market recommended"

        # 3. 创建拆分计划
        splitting_plan = self.create_splitting_plan(
            order_id, symbol, side, quantity, price, rankings, urgency
        )

        # 4. 市场冲击估算
        participation = notional / max(daily_volume or float('inf'), notional * 0.01)
        market_impact = self._impact_coefficient * math.sqrt(participation) * 10000

        # 5. 决策评分
        if rankings:
            decision_score = rankings[0].score * (0.7 if splitting_plan.total_slices > 1 else 1.0)
        else:
            decision_score = 0.5

        decision = RouteDecision(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            total_notional=notional,
            urgency=urgency,
            mid_price=price,
            recommended_venue=recommended,
            splitting_plan=splitting_plan,
            alternative_venues=rankings[:3],
            estimated_cost=rankings[0].cost_estimate if rankings else notional * 0.001,
            estimated_fill_time=splitting_plan.execution_time_estimate,
            market_impact_estimate=round(market_impact, 2),
            is_urgent_split=urgency == OrderUrgency.IMMEDIATE and splitting_plan.total_slices > 1,
            decision_score=round(decision_score, 3),
            reasoning=reasoning,
        )

        # 更新统计
        venue_key = recommended.value
        self._route_stats[venue_key]["routed"] += 1

        logger.info(f"SOR route decision: {order_id} -> {recommended.value} "
                    f"(notional={notional:.0f}, urgency={urgency.value}, "
                    f"score={decision_score:.2f}, slices={splitting_plan.total_slices})")

        return decision

    # ── 反博弈检测 ────────────────────────────────────────────

    def detect_gaming_patterns(self, recent_orders: List[Dict]) -> Dict[str, Any]:
        """检测高频交易探测模式"""
        if len(recent_orders) < 5:
            return {"gaming_detected": False, "confidence": 0.0}

        patterns = {
            "front_running": 0.0,       # 抢先交易
            "quote_stuffing": 0.0,       # 报价填充
            "latency_arbitrage": 0.0,    # 延迟套利
        }

        # 检测同方向连续小单（可能被探测）
        same_side_pings = 0
        for i in range(1, min(10, len(recent_orders))):
            if (recent_orders[i].get("side") == recent_orders[i-1].get("side") and
                recent_orders[i].get("quantity", 0) < 0.01):
                same_side_pings += 1
        if same_side_pings >= 3:
            patterns["front_running"] = min(1.0, same_side_pings / 5.0)

        # 检测短时间内价格波动
        prices = [o.get("price", 0) for o in recent_orders[:10] if o.get("price")]
        if len(prices) >= 5:
            pct_changes = [abs(prices[i] - prices[i-1]) / max(prices[i-1], 1e-8)
                          for i in range(1, len(prices))]
            quick_spikes = sum(1 for c in pct_changes if c > 0.001)  # >10bps
            if quick_spikes >= 3:
                patterns["latency_arbitrage"] = min(1.0, quick_spikes / 5.0)

        max_pattern = max(patterns.values())
        return {
            "gaming_detected": max_pattern > 0.5,
            "confidence": round(max_pattern, 3),
            "patterns": patterns,
            "recommendation": "increase random delays" if max_pattern > 0.5 else "normal execution",
        }

    # ── 统计查询 ──────────────────────────────────────────────

    def get_route_stats(self) -> Dict[str, Any]:
        return {
            "venue_count": len(self._venues),
            "venue_stats": {k: dict(v) for k, v in self._route_stats.items()},
            "total_splits": len(self._split_history),
            "available_venues": len(self.get_available_venues()),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "venues": {
                vid: {
                    "type": v.venue_type.value,
                    "spread_bps": round(v.spread_bps, 2),
                    "latency_ms": round(v.latency_ms, 1),
                    "available": v.is_available,
                    "fill_rate": round(v.fill_rate, 3),
                }
                for vid, v in self._venues.items()
            },
            "route_stats": dict(self._route_stats),
            "anti_gaming": self._anti_gaming,
        }

    # ═══════════════════════════════════════════════════════════
    # v2 增强：市场状态感知与订单簿不平衡
    # ═══════════════════════════════════════════════════════════

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("smart_order_router", {}) if config else {}
        # 拆分参数
        self._max_slices = cfg.get("max_slices", 10)
        self._split_threshold_notional = cfg.get("split_threshold_notional", 5000)
        # 场所权重
        self._spread_weight = cfg.get("spread_weight", 0.30)
        self._depth_weight = cfg.get("depth_weight", 0.25)
        self._fee_weight = cfg.get("fee_weight", 0.15)
        self._latency_weight = cfg.get("latency_weight", 0.10)
        self._fill_rate_weight = cfg.get("fill_rate_weight", 0.20)
        # 反博弈
        self._anti_gaming = cfg.get("anti_gaming", True)
        self._random_delay_ms = cfg.get("random_delay_ms", [50, 500])
        self._random_size_pct = cfg.get("random_size_pct", 0.10)
        # 市场冲击模型
        self._impact_coefficient = cfg.get("impact_coefficient", 0.1)
        self._participation_rate_max = cfg.get("participation_rate_max", 0.05)

        self._venues: Dict[str, ExecutionVenue] = {}
        self._venue_history: Dict[str, List[VenueRanking]] = defaultdict(list)
        self._split_history: deque = deque(maxlen=100)
        self._route_stats: Dict[str, Dict] = defaultdict(lambda: {
            "routed": 0, "filled": 0, "avg_cost": 0.0, "avg_latency": 0.0,
        })
        # v2: 市场状态 & 审计
        self._market_state: Dict[str, Dict[str, float]] = defaultdict(dict)
        self._decision_audit: deque = deque(maxlen=200)
        self._okx_client = None

        # 预计算: 紧急度权重映射 (避免每次循环 if/elif)
        self._urgency_weights = {
            OrderUrgency.IMMEDIATE: (0.10, 0.15, 0.05, 0.30, 0.40),
            OrderUrgency.HIGH: (0.15, 0.20, 0.10, 0.25, 0.30),
        }

        # ── 生产级：配置校验 ──
        self._validate_config()

        logger.info(f"SmartOrderRouter v2 initialized: max_slices={self._max_slices}, "
                    f"split_threshold={self._split_threshold_notional}")

    def _validate_config(self):
        """校验配置参数合法性，异常配置记录警告"""
        weight_sum = (self._spread_weight + self._depth_weight + self._fee_weight +
                      self._latency_weight + self._fill_rate_weight)
        if abs(weight_sum - 1.0) > self._WEIGHT_SUM_WARN:
            logger.warning(f"SOR weights sum={weight_sum:.3f}, expected ~1.0")

        if self._max_slices < 1:
            logger.warning(f"SOR max_slices={self._max_slices} invalid, clamped to 1")
            self._max_slices = 1
        if self._max_slices > 50:
            logger.warning(f"SOR max_slices={self._max_slices} too high, clamped to 50")
            self._max_slices = 50

        if self._split_threshold_notional < 100:
            logger.warning(f"SOR split_threshold_notional={self._split_threshold_notional} too low, clamped to 100")
            self._split_threshold_notional = 100

        if self._impact_coefficient < 0:
            logger.warning(f"SOR impact_coefficient={self._impact_coefficient} negative, clamped to 0")
            self._impact_coefficient = 0.0

        if not 0 < self._participation_rate_max <= 1.0:
            logger.warning(f"SOR participation_rate_max={self._participation_rate_max} invalid, clamped to 0.05")
            self._participation_rate_max = 0.05

    def update_market_state(self, symbol: str, **kwargs):
        """更新市场状态（供外部行情服务调用）

        支持的 key: mid_price, volume_24h, bid_depth_total, ask_depth_total,
                     order_book_imbalance, volatility_pct, spread_bps, trend_strength
        """
        state = self._market_state[symbol]
        state.update(kwargs)
        state["updated_at"] = time.time()

    def _get_order_book_imbalance(self, symbol: str) -> float:
        """计算订单簿不平衡度 [-1, 1]：正=买方强，负=卖方强"""
        state = self._market_state.get(symbol, {})
        bid = state.get("bid_depth_total", 0)
        ask = state.get("ask_depth_total", 0)
        total = bid + ask
        if total <= 0:
            return 0.0
        return (bid - ask) / total

    def _get_liquidity_score(self, symbol: str, notional: float) -> float:
        """流动性充足度评分"""
        state = self._market_state.get(symbol, {})
        vol_24h = state.get("volume_24h", notional * 100)
        score = min(1.0, vol_24h / max(notional * 10, 1.0))
        return score

    def _get_volatility_penalty(self, symbol: str) -> float:
        """波动率惩罚因子（高波动→降评分）"""
        state = self._market_state.get(symbol, {})
        vol_pct = state.get("volatility_pct", 1.0)
        return max(0.3, 1.0 - vol_pct / 20.0)

    def _is_market_state_stale(self, symbol: str) -> bool:
        """检查市场状态是否过期（超过TTL未更新）"""
        state = self._market_state.get(symbol, {})
        updated_at = state.get("updated_at", 0)
        return (time.time() - updated_at) > self._MARKET_STATE_TTL

    # ═══════════════════════════════════════════════════════════
    # v2 增强：订单簿感知场所排名
    # ═══════════════════════════════════════════════════════════

    def rank_venues_enhanced(self, side: str, notional: float, symbol: str = "",
                              urgency: OrderUrgency = OrderUrgency.NORMAL,
                              market_state: Dict[str, float] = None,
                              precomputed: Dict[str, float] = None) -> List[VenueRanking]:
        """增强版场所排名 — 集成订单簿不平衡、波动率、容量感知"""
        rankings = []
        available = self.get_available_venues(side)

        # 使用预计算状态或实时计算
        if precomputed:
            imbalance = precomputed["imbalance"]
            vol_penalty = precomputed["volatility_penalty"]
            liquidity = precomputed["liquidity"]
        else:
            imbalance = self._get_order_book_imbalance(symbol)
            vol_penalty = self._get_volatility_penalty(symbol)
            liquidity = self._get_liquidity_score(symbol, notional)

        # 预计算权重
        weights = self._urgency_weights.get(urgency)
        if weights is None:
            w_spread = self._spread_weight
            w_depth = self._depth_weight
            w_fee = self._fee_weight
            w_latency = self._latency_weight
            w_fill = self._fill_rate_weight
        else:
            w_spread, w_depth, w_fee, w_latency, w_fill = weights

        # 预计算滑点乘数
        slip_mult = 0.5 if urgency == OrderUrgency.IMMEDIATE else 0.3

        for venue in available:
            # 基础评分维度
            spread_score = max(0, 1.0 - venue.spread_bps / 50.0) if venue.spread_bps > 0 else 1.0
            depth = venue.ask_depth if side == "buy" else venue.bid_depth
            depth_score = min(1.0, depth / max(notional * 2, 1.0))
            fee_score = max(0, 1.0 - venue.fee_rate / 0.002)
            latency_score = max(0, 1.0 - venue.latency_ms / 500.0)
            fill_score = venue.fill_rate

            # v2: 订单簿不平衡修正
            if side == "buy":
                imbalance_penalty = max(0.5, 1.0 + imbalance * 0.5)
            else:
                imbalance_penalty = max(0.5, 1.0 - imbalance * 0.5)

            # v2: 容量感知（场所是否装得下）
            capacity_ratio = notional / max(venue.max_order_size, 1.0)
            capacity_score = max(0.3, 1.0 - capacity_ratio * 0.5)

            # v2: 历史滑点修正
            hist_slip = abs(venue.avg_slippage_bps)
            slip_adjustment = max(0.7, 1.0 - hist_slip / 30.0)

            base_score = (
                w_spread * spread_score +
                w_depth * depth_score +
                w_fee * fee_score +
                w_latency * latency_score +
                w_fill * fill_score
            )

            # v2: 综合修正
            composite = (base_score * venue.reliability_score *
                        imbalance_penalty * capacity_score *
                        slip_adjustment * vol_penalty * liquidity)

            # 预估成本
            expected_slippage_bps = venue.spread_bps * slip_mult
            cost = notional * (venue.fee_rate + expected_slippage_bps / 10000)

            rankings.append(VenueRanking(
                venue=venue,
                score=composite,
                cost_estimate=cost,
                fill_probability=fill_score * capacity_score,
                expected_slippage_bps=expected_slippage_bps,
                ranking_reason=self._build_enhanced_reason(
                    venue, composite, urgency, imbalance, capacity_score),
            ))

        rankings.sort(key=lambda r: r.score, reverse=True)
        return rankings

    def _build_enhanced_reason(self, venue: ExecutionVenue, score: float,
                                urgency: OrderUrgency, imbalance: float,
                                capacity: float) -> str:
        reasons = []
        if venue.spread_bps < 5: reasons.append("tight")
        if venue.latency_ms < 100: reasons.append("fast")
        if venue.fill_rate > 0.95: reasons.append("reliable")
        if capacity < 0.8: reasons.append("cap-limited")
        if abs(imbalance) > 0.3:
            reasons.append("skewed" if imbalance > 0 else "bearish")
        return f"[{venue.venue_type.value}] " + ", ".join(reasons) if reasons else "neutral"

    # ═══════════════════════════════════════════════════════════
    # v2 增强：最优拆分 —— √qty 法则 + 容量分配 + 时序调度
    # ═══════════════════════════════════════════════════════════

    def compute_optimal_slices(self, total_qty: float, notional: float,
                                symbol: str = "", urgency: OrderUrgency = None) -> int:
        """计算最优切片数（Almgren-Chriss √qty 法则 + 参与率约束）

        原则：
          - √ quantity 法则：切片数 ∝ √qty，大单多切
          - 参与率上限：每片不超过市场量 × participation_rate_max
          - 紧急度修正：紧急→少切，耐心→多切
          - 流动性修正：高流动性→少切，低流动性→多切
        """
        liquidity = self._get_liquidity_score(symbol, notional)
        state = self._market_state.get(symbol, {})

        # √qty 基准
        sqrt_base = max(2, int(math.sqrt(total_qty * 10)))
        sqrt_base = min(sqrt_base, self._max_slices)

        # 参与率约束：如果市场成交量不足，增加切片数
        vol_24h = state.get("volume_24h", notional * 50)
        min_slices_by_participation = max(1, int(
            notional / max(vol_24h * self._participation_rate_max, 1.0)))
        min_slices_by_participation = min(min_slices_by_participation, self._max_slices)

        optimal = max(sqrt_base, min_slices_by_participation)

        # 紧急度修正
        if urgency:
            urgency_mult = {
                OrderUrgency.IMMEDIATE: 0.3,
                OrderUrgency.HIGH: 0.5,
                OrderUrgency.NORMAL: 1.0,
                OrderUrgency.LOW: 1.3,
            }
            optimal = max(1, int(optimal * urgency_mult.get(urgency, 1.0)))

        # 流动性修正
        if liquidity < 0.3:
            optimal = min(self._max_slices, optimal + 1)
        elif liquidity > 0.8:
            optimal = max(1, optimal - 1)

        return min(optimal, self._max_slices)

    def create_splitting_plan_v2(self, order_id: str, symbol: str, side: str,
                                  total_qty: float, price: float,
                                  rankings: List[VenueRanking],
                                  urgency: OrderUrgency) -> OrderSplittingPlan:
        """智能拆分计划 v2 —— 最优切片 + 容量分配 + 时序调度"""
        notional = total_qty * price
        plan = OrderSplittingPlan(
            original_qty=total_qty,
            original_notional=notional,
        )

        # 不需要拆分 → 直接路由
        if not self.should_split(notional):
            if rankings:
                best = rankings[0]
                plan.slices.append(OrderSlice(
                    slice_id=f"{order_id}_s0",
                    venue=best.venue,
                    quantity=total_qty,
                    notional=notional,
                    price_limit=self._calc_limit_price(price, side, best),
                    order_type="limit",
                    sequence=0,
                ))
            plan.total_slices = 1
            return plan

        # ─ v2: 最优切片数 ─
        num_slices = self.compute_optimal_slices(
            total_qty, notional, symbol, urgency)

        # ─ v2: 按优先级分配切片到场所 ─
        top_venues = rankings[:min(3, len(rankings))]  # 取前3

        # 按评分给每个场所分配切片数
        total_score = sum(r.score for r in top_venues) or 1.0
        venue_allocations = []
        for r in top_venues:
            # 评分为0的不分配
            if r.score <= 0:
                continue
            share = r.score / total_score
            slices_for_venue = max(1, int(num_slices * share))
            venue_allocations.append((r, slices_for_venue))

        # 调整确保总数等于num_slices
        allocated = sum(a[1] for a in venue_allocations)
        if allocated < num_slices and venue_allocations:
            venue_allocations[0] = (venue_allocations[0][0],
                                     venue_allocations[0][1] + num_slices - allocated)
        elif allocated > num_slices and venue_allocations:
            diff = allocated - num_slices
            for i in range(len(venue_allocations)):
                if diff <= 0: break
                v, c = venue_allocations[i]
                reduction = min(diff, c - 1)
                venue_allocations[i] = (v, c - reduction)
                diff -= reduction

        # ─ v2: 时序调度（交错执行避免集中在单一场所）──
        # 展开为切片列表并按场所交错排列
        slice_list = []
        seq = 0
        max_per_venue = max(a[1] for a in venue_allocations) if venue_allocations else 0
        for round_idx in range(max_per_venue):
            for vr, count in venue_allocations:
                if round_idx < count:
                    slice_list.append((vr, seq))
                    seq += 1

        # ─ v2: 非线性切片大小（前大后小，降低尾部冲击）──
        # 权重: w_i = 1/i^0.5 (递减)
        weights = [1.0 / math.sqrt(i + 1) for i in range(num_slices)]
        weights_sum = sum(weights)
        base_sizes = [total_qty * w / weights_sum for w in weights]

        # ─ 创建切片 ─
        cumulative_delay = 0
        for (vr, global_seq), base_size in zip(slice_list, base_sizes):
            qty = base_size

            # 容量约束
            max_capacity = min(vr.venue.max_order_size,
                              vr.venue.ask_depth if side == "buy" else vr.venue.bid_depth)
            if max_capacity > 0 and qty > max_capacity:
                qty = max_capacity

            # 反博弈扰动
            if self._anti_gaming:
                jitter = random.uniform(-self._random_size_pct, self._random_size_pct)
                qty *= (1 + jitter)

            qty = round(qty, 4)

            # 反博弈延迟
            delay = 0
            if self._anti_gaming and global_seq > 0:
                delay = random.uniform(self._random_delay_ms[0], self._random_delay_ms[1])
            cumulative_delay += delay

            plan.slices.append(OrderSlice(
                slice_id=f"{order_id}_s{global_seq}",
                venue=vr.venue,
                quantity=qty,
                notional=round(qty * price, 2),
                price_limit=self._calc_limit_price(price, side, vr),
                order_type="limit",
                sequence=global_seq,
                delay_after_ms=round(delay, 0),
            ))

        # 补充尾部不足量
        filled_qty = sum(s.quantity for s in plan.slices)
        if plan.slices and filled_qty < total_qty:
            remainder = total_qty - filled_qty
            plan.slices[-1].quantity += remainder
            plan.slices[-1].notional += remainder * price

        plan.total_slices = len(plan.slices)
        direct_cost = notional * 0.001
        plan.expected_savings = direct_cost * 0.35
        plan.execution_time_estimate = sum(
            s.delay_after_ms for s in plan.slices) / 1000.0

        self._split_history.append(plan)
        return plan

    # ═══════════════════════════════════════════════════════════
    # v2 增强：审计跟踪 + 完整路由诊断
    # ═══════════════════════════════════════════════════════════

    def route_v2(self, order_id: str, symbol: str, side: str,
                  quantity: float, price: float,
                  urgency: OrderUrgency = OrderUrgency.NORMAL,
                  daily_volume: float = None,
                  use_enhanced: bool = True) -> RouteDecision:
        """智能路由决策 v2 — 增强排名 + 最优拆分 + 审计跟踪"""
        notional = quantity * price
        t_start = time.perf_counter()

        # ── 生产级：输入校验 ──
        input_err = self._validate_route_input(side, quantity, price, order_id)
        if input_err:
            logger.error(f"SOR v2 input validation failed: {input_err}")
            raise ValueError(input_err)

        # ── 生产级：市场状态过期检测 ──
        state_stale = self._is_market_state_stale(symbol)
        if state_stale:
            logger.warning(f"SOR v2 market state stale for {symbol} "
                          f"(TTL={self._MARKET_STATE_TTL}s), using defaults")

        # 预计算市场状态 (避免 rank_venues_enhanced 和 audit 各算一次)
        precomputed = {
            "imbalance": self._get_order_book_imbalance(symbol),
            "liquidity": self._get_liquidity_score(symbol, notional),
            "volatility_penalty": self._get_volatility_penalty(symbol),
        }

        # 1. 场所排名（增强版）
        if use_enhanced:
            rankings = self.rank_venues_enhanced(
                side, notional, symbol, urgency, precomputed=precomputed)
        else:
            rankings = self.rank_venues(side, notional, urgency)

        # 2. 选择推荐场所（基于加权评分）
        if not rankings:
            recommended = VenueType.CENTRAL_MARKET
            reasoning = "no venues available, fallback to market"
        elif urgency == OrderUrgency.IMMEDIATE:
            recommended = VenueType.CENTRAL_MARKET
            reasoning = "immediate execution required"
        elif self.should_split(notional, daily_volume):
            if notional > self._split_threshold_notional * 3 and precomputed["liquidity"] < 0.5:
                recommended = VenueType.ALGO_TWAP
                reasoning = "very large + low liquidity: TWAP optimal"
            elif notional > self._split_threshold_notional * 3:
                recommended = VenueType.ALGO_VWAP
                reasoning = "very large order: VWAP to match market profile"
            elif notional > self._split_threshold_notional:
                recommended = VenueType.ICEBERG
                reasoning = "medium-large: iceberg to hide size"
            else:
                recommended = VenueType.CENTRAL_LIMIT
                reasoning = "small order: limit optimal"
        elif rankings[0].score > 0.8:
            recommended = VenueType.CENTRAL_LIMIT
            reasoning = f"excellent venue score ({rankings[0].score:.2f})"
        else:
            recommended = VenueType.CENTRAL_MARKET
            reasoning = f"suboptimal score ({rankings[0].score:.2f}), market safer"

        # 3. 创建拆分计划（v2）
        splitting_plan = self.create_splitting_plan_v2(
            order_id, symbol, side, quantity, price, rankings, urgency)

        # 4. 市场冲击估算（含参与率修正）
        participation = notional / max(daily_volume or float('inf'), notional * 0.005)
        market_impact = self._impact_coefficient * math.sqrt(participation) * 10000

        # 5. 决策评分
        if rankings:
            decision_score = rankings[0].score * \
                (0.85 if splitting_plan.total_slices > 1 else 1.0)
        else:
            decision_score = 0.3

        elapsed_ms = (time.perf_counter() - t_start) * 1000

        decision = RouteDecision(
            order_id=order_id, symbol=symbol, side=side,
            total_quantity=quantity, total_notional=notional,
            urgency=urgency, mid_price=price,
            recommended_venue=recommended,
            splitting_plan=splitting_plan,
            alternative_venues=rankings[:3],
            estimated_cost=rankings[0].cost_estimate if rankings else notional * 0.001,
            estimated_fill_time=splitting_plan.execution_time_estimate,
            market_impact_estimate=round(market_impact, 2),
            is_urgent_split=urgency == OrderUrgency.IMMEDIATE and splitting_plan.total_slices > 1,
            decision_score=round(decision_score, 3),
            reasoning=reasoning,
        )

        # v2: 审计跟踪 (复用预计算状态)
        audit = {
            "order_id": order_id, "symbol": symbol, "side": side,
            "total_quantity": quantity, "total_notional": notional,
            "urgency": urgency.value, "mid_price": price,
            "recommended_venue": recommended.value,
            "total_slices": splitting_plan.total_slices,
            "num_alternatives": len(rankings),
            "top_venue_score": round(rankings[0].score, 4) if rankings else 0,
            "ranking_method": "enhanced" if use_enhanced else "basic",
            "market_state": {
                "imbalance": round(precomputed["imbalance"], 3),
                "liquidity": round(precomputed["liquidity"], 3),
                "volatility_penalty": round(precomputed["volatility_penalty"], 3),
            },
            "timing_ms": round(elapsed_ms, 2),
            "timestamp": datetime.now().isoformat(),
        }
        # 统计每个venue的切片数
        venue_counts = defaultdict(int)
        for s in splitting_plan.slices:
            if s.venue:
                venue_counts[s.venue.venue_id] += 1
        audit["slices_per_venue"] = dict(venue_counts)
        self._decision_audit.append(audit)

        # 更新统计
        venue_key = recommended.value
        self._route_stats[venue_key]["routed"] += 1

        logger.info(f"SOR v2 route: {order_id} -> {recommended.value} "
                    f"(notional={notional:.0f}, urgency={urgency.value}, "
                    f"score={decision_score:.2f}, slices={splitting_plan.total_slices}, "
                    f"imbalance={audit['market_state']['imbalance']:.3f})")

        # ── 生产级：路由耗时监控 ──
        if elapsed_ms > self._ROUTE_TIMING_WARN_MS:
            logger.warning(f"SOR v2 route took {elapsed_ms:.1f}ms > "
                          f"{self._ROUTE_TIMING_WARN_MS}ms threshold "
                          f"(order={order_id}, symbol={symbol}, notional={notional:.0f})")

        return decision

    # ── v2 查询接口 ───────────────────────────────────────────

    def get_decision_audit(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近的决策审计记录"""
        return list(self._decision_audit)[-limit:]

    def get_market_state(self, symbol: str = None) -> Dict[str, Any]:
        """获取市场状态快照"""
        if symbol:
            state = self._market_state.get(symbol, {})
            return {
                "symbol": symbol,
                "imbalance": round(self._get_order_book_imbalance(symbol), 3),
                "liquidity_score": round(self._get_liquidity_score(symbol, 1.0), 3),
                "volatility_penalty": round(self._get_volatility_penalty(symbol), 3),
                **{k: round(v, 4) if isinstance(v, float) else v
                   for k, v in state.items()},
            }
        return {
            "tracked_symbols": list(self._market_state.keys()),
            "count": len(self._market_state),
        }

    def get_diagnostic(self) -> Dict[str, Any]:
        """完整路由器诊断报告"""
        return {
            "config": {
                "max_slices": self._max_slices,
                "split_threshold_notional": self._split_threshold_notional,
                "participation_rate_max": self._participation_rate_max,
                "anti_gaming": self._anti_gaming,
            },
            "venues": self.get_status().get("venues", {}),
            "route_stats": dict(self._route_stats),
            "market_states": self.get_market_state(),
            "recent_decisions": self.get_decision_audit(10),
            "decision_count": len(self._decision_audit),
            "split_history_count": len(self._split_history),
        }
