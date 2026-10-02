"""
暗池路由器 (Dark Pool Router)

.. deprecated:: 实验性模块，未接入生产交易链路。

将大订单路由到隐蔽执行场所，避免市场冲击：
  - 暗池搜索：通过OKX的隐藏订单、大宗交易等方式模拟暗池
  - 隐藏订单管理：使用OKX的冰山单/IOC/FOK等订单类型
  - 流动性发现：通过探测小单发现暗池流动性
  - 交叉网络：在内部撮合系统中匹配反向订单（内部化）
  - 成本优化：暗池免手续费的优势
"""
import asyncio
import math
import random
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Set
import numpy as np
from loguru import logger

from core.direction_unifier import DirectionUnifier


class DarkPoolVenue(Enum):
    """暗池场所类型"""
    HIDDEN_ORDER = "hidden_order"          # OKX隐藏订单
    ICEBERG_ORDER = "iceberg_order"        # 冰山订单（OKX支持）
    BLOCK_TRADE = "block_trade"            # 大宗交易（场外）
    INTERNAL_CROSS = "internal_cross"      # 内部交叉（系统内部撮合）
    RFQ = "rfq"                            # 报价请求（OTC询价）


@dataclass
class DarkPoolOrder:
    """暗池订单"""
    order_id: str = ""
    symbol: str = ""
    side: str = "buy"
    quantity: float = 0.0
    limit_price: Optional[float] = None
    venue: DarkPoolVenue = DarkPoolVenue.HIDDEN_ORDER
    # 状态
    status: str = "created"               # created / active / partial / filled / cancelled / expired
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    # 时间线
    created_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    # 审计
    fill_count: int = 0
    total_cost: float = 0.0
    venue_latency_ms: float = 0.0
    # 匿名性
    anonymous_id: str = ""

    def __post_init__(self):
        if not self.order_id:
            self.order_id = f"dp_{uuid.uuid4().hex[:10]}"
        if not self.anonymous_id:
            self.anonymous_id = f"anon_{uuid.uuid4().hex[:8]}"
        if not self.created_at:
            self.created_at = datetime.now()

    @property
    def fill_rate(self) -> float:
        return self.filled_quantity / max(self.quantity, 1e-10)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "anonymous_id": self.anonymous_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": round(self.quantity, 4),
            "filled": round(self.filled_quantity, 4),
            "fill_rate": round(self.fill_rate, 3),
            "avg_price": round(self.avg_fill_price, 4),
            "venue": self.venue.value,
            "status": self.status,
            "latency_ms": round(self.venue_latency_ms, 1),
        }


@dataclass
class DarkPoolFill:
    """暗池成交"""
    fill_id: str
    order_id: str
    quantity: float
    price: float
    venue: DarkPoolVenue
    counterparty: str = "anonymous"    # 对手方（匿名）
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    saved_bps: float = 0.0             # 相比公开市场节省(bps)


# ═══════════════════════════════════════════════════════════════
# 暗池路由器
# ═══════════════════════════════════════════════════════════════

class DarkPoolRouter:
    """暗池路由器"""

    def __init__(self, config: Dict[str, Any] = None):
        cfg = config.get("dark_pool_router", {}) if config else {}
        self._enabled = cfg.get("enabled", True)
        # 场所权重
        self._venue_weights = {
            DarkPoolVenue.HIDDEN_ORDER: cfg.get("hidden_order_weight", 0.40),
            DarkPoolVenue.ICEBERG_ORDER: cfg.get("iceberg_weight", 0.35),
            DarkPoolVenue.BLOCK_TRADE: cfg.get("block_trade_weight", 0.10),
            DarkPoolVenue.INTERNAL_CROSS: cfg.get("internal_cross_weight", 0.10),
            DarkPoolVenue.RFQ: cfg.get("rfq_weight", 0.05),
        }
        # 内部交叉
        self._internal_cross_enabled = cfg.get("internal_cross_enabled", True)
        self._max_cross_wait_seconds = cfg.get("max_cross_wait_seconds", 60.0)
        # 探测
        self._probing_enabled = cfg.get("probing_enabled", True)
        self._probe_size_pct = cfg.get("probe_size_pct", 0.02)  # 探测量占总订单2%
        self._max_probes = cfg.get("max_probes", 3)
        # 最小订单量
        self._min_order_notional = cfg.get("min_order_notional", 100)  # 最小名义价值(USDT)
        self._min_hidden_qty = cfg.get("min_hidden_qty", 0.01)

        # 内部订单簿（系统内部交叉撮合）
        self._internal_book: Dict[str, List[DarkPoolOrder]] = defaultdict(list)  # symbol -> orders
        # 成交历史
        self._fill_history: List[DarkPoolFill] = []
        # 统计
        self._stats: Dict[str, Dict] = defaultdict(lambda: {
            "orders": 0, "total_volume": 0.0, "avg_savings_bps": 0.0,
            "fill_rate": 0.0, "latency": 0.0,
        })

        logger.info(f"DarkPoolRouter initialized: internal_cross={self._internal_cross_enabled}, "
                    f"probing={self._probing_enabled}")

    # ── 内部交叉撮合 ──────────────────────────────────────────

    async def submit_internal_order(self, order: DarkPoolOrder) -> DarkPoolFill:
        """提交到内部交叉网络（系统内部撮合）"""
        order.venue = DarkPoolVenue.INTERNAL_CROSS
        order.status = "active"

        symbol = order.symbol
        opposite_side = DirectionUnifier.opposite(order.side)

        # 查找匹配的反向订单
        matches = []
        for existing in self._internal_book.get(symbol, []):
            if existing.side == opposite_side and existing.status == "active":
                # 检查价格交叉
                if DirectionUnifier.is_long(order.side) and existing.limit_price:
                    if order.limit_price and order.limit_price < existing.limit_price:
                        continue
                elif DirectionUnifier.is_short(order.side) and existing.limit_price:
                    if order.limit_price and order.limit_price > existing.limit_price:
                        continue
                matches.append(existing)

        if not matches:
            # 无匹配：挂入内部簿
            self._internal_book[symbol].append(order)
            # 超时自动撤销
            await asyncio.sleep(self._max_cross_wait_seconds)
            if order.status == "active":
                order.status = "expired"
                self._internal_book[symbol] = [
                    o for o in self._internal_book[symbol]
                    if o.order_id != order.order_id
                ]
            return DarkPoolFill(
                fill_id=f"ic_{order.order_id}",
                order_id=order.order_id,
                quantity=0.0,
                price=0.0,
                venue=DarkPoolVenue.INTERNAL_CROSS,
                counterparty="none",
                saved_bps=0.0,
            )

        # 匹配成交
        match = matches[0]
        fill_qty = min(order.quantity, match.quantity)
        fill_price = order.limit_price or match.limit_price or 0.0

        order.filled_quantity = fill_qty
        order.avg_fill_price = fill_price
        order.status = "filled"
        order.filled_at = datetime.now()

        match.filled_quantity += fill_qty
        if match.filled_quantity >= match.quantity:
            match.status = "filled"
            match.filled_at = datetime.now()

        # 节省 = 双边手续费（内部交叉免手续费）
        saved = fill_qty * fill_price * 0.001  # 0.1%费率免除

        fill = DarkPoolFill(
            fill_id=f"ic_{uuid.uuid4().hex[:8]}",
            order_id=order.order_id,
            quantity=fill_qty,
            price=fill_price,
            venue=DarkPoolVenue.INTERNAL_CROSS,
            counterparty=match.anonymous_id,
            saved_bps=10.0,  # 双向免除 ~10bps
        )

        self._fill_history.append(fill)
        self._update_stats(DarkPoolVenue.INTERNAL_CROSS, saved)

        # 清理已成交订单
        self._internal_book[symbol] = [
            o for o in self._internal_book.get(symbol, [])
            if o.status in ("active", "partial")
        ]

        logger.info(f"Dark pool internal cross: {fill_qty:.4f} @ {fill_price:.4f}, "
                    f"saved={saved:.2f} USDT")

        return fill

    # ── 隐藏订单执行 ──────────────────────────────────────────

    async def execute_hidden_order(self, order: DarkPoolOrder,
                                    order_executor=None,
                                    market_price: float = 0.0) -> DarkPoolFill:
        """执行OKX隐藏订单（下单但不公开显示）"""
        order.venue = DarkPoolVenue.HIDDEN_ORDER
        order.status = "active"
        start_time = time.time()

        if not order_executor:
            # fail-closed：无执行器注入时不得静默保持 active 状态
            logger.error(f"Dark pool hidden order {order.order_id}: no order executor injected, fail-closed")
            order.status = "cancelled"
            order.venue_latency_ms = (time.time() - start_time) * 1000
            return DarkPoolFill(
                fill_id=f"ho_{order.order_id}",
                order_id=order.order_id,
                quantity=0.0,
                price=0.0,
                venue=DarkPoolVenue.HIDDEN_ORDER,
                counterparty="none",
                saved_bps=0.0,
            )

        try:
            params = {
                "symbol": order.symbol,
                "side": order.side,
                "pos_side": DirectionUnifier.to_pos_side(order.side),
                "quantity": order.quantity,
                "order_type": "limit",
                "price": order.limit_price,
                "hidden": True,  # OKX不直接支持hidden，用冰山模拟
                "trace_id": f"algo_{order.order_id}_{uuid.uuid4().hex[:8]}",
            }
            result = order_executor(params)
            if hasattr(result, '__await__'):
                result = await result

            if result:
                order.filled_quantity = float(result.get("filled", 0))
                order.avg_fill_price = float(result.get("avg_price", market_price))
                order.status = "filled" if order.filled_quantity >= order.quantity * 0.95 else "partial"
                order.filled_at = datetime.now()
        except Exception as e:
            logger.warning(f"Hidden order failed: {e}")
            order.status = "cancelled"

        order.venue_latency_ms = (time.time() - start_time) * 1000

        saved = 0.0
        if order.avg_fill_price > 0 and market_price > 0:
            slip_bps = (order.avg_fill_price - market_price) / market_price * 10000
            if DirectionUnifier.is_short(order.side):
                slip_bps *= -1
            saved = max(0, slip_bps)

        fill = DarkPoolFill(
            fill_id=f"ho_{order.order_id}",
            order_id=order.order_id,
            quantity=order.filled_quantity,
            price=order.avg_fill_price,
            venue=DarkPoolVenue.HIDDEN_ORDER,
            counterparty="market",
            saved_bps=round(saved, 2),
        )

        self._fill_history.append(fill)
        self._update_stats(DarkPoolVenue.HIDDEN_ORDER, saved)
        return fill

    # ── 流动性探测 ────────────────────────────────────────────

    async def probe_liquidity(self, symbol: str, side: str,
                               total_qty: float, mid_price: float,
                               order_executor=None) -> Dict[str, Any]:
        """发送小单探测暗池流动性"""
        if not self._probing_enabled:
            return {"liquidity_found": False, "estimated_depth": 0}

        probe_qty = total_qty * self._probe_size_pct
        probe_qty = max(self._min_hidden_qty, min(probe_qty, total_qty * 0.03))

        results = []
        for i in range(self._max_probes):
            if order_executor:
                try:
                    params = {
                        "symbol": symbol,
                        "side": side,
                        "pos_side": DirectionUnifier.to_pos_side(side),
                        "quantity": probe_qty,
                        "order_type": "limit",
                        "price": mid_price,
                        "trace_id": f"probe_{symbol}_{uuid.uuid4().hex[:8]}",
                    }
                    result = order_executor(params)
                    if hasattr(result, '__await__'):
                        result = await result
                    if result:
                        results.append({
                            "filled": float(result.get("filled", 0)),
                            "price": float(result.get("avg_price", mid_price)),
                            "latency_ms": float(result.get("latency_ms", 50)),
                        })
                except Exception:
                    pass
            await asyncio.sleep(random.uniform(0.5, 2.0))

        # 分析探测结果
        if results:
            fill_rates = [r["filled"] / max(probe_qty, 1e-10) for r in results]
            avg_fill = np.mean(fill_rates)
            # 基于成交率估算暗池深度
            estimated_depth = probe_qty / max(1 - avg_fill, 0.01)
            return {
                "liquidity_found": avg_fill > 0.5,
                "estimated_depth": round(estimated_depth, 4),
                "avg_fill_rate": round(avg_fill, 3),
                "probe_results": results,
                "recommendation": "visible_market" if avg_fill < 0.5 else "hidden_execution",
            }

        return {"liquidity_found": False, "estimated_depth": 0}

    # ── 路由入口 ──────────────────────────────────────────────

    async def route_to_dark_pool(self, symbol: str, side: str,
                                   quantity: float, price: float,
                                   order_executor=None) -> DarkPoolOrder:
        """路由到最佳暗池场所"""
        notional = quantity * price
        if notional < self._min_order_notional:
            logger.debug(f"Dark pool skipped: notional too small ({notional:.0f} USDT)")
            return DarkPoolOrder(
                symbol=symbol, side=side, quantity=quantity,
                limit_price=price, venue=DarkPoolVenue.HIDDEN_ORDER,
                status="skipped",
            )

        # 1. 探测流动性
        probe_result = await self.probe_liquidity(symbol, side, quantity, price, order_executor)

        # 2. 选择最佳场所
        # 加权随机选择
        venues = list(self._venue_weights.keys())
        weights = [self._venue_weights[v] for v in venues]

        if probe_result.get("liquidity_found"):
            # 增加暗池权重
            for i, v in enumerate(venues):
                if v == DarkPoolVenue.HIDDEN_ORDER:
                    weights[i] *= 1.5
                elif v == DarkPoolVenue.BLOCK_TRADE:
                    weights[i] *= 1.3

        total_w = sum(weights)
        if total_w <= 0:
            chosen = DarkPoolVenue.HIDDEN_ORDER
        else:
            weights = [w / total_w for w in weights]
            chosen = np.random.choice(venues, p=weights)

        # 3. 创建暗池订单
        order = DarkPoolOrder(
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=price,
            venue=chosen,
        )

        # 4. 执行
        if chosen == DarkPoolVenue.INTERNAL_CROSS and self._internal_cross_enabled:
            await self.submit_internal_order(order)
        else:
            await self.execute_hidden_order(order, order_executor, price)

        logger.info(f"Dark pool routed: {order.order_id} -> {chosen.value} "
                    f"({quantity:.4f} @ {price:.4f}, filled={order.filled_quantity:.4f})")

        return order

    # ── 统计 ──────────────────────────────────────────────────

    def _update_stats(self, venue: DarkPoolVenue, saved_bps: float):
        stat = self._stats[venue.value]
        stat["orders"] += 1
        n = stat["orders"]
        stat["avg_savings_bps"] = (stat["avg_savings_bps"] * (n - 1) + saved_bps) / n

    def get_stats(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "internal_cross_orders": len(self._internal_book),
            "total_fills": len(self._fill_history),
            "venue_stats": {k: dict(v) for k, v in self._stats.items()},
            "total_saved_usdt": round(sum(
                f.saved_bps * f.quantity * f.price / 10000
                for f in self._fill_history
            ), 2),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "internal_cross_enabled": self._internal_cross_enabled,
            "probing_enabled": self._probing_enabled,
            "venue_weights": {k.value: round(v, 2) for k, v in self._venue_weights.items()},
            **self.get_stats(),
        }
