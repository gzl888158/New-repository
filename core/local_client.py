"""本地模拟的 OKX 客户端，用于离线回测与开发调试时生成模拟行情、订单和账户数据。"""
import asyncio
import json
import random
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from loguru import logger

from core.models import TickData, BarData, Order, Position, AccountInfo, FundingRate


class LocalOKXClient:
    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None):
        self._seed = seed
        self._rng = random.Random(seed)
        self._symbols = config.get("symbols", ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"])
        self._initial_prices = {
            "BTC-USDT-SWAP": 30000.0,
            "ETH-USDT-SWAP": 1500.0,
            "SOL-USDT-SWAP": 100.0
        }
        self._current_prices = self._initial_prices.copy()
        self._price_volatility = {
            "BTC-USDT-SWAP": 0.002,
            "ETH-USDT-SWAP": 0.003,
            "SOL-USDT-SWAP": 0.005
        }
        
        self._account_balance = config.get("initial_balance", 100000.0)
        self._available_balance = self._account_balance
        self._used_margin = 0.0
        self._unrealized_pnl = 0.0
        
        self._positions: Dict[str, Position] = {}
        self._orders: Dict[str, Order] = {}
        self._order_id_counter = 1
        
        self._tick_callback = None
        self._order_callback = None
        self._position_callback = None
        
        self._running = False
        self._tick_interval = config.get("tick_interval_ms", 100) / 1000.0
        self._tick_task = None
        
        logger.info("Local OKX Client initialized")

    def _generate_order_id(self) -> str:
        order_id = f"LOCAL{int(time.time())}{self._order_id_counter:04d}"
        self._order_id_counter += 1
        return order_id

    def _update_prices(self):
        for symbol in self._symbols:
            volatility = self._price_volatility[symbol]
            change = (self._rng.random() - 0.5) * 2 * volatility
            self._current_prices[symbol] *= (1 + change)

    def get_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol not in self._current_prices:
            return None
        
        price = self._current_prices[symbol]
        spread = price * 0.0001
        return {
            "instId": symbol,
            "last": str(price),
            "bidPx": str(price - spread),
            "askPx": str(price + spread),
            "bidSz": str(self._rng.uniform(0.1, 1.0)),
            "askSz": str(self._rng.uniform(0.1, 1.0)),
            "vol24h": str(self._rng.uniform(10000, 50000)),
            "ts": str(int(time.time() * 1000))
        }

    def get_kline(self, symbol: str, interval: str, limit: int = 100) -> List[Dict[str, Any]]:
        if symbol not in self._current_prices:
            return []
        
        now = int(time.time() * 1000)
        interval_ms = self._parse_interval(interval)
        bars = []
        
        base_price = self._current_prices[symbol]
        volatility = self._price_volatility[symbol]
        
        for i in range(limit):
            timestamp = now - i * interval_ms
            open_price = base_price * (1 + (self._rng.random() - 0.5) * volatility)
            close_price = open_price * (1 + (self._rng.random() - 0.5) * volatility * 2)
            high_price = max(open_price, close_price) * (1 + self._rng.random() * volatility * 0.5)
            low_price = min(open_price, close_price) * (1 - self._rng.random() * volatility * 0.5)
            
            bars.append([
                str(timestamp),
                str(open_price),
                str(high_price),
                str(low_price),
                str(close_price),
                str(self._rng.uniform(100, 1000)),
                str(self._rng.uniform(50, 500)),
                "0"
            ])
        
        return bars[::-1]

    def _parse_interval(self, interval: str) -> int:
        interval_map = {
            "1m": 60000,
            "3m": 180000,
            "5m": 300000,
            "15m": 900000,
            "30m": 1800000,
            "1h": 3600000,
            "2h": 7200000,
            "4h": 14400000,
            "6h": 21600000,
            "12h": 43200000,
            "1d": 86400000
        }
        return interval_map.get(interval, 3600000)

    def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        if symbol not in self._current_prices:
            return None
        
        return {
            "instId": symbol,
            "fundingRate": str(self._rng.uniform(-0.003, 0.003)),
            "nextFundingTime": str(int((datetime.now() + timedelta(hours=8)).timestamp() * 1000)),
            "ts": str(int(time.time() * 1000))
        }

    def get_order_book(self, symbol: str, depth: int = 5) -> Optional[Dict[str, Any]]:
        if symbol not in self._current_prices:
            return None
        
        price = self._current_prices[symbol]
        spread = price * 0.0001
        
        asks = []
        bids = []
        
        for i in range(depth):
            ask_price = price + spread * (i + 1)
            bid_price = price - spread * (i + 1)
            asks.append([str(ask_price), str(self._rng.uniform(0.1, 0.5)), "0"])
            bids.append([str(bid_price), str(self._rng.uniform(0.1, 0.5)), "0"])
        
        return {
            "asks": asks,
            "bids": bids,
            "ts": str(int(time.time() * 1000))
        }

    def get_positions(self) -> List[Dict[str, Any]]:
        positions = []
        for pos in self._positions.values():
            positions.append({
                "instId": pos.symbol,
                "posSide": pos.side,
                "pos": str(pos.quantity),
                "avgPx": str(pos.avg_cost),
                "markPx": str(self._current_prices.get(pos.symbol, pos.avg_cost)),
                "upl": str(pos.unrealized_pnl),
                "margin": str(pos.margin),
                "lever": str(pos.leverage),
                "mmr": "0.005",
                "ts": str(int(time.time() * 1000))
            })
        return positions

    def get_account_info(self) -> Optional[Dict[str, Any]]:
        self._update_unrealized_pnl()
        return {
            "totalEq": str(self._account_balance + self._unrealized_pnl),
            "availBal": str(self._available_balance),
            "usedMargin": str(self._used_margin),
            "upl": str(self._unrealized_pnl),
            "marginRate": str(self._used_margin / (self._account_balance + self._unrealized_pnl) if self._account_balance > 0 else 0),
            "ts": str(int(time.time() * 1000))
        }

    def _update_unrealized_pnl(self):
        self._unrealized_pnl = 0.0
        for pos in self._positions.values():
            current_price = self._current_prices.get(pos.symbol, pos.avg_cost)
            if pos.side == "long":
                pnl = (current_price - pos.avg_cost) * pos.quantity
            else:
                pnl = (pos.avg_cost - current_price) * pos.quantity
            pos.unrealized_pnl = pnl
            self._unrealized_pnl += pnl

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float,
                   price: float = None, leverage: int = 1, stop_price: float = None) -> Optional[Dict[str, Any]]:
        if symbol not in self._current_prices:
            return None
        
        current_price = self._current_prices[symbol]
        
        if order_type == "market":
            fill_price = current_price
            status = "filled"
        elif order_type == "limit":
            fill_price = price if price else current_price
            if side == "buy" and fill_price >= current_price:
                status = "filled"
            elif side == "sell" and fill_price <= current_price:
                status = "filled"
            else:
                status = "pending"
        elif order_type == "stop":
            fill_price = stop_price if stop_price else current_price
            status = "pending"
        else:
            fill_price = current_price
            status = "filled"
        
        order_id = self._generate_order_id()
        margin = (fill_price * quantity) / leverage
        
        if status == "filled":
            if margin > self._available_balance:
                logger.warning(f"Insufficient balance for order: {symbol}")
                return None
            
            self._available_balance -= margin
            self._used_margin += margin
            
            if symbol in self._positions:
                pos = self._positions[symbol]
                total_qty = pos.quantity + quantity
                pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * quantity) / total_qty
                pos.quantity = total_qty
                pos.margin += margin
            else:
                pos_side = "long" if side == "buy" else "short"
                position = Position(
                    symbol=symbol,
                    side=pos_side,
                    quantity=quantity,
                    avg_cost=fill_price,
                    mark_price=fill_price,
                    unrealized_pnl=0.0,
                    margin=margin,
                    leverage=leverage,
                    maintenance_margin_rate=0.005,
                    timestamp=datetime.now()
                )
                self._positions[symbol] = position
            
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=fill_price,
                filled_quantity=quantity,
                status="filled",
                create_time=datetime.now()
            )
            self._orders[order_id] = order
        else:
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                type=order_type,
                quantity=quantity,
                price=fill_price,
                filled_quantity=0,
                status="pending",
                create_time=datetime.now(),
                stop_price=stop_price,
                leverage=leverage
            )
            self._orders[order_id] = order
        
        logger.info(f"Order placed: {order_id} {side} {symbol} @ {fill_price:.4f}, qty: {quantity:.4f}, status: {status}")
        return {"ordId": order_id, "sCode": "0", "sMsg": "success"}

    def cancel_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        if order.status == "filled":
            return None
        
        order.status = "cancelled"
        logger.info(f"Order cancelled: {order_id}")
        return {"ordId": order_id, "sCode": "0", "sMsg": "success"}

    def get_order(self, symbol: str, order_id: str) -> Optional[Dict[str, Any]]:
        if order_id not in self._orders:
            return None
        
        order = self._orders[order_id]
        return {
            "ordId": order.order_id,
            "instId": order.symbol,
            "side": order.side,
            "ordType": order.type,
            "sz": str(order.quantity),
            "filledSz": str(order.filled_quantity),
            "px": str(order.price),
            "state": order.status,
            "ts": str(int(order.timestamp.timestamp() * 1000))
        }

    def get_orders(self, inst_type: str = "SWAP") -> List[Dict[str, Any]]:
        pending_orders = []
        for order in self._orders.values():
            if order.status == "pending":
                pending_orders.append({
                    "ordId": order.order_id,
                    "instId": order.symbol,
                    "side": order.side,
                    "ordType": order.type,
                    "sz": str(order.quantity),
                    "filledSz": str(order.filled_quantity),
                    "px": str(order.price),
                    "state": order.status,
                    "ts": str(int(order.timestamp.timestamp() * 1000))
                })
        return pending_orders

    def get_order_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        filled_orders = []
        for order in self._orders.values():
            if order.status == "filled":
                filled_orders.append({
                    "ordId": order.order_id,
                    "instId": order.symbol,
                    "side": order.side,
                    "ordType": order.type,
                    "sz": str(order.quantity),
                    "filledSz": str(order.filled_quantity),
                    "px": str(order.price),
                    "state": order.status,
                    "ts": str(int(order.timestamp.timestamp() * 1000))
                })
        return filled_orders[-limit:]

    def set_leverage(self, symbol: str, leverage: int) -> Optional[Dict[str, Any]]:
        if symbol in self._positions:
            self._positions[symbol].leverage = leverage
        logger.info(f"Leverage set: {symbol} x{leverage}")
        return {"instId": symbol, "lever": str(leverage), "sCode": "0", "sMsg": "success"}

    def reduce_position(self, symbol: str, quantity: float) -> Optional[Dict[str, Any]]:
        if symbol not in self._positions:
            return None
        
        pos = self._positions[symbol]
        if quantity >= pos.quantity:
            self._available_balance += pos.margin + pos.unrealized_pnl
            self._used_margin -= pos.margin
            del self._positions[symbol]
            filled_qty = pos.quantity
            pnl = pos.unrealized_pnl
        else:
            pos.quantity -= quantity
            margin_released = (pos.margin / (pos.quantity + quantity)) * quantity
            pos.margin -= margin_released
            self._available_balance += margin_released
            self._used_margin -= margin_released
            filled_qty = quantity
            pnl = pos.unrealized_pnl * (filled_qty / (pos.quantity + filled_qty))
        
        logger.info(f"Position reduced: {symbol}, qty: {filled_qty:.4f}, PnL: {pnl:.2f}")
        return {"instId": symbol, "sz": str(filled_qty), "pnl": str(pnl), "sCode": "0", "sMsg": "success"}

    @property
    def tick_callback(self) -> Optional[Callable]:
        return self._tick_callback

    @tick_callback.setter
    def tick_callback(self, callback: Callable):
        self._tick_callback = callback

    @property
    def order_callback(self) -> Optional[Callable]:
        return self._order_callback

    @order_callback.setter
    def order_callback(self, callback: Callable):
        self._order_callback = callback

    @property
    def position_callback(self) -> Optional[Callable]:
        return self._position_callback

    @position_callback.setter
    def position_callback(self, callback: Callable):
        self._position_callback = callback

    async def subscribe_market_data(self, symbols: List[str]):
        logger.info(f"Subscribed to market data for: {symbols}")

    async def subscribe_private_data(self):
        logger.info("Subscribed to private data channels")

    async def start_tick_stream(self):
        if self._running:
            return
        
        self._running = True
        logger.info("Starting local tick stream...")
        self._tick_task = asyncio.create_task(self._tick_loop())

    async def _tick_loop(self):
        while self._running:
            self._update_prices()
            self._update_unrealized_pnl()
            
            for symbol in self._symbols:
                tick_data = self._create_tick_data(symbol)
                if self._tick_callback and tick_data:
                    await self._tick_callback(tick_data)
            
            for order_id in list(self._orders.keys()):
                order = self._orders[order_id]
                if order.status == "pending":
                    await self._check_pending_order(order)
            
            await asyncio.sleep(self._tick_interval)

    def _create_tick_data(self, symbol: str) -> Optional[TickData]:
        price = self._current_prices.get(symbol)
        if price is None:
            return None
        
        spread = price * 0.0001
        return TickData(
            symbol=symbol,
            price=price,
            volume=self._rng.uniform(10000, 50000),
            bid_price=price - spread,
            bid_volume=self._rng.uniform(0.1, 1.0),
            ask_price=price + spread,
            ask_volume=self._rng.uniform(0.1, 1.0),
            timestamp=datetime.now()
        )

    async def _check_pending_order(self, order: Order):
        current_price = self._current_prices.get(order.symbol)
        if current_price is None:
            return

        if order.type == "limit":
            if order.side == "buy" and order.price >= current_price:
                await self._fill_order(order, current_price)
            elif order.side == "sell" and order.price <= current_price:
                await self._fill_order(order, current_price)
        elif order.type == "stop":
            # stop单使用stop_price作为触发价
            trigger_price = order.stop_price if order.stop_price else order.price
            if order.side == "buy" and current_price >= trigger_price:
                await self._fill_order(order, current_price)
            elif order.side == "sell" and current_price <= trigger_price:
                await self._fill_order(order, current_price)

    async def _fill_order(self, order: Order, fill_price: float):
        leverage = order.leverage if order.leverage else 10
        margin = (fill_price * order.quantity) / leverage
        
        if margin > self._available_balance:
            return
        
        self._available_balance -= margin
        self._used_margin += margin
        
        order.status = "filled"
        order.filled_quantity = order.quantity
        order.price = fill_price
        
        if order.symbol in self._positions:
            pos = self._positions[order.symbol]
            total_qty = pos.quantity + order.quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + fill_price * order.quantity) / total_qty
            pos.quantity = total_qty
            pos.margin += margin
        else:
            pos_side = "long" if order.side == "buy" else "short"
            position = Position(
                symbol=order.symbol,
                side=pos_side,
                quantity=order.quantity,
                avg_cost=fill_price,
                mark_price=fill_price,
                unrealized_pnl=0.0,
                margin=margin,
                leverage=leverage,
                maintenance_margin_rate=0.005,
                timestamp=datetime.now()
            )
            self._positions[order.symbol] = position
        
        logger.info(f"Order filled: {order.order_id} {order.side} {order.symbol} @ {fill_price:.4f}")
        
        if self._order_callback:
            await self._order_callback({
                "ordId": order.order_id,
                "instId": order.symbol,
                "side": order.side,
                "ordType": order.type,
                "sz": str(order.quantity),
                "filledSz": str(order.filled_quantity),
                "px": str(order.price),
                "state": order.status,
                "ts": str(int(order.create_time.timestamp() * 1000))
            })

    async def close_websocket(self):
        self._running = False
        if self._tick_task:
            await self._tick_task
        logger.info("Local client stopped")

    def close(self):
        self._running = False

    def get_current_price(self, symbol: str) -> float:
        return self._current_prices.get(symbol, 0.0)

    def get_positions_dict(self) -> Dict[str, Position]:
        return self._positions.copy()

    def reset(self):
        self._current_prices = self._initial_prices.copy()
        self._available_balance = self._account_balance
        self._used_margin = 0.0
        self._unrealized_pnl = 0.0
        self._positions = {}
        self._orders = {}
        self._order_id_counter = 1
        self._rng = random.Random(self._seed)
        logger.info("Local client reset")