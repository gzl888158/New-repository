"""
模拟行情数据生成器，用于生成测试所需的 K 线与逐笔行情数据。
"""
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from loguru import logger


class SimulatedDataGenerator:
    def __init__(self):
        self._random_seed = None
        self._base_prices = {
            "BTC-USDT-SWAP": 30000.0,
            "ETH-USDT-SWAP": 1500.0,
            "SOL-USDT-SWAP": 100.0,
            "ADA-USDT-SWAP": 0.5,
            "DOT-USDT-SWAP": 7.5
        }
        self._volatility = {
            "BTC-USDT-SWAP": 0.002,
            "ETH-USDT-SWAP": 0.003,
            "SOL-USDT-SWAP": 0.005,
            "ADA-USDT-SWAP": 0.006,
            "DOT-USDT-SWAP": 0.004
        }
    
    def set_random_seed(self, seed: int):
        self._random_seed = seed
        random.seed(seed)
        np.random.seed(seed)
    
    def generate_kline_data(self, symbol: str, interval: str, count: int, 
                           start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if symbol not in self._base_prices:
            logger.error(f"Unknown symbol: {symbol}")
            return []
        
        if start_time is None:
            start_time = datetime.now()
        
        base_price = self._base_prices[symbol]
        volatility = self._volatility[symbol]
        
        interval_ms = self._get_interval_ms(interval)
        bars = []
        
        current_price = base_price
        trend_direction = random.choice([-1, 0, 1])
        trend_strength = random.uniform(0.3, 0.7)
        
        for i in range(count):
            timestamp = start_time - timedelta(milliseconds=(count - i - 1) * interval_ms)
            
            trend_change = trend_direction * trend_strength * volatility * random.uniform(0.5, 1.5)
            random_change = (random.random() - 0.5) * 2 * volatility
            
            open_price = current_price
            close_price = current_price * (1 + trend_change + random_change)
            high_price = max(open_price, close_price) * (1 + random.random() * volatility * 0.5)
            low_price = min(open_price, close_price) * (1 - random.random() * volatility * 0.5)
            volume = random.uniform(100, 1000) * (1 + random.random() * 0.5)
            
            bars.append({
                "timestamp": timestamp,
                "open": round(open_price, 4),
                "high": round(high_price, 4),
                "low": round(low_price, 4),
                "close": round(close_price, 4),
                "volume": round(volume, 4),
                "interval": interval
            })
            
            current_price = close_price
            
            if random.random() < 0.05:
                trend_direction *= -1
        
        return bars
    
    def generate_tick_data(self, symbol: str, count: int, 
                          start_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        if symbol not in self._base_prices:
            logger.error(f"Unknown symbol: {symbol}")
            return []
        
        if start_time is None:
            start_time = datetime.now()
        
        base_price = self._base_prices[symbol]
        volatility = self._volatility[symbol]
        
        ticks = []
        current_price = base_price
        
        for i in range(count):
            timestamp = start_time + timedelta(milliseconds=i * 100)
            
            change = (random.random() - 0.5) * 2 * volatility
            current_price *= (1 + change)
            
            spread = current_price * 0.0001
            bid_price = current_price - spread
            ask_price = current_price + spread
            bid_volume = random.uniform(0.1, 1.0)
            ask_volume = random.uniform(0.1, 1.0)
            
            ticks.append({
                "timestamp": timestamp,
                "symbol": symbol,
                "price": round(current_price, 4),
                "bid_price": round(bid_price, 4),
                "ask_price": round(ask_price, 4),
                "bid_volume": round(bid_volume, 4),
                "ask_volume": round(ask_volume, 4),
                "volume_24h": round(random.uniform(10000, 50000), 4)
            })
        
        return ticks
    
    def generate_order_book(self, symbol: str, depth: int = 5) -> Dict[str, Any]:
        if symbol not in self._base_prices:
            logger.error(f"Unknown symbol: {symbol}")
            return {}
        
        base_price = self._base_prices[symbol]
        volatility = self._volatility[symbol]
        
        current_price = base_price * (1 + (random.random() - 0.5) * 2 * volatility)
        spread = current_price * 0.0001
        
        asks = []
        bids = []
        
        for i in range(depth):
            ask_price = current_price + spread * (i + 1)
            bid_price = current_price - spread * (i + 1)
            ask_volume = random.uniform(0.1, 1.0) * (depth - i)
            bid_volume = random.uniform(0.1, 1.0) * (depth - i)
            
            asks.append({
                "price": round(ask_price, 4),
                "volume": round(ask_volume, 4)
            })
            bids.append({
                "price": round(bid_price, 4),
                "volume": round(bid_volume, 4)
            })
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "asks": asks,
            "bids": bids,
            "spread": round(spread, 4)
        }
    
    def generate_funding_rate(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._base_prices:
            logger.error(f"Unknown symbol: {symbol}")
            return {}
        
        funding_rate = random.uniform(-0.003, 0.003)
        
        return {
            "symbol": symbol,
            "funding_rate": round(funding_rate, 6),
            "next_funding_time": datetime.now() + timedelta(hours=8),
            "timestamp": datetime.now()
        }
    
    def generate_price_series(self, symbol: str, count: int, volatility_scale: float = 1.0,
                             trend: float = 0.0) -> np.ndarray:
        if symbol not in self._base_prices:
            logger.error(f"Unknown symbol: {symbol}")
            return np.array([])
        
        base_price = self._base_prices[symbol]
        volatility = self._volatility[symbol] * volatility_scale
        
        returns = np.random.normal(trend * volatility, volatility, count)
        prices = base_price * np.cumprod(1 + returns)
        
        return prices
    
    def generate_market_cycles(self, symbol: str, cycle_count: int = 5) -> List[Dict[str, Any]]:
        if symbol not in self._base_prices:
            logger.error(f"Unknown symbol: {symbol}")
            return []
        cycles = []
        base_price = self._base_prices[symbol]
        volatility = self._volatility[symbol]
        current_price = base_price
        
        cycle_types = ["trending_up", "trending_down", "sideways", "volatile"]
        
        for i in range(cycle_count):
            cycle_type = random.choice(cycle_types)
            duration_hours = random.randint(2, 8)
            
            if cycle_type == "trending_up":
                price_change = 1 + random.uniform(0.02, 0.08)
                volatility_factor = 0.8
            elif cycle_type == "trending_down":
                price_change = 1 - random.uniform(0.02, 0.08)
                volatility_factor = 0.8
            elif cycle_type == "volatile":
                price_change = 1 + random.uniform(-0.03, 0.03)
                volatility_factor = 2.0
            else:
                price_change = 1 + random.uniform(-0.01, 0.01)
                volatility_factor = 0.5
            
            cycles.append({
                "cycle": i + 1,
                "type": cycle_type,
                "duration_hours": duration_hours,
                "start_price": round(current_price, 4),
                "end_price": round(current_price * price_change, 4),
                "volatility_factor": round(volatility_factor, 2),
                "expected_move": round((price_change - 1) * 100, 2)
            })
            
            current_price *= price_change
        
        return cycles
    
    def _get_interval_ms(self, interval: str) -> int:
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
    
    def generate_strategy_signals(self, symbol: str, count: int) -> List[Dict[str, Any]]:
        if symbol not in self._base_prices:
            logger.error(f"Unknown symbol: {symbol}")
            return []
        signals = []
        base_price = self._base_prices[symbol]
        volatility = self._volatility[symbol]
        
        current_price = base_price
        signal_probability = 0.1
        
        for i in range(count):
            if random.random() < signal_probability:
                signal_type = random.choice(["buy", "sell"])
                confidence = random.uniform(0.5, 0.95)
                
                if signal_type == "buy":
                    entry_price = current_price * (1 - random.uniform(0.001, 0.005))
                    take_profit = entry_price * (1 + random.uniform(0.01, 0.03))
                    stop_loss = entry_price * (1 - random.uniform(0.015, 0.03))
                else:
                    entry_price = current_price * (1 + random.uniform(0.001, 0.005))
                    take_profit = entry_price * (1 - random.uniform(0.01, 0.03))
                    stop_loss = entry_price * (1 + random.uniform(0.015, 0.03))
                
                signals.append({
                    "timestamp": datetime.now() - timedelta(minutes=(count - i)),
                    "symbol": symbol,
                    "signal_type": signal_type,
                    "entry_price": round(entry_price, 4),
                    "take_profit": round(take_profit, 4),
                    "stop_loss": round(stop_loss, 4),
                    "confidence": round(confidence, 2),
                    "strategy": random.choice(["scalping", "trend", "grid"])
                })
            
            current_price *= (1 + (random.random() - 0.5) * 2 * volatility)
        
        return signals