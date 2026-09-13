"""
仓位动态规划单元
================
核心定位：激进加仓梯度算法，分阶梯倍率加仓，限定最大杠杆阈值

功能：
- 动态仓位计算（基于波动率、资金、风险偏好）
- 梯度加仓策略（阶梯倍率加仓）
- 杠杆控制（硬性上限）
- 仓位风控（最大仓位、单笔限额）
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import threading


class AddPositionMode(Enum):
    """加仓模式"""
    FIXED = "fixed"           # 固定比例加仓
    MARTINGALE = "martingale" # 马丁格尔倍数加仓
    GRADIENT = "gradient"     # 阶梯梯度加仓
    VOLATILITY = "volatility" # 波动率自适应加仓


class RiskLevel(Enum):
    """风险等级"""
    CONSERVATIVE = "conservative"   # 保守型
    BALANCED = "balanced"           # 平衡型
    AGGRESSIVE = "aggressive"       # 激进型


@dataclass
class PositionPlan:
    """仓位规划"""
    symbol: str
    side: str  # "long" or "short"
    total_size: float
    initial_size: float
    add_levels: List[Dict[str, Any]] = field(default_factory=list)
    stop_loss_price: float = 0.0
    take_profit_prices: List[float] = field(default_factory=list)
    leverage: float = 1.0
    margin_required: float = 0.0
    risk_amount: float = 0.0
    risk_percent: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_current_level(self, current_size: float) -> int:
        """根据当前仓位判断加仓层级"""
        cumulative = self.initial_size
        for i, level in enumerate(self.add_levels):
            if current_size <= cumulative:
                return i
            cumulative += level.get("size", 0)
        return len(self.add_levels)
    
    def get_next_add_size(self, current_size: float) -> float:
        """获取下一次加仓数量"""
        level = self.get_current_level(current_size)
        if level < len(self.add_levels):
            return self.add_levels[level].get("size", 0)
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "total_size": self.total_size,
            "initial_size": self.initial_size,
            "add_levels": self.add_levels,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_prices": self.take_profit_prices,
            "leverage": self.leverage,
            "margin_required": self.margin_required,
            "risk_amount": self.risk_amount,
            "risk_percent": self.risk_percent,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class AddPositionLevel:
    """加仓层级"""
    level: int
    trigger_price: float
    trigger_condition: str  # "price_drop", "price_rise", "pnl_loss", "pnl_profit"
    size: float
    multiplier: float  # 相对于初始仓位的倍数
    reason: str


class PositionPlanner:
    """
    仓位动态规划器
    
    核心算法：
    1. 初始仓位计算：基于ATR动态调整
    2. 加仓梯度：阶梯倍率加仓（1x, 1.5x, 2x, 3x...）
    3. 杠杆控制：硬性上限，超过则缩减仓位
    4. 风险预算：单笔最大亏损限定
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        trading_config = self.config.get("trading", {})
        self._total_capital = trading_config.get("total_capital", 100.0)
        self._max_leverage = trading_config.get("max_leverage", 10.0)
        self._max_position_ratio = trading_config.get("max_position_ratio", 0.8)
        self._risk_per_trade = trading_config.get("risk_per_trade", 0.02)
        self._max_risk_total = trading_config.get("max_risk_total", 0.1)
        
        position_config = self.config.get("position_planning", {})
        self._default_mode = AddPositionMode(position_config.get("add_mode", "gradient"))
        self._default_risk_level = RiskLevel(position_config.get("risk_level", "balanced"))
        
        self._gradient_multipliers = position_config.get("gradient_multipliers", 
            [1.0, 1.5, 2.0, 3.0, 4.0])
        self._martingale_base = position_config.get("martingale_base", 2.0)
        self._max_add_levels = position_config.get("max_add_levels", 5)
        
        self._atr_sl_multiplier = position_config.get("atr_sl_multiplier", 2.0)
        self._atr_tp_multiplier = position_config.get("atr_tp_multiplier", 3.0)
        
        self._min_add_interval_pct = position_config.get("min_add_interval_pct", 0.01)
        self._min_add_interval_bars = position_config.get("min_add_interval_bars", 3)
        
        self._position_plans: Dict[str, PositionPlan] = {}
        self._add_history: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.RLock()
        
        self._plan_count = 0
        self._add_count = 0
    
    def set_capital(self, capital: float) -> None:
        """设置总资金"""
        self._total_capital = capital
    
    def set_max_leverage(self, leverage: float) -> None:
        """设置最大杠杆"""
        self._max_leverage = leverage
    
    def calculate_initial_position(self, symbol: str, price: float, atr: float,
                                    side: str = "long", 
                                    risk_level: RiskLevel = None) -> PositionPlan:
        """
        计算初始仓位和加仓计划
        
        Args:
            symbol: 交易对
            price: 当前价格
            atr: 平均真实波幅
            side: 方向 "long" or "short"
            risk_level: 风险等级
        
        Returns:
            PositionPlan: 仓位规划
        """
        risk_level = risk_level or self._default_risk_level
        
        risk_multiplier = self._get_risk_multiplier(risk_level)
        
        atr_sl_distance = atr * self._atr_sl_multiplier
        sl_price = price - atr_sl_distance if side == "long" else price + atr_sl_distance
        
        risk_per_unit = abs(price - sl_price)
        
        max_risk_amount = self._total_capital * self._risk_per_trade * risk_multiplier
        max_risk_amount = min(max_risk_amount, self._total_capital * self._max_risk_total)
        
        initial_size = max_risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        max_position_value = self._total_capital * self._max_position_ratio
        max_size_by_capital = max_position_value / price
        
        initial_size = min(initial_size, max_size_by_capital)
        
        add_levels = self._calculate_add_levels(
            symbol, price, initial_size, atr, side, risk_level
        )
        
        total_size = initial_size + sum(level.get("size", 0) for level in add_levels)
        
        max_size_by_leverage = self._total_capital * self._max_leverage / price
        if total_size > max_size_by_leverage:
            scale_factor = max_size_by_leverage / total_size
            initial_size *= scale_factor
            for level in add_levels:
                level["size"] *= scale_factor
            total_size = initial_size + sum(level.get("size", 0) for level in add_levels)
        
        take_profit_prices = self._calculate_take_profit_levels(price, atr, side)
        
        margin_required = (total_size * price) / self._max_leverage
        
        plan = PositionPlan(
            symbol=symbol,
            side=side,
            total_size=total_size,
            initial_size=initial_size,
            add_levels=add_levels,
            stop_loss_price=sl_price,
            take_profit_prices=take_profit_prices,
            leverage=self._max_leverage,
            margin_required=margin_required,
            risk_amount=max_risk_amount,
            risk_percent=self._risk_per_trade * risk_multiplier
        )
        
        with self._lock:
            self._position_plans[symbol] = plan
            self._plan_count += 1
        
        logger.debug(f"Position plan created for {symbol}: initial={initial_size:.6f}, "
                    f"total={total_size:.6f}, levels={len(add_levels)}")
        
        return plan
    
    def _calculate_add_levels(self, symbol: str, price: float, initial_size: float,
                               atr: float, side: str, 
                               risk_level: RiskLevel) -> List[Dict[str, Any]]:
        """计算加仓层级"""
        levels = []
        
        mode = self._default_mode
        
        if mode == AddPositionMode.GRADIENT:
            multipliers = self._gradient_multipliers[1:]
        elif mode == AddPositionMode.MARTINGALE:
            multipliers = [self._martingale_base ** i for i in range(1, self._max_add_levels)]
        else:
            multipliers = [1.0] * (self._max_add_levels - 1)
        
        for i, multiplier in enumerate(multipliers[:self._max_add_levels - 1]):
            if side == "long":
                trigger_price = price * (1 - (i + 1) * self._min_add_interval_pct)
                trigger_condition = "price_drop"
            else:
                trigger_price = price * (1 + (i + 1) * self._min_add_interval_pct)
                trigger_condition = "price_rise"
            
            level_size = initial_size * multiplier
            
            level_size = min(level_size, initial_size * 5)
            
            levels.append({
                "level": i + 1,
                "trigger_price": round(trigger_price, 6),
                "trigger_condition": trigger_condition,
                "size": round(level_size, 8),
                "multiplier": multiplier,
                "reason": f"第{i+1}次加仓（倍率{multiplier:.1f}x）"
            })
        
        return levels
    
    def _calculate_take_profit_levels(self, price: float, atr: float, 
                                        side: str) -> List[float]:
        """计算止盈价格层级"""
        tp_multipliers = [2.0, 3.0, 5.0]
        
        tp_prices = []
        for mult in tp_multipliers:
            if side == "long":
                tp_price = price + atr * mult
            else:
                tp_price = price - atr * mult
            tp_prices.append(round(tp_price, 6))
        
        return tp_prices
    
    def _get_risk_multiplier(self, risk_level: RiskLevel) -> float:
        """获取风险偏好乘数"""
        multipliers = {
            RiskLevel.CONSERVATIVE: 0.5,
            RiskLevel.BALANCED: 1.0,
            RiskLevel.AGGRESSIVE: 1.5,
        }
        return multipliers.get(risk_level, 1.0)
    
    def should_add_position(self, symbol: str, current_price: float,
                            current_size: float, pnl_pct: float,
                            atr: float) -> Tuple[bool, float, str]:
        """
        判断是否应该加仓
        
        Returns:
            (should_add, size, reason)
        """
        with self._lock:
            if symbol not in self._position_plans:
                return False, 0.0, "无仓位计划"
            
            plan = self._position_plans[symbol]
            
            current_level = plan.get_current_level(current_size)
            
            if current_level >= len(plan.add_levels):
                return False, 0.0, "已达到最大加仓层级"
            
            next_level = plan.add_levels[current_level]
            trigger_price = next_level.get("trigger_price", 0)
            trigger_condition = next_level.get("trigger_condition", "")
            
            should_add = False
            reason = ""
            
            if trigger_condition == "price_drop":
                if current_price <= trigger_price:
                    should_add = True
                    reason = f"价格跌至{trigger_price:.4f}，触发加仓"
            elif trigger_condition == "price_rise":
                if current_price >= trigger_price:
                    should_add = True
                    reason = f"价格涨至{trigger_price:.4f}，触发加仓"
            elif trigger_condition == "pnl_loss":
                if pnl_pct <= -abs(trigger_price):
                    should_add = True
                    reason = f"亏损达{abs(trigger_price)*100:.1f}%，触发加仓"
            elif trigger_condition == "pnl_profit":
                if pnl_pct >= trigger_price:
                    should_add = True
                    reason = f"盈利达{trigger_price*100:.1f}%，触发加仓"
            
            if should_add:
                add_size = next_level.get("size", 0)
                
                max_additional = plan.total_size - current_size
                if add_size > max_additional:
                    add_size = max_additional
                
                self._add_count += 1
                
                if symbol not in self._add_history:
                    self._add_history[symbol] = []
                self._add_history[symbol].append({
                    "level": current_level + 1,
                    "price": current_price,
                    "size": add_size,
                    "timestamp": datetime.now().isoformat()
                })
                
                return True, add_size, reason
        
        return False, 0.0, ""
    
    def calculate_position_size_for_signal(self, symbol: str, price: float,
                                            signal_strength: float,
                                            volatility: float,
                                            account_balance: float = None) -> float:
        """
        根据信号强度和波动率计算仓位大小
        
        Args:
            symbol: 交易对
            price: 当前价格
            signal_strength: 信号强度 (0-1)
            volatility: 波动率 (ATR%)
            account_balance: 账户余额
        
        Returns:
            仓位大小
        """
        balance = account_balance or self._total_capital
        
        base_risk = self._risk_per_trade * balance
        
        volatility_factor = 1.0 / (1.0 + volatility * 10)
        
        strength_factor = 0.5 + signal_strength * 0.5
        
        adjusted_risk = base_risk * volatility_factor * strength_factor
        
        sl_distance = price * volatility * 2
        
        if sl_distance <= 0:
            return 0
        
        position_size = adjusted_risk / sl_distance
        
        max_size = balance * self._max_position_ratio / price
        position_size = min(position_size, max_size)
        
        return round(position_size, 8)
    
    def calculate_stop_loss(self, symbol: str, entry_price: float, 
                            side: str, atr: float,
                            position_size: float = None) -> float:
        """
        计算止损价格
        
        使用ATR动态止损，结合仓位大小调整
        """
        base_sl_distance = atr * self._atr_sl_multiplier
        
        if position_size and position_size > 0:
            capital = self._total_capital
            position_value = position_size * entry_price
            position_ratio = position_value / capital if capital > 0 else 0
            
            if position_ratio > 0.5:
                base_sl_distance *= 0.8
            elif position_ratio > 0.3:
                base_sl_distance *= 0.9
        
        if side == "long":
            sl_price = entry_price - base_sl_distance
        else:
            sl_price = entry_price + base_sl_distance
        
        return round(sl_price, 6)
    
    def calculate_take_profit(self, symbol: str, entry_price: float,
                               side: str, atr: float,
                               levels: int = 3) -> List[Dict[str, Any]]:
        """
        计算分批止盈价格
        
        Returns:
            [{price, ratio, reason}, ...]
        """
        tp_levels = []
        
        ratios = [0.3, 0.3, 0.4]
        multipliers = [1.5, 2.5, 4.0]
        
        for i in range(min(levels, 3)):
            distance = atr * multipliers[i]
            
            if side == "long":
                tp_price = entry_price + distance
            else:
                tp_price = entry_price - distance
            
            tp_levels.append({
                "price": round(tp_price, 6),
                "ratio": ratios[i],
                "reason": f"第{i+1}档止盈（盈亏比{multipliers[i]}:1）"
            })
        
        return tp_levels
    
    def get_position_plan(self, symbol: str) -> Optional[PositionPlan]:
        """获取仓位计划"""
        with self._lock:
            return self._position_plans.get(symbol)
    
    def remove_position_plan(self, symbol: str) -> None:
        """移除仓位计划（平仓后）"""
        with self._lock:
            if symbol in self._position_plans:
                del self._position_plans[symbol]
    
    def get_add_history(self, symbol: str) -> List[Dict[str, Any]]:
        """获取加仓历史"""
        with self._lock:
            return self._add_history.get(symbol, []).copy()
    
    def adjust_for_volatility(self, base_size: float, volatility_pct: float,
                               max_volatility: float = 0.05) -> float:
        """
        根据波动率调整仓位
        
        高波动环境下减小仓位，低波动环境下可适当增大
        """
        if volatility_pct <= 0:
            return base_size
        
        if volatility_pct > max_volatility:
            adjustment = 0.5
        elif volatility_pct > max_volatility * 0.7:
            adjustment = 0.7
        elif volatility_pct > max_volatility * 0.5:
            adjustment = 0.85
        elif volatility_pct < max_volatility * 0.3:
            adjustment = 1.2
        else:
            adjustment = 1.0
        
        return base_size * adjustment
    
    def calculate_leverage_usage(self, position_size: float, price: float,
                                   margin: float) -> Dict[str, Any]:
        """
        计算杠杆使用情况
        
        Returns:
            {leverage, margin_used, margin_ratio, is_safe}
        """
        position_value = position_size * price
        
        if margin <= 0:
            return {
                "leverage": 0,
                "margin_used": 0,
                "margin_ratio": 0,
                "is_safe": False
            }
        
        leverage = position_value / margin
        margin_used = position_value / self._max_leverage if self._max_leverage > 0 else position_value
        margin_ratio = margin_used / self._total_capital if self._total_capital > 0 else 0
        
        is_safe = leverage <= self._max_leverage and margin_ratio <= self._max_position_ratio
        
        return {
            "leverage": round(leverage, 2),
            "margin_used": round(margin_used, 4),
            "margin_ratio": round(margin_ratio, 4),
            "is_safe": is_safe
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "total_capital": self._total_capital,
                "max_leverage": self._max_leverage,
                "plans_count": len(self._position_plans),
                "total_plans_created": self._plan_count,
                "total_adds_executed": self._add_count,
                "active_symbols": list(self._position_plans.keys())
            }
    
    def clear_all_plans(self) -> None:
        """清除所有仓位计划"""
        with self._lock:
            self._position_plans.clear()
            self._add_history.clear()


_planner_instance: Optional[PositionPlanner] = None

def get_position_planner(config: Dict[str, Any] = None) -> PositionPlanner:
    """获取仓位规划器单例"""
    global _planner_instance
    if _planner_instance is None:
        _planner_instance = PositionPlanner(config)
    return _planner_instance