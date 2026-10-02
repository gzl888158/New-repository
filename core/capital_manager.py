"""
资金与仓位管理层系统
====================
核心定位：全局资金统一分配，动态调整20个币种仓位权重，防止单一币种重仓爆仓

五大核心模块：
1. 总资金池控制器 — 底仓资金 / 加仓备用资金 / 风险备用隔离金 三大分区
2. 币种权重动态分配 — 高波动率币种自动提升仓位占比，走弱币种自动降仓
3. 杠杆分级管理器 — 轻仓1-5倍 / 主仓5-15倍，严格禁止无上限高杠杆
4. 盈亏浮动再分配 — 当日盈利按比例划入加仓池；当日亏损强制收缩整体仓位
5. 多仓对冲调度 — 同币种多周期多方向仓位对冲，降低插针单边爆仓风险
"""

import asyncio
import threading
import time
import json
import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from loguru import logger

from core.capital_attrition_analyzer import (
    CapitalAttritionAnalyzer, AttritionType, AttritionBudget
)


# ============================================================================
# 模块一：总资金池控制器
# ============================================================================

class CapitalPoolType(Enum):
    """资金池类型"""
    BASE = "base"                    # 底仓资金（核心持仓）
    ADD_POSITION_RESERVE = "add_reserve"  # 加仓备用资金
    RISK_ISOLATION = "risk_isolation"     # 风险备用隔离金


@dataclass
class CapitalPool:
    """资金分区"""
    pool_type: CapitalPoolType
    total_amount: float              # 分区总额
    used_amount: float = 0.0         # 已用金额
    locked_amount: float = 0.0       # 锁定金额（挂单中）
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def available(self) -> float:
        """可用金额"""
        return max(0.0, self.total_amount - self.used_amount - self.locked_amount)

    @property
    def utilization(self) -> float:
        """使用率"""
        if self.total_amount <= 0:
            return 0.0
        return (self.used_amount + self.locked_amount) / self.total_amount

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pool_type": self.pool_type.value,
            "total": round(self.total_amount, 4),
            "used": round(self.used_amount, 4),
            "locked": round(self.locked_amount, 4),
            "available": round(self.available, 4),
            "utilization": round(self.utilization, 4),
            "last_updated": self.last_updated.isoformat()
        }


class CapitalPoolController:
    """
    总资金池控制器
    
    三大分区：
    - 底仓资金（60%）：用于初始建仓和核心持仓
    - 加仓备用资金（25%）：用于梯度加仓和马丁加仓
    - 风险备用隔离金（15%）：极端行情兜底，正常交易不可触碰
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        trading_config = self.config.get("trading", {})
        self._total_capital = trading_config.get("total_capital", 100.0)
        
        pool_config = self.config.get("capital_pool", {})
        self._base_ratio = pool_config.get("base_ratio", 0.60)
        self._add_reserve_ratio = pool_config.get("add_reserve_ratio", 0.25)
        self._risk_isolation_ratio = pool_config.get("risk_isolation_ratio", 0.15)
        
        # 确保比例之和为1
        total_ratio = self._base_ratio + self._add_reserve_ratio + self._risk_isolation_ratio
        if abs(total_ratio - 1.0) > 0.01:
            self._base_ratio = 0.60
            self._add_reserve_ratio = 0.25
            self._risk_isolation_ratio = 0.15
        
        self._pools: Dict[CapitalPoolType, CapitalPool] = {}
        self._lock = threading.RLock()
        
        # 每日盈亏记录
        self._daily_pnl: Dict[str, float] = {}  # date_str -> pnl
        self._daily_start_equity: Dict[str, float] = {}
        
        # 币种资金占用映射
        self._symbol_pool_usage: Dict[str, Dict[CapitalPoolType, float]] = defaultdict(
            lambda: {CapitalPoolType.BASE: 0.0, CapitalPoolType.ADD_POSITION_RESERVE: 0.0}
        )
        
        self._initialize_pools()
        
        self._rebalance_callback: Optional[Callable] = None
        
        logger.info(f"CapitalPoolController initialized: base={self._base_ratio:.0%}, "
                    f"add_reserve={self._add_reserve_ratio:.0%}, "
                    f"risk_isolation={self._risk_isolation_ratio:.0%}")

    def _initialize_pools(self) -> None:
        """初始化资金池"""
        with self._lock:
            base_amount = self._total_capital * self._base_ratio
            add_amount = self._total_capital * self._add_reserve_ratio
            risk_amount = self._total_capital * self._risk_isolation_ratio
            
            self._pools[CapitalPoolType.BASE] = CapitalPool(
                CapitalPoolType.BASE, base_amount
            )
            self._pools[CapitalPoolType.ADD_POSITION_RESERVE] = CapitalPool(
                CapitalPoolType.ADD_POSITION_RESERVE, add_amount
            )
            self._pools[CapitalPoolType.RISK_ISOLATION] = CapitalPool(
                CapitalPoolType.RISK_ISOLATION, risk_amount
            )

    def update_total_capital(self, total_capital: float) -> None:
        """更新总资金，按比例重新分配"""
        with self._lock:
            old_capital = self._total_capital
            self._total_capital = total_capital
            
            # 按当前使用比例缩放
            for pool in self._pools.values():
                if old_capital > 0:
                    utilization = pool.utilization
                    pool.total_amount = total_capital * self._get_pool_ratio(pool.pool_type)
                    pool.used_amount = pool.total_amount * utilization
                else:
                    pool.total_amount = total_capital * self._get_pool_ratio(pool.pool_type)
                pool.last_updated = datetime.now()
            
            today = date.today().isoformat()
            if today not in self._daily_start_equity:
                self._daily_start_equity[today] = total_capital
            
            logger.info(f"Capital updated: {old_capital:.2f} -> {total_capital:.2f}, "
                        f"pools rebalanced")

    def sync_total_capital_from_pools(self) -> float:
        """从各资金池总额反算_total_capital（PnL再分配后同步）"""
        with self._lock:
            self._total_capital = sum(p.total_amount for p in self._pools.values())
            return self._total_capital

    def _get_pool_ratio(self, pool_type: CapitalPoolType) -> float:
        """获取资金池比例"""
        if pool_type == CapitalPoolType.BASE:
            return self._base_ratio
        elif pool_type == CapitalPoolType.ADD_POSITION_RESERVE:
            return self._add_reserve_ratio
        else:
            return self._risk_isolation_ratio

    def allocate(self, symbol: str, amount: float, 
                 pool_type: CapitalPoolType = CapitalPoolType.BASE) -> bool:
        """
        从指定资金池分配资金
        
        Returns:
            是否分配成功
        """
        with self._lock:
            pool = self._pools.get(pool_type)
            if not pool:
                return False
            
            if amount <= 0:
                logger.warning(f"Invalid allocate amount {amount} for {symbol}, rejected")
                return False

            if pool.available < amount:
                logger.warning(f"Insufficient funds in {pool_type.value} pool: "
                             f"need {amount:.4f}, available {pool.available:.4f}")
                return False
            
            pool.used_amount += amount
            pool.last_updated = datetime.now()
            
            self._symbol_pool_usage[symbol][pool_type] += amount
            
            return True

    def release(self, symbol: str, amount: float,
                pool_type: CapitalPoolType = CapitalPoolType.BASE) -> None:
        """释放资金回到资金池"""
        with self._lock:
            pool = self._pools.get(pool_type)
            if not pool:
                return
            
            if amount <= 0:
                return

            release_amount = min(amount, pool.used_amount)
            pool.used_amount -= release_amount
            pool.last_updated = datetime.now()
            
            current = self._symbol_pool_usage[symbol].get(pool_type, 0)
            self._symbol_pool_usage[symbol][pool_type] = max(0, current - amount)

    def lock_funds(self, symbol: str, amount: float,
                   pool_type: CapitalPoolType = CapitalPoolType.BASE) -> bool:
        """锁定资金（挂单时）"""
        with self._lock:
            pool = self._pools.get(pool_type)
            if not pool or pool.available < amount:
                return False
            
            if amount <= 0:
                return False

            pool.locked_amount += amount
            pool.last_updated = datetime.now()
            return True

    def unlock_funds(self, symbol: str, amount: float,
                     pool_type: CapitalPoolType = CapitalPoolType.BASE) -> None:
        """解锁资金"""
        with self._lock:
            pool = self._pools.get(pool_type)
            if not pool:
                return
            
            if amount <= 0:
                return

            pool.locked_amount = max(0, pool.locked_amount - amount)
            pool.last_updated = datetime.now()

    def convert_locked_to_used(self, symbol: str, amount: float,
                                pool_type: CapitalPoolType = CapitalPoolType.BASE) -> None:
        """将锁定资金转为已用（订单成交）"""
        with self._lock:
            pool = self._pools.get(pool_type)
            if not pool:
                return
            
            if amount <= 0:
                return

            pool.locked_amount = max(0, pool.locked_amount - amount)
            pool.used_amount += amount
            pool.last_updated = datetime.now()
            
            self._symbol_pool_usage[symbol][pool_type] += amount

    def get_pool(self, pool_type: CapitalPoolType) -> CapitalPool:
        """获取资金池"""
        with self._lock:
            return self._pools.get(pool_type)

    def get_all_pools(self) -> Dict[str, CapitalPool]:
        """获取所有资金池"""
        with self._lock:
            return {pt.value: p for pt, p in self._pools.items()}

    def get_symbol_usage(self, symbol: str) -> Dict[str, float]:
        """获取币种资金使用情况"""
        with self._lock:
            usage = self._symbol_pool_usage.get(symbol, {})
            return {pt.value: amt for pt, amt in usage.items()}

    def get_total_available(self) -> float:
        """获取总可用资金"""
        with self._lock:
            return sum(p.available for p in self._pools.values())

    def get_total_used(self) -> float:
        """获取总已用资金"""
        with self._lock:
            return sum(p.used_amount for p in self._pools.values())

    def record_daily_pnl(self, pnl: float) -> None:
        """记录当日盈亏"""
        today = date.today().isoformat()
        with self._lock:
            self._daily_pnl[today] = self._daily_pnl.get(today, 0) + pnl

    def get_daily_pnl(self, day: str = None) -> float:
        """获取当日盈亏"""
        day = day or date.today().isoformat()
        with self._lock:
            return self._daily_pnl.get(day, 0)

    def trigger_risk_isolation(self, amount: float, reason: str) -> bool:
        """
        触发风险隔离金（极端情况）
        
        从风险隔离金中提取资金救市
        """
        with self._lock:
            pool = self._pools[CapitalPoolType.RISK_ISOLATION]
            if pool.available < amount:
                logger.error(f"Risk isolation fund insufficient: need {amount}, "
                            f"available {pool.available}")
                return False
            
            pool.used_amount += amount
            pool.last_updated = datetime.now()
            
            logger.warning(f"Risk isolation fund triggered: {amount:.4f} USDT, reason: {reason}")
            return True

    def rebalance_pools(self) -> None:
        """重新平衡资金池（每日结算时调用）。

        仅调整各池 total_amount 至目标比例，并清空挂单锁定金额（locked）。
        used_amount 表示仍在持仓中的保证金，不可清零，否则会错误释放仍在仓保证金。
        """
        with self._lock:
            current_total = self._total_capital
            
            for pool_type, ratio in [
                (CapitalPoolType.BASE, self._base_ratio),
                (CapitalPoolType.ADD_POSITION_RESERVE, self._add_reserve_ratio),
                (CapitalPoolType.RISK_ISOLATION, self._risk_isolation_ratio),
            ]:
                pool = self._pools[pool_type]
                target_amount = current_total * ratio
                pool.total_amount = target_amount
                # 保留在仓保证金（used_amount），仅约束其不超过池总额
                pool.used_amount = min(pool.used_amount, target_amount)
                pool.locked_amount = 0.0  # 每日结算清空挂单锁定
                pool.last_updated = datetime.now()
            
            logger.info("Capital pools rebalanced (used_amount preserved, locked reset)")

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_capital": round(self._total_capital, 4),
                "total_available": round(self.get_total_available(), 4),
                "total_used": round(self.get_total_used(), 4),
                "daily_pnl": round(self.get_daily_pnl(), 4),
                "pools": {pt.value: p.to_dict() for pt, p in self._pools.items()},
                "symbol_usage": {
                    sym: {pt.value: amt for pt, amt in usage.items()}
                    for sym, usage in self._symbol_pool_usage.items()
                    if any(v > 0 for v in usage.values())
                }
            }


# ============================================================================
# 模块二：币种权重动态分配算法
# ============================================================================

class SymbolWeightAllocator:
    """
    币种权重动态分配器
    
    核心算法：
    - 波动率因子：高波动率币种权重提升（激进策略需要）
    - 动量因子：强势币种权重提升
    - 流动性因子：高流动性币种基础权重更高
    - 相关性惩罚：高相关币种组合降权
    - 单币种上限：防止单一币种重仓
    """

    def __init__(self, config: Dict[str, Any] = None, 
                 capital_pool: CapitalPoolController = None):
        self.config = config or {}
        self._capital_pool = capital_pool
        
        alloc_config = self.config.get("symbol_allocation", {})
        self._max_symbol_weight = alloc_config.get("max_symbol_weight", 0.15)  # 单币种最大15%
        self._min_symbol_weight = alloc_config.get("min_symbol_weight", 0.02)  # 单币种最小2%
        self._rebalance_interval = alloc_config.get("rebalance_interval", 300)  # 5分钟
        self._max_weight_change = alloc_config.get("max_weight_change", 0.03)   # 单次最大变更3%
        
        # 因子权重
        self._volatility_weight = alloc_config.get("volatility_weight", 0.35)
        self._momentum_weight = alloc_config.get("momentum_weight", 0.25)
        self._liquidity_weight = alloc_config.get("liquidity_weight", 0.20)
        self._performance_weight = alloc_config.get("performance_weight", 0.20)
        
        self._symbol_weights: Dict[str, float] = {}
        self._symbol_metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        self._symbols: List[str] = []
        
        self._adjustment_count = 0
        self._last_rebalance = datetime.min

    def initialize(self, symbols: List[str]) -> None:
        """初始化币种列表，等权分配"""
        with self._lock:
            self._symbols = symbols
            equal_weight = 1.0 / len(symbols) if symbols else 0
            for symbol in symbols:
                self._symbol_weights[symbol] = equal_weight
                self._symbol_metrics[symbol] = {
                    "volatility": 0.02,
                    "momentum": 0.0,
                    "liquidity": 0.5,
                    "pnl": 0.0,
                    "win_rate": 0.5,
                }
            
            logger.info(f"SymbolWeightAllocator initialized for {len(symbols)} symbols, "
                        f"equal weight={equal_weight:.4f}")

    def update_symbol_metrics(self, symbol: str, volatility: float,
                               momentum: float, liquidity: float,
                               pnl: float = 0.0, win_rate: float = 0.5) -> None:
        """更新币种指标"""
        with self._lock:
            if symbol not in self._symbol_metrics:
                self._symbol_metrics[symbol] = {}
            
            self._symbol_metrics[symbol].update({
                "volatility": max(volatility, 0.001),
                "momentum": momentum,
                "liquidity": max(liquidity, 0.01),
                "pnl": pnl,
                "win_rate": max(0.0, min(1.0, win_rate)),
                "updated_at": datetime.now().isoformat()
            })

    def calculate_optimal_weights(self) -> Dict[str, float]:
        """
        计算最优权重

        高波动率币种自动降低仓位占比（风险控制）
        走弱币种自动降仓
        """
        with self._lock:
            if not self._symbols:
                return {}
            
            scores = {}
            total_score = 0.0
            
            for symbol in self._symbols:
                metrics = self._symbol_metrics.get(symbol, {})
                
                # 波动率因子：波动率越高，分数越低（风险控制：高波动币种分配更小仓位）
                vol = metrics.get("volatility", 0.02)
                vol_score = max(0.2, 2.0 - min(vol / 0.025, 1.8))  # 波动率0→2.0, 波动率5%→0.2
                
                # 动量因子：正动量加分，负动量减分
                momentum = metrics.get("momentum", 0.0)
                momentum_score = 1.0 + np.tanh(momentum * 10)  # tanh归一化到0-2
                
                # 流动性因子：高流动性基础权重更高
                liquidity = metrics.get("liquidity", 0.5)
                liq_score = min(liquidity * 2, 2.0)
                
                # 表现因子：盈利和胜率
                win_rate = metrics.get("win_rate", 0.5)
                perf_score = win_rate * 2.0
                
                # 综合得分
                composite_score = (
                    vol_score * self._volatility_weight +
                    momentum_score * self._momentum_weight +
                    liq_score * self._liquidity_weight +
                    perf_score * self._performance_weight
                )
                
                scores[symbol] = composite_score
                total_score += composite_score
            
            # 归一化为权重
            new_weights = {}
            if total_score > 0:
                for symbol, score in scores.items():
                    new_weights[symbol] = score / total_score
            else:
                equal = 1.0 / len(self._symbols)
                new_weights = {s: equal for s in self._symbols}
            
            # 限制单币种权重范围
            for symbol in new_weights:
                new_weights[symbol] = max(self._min_symbol_weight,
                                         min(self._max_symbol_weight, new_weights[symbol]))
            
            # 再次归一化
            total = sum(new_weights.values())
            if total > 0:
                new_weights = {s: w / total for s, w in new_weights.items()}
            
            return new_weights

    def rebalance(self) -> Dict[str, float]:
        """
        执行权重再平衡
        
        单次变更不超过 max_weight_change
        """
        with self._lock:
            new_weights = self.calculate_optimal_weights()
            
            old_weights = self._symbol_weights.copy()
            
            adjusted = {}
            for symbol in self._symbols:
                old_w = old_weights.get(symbol, 0)
                new_w = new_weights.get(symbol, 0)
                
                # 限制单次变更幅度
                change = new_w - old_w
                max_change = self._max_weight_change
                if abs(change) > max_change:
                    new_w = old_w + (max_change if change > 0 else -max_change)
                
                adjusted[symbol] = new_w
            
            # 归一化
            total = sum(adjusted.values())
            if total > 0:
                adjusted = {s: w / total for s, w in adjusted.items()}
            
            self._symbol_weights = adjusted
            self._last_rebalance = datetime.now()
            self._adjustment_count += 1
            
            # 记录显著变更
            for symbol in self._symbols:
                old_w = old_weights.get(symbol, 0)
                new_w = adjusted.get(symbol, 0)
                if abs(new_w - old_w) > 0.01:
                    logger.info(f"Symbol weight adjusted: {symbol} "
                               f"{old_w:.4f} -> {new_w:.4f}")
            
            return adjusted

    def get_weight(self, symbol: str) -> float:
        """获取币种权重"""
        with self._lock:
            return self._symbol_weights.get(symbol, 0.0)

    def get_all_weights(self) -> Dict[str, float]:
        """获取所有币种权重"""
        with self._lock:
            return self._symbol_weights.copy()

    def get_capital_allocation(self, symbol: str, pool_type: CapitalPoolType = None) -> float:
        """
        获取币种可分配资金
        
        权重 * 资金池金额
        """
        with self._lock:
            weight = self._symbol_weights.get(symbol, 0.0)
            
            if self._capital_pool and pool_type:
                pool = self._capital_pool.get_pool(pool_type)
                if pool:
                    return weight * pool.available
            
            if self._capital_pool:
                return weight * self._capital_pool.get_total_available()
            
            return 0.0

    def should_rebalance(self) -> bool:
        """是否需要再平衡"""
        elapsed = (datetime.now() - self._last_rebalance).total_seconds()
        return elapsed >= self._rebalance_interval

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "weights": {s: round(w, 4) for s, w in self._symbol_weights.items()},
                "metrics": {
                    s: {k: round(v, 4) if isinstance(v, float) else v 
                        for k, v in m.items()}
                    for s, m in self._symbol_metrics.items()
                },
                "adjustment_count": self._adjustment_count,
                "last_rebalance": self._last_rebalance.isoformat() if self._last_rebalance != datetime.min else None,
                "max_symbol_weight": self._max_symbol_weight,
                "min_symbol_weight": self._min_symbol_weight
            }


# ============================================================================
# 模块三：杠杆分级管理器
# ============================================================================

class LeverageTier(Enum):
    """杠杆等级"""
    LIGHT = "light"     # 轻仓 1-5倍
    MAIN = "main"       # 主仓 5-15倍
    HEAVY = "heavy"     # 重仓（仅特殊场景，严格限制）
    FORBIDDEN = "forbidden"  # 禁止（>15倍）


@dataclass
class LeverageAssignment:
    """杠杆分配"""
    symbol: str
    tier: LeverageTier
    leverage: float
    max_losing_pct: float  # 该杠杆下最大可承受亏损比例
    margin_per_position: float
    reason: str


class LeverageTierManager:
    """
    杠杆分级管理器
    
    严格分级：
    - 轻仓 1-5倍：试探性建仓、高波动币种初始仓位
    - 主仓 5-15倍：核心持仓、趋势确认后加仓
    - 严禁超过15倍：系统硬性限制
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        lev_config = self.config.get("leverage_tiers", {})
        self._light_min = lev_config.get("light_min", 1)
        self._light_max = lev_config.get("light_max", 5)
        self._main_min = lev_config.get("main_min", 5)
        self._main_max = lev_config.get("main_max", 15)
        self._absolute_max = lev_config.get("absolute_max", 15)  # 绝对上限
        
        # 币种杠杆配置（来自config.yaml currencies段）
        currencies = self.config.get("currencies", {})
        self._symbol_leverage_config: Dict[str, Dict[str, float]] = {}
        
        for tier_name in ["tier1_symbols", "tier2_symbols", "tier3_symbols"]:
            for base in currencies.get(tier_name, []):
                symbol = f"{base}-USDT-SWAP"
                tier_config = currencies.get(f"tier{'1' if 'tier1' in tier_name else '2' if 'tier2' in tier_name else '3'}", {})
                self._symbol_leverage_config[symbol] = {
                    "min": tier_config.get("leverage_min", 1),
                    "max": tier_config.get("leverage_max", 10),
                    "default": tier_config.get("leverage_default", 5),
                }
        
        # 当前持仓杠杆
        self._current_leverages: Dict[str, float] = {}
        
        # 杠杆调整历史
        self._adjustment_history: List[Dict[str, Any]] = []
        
        self._lock = threading.RLock()
        
        logger.info(f"LeverageTierManager initialized: light={self._light_min}-{self._light_max}x, "
                    f"main={self._main_min}-{self._main_max}x, max={self._absolute_max}x")

    def assign_leverage(self, symbol: str, position_type: str = "initial",
                        volatility: float = 0.02, signal_strength: float = 0.5,
                        account_drawdown: float = 0.0) -> LeverageAssignment:
        """
        为仓位分配杠杆
        
        Args:
            symbol: 交易对
            position_type: "initial"（初始建仓）, "add"（加仓）, "hedge"（对冲）
            volatility: 当前波动率
            signal_strength: 信号强度 0-1
            account_drawdown: 账户回撤比例
        
        Returns:
            LeverageAssignment
        """
        with self._lock:
            # 获取币种杠杆配置
            sym_config = self._symbol_leverage_config.get(symbol, {
                "min": 1, "max": 10, "default": 5
            })
            
            # 根据仓位类型确定基础杠杆等级
            if position_type == "initial":
                # 初始建仓：轻仓起步
                base_tier = LeverageTier.LIGHT
                base_leverage = min(sym_config["default"], self._light_max)
            elif position_type == "add":
                # 加仓：信号强且回撤小可用主仓
                if signal_strength > 0.7 and account_drawdown < 0.05:
                    base_tier = LeverageTier.MAIN
                    base_leverage = min(self._main_max, sym_config["max"])
                else:
                    base_tier = LeverageTier.LIGHT
                    base_leverage = min(self._light_max, sym_config["default"])
            elif position_type == "hedge":
                # 对冲仓：固定轻仓
                base_tier = LeverageTier.LIGHT
                base_leverage = min(3, self._light_max)
            else:
                base_tier = LeverageTier.LIGHT
                base_leverage = sym_config["default"]
            
            # 波动率调整：高波动降杠杆
            if volatility > 0.05:
                leverage_adjustment = 0.7  # 降30%
            elif volatility > 0.03:
                leverage_adjustment = 0.85
            elif volatility < 0.01:
                leverage_adjustment = 1.15  # 低波动可适当提高
            else:
                leverage_adjustment = 1.0
            
            # 回撤调整：有回撤时降杠杆
            if account_drawdown > 0.10:
                leverage_adjustment *= 0.5
            elif account_drawdown > 0.05:
                leverage_adjustment *= 0.7
            elif account_drawdown > 0.02:
                leverage_adjustment *= 0.85
            
            final_leverage = base_leverage * leverage_adjustment
            
            # 确保在范围内
            final_leverage = max(self._light_min, min(self._absolute_max, final_leverage))
            final_leverage = max(sym_config["min"], min(sym_config["max"], final_leverage))
            
            # 确定最终等级
            if final_leverage <= self._light_max:
                tier = LeverageTier.LIGHT
            elif final_leverage <= self._main_max:
                tier = LeverageTier.MAIN
            else:
                # 不应到达此处，但作为安全措施
                final_leverage = self._main_max
                tier = LeverageTier.MAIN
            
            # 计算该杠杆下最大可承受亏损（到强平距离的50%）
            max_losing_pct = 1.0 / (final_leverage * 2)  # 保守估计
            
            assignment = LeverageAssignment(
                symbol=symbol,
                tier=tier,
                leverage=round(final_leverage, 1),
                max_losing_pct=round(max_losing_pct, 4),
                margin_per_position=0.0,  # 由调用方计算
                reason=f"{position_type} position, vol={volatility:.4f}, "
                       f"signal={signal_strength:.2f}, drawdown={account_drawdown:.4f}"
            )
            
            self._current_leverages[symbol] = final_leverage
            self._adjustment_history.append({
                "symbol": symbol,
                "leverage": final_leverage,
                "tier": tier.value,
                "position_type": position_type,
                "timestamp": datetime.now().isoformat()
            })
            
            return assignment

    def check_leverage(self, symbol: str, requested_leverage: float) -> Tuple[bool, float, str]:
        """
        检查杠杆是否合规
        
        Returns:
            (is_compliant, adjusted_leverage, reason)
        """
        with self._lock:
            if requested_leverage > self._absolute_max:
                return False, self._absolute_max, f"杠杆{requested_leverage}超过绝对上限{self._absolute_max}"
            
            sym_config = self._symbol_leverage_config.get(symbol, {})
            sym_max = sym_config.get("max", self._absolute_max)
            
            if requested_leverage > sym_max:
                return False, sym_max, f"杠杆{requested_leverage}超过{symbol}最大杠杆{sym_max}"
            
            return True, requested_leverage, "合规"

    def get_current_leverage(self, symbol: str) -> float:
        """获取当前杠杆"""
        with self._lock:
            return self._current_leverages.get(symbol, 0)

    def get_tier_stats(self) -> Dict[str, Any]:
        """获取杠杆分布统计"""
        with self._lock:
            tier_counts = {tier.value: 0 for tier in LeverageTier}
            for lev in self._current_leverages.values():
                if lev <= self._light_max:
                    tier_counts[LeverageTier.LIGHT.value] += 1
                elif lev <= self._main_max:
                    tier_counts[LeverageTier.MAIN.value] += 1
                else:
                    tier_counts[LeverageTier.HEAVY.value] += 1
            
            return {
                "tier_distribution": tier_counts,
                "current_leverages": {s: round(l, 1) for s, l in self._current_leverages.items()},
                "absolute_max": self._absolute_max,
                "total_adjustments": len(self._adjustment_history)
            }

    def to_dict(self) -> Dict[str, Any]:
        return self.get_tier_stats()


# ============================================================================
# 模块四：盈亏浮动再分配单元
# ============================================================================

class PnLReallocationUnit:
    """
    盈亏浮动再分配单元
    
    核心逻辑：
    - 当日盈利：按比例（默认50%）划入加仓备用池，剩余留在底仓
    - 当日亏损：强制收缩整体仓位，从加仓池回补底仓
    - 连续盈利：逐步提升加仓池比例（最高40%）
    - 连续亏损：逐步收缩仓位（最低底仓50%）
    """

    def __init__(self, config: Dict[str, Any] = None,
                 capital_pool: CapitalPoolController = None):
        self.config = config or {}
        self._capital_pool = capital_pool
        
        realloc_config = self.config.get("pnl_reallocation", {})
        self._profit_to_add_ratio = realloc_config.get("profit_to_add_ratio", 0.50)
        self._loss_shrink_ratio = realloc_config.get("loss_shrink_ratio", 0.30)
        self._consecutive_profit_threshold = realloc_config.get("consecutive_profit_days", 3)
        self._consecutive_loss_threshold = realloc_config.get("consecutive_loss_days", 2)
        self._max_add_pool_ratio = realloc_config.get("max_add_pool_ratio", 0.40)
        self._min_base_pool_ratio = realloc_config.get("min_base_pool_ratio", 0.50)
        
        self._daily_pnl_history: deque = deque(maxlen=30)
        self._consecutive_profit_days = 0
        self._consecutive_loss_days = 0
        
        self._last_reallocation_date: Optional[str] = None
        self._reallocation_count = 0
        self._total_profit_allocated = 0.0
        self._total_loss_absorbed = 0.0
        
        self._lock = threading.RLock()
        
        logger.info(f"PnLReallocationUnit initialized: profit_to_add={self._profit_to_add_ratio:.0%}, "
                    f"loss_shrink={self._loss_shrink_ratio:.0%}")

    def process_daily_pnl(self, pnl: float, current_equity: float) -> Dict[str, Any]:
        """
        处理当日盈亏，执行再分配
        
        Returns:
            再分配详情
        """
        with self._lock:
            today = date.today().isoformat()
            
            if self._last_reallocation_date == today:
                return {"status": "already_processed", "action": "already_processed", "date": today}
            
            result = {
                "date": today,
                "daily_pnl": round(pnl, 4),
                "action": "none",
                "details": {},
                "timestamp": datetime.now().isoformat()
            }
            
            self._daily_pnl_history.append({"date": today, "pnl": pnl})
            
            if pnl > 0:
                # 当日盈利
                self._consecutive_profit_days += 1
                self._consecutive_loss_days = 0
                
                # 连续盈利提升加仓池比例
                add_ratio = self._profit_to_add_ratio
                if self._consecutive_profit_days >= self._consecutive_profit_threshold:
                    add_ratio = min(add_ratio + 0.1 * (self._consecutive_profit_days - self._consecutive_profit_threshold + 1),
                                   self._max_add_pool_ratio)
                
                profit_to_add = pnl * add_ratio
                profit_to_base = pnl - profit_to_add
                
                result["action"] = "profit_reallocation"
                result["details"] = {
                    "profit_to_add_pool": round(profit_to_add, 4),
                    "profit_to_base_pool": round(profit_to_base, 4),
                    "add_ratio": round(add_ratio, 4),
                    "consecutive_profit_days": self._consecutive_profit_days
                }
                
                self._total_profit_allocated += profit_to_add
                
                if self._capital_pool:
                    # 实际调整资金池
                    self._capital_pool.get_pool(CapitalPoolType.ADD_POSITION_RESERVE).total_amount += profit_to_add
                    self._capital_pool.get_pool(CapitalPoolType.BASE).total_amount += profit_to_base
                    self._capital_pool.sync_total_capital_from_pools()
                
                logger.info(f"Daily profit reallocation: +{pnl:.4f} USDT, "
                           f"add_pool +={profit_to_add:.4f}, base_pool +={profit_to_base:.4f}")
            
            elif pnl < 0:
                # 当日亏损
                self._consecutive_loss_days += 1
                self._consecutive_profit_days = 0
                
                # 连续亏损加速收缩
                shrink_ratio = self._loss_shrink_ratio
                if self._consecutive_loss_days >= self._consecutive_loss_threshold:
                    shrink_ratio = min(shrink_ratio + 0.1 * self._consecutive_loss_days, 0.6)
                
                # 从加仓池回补底仓
                add_pool = self._capital_pool.get_pool(CapitalPoolType.ADD_POSITION_RESERVE) if self._capital_pool else None
                add_pool_available = add_pool.available if add_pool else 0
                
                transfer_amount = min(abs(pnl) * shrink_ratio, add_pool_available)
                
                result["action"] = "loss_shrinkage"
                result["details"] = {
                    "loss_amount": round(abs(pnl), 4),
                    "shrink_ratio": round(shrink_ratio, 4),
                    "transfer_from_add_pool": round(transfer_amount, 4),
                    "consecutive_loss_days": self._consecutive_loss_days
                }
                
                self._total_loss_absorbed += transfer_amount
                
                if self._capital_pool:
                    # P0-9 关键修复：亏损必须扣减资金池总额。原实现仅在 add→base 回补时
                    # 调整 total_amount（且是中性转移），账户亏损 |pnl| 从未入账，
                    # 导致 _total_capital 被高估。此处先在 base 池扣减亏损，再执行回补。
                    base_pool = self._capital_pool.get_pool(CapitalPoolType.BASE)
                    base_pool.total_amount = max(0.0, base_pool.total_amount - abs(pnl))
                    if transfer_amount > 0:
                        add_pool.total_amount -= transfer_amount
                        base_pool.total_amount += transfer_amount
                    self._capital_pool.sync_total_capital_from_pools()
                
                logger.warning(f"Daily loss shrinkage: {pnl:.4f} USDT, "
                              f"add_pool -> base_pool: {transfer_amount:.4f}, "
                              f"shrink_ratio={shrink_ratio:.2f}")
            
            self._last_reallocation_date = today
            self._reallocation_count += 1
            
            return result

    def get_reallocation_stats(self) -> Dict[str, Any]:
        """获取再分配统计"""
        with self._lock:
            recent_pnl = list(self._daily_pnl_history)[-7:]
            
            return {
                "consecutive_profit_days": self._consecutive_profit_days,
                "consecutive_loss_days": self._consecutive_loss_days,
                "total_profit_allocated": round(self._total_profit_allocated, 4),
                "total_loss_absorbed": round(self._total_loss_absorbed, 4),
                "reallocation_count": self._reallocation_count,
                "last_date": self._last_reallocation_date,
                "recent_7d_pnl": [{"date": d["date"], "pnl": round(d["pnl"], 4)} for d in recent_pnl]
            }

    def to_dict(self) -> Dict[str, Any]:
        return self.get_reallocation_stats()


# ============================================================================
# 模块五：多仓对冲调度模块
# ============================================================================

class HedgeType(Enum):
    """对冲类型"""
    SAME_SYMBOL_DIFF_PERIOD = "same_symbol_diff_period"  # 同币种不同周期
    SAME_SYMBOL_DIFF_DIRECTION = "same_symbol_diff_direction"  # 同币种不同方向
    CORRELATED_PAIRS = "correlated_pairs"  # 高相关币种对冲
    DELTA_NEUTRAL = "delta_neutral"  # Delta中性对冲


@dataclass
class HedgePosition:
    """对冲仓位"""
    hedge_id: str
    hedge_type: HedgeType
    primary_symbol: str
    primary_side: str  # "long" or "short"
    primary_size: float
    primary_leverage: float
    primary_entry: float
    
    hedge_symbol: str
    hedge_side: str
    hedge_size: float
    hedge_leverage: float
    hedge_entry: float
    
    hedge_ratio: float  # 对冲比例
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, closed, expired
    
    def net_exposure(self, primary_price: float, hedge_price: float) -> float:
        """计算净敞口"""
        primary_value = self.primary_size * primary_price * (1 if self.primary_side == "long" else -1)
        hedge_value = self.hedge_size * hedge_price * (1 if self.hedge_side == "long" else -1)
        return primary_value + hedge_value


class HedgeScheduler:
    """
    多仓对冲调度模块
    
    核心功能：
    - 同币种多周期多方向仓位对冲
    - 降低插针单边爆仓风险
    - 动态调整对冲比例
    - 对冲成本控制
    """

    def __init__(self, config: Dict[str, Any] = None,
                 leverage_manager: LeverageTierManager = None):
        self.config = config or {}
        self._leverage_manager = leverage_manager
        
        hedge_config = self.config.get("hedge_scheduler", {})
        self._max_hedge_ratio = hedge_config.get("max_hedge_ratio", 0.7)
        self._min_hedge_ratio = hedge_config.get("min_hedge_ratio", 0.3)
        self._hedge_trigger_volatility = hedge_config.get("hedge_trigger_volatility", 0.04)
        self._hedge_trigger_drawdown = hedge_config.get("hedge_trigger_drawdown", 0.05)
        self._max_hedge_positions = hedge_config.get("max_hedge_positions", 10)
        self._hedge_check_interval = hedge_config.get("hedge_check_interval", 60)
        
        self._active_hedges: Dict[str, HedgePosition] = {}
        self._hedge_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        self._hedge_created_count = 0
        self._hedge_closed_count = 0
        
        logger.info("HedgeScheduler initialized for multi-position hedging")

    def evaluate_hedge_need(self, symbol: str, position_side: str,
                            position_size: float, entry_price: float,
                            current_price: float, volatility: float,
                            unrealized_pnl_pct: float) -> Optional[Dict[str, Any]]:
        """
        评估是否需要对冲
        
        Returns:
            对冲建议或None
        """
        # 高波动率触发对冲
        if volatility < self._hedge_trigger_volatility:
            return None
        
        # 亏损达到阈值触发对冲
        if unrealized_pnl_pct > -self._hedge_trigger_drawdown:
            return None
        
        with self._lock:
            # 检查是否已有该币种的对冲
            existing = [h for h in self._active_hedges.values() 
                       if h.primary_symbol == symbol and h.status == "active"]
            
            if len(existing) >= 2:  # 单币种最多2个对冲
                return None
            
            if len(self._active_hedges) >= self._max_hedge_positions:
                return None
            
            # 计算对冲比例
            # 波动率越高，对冲比例越大
            vol_factor = min(volatility / self._hedge_trigger_volatility, 2.0)
            hedge_ratio = min(self._min_hedge_ratio + 0.2 * (vol_factor - 1),
                            self._max_hedge_ratio)
            
            # 亏损越大，对冲比例越大
            loss_factor = abs(unrealized_pnl_pct) / self._hedge_trigger_drawdown
            hedge_ratio = min(hedge_ratio + 0.1 * (loss_factor - 1),
                            self._max_hedge_ratio)
            
            hedge_ratio = max(self._min_hedge_ratio, 
                            min(self._max_hedge_ratio, hedge_ratio))
            
            # 对冲方向
            hedge_side = "short" if position_side == "long" else "long"
            hedge_size = position_size * hedge_ratio
            
            # 对冲杠杆（轻仓）
            if self._leverage_manager:
                lev_assignment = self._leverage_manager.assign_leverage(
                    symbol, position_type="hedge", volatility=volatility
                )
                hedge_leverage = lev_assignment.leverage
            else:
                hedge_leverage = 3.0
            
            suggestion = {
                "symbol": symbol,
                "hedge_type": HedgeType.SAME_SYMBOL_DIFF_DIRECTION.value,
                "hedge_side": hedge_side,
                "hedge_size": round(hedge_size, 8),
                "hedge_ratio": round(hedge_ratio, 4),
                "hedge_leverage": hedge_leverage,
                "primary_side": position_side,
                "primary_size": position_size,
                "primary_entry": entry_price,
                "current_price": current_price,
                "reason": f"波动率{volatility*100:.2f}%超过阈值，亏损{unrealized_pnl_pct*100:.2f}%，启动对冲"
            }
            
            return suggestion

    def create_hedge(self, suggestion: Dict[str, Any]) -> Optional[HedgePosition]:
        """创建对冲仓位"""
        with self._lock:
            hedge_id = f"hedge_{suggestion['symbol']}_{int(time.time())}"
            
            hedge = HedgePosition(
                hedge_id=hedge_id,
                hedge_type=HedgeType(suggestion.get("hedge_type", "same_symbol_diff_direction")),
                primary_symbol=suggestion["symbol"],
                primary_side=suggestion["primary_side"],
                primary_size=suggestion["primary_size"],
                primary_leverage=suggestion.get("primary_leverage", 5.0),
                primary_entry=suggestion["primary_entry"],
                hedge_symbol=suggestion["symbol"],
                hedge_side=suggestion["hedge_side"],
                hedge_size=suggestion["hedge_size"],
                hedge_leverage=suggestion["hedge_leverage"],
                hedge_entry=suggestion["current_price"],
                hedge_ratio=suggestion["hedge_ratio"]
            )
            
            self._active_hedges[hedge_id] = hedge
            self._hedge_created_count += 1
            
            logger.info(f"Hedge created: {hedge_id}, "
                       f"{suggestion['symbol']} {suggestion['hedge_side']} "
                       f"size={suggestion['hedge_size']:.6f}, ratio={suggestion['hedge_ratio']:.2f}")
            
            return hedge

    def close_hedge(self, hedge_id: str, reason: str = "manual") -> bool:
        """关闭对冲仓位"""
        with self._lock:
            if hedge_id not in self._active_hedges:
                return False
            
            hedge = self._active_hedges[hedge_id]
            hedge.status = "closed"
            
            self._hedge_history.append({
                "hedge_id": hedge_id,
                "symbol": hedge.primary_symbol,
                "hedge_ratio": hedge.hedge_ratio,
                "created_at": hedge.created_at.isoformat(),
                "closed_at": datetime.now().isoformat(),
                "reason": reason
            })
            
            del self._active_hedges[hedge_id]
            self._hedge_closed_count += 1
            
            logger.info(f"Hedge closed: {hedge_id}, reason: {reason}")
            return True

    def get_active_hedges(self, symbol: str = None) -> List[HedgePosition]:
        """获取活跃对冲仓位"""
        with self._lock:
            hedges = list(self._active_hedges.values())
            if symbol:
                hedges = [h for h in hedges if h.primary_symbol == symbol]
            return hedges

    def calculate_net_exposure(self, symbol: str, current_price: float) -> float:
        """计算币种的净敞口（含对冲）"""
        with self._lock:
            total_exposure = 0.0
            for hedge in self._active_hedges.values():
                if hedge.primary_symbol == symbol and hedge.status == "active":
                    total_exposure += hedge.net_exposure(current_price, current_price)
            return total_exposure

    def get_hedge_stats(self) -> Dict[str, Any]:
        """获取对冲统计"""
        with self._lock:
            return {
                "active_hedges": len(self._active_hedges),
                "total_created": self._hedge_created_count,
                "total_closed": self._hedge_closed_count,
                "active_details": [
                    {
                        "hedge_id": h.hedge_id,
                        "symbol": h.primary_symbol,
                        "hedge_type": h.hedge_type.value,
                        "hedge_ratio": h.hedge_ratio,
                        "primary_side": h.primary_side,
                        "hedge_side": h.hedge_side,
                        "created_at": h.created_at.isoformat()
                    }
                    for h in self._active_hedges.values()
                ]
            }

    def to_dict(self) -> Dict[str, Any]:
        return self.get_hedge_stats()


# ============================================================================
# 统一管理入口：CapitalManager
# ============================================================================

class CapitalManager:
    """
    资金与仓位管理统一入口
    
    整合五大模块：
    1. 总资金池控制器
    2. 币种权重动态分配
    3. 杠杆分级管理器
    4. 盈亏浮动再分配
    5. 多仓对冲调度
    """

    def __init__(self, config: Dict[str, Any] = None, adaptive_controller=None):
        self.config = config or {}
        self._adaptive_controller = adaptive_controller
        
        # 初始化五大模块
        self.capital_pool = CapitalPoolController(config)
        self.symbol_allocator = SymbolWeightAllocator(config, self.capital_pool)
        self.leverage_manager = LeverageTierManager(config)
        self.pnl_reallocation = PnLReallocationUnit(config, self.capital_pool)
        self.hedge_scheduler = HedgeScheduler(config, self.leverage_manager)
        
        # 初始化资金磨损分析器（第六模块）
        self.attrition_analyzer = CapitalAttritionAnalyzer(config)
        self._init_attrition_budgets()
        
        self._running = False
        self._rebalance_task: Optional[asyncio.Task] = None
        self._lock = threading.RLock()
        
        # 风险预算缓存（从 AdaptiveController 同步）
        self._risk_budget_allocation: Dict[str, float] = {}
        self._last_risk_budget_sync: datetime = datetime.min
        
        logger.info("CapitalManager initialized with 6 modules (incl. attrition analyzer)")

    def initialize(self, symbols: List[str], total_capital: float = None) -> None:
        """初始化资金管理器"""
        if total_capital:
            self.capital_pool.update_total_capital(total_capital)
        
        self.symbol_allocator.initialize(symbols)
        
        logger.info(f"CapitalManager initialized: {len(symbols)} symbols, "
                    f"capital={self.capital_pool._total_capital:.2f} USDT")

    def set_adaptive_controller(self, adaptive_controller) -> None:
        """设置 AdaptiveController 引用（用于风险预算协作）"""
        self._adaptive_controller = adaptive_controller
        logger.info("AdaptiveController linked to CapitalManager for risk budget coordination")

    def update_capital(self, total_equity: float) -> None:
        """更新总权益"""
        self.capital_pool.update_total_capital(total_equity)

    def allocate_capital(self, symbol: str, amount: float,
                         pool_type: CapitalPoolType = CapitalPoolType.BASE) -> bool:
        """分配资金给币种"""
        return self.capital_pool.allocate(symbol, amount, pool_type)

    def get_symbol_capital(self, symbol: str) -> float:
        """获取币种可分配资金（含风险预算约束）"""
        base_capital = self.symbol_allocator.get_capital_allocation(symbol, CapitalPoolType.BASE)
        
        # 如果有 AdaptiveController，施加风险预算约束
        if self._adaptive_controller:
            try:
                risk_status = self._adaptive_controller.get_risk_budget_status()
                # 获取该symbol对应策略的风险预算使用率
                total_budget = risk_status.get("daily_total_budget", 0)
                total_consumed = risk_status.get("total_consumed", 0)
                if total_budget > 0 and total_consumed / total_budget > 0.8:
                    # 总体风险预算消耗超过80%，缩减单币种分配
                    utilization_ratio = total_consumed / total_budget
                    scale = max(0.3, 1.0 - (utilization_ratio - 0.8) * 5)
                    base_capital *= scale
            except Exception as e:
                logger.warning(f"Risk budget scale check failed: {e}")
        
        return base_capital

    def get_symbol_weight(self, symbol: str) -> float:
        """获取币种权重"""
        return self.symbol_allocator.get_weight(symbol)

    def assign_leverage(self, symbol: str, position_type: str = "initial",
                        volatility: float = 0.02, signal_strength: float = 0.5,
                        account_drawdown: float = 0.0) -> LeverageAssignment:
        """分配杠杆"""
        return self.leverage_manager.assign_leverage(
            symbol, position_type, volatility, signal_strength, account_drawdown
        )

    def check_leverage(self, symbol: str, leverage: float) -> Tuple[bool, float, str]:
        """检查杠杆合规性"""
        return self.leverage_manager.check_leverage(symbol, leverage)

    def evaluate_hedge(self, symbol: str, position_side: str, position_size: float,
                       entry_price: float, current_price: float, volatility: float,
                       unrealized_pnl_pct: float) -> Optional[Dict[str, Any]]:
        """评估对冲需求"""
        return self.hedge_scheduler.evaluate_hedge_need(
            symbol, position_side, position_size, entry_price,
            current_price, volatility, unrealized_pnl_pct
        )

    def record_daily_pnl(self, pnl: float, current_equity: float) -> Dict[str, Any]:
        """记录当日盈亏并执行再分配"""
        self.capital_pool.record_daily_pnl(pnl)
        return self.pnl_reallocation.process_daily_pnl(pnl, current_equity)

    def update_symbol_metrics(self, symbol: str, volatility: float,
                               momentum: float, liquidity: float,
                               pnl: float = 0.0, win_rate: float = 0.5) -> None:
        """更新币种指标"""
        self.symbol_allocator.update_symbol_metrics(
            symbol, volatility, momentum, liquidity, pnl, win_rate
        )

    async def start_periodic_tasks(self) -> None:
        """启动定期任务"""
        self._running = True
        self._rebalance_task = asyncio.create_task(self._rebalance_loop())
        logger.info("CapitalManager periodic tasks started")

    async def stop_periodic_tasks(self) -> None:
        """停止定期任务"""
        self._running = False
        if self._rebalance_task:
            self._rebalance_task.cancel()
            self._rebalance_task = None
        logger.info("CapitalManager periodic tasks stopped")

    async def _rebalance_loop(self) -> None:
        """定期再平衡循环（含风险预算同步）"""
        while self._running:
            try:
                await asyncio.sleep(self.symbol_allocator._rebalance_interval)
                
                # 权重再平衡
                if self.symbol_allocator.should_rebalance():
                    self.symbol_allocator.rebalance()
                
                # 风险预算同步（从 AdaptiveController 拉取最新预算分配）
                self._sync_risk_budget_allocations()
                
                # 资金池再平衡（每日）—— P0: 改用日期比较替代分钟窗口，避免错过
                if not hasattr(self.symbol_allocator, '_last_pool_rebalance_date'):
                    self.symbol_allocator._last_pool_rebalance_date = None
                today_str = datetime.now().strftime("%Y-%m-%d")
                if self.symbol_allocator._last_pool_rebalance_date != today_str:
                    self.capital_pool.rebalance_pools()
                    self.symbol_allocator._last_pool_rebalance_date = today_str
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in capital rebalance loop: {e}")
                await asyncio.sleep(60)

    def _sync_risk_budget_allocations(self):
        """从 AdaptiveController 同步风险预算分配到资金池"""
        if not self._adaptive_controller:
            return
        
        try:
            risk_status = self._adaptive_controller.get_risk_budget_status()
            strategy_status = risk_status.get("strategy_status", {})
            
            # 将风险预算映射到资金池分配
            for sname, sstatus in strategy_status.items():
                budget_ratio = sstatus.get("budget_ratio", 0)
                if budget_ratio > 0:
                    self._risk_budget_allocation[sname] = budget_ratio
            
            self._last_risk_budget_sync = datetime.now()
        except Exception as e:
            logger.warning(f"Risk budget sync error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 资金磨损分析集成
    # ═══════════════════════════════════════════════════════════════

    def _init_attrition_budgets(self) -> None:
        """从配置初始化各策略的磨损预算"""
        att_cfg = self.config.get("capital_attrition", {})
        strategy_budgets = att_cfg.get("strategy_budgets", {})
        for strategy_name, budget_cfg in strategy_budgets.items():
            self.attrition_analyzer.set_strategy_budget(
                strategy_name=strategy_name,
                daily_budget=budget_cfg.get("daily_budget_usdt"),
                weekly_budget=budget_cfg.get("weekly_budget_usdt"),
                max_attrition_rate=budget_cfg.get("max_attrition_rate"),
            )

    def check_attrition_budget(self, strategy_name: str) -> Tuple[bool, str]:
        """检查策略磨损预算 — 超限时返回(False, reason)"""
        return self.attrition_analyzer.check_budget(strategy_name)

    def record_trade_fee(
        self, symbol: str, strategy_name: str, fee_usdt: float,
        trade_value_usdt: float, side: str = "", is_maker: bool = False
    ) -> None:
        """记录交易手续费到磨损分析器"""
        self.attrition_analyzer.record_fee(
            symbol, strategy_name, fee_usdt, trade_value_usdt, side, is_maker
        )

    def record_trade_slippage(
        self, symbol: str, strategy_name: str, expected_price: float,
        filled_price: float, quantity: float, side: str = ""
    ) -> float:
        """记录滑点损耗到磨损分析器，返回损耗金额"""
        return self.attrition_analyzer.record_slippage(
            symbol, strategy_name, expected_price, filled_price, quantity, side
        )

    def record_funding_payment(
        self, symbol: str, strategy_name: str, payment_usdt: float,
        position_value_usdt: float, funding_rate: float = 0
    ) -> None:
        """记录资金费率支付到磨损分析器"""
        self.attrition_analyzer.record_funding_payment(
            symbol, strategy_name, payment_usdt, position_value_usdt, funding_rate
        )

    def record_strategy_profit(self, strategy_name: str, profit_usdt: float) -> None:
        """记录策略利润（用于磨损率计算）"""
        self.attrition_analyzer.record_profit(strategy_name, profit_usdt)

    def get_attrition_report(self) -> Dict[str, Any]:
        """获取资金磨损综合报告"""
        return self.attrition_analyzer.get_stats()

    def get_full_report(self) -> Dict[str, Any]:
        """获取完整资金管理报告"""
        report = {
            "capital_pool": self.capital_pool.to_dict(),
            "symbol_weights": self.symbol_allocator.to_dict(),
            "leverage_tiers": self.leverage_manager.to_dict(),
            "pnl_reallocation": self.pnl_reallocation.to_dict(),
            "hedge_scheduler": self.hedge_scheduler.to_dict(),
            "capital_attrition": self.attrition_analyzer.get_stats(),
        }
        
        # 附加风险预算状态
        if self._adaptive_controller:
            try:
                report["risk_budget"] = self._adaptive_controller.get_risk_budget_status()
            except Exception:
                report["risk_budget"] = {"error": "unavailable"}
        
        return report

    def shutdown(self) -> None:
        """关闭"""
        self._running = False
        logger.info("CapitalManager shutdown")


_manager_instance: Optional[CapitalManager] = None

def get_capital_manager(config: Dict[str, Any] = None) -> CapitalManager:
    """获取资金管理器单例"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = CapitalManager(config)
    return _manager_instance


__all__ = [
    "CapitalManager",
    "CapitalPoolController",
    "CapitalPool",
    "CapitalPoolType",
    "SymbolWeightAllocator",
    "LeverageTierManager",
    "LeverageTier",
    "LeverageAssignment",
    "PnLReallocationUnit",
    "HedgeScheduler",
    "HedgeType",
    "HedgePosition",
    "get_capital_manager",
]