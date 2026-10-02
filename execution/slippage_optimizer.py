"""
滑点优化执行器
激进行情动态调整限价偏移，平衡成交率与滑点损耗

核心功能：
1. 实时波动率检测 - 基于K线振幅、订单簿深度、成交异动判断行情激进程度
2. 动态限价偏移 - 根据激进程度自动调整限价单偏移量
3. 滑点预测模型 - 基于历史数据预测当前行情下的滑点
4. 成交率优化 - 在保证成交率的前提下最小化滑点损耗
5. 自适应学习 - 根据实际成交反馈持续优化参数
"""
import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from core.direction_unifier import DirectionUnifier


class MarketRegime(Enum):
    """市场行情状态"""
    CALM = "calm"              # 平静行情
    NORMAL = "normal"          # 正常行情
    ACTIVE = "active"          # 活跃行情
    VOLATILE = "volatile"      # 剧烈波动
    EXTREME = "extreme"        # 极端行情


class SlippageTolerance(Enum):
    """滑点容忍策略"""
    CONSERVATIVE = "conservative"    # 保守：低滑点优先，成交率可能低
    BALANCED = "balanced"            # 平衡：成交率与滑点均衡
    AGGRESSIVE = "aggressive"        # 激进：成交率优先，滑点容忍度高


@dataclass
class SlippageRecord:
    """滑点记录"""
    symbol: str
    side: str
    order_type: str
    expected_price: float
    filled_price: float
    quantity: float
    raw_slippage: float            # 原始滑点比例
    abs_slippage: float            # 绝对滑点比例
    market_regime: str             # 当时的市场状态
    offset_applied: float          # 应用的偏移比例
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketState:
    """市场状态快照"""
    symbol: str
    last_price: float = 0.0
    volatility_1m: float = 0.0     # 1分钟波动率
    volatility_5m: float = 0.0     # 5分钟波动率
    volatility_15m: float = 0.0    # 15分钟波动率
    volume_ratio: float = 1.0      # 成交量倍率（相对平均）
    spread_ratio: float = 0.0      # 买卖价差比例
    depth_imbalance: float = 0.0   # 订单簿深度失衡（-1到1，正=买盘强）
    price_change_5m: float = 0.0   # 5分钟涨跌幅
    regime: MarketRegime = MarketRegime.NORMAL
    last_update: float = 0.0


class SlippageOptimizer:
    """
    滑点优化执行器
    
    核心设计：
    - 根据市场波动动态调整限价单偏移
    - 平静行情：小偏移，控制滑点
    - 激进行情：大偏移，保证成交
    - 基于历史成交数据持续学习优化
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        slippage_config = config.get("execution", {}).get("slippage_optimizer") or {}
        if not isinstance(slippage_config, dict):
            slippage_config = {}
        self._enabled = slippage_config.get("enabled", True)
        self._default_strategy = SlippageTolerance(slippage_config.get("default_strategy", "balanced"))
        
        # 基础偏移配置（相对价格的比例）
        self._base_offset = slippage_config.get("base_offset", 0.0005)     # 基础偏移 0.05%
        self._max_offset = slippage_config.get("max_offset", 0.005)         # 最大偏移 0.5%
        self._min_offset = slippage_config.get("min_offset", 0.0001)        # 最小偏移 0.01%
        
        # 波动率阈值
        self._volatility_thresholds = slippage_config.get("volatility_thresholds", {
            "calm": 0.001,      # < 0.1% 平静
            "normal": 0.003,    # < 0.3% 正常
            "active": 0.008,    # < 0.8% 活跃
            "volatile": 0.02,   # < 2% 剧烈
        })
        
        # 偏移倍数（对应不同行情状态）
        self._offset_multipliers = slippage_config.get("offset_multipliers", {
            "calm": 0.5,        # 平静：0.5倍基础偏移
            "normal": 1.0,      # 正常：1倍
            "active": 2.0,      # 活跃：2倍
            "volatile": 4.0,    # 剧烈：4倍
            "extreme": 8.0,     # 极端：8倍
        })
        
        # 每币种市场状态
        self._market_states: Dict[str, MarketState] = {}
        
        # 滑点历史记录（内存中保留最近1000条）
        self._slippage_history: deque = deque(maxlen=1000)
        
        # 按币种聚合的滑点统计
        self._symbol_stats: Dict[str, Dict[str, Any]] = {}
        
        # 自适应学习参数
        self._learning_enabled = slippage_config.get("learning_enabled", True)
        self._learning_rate = slippage_config.get("learning_rate", 0.1)
        
        # 成交率目标
        self._target_fill_rate = slippage_config.get("target_fill_rate", 0.95)
        
        # 价格精度缓存
        self._price_precision_cache: Dict[str, int] = {}
        
        logger.info(f"SlippageOptimizer initialized: strategy={self._default_strategy.value}, "
                   f"base_offset={self._base_offset}, max_offset={self._max_offset}")

    def update_market_data(self, symbol: str, price: float, 
                          volatility: float = None, volume_ratio: float = None,
                          spread: float = None, depth_imbalance: float = None):
        """
        更新市场数据，用于滑点优化决策
        
        Args:
            symbol: 交易对
            price: 当前价格
            volatility: 波动率（可选）
            volume_ratio: 成交量倍率（可选）
            spread: 买卖价差（可选）
            depth_imbalance: 深度失衡（可选）
        """
        if symbol not in self._market_states:
            self._market_states[symbol] = MarketState(symbol=symbol)
        
        state = self._market_states[symbol]
        state.last_price = price
        state.last_update = time.time()
        
        if volatility is not None:
            state.volatility_5m = volatility
        
        if volume_ratio is not None:
            state.volume_ratio = volume_ratio
        
        if spread is not None and price > 0:
            state.spread_ratio = spread / price
        
        if depth_imbalance is not None:
            state.depth_imbalance = depth_imbalance
        
        # 重新计算市场状态
        self._update_market_regime(symbol)

    def _update_market_regime(self, symbol: str):
        """根据市场数据更新行情状态"""
        state = self._market_states.get(symbol)
        if not state or state.volatility_5m == 0:
            return
        
        vol = state.volatility_5m
        thresholds = self._volatility_thresholds
        
        if vol < thresholds["calm"]:
            state.regime = MarketRegime.CALM
        elif vol < thresholds["normal"]:
            state.regime = MarketRegime.NORMAL
        elif vol < thresholds["active"]:
            state.regime = MarketRegime.ACTIVE
        elif vol < thresholds["volatile"]:
            state.regime = MarketRegime.VOLATILE
        else:
            state.regime = MarketRegime.EXTREME

    def calculate_optimal_price(self, symbol: str, side: str, 
                                reference_price: float, order_type: str = "limit",
                                strategy: SlippageTolerance = None,
                                urgency: float = 0.5) -> Tuple[float, Dict[str, Any]]:
        """
        计算最优下单价格
        
        Args:
            symbol: 交易对
            side: buy/sell
            reference_price: 参考价格（通常是当前市价）
            order_type: limit/market
            strategy: 滑点容忍策略
            urgency: 紧急程度 0-1，1=最紧急
        
        Returns:
            (optimal_price, info_dict) - 最优价格和详细信息
        """
        if not self._enabled or order_type == "market":
            return reference_price, {"applied": False, "reason": "disabled_or_market"}

        # 方向归一化：long/short/buy/sell → buy/sell（避免 "long" 被误当 sell 处理）
        try:
            side = DirectionUnifier.to_side(side)
        except (ValueError, TypeError):
            pass

        strategy = strategy or self._default_strategy
        state = self._market_states.get(symbol)
        
        if not state or state.last_price <= 0:
            # 没有市场数据时使用基础偏移
            offset = self._base_offset
            regime = "unknown"
        else:
            regime = state.regime.value
            base_multiplier = self._offset_multipliers.get(regime, 1.0)
            
            # 根据策略调整
            strategy_multiplier = {
                "conservative": 0.6,
                "balanced": 1.0,
                "aggressive": 1.8,
            }.get(strategy.value, 1.0)
            
            # 根据紧急程度调整
            urgency_multiplier = 0.5 + urgency * 1.5  # 0.5x - 2.0x
            
            # 自适应调整：基于历史滑点
            adaptive_multiplier = self._get_adaptive_multiplier(symbol, side)
            
            total_multiplier = base_multiplier * strategy_multiplier * urgency_multiplier * adaptive_multiplier
            offset = self._base_offset * total_multiplier

            # spread-aware floor: offset must be >= half the current spread so the limit
            # price lands on the executable side rather than trapped inside the spread
            spread_floor = state.spread_ratio * 0.5 if state.spread_ratio > 0 else 0.0
            offset = max(offset, spread_floor)

            # 限制在合理范围
            offset = max(self._min_offset, min(self._max_offset, offset))
        
        # 根据方向计算价格
        if side == "buy":
            optimal_price = reference_price * (1 + offset)
        else:
            optimal_price = reference_price * (1 - offset)
        
        # 应用价格精度
        optimal_price = self._apply_price_precision(symbol, optimal_price)
        
        info = {
            "applied": True,
            "regime": regime,
            "strategy": strategy.value,
            "urgency": urgency,
            "offset_ratio": offset,
            "offset_value": abs(optimal_price - reference_price),
            "reference_price": reference_price,
            "optimal_price": optimal_price,
            "side": side,
            "spread_ratio": state.spread_ratio if state else 0.0,
        }
        
        return optimal_price, info

    def _get_adaptive_multiplier(self, symbol: str, side: str) -> float:
        """获取自适应倍数（基于历史滑点）"""
        if not self._learning_enabled:
            return 1.0
        
        stats = self._symbol_stats.get(symbol, {})
        if not stats:
            return 1.0
        
        # 取该方向的平均滑点
        side_key = f"avg_slippage_{side}"
        avg_slippage = stats.get(side_key, 0)
        
        if avg_slippage <= 0 or self._base_offset <= 0:
            return 1.0
        
        # 如果平均滑点 > 基础偏移，说明需要更大偏移才能成交
        ratio = avg_slippage / self._base_offset
        
        # 平滑处理，避免剧烈波动
        smoothed = 1.0 + (ratio - 1.0) * self._learning_rate
        
        return max(0.5, min(3.0, smoothed))

    def _apply_price_precision(self, symbol: str, price: float) -> float:
        """应用价格精度"""
        if symbol not in self._price_precision_cache:
            # 默认精度，后续可从合约信息中获取
            self._price_precision_cache[symbol] = 5
        
        precision = self._price_precision_cache[symbol]
        return round(price, precision)

    def record_fill_slippage(self, symbol: str, side: str, order_type: str,
                            expected_price: float, filled_price: float, quantity: float,
                            offset_applied: float = 0.0):
        """
        记录实际成交滑点，用于自适应学习
        
        Args:
            symbol: 交易对
            side: buy/sell
            order_type: limit/market
            expected_price: 期望价格（下单时的价格）
            filled_price: 实际成交价
            quantity: 成交数量
            offset_applied: 应用的偏移比例
        """
        if expected_price <= 0 or filled_price <= 0:
            return

        # 方向归一化：long/short/buy/sell → buy/sell
        try:
            side = DirectionUnifier.to_side(side)
        except (ValueError, TypeError):
            pass

        raw_slippage = (filled_price - expected_price) / expected_price
        abs_slippage = abs(raw_slippage)
        
        state = self._market_states.get(symbol)
        regime = state.regime.value if state else "unknown"
        
        record = SlippageRecord(
            symbol=symbol,
            side=side,
            order_type=order_type,
            expected_price=expected_price,
            filled_price=filled_price,
            quantity=quantity,
            raw_slippage=raw_slippage,
            abs_slippage=abs_slippage,
            market_regime=regime,
            offset_applied=offset_applied,
        )
        
        self._slippage_history.append(record)
        self._update_symbol_stats(record)
        
        if abs_slippage > self._max_offset:
            logger.warning(f"Large slippage detected: {symbol} {side} "
                          f"slippage={abs_slippage:.4%} regime={regime}")

    def _update_symbol_stats(self, record: SlippageRecord):
        """更新币种滑点统计"""
        symbol = record.symbol
        if symbol not in self._symbol_stats:
            self._symbol_stats[symbol] = {
                "total_fills": 0,
                "avg_slippage_buy": 0.0,
                "avg_slippage_sell": 0.0,
                "max_slippage": 0.0,
                "p95_slippage": 0.0,
                "buy_count": 0,
                "sell_count": 0,
                "by_regime": {},
            }
        
        stats = self._symbol_stats[symbol]
        stats["total_fills"] += 1
        stats["max_slippage"] = max(stats["max_slippage"], record.abs_slippage)
        
        side_key = "buy" if record.side == "buy" else "sell"
        count_key = f"{side_key}_count"
        avg_key = f"avg_slippage_{side_key}"
        
        stats[count_key] = stats.get(count_key, 0) + 1
        # 移动平均
        old_avg = stats.get(avg_key, 0)
        new_avg = old_avg + (record.abs_slippage - old_avg) / stats[count_key]
        stats[avg_key] = new_avg
        
        # 按行情状态统计
        regime = record.market_regime
        if regime not in stats["by_regime"]:
            stats["by_regime"][regime] = {"count": 0, "avg_slippage": 0.0}
        regime_stats = stats["by_regime"][regime]
        regime_stats["count"] += 1
        regime_stats["avg_slippage"] = regime_stats["avg_slippage"] + \
            (record.abs_slippage - regime_stats["avg_slippage"]) / regime_stats["count"]

    def get_symbol_slippage_stats(self, symbol: str) -> Dict[str, Any]:
        """获取指定币种的滑点统计"""
        return self._symbol_stats.get(symbol, {})

    def get_overall_stats(self) -> Dict[str, Any]:
        """获取整体滑点统计"""
        all_slippages = [r.abs_slippage for r in self._slippage_history]
        if not all_slippages:
            return {"total_records": 0}
        
        sorted_slippages = sorted(all_slippages)
        p95_idx = int(len(sorted_slippages) * 0.95)
        
        return {
            "total_records": len(self._slippage_history),
            "avg_slippage": sum(all_slippages) / len(all_slippages),
            "max_slippage": max(all_slippages),
            "min_slippage": min(all_slippages),
            "p95_slippage": sorted_slippages[p95_idx] if p95_idx < len(sorted_slippages) else max(all_slippages),
            "symbols_tracked": len(self._symbol_stats),
            "symbols": list(self._symbol_stats.keys()),
        }

    def get_market_regime(self, symbol: str) -> Optional[MarketRegime]:
        """获取指定币种的行情状态"""
        state = self._market_states.get(symbol)
        return state.regime if state else None

    def get_all_market_regimes(self) -> Dict[str, str]:
        """获取所有币种的行情状态"""
        return {symbol: state.regime.value for symbol, state in self._market_states.items()}
