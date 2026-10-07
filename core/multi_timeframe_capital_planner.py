"""企业级多时间框架资金规划。

三个层级：
1. 短期（1h）：流动性缓冲，确保开仓有足够资金
2. 中期（1d）：目标仓位，基于当日市场状态调整
3. 长期（1w）：资金增长目标，基于周度绩效调整

各层级独立约束并联动。
"""

from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from collections import deque
from loguru import logger


class MultiTimeframeCapitalPlanner:
    """多时间框架资金规划器。

    用法：
        planner = MultiTimeframeCapitalPlanner()
        planner.update_short_term(available_liquidity=30.0, reserved_for_orders=5.0)
        planner.update_mid_term(target_utilization=0.75, current_positions={...})
        planner.update_long_term(weekly_target=1.05, current_equity=100.0)

        plan = planner.get_plan()
        # plan 包含三个层级的目标与建议
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}

        # 短期（1h）：流动性缓冲
        self._short_term = {
            "available_liquidity": 0.0,
            "reserved_for_orders": 0.0,
            "min_buffer": config.get("short_term_min_buffer", 5.0) if config else 5.0,
            "history": deque(maxlen=24),  # 24小时
        }

        # 中期（1d）：目标仓位
        self._mid_term = {
            "target_utilization": 0.75,
            "current_positions": {},
            "target_positions": {},
            "history": deque(maxlen=7),  # 7天
        }

        # 长期（1w）：资金增长目标
        self._long_term = {
            "weekly_target": 1.05,  # 周增长5%
            "current_equity": 0.0,
            "target_equity": 0.0,
            "history": deque(maxlen=52),  # 52周
        }

        logger.info("MultiTimeframeCapitalPlanner initialized")

    def update_short_term(self, available_liquidity: float,
                          reserved_for_orders: float) -> None:
        """更新短期流动性状态"""
        self._short_term["available_liquidity"] = available_liquidity
        self._short_term["reserved_for_orders"] = reserved_for_orders
        self._short_term["history"].append({
            "timestamp": datetime.now().isoformat(),
            "available": available_liquidity,
            "reserved": reserved_for_orders,
        })

    def update_mid_term(self, target_utilization: float,
                        current_positions: Dict[str, float]) -> None:
        """更新中期目标仓位"""
        self._mid_term["target_utilization"] = target_utilization
        self._mid_term["current_positions"] = current_positions
        self._mid_term["history"].append({
            "timestamp": datetime.now().isoformat(),
            "target_utilization": target_utilization,
            "positions": current_positions,
        })

    def update_long_term(self, weekly_target: float,
                         current_equity: float) -> None:
        """更新长期资金增长目标"""
        self._long_term["weekly_target"] = weekly_target
        self._long_term["current_equity"] = current_equity
        self._long_term["target_equity"] = current_equity * weekly_target
        self._long_term["history"].append({
            "timestamp": datetime.now().isoformat(),
            "weekly_target": weekly_target,
            "current_equity": current_equity,
            "target_equity": current_equity * weekly_target,
        })

    def get_plan(self) -> Dict[str, Any]:
        """获取多时间框架规划"""
        # 短期建议
        short_term_buffer = self._short_term["available_liquidity"]
        min_buffer = self._short_term["min_buffer"]
        short_term_ok = short_term_buffer >= min_buffer

        # 中期建议
        mid_term_target = self._mid_term["target_utilization"]
        mid_term_current = sum(self._mid_term["current_positions"].values())
        mid_term_gap = mid_term_target - mid_term_current

        # 长期建议
        long_term_target = self._long_term["target_equity"]
        long_term_current = self._long_term["current_equity"]
        long_term_gap = long_term_target - long_term_current

        return {
            "short_term": {
                "available_liquidity": round(short_term_buffer, 4),
                "reserved_for_orders": round(self._short_term["reserved_for_orders"], 4),
                "min_buffer_required": round(min_buffer, 4),
                "status": "ok" if short_term_ok else "insufficient",
                "action": "hold" if short_term_ok else "reduce_positions",
            },
            "mid_term": {
                "target_utilization": round(mid_term_target, 4),
                "current_positions": round(mid_term_current, 4),
                "gap": round(mid_term_gap, 4),
                "action": "increase" if mid_term_gap > 0.05 else ("reduce" if mid_term_gap < -0.05 else "hold"),
            },
            "long_term": {
                "weekly_target": round(self._long_term["weekly_target"], 4),
                "current_equity": round(long_term_current, 4),
                "target_equity": round(long_term_target, 4),
                "gap": round(long_term_gap, 4),
                "action": "aggressive" if long_term_gap > 5 else ("conservative" if long_term_gap < -5 else "normal"),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def check_constraints(self) -> Dict[str, bool]:
        """检查各层级约束是否满足"""
        short_term_ok = self._short_term["available_liquidity"] >= self._short_term["min_buffer"]
        mid_term_ok = abs(self._mid_term["target_utilization"] - sum(self._mid_term["current_positions"].values())) < 0.10
        long_term_ok = self._long_term["current_equity"] >= self._long_term["target_equity"] * 0.95

        return {
            "short_term": short_term_ok,
            "mid_term": mid_term_ok,
            "long_term": long_term_ok,
            "all": short_term_ok and mid_term_ok and long_term_ok,
        }
