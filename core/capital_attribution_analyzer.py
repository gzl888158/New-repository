"""企业级资金分配绩效归因分析。

Brinson 式归因：将组合收益分解为三个维度：
1. 资产配置贡献（Asset Allocation）：策略权重选择带来的超额收益
2. 个券选择贡献（Security Selection）：策略内币种选择带来的超额收益
3. 交互贡献（Interaction）：权重与选择的交叉效应

= 使用场景 =
- 回答"哪个策略贡献了多少收益"
- 识别资金分配是否合理（高效策略是否获得更多权重）
- 为下一期权重调整提供数据支撑
"""

import os
import json
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from loguru import logger


@dataclass
class AttributionResult:
    """归因结果"""
    period_start: datetime
    period_end: datetime
    # 组合级
    portfolio_return: float              # 组合总收益
    benchmark_return: float              # 基准收益（等权或自定义）
    excess_return: float                 # 超额收益
    # 归因分解
    allocation_effect: float             # 资产配置贡献
    selection_effect: float              # 个券选择贡献
    interaction_effect: float            # 交互贡献
    # 策略级明细
    strategy_attribution: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # 元数据
    total_pnl: float = 0.0
    total_allocated: float = 0.0


class CapitalAttributionAnalyzer:
    """资金分配绩效归因分析器。

    用法：
        analyzer = CapitalAttributionAnalyzer()
        # 记录每期数据
        analyzer.record_period(
            period_key="2026-10-07",
            strategy_weights={"grid": 0.4, "trend": 0.3, "scalping": 0.3},
            strategy_returns={"grid": 0.02, "trend": 0.05, "scalping": -0.01},
            strategy_allocated={"grid": 20.0, "trend": 15.0, "scalping": 15.0},
            strategy_pnl={"grid": 0.4, "trend": 0.75, "scalping": -0.15},
        )
        # 执行归因
        result = analyzer.compute_attribution(
            period_start=datetime(2026, 10, 1),
            period_end=datetime(2026, 10, 7),
        )
    """

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        self._period_data: Dict[str, Dict[str, Any]] = {}
        self._history: deque = deque(maxlen=365)  # 保留365天数据

        # 持久化路径
        self._state_path = os.path.join("data", "attribution_state.json")

        logger.info("CapitalAttributionAnalyzer initialized")

    def record_period(self, period_key: str,
                      strategy_weights: Dict[str, float],
                      strategy_returns: Dict[str, float],
                      strategy_allocated: Dict[str, float],
                      strategy_pnl: Dict[str, float]) -> None:
        """记录单期数据。

        Args:
            period_key: 期间标识（如 "2026-10-07"）
            strategy_weights: 各策略权重（归一化为1）
            strategy_returns: 各策略收益率（PnL / allocated）
            strategy_allocated: 各策略分配资金
            strategy_pnl: 各策略盈亏
        """
        # 归一化权重
        total_weight = sum(strategy_weights.values())
        if total_weight > 0:
            norm_weights = {s: w / total_weight for s, w in strategy_weights.items()}
        else:
            norm_weights = strategy_weights

        self._period_data[period_key] = {
            "timestamp": datetime.now().isoformat(),
            "weights": norm_weights,
            "returns": strategy_returns,
            "allocated": strategy_allocated,
            "pnl": strategy_pnl,
        }

    def compute_attribution(self, period_start: datetime = None,
                            period_end: datetime = None,
                            benchmark_mode: str = "equal_weight") -> AttributionResult:
        """计算 Brinson 式归因。

        Args:
            period_start: 起始日期（None=全部历史）
            period_end: 截止日期（None=至今）
            benchmark_mode: 基准模式
                - "equal_weight": 等权基准（各策略等权）
                - "uniform": 统一收益率基准（0%）
                - "custom": 自定义（需提供 benchmark_returns）

        Returns:
            AttributionResult
        """
        # 筛选期间数据
        periods = self._filter_periods(period_start, period_end)
        if not periods:
            return self._empty_result(period_start, period_end)

        # 聚合多期数据
        agg_weights = defaultdict(float)
        agg_returns = defaultdict(float)
        agg_allocated = defaultdict(float)
        agg_pnl = defaultdict(float)
        count = 0

        for period_key, data in periods.items():
            count += 1
            for sname in data["weights"]:
                agg_weights[sname] += data["weights"].get(sname, 0)
                agg_returns[sname] += data["returns"].get(sname, 0)
                agg_allocated[sname] += data["allocated"].get(sname, 0)
                agg_pnl[sname] += data["pnl"].get(sname, 0)

        strategies = list(agg_weights.keys())
        if not strategies:
            return self._empty_result(period_start, period_end)

        # 平均权重和收益率
        avg_weights = {s: agg_weights[s] / count for s in strategies}
        avg_returns = {s: agg_returns[s] / count if count > 0 else 0 for s in strategies}

        # 计算基准收益率
        if benchmark_mode == "equal_weight":
            n_strategies = len(strategies)
            benchmark_weight = 1.0 / n_strategies if n_strategies > 0 else 0
            benchmark_return = sum(avg_returns.values()) * benchmark_weight
        elif benchmark_mode == "uniform":
            benchmark_return = 0.0
        else:
            benchmark_return = 0.0

        # 组合收益率（加权平均）
        portfolio_return = sum(avg_weights[s] * avg_returns[s] for s in strategies)
        excess_return = portfolio_return - benchmark_return

        # Brinson 归因分解
        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0
        strategy_attribution = {}

        for s in strategies:
            w_p = avg_weights[s]  # 实际权重
            w_b = 1.0 / len(strategies) if benchmark_mode == "equal_weight" else w_p  # 基准权重
            r_s = avg_returns[s]  # 策略收益
            r_b = benchmark_return  # 基准收益

            # 资产配置效应：(w_p - w_b) * (r_b)
            alloc_eff = (w_p - w_b) * r_b
            # 个券选择效应：w_b * (r_s - r_b)
            select_eff = w_b * (r_s - r_b)
            # 交互效应：(w_p - w_b) * (r_s - r_b)
            interact_eff = (w_p - w_b) * (r_s - r_b)

            allocation_effect += alloc_eff
            selection_effect += select_eff
            interaction_effect += interact_eff

            strategy_attribution[s] = {
                "weight": round(w_p, 4),
                "return": round(r_s, 4),
                "allocation_effect": round(alloc_eff, 4),
                "selection_effect": round(select_eff, 4),
                "interaction_effect": round(interact_eff, 4),
                "total_effect": round(alloc_eff + select_eff + interact_eff, 4),
                "allocated": round(agg_allocated[s], 4),
                "pnl": round(agg_pnl[s], 4),
            }

        total_allocated = sum(agg_allocated.values())
        total_pnl = sum(agg_pnl.values())

        result = AttributionResult(
            period_start=period_start or datetime.min,
            period_end=period_end or datetime.now(),
            portfolio_return=round(portfolio_return, 4),
            benchmark_return=round(benchmark_return, 4),
            excess_return=round(excess_return, 4),
            allocation_effect=round(allocation_effect, 4),
            selection_effect=round(selection_effect, 4),
            interaction_effect=round(interaction_effect, 4),
            strategy_attribution=strategy_attribution,
            total_pnl=round(total_pnl, 4),
            total_allocated=round(total_allocated, 4),
        )

        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "period_start": period_start.isoformat() if period_start else None,
            "period_end": period_end.isoformat() if period_end else None,
            "result": {
                "portfolio_return": result.portfolio_return,
                "excess_return": result.excess_return,
                "allocation_effect": result.allocation_effect,
                "selection_effect": result.selection_effect,
                "interaction_effect": result.interaction_effect,
            },
        })

        return result

    def _filter_periods(self, period_start: Optional[datetime],
                        period_end: Optional[datetime]) -> Dict[str, Dict[str, Any]]:
        """筛选指定期间的数据"""
        if period_start is None and period_end is None:
            return self._period_data

        filtered = {}
        for period_key, data in self._period_data.items():
            try:
                ts = datetime.fromisoformat(data["timestamp"])
                if period_start and ts < period_start:
                    continue
                if period_end and ts > period_end:
                    continue
                filtered[period_key] = data
            except (ValueError, KeyError):
                continue
        return filtered

    def _empty_result(self, period_start: Optional[datetime],
                      period_end: Optional[datetime]) -> AttributionResult:
        """返回空结果"""
        return AttributionResult(
            period_start=period_start or datetime.min,
            period_end=period_end or datetime.now(),
            portfolio_return=0.0,
            benchmark_return=0.0,
            excess_return=0.0,
            allocation_effect=0.0,
            selection_effect=0.0,
            interaction_effect=0.0,
            total_pnl=0.0,
            total_allocated=0.0,
        )

    def get_top_contributors(self, limit: int = 3) -> List[Tuple[str, float]]:
        """获取贡献最大的策略（按 total_effect 排序）"""
        if not self._history:
            return []

        latest = self._history[-1]
        # 需要重新计算最近一期的归因
        # 简化：直接从 period_data 取最后一期
        if not self._period_data:
            return []

        last_key = list(self._period_data.keys())[-1]
        last_data = self._period_data[last_key]

        contributions = {}
        for sname in last_data["weights"]:
            w = last_data["weights"].get(sname, 0)
            r = last_data["returns"].get(sname, 0)
            contributions[sname] = w * r

        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        return sorted_contrib[:limit]

    def to_dict(self) -> Dict[str, Any]:
        """导出状态"""
        return {
            "period_count": len(self._period_data),
            "history_count": len(self._history),
            "latest_periods": list(self._period_data.keys())[-10:],
        }

    def save_state(self) -> None:
        """持久化状态"""
        try:
            state = {
                "period_data": self._period_data,
                "history": list(self._history),
                "saved_at": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Attribution analyzer save error: {e}")

    def load_state(self) -> None:
        """加载状态"""
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._period_data = state.get("period_data", {})
            history = state.get("history", [])
            self._history = deque(history, maxlen=365)
            logger.info(f"Attribution analyzer loaded: {len(self._period_data)} periods")
        except Exception as e:
            logger.debug(f"Attribution analyzer load error: {e}")
