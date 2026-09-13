"""
资金分配优化Agent：基于策略表现动态分配资金
智能分析策略绩效、风险指标，自动优化资金配置

增强版：集成 DynamicAllocator 动态资金分配引擎
  - 多级资金池管理（底仓/加仓/风控隔离）
  - Kelly 公式最优仓位计算
  - 分配优先级瀑布模型
  - 市场状态感知的动态权重调整
  - 盈亏驱动池间资金流动
"""
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger
import numpy as np

from risk.dynamic_allocator import (
    DynamicAllocator, MarketRegime, AllocationPlan, get_dynamic_allocator,
)


class AllocationAgent:
    """智能资金分配Agent（增强版）"""

    def __init__(self, config: Dict[str, Any], trade_journal, profit_optimizer, account_manager):
        self.config = config
        self.trade_journal = trade_journal
        self.profit_optimizer = profit_optimizer
        self.account_manager = account_manager

        self._enabled = config.get("allocation_agent", {}).get("enabled", True)
        self._rebalance_interval = config.get("allocation_agent", {}).get("rebalance_interval", 3600)  # 每小时检查
        self._min_trade_count = config.get("allocation_agent", {}).get("min_trade_count", 20)  # 最小交易数
        self._max_allocation_change = config.get("allocation_agent", {}).get("max_allocation_change", 0.05)  # 单次最大变更5%
        self._allocation_method = config.get("allocation_agent", {}).get("method", "dynamic")  # equal, performance, risk_adjusted, dynamic

        self._strategy_weights: Dict[str, float] = {}
        self._performance_history: Dict[str, List[Dict[str, float]]] = {}
        self._last_rebalance_time: Optional[datetime] = None
        self._running = False

        self._strategy_names = [
            "grid", "trend", "scalping", "arbitrage", "spot_grid", "spot_martingale"
        ]

        # ── DynamicAllocator 集成 ───────────────────────────
        self._dynamic_allocator: Optional[DynamicAllocator] = None
        self._last_allocation_plan: Optional[AllocationPlan] = None
        self._market_regime: MarketRegime = MarketRegime.UNKNOWN
        self._portfolio_optimizer = None

    async def start(self):
        if not self._enabled:
            logger.info("Allocation Agent is disabled")
            return

        logger.info("Starting Allocation Agent")
        self._running = True
        self._last_rebalance_time = datetime.now()
        await self._load_initial_weights()

        asyncio.create_task(self._rebalance_loop())
        asyncio.create_task(self._performance_monitor_loop())

    async def stop(self):
        self._running = False
        logger.info("Allocation Agent stopped")

    async def _load_initial_weights(self):
        """从配置加载初始权重"""
        trading_cfg = self.config.get("trading", {})
        for strategy in self._strategy_names:
            key = f"{strategy}_allocation"
            self._strategy_weights[strategy] = trading_cfg.get(key, 0.167)

        self._normalize_weights()
        logger.info(f"Initial allocation weights: {self._strategy_weights}")

    def _enabled_strategy_names(self) -> List[str]:
        """返回 config 中 enabled=true 的策略名（未显式声明的默认启用）。"""
        strategies_cfg = self.config.get("strategies", {})
        return [
            name for name in self._strategy_names
            if strategies_cfg.get(name, {}).get("enabled", True)
        ]

    def _normalize_weights(self):
        """归一化权重：禁用策略强制归零，启用策略权重总和归一化为1"""
        enabled_names = set(self._enabled_strategy_names())
        for name in self._strategy_weights:
            if name not in enabled_names:
                self._strategy_weights[name] = 0.0
        total = sum(self._strategy_weights.get(n, 0.0) for n in enabled_names)
        if total > 0:
            for name in enabled_names:
                if name in self._strategy_weights:
                    self._strategy_weights[name] /= total

    def set_portfolio_optimizer(self, optimizer) -> None:
        """注入 PortfolioOptimizer，接收 MPT 最优权重推荐"""
        self._portfolio_optimizer = optimizer
        logger.info("AllocationAgent: PortfolioOptimizer injected")

    def set_dynamic_allocator(self, allocator: DynamicAllocator) -> None:
        """注入 DynamicAllocator，使用增强版资金分配引擎"""
        self._dynamic_allocator = allocator
        if self._portfolio_optimizer:
            allocator.set_portfolio_optimizer(self._portfolio_optimizer)
        logger.info("AllocationAgent: DynamicAllocator injected")

    def set_market_regime(self, regime_str: str) -> None:
        """设置当前市场状态（用于DynamicAllocator）"""
        try:
            self._market_regime = MarketRegime(regime_str)
        except ValueError:
            self._market_regime = MarketRegime.UNKNOWN

    def get_optimized_weights(self) -> Dict[str, float]:
        """从 PortfolioOptimizer 获取 MPT 最优权重作为再平衡目标"""
        if not hasattr(self, '_portfolio_optimizer') or not self._portfolio_optimizer:
            return dict(self._strategy_weights)

        result = self._portfolio_optimizer.optimize()
        if result and result.optimal_weights:
            # 合并优化结果：使用优化权重，未覆盖的策略保持原权重
            merged = dict(self._strategy_weights)
            merged.update(result.optimal_weights)
            return merged

        return dict(self._strategy_weights)

    async def _rebalance_loop(self):
        """定期重新平衡资金分配"""
        while self._running:
            try:
                await asyncio.sleep(self._rebalance_interval)
                await self._rebalance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rebalance loop error: {e}")

    async def _performance_monitor_loop(self):
        """监控策略表现，每10分钟记录一次"""
        while self._running:
            try:
                await asyncio.sleep(600)
                await self._record_performance()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitor loop error: {e}")

    async def _record_performance(self):
        """记录当前策略表现"""
        now = datetime.now()
        for strategy in self._strategy_names:
            metrics = self._get_strategy_metrics(strategy)
            if strategy not in self._performance_history:
                self._performance_history[strategy] = []

            self._performance_history[strategy].append({
                "timestamp": now.isoformat(),
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "sharpe_ratio": metrics["sharpe_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "total_pnl": metrics["total_pnl"],
                "trade_count": metrics["trade_count"]
            })

            # 保留最近100条记录
            if len(self._performance_history[strategy]) > 100:
                self._performance_history[strategy] = self._performance_history[strategy][-100:]

    async def _rebalance(self):
        """执行资金重新分配"""
        logger.info("Starting capital rebalance...")

        new_weights = await self._calculate_optimal_allocation()
        changes = await self._apply_allocation_changes(new_weights)

        if changes:
            logger.info(f"Capital rebalance completed: {changes}")
            await self._save_allocation_state()
        else:
            logger.info("No allocation changes needed")

        self._last_rebalance_time = datetime.now()

    async def _calculate_optimal_allocation(self) -> Dict[str, float]:
        """计算最优资金分配"""
        if self._allocation_method == "equal":
            return self._allocate_equal()
        elif self._allocation_method == "performance":
            return await self._allocate_by_performance()
        elif self._allocation_method == "risk_adjusted":
            return await self._allocate_by_risk_adjusted()
        elif self._allocation_method == "dynamic":
            return await self._allocate_dynamic()
        else:
            return await self._allocate_dynamic()

    def _allocate_equal(self) -> Dict[str, float]:
        """等权分配"""
        num_strategies = len(self._strategy_names)
        return {s: 1.0 / num_strategies for s in self._strategy_names}

    async def _allocate_by_performance(self) -> Dict[str, float]:
        """基于表现分配：盈利越多，分配越多"""
        scores = {}
        for strategy in self._strategy_names:
            metrics = self._get_strategy_metrics(strategy)
            # 评分 = 胜率 * 利润因子 * 夏普比率
            score = metrics["win_rate"] * metrics["profit_factor"] * max(0, metrics["sharpe_ratio"])
            scores[strategy] = max(0.01, score)

        total = sum(scores.values())
        if total > 0:
            return {s: scores[s] / total for s in self._strategy_names}
        return self._allocate_equal()

    async def _allocate_by_risk_adjusted(self) -> Dict[str, float]:
        """风险调整分配：兼顾收益和风险"""
        scores = {}
        for strategy in self._strategy_names:
            metrics = self._get_strategy_metrics(strategy)

            # 风险调整收益 = 收益 / 风险
            # 收益指标：胜率 * 利润因子
            # 风险指标：最大回撤 + (1 - 夏普比率)
            reward = metrics["win_rate"] * metrics["profit_factor"]
            risk = metrics["max_drawdown"] + max(0, 1 - metrics["sharpe_ratio"])

            if risk > 0:
                score = reward / risk
            else:
                score = reward * 10  # 低风险时给予高评分

            # 如果交易数不足，降低权重
            if metrics["trade_count"] < self._min_trade_count:
                confidence = min(1.0, metrics["trade_count"] / self._min_trade_count)
                score *= confidence

            scores[strategy] = max(0.001, score)

        total = sum(scores.values())
        if total > 0:
            weights = {s: scores[s] / total for s in self._strategy_names}
            return weights
        return self._allocate_equal()

    async def _allocate_dynamic(self) -> Dict[str, float]:
        """
        动态分配算法（增强版）

        使用 DynamicAllocator 执行以下步骤：
          1. 三级资金池划分（底仓/加仓/风控）
          2. Kelly 公式计算最优仓位
          3. 策略优先级评估（综合Sharpe/胜率/盈亏比/交易数）
          4. 分配优先级瀑布（HIGH→MEDIUM→LOW→FROZEN）
          5. 市场状态感知权重调整
          6. 闲置资金检测与效率评估
        """
        # 确保 DynamicAllocator 已初始化
        if self._dynamic_allocator is None:
            logger.warning("DynamicAllocator not available, falling back to risk_adjusted")
            return await self._allocate_by_risk_adjusted()

        # 收集所有策略的绩效指标
        strategy_metrics = {}
        for name in self._strategy_names:
            m = self._get_strategy_metrics(name)
            # 追加连续盈亏信息
            m["consecutive_wins"] = self._get_consecutive_count(name, "win")
            m["consecutive_losses"] = self._get_consecutive_count(name, "loss")
            m["volatility_30d"] = self._compute_30d_volatility(name)
            strategy_metrics[name] = m

        # 收集各策略已用保证金（修复资金效率数据断链）
        used_margin_by_strategy = {}
        if self.account_manager is not None and hasattr(self.account_manager, "get_strategy_margin"):
            for name in self._strategy_names:
                try:
                    used_margin_by_strategy[name] = self.account_manager.get_strategy_margin(name)
                except Exception:
                    used_margin_by_strategy[name] = 0.0

        # 获取总权益
        try:
            total_equity = self.account_manager.get_total_equity()
            total_capital = self.account_manager.get_total_capital()
        except Exception:
            total_equity = self.config.get("trading", {}).get("total_capital", 5000)
            total_capital = total_equity

        if total_equity <= 0:
            return self._allocate_equal()

        # 调用 DynamicAllocator 计算分配方案
        try:
            plan = await self._dynamic_allocator.compute_allocation_plan(
                total_capital=total_capital,
                total_equity=total_equity,
                strategy_names=self._enabled_strategy_names(),
                strategy_metrics=strategy_metrics,
                market_regime=self._market_regime,
                current_weights=dict(self._strategy_weights),
                used_margin_by_strategy=used_margin_by_strategy,
            )
            self._last_allocation_plan = plan

            # 更新 DynamicAllocator 绩效缓存
            for name in self._strategy_names:
                m = strategy_metrics[name]
                self._dynamic_allocator.update_strategy_pnl(name, m.get("total_pnl", 0))

            # 从分配方案提取权重
            new_weights = {}
            for name, alloc in plan.strategy_allocations.items():
                new_weights[name] = alloc.target_weight

            # 日志输出关键信息
            logger.info(
                f"DynamicAllocator plan: efficiency={plan.capital_efficiency:.1%}, "
                f"idle={plan.idle_cash:.0f}, frozen={len([a for a in plan.strategy_allocations.values() if a.is_frozen])}"
            )
            if plan.warnings:
                logger.warning(f"Allocation warnings: {plan.warnings}")
            if plan.recommendations:
                logger.info(f"Allocation recommendations: {plan.recommendations}")

            return new_weights

        except Exception as e:
            logger.error(f"DynamicAllocator failed: {e}, falling back to risk_adjusted")
            return await self._allocate_by_risk_adjusted()

    def _get_consecutive_count(self, strategy: str, streak_type: str) -> int:
        """获取连续盈利/亏损天数"""
        if strategy not in self._performance_history:
            return 0

        history = self._performance_history[strategy]
        count = 0
        for entry in reversed(history):
            pnl = entry.get("total_pnl", 0)
            if streak_type == "win" and pnl > 0:
                count += 1
            elif streak_type == "loss" and pnl < 0:
                count += 1
            else:
                break
        return count

    def _compute_30d_volatility(self, strategy: str) -> float:
        """计算策略30日年化波动率"""
        if strategy not in self._performance_history:
            return 0.02  # 默认2%

        history = self._performance_history[strategy][-30:]
        if len(history) < 5:
            return 0.02

        pnl_values = [h.get("total_pnl", 0) for h in history]
        std = float(np.std(pnl_values))
        if std > 0:
            return float(std * np.sqrt(365))  # 日波动率年化
        return 0.02

    def _get_strategy_metrics(self, strategy_name: str) -> Dict[str, float]:
        """获取策略性能指标（企业级：USDT 净额口径 + 已实现/未实现拆账）"""
        try:
            trades = self.trade_journal.get_trades_by_strategy(strategy_name)
        except Exception:
            trades = []

        # 未实现盈亏（真实数据源：AccountManager 逐策略浮动盈亏）
        unrealized_pnl = 0.0
        if self.account_manager is not None and hasattr(self.account_manager, "get_strategy_equity"):
            try:
                unrealized_pnl = self.account_manager.get_strategy_equity(strategy_name)
            except Exception:
                unrealized_pnl = 0.0

        base_metrics = {
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "sharpe_ratio": 0.5,
            "max_drawdown": 0.2,
            "total_pnl": 0.0,
            "trade_count": 0,
            "realized_pnl": 0.0,
            "unrealized_pnl": unrealized_pnl,
        }

        if not trades:
            return base_metrics

        # 企业级修复：统一 USDT 净额口径（t.pnl_usdt），废弃百分比 t.pnl
        pnl_values = [t.pnl_usdt for t in trades if t.pnl_usdt is not None]
        if not pnl_values:
            base_metrics["trade_count"] = len(trades)
            return base_metrics

        wins = [p for p in pnl_values if p > 0]
        losses = [abs(p) for p in pnl_values if p < 0]

        win_rate = len(wins) / len(pnl_values) if pnl_values else 0.5
        avg_win = sum(wins) / len(wins) if wins else 1.0
        avg_loss = sum(losses) / len(losses) if losses else 1.0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0

        realized_pnl = sum(pnl_values)
        trade_count = len(trades)

        # 计算夏普比率（简化版：假设无风险利率为0）
        if len(pnl_values) >= 2:
            returns = np.array(pnl_values)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = mean_return / std_return if std_return > 0 else 0.0
        else:
            sharpe_ratio = 0.5

        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(pnl_values)

        return {
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_pnl": realized_pnl,
            "trade_count": trade_count,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
        }

    def _calculate_max_drawdown(self, pnl_values: List[float]) -> float:
        """计算最大回撤"""
        if not pnl_values:
            return 0.0

        cumulative = np.cumsum(pnl_values)
        peak = cumulative[0]
        max_dd = 0.0

        for val in cumulative:
            if val > peak:
                peak = val
            dd = (peak - val) / peak if peak != 0 else 0
            if dd > max_dd:
                max_dd = dd

        return max_dd

    async def _apply_allocation_changes(self, new_weights: Dict[str, float]) -> Dict[str, float]:
        """应用资金分配变更（带限制）。

        单一写入口（P1-④）：归一化后的最终权重经 _publish_allocations 统一写入
        AccountManager，并回写 config，确保 AccountManager / AdaptivePositionSizer /
        AllocationAgent 三处读取同一数据源，避免各自快照导致分配值分歧。
        """
        changes = {}

        # 禁用策略强制归零（config 未启用的策略不应占用资金）
        enabled_names = set(self._enabled_strategy_names())
        for strategy in self._strategy_names:
            if strategy not in enabled_names:
                current = self._strategy_weights.get(strategy, 0.0)
                if abs(current) > 0.001:
                    self._strategy_weights[strategy] = 0.0
                    changes[strategy] = -current
                new_weights.pop(strategy, None)

        for strategy, new_weight in new_weights.items():
            current_weight = self._strategy_weights.get(strategy, 0)
            diff = new_weight - current_weight

            # 限制单次变更幅度
            max_change = self._max_allocation_change
            if abs(diff) > max_change:
                diff = max_change if diff > 0 else -max_change
                new_weight = current_weight + diff

            if abs(diff) > 0.001:
                self._strategy_weights[strategy] = new_weight
                changes[strategy] = diff

        if changes:
            self._normalize_weights()
            self._publish_allocations(dict(self._strategy_weights))

        return changes

    def _publish_allocations(self, weights: Dict[str, float]) -> None:
        """将归一化后的最终权重发布到单一权威源（AccountManager），回写 config 兜底。"""
        if self.account_manager is not None and hasattr(self.account_manager, "set_strategy_allocations"):
            try:
                self.account_manager.set_strategy_allocations(weights)
                return
            except Exception as e:
                logger.error(f"Failed to publish allocations via AccountManager: {e}")
        # 兜底：直接回写 config（保持原行为）
        trading_cfg = self.config.get("trading", {})
        for strategy, weight in weights.items():
            trading_cfg[f"{strategy}_allocation"] = round(weight, 6)

    async def _save_allocation_state(self):
        """保存分配状态到文件（原子写入，防止崩溃损坏）"""
        state = {
            "last_rebalance_time": datetime.now().isoformat(),
            "strategy_weights": self._strategy_weights,
            "allocation_method": self._allocation_method,
            "performance_history": self._performance_history
        }

        try:
            import os
            from core.atomic_writer import atomic_write_json
            data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
            os.makedirs(data_dir, exist_ok=True)
            path = os.path.join(data_dir, "allocation_state.json")

            if atomic_write_json(path, state):
                logger.debug(f"Allocation state saved to {path}")
            else:
                logger.error(f"Failed to save allocation state atomically")
        except Exception as e:
            logger.error(f"Failed to save allocation state: {e}")

    def get_current_allocation(self) -> Dict[str, float]:
        """获取当前资金分配"""
        return dict(self._strategy_weights)

    def get_allocation_report(self) -> Dict[str, Any]:
        """生成资金分配报告（增强版：含 DynamicAllocator 数据）"""
        report = {
            "last_rebalance_time": self._last_rebalance_time.isoformat() if self._last_rebalance_time else None,
            "allocation_method": self._allocation_method,
            "current_weights": self._strategy_weights,
            "strategy_metrics": {}
        }

        for strategy in self._strategy_names:
            report["strategy_metrics"][strategy] = self._get_strategy_metrics(strategy)

        # DynamicAllocator 数据
        if self._last_allocation_plan:
            plan_dict = self._last_allocation_plan.to_dict()
            report["dynamic_allocation"] = {
                "pools": plan_dict.get("pools", {}),
                "capital_efficiency": plan_dict.get("capital_efficiency", 0),
                "idle_cash": plan_dict.get("idle_cash", 0),
                "concentration_ratio": plan_dict.get("concentration_ratio", 0),
                "warnings": plan_dict.get("warnings", []),
                "recommendations": plan_dict.get("recommendations", []),
                "pool_flows": plan_dict.get("pool_flows", {}),
                "strategy_allocations": plan_dict.get("strategy_allocations", {}),
            }
            report["market_regime"] = self._market_regime.value

        return report

    async def get_recommendations(self) -> Dict[str, Any]:
        """获取资金分配建议"""
        optimal = await self._calculate_optimal_allocation()
        recommendations = {}

        for strategy, optimal_weight in optimal.items():
            current_weight = self._strategy_weights.get(strategy, 0)
            diff = optimal_weight - current_weight

            recommendations[strategy] = {
                "current": round(current_weight, 4),
                "optimal": round(optimal_weight, 4),
                "change": round(diff, 4),
                "action": "increase" if diff > 0.01 else "decrease" if diff < -0.01 else "hold"
            }

        return recommendations

    async def manual_rebalance(self):
        """手动触发重新平衡"""
        logger.info("Manual rebalance requested")
        await self._rebalance()