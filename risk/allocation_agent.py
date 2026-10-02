"""
资金分配监控 Agent（Observer 模式）：采集策略绩效与风险指标，为 Dashboard / RiskBudgetEngine 提供只读视图。

.. deprecated::
    实验性模块，未接入生产交易链路。仅提供只读指标视图，不执行资金再平衡。
    实际资金分配由 AdaptiveController 负责。

注意：本模块 **不执行** 实际资金再平衡。Live 策略权重的写入与执行由 AdaptiveController 独占负责。
AllocationAgent 的职责边界：
  - 采集策略表现（胜率、Sharpe、回撤、连胜连败）
  - 计算 MPT 最优权重建议（仅供查看，不下发）
  - 为 DynamicAllocator 提供分配计划数据
  - 为 Dashboard API 和 RiskBudgetEngine 提供策略指标查询

增强版：集成 DynamicAllocator 动态资金分配引擎（只读数据源）
  - 多级资金池管理（底仓/加仓/风控隔离）
  - Kelly 公式最优仓位计算
  - 分配优先级瀑布模型
  - 市场状态感知的动态权重调整
  - 盈亏驱动池间资金流动
"""
import asyncio
import json
import time
import math
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger
import numpy as np

from risk.dynamic_allocator import (
    DynamicAllocator, MarketRegime, AllocationPlan, get_dynamic_allocator,
)


def _finite(value: Any, default: float = 0.0) -> float:
    """安全数值转换：None/非法字符串/NaN/Inf 统一回退到 default。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


class AllocationAgent:
    """资金分配监控 Agent（Observer 模式）。

    本模块仅采集策略指标并提供只读视图，**不执行**实际资金再平衡。
    Live 策略权重由 AdaptiveController 独占管理。详见模块级 docstring。
    """

    def __init__(self, config: Dict[str, Any], trade_journal, profit_optimizer, account_manager):
        self.config = config
        self.trade_journal = trade_journal
        self.profit_optimizer = profit_optimizer
        self.account_manager = account_manager
        self._learning_memory = None

        self._enabled = config.get("allocation_agent", {}).get("enabled", True)
        self._min_trade_count = config.get("allocation_agent", {}).get("min_trade_count", 20)  # 最小交易数
        self._allocation_method = config.get("allocation_agent", {}).get("method", "dynamic")  # equal, performance, risk_adjusted, dynamic
        legacy_rebalance_requested = bool(
            config.get("allocation_agent", {}).get("rebalance_enabled", False)
        )
        self._rebalance_enabled = False
        if legacy_rebalance_requested:
            logger.warning(
                "Ignoring allocation_agent.rebalance_enabled: AdaptiveController is the sole live allocation authority"
            )

        self._strategy_weights: Dict[str, float] = {}
        self._performance_history: Dict[str, List[Dict[str, float]]] = {}
        self._last_rebalance_time: Optional[datetime] = None
        self._running = False
        self._tasks: List[asyncio.Task] = []

        self._strategy_names = self._load_strategy_names_from_config()
        self._strategy_manager = None  # 由 Scheduler 注入，作为策略名称单一事实来源
        self._adaptive_controller = None  # 由 Scheduler 注入，作为实时权重权威来源

        # ── DynamicAllocator 集成 ───────────────────────────
        self._dynamic_allocator: Optional[DynamicAllocator] = None
        self._last_allocation_plan: Optional[AllocationPlan] = None
        self._market_regime: MarketRegime = MarketRegime.UNKNOWN
        self._portfolio_optimizer = None

    async def start(self):
        if not self._enabled:
            logger.info("Allocation Agent is disabled")
            return

        if self._running:
            return

        logger.info("Starting Allocation Agent")
        self._running = True
        self._last_rebalance_time = datetime.now()
        await self._load_initial_weights()

        logger.info(
            "AllocationAgent running in observer mode; AdaptiveController owns live strategy weights"
        )
        self._tasks.append(asyncio.create_task(self._performance_monitor_loop()))

    async def stop(self):
        self._running = False
        tasks, self._tasks = self._tasks, []
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("Allocation Agent stopped")

    async def _load_initial_weights(self):
        """从配置加载初始权重"""
        trading_cfg = self.config.get("trading", {})
        for strategy in self._strategy_names:
            key = f"{strategy}_allocation"
            self._strategy_weights[strategy] = _finite(trading_cfg.get(key, 0.167), 0.0)

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

    def set_learning_memory(self, memory) -> None:
        """Inject shared cross-agent performance memory."""
        self._learning_memory = memory

    def set_dynamic_allocator(self, allocator: DynamicAllocator) -> None:
        """注入 DynamicAllocator，使用增强版资金分配引擎"""
        self._dynamic_allocator = allocator
        if self._portfolio_optimizer:
            allocator.set_portfolio_optimizer(self._portfolio_optimizer)
        logger.info("AllocationAgent: DynamicAllocator injected")

    def set_strategy_manager(self, strategy_manager) -> None:
        """注入 StrategyManager 作为策略名称的单一事实来源。

        注入后刷新 self._strategy_names，确保与系统其他模块使用同一套
        策略列表（含 spot_grid / spot_martingale 等）。
        """
        self._strategy_manager = strategy_manager
        if strategy_manager is not None:
            try:
                # 取所有已注册策略名（不限于 enabled），与 StrategyManager 注册表一致
                self._strategy_names = list(strategy_manager.get_all_descriptors().keys())
                logger.info(
                    f"AllocationAgent: strategy names refreshed from StrategyManager "
                    f"({len(self._strategy_names)} strategies)"
                )
            except Exception as e:
                logger.warning(f"StrategyManager strategy name refresh failed: {e}")

    def _load_strategy_names_from_config(self) -> List[str]:
        """从 config 读取策略名称列表（strategy_manager 未注入时的兜底）。

        遍历 strategies.* 配置项，保持与 StrategyManager 同口径。
        """
        strategies_cfg = self.config.get("strategies", {})
        if strategies_cfg:
            return list(strategies_cfg.keys())
        # 极端兜底：config 中无 strategies 段时使用已知策略名
        return ["grid", "trend", "scalping", "arbitrage", "spot_grid", "spot_martingale"]

    def set_adaptive_controller(self, adaptive_controller) -> None:
        """注入 AdaptiveController 作为实时权重的权威来源。

        注入后，get_recommendations / get_current_allocation / get_allocation_report
        将以 AdaptiveController.get_allocations() 返回的实时可部署权重为基准，
        避免使用启动时加载的陈旧 config 快照产生误导性 diff。
        """
        self._adaptive_controller = adaptive_controller
        logger.info(
            "AllocationAgent: AdaptiveController injected as live-weight authority"
        )

    def _get_live_weights(self) -> Dict[str, float]:
        """获取实时权威权重（供建议计算使用）。

        优先使用 AdaptiveController.get_allocations()（单一权威读取入口，
        含可部署性门控）；未注入时回退到本地 config 快照（仅用于离线/测试场景）。
        """
        if self._adaptive_controller is not None and hasattr(self._adaptive_controller, "get_allocations"):
            try:
                return dict(self._adaptive_controller.get_allocations())
            except Exception as e:
                logger.debug(f"AdaptiveController.get_allocations failed: {e}")
        return dict(self._strategy_weights)

    def set_market_regime(self, regime_str: str) -> None:
        """设置当前市场状态（用于DynamicAllocator）"""
        try:
            self._market_regime = MarketRegime(regime_str)
        except ValueError:
            self._market_regime = MarketRegime.UNKNOWN

    def get_optimized_weights(self) -> Dict[str, float]:
        """从 PortfolioOptimizer 获取 MPT 最优权重作为再平衡目标。

        fail-closed：优化器缺失或异常时返回空权重（不编造建议），由权威
        AdaptiveController 继续持有真实分配口径。
        """
        if not hasattr(self, '_portfolio_optimizer') or not self._portfolio_optimizer:
            return {}

        try:
            result = self._portfolio_optimizer.optimize()
            if result and result.optimal_weights:
                merged = {}
                merged.update(result.optimal_weights)
                enabled = set(self._enabled_strategy_names())
                merged = {
                    name: max(0.0, _finite(weight, 0.0))
                    for name, weight in merged.items() if name in enabled
                }
                total = sum(merged.values())
                if total > 0:
                    return {name: weight / total for name, weight in merged.items()}
        except Exception as e:
            logger.warning(f"PortfolioOptimizer failed; returning empty recommendation (fail-closed): {e}")
        return {}

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
            if self._learning_memory is not None:
                try:
                    self._learning_memory.record(
                        agent="allocation_agent",
                        kind="strategy_performance",
                        context={"strategy": strategy},
                        outcome={
                            "win_rate": metrics["win_rate"],
                            "profit_factor": metrics["profit_factor"],
                            "sharpe_ratio": metrics["sharpe_ratio"],
                            "max_drawdown": metrics["max_drawdown"],
                            "total_pnl": metrics["total_pnl"],
                            "trade_count": metrics["trade_count"],
                        },
                    )
                except Exception as e:
                    logger.debug(f"Shared strategy memory write failed: {e}")

    async def _rebalance(self):
        """Observer-mode no-op.

        AdaptiveController 是唯一的实时分配权威；AllocationAgent 不再执行
        任何写入（不修改 strategy_weights、不调用 AccountManager、不回写 config）。
        若需触发真实再平衡，请通过 AdaptiveController 的权威入口。
        """
        logger.info(
            "AllocationAgent._rebalance is a no-op in observer mode; "
            "AdaptiveController is the sole live allocation authority"
        )
        self._last_rebalance_time = datetime.now()

    async def _calculate_optimal_allocation(self) -> Dict[str, float]:
        """计算最优资金分配"""
        if self._allocation_method == "equal":
            return self.get_optimized_weights()
        elif self._allocation_method == "performance":
            return self.get_optimized_weights()
        elif self._allocation_method == "risk_adjusted":
            return self.get_optimized_weights()
        elif self._allocation_method == "dynamic":
            return await self._allocate_dynamic()
        else:
            return await self._allocate_dynamic()

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
            logger.warning("DynamicAllocator unavailable; returning empty recommendation (fail-closed)")
            return {}

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

        total_equity = _finite(total_equity, 0.0)
        total_capital = _finite(total_capital, total_equity)
        if total_equity <= 0:
            logger.warning("Allocation skipped: equity unavailable; returning empty recommendation (fail-closed)")
            return {}

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
            logger.error(f"DynamicAllocator failed: {e}; returning empty recommendation (fail-closed)")
            return {}

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

        pnl_values = []
        for h in history:
            v = _finite(h.get("total_pnl"), None)
            if v is None:
                continue
            pnl_values.append(v)
        if len(pnl_values) < 5:
            return 0.02

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
        pnl_values = []
        for t in trades:
            try:
                v = float(t.pnl_usdt)
            except (TypeError, ValueError):
                continue
            if math.isnan(v) or math.isinf(v):
                continue
            pnl_values.append(v)
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
        """Observer-mode no-op.

        单一权威写入已收敛到 AdaptiveController；AllocationAgent 不再写入
        AccountManager 或 config，避免双引擎改写策略权重。返回空变更集。
        """
        logger.debug(
            "AllocationAgent._apply_allocation_changes is a no-op in observer mode; "
            "AdaptiveController owns live strategy weights"
        )
        return {}

    def _publish_allocations(self, weights: Dict[str, float]) -> None:
        """Observer-mode no-op.

        单一权威写入已收敛到 AdaptiveController；此处不再向 AccountManager 或
        config 发布权重，避免双引擎改写策略权重。
        """
        logger.debug(
            "AllocationAgent._publish_allocations is a no-op in observer mode; "
            "AdaptiveController is the sole live allocation authority"
        )

    def get_current_allocation(self) -> Dict[str, float]:
        """获取当前资金分配（实时权威口径）。

        优先返回 AdaptiveController 的实时可部署权重；未注入时回退到本地快照。
        """
        return self._get_live_weights()

    def get_allocation_report(self) -> Dict[str, Any]:
        """生成资金分配报告（增强版：含 DynamicAllocator 数据）。

        current_weights 字段使用实时权威权重，避免与真实分配口径脱节。
        """
        report = {
            "last_rebalance_time": self._last_rebalance_time.isoformat() if self._last_rebalance_time else None,
            "allocation_method": self._allocation_method,
            "current_weights": self._get_live_weights(),
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
        """获取资金分配建议（advisory only，仅供参考，不直接执行）。

        以 AdaptiveController 的实时可部署权重为 current 基准，计算与
        优化器建议权重的差异，避免使用陈旧 config 快照产生误导性 diff。
        真实再平衡由 AdaptiveController 权威执行。
        """
        optimal = await self._calculate_optimal_allocation()
        live_weights = self._get_live_weights()
        recommendations = {}

        for strategy, optimal_weight in optimal.items():
            current_weight = live_weights.get(strategy, 0)
            diff = optimal_weight - current_weight

            recommendations[strategy] = {
                "current": round(current_weight, 4),
                "optimal": round(optimal_weight, 4),
                "change": round(diff, 4),
                "action": "increase" if diff > 0.01 else "decrease" if diff < -0.01 else "hold"
            }

        return recommendations

    async def manual_rebalance(self):
        """Observer-mode no-op.

        手动触发再平衡在 observer 模式下被禁用；真实再平衡需通过
        AdaptiveController 的权威入口执行，避免双引擎冲突。
        """
        logger.warning(
            "AllocationAgent.manual_rebalance is disabled in observer mode; "
            "use AdaptiveController for live allocation changes"
        )