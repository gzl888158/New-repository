"""
统一自治协调器 QuantAGIOrchestrator
====================================

资金侧自治闭环的「大脑」，负责把已成熟的智能组件编排成一条可自愈、可解释、
可观测的闭环，而不重新实现它们内部的算法（Kelly、regime、归因等）。

闭环流程（一个 tick = 一次 run_cycle）：
    感知 Perceive  → 诊断 Diagnose  → 决策 Decide  → 执行 Act  → 反馈 Reflect

设计约束：
  - 纯编排：只调用现有模块接口，不复制/重写任何算法。
  - fail-closed：任一感知/决策环节异常都降级跳过并记日志，不放大资金风险；
    无法确认权益（None 或 <=0）时返回保守的空计划，绝不放行。
  - JSON 安全：所有输出 dict 经 utils.helpers 的 safe_* 清洗，json.dumps 无 NaN/Inf。
  - 幂等 + 冷却：run_cycle 带最小间隔 cooldown_seconds（默认 300s，config 可配），
    冷却期内直接返回上次报告的缓存副本。
  - 不污染调用方：接收的 dict 先 copy 再操作，返回的报告亦是副本。
"""

import copy
import inspect
import json
import math
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from decision.rl_agent import StateEncoding
from risk.dynamic_allocator import MarketRegime
from utils.helpers import safe_div, safe_finite, safe_float, safe_int

# regime_engine 输出值 → DynamicAllocator 的 MarketRegime（仅编排映射，不重算）
_REGIME_TO_ALLOCATOR: Dict[str, MarketRegime] = {
    "trend_bullish": MarketRegime.TRENDING_UP,
    "trend_bearish": MarketRegime.TRENDING_DOWN,
    "breakout": MarketRegime.BREAKOUT,
    "breakdown": MarketRegime.BREAKDOWN,
    "reversal": MarketRegime.REVERSAL,
    "trend_up": MarketRegime.TRENDING_UP,
    "trend_down": MarketRegime.TRENDING_DOWN,
    "range_bound": MarketRegime.RANGING,
    "ranging": MarketRegime.RANGING,
    "range": MarketRegime.RANGING,
    "extreme_volatility": MarketRegime.HIGH_VOLATILITY,
    "funding_crush": MarketRegime.HIGH_VOLATILITY,
    "liquidity_crisis": MarketRegime.HIGH_VOLATILITY,
}

_PROJECTION_REGIME_ALIASES = {
    "trend_bullish": "trend_up",
    "trending_up": "trend_up",
    "trend_up": "trend_up",
    "bullish": "trend_up",
    "trend_bearish": "trend_down",
    "trending_down": "trend_down",
    "trend_down": "trend_down",
    "bearish": "trend_down",
    "range": "range_bound",
    "ranging": "range_bound",
    "range_bound": "range_bound",
    "extreme_volatility": "high_volatility",
    "funding_crush": "high_volatility",
    "liquidity_crisis": "high_volatility",
    "high_volatility": "high_volatility",
    "low_volatility": "low_volatility",
}


def _canonical_projection_regime(regime_value: Any) -> str:
    """Canonical regime key shared by forecast and accuracy samples."""
    regime = str(regime_value or "unknown").strip().lower()
    return _PROJECTION_REGIME_ALIASES.get(regime, regime or "unknown")


def _map_regime(regime_value: Any) -> MarketRegime:
    """把 regime_engine 的 regime 字符串映射到 DynamicAllocator 的枚举，未知回退 UNKNOWN。"""
    if regime_value is None:
        return MarketRegime.UNKNOWN
    s = str(regime_value)
    mapped = _REGIME_TO_ALLOCATOR.get(s)
    if mapped is not None:
        return mapped
    try:
        return MarketRegime(s)
    except ValueError:
        return MarketRegime.UNKNOWN


def _health_grade_for(score: float) -> str:
    """健康度评分 → 等级（与 ContributionAnalyzer 口径一致）。"""
    score = safe_float(score, 0.0)
    if score >= 80:
        return "A"
    if score >= 60:
        return "B"
    if score >= 45:
        return "C"
    if score >= 25:
        return "D"
    return "F"


def _coerce_dict(value: Any) -> Dict[str, Any]:
    """把 value 规范为 dict；非 dict（str/list/None 等）回退空 dict。

    防御「协议不兼容 / 脏数据」：contribution/strategies/net_exposure 等感知子结构
    若被注入非 dict（如字符串）会导致 .get()/.items() 抛 AttributeError，进而使整个
    run_cycle 崩溃。本 helper 保证这些读取永不因类型错误而崩溃（fail-closed）。
    """
    return value if isinstance(value, dict) else {}


def snapshot_to_dict(snapshot: Any) -> Dict[str, Any]:
    """独立算子：把 ContributionSnapshot（或兼容对象）转为安全 dict。

    纯函数、无副作用、可独立单测。只保留编排所需字段，所有数值经 safe_* 清洗。
    """
    if snapshot is None:
        return {"available": False, "strategies": {}, "total_pnl": 0.0, "overall_health_score": 0.0}
    if isinstance(snapshot, dict):
        return copy.deepcopy(snapshot)

    strategies: Dict[str, Any] = {}
    raw_strategies = getattr(snapshot, "strategies", None) or {}
    for name, c in raw_strategies.items():
        strategies[str(name)] = {
            "total_pnl": safe_finite(getattr(c, "total_pnl", 0.0), 0.0),
            "win_rate": safe_finite(getattr(c, "win_rate", 0.0), 0.0),
            "profit_factor": safe_finite(getattr(c, "profit_factor", 0.0), 0.0),
            "max_drawdown": safe_finite(getattr(c, "max_drawdown", 0.0), 0.0),
            "max_drawdown_duration_hours": safe_finite(getattr(c, "max_drawdown_duration_hours", 0.0), 0.0),
            "total_trades": safe_int(getattr(c, "total_trades", 0), 0),
            "consecutive_losses": safe_int(getattr(c, "consecutive_losses", 0), 0),
            "stop_loss_count": safe_int(getattr(c, "stop_loss_count", 0), 0),
            "take_profit_count": safe_int(getattr(c, "take_profit_count", 0), 0),
            "health_score": safe_finite(getattr(c, "health_score", 0.0), 0.0),
            "health_grade": getattr(c, "health_grade", "N/A"),
            "lifecycle": getattr(c, "lifecycle", "unknown"),
            "lifecycle_last_trade_age_hours": safe_finite(getattr(c, "lifecycle_last_trade_age_hours", 0.0), 0.0),
            "trend": getattr(c, "trend", "stable"),
            "trend_pnl_7d_vs_30d": safe_finite(getattr(c, "trend_pnl_7d_vs_30d", 0.0), 0.0),
            "pnl_per_capital_pct": safe_finite(getattr(c, "pnl_per_capital_pct", 0.0), 0.0),
            "sharpe_ratio": safe_finite(getattr(c, "sharpe_ratio", 0.0), 0.0),
            "risk_adjusted_contribution": safe_finite(getattr(c, "risk_adjusted_contribution", 0.0), 0.0),
            "volatility": safe_finite(getattr(c, "volatility", 0.0), 0.0),
            "realized_pnl": safe_finite(getattr(c, "realized_pnl", 0.0), 0.0),
            "unrealized_pnl": safe_finite(getattr(c, "unrealized_pnl", 0.0), 0.0),
            "total_fees": safe_finite(getattr(c, "total_fees", 0.0), 0.0),
            "total_funding_cost": safe_finite(getattr(c, "total_funding_cost", 0.0), 0.0),
            "total_slippage_cost": safe_finite(getattr(c, "total_slippage_cost", 0.0), 0.0),
            "total_spread_cost": safe_finite(getattr(c, "total_spread_cost", 0.0), 0.0),
            "long_pnl": safe_finite(getattr(c, "long_pnl", 0.0), 0.0),
            "short_pnl": safe_finite(getattr(c, "short_pnl", 0.0), 0.0),
            "long_trades": safe_int(getattr(c, "long_trades", 0), 0),
            "short_trades": safe_int(getattr(c, "short_trades", 0), 0),
            "avg_win": safe_finite(getattr(c, "avg_win", 0.0), 0.0),
            "avg_loss": safe_finite(getattr(c, "avg_loss", 0.0), 0.0),
            "stop_loss_pnl": safe_finite(getattr(c, "stop_loss_pnl", 0.0), 0.0),
            "take_profit_pnl": safe_finite(getattr(c, "take_profit_pnl", 0.0), 0.0),
            "pnl_per_trade": safe_finite(getattr(c, "pnl_per_trade", 0.0), 0.0),
            "active_hours": safe_finite(getattr(c, "active_hours", 0.0), 0.0),
            "pnl_per_hour": safe_finite(getattr(c, "pnl_per_hour", 0.0), 0.0),
            "delta_pnl": safe_finite(getattr(c, "delta_pnl", 0.0), 0.0),
            "delta_health": safe_finite(getattr(c, "delta_health", 0.0), 0.0),
        }

    return {
        "available": bool(getattr(snapshot, "available", True)),
        "total_pnl": safe_finite(getattr(snapshot, "total_pnl", 0.0), 0.0),
        "total_trades": safe_int(getattr(snapshot, "total_trades", 0), 0),
        "total_fees": safe_finite(getattr(snapshot, "total_fees", 0.0), 0.0),
        "total_unrealized_pnl": safe_finite(getattr(snapshot, "total_unrealized_pnl", 0.0), 0.0),
        "overall_health_score": safe_finite(getattr(snapshot, "overall_health_score", 0.0), 0.0),
        "concentration": safe_finite(getattr(snapshot, "concentration", 0.0), 0.0),
        "diversification_score": safe_finite(getattr(snapshot, "diversification_score", 0.0), 0.0),
        "efficiency_score": safe_finite(getattr(snapshot, "efficiency_score", 0.0), 0.0),
        "synergy_score": safe_finite(getattr(snapshot, "synergy_score", 0.0), 0.0),
        "lifecycle_summary": copy.deepcopy(getattr(snapshot, "lifecycle_summary", {}) or {}),
        "strategies": strategies,
    }


def build_strategy_metrics(contrib: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """独立算子：从贡献快照构建 DynamicAllocator 所需的 strategy_metrics。

    纯函数、无副作用、可独立单测。DynamicAllocator 优先级评分里 sharpe_ratio 权重
    最高（0.4），缺省为 0 会劣化分配质量。快照未直接提供 sharpe，这里用
    「资本回报率 / 最大回撤」构造无量纲、有界的 Sharpe 代理，并补资本效率字段。
    """
    metrics: Dict[str, Dict[str, Any]] = {}
    for name, c in _coerce_dict(contrib.get("strategies")).items():
        cap_ret = safe_float(c.get("pnl_per_capital_pct"), 0.0)   # 资本回报率（百分比）
        max_dd = safe_float(c.get("max_drawdown"), 0.0)

        # 优先使用真实 sharpe（来自 contribution_analyzer 的 mean/std）；不可用时回退到代理
        real_sharpe = safe_float(c.get("sharpe_ratio"), 0.0)
        if real_sharpe != 0.0:
            sharpe = max(-5.0, min(5.0, real_sharpe))
        else:
            # 收益/回撤 → Sharpe 代理，分母下限 0.01 防除零，结果钳制到 [-5, 5] 防量级失控
            sharpe = safe_div(cap_ret, max(max_dd, 0.01), 0.0)
            sharpe = max(-5.0, min(5.0, sharpe))

        metrics[str(name)] = {
            "win_rate": safe_float(c.get("win_rate"), 0.5),
            "profit_factor": safe_float(c.get("profit_factor"), 1.0),
            "max_drawdown": max_dd,
            "trade_count": safe_float(c.get("total_trades"), 0),
            "consecutive_losses": safe_int(c.get("consecutive_losses"), 0),
            "total_pnl": safe_float(c.get("total_pnl"), 0.0),
            "realized_pnl": safe_float(c.get("realized_pnl"), 0.0),
            "unrealized_pnl": safe_float(c.get("unrealized_pnl"), 0.0),
            "pnl_per_capital_pct": cap_ret,
            "sharpe_ratio": sharpe,
        }
    return metrics


# 动作优先级（越大越优先保留/执行）：风险收敛方向 > 进攻方向 > 通知型。
# 供 _apply_decision_quality 稳定排序，并与 _reconcile_actions 的冲突消解衔接。
_ACTION_PRIORITY = {
    "strategy_pause": 100,   # 停开新仓（防持续亏损，最高）
    "param_adjust": 90,      # 降杠杆
    "profit_take_close": 85,  # 落袋平仓
    "reallocate": 70,        # 资金重分配（decrease 优先于 increase，由冲突消解细化）
    "idle_cash_deploy": 50,  # 闲置资金归集
    "self_heal": 40,         # 通知型自愈事件
    "allocation_auto_action": 30,
    "allocation_recommendation": 30,
    "alert_action": 10,      # 告警（仅通知）
}

# 事前风控趋势守卫的告警类型集合（供多守卫共振检测：同一策略多维度同时衰退 → 升级收敛）
# 注意：consecutive_losses（strategy_losing_streak）是阈值型守卫，不纳入共振维度，
# 共振只统计「趋势型守卫」——health/win_rate/sharpe/profit_factor/max_drawdown/capital_return/
# volatility/pnl_momentum/pnl_per_trade/delta_pnl 各看一种衰退模式。
_DECAY_ALERT_TYPES = frozenset({
    "health_deteriorating",
    "win_rate_deteriorating",
    "sharpe_deteriorating",
    "profit_factor_deteriorating",
    "max_drawdown_deteriorating",
    "drawdown_duration_deteriorating",
    "capital_return_deteriorating",
    "volatility_deteriorating",
    "pnl_momentum_deteriorating",
    "pnl_per_trade_deteriorating",
    "delta_pnl_deteriorating",
})

# 组合级风险共振检测的维度映射（供 portfolio_risk_resonance_guard 检测多维度同时恶化）。
# 组合级风险四个独立维度：correlation（策略间相关性）、weight_concentration（资金配置权重
# 集中）、profit_concentration（盈利来源集中）、tail_risk（尾部深度）。每个维度可能对应多个
# 告警类型（含阈值型与趋势型），按维度去重后统计触发的维度数——避免同一维度多个告警重复
# 计数（如 high_concentration 与 low_diversification 同为「权重集中」维度，portfolio_
# concentration_rising 也同属该维度的趋势层）。
_PORTFOLIO_RISK_DIMENSIONS = {
    "correlation": frozenset({"high_correlation", "diversification_eroding", "portfolio_correlation_rising"}),
    "weight_concentration": frozenset({"high_concentration", "low_diversification", "portfolio_concentration_rising"}),
    "profit_concentration": frozenset({"profit_concentration", "profit_concentration_rising"}),
    "tail_risk": frozenset({"high_tail_risk", "tail_risk_rising"}),
}


class QuantAGIOrchestrator:
    """统一自治协调器：编排 regime/contribution/capital/dynamic 四大组件形成自治闭环。"""

    def __init__(
        self,
        config: Dict = None,
        regime_engine=None,
        contribution_analyzer=None,
        capital_allocator=None,
        dynamic_allocator=None,
        account_manager=None,
        equity_monitor=None,
        strategy_correlation=None,
        regime_arbiter=None,
        rl_agent=None,
    ):
        # 不污染调用方：config 先 copy
        self.config = dict(config or {})
        agi_cfg = self.config.get("agi_orchestrator") or {}

        self.cooldown_seconds = safe_float(agi_cfg.get("cooldown_seconds"), 300.0)
        if self.cooldown_seconds < 0:
            self.cooldown_seconds = 300.0
        self.utilization_low_threshold = safe_float(agi_cfg.get("utilization_low_threshold"), 0.10)
        # 市场状态突变告警的置信度门槛：低于该值只作「疑似变化」提示，不触发突变告警
        self.regime_confidence_threshold = safe_float(agi_cfg.get("regime_confidence_threshold"), 0.5)
        if not 0.0 <= self.regime_confidence_threshold <= 1.0:
            self.regime_confidence_threshold = 0.5
        # regime 强度低门槛：低于该值标注分配建议低置信度（strength 入分配）
        self.regime_strength_low_threshold = safe_float(agi_cfg.get("regime_strength_low_threshold"), 0.5)
        if not 0.0 <= self.regime_strength_low_threshold <= 1.0:
            self.regime_strength_low_threshold = 0.5
        # 资金分析阈值（集中度 HHI / 分散化得分 / 资金效率得分，均为 0-1）
        self.concentration_high_threshold = safe_float(agi_cfg.get("concentration_high_threshold"), 0.6)
        self.diversification_low_threshold = safe_float(agi_cfg.get("diversification_low_threshold"), 0.4)
        self.capital_efficiency_low_threshold = safe_float(agi_cfg.get("capital_efficiency_low_threshold"), 0.15)
        # 低风险闲置资金归集：闲置占比超过此阈值且存在 A/B 级核心策略时，自动生成归集动作
        self.idle_deploy_threshold = safe_float(agi_cfg.get("idle_deploy_threshold"), 0.10)
        # 健康度最小样本量：低于此成交数不评估健康度，判为 N/A（与 contribution_analyzer 口径一致）
        self.min_health_sample = int(safe_float(agi_cfg.get("min_health_sample"), 5))
        self.state_path = agi_cfg.get("state_path") or os.path.join("data", "agi_orchestrator_state.json")
        simulation_cfg = agi_cfg.get("simulation_safety") or {}
        paper_cfg = self.config.get("paper_trading") or {}
        self._simulation_safety_enabled = (
            isinstance(simulation_cfg, dict)
            and bool(simulation_cfg.get("enabled", False))
            and isinstance(paper_cfg, dict)
            and bool(paper_cfg.get("enabled", False))
            and str(paper_cfg.get("mode", "")).lower() == "sandbox"
        )
        self._simulation_fail_closed_threshold = max(
            1,
            int(safe_float(simulation_cfg.get("fail_closed_streak_threshold"), 3))
            if isinstance(simulation_cfg, dict) else 3,
        )
        self._simulation_bear_case_reduce_ratio = max(
            0.0,
            min(
                1.0,
                safe_float(simulation_cfg.get("bear_case_reduce_ratio"), 0.25)
                if isinstance(simulation_cfg, dict) else 0.25,
            ),
        )
        self._simulation_fail_closed_streak = 0
        self._simulation_kill_switch_enabled = False
        self._simulation_kill_switch_reason = ""

        # ── 循环失败熔断（cycle circuit breaker）──
        # run_cycle 连续失败超过阈值时自停，防止静默持续异常
        self._cycle_failure_streak = 0
        self._cycle_failure_threshold = max(
            3, safe_int(agi_cfg.get("cycle_failure_threshold"), 10)
        )
        self._cycle_halted = False
        self._cycle_halt_reason = ""

        # ── 账户级收益检测自动平仓（profit_take）──
        # AGI 综合账户整体浮盈，达到阈值后生成 profit_take_close 动作，
        # 由 RestrictedExecutionChannel 低风险自动执行（平仓降风险），
        # 与持仓级 ProfitLockEngine 互补：前者锁单仓微利，后者落袋账户级利润。
        pt_cfg = agi_cfg.get("profit_take", {}) or {}
        self._profit_take_enabled = bool(pt_cfg.get("enabled", False))
        self._profit_take_activation_pct = safe_float(pt_cfg.get("activation_pct"), 0.03)
        self._profit_take_max_pct = safe_float(pt_cfg.get("max_pct"), 0.10)
        self._profit_take_max_close_ratio = safe_float(pt_cfg.get("max_close_ratio"), 0.5)
        # 下限护栏：激活阈值不得低于往返手续费，避免无意义平仓
        self._profit_take_activation_pct = max(self._profit_take_activation_pct, 0.005)
        self._profit_take_max_pct = max(self._profit_take_max_pct, self._profit_take_activation_pct)
        self._profit_take_max_close_ratio = max(0.05, min(0.9, self._profit_take_max_close_ratio))

        # ── 主动风控响应（risk_response）──
        # 诊断到策略健康度 F / 连续亏损 / 趋势恶化时，主动生成 reallocate 减配动作，
        # 由 RestrictedExecutionChannel 落地（autonomous 模式），风险早期即收敛敞口，
        # 不再停留在「仅告警」。
        rr_cfg = agi_cfg.get("risk_response", {}) or {}
        self._risk_response_enabled = bool(rr_cfg.get("enabled", False))
        self._risk_response_reduce_target = safe_float(rr_cfg.get("reduce_target"), 0.0)
        self._risk_response_reduce_target = max(0.0, min(1.0, self._risk_response_reduce_target))

        # ── 市场状态自适应（regime_adaptive）──
        # profit_take 落袋力度按市场状态动态调整：趋势市让利润奔跑（放宽落袋），
        # 震荡市积极落袋（收紧落袋）。
        ra_cfg = agi_cfg.get("regime_adaptive", {}) or {}
        self._regime_adaptive_enabled = bool(ra_cfg.get("enabled", False))
        self._trend_profit_take_mult = safe_float(ra_cfg.get("trend_multiplier"), 1.3)
        self._range_profit_take_mult = safe_float(ra_cfg.get("range_multiplier"), 0.8)

        # ── 跨周期学习记忆（learning）──
        # 记录最近若干周期的决策结果（健康度/收益/市场状态），据健康度趋势自适应
        # 微调 profit_take 落袋阈值：改善→放宽（让利润跑）、恶化→收紧（积极落袋）。
        lr_cfg = agi_cfg.get("learning", {}) or {}
        self._learning_enabled = bool(lr_cfg.get("enabled", False))
        self._learning_memory_size = int(safe_float(lr_cfg.get("memory_size"), 10))
        self._learning_memory_size = max(3, min(50, self._learning_memory_size))
        self._learning_max_adjust_pct = safe_float(lr_cfg.get("max_adjust_pct"), 0.4)
        self._learning_max_adjust_pct = max(0.0, min(0.6, self._learning_max_adjust_pct))
        # 决策记忆指数衰减：近期周期权重更高，让 _decision_quality_score 对市场状态
        # 切换更快响应（旧记忆等权导致质量分数滞后于真实状态）。alpha=1.0 等价关闭
        # （默认关闭，向后兼容；仅在 learning 配置显式指定 memory_decay_alpha 时启用）。
        self._learning_memory_decay_alpha = safe_float(
            lr_cfg.get("memory_decay_alpha"), 1.0)
        self._learning_memory_decay_alpha = max(0.5, min(1.0, self._learning_memory_decay_alpha))
        self._decision_memory: deque = deque(maxlen=self._learning_memory_size)

        # ── 目标导向规划（goal_planning）──
        # 基于日/周收益目标与最大回撤约束，动态调整风险预算：
        # 收益达标 → 减仓落袋；接近回撤约束 → 降仓防守。
        gp_cfg = agi_cfg.get("goal_planning", {}) or {}
        self._goal_planning_enabled = bool(gp_cfg.get("enabled", False))
        self._goal_daily_target_pct = safe_float(gp_cfg.get("daily_target_return_pct"), 0.02)
        self._goal_max_drawdown_pct = safe_float(gp_cfg.get("max_drawdown_pct"), 0.08)
        self._goal_near_drawdown_ratio = safe_float(gp_cfg.get("near_drawdown_ratio"), 0.8)
        self._goal_reached_reduce_target = safe_float(gp_cfg.get("goal_reached_reduce_target"), 0.3)
        self._goal_reached_reduce_target = max(0.0, min(1.0, self._goal_reached_reduce_target))

        # ── 策略参数自适应（param_adaptation）──
        # AGI 对健康度 F / 趋势恶化的策略，生成 param_adjust 动作降低杠杆，
        # 由 scheduler 落地（调策略 update_config），实现参数级风险收敛。
        pa_cfg = agi_cfg.get("param_adaptation", {}) or {}
        self._param_adaptation_enabled = bool(pa_cfg.get("enabled", False))
        self._param_health_f_leverage = safe_float(pa_cfg.get("health_f_leverage"), 1.0)
        self._param_declining_leverage = safe_float(pa_cfg.get("declining_leverage"), 2.0)
        # 参数自适应限速：同一策略在 min_interval_cycles 个周期内只调整一次参数，
        # 防止 AGI 过度迭代（曾出现 sniper 在 1.5h 内迭代 20 版杠杆），让参数有
        # 时间沉淀验证后再调整。与 _offensive_min_interval_cycles 同构。
        self._param_adapt_min_interval_cycles = int(safe_float(
            pa_cfg.get("min_interval_cycles"), 3))
        self._param_adapt_min_interval_cycles = max(0, min(100, self._param_adapt_min_interval_cycles))
        # 参数自适应恢复（对称闭环）：此前 param_adaptation 只能降杠杆（单向），策略
        # 健康度 F 降到 1.0 后即使恢复 A/B，杠杆也永久停留低位 → 过度保守、错失盈利，
        # 违背「激进全自动」诉求。健康度恢复（strategy_recovered：grade A/B + improving）
        # 且无回吐/非暂停时，恢复杠杆到 restore_leverage。默认关（升杠杆风险高于降杠杆，
        # 向后兼容），生产按需开启。restore_min_interval_cycles 默认更长（更保守防振荡）。
        self._param_restore_enabled = bool(pa_cfg.get("restore_enabled", False))
        self._param_restore_leverage = safe_float(pa_cfg.get("restore_leverage"), 3.0)
        self._param_restore_leverage = max(0.0, self._param_restore_leverage)
        # 恢复分级：A 级（健康稳定）恢复到 restore_leverage 完整水平；B 级（刚恢复、
        # 尚不稳定）仅恢复到 restore_leverage_b 中间值（更保守），避免 B 级就完全升杠杆。
        self._param_restore_leverage_b = safe_float(pa_cfg.get("restore_leverage_b"), 2.0)
        self._param_restore_leverage_b = max(0.0, self._param_restore_leverage_b)
        self._param_restore_min_interval_cycles = int(safe_float(
            pa_cfg.get("restore_min_interval_cycles"), 5))
        self._param_restore_min_interval_cycles = max(
            0, min(100, self._param_restore_min_interval_cycles))

        # ── 逐币种精细杠杆守卫（symbol_param_guard）──
        # AGI 感知逐币种浮亏（account_manager.get_symbol_unrealized_pnl），单币种浮亏
        # 超阈值时对「该币种」单独精细下调杠杆（param_adjust + symbol 维度），落到
        # currencies.symbol_overrides，实现 ARB 等主流币种在 tier 默认参数之上单独定制
        # 更精细的杠杆/间距——「为 ARB 单独定制更精细的杠杆/间距参数，增强 AGI 能力」。
        spg_cfg = agi_cfg.get("symbol_param_guard", {}) or {}
        self._symbol_param_guard_enabled = bool(spg_cfg.get("enabled", False))
        self._symbol_loss_threshold = safe_float(spg_cfg.get("loss_threshold"), 5.0)
        self._symbol_loss_threshold = max(0.0, self._symbol_loss_threshold)
        self._symbol_reduce_leverage_step = safe_float(spg_cfg.get("reduce_leverage_step"), 1.0)
        self._symbol_reduce_leverage_step = max(0.0, self._symbol_reduce_leverage_step)
        self._symbol_min_leverage = safe_float(spg_cfg.get("min_leverage"), 1.0)
        self._symbol_min_leverage = max(0.0, self._symbol_min_leverage)
        self._symbol_cooldown_cycles = int(safe_float(spg_cfg.get("cooldown_cycles"), 5))
        self._symbol_cooldown_cycles = max(0, self._symbol_cooldown_cycles)

        # ── 现货持有守卫（spot_hold_guard）──
        # AGI 感知逐币种现货持币余额（account_manager.get_spot_holdings），
        # 对现货过度分散（持币种类超限）与现货策略累计盈利达标两种情况收敛现货敞口，
        # 实现「现货不占用过多资金、稳步增长」的现货侧闭环。
        shg_cfg = agi_cfg.get("spot_hold_guard", {}) or {}
        self._spot_hold_guard_enabled = bool(shg_cfg.get("enabled", False))
        self._spot_max_currencies = int(safe_float(shg_cfg.get("max_spot_currencies"), 3))
        self._spot_max_currencies = max(1, self._spot_max_currencies)
        self._spot_profit_take_pct = safe_float(shg_cfg.get("profit_take_pct"), 0.02)
        self._spot_profit_take_pct = max(0.0, self._spot_profit_take_pct)
        # 现货亏损止损回收：现货策略累计亏损占总资金比例达阈值时收敛现货（止损回收资金
        # 到合约），与 profit_take_pct（盈利止盈）对称——现货既「赚了落袋」也「亏了止损」，
        # 不长期占用资金，对应「现货不占用资金、增加合约资金占比」诉求。
        self._spot_loss_take_pct = safe_float(shg_cfg.get("loss_take_pct"), 0.02)
        self._spot_loss_take_pct = max(0.0, self._spot_loss_take_pct)
        self._spot_reduce_target = safe_float(shg_cfg.get("reduce_target"), 0.1)
        self._spot_reduce_target = max(0.0, min(1.0, self._spot_reduce_target))
        self._spot_strategies = ["spot_grid", "spot_martingale"]

        # ── 每小时盈利效率守卫（hourly_pnl_guard）──
        # AGI 感知策略「每小时盈利」（contribution_analyzer.pnl_per_hour），对活跃时长足够、
        # 交易量足够但每小时仍在持续失血（时间价值流失）的策略收敛资金权重，消除「长时间
        # 占用资金却负期望」的低效策略（与频繁交易/杀跌收敛同构，补齐时间维度缺口）。
        hpg_cfg = agi_cfg.get("hourly_pnl_guard", {}) or {}
        self._hourly_pnl_guard_enabled = bool(hpg_cfg.get("enabled", False))
        self._hourly_min_active_hours = safe_float(hpg_cfg.get("min_active_hours"), 24.0)
        self._hourly_min_active_hours = max(0.0, self._hourly_min_active_hours)
        self._hourly_min_trades = int(safe_float(hpg_cfg.get("min_trades"), 3))
        self._hourly_min_trades = max(1, self._hourly_min_trades)
        self._hourly_loss_threshold = safe_float(hpg_cfg.get("hourly_loss_threshold"), 0.5)
        self._hourly_loss_threshold = max(0.0, self._hourly_loss_threshold)
        self._hourly_reduce_target = safe_float(hpg_cfg.get("reduce_target"), 0.1)
        self._hourly_reduce_target = max(0.0, min(1.0, self._hourly_reduce_target))

        # ── 交易频率守卫（trade_frequency_guard）──
        # 与 hourly_pnl_guard（看「每小时盈亏」pnl_per_hour，时间价值流失）区分：本守卫看
        # 「每小时交易笔数」trades_per_hour = total_trades/active_hours——策略即使盈利、手续费
        # 不高，若开仓频率过高（高频刷单）仍属「频繁交易」，单笔滑点/手续费/冲击成本叠加侵蚀利润，
        # 直接对应「消除频繁交易消耗资金」诉求。仅在活跃时长达 min_active_hours 且笔数达 min_trades
        # 时评估。阈值型守卫（非趋势型），从当周期 contribution 读取，不持久化。
        tf_cfg = agi_cfg.get("trade_frequency_guard", {}) or {}
        self._trade_frequency_enabled = bool(tf_cfg.get("enabled", False))
        self._trade_frequency_min_active_hours = safe_float(tf_cfg.get("min_active_hours"), 24.0)
        self._trade_frequency_min_active_hours = max(0.0, self._trade_frequency_min_active_hours)
        self._trade_frequency_min_trades = int(safe_float(tf_cfg.get("min_trades"), 10))
        self._trade_frequency_min_trades = max(1, self._trade_frequency_min_trades)
        self._trade_frequency_threshold = safe_float(tf_cfg.get("frequency_threshold"), 2.0)
        self._trade_frequency_threshold = max(0.0, self._trade_frequency_threshold)
        self._trade_frequency_reduce_target = safe_float(tf_cfg.get("reduce_target"), 0.1)
        self._trade_frequency_reduce_target = max(0.0, min(1.0, self._trade_frequency_reduce_target))

        # ── 风险调整后贡献守卫（risk_adjusted_contribution_guard）──
        # AGI 感知策略「风险调整后贡献」（contribution_analyzer.risk_adjusted_contribution =
        # PnL / MaxDD），对仍盈利但盈利相对所承受回撤过少（小赚大扛，风险收益比失衡）的
        # 策略降杠杆收敛，消除「追涨杀跌、频繁交易」的收益/风险失衡特征。
        rac_cfg = agi_cfg.get("risk_adjusted_contribution_guard", {}) or {}
        self._risk_adjusted_contribution_enabled = bool(rac_cfg.get("enabled", False))
        self._risk_adjusted_contribution_min_trades = int(safe_float(rac_cfg.get("min_trades"), 10))
        self._risk_adjusted_contribution_min_trades = max(1, self._risk_adjusted_contribution_min_trades)
        self._risk_adjusted_contribution_min_ratio = safe_float(rac_cfg.get("min_risk_adjusted_ratio"), 1.0)
        self._risk_adjusted_contribution_min_ratio = max(0.0, self._risk_adjusted_contribution_min_ratio)
        self._risk_adjusted_contribution_leverage = safe_float(rac_cfg.get("deterioration_leverage"), 1.0)
        self._risk_adjusted_contribution_leverage = max(0.0, self._risk_adjusted_contribution_leverage)

        # ── 夏普比率绝对阈值守卫（sharpe_ratio_guard）──
        # AGI 感知策略「夏普比率」（contribution_analyzer.sharpe_ratio = 单笔盈亏均值/标准差），
        # 对「负夏普」（sharpe < min_sharpe_ratio，风险调整后为负收益）的策略降杠杆收敛。
        # 与 sharpe_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」——一个策略
        # 夏普长期为负但未连续下降（平坦负值）时，趋势守卫不触发，但策略实际在「承担风险
        # 却负收益」，是纯亏损信号（均值负 = 平均每笔亏损）。构成夏普维度「阈值 + 趋势」双层
        # （与 volatility/资本回报/手续费等指标的阈值+趋势双层对称）。阈值型守卫，无跨周期状态。
        srg_cfg = agi_cfg.get("sharpe_ratio_guard", {}) or {}
        self._sharpe_ratio_enabled = bool(srg_cfg.get("enabled", False))
        self._sharpe_ratio_min_trades = int(safe_float(srg_cfg.get("min_trades"), 10))
        self._sharpe_ratio_min_trades = max(1, self._sharpe_ratio_min_trades)
        self._sharpe_ratio_min_ratio = safe_float(srg_cfg.get("min_sharpe_ratio"), 0.0)
        self._sharpe_ratio_leverage = safe_float(srg_cfg.get("deterioration_leverage"), 1.0)
        self._sharpe_ratio_leverage = max(0.0, self._sharpe_ratio_leverage)

        # ── 策略休眠资金回收守卫（strategy_staleness_guard）──
        # AGI 感知策略「距上次交易小时数」（contribution_analyzer.lifecycle_last_trade_age_hours），
        # 对曾活跃但长时间未交易（资金闲置）的策略回收资金（reallocate decrease），事前释放
        # 闲置资金到活跃策略，对应「资金利用率」诉求。
        ssg_cfg = agi_cfg.get("strategy_staleness_guard", {}) or {}
        self._strategy_staleness_enabled = bool(ssg_cfg.get("enabled", False))
        self._strategy_staleness_reclaim_idle_hours = safe_float(ssg_cfg.get("reclaim_idle_hours"), 48.0)
        self._strategy_staleness_reclaim_idle_hours = max(0.0, self._strategy_staleness_reclaim_idle_hours)
        self._strategy_staleness_min_trades = int(safe_float(ssg_cfg.get("min_trades"), 5))
        self._strategy_staleness_min_trades = max(1, self._strategy_staleness_min_trades)
        self._strategy_staleness_reclaim_target = safe_float(ssg_cfg.get("reclaim_target"), 0.05)
        self._strategy_staleness_reclaim_target = max(0.0, min(1.0, self._strategy_staleness_reclaim_target))

        # ── 健康度骤降守卫（health_crash_guard）──
        # AGI 感知策略「健康度增量」（contribution_analyzer.delta_health = 本周期 health_score
        # 相对上一快照的增量），对单周期健康度骤降（突然恶化）的策略降杠杆收敛。与
        # health_trend_guard（连续 window 周期渐降，事前慢信号）区分：本守卫看「单周期骤降」
        # （急信号）——健康度在一个周期内暴跌，即使尚未连续下降也应收敛。
        hc_cfg = agi_cfg.get("health_crash_guard", {}) or {}
        self._health_crash_enabled = bool(hc_cfg.get("enabled", False))
        self._health_crash_threshold = safe_float(hc_cfg.get("crash_threshold"), 20.0)
        self._health_crash_threshold = max(0.0, self._health_crash_threshold)
        self._health_crash_min_trades = int(safe_float(hc_cfg.get("min_trades"), 10))
        self._health_crash_min_trades = max(1, self._health_crash_min_trades)
        self._health_crash_leverage = safe_float(hc_cfg.get("deterioration_leverage"), 1.0)
        self._health_crash_leverage = max(0.0, self._health_crash_leverage)

        # ── 决策震荡抑制（decision_oscillation_guard）──
        # AGI 抑制「减了又加」的决策震荡：记录各策略上次被防守性减仓的周期号，进攻加仓前
        # 检查该策略是否在 cooldown_cycles 内刚被防守减仓，是则跳过（避免频繁交易消耗资金）。
        do_cfg = agi_cfg.get("decision_oscillation_guard", {}) or {}
        self._oscillation_enabled = bool(do_cfg.get("enabled", False))
        self._oscillation_cooldown_cycles = int(safe_float(do_cfg.get("cooldown_cycles"), 3))
        self._oscillation_cooldown_cycles = max(0, self._oscillation_cooldown_cycles)
        self._last_defensive_reduce_cycle: Dict[str, int] = {}

        # ── 组合多空净敞口监控（net_exposure_guard）──
        # AGI 感知账户多空敞口（account_manager.get_long_exposure/get_short_exposure），
        # 对方向性失衡（净敞口占毛敞口比例过高）收敛敞口，避免组合过度单边押注。
        ne_cfg = agi_cfg.get("net_exposure_guard", {}) or {}
        self._net_exposure_enabled = bool(ne_cfg.get("enabled", False))
        self._net_exposure_imbalance_threshold = safe_float(ne_cfg.get("imbalance_ratio_threshold"), 0.6)
        self._net_exposure_imbalance_threshold = max(0.0, min(1.0, self._net_exposure_imbalance_threshold))
        self._net_exposure_min_gross = safe_float(ne_cfg.get("min_gross_exposure"), 1.0)
        self._net_exposure_min_gross = max(0.0, self._net_exposure_min_gross)
        self._net_exposure_reduce_target = safe_float(ne_cfg.get("reduce_target"), 0.15)
        self._net_exposure_reduce_target = max(0.0, min(1.0, self._net_exposure_reduce_target))

        # ── 账户级杠杆守卫（account_leverage_guard）──
        # account_manager 已维护账户级总杠杆 _current_total_leverage = (已用保证金 + |未实现盈亏|)/权益，
        # 暴露 get_current_leverage()/get_account_summary()["current_leverage"]，且 _check_leverage_exposure
        # 在杠杆 ≥ max_total_leverage 时硬性减仓（事后被动、市场单有滑点）。但 AGI 编排层完全感知不到杠杆，
        # 只用 utilization = used_margin/equity 近似——忽略未实现浮亏对杠杆的放大（浮亏会抬高杠杆、更接近强平），
        # 且无「事前主动收敛 + 暂停进攻」软门控。本守卫补足：AGI 感知账户级杠杆，在杠杆 ≥ soft_threshold_ratio
        # （默认 0.8）× max_total_leverage 时，事前主动收敛最高敞口策略 + 暂停进攻，在硬减仓之前化解强平风险。
        al_cfg = agi_cfg.get("account_leverage_guard", {}) or {}
        self._account_leverage_enabled = bool(al_cfg.get("enabled", False))
        self._account_leverage_soft_ratio = safe_float(al_cfg.get("soft_threshold_ratio"), 0.8)
        self._account_leverage_soft_ratio = max(0.0, min(1.0, self._account_leverage_soft_ratio))
        self._account_leverage_reduce_target = safe_float(al_cfg.get("reduce_target"), 0.15)
        self._account_leverage_reduce_target = max(0.0, min(1.0, self._account_leverage_reduce_target))

        # ── 挂单保证金守卫（pending_margin_guard）──
        # account_manager 已维护挂单保证金 _pending_orders_margin（遍历 live/pending 挂单的
        # quantity×price/leverage 累加）并暴露 get_pending_margin()，且 _check_leverage_exposure
        # 把挂单保证金纳入杠杆计算（total_leverage_with_pending）。但 AGI 编排层感知不到挂单保证金——
        # _available_margin_ratio 用 used_margin（已成交持仓）计算，忽略挂单锁定的隐性资金，导致进攻门控
        # 高估可用资金、挂单成交后总敞口超预期。本守卫补足：AGI 感知挂单保证金占权益比例，占比过高
        # （大量未成交挂单锁定资金，如网格策略密集挂单）时事前收敛 + 暂停进攻，避免隐性敞口兑现时强平。
        pm_cfg = agi_cfg.get("pending_margin_guard", {}) or {}
        self._pending_margin_enabled = bool(pm_cfg.get("enabled", False))
        self._pending_margin_ratio_threshold = safe_float(pm_cfg.get("ratio_threshold"), 0.15)
        self._pending_margin_ratio_threshold = max(0.0, min(1.0, self._pending_margin_ratio_threshold))
        self._pending_margin_reduce_target = safe_float(pm_cfg.get("reduce_target"), 0.15)
        self._pending_margin_reduce_target = max(0.0, min(1.0, self._pending_margin_reduce_target))

        # ── 决策置信度自适应（confidence_adaptation）──
        # 使进攻置信门槛随账户状态联动：权益恶化（风险偏好低）→ 提高门槛（更保守），
        # 权益健康（风险偏好高）→ 保持基础门槛。避免账户回撤时仍低置信进攻。
        cfa_cfg = agi_cfg.get("confidence_adaptation", {}) or {}
        self._confidence_adaptation_enabled = bool(cfa_cfg.get("enabled", False))
        self._confidence_adapt_span = safe_float(cfa_cfg.get("adapt_span"), 0.15)
        self._confidence_adapt_span = max(0.0, min(1.0, self._confidence_adapt_span))

        # ── 成本预算自适应（cost_budget_adaptation）──
        # 滑点/资金费成本比例阈值随权益状态联动：权益健康 → 放宽预算（允许更高成本），
        # 权益恶化 → 收紧预算（更严格），与 confidence_adaptation 同构但方向相反。
        cba_cfg = agi_cfg.get("cost_budget_adaptation", {}) or {}
        self._cost_budget_adaptation_enabled = bool(cba_cfg.get("enabled", False))
        self._cost_budget_adapt_span = safe_float(cba_cfg.get("adapt_span"), 0.15)
        self._cost_budget_adapt_span = max(0.0, min(1.0, self._cost_budget_adapt_span))

        # ── 组合波动率预算守卫（volatility_budget_guard）──
        # AGI 感知策略「单笔盈亏标准差」（contribution_analyzer.volatility），对波动率超过
        # 预算（绝对阈值）的策略降杠杆收敛。与 volatility_trend_guard（波动率连续上升，趋势型）
        # 区分：本守卫看「绝对阈值」，构成波动率维度的「阈值 + 趋势」双层。
        vb_cfg = agi_cfg.get("volatility_budget_guard", {}) or {}
        self._volatility_budget_enabled = bool(vb_cfg.get("enabled", False))
        self._volatility_budget = safe_float(vb_cfg.get("volatility_budget"), 50.0)
        self._volatility_budget = max(0.0, self._volatility_budget)
        self._volatility_budget_min_trades = int(safe_float(vb_cfg.get("min_trades"), 10))
        self._volatility_budget_min_trades = max(1, self._volatility_budget_min_trades)
        self._volatility_budget_leverage = safe_float(vb_cfg.get("deterioration_leverage"), 1.0)
        self._volatility_budget_leverage = max(0.0, self._volatility_budget_leverage)

        # ── 决策记忆质量评估（decision_quality_guard）──
        # AGI 评估自身近期决策质量（盈利周期占比），质量过低时暂停进攻（避免在决策持续
        # 失误时继续进攻）。从 _decision_memory（每周期 total_pnl）计算盈利周期占比。
        dq_cfg = agi_cfg.get("decision_quality_guard", {}) or {}
        self._decision_quality_enabled = bool(dq_cfg.get("enabled", False))
        self._decision_quality_threshold = safe_float(dq_cfg.get("quality_threshold"), 0.4)
        self._decision_quality_threshold = max(0.0, min(1.0, self._decision_quality_threshold))
        self._decision_quality_min_samples = int(safe_float(dq_cfg.get("min_samples"), 3))
        self._decision_quality_min_samples = max(2, self._decision_quality_min_samples)
        # 死锁自愈：触发后完全暂停进攻会形成「不开仓→无盈利→质量分不变→继续暂停」
        # 死循环。recovery_after_cycles 个周期后允许最小试探开单（recovery_boost_scale ×
        # boost_step），给系统一个恢复出口。recovery_lock_count 记录持续触发周期数。
        self._decision_quality_recovery_cycles = int(
            safe_float(dq_cfg.get("recovery_after_cycles"), 8))
        self._decision_quality_recovery_cycles = max(1, self._decision_quality_recovery_cycles)
        self._decision_quality_recovery_scale = safe_float(
            dq_cfg.get("recovery_boost_scale"), 0.5)
        self._decision_quality_recovery_scale = max(
            0.0, min(1.0, self._decision_quality_recovery_scale))
        self._decision_quality_lock_count: int = 0
        # 市场状态条件化决策质量：按 regime（trend_bullish/trend_bearish/range_bound）分桶
        # 评估决策质量，避免不同市场状态的决策互相污染（趋势市的盈利掩盖震荡市的亏损，
        # 导致质量分偏高、guard 该触发却不触发）。当前 regime 样本不足 min_samples 时
        # 回退混合计算（保证 guard 仍能工作）。默认关闭，向后兼容。
        self._decision_quality_regime_conditioned = bool(
            dq_cfg.get("regime_conditioned", False))
        # 单周期盈亏增量（cycle_pnl delta）：_decision_memory 原存累计 total_pnl，
        # 跨周期不变时所有条目同值，导致质量分恒为 0（累计为负）或 1（累计为正），
        # 无法区分单周期决策好坏。改为记录本周期相对上周期的 closed-PnL 增量，
        # _compute_quality_score 据此判定「该周期决策是否盈利」。默认开启（修复语义），
        # 设为 false 回退到累计值（向后兼容旧行为）。
        self._decision_quality_use_cycle_pnl = bool(
            dq_cfg.get("use_cycle_pnl_delta", True))
        # 忽略无平仓周期（cycle_pnl==0）：震荡市长时间无平仓时，cycle_pnl 恒 0，若按
        # p>0 判定盈亏会把「无交易」误判为「非盈利」，导致质量分恒 0、guard 持续误触发
        # 暂停进攻（而实际是没交易，非决策错误）。开启后 cycle_pnl==0 的样本视为中性，
        # 从质量分计算中排除；全无平仓时返回 0.5（中性）。默认开启（修复误触发），
        # 设为 false 回退旧行为。
        self._decision_quality_ignore_zero_pnl = bool(
            dq_cfg.get("ignore_zero_cycle_pnl", True))
        self._last_decision_total_pnl: Optional[float] = None

        # ── 组合压力测试守卫（portfolio_stress_guard）──
        # AGI 对组合做尾部压力测试：假设所有敞口同时下跌 stress_scenario_pct（尾部场景），
        # 估算压力损失 = 毛敞口 × 场景跌幅；压力损失占权益比例 ≥ 预算时收敛敞口。
        ps_cfg = agi_cfg.get("portfolio_stress_guard", {}) or {}
        self._stress_guard_enabled = bool(ps_cfg.get("enabled", False))
        self._stress_scenario_pct = safe_float(ps_cfg.get("stress_scenario_pct"), 0.20)
        self._stress_scenario_pct = max(0.0, min(1.0, self._stress_scenario_pct))
        self._stress_loss_budget = safe_float(ps_cfg.get("stress_loss_budget"), 0.50)
        self._stress_loss_budget = max(0.0, min(1.0, self._stress_loss_budget))
        self._stress_reduce_target = safe_float(ps_cfg.get("reduce_target"), 0.15)
        self._stress_reduce_target = max(0.0, min(1.0, self._stress_reduce_target))
        # 压力场景多样化：除基础场景外增加「严重尾部场景」（更大幅度、更高预算），分级告警
        self._stress_severe_scenario_pct = safe_float(ps_cfg.get("severe_scenario_pct"), 0.40)
        self._stress_severe_scenario_pct = max(0.0, min(1.0, self._stress_severe_scenario_pct))
        self._stress_severe_loss_budget = safe_float(ps_cfg.get("severe_loss_budget"), 0.80)
        self._stress_severe_loss_budget = max(0.0, min(1.0, self._stress_severe_loss_budget))

        # ── 决策溯源与可观测性（decision_lineage）──
        # 每个周期产出 decision_id + 输入快照 + 决策依据(rationale) + 动作清单，
        # 追加到有界持久化历史，供 Dashboard 回溯每一条决策的完整依据链。
        dl_cfg = agi_cfg.get("decision_lineage", {}) or {}
        self._lineage_enabled = bool(dl_cfg.get("enabled", False))
        self._lineage_max_entries = int(safe_float(dl_cfg.get("max_entries"), 200))
        self._lineage_max_entries = max(10, min(2000, self._lineage_max_entries))
        self._lineage_path = dl_cfg.get("path") or os.path.join("data", "agi_decision_lineage.json")

        # ── 自主进攻性资金分配（offensive_allocation）──
        # 趋势确认（趋势市 + 足够强度）且账户健康（回撤受控）时，主动提升
        # A/B 级健康策略的资金权重（进攻加仓），补足「只减仓/归集」的进攻缺口。
        oa_cfg = agi_cfg.get("offensive_allocation", {}) or {}
        self._offensive_enabled = bool(oa_cfg.get("enabled", False))
        self._offensive_min_regime_strength = safe_float(oa_cfg.get("min_regime_strength"), 0.6)
        # 趋势强度上限（开单精准度）：strength 过高说明趋势已极端单边运行，此时追涨杀跌
        # 风险大（追高做多/追低做空）。与 min_regime_strength 构成「够强但不极端」的
        # 健康开单区间，对应用户「只趋势确认后开单、不追涨杀跌」诉求。默认 1.0 = 不设上限
        # （向后兼容），实盘设略低于 1.0 抑制趋势末端追涨杀跌。
        self._offensive_max_regime_strength = safe_float(oa_cfg.get("max_regime_strength"), 1.0)
        self._offensive_max_regime_strength = max(0.0, min(1.0, self._offensive_max_regime_strength))
        self._offensive_max_drawdown_pct = safe_float(oa_cfg.get("max_drawdown_pct"), 0.05)
        self._offensive_boost_step = safe_float(oa_cfg.get("boost_step"), 0.05)
        self._offensive_max_target = safe_float(oa_cfg.get("max_target"), 0.4)
        self._offensive_boost_step = max(0.0, self._offensive_boost_step)
        self._offensive_max_target = max(0.0, min(1.0, self._offensive_max_target))
        # 信号质量门槛强化：进攻开单不仅要趋势够强（strength），还要够确定（confidence）。
        self._offensive_min_regime_confidence = safe_float(oa_cfg.get("min_regime_confidence"), 0.0)
        self._offensive_min_regime_confidence = max(0.0, min(1.0, self._offensive_min_regime_confidence))
        # 信号共振过滤：进攻候选不仅健康度 A/B，还需趋势不衰退 + 累计盈利为正（多信号共振）
        self._signal_resonance_enabled = bool(oa_cfg.get("signal_resonance", False))
        # 开单时机校准：趋势确认周期——市场状态需持续 ≥ trend_confirmation_cycles 个周期才开单
        self._trend_confirmation_cycles = int(safe_float(oa_cfg.get("trend_confirmation_cycles"), 0))
        self._trend_confirmation_cycles = max(0, self._trend_confirmation_cycles)
        # 状态机：当前市场状态 + 已持续周期数（供趋势确认判断）
        # 注：独立变量名，避免与既有 regime_shift 检测使用的 _last_regime 冲突
        self._trend_confirmed_regime: Optional[str] = None
        self._trend_confirmed_streak: int = 0
        # 进攻观察期冷却：同一策略至少间隔 min_interval_cycles 个周期才能再次加仓，
        # 防止 AGI 连续追高加仓（对应用户「消除追涨杀跌」诉求）。
        self._offensive_min_interval_cycles = int(safe_float(oa_cfg.get("min_interval_cycles"), 5))
        self._offensive_min_interval_cycles = max(1, min(1000, self._offensive_min_interval_cycles))
        # 震荡市进攻（增强开单能力）：range_bound 下为震荡策略（grid 高抛低吸）生成进攻机会，
        # 补足「仅趋势市进攻」的开单缺口——震荡市是网格/震荡收割策略的主战场，低买高卖
        # 符合「低位开多、高位开空」原则，不追涨杀跌。
        self._offensive_range_bound_enabled = bool(oa_cfg.get("range_bound_enabled", False))
        self._offensive_range_bound_min_strength = safe_float(oa_cfg.get("range_bound_min_strength"), 0.3)
        rb_strats = oa_cfg.get("range_bound_strategies", ["grid", "oscillation_harvest"])
        if isinstance(rb_strats, str):
            rb_strats = [s.strip() for s in rb_strats.split(",") if s.strip()]
        self._offensive_range_bound_strategies = set(rb_strats)
        # 震荡市进攻参数独立化：震荡市「快进快出」节奏与趋势市「确认后持续加仓」不同，
        # 需更小幅试探（小 boost_step）、更低封顶（低 max_target）、更短冷却（机会转瞬即逝）。
        # 缺省回退到趋势市参数（向后兼容）。
        self._offensive_range_bound_boost_step = safe_float(
            oa_cfg.get("range_bound_boost_step"), self._offensive_boost_step)
        self._offensive_range_bound_max_target = safe_float(
            oa_cfg.get("range_bound_max_target"), self._offensive_max_target)
        self._offensive_range_bound_min_interval_cycles = int(safe_float(
            oa_cfg.get("range_bound_min_interval_cycles"), self._offensive_min_interval_cycles))
        self._offensive_range_bound_boost_step = max(0.0, self._offensive_range_bound_boost_step)
        self._offensive_range_bound_max_target = max(0.0, min(1.0, self._offensive_range_bound_max_target))
        self._offensive_range_bound_min_interval_cycles = max(1, min(1000, self._offensive_range_bound_min_interval_cycles))
        # 恢复期渐进进攻（recovery_offense）：账户从 EMERGENCY 恢复到 RECOVERY 后，默认 AGI 完全
        # 禁止进攻（mode not in decline/recovery/emergency）。本开关让 AGI 在 RECOVERY 恢复期
        # 利用 equity_monitor 的 position_multiplier（渐进加仓乘数）做谨慎渐进进攻——乘数从
        # floor 按 bar 回升到 ceiling，进攻力度随恢复进度逐步放大，而非「恢复期零进攻 →
        # NORMAL 满额进攻」的跳变，对应「稳步增长」诉求。
        self._recovery_offense_enabled = bool(oa_cfg.get("recovery_offense_enabled", False))
        self._recovery_offense_min_multiplier = max(0.0, safe_float(
            oa_cfg.get("recovery_offense_min_multiplier"), 0.1))
        # 进攻火力集中：每周期最多同时对 max_offensive_strategies 个策略进攻加仓，
        # 按 health_score 降序取最优——避免资金撒网过宽（单策略加仓不足）且交易笔数过多
        # （手续费侵蚀）。集中优势火力到质量最高的策略，呼应「消除频繁交易消耗资金」。
        self._offensive_max_offensive_strategies = int(safe_float(
            oa_cfg.get("max_offensive_strategies"), 3))
        self._offensive_max_offensive_strategies = max(1, min(100, self._offensive_max_offensive_strategies))
        # 震荡市火力集中独立上限：震荡市「快进快出、机会转瞬即逝」更应集中到更少最优策略
        # （火力更集中），趋势市「确认后持续加仓」可略分散。缺省回退到 max_offensive_strategies
        # （向后兼容），与震荡市其他参数（boost_step/max_target/min_interval_cycles）分轨一致。
        self._offensive_range_bound_max_offensive_strategies = int(safe_float(
            oa_cfg.get("range_bound_max_offensive_strategies"), self._offensive_max_offensive_strategies))
        self._offensive_range_bound_max_offensive_strategies = max(
            1, min(100, self._offensive_range_bound_max_offensive_strategies))
        # 进攻止盈回落：上次进攻加仓后策略累计盈亏超过阈值 → 落袋为安，减仓回落，
        # 闭环「进攻加仓→盈利达标→自动回落」，避免进攻敞口永远不收缩、利润无法锁定。
        pt_cfg = oa_cfg.get("profit_take", {}) or {}
        self._offensive_profit_take_enabled = bool(pt_cfg.get("enabled", False))
        self._offensive_profit_take_pnl_threshold = safe_float(
            pt_cfg.get("pnl_threshold"), 5.0)
        self._offensive_profit_take_revert_step = safe_float(
            pt_cfg.get("revert_step"), self._offensive_boost_step)
        self._offensive_profit_take_pnl_threshold = max(0.0, self._offensive_profit_take_pnl_threshold)
        self._offensive_profit_take_revert_step = max(0.0, self._offensive_profit_take_revert_step)
        # 止盈阶梯式减仓：盈利远超阈值时按超阈值幅度放大减仓（多赚多减、锁定更多利润），
        # 封顶 revert_max_multiplier（默认 1.0 = 不缩放，向后兼容）。
        self._offensive_profit_take_revert_max_mult = max(1.0, safe_float(
            pt_cfg.get("revert_max_multiplier"), 1.0))
        # 止盈后冷却期：止盈后 cooldown_cycles 内禁止重新进攻该策略，
        # 与止损后冷却对称——「刚止盈就再进」同样是追涨、频繁交易消耗资金。
        self._offensive_profit_take_cooldown_cycles = int(safe_float(
            pt_cfg.get("cooldown_cycles"), 0))
        self._offensive_profit_take_cooldown_cycles = max(0, min(1000, self._offensive_profit_take_cooldown_cycles))
        # 进攻止损回落：上次进攻加仓后策略累计亏损超过阈值 → 止损减仓回落，
        # 与止盈回落对称，闭环进攻风控——避免亏损敞口持续放大、越亏越多。
        sl_cfg = oa_cfg.get("stop_loss", {}) or {}
        self._offensive_stop_loss_enabled = bool(sl_cfg.get("enabled", False))
        self._offensive_stop_loss_pnl_threshold = safe_float(
            sl_cfg.get("pnl_threshold"), 5.0)
        self._offensive_stop_loss_revert_step = safe_float(
            sl_cfg.get("revert_step"), self._offensive_boost_step)
        self._offensive_stop_loss_pnl_threshold = max(0.0, self._offensive_stop_loss_pnl_threshold)
        self._offensive_stop_loss_revert_step = max(0.0, self._offensive_stop_loss_revert_step)
        # 止损阶梯式减仓：亏损远超阈值时按超阈值幅度放大减仓（多亏多减、快速止血），
        # 封顶 revert_max_multiplier（默认 1.0 = 不缩放，向后兼容）。
        self._offensive_stop_loss_revert_max_mult = max(1.0, safe_float(
            sl_cfg.get("revert_max_multiplier"), 1.0))
        # 止损后冷却期：止损后 cooldown_cycles 内禁止重新进攻该策略，
        # 避免「刚亏就加」的频繁交易消耗资金。
        self._offensive_stop_loss_cooldown_cycles = int(safe_float(
            sl_cfg.get("cooldown_cycles"), 10))
        self._offensive_stop_loss_cooldown_cycles = max(1, min(1000, self._offensive_stop_loss_cooldown_cycles))
        # 进攻力度动量缩放：按 regime strength 归一化后映射到 [min_mult, max_mult]，
        # 强信号加大仓、弱信号小仓试——精准匹配信号质量，避免「一刀切」固定加仓。
        self._offensive_momentum_scaling_enabled = bool(oa_cfg.get("momentum_scaling_enabled", False))
        self._offensive_momentum_min_mult = safe_float(
            oa_cfg.get("momentum_scaling_min_multiplier"), 0.5)
        self._offensive_momentum_max_mult = safe_float(
            oa_cfg.get("momentum_scaling_max_multiplier"), 1.5)
        self._offensive_momentum_min_mult = max(0.0, self._offensive_momentum_min_mult)
        self._offensive_momentum_max_mult = max(self._offensive_momentum_min_mult, self._offensive_momentum_max_mult)
        # 账户级总进攻敞口上限：所有策略目标权重之和（总部署比例）不得突破该上限。
        # 单策略封顶（max_target）只约束「单点火力」，火力集中会同时加仓多策略，
        # 若无账户级总敞口封顶，多策略叠加仍可能把组合推到超配甚至杠杆状态——
        # 「进攻」也要有账户级总敞口封顶，呼应「稳健资金增长、不冒进」诉求。
        self._offensive_max_total_allocation = safe_float(oa_cfg.get("max_total_allocation"), 1.0)
        self._offensive_max_total_allocation = max(0.0, min(10.0, self._offensive_max_total_allocation))
        # 进攻归因周期过期（TTL）：entry_pnl 基线超过 attribution_ttl_cycles 未触发止盈/止损
        # 则自动过期清除，避免陈旧基线长期阻塞重新进攻（默认 0 = 永不过期，向后兼容）。
        self._offensive_attribution_ttl_cycles = int(safe_float(
            oa_cfg.get("attribution_ttl_cycles"), 0))
        self._offensive_attribution_ttl_cycles = max(0, min(10000, self._offensive_attribution_ttl_cycles))
        # 连续进攻次数上限：同一策略「无了结」（未经历止盈/止损/防守减仓）连续进攻加仓
        # 达到该上限后暂停，防止无了结地一路追高加仓（对应「消除追涨杀跌」）。
        # 默认 0 = 不限，向后兼容。
        self._offensive_max_consecutive = int(safe_float(
            oa_cfg.get("max_consecutive_offenses"), 0))
        self._offensive_max_consecutive = max(0, min(1000, self._offensive_max_consecutive))
        # 进攻与账户可用保证金联动：仅当账户可用保证金占总资金比例 ≥ min_available_margin_ratio
        # 时才进攻加仓，避免保证金不足时强行加仓被拒/被动减仓。默认 disabled，向后兼容。
        self._offensive_margin_check_enabled = bool(oa_cfg.get("available_margin_check_enabled", False))
        self._offensive_min_available_margin_ratio = safe_float(
            oa_cfg.get("min_available_margin_ratio"), 0.1)
        self._offensive_min_available_margin_ratio = max(0.0, self._offensive_min_available_margin_ratio)
        # 进攻最小增幅保护：动量缩放/自适应风险缩放后单次加仓幅度若低于 min_boost_delta 则跳过，
        # 避免微调（如 0.5% 仓位）徒耗手续费，且避免为「会被 cost_guard 过滤的微动作」污染
        # 归因基线/观察期冷却/连续进攻计数状态。默认 0 = 无下限，向后兼容。
        self._offensive_min_boost_delta = safe_float(oa_cfg.get("min_boost_delta"), 0.0)
        self._offensive_min_boost_delta = max(0.0, self._offensive_min_boost_delta)
        # 进攻强化学习反馈循环：止盈→正反馈（+1）、止损→负反馈（-1），EMA 平滑为
        # 滚动分数 [-1,1]，下次对该策略进攻时按分数缩放 boost_step——对的决策加大
        # 步长（奖励）、错的决策缩小步长（惩罚），形成「进攻→了结→学习→调整」闭环。
        # 与 attribution（控方向对错，binary gate）互补：attribution 是硬门控（转差就不加），
        # rl_feedback 是软调节（加多少按历史胜率调整）。默认关闭，向后兼容。
        self._offensive_rl_feedback_enabled = bool(oa_cfg.get("rl_feedback_enabled", False))
        self._offensive_rl_adaptation_span = safe_float(
            oa_cfg.get("rl_adaptation_span"), 0.1)
        self._offensive_rl_adaptation_span = max(0.0, min(1.0, self._offensive_rl_adaptation_span))
        self._offensive_rl_min_mult = safe_float(oa_cfg.get("rl_min_multiplier"), 0.5)
        self._offensive_rl_max_mult = safe_float(oa_cfg.get("rl_max_multiplier"), 1.5)
        self._offensive_rl_min_mult = max(0.0, self._offensive_rl_min_mult)
        self._offensive_rl_max_mult = max(self._offensive_rl_min_mult, self._offensive_rl_max_mult)
        self._offensive_rl_ema_alpha = safe_float(oa_cfg.get("rl_ema_alpha"), 0.3)
        self._offensive_rl_ema_alpha = max(0.01, min(1.0, self._offensive_rl_ema_alpha))
        # 进攻反馈暖启动：策略尚无进攻反馈历史时，用账户级 decision_quality 作为先验
        # 初始化其反馈倍率——避免所有新策略恒以 mult=1.0 起步（RL 循环需首次进攻了结
        # 才能学习，暖启动让循环从首个进攻就有方向）。映射：dq∈[0,1] → score∈[-1,1]。
        # 默认开启，可关闭回退到无历史=1.0。
        self._offensive_rl_warm_start = bool(oa_cfg.get("rl_warm_start_enabled", True))

        # ── 自主策略生命周期管理（strategy_lifecycle）──
        # 诊断到「永久冻结（试探耗尽）/ 休眠」策略时，主动生成 strategy_pause 动作，
        # 由 scheduler 落地（调 strategy_manager.pause_strategy 停开新仓），完成
        # 「诊断→执行」闭环，避免持续亏损/闲置策略继续占用算力与资金。
        sl_cfg = agi_cfg.get("strategy_lifecycle", {}) or {}
        self._strategy_lifecycle_enabled = bool(sl_cfg.get("enabled", False))
        self._pause_permanently_frozen = bool(sl_cfg.get("pause_permanently_frozen", True))
        self._pause_dormant = bool(sl_cfg.get("pause_dormant", True))
        # 自主恢复：AGI 暂停过的策略在健康度改善后自动 resume（补足 pause 的对称闭环）
        self._resume_recovered = bool(sl_cfg.get("resume_recovered", True))

        # ── 自适应风险偏好（adaptive_risk）──
        # 基于近期权益轨迹维护一个 [0,1] 的风险偏好：权益上升 → 偏好升高（进攻），
        # 权益下跌 → 偏好降低（收敛）。用于动态抑制进攻性加仓，使 AGI 从自身
        # 决策结果中学习，避免在账户回撤时继续进攻。
        ar_cfg = agi_cfg.get("adaptive_risk", {}) or {}
        self._adaptive_risk_enabled = bool(ar_cfg.get("enabled", False))
        self._adaptive_risk_window = int(safe_float(ar_cfg.get("window"), 5))
        self._adaptive_risk_window = max(3, min(50, self._adaptive_risk_window))
        self._adaptive_risk_min_appetite = safe_float(ar_cfg.get("min_appetite"), 0.2)
        self._adaptive_risk_min_boost_ratio = safe_float(ar_cfg.get("min_boost_ratio"), 0.3)
        self._adaptive_risk_min_appetite = max(0.0, min(1.0, self._adaptive_risk_min_appetite))
        self._adaptive_risk_min_boost_ratio = max(0.0, min(1.0, self._adaptive_risk_min_boost_ratio))
        # 回撤感知风险偏好：仅看权益斜率会忽略「回撤深度」——深度回撤未恢复时即使
        # 近期微升，斜率型偏好仍偏高，导致在回撤未恢复时过快加仓。开启后按当前回撤
        # 深度叠加乘法惩罚：drawdown_scale（默认 0.2）处回撤 20% → 惩罚归零（完全收敛）。
        self._adaptive_risk_drawdown_aware = bool(ar_cfg.get("drawdown_aware", True))
        self._adaptive_risk_drawdown_scale = safe_float(ar_cfg.get("drawdown_scale"), 0.2)
        self._adaptive_risk_drawdown_scale = max(
            0.01, min(1.0, self._adaptive_risk_drawdown_scale))

        # ── 动作冲突消解与优先级仲裁（action_reconciliation）──
        # 多个动作生成器（进攻加仓 / 风控减仓 / 停开新仓 / 降杠杆）在同一周期可能对
        # 同一策略产出相互矛盾的动作。按「风险收敛优先」原则仲裁：降风险动作拥有
        # 最高优先级，凡被降风险的策略，其进攻动作一律抑制；同策略 reallocate 去重。
        rc_cfg = agi_cfg.get("action_reconciliation", {}) or {}
        self._reconciliation_enabled = bool(rc_cfg.get("enabled", False))
        self._min_confidence = safe_float(rc_cfg.get("min_confidence"), 0.5)
        self._min_confidence = max(0.0, min(1.0, self._min_confidence))

        # ── 多时间尺度分层自治（timescale）──
        # 战略性决策（资金重分配/进攻加仓/目标规划/策略生命周期）需要长观察窗口，
        # 战术性决策（风险响应/参数自适应/收益落袋/归集）需要快速响应。二者在同一
        # cooldown 周期混跑会引入噪声震荡，故按快/慢尺度分层：慢尺度动作生成器仅在
        # 每 slow_cycle_interval 个周期执行一次。未启用时保持向后兼容（每周期全执行）。
        ts_cfg = agi_cfg.get("timescale", {}) or {}
        self._timescale_enabled = bool(ts_cfg.get("enabled", False))
        self._slow_cycle_interval = int(safe_float(ts_cfg.get("slow_cycle_interval"), 10))
        self._slow_cycle_interval = max(1, min(1000, self._slow_cycle_interval))

        # ── 组合级分散化响应（diversification）──
        # 诊断到「集中度过高（HHI 跨阈值）」时，主动降低最高权重策略的权重（分散化），
        # 补足「组合级风险诊断→响应」的缺口（此前 high_concentration 只告警不动作）。
        dv_cfg = agi_cfg.get("diversification", {}) or {}
        self._diversification_enabled = bool(dv_cfg.get("enabled", False))
        self._diversification_reduce_target = safe_float(dv_cfg.get("reduce_target"), 0.2)
        self._diversification_reduce_target = max(0.0, min(1.0, self._diversification_reduce_target))

        # ── 组合级相关性感知与响应（correlation）──
        # 感知 StrategyCorrelationAnalyzer 的策略间相关性（平均/最大 pair 相关性、
        # 有效 N 侵蚀），诊断到高相关/有效 N 侵蚀时降低最高权重策略权重，
        # 补足「组合级风险」的第二个维度（集中度之外的相关性风险）。
        corr_cfg = agi_cfg.get("correlation", {}) or {}
        self._correlation_enabled = bool(corr_cfg.get("enabled", False))
        self._correlation_high_threshold = safe_float(corr_cfg.get("high_corr_threshold"), 0.7)
        self._correlation_high_threshold = max(0.0, min(1.0, self._correlation_high_threshold))
        self._correlation_reduce_target = safe_float(corr_cfg.get("reduce_target"), 0.2)
        self._correlation_reduce_target = max(0.0, min(1.0, self._correlation_reduce_target))

        # ── 组合级资金效率响应（portfolio_efficiency_guard）──
        # 诊断到「资金效率偏低（low_capital_efficiency，efficiency_score 跨阈值）」时，
        # 主动降低最高权重策略的权重（收敛低效资金暴露），补足「组合级资金效率诊断→响应」
        # 缺口（此前 low_capital_efficiency 只告警不动作）。与 diversification（集中度）、
        # correlation（相关性）、synergy（协同度）、tail_risk（尾部风险）并列，是组合级
        # 「资金效率」维度的响应——效率低 = 资金未被高效利用（费率/资本回报/风险调整后
        # 收益综合偏弱），收敛最大敞口以释放低效占用资金。阈值型守卫，无跨周期状态。
        pe_cfg = agi_cfg.get("portfolio_efficiency_guard", {}) or {}
        self._portfolio_efficiency_enabled = bool(pe_cfg.get("enabled", False))
        self._portfolio_efficiency_reduce_target = safe_float(pe_cfg.get("reduce_target"), 0.15)
        self._portfolio_efficiency_reduce_target = max(0.0, min(1.0, self._portfolio_efficiency_reduce_target))

        # ── 组合级资金效率趋势外推（portfolio_efficiency_trend_guard）──
        # 与 portfolio_efficiency_guard（阈值型：efficiency_score 跌破阈值才收敛，事后）区分：
        # 本守卫追踪 efficiency_score 跨周期时序，检测连续 window 周期严格下降——资金效率
        # 持续劣化（费率/资本回报/风险调整后收益综合走弱）的事前预警，尚未跌破阈值即收敛。
        # 与 portfolio_health_trend（组合健康度下降，看收益/风险质量）区分：本守卫看「资金
        # 利用效率」下降（资金未高效利用），二者正交。至此组合级资金效率维度形成「阈值 +
        # 趋势」双层。仅在成交数 ≥ min_sample 时追踪（无成交时 efficiency 恒 0 稳定不下降）。
        pet_cfg = agi_cfg.get("portfolio_efficiency_trend_guard", {}) or {}
        self._portfolio_efficiency_trend_enabled = bool(pet_cfg.get("enabled", False))
        self._portfolio_efficiency_trend_window = int(safe_float(pet_cfg.get("window"), 3))
        self._portfolio_efficiency_trend_window = max(2, min(20, self._portfolio_efficiency_trend_window))
        self._portfolio_efficiency_trend_min_sample = int(safe_float(pet_cfg.get("min_sample"), 10))
        self._portfolio_efficiency_trend_min_sample = max(1, min(10000, self._portfolio_efficiency_trend_min_sample))
        self._portfolio_efficiency_trend_reduce_target = safe_float(pet_cfg.get("reduce_target"), 0.15)
        self._portfolio_efficiency_trend_reduce_target = max(0.0, min(1.0, self._portfolio_efficiency_trend_reduce_target))

        # ── 成本意识调仓门控（cost_guard）──
        # 对 AGI 生成的所有 reallocate 动作统一施加「最小调仓幅度」过滤：
        # 目标权重与当前权重差异小于 min_delta 时丢弃，避免微小调仓被手续费/滑点
        # 吞噬（对应用户「消除频繁交易消耗资金」诉求）。无法读取当前权重时放行
        # （向后兼容，不误杀）。
        cg_cfg = agi_cfg.get("cost_guard", {}) or {}
        self._cost_guard_enabled = bool(cg_cfg.get("enabled", False))
        self._cost_guard_min_delta = safe_float(cg_cfg.get("min_delta"), 0.02)
        self._cost_guard_min_delta = max(0.0, min(1.0, self._cost_guard_min_delta))

        # ── 策略健康度趋势外推（health_trend_guard）──
        # 从「反应式」升级到「预测式」：追踪各策略 health_score 跨周期时序，
        # 检测到连续 window 个周期健康度下降时（即使当前仍在 B/C 级），提前降杠杆，
        # 在策略真正掉到 F 之前收敛敞口（事前风控）。
        htg_cfg = agi_cfg.get("health_trend_guard", {}) or {}
        self._health_trend_enabled = bool(htg_cfg.get("enabled", False))
        self._health_trend_window = int(safe_float(htg_cfg.get("window"), 3))
        self._health_trend_window = max(2, min(20, self._health_trend_window))
        self._health_trend_leverage = safe_float(htg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略胜率趋势外推（win_rate_trend_guard）──
        # health_trend_guard 的兄弟守卫：追踪各策略 win_rate 跨周期时序，
        # 检测到连续 window 个周期胜率下降时（即使当前 PnL 仍为正、健康度仍 A/B），
        # 提前降杠杆——胜率下降是策略衰退的先行指标，在 PnL 转负之前收敛敞口。
        # 与 health_trend_guard 区分：后者看综合健康度（含 PnL/回撤/Sharpe），
        # 本方法单看胜率趋势——一个策略可能「赢少输多但单笔赢大于输」导致 PnL 正
        # 但胜率持续下滑，此时综合健康度尚未恶化，但策略实际已在劣化。
        wrtg_cfg = agi_cfg.get("win_rate_trend_guard", {}) or {}
        self._win_rate_trend_enabled = bool(wrtg_cfg.get("enabled", False))
        self._win_rate_trend_window = int(safe_float(wrtg_cfg.get("window"), 3))
        self._win_rate_trend_window = max(2, min(20, self._win_rate_trend_window))
        self._win_rate_trend_min_trades = int(safe_float(wrtg_cfg.get("min_trades"), 10))
        self._win_rate_trend_min_trades = max(1, min(1000, self._win_rate_trend_min_trades))
        self._win_rate_trend_leverage = safe_float(wrtg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略胜率绝对值守卫（win_rate_guard）──
        # 与 win_rate_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」——策略胜率
        # 已低于 min_win_rate（默认0.35）但尚未连续下降（可能长期在低位徘徊）时趋势守卫不触发，
        # 但策略实际在「长期低胜率」状态（赢少输多，策略有效性存疑）。构成胜率维度「阈值+趋势」
        # 双层。仅在交易笔数达 min_trades 时评估（样本不足的胜率噪声大）。阈值型守卫（非趋势型），
        # 从当周期 contribution 读取 win_rate（已在 snapshot_to_dict 透出），无跨周期状态。
        wrg_cfg = agi_cfg.get("win_rate_guard", {}) or {}
        self._win_rate_guard_enabled = bool(wrg_cfg.get("enabled", False))
        self._win_rate_guard_min_trades = int(safe_float(wrg_cfg.get("min_trades"), 10))
        self._win_rate_guard_min_trades = max(1, self._win_rate_guard_min_trades)
        self._win_rate_guard_min_win_rate = safe_float(wrg_cfg.get("min_win_rate"), 0.35)
        self._win_rate_guard_min_win_rate = max(0.0, min(1.0, self._win_rate_guard_min_win_rate))
        self._win_rate_guard_leverage = safe_float(wrg_cfg.get("deterioration_leverage"), 1.0)
        self._win_rate_guard_leverage = max(0.0, self._win_rate_guard_leverage)

        # ── 策略夏普比率趋势外推（sharpe_trend_guard）──
        # health_trend_guard / win_rate_trend_guard 的第三个兄弟守卫：
        # 追踪各策略 sharpe_ratio 跨周期时序，连续下降 → 提前降杠杆。
        # 夏普度量「风险调整后收益」——夏普下降意味着策略在承担更多风险获取同等收益，
        # 这是 PnL 转负前的先行指标，且独立于胜率（胜率高但波动大 → 夏普低）。
        # 与 win_rate_trend 区分：胜率看「赢的频率」，夏普看「单位风险收益」——
        # 一个策略可能胜率高（0.7）但每笔赢小输大导致夏普低；或胜率低（0.3）但单笔赢大
        # 导致夏普高。两者各捕获一种衰退模式。
        stg_cfg = agi_cfg.get("sharpe_trend_guard", {}) or {}
        self._sharpe_trend_enabled = bool(stg_cfg.get("enabled", False))
        self._sharpe_trend_window = int(safe_float(stg_cfg.get("window"), 3))
        self._sharpe_trend_window = max(2, min(20, self._sharpe_trend_window))
        self._sharpe_trend_min_trades = int(safe_float(stg_cfg.get("min_trades"), 10))
        self._sharpe_trend_min_trades = max(1, min(1000, self._sharpe_trend_min_trades))
        self._sharpe_trend_leverage = safe_float(stg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级连续亏损收敛（consecutive_losses_guard）──
        # 填补「策略健康」与「冻结（≥5连续亏损）」之间的 AGI 响应空白：
        # 此前 consecutive_losses≥5 触发 FROZEN 是策略管理器级硬冻结，
        # AGI 应在 loss_threshold（默认3）时提前收敛——降杠杆+抑制进攻，
        # 在硬冻结前先收敛敞口。阈值型守卫（非趋势型），与三子趋势守卫互补。
        clg_cfg = agi_cfg.get("consecutive_losses_guard", {}) or {}
        self._consecutive_losses_enabled = bool(clg_cfg.get("enabled", False))
        self._consecutive_losses_threshold = int(safe_float(clg_cfg.get("loss_threshold"), 3))
        self._consecutive_losses_threshold = max(1, min(100, self._consecutive_losses_threshold))
        self._consecutive_losses_leverage = safe_float(clg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级止损频率收敛（stop_loss_frequency_guard）──
        # 止损率（stop_loss_count/total_trades）过高说明策略入场时机差——频繁触发止损往往
        # 是「追高开多/追低开空」的直接结果，直接对应「消除追涨杀跌、趋势确认后才开仓」诉求。
        # 与 consecutive_losses_guard（连续亏损笔数，看连败）区分：本守卫看「止损占比」——
        # 即使亏损不连续，只要止损交易占比过高，也说明开仓位置不佳。与 win_rate_trend_guard
        # （胜率下降，含所有亏损单）区分：本守卫单看「触及止损」的亏损（入场时机最差的一类）。
        # 阈值型守卫（非趋势型），从当周期 contribution 读取 stop_loss_count，不持久化。
        slfg_cfg = agi_cfg.get("stop_loss_frequency_guard", {}) or {}
        self._stop_loss_frequency_enabled = bool(slfg_cfg.get("enabled", False))
        self._stop_loss_frequency_ratio_threshold = safe_float(slfg_cfg.get("loss_ratio_threshold"), 0.5)
        self._stop_loss_frequency_ratio_threshold = max(0.0, min(1.0, self._stop_loss_frequency_ratio_threshold))
        self._stop_loss_frequency_min_trades = int(safe_float(slfg_cfg.get("min_trades"), 10))
        self._stop_loss_frequency_min_trades = max(1, min(10000, self._stop_loss_frequency_min_trades))
        self._stop_loss_frequency_leverage = safe_float(slfg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略盈亏比趋势外推（profit_factor_trend_guard）──
        # 事前风控趋势守卫的第四子：追踪各策略 profit_factor 跨周期时序，连续下降 → 提前降杠杆。
        # profit_factor（平均盈利/平均亏损）直接度量「赢的幅度 vs 输的幅度」，独立于胜率频率
        # 与夏普（风险调整收益）——一个策略可能胜率高（0.7）但每笔赢小输大导致 profit_factor<1
        # 亏损；或胜率低（0.3）但单笔赢大导致 profit_factor 很高。三者各捕获一种衰退模式：
        # win_rate 看频率、sharpe 看单位风险收益、profit_factor 看盈亏幅度比。
        pftg_cfg = agi_cfg.get("profit_factor_trend_guard", {}) or {}
        self._profit_factor_trend_enabled = bool(pftg_cfg.get("enabled", False))
        self._profit_factor_trend_window = int(safe_float(pftg_cfg.get("window"), 3))
        self._profit_factor_trend_window = max(2, min(20, self._profit_factor_trend_window))
        self._profit_factor_trend_min_trades = int(safe_float(pftg_cfg.get("min_trades"), 10))
        self._profit_factor_trend_min_trades = max(1, min(1000, self._profit_factor_trend_min_trades))
        self._profit_factor_trend_leverage = safe_float(pftg_cfg.get("deterioration_leverage"), 1.0)

        # ── 盈亏比绝对阈值守卫（profit_factor_guard）──
        # AGI 感知策略「盈亏比」（contribution_analyzer.profit_factor = 总盈利/总亏损），
        # 对「盈亏比 < 1」（总亏损 > 总盈利，已实现净亏损）的策略降杠杆收敛。
        # 与 profit_factor_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」——一个策略
        # 盈亏比长期 <1 但未连续下降（平坦低位）时趋势守卫不触发，但策略实际在「总亏损超过总盈利」
        # （已实现交易净亏，频繁交易消耗资金的直接体现）。构成盈亏比维度「阈值 + 趋势」双层
        # （与 volatility/资本回报/手续费/夏普等指标的阈值+趋势双层对称）。阈值型守卫，无跨周期状态。
        pfag_cfg = agi_cfg.get("profit_factor_guard", {}) or {}
        self._profit_factor_guard_enabled = bool(pfag_cfg.get("enabled", False))
        self._profit_factor_guard_min_trades = int(safe_float(pfag_cfg.get("min_trades"), 10))
        self._profit_factor_guard_min_trades = max(1, self._profit_factor_guard_min_trades)
        self._profit_factor_guard_min_ratio = safe_float(pfag_cfg.get("min_profit_factor"), 1.0)
        self._profit_factor_guard_leverage = safe_float(pfag_cfg.get("deterioration_leverage"), 1.0)
        self._profit_factor_guard_leverage = max(0.0, self._profit_factor_guard_leverage)

        # ── 策略最大回撤趋势外推（max_drawdown_trend_guard）──
        # 事前风控趋势守卫的第五子：追踪各策略 max_drawdown 跨周期时序，连续上升（加深）→ 提前降杠杆。
        # max_drawdown 是纯风险维度的独立信号——一个策略可能 PnL 正、胜率高、夏普高、盈亏比高
        # 但回撤在持续加深（承担更大风险获取收益），这是前四子守卫（收益维度）无法捕获的。
        # 前四子看收益质量（health 综合/win_rate 频率/sharpe 单位风险收益/profit_factor 盈亏幅度），
        # 本守卫看回撤深度轨迹，构成「收益质量 + 风险深度」的双维度事前风控体系。
        mdtg_cfg = agi_cfg.get("max_drawdown_trend_guard", {}) or {}
        self._max_drawdown_trend_enabled = bool(mdtg_cfg.get("enabled", False))
        self._max_drawdown_trend_window = int(safe_float(mdtg_cfg.get("window"), 3))
        self._max_drawdown_trend_window = max(2, min(20, self._max_drawdown_trend_window))
        self._max_drawdown_trend_min_trades = int(safe_float(mdtg_cfg.get("min_trades"), 10))
        self._max_drawdown_trend_min_trades = max(1, min(1000, self._max_drawdown_trend_min_trades))
        self._max_drawdown_trend_leverage = safe_float(mdtg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略最大回撤绝对值守卫（max_drawdown_guard）──
        # 与 max_drawdown_trend_guard（连续加深，趋势型）区分：本守卫看「绝对阈值」——策略最大
        # 回撤已超过 max_drawdown_threshold（默认0.2，即20%）但尚未连续加深（可能长期在高位
        # 徘徊未恢复）时趋势守卫不触发，但策略实际在「深度回撤」状态（风险敞口过大、抗风险能力弱）。
        # 构成回撤深度维度「阈值+趋势」双层。仅在交易笔数达 min_trades 时评估（样本不足的回撤
        # 统计不稳定）。阈值型守卫（非趋势型），从当周期 contribution 读取 max_drawdown（已在
        # snapshot_to_dict 透出），无跨周期状态。
        mdg_cfg = agi_cfg.get("max_drawdown_guard", {}) or {}
        self._max_drawdown_guard_enabled = bool(mdg_cfg.get("enabled", False))
        self._max_drawdown_guard_min_trades = int(safe_float(mdg_cfg.get("min_trades"), 10))
        self._max_drawdown_guard_min_trades = max(1, self._max_drawdown_guard_min_trades)
        self._max_drawdown_guard_threshold = safe_float(mdg_cfg.get("max_drawdown_threshold"), 0.2)
        self._max_drawdown_guard_threshold = max(0.0, min(1.0, self._max_drawdown_guard_threshold))
        self._max_drawdown_guard_leverage = safe_float(mdg_cfg.get("deterioration_leverage"), 1.0)
        self._max_drawdown_guard_leverage = max(0.0, self._max_drawdown_guard_leverage)

        # ── 策略回撤持续时间趋势外推（drawdown_duration_trend_guard）──
        # 与 max_drawdown_trend_guard（回撤「深度」连续加深）区分：本守卫追踪各策略
        # max_drawdown_duration_hours（回撤「持续时长」）跨周期时序，检测连续 window 周期
        # 严格上升——回撤持续时间拉长（恢复能力下降），资金被套牢。回撤深度与回撤时长正交：
        # 一个策略可能回撤不深但长时间无法收复前高（阴跌、慢恢复），这是 max_drawdown 无法
        # 捕获的独立「资金时间价值」维度。仅在交易笔数 ≥ min_trades 时追踪，避免样本不足
        # 噪声误触发。
        ddg_cfg = agi_cfg.get("drawdown_duration_trend_guard", {}) or {}
        self._drawdown_duration_trend_enabled = bool(ddg_cfg.get("enabled", False))
        self._drawdown_duration_trend_window = int(safe_float(ddg_cfg.get("window"), 3))
        self._drawdown_duration_trend_window = max(2, min(20, self._drawdown_duration_trend_window))
        self._drawdown_duration_trend_min_trades = int(safe_float(ddg_cfg.get("min_trades"), 10))
        self._drawdown_duration_trend_min_trades = max(1, min(1000, self._drawdown_duration_trend_min_trades))
        self._drawdown_duration_trend_leverage = safe_float(ddg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略回撤持续时间绝对阈值守卫（drawdown_duration_guard）──
        # AGI 感知策略「最长回撤持续时长」（contribution_analyzer.max_drawdown_duration_hours =
        # 权益从峰值回落到恢复峰值的最长时间跨度），对「回撤持续过久」（恢复能力差、资金被套牢）
        # 的策略降杠杆收敛。与 drawdown_duration_trend_guard（连续拉长，趋势型）区分：本守卫看
        # 「绝对阈值」——一个策略历史最长回撤持续 ≥ max_drawdown_hours（默认168h=7天）但未连续
        # 拉长时趋势守卫不触发，但策略实际已证明「恢复能力差」（资金时间价值差），应提前收敛。
        # 构成回撤持续维度「阈值 + 趋势」双层（与回撤深度 max_drawdown_trend + tail_risk 双层对称）。
        # 阈值型守卫（非趋势型），从当周期 contribution 读取，不持久化、不纳入 _DECAY_ALERT_TYPES。
        ddug_cfg = agi_cfg.get("drawdown_duration_guard", {}) or {}
        self._drawdown_duration_guard_enabled = bool(ddug_cfg.get("enabled", False))
        self._drawdown_duration_guard_min_trades = int(safe_float(ddug_cfg.get("min_trades"), 10))
        self._drawdown_duration_guard_min_trades = max(1, self._drawdown_duration_guard_min_trades)
        self._drawdown_duration_guard_max_hours = safe_float(ddug_cfg.get("max_drawdown_hours"), 168.0)
        self._drawdown_duration_guard_leverage = safe_float(ddug_cfg.get("deterioration_leverage"), 1.0)
        self._drawdown_duration_guard_leverage = max(0.0, self._drawdown_duration_guard_leverage)

        # ── 资本回报率趋势外推（capital_return_trend_guard，第六子）──
        # 追踪各策略 pnl_per_capital_pct（资本回报率 = PnL / 配置资本）跨周期时序，连续下降
        # → 提前降杠杆。与前五子区分：前五子看策略自身收益/风险质量（health 综合/win_rate
        # 频率/sharpe 单位风险收益/profit_factor 盈亏幅度/max_drawdown 回撤深度），本守卫看
        # 「资本配置效率」——单位资本的产出。一个策略可能 health_grade=A（盈利、低回撤），但
        # 配置了过多资本导致单位资本产出持续递减（pnl_per_capital_pct 下降），这是资本配置
        # 效率恶化的独立先行指标，直接对应「稳定资金增长」诉求。仅在交易笔数 ≥ min_trades
        # 时追踪，避免样本不足噪声误触发。
        crtg_cfg = agi_cfg.get("capital_return_trend_guard", {}) or {}
        self._capital_return_trend_enabled = bool(crtg_cfg.get("enabled", False))
        self._capital_return_trend_window = int(safe_float(crtg_cfg.get("window"), 3))
        self._capital_return_trend_window = max(2, min(20, self._capital_return_trend_window))
        self._capital_return_trend_min_trades = int(safe_float(crtg_cfg.get("min_trades"), 10))
        self._capital_return_trend_min_trades = max(1, min(1000, self._capital_return_trend_min_trades))
        self._capital_return_trend_leverage = safe_float(crtg_cfg.get("deterioration_leverage"), 1.0)

        # ── 资本回报率阈值守卫（capital_return_guard，补足资本效率维度的「阈值层」）──
        # 与 capital_return_trend_guard（pnl_per_capital_pct 连续下降，趋势型）区分：本守卫看
        # 「绝对阈值」——单位资本产出低于阈值（如 -10%，即每配置 100 USDT 资本亏 10 USDT）即
        # 收敛，不必等待连续下降确认，构成资本效率维度「阈值 + 趋势」双层（与 volatility_budget
        # 补充 volatility_trend 对称）。仅在交易笔数达 min_trades 时评估。阈值型守卫，无跨周期状态。
        crg_cfg = agi_cfg.get("capital_return_guard", {}) or {}
        self._capital_return_guard_enabled = bool(crg_cfg.get("enabled", False))
        self._capital_return_threshold = safe_float(crg_cfg.get("capital_return_threshold"), -10.0)
        self._capital_return_min_trades = int(safe_float(crg_cfg.get("min_trades"), 10))
        self._capital_return_min_trades = max(1, min(1000, self._capital_return_min_trades))
        self._capital_return_leverage = safe_float(crg_cfg.get("deterioration_leverage"), 1.0)
        self._capital_return_leverage = max(0.0, self._capital_return_leverage)

        # ── 策略波动率趋势外推（volatility_trend_guard，第七子）──
        # 追踪各策略 volatility（单笔盈亏标准差）跨周期时序，连续上升（波动放大）→ 提前降杠杆。
        # 与前六子区分：前六子看收益质量（health 综合/win_rate 频率/sharpe 单位风险收益/
        # profit_factor 盈亏幅度）、风险深度（max_drawdown）、资本效率（capital_return），
        # 本守卫看「绝对不确定性」——波动率是纯风险维度的独立信号，与夏普正交：夏普持平但
        # 波动率上升 = 策略承担更多风险换取同等收益，这是夏普（比值）无法捕获的独立衰退模式。
        # 仅在交易笔数 ≥ min_trades 时追踪，避免样本不足噪声误触发。
        vtg_cfg = agi_cfg.get("volatility_trend_guard", {}) or {}
        self._volatility_trend_enabled = bool(vtg_cfg.get("enabled", False))
        self._volatility_trend_window = int(safe_float(vtg_cfg.get("window"), 3))
        self._volatility_trend_window = max(2, min(20, self._volatility_trend_window))
        self._volatility_trend_min_trades = int(safe_float(vtg_cfg.get("min_trades"), 10))
        self._volatility_trend_min_trades = max(1, min(1000, self._volatility_trend_min_trades))
        self._volatility_trend_leverage = safe_float(vtg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级 PnL 动量趋势外推（pnl_momentum_trend_guard）──
        # 与七子守卫（health/win_rate/sharpe/profit_factor/max_drawdown/capital_return/volatility）
        # 区分：本守卫看「PnL 动量」——trend_pnl_7d_vs_30d（近7天 vs 近30天平均 PnL 比率，来自
        # contribution_analyzer 的趋势分解）。比率连续下降意味着策略近期盈利动量相对中期持续
        # 减弱（虽仍盈利但增长动能衰竭），是「从盈利走向停滞/转亏」的先行指标，独立于收益质量/
        # 风险深度/资本效率/波动率。仅在交易笔数达 min_trades 且比率>0（有数据）时追踪。
        pmtg_cfg = agi_cfg.get("pnl_momentum_trend_guard", {}) or {}
        self._pnl_momentum_trend_enabled = bool(pmtg_cfg.get("enabled", False))
        self._pnl_momentum_trend_window = int(safe_float(pmtg_cfg.get("window"), 3))
        self._pnl_momentum_trend_window = max(2, min(20, self._pnl_momentum_trend_window))
        self._pnl_momentum_trend_min_trades = int(safe_float(pmtg_cfg.get("min_trades"), 10))
        self._pnl_momentum_trend_min_trades = max(1, min(10000, self._pnl_momentum_trend_min_trades))
        self._pnl_momentum_trend_leverage = safe_float(pmtg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级 PnL 动量绝对阈值守卫（pnl_momentum_guard）──
        # AGI 感知策略「PnL 动量」（contribution_analyzer.trend_pnl_7d_vs_30d = 近7天 vs 近30天
        # 平均 PnL 比率），对「动量衰竭」（比率 < min_momentum_ratio，近期盈利动量相对中期明显减弱）
        # 的策略降杠杆收敛。与 pnl_momentum_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」
        # ——一个策略近期动量已衰减到中期 70% 以下但未连续下降（平坦低位）时趋势守卫不触发，但策略
        # 实际在「增长动能衰竭」（从盈利走向停滞/转亏的先行信号）。与 strategy_declining（需 grade D/F）
        # 区分：本守卫不看健康度——动量衰竭但健康度仍 A/B/C 的策略也应提前收敛。构成 PnL 动量维度
        # 「阈值 + 趋势」双层。阈值型守卫（非趋势型），无跨周期状态，不持久化、不纳入 _DECAY_ALERT_TYPES。
        pmg_cfg = agi_cfg.get("pnl_momentum_guard", {}) or {}
        self._pnl_momentum_guard_enabled = bool(pmg_cfg.get("enabled", False))
        self._pnl_momentum_guard_min_trades = int(safe_float(pmg_cfg.get("min_trades"), 10))
        self._pnl_momentum_guard_min_trades = max(1, self._pnl_momentum_guard_min_trades)
        self._pnl_momentum_guard_min_ratio = safe_float(pmg_cfg.get("min_momentum_ratio"), 0.7)
        self._pnl_momentum_guard_leverage = safe_float(pmg_cfg.get("deterioration_leverage"), 1.0)
        self._pnl_momentum_guard_leverage = max(0.0, self._pnl_momentum_guard_leverage)

        # ── 策略级单笔期望值趋势外推（pnl_per_trade_trend_guard）──
        # 与九子守卫（health/win_rate/sharpe/profit_factor/max_drawdown/drawdown_duration/capital_return/
        # volatility/pnl_momentum）区分：本守卫看「单笔期望值」——pnl_per_trade（total_pnl/total_trades，
        # 每笔平均盈亏）连续下降，意味着策略「每笔交易的价值持续萎缩」（即使总盈亏仍增长，也是靠更多
        # 交易堆砌，规模换质量），是「交易质量恶化」的先行指标，独立于收益质量/风险深度/资本效率/波动率/
        # PnL 动量。仅在交易笔数达 min_trades 时追踪，避免小样本噪声误触发。
        pttrg_cfg = agi_cfg.get("pnl_per_trade_trend_guard", {}) or {}
        self._pnl_per_trade_trend_enabled = bool(pttrg_cfg.get("enabled", False))
        self._pnl_per_trade_trend_window = int(safe_float(pttrg_cfg.get("window"), 3))
        self._pnl_per_trade_trend_window = max(2, min(20, self._pnl_per_trade_trend_window))
        self._pnl_per_trade_trend_min_trades = int(safe_float(pttrg_cfg.get("min_trades"), 10))
        self._pnl_per_trade_trend_min_trades = max(1, min(10000, self._pnl_per_trade_trend_min_trades))
        self._pnl_per_trade_trend_leverage = safe_float(pttrg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级单笔期望值绝对值守卫（pnl_per_trade_guard）──
        # 与 pnl_per_trade_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」——策略
        # pnl_per_trade 已低于 min_pnl_per_trade（默认0.0，即每笔平均盈亏非正）但尚未连续下降
        # （可能长期在低位徘徊）时趋势守卫不触发，但策略实际在「每笔交易无正期望」状态（交易
        # 质量差，靠规模堆砌或亏损）。构成单笔期望值维度「阈值+趋势」双层。仅在交易笔数达
        # min_trades 时评估。阈值型守卫（非趋势型），从当周期 contribution 读取 pnl_per_trade
        # （已在 snapshot_to_dict 透出），无跨周期状态。
        pptg_cfg = agi_cfg.get("pnl_per_trade_guard", {}) or {}
        self._pnl_per_trade_guard_enabled = bool(pptg_cfg.get("enabled", False))
        self._pnl_per_trade_guard_min_trades = int(safe_float(pptg_cfg.get("min_trades"), 10))
        self._pnl_per_trade_guard_min_trades = max(1, self._pnl_per_trade_guard_min_trades)
        self._pnl_per_trade_guard_min_value = safe_float(pptg_cfg.get("min_pnl_per_trade"), 0.0)
        self._pnl_per_trade_guard_leverage = safe_float(pptg_cfg.get("deterioration_leverage"), 1.0)
        self._pnl_per_trade_guard_leverage = max(0.0, self._pnl_per_trade_guard_leverage)

        # ── 前瞻盈利推算与盈亏归因（pnl_projection）──
        # 在 run_cycle 中新增「归因→推算」步骤（diagnose 之后、decide 之前），构成
        # 「诊断→归因→推算→决策→执行→反思」六段式闭环。推算结果注入 _decide 和
        # _offensive_allocation_actions，影响资金分配与进攻性加仓 gate。
        # fail-closed：缺数据/异常时返回 available=false + correction_factor=1.0，不影响原闭环。
        pp_cfg = agi_cfg.get("pnl_projection", {}) or {}
        self._pnl_projection_enabled = bool(pp_cfg.get("enabled", False))
        self._projection_horizon = int(safe_float(pp_cfg.get("horizon_cycles"), 5))
        self._projection_horizon = max(1, min(50, self._projection_horizon))
        self._projection_accuracy_window = int(safe_float(pp_cfg.get("accuracy_window"), 20))
        self._projection_accuracy_window = max(1, self._projection_accuracy_window)
        self._projection_min_accuracy_samples = int(safe_float(pp_cfg.get("min_accuracy_samples"), 5))
        self._projection_min_accuracy_samples = max(1, self._projection_min_accuracy_samples)
        self._regime_correction_max_age_cycles = max(
            1, int(safe_float(pp_cfg.get("regime_correction_max_age_cycles"), 100))
        )
        self._projection_min_correction = safe_float(pp_cfg.get("min_correction"), 0.5)
        self._projection_max_correction = safe_float(pp_cfg.get("max_correction"), 2.0)
        self._projection_trend_weight = safe_float(pp_cfg.get("trend_weight"), 0.3)
        self._projection_max_trend_adj_ratio = safe_float(pp_cfg.get("max_trend_adj_ratio"), 0.5)
        self._projection_confidence_level = safe_float(pp_cfg.get("confidence_level"), 0.95)
        self._projection_adaptation_span = safe_float(pp_cfg.get("projection_adaptation_span"), 0.2)
        self._projection_min_mult = safe_float(pp_cfg.get("projection_min_mult"), 0.5)
        self._projection_max_mult = safe_float(pp_cfg.get("projection_max_mult"), 1.5)
        self._projection_fail_closed_loss = safe_float(pp_cfg.get("fail_closed_loss_threshold"), 0.05)
        self._offensive_projection_loss_threshold = safe_float(
            pp_cfg.get("offensive_projection_loss_threshold"), 0.0)
        self._projection_unit_pnl = safe_float(pp_cfg.get("projection_unit_pnl"), 1.0)
        self._projection_unit_pnl = max(0.0001, self._projection_unit_pnl)
        self._projection_min_attribution_trades = int(
            safe_float(pp_cfg.get("min_attribution_trades"), 20))
        self._projection_min_attribution_trades = max(1, self._projection_min_attribution_trades)

        # ── 策略级边际盈亏趋势外推（delta_pnl_trend_guard）──
        # 与十子守卫（health/win_rate/sharpe/profit_factor/max_drawdown/drawdown_duration/capital_return/
        # volatility/pnl_momentum/pnl_per_trade）区分：本守卫看「边际盈亏」——delta_pnl（本周期 vs
        # 上一快照的盈亏增量）连续为负，意味着策略「最近持续失血」（即使历史累计 total_pnl 仍为正，
        # 近期也在亏），是「从盈利转亏损」的周期级先行指标，独立于累计盈亏/比率趋势/笔数连亏。
        # 仅在交易笔数达 min_trades 时追踪，避免小样本噪声误触发。
        dptg_cfg = agi_cfg.get("delta_pnl_trend_guard", {}) or {}
        self._delta_pnl_trend_enabled = bool(dptg_cfg.get("enabled", False))
        self._delta_pnl_trend_window = int(safe_float(dptg_cfg.get("window"), 3))
        self._delta_pnl_trend_window = max(2, min(20, self._delta_pnl_trend_window))
        self._delta_pnl_trend_min_trades = int(safe_float(dptg_cfg.get("min_trades"), 10))
        self._delta_pnl_trend_min_trades = max(1, min(10000, self._delta_pnl_trend_min_trades))
        self._delta_pnl_trend_leverage = safe_float(dptg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级单周期亏损急跌守卫（delta_pnl_guard）──
        # AGI 感知策略「边际盈亏」（contribution_analyzer.delta_pnl = 本周期 vs 上一快照盈亏增量），
        # 对「单周期大额急跌」（delta_pnl < 0 且 |delta_pnl|/equity ≥ loss_threshold，即单周期亏损
        # 占账户权益超阈值）的策略降杠杆收敛。与 delta_pnl_trend_guard（连续 N 周期为负，趋势型）
        # 区分：本守卫看「绝对阈值」——策略单周期突然大幅亏损（急跌尖峰）但未连续为负时趋势守卫
        # 不触发，但策略实际在「单周期急跌」（可能遭遇突发不利行情），是「从盈利转亏损」的即时
        # 硬信号。与 unrealized_loss_guard（浮亏占比，当前状态）区分：本守卫看「单周期盈亏增量」
        # （急跌幅度）；与 realized_loss_guard（已实现亏损，永久）区分：本守卫看单周期增量（含浮亏
        # 变动）。构成边际盈亏维度「阈值 + 趋势」双层。阈值型守卫（非趋势型），无跨周期状态。
        dpg_cfg = agi_cfg.get("delta_pnl_guard", {}) or {}
        self._delta_pnl_guard_enabled = bool(dpg_cfg.get("enabled", False))
        self._delta_pnl_guard_min_trades = int(safe_float(dpg_cfg.get("min_trades"), 10))
        self._delta_pnl_guard_min_trades = max(1, self._delta_pnl_guard_min_trades)
        self._delta_pnl_guard_loss_threshold = safe_float(dpg_cfg.get("loss_threshold"), 0.02)
        self._delta_pnl_guard_loss_threshold = max(0.0, min(1.0, self._delta_pnl_guard_loss_threshold))
        self._delta_pnl_guard_leverage = safe_float(dpg_cfg.get("deterioration_leverage"), 1.0)
        self._delta_pnl_guard_leverage = max(0.0, self._delta_pnl_guard_leverage)

        # ── 浮盈占比过高检测（unrealized_profit_ratio_guard）──
        # 收益侧闭环的「盈利质量」维度：当策略 total_pnl>0 但 unrealized_pnl 占比过高时，
        # 说明账面盈利主要靠未兑现浮盈支撑，一旦回撤浮盈快速蒸发。与 give_back（浮盈已
        # 从峰值回吐，事后/时序）区分：本守卫看「浮盈占比过高」（事前/结构），在回吐发生
        # 之前预警盈利脆弱性。与 profit_take（账户级浮盈落袋）区分：本守卫是策略级（单
        # 策略浮盈占比）。阈值型守卫，无跨周期状态，不持久化。
        uprg_cfg = agi_cfg.get("unrealized_profit_ratio_guard", {}) or {}
        self._unrealized_profit_ratio_enabled = bool(uprg_cfg.get("enabled", False))
        self._unrealized_profit_ratio_threshold = safe_float(uprg_cfg.get("ratio_threshold"), 0.8)
        self._unrealized_profit_ratio_threshold = max(0.5, min(1.0, self._unrealized_profit_ratio_threshold))
        self._unrealized_profit_ratio_reduce_target = safe_float(uprg_cfg.get("reduce_target"), 0.1)
        self._unrealized_profit_ratio_reduce_target = max(0.0, min(1.0, self._unrealized_profit_ratio_reduce_target))

        # ── 浮盈占比趋势外推（unrealized_profit_ratio_trend_guard）──
        # 与 unrealized_profit_ratio_guard（阈值型：浮盈占比超 ratio_threshold 才收敛，事后）区分：
        # 本守卫追踪各策略「浮盈占比」（unrealized_pnl/total_pnl）跨周期时序，检测连续 window
        # 周期严格上升——盈利质量持续劣化的轨迹（事前），尚未触及 ratio_threshold 即提前锁定
        # 浮盈。与 unrealized_loss_trend_guard（浮亏连续加深，亏损侧趋势）形成对称：浮亏侧看
        # 「亏损加深」，本守卫看「盈利变虚」（账面盈利越来越依赖未兑现浮盈），共同对应「稳定
        # 资金增长」诉求。
        uprtg_cfg = agi_cfg.get("unrealized_profit_ratio_trend_guard", {}) or {}
        self._unrealized_profit_ratio_trend_enabled = bool(uprtg_cfg.get("enabled", False))
        self._unrealized_profit_ratio_trend_window = int(safe_float(uprtg_cfg.get("window"), 3))
        self._unrealized_profit_ratio_trend_window = max(2, min(20, self._unrealized_profit_ratio_trend_window))
        self._unrealized_profit_ratio_trend_reduce_target = safe_float(uprtg_cfg.get("reduce_target"), 0.1)
        self._unrealized_profit_ratio_trend_reduce_target = max(0.0, min(1.0, self._unrealized_profit_ratio_trend_reduce_target))

        # ── 多守卫共振收敛（resonance_guard）──
        # 事前风控六子守卫各自独立检测单维度衰退，本守卫补足「多维度同时衰退」的升级响应：
        # 当同一策略在同一周期触发 ≥ resonance_threshold（默认2）个趋势守卫（health/win_rate/
        # sharpe/profit_factor/max_drawdown/capital_return 任意组合）时，判定为「多维度共振衰退」，
        # 比单维度衰退严重得多，升级为更大力度的降杠杆（resonance_leverage 默认0.5，比单守卫
        # 1.0 更保守）。无跨周期状态（直接从当周期告警统计），故不持久化。
        res_cfg = agi_cfg.get("resonance_guard", {}) or {}
        self._resonance_enabled = bool(res_cfg.get("enabled", False))
        self._resonance_threshold = int(safe_float(res_cfg.get("resonance_threshold"), 2))
        self._resonance_threshold = max(2, min(11, self._resonance_threshold))
        self._resonance_leverage = safe_float(res_cfg.get("resonance_leverage"), 0.5)

        # ── 恢复纯度门控（recovery_purity_guard）──
        # strategy_recovered（trend=improving 且 grade A/B）与趋势型衰退告警可能对同一策略
        # 同时触发（如 health_grade=A 但 health_score 连续下降），产生「已恢复」与「衰退中」
        # 矛盾信号，导致 _strategy_resume_actions 恢复开仓与 _health_trend_actions 降杠杆并存。
        # 本门控强化 strategy_recovered 判定：仅当本周期无任何趋势型衰退告警时才判定「已恢复」，
        # 消除矛盾信号与 pause/resume 震荡。
        rpg_cfg = agi_cfg.get("recovery_purity_guard", {}) or {}
        self._recovery_purity_enabled = bool(rpg_cfg.get("enabled", False))

        # ── 组合级盈利集中度收敛（profit_concentration_guard）──
        # 组合级新维度：盈利来源的集中度（单一盈利支柱风险）。与 correlation（策略间收益
        # 相关性）和 diversification（资金配置权重 HHI）区分——即使资金分散（HHI 低）、
        # 策略不相关，若组合盈利过度依赖单一策略，一旦该支柱衰退，组合将无支撑快速转亏。
        # 计算「最大盈利策略 total_pnl / 组合 total_pnl」，超过 threshold 时告警，
        # 并抑制对该支柱策略的进攻性加仓（避免把资金进一步压到单一支柱上）。
        pcg_cfg = agi_cfg.get("profit_concentration_guard", {}) or {}
        self._profit_concentration_enabled = bool(pcg_cfg.get("enabled", False))
        self._profit_concentration_threshold = safe_float(pcg_cfg.get("concentration_threshold"), 0.8)
        self._profit_concentration_threshold = max(0.5, min(2.0, self._profit_concentration_threshold))

        # ── 组合级尾部风险收敛（tail_risk_guard，CVaR 代理）──
        # 组合级风险四子维度之四：尾部风险（最坏情况下的极端亏损）。组合级 CVaR 的务实代理——
        # 以「组合中最深策略回撤 max(max_drawdown_i)」作为尾部亏损的保守下界（组合整体回撤
        # 至少等于最坏单策略回撤）。与 max_drawdown_trend_guard（单策略回撤趋势）区分：
        # 后者看单策略回撤的「趋势加深」，本守卫看组合级回撤的「绝对水平」——即使各策略
        # 回撤未在加深，只要组合中存在一个高回撤策略，尾部风险就高。
        trg_cfg = agi_cfg.get("tail_risk_guard", {}) or {}
        self._tail_risk_enabled = bool(trg_cfg.get("enabled", False))
        self._tail_risk_threshold = safe_float(trg_cfg.get("tail_risk_threshold"), 0.15)
        self._tail_risk_threshold = max(0.05, min(1.0, self._tail_risk_threshold))
        self._tail_risk_reduce_target = safe_float(trg_cfg.get("reduce_target"), 0.1)
        self._tail_risk_reduce_target = max(0.0, min(1.0, self._tail_risk_reduce_target))

        # ── 组合级协同度收敛（synergy_guard）──
        # 组合级第五个结构维度：协同质量（synergy_score，来自 ContributionAnalyzer）。与四子
        # 结构维度（correlation/weight_concentration/profit_concentration/tail_risk）区分——四子
        # 看风险（联动/集中/盈利集中/尾部），本守卫看「策略组合的协同质量」：低相关+正 PnL =
        # 高协同（互补），高相关+负 PnL = 低协同（相互拖累）。synergy_score 低于阈值说明策略
        # 组合在「相关地一起亏」（协同劣化），即使相关性未单独触发、各策略健康度尚可，也应收敛。
        sg_cfg = agi_cfg.get("synergy_guard", {}) or {}
        self._synergy_enabled = bool(sg_cfg.get("enabled", False))
        self._synergy_threshold = safe_float(sg_cfg.get("synergy_threshold"), 0.3)
        self._synergy_threshold = max(0.0, min(1.0, self._synergy_threshold))
        self._synergy_min_strategies = int(safe_float(sg_cfg.get("min_strategies"), 2))
        self._synergy_min_strategies = max(2, min(50, self._synergy_min_strategies))
        self._synergy_reduce_target = safe_float(sg_cfg.get("reduce_target"), 0.15)
        self._synergy_reduce_target = max(0.0, min(1.0, self._synergy_reduce_target))

        # ── 组合级协同度趋势外推（synergy_trend_guard）──
        # 与 synergy_guard（阈值型：synergy_score 跌破阈值才收敛，事后）区分：本守卫追踪
        # synergy_score 跨周期时序，检测连续 window 周期严格下降——协同质量持续劣化的轨迹
        # （事前），即使尚未跌破 synergy_threshold 也应收敛。与 portfolio_correlation_trend
        # （相关性上升，看联动）区分：本守卫看协同质量下降（相关性 + 盈亏同向性综合恶化），
        # 是更综合的联动劣化信号。至此组合级协同质量维度也形成「阈值 + 趋势」双层。
        stg_cfg = agi_cfg.get("synergy_trend_guard", {}) or {}
        self._synergy_trend_enabled = bool(stg_cfg.get("enabled", False))
        self._synergy_trend_window = int(safe_float(stg_cfg.get("window"), 3))
        self._synergy_trend_window = max(2, min(20, self._synergy_trend_window))
        self._synergy_trend_min_strategies = int(safe_float(stg_cfg.get("min_strategies"), 2))
        self._synergy_trend_min_strategies = max(2, min(50, self._synergy_trend_min_strategies))
        self._synergy_trend_reduce_target = safe_float(stg_cfg.get("reduce_target"), 0.15)
        self._synergy_trend_reduce_target = max(0.0, min(1.0, self._synergy_trend_reduce_target))

        # ── 组合级风险共振收敛（portfolio_risk_resonance_guard）──
        # 组合级风险四子维度（correlation / weight_concentration / profit_concentration /
        # tail_risk）各自独立响应，但缺「多维度同时恶化」的聚合升级。本守卫检测当周期
        # 触发的独立组合级风险维度数 ≥ resonance_threshold 时生成组合级共振告警，触发
        # 比单维度更保守的全局收敛。与策略级 resonance_guard（六子守卫共振）对称：
        # 策略级看单策略多维衰退，组合级看多维度组合风险同时恶化。
        prrg_cfg = agi_cfg.get("portfolio_risk_resonance_guard", {}) or {}
        self._portfolio_resonance_enabled = bool(prrg_cfg.get("enabled", False))
        self._portfolio_resonance_threshold = int(safe_float(prrg_cfg.get("resonance_threshold"), 2))
        self._portfolio_resonance_threshold = max(2, min(4, self._portfolio_resonance_threshold))
        self._portfolio_resonance_reduce_target = safe_float(prrg_cfg.get("reduce_target"), 0.1)
        self._portfolio_resonance_reduce_target = max(0.0, min(1.0, self._portfolio_resonance_reduce_target))

        # ── 组合级健康度收敛（portfolio_health_guard，阈值型）──
        # 组合级健康度双层之「阈值层」：策略级健康度有「阈值型（grade F/D → strategy_health_
        # critical/strategy_declining）+ 趋势型（health_score 下降 → health_deteriorating）」双层，
        # 组合级健康度此前只有趋势型（portfolio_health_trend_guard 追踪 overall_health_score
        # 连续下降），缺阈值型——组合整体健康度即使未在连续下降，只要已跌破健康线（绝对值差）
        # 就应收敛。本守卫检测 overall_health_score < health_threshold（默认45，对应 C 线以下
        # D/F 级）时生成 portfolio_health_low 告警并收敛组合敞口。仅在成交数 ≥ min_sample
        # （默认10，与 portfolio_health_trend_guard 口径一致）时评估，避免样本不足误触发。
        phg_cfg = agi_cfg.get("portfolio_health_guard", {}) or {}
        self._portfolio_health_enabled = bool(phg_cfg.get("enabled", False))
        self._portfolio_health_threshold = safe_float(phg_cfg.get("health_threshold"), 45.0)
        self._portfolio_health_threshold = max(0.0, min(100.0, self._portfolio_health_threshold))
        self._portfolio_health_min_sample = int(safe_float(phg_cfg.get("min_sample"), 10))
        self._portfolio_health_min_sample = max(1, min(1000, self._portfolio_health_min_sample))
        self._portfolio_health_reduce_target = safe_float(phg_cfg.get("reduce_target"), 0.15)
        self._portfolio_health_reduce_target = max(0.0, min(1.0, self._portfolio_health_reduce_target))

        # ── 组合级健康度趋势外推（portfolio_health_trend_guard）──
        # 策略级 health_trend_guard 追踪单策略 health_score 时序，本守卫追踪组合
        # overall_health_score 时序——捕获「所有策略都在轻微衰退，单策略均未触发」的
        # 温水煮青蛙式组合级衰退（组合整体健康度连续下降）。与策略级 health_trend 对称。
        phtg_cfg = agi_cfg.get("portfolio_health_trend_guard", {}) or {}
        self._portfolio_health_trend_enabled = bool(phtg_cfg.get("enabled", False))
        self._portfolio_health_trend_window = int(safe_float(phtg_cfg.get("window"), 3))
        self._portfolio_health_trend_window = max(2, min(20, self._portfolio_health_trend_window))
        self._portfolio_health_trend_min_sample = int(safe_float(phtg_cfg.get("min_sample"), 10))
        self._portfolio_health_trend_min_sample = max(1, min(1000, self._portfolio_health_trend_min_sample))
        self._portfolio_health_trend_reduce_target = safe_float(phtg_cfg.get("reduce_target"), 0.15)
        self._portfolio_health_trend_reduce_target = max(0.0, min(1.0, self._portfolio_health_trend_reduce_target))

        # ── 组合级集中度趋势外推（portfolio_concentration_trend_guard）──
        # 追踪组合 concentration（资金配置权重 HHI）跨周期时序，连续上升 → 组合分散度侵蚀预警。
        # 与 diversification（阈值型：HHI 超阈值才收敛，事后）区分：本守卫看「集中度持续上升」
        # 的轨迹（事前）——即使尚未触及 concentration_high_threshold，分散度正在被侵蚀也应
        # 提前收敛。与 portfolio_health_trend（组合健康度下降）区分：本守卫看资金配置集中度
        # （结构维度），健康度看综合收益/风险质量。二者正交。
        pctg_cfg = agi_cfg.get("portfolio_concentration_trend_guard", {}) or {}
        self._portfolio_concentration_trend_enabled = bool(pctg_cfg.get("enabled", False))
        self._portfolio_concentration_trend_window = int(safe_float(pctg_cfg.get("window"), 3))
        self._portfolio_concentration_trend_window = max(2, min(20, self._portfolio_concentration_trend_window))
        self._portfolio_concentration_trend_reduce_target = safe_float(pctg_cfg.get("reduce_target"), 0.2)
        self._portfolio_concentration_trend_reduce_target = max(0.0, min(1.0, self._portfolio_concentration_trend_reduce_target))

        # ── 组合级相关性趋势外推（portfolio_correlation_trend_guard）──
        # 追踪组合 max_pair_correlation（策略间最大成对相关性）跨周期时序，连续上升 →
        # 联动风险加剧预警。与 correlation（阈值型：max_pair 超阈值才收敛，事后）区分：
        # 本守卫看「相关性持续上升」轨迹（事前）——即使尚未触及 high_corr_threshold，策略
        # 收益趋同正在加剧也应提前收敛。与 portfolio_concentration_trend（资金配置集中度
        # 上升）区分：本守卫看策略间收益联动（相关性），集中度看资金配置权重（结构），二者正交。
        pctg2_cfg = agi_cfg.get("portfolio_correlation_trend_guard", {}) or {}
        self._portfolio_correlation_trend_enabled = bool(pctg2_cfg.get("enabled", False))
        self._portfolio_correlation_trend_window = int(safe_float(pctg2_cfg.get("window"), 3))
        self._portfolio_correlation_trend_window = max(2, min(20, self._portfolio_correlation_trend_window))
        self._portfolio_correlation_trend_reduce_target = safe_float(pctg2_cfg.get("reduce_target"), 0.2)
        self._portfolio_correlation_trend_reduce_target = max(0.0, min(1.0, self._portfolio_correlation_trend_reduce_target))

        # ── 组合级尾部风险趋势外推（portfolio_tail_risk_trend_guard）──
        # 追踪组合 tail_risk（max(max_drawdown_i)，最深单策略回撤）跨周期时序，连续上升 →
        # 尾部风险加深预警。与 tail_risk_guard（阈值型：max_dd 超阈值才收敛，事后）区分：
        # 本守卫看「尾部风险持续加深」轨迹（事前）——组合最深策略回撤在创新低，即使尚未触及
        # tail_risk_threshold 也应提前收敛。与 max_drawdown_trend_guard（单策略回撤趋势）区分：
        # 本守卫看组合级最深回撤（尾部），后者看每个策略自身回撤，二者响应粒度不同。
        ptrtg_cfg = agi_cfg.get("portfolio_tail_risk_trend_guard", {}) or {}
        self._portfolio_tail_risk_trend_enabled = bool(ptrtg_cfg.get("enabled", False))
        self._portfolio_tail_risk_trend_window = int(safe_float(ptrtg_cfg.get("window"), 3))
        self._portfolio_tail_risk_trend_window = max(2, min(20, self._portfolio_tail_risk_trend_window))
        self._portfolio_tail_risk_trend_reduce_target = safe_float(ptrtg_cfg.get("reduce_target"), 0.2)
        self._portfolio_tail_risk_trend_reduce_target = max(0.0, min(1.0, self._portfolio_tail_risk_trend_reduce_target))

        # ── 组合级盈利集中度趋势外推（profit_concentration_trend_guard）──
        # 追踪组合 profit_concentration（max(total_pnl_i) / total_pnl）跨周期时序，连续上升 →
        # 单一盈利支柱风险加剧预警。与 profit_concentration_guard（阈值型：concentration 超阈值
        # 才门控，事后）区分：本守卫看「盈利来源集中度持续上升」轨迹（事前）——即使尚未触及
        # concentration_threshold，盈利支柱正在单一化也应提前收敛。与 portfolio_concentration_
        # trend_guard（资金配置权重 HHI 上升，看配置结构）区分：本守卫看盈利来源集中（收益结构），
        # 二者正交。至此组合级四子维度（correlation/weight_concentration/profit_concentration/
        # tail_risk）均已形成「阈值型 + 趋势型」双层结构。
        pctg3_cfg = agi_cfg.get("profit_concentration_trend_guard", {}) or {}
        self._profit_concentration_trend_enabled = bool(pctg3_cfg.get("enabled", False))
        self._profit_concentration_trend_window = int(safe_float(pctg3_cfg.get("window"), 3))
        self._profit_concentration_trend_window = max(2, min(20, self._profit_concentration_trend_window))
        self._profit_concentration_trend_reduce_target = safe_float(pctg3_cfg.get("reduce_target"), 0.2)
        self._profit_concentration_trend_reduce_target = max(0.0, min(1.0, self._profit_concentration_trend_reduce_target))

        # ── 资金利用率趋势外推（utilization_trend_guard）──
        # 与 utilization_guard（阈值型，事后）区分：utilization_guard 在资金利用率
        # used_margin/equity 超过 high_utilization_threshold（默认0.8）时才收敛，本守卫
        # 追踪利用率跨周期时序，检测连续 window 周期严格上升——杠杆/敞口持续放大的
        # 轨迹，即使尚未触及强平危险线。与 drawdown_acceleration（回撤深度加深，亏损侧）
        # 区分：本守卫看利用率上升（杠杆侧），二者正交。
        utg_cfg = agi_cfg.get("utilization_trend_guard", {}) or {}
        self._utilization_trend_enabled = bool(utg_cfg.get("enabled", False))
        self._utilization_trend_window = int(safe_float(utg_cfg.get("window"), 3))
        self._utilization_trend_window = max(2, min(20, self._utilization_trend_window))
        self._utilization_trend_reduce_target = safe_float(utg_cfg.get("reduce_target"), 0.15)
        self._utilization_trend_reduce_target = max(0.0, min(1.0, self._utilization_trend_reduce_target))

        # ── 市场状态突变进攻冷却（regime_shift_guard）──
        # 市场状态突变说明趋势尚未确认，AGI 应在突变后冷却 cooldown_cycles 个周期
        # 暂停进攻性加仓，等新状态稳定、趋势重新确认后再行动（对应用户「趋势确认
        # 后才开仓」诉求）。仅高置信度突变触发；低置信度疑似变化不触发冷却。
        rsg_cfg = agi_cfg.get("regime_shift_guard", {}) or {}
        self._regime_shift_guard_enabled = bool(rsg_cfg.get("enabled", False))
        self._regime_shift_cooldown_cycles = int(safe_float(rsg_cfg.get("cooldown_cycles"), 3))
        self._regime_shift_cooldown_cycles = max(1, min(100, self._regime_shift_cooldown_cycles))

        # ── 策略级利润回吐保护（give_back_guard）──
        # 收益侧闭环：追踪各策略累计盈亏峰值，浮盈从峰值回吐超过 give_back_threshold
        # 时主动减仓锁利，避免「赚过又吐回去」（对应用户「稳定资金增长」诉求）。
        # 与 profit_take（账户级浮盈落袋）、ProfitLockEngine（持仓级锁微利）分层互补。
        # 分级响应：基础（20%→减仓10%）→ 严重（50%→减仓2%）→ 危急（80%→冻结停开仓），
        # 防止单一减仓力度不足以阻止「盈利全回吐」（如 sync 26.76→-0.40）。
        gbg_cfg = agi_cfg.get("give_back_guard", {}) or {}
        self._give_back_enabled = bool(gbg_cfg.get("enabled", False))
        self._give_back_threshold = safe_float(gbg_cfg.get("give_back_threshold"), 0.2)
        self._give_back_threshold = max(0.0, min(1.0, self._give_back_threshold))
        self._give_back_reduce_target = safe_float(gbg_cfg.get("reduce_target"), 0.1)
        self._give_back_reduce_target = max(0.0, min(1.0, self._give_back_reduce_target))
        # 严重回吐分级（默认关，向后兼容）
        self._give_back_severe_enabled = bool(gbg_cfg.get("severe_enabled", False))
        self._give_back_severe_threshold = safe_float(gbg_cfg.get("severe_threshold"), 0.5)
        self._give_back_severe_threshold = max(
            self._give_back_threshold, min(1.0, self._give_back_severe_threshold))
        self._give_back_severe_reduce_target = safe_float(
            gbg_cfg.get("severe_reduce_target"), 0.02)
        self._give_back_severe_reduce_target = max(
            0.0, min(1.0, self._give_back_severe_reduce_target))
        # 危急回吐分级（默认关，向后兼容）：回吐超阈值 → 暂停策略开新仓
        self._give_back_critical_enabled = bool(gbg_cfg.get("critical_enabled", False))
        self._give_back_critical_threshold = safe_float(gbg_cfg.get("critical_threshold"), 0.8)
        self._give_back_critical_threshold = max(
            self._give_back_severe_threshold, min(1.0, self._give_back_critical_threshold))
        # 峰值生命周期管理（防陈旧峰值反复触发 → 策略恢复后立即再次暂停的死锁）
        # reset_on_resume: 策略被 resume 后重置 peak=0（新周期从零计起）
        # reset_on_critical: 危急回吐清仓后重置 peak=0（仓位已清，旧峰值无意义）
        # peak_decay_cycles: peak 每 N 周期衰减一次（alpha=0.9），使近期表现权重更高
        self._give_back_peak_reset_on_resume = bool(
            gbg_cfg.get("peak_reset_on_resume", True))
        self._give_back_peak_reset_on_critical = bool(
            gbg_cfg.get("peak_reset_on_critical", True))
        self._give_back_peak_decay_cycles = int(
            gbg_cfg.get("peak_decay_cycles", 0))  # 默认 0=禁用衰减
        self._give_back_peak_decay_cycles = max(0, min(1000, self._give_back_peak_decay_cycles))
        self._give_back_peak_decay_alpha = 0.9  # 衰减因子：每轮 peak *= 0.9
        self._give_back_peak_last_decay_cycle: Dict[str, int] = {}

        # ── 交易成本感知与响应（cost_awareness）──
        # 手续费侵蚀度量：当账户累计手续费占毛利（手续费前利润）比例超过阈值时，
        # 说明交易过频、换手过度，主动抑制进攻性加仓（降低交易频率），
        # 与 cost_guard（事前防微小调仓）形成「事前+事后」双重成本控制。
        ca_cfg = agi_cfg.get("cost_awareness", {}) or {}
        self._cost_awareness_enabled = bool(ca_cfg.get("enabled", False))
        self._cost_awareness_threshold = safe_float(ca_cfg.get("fee_ratio_threshold"), 0.3)
        self._cost_awareness_threshold = max(0.0, min(1.0, self._cost_awareness_threshold))
        self._cost_awareness_reduce_target = safe_float(ca_cfg.get("reduce_target"), 0.1)
        self._cost_awareness_reduce_target = max(0.0, min(1.0, self._cost_awareness_reduce_target))

        # ── 追涨抑制（momentum_guard）──
        # 感知 EquityMonitor 的连续上涨周期数（consecutive_up），连续上涨过多说明市场
        # 过热，AGI 应暂停进攻性加仓（不追高），直接对应「消除追涨杀跌」诉求。
        mg_cfg = agi_cfg.get("momentum_guard", {}) or {}
        self._momentum_guard_enabled = bool(mg_cfg.get("enabled", False))
        self._momentum_max_consecutive_up = int(safe_float(mg_cfg.get("max_consecutive_up"), 3))
        self._momentum_max_consecutive_up = max(1, min(100, self._momentum_max_consecutive_up))

        # ── 资金利用率过高收敛（utilization_guard）──
        # 资金利用率（已用保证金/权益）过高说明敞口接近满仓/过度杠杆，有强平风险，
        # 主动抑制进攻性加仓（收敛敞口）。与 low_capital_utilization→idle_cash_deploy
        # （过低→归集）形成对称：过低归集，过高收敛。
        ug_cfg = agi_cfg.get("utilization_guard", {}) or {}
        self._utilization_guard_enabled = bool(ug_cfg.get("enabled", False))
        self._utilization_high_threshold = safe_float(ug_cfg.get("high_utilization_threshold"), 0.8)
        self._utilization_high_threshold = max(0.0, min(1.0, self._utilization_high_threshold))
        self._utilization_reduce_target = safe_float(ug_cfg.get("reduce_target"), 0.15)
        self._utilization_reduce_target = max(0.0, min(1.0, self._utilization_reduce_target))

        # ── 策略级浮亏止损（unrealized_loss_guard）──
        # 收益侧闭环（profit_take 账户浮盈 / give_back 策略回吐 / ProfitLock 持仓锁微利）
        # 已覆盖「保住利润」，本 guard 补足亏损侧闭环：单策略当前浮亏（unrealized_pnl<0）
        # 占账户权益比例超阈值 → 主动止损减仓，限制单策略亏损，对应用户「稳定资金增长」。
        ulg_cfg = agi_cfg.get("unrealized_loss_guard", {}) or {}
        self._unrealized_loss_enabled = bool(ulg_cfg.get("enabled", False))
        self._unrealized_loss_threshold = safe_float(ulg_cfg.get("loss_threshold"), 0.03)
        self._unrealized_loss_threshold = max(0.0, min(1.0, self._unrealized_loss_threshold))
        self._unrealized_loss_reduce_target = safe_float(ulg_cfg.get("reduce_target"), 0.0)
        self._unrealized_loss_reduce_target = max(0.0, min(1.0, self._unrealized_loss_reduce_target))

        # ── 策略级浮亏加深趋势外推（unrealized_loss_trend_guard）──
        # unrealized_loss_guard（阈值型，事后：浮亏占权益超阈值才清仓）的「事前」补充：
        # 追踪各策略 unrealized_pnl 跨周期时序，连续下降（浮亏加深）时提前收敛敞口，
        # 在浮亏触及 loss_threshold 之前止损，避免单策略亏损扩散。与 give_back（浮盈从
        # 峰值回吐，收益侧趋势）形成对称：give_back 看收益侧回吐，本守卫看亏损侧加深。
        ultg_cfg = agi_cfg.get("unrealized_loss_trend_guard", {}) or {}
        self._unrealized_loss_trend_enabled = bool(ultg_cfg.get("enabled", False))
        self._unrealized_loss_trend_window = int(safe_float(ultg_cfg.get("window"), 3))
        self._unrealized_loss_trend_window = max(2, min(20, self._unrealized_loss_trend_window))
        self._unrealized_loss_trend_reduce_target = safe_float(ultg_cfg.get("reduce_target"), 0.1)
        self._unrealized_loss_trend_reduce_target = max(0.0, min(1.0, self._unrealized_loss_trend_reduce_target))

        # ── 策略级已实现亏损守卫（realized_loss_guard）──
        # 与 unrealized_loss_guard（看「浮亏」unrealized_pnl<0，可能回本，事后止损）区分：本守卫看
        # 「已实现亏损」realized_pnl<0——已平仓交易累计为负（永久损失、不可逆）。一个策略即使当前
        # 浮盈（unrealized>0），若已实现亏损（realized<0）持续累积，说明它「平仓的都是亏的」，
        # 是「频繁交易消耗资金」的直接体现（频繁开平仓、每笔小亏、手续费叠加）。阈值型守卫，无跨周期状态。
        rlg_cfg = agi_cfg.get("realized_loss_guard", {}) or {}
        self._realized_loss_enabled = bool(rlg_cfg.get("enabled", False))
        self._realized_loss_threshold = safe_float(rlg_cfg.get("loss_threshold"), 0.03)
        self._realized_loss_threshold = max(0.0, min(1.0, self._realized_loss_threshold))
        self._realized_loss_reduce_target = safe_float(rlg_cfg.get("reduce_target"), 0.1)
        self._realized_loss_reduce_target = max(0.0, min(1.0, self._realized_loss_reduce_target))

        # ── 策略级手续费率守卫（strategy_fee_ratio_guard）──
        # 与 cost_awareness（账户级手续费占毛利比例过高，抑制进攻）区分：本守卫看「单策略」
        # 手续费侵蚀——单策略手续费占毛利（total_fees/(total_pnl+total_fees)）比例过高说明该
        # 策略交易过频、换手过度，直接对应「消除频繁交易消耗资金」诉求。阈值型守卫，无跨周期状态。
        sfrg_cfg = agi_cfg.get("strategy_fee_ratio_guard", {}) or {}
        self._strategy_fee_ratio_enabled = bool(sfrg_cfg.get("enabled", False))
        self._strategy_fee_ratio_threshold = safe_float(sfrg_cfg.get("fee_ratio_threshold"), 0.3)
        self._strategy_fee_ratio_threshold = max(0.0, min(1.0, self._strategy_fee_ratio_threshold))
        self._strategy_fee_ratio_reduce_target = safe_float(sfrg_cfg.get("reduce_target"), 0.1)
        self._strategy_fee_ratio_reduce_target = max(0.0, min(1.0, self._strategy_fee_ratio_reduce_target))

        # ── 策略级手续费率趋势外推（strategy_fee_ratio_trend_guard）──
        # 与 strategy_fee_ratio_guard（阈值型：fee_ratio 超 threshold 才收敛，事后）区分：本守卫
        # 追踪各策略「手续费率」（total_fees/(total_pnl+total_fees)）跨周期时序，检测连续 window
        # 周期严格上升——交易成本相对盈利持续恶化（换手/频率增加）的事前预警，尚未触及 threshold
        # 即降频收敛。与 cost_awareness（账户级，阈值型）区分：本守卫看单策略粒度的手续费率趋势。
        # 直接对应「消除频繁交易消耗资金」诉求。
        sfrtg_cfg = agi_cfg.get("strategy_fee_ratio_trend_guard", {}) or {}
        self._strategy_fee_ratio_trend_enabled = bool(sfrtg_cfg.get("enabled", False))
        self._strategy_fee_ratio_trend_window = int(safe_float(sfrtg_cfg.get("window"), 3))
        self._strategy_fee_ratio_trend_window = max(2, min(20, self._strategy_fee_ratio_trend_window))
        self._strategy_fee_ratio_trend_reduce_target = safe_float(sfrtg_cfg.get("reduce_target"), 0.1)
        self._strategy_fee_ratio_trend_reduce_target = max(0.0, min(1.0, self._strategy_fee_ratio_trend_reduce_target))

        # ── 策略级资金费率守卫（funding_cost_guard）──
        # 与 strategy_fee_ratio_guard（手续费 = 交易频率成本）区分：本守卫看「资金费率」——
        # OKX 永续合约特有的持仓时间成本。即使不交易，持仓过久（尤其隔夜单边持仓）也会被
        # funding 持续侵蚀，是独立于手续费的持仓侧成本维度。检测单策略资金费占毛利比例
        # （total_funding_cost/(total_pnl+total_fees)）过高（仅 total_pnl>0 且 funding_cost>0
        # 时评估，负资金费=收到资金费不算成本），直接对应「稳定资金增长」诉求。阈值型守卫，
        # 无跨周期状态。
        fcg_cfg = agi_cfg.get("funding_cost_guard", {}) or {}
        self._funding_cost_enabled = bool(fcg_cfg.get("enabled", False))
        self._funding_cost_ratio_threshold = safe_float(fcg_cfg.get("funding_ratio_threshold"), 0.3)
        self._funding_cost_ratio_threshold = max(0.0, min(1.0, self._funding_cost_ratio_threshold))
        self._funding_cost_reduce_target = safe_float(fcg_cfg.get("reduce_target"), 0.1)
        self._funding_cost_reduce_target = max(0.0, min(1.0, self._funding_cost_reduce_target))

        # ── 策略级资金费率趋势外推（funding_cost_trend_guard）──
        # 与 funding_cost_guard（阈值型：funding_ratio 超 threshold 才收敛，事后）区分：本守卫
        # 追踪各策略「资金费率」（total_funding_cost/(total_pnl+total_fees)）跨周期时序，检测连续
        # window 周期严格上升——持仓时间成本相对盈利持续恶化（持仓变久/资金费增加）的事前预警，
        # 尚未触及 threshold 即收敛持仓。与 strategy_fee_ratio_trend_guard（手续费率趋势，看换手）
        # 区分：本守卫看持仓时间成本趋势（funding），即使不交易持仓过久也持续侵蚀。直接对应
        # 「稳定资金增长」诉求。
        fctg_cfg = agi_cfg.get("funding_cost_trend_guard", {}) or {}
        self._funding_cost_trend_enabled = bool(fctg_cfg.get("enabled", False))
        self._funding_cost_trend_window = int(safe_float(fctg_cfg.get("window"), 3))
        self._funding_cost_trend_window = max(2, min(20, self._funding_cost_trend_window))
        self._funding_cost_trend_reduce_target = safe_float(fctg_cfg.get("reduce_target"), 0.1)
        self._funding_cost_trend_reduce_target = max(0.0, min(1.0, self._funding_cost_trend_reduce_target))

        # ── 策略级多空方向失衡守卫（long_short_imbalance_guard）──
        # 与收益质量/风险深度/成本/资本效率各维度守卫区分：本守卫看「方向判断质量」——策略
        # 多空两个方向都有交易，但某一方向（多或空）的累计盈亏为负且绝对亏损占两方向总毛利
        # （|long_pnl|+|short_pnl|）比例过高，说明该方向持续逆势开仓判断错误，直接对应
        # 「只在高位开空、低位开多、趋势确认后才开仓」诉求。阈值型守卫，无跨周期状态。
        # 仅当两个方向交易笔数均 ≥ min_trades 时评估（避免单边样本不足的噪声误触发）。
        lsig_cfg = agi_cfg.get("long_short_imbalance_guard", {}) or {}
        self._long_short_imbalance_enabled = bool(lsig_cfg.get("enabled", False))
        self._long_short_imbalance_min_trades = int(safe_float(lsig_cfg.get("min_trades"), 10))
        self._long_short_imbalance_min_trades = max(1, min(10000, self._long_short_imbalance_min_trades))
        self._long_short_imbalance_loss_ratio = safe_float(lsig_cfg.get("loss_ratio_threshold"), 0.3)
        self._long_short_imbalance_loss_ratio = max(0.0, min(1.0, self._long_short_imbalance_loss_ratio))
        self._long_short_imbalance_leverage = safe_float(lsig_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略方向偏好失衡守卫（strategy_direction_bias_guard）──
        # 与 long_short_imbalance_guard（看「方向盈亏」：多头/空头累计盈亏为负，事后亏损信号）区分：
        # 本守卫看「方向笔数失衡」——策略几乎只做多或只做空（单边笔数占比过高），方向过于单一、
        # 缺乏双向灵活性，是「追涨杀跌」的结构性风险（即使暂时未亏，行情反转时单边敞口无对冲）。
        # 仅在多空总笔数达 min_trades 时评估。阈值型守卫，无跨周期状态。
        sdb_cfg = agi_cfg.get("strategy_direction_bias_guard", {}) or {}
        self._direction_bias_enabled = bool(sdb_cfg.get("enabled", False))
        self._direction_bias_min_trades = int(safe_float(sdb_cfg.get("min_trades"), 10))
        self._direction_bias_min_trades = max(2, min(10000, self._direction_bias_min_trades))
        self._direction_bias_ratio_threshold = safe_float(sdb_cfg.get("bias_ratio_threshold"), 0.9)
        self._direction_bias_ratio_threshold = max(0.5, min(1.0, self._direction_bias_ratio_threshold))
        self._direction_bias_leverage = safe_float(sdb_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级止盈止损比守卫（take_profit_ratio_guard）──
        # 与 stop_loss_frequency_guard（止损率过高，看「入场时机差」）区分：本守卫看「离场质量」
        # ——止盈笔数相对止损笔数过少（take_profit_count/stop_loss_count < 阈值），说明策略「善止损
        # 不善止盈」，赚的时候不落袋、亏的时候才离场，是「稳定资金增长」的另一隐患。仅在有止损单
        # （stop_loss_count>0）且交易笔数达 min_trades 时评估。阈值型守卫，无跨周期状态。
        tprg_cfg = agi_cfg.get("take_profit_ratio_guard", {}) or {}
        self._take_profit_ratio_enabled = bool(tprg_cfg.get("enabled", False))
        self._take_profit_ratio_min_trades = int(safe_float(tprg_cfg.get("min_trades"), 10))
        self._take_profit_ratio_min_trades = max(1, min(10000, self._take_profit_ratio_min_trades))
        self._take_profit_ratio_min_ratio = safe_float(tprg_cfg.get("min_take_profit_ratio"), 0.5)
        self._take_profit_ratio_min_ratio = max(0.0, min(10.0, self._take_profit_ratio_min_ratio))
        self._take_profit_ratio_leverage = safe_float(tprg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级盈亏比守卫（win_loss_ratio_guard）──
        # 与 take_profit_ratio_guard（止盈笔数/止损笔数，看「离场笔数」质量）区分：本守卫看
        # 「单笔盈亏金额结构」——avg_win/avg_loss（平均盈利/平均亏损）低于阈值，说明赚的时候赚得
        # 少、亏的时候亏得多（追涨杀跌的典型盈亏结构：小赚就跑、亏了死扛），是「稳定资金增长」
        # 的另一隐患。仅在有盈亏单（avg_win>0 且 avg_loss>0）且交易笔数达 min_trades 时评估。
        # 阈值型守卫，无跨周期状态。
        wlrg_cfg = agi_cfg.get("win_loss_ratio_guard", {}) or {}
        self._win_loss_ratio_enabled = bool(wlrg_cfg.get("enabled", False))
        self._win_loss_ratio_min_trades = int(safe_float(wlrg_cfg.get("min_trades"), 10))
        self._win_loss_ratio_min_trades = max(1, min(10000, self._win_loss_ratio_min_trades))
        self._win_loss_ratio_min_ratio = safe_float(wlrg_cfg.get("min_win_loss_ratio"), 1.0)
        self._win_loss_ratio_min_ratio = max(0.0, min(10.0, self._win_loss_ratio_min_ratio))
        self._win_loss_ratio_leverage = safe_float(wlrg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级止盈止损盈亏金额比守卫（take_profit_pnl_ratio_guard）──
        # 与 take_profit_ratio_guard（止盈/止损「笔数」比）、win_loss_ratio_guard（平均「单笔」盈利/
        # 亏损比）区分：本守卫看「累计金额」——止盈单累计盈亏（take_profit_pnl）相对止损单累计亏损
        # （stop_loss_pnl 绝对值）过少，说明盈利单轻仓、亏损单重仓（追涨杀跌的仓位管理特征）。仅在
        # 有止盈盈利（take_profit_pnl>0）且有止损亏损（stop_loss_pnl<0）且交易笔数达 min_trades 时
        # 评估。阈值型守卫，无跨周期状态。
        tpprg_cfg = agi_cfg.get("take_profit_pnl_ratio_guard", {}) or {}
        self._take_profit_pnl_ratio_enabled = bool(tpprg_cfg.get("enabled", False))
        self._take_profit_pnl_ratio_min_trades = int(safe_float(tpprg_cfg.get("min_trades"), 10))
        self._take_profit_pnl_ratio_min_trades = max(1, min(10000, self._take_profit_pnl_ratio_min_trades))
        self._take_profit_pnl_ratio_min_ratio = safe_float(tpprg_cfg.get("min_take_profit_pnl_ratio"), 1.0)
        self._take_profit_pnl_ratio_min_ratio = max(0.0, min(10.0, self._take_profit_pnl_ratio_min_ratio))
        self._take_profit_pnl_ratio_leverage = safe_float(tpprg_cfg.get("deterioration_leverage"), 1.0)

        # ── 策略级执行质量成本守卫（execution_cost_guard）──
        # 与 strategy_fee_ratio_guard（手续费 = 交易频率成本）、funding_cost_guard（资金费 = 持仓
        # 时间成本）区分：本守卫看「执行质量成本」——滑点 + 点差（下单执行时点差/滑点损失），
        # 反映流动性、下单时机、市价 vs 限价的执行质量。三者构成交易成本三分类（频率/时间/执行）。
        # 检测单策略执行成本占毛利比例（(slippage+spread)/(total_pnl+total_fees)）过高（仅
        # total_pnl>0 时评估），直接对应「消除追涨杀跌」（追单多用市价单导致高滑点）。阈值型
        # 守卫，无跨周期状态。
        ecg_cfg = agi_cfg.get("execution_cost_guard", {}) or {}
        self._execution_cost_enabled = bool(ecg_cfg.get("enabled", False))
        self._execution_cost_ratio_threshold = safe_float(ecg_cfg.get("cost_ratio_threshold"), 0.3)
        self._execution_cost_ratio_threshold = max(0.0, min(1.0, self._execution_cost_ratio_threshold))
        self._execution_cost_reduce_target = safe_float(ecg_cfg.get("reduce_target"), 0.1)
        self._execution_cost_reduce_target = max(0.0, min(1.0, self._execution_cost_reduce_target))

        # ── 策略级执行成本趋势外推（execution_cost_trend_guard）──
        # 与 execution_cost_guard（阈值型：exec_ratio 超 threshold 才收敛，事后）区分：本守卫
        # 追踪各策略「执行成本率」（(slippage+spread)/(total_pnl+total_fees)）跨周期时序，检测
        # 连续 window 周期严格上升——执行质量持续恶化（滑点/点差相对盈利不断加剧）的事前预警，
        # 尚未触及 threshold 即收敛。与 strategy_fee_ratio_trend_guard（手续费率趋势，看换手频率）、
        # funding_cost_trend_guard（资金费率趋势，看持仓时间）区分：本守卫看执行质量成本趋势
        # （下单执行时点差/滑点损失），反映流动性、下单时机、市价 vs 限价的执行质量恶化。
        # 直接对应「消除追涨杀跌」（追单多用市价单导致高滑点）。
        ectg_cfg = agi_cfg.get("execution_cost_trend_guard", {}) or {}
        self._execution_cost_trend_enabled = bool(ectg_cfg.get("enabled", False))
        self._execution_cost_trend_window = int(safe_float(ectg_cfg.get("window"), 3))
        self._execution_cost_trend_window = max(2, min(20, self._execution_cost_trend_window))
        self._execution_cost_trend_reduce_target = safe_float(ectg_cfg.get("reduce_target"), 0.1)
        self._execution_cost_trend_reduce_target = max(0.0, min(1.0, self._execution_cost_trend_reduce_target))

        # ── 回撤加速预警（drawdown_acceleration_guard）──
        # 跟踪账户回撤跨周期时序，连续加深（每周期都在变差）时生成 drawdown_accelerating 告警，
        # 抑制进攻性加仓。与 equity_status mode（绝对状态）、near_drawdown_limit（接近约束）、
        # EquityMonitor drawdown_velocity（短窗口速率）互补：本 guard 是中周期轨迹维度的先行指标。
        dag_cfg = agi_cfg.get("drawdown_acceleration_guard", {}) or {}
        self._drawdown_accel_enabled = bool(dag_cfg.get("enabled", False))
        self._drawdown_accel_window = int(safe_float(dag_cfg.get("window"), 3))
        self._drawdown_accel_window = max(2, min(100, self._drawdown_accel_window))
        self._drawdown_accel_reduce_target = safe_float(dag_cfg.get("reduce_target"), 0.1)
        self._drawdown_accel_reduce_target = max(0.0, min(1.0, self._drawdown_accel_reduce_target))

        # ── 连续下跌收敛（downside_momentum_guard，momentum_guard 的镜像）──
        # momentum_guard 用 consecutive_up（连续上涨）抑制追涨；本 guard 用 consecutive_down
        # （连续下跌周期数）触发「杀跌」防御——权益持续失血时主动降杠杆收敛 + 暂停进攻（不抄底），
        # 对应用户「消除追涨杀跌」。与 drawdown_acceleration_guard 区分：后者看回撤深度时序加速，
        # 本 guard 看连续下跌周期的持续性（绝对值）。
        dmg_cfg = agi_cfg.get("downside_momentum_guard", {}) or {}
        self._downside_momentum_enabled = bool(dmg_cfg.get("enabled", False))
        self._downside_momentum_max_consecutive_down = int(safe_float(dmg_cfg.get("max_consecutive_down"), 3))
        self._downside_momentum_max_consecutive_down = max(1, min(100, self._downside_momentum_max_consecutive_down))
        self._downside_momentum_reduce_target = safe_float(dmg_cfg.get("reduce_target"), 0.1)
        self._downside_momentum_reduce_target = max(0.0, min(1.0, self._downside_momentum_reduce_target))

        # ── 权益状态机主动收敛（equity_mode_guard）──
        # equity_decline（DECLINE 衰退状态）/equity_emergency（EMERGENCY 紧急状态）告警此前仅
        # 由 EquityMonitor 生成（emergency 冻结新仓、decline 仅标 severity），orchestrator 侧
        # 只告警不动作。本守卫补足主动收敛侧：emergency → 全面降仓到 emergency_reduce_target
        # （默认0，清空敞口）；decline → 谨慎降仓到 decline_reduce_target（默认0.1）。降仓对象取
        # _used_margin_by_strategy() 中保证金占用最多（回退到 target_weight 最高）的策略。
        emg_cfg = agi_cfg.get("equity_mode_guard", {}) or {}
        self._equity_mode_enabled = bool(emg_cfg.get("enabled", False))
        self._equity_mode_emergency_reduce_target = safe_float(emg_cfg.get("emergency_reduce_target"), 0.0)
        self._equity_mode_emergency_reduce_target = max(0.0, min(1.0, self._equity_mode_emergency_reduce_target))
        self._equity_mode_decline_reduce_target = safe_float(emg_cfg.get("decline_reduce_target"), 0.1)
        self._equity_mode_decline_reduce_target = max(0.0, min(1.0, self._equity_mode_decline_reduce_target))

        self.regime_engine = regime_engine
        # RegimeArbiter：融合主引擎+检测器输出，None 时 _perceive 透明回退 regime_engine
        self.regime_arbiter = regime_arbiter
        self.contribution_analyzer = contribution_analyzer
        self.capital_allocator = capital_allocator
        self.dynamic_allocator = dynamic_allocator
        self.account_manager = account_manager
        self.equity_monitor = equity_monitor
        self.strategy_correlation = strategy_correlation
        self.rl_agent = rl_agent

        # 状态
        self._cycle_count = 0
        self._last_run_ts: Optional[float] = None
        self._last_report: Optional[Dict[str, Any]] = None
        self._last_regime: Optional[str] = None
        self._last_health_score: Optional[float] = None  # 跨周期时序：上一轮健康度
        self._last_snapshot = None  # 本周期贡献快照（原始对象），供决策复用，避免重复 analyze 口径不一致

        # 逐币种精细杠杆守卫：{symbol: 上次下调杠杆的周期号}，冷却期内不重复下调。
        self._symbol_leverage_cooldown: Dict[str, int] = {}

        # 目标导向规划运行时状态：收益基准（首次权益/初始资本）与峰值权益，
        # 用于计算「收益达成度」与「距最大回撤约束的距离」。
        self._goal_baseline_equity: Optional[float] = None
        self._goal_peak_equity: float = 0.0

        # 决策溯源历史（有界 FIFO，跨重启续用）
        self._decision_lineage: deque = deque(maxlen=self._lineage_max_entries)

        # 权益轨迹（有界 FIFO，供自适应风险偏好计算权益斜率）
        self._equity_window: deque = deque(maxlen=self._adaptive_risk_window)

        # 本周期动作冲突消解审计结果（kept/dropped），供报告与决策溯源回填
        self._last_reconciliation: Optional[Dict[str, Any]] = None

        # 本周期决策置信度门控结果（低置信进攻动作被收敛），供报告与决策溯源回填
        self._last_confidence_gate: Optional[Dict[str, Any]] = None

        # 本周期时间尺度（slow/strategic 或 fast/tactical），供报告与决策溯源回填
        self._last_timescale: str = "fast"

        # AGI 已暂停（生成过 strategy_pause）的策略集合，供健康度改善后自主恢复
        self._paused_strategies: set = set()

        # 上一周期动作的执行结果（deployed/queued/rejected/notified），由 scheduler 在
        # route 后通过 report_execution_result 回调，供本周期诊断感知（决策→执行→反馈闭环）
        self._last_execution_result: Optional[Dict[str, Any]] = None
        self._execution_memory: deque = deque(maxlen=50)

        # 各策略上次进攻加仓的周期号（进攻观察期冷却：防连续追高加仓）
        self._last_offensive_cycle: Dict[str, int] = {}

        # 各策略上次参数自适应调整的周期号（param_adaptation 限速冷却：
        # 防过度迭代杠杆，让参数沉淀验证后再调整）
        self._param_adapt_last_cycle: Dict[str, int] = {}

        # 各策略上次进攻时的累计盈亏基线（决策效果归因：检测加仓后是否转差）
        self._offensive_attribution: Dict[str, float] = {}

        # 止损后冷却期：记录各策略上次止损周期，cooldown_cycles 内禁止重新进攻
        # （刚亏了钱不加仓，避免频繁交易消耗资金）
        self._offensive_stop_loss_cooldown: Dict[str, int] = {}

        # 止盈后冷却期：记录各策略上次止盈周期，cooldown_cycles 内禁止重新进攻
        # （刚止盈就再进同样是追涨、频繁交易消耗资金）
        self._offensive_profit_take_cooldown: Dict[str, int] = {}

        # 进攻归因周期过期（TTL）：记录各策略归因基线写入时的周期号，
        # 超过 attribution_ttl_cycles 未了结则自动过期清除，避免陈旧基线长期阻塞重新进攻
        self._offensive_attribution_cycle: Dict[str, int] = {}

        # 进攻强化学习反馈分数：各策略进攻了结结果的 EMA（止盈→+1/止损→-1），
        # 供下次进攻时缩放 boost_step（正分→加大、负分→缩小），形成学习闭环
        self._offensive_feedback_score: Dict[str, float] = {}

        # 连续进攻次数：记录各策略「无了结」连续进攻加仓次数，
        # 达到 max_consecutive_offenses 上限后暂停（防止一路追高加仓）
        self._offensive_consecutive: Dict[str, int] = {}

        # 本周期策略相关性快照（average/max_pair correlation、有效 N 侵蚀），供报告审计
        self._last_correlation: Dict[str, Any] = {}

        # 本周期成本意识调仓门控结果（微小调仓被丢弃），供报告与决策溯源回填
        self._last_cost_guard: Optional[Dict[str, Any]] = None

        # 各策略 health_score 跨周期时序（供健康度趋势外推：连续下降 → 事前降杠杆）
        self._strategy_health_history: Dict[str, deque] = {}

        # 上次高置信度市场状态突变的周期号（供进攻冷却：突变后暂停进攻等趋势确认）
        self._last_regime_shift_cycle: Optional[int] = None

        # 各策略累计盈亏峰值（供利润回吐保护：浮盈从峰值回吐超阈值 → 减仓锁利）
        self._strategy_pnl_peak: Dict[str, float] = {}

        # 账户回撤跨周期时序（供回撤加速预警：连续加深 → 抑制进攻，先行指标）
        self._drawdown_history: deque = deque(maxlen=self._drawdown_accel_window)

        # 组合 overall_health_score 跨周期时序（供组合级健康度趋势外推：连续下降 → 收敛组合）
        self._portfolio_health_history: deque = deque(maxlen=self._portfolio_health_trend_window)

        # 组合 concentration（HHI）跨周期时序（供组合级集中度趋势外推：连续上升 → 收敛组合）
        self._portfolio_concentration_history: deque = deque(maxlen=self._portfolio_concentration_trend_window)

        # 组合 max_pair_correlation 跨周期时序（供组合级相关性趋势外推：连续上升 → 收敛组合）
        self._portfolio_correlation_history: deque = deque(maxlen=self._portfolio_correlation_trend_window)

        # 组合 tail_risk（max(max_drawdown_i)）跨周期时序（供组合级尾部风险趋势外推：连续上升 → 收敛组合）
        self._portfolio_tail_risk_history: deque = deque(maxlen=self._portfolio_tail_risk_trend_window)

        # 组合 profit_concentration（max_pnl/total_pnl）跨周期时序（供组合级盈利集中度趋势外推：连续上升 → 收敛主导盈利策略）
        self._portfolio_profit_concentration_history: deque = deque(maxlen=self._profit_concentration_trend_window)

        # 组合 synergy_score 跨周期时序（供组合级协同度趋势外推：连续下降 → 收敛组合）
        self._portfolio_synergy_history: deque = deque(maxlen=self._synergy_trend_window)

        # 组合 efficiency_score 跨周期时序（供组合级资金效率趋势外推：连续下降 → 收敛组合）
        self._portfolio_efficiency_history: deque = deque(maxlen=self._portfolio_efficiency_trend_window)

        # 资金利用率（used_margin/equity）跨周期时序（供利用率趋势外推：连续上升 → 收敛敞口）
        self._utilization_history: deque = deque(maxlen=self._utilization_trend_window)

        # 各策略 win_rate 跨周期时序（供胜率趋势外推：连续下降 → 事前降杠杆）
        self._strategy_win_rate_history: Dict[str, deque] = {}

        # 各策略 sharpe_ratio 跨周期时序（供夏普趋势外推：连续下降 → 事前降杠杆）
        self._strategy_sharpe_history: Dict[str, deque] = {}

        # 各策略 profit_factor 跨周期时序（供盈亏比趋势外推：连续下降 → 事前降杠杆）
        self._strategy_profit_factor_history: Dict[str, deque] = {}

        # 各策略 max_drawdown 跨周期时序（供最大回撤趋势外推：连续加深 → 事前降杠杆）
        self._strategy_max_drawdown_history: Dict[str, deque] = {}

        # 各策略 max_drawdown_duration_hours 跨周期时序（供回撤持续时间趋势外推：连续拉长 → 事前降杠杆）
        self._strategy_drawdown_duration_history: Dict[str, deque] = {}

        # 各策略 pnl_per_capital_pct 跨周期时序（供资本回报率趋势外推：连续下降 → 事前降杠杆）
        self._strategy_capital_return_history: Dict[str, deque] = {}

        # 各策略 volatility 跨周期时序（供波动率趋势外推：连续上升 → 事前降杠杆）
        self._strategy_volatility_history: Dict[str, deque] = {}

        # 各策略 trend_pnl_7d_vs_30d 跨周期时序（供 PnL 动量趋势外推：连续下降 → 事前降杠杆）
        self._strategy_pnl_momentum_history: Dict[str, deque] = {}

        # 各策略 pnl_per_trade 跨周期时序（供单笔期望值趋势外推：连续下降 → 事前降杠杆）
        self._strategy_pnl_per_trade_history: Dict[str, deque] = {}

        # 各策略 delta_pnl 跨周期时序（供边际盈亏趋势外推：连续为负 → 事前降杠杆）
        self._strategy_delta_pnl_history: Dict[str, deque] = {}

        # 各策略 unrealized_pnl 跨周期时序（供浮亏加深趋势外推：连续下降 → 提前止损）
        self._strategy_unrealized_loss_history: Dict[str, deque] = {}

        # 各策略 浮盈占比（unrealized_pnl/total_pnl）跨周期时序（供浮盈占比趋势外推：连续上升 → 锁定浮盈）
        self._strategy_unrealized_profit_ratio_history: Dict[str, deque] = {}

        # 各策略 手续费率（total_fees/(total_pnl+total_fees)）跨周期时序（供手续费率趋势外推：连续上升 → 降频收敛）
        self._strategy_fee_ratio_history: Dict[str, deque] = {}

        # 各策略 资金费率（total_funding_cost/(total_pnl+total_fees)）跨周期时序（供资金费率趋势外推：连续上升 → 收敛持仓）
        self._strategy_funding_cost_history: Dict[str, deque] = {}

        # 各策略 执行成本率（(slippage+spread)/(total_pnl+total_fees)）跨周期时序（供执行成本趋势外推：连续上升 → 收敛）
        self._strategy_execution_cost_history: Dict[str, deque] = {}

        # 推算准确度跨周期时序（供下一周期 _project_pnl 的 correction_factor 计算）
        self._projection_accuracy: deque = deque(maxlen=self._projection_accuracy_window)
        # 上一周期推算结果（供本周期 _track_projection_accuracy 对比实际）
        self._last_projection: Optional[Dict[str, Any]] = None
        # 当前校正系数（由 _track_projection_accuracy 写入，供 _project_pnl 读取）
        self._correction_factor: float = 1.0
        # 按 regime 分桶的校正系数（条件化准确度：不同 regime 推算偏差不同）
        # regime → correction；缺失/过期 regime 时回退到全局 _correction_factor
        self._correction_factor_by_regime: Dict[str, float] = {}
        self._correction_factor_by_regime_updated_cycle: Dict[str, int] = {}

        logger.info(
            f"QuantAGIOrchestrator initialized: cooldown={self.cooldown_seconds:.0f}s, "
            f"regime_engine={'on' if self.regime_engine else 'off'}, "
            f"contribution_analyzer={'on' if self.contribution_analyzer else 'off'}, "
            f"capital_allocator={'on' if self.capital_allocator else 'off'}, "
            f"dynamic_allocator={'on' if self.dynamic_allocator else 'off'}, "
            f"profit_take={'on' if self._profit_take_enabled else 'off'}"
            f"(act={self._profit_take_activation_pct:.1%},max={self._profit_take_max_pct:.1%},"
            f"close<= {self._profit_take_max_close_ratio:.0%}), "
            f"risk_response={'on' if self._risk_response_enabled else 'off'}"
            f"(reduce_to={self._risk_response_reduce_target:.0%}), "
            f"regime_adaptive={'on' if self._regime_adaptive_enabled else 'off'}"
            f"(trend={self._trend_profit_take_mult:.1f}x,range={self._range_profit_take_mult:.1f}x), "
            f"learning={'on' if self._learning_enabled else 'off'}"
            f"(mem={self._learning_memory_size},max_adj={self._learning_max_adjust_pct:.0%}), "
            f"goal_planning={'on' if self._goal_planning_enabled else 'off'}"
            f"(target={self._goal_daily_target_pct:.1%},max_dd={self._goal_max_drawdown_pct:.1%}), "
            f"param_adaptation={'on' if self._param_adaptation_enabled else 'off'}"
        )

        # 跨周期学习记忆：从 state 文件恢复历史决策（跨重启续用）
        self._load_learning_memory()

        # 已暂停策略集合：从 state 文件恢复（跨重启续用，供健康度改善后自主恢复）
        self._load_paused_strategies()

        # 决策溯源历史：从 lineage 文件恢复（跨重启续用）
        self._load_decision_lineage()

        # 学习型状态（进攻归因基线/利润峰值/健康度时序/突变冷却）：从 state 文件恢复（跨重启续用）
        self._load_learning_state()

    def _load_learning_state(self) -> None:
        """从 state 文件恢复学习型状态（进攻归因基线 / 观察期冷却 / 止损止盈冷却 /
        利润峰值 / 健康度时序 / 突变冷却），使「从自身决策结果中学习」跨重启续用，
        不因重启失忆——尤其止损/止盈后冷却期跨重启保留，防止「刚亏就加」因重启而失效。"""
        try:
            if not os.path.exists(self.state_path):
                return
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            ls = state.get("learning_state")
            if not isinstance(ls, dict):
                return
            self._cycle_count = max(
                self._cycle_count,
                safe_int(ls.get("cycle_count", state.get("cycle")), 0),
            )
            self._offensive_attribution = {
                str(k): safe_float(v, 0.0)
                for k, v in (ls.get("offensive_attribution") or {}).items()
            }
            self._last_offensive_cycle = {
                str(k): int(safe_float(v, 0.0))
                for k, v in (ls.get("last_offensive_cycle") or {}).items()
            }
            self._param_adapt_last_cycle = {
                str(k): int(safe_float(v, 0.0))
                for k, v in (ls.get("param_adapt_last_cycle") or {}).items()
            }
            self._offensive_stop_loss_cooldown = {
                str(k): int(safe_float(v, 0.0))
                for k, v in (ls.get("offensive_stop_loss_cooldown") or {}).items()
            }
            self._offensive_profit_take_cooldown = {
                str(k): int(safe_float(v, 0.0))
                for k, v in (ls.get("offensive_profit_take_cooldown") or {}).items()
            }
            self._offensive_attribution_cycle = {
                str(k): int(safe_float(v, 0.0))
                for k, v in (ls.get("offensive_attribution_cycle") or {}).items()
            }
            self._offensive_consecutive = {
                str(k): int(safe_float(v, 0.0))
                for k, v in (ls.get("offensive_consecutive") or {}).items()
            }
            self._offensive_feedback_score = {
                str(k): safe_float(v, 0.0)
                for k, v in (ls.get("offensive_feedback_score") or {}).items()
            }
            # 恢复决策质量单周期增量基线（无则保持 None，首周期 cycle_pnl=0）
            _ldtp = ls.get("last_decision_total_pnl")
            if _ldtp is not None:
                _val = safe_float(_ldtp, None)
                # safe_finite 仅接受有限数值，NaN/Inf 回退 None
                self._last_decision_total_pnl = safe_finite(_val, None) if _val is not None else None
            self._strategy_pnl_peak = {
                str(k): safe_float(v, 0.0)
                for k, v in (ls.get("strategy_pnl_peak") or {}).items()
            }
            self._give_back_peak_last_decay_cycle = {
                str(k): int(safe_float(v, 0.0))
                for k, v in (ls.get("give_back_peak_last_decay_cycle") or {}).items()
            }
            for k, v in (ls.get("strategy_health_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_health_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._health_trend_window,
                    )
            for k, v in (ls.get("strategy_win_rate_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_win_rate_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._win_rate_trend_window,
                    )
            for k, v in (ls.get("strategy_sharpe_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_sharpe_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._sharpe_trend_window,
                    )
            for k, v in (ls.get("strategy_profit_factor_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_profit_factor_history[str(k)] = deque(
                        [safe_float(x, 1.0) for x in v],
                        maxlen=self._profit_factor_trend_window,
                    )
            for k, v in (ls.get("strategy_max_drawdown_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_max_drawdown_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._max_drawdown_trend_window,
                    )
            for k, v in (ls.get("strategy_drawdown_duration_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_drawdown_duration_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._drawdown_duration_trend_window,
                    )
            for k, v in (ls.get("strategy_capital_return_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_capital_return_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._capital_return_trend_window,
                    )
            for k, v in (ls.get("strategy_volatility_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_volatility_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._volatility_trend_window,
                    )
            for k, v in (ls.get("strategy_pnl_momentum_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_pnl_momentum_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._pnl_momentum_trend_window,
                    )
            for k, v in (ls.get("strategy_pnl_per_trade_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_pnl_per_trade_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._pnl_per_trade_trend_window,
                    )
            for k, v in (ls.get("strategy_delta_pnl_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_delta_pnl_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._delta_pnl_trend_window,
                    )
            for k, v in (ls.get("strategy_unrealized_loss_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_unrealized_loss_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._unrealized_loss_trend_window,
                    )
            for k, v in (ls.get("strategy_unrealized_profit_ratio_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_unrealized_profit_ratio_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._unrealized_profit_ratio_trend_window,
                    )
            for k, v in (ls.get("strategy_fee_ratio_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_fee_ratio_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._strategy_fee_ratio_trend_window,
                    )
            for k, v in (ls.get("strategy_funding_cost_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_funding_cost_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._funding_cost_trend_window,
                    )
            for k, v in (ls.get("strategy_execution_cost_history") or {}).items():
                if isinstance(v, list):
                    self._strategy_execution_cost_history[str(k)] = deque(
                        [safe_float(x, 0.0) for x in v],
                        maxlen=self._execution_cost_trend_window,
                    )
            rs = ls.get("last_regime_shift_cycle")
            self._last_regime_shift_cycle = (
                int(safe_float(rs, 0.0)) if rs is not None else None
            )
            dh = ls.get("drawdown_history")
            if isinstance(dh, list):
                self._drawdown_history = deque(
                    [safe_float(x, 0.0) for x in dh],
                    maxlen=self._drawdown_accel_window,
                )
            ph = ls.get("portfolio_health_history")
            if isinstance(ph, list):
                self._portfolio_health_history = deque(
                    [safe_float(x, 0.0) for x in ph],
                    maxlen=self._portfolio_health_trend_window,
                )
            pch = ls.get("portfolio_concentration_history")
            if isinstance(pch, list):
                self._portfolio_concentration_history = deque(
                    [safe_float(x, 0.0) for x in pch],
                    maxlen=self._portfolio_concentration_trend_window,
                )
            pcorr = ls.get("portfolio_correlation_history")
            if isinstance(pcorr, list):
                self._portfolio_correlation_history = deque(
                    [safe_float(x, 0.0) for x in pcorr],
                    maxlen=self._portfolio_correlation_trend_window,
                )
            ptr = ls.get("portfolio_tail_risk_history")
            if isinstance(ptr, list):
                self._portfolio_tail_risk_history = deque(
                    [safe_float(x, 0.0) for x in ptr],
                    maxlen=self._portfolio_tail_risk_trend_window,
                )
            ppc = ls.get("portfolio_profit_concentration_history")
            if isinstance(ppc, list):
                self._portfolio_profit_concentration_history = deque(
                    [safe_float(x, 0.0) for x in ppc],
                    maxlen=self._profit_concentration_trend_window,
                )
            psy = ls.get("portfolio_synergy_history")
            if isinstance(psy, list):
                self._portfolio_synergy_history = deque(
                    [safe_float(x, 0.0) for x in psy],
                    maxlen=self._synergy_trend_window,
                )
            pef = ls.get("portfolio_efficiency_history")
            if isinstance(pef, list):
                self._portfolio_efficiency_history = deque(
                    [safe_float(x, 0.0) for x in pef],
                    maxlen=self._portfolio_efficiency_trend_window,
                )
            uh = ls.get("utilization_history")
            if isinstance(uh, list):
                self._utilization_history = deque(
                    [safe_float(x, 0.0) for x in uh],
                    maxlen=self._utilization_trend_window,
                )
            pa = ls.get("projection_accuracy")
            if isinstance(pa, list):
                self._projection_accuracy = deque(
                    [
                        {
                            "cycle": int(x.get("cycle", 0)) if isinstance(x, dict) else 0,
                            "projected": safe_float(x.get("projected"), 0.0) if isinstance(x, dict) else 0.0,
                            "actual": safe_float(x.get("actual"), 0.0) if isinstance(x, dict) else 0.0,
                            "bias_ratio": safe_float(x.get("bias_ratio"), 1.0) if isinstance(x, dict) else 1.0,
                            "regime": _canonical_projection_regime(
                                x.get("regime", "unknown") if isinstance(x, dict) else "unknown"
                            ),
                        }
                        for x in pa
                    ],
                    maxlen=self._projection_accuracy_window,
                )
            lp = ls.get("last_projection")
            self._last_projection = dict(lp) if isinstance(lp, dict) else None
            cf = safe_float(ls.get("correction_factor"), 1.0)
            self._correction_factor = max(self._projection_min_correction,
                                          min(self._projection_max_correction, cf))
            # regime 分桶 correction 恢复
            cfbr = ls.get("correction_factor_by_regime")
            cfbr_updated = ls.get("correction_factor_by_regime_updated_cycle")
            if isinstance(cfbr, dict):
                normalized_corrections: Dict[str, List[float]] = {}
                for key, value in cfbr.items():
                    canonical_key = _canonical_projection_regime(key)
                    normalized_corrections.setdefault(canonical_key, []).append(
                        safe_float(value, 1.0)
                    )
                self._correction_factor_by_regime = {
                    key: max(
                        self._projection_min_correction,
                        min(
                            self._projection_max_correction,
                            sum(values) / len(values),
                        ),
                    )
                    for key, values in normalized_corrections.items()
                }
                if isinstance(cfbr_updated, dict):
                    normalized_updated_cycles: Dict[str, List[int]] = {}
                    for key, value in cfbr_updated.items():
                        canonical_key = _canonical_projection_regime(key)
                        normalized_updated_cycles.setdefault(canonical_key, []).append(
                            safe_int(value, -1)
                        )
                    self._correction_factor_by_regime_updated_cycle = {
                        key: max(cycles)
                        for key, cycles in normalized_updated_cycles.items()
                        if cycles and max(cycles) >= 0
                    }
                else:
                    # Legacy state can recover freshness from the retained accuracy samples.
                    recovered_update_cycles = {
                        regime: max(
                            (
                                safe_int(sample.get("cycle"), -1)
                                for sample in self._projection_accuracy
                                if _canonical_projection_regime(sample.get("regime")) == regime
                            ),
                            default=-1,
                        )
                        for regime in self._correction_factor_by_regime
                    }
                    self._correction_factor_by_regime_updated_cycle = {
                        regime: cycle
                        for regime, cycle in recovered_update_cycles.items()
                        if cycle >= 0
                    }
            execution_memory = ls.get("execution_memory")
            if isinstance(execution_memory, list):
                self._execution_memory = deque(
                    [item for item in execution_memory if isinstance(item, dict)][-50:],
                    maxlen=50,
                )
                if self._execution_memory:
                    self._last_execution_result = copy.deepcopy(
                        self._execution_memory[-1].get("result")
                    )
            sks = ls.get("simulation_kill_switch")
            if isinstance(sks, dict) and sks.get("enabled"):
                self._simulation_kill_switch_enabled = True
                self._simulation_kill_switch_reason = str(sks.get("reason", "persisted_from_previous_session"))
                self._simulation_fail_closed_streak = safe_int(sks.get("fail_closed_streak"), 0)
                logger.warning(
                    f"[AGI-Simulation] simulated Kill Switch restored from persistence: "
                    f"reason={self._simulation_kill_switch_reason}, streak={self._simulation_fail_closed_streak}"
                )
            ccb = ls.get("cycle_circuit_breaker")
            if isinstance(ccb, dict):
                self._cycle_failure_streak = safe_int(ccb.get("failure_streak"), 0)
                if ccb.get("halted"):
                    self._cycle_halted = True
                    self._cycle_halt_reason = str(ccb.get("halt_reason", "consecutive_cycle_failures"))
                    logger.warning(
                        f"[AGI] cycle circuit breaker restored: halted=True, "
                        f"reason={self._cycle_halt_reason}, streak={self._cycle_failure_streak}"
                    )
            logger.info("[AGI] learning state restored")
        except Exception as e:
            logger.debug(f"[AGI] load learning state failed: {e}")

    def _serialize_learning_state(self) -> Dict[str, Any]:
        """把学习型状态序列化为 JSON 安全的 dict（deque → list，数值 safe_finite）。"""
        return {
            "offensive_attribution": {
                str(k): safe_finite(v, 0.0) for k, v in self._offensive_attribution.items()
            },
            "last_offensive_cycle": {
                str(k): int(v) for k, v in self._last_offensive_cycle.items()
            },
            "param_adapt_last_cycle": {
                str(k): int(v) for k, v in self._param_adapt_last_cycle.items()
            },
            "offensive_stop_loss_cooldown": {
                str(k): int(v) for k, v in self._offensive_stop_loss_cooldown.items()
            },
            "offensive_profit_take_cooldown": {
                str(k): int(v) for k, v in self._offensive_profit_take_cooldown.items()
            },
            "offensive_attribution_cycle": {
                str(k): int(v) for k, v in self._offensive_attribution_cycle.items()
            },
            "offensive_consecutive": {
                str(k): int(v) for k, v in self._offensive_consecutive.items()
            },
            "offensive_feedback_score": {
                str(k): safe_finite(v, 0.0)
                for k, v in self._offensive_feedback_score.items()
            },
            # 决策质量单周期增量基线：跨重启保留，避免重启后首个周期 cycle_pnl=0 丢失连续性
            "last_decision_total_pnl": (
                safe_finite(self._last_decision_total_pnl, None)
                if self._last_decision_total_pnl is not None else None
            ),
            "strategy_pnl_peak": {
                str(k): safe_finite(v, 0.0) for k, v in self._strategy_pnl_peak.items()
            },
            "give_back_peak_last_decay_cycle": {
                str(k): int(v) for k, v in self._give_back_peak_last_decay_cycle.items()
            },
            "strategy_health_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_health_history.items()
            },
            "strategy_win_rate_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_win_rate_history.items()
            },
            "strategy_sharpe_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_sharpe_history.items()
            },
            "strategy_profit_factor_history": {
                str(k): [safe_finite(x, 1.0) for x in v]
                for k, v in self._strategy_profit_factor_history.items()
            },
            "strategy_max_drawdown_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_max_drawdown_history.items()
            },
            "strategy_drawdown_duration_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_drawdown_duration_history.items()
            },
            "strategy_capital_return_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_capital_return_history.items()
            },
            "strategy_volatility_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_volatility_history.items()
            },
            "strategy_pnl_momentum_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_pnl_momentum_history.items()
            },
            "strategy_pnl_per_trade_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_pnl_per_trade_history.items()
            },
            "strategy_delta_pnl_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_delta_pnl_history.items()
            },
            "strategy_unrealized_loss_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_unrealized_loss_history.items()
            },
            "strategy_unrealized_profit_ratio_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_unrealized_profit_ratio_history.items()
            },
            "strategy_fee_ratio_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_fee_ratio_history.items()
            },
            "strategy_funding_cost_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_funding_cost_history.items()
            },
            "strategy_execution_cost_history": {
                str(k): [safe_finite(x, 0.0) for x in v]
                for k, v in self._strategy_execution_cost_history.items()
            },
            "last_regime_shift_cycle": self._last_regime_shift_cycle,
            "drawdown_history": [safe_finite(x, 0.0) for x in self._drawdown_history],
            "portfolio_health_history": [safe_finite(x, 0.0) for x in self._portfolio_health_history],
            "portfolio_concentration_history": [safe_finite(x, 0.0) for x in self._portfolio_concentration_history],
            "portfolio_correlation_history": [safe_finite(x, 0.0) for x in self._portfolio_correlation_history],
            "portfolio_tail_risk_history": [safe_finite(x, 0.0) for x in self._portfolio_tail_risk_history],
            "portfolio_profit_concentration_history": [safe_finite(x, 0.0) for x in self._portfolio_profit_concentration_history],
            "portfolio_synergy_history": [safe_finite(x, 0.0) for x in self._portfolio_synergy_history],
            "portfolio_efficiency_history": [safe_finite(x, 0.0) for x in self._portfolio_efficiency_history],
            "utilization_history": [safe_finite(x, 0.0) for x in self._utilization_history],
            "projection_accuracy": [
                {
                    "cycle": int(x.get("cycle", 0)),
                    "projected": safe_finite(x.get("projected"), 0.0),
                    "actual": safe_finite(x.get("actual"), 0.0),
                    "bias_ratio": safe_finite(x.get("bias_ratio"), 1.0),
                    "regime": str(x.get("regime", "unknown")),
                }
                for x in self._projection_accuracy
            ],
            "last_projection": (
                None if self._last_projection is None
                else {
                    "total": self._last_projection.get("total"),
                    "projection_cycle": self._last_projection.get("projection_cycle"),
                    "horizon_cycles": self._last_projection.get("horizon_cycles"),
                }
            ),
            "correction_factor": safe_finite(self._correction_factor, 1.0),
            "correction_factor_by_regime": {
                str(k): safe_finite(v, 1.0)
                for k, v in self._correction_factor_by_regime.items()
            },
            "correction_factor_by_regime_updated_cycle": {
                str(k): int(v)
                for k, v in self._correction_factor_by_regime_updated_cycle.items()
            },
            "cycle_count": int(self._cycle_count),
            "execution_memory": [
                copy.deepcopy(item) for item in self._execution_memory
            ],
            "simulation_kill_switch": {
                "enabled": self._simulation_kill_switch_enabled,
                "reason": self._simulation_kill_switch_reason,
                "fail_closed_streak": int(self._simulation_fail_closed_streak),
            },
            "cycle_circuit_breaker": {
                "failure_streak": int(self._cycle_failure_streak),
                "halted": self._cycle_halted,
                "halt_reason": self._cycle_halt_reason,
            },
        }

    def _load_paused_strategies(self) -> None:
        """从 state 文件恢复 AGI 已暂停策略集合（跨重启续用）。"""
        try:
            if not os.path.exists(self.state_path):
                return
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            paused = state.get("paused_strategies")
            if isinstance(paused, list):
                self._paused_strategies = {str(s) for s in paused if isinstance(s, str)}
                if self._paused_strategies:
                    logger.info(
                        f"[AGI] paused strategies restored: {sorted(self._paused_strategies)}"
                    )
        except Exception as e:
            logger.debug(f"[AGI] load paused strategies failed: {e}")


    def _load_learning_memory(self) -> None:
        """从 state 文件恢复决策记忆（跨重启续用学习状态）。"""
        if not self._learning_enabled:
            return
        try:
            if not os.path.exists(self.state_path):
                return
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            mem = state.get("decision_memory")
            if isinstance(mem, list):
                self._decision_memory = deque(
                    [m for m in mem if isinstance(m, dict)][-self._learning_memory_size:],
                    maxlen=self._learning_memory_size,
                )
                if self._decision_memory:
                    logger.info(
                        f"[AGI] learning memory restored: {len(self._decision_memory)} entries"
                    )
        except Exception as e:
            logger.debug(f"[AGI] load learning memory failed: {e}")

    def _load_decision_lineage(self) -> None:
        """从 lineage 文件恢复决策溯源历史（跨重启续用）。"""
        if not self._lineage_enabled:
            return
        try:
            if not os.path.exists(self._lineage_path):
                return
            with open(self._lineage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                self._decision_lineage = deque(
                    [d for d in data if isinstance(d, dict)][-self._lineage_max_entries:],
                    maxlen=self._lineage_max_entries,
                )
        except Exception as e:
            logger.debug(f"[AGI] load decision lineage failed: {e}")

    def _save_decision_lineage(self) -> None:
        """持久化决策溯源历史到 lineage 文件（fail-closed：失败仅记日志）。"""
        if not self._lineage_enabled:
            return
        try:
            dirname = os.path.dirname(self._lineage_path)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            with open(self._lineage_path, "w", encoding="utf-8") as f:
                json.dump(list(self._decision_lineage), f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.debug(f"[AGI] save decision lineage failed: {e}")

    @staticmethod
    def _guard_from_reason(reason: Any) -> Optional[str]:
        """从动作 reason 提取触发守卫名（reason 形如 "guard_name: message"），供可解释审计。"""
        if not reason:
            return None
        return str(reason).split(":")[0].strip() or None

    def _append_decision_lineage(self, report: Dict[str, Any]) -> None:
        """把本周期决策摘要追加到溯源历史（输入快照 + rationale + 动作清单）。"""
        if not self._lineage_enabled:
            return
        try:
            decision = report.get("decision") or {}
            reflection = report.get("reflection") or {}
            perception = report.get("perception") or {}
            entry = {
                "decision_id": report.get("decision_id"),
                "cycle": report.get("cycle"),
                "timestamp": report.get("timestamp"),
                "status": report.get("status"),
                "inputs": {
                    "equity": safe_float(perception.get("equity"), 0.0),
                    "regime": (perception.get("market_regime") or {}).get("regime"),
                    "health_grade": reflection.get("health_grade"),
                },
                "rationale": decision.get("rationale") or [],
                "summary": {
                    "alerts": len(report.get("alerts") or []),
                    "actions": len(report.get("actions") or []),
                    "decision_quality": reflection.get("decision_quality"),
                },
                "actions": [
                    {
                        "type": a.get("type"),
                        "strategy": a.get("strategy"),
                        "action": a.get("action"),
                        "param": a.get("param"),
                        "value": a.get("value"),
                        "target_allocation": a.get("target_allocation"),
                        "guard": self._guard_from_reason(a.get("reason")),
                        "reason": a.get("reason"),
                        "confidence": a.get("confidence"),
                        "priority": a.get("priority"),
                        "rationale": a.get("rationale"),
                    }
                    for a in (report.get("actions") or [])
                ],
            }
            self._decision_lineage.append(entry)
            self._save_decision_lineage()
        except Exception as e:
            logger.debug(f"[AGI] append decision lineage failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # 核心闭环入口
    # ─────────────────────────────────────────────────────────────

    async def run_cycle(self) -> Dict[str, Any]:
        """执行一个自治闭环 tick，返回结构化的 decision_report。"""
        now = time.monotonic()

        # 幂等 + 冷却：冷却期内直接返回缓存报告
        if (
            self._last_report is not None
            and self._last_run_ts is not None
            and (now - self._last_run_ts) < self.cooldown_seconds
        ):
            cached = copy.deepcopy(self._last_report)
            cached["status"] = "cooldown"
            cached["cooldown"] = True
            logger.debug(f"[AGI] cooldown active ({self.cooldown_seconds:.0f}s), returning cached report")
            return cached

        self._cycle_count += 1

        # 循环熔断检查：连续失败超过阈值后拒绝执行
        if self._cycle_halted:
            report_halted: Dict[str, Any] = {
                "decision_id": f"agi-halted-{self._cycle_count}",
                "timestamp": datetime.now().isoformat(),
                "cycle": self._cycle_count,
                "status": "halted",
                "cooldown": False,
                "halt_reason": self._cycle_halt_reason,
                "failure_streak": self._cycle_failure_streak,
                "perception": {},
                "diagnosis": {"alerts": []},
                "decision": {"allocation_plan": None, "reallocation_suggestions": [],
                             "market_regime": "unknown", "strategy_names": [], "strategy_metrics": {}},
                "actions": [],
                "projection": {},
                "attribution": {},
                "errors": [{"stage": "circuit_breaker", "type": "CycleHalted",
                            "message": self._cycle_halt_reason}],
                "reflection": {"health_score": 0.0, "health_grade": "HALTED",
                               "alerts_count": 0, "cycle_summary": self._cycle_halt_reason},
            }
            return report_halted

        decision_id = f"agi-dec-{self._cycle_count}-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        report: Dict[str, Any] = {
            "decision_id": decision_id,
            "timestamp": datetime.now().isoformat(),
            "cycle": self._cycle_count,
            "status": "ok",
            "cooldown": False,
            "perception": {},
            "diagnosis": {"alerts": []},
            "decision": {
                "allocation_plan": None,
                "reallocation_suggestions": [],
                "market_regime": "unknown",
                "strategy_names": [],
                "strategy_metrics": {},
            },
            "actions": [],
            "projection": {},
            "attribution": {},
            "errors": [],
            "reflection": {
                "health_score": 0.0,
                "health_grade": "N/A",
                "alerts_count": 0,
                "cycle_summary": "",
            },
        }

        # 1. 感知 Perceive
        try:
            perception = await self._perceive()
            if not isinstance(perception, dict):
                raise TypeError("perception stage must return a dictionary")
        except Exception as exc:
            logger.exception("[AGI] perception stage failed")
            perception = {}
            report["errors"].append({
                "stage": "perception",
                "type": type(exc).__name__,
                "message": str(exc),
            })
        report["perception"] = perception

        # 2. 诊断 Diagnose
        try:
            alerts = self._diagnose(perception)
        except Exception as exc:
            logger.exception("[AGI] diagnosis stage failed")
            alerts = []
            report["errors"].append({
                "stage": "diagnosis",
                "type": type(exc).__name__,
                "message": str(exc),
            })
        report["diagnosis"]["alerts"] = alerts

        # 2.5 归因+推算（前瞻盈利推算）
        try:
            attribution = self._attribute_pnl(perception)
        except Exception as exc:
            logger.exception("[AGI] PnL attribution stage failed")
            attribution = {}
            report["errors"].append({
                "stage": "attribution",
                "type": type(exc).__name__,
                "message": str(exc),
            })
        try:
            projection = self._project_pnl(perception, attribution)
        except Exception as exc:
            logger.exception("[AGI] PnL projection stage failed")
            projection = {}
            report["errors"].append({
                "stage": "projection",
                "type": type(exc).__name__,
                "message": str(exc),
            })
        report["attribution"] = attribution
        report["projection"] = projection

        # 3. 决策 Decide
        try:
            decision = await self._decide(perception, projection=projection)
            if not isinstance(decision, dict):
                raise TypeError("decision stage must return a dictionary")
        except Exception as exc:
            logger.exception("[AGI-Decide] decision stage failed; failing closed")
            decision = {
                "allocation_plan": None,
                "reallocation_suggestions": [],
                "market_regime": "unknown",
                "strategy_names": [],
                "strategy_metrics": {},
                "rationale": ["决策阶段异常，未生成资金分配计划"],
                "fail_closed": True,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
            report["errors"].append({
                "stage": "decision",
                "type": type(exc).__name__,
                "message": str(exc),
            })
        decision_error = decision.get("error")
        if isinstance(decision_error, dict) and not any(
            error.get("stage") == "decision" for error in report["errors"]
        ):
            report["errors"].append({
                "stage": "decision",
                "type": str(decision_error.get("type", "DecisionError")),
                "message": str(decision_error.get("message", "decision failed")),
            })
        report["decision"] = decision

        if decision.get("fail_closed") is True:
            report["status"] = "fail_closed"
            report["actions"] = []
            report["timescale"] = self._last_timescale
            self._apply_simulation_safety(report)
            # 循环失败熔断：decide 阶段 fail_closed 也算一次失败（仅内存计数，不持久化）
            if report.get("errors"):
                self._cycle_failure_streak += 1
                if self._cycle_failure_streak >= self._cycle_failure_threshold:
                    self._cycle_halted = True
                    self._cycle_halt_reason = (
                        f"consecutive_cycle_failures:{self._cycle_failure_streak}"
                    )
                    logger.error(
                        f"[AGI] cycle circuit breaker tripped (decide fail_closed): "
                        f"{self._cycle_failure_streak} consecutive failures"
                    )
            report = self._sanitize(report)
            return report

        # 4. 执行 Act（只生成动作指令，不直接下单）
        if isinstance(decision_error, dict):
            report["actions"] = []
        else:
            try:
                report["actions"] = self._act(decision, alerts, projection=projection)
            except Exception as exc:
                logger.exception("[AGI] action generation stage failed")
                report["actions"] = []
                report["errors"].append({
                    "stage": "actions",
                    "type": type(exc).__name__,
                    "message": str(exc),
                })

        # 5. 反馈 Reflect
        try:
            self._reflect(report, perception, alerts)
        except Exception as exc:
            logger.exception("[AGI] reflection stage failed")
            report["errors"].append({
                "stage": "reflection",
                "type": type(exc).__name__,
                "message": str(exc),
            })

        # 终态判定
        equity = perception.get("equity")
        if equity is None or safe_float(equity, 0.0) <= 0:
            report["status"] = "fail_closed"
            report["actions"] = []  # 无法确认权益时清空动作指令，绝不放行
        elif isinstance(decision_error, dict) or report["errors"]:
            report["status"] = "degraded"
        elif self._has_partial_dependency():
            report["status"] = "degraded"
        else:
            report["status"] = "ok"

        # 动作冲突消解审计信息（fail_closed 已清空动作，不回填审计）
        if (
            self._reconciliation_enabled
            and self._last_reconciliation is not None
            and report["status"] != "fail_closed"
        ):
            report["reconciliation"] = self._last_reconciliation

        # 决策置信度门控审计信息（fail_closed 已清空动作，不回填审计）
        if (
            self._reconciliation_enabled
            and self._last_confidence_gate is not None
            and report["status"] != "fail_closed"
        ):
            report["confidence_gate"] = self._last_confidence_gate

        # 成本意识调仓门控审计信息（fail_closed 已清空动作，不回填审计）
        if (
            self._cost_guard_enabled
            and self._last_cost_guard is not None
            and report["status"] != "fail_closed"
        ):
            report["cost_guard"] = self._last_cost_guard

        # 本周期时间尺度（slow/strategic 或 fast/tactical），供 Dashboard/溯源审计
        report["timescale"] = self._last_timescale

        self._apply_simulation_safety(report)

        # JSON 安全清洗 + 持久化 + 缓存
        report = self._sanitize(report)
        if self._learning_enabled:
            report["decision_memory"] = [dict(m) for m in self._decision_memory]
        # 已暂停策略集合（sorted list 保证 JSON 可序列化，跨重启续用）
        report["paused_strategies"] = sorted(self._paused_strategies)

        # 循环失败熔断：连续 fail_closed 且有 errors 时累计，超过阈值则自停
        has_errors = bool(report.get("errors"))
        is_fail_closed = report.get("status") == "fail_closed"
        if has_errors and is_fail_closed:
            self._cycle_failure_streak += 1
            if self._cycle_failure_streak >= self._cycle_failure_threshold:
                self._cycle_halted = True
                self._cycle_halt_reason = (
                    f"consecutive_cycle_failures:{self._cycle_failure_streak}"
                )
                logger.error(
                    f"[AGI] cycle circuit breaker tripped: "
                    f"{self._cycle_failure_streak} consecutive failures, "
                    f"threshold={self._cycle_failure_threshold}"
                )
        else:
            self._cycle_failure_streak = 0

        self._persist_state(report)
        self._append_decision_lineage(report)
        self._last_report = copy.deepcopy(report)
        self._last_run_ts = time.monotonic()
        return report

    def _has_partial_dependency(self) -> bool:
        return (
            self.regime_engine is None
            or self.contribution_analyzer is None
            or self.capital_allocator is None
            or self.dynamic_allocator is None
        )

    def get_dashboard_snapshot(self) -> Dict[str, Any]:
        """Return a detached, JSON-safe snapshot for dashboard consumers."""
        report = self._last_report if isinstance(self._last_report, dict) else {}
        decision = _coerce_dict(report.get("decision"))
        projection = report.get("projection")
        allocation_plan = decision.get("allocation_plan")

        return {
            "projection": copy.deepcopy(projection) if isinstance(projection, dict) else {},
            "attribution": copy.deepcopy(report.get("attribution"))
            if isinstance(report.get("attribution"), dict) else {},
            "allocation_plan": (
                copy.deepcopy(allocation_plan) if isinstance(allocation_plan, dict) else {}
            ),
            "projection_accuracy": [
                {
                    "cycle": safe_int(item.get("cycle"), 0),
                    "projected": safe_finite(item.get("projected"), 0.0),
                    "actual": safe_finite(item.get("actual"), 0.0),
                    "bias_ratio": safe_finite(item.get("bias_ratio"), 1.0),
                    "regime": str(item.get("regime", "unknown")),
                }
                for item in self._projection_accuracy
                if isinstance(item, dict)
            ],
            "correction_factor_by_regime": {
                str(key): safe_finite(value, 1.0)
                for key, value in self._correction_factor_by_regime.items()
            },
            "correction_factor_by_regime_updated_cycle": {
                str(key): int(value)
                for key, value in self._correction_factor_by_regime_updated_cycle.items()
            },
        }

    # ─────────────────────────────────────────────────────────────
    # 1. 感知 Perceive
    # ─────────────────────────────────────────────────────────────

    async def _perceive(self) -> Dict[str, Any]:
        perception: Dict[str, Any] = {
            "market_regime": None,
            "contribution": None,
            "equity": None,
            "total_capital": None,
            "used_margin": None,
            "unrealized_pnl": None,
            "freeze_state": {},
            "equity_status": {},
            "correlation": {},
        }

        # 市场状态：优先 RegimeArbiter（融合主引擎+检测器），None 时回退 regime_engine
        source = self.regime_arbiter or self.regime_engine
        if source is not None:
            try:
                # arbiter 有 arbitrate()，主引擎有 get_regime()
                get_regime = getattr(source, "arbitrate", None) or getattr(source, "get_regime", None)
                if callable(get_regime):
                    regime = get_regime()
                    if isinstance(regime, dict):
                        perception["market_regime"] = copy.deepcopy(regime)
                    else:
                        perception["market_regime"] = {"regime": str(regime)}
                    logger.info(
                        f"[AGI-Perceive] regime={perception['market_regime'].get('regime')}"
                    )
                else:
                    logger.debug("[AGI-Perceive] regime source lacks get_regime/arbitrate; skipping")
            except Exception as e:
                logger.warning(f"[AGI-Perceive] regime perception failed (degraded): {e}")
        else:
            logger.debug("[AGI-Perceive] no regime_engine; skipping")

        # 策略贡献
        if self.contribution_analyzer is not None:
            try:
                analyze = getattr(self.contribution_analyzer, "analyze", None)
                if callable(analyze):
                    snapshot = analyze(window="24h")
                    self._last_snapshot = snapshot  # 保留原始快照供决策复用（口径一致）
                    perception["contribution"] = snapshot_to_dict(snapshot)
                    contrib = perception["contribution"]
                    logger.info(
                        f"[AGI-Perceive] contribution total_pnl={contrib.get('total_pnl')}, "
                        f"strategies={len(contrib.get('strategies') or {})}"
                    )
                else:
                    logger.debug("[AGI-Perceive] contribution_analyzer lacks analyze; skipping")
            except Exception as e:
                logger.warning(f"[AGI-Perceive] contribution perception failed (degraded): {e}")
        else:
            logger.debug("[AGI-Perceive] no contribution_analyzer; skipping")

        # 权益 / 总资金
        if self.capital_allocator is not None:
            try:
                get_equity = getattr(self.capital_allocator, "get_equity", None)
                if callable(get_equity):
                    eq = get_equity()
                    perception["equity"] = None if eq is None else safe_finite(eq, 0.0)
                get_capital = getattr(self.capital_allocator, "get_total_capital", None)
                if callable(get_capital):
                    cap = get_capital()
                    perception["total_capital"] = None if cap is None else safe_finite(cap, 0.0)
                logger.info(f"[AGI-Perceive] equity={perception['equity']}")
            except Exception as e:
                logger.warning(f"[AGI-Perceive] capital perception failed (degraded): {e}")
        else:
            logger.debug("[AGI-Perceive] no capital_allocator; equity unknown")

        # 账户级未实现盈亏（供收益检测自动平仓）
        if self.account_manager is not None:
            try:
                get_upl = getattr(self.account_manager, "get_total_unrealized_pnl", None)
                if callable(get_upl):
                    perception["unrealized_pnl"] = safe_finite(get_upl(), 0.0)
                    logger.info(f"[AGI-Perceive] unrealized_pnl={perception['unrealized_pnl']}")
            except Exception as e:
                logger.warning(f"[AGI-Perceive] unrealized_pnl perception failed (degraded): {e}")

        # 已用保证金（供诊断「资金利用率过低」）
        perception["used_margin"] = self._total_used_margin()

        # 冻结策略观察期/缩量试探状态（AGI 能力：感知自动解冻状态机）
        perception["freeze_state"] = self._perceive_freeze_state()

        # 账户状态机（EquityMonitor 的 NORMAL/GROWTH/DECLINE/RECOVERY/EMERGENCY）
        perception["equity_status"] = self._perceive_equity_status()

        # 策略间相关性（StrategyCorrelationAnalyzer 的平均/最大 pair 相关性、有效 N 侵蚀）
        perception["correlation"] = self._perceive_correlation()

        # 逐币种未实现盈亏（供逐币种精细杠杆/间距守卫）
        perception["symbol_pnl"] = self._perceive_symbol_pnl()

        # 逐币种现货持币余额（供现货持有守卫）
        perception["spot_holdings"] = self._perceive_spot_holdings()

        # 账户多空敞口（供组合多空净敞口监控）
        perception["net_exposure"] = self._perceive_net_exposure()

        # 账户级杠杆（供 account_leverage_guard 事前收敛高杠杆）
        perception["account_leverage"] = self._perceive_account_leverage()

        # 挂单保证金（供 pending_margin_guard 事前收敛隐性挂单敞口）
        perception["pending_margin"] = self._perceive_pending_margin()

        return perception

    def _perceive_spot_holdings(self) -> Dict[str, Any]:
        """感知逐币种现货持币余额（供 spot_hold_guard 收敛过度分散现货敞口）。"""
        am = getattr(self, "account_manager", None)
        if am is None:
            return {"currencies": {}, "count": 0}
        try:
            getter = getattr(am, "get_spot_holdings", None)
            if callable(getter):
                data = getter()
                if isinstance(data, dict):
                    return {"currencies": data, "count": len(data)}
        except Exception as e:
            logger.debug(f"[AGI-Perceive] spot_holdings perception failed (degraded): {e}")
        return {"currencies": {}, "count": 0}

    def _perceive_symbol_pnl(self) -> Dict[str, float]:
        """感知逐币种未实现盈亏 {symbol: upl}（供 symbol_param_guard 精细下调杠杆）。"""
        am = getattr(self, "account_manager", None)
        if am is None:
            return {}
        try:
            getter = getattr(am, "get_symbol_unrealized_pnl", None)
            if callable(getter):
                data = getter()
                if isinstance(data, dict):
                    return data
        except Exception as e:
            logger.debug(f"[AGI-Perceive] symbol_pnl perception failed (degraded): {e}")
        return {}

    def _perceive_net_exposure(self) -> Dict[str, float]:
        """感知账户多空敞口 {long, short}（供 net_exposure_guard 收敛方向性失衡）。"""
        am = getattr(self, "account_manager", None)
        if am is None:
            return {}
        try:
            get_long = getattr(am, "get_long_exposure", None)
            get_short = getattr(am, "get_short_exposure", None)
            long_exp = safe_float(get_long(), 0.0) if callable(get_long) else 0.0
            short_exp = safe_float(get_short(), 0.0) if callable(get_short) else 0.0
            return {"long": long_exp, "short": short_exp}
        except Exception as e:
            logger.debug(f"[AGI-Perceive] net_exposure perception failed (degraded): {e}")
            return {}

    def _perceive_account_leverage(self) -> Dict[str, float]:
        """感知账户级总杠杆 {current, max}（供 account_leverage_guard 事前收敛高杠杆）。

        读 account_manager.get_current_leverage() 取当前杠杆，max 从
        get_account_summary()["max_leverage"] 取（与 account_manager 硬减仓阈值同源）。
        依赖缺失或无法获取时返回空 dict（向后兼容，守卫放行）。
        """
        am = getattr(self, "account_manager", None)
        if am is None:
            return {}
        try:
            get_lev = getattr(am, "get_current_leverage", None)
            current = safe_float(get_lev(), 0.0) if callable(get_lev) else 0.0
            max_lev = 0.0
            get_summary = getattr(am, "get_account_summary", None)
            if callable(get_summary):
                summary = get_summary()
                if isinstance(summary, dict):
                    max_lev = safe_float(summary.get("max_leverage"), 0.0)
            return {"current": current, "max": max_lev}
        except Exception as e:
            logger.debug(f"[AGI-Perceive] account_leverage perception failed (degraded): {e}")
            return {}

    def _perceive_pending_margin(self) -> Dict[str, float]:
        """感知账户挂单保证金 {pending}（供 pending_margin_guard 收敛隐性挂单敞口）。

        读 account_manager.get_pending_margin() 取挂单锁定保证金（live/pending 挂单的
        quantity×price/leverage 累加）。依赖缺失或无法获取时返回空 dict（向后兼容，守卫放行）。
        """
        am = getattr(self, "account_manager", None)
        if am is None:
            return {}
        try:
            getter = getattr(am, "get_pending_margin", None)
            pending = safe_float(getter(), 0.0) if callable(getter) else 0.0
            return {"pending": pending}
        except Exception as e:
            logger.debug(f"[AGI-Perceive] pending_margin perception failed (degraded): {e}")
            return {}

    def _perceive_correlation(self) -> Dict[str, Any]:
        """感知 StrategyCorrelationAnalyzer 的策略间相关性快照（同步 get_summary）。

        读取 average_correlation / max_pair_correlation / correlation_regime /
        effective_n / diversification_erosion，供诊断高相关组合风险。
        依赖缺失或尚无数据时返回空 dict（向后兼容，不阻塞闭环）。
        """
        self._last_correlation = {}
        if self.strategy_correlation is None:
            return {}
        try:
            getter = getattr(self.strategy_correlation, "get_summary", None)
            if not callable(getter):
                return {}
            result = getter()
            if not isinstance(result, dict) or result.get("status") != "ready":
                return {}
            summary = {
                "average_correlation": safe_float(result.get("average_correlation"), 0.0),
                "max_pair_correlation": safe_float(result.get("max_pair_correlation"), 0.0),
                "correlation_regime": result.get("correlation_regime"),
                "effective_n": safe_float(result.get("effective_n"), 0.0),
                "diversification_erosion": bool(result.get("diversification_erosion", False)),
            }
            self._last_correlation = summary
            return summary
        except Exception as e:
            logger.debug(f"[AGI-Perceive] correlation query failed: {e}")
            return {}

    def _perceive_equity_status(self) -> Dict[str, Any]:
        """感知 EquityMonitor 的账户状态机（mode/回撤/权益乘数）。"""
        if self.equity_monitor is None:
            return {}
        try:
            getter = getattr(self.equity_monitor, "get_equity_status", None)
            if not callable(getter):
                return {}
            result = getter()
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.debug(f"[AGI-Perceive] equity status query failed: {e}")
            return {}

    def _perceive_freeze_state(self) -> Dict[str, Any]:
        """感知 dynamic_allocator 的冻结状态机（观察期/缩量试探/永久冻结）。"""
        if self.dynamic_allocator is None:
            return {}
        try:
            getter = getattr(self.dynamic_allocator, "get_freeze_state", None)
            if not callable(getter):
                return {}
            result = getter()
            return result if isinstance(result, dict) else {}
        except Exception as e:
            logger.debug(f"[AGI-Perceive] freeze state query failed: {e}")
            return {}

    def _total_used_margin(self) -> Optional[float]:
        """汇总 dynamic_allocator 各策略已用保证金；无法获取时返回 None。"""
        if self.dynamic_allocator is None:
            return None
        try:
            getter = getattr(self.dynamic_allocator, "get_strategy_allocations", None)
            if not callable(getter):
                return None
            allocs = getter()
            if not isinstance(allocs, dict):
                return None
            total = 0.0
            for info in allocs.values():
                if isinstance(info, dict):
                    total += safe_float(info.get("used_margin"), 0.0)
            return total
        except Exception as e:
            logger.debug(f"[AGI-Perceive] used_margin query failed: {e}")
            return None

    def _available_margin_ratio(self) -> Optional[float]:
        """账户可用保证金占总资金比例 = (权益 - 已用保证金) / 总资金。

        供进攻与账户可用保证金联动检查使用。权益/总资金/已用保证金任一无法获取时
        返回 None（守卫放行，向后兼容——无法度量则不阻断进攻）。
        """
        equity = None
        total_capital = None
        if self.capital_allocator is not None:
            get_equity = getattr(self.capital_allocator, "get_equity", None)
            get_cap = getattr(self.capital_allocator, "get_total_capital", None)
            if callable(get_equity):
                try:
                    eq = get_equity()
                    if eq is not None:
                        equity = safe_finite(eq, 0.0)
                except Exception:
                    equity = None
            if callable(get_cap):
                try:
                    cap = get_cap()
                    if cap is not None:
                        total_capital = safe_finite(cap, 0.0)
                except Exception:
                    total_capital = None
        used = self._total_used_margin()
        if equity is None or used is None or not total_capital or total_capital <= 0:
            return None
        return (equity - used) / total_capital

    # ─────────────────────────────────────────────────────────────
    # 2. 诊断 Diagnose
    # ─────────────────────────────────────────────────────────────

    def _diagnose(self, perception: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []

        equity = perception.get("equity")

        # 权益非正 → 最高优先级告警
        if equity is not None and safe_float(equity, 0.0) <= 0:
            alerts.append({
                "level": "critical",
                "type": "non_positive_equity",
                "message": "账户权益 <= 0，自治闭环进入 fail-closed 状态",
            })

        # 账户状态机（EquityMonitor 的 mode）：按状态分层收敛/进攻
        equity_status = perception.get("equity_status") or {}
        mode = equity_status.get("mode")
        if mode == "emergency":
            alerts.append({
                "level": "critical",
                "type": "equity_emergency",
                "mode": mode,
                "message": "账户处于 EMERGENCY 紧急状态，全面收敛（禁开新仓）",
            })
        elif mode == "decline":
            alerts.append({
                "level": "warning",
                "type": "equity_decline",
                "mode": mode,
                "message": "账户处于 DECLINE 衰退状态，谨慎收敛敞口",
            })
        elif mode == "recovery":
            alerts.append({
                "level": "info",
                "type": "equity_recovery",
                "mode": mode,
                "message": "账户处于 RECOVERY 恢复期，渐进恢复（谨慎进攻）",
            })

        # 回撤加速预警：跟踪 max_drawdown_pct 跨周期时序，连续加深 → 先行指标抑制进攻
        if self._drawdown_accel_enabled:
            dd = safe_float(equity_status.get("max_drawdown_pct"), 0.0)
            self._drawdown_history.append(dd)
            if (
                len(self._drawdown_history) == self._drawdown_accel_window
                and all(
                    self._drawdown_history[i] > self._drawdown_history[i - 1]
                    for i in range(1, len(self._drawdown_history))
                )
                and self._drawdown_history[-1] > 0  # 确有回撤（非全 0）
            ):
                alerts.append({
                    "level": "warning",
                    "type": "drawdown_accelerating",
                    "current_drawdown": dd,
                    "window": self._drawdown_accel_window,
                    "message": (
                        f"账户回撤连续 {self._drawdown_accel_window} 周期加深"
                        f"（当前 {dd:.1%}），趋势恶化，暂停进攻防扩散"
                    ),
                })

        # 追涨抑制：连续上涨周期过多 → 市场过热，暂停进攻性加仓（不追高）
        if self._momentum_guard_enabled:
            consecutive_up = safe_int(equity_status.get("consecutive_up"), 0)
            if consecutive_up >= self._momentum_max_consecutive_up:
                alerts.append({
                    "level": "warning",
                    "type": "market_overheated",
                    "consecutive_up": consecutive_up,
                    "message": (
                        f"账户连续上涨 {consecutive_up} 周期（≥ "
                        f"{self._momentum_max_consecutive_up}），市场过热，"
                        f"暂停追高加仓"
                    ),
                })

        # 连续下跌收敛（momentum_guard 的镜像）：连续下跌周期过多 → 权益持续失血，
        # 主动降杠杆收敛 + 暂停进攻（不抄底），对应用户「消除追涨杀跌」
        if self._downside_momentum_enabled:
            consecutive_down = safe_int(equity_status.get("consecutive_down"), 0)
            if consecutive_down >= self._downside_momentum_max_consecutive_down:
                alerts.append({
                    "level": "warning",
                    "type": "market_panicking",
                    "consecutive_down": consecutive_down,
                    "message": (
                        f"账户连续下跌 {consecutive_down} 周期（≥ "
                        f"{self._downside_momentum_max_consecutive_down}），权益持续失血，"
                        f"主动降杠杆收敛"
                    ),
                })

        # 资金利用率过低
        used_margin = perception.get("used_margin")
        if equity is not None and safe_float(equity, 0.0) > 0 and used_margin is not None:
            utilization = safe_div(used_margin, equity, 1.0)
            if utilization < self.utilization_low_threshold:
                alerts.append({
                    "level": "warning",
                    "type": "low_capital_utilization",
                    "utilization": utilization,
                    "message": (
                        f"资金利用率 {utilization:.1%} 低于阈值 "
                        f"{self.utilization_low_threshold:.0%}，存在闲置资金"
                    ),
                })
            # 资金利用率过高：敞口接近满仓/过度杠杆，有强平风险
            if self._utilization_guard_enabled and utilization > self._utilization_high_threshold:
                alerts.append({
                    "level": "warning",
                    "type": "high_capital_utilization",
                    "utilization": utilization,
                    "message": (
                        f"资金利用率 {utilization:.1%} 超过阈值 "
                        f"{self._utilization_high_threshold:.0%}，敞口过高需收敛"
                    ),
                })
            # 资金利用率趋势外推：追踪 utilization 时序，连续上升 → 杠杆/敞口持续放大预警。
            # 与 utilization_guard（阈值型，事后）区分：本守卫看「杠杆持续放大」的轨迹，
            # 即使尚未触及 high_utilization_threshold。与 drawdown_acceleration（回撤深度
            # 加深，亏损侧）区分：本守卫看利用率上升（杠杆侧），二者正交。
            if self._utilization_trend_enabled:
                self._utilization_history.append(utilization)
                if (
                    len(self._utilization_history) == self._utilization_trend_window
                    and all(
                        self._utilization_history[i] > self._utilization_history[i - 1]
                        for i in range(1, len(self._utilization_history))
                    )
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "utilization_rising",
                        "utilization": utilization,
                        "message": (
                            f"资金利用率连续 {self._utilization_trend_window} 周期上升"
                            f"（当前 {utilization:.1%}），杠杆持续放大，收敛敞口防强平"
                        ),
                    })

        # 账户级杠杆守卫：当前杠杆 ≥ soft_threshold_ratio × max_total_leverage → 事前收敛高杠杆。
        # 与 utilization_guard（used_margin/equity，保证金占用率）区分：本守卫看「杠杆倍数」
        # （含未实现浮亏放大），更直接反映强平风险——浮亏抬杠杆时 utilization 未必高，但杠杆已逼近上限。
        if self._account_leverage_enabled:
            al = _coerce_dict(perception.get("account_leverage"))
            cur_lev = safe_float(al.get("current"), 0.0)
            max_lev = safe_float(al.get("max"), 0.0)
            if cur_lev > 0 and max_lev > 0 and cur_lev >= self._account_leverage_soft_ratio * max_lev:
                alerts.append({
                    "level": "warning",
                    "type": "high_account_leverage",
                    "current_leverage": cur_lev,
                    "max_leverage": max_lev,
                    "leverage_ratio": cur_lev / max_lev,
                    "message": (
                        f"账户级杠杆 {cur_lev:.2f}x 达到上限 {max_lev:.2f}x 的 "
                        f"{self._account_leverage_soft_ratio:.0%}（{cur_lev / max_lev:.0%}），"
                        f"事前收敛敞口避免强平"
                    ),
                })

        # 挂单保证金守卫：挂单锁定保证金占权益比例过高 → 事前收敛隐性挂单敞口。
        # 与 utilization_guard（已成交持仓的 used_margin/equity）区分：本守卫看「未成交挂单」锁定的
        # 隐性保证金——大量挂单（如网格密集挂单）会锁定资金，成交后敞口立即兑现、杠杆突增，
        # 但 utilization/leverage 都感知不到（它们只看已成交持仓）。与 account_manager._check_leverage_exposure
        # （把挂单保证金纳入杠杆的硬减仓）区分：本守卫在挂单保证金占比过高时「事前主动」收敛 + 暂停进攻。
        if self._pending_margin_enabled:
            pm = _coerce_dict(perception.get("pending_margin"))
            pending = safe_float(pm.get("pending"), 0.0)
            if equity is not None and safe_float(equity, 0.0) > 0 and pending > 0:
                pending_ratio = pending / safe_float(equity, 0.0)
                if pending_ratio >= self._pending_margin_ratio_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "high_pending_margin",
                        "pending_margin": pending,
                        "pending_margin_ratio": pending_ratio,
                        "message": (
                            f"挂单保证金 {pending:.2f} 占权益 {pending_ratio:.0%} ≥ "
                            f"{self._pending_margin_ratio_threshold:.0%}，隐性挂单敞口过高，"
                            f"事前收敛避免成交后敞口突增"
                        ),
                    })

        # 策略衰退 / 休眠 / 健康度恶化 / 疑似连续亏损
        contrib = _coerce_dict(perception.get("contribution"))
        # 成本预算自适应阈值（权益状态联动：健康放宽、恶化收紧）
        adaptive_funding_threshold = self._adaptive_cost_threshold(self._funding_cost_ratio_threshold)
        adaptive_exec_cost_threshold = self._adaptive_cost_threshold(self._execution_cost_ratio_threshold)
        for name, c in _coerce_dict(contrib.get("strategies")).items():
            lifecycle = c.get("lifecycle")
            trend = c.get("trend")
            grade = c.get("health_grade")
            trades = safe_int(c.get("total_trades"), 0)
            total_pnl = safe_float(c.get("total_pnl"), 0.0)

            # 策略级浮亏止损：单策略浮亏（unrealized_pnl<0）占账户权益比例超阈值 → 止损
            if self._unrealized_loss_enabled:
                upl = safe_float(c.get("unrealized_pnl"), 0.0)
                eq = safe_float(equity, 0.0)
                if upl < 0 and eq > 0:
                    loss_pct = -upl / eq
                    if loss_pct >= self._unrealized_loss_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "strategy_floating_loss",
                            "strategy": name,
                            "unrealized_pnl": upl,
                            "loss_pct": loss_pct,
                            "message": (
                                f"策略 {name} 当前浮亏 {upl:.2f}（占权益 {loss_pct:.1%} ≥ "
                                f"{self._unrealized_loss_threshold:.1%}），建议止损减仓"
                            ),
                        })

            # 策略级已实现亏损守卫：已平仓交易累计为负（永久损失）占权益比例超阈值 → 收敛资金权重。
            # 与 unrealized_loss_guard（浮亏，可能回本）区分：本守卫看「已实现亏损」（不可逆、永久）。
            if self._realized_loss_enabled:
                rl = safe_float(c.get("realized_pnl"), 0.0)
                eq = safe_float(equity, 0.0)
                if rl < 0 and eq > 0:
                    rloss_pct = -rl / eq
                    if rloss_pct >= self._realized_loss_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "realized_loss",
                            "strategy": name,
                            "realized_pnl": rl,
                            "loss_pct": rloss_pct,
                            "message": (
                                f"策略 {name} 已实现亏损 {rl:.2f}（占权益 {rloss_pct:.1%} ≥ "
                                f"{self._realized_loss_threshold:.1%}），频繁交易累积永久损失，收敛资金权重"
                            ),
                        })

            # 策略级浮亏加深趋势外推：追踪 unrealized_pnl 时序，连续下降（浮亏加深）→ 事前预警。
            # 与 unrealized_loss_guard（阈值型：浮亏占权益超 loss_threshold 才清仓）区分：
            # 本守卫在浮亏触及阈值之前收敛敞口（事前），仅当当前处于浮亏（unrealized_pnl<0）
            # 且连续 window 周期下降时触发。与 give_back（浮盈回吐，收益侧）形成对称。
            if self._unrealized_loss_trend_enabled:
                upl = safe_float(c.get("unrealized_pnl"), 0.0)
                upl_hist = self._strategy_unrealized_loss_history.setdefault(
                    str(name), deque(maxlen=self._unrealized_loss_trend_window)
                )
                upl_hist.append(upl)
                if (
                    len(upl_hist) == self._unrealized_loss_trend_window
                    and upl < 0  # 当前处于浮亏
                    and all(upl_hist[i] < upl_hist[i - 1] for i in range(1, len(upl_hist)))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "unrealized_loss_deteriorating",
                        "strategy": name,
                        "unrealized_pnl": upl,
                        "message": (
                            f"策略 {name} 浮亏连续 {self._unrealized_loss_trend_window} 周期加深"
                            f"（当前 {upl:.2f}），提前收敛敞口止损"
                        ),
                    })

            # 策略级手续费率守卫：单策略手续费占毛利比例过高 → 过度交易侵蚀利润。
            # 与 cost_awareness（账户级：账户整体手续费侵蚀）区分：本守卫看单策略粒度，
            # 识别「哪个策略在过度交易」。阈值型守卫，无跨周期状态。仅在总盈亏为正时评估。
            if self._strategy_fee_ratio_enabled and total_pnl > 0:
                s_total_fees = safe_float(c.get("total_fees"), 0.0)
                gross = total_pnl + s_total_fees
                if s_total_fees > 0 and gross > 0:
                    fee_ratio = s_total_fees / gross
                    if fee_ratio >= self._strategy_fee_ratio_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "high_strategy_fee",
                            "strategy": name,
                            "fee_ratio": fee_ratio,
                            "total_fees": s_total_fees,
                            "message": (
                                f"策略 {name} 手续费侵蚀过高（{fee_ratio:.0%} ≥ "
                                f"{self._strategy_fee_ratio_threshold:.0%}），交易过频，建议降频"
                            ),
                        })

            # 策略级资金费率守卫：单策略资金费占毛利比例过高 → 持仓时间成本侵蚀。
            # 与 strategy_fee_ratio_guard（手续费 = 交易频率成本）区分：本守卫看「资金费率」——
            # 持仓过久（尤其隔夜单边持仓）被 funding 持续侵蚀，即使不交易也亏钱。阈值型守卫，
            # 无跨周期状态。仅在总盈亏为正且资金费为正（支付资金费）时评估。
            if self._funding_cost_enabled and total_pnl > 0:
                s_funding = safe_float(c.get("total_funding_cost"), 0.0)
                s_total_fees = safe_float(c.get("total_fees"), 0.0)
                gross = total_pnl + s_total_fees
                if s_funding > 0 and gross > 0:
                    funding_ratio = s_funding / gross
                    if funding_ratio >= adaptive_funding_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "high_funding_cost",
                            "strategy": name,
                            "funding_ratio": funding_ratio,
                            "total_funding_cost": s_funding,
                            "message": (
                                f"策略 {name} 资金费侵蚀过高（{funding_ratio:.0%} ≥ "
                                f"{adaptive_funding_threshold:.0%}），持仓过久，建议收敛敞口"
                            ),
                        })

            # 策略级资金费率趋势外推：追踪「资金费率」跨周期时序，连续 window 周期严格上升
            # （持仓时间成本相对盈利持续恶化）→ 事前收敛持仓。与 funding_cost_guard（阈值型：
            # funding_ratio 超 threshold 才收敛）区分：本守卫看「资金费率持续上升」轨迹（事前）。
            # 与 strategy_fee_ratio_trend_guard（手续费率趋势，看换手频率）区分：本守卫看持仓
            # 时间成本趋势。total_pnl<=0 或 funding<=0 时 funding_ratio 记为 0（重置趋势）。
            if self._funding_cost_trend_enabled and total_pnl > 0:
                s_funding = safe_float(c.get("total_funding_cost"), 0.0)
                s_total_fees = safe_float(c.get("total_fees"), 0.0)
                gross = total_pnl + s_total_fees
                funding_ratio = (s_funding / gross) if s_funding > 0 and gross > 0 else 0.0
                funding_hist = self._strategy_funding_cost_history.setdefault(
                    str(name), deque(maxlen=self._funding_cost_trend_window)
                )
                funding_hist.append(funding_ratio)
                if (
                    len(funding_hist) == self._funding_cost_trend_window
                    and funding_ratio > 0
                    and all(funding_hist[i] > funding_hist[i - 1] for i in range(1, len(funding_hist)))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "funding_cost_rising",
                        "strategy": name,
                        "funding_ratio": funding_ratio,
                        "message": (
                            f"策略 {name} 资金费率连续 {self._funding_cost_trend_window} 周期上升"
                            f"（当前 {funding_ratio:.0%}），持仓时间成本持续恶化，提前收敛持仓"
                        ),
                    })

            # 策略级执行质量成本守卫：滑点 + 点差成本占毛利比例过高 → 执行质量差。
            # 与 strategy_fee_ratio_guard（手续费 = 交易频率成本）、funding_cost_guard（资金费 =
            # 持仓时间成本）区分：本守卫看「执行质量成本」（下单执行时点差/滑点损失），反映流动性、
            # 下单时机、市价 vs 限价的执行质量。阈值型守卫，无跨周期状态。仅在总盈亏为正时评估。
            if self._execution_cost_enabled and total_pnl > 0:
                s_slippage = safe_float(c.get("total_slippage_cost"), 0.0)
                s_spread = safe_float(c.get("total_spread_cost"), 0.0)
                s_total_fees = safe_float(c.get("total_fees"), 0.0)
                gross = total_pnl + s_total_fees
                exec_cost = s_slippage + s_spread
                if exec_cost > 0 and gross > 0:
                    exec_ratio = exec_cost / gross
                    if exec_ratio >= adaptive_exec_cost_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "high_execution_cost",
                            "strategy": name,
                            "execution_cost_ratio": exec_ratio,
                            "slippage_cost": s_slippage,
                            "spread_cost": s_spread,
                            "message": (
                                f"策略 {name} 执行成本侵蚀过高（{exec_ratio:.0%} ≥ "
                                f"{adaptive_exec_cost_threshold:.0%}），滑点/点差过大，"
                                f"建议改善下单执行质量"
                            ),
                        })

            # 策略级执行成本趋势外推：追踪「执行成本率」跨周期时序，连续 window 周期严格上升
            # （滑点/点差相对盈利持续加剧）→ 事前收敛。与 execution_cost_guard（阈值型：
            # exec_ratio 超 threshold 才收敛）区分：本守卫看「执行成本率持续上升」轨迹（事前）。
            # total_pnl<=0 或 exec_cost<=0 时 exec_ratio 记为 0（重置趋势）。
            if self._execution_cost_trend_enabled and total_pnl > 0:
                s_slippage = safe_float(c.get("total_slippage_cost"), 0.0)
                s_spread = safe_float(c.get("total_spread_cost"), 0.0)
                s_total_fees = safe_float(c.get("total_fees"), 0.0)
                gross = total_pnl + s_total_fees
                exec_cost = s_slippage + s_spread
                exec_ratio = (exec_cost / gross) if exec_cost > 0 and gross > 0 else 0.0
                exec_hist = self._strategy_execution_cost_history.setdefault(
                    str(name), deque(maxlen=self._execution_cost_trend_window)
                )
                exec_hist.append(exec_ratio)
                if (
                    len(exec_hist) == self._execution_cost_trend_window
                    and exec_ratio > 0
                    and all(exec_hist[i] > exec_hist[i - 1] for i in range(1, len(exec_hist)))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "execution_cost_rising",
                        "strategy": name,
                        "execution_cost_ratio": exec_ratio,
                        "message": (
                            f"策略 {name} 执行成本率连续 {self._execution_cost_trend_window} 周期上升"
                            f"（当前 {exec_ratio:.0%}），滑点/点差持续加剧，提前收敛敞口"
                        ),
                    })

            # 策略级多空方向失衡守卫：某一方向累计亏损且绝对亏损占比过高 → 方向判断错误。
            # 与连续亏损（consecutive_losses，看连败时序）区分：本守卫看「方向」——策略多空
            # 两个方向都有交易，但一个方向持续逆势开仓（该方向累计盈亏为负）。仅当两个方向
            # 交易笔数均达 min_trades 时评估。阈值型守卫，无跨周期状态。
            if self._long_short_imbalance_enabled:
                long_trades = safe_int(c.get("long_trades"), 0)
                short_trades = safe_int(c.get("short_trades"), 0)
                if (
                    long_trades >= self._long_short_imbalance_min_trades
                    and short_trades >= self._long_short_imbalance_min_trades
                ):
                    long_pnl = safe_float(c.get("long_pnl"), 0.0)
                    short_pnl = safe_float(c.get("short_pnl"), 0.0)
                    gross_abs = abs(long_pnl) + abs(short_pnl)
                    if gross_abs > 0:
                        # 多头方向亏损（空头盈利或至少多头亏损显著）
                        if long_pnl < 0:
                            loss_ratio = abs(long_pnl) / gross_abs
                            if loss_ratio >= self._long_short_imbalance_loss_ratio:
                                alerts.append({
                                    "level": "warning",
                                    "type": "long_side_losing",
                                    "strategy": name,
                                    "long_pnl": long_pnl,
                                    "short_pnl": short_pnl,
                                    "loss_ratio": loss_ratio,
                                    "message": (
                                        f"策略 {name} 多头方向持续亏损（多头 {long_pnl:.2f}，"
                                        f"空头 {short_pnl:.2f}，多头亏损占比 {loss_ratio:.0%}），"
                                        f"逆势开多，提前降杠杆收敛"
                                    ),
                                })
                        # 空头方向亏损（多头盈利或至少空头亏损显著）
                        if short_pnl < 0:
                            loss_ratio = abs(short_pnl) / gross_abs
                            if loss_ratio >= self._long_short_imbalance_loss_ratio:
                                alerts.append({
                                    "level": "warning",
                                    "type": "short_side_losing",
                                    "strategy": name,
                                    "long_pnl": long_pnl,
                                    "short_pnl": short_pnl,
                                    "loss_ratio": loss_ratio,
                                    "message": (
                                        f"策略 {name} 空头方向持续亏损（多头 {long_pnl:.2f}，"
                                        f"空头 {short_pnl:.2f}，空头亏损占比 {loss_ratio:.0%}），"
                                        f"逆势开空，提前降杠杆收敛"
                                    ),
                                })

            # 策略方向偏好失衡：多空总笔数达 min_trades 但单边笔数占比过高（几乎只做多/只做空）
            # → 方向过于单一、缺乏双向灵活性（追涨杀跌的结构性风险），提前降杠杆收敛。
            # 与 long_short_imbalance_guard（看方向盈亏，事后）区分：本守卫看方向笔数失衡（事前）。
            if self._direction_bias_enabled:
                long_trades = safe_int(c.get("long_trades"), 0)
                short_trades = safe_int(c.get("short_trades"), 0)
                total_dir_trades = long_trades + short_trades
                if total_dir_trades >= self._direction_bias_min_trades:
                    bias_ratio = max(long_trades, short_trades) / total_dir_trades
                    if bias_ratio >= self._direction_bias_ratio_threshold:
                        bias_side = "多" if long_trades >= short_trades else "空"
                        alerts.append({
                            "level": "warning",
                            "type": "direction_bias",
                            "strategy": name,
                            "long_trades": long_trades,
                            "short_trades": short_trades,
                            "bias_ratio": bias_ratio,
                            "message": (
                                f"策略 {name} 方向偏好失衡（{bias_side}头 {max(long_trades, short_trades)} 笔"
                                f" / 总 {total_dir_trades} 笔，单边占比 {bias_ratio:.0%} ≥ "
                                f"{self._direction_bias_ratio_threshold:.0%}），方向单一缺乏对冲，"
                                f"提前降杠杆收敛"
                            ),
                        })

            # 策略级手续费率趋势外推：追踪 fee_ratio 时序，连续上升 → 交易成本侵蚀加剧预警。
            # 与 strategy_fee_ratio_guard（阈值型：超 threshold 才收敛）区分：本守卫看「手续费率
            # 持续上升」轨迹（事前），尚未触及阈值即降频收敛。total_pnl<=0 时 fee_ratio 记为 0（重置趋势）。
            if self._strategy_fee_ratio_trend_enabled and total_pnl > 0:
                s_total_fees = safe_float(c.get("total_fees"), 0.0)
                gross = total_pnl + s_total_fees
                fee_ratio = (s_total_fees / gross) if s_total_fees > 0 and gross > 0 else 0.0
                fee_hist = self._strategy_fee_ratio_history.setdefault(
                    str(name), deque(maxlen=self._strategy_fee_ratio_trend_window)
                )
                fee_hist.append(fee_ratio)
                if (
                    len(fee_hist) == self._strategy_fee_ratio_trend_window
                    and fee_ratio > 0
                    and all(fee_hist[i] > fee_hist[i - 1] for i in range(1, len(fee_hist)))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "strategy_fee_ratio_rising",
                        "strategy": name,
                        "fee_ratio": fee_ratio,
                        "message": (
                            f"策略 {name} 手续费率连续 {self._strategy_fee_ratio_trend_window} 周期上升"
                            f"（当前 {fee_ratio:.0%}），交易成本侵蚀加剧，提前降频收敛"
                        ),
                    })

            # 健康度趋势外推：追踪 health_score 时序，连续下降 → 事前预警（降杠杆）
            if self._health_trend_enabled:
                hs = safe_float(c.get("health_score"), 0.0)
                hist = self._strategy_health_history.setdefault(
                    str(name), deque(maxlen=self._health_trend_window)
                )
                hist.append(hs)
                if len(hist) == self._health_trend_window and all(
                    hist[i] < hist[i - 1] for i in range(1, len(hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "health_deteriorating",
                        "strategy": name,
                        "health_score": hs,
                        "message": (
                            f"策略 {name} 健康度连续 {self._health_trend_window} 周期下降"
                            f"（当前 {hs:.0f}），提前降杠杆防恶化"
                        ),
                    })

            # 胜率趋势外推：追踪 win_rate 时序，连续下降 → 事前预警（降杠杆）
            # 仅在交易笔数达 min_trades 时追踪，避免样本不足的噪声误触发
            if self._win_rate_trend_enabled and trades >= self._win_rate_trend_min_trades:
                wr = safe_float(c.get("win_rate"), 0.0)
                wr_hist = self._strategy_win_rate_history.setdefault(
                    str(name), deque(maxlen=self._win_rate_trend_window)
                )
                wr_hist.append(wr)
                if len(wr_hist) == self._win_rate_trend_window and all(
                    wr_hist[i] < wr_hist[i - 1] for i in range(1, len(wr_hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "win_rate_deteriorating",
                        "strategy": name,
                        "win_rate": wr,
                        "message": (
                            f"策略 {name} 胜率连续 {self._win_rate_trend_window} 周期下降"
                            f"（当前 {wr:.1%}），提前降杠杆防恶化"
                        ),
                    })

            # 胜率绝对值守卫：胜率已低于 min_win_rate（绝对阈值）但尚未连续下降时趋势守卫不触发，
            # 但策略实际在「长期低胜率」状态（赢少输多，策略有效性存疑）。仅在交易笔数达 min_trades
            # 时评估（样本不足的胜率噪声大）。阈值型守卫（非趋势型），无跨周期状态。
            if (self._win_rate_guard_enabled
                    and trades >= self._win_rate_guard_min_trades):
                wr = safe_float(c.get("win_rate"), 0.0)
                if wr < self._win_rate_guard_min_win_rate:
                    alerts.append({
                        "level": "warning",
                        "type": "low_win_rate",
                        "strategy": name,
                        "win_rate": wr,
                        "message": (
                            f"策略 {name} 胜率 {wr:.1%} 低于阈值 "
                            f"{self._win_rate_guard_min_win_rate:.1%}，长期低胜率，提前降杠杆收敛"
                        ),
                    })

            # 夏普趋势外推：追踪 sharpe_ratio 时序，连续下降 → 事前预警（降杠杆）
            # 夏普下降意味着策略在承担更多风险获取同等收益，是 PnL 转负前的先行指标
            # 仅在交易笔数达 min_trades 时追踪，避免样本不足的噪声误触发
            if self._sharpe_trend_enabled and trades >= self._sharpe_trend_min_trades:
                sr = safe_float(c.get("sharpe_ratio"), 0.0)
                sr_hist = self._strategy_sharpe_history.setdefault(
                    str(name), deque(maxlen=self._sharpe_trend_window)
                )
                sr_hist.append(sr)
                if len(sr_hist) == self._sharpe_trend_window and all(
                    sr_hist[i] < sr_hist[i - 1] for i in range(1, len(sr_hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "sharpe_deteriorating",
                        "strategy": name,
                        "sharpe_ratio": sr,
                        "message": (
                            f"策略 {name} 夏普比率连续 {self._sharpe_trend_window} 周期下降"
                            f"（当前 {sr:.2f}），风险调整后收益恶化，提前降杠杆"
                        ),
                    })

            # 连续亏损收敛：consecutive_losses ≥ loss_threshold → 提前降杠杆（事前风控）
            # 阈值型守卫（非趋势型），直接读 contribution 实时数据，无需 AGI 追踪时序。
            # 填补「策略健康」与「硬冻结（consecutive_losses≥5）」之间的响应空白：
            # 在策略管理器级 FROZEN 之前，AGI 先降杠杆收敛敞口，避免亏损扩散。
            if self._consecutive_losses_enabled:
                cl = safe_int(c.get("consecutive_losses"), 0)
                if cl >= self._consecutive_losses_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "strategy_losing_streak",
                        "strategy": name,
                        "consecutive_losses": cl,
                        "message": (
                            f"策略 {name} 连续亏损 {cl} 笔（≥ "
                            f"{self._consecutive_losses_threshold}），提前降杠杆收敛"
                        ),
                    })

            # 止损频率收敛：止损率（stop_loss_count/total_trades）过高 → 入场时机差预警。
            # 与 consecutive_losses（连续亏损笔数，看连败）区分：本守卫看「止损占比」——
            # 即使亏损不连续，止损交易占比过高也说明开仓位置不佳（追高开多/追低开空）。
            # 仅在交易笔数达 min_trades 时评估，避免样本不足噪声误触发。
            if self._stop_loss_frequency_enabled and trades >= self._stop_loss_frequency_min_trades:
                slc = safe_int(c.get("stop_loss_count"), 0)
                if slc > 0:
                    slr = slc / trades
                    if slr >= self._stop_loss_frequency_ratio_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "high_stop_loss_rate",
                            "strategy": name,
                            "stop_loss_count": slc,
                            "stop_loss_ratio": slr,
                            "message": (
                                f"策略 {name} 止损率过高（{slr:.0%} ≥ "
                                f"{self._stop_loss_frequency_ratio_threshold:.0%}），"
                                f"入场时机差，提前降杠杆收敛"
                            ),
                        })

            # 策略级止盈止损比守卫：止盈笔数相对止损笔数过少 → 离场质量差（善止损不善止盈）。
            # 与 stop_loss_frequency_guard（止损率过高，看入场时机差）区分：本守卫看「止盈/止损比」，
            # 赚的时候不落袋、亏的时候才离场，是「稳定资金增长」的另一隐患。仅在交易笔数达
            # min_trades 且有止损单（stop_loss_count>0）时评估。阈值型守卫，无跨周期状态。
            if self._take_profit_ratio_enabled and trades >= self._take_profit_ratio_min_trades:
                slc = safe_int(c.get("stop_loss_count"), 0)
                tpc = safe_int(c.get("take_profit_count"), 0)
                if slc > 0:
                    tp_ratio = tpc / slc
                    if tp_ratio < self._take_profit_ratio_min_ratio:
                        alerts.append({
                            "level": "warning",
                            "type": "low_take_profit_ratio",
                            "strategy": name,
                            "take_profit_count": tpc,
                            "stop_loss_count": slc,
                            "take_profit_ratio": tp_ratio,
                            "message": (
                                f"策略 {name} 止盈止损比过低（止盈 {tpc} / 止损 {slc} = "
                                f"{tp_ratio:.2f} < {self._take_profit_ratio_min_ratio:.2f}），"
                                f"善止损不善止盈，提前降杠杆收敛"
                            ),
                        })

            # 策略级盈亏比守卫：平均盈利/平均亏损过低 → 单笔盈亏结构差（小赚大亏）。
            # 与 take_profit_ratio_guard（止盈笔数/止损笔数，看「离场笔数」质量）区分：本守卫看
            # 「金额」——avg_win/avg_loss 低于阈值，赚的时候赚得少、亏的时候亏得多（追涨杀跌的
            # 典型盈亏结构）。与 profit_factor_trend_guard（累计盈亏比，被胜率稀释，看趋势）区分：
            # 本守卫看纯单笔金额结构。仅在交易笔数达 min_trades 且有盈亏单时评估。阈值型守卫。
            if self._win_loss_ratio_enabled and trades >= self._win_loss_ratio_min_trades:
                avg_win = safe_float(c.get("avg_win"), 0.0)
                avg_loss = safe_float(c.get("avg_loss"), 0.0)
                if avg_win > 0 and avg_loss > 0:
                    wl_ratio = avg_win / avg_loss
                    if wl_ratio < self._win_loss_ratio_min_ratio:
                        alerts.append({
                            "level": "warning",
                            "type": "low_win_loss_ratio",
                            "strategy": name,
                            "avg_win": avg_win,
                            "avg_loss": avg_loss,
                            "win_loss_ratio": wl_ratio,
                            "message": (
                                f"策略 {name} 盈亏比过低（平均盈利 {avg_win:.2f} / 平均亏损 "
                                f"{avg_loss:.2f} = {wl_ratio:.2f} < "
                                f"{self._win_loss_ratio_min_ratio:.2f}），小赚大亏，提前降杠杆收敛"
                            ),
                        })

            # 策略级止盈止损盈亏金额比守卫：止盈累计盈亏相对止损累计亏损过少 → 仓位管理失衡
            # （赚的时候轻仓、亏的时候重仓）。与 take_profit_ratio_guard（止盈/止损「笔数」比）
            # 和 win_loss_ratio_guard（平均「单笔」盈利/亏损比）区分：本守卫看「累计金额」——
            # 止盈单总赚的 vs 止损单总亏的，即使笔数比、单笔金额比都健康，若盈利单轻仓、亏损单
            # 重仓，止盈累计金额仍会低于止损累计亏损，是追涨杀跌的仓位管理特征。阈值型守卫。
            if self._take_profit_pnl_ratio_enabled and trades >= self._take_profit_pnl_ratio_min_trades:
                tp_pnl = safe_float(c.get("take_profit_pnl"), 0.0)
                sl_pnl = safe_float(c.get("stop_loss_pnl"), 0.0)
                if tp_pnl > 0 and sl_pnl < 0:
                    pnl_ratio = tp_pnl / abs(sl_pnl)
                    if pnl_ratio < self._take_profit_pnl_ratio_min_ratio:
                        alerts.append({
                            "level": "warning",
                            "type": "low_take_profit_pnl_ratio",
                            "strategy": name,
                            "take_profit_pnl": tp_pnl,
                            "stop_loss_pnl": sl_pnl,
                            "take_profit_pnl_ratio": pnl_ratio,
                            "message": (
                                f"策略 {name} 止盈止损盈亏金额比过低（止盈累计 {tp_pnl:.2f} / "
                                f"止损累计 |{sl_pnl:.2f}| = {pnl_ratio:.2f} < "
                                f"{self._take_profit_pnl_ratio_min_ratio:.2f}），"
                                f"赚的时候轻仓、亏的时候重仓，提前降杠杆收敛"
                            ),
                        })

            # 风险调整后贡献守卫：盈利但风险调整后贡献过低 → 小赚大扛（盈利/回撤比失衡）。
            # 与 max_drawdown_trend_guard（回撤深度趋势，事前）、tail_risk_guard（回撤绝对水平，
            # 阈值）区分：本守卫看「风险调整后贡献」（PnL / MaxDD）——一个策略 total_pnl>0
            # （仍盈利）但盈利相对所承受的回撤过少（赚的没扛的多），是「追涨杀跌、频繁交易」
            # 的风险收益比失衡特征。仅在交易笔数达 min_trades 且 total_pnl>0 时评估。阈值型守卫。
            if (self._risk_adjusted_contribution_enabled
                    and trades >= self._risk_adjusted_contribution_min_trades
                    and total_pnl > 0):
                rac = safe_float(c.get("risk_adjusted_contribution"), 0.0)
                if 0 < rac < self._risk_adjusted_contribution_min_ratio:
                    alerts.append({
                        "level": "warning",
                        "type": "low_risk_adjusted_contribution",
                        "strategy": name,
                        "risk_adjusted_contribution": rac,
                        "max_drawdown": safe_float(c.get("max_drawdown"), 0.0),
                        "message": (
                            f"策略 {name} 风险调整后贡献过低（{rac:.2f} < "
                            f"{self._risk_adjusted_contribution_min_ratio:.2f}），"
                            f"盈利不足以覆盖所承受的回撤风险，提前降杠杆收敛"
                        ),
                    })

            # 夏普比率绝对阈值守卫：负夏普（sharpe < min_sharpe_ratio）→ 提前降杠杆（事前风控）。
            # 与 sharpe_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」——策略夏普长期
            # 为负（风险调整后负收益）但未连续下降（平坦负值）时趋势守卫不触发，但策略实际在
            # 「承担风险却负收益」（单笔盈亏均值为负 = 平均每笔亏损），是纯亏损硬信号，应提前收敛。
            # 仅在交易笔数达 min_trades 时评估（样本不足时夏普噪声大）。阈值型守卫，无跨周期状态。
            if (self._sharpe_ratio_enabled
                    and trades >= self._sharpe_ratio_min_trades):
                sr = safe_float(c.get("sharpe_ratio"), 0.0)
                if sr < self._sharpe_ratio_min_ratio:
                    alerts.append({
                        "level": "warning",
                        "type": "negative_sharpe",
                        "strategy": name,
                        "sharpe_ratio": sr,
                        "message": (
                            f"策略 {name} 夏普比率为负（{sr:.2f} < "
                            f"{self._sharpe_ratio_min_ratio:.2f}），风险调整后负收益，"
                            f"提前降杠杆收敛"
                        ),
                    })

            # 策略休眠资金回收守卫：曾活跃但长时间未交易 → 回收闲置资金（激活 lifecycle_last_trade_age_hours）
            # 与 strategy_dormant（lifecycle=="dormant"，idle>72h → strategy_pause 停开新仓）区分：
            # 本守卫在更早的 reclaim_idle_hours（默认48h）回收资金（reallocate decrease），事前释放
            # 闲置资金到活跃策略；strategy_dormant 是事后停开新仓。二者互补：先回收资金、再停开新仓。
            # 仅在曾活跃（total_trades≥min_trades）时评估，避免误回收无数据策略。阈值型守卫。
            if (self._strategy_staleness_enabled
                    and trades >= self._strategy_staleness_min_trades):
                idle_hours = safe_float(c.get("lifecycle_last_trade_age_hours"), 0.0)
                if idle_hours >= self._strategy_staleness_reclaim_idle_hours:
                    alerts.append({
                        "level": "warning",
                        "type": "strategy_stale",
                        "strategy": name,
                        "idle_hours": idle_hours,
                        "message": (
                            f"策略 {name} 已闲置 {idle_hours:.1f}h（≥ "
                            f"{self._strategy_staleness_reclaim_idle_hours:.0f}h），"
                            f"回收闲置资金到活跃策略"
                        ),
                    })

            # 健康度骤降守卫：单周期健康度骤降（突然恶化）→ 提前降杠杆（激活 delta_health）
            # 与 health_trend_guard（连续 window 周期渐降，事前慢信号）区分：本守卫看「单周期
            # 骤降」（急信号）——健康度在一个周期内暴跌 ≥ crash_threshold（默认20分），即使
            # 尚未连续下降也应收敛，是比渐降更即时的健康恶化信号。仅在交易笔数达 min_trades
            # 时评估（样本不足的 delta_health 噪声大）。阈值型守卫。
            if (self._health_crash_enabled
                    and trades >= self._health_crash_min_trades):
                dh = safe_float(c.get("delta_health"), 0.0)
                if dh <= -self._health_crash_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "health_crash",
                        "strategy": name,
                        "delta_health": dh,
                        "message": (
                            f"策略 {name} 健康度单周期骤降 {dh:.1f} 分（≤ "
                            f"-{self._health_crash_threshold:.0f}），健康状态突然恶化，提前降杠杆收敛"
                        ),
                    })

            # 组合波动率预算守卫：单策略单笔盈亏标准差超预算（绝对阈值）→ 提前降杠杆。
            # 与 volatility_trend_guard（波动率连续上升，趋势型）区分：本守卫看「绝对阈值」——
            # 单笔盈亏标准差 volatility > volatility_budget 即收敛，即使尚未连续上升，构成
            # 波动率维度的「阈值 + 趋势」双层。仅在交易笔数达 min_trades 时评估。阈值型守卫。
            if (self._volatility_budget_enabled
                    and trades >= self._volatility_budget_min_trades):
                vol = safe_float(c.get("volatility"), 0.0)
                if vol > self._volatility_budget:
                    alerts.append({
                        "level": "warning",
                        "type": "volatility_over_budget",
                        "strategy": name,
                        "volatility": vol,
                        "message": (
                            f"策略 {name} 单笔盈亏标准差 {vol:.2f} 超波动率预算 "
                            f"{self._volatility_budget:.2f}，波动过大，提前降杠杆收敛"
                        ),
                    })

            # 盈亏比趋势外推：追踪 profit_factor 时序，连续下降 → 事前预警（降杠杆）
            # profit_factor 度量「赢的幅度 vs 输的幅度」，独立于胜率频率与夏普——
            # 一个策略可能胜率高但每笔赢小输大导致 profit_factor 持续下滑（盈亏比恶化）。
            # 仅在交易笔数达 min_trades 时追踪，避免样本不足的噪声误触发。
            if self._profit_factor_trend_enabled and trades >= self._profit_factor_trend_min_trades:
                pf = safe_float(c.get("profit_factor"), 1.0)
                pf_hist = self._strategy_profit_factor_history.setdefault(
                    str(name), deque(maxlen=self._profit_factor_trend_window)
                )
                pf_hist.append(pf)
                if len(pf_hist) == self._profit_factor_trend_window and all(
                    pf_hist[i] < pf_hist[i - 1] for i in range(1, len(pf_hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "profit_factor_deteriorating",
                        "strategy": name,
                        "profit_factor": pf,
                        "message": (
                            f"策略 {name} 盈亏比连续 {self._profit_factor_trend_window} 周期下降"
                            f"（当前 {pf:.2f}），赢亏幅度比恶化，提前降杠杆"
                        ),
                    })

            # 盈亏比绝对阈值守卫：盈亏比 < min_profit_factor（总亏损>总盈利，已实现净亏损）→ 提前降杠杆。
            # 与 profit_factor_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」——策略盈亏比
            # 长期 <1（已实现净亏损）但未连续下降（平坦低位）时趋势守卫不触发，但策略实际在
            # 「总亏损超过总盈利」，是频繁交易消耗资金的直接体现（已实现口径），应提前收敛。
            # 仅在交易笔数达 min_trades 时评估（样本不足时盈亏比噪声大）。阈值型守卫，无跨周期状态。
            if (self._profit_factor_guard_enabled
                    and trades >= self._profit_factor_guard_min_trades):
                pf = safe_float(c.get("profit_factor"), 0.0)
                if 0 < pf < self._profit_factor_guard_min_ratio:
                    alerts.append({
                        "level": "warning",
                        "type": "low_profit_factor",
                        "strategy": name,
                        "profit_factor": pf,
                        "message": (
                            f"策略 {name} 盈亏比过低（{pf:.2f} < "
                            f"{self._profit_factor_guard_min_ratio:.2f}），总亏损超过总盈利"
                            f"（已实现净亏损），提前降杠杆收敛"
                        ),
                    })

            # 最大回撤趋势外推：追踪 max_drawdown 时序，连续上升（加深）→ 事前预警（降杠杆）
            # max_drawdown 是纯风险维度的独立信号——策略可能收益好但回撤持续加深（承担更大风险）。
            # 与前四子守卫（收益质量维度）区分：本守卫看回撤深度轨迹（风险深度维度）。
            # 仅在交易笔数达 min_trades 时追踪，避免样本不足的噪声误触发。
            if self._max_drawdown_trend_enabled and trades >= self._max_drawdown_trend_min_trades:
                dd = safe_float(c.get("max_drawdown"), 0.0)
                dd_hist = self._strategy_max_drawdown_history.setdefault(
                    str(name), deque(maxlen=self._max_drawdown_trend_window)
                )
                dd_hist.append(dd)
                if len(dd_hist) == self._max_drawdown_trend_window and all(
                    dd_hist[i] > dd_hist[i - 1] for i in range(1, len(dd_hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "max_drawdown_deteriorating",
                        "strategy": name,
                        "max_drawdown": dd,
                        "message": (
                            f"策略 {name} 最大回撤连续 {self._max_drawdown_trend_window} 周期加深"
                            f"（当前 {dd:.2%}），风险深度扩大，提前降杠杆"
                        ),
                    })

            # 最大回撤绝对值守卫：max_drawdown 已超过阈值（绝对深度）但尚未连续加深时趋势守卫不触发，
            # 但策略实际在「深度回撤」状态（风险敞口过大、抗风险能力弱）。仅在交易笔数达 min_trades
            # 时评估（样本不足的回撤统计不稳定）。阈值型守卫（非趋势型），无跨周期状态。
            if (self._max_drawdown_guard_enabled
                    and trades >= self._max_drawdown_guard_min_trades):
                dd = safe_float(c.get("max_drawdown"), 0.0)
                if dd >= self._max_drawdown_guard_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "high_max_drawdown",
                        "strategy": name,
                        "max_drawdown": dd,
                        "message": (
                            f"策略 {name} 最大回撤 {dd:.2%} 超阈值 "
                            f"{self._max_drawdown_guard_threshold:.2%}，深度回撤，提前降杠杆收敛"
                        ),
                    })

            # 回撤持续时间趋势外推：追踪 max_drawdown_duration_hours 时序，连续上升（拉长）→
            # 恢复能力下降预警。与 max_drawdown_trend（回撤「深度」加深）区分：本守卫看回撤
            # 「时长」拉长——策略可能回撤不深但长时间无法收复前高（阴跌、慢恢复），是独立于
            # 回撤深度的「资金时间价值」维度。仅在交易笔数达 min_trades 时追踪。
            if self._drawdown_duration_trend_enabled and trades >= self._drawdown_duration_trend_min_trades:
                ddh = safe_float(c.get("max_drawdown_duration_hours"), 0.0)
                ddh_hist = self._strategy_drawdown_duration_history.setdefault(
                    str(name), deque(maxlen=self._drawdown_duration_trend_window)
                )
                ddh_hist.append(ddh)
                if len(ddh_hist) == self._drawdown_duration_trend_window and all(
                    ddh_hist[i] > ddh_hist[i - 1] for i in range(1, len(ddh_hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "drawdown_duration_deteriorating",
                        "strategy": name,
                        "max_drawdown_duration_hours": ddh,
                        "message": (
                            f"策略 {name} 回撤持续时长连续 {self._drawdown_duration_trend_window} 周期拉长"
                            f"（当前 {ddh:.1f}h），恢复能力下降，提前降杠杆"
                        ),
                    })

            # 回撤持续时间绝对阈值守卫：最长回撤持续 ≥ max_drawdown_hours（恢复能力差）→ 提前降杠杆。
            # 与 drawdown_duration_trend_guard（连续拉长，趋势型）区分：本守卫看「绝对阈值」——策略
            # 历史最长回撤持续已达阈值（资金被套牢过久、恢复能力差）但未连续拉长时趋势守卫不触发，
            # 但策略实际已证明「资金时间价值差」（回撤后迟迟无法收复前高），应提前收敛。
            # 仅在交易笔数达 min_trades 时评估（样本不足时回撤时长无意义）。阈值型守卫，无跨周期状态。
            if (self._drawdown_duration_guard_enabled
                    and trades >= self._drawdown_duration_guard_min_trades):
                ddh = safe_float(c.get("max_drawdown_duration_hours"), 0.0)
                if ddh >= self._drawdown_duration_guard_max_hours:
                    alerts.append({
                        "level": "warning",
                        "type": "long_drawdown_duration",
                        "strategy": name,
                        "max_drawdown_duration_hours": ddh,
                        "message": (
                            f"策略 {name} 最长回撤持续 {ddh:.1f}h ≥ "
                            f"{self._drawdown_duration_guard_max_hours:.0f}h，"
                            f"恢复能力差（资金被套牢过久），提前降杠杆收敛"
                        ),
                    })

            # 资本回报率趋势外推：追踪 pnl_per_capital_pct 时序，连续下降 → 资本配置效率恶化预警。
            # 与前五子区分：本守卫看「单位资本产出」——策略可能 health_grade=A（盈利、低回撤），
            # 但配置过多资本导致单位资本产出持续递减（pnl_per_capital_pct 下降），这是资本配置
            # 效率恶化的独立先行指标。仅在交易笔数达 min_trades 时追踪，避免样本不足噪声误触发。
            if self._capital_return_trend_enabled and trades >= self._capital_return_trend_min_trades:
                cr = safe_float(c.get("pnl_per_capital_pct"), 0.0)
                cr_hist = self._strategy_capital_return_history.setdefault(
                    str(name), deque(maxlen=self._capital_return_trend_window)
                )
                cr_hist.append(cr)
                if len(cr_hist) == self._capital_return_trend_window and all(
                    cr_hist[i] < cr_hist[i - 1] for i in range(1, len(cr_hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "capital_return_deteriorating",
                        "strategy": name,
                        "pnl_per_capital_pct": cr,
                        "message": (
                            f"策略 {name} 资本回报率连续 {self._capital_return_trend_window} 周期下降"
                            f"（当前 {cr:.2f}%），资本配置效率恶化，提前降杠杆"
                        ),
                    })

            # 资本回报率阈值守卫：单位资本产出低于绝对阈值（如 pnl_per_capital_pct < -10%）→ 收敛。
            # 与 capital_return_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」——资本
            # 回报率低于阈值即收敛，不必等待连续下降确认，构成资本效率维度「阈值 + 趋势」双层
            # （与 volatility_budget 补充 volatility_trend 对称）。仅在交易笔数达 min_trades 时评估。
            if (self._capital_return_guard_enabled
                    and trades >= self._capital_return_min_trades):
                cr = safe_float(c.get("pnl_per_capital_pct"), 0.0)
                if cr < self._capital_return_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "capital_return_low",
                        "strategy": name,
                        "pnl_per_capital_pct": cr,
                        "message": (
                            f"策略 {name} 资本回报率 {cr:.2f}% 低于阈值 "
                            f"{self._capital_return_threshold:.2f}%，单位资本产出过低，收敛"
                        ),
                    })

            # 波动率趋势外推：追踪 volatility（单笔盈亏标准差）时序，连续上升（波动放大）→ 事前预警。
            # volatility 是纯风险维度的独立信号——策略可能 PnL 正、胜率高、夏普高、盈亏比高，
            # 但波动率持续上升意味着单笔盈亏离散度扩大（承担更多不确定性），这是前六子守卫
            # （收益质量/风险深度/资本效率）无法捕获的独立衰退模式。仅在交易笔数达 min_trades 时追踪。
            if self._volatility_trend_enabled and trades >= self._volatility_trend_min_trades:
                vol = safe_float(c.get("volatility"), 0.0)
                vol_hist = self._strategy_volatility_history.setdefault(
                    str(name), deque(maxlen=self._volatility_trend_window)
                )
                vol_hist.append(vol)
                if len(vol_hist) == self._volatility_trend_window and all(
                    vol_hist[i] > vol_hist[i - 1] for i in range(1, len(vol_hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "volatility_deteriorating",
                        "strategy": name,
                        "volatility": vol,
                        "message": (
                            f"策略 {name} 波动率连续 {self._volatility_trend_window} 周期上升"
                            f"（当前 {vol:.2f}），单笔盈亏离散度扩大，提前降杠杆"
                        ),
                    })

            # 策略级 PnL 动量趋势外推：追踪 trend_pnl_7d_vs_30d 时序，连续下降 → 盈利动量衰竭。
            # 与七子守卫（health/win_rate/sharpe/profit_factor/max_drawdown/capital_return/volatility）
            # 区分：本守卫看「PnL 动量」——近7天 vs 近30天平均 PnL 比率连续下降，意味着策略近期
            # 盈利动能相对中期持续衰竭（虽仍盈利但增长停滞），是七子守卫无法捕获的独立先行指标。
            # 仅在交易笔数达 min_trades 且比率>0（有数据）时追踪，避免无数据（恒0）误触发。
            if self._pnl_momentum_trend_enabled and trades >= self._pnl_momentum_trend_min_trades:
                mom = safe_float(c.get("trend_pnl_7d_vs_30d"), 0.0)
                if mom > 0:
                    mom_hist = self._strategy_pnl_momentum_history.setdefault(
                        str(name), deque(maxlen=self._pnl_momentum_trend_window)
                    )
                    mom_hist.append(mom)
                    if len(mom_hist) == self._pnl_momentum_trend_window and all(
                        mom_hist[i] < mom_hist[i - 1] for i in range(1, len(mom_hist))
                    ):
                        alerts.append({
                            "level": "warning",
                            "type": "pnl_momentum_deteriorating",
                            "strategy": name,
                            "pnl_momentum": mom,
                            "message": (
                                f"策略 {name} PnL 动量连续 {self._pnl_momentum_trend_window} 周期下降"
                                f"（当前 {mom:.2f}），近期盈利动能衰竭，提前降杠杆"
                            ),
                        })

            # 策略级 PnL 动量绝对阈值守卫：动量 < min_momentum_ratio（近期动能相对中期明显减弱）→ 提前降杠杆。
            # 与 pnl_momentum_trend_guard（连续下降，趋势型）区分：本守卫看「绝对阈值」——动量已衰减到
            # 阈值以下但未连续下降（平坦低位）时趋势守卫不触发，但策略实际在「增长动能衰竭」。
            # 与 strategy_declining（需 grade D/F）区分：本守卫不看健康度，动量衰竭但健康度仍 A/B/C 也收敛。
            # 仅在交易笔数达 min_trades 且 0<动量<阈值（有数据且为正，排除无数据恒0）时评估。阈值型守卫。
            if (self._pnl_momentum_guard_enabled
                    and trades >= self._pnl_momentum_guard_min_trades):
                mom = safe_float(c.get("trend_pnl_7d_vs_30d"), 0.0)
                if 0 < mom < self._pnl_momentum_guard_min_ratio:
                    alerts.append({
                        "level": "warning",
                        "type": "pnl_momentum_faded",
                        "strategy": name,
                        "pnl_momentum": mom,
                        "message": (
                            f"策略 {name} PnL 动量衰竭（{mom:.2f} < "
                            f"{self._pnl_momentum_guard_min_ratio:.2f}），"
                            f"近期盈利动能相对中期明显减弱，提前降杠杆收敛"
                        ),
                    })

            # 策略级单笔期望值趋势外推：追踪 pnl_per_trade 时序，连续下降 → 交易质量恶化。
            # 与九子守卫区分：本守卫看「单笔期望」——pnl_per_trade（每笔平均盈亏）连续下降意味着
            # 策略每笔交易价值持续萎缩（靠更多交易堆砌总利润，规模换质量），是独立先行指标。
            # 仅在交易笔数达 min_trades 时追踪，避免小样本噪声误触发。
            if self._pnl_per_trade_trend_enabled and trades >= self._pnl_per_trade_trend_min_trades:
                ppt = safe_float(c.get("pnl_per_trade"), 0.0)
                ppt_hist = self._strategy_pnl_per_trade_history.setdefault(
                    str(name), deque(maxlen=self._pnl_per_trade_trend_window)
                )
                ppt_hist.append(ppt)
                if len(ppt_hist) == self._pnl_per_trade_trend_window and all(
                    ppt_hist[i] < ppt_hist[i - 1] for i in range(1, len(ppt_hist))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "pnl_per_trade_deteriorating",
                        "strategy": name,
                        "pnl_per_trade": ppt,
                        "message": (
                            f"策略 {name} 单笔期望值连续 {self._pnl_per_trade_trend_window} 周期下降"
                            f"（当前 {ppt:.4f}），每笔交易价值持续萎缩，提前降杠杆"
                        ),
                    })

            # 单笔期望值绝对值守卫：pnl_per_trade 已低于 min_pnl_per_trade（绝对阈值）但尚未连续
            # 下降时趋势守卫不触发，但策略实际在「每笔交易无正期望」状态（交易质量差，靠规模堆砌
            # 或亏损）。仅在交易笔数达 min_trades 时评估。阈值型守卫（非趋势型），无跨周期状态。
            if (self._pnl_per_trade_guard_enabled
                    and trades >= self._pnl_per_trade_guard_min_trades):
                ppt = safe_float(c.get("pnl_per_trade"), 0.0)
                if ppt < self._pnl_per_trade_guard_min_value:
                    alerts.append({
                        "level": "warning",
                        "type": "low_pnl_per_trade",
                        "strategy": name,
                        "pnl_per_trade": ppt,
                        "message": (
                            f"策略 {name} 单笔期望值 {ppt:.4f} 低于阈值 "
                            f"{self._pnl_per_trade_guard_min_value:.4f}，每笔交易无正期望，提前降杠杆收敛"
                        ),
                    })

            # 策略级边际盈亏趋势外推：追踪 delta_pnl 时序，连续为负 → 持续失血。
            # 与十子守卫区分：本守卫看「边际盈亏」——delta_pnl（本周期 vs 上一快照的盈亏增量）
            # 连续为负意味着策略最近持续失血（即使累计 total_pnl 仍为正，近期也在亏），是周期级
            # 先行指标。仅在交易笔数达 min_trades 时追踪，避免小样本噪声误触发。
            if self._delta_pnl_trend_enabled and trades >= self._delta_pnl_trend_min_trades:
                dp = safe_float(c.get("delta_pnl"), 0.0)
                dp_hist = self._strategy_delta_pnl_history.setdefault(
                    str(name), deque(maxlen=self._delta_pnl_trend_window)
                )
                dp_hist.append(dp)
                if len(dp_hist) == self._delta_pnl_trend_window and all(x < 0 for x in dp_hist):
                    alerts.append({
                        "level": "warning",
                        "type": "delta_pnl_deteriorating",
                        "strategy": name,
                        "delta_pnl": dp,
                        "message": (
                            f"策略 {name} 边际盈亏连续 {self._delta_pnl_trend_window} 周期为负"
                            f"（当前 {dp:.4f}），持续失血，提前降杠杆"
                        ),
                    })

            # 单周期亏损急跌守卫：delta_pnl 单周期大额急跌（|delta_pnl|/equity ≥ loss_threshold）→ 提前降杠杆。
            # 与 delta_pnl_trend_guard（连续 N 周期为负，趋势型）区分：本守卫看「绝对阈值」——策略单周期
            # 突然大幅亏损（急跌尖峰）但未连续为负时趋势守卫不触发，但策略实际在「单周期急跌」（突发不利行情）。
            # 仅在交易笔数达 min_trades 且 delta_pnl<0 时评估（|delta_pnl|/equity 归一化，阈值型守卫）。
            if (self._delta_pnl_guard_enabled
                    and trades >= self._delta_pnl_guard_min_trades):
                dp = safe_float(c.get("delta_pnl"), 0.0)
                eq = safe_float(equity, 0.0)
                if dp < 0 and eq > 0:
                    loss_ratio = -dp / eq
                    if loss_ratio >= self._delta_pnl_guard_loss_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "delta_pnl_spike",
                            "strategy": name,
                            "delta_pnl": dp,
                            "loss_ratio": loss_ratio,
                            "message": (
                                f"策略 {name} 单周期急跌 {dp:.4f}（占权益 {loss_ratio:.1%} ≥ "
                                f"{self._delta_pnl_guard_loss_threshold:.1%}），突发大幅亏损，提前降杠杆收敛"
                            ),
                        })

            # 利润回吐保护：追踪累计盈亏峰值，浮盈从峰值回吐超阈值 → 减仓锁利
            if self._give_back_enabled:
                # 峰值时间衰减：陈旧峰值每 peak_decay_cycles 周期衰减一次（×0.9），
                # 使回吐保护以近期峰值（而非历史最高点）为基准，避免旧高盈利峰值
                # 长期压制策略恢复（回吐基线永久虚高）。
                if self._give_back_peak_decay_cycles > 0:
                    last_decay = self._give_back_peak_last_decay_cycle.get(str(name), 0)
                    elapsed = self._cycle_count - last_decay
                    if elapsed >= self._give_back_peak_decay_cycles:
                        self._strategy_pnl_peak[str(name)] = (
                            self._strategy_pnl_peak.get(str(name), 0.0)
                            * self._give_back_peak_decay_alpha
                        )
                        self._give_back_peak_last_decay_cycle[str(name)] = self._cycle_count
                peak = self._strategy_pnl_peak.get(str(name), 0.0)
                peak = max(peak, total_pnl)
                self._strategy_pnl_peak[str(name)] = peak
                if peak > 0 and total_pnl <= peak * (1.0 - self._give_back_threshold):
                    alerts.append({
                        "level": "warning",
                        "type": "pnl_give_back",
                        "strategy": name,
                        "peak_pnl": peak,
                        "current_pnl": total_pnl,
                        "message": (
                            f"策略 {name} 利润从峰值 {peak:.2f} 回吐至 {total_pnl:.2f}"
                            f"（超 {self._give_back_threshold:.0%}），建议减仓锁利"
                        ),
                    })
                    # 严重回吐：回吐超 severe_threshold → 更激进减仓
                    if (self._give_back_severe_enabled
                            and total_pnl <= peak * (1.0 - self._give_back_severe_threshold)):
                        alerts.append({
                            "level": "high",
                            "type": "pnl_give_back_severe",
                            "strategy": name,
                            "peak_pnl": peak,
                            "current_pnl": total_pnl,
                            "message": (
                                f"策略 {name} 利润严重回吐：峰值 {peak:.2f} → {total_pnl:.2f}"
                                f"（超 {self._give_back_severe_threshold:.0%}），"
                                f"建议大幅减仓"
                            ),
                        })
                    # 危急回吐：回吐超 critical_threshold → 暂停策略开新仓
                    if (self._give_back_critical_enabled
                            and total_pnl <= peak * (1.0 - self._give_back_critical_threshold)):
                        alerts.append({
                            "level": "critical",
                            "type": "pnl_give_back_critical",
                            "strategy": name,
                            "peak_pnl": peak,
                            "current_pnl": total_pnl,
                            "message": (
                                f"策略 {name} 利润危急回吐：峰值 {peak:.2f} → {total_pnl:.2f}"
                                f"（超 {self._give_back_critical_threshold:.0%}），"
                                f"暂停开新仓止血"
                            ),
                        })

            # 浮盈占比过高检测：策略账面盈利主要靠未兑现浮盈支撑 → 盈利脆弱性预警。
            # 与 give_back（浮盈已回吐，事后/时序）区分：本守卫看「浮盈占比过高」（事前/
            # 结构），在回吐发生之前预警。阈值型守卫，无跨周期状态。
            if self._unrealized_profit_ratio_enabled and total_pnl > 0:
                upl = safe_float(c.get("unrealized_pnl"), 0.0)
                if upl > 0 and upl / total_pnl > self._unrealized_profit_ratio_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "unrealized_profit_concentration",
                        "strategy": name,
                        "unrealized_pnl": upl,
                        "total_pnl": total_pnl,
                        "message": (
                            f"策略 {name} 浮盈占比 {upl / total_pnl:.0%} 过高"
                            f"（浮盈 {upl:.2f}/{total_pnl:.2f}），盈利脆弱需锁定浮盈"
                        ),
                    })

            # 浮盈占比趋势外推：追踪 unrealized_pnl/total_pnl 时序，连续上升 → 盈利质量劣化预警。
            # 与 unrealized_profit_ratio_guard（阈值型：超 ratio_threshold 才收敛）区分：本守卫看
            # 「浮盈占比持续上升」轨迹（事前），尚未触及阈值即锁定浮盈。与 unrealized_loss_trend
            # （浮亏连续加深，亏损侧）形成对称。total_pnl<=0 或 upl<=0 时 ratio 记为 0（重置趋势）。
            if self._unrealized_profit_ratio_trend_enabled and total_pnl > 0:
                upl = safe_float(c.get("unrealized_pnl"), 0.0)
                ratio = (upl / total_pnl) if upl > 0 else 0.0
                ratio_hist = self._strategy_unrealized_profit_ratio_history.setdefault(
                    str(name), deque(maxlen=self._unrealized_profit_ratio_trend_window)
                )
                ratio_hist.append(ratio)
                if (
                    len(ratio_hist) == self._unrealized_profit_ratio_trend_window
                    and ratio > 0
                    and all(ratio_hist[i] > ratio_hist[i - 1] for i in range(1, len(ratio_hist)))
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "unrealized_profit_ratio_rising",
                        "strategy": name,
                        "unrealized_profit_ratio": ratio,
                        "message": (
                            f"策略 {name} 浮盈占比连续 {self._unrealized_profit_ratio_trend_window} 周期上升"
                            f"（当前 {ratio:.0%}），盈利质量劣化，提前锁定浮盈"
                        ),
                    })

            if lifecycle == "dormant":
                alerts.append({
                    "level": "info",
                    "type": "strategy_dormant",
                    "strategy": name,
                    "message": f"策略 {name} 已休眠，建议回收资金",
                })
            if grade == "F":
                alerts.append({
                    "level": "critical",
                    "type": "strategy_health_critical",
                    "strategy": name,
                    "message": f"策略 {name} 健康度为 F，建议暂停审查",
                })
            if trend == "declining" and grade in ("D", "F"):
                alerts.append({
                    "level": "warning",
                    "type": "strategy_declining",
                    "strategy": name,
                    "message": f"策略 {name} 趋势恶化且健康度 {grade}，需密切关注",
                })
            if trend == "improving" and grade in ("A", "B"):
                # 恢复纯度门控：仅当本周期无趋势型衰退告警时才判定「已恢复」，
                # 避免「健康度改善」与「健康度/胜率/夏普等趋势下降」矛盾信号共存，
                # 从而消除 pause/resume 震荡。
                if self._recovery_purity_enabled and any(
                    a.get("strategy") == name and a.get("type") in _DECAY_ALERT_TYPES
                    for a in alerts
                ):
                    pass  # 存在衰退信号，抑制 recovered
                else:
                    alerts.append({
                        "level": "info",
                        "type": "strategy_recovered",
                        "strategy": name,
                        "grade": grade,
                        "message": f"策略 {name} 健康度改善至 {grade}，可考虑恢复",
                    })
            if trades > 0 and total_pnl < 0:
                alerts.append({
                    "level": "warning",
                    "type": "possible_consecutive_losses",
                    "strategy": name,
                    "message": f"策略 {name} 累计亏损 {total_pnl:.2f}（{trades} 笔），存在连续亏损风险",
                })

            # 多守卫共振：同一策略本周期触发 ≥ resonance_threshold 个趋势守卫 → 升级收敛。
            # 扫描本周期为该策略生成的趋势型衰退告警（health/win_rate/sharpe/profit_factor/
            # max_drawdown），多维度同时衰退比单维度严重得多，需要更大力度的降杠杆。
            if self._resonance_enabled:
                decay_types = {
                    a.get("type") for a in alerts
                    if a.get("strategy") == name and a.get("type") in _DECAY_ALERT_TYPES
                }
                if len(decay_types) >= self._resonance_threshold:
                    alerts.append({
                        "level": "critical",
                        "type": "multi_guard_resonance",
                        "strategy": name,
                        "decay_types": sorted(decay_types),
                        "message": (
                            f"策略 {name} 多维度共振衰退（{len(decay_types)} 维："
                            f"{'、'.join(sorted(decay_types))}），升级降杠杆收敛"
                        ),
                    })

        # 组合级盈利集中度：盈利来源是否过度依赖单一策略（单一支柱风险）。
        # 与 correlation（策略间收益相关性）和 diversification（资金配置 HHI）区分：
        # 本维度看「盈利来源的集中度」——即使资金分散（HHI 低）、策略不相关，
        # 若组合盈利过度依赖单一策略，一旦该支柱衰退，组合将无支撑快速转亏。
        if self._profit_concentration_enabled:
            strat_map = _coerce_dict(contrib.get("strategies"))
            total_pnl = safe_float(contrib.get("total_pnl"), 0.0)
            if total_pnl > 0:
                max_pnl = 0.0
                dominant = None
                for sname, sc in strat_map.items():
                    if not isinstance(sc, dict):
                        continue
                    pnl = safe_float(sc.get("total_pnl"), 0.0)
                    if pnl > max_pnl:
                        max_pnl = pnl
                        dominant = sname
                if max_pnl > 0:
                    concentration = max_pnl / total_pnl
                    if concentration > self._profit_concentration_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "profit_concentration",
                            "strategy": dominant,
                            "concentration": concentration,
                            "message": (
                                f"组合盈利集中度 {concentration:.0%}，过度依赖策略 "
                                f"{dominant}，单一支柱风险"
                            ),
                        })

        # 组合级盈利集中度趋势外推：追踪 profit_concentration（max_pnl/total_pnl）时序，连续
        # 上升 → 单一盈利支柱风险加剧。与 profit_concentration_guard（阈值型，事后）区分：本守卫
        # 看「盈利来源集中度持续上升」轨迹（事前），尚未触及 threshold 即提前收敛。total_pnl<=0
        # 时 concentration 记为 0（盈利转负，集中度度量失效并重置趋势，避免陈旧高值污染）。
        if self._profit_concentration_trend_enabled:
            strat_map = _coerce_dict(contrib.get("strategies"))
            total_pnl = safe_float(contrib.get("total_pnl"), 0.0)
            concentration = 0.0
            dominant = None
            if total_pnl > 0:
                max_pnl = 0.0
                for sname, sc in strat_map.items():
                    if not isinstance(sc, dict):
                        continue
                    pnl = safe_float(sc.get("total_pnl"), 0.0)
                    if pnl > max_pnl:
                        max_pnl = pnl
                        dominant = sname
                if max_pnl > 0:
                    concentration = max_pnl / total_pnl
            self._portfolio_profit_concentration_history.append(concentration)
            if (
                len(self._portfolio_profit_concentration_history) == self._profit_concentration_trend_window
                and all(
                    self._portfolio_profit_concentration_history[i] > self._portfolio_profit_concentration_history[i - 1]
                    for i in range(1, len(self._portfolio_profit_concentration_history))
                )
            ):
                alerts.append({
                    "level": "warning",
                    "type": "profit_concentration_rising",
                    "strategy": dominant,
                    "concentration": concentration,
                    "message": (
                        f"组合盈利来源集中度连续 {self._profit_concentration_trend_window} 周期上升"
                        f"（当前 {concentration:.0%}），单一盈利支柱风险加剧，收敛支柱敞口"
                    ),
                })

        # 组合级尾部风险（CVaR 代理）：组合中最深策略回撤 max(max_drawdown_i) 作为尾部亏损的
        # 保守下界。与 max_drawdown_trend_guard（单策略回撤趋势）区分：本守卫看组合级回撤的
        # 绝对水平——即使各策略回撤未在加深，只要组合中存在一个高回撤策略，尾部风险就高。
        if self._tail_risk_enabled:
            strat_map = _coerce_dict(contrib.get("strategies"))
            max_dd = 0.0
            tail_strategy = None
            for sname, sc in strat_map.items():
                if not isinstance(sc, dict):
                    continue
                dd = safe_float(sc.get("max_drawdown"), 0.0)
                if dd > max_dd:
                    max_dd = dd
                    tail_strategy = sname
            if max_dd > self._tail_risk_threshold:
                alerts.append({
                    "level": "warning",
                    "type": "high_tail_risk",
                    "strategy": tail_strategy,
                    "tail_risk": max_dd,
                    "message": (
                        f"组合尾部风险偏高：策略 {tail_strategy} 最大回撤 {max_dd:.1%} "
                        f"超过阈值 {self._tail_risk_threshold:.1%}，需收敛尾部暴露"
                    ),
                })

        # 组合级协同度收敛：synergy_score 低于阈值 → 策略组合协同劣化（相关地一起亏）。
        # 与 correlation（高相关）、diversification（权重集中）区分：本守卫看「协同质量」——
        # 综合相关性 + 盈亏同向性，捕捉策略「相关且一起亏」的相互拖累。仅策略数 ≥ min_strategies
        # 时评估（synergy 需多策略才有意义，无相关性数据时 synergy 默认 0.5 不会误触发）。
        if self._synergy_enabled:
            strat_map = _coerce_dict(contrib.get("strategies"))
            synergy = safe_float(contrib.get("synergy_score"), 0.5)
            if len(strat_map) >= self._synergy_min_strategies and synergy < self._synergy_threshold:
                alerts.append({
                    "level": "warning",
                    "type": "low_synergy",
                    "synergy_score": synergy,
                    "message": (
                        f"组合协同度 {synergy:.2f} 低于阈值 {self._synergy_threshold:.2f}"
                        f"，策略相关且一起亏（协同劣化），收敛组合敞口防相互拖累"
                    ),
                })

        # 组合级协同度趋势外推：追踪 synergy_score 时序，连续下降 → 协同质量劣化预警。
        # 与 synergy_guard（阈值型：跌破阈值才收敛）区分：本守卫看「协同质量持续下降」轨迹
        # （事前），尚未跌破阈值即收敛。仅策略数 ≥ min_strategies 时追踪（无相关性数据时
        # synergy 默认 0.5 稳定不下降）。
        if self._synergy_trend_enabled:
            strat_map = _coerce_dict(contrib.get("strategies"))
            synergy = safe_float(contrib.get("synergy_score"), 0.5)
            if len(strat_map) >= self._synergy_trend_min_strategies:
                self._portfolio_synergy_history.append(synergy)
                if (
                    len(self._portfolio_synergy_history) == self._synergy_trend_window
                    and all(
                        self._portfolio_synergy_history[i] < self._portfolio_synergy_history[i - 1]
                        for i in range(1, len(self._portfolio_synergy_history))
                    )
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "synergy_deteriorating",
                        "synergy_score": synergy,
                        "message": (
                            f"组合协同度连续 {self._synergy_trend_window} 周期下降"
                            f"（当前 {synergy:.2f}），协同质量劣化，收敛组合敞口防相互拖累"
                        ),
                    })

        # 组合级资金效率趋势外推：追踪 efficiency_score 时序，连续下降 → 资金效率劣化预警。
        # 与 portfolio_efficiency_guard（阈值型：跌破阈值才收敛，事后）区分：本守卫看「资金
        # 效率持续下降」轨迹（事前），尚未跌破阈值即收敛；与 portfolio_health_trend（组合
        # 健康度下降，看收益/风险质量）区分：本守卫看资金利用效率（费率/资本回报/风险调整）。
        # 仅在成交数 ≥ min_sample 时追踪（无成交时 efficiency 恒 0 稳定不下降）。
        if self._portfolio_efficiency_trend_enabled:
            eff_total_trades = safe_int(contrib.get("total_trades"), 0)
            efficiency = safe_float(contrib.get("efficiency_score"), 0.0)
            if eff_total_trades >= self._portfolio_efficiency_trend_min_sample:
                self._portfolio_efficiency_history.append(efficiency)
                if (
                    len(self._portfolio_efficiency_history) == self._portfolio_efficiency_trend_window
                    and all(
                        self._portfolio_efficiency_history[i] < self._portfolio_efficiency_history[i - 1]
                        for i in range(1, len(self._portfolio_efficiency_history))
                    )
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "efficiency_deteriorating",
                        "efficiency_score": efficiency,
                        "message": (
                            f"组合资金效率连续 {self._portfolio_efficiency_trend_window} 周期下降"
                            f"（当前 {efficiency:.0%}），资金利用效率劣化，收敛组合敞口释放低效占用"
                        ),
                    })

        # 组合级尾部风险趋势外推：追踪 tail_risk（最深单策略回撤）时序，连续上升 → 尾部风险
        # 加深预警。与 tail_risk_guard（阈值型：max_dd 超阈值才收敛）区分：本守卫看「尾部风险
        # 持续加深」轨迹（事前），尚未触及阈值即收敛；与 max_drawdown_trend_guard（单策略回撤
        # 趋势）区分：本守卫看组合级最深回撤（尾部），后者看每个策略自身回撤。
        if self._portfolio_tail_risk_trend_enabled:
            strat_map = _coerce_dict(contrib.get("strategies"))
            max_dd = 0.0
            for sc in strat_map.values():
                if isinstance(sc, dict):
                    max_dd = max(max_dd, safe_float(sc.get("max_drawdown"), 0.0))
            self._portfolio_tail_risk_history.append(max_dd)
            if (
                len(self._portfolio_tail_risk_history) == self._portfolio_tail_risk_trend_window
                and all(
                    self._portfolio_tail_risk_history[i] > self._portfolio_tail_risk_history[i - 1]
                    for i in range(1, len(self._portfolio_tail_risk_history))
                )
            ):
                alerts.append({
                    "level": "warning",
                    "type": "tail_risk_rising",
                    "tail_risk": max_dd,
                    "message": (
                        f"组合尾部风险连续 {self._portfolio_tail_risk_trend_window} 周期加深"
                        f"（当前最深回撤 {max_dd:.0%}），收敛敞口防尾部暴露"
                    ),
                })

        # 连续亏损（借助 dynamic_allocator 的 streaks 检测，若可用）
        if self.dynamic_allocator is not None and _coerce_dict(contrib.get("strategies")):
            try:
                check = getattr(self.dynamic_allocator, "check_consecutive_streaks", None)
                if callable(check):
                    streaks = check(build_strategy_metrics(contrib))
                    for name, s in (streaks or {}).items():
                        if isinstance(s, dict) and s.get("action") == "decrease":
                            alerts.append({
                                "level": "warning",
                                "type": "consecutive_losses",
                                "strategy": name,
                                "message": f"策略 {name} 连续亏损，建议减配",
                                "detail": {k: safe_finite(v, 0.0) if isinstance(v, (int, float)) else v
                                           for k, v in s.items()},
                            })
            except Exception as e:
                logger.debug(f"[AGI-Diagnose] consecutive streaks check failed: {e}")

        # 冻结策略观察期 / 缩量试探 / 永久冻结（AGI 能力：感知自动解冻状态机）
        freeze_state = _coerce_dict(perception.get("freeze_state"))
        for name, fs in freeze_state.items():
            if not isinstance(fs, dict):
                continue
            reason = fs.get("reason", "")
            probing = bool(fs.get("probing", False))
            attempts = safe_int(fs.get("probe_attempts"), 0)
            max_attempts = safe_int(fs.get("probe_max_attempts"), 0)
            remaining = safe_float(fs.get("remaining_seconds"), 0.0)
            if fs.get("permanently_frozen"):
                alerts.append({
                    "level": "warning",
                    "type": "strategy_permanently_frozen",
                    "strategy": name,
                    "message": f"策略 {name} 缩量试探 {max_attempts} 次仍未恢复，永久冻结需人工确认",
                })
            elif probing:
                alerts.append({
                    "level": "info",
                    "type": "strategy_probing",
                    "strategy": name,
                    "message": f"策略 {name} 缩量试探中（第 {attempts}/{max_attempts} 次）",
                })
            else:
                alerts.append({
                    "level": "info",
                    "type": "strategy_frozen_observing",
                    "strategy": name,
                    "message": (
                        f"策略 {name} 已冻结（{reason}），"
                        f"观察期剩余 {remaining / 60:.0f} 分钟"
                    ),
                })

        # 资金集中度 / 分散化 / 资金效率（贡献快照跨策略指标）
        if contrib:
            concentration = safe_float(contrib.get("concentration"), 0.0)
            diversification = safe_float(contrib.get("diversification_score"), 0.0)
            efficiency = safe_float(contrib.get("efficiency_score"), 0.0)
            total_trades = safe_int(contrib.get("total_trades"), 0)

            if concentration > self.concentration_high_threshold:
                alerts.append({
                    "level": "warning",
                    "type": "high_concentration",
                    "concentration": concentration,
                    "message": (
                        f"资金集中度过高 {concentration:.0%}（HHI > "
                        f"{self.concentration_high_threshold:.0%}），单一策略风险敞口过大"
                    ),
                })
            # 仅当存在多个策略时，分散化评分才有意义
            if len(_coerce_dict(contrib.get("strategies"))) > 1 and diversification < self.diversification_low_threshold:
                alerts.append({
                    "level": "warning",
                    "type": "low_diversification",
                    "diversification_score": diversification,
                    "message": (
                        f"策略分散化不足 {diversification:.0%}（< "
                        f"{self.diversification_low_threshold:.0%}），组合相关性风险偏高"
                    ),
                })
            # 无成交时效率恒为 0，仅在存在成交记录时才评估资金效率
            if total_trades > 0 and efficiency < self.capital_efficiency_low_threshold:
                alerts.append({
                    "level": "info",
                    "type": "low_capital_efficiency",
                    "efficiency_score": efficiency,
                    "message": (
                        f"资金效率偏低 {efficiency:.0%}（< "
                        f"{self.capital_efficiency_low_threshold:.0%}）"
                    ),
                })

            # 交易成本感知：手续费占毛利比例过高 → 交易过频侵蚀利润
            if self._cost_awareness_enabled:
                total_fees = safe_float(contrib.get("total_fees"), 0.0)
                total_pnl = safe_float(contrib.get("total_pnl"), 0.0)
                gross = total_pnl + total_fees  # 毛利（手续费前利润）
                if gross > 0 and total_fees > 0:
                    fee_ratio = total_fees / gross
                    if fee_ratio >= self._cost_awareness_threshold:
                        alerts.append({
                            "level": "warning",
                            "type": "high_trading_cost",
                            "fee_ratio": fee_ratio,
                            "total_fees": total_fees,
                            "message": (
                                f"手续费侵蚀过高（{fee_ratio:.0%} ≥ "
                                f"{self._cost_awareness_threshold:.0%}），交易过频消耗资金，"
                                f"建议降低交易频率"
                            ),
                        })

        # 策略间相关性（StrategyCorrelationAnalyzer）：高相关 / 有效 N 侵蚀 → 组合相关性风险
        if self._correlation_enabled:
            corr = perception.get("correlation") or {}
            max_pair = safe_float(corr.get("max_pair_correlation"), 0.0)
            avg_corr = safe_float(corr.get("average_correlation"), 0.0)
            if max_pair >= self._correlation_high_threshold:
                alerts.append({
                    "level": "warning",
                    "type": "high_correlation",
                    "max_pair_correlation": max_pair,
                    "average_correlation": avg_corr,
                    "message": (
                        f"策略间相关性过高（最大 pair {max_pair:.0%} ≥ "
                        f"{self._correlation_high_threshold:.0%}），组合分散失效"
                    ),
                })
            if bool(corr.get("diversification_erosion", False)):
                alerts.append({
                    "level": "warning",
                    "type": "diversification_eroding",
                    "effective_n": safe_float(corr.get("effective_n"), 0.0),
                    "message": (
                        "有效 N 持续侵蚀（策略收益趋同），组合实际分散度下降"
                    ),
                })

        # 组合级相关性趋势外推：追踪 max_pair_correlation 时序，连续上升 → 联动风险加剧预警。
        # 与 correlation（阈值型：max_pair 超阈值才收敛）区分：本守卫在相关性触及阈值之前
        # 收敛（事前）。与 portfolio_concentration_trend（集中度上升）区分：本守卫看策略间
        # 收益联动（相关性），集中度看资金配置权重，二者正交。
        if self._portfolio_correlation_trend_enabled:
            corr = perception.get("correlation") or {}
            max_pair = safe_float(corr.get("max_pair_correlation"), 0.0)
            self._portfolio_correlation_history.append(max_pair)
            if (
                len(self._portfolio_correlation_history) == self._portfolio_correlation_trend_window
                and all(
                    self._portfolio_correlation_history[i] > self._portfolio_correlation_history[i - 1]
                    for i in range(1, len(self._portfolio_correlation_history))
                )
            ):
                alerts.append({
                    "level": "warning",
                    "type": "portfolio_correlation_rising",
                    "max_pair_correlation": max_pair,
                    "message": (
                        f"策略间相关性连续 {self._portfolio_correlation_trend_window} 周期上升"
                        f"（当前 max pair {max_pair:.0%}），联动风险加剧，收敛敞口防同步下跌"
                    ),
                })

        # 组合级风险共振：多维度组合风险同时恶化 → 升级全局收敛。
        # 与策略级 resonance_guard（单策略多维衰退共振）对称：组合级看多维度组合风险
        # （相关性/权重集中/盈利集中/尾部风险）同时触发，比单维度风险严重得多，需要
        # 更保守的全局收敛。按独立维度去重计数（避免同一维度多告警重复计数）。
        if self._portfolio_resonance_enabled:
            triggered_dims = {
                dim for dim, types in _PORTFOLIO_RISK_DIMENSIONS.items()
                if any(a.get("type") in types for a in alerts)
            }
            if len(triggered_dims) >= self._portfolio_resonance_threshold:
                alerts.append({
                    "level": "critical",
                    "type": "portfolio_risk_resonance",
                    "dimensions": sorted(triggered_dims),
                    "message": (
                        f"组合级风险多维度共振（{len(triggered_dims)} 维："
                        f"{'、'.join(sorted(triggered_dims))}），升级全局收敛"
                    ),
                })

        # 组合级健康度阈值收敛：overall_health_score 跌破健康线（绝对值差）→ 组合级健康度差。
        # 与 portfolio_health_trend_guard（趋势型：连续下降）区分：本守卫看组合整体健康度的
        # 「绝对水平」——即使未在连续下降，只要已跌破 health_threshold 就应收敛。与策略级
        # strategy_health_critical/strategy_declining（grade F/D 阈值型）对称：组合级补足阈值层。
        if self._portfolio_health_enabled:
            total_trades = safe_int(contrib.get("total_trades"), 0)
            if total_trades >= self._portfolio_health_min_sample:
                oh = safe_float(contrib.get("overall_health_score"), 0.0)
                if oh < self._portfolio_health_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "portfolio_health_low",
                        "health_score": oh,
                        "message": (
                            f"组合整体健康度 {oh:.0f} 低于阈值 {self._portfolio_health_threshold:.0f}"
                            f"，组合级健康度差，收敛敞口防恶化"
                        ),
                    })

        # 组合级健康度趋势外推：追踪 overall_health_score 时序，连续下降 → 组合级衰退预警。
        # 与策略级 health_trend_guard 对称：策略级看单策略 health_score 连续下降，本守卫看
        # 组合整体健康度连续下降（可能所有策略轻微衰退，单策略均未触发）——温水煮青蛙式
        # 组合级衰退。仅在成交数达 min_sample 时追踪，避免样本不足的噪声误触发。
        if self._portfolio_health_trend_enabled:
            total_trades = safe_int(contrib.get("total_trades"), 0)
            if total_trades >= self._portfolio_health_trend_min_sample:
                oh = safe_float(contrib.get("overall_health_score"), 0.0)
                self._portfolio_health_history.append(oh)
                if (
                    len(self._portfolio_health_history) == self._portfolio_health_trend_window
                    and all(
                        self._portfolio_health_history[i] < self._portfolio_health_history[i - 1]
                        for i in range(1, len(self._portfolio_health_history))
                    )
                ):
                    alerts.append({
                        "level": "warning",
                        "type": "portfolio_health_deteriorating",
                        "health_score": oh,
                        "message": (
                            f"组合整体健康度连续 {self._portfolio_health_trend_window} 周期下降"
                            f"（当前 {oh:.0f}），组合级衰退，收敛敞口防恶化"
                        ),
                    })

        # 组合级集中度趋势外推：追踪 concentration（HHI）时序，连续上升 → 分散度侵蚀预警。
        # 与 portfolio_health_trend（健康度下降）区分：本守卫看资金配置集中度（结构维度）。
        # HHI 是结构性指标，无需成交数门控（无成交时 concentration 为 0，稳定不上升）。
        if self._portfolio_concentration_trend_enabled:
            concentration = safe_float(contrib.get("concentration"), 0.0)
            self._portfolio_concentration_history.append(concentration)
            if (
                len(self._portfolio_concentration_history) == self._portfolio_concentration_trend_window
                and all(
                    self._portfolio_concentration_history[i] > self._portfolio_concentration_history[i - 1]
                    for i in range(1, len(self._portfolio_concentration_history))
                )
            ):
                alerts.append({
                    "level": "warning",
                    "type": "portfolio_concentration_rising",
                    "concentration": concentration,
                    "message": (
                        f"组合集中度连续 {self._portfolio_concentration_trend_window} 周期上升"
                        f"（当前 HHI {concentration:.0%}），分散度被侵蚀，收敛敞口防集中"
                    ),
                })

        # 市场状态突变（结合置信度过滤：低置信度只作「疑似变化」提示，不触发突变告警）
        market_regime = perception.get("market_regime") or {}
        regime = market_regime.get("regime")
        confidence = safe_float(market_regime.get("confidence"), 0.0)
        if regime:
            if self._last_regime is not None and regime != self._last_regime:
                if confidence >= self.regime_confidence_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "regime_shift",
                        "confidence": confidence,
                        "message": (
                            f"市场状态突变：{self._last_regime} → {regime} "
                            f"(置信度 {confidence:.0%})"
                        ),
                    })
                    # 记录突变周期，供进攻冷却（突变后暂停进攻，等趋势重新确认）
                    self._last_regime_shift_cycle = self._cycle_count
                else:
                    alerts.append({
                        "level": "info",
                        "type": "regime_shift_low_confidence",
                        "confidence": confidence,
                        "message": (
                            f"市场状态疑似变化：{self._last_regime} → {regime}，"
                            f"置信度偏低 {confidence:.0%}，暂不触发突变告警"
                        ),
                    })
            self._last_regime = str(regime)

        # 账户级收益检测自动平仓（profit_take）：账户整体浮盈达阈值 → 生成平仓建议
        if self._profit_take_enabled:
            upl = safe_float(perception.get("unrealized_pnl"), 0.0)
            eq = safe_float(perception.get("equity"), 0.0)
            if eq > 0 and upl > 0:
                upl_pct = upl / eq
                # 市场状态自适应 + 学习记忆：动态调整落袋激活阈值
                regime_now = (perception.get("market_regime") or {}).get("regime")
                activation_pct = self._effective_profit_take_activation(regime_now)
                if upl_pct >= activation_pct:
                    span = max(self._profit_take_max_pct - activation_pct, 1e-6)
                    intensity = min(1.0, (upl_pct - activation_pct) / span)
                    close_ratio = self._profit_take_max_close_ratio * intensity
                    alerts.append({
                        "level": "info",
                        "type": "profit_take_signal",
                        "unrealized_pnl": upl,
                        "unrealized_pnl_pct": upl_pct,
                        "close_ratio": close_ratio,
                        "activation_pct": activation_pct,
                        "message": (
                            f"账户整体浮盈 {upl_pct:.1%}，建议落袋 {close_ratio:.0%} 仓位"
                        ),
                    })

        # 目标导向规划（goal_planning）：收益目标达成 / 接近最大回撤约束
        if self._goal_planning_enabled:
            eq = safe_float(perception.get("equity"), 0.0)
            if eq > 0:
                gs = self._update_goal_planning_state(eq, perception.get("total_capital"))
                if gs["return_pct"] >= self._goal_daily_target_pct:
                    alerts.append({
                        "level": "info",
                        "type": "goal_reached",
                        "return_pct": gs["return_pct"],
                        "target_pct": self._goal_daily_target_pct,
                        "message": (
                            f"收益目标达成 {gs['return_pct']:.1%} "
                            f"(目标 {self._goal_daily_target_pct:.1%})，建议减仓落袋"
                        ),
                    })
                near_dd = self._goal_max_drawdown_pct * self._goal_near_drawdown_ratio
                if gs["drawdown"] >= near_dd:
                    alerts.append({
                        "level": "warning",
                        "type": "near_drawdown_limit",
                        "drawdown_pct": gs["drawdown"],
                        "limit_pct": self._goal_max_drawdown_pct,
                        "message": (
                            f"回撤 {gs['drawdown']:.1%} 接近约束 "
                            f"{self._goal_max_drawdown_pct:.1%}，建议降仓防守"
                        ),
                    })

        # 自主进攻性资金分配：趋势确认 + 账户健康 + A/B 健康策略 + 风险偏好充足 → 进攻机会
        if self._offensive_enabled:
            regime = (perception.get("market_regime") or {}).get("regime")
            strength = safe_float((perception.get("market_regime") or {}).get("strength"), 0.0)
            # 账户状态机门控：DECLINE/EMERGENCY 下收敛不进攻；RECOVERY 下默认不进攻，
            # 但启用 recovery_offense 且恢复期乘数达标时允许渐进进攻（谨慎恢复）。
            # mode 未知时放行（向后兼容）。
            offense_allowed = mode not in ("decline", "recovery", "emergency")
            recovery_multiplier: Optional[float] = None
            if mode == "recovery" and self._recovery_offense_enabled:
                _pm = safe_float(equity_status.get("position_multiplier"), 0.0)
                if _pm >= self._recovery_offense_min_multiplier:
                    offense_allowed = True
                    recovery_multiplier = _pm
            # 趋势市进攻：趋势确认（趋势市 + 足够强度）且不极端（≤ max_regime_strength，
            # 避免趋势末端追涨杀跌）
            trend_offense = (
                regime in ("trend_bullish", "trend_bearish")
                and strength >= self._offensive_min_regime_strength
                and strength <= self._offensive_max_regime_strength
            )
            # 震荡市进攻：range_bound 下为震荡策略（grid 高抛低吸）开单，强度要求略低
            range_offense = (
                self._offensive_range_bound_enabled
                and regime == "range_bound"
                and strength >= self._offensive_range_bound_min_strength
            )
            # 信号质量门槛强化：进攻开单还需市场状态置信度达标（够确定才开单）
            regime_conf = safe_float(
                (perception.get("market_regime") or {}).get("confidence"), 0.0)
            confident = regime_conf >= self._offensive_min_regime_confidence
            # 开单时机校准：趋势确认周期——市场状态需持续 ≥ trend_confirmation_cycles 个周期
            confirmed_streak = (
                (self._trend_confirmed_streak + 1)
                if regime == self._trend_confirmed_regime else 1
            )
            regime_confirmed = (
                self._trend_confirmation_cycles <= 0
                or confirmed_streak >= self._trend_confirmation_cycles
            )
            if offense_allowed and (trend_offense or range_offense) and confident and regime_confirmed:
                eq = safe_float(perception.get("equity"), 0.0)
                if eq > 0 and self._risk_appetite() >= self._adaptive_risk_min_appetite:
                    gs = self._update_goal_planning_state(eq, perception.get("total_capital"))
                    if gs["drawdown"] <= self._offensive_max_drawdown_pct:
                        contrib = _coerce_dict(perception.get("contribution"))
                        strat_map = _coerce_dict(contrib.get("strategies"))
                        # 信号共振过滤：进攻候选不仅健康度 A/B，还需趋势不衰退 + 累计盈利为正
                        healthy = []
                        for name, c in strat_map.items():
                            if c.get("health_grade") not in ("A", "B"):
                                continue
                            if self._signal_resonance_enabled:
                                if c.get("trend") == "declining":
                                    continue
                                if safe_float(c.get("total_pnl"), 0.0) <= 0:
                                    continue
                            healthy.append(name)
                        # 震荡市只进攻震荡策略（grid/oscillation_harvest），趋势市进攻所有 A/B 策略
                        is_range = bool(range_offense and not trend_offense)
                        if is_range:
                            healthy = [
                                n for n in healthy
                                if n in self._offensive_range_bound_strategies
                            ]
                        # 进攻火力集中：按 health_score 降序取前 N 个，集中优势火力到质量最高的策略，
                        # 避免撒网过宽。震荡市独立火力集中上限（range_bound_max_offensive_strategies）
                        # ——震荡市「快进快出、机会转瞬即逝」更应集中到更少最优策略，趋势市可略分散。
                        max_offensive = (
                            self._offensive_range_bound_max_offensive_strategies if is_range
                            else self._offensive_max_offensive_strategies
                        )
                        healthy = sorted(
                            healthy,
                            key=lambda n: safe_float(
                                (strat_map.get(n) or {}).get("health_score"), 0.0),
                            reverse=True,
                        )[: max_offensive]
                        if healthy:
                            mode_label = "震荡市高抛低吸" if is_range else "趋势确认"
                            # 进攻力度动量缩放：按 regime strength 归一化后映射到 [min_mult, max_mult]
                            raw_boost = (
                                self._offensive_range_bound_boost_step if is_range
                                else self._offensive_boost_step
                            )
                            if self._offensive_momentum_scaling_enabled:
                                min_str = (
                                    self._offensive_range_bound_min_strength if is_range
                                    else self._offensive_min_regime_strength
                                )
                                norm = max(0.0, min(1.0,
                                    (strength - min_str) / max(0.01, 1.0 - min_str)))
                                scaled_boost = raw_boost * (
                                    self._offensive_momentum_min_mult
                                    + (self._offensive_momentum_max_mult
                                       - self._offensive_momentum_min_mult) * norm
                                )
                            else:
                                scaled_boost = raw_boost
                            # 恢复期渐进进攻：RECOVERY 模式按 position_multiplier 缩放进攻力度，
                            # 恢复期乘数越低（刚脱离 EMERGENCY）进攻越保守，随恢复进度逐步放大。
                            if recovery_multiplier is not None:
                                scaled_boost *= recovery_multiplier
                            alerts.append({
                                "level": "info",
                                "type": "offensive_opportunity",
                                "strategies": healthy,
                                "mode": "range_bound" if is_range else "trend",
                                "boost_step": scaled_boost,
                                "message": (
                                    f"{mode_label}({regime},强度{strength:.0%}) + 账户健康，"
                                    f"进攻性加仓 A/B 策略 {healthy}"
                                ),
                            })

        # 决策执行结果反馈：上一周期动作被 fail-closed 拒绝（如 Kill Switch 熔断）→ 收敛告警
        exec_result = self._last_execution_result
        if exec_result and safe_int(exec_result.get("rejected"), 0) > 0:
            alerts.append({
                "level": "warning",
                "type": "execution_rejected",
                "rejected": safe_int(exec_result.get("rejected"), 0),
                "message": (
                    f"上一周期 {safe_int(exec_result.get('rejected'), 0)} 条动作被执行通道拒绝"
                    f"（可能因 Kill Switch 熔断），建议收敛进攻"
                ),
            })

        # 逐币种精细杠杆守卫（symbol_param_guard）：单币种浮亏超阈值 → 该币种单独下调杠杆
        if self._symbol_param_guard_enabled:
            symbol_pnl = perception.get("symbol_pnl") or {}
            for symbol, upl in symbol_pnl.items():
                upl_f = safe_float(upl, 0.0)
                if upl_f < -self._symbol_loss_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "symbol_losing",
                        "symbol": symbol,
                        "upl": upl_f,
                        "message": (
                            f"币种 {symbol} 浮亏 {upl_f:.2f} 超过阈值 "
                            f"{self._symbol_loss_threshold:.2f}，精细下调该币种杠杆"
                        ),
                    })

        # 现货持有守卫（spot_hold_guard）：现货过度分散 / 现货累计盈利达标 → 收敛现货敞口
        if self._spot_hold_guard_enabled:
            spot = perception.get("spot_holdings") or {}
            spot_count = safe_int(spot.get("count"), 0)
            if spot_count > self._spot_max_currencies:
                alerts.append({
                    "level": "warning",
                    "type": "spot_overweight",
                    "spot_count": spot_count,
                    "message": (
                        f"现货持币种类 {spot_count} 超过上限 {self._spot_max_currencies}，"
                        f"收敛现货敞口（现货不占用过多资金）"
                    ),
                })
            # 现货策略累计盈亏达阈值 → 止盈落袋（稳步增长）
            contrib = _coerce_dict(perception.get("contribution"))
            strategies = _coerce_dict(contrib.get("strategies"))
            spot_pnl = sum(
                safe_float(strategies.get(s, {}).get("total_pnl"), 0.0)
                for s in self._spot_strategies
            )
            equity = safe_float(perception.get("equity"), 0.0)
            if equity > 0 and spot_pnl / equity >= self._spot_profit_take_pct:
                alerts.append({
                    "level": "info",
                    "type": "spot_profit",
                    "spot_pnl": spot_pnl,
                    "message": (
                        f"现货策略累计盈亏 {spot_pnl:.2f} 占比达阈值 "
                        f"{self._spot_profit_take_pct:.1%}，止盈落袋锁定稳步增长"
                    ),
                })
            # 现货策略累计亏损达阈值 → 止损回收（现货不占用资金、腾挪到合约），与 spot_profit 对称
            if equity > 0 and self._spot_loss_take_pct > 0 and spot_pnl / equity <= -self._spot_loss_take_pct:
                alerts.append({
                    "level": "warning",
                    "type": "spot_loss",
                    "spot_pnl": spot_pnl,
                    "message": (
                        f"现货策略累计亏损 {spot_pnl:.2f} 占比达阈值 "
                        f"{self._spot_loss_take_pct:.1%}，止损回收资金到合约（现货不占用资金）"
                    ),
                })

        # 每小时盈利效率守卫（hourly_pnl_guard）：活跃时长与交易量足够但每小时持续失血 → 收敛
        if self._hourly_pnl_guard_enabled:
            contrib = _coerce_dict(perception.get("contribution"))
            strategies = _coerce_dict(contrib.get("strategies"))
            for name, s in strategies.items():
                active_h = safe_float(s.get("active_hours"), 0.0)
                trades = safe_int(s.get("total_trades"), 0)
                pph = safe_float(s.get("pnl_per_hour"), 0.0)
                if (active_h >= self._hourly_min_active_hours
                        and trades >= self._hourly_min_trades
                        and pph < -self._hourly_loss_threshold):
                    alerts.append({
                        "level": "warning",
                        "type": "hourly_pnl_negative",
                        "strategy": name,
                        "pnl_per_hour": pph,
                        "active_hours": active_h,
                        "message": (
                            f"策略 {name} 每小时盈利 {pph:.2f} 低于 "
                            f"-{self._hourly_loss_threshold:.2f}"
                            f"（活跃 {active_h:.1f}h / {trades} 笔），时间价值持续流失，收敛资金权重"
                        ),
                    })

        # 交易频率守卫（trade_frequency_guard）：活跃时长与交易量足够但每小时交易笔数过高 → 高频交易收敛
        if self._trade_frequency_enabled:
            contrib = _coerce_dict(perception.get("contribution"))
            strategies = _coerce_dict(contrib.get("strategies"))
            for name, s in strategies.items():
                active_h = safe_float(s.get("active_hours"), 0.0)
                trades = safe_int(s.get("total_trades"), 0)
                if active_h <= 0 or trades < self._trade_frequency_min_trades:
                    continue
                if active_h < self._trade_frequency_min_active_hours:
                    continue
                tph = trades / active_h
                if tph >= self._trade_frequency_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "high_trade_frequency",
                        "strategy": name,
                        "trades_per_hour": tph,
                        "active_hours": active_h,
                        "total_trades": trades,
                        "message": (
                            f"策略 {name} 交易频率 {tph:.2f} 笔/小时 ≥ "
                            f"{self._trade_frequency_threshold:.2f}（{trades} 笔 / {active_h:.1f}h），"
                            f"高频刷单侵蚀利润，收敛资金权重"
                        ),
                    })

        # 组合多空净敞口监控：净敞口占毛敞口比例过高 → 方向性失衡收敛
        if self._net_exposure_enabled:
            ne = _coerce_dict(perception.get("net_exposure"))
            long_exp = safe_float(ne.get("long"), 0.0)
            short_exp = safe_float(ne.get("short"), 0.0)
            gross = long_exp + short_exp
            if gross >= self._net_exposure_min_gross:
                imbalance = abs(long_exp - short_exp) / gross
                if imbalance >= self._net_exposure_imbalance_threshold:
                    alerts.append({
                        "level": "warning",
                        "type": "directional_imbalance",
                        "long_exposure": long_exp,
                        "short_exposure": short_exp,
                        "imbalance_ratio": imbalance,
                        "message": (
                            f"组合净敞口方向性失衡（{'多' if long_exp > short_exp else '空'}头 "
                            f"{max(long_exp, short_exp):.2f} vs 反向 {min(long_exp, short_exp):.2f}，"
                            f"净占比 {imbalance:.0%} ≥ {self._net_exposure_imbalance_threshold:.0%}），收敛敞口"
                        ),
                    })

        # 决策记忆质量评估：近期决策盈利周期占比过低 → 暂停进攻
        # 死锁自愈：持续触发 recovery_after_cycles 周期后，标记 recovery_mode=True，
        # 允许最小试探开单（recovery_boost_scale × boost_step），给系统恢复出口。
        if self._decision_quality_enabled:
            _cur_regime = (perception.get("market_regime") or {}).get("regime")
            score = self._decision_quality_score(_cur_regime)
            if (len(self._decision_memory) >= self._decision_quality_min_samples
                    and score < self._decision_quality_threshold):
                self._decision_quality_lock_count += 1
                in_recovery = (
                    self._decision_quality_lock_count >= self._decision_quality_recovery_cycles
                )
                alerts.append({
                    "level": "warning",
                    "type": "decision_quality_low",
                    "score": score,
                    "lock_count": self._decision_quality_lock_count,
                    "recovery_mode": in_recovery,
                    "message": (
                        f"近期决策质量过低（盈利周期占比 {score:.0%} < "
                        f"{self._decision_quality_threshold:.0%}），"
                        + ("已进入恢复试探" if in_recovery else "暂停进攻")
                    ),
                })
            else:
                # 质量回升 → 重置锁计数
                self._decision_quality_lock_count = 0

        # 组合压力测试：尾部场景下毛敞口压力损失占权益比例超预算 → 收敛敞口
        if self._stress_guard_enabled:
            ne = _coerce_dict(perception.get("net_exposure"))
            gross = safe_float(ne.get("long"), 0.0) + safe_float(ne.get("short"), 0.0)
            equity = safe_float(perception.get("equity"), 0.0)
            if gross > 0 and equity > 0:
                stress_loss = gross * self._stress_scenario_pct
                if stress_loss / equity >= self._stress_loss_budget:
                    alerts.append({
                        "level": "warning",
                        "type": "stress_test_failed",
                        "stress_loss": stress_loss,
                        "gross_exposure": gross,
                        "message": (
                            f"组合压力测试失败：尾部场景（-{self._stress_scenario_pct:.0%}）"
                            f"压力损失 {stress_loss:.2f} 占权益 {stress_loss/equity:.0%} ≥ "
                            f"预算 {self._stress_loss_budget:.0%}，收敛敞口"
                        ),
                    })
                # 严重尾部场景（多样化）：更大幅度、更高预算，分级告警（critical）
                severe_loss = gross * self._stress_severe_scenario_pct
                if severe_loss / equity >= self._stress_severe_loss_budget:
                    alerts.append({
                        "level": "critical",
                        "type": "severe_stress_test_failed",
                        "stress_loss": severe_loss,
                        "gross_exposure": gross,
                        "message": (
                            f"组合严重压力测试失败：严重尾部场景（-{self._stress_severe_scenario_pct:.0%}）"
                            f"压力损失 {severe_loss:.2f} 占权益 {severe_loss/equity:.0%} ≥ "
                            f"预算 {self._stress_severe_loss_budget:.0%}，紧急收敛敞口"
                        ),
                    })

        logger.info(f"[AGI-Diagnose] {len(alerts)} alerts")
        return alerts

    def _effective_profit_take_activation(self, regime: Optional[str]) -> float:
        """计算经过市场状态自适应 + 学习记忆调整后的 profit_take 激活阈值。

        市场状态自适应：趋势市放大阈值（让利润奔跑）、震荡市缩小阈值（积极落袋）。
        学习记忆：健康度趋势改善 → 放大阈值（让利润跑）；恶化 → 缩小阈值（积极落袋）。
        两者连乘，最终钳制在 [0.005, 0.50]，避免阈值失控。
        """
        activation = self._profit_take_activation_pct
        if self._regime_adaptive_enabled and regime:
            r = str(regime)
            if r in ("trend_bullish", "trend_bearish"):
                activation *= self._trend_profit_take_mult
            elif r == "range_bound":
                activation *= self._range_profit_take_mult
        if self._learning_enabled:
            activation *= self._learning_profit_take_adjust()
        return max(0.005, min(0.50, activation))

    def _learning_profit_take_adjust(self) -> float:
        """学习记忆：据最近决策的健康度趋势，返回 profit_take 阈值调整乘数。

        健康度持续改善 → 放宽落袋（>1，让利润跑）；持续恶化 → 收紧落袋（<1，积极落袋）。
        用首尾差值斜率判断趋势；样本不足或平稳时返回 1.0（不调整）。
        """
        if len(self._decision_memory) < 3:
            return 1.0
        healths = [safe_float(m.get("health_score"), 0.0) for m in self._decision_memory]
        n = len(healths)
        span = n - 1
        if span <= 0:
            return 1.0
        slope = (healths[-1] - healths[0]) / span
        # 健康度每周期变化 < 0.5 分视为平稳，不调整
        if abs(slope) < 0.5:
            return 1.0
        norm = max(-1.0, min(1.0, slope / 5.0))
        return 1.0 + self._learning_max_adjust_pct * norm

    def _update_goal_planning_state(self, equity: float,
                                    total_capital: Optional[float]) -> Dict[str, float]:
        """更新目标导向规划运行时状态，返回收益达成度与当前回撤。

        收益基准：优先用初始资本 total_capital（>0 时），否则用首次观察到的权益。
        峰值权益：跨周期单调上升（只在外部资金变动等重建场景由感知层重置，本编排器
        不主动重建，避免把正常波动误判为基准漂移）。
        """
        eq = safe_float(equity, 0.0)
        if self._goal_baseline_equity is None:
            base = safe_float(total_capital, 0.0)
            self._goal_baseline_equity = base if base > 0 else eq
        if self._goal_baseline_equity is None or self._goal_baseline_equity <= 0:
            self._goal_baseline_equity = eq if eq > 0 else 1.0
        self._goal_peak_equity = max(self._goal_peak_equity, eq)

        baseline = self._goal_baseline_equity
        return_pct = (eq - baseline) / baseline if baseline > 0 else 0.0
        drawdown = (self._goal_peak_equity - eq) / self._goal_peak_equity if self._goal_peak_equity > 0 else 0.0
        return {
            "return_pct": max(return_pct, 0.0),
            "drawdown": max(drawdown, 0.0),
            "baseline": baseline,
            "peak": self._goal_peak_equity,
        }

    def _risk_appetite(self) -> float:
        """自适应风险偏好 [0,1]：基于近期权益轨迹斜率 + 回撤深度惩罚。

        权益上升 → 偏好升高（进攻）；权益下跌 → 偏好降低（收敛）。
        样本不足（<2）时返回中性 0.5。斜率归一化：±10% 权益变动映射到 [0,1]。

        回撤感知（drawdown_aware）：斜率只看窗口首尾，会忽略「回撤深度」——
        深度回撤未恢复时即使近期微升，斜率型偏好仍偏高。故叠加乘法惩罚：
        penalty = 1 - drawdown / drawdown_scale（drawdown_scale 处回撤 → 惩罚归零）。
        """
        if not self._adaptive_risk_enabled or len(self._equity_window) < 2:
            return 0.5
        arr = list(self._equity_window)
        start = arr[0]
        if start <= 0:
            return 0.5
        slope = (arr[-1] - start) / start
        appetite = 0.5 + slope / 0.10
        # 回撤感知：从窗口内峰值回撤越深，风险偏好越低（clamp 到 [0,1]）
        if self._adaptive_risk_drawdown_aware:
            peak = max(arr)
            if peak > 0:
                drawdown = (peak - arr[-1]) / peak
                penalty = max(0.0, 1.0 - drawdown / self._adaptive_risk_drawdown_scale)
                appetite *= penalty
        return max(0.0, min(1.0, appetite))

    def _decision_quality_score(self, regime: Optional[str] = None) -> float:
        """评估决策记忆质量 [0,1]：最近决策中「盈利周期」加权占比。

        决策记忆（_decision_memory）记录每周期 total_pnl/health_score/equity/regime；
        这里用「盈利周期占比」度量决策质量——最近决策导致盈利的比例越高，决策质量越好。
        样本不足（< min_samples）时返回 0.5（中性）。

        指数衰减加权：近期周期权重 = alpha^(n-1-i)（i 从旧到新），让质量分数对市场状态
        切换更快响应——旧市场状态下的决策不应等权拉低/抬高当前质量评估。alpha=1.0
        退化为等权（向后兼容）。

        regime 条件化（可选）：传 regime 时按该市场状态分桶过滤样本，仅用同 regime 的
        决策评估质量，避免不同市场状态互相污染。同 regime 样本不足 min_samples 时回退
        混合计算（保证 guard 仍能工作）。
        """
        # regime 条件化分桶：过滤同 regime 样本
        if (self._decision_quality_regime_conditioned and regime is not None):
            bucket = [m for m in self._decision_memory
                      if (m.get("regime") or None) == regime]
            if len(bucket) >= self._decision_quality_min_samples:
                return self._compute_quality_score(bucket)
            # 同 regime 样本不足 → 回退混合计算
        if len(self._decision_memory) < self._decision_quality_min_samples:
            return 0.5
        return self._compute_quality_score(self._decision_memory)

    def _compute_quality_score(self, memory) -> float:
        """从给定决策记忆（list/deque）计算盈利周期加权占比，复用指数衰减加权。

        优先用 cycle_pnl（单周期 closed-PnL 增量）判定该周期决策盈亏——累计 total_pnl
        跨周期不变会导致质量分恒为 0/1。旧持久化条目无 cycle_pnl 字段时回退 total_pnl
        （向后兼容）。

        ignore_zero_cycle_pnl（默认开）：cycle_pnl==0 表示「本周期无平仓」（中性，既非
        盈也非亏），不应计入质量分——震荡市长时间无平仓时若按 p>0 判定会把「无交易」
        误判为「非盈利」拉低质量分。过滤后全无平仓 → 返回 0.5（中性，不做判定）。
        """
        if self._decision_quality_use_cycle_pnl:
            pnl_list = [
                safe_float(m.get("cycle_pnl", m.get("total_pnl", 0.0)), 0.0)
                for m in memory
            ]
            if self._decision_quality_ignore_zero_pnl:
                pnl_list = [p for p in pnl_list if p != 0.0]
                if not pnl_list:
                    return 0.5  # 全无平仓 → 中性，不做盈亏判定
        else:
            pnl_list = [safe_float(m.get("total_pnl"), 0.0) for m in memory]
        alpha = self._learning_memory_decay_alpha
        n = len(pnl_list)
        # 权重从旧到新：alpha^(n-1), alpha^(n-2), ..., alpha^0
        weights = [alpha ** (n - 1 - i) for i in range(n)]
        total_w = sum(weights)
        if total_w <= 0:
            # 退化保护：全零权重时回退等权
            wins = sum(1 for p in pnl_list if p > 0)
            return wins / n
        win_w = sum(w for w, p in zip(weights, pnl_list, strict=True) if p > 0)
        return win_w / total_w

    def _adaptive_cost_threshold(self, base: float) -> float:
        """成本预算自适应：权益健康 → 放宽成本预算（允许更高滑点/资金费）；权益恶化 → 收紧。

        返回调整后的成本比例阈值（[0.05, 0.9]）。与 confidence_adaptation 同构但方向相反：
        置信门槛在恶化时提高，成本预算在恶化时收紧（更严格）。
        """
        if not self._cost_budget_adaptation_enabled:
            return base
        appetite = self._risk_appetite()
        adjusted = base + (appetite - 0.5) * self._cost_budget_adapt_span
        return max(0.05, min(0.9, adjusted))

    # ─────────────────────────────────────────────────────────────
    # 3. 决策 Decide
    # ─────────────────────────────────────────────────────────────

    async def _decide(self, perception: Dict[str, Any],
                      projection: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        try:
            return await self._decide_impl(perception, projection=projection)
        except Exception as exc:
            perception_fields = sorted(perception) if isinstance(perception, dict) else []
            contribution = _coerce_dict(perception.get("contribution")) if isinstance(perception, dict) else {}
            projection_available = bool(
                projection.get("available") if isinstance(projection, dict) else False
            )
            logger.bind(
                cycle=self._cycle_count,
                perception_fields=perception_fields,
                contribution_available=bool(contribution.get("available")),
                projection_available=projection_available,
                exception_type=type(exc).__name__,
            ).exception(
                "[AGI-Decide] unhandled decision failure; returning degraded decision"
            )
            return {
                "allocation_plan": None,
                "reallocation_suggestions": [],
                "market_regime": "unknown",
                "market_regime_subtype": "unknown",
                "market_regime_strength": 0.0,
                "market_regime_confidence": 0.0,
                "strategy_names": [],
                "strategy_metrics": {},
                "rationale": ["决策阶段异常，未生成资金分配计划"],
                "fail_closed": True,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }

    async def _decide_impl(self, perception: Dict[str, Any],
                           projection: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        equity = perception.get("equity")
        total_capital = perception.get("total_capital")
        contrib = _coerce_dict(perception.get("contribution"))

        strategies = list(_coerce_dict(contrib.get("strategies")).keys())
        if not strategies:
            strategies = self._enabled_strategy_names_from_config()

        market_regime = perception.get("market_regime") or {}
        regime_str = market_regime.get("regime")
        allocator_regime = _map_regime(regime_str)

        decision: Dict[str, Any] = {
            "allocation_plan": None,
            "reallocation_suggestions": [],
            "market_regime": str(regime_str or "unknown"),
            "market_regime_subtype": str(market_regime.get("subtype") or "unknown"),
            "market_regime_strength": safe_float(market_regime.get("strength"), 0.0),
            "market_regime_confidence": safe_float(market_regime.get("confidence"), 0.0),
            "strategy_names": list(strategies),
            "strategy_metrics": {},
            "rationale": [],
        }
        rationale: List[str] = decision["rationale"]

        # fail-closed：无法确认权益（None 或 <=0）→ 保守空计划，不调用分配器
        if equity is None or safe_float(equity, 0.0) <= 0:
            logger.warning("[AGI-Decide] equity unavailable/non-positive; fail-closed, no allocation plan")
            rationale.append("权益不可用或非正，fail-closed 中止分配")
            decision["allocation_plan"] = {
                "timestamp": datetime.now().isoformat(),
                "total_capital": safe_float(total_capital, 0.0),
                "total_equity": safe_float(equity, 0.0) if equity is not None else 0.0,
                "warnings": ["equity unavailable or non-positive; allocation aborted (fail-closed)"],
                "strategy_allocations": {},
            }
            return decision

        # 贡献归因 → 资金重分配建议
        if self.contribution_analyzer is not None:
            try:
                sugg = getattr(self.contribution_analyzer, "get_capital_reallocation_suggestions", None)
                if callable(sugg):
                    raw = sugg(snapshot=getattr(self, "_last_snapshot", None))
                    if isinstance(raw, list):
                        decision["reallocation_suggestions"] = [
                            dict(s) for s in raw if isinstance(s, dict)
                        ]
                    logger.info(
                        f"[AGI-Decide] {len(decision['reallocation_suggestions'])} reallocation suggestions"
                    )
                    if decision["reallocation_suggestions"]:
                        rationale.append(
                            f"贡献归因生成 {len(decision['reallocation_suggestions'])} 条资金重分配建议"
                        )
            except Exception as e:
                logger.warning(f"[AGI-Decide] reallocation suggestions failed (degraded): {e}")

        # 动态分配计划
        if self.dynamic_allocator is not None:
            try:
                compute = getattr(self.dynamic_allocator, "compute_allocation_plan", None)
                if callable(compute):
                    metrics = build_strategy_metrics(contrib)
                    # 推算结果注入 strategy_metrics（供 DynamicAllocator 优先级评分参考）
                    if self._pnl_projection_enabled and projection and projection.get("available"):
                        # 全局 bear_case fail-closed 阈值（相对权益的亏损比例）
                        _bear_threshold_abs = -abs(
                            safe_float(self._projection_fail_closed_loss, 0.05)
                        ) * safe_float(equity, 0.0)
                        for _n, _sp in (projection.get("per_strategy") or {}).items():
                            if _n in metrics:
                                metrics[_n]["projected_pnl_per_cycle"] = _sp["corrected_expect"]
                                metrics[_n]["projection_confidence"] = _sp["confidence"]
                                # bear_case 场景 horizon 注入（供 allocator 评估下行风险）
                                _bc_h = safe_float(_sp.get("bear_case_horizon"), 0.0)
                                metrics[_n]["bear_case_horizon"] = _bc_h
                                # per-strategy bear_case fail-closed：悲观场景下预期亏损超过阈值 → 标记
                                metrics[_n]["bear_case_fail_closed"] = bool(
                                    _bc_h < _bear_threshold_abs
                                )
                    current_weights = self._current_weights(strategies)
                    used_margin = self._used_margin_by_strategy()

                    result = compute(
                        total_capital=safe_float(total_capital, safe_float(equity, 0.0)),
                        total_equity=safe_float(equity, 0.0),
                        strategy_names=list(strategies),
                        strategy_metrics=metrics,
                        market_regime=allocator_regime,
                        current_weights=current_weights,
                        used_margin_by_strategy=used_margin,
                        persist_last_plan=False,
                    )
                    if inspect.isawaitable(result):
                        plan = await result
                    else:
                        plan = result

                    decision["allocation_plan"] = self._plan_to_dict(plan)
                    decision["strategy_metrics"] = metrics
                    # strength 入分配：低强度 regime 信号时标注分配建议低置信度（不改权重，仅告警）
                    strength = safe_float(market_regime.get("strength"), 0.0)
                    if strength < self.regime_strength_low_threshold and isinstance(decision["allocation_plan"], dict):
                        warnings = decision["allocation_plan"].setdefault("warnings", [])
                        warnings.append(
                            f"low regime strength {strength:.2f} (< {self.regime_strength_low_threshold:.2f}); "
                            "allocation suggestion is low-confidence"
                        )
                    logger.info(f"[AGI-Decide] allocation plan computed for {len(strategies)} strategies")
                    rationale.append(
                        f"动态分配器为 {len(strategies)} 个策略生成资金配置计划"
                        f"(regime={allocator_regime.value})"
                    )
                else:
                    logger.debug("[AGI-Decide] dynamic_allocator lacks compute_allocation_plan; skipping")
            except Exception as e:
                logger.warning(f"[AGI-Decide] allocation plan failed (degraded): {e}")
                decision["allocation_plan"] = None
        else:
            logger.debug("[AGI-Decide] no dynamic_allocator; skipping")

        return decision

    def _current_weights(self, strategy_names: List[str]) -> Dict[str, float]:
        weights: Dict[str, float] = {}
        try:
            getter = getattr(self.dynamic_allocator, "get_strategy_allocations", None)
            if callable(getter):
                allocs = getter()
                if isinstance(allocs, dict):
                    for name, info in allocs.items():
                        if isinstance(info, dict) and str(name) in strategy_names:
                            weights[str(name)] = safe_float(info.get("target_weight"), 0.0)
        except Exception as e:
            logger.debug(f"[AGI-Decide] current weights query failed: {e}")

        if not weights and self.capital_allocator is not None:
            try:
                gsa = getattr(self.capital_allocator, "get_strategy_allocation", None)
                if callable(gsa):
                    for name in strategy_names:
                        weights[str(name)] = safe_float(gsa(name), 0.0)
            except Exception as e:
                logger.debug(f"[AGI-Decide] capital allocation fallback failed: {e}")
        return weights

    def _used_margin_by_strategy(self) -> Dict[str, float]:
        used: Dict[str, float] = {}
        try:
            getter = getattr(self.dynamic_allocator, "get_strategy_allocations", None)
            if callable(getter):
                allocs = getter()
                if isinstance(allocs, dict):
                    for name, info in allocs.items():
                        if isinstance(info, dict):
                            used[str(name)] = safe_float(info.get("used_margin"), 0.0)
        except Exception as e:
            logger.debug(f"[AGI-Decide] used margin query failed: {e}")
        return used

    def _enabled_strategy_names_from_config(self) -> List[str]:
        strategies = self.config.get("strategies") or {}
        names: List[str] = []
        if isinstance(strategies, dict):
            for name, sec in strategies.items():
                if isinstance(sec, dict) and sec.get("enabled", True):
                    names.append(str(name))
        return names

    @staticmethod
    def _plan_to_dict(plan: Any) -> Optional[Dict[str, Any]]:
        if plan is None:
            return None
        if isinstance(plan, dict):
            return copy.deepcopy(plan)
        to_dict = getattr(plan, "to_dict", None)
        if callable(to_dict):
            try:
                d = to_dict()
                if isinstance(d, dict):
                    return d
            except Exception:
                pass
        return None

    # ─────────────────────────────────────────────────────────────
    # 4. 执行 Act（只产出动作指令，不直接下单）
    # ─────────────────────────────────────────────────────────────

    def _is_slow_cycle(self) -> bool:
        """判断本周期是否为慢尺度（strategic）决策周期。

        未启用时间尺度分层时返回 True（每个周期都执行全部决策，向后兼容）。
        启用后：仅当 cycle_count 为 slow_cycle_interval 的整数倍时执行慢尺度决策。
        """
        if not self._timescale_enabled:
            return True
        return self._cycle_count % self._slow_cycle_interval == 0

    def _act(self, decision: Dict[str, Any], alerts: List[Dict[str, Any]],
             projection: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        actions: List[Dict[str, Any]] = []

        # 慢尺度（strategic）：战略性决策仅在慢周期执行，避免短窗口噪声震荡
        slow = self._is_slow_cycle()
        self._last_timescale = "slow" if slow else "fast"

        plan = decision.get("allocation_plan") or {}
        if slow:
            for a in plan.get("auto_actions") or []:
                actions.append({"type": "allocation_auto_action", "detail": str(a)})
            for r in plan.get("recommendations") or []:
                actions.append({"type": "allocation_recommendation", "detail": str(r)})

            for s in decision.get("reallocation_suggestions") or []:
                actions.append({
                    "type": "reallocate",
                    "strategy": s.get("strategy"),
                    "action": s.get("action"),
                    "target_allocation": s.get("target_allocation"),
                    "reason": s.get("reason"),
                })

        # 快尺度（tactical）：以下动作每个周期都评估，保证风险响应/落袋的及时性
        # 低风险闲置资金自动归集：仅流入 A/B 级核心策略（无需人工确认）
        actions.extend(self._idle_cash_deploy_actions(decision))

        # 账户级收益检测自动平仓：profit_take_signal → profit_take_close 动作（低风险自动执行）
        for alert in alerts or []:
            if alert.get("type") == "profit_take_signal":
                actions.append({
                    "type": "profit_take_close",
                    "close_ratio": safe_float(alert.get("close_ratio"), 0.0),
                    "unrealized_pnl_pct": safe_float(alert.get("unrealized_pnl_pct"), 0.0),
                    "reason": alert.get("message"),
                })

        # 主动风控响应：诊断到风险策略时主动减配（而非仅告警）
        actions.extend(self._risk_reduce_actions(alerts))

        # 资金利用率过高收敛：敞口过高 → 主动降低保证金占用最多的策略权重（对称闭环）
        actions.extend(self._utilization_reduce_actions(decision, alerts))

        # 资金利用率趋势外推：杠杆/敞口持续放大 → 提前收敛（与阈值型 utilization_guard 对称）
        actions.extend(self._utilization_trend_actions(decision, alerts))

        # 利润回吐保护：浮盈从峰值回吐超阈值 → 减仓锁利（收益侧闭环）
        actions.extend(self._give_back_actions(alerts))

        # 策略级浮亏止损：单策略浮亏超阈值 → 止损减仓（亏损侧闭环，与收益侧对称）
        actions.extend(self._unrealized_loss_actions(alerts))

        # 策略级已实现亏损：已平仓交易累计永久损失超阈值 → 收敛资金权重
        actions.extend(self._realized_loss_actions(alerts))

        # 策略级浮亏加深趋势：浮亏连续加深 → 提前止损（亏损侧趋势，与收益侧 give_back 对称）
        actions.extend(self._unrealized_loss_trend_actions(decision, alerts))

        # 策略级手续费率守卫：单策略手续费侵蚀过高 → 降低过度交易策略权重（消除频繁交易）
        actions.extend(self._strategy_fee_ratio_actions(decision, alerts))

        # 策略级资金费率守卫：单策略资金费侵蚀过高 → 收敛持仓过久策略（持仓时间成本）
        actions.extend(self._funding_cost_actions(decision, alerts))

        # 策略级执行质量成本守卫：单策略滑点/点差侵蚀过高 → 收敛执行质量差策略（执行成本）
        actions.extend(self._execution_cost_actions(decision, alerts))

        # 策略级手续费率趋势：手续费率连续上升 → 提前降频收敛（消除频繁交易的事前预警）
        actions.extend(self._strategy_fee_ratio_trend_actions(decision, alerts))

        # 策略级资金费率趋势：资金费率连续上升 → 提前收敛持仓（持仓时间成本的事前预警）
        actions.extend(self._funding_cost_trend_actions(decision, alerts))

        # 策略级执行成本趋势：执行成本率连续上升 → 提前收敛敞口（执行质量的事前预警）
        actions.extend(self._execution_cost_trend_actions(decision, alerts))

        # 浮盈占比过高：单策略账面盈利靠浮盈支撑 → 减仓锁定浮盈（收益侧闭环，与亏损侧对称）
        actions.extend(self._unrealized_profit_ratio_actions(decision, alerts))

        # 浮盈占比趋势：浮盈占比连续上升 → 提前锁定浮盈（收益侧趋势，与浮亏加深对称）
        actions.extend(self._unrealized_profit_ratio_trend_actions(decision, alerts))

        # 连续下跌收敛：权益持续失血 → 主动降杠杆（momentum_guard 的镜像，消除杀跌）
        actions.extend(self._downside_momentum_actions(decision, alerts))

        # 回撤加速收敛：回撤连续加深 → 主动降杠杆（中周期轨迹先行指标的主动收敛侧）
        actions.extend(self._drawdown_acceleration_actions(decision, alerts))

        # 交易成本侵蚀主动降仓：手续费占毛利比例过高 → 主动降低交易最活跃策略权重（消除频繁交易）
        actions.extend(self._high_trading_cost_actions(decision, alerts))

        # 权益状态机主动收敛：emergency/decline → 主动降杠杆（账户级权益状态收敛）
        actions.extend(self._equity_mode_actions(decision, alerts))

        # 策略参数自适应：健康度 F / 趋势恶化 → 降低杠杆（参数级风险收敛）
        actions.extend(self._param_adjust_actions(alerts, decision))

        # 策略参数自适应恢复：健康度恢复 A/B → 恢复杠杆（降杠杆的对称闭环，防永久低杠杆）
        actions.extend(self._param_restore_actions(alerts))

        # 逐币种精细杠杆守卫：单币种浮亏超阈值 → 该币种单独下调杠杆（symbol-level 参数收敛）
        actions.extend(self._symbol_param_actions(alerts))

        # 现货持有守卫：现货过度分散 / 累计盈利达标 → 收敛现货敞口（现货不占用过多资金）
        actions.extend(self._spot_hold_actions(alerts))

        # 每小时盈利效率守卫：每小时持续失血 → 收敛资金权重（消除时间价值流失的低效策略）
        actions.extend(self._hourly_pnl_actions(alerts))

        # 交易频率守卫：每小时交易笔数过高 → 收敛资金权重（消除高频刷单侵蚀利润）
        actions.extend(self._trade_frequency_actions(alerts))

        # 风险调整后贡献守卫：盈利但风险收益比失衡（小赚大扛）→ 提前降杠杆
        actions.extend(self._risk_adjusted_contribution_actions(alerts))

        # 策略休眠资金回收守卫：曾活跃但长时间未交易 → 回收闲置资金到活跃策略
        actions.extend(self._strategy_staleness_actions(alerts))

        # 健康度骤降守卫：健康度单周期骤降（突然恶化）→ 提前降杠杆
        actions.extend(self._health_crash_actions(alerts))

        # 组合多空净敞口监控：方向性失衡 → 收敛最高权重策略敞口
        actions.extend(self._net_exposure_actions(decision, alerts))

        # 账户级杠杆守卫：杠杆逼近上限 → 收敛最高权重策略敞口
        actions.extend(self._account_leverage_actions(decision, alerts))

        # 挂单保证金守卫：隐性挂单敞口过高 → 收敛最高权重策略敞口
        actions.extend(self._pending_margin_actions(decision, alerts))

        # 组合压力测试：尾部压力测试失败 → 收敛最高权重策略敞口
        actions.extend(self._stress_actions(decision, alerts))

        # 组合波动率预算守卫：单笔盈亏标准差超预算 → 提前降杠杆
        actions.extend(self._volatility_budget_actions(alerts))

        # 健康度趋势外推：连续下降 → 提前降杠杆（事前风控，独立于 param_adaptation 开关）
        actions.extend(self._health_trend_actions(alerts))

        # 胜率趋势外推：连续下降 → 提前降杠杆（事前风控，与 health_trend 互补）
        actions.extend(self._win_rate_trend_actions(alerts))

        # 胜率绝对值：胜率低于阈值 → 提前降杠杆（阈值型，长期低胜率硬信号）
        actions.extend(self._win_rate_guard_actions(alerts))

        # 夏普趋势外推：连续下降 → 提前降杠杆（事前风控，风险调整后收益恶化先行指标）
        actions.extend(self._sharpe_trend_actions(alerts))

        # 夏普比率绝对阈值：负夏普（风险调整后负收益）→ 提前降杠杆（事前风控，纯亏损硬信号）
        actions.extend(self._sharpe_ratio_actions(alerts))

        # 连续亏损收敛：consecutive_losses 达阈值 → 提前降杠杆（事前风控，硬冻结前收敛）
        actions.extend(self._consecutive_losses_actions(alerts))

        # 止损频率收敛：止损率过高 → 提前降杠杆（事前风控，入场时机差收敛）
        actions.extend(self._stop_loss_frequency_actions(alerts))

        # 止盈止损比收敛：止盈相对止损过少 → 提前降杠杆（离场质量差收敛）
        actions.extend(self._take_profit_ratio_actions(alerts))

        # 盈亏比收敛：平均盈利/平均亏损过低 → 提前降杠杆（单笔盈亏结构差收敛）
        actions.extend(self._win_loss_ratio_actions(alerts))

        # 止盈止损盈亏金额比收敛：止盈累计相对止损累计过少 → 提前降杠杆（仓位管理失衡收敛）
        actions.extend(self._take_profit_pnl_ratio_actions(alerts))

        # 多空方向失衡收敛：单方向持续逆势亏损 → 提前降杠杆（方向判断质量收敛）
        actions.extend(self._long_short_imbalance_actions(alerts))

        # 策略方向偏好失衡收敛：单边笔数占比过高 → 提前降杠杆（方向单一缺乏对冲）
        actions.extend(self._direction_bias_actions(alerts))

        # 盈亏比趋势外推：连续下降 → 提前降杠杆（事前风控，赢亏幅度比恶化先行指标）
        actions.extend(self._profit_factor_trend_actions(alerts))

        # 盈亏比绝对阈值：总亏损超过总盈利（已实现净亏损）→ 提前降杠杆（事前风控，纯亏损硬信号）
        actions.extend(self._profit_factor_guard_actions(alerts))

        # 最大回撤趋势外推：连续加深 → 提前降杠杆（事前风控，风险深度扩大先行指标）
        actions.extend(self._max_drawdown_trend_actions(alerts))

        # 最大回撤绝对值：回撤超阈值 → 提前降杠杆（阈值型，深度回撤硬信号）
        actions.extend(self._max_drawdown_guard_actions(alerts))

        # 回撤持续时间趋势外推：连续拉长 → 提前降杠杆（事前风控，恢复能力下降先行指标）
        actions.extend(self._drawdown_duration_trend_actions(alerts))

        # 回撤持续时间绝对阈值：最长回撤持续过久（恢复能力差）→ 提前降杠杆（事前风控，资金时间价值差）
        actions.extend(self._drawdown_duration_guard_actions(alerts))

        # 资本回报率趋势外推：连续下降 → 提前降杠杆（事前风控，资本配置效率恶化先行指标）
        actions.extend(self._capital_return_trend_actions(alerts))

        # 资本回报率阈值：单位资本产出低于绝对阈值 → 降杠杆收敛（阈值层，与趋势层互补）
        actions.extend(self._capital_return_actions(alerts))

        # 波动率趋势外推：连续上升 → 提前降杠杆（事前风控，绝对不确定性放大先行指标）
        actions.extend(self._volatility_trend_actions(alerts))

        # PnL 动量趋势外推：连续下降 → 提前降杠杆（事前风控，盈利动能衰竭先行指标）
        actions.extend(self._pnl_momentum_trend_actions(alerts))

        # PnL 动量绝对阈值：近期动能相对中期明显减弱 → 提前降杠杆（事前风控，增长动能衰竭硬信号）
        actions.extend(self._pnl_momentum_guard_actions(alerts))

        # 单笔期望值趋势外推：连续下降 → 提前降杠杆（事前风控，交易质量恶化先行指标）
        actions.extend(self._pnl_per_trade_trend_actions(alerts))

        # 单笔期望值绝对值：低于阈值 → 提前降杠杆（阈值型，每笔无正期望硬信号）
        actions.extend(self._pnl_per_trade_guard_actions(alerts))

        # 边际盈亏趋势外推：连续为负 → 提前降杠杆（事前风控，持续失血先行指标）
        actions.extend(self._delta_pnl_trend_actions(alerts))

        # 单周期亏损急跌：单周期大额急跌（突发大幅亏损）→ 提前降杠杆（事前风控，急跌尖峰硬信号）
        actions.extend(self._delta_pnl_guard_actions(alerts))

        # 多守卫共振收敛：多维度同时衰退 → 升级降杠杆（事前风控，比单守卫更大力度的收敛）
        actions.extend(self._resonance_actions(alerts))

        # 组合级健康度阈值收敛：组合整体健康度跌破健康线 → 收敛最高权重策略（组合级阈值风控）
        actions.extend(self._portfolio_health_actions(decision, alerts))

        # 组合级健康度趋势外推：组合整体健康度连续下降 → 收敛最高权重策略（组合级事前风控）
        actions.extend(self._portfolio_health_trend_actions(decision, alerts))

        # 组合级集中度趋势外推：集中度连续上升 → 收敛最高权重策略（组合级事前风控）
        actions.extend(self._portfolio_concentration_trend_actions(decision, alerts))

        # 组合级相关性趋势外推：相关性连续上升 → 收敛最高权重策略（组合级事前风控）
        actions.extend(self._portfolio_correlation_trend_actions(decision, alerts))

        # 组合级尾部风险趋势外推：尾部风险连续加深 → 收敛最高权重策略（组合级事前风控）
        actions.extend(self._portfolio_tail_risk_trend_actions(decision, alerts))

        # 组合级盈利集中度趋势外推：盈利来源集中度连续上升 → 收敛主导盈利策略（组合级事前风控）
        actions.extend(self._profit_concentration_trend_actions(decision, alerts))

        # 组合级协同度趋势外推：协同度连续下降 → 收敛最高权重策略（组合级事前风控）
        actions.extend(self._synergy_trend_actions(decision, alerts))

        # 组合级资金效率趋势外推：资金效率连续下降 → 收敛最高权重策略（组合级事前风控）
        actions.extend(self._portfolio_efficiency_trend_actions(decision, alerts))

        # 风险自愈闭环：冻结策略观察期/缩量试探 → 自愈事件（通知型，流入决策溯源）
        actions.extend(self._self_heal_actions(alerts))

        # 慢尺度（strategic）：目标规划 / 进攻加仓 / 策略生命周期
        if slow:
            # 目标导向规划：收益达标 → 减仓落袋；接近回撤约束 → 降仓防守
            actions.extend(self._goal_planning_actions(decision, alerts))

            # 组合级分散化响应：集中度过高 → 降低最高权重策略（分散化）
            actions.extend(self._diversification_actions(decision, alerts))

            # 组合级相关性响应：高相关/有效N侵蚀 → 降低最高权重策略（相关性风险）
            actions.extend(self._correlation_response_actions(decision, alerts))

            # 组合级资金效率响应：资金效率偏低 → 降低最高权重策略（资金效率风险）
            actions.extend(self._portfolio_efficiency_actions(decision, alerts))

            # 组合级协同度响应：协同度低于阈值 → 降低最高权重策略（协同劣化风险）
            actions.extend(self._synergy_actions(decision, alerts))

            # 组合级尾部风险响应：组合最深回撤超阈值 → 降尾部暴露最高策略（尾部风险）
            actions.extend(self._tail_risk_actions(decision, alerts))

            # 组合级风险共振响应：多维度组合风险同时恶化 → 更保守全局收敛（组合级共振）
            actions.extend(self._portfolio_resonance_actions(decision, alerts))

            # 自主进攻性资金分配：趋势确认+账户健康 → 进攻加仓 A/B 策略
            actions.extend(self._offensive_allocation_actions(decision, alerts, projection))

            # 自主策略生命周期管理：永久冻结/休眠 → strategy_pause 停开新仓
            actions.extend(self._strategy_lifecycle_actions(alerts))

            # 自主策略恢复：AGI 暂停过的策略健康度改善 → strategy_resume 恢复开仓
            actions.extend(self._strategy_resume_actions(alerts))

        for alert in alerts or []:
            if alert.get("level") in ("critical", "warning"):
                actions.append({
                    "type": "alert_action",
                    "level": alert.get("level"),
                    "alert_type": alert.get("type"),
                    "detail": alert.get("message"),
                })

        # 动作冲突消解与优先级仲裁：风险收敛优先，抑制与降风险动作矛盾的进攻动作
        actions = self._reconcile_actions(actions)

        # 成本意识调仓门控：过滤调仓幅度过小（< min_delta）的 reallocate 动作，避免频繁交易消耗资金
        actions = self._apply_cost_guard(actions, decision)

        # 决策质量仲裁：附 priority + confidence，门控低置信进攻动作，稳定排序
        actions = self._apply_decision_quality(actions, decision)

        # 决策可解释审计：为每条动作指令补 rationale（若未显式标注），
        # 使动作依据可追溯到 reason/detail/message，形成完整审计链。
        for a in actions:
            if not a.get("rationale"):
                a["rationale"] = (
                    a.get("reason") or a.get("detail") or a.get("message") or ""
                )

        logger.info(
            f"[AGI-Act] {len(actions)} action directives generated "
            f"(scale={self._last_timescale}, no direct order execution)"
        )
        return actions

    def _reconcile_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """动作冲突消解与优先级仲裁（风险收敛优先）。

        多个动作生成器（进攻加仓 / 风控减仓 / 停开新仓 / 降杠杆）在同一周期可能对
        同一策略产出相互矛盾的动作。仲裁规则：
          1. 降风险动作（reallocate decrease / param_adjust 降杠杆 / strategy_pause）
             拥有最高优先级；凡被降风险的策略，其进攻动作（reallocate increase /
             idle_cash_deploy）一律被抑制。
          2. 同策略 reallocate 去重：保留目标权重最小（最保守）的一条；当
             increase 与 decrease 并存时 decrease 优先。
          3. 账户级/通知型动作（alert / self_heal / profit_take_close / allocation_*）
             不参与冲突，原样保留。
        被抑制的动作记入 self._last_reconciliation，供报告与决策溯源审计。
        """
        if not self._reconciliation_enabled:
            self._last_reconciliation = None
            return actions

        def _is_risk_reduce(a: Dict[str, Any]) -> bool:
            atype = a.get("type")
            if atype == "reallocate":
                return a.get("action") == "decrease"
            if atype == "param_adjust":
                return safe_float(a.get("value"), 0.0) <= 1.0
            return atype == "strategy_pause"

        def _is_offensive(a: Dict[str, Any]) -> bool:
            atype = a.get("type")
            if atype == "reallocate":
                return a.get("action") == "increase"
            return atype == "idle_cash_deploy"

        # 1. 识别被降风险的策略集合（先全量扫描，确保顺序无关）
        risk_reduced: set = set()
        for a in actions:
            strat = a.get("strategy")
            if strat and _is_risk_reduce(a):
                risk_reduced.add(str(strat))

        # 清除被降风险策略的进攻归因基线：防守性减仓（或进攻加仓被仲裁丢弃）意味着该策略
        # 的「进攻仓位」已被了结，entry_pnl 基线不再有效，清除之避免陈旧基线误判后续进攻。
        # 注意：进攻止盈/止损减仓已在 _offensive_allocation_actions 内清除归因，此处为幂等 no-op；
        # 观察期冷却 _last_offensive_cycle 保留——防守减仓后仍应冷却，避免立即重新进攻振荡。
        for _strat in risk_reduced:
            self._offensive_attribution.pop(_strat, None)
            self._offensive_attribution_cycle.pop(_strat, None)
            # 防守减仓同样「了结」了进攻仓位 → 连续进攻计数归零
            self._offensive_consecutive.pop(_strat, None)
            # 决策震荡抑制：记录本次防守减仓周期，供下一周期进攻加仓前检查（减了又加振荡）
            self._last_defensive_reduce_cycle[_strat] = self._cycle_count

        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        best_realloc: Dict[str, Dict[str, Any]] = {}

        for a in actions:
            atype = a.get("type")
            strat = a.get("strategy")

            # 2. 抑制进攻动作：目标策略已被降风险
            if strat and _is_offensive(a) and str(strat) in risk_reduced:
                dropped.append({**a, "dropped_reason": "risk_reduce_priority"})
                continue

            # 3. 同策略 reallocate 去重（decrease 优先，同方向取更小 target）
            if atype == "reallocate" and strat:
                key = str(strat)
                if key in best_realloc:
                    existing = best_realloc[key]
                    if a.get("action") == "decrease" and existing.get("action") != "decrease":
                        dropped.append({**existing, "dropped_reason": "reallocate_dedup"})
                        best_realloc[key] = a
                    elif a.get("action") == existing.get("action"):
                        ta = safe_float(a.get("target_allocation"), 1.0)
                        te = safe_float(existing.get("target_allocation"), 1.0)
                        if ta < te:
                            dropped.append({**existing, "dropped_reason": "reallocate_dedup"})
                            best_realloc[key] = a
                        else:
                            dropped.append({**a, "dropped_reason": "reallocate_dedup"})
                    else:
                        # increase 与已有 decrease 冲突 → 丢弃 increase
                        dropped.append({**a, "dropped_reason": "reallocate_dedup"})
                    continue
                best_realloc[key] = a
                continue

            kept.append(a)

        # 合并去重后的 reallocate 动作
        kept.extend(best_realloc.values())

        self._last_reconciliation = {
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "dropped": dropped,
        }
        logger.info(
            f"[AGI-Act] reconciliation: kept={len(kept)}, dropped={len(dropped)}"
        )
        return kept

    def _apply_cost_guard(self, actions: List[Dict[str, Any]],
                          decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """成本意识调仓门控：过滤调仓幅度低于 min_delta 的 reallocate 动作。

        - 对每个 reallocate 动作，从 decision.allocation_plan.strategy_allocations 读取
          该策略当前 target_weight；目标权重与当前权重差异 < min_delta 时丢弃。
        - 无法读取当前权重（策略不在 allocations / 结构异常）时放行（向后兼容，不误杀）。
        - 仅门控 reallocate 类型；非调仓动作（profit_take_close/param_adjust 等）不受影响。
        被丢弃动作记入 self._last_cost_guard，供决策溯源审计。
        """
        if not self._cost_guard_enabled:
            self._last_cost_guard = None
            return actions

        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for a in actions:
            if a.get("type") != "reallocate":
                kept.append(a)
                continue
            strategy = a.get("strategy")
            info = allocs.get(strategy) if isinstance(allocs, dict) else None
            current = safe_float(info.get("target_weight"), -1.0) if isinstance(info, dict) else -1.0
            if current < 0.0:
                kept.append(a)  # 无法读取当前权重 → 放行
                continue
            target = safe_float(a.get("target_allocation"), current)
            delta = abs(target - current)
            if delta < self._cost_guard_min_delta:
                dropped.append({
                    "strategy": strategy,
                    "current_weight": current,
                    "target_allocation": target,
                    "delta": round(delta, 4),
                    "reason": a.get("reason") or "",
                })
                continue
            kept.append(a)

        self._last_cost_guard = {
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "dropped": dropped,
        }
        logger.info(
            f"[AGI-Act] cost_guard: kept={len(kept)}, dropped={len(dropped)}"
        )
        return kept

    def _action_confidence(self, action: Dict[str, Any],
                           decision: Dict[str, Any]) -> float:
        """计算动作的量化置信度 [0,1]。

        - 降风险/收敛方向（暂停/降杠杆/落袋/减仓）：方向安全，天然高置信。
        - 通知型：中高置信（无资金风险）。
        - 进攻动作（reallocate increase / idle_cash_deploy）：需证据支撑，置信度
          = 0.3 基础 + 0.3 市场强度 + 0.2 盈亏因子 + 0.2 Sharpe 代理，
          证据不足（弱趋势/无正收益）时置信度走低，供 _apply_decision_quality 门控。
        """
        atype = action.get("type")
        if atype in ("strategy_pause", "param_adjust", "profit_take_close"):
            return 0.95
        if atype == "reallocate" and action.get("action") == "decrease":
            return 0.9
        if atype in ("self_heal", "alert_action",
                     "allocation_auto_action", "allocation_recommendation"):
            return 0.8

        # 进攻动作：reallocate increase / idle_cash_deploy
        strength = safe_float(decision.get("market_regime_strength"), 0.0)
        name = action.get("strategy")
        metrics = (decision.get("strategy_metrics") or {}).get(name) or {}
        pf = safe_float(metrics.get("profit_factor"), 1.0)
        sharpe = safe_float(metrics.get("sharpe_ratio"), 0.0)
        conf = (
            0.3
            + 0.3 * max(0.0, min(1.0, strength))
            + 0.2 * max(0.0, min(1.0, pf - 1.0))
            + 0.2 * max(0.0, min(1.0, sharpe))
        )
        return max(0.0, min(1.0, conf))

    def _apply_decision_quality(self, actions: List[Dict[str, Any]],
                                decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """决策质量仲裁：为动作附 priority + confidence，门控低置信进攻动作，稳定排序。

        - 进攻动作（reallocate increase / idle_cash_deploy）置信度低于 min_confidence
          时被 fail-closed 收敛（丢弃），绝不低置信进攻。
        - 降风险/通知型动作不受门控（方向安全，天然保留）。
        - 最终按 priority 降序稳定排序，风险收敛动作先执行。
        被丢弃动作记入 self._last_confidence_gate，供决策溯源审计。
        """
        # 决策置信度自适应：权益恶化（风险偏好低）→ 提高置信门槛（更保守）；
        # 权益健康（风险偏好高）→ 保持基础门槛。使「要不要进攻」与账户状态联动。
        effective_min_confidence = self._min_confidence
        if self._confidence_adaptation_enabled:
            appetite = self._risk_appetite()
            effective_min_confidence = self._min_confidence + (
                1.0 - appetite
            ) * self._confidence_adapt_span
            effective_min_confidence = max(0.0, min(1.0, effective_min_confidence))

        kept: List[Dict[str, Any]] = []
        dropped: List[Dict[str, Any]] = []
        for a in actions:
            atype = a.get("type")
            a["priority"] = int(_ACTION_PRIORITY.get(atype, 0))
            a["confidence"] = round(self._action_confidence(a, decision), 3)

            offensive = (
                (atype == "reallocate" and a.get("action") == "increase")
                or atype == "idle_cash_deploy"
            )
            # 门控仅在与冲突消解同开关（action_reconciliation.enabled）时生效，
            # 避免默认配置下误杀低置信进攻动作。
            if (
                self._reconciliation_enabled
                and offensive
                and a["confidence"] < effective_min_confidence
            ):
                dropped.append({**a, "dropped_reason": "low_confidence"})
                continue
            kept.append(a)

        # 稳定排序：priority 降序（同级保持原相对顺序）
        kept.sort(key=lambda a: -int(a.get("priority", 0)))

        self._last_confidence_gate = {
            "kept_count": len(kept),
            "dropped_count": len(dropped),
            "min_confidence": round(effective_min_confidence, 3),
            "dropped": dropped,
        } if self._reconciliation_enabled else None
        if dropped:
            logger.info(
                f"[AGI-Act] confidence gate: dropped={len(dropped)} low-confidence actions"
            )
        return kept

    def _risk_reduce_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """主动风控响应：诊断到风险策略时，生成 reallocate 减配动作。

        健康度 F → 减配到 0；连续亏损/累计亏损/趋势恶化 → 减配到 reduce_target。
        动作经 RestrictedExecutionChannel 落地（autonomous 模式），风险早期即收敛敞口，
        不再停留在「仅告警」。同一策略去重，只生成一条减配动作。
        """
        if not self._risk_response_enabled:
            return []
        risk_types = (
            "strategy_health_critical", "consecutive_losses",
            "possible_consecutive_losses", "strategy_declining",
        )
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            atype = alert.get("type")
            strategy = alert.get("strategy")
            if not strategy or atype not in risk_types or strategy in seen:
                continue
            seen.add(strategy)
            target = 0.0 if atype == "strategy_health_critical" else self._risk_response_reduce_target
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": target,
                "reason": f"risk_response:{atype}: {alert.get('message', '')}",
            })
        return actions

    def _give_back_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """利润回吐保护动作：pnl_give_back → reallocate decrease 减仓锁利。

        - 收益侧闭环：策略浮盈从峰值回吐超阈值时，主动降低该策略资金权重，
          把已赚利润部分落袋，避免「赚过又吐回去」。
        - 分级响应：
          * pnl_give_back（基础）→ reallocate decrease 到 reduce_target（默认 10%）
          * pnl_give_back_severe（严重）→ reallocate decrease 到 severe_reduce_target（默认 2%）
          * pnl_give_back_critical（危急）→ reallocate decrease 到 0.0 + strategy_pause 暂停开新仓
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条减仓动作（取最严重级别）。
        """
        if not self._give_back_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        # 收集每个策略的最严重回吐级别：critical > severe > basic
        rank_map = {"pnl_give_back": 1, "pnl_give_back_severe": 2,
                    "pnl_give_back_critical": 3}
        worst_rank: Dict[str, int] = {}
        for alert in alerts or []:
            atype = alert.get("type")
            if atype not in rank_map:
                continue
            strategy = alert.get("strategy")
            if not strategy:
                continue
            rank = rank_map[atype]
            if rank > worst_rank.get(strategy, 0):
                worst_rank[strategy] = rank
        inv_rank = {v: k for k, v in rank_map.items()}
        for strategy, rank in worst_rank.items():
            atype = inv_rank[rank]
            if atype == "pnl_give_back_critical" and self._give_back_critical_enabled:
                # 危急：清仓 + 暂停开新仓
                actions.append({
                    "type": "reallocate",
                    "strategy": strategy,
                    "action": "decrease",
                    "target_allocation": 0.0,
                    "reason": "give_back_guard_critical: 利润危急回吐，清仓止血",
                })
                actions.append({
                    "type": "strategy_pause",
                    "strategy": strategy,
                    "reason": "give_back_guard_critical: 利润危急回吐，暂停开新仓",
                })
                self._paused_strategies.add(str(strategy))
                # 危急回吐已清仓 → 重置峰值，避免旧峰值在策略恢复后立即重新触发回吐
                if self._give_back_peak_reset_on_critical:
                    self._strategy_pnl_peak[str(strategy)] = 0.0
                    self._give_back_peak_last_decay_cycle.pop(str(strategy), None)
            elif atype == "pnl_give_back_severe" and self._give_back_severe_enabled:
                actions.append({
                    "type": "reallocate",
                    "strategy": strategy,
                    "action": "decrease",
                    "target_allocation": self._give_back_severe_reduce_target,
                    "reason": "give_back_guard_severe: 利润严重回吐，大幅减仓",
                })
            else:
                actions.append({
                    "type": "reallocate",
                    "strategy": strategy,
                    "action": "decrease",
                    "target_allocation": self._give_back_reduce_target,
                    "reason": "give_back_guard: 利润回吐，减仓锁利",
                })
        return actions

    def _unrealized_loss_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级浮亏止损动作：strategy_floating_loss → reallocate decrease 止损。

        - 亏损侧闭环：单策略当前浮亏占账户权益比例超阈值时，主动止损减仓，
          限制单策略亏损扩散，对应用户「稳定资金增长」诉求。
        - 与收益侧闭环（profit_take 账户浮盈 / give_back 策略回吐 / ProfitLock 持仓锁微利）
          形成对称：收益侧保住利润，亏损侧限制亏损。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条止损动作。
        """
        if not self._unrealized_loss_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "strategy_floating_loss":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._unrealized_loss_reduce_target,
                "reason": f"unrealized_loss_guard: {alert.get('message', '')}",
            })
        return actions

    def _realized_loss_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级已实现亏损动作：realized_loss → reallocate decrease 收敛资金权重。

        - 已平仓交易累计为负（永久损失）占权益比例超阈值时，降低该策略资金权重，
          收敛「频繁交易累积永久损失」的策略，对应「消除频繁交易消耗资金」诉求。
        - 与 unrealized_loss_actions（浮亏止损，看浮亏）区分：本方法看「已实现亏损」（不可逆）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条收敛动作。
        """
        if not self._realized_loss_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "realized_loss":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._realized_loss_reduce_target,
                "reason": f"realized_loss_guard: {alert.get('message', '')}",
            })
        return actions

    def _unrealized_loss_trend_actions(self, decision: Dict[str, Any],
                                       alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级浮亏加深趋势动作：unrealized_loss_deteriorating → reallocate decrease 提前止损。

        - unrealized_loss_guard（阈值型：浮亏占权益超 loss_threshold 才清仓）的「事前」补充：
          浮亏连续 window 周期加深时提前收敛敞口到 reduce_target（默认0.1，比清仓更温和），
          在浮亏触及硬阈值之前止损，避免单策略亏损扩散。
        - 与 give_back（浮盈从峰值回吐，收益侧趋势）形成对称：give_back 看收益侧回吐，
          本方法看亏损侧加深，共同对应用户「稳定资金增长」诉求。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条止损动作。
        """
        if not self._unrealized_loss_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "unrealized_loss_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._unrealized_loss_trend_reduce_target,
                "reason": f"unrealized_loss_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _strategy_fee_ratio_actions(self, decision: Dict[str, Any],
                                    alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级手续费率动作：high_strategy_fee → reallocate decrease 降低过度交易策略权重。

        - 单策略手续费占毛利比例过高（交易过频、换手过度）时，主动降低该策略权重到
          reduce_target，抑制频繁交易消耗资金（直接对应「消除频繁交易消耗资金」诉求）。
        - 与 cost_awareness（账户级：抑制进攻性加仓）区分：本方法针对单策略收敛敞口。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条降权动作。
        """
        if not self._strategy_fee_ratio_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "high_strategy_fee":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._strategy_fee_ratio_reduce_target,
                "reason": f"strategy_fee_ratio_guard: {alert.get('message', '')}",
            })
        return actions

    def _funding_cost_actions(self, decision: Dict[str, Any],
                              alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级资金费率动作：high_funding_cost → reallocate decrease 收敛持仓过久策略。

        - 单策略资金费占毛利比例过高（持仓过久被 funding 持续侵蚀）时，主动降低该策略权重
          到 reduce_target，收敛持仓敞口、缩短持仓时间（直接对应「稳定资金增长」诉求）。
        - 与 strategy_fee_ratio_guard（手续费 = 交易频率成本，看换手/交易过频）区分：本方法
          针对「持仓时间成本」（funding），即使不交易也亏钱。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条降权动作。
        """
        if not self._funding_cost_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "high_funding_cost":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._funding_cost_reduce_target,
                "reason": f"funding_cost_guard: {alert.get('message', '')}",
            })
        return actions

    def _funding_cost_trend_actions(self, decision: Dict[str, Any],
                                    alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级资金费率趋势动作：funding_cost_rising → reallocate decrease 提前收敛持仓。

        - funding_cost_guard（阈值型：funding_ratio 超 threshold 才降权）的「事前」补充：
          资金费率连续 window 周期上升时提前收敛敞口到 reduce_target（默认0.1），在持仓时间
          成本进一步侵蚀之前收敛持仓、缩短持仓时间（直接对应「稳定资金增长」诉求）。
        - 与 strategy_fee_ratio_trend_actions（手续费率趋势，看换手频率）区分：本方法针对
          持仓时间成本趋势（funding），即使不交易持仓过久也持续侵蚀。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条降权动作。
        """
        if not self._funding_cost_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "funding_cost_rising":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._funding_cost_trend_reduce_target,
                "reason": f"funding_cost_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _execution_cost_trend_actions(self, decision: Dict[str, Any],
                                      alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级执行成本趋势动作：execution_cost_rising → reallocate decrease 提前收敛敞口。

        - execution_cost_guard（阈值型：exec_ratio 超 threshold 才降权）的「事前」补充：
          执行成本率连续 window 周期上升时提前收敛敞口到 reduce_target（默认0.1），在执行
          质量进一步恶化之前收敛、减少市价单追单（直接对应「消除追涨杀跌」诉求）。
        - 与 strategy_fee_ratio_trend_actions（手续费率趋势，看换手频率）、
          _funding_cost_trend_actions（资金费率趋势，看持仓时间）区分：本方法针对执行质量
          成本趋势（滑点/点差），反映流动性、下单时机、市价 vs 限价的执行质量恶化。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条降权动作。
        """
        if not self._execution_cost_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "execution_cost_rising":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._execution_cost_trend_reduce_target,
                "reason": f"execution_cost_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _execution_cost_actions(self, decision: Dict[str, Any],
                                alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级执行质量成本动作：high_execution_cost → reallocate decrease 收敛执行质量差策略。

        - 单策略滑点 + 点差成本占毛利比例过高（下单执行时点差/滑点损失大）时，主动降低该策略
          权重到 reduce_target，收敛敞口、倒逼改善下单执行质量（限价替代市价、避免追单）。
        - 与 strategy_fee_ratio_guard（手续费 = 交易频率成本）、funding_cost_guard（资金费 =
          持仓时间成本）区分：本方法针对「执行质量成本」（滑点/点差），直接对应「消除追涨杀跌」。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条降权动作。
        """
        if not self._execution_cost_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "high_execution_cost":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._execution_cost_reduce_target,
                "reason": f"execution_cost_guard: {alert.get('message', '')}",
            })
        return actions

    def _long_short_imbalance_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级多空方向失衡动作：long_side_losing / short_side_losing → param_adjust 降杠杆。

        - 某方向持续逆势开仓（该方向累计盈亏为负）时，提前降低该策略杠杆（低风险自动执行），
          收敛逆势方向的敞口，直接对应「只在高位开空、低位开多、趋势确认后才开仓」诉求。
        - 与 _consecutive_losses_actions（连续亏损，看连败时序）区分：本方法看「方向」——
          即使亏损不连续，单方向持续亏损也说明该方向开仓判断错误。
        - 同一策略去重，只生成一条降杠杆动作（可能同时触发 long/short 两个告警）。
        """
        if not self._long_short_imbalance_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") not in ("long_side_losing", "short_side_losing"):
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._long_short_imbalance_leverage,
                "reason": f"long_short_imbalance_guard: {alert.get('message', '')}",
            })
        return actions

    def _direction_bias_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略方向偏好失衡动作：direction_bias → param_adjust 降杠杆。

        - 策略几乎只做多或只做空（单边笔数占比过高，方向单一缺乏对冲）时，提前降低该策略
          杠杆（低风险自动执行），收敛方向偏好带来的反转风险，对应「消除追涨杀跌」诉求。
        - 同一策略去重，只生成一条降杠杆动作。
        """
        if not self._direction_bias_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "direction_bias":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._direction_bias_leverage,
                "reason": f"strategy_direction_bias_guard: {alert.get('message', '')}",
            })
        return actions

    def _strategy_fee_ratio_trend_actions(self, decision: Dict[str, Any],
                                          alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略级手续费率趋势动作：strategy_fee_ratio_rising → reallocate decrease 提前降频收敛。

        - strategy_fee_ratio_guard（阈值型：fee_ratio 超 threshold 才降权）的「事前」补充：
          手续费率连续 window 周期上升时提前收敛敞口到 reduce_target（默认0.1），在交易成本
          进一步侵蚀之前降频收敛。
        - 与 cost_awareness（账户级，阈值型抑制进攻）区分：本方法针对单策略收敛敞口（事前趋势）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条降权动作。
        """
        if not self._strategy_fee_ratio_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "strategy_fee_ratio_rising":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._strategy_fee_ratio_trend_reduce_target,
                "reason": f"strategy_fee_ratio_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _unrealized_profit_ratio_actions(self, decision: Dict[str, Any],
                                         alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """浮盈占比过高动作：unrealized_profit_concentration → reallocate decrease 锁定浮盈。

        - 收益侧闭环的「盈利质量」维度：单策略账面盈利主要靠未兑现浮盈支撑时，主动减仓
          锁定浮盈，把脆弱的账面利润部分落袋（对应用户「稳定资金增长」诉求）。
        - 与 give_back（浮盈已回吐，事后减仓锁利）区分：本方法在回吐发生之前减仓（事前），
          与 profit_take（账户级浮盈落袋）区分：本方法是策略级（单策略浮盈占比过高）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条减仓动作。
        """
        if not self._unrealized_profit_ratio_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "unrealized_profit_concentration":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._unrealized_profit_ratio_reduce_target,
                "reason": f"unrealized_profit_ratio_guard: {alert.get('message', '')}",
            })
        return actions

    def _unrealized_profit_ratio_trend_actions(self, decision: Dict[str, Any],
                                               alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """浮盈占比趋势动作：unrealized_profit_ratio_rising → reallocate decrease 锁定浮盈。

        - unrealized_profit_ratio_guard（阈值型：浮盈占比超 ratio_threshold 才减仓）的「事前」
          补充：浮盈占比连续 window 周期上升时提前收敛敞口到 reduce_target（默认0.1），在盈利
          质量进一步劣化之前锁定浮盈。
        - 与 unrealized_loss_trend_actions（浮亏连续加深，亏损侧趋势）形成对称：浮亏侧提前止损，
          本方法看「盈利变虚」提前锁定浮盈，共同对应用户「稳定资金增长」诉求。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        - 同一策略去重，只生成一条减仓动作。
        """
        if not self._unrealized_profit_ratio_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "unrealized_profit_ratio_rising":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._unrealized_profit_ratio_trend_reduce_target,
                "reason": f"unrealized_profit_ratio_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _downside_momentum_actions(self, decision: Dict[str, Any],
                                   alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """连续下跌收敛动作：market_panicking → reallocate decrease 降杠杆。

        - momentum_guard 的镜像：consecutive_up（连续上涨）抑制追涨，本方法响应
          consecutive_down（连续下跌）触发的 market_panicking 告警，主动降低保证金占用
          最多（回退到 target_weight 最高）策略的权重，权益持续失血时收敛敞口（不抄底）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._downside_momentum_enabled:
            return []
        if not any(a.get("type") == "market_panicking" for a in (alerts or [])):
            return []

        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}

        target_name: Optional[str] = None
        used_margin = self._used_margin_by_strategy()
        if used_margin:
            target_name = max(
                used_margin,
                key=lambda n: safe_float(used_margin.get(n), 0.0),
            )
        elif isinstance(allocs, dict) and allocs:
            target_name = max(
                allocs,
                key=lambda n: safe_float(
                    (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                    0.0,
                ),
            )
        if not target_name:
            return []

        current_weight = safe_float(
            (allocs.get(target_name) or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(target_name), dict) else 0.0
        if current_weight <= self._downside_momentum_reduce_target:
            return []

        return [{
            "type": "reallocate",
            "strategy": target_name,
            "action": "decrease",
            "target_allocation": self._downside_momentum_reduce_target,
            "reason": (
                f"downside_momentum_guard: 连续下跌收敛，降低 {target_name} 权重 "
                f"{current_weight:.0%} → {self._downside_momentum_reduce_target:.0%}"
            ),
        }]

    def _drawdown_acceleration_actions(self, decision: Dict[str, Any],
                                       alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """回撤加速收敛动作：drawdown_accelerating → reallocate decrease 降杠杆。

        - drawdown_acceleration_guard 此前仅抑制进攻性加仓（被动门控），本方法补足其主动
          收敛侧（与 _downside_momentum_actions 对称）：回撤连续加深（中周期轨迹先行指标）
          时，主动降低保证金占用最多（回退到 target_weight 最高）策略的权重，在回撤趋势
          恶化时收敛敞口，而非仅等待不再加仓。
        - 与 _downside_momentum_actions（market_panicking，连续下跌周期绝对值）区分：
          本方法看回撤深度时序加速。二者均为账户级风险主动收敛。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._drawdown_accel_enabled:
            return []
        if not any(a.get("type") == "drawdown_accelerating" for a in (alerts or [])):
            return []

        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}

        target_name: Optional[str] = None
        used_margin = self._used_margin_by_strategy()
        if used_margin:
            target_name = max(
                used_margin,
                key=lambda n: safe_float(used_margin.get(n), 0.0),
            )
        elif isinstance(allocs, dict) and allocs:
            target_name = max(
                allocs,
                key=lambda n: safe_float(
                    (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                    0.0,
                ),
            )
        if not target_name:
            return []

        current_weight = safe_float(
            (allocs.get(target_name) or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(target_name), dict) else 0.0
        if current_weight <= self._drawdown_accel_reduce_target:
            return []

        return [{
            "type": "reallocate",
            "strategy": target_name,
            "action": "decrease",
            "target_allocation": self._drawdown_accel_reduce_target,
            "reason": (
                f"drawdown_acceleration_guard: 回撤连续加深，降低 {target_name} 权重 "
                f"{current_weight:.0%} → {self._drawdown_accel_reduce_target:.0%}"
            ),
        }]

    def _high_trading_cost_actions(self, decision: Dict[str, Any],
                                   alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """交易成本侵蚀主动降仓：high_trading_cost → reallocate decrease 降杠杆。

        - cost_awareness 此前仅抑制进攻性加仓（被动门控 `return []`），本方法补足其主动
          收敛侧（与 _drawdown_acceleration_actions / _utilization_reduce_actions 对称）：
          账户手续费占毛利比例过高（交易过频）时，主动降低保证金占用最多（回退到 target_weight
          最高）策略的权重，收敛换手、降低交易频率，直接对应用户「消除频繁交易消耗资金」诉求。
        - 降仓对象：优先 _used_margin_by_strategy() 中占用保证金最多的策略（交易最活跃、手续费
          贡献最高的代理）；无法获取时回退到 allocation_plan 中 target_weight 最高的策略。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._cost_awareness_enabled:
            return []
        if not any(a.get("type") == "high_trading_cost" for a in (alerts or [])):
            return []

        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}

        target_name: Optional[str] = None
        used_margin = self._used_margin_by_strategy()
        if used_margin:
            target_name = max(
                used_margin,
                key=lambda n: safe_float(used_margin.get(n), 0.0),
            )
        elif isinstance(allocs, dict) and allocs:
            target_name = max(
                allocs,
                key=lambda n: safe_float(
                    (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                    0.0,
                ),
            )
        if not target_name:
            return []

        current_weight = safe_float(
            (allocs.get(target_name) or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(target_name), dict) else 0.0
        if current_weight <= self._cost_awareness_reduce_target:
            return []

        return [{
            "type": "reallocate",
            "strategy": target_name,
            "action": "decrease",
            "target_allocation": self._cost_awareness_reduce_target,
            "reason": (
                f"cost_awareness: 手续费侵蚀过高（交易过频），降低 {target_name} 权重 "
                f"{current_weight:.0%} → {self._cost_awareness_reduce_target:.0%}"
            ),
        }]

    def _equity_mode_actions(self, decision: Dict[str, Any],
                             alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """权益状态机主动收敛：equity_emergency/equity_decline → reallocate decrease 降杠杆。

        - equity_emergency（EMERGENCY 紧急）/equity_decline（DECLINE 衰退）告警此前仅由
          EquityMonitor 生成（emergency 冻结新仓、decline 仅标 severity），orchestrator 侧
          只告警不动作。本方法补足主动收敛侧（与 _drawdown_acceleration_actions /
          _utilization_reduce_actions / _high_trading_cost_actions 同构）：权益状态进入
          紧急/衰退时，主动降低保证金占用最多（回退到 target_weight 最高）策略的权重，收敛敞口，
          而非仅靠「不再开新仓」被动等待。
        - 力度分层：emergency → emergency_reduce_target（默认0，全面清空）；decline →
          decline_reduce_target（默认0.1，谨慎收敛）。与 drawdown_acceleration（回撤深度加速）、
          downside_momentum（连续下跌持续性）区分：本方法看「权益状态机的绝对状态」。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._equity_mode_enabled:
            return []
        emergency = any(a.get("type") == "equity_emergency" for a in (alerts or []))
        decline = any(a.get("type") == "equity_decline" for a in (alerts or []))
        if not emergency and not decline:
            return []
        target = (
            self._equity_mode_emergency_reduce_target if emergency
            else self._equity_mode_decline_reduce_target
        )

        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}

        target_name: Optional[str] = None
        used_margin = self._used_margin_by_strategy()
        if used_margin:
            target_name = max(
                used_margin,
                key=lambda n: safe_float(used_margin.get(n), 0.0),
            )
        elif isinstance(allocs, dict) and allocs:
            target_name = max(
                allocs,
                key=lambda n: safe_float(
                    (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                    0.0,
                ),
            )
        if not target_name:
            return []

        current_weight = safe_float(
            (allocs.get(target_name) or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(target_name), dict) else 0.0
        if current_weight <= target:
            return []

        state_label = "EMERGENCY 紧急" if emergency else "DECLINE 衰退"
        return [{
            "type": "reallocate",
            "strategy": target_name,
            "action": "decrease",
            "target_allocation": target,
            "reason": (
                f"equity_mode_guard: {state_label}，降低 {target_name} 权重 "
                f"{current_weight:.0%} → {target:.0%}"
            ),
        }]

    def _utilization_reduce_actions(self, decision: Dict[str, Any],
                                    alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """资金利用率过高主动降仓：high_capital_utilization → reallocate decrease。

        - 补足资金利用率对称闭环的「过高→收敛」动作侧：此前 high_capital_utilization
          仅抑制进攻性加仓（被动门控），本方法主动降低保证金占用最多的策略权重，
          把敞口降下来，缓解强平风险。
        - 降仓对象：优先 _used_margin_by_strategy() 中占用保证金最多的策略（精准降杠杆）；
          无法获取保证金占用时回退到 allocation_plan 中 target_weight 最高的策略。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        与 low_capital_utilization→idle_cash_deploy（过低→归集）形成对称闭环。
        """
        if not self._utilization_guard_enabled:
            return []
        triggered = any(
            alert.get("type") == "high_capital_utilization"
            for alert in (alerts or [])
        )
        if not triggered:
            return []

        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}

        # 降仓对象：优先保证金占用最多的策略，回退到权重最高的策略
        target_name: Optional[str] = None
        used_margin = self._used_margin_by_strategy()
        if used_margin:
            target_name = max(
                used_margin,
                key=lambda n: safe_float(used_margin.get(n), 0.0),
            )
        elif isinstance(allocs, dict) and allocs:
            target_name = max(
                allocs,
                key=lambda n: safe_float(
                    (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                    0.0,
                ),
            )
        if not target_name:
            return []

        current_weight = safe_float(
            (allocs.get(target_name) or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(target_name), dict) else 0.0
        # 已足够低则无需降仓（避免无谓调仓）
        if current_weight <= self._utilization_reduce_target:
            return []

        return [{
            "type": "reallocate",
            "strategy": target_name,
            "action": "decrease",
            "target_allocation": self._utilization_reduce_target,
            "reason": (
                f"utilization_guard: 资金利用率过高，降低 {target_name} 权重 "
                f"{current_weight:.0%} → {self._utilization_reduce_target:.0%}"
            ),
        }]

    def _utilization_trend_actions(self, decision: Dict[str, Any],
                                   alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """资金利用率趋势响应：utilization_rising → 收敛保证金占用最多的策略。

        - utilization_rising 告警说明资金利用率（杠杆/敞口）连续 window 周期上升，
          即使尚未触及 high_utilization_threshold，也属「杠杆持续放大」的先行信号，
          需主动收敛敞口防强平。
        - 降仓对象：优先 _used_margin_by_strategy() 中占用保证金最多的策略（精准降杠杆）；
          无法获取保证金占用时回退到 allocation_plan 中 target_weight 最高的策略。
        - 与 _utilization_reduce_actions（阈值型，事后）对称：本方法对应趋势型（事前）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._utilization_trend_enabled:
            return []
        triggered = any(
            alert.get("type") == "utilization_rising"
            for alert in (alerts or [])
        )
        if not triggered:
            return []

        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}

        # 降仓对象：优先保证金占用最多的策略，回退到权重最高的策略
        target_name: Optional[str] = None
        used_margin = self._used_margin_by_strategy()
        if used_margin:
            target_name = max(
                used_margin,
                key=lambda n: safe_float(used_margin.get(n), 0.0),
            )
        elif isinstance(allocs, dict) and allocs:
            target_name = max(
                allocs,
                key=lambda n: safe_float(
                    (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                    0.0,
                ),
            )
        if not target_name:
            return []

        current_weight = safe_float(
            (allocs.get(target_name) or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(target_name), dict) else 0.0
        # 已足够低则无需降仓（避免无谓调仓）
        if current_weight <= self._utilization_trend_reduce_target:
            return []

        return [{
            "type": "reallocate",
            "strategy": target_name,
            "action": "decrease",
            "target_allocation": self._utilization_trend_reduce_target,
            "reason": (
                f"utilization_trend: 资金利用率持续上升，降低 {target_name} 权重 "
                f"{current_weight:.0%} → {self._utilization_trend_reduce_target:.0%}"
            ),
        }]

    def _diversification_actions(self, decision: Dict[str, Any],
                                 alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级分散化响应：集中度过高（HHI 跨阈值）→ 降低最高权重策略的权重。

        - 从 decision.allocation_plan.strategy_allocations 找出 target_weight 最高的策略，
          生成 reallocate decrease 动作将其降到 reduce_target（分散化）。
        - 仅当最高权重 > reduce_target 时才生成（否则已足够分散，无需动作）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        补足「组合级风险诊断→响应」的缺口（此前 high_concentration 只告警不动作）。
        """
        if not self._diversification_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        for alert in alerts or []:
            if alert.get("type") != "high_concentration":
                continue
            plan = decision.get("allocation_plan") or {}
            allocs = plan.get("strategy_allocations") or {}
            if not isinstance(allocs, dict) or not allocs:
                continue
            top_name = max(
                allocs,
                key=lambda n: safe_float(
                    (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                    0.0,
                ),
            )
            top_weight = safe_float(
                (allocs[top_name] or {}).get("target_weight"), 0.0
            ) if isinstance(allocs.get(top_name), dict) else 0.0
            if top_weight <= self._diversification_reduce_target:
                continue
            actions.append({
                "type": "reallocate",
                "strategy": top_name,
                "action": "decrease",
                "target_allocation": self._diversification_reduce_target,
                "reason": (
                    f"diversification: 集中度过高，降低 {top_name} 权重 "
                    f"{top_weight:.0%} → {self._diversification_reduce_target:.0%}"
                ),
            })
        return actions

    def _correlation_response_actions(self, decision: Dict[str, Any],
                                      alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级相关性响应：高相关 / 有效 N 侵蚀 → 降低最高权重策略的权重。

        - 从 decision.allocation_plan.strategy_allocations 找出 target_weight 最高的策略，
          生成 reallocate decrease 动作降到 reduce_target（降低相关性集中暴露）。
        - 仅当最高权重 > reduce_target 时才生成（否则已足够保守，无需动作）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        与 diversification（集中度 HHI）互补，覆盖组合级风险的相关性维度。
        """
        if not self._correlation_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        triggered = any(
            alert.get("type") in ("high_correlation", "diversification_eroding")
            for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._correlation_reduce_target:
            return []
        actions.append({
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._correlation_reduce_target,
            "reason": (
                f"correlation: 策略间高相关/有效N侵蚀，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._correlation_reduce_target:.0%}"
            ),
        })
        return actions

    def _portfolio_efficiency_actions(self, decision: Dict[str, Any],
                                      alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级资金效率响应：资金效率偏低 → 收敛最高权重策略。

        - low_capital_efficiency 告警说明组合整体资金效率（efficiency_score）低于阈值——
          费率效率、资本回报、风险调整后收益综合偏弱，资金未被高效利用。生成 reallocate
          decrease 将最高权重策略降到 reduce_target，收敛低效资金暴露、释放占用资金。
        - 与 _diversification_actions（集中度过高）、_correlation_response_actions（高相关）、
          _synergy_actions（协同劣化）、_tail_risk_actions（尾部风险）区分：前四者看组合
          结构/联动/尾部维度，本方法看「资金效率」维度（资金是否被高效利用）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._portfolio_efficiency_enabled:
            return []
        triggered = any(
            alert.get("type") == "low_capital_efficiency" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._portfolio_efficiency_reduce_target:
            return []
        actions: List[Dict[str, Any]] = []
        actions.append({
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._portfolio_efficiency_reduce_target,
            "reason": (
                f"portfolio_efficiency: 组合资金效率偏低，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._portfolio_efficiency_reduce_target:.0%}"
            ),
        })
        return actions

    def _synergy_actions(self, decision: Dict[str, Any],
                         alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级协同度响应：协同度低于阈值 → 收敛最高权重策略。

        - low_synergy 告警说明组合协同度（synergy_score）低于阈值——策略「相关且一起亏」
          （协同劣化），生成 reallocate decrease 将最高权重策略降到 reduce_target，收敛
          组合敞口防相互拖累。
        - 与 _correlation_response_actions（高相关/有效N侵蚀）区分：后者看策略间收益联动
          （相关性），本方法看综合相关性 + 盈亏同向性的「协同质量」（低协同 = 相关 + 一起亏），
          是更严重的联动风险信号。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._synergy_enabled:
            return []
        triggered = any(
            alert.get("type") == "low_synergy" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._synergy_reduce_target:
            return []
        actions: List[Dict[str, Any]] = [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._synergy_reduce_target,
            "reason": (
                f"synergy: 组合协同度低于阈值，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._synergy_reduce_target:.0%}"
            ),
        }]
        return actions

    def _tail_risk_actions(self, decision: Dict[str, Any],
                           alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级尾部风险响应：high_tail_risk → 降低尾部暴露最高的策略权重。

        - high_tail_risk 告警已记录「最深回撤策略」（tail_strategy），直接对该策略生成
          reallocate decrease 降到 reduce_target，收敛尾部暴露。
        - 仅当该策略当前权重 > reduce_target 时才生成（否则已足够收敛）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        与 max_drawdown_trend_guard（单策略回撤趋势）互补，覆盖组合级回撤绝对水平维度。
        """
        if not self._tail_risk_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        for alert in alerts or []:
            if alert.get("type") != "high_tail_risk":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            info = allocs.get(strategy) if isinstance(allocs, dict) else None
            current = safe_float(info.get("target_weight"), 1.0) if isinstance(info, dict) else 1.0
            if current <= self._tail_risk_reduce_target:
                continue
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._tail_risk_reduce_target,
                "reason": f"tail_risk: {alert.get('message', '')}",
            })
        return actions

    def _portfolio_resonance_actions(self, decision: Dict[str, Any],
                                     alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级风险共振响应：多维度组合风险同时恶化 → 更保守的全局收敛。

        - 组合级四子维度（相关性/权重集中/盈利集中/尾部风险）≥ resonance_threshold 个
          同时触发时，生成 reallocate decrease 将最高权重策略降到 reduce_target（默认0.1，
          比单维度 reduce_target 0.2 更保守）。
        - 与策略级 _resonance_actions（单策略多维度衰退 → 降杠杆 0.5）对称：组合级共振
          说明组合结构整体恶化，需要更大力度的收敛。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._portfolio_resonance_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        triggered = any(
            alert.get("type") == "portfolio_risk_resonance" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._portfolio_resonance_reduce_target:
            return []
        actions.append({
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._portfolio_resonance_reduce_target,
            "reason": (
                f"portfolio_resonance: 组合级风险多维度共振，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._portfolio_resonance_reduce_target:.0%}"
            ),
        })
        return actions

    def _portfolio_health_actions(self, decision: Dict[str, Any],
                                  alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级健康度阈值响应：组合整体健康度跌破健康线 → 收敛最高权重策略。

        - portfolio_health_low 告警说明组合整体健康度（overall_health_score）已跌破
          health_threshold（绝对值差），生成 reallocate decrease 将最高权重策略降到
          reduce_target，收敛组合敞口防恶化。
        - 与 _portfolio_health_trend_actions（趋势型：连续下降）区分：本方法看组合整体健康度
          的绝对水平（阈值型）；与策略级 _health_trend_actions（param_adjust 降杠杆）对称：
          策略级降单策略杠杆，组合级收敛组合敞口（无单一策略目标，故降最高权重策略）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._portfolio_health_enabled:
            return []
        triggered = any(
            alert.get("type") == "portfolio_health_low" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._portfolio_health_reduce_target:
            return []
        actions: List[Dict[str, Any]] = [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._portfolio_health_reduce_target,
            "reason": (
                f"portfolio_health: 组合整体健康度低于阈值，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._portfolio_health_reduce_target:.0%}"
            ),
        }]
        return actions

    def _portfolio_health_trend_actions(self, decision: Dict[str, Any],
                                        alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级健康度趋势响应：组合整体健康度连续下降 → 收敛最高权重策略。

        - portfolio_health_deteriorating 告警说明组合整体健康度连续 window 周期下降
          （温水煮青蛙式组合级衰退），生成 reallocate decrease 将最高权重策略降到
          reduce_target，收敛组合敞口防恶化。
        - 与策略级 _health_trend_actions（param_adjust 降杠杆）对称：策略级降单策略杠杆，
          组合级收敛组合敞口（无单一策略目标，故降最高权重策略）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._portfolio_health_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        triggered = any(
            alert.get("type") == "portfolio_health_deteriorating" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._portfolio_health_trend_reduce_target:
            return []
        actions.append({
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._portfolio_health_trend_reduce_target,
            "reason": (
                f"portfolio_health_trend: 组合整体健康度连续下降，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._portfolio_health_trend_reduce_target:.0%}"
            ),
        })
        return actions

    def _portfolio_concentration_trend_actions(self, decision: Dict[str, Any],
                                               alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级集中度趋势响应：集中度连续上升 → 收敛最高权重策略。

        - portfolio_concentration_rising 告警说明组合资金配置集中度（HHI）连续 window 周期
          上升（分散度被侵蚀），生成 reallocate decrease 将最高权重策略降到 reduce_target，
          收敛集中敞口防风险堆积。
        - 与 _diversification_actions（阈值型：HHI 超阈值才收敛，事后）区分：本方法在集中度
          上升轨迹中提前收敛（事前）；与 _portfolio_health_trend_actions（健康度下降）区分：
          本方法看结构维度的集中度，后者看综合质量。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._portfolio_concentration_trend_enabled:
            return []
        triggered = any(
            alert.get("type") == "portfolio_concentration_rising" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._portfolio_concentration_trend_reduce_target:
            return []
        actions: List[Dict[str, Any]] = [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._portfolio_concentration_trend_reduce_target,
            "reason": (
                f"portfolio_concentration_trend: 组合集中度连续上升，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._portfolio_concentration_trend_reduce_target:.0%}"
            ),
        }]
        return actions

    def _portfolio_correlation_trend_actions(self, decision: Dict[str, Any],
                                             alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级相关性趋势响应：相关性连续上升 → 收敛最高权重策略。

        - portfolio_correlation_rising 告警说明策略间最大成对相关性（max_pair_correlation）
          连续 window 周期上升（收益趋同、联动风险加剧），生成 reallocate decrease 将最高
          权重策略降到 reduce_target，收敛敞口防同步下跌。
        - 与 _correlation_response_actions（阈值型：max_pair 超阈值才收敛，事后）区分：
          本方法在相关性上升轨迹中提前收敛（事前）；与 _portfolio_concentration_trend_actions
          （集中度上升）区分：本方法看收益联动（相关性），后者看资金配置权重，二者正交。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._portfolio_correlation_trend_enabled:
            return []
        triggered = any(
            alert.get("type") == "portfolio_correlation_rising" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._portfolio_correlation_trend_reduce_target:
            return []
        actions: List[Dict[str, Any]] = [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._portfolio_correlation_trend_reduce_target,
            "reason": (
                f"portfolio_correlation_trend: 组合相关性连续上升，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._portfolio_correlation_trend_reduce_target:.0%}"
            ),
        }]
        return actions

    def _portfolio_tail_risk_trend_actions(self, decision: Dict[str, Any],
                                           alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级尾部风险趋势响应：尾部风险连续加深 → 收敛最高权重策略。

        - tail_risk_rising 告警说明组合最深单策略回撤（tail_risk = max(max_drawdown_i)）连续
          window 周期加深（尾部暴露扩大），生成 reallocate decrease 将最高权重策略降到
          reduce_target，收敛敞口防尾部亏损扩散。
        - 与 _tail_risk_actions（阈值型：max_dd 超阈值才收敛，事后）区分：本方法在尾部风险
          加深轨迹中提前收敛（事前）；与 _max_drawdown_trend_actions（单策略回撤趋势，降单
          策略杠杆）区分：本方法看组合级最深回撤，收敛组合敞口。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._portfolio_tail_risk_trend_enabled:
            return []
        triggered = any(
            alert.get("type") == "tail_risk_rising" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._portfolio_tail_risk_trend_reduce_target:
            return []
        actions: List[Dict[str, Any]] = [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._portfolio_tail_risk_trend_reduce_target,
            "reason": (
                f"portfolio_tail_risk_trend: 组合尾部风险连续加深，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._portfolio_tail_risk_trend_reduce_target:.0%}"
            ),
        }]
        return actions

    def _profit_concentration_trend_actions(self, decision: Dict[str, Any],
                                            alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级盈利集中度趋势响应：盈利来源集中度连续上升 → 收敛主导盈利策略。

        - profit_concentration_rising 告警说明组合盈利来源集中度（max(total_pnl_i)/total_pnl）
          连续 window 周期上升（单一盈利支柱风险加剧），生成 reallocate decrease 将主导盈利
          策略（告警 strategy）降到 reduce_target，收敛支柱暴露防单一支柱塌陷。
        - 与 profit_concentration_guard（阈值型：concentration 超阈值才门控，且仅抑制该支柱
          进攻不加仓、无 reallocate）区分：本方法在集中度上升轨迹中提前主动收敛主导策略权重
          （事前）；与 _portfolio_concentration_trend_actions（资金配置权重 HHI 上升，降最高权重
          策略）区分：本方法看盈利来源集中（收益结构），降主导盈利策略，后者看配置结构，正交。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._profit_concentration_trend_enabled:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "profit_concentration_rising":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            info = allocs.get(strategy) if isinstance(allocs, dict) else None
            current = safe_float(info.get("target_weight"), 1.0) if isinstance(info, dict) else 1.0
            if current <= self._profit_concentration_trend_reduce_target:
                continue
            actions.append({
                "type": "reallocate",
                "strategy": strategy,
                "action": "decrease",
                "target_allocation": self._profit_concentration_trend_reduce_target,
                "reason": (
                    f"profit_concentration_trend: 组合盈利来源集中度连续上升，降低主导盈利策略 "
                    f"{strategy} 权重 {current:.0%} → {self._profit_concentration_trend_reduce_target:.0%}"
                ),
            })
        return actions

    def _synergy_trend_actions(self, decision: Dict[str, Any],
                               alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级协同度趋势响应：协同度连续下降 → 收敛最高权重策略。

        - synergy_deteriorating 告警说明组合协同度（synergy_score）连续 window 周期下降
          （协同质量劣化轨迹），生成 reallocate decrease 将最高权重策略降到 reduce_target，
          收敛组合敞口防相互拖累。
        - 与 _synergy_actions（阈值型：跌破阈值才收敛）区分：本方法在协同度下降轨迹中提前
          收敛（事前）；与其他组合级趋势守卫（portfolio_concentration_trend 等）同构：无单一
          策略目标，故降最高权重策略。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._synergy_trend_enabled:
            return []
        triggered = any(
            alert.get("type") == "synergy_deteriorating" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._synergy_trend_reduce_target:
            return []
        actions: List[Dict[str, Any]] = [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._synergy_trend_reduce_target,
            "reason": (
                f"synergy_trend: 组合协同度连续下降，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._synergy_trend_reduce_target:.0%}"
            ),
        }]
        return actions

    def _portfolio_efficiency_trend_actions(self, decision: Dict[str, Any],
                                            alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合级资金效率趋势响应：资金效率连续下降 → 收敛最高权重策略。

        - efficiency_deteriorating 告警说明组合资金效率（efficiency_score）连续 window 周期
          下降（资金利用效率劣化轨迹），生成 reallocate decrease 将最高权重策略降到
          reduce_target，收敛低效资金暴露、释放占用资金。
        - 与 _portfolio_efficiency_actions（阈值型：跌破阈值才收敛）区分：本方法在资金效率
          下降轨迹中提前收敛（事前）；与其他组合级趋势守卫（synergy_trend 等）同构：无单一
          策略目标，故降最高权重策略。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._portfolio_efficiency_trend_enabled:
            return []
        triggered = any(
            alert.get("type") == "efficiency_deteriorating" for alert in (alerts or [])
        )
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._portfolio_efficiency_trend_reduce_target:
            return []
        actions: List[Dict[str, Any]] = [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._portfolio_efficiency_trend_reduce_target,
            "reason": (
                f"portfolio_efficiency_trend: 组合资金效率连续下降，降低 {top_name} 权重 "
                f"{top_weight:.0%} → {self._portfolio_efficiency_trend_reduce_target:.0%}"
            ),
        }]
        return actions

    def _goal_planning_actions(self, decision: Dict[str, Any],
                               alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """目标导向规划动作：据收益/回撤约束动态调整风险。

        - goal_reached（收益达标）→ 复用 profit_take_close 落袋减仓（低风险自动执行）。
        - near_drawdown_limit（接近回撤约束）→ 对每个策略生成 reallocate 减配动作，
          目标权重回落到 risk_response 的 defensive 目标（完全自主模式下降仓防守）。
        """
        if not self._goal_planning_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        for alert in alerts or []:
            atype = alert.get("type")
            if atype == "goal_reached":
                actions.append({
                    "type": "profit_take_close",
                    "close_ratio": self._goal_reached_reduce_target,
                    "reason": (
                        f"goal_planning: 收益达标 {alert.get('return_pct', 0):.1%}，"
                        f"落袋 {self._goal_reached_reduce_target:.0%} 仓位"
                    ),
                })
            elif atype == "near_drawdown_limit":
                for name in decision.get("strategy_names") or []:
                    actions.append({
                        "type": "reallocate",
                        "strategy": name,
                        "action": "decrease",
                        "target_allocation": self._risk_response_reduce_target,
                        "reason": (
                            f"goal_planning: 回撤接近约束，策略 {name} 降仓防守"
                        ),
                    })
        return actions

    def _param_adjust_actions(
        self,
        alerts: List[Dict[str, Any]],
        decision: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """策略参数自适应动作：健康度 F → 降到 health_f_leverage；
        趋势恶化 → 降到 declining_leverage。由 scheduler 落地（策略热重载杠杆）。

        限速冷却：同一策略距上次调整不足 min_interval_cycles 周期则跳过，防止
        AGI 过度迭代杠杆（曾出现 1.5h 迭代 20 版），让参数沉淀验证后再调整。
        """
        if not self._param_adaptation_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            atype = alert.get("type")
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            if atype == "strategy_health_critical":
                value = self._param_health_f_leverage
            elif atype == "strategy_declining":
                value = self._param_declining_leverage
            else:
                continue
            # 限速冷却：距上次调整不足 min_interval_cycles 周期则跳过
            if self._param_adapt_min_interval_cycles > 0:
                last = self._param_adapt_last_cycle.get(str(strategy))
                if (last is not None
                        and self._cycle_count - last < self._param_adapt_min_interval_cycles):
                    continue

            if self.rl_agent is not None:
                decision_data = decision or {}
                metrics = _coerce_dict(decision_data.get("strategy_metrics"))
                strategy_metrics = _coerce_dict(metrics.get(strategy))
                state = StateEncoding(
                    market_regime=str(decision_data.get("market_regime") or "unknown"),
                    volatility_percentile=safe_float(
                        strategy_metrics.get("volatility_percentile"), 0.5
                    ),
                    trend_strength=safe_float(
                        strategy_metrics.get("trend_strength"),
                        safe_float(decision_data.get("market_regime_strength"), 0.0),
                    ),
                    current_drawdown_pct=safe_float(
                        strategy_metrics.get("max_drawdown"), 0.0
                    ),
                    position_count=safe_int(strategy_metrics.get("position_count"), 0),
                    strategy_id=str(strategy),
                    regime_confidence=safe_float(
                        decision_data.get("market_regime_confidence"), 0.0
                    ),
                    decision_id=str(decision_data.get("decision_id") or ""),
                )
                try:
                    recommended = safe_float(
                        self.rl_agent.get_parameter_adjustment("leverage", value, state),
                        value,
                    )
                    value = min(value, recommended)
                except Exception as exc:
                    logger.warning(
                        f"[AGI] RL leverage recommendation failed for {strategy}; "
                        f"keeping risk target: {exc}"
                    )

            seen.add(strategy)
            self._param_adapt_last_cycle[str(strategy)] = self._cycle_count
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": value,
                "reason": f"param_adaptation:{atype}: {alert.get('message', '')}",
            })
        return actions

    def _param_restore_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """参数自适应恢复动作（降杠杆的对称闭环）：健康度恢复 → 恢复杠杆到 restore_leverage。

        - 补足 param_adaptation 的「单向降杠杆」缺陷：策略健康度 F 降到 1.0 后，
          即使恢复 A/B，杠杆也永久停留低位 → 过度保守、错失盈利。
        - 触发：strategy_recovered（grade A/B + trend improving + 恢复纯度门控已通过）。
        - 安全门控（升杠杆风险高于降杠杆，三重保护）：
          1) 策略未被 AGI 暂停（_paused_strategies 之外，暂停策略不升杠杆）
          2) 本周期该策略无回吐告警（pnl_give_back*，避免回吐中升杠杆）
          3) 冷却：距上次 param 调整不足 restore_min_interval_cycles 周期则跳过
             （与降杠杆共用 _param_adapt_last_cycle，天然防「降了又升」振荡）
        - 由 scheduler 落地后仍受 ParameterRollbackGuard 30 分钟验证（效果差回滚），
          双保险兜底。
        """
        if not self._param_restore_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        # 本周期有回吐告警的策略集合（回吐中不升杠杆）
        giving_back = {
            a.get("strategy") for a in alerts or []
            if a.get("type") in ("pnl_give_back", "pnl_give_back_severe",
                                 "pnl_give_back_critical")
        }
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "strategy_recovered":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            # 安全门控 1：暂停策略不升杠杆
            if str(strategy) in self._paused_strategies:
                continue
            # 安全门控 2：回吐中不升杠杆
            if strategy in giving_back:
                continue
            # 安全门控 3：冷却（与降杠杆共用 _param_adapt_last_cycle）
            if self._param_restore_min_interval_cycles > 0:
                last = self._param_adapt_last_cycle.get(str(strategy))
                if (last is not None and
                        self._cycle_count - last < self._param_restore_min_interval_cycles):
                    continue
            seen.add(strategy)
            self._param_adapt_last_cycle[str(strategy)] = self._cycle_count
            # 恢复分级：A 级完整恢复，B 级（或未知）保守恢复到中间值
            grade = alert.get("grade")
            restore_value = (
                self._param_restore_leverage if grade == "A"
                else self._param_restore_leverage_b
            )
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": restore_value,
                "reason": f"param_adaptation:restore: {alert.get('message', '健康度恢复，恢复杠杆')}",
            })
        return actions

    def _symbol_param_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """逐币种精细杠杆守卫动作：单币种浮亏超阈值 → 该币种单独下调 leverage_default。

        与 _param_adjust_actions 区分：后者按「策略」维度降杠杆（strategy-level），
        本方法按「币种」维度精细下调（symbol-level），落到 currencies.symbol_overrides，
        实现 ARB 等币种在 tier 默认参数之上单独定制更精细的杠杆。复用 param_adjust
        低风险自动执行通道（scheduler 的 _param_adjust_deployer 支持 symbol 维度落地）。
        """
        if not self._symbol_param_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        for alert in alerts or []:
            if alert.get("type") != "symbol_losing":
                continue
            symbol = alert.get("symbol")
            if not symbol:
                continue
            # 冷却期内不重复下调（与 min_interval_cycles 同构：控频率防反复降杠杆）
            cd = self._symbol_leverage_cooldown.get(symbol)
            if cd is not None and self._symbol_cooldown_cycles > 0:
                if self._cycle_count - cd < self._symbol_cooldown_cycles:
                    continue
            # 当前有效杠杆（含 symbol_overrides 覆盖），向下夹紧到 min_leverage
            try:
                from configs.settings import get_symbol_config
                ts = get_symbol_config(symbol, self.config)
            except Exception:
                ts = {}
            cur = safe_float(ts.get("leverage_default"), safe_float(alert.get("leverage"), 0.0))
            if cur <= 0:
                cur = 1.0
            new_lev = max(self._symbol_min_leverage, cur - self._symbol_reduce_leverage_step)
            if new_lev >= cur:
                continue  # 已在杠杆下限，不再下调
            self._symbol_leverage_cooldown[symbol] = self._cycle_count
            actions.append({
                "type": "param_adjust",
                "symbol": symbol,
                "param": "leverage_default",
                "value": new_lev,
                "reason": (
                    f"symbol_param_guard: {symbol} 浮亏 {safe_float(alert.get('upl'), 0.0):.2f} "
                    f"超阈值，杠杆 {cur:.0f}x -> {new_lev:.0f}x"
                ),
            })
        return actions

    def _spot_hold_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """现货持有守卫动作：spot_overweight / spot_profit → reallocate decrease 收敛现货。

        - 现货过度分散（持币种类超限）或现货累计盈利达标时，降低现货策略
          （spot_grid/spot_martingale）资金权重到 reduce_target，兑现「现货不占用
          过多资金、稳步增长」。
        - 复用 reallocate 高风险通道，autonomous 模式受 Kill Switch 约束。
        """
        if not self._spot_hold_guard_enabled:
            return []
        triggered = {a.get("type") for a in alerts or []}
        if not ({"spot_overweight", "spot_profit", "spot_loss"} & triggered):
            return []
        actions: List[Dict[str, Any]] = []
        if "spot_overweight" in triggered:
            reason = "spot_hold_guard: 现货过度分散，收敛敞口"
        elif "spot_loss" in triggered:
            reason = "spot_hold_guard: 现货累计亏损达标，止损回收资金到合约"
        else:
            reason = "spot_hold_guard: 现货累计盈利达标，止盈落袋"
        for s in self._spot_strategies:
            actions.append({
                "type": "reallocate",
                "strategy": s,
                "action": "decrease",
                "target_allocation": self._spot_reduce_target,
                "reason": reason,
            })
        return actions

    def _hourly_pnl_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """每小时盈利效率守卫动作：hourly_pnl_negative → reallocate decrease 收敛资金权重。

        - 活跃时长足够、交易量足够但每小时仍在持续失血的策略，说明其长时间占用资金
          却负期望（时间价值流失），降低其资金权重到 reduce_target，消除低效策略。
        - 复用 reallocate 高风险通道，autonomous 模式受 Kill Switch 约束。
        """
        if not self._hourly_pnl_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        for alert in alerts or []:
            if alert.get("type") != "hourly_pnl_negative":
                continue
            name = alert.get("strategy")
            if not name:
                continue
            actions.append({
                "type": "reallocate",
                "strategy": name,
                "action": "decrease",
                "target_allocation": self._hourly_reduce_target,
                "reason": f"hourly_pnl_guard: {alert.get('message', '')}",
            })
        return actions

    def _trade_frequency_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """交易频率守卫动作：high_trade_frequency → reallocate decrease 收敛资金权重。

        - 每小时交易笔数过高（高频刷单）的策略，即使盈利，单笔滑点/手续费/冲击成本叠加
          也持续侵蚀利润，降低其资金权重到 reduce_target，消除「频繁交易消耗资金」。
        - 复用 reallocate 高风险通道，autonomous 模式受 Kill Switch 约束。
        """
        if not self._trade_frequency_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        for alert in alerts or []:
            if alert.get("type") != "high_trade_frequency":
                continue
            name = alert.get("strategy")
            if not name:
                continue
            actions.append({
                "type": "reallocate",
                "strategy": name,
                "action": "decrease",
                "target_allocation": self._trade_frequency_reduce_target,
                "reason": f"trade_frequency_guard: {alert.get('message', '')}",
            })
        return actions

    def _health_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """健康度趋势外推动作：连续下降 → param_adjust 提前降杠杆（事前风控）。

        - 与 _param_adjust_actions 区分：前者响应「已 F / 已趋势恶化」（事后），
          本方法响应「health_deteriorating」告警（健康度连续下降、仍处 B/C 级时），
          在策略真正掉到 F 之前收敛敞口。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._health_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "health_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._health_trend_leverage,
                "reason": f"health_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _win_rate_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """胜率趋势外推动作：连续下降 → param_adjust 提前降杠杆（事前风控）。

        - 与 _health_trend_actions 互补：后者看综合健康度（含 PnL/回撤/Sharpe），
          本方法单看胜率趋势——一个策略可能 PnL 正、健康度 A/B 但胜率持续下滑
          （赢少输多但单笔赢大于输），此时综合健康度尚未恶化，但策略实际已在劣化。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._win_rate_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "win_rate_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._win_rate_trend_leverage,
                "reason": f"win_rate_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _win_rate_guard_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """胜率绝对值守卫动作：low_win_rate → param_adjust 提前降杠杆（阈值型风控）。

        - 与 _win_rate_trend_actions（连续下降，趋势型）区分：本方法看「绝对阈值」——
          胜率已低于 min_win_rate 但尚未连续下降（可能长期在低位徘徊）时也应降杠杆收敛，
          构成胜率维度「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        - 同一策略去重，只生成一条降杠杆动作。
        """
        if not self._win_rate_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "low_win_rate":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._win_rate_guard_leverage,
                "reason": f"win_rate_guard: {alert.get('message', '')}",
            })
        return actions

    def _sharpe_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """夏普趋势外推动作：连续下降 → param_adjust 提前降杠杆（事前风控）。

        - 与 _health_trend_actions / _win_rate_trend_actions 互补：
          health_trend 看综合健康度，win_rate_trend 看胜率频率，本方法看风险调整后收益。
          夏普下降意味着策略在承担更多风险获取同等收益——一个策略可能胜率高、
          PnL 正、健康度 A/B 但夏普持续下滑（波动增大），此时其他守卫尚未告警，
          但策略实际已在劣化（单位风险收益递减）。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._sharpe_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "sharpe_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._sharpe_trend_leverage,
                "reason": f"sharpe_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _sharpe_ratio_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """夏普比率绝对阈值动作：negative_sharpe → param_adjust 提前降杠杆（事前风控）。

        - 与 _sharpe_trend_actions（连续下降，趋势型）区分：本方法看「绝对阈值」——负夏普
          （风险调整后负收益）是纯亏损硬信号，即使未连续下降也应提前收敛敞口，避免在
          硬冻结/深度回撤前继续「承担风险却负收益」。构成夏普维度「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._sharpe_ratio_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "negative_sharpe":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._sharpe_ratio_leverage,
                "reason": f"sharpe_ratio_guard: {alert.get('message', '')}",
            })
        return actions

    def _consecutive_losses_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """连续亏损收敛动作：strategy_losing_streak → param_adjust 提前降杠杆（事前风控）。

        - 与事前风控三子守卫（health_trend/win_rate_trend/sharpe_trend）互补：
          三者看趋势（连续下降），本方法看阈值（consecutive_losses 直接达 loss_threshold）。
          连续亏损是亏损侧的硬信号——即使健康度/胜率/夏普尚未趋势恶化，连续亏损达阈值
          即应提前收敛敞口，避免在硬冻结（≥5）前继续放大亏损。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._consecutive_losses_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "strategy_losing_streak":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._consecutive_losses_leverage,
                "reason": f"consecutive_losses_guard: {alert.get('message', '')}",
            })
        return actions

    def _stop_loss_frequency_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """止损频率收敛动作：high_stop_loss_rate → param_adjust 提前降杠杆（事前风控）。

        - 与 _consecutive_losses_actions（连续亏损达阈值）互补：后者看连败（亏损连续），
          本方法看止损占比（即使亏损不连续，止损交易占比过高也说明入场时机差）。止损频繁
          往往是「追高开多/追低开空」的直接结果——收敛杠杆倒逼改善入场时机。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._stop_loss_frequency_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "high_stop_loss_rate":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._stop_loss_frequency_leverage,
                "reason": f"stop_loss_frequency_guard: {alert.get('message', '')}",
            })
        return actions

    def _take_profit_ratio_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """止盈止损比收敛动作：low_take_profit_ratio → param_adjust 提前降杠杆（事前风控）。

        - 与 _stop_loss_frequency_actions（止损率过高，看入场时机差）互补：后者看「止损占比」，
          本方法看「止盈/止损比」（离场质量）——策略善止损不善止盈、赚的时候不落袋，收敛杠杆
          倒逼改善离场纪律。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._take_profit_ratio_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "low_take_profit_ratio":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._take_profit_ratio_leverage,
                "reason": f"take_profit_ratio_guard: {alert.get('message', '')}",
            })
        return actions

    def _win_loss_ratio_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """盈亏比收敛动作：low_win_loss_ratio → param_adjust 提前降杠杆（事前风控）。

        - 与 _take_profit_ratio_actions（止盈止损笔数比，看离场「笔数」质量）互补：本方法看
          「单笔盈亏金额结构」——平均盈利相对平均亏损过小（小赚大亏），是追涨杀跌的典型盈亏
          特征，收敛杠杆倒逼改善盈亏结构。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._win_loss_ratio_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "low_win_loss_ratio":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._win_loss_ratio_leverage,
                "reason": f"win_loss_ratio_guard: {alert.get('message', '')}",
            })
        return actions

    def _take_profit_pnl_ratio_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """止盈止损盈亏金额比收敛动作：low_take_profit_pnl_ratio → param_adjust 提前降杠杆。

        - 与 _take_profit_ratio_actions（止盈/止损笔数比）、_win_loss_ratio_actions（单笔盈亏金额比）
          互补：本方法看「累计金额」——止盈单总赚的少于止损单总亏的（盈利单轻仓、亏损单重仓），
          是追涨杀跌的仓位管理特征，收敛杠杆倒逼改善仓位纪律。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._take_profit_pnl_ratio_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "low_take_profit_pnl_ratio":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._take_profit_pnl_ratio_leverage,
                "reason": f"take_profit_pnl_ratio_guard: {alert.get('message', '')}",
            })
        return actions

    def _risk_adjusted_contribution_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """风险调整后贡献守卫动作：low_risk_adjusted_contribution → param_adjust 提前降杠杆。

        - 与 _max_drawdown_trend_actions（回撤深度趋势，事前）、tail_risk_guard（回撤绝对水平，
          阈值）区分：本方法看「风险调整后贡献」（PnL/MaxDD 比值）——策略仍盈利但盈利相对
          所承受回撤过少（小赚大扛），是风险收益比失衡，收敛杠杆倒逼改善盈亏质量。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._risk_adjusted_contribution_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "low_risk_adjusted_contribution":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._risk_adjusted_contribution_leverage,
                "reason": f"risk_adjusted_contribution_guard: {alert.get('message', '')}",
            })
        return actions

    def _strategy_staleness_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """策略休眠资金回收动作：strategy_stale → reallocate decrease 回收闲置资金。

        - 与 strategy_lifecycle（strategy_dormant → strategy_pause 停开新仓）区分：本方法在更早的
          reclaim_idle_hours 回收资金（reallocate decrease 到 reclaim_target），事前释放闲置资金；
          strategy_pause 是事后停开新仓。二者互补：先回收资金、再停开新仓。
        - 复用 reallocate 高风险通道，autonomous 模式受 Kill Switch 约束。
        """
        if not self._strategy_staleness_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        for alert in alerts or []:
            if alert.get("type") != "strategy_stale":
                continue
            name = alert.get("strategy")
            if not name:
                continue
            actions.append({
                "type": "reallocate",
                "strategy": name,
                "action": "decrease",
                "target_allocation": self._strategy_staleness_reclaim_target,
                "reason": f"strategy_staleness_guard: {alert.get('message', '')}",
            })
        return actions

    def _health_crash_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """健康度骤降动作：health_crash → param_adjust 提前降杠杆（急信号）。

        - 与 _health_trend_actions（连续 window 周期渐降，慢信号）区分：本方法响应「单周期
          骤降」（delta_health 暴跌），在健康度突然恶化时立即收敛，不必等待连续下降确认。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._health_crash_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "health_crash":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._health_crash_leverage,
                "reason": f"health_crash_guard: {alert.get('message', '')}",
            })
        return actions

    def _net_exposure_actions(self, decision: Dict[str, Any],
                              alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合多空净敞口响应：directional_imbalance → 降低最高权重策略敞口。

        - 方向性失衡（净敞口占毛敞口比例过高）时，降低保证金占用最多（回退到 target_weight
          最高）策略的权重，收敛组合单边敞口，避免过度单边押注。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._net_exposure_enabled:
            return []
        triggered = any(a.get("type") == "directional_imbalance" for a in (alerts or []))
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._net_exposure_reduce_target:
            return []
        return [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._net_exposure_reduce_target,
            "reason": "net_exposure_guard: 组合方向性失衡，收敛最高权重策略敞口",
        }]

    def _account_leverage_actions(self, decision: Dict[str, Any],
                                  alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """账户级高杠杆响应：high_account_leverage → 降低最高权重策略敞口。

        - 账户级杠杆逼近 max_total_leverage（soft_threshold_ratio 触发）时，降低保证金占用
          最多（回退到 target_weight 最高）策略的权重，事前收敛总敞口，避免触发
          account_manager 硬性减仓（市场单、有滑点损失）。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._account_leverage_enabled:
            return []
        triggered = any(a.get("type") == "high_account_leverage" for a in (alerts or []))
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._account_leverage_reduce_target:
            return []
        return [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._account_leverage_reduce_target,
            "reason": "account_leverage_guard: 账户杠杆逼近上限，收敛最高权重策略敞口",
        }]

    def _pending_margin_actions(self, decision: Dict[str, Any],
                                alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """挂单保证金响应：high_pending_margin → 降低最高权重策略敞口。

        - 挂单锁定保证金占权益比例过高（大量未成交挂单，如网格密集挂单）时，降低保证金占用
          最多（回退到 target_weight 最高）策略的权重，事前收敛总敞口，避免挂单成交后敞口突增。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._pending_margin_enabled:
            return []
        triggered = any(a.get("type") == "high_pending_margin" for a in (alerts or []))
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._pending_margin_reduce_target:
            return []
        return [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._pending_margin_reduce_target,
            "reason": "pending_margin_guard: 隐性挂单敞口过高，收敛最高权重策略敞口",
        }]

    def _stress_actions(self, decision: Dict[str, Any],
                        alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合压力测试响应：stress_test_failed → 降低最高权重策略敞口。

        - 尾部压力测试失败（压力损失占权益超预算）时，降低最高权重策略权重，收敛总敞口。
        - 复用 reallocate 高风险动作：autonomous 模式受 Kill Switch 约束，非自主排队。
        """
        if not self._stress_guard_enabled:
            return []
        triggered = any(a.get("type") in ("stress_test_failed", "severe_stress_test_failed")
                        for a in (alerts or []))
        if not triggered:
            return []
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if not isinstance(allocs, dict) or not allocs:
            return []
        top_name = max(
            allocs,
            key=lambda n: safe_float(
                (allocs[n] or {}).get("target_weight") if isinstance(allocs[n], dict) else 0.0,
                0.0,
            ),
        )
        top_weight = safe_float(
            (allocs[top_name] or {}).get("target_weight"), 0.0
        ) if isinstance(allocs.get(top_name), dict) else 0.0
        if top_weight <= self._stress_reduce_target:
            return []
        return [{
            "type": "reallocate",
            "strategy": top_name,
            "action": "decrease",
            "target_allocation": self._stress_reduce_target,
            "reason": "portfolio_stress_guard: 尾部压力测试失败，收敛最高权重策略敞口",
        }]

    def _volatility_budget_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """组合波动率预算动作：volatility_over_budget → param_adjust 提前降杠杆。

        - 与 _volatility_trend_actions（波动率连续上升，趋势型）互补：本方法看「绝对阈值」——
          单笔盈亏标准差超预算即收敛，不必等待连续上升确认，构成波动率维度「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._volatility_budget_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "volatility_over_budget":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._volatility_budget_leverage,
                "reason": f"volatility_budget_guard: {alert.get('message', '')}",
            })
        return actions

    def _profit_factor_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """盈亏比趋势外推动作：连续下降 → param_adjust 提前降杠杆（事前风控）。

        - 与 _health_trend_actions / _win_rate_trend_actions / _sharpe_trend_actions 互补：
          health_trend 看综合健康度，win_rate_trend 看胜率频率，sharpe_trend 看风险调整后收益，
          本方法看盈亏幅度比（profit_factor）。profit_factor 下降意味着策略在「赢的幅度
          相对输的幅度」收窄——即使胜率高、PnL 正、健康度 A/B，盈亏比持续下滑也说明
          策略盈利质量在恶化（赢小输大），是独立于夏普的又一先行指标。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._profit_factor_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "profit_factor_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._profit_factor_trend_leverage,
                "reason": f"profit_factor_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _profit_factor_guard_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """盈亏比绝对阈值动作：low_profit_factor → param_adjust 提前降杠杆（事前风控）。

        - 与 _profit_factor_trend_actions（连续下降，趋势型）区分：本方法看「绝对阈值」——盈亏比
          长期 <1（总亏损超过总盈利，已实现净亏损）是纯亏损硬信号，即使未连续下降也应提前收敛
          敞口，避免在硬冻结/深度回撤前继续「频繁交易消耗资金」。构成盈亏比维度「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._profit_factor_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "low_profit_factor":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._profit_factor_guard_leverage,
                "reason": f"profit_factor_guard: {alert.get('message', '')}",
            })
        return actions

    def _max_drawdown_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最大回撤趋势外推动作：连续加深 → param_adjust 提前降杠杆（事前风控）。

        - 与前四子守卫（health_trend/win_rate_trend/sharpe_trend/profit_factor_trend）互补：
          前四子看收益质量维度（综合健康度/胜率频率/风险调整收益/盈亏幅度比），
          本方法看风险深度维度（max_drawdown）。max_drawdown 连续加深意味着策略在承担
          更大风险获取同等收益——即使收益指标全部健康，回撤持续扩大也说明风险敞口
          在劣化，是收益维度守卫无法捕获的独立先行指标。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._max_drawdown_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "max_drawdown_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._max_drawdown_trend_leverage,
                "reason": f"max_drawdown_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _max_drawdown_guard_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """最大回撤绝对值守卫动作：high_max_drawdown → param_adjust 提前降杠杆（阈值型风控）。

        - 与 _max_drawdown_trend_actions（连续加深，趋势型）区分：本方法看「绝对阈值」——
          max_drawdown 已超过阈值但尚未连续加深（可能长期在高位徘徊未恢复）时也应降杠杆收敛，
          构成回撤深度维度「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        - 同一策略去重，只生成一条降杠杆动作。
        """
        if not self._max_drawdown_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "high_max_drawdown":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._max_drawdown_guard_leverage,
                "reason": f"max_drawdown_guard: {alert.get('message', '')}",
            })
        return actions

    def _drawdown_duration_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """回撤持续时间趋势外推动作：连续拉长 → param_adjust 提前降杠杆（事前风控）。

        - 与 _max_drawdown_trend_actions（回撤「深度」连续加深）互补：后者看回撤幅度，
          本方法看回撤「时长」（max_drawdown_duration_hours）。回撤时长连续拉长意味着策略
          长时间无法收复前高、恢复能力下降——即使回撤不深，资金被套牢也是「资金时间价值」
          的损失，是回撤深度守卫无法捕获的独立先行指标。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._drawdown_duration_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "drawdown_duration_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._drawdown_duration_trend_leverage,
                "reason": f"drawdown_duration_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _drawdown_duration_guard_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """回撤持续时间绝对阈值动作：long_drawdown_duration → param_adjust 提前降杠杆（事前风控）。

        - 与 _drawdown_duration_trend_actions（连续拉长，趋势型）区分：本方法看「绝对阈值」——最长
          回撤持续已达阈值（恢复能力差、资金被套牢过久）是「资金时间价值」的硬信号，即使未连续
          拉长也应提前收敛敞口。构成回撤持续维度「阈值 + 趋势」双层（与回撤深度双层对称）。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._drawdown_duration_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "long_drawdown_duration":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._drawdown_duration_guard_leverage,
                "reason": f"drawdown_duration_guard: {alert.get('message', '')}",
            })
        return actions

    def _capital_return_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """资本回报率趋势外推动作：连续下降 → param_adjust 提前降杠杆（事前风控，第六子）。

        - 与前五子守卫互补：前五子看策略自身收益/风险质量（health 综合/win_rate 频率/
          sharpe 单位风险收益/profit_factor 盈亏幅度/max_drawdown 回撤深度），本方法看
          「资本配置效率」（pnl_per_capital_pct = PnL / 配置资本）。资本回报率连续下降
          意味着单位资本产出在递减——即使策略健康度 A/B（盈利、低回撤），配置过多资本
          也会导致资本效率恶化，是收益/风险维度守卫无法捕获的独立先行指标。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._capital_return_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "capital_return_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._capital_return_trend_leverage,
                "reason": f"capital_return_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _capital_return_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """资本回报率阈值动作：capital_return_low → param_adjust 降杠杆收敛。

        - 与 _capital_return_trend_actions（连续下降，趋势型）互补：本方法看「绝对阈值」——
          单位资本产出低于阈值即收敛，不必等待连续下降确认，构成资本效率维度「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._capital_return_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "capital_return_low":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._capital_return_leverage,
                "reason": f"capital_return_guard: {alert.get('message', '')}",
            })
        return actions

    def _volatility_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """波动率趋势外推动作：连续上升 → param_adjust 提前降杠杆（事前风控，第七子）。

        - 与前六子守卫互补：前六子看收益质量（health 综合/win_rate 频率/sharpe 单位风险收益/
          profit_factor 盈亏幅度）、风险深度（max_drawdown）、资本效率（capital_return），
          本方法看「绝对不确定性」（volatility = 单笔盈亏标准差）。波动率连续上升意味着单笔
          盈亏离散度扩大——即使策略 PnL 正、胜率高、健康度 A/B、夏普持平，承担更多不确定性
          换取同等收益也是独立衰退模式，是前六子（含夏普比值）无法捕获的。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._volatility_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "volatility_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._volatility_trend_leverage,
                "reason": f"volatility_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _pnl_momentum_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """PnL 动量趋势外推动作：连续下降 → param_adjust 提前降杠杆（事前风控）。

        - 与七子守卫（health/win_rate/sharpe/profit_factor/max_drawdown/capital_return/volatility）
          互补：本方法看「PnL 动量」——trend_pnl_7d_vs_30d 连续下降意味着策略近期盈利动能相对
          中期持续衰竭（虽仍盈利但增长停滞），是「从盈利走向转亏」的先行指标。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._pnl_momentum_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "pnl_momentum_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._pnl_momentum_trend_leverage,
                "reason": f"pnl_momentum_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _pnl_momentum_guard_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """PnL 动量绝对阈值动作：pnl_momentum_faded → param_adjust 提前降杠杆（事前风控）。

        - 与 _pnl_momentum_trend_actions（连续下降，趋势型）区分：本方法看「绝对阈值」——动量已衰减到
          阈值以下（近期盈利动能相对中期明显减弱）是「增长动能衰竭」的硬信号，即使未连续下降也应提前
          收敛。与 strategy_declining（需 grade D/F）区分：本方法不看健康度。构成 PnL 动量「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._pnl_momentum_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "pnl_momentum_faded":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._pnl_momentum_guard_leverage,
                "reason": f"pnl_momentum_guard: {alert.get('message', '')}",
            })
        return actions

    def _pnl_per_trade_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """单笔期望值趋势外推动作：连续下降 → param_adjust 提前降杠杆（事前风控）。

        - 与九子守卫（health/win_rate/sharpe/profit_factor/max_drawdown/drawdown_duration/capital_return/
          volatility/pnl_momentum）互补：本方法看「单笔期望值」——pnl_per_trade 连续下降意味着策略
          每笔交易价值持续萎缩（靠更多交易堆砌总利润，规模换质量），是「交易质量恶化」的先行指标。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._pnl_per_trade_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "pnl_per_trade_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._pnl_per_trade_trend_leverage,
                "reason": f"pnl_per_trade_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _pnl_per_trade_guard_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """单笔期望值绝对值守卫动作：low_pnl_per_trade → param_adjust 提前降杠杆（阈值型风控）。

        - 与 _pnl_per_trade_trend_actions（连续下降，趋势型）区分：本方法看「绝对阈值」——
          pnl_per_trade 已低于 min_pnl_per_trade 但尚未连续下降（可能长期在低位徘徊）时也应
          降杠杆收敛，构成单笔期望值维度「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        - 同一策略去重，只生成一条降杠杆动作。
        """
        if not self._pnl_per_trade_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "low_pnl_per_trade":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._pnl_per_trade_guard_leverage,
                "reason": f"pnl_per_trade_guard: {alert.get('message', '')}",
            })
        return actions

    # ─────────────────────────────────────────────────────────────
    # 前瞻盈利推算与盈亏归因（pnl_projection）
    # ─────────────────────────────────────────────────────────────

    def _attribute_pnl(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """盈亏归因分析：将 total_pnl 按策略 / regime / 方向 / 退出原因四维分解。

        - 归因先于推算：_project_pnl 依赖 by_regime 提供 regime 历史表现作为先验。
        - fail-closed：contribution 不可用时返回 available=false，不影响后续推算/决策。
        """
        contrib = _coerce_dict(perception.get("contribution"))
        if not contrib.get("available"):
            return {"available": False, "total_pnl": 0.0, "by_strategy": {},
                    "by_regime": {}, "by_direction": {}, "by_exit_reason": {},
                    "attribution_quality": 0.0, "dominant_strategy": None,
                    "worst_strategy": None, "timestamp": datetime.now().isoformat()}

        raw_strategies = _coerce_dict(contrib.get("strategies"))
        strategies = {
            str(name): _coerce_dict(strategy)
            for name, strategy in raw_strategies.items()
            if isinstance(strategy, dict)
        }
        degraded_dimensions = ["strategy_records"] if len(strategies) != len(raw_strategies) else []
        total_pnl = safe_float(contrib.get("total_pnl"), 0.0)
        total_trades = safe_int(contrib.get("total_trades"), 0)

        # ── 按策略归因 ──
        by_strategy: Dict[str, Any] = {}
        try:
            abs_sum = sum(
                abs(safe_float(c.get("total_pnl"), 0.0))
                for c in strategies.values()
            )
            for name, c in strategies.items():
                pnl = safe_float(c.get("total_pnl"), 0.0)
                share = abs(pnl) / abs_sum if abs_sum > 0 else 0.0
                hs = safe_float(c.get("health_score"), 0.0)
                risk_adj = share / max(hs / 100.0, 0.01) if hs > 0 else share
                by_strategy[str(name)] = {
                    "pnl": pnl,
                    "share": share,
                    "risk_adjusted_share": risk_adj,
                    "health_score": hs,
                    "lifecycle": str(c.get("lifecycle", "unknown")),
                }
            dominant = (
                max(by_strategy, key=lambda k: by_strategy[k]["pnl"])
                if by_strategy else None
            )
            worst = (
                min(by_strategy, key=lambda k: by_strategy[k]["pnl"])
                if by_strategy else None
            )
        except Exception:
            logger.exception("[AGI-PnL] strategy attribution failed")
            by_strategy = {}
            dominant = None
            worst = None
            degraded_dimensions.append("by_strategy")

        # ── 按 regime 归因（从 _decision_memory 取跨周期记录） ──
        by_regime: Dict[str, Any] = {}
        try:
            if self._learning_enabled and len(self._decision_memory) > 0:
                regime_buckets: Dict[str, List[float]] = {}
                for mem in self._decision_memory:
                    if not isinstance(mem, dict):
                        continue
                    r = str(mem.get("regime") or "unknown")
                    regime_buckets.setdefault(r, []).append(
                        safe_float(mem.get("cycle_pnl"), 0.0)
                    )
                for r, pnls in regime_buckets.items():
                    cycles = len(pnls)
                    total = sum(pnls)
                    by_regime[r] = {
                        "pnl": total,
                        "cycles": cycles,
                        "avg_per_cycle": total / cycles if cycles > 0 else 0.0,
                        "low_confidence": cycles < 3,
                    }
        except Exception:
            logger.exception("[AGI-PnL] regime attribution failed")
            by_regime = {}
            degraded_dimensions.append("by_regime")

        # ── 按方向归因 ──
        by_direction = {
            "long_pnl": 0.0, "long_share": 0.0, "long_trades": 0,
            "short_pnl": 0.0, "short_share": 0.0, "short_trades": 0,
            "long_short_ratio": 0.0,
        }
        try:
            long_pnl_sum = sum(
                safe_float(c.get("long_pnl"), 0.0) for c in strategies.values()
            )
            short_pnl_sum = sum(
                safe_float(c.get("short_pnl"), 0.0) for c in strategies.values()
            )
            long_trades_sum = sum(
                safe_int(c.get("long_trades"), 0) for c in strategies.values()
            )
            short_trades_sum = sum(
                safe_int(c.get("short_trades"), 0) for c in strategies.values()
            )
            dir_abs = abs(long_pnl_sum) + abs(short_pnl_sum)
            by_direction = {
                "long_pnl": long_pnl_sum,
                "long_share": abs(long_pnl_sum) / dir_abs if dir_abs > 0 else 0.0,
                "long_trades": long_trades_sum,
                "short_pnl": short_pnl_sum,
                "short_share": abs(short_pnl_sum) / dir_abs if dir_abs > 0 else 0.0,
                "short_trades": short_trades_sum,
                "long_short_ratio": (
                    long_pnl_sum / short_pnl_sum if short_pnl_sum != 0 else 0.0
                ),
            }
        except Exception:
            logger.exception("[AGI-PnL] direction attribution failed")
            degraded_dimensions.append("by_direction")

        # ── 按退出原因归因 ──
        by_exit_reason = {
            "take_profit": {"pnl": 0.0, "count": 0, "share": 0.0},
            "stop_loss": {"pnl": 0.0, "count": 0, "share": 0.0},
            "realized": {"pnl": 0.0, "share": 0.0},
            "unrealized": {"pnl": 0.0, "share": 0.0},
        }
        try:
            tp_pnl = sum(
                safe_float(c.get("take_profit_pnl"), 0.0)
                for c in strategies.values()
            )
            sl_pnl = sum(
                safe_float(c.get("stop_loss_pnl"), 0.0)
                for c in strategies.values()
            )
            tp_count = sum(
                safe_int(c.get("take_profit_count"), 0)
                for c in strategies.values()
            )
            sl_count = sum(
                safe_int(c.get("stop_loss_count"), 0)
                for c in strategies.values()
            )
            realized = sum(
                safe_float(c.get("realized_pnl"), 0.0)
                for c in strategies.values()
            )
            unrealized = sum(
                safe_float(c.get("unrealized_pnl"), 0.0)
                for c in strategies.values()
            )
            exit_abs = abs(tp_pnl) + abs(sl_pnl) + 1e-9
            total_ru = abs(realized) + abs(unrealized) + 1e-9
            by_exit_reason = {
                "take_profit": {
                    "pnl": tp_pnl, "count": tp_count, "share": tp_pnl / exit_abs,
                },
                "stop_loss": {
                    "pnl": sl_pnl, "count": sl_count, "share": sl_pnl / exit_abs,
                },
                "realized": {"pnl": realized, "share": realized / total_ru},
                "unrealized": {"pnl": unrealized, "share": unrealized / total_ru},
            }
        except Exception:
            logger.exception("[AGI-PnL] exit-reason attribution failed")
            degraded_dimensions.append("by_exit_reason")

        quality = min(1.0, total_trades / self._projection_min_attribution_trades)

        return {
            "available": True,
            "total_pnl": total_pnl,
            "by_strategy": by_strategy,
            "by_regime": by_regime,
            "by_direction": by_direction,
            "by_exit_reason": by_exit_reason,
            "attribution_quality": quality,
            "dominant_strategy": dominant,
            "worst_strategy": worst,
            "timestamp": datetime.now().isoformat(),
            "degraded_dimensions": degraded_dimensions,
        }

    def _project_pnl(
        self,
        perception: Dict[str, Any],
        attribution: Dict[str, Any],
    ) -> Dict[str, Any]:
        """前瞻盈利推算：基于历史表现 + 趋势调整 + regime 调整估算未来 N 周期预期盈亏。

        - 按策略逐个推算后聚合为组合级推算。
        - fail-closed：缺数据/异常返回 available=false + correction_factor=1.0。
        """
        if not self._pnl_projection_enabled:
            return {"available": False, "horizon_cycles": self._projection_horizon,
                    "regime": "unknown", "correction_factor": 1.0,
                    "total": {}, "per_strategy": {},
                    "projection_cycle": self._cycle_count,
                    "timestamp": datetime.now().isoformat()}

        contrib = _coerce_dict(perception.get("contribution"))
        if not contrib.get("available"):
            return {"available": False, "horizon_cycles": self._projection_horizon,
                    "regime": "unknown", "correction_factor": self._correction_factor,
                    "total": {}, "per_strategy": {},
                    "projection_cycle": self._cycle_count,
                    "timestamp": datetime.now().isoformat()}

        raw_strategies = _coerce_dict(contrib.get("strategies"))
        strategies = {
            str(name): _coerce_dict(strategy)
            for name, strategy in raw_strategies.items()
            if isinstance(strategy, dict)
        }
        market_regime = _coerce_dict(perception.get("market_regime"))
        regime_str = str(market_regime.get("regime") or "unknown")
        regime_key = _canonical_projection_regime(regime_str)
        regime_conf = safe_float(market_regime.get("confidence"), 0.5)
        horizon = self._projection_horizon
        # regime 条件化准确度：优先用对应 regime 分桶的 correction，缺失时回退全局
        regime_sample_count = sum(
            1 for sample in self._projection_accuracy
            if isinstance(sample, dict)
            and _canonical_projection_regime(sample.get("regime")) == regime_key
        )
        regime_correction_updated_cycle = (
            self._correction_factor_by_regime_updated_cycle.get(regime_key)
        )
        if (
            regime_sample_count >= self._projection_min_accuracy_samples
            and regime_key in self._correction_factor_by_regime
            and regime_correction_updated_cycle is not None
            and 0 <= self._cycle_count - regime_correction_updated_cycle
            <= self._regime_correction_max_age_cycles
        ):
            correction = self._correction_factor_by_regime[regime_key]
        else:
            correction = self._correction_factor

        # regime 乘子（复用 regime_adaptive 的 trend/range 乘子）
        if regime_key in ("trend_up", "trend_down", "trending"):
            regime_mult = self._trend_profit_take_mult
        elif regime_key == "range_bound":
            regime_mult = self._range_profit_take_mult
        else:
            regime_mult = 1.0
        # confidence 低于阈值时按比例衰减
        if regime_conf < 0.8:
            regime_mult *= max(0.1, regime_conf)

        per_strategy: Dict[str, Any] = {}
        total_base_per_cycle = 0.0
        total_vol = 0.0
        # bear_case 场景乘数（与 _scenario 中 bear_case 一致）
        bear_mult = self._range_profit_take_mult / max(regime_mult, 0.01)

        for name, c in strategies.items():
            wr = safe_float(c.get("win_rate"), 0.0)
            aw = safe_float(c.get("avg_win"), 0.0)
            al = safe_float(c.get("avg_loss"), 0.0)
            trades = safe_int(c.get("total_trades"), 0)
            vol = safe_float(c.get("volatility"), 0.0)
            ppt = safe_float(c.get("pnl_per_trade"), 0.0)
            momentum = safe_float(c.get("trend_pnl_7d_vs_30d"), 0.0)

            # 基础期望：EV = win_rate × avg_win - (1-win_rate) × avg_loss
            if wr > 0 or aw > 0 or al > 0:
                base_expect = wr * aw - (1.0 - wr) * abs(al)
            elif ppt != 0:
                # 兜底：用 pnl_per_trade 作为期望值近似
                base_expect = ppt
            else:
                base_expect = 0.0

            # 趋势调整：从 history deque 取线性回归斜率
            trend_adj = 0.0
            ppt_hist = self._strategy_pnl_per_trade_history.get(str(name))
            if ppt_hist and len(ppt_hist) >= 2:
                slope = self._deque_linear_slope(ppt_hist)
                max_adj = self._projection_max_trend_adj_ratio * abs(base_expect + 1e-9)
                trend_adj = max(-max_adj, min(max_adj, slope * self._projection_trend_weight))
            # trend_pnl_7d_vs_30d 次级修正
            if momentum > 0:
                trend_adj += abs(trend_adj) * 0.1
            elif momentum < 0:
                trend_adj -= abs(trend_adj) * 0.1

            # regime 调整 + 准确度校正
            corrected = (base_expect + trend_adj) * regime_mult * correction

            # 推算置信度（样本量 + 波动率稳定性）
            conf = min(1.0, trades / 20.0) * (1.0 - min(0.5, vol / max(abs(base_expect), 1.0, 1e-9)))
            conf = max(0.0, min(1.0, conf))

            per_strategy[str(name)] = {
                "base_expect": base_expect,
                "trend_adjustment": trend_adj,
                "regime_mult": regime_mult,
                "corrected_expect": corrected,
                "horizon": corrected * horizon,
                "bear_case_horizon": corrected * horizon * bear_mult,
                "confidence": conf,
            }
            total_base_per_cycle += corrected
            total_vol += vol

        # 场景推演及 95% 置信区间（±1.96 × vol × √horizon）
        def _scenario(mult: float, conf_scale: float) -> Dict[str, Any]:
            per_cycle = total_base_per_cycle * mult
            sc_horizon = per_cycle * horizon
            sc_sigma = max(total_vol, abs(per_cycle) * 0.5) * (horizon ** 0.5) * conf_scale
            return {
                "per_cycle": per_cycle,
                "horizon": sc_horizon,
                "ci_lower": sc_horizon - 1.96 * sc_sigma,
                "ci_upper": sc_horizon + 1.96 * sc_sigma,
            }

        base_case = _scenario(1.0, 1.0)
        bear_case = _scenario(self._range_profit_take_mult / max(regime_mult, 0.01), 1.5)
        bull_case = _scenario(1.0 + 0.2, 0.8)

        return {
            "available": True,
            "horizon_cycles": horizon,
            "regime": regime_key,
            "correction_factor": correction,
            "total": {
                "base_case": base_case,
                "bear_case": bear_case,
                "bull_case": bull_case,
            },
            "per_strategy": per_strategy,
            "projection_cycle": self._cycle_count,
            "timestamp": datetime.now().isoformat(),
        }

    @staticmethod
    def _deque_linear_slope(d: deque) -> float:
        """对 deque 做简单线性回归斜率（用于趋势调整）。"""
        n = len(d)
        if n < 2:
            return 0.0
        vals = [safe_float(x, 0.0) for x in d]
        x_mean = (n - 1) / 2.0
        y_mean = sum(vals) / n
        num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(vals))
        den = sum((i - x_mean) ** 2 for i in range(n))
        return num / den if den != 0 else 0.0

    def _track_projection_accuracy(self, perception: Dict[str, Any]) -> Dict[str, Any]:
        """追踪历史推算准确度（实际 vs 上次推算），用于校准未来推算。

        自适应学习闭环：每周期对比「上一周期推算的 per_cycle」与「本周期实际 cycle_pnl」，
        维护 bias_ratio 滚动窗口（deque），用中位数作为下一周期推算的 correction_factor。
        """
        if self._last_projection is None:
            return {"last_bias_ratio": 1.0, "median_bias_ratio": 1.0,
                    "correction_factor": 1.0, "sample_count": 0,
                    "low_confidence": True, "history": []}

        last_total = (self._last_projection.get("total") or {}).get("base_case") or {}
        projected_per_cycle = safe_float(last_total.get("per_cycle"), 0.0)

        # 从 _decision_memory 取本周期实际 cycle_pnl
        contrib = _coerce_dict(perception.get("contribution"))
        cur_total_pnl = safe_float(contrib.get("total_pnl"), 0.0)
        if (self._last_decision_total_pnl is not None
                and self._decision_quality_use_cycle_pnl):
            actual_cycle_pnl = cur_total_pnl - self._last_decision_total_pnl
        else:
            actual_cycle_pnl = 0.0

        # bias_ratio = actual / projected（除零时 1.0）
        if abs(projected_per_cycle) > 1e-9:
            bias = actual_cycle_pnl / projected_per_cycle
        else:
            bias = 1.0
        # 安全清洗
        if not (math.isfinite(bias) and abs(bias) < 1000):
            bias = 1.0

        # The sample scores the prior forecast, so classify it by the forecast's
        # regime rather than the regime observed after that forecast.
        forecast_regime = _canonical_projection_regime(
            self._last_projection.get("regime")
        )
        self._projection_accuracy.append({
            "cycle": self._cycle_count,
            "projected": projected_per_cycle,
            "actual": actual_cycle_pnl,
            "bias_ratio": bias,
            "regime": forecast_regime,
        })

        samples = len(self._projection_accuracy)
        if samples < self._projection_min_accuracy_samples:
            correction = 1.0
            low_conf = True
            median_bias = 1.0
        else:
            biases = sorted(safe_float(x.get("bias_ratio"), 1.0)
                           for x in self._projection_accuracy)
            mid = len(biases) // 2
            median_bias = biases[mid]
            correction = max(self._projection_min_correction,
                             min(self._projection_max_correction, median_bias))
            low_conf = False

        self._correction_factor = correction

        # regime 条件化准确度：按 regime 分桶维护独立 correction
        # 不同 regime 下推算偏差不同（趋势市推算更准、震荡市偏差大），分桶校正更精确
        cur_regime = forecast_regime
        regime_samples = [x for x in self._projection_accuracy
                          if str(x.get("regime")) == cur_regime]
        if len(regime_samples) >= self._projection_min_accuracy_samples:
            r_biases = sorted(safe_float(x.get("bias_ratio"), 1.0)
                              for x in regime_samples)
            r_mid = len(r_biases) // 2
            r_median = r_biases[r_mid]
            r_correction = max(self._projection_min_correction,
                               min(self._projection_max_correction, r_median))
            self._correction_factor_by_regime[cur_regime] = r_correction
            self._correction_factor_by_regime_updated_cycle[cur_regime] = self._cycle_count

        return {
            "last_bias_ratio": bias,
            "median_bias_ratio": median_bias,
            "correction_factor": correction,
            "sample_count": samples,
            "low_confidence": low_conf,
            "history": [safe_float(x.get("bias_ratio"), 1.0)
                       for x in self._projection_accuracy],
        }

    def _projection_boost_scale(self, expect: float, confidence: float) -> float:
        """推算缩放因子 [min_mult, max_mult]：正期望高置信→加大，负→缩小。

        - 与 _offensive_feedback_multiplier 同构：不生成新动作类型，仅缩放 boost_step。
        - fail-closed：expect=None 或 disabled 时返回 1.0（不影响原有 boost）。
        """
        if not self._pnl_projection_enabled or expect is None:
            return 1.0
        sign = 1.0 if safe_float(expect, 0.0) >= 0 else -1.0
        mag = min(1.0, abs(safe_float(expect, 0.0)) / self._projection_unit_pnl)
        conf = max(0.0, min(1.0, safe_float(confidence, 0.5)))
        scale = 1.0 + sign * mag * conf * self._projection_adaptation_span
        return max(self._projection_min_mult,
                   min(self._projection_max_mult, scale))

    def _delta_pnl_trend_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """边际盈亏趋势外推动作：连续为负 → param_adjust 提前降杠杆（事前风控）。

        - 与十子守卫（health/win_rate/sharpe/profit_factor/max_drawdown/drawdown_duration/capital_return/
          volatility/pnl_momentum/pnl_per_trade）互补：本方法看「边际盈亏」——delta_pnl 连续为负
          意味着策略最近持续失血（即使累计 total_pnl 仍为正，近期也在亏），是「从盈利转亏损」的
          周期级先行指标。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._delta_pnl_trend_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "delta_pnl_deteriorating":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._delta_pnl_trend_leverage,
                "reason": f"delta_pnl_trend_guard: {alert.get('message', '')}",
            })
        return actions

    def _delta_pnl_guard_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """单周期亏损急跌动作：delta_pnl_spike → param_adjust 提前降杠杆（事前风控）。

        - 与 _delta_pnl_trend_actions（连续 N 周期为负，趋势型）区分：本方法看「绝对阈值」——单周期
          大额急跌（|delta_pnl|/equity ≥ loss_threshold）是突发大幅亏损的硬信号，即使未连续为负也应
          提前收敛敞口。构成边际盈亏维度「阈值 + 趋势」双层。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._delta_pnl_guard_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "delta_pnl_spike":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._delta_pnl_guard_leverage,
                "reason": f"delta_pnl_guard: {alert.get('message', '')}",
            })
        return actions

    def _resonance_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """多守卫共振收敛动作：multi_guard_resonance → param_adjust 升级降杠杆（事前风控）。

        - 与六子守卫（单维度趋势衰退 → 降杠杆 1.0）区分：本方法响应「多维度同时衰退」，
          降杠杆力度更大（resonance_leverage 默认0.5），因为多维度共振衰退意味着策略在
          多个独立维度同时劣化，风险远高于单维度衰退，需要更大力度的敞口收敛。
        - 复用 param_adjust 低风险自动执行通道（scheduler 热重载杠杆），不直接下单。
        """
        if not self._resonance_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            if alert.get("type") != "multi_guard_resonance":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            seen.add(strategy)
            actions.append({
                "type": "param_adjust",
                "strategy": strategy,
                "param": "leverage",
                "value": self._resonance_leverage,
                "reason": f"resonance_guard: {alert.get('message', '')}",
            })
        return actions

    def _self_heal_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """风险自愈闭环：冻结策略观察期/缩量试探 → self_heal 事件（通知型）。

        将「自动解冻状态机」的进展（观察期等待 / 缩量试探中）显式化为动作，
        使自愈进程可被决策溯源历史与 Dashboard 追踪。该动作为通知型，不触碰资金。
        """
        actions: List[Dict[str, Any]] = []
        for alert in alerts or []:
            atype = alert.get("type")
            strategy = alert.get("strategy")
            if atype in ("strategy_probing", "strategy_frozen_observing") and strategy:
                actions.append({
                    "type": "self_heal",
                    "strategy": strategy,
                    "reason": f"self_heal:{atype}: {alert.get('message', '')}",
                })
        return actions

    def _offensive_allocation_actions(self, decision: Dict[str, Any],
                                      alerts: List[Dict[str, Any]],
                                      projection: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """自主进攻性资金分配：offensive_opportunity → reallocate increase 加仓 A/B 策略。

        - 目标权重 = 当前目标权重 + boost_step，封顶 max_target（有界进攻，不梭哈）。
        - 进攻观察期冷却：同一策略在 min_interval_cycles 个周期内只加仓一次，
          防止 AGI 连续追高加仓（对应用户「消除追涨杀跌」诉求）。
        - 复用 reallocate 高风险动作：autonomous 模式受全局 Kill Switch 约束，非自主排队。
        """
        if not self._offensive_enabled:
            return []
        # 市场状态突变进攻冷却：突变后 cooldown_cycles 个周期内暂停进攻，
        # 等新状态稳定、趋势重新确认后再加仓（对应用户「趋势确认后才开仓」诉求）
        if (
            self._regime_shift_guard_enabled
            and self._last_regime_shift_cycle is not None
            and self._cycle_count - self._last_regime_shift_cycle < self._regime_shift_cooldown_cycles
        ):
            return []
        # 交易成本感知：手续费侵蚀过高 → 抑制进攻性加仓，降低交易频率
        if self._cost_awareness_enabled and any(
            a.get("type") == "high_trading_cost" for a in (alerts or [])
        ):
            return []
        # 追涨抑制：市场过热（连续上涨过多）→ 暂停进攻性加仓，不追高
        if self._momentum_guard_enabled and any(
            a.get("type") == "market_overheated" for a in (alerts or [])
        ):
            return []
        # 资金利用率过高收敛：敞口过高 → 暂停进攻性加仓，避免进一步放大杠杆
        if self._utilization_guard_enabled and any(
            a.get("type") == "high_capital_utilization" for a in (alerts or [])
        ):
            return []
        # 资金利用率趋势外推：杠杆/敞口持续放大 → 暂停进攻性加仓（事前，与阈值型对称）
        if self._utilization_trend_enabled and any(
            a.get("type") == "utilization_rising" for a in (alerts or [])
        ):
            return []
        # 回撤加速预警：回撤连续加深 → 暂停进攻，防止在趋势恶化时继续放大敞口
        if self._drawdown_accel_enabled and any(
            a.get("type") == "drawdown_accelerating" for a in (alerts or [])
        ):
            return []
        # 连续下跌收敛：权益持续失血 → 暂停进攻（不抄底），等止跌企稳再行动
        if self._downside_momentum_enabled and any(
            a.get("type") == "market_panicking" for a in (alerts or [])
        ):
            return []
        # 组合级尾部风险：组合最深回撤超阈值 → 暂停进攻性加仓，避免放大尾部暴露
        if self._tail_risk_enabled and any(
            a.get("type") == "high_tail_risk" for a in (alerts or [])
        ):
            return []
        # 组合级协同度：协同度低于阈值 → 暂停进攻性加仓（策略相关且一起亏）
        if self._synergy_enabled and any(
            a.get("type") == "low_synergy" for a in (alerts or [])
        ):
            return []
        # 组合级协同度趋势：协同度连续下降 → 暂停进攻性加仓（协同质量劣化）
        if self._synergy_trend_enabled and any(
            a.get("type") == "synergy_deteriorating" for a in (alerts or [])
        ):
            return []
        # 组合级资金效率趋势：资金效率连续下降 → 暂停进攻性加仓（资金利用效率劣化）
        if self._portfolio_efficiency_trend_enabled and any(
            a.get("type") == "efficiency_deteriorating" for a in (alerts or [])
        ):
            return []
        # 组合级风险共振：多维度组合风险同时恶化 → 暂停进攻性加仓（组合级全局收敛）
        if self._portfolio_resonance_enabled and any(
            a.get("type") == "portfolio_risk_resonance" for a in (alerts or [])
        ):
            return []
        # 组合级健康度阈值：组合整体健康度跌破健康线 → 暂停进攻性加仓（组合级健康度差）
        if self._portfolio_health_enabled and any(
            a.get("type") == "portfolio_health_low" for a in (alerts or [])
        ):
            return []
        # 组合级健康度趋势：组合整体健康度连续下降 → 暂停进攻性加仓（温水煮青蛙式衰退）
        if self._portfolio_health_trend_enabled and any(
            a.get("type") == "portfolio_health_deteriorating" for a in (alerts or [])
        ):
            return []
        # 组合级集中度趋势：集中度连续上升 → 暂停进攻性加仓（分散度被侵蚀）
        if self._portfolio_concentration_trend_enabled and any(
            a.get("type") == "portfolio_concentration_rising" for a in (alerts or [])
        ):
            return []
        # 组合级相关性趋势：相关性连续上升 → 暂停进攻性加仓（联动风险加剧）
        if self._portfolio_correlation_trend_enabled and any(
            a.get("type") == "portfolio_correlation_rising" for a in (alerts or [])
        ):
            return []
        # 组合级尾部风险趋势：尾部风险连续加深 → 暂停进攻性加仓（尾部暴露扩大）
        if self._portfolio_tail_risk_trend_enabled and any(
            a.get("type") == "tail_risk_rising" for a in (alerts or [])
        ):
            return []
        # 组合级盈利集中度趋势：盈利来源集中度连续上升 → 暂停进攻性加仓（单一盈利支柱风险加剧）
        if self._profit_concentration_trend_enabled and any(
            a.get("type") == "profit_concentration_rising" for a in (alerts or [])
        ):
            return []
        # 组合多空净敞口失衡：方向性失衡 → 暂停进攻性加仓（避免放大单边敞口）
        if self._net_exposure_enabled and any(
            a.get("type") == "directional_imbalance" for a in (alerts or [])
        ):
            return []
        # 账户级杠杆过高：杠杆逼近上限 → 暂停进攻性加仓（避免进一步放大杠杆、逼近强平）
        if self._account_leverage_enabled and any(
            a.get("type") == "high_account_leverage" for a in (alerts or [])
        ):
            return []
        # 挂单保证金过高：隐性挂单敞口过高 → 暂停进攻性加仓（避免挂单成交后敞口突增）
        if self._pending_margin_enabled and any(
            a.get("type") == "high_pending_margin" for a in (alerts or [])
        ):
            return []
        # 决策记忆质量低：近期决策盈利周期占比过低 → 暂停进攻（避免决策持续失误时继续进攻）
        # 死锁自愈：持续触发 recovery_after_cycles 周期后，不直接 return []，而是设置
        # recovery_boost_scale 缩放 boost_step，允许最小试探开单，给系统恢复出口。
        _dq_recovery_scale: Optional[float] = None
        if self._decision_quality_enabled:
            dq_alert = next(
                (a for a in (alerts or []) if a.get("type") == "decision_quality_low"),
                None,
            )
            if dq_alert is not None:
                if dq_alert.get("recovery_mode"):
                    # 已过冷却期 → 允许最小试探开单（缩放 boost_step）
                    _dq_recovery_scale = self._decision_quality_recovery_scale
                else:
                    return []
        # 组合压力测试失败：尾部场景压力损失超预算 → 暂停进攻（避免放大敞口）
        if self._stress_guard_enabled and any(
            a.get("type") == "stress_test_failed" for a in (alerts or [])
        ):
            return []
        # 进攻与账户可用保证金联动：可用保证金占总资金比例不足 → 暂停进攻加仓，
        # 避免保证金不足时强行加仓被拒/被动减仓（无法度量时放行，向后兼容）
        if self._offensive_margin_check_enabled:
            avail_ratio = self._available_margin_ratio()
            if avail_ratio is not None and avail_ratio < self._offensive_min_available_margin_ratio:
                return []
        # 胜率趋势恶化的策略集合（per-strategy gate）
        # 仅抑制这些策略的进攻加仓，不影响其他健康策略——胜率下降是策略级信号而非市场级
        wr_deteriorating: set = set()
        if self._win_rate_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "win_rate_deteriorating" and _a.get("strategy"):
                    wr_deteriorating.add(str(_a.get("strategy")))
        # 胜率绝对值过低的策略集合（per-strategy gate，胜率阈值维度）
        # 胜率低于阈值（长期低胜率）→ 不继续追高加仓放大低胜率敞口，先降杠杆收敛
        low_win_rate: set = set()
        if self._win_rate_guard_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "low_win_rate" and _a.get("strategy"):
                    low_win_rate.add(str(_a.get("strategy")))
        # 夏普趋势恶化的策略集合（per-strategy gate，与胜率独立判定）
        # 夏普下降意味着单位风险收益递减，该策略不应继续追高加仓
        sr_deteriorating: set = set()
        if self._sharpe_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "sharpe_deteriorating" and _a.get("strategy"):
                    sr_deteriorating.add(str(_a.get("strategy")))
        # 连续亏损策略集合（per-strategy gate，与胜率/夏普独立判定）
        # 连续亏损达阈值是亏损侧硬信号——该策略不应继续追高加仓放大亏损
        cl_strategies: set = set()
        if self._consecutive_losses_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "strategy_losing_streak" and _a.get("strategy"):
                    cl_strategies.add(str(_a.get("strategy")))
        # 止损率过高的策略集合（per-strategy gate，与连续亏损独立判定）
        # 止损占比过高意味着入场时机差（追高开多/追低开空）——该策略不应继续追高加仓
        high_slr_strategies: set = set()
        if self._stop_loss_frequency_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "high_stop_loss_rate" and _a.get("strategy"):
                    high_slr_strategies.add(str(_a.get("strategy")))
        # 止盈止损比过低的策略集合（per-strategy gate，离场质量维度）
        # 止盈相对止损过少意味着善止损不善止盈（赚的时候不落袋）——该策略不应继续追高加仓
        low_tp_strategies: set = set()
        if self._take_profit_ratio_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "low_take_profit_ratio" and _a.get("strategy"):
                    low_tp_strategies.add(str(_a.get("strategy")))
        # 盈亏比过低的策略集合（per-strategy gate，单笔盈亏结构维度）
        # 平均盈利相对平均亏损过小（小赚大亏）→ 该策略不应继续追高加仓
        low_wlr_strategies: set = set()
        if self._win_loss_ratio_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "low_win_loss_ratio" and _a.get("strategy"):
                    low_wlr_strategies.add(str(_a.get("strategy")))
        # 止盈止损盈亏金额比过低的策略集合（per-strategy gate，累计金额仓位结构维度）
        # 止盈累计盈亏相对止损累计亏损过少（赚的时候轻仓、亏的时候重仓）→ 该策略不应继续追高加仓
        low_tppr_strategies: set = set()
        if self._take_profit_pnl_ratio_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "low_take_profit_pnl_ratio" and _a.get("strategy"):
                    low_tppr_strategies.add(str(_a.get("strategy")))
        # 盈亏比恶化的策略集合（per-strategy gate，与胜率/夏普/连续亏损独立判定）
        # profit_factor 连续下降意味着赢亏幅度比收窄——该策略不应继续追高加仓
        pf_deteriorating: set = set()
        if self._profit_factor_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "profit_factor_deteriorating" and _a.get("strategy"):
                    pf_deteriorating.add(str(_a.get("strategy")))
        # 最大回撤加深的策略集合（per-strategy gate，与前四子独立判定）
        # max_drawdown 连续上升意味着风险深度扩大——该策略不应继续追高加仓放大风险
        mdd_deteriorating: set = set()
        if self._max_drawdown_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "max_drawdown_deteriorating" and _a.get("strategy"):
                    mdd_deteriorating.add(str(_a.get("strategy")))
        # 最大回撤超阈值的策略集合（per-strategy gate，回撤深度阈值维度）
        # max_drawdown 已超阈值（深度回撤）→ 不继续追高加仓放大风险敞口，先降杠杆收敛
        high_mdd: set = set()
        if self._max_drawdown_guard_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "high_max_drawdown" and _a.get("strategy"):
                    high_mdd.add(str(_a.get("strategy")))
        # 回撤持续时间拉长的策略集合（per-strategy gate，与回撤深度独立判定）
        # max_drawdown_duration_hours 连续上升意味着恢复能力下降——该策略不应继续追高加仓
        ddh_deteriorating: set = set()
        if self._drawdown_duration_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "drawdown_duration_deteriorating" and _a.get("strategy"):
                    ddh_deteriorating.add(str(_a.get("strategy")))
        # 资本回报率下降的策略集合（per-strategy gate，与前五子独立判定）
        # pnl_per_capital_pct 连续下降意味着资本配置效率恶化——该策略不应继续追高加仓
        cr_deteriorating: set = set()
        if self._capital_return_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "capital_return_deteriorating" and _a.get("strategy"):
                    cr_deteriorating.add(str(_a.get("strategy")))
        # 资本回报率过低的策略集合（per-strategy gate，资本效率阈值维度）
        # 单位资本产出低于绝对阈值 → 该策略不应继续追高加仓（阈值层，与 cr_deteriorating 趋势层互补）
        low_cr_strategies: set = set()
        if self._capital_return_guard_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "capital_return_low" and _a.get("strategy"):
                    low_cr_strategies.add(str(_a.get("strategy")))
        # 波动率上升的策略集合（per-strategy gate，与前六子独立判定）
        # volatility 连续上升意味着单笔盈亏离散度扩大——该策略不应继续追高加仓放大不确定性
        vol_deteriorating: set = set()
        if self._volatility_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "volatility_deteriorating" and _a.get("strategy"):
                    vol_deteriorating.add(str(_a.get("strategy")))
        # 多守卫共振的策略集合（per-strategy gate，与六子独立判定）
        # 多维度同时衰退是最严重的衰退信号——该策略不应继续追高加仓
        resonance_strategies: set = set()
        if self._resonance_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "multi_guard_resonance" and _a.get("strategy"):
                    resonance_strategies.add(str(_a.get("strategy")))
        # 盈利集中度的主导策略集合（per-strategy gate，组合级维度）
        # 组合盈利过度依赖单一策略 → 不把资金进一步压到该支柱上，避免过度集中
        pc_dominant: set = set()
        if self._profit_concentration_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "profit_concentration" and _a.get("strategy"):
                    pc_dominant.add(str(_a.get("strategy")))
        # 浮盈占比过高的策略集合（per-strategy gate，收益侧盈利质量维度）
        # 账面盈利主要靠浮盈支撑 → 不继续加仓放大浮盈暴露，先锁定浮盈
        upr_concentrated: set = set()
        if self._unrealized_profit_ratio_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "unrealized_profit_concentration" and _a.get("strategy"):
                    upr_concentrated.add(str(_a.get("strategy")))
        # 浮盈占比持续上升的策略集合（per-strategy gate，收益侧盈利质量趋势维度）
        # 账面盈利越来越依赖浮盈 → 不继续追高加仓放大浮盈暴露，先锁定浮盈
        upr_rising: set = set()
        if self._unrealized_profit_ratio_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "unrealized_profit_ratio_rising" and _a.get("strategy"):
                    upr_rising.add(str(_a.get("strategy")))
        # 浮亏加深的策略集合（per-strategy gate，亏损侧趋势维度）
        # 浮亏连续加深 → 不继续追高加仓放大浮亏暴露，先止损收敛
        uld_deteriorating: set = set()
        if self._unrealized_loss_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "unrealized_loss_deteriorating" and _a.get("strategy"):
                    uld_deteriorating.add(str(_a.get("strategy")))
        # 手续费侵蚀过高的策略集合（per-strategy gate，交易频率维度）
        # 单策略交易过频消耗资金 → 不继续追高加仓放大换手，先降频收敛
        high_fee_strategies: set = set()
        if self._strategy_fee_ratio_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "high_strategy_fee" and _a.get("strategy"):
                    high_fee_strategies.add(str(_a.get("strategy")))
        # 手续费率持续上升的策略集合（per-strategy gate，交易频率趋势维度）
        # 手续费率连续上升 → 不继续追高加仓放大换手，先降频收敛
        high_fee_rising: set = set()
        if self._strategy_fee_ratio_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "strategy_fee_ratio_rising" and _a.get("strategy"):
                    high_fee_rising.add(str(_a.get("strategy")))
        # 资金费侵蚀过高的策略集合（per-strategy gate，持仓时间成本维度）
        # 单策略持仓过久被 funding 持续侵蚀 → 不继续追高加仓放大持仓敞口，先收敛持仓
        high_funding_strategies: set = set()
        if self._funding_cost_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "high_funding_cost" and _a.get("strategy"):
                    high_funding_strategies.add(str(_a.get("strategy")))
        # 资金费率持续上升的策略集合（per-strategy gate，持仓时间成本趋势维度）
        # 资金费率连续上升 → 不继续追高加仓放大持仓敞口，先收敛持仓（持仓时间成本事前预警）
        funding_rising: set = set()
        if self._funding_cost_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "funding_cost_rising" and _a.get("strategy"):
                    funding_rising.add(str(_a.get("strategy")))
        # 执行质量成本过高的策略集合（per-strategy gate，执行成本维度）
        # 单策略滑点/点差侵蚀过高（下单执行质量差）→ 不继续追高加仓放大敞口，先改善执行质量
        high_exec_cost_strategies: set = set()
        if self._execution_cost_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "high_execution_cost" and _a.get("strategy"):
                    high_exec_cost_strategies.add(str(_a.get("strategy")))
        # 执行成本率持续上升的策略集合（per-strategy gate，执行质量趋势维度）
        # 执行成本率连续上升 → 不继续追高加仓放大滑点敞口，先收敛（执行质量事前预警）
        exec_cost_rising: set = set()
        if self._execution_cost_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "execution_cost_rising" and _a.get("strategy"):
                    exec_cost_rising.add(str(_a.get("strategy")))
        # 多空方向失衡的策略集合（per-strategy gate，方向判断质量维度）
        # 单方向持续逆势亏损 → 不继续追高加仓放大逆势方向敞口，先降杠杆收敛
        imbalanced_strategies: set = set()
        if self._long_short_imbalance_enabled:
            for _a in (alerts or []):
                if _a.get("type") in ("long_side_losing", "short_side_losing") and _a.get("strategy"):
                    imbalanced_strategies.add(str(_a.get("strategy")))
        # 方向偏好失衡（单边笔数占比过高）→ 不继续追高加仓放大单边敞口，先降杠杆收敛
        direction_bias_strategies: set = set()
        if self._direction_bias_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "direction_bias" and _a.get("strategy"):
                    direction_bias_strategies.add(str(_a.get("strategy")))
        # PnL 动量趋势恶化的策略集合（per-strategy gate，收益趋势维度）
        # 近期（7日）PnL 相对中期（30日）持续收敛 → 盈利动能衰减 → 不继续追高加仓放大敞口
        pnl_momentum_deteriorating: set = set()
        if self._pnl_momentum_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "pnl_momentum_deteriorating" and _a.get("strategy"):
                    pnl_momentum_deteriorating.add(str(_a.get("strategy")))
        # PnL 动量衰竭的策略集合（per-strategy gate，增长动能衰竭维度）
        # 近期（7日）动量相对中期（30日）衰减到阈值以下 → 不继续追高加仓放大动能衰竭敞口
        pnl_momentum_faded_strategies: set = set()
        if self._pnl_momentum_guard_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "pnl_momentum_faded" and _a.get("strategy"):
                    pnl_momentum_faded_strategies.add(str(_a.get("strategy")))
        # 单笔期望值持续下降的策略集合（per-strategy gate，交易质量维度）
        # pnl_per_trade 连续下降意味着每笔交易价值持续萎缩 → 该策略不应继续追高加仓
        pptr_deteriorating: set = set()
        if self._pnl_per_trade_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "pnl_per_trade_deteriorating" and _a.get("strategy"):
                    pptr_deteriorating.add(str(_a.get("strategy")))
        # 单笔期望值低于阈值的策略集合（per-strategy gate，单笔期望值阈值维度）
        # pnl_per_trade 低于阈值（每笔无正期望）→ 不继续追高加仓放大低质量敞口，先降杠杆收敛
        low_ppt: set = set()
        if self._pnl_per_trade_guard_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "low_pnl_per_trade" and _a.get("strategy"):
                    low_ppt.add(str(_a.get("strategy")))
        # 推算预期为负的策略集合（per-strategy gate，前瞻期望维度）
        # 该策略未来 horizon 期望亏损 → 不追高加仓放大亏损敞口
        negative_projection_strategies: set = set()
        # bear_case fail-closed 策略集合（per-strategy gate，下行风险维度）
        # 该策略悲观场景 horizon 亏损超过 fail_closed_loss_threshold × equity → 不追高加仓放大下行敞口
        bear_case_fail_closed_strategies: set = set()
        if self._pnl_projection_enabled and projection and projection.get("available"):
            # 权益基准：从 decision 的 allocation_plan 间接取 total_equity
            _eq_base = 0.0
            _ap = decision.get("allocation_plan") if isinstance(decision, dict) else None
            if isinstance(_ap, dict):
                _eq_base = safe_float(_ap.get("total_equity"), 0.0)
            _bear_threshold_abs = -abs(
                safe_float(self._projection_fail_closed_loss, 0.05)
            ) * max(_eq_base, 0.0)
            for _n, _sp in (projection.get("per_strategy") or {}).items():
                if safe_float(_sp.get("corrected_expect"), 0.0) < -self._offensive_projection_loss_threshold:
                    negative_projection_strategies.add(str(_n))
                # bear_case horizon 低于阈值 → 加入 fail-closed gate
                # _eq_base<=0 时阈值退化为 0，仅 bear_case_horizon<0 即触发（安全降级）
                if safe_float(_sp.get("bear_case_horizon"), 0.0) < _bear_threshold_abs:
                    bear_case_fail_closed_strategies.add(str(_n))
        # 边际盈亏持续为负的策略集合（per-strategy gate，持续失血维度）
        # delta_pnl 连续为负意味着策略最近持续失血 → 该策略不应继续追高加仓
        dp_deteriorating: set = set()
        if self._delta_pnl_trend_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "delta_pnl_deteriorating" and _a.get("strategy"):
                    dp_deteriorating.add(str(_a.get("strategy")))
        # 单周期亏损急跌的策略集合（per-strategy gate，急跌尖峰维度）
        # delta_pnl 单周期大额急跌（突发大幅亏损）→ 该策略不应继续追高加仓放大急跌敞口
        delta_pnl_spike_strategies: set = set()
        if self._delta_pnl_guard_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "delta_pnl_spike" and _a.get("strategy"):
                    delta_pnl_spike_strategies.add(str(_a.get("strategy")))
        # 风险调整后贡献过低的策略集合（per-strategy gate，风险收益比维度）
        # 盈利相对回撤过少（小赚大扛）→ 该策略不应继续追高加仓放大风险
        low_rac_strategies: set = set()
        if self._risk_adjusted_contribution_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "low_risk_adjusted_contribution" and _a.get("strategy"):
                    low_rac_strategies.add(str(_a.get("strategy")))
        # 闲置策略集合（per-strategy gate，资金闲置维度）
        # 曾活跃但长时间未交易 → 该策略不应继续追高加仓，先回收闲置资金
        stale_strategies: set = set()
        if self._strategy_staleness_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "strategy_stale" and _a.get("strategy"):
                    stale_strategies.add(str(_a.get("strategy")))
        # 健康度骤降的策略集合（per-strategy gate，健康恶化急信号维度）
        # 健康度单周期暴跌 → 该策略不应继续追高加仓，先降杠杆收敛
        health_crash_strategies: set = set()
        if self._health_crash_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "health_crash" and _a.get("strategy"):
                    health_crash_strategies.add(str(_a.get("strategy")))
        # 波动率超预算的策略集合（per-strategy gate，波动率绝对阈值维度）
        # 单笔盈亏标准差超预算 → 该策略不应继续追高加仓放大不确定性
        vol_budget_strategies: set = set()
        if self._volatility_budget_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "volatility_over_budget" and _a.get("strategy"):
                    vol_budget_strategies.add(str(_a.get("strategy")))
        # 负夏普策略集合（per-strategy gate，风险调整后负收益维度）
        # 夏普为负（风险调整后负收益）→ 该策略不应继续追高加仓放大风险敞口
        negative_sharpe_strategies: set = set()
        if self._sharpe_ratio_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "negative_sharpe" and _a.get("strategy"):
                    negative_sharpe_strategies.add(str(_a.get("strategy")))
        # 盈亏比过低策略集合（per-strategy gate，已实现净亏损维度）
        # 盈亏比<1（总亏损超过总盈利）→ 该策略不应继续追高加仓放大亏损敞口
        low_profit_factor_strategies: set = set()
        if self._profit_factor_guard_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "low_profit_factor" and _a.get("strategy"):
                    low_profit_factor_strategies.add(str(_a.get("strategy")))
        # 回撤持续过久策略集合（per-strategy gate，资金时间价值维度）
        # 最长回撤持续过久（恢复能力差）→ 该策略不应继续追高加仓放大被套牢资金
        long_drawdown_duration_strategies: set = set()
        if self._drawdown_duration_guard_enabled:
            for _a in (alerts or []):
                if _a.get("type") == "long_drawdown_duration" and _a.get("strategy"):
                    long_drawdown_duration_strategies.add(str(_a.get("strategy")))
        actions: List[Dict[str, Any]] = []
        # ── 进攻归因周期过期（TTL）预清理 ──
        # 归因基线超过 attribution_ttl_cycles 未触发止盈/止损则自动清除，
        # 避免陈旧 entry_pnl 长期阻塞重新进攻（TTL=0 表示永不过期，向后兼容）。
        if self._offensive_attribution_ttl_cycles > 0 and self._offensive_attribution_cycle:
            for _name in list(self._offensive_attribution_cycle.keys()):
                _set_cycle = self._offensive_attribution_cycle.get(_name)
                if _set_cycle is not None and self._cycle_count - _set_cycle > self._offensive_attribution_ttl_cycles:
                    self._offensive_attribution.pop(_name, None)
                    self._offensive_attribution_cycle.pop(_name, None)
        # ── 进攻止盈回落预扫描 ──
        # 上次进攻加仓后策略累计盈亏超过阈值 → 落袋为安，减仓回落，
        # 闭环「进攻加仓→盈利达标→自动回落」，避免进攻敞口永远不收缩。
        profit_taken: set = set()
        if self._offensive_profit_take_enabled and self._offensive_attribution:
            for pt_name, entry_pnl in list(self._offensive_attribution.items()):
                pt_metrics = (decision.get("strategy_metrics") or {}).get(pt_name) or {}
                pt_current_pnl = safe_float(pt_metrics.get("total_pnl"), 0.0)
                pt_gain = pt_current_pnl - entry_pnl
                if pt_gain >= self._offensive_profit_take_pnl_threshold:
                    pt_base = self._base_weight_for(decision, pt_name)
                    # 阶梯式减仓：超出阈值越多，减仓幅度越大（多赚多减），封顶 revert_max_multiplier
                    pt_over_ratio = pt_gain / max(0.01, self._offensive_profit_take_pnl_threshold)
                    pt_mult = min(self._offensive_profit_take_revert_max_mult, pt_over_ratio)
                    pt_revert = self._offensive_profit_take_revert_step * pt_mult
                    pt_target = max(0.0, pt_base - pt_revert)
                    if pt_target < pt_base:
                        actions.append({
                            "type": "reallocate",
                            "strategy": pt_name,
                            "action": "decrease",
                            "target_allocation": pt_target,
                            "reason": (
                                f"offensive_allocation: 止盈回落 策略{pt_name}"
                                f"盈利{pt_gain:.2f}≥阈值{self._offensive_profit_take_pnl_threshold:.2f}"
                                f"，减仓回落到{pt_target:.1%}"
                            ),
                        })
                    # 无论是否生成动作，止盈后清除该策略的进攻归因（不再追踪已了结的进攻）
                    # 强化学习正反馈：止盈=上次进攻决策正确→EMA 记 +1，下次加大 boost
                    self._update_offensive_feedback(pt_name, 1.0)
                    del self._offensive_attribution[pt_name]
                    self._offensive_attribution_cycle.pop(pt_name, None)
                    # 止盈了结 → 连续进攻计数归零（重新开始累积）
                    self._offensive_consecutive.pop(pt_name, None)
                    # 记录止盈冷却周期：cooldown_cycles 内禁止重新进攻该策略
                    self._offensive_profit_take_cooldown[pt_name] = self._cycle_count
                    profit_taken.add(pt_name)
        # ── 进攻止损回落预扫描 ──
        # 上次进攻加仓后策略累计亏损超过阈值 → 止损减仓回落，
        # 与止盈回落对称，闭环进攻风控，避免亏损敞口持续放大。
        if self._offensive_stop_loss_enabled and self._offensive_attribution:
            for sl_name, sl_entry_pnl in list(self._offensive_attribution.items()):
                if sl_name in profit_taken:
                    continue
                sl_metrics = (decision.get("strategy_metrics") or {}).get(sl_name) or {}
                sl_current_pnl = safe_float(sl_metrics.get("total_pnl"), 0.0)
                sl_loss = sl_entry_pnl - sl_current_pnl  # 正数 = 亏损量
                if sl_loss >= self._offensive_stop_loss_pnl_threshold:
                    sl_base = self._base_weight_for(decision, sl_name)
                    # 阶梯式减仓：超出阈值越多，减仓幅度越大（多亏多减、快速止血），封顶 revert_max_multiplier
                    sl_over_ratio = sl_loss / max(0.01, self._offensive_stop_loss_pnl_threshold)
                    sl_mult = min(self._offensive_stop_loss_revert_max_mult, sl_over_ratio)
                    sl_revert = self._offensive_stop_loss_revert_step * sl_mult
                    sl_target = max(0.0, sl_base - sl_revert)
                    if sl_target < sl_base:
                        actions.append({
                            "type": "reallocate",
                            "strategy": sl_name,
                            "action": "decrease",
                            "target_allocation": sl_target,
                            "reason": (
                                f"offensive_allocation: 止损回落 策略{sl_name}"
                                f"亏损{sl_loss:.2f}≥阈值{self._offensive_stop_loss_pnl_threshold:.2f}"
                                f"，减仓回落到{sl_target:.1%}"
                            ),
                        })
                    del self._offensive_attribution[sl_name]
                    self._offensive_attribution_cycle.pop(sl_name, None)
                    # 强化学习负反馈：止损=上次进攻决策错误→EMA 记 -1，下次缩小 boost
                    self._update_offensive_feedback(sl_name, -1.0)
                    # 止损了结 → 连续进攻计数归零（重新开始累积）
                    self._offensive_consecutive.pop(sl_name, None)
                    # 记录止损冷却周期：cooldown_cycles 内禁止重新进攻该策略
                    self._offensive_stop_loss_cooldown[sl_name] = self._cycle_count
                    profit_taken.add(sl_name)
        # 账户级总进攻敞口：当前分配计划所有策略目标权重之和，作为加仓封顶的基数。
        # 随本轮每次加仓累加（每次加仓抬升组合总权重 boost_step），突破上限即停。
        # 注意：本轮止盈/止损减仓不抵扣基数——「刚止损就补仓」正是冷却守卫要禁止的，
        # 故保持基数保守（不减），避免同一周期内止损减仓又立刻被进攻加仓对冲掉。
        total_alloc = self._total_alloc_weight(decision)
        for alert in alerts or []:
            if alert.get("type") != "offensive_opportunity":
                continue
            boost_step = safe_float(alert.get("boost_step"), self._offensive_boost_step)
            # 震荡市进攻参数独立化：mode == range_bound 时用独立的 max_target/cooldown
            is_range_mode = alert.get("mode") == "range_bound"
            max_target = (self._offensive_range_bound_max_target if is_range_mode
                          else self._offensive_max_target)
            min_interval = (self._offensive_range_bound_min_interval_cycles if is_range_mode
                            else self._offensive_min_interval_cycles)
            # 自适应风险偏好：权益上升→接近全额进攻；下跌→按 min_boost_ratio 收敛
            if self._adaptive_risk_enabled:
                appetite = self._risk_appetite()
                boost_step *= (self._adaptive_risk_min_boost_ratio
                               + (1.0 - self._adaptive_risk_min_boost_ratio) * appetite)
            # 决策质量死锁自愈：持续触发 recovery_after_cycles 周期后，
            # 按 recovery_boost_scale 缩放 boost_step，允许最小试探开单
            if _dq_recovery_scale is not None:
                boost_step *= _dq_recovery_scale
            # 进攻最小增幅保护：缩放后加仓幅度过小 → 跳过整个告警（微调徒耗手续费，
            # 且避免污染归因/冷却/连续计数状态——这些微动作本会被 cost_guard 过滤）。
            if self._offensive_min_boost_delta > 0 and boost_step < self._offensive_min_boost_delta:
                continue
            for name in (alert.get("strategies") or []):
                # 观察期冷却：距上次进攻不足 min_interval 个周期则跳过
                # （首次进攻 last is None，始终放行）
                last = self._last_offensive_cycle.get(str(name))
                if last is not None and self._cycle_count - last < min_interval:
                    continue
                # 本周期已止盈/止损回落 → 不在同周期重新加仓（避免「减了又加」自相矛盾）
                if str(name) in profit_taken:
                    continue
                # 止损后冷却期：刚亏了钱不加仓，避免频繁交易消耗资金
                sl_cd = self._offensive_stop_loss_cooldown.get(str(name))
                if sl_cd is not None and self._cycle_count - sl_cd < self._offensive_stop_loss_cooldown_cycles:
                    continue
                # 止盈后冷却期：刚止盈就再进同样是追涨，避免频繁交易消耗资金
                pt_cd = self._offensive_profit_take_cooldown.get(str(name))
                if pt_cd is not None and self._cycle_count - pt_cd < self._offensive_profit_take_cooldown_cycles:
                    continue
                # 决策震荡抑制：该策略最近刚被防守性减仓 → 不立即重新加仓（减了又加振荡）
                dr_cd = self._last_defensive_reduce_cycle.get(str(name))
                if (self._oscillation_enabled
                        and dr_cd is not None
                        and self._cycle_count - dr_cd < self._oscillation_cooldown_cycles):
                    continue
                # 胜率趋势恶化：该策略胜率连续下降 → 不追高加仓（per-strategy gate）
                if str(name) in wr_deteriorating:
                    continue
                # 胜率绝对值过低：该策略胜率低于阈值（长期低胜率）→ 不追高加仓（per-strategy gate）
                if str(name) in low_win_rate:
                    continue
                # 夏普趋势恶化：该策略夏普连续下降 → 不追高加仓（per-strategy gate）
                if str(name) in sr_deteriorating:
                    continue
                # 连续亏损：该策略 consecutive_losses 达阈值 → 不追高加仓（per-strategy gate）
                if str(name) in cl_strategies:
                    continue
                # 止损率过高：该策略入场时机差 → 不追高加仓（per-strategy gate）
                if str(name) in high_slr_strategies:
                    continue
                # 止盈止损比过低：该策略离场质量差（善止损不善止盈） → 不追高加仓（per-strategy gate）
                if str(name) in low_tp_strategies:
                    continue
                # 盈亏比过低：该策略平均盈利相对平均亏损过小（小赚大亏）→ 不追高加仓（per-strategy gate）
                if str(name) in low_wlr_strategies:
                    continue
                # 止盈止损盈亏金额比过低：该策略赚的时候轻仓、亏的时候重仓 → 不追高加仓（per-strategy gate）
                if str(name) in low_tppr_strategies:
                    continue
                # 盈亏比恶化：该策略 profit_factor 连续下降 → 不追高加仓（per-strategy gate）
                if str(name) in pf_deteriorating:
                    continue
                # 最大回撤加深：该策略 max_drawdown 连续上升 → 不追高加仓（per-strategy gate）
                if str(name) in mdd_deteriorating:
                    continue
                # 最大回撤超阈值：该策略深度回撤 → 不追高加仓放大风险敞口（per-strategy gate）
                if str(name) in high_mdd:
                    continue
                # 回撤持续时间拉长：该策略恢复能力下降 → 不追高加仓（per-strategy gate）
                if str(name) in ddh_deteriorating:
                    continue
                # 资本回报率下降：该策略 pnl_per_capital_pct 连续下降 → 不追高加仓（per-strategy gate）
                if str(name) in cr_deteriorating:
                    continue
                # 资本回报率过低：该策略单位资本产出低于绝对阈值 → 不追高加仓（per-strategy gate）
                if str(name) in low_cr_strategies:
                    continue
                # 波动率上升：该策略 volatility 连续上升 → 不追高加仓（per-strategy gate）
                if str(name) in vol_deteriorating:
                    continue
                # 多守卫共振：该策略多维度同时衰退 → 不追高加仓（per-strategy gate）
                if str(name) in resonance_strategies:
                    continue
                # 盈利集中度：该策略是组合盈利支柱 → 不追高加仓（组合级 per-strategy gate）
                if str(name) in pc_dominant:
                    continue
                # 浮盈占比过高：该策略账面盈利靠浮盈支撑 → 不追高加仓放大浮盈暴露（per-strategy gate）
                if str(name) in upr_concentrated:
                    continue
                # 浮盈占比持续上升：该策略盈利越来越依赖浮盈 → 不追高加仓放大浮盈暴露（per-strategy gate）
                if str(name) in upr_rising:
                    continue
                # 浮亏加深：该策略浮亏连续加深 → 不追高加仓放大浮亏暴露（per-strategy gate）
                if str(name) in uld_deteriorating:
                    continue
                # 手续费侵蚀过高：该策略交易过频 → 不追高加仓放大换手（per-strategy gate）
                if str(name) in high_fee_strategies:
                    continue
                # 手续费率持续上升：该策略交易成本侵蚀加剧 → 不追高加仓放大换手（per-strategy gate）
                if str(name) in high_fee_rising:
                    continue
                # 资金费侵蚀过高：该策略持仓过久被 funding 持续侵蚀 → 不追高加仓放大持仓（per-strategy gate）
                if str(name) in high_funding_strategies:
                    continue
                # 资金费率持续上升：该策略持仓时间成本侵蚀加剧 → 不追高加仓放大持仓（per-strategy gate）
                if str(name) in funding_rising:
                    continue
                # 执行质量成本过高：该策略滑点/点差侵蚀过大 → 不追高加仓放大敞口（per-strategy gate）
                if str(name) in high_exec_cost_strategies:
                    continue
                # 执行成本率持续上升：该策略执行质量持续恶化 → 不追高加仓放大滑点敞口（per-strategy gate）
                if str(name) in exec_cost_rising:
                    continue
                # 多空方向失衡：该策略单方向持续逆势亏损 → 不追高加仓放大逆势敞口（per-strategy gate）
                if str(name) in imbalanced_strategies:
                    continue
                # 方向偏好失衡：该策略单边笔数占比过高（方向单一）→ 不追高加仓放大单边敞口（per-strategy gate）
                if str(name) in direction_bias_strategies:
                    continue
                # PnL 动量趋势恶化：该策略近期盈利动能持续衰减 → 不追高加仓放大敞口（per-strategy gate）
                if str(name) in pnl_momentum_deteriorating:
                    continue
                # PnL 动量衰竭：该策略近期动能相对中期衰减到阈值以下 → 不追高加仓放大动能衰竭敞口（per-strategy gate）
                if str(name) in pnl_momentum_faded_strategies:
                    continue
                # 单笔期望值持续下降：该策略每笔交易价值持续萎缩 → 不追高加仓放大敞口（per-strategy gate）
                if str(name) in pptr_deteriorating:
                    continue
                # 单笔期望值低于阈值：该策略每笔无正期望（交易质量差）→ 不追高加仓放大低质量敞口（per-strategy gate）
                if str(name) in low_ppt:
                    continue
                # 推算预期为负：该策略未来 horizon 期望亏损 → 不追高加仓放大亏损敞口（per-strategy gate）
                if str(name) in negative_projection_strategies:
                    continue
                # bear_case fail-closed：该策略悲观场景 horizon 亏损超过阈值 → 不追高加仓放大下行敞口（per-strategy gate）
                if str(name) in bear_case_fail_closed_strategies:
                    continue
                # 边际盈亏持续为负：该策略最近持续失血 → 不追高加仓放大敞口（per-strategy gate）
                if str(name) in dp_deteriorating:
                    continue
                # 单周期亏损急跌：该策略单周期大额急跌（突发大幅亏损）→ 不追高加仓放大急跌敞口（per-strategy gate）
                if str(name) in delta_pnl_spike_strategies:
                    continue
                # 风险调整后贡献过低：该策略盈利相对回撤过少（小赚大扛）→ 不追高加仓放大风险（per-strategy gate）
                if str(name) in low_rac_strategies:
                    continue
                # 闲置策略：该策略长时间未交易 → 不追高加仓，先回收闲置资金（per-strategy gate）
                if str(name) in stale_strategies:
                    continue
                # 健康度骤降：该策略健康度单周期暴跌 → 不追高加仓，先降杠杆收敛（per-strategy gate）
                if str(name) in health_crash_strategies:
                    continue
                # 波动率超预算：该策略单笔盈亏标准差过大 → 不追高加仓放大不确定性（per-strategy gate）
                if str(name) in vol_budget_strategies:
                    continue
                # 负夏普：该策略风险调整后负收益 → 不追高加仓放大风险敞口（per-strategy gate）
                if str(name) in negative_sharpe_strategies:
                    continue
                # 盈亏比过低：该策略总亏损超过总盈利（已实现净亏损）→ 不追高加仓放大亏损敞口（per-strategy gate）
                if str(name) in low_profit_factor_strategies:
                    continue
                # 回撤持续过久：该策略最长回撤持续过久（恢复能力差）→ 不追高加仓放大被套牢资金（per-strategy gate）
                if str(name) in long_drawdown_duration_strategies:
                    continue
                # 连续进攻次数上限：无了结（未经历止盈/止损/防守减仓）连续加仓达上限 → 暂停，
                # 防止无了结地一路追高加仓（max_consecutive_offenses=0 表示不限）
                if self._offensive_max_consecutive > 0:
                    consec = self._offensive_consecutive.get(str(name), 0)
                    if consec >= self._offensive_max_consecutive:
                        continue
                # 决策效果归因：上次加仓后该策略累计盈亏转差 → 上次决策错误，本次不追高加仓
                metrics = (decision.get("strategy_metrics") or {}).get(name) or {}
                current_pnl = safe_float(metrics.get("total_pnl"), 0.0)
                entry_pnl = self._offensive_attribution.get(str(name))
                if entry_pnl is not None and current_pnl < entry_pnl:
                    continue
                base = self._base_weight_for(decision, name)
                # 进攻封顶守卫：当前目标权重已达/超过 max_target 时，进攻加仓无意义
                # （target = min(base+boost, max_target) 会 ≤ base，实际是减仓甚至反向），
                # 直接跳过，避免生成「标记 increase 实为 decrease」的无效动作。
                if base >= max_target:
                    continue
                # 账户级总进攻敞口封顶：本轮加仓后组合总权重突破 max_total_allocation 时跳过，
                # 避免多策略火力集中叠加把组合推到超配/杠杆状态（账户级总敞口守卫）。
                # 强化学习反馈缩放：该策略历史进攻胜率→调整 boost（对的加大、错的缩小）
                rl_mult = self._offensive_feedback_multiplier(str(name))
                strategy_boost = boost_step * rl_mult
                # 推算缩放：正期望高置信→加大 boost；负期望→缩小（前瞻期望维度）
                if self._pnl_projection_enabled and projection and projection.get("available"):
                    _sp = (projection.get("per_strategy") or {}).get(str(name)) or {}
                    _proj_mult = self._projection_boost_scale(
                        _sp.get("corrected_expect"), _sp.get("confidence"))
                    strategy_boost *= _proj_mult
                if total_alloc + strategy_boost > self._offensive_max_total_allocation:
                    continue
                target = min(base + strategy_boost, max_target)
                actions.append({
                    "type": "reallocate",
                    "strategy": name,
                    "action": "increase",
                    "target_allocation": target,
                    "reason": (
                        f"offensive_allocation: {'震荡市高抛低吸' if is_range_mode else '趋势确认'}"
                        f"+健康策略 {name}，进攻加仓到 {target:.1%}"
                    ),
                })
                self._last_offensive_cycle[str(name)] = self._cycle_count
                self._offensive_attribution[str(name)] = current_pnl
                # 记录归因基线写入周期号（供 TTL 过期清理），并累计连续进攻次数
                self._offensive_attribution_cycle[str(name)] = self._cycle_count
                self._offensive_consecutive[str(name)] = self._offensive_consecutive.get(str(name), 0) + 1
                # 累计本轮已加仓抬升的组合总权重，供后续策略的账户级总敞口封顶判断
                total_alloc += strategy_boost
        return actions

    def _update_offensive_feedback(self, strategy: str, outcome: float) -> None:
        """强化学习反馈更新：outcome=+1（止盈/正确）或 -1（止损/错误），
        用 EMA 平滑为滚动分数 [-1,1]，存入 _offensive_feedback_score。
        首次记录直接取 outcome（无历史基线时 EMA 从 0 开始会过度平滑首样本）。
        """
        if not self._offensive_rl_feedback_enabled:
            return
        old = self._offensive_feedback_score.get(str(strategy), 0.0)
        # 首次：直接置 outcome；后续：EMA 平滑
        if str(strategy) not in self._offensive_feedback_score:
            self._offensive_feedback_score[str(strategy)] = max(-1.0, min(1.0, outcome))
        else:
            new = old + self._offensive_rl_ema_alpha * (outcome - old)
            self._offensive_feedback_score[str(strategy)] = max(-1.0, min(1.0, new))

    def _offensive_feedback_multiplier(self, strategy: str) -> float:
        """根据策略的强化学习反馈分数计算 boost 缩放倍率。

        score∈[-1,1] → mult∈[1-span, 1+span]，再 clamp 到 [min_mult, max_mult]。
        无历史分数：暖启动开启时用账户级 decision_quality 作为先验（dq∈[0,1]→score∈[-1,1]），
        让 RL 循环从首个进攻就有方向；关闭时 score=0→mult=1.0（向后兼容）。
        """
        if not self._offensive_rl_feedback_enabled:
            return 1.0
        key = str(strategy)
        if key in self._offensive_feedback_score:
            score = self._offensive_feedback_score[key]
        elif self._offensive_rl_warm_start:
            # 暖启动：用账户级决策质量作为先验，dq=0.5→score=0（中性）
            dq = self._decision_quality_score()
            score = (dq - 0.5) * 2.0
        else:
            score = 0.0
        mult = 1.0 + score * self._offensive_rl_adaptation_span
        return max(self._offensive_rl_min_mult,
                   min(self._offensive_rl_max_mult, mult))

    def _base_weight_for(self, decision: Dict[str, Any], name: str) -> float:
        """从分配计划/重分配建议中提取策略的当前目标权重，缺省回退 0.0。"""
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        if isinstance(allocs, dict):
            info = allocs.get(name)
            if isinstance(info, dict):
                w = safe_float(info.get("target_weight"), -1.0)
                if w >= 0.0:
                    return w
        for s in (decision.get("reallocation_suggestions") or []):
            if s.get("strategy") == name and s.get("target_allocation") is not None:
                return safe_float(s.get("target_allocation"), 0.0)
        return 0.0

    def _total_alloc_weight(self, decision: Dict[str, Any]) -> float:
        """计算分配计划中所有策略目标权重之和，用于账户级总进攻敞口上限守卫。"""
        plan = decision.get("allocation_plan") or {}
        allocs = plan.get("strategy_allocations") or {}
        total = 0.0
        if isinstance(allocs, dict):
            for info in allocs.values():
                if isinstance(info, dict):
                    w = safe_float(info.get("target_weight"), -1.0)
                    if w >= 0.0:
                        total += w
        return total

    def _strategy_lifecycle_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """自主策略生命周期管理：永久冻结/休眠 → strategy_pause 动作。

        - strategy_permanently_frozen（试探耗尽，需人工）→ 暂停停开新仓。
        - strategy_dormant（休眠）→ 暂停回收算力/资金。
        均属降风险方向，由 RestrictedExecutionChannel 低风险自动执行。
        """
        if not self._strategy_lifecycle_enabled:
            return []
        actions: List[Dict[str, Any]] = []
        seen: set = set()
        for alert in alerts or []:
            atype = alert.get("type")
            strategy = alert.get("strategy")
            if not strategy or strategy in seen:
                continue
            if atype == "strategy_permanently_frozen" and self._pause_permanently_frozen:
                pass
            elif atype == "strategy_dormant" and self._pause_dormant:
                pass
            else:
                continue
            seen.add(strategy)
            actions.append({
                "type": "strategy_pause",
                "strategy": strategy,
                "reason": f"strategy_lifecycle:{atype}: {alert.get('message', '')}",
            })
            # 记录 AGI 已暂停策略，供健康度改善后自主恢复（_strategy_resume_actions）
            self._paused_strategies.add(str(strategy))
        return actions

    def _strategy_resume_actions(self, alerts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """自主策略恢复：AGI 暂停过的策略健康度改善 → strategy_resume 动作。

        - 仅恢复本实例此前生成过 strategy_pause 的策略（_paused_strategies 追踪），
          避免误恢复从未被 AGI 暂停的策略。
        - 触发条件：strategy_recovered（健康度改善至 A/B）alert。
        - 生成 resume 后从集合移除（幂等：同周期只恢复一次）。
        由 RestrictedExecutionChannel 低风险自动执行（strategy_manager.resume_strategy
        有 P2-12 prelaunch_gate 兜底），不直接下单。
        """
        if not self._strategy_lifecycle_enabled or not self._resume_recovered:
            return []
        actions: List[Dict[str, Any]] = []
        for alert in alerts or []:
            if alert.get("type") != "strategy_recovered":
                continue
            strategy = alert.get("strategy")
            if not strategy or strategy not in self._paused_strategies:
                continue
            self._paused_strategies.discard(strategy)
            # 策略恢复开仓 → 重置回吐峰值，新周期从零计起，避免陈旧峰值立即重新触发回吐
            if self._give_back_peak_reset_on_resume:
                self._strategy_pnl_peak[str(strategy)] = 0.0
                self._give_back_peak_last_decay_cycle.pop(str(strategy), None)
            actions.append({
                "type": "strategy_resume",
                "strategy": strategy,
                "reason": f"strategy_lifecycle: {alert.get('message', '健康度改善，恢复开仓')}",
            })
        return actions

    def _idle_cash_deploy_actions(self, decision: Dict[str, Any]) -> List[Dict[str, Any]]:
        """低风险闲置资金自动归集动作。

        触发条件：
        - 闲置资金占比 > idle_deploy_threshold（默认 10%）
        - 存在可承接资金的目标策略：
          1) 优先 health_grade ∈ {A, B} 的核心策略
          2) 无 A/B 级时，兜底识别「资金池内正收益策略」
             （profit_factor > 1 且 total_pnl > 0）做缩量试探归集，
             打破「无 A/B 级 → 不归集 → 无交易 → 无评级」死锁

        该动作被 RestrictedExecutionChannel 判为低风险自动执行（调用 deployer），
        不进入人工确认队列；实际下单仍受下游 RiskGate 六层约束。
        """
        plan = decision.get("allocation_plan") or {}
        idle_cash = safe_float(plan.get("idle_cash"), 0.0)
        total_equity = safe_float(plan.get("total_equity"), 0.0)
        if idle_cash <= 0 or total_equity <= 0:
            return []
        idle_pct = idle_cash / total_equity
        if idle_pct <= self.idle_deploy_threshold:
            return []

        # 资金池策略集合：reallocation_suggestions 已过滤 sync/manual_override 等非资金池标签
        pool_strategies = {
            str(s.get("strategy"))
            for s in (decision.get("reallocation_suggestions") or [])
            if s.get("strategy")
        }

        # 1) 优先 A/B 级核心策略（原有逻辑）
        core = [
            s for s in (decision.get("reallocation_suggestions") or [])
            if s.get("health_grade") in ("A", "B")
        ]

        # 2) 兜底：无 A/B 级时，识别资金池内「正收益策略」做缩量试探，打破死锁
        if not core:
            metrics = decision.get("strategy_metrics") or {}
            for name, m in metrics.items():
                if name not in pool_strategies:
                    continue  # 排除 sync/manual_override 等非资金池标签（无法被 deployer 落地）
                pf = safe_float(m.get("profit_factor"), 1.0)
                pnl = safe_float(m.get("total_pnl"), 0.0)
                if pf > 1.0 and pnl > 0:
                    core.append({
                        "strategy": name,
                        "health_grade": "positive",
                        "target_allocation": None,
                    })

        if not core:
            return []

        actions: List[Dict[str, Any]] = []
        for s in core:
            grade = s.get("health_grade")
            actions.append({
                "type": "idle_cash_deploy",
                "strategy": s.get("strategy"),
                "health_grade": grade,
                "target_allocation": s.get("target_allocation"),
                "detail": (
                    f"deploy idle cash ({idle_cash:.1f} USDT, {idle_pct:.0%}) "
                    f"to {s.get('strategy')} "
                    f"(grade {grade}{' positive-return probe' if grade == 'positive' else ''})"
                ),
            })
        return actions

    # ─────────────────────────────────────────────────────────────
    # 5. 反馈 Reflect
    # ─────────────────────────────────────────────────────────────

    def _reflect(self, report: Dict[str, Any], perception: Dict[str, Any],
                 alerts: List[Dict[str, Any]]) -> None:
        contrib = _coerce_dict(perception.get("contribution"))
        health = safe_float(contrib.get("overall_health_score"), 0.0)
        total_trades = safe_int(contrib.get("total_trades"), 0)
        # 区分「数据不足」与「健康度差」：成交数低于最小样本量时不得评估健康度。
        # overall_health_score 在无 active contribs 时返回 0.0，若直接套等级会
        # 把「样本不足」误判为「健康度 F」，进而在顶层联动里反复触发 health_degraded。
        if total_trades < self.min_health_sample:
            grade = "N/A"
        else:
            grade = _health_grade_for(health)

        # 跨周期时序：健康度趋势（仅样本充足时才有意义，±2 分为滞回带）
        health_trend = "stable"
        if total_trades >= self.min_health_sample:
            if self._last_health_score is not None:
                delta = health - self._last_health_score
                if delta >= 2.0:
                    health_trend = "improving"
                elif delta <= -2.0:
                    health_trend = "declining"
            self._last_health_score = health

        report["reflection"]["health_score"] = health
        report["reflection"]["health_grade"] = grade
        report["reflection"]["health_trend"] = health_trend
        report["reflection"]["alerts_count"] = len(alerts)
        report["reflection"]["cycle_summary"] = (
            f"cycle={report['cycle']} health={grade}({health:.0f}) trend={health_trend} "
            f"trades={total_trades} alerts={len(alerts)}"
        )
        logger.info(
            f"[AGI-Reflect] health={health:.0f} grade={grade} trend={health_trend} "
            f"trades={total_trades} alerts={len(alerts)}"
        )

        # 跨周期学习记忆：记录本周期决策结果（供后续周期自适应微调落袋阈值）
        if self._learning_enabled:
            cur_total_pnl = safe_float(contrib.get("total_pnl"), 0.0)
            # cycle_pnl = 本周期相对上周期的 closed-PnL 增量（而非累计值），
            # 使 _decision_quality_score 能区分单周期决策的盈亏；
            # 首周期无上期基线时记 0（中性，既非盈也非亏）。
            if (self._last_decision_total_pnl is not None
                    and self._decision_quality_use_cycle_pnl):
                cycle_pnl = cur_total_pnl - self._last_decision_total_pnl
            else:
                cycle_pnl = 0.0
            self._last_decision_total_pnl = cur_total_pnl
            self._decision_memory.append({
                "cycle": report.get("cycle"),
                "health_score": health,
                "total_pnl": cur_total_pnl,
                "cycle_pnl": cycle_pnl,
                "equity": safe_float(perception.get("equity"), 0.0),
                "regime": (perception.get("market_regime") or {}).get("regime"),
            })

        # 自适应风险偏好：记录本周期权益轨迹（供下一周期风险偏好计算）
        self._equity_window.append(safe_float(perception.get("equity"), 0.0))
        report["reflection"]["risk_appetite"] = self._risk_appetite()
        report["reflection"]["decision_quality"] = self._decision_quality_score(
            (perception.get("market_regime") or {}).get("regime"))

        # 开单时机校准：更新市场状态持续周期（供下一周期进攻的趋势确认判断）
        regime = (perception.get("market_regime") or {}).get("regime")
        if regime == self._trend_confirmed_regime:
            self._trend_confirmed_streak += 1
        else:
            self._trend_confirmed_regime = regime
            self._trend_confirmed_streak = 1

        # 决策执行结果反馈：记录上一周期动作的执行结果（供 Dashboard/溯源审计）
        report["reflection"]["execution_result"] = (
            copy.deepcopy(self._last_execution_result)
            if self._last_execution_result is not None else None
        )

        # 推算准确度追踪（闭环：实际 vs 上次推算），用于校准未来推算
        if self._pnl_projection_enabled and self._last_projection is not None:
            try:
                accuracy = self._track_projection_accuracy(perception)
                report["reflection"]["projection_accuracy"] = accuracy
            except Exception as e:
                logger.warning(f"[AGI-Reflect] projection accuracy failed: {e}")
        # 本周期推算结果保存为下一周期准确度追踪基线
        self._last_projection = report.get("projection") or None

    # ─────────────────────────────────────────────────────────────
    # 持久化与 JSON 安全清洗
    # ─────────────────────────────────────────────────────────────

    def _persist_state(self, report: Dict[str, Any]) -> None:
        """持久化闭环状态到 data/agi_orchestrator_state.json（atomic write，fail-closed）。"""
        try:
            state_dir = os.path.dirname(self.state_path)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            state = dict(report)
            # 学习型状态（跨重启续用）：进攻归因/观察期冷却/利润峰值/健康度时序/突变冷却
            state["learning_state"] = self._serialize_learning_state()
            # Atomic write: write to temp file, then atomically replace
            temp_path = f"{self.state_path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
            os.replace(temp_path, self.state_path)
            logger.debug(f"[AGI-Reflect] state persisted to {self.state_path}")
        except Exception as e:
            logger.error(f"[AGI-Reflect] state persist failed (fail-closed): {e}")

    def _sanitize(self, obj: Any) -> Any:
        """递归清洗，保证 json.dumps 无 NaN/Infinity，且不污染调用方数据结构。"""
        if obj is None:
            return None
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, float):
            return safe_finite(obj, 0.0)
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._sanitize(v) for v in obj]
        # 兜底：numpy 数值 / 枚举 / datetime 等
        try:
            return safe_finite(float(obj), 0.0)
        except (TypeError, ValueError):
            return str(obj)

    def get_last_report(self) -> Optional[Dict[str, Any]]:
        """返回最近一次闭环报告的副本（供外部观测，不暴露内部引用）。"""
        if self._last_report is None:
            return None
        return copy.deepcopy(self._last_report)

    def get_bear_case_open_block_reason(self, strategy_name: str) -> Optional[str]:
        """Return the latest bear-case threshold rejection reason for an opening signal."""
        if not self._pnl_projection_enabled or not strategy_name:
            return None

        report = self._last_report
        if not isinstance(report, dict):
            return None
        projection = report.get("projection")
        if not isinstance(projection, dict) or not projection.get("available"):
            return None
        per_strategy = projection.get("per_strategy")
        if not isinstance(per_strategy, dict):
            return None
        strategy_projection = per_strategy.get(str(strategy_name))
        if not isinstance(strategy_projection, dict):
            return None

        perception = report.get("perception")
        equity = safe_float(
            perception.get("equity"), 0.0
        ) if isinstance(perception, dict) else 0.0
        if equity <= 0:
            return None

        bear_case_horizon = safe_float(
            strategy_projection.get("bear_case_horizon"), 0.0
        )
        threshold = -abs(safe_float(self._projection_fail_closed_loss, 0.05)) * equity
        if bear_case_horizon < threshold:
            return (
                f"bear_case_horizon={bear_case_horizon:.8g} "
                f"below_threshold={threshold:.8g}"
            )
        return None

    def _apply_simulation_safety(self, report: Dict[str, Any]) -> None:
        """Attach paper-sandbox-only risk simulations without creating executable actions."""
        if not self._simulation_safety_enabled:
            return

        if report.get("status") == "fail_closed":
            self._simulation_fail_closed_streak += 1
        else:
            self._simulation_fail_closed_streak = 0

        if (
            self._simulation_fail_closed_streak >= self._simulation_fail_closed_threshold
            and not self._simulation_kill_switch_enabled
        ):
            self._simulation_kill_switch_enabled = True
            self._simulation_kill_switch_reason = (
                "consecutive_agi_fail_closed:"
                f"{self._simulation_fail_closed_streak}"
            )
            logger.warning(
                "[AGI-Simulation] simulated Kill Switch threshold reached "
                f"(streak={self._simulation_fail_closed_streak})"
            )

        simulated_reductions = []
        perception = report.get("perception")
        projection = report.get("projection")
        equity = (
            safe_float(perception.get("equity"), 0.0)
            if isinstance(perception, dict) else 0.0
        )
        if (
            report.get("status") != "fail_closed"
            and equity > 0
            and isinstance(projection, dict)
            and projection.get("available")
        ):
            threshold = (
                -abs(self._projection_fail_closed_loss) * equity
            )
            per_strategy = projection.get("per_strategy")
            if isinstance(per_strategy, dict):
                for strategy, metrics in per_strategy.items():
                    if not isinstance(metrics, dict):
                        continue
                    bear_case_horizon = safe_float(
                        metrics.get("bear_case_horizon"), 0.0
                    )
                    if bear_case_horizon >= threshold:
                        continue
                    simulated_reductions.append({
                        "type": "simulated_bear_case_reduction",
                        "strategy": str(strategy),
                        "reduce_ratio": self._simulation_bear_case_reduce_ratio,
                        "bear_case_horizon": bear_case_horizon,
                        "simulated_bear_case_horizon_after_reduction": (
                            bear_case_horizon
                            * (1.0 - self._simulation_bear_case_reduce_ratio)
                        ),
                        "simulation_assumption": "linear_exposure_scaling",
                        "threshold": threshold,
                        "reason": "bear_case_projection_below_loss_threshold",
                        "simulation_only": True,
                        "execution_applied": False,
                    })

        report["simulation_actions"] = simulated_reductions
        report["simulation_safety"] = {
            "enabled": True,
            "mode": "paper_sandbox",
            "state_scope": "process",
            "fail_closed_streak": self._simulation_fail_closed_streak,
            "fail_closed_threshold": self._simulation_fail_closed_threshold,
            "simulated_kill_switch": {
                "enabled": self._simulation_kill_switch_enabled,
                "would_block_new_openings": self._simulation_kill_switch_enabled,
                "reason": self._simulation_kill_switch_reason,
                "execution_applied": False,
            },
            "execution_applied": False,
        }

    def reset_simulated_kill_switch(self) -> None:
        """Reset the in-memory simulation latch; never changes the production Kill Switch."""
        self._simulation_kill_switch_enabled = False
        self._simulation_kill_switch_reason = ""
        self._simulation_fail_closed_streak = 0

    def reset_cycle_circuit_breaker(self) -> None:
        """Reset the cycle failure circuit breaker after operator intervention."""
        self._cycle_halted = False
        self._cycle_halt_reason = ""
        self._cycle_failure_streak = 0
        logger.info("[AGI] cycle circuit breaker manually reset")

    def get_decision_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """返回最近决策溯源历史（深拷贝，最新在前），供 Dashboard 回溯决策依据链。"""
        items = list(self._decision_lineage)
        limit = max(1, min(1000, int(limit)))
        return [copy.deepcopy(x) for x in reversed(items[-limit:])]

    def get_risk_appetite(self) -> float:
        """返回当前自适应风险偏好 [0,1]（供外部观测/联动用）。"""
        return self._risk_appetite()

    def report_execution_result(self, result: Dict[str, Any]) -> None:
        """接收上一周期动作的执行结果（由 scheduler 在 route 后回调）。

        形成「决策→执行→反馈」闭环：本周期诊断可感知上一周期动作是否被
        deployed/queued/rejected（如被 Kill Switch 拒绝），据此收敛进攻。
        """
        if not isinstance(result, dict):
            return
        action_results = []
        raw_actions = result.get("action_results")
        if isinstance(raw_actions, list):
            for item in raw_actions[-100:]:
                if not isinstance(item, dict):
                    continue
                action_results.append({
                    "trace_id": str(item.get("trace_id") or ""),
                    "type": str(item.get("type") or "unknown"),
                    "status": str(item.get("status") or "unknown"),
                    "reason": str(item.get("reason") or item.get("previous_status") or "")[:200],
                    "result": {
                        key: self._sanitize(item["result"][key])
                        for key in ("deployed", "status", "reason", "detail")
                        if isinstance(item.get("result"), dict) and key in item["result"]
                    },
                })

        self._last_execution_result = {
            "queued": safe_int(result.get("queued"), 0),
            "deployed": safe_int(result.get("deployed"), 0),
            "rejected": safe_int(result.get("rejected"), 0),
            "notified": safe_int(result.get("notified"), 0),
            "skipped": safe_int(result.get("skipped"), 0),
            "action_results": action_results,
        }
        memory_item = {
            "cycle": safe_int(result.get("cycle"), 0),
            "decision_id": str(result.get("decision_id") or ""),
            "timestamp": datetime.now().isoformat(),
            "result": copy.deepcopy(self._last_execution_result),
        }
        self._execution_memory.append(memory_item)
        if self._last_report is not None:
            self._last_report.setdefault("reflection", {})["execution_result"] = copy.deepcopy(
                self._last_execution_result
            )
            self._persist_state(self._last_report)

    def get_execution_result(self) -> Optional[Dict[str, Any]]:
        """返回上一周期动作执行结果（深拷贝，供 Dashboard 观测）。"""
        if self._last_execution_result is None:
            return None
        return copy.deepcopy(self._last_execution_result)

    def get_execution_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Return recent per-cycle execution feedback, newest first."""
        limit = max(1, min(50, int(limit)))
        return [copy.deepcopy(item) for item in list(self._execution_memory)[-limit:][::-1]]


__all__ = ["QuantAGIOrchestrator"]
