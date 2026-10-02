"""
策略贡献度分析引擎（强化版）
========================
统一管理所有策略的贡献度计算、趋势追踪、效率评估、相关性分析、
生命周期分类、健康度评分和资金重分配建议。

核心能力：
1. 多时间窗口贡献度拆解（1h / 6h / 24h / 7d / 30d）
2. 风险调整后贡献度（PnL / MaxDD / Capital）
3. 资金效率矩阵（PnL per unit of capital / per trade / per hour）
4. 策略间 PnL 相关性分析（互补/重叠识别）
5. 贡献趋势检测（改善 / 恶化 / 稳定）
6. 贡献度快照持久化（供 Dashboard 和 ReviewEngine 消费）
7. 【强化】策略生命周期分类（新生期/成熟期/衰退期/休眠期）
8. 【强化】综合健康度评分（0-100分）
9. 【强化】多时间窗口对比分析
10.【强化】贡献度增量追踪（δ 变化）
11.【强化】策略协同效应分析
12.【强化】未实现盈亏追踪
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from loguru import logger

from utils.helpers import safe_finite


def _safe_float(value, default=0.0):
    """安全转换为有限浮点数；None/NaN/Inf/不可解析值回退为 default。"""
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(f) or np.isinf(f):
        return default
    return f


# ============================================================
# 数据模型
# ============================================================

@dataclass
class StrategyContribution:
    """单策略贡献度快照"""
    strategy: str
    # PnL 贡献
    total_pnl: float = 0.0
    pnl_contribution_pct: float = 0.0          # 占全部 PnL 的百分比
    realized_pnl: float = 0.0                   # 已实现盈亏
    unrealized_pnl: float = 0.0                 # 未实现盈亏
    # 交易统计
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    consecutive_losses: int = 0                 # 最大连续亏损笔数（负期望早期熔断信号）
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    # 风险指标
    max_drawdown: float = 0.0
    max_drawdown_duration_hours: float = 0.0
    # 资金效率
    avg_capital_used: float = 0.0               # 平均占用资金
    pnl_per_capital_pct: float = 0.0            # PnL / 平均资本 (资本回报率)
    pnl_per_trade: float = 0.0                  # 每笔交易平均盈亏
    total_fees: float = 0.0                     # 总手续费
    fee_ratio: float = 0.0                      # 手续费 / |PnL| (手续费损耗率)
    # 成本细分归因（企业级：区分手续费 / 资金费 / 滑点 / 点差）
    total_funding_cost: float = 0.0             # 总资金费
    total_slippage_cost: float = 0.0            # 总滑点成本
    total_spread_cost: float = 0.0              # 总点差成本
    # 方向归因（long / short 分解）
    long_pnl: float = 0.0                       # 多单累计盈亏
    short_pnl: float = 0.0                      # 空单累计盈亏
    long_trades: int = 0                        # 多单笔数
    short_trades: int = 0                       # 空单笔数
    # 退出原因归因（止损 / 止盈 分解）
    stop_loss_count: int = 0                    # 止损平仓笔数
    stop_loss_pnl: float = 0.0                  # 止损单累计盈亏
    take_profit_count: int = 0                  # 止盈平仓笔数
    take_profit_pnl: float = 0.0                # 止盈单累计盈亏
    # 时间维度
    first_trade_time: Optional[str] = None
    last_trade_time: Optional[str] = None
    active_hours: float = 0.0                   # 策略活跃时长
    pnl_per_hour: float = 0.0                   # 每小时盈利
    # 贡献趋势
    trend: str = "stable"                       # improving / declining / stable
    trend_pnl_7d_vs_30d: float = 0.0            # 近7天 vs 近30天 PnL 比率
    # 风险调整贡献
    risk_adjusted_contribution: float = 0.0     # PnL / MaxDD (风险调整后贡献)
    sharpe_ratio: float = 0.0                   # 真实夏普比率（mean/std，与寻优侧口径一致）
    volatility: float = 0.0                     # 单笔盈亏标准差（波动率，独立风险维度）
    # 【强化】生命周期
    lifecycle: str = "unknown"                   # newborn / mature / declining / dormant
    lifecycle_age_days: float = 0.0              # 首次交易至今的天数
    lifecycle_last_trade_age_hours: float = 0.0  # 距上次交易的小时数
    # 【强化】健康度
    health_score: float = 0.0                    # 综合健康度 0-100
    health_grade: str = "N/A"                    # A/B/C/D/F
    # 【强化】增量变化（相对于上一快照）
    delta_pnl: float = 0.0
    delta_contribution_pct: float = 0.0
    delta_health: float = 0.0


@dataclass
class ContributionSnapshot:
    """完整贡献度快照"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    window: str = "24h"
    total_pnl: float = 0.0
    total_trades: int = 0
    total_fees: float = 0.0
    total_unrealized_pnl: float = 0.0            # 全部未实现盈亏
    strategies: Dict[str, StrategyContribution] = field(default_factory=dict)
    # 跨策略分析
    correlation_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)
    concentration: float = 0.0                   # Herfindahl指数 (集中度)
    diversification_score: float = 0.0           # 分散化得分 (0-1)
    efficiency_score: float = 0.0                # 整体资金效率得分
    # 【强化】综合评分
    overall_health_score: float = 0.0            # 系统整体健康度 0-100
    synergy_score: float = 0.0                   # 策略协同效应得分
    # 【强化】生命周期统计
    lifecycle_summary: Dict[str, int] = field(default_factory=dict)  # 各生命周期策略数
    # 数据可用性标记（fail-closed）
    available: bool = True
    data_error: Optional[str] = None


# ============================================================
# 核心引擎
# ============================================================

class ContributionAnalyzer:
    """策略贡献度分析引擎（强化版）"""

    def __init__(self, sqlite_storage=None, trade_journal=None,
                 config: Dict[str, Any] = None, okx_client=None,
                 account_manager=None):
        self._sqlite = sqlite_storage
        self._journal = trade_journal
        self._okx_client = okx_client
        self._account_manager = account_manager
        self.config = config or {}

        # 动态分配权重（从 AdaptiveController 注入）
        self._dynamic_allocations: Dict[str, float] = {}
        # 历史快照（内存缓存）
        self._snapshot_history: List[ContributionSnapshot] = []
        self._max_history = 168  # 保留最近 168 个快照 (7天 * 24小时)
        # 交易数据查询状态（fail-closed）
        self._trade_query_error: Optional[str] = None

        # 策略生命周期配置
        contrib_cfg = self.config.get("contribution", {})
        self._lifecycle_config = contrib_cfg.get("lifecycle", {
            "newborn_max_days": 7,           # 7天内为新生期
            "mature_min_trades": 10,         # 成熟期最少10笔交易
            "dormant_max_idle_hours": 72,    # 72小时无交易=休眠
            "declining_7d_vs_30d_ratio": 0.5,  # 7天vs30天<50%=衰退
        })

        # 健康度评分权重
        self._health_weights = contrib_cfg.get("health_weights", {
            "profit_factor": 0.20,
            "win_rate": 0.15,
            "risk_adjusted": 0.20,
            "capital_efficiency": 0.15,
            "fee_efficiency": 0.10,
            "trend": 0.10,
            "stability": 0.10,
        })

        # 健康度最小样本量：低于此笔成交的策略不评估健康度（噪声太大），标记为 N/A
        self._min_health_sample = int(contrib_cfg.get("health_min_sample", 5))

        # 窗口配置
        self._windows = {
            "1h": timedelta(hours=1),
            "6h": timedelta(hours=6),
            "24h": timedelta(days=1),
            "7d": timedelta(days=7),
            "30d": timedelta(days=30),
        }

        logger.info("ContributionAnalyzer initialized (enhanced)")

    def set_dynamic_allocations(self, allocations: Dict[str, float]):
        """注入动态分配权重（从 AdaptiveController）"""
        self._dynamic_allocations = dict(allocations or {})

    # ============================================================
    # 核心分析
    # ============================================================

    def analyze(self, window: str = "24h",
                include_trend: bool = True,
                include_lifecycle: bool = True,
                include_health: bool = True,
                since_override: Optional[datetime] = None) -> ContributionSnapshot:
        """执行完整贡献度分析（强化版）

        Args:
            window: 时间窗口 (1h/6h/24h/7d/30d)
            include_trend: 是否包含趋势分析
            include_lifecycle: 是否包含生命周期分类
            include_health: 是否包含健康度评分
            since_override: 可选的分析时间线起点；非 None 时只统计该时间之后的交易
        """
        now = datetime.now()
        since = now - self._windows.get(window, self._windows["24h"])
        if since_override is not None:
            since = max(since, since_override)

        # 1. 从 SQLite 获取交易数据
        trades = self._get_trades_since(since)
        data_error = self._trade_query_error
        # 2. 按策略聚合
        strategy_data = self._aggregate_by_strategy(trades)
        # 3. 获取未实现盈亏（从持仓数据）
        unrealized_by_strategy = self._get_unrealized_pnl_by_strategy()
        for sname in strategy_data:
            strategy_data[sname]["unrealized_pnl"] = unrealized_by_strategy.get(sname, 0.0)

        # 4. 构建贡献度对象
        contributions = {}
        total_pnl = sum(d["total_pnl"] for d in strategy_data.values())
        total_fees = sum(d["total_fees"] for d in strategy_data.values())
        total_unrealized = sum(unrealized_by_strategy.values())

        for sname, data in strategy_data.items():
            contrib = StrategyContribution(
                strategy=sname,
                total_pnl=round(data["total_pnl"], 4),
                pnl_contribution_pct=round(
                    data["total_pnl"] / total_pnl * 100, 2
                ) if total_pnl != 0 else 0.0,
                realized_pnl=round(data["realized_pnl"], 4),
                unrealized_pnl=round(data.get("unrealized_pnl", 0), 4),
                total_trades=data["total_trades"],
                winning_trades=data["winning_trades"],
                losing_trades=data["losing_trades"],
                consecutive_losses=data.get("consecutive_losses", 0),
                win_rate=round(data["win_rate"], 4),
                avg_win=round(data["avg_win"], 4),
                avg_loss=round(data["avg_loss"], 4),
                profit_factor=round(data["profit_factor"], 4),
                max_drawdown=round(data["max_drawdown"], 4),
                max_drawdown_duration_hours=round(
                    data.get("max_drawdown_duration_hours", 0), 2),
                avg_capital_used=round(data["avg_capital_used"], 4),
                pnl_per_capital_pct=round(data["pnl_per_capital_pct"], 4),
                pnl_per_trade=round(data["pnl_per_trade"], 4),
                total_fees=round(data["total_fees"], 4),
                fee_ratio=round(data["fee_ratio"], 4),
                total_funding_cost=round(data.get("total_funding_cost", 0), 4),
                total_slippage_cost=round(data.get("total_slippage_cost", 0), 4),
                total_spread_cost=round(data.get("total_spread_cost", 0), 4),
                long_pnl=round(data.get("long_pnl", 0), 4),
                short_pnl=round(data.get("short_pnl", 0), 4),
                long_trades=data.get("long_trades", 0),
                short_trades=data.get("short_trades", 0),
                stop_loss_count=data.get("stop_loss_count", 0),
                stop_loss_pnl=round(data.get("stop_loss_pnl", 0), 4),
                take_profit_count=data.get("take_profit_count", 0),
                take_profit_pnl=round(data.get("take_profit_pnl", 0), 4),
                first_trade_time=data.get("first_trade_time"),
                last_trade_time=data.get("last_trade_time"),
                active_hours=round(data.get("active_hours", 0), 2),
                pnl_per_hour=round(data["pnl_per_hour"], 4),
                risk_adjusted_contribution=round(
                    data["risk_adjusted_contribution"], 4),
                sharpe_ratio=round(data["sharpe_ratio"], 4),
                volatility=round(data.get("volatility", 0), 4),
            )
            contributions[sname] = contrib

        # 5. 趋势分析
        if include_trend and window in ("24h", "7d", "30d"):
            self._compute_trends(contributions)

        # 6. 生命周期分类
        if include_lifecycle:
            self._compute_lifecycle(contributions, now)

        # 7. 健康度评分
        if include_health:
            self._compute_health_scores(contributions)

        # 8. 增量变化追踪
        self._compute_deltas(contributions)

        # 9. 相关性矩阵
        correlation = self._compute_correlation_matrix(strategy_data)

        # 10. 集中度与分散化得分
        concentration = self._compute_concentration(contributions)
        diversification = self._compute_diversification(correlation)
        efficiency = self._compute_efficiency_score(contributions)

        # 11. 协同效应
        synergy = self._compute_synergy(correlation, contributions)

        # 12. 系统整体健康度
        overall_health = self._compute_overall_health(contributions)

        # 13. 生命周期统计
        lifecycle_summary = defaultdict(int)
        for c in contributions.values():
            lifecycle_summary[c.lifecycle] += 1

        snapshot = ContributionSnapshot(
            timestamp=now.isoformat(),
            window=window,
            total_pnl=round(total_pnl, 4),
            total_trades=sum(d["total_trades"] for d in strategy_data.values()),
            total_fees=round(total_fees, 4),
            total_unrealized_pnl=round(total_unrealized, 4),
            strategies=contributions,
            correlation_matrix=correlation,
            concentration=round(concentration, 4),
            diversification_score=round(diversification, 4),
            efficiency_score=round(efficiency, 4),
            overall_health_score=round(overall_health, 2),
            synergy_score=round(synergy, 4),
            lifecycle_summary=dict(lifecycle_summary),
            available=data_error is None,
            data_error=data_error,
        )

        # 14. 缓存历史
        self._snapshot_history.append(snapshot)
        if len(self._snapshot_history) > self._max_history:
            self._snapshot_history = self._snapshot_history[-self._max_history:]

        return snapshot

    # ============================================================
    # 数据获取
    # ============================================================

    def _get_trades_since(self, since: datetime) -> List[Dict[str, Any]]:
        """获取指定时间范围内的交易记录"""
        self._trade_query_error = None
        trades = []
        try:
            if self._sqlite:
                getter = getattr(self._sqlite, "get_trade_records_checked", None)
                if getter is not None:
                    records, err = getter(limit=5000)
                else:
                    records, err = self._sqlite.get_trade_records(limit=5000), None
                if err:
                    self._trade_query_error = f"sqlite_storage.get_trade_records failed: {err}"
                    return []
                if records is None:
                    self._trade_query_error = "sqlite_storage.get_trade_records returned None"
                    return []
                for tr in records:
                    rec = self._normalize_trade_record(tr)
                    ts = self._parse_close_time(rec.get("close_time"))
                    if ts is None or ts < since:
                        continue
                    trades.append(rec)
            elif self._journal:
                getter = getattr(self._journal, "get_trades_since", None)
                if getter is None:
                    self._trade_query_error = "trade_journal.get_trades_since unavailable"
                    return []
                all_trades = getter(since)
                if all_trades is None:
                    self._trade_query_error = "trade_journal.get_trades_since returned None"
                    return []
                for tr in all_trades:
                    rec = self._normalize_trade_record(tr)
                    ts = self._parse_close_time(rec.get("close_time"))
                    if ts is None or ts < since:
                        continue
                    trades.append(rec)
            else:
                self._trade_query_error = "no data source configured (sqlite_storage/trade_journal)"
                return []
        except Exception as e:
            self._trade_query_error = f"Trade retrieval error: {e}"
            logger.debug(self._trade_query_error)
        return self._dedup_trades(trades)

    @staticmethod
    def _parse_close_time(value) -> Optional[datetime]:
        """安全解析平仓时间，解析失败返回 None。"""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if hasattr(value, "timestamp"):
            try:
                return datetime.fromtimestamp(value.timestamp())
            except (OSError, OverflowError, ValueError, TypeError):
                return None
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _normalize_trade_record(tr: Dict[str, Any]) -> Dict[str, Any]:
        """统一交易记录字段（兼容 sqlite_storage 与 trade_journal 两种口径）。"""
        if not isinstance(tr, dict):
            return {}
        rec = dict(tr)
        if rec.get("pnl") is None:
            rec["pnl"] = rec.get("pnl_usdt", 0)
        if rec.get("fee") is None:
            rec["fee"] = rec.get("fees", rec.get("commission", 0))
        # 成本细分归因：兼容 funding_cost/slippage_cost/spread_cost 及缩写别名
        rec["funding_cost"] = rec.get("funding_cost", rec.get("funding", 0))
        rec["slippage_cost"] = rec.get("slippage_cost", rec.get("slippage", 0))
        rec["spread_cost"] = rec.get("spread_cost", rec.get("spread", 0))
        if rec.get("close_time") is None:
            rec["close_time"] = rec.get("exit_time")
        if rec.get("strategy_name") is None:
            rec["strategy_name"] = rec.get("strategy", "unknown")
        if "status" not in rec:
            rec["status"] = "closed"
        return rec

    @staticmethod
    def _dedup_trades(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按交易 id 去重，避免重复统计。"""
        seen = set()
        result = []
        for tr in trades:
            if not isinstance(tr, dict):
                continue
            key = None
            for k in ("id", "trade_id", "tradeId"):
                v = tr.get(k)
                if v is not None:
                    key = v
                    break
            if key is None:
                result.append(tr)
                continue
            if key in seen:
                continue
            seen.add(key)
            result.append(tr)
        return result

    def _aggregate_by_strategy(self,
                                trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """按策略聚合交易数据"""
        data = defaultdict(lambda: {
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "pnl_list": [],
            "total_fees": 0.0,
            "total_funding_cost": 0.0,
            "total_slippage_cost": 0.0,
            "total_spread_cost": 0.0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "stop_loss_count": 0,
            "stop_loss_pnl": 0.0,
            "take_profit_count": 0,
            "take_profit_pnl": 0.0,
            "equity_curve": [0.0],
            "drawdown_durations": [],
            "trade_times": [],
            "equity_timeline": [],
        })

        for tr in trades:
            tr = self._normalize_trade_record(tr)
            if not tr:
                continue
            sname = tr.get("strategy_name") or tr.get("strategy") or "unknown"
            pnl = _safe_float(tr.get("pnl"))
            fee = _safe_float(tr.get("fee"))
            funding = _safe_float(tr.get("funding_cost"))
            slippage = _safe_float(tr.get("slippage_cost"))
            spread = _safe_float(tr.get("spread_cost"))
            direction = (tr.get("direction") or tr.get("side") or "").lower()
            exit_reason = (tr.get("exit_reason") or tr.get("close_reason") or "").lower()
            status = tr.get("status", "")

            d = data[sname]
            d["total_pnl"] += pnl
            d["total_fees"] += fee
            d["total_funding_cost"] += funding
            d["total_slippage_cost"] += slippage
            d["total_spread_cost"] += spread
            d["pnl_list"].append(pnl)

            # 方向归因（long / short 分解）
            if direction == "long":
                d["long_pnl"] += pnl
                d["long_trades"] += 1
            elif direction == "short":
                d["short_pnl"] += pnl
                d["short_trades"] += 1

            # 退出原因归因（止损 / 止盈 分解）
            if exit_reason in ("stop_loss", "stoploss", "sl", "止损"):
                d["stop_loss_count"] += 1
                d["stop_loss_pnl"] += pnl
            elif exit_reason in ("take_profit", "takeprofit", "tp", "止盈"):
                d["take_profit_count"] += 1
                d["take_profit_pnl"] += pnl

            if status == "closed":
                d["total_trades"] += 1
                if pnl > 0:
                    d["winning_trades"] += 1
                elif pnl < 0:
                    d["losing_trades"] += 1

            if pnl != 0:
                d["realized_pnl"] += pnl

            # 权益曲线
            if d["equity_curve"]:
                d["equity_curve"].append(d["equity_curve"][-1] + pnl)
            else:
                d["equity_curve"] = [pnl]

            # 交易时间
            ts = self._parse_close_time(tr.get("close_time"))
            if ts is not None:
                d["trade_times"].append(ts)
                # 权益时间线（供回撤持续时长计算：与 equity_curve 对齐，仅含有效平仓时间的交易）
                d["equity_timeline"].append((ts, d["equity_curve"][-1]))

        # 计算派生指标
        for sname, d in data.items():
            n = d["total_trades"]
            pnl_list = d["pnl_list"]

            d["win_rate"] = d["winning_trades"] / n if n > 0 else 0.0

            wins = [p for p in pnl_list if p > 0]
            losses = [p for p in pnl_list if p < 0]
            d["avg_win"] = np.mean(wins) if wins else 0.0
            d["avg_loss"] = abs(np.mean(losses)) if losses else 1.0
            d["profit_factor"] = (d["avg_win"] / d["avg_loss"]
                                  if d["avg_loss"] > 0 and d["avg_win"] > 0
                                  else (2.0 if d["winning_trades"] > d["losing_trades"]
                                        else 0.5))

            # 真实夏普比率（mean/std，与 allocation_agent / DataAnalysisEngine 口径一致）。
            # 同时暴露波动率（单笔盈亏标准差）：独立于夏普的风险维度——夏普看单位风险收益，
            # 波动率看绝对不确定性，二者正交（夏普持平但波动率上升 = 承担更多风险换同等收益）。
            if len(pnl_list) >= 2:
                pnl_arr = np.array(pnl_list, dtype=float)
                pnl_std = float(np.std(pnl_arr))
                d["volatility"] = pnl_std
                d["sharpe_ratio"] = float(np.mean(pnl_arr) / pnl_std) if pnl_std > 0 else 0.0
            else:
                d["volatility"] = 0.0
                d["sharpe_ratio"] = 0.0

            # 最大回撤
            d["max_drawdown"] = self._calc_max_drawdown(d["equity_curve"])

            # 最大回撤持续时长（小时）：独立于回撤深度（max_drawdown）的「资金时间价值」维度——
            # 策略可能回撤不深但长时间无法收复前高，资金被套牢、恢复能力差。
            d["max_drawdown_duration_hours"] = self._calc_max_drawdown_duration_hours(
                d["equity_timeline"]
            )

            # 最大连续亏损（负期望早期熔断信号）
            d["consecutive_losses"] = self._calc_consecutive_losses(d["pnl_list"])

            # 平均占用资金
            alloc = _safe_float(self._dynamic_allocations.get(sname), 0.2)
            trading_cfg = self.config.get("trading") or {}
            total_capital = _safe_float(trading_cfg.get("total_capital", 559), 559.0)
            d["avg_capital_used"] = total_capital * alloc

            d["pnl_per_capital_pct"] = (d["total_pnl"] / d["avg_capital_used"] * 100
                                         if d["avg_capital_used"] > 0 else 0)
            d["pnl_per_trade"] = d["total_pnl"] / n if n > 0 else 0
            d["fee_ratio"] = (d["total_fees"] / abs(d["total_pnl"]) * 100
                              if abs(d["total_pnl"]) > 0 else 0)

            # 活跃时长
            if d["trade_times"]:
                d["first_trade_time"] = min(d["trade_times"]).isoformat()
                d["last_trade_time"] = max(d["trade_times"]).isoformat()
                span = (max(d["trade_times"]) - min(d["trade_times"])).total_seconds()
                d["active_hours"] = span / 3600
                d["pnl_per_hour"] = (d["total_pnl"] / d["active_hours"]
                                      if d["active_hours"] > 0 else 0)
            else:
                d["active_hours"] = 0
                d["pnl_per_hour"] = 0

            # 风险调整后贡献
            dd = d["max_drawdown"]
            d["risk_adjusted_contribution"] = (d["total_pnl"] / dd
                                                if dd > 0 else d["total_pnl"])

        return dict(data)

    # ============================================================
    # 辅助计算
    # ============================================================

    @staticmethod
    def _calc_max_drawdown(equity_curve: List[float]) -> float:
        """计算最大回撤（结果 clamp 到 [0,1]）。

        权益曲线为累计 PnL（从 0 起），可为负；当曲线先正后深负时
        (peak-val)/peak 会 >1（曾产出 119.48 = 11948%），污染 stability_score
        与 risk_adjusted_contribution。此处 clamp 到 [0,1]。
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        peak = equity_curve[0]
        max_dd = 0.0
        for val in equity_curve:
            peak = max(peak, val)
            if peak > 0:
                dd = max(0.0, min(1.0, (peak - val) / peak))
                max_dd = max(max_dd, dd)
        return max_dd

    @staticmethod
    def _calc_max_drawdown_duration_hours(timeline: List[Tuple[Any, float]]) -> float:
        """计算最大回撤持续时长（小时）：权益从峰值回落到恢复峰值的最长时间跨度。

        timeline 为 [(close_time, cumulative_pnl), ...] 时间升序序列（仅含有效平仓时间的
        交易），权益为累计 PnL。用「时间跨度」度量回撤持续——策略可能回撤不深但长时间
        无法收复前高，资金被套牢、恢复能力差，这是独立于回撤深度（max_drawdown）的
        「资金时间价值」维度。时间解析失败或类型不支持时该段时长记为 0（不污染结果）。
        """
        if not timeline or len(timeline) < 2:
            return 0.0

        def _hours(a, b):
            try:
                return max(0.0, (b - a).total_seconds() / 3600.0)
            except (TypeError, AttributeError):
                return 0.0

        peak = timeline[0][1]
        peak_time = timeline[0][0]
        dd_start = None
        max_dur = 0.0

        for t, val in timeline[1:]:
            if val >= peak:
                # 恢复或创新高：结束回撤区间，并推进峰值时间（下次回撤起点从当前算起）
                if dd_start is not None:
                    max_dur = max(max_dur, _hours(dd_start, t))
                    dd_start = None
                if val > peak:
                    peak = val
                peak_time = t
            else:
                # 回撤中：记录起点（从最近一次峰值时间算起）
                if dd_start is None:
                    dd_start = peak_time

        # 未恢复的尾部回撤（截至最后一笔交易）
        if dd_start is not None and timeline:
            max_dur = max(max_dur, _hours(dd_start, timeline[-1][0]))
        return max_dur

    @staticmethod
    def _calc_consecutive_losses(pnl_list: List[float]) -> int:
        """计算最大连续亏损笔数（负期望策略早期熔断信号）。"""
        max_streak = 0
        streak = 0
        for pnl in pnl_list or []:
            if pnl < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0
        return max_streak

    def _compute_trends(self, contributions: Dict[str, StrategyContribution]):
        """计算贡献趋势（对比不同窗口）"""
        now = datetime.now()
        window_7d = now - timedelta(days=7)
        window_30d = now - timedelta(days=30)

        trades_7d = self._get_trades_since(window_7d)
        trades_30d = self._get_trades_since(window_30d)

        data_7d = self._aggregate_by_strategy(trades_7d)
        data_30d = self._aggregate_by_strategy(trades_30d)

        for sname, contrib in contributions.items():
            pnl_7d = data_7d.get(sname, {}).get("total_pnl", 0)
            pnl_30d = data_30d.get(sname, {}).get("total_pnl", 0)
            avg_30d = pnl_30d / 30 if pnl_30d != 0 else 0
            avg_7d = pnl_7d / 7 if pnl_7d != 0 else 0

            if avg_30d != 0 and avg_7d != 0:
                contrib.trend_pnl_7d_vs_30d = round(avg_7d / avg_30d, 4) if avg_30d > 0 else round(avg_7d / max(abs(avg_30d), 0.01), 4)
            elif avg_7d != 0:
                contrib.trend_pnl_7d_vs_30d = 2.0  # 从无到有，视为改善
            else:
                contrib.trend_pnl_7d_vs_30d = 0.0

            if contrib.trend_pnl_7d_vs_30d > 1.2:
                contrib.trend = "improving"
            elif contrib.trend_pnl_7d_vs_30d < 0.7:
                contrib.trend = "declining"
            else:
                contrib.trend = "stable"

    def _compute_correlation_matrix(self,
                                     strategy_data: Dict[str, Dict[str, Any]]
                                     ) -> Dict[str, Dict[str, float]]:
        """计算策略间 PnL 序列的相关性矩阵"""
        matrix = {}
        strategies = list(strategy_data.keys())
        if len(strategies) < 2:
            return matrix

        for s1 in strategies:
            matrix[s1] = {}
            pnl1 = strategy_data[s1].get("pnl_list", [])
            for s2 in strategies:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                    continue
                pnl2 = strategy_data[s2].get("pnl_list", [])
                corr = self._calc_correlation(pnl1, pnl2)
                matrix[s1][s2] = round(corr, 4)

        return matrix

    @staticmethod
    def _calc_correlation(a: List[float], b: List[float]) -> float:
        """计算皮尔逊相关系数"""
        n = min(len(a), len(b))
        if n < 3:
            return 0.0
        a_arr = np.array(a[:n], dtype=float)
        b_arr = np.array(b[:n], dtype=float)
        std_a = np.std(a_arr)
        std_b = np.std(b_arr)
        if not np.isfinite(std_a) or not np.isfinite(std_b) or std_a == 0 or std_b == 0:
            return 0.0
        corr = float(np.corrcoef(a_arr, b_arr)[0, 1])
        return corr if np.isfinite(corr) else 0.0

    def _compute_concentration(self,
                                contributions: Dict[str, StrategyContribution]
                                ) -> float:
        """Herfindahl-Hirschman Index (HHI) 计算集中度（基于资金配置权重）。

        HHI = sum(weight_i^2)，weight_i 为归一化后的资金配置权重（非负、和=1），
        取值 ∈ [1/N, 1]。不再使用 PnL 份额：PnL 可为负、总和近 0，导致 HHI 无界
        （曾出现 300+，告警显示 30055%）。
        """
        alloc = self._dynamic_allocations or {}
        weights = [w for w in alloc.values() if _safe_float(w, 0.0) > 0]
        if weights:
            total = sum(weights)
            if total > 0:
                hhi = sum((w / total) ** 2 for w in weights)
                n = len(weights)
                return max(1.0 / max(n, 1), min(1.0, hhi))

        # 回退：无正权重时等权（最分散）
        n = max(len(contributions), 1)
        return 1.0 / n

    def _compute_diversification(self,
                                  correlation: Dict[str, Dict[str, float]]
                                  ) -> float:
        """基于相关性的分散化得分（0=完全同向, 1=完全分散）"""
        if not correlation or len(correlation) < 2:
            return 0.5

        # 计算平均绝对交叉相关性
        all_corrs = []
        strategies = list(correlation.keys())
        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1:]:
                corr = correlation.get(s1, {}).get(s2, 0)
                all_corrs.append(abs(corr))

        if not all_corrs:
            return 0.5

        avg_corr = np.mean(all_corrs)
        # 低相关性 → 高分散化得分
        return round(max(0.0, min(1.0, 1.0 - avg_corr)), 4)

    @staticmethod
    def _compute_efficiency_score(contributions: Dict[str, StrategyContribution]
                                   ) -> float:
        """整体资金效率得分"""
        if not contributions:
            return 0.0

        scores = []
        for c in contributions.values():
            if c.total_trades < 1:
                continue
            # 综合费率 + 资本回报 + 风险调整
            fee_score = max(0, 1 - c.fee_ratio / 50)  # 手续费<50%时得分递减
            capital_score = min(1, max(0, c.pnl_per_capital_pct / 10))  # 10%回报=满分
            risk_score = min(1, max(0, c.risk_adjusted_contribution / 5))

            composite = fee_score * 0.3 + capital_score * 0.4 + risk_score * 0.3
            scores.append(composite)

        return round(np.mean(scores) if scores else 0.0, 4)

    # ============================================================
    # 【强化】未实现盈亏
    # ============================================================

    def _get_unrealized_pnl_by_strategy(self) -> Dict[str, float]:
        """从持仓数据获取各策略的未实现盈亏"""
        result = {}
        try:
            # 方式1：从 OKX 客户端获取持仓
            if self._okx_client:
                positions = self._okx_client.get_positions()
                if positions:
                    for pos in positions:
                        upl = _safe_float(pos.get("upl"))
                        if upl == 0:
                            continue
                        # 从持仓推断策略（通过 posSide 和 instId）
                        inst_id = pos.get("instId", "")
                        # 通过 trade_records 反查策略
                        sname = self._resolve_strategy_for_position(inst_id)
                        result[sname] = result.get(sname, 0.0) + upl

            # 方式2：从 trade_journal 获取活跃持仓
            if not result and self._journal:
                active = self._journal.get_active_positions()
                if active:
                    for pos in active:
                        sname = pos.get("strategy_name", pos.get("strategy", "unknown"))
                        upl = _safe_float(pos.get("unrealized_pnl"))
                        result[sname] = result.get(sname, 0.0) + upl

            # 方式3：从 SQLite 查询 open 状态的交易
            if not result and self._sqlite:
                try:
                    records = self._sqlite.get_trade_records(limit=1000)
                    for tr in records:
                        if tr.get("status") == "open":
                            sname = tr.get("strategy_name", tr.get("strategy", "unknown"))
                            upl = _safe_float(tr.get("unrealized_pnl"))
                            result[sname] = result.get(sname, 0.0) + upl
                except Exception as e:
                    logger.debug(f"Unrealized PnL via sqlite failed: {e}")
        except Exception as e:
            logger.debug(f"Unrealized PnL query error: {e}")

        return result

    def _resolve_strategy_for_position(self, inst_id: str) -> str:
        """通过 trade_records 反查持仓对应的策略"""
        try:
            if self._sqlite:
                records = self._sqlite.get_trade_records(limit=100)
                for tr in records:
                    if tr.get("symbol") == inst_id and tr.get("status") == "open":
                        return tr.get("strategy_name", tr.get("strategy", "unknown"))
        except Exception as e:
            logger.debug(f"Resolve strategy for position {inst_id} failed: {e}")
        return "unknown"

    # ============================================================
    # 【强化】生命周期分类
    # ============================================================

    def _compute_lifecycle(self, contributions: Dict[str, StrategyContribution],
                            now: datetime):
        """分类策略生命周期：newborn / mature / declining / dormant"""
        cfg = self._lifecycle_config

        for sname, c in contributions.items():
            # 计算首次交易至今的天数
            age_days = 0.0
            if c.first_trade_time:
                try:
                    first_dt = datetime.fromisoformat(c.first_trade_time)
                    age_days = (now - first_dt).total_seconds() / 86400
                except (ValueError, TypeError):
                    pass
            c.lifecycle_age_days = round(age_days, 1)

            # 计算距上次交易的小时数
            idle_hours = 0.0
            if c.last_trade_time:
                try:
                    last_dt = datetime.fromisoformat(c.last_trade_time)
                    idle_hours = (now - last_dt).total_seconds() / 3600
                except (ValueError, TypeError):
                    pass
            c.lifecycle_last_trade_age_hours = round(idle_hours, 1)

            # 判断生命周期
            if c.total_trades == 0:
                c.lifecycle = "dormant"
            elif idle_hours > _safe_float(cfg.get("dormant_max_idle_hours"), 72.0):
                c.lifecycle = "dormant"
            elif age_days <= _safe_float(cfg.get("newborn_max_days"), 7.0):
                c.lifecycle = "newborn"
            elif c.trend == "declining" and c.trend_pnl_7d_vs_30d < _safe_float(cfg.get("declining_7d_vs_30d_ratio"), 0.5):
                c.lifecycle = "declining"
            elif c.total_trades >= _safe_float(cfg.get("mature_min_trades"), 10.0) and c.win_rate > 0:
                c.lifecycle = "mature"
            elif c.trend == "declining":
                c.lifecycle = "declining"
            else:
                c.lifecycle = "mature"

    # ============================================================
    # 【强化】健康度评分
    # ============================================================

    def _compute_health_scores(self, contributions: Dict[str, StrategyContribution]):
        """计算综合健康度评分（0-100分）"""
        w = self._health_weights

        for sname, c in contributions.items():
            if c.total_trades < self._min_health_sample:
                c.health_score = 0.0
                c.health_grade = "N/A"
                continue

            # 1. 盈亏比得分 (0-100)
            pf_score = min(100, c.profit_factor * 50) if c.profit_factor > 0 else 0

            # 2. 胜率得分 (0-100)
            wr_score = min(100, c.win_rate * 100)

            # 3. 风险调整得分 (0-100) — 负贡献 clamp 到 0，防健康度越界为负
            ra_score = max(0.0, min(100.0, c.risk_adjusted_contribution * 20))

            # 4. 资金效率得分 (0-100) — 负资本回报 clamp 到 0，防健康度越界为负
            ce_score = max(0.0, min(100.0, c.pnl_per_capital_pct * 10))

            # 5. 费率效率得分 (0-100) — 费率越低越好
            fe_score = max(0, 100 - c.fee_ratio * 2)

            # 6. 趋势得分 (0-100)
            trend_map = {"improving": 90, "stable": 60, "declining": 25}
            trend_score = trend_map.get(c.trend, 50)

            # 7. 稳定性得分 (0-100) — 基于回撤
            stability_score = max(0, 100 - c.max_drawdown * 200)

            # 综合加权
            composite = (
                pf_score * w.get("profit_factor", 0.20) +
                wr_score * w.get("win_rate", 0.15) +
                ra_score * w.get("risk_adjusted", 0.20) +
                ce_score * w.get("capital_efficiency", 0.15) +
                fe_score * w.get("fee_efficiency", 0.10) +
                trend_score * w.get("trend", 0.10) +
                stability_score * w.get("stability", 0.10)
            )

            c.health_score = round(max(0.0, min(100.0, composite)), 1)

            # 等级映射
            if composite >= 80:
                c.health_grade = "A"
            elif composite >= 60:
                c.health_grade = "B"
            elif composite >= 45:
                c.health_grade = "C"
            elif composite >= 25:
                c.health_grade = "D"
            else:
                c.health_grade = "F"

    # ============================================================
    # 【强化】增量变化追踪
    # ============================================================

    def _compute_deltas(self, contributions: Dict[str, StrategyContribution]):
        """计算相对于上一快照的增量变化"""
        if len(self._snapshot_history) < 1:
            return

        prev = self._snapshot_history[-1]
        for sname, c in contributions.items():
            if sname in prev.strategies:
                prev_c = prev.strategies[sname]
                c.delta_pnl = round(c.total_pnl - prev_c.total_pnl, 4)
                c.delta_contribution_pct = round(
                    c.pnl_contribution_pct - prev_c.pnl_contribution_pct, 2)
                c.delta_health = round(c.health_score - prev_c.health_score, 1)
            else:
                c.delta_pnl = c.total_pnl
                c.delta_contribution_pct = c.pnl_contribution_pct

    # ============================================================
    # 【强化】协同效应分析
    # ============================================================

    def _compute_synergy(self,
                          correlation: Dict[str, Dict[str, float]],
                          contributions: Dict[str, StrategyContribution]) -> float:
        """计算策略协同效应得分

        低相关性 + 正PnL贡献 = 高协同（策略有互补效果）
        高相关性 + 正PnL贡献 = 中等协同
        高相关性 + 负PnL贡献 = 低协同（策略相互拖累）
        """
        if not correlation or len(correlation) < 2:
            return 0.5

        strategies = list(correlation.keys())
        synergy_scores = []

        for i, s1 in enumerate(strategies):
            for s2 in strategies[i + 1:]:
                corr = correlation.get(s1, {}).get(s2, 0)
                c1 = contributions.get(s1)
                c2 = contributions.get(s2)
                if not c1 or not c2:
                    continue

                # 两策略都盈利 = 好组合
                both_profitable = c1.total_pnl > 0 and c2.total_pnl > 0
                # 低相关性 = 分散化加分
                diversification = 1 - abs(corr)

                if both_profitable and diversification > 0.5:
                    synergy_scores.append(1.0)  # 完美互补
                elif both_profitable:
                    synergy_scores.append(0.7)  # 同向盈利
                elif diversification > 0.5:
                    synergy_scores.append(0.5)  # 分散但盈亏不一
                else:
                    synergy_scores.append(0.2)  # 低协同

        return round(np.mean(synergy_scores), 4) if synergy_scores else 0.5

    # ============================================================
    # 【强化】系统整体健康度
    # ============================================================

    def _compute_overall_health(self,
                                 contributions: Dict[str, StrategyContribution]
                                 ) -> float:
        """计算系统整体健康度（加权平均）"""
        if not contributions:
            return 0.0

        active_contribs = [c for c in contributions.values()
                           if c.total_trades > 0 and c.health_score > 0]
        if not active_contribs:
            return 0.0

        # 按交易量加权
        total_trades = sum(c.total_trades for c in active_contribs)
        if total_trades == 0:
            return np.mean([c.health_score for c in active_contribs])

        weighted = sum(c.health_score * c.total_trades for c in active_contribs) / total_trades
        return round(weighted, 2)

    # ============================================================
    # 多窗口分析
    # ============================================================

    def analyze_all_windows(self) -> Dict[str, ContributionSnapshot]:
        """分析所有时间窗口"""
        results = {}
        for window in self._windows:
            try:
                results[window] = self.analyze(window=window, include_trend=False)
            except Exception as e:
                logger.warning(f"Contribution analysis failed for window '{window}': {e}")
        return results

    # ============================================================
    # 【强化】多窗口对比
    # ============================================================

    def compare_windows(self, windows: List[str] = None) -> Dict[str, Any]:
        """多时间窗口贡献度对比分析

        对比不同窗口下的各策略表现变化，识别：
        - 短期vs长期表现背离的策略
        - 哪个窗口贡献度变化最大
        - 风险收益比随时间变化的策略
        """
        if windows is None:
            windows = ["1h", "6h", "24h", "7d"]

        snapshots = {}
        for w in windows:
            try:
                snapshots[w] = self.analyze(window=w, include_trend=False,
                                            include_lifecycle=False, include_health=False)
            except Exception as e:
                logger.warning(f"Compare window '{w}' failed: {e}")

        if not snapshots:
            return {"error": "No valid window snapshots"}

        # 构建对比矩阵
        comparison = {
            "windows": windows,
            "by_strategy": {},
            "aggregate": {
                "total_pnl": {w: snapshots[w].total_pnl for w in snapshots},
                "total_trades": {w: snapshots[w].total_trades for w in snapshots},
                "total_fees": {w: snapshots[w].total_fees for w in snapshots},
                "concentration": {w: snapshots[w].concentration for w in snapshots},
                "diversification": {w: snapshots[w].diversification_score for w in snapshots},
                "efficiency": {w: snapshots[w].efficiency_score for w in snapshots},
            },
        }

        # 每个策略在各窗口的表现
        all_strategies = set()
        for s in snapshots.values():
            all_strategies.update(s.strategies.keys())

        for sname in all_strategies:
            comp = {"pnl": {}, "contribution_pct": {}, "trades": {},
                     "win_rate": {}, "profit_factor": {}, "fee_ratio": {}}
            for w, snap in snapshots.items():
                c = snap.strategies.get(sname)
                if c:
                    comp["pnl"][w] = c.total_pnl
                    comp["contribution_pct"][w] = c.pnl_contribution_pct
                    comp["trades"][w] = c.total_trades
                    comp["win_rate"][w] = c.win_rate
                    comp["profit_factor"][w] = c.profit_factor
                    comp["fee_ratio"][w] = c.fee_ratio

            # 判断短期vs长期背离
            if "1h" in comp["pnl"] and "7d" in comp["pnl"]:
                pnl_1h = comp["pnl"]["1h"]
                pnl_7d = comp["pnl"]["7d"]
                if pnl_1h > 0 and pnl_7d < 0:
                    comp["alert"] = "短期盈利但长期亏损 → 可能是运气/过拟合"
                elif pnl_1h < 0 and pnl_7d > 0:
                    comp["alert"] = "短期亏损但长期盈利 → 可能是临时回调"
            comparison["by_strategy"][sname] = comp

        return comparison

    # ============================================================
    # 【强化】健康度摘要
    # ============================================================

    def get_health_summary(self, since_override: Optional[datetime] = None) -> Dict[str, Any]:
        """获取系统健康度摘要（供 Dashboard 消费）"""
        snapshot = self.analyze(window="7d", since_override=since_override)
        strategies_health = []
        alerts = []

        for sname, c in snapshot.strategies.items():
            health_entry = {
                "strategy": sname,
                "health_score": c.health_score,
                "health_grade": c.health_grade,
                "lifecycle": c.lifecycle,
                "total_pnl": c.total_pnl,
                "pnl_contribution_pct": c.pnl_contribution_pct,
                "win_rate": c.win_rate,
                "profit_factor": c.profit_factor,
                "trend": c.trend,
                "delta_health": c.delta_health,
                "delta_pnl": c.delta_pnl,
                # 方向归因
                "long_pnl": c.long_pnl,
                "short_pnl": c.short_pnl,
                "long_trades": c.long_trades,
                "short_trades": c.short_trades,
                # 成本细分归因
                "total_fees": c.total_fees,
                "total_funding_cost": c.total_funding_cost,
                "total_slippage_cost": c.total_slippage_cost,
                "total_spread_cost": c.total_spread_cost,
                # 退出原因归因
                "stop_loss_count": c.stop_loss_count,
                "stop_loss_pnl": c.stop_loss_pnl,
                "take_profit_count": c.take_profit_count,
                "take_profit_pnl": c.take_profit_pnl,
            }
            strategies_health.append(health_entry)

            # 告警生成
            if c.health_grade == "F":
                alerts.append({
                    "strategy": sname,
                    "level": "critical",
                    "message": f"{sname} 健康度评分为F（{c.health_score}分），建议立即暂停审查",
                    "score": c.health_score,
                })
            elif c.health_grade == "D" and c.trend == "declining":
                alerts.append({
                    "strategy": sname,
                    "level": "warning",
                    "message": f"{sname} 健康度D级且趋势恶化，需密切关注",
                    "score": c.health_score,
                })
            elif c.lifecycle == "dormant" and c.total_trades > 5:
                alerts.append({
                    "strategy": sname,
                    "level": "info",
                    "message": f"{sname} 已进入休眠状态，距上次交易{c.lifecycle_last_trade_age_hours:.0f}小时",
                })

        # 按健康度排序（低的在前）
        strategies_health.sort(key=lambda x: x["health_score"])

        return {
            "timestamp": snapshot.timestamp,
            "available": snapshot.available,
            "data_error": snapshot.data_error,
            "overall_health": snapshot.overall_health_score,
            "overall_health_grade": self._health_grade_for(snapshot.overall_health_score),
            "total_pnl": snapshot.total_pnl,
            "total_unrealized_pnl": snapshot.total_unrealized_pnl,
            "synergy_score": snapshot.synergy_score,
            "diversification_score": snapshot.diversification_score,
            "efficiency_score": snapshot.efficiency_score,
            "concentration": snapshot.concentration,
            "lifecycle_summary": snapshot.lifecycle_summary,
            "strategies": strategies_health,
            "alerts": alerts,
            "top_performer": strategies_health[-1] if strategies_health else None,
            "worst_performer": strategies_health[0] if strategies_health else None,
        }

    @staticmethod
    def _health_grade_for(score: float) -> str:
        if score >= 80:
            return "A"
        elif score >= 60:
            return "B"
        elif score >= 45:
            return "C"
        elif score >= 25:
            return "D"
        return "F"

    def get_contribution_timeline(self, hours: int = 168) -> List[Dict[str, Any]]:
        """获取贡献度时间线（用于趋势图）

        返回每个小时的各策略 PnL 贡献汇总
        """
        now = datetime.now()
        since = now - timedelta(hours=hours)
        trades = self._get_trades_since(since)

        # 按时段分组
        hourly = defaultdict(lambda: defaultdict(float))
        for tr in trades:
            tr = self._normalize_trade_record(tr)
            if not tr:
                continue
            ts = self._parse_close_time(tr.get("close_time"))
            if ts is None:
                continue

            hour_key = ts.strftime("%Y-%m-%dT%H:00")
            sname = tr.get("strategy_name") or tr.get("strategy") or "unknown"
            pnl = _safe_float(tr.get("pnl"))
            hourly[hour_key][sname] += pnl

        # 排序输出
        timeline = []
        for hour_key in sorted(hourly.keys()):
            entry = {"hour": hour_key, "strategies": dict(hourly[hour_key])}
            entry["total_pnl"] = round(sum(hourly[hour_key].values()), 4)
            timeline.append(entry)

        return timeline

    # ============================================================
    # 排名与推荐
    # ============================================================

    def get_strategy_ranking(self, metric: str = "total_pnl",
                             since_override: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """按指定指标对策略排名

        Args:
            metric: total_pnl / risk_adjusted / capital_efficiency / win_rate /
                    profit_factor / health_score / synergy_contribution
        """
        snapshot = self.analyze(window="7d", include_trend=False,
                                since_override=since_override)
        strategies = list(snapshot.strategies.values())

        if metric == "risk_adjusted":
            key = lambda c: c.risk_adjusted_contribution
        elif metric == "capital_efficiency":
            key = lambda c: c.pnl_per_capital_pct
        elif metric == "win_rate":
            key = lambda c: c.win_rate
        elif metric == "profit_factor":
            key = lambda c: c.profit_factor
        elif metric == "health_score":
            key = lambda c: c.health_score
        elif metric == "synergy_contribution":
            key = lambda c: c.total_pnl * (1 - c.fee_ratio / 100)
        else:
            key = lambda c: c.total_pnl

        ranked = sorted(strategies, key=key, reverse=True)

        return [
            {
                "rank": i + 1,
                "strategy": c.strategy,
                "total_pnl": c.total_pnl,
                "pnl_contribution_pct": c.pnl_contribution_pct,
                "win_rate": c.win_rate,
                "profit_factor": c.profit_factor,
                "pnl_per_capital_pct": c.pnl_per_capital_pct,
                "risk_adjusted_contribution": c.risk_adjusted_contribution,
                "health_score": c.health_score,
                "health_grade": c.health_grade,
                "lifecycle": c.lifecycle,
                "trend": c.trend,
            }
            for i, c in enumerate(ranked)
        ]

    def _normalized_allocation_pool(self) -> Dict[str, float]:
        """返回归一化后的资金配置权重（仅正权重、和=1.0）。

        仅包含真实资金池策略；manual_override 等非策略标签不在 _dynamic_allocations
        中，天然被排除。负值/零值剔除，剩余正权重归一化到 1.0。
        """
        result: Dict[str, float] = {}
        total = 0.0
        for k, v in (self._dynamic_allocations or {}).items():
            w = _safe_float(v, 0.0)
            if w > 0:
                result[k] = w
                total += w
        if total <= 0:
            return result
        return {k: w / total for k, w in result.items()}

    def get_capital_reallocation_suggestions(self, since_override: Optional[datetime] = None,
                                              snapshot: Optional[ContributionSnapshot] = None) -> List[Dict[str, Any]]:
        """基于贡献度分析生成资金重分配建议（强化版）

        规则：
        - 高健康度 + 改善趋势 → 建议增配
        - 低健康度 + 恶化趋势 → 建议减配
        - 高效率高贡献 → 建议作为核心策略
        - 休眠策略 → 建议回收资金

        当传入 snapshot 时复用该快照（避免与调用方感知快照口径不一致），
        否则回退到 analyze(window="7d")。
        """
        if snapshot is None:
            snapshot = self.analyze(window="7d", since_override=since_override)
        suggestions = []

        pool_weights = self._normalized_allocation_pool()

        for sname, contrib in snapshot.strategies.items():
            if sname not in pool_weights:
                continue  # 非资金池标签（manual_override / 未配置权重），不参与重配
            current_alloc = pool_weights[sname]
            suggestion = {
                "strategy": sname,
                "current_allocation": round(current_alloc, 4),
                "contribution_pct": contrib.pnl_contribution_pct,
                "health_score": contrib.health_score,
                "health_grade": contrib.health_grade,
                "lifecycle": contrib.lifecycle,
                "efficiency": round(contrib.pnl_per_capital_pct, 2),
                "trend": contrib.trend,
                "action": "hold",
                "target_allocation": round(current_alloc, 4),
                "reason": "",
            }

            # 判断规则
            is_high_health = contrib.health_grade in ("A", "B")
            is_low_health = contrib.health_grade in ("D", "F")
            is_high_contrib = contrib.pnl_contribution_pct > 25
            is_low_contrib = contrib.pnl_contribution_pct < 5 and contrib.total_trades > 3
            is_improving = contrib.trend == "improving"
            is_declining = contrib.trend == "declining"
            is_efficient = contrib.pnl_per_capital_pct > 5
            is_dormant = contrib.lifecycle == "dormant"

            if is_dormant and contrib.total_trades > 5:
                suggestion["action"] = "reclaim"
                suggestion["target_allocation"] = round(current_alloc * 0.1, 4)
                suggestion["reason"] = "策略休眠 → 回收大部分资金到活跃策略"
            elif is_high_health and is_improving and is_efficient:
                suggestion["action"] = "increase"
                suggestion["target_allocation"] = round(
                    min(current_alloc * 1.25, 0.5), 4)
                suggestion["reason"] = "A/B级健康度+改善趋势+高效率 → 核心策略，建议增配25%"
            elif is_low_health and is_declining:
                suggestion["action"] = "reduce"
                suggestion["target_allocation"] = round(
                    max(current_alloc * 0.5, 0.03), 4)
                suggestion["reason"] = "D/F级健康度+恶化趋势 → 建议大幅减配50%或暂停"
            elif is_low_health and not is_declining:
                suggestion["action"] = "reduce"
                suggestion["target_allocation"] = round(
                    max(current_alloc * 0.75, 0.05), 4)
                suggestion["reason"] = f"{contrib.health_grade}级健康度 → 建议减配25%观察"
            elif is_high_contrib and is_declining:
                suggestion["action"] = "reduce"
                suggestion["target_allocation"] = round(
                    max(current_alloc * 0.8, 0.05), 4)
                suggestion["reason"] = "高贡献但趋势恶化 → 建议减配控制风险"
            elif is_low_contrib and is_declining:
                suggestion["action"] = "reduce"
                suggestion["target_allocation"] = round(
                    max(current_alloc * 0.5, 0.03), 4)
                suggestion["reason"] = "低贡献+恶化趋势 → 大幅减配或考虑暂停"
            elif is_low_contrib and contrib.total_trades < 3:
                suggestion["action"] = "hold"
                suggestion["reason"] = "交易量不足，暂不调整"
            elif is_high_contrib and is_improving:
                suggestion["action"] = "increase"
                suggestion["target_allocation"] = round(
                    min(current_alloc * 1.15, 0.5), 4)
                suggestion["reason"] = "高贡献+改善趋势 → 建议增配15%"
            elif contrib.lifecycle == "newborn":
                suggestion["action"] = "hold"
                suggestion["reason"] = "新生策略，给予更多观察时间"

            suggestions.append(suggestion)

        return suggestions

    # ============================================================
    # 持久化
    # ============================================================

    def persist_snapshot(self, snapshot: ContributionSnapshot = None) -> bool:
        """持久化贡献度快照到 JSON 文件（fail-closed：成功返回 True，失败返回 False）"""
        if snapshot is None:
            snapshot = self.analyze(window="24h")

        try:
            data_dir = os.path.join("data", "contribution")
            os.makedirs(data_dir, exist_ok=True)

            # 最新快照
            latest_path = os.path.join(data_dir, "contribution_latest.json")
            with open(latest_path, "w", encoding="utf-8") as f:
                json.dump(self._snapshot_to_dict(snapshot), f,
                          ensure_ascii=False, indent=2)

            # 时间戳快照
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            hist_path = os.path.join(data_dir, f"contribution_{ts}.json")
            with open(hist_path, "w", encoding="utf-8") as f:
                json.dump(self._snapshot_to_dict(snapshot), f,
                          ensure_ascii=False, indent=2)

            # 保留最近 90 天的快照
            self._cleanup_old_snapshots(data_dir, keep=90)

            logger.debug(f"Contribution snapshot persisted: {latest_path}")
            return True

        except Exception as e:
            logger.error(f"Persist contribution snapshot error: {e}")
            return False

    def _snapshot_to_dict(self, snapshot: ContributionSnapshot) -> Dict[str, Any]:
        """将 ContributionSnapshot 转为 JSON 可序列化字典"""
        strategies = {}
        for sname, c in snapshot.strategies.items():
            strategies[sname] = {
                "total_pnl": safe_finite(c.total_pnl),
                "pnl_contribution_pct": safe_finite(c.pnl_contribution_pct),
                "realized_pnl": safe_finite(c.realized_pnl),
                "unrealized_pnl": safe_finite(c.unrealized_pnl),
                "total_trades": int(c.total_trades),
                "winning_trades": int(c.winning_trades),
                "losing_trades": int(c.losing_trades),
                "win_rate": safe_finite(c.win_rate),
                "avg_win": safe_finite(c.avg_win),
                "avg_loss": safe_finite(c.avg_loss),
                "profit_factor": safe_finite(c.profit_factor),
                "max_drawdown": safe_finite(c.max_drawdown),
                "volatility": safe_finite(c.volatility),
                "avg_capital_used": safe_finite(c.avg_capital_used),
                "pnl_per_capital_pct": safe_finite(c.pnl_per_capital_pct),
                "pnl_per_trade": safe_finite(c.pnl_per_trade),
                "total_fees": safe_finite(c.total_fees),
                "fee_ratio": safe_finite(c.fee_ratio),
                "active_hours": safe_finite(c.active_hours),
                "pnl_per_hour": safe_finite(c.pnl_per_hour),
                "trend": c.trend,
                "trend_pnl_7d_vs_30d": safe_finite(c.trend_pnl_7d_vs_30d),
                "risk_adjusted_contribution": safe_finite(c.risk_adjusted_contribution),
                # 成本细分归因
                "total_funding_cost": safe_finite(c.total_funding_cost),
                "total_slippage_cost": safe_finite(c.total_slippage_cost),
                "total_spread_cost": safe_finite(c.total_spread_cost),
                # 方向归因
                "long_pnl": safe_finite(c.long_pnl),
                "short_pnl": safe_finite(c.short_pnl),
                "long_trades": int(c.long_trades),
                "short_trades": int(c.short_trades),
                # 退出原因归因
                "stop_loss_count": int(c.stop_loss_count),
                "stop_loss_pnl": safe_finite(c.stop_loss_pnl),
                "take_profit_count": int(c.take_profit_count),
                "take_profit_pnl": safe_finite(c.take_profit_pnl),
                # 强化字段
                "lifecycle": c.lifecycle,
                "lifecycle_age_days": safe_finite(c.lifecycle_age_days),
                "lifecycle_last_trade_age_hours": safe_finite(c.lifecycle_last_trade_age_hours),
                "health_score": safe_finite(c.health_score),
                "health_grade": c.health_grade,
                "delta_pnl": safe_finite(c.delta_pnl),
                "delta_contribution_pct": safe_finite(c.delta_contribution_pct),
                "delta_health": safe_finite(c.delta_health),
            }

        return {
            "timestamp": snapshot.timestamp,
            "window": snapshot.window,
            "total_pnl": safe_finite(snapshot.total_pnl),
            "total_trades": int(snapshot.total_trades),
            "total_fees": safe_finite(snapshot.total_fees),
            "total_unrealized_pnl": safe_finite(snapshot.total_unrealized_pnl),
            "strategies": strategies,
            "correlation_matrix": snapshot.correlation_matrix,
            "concentration": safe_finite(snapshot.concentration),
            "diversification_score": safe_finite(snapshot.diversification_score),
            "efficiency_score": safe_finite(snapshot.efficiency_score),
            "overall_health_score": safe_finite(snapshot.overall_health_score),
            "synergy_score": safe_finite(snapshot.synergy_score),
            "lifecycle_summary": snapshot.lifecycle_summary,
            "available": snapshot.available,
            "data_error": snapshot.data_error,
        }

    @staticmethod
    def _cleanup_old_snapshots(data_dir: str, keep: int = 90):
        """清理过期快照文件"""
        try:
            files = [f for f in os.listdir(data_dir)
                     if f.startswith("contribution_") and f.endswith(".json")
                     and f != "contribution_latest.json"]
            files.sort(reverse=True)
            for old_file in files[keep:]:
                os.remove(os.path.join(data_dir, old_file))
        except Exception as e:
            logger.debug(f"Cleanup contribution snapshots error: {e}")

    def load_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        """加载最新持久化的快照"""
        try:
            latest_path = os.path.join("data", "contribution",
                                       "contribution_latest.json")
            if not os.path.exists(latest_path):
                return None
            with open(latest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.debug(f"Load contribution snapshot error: {e}")
            return None

    # ============================================================
    # 状态查询
    # ============================================================

    def get_status(self) -> Dict[str, Any]:
        """获取分析器状态"""
        return {
            "snapshots_cached": len(self._snapshot_history),
            "dynamic_allocations": self._dynamic_allocations,
            "available_windows": list(self._windows.keys()),
            "last_snapshot": (
                self._snapshot_history[-1].timestamp
                if self._snapshot_history else None
            ),
            "lifecycle_config": self._lifecycle_config,
            "health_weights": self._health_weights,
            "has_okx_client": self._okx_client is not None,
            "trade_query_error": self._trade_query_error,
        }


# ============================================================
# 单例
# ============================================================

_analyzer_instance: Optional[ContributionAnalyzer] = None


def get_contribution_analyzer(sqlite_storage=None, trade_journal=None,
                               config: Dict[str, Any] = None,
                               okx_client=None, account_manager=None) -> ContributionAnalyzer:
    """获取贡献度分析器单例"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = ContributionAnalyzer(
            sqlite_storage=sqlite_storage,
            trade_journal=trade_journal,
            config=config,
            okx_client=okx_client,
            account_manager=account_manager,
        )
    return _analyzer_instance


def reset_contribution_analyzer():
    """重置单例（用于重启）"""
    global _analyzer_instance
    _analyzer_instance = None


__all__ = [
    "ContributionAnalyzer",
    "StrategyContribution",
    "ContributionSnapshot",
    "get_contribution_analyzer",
    "reset_contribution_analyzer",
]
