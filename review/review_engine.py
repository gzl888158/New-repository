"""
复盘迭代指导框架系统
========================
1. 每日固定复盘维度：
   - 币池表现：20个币种当日盈亏、策略有效/失效标记
   - 风控拦截统计：拦截次数、拦截原因，针对性优化参数
   - 执行损耗统计：滑点、API报错、断流时长，优化数据链路
   - 资金曲线：当日最大回撤、盈亏比例、杠杆使用均值

2. 月度迭代标准：
   - 币池更新：淘汰低收益高回撤币种，替换高波动优质标的
   - 策略参数优化：根据全月回测数据调整加仓、止盈止损参数
   - 系统架构优化：修复当月高频出现的程序、网络、风控漏洞
"""

import os
import json
import sqlite3
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from loguru import logger


# ============================================================
# 数据模型
# ============================================================

@dataclass
class SymbolPerformance:
    """币种表现"""
    symbol: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_fees: float = 0.0
    max_drawdown: float = 0.0
    avg_leverage: float = 0.0
    net_pnl: float = 0.0  # total_pnl - fees
    win_rate: float = 0.0
    strategy_effectiveness: Dict[str, Any] = field(default_factory=dict)
    effective: bool = True  # 策略是否有效


@dataclass
class RiskInterceptionStats:
    """风控拦截统计"""
    layer: str
    total_checks: int = 0
    blocks: int = 0
    reduces: int = 0
    close_alls: int = 0
    block_reasons: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    top_reasons: List[Tuple[str, int]] = field(default_factory=list)


@dataclass
class ExecutionLossStats:
    """执行损耗统计"""
    avg_slippage: float = 0.0
    max_slippage: float = 0.0
    p95_slippage: float = 0.0
    tolerance_exceeded_rate: float = 0.0
    api_errors: int = 0
    api_error_rate: float = 0.0
    disconnection_seconds: float = 0.0
    data_gap_count: int = 0
    symbol_slippage_rank: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CapitalCurveSnapshot:
    """资金曲线快照"""
    date: str
    start_equity: float = 0.0
    end_equity: float = 0.0
    max_equity: float = 0.0
    min_equity: float = 0.0
    max_drawdown_pct: float = 0.0
    daily_return_pct: float = 0.0
    avg_leverage: float = 0.0
    max_leverage: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_approx: float = 0.0


@dataclass
class DailyReviewReport:
    """每日复盘报告"""
    date: str
    generated_at: str
    symbol_performance: List[SymbolPerformance]
    risk_interceptions: List[RiskInterceptionStats]
    execution_loss: ExecutionLossStats
    capital_curve: CapitalCurveSnapshot
    summary: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "symbol_performance": [
                {
                    "symbol": s.symbol,
                    "trades": s.trades,
                    "win_rate": round(s.win_rate, 4),
                    "total_pnl": round(s.total_pnl, 4),
                    "total_fees": round(s.total_fees, 4),
                    "net_pnl": round(s.net_pnl, 4),
                    "max_drawdown": round(s.max_drawdown, 4),
                    "avg_leverage": round(s.avg_leverage, 2),
                    "effective": s.effective,
                    "strategy_effectiveness": s.strategy_effectiveness,
                }
                for s in self.symbol_performance
            ],
            "risk_interceptions": [
                {
                    "layer": r.layer,
                    "total_checks": r.total_checks,
                    "blocks": r.blocks,
                    "reduces": r.reduces,
                    "close_alls": r.close_alls,
                    "top_reasons": r.top_reasons,
                }
                for r in self.risk_interceptions
            ],
            "execution_loss": {
                "avg_slippage": round(self.execution_loss.avg_slippage, 6),
                "max_slippage": round(self.execution_loss.max_slippage, 6),
                "p95_slippage": round(self.execution_loss.p95_slippage, 6),
                "tolerance_exceeded_rate": round(self.execution_loss.tolerance_exceeded_rate, 4),
                "api_errors": self.execution_loss.api_errors,
                "api_error_rate": round(self.execution_loss.api_error_rate, 4),
                "disconnection_seconds": round(self.execution_loss.disconnection_seconds, 1),
                "data_gap_count": self.execution_loss.data_gap_count,
                "symbol_slippage_rank": self.execution_loss.symbol_slippage_rank,
            },
            "capital_curve": {
                "start_equity": round(self.capital_curve.start_equity, 2),
                "end_equity": round(self.capital_curve.end_equity, 2),
                "max_drawdown_pct": round(self.capital_curve.max_drawdown_pct, 4),
                "daily_return_pct": round(self.capital_curve.daily_return_pct, 4),
                "avg_leverage": round(self.capital_curve.avg_leverage, 2),
                "max_leverage": round(self.capital_curve.max_leverage, 2),
                "win_rate": round(self.capital_curve.win_rate, 4),
                "profit_factor": round(self.capital_curve.profit_factor, 4),
                "sharpe_approx": round(self.capital_curve.sharpe_approx, 4),
            },
            "recommendations": self.recommendations,
        }


@dataclass
class MonthlyIterationPlan:
    """月度迭代计划"""
    month: str
    generated_at: str
    # 币池更新
    symbols_to_remove: List[str]
    symbols_to_add_candidates: List[str]
    symbol_ranking: List[Dict[str, Any]]
    # 策略参数优化
    strategy_param_adjustments: Dict[str, Dict[str, Any]]
    # 系统架构优化
    system_fixes: List[Dict[str, Any]]
    # 综合
    summary: Dict[str, Any]
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "month": self.month,
            "generated_at": self.generated_at,
            "summary": self.summary,
            "symbols_to_remove": self.symbols_to_remove,
            "symbols_to_add_candidates": self.symbols_to_add_candidates,
            "symbol_ranking": self.symbol_ranking,
            "strategy_param_adjustments": self.strategy_param_adjustments,
            "system_fixes": self.system_fixes,
            "confidence": round(self.confidence, 4),
        }


# ============================================================
# 每日复盘收集器
# ============================================================

class DailyReviewCollector:
    """每日复盘数据收集"""

    def __init__(self, db_path: str):
        self._db_path = db_path

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def collect_symbol_performance(self, date: datetime) -> List[SymbolPerformance]:
        """收集币种表现：盈亏、策略有效性"""
        date_str = date.strftime("%Y-%m-%d")
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        conn = self._conn()
        try:
            rows = conn.execute("""
                SELECT symbol, strategy_name, pnl, fees, leverage
                FROM trade_records
                WHERE status = 'closed' AND close_time >= ? AND close_time < ?
            """, (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"))).fetchall()
        except Exception:
            conn.close()
            return []
        conn.close()

        sym_map: Dict[str, SymbolPerformance] = {}
        for row in rows:
            sym = row["symbol"] or "UNKNOWN"
            pnl = row["pnl"] or 0
            fees = row["fees"] or 0
            win = pnl > 0
            if sym not in sym_map:
                sym_map[sym] = SymbolPerformance(symbol=sym)
            sp = sym_map[sym]
            sp.trades += 1
            sp.total_pnl += pnl
            sp.total_fees += fees
            sp.avg_leverage += row["leverage"] or 0
            if win:
                sp.wins += 1
            else:
                sp.losses += 1

            # 策略有效性追踪
            strat = row["strategy_name"] or "unknown"
            if strat not in sp.strategy_effectiveness:
                sp.strategy_effectiveness[strat] = {"trades": 0, "pnl": 0.0, "wins": 0}
            sp.strategy_effectiveness[strat]["trades"] += 1
            sp.strategy_effectiveness[strat]["pnl"] += pnl
            if win:
                sp.strategy_effectiveness[strat]["wins"] += 1

        # 计算派生指标
        for sp in sym_map.values():
            # trade_records.pnl 已为扣除手续费后的净盈亏，net_pnl 与 total_pnl 一致
            sp.net_pnl = sp.total_pnl
            sp.win_rate = sp.wins / sp.trades if sp.trades > 0 else 0
            sp.avg_leverage = sp.avg_leverage / sp.trades if sp.trades > 0 else 0
            # 策略有效/失效判定：净亏损且胜率<40% 或 回撤过大
            sp.effective = not (sp.net_pnl < -5 and sp.win_rate < 0.4 and sp.trades >= 3)
            for strat, stats in sp.strategy_effectiveness.items():
                stats["win_rate"] = round(stats["wins"] / stats["trades"], 4) if stats["trades"] > 0 else 0

        return sorted(sym_map.values(), key=lambda x: x.net_pnl, reverse=True)

    def collect_risk_interceptions(self, date: datetime,
                                   risk_gate=None) -> List[RiskInterceptionStats]:
        """收集风控拦截统计"""
        stats_list = []

        # 尝试从risk_gate获取统计
        if risk_gate and hasattr(risk_gate, '_interception_stats'):
            raw = risk_gate._interception_stats
            for layer_name, data in raw.items():
                stats = RiskInterceptionStats(
                    layer=layer_name,
                    total_checks=data.get("total", 0),
                    blocks=data.get("blocks", 0),
                    reduces=data.get("reduces", 0),
                    close_alls=data.get("close_alls", 0),
                )
                reasons = data.get("reasons", {})
                stats.block_reasons = dict(reasons)
                stats.top_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                stats_list.append(stats)

        # 兜底：从SQLite的risk_events表读取（适配实际表结构）
        if not stats_list:
            date_str = date.strftime("%Y-%m-%d")
            conn = self._conn()
            try:
                rows = conn.execute("""
                    SELECT event_type as layer, severity, message, COUNT(*) as cnt
                    FROM risk_events
                    WHERE date(timestamp) = ?
                    GROUP BY event_type, severity, message
                """, (date_str,)).fetchall()
                layer_map: Dict[str, RiskInterceptionStats] = {}
                for row in rows:
                    layer = row["layer"] or "unknown"
                    if layer not in layer_map:
                        layer_map[layer] = RiskInterceptionStats(layer=layer)
                    st = layer_map[layer]
                    st.total_checks += row["cnt"]
                    severity = row["severity"] or ""
                    if "reject" in severity or "block" in severity or "critical" in severity:
                        st.blocks += row["cnt"]
                    elif "reduce" in severity or "warning" in severity:
                        st.reduces += row["cnt"]
                    elif "close_all" in severity or "emergency" in severity:
                        st.close_alls += row["cnt"]
                    reason = row["message"] or "unknown"
                    st.block_reasons[reason] += row["cnt"]
                for st in layer_map.values():
                    st.top_reasons = sorted(st.block_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                stats_list = list(layer_map.values())
            except Exception:
                pass
            conn.close()

        # 仍然没有数据：创建空占位
        if not stats_list:
            for layer in ["L1_pre_trade", "L2_in_trade", "L3_position", "L4_daily", "L5_emergency"]:
                stats_list.append(RiskInterceptionStats(layer=layer))

        return stats_list

    def collect_execution_loss(self, date: datetime, risk_gate=None) -> ExecutionLossStats:
        """收集执行损耗统计：滑点、API报错、断流"""
        date_str = date.strftime("%Y-%m-%d")
        cutoff = date.replace(hour=0, minute=0, second=0).isoformat()
        stats = ExecutionLossStats()

        conn = self._conn()
        try:
            # 滑点统计（fill_quality表）
            row = conn.execute("""
                SELECT AVG(abs_slippage) as avg_slippage,
                       MAX(abs_slippage) as max_slippage,
                       SUM(exceeds_tolerance) as tol_exceeded,
                       COUNT(*) as total
                FROM fill_quality
                WHERE timestamp >= ?
            """, (cutoff,)).fetchone()
            if row and row["total"]:
                stats.avg_slippage = row["avg_slippage"] or 0
                stats.max_slippage = row["max_slippage"] or 0
                stats.tolerance_exceeded_rate = (row["tol_exceeded"] or 0) / row["total"]

            # 按币种滑点排名
            rows = conn.execute("""
                SELECT symbol, AVG(abs_slippage) as avg_slippage, COUNT(*) as cnt
                FROM fill_quality
                WHERE timestamp >= ?
                GROUP BY symbol
                ORDER BY avg_slippage DESC
                LIMIT 10
            """, (cutoff,)).fetchall()
            stats.symbol_slippage_rank = [
                {"symbol": r["symbol"], "avg_slippage": round(r["avg_slippage"] or 0, 6), "count": r["cnt"]}
                for r in rows
            ]

            # P95滑点
            srows = conn.execute("""
                SELECT abs_slippage FROM fill_quality WHERE timestamp >= ?
            """, (cutoff,)).fetchall()
            slippages = sorted([r["abs_slippage"] for r in srows if r["abs_slippage"] is not None])
            if slippages:
                stats.p95_slippage = slippages[int(len(slippages) * 0.95)] if len(slippages) >= 20 else slippages[-1]

        except Exception as e:
            logger.debug(f"Execution loss collection error: {e}")
        finally:
            conn.close()

        # 从risk_gate获取API错误和断流统计（内存数据，非数据库表）
        if risk_gate:
            try:
                # API调用统计
                l2 = getattr(risk_gate, '_l2', None)
                if l2 and hasattr(l2, '_api_call_times'):
                    now = time.time()
                    # 统计当日API调用总数和窗口内频率
                    today_calls = [t for t in l2._api_call_times if now - t < 86400]
                    stats.api_errors = getattr(l2, '_api_error_count', 0)
                    total_calls = len(today_calls)
                    stats.api_error_rate = stats.api_errors / total_calls if total_calls > 0 else 0

                # 断流检测（L5的_last_data_time与当前时间差）
                l5 = getattr(risk_gate, '_l5', None)
                if l5 and hasattr(l5, '_last_data_time') and l5._last_data_time:
                    import time as _time
                    gap = _time.time() - l5._last_data_time
                    if gap > 30:  # 超过30秒算断流
                        stats.disconnection_seconds = round(gap, 0)
                        stats.data_gap_count = 1
            except Exception as e:
                logger.debug(f"Risk gate stats collection error: {e}")

        return stats

    def collect_capital_curve(self, date: datetime) -> CapitalCurveSnapshot:
        """收集资金曲线：当日最大回撤、盈亏比例、杠杆均值"""
        date_str = date.strftime("%Y-%m-%d")
        start = date.replace(hour=0, minute=0, second=0)
        end = start + timedelta(days=1)

        snap = CapitalCurveSnapshot(date=date_str)
        conn = self._conn()
        try:
            rows = conn.execute("""
                SELECT timestamp, total_equity, used_margin, unrealized_pnl, realized_pnl,
                       win_rate, total_trades
                FROM equity_curve
                WHERE timestamp >= ? AND timestamp < ?
                ORDER BY timestamp
            """, (start.isoformat(), end.isoformat())).fetchall()

            if rows:
                equities = [r["total_equity"] or 0 for r in rows]
                snap.start_equity = equities[0]
                snap.end_equity = equities[-1]
                snap.max_equity = max(equities)
                snap.min_equity = min(equities)

                if snap.start_equity > 0:
                    snap.daily_return_pct = (snap.end_equity - snap.start_equity) / snap.start_equity
                    snap.max_drawdown_pct = (snap.max_equity - snap.min_equity) / snap.max_equity if snap.max_equity > 0 else 0

                leverages = []
                for r in rows:
                    eq = r["total_equity"] or 0
                    margin = r["used_margin"] or 0
                    # 杠杆=used_margin/total_equity，仅当权益>0时才有意义
                    # equity=0 时跳过，避免 margin/1 误报为杠杆值
                    if eq > 0:
                        leverages.append(margin / eq)
                if leverages:
                    snap.avg_leverage = np.mean(leverages)
                    snap.max_leverage = max(leverages)

                snap.win_rate = rows[-1]["win_rate"] or 0

            # 盈亏比 & 近似Sharpe（从trade_records表，pnl已为净盈亏）
            trows = conn.execute("""
                SELECT pnl FROM trade_records
                WHERE status = 'closed' AND close_time >= ? AND close_time < ? AND pnl IS NOT NULL
            """, (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"))).fetchall()
            pnls = [r["pnl"] for r in trows]
            if pnls:
                profits = [p for p in pnls if p > 0]
                losses = [abs(p) for p in pnls if p < 0]
                snap.profit_factor = sum(profits) / sum(losses) if losses else float('inf')
                if len(pnls) >= 2:
                    mean_pnl = np.mean(pnls)
                    std_pnl = np.std(pnls, ddof=1)
                    snap.sharpe_approx = mean_pnl / std_pnl if std_pnl > 0 else 0

        except Exception as e:
            logger.debug(f"Capital curve collection error: {e}")
        conn.close()

        return snap


# ============================================================
# 月度迭代决策器
# ============================================================

class MonthlyIterationPlanner:
    """月度迭代计划生成"""

    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._review_dir = config.get("review", {}).get("data_dir", "./data/reviews")
        os.makedirs(self._review_dir, exist_ok=True)

    def _load_daily_reviews(self, month: datetime) -> List[Dict[str, Any]]:
        """加载指定月份的所有每日复盘报告"""
        month_str = month.strftime("%Y-%m")
        reviews = []
        for fname in os.listdir(self._review_dir):
            if fname.startswith(f"daily_review_{month_str}") and fname.endswith(".json"):
                try:
                    with open(os.path.join(self._review_dir, fname), "r", encoding="utf-8") as f:
                        reviews.append(json.load(f))
                except Exception:
                    pass
        return reviews

    def plan_symbol_pool_update(self, reviews: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """币池更新决策：淘汰+候选替换"""
        # 聚合币种表现
        sym_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "days": 0, "total_pnl": 0.0, "total_trades": 0, "wins": 0, "max_dd": 0.0,
            "avg_leverage": 0.0, "effective_days": 0,
        })

        for review in reviews:
            for sp in review.get("symbol_performance", []):
                sym = sp["symbol"]
                s = sym_stats[sym]
                s["days"] += 1
                s["total_pnl"] += sp.get("net_pnl", 0)
                s["total_trades"] += sp.get("trades", 0)
                s["wins"] += sp.get("wins", 0)
                s["max_dd"] = max(s["max_dd"], abs(sp.get("max_drawdown", 0)))
                s["avg_leverage"] += sp.get("avg_leverage", 0)
                if sp.get("effective", True):
                    s["effective_days"] += 1

        # 评分排序
        ranking = []
        for sym, s in sym_stats.items():
            win_rate = s["wins"] / s["total_trades"] if s["total_trades"] > 0 else 0
            avg_pnl_per_day = s["total_pnl"] / s["days"] if s["days"] > 0 else 0
            effectiveness = s["effective_days"] / s["days"] if s["days"] > 0 else 1
            avg_leverage = s["avg_leverage"] / s["days"] if s["days"] > 0 else 0

            # 综合评分：高收益 + 高胜率 + 低回撤 + 高有效性
            score = (
                avg_pnl_per_day * 10 +
                win_rate * 50 -
                s["max_dd"] * 100 +
                effectiveness * 20 -
                avg_leverage * 2  # 高杠杆惩罚
            )
            ranking.append({
                "symbol": sym,
                "score": round(score, 2),
                "avg_pnl_per_day": round(avg_pnl_per_day, 4),
                "win_rate": round(win_rate, 4),
                "max_drawdown": round(s["max_dd"], 4),
                "effectiveness": round(effectiveness, 4),
                "total_trades": s["total_trades"],
            })

        ranking.sort(key=lambda x: x["score"], reverse=True)

        # 淘汰：低收益+高回撤 或 多日积分为负+无效天数多
        to_remove = []
        for r in ranking:
            if r["avg_pnl_per_day"] < -1 and r["max_drawdown"] > 0.05:
                to_remove.append(r["symbol"])
            elif r["effectiveness"] < 0.3 and r["total_trades"] >= 10:
                to_remove.append(r["symbol"])

        # 候选替换：从高波动优质标的中选（简化版，实际可从外部数据源推荐）
        candidates = []
        cfg_symbols = self._config.get("trading", {}).get("symbols", [])
        active_symbols = set(r["symbol"] for r in ranking)
        # 从tier1/tier2配置中找未交易的币种
        all_candidate_symbols = set()
        for tier in ["tier1_symbols", "tier2_symbols", "tier3_symbols"]:
            syms = self._config.get("currencies", {}).get(tier, [])
            for s in syms:
                all_candidate_symbols.add(f"{s}-USDT-SWAP")
        for sym in all_candidate_symbols:
            if sym not in active_symbols:
                candidates.append(sym)

        return to_remove, candidates, ranking

    def plan_strategy_param_adjustments(self, reviews: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """策略参数优化建议"""
        adjustments = {}

        # 按策略聚合表现
        strat_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_pnl": 0.0, "trades": 0, "wins": 0, "avg_slippage": 0.0,
            "max_dd": 0.0, "interceptions": 0,
        })

        for review in reviews:
            for sp in review.get("symbol_performance", []):
                for strat, se in sp.get("strategy_effectiveness", {}).items():
                    s = strat_stats[strat]
                    s["total_pnl"] += se.get("pnl", 0)
                    s["trades"] += se.get("trades", 0)
                    s["wins"] += se.get("wins", 0)

            el = review.get("execution_loss", {})
            for rank in el.get("symbol_slippage_rank", []):
                for strat in strat_stats:
                    strat_stats[strat]["avg_slippage"] += rank.get("avg_slippage", 0)

            for ri in review.get("risk_interceptions", []):
                for strat in strat_stats:
                    strat_stats[strat]["interceptions"] += ri.get("blocks", 0)

        for strat, stats in strat_stats.items():
            if stats["trades"] < 10:
                continue

            win_rate = stats["wins"] / stats["trades"]
            avg_pnl = stats["total_pnl"] / stats["trades"]
            adjustments[strat] = {}

            # 胜率低 → 收紧信号质量阈值
            if win_rate < 0.35:
                adjustments[strat]["min_signal_quality"] = {"action": "increase", "reason": f"胜率过低({win_rate:.0%})，收紧信号门槛"}
            elif win_rate > 0.65:
                adjustments[strat]["min_signal_quality"] = {"action": "decrease", "reason": f"胜率过高({win_rate:.0%})，可适当放宽以增交易频率"}

            # 单笔亏损大 → 收紧止损
            if avg_pnl < -2:
                adjustments[strat]["stop_loss_pct"] = {"action": "decrease", "reason": f"单笔均亏{avg_pnl:.2f}U，收紧止损"}

            # 滑点高 → 增大滑点容忍或优化下单策略
            avg_slip = stats["avg_slippage"] / max(len(reviews), 1)
            if avg_slip > 0.001:
                adjustments[strat]["slippage_tolerance"] = {"action": "increase", "reason": f"平均滑点{avg_slip:.4%}，增大容忍度"}

            # 拦截多 → 降低单笔仓位或杠杆
            if stats["interceptions"] > 20:
                adjustments[strat]["position_size"] = {"action": "decrease", "reason": f"月拦截{stats['interceptions']}次，降低单笔规模"}

        return adjustments

    def plan_system_fixes(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """系统架构优化建议：高频漏洞修复"""
        fixes = []

        # 聚合全月风控拦截原因
        all_reasons: Dict[str, int] = defaultdict(int)
        total_api_errors = 0
        total_disconnect = 0.0
        total_gaps = 0

        for review in reviews:
            for ri in review.get("risk_interceptions", []):
                for reason, cnt in ri.get("top_reasons", []):
                    all_reasons[reason] += cnt
            el = review.get("execution_loss", {})
            total_api_errors += el.get("api_errors", 0)
            total_disconnect += el.get("disconnection_seconds", 0)
            total_gaps += el.get("data_gap_count", 0)

        # 高频拦截原因 → 针对性修复
        sorted_reasons = sorted(all_reasons.items(), key=lambda x: x[1], reverse=True)
        for reason, cnt in sorted_reasons[:3]:
            if cnt >= 5:
                fixes.append({
                    "category": "risk_gate",
                    "issue": reason,
                    "frequency": cnt,
                    "suggested_fix": self._suggest_fix_for_reason(reason),
                    "priority": "high" if cnt >= 10 else "medium",
                })

        # API错误多 → 网络/请求链路优化
        if total_api_errors >= 10:
            fixes.append({
                "category": "api_stability",
                "issue": f"月API错误{total_api_errors}次",
                "frequency": total_api_errors,
                "suggested_fix": "检查代理稳定性、增加请求重试、启用备用REST端点",
                "priority": "high",
            })

        # 断流时间长 → 数据链路优化
        if total_disconnect >= 300:
            fixes.append({
                "category": "data_pipeline",
                "issue": f"月数据断流{total_disconnect:.0f}秒/{total_gaps}次",
                "frequency": total_gaps,
                "suggested_fix": "检查WebSocket连接、增加REST兜底拉取频率、部署本地行情缓存",
                "priority": "high" if total_disconnect >= 600 else "medium",
            })

        return fixes

    def _suggest_fix_for_reason(self, reason: str) -> str:
        """根据拦截原因给出修复建议"""
        reason_lower = reason.lower()
        if "余额" in reason_lower or "balance" in reason_lower or "margin" in reason_lower:
            return "优化资金分配算法，确保各策略间资金不重叠占用"
        elif "杠杆" in reason_lower or "leverage" in reason_lower:
            return "降低全局杠杆上限或启用动态杠杆分级"
        elif "滑点" in reason_lower or "slippage" in reason_lower:
            return "启用滑点优化执行器、扩大限价单偏移、避开高波动时段"
        elif "延迟" in reason_lower or "latency" in reason_lower:
            return "切换更近的代理节点、启用请求并发池、优化本地网络"
        elif "频率" in reason_lower or "rate" in reason_lower:
            return "降低策略信号频率、启用批量下单、优化API调用合并"
        elif "亏损" in reason_lower or "loss" in reason_lower:
            return "收紧单日亏损限制、启用更严格的止损策略"
        elif "仓位" in reason_lower or "position" in reason_lower:
            return "降低单币种仓位上限、启用集中度监控"
        else:
            return "针对性调整对应风控阈值参数"

    def generate_plan(self, month: datetime) -> MonthlyIterationPlan:
        """生成月度迭代计划"""
        reviews = self._load_daily_reviews(month)
        if not reviews:
            return MonthlyIterationPlan(
                month=month.strftime("%Y-%m"),
                generated_at=datetime.now().isoformat(),
                symbols_to_remove=[],
                symbols_to_add_candidates=[],
                symbol_ranking=[],
                strategy_param_adjustments={},
                system_fixes=[],
                summary={"error": "No daily reviews found for this month"},
                confidence=0.0,
            )

        to_remove, candidates, ranking = self.plan_symbol_pool_update(reviews)
        param_adj = self.plan_strategy_param_adjustments(reviews)
        fixes = self.plan_system_fixes(reviews)

        # 汇总统计
        total_pnl = sum(r.get("capital_curve", {}).get("daily_return_pct", 0) for r in reviews)
        avg_win_rate = np.mean([r.get("capital_curve", {}).get("win_rate", 0) for r in reviews])
        max_dd = max(r.get("capital_curve", {}).get("max_drawdown_pct", 0) for r in reviews)

        summary = {
            "review_days": len(reviews),
            "monthly_return_pct": round(total_pnl, 4),
            "avg_daily_win_rate": round(avg_win_rate, 4),
            "max_drawdown_pct": round(max_dd, 4),
            "symbols_evaluated": len(ranking),
            "symbols_to_remove": len(to_remove),
            "param_adjustments": len(param_adj),
            "system_fixes": len(fixes),
        }

        confidence = min(1.0, len(reviews) / 25)  # 数据越完整置信度越高

        return MonthlyIterationPlan(
            month=month.strftime("%Y-%m"),
            generated_at=datetime.now().isoformat(),
            symbols_to_remove=to_remove,
            symbols_to_add_candidates=candidates,
            symbol_ranking=ranking,
            strategy_param_adjustments=param_adj,
            system_fixes=fixes,
            summary=summary,
            confidence=confidence,
        )


# ============================================================
# 复盘迭代主引擎
# ============================================================

class ReviewEngine:
    """
    复盘迭代指导框架主引擎

    职责：
    1. 每日定时执行复盘，生成DailyReviewReport
    2. 每月初生成月度迭代计划MonthlyIterationPlan
    3. 持久化复盘报告，供Dashboard API查询
    4. 触发迭代建议通知
    """

    def __init__(self, config: Dict[str, Any], okx_client=None, risk_gate=None,
                 trade_journal=None, db_path: str = ""):
        self._config = config
        self._okx_client = okx_client
        self._risk_gate = risk_gate
        self._trade_journal = trade_journal

        self._db_path = db_path or config.get("sqlite", {}).get("db_path", "./data/trading.db")
        self._review_dir = config.get("review", {}).get("data_dir", "./data/reviews")
        os.makedirs(self._review_dir, exist_ok=True)

        self._daily_collector = DailyReviewCollector(self._db_path)
        self._monthly_planner = MonthlyIterationPlanner(config)

        # 历史记录
        self._daily_reports: deque = deque(maxlen=90)
        self._monthly_plans: deque = deque(maxlen=12)

        # 通知回调
        self._on_daily_review: Any = None
        self._on_monthly_plan: Any = None

    def set_callbacks(self, on_daily_review=None, on_monthly_plan=None):
        """设置通知回调"""
        self._on_daily_review = on_daily_review
        self._on_monthly_plan = on_monthly_plan

    async def run_daily_review(self, date: Optional[datetime] = None) -> DailyReviewReport:
        """执行每日复盘"""
        if date is None:
            date = datetime.now() - timedelta(days=1)  # 复盘昨日

        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"Running daily review for {date_str}")

        # 1. 币池表现
        sym_perf = self._daily_collector.collect_symbol_performance(date)

        # 2. 风控拦截统计
        risk_stats = self._daily_collector.collect_risk_interceptions(date, self._risk_gate)

        # 3. 执行损耗统计
        exec_loss = self._daily_collector.collect_execution_loss(date, self._risk_gate)

        # 4. 资金曲线
        capital = self._daily_collector.collect_capital_curve(date)

        # 汇总
        total_trades = sum(s.trades for s in sym_perf)
        total_net_pnl = sum(s.net_pnl for s in sym_perf)
        effective_symbols = sum(1 for s in sym_perf if s.effective)
        ineffective_symbols = [s.symbol for s in sym_perf if not s.effective]

        summary = {
            "total_trades": total_trades,
            "total_net_pnl": round(total_net_pnl, 4),
            "effective_symbols": effective_symbols,
            "ineffective_symbols": ineffective_symbols,
            "symbols_reviewed": len(sym_perf),
        }

        # 生成建议
        recommendations = self._generate_daily_recommendations(
            sym_perf, risk_stats, exec_loss, capital
        )

        report = DailyReviewReport(
            date=date_str,
            generated_at=datetime.now().isoformat(),
            symbol_performance=sym_perf,
            risk_interceptions=risk_stats,
            execution_loss=exec_loss,
            capital_curve=capital,
            summary=summary,
            recommendations=recommendations,
        )

        # 持久化
        self._persist_daily_report(report)
        self._daily_reports.append(report)

        logger.info(f"Daily review completed: {date_str}, trades={total_trades}, pnl={total_net_pnl:.2f}")

        # 通知
        if self._on_daily_review:
            try:
                await self._on_daily_review(report)
            except Exception as e:
                logger.debug(f"Daily review callback error: {e}")

        # 重置风控拦截统计，为新的一天做准备
        if self._risk_gate and hasattr(self._risk_gate, 'reset_interception_stats'):
            try:
                self._risk_gate.reset_interception_stats()
            except Exception as e:
                logger.debug(f"Failed to reset interception stats: {e}")

        return report

    def _generate_daily_recommendations(self, sym_perf: List[SymbolPerformance],
                                        risk_stats: List[RiskInterceptionStats],
                                        exec_loss: ExecutionLossStats,
                                        capital: CapitalCurveSnapshot) -> List[str]:
        """生成每日复盘建议"""
        recs = []

        # 币种层面
        bad_symbols = [s for s in sym_perf if not s.effective and s.trades >= 3]
        if bad_symbols:
            recs.append(f"币种失效预警: {', '.join(s.symbol for s in bad_symbols[:3])} — 建议冻结或降低权重")

        top_losers = [s for s in sym_perf if s.net_pnl < -5][:3]
        if top_losers:
            recs.append(f"当日亏损TOP: {', '.join(f'{s.symbol}({s.net_pnl:.1f}U)' for s in top_losers)} — 检查策略适配性")

        # 风控层面
        total_blocks = sum(r.blocks for r in risk_stats)
        if total_blocks >= 10:
            top_reason = risk_stats[0].top_reasons[0] if risk_stats[0].top_reasons else ("unknown", 0)
            recs.append(f"风控拦截频繁({total_blocks}次)，主因: {top_reason[0]}({top_reason[1]}次) — 建议优化对应参数")

        # 执行层面
        if exec_loss.tolerance_exceeded_rate > 0.2:
            recs.append(f"滑点超限率{exec_loss.tolerance_exceeded_rate:.1%}偏高 — 建议启用滑点优化执行器或避开高波动时段")
        if exec_loss.api_errors >= 5:
            recs.append(f"API错误{exec_loss.api_errors}次 — 检查网络代理稳定性")
        if exec_loss.disconnection_seconds >= 60:
            recs.append(f"数据断流{exec_loss.disconnection_seconds:.0f}秒 — 检查行情链路")

        # 资金层面
        if capital.max_drawdown_pct >= 0.05:
            recs.append(f"当日最大回撤{capital.max_drawdown_pct:.2%} — 建议收紧止损或降低杠杆")
        if capital.daily_return_pct < -0.03:
            recs.append(f"当日亏损{capital.daily_return_pct:.2%}超过3% — 触发单日风控复核")
        if capital.avg_leverage >= 10:
            recs.append(f"平均杠杆{capital.avg_leverage:.1f}x偏高 — 建议降杠杆保护本金")

        return recs

    async def run_monthly_iteration(self, month: Optional[datetime] = None) -> MonthlyIterationPlan:
        """执行月度迭代计划"""
        if month is None:
            # 上一个月
            today = datetime.now()
            month = datetime(today.year, today.month, 1) - timedelta(days=1)

        month_str = month.strftime("%Y-%m")
        logger.info(f"Running monthly iteration plan for {month_str}")

        plan = self._monthly_planner.generate_plan(month)
        self._monthly_plans.append(plan)

        # 持久化
        path = os.path.join(self._review_dir, f"monthly_plan_{month_str}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(plan.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist monthly plan: {e}")

        logger.info(f"Monthly iteration plan generated: {month_str}, "
                    f"remove={len(plan.symbols_to_remove)}, "
                    f"adjust={len(plan.strategy_param_adjustments)}, "
                    f"fixes={len(plan.system_fixes)}")

        # 通知
        if self._on_monthly_plan:
            try:
                await self._on_monthly_plan(plan)
            except Exception as e:
                logger.debug(f"Monthly plan callback error: {e}")

        return plan

    def _persist_daily_report(self, report: DailyReviewReport):
        """持久化每日复盘报告"""
        path = os.path.join(self._review_dir, f"daily_review_{report.date}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist daily review: {e}")

    # ---- 查询接口 ----

    def get_daily_report(self, date_str: str) -> Optional[Dict[str, Any]]:
        """获取指定日期复盘报告"""
        path = os.path.join(self._review_dir, f"daily_review_{date_str}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def get_recent_daily_reports(self, days: int = 7) -> List[Dict[str, Any]]:
        """获取最近N天复盘报告"""
        reports = []
        for i in range(days):
            d = (datetime.now() - timedelta(days=i + 1)).strftime("%Y-%m-%d")
            r = self.get_daily_report(d)
            if r:
                reports.append(r)
        return reports

    def get_monthly_plan(self, month_str: str) -> Optional[Dict[str, Any]]:
        """获取指定月度迭代计划"""
        path = os.path.join(self._review_dir, f"monthly_plan_{month_str}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    def get_latest_monthly_plan(self) -> Optional[Dict[str, Any]]:
        """获取最新月度迭代计划"""
        for i in range(12):
            d = datetime.now() - timedelta(days=i * 30)
            month_str = d.strftime("%Y-%m")
            plan = self.get_monthly_plan(month_str)
            if plan:
                return plan
        return None

    def get_symbol_trend(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """获取指定币种最近N天趋势"""
        daily_pnls = []
        daily_trades = []
        daily_win_rates = []

        for i in range(days):
            d = (datetime.now() - timedelta(days=i + 1)).strftime("%Y-%m-%d")
            report = self.get_daily_report(d)
            if not report:
                continue
            for sp in report.get("symbol_performance", []):
                if sp["symbol"] == symbol:
                    daily_pnls.append(sp.get("net_pnl", 0))
                    daily_trades.append(sp.get("trades", 0))
                    daily_win_rates.append(sp.get("win_rate", 0))
                    break

        if not daily_pnls:
            return {"symbol": symbol, "days": 0, "trend": "no_data"}

        return {
            "symbol": symbol,
            "days": len(daily_pnls),
            "total_pnl": round(sum(daily_pnls), 4),
            "avg_daily_pnl": round(np.mean(daily_pnls), 4),
            "avg_trades": round(np.mean(daily_trades), 2),
            "avg_win_rate": round(np.mean(daily_win_rates), 4),
            "trend": "improving" if daily_pnls[-1] > daily_pnls[0] else "declining" if len(daily_pnls) >= 2 else "stable",
        }
