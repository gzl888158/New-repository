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

企业级强化：
- profit_factor 无亏损时返回 None（非 float('inf')），保证 JSON 可序列化
- DB 数值字段经 safe_float 转换，防止畸形值导致类型错误
- 静默 except 补充 debug 日志（不改变降级行为）
- 所有除法经 safe_div 保护
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

from eval._base import safe_float, safe_int, safe_div


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
    net_pnl: float = 0.0
    win_rate: float = 0.0
    strategy_effectiveness: Dict[str, Any] = field(default_factory=dict)
    effective: bool = True


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
    profit_factor: Optional[float] = None
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
                # 企业级：profit_factor 无亏损时为 None（JSON null），非 Infinity
                "profit_factor": (round(self.capital_curve.profit_factor, 4)
                                  if self.capital_curve.profit_factor is not None else None),
                "sharpe_approx": round(self.capital_curve.sharpe_approx, 4),
            },
            "recommendations": self.recommendations,
        }


@dataclass
class MonthlyIterationPlan:
    """月度迭代计划"""
    month: str
    generated_at: str
    symbols_to_remove: List[str]
    symbols_to_add_candidates: List[str]
    symbol_ranking: List[Dict[str, Any]]
    strategy_param_adjustments: Dict[str, Dict[str, Any]]
    system_fixes: List[Dict[str, Any]]
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
        except Exception as e:
            logger.debug(f"review_engine: collect_symbol_performance query failed: {type(e).__name__}: {e}")
            conn.close()
            return []
        conn.close()

        sym_map: Dict[str, SymbolPerformance] = {}
        for row in rows:
            sym = row["symbol"] or "UNKNOWN"
            # 企业级：DB 数值安全转换，防止畸形值污染统计
            pnl = safe_float(row["pnl"], 0.0)
            fees = safe_float(row["fees"], 0.0)
            win = pnl > 0
            if sym not in sym_map:
                sym_map[sym] = SymbolPerformance(symbol=sym)
            sp = sym_map[sym]
            sp.trades += 1
            sp.total_pnl += pnl
            sp.total_fees += fees
            sp.avg_leverage += safe_float(row["leverage"], 0.0)
            if win:
                sp.wins += 1
            else:
                sp.losses += 1

            strat = row["strategy_name"] or "unknown"
            if strat not in sp.strategy_effectiveness:
                sp.strategy_effectiveness[strat] = {"trades": 0, "pnl": 0.0, "wins": 0}
            sp.strategy_effectiveness[strat]["trades"] += 1
            sp.strategy_effectiveness[strat]["pnl"] += pnl
            if win:
                sp.strategy_effectiveness[strat]["wins"] += 1

        for sp in sym_map.values():
            sp.net_pnl = sp.total_pnl
            sp.win_rate = safe_div(sp.wins, sp.trades, 0.0)
            sp.avg_leverage = safe_div(sp.avg_leverage, sp.trades, 0.0)
            sp.effective = not (sp.net_pnl < -5 and sp.win_rate < 0.4 and sp.trades >= 3)
            for strat, stats in sp.strategy_effectiveness.items():
                stats["win_rate"] = round(safe_div(stats["wins"], stats["trades"], 0.0), 4)

        return sorted(sym_map.values(), key=lambda x: x.net_pnl, reverse=True)

    def collect_risk_interceptions(self, date: datetime,
                                   risk_gate=None) -> List[RiskInterceptionStats]:
        """收集风控拦截统计"""
        stats_list = []

        if risk_gate and hasattr(risk_gate, '_interception_stats'):
            raw = risk_gate._interception_stats
            for layer_name, data in raw.items():
                stats = RiskInterceptionStats(
                    layer=layer_name,
                    total_checks=safe_int(data.get("total"), 0),
                    blocks=safe_int(data.get("blocks"), 0),
                    reduces=safe_int(data.get("reduces"), 0),
                    close_alls=safe_int(data.get("close_alls"), 0),
                )
                reasons = data.get("reasons", {})
                stats.block_reasons = dict(reasons)
                stats.top_reasons = sorted(reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                stats_list.append(stats)

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
                    cnt = safe_int(row["cnt"], 0)
                    st.total_checks += cnt
                    severity = row["severity"] or ""
                    if "reject" in severity or "block" in severity or "critical" in severity:
                        st.blocks += cnt
                    elif "reduce" in severity or "warning" in severity:
                        st.reduces += cnt
                    elif "close_all" in severity or "emergency" in severity:
                        st.close_alls += cnt
                    reason = row["message"] or "unknown"
                    st.block_reasons[reason] += cnt
                for st in layer_map.values():
                    st.top_reasons = sorted(st.block_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
                stats_list = list(layer_map.values())
            except Exception as e:
                logger.debug(f"review_engine: risk_events query failed: {type(e).__name__}: {e}")
            conn.close()

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
            row = conn.execute("""
                SELECT AVG(abs_slippage) as avg_slippage,
                       MAX(abs_slippage) as max_slippage,
                       SUM(exceeds_tolerance) as tol_exceeded,
                       COUNT(*) as total
                FROM fill_quality
                WHERE timestamp >= ?
            """, (cutoff,)).fetchone()
            if row and row["total"]:
                stats.avg_slippage = safe_float(row["avg_slippage"], 0.0)
                stats.max_slippage = safe_float(row["max_slippage"], 0.0)
                stats.tolerance_exceeded_rate = safe_div(safe_float(row["tol_exceeded"], 0.0), row["total"], 0.0)

            rows = conn.execute("""
                SELECT symbol, AVG(abs_slippage) as avg_slippage, COUNT(*) as cnt
                FROM fill_quality
                WHERE timestamp >= ?
                GROUP BY symbol
                ORDER BY avg_slippage DESC
                LIMIT 10
            """, (cutoff,)).fetchall()
            stats.symbol_slippage_rank = [
                {"symbol": r["symbol"], "avg_slippage": round(safe_float(r["avg_slippage"], 0.0), 6), "count": safe_int(r["cnt"], 0)}
                for r in rows
            ]

            srows = conn.execute("""
                SELECT abs_slippage FROM fill_quality WHERE timestamp >= ?
            """, (cutoff,)).fetchall()
            slippages = sorted([safe_float(r["abs_slippage"], 0.0) for r in srows if r["abs_slippage"] is not None])
            if slippages:
                idx = min(int(len(slippages) * 0.95), len(slippages) - 1)
                stats.p95_slippage = slippages[idx]

        except Exception as e:
            logger.debug(f"review_engine: execution loss collection error: {type(e).__name__}: {e}")
        finally:
            conn.close()

        if risk_gate:
            try:
                l2 = getattr(risk_gate, '_l2', None)
                if l2 and hasattr(l2, '_api_call_times'):
                    now = time.time()
                    today_calls = [t for t in l2._api_call_times if now - t < 86400]
                    stats.api_errors = safe_int(getattr(l2, '_api_error_count', 0), 0)
                    total_calls = len(today_calls)
                    stats.api_error_rate = safe_div(stats.api_errors, total_calls, 0.0)

                l5 = getattr(risk_gate, '_l5', None)
                if l5 and hasattr(l5, '_last_data_time') and l5._last_data_time:
                    gap = time.time() - l5._last_data_time
                    if gap > 30:
                        stats.disconnection_seconds = round(gap, 0)
                        stats.data_gap_count = 1
            except Exception as e:
                logger.debug(f"review_engine: risk gate stats collection error: {type(e).__name__}: {e}")

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
                equities = [safe_float(r["total_equity"], 0.0) for r in rows]
                snap.start_equity = equities[0]
                snap.end_equity = equities[-1]
                snap.max_equity = max(equities)
                snap.min_equity = min(equities)

                if snap.start_equity > 0:
                    snap.daily_return_pct = (snap.end_equity - snap.start_equity) / snap.start_equity
                    snap.max_drawdown_pct = safe_div(snap.max_equity - snap.min_equity, snap.max_equity, 0.0)

                leverages = []
                for r in rows:
                    eq = safe_float(r["total_equity"], 0.0)
                    margin = safe_float(r["used_margin"], 0.0)
                    if eq > 0:
                        leverages.append(margin / eq)
                if leverages:
                    snap.avg_leverage = float(np.mean(leverages))
                    snap.max_leverage = max(leverages)

                snap.win_rate = safe_float(rows[-1]["win_rate"], 0.0)

            trows = conn.execute("""
                SELECT pnl FROM trade_records
                WHERE status = 'closed' AND close_time >= ? AND close_time < ? AND pnl IS NOT NULL
            """, (start.strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"))).fetchall()
            pnls = [safe_float(r["pnl"], 0.0) for r in trows]
            if pnls:
                profits = [p for p in pnls if p > 0]
                losses = [abs(p) for p in pnls if p < 0]
                # 企业级：无亏损时 profit_factor = None（JSON null），禁止 float('inf')
                snap.profit_factor = safe_div(sum(profits), sum(losses), None) if losses else None
                if len(pnls) >= 2:
                    mean_pnl = float(np.mean(pnls))
                    std_pnl = float(np.std(pnls, ddof=1))
                    snap.sharpe_approx = safe_div(mean_pnl, std_pnl, 0.0)

        except Exception as e:
            logger.debug(f"review_engine: capital curve collection error: {type(e).__name__}: {e}")
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
                except Exception as e:
                    logger.debug(f"review_engine: failed to load daily review {fname}: {type(e).__name__}: {e}")
        return reviews

    def plan_symbol_pool_update(self, reviews: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """币池更新决策：淘汰+候选替换"""
        sym_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "days": 0, "total_pnl": 0.0, "total_trades": 0, "wins": 0, "max_dd": 0.0,
            "avg_leverage": 0.0, "effective_days": 0,
        })

        for review in reviews:
            for sp in review.get("symbol_performance", []):
                sym = sp["symbol"]
                s = sym_stats[sym]
                s["days"] += 1
                s["total_pnl"] += safe_float(sp.get("net_pnl"), 0.0)
                s["total_trades"] += safe_int(sp.get("trades"), 0)
                s["wins"] += safe_int(sp.get("wins"), 0)
                s["max_dd"] = max(s["max_dd"], abs(safe_float(sp.get("max_drawdown"), 0.0)))
                s["avg_leverage"] += safe_float(sp.get("avg_leverage"), 0.0)
                if sp.get("effective", True):
                    s["effective_days"] += 1

        ranking = []
        for sym, s in sym_stats.items():
            win_rate = safe_div(s["wins"], s["total_trades"], 0.0)
            avg_pnl_per_day = safe_div(s["total_pnl"], s["days"], 0.0)
            effectiveness = safe_div(s["effective_days"], s["days"], 1.0)
            avg_leverage = safe_div(s["avg_leverage"], s["days"], 0.0)

            score = (
                avg_pnl_per_day * 10 +
                win_rate * 50 -
                s["max_dd"] * 100 +
                effectiveness * 20 -
                avg_leverage * 2
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

        to_remove = []
        for r in ranking:
            if r["avg_pnl_per_day"] < -1 and r["max_drawdown"] > 0.05:
                to_remove.append(r["symbol"])
            elif r["effectiveness"] < 0.3 and r["total_trades"] >= 10:
                to_remove.append(r["symbol"])

        candidates = []
        active_symbols = set(r["symbol"] for r in ranking)
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

        strat_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "total_pnl": 0.0, "trades": 0, "wins": 0, "avg_slippage": 0.0,
            "max_dd": 0.0, "interceptions": 0,
        })

        for review in reviews:
            for sp in review.get("symbol_performance", []):
                for strat, se in sp.get("strategy_effectiveness", {}).items():
                    s = strat_stats[strat]
                    s["total_pnl"] += safe_float(se.get("pnl"), 0.0)
                    s["trades"] += safe_int(se.get("trades"), 0)
                    s["wins"] += safe_int(se.get("wins"), 0)

            el = review.get("execution_loss", {})
            for rank in el.get("symbol_slippage_rank", []):
                for strat in strat_stats:
                    strat_stats[strat]["avg_slippage"] += safe_float(rank.get("avg_slippage"), 0.0)

            for ri in review.get("risk_interceptions", []):
                for strat in strat_stats:
                    strat_stats[strat]["interceptions"] += safe_int(ri.get("blocks"), 0)

        for strat, stats in strat_stats.items():
            if stats["trades"] < 10:
                continue

            win_rate = safe_div(stats["wins"], stats["trades"], 0.0)
            avg_pnl = safe_div(stats["total_pnl"], stats["trades"], 0.0)
            adjustments[strat] = {}

            if win_rate < 0.35:
                adjustments[strat]["min_signal_quality"] = {"action": "increase", "reason": f"胜率过低({win_rate:.0%})，收紧信号门槛"}
            elif win_rate > 0.65:
                adjustments[strat]["min_signal_quality"] = {"action": "decrease", "reason": f"胜率过高({win_rate:.0%})，可适当放宽以增交易频率"}

            if avg_pnl < -2:
                adjustments[strat]["stop_loss_pct"] = {"action": "decrease", "reason": f"单笔均亏{avg_pnl:.2f}U，收紧止损"}

            avg_slip = safe_div(stats["avg_slippage"], max(len(reviews), 1), 0.0)
            if avg_slip > 0.001:
                adjustments[strat]["slippage_tolerance"] = {"action": "increase", "reason": f"平均滑点{avg_slip:.4%}，增大容忍度"}

            if stats["interceptions"] > 20:
                adjustments[strat]["position_size"] = {"action": "decrease", "reason": f"月拦截{stats['interceptions']}次，降低单笔规模"}

        return adjustments

    def plan_system_fixes(self, reviews: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """系统架构优化建议：高频漏洞修复"""
        fixes = []

        all_reasons: Dict[str, int] = defaultdict(int)
        total_api_errors = 0
        total_disconnect = 0.0
        total_gaps = 0

        for review in reviews:
            for ri in review.get("risk_interceptions", []):
                for reason, cnt in ri.get("top_reasons", []):
                    all_reasons[reason] += safe_int(cnt, 0)
            el = review.get("execution_loss", {})
            total_api_errors += safe_int(el.get("api_errors"), 0)
            total_disconnect += safe_float(el.get("disconnection_seconds"), 0.0)
            total_gaps += safe_int(el.get("data_gap_count"), 0)

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

        if total_api_errors >= 10:
            fixes.append({
                "category": "api_stability",
                "issue": f"月API错误{total_api_errors}次",
                "frequency": total_api_errors,
                "suggested_fix": "检查代理稳定性、增加请求重试、启用备用REST端点",
                "priority": "high",
            })

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

        total_pnl = sum(safe_float(r.get("capital_curve", {}).get("daily_return_pct"), 0.0) for r in reviews)
        avg_win_rate = float(np.mean([safe_float(r.get("capital_curve", {}).get("win_rate"), 0.0) for r in reviews]))
        max_dd = max(safe_float(r.get("capital_curve", {}).get("max_drawdown_pct"), 0.0) for r in reviews)

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

        confidence = min(1.0, safe_div(len(reviews), 25, 0.0))

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

        self._daily_reports: deque = deque(maxlen=90)
        self._monthly_plans: deque = deque(maxlen=12)

        self._on_daily_review: Any = None
        self._on_monthly_plan: Any = None

    def set_callbacks(self, on_daily_review=None, on_monthly_plan=None):
        """设置通知回调"""
        self._on_daily_review = on_daily_review
        self._on_monthly_plan = on_monthly_plan

    async def run_daily_review(self, date: Optional[datetime] = None) -> DailyReviewReport:
        """执行每日复盘"""
        if date is None:
            date = datetime.now() - timedelta(days=1)

        date_str = date.strftime("%Y-%m-%d")
        logger.info(f"Running daily review for {date_str}")

        sym_perf = self._daily_collector.collect_symbol_performance(date)
        risk_stats = self._daily_collector.collect_risk_interceptions(date, self._risk_gate)
        exec_loss = self._daily_collector.collect_execution_loss(date, self._risk_gate)
        capital = self._daily_collector.collect_capital_curve(date)

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

        self._persist_daily_report(report)
        self._daily_reports.append(report)

        logger.info(f"Daily review completed: {date_str}, trades={total_trades}, pnl={total_net_pnl:.2f}")

        if self._on_daily_review:
            try:
                await self._on_daily_review(report)
            except Exception as e:
                logger.debug(f"Daily review callback error: {type(e).__name__}: {e}")

        if self._risk_gate and hasattr(self._risk_gate, 'reset_interception_stats'):
            try:
                self._risk_gate.reset_interception_stats()
            except Exception as e:
                logger.debug(f"Failed to reset interception stats: {type(e).__name__}: {e}")

        return report

    def _generate_daily_recommendations(self, sym_perf: List[SymbolPerformance],
                                        risk_stats: List[RiskInterceptionStats],
                                        exec_loss: ExecutionLossStats,
                                        capital: CapitalCurveSnapshot) -> List[str]:
        """生成每日复盘建议"""
        recs = []

        bad_symbols = [s for s in sym_perf if not s.effective and s.trades >= 3]
        if bad_symbols:
            recs.append(f"币种失效预警: {', '.join(s.symbol for s in bad_symbols[:3])} — 建议冻结或降低权重")

        top_losers = [s for s in sym_perf if s.net_pnl < -5][:3]
        if top_losers:
            recs.append(f"当日亏损TOP: {', '.join(f'{s.symbol}({s.net_pnl:.1f}U)' for s in top_losers)} — 检查策略适配性")

        total_blocks = sum(r.blocks for r in risk_stats)
        if total_blocks >= 10:
            top_reason = risk_stats[0].top_reasons[0] if risk_stats[0].top_reasons else ("unknown", 0)
            recs.append(f"风控拦截频繁({total_blocks}次)，主因: {top_reason[0]}({top_reason[1]}次) — 建议优化对应参数")

        if exec_loss.tolerance_exceeded_rate > 0.2:
            recs.append(f"滑点超限率{exec_loss.tolerance_exceeded_rate:.1%}偏高 — 建议启用滑点优化执行器或避开高波动时段")
        if exec_loss.api_errors >= 5:
            recs.append(f"API错误{exec_loss.api_errors}次 — 检查网络代理稳定性")
        if exec_loss.disconnection_seconds >= 60:
            recs.append(f"数据断流{exec_loss.disconnection_seconds:.0f}秒 — 检查行情链路")

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
            today = datetime.now()
            month = datetime(today.year, today.month, 1) - timedelta(days=1)

        month_str = month.strftime("%Y-%m")
        logger.info(f"Running monthly iteration plan for {month_str}")

        plan = self._monthly_planner.generate_plan(month)
        self._monthly_plans.append(plan)

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

        if self._on_monthly_plan:
            try:
                await self._on_monthly_plan(plan)
            except Exception as e:
                logger.debug(f"Monthly plan callback error: {type(e).__name__}: {e}")

        return plan

    def _persist_daily_report(self, report: DailyReviewReport):
        """持久化每日复盘报告"""
        path = os.path.join(self._review_dir, f"daily_review_{report.date}.json")
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to persist daily review: {e}")

    def get_daily_report(self, date_str: str) -> Optional[Dict[str, Any]]:
        """获取指定日期复盘报告"""
        path = os.path.join(self._review_dir, f"daily_review_{date_str}.json")
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"review_engine: failed to read daily report {date_str}: {type(e).__name__}: {e}")
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
            except Exception as e:
                logger.debug(f"review_engine: failed to read monthly plan {month_str}: {type(e).__name__}: {e}")
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
                    daily_pnls.append(safe_float(sp.get("net_pnl"), 0.0))
                    daily_trades.append(safe_float(sp.get("trades"), 0.0))
                    daily_win_rates.append(safe_float(sp.get("win_rate"), 0.0))
                    break

        if not daily_pnls:
            return {"symbol": symbol, "days": 0, "trend": "no_data"}

        return {
            "symbol": symbol,
            "days": len(daily_pnls),
            "total_pnl": round(sum(daily_pnls), 4),
            "avg_daily_pnl": round(float(np.mean(daily_pnls)), 4),
            "avg_trades": round(float(np.mean(daily_trades)), 2),
            "avg_win_rate": round(float(np.mean(daily_win_rates)), 4),
            "trend": "improving" if daily_pnls[-1] > daily_pnls[0] else "declining" if len(daily_pnls) >= 2 else "stable",
        }
