"""
历史交易数据分析器：分析各策略、交易对与时间模式并识别短板、提取成功模式。
"""
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from core.trade_journal import TradeJournal


def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


def _safe_datetime(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


class HistoricalAnalyzer:
    def __init__(self, trade_journal: TradeJournal):
        self.trade_journal = trade_journal
        self._analysis_cache: Dict[str, Any] = {}
        self._data_load_failed = False
    
    def analyze_all_strategies(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        self._data_load_failed = False
        stats = {
            "overview": self._analyze_overview(since),
            "strategies": self._analyze_strategies(since),
            "symbols": self._analyze_symbols(since),
            "time_patterns": self._analyze_time_patterns(since),
            "shortcomings": self._identify_shortcomings(since),
            "success_patterns": self._extract_success_patterns(since)
        }
        stats["available"] = not self._data_load_failed
        return stats

    def _get_trades(self, since: Optional[datetime], limit: int = 1000) -> List[Dict[str, Any]]:
        """获取交易列表：since 非 None 时按时间线过滤（仅时间线之后平仓的交易）。"""
        try:
            if since is not None:
                trades = self.trade_journal.get_trades_since(since, limit=limit)
            else:
                trades = self.trade_journal.get_recent_trades(limit=limit)
            return trades or []
        except Exception as e:
            logger.error(f"Failed to load trades from journal: {e}")
            self._data_load_failed = True
            return []

    def _analyze_overview(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        try:
            if since is not None:
                return self.trade_journal.get_trade_stats_since(since) or {}
            return self.trade_journal.get_trade_stats() or {}
        except Exception as e:
            logger.error(f"Failed to analyze overview: {e}")
            self._data_load_failed = True
            return {}
    
    def _analyze_strategies(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        # P0: 包含所有策略类型，spot_grid和spot_martingale之前被遗漏
        strategies = ["grid", "trend", "scalping", "arbitrage", "spot_grid", "spot_martingale"]
        results = []
        
        for strategy in strategies:
            try:
                if since is not None:
                    stats = self.trade_journal.get_strategy_stats_since(since, strategy)
                else:
                    stats = self.trade_journal.get_strategy_stats(strategy)
            except Exception as e:
                logger.error(f"Failed to get strategy stats for {strategy}: {e}")
                self._data_load_failed = True
                stats = None
            stats = stats or {
                "strategy_name": strategy,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0
            }
            avg_win = _safe_float(stats.get("avg_win"))
            avg_loss = _safe_float(stats.get("avg_loss"))
            stats["avg_win"] = avg_win
            stats["avg_loss"] = avg_loss
            # 无亏损时（avg_loss==0）盈亏比无数学意义，用 None 表示「不适用」，
            # 避免输出 float('inf') 污染 JSON 序列化（非标准 Infinity）
            stats["profit_factor"] = abs(avg_win / avg_loss) if avg_loss > 0 else None
            stats["risk_reward"] = abs(avg_win / avg_loss) if avg_loss > 0 else None
            results.append(stats)
        
        return results
    
    def _analyze_symbols(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        all_trades = self._get_trades(since)
        
        symbol_stats = {}
        for trade in all_trades:
            symbol = trade.get("symbol")
            if not symbol:
                continue
            pnl_usdt = _safe_float(trade.get("pnl_usdt"))
            is_win = bool(trade.get("win"))
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {
                    "symbol": symbol,
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0,
                    "avg_win": 0,
                    "avg_loss": 0,
                    "strategies": set()
                }
            
            symbol_stats[symbol]["total_trades"] += 1
            symbol_stats[symbol]["total_pnl"] += pnl_usdt
            symbol_stats[symbol]["strategies"].add(trade.get("strategy_name") or "unknown")
            
            if is_win:
                symbol_stats[symbol]["wins"] += 1
            else:
                symbol_stats[symbol]["losses"] += 1
        
        results = []
        for symbol, stats in symbol_stats.items():
            stats["win_rate"] = stats["wins"] / stats["total_trades"] if stats["total_trades"] > 0 else 0
            stats["strategies"] = list(stats["strategies"])
            
            wins = [_safe_float(t.get("pnl_usdt")) for t in all_trades if t.get("symbol") == symbol and t.get("win")]
            losses = [_safe_float(t.get("pnl_usdt")) for t in all_trades if t.get("symbol") == symbol and not t.get("win")]
            
            avg_win = _safe_float(np.mean(wins)) if wins else 0
            avg_loss = _safe_float(np.mean(losses)) if losses else 0
            stats["avg_win"] = avg_win
            stats["avg_loss"] = avg_loss
            stats["profit_factor"] = abs(avg_win / avg_loss) if avg_loss > 0 else None
            
            results.append(stats)
        
        return sorted(results, key=lambda x: x["total_pnl"], reverse=True)
    
    def _analyze_time_patterns(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        all_trades = self._get_trades(since)
        
        hourly_stats = {}
        for trade in all_trades:
            entry_time = _safe_datetime(trade.get("entry_time"))
            if entry_time is None:
                continue
            hour = entry_time.hour
            pnl_usdt = _safe_float(trade.get("pnl_usdt"))
            is_win = bool(trade.get("win"))
            
            if hour not in hourly_stats:
                hourly_stats[hour] = {
                    "hour": hour,
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0,
                    "avg_pnl": 0
                }
            
            hourly_stats[hour]["total_trades"] += 1
            hourly_stats[hour]["total_pnl"] += pnl_usdt
            
            if is_win:
                hourly_stats[hour]["wins"] += 1
            else:
                hourly_stats[hour]["losses"] += 1
        
        for hour, stats in hourly_stats.items():
            stats["win_rate"] = stats["wins"] / stats["total_trades"] if stats["total_trades"] > 0 else 0
            stats["avg_pnl"] = stats["total_pnl"] / stats["total_trades"] if stats["total_trades"] > 0 else 0
        
        weekday_stats = {}
        for trade in all_trades:
            entry_time = _safe_datetime(trade.get("entry_time"))
            if entry_time is None:
                continue
            weekday = entry_time.weekday()
            pnl_usdt = _safe_float(trade.get("pnl_usdt"))
            is_win = bool(trade.get("win"))
            
            if weekday not in weekday_stats:
                weekday_stats[weekday] = {
                    "weekday": weekday,
                    "day_name": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][weekday],
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0
                }
            
            weekday_stats[weekday]["total_trades"] += 1
            weekday_stats[weekday]["total_pnl"] += pnl_usdt
            
            if is_win:
                weekday_stats[weekday]["wins"] += 1
            else:
                weekday_stats[weekday]["losses"] += 1
        
        for weekday, stats in weekday_stats.items():
            stats["win_rate"] = stats["wins"] / stats["total_trades"] if stats["total_trades"] > 0 else 0
        
        return {
            "hourly": sorted(hourly_stats.values(), key=lambda x: x["hour"]),
            "weekday": sorted(weekday_stats.values(), key=lambda x: x["weekday"]),
            "best_hours": sorted(hourly_stats.values(), key=lambda x: x["avg_pnl"], reverse=True)[:5],
            "worst_hours": sorted(hourly_stats.values(), key=lambda x: x["avg_pnl"])[:5]
        }
    
    def _identify_shortcomings(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        all_trades = self._get_trades(since)
        strategies_stats = self._analyze_strategies(since)
        
        shortcomings = []
        
        for strategy in strategies_stats:
            strategy_name = strategy.get("strategy_name") or "unknown"
            total_trades = _safe_float(strategy.get("total_trades"))
            if total_trades < 10:
                continue
            
            win_rate = _safe_float(strategy.get("win_rate"))
            if win_rate < 0.45:
                shortcomings.append({
                    "category": "win_rate",
                    "strategy": strategy_name,
                    "value": win_rate,
                    "threshold": 0.45,
                    "severity": "high",
                    "description": f"{strategy_name}策略胜率低于45%，需要优化入场条件",
                    "suggestion": "增加信号确认条件，优化指标参数，考虑多时间框架确认"
                })
            
            profit_factor = strategy.get("profit_factor")
            # None 表示无亏损（全胜/无数据），盈亏比不适用，不应误报为「盈亏比低于1.0」
            if profit_factor is not None and profit_factor < 1.0:
                shortcomings.append({
                    "category": "profit_factor",
                    "strategy": strategy_name,
                    "value": profit_factor,
                    "threshold": 1.0,
                    "severity": "high",
                    "description": f"{strategy_name}策略盈亏比低于1.0，亏损交易平均金额大于盈利交易",
                    "suggestion": "调整止盈止损比例，优化出场策略，考虑追踪止损"
                })
            
            total_pnl = _safe_float(strategy.get("total_pnl"))
            if total_pnl < 0:
                shortcomings.append({
                    "category": "negative_pnl",
                    "strategy": strategy_name,
                    "value": total_pnl,
                    "threshold": 0,
                    "severity": "critical",
                    "description": f"{strategy_name}策略总盈亏为负",
                    "suggestion": "全面审查策略逻辑，检查手续费影响，考虑暂时关闭或大幅调整参数"
                })
        
        exit_reasons = {}
        for trade in all_trades:
            reason = trade.get("exit_reason") or "unknown"
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "wins": 0, "total_pnl": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["total_pnl"] += _safe_float(trade.get("pnl_usdt"))
            if trade.get("win"):
                exit_reasons[reason]["wins"] += 1
        
        for reason, stats in exit_reasons.items():
            if stats["count"] > 10:
                win_rate = stats["wins"] / stats["count"]
                if win_rate < 0.3:
                    shortcomings.append({
                        "category": "exit_reason",
                        "exit_reason": reason,
                        "win_rate": win_rate,
                        "count": stats["count"],
                        "severity": "medium",
                        "description": f"因{reason}退出的交易胜率仅为{win_rate:.1%}",
                        "suggestion": f"优化{reason}相关的退出逻辑，调整触发条件"
                    })
        
        symbol_stats = self._analyze_symbols(since)
        for symbol in symbol_stats:
            symbol_name = symbol.get("symbol")
            total_trades = _safe_float(symbol.get("total_trades"))
            win_rate = _safe_float(symbol.get("win_rate"))
            if total_trades > 20 and win_rate < 0.4:
                shortcomings.append({
                    "category": "symbol_performance",
                    "symbol": symbol_name,
                    "win_rate": win_rate,
                    "total_trades": total_trades,
                    "severity": "medium",
                    "description": f"{symbol_name}标的交易胜率低于40%",
                    "suggestion": "减少该标的交易权重，或调整针对该标的的策略参数"
                })
        
        time_patterns = self._analyze_time_patterns(since)
        for hour in time_patterns.get("worst_hours") or []:
            total_trades = _safe_float(hour.get("total_trades"))
            avg_pnl = _safe_float(hour.get("avg_pnl"))
            if total_trades > 10 and avg_pnl < -10:
                shortcomings.append({
                    "category": "time_pattern",
                    "hour": hour.get("hour"),
                    "avg_pnl": avg_pnl,
                    "total_trades": total_trades,
                    "severity": "low",
                    "description": f"{hour.get('hour')}:00时段平均每笔亏损{abs(avg_pnl):.2f} USDT",
                    "suggestion": "考虑在该时段减少交易频率，或调整策略参数"
                })
        
        return {
            "available": not self._data_load_failed,
            "total_shortcomings": len(shortcomings),
            "critical": [s for s in shortcomings if s["severity"] == "critical"],
            "high": [s for s in shortcomings if s["severity"] == "high"],
            "medium": [s for s in shortcomings if s["severity"] == "medium"],
            "low": [s for s in shortcomings if s["severity"] == "low"],
            "all": shortcomings
        }
    
    def _extract_success_patterns(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        all_trades = self._get_trades(since)
        
        profitable_trades = [t for t in all_trades if t.get("win") and _safe_float(t.get("pnl_usdt")) > 0]
        
        if len(profitable_trades) < 10:
            return {
                "available": not self._data_load_failed,
                "success_trades_count": len(profitable_trades),
                "patterns": [],
                "key_factors": []
            }
        
        strategy_success = {}
        for trade in profitable_trades:
            strategy = trade.get("strategy_name") or "unknown"
            if strategy not in strategy_success:
                strategy_success[strategy] = []
            strategy_success[strategy].append(_safe_float(trade.get("pnl_usdt")))
        
        success_patterns = []
        for strategy, pnls in strategy_success.items():
            avg_pnl = _safe_float(np.mean(pnls))
            max_pnl = max(pnls)
            count = len(pnls)
            
            success_patterns.append({
                "strategy": strategy,
                "count": count,
                "avg_pnl": avg_pnl,
                "max_pnl": max_pnl,
                "description": f"{strategy}策略盈利交易平均盈利{avg_pnl:.2f} USDT，最高{max_pnl:.2f} USDT"
            })
        
        time_success = {}
        for trade in profitable_trades:
            entry_time = _safe_datetime(trade.get("entry_time"))
            if entry_time is None:
                continue
            hour = entry_time.hour
            if hour not in time_success:
                time_success[hour] = []
            time_success[hour].append(_safe_float(trade.get("pnl_usdt")))
        
        top_hours = sorted(time_success.items(), key=lambda x: np.mean(x[1]), reverse=True)[:3]
        hour_patterns = []
        for hour, pnls in top_hours:
            avg_pnl = _safe_float(np.mean(pnls))
            hour_patterns.append({
                "hour": hour,
                "count": len(pnls),
                "avg_pnl": avg_pnl,
                "description": f"{hour}:00时段盈利交易平均盈利{avg_pnl:.2f} USDT"
            })
        
        symbol_success = {}
        for trade in profitable_trades:
            symbol = trade.get("symbol")
            if not symbol:
                continue
            if symbol not in symbol_success:
                symbol_success[symbol] = []
            symbol_success[symbol].append(_safe_float(trade.get("pnl_usdt")))
        
        top_symbols = sorted(symbol_success.items(), key=lambda x: np.mean(x[1]), reverse=True)[:5]
        symbol_patterns = []
        for symbol, pnls in top_symbols:
            avg_pnl = _safe_float(np.mean(pnls))
            symbol_patterns.append({
                "symbol": symbol,
                "count": len(pnls),
                "avg_pnl": avg_pnl,
                "description": f"{symbol}标的盈利交易平均盈利{avg_pnl:.2f} USDT"
            })
        
        exit_reason_success = {}
        for trade in profitable_trades:
            reason = trade.get("exit_reason") or "unknown"
            if reason not in exit_reason_success:
                exit_reason_success[reason] = []
            exit_reason_success[reason].append(_safe_float(trade.get("pnl_usdt")))
        
        top_reasons = sorted(exit_reason_success.items(), key=lambda x: np.mean(x[1]), reverse=True)[:3]
        reason_patterns = []
        for reason, pnls in top_reasons:
            avg_pnl = _safe_float(np.mean(pnls))
            reason_patterns.append({
                "exit_reason": reason,
                "count": len(pnls),
                "avg_pnl": avg_pnl,
                "description": f"因{reason}退出的盈利交易平均盈利{avg_pnl:.2f} USDT"
            })
        
        key_factors = []
        overview = self._analyze_overview(since) or {}
        
        win_rate = _safe_float(overview.get("win_rate"))
        if win_rate > 0.55:
            key_factors.append({
                "factor": "win_rate",
                "value": win_rate,
                "description": "整体胜率高于55%，表明入场信号质量较好",
                "action": "保持现有入场条件，优化出场策略以扩大盈利"
            })
        
        profit_factor = overview.get("profit_factor")
        if profit_factor is not None and profit_factor > 1.5:
            key_factors.append({
                "factor": "profit_factor",
                "value": profit_factor,
                "description": "盈亏比高于1.5，表明盈利交易平均金额显著大于亏损交易",
                "action": "继续优化止损策略，控制亏损规模"
            })
        
        for pattern in success_patterns:
            if pattern["avg_pnl"] > 50:
                key_factors.append({
                    "factor": "strategy_performance",
                    "strategy": pattern["strategy"],
                    "value": pattern["avg_pnl"],
                    "description": f"{pattern['strategy']}策略表现优秀",
                    "action": "增加该策略资金分配比例"
                })
        
        return {
            "available": not self._data_load_failed,
            "success_trades_count": len(profitable_trades),
            "patterns": success_patterns,
            "hour_patterns": hour_patterns,
            "symbol_patterns": symbol_patterns,
            "exit_reason_patterns": reason_patterns,
            "key_factors": key_factors
        }
    
    def generate_analysis_report(self) -> Dict[str, Any]:
        analysis = self.analyze_all_strategies()
        
        report = {
            "report_time": datetime.now().isoformat(),
            "summary": self._generate_summary(analysis),
            "detailed_analysis": analysis
        }
        
        return report
    
    def _generate_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        overview = analysis.get("overview") or {}
        total_return = _safe_float(overview.get("total_return"))
        profit_factor = overview.get("profit_factor")
        if profit_factor is None:
            profit_factor = 0
        
        summary = {
            "total_trades": overview.get("total_trades") or 0,
            "win_rate": _safe_float(overview.get("win_rate")),
            "total_pnl": _safe_float(overview.get("total_pnl")),
            "total_return": total_return,
            "max_drawdown": _safe_float(overview.get("max_drawdown")),
            "sharpe_ratio": _safe_float(overview.get("sharpe_ratio")),
            "profit_factor": profit_factor,
            "current_equity": _safe_float(overview.get("current_equity")),
            "starting_capital": _safe_float(overview.get("starting_capital")),
            "shortcomings_count": (analysis.get("shortcomings") or {}).get("total_shortcomings", 0),
            "critical_shortcomings": len((analysis.get("shortcomings") or {}).get("critical", [])),
            "success_patterns_count": len((analysis.get("success_patterns") or {}).get("patterns", []))
        }
        
        if total_return > 0.1:
            summary["performance_rating"] = "优秀"
            summary["performance_comment"] = "资金增长超过10%，策略表现良好"
        elif total_return > 0:
            summary["performance_rating"] = "良好"
            summary["performance_comment"] = "资金正增长，继续优化可提升表现"
        elif total_return > -0.05:
            summary["performance_rating"] = "一般"
            summary["performance_comment"] = "小幅亏损，需要调整策略参数"
        else:
            summary["performance_rating"] = "差"
            summary["performance_comment"] = "亏损超过5%，需要全面审查策略"
        
        return summary