"""
历史交易数据分析器：分析各策略、交易对与时间模式并识别短板、提取成功模式。
"""
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from core.trade_journal import TradeJournal

class HistoricalAnalyzer:
    def __init__(self, trade_journal: TradeJournal):
        self.trade_journal = trade_journal
        self._analysis_cache: Dict[str, Any] = {}
    
    def analyze_all_strategies(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        stats = {
            "overview": self._analyze_overview(since),
            "strategies": self._analyze_strategies(since),
            "symbols": self._analyze_symbols(since),
            "time_patterns": self._analyze_time_patterns(since),
            "shortcomings": self._identify_shortcomings(since),
            "success_patterns": self._extract_success_patterns(since)
        }
        return stats

    def _get_trades(self, since: Optional[datetime], limit: int = 1000) -> List[Dict[str, Any]]:
        """获取交易列表：since 非 None 时按时间线过滤（仅时间线之后平仓的交易）。"""
        if since is not None:
            return self.trade_journal.get_trades_since(since, limit=limit)
        return self.trade_journal.get_recent_trades(limit=limit)

    def _analyze_overview(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        if since is not None:
            return self.trade_journal.get_trade_stats_since(since)
        return self.trade_journal.get_trade_stats()
    
    def _analyze_strategies(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        # P0: 包含所有策略类型，spot_grid和spot_martingale之前被遗漏
        strategies = ["grid", "trend", "scalping", "arbitrage", "spot_grid", "spot_martingale"]
        results = []
        
        for strategy in strategies:
            if since is not None:
                stats = self.trade_journal.get_strategy_stats_since(since, strategy)
            else:
                stats = self.trade_journal.get_strategy_stats(strategy)
            stats["profit_factor"] = abs(stats["avg_win"] / stats["avg_loss"]) if stats["avg_loss"] != 0 else float('inf')
            stats["risk_reward"] = abs(stats["avg_win"] / stats["avg_loss"]) if stats["avg_loss"] != 0 else float('inf')
            results.append(stats)
        
        return results
    
    def _analyze_symbols(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        all_trades = self._get_trades(since)
        
        symbol_stats = {}
        for trade in all_trades:
            symbol = trade["symbol"]
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
            symbol_stats[symbol]["total_pnl"] += trade["pnl_usdt"]
            symbol_stats[symbol]["strategies"].add(trade["strategy_name"])
            
            if trade["win"]:
                symbol_stats[symbol]["wins"] += 1
            else:
                symbol_stats[symbol]["losses"] += 1
        
        results = []
        for symbol, stats in symbol_stats.items():
            stats["win_rate"] = stats["wins"] / stats["total_trades"] if stats["total_trades"] > 0 else 0
            stats["strategies"] = list(stats["strategies"])
            
            wins = [t["pnl_usdt"] for t in all_trades if t["symbol"] == symbol and t["win"]]
            losses = [t["pnl_usdt"] for t in all_trades if t["symbol"] == symbol and not t["win"]]
            
            stats["avg_win"] = np.mean(wins) if wins else 0
            stats["avg_loss"] = np.mean(losses) if losses else 0
            stats["profit_factor"] = abs(stats["avg_win"] / stats["avg_loss"]) if stats["avg_loss"] != 0 else float('inf')
            
            results.append(stats)
        
        return sorted(results, key=lambda x: x["total_pnl"], reverse=True)
    
    def _analyze_time_patterns(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        all_trades = self._get_trades(since)
        
        hourly_stats = {}
        for trade in all_trades:
            entry_time = datetime.fromisoformat(trade["entry_time"])
            hour = entry_time.hour
            
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
            hourly_stats[hour]["total_pnl"] += trade["pnl_usdt"]
            
            if trade["win"]:
                hourly_stats[hour]["wins"] += 1
            else:
                hourly_stats[hour]["losses"] += 1
        
        for hour, stats in hourly_stats.items():
            stats["win_rate"] = stats["wins"] / stats["total_trades"] if stats["total_trades"] > 0 else 0
            stats["avg_pnl"] = stats["total_pnl"] / stats["total_trades"] if stats["total_trades"] > 0 else 0
        
        weekday_stats = {}
        for trade in all_trades:
            entry_time = datetime.fromisoformat(trade["entry_time"])
            weekday = entry_time.weekday()
            
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
            weekday_stats[weekday]["total_pnl"] += trade["pnl_usdt"]
            
            if trade["win"]:
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
            if strategy["total_trades"] < 10:
                continue
            
            if strategy["win_rate"] < 0.45:
                shortcomings.append({
                    "category": "win_rate",
                    "strategy": strategy["strategy_name"],
                    "value": strategy["win_rate"],
                    "threshold": 0.45,
                    "severity": "high",
                    "description": f"{strategy['strategy_name']}策略胜率低于45%，需要优化入场条件",
                    "suggestion": "增加信号确认条件，优化指标参数，考虑多时间框架确认"
                })
            
            if strategy["profit_factor"] < 1.0:
                shortcomings.append({
                    "category": "profit_factor",
                    "strategy": strategy["strategy_name"],
                    "value": strategy["profit_factor"],
                    "threshold": 1.0,
                    "severity": "high",
                    "description": f"{strategy['strategy_name']}策略盈亏比低于1.0，亏损交易平均金额大于盈利交易",
                    "suggestion": "调整止盈止损比例，优化出场策略，考虑追踪止损"
                })
            
            if strategy["total_pnl"] < 0:
                shortcomings.append({
                    "category": "negative_pnl",
                    "strategy": strategy["strategy_name"],
                    "value": strategy["total_pnl"],
                    "threshold": 0,
                    "severity": "critical",
                    "description": f"{strategy['strategy_name']}策略总盈亏为负",
                    "suggestion": "全面审查策略逻辑，检查手续费影响，考虑暂时关闭或大幅调整参数"
                })
        
        exit_reasons = {}
        for trade in all_trades:
            reason = trade.get("exit_reason", "unknown")
            if reason not in exit_reasons:
                exit_reasons[reason] = {"count": 0, "wins": 0, "total_pnl": 0}
            exit_reasons[reason]["count"] += 1
            exit_reasons[reason]["total_pnl"] += trade["pnl_usdt"]
            if trade["win"]:
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
            if symbol["total_trades"] > 20 and symbol["win_rate"] < 0.4:
                shortcomings.append({
                    "category": "symbol_performance",
                    "symbol": symbol["symbol"],
                    "win_rate": symbol["win_rate"],
                    "total_trades": symbol["total_trades"],
                    "severity": "medium",
                    "description": f"{symbol['symbol']}标的交易胜率低于40%",
                    "suggestion": "减少该标的交易权重，或调整针对该标的的策略参数"
                })
        
        time_patterns = self._analyze_time_patterns(since)
        for hour in time_patterns["worst_hours"]:
            if hour["total_trades"] > 10 and hour["avg_pnl"] < -10:
                shortcomings.append({
                    "category": "time_pattern",
                    "hour": hour["hour"],
                    "avg_pnl": hour["avg_pnl"],
                    "total_trades": hour["total_trades"],
                    "severity": "low",
                    "description": f"{hour['hour']}:00时段平均每笔亏损{abs(hour['avg_pnl']):.2f} USDT",
                    "suggestion": "考虑在该时段减少交易频率，或调整策略参数"
                })
        
        return {
            "total_shortcomings": len(shortcomings),
            "critical": [s for s in shortcomings if s["severity"] == "critical"],
            "high": [s for s in shortcomings if s["severity"] == "high"],
            "medium": [s for s in shortcomings if s["severity"] == "medium"],
            "low": [s for s in shortcomings if s["severity"] == "low"],
            "all": shortcomings
        }
    
    def _extract_success_patterns(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        all_trades = self._get_trades(since)
        
        profitable_trades = [t for t in all_trades if t["win"] and t["pnl_usdt"] > 0]
        
        if len(profitable_trades) < 10:
            return {
                "success_trades_count": len(profitable_trades),
                "patterns": [],
                "key_factors": []
            }
        
        strategy_success = {}
        for trade in profitable_trades:
            strategy = trade["strategy_name"]
            if strategy not in strategy_success:
                strategy_success[strategy] = []
            strategy_success[strategy].append(trade["pnl_usdt"])
        
        success_patterns = []
        for strategy, pnls in strategy_success.items():
            avg_pnl = np.mean(pnls)
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
            hour = datetime.fromisoformat(trade["entry_time"]).hour
            if hour not in time_success:
                time_success[hour] = []
            time_success[hour].append(trade["pnl_usdt"])
        
        top_hours = sorted(time_success.items(), key=lambda x: np.mean(x[1]), reverse=True)[:3]
        hour_patterns = [{
            "hour": hour,
            "count": len(pnls),
            "avg_pnl": np.mean(pnls),
            "description": f"{hour}:00时段盈利交易平均盈利{np.mean(pnls):.2f} USDT"
        } for hour, pnls in top_hours]
        
        symbol_success = {}
        for trade in profitable_trades:
            symbol = trade["symbol"]
            if symbol not in symbol_success:
                symbol_success[symbol] = []
            symbol_success[symbol].append(trade["pnl_usdt"])
        
        top_symbols = sorted(symbol_success.items(), key=lambda x: np.mean(x[1]), reverse=True)[:5]
        symbol_patterns = [{
            "symbol": symbol,
            "count": len(pnls),
            "avg_pnl": np.mean(pnls),
            "description": f"{symbol}标的盈利交易平均盈利{np.mean(pnls):.2f} USDT"
        } for symbol, pnls in top_symbols]
        
        exit_reason_success = {}
        for trade in profitable_trades:
            reason = trade.get("exit_reason", "unknown")
            if reason not in exit_reason_success:
                exit_reason_success[reason] = []
            exit_reason_success[reason].append(trade["pnl_usdt"])
        
        top_reasons = sorted(exit_reason_success.items(), key=lambda x: np.mean(x[1]), reverse=True)[:3]
        reason_patterns = [{
            "exit_reason": reason,
            "count": len(pnls),
            "avg_pnl": np.mean(pnls),
            "description": f"因{reason}退出的盈利交易平均盈利{np.mean(pnls):.2f} USDT"
        } for reason, pnls in top_reasons]
        
        key_factors = []
        overview = self._analyze_overview(since)
        
        if overview["win_rate"] > 0.55:
            key_factors.append({
                "factor": "win_rate",
                "value": overview["win_rate"],
                "description": "整体胜率高于55%，表明入场信号质量较好",
                "action": "保持现有入场条件，优化出场策略以扩大盈利"
            })
        
        if overview["profit_factor"] > 1.5:
            key_factors.append({
                "factor": "profit_factor",
                "value": overview["profit_factor"],
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
        overview = analysis["overview"]
        
        summary = {
            "total_trades": overview["total_trades"],
            "win_rate": overview["win_rate"],
            "total_pnl": overview["total_pnl"],
            "total_return": overview["total_return"],
            "max_drawdown": overview["max_drawdown"],
            "sharpe_ratio": overview["sharpe_ratio"],
            "profit_factor": overview["profit_factor"],
            "current_equity": overview["current_equity"],
            "starting_capital": overview["starting_capital"],
            "shortcomings_count": analysis["shortcomings"]["total_shortcomings"],
            "critical_shortcomings": len(analysis["shortcomings"]["critical"]),
            "success_patterns_count": len(analysis["success_patterns"]["patterns"])
        }
        
        if overview["total_return"] > 0.1:
            summary["performance_rating"] = "优秀"
            summary["performance_comment"] = "资金增长超过10%，策略表现良好"
        elif overview["total_return"] > 0:
            summary["performance_rating"] = "良好"
            summary["performance_comment"] = "资金正增长，继续优化可提升表现"
        elif overview["total_return"] > -0.05:
            summary["performance_rating"] = "一般"
            summary["performance_comment"] = "小幅亏损，需要调整策略参数"
        else:
            summary["performance_rating"] = "差"
            summary["performance_comment"] = "亏损超过5%，需要全面审查策略"
        
        return summary