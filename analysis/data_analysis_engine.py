"""
多维数据分析引擎
提供策略绩效、交易统计、市场分析、组合风险等维度的综合分析能力。
"""
import asyncio
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger


class DataAnalysisEngine:
    """多维数据分析引擎"""

    def __init__(self, config: Dict[str, Any], sqlite_storage=None, trade_journal=None):
        self.config = config
        self._sqlite_storage = sqlite_storage
        self._trade_journal = trade_journal
        
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 60
        
        logger.info("DataAnalysisEngine initialized")

    def _get_cached_value(self, key: str) -> Any:
        """命中且未过期的缓存值；未命中或过期返回 None。"""
        cached = self._cache.get(key)
        if cached and time.time() - cached.get("_ts", 0) < self._cache_ttl:
            return cached.get("data")
        return None

    def _set_cached_value(self, key: str, data: Any):
        self._cache[key] = {"_ts": time.time(), "data": data}

    def clear_cache(self):
        """清空分析缓存（数据源变化或手动刷新时调用）。"""
        self._cache.clear()

    # ===================== 策略绩效分析 =====================

    def analyze_strategy_performance(self, strategy_name: str = None, 
                                     start_date: str = None, 
                                     end_date: str = None) -> Dict[str, Any]:
        """分析策略绩效（支持多策略对比）"""
        cache_key = f"perf:{strategy_name}:{start_date}:{end_date}"
        cached = self._get_cached_value(cache_key)
        if cached is not None:
            return cached

        strategies = [strategy_name] if strategy_name else ["grid", "trend", "scalping", "arbitrage", "spot_grid", "spot_martingale"]
        
        results = {}
        
        for strategy in strategies:
            records = self._get_trade_records(strategy, start_date, end_date)
            
            if not records:
                results[strategy] = self._empty_performance(strategy)
                continue
            
            results[strategy] = self._compute_strategy_metrics(records, strategy)
        
        if len(results) > 1:
            results["comparison"] = self._compare_strategies(results)
        
        self._set_cached_value(cache_key, results)
        return results

    def _get_trade_records(self, strategy_name: str, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """获取交易记录"""
        if not self._sqlite_storage:
            return []
        
        try:
            records = self._sqlite_storage.get_trade_records(
                strategy_name=strategy_name,
                limit=1000
            )
            
            filtered = []
            for r in records:
                if r.get("status") != "closed":
                    continue
                
                close_time = r.get("close_time", "")
                if close_time:
                    try:
                        if isinstance(close_time, str):
                            ct = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                        else:
                            ct = close_time
                        
                        if start_date:
                            sd = datetime.fromisoformat(start_date)
                            if ct < sd:
                                continue
                        if end_date:
                            ed = datetime.fromisoformat(end_date)
                            if ct > ed:
                                continue
                        
                        filtered.append(r)
                    except Exception:
                        continue
            
            return filtered
        except Exception as e:
            logger.debug(f"Failed to get trade records: {e}")
            return []

    def _compute_strategy_metrics(self, records: List[Dict[str, Any]], strategy_name: str) -> Dict[str, Any]:
        """计算策略绩效指标"""
        if not records:
            return self._empty_performance(strategy_name)
        
        closed = [r for r in records if r.get("status") == "closed"]
        wins = [r for r in closed if r.get("pnl", 0) > 0]
        losses = [r for r in closed if r.get("pnl", 0) < 0]
        
        total_trades = len(closed)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0.5
        
        total_pnl = sum(r.get("pnl", 0) for r in closed)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([r["pnl"] for r in wins]) if wins else 0
        avg_loss = abs(np.mean([r["pnl"] for r in losses])) if losses else 1
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        equity_curve = self._compute_equity_curve(closed)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        returns = self._compute_returns(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        avg_holding_period = self._calculate_avg_holding_period(closed)
        
        largest_win = max([r["pnl"] for r in wins], default=0)
        largest_loss = min([r["pnl"] for r in losses], default=0)
        
        consecutive_wins, consecutive_losses = self._calculate_consecutive_streaks(closed)
        
        return {
            "strategy": strategy_name,
            "total_trades": total_trades,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(total_pnl, 4),
            "avg_pnl": round(avg_pnl, 4),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "max_drawdown": round(max_drawdown, 4),
            "sharpe_ratio": round(sharpe_ratio, 4),
            "avg_holding_minutes": round(avg_holding_period, 2),
            "largest_win": round(largest_win, 4),
            "largest_loss": round(largest_loss, 4),
            "consecutive_wins": consecutive_wins,
            "consecutive_losses": consecutive_losses,
            "equity_curve": equity_curve,
            "returns": returns,
            "risk_adjusted_return": round(total_pnl / max(max_drawdown, 0.01), 4) if total_pnl > 0 else 0,
        }

    def _empty_performance(self, strategy_name: str) -> Dict[str, Any]:
        """空绩效数据"""
        return {
            "strategy": strategy_name,
            "total_trades": 0,
            "win_rate": 0.5,
            "total_pnl": 0,
            "avg_pnl": 0,
            "avg_win": 0,
            "avg_loss": 0,
            "profit_factor": 1.0,
            "max_drawdown": 0,
            "sharpe_ratio": 0,
            "avg_holding_minutes": 0,
            "largest_win": 0,
            "largest_loss": 0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "equity_curve": [],
            "returns": [],
            "risk_adjusted_return": 0,
        }

    def _compare_strategies(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """多策略对比"""
        strategies = {k: v for k, v in results.items() if k != "comparison"}
        
        best_pnl = max(strategies.values(), key=lambda x: x["total_pnl"])
        best_win_rate = max(strategies.values(), key=lambda x: x["win_rate"])
        best_profit_factor = max(strategies.values(), key=lambda x: x["profit_factor"])
        best_sharpe = max(strategies.values(), key=lambda x: x["sharpe_ratio"])
        
        total_pnl = sum(s["total_pnl"] for s in strategies.values())
        total_trades = sum(s["total_trades"] for s in strategies.values())
        
        return {
            "best_pnl": {"strategy": best_pnl["strategy"], "value": best_pnl["total_pnl"]},
            "best_win_rate": {"strategy": best_win_rate["strategy"], "value": best_win_rate["win_rate"]},
            "best_profit_factor": {"strategy": best_profit_factor["strategy"], "value": best_profit_factor["profit_factor"]},
            "best_sharpe_ratio": {"strategy": best_sharpe["strategy"], "value": best_sharpe["sharpe_ratio"]},
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "strategy_count": len(strategies),
        }

    # ===================== 交易统计 =====================

    def analyze_trading_statistics(self, start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """分析交易统计"""
        cache_key = f"stats:{start_date}:{end_date}"
        cached = self._get_cached_value(cache_key)
        if cached is not None:
            return cached

        records = self._get_all_trade_records(start_date, end_date)
        
        if not records:
            result = self._empty_trading_stats()
        else:
            result = {
                "overview": self._compute_trade_overview(records),
                "daily_stats": self._compute_daily_stats(records),
                "hourly_stats": self._compute_hourly_stats(records),
                "symbol_stats": self._compute_symbol_stats(records),
                "fee_analysis": self._compute_fee_analysis(records),
            }

        self._set_cached_value(cache_key, result)
        return result

    def _get_all_trade_records(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """获取所有交易记录"""
        if not self._sqlite_storage:
            return []
        
        try:
            records = self._sqlite_storage.get_trade_records(limit=5000)
            
            filtered = []
            for r in records:
                if r.get("status") != "closed":
                    continue
                
                close_time = r.get("close_time", "")
                if close_time:
                    try:
                        if isinstance(close_time, str):
                            ct = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                        else:
                            ct = close_time
                        
                        if start_date:
                            sd = datetime.fromisoformat(start_date)
                            if ct < sd:
                                continue
                        if end_date:
                            ed = datetime.fromisoformat(end_date)
                            if ct > ed:
                                continue
                        
                        filtered.append(r)
                    except Exception:
                        continue
            
            return filtered
        except Exception:
            return []

    def _compute_trade_overview(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """交易概览"""
        closed = [r for r in records if r.get("status") == "closed"]
        wins = [r for r in closed if r.get("pnl", 0) > 0]
        losses = [r for r in closed if r.get("pnl", 0) < 0]
        
        return {
            "total_trades": len(closed),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(len(wins) / max(len(closed), 1), 4),
            "total_pnl": round(sum(r.get("pnl", 0) for r in closed), 4),
            "total_fees": round(sum(r.get("fees", 0) for r in closed), 4),
            "net_pnl": round(sum(r.get("pnl", 0) - (r.get("fees", 0) or 0) for r in closed), 4),
        }

    def _compute_daily_stats(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """每日统计"""
        daily = {}
        
        for r in records:
            close_time = r.get("close_time", "")
            try:
                if isinstance(close_time, str):
                    ct = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                else:
                    ct = close_time
                
                date_key = ct.strftime("%Y-%m-%d")
                
                if date_key not in daily:
                    daily[date_key] = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "pnl": 0,
                        "fees": 0,
                    }
                
                daily[date_key]["trades"] += 1
                daily[date_key]["pnl"] += r.get("pnl", 0)
                daily[date_key]["fees"] += r.get("fees", 0) or 0
                
                if r.get("pnl", 0) > 0:
                    daily[date_key]["wins"] += 1
                else:
                    daily[date_key]["losses"] += 1
            except Exception:
                continue
        
        sorted_dates = sorted(daily.keys())
        
        return {
            "daily": [
                {
                    "date": date,
                    "trades": daily[date]["trades"],
                    "wins": daily[date]["wins"],
                    "losses": daily[date]["losses"],
                    "pnl": round(daily[date]["pnl"], 4),
                    "fees": round(daily[date]["fees"], 4),
                    "win_rate": round(daily[date]["wins"] / max(daily[date]["trades"], 1), 4),
                }
                for date in sorted_dates[-30:]
            ],
            "best_day": max(daily.items(), key=lambda x: x[1]["pnl"], default=(None, {"pnl": 0})),
            "worst_day": min(daily.items(), key=lambda x: x[1]["pnl"], default=(None, {"pnl": 0})),
        }

    def _compute_hourly_stats(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """小时统计"""
        hourly = {}
        
        for r in records:
            close_time = r.get("close_time", "")
            try:
                if isinstance(close_time, str):
                    ct = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                else:
                    ct = close_time
                
                hour_key = ct.hour
                
                if hour_key not in hourly:
                    hourly[hour_key] = {
                        "trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "pnl": 0,
                    }
                
                hourly[hour_key]["trades"] += 1
                hourly[hour_key]["pnl"] += r.get("pnl", 0)
                
                if r.get("pnl", 0) > 0:
                    hourly[hour_key]["wins"] += 1
                else:
                    hourly[hour_key]["losses"] += 1
            except Exception:
                continue
        
        return {
            "hourly": [
                {
                    "hour": h,
                    "trades": hourly[h]["trades"],
                    "wins": hourly[h]["wins"],
                    "losses": hourly[h]["losses"],
                    "pnl": round(hourly[h]["pnl"], 4),
                    "win_rate": round(hourly[h]["wins"] / max(hourly[h]["trades"], 1), 4),
                }
                for h in sorted(hourly.keys())
            ],
            "best_hour": max(hourly.items(), key=lambda x: x[1]["pnl"], default=(None, {"pnl": 0})),
            "worst_hour": min(hourly.items(), key=lambda x: x[1]["pnl"], default=(None, {"pnl": 0})),
        }

    def _compute_symbol_stats(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """标的统计"""
        symbols = {}
        
        for r in records:
            symbol = r.get("symbol", "")
            
            if symbol not in symbols:
                symbols[symbol] = {
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "pnl": 0,
                }
            
            symbols[symbol]["trades"] += 1
            symbols[symbol]["pnl"] += r.get("pnl", 0)
            
            if r.get("pnl", 0) > 0:
                symbols[symbol]["wins"] += 1
            else:
                symbols[symbol]["losses"] += 1
        
        sorted_symbols = sorted(symbols.items(), key=lambda x: x[1]["trades"], reverse=True)
        
        return {
            "symbols": [
                {
                    "symbol": s,
                    "trades": symbols[s]["trades"],
                    "wins": symbols[s]["wins"],
                    "losses": symbols[s]["losses"],
                    "pnl": round(symbols[s]["pnl"], 4),
                    "win_rate": round(symbols[s]["wins"] / max(symbols[s]["trades"], 1), 4),
                }
                for s, _ in sorted_symbols[:10]
            ],
            "top_performer": max(symbols.items(), key=lambda x: x[1]["pnl"], default=(None, {"pnl": 0})),
            "worst_performer": min(symbols.items(), key=lambda x: x[1]["pnl"], default=(None, {"pnl": 0})),
        }

    def _compute_fee_analysis(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """手续费分析"""
        total_fees = 0
        total_pnl = 0
        
        for r in records:
            total_fees += r.get("fees", 0) or 0
            total_pnl += r.get("pnl", 0)
        
        net_pnl = total_pnl - total_fees
        fee_ratio = total_fees / max(abs(total_pnl), 0.01) if total_pnl != 0 else 0
        
        return {
            "total_fees": round(total_fees, 4),
            "total_pnl": round(total_pnl, 4),
            "net_pnl": round(net_pnl, 4),
            "fee_to_pnl_ratio": round(fee_ratio, 4),
            "fee_per_trade": round(total_fees / max(len(records), 1), 4),
        }

    def _empty_trading_stats(self) -> Dict[str, Any]:
        """空交易统计"""
        return {
            "overview": {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "total_fees": 0,
                "net_pnl": 0,
            },
            "daily_stats": {"daily": [], "best_day": (None, {"pnl": 0}), "worst_day": (None, {"pnl": 0})},
            "hourly_stats": {"hourly": [], "best_hour": (None, {"pnl": 0}), "worst_hour": (None, {"pnl": 0})},
            "symbol_stats": {"symbols": [], "top_performer": (None, {"pnl": 0}), "worst_performer": (None, {"pnl": 0})},
            "fee_analysis": {"total_fees": 0, "total_pnl": 0, "net_pnl": 0, "fee_to_pnl_ratio": 0, "fee_per_trade": 0},
        }

    # ===================== 辅助函数 =====================

    def _compute_equity_curve(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """计算权益曲线"""
        sorted_records = sorted(records, key=lambda x: x.get("close_time", ""))
        
        equity = 0
        curve = []
        
        for r in sorted_records:
            equity += r.get("pnl", 0)
            curve.append({
                "time": r.get("close_time", ""),
                "equity": round(equity, 4),
            })
        
        return curve

    def _compute_returns(self, equity_curve: List[Dict[str, Any]]) -> List[float]:
        """计算收益率序列"""
        returns = []
        
        for i in range(1, len(equity_curve)):
            prev_eq = equity_curve[i-1]["equity"]
            curr_eq = equity_curve[i]["equity"]
            
            if prev_eq != 0:
                returns.append((curr_eq - prev_eq) / abs(prev_eq))
        
        return returns

    def _calculate_max_drawdown(self, equity_curve: List[Dict[str, Any]]) -> float:
        """计算最大回撤"""
        if not equity_curve:
            return 0.0
        
        equities = [e["equity"] for e in equity_curve]
        max_equity = equities[0]
        max_dd = 0.0
        
        for eq in equities[1:]:
            max_equity = max(max_equity, eq)
            if max_equity > 0:
                dd = (max_equity - eq) / max_equity
                max_dd = max(max_dd, dd)
        
        return max_dd

    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.0) -> float:
        """计算夏普比率"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = [r - risk_free_rate for r in returns]
        mean_return = np.mean(excess_returns)
        std_dev = np.std(excess_returns)
        
        if std_dev == 0:
            return 0.0
        
        return mean_return / std_dev

    def _calculate_avg_holding_period(self, records: List[Dict[str, Any]]) -> float:
        """计算平均持仓时间"""
        total_minutes = 0
        count = 0
        
        for r in records:
            create_time = r.get("create_time", "")
            close_time = r.get("close_time", "")
            
            try:
                if isinstance(create_time, str):
                    ct = datetime.fromisoformat(create_time.replace('Z', '+00:00'))
                else:
                    ct = create_time
                
                if isinstance(close_time, str):
                    lt = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
                else:
                    lt = close_time
                
                diff = (lt - ct).total_seconds() / 60
                if diff > 0:
                    total_minutes += diff
                    count += 1
            except Exception:
                continue
        
        return total_minutes / max(count, 1) if count > 0 else 0

    def _calculate_consecutive_streaks(self, records: List[Dict[str, Any]]) -> Tuple[int, int]:
        """基于完整平仓序列计算最大连胜/连亏（按 close_time 排序的真实连续口径）。

        与旧实现（分别在盈利/亏损子集中按 1 小时时间间隔判断）不同，这里在
        全部已平仓交易序列中统计连续盈利与连续亏损的最大长度，避免口径失真。
        """
        if not records:
            return 0, 0

        sorted_records = sorted(records, key=lambda x: x.get("close_time", ""))
        max_wins = 0
        max_losses = 0
        cur_wins = 0
        cur_losses = 0

        for r in sorted_records:
            pnl = r.get("pnl", 0) or 0
            if pnl > 0:
                cur_wins += 1
                cur_losses = 0
            elif pnl < 0:
                cur_losses += 1
                cur_wins = 0
            else:
                cur_wins = 0
                cur_losses = 0
            max_wins = max(max_wins, cur_wins)
            max_losses = max(max_losses, cur_losses)

        return max_wins, max_losses
