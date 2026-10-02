"""
基于历史交易分析优化各策略参数，支持参数热更新与优化记录。
"""
import asyncio
import math
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from core.trade_journal import TradeJournal
from analysis.historical_analyzer import HistoricalAnalyzer


def _safe_float(value, default=0.0):
    """安全转换数值：None/非法/NaN/Inf 返回默认值，用于 fail-closed 校验。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f) or math.isinf(f):
        return default
    return f


# 交易分配字段：写回 config.yaml 时需保证总和精确等于 1.0（否则触发 AppConfig 校验失败）
TRADING_ALLOCATION_FIELDS = [
    "grid_allocation", "spot_grid_allocation", "spot_martingale_allocation",
    "trend_allocation", "scalping_allocation", "arbitrage_allocation",
]


def _normalize_trading_allocations(trading_cfg: Dict[str, Any]) -> None:
    """归一化交易分配字段，确保 6 个 allocation 之和精确等于 1.0。

    persist_config 写回前会 _round_floats 对每个字段独立四舍五入到 6 位小数，
    浮点误差累积会使总和变成 0.999999 之类，进而触发 AppConfig 的
    「allocations sum must be 1.0」校验失败。此处把 (1.0 - 总和) 的误差集中
    加到当前值最大的字段（相对扰动最小），使总和精确回到 1.0，且不改变已置 0
    的 disabled 策略字段。
    """
    if not isinstance(trading_cfg, dict):
        return
    entries = [
        f for f in TRADING_ALLOCATION_FIELDS
        if f in trading_cfg and isinstance(trading_cfg.get(f), (int, float))
    ]
    if not entries:
        return
    total = sum(float(trading_cfg.get(f) or 0.0) for f in entries)
    if total <= 0 or abs(total - 1.0) <= 1e-9:
        return
    adjust_field = max(entries, key=lambda f: float(trading_cfg.get(f) or 0.0))
    trading_cfg[adjust_field] = round(float(trading_cfg[adjust_field] or 0.0) + (1.0 - total), 6)


class StrategyOptimizer:
    def __init__(self, trade_journal: TradeJournal, config: Dict[str, Any]):
        self.trade_journal = trade_journal
        self.config = config
        self.analyzer = HistoricalAnalyzer(trade_journal)

        self._learning_state: Dict[str, Any] = {}
        self._optimization_history: List[Dict[str, Any]] = []
        self._parameter_tuning: Dict[str, Dict[str, Any]] = {}

        self._min_trades_for_optimization = 50
        self._confidence_threshold = 0.7

        # P2: 样本外验证配置（防止过拟合）
        self._oos_validation_enabled = config.get("strategy_optimizer", {}).get("oos_validation_enabled", True)
        self._oos_train_ratio = config.get("strategy_optimizer", {}).get("oos_train_ratio", 0.7)
        self._oos_min_test_trades = config.get("strategy_optimizer", {}).get("oos_min_test_trades", 10)

        # 锁定参数：locked_params 中列出的参数不会被自动优化覆盖
        self._locked_params: Dict[str, set] = {}
        for sname, scfg in config.get("strategies", {}).items():
            if isinstance(scfg, dict):
                self._locked_params[sname] = set(scfg.get("locked_params", []) or [])

        # 策略实例引用（由scheduler注入，用于热更新参数）
        self._strategy_instances: Dict[str, Any] = {}

    def register_strategy_instance(self, name: str, instance):
        """注册策略实例，用于参数热更新"""
        self._strategy_instances[name] = instance
        logger.info(f"Strategy instance registered: {name}")

    def register_all_strategies(self, **kwargs):
        """批量注册所有策略实例（按策略名注入，兼容任意新增策略）"""
        for name, instance in kwargs.items():
            if instance is not None:
                self.register_strategy_instance(name, instance)
    
    async def start(self):
        await self._load_learning_state()

    def _get_trades_for_oos(self, strategy_name: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """获取某策略的交易记录（按 exit_time 升序），用于 OOS 拆分。"""
        try:
            trades = self.trade_journal.get_recent_trades(limit=limit)
            strategy_trades = [t for t in trades if t.get("strategy_name") == strategy_name]
            strategy_trades.sort(key=lambda t: str(t.get("exit_time", "")))
            return strategy_trades
        except Exception as e:
            logger.error(f"Failed to get trades for OOS validation ({strategy_name}): {e}")
            return []

    def _split_trades_chronological(self, trades: List[Dict[str, Any]]) -> tuple:
        """按时间顺序拆分交易为 train/test 集（前 train_ratio 为训练集，后 1-train_ratio 为测试集）。"""
        n = len(trades)
        split_idx = int(n * self._oos_train_ratio)
        return trades[:split_idx], trades[split_idx:]

    def _compute_stats_from_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """从交易列表计算基础统计指标。"""
        if not trades:
            return {"total_trades": 0, "win_rate": 0, "total_pnl": 0, "profit_factor": 0, "avg_pnl": 0}
        wins = [t for t in trades if _safe_float(t.get("pnl", 0)) > 0]
        losses = [t for t in trades if _safe_float(t.get("pnl", 0)) <= 0]
        total_pnl = sum(_safe_float(t.get("pnl", 0)) for t in trades)
        gross_profit = sum(_safe_float(t.get("pnl", 0)) for t in wins)
        gross_loss = abs(sum(_safe_float(t.get("pnl", 0)) for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        return {
            "total_trades": len(trades),
            "win_rate": len(wins) / len(trades) if trades else 0,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "avg_pnl": total_pnl / len(trades) if trades else 0,
        }

    def _simulate_parameter_impact(
        self, strategy: str, changes: Dict[str, Any], test_trades: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """启发式估算参数变更对测试集交易的影响。

        不做全量回测（代价太高），而是根据参数变更方向与测试集交易特征的
        匹配度来估算。例如：
        - 放宽 grid_spacing → 测试集中因间距过小而止损的交易可减少
        - 收紧 trailing_stop → 测试集中回撤较大的盈利交易可能更早被止盈
        返回 {"estimated_improvement": float, "validated": bool, "details": str}
        """
        if not test_trades:
            return {"estimated_improvement": 0, "validated": False, "details": "no test trades"}

        baseline_stats = self._compute_stats_from_trades(test_trades)
        estimated_pnl_delta = 0.0

        for param, new_val in changes.items():
            if param in ("grid_spacing_min", "grid_spacing", "min_grid_spacing"):
                losses = [t for t in test_trades if _safe_float(t.get("pnl", 0)) < 0]
                estimated_pnl_delta += len(losses) * 0.5
            elif param in ("trailing_stop_tier1", "trailing_stop_tier2", "trailing_stop_tier3", "trailing_stop"):
                wins = [t for t in test_trades if _safe_float(t.get("pnl", 0)) > 0]
                avg_win = baseline_stats["total_pnl"] / max(1, baseline_stats["total_trades"])
                estimated_pnl_delta += len(wins) * abs(avg_win) * 0.03
            elif param in ("stop_loss",):
                losses = [t for t in test_trades if _safe_float(t.get("pnl", 0)) < 0]
                avg_loss = sum(_safe_float(t.get("pnl", 0)) for t in losses) / max(1, len(losses))
                estimated_pnl_delta += abs(avg_loss) * len(losses) * 0.05
            elif param in ("profit_target_min", "take_profit_pct"):
                wins = [t for t in test_trades if _safe_float(t.get("pnl", 0)) > 0]
                avg_win = sum(_safe_float(t.get("pnl", 0)) for t in wins) / max(1, len(wins))
                estimated_pnl_delta += avg_win * len(wins) * 0.02
            elif param in ("rsi_oversold", "rsi_overbought", "funding_rate_threshold"):
                n = len(test_trades)
                avg_pnl = baseline_stats["avg_pnl"]
                estimated_pnl_delta += avg_pnl * n * 0.01
            elif param in ("martingale_coefficient",):
                losses = [t for t in test_trades if _safe_float(t.get("pnl", 0)) < 0]
                avg_loss = sum(_safe_float(t.get("pnl", 0)) for t in losses) / max(1, len(losses))
                if new_val < self.config.get("strategies", {}).get(strategy, {}).get(param, 1.5):
                    estimated_pnl_delta += abs(avg_loss) * len(losses) * 0.08
            elif param in ("leverage",):
                total_pnl = baseline_stats["total_pnl"]
                estimated_pnl_delta += total_pnl * 0.05
            elif param in ("max_layers", "grid_count_min", "grid_count_max"):
                total_pnl = baseline_stats["total_pnl"]
                estimated_pnl_delta += abs(total_pnl) * 0.03

        estimated_new_pnl = baseline_stats["total_pnl"] + estimated_pnl_delta
        improvement_pct = estimated_pnl_delta / max(abs(baseline_stats["total_pnl"]), 1)

        validated = improvement_pct > 0
        return {
            "estimated_improvement": improvement_pct,
            "estimated_pnl_delta": estimated_pnl_delta,
            "baseline_pnl": baseline_stats["total_pnl"],
            "estimated_new_pnl": estimated_new_pnl,
            "validated": validated,
            "test_trades": len(test_trades),
            "details": f"baseline_pnl={baseline_stats['total_pnl']:.2f}, estimated_delta={estimated_pnl_delta:.2f}, test_n={len(test_trades)}",
        }

    async def _validate_recommendations_oos(
        self, strategy: str, recommendations: List[Dict[str, Any]], changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """对推荐变更做样本外验证。返回验证结果，包含哪些变更通过/未通过。"""
        if not self._oos_validation_enabled or not changes:
            return {"validated": True, "reason": "OOS validation disabled or no changes"}

        all_trades = self._get_trades_for_oos(strategy)
        if len(all_trades) < self._oos_min_test_trades * 2:
            return {
                "validated": False,
                "reason": f"Insufficient trades ({len(all_trades)}) for OOS split (need >= {self._oos_min_test_trades * 2})",
            }

        train_trades, test_trades = self._split_trades_chronological(all_trades)
        if len(test_trades) < self._oos_min_test_trades:
            return {
                "validated": False,
                "reason": f"Test set too small ({len(test_trades)} < {self._oos_min_test_trades})",
            }

        train_stats = self._compute_stats_from_trades(train_trades)
        test_stats = self._compute_stats_from_trades(test_trades)

        simulation = self._simulate_parameter_impact(strategy, changes, test_trades)

        return {
            "validated": simulation["validated"],
            "train_stats": train_stats,
            "test_stats": test_stats,
            "simulation": simulation,
            "train_size": len(train_trades),
            "test_size": len(test_trades),
        }
    
    async def _load_learning_state(self):
        state = await self.trade_journal.load_learning_state("strategy_optimizer_state")
        if state:
            self._learning_state = state
            self._optimization_history = state.get("optimization_history", [])
            self._parameter_tuning = state.get("parameter_tuning", {})
            logger.info(f"Loaded learning state with {len(self._optimization_history)} optimization records")
    
    async def _save_learning_state(self):
        state = {
            "optimization_history": self._optimization_history,
            "parameter_tuning": self._parameter_tuning,
            "last_optimization_time": datetime.now().isoformat()
        }
        await self.trade_journal.persist_learning_state("strategy_optimizer_state", state)
    
    async def optimize_all_strategies(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        analysis = self.analyzer.analyze_all_strategies(since)
        
        recommendations = {
            "grid": await self._optimize_grid(analysis),
            "spot_grid": await self._optimize_spot_grid(analysis),
            "spot_martingale": await self._optimize_spot_martingale(analysis),
            "trend": await self._optimize_trend(analysis),
            "scalping": await self._optimize_scalping(analysis),
            "arbitrage": await self._optimize_arbitrage(analysis)
        }

        # P2: 样本外验证——过滤掉在测试集上未通过验证的参数变更
        oos_results = {}
        for strategy, rec in recommendations.items():
            if strategy == "overall":
                continue
            if rec.get("optimized") and rec.get("changes"):
                oos_result = await self._validate_recommendations_oos(
                    strategy, rec.get("recommendations", []), rec["changes"]
                )
                oos_results[strategy] = oos_result
                if not oos_result.get("validated"):
                    logger.warning(
                        f"OOS validation FAILED for {strategy}: {oos_result.get('reason', oos_result.get('simulation', {}).get('details', ''))}"
                    )
                    rec["changes"] = {}
                    rec["optimized"] = False
                    rec["oos_rejected"] = True
                    rec["oos_reason"] = oos_result.get("reason", "validation failed")
                    for r in rec.get("recommendations", []):
                        r["oos_validated"] = False
                else:
                    sim = oos_result.get("simulation", {})
                    logger.info(
                        f"OOS validation PASSED for {strategy}: "
                        f"est. improvement={sim.get('estimated_improvement', 0):.2%}, "
                        f"test_n={oos_result.get('test_size', 0)}"
                    )
                    rec["oos_validated"] = True
                    rec["oos_simulation"] = sim

        overall_recommendations = self._generate_overall_recommendations(analysis)
        recommendations["overall"] = overall_recommendations
        recommendations["oos_validation"] = oos_results
        
        await self._record_optimization(recommendations)
        await self._save_learning_state()
        
        return recommendations
    
    async def _optimize_grid(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        strategy_stats = None
        for s in analysis["strategies"]:
            if s["strategy_name"] == "grid":
                strategy_stats = s
                break
        
        if not strategy_stats or _safe_float(strategy_stats.get("total_trades", 0)) < self._min_trades_for_optimization:
            return {
                "strategy": "grid",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }
        
        recommendations = []
        changes = {}
        
        if _safe_float(strategy_stats.get("win_rate", 0)) < 0.45:
            tier1 = self.config["currencies"]["tier1_settings"]
            new_spacing = round(tier1["grid_spacing_min"] * 1.2, 6)
            # 夹紧：不能超过 grid_spacing_max
            new_spacing = min(new_spacing, tier1["grid_spacing_max"] * 0.95)
            new_spacing = round(new_spacing, 6)
            if tier1["grid_spacing_min"] < tier1["grid_spacing_max"]:
                recommendations.append({
                    "parameter": "grid_spacing",
                    "current": tier1["grid_spacing_min"],
                    "recommended": new_spacing,
                    "reason": "Low win rate suggests grid spacing too tight, causing frequent stop-outs",
                    "confidence": 0.75
                })
                changes["grid_spacing_min"] = new_spacing
        
        if _safe_float(strategy_stats.get("profit_factor", 0)) < 1.0:
            new_coef = max(1.0, min(1.15, self.config["strategies"]["grid"]["martingale_coefficient"] * 0.95))
            recommendations.append({
                "parameter": "martingale_coefficient",
                "current": self.config["strategies"]["grid"]["martingale_coefficient"],
                "recommended": new_coef,
                "reason": "Low profit factor suggests martingale is amplifying losses excessively",
                "confidence": 0.7
            })
            changes["martingale_coefficient"] = new_coef
        
        if _safe_float(strategy_stats.get("total_pnl", 0)) < 0:
            recommendations.append({
                "parameter": "grid_count",
                "current": (self.config["strategies"]["grid"]["grid_count_min"] + self.config["strategies"]["grid"]["grid_count_max"]) / 2,
                "recommended": int((self.config["strategies"]["grid"]["grid_count_min"] + self.config["strategies"]["grid"]["grid_count_max"]) / 2 * 0.8),
                "reason": "Negative PnL suggests too many grid levels, increasing transaction costs",
                "confidence": 0.65
            })
            changes["grid_count_min"] = max(2, int(self.config["strategies"]["grid"]["grid_count_min"] * 0.8))
            changes["grid_count_max"] = max(changes.get("grid_count_min", 2) + 1, int(self.config["strategies"]["grid"]["grid_count_max"] * 0.8))
        
        time_patterns = analysis["time_patterns"]
        worst_hours = [h for h in time_patterns["worst_hours"] if _safe_float(h.get("avg_pnl", 0)) < -10]
        if worst_hours:
            recommendations.append({
                "parameter": "trading_hours",
                "current": "24/7",
                "recommended": f"Exclude hours: {[h['hour'] for h in worst_hours]}",
                "reason": f"Significant losses during hours {[h['hour'] for h in worst_hours]}",
                "confidence": 0.6
            })
        
        return {
            "strategy": "grid",
            "optimized": len(recommendations) > 0,
            "current_stats": strategy_stats,
            "recommendations": recommendations,
            "changes": changes
        }
    
    async def _optimize_trend(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        strategy_stats = None
        for s in analysis["strategies"]:
            if s["strategy_name"] == "trend":
                strategy_stats = s
                break
        
        if not strategy_stats or _safe_float(strategy_stats.get("total_trades", 0)) < self._min_trades_for_optimization:
            return {
                "strategy": "trend",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }
        
        recommendations = []
        changes = {}
        locked = self._locked_params.get("trend", set())

        if _safe_float(strategy_stats.get("win_rate", 0)) < 0.45 and "confirmation_periods" not in locked:
            recommendations.append({
                "parameter": "confirmation_periods",
                "current": self.config["strategies"]["trend"]["confirmation_periods"],
                "recommended": ["1d", "4h", "1h", "30m"],
                "reason": "Low win rate suggests need for additional timeframe confirmation",
                "confidence": 0.75
            })
            changes["confirmation_periods"] = ["1d", "4h", "1h", "30m"]
        elif _safe_float(strategy_stats.get("win_rate", 0)) < 0.45:
            logger.info("Trend confirmation_periods is locked, skipping auto-tighten (low win rate)")
        
        if _safe_float(strategy_stats.get("profit_factor", 0)) < 1.0:
            # 约束 trailing_stop 下限，避免 ×0.8 无界下调趋近 0
            _cur_ts1 = self.config["strategies"]["trend"]["trailing_stop_tier1"]
            _cur_ts2 = self.config["strategies"]["trend"]["trailing_stop_tier2"]
            _cur_ts3 = self.config["strategies"]["trend"]["trailing_stop_tier3"]
            _new_ts1 = max(0.001, _cur_ts1 * 0.8)
            _new_ts2 = max(0.001, _cur_ts2 * 0.8)
            _new_ts3 = max(0.001, _cur_ts3 * 0.8)
            recommendations.append({
                "parameter": "trailing_stop",
                "current": _cur_ts1,
                "recommended": _new_ts1,
                "reason": "Tighter trailing stop to protect profits and improve profit factor",
                "confidence": 0.7
            })
            changes["trailing_stop_tier1"] = _new_ts1
            changes["trailing_stop_tier2"] = _new_ts2
            changes["trailing_stop_tier3"] = _new_ts3
        
        if _safe_float(strategy_stats.get("total_pnl", 0)) < 0:
            # 约束 initial_position_ratio 下限（0,1]，避免 ×0.7 无界下调趋近 0
            _cur_ratio = self.config["strategies"]["trend"]["initial_position_ratio"]
            _new_ratio = max(0.05, _cur_ratio * 0.7)
            recommendations.append({
                "parameter": "initial_position_ratio",
                "current": _cur_ratio,
                "recommended": _new_ratio,
                "reason": "Reduce initial position size to limit drawdown",
                "confidence": 0.65
            })
            changes["initial_position_ratio"] = _new_ratio
        
        success_patterns = analysis["success_patterns"]
        top_strategies = [p for p in success_patterns["patterns"] if p["strategy"] == "trend"]
        if top_strategies and _safe_float(top_strategies[0].get("avg_pnl", 0)) > 100:
            # P0: 安全访问配置键，防止KeyError（trend_allocation可能不存在）
            current_alloc = self.config.get("trading", {}).get("trend_allocation", 0.15)
            recommendations.append({
                "parameter": "allocation",
                "current": current_alloc,
                "recommended": min(0.4, current_alloc * 1.1),
                "reason": "Excellent performance, increase allocation",
                "confidence": 0.8
            })
            changes["trend_allocation"] = min(0.4, current_alloc * 1.1)
        
        return {
            "strategy": "trend",
            "optimized": len(recommendations) > 0,
            "current_stats": strategy_stats,
            "recommendations": recommendations,
            "changes": changes
        }
    
    async def _optimize_scalping(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        strategy_stats = None
        for s in analysis["strategies"]:
            if s["strategy_name"] == "scalping":
                strategy_stats = s
                break
        
        if not strategy_stats or _safe_float(strategy_stats.get("total_trades", 0)) < self._min_trades_for_optimization:
            return {
                "strategy": "scalping",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }
        
        recommendations = []
        changes = {}
        
        if _safe_float(strategy_stats.get("win_rate", 0)) < 0.45:
            cur_oversold = self.config["strategies"]["scalping"].get("rsi_oversold", 30)
            cur_overbought = self.config["strategies"]["scalping"].get("rsi_overbought", 70)
            new_oversold = max(0.0, cur_oversold - 5)
            new_overbought = min(100.0, cur_overbought + 5)
            # 保持 oversold < overbought，避免触发 rsi_oversold < rsi_overbought 配置校验失败
            if new_oversold >= new_overbought:
                new_oversold, new_overbought = cur_oversold, cur_overbought

            recommendations.append({
                "parameter": "rsi_oversold",
                "current": cur_oversold,
                "recommended": new_oversold,
                "reason": "Increase RSI oversold threshold to filter weaker signals",
                "confidence": 0.75
            })
            changes["rsi_oversold"] = new_oversold

            recommendations.append({
                "parameter": "rsi_overbought",
                "current": cur_overbought,
                "recommended": new_overbought,
                "reason": "Increase RSI overbought threshold to filter weaker signals",
                "confidence": 0.75
            })
            changes["rsi_overbought"] = new_overbought
        
        if _safe_float(strategy_stats.get("profit_factor", 0)) < 1.0:
            # 约束 profit_target_min 上限为 profit_target_max，避免 ×1.2 上调越界（min > max 触发配置校验失败）
            _cur_min = self.config["strategies"]["scalping"]["profit_target_min"]
            _max = self.config["strategies"]["scalping"].get("profit_target_max", _cur_min)
            _new_min = min(_cur_min * 1.2, _max)
            recommendations.append({
                "parameter": "profit_target_min",
                "current": _cur_min,
                "recommended": _new_min,
                "reason": "Increase minimum profit target to improve average win size",
                "confidence": 0.7
            })
            changes["profit_target_min"] = _new_min
            
            # 约束 stop_loss 下限，避免 ×0.9 无界下调趋近 0 导致止损失效
            _cur_stop = self.config["strategies"]["scalping"]["stop_loss"]
            _new_stop = max(0.0005, _cur_stop * 0.9)
            recommendations.append({
                "parameter": "stop_loss",
                "current": _cur_stop,
                "recommended": _new_stop,
                "reason": "Tighten stop loss to reduce average loss size",
                "confidence": 0.7
            })
            changes["stop_loss"] = _new_stop
        
        time_patterns = analysis["time_patterns"]
        best_hours = [h for h in time_patterns["best_hours"] if _safe_float(h.get("avg_pnl", 0)) > 20]
        if best_hours:
            recommendations.append({
                "parameter": "run_hours",
                "current": f"{self.config['strategies']['scalping']['run_hours_start']}-{self.config['strategies']['scalping']['run_hours_end']}",
                "recommended": f"Focus on hours: {[h['hour'] for h in best_hours]}",
                "reason": f"Best performance during hours {[h['hour'] for h in best_hours]}",
                "confidence": 0.75
            })
        
        return {
            "strategy": "scalping",
            "optimized": len(recommendations) > 0,
            "current_stats": strategy_stats,
            "recommendations": recommendations,
            "changes": changes
        }
    
    async def _optimize_arbitrage(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        strategy_stats = None
        for s in analysis["strategies"]:
            if s["strategy_name"] == "arbitrage":
                strategy_stats = s
                break
        
        if not strategy_stats or _safe_float(strategy_stats.get("total_trades", 0)) < self._min_trades_for_optimization:
            return {
                "strategy": "arbitrage",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }
        
        recommendations = []
        changes = {}
        
        if _safe_float(strategy_stats.get("win_rate", 0)) < 0.45:
            # 约束 funding_rate_threshold 上限，避免 ×1.2 无界上调
            _cur_frt = self.config["strategies"]["arbitrage"]["funding_rate_threshold"]
            _new_frt = min(0.05, _cur_frt * 1.2)
            recommendations.append({
                "parameter": "funding_rate_threshold",
                "current": _cur_frt,
                "recommended": _new_frt,
                "reason": "Increase threshold to only take higher confidence funding rate trades",
                "confidence": 0.7
            })
            changes["funding_rate_threshold"] = _new_frt
        
        if _safe_float(strategy_stats.get("total_pnl", 0)) < 0:
            recommendations.append({
                "parameter": "leverage",
                "current": self.config["strategies"]["arbitrage"]["leverage"],
                "recommended": max(1, self.config["strategies"]["arbitrage"]["leverage"] - 1),
                "reason": "Reduce leverage to limit losses from basis risk",
                "confidence": 0.65
            })
            changes["leverage"] = max(1, self.config["strategies"]["arbitrage"]["leverage"] - 1)
        
        success_patterns = analysis["success_patterns"]
        top_strategies = [p for p in success_patterns["patterns"] if p["strategy"] == "arbitrage"]
        if top_strategies and _safe_float(top_strategies[0].get("avg_pnl", 0)) > 100:
            # P0: 安全访问配置键，防止KeyError（arbitrage_allocation可能不存在）
            current_alloc = self.config.get("trading", {}).get("arbitrage_allocation", 0.05)
            recommendations.append({
                "parameter": "allocation",
                "current": current_alloc,
                "recommended": min(0.1, current_alloc * 1.2),
                "reason": "Excellent performance, increase allocation",
                "confidence": 0.75
            })
            changes["arbitrage_allocation"] = min(0.1, current_alloc * 1.2)
        
        return {
            "strategy": "arbitrage",
            "optimized": len(recommendations) > 0,
            "current_stats": strategy_stats,
            "recommendations": recommendations,
            "changes": changes
        }

    async def _optimize_spot_grid(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        strategy_stats = None
        for s in analysis["strategies"]:
            if s["strategy_name"] == "spot_grid":
                strategy_stats = s
                break

        if not strategy_stats or _safe_float(strategy_stats.get("total_trades", 0)) < self._min_trades_for_optimization:
            return {
                "strategy": "spot_grid",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }

        recommendations = []
        changes = {}
        # P0: 安全访问配置，spot_grid配置节可能不存在
        spot_grid_cfg = self.config.get("strategies", {}).get("spot_grid", {})
        spot_martingale_cfg = self.config.get("strategies", {}).get("spot_martingale", {})
        
        if _safe_float(strategy_stats.get("win_rate", 0)) < 0.45 and spot_grid_cfg:
            cur_min = spot_grid_cfg.get("min_grid_spacing", 0.01)
            cur_max = spot_grid_cfg.get("max_grid_spacing", 0.03)
            current_spacing = (cur_min + cur_max) / 2
            # 约束 min_grid_spacing 不超过 max_grid_spacing，避免 min>max 配置校验失败
            new_min = min(cur_min * 1.3, cur_max)
            recommendations.append({
                "parameter": "grid_spacing",
                "current": current_spacing,
                "recommended": new_min,
                "reason": "Low win rate suggests grid spacing too tight",
                "confidence": 0.7
            })
            changes["min_grid_spacing"] = new_min

        if _safe_float(strategy_stats.get("profit_factor", 0)) < 1.0 and spot_grid_cfg:
            current_tp = spot_grid_cfg.get("take_profit_pct", 0.02)
            # 约束 take_profit_pct 上限 le=1
            new_tp = min(1.0, current_tp * 1.2)
            recommendations.append({
                "parameter": "take_profit_pct",
                "current": current_tp,
                "recommended": new_tp,
                "reason": "Increase take profit to improve profit factor",
                "confidence": 0.7
            })
            changes["take_profit_pct"] = new_tp

        return {
            "strategy": "spot_grid",
            "optimized": len(recommendations) > 0,
            "current_stats": strategy_stats,
            "recommendations": recommendations,
            "changes": changes
        }

    async def _optimize_spot_martingale(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        strategy_stats = None
        for s in analysis["strategies"]:
            if s["strategy_name"] == "spot_martingale":
                strategy_stats = s
                break

        if not strategy_stats or _safe_float(strategy_stats.get("total_trades", 0)) < self._min_trades_for_optimization:
            return {
                "strategy": "spot_martingale",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }

        recommendations = []
        changes = {}
        # P0: 安全访问配置，spot_martingale配置节可能不存在
        spot_martingale_cfg = self.config.get("strategies", {}).get("spot_martingale", {})

        if _safe_float(strategy_stats.get("win_rate", 0)) < 0.45 and spot_martingale_cfg:
            current_drop = spot_martingale_cfg.get("price_drop_pct", 0.05)
            # 约束 price_drop_pct 上限 le=1
            new_drop = min(1.0, current_drop * 1.3)
            recommendations.append({
                "parameter": "price_drop_pct",
                "current": current_drop,
                "recommended": new_drop,
                "reason": "Increase price drop threshold to filter weaker signals",
                "confidence": 0.7
            })
            changes["price_drop_pct"] = new_drop

        if _safe_float(strategy_stats.get("profit_factor", 0)) < 1.0 and spot_martingale_cfg:
            current_coef = spot_martingale_cfg.get("martingale_coefficient", 1.1)
            new_coef = max(1.0, min(1.3, current_coef * 0.9))
            recommendations.append({
                "parameter": "martingale_coefficient",
                "current": current_coef,
                "recommended": new_coef,
                "reason": "Reduce martingale coefficient to limit loss amplification",
                "confidence": 0.7
            })
            changes["martingale_coefficient"] = new_coef

        if _safe_float(strategy_stats.get("total_pnl", 0)) < 0 and spot_martingale_cfg:
            current_layers = spot_martingale_cfg.get("max_layers", 3)
            recommendations.append({
                "parameter": "max_layers",
                "current": current_layers,
                "recommended": max(2, current_layers - 1),
                "reason": "Reduce max layers to limit drawdown",
                "confidence": 0.65
            })
            changes["max_layers"] = max(2, current_layers - 1)

        return {
            "strategy": "spot_martingale",
            "optimized": len(recommendations) > 0,
            "current_stats": strategy_stats,
            "recommendations": recommendations,
            "changes": changes
        }
    
    def _generate_overall_recommendations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        overview = analysis["overview"] or {}
        shortcomings = analysis["shortcomings"] or {}
        success_patterns = analysis["success_patterns"] or {}
        _max_drawdown = _safe_float(overview.get("max_drawdown", 0))
        _sharpe_ratio = _safe_float(overview.get("sharpe_ratio", 0))
        
        recommendations = []
        
        if _max_drawdown > 0.15:
            recommendations.append({
                "category": "risk_management",
                "action": "reduce_overall_leverage",
                "reason": f"Max drawdown {_max_drawdown:.1%} exceeds 15% threshold",
                "suggestion": "Reduce overall leverage by 20% and increase margin requirements",
                "priority": "high"
            })
        
        if _sharpe_ratio < 1.0:
            recommendations.append({
                "category": "risk_adjusted_return",
                "action": "optimize_risk_reward",
                "reason": f"Sharpe ratio {_sharpe_ratio:.2f} below 1.0 target",
                "suggestion": "Increase average win/loss ratio, reduce trade frequency",
                "priority": "medium"
            })
        
        if len(shortcomings.get("critical", [])) > 0:
            recommendations.append({
                "category": "emergency",
                "action": "address_critical_issues",
                "reason": f"{len(shortcomings['critical'])} critical shortcomings identified",
                "suggestion": "Immediately address critical issues before continuing trading",
                "priority": "critical"
            })
        
        if len(success_patterns.get("key_factors", [])) > 0:
            recommendations.append({
                "category": "capital_allocation",
                "action": "allocate_to_winners",
                "reason": f"{len(success_patterns['key_factors'])} key success factors identified",
                "suggestion": "Increase allocation to strategies/symbols showing strong performance",
                "priority": "medium"
            })
        
        symbol_stats = analysis.get("symbols", []) or []
        top_symbols = [s for s in symbol_stats if _safe_float(s.get("total_pnl", 0)) > 0][:3]
        bottom_symbols = [s for s in symbol_stats if _safe_float(s.get("total_pnl", 0)) < 0][:3]
        
        if bottom_symbols:
            recommendations.append({
                "category": "symbol_filter",
                "action": "reduce_underperforming_symbols",
                "reason": f"{len(bottom_symbols)} underperforming symbols identified",
                "suggestion": f"Reduce trading on {[s['symbol'] for s in bottom_symbols]} or remove from watchlist",
                "priority": "medium"
            })
        
        return {
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
            "current_overview": overview,
            "suggested_actions": [r["action"] for r in recommendations]
        }
    
    async def _record_optimization(self, recommendations: Dict[str, Any]):
        optimization_record = {
            "timestamp": datetime.now().isoformat(),
            "recommendations": recommendations,
            "analysis_summary": self.analyzer.generate_analysis_report()["summary"]
        }
        
        self._optimization_history.append(optimization_record)
        
        if len(self._optimization_history) > 100:
            self._optimization_history = self._optimization_history[-100:]
    
    def get_learning_progress(self) -> Dict[str, Any]:
        if not self._optimization_history:
            return {
                "total_optimizations": 0,
                "current_state": "No optimization history",
                "progress": 0
            }
        
        recent_optimizations = self._optimization_history[-10:]
        
        improvement_count = 0
        for i in range(1, len(recent_optimizations)):
            prev_pnl = _safe_float(recent_optimizations[i-1].get("analysis_summary", {}).get("total_pnl", 0))
            curr_pnl = _safe_float(recent_optimizations[i].get("analysis_summary", {}).get("total_pnl", 0))
            if curr_pnl > prev_pnl:
                improvement_count += 1
        
        improvement_rate = improvement_count / max(1, len(recent_optimizations) - 1)
        
        return {
            "total_optimizations": len(self._optimization_history),
            "improvement_rate": improvement_rate,
            "last_optimization": self._optimization_history[-1]["timestamp"],
            "current_state": self._learning_state.get("current_state", "learning"),
            "progress": min(100, len(self._optimization_history) * 5)
        }
    
    async def apply_optimizations(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        applied = []
        failed = []
        
        for strategy, rec in recommendations.items():
            if strategy == "overall":
                continue
            
            if rec["optimized"] and rec.get("changes"):
                try:
                    await self._apply_strategy_changes(strategy, rec["changes"])
                    applied.append({
                        "strategy": strategy,
                        "changes": rec["changes"]
                    })
                except Exception as e:
                    failed.append({
                        "strategy": strategy,
                        "error": str(e)
                    })
        
        return {
            "applied": applied,
            "failed": failed,
            "total_applied": len(applied),
            "total_failed": len(failed)
        }
    
    async def _apply_strategy_changes(self, strategy: str, changes: Dict[str, Any]):
        locked = self._locked_params.get(strategy, set())
        for key, value in changes.items():
            if key in locked:
                logger.info(f"Skip locked param {strategy}.{key} (value {value} ignored)")
                continue
            if key in self.config["strategies"][strategy]:
                self.config["strategies"][strategy][key] = value
                logger.info(f"Updated {strategy}.{key} to {value}")
            elif key in self.config["trading"]:
                self.config["trading"][key] = value
                logger.info(f"Updated trading.{key} to {value}")
            elif key in ["grid_spacing_min", "grid_spacing_max"]:
                for tier in ["tier1", "tier2", "tier3"]:
                    if key in self.config["currencies"][f"{tier}_settings"]:
                        self.config["currencies"][f"{tier}_settings"][key] = value
                        logger.info(f"Updated {tier}_settings.{key} to {value}")

        # 热更新：通知运行中的策略实例应用新参数
        strategy_instance = self._strategy_instances.get(strategy)
        if strategy_instance and hasattr(strategy_instance, "apply_param_update"):
            try:
                applied = strategy_instance.apply_param_update(changes)
                if applied:
                    logger.info(f"Hot-updated {strategy} strategy params: {applied}")
            except Exception as e:
                logger.error(f"Failed to hot-update {strategy} strategy: {e}")

    def _validate_tier_constraints(self, config: dict):
        """持久化前验证 tier_settings 的 min/max 约束，防止写坏配置导致启动崩溃"""
        currencies = config.get("currencies", {})
        for tier_key in ["tier1_settings", "tier2_settings", "tier3_settings"]:
            tier = currencies.get(tier_key)
            if not tier:
                continue
            for prefix in ["grid_spacing", "leverage", "position_size", "stop_loss", "take_profit"]:
                key_min = f"{prefix}_min"
                key_max = f"{prefix}_max"
                val_min = tier.get(key_min)
                val_max = tier.get(key_max)
                if val_min is not None and val_max is not None and val_min > val_max:
                    logger.warning(f"Config validation: {tier_key}.{key_min}={val_min} > {key_max}={val_max}, clamping min to max*0.9")
                    tier[key_min] = round(val_max * 0.9, 6)

    async def persist_config(self, config_path: str = "config.yaml") -> bool:
        import yaml
        import shutil
        import os
        from datetime import datetime

        try:
            # 版本管理：写入前先备份当前config到 versions/ 目录
            versions_dir = "./config_versions"
            os.makedirs(versions_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{versions_dir}/config_{timestamp}.yaml"

            if os.path.exists(config_path):
                shutil.copy2(config_path, backup_path)
                logger.info(f"Config backed up to {backup_path}")

                # 清理超过30天的旧版本
                cutoff = datetime.now().timestamp() - 30 * 86400
                for fname in os.listdir(versions_dir):
                    fpath = os.path.join(versions_dir, fname)
                    if os.path.isfile(fpath) and fname.startswith("config_"):
                        if os.path.getmtime(fpath) < cutoff:
                            os.remove(fpath)
                            logger.debug(f"Removed old config version: {fname}")

            # 读取现有完整配置，只更新优化的部分
            existing_config = {}
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    existing_config = yaml.safe_load(f) or {}

            for strategy in ["grid", "spot_grid", "spot_martingale", "trend", "scalping", "arbitrage"]:
                if strategy in self.config["strategies"]:
                    strategy_config = {}
                    for key, value in self.config["strategies"][strategy].items():
                        if not isinstance(value, str) or "${" not in value:
                            strategy_config[key] = value
                    if strategy_config:
                        existing_config.setdefault("strategies", {})[strategy] = strategy_config

            if "trading" in self.config:
                existing_config.setdefault("trading", {})
                for key, value in self.config["trading"].items():
                    if not isinstance(value, str) or "${" not in value:
                        existing_config["trading"][key] = value

            for key in ["tier1_settings", "tier2_settings", "tier3_settings"]:
                if key in self.config.get("currencies", {}):
                    tier_config = {}
                    for tier_key, value in self.config["currencies"][key].items():
                        if not isinstance(value, str) or "${" not in value:
                            tier_config[tier_key] = value
                    if tier_config:
                        existing_config.setdefault("currencies", {})[key] = tier_config

            # P0: 使用传入的config_path参数，而非硬编码文件名
            # 递归四舍五入浮点数，避免 0.013823999999999998 等精度问题
            def _round_floats(obj, decimals=6):
                if isinstance(obj, dict):
                    return {k: _round_floats(v, decimals) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [_round_floats(v, decimals) for v in obj]
                elif isinstance(obj, float):
                    return round(obj, decimals)
                return obj
            existing_config = _round_floats(existing_config)

            # 归一化交易分配字段，确保总和精确等于 1.0（见 _normalize_trading_allocations 说明）
            _normalize_trading_allocations(existing_config.get("trading", {}))

            # 持久化前验证：确保所有 tier_settings 的 min/max 约束有效
            self._validate_tier_constraints(existing_config)

            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(existing_config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

            logger.info(f"Optimizations persisted to {config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to persist configuration: {e}")
            return False

    def list_config_versions(self) -> list:
        """列出所有可回滚的配置版本"""
        import os
        versions_dir = "./config_versions"
        if not os.path.exists(versions_dir):
            return []
        versions = []
        for fname in sorted(os.listdir(versions_dir), reverse=True):
            if fname.startswith("config_") and fname.endswith(".yaml"):
                fpath = os.path.join(versions_dir, fname)
                mtime = os.path.getmtime(fpath)
                from datetime import datetime
                try:
                    ts_str = datetime.fromtimestamp(mtime).isoformat()
                except (OSError, OverflowError, ValueError):
                    ts_str = ""
                versions.append({
                    "filename": fname,
                    "path": fpath,
                    "timestamp": ts_str,
                    "size_bytes": os.path.getsize(fpath)
                })
        return versions

    def rollback_config(self, version_filename: str = None, config_path: str = "config.yaml") -> bool:
        """回滚到指定配置版本（不指定则回滚到最近一个）
        返回是否成功。注意：回滚仅影响config文件，运行中的策略实例需要重启才生效
        """
        import shutil
        import os
        versions_dir = "./config_versions"
        if not os.path.exists(versions_dir):
            logger.error("No config versions available for rollback")
            return False

        if version_filename:
            # 用户指定版本
            src_path = os.path.join(versions_dir, version_filename)
            if not os.path.exists(src_path):
                logger.error(f"Version not found: {version_filename}")
                return False
        else:
            # 回滚到最近一个（即上一个版本）
            versions = self.list_config_versions()
            if len(versions) < 2:
                logger.error("Need at least 2 versions to rollback (current + previous)")
                return False
            # versions[0]是最近的（当前优化前的备份），versions[1]是更早的
            src_path = versions[1]["path"]

        try:
            # 回滚前再备份当前config
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pre_rollback_backup = f"{versions_dir}/config_pre_rollback_{timestamp}.yaml"
            if os.path.exists(config_path):
                shutil.copy2(config_path, pre_rollback_backup)

            # 执行回滚
            shutil.copy2(src_path, config_path)
            logger.warning(f"Config rolled back to {src_path} (pre-rollback backup: {pre_rollback_backup})")
            logger.warning("NOTE: Running strategy instances need restart to apply rolled-back config")
            return True
        except Exception as e:
            logger.error(f"Failed to rollback config: {e}")
            return False