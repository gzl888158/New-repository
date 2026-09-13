"""
基于历史交易分析优化各策略参数，支持参数热更新与优化记录。
"""
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from core.trade_journal import TradeJournal
from analysis.historical_analyzer import HistoricalAnalyzer

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
        
        overall_recommendations = self._generate_overall_recommendations(analysis)
        recommendations["overall"] = overall_recommendations
        
        await self._record_optimization(recommendations)
        await self._save_learning_state()
        
        return recommendations
    
    async def _optimize_grid(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        strategy_stats = None
        for s in analysis["strategies"]:
            if s["strategy_name"] == "grid":
                strategy_stats = s
                break
        
        if not strategy_stats or strategy_stats["total_trades"] < self._min_trades_for_optimization:
            return {
                "strategy": "grid",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }
        
        recommendations = []
        changes = {}
        
        if strategy_stats["win_rate"] < 0.45:
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
        
        if strategy_stats["profit_factor"] < 1.0:
            recommendations.append({
                "parameter": "martingale_coefficient",
                "current": self.config["strategies"]["grid"]["martingale_coefficient"],
                "recommended": min(1.15, self.config["strategies"]["grid"]["martingale_coefficient"] * 0.95),
                "reason": "Low profit factor suggests martingale is amplifying losses excessively",
                "confidence": 0.7
            })
            changes["martingale_coefficient"] = min(1.15, self.config["strategies"]["grid"]["martingale_coefficient"] * 0.95)
        
        if strategy_stats["total_pnl"] < 0:
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
        worst_hours = [h for h in time_patterns["worst_hours"] if h["avg_pnl"] < -10]
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
        
        if not strategy_stats or strategy_stats["total_trades"] < self._min_trades_for_optimization:
            return {
                "strategy": "trend",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }
        
        recommendations = []
        changes = {}
        locked = self._locked_params.get("trend", set())

        if strategy_stats["win_rate"] < 0.45 and "confirmation_periods" not in locked:
            recommendations.append({
                "parameter": "confirmation_periods",
                "current": self.config["strategies"]["trend"]["confirmation_periods"],
                "recommended": ["1d", "4h", "1h", "30m"],
                "reason": "Low win rate suggests need for additional timeframe confirmation",
                "confidence": 0.75
            })
            changes["confirmation_periods"] = ["1d", "4h", "1h", "30m"]
        elif strategy_stats["win_rate"] < 0.45:
            logger.info("Trend confirmation_periods is locked, skipping auto-tighten (low win rate)")
        
        if strategy_stats["profit_factor"] < 1.0:
            recommendations.append({
                "parameter": "trailing_stop",
                "current": self.config["strategies"]["trend"]["trailing_stop_tier1"],
                "recommended": self.config["strategies"]["trend"]["trailing_stop_tier1"] * 0.8,
                "reason": "Tighter trailing stop to protect profits and improve profit factor",
                "confidence": 0.7
            })
            changes["trailing_stop_tier1"] = self.config["strategies"]["trend"]["trailing_stop_tier1"] * 0.8
            changes["trailing_stop_tier2"] = self.config["strategies"]["trend"]["trailing_stop_tier2"] * 0.8
            changes["trailing_stop_tier3"] = self.config["strategies"]["trend"]["trailing_stop_tier3"] * 0.8
        
        if strategy_stats["total_pnl"] < 0:
            recommendations.append({
                "parameter": "initial_position_ratio",
                "current": self.config["strategies"]["trend"]["initial_position_ratio"],
                "recommended": self.config["strategies"]["trend"]["initial_position_ratio"] * 0.7,
                "reason": "Reduce initial position size to limit drawdown",
                "confidence": 0.65
            })
            changes["initial_position_ratio"] = self.config["strategies"]["trend"]["initial_position_ratio"] * 0.7
        
        success_patterns = analysis["success_patterns"]
        top_strategies = [p for p in success_patterns["patterns"] if p["strategy"] == "trend"]
        if top_strategies and top_strategies[0]["avg_pnl"] > 100:
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
        
        if not strategy_stats or strategy_stats["total_trades"] < self._min_trades_for_optimization:
            return {
                "strategy": "scalping",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }
        
        recommendations = []
        changes = {}
        
        if strategy_stats["win_rate"] < 0.45:
            recommendations.append({
                "parameter": "rsi_oversold",
                "current": self.config["strategies"]["scalping"].get("rsi_oversold", 30),
                "recommended": self.config["strategies"]["scalping"].get("rsi_oversold", 30) - 5,
                "reason": "Increase RSI oversold threshold to filter weaker signals",
                "confidence": 0.75
            })
            changes["rsi_oversold"] = self.config["strategies"]["scalping"].get("rsi_oversold", 30) - 5
            
            recommendations.append({
                "parameter": "rsi_overbought",
                "current": self.config["strategies"]["scalping"].get("rsi_overbought", 70),
                "recommended": self.config["strategies"]["scalping"].get("rsi_overbought", 70) + 5,
                "reason": "Increase RSI overbought threshold to filter weaker signals",
                "confidence": 0.75
            })
            changes["rsi_overbought"] = self.config["strategies"]["scalping"].get("rsi_overbought", 70) + 5
        
        if strategy_stats["profit_factor"] < 1.0:
            recommendations.append({
                "parameter": "profit_target_min",
                "current": self.config["strategies"]["scalping"]["profit_target_min"],
                "recommended": self.config["strategies"]["scalping"]["profit_target_min"] * 1.2,
                "reason": "Increase minimum profit target to improve average win size",
                "confidence": 0.7
            })
            changes["profit_target_min"] = self.config["strategies"]["scalping"]["profit_target_min"] * 1.2
            
            recommendations.append({
                "parameter": "stop_loss",
                "current": self.config["strategies"]["scalping"]["stop_loss"],
                "recommended": self.config["strategies"]["scalping"]["stop_loss"] * 0.9,
                "reason": "Tighten stop loss to reduce average loss size",
                "confidence": 0.7
            })
            changes["stop_loss"] = self.config["strategies"]["scalping"]["stop_loss"] * 0.9
        
        time_patterns = analysis["time_patterns"]
        best_hours = [h for h in time_patterns["best_hours"] if h["avg_pnl"] > 20]
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
        
        if not strategy_stats or strategy_stats["total_trades"] < self._min_trades_for_optimization:
            return {
                "strategy": "arbitrage",
                "optimized": False,
                "reason": "Not enough trades for optimization",
                "recommendations": []
            }
        
        recommendations = []
        changes = {}
        
        if strategy_stats["win_rate"] < 0.45:
            recommendations.append({
                "parameter": "funding_rate_threshold",
                "current": self.config["strategies"]["arbitrage"]["funding_rate_threshold"],
                "recommended": self.config["strategies"]["arbitrage"]["funding_rate_threshold"] * 1.2,
                "reason": "Increase threshold to only take higher confidence funding rate trades",
                "confidence": 0.7
            })
            changes["funding_rate_threshold"] = self.config["strategies"]["arbitrage"]["funding_rate_threshold"] * 1.2
        
        if strategy_stats["total_pnl"] < 0:
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
        if top_strategies and top_strategies[0]["avg_pnl"] > 100:
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

        if not strategy_stats or strategy_stats["total_trades"] < self._min_trades_for_optimization:
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
        
        if strategy_stats["win_rate"] < 0.45 and spot_grid_cfg:
            current_spacing = (spot_grid_cfg.get("min_grid_spacing", 0.01) + spot_grid_cfg.get("max_grid_spacing", 0.03)) / 2
            recommendations.append({
                "parameter": "grid_spacing",
                "current": current_spacing,
                "recommended": spot_grid_cfg.get("min_grid_spacing", 0.01) * 1.3,
                "reason": "Low win rate suggests grid spacing too tight",
                "confidence": 0.7
            })
            changes["min_grid_spacing"] = spot_grid_cfg.get("min_grid_spacing", 0.01) * 1.3

        if strategy_stats["profit_factor"] < 1.0 and spot_grid_cfg:
            current_tp = spot_grid_cfg.get("take_profit_pct", 0.02)
            recommendations.append({
                "parameter": "take_profit_pct",
                "current": current_tp,
                "recommended": current_tp * 1.2,
                "reason": "Increase take profit to improve profit factor",
                "confidence": 0.7
            })
            changes["take_profit_pct"] = current_tp * 1.2

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

        if not strategy_stats or strategy_stats["total_trades"] < self._min_trades_for_optimization:
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

        if strategy_stats["win_rate"] < 0.45 and spot_martingale_cfg:
            current_drop = spot_martingale_cfg.get("price_drop_pct", 0.05)
            recommendations.append({
                "parameter": "price_drop_pct",
                "current": current_drop,
                "recommended": current_drop * 1.3,
                "reason": "Increase price drop threshold to filter weaker signals",
                "confidence": 0.7
            })
            changes["price_drop_pct"] = current_drop * 1.3

        if strategy_stats["profit_factor"] < 1.0 and spot_martingale_cfg:
            current_coef = spot_martingale_cfg.get("martingale_coefficient", 1.1)
            recommendations.append({
                "parameter": "martingale_coefficient",
                "current": current_coef,
                "recommended": min(1.3, current_coef * 0.9),
                "reason": "Reduce martingale coefficient to limit loss amplification",
                "confidence": 0.7
            })
            changes["martingale_coefficient"] = min(1.3, current_coef * 0.9)

        if strategy_stats["total_pnl"] < 0 and spot_martingale_cfg:
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
        overview = analysis["overview"]
        shortcomings = analysis["shortcomings"]
        success_patterns = analysis["success_patterns"]
        
        recommendations = []
        
        if overview["max_drawdown"] > 0.15:
            recommendations.append({
                "category": "risk_management",
                "action": "reduce_overall_leverage",
                "reason": f"Max drawdown {overview['max_drawdown']:.1%} exceeds 15% threshold",
                "suggestion": "Reduce overall leverage by 20% and increase margin requirements",
                "priority": "high"
            })
        
        if overview["sharpe_ratio"] < 1.0:
            recommendations.append({
                "category": "risk_adjusted_return",
                "action": "optimize_risk_reward",
                "reason": f"Sharpe ratio {overview['sharpe_ratio']:.2f} below 1.0 target",
                "suggestion": "Increase average win/loss ratio, reduce trade frequency",
                "priority": "medium"
            })
        
        if len(shortcomings["critical"]) > 0:
            recommendations.append({
                "category": "emergency",
                "action": "address_critical_issues",
                "reason": f"{len(shortcomings['critical'])} critical shortcomings identified",
                "suggestion": "Immediately address critical issues before continuing trading",
                "priority": "critical"
            })
        
        if len(success_patterns["key_factors"]) > 0:
            recommendations.append({
                "category": "capital_allocation",
                "action": "allocate_to_winners",
                "reason": f"{len(success_patterns['key_factors'])} key success factors identified",
                "suggestion": "Increase allocation to strategies/symbols showing strong performance",
                "priority": "medium"
            })
        
        symbol_stats = analysis["symbols"]
        top_symbols = [s for s in symbol_stats if s["total_pnl"] > 0][:3]
        bottom_symbols = [s for s in symbol_stats if s["total_pnl"] < 0][:3]
        
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
            prev_pnl = recent_optimizations[i-1]["analysis_summary"].get("total_pnl", 0)
            curr_pnl = recent_optimizations[i]["analysis_summary"].get("total_pnl", 0)
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
                versions.append({
                    "filename": fname,
                    "path": fpath,
                    "timestamp": datetime.fromtimestamp(mtime).isoformat(),
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