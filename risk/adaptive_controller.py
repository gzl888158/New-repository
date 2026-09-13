"""
自主监控与资金自适应系统（强化版）
- 账户健康监控
- 策略性能分析
- 动态资金分配（多因子决策）
- 自动参数调优
- 异常告警
- 市场状态感知（与 MarketRegimeEngine 集成）
- 置信度衰减机制
- 风险预算动态绑定
"""
import asyncio
import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from core.capital_utilization_engine import (
    CapitalUtilizationEngine, UtilizationTier, UtilizationAction, UtilizationReport
)


class AdaptiveController:
    """自适应控制器：监控 + 分析 + 调优一体化（强化版）"""

    def __init__(self, config: Dict[str, Any], okx_client, sqlite_storage,
                 trade_journal, profit_optimizer, account_manager=None, equity_monitor=None):
        self.config = config
        self.okx_client = okx_client
        self.sqlite_storage = sqlite_storage
        self.trade_journal = trade_journal
        self.profit_optimizer = profit_optimizer
        self.account_manager = account_manager
        self._equity_monitor = equity_monitor
        
        self._regime_engine = None
        self._adaptive_tp_sl_engine = None
        self._adaptive_position_sizer = None

        self._health_check_interval = 60
        self._rebalance_interval = 3600
        self._optimization_interval = 7200
        self._utilization_check_interval = 120  # 2分钟检查一次资金利用率和idle cash优化
        
        self._health_status: Dict[str, Any] = {
            "account": "unknown",
            "strategies": {},
            "system": "unknown",
            "last_check": None
        }
        
        self._dynamic_allocations: Dict[str, float] = {}
        self._allocation_history: List[Dict[str, Any]] = []
        
        self._adaptive_params: Dict[str, Dict[str, Any]] = {}
        
        self._alert_cooldown: Dict[str, datetime] = {}
        
        self._performance_window_hours = 24
        
        trading_cfg = config.get("trading", {})
        self._base_allocations = {
            "grid": trading_cfg.get("grid_allocation", 0.12),
            "trend": trading_cfg.get("trend_allocation", 0.25),
            "scalping": trading_cfg.get("scalping_allocation", 0.28),
            "arbitrage": trading_cfg.get("arbitrage_allocation", 0.13),
            "spot_grid": trading_cfg.get("spot_grid_allocation", 0.12),
            "spot_martingale": trading_cfg.get("spot_martingale_allocation", 0.10),
        }
        self._dynamic_allocations = dict(self._base_allocations)
        
        self._pnl_verification_log: List[Dict[str, Any]] = []
        
        self._equity_history: List[Dict[str, Any]] = []
        self._last_equity_check: Optional[datetime] = None
        
        self._factor_log: List[Dict[str, Any]] = []
        
        self._sizing_log: List[Dict[str, Any]] = []
        
        self._capital_utilization: Dict[str, Any] = {
            "total_used": 0.0,
            "total_available": 0.0,
            "utilization_rate": 0.0,
            "by_strategy": {},
            "history": []
        }
        self._target_utilization = config.get("trading", {}).get("target_utilization", 0.85)
        self._min_utilization = config.get("trading", {}).get("min_utilization", 0.5)
        self._utilization_adjustment_factor = 0.1
        self._start_time = datetime.now()
        self._warmup_minutes = 30
        self._warmup_minutes_with_positions = 10  # P28: 已有持仓时缩短预热期
        
        # P28: 初始化时检测已有持仓，自动缩短预热期
        self._detect_existing_positions_for_warmup()
        
        self._idle_cash_allocation_enabled = config.get("trading", {}).get("idle_cash_allocation", True)
        self._idle_cash_strategy_priority = ["scalping", "arbitrage", "grid", "trend"]
        self._idle_cash_position_boost = 1.0
        self._strategy_optimizer = None

        # ── 企业级自适应资金利用率引擎 ──
        self._utilization_engine = CapitalUtilizationEngine(config)
        self._utilization_engine.set_equity_monitor(equity_monitor)
        # 最近一次利用率报告（供外部获取）
        self._latest_utilization_report: Optional[UtilizationReport] = None

        # 保存策略初始信号质量阈值，用于 auto-tune 回调参考
        self._initial_signal_quality = {}
        for sname, scfg in config.get("strategies", {}).items():
            self._initial_signal_quality[sname] = scfg.get("min_signal_quality", 0.30)

        # 分策略信号质量 floor：放松机制不得低于此值（防止阈值被下调漂移）
        self._signal_quality_floor = {
            "grid": 0.35,
            "trend": 0.30,
            "scalping": 0.15,
            "arbitrage": 0.10,
        }
        
        # 加载锁定参数列表：locked_params 中的参数不会被自动调优覆盖
        self._locked_params: Dict[str, set] = {}
        for sname, scfg in config.get("strategies", {}).items():
            if isinstance(scfg, dict):
                locked = scfg.get("locked_params", [])
                self._locked_params[sname] = set(locked) if locked else set()
        if self._locked_params:
            logger.info(f"Loaded locked params: { {k: list(v) for k, v in self._locked_params.items() if v} }")
        
        # 风险预算系统 —— 从 config.yaml risk_budget 段初始化
        risk_budget_cfg = config.get("risk_budget", {})
        self._risk_budget_enabled = risk_budget_cfg.get("enabled", True)
        self._daily_risk_budget_pct = risk_budget_cfg.get("daily_risk_budget_pct", 0.03)
        self._hourly_max_loss_pct = risk_budget_cfg.get("hourly_max_loss_pct", 0.015)
        self._max_per_trade_risk_pct = risk_budget_cfg.get("max_per_trade_risk_pct", 0.008)
        # 策略级风险预算
        self._risk_budget: Dict[str, float] = {}
        self._strategy_risk_limits: Dict[str, float] = {}
        strategy_budgets = risk_budget_cfg.get("strategy_budgets", {})
        for sname, budget_ratio in strategy_budgets.items():
            self._strategy_risk_limits[sname] = float(budget_ratio)
            self._risk_budget[sname] = float(budget_ratio)
        # 风险预算转移配置
        realloc_cfg = risk_budget_cfg.get("reallocation", {})
        self._rb_realloc_enabled = realloc_cfg.get("enabled", True)
        self._rb_realloc_interval = realloc_cfg.get("interval_minutes", 60) * 60
        self._rb_max_shift = realloc_cfg.get("max_shift_ratio", 0.15)
        self._rb_transfer_out_wr = realloc_cfg.get("transfer_out_winrate_threshold", 0.35)
        self._rb_transfer_out_dd = realloc_cfg.get("transfer_out_drawdown_threshold", 0.08)
        self._rb_min_trades_shift = realloc_cfg.get("min_trades_for_shift", 5)
        self._rb_transfer_in_wr = realloc_cfg.get("transfer_in_winrate_threshold", 0.50)
        self._rb_transfer_in_pf = realloc_cfg.get("transfer_in_profit_factor_threshold", 1.2)
        # 集中度限制
        conc_cfg = risk_budget_cfg.get("concentration", {})
        self._max_symbol_risk_pct = conc_cfg.get("max_single_symbol_risk_pct", 0.008)
        self._max_correlated_risk_pct = conc_cfg.get("max_correlated_group_risk_pct", 0.02)
        self._correlation_threshold = conc_cfg.get("high_correlation_threshold", 0.7)
        # 连续亏损熔断
        streak_cfg = risk_budget_cfg.get("loss_streak", {})
        self._max_consecutive_losses = streak_cfg.get("max_consecutive_losses", 5)
        self._streak_reduce_pct = streak_cfg.get("loss_streak_reduce_pct", 0.50)
        self._streak_recovery_wins = streak_cfg.get("recovery_consecutive_wins", 3)
        # 运行时风险跟踪
        self._daily_risk_consumed: Dict[str, float] = {}  # 策略 -> 当日已消耗风险(USDT)
        self._hourly_pnl: float = 0.0  # 当前小时盈亏
        self._hour_start: datetime = datetime.now()
        self._last_risk_realloc: datetime = datetime.min
        self._consecutive_loss_count: int = 0
        self._consecutive_win_count: int = 0
        self._streak_lock_active: bool = False  # 熔断锁
        self._symbol_risk_exposure: Dict[str, float] = {}  # 币种 -> 风险敞口
        self._risk_budget_log: List[Dict[str, Any]] = []   # 风险预算操作日志
        
        self._dynamic_leverage_enabled = config.get("trading", {}).get("dynamic_leverage", True)
        self._base_leverage = config.get("trading", {}).get("default_leverage", 5)
        
        self._confidence_decay: Dict[str, float] = {}
        self._confidence_half_life_hours = 24
        self._confidence_min_threshold = 0.1
        
        self._multi_factor_weights = {
            "performance": 0.40,
            "market_regime": 0.30,
            "risk_budget": 0.15,
            "utilization": 0.15,
        }
        
        logger.info("AdaptiveController initialized (enhanced)")

    def set_regime_engine(self, engine):
        """注入MarketRegimeEngine实例"""
        self._regime_engine = engine
        logger.info("MarketRegimeEngine injected")

    def set_adaptive_tp_sl_engine(self, engine):
        """注入统一自适应止损止盈引擎（供策略/上层获取自适应 SL/TP 参考）"""
        self._adaptive_tp_sl_engine = engine
        logger.info("AdaptiveTpSlEngine injected into AdaptiveController")

    def set_adaptive_position_sizer(self, engine):
        """注入统一自适应仓位引擎（供策略/上层复用统一仓位口径）"""
        self._adaptive_position_sizer = engine
        logger.info("AdaptivePositionSizer injected into AdaptiveController")

    def compute_adaptive_tp_sl(self, symbol: str, entry_price: float, direction: str, **kwargs) -> dict:
        """基于统一引擎计算自适应止损止盈（策略/上层复用入口）。

        引擎未注入时返回空 dict，调用方应自行回退。
        """
        if self._adaptive_tp_sl_engine is None:
            return {}
        try:
            ctx = self._adaptive_tp_sl_engine.build_context(symbol)
            params = {k: v for k, v in ctx.items() if k != "symbol"}
            params.update(kwargs)
            return self._adaptive_tp_sl_engine.compute(
                symbol=symbol, entry_price=entry_price, direction=direction, **params
            )
        except Exception as e:
            logger.warning(f"compute_adaptive_tp_sl failed for {symbol}: {e}")
            return {}

    def compute_position_size(self, symbol: str, price: float, account_balance: float,
                              strategy_name: str = "grid", **kwargs) -> dict:
        """基于统一仓位引擎计算自适应仓位（策略/上层复用入口）。

        引擎未注入时返回空 dict，调用方应回退到旧路径
        （PositionPlanner / AdaptiveKelly / StrategyRisk）。
        """
        if self._adaptive_position_sizer is None:
            return {}
        try:
            result = self._adaptive_position_sizer.compute(
                symbol=symbol,
                price=price,
                account_balance=account_balance,
                strategy_name=strategy_name,
                **kwargs,
            )
            return result.to_dict()
        except Exception as e:
            logger.warning(f"compute_position_size failed for {symbol}: {e}")
            return {}

    def set_strategy_optimizer(self, optimizer):
        """注入StrategyOptimizer实例，用于热更新策略参数"""
        self._strategy_optimizer = optimizer

    def set_equity_monitor(self, monitor):
        """注入EquityMonitor实例，用于资金变动自适应检测"""
        self._equity_monitor = monitor
        logger.info("EquityMonitor injected into AdaptiveController")

    async def start(self):
        """启动所有监控和自适应循环"""
        asyncio.create_task(self._health_monitor_loop())
        asyncio.create_task(self._rebalance_loop())
        asyncio.create_task(self._optimization_loop())
        asyncio.create_task(self._pnl_verification_loop())
        asyncio.create_task(self._capital_utilization_loop())
        # 启动企业级资金利用率引擎（加载状态 + 持久化循环）
        await self._utilization_engine.start()
        logger.info("AdaptiveController started")

    async def stop(self):
        """停止所有监控和自适应循环（优雅关闭）"""
        await self._utilization_engine.stop()
        logger.info("AdaptiveController stopped")

    # ===================== 健康监控 =====================

    async def _health_monitor_loop(self):
        """账户和系统健康监控循环"""
        while True:
            try:
                self._reset_daily_risk_counters()
                await self._check_account_health()
                await self._check_system_health()
                await self._check_position_consistency()
                self._persist_risk_budget_state()
                self._health_status["last_check"] = datetime.now().isoformat()
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
            await asyncio.sleep(self._health_check_interval)

    async def _check_account_health(self):
        """检查账户健康状态"""
        try:
            account_info = self.okx_client.get_account_info()
            if not account_info:
                self._health_status["account"] = "unreachable"
                await self._alert("account_unreachable", "无法获取账户信息", "critical")
                return

            account = self.okx_client._parse_account_info(account_info)
            equity = account.total_equity
            initial_capital = self.config["trading"].get("total_capital", 100)

            # 更新 ProfitOptimizer 权益
            if equity > 0:
                self.profit_optimizer.update_equity(equity)

            # 回撤检查
            stats = self.profit_optimizer.get_stats()
            drawdown = stats.get("drawdown", 0)
            max_dd = self.config["trading"].get("max_drawdown", 0.12)

            if drawdown > max_dd:
                self._health_status["account"] = "critical"
                await self._alert("max_drawdown", f"最大回撤 {drawdown:.1%} 超过阈值 {max_dd:.1%}", "critical")
                await self._emergency_drawdown_protection(drawdown)
            elif drawdown > max_dd * 0.7:
                self._health_status["account"] = "warning"
                await self._alert("drawdown_warning", f"回撤 {drawdown:.1%} 接近阈值 {max_dd:.1%}", "warning")
            else:
                self._health_status["account"] = "healthy"

            # 每日亏损检查 —— P0: 仅检查负值（亏损），abs()会导致正盈利也触警
            daily_pnl = self._calc_daily_loss()
            daily_max = self.config["trading"].get("daily_max_loss", 0.04)
            if daily_pnl < -initial_capital * daily_max:
                await self._alert("daily_loss_limit", f"当日亏损 {daily_pnl:.2f} 超过阈值", "warning")

            # 保证金使用率检查：account.margin_rate = used_margin / total_equity
            # 阈值对齐资金利用率逻辑：85% 是目标利用率（仅 warning）；
            # 只有 >95%（与 _check_capital_utilization 的 high 阈值一致）才视为强平风险 CRITICAL。
            # 注意：这是账户级使用率，单仓位的维持保证金率检查由 global_risk.py 负责
            if account.margin_rate > 0.95:
                await self._alert("high_margin_usage", f"保证金使用率过高: {account.margin_rate:.2%}", "critical")
            elif account.margin_rate > 0.85:
                await self._alert("high_margin_usage", f"保证金使用率偏高: {account.margin_rate:.2%}", "warning")

            # 权益异常检测：对比OKX权益变化与数据库已实现PnL
            await self._check_equity_anomaly(equity)

        except Exception as e:
            logger.error(f"Account health check failed: {e}")

    async def _check_equity_anomaly(self, current_equity: float):
        """检测权益异常：对比实际权益变化和数据库记录的PnL"""
        try:
            now = datetime.now()
            self._equity_history.append({
                "timestamp": now,
                "equity": current_equity
            })
            if len(self._equity_history) > 100:
                self._equity_history = self._equity_history[-100:]

            # 至少有2个数据点才检测
            if len(self._equity_history) < 2:
                return

            # 计算这段时间的实际权益变化
            first_equity = self._equity_history[0]["equity"]
            actual_change = current_equity - first_equity

            # 计算数据库记录的已实现PnL（从同一时间段开始）
            start_time = self._equity_history[0]["timestamp"]
            all_trades = self.sqlite_storage.get_trade_records(limit=200)
            db_realized_pnl = 0.0
            for t in all_trades:
                if t.get("status") == "closed" and (t.get("pnl") or 0) != 0:
                    ct = t.get("close_time")
                    if ct and hasattr(ct, "timestamp"):
                        if ct.timestamp() >= start_time.timestamp():
                            db_realized_pnl += (t.get("pnl") or 0.0)

            # 偏差 = 实际变化 - 已实现PnL - 未实现PnL（近似）
            # 正常情况下偏差应该很小（主要是手续费误差）
            deviation = abs(actual_change - db_realized_pnl)
            deviation_pct = deviation / first_equity if first_equity > 0 else 0

            if deviation_pct > 0.05:  # 偏差超过5%
                await self._alert(
                    "equity_mismatch",
                    f"权益偏差过大: 实际变化 {actual_change:+.2f}, 数据库已实现 {db_realized_pnl:+.2f}, "
                    f"偏差 {deviation:.2f} ({deviation_pct:.1%})",
                    "warning"
                )
                logger.warning(f"Equity anomaly detected: actual={actual_change:+.4f}, db_pnl={db_realized_pnl:+.4f}, "
                               f"deviation={deviation:.4f} ({deviation_pct:.1%})")
        except Exception as e:
            logger.error(f"Equity anomaly check failed: {e}")

    def _calc_daily_loss(self) -> float:
        """计算当日盈亏"""
        try:
            today = datetime.now().date()
            closed_trades = self.sqlite_storage.get_trade_records(limit=200)
            daily_pnl = 0.0
            for t in closed_trades:
                if t.get("status") == "closed":
                    ct = t.get("close_time")
                    if ct and hasattr(ct, "date"):
                        if ct.date() == today:
                            daily_pnl += (t.get("pnl") or 0.0)
            return daily_pnl
        except Exception:
            return 0.0

    async def _check_system_health(self):
        """检查系统组件健康状态"""
        checks = {}

        # SQLite
        try:
            checks["sqlite"] = self.sqlite_storage.health_check()
        except Exception:
            checks["sqlite"] = False

        # OKX API
        try:
            ticker = self.okx_client.get_ticker("BTC-USDT-SWAP")
            checks["okx_api"] = ticker is not None
        except Exception:
            checks["okx_api"] = False

        # 策略状态
        for strategy_name in ["grid", "trend", "scalping", "arbitrage"]:
            strategy_cfg = self.config["strategies"].get(strategy_name, {})
            enabled = strategy_cfg.get("enabled", True)
            checks[f"strategy_{strategy_name}"] = "enabled" if enabled else "disabled"
            self._health_status["strategies"][strategy_name] = {
                "enabled": enabled,
                "allocation": self._dynamic_allocations.get(strategy_name, 0)
            }

        all_healthy = all(v for k, v in checks.items() if isinstance(v, bool))
        self._health_status["system"] = "healthy" if all_healthy else "degraded"

        if not checks.get("okx_api", False):
            await self._alert("okx_api_down", "OKX API 无法访问", "critical")
        if not checks.get("sqlite", False):
            await self._alert("sqlite_down", "SQLite 数据库异常", "critical")

    async def _check_position_consistency(self):
        """检查持仓一致性（对账）"""
        try:
            positions = self.okx_client.get_positions()
            okx_positions = {}
            for pos_data in positions:
                pos = self.okx_client._parse_position(pos_data)
                if pos and abs(pos.quantity) > 0:
                    key = f"{pos.symbol}:{pos.side}"
                    okx_positions[key] = pos

            db_open = self.sqlite_storage.get_trade_records_by_status("open")
            db_positions = {}
            for rec in db_open:
                symbol = rec.get("symbol", "")
                side = (rec.get("side", "") or rec.get("direction", "") or "").lower()
                if side in ("buy", "long"):
                    norm_side = "long"
                elif side in ("sell", "short"):
                    norm_side = "short"
                else:
                    norm_side = side
                key = f"{symbol}:{norm_side}"
                db_positions[key] = rec

            # 检测幽灵持仓（数据库有但OKX没有）
            ghosts = [k for k in db_positions if k not in okx_positions]
            if ghosts:
                logger.warning(f"Found {len(ghosts)} ghost positions: {ghosts}")
                for g in ghosts:
                    rec = db_positions[g]
                    # 幽灵持仓的 pnl 无法从本地数据准确核算（entry/quantity 可能陈旧、量纲也可能不一致），
                    # 一律不写 pnl（留 NULL），交由 PnLReconciler 用 OKX 平仓账单的权威 pnl 精确对账，
                    # 避免用 mark_price 估算出假数据（历史事故：单笔被算成 +71% 或真实亏损 3.7 倍）。
                    updates = {
                        "status": "closed",
                        "close_time": datetime.now().isoformat(),
                        "exit_reason": "ghost_cleanup",
                    }
                    self.sqlite_storage.update_trade_record(rec.get("id"), updates)
                logger.info(f"Cleaned {len(ghosts)} ghost positions")

            # 检测丢失持仓（OKX有但数据库没有）
            missing = [k for k in okx_positions if k not in db_positions]
            if missing:
                logger.info(f"Found {len(missing)} positions not in DB: {missing}")
                # P7-4: 自动同步交易所持仓到数据库
                for m in missing:
                    pos = okx_positions[m]
                    try:
                        self.sqlite_storage.save_trade_record({
                            "symbol": pos.symbol,
                            "side": pos.side,
                            "price": pos.avg_cost,
                            # 账本一致性（修复3）：pos.quantity 是 OKX 返回的合约张数，须转成币数写入，
                            # 与 trades.quantity 及 order_executor 开仓写入的币数量口径对齐，消除量纲错乱。
                            "quantity": self.okx_client.contracts_to_coins(pos.symbol, abs(pos.quantity)),
                            "leverage": pos.leverage,
                            "create_time": datetime.now(),
                            "status": "open",
                            "strategy_name": "sync",
                            "pnl": 0,
                            "pnl_percent": 0,
                        })
                        logger.info(f"P7-4: Synced missing position {m} to DB")
                    except Exception as sync_err:
                        logger.warning(f"P7-4: Failed to sync missing position {m}: {sync_err}")

        except Exception as e:
            logger.error(f"Position consistency check failed: {e}")

    async def _emergency_drawdown_protection(self, drawdown: float):
        """紧急回撤保护：降低仓位和杠杆"""
        logger.warning(f"Emergency drawdown protection activated: {drawdown:.1%}")

        # 按回撤程度降权
        reduction_factor = max(0.3, 1 - drawdown * 3)

        for strategy in self._dynamic_allocations:
            self._dynamic_allocations[strategy] *= reduction_factor

        # 不再归一化：归一化会把各策略权重按比例拉回 1.0，导致统一降权被完全抵消。
        # 保留权重和 < 1.0 的缺口，剩余资金自动回到现金，从而真实降低总敞口。
        total = sum(self._dynamic_allocations.values())
        logger.info(f"Emergency allocation adjustment (total={total:.3f}): {self._dynamic_allocations}")

    # ===================== 资金自适应 =====================

    async def _rebalance_loop(self):
        """资金再平衡循环（含风险预算转移）"""
        while True:
            try:
                await self._enhanced_rebalance_allocations()
                # 风险预算转移：从表现差的策略转出预算
                await self._reallocate_risk_budgets()
            except Exception as e:
                logger.error(f"Rebalance loop error: {e}")
            await asyncio.sleep(self._rebalance_interval)

    async def _rebalance_allocations(self):
        """基于多因子决策动态调整资金分配（强化版）
        
        已废弃：当前使用 _enhanced_rebalance_allocations，此方法保留用于未来AB测试
        """
        strategy_perf = self._analyze_strategy_performance()

        if not strategy_perf:
            return

        performance_scores = self._calculate_performance_scores(strategy_perf)
        
        regime_scores = self._calculate_regime_scores()
        
        risk_scores = self._calculate_risk_scores(strategy_perf)
        
        utilization_scores = self._calculate_utilization_scores()
        
        composite_scores = self._calculate_composite_scores(
            performance_scores, regime_scores, risk_scores, utilization_scores
        )

        self._update_confidence_decay(composite_scores)

        new_allocations = self._compute_final_allocations(composite_scores)

        old_alloc = dict(self._dynamic_allocations)
        self._dynamic_allocations = new_allocations

        if old_alloc != new_allocations:
            logger.info(f"Allocation rebalanced (multi-factor): {old_alloc} -> {new_allocations}")
            self._allocation_history.append({
                "timestamp": datetime.now().isoformat(),
                "old": old_alloc,
                "new": new_allocations,
                "reason": "multi_factor_decision",
                "factor_scores": {
                    "performance": performance_scores,
                    "regime": regime_scores,
                    "risk": risk_scores,
                    "utilization": utilization_scores,
                    "composite": composite_scores,
                }
            })

    def _calculate_performance_scores(self, strategy_perf: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """计算性能因子得分（0-1），融入夏普/卡尔玛/索提诺比率"""
        scores = {}

        for strategy, perf in strategy_perf.items():
            if perf["total_trades"] < 3:
                scores[strategy] = self._base_allocations.get(strategy, 0.25)
            else:
                win_rate = perf.get("win_rate", 0.5)
                profit_factor = perf.get("profit_factor", 1.0)
                avg_pnl = perf.get("avg_pnl", 0)
                max_drawdown = perf.get("max_drawdown", 0)
                sharpe = perf.get("sharpe_ratio", 0)
                calmar = perf.get("calmar_ratio", 0)
                sortino = perf.get("sortino_ratio", 0)

                # 夏普得分：0附近=0.5，正值加、负值减 (限[-5, 5]区间)
                sharpe_score = 0.5 + max(-0.5, min(0.5, sharpe * 0.1))
                # 卡尔玛得分：>0为优秀
                calmar_score = 0.5 + max(-0.5, min(0.5, calmar * 0.05))
                # 索提诺得分：同理
                sortino_score = 0.5 + max(-0.5, min(0.5, sortino * 0.1))

                score = (
                    win_rate * 0.20 +
                    min(profit_factor, 3.0) / 3.0 * 0.15 +
                    (1 if avg_pnl > 0 else 0.3) * 0.15 +
                    min(perf["total_trades"] / 20, 1.0) * 0.10 +
                    max(0, 1 - max_drawdown * 5) * 0.10 +
                    sharpe_score * 0.10 +
                    calmar_score * 0.10 +
                    sortino_score * 0.10
                )
                scores[strategy] = max(0.1, min(1.0, score))

        return scores

    def _calculate_regime_scores(self) -> Dict[str, float]:
        """计算市场状态因子得分（基于 MarketRegimeEngine）"""
        scores = {s: 0.5 for s in self._base_allocations}

        if self._regime_engine:
            adjustment = self._regime_engine.get_position_adjustment()
            for strategy in self._base_allocations:
                strat_adjust = adjustment.get(strategy, adjustment.get("overall", 1.0))
                scores[strategy] = min(1.0, max(0.1, strat_adjust))

            # 趋势/震荡 regime 下的方向性倾斜：趋势跟随类（trend/arbitrage）与
            # 均值回归类（grid/scalping）在相反 regime 中调低权重，使资金向当前
            # regime 下更易开仓的策略集中，提升小账户资金利用率（配合 RegimeGate 顺趋势放行）。
            regime_str = ""
            try:
                ro = self._regime_engine.get_regime()
                regime_str = str(ro.get("regime", "") or "").lower()
            except Exception:
                regime_str = ""

            if regime_str in ("trend_bullish", "trend_bearish"):
                for s in ("trend", "arbitrage"):
                    if s in scores:
                        scores[s] = min(1.0, scores[s] + 0.2)
                for s in ("grid", "scalping"):
                    if s in scores:
                        scores[s] = max(0.1, scores[s] - 0.2)
            elif regime_str == "range_bound":
                for s in ("grid", "scalping"):
                    if s in scores:
                        scores[s] = min(1.0, scores[s] + 0.2)
                for s in ("trend", "arbitrage"):
                    if s in scores:
                        scores[s] = max(0.1, scores[s] - 0.2)

        return scores

    def _calculate_risk_scores(self, strategy_perf: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """计算风险预算因子得分（0-1）
        
        基于已消耗风险预算 vs 分配预算的比率：
        - 已消耗少 → 高分（风险空间充足）
        - 已消耗多 → 低分（风险空间不足，需要收紧）
        """
        scores = {}
        if not self._risk_budget_enabled:
            return {s: 0.5 for s in strategy_perf}

        # 获取当前权益作为计算基准
        total_equity = self._capital_utilization.get("total_equity", 0)
        if total_equity <= 0:
            total_equity = self.config.get("trading", {}).get("total_capital", 559.29)

        for strategy in strategy_perf:
            budget_ratio = self._strategy_risk_limits.get(strategy, 0.2)
            # 该策略当日预算上限(USDT)
            daily_budget_usdt = total_equity * self._daily_risk_budget_pct * budget_ratio
            consumed = self._daily_risk_consumed.get(strategy, 0)
            # 已用比例
            used_ratio = abs(consumed) / daily_budget_usdt if daily_budget_usdt > 0 else 0
            # 剩余空间大→高分, 已耗尽→低分
            score = max(0.1, 1.0 - used_ratio)

            # 额外惩罚：如果该策略正在亏损且已消耗超过60%
            if strategy_perf.get(strategy, {}).get("total_pnl", 0) < 0 and used_ratio > 0.6:
                score *= 0.7

            scores[strategy] = score

        return scores

    def _calculate_utilization_scores(self) -> Dict[str, float]:
        """计算资金利用率因子得分（0-1）"""
        scores = {s: 0.5 for s in self._base_allocations}
        
        utilization = self._capital_utilization.get("utilization_rate", 0)
        strategy_usage = self._capital_utilization.get("by_strategy", {})
        
        for strategy in self._base_allocations:
            strat_usage = strategy_usage.get(strategy, 0)
            alloc = self._dynamic_allocations.get(strategy, 0)
            
            if alloc > 0:
                actual_ratio = strat_usage / (utilization + 0.01) if utilization > 0 else 0
                target_ratio = alloc
                
                if actual_ratio < target_ratio * 0.5:
                    scores[strategy] = 0.8
                elif actual_ratio > target_ratio * 1.5:
                    scores[strategy] = 0.4
                else:
                    scores[strategy] = 0.6

        return scores

    def _calculate_composite_scores(self, performance: Dict[str, float],
                                    regime: Dict[str, float], risk: Dict[str, float],
                                    utilization: Dict[str, float]) -> Dict[str, float]:
        """计算多因子综合得分"""
        scores = {}
        
        for strategy in self._base_allocations:
            perf_score = performance.get(strategy, 0.5)
            regime_score = regime.get(strategy, 0.5)
            risk_score = risk.get(strategy, 0.5)
            util_score = utilization.get(strategy, 0.5)
            
            confidence = self._confidence_decay.get(strategy, 1.0)
            
            composite = (
                perf_score * self._multi_factor_weights["performance"] +
                regime_score * self._multi_factor_weights["market_regime"] +
                risk_score * self._multi_factor_weights["risk_budget"] +
                util_score * self._multi_factor_weights["utilization"]
            ) * confidence
            
            scores[strategy] = max(0.1, min(1.0, composite))
        
        return scores

    def _update_confidence_decay(self, scores: Dict[str, float]):
        """更新置信度衰减（基于时间和表现）"""
        now = datetime.now()
        
        for strategy in self._base_allocations:
            score = scores[strategy]
            
            if strategy not in self._confidence_decay:
                self._confidence_decay[strategy] = 1.0
            
            if score > 0.7:
                self._confidence_decay[strategy] = min(1.0, self._confidence_decay[strategy] + 0.05)
            elif score < 0.3:
                self._confidence_decay[strategy] = max(self._confidence_min_threshold, 
                                                       self._confidence_decay[strategy] * 0.9)
            else:
                half_life_seconds = self._confidence_half_life_hours * 3600
                decay_factor = np.exp(-np.log(2) / half_life_seconds * 3600)
                self._confidence_decay[strategy] = max(self._confidence_min_threshold,
                                                       self._confidence_decay[strategy] * decay_factor)

    def _compute_final_allocations(self, composite_scores: Dict[str, float]) -> Dict[str, float]:
        """计算最终资金分配"""
        new_allocations = {}
        
        for strategy, score in composite_scores.items():
            base = self._base_allocations.get(strategy, 0.25)
            scores_sum = sum(composite_scores.values())
            
            if scores_sum > 0:
                perf_ratio = score / scores_sum
            else:
                perf_ratio = base
            
            new_allocations[strategy] = base * 0.6 + perf_ratio * 0.4

        total_new = sum(new_allocations.values())
        if total_new > 0:
            for s in new_allocations:
                new_allocations[s] = new_allocations[s] / total_new

        for strategy in new_allocations:
            old = self._dynamic_allocations.get(strategy, 0.25)
            new = new_allocations[strategy]
            max_change = old * 0.15
            if abs(new - old) > max_change:
                if new > old:
                    new_allocations[strategy] = old + max_change
                else:
                    new_allocations[strategy] = old - max_change

        total_final = sum(new_allocations.values())
        if total_final > 0:
            for s in new_allocations:
                new_allocations[s] = new_allocations[s] / total_final

        return new_allocations

    def _analyze_strategy_performance(self, days: int = 3) -> Dict[str, Dict[str, Any]]:
        """分析各策略表现（强化版：增加最大回撤+夏普/卡尔玛/索提诺比率）
        
        Args:
            days: 仅分析最近N天的交易数据，避免陈旧数据误导auto-tune
        """
        performances = {}
        cutoff = datetime.now() - timedelta(days=days)

        for strategy_name in ["grid", "trend", "scalping", "arbitrage"]:
            records = self.sqlite_storage.get_trade_records(
                strategy_name=strategy_name, limit=200
            )

            closed = [r for r in records if r.get("status") == "closed"]
            
            # P6-1: 时间过滤 - 仅使用最近N天的交易数据
            recent_closed = []
            for r in closed:
                close_time = r.get("close_time") or r.get("exit_time")
                if close_time:
                    if isinstance(close_time, str):
                        try:
                            close_time = datetime.fromisoformat(close_time)
                        except (ValueError, TypeError):
                            close_time = None
                    if close_time and close_time >= cutoff:
                        recent_closed.append(r)
                else:
                    # 无close_time的记录保守处理：跳过
                    pass
            
            if not recent_closed:
                performances[strategy_name] = {
                    "total_trades": 0,
                    "win_rate": 0.5,
                    "profit_factor": 1.0,
                    "total_pnl": 0,
                    "avg_pnl": 0,
                    "avg_win": 0,
                    "avg_loss": 1,
                    "max_drawdown": 0.0,
                    "sharpe_ratio": 0.0,
                    "calmar_ratio": 0.0,
                    "sortino_ratio": 0.0,
                    "stale": True,  # P6-1: 标记数据陈旧，禁止auto-tune
                }
                continue
            
            # 使用过滤后的recent_closed替换原closed
            closed = recent_closed
            # 归一化 pnl：数据库记录 pnl 可能为 None（未平仓即写入/历史脏数据），
            # 统一按 0.0 处理，避免 `None > 0` 等比较抛 TypeError（fail-safe）
            for r in closed:
                if r.get("pnl") is None:
                    r["pnl"] = 0.0
            wins = [r for r in closed if r.get("pnl", 0) > 0]
            losses = [r for r in closed if r.get("pnl", 0) < 0]

            total_pnl = sum(r.get("pnl", 0) for r in closed)
            win_rate = len(wins) / len(closed) if closed else 0.5

            avg_win = np.mean([r["pnl"] for r in wins]) if wins else 0
            avg_loss = abs(np.mean([r["pnl"] for r in losses])) if losses else 1
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0

            max_drawdown = self._calculate_max_drawdown(closed)

            # === P1: 夏普/卡尔玛/索提诺比率 ===
            pnl_list = [r.get("pnl", 0) for r in closed]
            sharpe = 0.0
            calmar = 0.0
            sortino = 0.0

            if len(pnl_list) >= 3:
                pnl_arr = np.array(pnl_list, dtype=float)
                mean_pnl = float(np.mean(pnl_arr))
                std_pnl = float(np.std(pnl_arr))
                # 年化近似：交易次数开方
                trade_factor = np.sqrt(max(len(pnl_arr), 1))
                if std_pnl > 0:
                    sharpe = mean_pnl / std_pnl * trade_factor
                # 卡尔玛 = 总收益 / 最大回撤
                if max_drawdown > 0:
                    calmar = total_pnl / max_drawdown
                # 索提诺 = 均值 / 下行标准差
                down_pnl = pnl_arr[pnl_arr < 0]
                if len(down_pnl) > 0:
                    down_std = float(np.std(down_pnl))
                    if down_std > 0:
                        sortino = mean_pnl / down_std * trade_factor

            performances[strategy_name] = {
                "total_trades": len(closed),
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / len(closed) if closed else 0,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe,
                "calmar_ratio": calmar,
                "sortino_ratio": sortino,
            }

        return performances

    def _calculate_max_drawdown(self, trades: List[Dict[str, Any]]) -> float:
        """计算最大回撤"""
        if not trades:
            return 0.0

        equity_curve = []
        running_total = 0.0

        for trade in sorted(trades, key=lambda x: x.get("close_time", "")):
            running_total += (trade.get("pnl") or 0.0)
            equity_curve.append(running_total)

        if not equity_curve:
            return 0.0

        max_equity = equity_curve[0]
        max_dd = 0.0

        for eq in equity_curve:
            max_equity = max(max_equity, eq)
            if max_equity != 0:
                dd = (max_equity - eq) / abs(max_equity)
                max_dd = max(max_dd, dd)

        return max_dd

    def get_allocation(self, strategy_name: str) -> float:
        """获取策略的动态分配比例"""
        return self._dynamic_allocations.get(
            strategy_name,
            self._base_allocations.get(strategy_name, 0.25)
        )

    def _is_strategy_enabled(self, strategy_name: str) -> bool:
        """判断策略是否启用（未显式配置 enabled 时默认启用）。

        用于资金分配链路：disabled 策略（如现货 spot_grid/spot_martingale）
        不应占用资金权重，避免其分配被 allocation_shift / dynamic_allocator 重新拉起。
        """
        return bool(self.config.get("strategies", {}).get(strategy_name, {}).get("enabled", True))

    # ===================== 风险预算系统：核心检查与执行 =====================

    def check_trade_risk_budget(self, strategy_name: str, symbol: str,
                                 trade_risk_usdt: float, equity: float) -> Tuple[bool, str]:
        """在每笔交易前检查风险预算是否允许
        
        Args:
            strategy_name: 策略名称
            symbol: 交易币种
            trade_risk_usdt: 该笔交易的预估风险(USDT)
            equity: 当前总权益
        
        Returns:
            (是否允许, 原因说明)
        """
        if not self._risk_budget_enabled:
            return True, "risk_budget_disabled"

        # 1. 连续亏损熔断检查
        if self._streak_lock_active:
            return False, f"loss_streak_lock: consecutive_losses={self._consecutive_loss_count}"

        # 2. 单笔风险上限检查
        max_per_trade_risk = equity * self._max_per_trade_risk_pct
        if trade_risk_usdt > max_per_trade_risk:
            return False, f"per_trade_risk_exceeded: {trade_risk_usdt:.4f} > {max_per_trade_risk:.4f}"

        # 3. 策略当日风险预算检查
        budget_ratio = self._strategy_risk_limits.get(strategy_name, 0.2)
        daily_budget_usdt = equity * self._daily_risk_budget_pct * budget_ratio
        consumed = abs(self._daily_risk_consumed.get(strategy_name, 0))

        if consumed + trade_risk_usdt > daily_budget_usdt:
            remaining = max(0, daily_budget_usdt - consumed)
            return False, (f"strategy_risk_budget_exhausted: {strategy_name} "
                          f"consumed={consumed:.4f}/{daily_budget_usdt:.4f}, "
                          f"remaining={remaining:.4f}, needed={trade_risk_usdt:.4f}")

        # 4. 单币种集中度检查
        symbol_exposure = self._symbol_risk_exposure.get(symbol, 0)
        max_symbol_risk = equity * self._max_symbol_risk_pct
        if symbol_exposure + trade_risk_usdt > max_symbol_risk:
            return False, (f"symbol_concentration_exceeded: {symbol} "
                          f"exposure={symbol_exposure:.4f}, "
                          f"adding={trade_risk_usdt:.4f}, limit={max_symbol_risk:.4f}")

        # 5. 每小时亏损上限检查
        if self._hourly_pnl < -equity * self._hourly_max_loss_pct:
            return False, (f"hourly_loss_limit: pnl={self._hourly_pnl:.4f}, "
                          f"limit={-equity * self._hourly_max_loss_pct:.4f}")

        return True, "approved"

    def record_risk_consumption(self, strategy_name: str, symbol: str,
                                 pnl: float, risk_amount: float):
        """记录风险消耗（每次交易完成后调用）"""
        if not self._risk_budget_enabled:
            return

        # 累计当日风险消耗（亏损才算消耗）
        if pnl < 0:
            self._daily_risk_consumed[strategy_name] = \
                self._daily_risk_consumed.get(strategy_name, 0) + abs(pnl)

        # 更新币种风险敞口
        if pnl < 0:
            self._symbol_risk_exposure[symbol] = \
                self._symbol_risk_exposure.get(symbol, 0) + risk_amount

        # 更新连续盈亏计数
        if pnl < 0:
            self._consecutive_loss_count += 1
            self._consecutive_win_count = 0
            # 连续亏损熔断检查
            if self._consecutive_loss_count >= self._max_consecutive_losses:
                self._streak_lock_active = True
                self._apply_loss_streak_lock()
        elif pnl > 0:
            self._consecutive_win_count += 1
            self._consecutive_loss_count = 0
            # 连续盈利恢复检查
            if self._streak_lock_active and \
               self._consecutive_win_count >= self._streak_recovery_wins:
                self._release_loss_streak_lock()

        # 更新小时盈亏
        self._hourly_pnl += pnl
        now = datetime.now()
        if (now - self._hour_start).total_seconds() >= 3600:
            self._hourly_pnl = 0
            self._hour_start = now

        # 日志记录
        self._risk_budget_log.append({
            "timestamp": now.isoformat(),
            "strategy": strategy_name,
            "symbol": symbol,
            "pnl": round(pnl, 4),
            "risk_amount": round(risk_amount, 4),
            "daily_consumed": round(self._daily_risk_consumed.get(strategy_name, 0), 4),
            "consecutive_losses": self._consecutive_loss_count,
            "streak_lock": self._streak_lock_active,
        })
        if len(self._risk_budget_log) > 200:
            self._risk_budget_log = self._risk_budget_log[-200:]

    def _apply_loss_streak_lock(self):
        """连续亏损熔断：削减所有策略风险预算"""
        reduce_pct = self._streak_reduce_pct
        for sname in self._strategy_risk_limits:
            old_budget = self._risk_budget.get(sname, 0)
            self._risk_budget[sname] = old_budget * (1 - reduce_pct)
        # 重新归一化
        total = sum(self._risk_budget.values())
        if total > 0:
            for sname in self._risk_budget:
                self._risk_budget[sname] /= total

        logger.warning(f"[RISK_BUDGET] Loss streak lock activated: "
                       f"{self._consecutive_loss_count} consecutive losses, "
                       f"budgets reduced by {reduce_pct:.0%}")
        self._risk_budget_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "streak_lock_activated",
            "consecutive_losses": self._consecutive_loss_count,
            "new_budgets": dict(self._risk_budget),
        })

    def _release_loss_streak_lock(self):
        """连续盈利恢复：恢复原始风险预算"""
        self._streak_lock_active = False
        strategy_budgets = self.config.get("risk_budget", {}).get("strategy_budgets", {})
        for sname in self._strategy_risk_limits:
            self._risk_budget[sname] = float(strategy_budgets.get(sname, self._strategy_risk_limits.get(sname, 0.2)))

        logger.info(f"[RISK_BUDGET] Loss streak lock released: "
                    f"{self._consecutive_win_count} consecutive wins, budgets restored")
        self._risk_budget_log.append({
            "timestamp": datetime.now().isoformat(),
            "event": "streak_lock_released",
            "consecutive_wins": self._consecutive_win_count,
        })

    async def _reallocate_risk_budgets(self):
        """风险预算转移：从表现差的策略转出预算给表现好的策略"""
        if not self._risk_budget_enabled or not self._rb_realloc_enabled:
            return

        now = datetime.now()
        if (now - self._last_risk_realloc).total_seconds() < self._rb_realloc_interval:
            return
        self._last_risk_realloc = now

        strategy_perf = self._analyze_strategy_performance()
        transfers = []

        # 找出需要转出的策略（表现差）
        donors = []
        receivers = []
        for sname, perf in strategy_perf.items():
            if perf["total_trades"] < self._rb_min_trades_shift:
                continue
            wr = perf.get("win_rate", 0.5)
            dd = perf.get("max_drawdown", 0)
            pf = perf.get("profit_factor", 1.0)

            # 转出条件
            if wr < self._rb_transfer_out_wr or dd > self._rb_transfer_out_dd:
                donors.append((sname, wr, dd))
            # 接收条件
            if wr >= self._rb_transfer_in_wr and pf >= self._rb_transfer_in_pf:
                receivers.append((sname, wr, pf))

        if not donors or not receivers:
            return

        # 执行转移：每个donor转出max_shift给最好的receiver
        best_receiver = max(receivers, key=lambda x: x[1])[0]

        for donor_name, donor_wr, donor_dd in donors:
            donor_budget = self._risk_budget.get(donor_name, 0)
            shift = min(donor_budget * 0.5, self._rb_max_shift)
            if shift < 0.01:
                continue

            self._risk_budget[donor_name] -= shift
            self._risk_budget[best_receiver] += shift
            transfers.append({
                "from": donor_name,
                "to": best_receiver,
                "amount": round(shift, 4),
                "reason": f"donor_wr={donor_wr:.1%}, dd={donor_dd:.1%}"
            })

        # 归一化
        total = sum(self._risk_budget.values())
        if total > 0:
            for sname in self._risk_budget:
                self._risk_budget[sname] /= total

        if transfers:
            logger.info(f"[RISK_BUDGET_REALLOC] {len(transfers)} transfers: {transfers}")
            self._risk_budget_log.append({
                "timestamp": now.isoformat(),
                "event": "reallocation",
                "transfers": transfers,
                "new_budgets": {s: round(v, 4) for s, v in self._risk_budget.items()},
            })

    def update_symbol_exposure(self, symbol: str, risk_usdt: float, action: str = "add"):
        """更新单币种风险敞口
        
        Args:
            symbol: 交易币种
            risk_usdt: 风险敞口变化量
            action: 'add' 增加, 'remove' 减少
        """
        if action == "add":
            self._symbol_risk_exposure[symbol] = \
                self._symbol_risk_exposure.get(symbol, 0) + risk_usdt
        elif action == "remove":
            self._symbol_risk_exposure[symbol] = \
                max(0, self._symbol_risk_exposure.get(symbol, 0) - risk_usdt)

    def _reset_daily_risk_counters(self):
        """每日重置风险计数器（跨日时调用）"""
        today = datetime.now().strftime("%Y-%m-%d")
        if not hasattr(self, '_last_reset_date'):
            self._last_reset_date = today
        if self._last_reset_date != today:
            logger.info(f"[RISK_BUDGET] Daily reset: {self._last_reset_date} -> {today} "
                       f"(consumed={sum(self._daily_risk_consumed.values()):.4f})")
            self._daily_risk_consumed = {}
            self._hourly_pnl = 0.0
            self._hour_start = datetime.now()
            self._symbol_risk_exposure = {}
            self._last_reset_date = today

    def get_risk_budget_status(self) -> Dict[str, Any]:
        """获取风险预算状态（供 API 和策略查询）"""
        total_equity = self._capital_utilization.get("total_equity", 0)
        if total_equity <= 0:
            total_equity = self.config.get("trading", {}).get("total_capital", 559.29)

        strategy_status = {}
        for sname in self._strategy_risk_limits:
            budget_ratio = self._risk_budget.get(sname, self._strategy_risk_limits.get(sname, 0))
            daily_limit = total_equity * self._daily_risk_budget_pct * budget_ratio
            consumed = abs(self._daily_risk_consumed.get(sname, 0))
            strategy_status[sname] = {
                "budget_ratio": round(budget_ratio, 4),
                "daily_limit_usdt": round(daily_limit, 4),
                "consumed_usdt": round(consumed, 4),
                "remaining_usdt": round(max(0, daily_limit - consumed), 4),
                "usage_pct": round(consumed / daily_limit * 100, 1) if daily_limit > 0 else 0,
                "healthy": consumed < daily_limit * 0.8,
            }

        return {
            "enabled": self._risk_budget_enabled,
            "total_equity": round(total_equity, 2),
            "daily_total_budget": round(total_equity * self._daily_risk_budget_pct, 4),
            "total_consumed": round(sum(abs(v) for v in self._daily_risk_consumed.values()), 4),
            "max_per_trade_risk": round(total_equity * self._max_per_trade_risk_pct, 4),
            "streak_lock_active": self._streak_lock_active,
            "consecutive_losses": self._consecutive_loss_count,
            "consecutive_wins": self._consecutive_win_count,
            "hourly_pnl": round(self._hourly_pnl, 4),
            "strategy_status": strategy_status,
            "symbol_exposure": {s: round(v, 4) for s, v in self._symbol_risk_exposure.items()},
            "last_update": datetime.now().isoformat(),
        }

    def _persist_risk_budget_state(self):
        """将风险预算状态持久化到 data/risk_budget_state.json（供 dashboard API 读取）"""
        try:
            state = self.get_risk_budget_status()
            state_path = os.path.join("data", "risk_budget_state.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Persist risk budget state error: {e}")

    # ===================== 自动调优 =====================

    async def _optimization_loop(self):
        """自动参数优化循环"""
        while True:
            try:
                await self._auto_tune_parameters()
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
            await asyncio.sleep(self._optimization_interval)

    async def _auto_tune_parameters(self):
        """基于历史表现自动调优策略参数（受 locked_params 约束）
        
        P6-1: 仅使用最近3天的交易数据，陈旧数据不参与调优
        """
        strategy_perf = self._analyze_strategy_performance(days=3)

        for strategy_name, perf in strategy_perf.items():
            # P6-1: 数据陈旧或无近期交易，跳过调优
            if perf.get("stale"):
                logger.debug(f"Auto-tune {strategy_name}: no recent trades (3d), skipping")
                continue
            
            if perf["total_trades"] < 5:
                logger.debug(f"Auto-tune {strategy_name}: only {perf['total_trades']} recent trades, need >=5, skipping")
                continue

            strategy_cfg = self.config["strategies"].get(strategy_name, {})
            changes = {}
            locked = self._locked_params.get(strategy_name, set())

            # 胜率低于40%，提高信号质量要求
            if perf["win_rate"] < 0.40 and "min_signal_quality" not in locked:
                current_quality = strategy_cfg.get("min_signal_quality", 0.30)
                new_quality = min(0.70, current_quality + 0.05)
                if new_quality > current_quality:
                    changes["min_signal_quality"] = new_quality
                    logger.info(f"Auto-tune {strategy_name}: min_signal_quality {current_quality} -> {new_quality} (low win_rate)")

            # 胜率恢复正常或资金利用率低，回调阈值（避免只升不降）
            elif perf["win_rate"] >= 0.45 and "min_signal_quality" not in locked:
                current_quality = strategy_cfg.get("min_signal_quality", 0.30)
                # 读取该策略的初始阈值
                initial_quality = self._initial_signal_quality.get(strategy_name, current_quality)
                if current_quality > initial_quality:
                    new_quality = max(initial_quality, current_quality - 0.05)
                    if new_quality < current_quality:
                        changes["min_signal_quality"] = new_quality
                        logger.info(f"Auto-tune {strategy_name}: min_signal_quality {current_quality} -> {new_quality} (win_rate recovered)")

            # 资金利用率过低时，强制回调阈值
            utilization = self._capital_utilization.get("utilization_rate", 0)
            if utilization < self._min_utilization and not changes.get("min_signal_quality") and "min_signal_quality" not in locked:
                current_quality = strategy_cfg.get("min_signal_quality", 0.30)
                initial_quality = self._initial_signal_quality.get(strategy_name, current_quality)
                if current_quality > initial_quality:
                    new_quality = max(initial_quality, current_quality - 0.05)
                    changes["min_signal_quality"] = new_quality
                    logger.info(f"Auto-tune {strategy_name}: min_signal_quality {current_quality} -> {new_quality} (low utilization {utilization:.0%})")

            # 盈亏比低于1.0，收紧止损
            if perf["profit_factor"] < 1.0 and "stop_loss_pct" not in locked:
                if strategy_name == "grid":
                    current_sl = strategy_cfg.get("stop_loss_pct", 0.02)
                    new_sl = max(0.008, current_sl * 0.9)
                    changes["stop_loss_pct"] = new_sl
                    logger.info(f"Auto-tune {strategy_name}: stop_loss {current_sl} -> {new_sl} (low PF)")

            # 连续亏损，降低仓位
            if perf["total_pnl"] < 0 and perf["total_trades"] > 5:
                current_alloc = self._dynamic_allocations.get(strategy_name, 0.25)
                self._dynamic_allocations[strategy_name] = current_alloc * 0.9
                logger.info(f"Auto-tune {strategy_name}: allocation reduced by 10% (losing streak)")

            # 应用配置变更
            if changes:
                if strategy_name in self.config["strategies"]:
                    self.config["strategies"][strategy_name].update(changes)

        # 重新归一化分配
        total = sum(self._dynamic_allocations.values())
        if total > 0:
            for s in self._dynamic_allocations:
                self._dynamic_allocations[s] /= total

        # 更新 ProfitOptimizer 的交易历史
        self._sync_trade_history_to_optimizer()

        # 记录因子状态
        if self.profit_optimizer:
            stats = self.profit_optimizer.get_stats()
            self._factor_log.append({
                "timestamp": datetime.now().isoformat(),
                "compound_factor": stats.get("compound_factor", 1.0),
                "kelly_factor": stats.get("kelly_factor", 1.0),
                "drawdown_factor": stats.get("drawdown_factor", 1.0),
                "current_equity": stats.get("current_equity", 0),
                "peak_equity": stats.get("peak_equity", 0),
                "drawdown": stats.get("drawdown", 0),
                "win_rate": stats.get("win_rate", 0),
                "profit_factor": stats.get("profit_factor", 0),
                "total_trades": stats.get("total_trades", 0),
            })
            if len(self._factor_log) > 100:
                self._factor_log = self._factor_log[-100:]
            logger.info(
                f"[FACTORS] compound={stats.get('compound_factor', 1.0):.3f}, "
                f"kelly={stats.get('kelly_factor', 1.0):.3f}, "
                f"drawdown={stats.get('drawdown_factor', 1.0):.3f}, "
                f"equity={stats.get('current_equity', 0):.2f}, "
                f"win_rate={stats.get('win_rate', 0):.1%}, "
                f"trades={stats.get('total_trades', 0)}"
            )

    def _sync_trade_history_to_optimizer(self):
        """同步交易历史到ProfitOptimizer（用于凯利公式）
        
        P0: 使用已同步交易ID集合去重，避免重复记录导致胜率统计膨胀
        """
        if not self.profit_optimizer:
            return

        # P1: 初始化已同步集合（persistent across calls）
        if not hasattr(self, '_synced_trade_ids'):
            self._synced_trade_ids = set()
        
        closed_trades = self.sqlite_storage.get_trade_records(limit=100)
        new_trades = 0

        for t in closed_trades:
            trade_id = t.get("id") or t.get("order_id", "")
            if trade_id and trade_id in self._synced_trade_ids:
                continue
            if t.get("status") == "closed" and (t.get("pnl") or 0) != 0:
                pnl = t.get("pnl") or 0.0
                strategy = t.get("strategy_name", "")
                self.profit_optimizer.record_trade_result(pnl, strategy)
                new_trades += 1
                if trade_id:
                    self._synced_trade_ids.add(trade_id)
        
        # 防止已同步集合无限膨胀（保留最近200条）
        if len(self._synced_trade_ids) > 200:
            self._synced_trade_ids = set(list(self._synced_trade_ids)[-100:])

        if new_trades > 0:
            logger.debug(f"Synced {new_trades} trades to ProfitOptimizer")

    # ===================== 盈亏验证 =====================

    async def _pnl_verification_loop(self):
        """盈亏计算验证循环"""
        while True:
            try:
                await self._verify_pnl_calculation()
            except Exception as e:
                logger.error(f"PnL verification error: {e}")
            await asyncio.sleep(self._rebalance_interval)

    async def _verify_pnl_calculation(self):
        """验证盈亏计算准确性（对比OKX数据和数据库数据）"""
        try:
            positions = self.okx_client.get_positions()
            verification_results = []

            for pos_data in positions:
                pos = self.okx_client._parse_position(pos_data)
                if not pos or abs(pos.quantity) == 0:
                    continue

                # 查找数据库中对应的open持仓
                db_open = self.sqlite_storage.get_trade_records_by_status("open")
                matched = None
                for rec in db_open:
                    if rec.get("symbol") == pos.symbol:
                        side = (rec.get("side", "") or rec.get("direction", "") or "").lower()
                        if side in ("buy", "long") and pos.side == "long":
                            matched = rec
                            break
                        elif side in ("sell", "short") and pos.side == "short":
                            matched = rec
                            break

                if matched:
                    db_entry_price = matched.get("price", 0)
                    okx_entry_price = pos.avg_cost
                    price_diff_pct = abs(db_entry_price - okx_entry_price) / okx_entry_price if okx_entry_price > 0 else 0

                    # 计算期望PnL vs OKX实际PnL
                    direction = 1 if pos.side == "long" else -1
                    expected_pnl = (pos.mark_price - okx_entry_price) * abs(pos.quantity) * direction
                    actual_pnl = pos.unrealized_pnl
                    pnl_diff = abs(expected_pnl - actual_pnl)

                    verification_results.append({
                        "symbol": pos.symbol,
                        "side": pos.side,
                        "db_entry_price": db_entry_price,
                        "okx_entry_price": okx_entry_price,
                        "price_diff_pct": price_diff_pct,
                        "expected_pnl": expected_pnl,
                        "actual_pnl": actual_pnl,
                        "pnl_diff": pnl_diff,
                        "accurate": price_diff_pct < 0.01 and pnl_diff < abs(actual_pnl) * 0.05
                    })

            if verification_results:
                accurate_count = sum(1 for v in verification_results if v["accurate"])
                accuracy = accurate_count / len(verification_results)
                self._pnl_verification_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "total": len(verification_results),
                    "accurate": accurate_count,
                    "accuracy": accuracy
                })

                if accuracy < 0.8:
                    logger.warning(f"PnL verification accuracy low: {accuracy:.1%}")

        except Exception as e:
            logger.error(f"PnL verification failed: {e}")

    # ===================== 告警系统 =====================

    async def _alert(self, alert_id: str, message: str, severity: str = "info"):
        """发送告警（带冷却）"""
        cooldown_map = {
            "critical": 600,
            "warning": 1800,
            "info": 3600,
        }
        cooldown = cooldown_map.get(severity, 3600)

        last_alert = self._alert_cooldown.get(alert_id)
        now = datetime.now()
        if last_alert and (now - last_alert).total_seconds() < cooldown:
            return

        self._alert_cooldown[alert_id] = now
        log_func = {
            "critical": logger.critical,
            "warning": logger.warning,
            "info": logger.info,
        }.get(severity, logger.info)

        log_func(f"[ALERT-{severity.upper()}] {alert_id}: {message}")

    # ===================== 状态查询 =====================

    def get_status(self) -> Dict[str, Any]:
        """获取当前监控状态"""
        return {
            "health": self._health_status,
            "allocations": self._dynamic_allocations,
            "base_allocations": self._base_allocations,
            "allocation_history_count": len(self._allocation_history),
            "pnl_verification": self._pnl_verification_log[-1] if self._pnl_verification_log else None,
            "profit_optimizer": self.profit_optimizer.get_stats() if self.profit_optimizer else {},
            "factor_log_count": len(self._factor_log),
            "sizing_log_count": len(self._sizing_log),
            "equity_history_count": len(self._equity_history),
            "latest_factors": self._factor_log[-1] if self._factor_log else None,
            "equity_monitor": self.get_equity_monitor_status(),
        }

    def get_equity_monitor_status(self) -> Dict[str, Any]:
        """获取EquityMonitor资金变动状态"""
        if self._equity_monitor is None:
            return {"available": False}
        return self._equity_monitor.get_equity_status()

    def get_adjusted_position_size(self, strategy_name: str, base_margin: float) -> float:
        """获取调整后的仓位大小（融合动态分配+复利+凯利+增强凯利）"""
        # 动态分配比例
        alloc_ratio = self.get_allocation(strategy_name)
        base_ratio = self._base_allocations.get(strategy_name, 0.25)
        alloc_multiplier = alloc_ratio / base_ratio if base_ratio > 0 else 1.0

        # P2-3: 增强凯利因子（策略级贝叶斯+波动率惩罚）
        enhanced_kelly_factor = self._get_enhanced_kelly_factor(strategy_name)

        # ProfitOptimizer 优化
        if self.profit_optimizer:
            optimized = self.profit_optimizer.get_optimal_position_size(
                base_margin * alloc_multiplier, 5
            )

            # P2-3: 应用增强凯利因子
            optimized *= enhanced_kelly_factor

            # 记录因子明细（每10次记录一次，避免过多日志）
            stats = self.profit_optimizer.get_stats()
            if len(self._sizing_log) == 0 or \
               (datetime.now() - self._sizing_log[-1]["timestamp"]).total_seconds() > 300:
                self._sizing_log.append({
                    "timestamp": datetime.now(),
                    "strategy": strategy_name,
                    "base_margin": base_margin,
                    "alloc_multiplier": alloc_multiplier,
                    "compound_factor": stats.get("compound_factor", 1.0),
                    "kelly_factor": stats.get("kelly_factor", 1.0),
                    "drawdown_factor": stats.get("drawdown_factor", 1.0),
                    "enhanced_kelly_factor": enhanced_kelly_factor,
                    "final_margin": optimized,
                    "adjustment_ratio": optimized / base_margin if base_margin > 0 else 1.0
                })
                if len(self._sizing_log) > 100:
                    self._sizing_log = self._sizing_log[-100:]
                logger.info(
                    f"[SIZING] {strategy_name}: base={base_margin:.4f}, "
                    f"alloc_x{alloc_multiplier:.2f}, "
                    f"compound_x{stats.get('compound_factor', 1.0):.2f}, "
                    f"kelly_x{stats.get('kelly_factor', 1.0):.2f}, "
                    f"dd_x{stats.get('drawdown_factor', 1.0):.2f}, "
                    f"enhkelly_x{enhanced_kelly_factor:.2f}, "
                    f"final={optimized:.4f} "
                    f"({optimized/base_margin*100:.1f}% of base)" if base_margin > 0 else ""
                )

            return optimized

        return base_margin * alloc_multiplier * enhanced_kelly_factor

    def get_sizing_log(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取仓位调整日志"""
        return self._sizing_log[-limit:] if self._sizing_log else []

    def get_equity_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取权益历史"""
        return self._equity_history[-limit:] if self._equity_history else []

    # ===================== 资金利用率优化 =====================

    async def _capital_utilization_loop(self):
        """资金利用率监控循环"""
        # 记录上次处理的重置信号时间戳，避免重复处理
        self._last_reset_signal_ts: float = 0.0
        while True:
            try:
                # 检测重置信号文件（由 dashboard_api 写入）
                await self._check_reset_signal()
                await self._check_capital_utilization()
                await self._optimize_idle_cash_allocation()
                await self._sync_dynamic_allocator()
            except Exception as e:
                logger.error(f"Capital utilization check error: {e}")
            await asyncio.sleep(self._utilization_check_interval)

    async def _check_reset_signal(self):
        """检测资金利用率重置信号文件"""
        try:
            reset_path = os.path.join("data", "capital_utilization_reset.json")
            if not os.path.exists(reset_path):
                return
            with open(reset_path, "r", encoding="utf-8") as f:
                signal = json.load(f)
            signal_ts = float(signal.get("timestamp", 0) or 0)
            # 仅处理比上次更新的信号
            if signal_ts <= self._last_reset_signal_ts:
                return
            self._last_reset_signal_ts = signal_ts
            source = signal.get("source", "unknown")
            logger.info(f"Capital utilization reset signal received (source={source}, ts={signal_ts})")
            self.reset_capital_utilization(source=source)
            # 处理完成后删除信号文件
            try:
                os.remove(reset_path)
            except OSError:
                pass
        except Exception as e:
            logger.debug(f"Reset signal check error: {e}")

    def reset_capital_utilization(self, source: str = "manual") -> Dict[str, Any]:
        """重置资金利用率状态：
        - 清空利用率历史
        - 重置 position_boost 为 1.0
        - 恢复信号质量阈值到初始值
        - 重置 warmup 起始时间，重新进入预热期

        Returns:
            重置前的状态快照
        """
        try:
            # 保存重置前快照
            snapshot = {
                "utilization_rate": self._capital_utilization.get("utilization_rate", 0),
                "position_boost": getattr(self, "_idle_cash_position_boost", 1.0),
                "status": self._capital_utilization.get("status", "normal"),
                "history_len": len(self._capital_utilization.get("history", [])),
            }

            # 1. 重置利用率状态（保留 by_strategy 结构，清零数值）
            self._capital_utilization = {
                "total_used": 0.0,
                "total_available": 0.0,
                "utilization_rate": 0.0,
                "target_utilization": self._target_utilization,
                "by_strategy": {},
                "history": [],
                "status": "warming_up",
                "reset_at": datetime.now().isoformat(),
                "reset_source": source,
            }

            # 2. 重置 position_boost 为 1.0
            self._idle_cash_position_boost = 1.0

            # 3. 恢复信号质量阈值到初始值（关闭 idle cash 放松）
            self._reset_idle_cash_optimizations()

            # 4. 重置 warmup 起始时间，重新进入预热期
            # P28: 持有仓位时缩短预热期
            self._start_time = datetime.now()
            self._detect_existing_positions_for_warmup()

            # 5. 企业级引擎完整重置（清空历史/快照/振荡状态/动态目标）
            engine_before = None
            try:
                if self._utilization_engine is not None:
                    engine_before = self._utilization_engine.reset_full_state()
            except Exception as e:
                logger.error(f"Utilization engine full reset failed: {e}")

            # 6. 清空最近一次引擎报告，避免下游读到过期建议
            self._latest_utilization_report = None
            snapshot["engine_reset"] = engine_before

            logger.info(
                f"Capital utilization reset by '{source}': "
                f"cleared {snapshot['history_len']} history entries, "
                f"position_boost {snapshot['position_boost']:.2f} -> 1.00, "
                f"warmup restarted. Previous utilization={snapshot['utilization_rate']:.1%}"
            )
            return snapshot
        except Exception as e:
            logger.error(f"reset_capital_utilization error: {e}")
            return {"error": str(e)}

    async def _check_capital_utilization(self):
        """检查资金利用率"""
        try:
            # 优先从OKX获取实时账户数据，回退到account_manager
            total_equity = 0.0
            used_margin = 0.0
            available = 0.0

            try:
                account_info = self.okx_client.get_account_info()
                if account_info:
                    # 优先从USDT明细读取（account级别availEq常为空字符串）
                    usdt_frozen = 0.0
                    details = account_info.get("details", [])
                    for detail in details:
                        if detail.get("ccy") == "USDT":
                            total_equity = float(detail.get("eq", 0) or 0)
                            available = float(detail.get("availBal", 0) or 0)
                            usdt_frozen = float(detail.get("frozenBal", 0) or 0)
                            break
                    if total_equity <= 0:
                        total_equity = float(account_info.get("totalEq", 0) or 0)
                    # frozenBal 只反映挂单冻结，不含持仓保证金（尤其cross模式）
                    # 因此使用 frozenBal 和 (total_equity - available) 中较大值，并从 positions API 获取最准确值
                    used_margin = max(usdt_frozen, max(0.0, total_equity - available))
                    # 从 positions API 汇总实际保证金占用（cross用imr，isolated用margin）
                    try:
                        positions = self.okx_client.get_positions()
                        positions_margin = 0.0
                        for pos in positions or []:
                            pos_qty = float(pos.get("pos", 0) or 0)
                            if pos_qty == 0:
                                continue
                            margin = pos.get("margin", "")
                            imr = pos.get("imr", "")
                            lever = float(pos.get("lever", 1) or 1)
                            notional = float(pos.get("notionalUsd", 0) or 0)
                            if margin and margin != "":
                                m = float(margin)
                            elif imr and imr != "":
                                m = float(imr)
                            elif lever > 0 and notional > 0:
                                m = notional / lever
                            else:
                                m = 0.0
                            positions_margin += m
                        if positions_margin > used_margin:
                            used_margin = positions_margin
                    except Exception as pe:
                        logger.debug(f"Positions margin calc fallback failed: {pe}")
            except Exception:
                pass

            # 回退到account_manager获取已用保证金
            if used_margin <= 0 and self.account_manager:
                summary = self.account_manager.get_account_summary()
                used_margin = float(summary.get("total_margin_used", 0) or 0)
                if total_equity <= 0:
                    total_equity = float(summary.get("total_capital", 0) or 0)
                if available <= 0:
                    available = total_equity - used_margin

            utilization_rate = used_margin / total_equity if total_equity > 0 else 0
            
            strategy_usage = {}
            positions = self.sqlite_storage.get_trade_records_by_status("open")
            for pos in positions:
                strategy = pos.get("strategy_name") or pos.get("strategy") or "unknown"
                margin = pos.get("margin", 0)
                strategy_usage[strategy] = strategy_usage.get(strategy, 0) + margin
            
            self._capital_utilization = {
                "total_used": used_margin,
                "total_available": available,
                "total_equity": total_equity,
                "utilization_rate": utilization_rate,
                "target_utilization": self._target_utilization,
                "by_strategy": strategy_usage,
                "timestamp": datetime.now().isoformat(),
                "history": self._capital_utilization.get("history", []),
                "reset_at": self._capital_utilization.get("reset_at"),
                "reset_source": self._capital_utilization.get("reset_source"),
            }
            
            self._capital_utilization["history"].append({
                "total_used": used_margin,
                "utilization_rate": utilization_rate,
                "timestamp": datetime.now().isoformat()
            })
            if len(self._capital_utilization["history"]) > 100:
                self._capital_utilization["history"] = self._capital_utilization["history"][-100:]
            
            # ── 企业级自适应资金利用率引擎分析 ──
            atr_ratio = self._get_atr_ratio()
            recent_pnl = self._get_recent_strategy_pnl()
            drawdown_pct = self._get_total_drawdown_pct()
            report = self._utilization_engine.analyze(
                total_equity=total_equity,
                used_margin=used_margin,
                available=available,
                strategy_usage=strategy_usage,
                recent_pnl=recent_pnl,
                atr_ratio=atr_ratio,
                total_drawdown_pct=drawdown_pct,
            )
            self._latest_utilization_report = report

            # 同步引擎输出到旧字段（兼容 dashboard / 下游）
            self._capital_utilization["target_utilization"] = report.dynamic_target
            self._capital_utilization["utilization_tier"] = report.utilization_tier.value
            self._capital_utilization["recommended_action"] = report.recommended_action.value
            self._capital_utilization["capital_efficiency"] = report.capital_efficiency
            self._capital_utilization["utilization_trend"] = report.utilization_trend
            self._capital_utilization["utilization_volatility"] = report.utilization_volatility
            self._capital_utilization["volatility_regime"] = report.volatility_regime
            self._capital_utilization["session"] = report.session
            self._capital_utilization["strategy_targets"] = report.strategy_targets
            # 用引擎建议覆盖 position_boost，并施加 risk_budget 封顶
            self._idle_cash_position_boost = self._cap_position_boost_by_risk_budget(report.position_boost)
            self._capital_utilization["position_boost"] = self._idle_cash_position_boost

            # 接通 allocation_shift → 动态分配
            self._apply_allocation_shift(report.allocation_shift)

            logger.info(
                f"Capital utilization: {utilization_rate:.1%} (target: {report.dynamic_target:.0%}, "
                f"tier: {report.utilization_tier.value}, action: {report.recommended_action.value}, "
                f"vol: {report.volatility_regime}, session: {report.session}, equity: {total_equity:.2f}, used: {used_margin:.2f})"
            )

            elapsed_minutes = (datetime.now() - self._start_time).total_seconds() / 60
            is_warmup = elapsed_minutes < self._warmup_minutes

            # ── 状态判断（基于引擎报告，迟滞防振荡）──
            action = report.recommended_action
            tier = report.utilization_tier

            if action == UtilizationAction.EMERGENCY_FREEZE:
                self._capital_utilization["status"] = "emergency_blocked"
            elif action == UtilizationAction.FORCE_REBALANCE:
                logger.warning(
                    f"Force rebalance triggered: capital efficiency {report.capital_efficiency:.4f} "
                    f"< {CapitalUtilizationEngine.FORCE_REBALANCE_EFFICIENCY_THRESHOLD} "
                    f"with equity {total_equity:.2f} > {CapitalUtilizationEngine.FORCE_REBALANCE_MIN_EQUITY}"
                )
                self._capital_utilization["status"] = "force_rebalance"
                await self._alert(
                    "utilization_force_rebalance",
                    f"资本效率过低触发强制再平衡：效率 {report.capital_efficiency:.4f}，权益 {total_equity:.2f}",
                    "warning",
                )
            elif tier == UtilizationTier.CRITICAL_HIGH:
                logger.warning(f"Critical high capital utilization: {utilization_rate:.1%} > 98%")
                self._capital_utilization["status"] = "high"
                await self._trigger_utilization_alert("high", utilization_rate)
            elif tier == UtilizationTier.HIGH:
                logger.warning(f"High capital utilization: {utilization_rate:.1%} > 95%")
                self._capital_utilization["status"] = "high"
                await self._trigger_utilization_alert("high", utilization_rate)
            elif tier in (UtilizationTier.CRITICAL_LOW, UtilizationTier.LOW):
                if is_warmup:
                    logger.info(f"Capital utilization low ({utilization_rate:.1%}) - system warming up ({elapsed_minutes:.0f}/{self._warmup_minutes} min), building positions")
                    self._capital_utilization["status"] = "warming_up"
                else:
                    logger.warning(f"Low capital utilization: {utilization_rate:.1%} (dynamic target: {report.dynamic_target:.0%})")
                    self._capital_utilization["status"] = "low"
                    await self._trigger_utilization_alert("low", utilization_rate)
            else:
                self._capital_utilization["status"] = "normal"

            # ── EquityMonitor 自适应参数注入 ──
            if self._equity_monitor is not None:
                eq_params = self._equity_monitor.get_adaptive_params()
                self._capital_utilization["equity_mode"] = eq_params.get("mode", "normal")
                self._capital_utilization["equity_position_multiplier"] = eq_params.get("position_multiplier", 1.0)
                self._capital_utilization["equity_signal_quality_offset"] = eq_params.get("signal_quality_offset", 0.0)
                self._capital_utilization["equity_risk_budget_ratio"] = eq_params.get("risk_budget_ratio", 1.0)
                self._capital_utilization["equity_account_tier"] = eq_params.get("account_tier", "nano")
                # 紧急模式：禁止开新仓
                if not self._equity_monitor.is_new_position_allowed():
                    logger.warning("EquityMonitor EMERGENCY mode: blocking new positions")
                    self._capital_utilization["status"] = "emergency_blocked"

            # 持久化状态到文件，供 dashboard_api 和客户端读取
            self._persist_utilization_state()

        except Exception as e:
            logger.error(f"Error checking capital utilization: {e}")

    def _persist_utilization_state(self):
        """将资金利用率状态持久化到 data/capital_utilization_state.json"""
        try:
            state = {
                "utilization_rate": self._capital_utilization.get("utilization_rate", 0),
                "total_used": self._capital_utilization.get("total_used", 0),
                "total_available": self._capital_utilization.get("total_available", 0),
                "total_equity": self._capital_utilization.get("total_equity", 0),
                "target_utilization": self._capital_utilization.get("target_utilization", self._target_utilization),
                "status": self._capital_utilization.get("status", "normal"),
                "reset_at": self._capital_utilization.get("reset_at"),
                "reset_source": self._capital_utilization.get("reset_source"),
                "position_boost": getattr(self, "_idle_cash_position_boost", 1.0),
                "timestamp": self._capital_utilization.get("timestamp", datetime.now().isoformat()),
                "by_strategy": self._capital_utilization.get("by_strategy", {}),
                "equity_mode": self._capital_utilization.get("equity_mode", "normal"),
                "equity_position_multiplier": self._capital_utilization.get("equity_position_multiplier", 1.0),
                "equity_account_tier": self._capital_utilization.get("equity_account_tier", "nano"),
                # 企业级引擎新增字段
                "utilization_tier": self._capital_utilization.get("utilization_tier", "unknown"),
                "recommended_action": self._capital_utilization.get("recommended_action", "none"),
                "capital_efficiency": self._capital_utilization.get("capital_efficiency", 0.0),
                "utilization_trend": self._capital_utilization.get("utilization_trend", 0.0),
                "utilization_volatility": self._capital_utilization.get("utilization_volatility", 0.0),
                "volatility_regime": self._capital_utilization.get("volatility_regime", "normal"),
                "session": self._capital_utilization.get("session", "unknown"),
                "strategy_targets": self._capital_utilization.get("strategy_targets", {}),
            }
            state_path = os.path.join("data", "capital_utilization_state.json")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            with open(state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"Persist utilization state error: {e}")

    def _detect_existing_positions_for_warmup(self):
        """P28: 检测已有持仓，自动缩短预热期
        如果系统中已有持仓记录（如重启后恢复），预热期从30分钟缩短到10分钟
        """
        try:
            # 检查是否有持仓状态文件
            state_path = os.path.join("data", "capital_utilization_state.json")
            if os.path.exists(state_path):
                with open(state_path, "r", encoding="utf-8") as f:
                    state = json.load(f)
                total_used = state.get("total_used", 0)
                if total_used > 0:
                    self._warmup_minutes = self._warmup_minutes_with_positions
                    logger.info(
                        f"P28: Existing positions detected (used={total_used:.2f} USDT), "
                        f"warmup shortened to {self._warmup_minutes} min"
                    )
        except Exception as e:
            logger.debug(f"P28: Warmup detection error: {e}")

    async def _optimize_idle_cash_allocation(self):
        """优化空闲资金分配：企业级引擎驱动（强化版）
        1. 调整策略分配比例（向表现好的策略倾斜）
        2. 应用引擎的信号质量阈值放松量（连续函数，可正可负）
        3. 应用引擎的仓位乘数（连续函数，已在上游 _check_capital_utilization 更新）
        """
        if not self._idle_cash_allocation_enabled:
            return

        try:
            utilization = self._capital_utilization.get("utilization_rate", 0)
            status = self._capital_utilization.get("status", "normal")
            report = self._latest_utilization_report

            # 目标利用率：优先使用引擎动态目标
            target = report.dynamic_target if report else self._target_utilization

            # 预热期内允许阈值放松和仓位放大，但不调整策略分配比例
            if status == "warming_up":
                if report is not None:
                    self._apply_engine_signal_relaxation(report.signal_relaxation)
                else:
                    severity = self._calculate_idle_severity(utilization)
                    if severity > 0:
                        self._apply_signal_threshold_relaxation(severity)
                        self._apply_position_size_boost(severity, utilization)
                return

            if utilization >= target:
                # 利用率达到动态目标，恢复默认参数
                if report is not None:
                    self._apply_engine_signal_relaxation(report.signal_relaxation)
                else:
                    self._reset_idle_cash_optimizations()
                return

            idle_ratio = 1.0 - utilization
            if idle_ratio < 0.1:
                return

            # === 1. 调整策略分配比例 ===
            target_increase = (target - utilization) * 0.5
            if target_increase > 0:
                strategy_performances = self._get_strategy_performances()

                allocation_adjustments = {}
                remaining_increase = target_increase

                for strategy in self._idle_cash_strategy_priority:
                    if remaining_increase <= 0:
                        break

                    perf = strategy_performances.get(strategy, {})
                    win_rate = perf.get("win_rate", 0.5)
                    current_alloc = self._dynamic_allocations.get(strategy, 0)

                    if win_rate >= 0.5 or perf.get("total_trades", 0) < 3:
                        adjustment = min(remaining_increase, current_alloc * self._utilization_adjustment_factor)
                        if adjustment > 0.001:
                            allocation_adjustments[strategy] = adjustment
                            remaining_increase -= adjustment

                for strategy, adjustment in allocation_adjustments.items():
                    old_alloc = self._dynamic_allocations.get(strategy, 0)
                    new_alloc = old_alloc + adjustment
                    total_alloc = sum(self._dynamic_allocations.values())

                    if total_alloc + adjustment <= 1.0:
                        self._dynamic_allocations[strategy] = new_alloc
                        logger.info(f"Idle cash allocation: {strategy} {old_alloc:.1%} -> {new_alloc:.1%} (+{adjustment:.1%})")

                self._allocation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "reason": "idle_cash_optimization",
                    "allocations": dict(self._dynamic_allocations)
                })

            # === 2. 应用信号质量阈值放松（引擎优先）===
            # === 3. 仓位乘数已在 _check_capital_utilization 由引擎更新 ===
            if report is not None:
                self._apply_engine_signal_relaxation(report.signal_relaxation)
                logger.info(
                    f"Engine idle cash optimization: utilization={utilization:.1%}, "
                    f"target={target:.0%}, action={report.recommended_action.value}, "
                    f"boost={report.position_boost:.2f}, relax={report.signal_relaxation:+.3f}"
                )
            else:
                severity = self._calculate_idle_severity(utilization)
                if severity > 0:
                    self._apply_signal_threshold_relaxation(severity)
                    self._apply_position_size_boost(severity, utilization)
                    logger.info(f"Idle cash optimization: utilization={utilization:.1%}, severity={severity:.2f}, "
                               f"adjusting allocations + relaxing thresholds + boosting position size")

        except Exception as e:
            logger.error(f"Error optimizing idle cash allocation: {e}")
    
    def _calculate_idle_severity(self, utilization: float) -> float:
        """计算空闲资金严重程度：0.0-1.0，越高越严重"""
        if utilization >= self._target_utilization:
            return 0.0
        if utilization >= self._min_utilization:
            # 介于最低利用率和目标之间，轻微调整
            return 0.3
        if utilization >= 0.1:
            return 0.6
        if utilization > 0:
            return 0.8
        return 1.0
    
    def _apply_signal_threshold_relaxation(self, severity: float):
        """根据严重程度降低信号质量阈值（受 locked_params 约束）"""
        try:
            relaxation = severity * 0.15  # 最多降低15%
            strategies_config = self.config.get("strategies", {})

            for strategy_name in ["grid", "trend", "scalping", "arbitrage"]:
                # locked_params 锁定的参数（如 min_signal_quality）不允许被放松下调
                if "min_signal_quality" in self._locked_params.get(strategy_name, set()):
                    continue

                # 以初始阈值为基准，避免阈值累积漂移
                base_quality = self._initial_signal_quality.get(strategy_name, 0.25)
                floor = self._signal_quality_floor.get(strategy_name, 0.10)
                relaxed_quality = max(floor, base_quality - relaxation)

                # 同步写入 config（order_executor 从 config 读取阈值）
                if strategy_name in strategies_config:
                    strategies_config[strategy_name]["min_signal_quality"] = relaxed_quality

                # 同时热更新策略实例
                if self._strategy_optimizer:
                    instance = self._strategy_optimizer._strategy_instances.get(strategy_name)
                    if instance and hasattr(instance, "apply_param_update"):
                        try:
                            instance.apply_param_update({"min_signal_quality": relaxed_quality})
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f"Signal threshold relaxation error: {e}")

    def _apply_engine_signal_relaxation(self, relaxation: float):
        """应用企业级引擎的信号阈值放松量（relaxation 可为负表示收紧）。

        以 _initial_signal_quality 为基准，避免阈值累积漂移。
        relaxation > 0 → 降低阈值（放松）；relaxation < 0 → 提高阈值（收紧）。
        """
        try:
            if abs(relaxation) < 1e-6:
                # 无放松需求，恢复到初始阈值
                self._restore_initial_signal_quality()
                return

            strategies_config = self.config.get("strategies", {})

            for strategy_name in ["grid", "trend", "scalping", "arbitrage"]:
                # locked_params 锁定的参数（如 min_signal_quality）不允许被放松下调
                if "min_signal_quality" in self._locked_params.get(strategy_name, set()):
                    continue

                base_quality = self._initial_signal_quality.get(strategy_name, 0.25)
                # relaxation > 0 → base - relaxation（降低）；relaxation < 0 → base + |relaxation|（提高）
                adjusted_quality = base_quality - relaxation
                floor = self._signal_quality_floor.get(strategy_name, 0.10)
                adjusted_quality = max(floor, min(0.80, adjusted_quality))

                if strategy_name in strategies_config:
                    strategies_config[strategy_name]["min_signal_quality"] = adjusted_quality

                if self._strategy_optimizer:
                    instance = self._strategy_optimizer._strategy_instances.get(strategy_name)
                    if instance and hasattr(instance, "apply_param_update"):
                        try:
                            instance.apply_param_update({"min_signal_quality": adjusted_quality})
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f"Engine signal relaxation error: {e}")

    def _restore_initial_signal_quality(self):
        """恢复到初始信号质量阈值"""
        try:
            strategies_config = self.config.get("strategies", {})
            for strategy_name in ["grid", "trend", "scalping", "arbitrage"]:
                initial_quality = self._initial_signal_quality.get(strategy_name, 0.25)
                if strategy_name in strategies_config:
                    strategies_config[strategy_name]["min_signal_quality"] = initial_quality
                if self._strategy_optimizer:
                    instance = self._strategy_optimizer._strategy_instances.get(strategy_name)
                    if instance and hasattr(instance, "apply_param_update"):
                        try:
                            instance.apply_param_update({"min_signal_quality": initial_quality})
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f"Restore initial signal quality error: {e}")
    
    def _apply_position_size_boost(self, severity: float, utilization: float):
        """利用率低时提高仓位乘数
        
        P24: 根据账户规模动态限制最大boost倍数，防止小账户过度放大仓位
        - < 50 USDT: max 1.5x
        - 50-100 USDT: max 2.0x
        - 100-500 USDT: max 3.0x
        - > 500 USDT: max 5.0x
        """
        try:
            # 获取当前账户权益
            equity = self._capital_utilization.get("equity", 0)
            if equity <= 0:
                try:
                    equity = self._capital_utilization.get("total_equity", 0)
                except Exception:
                    pass
            
            # P24: 根据账户规模限制最大boost
            if equity < 50:
                max_boost = 1.5
            elif equity < 100:
                max_boost = 2.0
            elif equity < 500:
                max_boost = 3.0
            else:
                max_boost = 5.0
            
            # 计算目标仓位乘数：利用率越低，乘数越大
            idle_ratio = 1.0 - utilization
            raw_boost = 1.0 + min(4.0, idle_ratio * severity * 5)
            boost_factor = min(raw_boost, max_boost)
            
            self._idle_cash_position_boost = boost_factor
            self._capital_utilization["position_boost"] = boost_factor
            
            if severity >= 0.3:
                logger.info(
                    f"Position size boost: x{boost_factor:.2f} (raw={raw_boost:.2f}, "
                    f"max={max_boost:.1f}, equity={equity:.1f}, utilization={utilization:.1%})"
                )
        except Exception as e:
            logger.debug(f"Position size boost error: {e}")
    
    def get_position_boost(self) -> float:
        """获取空闲资金仓位乘数（供scheduler调用）"""
        return getattr(self, '_idle_cash_position_boost', 1.0)

    def _cap_position_boost_by_risk_budget(self, boost: float) -> float:
        """给 position_boost 加 risk_budget 封顶，防止风险预算耗尽时仍放大仓位。

        以 risk_budget.max_position_boost 为硬上限（默认 2.0），
        同时兜底不小于 0（紧急冻结时引擎已返回 0.0）。
        """
        try:
            rb_cfg = self.config.get("risk_budget", {}) or {}
            if not rb_cfg.get("enabled", True):
                return boost
            max_boost = float(rb_cfg.get("max_position_boost", 2.0) or 2.0)
            return max(0.0, min(float(boost), max_boost))
        except Exception as e:
            logger.debug(f"Cap position boost error: {e}")
            return boost

    def _apply_allocation_shift(self, shifts: Dict[str, float]):
        """将资金利用率引擎的 allocation_shift 应用到 _dynamic_allocations。

        以 risk_budget.reallocation.max_shift_ratio 作为单步偏移上限，
        应用后做归一化与单策略分配区间封顶，避免权重漂移/极端集中。
        """
        if not shifts:
            return
        try:
            rb_cfg = self.config.get("risk_budget", {}) or {}
            max_shift = float(rb_cfg.get("reallocation", {}).get("max_shift_ratio", 0.15) or 0.15)
            min_alloc = 0.02
            max_alloc = 0.60

            old = dict(self._dynamic_allocations)
            for sname, shift in shifts.items():
                if sname not in self._dynamic_allocations:
                    continue
                if not self._is_strategy_enabled(sname):
                    # disabled 策略（如现货 spot_grid/spot_martingale）不占用资金权重
                    self._dynamic_allocations[sname] = 0.0
                    continue
                clamped = max(-max_shift, min(max_shift, float(shift)))
                self._dynamic_allocations[sname] = max(
                    min_alloc,
                    min(max_alloc, self._dynamic_allocations[sname] + clamped),
                )

            # 仅对启用策略归一化，disabled 权重保持 0
            enabled_names = [s for s in self._dynamic_allocations if self._is_strategy_enabled(s)]
            total = sum(self._dynamic_allocations[s] for s in enabled_names)
            if total > 0:
                for s in enabled_names:
                    self._dynamic_allocations[s] /= total

            if old != self._dynamic_allocations:
                logger.info(f"Allocation shift applied: {old} -> {self._dynamic_allocations}")
                self._allocation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "old": old,
                    "new": dict(self._dynamic_allocations),
                    "reason": "capital_utilization_allocation_shift",
                })
        except Exception as e:
            logger.debug(f"Apply allocation shift error: {e}")

    def _reset_idle_cash_optimizations(self):
        """利用率达标后恢复默认参数"""
        try:
            if hasattr(self, '_idle_cash_position_boost') and self._idle_cash_position_boost > 1.0:
                logger.info(f"Resetting idle cash optimizations (utilization target reached)")
                self._idle_cash_position_boost = 1.0
                self._capital_utilization["position_boost"] = 1.0

            strategies_config = self.config.get("strategies", {})
            for strategy_name in ["grid", "trend", "scalping", "arbitrage"]:
                # 恢复到初始阈值，而非当前被放松的值
                initial_quality = self._initial_signal_quality.get(strategy_name, 0.25)
                if strategy_name in strategies_config:
                    strategies_config[strategy_name]["min_signal_quality"] = initial_quality
                if self._strategy_optimizer:
                    instance = self._strategy_optimizer._strategy_instances.get(strategy_name)
                    if instance and hasattr(instance, "apply_param_update"):
                        try:
                            instance.apply_param_update({"min_signal_quality": initial_quality})
                        except Exception:
                            pass
        except Exception as e:
            logger.debug(f"Reset idle cash optimizations error: {e}")

    def _get_strategy_performances(self) -> Dict[str, Dict[str, Any]]:
        """获取各策略表现"""
        performances = {}
        
        try:
            records = self.sqlite_storage.get_trade_records(limit=500)
            strategy_trades = {}
            
            for rec in records:
                strategy = rec.get("strategy_name", rec.get("strategy", "unknown"))
                if strategy not in strategy_trades:
                    strategy_trades[strategy] = {"wins": 0, "losses": 0, "total_pnl": 0}
                
                pnl = rec.get("pnl", 0) or 0
                if pnl > 0:
                    strategy_trades[strategy]["wins"] += 1
                else:
                    strategy_trades[strategy]["losses"] += 1
                strategy_trades[strategy]["total_pnl"] += pnl
            
            for strategy, data in strategy_trades.items():
                total = data["wins"] + data["losses"]
                performances[strategy] = {
                    "win_rate": data["wins"] / total if total > 0 else 0.5,
                    "total_trades": total,
                    "total_pnl": data["total_pnl"]
                }
        except Exception as e:
            logger.error(f"Error getting strategy performances: {e}")
        
        return performances

    async def _trigger_utilization_alert(self, level: str, utilization: float):
        """触发资金利用率告警"""
        alert_id = f"utilization_{level}"
        message = f"Capital utilization {level}: {utilization:.1%}"
        severity = "warning" if level in ("high", "low") else "info"
        await self._alert(alert_id, message, severity)

    def get_capital_utilization(self) -> Dict[str, Any]:
        """获取资金利用率信息"""
        return dict(self._capital_utilization)

    # ── 企业级利用率引擎辅助方法 ──

    def _get_atr_ratio(self) -> float:
        """从 MarketRegimeEngine 提取波动率，映射为 ATR 比率。

        volatility score ∈ [-1, 1]，映射到 atr_ratio ∈ [0.3, 2.5]：
        - score=0（中性）→ atr_ratio=1.0
        - score=0.6（极端高波动）→ atr_ratio=1.6
        - score=-0.5（极低波动）→ atr_ratio=0.5
        """
        try:
            if self._regime_engine is not None:
                regime = self._regime_engine.get_regime()
                vol_score = float(regime.get("factor_scores", {}).get("volatility", 0) or 0)
                return max(0.3, min(2.5, 1.0 + vol_score))
        except Exception as e:
            logger.debug(f"Failed to get atr_ratio from regime engine: {e}")
        return 1.0

    def _get_recent_strategy_pnl(self) -> Dict[str, float]:
        """提取各策略近期累计盈亏"""
        try:
            performances = self._get_strategy_performances()
            return {s: float(p.get("total_pnl", 0) or 0) for s, p in performances.items()}
        except Exception as e:
            logger.debug(f"Failed to get recent strategy pnl: {e}")
            return {}

    def _get_total_drawdown_pct(self) -> float:
        """从 EquityMonitor 提取当前总回撤百分比"""
        try:
            if self._equity_monitor is not None:
                status = self._equity_monitor.get_equity_status()
                return float(status.get("max_drawdown_pct", 0) or 0)
        except Exception as e:
            logger.debug(f"Failed to get total drawdown pct: {e}")
        return 0.0

    def _get_dynamic_allocator(self):
        """懒加载全局单例 DynamicAllocator（三级资金池引擎）。"""
        try:
            from risk.dynamic_allocator import get_dynamic_allocator
            return get_dynamic_allocator(self.config)
        except Exception as e:
            logger.debug(f"Get dynamic allocator error: {e}")
            return None

    def _map_market_regime(self):
        """将 MarketRegimeEngine 的 regime 字符串映射为 DynamicAllocator.MarketRegime。"""
        try:
            from risk.dynamic_allocator import MarketRegime
            regime_str = "unknown"
            if self._regime_engine is not None:
                ro = self._regime_engine.get_regime()
                if isinstance(ro, dict):
                    regime_str = str(ro.get("regime", "unknown") or "unknown").lower()
            mapping = {
                "trending_up": MarketRegime.TRENDING_UP,
                "trending_down": MarketRegime.TRENDING_DOWN,
                "ranging": MarketRegime.RANGING,
                "high_volatility": MarketRegime.HIGH_VOLATILITY,
                "low_volatility": MarketRegime.LOW_VOLATILITY,
            }
            return mapping.get(regime_str, MarketRegime.UNKNOWN)
        except Exception:
            from risk.dynamic_allocator import MarketRegime
            return MarketRegime.UNKNOWN

    async def _sync_dynamic_allocator(self):
        """打通 DynamicAllocator 三级资金池 → _dynamic_allocations。

        调用 compute_allocation_plan() 得到各策略目标权重，
        平滑融合进 _dynamic_allocations 并归一化，让三级资金池分配结果真正驱动下单仓位。
        """
        try:
            total_equity = float(self._capital_utilization.get("total_equity", 0) or 0)
            if total_equity <= 0:
                return

            # 紧急模式：保持 _enhanced_rebalance_allocations 的归零结果，禁止重新放大
            if self._equity_monitor is not None:
                eq_params = self._equity_monitor.get_adaptive_params()
                if eq_params.get("mode", "normal") == "emergency":
                    return

            allocator = self._get_dynamic_allocator()
            if allocator is None:
                return

            strategy_names = list(self._dynamic_allocations.keys())
            strategy_perf = self._analyze_strategy_performance(days=3)
            strategy_metrics = {
                s: {
                    "win_rate": float(p.get("win_rate", 0.5) or 0.5),
                    "sharpe_ratio": float(p.get("sharpe_ratio", 0.0) or 0.0),
                    "profit_factor": float(p.get("profit_factor", 1.0) or 1.0),
                    "max_drawdown": float(p.get("max_drawdown", 0.0) or 0.0),
                    "trade_count": int(p.get("total_trades", 0) or 0),
                }
                for s, p in strategy_perf.items() if s in self._dynamic_allocations
            }

            regime = self._map_market_regime()

            plan = await allocator.compute_allocation_plan(
                total_capital=total_equity,
                total_equity=total_equity,
                strategy_names=strategy_names,
                strategy_metrics=strategy_metrics,
                market_regime=regime,
                current_weights=dict(self._dynamic_allocations),
            )

            old = dict(self._dynamic_allocations)
            new_weights = {}
            for sname in self._dynamic_allocations:
                if not self._is_strategy_enabled(sname):
                    # disabled 策略不占用资金权重，强制归零
                    new_weights[sname] = 0.0
                    continue
                sa = plan.strategy_allocations.get(sname)
                target = float(sa.target_weight) if sa is not None else self._dynamic_allocations.get(sname, 0.0)
                new_weights[sname] = old.get(sname, 0.0) * 0.7 + target * 0.3

            # 仅对启用策略归一化，disabled 权重保持 0
            enabled_names = [s for s in new_weights if self._is_strategy_enabled(s)]
            total = sum(new_weights[s] for s in enabled_names)
            if total > 0:
                for s in enabled_names:
                    new_weights[s] /= total
            else:
                new_weights = {
                    s: w for s, w in self._base_allocations.items()
                    if self._is_strategy_enabled(s)
                }

            self._dynamic_allocations = new_weights

            if old != self._dynamic_allocations:
                logger.info(f"DynamicAllocator synced: {old} -> {self._dynamic_allocations}")
                self._allocation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "old": old,
                    "new": dict(self._dynamic_allocations),
                    "reason": "dynamic_allocator_pools",
                    "method": "dynamic_allocator_3pool",
                })
        except Exception as e:
            logger.debug(f"Sync dynamic allocator error: {e}")

    def get_utilization_engine_summary(self) -> Dict[str, Any]:
        """获取企业级利用率引擎摘要（供 dashboard 使用）"""
        try:
            if self._utilization_engine is None:
                return {"available": False}
            return self._utilization_engine.get_utilization_summary()
        except Exception as e:
            logger.debug(f"Failed to get utilization engine summary: {e}")
            return {"available": False, "error": str(e)}

    def get_utilization_heatmap(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取利用率热力图数据"""
        try:
            if self._utilization_engine is None:
                return []
            return self._utilization_engine.get_utilization_heatmap(limit)
        except Exception as e:
            logger.debug(f"Failed to get utilization heatmap: {e}")
            return []

    def get_utilization_oscillation_status(self) -> Dict[str, Any]:
        """获取利用率振荡防抖状态"""
        try:
            if self._utilization_engine is None:
                return {"available": False}
            return self._utilization_engine.get_oscillation_status()
        except Exception as e:
            logger.debug(f"Failed to get oscillation status: {e}")
            return {"available": False, "error": str(e)}

    # ===================== 强化：风险平价资金分配 =====================

    def _calculate_risk_parity_allocation(self, strategy_perf: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """风险平价分配：让每个策略贡献相等的风险
        风险 = 波动率 * 仓位占比，目标是让各策略的风险贡献相等
        """
        risks = {}
        for strategy, perf in strategy_perf.items():
            max_dd = perf.get("max_drawdown", 0.05)
            avg_loss = perf.get("avg_loss", 0.01)
            win_rate = perf.get("win_rate", 0.5)
            total_trades = perf.get("total_trades", 0)

            if total_trades < 5:
                risks[strategy] = 0.02
                continue

            volatility = max_dd * 0.7 + avg_loss * win_rate * 0.3
            risks[strategy] = max(0.005, volatility)

        total_risk_budget = sum(1.0 / r for r in risks.values()) if risks else 1.0

        allocations = {}
        for strategy, risk in risks.items():
            allocations[strategy] = (1.0 / risk) / total_risk_budget if total_risk_budget > 0 else 0.2

        return allocations

    def _blend_allocation_methods(self,
                                   performance_alloc: Dict[str, float],
                                   risk_parity_alloc: Dict[str, float]) -> Dict[str, float]:
        """混合多种分配方法：性能40% + 风险平价30% + 基准30%"""
        blended = {}
        all_strategies = set(performance_alloc.keys()) | set(risk_parity_alloc.keys()) | set(self._base_allocations.keys())

        for strategy in all_strategies:
            perf_weight = performance_alloc.get(strategy, 0)
            rp_weight = risk_parity_alloc.get(strategy, 0)
            base_weight = self._base_allocations.get(strategy, 0)

            blended[strategy] = (
                perf_weight * 0.40 +
                rp_weight * 0.30 +
                base_weight * 0.30
            )

        total = sum(blended.values())
        if total > 0:
            for s in blended:
                blended[s] /= total

        return blended

    # ===================== P2-3: 增强凯利公式 =====================

    def _get_enhanced_kelly_factor(self, strategy_name: str) -> float:
        """P2-3: 获取增强凯利仓位因子（纯乘数，1.0=无变化）
        
        融合：贝叶斯先验胜率 + 分数凯利(0.25) + 波动率惩罚 + 回撤惩罚
        返回因子：0.3~2.0，默认1.0
        """
        try:
            perf = self._analyze_strategy_performance()
            strat_perf = perf.get(strategy_name, {})
            total_trades = strat_perf.get("total_trades", 0)
            win_rate = strat_perf.get("win_rate", 0.5)
            profit_factor = strat_perf.get("profit_factor", 1.0)
            max_dd = strat_perf.get("max_drawdown", 0.1)

            if total_trades < 5:
                return 1.0  # 数据不足，保持原始仓位

            # 贝叶斯后验胜率（先验Beta(15,15) → 50%胜率，等效30次交易经验）
            prior_alpha = 15.0
            prior_beta = 15.0
            wins = int(total_trades * win_rate)
            losses = total_trades - wins
            bayesian_win_rate = (prior_alpha + wins) / (prior_alpha + prior_beta + total_trades)

            # 赔率（profit_factor = 总盈利/总亏损）
            b = max(0.5, profit_factor)
            q = 1.0 - bayesian_win_rate

            # 标准凯利: f* = (p*b - q) / b
            kelly_fraction = (bayesian_win_rate * b - q) / b if b > 0 else 0
            kelly_fraction = max(0.0, min(1.0, kelly_fraction))

            # 分数凯利：使用25%以降低波动
            fractional_kelly = kelly_fraction * 0.25

            # 波动率惩罚：回撤越大，仓位越小
            vol_penalty = max(0.3, 1.0 - max_dd * 3)
            fractional_kelly *= vol_penalty

            if fractional_kelly <= 0:
                return 0.5  # 负凯利 → 最小仓位

            # 归一化到合理范围：以0.15为基准
            kelly_multiplier = fractional_kelly / 0.15
            kelly_multiplier = max(0.3, min(2.0, kelly_multiplier))

            logger.debug(
                f"P2-3: Enhanced Kelly for {strategy_name}: "
                f"bayes_wr={bayesian_win_rate:.3f}, pf={profit_factor:.2f}, "
                f"kelly_raw={kelly_fraction:.4f}, frac={fractional_kelly:.4f}, "
                f"vol_pen={vol_penalty:.2f}, factor={kelly_multiplier:.3f}"
            )

            return kelly_multiplier
        except Exception as e:
            logger.debug(f"Enhanced kelly factor calc error: {e}")
            return 1.0

    def _calculate_enhanced_kelly(self, strategy_name: str, base_size: float) -> float:
        """增强凯利公式：融合贝叶斯先验 + 分数凯利 + 波动率调整
        f* = (p*b - q) / b  (标准凯利)
        增强: 贝叶斯胜率 + 分数凯利(0.3-0.5) + 波动率惩罚
        
        NOTE: 此方法已由 _get_enhanced_kelly_factor 替代，保留用于向后兼容
        """
        return base_size * self._get_enhanced_kelly_factor(strategy_name)

    # ===================== 强化：动态杠杆调整 =====================

    def _calculate_dynamic_leverage(self) -> float:
        """动态杠杆：基于当前回撤、波动率、胜率自动调整杠杆
        - 回撤越深，杠杆越低
        - 波动率越高，杠杆越低
        - 胜率越高，杠杆越高
        """
        try:
            if not self._dynamic_leverage_enabled:
                return self._base_leverage

            stats = self.profit_optimizer.get_stats() if self.profit_optimizer else {}
            drawdown = stats.get("drawdown", 0)
            win_rate = stats.get("win_rate", 0.5)

            perf = self._analyze_strategy_performance()
            total_trades = sum(p.get("total_trades", 0) for p in perf.values())

            drawdown_factor = max(0.3, 1.0 - drawdown * 4)

            if total_trades >= 10:
                win_rate_factor = 0.7 + win_rate * 0.6
            else:
                win_rate_factor = 1.0

            vol_factor = 1.0
            if self._equity_history and len(self._equity_history) >= 10:
                equities = [e["equity"] for e in self._equity_history[-20:]]
                if len(equities) >= 5:
                    returns = np.diff(equities) / equities[:-1]
                    volatility = np.std(returns) if len(returns) > 1 else 0
                    if volatility > 0:
                        vol_factor = max(0.5, min(1.5, 0.02 / (volatility * 100 + 0.01)))

            dynamic_lev = self._base_leverage * drawdown_factor * win_rate_factor * vol_factor
            dynamic_lev = max(1.0, min(self._base_leverage * 1.5, dynamic_lev))

            return round(dynamic_lev, 1)
        except Exception as e:
            logger.debug(f"Dynamic leverage calc error: {e}")
            return self._base_leverage

    def get_dynamic_leverage(self) -> float:
        """获取当前动态杠杆倍数"""
        return self._calculate_dynamic_leverage()

    # ===================== 强化：波动率加权资金分配 =====================

    def _calculate_volatility_adjusted_allocations(self,
                                                    base_allocations: Dict[str, float]) -> Dict[str, float]:
        """波动率调整分配：高波动策略降低权重，低波动策略提高权重"""
        try:
            perf = self._analyze_strategy_performance()
            adjusted = dict(base_allocations)

            for strategy in base_allocations:
                strat_perf = perf.get(strategy, {})
                max_dd = strat_perf.get("max_drawdown", 0.05)
                total_trades = strat_perf.get("total_trades", 0)

                if total_trades < 10:
                    continue

                if max_dd > 0.15:
                    adjustment = max(0.6, 1.0 - (max_dd - 0.15) * 3)
                    adjusted[strategy] *= adjustment
                elif max_dd < 0.05:
                    adjustment = min(1.3, 1.0 + (0.05 - max_dd) * 4)
                    adjusted[strategy] *= adjustment

            total = sum(adjusted.values())
            if total > 0:
                for s in adjusted:
                    adjusted[s] /= total

            return adjusted
        except Exception as e:
            logger.debug(f"Volatility adjusted allocation error: {e}")
            return base_allocations

    # ===================== 强化：综合增强分配 =====================

    async def _enhanced_rebalance_allocations(self):
        """增强版再平衡：风险平价 + 波动率调整 + 市场状态"""
        try:
            strategy_perf = self._analyze_strategy_performance()
            if not strategy_perf:
                return

            # 禁用策略不参与再平衡（避免 trend/arbitrage 等被重新拉起权重）
            strategy_perf = {
                s: p for s, p in strategy_perf.items()
                if self._is_strategy_enabled(s)
            }
            if not strategy_perf:
                return

            performance_scores = self._calculate_performance_scores(strategy_perf)
            perf_total = sum(performance_scores.values())
            perf_alloc = {s: performance_scores[s] / perf_total if perf_total > 0 else 0.25
                         for s in performance_scores}

            rp_alloc = self._calculate_risk_parity_allocation(strategy_perf)

            blended = self._blend_allocation_methods(perf_alloc, rp_alloc)

            vol_adjusted = self._calculate_volatility_adjusted_allocations(blended)

            regime_scores = self._calculate_regime_scores()
            final_alloc = {}
            for s in vol_adjusted:
                regime_factor = regime_scores.get(s, 0.5)
                final_alloc[s] = vol_adjusted[s] * (0.7 + regime_factor * 0.6)

            final_total = sum(final_alloc.values())
            if final_total > 0:
                for s in final_alloc:
                    final_alloc[s] /= final_total

            old_alloc = dict(self._dynamic_allocations)
            smoothed = {}
            for s in final_alloc:
                old = old_alloc.get(s, self._base_allocations.get(s, 0.2))
                smoothed[s] = old * 0.7 + final_alloc[s] * 0.3

            smoothed_total = sum(smoothed.values())
            if smoothed_total > 0:
                for s in smoothed:
                    smoothed[s] /= smoothed_total

            # 禁用策略强制归零并保留 key（避免从 _dynamic_allocations 丢失结构）
            self._dynamic_allocations = {
                s: (smoothed.get(s, 0.0) if self._is_strategy_enabled(s) else 0.0)
                for s in self._base_allocations
            }

            # ── EquityMonitor 自适应修正：紧急/衰退模式下降低高风险策略权重 ──
            if self._equity_monitor is not None:
                eq_params = self._equity_monitor.get_adaptive_params()
                position_mult = eq_params.get("position_multiplier", 1.0)
                strategy_mod = eq_params.get("strategy_allocation_modifier", {})
                eq_mode = eq_params.get("mode", "normal")
                
                if eq_mode == "emergency":
                    # 紧急模式：所有策略权重归零
                    self._dynamic_allocations = {s: 0.0 for s in self._dynamic_allocations}
                    logger.warning("EquityMonitor EMERGENCY: all strategy allocations zeroed")
                elif eq_mode == "decline":
                    # 衰退模式：高风险策略（trend/scalping）降权
                    for s in self._dynamic_allocations:
                        if s in ("trend", "scalping"):
                            self._dynamic_allocations[s] *= max(0.3, position_mult)
                        else:
                            self._dynamic_allocations[s] *= max(0.5, position_mult)
                elif strategy_mod:
                    for s, mod in strategy_mod.items():
                        if s in self._dynamic_allocations:
                            self._dynamic_allocations[s] *= mod
                
                # 重新归一化
                alloc_total = sum(self._dynamic_allocations.values())
                if alloc_total > 0:
                    for s in self._dynamic_allocations:
                        self._dynamic_allocations[s] /= alloc_total

            if old_alloc != self._dynamic_allocations:
                logger.info(f"Enhanced rebalance completed: {old_alloc} -> {self._dynamic_allocations}")
                self._allocation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "old": old_alloc,
                    "new": dict(self._dynamic_allocations),
                    "reason": "enhanced_risk_parity",
                    "method": "performance+risk_parity+volatility+regime"
                })
        except Exception as e:
            logger.error(f"Enhanced rebalance error: {e}")
