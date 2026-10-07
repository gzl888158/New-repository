"""超短线剥头皮策略：基于 RSI/随机指标/MACD 等指标捕捉短期价格波动，快进快出并带移动止损。"""
import asyncio
import math
import time
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from core.models import Signal, TickData, Position
from configs.settings import get_currency_tier
from utils.helpers import calculate_take_profit, calculate_stop_loss, validate_tp_sl_prices, calculate_risk_reward_ratio, calculate_trailing_stop, calculate_scaled_take_profit, adjust_stop_loss_for_volatility, check_profit_protection, calculate_partial_close_quantity, detect_market_state, calculate_adaptive_position_size, adjust_signal_thresholds, evaluate_signal_quality, get_price_precision, map_market_state_to_regime
from utils.state_persistence import PersistentStrategy
from risk.dynamic_allocator import AdaptiveKelly

class ScalpingStrategy(PersistentStrategy):
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        super().__init__()
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache
        
        self._enabled = config["strategies"]["scalping"]["enabled"]
        self._run_hours_start = config["strategies"]["scalping"]["run_hours_start"]
        self._run_hours_end = config["strategies"]["scalping"]["run_hours_end"]
        # 非运行时段是否强制全平（默认 False：停止开新仓但保留现有仓位至自然退出）
        self._force_close_outside_hours = config["strategies"]["scalping"].get("force_close_outside_hours", False)
        self._min_drop = config["strategies"]["scalping"]["min_drop"]
        self._min_rise = config["strategies"]["scalping"]["min_rise"]
        self._price_deviation = config["strategies"]["scalping"]["price_deviation"]
        self._max_hold_minutes = config["strategies"]["scalping"]["max_hold_minutes"]
        self._profit_target_min = config["strategies"]["scalping"]["profit_target_min"]
        self._profit_target_max = config["strategies"]["scalping"]["profit_target_max"]
        self._stop_loss = config["strategies"]["scalping"]["stop_loss"]
        
        self._rsi_period = config["strategies"]["scalping"].get("rsi_period", 14)
        self._rsi_overbought = config["strategies"]["scalping"].get("rsi_overbought", 70)
        self._rsi_oversold = config["strategies"]["scalping"].get("rsi_oversold", 30)
        self._stoch_period = config["strategies"]["scalping"].get("stoch_period", 14)
        self._stoch_smooth = 3
        self._vwap_period = 60
        self._trailing_stop_activation = config["strategies"]["scalping"].get("trailing_stop_activation", 0.5)
        
        self._macd_fast = config["strategies"]["scalping"].get("macd_fast", 12)
        self._macd_slow = config["strategies"]["scalping"].get("macd_slow", 26)
        self._macd_signal = config["strategies"]["scalping"].get("macd_signal", 9)
        self._bb_period = config["strategies"]["scalping"].get("bb_period", 20)
        self._bb_std = config["strategies"]["scalping"].get("bb_std", 2.0)
        self._atr_period = config["strategies"]["scalping"].get("atr_period", 14)
        
        self._min_quantity_threshold = 0.0001
        self._volume_spike_factor = 3.0
        self._order_book_depth_check = 5
        self._volume_delta_threshold = config["strategies"]["scalping"].get("volume_delta_threshold", 0.3)
        
        self._aggressive_mode = config["strategies"]["scalping"].get("aggressive_mode", False)
        self._max_positions = config["strategies"]["scalping"].get("max_positions", 3)
        self._position_sizing_mode = config["strategies"]["scalping"].get("position_sizing_mode", "fixed")
        
        self._profit_taking_levels = config["strategies"]["scalping"].get("profit_taking_levels", [
            {"threshold": 0.003, "close_ratio": 0.3},
            {"threshold": 0.006, "close_ratio": 0.3},
            {"threshold": 0.01, "close_ratio": 0.4}
        ])
        self._trailing_stop_initial = config["strategies"]["scalping"].get("trailing_stop_initial", 0.015)
        self._trailing_stop_min = config["strategies"]["scalping"].get("trailing_stop_min", 0.003)
        self._volatility_adaptive_sl = config["strategies"]["scalping"].get("volatility_adaptive_sl", True)
        self._profit_protection_max_drawdown = config["strategies"]["scalping"].get("profit_protection_max_drawdown", 0.5)
        
        # P0: 硬性最大止损比例 - 防止ATR扩大时止损反向变宽导致巨亏（如AVAX单笔-12.7USDT）
        self._max_stop_loss_pct = config["strategies"]["scalping"].get("max_stop_loss_pct", 0.025)
        # P0: 单笔最大亏损USDT硬限制 - 基于总资金动态计算，保护账户安全
        self._max_single_trade_loss_usdt = config["strategies"]["scalping"].get("max_single_trade_loss_usdt", 0)
        # P32: 日内最大交易次数限制，防止过度交易
        self._max_daily_trades = config["strategies"]["scalping"].get("max_daily_trades", 20)
        self._daily_trade_count = 0
        self._daily_trade_reset_date = None

        # === 生产级 v2.0 参数 ===
        # 移动止损
        self._trailing_stop_enabled = config["strategies"]["scalping"].get("trailing_stop_enabled", True)
        self._trailing_stop_pct = config["strategies"]["scalping"].get("trailing_stop_pct", 0.018)
        # 保本止损
        self._breakeven_trigger_pct = config["strategies"]["scalping"].get("breakeven_trigger_pct", 0.012)
        self._breakeven_stop_pct = config["strategies"]["scalping"].get("breakeven_stop_pct", 0.003)
        # 时间止盈
        self._time_exit_enabled = config["strategies"]["scalping"].get("time_exit_enabled", True)
        self._max_hold_hours = config["strategies"]["scalping"].get("max_hold_hours", 72.0)
        self._time_exit_after_hours = config["strategies"]["scalping"].get("time_exit_after_hours", 48.0)
        self._time_exit_partial_pct = config["strategies"]["scalping"].get("time_exit_partial_pct", 0.5)
        # 波动率止盈
        self._volatility_stop_enabled = config["strategies"]["scalping"].get("volatility_stop_enabled", True)
        self._vol_spike_threshold = config["strategies"]["scalping"].get("vol_spike_threshold", 2.5)
        self._vol_stop_partial_pct = config["strategies"]["scalping"].get("vol_stop_partial_pct", 0.3)
        self._volatility_lockout_minutes = config["strategies"]["scalping"].get("volatility_lockout_minutes", 30)
        # 多级止盈
        self._take_profit_enabled = config["strategies"]["scalping"].get("take_profit_enabled", True)
        self._tp1_ratio = config["strategies"]["scalping"].get("tp1_ratio", 0.4)
        self._tp1_pct = config["strategies"]["scalping"].get("tp1_pct", 0.008)
        self._tp2_ratio = config["strategies"]["scalping"].get("tp2_ratio", 0.5)
        self._tp2_pct = config["strategies"]["scalping"].get("tp2_pct", 0.015)
        self._tp3_ratio = config["strategies"]["scalping"].get("tp3_ratio", 0.1)
        self._tp3_trailing_pct = config["strategies"]["scalping"].get("tp3_trailing_pct", 0.02)
        # 波动率尖峰后锁仓
        self._volatility_lockout_until: Dict[str, datetime] = {}
        
        self._adaptive_enabled = config["strategies"]["scalping"].get("adaptive_enabled", True)
        self._min_signal_quality = config["strategies"]["scalping"].get("min_signal_quality", 0.50)
        self._market_state_lookback = config["strategies"]["scalping"].get("market_state_lookback", 20)

        # 措施5: 高滑点币降权/禁用 — 滑点成本翻转治理（DOGE 实测滑点 0.32% vs tp1 0.5%）
        self._slippage_filter_enabled = config["strategies"]["scalping"].get("slippage_filter_enabled", False)
        self._max_slippage_ratio = config["strategies"]["scalping"].get("max_slippage_ratio", 0.5)
        self._high_slippage_downgrade_ratio = config["strategies"]["scalping"].get("high_slippage_downgrade_ratio", 0.3)
        self._high_slippage_downgrade_factor = config["strategies"]["scalping"].get("high_slippage_downgrade_factor", 0.5)
        self._symbol_slippage_overrides = config["strategies"]["scalping"].get("symbol_slippage_overrides", {})

        # 企业级增强：风险预算感知动态阈值（连续亏损/熔断时上浮信号质量阈值收紧开仓）
        self._risk_lock_quality_boost = config["strategies"]["scalping"].get("risk_lock_quality_boost", 0.10)
        # 企业级增强：过滤理由本地计数（get_health 暴露 + MetricsPipeline 埋点）
        self._filter_stats: Dict[str, int] = {}
        
        self._signal_types_enabled = config["strategies"]["scalping"].get("signal_types", ["momentum", "mean_reversion", "breakout", "range"])
        self._auto_select_signal_type = config["strategies"]["scalping"].get("auto_select_signal_type", True)
        self._signal_type_performance: Dict[str, Dict[str, Any]] = {
            "momentum": {"wins": 0, "losses": 0, "total_pnl": 0, "total_profit": 0, "total_loss": 0, "count": 0, "pnl_list": [], "consecutive_wins": 0, "consecutive_losses": 0, "peak_equity": 0, "max_drawdown": 0},
            "mean_reversion": {"wins": 0, "losses": 0, "total_pnl": 0, "total_profit": 0, "total_loss": 0, "count": 0, "pnl_list": [], "consecutive_wins": 0, "consecutive_losses": 0, "peak_equity": 0, "max_drawdown": 0},
            "breakout": {"wins": 0, "losses": 0, "total_pnl": 0, "total_profit": 0, "total_loss": 0, "count": 0, "pnl_list": [], "consecutive_wins": 0, "consecutive_losses": 0, "peak_equity": 0, "max_drawdown": 0},
            "range": {"wins": 0, "losses": 0, "total_pnl": 0, "total_profit": 0, "total_loss": 0, "count": 0, "pnl_list": [], "consecutive_wins": 0, "consecutive_losses": 0, "peak_equity": 0, "max_drawdown": 0}
        }
        self._adaptive_kelly = AdaptiveKelly(config.get("adaptive_kelly", {}))
        self._position_signal_type: Dict[str, str] = {}
        # ADX 趋势确认：每 symbol 的 DX 历史缓存，用于 Wilder 平滑（修复原 adx=dx 单根K线噪声）
        self._dx_history: Dict[str, List[float]] = {}
        
        self._mean_reversion_rsi_oversold = config["strategies"]["scalping"].get("mr_rsi_oversold", 30)
        self._mean_reversion_rsi_overbought = config["strategies"]["scalping"].get("mr_rsi_overbought", 70)
        self._mean_reversion_bb_threshold = config["strategies"]["scalping"].get("mr_bb_threshold", 0.15)
        self._range_scalp_threshold = config["strategies"]["scalping"].get("range_scalp_threshold", 0.7)
        
        self._base_thresholds = {
            "rsi_overbought": self._rsi_overbought,
            "rsi_oversold": self._rsi_oversold,
            "volume_delta": self._volume_delta_threshold,
            "price_deviation": self._price_deviation,
            "profit_target_min": self._profit_target_min,
            "stop_loss": self._stop_loss
        }
        
        self._market_state: Dict[str, Dict[str, Any]] = {}
        self._adjusted_thresholds: Dict[str, Dict[str, float]] = {}
        
        self._is_active = False
        self._positions: Dict[str, Dict[str, Any]] = {}
        # ghost_close 专项 Phase 2: 待确认开仓意图（成交回执驱动仓位写入）
        self._pending_entries: Dict[str, Dict[str, Any]] = {}
        self._use_fill_driven_position = config["strategies"]["scalping"].get("use_fill_driven_position", False)
        # ghost_close 专项 Phase 4: 观测指标
        self._fill_callback_hits = 0
        self._pending_timeout_cleaned = 0
        self._last_klines: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        self._vwap_cache: Dict[str, float] = {}
        self._momentum_cache: Dict[str, Dict[str, float]] = {}
        
        self._signal_cooldown: Dict[str, datetime] = {}
        self._breakout_levels: Dict[str, Dict[str, float]] = {}

        # P2: 全局信号频率控制 - 记录最近信号时间戳
        self._recent_signals_list: deque = deque(maxlen=50)

        # 缓存清理周期（秒）
        self._cache_cleanup_interval = 300  # 每5分钟清理一次过期缓存
        self._last_cache_cleanup = 0.0
        
        self._scalping_symbols = []
        # 小账户(<2000 USDT)下 tier1(BTC/ETH) 最小下单保证金过高(约154 USDT)远超单仓额度，
        # 剥头皮这类频繁小单策略跳过 tier1，聚焦 tier2/tier3 低价币（阈值与 signal_processor 一致）
        _total_capital = float(config.get("trading", {}).get("total_capital", 0) or 0)
        _btc_min_equity = float(config.get("trading", {}).get("high_value_equity_threshold", 2000.0) or 2000.0)
        _skip_high_value = _total_capital > 0 and _total_capital < _btc_min_equity
        for tier in ["tier1", "tier2", "tier3"]:
            if _skip_high_value and tier == "tier1":
                continue
            for base in config["currencies"][f"{tier}_symbols"]:
                # 剥头皮策略使用合约格式
                self._scalping_symbols.append(f"{base}-USDT-SWAP")
        
        self._ticker_cache: Dict[str, Dict[str, float]] = {}
        self._ticker_cache_time: Dict[str, datetime] = {}
        self._position_verify_cache: Dict[str, tuple] = {}  # {symbol:direction: (timestamp, result)}
        self._kline_update_interval = 60
        self._last_kline_update = datetime.now() - timedelta(seconds=60)
        self._indicators_dirty = True  # 标记指标是否需要重新计算（K线更新后置True）
        
        self._daily_pnl = 0.0
        self._daily_start_equity = 0.0
        self._max_daily_equity = 0.0
        self._current_drawdown = 0.0
        self._daily_reset_date = None
        self._equity_initialized = False

        self._adaptive_controller = None  # 动态分配控制器（由scheduler注入）
        self._stop_loss_manager = None    # 统一止损管理器（由scheduler注入）
        self._regime_engine = None        # MarketRegimeEngine（由StrategyFactory注入，供统一 ADX/DI 趋势明细复用）
        self.sqlite_storage = None        # 数据库存储（由scheduler注入，用于孤儿持仓恢复与对账）
        self._orphan_recovery_last_ts = 0.0  # 孤儿持仓恢复节流时间戳

    def set_adaptive_controller(self, controller):
        """注入AdaptiveController实例，用于获取动态资金分配"""
        self._adaptive_controller = controller

    def set_stop_loss_manager(self, manager):
        """注入StopLossManager实例，用于统一止损管理"""
        self._stop_loss_manager = manager

    def set_sqlite_storage(self, storage):
        """注入SQLiteStorage实例，用于孤儿持仓恢复与对账"""
        self.sqlite_storage = storage

    def set_coordinator(self, coordinator):
        """注入StrategyCoordinator实例"""
        self._coordinator = coordinator

    def set_regime_engine(self, engine):
        """注入MarketRegimeEngine，供内部评分复用统一 ADX/DI 趋势明细（消除口径漂移）。"""
        self._regime_engine = engine

    def _get_unified_trend_detail(self, symbol: str) -> Optional[Dict[str, Any]]:
        """读取 MarketRegimeEngine 的统一 ADX/DI 趋势明细（get_symbol_trend_detail）。

        与框架层 TrendConfirmationFilter 同源，供 scalping 内部评分复用，避免与策略内部
        自算 ADX 口径漂移。引擎未注入/币种未被监控/数据不足时返回 None（调用方回退）。
        """
        engine = getattr(self, "_regime_engine", None)
        if engine is None:
            return None
        try:
            return engine.get_symbol_trend_detail(symbol)
        except Exception:
            return None

    def _trend_alignment_adjust(self, symbol: str, side: str) -> float:
        """返回趋势对齐的 confidence 修正值（±0.10 * adx_strength）。

        读取 MarketRegimeEngine 的统一 ADX/DI 趋势明细（get_symbol_trend_detail），与框架层
        TrendConfirmationFilter 同源：顺势（side 与 direction 同向）加分，逆势减分，
        方向不明（direction==0）或无明细时返回 0.0 不干预。引擎未注入/币种未被监控/数据不足
        时返回 0.0（回退为不调整）。
        """
        detail = self._get_unified_trend_detail(symbol)
        if not detail:
            return 0.0
        try:
            direction = float(detail.get("direction", 0.0))
            adx_strength = float(detail.get("adx_strength", 0.0))
            if not math.isfinite(direction) or not math.isfinite(adx_strength):
                return 0.0
            if direction == 0.0:
                return 0.0
            aligned = (side == "buy" and direction > 0) or (side == "sell" and direction < 0)
            opposing = (side == "buy" and direction < 0) or (side == "sell" and direction > 0)
            weight = max(0.0, min(1.0, adx_strength))
            if aligned:
                return 0.10 * weight
            if opposing:
                return -0.10 * weight
        except (TypeError, ValueError):
            return 0.0
        return 0.0

    async def update_config(self, updates: Dict[str, Any]):
        """运行时热更新策略配置（不重启策略）

        支持的配置键：max_positions, max_hold_minutes, min_signal_quality,
        max_stop_loss_pct, max_single_trade_loss_usdt, leverage,
        position_sizing_mode 等
        """
        strategy_cfg = self.config.get("strategies", {}).get("scalping", {})
        strategy_cfg.update(updates)
        self.config.setdefault("strategies", {})["scalping"] = strategy_cfg

        attr_map = {
            "max_positions": "_max_positions",
            "max_hold_minutes": "_max_hold_minutes",
            "min_signal_quality": "_min_signal_quality",
            "max_stop_loss_pct": "_max_stop_loss_pct",
            "max_single_trade_loss_usdt": "_max_single_trade_loss_usdt",
            "leverage": "_leverage",
            "position_sizing_mode": "_position_sizing_mode",
            "stop_loss": "_stop_loss",
            "trailing_stop_activation": "_trailing_stop_activation",
            "trailing_stop_initial": "_trailing_stop_initial",
            "trailing_stop_min": "_trailing_stop_min",
            "rsi_overbought": "_rsi_overbought",
            "rsi_oversold": "_rsi_oversold",
            "take_profit_enabled": "_take_profit_enabled",
            "tp1_ratio": "_tp1_ratio", "tp2_ratio": "_tp2_ratio", "tp3_ratio": "_tp3_ratio",
            "tp1_pct": "_tp1_pct", "tp2_pct": "_tp2_pct", "tp3_trailing_pct": "_tp3_trailing_pct",
            "volatility_stop_enabled": "_volatility_stop_enabled",
            "vol_spike_threshold": "_vol_spike_threshold",
            "vol_stop_partial_pct": "_vol_stop_partial_pct",
            "volatility_lockout_minutes": "_volatility_lockout_minutes",
            "trailing_stop_enabled": "_trailing_stop_enabled",
            "trailing_stop_pct": "_trailing_stop_pct",
            "breakeven_trigger_pct": "_breakeven_trigger_pct",
            "breakeven_stop_pct": "_breakeven_stop_pct",
            "time_exit_enabled": "_time_exit_enabled",
            "max_hold_hours": "_max_hold_hours",
            "time_exit_after_hours": "_time_exit_after_hours",
            "time_exit_partial_pct": "_time_exit_partial_pct",
        }
        for cfg_key, attr_name in attr_map.items():
            if cfg_key in updates:
                setattr(self, attr_name, updates[cfg_key])
                logger.info(f"Scalping config hot-updated: {attr_name}={updates[cfg_key]}")

    def _dynamic_min_quality(self) -> float:
        """自适应信号质量阈值：连续亏损/风险熔断时上浮，收紧开仓。"""
        quality = self._min_signal_quality
        if not self._adaptive_enabled or self._adaptive_controller is None:
            return quality
        try:
            status = self._adaptive_controller.get_risk_budget_status()
            if status.get("streak_lock_active"):
                quality += self._risk_lock_quality_boost
        except Exception:
            # 风险锁查询失败时保守收紧阈值（fail-closed），避免在风控状态未知时放行
            logger.warning("Scalping risk budget status query failed; tightening min quality")
            quality += self._risk_lock_quality_boost
        return quality

    def _record_filter(self, symbol: str, reason: str):
        """记录一次过滤/拒绝原因（本地计数 + MetricsPipeline 埋点）。"""
        self._filter_stats[reason] = self._filter_stats.get(reason, 0) + 1
        try:
            self._increment_metric("scalping_filter_total", 1.0,
                                   {"reason": reason, "symbol": symbol})
        except Exception:
            pass
        logger.debug(f"Scalping filter: {symbol} {reason}")

    def _get_allocation(self) -> float:
        """获取当前资金分配比例：优先使用AdaptiveController动态分配，回退到config"""
        if self._adaptive_controller:
            try:
                return self._adaptive_controller.get_allocation("scalping")
            except Exception as e:
                logger.debug(f"[scalping] get_allocation failed, fallback to config: {e}")
        return self.config["trading"].get("scalping_allocation", 0.20)

    def _get_effective_capital(self) -> float:
        """获取有效资金：优先使用实际账户权益，回退到配置中的total_capital"""
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                details = account_info.get("details", [])
                for detail in details:
                    if detail.get("ccy") == "USDT":
                        eq = float(detail.get("eq", 0))
                        if eq > 0:
                            return eq
                total_eq = float(account_info.get("totalEq", 0))
                if total_eq > 0:
                    return total_eq
        except Exception as e:
            logger.warning(f"[scalping] get_account_info failed, fallback to config total_capital: {e}")
        return self.config["trading"].get("total_capital", 100.0)

    def apply_param_update(self, params: Dict[str, Any]):
        """热更新策略参数（由StrategyOptimizer调用，无需重启）
        通用方法：遍历params，更新self.config和同名属性（若存在）
        """
        applied = []
        for key, value in params.items():
            # 更新config
            if key in self.config["strategies"].get("scalping", {}):
                self.config["strategies"]["scalping"][key] = value
                # 同名属性（前缀_）
                attr_name = f"_{key}"
                if hasattr(self, attr_name):
                    setattr(self, attr_name, value)
                applied.append(key)
        if applied:
            logger.info(f"Scalping strategy params hot-updated: {applied}")
        return applied

    async def start(self):
        if not self._enabled:
            logger.info("Scalping strategy is disabled")
            return

        # P0-5: 初始化状态持久化（Redis 优先，JSON 兜底）
        self.init_state_persistence("scalping", self.redis_cache)
        # 恢复上次运行的状态（持仓、日内 PnL、回撤等）
        await self.load_state_async()
        # 启动周期性状态保存协程
        asyncio.create_task(self.periodic_save_loop())

        # P0-2: 启动时初始化日内权益基准（从 OKX 拉取实际权益）
        await self._init_daily_equity()

        logger.info("Starting Scalping Strategy")
        asyncio.create_task(self._schedule_loop())
        asyncio.create_task(self._monitor_loop())

    async def _schedule_loop(self):
        try:
            while True:
                now = datetime.now()
                hour = now.hour

                # 支持跨午夜时段（如 22:00-06:00）：start > end 时为跨午夜逻辑
                if self._run_hours_start <= self._run_hours_end:
                    is_within_hours = self._run_hours_start <= hour < self._run_hours_end
                else:
                    is_within_hours = hour >= self._run_hours_start or hour < self._run_hours_end

                if is_within_hours and not self._is_active:
                    logger.info("Scalping strategy activated")
                    self._is_active = True
                elif not is_within_hours and self._is_active:
                    logger.info("Scalping strategy deactivated (outside run hours)")
                    self._is_active = False
                    if self._force_close_outside_hours:
                        logger.info("force_close_outside_hours=true: closing all positions")
                        await self._close_all_positions()
                    else:
                        logger.info("force_close_outside_hours=false: positions left to natural exit (SL/TP)")

                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            logger.info("Scalping _schedule_loop cancelled")
            self._is_active = False
            # 取消时不关闭仓位，由外部控制

    async def _get_ticker_cached(self, symbol: str):
        now = datetime.now()
        if symbol in self._ticker_cache and (now - self._ticker_cache_time[symbol]).total_seconds() < 2:
            return self._ticker_cache[symbol]
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if ticker:
            self._ticker_cache[symbol] = ticker
            self._ticker_cache_time[symbol] = now
        
        return ticker
    
    async def _verify_exchange_position(self, symbol: str, direction: str) -> bool:
        """验证交易所是否实际持有该方向的持仓，用于防止幽灵仓位
        
        使用缓存机制，避免频繁调用get_positions API
        """
        now = time.time()
        cache_key = f"{symbol}:{direction}"
        if cache_key in self._position_verify_cache:
            cached_time, cached_result = self._position_verify_cache[cache_key]
            if now - cached_time < 1.5:  # 1.5秒缓存（缩短TTL防止幽灵仓位竞态）
                return cached_result
        
        try:
            positions = await self.okx_client.get_positions_async()
            found = False
            for pos_data in positions:
                pos = self.okx_client._parse_position(pos_data)
                if pos and pos.symbol == symbol and pos.side == direction and abs(pos.quantity) > 0:
                    found = True
                    break
            self._position_verify_cache[cache_key] = (now, found)
            return found
        except Exception as e:
            logger.error(f"Failed to verify exchange position for {symbol}: {e}")
            # 查询失败时保守处理：返回False，不冒51169风险
            return False
    
    def _invalidate_position_cache(self, symbol: str, direction: str = None):
        """平仓后立即失效缓存，防止幽灵仓位"""
        if direction:
            self._position_verify_cache.pop(f"{symbol}:{direction}", None)
        else:
            # 清除该symbol所有方向的缓存
            keys_to_remove = [k for k in self._position_verify_cache if k.startswith(f"{symbol}:")]
            for k in keys_to_remove:
                self._position_verify_cache.pop(k, None)
    
    async def verify_exchange_positions(self) -> Dict[str, Any]:
        """P22: 状态无漂移 - 校验本地self._positions与交易所真实仓位一致性
        
        每tick/每K线前由StrategyContainer调用，确保策略不依赖内存变量记仓位。
        检测到漂移时自动清理本地幽灵仓位记录。
        """
        corrections = []
        try:
            # 获取交易所真实仓位
            positions = None
            position_provider = getattr(self, "_position_provider", None)
            if position_provider:
                positions = position_provider()
            if not positions:
                positions = await self.okx_client.get_positions_async()
            
            if not positions:
                positions = []
            
            # 构建交易所仓位映射 {symbol: {direction: quantity}}
            exchange_positions: Dict[str, Dict[str, float]] = {}
            for pos_data in positions:
                try:
                    pos = self.okx_client._parse_position(pos_data)
                    if pos and abs(pos.quantity) > 0:
                        exchange_positions.setdefault(pos.symbol, {})[pos.side] = abs(pos.quantity)
                except Exception:
                    continue
            
            # 校验本地self._positions中标记为"open"的仓位
            for symbol, local_pos in list(self._positions.items()):
                if local_pos.get("status") != "open":
                    continue
                
                direction = local_pos.get("direction", "")
                exchange_qty = exchange_positions.get(symbol, {}).get(direction, 0)
                
                if exchange_qty <= 0:
                    # 交易所不存在该仓位 → 幽灵仓位，清理
                    logger.warning(
                        f"P22: Ghost position detected in scalping: {symbol} {direction}, "
                        f"local status=open but exchange has no position. Cleaning up."
                    )
                    self._positions[symbol]["status"] = "closed"
                    self._positions[symbol]["close_reason"] = "ghost_position_cleaned"
                    self._positions[symbol]["pnl"] = local_pos.get("pnl", 0)
                    self._invalidate_position_cache(symbol, direction)
                    corrections.append({
                        "symbol": symbol,
                        "action": "ghost_cleaned",
                        "direction": direction,
                        "local_qty": local_pos.get("quantity", 0),
                    })
            
            # 清理超时未更新的"open"仓位（超过24小时无活动的视为幽灵）
            stale_threshold = time.time() - 86400
            for symbol, local_pos in list(self._positions.items()):
                if local_pos.get("status") == "open":
                    last_update = local_pos.get("last_update", 0)
                    if last_update < stale_threshold:
                        logger.warning(
                            f"P22: Stale position cleaned in scalping: {symbol}, "
                            f"last_update={last_update:.0f} ({time.time() - last_update:.0f}s ago)"
                        )
                        self._positions[symbol]["status"] = "closed"
                        self._positions[symbol]["close_reason"] = "stale_position_cleaned"
                        corrections.append({
                            "symbol": symbol,
                            "action": "stale_cleaned",
                            "age_seconds": time.time() - last_update,
                        })

            # ghost_close 专项 Phase 3: 成交回执丢失补偿 — 交易所已有仓位则反向物化 pending intent
            for symbol in list(self._pending_entries.keys()):
                entry = self._pending_entries[symbol]
                direction = entry.get("direction", "")
                exchange_qty = exchange_positions.get(symbol, {}).get(direction, 0)
                if exchange_qty > 0:
                    self._pending_entries.pop(symbol, None)
                    entry_price = entry.get("entry_price", 0)
                    self._positions[symbol] = {
                        "direction": direction,
                        "entry_price": entry_price,
                        "quantity": entry.get("quantity", exchange_qty),
                        "leverage": entry.get("leverage", 1),
                        "stop_loss": entry.get("stop_loss"),
                        "take_profit": entry.get("take_profit"),
                        "entry_time": datetime.now(),
                        "status": "open",
                        "peak_price": entry_price if direction == "long" else 0,
                        # 用 entry_price 作为 trough 初始值（有限值，避免 float('inf') 污染 JSON 持久化）；
                        # 多头 trough 作为 min 累加器，初始=入场价语义正确（最低价不会高于入场价）
                        "trough_price": entry_price,
                        "trailing_stop_initial": entry.get("stop_loss"),
                        "vwap_entry": entry.get("vwap_entry", entry_price),
                        "current_profit": 0.0,
                        "partial_closes": [],
                    }
                    logger.warning(
                        f"[ghost_close-P3] position materialized from exchange (fill callback lost): "
                        f"{symbol} {direction} qty={entry.get('quantity', exchange_qty)}"
                    )
                    corrections.append({
                        "symbol": symbol,
                        "action": "pending_compensated",
                        "direction": direction,
                    })

            # ghost_close 专项 Phase 2: 清理超时的 pending intent（限价单 TTL 后仍未成交）
            pending_ttl = self.config["strategies"]["scalping"].get("pending_ttl_seconds", 15)
            now_ts = time.time()
            for symbol in list(self._pending_entries.keys()):
                entry = self._pending_entries[symbol]
                if now_ts - entry.get("timestamp", 0) > pending_ttl:
                    logger.warning(
                        f"P22: Pending entry expired for {symbol} "
                        f"(age={now_ts - entry.get('timestamp', 0):.0f}s > {pending_ttl}s), dropping"
                    )
                    del self._pending_entries[symbol]
                    corrections.append({
                        "symbol": symbol,
                        "action": "pending_expired",
                    })
                    self._pending_timeout_cleaned += 1

            return {
                "drifted": len(corrections) > 0,
                "corrections": corrections,
                "exchange_positions": list(exchange_positions.keys()),
            }
        except Exception as e:
            logger.debug(f"P22: Scalping position verification error: {e}")
            return {"drifted": False, "corrections": [], "exchange_positions": []}
    
    async def _monitor_loop(self):
        try:
            while True:
                if not self._is_active:
                    await asyncio.sleep(10)
                    continue

                if not self._equity_initialized:
                    await asyncio.sleep(2)
                    continue

                self._check_daily_reset()

                # 定期清理过期缓存，防止内存泄漏
                now_ts = time.time()
                if now_ts - self._last_cache_cleanup > self._cache_cleanup_interval:
                    self._cleanup_stale_caches()

                now = datetime.now()
                if (now - self._last_kline_update).total_seconds() >= self._kline_update_interval:
                    await self._update_klines()
                    self._last_kline_update = now

                # CPU节流：仅在K线更新后重新计算指标，避免每0.5秒无效重算
                if self._indicators_dirty:
                    await self._calculate_indicators()
                    if self._adaptive_enabled:
                        await self._update_market_state()
                    self._indicators_dirty = False

                await self._scan_for_signals()
                await self._manage_positions()
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            logger.info("Scalping _monitor_loop cancelled")

    async def _update_klines(self):
        for symbol in self._scalping_symbols:
            if symbol not in self._last_klines:
                self._last_klines[symbol] = {}
            
            for timeframe in ["1m", "5m", "15m"]:
                klines = await self.okx_client.get_kline_async(symbol, timeframe, limit=120)
                if len(klines) >= 2:
                    self._last_klines[symbol][timeframe] = klines
        
        # 标记指标需要重新计算
        self._indicators_dirty = True

    async def _calculate_indicators(self):
        for symbol in self._scalping_symbols:
            klines_1m = self._last_klines.get(symbol, {}).get("1m", [])
            klines_5m = self._last_klines.get(symbol, {}).get("5m", [])
            klines_15m = self._last_klines.get(symbol, {}).get("15m", [])
            
            if len(klines_1m) >= max(self._rsi_period, self._macd_slow, self._bb_period, self._atr_period) + 1:
                closes_1m = np.array([float(k[4]) for k in klines_1m])
                highs_1m = np.array([float(k[2]) for k in klines_1m])
                lows_1m = np.array([float(k[3]) for k in klines_1m])
                volumes_1m = np.array([float(k[5]) for k in klines_1m])
                
                rsi_1m = self._calculate_rsi(closes_1m)
                stoch_1m = self._calculate_stochastic(closes_1m)
                vwap_1m = self._calculate_vwap(klines_1m)
                volume_delta = self._calculate_volume_delta(klines_1m)
                macd, macd_signal, macd_hist = self._calculate_macd(closes_1m)
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes_1m)
                atr_1m = self._calculate_atr(highs_1m, lows_1m, closes_1m)
                volume_spike = self._detect_volume_spike(volumes_1m)
                price_action_patterns = self._detect_price_action_patterns(klines_1m[:5])
                
                rsi_values = [self._calculate_rsi(closes_1m[i:i+self._rsi_period]) for i in range(len(closes_1m) - self._rsi_period + 1)]
                divergence = self._detect_rsi_divergence(klines_1m, rsi_values)
                trendline_break_long = self._detect_trendline_break(klines_1m, "long")
                trendline_break_short = self._detect_trendline_break(klines_1m, "short")
                volume_profile = self._check_volume_profile(klines_1m)
                
                self._momentum_cache[symbol] = {
                    "rsi_1m": rsi_1m,
                    "stoch_1m": stoch_1m,
                    "vwap_1m": vwap_1m,
                    "volume_delta": volume_delta,
                    "price_above_vwap": float(klines_1m[0][4]) > vwap_1m if klines_1m else False,
                    "macd": macd,
                    "macd_signal": macd_signal,
                    "macd_hist": macd_hist,
                    "bb_upper": bb_upper,
                    "bb_middle": bb_middle,
                    "bb_lower": bb_lower,
                    "atr_1m": atr_1m,
                    "volume_spike": volume_spike,
                    "price_action_patterns": price_action_patterns,
                    "divergence": divergence,
                    "trendline_break_long": trendline_break_long,
                    "trendline_break_short": trendline_break_short,
                    "support_zones": volume_profile["support_zones"],
                    "resistance_zones": volume_profile["resistance_zones"]
                }
            
            if len(klines_5m) >= max(self._rsi_period, self._macd_slow, self._bb_period) + 1:
                closes_5m = np.array([float(k[4]) for k in klines_5m])
                highs_5m = np.array([float(k[2]) for k in klines_5m])
                lows_5m = np.array([float(k[3]) for k in klines_5m])
                
                rsi_5m = self._calculate_rsi(closes_5m)
                macd_5m, macd_signal_5m, _ = self._calculate_macd(closes_5m)
                bb_upper_5m, bb_middle_5m, bb_lower_5m = self._calculate_bollinger_bands(closes_5m)
                atr_5m = self._calculate_atr(highs_5m, lows_5m, closes_5m)
                
                if symbol in self._momentum_cache:
                    self._momentum_cache[symbol]["rsi_5m"] = rsi_5m
                    self._momentum_cache[symbol]["macd_5m"] = macd_5m
                    self._momentum_cache[symbol]["macd_signal_5m"] = macd_signal_5m
                    self._momentum_cache[symbol]["bb_upper_5m"] = bb_upper_5m
                    self._momentum_cache[symbol]["bb_middle_5m"] = bb_middle_5m
                    self._momentum_cache[symbol]["bb_lower_5m"] = bb_lower_5m
                    self._momentum_cache[symbol]["atr_5m"] = atr_5m
            
            if len(klines_15m) >= self._rsi_period + 1:
                closes_15m = np.array([float(k[4]) for k in klines_15m])
                rsi_15m = self._calculate_rsi(closes_15m)
                
                if symbol in self._momentum_cache:
                    self._momentum_cache[symbol]["rsi_15m"] = rsi_15m

    async def _update_market_state(self):
        for symbol in self._scalping_symbols:
            klines_1m = self._last_klines.get(symbol, {}).get("1m", [])
            
            if len(klines_1m) >= self._market_state_lookback:
                prices = [float(k[4]) for k in klines_1m[-self._market_state_lookback:]]
                volumes = [float(k[5]) for k in klines_1m[-self._market_state_lookback:]]
                
                atr = self._get_atr(symbol)
                
                market_state = detect_market_state(prices, volumes, atr, self._market_state_lookback)
                self._market_state[symbol] = market_state
                
                adjusted_thresholds = adjust_signal_thresholds(self._base_thresholds, market_state)
                self._adjusted_thresholds[symbol] = adjusted_thresholds

    def _calculate_rsi(self, prices: np.ndarray) -> float:
        if len(prices) < self._rsi_period + 1:
            return 50
        
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return 50
        
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-self._rsi_period:])
        avg_loss = np.mean(losses[-self._rsi_period:])
        
        if np.isnan(avg_gain) or np.isinf(avg_gain):
            avg_gain = 0
        if np.isnan(avg_loss) or np.isinf(avg_loss):
            avg_loss = 0
        
        if avg_loss == 0:
            return 100
        if avg_gain == 0:
            return 0
        
        rs = avg_gain / avg_loss
        if np.isnan(rs) or np.isinf(rs):
            return 50
        
        rsi = 100 - (100 / (1 + rs))
        
        return max(0, min(100, rsi))

    def _calculate_stochastic(self, prices: np.ndarray) -> float:
        if len(prices) < self._stoch_period:
            return 50
        
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return 50
        
        recent_prices = prices[-self._stoch_period:]
        lowest_low = np.min(recent_prices)
        highest_high = np.max(recent_prices)
        
        if np.isnan(lowest_low) or np.isinf(lowest_low) or np.isnan(highest_high) or np.isinf(highest_high):
            return 50
        
        if highest_high == lowest_low:
            return 50
        
        k = ((prices[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        if np.isnan(k) or np.isinf(k):
            return 50
        
        return max(0, min(100, k))

    def _calculate_macd(self, prices: np.ndarray):
        if len(prices) < self._macd_slow:
            return 0, 0, 0
        
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return 0, 0, 0
        
        ema_fast = self._calculate_ema(prices, self._macd_fast)
        ema_slow = self._calculate_ema(prices, self._macd_slow)

        # EMA不足period返回NaN时，MACD降级为0
        if np.isnan(ema_fast) or np.isnan(ema_slow):
            return 0, 0, 0

        # P0: 修复MACD信号线计算 —— 需要历史MACD值序列来计算EMA
        # 使用滚动缓冲区累积MACD值，取最近 signal_period 个值计算EMA
        macd_value = ema_fast - ema_slow
        buffer_key = f"macd_{self._macd_signal}"
        if not hasattr(self, '_macd_buffer') or self._macd_buffer is None:
            self._macd_buffer = {}
        if buffer_key not in self._macd_buffer:
            self._macd_buffer[buffer_key] = []
        buf = self._macd_buffer[buffer_key]
        buf.append(macd_value)
        if len(buf) > self._macd_signal * 3:
            buf.pop(0)
        
        macd = macd_value
        if len(buf) >= self._macd_signal:
            recent = buf[-self._macd_signal:]
            macd_signal = self._calculate_ema(recent, self._macd_signal)
        else:
            macd_signal = macd_value  # 数据不足时信号线=MACD线，柱状图为0
        
        macd_hist = macd - macd_signal
        
        if np.isnan(macd) or np.isinf(macd):
            macd = 0
        if np.isnan(macd_signal) or np.isinf(macd_signal):
            macd_signal = 0
        if np.isnan(macd_hist) or np.isinf(macd_hist):
            macd_hist = 0
        
        return macd, macd_signal, macd_hist

    def _calculate_ema(self, prices, period: int) -> float:
        if np.isscalar(prices):
            return float(prices)
        
        if not isinstance(prices, np.ndarray):
            prices = np.array(prices)
        
        if len(prices) < period:
            # 不足period时返回NaN，让上层降级处理，避免ema_fast-ema_slow始终为0
            return float('nan')
        
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return float(np.nanmean(prices))
        
        alpha = 2.0 / (period + 1)
        ema = prices[-period]
        
        for price in prices[-period + 1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        if np.isnan(ema) or np.isinf(ema):
            return float(np.mean(prices[-period:]))
        
        return float(ema)

    def _calculate_bollinger_bands(self, prices: np.ndarray):
        if len(prices) < self._bb_period:
            return 0, 0, 0
        
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return 0, 0, 0
        
        middle = np.mean(prices[-self._bb_period:])
        std_dev = np.std(prices[-self._bb_period:])
        
        if np.isnan(middle) or np.isinf(middle):
            middle = 0
        if np.isnan(std_dev) or np.isinf(std_dev):
            std_dev = 0
        
        upper = middle + self._bb_std * std_dev
        lower = middle - self._bb_std * std_dev
        
        return upper, middle, lower

    def _get_atr(self, symbol: str) -> float:
        if symbol in self._momentum_cache:
            atr_5m = self._momentum_cache[symbol].get("atr_5m", 0)
            atr_1m = self._momentum_cache[symbol].get("atr_1m", 0)
            return max(atr_5m, atr_1m) if atr_5m > 0 else atr_1m
        return 0

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        if len(highs) < self._atr_period or len(lows) < self._atr_period or len(closes) < self._atr_period:
            return 0
        
        if np.any(np.isnan(highs)) or np.any(np.isinf(highs)) or np.any(np.isnan(lows)) or np.any(np.isinf(lows)) or np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
            return 0
        
        tr1 = highs[-self._atr_period:] - lows[-self._atr_period:]
        tr2 = np.abs(highs[-self._atr_period:] - np.roll(closes[-self._atr_period:], 1))
        tr3 = np.abs(lows[-self._atr_period:] - np.roll(closes[-self._atr_period:], 1))
        
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = np.mean(tr)
        
        if np.isnan(atr) or np.isinf(atr):
            return 0
        
        return atr

    def _detect_volume_spike(self, volumes: np.ndarray) -> bool:
        if len(volumes) < 20:
            return False
        
        if np.any(np.isnan(volumes)) or np.any(np.isinf(volumes)):
            return False
        
        recent_volume = volumes[-1]
        avg_volume = np.mean(volumes[-20:-1])
        
        if avg_volume == 0:
            return False
        
        return recent_volume > avg_volume * self._volume_spike_factor

    def _detect_price_action_patterns(self, klines: List) -> List[str]:
        patterns = []
        
        if len(klines) < 3:
            return patterns
        
        try:
            recent = [
                {
                    "open": float(k[1]),
                    "close": float(k[4]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "body": abs(float(k[4]) - float(k[1])),
                    "range": float(k[2]) - float(k[3])
                }
                for k in klines[:3]
            ]
            
            if len(recent) >= 3:
                prev_prev, prev, curr = recent
                
                if curr["body"] > prev["body"] * 1.5 and curr["body"] > prev_prev["body"] * 1.5:
                    if curr["close"] > curr["open"] and prev["close"] < prev["open"]:
                        patterns.append("bullish_engulfing")
                    elif curr["close"] < curr["open"] and prev["close"] > prev["open"]:
                        patterns.append("bearish_engulfing")
                
                if prev_prev["body"] > prev["body"] * 2 and curr["body"] > prev["body"] * 2:
                    if curr["close"] > curr["open"] and prev["close"] < prev["open"]:
                        patterns.append("morning_star")
                    elif curr["close"] < curr["open"] and prev["close"] > prev["open"]:
                        patterns.append("evening_star")
                
                if curr["range"] < prev["range"] * 0.5:
                    if curr["close"] > curr["open"]:
                        patterns.append("bullish_hammer")
                    else:
                        patterns.append("bearish_hanging_man")
                
                if curr["body"] < prev["body"] * 0.3:
                    if curr["close"] > curr["open"]:
                        patterns.append("bullish_doji")
                    else:
                        patterns.append("bearish_doji")
        except (ValueError, IndexError):
            pass
        
        return patterns

    def _detect_rsi_divergence(self, klines: List, rsi_values: List[float]) -> str:
        if len(klines) < 10 or len(rsi_values) < 10:
            return ""
        
        try:
            closes = np.array([float(k[4]) for k in klines[:10]])
            
            recent_low = np.min(closes[-5:])
            earlier_low = np.min(closes[-10:-5])
            
            recent_rsi = np.min(rsi_values[-5:])
            earlier_rsi = np.min(rsi_values[-10:-5])
            
            if recent_low < earlier_low and recent_rsi > earlier_rsi and recent_rsi < 40:
                return "bullish_divergence"
            
            recent_high = np.max(closes[-5:])
            earlier_high = np.max(closes[-10:-5])
            
            recent_rsi_high = np.max(rsi_values[-5:])
            earlier_rsi_high = np.max(rsi_values[-10:-5])
            
            if recent_high > earlier_high and recent_rsi_high < earlier_rsi_high and recent_rsi_high > 60:
                return "bearish_divergence"
        except (ValueError, IndexError):
            pass
        
        return ""

    def _detect_trendline_break(self, klines: List, direction: str) -> bool:
        if len(klines) < 15:
            return False
        
        try:
            closes = np.array([float(k[4]) for k in klines[:15]])
            
            if direction == "long":
                lows = np.array([float(k[3]) for k in klines[:15]])
                x = np.arange(len(lows))
                
                coef = np.polyfit(x, lows, 1)
                trendline = np.polyval(coef, x)
                
                recent_low = lows[-1]
                prev_low = lows[-2]
                
                if prev_low <= trendline[-2] * 0.999 and recent_low > trendline[-1] * 1.001:
                    return True
            else:
                highs = np.array([float(k[2]) for k in klines[:15]])
                x = np.arange(len(highs))
                
                coef = np.polyfit(x, highs, 1)
                trendline = np.polyval(coef, x)
                
                recent_high = highs[-1]
                prev_high = highs[-2]
                
                if prev_high >= trendline[-2] * 1.001 and recent_high < trendline[-1] * 0.999:
                    return True
        except (ValueError, IndexError):
            pass
        
        return False

    def _check_volume_profile(self, klines: List) -> Dict[str, Any]:
        if len(klines) < 30:
            return {"support_zones": [], "resistance_zones": []}
        
        try:
            volume_profile = {}
            
            for k in klines[:30]:
                close = round(float(k[4]), 2)
                volume = float(k[5])
                
                if close not in volume_profile:
                    volume_profile[close] = 0
                volume_profile[close] += volume
            
            sorted_prices = sorted(volume_profile.keys())
            total_volume = sum(volume_profile.values())
            # 零成交量时无法计算量价分布，直接返回空（避免除零）
            if total_volume <= 0:
                return {"support_zones": [], "resistance_zones": []}
            
            support_zones = []
            resistance_zones = []
            
            for i in range(1, len(sorted_prices) - 1):
                current_price = sorted_prices[i]
                current_volume = volume_profile[current_price]
                prev_volume = volume_profile[sorted_prices[i-1]]
                next_volume = volume_profile[sorted_prices[i+1]]
                
                if current_volume > prev_volume * 1.5 and current_volume > next_volume * 1.5:
                    if current_volume / total_volume > 0.02:
                        support_zones.append(current_price)
            
            for i in range(len(sorted_prices) - 1, 1, -1):
                current_price = sorted_prices[i]
                current_volume = volume_profile[current_price]
                
                if i > 0:
                    prev_volume = volume_profile[sorted_prices[i-1]]
                else:
                    prev_volume = 0
                if i < len(sorted_prices) - 1:
                    next_volume = volume_profile[sorted_prices[i+1]]
                else:
                    next_volume = 0
                
                if current_volume > prev_volume * 1.5 and current_volume > next_volume * 1.5:
                    if current_volume / total_volume > 0.02:
                        resistance_zones.append(current_price)
            
            return {
                "support_zones": sorted(support_zones),
                "resistance_zones": sorted(resistance_zones, reverse=True)
            }
        except (ValueError, IndexError):
            return {"support_zones": [], "resistance_zones": []}

    def _calculate_vwap(self, klines: List) -> float:
        if len(klines) < self._vwap_period:
            return float(klines[0][4]) if klines else 0
        
        total_pv = 0
        total_volume = 0
        
        for k in klines[:self._vwap_period]:
            try:
                close = float(k[4])
                volume = float(k[5])
                
                if np.isnan(close) or np.isinf(close) or np.isnan(volume) or np.isinf(volume) or volume < 0:
                    continue
                
                total_pv += close * volume
                total_volume += volume
            except (ValueError, IndexError):
                continue
        
        if total_volume <= 0:
            return float(klines[0][4]) if klines else 0
        
        vwap = total_pv / total_volume
        
        if np.isnan(vwap) or np.isinf(vwap):
            return float(klines[0][4]) if klines else 0
        
        return vwap

    def _calculate_volume_delta(self, klines: List) -> float:
        if len(klines) < 10:
            return 0
        
        recent_up = 0
        recent_down = 0
        
        for i in range(1, min(10, len(klines))):
            current_close = float(klines[i-1][4])
            prev_close = float(klines[i][4])
            volume = float(klines[i-1][5])
            
            if current_close > prev_close:
                recent_up += volume
            else:
                recent_down += volume
        
        if recent_up + recent_down == 0:
            return 0
        
        return (recent_up - recent_down) / (recent_up + recent_down)

    async def _scan_for_signals(self):
        open_positions_count = sum(1 for p in self._positions.values() if p["status"] == "open")
        if open_positions_count >= self._max_positions:
            self._record_filter("global", "max_positions_reached")
            return
        
        if not self._check_drawdown_limit():
            self._record_filter("global", "drawdown_limit")
            return
        
        if not self._check_daily_loss_limit():
            self._record_filter("global", "daily_loss_limit")
            return

        # P2: 全局信号密度控制 - 每分钟最多N个信号
        now = datetime.now()
        recent_count = sum(1 for s in self._recent_signals_list
                           if (now - s["timestamp"]).total_seconds() < 60)
        max_per_minute = self.config["strategies"]["scalping"].get("max_signals_per_minute", 4)
        if recent_count >= max_per_minute:
            self._record_filter("global", "signal_density")
            return
        
        for symbol in self._scalping_symbols:
            if symbol in self._positions and self._positions[symbol]["status"] == "open":
                continue
            
            if symbol in self._signal_cooldown:
                cooldown_remaining = (datetime.now() - self._signal_cooldown[symbol]).total_seconds()
                if cooldown_remaining < 60:
                    continue

            # 波动率尖峰锁仓检查：锁仓期内不新开仓
            lockout_until = self._volatility_lockout_until.get(symbol)
            if lockout_until and datetime.now() < lockout_until:
                continue
            
            momentum = self._momentum_cache.get(symbol)
            if not momentum:
                continue
            
            await self._update_breakout_levels(symbol)
            
            market_state = self._market_state.get(symbol, {"state": "range", "volatility": "normal"})
            
            active_signal_types = self._get_active_signal_types(market_state)
            
            if "breakout" in active_signal_types and self._aggressive_mode:
                await self._check_aggressive_long_signal(symbol, momentum)
                await self._check_aggressive_short_signal(symbol, momentum)
            
            if "momentum" in active_signal_types:
                if self._check_position_correlation(symbol, "long"):
                    await self._check_long_signal(symbol, momentum)
                if self._check_position_correlation(symbol, "short"):
                    await self._check_short_signal(symbol, momentum)
            
            if "mean_reversion" in active_signal_types:
                if self._check_position_correlation(symbol, "long"):
                    await self._check_mean_reversion_long(symbol, momentum)
                if self._check_position_correlation(symbol, "short"):
                    await self._check_mean_reversion_short(symbol, momentum)
            
            if "range" in active_signal_types:
                if self._check_position_correlation(symbol, "long"):
                    await self._check_range_long(symbol, momentum)
                if self._check_position_correlation(symbol, "short"):
                    await self._check_range_short(symbol, momentum)

    def _get_active_signal_types(self, market_state: Dict[str, Any]) -> List[str]:
        if not self._auto_select_signal_type:
            return self._signal_types_enabled
        
        state = market_state.get("state", "range")
        volatility = market_state.get("volatility", "normal")
        
        active_types = []
        
        if state in ["uptrend", "downtrend"]:
            active_types.append("momentum")
            if volatility == "high":
                active_types.append("breakout")
            active_types.append("mean_reversion")
        elif state == "range":
            active_types.append("range")
            active_types.append("mean_reversion")
            active_types.append("momentum")
        elif volatility == "high":
            active_types.append("breakout")
            active_types.append("momentum")
        elif volatility == "low":
            active_types.append("mean_reversion")
            active_types.append("range")
        else:
            active_types = ["momentum", "mean_reversion"]
        
        return [t for t in active_types if t in self._signal_types_enabled]

    def _get_best_signal_type(self) -> str:
        best_type = "momentum"
        best_score = -1
        
        for sig_type, perf in self._signal_type_performance.items():
            total = perf["wins"] + perf["losses"]
            if total < 3:
                continue
            win_rate = perf["wins"] / total
            total_profit = perf.get("total_profit", 0)
            total_loss = perf.get("total_loss", 0)
            profit_factor = total_profit / max(total_loss, 0.01)
            score = win_rate * 0.5 + min(profit_factor, 3) / 3 * 0.3 + min(total / 20, 1) * 0.2
            if score > best_score:
                best_score = score
                best_type = sig_type
        
        return best_type

    async def _check_mean_reversion_long(self, symbol: str, momentum: Dict[str, float]):
        rsi_1m = momentum.get("rsi_1m", 50)
        rsi_5m = momentum.get("rsi_5m", 50)
        bb_upper = momentum.get("bb_upper", 0)
        bb_middle = momentum.get("bb_middle", 0)
        bb_lower = momentum.get("bb_lower", 0)
        vwap_1m = momentum.get("vwap_1m", 0)
        stoch_1m = momentum.get("stoch_1m", 50)
        atr_1m = momentum.get("atr_1m", 0)
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        
        if bb_lower == 0 or bb_middle == 0:
            return
        
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        vwap_distance = (current_price - vwap_1m) / vwap_1m if vwap_1m > 0 else 0
        
        conditions = []
        
        if rsi_1m <= self._mean_reversion_rsi_oversold:
            conditions.append(("rsi_oversold", 20))
        elif rsi_1m <= self._mean_reversion_rsi_oversold + 10:
            conditions.append(("rsi_low", 10))
        
        if bb_position <= self._mean_reversion_bb_threshold:
            conditions.append(("bb_lower", 20))
        elif bb_position <= 0.3:
            conditions.append(("bb_low", 10))
        
        if vwap_distance < -0.005:
            conditions.append(("below_vwap", 10))
        elif vwap_distance < -0.002:
            conditions.append(("slightly_below_vwap", 5))
        
        if stoch_1m < 20:
            conditions.append(("stoch_oversold", 15))
        elif stoch_1m < 30:
            conditions.append(("stoch_low", 8))
        
        if atr_1m > 0 and current_price < bb_lower + atr_1m * 0.5:
            conditions.append(("extreme_low", 15))
        
        rsi_divergence = momentum.get("divergence", "")
        if rsi_divergence == "bullish_divergence":
            conditions.append(("bullish_divergence", 20))
        
        support_zones = momentum.get("support_zones", [])
        for zone in support_zones:
            if zone * 0.99 <= current_price <= zone * 1.01:
                conditions.append(("support_zone", 15))
                break
        
        weighted_score = sum(c[1] for c in conditions)
        max_score = 100.0
        
        if max_score == 0:
            return
        
        raw_confidence = min(0.9, weighted_score / max_score)
        
        threshold = 0.45
        market_state = self._market_state.get(symbol, {"state": "range"})
        if market_state.get("state") == "range":
            threshold = 0.40
        if market_state.get("volatility") == "low":
            threshold = 0.38
        
        if weighted_score >= max_score * threshold:
            # 归一化 confidence：mean_reversion 的「加权评分/100」天然偏低（典型 0.38~0.6），
            # 与智能决策引擎全局 0.43 阈值系统性错配。将「刚过 threshold」映射到 ~0.5、
            # 「满分」映射到 ~0.9，使置信度量纲与 momentum 等信号对齐。
            span = max(1e-9, 0.9 - threshold)
            confidence = 0.5 + (raw_confidence - threshold) / span * 0.4
            confidence += self._trend_alignment_adjust(symbol, "buy")
            confidence = min(0.95, max(0.35, confidence))
            await self._generate_signal_with_type(symbol, "long", current_price, confidence, "mean_reversion")

    async def _check_mean_reversion_short(self, symbol: str, momentum: Dict[str, float]):
        rsi_1m = momentum.get("rsi_1m", 50)
        rsi_5m = momentum.get("rsi_5m", 50)
        bb_upper = momentum.get("bb_upper", 0)
        bb_middle = momentum.get("bb_middle", 0)
        bb_lower = momentum.get("bb_lower", 0)
        vwap_1m = momentum.get("vwap_1m", 0)
        stoch_1m = momentum.get("stoch_1m", 50)
        atr_1m = momentum.get("atr_1m", 0)
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        
        if bb_upper == 0 or bb_middle == 0:
            return
        
        bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
        vwap_distance = (current_price - vwap_1m) / vwap_1m if vwap_1m > 0 else 0
        
        conditions = []
        
        if rsi_1m >= self._mean_reversion_rsi_overbought:
            conditions.append(("rsi_overbought", 20))
        elif rsi_1m >= self._mean_reversion_rsi_overbought - 10:
            conditions.append(("rsi_high", 10))
        
        if bb_position >= 1 - self._mean_reversion_bb_threshold:
            conditions.append(("bb_upper", 20))
        elif bb_position >= 0.7:
            conditions.append(("bb_high", 10))
        
        if vwap_distance > 0.005:
            conditions.append(("above_vwap", 10))
        elif vwap_distance > 0.002:
            conditions.append(("slightly_above_vwap", 5))
        
        if stoch_1m > 80:
            conditions.append(("stoch_overbought", 15))
        elif stoch_1m > 70:
            conditions.append(("stoch_high", 8))
        
        if atr_1m > 0 and current_price > bb_upper - atr_1m * 0.5:
            conditions.append(("extreme_high", 15))
        
        rsi_divergence = momentum.get("divergence", "")
        if rsi_divergence == "bearish_divergence":
            conditions.append(("bearish_divergence", 20))
        
        resistance_zones = momentum.get("resistance_zones", [])
        for zone in resistance_zones:
            if zone * 0.99 <= current_price <= zone * 1.01:
                conditions.append(("resistance_zone", 15))
                break
        
        weighted_score = sum(c[1] for c in conditions)
        max_score = 100.0
        
        if max_score == 0:
            return
        
        raw_confidence = min(0.9, weighted_score / max_score)
        
        threshold = 0.45
        market_state = self._market_state.get(symbol, {"state": "range"})
        if market_state.get("state") == "range":
            threshold = 0.40
        if market_state.get("volatility") == "low":
            threshold = 0.38
        
        if weighted_score >= max_score * threshold:
            # 归一化 confidence：mean_reversion 的「加权评分/100」天然偏低（典型 0.38~0.6），
            # 与智能决策引擎全局 0.43 阈值系统性错配。将「刚过 threshold」映射到 ~0.5、
            # 「满分」映射到 ~0.9，使置信度量纲与 momentum 等信号对齐。
            span = max(1e-9, 0.9 - threshold)
            confidence = 0.5 + (raw_confidence - threshold) / span * 0.4
            confidence += self._trend_alignment_adjust(symbol, "sell")
            confidence = min(0.95, max(0.35, confidence))
            await self._generate_signal_with_type(symbol, "short", current_price, confidence, "mean_reversion")

    async def _check_range_long(self, symbol: str, momentum: Dict[str, float]):
        bb_lower = momentum.get("bb_lower", 0)
        bb_upper = momentum.get("bb_upper", 0)
        bb_middle = momentum.get("bb_middle", 0)
        rsi_1m = momentum.get("rsi_1m", 50)
        stoch_1m = momentum.get("stoch_1m", 50)
        vwap_1m = momentum.get("vwap_1m", 0)
        volume_delta = momentum.get("volume_delta", 0)
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        breakout_levels = self._breakout_levels.get(symbol)
        
        if not breakout_levels or bb_lower == 0 or bb_upper == 0:
            return
        
        support = breakout_levels["support"]
        resistance = breakout_levels["resistance"]
        range_size = breakout_levels["range"]
        
        if range_size == 0:
            return
        
        price_in_range = support <= current_price <= resistance
        range_width = range_size / current_price if current_price > 0 else 0
        
        if not price_in_range or range_width > 0.03:
            return
        
        position_in_range = (current_price - support) / range_size if range_size > 0 else 0.5
        
        conditions = []
        
        if position_in_range < 0.25:
            conditions.append(("near_support", 25))
        elif position_in_range < 0.35:
            conditions.append(("lower_range", 15))
        
        if rsi_1m < 40:
            conditions.append(("rsi_low", 15))
        elif rsi_1m < 50:
            conditions.append(("rsi_neutral_low", 8))
        
        if stoch_1m < 30:
            conditions.append(("stoch_low", 15))
        elif stoch_1m < 50:
            conditions.append(("stoch_neutral_low", 8))
        
        if current_price > bb_middle and current_price < bb_middle * 1.002:
            conditions.append(("below_middle", 10))
        
        if volume_delta > 0.2:
            conditions.append(("volume_support", 10))
        
        support_zones = momentum.get("support_zones", [])
        for zone in support_zones:
            if zone * 0.995 <= current_price <= zone * 1.005:
                conditions.append(("volume_support_zone", 15))
                break
        
        weighted_score = sum([c[1] for c in conditions])
        max_score = 80.0
        
        if max_score == 0:
            return
        
        confidence = min(0.85, weighted_score / max_score)
        confidence += self._trend_alignment_adjust(symbol, "buy")
        confidence = min(0.95, max(0.0, confidence))
        
        threshold = self._range_scalp_threshold
        if weighted_score >= max_score * threshold:
            await self._generate_signal_with_type(symbol, "long", current_price, confidence, "range")

    async def _check_range_short(self, symbol: str, momentum: Dict[str, float]):
        bb_lower = momentum.get("bb_lower", 0)
        bb_upper = momentum.get("bb_upper", 0)
        bb_middle = momentum.get("bb_middle", 0)
        rsi_1m = momentum.get("rsi_1m", 50)
        stoch_1m = momentum.get("stoch_1m", 50)
        vwap_1m = momentum.get("vwap_1m", 0)
        volume_delta = momentum.get("volume_delta", 0)
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        breakout_levels = self._breakout_levels.get(symbol)
        
        if not breakout_levels or bb_lower == 0 or bb_upper == 0:
            return
        
        support = breakout_levels["support"]
        resistance = breakout_levels["resistance"]
        range_size = breakout_levels["range"]
        
        if range_size == 0:
            return
        
        price_in_range = support <= current_price <= resistance
        range_width = range_size / current_price if current_price > 0 else 0
        
        if not price_in_range or range_width > 0.03:
            return
        
        position_in_range = (current_price - support) / range_size if range_size > 0 else 0.5
        
        conditions = []
        
        if position_in_range > 0.75:
            conditions.append(("near_resistance", 25))
        elif position_in_range > 0.65:
            conditions.append(("upper_range", 15))
        
        if rsi_1m > 60:
            conditions.append(("rsi_high", 15))
        elif rsi_1m > 50:
            conditions.append(("rsi_neutral_high", 8))
        
        if stoch_1m > 70:
            conditions.append(("stoch_high", 15))
        elif stoch_1m > 50:
            conditions.append(("stoch_neutral_high", 8))
        
        if current_price > bb_middle and current_price < bb_middle * 1.002:
            conditions.append(("above_middle", 10))
        
        if volume_delta < -0.2:
            conditions.append(("volume_resistance", 10))
        
        resistance_zones = momentum.get("resistance_zones", [])
        for zone in resistance_zones:
            if zone * 0.995 <= current_price <= zone * 1.005:
                conditions.append(("volume_resistance_zone", 15))
                break
        
        weighted_score = sum([c[1] for c in conditions])
        max_score = 80.0
        
        if max_score == 0:
            return
        
        confidence = min(0.85, weighted_score / max_score)
        confidence += self._trend_alignment_adjust(symbol, "sell")
        confidence = min(0.95, max(0.0, confidence))
        
        threshold = self._range_scalp_threshold
        if weighted_score >= max_score * threshold:
            await self._generate_signal_with_type(symbol, "short", current_price, confidence, "range")

    async def _generate_signal_with_type(self, symbol: str, direction: str, price: float, confidence: float, signal_type: str):
        self._position_signal_type[symbol] = signal_type
        await self._generate_signal(symbol, direction, price, confidence)

    async def _update_breakout_levels(self, symbol: str):
        klines_1m = self._last_klines.get(symbol, {}).get("1m", [])
        if len(klines_1m) < 20:
            return
        
        highs = np.array([float(k[2]) for k in klines_1m[:20]])
        lows = np.array([float(k[3]) for k in klines_1m[:20]])
        
        self._breakout_levels[symbol] = {
            "resistance": np.max(highs),
            "support": np.min(lows),
            "range": np.max(highs) - np.min(lows)
        }

    async def _check_long_signal(self, symbol: str, momentum: Dict[str, float]):
        rsi_1m = momentum.get("rsi_1m", 50)
        rsi_5m = momentum.get("rsi_5m", 50)
        rsi_15m = momentum.get("rsi_15m", 50)
        stoch_1m = momentum.get("stoch_1m", 50)
        vwap_1m = momentum.get("vwap_1m", 0)
        volume_delta = momentum.get("volume_delta", 0)
        price_above_vwap = momentum.get("price_above_vwap", False)
        macd = momentum.get("macd", 0)
        macd_signal = momentum.get("macd_signal", 0)
        macd_hist = momentum.get("macd_hist", 0)
        macd_5m = momentum.get("macd_5m", 0)
        macd_signal_5m = momentum.get("macd_signal_5m", 0)
        bb_lower = momentum.get("bb_lower", 0)
        bb_middle = momentum.get("bb_middle", 0)
        atr_1m = momentum.get("atr_1m", 0)
        volume_spike = momentum.get("volume_spike", False)
        price_action_patterns = momentum.get("price_action_patterns", [])
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        
        weighted_score = 0.0
        max_score = 0.0
        
        if rsi_1m < self._rsi_oversold + 15:
            weighted_score += 15.0
            if rsi_1m < self._rsi_oversold + 5:
                weighted_score += 10.0
        max_score += 25.0
        
        if rsi_5m < 45:
            weighted_score += 10.0
        elif rsi_5m < 55:
            weighted_score += 5.0
        max_score += 10.0
        
        if rsi_15m < 50:
            weighted_score += 5.0
        max_score += 5.0
        
        if stoch_1m < 25:
            weighted_score += 10.0
            if stoch_1m < 15:
                weighted_score += 5.0
        max_score += 15.0
        
        if volume_delta > self._volume_delta_threshold:
            weighted_score += 10.0
            if volume_delta > 0.5:
                weighted_score += 5.0
        max_score += 15.0
        
        if volume_spike:
            weighted_score += 10.0
        max_score += 10.0
        
        if current_price > vwap_1m * 1.001:
            weighted_score += 10.0
        max_score += 10.0
        
        if macd > macd_signal and macd_hist > 0:
            weighted_score += 10.0
            if macd_5m > macd_signal_5m:
                weighted_score += 5.0
        max_score += 15.0
        
        if current_price >= bb_lower and current_price <= bb_middle:
            weighted_score += 10.0
        elif current_price < bb_lower:
            weighted_score += 5.0
        max_score += 10.0
        
        if "bullish_engulfing" in price_action_patterns:
            weighted_score += 15.0
        elif "morning_star" in price_action_patterns:
            weighted_score += 12.0
        elif "bullish_hammer" in price_action_patterns:
            weighted_score += 8.0
        max_score += 15.0
        
        if await self._check_momentum_reversal(symbol, "long"):
            weighted_score += 10.0
        max_score += 10.0
        
        if await self._check_order_book_strength(symbol, "long"):
            weighted_score += 10.0
        max_score += 10.0
        
        divergence = momentum.get("divergence", "")
        trendline_break_long = momentum.get("trendline_break_long", False)
        support_zones = momentum.get("support_zones", [])
        
        if divergence == "bullish_divergence":
            weighted_score += 20.0
        max_score += 20.0
        
        if trendline_break_long:
            weighted_score += 15.0
        max_score += 15.0
        
        for zone in support_zones:
            if zone * 0.995 <= current_price <= zone * 1.005:
                weighted_score += 10.0
                break
        max_score += 10.0
        
        confidence = min(0.95, weighted_score / max(max_score, 1e-9))
        
        threshold_ratio = 0.40
        if self._adaptive_enabled:
            market_state = self._market_state.get(symbol, {"state": "range", "volatility": "normal"})
            if market_state.get("volatility") == "low":
                threshold_ratio = 0.35
            elif market_state.get("volatility") == "high":
                threshold_ratio = 0.45
            if market_state.get("state") == "range":
                threshold_ratio -= 0.03
        
        if weighted_score >= max_score * threshold_ratio:
            await self._generate_signal(symbol, "long", current_price, confidence)

    async def _check_short_signal(self, symbol: str, momentum: Dict[str, float]):
        rsi_1m = momentum.get("rsi_1m", 50)
        rsi_5m = momentum.get("rsi_5m", 50)
        rsi_15m = momentum.get("rsi_15m", 50)
        stoch_1m = momentum.get("stoch_1m", 50)
        vwap_1m = momentum.get("vwap_1m", 0)
        volume_delta = momentum.get("volume_delta", 0)
        price_above_vwap = momentum.get("price_above_vwap", False)
        macd = momentum.get("macd", 0)
        macd_signal = momentum.get("macd_signal", 0)
        macd_hist = momentum.get("macd_hist", 0)
        macd_5m = momentum.get("macd_5m", 0)
        macd_signal_5m = momentum.get("macd_signal_5m", 0)
        bb_upper = momentum.get("bb_upper", 0)
        bb_middle = momentum.get("bb_middle", 0)
        atr_1m = momentum.get("atr_1m", 0)
        volume_spike = momentum.get("volume_spike", False)
        price_action_patterns = momentum.get("price_action_patterns", [])
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        
        weighted_score = 0.0
        max_score = 0.0
        
        if rsi_1m > self._rsi_overbought - 15:
            weighted_score += 15.0
            if rsi_1m > self._rsi_overbought - 5:
                weighted_score += 10.0
        max_score += 25.0
        
        if rsi_5m > 55:
            weighted_score += 10.0
        elif rsi_5m > 45:
            weighted_score += 5.0
        max_score += 10.0
        
        if rsi_15m > 50:
            weighted_score += 5.0
        max_score += 5.0
        
        if stoch_1m > 75:
            weighted_score += 10.0
            if stoch_1m > 85:
                weighted_score += 5.0
        max_score += 15.0
        
        if volume_delta < -self._volume_delta_threshold:
            weighted_score += 10.0
            if volume_delta < -0.5:
                weighted_score += 5.0
        max_score += 15.0
        
        if volume_spike:
            weighted_score += 10.0
        max_score += 10.0
        
        if current_price < vwap_1m * 0.999:
            weighted_score += 10.0
        max_score += 10.0
        
        if macd < macd_signal and macd_hist < 0:
            weighted_score += 10.0
            if macd_5m < macd_signal_5m:
                weighted_score += 5.0
        max_score += 15.0
        
        if current_price <= bb_upper and current_price >= bb_middle:
            weighted_score += 10.0
        elif current_price > bb_upper:
            weighted_score += 5.0
        max_score += 10.0
        
        if "bearish_engulfing" in price_action_patterns:
            weighted_score += 15.0
        elif "evening_star" in price_action_patterns:
            weighted_score += 12.0
        elif "bearish_hanging_man" in price_action_patterns:
            weighted_score += 8.0
        max_score += 15.0
        
        if await self._check_momentum_reversal(symbol, "short"):
            weighted_score += 10.0
        max_score += 10.0
        
        if await self._check_order_book_strength(symbol, "short"):
            weighted_score += 10.0
        max_score += 10.0
        
        divergence = momentum.get("divergence", "")
        trendline_break_short = momentum.get("trendline_break_short", False)
        resistance_zones = momentum.get("resistance_zones", [])
        
        if divergence == "bearish_divergence":
            weighted_score += 20.0
        max_score += 20.0
        
        if trendline_break_short:
            weighted_score += 15.0
        max_score += 15.0
        
        for zone in resistance_zones:
            if zone * 0.995 <= current_price <= zone * 1.005:
                weighted_score += 10.0
                break
        max_score += 10.0
        
        confidence = min(0.95, weighted_score / max_score)
        
        threshold_ratio = 0.40
        if self._adaptive_enabled:
            market_state = self._market_state.get(symbol, {"state": "range", "volatility": "normal"})
            if market_state.get("volatility") == "low":
                threshold_ratio = 0.35
            elif market_state.get("volatility") == "high":
                threshold_ratio = 0.45
            if market_state.get("state") == "range":
                threshold_ratio -= 0.03
        
        if weighted_score >= max_score * threshold_ratio:
            await self._generate_signal(symbol, "short", current_price, confidence)

    async def _check_aggressive_long_signal(self, symbol: str, momentum: Dict[str, float]):
        rsi_1m = momentum.get("rsi_1m", 50)
        rsi_5m = momentum.get("rsi_5m", 50)
        vwap_1m = momentum.get("vwap_1m", 0)
        volume_delta = momentum.get("volume_delta", 0)
        price_above_vwap = momentum.get("price_above_vwap", False)
        stoch_1m = momentum.get("stoch_1m", 50)
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        breakout_levels = self._breakout_levels.get(symbol)
        
        if not breakout_levels:
            return
        
        base_conditions = [
            current_price > breakout_levels["resistance"] * 1.001,
            volume_delta > 0.5,
            current_price > vwap_1m * 1.002,
            rsi_1m < 75
        ]
        
        quality_conditions = [
            rsi_5m > 50,
            stoch_1m < 80,
            price_above_vwap,
            breakout_levels["range"] > vwap_1m * 0.005
        ]
        
        signal_strength = sum(base_conditions) * 0.2
        signal_strength += sum(quality_conditions) * 0.05
        signal_strength += min(volume_delta, 1.0) * 0.1
        
        if all(base_conditions) and signal_strength >= 0.7:
            confidence = 0.6 + signal_strength * 0.3
            await self._generate_signal(symbol, "long", current_price, min(0.95, confidence))
        elif all(base_conditions):
            logger.debug(f"Aggressive long signal filtered - insufficient quality: {signal_strength:.2f} for {symbol}")

    async def _check_aggressive_short_signal(self, symbol: str, momentum: Dict[str, float]):
        rsi_1m = momentum.get("rsi_1m", 50)
        rsi_5m = momentum.get("rsi_5m", 50)
        vwap_1m = momentum.get("vwap_1m", 0)
        volume_delta = momentum.get("volume_delta", 0)
        price_above_vwap = momentum.get("price_above_vwap", False)
        stoch_1m = momentum.get("stoch_1m", 50)
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        breakout_levels = self._breakout_levels.get(symbol)
        
        if not breakout_levels:
            return
        
        base_conditions = [
            current_price < breakout_levels["support"] * 0.999,
            volume_delta < -0.5,
            current_price < vwap_1m * 0.998,
            rsi_1m > 25
        ]
        
        quality_conditions = [
            rsi_5m < 50,
            stoch_1m > 20,
            not price_above_vwap,
            breakout_levels["range"] > vwap_1m * 0.005
        ]
        
        signal_strength = sum(base_conditions) * 0.2
        signal_strength += sum(quality_conditions) * 0.05
        signal_strength += min(abs(volume_delta), 1.0) * 0.1
        
        if all(base_conditions) and signal_strength >= 0.7:
            confidence = 0.6 + signal_strength * 0.3
            await self._generate_signal(symbol, "short", current_price, min(0.95, confidence))
        elif all(base_conditions):
            logger.debug(f"Aggressive short signal filtered - insufficient quality: {signal_strength:.2f} for {symbol}")

    def _check_stop_hunt(self, symbol: str, price: float, direction: str) -> bool:
        """成交量优先的停猎检测：异常放量 + 价格在关键位附近"""
        klines_1m = self._last_klines.get(symbol, {}).get("1m", [])
        if len(klines_1m) < 15:
            return False
        
        try:
            recent_volumes = np.array([float(k[5]) for k in klines_1m[:15]])
            recent_lows = np.array([float(k[3]) for k in klines_1m[:15]])
            recent_highs = np.array([float(k[2]) for k in klines_1m[:15]])
            recent_closes = np.array([float(k[4]) for k in klines_1m[:15]])
            
            # 计算 ATR（14周期）
            trs = []
            for i in range(1, min(15, len(klines_1m))):
                h = float(klines_1m[i][2]); l = float(klines_1m[i][3]); pc = float(klines_1m[i-1][4])
                trs.append(max(h - l, abs(h - pc), abs(l - pc)))
            atr = np.mean(trs[-14:]) if trs else price * 0.005
            
            # ATR 动态阈值：ATR/价格的 3 倍作为容忍度
            tol = min(max((atr / price) * 3, 0.002), 0.015)
            
            # 1. 成交量异常检测（核心条件）
            avg_volume = np.mean(recent_volumes[1:])  # 排除最新K线算均值
            latest_volume = recent_volumes[0]
            vol_spike_2x = avg_volume > 0 and latest_volume > avg_volume * 2.0
            vol_spike_3x = avg_volume > 0 and latest_volume > avg_volume * 3.0
            
            if not vol_spike_2x:
                return False  # 无异常放量，不过滤
            
            # 2. 价格位置检测（只在放量时检查）
            if direction == "long":
                # 支撑位：近期低点、10分位低点
                support = float(np.min(recent_lows))
                p10_low = float(np.percentile(recent_lows, 10))
                
                # 价格是否在支撑位附近
                at_support = abs(price - support) / price < tol
                at_p10 = abs(price - p10_low) / price < tol
                
                # 下影线检测（停猎常见特征）
                candle_low = float(klines_1m[0][3])
                candle_close = float(klines_1m[0][4])
                lower_wick = (min(candle_close, float(klines_1m[0][1])) - candle_low) / price
                long_wick = lower_wick > tol * 1.5
                
                # 3x 放量直接判定；2x 放量需要支撑位+影线同时确认
                if vol_spike_3x and at_support:
                    return True
                if vol_spike_2x and (at_support or at_p10) and long_wick:
                    return True
            else:
                resistance = float(np.max(recent_highs))
                p90_high = float(np.percentile(recent_highs, 90))
                
                at_resistance = abs(price - resistance) / price < tol
                at_p90 = abs(price - p90_high) / price < tol
                
                candle_high = float(klines_1m[0][2])
                candle_close = float(klines_1m[0][4])
                upper_wick = (candle_high - max(candle_close, float(klines_1m[0][1]))) / price
                long_wick = upper_wick > tol * 1.5
                
                if vol_spike_3x and at_resistance:
                    return True
                if vol_spike_2x and (at_resistance or at_p90) and long_wick:
                    return True
        except (ValueError, IndexError) as e:
            pass
        
        return False

    def _check_low_liquidity(self, symbol: str) -> bool:
        data = self.okx_client.get_order_book(symbol, 5)
        if not data:
            return False
        
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        
        if len(bids) < 3 or len(asks) < 3:
            return True
        
        try:
            bid_depth = sum(float(b[1]) for b in bids[:3])
            ask_depth = sum(float(a[1]) for a in asks[:3])
            
            avg_depth = (bid_depth + ask_depth) / 2
            
            if avg_depth < 100:
                return True
            
            spread = float(asks[0][0]) - float(bids[0][0])
            price = float(bids[0][0])
            
            if spread / price > 0.002:
                return True
        except (ValueError, IndexError):
            pass
        
        return False

    async def _check_entry_filter(self, symbol: str, price: float, direction: str) -> bool:
        if self._check_stop_hunt(symbol, price, direction):
            logger.debug(f"Stop hunt detected for {symbol}, filtering signal")
            return False
        
        if self._check_low_liquidity(symbol):
            logger.debug(f"Low liquidity detected for {symbol}, filtering signal")
            return False
        
        return True

    async def _check_trend_ready_for_entry(self, symbol: str, direction: str, price: float) -> bool:
        """P33: 趋势确认门禁 — 趋势确认后才允许入场，等位入场（拒绝追行情）

        - ADX > 20 且 DI 方向对齐
        - 做多：等回踩到低位（RSI < 40 或价格在近期低位 25% 区间）
        - 做空：等反弹到高位（RSI > 60 或价格在近期高位 75% 区间）
        """
        try:
            import numpy as np
            klines = await self.okx_client.get_kline_async(symbol, "15m", limit=50)
            if len(klines) < 20:
                return True  # 数据不足时放行

            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            closes = np.array([float(k[4]) for k in klines])

            if np.any(np.isnan(highs)) or np.any(np.isinf(highs)) or \
               np.any(np.isnan(lows)) or np.any(np.isinf(lows)) or \
               np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
                return True

            # 计算简易 ADX
            plus_di_vals = []
            minus_di_vals = []
            tr_vals = []
            for i in range(1, len(highs)):
                tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                tr_vals.append(tr)
                up = highs[i] - highs[i-1] if highs[i] > highs[i-1] else 0
                # 修复：-DM 条件此前写反（lows[i] > lows[i-1]），导致 down 恒为 0、-DI 恒为 0，
                #       空头方向 100% 被拒。正确应为「最低价下跌」（lows[i-1] > lows[i]）。
                down = lows[i-1] - lows[i] if lows[i-1] > lows[i] else 0
                plus_di_vals.append(up if up > down else 0)
                minus_di_vals.append(down if down > up else 0)

            if len(plus_di_vals) < 14:
                return True

            atr14 = np.mean(tr_vals[-14:]) if len(tr_vals) >= 14 else np.mean(tr_vals)
            if atr14 <= 0:
                return True

            plus_di = np.mean(plus_di_vals[-14:]) / atr14 * 100
            minus_di = np.mean(minus_di_vals[-14:]) / atr14 * 100

            if not (math.isfinite(plus_di) and math.isfinite(minus_di)):
                return True

            dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0

            # 修复：原实现 adx = dx 仅用单根 K 线的 DX，噪声极大（常出现 ADX=0.0 或 4~8 的假低值），
            # 导致趋势信号被误拒。改为维护每 symbol 的 DX 历史并取均值（Wilder 平滑的简化），
            # 与 grid/trend 策略的 _calculate_adx 保持同一口径。
            if not math.isfinite(dx):
                dx = 0.0
            if symbol not in self._dx_history:
                self._dx_history[symbol] = []
            self._dx_history[symbol].append(dx)
            if len(self._dx_history[symbol]) > 14:
                self._dx_history[symbol] = self._dx_history[symbol][-14:]
            adx = float(np.mean(self._dx_history[symbol]))

            adx_floor = 20.0

            if direction == "long":
                if adx < adx_floor:
                    logger.debug(f"P33: Scalping {symbol} long rejected — ADX={adx:.1f} < {adx_floor}")
                    return False
                if plus_di <= minus_di:
                    logger.debug(f"P33: Scalping {symbol} long rejected — +DI={plus_di:.1f} <= -DI={minus_di:.1f}")
                    return False

                # 低位等位做多：价格在近期低位区间
                recent_high = float(np.max(highs[-20:]))
                recent_low = float(np.min(lows[-20:]))
                if recent_high > recent_low:
                    price_position = (price - recent_low) / (recent_high - recent_low)
                    if price_position > 0.65:
                        logger.debug(
                            f"P33: Scalping {symbol} long rejected — price at {price_position:.1%} "
                            f"of range (too high, wait for pullback)"
                        )
                        return False
            else:
                if adx < adx_floor:
                    logger.debug(f"P33: Scalping {symbol} short rejected — ADX={adx:.1f} < {adx_floor}")
                    return False
                if minus_di <= plus_di:
                    logger.debug(f"P33: Scalping {symbol} short rejected — -DI={minus_di:.1f} <= +DI={plus_di:.1f}")
                    return False

                # 高位等位开空：价格在近期高位区间
                recent_high = float(np.max(highs[-20:]))
                recent_low = float(np.min(lows[-20:]))
                if recent_high > recent_low:
                    price_position = (price - recent_low) / (recent_high - recent_low)
                    if price_position < 0.35:
                        logger.debug(
                            f"P33: Scalping {symbol} short rejected — price at {price_position:.1%} "
                            f"of range (too low, wait for rally)"
                        )
                        return False

            return True
        except Exception as e:
            logger.debug(f"P33: Scalping trend check error for {symbol}: {e}, rejecting entry (fail-closed)")
            return False

    def _check_position_correlation(self, symbol: str, direction: str) -> bool:
        open_positions = [p for p in self._positions.values() if p["status"] == "open"]
        
        if len(open_positions) == 0:
            return True
        
        same_direction_count = sum(1 for p in open_positions if p["direction"] == direction)
        # 剥头皮策略独立运行，放宽方向集中度限制（避免被趋势策略空单连累）
        max_same_direction = max(5, self._max_positions * 2)
        
        if same_direction_count >= max_same_direction:
            logger.debug(f"Too many {direction} positions, filtering signal for {symbol}")
            return False
        
        return True

    def _check_drawdown_limit(self) -> bool:
        max_drawdown = self.config["trading"].get("max_drawdown", 0.15)
        
        if self._current_drawdown >= max_drawdown:
            logger.warning(f"Maximum drawdown reached ({self._current_drawdown:.2%}), no new signals")
            return False
        
        return True

    def _check_daily_loss_limit(self) -> bool:
        daily_max_loss = self.config["trading"].get("daily_max_loss", 0.03)
        effective_capital = self._get_effective_capital()

        if self._daily_pnl < -daily_max_loss * effective_capital:
            logger.warning(f"Daily loss limit reached ({self._daily_pnl:.2f} USDT), no new signals")
            return False

        return True

    def _update_daily_pnl(self, pnl_usdt: float):
        self._daily_pnl += pnl_usdt

        current_equity = self._daily_start_equity + self._daily_pnl
        
        if current_equity < 0:
            current_equity = 0

        self._max_daily_equity = max(self._max_daily_equity, current_equity)

        if self._max_daily_equity > 0:
            self._current_drawdown = (self._max_daily_equity - current_equity) / self._max_daily_equity
            
            if self._current_drawdown > 1.0:
                self._current_drawdown = 1.0

    def _update_signal_performance(self, signal_type: str, pnl_usdt: float):
        """P0: 按信号类型累计企业级性能追踪（供 AdaptiveKelly 使用）。"""
        perf = self._signal_type_performance.setdefault(signal_type, {
            "wins": 0, "losses": 0, "total_pnl": 0, "total_profit": 0, "total_loss": 0,
            "count": 0, "pnl_list": [], "consecutive_wins": 0, "consecutive_losses": 0,
            "peak_equity": 0, "max_drawdown": 0,
        })
        perf["count"] = perf.get("count", 0) + 1
        perf["total_pnl"] = perf.get("total_pnl", 0) + pnl_usdt
        if pnl_usdt > 0:
            perf["wins"] = perf.get("wins", 0) + 1
            perf["total_profit"] = perf.get("total_profit", 0) + pnl_usdt
            perf["consecutive_wins"] = perf.get("consecutive_wins", 0) + 1
            perf["consecutive_losses"] = 0
        else:
            perf["losses"] = perf.get("losses", 0) + 1
            perf["total_loss"] = perf.get("total_loss", 0) + pnl_usdt
            perf["consecutive_losses"] = perf.get("consecutive_losses", 0) + 1
            perf["consecutive_wins"] = 0
        pnl_list = perf.setdefault("pnl_list", [])
        pnl_list.append(pnl_usdt)
        if len(pnl_list) > 200:
            perf["pnl_list"] = pnl_list[-200:]
        perf["peak_equity"] = max(perf.get("peak_equity", 0), perf["total_pnl"])
        drawdown = perf["peak_equity"] - perf["total_pnl"]
        perf["max_drawdown"] = max(perf.get("max_drawdown", 0), drawdown)

    def _calculate_kelly_position(self, base_position: float, symbol: str) -> float:
        """P0: 企业级凯利仓位调整（跨信号类型聚合后叠加在自适应仓位之上）。"""
        wins = sum(p.get("wins", 0) for p in self._signal_type_performance.values())
        losses = sum(p.get("losses", 0) for p in self._signal_type_performance.values())
        total = wins + losses
        if total < 10:
            return base_position

        total_profit = sum(p.get("total_profit", 0) for p in self._signal_type_performance.values())
        total_loss = sum(p.get("total_loss", 0) for p in self._signal_type_performance.values())
        total_pnl = sum(p.get("total_pnl", 0) for p in self._signal_type_performance.values())

        avg_win = total_profit / wins if wins > 0 else 0.0
        avg_loss = abs(total_loss) / losses if losses > 0 else 0.0
        if avg_win <= 0 or avg_loss <= 0:
            return base_position

        consecutive_wins = max(p.get("consecutive_wins", 0) for p in self._signal_type_performance.values())
        consecutive_losses = max(p.get("consecutive_losses", 0) for p in self._signal_type_performance.values())

        peak = max(p.get("peak_equity", 0) for p in self._signal_type_performance.values())
        drawdown = max(0.0, peak - total_pnl)
        drawdown_pct = drawdown / max(abs(peak), 0.01)

        regime = map_market_state_to_regime(self._market_state.get(symbol, {}).get("state", "unknown"))

        result = self._adaptive_kelly.compute_kelly(
            win_rate=wins / total,
            avg_win=avg_win,
            avg_loss=avg_loss,
            regime=regime,
            drawdown=drawdown_pct,
            consecutive_wins=int(consecutive_wins),
            consecutive_losses=int(consecutive_losses),
            trade_count=total,
        )
        final_kelly = result.get("final_kelly", 0.05)
        if final_kelly <= 0:
            adjusted = base_position * 0.5
        else:
            adjusted = base_position * (1.0 + (final_kelly - 0.05) * 4.0)
            adjusted = max(base_position * 0.3, min(adjusted, base_position * 2.0))
        return adjusted

    async def _init_daily_equity(self):
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                equity = float(account_info.get("totalEq", 0))
                if equity > 0:
                    self._daily_start_equity = equity
                    self._max_daily_equity = equity
                    self._daily_pnl = 0.0
                    self._current_drawdown = 0.0
                    self._daily_reset_date = datetime.now().date()
                    self._equity_initialized = True
                    logger.info(f"Scalping daily equity initialized: {equity:.2f} USDT (date={self._daily_reset_date}), drawdown reset to 0.0")
                    return
            logger.warning("Scalping _init_daily_equity: account_info empty or totalEq=0, fallback to 0")
        except Exception as e:
            logger.error(f"Scalping _init_daily_equity failed: {e}")

    def _check_daily_reset(self):
        """P0-2: 检测跨日，重置日内 PnL 基准。每日 0 点触发一次。"""
        today = datetime.now().date()
        if self._daily_reset_date is None:
            # 首次启动尚未初始化（理论上 start() 已调用 _init_daily_equity），不做同步重置
            return
        if today != self._daily_reset_date:
            logger.info(f"Scalping daily reset triggered: {self._daily_reset_date} -> {today}, re-initializing equity")
            # 触发异步重置（不阻塞当前循环），用 create_task 避免 await 阻塞
            asyncio.create_task(self._init_daily_equity())

    async def _check_momentum_reversal(self, symbol: str, direction: str) -> bool:
        klines_1m = self._last_klines.get(symbol, {}).get("1m", [])
        # P0-1: 数据不足不应获得"动量反转"加分，返回 False
        if len(klines_1m) < 5:
            return False
        
        recent_closes = [float(k[4]) for k in klines_1m[:5]]
        
        if direction == "long":
            if recent_closes[-1] > recent_closes[-2] and recent_closes[-2] < recent_closes[-3]:
                return True
        else:
            if recent_closes[-1] < recent_closes[-2] and recent_closes[-2] > recent_closes[-3]:
                return True
        
        return False

    async def _check_order_book_strength(self, symbol: str, direction: str) -> bool:
        data = self.okx_client.get_order_book(symbol, 10)
        # P0-1: 数据不足不应获得"盘口强度"加分，返回 False
        if not data:
            return False

        bids = data.get("bids", [])
        asks = data.get("asks", [])

        if len(bids) < 5 or len(asks) < 5:
            return False
        
        bid_0_3 = sum(float(b[1]) for b in bids[:3])
        ask_0_3 = sum(float(a[1]) for a in asks[:3])
        bid_0_5 = sum(float(b[1]) for b in bids[:5])
        ask_0_5 = sum(float(a[1]) for a in asks[:5])
        
        bid_spread = float(bids[0][0]) - float(bids[4][0]) if len(bids) >= 5 else 0
        ask_spread = float(asks[4][0]) - float(asks[0][0]) if len(asks) >= 5 else 0
        current_price = float(bids[0][0]) if bids else 0
        
        depth_ratio = bid_0_5 / ask_0_5 if ask_0_5 > 0 else 1.0
        tightness_ratio = bid_spread / ask_spread if ask_spread > 0 else 1.0
        
        if direction == "long":
            if bid_0_3 > ask_0_3 * 1.15:
                return True
            if depth_ratio > 1.2 and tightness_ratio < 1.8:
                return True
            if bid_0_5 > ask_0_5 * 1.1 and current_price < float(asks[0][0]) * 1.0005:
                return True
            return False
        else:
            if ask_0_3 > bid_0_3 * 1.15:
                return True
            if depth_ratio < 0.83 and tightness_ratio > 0.55:
                return True
            if ask_0_5 > bid_0_5 * 1.1 and current_price > float(bids[0][0]) * 0.9995:
                return True
            return False

    async def _check_order_book_liquidity(self, symbol: str) -> bool:
        """盘口流动性检查：买卖5档名义深度均达到最小阈值，确保可成交性。

        用于均值回归等依赖价格偏离（RSI/布林带/VWAP）、不依赖盘口方向偏斜的信号类型。
        """
        data = self.okx_client.get_order_book(symbol, 5)
        if not data:
            return False
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if len(bids) < 5 or len(asks) < 5:
            return False
        min_depth = float(self.config["strategies"]["scalping"].get("min_order_book_depth_usdt", 100.0))
        bid_depth = sum(float(b[0]) * float(b[1]) for b in bids[:5])
        ask_depth = sum(float(a[0]) * float(a[1]) for a in asks[:5])
        return bid_depth >= min_depth and ask_depth >= min_depth

    def _calculate_confidence(self, conditions: List[bool]) -> float:
        base_confidence = 0.5
        for condition in conditions:
            if condition:
                base_confidence += 0.07
        
        return min(0.95, base_confidence)

    def _map_indicators_for_quality(self, raw: Dict[str, Any], price: float) -> Dict[str, float]:
        """将 _momentum_cache 的键名映射为 evaluate_signal_quality 期望的格式"""
        mapped = {}
        
        # rsi: 使用 rsi_1m（最敏感的周期）
        if "rsi_1m" in raw:
            mapped["rsi"] = float(raw["rsi_1m"])
        
        # volume_delta: 键名一致，直接传递
        if "volume_delta" in raw:
            mapped["volume_delta"] = float(raw["volume_delta"])
        
        # vwap_distance: (price - vwap) / vwap
        vwap = raw.get("vwap_1m", price)
        if vwap > 0 and "vwap_1m" in raw:
            mapped["vwap_distance"] = (price - float(vwap)) / float(vwap)
        
        # momentum: 从 macd_hist 近似（macd柱状图本身就是动量指标）
        if "macd_hist" in raw:
            mapped["momentum"] = float(raw["macd_hist"])
        
        # macd_histogram: macd_hist → macd_histogram
        if "macd_hist" in raw:
            mapped["macd_histogram"] = float(raw["macd_hist"])
        
        # bb_position: (price - bb_lower) / (bb_upper - bb_lower)
        bb_upper = raw.get("bb_upper")
        bb_lower = raw.get("bb_lower")
        if bb_upper is not None and bb_lower is not None and float(bb_upper) > float(bb_lower):
            mapped["bb_position"] = (price - float(bb_lower)) / (float(bb_upper) - float(bb_lower))
        
        # stoch_k / stoch_d: 用 stoch_1m 近似（两者相同，简化处理）
        if "stoch_1m" in raw:
            mapped["stoch_k"] = float(raw["stoch_1m"])
            mapped["stoch_d"] = float(raw["stoch_1m"])
        
        # atr_ratio: atr_1m / price
        atr = raw.get("atr_1m")
        if atr is not None and price > 0:
            mapped["atr_ratio"] = float(atr) / price
        
        return mapped

    def _generate_clordid(self, symbol: str, direction: str, signal_type: str) -> str:
        """生成合规 clOrdId（纯字母+数字，≤32位，无特殊字符），用于成交回执关联。

        与 order_executor._generate_idempotency_key 保持同一约束，但由策略侧生成，
        以便在 on_order_filled 中通过 clOrdId 精确匹配 pending intent。
        """
        import re
        ts_ms = int(time.time() * 1000)
        short_symbol = re.sub(r'[^a-zA-Z0-9]', '', symbol.replace("-USDT", ""))[:6]
        short_type = re.sub(r'[^a-zA-Z0-9]', '', str(signal_type or ""))[:6]
        dir_flag = "L" if direction in ("long", "buy") else "S"
        clordid = f"scal{short_symbol}{dir_flag}{ts_ms}{short_type}"
        if len(clordid) > 32:
            clordid = clordid[:32]
        return clordid

    def _cleanup_expired_pending_entries(self) -> int:
        """ghost_close 专项：清理超时未成交的 pending intent（限价单 TTL 后仍未成交）。

        独立于交易所仓位获取——verify_exchange_positions 依赖 get_positions 成功，
        网络故障时 get_positions 返回 None 导致 TTL 清理不执行，pending_entry_exists
        会在网络抖动期间无限期阻塞新开仓。本方法每次信号生成前兜底清理。
        """
        pending_ttl = self.config["strategies"]["scalping"].get("pending_ttl_seconds", 15)
        now_ts = time.time()
        cleaned = 0
        for symbol in list(self._pending_entries.keys()):
            entry = self._pending_entries[symbol]
            try:
                age = now_ts - float(entry.get("timestamp", 0))
            except (TypeError, ValueError):
                age = float("inf")
            if age > pending_ttl:
                logger.warning(
                    f"Pending entry expired for {symbol} (age={age:.0f}s > {pending_ttl}s), dropping"
                )
                del self._pending_entries[symbol]
                self._pending_timeout_cleaned += 1
                cleaned += 1
        return cleaned

    async def _generate_signal(self, symbol: str, direction: str, price: float, confidence: float):
        # 企业级：参数前置校验，非法输入直接拒绝并埋点
        if not self._validate_symbol(symbol) or not self._validate_direction(direction) \
                or not self._validate_price(price) or not self._validate_confidence(confidence):
            logger.warning(
                f"Scalping signal rejected: invalid params "
                f"symbol={symbol!r} direction={direction!r} price={price!r} confidence={confidence!r}"
            )
            self._increment_metric("scalping_signal_rejected_total", 1.0, {"reason": "invalid_params", "symbol": symbol})
            return

        # ghost_close 专项：先清理过期 pending，避免网络故障时 TTL 清理失效阻塞开仓
        if self._pending_entries:
            self._cleanup_expired_pending_entries()

        if symbol in self._positions and self._positions[symbol]["status"] == "open":
            self._record_filter(symbol, "position_already_open")
            return
        # ghost_close 专项 Phase 2: 待确认开仓意图同样阻塞同 symbol 新开仓，避免重复下单
        if symbol in self._pending_entries:
            self._record_filter(symbol, "pending_entry_exists")
            return

        # P32: 日内交易次数限制（max_daily_trades<=0 表示不限制）
        today = datetime.now().date()
        if self._daily_trade_reset_date != today:
            self._daily_trade_count = 0
            self._daily_trade_reset_date = today
        if self._max_daily_trades > 0 and self._daily_trade_count >= self._max_daily_trades:
            logger.debug(f"P32: Scalping daily trade limit reached ({self._max_daily_trades}), skipping {symbol}")
            self._record_filter(symbol, "daily_trade_limit")
            return

        # P1-Fix: 信号TTL过期机制 - ticker缓存超过3秒则丢弃信号
        now = datetime.now()
        ticker_cache_age = self._ticker_cache_time.get(symbol)
        if ticker_cache_age and (now - ticker_cache_age).total_seconds() > 3:
            logger.debug(f"Scalping signal TTL expired for {symbol}: ticker cache age={((now - ticker_cache_age).total_seconds()):.1f}s > 3s")
            self._record_filter(symbol, "signal_ttl_expired")
            return

        # P1-Fix: 盘口价差检查 - spread > 0.2% 则跳过
        try:
            order_book = self.okx_client.get_order_book(symbol, 3)
            if order_book:
                bids = order_book.get("bids", [])
                asks = order_book.get("asks", [])
                if bids and asks and len(bids) > 0 and len(asks) > 0:
                    best_bid = float(bids[0][0]) if bids[0] else 0
                    best_ask = float(asks[0][0]) if asks[0] else 0
                    if best_bid > 0 and best_ask > 0:
                        spread = (best_ask - best_bid) / best_bid
                        if spread > 0.002:
                            logger.debug(f"Scalping signal skipped for {symbol}: spread={spread:.4%} > 0.2%")
                            self._record_filter(symbol, "spread_too_wide")
                            return
        except Exception as e:
            logger.debug(f"Scalping spread check failed for {symbol}: {e}")

        # P1-Fix: 盘口拦截改为必要条件 - 非动量型信号必须通过盘口检查
        signal_type = self._position_signal_type.get(symbol, "momentum")
        if signal_type == "mean_reversion":
            # 均值回归依赖价格偏离（RSI/布林带/VWAP）而非盘口方向偏斜，仅需盘口流动性充足
            if not await self._check_order_book_liquidity(symbol):
                logger.debug(f"Scalping signal rejected for {symbol}: order book liquidity insufficient ({signal_type})")
                self._record_filter(symbol, "order_book_liquidity_insufficient")
                return
        elif signal_type != "momentum":
            if not await self._check_order_book_strength(symbol, direction):
                logger.debug(f"Scalping signal rejected for {symbol}: order book strength insufficient ({signal_type})")
                self._record_filter(symbol, "order_book_strength_insufficient")
                return

        # P0-4: 临近 funding 结算暂停开仓（避免结算瞬间价格剧烈波动）
        try:
            funding_data = await self.okx_client.get_funding_rate_async(symbol)
            if funding_data:
                next_funding_ms = funding_data.get("nextFundingTime")
                funding_rate = float(funding_data.get("fundingRate", 0))
                if next_funding_ms:
                    next_funding_ts = int(next_funding_ms) / 1000.0
                    seconds_to_funding = next_funding_ts - datetime.now().timestamp()
                    # 距 nextFundingTime < 5 分钟且 |rate| > 0.0003 (0.03%) 时跳过
                    if seconds_to_funding < 300 and abs(funding_rate) > 0.0003:
                        logger.debug(
                            f"Scalping signal skipped for {symbol}: near funding settlement "
                            f"(seconds_to_funding={seconds_to_funding:.0f}s, rate={funding_rate:.4%})"
                        )
                        self._record_filter(symbol, "near_funding_settlement")
                        return
        except Exception as e:
            logger.debug(f"Scalping funding rate check failed for {symbol}: {e}")

        if not await self._check_entry_filter(symbol, price, direction):
            self._record_filter(symbol, "entry_filter_failed")
            return

        # P33: 趋势确认 + 等位入场 — 趋势确认后才入场，等回踩/反弹到合理位置
        # mean_reversion / range 均内建等位入场（RSI超卖/布林带下轨/VWAP偏离 或 区间上下沿高抛低吸），
        # 且两者都需震荡市（低 ADX），与 ADX≥20 趋势门禁矛盾，故跳过；momentum/breakout 保留趋势确认
        if signal_type not in ("mean_reversion", "range"):
            if not await self._check_trend_ready_for_entry(symbol, direction, price):
                self._record_filter(symbol, "trend_not_ready")
                return
        
        if self._adaptive_enabled:
            market_state = self._market_state.get(symbol, {"state": "unknown", "volatility": "normal"})
            adjusted_thresholds = self._adjusted_thresholds.get(symbol, self._base_thresholds)
            
            # 修复：键名映射 - _momentum_cache 的键名与 evaluate_signal_quality 期望不一致
            # momentum_cache 用 rsi_1m/macd_hist/bb_upper 等，evaluate_signal_quality 期望 rsi/macd_histogram/bb_position
            raw = self._momentum_cache.get(symbol, {})
            indicators = self._map_indicators_for_quality(raw, price)
            signal_quality = evaluate_signal_quality(indicators, market_state, direction)
            
            if signal_quality < self._dynamic_min_quality():
                logger.debug(f"Signal quality {signal_quality:.2f} below threshold {self._dynamic_min_quality()} for {symbol}")
                self._record_filter(symbol, "signal_quality_low")
                return
            
            confidence = confidence * (0.8 + signal_quality * 0.2)
        
        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        leverage = min(tier_settings["leverage_max"], 8)
        
        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        position_limit = tier_settings["position_limit"]

        tier_count = len(self._scalping_symbols)
        if tier == "tier1":
            tier_count = len(self.config["currencies"]["tier1_symbols"])
        elif tier == "tier2":
            tier_count = len(self.config["currencies"]["tier2_symbols"])
        elif tier == "tier3":
            tier_count = len(self.config["currencies"]["tier3_symbols"])
        
        if self._position_sizing_mode == "confidence":
            base_position = trading_capital * min(allocation, position_limit) * confidence
        elif self._position_sizing_mode == "aggressive":
            base_position = trading_capital * min(allocation, position_limit) * 2.0
        else:
            base_position = trading_capital * min(allocation, position_limit)

        price_factor = 1.0
        if tier == "tier2":
            price_factor = 1.3
        elif tier == "tier3":
            price_factor = 1.6

        base_position *= price_factor

        if self._adaptive_enabled:
            market_state = self._market_state.get(symbol, {"state": "range", "volatility": "normal", "volume_ratio": 1.0})
            base_position = calculate_adaptive_position_size(base_position, market_state, "scalping")

        # P0: 企业级凯利仓位调整（在静态自适应之上叠加数据驱动的凯利）
        base_position = self._calculate_kelly_position(base_position, symbol)

        # P-复盘修复：单仓保证金硬上限，防止仓位累积导致保证金超过账户权益
        # （历史事故：AVAX 单仓保证金 102 USDT 超过 46 USDT 账户权益，单笔亏损 -12.70 USDT）
        max_single_margin_pct = self.config["strategies"]["scalping"].get("max_single_position_margin_pct", 0.20)
        base_position = min(base_position, trading_capital * max_single_margin_pct)

        # 注入空闲资金放大乘数（来自 AdaptiveController，受 adaptive_enabled 门控）
        # 在硬上限之后应用，确保 boost 不会被上限抵消；但 boost 后仍受 2x 硬顶保护
        if self._adaptive_enabled and self._adaptive_controller:
            try:
                boost = self._adaptive_controller.get_position_boost()
                if boost > 1.0:
                    boosted = base_position * boost
                    cap = trading_capital * max_single_margin_pct * 2.0
                    base_position = min(boosted, cap)
            except Exception as e:
                logger.debug(f"[scalping] get_position_boost failed: {e}")

        # 措施5: 高滑点币降权/禁用 — 滑点成本翻转治理
        # 当有效滑点相对最小止盈目标(tp1)占比过高时，单边成交成本吃掉大部分盈利，导致成本翻转。
        # 占比 >= max_slippage_ratio 直接禁用；介于 downgrade_ratio~max_slippage_ratio 之间降权减仓。
        if self._slippage_filter_enabled:
            tier_slippage = tier_settings.get("slippage", 0.001)
            effective_slippage = max(float(tier_slippage), float(self._symbol_slippage_overrides.get(symbol, 0.0)))
            min_profit_target = self._tp1_pct if (self._take_profit_enabled and self._tp1_ratio > 0 and self._tp1_pct > 0) else self._profit_target_min
            if min_profit_target > 0:
                slippage_ratio = effective_slippage / min_profit_target
                if slippage_ratio >= self._max_slippage_ratio:
                    logger.warning(
                        f"Scalping signal disabled for {symbol}: effective slippage {effective_slippage:.4%} "
                        f"= {slippage_ratio:.0%} of min profit target {min_profit_target:.4%} (cost reversal risk)"
                    )
                    self._record_filter(symbol, "slippage_cost_reversal")
                    return
                if slippage_ratio >= self._high_slippage_downgrade_ratio:
                    base_position *= self._high_slippage_downgrade_factor
                    logger.info(
                        f"Scalping position downgraded for {symbol}: slippage ratio {slippage_ratio:.2f} "
                        f"-> position scaled x{self._high_slippage_downgrade_factor}"
                    )

        quantity = base_position * leverage / price
        
        min_lot_size = self._safe_float((await self.okx_client.get_instrument_info_async(symbol) or {}).get("lotSz", "1"), 1.0)
        margin_needed_for_min_lot = price * min_lot_size / leverage
        
        if base_position < margin_needed_for_min_lot:
            logger.warning(f"Insufficient capital for scalping {symbol}: need {margin_needed_for_min_lot:.2f} USDT, have {base_position:.2f} USDT. Skipping.")
            self._record_filter(symbol, "insufficient_capital_min_lot")
            return
        
        if quantity < self._min_quantity_threshold:
            logger.debug(f"Quantity {quantity} below threshold for {symbol}")
            self._record_filter(symbol, "quantity_below_threshold")
            return
        
        vwap = self._momentum_cache.get(symbol, {}).get("vwap_1m", price)
        precision = get_price_precision(symbol)
        
        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        slippage = tier_settings.get("slippage", 0.001)
        
        if self._adaptive_enabled:
            adjusted_thresholds = self._adjusted_thresholds.get(symbol, self._base_thresholds)
            profit_target = adjusted_thresholds.get("profit_target_min", self._profit_target_min) + \
                           (self._profit_target_max - adjusted_thresholds.get("profit_target_min", self._profit_target_min)) * confidence
            stop_loss_pct = adjusted_thresholds.get("stop_loss", self._stop_loss)
        else:
            profit_target = self._profit_target_min + (self._profit_target_max - self._profit_target_min) * confidence
            stop_loss_pct = self._stop_loss
        
        take_profit = calculate_take_profit(price, direction, profit_target, slippage, precision)
        
        if direction == "long":
            sl_calculated = calculate_stop_loss(price, direction, stop_loss_pct, slippage, precision)
            vwap_distance = (price - vwap) / vwap if vwap > 0 else 0
            if vwap_distance < -0.003:
                vwap_bound = round(vwap * (1 - stop_loss_pct * 1.5), precision)
                # 确保 VWAP 边界止损不高于入场价（长仓止损必须在入场价下方）
                stop_loss = max(sl_calculated, min(vwap_bound, price - price * max(slippage, 0.0005)))
            else:
                stop_loss = sl_calculated
        else:
            sl_calculated = calculate_stop_loss(price, direction, stop_loss_pct, slippage, precision)
            vwap_distance = (price - vwap) / vwap if vwap > 0 else 0
            if vwap_distance > 0.003:
                vwap_bound = round(vwap * (1 + stop_loss_pct * 1.5), precision)
                # 确保 VWAP 边界止损不低于入场价（短仓止损必须在入场价上方）
                stop_loss = min(sl_calculated, max(vwap_bound, price + price * max(slippage, 0.0005)))
            else:
                stop_loss = sl_calculated
        
        validation = validate_tp_sl_prices(price, direction, take_profit, stop_loss, price)
        if not validation["valid"]:
            logger.error(f"TP/SL validation failed for {symbol}: {validation['errors']}")
            self._record_filter(symbol, "tp_sl_validation_failed")
            return
        
        for warning in validation["warnings"]:
            logger.warning(f"TP/SL warning for {symbol}: {warning}")
        
        risk_reward = calculate_risk_reward_ratio(price, take_profit, stop_loss, direction)
        if risk_reward < 1.0:
            logger.warning(f"Low risk-reward ratio ({risk_reward:.2f}) for {symbol}, skipping signal")
            self._record_filter(symbol, "risk_reward_low")
            return

        # P1-Fix: 限价单内侧挂单 - 使用bid价(买单)或ask价(卖单)减少滑点
        limit_price = price
        try:
            ob = self.okx_client.get_order_book(symbol, 3)
            if ob:
                ob_bids = ob.get("bids", [])
                ob_asks = ob.get("asks", [])
                if ob_bids and ob_asks and len(ob_bids) > 0 and len(ob_asks) > 0:
                    best_bid = float(ob_bids[0][0])
                    best_ask = float(ob_asks[0][0])
                    if direction == "long":
                        limit_price = best_bid
                    else:
                        limit_price = best_ask
        except Exception:
            pass

        # P32: 日内计数改在成交回执(on_order_filled)中递增，信号生成阶段不计数，
        # 避免信号被 RegimeGate/Agent 拒绝或限价单未成交时白白空烧日内额度。
        logger.info(f"P32: Scalping signal generated for {symbol} {direction}")

        # 信号子类型（momentum/mean_reversion/breakout/range）编码进 signal_type，落库供复盘
        signal_subtype = self._position_signal_type.get(symbol, "momentum")
        signal = Signal(
            symbol=symbol,
            strategy_name="scalping",
            signal_type=f"scalping_entry_{signal_subtype}",
            direction=direction,
            price=limit_price,
            quantity=quantity,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=datetime.now()
        )

        clordid = self._generate_clordid(symbol, direction, signal.signal_type)

        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
                "direction": signal.direction,
                "price": signal.price,
                "quantity": signal.quantity,
                "clOrdId": clordid,
                "leverage": signal.leverage,
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat(),
                # P0-3: 剥头皮开仓使用限价单挂在 bid/ask 内侧减少滑价（GTT + 10s TTL）
                "order_type": "limit",
                "time_in_force": "GTT",
                "ttl_seconds": 10
            }
        })

        if self._use_fill_driven_position:
            # ghost_close 专项 Phase 2: 记录待确认开仓意图，等成交回执再物化 _positions
            self._pending_entries[symbol] = {
                "direction": direction,
                "entry_price": limit_price,
                "quantity": quantity,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "vwap_entry": vwap,
                "clOrdId": clordid,
                "signal_type": signal.signal_type,
                "timestamp": time.time(),
            }
            logger.info(f"Scalping pending entry recorded: {symbol} {direction} clOrdId={clordid}")
        else:
            self._positions[symbol] = {
                "direction": direction,
                "entry_price": limit_price,
                "quantity": quantity,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "entry_time": datetime.now(),
                "status": "open",
                "peak_price": limit_price if direction == "long" else 0,
                "trough_price": limit_price,
                "trailing_stop_initial": stop_loss,
                "vwap_entry": vwap,
                "current_profit": 0.0,
                "partial_closes": []
            }

        # P2: 记录信号时间戳用于全局频率控制
        self._recent_signals_list.append({"timestamp": datetime.now(), "symbol": symbol, "direction": direction})
        
        self._record_metric("scalping_signal_generated_total", 1.0, {"symbol": symbol, "direction": direction})
        self._record_metric("scalping_signal_confidence", confidence, {"symbol": symbol, "direction": direction})

        logger.info(f"Scalping signal: {direction} {symbol} @ {limit_price:.4f}, qty: {quantity:.4f}, TP: {take_profit:.4f}, SL: {stop_loss:.4f}, conf: {confidence:.2f}")

    def _cleanup_stale_caches(self):
        """定期清理过期缓存，防止内存泄漏
        
        清理目标：
        - _signal_cooldown: 超过10分钟的冷却记录
        - _position_verify_cache: 超过30秒的验证缓存
        - _ticker_cache / _ticker_cache_time: 超过60秒的ticker缓存
        """
        now = time.time()
        
        # 清理过期冷却记录（10分钟阈值）
        stale_cooldowns = []
        for symbol, cooldown_time in self._signal_cooldown.items():
            if (datetime.now() - cooldown_time).total_seconds() > 600:
                stale_cooldowns.append(symbol)
        for symbol in stale_cooldowns:
            self._signal_cooldown.pop(symbol, None)
        if stale_cooldowns:
            logger.debug(f"Scalping cache cleanup: removed {len(stale_cooldowns)} stale cooldowns")
        
        # 清理过期持仓验证缓存（30秒阈值）
        stale_verifies = []
        for cache_key, (cached_time, _) in self._position_verify_cache.items():
            if now - cached_time > 30:
                stale_verifies.append(cache_key)
        for key in stale_verifies:
            self._position_verify_cache.pop(key, None)
        
        # 清理过期ticker缓存（60秒阈值）
        stale_tickers = []
        for symbol, ticker_time in self._ticker_cache_time.items():
            if (datetime.now() - ticker_time).total_seconds() > 60:
                stale_tickers.append(symbol)
        for symbol in stale_tickers:
            self._ticker_cache.pop(symbol, None)
            self._ticker_cache_time.pop(symbol, None)
        
        total_cleaned = len(stale_cooldowns) + len(stale_verifies) + len(stale_tickers)
        if total_cleaned > 0:
            logger.debug(f"Scalping cache cleanup: {total_cleaned} stale entries removed "
                        f"(cooldowns={len(stale_cooldowns)}, verifies={len(stale_verifies)}, tickers={len(stale_tickers)})")
        
        self._last_cache_cleanup = now

    def _recover_orphan_positions(self):
        """从数据库恢复孤儿持仓：DB 中 open 但内存 self._positions 缺失的 scalping 持仓。

        场景：策略重启后内存状态丢失，但交易所仍持有真实仓位（DB open 记录未关闭）。
        若不物化，这些持仓将永远不被止损/超时检查，成为「僵尸仓」。

        物化后进入 _manage_positions 的止损/超时检查链：
        - _verify_exchange_position 会在 30s 内校验交易所是否仍持有该仓位，
          若交易所已无仓位则 2 次失败后 cleanup_position 关闭 DB 记录（自愈）。
        - 若交易所仍有仓位，则止损/超时逻辑正常接管。
        """
        if not self.sqlite_storage:
            return
        now_ts = time.time()
        if now_ts - self._orphan_recovery_last_ts < 60:
            return
        self._orphan_recovery_last_ts = now_ts

        try:
            open_records = self.sqlite_storage.get_all_open_records() or []
        except Exception as e:
            logger.warning(f"Scalping orphan recovery: failed to query open records: {e}")
            return

        recovered = 0
        for rec in open_records:
            try:
                if str(rec.get("strategy_name", "")).strip().lower() != "scalping":
                    continue
                symbol = rec.get("symbol", "")
                if not symbol:
                    continue
                # 已存在 open 持仓则跳过，避免覆盖内存中的实时状态
                if symbol in self._positions and self._positions[symbol].get("status") == "open":
                    continue

                # 归一化方向：side 可能存 long/short 或 buy/sell
                side = str(rec.get("side", "")).strip().lower()
                if side in ("long", "buy"):
                    direction = "long"
                elif side in ("short", "sell"):
                    direction = "short"
                else:
                    logger.debug(f"Scalping orphan recovery: unknown side '{side}' for {symbol}, skip")
                    continue

                entry_price = float(rec.get("filled_price") or rec.get("price") or 0)
                if entry_price <= 0:
                    logger.warning(f"Scalping orphan recovery: no entry price for {symbol}, skip")
                    continue
                quantity = float(rec.get("quantity", 0) or 0)
                if quantity <= 0:
                    logger.warning(f"Scalping orphan recovery: no quantity for {symbol}, skip")
                    continue

                entry_time = rec.get("create_time")
                if isinstance(entry_time, str):
                    try:
                        entry_time = datetime.fromisoformat(entry_time)
                    except (ValueError, TypeError):
                        entry_time = datetime.now()
                if not isinstance(entry_time, datetime):
                    entry_time = datetime.now()

                stop_loss = entry_price * (1 - self._stop_loss) if direction == "long" else entry_price * (1 + self._stop_loss)
                take_profit = entry_price * (1 + self._profit_target_min) if direction == "long" else entry_price * (1 - self._profit_target_min)

                self._positions[symbol] = {
                    "direction": direction,
                    "entry_price": entry_price,
                    "quantity": quantity,
                    "leverage": rec.get("leverage", 1) or 1,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": entry_time,
                    "status": "open",
                    "peak_price": entry_price if direction == "long" else 0,
                    "trough_price": entry_price,
                    "trailing_stop_initial": stop_loss,
                    "vwap_entry": entry_price,
                    "current_profit": 0.0,
                    "partial_closes": [],
                    "_orphan_recovered": True,
                }
                recovered += 1
                logger.warning(
                    f"Scalping orphan recovered: {symbol} {direction} qty={quantity} "
                    f"entry={entry_price:.4f} held since {entry_time} (from DB open record)"
                )
            except Exception as e:
                logger.warning(f"Scalping orphan recovery: failed to recover record {rec.get('id', '?')}: {e}")

        if recovered:
            logger.warning(f"Scalping orphan recovery: materialized {recovered} orphan position(s) from DB")

    async def _run_position_check(self, symbol: str, check_fn, *args):
        """执行单个持仓检查；持仓已在上一检查中被平掉时安全跳过。

        修复反复出现的 KeyError：_manage_positions 对同一 symbol 依次调用多个
        _check_* 方法，前一个方法可能 _close_position/_close_partial 平仓并触发
        cleanup_position 删除 _positions[symbol]，后一个方法直接下标会抛 KeyError
        （历史 crash_log 多次出现 XRP/SOL/DOGE KeyError）。此处统一在调用前做存在性守卫。
        """
        if symbol not in self._positions:
            return
        await check_fn(symbol, *args)

    async def _manage_positions(self):
        self._recover_orphan_positions()
        for symbol in list(self._positions.keys()):
            state = self._positions[symbol]
            if state["status"] != "open":
                continue
            
            # 定期验证交易所实际持仓（每30秒检查一次），防止幽灵仓位
            now_ts = time.time()
            last_verify = state.get("_last_verify_ts", 0)
            if now_ts - last_verify > 30:
                if not await self._verify_exchange_position(symbol, state["direction"]):
                    # 连续验证失败计数：2次失败后强制清理
                    fail_count = state.get("_verify_fail_count", 0) + 1
                    state["_verify_fail_count"] = fail_count
                    if fail_count >= 2:
                        logger.warning(f"Scalping ghost position confirmed for {symbol} ({state['direction']}) after {fail_count} failed verifications, cleaning up")
                        self.cleanup_position(symbol, "scalping")
                        continue
                    else:
                        logger.warning(f"Scalping position verify failed for {symbol} ({state['direction']}), attempt {fail_count}/2")
                else:
                    state["_verify_fail_count"] = 0  # 验证通过，重置计数
                state["_last_verify_ts"] = now_ts
            
            ticker = await self._get_ticker_cached(symbol)
            if not ticker:
                continue
            
            current_price = float(ticker["last"])
            entry_price = state["entry_price"]
            direction = state["direction"]
            
            if direction == "long":
                state["peak_price"] = max(state["peak_price"], current_price)
            else:
                state["trough_price"] = min(state["trough_price"], current_price)
            
            profit = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price
            state["current_profit"] = profit
            
            if self._volatility_adaptive_sl:
                atr = self._get_atr(symbol)
                old_sl = state["stop_loss"]
                new_sl = adjust_stop_loss_for_volatility(entry_price, direction, atr, self._stop_loss)
                
                # P0: 止损只能收紧不能放宽 - 防止ATR变大时止损反向扩大
                if direction == "long":
                    state["stop_loss"] = max(old_sl, new_sl)  # 多头止损只能上移
                else:
                    state["stop_loss"] = min(old_sl, new_sl)  # 空头止损只能下移
                
                # P0: 硬性最大止损限制 - 无论ATR多大，止损不超过max_stop_loss_pct
                if direction == "long":
                    max_sl = entry_price * (1 - self._max_stop_loss_pct)
                    if state["stop_loss"] < max_sl:
                        state["stop_loss"] = max_sl
                        logger.warning(f"Scalping {symbol}: stop_loss capped at max {self._max_stop_loss_pct*100:.1f}% (was being pushed below)")
                else:
                    max_sl = entry_price * (1 + self._max_stop_loss_pct)
                    if state["stop_loss"] > max_sl:
                        state["stop_loss"] = max_sl
                        logger.warning(f"Scalping {symbol}: stop_loss capped at max {self._max_stop_loss_pct*100:.1f}% (was being pushed above)")
                
                # P0: 单笔最大亏损USDT硬限制 - 基于实际资金检查
                if self._max_single_trade_loss_usdt > 0:
                    max_loss_price = self._calculate_max_loss_price(entry_price, direction, state["quantity"], self._max_single_trade_loss_usdt)
                    if max_loss_price > 0:
                        if direction == "long":
                            if state["stop_loss"] < max_loss_price:
                                state["stop_loss"] = max_loss_price
                                logger.warning(f"Scalping {symbol}: stop_loss tightened to cap single-trade loss at ${self._max_single_trade_loss_usdt}")
                        else:
                            if state["stop_loss"] > max_loss_price:
                                state["stop_loss"] = max_loss_price
            
            await self._run_position_check(symbol, self._check_momentum_decay, current_price)
            await self._run_position_check(symbol, self._check_scaled_take_profit, current_price)
            await self._run_position_check(symbol, self._check_multiple_take_profit, current_price)
            await self._run_position_check(symbol, self._check_breakeven_stop, current_price)
            await self._run_position_check(symbol, self._check_dynamic_trailing_stop, current_price)
            await self._run_position_check(symbol, self._check_atr_trailing_stop, current_price)
            await self._run_position_check(symbol, self._check_structural_level_exit, current_price)
            await self._run_position_check(symbol, self._check_volatility_spike_exit, current_price)
            await self._run_position_check(symbol, self._check_volatility_based_exit, current_price)
            await self._run_position_check(symbol, self._check_stop_loss, current_price)
            await self._run_position_check(symbol, self._check_profit_protection, current_price)
            await self._run_position_check(symbol, self._check_time_limit)

            # 行情反转智能落袋：动态止盈 + 反转减仓（HMM 为主 + 指标兜底）
            if self._stop_loss_manager and self._positions.get(symbol, {}).get("status") == "open":
                await self._stop_loss_manager.check_reversal_take_profit(
                    symbol=symbol,
                    strategy_name="scalping",
                    current_price=current_price,
                    position_state=self._positions.get(symbol, {}),
                )

    async def _check_scaled_take_profit(self, symbol: str, current_price: float):
        # 现代多级止盈(tp1/tp2/tp3 由 take_profit_enabled 控制)启用时，
        # 旧版 scaled TP(profit_taking_levels)不再叠加执行，避免双重部分平仓。
        if self._take_profit_enabled:
            return
        state = self._positions[symbol]
        direction = state["direction"]
        entry_price = state["entry_price"]
        current_profit = state["current_profit"]
        
        profit_info = calculate_scaled_take_profit(entry_price, direction, self._profit_taking_levels, current_profit)
        
        if profit_info["next_level"]:
            next_level = profit_info["next_level"]
            if direction == "long" and current_price >= next_level["price"]:
                close_qty = state["quantity"] * next_level["close_ratio"]
                await self._close_partial(symbol, close_qty, f"profit_level_{next_level['level']}")
                logger.info(f"Scaled profit taking: {symbol}, level {next_level['level']}, closed {close_qty:.4f} at {current_price:.4f}")
            elif direction == "short" and current_price <= next_level["price"]:
                close_qty = state["quantity"] * next_level["close_ratio"]
                await self._close_partial(symbol, close_qty, f"profit_level_{next_level['level']}")
                logger.info(f"Scaled profit taking: {symbol}, level {next_level['level']}, closed {close_qty:.4f} at {current_price:.4f}")

    async def _check_dynamic_trailing_stop(self, symbol: str, current_price: float):
        state = self._positions[symbol]
        direction = state["direction"]
        entry_price = state["entry_price"]
        current_profit = state["current_profit"]
        
        if current_profit >= self._profit_target_min * self._trailing_stop_activation:
            new_trailing_stop = calculate_trailing_stop(entry_price, current_price, direction, 
                                                       self._trailing_stop_initial, self._trailing_stop_min)
            
            if direction == "long":
                if new_trailing_stop > state["stop_loss"]:
                    state["stop_loss"] = new_trailing_stop
            else:
                if new_trailing_stop < state["stop_loss"]:
                    state["stop_loss"] = new_trailing_stop

    async def _check_atr_trailing_stop(self, symbol: str, current_price: float):
        state = self._positions[symbol]
        direction = state["direction"]
        entry_price = state["entry_price"]
        current_profit = state["current_profit"]
        
        atr = self._get_atr(symbol)
        if atr == 0:
            return
        
        if current_profit >= self._profit_target_min:
            atr_trailing_distance = atr * 1.5
            
            if direction == "long":
                atr_stop = current_price - atr_trailing_distance
                if atr_stop > state["stop_loss"]:
                    state["stop_loss"] = atr_stop
            else:
                atr_stop = current_price + atr_trailing_distance
                if atr_stop < state["stop_loss"]:
                    state["stop_loss"] = atr_stop

    async def _check_structural_level_exit(self, symbol: str, current_price: float):
        state = self._positions[symbol]
        direction = state["direction"]
        entry_price = state["entry_price"]
        
        # P4: 级联平仓保护 - 同symbol 30秒内不重复触发结构性退出
        now_ts = time.time()
        last_structural_exit = state.get("_last_structural_exit_ts", 0)
        if now_ts - last_structural_exit < 30:
            return
        
        # P4: 最小数量保护 - 防止无限递归平仓（剩余数量过小不触发部分平仓）
        min_close_qty = max(self._min_quantity_threshold * 3, state.get("_initial_quantity", state["quantity"]) * 0.05)
        if state["quantity"] <= min_close_qty:
            return
        
        momentum = self._momentum_cache.get(symbol)
        if not momentum:
            return
        
        if direction == "long":
            resistance_zones = momentum.get("resistance_zones", [])
            for zone in resistance_zones:
                if zone * 0.995 <= current_price <= zone * 1.005:
                    if state["current_profit"] > 0:
                        state["_last_structural_exit_ts"] = now_ts
                        await self._close_partial(symbol, state["quantity"] * 0.5, "resistance_level")
                        logger.info(f"Resistance level exit: {symbol}, closed 50% at {current_price:.4f}")
                    break
        else:
            support_zones = momentum.get("support_zones", [])
            for zone in support_zones:
                if zone * 0.995 <= current_price <= zone * 1.005:
                    if state["current_profit"] > 0:
                        state["_last_structural_exit_ts"] = now_ts
                        await self._close_partial(symbol, state["quantity"] * 0.5, "support_level")
                        logger.info(f"Support level exit: {symbol}, closed 50% at {current_price:.4f}")
                    break

    async def _check_stop_loss(self, symbol: str, current_price: float):
        state = self._positions[symbol]
        direction = state["direction"]
        stop_loss = state["stop_loss"]
        entry_price = state["entry_price"]
        
        # P0: 硬性最大止损检查 - 双保险：价格止损+百分比止损，取更严格的
        if direction == "long":
            max_sl_price = entry_price * (1 - self._max_stop_loss_pct)
            effective_sl = max(stop_loss, max_sl_price)  # 取更高值（更严格的止损）
        else:
            max_sl_price = entry_price * (1 + self._max_stop_loss_pct)
            effective_sl = min(stop_loss, max_sl_price)  # 取更低值（更严格的止损）
        
        triggered = False
        if direction == "long" and current_price <= effective_sl:
            triggered = True
        elif direction == "short" and current_price >= effective_sl:
            triggered = True
        
        if triggered:
            state["exit_reason"] = "stop_loss"
            # P0: 使用StopLossManager统一执行止损
            if self._stop_loss_manager:
                sl_executed = await self._stop_loss_manager.check_and_execute_stop_loss(
                    symbol=symbol,
                    strategy_name="scalping",
                    current_price=current_price,
                    position_state=state
                )
                if not sl_executed:
                    # StopLossManager返回False时降级到直接平仓
                    await self._close_position(symbol, "stop_loss")
            else:
                await self._close_position(symbol, "stop_loss")

    async def _check_profit_protection(self, symbol: str, current_price: float):
        state = self._positions[symbol]
        direction = state["direction"]
        entry_price = state["entry_price"]
        
        if not check_profit_protection(entry_price, current_price, direction, self._profit_protection_max_drawdown):
            await self._close_position(symbol, "profit_protection")
            logger.info(f"Profit protection triggered: {symbol}, current_price: {current_price:.4f}")

    async def _check_time_limit(self, symbol: str):
        """生产级时间退出：支持部分平仓+强制全平"""
        if not self._time_exit_enabled:
            # 回退到旧版简单超时逻辑
            state = self._positions[symbol]
            elapsed = (datetime.now() - state["entry_time"]).total_seconds() / 60
            if elapsed >= self._max_hold_minutes:
                await self._close_position(symbol, "timeout")
            return

        state = self._positions[symbol]
        elapsed_hours = (datetime.now() - state["entry_time"]).total_seconds() / 3600

        # 阶段1：超过 time_exit_after_hours，部分平仓
        if elapsed_hours >= self._time_exit_after_hours:
            if not state.get("_time_partial_closed"):
                close_qty = state["quantity"] * self._time_exit_partial_pct
                await self._close_partial(symbol, close_qty, "time_exit_partial")
                state["_time_partial_closed"] = True
                logger.info(f"Time exit partial: {symbol}, closed {self._time_exit_partial_pct*100:.0f}% after {elapsed_hours:.1f}h")

        # 阶段2：超过 max_hold_hours，强制全平
        if elapsed_hours >= self._max_hold_hours:
            await self._close_position(symbol, "time_exit_full")
            logger.info(f"Time exit full: {symbol}, force closed after {elapsed_hours:.1f}h")

        # 回退：如果 max_hold_hours 未配置，使用旧版 max_hold_minutes
        if self._max_hold_hours <= 0:
            elapsed_min = elapsed_hours * 60
            if elapsed_min >= self._max_hold_minutes:
                await self._close_position(symbol, "timeout")
    
    async def _check_momentum_decay(self, symbol: str, current_price: float):
        state = self._positions[symbol]
        direction = state["direction"]
        
        # P4: 动量衰减退出冷却 - 同symbol 30秒内不重复触发
        now_ts = time.time()
        last_momentum_exit = state.get("_last_momentum_exit_ts", 0)
        if now_ts - last_momentum_exit < 30:
            return
        
        # P4: 最小数量保护 - 防止无限递归平仓
        min_close_qty = max(self._min_quantity_threshold * 3, state.get("_initial_quantity", state["quantity"]) * 0.05)
        if state["quantity"] <= min_close_qty:
            return
        
        momentum = self._momentum_cache.get(symbol)
        if not momentum:
            return
        
        rsi_1m = momentum.get("rsi_1m", 50)
        macd = momentum.get("macd", 0)
        macd_signal = momentum.get("macd_signal", 0)
        
        if direction == "long":
            if rsi_1m > 75 or (macd < macd_signal and macd_signal > 0):
                if state["current_profit"] > 0:
                    state["_last_momentum_exit_ts"] = now_ts
                    await self._close_partial(symbol, state["quantity"] * 0.5, "momentum_decay")
                    logger.info(f"Momentum decay exit: {symbol}, closed 50% at {current_price:.4f}")
        else:
            if rsi_1m < 25 or (macd > macd_signal and macd_signal < 0):
                if state["current_profit"] > 0:
                    state["_last_momentum_exit_ts"] = now_ts
                    await self._close_partial(symbol, state["quantity"] * 0.5, "momentum_decay")
                    logger.info(f"Momentum decay exit: {symbol}, closed 50% at {current_price:.4f}")
    
    async def _check_volatility_based_exit(self, symbol: str, current_price: float):
        state = self._positions[symbol]
        direction = state["direction"]
        
        atr = self._get_atr(symbol)
        if atr == 0:
            return
        
        entry_price = state["entry_price"]
        price_range = atr * 3
        
        if direction == "long":
            if current_price > entry_price + price_range * 0.5 and state["current_profit"] > 0:
                if state["peak_price"] - current_price > atr * 0.8:
                    await self._close_partial(symbol, state["quantity"] * 0.3, "volatility_taking")
                    logger.info(f"Volatility profit taking: {symbol}, closed 30% at {current_price:.4f}")
        else:
            if current_price < entry_price - price_range * 0.5 and state["current_profit"] > 0:
                if current_price - state["trough_price"] > atr * 0.8:
                    await self._close_partial(symbol, state["quantity"] * 0.3, "volatility_taking")
                    logger.info(f"Volatility profit taking: {symbol}, closed 30% at {current_price:.4f}")

    async def _check_multiple_take_profit(self, symbol: str, current_price: float):
        """生产级多级止盈：tp1部分止盈 -> tp2部分止盈 -> tp3移动止损"""
        if not self._take_profit_enabled:
            return

        state = self._positions[symbol]
        direction = state["direction"]
        entry_price = state["entry_price"]
        current_profit = state["current_profit"]

        # 防止重复触发
        tp_state = state.setdefault("_tp_state", {"tp1_done": False, "tp2_done": False, "tp3_done": False})

        # TP1: 部分止盈
        if not tp_state["tp1_done"] and self._tp1_ratio > 0 and self._tp1_pct > 0:
            if current_profit >= self._tp1_pct:
                close_qty = state["quantity"] * self._tp1_ratio
                await self._close_partial(symbol, close_qty, "tp1")
                tp_state["tp1_done"] = True
                logger.info(f"Multi-TP Level 1: {symbol}, closed {self._tp1_ratio*100:.0f}% at +{self._tp1_pct*100:.2f}% profit")

        # TP2: 第二部分止盈
        if not tp_state["tp2_done"] and self._tp2_ratio > 0 and self._tp2_pct > 0:
            if current_profit >= self._tp2_pct:
                remaining_qty = state["quantity"]
                close_qty = remaining_qty * self._tp2_ratio
                await self._close_partial(symbol, close_qty, "tp2")
                tp_state["tp2_done"] = True
                logger.info(f"Multi-TP Level 2: {symbol}, closed {self._tp2_ratio*100:.0f}% at +{self._tp2_pct*100:.2f}% profit")

        # TP3: 移动止损跟踪剩余仓位
        if not tp_state["tp3_done"] and self._tp3_ratio > 0 and self._tp3_trailing_pct > 0:
            if current_profit >= self._tp2_pct:  # TP2触发后启动TP3跟踪
                if direction == "long":
                    trailing_sl = current_price * (1 - self._tp3_trailing_pct)
                    if trailing_sl > state["stop_loss"]:
                        state["stop_loss"] = trailing_sl
                else:
                    trailing_sl = current_price * (1 + self._tp3_trailing_pct)
                    if trailing_sl < state["stop_loss"]:
                        state["stop_loss"] = trailing_sl

    async def _check_breakeven_stop(self, symbol: str, current_price: float):
        """生产级保本止损：利润达到触发阈值后，止损移至成本价+安全边际"""
        state = self._positions[symbol]
        direction = state["direction"]
        entry_price = state["entry_price"]
        current_profit = state["current_profit"]

        # 防止重复设置
        if state.get("_breakeven_set"):
            return

        if current_profit >= self._breakeven_trigger_pct:
            precision = get_price_precision(symbol)
            if direction == "long":
                breakeven_sl = round(entry_price * (1 + self._breakeven_stop_pct), precision)
                if breakeven_sl > state["stop_loss"]:
                    state["stop_loss"] = breakeven_sl
                    state["_breakeven_set"] = True
                    logger.info(f"Breakeven stop set: {symbol}, SL moved to {breakeven_sl:.4f} (+{self._breakeven_stop_pct*100:.2f}%)")
            else:
                breakeven_sl = round(entry_price * (1 - self._breakeven_stop_pct), precision)
                if breakeven_sl < state["stop_loss"]:
                    state["stop_loss"] = breakeven_sl
                    state["_breakeven_set"] = True
                    logger.info(f"Breakeven stop set: {symbol}, SL moved to {breakeven_sl:.4f} (-{self._breakeven_stop_pct*100:.2f}%)")

    async def _check_volatility_spike_exit(self, symbol: str, current_price: float):
        """生产级波动率尖峰退出：检测异常波动率飙升，部分平仓+锁仓"""
        if not self._volatility_stop_enabled:
            return

        # 检查锁仓期
        lockout_until = self._volatility_lockout_until.get(symbol)
        if lockout_until and datetime.now() < lockout_until:
            return

        state = self._positions[symbol]
        atr = self._get_atr(symbol)
        if atr <= 0 or state["entry_price"] <= 0:
            return

        # 计算当前波动率 vs 历史波动率
        entry_price = state["entry_price"]
        current_atr_pct = atr / current_price

        # 历史ATR基准（用入场时的ATR近似）
        klines_1m = self._last_klines.get(symbol, {}).get("1m", [])
        if len(klines_1m) < 30:
            return

        try:
            historical_atr_values = []
            for i in range(min(30, len(klines_1m))):
                h = float(klines_1m[i][2])
                l = float(klines_1m[i][3])
                c = float(klines_1m[i][4])
                historical_atr_values.append((h - l) / c if c > 0 else 0)

            avg_historical_atr_pct = np.mean(historical_atr_values) if historical_atr_values else current_atr_pct
            if avg_historical_atr_pct <= 0:
                return

            vol_ratio = current_atr_pct / avg_historical_atr_pct

            if vol_ratio >= self._vol_spike_threshold:
                close_qty = state["quantity"] * self._vol_stop_partial_pct
                await self._close_partial(symbol, close_qty, "vol_spike")
                # 设置锁仓期
                self._volatility_lockout_until[symbol] = datetime.now() + timedelta(minutes=self._volatility_lockout_minutes)
                logger.warning(
                    f"Volatility spike exit: {symbol}, vol_ratio={vol_ratio:.1f}x > {self._vol_spike_threshold:.1f}x, "
                    f"closed {self._vol_stop_partial_pct*100:.0f}%, lockout {self._volatility_lockout_minutes}min"
                )
        except (ValueError, IndexError) as e:
            pass

    async def _close_partial(self, symbol: str, quantity: float, reason: str):
        state = self._positions.get(symbol)
        if not state:
            return
        
        # 验证交易所实际持仓，防止幽灵仓位
        if not await self._verify_exchange_position(symbol, state["direction"]):
            logger.warning(f"Scalping ghost position detected for {symbol}, cleaning up internal state")
            self.cleanup_position(symbol, "scalping")
            return
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        price = float(ticker["last"])
        direction = state["direction"]
        close_direction = "sell" if direction == "long" else "buy"
        
        signal = Signal(
            symbol=symbol,
            strategy_name="scalping",
            signal_type=f"scalping_close_{reason}",
            direction=close_direction,
            price=price,
            quantity=quantity,
            leverage=state["leverage"],
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
                "exit_reason": reason,
                "direction": signal.direction,
                "price": signal.price,
                "quantity": signal.quantity,
                "leverage": signal.leverage,
                "stop_loss": None,
                "take_profit": None,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat()
            }
        })
        
        state["quantity"] -= quantity
        state["partial_closes"].append({
            "price": price,
            "quantity": quantity,
            "reason": reason,
            "timestamp": datetime.now()
        })
        
        if state["quantity"] <= self._min_quantity_threshold:
            state["status"] = "closed"
            state["close_price"] = price
            state["close_time"] = datetime.now()
        
        pnl = (price - state["entry_price"]) / state["entry_price"] if direction == "long" else (state["entry_price"] - price) / state["entry_price"]
        pnl_usdt = pnl * quantity * state["entry_price"]
        self._update_daily_pnl(pnl_usdt)
        self._update_signal_performance(self._position_signal_type.get(symbol, "momentum"), pnl_usdt)
        # 平仓后立即失效缓存，防止后续操作误判为幽灵仓位
        self._invalidate_position_cache(symbol, direction)
        logger.info(f"Scalping partial close: {symbol}, reason: {reason}, qty: {quantity:.4f}, price: {price:.4f}, PnL: {pnl_usdt:.2f} USDT ({pnl:.2%})")

    def _calculate_max_loss_price(self, entry_price: float, direction: str, quantity: float, max_loss_usdt: float) -> float:
        """根据单笔最大亏损USDT反算止损价格"""
        try:
            if quantity <= 0 or entry_price <= 0 or max_loss_usdt <= 0:
                return 0
            # max_loss_usdt = |price - entry_price| * quantity
            # => |price - entry_price| = max_loss_usdt / quantity
            loss_per_unit = max_loss_usdt / quantity
            if direction == "long":
                return entry_price - loss_per_unit
            else:
                return entry_price + loss_per_unit
        except Exception:
            return 0

    async def _close_position(self, symbol: str, reason: str):
        state = self._positions.get(symbol)
        if not state:
            return
        
        # 验证交易所实际持仓，防止幽灵仓位
        if not await self._verify_exchange_position(symbol, state["direction"]):
            logger.warning(f"Scalping ghost position detected for {symbol}, cleaning up internal state")
            self.cleanup_position(symbol, "scalping")
            return
        
        ticker = await self._get_ticker_cached(symbol)
        if not ticker:
            return
        
        price = float(ticker["last"])
        direction = state["direction"]
        close_direction = "sell" if direction == "long" else "buy"
        
        pnl = (price - state["entry_price"]) / state["entry_price"] if direction == "long" else (state["entry_price"] - price) / state["entry_price"]
        pnl_usdt = pnl * state["quantity"] * state["entry_price"]
        
        signal = Signal(
            symbol=symbol,
            strategy_name="scalping",
            signal_type=f"scalping_close_{reason}",
            direction=close_direction,
            price=price,
            quantity=state["quantity"],
            leverage=state["leverage"],
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
                "exit_reason": reason,
                "direction": signal.direction,
                "price": signal.price,
                "quantity": signal.quantity,
                "leverage": signal.leverage,
                "stop_loss": None,
                "take_profit": None,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat()
            }
        })
        
        state["status"] = "closed"
        state["close_price"] = price
        state["close_time"] = datetime.now()
        state["pnl"] = pnl
        state["pnl_usdt"] = pnl_usdt
        
        self._update_daily_pnl(pnl_usdt)
        self._update_signal_performance(self._position_signal_type.get(symbol, "momentum"), pnl_usdt)
        
        # 企业级增强：出场埋点（按出场原因分类统计 + PnL 指标）
        self._record_metric("scalping_position_closed_total", 1.0,
                            {"reason": reason, "symbol": symbol, "direction": direction})
        self._record_metric("scalping_position_pnl", pnl_usdt,
                            {"reason": reason, "symbol": symbol})
        
        # 平仓后立即失效缓存，防止后续操作误判为幽灵仓位
        self._invalidate_position_cache(symbol, direction)
        
        logger.info(f"Scalping closed: {symbol}, reason: {reason}, price: {price:.4f}, PnL: {pnl_usdt:.2f} USDT ({pnl:.2%})")

    async def _close_all_positions(self):
        for symbol in list(self._positions.keys()):
            await self._close_position(symbol, "scheduled")

    def cleanup_position(self, symbol: str, strategy_name: str = ""):
        """清理指定symbol的持仓状态（用于订单执行失败时的状态同步）
        
        P5增强：同步清理数据库记录，防止幽灵仓位反复出现
        """
        # ghost_close 专项 Phase 2: 同步清理 pending intent，避免订单失败后阻塞 symbol
        self._pending_entries.pop(symbol, None)

        if symbol in self._positions:
            direction = self._positions[symbol].get("direction", "")
            del self._positions[symbol]
            logger.info(f"Scalping cleaned up ghost position for {symbol}")
            
            # P5: 同步清理数据库中的开仓记录
            try:
                if hasattr(self, 'sqlite_storage') and self.sqlite_storage:
                    # 将该symbol的开仓记录标记为已关闭
                    self.sqlite_storage.close_open_record(symbol, "scalping")
            except Exception as e:
                logger.debug(f"Failed to close DB record for ghost position {symbol}: {e}")
            
            # P5: 同步清理trade_journal
            try:
                if hasattr(self, 'trade_journal') and self.trade_journal:
                    self.trade_journal.mark_position_closed(symbol, "scalping", reason="ghost_close")
            except Exception as e:
                logger.debug(f"Failed to sync trade_journal for ghost position {symbol}: {e}")
            
            # P5: 失效缓存
            if direction:
                self._invalidate_position_cache(symbol, direction)

    def on_order_filled(self, fill_payload: Dict[str, Any]):
        """成交回执回调（ghost_close 专项 Phase 2）：匹配 pending intent 并物化仓位。

        仅处理 scalping 开仓成交；按 clOrdId 精确匹配，symbol 作为兜底（剥头皮单币种单仓）。
        """
        try:
            strategy_name = fill_payload.get("strategy_name", "")
            if strategy_name != "scalping":
                return
            if not self._use_fill_driven_position:
                return

            symbol = fill_payload.get("symbol", "")
            clordid = fill_payload.get("clOrdId", "")
            try:
                filled_qty = float(fill_payload.get("filled_qty", 0) or 0)
            except (TypeError, ValueError):
                filled_qty = 0.0
            try:
                avg_price = float(fill_payload.get("avg_price", 0) or 0)
            except (TypeError, ValueError):
                avg_price = 0.0

            pending = self._pending_entries.pop(symbol, None)
            if pending is None:
                return

            # clOrdId 精确匹配；不匹配则说明是旧单/其他信号，恢复 pending
            if clordid and pending.get("clOrdId") and clordid != pending.get("clOrdId"):
                self._pending_entries[symbol] = pending
                return

            direction = pending.get("direction", fill_payload.get("direction", ""))
            qty = filled_qty if filled_qty > 0 else pending.get("quantity", 0)
            entry_price = avg_price if avg_price > 0 else pending.get("entry_price", 0)

            self._positions[symbol] = {
                "direction": direction,
                "entry_price": entry_price,
                "quantity": qty,
                "leverage": pending.get("leverage", 1),
                "stop_loss": pending.get("stop_loss"),
                "take_profit": pending.get("take_profit"),
                "entry_time": datetime.now(),
                "status": "open",
                "peak_price": entry_price if direction == "long" else 0,
                "trough_price": entry_price,
                "trailing_stop_initial": pending.get("stop_loss"),
                "vwap_entry": pending.get("vwap_entry", entry_price),
                "current_profit": 0.0,
                "partial_closes": [],
            }
            logger.info(
                f"[ghost_close-P2] position materialized: {symbol} {direction} "
                f"qty={qty} avg_px={entry_price} clOrdId={clordid}"
            )
            self._fill_callback_hits += 1

            # P32: 日内交易计数——成交后才递增，避免信号空烧额度
            today = datetime.now().date()
            if self._daily_trade_reset_date != today:
                self._daily_trade_count = 0
                self._daily_trade_reset_date = today
            self._daily_trade_count += 1
            logger.info(
                f"P32: Scalping fill counted {symbol} {direction} "
                f"(daily: {self._daily_trade_count}/"
                f"{self._max_daily_trades if self._max_daily_trades > 0 else '∞'})"
            )
        except Exception as e:
            logger.debug(f"on_order_filled error: {e}")

    def get_health(self) -> Dict[str, Any]:
        """企业级健康状态：暴露自适应参数与过滤埋点统计。"""
        open_positions = sum(1 for p in self._positions.values() if p["status"] == "open")
        return {
            "strategy": "scalping",
            "is_active": self._is_active,
            "open_positions": open_positions,
            "adaptive_enabled": self._adaptive_enabled,
            "min_signal_quality": self._min_signal_quality,
            "dynamic_min_quality": self._dynamic_min_quality(),
            "risk_lock_quality_boost": self._risk_lock_quality_boost,
            "filter_stats": dict(self._filter_stats),
            "daily_pnl": self._daily_pnl,
            "current_drawdown": self._current_drawdown,
        }

    def get_stats(self) -> Dict[str, Any]:
        open_positions = sum(1 for p in self._positions.values() if p["status"] == "open")
        closed_positions = sum(1 for p in self._positions.values() if p["status"] == "closed")
        
        total_pnl = sum(p.get("pnl_usdt", 0) for p in self._positions.values() if p["status"] == "closed")
        win_count = sum(1 for p in self._positions.values() if p["status"] == "closed" and p.get("pnl", 0) > 0)
        loss_count = sum(1 for p in self._positions.values() if p["status"] == "closed" and p.get("pnl", 0) <= 0)
        
        win_rate = win_count / (win_count + loss_count) if (win_count + loss_count) > 0 else 0
        
        winning_pnl = sum(p.get("pnl_usdt", 0) for p in self._positions.values() if p["status"] == "closed" and p.get("pnl", 0) > 0)
        losing_pnl = sum(p.get("pnl_usdt", 0) for p in self._positions.values() if p["status"] == "closed" and p.get("pnl", 0) <= 0)
        
        avg_win = winning_pnl / win_count if win_count > 0 else 0
        avg_loss = abs(losing_pnl) / loss_count if loss_count > 0 else 0
        # 无亏损时盈亏比无数学意义，用 None 表示「不适用」，避免 float('inf') 污染 JSON 输出
        profit_factor = avg_win / avg_loss if avg_loss > 0 else None
        
        active_symbols = [s for s in self._scalping_symbols if s in self._momentum_cache]
        
        return {
            "strategy": "scalping",
            "is_active": self._is_active,
            "open_positions": open_positions,
            "closed_positions": closed_positions,
            "total_pnl": total_pnl,
            "win_count": win_count,
            "loss_count": loss_count,
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "monitored_symbols": len(active_symbols),
            "daily_pnl": self._daily_pnl,
            "current_drawdown": self._current_drawdown,
            "pending_entry_count": len(self._pending_entries),
            "fill_callback_hits": self._fill_callback_hits,
            "pending_timeout_cleaned": self._pending_timeout_cleaned,
            "ghost_close_count": sum(1 for p in self._positions.values() if p.get("close_reason") == "ghost_position_cleaned"),
        }

    # ===== P0-5: 状态持久化 =====

    def collect_persistent_state(self) -> Dict[str, Any]:
        """收集需要持久化的策略状态（持仓 + 日内 PnL 基准 + 回撤）。"""
        # 将 volatility_lockout_until 的 datetime 值序列化为字符串
        lockout_serialized = {}
        for sym, dt in self._volatility_lockout_until.items():
            lockout_serialized[sym] = dt.isoformat()
        return {
            "positions": self._positions,
            "daily_pnl": self._daily_pnl,
            "daily_start_equity": self._daily_start_equity,
            "max_daily_equity": self._max_daily_equity,
            "current_drawdown": self._current_drawdown,
            "daily_reset_date": self._daily_reset_date.isoformat() if self._daily_reset_date else None,
            "signal_type_performance": self._signal_type_performance,
            "position_signal_type": self._position_signal_type,
            "volatility_lockout_until": lockout_serialized,
        }

    def restore_persistent_state(self, state: Dict[str, Any]):
        """从持久化数据恢复策略状态。"""
        try:
            if "positions" in state and isinstance(state["positions"], dict):
                # 仅恢复仍处于 open 状态的持仓，closed 持仓无意义
                restored = {}
                for sym, pos in state["positions"].items():
                    if isinstance(pos, dict) and pos.get("status") == "open":
                        # 还原 datetime 字段（JSON 序列化时变为字符串）
                        # 注意：peak_price/trough_price 是价格（float），不是时间，不应在此解析
                        for key in ("entry_time", "close_time"):
                            if key in pos and isinstance(pos[key], str):
                                try:
                                    pos[key] = datetime.fromisoformat(pos[key])
                                except (ValueError, TypeError):
                                    pass
                        restored[sym] = pos
                self._positions = restored
                logger.info(f"Scalping restored {len(restored)} open positions")

            self._daily_pnl = float(state.get("daily_pnl", 0.0))
            self._daily_start_equity = float(state.get("daily_start_equity", 0.0))
            self._max_daily_equity = float(state.get("max_daily_equity", 0.0))
            self._current_drawdown = float(state.get("current_drawdown", 0.0))
            
            if self._current_drawdown < 0 or self._current_drawdown > 1.0:
                self._current_drawdown = 0.0

            reset_date_str = state.get("daily_reset_date")
            if reset_date_str:
                try:
                    self._daily_reset_date = datetime.fromisoformat(reset_date_str).date()
                except (ValueError, TypeError):
                    self._daily_reset_date = None

            if "signal_type_performance" in state and isinstance(state["signal_type_performance"], dict):
                self._signal_type_performance = state["signal_type_performance"]
                # P0: 补齐旧状态缺失的企业级追踪字段
                for _sig_type, _perf in self._signal_type_performance.items():
                    for _key in ("count", "pnl_list", "consecutive_wins", "consecutive_losses", "peak_equity", "max_drawdown"):
                        if _key not in _perf:
                            _perf[_key] = [] if _key == "pnl_list" else 0
            if "position_signal_type" in state and isinstance(state["position_signal_type"], dict):
                self._position_signal_type = state["position_signal_type"]

            # 恢复波动率锁仓状态
            if "volatility_lockout_until" in state and isinstance(state["volatility_lockout_until"], dict):
                self._volatility_lockout_until = {}
                for sym, dt_str in state["volatility_lockout_until"].items():
                    try:
                        lockout_dt = datetime.fromisoformat(dt_str)
                        if lockout_dt > datetime.now():
                            self._volatility_lockout_until[sym] = lockout_dt
                    except (ValueError, TypeError):
                        pass
                if self._volatility_lockout_until:
                    logger.info(f"Scalping restored {len(self._volatility_lockout_until)} volatility lockouts")
            
            self._equity_initialized = False
        except Exception as e:
            logger.error(f"Scalping restore_persistent_state failed: {e}")