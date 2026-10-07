"""趋势跟踪主策略：基于均线/突破等多周期确认信号开仓，支持分批加仓与移动止损。"""
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from core.models import Signal, BarData, Position
from configs.settings import get_currency_tier
from utils.helpers import calculate_take_profit, calculate_stop_loss, validate_tp_sl_prices, calculate_tp_sl_from_atr, calculate_trailing_stop, calculate_scaled_take_profit, adjust_stop_loss_for_volatility, check_profit_protection, detect_market_state, calculate_adaptive_position_size, evaluate_signal_quality, get_price_precision, map_market_state_to_regime
from utils.state_persistence import PersistentStrategy
from risk.dynamic_allocator import AdaptiveKelly
from strategies.trend_sub_strategies import (
    evaluate_ma_trend,
    evaluate_donchian,
    macd_zero_axis,
    rolling_return,
    rank_momentum,
)
from strategies.funding_rate_enhancer import FundingRateEnhancer
from services.trend_vote import compute_trend_vote

class TrendStrategy(PersistentStrategy):
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        # P0-4: 初始化 PersistentStrategy 基类
        super().__init__()
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache
        # P-资金费率: 统一挂载资金费率增强模块，替换旧的内联费率逻辑
        self._funding_enhancer = FundingRateEnhancer(config, okx_client)
        
        self._enabled = config["strategies"]["trend"]["enabled"]
        self._confirmation_periods = config["strategies"]["trend"]["confirmation_periods"]
        self._max_additions = config["strategies"]["trend"]["max_additions"]
        self._initial_position_ratio = config["strategies"]["trend"]["initial_position_ratio"]
        self._addition_ratio = config["strategies"]["trend"]["addition_ratio"]
        # P0: 最大并发持仓数 - 防止资金过度分散，历史数据显示12个持仓无法有效管理
        self._max_concurrent_positions = config["strategies"]["trend"].get("max_concurrent_positions", 6)
        self._profit_targets = config["strategies"]["trend"]["profit_targets"]
        self._trailing_stop_tier1 = config["strategies"]["trend"]["trailing_stop_tier1"]
        self._trailing_stop_tier2 = config["strategies"]["trend"]["trailing_stop_tier2"]
        self._trailing_stop_tier3 = config["strategies"]["trend"]["trailing_stop_tier3"]
        self._false_break_threshold = config["strategies"]["trend"]["false_break_threshold"]
        
        self._trailing_stop_initial = config["strategies"]["trend"].get("trailing_stop_initial", 0.02)
        self._trailing_stop_min = config["strategies"]["trend"].get("trailing_stop_min", 0.005)
        self._volatility_adaptive_sl = config["strategies"]["trend"].get("volatility_adaptive_sl", True)
        self._max_stop_loss_pct = config["strategies"]["trend"].get("max_stop_loss_pct", 0.025)
        self._profit_protection_max_drawdown = config["strategies"]["trend"].get("profit_protection_max_drawdown", 0.5)
        self._reversal_confirmation_bars = config["strategies"]["trend"].get("reversal_confirmation_bars", 2)
        self._reversal_adx_threshold = config["strategies"]["trend"].get("reversal_adx_threshold", 30)
        # 最小持仓时长（分钟）：避免闪开闪平吃手续费
        self._min_hold_minutes = config.get("trading", {}).get("min_hold_minutes", 5)
        
        self._adaptive_enabled = config["strategies"]["trend"].get("adaptive_enabled", True)
        # P32: 降低信号质量门槛，从0.5降至0.35，激活趋势策略信号产出
        self._min_signal_quality = config["strategies"]["trend"].get("min_signal_quality", 0.35)
        self._market_state_lookback = config["strategies"]["trend"].get("market_state_lookback", 20)
        
        self._breakout_confirmation = config["strategies"]["trend"].get("breakout_confirmation", True)
        self._multi_timeframe_confirmation = config["strategies"]["trend"].get("multi_timeframe_confirmation", True)
        self._trend_strength_filter = config["strategies"]["trend"].get("trend_strength_filter", True)
        
        # P32: 降低趋势强度阈值，从0.3降至0.2，允许弱趋势入场
        self._min_trend_strength = config["strategies"]["trend"].get("min_trend_strength", 0.2)
        self._breakout_volume_factor = config["strategies"]["trend"].get("breakout_volume_factor", 1.5)
        self._pullback_entry = config["strategies"]["trend"].get("pullback_entry", True)
        self._pullback_depth = config["strategies"]["trend"].get("pullback_depth", 0.005)

        # 企业级趋势开单强化：软加权门控（把 MTF/量/ADX 硬否决改为综合评分，减少误杀）
        #   min_confirmation_count: MTF/量/ADX 三项至少通过几项（默认 1，替代原"三项全过"）
        #   confidence_threshold: 融合后置信度门槛（默认 0.30，略高于原 0.20 以补偿放宽的门控）
        #   detector_confidence_weight: 检测器置信度在融合中的权重（0-1，剩余给多因子软评分）
        self._min_confirmation_count = config["strategies"]["trend"].get("min_confirmation_count", 1)
        self._confidence_threshold = config["strategies"]["trend"].get("confidence_threshold", 0.30)
        self._detector_confidence_weight = config["strategies"]["trend"].get("detector_confidence_weight", 0.4)

        # 共享趋势判断投票（services/trend_vote）：与 MarketRegimeEngine 复用同一套
        # ADX+结构+EMA排列+DI 投票，校准 _detect_trend 的方向与置信度
        self._shared_trend_vote_enabled = config["strategies"]["trend"].get("shared_trend_vote_enabled", True)
        self._shared_trend_vote_weight = config["strategies"]["trend"].get("shared_trend_vote_weight", 0.15)
        self._shared_vote_adx_floor = config["strategies"]["trend"].get("shared_vote_adx_floor", 15.0)
        self._shared_vote_adx_saturation = config["strategies"]["trend"].get("shared_vote_adx_saturation", 40.0)
        # 等位挂单：post_only 开启后，开仓用 post_only 限价单（maker 单）挂在盘口 maker 侧，避免 taker 手续费与追单
        self._post_only = config["strategies"]["trend"].get("post_only", False)
        
        self._trend_performance: Dict[str, Any] = {
            "wins": 0, "losses": 0, "total_pnl": 0, "count": 0,
            "pnl_list": [], "total_profit": 0, "total_loss": 0,
            "consecutive_wins": 0, "consecutive_losses": 0,
            "peak_equity": 0, "max_drawdown": 0,
        }
        # P33: 企业级凯利引擎（贝叶斯胜率收缩 + 赔率收缩 + 半方差 + 负偏度惩罚）
        self._adaptive_kelly = AdaptiveKelly(config.get("adaptive_kelly", {}))
        self._consecutive_periods_required = config["strategies"]["trend"].get("consecutive_periods", 2)
        
        # === 生产级 v2.0 参数 ===
        # 多级止盈
        self._take_profit_enabled = config["strategies"]["trend"].get("take_profit_enabled", True)
        self._tp1_ratio = config["strategies"]["trend"].get("tp1_ratio", 0.4)
        self._tp1_pct = config["strategies"]["trend"].get("tp1_pct", 0.6)
        self._tp2_ratio = config["strategies"]["trend"].get("tp2_ratio", 0.5)
        self._tp2_pct = config["strategies"]["trend"].get("tp2_pct", 1.0)
        self._tp3_ratio = config["strategies"]["trend"].get("tp3_ratio", 0.1)
        self._tp3_trailing_pct = config["strategies"]["trend"].get("tp3_trailing_pct", 0.02)
        # 时间退出
        self._time_exit_enabled = config["strategies"]["trend"].get("time_exit_enabled", True)
        self._max_hold_hours = config["strategies"]["trend"].get("max_hold_hours", 72.0)
        self._time_exit_after_hours = config["strategies"]["trend"].get("time_exit_after_hours", 48.0)
        self._time_exit_partial_pct = config["strategies"]["trend"].get("time_exit_partial_pct", 0.5)
        # 波动率退出
        self._volatility_stop_enabled = config["strategies"]["trend"].get("volatility_stop_enabled", True)
        self._vol_spike_threshold = config["strategies"]["trend"].get("vol_spike_threshold", 2.0)
        self._vol_stop_partial_pct = config["strategies"]["trend"].get("vol_stop_partial_pct", 0.3)
        self._volatility_lockout_minutes = config["strategies"]["trend"].get("volatility_lockout_minutes", 30)
        # 动态ADX
        self._dynamic_adx_threshold = config["strategies"]["trend"].get("dynamic_adx_threshold", True)
        self._adx_vol_high = max(20, self._safe_float(config["strategies"]["trend"].get("adx_vol_high", 30), 30))
        self._adx_vol_low = max(20, self._safe_float(config["strategies"]["trend"].get("adx_vol_low", 20), 20))
        self._adx_vol_normal = max(20, self._safe_float(config["strategies"]["trend"].get("adx_vol_normal", 25), 25))
        # Chandelier Exit
        self._chandelier_exit_enabled = config["strategies"]["trend"].get("chandelier_exit_enabled", True)
        self._chandelier_atr_period = config["strategies"]["trend"].get("chandelier_atr_period", 22)
        self._chandelier_multiplier_base = config["strategies"]["trend"].get("chandelier_multiplier_base", 3.0)
        self._chandelier_profit_threshold = config["strategies"]["trend"].get("chandelier_profit_threshold", 0.02)
        # 背离检测
        self._divergence_detection = config["strategies"]["trend"].get("divergence_detection", True)
        self._divergence_lookback_bars = config["strategies"]["trend"].get("divergence_lookback_bars", 30)
        self._divergence_min_strength = config["strategies"]["trend"].get("divergence_min_strength", 0.6)
        # 市场结构
        self._market_structure_enabled = config["strategies"]["trend"].get("market_structure_enabled", True)
        self._min_swing_points = config["strategies"]["trend"].get("min_swing_points", 3)
        self._swing_window = config["strategies"]["trend"].get("swing_window", 5)
        
        self._market_state: Dict[str, Dict[str, Any]] = {}
        
        self._rsi_period = 14
        self._rsi_overbought = 70
        self._rsi_oversold = 30
        self._adx_period = 14
        self._adx_threshold = 25
        self._atr_period = 14
        self._atr_multiplier = 2.0
        
        self._position_state: Dict[str, Dict[str, Any]] = {}
        # ghost_close 专项 Phase 4: 待确认开仓意图（成交回执驱动仓位写入）
        self._pending_entries: Dict[str, Dict[str, Any]] = {}
        self._use_fill_driven_position = config["strategies"]["trend"].get("use_fill_driven_position", False)
        # ghost_close 专项 Phase 4: 观测指标
        self._fill_callback_hits = 0
        self._pending_timeout_cleaned = 0
        self._last_signal_time: Dict[str, datetime] = {}
        self._last_cleanup_time = datetime.min
        self._last_exchange_sync_time = datetime.min  # P7-1: 交易所持仓同步时间戳
        # 信号饥饿诊断：周期性汇总开单漏斗各层通过情况，定位"长时间不开单"瓶颈
        self._signal_starvation_stats: Dict[str, int] = {
            "scanned": 0, "detected": 0, "macd_pass": 0,
            "mtf_pass": 0, "vol_pass": 0, "adx_pass": 0, "generated": 0,
        }
        self._starvation_last_log = datetime.min
        self._starvation_log_interval = config["strategies"]["trend"].get("starvation_log_interval", 1800)
        self._indicator_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._dx_cache: Dict[str, List[float]] = {}  # ADX计算缓存：每个symbol的dx历史序列
        # 生产级状态变量
        self._volatility_lockout_until: Dict[str, datetime] = {}  # 波动率尖峰后锁仓
        self._position_entry_times: Dict[str, datetime] = {}      # 持仓入场时间（用于时间退出）
        # 动量排名缓存（多品种相对强弱，减少 4h K线 API 调用）
        self._momentum_rank_cache: Dict[str, float] = {}
        self._momentum_rank_ts: float = 0.0
        self._momentum_rank_ttl: float = 1800.0  # 30分钟刷新一次

        # === 企业级趋势子策略配置（均线/唐奇安/MACD零轴/动量） ===
        # 主评估周期：均线/唐奇安/MACD 在此周期上计算（中长线基准）
        self._sub_trend_bar = config["strategies"]["trend"].get("sub_strategy_bar", "1H")
        self._sub_ma_atr_threshold = config["strategies"]["trend"].get("sub_ma_atr_threshold", 0.003)
        self._sub_donchian_period = config["strategies"]["trend"].get("sub_donchian_period", 20)
        self._sub_donchian_atr_sl = config["strategies"]["trend"].get("sub_donchian_atr_sl", 2.0)
        self._sub_donchian_atr_tp = config["strategies"]["trend"].get("sub_donchian_atr_tp", 3.0)
        self._sub_momentum_top_n = config["strategies"]["trend"].get("sub_momentum_top_n", 3)
        self._sub_momentum_bottom_n = config["strategies"]["trend"].get("sub_momentum_bottom_n", 3)
        
        self._all_symbols = []
        # 小账户(<2000 USDT)下 tier1(BTC/ETH) 最小下单保证金过高(约154 USDT)远超单仓额度，
        # 趋势类频繁小单策略跳过 tier1，聚焦 tier2/tier3 低价币（阈值与 signal_processor 一致）
        _total_capital = float(config.get("trading", {}).get("total_capital", 0) or 0)
        _btc_min_equity = float(config.get("trading", {}).get("high_value_equity_threshold", 2000.0) or 2000.0)
        _skip_high_value = _total_capital > 0 and _total_capital < _btc_min_equity
        for tier in ["tier1", "tier2", "tier3"]:
            if _skip_high_value and tier == "tier1":
                continue
            for base in config["currencies"][f"{tier}_symbols"]:
                # 使用合约格式
                self._all_symbols.append(f"{base}-USDT-SWAP")

        self._adaptive_controller = None  # 动态分配控制器（由scheduler注入）
        self._stop_loss_manager = None    # 统一止损管理器（由scheduler注入）

        # P0-4: 启用状态持久化（redis_cache 当前同步接口，StatePersistence 内部会自动降级到 JSON 文件）
        self.init_state_persistence("trend", redis_cache)

        self._capital_cache_value = 0.0
        self._capital_cache_ts = 0.0
        self._capital_cache_ttl = 30.0

    def set_adaptive_controller(self, controller):
        """注入AdaptiveController实例，用于获取动态资金分配"""
        self._adaptive_controller = controller

    def set_stop_loss_manager(self, manager):
        """注入StopLossManager实例，用于统一止损管理"""
        self._stop_loss_manager = manager

    def set_coordinator(self, coordinator):
        """注入StrategyCoordinator实例"""
        self._coordinator = coordinator

    def set_conditional_order_manager(self, manager):
        """注入ConditionalOrderManager实例，用于平仓时立即撤销条件单"""
        self._conditional_order_manager = manager

    async def update_config(self, updates: Dict[str, Any]):
        """运行时热更新策略配置（不重启策略）

        支持的配置键：max_concurrent_positions, initial_position_ratio,
        addition_ratio, max_additions, min_signal_quality, max_stop_loss_pct,
        trailing_stop_initial 等
        """
        strategy_cfg = self.config.get("strategies", {}).get("trend", {})
        strategy_cfg.update(updates)
        self.config.setdefault("strategies", {})["trend"] = strategy_cfg

        attr_map = {
            "max_concurrent_positions": "_max_concurrent_positions",
            "initial_position_ratio": "_initial_position_ratio",
            "addition_ratio": "_addition_ratio",
            "max_additions": "_max_additions",
            "min_signal_quality": "_min_signal_quality",
            "max_stop_loss_pct": "_max_stop_loss_pct",
            "trailing_stop_initial": "_trailing_stop_initial",
            "trailing_stop_min": "_trailing_stop_min",
            "take_profit_enabled": "_take_profit_enabled",
            "tp1_ratio": "_tp1_ratio", "tp2_ratio": "_tp2_ratio", "tp3_ratio": "_tp3_ratio",
            "tp1_pct": "_tp1_pct", "tp2_pct": "_tp2_pct", "tp3_trailing_pct": "_tp3_trailing_pct",
            "time_exit_enabled": "_time_exit_enabled",
            "time_exit_after_hours": "_time_exit_after_hours",
            "max_hold_hours": "_max_hold_hours",
            "time_exit_partial_pct": "_time_exit_partial_pct",
            "volatility_stop_enabled": "_volatility_stop_enabled",
            "vol_spike_threshold": "_vol_spike_threshold",
            "vol_stop_partial_pct": "_vol_stop_partial_pct",
            "volatility_lockout_minutes": "_volatility_lockout_minutes",
            "divergence_detection": "_divergence_detection",
            "market_structure_enabled": "_market_structure_enabled",
            "chandelier_exit_enabled": "_chandelier_exit_enabled",
            "chandelier_multiplier_base": "_chandelier_multiplier_base",
            "chandelier_profit_threshold": "_chandelier_profit_threshold",
            "dynamic_adx_threshold": "_dynamic_adx_threshold",
            "adx_vol_high": "_adx_vol_high",
            "adx_vol_low": "_adx_vol_low",
            "adx_vol_normal": "_adx_vol_normal",
            "false_break_threshold": "_false_break_threshold",
            "profit_protection_max_drawdown": "_profit_protection_max_drawdown",
        }
        for cfg_key, attr_name in attr_map.items():
            if cfg_key in updates:
                setattr(self, attr_name, updates[cfg_key])
                logger.info(f"Trend config hot-updated: {attr_name}={updates[cfg_key]}")

        # ADX阈值硬下限：不得低于20（历史教训：ADX<20 趋势无效）
        self._adx_vol_high = max(20, self._safe_float(self._adx_vol_high, 30))
        self._adx_vol_low = max(20, self._safe_float(self._adx_vol_low, 20))
        self._adx_vol_normal = max(20, self._safe_float(self._adx_vol_normal, 25))

    def _get_allocation(self) -> float:
        """获取当前资金分配比例：优先使用AdaptiveController动态分配，回退到config"""
        if self._adaptive_controller:
            try:
                return self._adaptive_controller.get_allocation("trend")
            except Exception as e:
                logger.debug(f"[trend] get_allocation failed, fallback to config: {e}")
        return self.config["trading"].get("trend_allocation", 0.35)

    def _get_effective_capital(self) -> float:
        """获取有效资金：优先使用实际账户权益，回退到配置中的total_capital。30s TTL 缓存。"""
        import time
        now = time.time()
        if self._capital_cache_value > 0 and (now - self._capital_cache_ts) < self._capital_cache_ttl:
            return self._capital_cache_value
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                details = account_info.get("details", [])
                for detail in details:
                    if detail.get("ccy") == "USDT":
                        eq = float(detail.get("eq", 0))
                        if eq > 0:
                            self._capital_cache_value = eq
                            self._capital_cache_ts = now
                            return eq
                total_eq = float(account_info.get("totalEq", 0))
                if total_eq > 0:
                    self._capital_cache_value = total_eq
                    self._capital_cache_ts = now
                    return total_eq
        except Exception as e:
            logger.warning(f"[trend] get_account_info failed, fallback to config total_capital: {e}")
        fallback = self.config["trading"].get("total_capital", 100.0)
        if self._capital_cache_value <= 0:
            self._capital_cache_value = fallback
            self._capital_cache_ts = now
        return fallback

    def _get_small_cap_multiplier(self, total_capital: float) -> float:
        """小资金适配乘数：资金越少，乘数越大，确保单笔保证金足够"""
        if total_capital >= 1500:
            return 1.0
        elif total_capital >= 800:
            return 1.5
        elif total_capital >= 500:
            return 2.0
        elif total_capital >= 200:
            return 3.0
        elif total_capital >= 100:
            return 4.0
        elif total_capital >= 50:
            return 6.0
        else:
            return 10.0

    def apply_param_update(self, params: Dict[str, Any]):
        """热更新策略参数（由StrategyOptimizer调用，无需重启）"""
        applied = []
        param_map = {
            "confirmation_periods": "_confirmation_periods",
            "max_additions": "_max_additions",
            "initial_position_ratio": "_initial_position_ratio",
            "addition_ratio": "_addition_ratio",
            "profit_targets": "_profit_targets",
            "trailing_stop_tier1": "_trailing_stop_tier1",
            "trailing_stop_tier2": "_trailing_stop_tier2",
            "trailing_stop_tier3": "_trailing_stop_tier3",
        }
        for key, attr_name in param_map.items():
            if key in params:
                setattr(self, attr_name, params[key])
                self.config["strategies"]["trend"][key] = params[key]
                applied.append(key)
        if applied:
            logger.info(f"Trend strategy params hot-updated: {applied}")
        return applied

    async def start(self):
        if not self._enabled:
            logger.info("Trend strategy is disabled")
            return

        logger.info("Starting Trend Strategy")
        # P0-4: 启动时尝试恢复持久化状态（失败不阻塞启动）
        try:
            await self.load_state_async()
        except Exception as e:
            logger.warning(f"Trend state load failed, starting fresh: {e}")

        # P7-1: 启动后同步交易所实际持仓，清理不在交易所的幽灵仓位
        await self._sync_positions_with_exchange()

        asyncio.create_task(self._monitor_loop())
        # P0-4: 启动周期性状态保存协程
        asyncio.create_task(self.periodic_save_loop())

    async def _sync_positions_with_exchange(self):
        """P7-1: 同步交易所实际持仓，清理已不存在的恢复仓位"""
        if not self._position_state and not self._pending_entries:
            return
        try:
            positions = await self.okx_client.get_positions_async()
            # 空列表 = 交易所当前无持仓（合法响应），不可与"抓取失败"混为一谈；
            # 否则交易所净敞口为 0 时，本地幽灵仓位永远无法被清理。
            if positions is None:
                logger.warning("Trend _sync_positions_with_exchange: failed to fetch positions")
                return

            exchange_positions: Dict[str, Dict[str, float]] = {}
            for pos in positions:
                sym = pos.get("instId", "")
                side = pos.get("posSide", "")
                qty = abs(float(pos.get("pos", 0) or 0))
                if sym and side and qty > 0:
                    exchange_positions.setdefault(sym, {})[side] = qty
            exchange_symbols = set(exchange_positions.keys())

            stale = []
            for sym in list(self._position_state.keys()):
                if self._position_state[sym].get("status") == "open" and sym not in exchange_symbols:
                    stale.append(sym)
                    del self._position_state[sym]
                    # 清理相关状态
                    self._last_signal_time.pop(sym, None)
                    self._volatility_lockout_until.pop(sym, None)
                    self._position_entry_times.pop(sym, None)

            if stale:
                logger.info(f"Trend _sync_positions_with_exchange: removed {len(stale)} stale positions: {stale}")
                open_count = sum(1 for ps in self._position_state.values() if ps.get("status") == "open")
                logger.info(f"Trend active positions after sync: {open_count} (max={self._max_concurrent_positions})")

            # ghost_close 专项 Phase 4: 成交回执丢失补偿 — 交易所已有仓位则反向物化 pending intent
            for sym in list(self._pending_entries.keys()):
                entry = self._pending_entries[sym]
                direction = entry.get("position", {}).get("direction", "")
                if exchange_positions.get(sym, {}).get(direction, 0) > 0:
                    pos = self._pending_entries.pop(sym).get("position", {})
                    self._position_state[sym] = pos
                    self._position_entry_times[sym] = pos.get("entry_time")
                    logger.warning(
                        f"[ghost_close-P4] trend position materialized from exchange (fill lost): {sym} {direction}"
                    )

            # ghost_close 专项 Phase 4: 清理超时的 pending intent
            pending_ttl = self.config["strategies"]["trend"].get("pending_ttl_seconds", 15)
            now_ts = time.time()
            for sym in list(self._pending_entries.keys()):
                entry = self._pending_entries[sym]
                if now_ts - entry.get("timestamp", 0) > pending_ttl:
                    del self._pending_entries[sym]
                    logger.warning(f"[ghost_close-P4] trend pending expired: {sym}")
                    self._pending_timeout_cleaned += 1
        except Exception as e:
            logger.warning(f"Trend _sync_positions_with_exchange failed: {e}")

    # P0-4: 状态持久化 - 收集/恢复
    def collect_persistent_state(self) -> Dict[str, Any]:
        """收集需要持久化的状态。"""
        return {
            "_position_state": self._position_state,
            "_last_signal_time": self._last_signal_time,
            "_volatility_lockout_until": self._volatility_lockout_until,
            "_position_entry_times": self._position_entry_times,
            "_trend_performance": self._trend_performance,
        }

    def restore_persistent_state(self, state: Dict[str, Any]):
        """从持久化数据恢复状态。"""
        # 恢复 _position_state，并解析 entry_time 字段（JSON 序列化为字符串）
        raw_pos_state = state.get("_position_state", {}) or {}
        self._position_state = {}
        for sym, ps in raw_pos_state.items():
            if not isinstance(ps, dict):
                continue
            entry_time = ps.get("entry_time")
            if isinstance(entry_time, str):
                try:
                    ps["entry_time"] = datetime.fromisoformat(entry_time)
                except (ValueError, TypeError):
                    try:
                        ps["entry_time"] = datetime.strptime(entry_time, "%Y-%m-%d %H:%M:%S.%f")
                    except (ValueError, TypeError):
                        ps["entry_time"] = datetime.now()
            self._position_state[sym] = ps

        # 恢复 _last_signal_time（Dict[str, datetime]）
        raw_last_signal = state.get("_last_signal_time", {}) or {}
        self._last_signal_time = {}
        for sym, ts in raw_last_signal.items():
            if isinstance(ts, datetime):
                self._last_signal_time[sym] = ts
            elif isinstance(ts, str):
                try:
                    self._last_signal_time[sym] = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    try:
                        self._last_signal_time[sym] = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
                    except (ValueError, TypeError):
                        pass

        # 恢复 _volatility_lockout_until
        raw_lockout = state.get("_volatility_lockout_until", {}) or {}
        self._volatility_lockout_until = {}
        for sym, ts in raw_lockout.items():
            if isinstance(ts, datetime):
                self._volatility_lockout_until[sym] = ts
            elif isinstance(ts, str):
                try:
                    self._volatility_lockout_until[sym] = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    pass

        # 恢复 _position_entry_times
        raw_entry_times = state.get("_position_entry_times", {}) or {}
        self._position_entry_times = {}
        for sym, ts in raw_entry_times.items():
            if isinstance(ts, datetime):
                self._position_entry_times[sym] = ts
            elif isinstance(ts, str):
                try:
                    self._position_entry_times[sym] = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    pass

        # P33: 恢复趋势性能追踪（兼容旧数据缺失字段）
        raw_perf = state.get("_trend_performance") or {}
        if isinstance(raw_perf, dict):
            default_perf = {
                "wins": 0, "losses": 0, "total_pnl": 0, "count": 0,
                "pnl_list": [], "total_profit": 0, "total_loss": 0,
                "consecutive_wins": 0, "consecutive_losses": 0,
                "peak_equity": 0, "max_drawdown": 0,
            }
            for k, v in default_perf.items():
                raw_perf.setdefault(k, v)
            self._trend_performance = raw_perf

        logger.info(
            f"Trend state restored: {len(self._position_state)} positions, "
            f"{len(self._last_signal_time)} signal times, "
            f"{len(self._volatility_lockout_until)} lockouts, "
            f"{len(self._position_entry_times)} entry times"
        )

    async def _monitor_loop(self):
        while True:
            try:
                await self._update_indicators()
                if self._adaptive_enabled:
                    await self._update_market_state()
                await self._check_trend_signals()
                await self._manage_positions()
                
                # 每小时清理一次 _last_signal_time 中已不在监控范围内的symbol
                now = datetime.now()
                if (now - self._last_cleanup_time).total_seconds() >= 3600:
                    stale = [s for s in self._last_signal_time if s not in self._all_symbols]
                    for s in stale:
                        del self._last_signal_time[s]
                    if stale:
                        logger.debug(f"Cleaned up {len(stale)} stale entries from _last_signal_time")
                    self._last_cleanup_time = now
                
                # P7-1: 定期同步交易所持仓，清理幽灵仓位
                # ghost_close 专项 Phase 4: 存在待确认开仓意图时缩短对账周期，
                # 以便及时反向物化（回执丢失）或超时清理（未成交）
                sync_interval = 30 if self._pending_entries else 300
                if (now - self._last_exchange_sync_time).total_seconds() >= sync_interval:
                    open_count = sum(1 for ps in self._position_state.values() if ps.get("status") == "open")
                    if open_count > self._max_concurrent_positions or self._pending_entries:
                        self._last_exchange_sync_time = now
                        await self._sync_positions_with_exchange()
                        # 同步后如果仍然超限，记录警告
                        open_count = sum(1 for ps in self._position_state.values() if ps.get("status") == "open")
                        if open_count > self._max_concurrent_positions:
                            logger.warning(
                                f"Trend position count ({open_count}) still exceeds max "
                                f"({self._max_concurrent_positions}) after sync - "
                                f"positions exist on exchange but exceed limit"
                            )
                
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                logger.info("Trend _monitor_loop cancelled")
                raise
            except Exception as e:
                # 单点异常（脏数据/接口异常）不得终止监控循环，记录后继续
                logger.error(f"Trend _monitor_loop error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _update_indicators(self):
        async def _fetch_symbol(symbol):
            if symbol not in self._indicator_cache:
                self._indicator_cache[symbol] = {}
            for period in self._confirmation_periods:
                try:
                    klines = await self.okx_client.get_kline_async(symbol, period, limit=120)
                    if len(klines) >= 50:
                        self._indicator_cache[symbol][period] = self._calculate_all_indicators(klines, symbol)
                except Exception as e:
                    logger.debug(f"Trend _update_indicators failed for {symbol} {period}: {e}")

        await asyncio.gather(*[_fetch_symbol(sym) for sym in self._all_symbols])

    async def _update_market_state(self):
        async def _fetch_state(symbol):
            try:
                klines = await self.okx_client.get_kline_async(symbol, "1H", limit=self._market_state_lookback + 10)
                if len(klines) >= self._market_state_lookback:
                    klines = sorted(klines, key=lambda k: int(k[0]))
                    prices = [float(k[4]) for k in klines[-self._market_state_lookback:]]
                    volumes = [float(k[5]) for k in klines[-self._market_state_lookback:]]
                    atr = self._indicator_cache.get(symbol, {}).get("1H", {}).get("atr", 0)
                    market_state = detect_market_state(prices, volumes, atr, self._market_state_lookback)
                    self._market_state[symbol] = market_state
            except Exception as e:
                logger.debug(f"Trend _update_market_state failed for {symbol}: {e}")

        await asyncio.gather(*[_fetch_state(sym) for sym in self._all_symbols])

    def _calculate_all_indicators(self, klines, symbol: str = ""):
        data = self._kline_to_df(klines)
        closes = np.array([d["close"] for d in data])
        highs = np.array([d["high"] for d in data])
        lows = np.array([d["low"] for d in data])
        volumes = np.array([d["volume"] for d in data])
        
        ma20 = np.mean(closes[-20:])
        ma50 = np.mean(closes[-50:])
        ma5 = np.mean(closes[-5:]) if len(closes) >= 5 else (np.mean(closes) if len(closes) > 0 else 0)
        ma60 = np.mean(closes[-60:]) if len(closes) >= 60 else ma50
        ema20 = self._calculate_ema(closes, 20)
        vwma20 = self._calculate_vwma(closes, volumes, 20)
        vwma50 = self._calculate_vwma(closes, volumes, 50)
        
        rsi = self._calculate_rsi(closes)
        macd, signal_line, histogram = self._calculate_macd(closes)
        adx, plus_di, minus_di = self._calculate_adx(highs, lows, closes, symbol)
        atr = self._calculate_atr(highs, lows, closes)
        
        recent_close = closes[-1]
        recent_high = np.max(highs[-20:])
        recent_low = np.min(lows[-20:])
        
        # 唐奇安通道（20周期高低点）与 ATR 波动率占比
        donchian_upper = recent_high
        donchian_lower = recent_low
        donchian_mid = (recent_high + recent_low) / 2 if recent_high and recent_low else 0
        atr_pct = atr / recent_close if recent_close > 0 else 0
        
        volume_ma20 = np.mean(volumes[-20:])
        recent_volume = volumes[-1]
        
        # 布林带 (20周期, 2标准差)
        bb_middle = ma20
        bb_std = np.std(closes[-20:]) if len(closes) >= 20 else np.std(closes[-10:]) if len(closes) >= 10 else 0
        bb_upper = bb_middle + bb_std * 2
        bb_lower = bb_middle - bb_std * 2
        
        # 随机指标近似 (14周期, 3周期平滑)
        stoch_k = 50
        stoch_d = 50
        if len(highs) >= 14 and len(lows) >= 14:
            h14 = np.max(highs[-14:])
            l14 = np.min(lows[-14:])
            if h14 != l14:
                raw_k = (recent_close - l14) / (h14 - l14) * 100
                stoch_k = float(np.clip(raw_k, 0, 100))
                stoch_d = float(np.clip((np.mean([raw_k] + list(np.clip(
                    [(closes[-i] - l14) / (h14 - l14) * 100 for i in range(2, min(5, len(closes)+1))], 0, 100
                )))), 0, 100))
        
        return {
            "ma20": ma20,
            "ma50": ma50,
            "ma5": ma5,
            "ma60": ma60,
            "ema20": ema20,
            "vwma20": vwma20,
            "vwma50": vwma50,
            "rsi": rsi,
            "macd": macd,
            "signal_line": signal_line,
            "histogram": histogram,
            "adx": adx,
            "+di": plus_di,
            "-di": minus_di,
            "atr": atr,
            "atr_pct": atr_pct,
            "recent_close": recent_close,
            "recent_high": recent_high,
            "recent_low": recent_low,
            "donchian_upper": donchian_upper,
            "donchian_lower": donchian_lower,
            "donchian_mid": donchian_mid,
            "volume_ma20": volume_ma20,
            "recent_volume": recent_volume,
            "bb_upper": bb_upper,
            "bb_middle": bb_middle,
            "bb_lower": bb_lower,
            "stoch_k": stoch_k,
            "stoch_d": stoch_d,
        }

    def _kline_to_df(self, klines):
        data = []
        for kline in klines:
            data.append({
                "timestamp": int(kline[0]),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5])
            })
        return sorted(data, key=lambda x: x["timestamp"])

    def _calculate_vwma(self, prices: np.ndarray, volumes: np.ndarray, period: int) -> float:
        if len(prices) < period:
            safe_prices = prices[~np.isnan(prices)]
            return np.mean(safe_prices) if len(safe_prices) > 0 else 0
        
        recent_prices = prices[-period:]
        recent_volumes = volumes[-period:]
        
        if np.any(np.isnan(recent_prices)) or np.any(np.isinf(recent_prices)) or \
           np.any(np.isnan(recent_volumes)) or np.any(np.isinf(recent_volumes)) or \
           np.any(recent_volumes < 0):
            safe_prices = recent_prices[~np.isnan(recent_prices)]
            return np.mean(safe_prices) if len(safe_prices) > 0 else 0
        
        total_pv = np.sum(recent_prices * recent_volumes)
        total_volume = np.sum(recent_volumes)
        
        if total_volume <= 0:
            return np.mean(recent_prices)
        
        vwma = total_pv / total_volume
        
        if np.isnan(vwma) or np.isinf(vwma):
            return np.mean(recent_prices)
        
        return vwma

    def _calculate_rsi(self, prices: np.ndarray) -> float:
        if len(prices) < self._rsi_period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-self._rsi_period:])
        avg_loss = np.mean(losses[-self._rsi_period:])
        
        if avg_loss == 0:
            return 100
        if avg_gain == 0:
            return 0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, prices: np.ndarray) -> tuple:
        if len(prices) < 35:
            return 0, 0, 0
        
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return 0, 0, 0
        
        ema12 = self._calculate_ema_array(prices, 12)
        ema26 = self._calculate_ema_array(prices, 26)
        
        if len(ema12) == 0 or len(ema26) == 0:
            return 0, 0, 0
        
        if np.any(np.isnan(ema12)) or np.any(np.isinf(ema12)) or \
           np.any(np.isnan(ema26)) or np.any(np.isinf(ema26)):
            return 0, 0, 0
        
        min_len = min(len(ema12), len(ema26))
        ema12 = ema12[-min_len:]
        ema26 = ema26[-min_len:]
        
        macd_line = ema12 - ema26
        
        if len(macd_line) < 9:
            return 0, 0, 0
        
        if np.any(np.isnan(macd_line)) or np.any(np.isinf(macd_line)):
            return 0, 0, 0
        
        signal_line = self._calculate_ema_array(macd_line, 9)
        
        if len(signal_line) == 0 or np.any(np.isnan(signal_line)) or np.any(np.isinf(signal_line)):
            return 0, 0, 0
        
        histogram = macd_line[-len(signal_line):] - signal_line
        
        if np.any(np.isnan(histogram)) or np.any(np.isinf(histogram)):
            return 0, 0, 0
        
        if len(macd_line) > 0 and len(signal_line) > 0:
            return macd_line[-1], signal_line[-1], histogram[-1]
        return 0, 0, 0

    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, symbol: str = "") -> tuple:
        if len(highs) < self._adx_period + 1:
            return 20, 25, 25
        
        if np.any(np.isnan(highs)) or np.any(np.isinf(highs)) or \
           np.any(np.isnan(lows)) or np.any(np.isinf(lows)) or \
           np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
            return 20, 25, 25
        
        tr_values = []
        plus_di_values = []
        minus_di_values = []
        
        for i in range(1, len(highs)):
            try:
                tr = max(highs[i] - lows[i], 
                         abs(highs[i] - closes[i-1]), 
                         abs(lows[i] - closes[i-1]))
                tr_values.append(tr)
                
                plus_dm = highs[i] - highs[i-1]
                minus_dm = lows[i-1] - lows[i]
                
                if plus_dm > minus_dm and plus_dm > 0:
                    plus_dm_val = plus_dm
                else:
                    plus_dm_val = 0
                    
                if minus_dm > plus_dm and minus_dm > 0:
                    minus_dm_val = minus_dm
                else:
                    minus_dm_val = 0
                    
                plus_di_values.append(plus_dm_val)
                minus_di_values.append(minus_dm_val)
            except (ValueError, TypeError):
                continue
        
        if len(tr_values) < self._adx_period:
            return 20, 25, 25
        
        tr_smooth = np.mean(tr_values[-self._adx_period:])
        plus_di_smooth = np.mean(plus_di_values[-self._adx_period:])
        minus_di_smooth = np.mean(minus_di_values[-self._adx_period:])
        
        if np.isnan(tr_smooth) or np.isinf(tr_smooth) or tr_smooth == 0:
            return 20, 0, 0
        
        plus_di = (plus_di_smooth / tr_smooth) * 100
        minus_di = (minus_di_smooth / tr_smooth) * 100
        
        if np.isnan(plus_di) or np.isinf(plus_di):
            plus_di = 0
        if np.isnan(minus_di) or np.isinf(minus_di):
            minus_di = 0
        
        if plus_di + minus_di == 0:
            dx = 0
        else:
            dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        
        if np.isnan(dx) or np.isinf(dx):
            dx = 0
        
        # 修复：缓存dx历史序列，计算正确的 _adx_period 均值
        if symbol:
            if symbol not in self._dx_cache:
                self._dx_cache[symbol] = []
            self._dx_cache[symbol].append(dx)
            # 只保留最近 adx_period*2 个值，防止内存增长
            if len(self._dx_cache[symbol]) > self._adx_period * 3:
                self._dx_cache[symbol] = self._dx_cache[symbol][-self._adx_period * 2:]
            
            if len(self._dx_cache[symbol]) >= self._adx_period:
                adx = np.mean(self._dx_cache[symbol][-self._adx_period:])
            else:
                adx = np.mean(self._dx_cache[symbol])
        else:
            # 无symbol上下文时（如_reversal检测），用退避方案
            adx = np.mean([dx] * min(3, self._adx_period))
        
        if np.isnan(adx) or np.isinf(adx):
            adx = 20
        
        return max(0, min(100, adx)), max(0, min(100, plus_di)), max(0, min(100, minus_di))

    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        if len(highs) < self._atr_period + 1:
            if len(highs) >= 5:
                safe_highs = highs[-5:]
                safe_lows = lows[-5:]
                if np.any(np.isnan(safe_highs)) or np.any(np.isinf(safe_highs)) or \
                   np.any(np.isnan(safe_lows)) or np.any(np.isinf(safe_lows)):
                    return 0.01
                return np.mean(safe_highs - safe_lows)
            return 0.01
        
        if np.any(np.isnan(highs)) or np.any(np.isinf(highs)) or \
           np.any(np.isnan(lows)) or np.any(np.isinf(lows)) or \
           np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
            return 0.01
        
        tr_values = []
        for i in range(1, len(highs)):
            try:
                tr = max(highs[i] - lows[i], 
                         abs(highs[i] - closes[i-1]), 
                         abs(lows[i] - closes[i-1]))
                tr_values.append(tr)
            except (ValueError, TypeError):
                continue
        
        if len(tr_values) < self._atr_period:
            return 0.01
        
        atr = np.mean(tr_values[-self._atr_period:])
        
        if np.isnan(atr) or np.isinf(atr) or atr <= 0:
            return 0.01
        
        return atr

    def _calculate_ema_array(self, prices: np.ndarray, period: int) -> np.ndarray:
        if len(prices) < period:
            return np.array([np.mean(prices)])
        
        multiplier = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)
        
        return ema[period - 1:]
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return np.mean(prices)
        
        multiplier = 2 / (period + 1)
        ema = np.zeros(len(prices))
        ema[period - 1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = prices[i] * multiplier + ema[i - 1] * (1 - multiplier)
        
        return ema[-1]

    async def _check_trend_signals(self):
        # P0: 并发持仓数限制 - 防止过度分散资金
        open_count = sum(1 for s, ps in self._position_state.items() if ps.get("status") == "open")
        if open_count >= self._max_concurrent_positions:
            if open_count > self._max_concurrent_positions:
                logger.warning(
                    f"Trend _check_trend_signals: open_count={open_count} exceeds "
                    f"max={self._max_concurrent_positions}, skipping signal generation"
                )
            return
        
        async def _check_one(symbol):
            now = datetime.now()
            last_signal = self._last_signal_time.get(symbol)

            if last_signal and (now - last_signal).total_seconds() / 3600 < 1:
                return

            pos_state = self._position_state.get(symbol)
            if pos_state and pos_state.get("status") == "open":
                return

            lockout = self._volatility_lockout_until.get(symbol)
            if lockout and datetime.now() < lockout:
                return

            trend_direction, confidence = await self._detect_trend(symbol)
            self._signal_starvation_stats["scanned"] += 1
            if trend_direction:
                self._signal_starvation_stats["detected"] += 1
                try:
                    sub = await self._evaluate_trend_sub_strategies(symbol)
                    if sub.get("available"):
                        macd_axis = sub.get("macd_axis")
                        if trend_direction == "long" and macd_axis == "bear":
                            confidence = max(0.0, confidence - 0.05)
                        elif trend_direction == "short" and macd_axis == "bull":
                            confidence = max(0.0, confidence - 0.05)
                        elif (trend_direction == "long" and macd_axis == "bull") or \
                                (trend_direction == "short" and macd_axis == "bear"):
                            confidence = min(0.95, confidence + 0.03)
                        ma_sig = sub.get("ma", {}).get("signal")
                        don_sig = sub.get("donchian", {}).get("signal")
                        agreement = sum(1 for s in (ma_sig, don_sig) if s == trend_direction)
                        if agreement >= 1:
                            confidence = min(0.95, confidence + 0.05 * agreement)
                        if sub.get("ma", {}).get("range_market") and ma_sig != trend_direction:
                            confidence = max(0.0, confidence - 0.05)
                        mom = sub.get("momentum", {}).get("composite", 0.0)
                        if (trend_direction == "long" and mom > 0) or (trend_direction == "short" and mom < 0):
                            confidence = min(0.95, confidence + 0.03)
                        elif (trend_direction == "long" and mom < 0) or (trend_direction == "short" and mom > 0):
                            confidence = max(0.0, confidence - 0.05)
                except Exception as e:
                    logger.debug(f"Trend {symbol}: sub-strategy confirmation error: {e}")

                self._signal_starvation_stats["macd_pass"] += 1

                try:
                    mtf_passed = await self._check_multi_timeframe_direction(symbol, trend_direction)
                    vol_passed = await self._check_volume_confirmation(symbol, trend_direction)
                    adx_passed = await self._check_adx_confirmation(symbol)

                    if mtf_passed:
                        self._signal_starvation_stats["mtf_pass"] += 1
                    if vol_passed:
                        self._signal_starvation_stats["vol_pass"] += 1
                    if adx_passed:
                        self._signal_starvation_stats["adx_pass"] += 1

                    confirmation_count = sum(1 for p in (mtf_passed, vol_passed, adx_passed) if p)
                    if confirmation_count < self._min_confirmation_count:
                        self._increment_metric("trend_gate_rejected_total", 1.0,
                                               {"gate": "min_confirmation", "symbol": symbol})
                        return

                    trend_confidence = await self._calculate_trend_confidence(
                        symbol, trend_direction, confidence,
                        mtf_passed=mtf_passed,
                        vol_passed=vol_passed,
                        adx_passed=adx_passed
                    )
                    if trend_confidence < self._confidence_threshold:
                        self._increment_metric("trend_gate_rejected_total", 1.0,
                                               {"gate": "confidence", "symbol": symbol})
                        return

                    confidence = trend_confidence
                except Exception as e:
                    logger.debug(f"Trend {symbol}: enhanced confirmation check error: {e}, using original confidence={confidence:.2f}")

                self._signal_starvation_stats["generated"] += 1
                await self._generate_trend_signal(symbol, trend_direction, confidence)

        await asyncio.gather(*[_check_one(sym) for sym in self._all_symbols])

        # 信号饥饿诊断：周期性汇总开单漏斗各层通过情况
        self._log_starvation_summary()

    def _log_starvation_summary(self):
        """周期性输出开单漏斗诊断摘要，定位"长时间不开单"的瓶颈层。

        每 `starvation_log_interval` 秒（默认 1800s=30min）打印一次各层通过计数，
        并复位计数，帮助运维快速判断信号卡在哪一层（检测/MTF/量/ADX/生成）。
        """
        now = datetime.now()
        if self._starvation_log_interval <= 0:
            return
        if (now - self._starvation_last_log).total_seconds() < self._starvation_log_interval:
            return
        s = self._signal_starvation_stats
        logger.info(
            f"Trend 开单漏斗诊断(近{self._starvation_log_interval // 60}分钟): "
            f"扫描={s['scanned']} 检测方向={s['detected']} MACD通过={s['macd_pass']} "
            f"MTF通过={s['mtf_pass']} 量通过={s['vol_pass']} ADX通过={s['adx_pass']} "
            f"进入下单={s['generated']}"
        )
        self._starvation_last_log = now
        for k in s:
            s[k] = 0

    async def _evaluate_trend_sub_strategies(self, symbol: str) -> Dict[str, Any]:
        """企业级趋势子策略综合评估（均线/唐奇安/MACD零轴/动量）。

        四类子策略均为辅助确认层，叠加在原有 `_detect_trend` 方向之上：
          - 均线趋势：MA5/MA20/MA60 金叉死叉 + 多均线排列 + ATR 震荡过滤
          - 唐奇安通道：N 周期高低点突破 + ATR 动态止盈止损
          - MACD 零轴：零轴上方只做多、零轴下方只做空（仅否决，不产生信号）
          - 动量：2h/4h 滚动涨跌幅 + 多品种相对强弱

        返回供 `_check_trend_signals` 使用的字典：
          {"available": bool, "ma": {...}, "donchian": {...},
           "macd_axis": 'bull'|'bear'|'neutral', "momentum": {"composite": float, ...}}
        """
        result: Dict[str, Any] = {
            "available": False,
            "ma": {},
            "donchian": {},
            "macd_axis": "neutral",
            "momentum": {"composite": 0.0},
        }
        try:
            klines = await self.okx_client.get_kline_async(symbol, self._sub_trend_bar, limit=120)
            if not klines or len(klines) < 30:
                return result

            # OKX 返回倒序（最新在前），按时间戳正序排序后再转数组
            klines = sorted(klines, key=lambda k: int(k[0]))
            closes = np.array([float(k[4]) for k in klines])
            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])

            # 1. 均线趋势（含 ATR 震荡过滤）
            result["ma"] = evaluate_ma_trend(
                closes, highs, lows,
                {"atr_pct_threshold": self._sub_ma_atr_threshold},
            )

            # 2. 唐奇安通道突破
            result["donchian"] = evaluate_donchian(
                highs, lows, closes,
                {
                    "donchian_period": self._sub_donchian_period,
                    "atr_sl_multiplier": self._sub_donchian_atr_sl,
                    "atr_tp_multiplier": self._sub_donchian_atr_tp,
                },
            )

            # 3. MACD 零轴（优先复用指标缓存，避免重复计算）
            ind = self._indicator_cache.get(symbol, {}).get(self._sub_trend_bar)
            if ind:
                macd_val = float(ind.get("macd", 0.0))
                signal_val = float(ind.get("signal_line", 0.0))
            else:
                macd_val, signal_val, _ = self._calculate_macd(closes)
            result["macd_axis"] = macd_zero_axis(macd_val, signal_val)

            # 4. 动量（2h/4h 滚动涨跌幅 + 多品种相对强弱）
            result["momentum"] = await self._compute_momentum(symbol)

            result["available"] = True
        except Exception as e:
            logger.debug(f"Trend {symbol}: _evaluate_trend_sub_strategies error: {e}")
        return result

    async def _compute_momentum(self, symbol: str) -> Dict[str, Any]:
        """计算 2h/4h 动量 + 多品种相对强弱复合值。

        独立拉取 2H / 4H K线计算滚动涨跌幅（保持 2h/4h 语义，不依赖
        主评估周期），再叠加当前品种相对全场中位数的强弱（强者恒强），
        得到带符号的 composite。
        """
        ret_2h = 0.0
        ret_4h = 0.0
        for bar, limit in (("2H", 3), ("4H", 3)):
            try:
                k = await self.okx_client.get_kline_async(symbol, bar, limit=limit)
                if k and len(k) >= 2:
                    k = sorted(k, key=lambda x: int(x[0]))
                    c = np.array([float(x[4]) for x in k])
                    if bar == "2H":
                        ret_2h = rolling_return(c, 1)
                    else:
                        ret_4h = rolling_return(c, 1)
            except Exception:
                continue

        rank = await self._get_momentum_rank()
        rank_ret = rank.get(symbol, ret_4h)

        composite = 0.4 * ret_2h + 0.6 * ret_4h
        if rank:
            median = float(np.median(list(rank.values())))
            composite += 0.5 * (rank_ret - median)

        return {
            "composite": composite,
            "ret_2h": ret_2h,
            "ret_4h": ret_4h,
            "rank_ret": rank_ret,
        }

    async def _get_momentum_rank(self) -> Dict[str, float]:
        """多品种 4h 相对强弱排名（缓存 _momentum_rank_ttl 秒，减少 API 调用）。"""
        import time
        now = time.time()
        if self._momentum_rank_cache and (now - self._momentum_rank_ts) < self._momentum_rank_ttl:
            return self._momentum_rank_cache

        ranks: Dict[str, float] = {}
        for sym in self._all_symbols:
            try:
                k4 = await self.okx_client.get_kline_async(sym, "4H", limit=2)
                if k4 and len(k4) >= 2:
                    k4 = sorted(k4, key=lambda k: int(k[0]))
                    c = np.array([float(k[4]) for k in k4])
                    ranks[sym] = rolling_return(c, 1)
            except Exception:
                continue

        self._momentum_rank_cache = ranks
        self._momentum_rank_ts = now
        return ranks

    async def _detect_trend(self, symbol: str) -> tuple:
        if not self._validate_symbol(symbol):
            return None, 0
        try:
            signals = []
            trend_confidence = []
            period_scores = []
            
            for period in self._confirmation_periods:
                indicators = self._indicator_cache.get(symbol, {}).get(period)
                if not indicators:
                    try:
                        klines = await self.okx_client.get_kline_async(symbol, period, limit=120)
                        if len(klines) < 50:
                            continue
                        indicators = self._calculate_all_indicators(klines, symbol)
                    except Exception as e:
                        logger.debug(f"Trend _detect_trend kline failed for {symbol} {period}: {e}")
                        continue
                
                # 提取原始K线数据用于背离检测和市场结构分析
                try:
                    klines_raw = await self.okx_client.get_kline_async(symbol, period, limit=100)
                except Exception:
                    klines_raw = []
                closes_arr = None
                highs_arr = None
                lows_arr = None
                if len(klines_raw) >= 30:
                    klines_raw = sorted(klines_raw, key=lambda k: int(k[0]))
                    closes_arr = np.array([float(k[4]) for k in klines_raw[-80:]])
                    highs_arr = np.array([float(k[2]) for k in klines_raw[-80:]])
                    lows_arr = np.array([float(k[3]) for k in klines_raw[-80:]])
                
                signal, confidence = self._analyze_period_with_indicators(
                    indicators, closes_arr, highs_arr, lows_arr
                )
                signals.append(signal)
                trend_confidence.append(confidence)
                period_scores.append({
                    "period": period,
                    "signal": signal,
                    "confidence": confidence,
                    "indicators": indicators,
                    "closes": closes_arr,
                    "highs": highs_arr,
                    "lows": lows_arr,
                })
            
            if not signals:
                return None, 0
            
            bull_periods = sum(1 for s in signals if s == "long")
            bear_periods = sum(1 for s in signals if s == "short")
            total_periods = len(signals)
            
            if self._multi_timeframe_confirmation:
                # P32: 降低多时间框架确认比例从60%到50%，允许更多信号通过
                bull_ratio = bull_periods / total_periods
                bear_ratio = bear_periods / total_periods
                
                if bull_ratio >= 0.5:
                    direction = "long"
                    base_confidence = sum(c for s, c in zip(signals, trend_confidence) if s == "long") / bull_periods if bull_periods > 0 else 0
                elif bear_ratio >= 0.5:
                    direction = "short"
                    base_confidence = sum(c for s, c in zip(signals, trend_confidence) if s == "short") / bear_periods if bear_periods > 0 else 0
                else:
                    return None, 0
            else:
                if bull_periods > bear_periods and bull_periods >= 1:
                    direction = "long"
                    base_confidence = sum(trend_confidence) / total_periods
                elif bear_periods > bull_periods and bear_periods >= 1:
                    direction = "short"
                    base_confidence = sum(trend_confidence) / total_periods
                else:
                    return None, 0
            
            if self._trend_strength_filter:
                adx_values = [ps["indicators"].get("adx", 0) for ps in period_scores if ps["signal"] == direction]
                if adx_values:
                    avg_adx = np.mean(adx_values)
                    dynamic_adx_threshold = self._get_dynamic_adx_threshold(symbol)
                    # 趋势识别精度强化：ADX 强度门槛从 0.5 倍提升至 0.8 倍，
                    # 将 normal 波动下的有效 ADX 下限从 12.5（噪声级）恢复到 20（真趋势级），
                    # 过滤震荡市弱趋势的磨损性入场，满足硬约束「ADX阈值≥20」。
                    if avg_adx < dynamic_adx_threshold * 0.8:
                        logger.debug(f"Trend strength too low for {symbol}: ADX={avg_adx:.1f} < {dynamic_adx_threshold*0.8:.1f}")
                        return None, 0
                    trend_strength_bonus = min(0.1, (avg_adx - dynamic_adx_threshold * 0.8) / 100)
                    base_confidence += trend_strength_bonus
            
            if self._breakout_confirmation:
                breakout_confirmed = self._check_breakout_confirmation(symbol, direction, period_scores)
                if breakout_confirmed:
                    base_confidence += 0.05
            
            if self._pullback_entry:
                pullback_score = self._check_pullback_entry(symbol, direction, period_scores)
                if pullback_score > 0:
                    base_confidence += pullback_score * 0.05

            # 共享趋势判断投票：与 MarketRegimeEngine 复用同一套 ADX+结构+EMA排列+DI 投票，
            # 对多周期方向做加权校准（正向强化、负向衰减，不硬否决）
            if self._shared_trend_vote_enabled:
                vote_alignment = self._compute_shared_trend_vote_alignment(period_scores, direction)
                if vote_alignment is not None:
                    base_confidence += vote_alignment * self._shared_trend_vote_weight
            
            final_confidence = min(0.95, max(0.4, base_confidence))

            self._record_metric(
                "trend_detect_direction",
                1.0 if direction == "long" else -1.0,
                {"symbol": symbol},
            )
            self._record_metric(
                "trend_detect_confidence",
                final_confidence,
                {"symbol": symbol, "direction": direction},
            )

            return direction, final_confidence
        except Exception as e:
            self._handle_exception(
                e,
                context={"symbol": symbol},
                module="TrendStrategy",
                function="_detect_trend",
                severity="medium",
                category="business",
            )
            return None, 0

    def _compute_shared_trend_vote_alignment(self, period_scores: List[Dict[str, Any]], direction: str) -> Optional[float]:
        """共享趋势投票与主方向的对齐度（-1..1）

        对每个周期用 compute_trend_vote 计算 ADX+结构+EMA排列+DI 的方向投票强度，
        以主方向为符号基准加权平均，返回与主方向的对齐程度：
        正=投票一致（强化），负=投票背离（衰减）。低 ADX 时强度趋近 0，影响最小。
        """
        sign = 1.0 if direction == "long" else -1.0
        weighted = 0.0
        count = 0
        for ps in period_scores:
            closes = ps.get("closes")
            highs = ps.get("highs")
            lows = ps.get("lows")
            if closes is None or highs is None or lows is None or len(closes) < 20:
                continue
            try:
                vote = compute_trend_vote(
                    closes, highs, lows,
                    adx_period=self._adx_period,
                    adx_floor=self._shared_vote_adx_floor,
                    adx_saturation=self._shared_vote_adx_saturation,
                )
            except Exception as e:
                logger.debug(f"Shared trend vote failed: {e}")
                continue
            aligned = vote["direction"] * vote["adx_strength"] * sign
            weighted += aligned
            count += 1
        if count == 0:
            return None
        return max(-1.0, min(1.0, weighted / count))

    def _check_breakout_confirmation(self, symbol: str, direction: str, period_scores: List[Dict[str, Any]]) -> bool:
        if not period_scores:
            return False
        
        latest = period_scores[-1]["indicators"]
        recent_close = latest.get("recent_close", 0)
        recent_high = latest.get("recent_high", 0)
        recent_low = latest.get("recent_low", 0)
        recent_volume = latest.get("recent_volume", 0)
        volume_ma20 = latest.get("volume_ma20", 0)
        
        if volume_ma20 <= 0:
            return False
        
        volume_ratio = recent_volume / volume_ma20
        
        if direction == "long":
            price_near_high = recent_close >= recent_high * 0.98
            volume_surge = volume_ratio >= self._breakout_volume_factor
            return price_near_high and volume_surge
        else:
            price_near_low = recent_close <= recent_low * 1.02
            volume_surge = volume_ratio >= self._breakout_volume_factor
            return price_near_low and volume_surge

    def _check_pullback_entry(self, symbol: str, direction: str, period_scores: List[Dict[str, Any]]) -> float:
        if len(period_scores) < 2:
            return 0
        
        latest = period_scores[-1]["indicators"]
        previous = period_scores[-2]["indicators"] if len(period_scores) >= 2 else latest
        
        recent_close = latest.get("recent_close", 0)
        ma20 = latest.get("ma20", 0)
        
        if ma20 <= 0 or recent_close <= 0:
            return 0
        
        pullback_score = 0
        
        if direction == "long":
            if ma20 < recent_close:
                distance_to_ma = (recent_close - ma20) / ma20
                if 0 < distance_to_ma < self._pullback_depth:
                    pullback_score = 1.0 - distance_to_ma / self._pullback_depth
        else:
            if ma20 > recent_close:
                distance_to_ma = (ma20 - recent_close) / ma20
                if 0 < distance_to_ma < self._pullback_depth:
                    pullback_score = 1.0 - distance_to_ma / self._pullback_depth
        
        return pullback_score

    def _analyze_period_with_indicators(self, indicators: Dict[str, float], 
                                         closes: np.ndarray = None,
                                         highs: np.ndarray = None,
                                         lows: np.ndarray = None) -> tuple:
        """多因子打分制趋势检测，替代原始严格条件链
        增强版：新增背离检测(10)和市场结构(11)因子
        """
        if not indicators:
            return None, 0

        def _g(key: str, default: float = 0.0) -> float:
            return self._safe_float(indicators.get(key), default)

        ma20 = _g("ma20")
        ma50 = _g("ma50")
        vwma20 = _g("vwma20")
        vwma50 = _g("vwma50")
        rsi = _g("rsi", 50.0)
        macd = _g("macd")
        signal_line = _g("signal_line")
        histogram = _g("histogram")
        adx = _g("adx")
        plus_di = _g("+di")
        minus_di = _g("-di")
        recent_close = _g("recent_close")
        recent_high = _g("recent_high")
        recent_low = _g("recent_low")
        recent_volume = _g("recent_volume")
        volume_ma20 = _g("volume_ma20")

        # 核心价格/均线数据缺失或无效时，趋势判断无意义，直接放弃
        if recent_close <= 0 or ma20 <= 0 or ma50 <= 0:
            return None, 0

        bull_score = 0.0
        bear_score = 0.0
        max_score = 13.0
        
        # 1. 均线排列 (+2)
        if ma20 > ma50 and recent_close > ma20:
            bull_score += 2
        elif ma20 < ma50 and recent_close < ma20:
            bear_score += 2
        elif recent_close > ma20:
            bull_score += 0.5
        elif recent_close < ma20:
            bear_score += 0.5
        
        # 2. VWMA确认 (+1)
        if recent_close > vwma20 and vwma20 > vwma50:
            bull_score += 1
        elif recent_close < vwma20 and vwma20 < vwma50:
            bear_score += 1
        
        # 3. 突破确认 (+2)
        if recent_high > 0 and recent_close >= recent_high * 0.98:
            bull_score += 2
        elif recent_low > 0 and recent_close <= recent_low * 1.02:
            bear_score += 2
        elif recent_high > 0 and recent_close >= recent_high * 0.99:
            bull_score += 1
        elif recent_low > 0 and recent_close <= recent_low * 1.01:
            bear_score += 1
        
        # 4. 成交量确认 (+1)
        if volume_ma20 > 0:
            if recent_volume >= volume_ma20 * 1.5:
                if recent_close > ma20:
                    bull_score += 1
                else:
                    bear_score += 1
            elif recent_volume >= volume_ma20 * 1.2:
                if recent_close > ma20:
                    bull_score += 0.5
                else:
                    bear_score += 0.5
        
        # 5. MACD方向 (+1)
        if macd > signal_line and histogram > 0:
            bull_score += 1
        elif macd < signal_line and histogram < 0:
            bear_score += 1
        
        # 6. ADX趋势强度 (+1)
        if adx > self._adx_threshold:
            if plus_di > minus_di:
                bull_score += 1
            else:
                bear_score += 1
        elif adx > 20:
            if plus_di > minus_di:
                bull_score += 0.5
            else:
                bear_score += 0.5
        
        # 7. RSI状态 (+1)
        if self._rsi_oversold < rsi < self._rsi_overbought:
            if rsi > 50:
                bull_score += 1
            else:
                bear_score += 1
        elif rsi <= self._rsi_oversold:
            bull_score += 0.5
        elif rsi >= self._rsi_overbought:
            bear_score += 0.5
        
        # 8. MACD柱状图动量 (+1)
        hist_threshold = recent_close * 0.0003
        if histogram > hist_threshold:
            bull_score += 1
        elif histogram < -hist_threshold:
            bear_score += 1
        
        # 9. DI差值 (+1)
        di_diff = abs(plus_di - minus_di)
        if plus_di > minus_di and di_diff > 5:
            bull_score += 1
        elif minus_di > plus_di and di_diff > 5:
            bear_score += 1
        
        # 10. 背离检测 (+1) — 顶背离=做空加分，底背离=做多加分
        if self._divergence_detection and closes is not None and highs is not None and lows is not None:
            rsi_seq = None
            if len(closes) >= 14:
                rsi_seq = np.array([self._calculate_rsi(closes[:i+1]) for i in range(14, len(closes))])
            
            divergence = self._detect_divergence(closes, highs, lows, rsi_seq)
            
            if divergence.get("rsi") == "bearish":
                bear_score += 1  # 顶背离，做空信号增强
            elif divergence.get("rsi") == "bullish":
                bull_score += 1  # 底背离，做多信号增强
            
            if divergence.get("macd") == "bearish":
                bear_score += 0.5
            elif divergence.get("macd") == "bullish":
                bull_score += 0.5
        
        # 11. 市场结构分析 (+1) — HH/HL=多头加分，LH/LL=空头加分
        if self._market_structure_enabled and highs is not None and lows is not None and closes is not None:
            structure = self._analyze_market_structure(highs, lows, closes)
            if structure == "strong_uptrend":
                bull_score += 1
            elif structure == "uptrend":
                bull_score += 0.7
            elif structure == "strong_downtrend":
                bear_score += 1
            elif structure == "downtrend":
                bear_score += 0.7
        
        threshold = max_score * 0.30  # P6-2: 从0.40降至0.30，激活更多趋势信号
        
        if bull_score >= threshold and bull_score > bear_score:
            confidence = min(0.95, 0.40 + bull_score / max_score * 0.55)
            return "long", confidence
        elif bear_score >= threshold and bear_score > bull_score:
            confidence = min(0.95, 0.40 + bear_score / max_score * 0.55)
            return "short", confidence
        
        return None, 0

    def _generate_clordid(self, symbol: str, direction: str, signal_type: str) -> str:
        """生成合规 clOrdId（纯字母+数字，≤32位），用于成交回执关联（ghost_close 专项 Phase 4）。"""
        import re
        ts_ms = int(time.time() * 1000)
        short_symbol = re.sub(r'[^a-zA-Z0-9]', '', symbol.replace("-USDT", ""))[:6]
        short_type = re.sub(r'[^a-zA-Z0-9]', '', str(signal_type or ""))[:6]
        dir_flag = "L" if direction in ("long", "buy") else "S"
        clordid = f"trd{short_symbol}{dir_flag}{ts_ms}{short_type}"
        if len(clordid) > 32:
            clordid = clordid[:32]
        return clordid

    async def _generate_trend_signal(self, symbol: str, direction: str, confidence: float):
        if not self._validate_symbol(symbol) or not self._validate_direction(direction) or not self._validate_confidence(confidence):
            logger.warning(
                f"Trend signal rejected: invalid params "
                f"symbol={symbol!r} direction={direction!r} confidence={confidence!r}"
            )
            self._increment_metric("trend_signal_rejected_total", 1.0, {"reason": "invalid_params", "symbol": symbol})
            return
        # P-资金费率: 统一由 FundingRateEnhancer 调整开仓置信度（择时开仓，失败默认放行）
        confidence = await self._funding_enhancer.adjust_entry_confidence(symbol, direction, confidence)

        if self._adaptive_enabled:
            market_state = self._market_state.get(symbol, {"state": "unknown", "volatility": "normal"})
            
            if market_state["state"] == "range" and market_state.get("confidence", 0) > 0.75:
                logger.debug(f"Strong range market state, skipping trend signal for {symbol}")
                return
            
            timeframe = self.config["strategies"]["trend"]["timeframe"]
            ind = self._indicator_cache.get(symbol, {}).get(timeframe, {})
            # 修复：补充 evaluate_signal_quality 期望的所有字段（之前仅5个，缺失6个导致评分系统性偏低）
            relevant_close = ind.get("recent_close", 0)
            relevant_atr = ind.get("atr", 0)
            indicators = {
                "rsi": ind.get("rsi", 50),
                "adx": ind.get("adx", 0),
                "volume_delta": ind.get("recent_volume", 0) / max(ind.get("volume_ma20", 1), 1) - 1,
                "vwap_distance": (relevant_close - ind.get("vwma20", 0)) / max(ind.get("vwma20", 1), 1) if relevant_close > 0 else 0,
                "momentum": ind.get("macd", 0) - ind.get("signal_line", 0),
                "macd_histogram": ind.get("histogram", 0),
                "bb_position": (relevant_close - ind.get("bb_lower", relevant_close)) / max(ind.get("bb_upper", 1) - ind.get("bb_lower", 0), 1) if relevant_close > 0 else 0.5,
                "stoch_k": ind.get("stoch_k", 50),
                "stoch_d": ind.get("stoch_d", 50),
                "atr_ratio": relevant_atr / max(relevant_close, 1) if relevant_close > 0 else 0,
            }
            
            signal_quality = evaluate_signal_quality(indicators, market_state, direction)
            
            adaptive_threshold = self._min_signal_quality
            # P1: 低波动/震荡市应提高门槛（趋势难形成），而非降低门槛
            if market_state.get("volatility") == "low":
                adaptive_threshold = min(0.35, self._min_signal_quality + 0.10)
            elif market_state.get("state") == "range":
                adaptive_threshold = min(0.32, self._min_signal_quality + 0.08)
            elif market_state.get("volatility") == "high":
                adaptive_threshold = max(0.15, self._min_signal_quality - 0.05)  # 高波动更易形成趋势
            
            if signal_quality < adaptive_threshold:
                logger.debug(f"Signal quality {signal_quality:.2f} below threshold {adaptive_threshold:.2f} for {symbol}")
                return
            
            confidence = confidence * (0.8 + signal_quality * 0.2)
        
        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        leverage = tier_settings["leverage_max"]
        
        # P0: 首仓杠杆上限检查——防止小账户使用过高杠杆（与加仓逻辑一致）
        abs_max_leverage = self.config.get("leverage_tiers", {}).get("absolute_max", 5)
        if leverage > abs_max_leverage:
            logger.warning(f"Trend {symbol}: leverage {leverage}x capped to absolute max {abs_max_leverage}x")
            leverage = abs_max_leverage
        
        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        position_limit = tier_settings["position_limit"]
        
        base_position = trading_capital * min(allocation, position_limit)

        # 注入空闲资金放大乘数（来自 AdaptiveController）
        if self._adaptive_controller:
            try:
                boost = self._adaptive_controller.get_position_boost()
                if boost > 1.0:
                    base_position *= boost
            except Exception as e:
                logger.debug(f"[trend] get_position_boost failed: {e}")

        price_factor = 1.0
        if tier == "tier2":
            price_factor = 1.2
        elif tier == "tier3":
            price_factor = 1.4
        
        base_position *= price_factor
        
        if total_capital < 500:
            small_cap_multiplier = self._get_small_cap_multiplier(total_capital)
            base_position *= small_cap_multiplier
        
        if self._adaptive_enabled:
            market_state = self._market_state.get(symbol, {"state": "range", "volatility": "normal", "volume_ratio": 1.0})
            base_position = calculate_adaptive_position_size(base_position, market_state, "trend")
        
        # P33: 企业级凯利仓位调整（在静态自适应之上叠加数据驱动的凯利）
        base_position = self._calculate_kelly_position(base_position, symbol)
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        
        price = float(ticker["last"])
        atr = self._get_atr(symbol)
        init_ratio = self._initial_position_ratio if total_capital >= 1500 else min(1.0, self._initial_position_ratio * 1.5)
        
        # P1-Fix: 波动率目标仓位 - 使用日ATR归一化仓位，控制目标日波动率1%
        daily_atr_pct = (atr * np.sqrt(24)) / price if price > 0 else 0  # 1H ATR转日ATR
        target_vol = 0.01  # 目标日波动1%
        vol_adjustment = min(2.0, max(0.3, target_vol / daily_atr_pct)) if daily_atr_pct > 0 else 1.0
        init_ratio *= vol_adjustment

        # P1-Fix: 单笔仓位保证金上限（防止单笔过大，0 表示不限制）
        max_position_usdt = self.config.get("strategies", {}).get("trend", {}).get("max_position_usdt", 0)
        if max_position_usdt and max_position_usdt > 0 and base_position > max_position_usdt:
            logger.info(f"Trend {symbol}: base_position {base_position:.2f} capped to {max_position_usdt} USDT")
            base_position = max_position_usdt

        quantity = base_position * init_ratio * leverage / price
        
        # P0: 入场价格合理性检查 - 验证价格在近期K线范围内
        try:
            klines_1h = await self.okx_client.get_kline_async(symbol, "1H", limit=4)
            if klines_1h and len(klines_1h) >= 3:
                recent_highs = [float(k[2]) for k in klines_1h]  # high
                recent_lows = [float(k[3]) for k in klines_1h]    # low
                range_high = max(recent_highs)
                range_low = min(recent_lows)
                range_width = (range_high - range_low) / range_low if range_low > 0 else 0
                # 允许价格在近期范围的+-1.5%内（容忍突破/跳空）
                if price > range_high * 1.015 or price < range_low * 0.985:
                    logger.warning(f"Trend {symbol}: price {price:.4f} outside recent range "
                                  f"[{range_low:.4f}-{range_high:.4f}] (width={range_width:.2%}), "
                                  f"possible data anomaly, skipping")
                    return
        except Exception:
            pass  # 数据获取失败时不阻塞，正常流程继续
        
        # P0: quantity 有效性检查
        if quantity <= 0 or np.isnan(quantity) or np.isinf(quantity):
            logger.warning(f"Trend {symbol}: invalid quantity={quantity}, skipping")
            return
        
        min_lot_size = self._safe_float((await self.okx_client.get_instrument_info_async(symbol) or {}).get("lotSz", "1"), 1.0)
        margin_needed_for_min_lot = price * min_lot_size / leverage
        
        if base_position < margin_needed_for_min_lot:
            # 小账户适配：若 base_position 不足最小手数保证金，提升至最小手数所需金额
            # 但不超过策略可用资金的 80%，避免单仓占用全部资金
            strategy_cap = trading_capital * allocation
            if total_capital < 1500:
                adjusted = min(margin_needed_for_min_lot, strategy_cap * 0.8)
                if adjusted >= margin_needed_for_min_lot:
                    logger.debug(f"Trend {symbol}: small-cap adjusted margin {base_position:.2f} -> {adjusted:.2f}")
                    base_position = adjusted
                    quantity = base_position / price
                else:
                    logger.debug(f"Trend {symbol}: margin {base_position:.2f} < min lot margin {margin_needed_for_min_lot:.2f}, skip")
                    return
            else:
                logger.warning(f"Insufficient capital for trend {symbol}: need {margin_needed_for_min_lot:.2f} USDT, have {base_position:.2f} USDT. Skipping.")
                return
        
        # 追单铁则硬拦截（文档 5.2.3）：1.5×ATR 偏离 + 单根K线涨跌幅超限 → 拒绝追单，等待回踩/反弹
        pending_price = 0.0
        try:
            klines_chase = await self.okx_client.get_kline_async(symbol, "1H", limit=30)
            # OKX history-candles 返回降序（最新在前），升序后 klines[-1] 才是最新K线
            if klines_chase:
                klines_chase = sorted(klines_chase, key=lambda k: int(k[0]))
            is_chasing, chase_reason = self._detect_chase(symbol, direction, price, atr, tier, klines_chase)
            if is_chasing:
                logger.warning(f"Trend {symbol}: 追单拦截 - {chase_reason}")
                self._increment_metric("trend_chase_intercepted_total", 1.0, {"symbol": symbol, "direction": direction})
                return
            # 等位挂单锚点：post_only 开启时计算支撑/压力位，作为执行层分档挂单基准
            if self._post_only:
                pending_price = self._compute_pending_anchor(symbol, direction, price, klines_chase)
        except Exception:
            pass  # 追单/锚点检测数据获取失败时不阻塞正常下单链路

        # 震荡期抑制门控（文档 5.4）：4 条件识别震荡市
        # score>=3 禁开新仓；score==2 挂单间距翻倍（更保守等更深回踩）
        pending_spread_multiplier = 1.0
        osc_score, osc_reasons = self._detect_oscillation(symbol)
        if osc_score >= 3:
            logger.warning(
                f"Trend {symbol}: 震荡期抑制 - {osc_score}/4 条件命中({'; '.join(osc_reasons)})，禁开新仓"
            )
            self._increment_metric("trend_oscillation_blocked_total", 1.0, {"symbol": symbol, "direction": direction})
            return
        if osc_score >= 2:
            pending_spread_multiplier = 2.0
            logger.info(f"Trend {symbol}: 震荡期边缘({osc_score}/4: {'; '.join(osc_reasons)})，挂单间距翻倍")

        stop_loss = self._calculate_stop_loss(symbol, price, direction, atr)
        take_profit = self._calculate_take_profit(symbol, price, direction, atr)
        
        # P0: 验证止盈止损价格有效性（方向正确 + 不立即触发 + 符合最小距离）
        tp_sl_valid = validate_tp_sl_prices(price, direction, take_profit, stop_loss, price)
        if not tp_sl_valid.get("valid", False):
            logger.warning(f"Trend {symbol}: TP/SL validation failed for {direction} - "
                          f"errors={tp_sl_valid.get('errors', [])}")
            return
        
        # 止盈覆盖手续费检查：预期收益必须 > 3倍手续费（从config读取，避免硬编码）
        taker_fee = self.config.get("trading", {}).get("taker_fee_rate", 0.0005)
        round_trip_fee = taker_fee * 2  # 开仓+平仓各一次
        position_value = price * quantity
        total_fee_cost = position_value * round_trip_fee
        if direction == "long":
            expected_profit = (take_profit - price) * quantity
        else:
            expected_profit = (price - take_profit) * quantity
        if expected_profit <= total_fee_cost * 1.5:
            logger.debug(f"Trend {symbol}: expected profit {expected_profit:.4f} <= 1.5x fee cost {total_fee_cost*1.5:.4f}, skipping")
            return
        
        signal = Signal(
            symbol=symbol,
            strategy_name="trend",
            signal_type="trend_entry",
            direction=direction,
            price=price,
            quantity=quantity,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            order_type="post_only" if self._post_only else "",
            pending_price=pending_price,
            pending_spread_multiplier=pending_spread_multiplier,
            timestamp=datetime.now()
        )

        clordid = self._generate_clordid(symbol, direction, signal.signal_type)

        # ghost_close 专项 Phase 4: 待确认开仓意图阻塞同 symbol 新开仓，避免重复下单
        if self._use_fill_driven_position and symbol in self._pending_entries:
            logger.debug(f"Skipping signal: {symbol} has a pending entry awaiting fill")
            return

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
                "order_type": signal.order_type,
                "pending_price": signal.pending_price,
                "pending_spread_multiplier": signal.pending_spread_multiplier,
                "timestamp": signal.timestamp.isoformat()
            }
        })
        
        # 防止覆盖已有开仓
        if symbol in self._position_state and self._position_state[symbol].get("status") == "open":
            logger.warning(f"Skipping signal: {symbol} already has an open position")
            return
        
        entry_time = datetime.now()
        position_record = {
            "direction": direction,
            "entry_price": price,
            "peak_price": price,
            "trough_price": price,
            "initial_quantity": quantity,
            "current_quantity": quantity,
            "additions": 0,
            "status": "open",
            "profit_taken": 0,
            "atr": atr,
            "initial_stop_loss": stop_loss,
            "current_profit": 0.0,
            "dynamic_stop_loss": stop_loss,
            "adjusted_stop_loss": stop_loss,
            "entry_time": entry_time,
            "tp1_triggered": False,
            "tp2_triggered": False,
            "tp3_triggered": False,
        }
        if self._use_fill_driven_position:
            # ghost_close 专项 Phase 4: 记录待确认开仓意图，等成交回执再物化 _position_state
            self._pending_entries[symbol] = {
                "position": position_record,
                "clOrdId": clordid,
                "timestamp": time.time(),
            }
            logger.info(f"Trend pending entry recorded: {symbol} {direction} clOrdId={clordid}")
        else:
            self._position_state[symbol] = position_record
            self._position_entry_times[symbol] = entry_time
        
        self._last_signal_time[symbol] = datetime.now()
        self._record_metric("trend_signal_generated_total", 1.0, {"symbol": symbol, "direction": direction})
        self._record_metric("trend_signal_confidence", confidence, {"symbol": symbol, "direction": direction})
        logger.info(f"Trend signal: {direction} {symbol} @ {price:.4f}, qty: {quantity:.4f}, conf: {confidence:.2f}, ATR: {atr:.4f}")

    def on_order_filled(self, fill_payload: Dict[str, Any]):
        """成交回执回调（ghost_close 专项 Phase 4）：匹配 pending intent 并物化仓位。"""
        try:
            strategy_name = fill_payload.get("strategy_name", "")
            if strategy_name != "trend":
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
            if clordid and pending.get("clOrdId") and clordid != pending.get("clOrdId"):
                self._pending_entries[symbol] = pending
                return

            pos = pending.get("position", {})
            direction = pos.get("direction", fill_payload.get("direction", ""))
            if filled_qty > 0:
                pos["current_quantity"] = filled_qty
                pos["initial_quantity"] = filled_qty
            if avg_price > 0:
                pos["entry_price"] = avg_price
                if direction == "long":
                    pos["peak_price"] = avg_price
                else:
                    pos["trough_price"] = avg_price

            self._position_state[symbol] = pos
            self._position_entry_times[symbol] = pos.get("entry_time")
            self._fill_callback_hits += 1
            logger.info(
                f"[ghost_close-P4] trend position materialized: {symbol} {direction} "
                f"qty={pos.get('current_quantity')} avg_px={pos.get('entry_price')} clOrdId={clordid}"
            )
        except Exception as e:
            logger.debug(f"on_order_filled error: {e}")

    def _detect_chase(self, symbol: str, direction: str, price: float, atr: float, tier: str, klines) -> Tuple[bool, str]:
        """追单铁则检测（文档 5.2.3）。

        规则：
        1. 单根K线涨跌幅超过阈值 → 追涨/追跌拦截（大盘 tier1=3%，小盘 tier2/3=8%）
        2. 价格偏离趋势中枢（近20根1H收盘均价）超过 1.5×ATR → 追高/追低拦截

        返回 (is_chasing: bool, reason: str)。数据不足或异常时放行（False）。
        """
        try:
            candle_limit = 0.03 if tier == "tier1" else 0.08

            if klines and len(klines) >= 2:
                # 最近一根K线（可能未收盘），用当前价作为实时 close
                last_open = float(klines[-1][1])
                if last_open > 0:
                    candle_change = (price - last_open) / last_open
                    if direction == "long" and candle_change >= candle_limit:
                        return True, (
                            f"单根K线涨幅 {candle_change:.2%} 超过 {candle_limit:.0%}(tier={tier})，追高禁止"
                        )
                    if direction == "short" and candle_change <= -candle_limit:
                        return True, (
                            f"单根K线跌幅 {candle_change:.2%} 超过 {candle_limit:.0%}(tier={tier})，追低禁止"
                        )

            # 1.5×ATR 偏离：近20根1H收盘均价作为趋势中枢锚点
            if klines and len(klines) >= 5 and atr > 0:
                closes = [float(k[4]) for k in klines[-20:] if float(k[4]) > 0]
                if closes:
                    anchor = sum(closes) / len(closes)
                    dev_atr = (price - anchor) / atr if direction == "long" else (anchor - price) / atr
                    if dev_atr > 1.5:
                        return True, (
                            f"价格偏离趋势中枢 {dev_atr:.2f}×ATR (>1.5)，"
                            f"{'追高' if direction == 'long' else '追低'}，应等回踩/反弹"
                        )
        except Exception as e:
            logger.debug(f"_detect_chase error for {symbol}: {e}")

        return False, ""

    def _compute_pending_anchor(self, symbol: str, direction: str, price: float, klines) -> float:
        """等位挂单锚点计算（文档 5.3）。

        趋势确认后不追行情，改在支撑/压力位挂 maker 单等回踩/反弹：
        - 开多：取近 12 根 1H K 线最低价作为支撑位（锚点必须低于现价，否则回退 0）
        - 开空：取近 12 根 1H K 线最高价作为压力位（锚点必须高于现价，否则回退 0）

        返回锚点价格；数据不足或锚点与方向矛盾时返回 0（执行层回退为普通盘口挂单）。
        """
        try:
            if not klines:
                return 0.0
            recent = klines[-12:]
            if len(recent) < 5:
                return 0.0
            lows = [float(k[3]) for k in recent if float(k[3]) > 0]
            highs = [float(k[2]) for k in recent if float(k[2]) > 0]
            if not lows or not highs:
                return 0.0
            if direction == "long":
                anchor = min(lows)  # 支撑位
                if anchor >= price:
                    return 0.0
            else:
                anchor = max(highs)  # 压力位
                if anchor <= price:
                    return 0.0
            return round(anchor, 6)
        except Exception as e:
            logger.debug(f"_compute_pending_anchor error for {symbol}: {e}")
            return 0.0

    def _detect_oscillation(self, symbol: str) -> Tuple[int, List[str]]:
        """震荡期识别（文档 5.4）：4 条件打分，识别无趋势震荡市。

        条件：
        1. ADX < 20（趋势强度不足）
        2. 均线缠绕（MA20 与 MA50 粘合 < 1%）
        3. 价格处于区间中部（price_position ∈ [0.4, 0.6]，无方向）
        4. 低波动 / 缩量（volatility=low 或 volume_ratio < 0.8）

        返回 (score: 0-4, reasons: 命中原因列表)。
        """
        score = 0
        reasons: List[str] = []
        try:
            ind = self._indicator_cache.get(symbol, {}).get("1H", {})
            ms = self._market_state.get(symbol, {})

            adx = ind.get("adx", 0) or 0
            if adx < 20:
                score += 1
                reasons.append(f"ADX={adx:.1f}<20")

            ma20 = ind.get("ma20", 0) or 0
            ma50 = ind.get("ma50", 0) or 0
            if ma20 > 0 and ma50 > 0 and abs(ma20 - ma50) / ma20 < 0.01:
                score += 1
                reasons.append("MA20/MA50缠绕")

            pp = float(ms.get("price_position", 0.5) if ms else 0.5)
            if 0.4 <= pp <= 0.6:
                score += 1
                reasons.append(f"价格居中({pp:.2f})")

            vol = ms.get("volatility", "normal") if ms else "normal"
            vr = float(ms.get("volume_ratio", 1.0) if ms else 1.0)
            if vol == "low" or vr < 0.8:
                score += 1
                reasons.append(f"低波/缩量(vol={vol},vr={vr:.2f})")
        except Exception as e:
            logger.debug(f"_detect_oscillation error for {symbol}: {e}")

        return score, reasons

    def _get_atr(self, symbol: str) -> float:
        for period in self._confirmation_periods:
            indicators = self._indicator_cache.get(symbol, {}).get(period)
            if indicators:
                return indicators.get("atr", 0.01)
        return 0.01

    def _calculate_stop_loss(self, symbol: str, price: float, direction: str, atr: float) -> float:
        tier = get_currency_tier(symbol, self.config)
        precision = get_price_precision(symbol)
        
        if atr > 0:
            sl_pct = (atr * self._atr_multiplier) / price
        else:
            sl_pct = self._false_break_threshold
        
        sl_pct = max(self._false_break_threshold, sl_pct)
        
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        slippage = tier_settings.get("slippage", 0.001)
        
        return calculate_stop_loss(price, direction, sl_pct, slippage, precision)

    def _calculate_take_profit(self, symbol: str, price: float, direction: str, atr: float) -> float:
        tier = get_currency_tier(symbol, self.config)
        precision = get_price_precision(symbol)
        
        if tier == "tier1":
            trailing = self._trailing_stop_tier1
        elif tier == "tier2":
            trailing = self._trailing_stop_tier2
        else:
            trailing = self._trailing_stop_tier3
        
        if atr > 0:
            tp_pct = (atr * 3) / price
        else:
            tp_pct = trailing * 5
        
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        slippage = tier_settings.get("slippage", 0.001)
        
        return calculate_take_profit(price, direction, tp_pct, slippage, precision)

    async def _manage_positions(self):
        for symbol in list(self._position_state.keys()):
            state = self._position_state[symbol]
            if state["status"] != "open":
                continue
            
            ticker = await self.okx_client.get_ticker_async(symbol)
            if not ticker:
                continue
            
            current_price = float(ticker["last"])
            entry_price = state["entry_price"]
            direction = state["direction"]
            
            profit = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price
            state["current_profit"] = profit
            
            # P1-Fix: 统一止损比较 - 比较 adjusted_stop_loss 和 dynamic_stop_loss，取更严格的值
            adj_sl = state.get("adjusted_stop_loss")
            dyn_sl = state.get("dynamic_stop_loss")
            if adj_sl is not None and dyn_sl is not None:
                if direction == "long":
                    effective_stop = max(adj_sl, dyn_sl)
                else:
                    effective_stop = min(adj_sl, dyn_sl)
                state["adjusted_stop_loss"] = effective_stop
                state["dynamic_stop_loss"] = effective_stop
            
            if not check_profit_protection(entry_price, current_price, direction, self._profit_protection_max_drawdown):
                await self._close_position(symbol, force=True)
                state["status"] = "closed"
                logger.info(f"Trend profit protection triggered: {symbol}")
                continue
            
            await self._check_addition(symbol, current_price)
            await self._check_multiple_take_profit(symbol, current_price)
            await self._check_take_profit(symbol, current_price)
            
            # P0-4: 双重触发协调 — 如果交易所侧已有活跃的条件止损单，跳过策略级止损检查
            # 防止策略级轮询和交易所条件单同时触发导致双重平仓
            if hasattr(self, "_conditional_order_manager") and self._conditional_order_manager:
                if self._conditional_order_manager.has_active_conditional_orders(symbol, "stop_loss"):
                    logger.debug(f"P0-4: Skipping strategy SL check for {symbol} — exchange conditional SL active")
                else:
                    await self._check_stop_loss(symbol, current_price)
            else:
                await self._check_stop_loss(symbol, current_price)
            
            await self._check_trailing_stop(symbol, current_price)
            await self._check_trend_reversal(symbol, current_price)
            await self._check_time_exit(symbol, current_price)
            await self._check_volatility_spike_exit(symbol, current_price)

            # 行情反转智能落袋：动态止盈 + 反转减仓（HMM 为主 + 指标兜底）
            if self._stop_loss_manager and self._position_state[symbol].get("status") == "open":
                await self._stop_loss_manager.check_reversal_take_profit(
                    symbol=symbol,
                    strategy_name="trend",
                    current_price=current_price,
                    position_state=self._position_state[symbol],
                )

    async def _check_addition(self, symbol: str, current_price: float):
        state = self._position_state[symbol]
        if state.get("status") != "open":
            return
        if state["additions"] >= self._max_additions:
            return
        
        direction = state["direction"]
        entry_price = state["entry_price"]
        atr = state["atr"]
        
        profit = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price
        
        if profit < 0.01:
            return
        
        atr_distance = atr / entry_price
        
        if direction == "long" and current_price > entry_price * (1 + atr_distance * (state["additions"] + 1)):
            await self._add_position(symbol)
        elif direction == "short" and current_price < entry_price * (1 - atr_distance * (state["additions"] + 1)):
            await self._add_position(symbol)

    async def _add_position(self, symbol: str):
        state = self._position_state[symbol]
        direction = state["direction"]

        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return

        price = float(ticker["last"])
        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        leverage = tier_settings["leverage_max"]

        # P0-1: 使用 _get_effective_capital() 替代硬编码 total_capital（与 _generate_trend_signal 一致）
        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        position_limit = tier_settings["position_limit"]

        base_position = trading_capital * min(allocation, position_limit)
        # P0-1: 删除 base_position /= tier_count（与开仓逻辑 _generate_trend_signal 保持一致，
        # 原逻辑会让加仓仓位远小于开仓仓位，导致加仓无效）

        price_factor = 1.0
        if tier == "tier2":
            price_factor = 1.2
        elif tier == "tier3":
            price_factor = 1.4

        base_position *= price_factor

        min_lot_size = self._safe_float((await self.okx_client.get_instrument_info_async(symbol) or {}).get("lotSz", "1"), 1.0)
        margin_needed_for_min_lot = price * min_lot_size / leverage

        if base_position < margin_needed_for_min_lot:
            logger.warning(f"Insufficient capital for trend addition {symbol}: need {margin_needed_for_min_lot:.2f} USDT, have {base_position:.2f} USDT. Skipping.")
            return

        pyramid_factor = 0.7 ** state["additions"]
        add_quantity = base_position * self._addition_ratio * pyramid_factor * leverage / price

        # P0-1: 加仓前检查总持仓杠杆（不超过 max_leverage * 0.8）
        current_qty = state.get("current_quantity", 0)
        expected_total_value = (current_qty + add_quantity) * price
        if total_capital > 0 and expected_total_value / total_capital > leverage * 0.8:
            logger.warning(
                f"Trend add {symbol} skipped: expected total leverage "
                f"{expected_total_value/total_capital:.2f}x > {leverage*0.8:.2f}x (80% of max {leverage}x)"
            )
            return
        
        signal = Signal(
            symbol=symbol,
            strategy_name="trend",
            signal_type="trend_addition",
            direction=direction,
            price=price,
            quantity=add_quantity,
            leverage=leverage,
            confidence=0.6 + state["additions"] * 0.05,
            timestamp=datetime.now()
        )
        
        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
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
        
        # Bug fix: 加仓后更新 entry_price 为加权平均成本
        old_entry = state["entry_price"]
        old_qty = state["current_quantity"]
        state["entry_price"] = (old_entry * old_qty + price * add_quantity) / (old_qty + add_quantity)
        state["current_quantity"] += add_quantity
        state["additions"] += 1

        # 更新 atr（从当前 ticker 重新获取，确保止损基于最新波动率）
        new_atr = self._get_atr(symbol)
        if new_atr > 0:
            state["atr"] = new_atr

        # 基于新的 entry_price 重新计算 dynamic_stop_loss
        direction = state["direction"]
        new_stop_loss = self._calculate_stop_loss(symbol, state["entry_price"], direction, state["atr"])
        state["dynamic_stop_loss"] = new_stop_loss
        state["adjusted_stop_loss"] = new_stop_loss
        
        logger.info(f"Trend addition: {direction} {symbol} @ {price:.4f}, qty: {add_quantity:.4f}, "
                    f"new_entry: {state['entry_price']:.4f}, additions: {state['additions']}, pyramid: {pyramid_factor:.2f}")

    async def _check_take_profit(self, symbol: str, current_price: float):
        state = self._position_state[symbol]
        direction = state["direction"]
        entry_price = state["entry_price"]
        
        if direction == "long":
            state["peak_price"] = max(state["peak_price"], current_price)
        else:
            state["trough_price"] = min(state["trough_price"], current_price)
        
        profit = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price
        
        for target in self._profit_targets:
            if profit >= target["threshold"] and state["profit_taken"] < target["threshold"]:
                close_qty = state["current_quantity"] * target["close_ratio"]
                
                if close_qty > 0:
                    state["exit_reason"] = "take_profit"
                    await self._close_partial(symbol, close_qty, record_pnl=True)
                    state["current_quantity"] -= close_qty
                    state["profit_taken"] = target["threshold"]
                    
                    if state["current_quantity"] <= 0:
                        state["status"] = "closed"
                        logger.info(f"Trend position fully closed via take profit: {symbol}")
                    return

    async def _check_trailing_stop(self, symbol: str, current_price: float):
        state = self._position_state[symbol]
        if state.get("status") != "open":
            return
        direction = state["direction"]
        entry_price = state["entry_price"]
        
        profit = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price
        
        if profit > 0:
            tier = get_currency_tier(symbol, self.config)
            if tier == "tier1":
                trailing = self._trailing_stop_tier1
            elif tier == "tier2":
                trailing = self._trailing_stop_tier2
            else:
                trailing = self._trailing_stop_tier3
            
            # P2: Chandelier Exit - 盈利>2%后启用更智能的追踪止损
            chandelier_stop = 0.0
            if self._chandelier_exit_enabled:
                chandelier_stop = await self._calculate_chandelier_exit(symbol, direction, current_price)
            
            new_trailing_stop = calculate_trailing_stop(entry_price, current_price, direction, 
                                                       trailing, self._trailing_stop_min)
            
            # 使用Chandelier Exit和传统追踪止损中更优的（更能保护利润的）
            if chandelier_stop > 0:
                if direction == "long":
                    effective_stop = max(new_trailing_stop, chandelier_stop)
                else:
                    effective_stop = min(new_trailing_stop, chandelier_stop)
            else:
                effective_stop = new_trailing_stop
            
            if direction == "long":
                if effective_stop > state.get("dynamic_stop_loss", entry_price * (1 - trailing)):
                    state["dynamic_stop_loss"] = effective_stop
                peak_price = state["peak_price"]
                if current_price <= state.get("dynamic_stop_loss", peak_price * (1 - trailing)):
                    state["exit_reason"] = "trailing_stop"
                    await self._close_position(symbol, force=True)
                    logger.info(f"Trend trailing stop triggered: {symbol}, SL: {state['dynamic_stop_loss']:.4f}")
            else:
                if effective_stop < state.get("dynamic_stop_loss", entry_price * (1 + trailing)):
                    state["dynamic_stop_loss"] = effective_stop
                trough_price = state["trough_price"]
                if current_price >= state.get("dynamic_stop_loss", trough_price * (1 + trailing)):
                    state["exit_reason"] = "trailing_stop"
                    await self._close_position(symbol, force=True)
                    logger.info(f"Trend trailing stop triggered: {symbol}, SL: {state['dynamic_stop_loss']:.4f}")

    async def _check_stop_loss(self, symbol: str, current_price: float):
        state = self._position_state[symbol]
        if state.get("status") != "open":
            return
        direction = state["direction"]
        entry_price = state["entry_price"]

        if self._volatility_adaptive_sl:
            atr = state.get("atr", 0)
            adjusted_sl = adjust_stop_loss_for_volatility(entry_price, direction, atr, self._false_break_threshold)
            state["adjusted_stop_loss"] = adjusted_sl
        else:
            adjusted_sl = entry_price * (1 - self._false_break_threshold) if direction == "long" else entry_price * (1 + self._false_break_threshold)

        # 硬性最大止损：无论ATR多大，止损不超过max_stop_loss_pct
        if direction == "long":
            max_sl = entry_price * (1 - self._max_stop_loss_pct)
            if adjusted_sl > max_sl:
                adjusted_sl = max_sl
                state["adjusted_stop_loss"] = adjusted_sl
        else:
            max_sl = entry_price * (1 + self._max_stop_loss_pct)
            if adjusted_sl < max_sl:
                adjusted_sl = max_sl
                state["adjusted_stop_loss"] = adjusted_sl

        # 保本止损：盈利达到阈值后将止损移至保本位
        profit = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price
        breakeven_trigger = self.config.get("strategies", {}).get("trend", {}).get("breakeven_trigger_pct", 0.015)
        breakeven_stop = self.config.get("strategies", {}).get("trend", {}).get("breakeven_stop_pct", 0.003)

        if profit >= breakeven_trigger:
            if direction == "long":
                breakeven_sl = entry_price * (1 + breakeven_stop)
                if breakeven_sl > adjusted_sl:
                    adjusted_sl = breakeven_sl
                    state["adjusted_stop_loss"] = adjusted_sl
            else:
                breakeven_sl = entry_price * (1 - breakeven_stop)
                if breakeven_sl < adjusted_sl:
                    adjusted_sl = breakeven_sl
                    state["adjusted_stop_loss"] = adjusted_sl

        loss = (entry_price - current_price) / entry_price if direction == "long" else (current_price - entry_price) / entry_price

        if direction == "long" and current_price <= adjusted_sl:
            state["exit_reason"] = "stop_loss"
            # P0: 使用StopLossManager统一执行止损（审计日志+强制执行）
            if self._stop_loss_manager:
                sl_executed = await self._stop_loss_manager.check_and_execute_stop_loss(
                    symbol=symbol,
                    strategy_name="trend",
                    current_price=current_price,
                    position_state=state
                )
                if not sl_executed:
                    # StopLossManager 返回 False 时降级到直接平仓
                    await self._close_position(symbol, force=True)
            else:
                await self._close_position(symbol, force=True)
            logger.info(f"Trend stop loss triggered: {symbol}, SL: {adjusted_sl:.4f}, Loss: {loss:.2%}")
        elif direction == "short" and current_price >= adjusted_sl:
            state["exit_reason"] = "stop_loss"
            if self._stop_loss_manager:
                sl_executed = await self._stop_loss_manager.check_and_execute_stop_loss(
                    symbol=symbol,
                    strategy_name="trend",
                    current_price=current_price,
                    position_state=state
                )
                if not sl_executed:
                    await self._close_position(symbol, force=True)
            else:
                await self._close_position(symbol, force=True)
            logger.info(f"Trend stop loss triggered: {symbol}, SL: {adjusted_sl:.4f}, Loss: {loss:.2%}")

    async def _check_trend_reversal(self, symbol: str, current_price: float):
        state = self._position_state[symbol]
        if state.get("status") != "open":
            return
        # P0-3: 反转检查对所有持仓生效（原逻辑 additions < 2 直接 return，导致单层仓位无法反平）
        klines = await self.okx_client.get_kline_async(symbol, "1H", limit=12)
        if len(klines) < 6:
            return

        closes = np.array([float(kline[4]) for kline in klines])
        highs = np.array([float(kline[2]) for kline in klines])
        lows = np.array([float(kline[3]) for kline in klines])

        recent_avg = np.mean(closes[-3:])
        prev_avg = np.mean(closes[-6:-3])

        adx, plus_di, minus_di = self._calculate_adx(highs, lows, closes)
        rsi = self._calculate_rsi(closes)

        # P0-3: 打分制反转检测 - 每个条件贡献 1 分，>=3 平仓 50%，>=4 全部平仓
        if state["direction"] == "long":
            conditions = [
                recent_avg < prev_avg * 0.99,
                minus_di > plus_di,
                adx > self._reversal_adx_threshold,
                rsi > 50,
            ]
        else:
            conditions = [
                recent_avg > prev_avg * 1.01,
                plus_di > minus_di,
                adx > self._reversal_adx_threshold,
                rsi < 50,
            ]

        reversal_score = sum(1 for c in conditions if c)

        # 15m 级别确认：score >= 3 时拉取 15m K线验证反转方向
        # 15m 确认 → 维持原动作；15m 未确认 → 降级（全平→半仓，半仓→不动）
        confirmed_15m = False
        if reversal_score >= 3:
            confirmed_15m = await self._confirm_reversal_15m(symbol, state["direction"])

        if reversal_score >= 4 and confirmed_15m:
            # 1H 强反转 + 15m 确认 → 全部平仓
            await self._close_position(symbol, force=True)
            state["status"] = "reversed"
            logger.info(
                f"Trend reversal (full close) {symbol} {state['direction']}: "
                f"score={reversal_score}/4, 15m=confirmed, ADX={adx:.1f}, RSI={rsi:.1f}"
            )
        elif reversal_score >= 4 and not confirmed_15m:
            # 1H 强反转但 15m 未确认 → 降级为 50% 平仓
            state["exit_reason"] = "reversal"
            close_qty = state["current_quantity"] * 0.5
            if close_qty > 0:
                await self._close_partial(symbol, close_qty, record_pnl=True)
                state["current_quantity"] -= close_qty
                if state["current_quantity"] <= 0:
                    state["status"] = "reversed"
            logger.info(
                f"Trend reversal (50% close, 15m unconfirmed) {symbol} {state['direction']}: "
                f"score={reversal_score}/4, 15m=rejected, ADX={adx:.1f}, RSI={rsi:.1f}"
            )
        elif reversal_score >= 3 and confirmed_15m:
            # 1H 中等反转 + 15m 确认 → 平仓 50%
            state["exit_reason"] = "reversal"
            close_qty = state["current_quantity"] * 0.5
            if close_qty > 0:
                await self._close_partial(symbol, close_qty, record_pnl=True)
                state["current_quantity"] -= close_qty
                if state["current_quantity"] <= 0:
                    state["status"] = "reversed"
            logger.info(
                f"Trend reversal (50% close) {symbol} {state['direction']}: "
                f"score={reversal_score}/4, 15m=confirmed, ADX={adx:.1f}, RSI={rsi:.1f}"
            )
        elif reversal_score >= 3 and not confirmed_15m:
            # 1H 中等反转但 15m 未确认 → 不操作，继续观察
            logger.info(
                f"Trend reversal (skipped, 15m unconfirmed) {symbol} {state['direction']}: "
                f"score={reversal_score}/4, 15m=rejected, ADX={adx:.1f}, RSI={rsi:.1f}"
            )

    async def _confirm_reversal_15m(self, symbol: str, direction: str) -> bool:
        """15m 级别反转确认：EMA9/EMA21 交叉 + 近 3 根收盘趋势"""
        klines_15m = await self.okx_client.get_kline_async(symbol, "15m", limit=24)
        if len(klines_15m) < 22:
            return False

        closes_15m = np.array([float(k[4]) for k in klines_15m])
        ema9 = self._calculate_ema(closes_15m, 9)
        ema21 = self._calculate_ema(closes_15m, 21)

        # 15m EMA 交叉方向确认
        if direction == "long":
            ema_confirms = ema9 < ema21
        else:
            ema_confirms = ema9 > ema21

        # 近 3 根 15m 收盘趋势确认
        recent_3 = closes_15m[-3:]
        if direction == "long":
            trend_confirms = recent_3[-1] < recent_3[0]
        else:
            trend_confirms = recent_3[-1] > recent_3[0]

        return ema_confirms and trend_confirms

    async def _check_multiple_take_profit(self, symbol: str, current_price: float):
        """多级止盈：tp1/tp2/tp3 三级渐进止盈
        - tp1: 盈利达到 tp1_pct，平仓 tp1_ratio 比例
        - tp2: 盈利达到 tp2_pct，平仓 tp2_ratio 比例
        - tp3: 剩余仓位跟随 tp3_trailing_pct 移动止损
        """
        if not self._take_profit_enabled:
            return
        state = self._position_state[symbol]
        if state.get("status") != "open":
            return
        direction = state["direction"]
        entry_price = state["entry_price"]
        profit = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price

        if profit <= 0:
            return

        # TP1: 第一级止盈
        if not state.get("tp1_triggered", False) and profit >= self._tp1_pct:
            close_qty = state["current_quantity"] * self._tp1_ratio
            if close_qty > 0:
                state["exit_reason"] = "tp1"
                await self._close_partial(symbol, close_qty, record_pnl=True)
                state["current_quantity"] -= close_qty
                state["tp1_triggered"] = True
                logger.info(f"Trend TP1 triggered: {symbol} profit={profit:.2%}, closed {self._tp1_ratio:.0%}")
                if state["current_quantity"] <= 0:
                    state["status"] = "closed"
                    return

        # TP2: 第二级止盈
        if not state.get("tp2_triggered", False) and profit >= self._tp2_pct:
            close_qty = state["current_quantity"] * self._tp2_ratio
            if close_qty > 0:
                state["exit_reason"] = "tp2"
                await self._close_partial(symbol, close_qty, record_pnl=True)
                state["current_quantity"] -= close_qty
                state["tp2_triggered"] = True
                logger.info(f"Trend TP2 triggered: {symbol} profit={profit:.2%}, closed {self._tp2_ratio:.0%}")
                if state["current_quantity"] <= 0:
                    state["status"] = "closed"
                    return

        # TP3: 剩余仓位移动止损（基于 tp3_trailing_pct）
        if state.get("tp3_triggered", False):
            return
        if state.get("tp1_triggered") and state.get("tp2_triggered") and state["current_quantity"] > 0:
            state["tp3_triggered"] = True
            logger.info(f"Trend TP3 activated: {symbol} remaining qty={state['current_quantity']:.4f}, trailing={self._tp3_trailing_pct:.2%}")

    async def _check_time_exit(self, symbol: str, current_price: float):
        """时间退出：持仓超过最大时长后强制平仓
        - 超过 time_exit_after_hours：部分平仓 time_exit_partial_pct
        - 超过 max_hold_hours：全部平仓
        """
        if not self._time_exit_enabled:
            return
        state = self._position_state[symbol]
        if state.get("status") != "open":
            return

        entry_time = self._position_entry_times.get(symbol)
        if not entry_time:
            return

        hold_hours = (datetime.now() - entry_time).total_seconds() / 3600

        # 超过最大持仓时长：全部平仓
        if hold_hours >= self._max_hold_hours:
            state["exit_reason"] = "time_exit_full"
            await self._close_position(symbol, force=True)
            state["status"] = "closed"
            logger.info(f"Trend time exit (full): {symbol} held {hold_hours:.1f}h >= {self._max_hold_hours}h")
            return

        # 超过收紧止损时长：部分平仓
        if hold_hours >= self._time_exit_after_hours and not state.get("time_exit_partial_triggered", False):
            close_qty = state["current_quantity"] * self._time_exit_partial_pct
            if close_qty > 0:
                state["exit_reason"] = "time_exit_partial"
                await self._close_partial(symbol, close_qty, record_pnl=True)
                state["current_quantity"] -= close_qty
                state["time_exit_partial_triggered"] = True
                logger.info(f"Trend time exit (partial): {symbol} held {hold_hours:.1f}h >= {self._time_exit_after_hours}h, closed {self._time_exit_partial_pct:.0%}")
                if state["current_quantity"] <= 0:
                    state["status"] = "closed"

    async def _check_volatility_spike_exit(self, symbol: str, current_price: float):
        """波动率尖峰退出：检测到异常波动率放大时部分平仓
        - 当前ATR / 近N期ATR均值的比值 > vol_spike_threshold
        - 触发后部分平仓 vol_stop_partial_pct，并锁仓 volatility_lockout_minutes
        """
        if not self._volatility_stop_enabled:
            return
        state = self._position_state[symbol]
        if state.get("status") != "open":
            return

        # 检查是否在锁仓期内
        lockout = self._volatility_lockout_until.get(symbol)
        if lockout and datetime.now() < lockout:
            return

        direction = state["direction"]
        atr = state.get("atr", 0)
        entry_price = state["entry_price"]
        if atr <= 0 or entry_price <= 0:
            return

        # 获取最近20根1H K线计算ATR均值
        try:
            klines = await self.okx_client.get_kline_async(symbol, "1H", limit=25)
            if len(klines) < 20:
                return
            highs = np.array([float(k[2]) for k in klines[-20:]])
            lows = np.array([float(k[3]) for k in klines[-20:]])
            closes = np.array([float(k[4]) for k in klines[-20:]])
            avg_atr = self._calculate_atr(highs, lows, closes)
            if avg_atr <= 0:
                return
        except Exception:
            return

        atr_ratio = atr / avg_atr

        if atr_ratio >= self._vol_spike_threshold:
            close_qty = state["current_quantity"] * self._vol_stop_partial_pct
            if close_qty > 0:
                state["exit_reason"] = "vol_spike"
                await self._close_partial(symbol, close_qty, record_pnl=True)
                state["current_quantity"] -= close_qty
                # 锁仓
                self._volatility_lockout_until[symbol] = datetime.now() + timedelta(minutes=self._volatility_lockout_minutes)
                logger.info(
                    f"Trend volatility spike exit: {symbol} ATR ratio={atr_ratio:.2f} >= {self._vol_spike_threshold}, "
                    f"closed {self._vol_stop_partial_pct:.0%}, locked until {self._volatility_lockout_until[symbol].strftime('%H:%M:%S')}"
                )
                if state["current_quantity"] <= 0:
                    state["status"] = "closed"

    def _get_dynamic_adx_threshold(self, symbol: str) -> float:
        """动态ADX阈值：根据市场波动率自适应调整ADX阈值
        - 高波动（volatility=high）：使用 adx_vol_high（更严格的趋势确认）
        - 低波动（volatility=low）：使用 adx_vol_low（更宽松的入场条件）
        - 正常：使用 adx_vol_normal
        """
        if not self._dynamic_adx_threshold:
            return self._adx_threshold
        market_state = self._market_state.get(symbol, {})
        volatility = market_state.get("volatility", "normal")
        if volatility == "high":
            return self._adx_vol_high
        elif volatility == "low":
            return self._adx_vol_low
        else:
            return self._adx_vol_normal

    def _update_trend_performance(self, pnl: float):
        """P33: 更新趋势策略性能追踪（含连续盈亏、PnL序列、回撤）"""
        perf = self._trend_performance
        perf["count"] += 1
        perf["total_pnl"] += pnl
        if pnl > 0:
            perf["wins"] += 1
            perf["total_profit"] = perf.get("total_profit", 0) + pnl
            perf["consecutive_wins"] = perf.get("consecutive_wins", 0) + 1
            perf["consecutive_losses"] = 0
        else:
            perf["losses"] += 1
            perf["total_loss"] = perf.get("total_loss", 0) + abs(pnl)
            perf["consecutive_losses"] = perf.get("consecutive_losses", 0) + 1
            perf["consecutive_wins"] = 0

        pnl_list = perf.setdefault("pnl_list", [])
        pnl_list.append(pnl)
        if len(pnl_list) > 200:
            perf["pnl_list"] = pnl_list[-200:]

        cum_pnl = float(sum(pnl_list))
        perf["peak_equity"] = max(perf.get("peak_equity", 0), cum_pnl)
        perf["max_drawdown"] = max(perf.get("max_drawdown", 0), perf["peak_equity"] - cum_pnl)

    def _calculate_kelly_position(self, base_position: float, symbol: str) -> float:
        """P33: 企业级凯利仓位计算（接入 AdaptiveKelly）

        用企业级凯利引擎替代/增强静态自适应仓位：
        - 贝叶斯胜率收缩 + 赔率收缩 + 市场状态 + 回撤 + 连续盈亏 + 样本量折扣

        Returns:
            调整后的仓位
        """
        perf = self._trend_performance
        total = perf.get("wins", 0) + perf.get("losses", 0)
        if total < 10:
            return base_position

        win_rate = perf["wins"] / total if total > 0 else 0
        avg_win = perf.get("total_profit", 0) / perf["wins"] if perf["wins"] > 0 else 0
        avg_loss = perf.get("total_loss", 0) / perf["losses"] if perf["losses"] > 0 else 0
        if avg_win <= 0 or avg_loss <= 0:
            return base_position

        state = self._market_state.get(symbol, {}).get("state", "unknown")
        regime = map_market_state_to_regime(state)

        total_equity = self._get_effective_capital()
        peak_equity = float(perf.get("peak_equity", 0.0))
        cum_pnl = float(sum(perf.get("pnl_list", [])))
        drawdown_pct = max(0.0, (peak_equity - cum_pnl) / total_equity) if total_equity > 0 else 0.0

        result = self._adaptive_kelly.compute_kelly(
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            regime=regime,
            drawdown=drawdown_pct,
            consecutive_wins=int(perf.get("consecutive_wins", 0)),
            consecutive_losses=int(perf.get("consecutive_losses", 0)),
            trade_count=total,
        )
        final_kelly = result["final_kelly"]

        if final_kelly <= 0:
            adjusted = base_position * 0.5  # 无边际优势，减半
        else:
            adjusted = base_position * (1.0 + (final_kelly - 0.05) * 4.0)
            adjusted = max(base_position * 0.3, min(adjusted, base_position * 2.0))

        logger.debug(
            f"Enterprise Kelly sizing [trend]: win_rate={win_rate:.2%} → shrunk={result['wilson_win_rate']:.2%}, "
            f"b={result['shrunk_odds']:.2f}, kelly={final_kelly:.3f}, regime={regime}, "
            f"base={base_position:.2f}, adjusted={adjusted:.2f}"
        )
        return adjusted

    async def _close_partial(self, symbol: str, quantity: float, record_pnl: bool = False):
        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]
        leverage = tier_settings["leverage_max"]
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        
        price = float(ticker["last"])
        direction = self._position_state[symbol]["direction"]
        
        close_direction = "sell" if direction == "long" else "buy"

        state = self._position_state[symbol]
        exit_reason = state.get("exit_reason") or "manual"
        signal_type = "trend_close"
        if exit_reason == "stop_loss":
            signal_type = "stop_loss"
        elif exit_reason == "trailing_stop":
            signal_type = "trailing_stop"
        elif exit_reason == "take_profit":
            signal_type = "take_profit"
        elif exit_reason == "trend_reversal":
            signal_type = "trend_reversal"
        elif exit_reason == "profit_protection":
            signal_type = "profit_protection"
        elif exit_reason == "time_exit_full":
            signal_type = "time_exit"
        elif exit_reason == "vol_spike":
            signal_type = "vol_spike"

        # 记录部分平仓的 PNL（止盈/反转场景下调用时 record_pnl=True）
        if record_pnl and exit_reason in ("take_profit", "reversal"):
            entry_price = state.get("entry_price", 0)
            pnl_per_unit = (price - entry_price) if direction == "long" else (entry_price - price)
            partial_pnl = pnl_per_unit * quantity
            self._update_trend_performance(partial_pnl)
            logger.info(f"Trend partial close PNL recorded: {symbol} qty={quantity:.4f} pnl={partial_pnl:.4f}")

        signal = Signal(
            symbol=symbol,
            strategy_name="trend",
            signal_type=signal_type,
            direction=close_direction,
            price=price,
            quantity=quantity,
            leverage=leverage,
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        self.redis_cache.publish_signal({
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
                "direction": signal.direction,
                "price": signal.price,
                "quantity": signal.quantity,
                "leverage": signal.leverage,
                "stop_loss": None,
                "take_profit": None,
                "confidence": signal.confidence,
                "exit_reason": exit_reason,
                "timestamp": signal.timestamp.isoformat()
            }
        })

    async def _close_position(self, symbol: str, force: bool = False):
        state = self._position_state[symbol]
        # 最小持仓时长检查：避免闪开闪平吃手续费
        # 历史数据显示 25 笔 < 5min 平仓全部 pnl=0，纯损耗手续费
        # force=True 时跳过检查（用于止损/强平场景）
        if not force:
            entry_time = state.get("entry_time")
            if entry_time:
                hold_seconds = (datetime.now() - entry_time).total_seconds()
                if hold_seconds < self._min_hold_minutes * 60:
                    logger.debug(
                        f"Trend close skipped: {symbol} held {hold_seconds:.0f}s < min_hold {self._min_hold_minutes}min"
                    )
                    return
        # P0-1: 平仓前立即撤销该品种的所有条件单（TP/SL），防止平仓后120s内条件单仍挂在网上
        if hasattr(self, "_conditional_order_manager") and self._conditional_order_manager:
            try:
                self._conditional_order_manager.cancel_all_conditional_orders(symbol)
                logger.debug(f"P0-1: Canceled conditional orders for {symbol} before close")
            except Exception as e:
                logger.error(f"P0-1: Failed to cancel conditional orders for {symbol}: {e}")
        await self._close_partial(symbol, state["current_quantity"])
        # P0: 保留 "reversed" 状态，不覆盖为 "closed"，以便 get_stats 区分正常平仓和反转平仓
        if state.get("status") != "reversed":
            state["status"] = "closed"
        
        pnl = state.get("current_profit", 0) * state.get("current_quantity", 0) * state.get("entry_price", 0)
        self._update_trend_performance(pnl)
        
        # 清理状态：移除入场时间记录
        if symbol in self._position_entry_times:
            del self._position_entry_times[symbol]
    
    def get_stats(self) -> Dict[str, Any]:
        open_positions = sum(1 for p in self._position_state.values() if p["status"] == "open")
        total_additions = sum(p.get("additions", 0) for p in self._position_state.values())
        
        return {
            "strategy": "trend",
            "is_enabled": self._enabled,
            "open_positions": open_positions,
            "total_additions": total_additions,
            "monitored_symbols": len(self._all_symbols),
            "trend_performance": self._trend_performance,
            "multi_timeframe_confirmation": self._multi_timeframe_confirmation,
            "breakout_confirmation": self._breakout_confirmation,
            "position_details": [{
                "symbol": s,
                "direction": p["direction"],
                "entry_price": p["entry_price"],
                "current_quantity": p["current_quantity"],
                "additions": p.get("additions", 0),
                "status": p["status"],
                "atr": p.get("atr", 0),
                "current_profit": p.get("current_profit", 0)
            } for s, p in self._position_state.items() if p["status"] == "open"],
            # ghost_close 专项 Phase 4: 观测指标
            "use_fill_driven_position": self._use_fill_driven_position,
            "pending_entry_count": len(self._pending_entries),
            "fill_callback_hits": self._fill_callback_hits,
            "pending_timeout_cleaned": self._pending_timeout_cleaned,
            "ghost_close_count": sum(
                1 for p in self._position_state.values()
                if p.get("close_reason") == "ghost_position_cleaned"
            ),
        }

    # ===================== 强化：多周期趋势共振检测 =====================

    async def _detect_multi_timeframe_resonance(self, symbol: str) -> Dict[str, Any]:
        """多周期趋势共振检测：检查多个时间框架是否指向同一方向
        返回: {"direction": "long/short/None", "resonance_score": 0-1, "agreeing_periods": int, "total_periods": int}
        """
        try:
            periods = self._confirmation_periods
            signals = []
            confidences = []

            for period in periods:
                indicators = self._indicator_cache.get(symbol, {}).get(period)
                if not indicators:
                    continue

                signal, conf = self._analyze_period_with_indicators(indicators)
                if signal:
                    signals.append(signal)
                    confidences.append(conf)

            if not signals:
                return {"direction": None, "resonance_score": 0, "agreeing_periods": 0, "total_periods": len(periods)}

            bull_count = sum(1 for s in signals if s == "long")
            bear_count = sum(1 for s in signals if s == "short")
            total = len(signals)

            if bull_count > bear_count and bull_count >= max(2, total * 0.6):
                direction = "long"
                agree_count = bull_count
            elif bear_count > bull_count and bear_count >= max(2, total * 0.6):
                direction = "short"
                agree_count = bear_count
            else:
                return {"direction": None, "resonance_score": 0, "agreeing_periods": 0, "total_periods": total}

            avg_conf = sum(c for s, c in zip(signals, confidences) if s == direction) / agree_count if agree_count > 0 else 0
            resonance_score = (agree_count / total) * 0.6 + avg_conf * 0.4

            return {
                "direction": direction,
                "resonance_score": min(1.0, resonance_score),
                "agreeing_periods": agree_count,
                "total_periods": total,
                "avg_confidence": avg_conf
            }
        except Exception as e:
            logger.debug(f"Multi-timeframe resonance error for {symbol}: {e}")
            return {"direction": None, "resonance_score": 0, "agreeing_periods": 0, "total_periods": 0}

    # ===================== 强化：背离检测 =====================

    def _detect_divergence(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                           rsi_values: np.ndarray = None) -> Dict[str, Any]:
        """检测RSI和MACD背离
        Returns: {"rsi": "bearish"/"bullish"/None, "macd": "bearish"/"bullish"/None}
        """
        result = {"rsi": None, "macd": None}
        try:
            if len(closes) < 30:
                return result

            mid = len(closes) // 2
            front_closes = closes[:mid]
            back_closes = closes[mid:]

            front_high = float(np.max(front_closes))
            back_high = float(np.max(back_closes))
            front_low = float(np.min(front_closes))
            back_low = float(np.min(back_closes))

            # RSI背离检测
            if rsi_values is not None and len(rsi_values) >= 30:
                front_rsi = rsi_values[:mid]
                back_rsi = rsi_values[mid:]
                front_rsi_high = float(np.max(front_rsi))
                back_rsi_high = float(np.max(back_rsi))
                front_rsi_low = float(np.min(front_rsi))
                back_rsi_low = float(np.min(back_rsi))

                # 顶背离：价格新高但RSI走低
                if back_high > front_high and back_rsi_high < front_rsi_high:
                    result["rsi"] = "bearish"
                # 底背离：价格新低但RSI走高
                elif back_low < front_low and back_rsi_low > front_rsi_low:
                    result["rsi"] = "bullish"

            # MACD背离检测
            if len(closes) >= 35:
                macd_line, _, _ = self._calculate_macd(closes)
                if macd_line != 0:
                    # 简化MACD背离：比较最近两段的MACD极值
                    # 注：完整检测需要MACD历史序列，此处基于现有MACD值做简化判断
                    pass  # MACD序列背离在主检测流程中使用

            # 无RSI数据时的简化价格背离
            if rsi_values is None:
                front_range = front_high - front_low
                back_range = back_high - back_low
                if back_high > front_high and back_range < front_range * 0.7:
                    result["rsi"] = "bearish"
                if back_low < front_low and back_range < front_range * 0.7:
                    result["rsi"] = "bullish"

        except Exception as e:
            logger.debug(f"Divergence detection error: {e}")
        return result

    def _analyze_market_structure(self, highs: np.ndarray, lows: np.ndarray,
                                  closes: np.ndarray, window: int = 5) -> str:
        """分析市场结构：基于HH/HL/LH/LL的摆动点分析
        
        Returns: strong_uptrend/uptrend/ranging/downtrend/strong_downtrend
        """
        try:
            if len(highs) < window * 2 + 1 or len(lows) < window * 2 + 1:
                return "ranging"

            # 寻找摆动高点（用closes近似）
            n = len(closes)
            swing_highs = []
            swing_lows = []

            for i in range(window, n - window):
                is_high = all(closes[i] >= closes[j] for j in range(i - window, i + window + 1) if j != i)
                is_low = all(closes[i] <= closes[j] for j in range(i - window, i + window + 1) if j != i)
                if is_high:
                    swing_highs.append(float(closes[i]))
                if is_low:
                    swing_lows.append(float(closes[i]))

            if len(swing_highs) < 3 or len(swing_lows) < 3:
                return "ranging"

            recent_highs = swing_highs[-3:]
            recent_lows = swing_lows[-3:]

            hh = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
            hl = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
            lh = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
            ll = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])

            total = len(recent_highs) - 1

            if hh == total and hl == total:
                return "strong_uptrend"
            elif hh >= total * 0.7 and hl >= total * 0.7:
                return "uptrend"
            elif lh == total and ll == total:
                return "strong_downtrend"
            elif lh >= total * 0.7 and ll >= total * 0.7:
                return "downtrend"
            else:
                return "ranging"
        except Exception:
            return "ranging"

    # ===================== 强化：Chandelier Exit 追踪止损 =====================

    async def _calculate_chandelier_exit(self, symbol: str, direction: str, current_price: float) -> float:
        """Chandelier Exit：基于ATR的动态追踪止损
        
        - 多头：Highest High(N) - ATR * multiplier
        - 空头：Lowest Low(N) + ATR * multiplier
        
        关键参数：ATR周期22（月波动），乘数3.0（经典设置）
        盈利 > 2% 后启用，让利润奔跑同时保护浮盈
        """
        try:
            state = self._position_state.get(symbol)
            if not state or state.get("status") != "open":
                return 0.0

            profit = state.get("current_profit", 0)
            
            # 盈利 > 阈值才启用Chandelier Exit（避免过早触发）
            if profit < self._chandelier_profit_threshold:
                return 0.0

            atr = state.get("atr", 0)
            if atr <= 0:
                return 0.0

            # 获取指定周期K线计算Highest High / Lowest Low
            klines = await self.okx_client.get_kline_async(symbol, "1H", limit=self._chandelier_atr_period + 3)
            if len(klines) < self._chandelier_atr_period:
                return 0.0

            klines = sorted(klines, key=lambda k: int(k[0]))
            highs_n = np.array([float(k[2]) for k in klines[-self._chandelier_atr_period:]])
            lows_n = np.array([float(k[3]) for k in klines[-self._chandelier_atr_period:]])

            # 自适应乘数：盈利越多，乘数越小（保护更多利润）
            if profit > 0.10:
                multiplier = self._chandelier_multiplier_base * 0.67
            elif profit > 0.06:
                multiplier = self._chandelier_multiplier_base * 0.83
            elif profit > 0.04:
                multiplier = self._chandelier_multiplier_base * 0.93
            else:
                multiplier = self._chandelier_multiplier_base

            if direction == "long":
                highest_high = float(np.max(highs_n))
                chandelier_stop = highest_high - atr * multiplier
                return chandelier_stop
            else:
                lowest_low = float(np.min(lows_n))
                chandelier_stop = lowest_low + atr * multiplier
                return chandelier_stop

        except Exception as e:
            logger.debug(f"Chandelier exit calculation error: {e}")
            return 0.0

    # ===================== 强化：趋势强度分级过滤 =====================

    def _calculate_trend_strength_level(self, indicators: Dict[str, float]) -> str:
        """趋势强度分级：strong / moderate / weak / none
        基于ADX、DI差、均线斜率、MACD柱状图综合判断
        """
        try:
            adx = indicators.get("adx", 0)
            plus_di = indicators.get("+di", 0)
            minus_di = indicators.get("-di", 0)
            ma20 = indicators.get("ma20", 0)
            ma50 = indicators.get("ma50", 0)
            histogram = indicators.get("histogram", 0)
            recent_close = indicators.get("recent_close", 0)

            score = 0

            if adx > 35:
                score += 2
            elif adx > 25:
                score += 1

            di_diff = abs(plus_di - minus_di)
            if di_diff > 20:
                score += 2
            elif di_diff > 10:
                score += 1

            if ma50 > 0:
                ma_slope = (ma20 - ma50) / ma50
                if abs(ma_slope) > 0.05:
                    score += 2
                elif abs(ma_slope) > 0.02:
                    score += 1

            if recent_close > 0:
                hist_ratio = abs(histogram) / recent_close
                if hist_ratio > 0.002:
                    score += 1

            if score >= 5:
                return "strong"
            elif score >= 3:
                return "moderate"
            elif score >= 1:
                return "weak"
            else:
                return "none"
        except Exception:
            return "none"

    # ===================== 强化：自适应参数学习 =====================

    def _learn_from_trade_result(self, symbol: str, profit: float, hold_minutes: float):
        """从交易结果中学习，自适应调整参数
        - 盈利交易：加强类似信号的权重
        - 亏损交易：提高过滤阈值
        """
        try:
            if not hasattr(self, '_param_learning'):
                self._param_learning = {
                    "total_trades": 0,
                    "winning_params": {"adx_threshold_bonus": 0, "rsi_mild_bonus": 0},
                    "losing_params": {"adx_threshold_penalty": 0, "quality_min_increase": 0},
                }

            self._param_learning["total_trades"] += 1

            if profit > 0:
                self._param_learning["winning_params"]["adx_threshold_bonus"] = min(
                    0.1, self._param_learning["winning_params"]["adx_threshold_bonus"] + 0.002
                )
            else:
                self._param_learning["losing_params"]["quality_min_increase"] = min(
                    0.15, self._param_learning["losing_params"]["quality_min_increase"] + 0.005
                )

            if self._param_learning["total_trades"] % 20 == 0:
                self._param_learning["winning_params"]["adx_threshold_bonus"] *= 0.9
                self._param_learning["losing_params"]["quality_min_increase"] *= 0.9

        except Exception as e:
            logger.debug(f"Param learning error: {e}")

    def _get_adaptive_params(self) -> Dict[str, float]:
        """获取自适应调整后的参数"""
        default_min_quality = self._min_signal_quality

        if not hasattr(self, '_param_learning'):
            return {"min_signal_quality": default_min_quality}

        quality_increase = self._param_learning["losing_params"].get("quality_min_increase", 0)

        return {
            "min_signal_quality": min(0.75, default_min_quality + quality_increase),
        }

    # ===================== 强化：成交量趋势确认 =====================

    def _check_volume_trend_confirmation(self, symbol: str, direction: str) -> float:
        """成交量趋势确认：价格上涨+放量/价格下跌+放量 = 确认
        返回确认分数 0-1，越高越确认
        """
        try:
            indicators = self._indicator_cache.get(symbol, {})
            if not indicators:
                return 0.5

            scores = []
            for period in self._confirmation_periods:
                ind = indicators.get(period)
                if not ind:
                    continue

                recent_volume = ind.get("recent_volume", 0)
                volume_ma20 = ind.get("volume_ma20", 0)
                recent_close = ind.get("recent_close", 0)
                ma20 = ind.get("ma20", 0)

                if volume_ma20 <= 0 or recent_close <= 0 or ma20 <= 0:
                    continue

                volume_ratio = recent_volume / volume_ma20
                price_above_ma = recent_close > ma20

                if direction == "long" and price_above_ma and volume_ratio > 1.2:
                    score = min(1.0, (volume_ratio - 1.0) * 2)
                elif direction == "short" and not price_above_ma and volume_ratio > 1.2:
                    score = min(1.0, (volume_ratio - 1.0) * 2)
                elif volume_ratio < 0.8:
                    score = 0.2
                else:
                    score = 0.5

                scores.append(score)

            if scores:
                return float(np.mean(scores))
            return 0.5
        except Exception as e:
            logger.debug(f"Volume trend confirmation error: {e}")
            return 0.5

    # ===================== 强化：布林带突破确认 =====================

    def _check_bollinger_breakout(self, indicators: Dict[str, float], direction: str) -> float:
        """布林带突破确认：价格突破上轨（做多）或下轨（做空）+ 带宽扩张
        返回确认分数 0-1
        """
        try:
            bb_upper = indicators.get("bb_upper", 0)
            bb_lower = indicators.get("bb_lower", 0)
            bb_middle = indicators.get("bb_middle", 0)
            recent_close = indicators.get("recent_close", 0)

            if bb_upper <= 0 or bb_lower <= 0 or bb_middle <= 0 or recent_close <= 0:
                return 0.5

            bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0

            if direction == "long":
                distance_to_upper = (bb_upper - recent_close) / recent_close
                if recent_close >= bb_upper:
                    score = 0.9
                elif distance_to_upper < 0.01:
                    score = 0.7
                elif recent_close > bb_middle:
                    score = 0.5
                else:
                    score = 0.2
            else:
                distance_to_lower = (recent_close - bb_lower) / recent_close
                if recent_close <= bb_lower:
                    score = 0.9
                elif distance_to_lower < 0.01:
                    score = 0.7
                elif recent_close < bb_middle:
                    score = 0.5
                else:
                    score = 0.2

            if bb_width > 0.1:
                score = min(1.0, score + 0.1)

            return score
        except Exception as e:
            logger.debug(f"Bollinger breakout error: {e}")
            return 0.5

    # ===================== 强化：综合信号质量评分 =====================

    def _calculate_composite_signal_quality(self, symbol: str, direction: str,
                                            period_scores: List[Dict[str, Any]]) -> float:
        """综合信号质量评分：融合多维度确认
        1. 多周期共振 (25%)
        2. 趋势强度 (20%)
        3. 成交量确认 (20%)
        4. 布林带突破 (15%)
        5. 市场状态适配 (10%)
        6. 资金费率 (10%)
        """
        try:
            if not period_scores:
                return self._min_signal_quality

            sub_scores = {}

            resonance = 0.5
            bull_periods = sum(1 for ps in period_scores if ps["signal"] == "long")
            bear_periods = sum(1 for ps in period_scores if ps["signal"] == "short")
            total = len(period_scores)
            if direction == "long" and total > 0:
                resonance = bull_periods / total
            elif direction == "short" and total > 0:
                resonance = bear_periods / total
            sub_scores["resonance"] = resonance

            avg_adx = np.mean([ps["indicators"].get("adx", 0) for ps in period_scores if ps["indicators"]])
            if avg_adx > 35:
                sub_scores["trend_strength"] = 0.9
            elif avg_adx > 25:
                sub_scores["trend_strength"] = 0.7
            elif avg_adx > 15:
                sub_scores["trend_strength"] = 0.5
            else:
                sub_scores["trend_strength"] = 0.3

            volume_score = self._check_volume_trend_confirmation(symbol, direction)
            sub_scores["volume"] = volume_score

            if period_scores:
                latest = period_scores[-1]["indicators"]
                bb_score = self._check_bollinger_breakout(latest, direction)
                sub_scores["bollinger"] = bb_score
            else:
                sub_scores["bollinger"] = 0.5

            market_state = self._market_state.get(symbol, {})
            state = market_state.get("state", "unknown")
            if state == "trending":
                sub_scores["market_state"] = 0.9
            elif state == "ranging":
                sub_scores["market_state"] = 0.3
            else:
                sub_scores["market_state"] = 0.6

            try:
                quality = self._funding_enhancer.get_quality_sync(symbol, direction)
                if quality is None:
                    sub_scores["funding"] = 0.7
                else:
                    # 质量分 [-1,1] 映射到子分 0.5~0.9（中性 0.7）
                    sub_scores["funding"] = 0.7 + 0.2 * quality
            except Exception:
                sub_scores["funding"] = 0.7

            weights = {
                "resonance": 0.25,
                "trend_strength": 0.20,
                "volume": 0.20,
                "bollinger": 0.15,
                "market_state": 0.10,
                "funding": 0.10,
            }

            composite = sum(sub_scores[k] * weights[k] for k in weights if k in sub_scores)

            return min(0.95, max(0.1, composite))
        except Exception as e:
            logger.debug(f"Composite signal quality error: {e}")
            return self._min_signal_quality

    # ===================== 强化：趋势追踪增强退出 =====================

    def _calculate_enhanced_trailing_stop(self, symbol: str, current_price: float) -> float:
        """增强追踪止损：基于ATR + 波动率 + 盈利保护
        1. 盈利越多，止盈位越远（让利润奔跑）
        2. 波动率越高，止损越宽
        3. 趋势越强，止损越宽
        """
        try:
            state = self._position_state.get(symbol)
            if not state:
                return 0.0

            direction = state["direction"]
            entry_price = state["entry_price"]
            atr = state.get("atr", 0.01)
            profit = state.get("current_profit", 0)

            indicators = {}
            for period in self._confirmation_periods:
                ind = self._indicator_cache.get(symbol, {}).get(period)
                if ind:
                    indicators = ind
                    break

            adx = indicators.get("adx", 20)

            base_atr_multiplier = self._atr_multiplier

            if profit > 0.05:
                profit_protection_factor = 1.0 + min(0.5, profit * 4)
            elif profit > 0.02:
                profit_protection_factor = 1.0 + (profit - 0.02) * 5
            else:
                profit_protection_factor = 1.0

            if adx > 35:
                trend_factor = 1.3
            elif adx > 25:
                trend_factor = 1.1
            else:
                trend_factor = 1.0

            effective_multiplier = base_atr_multiplier * profit_protection_factor * trend_factor

            if direction == "long":
                trailing_stop = current_price - atr * effective_multiplier
                if profit > 0.02:
                    min_protection = entry_price * (1 + profit * 0.5)
                    trailing_stop = max(trailing_stop, min_protection)
            else:
                trailing_stop = current_price + atr * effective_multiplier
                if profit > 0.02:
                    min_protection = entry_price * (1 - profit * 0.5)
                    trailing_stop = min(trailing_stop, min_protection)

            return trailing_stop
        except Exception as e:
            logger.debug(f"Enhanced trailing stop error: {e}")
            return 0.0

    # ===================== 增强趋势确认 =====================

    async def _check_multi_timeframe_direction(self, symbol: str, direction: str) -> bool:
        """多时间框架方向确认：至少 2/3 周期 (1h, 4h, 1d) 方向一致

        Returns:
            bool: 是否通过多时间框架确认
        """
        try:
            mtf_periods = ["1H", "4H", "1D"]
            agree_count = 0
            total_checked = 0

            for period in mtf_periods:
                try:
                    klines = await self.okx_client.get_kline_async(symbol, period, limit=120)
                    if len(klines) < 50:
                        continue
                    indicators = self._calculate_all_indicators(klines, symbol)
                    if not indicators:
                        continue

                    ma20 = indicators.get("ma20", 0)
                    ma50 = indicators.get("ma50", 0)
                    adx = indicators.get("adx", 0)
                    plus_di = indicators.get("+di", 0)
                    minus_di = indicators.get("-di", 0)
                    recent_close = indicators.get("recent_close", 0)

                    # 判断该周期方向
                    if direction == "long":
                        # 多头确认：价格>MA20, MA20>MA50, +DI > -DI
                        period_bull = (recent_close > ma20 and ma20 > ma50) or (plus_di > minus_di and adx > 20)
                        if period_bull:
                            agree_count += 1
                    else:
                        # 空头确认：价格<MA20, MA20<MA50, -DI > +DI
                        period_bear = (recent_close < ma20 and ma20 < ma50) or (minus_di > plus_di and adx > 20)
                        if period_bear:
                            agree_count += 1
                    total_checked += 1
                except Exception:
                    continue

            # 至少检查了2个周期，且至少2个一致
            passed = total_checked >= 2 and agree_count >= 2
            if not passed:
                logger.debug(
                    f"MTF check {symbol}: {direction} agree={agree_count}/{total_checked} "
                    f"(need >= 2/3), passed={passed}"
                )
            return passed
        except Exception as e:
            logger.debug(f"Multi-timeframe direction check error for {symbol}: {e}")
            return True  # 出错时不阻塞，默认放行

    async def _check_volume_confirmation(self, symbol: str, direction: str) -> bool:
        """成交量确认：当前volume > 20周期均量

        Returns:
            bool: 是否通过成交量确认
        """
        try:
            # 获取1H K线数据
            klines = await self.okx_client.get_kline_async(symbol, "1H", limit=25)
            if len(klines) < 20:
                return True  # 数据不足，默认放行

            volumes = [float(k[5]) for k in klines]

            recent_volume = volumes[-1]
            avg_volume_20 = sum(volumes[-20:]) / 20

            if avg_volume_20 <= 0:
                return True  # 无法计算，默认放行

            passed = recent_volume > avg_volume_20
            if not passed:
                logger.debug(
                    f"Volume check {symbol}: recent_vol={recent_volume:.2f} <= "
                    f"avg_20={avg_volume_20:.2f}"
                )
            return passed
        except Exception as e:
            logger.debug(f"Volume confirmation error for {symbol}: {e}")
            return True  # 出错时不阻塞

    async def _check_adx_confirmation(self, symbol: str) -> bool:
        """ADX 确认：多周期取最大 ADX > 20（任一周期确认趋势强度即通过）。

        原实现只检查第一个有缓存指标的周期就返回，若该周期（如 4h）恰好弱趋势、
        而更短周期（1h/15m）已走强，会误杀信号。改为取所有周期最大 ADX 判定。
        """
        try:
            # 优先使用缓存：多周期取最大 ADX，避免首个周期弱趋势时误判
            max_adx = 0.0
            has_cache = False
            for period in self._confirmation_periods:
                indicators = self._indicator_cache.get(symbol, {}).get(period)
                if indicators:
                    has_cache = True
                    adx = indicators.get("adx", 0)
                    if adx > max_adx:
                        max_adx = adx

            if has_cache:
                passed = max_adx > 20
                if not passed:
                    logger.debug(f"ADX check {symbol}: max ADX={max_adx:.1f} <= 20")
                return passed

            # 无缓存时实时计算
            klines = await self.okx_client.get_kline_async(symbol, "1H", limit=120)
            if len(klines) >= 50:
                indicators = self._calculate_all_indicators(klines, symbol)
                adx = indicators.get("adx", 0)
                passed = adx > 20
                if not passed:
                    logger.debug(f"ADX check {symbol}: ADX={adx:.1f} <= 20")
                return passed

            return True  # 数据不足，默认放行
        except Exception as e:
            logger.debug(f"ADX confirmation error for {symbol}: {e}")
            return True  # 出错时不阻塞

    async def _calculate_trend_confidence(self, symbol: str, direction: str, base_confidence: float,
                                     mtf_passed: bool = True, vol_passed: bool = True,
                                     adx_passed: bool = True) -> float:
        """综合趋势置信度计算

        组合以下维度：
        1. 多时间框架一致性 (0-0.4)
        2. 成交量确认 (0-0.2)
        3. ADX 强度 (0-0.2)
        4. 价格结构/HH-HL (0-0.2)

        Returns:
            float: 0-1 的置信度分数
        """
        try:
            score = 0.0

            # 1. 多时间框架一致性 (0-0.4)
            if mtf_passed:
                # 进一步细化：获取实际一致周期数
                mtf_detail = 0.2  # 基础分
                try:
                    mtf_periods = ["1H", "4H", "1D"]
                    agree_count = 0
                    total_checked = 0
                    for period in mtf_periods:
                        try:
                            klines = await self.okx_client.get_kline_async(symbol, period, limit=100)
                            if len(klines) >= 50:
                                indicators = self._calculate_all_indicators(klines, symbol)
                                if indicators:
                                    ma20 = indicators.get("ma20", 0)
                                    ma50 = indicators.get("ma50", 0)
                                    adx = indicators.get("adx", 0)
                                    plus_di = indicators.get("+di", 0)
                                    minus_di = indicators.get("-di", 0)
                                    recent_close = indicators.get("recent_close", 0)
                                    if direction == "long":
                                        if (recent_close > ma20 and ma20 > ma50) or (plus_di > minus_di and adx > 20):
                                            agree_count += 1
                                    else:
                                        if (recent_close < ma20 and ma20 < ma50) or (minus_di > plus_di and adx > 20):
                                            agree_count += 1
                                    total_checked += 1
                        except Exception:
                            continue
                    if total_checked >= 2:
                        agreement_ratio = agree_count / total_checked
                        mtf_detail = 0.2 + agreement_ratio * 0.2  # 0.2-0.4
                except Exception:
                    mtf_detail = 0.2
                score += mtf_detail
            else:
                score += 0.05  # 不通过也有一点基础分，不直接归零

            # 2. 成交量确认 (0-0.2)
            if vol_passed:
                try:
                    klines = await self.okx_client.get_kline_async(symbol, "1H", limit=25)
                    if len(klines) >= 20:
                        volumes = [float(k[5]) for k in klines]
                        recent_vol = volumes[-1]
                        avg_vol_20 = sum(volumes[-20:]) / 20
                        if avg_vol_20 > 0:
                            vol_ratio = recent_vol / avg_vol_20
                            vol_score = min(0.2, (vol_ratio - 1.0) * 0.2)  # 1x=0, 2x=0.2
                            score += max(0.0, vol_score)
                        else:
                            score += 0.1
                    else:
                        score += 0.1
                except Exception:
                    score += 0.1
            else:
                score += 0.02

            # 3. ADX 强度 (0-0.2)
            if adx_passed:
                try:
                    adx = 0
                    for period in self._confirmation_periods:
                        indicators = self._indicator_cache.get(symbol, {}).get(period)
                        if indicators:
                            adx = indicators.get("adx", 0)
                            break
                    if adx <= 0:
                        klines = await self.okx_client.get_kline_async(symbol, "1H", limit=100)
                        if len(klines) >= 50:
                            indicators = self._calculate_all_indicators(klines, symbol)
                            adx = indicators.get("adx", 0)
                    if adx > 50:
                        adx_score = 0.2
                    elif adx > 35:
                        adx_score = 0.15
                    elif adx > 25:
                        adx_score = 0.10
                    elif adx > 20:
                        adx_score = 0.05
                    else:
                        adx_score = 0.0
                    score += adx_score
                except Exception:
                    score += 0.05
            else:
                score += 0.0

            # 4. 价格结构 HH/HL (0-0.2)
            try:
                klines = await self.okx_client.get_kline_async(symbol, "1H", limit=30)
                if len(klines) >= 20:
                    klines = sorted(klines, key=lambda k: int(k[0]))
                    closes = np.array([float(k[4]) for k in klines])
                    highs = np.array([float(k[2]) for k in klines])
                    lows = np.array([float(k[3]) for k in klines])

                    structure = self._analyze_market_structure(highs, lows, closes)
                    if direction == "long":
                        if structure == "strong_uptrend":
                            score += 0.2
                        elif structure == "uptrend":
                            score += 0.15
                        elif structure == "ranging":
                            score += 0.05
                        else:
                            score += 0.0
                    else:
                        if structure == "strong_downtrend":
                            score += 0.2
                        elif structure == "downtrend":
                            score += 0.15
                        elif structure == "ranging":
                            score += 0.05
                        else:
                            score += 0.0
                else:
                    score += 0.05
            except Exception:
                score += 0.05

            # 融合检测器置信度（含多周期一致 + ADX 强度 + 突破/回调）与多因子软评分，
            # 避免单一维度主导（修复 Bug：此前 base_confidence 被完全丢弃）
            detector_w = max(0.0, min(1.0, self._detector_confidence_weight))
            blended = detector_w * base_confidence + (1.0 - detector_w) * score
            return min(1.0, max(0.0, blended))
        except Exception as e:
            logger.debug(f"Trend confidence calculation error for {symbol}: {e}")
            return max(0.25, base_confidence)  # 出错时保守返回