"""合约网格交易策略：在震荡区间内布设网格高抛低吸，并支持马丁格尔加仓与波动率自适应调整。"""
import asyncio
import math
import time
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger

from core.models import Signal, TickData, Position
from configs.settings import get_symbol_config
from utils.helpers import calculate_take_profit, calculate_stop_loss, validate_tp_sl_prices, calculate_tp_sl_from_atr, get_price_precision, map_market_state_to_regime
from utils.state_persistence import PersistentStrategy
from risk.dynamic_allocator import AdaptiveKelly

class GridStrategy(PersistentStrategy):
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        # P0-4: 初始化 PersistentStrategy 基类（设置 _state_persistence 等属性）
        super().__init__()
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache
        
        # 安全获取策略配置
        grid_cfg = config.get("strategies", {}).get("grid", {})
        
        self._enabled = grid_cfg.get("enabled", False)
        self._grid_count_min = grid_cfg.get("grid_count_min", 10)
        self._grid_count_max = grid_cfg.get("grid_count_max", 30)
        self._martingale_layers = grid_cfg.get("martingale_layers", 4)
        self._martingale_coefficient = grid_cfg.get("martingale_coefficient", 1.5)
        self._dynamic_adjust_interval = grid_cfg.get("dynamic_adjust_interval", 60)
        self._volatility_threshold = grid_cfg.get("volatility_threshold", 0.02)
        self._vol_adjust_cooldown = grid_cfg.get("vol_adjust_cooldown", 300)  # P23: 波动调整冷却时间(秒)，默认5分钟
        self._last_vol_adjust_ts: Dict[str, float] = {}  # P23: 每个币种上次波动调整时间戳
        self._atr_period = grid_cfg.get("atr_period", 14)
        self._atr_multiplier = grid_cfg.get("atr_multiplier", 0.5)
        self._volume_profile_period = grid_cfg.get("volume_profile_period", 60)
        
        self._grids: Dict[str, List[Dict[str, float]]] = {}
        self._martingale_state: Dict[str, int] = {}
        self._last_adjust_time: Dict[str, datetime] = {}
        self._last_tick_price: Dict[str, float] = {}
        self._trade_history: Dict[str, List[Dict[str, Any]]] = {}
        self._position_side: Dict[str, Optional[str]] = {}
        self._trend_mode: Dict[str, bool] = {}
        self._trend_mode_last_switch: Dict[str, float] = {}
        self._trend_mode_cooldown = grid_cfg.get("trend_mode_cooldown", 120.0)  # R6: 300s→120s，减少趋势模式切换冷却期的资金闲置
        self._atr_cache: Dict[str, float] = {}
        self._volume_profile_cache: Dict[str, Dict[str, float]] = {}
        self._trend_bias_cache: Dict[str, Dict[str, Any]] = {}
        self._dx_history: Dict[str, List[float]] = {}
        # P33: 退出后冷却追踪 {symbol: exit_timestamp}
        self._last_exit_time: Dict[str, float] = {}
        self._post_exit_cooldown = grid_cfg.get("post_exit_cooldown", 120)  # R7: 默认从 300s 降至 120s，加速资金再部署
        # 参数层：post_exit_cooldown 运行时校验
        try:
            self._post_exit_cooldown = float(self._post_exit_cooldown)
        except (TypeError, ValueError):
            self._post_exit_cooldown = 120.0
        if not math.isfinite(self._post_exit_cooldown) or self._post_exit_cooldown < 0:
            self._post_exit_cooldown = 120.0
        
        self._grid_pending_at: Dict[str, Dict[int, float]] = {}  # {symbol: {grid_index: pending_timestamp}}

        # ghost_close 专项 Phase 4: 成交回执驱动仓位写入开关（false=回退到旧的乐观 pending 标记）
        self._long_only = grid_cfg.get("long_only", False)
        self._use_fill_driven_position = grid_cfg.get("use_fill_driven_position", False)
        # ghost_close 专项 Phase 4: 观测指标
        self._fill_callback_hits = 0
        self._pending_timeout_cleaned = 0
        self._ghost_close_count = 0
        
        # P0: 网格回滚追踪 - 防止同一层反复触发
        self._grid_retry_count: Dict[str, Dict[int, int]] = {}  # {symbol: {grid_index: retry_count}}
        self._grid_retry_first_ts: Dict[str, Dict[int, float]] = {}  # R15: {symbol: {grid_index: first_retry_timestamp}}
        self._grid_rollback_cooldown: Dict[str, Dict[int, float]] = {}  # {symbol: {grid_index: cooldown_until_timestamp}}
        self._grid_stale_count: Dict[str, int] = {}  # P10: {symbol: 连续"too far"拒绝次数}

        # R1: 跨策略持仓冲突检查（GridStrategy 不继承 TrendStrategyBase，需自行支持注入）
        self._position_manager = None

        # REST API降级缓存：避免频繁调用REST接口
        self._tick_rest_cache: Dict[str, tuple] = {}  # {symbol: (timestamp, TickData)}
        self._tick_rest_interval = 1.0  # REST tick缓存1秒
        
        self._pending_orders: Dict[str, List[Dict[str, Any]]] = {}
        self._stop_loss_orders: Dict[str, Dict[str, Any]] = {}
        self._order_check_interval = 60

        # P34: 单方向连续止损熔断 — 同 symbol+side 在观察窗口内连续止损达阈值后，冷却该方向开仓
        # 目标：阻断「同方向分层加仓→趋势反向→多层同时止损」的密集爆损（如 XRP 三连止损 -6.2）
        self._stop_loss_history: Dict[str, List[float]] = {}  # {symbol:buy|sell: [止损unix时间戳]}
        self._sl_circuit_breaker_enabled = grid_cfg.get("sl_circuit_breaker_enabled", True)
        self._sl_circuit_breaker_window = grid_cfg.get("sl_circuit_breaker_window", 7200)      # 观察窗口2h
        self._sl_circuit_breaker_threshold = grid_cfg.get("sl_circuit_breaker_threshold", 2)   # 连续止损≥2次
        self._sl_circuit_breaker_cooldown = grid_cfg.get("sl_circuit_breaker_cooldown", 3600)  # 冷却60min

        # 信号去重和节流机制
        self._last_signal_time: Dict[str, float] = {}  # {symbol: last_signal_timestamp}
        self._signal_fingerprints: Dict[str, float] = {}  # {fingerprint: timestamp}

        # grid skip 日志防抖: 同一 symbol+side 30 秒内不重复打印
        self._last_grid_skip_log: Dict[str, float] = {}  # {symbol_side: timestamp}
        self._symbol_signal_timestamps: Dict[str, deque] = {}  # {symbol: deque of timestamps} 滑动窗口计数
        self._min_signal_interval = config["strategies"]["grid"].get("min_signal_interval", 5.0)  # 最小信号间隔5秒
        self._signal_cooldown = config["strategies"]["grid"].get("signal_cooldown", 5.0)  # R18: 同层信号冷却5秒
        self._max_signals_per_window = config["strategies"]["grid"].get("max_signals_per_window", 2)  # 每窗口最多2个信号
        self._signal_window_seconds = config["strategies"]["grid"].get("signal_window_seconds", 10.0)  # 滑动窗口10秒
        # P0-4: 信号质量阈值接入config（buy方向）；sell方向保持 +0.25 风险溢价
        self._min_signal_quality = config["strategies"]["grid"].get("min_signal_quality", 0.50)
        # P2: sell 门槛上限，防止高风时段 + 做空溢价叠加形成过高的绝对门槛
        self._sell_signal_quality_cap = config["strategies"]["grid"].get("sell_signal_quality_cap", 0.75)
        # P33: 趋势确认门禁模式 — strict（ADX<20 拒绝，旧行为）/ regime_aware（震荡市 ADX<20 放行）
        self._trend_confirmation_mode = config["strategies"]["grid"].get("trend_confirmation_mode", "regime_aware")

        # P0-3: 极端波动暂停截止时间 {symbol: unix_timestamp}，到期前禁止开新网格
        self._extreme_vol_until: Dict[str, float] = {}

        # P32: 时段风险控制标志
        self._high_risk_quality_margin = 0.0  # 高风险时段：提高信号质量门槛（优质信号仍可开仓）
        self._medium_risk_reduce = False      # 中风险时段：仓位减半
        self._last_risk_hour_log: Dict[str, float] = {}  # 防抖日志

        self._adaptive_controller = None  # 动态分配控制器（由scheduler注入）
        self._stop_loss_manager = None    # 统一止损管理器（由scheduler注入）
        self._regime_engine = None        # MarketRegimeEngine（由StrategyFactory注入，供统一 ADX/DI 趋势明细复用）

        # 企业级：Grid 自适应利用率引擎
        self._grid_adaptive_engine = None
        try:
            from core.grid_adaptive_utilization import get_grid_adaptive_engine
            self._grid_adaptive_engine = get_grid_adaptive_engine(config)
        except Exception as e:
            logger.debug(f"GridAdaptiveUtilizationEngine init skipped: {e}")

        self._dynamic_grid_count = config["strategies"]["grid"].get("dynamic_grid_count", True)
        self._volatility_adaptive_spacing = config["strategies"]["grid"].get("volatility_adaptive_spacing", True)
        self._multi_symbol_scheduling = config["strategies"]["grid"].get("multi_symbol_scheduling", True)
        self._min_grid_spacing = config["strategies"]["grid"].get("min_grid_spacing", 0.002)
        self._max_grid_spacing = config["strategies"]["grid"].get("max_grid_spacing", 0.05)

        # ── 参数层：min/max grid_spacing 运行时类型/范围/合法性校验 ──
        # Pydantic 已做 min <= max 校验，此处兜底 dict.get 返回的 None/非数字/NaN/Inf
        _default = (0.002, 0.05)
        try:
            self._min_grid_spacing = float(self._min_grid_spacing)
            self._max_grid_spacing = float(self._max_grid_spacing)
        except (TypeError, ValueError):
            self._min_grid_spacing, self._max_grid_spacing = _default
        if not (math.isfinite(self._min_grid_spacing) and math.isfinite(self._max_grid_spacing)):
            self._min_grid_spacing, self._max_grid_spacing = _default
        if self._min_grid_spacing <= 0:
            self._min_grid_spacing = _default[0]
        if self._max_grid_spacing <= 0:
            self._max_grid_spacing = _default[1]
        if self._min_grid_spacing > self._max_grid_spacing:
            self._min_grid_spacing, self._max_grid_spacing = _default
        # 每symbol实际网格间距（ATR动态计算），用于手续费检查和利润保障
        self._actual_grid_spacing: Dict[str, float] = {}
        # 硬性最大止损：无论波动多大，止损不超过此比例（避免ATR自适应时止损过远）
        self._max_stop_loss_pct = config["strategies"]["grid"].get("max_stop_loss_pct", 0.04)

        # 移动止损追踪 {symbol: {"highest_price": float, "lowest_price": float, "entry_price": float}}
        self._trailing_state: Dict[str, Dict[str, float]] = {}
        self._trailing_stop_enabled = config["strategies"]["grid"].get("trailing_stop_enabled", True)
        self._trailing_stop_pct = config["strategies"]["grid"].get("trailing_stop_pct", 0.018)
        self._breakeven_trigger_pct = config["strategies"]["grid"].get("breakeven_trigger_pct", 0.012)
        self._breakeven_stop_pct = config["strategies"]["grid"].get("breakeven_stop_pct", 0.003)

        # 时间止盈 {symbol: entry_timestamp}
        self._position_entry_time: Dict[str, float] = {}
        self._time_exit_enabled = config["strategies"]["grid"].get("time_exit_enabled", True)
        self._max_hold_hours = config["strategies"]["grid"].get("max_hold_hours", 72.0)
        self._time_exit_after_hours = config["strategies"]["grid"].get("time_exit_after_hours", 48.0)
        self._time_exit_partial_pct = config["strategies"]["grid"].get("time_exit_partial_pct", 0.5)

        # 波动率止盈 {symbol: last_volatility_check_timestamp}
        self._volatility_stop_enabled = config["strategies"]["grid"].get("volatility_stop_enabled", True)
        self._vol_spike_threshold = config["strategies"]["grid"].get("vol_spike_threshold", 2.5)
        self._vol_stop_partial_pct = config["strategies"]["grid"].get("vol_stop_partial_pct", 0.3)
        self._volatility_lockout_minutes = config["strategies"]["grid"].get("volatility_lockout_minutes", 30)

        # P13: 网格重建冷却 {symbol: last_rebuild_timestamp}
        self._grid_rebuild_cooldown: Dict[str, float] = {}
        self._grid_rebuild_cooldown_seconds = config["strategies"]["grid"].get("grid_rebuild_cooldown_seconds", 300)

        # 多级止盈配置
        self._tp1_ratio = config["strategies"]["grid"].get("tp1_ratio", 0.4)
        self._tp1_pct = config["strategies"]["grid"].get("tp1_pct", 0.008)
        self._tp2_ratio = config["strategies"]["grid"].get("tp2_ratio", 0.5)
        self._tp2_pct = config["strategies"]["grid"].get("tp2_pct", 0.015)
        self._tp3_ratio = config["strategies"]["grid"].get("tp3_ratio", 0.1)
        self._tp3_trailing_pct = config["strategies"]["grid"].get("tp3_trailing_pct", 0.02)
        self._take_profit_enabled = config["strategies"]["grid"].get("take_profit_enabled", True)

        # P36: 分钟级反转过滤器 — EMA20-EMA50/ADX(15m) 均为滞后指标，捕捉不到分钟级反弹。
        # grid 做空在长期下跌趋势中被短期反弹反复止损（XRP/SOL short 累计约 -3.9）。
        self._minute_rebound_window = config["strategies"]["grid"].get("minute_rebound_window", 15)  # 观察窗口（分钟）
        self._minute_rebound_threshold = config["strategies"]["grid"].get("minute_rebound_threshold", 0.005)  # 反转幅度阈值（默认0.5%）
        self._minute_rebound_cache: Dict[str, Dict[str, Any]] = {}  # {symbol: {ts, rise, drop}}
        self._minute_rebound_cache_ttl = 30.0  # 1m K线结果缓存30秒，避免每tick重复请求

        self._grid_performance: Dict[str, Dict[str, Any]] = {}
        self._adaptive_kelly = AdaptiveKelly(config.get("adaptive_kelly", {}))

        # P1: 日内亏损熔断 —— 当日累计亏损超过阈值则停摆开仓
        self._max_daily_loss_usdt = float(config["strategies"]["grid"].get("max_daily_loss_usdt", 0.0) or 0.0)
        self._daily_pnl = 0.0
        self._daily_pnl_date = None
        self._symbol_activity: Dict[str, Dict[str, Any]] = {}
        self._last_rebalance_time: Optional[datetime] = None
        self._rebalance_interval = 3600  # 1小时重新平衡一次

        # ===================== 生产级各币种网格状态管理 =====================
        # 逐币健康状态: {symbol: {status, last_check, error_count, warnings, grid_integrity_score}}
        self._coin_health: Dict[str, Dict[str, Any]] = {}
        # 逐币生命周期事件日志: {symbol: [{timestamp, event_type, details}]}
        self._coin_lifecycle: Dict[str, List[Dict[str, Any]]] = {}
        # 逐币暂停标记: {symbol: bool} - 暂停后该币种不触发新网格、不调整止损
        self._coin_paused: Dict[str, bool] = {}
        # 逐币上次状态快照: {symbol: snapshot_dict} - 用于diff变更检测
        self._coin_last_state_snapshot: Dict[str, Dict[str, Any]] = {}
        # 健康检查配置
        self._coin_health_check_interval = config["strategies"]["grid"].get("coin_health_check_interval", 120)
        self._coin_max_lifecycle_events = config["strategies"]["grid"].get("coin_max_lifecycle_events", 100)
        self._coin_health_max_errors = config["strategies"]["grid"].get("coin_health_max_errors", 5)
        self._coin_health_degraded_score = config["strategies"]["grid"].get("coin_health_degraded_score", 0.6)
        self._coin_auto_pause_on_error = config["strategies"]["grid"].get("coin_auto_pause_on_error", True)

        self._all_symbols = []
        for tier in ["tier1", "tier2", "tier3"]:
            for base in config["currencies"][f"{tier}_symbols"]:
                # 网格策略使用合约格式
                self._all_symbols.append(f"{base}-USDT-SWAP")

        # P0-4: 启用状态持久化（redis_cache 当前同步接口，StatePersistence 内部会自动降级到 JSON 文件）
        self.init_state_persistence("grid", redis_cache)

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

    def set_regime_engine(self, engine):
        """注入MarketRegimeEngine，供内部评分复用统一 ADX/DI 趋势明细（消除口径漂移）。"""
        self._regime_engine = engine

    def set_position_manager(self, manager):
        """注入 PositionManager，用于跨策略持仓冲突检查（R1）。"""
        self._position_manager = manager

    def _check_cross_strategy_conflict(self, symbol: str, side: str) -> bool:
        """检查跨策略持仓冲突：同 symbol 同方向是否已被其他策略持有。

        返回 True 表示存在冲突，应拒绝开仓。
        """
        if self._position_manager is None:
            return False
        try:
            allowed, reason = self._position_manager.would_create_cross_strategy_duplicate(
                symbol, side, "grid"
            )
            if not allowed:
                logger.debug(f"[grid] 跨策略持仓冲突: {symbol} {side} — {reason}")
                return True
        except Exception:
            pass
        return False

    def _get_unified_trend_detail(self, symbol: str) -> Optional[Dict[str, Any]]:
        """读取 MarketRegimeEngine 的统一 ADX/DI 趋势明细（get_symbol_trend_detail）。

        与框架层 TrendConfirmationFilter 同源，供 grid 内部评分复用，避免与策略内部
        自算 ADX 口径漂移。引擎未注入/币种未被监控/数据不足时返回 None（调用方回退）。
        """
        engine = getattr(self, "_regime_engine", None)
        if engine is None:
            return None
        try:
            return engine.get_symbol_trend_detail(symbol)
        except Exception:
            return None

    def set_trade_journal(self, trade_journal):
        """注入TradeJournal实例，用于网格对账时查询实际成交"""
        self._trade_journal = trade_journal

    async def update_config(self, updates: Dict[str, Any]):
        """运行时热更新策略配置（不重启策略）

        支持的配置键：grid_count_max, grid_count_min, min_grid_spacing,
        max_grid_spacing, atr_multiplier, stop_loss_pct, min_signal_quality,
        leverage, martingale_coefficient, martingale_layers 等
        """
        strategy_cfg = self.config.get("strategies", {}).get("grid", {})
        strategy_cfg.update(updates)
        self.config.setdefault("strategies", {})["grid"] = strategy_cfg

        # 更新实例属性
        attr_map = {
            "grid_count_max": "grid_count_max",
            "grid_count_min": "grid_count_min",
            "min_grid_spacing": "min_grid_spacing",
            "max_grid_spacing": "max_grid_spacing",
            "atr_multiplier": "atr_multiplier",
            "stop_loss_pct": "stop_loss_pct",
            "min_signal_quality": "min_signal_quality",
            "leverage": "leverage",
            "martingale_coefficient": "martingale_coefficient",
            "martingale_layers": "martingale_layers",
            "take_profit_enabled": "take_profit_enabled",
            "tp1_ratio": "tp1_ratio", "tp2_ratio": "tp2_ratio", "tp3_ratio": "tp3_ratio",
            "tp1_pct": "tp1_pct", "tp2_pct": "tp2_pct",
            "time_exit_enabled": "time_exit_enabled",
            "time_exit_after_hours": "time_exit_after_hours",
            "max_hold_hours": "max_hold_hours",
            "volatility_stop_enabled": "volatility_stop_enabled",
            "vol_spike_threshold": "vol_spike_threshold",
            "volatility_lockout_minutes": "volatility_lockout_minutes",
            "trailing_stop_enabled": "trailing_stop_enabled",
            "trailing_stop_pct": "trailing_stop_pct",
            "breakeven_trigger_pct": "breakeven_trigger_pct",
            "breakeven_stop_pct": "breakeven_stop_pct",
            "max_stop_loss_pct": "max_stop_loss_pct",
            "dynamic_grid_count": "dynamic_grid_count",
            "volatility_adaptive_spacing": "volatility_adaptive_spacing",
            "multi_symbol_scheduling": "multi_symbol_scheduling",
            # 生产级各币种网格状态管理
            "coin_health_check_interval": "coin_health_check_interval",
            "coin_max_lifecycle_events": "coin_max_lifecycle_events",
            "coin_health_max_errors": "coin_health_max_errors",
            "coin_health_degraded_score": "coin_health_degraded_score",
            "coin_auto_pause_on_error": "coin_auto_pause_on_error",
        }
        for cfg_key, attr_name in attr_map.items():
            if cfg_key in updates:
                setattr(self, attr_name, updates[cfg_key])
                logger.info(f"Grid config hot-updated: {attr_name}={updates[cfg_key]}")

    def _get_signal_relaxation(self) -> float:
        """获取资金利用率引擎的信号质量放松量（仅取正数放松部分）。

        正数=放松（降低门槛，让更多信号通过），作用于最终门槛；
        负数=收紧方向由上层 AdaptiveController 通过 min_signal_quality 路径处理，
        此处只取正数避免双重计数。
        """
        try:
            if self._adaptive_controller and hasattr(self._adaptive_controller, "get_signal_relaxation"):
                return max(0.0, float(self._adaptive_controller.get_signal_relaxation()))
        except Exception as e:
            logger.debug(f"[grid] get_signal_relaxation failed: {e}")
        return 0.0

    def _get_allocation(self) -> float:
        """获取当前资金分配比例：优先使用AdaptiveController动态分配，回退到config"""
        if self._adaptive_controller:
            try:
                return self._adaptive_controller.get_allocation("grid")
            except Exception as e:
                logger.debug(f"[grid] get_allocation failed, fallback to config: {e}")
        return self.config["trading"].get("grid_allocation", 0.30)

    def _get_effective_capital(self) -> float:
        """获取有效资金；账户权益无法确认时返回 0，禁止按静态配置扩大仓位。30s TTL 缓存。"""
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
            logger.warning(f"[grid] get_account_info failed, refusing new exposure: {e}")
        return 0.0

    def _get_existing_symbol_margin(self, symbol: str) -> Optional[float]:
        """查询该币种现有持仓的累计保证金（与 global_risk 单币种上限口径一致）。

        返回该币种所有方向持仓的 margin 之和；margin 为 0 时按 notionalUsd/lever 估算，
        避免跨/逐仓模式下 margin 字段为空导致累计上限失效。
        """
        try:
            positions = self.okx_client.get_positions()
            if positions is None:
                return None
            total_margin = 0.0
            for p in positions:
                if p.get("instId", "") != symbol:
                    continue
                margin = float(p.get("margin", 0) or 0)
                if margin <= 0:
                    notional = float(p.get("notionalUsd", 0) or 0)
                    lever = float(p.get("lever", 1) or 1)
                    if notional > 0 and lever > 0:
                        margin = notional / lever
                total_margin += margin
            return total_margin
        except Exception as e:
            logger.warning(f"Grid: failed to query existing margin for {symbol}; refusing new exposure: {e}")
            return None

    def apply_param_update(self, params: Dict[str, Any]):
        """热更新策略参数（由StrategyOptimizer调用，无需重启）
        params: 参数字典，键为参数名，值为新值
        支持的参数：grid_count_min, grid_count_max, martingale_layers,
                   martingale_coefficient, volatility_threshold, atr_period, atr_multiplier,
                   stop_loss_pct, trailing_stop_pct, breakeven_trigger_pct, breakeven_stop_pct,
                   tp1_pct, tp2_pct, max_hold_hours, time_exit_after_hours
        """
        applied = []
        param_map = {
            "grid_count_min": "_grid_count_min",
            "grid_count_max": "_grid_count_max",
            "martingale_layers": "_martingale_layers",
            "martingale_coefficient": "_martingale_coefficient",
            "volatility_threshold": "_volatility_threshold",
            "atr_period": "_atr_period",
            "atr_multiplier": "_atr_multiplier",
            "stop_loss_pct": "_stop_loss_pct",
            "trailing_stop_pct": "_trailing_stop_pct",
            "breakeven_trigger_pct": "_breakeven_trigger_pct",
            "breakeven_stop_pct": "_breakeven_stop_pct",
            "tp1_pct": "_tp1_pct",
            "tp2_pct": "_tp2_pct",
            "max_hold_hours": "_max_hold_hours",
            "time_exit_after_hours": "_time_exit_after_hours",
            "max_stop_loss_pct": "_max_stop_loss_pct",
            "long_only": "_long_only",
            "min_signal_quality": "_min_signal_quality",
            "trend_mode_cooldown": "_trend_mode_cooldown",
            "post_exit_cooldown": "_post_exit_cooldown",
            # 生产级各币种网格状态管理
            "coin_health_check_interval": "_coin_health_check_interval",
            "coin_health_max_errors": "_coin_health_max_errors",
            "coin_health_degraded_score": "_coin_health_degraded_score",
        }
        for cfg_key, attr_name in param_map.items():
            if cfg_key in params:
                setattr(self, attr_name, params[cfg_key])
                self.config["strategies"]["grid"][cfg_key] = params[cfg_key]
                applied.append(cfg_key)
        if applied:
            logger.info(f"Grid strategy params hot-updated: {applied}")
        return applied

    async def start(self):
        if not self._enabled:
            logger.info("Grid strategy is disabled")
            return

        logger.info("Starting Grid Strategy")
        # P0-4: 启动时尝试恢复持久化状态（失败不阻塞启动）
        try:
            await self.load_state_async()
        except Exception as e:
            logger.warning(f"Grid state load failed, starting fresh: {e}")

        await self._initialize_grids()

        asyncio.create_task(self._monitor_ticks())
        asyncio.create_task(self._dynamic_adjust_loop())
        asyncio.create_task(self._trend_monitor_loop())
        asyncio.create_task(self._update_indicators_loop())
        asyncio.create_task(self._order_check_loop())
        asyncio.create_task(self._multi_symbol_rebalance_loop())
        # P0-4: 启动周期性状态保存协程
        asyncio.create_task(self.periodic_save_loop())
        # 生产级：启动逐币健康检查协程
        asyncio.create_task(self._coin_health_check_loop())

    # P0-4: 状态持久化 - 收集/恢复
    def collect_persistent_state(self) -> Dict[str, Any]:
        """收集需要持久化的状态。"""
        return {
            "_grids": self._grids,
            "_pending_orders": self._pending_orders,
            "_stop_loss_orders": self._stop_loss_orders,
            "_position_state": self._position_side,
            "_martingale_state": self._martingale_state,
            "_trend_mode": self._trend_mode,
            "_extreme_vol_until": self._extreme_vol_until,
            "_grid_retry_count": self._grid_retry_count,
            "_grid_retry_first_ts": self._grid_retry_first_ts,
            "_grid_rollback_cooldown": self._grid_rollback_cooldown,
            "_trailing_state": self._trailing_state,
            "_position_entry_time": self._position_entry_time,
            # P34: 单方向止损熔断历史
            "_stop_loss_history": self._stop_loss_history,
            # 生产级各币种网格状态管理
            "_coin_health": self._coin_health,
            "_coin_paused": self._coin_paused,
            "_coin_lifecycle": self._coin_lifecycle,
            "_grid_performance": self._grid_performance,
        }

    @staticmethod
    def _restore_int_keyed_dict(raw):
        """将 JSON 序列化后 int 键变成字符串键的嵌套 dict 恢复为 int 键。

        _grid_retry_count / _grid_rollback_cooldown 使用 grid_index(int) 作内层键，
        json.dumps 会把 int 键转成字符串，恢复后若直接使用 int 键查询必然 miss，
        导致重试/冷却防护在重启后静默失效。
        """
        out = {}
        for symbol, inner in (raw or {}).items():
            if not isinstance(inner, dict):
                continue
            out[symbol] = {}
            for k, v in inner.items():
                try:
                    out[symbol][int(k)] = v
                except (TypeError, ValueError):
                    out[symbol][k] = v
        return out

    def restore_persistent_state(self, state: Dict[str, Any]):
        """从持久化数据恢复状态。"""
        self._grids = state.get("_grids", {}) or {}
        self._pending_orders = state.get("_pending_orders", {}) or {}
        self._stop_loss_orders = state.get("_stop_loss_orders", {}) or {}
        # 兼容字段：优先用持久化的 _position_state，否则回退到 _position_side
        self._position_side = state.get("_position_state") or state.get("_position_side") or {}
        self._martingale_state = state.get("_martingale_state", {}) or {}
        self._trend_mode = state.get("_trend_mode", {}) or {}
        self._extreme_vol_until = state.get("_extreme_vol_until", {}) or {}
        self._grid_retry_count = self._restore_int_keyed_dict(state.get("_grid_retry_count", {}))
        self._grid_retry_first_ts = self._restore_int_keyed_dict(state.get("_grid_retry_first_ts", {}))
        self._grid_rollback_cooldown = self._restore_int_keyed_dict(state.get("_grid_rollback_cooldown", {}))
        self._trailing_state = state.get("_trailing_state", {}) or {}
        self._position_entry_time = state.get("_position_entry_time", {}) or {}
        # P34: 单方向止损熔断历史（恢复时保留窗口内的止损时间戳）
        self._stop_loss_history = state.get("_stop_loss_history", {}) or {}
        # 生产级各币种网格状态管理
        self._coin_health = state.get("_coin_health", {}) or {}
        self._coin_paused = state.get("_coin_paused", {}) or {}
        self._coin_lifecycle = state.get("_coin_lifecycle", {}) or {}
        self._grid_performance = state.get("_grid_performance", {}) or {}
        logger.info(
            f"Grid state restored: {len(self._grids)} symbols, "
            f"{len(self._pending_orders)} pending, {len(self._stop_loss_orders)} SL orders, "
            f"{len(self._coin_paused)} paused coins"
        )

    def get_active_position_symbols(self) -> set:
        """返回 grid 当前认为持有持仓的 symbol 集合。

        供 OrderExecutor 重启对账使用：这些 symbol 由 grid 主动管理（即使方向或
        数据库记录被 P7-4 sync 污染），不应被误判为框架外 orphan 持仓。
        """
        try:
            return set(self._position_side.keys())
        except Exception:
            return set()

    async def _initialize_grids(self):
        for symbol in self._all_symbols:
            await self._build_grid(symbol)
            self._trend_mode[symbol] = False

    async def _update_indicators_loop(self):
        while True:
            for symbol in self._all_symbols:
                await self._update_atr(symbol)
                await self._update_volume_profile(symbol)
            await asyncio.sleep(300)

    async def _update_atr(self, symbol: str):
        klines = await self.okx_client.get_kline_async(symbol, "1H", limit=self._atr_period + 10)
        if len(klines) < self._atr_period + 1:
            return
        
        try:
            highs = np.array([float(kline[2]) for kline in klines])
            lows = np.array([float(kline[3]) for kline in klines])
            closes = np.array([float(kline[4]) for kline in klines])
            
            if np.any(np.isnan(highs)) or np.any(np.isinf(highs)) or \
               np.any(np.isnan(lows)) or np.any(np.isinf(lows)) or \
               np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
                return
            
            tr_values = []
            for i in range(1, len(highs)):
                tr = max(highs[i] - lows[i], 
                         abs(highs[i] - closes[i-1]), 
                         abs(lows[i] - closes[i-1]))
                tr_values.append(tr)
            
            if not tr_values:
                return
            
            atr = np.mean(tr_values[-self._atr_period:])
            
            if np.isnan(atr) or np.isinf(atr) or atr <= 0:
                return
            
            self._atr_cache[symbol] = atr

            # 顺便计算EMA20/EMA50用于趋势方向过滤
            try:
                ema20 = closes[-1]
                ema50 = closes[-1]
                if len(closes) >= 50:
                    alpha20 = 2.0 / (20 + 1)
                    alpha50 = 2.0 / (50 + 1)
                    for c in closes[-50:]:
                        ema50 = alpha50 * c + (1 - alpha50) * ema50
                    for c in closes[-20:]:
                        ema20 = alpha20 * c + (1 - alpha20) * ema20
                elif len(closes) >= 20:
                    alpha20 = 2.0 / (20 + 1)
                    for c in closes[-20:]:
                        ema20 = alpha20 * c + (1 - alpha20) * ema20
                    ema50 = float(np.mean(closes))
                # 趋势强度：|EMA20-EMA50|/EMA50
                if ema50 > 0:
                    trend_strength = abs(ema20 - ema50) / ema50
                    # 趋势方向: 1=上涨, -1=下跌, 0=震荡
                    if trend_strength < 0.005:
                        trend_dir = 0
                    elif ema20 > ema50:
                        trend_dir = 1
                    else:
                        trend_dir = -1
                    self._trend_bias_cache[symbol] = {
                        "direction": trend_dir,
                        "strength": trend_strength,
                        "ema20": ema20,
                        "ema50": ema50,
                    }
            except Exception:
                pass
        except (ValueError, IndexError):
            return

    async def _update_volume_profile(self, symbol: str):
        klines = await self.okx_client.get_kline_async(symbol, "1H", limit=self._volume_profile_period)
        if len(klines) < 20:
            return
        
        try:
            prices = np.array([float(kline[4]) for kline in klines])
            volumes = np.array([float(kline[5]) for kline in klines])
            
            if np.any(np.isnan(prices)) or np.any(np.isinf(prices)) or \
               np.any(np.isnan(volumes)) or np.any(np.isinf(volumes)) or \
               np.any(volumes < 0):
                return
            
            min_price = np.min(prices)
            max_price = np.max(prices)
            price_range = max_price - min_price
            
            if np.isnan(price_range) or np.isinf(price_range) or price_range == 0:
                return
            
            bins = 20
            bin_width = price_range / bins
            
            volume_by_bin = np.zeros(bins)
            price_by_bin = np.zeros(bins)
            
            for i in range(bins):
                lower = min_price + i * bin_width
                upper = min_price + (i + 1) * bin_width
                mask = (prices >= lower) & (prices < upper)
                if np.any(mask):
                    volume_by_bin[i] = np.sum(volumes[mask])
                    price_by_bin[i] = np.mean(prices[mask])
            
            max_volume_bin = np.argmax(volume_by_bin)
            
            total_volume = np.sum(volumes)
            if total_volume <= 0:
                return
            
            vwap_price = np.sum(prices * volumes) / total_volume
            
            if np.isnan(vwap_price) or np.isinf(vwap_price):
                vwap_price = np.mean(prices)
            
            poc_price = price_by_bin[max_volume_bin]
            if np.isnan(poc_price) or np.isinf(poc_price):
                poc_price = np.mean(prices)
            
            self._volume_profile_cache[symbol] = {
                "poc_price": poc_price,
                "vwap_price": vwap_price,
                "min_price": min_price,
                "max_price": max_price,
                "high_volume_zones": self._identify_high_volume_zones(bin_width, price_by_bin, volume_by_bin)
            }
        except (ValueError, IndexError):
            return

    def _identify_high_volume_zones(self, bin_width: float, price_by_bin: np.ndarray, volume_by_bin: np.ndarray) -> List[Dict[str, float]]:
        zones = []
        avg_volume = np.mean(volume_by_bin)
        threshold = avg_volume * 1.5
        
        in_zone = False
        zone_start = 0
        
        for i, volume in enumerate(volume_by_bin):
            if volume >= threshold and not in_zone:
                in_zone = True
                zone_start = price_by_bin[i] - bin_width / 2
            elif volume < threshold and in_zone:
                in_zone = False
                zones.append({
                    "start": zone_start,
                    "end": price_by_bin[i] + bin_width / 2
                })
        
        if in_zone:
            zones.append({
                "start": zone_start,
                "end": price_by_bin[-1] + bin_width / 2
            })
        
        return zones

    @staticmethod
    def _safe_avg_spacing(grids: list, current_price: float, fallback: float = 0.01) -> float:
        """安全计算网格平均间距（%），除零/NaN/Inf 防护，返回非负有限值。

        风险：grid 价格可能含 NaN/Inf、current_price 可能为 0 或非有限、
        len(grids) 可能为 1 导致除零，任一情况都返回 fallback 避免污染下游。
        """
        if current_price <= 0 or not math.isfinite(current_price):
            return fallback
        n = len(grids)
        if n < 2:
            return fallback
        try:
            p0 = float(grids[0].get("price", 0))
            pn = float(grids[-1].get("price", 0))
        except (TypeError, ValueError):
            return fallback
        if not (math.isfinite(p0) and math.isfinite(pn)):
            return fallback
        raw = abs(pn - p0) / current_price / (n - 1)
        if not math.isfinite(raw) or raw < 0:
            return fallback
        return raw

    async def _build_grid(self, symbol: str, current_price: float = None):
        # P14: 优先使用传入的current_price，避免缓存过期导致重建到错误价格
        if current_price is not None:
            price = current_price
        else:
            ticker = await self.okx_client.get_ticker_async(symbol)
            if not ticker:
                return
            price = float(ticker["last"])
        tier_settings = get_symbol_config(symbol, self.config)
        
        atr = self._atr_cache.get(symbol, 0)
        if atr > 0 and price > 0:
            atr_spacing = atr * self._atr_multiplier / price
            if self._volatility_adaptive_spacing:
                base_spacing = max(self._min_grid_spacing, min(self._max_grid_spacing, atr_spacing))
            else:
                base_spacing = max(tier_settings["grid_spacing_min"], min(tier_settings["grid_spacing_max"], atr_spacing))
        else:
            base_spacing = (tier_settings["grid_spacing_min"] + tier_settings["grid_spacing_max"]) // 2
        # ── 运算层：base_spacing NaN/Inf 防护（price=0 或 atr 异常时除法可能产生 NaN/Inf） ──
        if not math.isfinite(base_spacing) or base_spacing <= 0:
            base_spacing = max(self._min_grid_spacing, tier_settings.get("grid_spacing_min", 0.005))
        
        # P32: 强制最小网格间距，防止ATR过小导致间距为0
        # min_profitable_spacing = taker_fee * 2 * 4 = 0.004 (0.4%)
        # 最小间距至少为 min_profitable * 2 = 0.008 (0.8%)，确保每格有充足利润空间
        taker_fee = float(self.config.get("trading", {}).get("taker_fee_rate", 0.0005) or 0.0005)
        min_profitable = taker_fee * 2 * 4
        min_grid_spacing = max(
            tier_settings.get("grid_spacing_min", 0.005),
            min_profitable * 2  # 0.8%
        )
        if base_spacing < min_grid_spacing:
            logger.debug(
                f"P16: Grid {symbol} base_spacing {base_spacing:.4%} below minimum {min_grid_spacing:.4%}, "
                f"enforcing minimum"
            )
            base_spacing = min_grid_spacing
        
        # 保存实际网格间距，供手续费检查使用
        self._actual_grid_spacing[symbol] = base_spacing
        
        if self._dynamic_grid_count and atr > 0 and price > 0:
            volatility_ratio = atr / price
            if volatility_ratio > 0.02:
                grid_count = self._grid_count_min
            elif volatility_ratio > 0.01:
                grid_count = (self._grid_count_min + self._grid_count_max) // 2
            else:
                grid_count = self._grid_count_max
        else:
            grid_count = (self._grid_count_min + self._grid_count_max) // 2
        
        grid_count = max(3, min(grid_count, 50))
        half_grids = grid_count // 2
        
        # P32: 方向感知网格 - 根据EMA趋势非对称分配buy/sell层
        trend_direction, trend_strength = await self._detect_grid_trend(symbol)
        if trend_direction == "up" and trend_strength >= 0.15:
            # 上升趋势：buy层更多，sell层更少 (60/40)
            # 趋势越强，buy层越多；上限0.6防止极端非对称
            buy_ratio = min(0.6, 0.5 + trend_strength * 0.2)
            split_point = int(grid_count * buy_ratio)
            logger.info(
                f"P32: Grid {symbol} asymmetric UP - buy_ratio={buy_ratio:.0%} "
                f"split={split_point}/{grid_count} strength={trend_strength:.2f}"
            )
        elif trend_direction == "down" and trend_strength >= 0.15:
            # 下降趋势：sell层更多，buy层更少 (40/60)
            # 趋势越强，sell层越多；上限0.6防止极端非对称
            sell_ratio = min(0.6, 0.5 + trend_strength * 0.2)
            split_point = int(grid_count * (1 - sell_ratio))  # buy层更少
            logger.info(
                f"P32: Grid {symbol} asymmetric DOWN - sell_ratio={sell_ratio:.0%} "
                f"split={split_point}/{grid_count} strength={trend_strength:.2f}"
            )
        else:
            # 震荡/弱趋势：保持对称网格
            split_point = half_grids
            logger.debug(f"P32: Grid {symbol} symmetric (trend={trend_direction} strength={trend_strength:.2f})")
        
        # P0-1: long_only模式 - 所有层均为buy，禁止做空
        if self._long_only:
            split_point = grid_count
            logger.info(f"P0-1: Grid {symbol} long_only mode - all {grid_count} layers set to buy")
        
        # 边界保护：long_only模式允许split_point=grid_count，否则至少留1层sell
        if self._long_only:
            split_point = max(1, min(split_point, grid_count))
        else:
            split_point = max(1, min(split_point, grid_count - 1))
        
        vp = self._volume_profile_cache.get(symbol)
        use_vp_bounds = False
        if vp and vp.get("min_price") is not None and vp.get("max_price") is not None:
            vp_min = float(vp["min_price"])
            vp_max = float(vp["max_price"])
            if vp_min > 0 and vp_max > vp_min:
                # P15: 验证VP范围与当前价格是否兼容
                # 如果VP中心偏离当前价格超过20%，VP缓存已过期，忽略VP使用价格边界
                vp_center = (vp_min + vp_max) / 2
                vp_deviation = abs(vp_center - price) / price if price > 0 else 0
                if vp_deviation > 0.20:
                    logger.info(
                        f"P15: VP cache stale for {symbol} "
                        f"(VP center={vp_center:.4f}, current={price:.4f}, deviation={vp_deviation:.1%}), "
                        f"using price-based bounds"
                    )
                    # 清除过期VP缓存
                    del self._volume_profile_cache[symbol]
                else:
                    use_vp_bounds = True
                    upper_bound = vp_max * 1.01
                    lower_bound = vp_min * 0.99
        
        if not use_vp_bounds:
            upper_bound = price * (1 + base_spacing * half_grids)
            lower_bound = price * (1 - base_spacing * half_grids)
        
        lower_bound = max(lower_bound, price * 0.5)
        upper_bound = min(upper_bound, price * 1.5)
        
        if lower_bound >= upper_bound:
            lower_bound = price * (1 - base_spacing * half_grids)
            upper_bound = price * (1 + base_spacing * half_grids)
        
        prices = np.linspace(lower_bound, upper_bound, grid_count)
        
        self._grids[symbol] = []
        for i, price in enumerate(prices):
            side = "buy" if i < split_point else "sell"  # P32: 方向感知非对称分配
            
            density_factor = self._calculate_density_factor(symbol, price, vp)
            adjusted_spacing = base_spacing * density_factor
            
            self._grids[symbol].append({
                "price": self.okx_client.round_price_to_tick(symbol, price),
                "side": side,
                "filled": False,
                "layer": abs(i - split_point),  # P32: 使用split_point而非half_grids
                "quantity": 0,
                "density_factor": density_factor,
                "adjusted_spacing": adjusted_spacing
            })
        
        self._martingale_state[symbol] = 0
        self._last_adjust_time[symbol] = datetime.now()
        self._trend_mode[symbol] = False
        
        logger.info(f"Grid built for {symbol}: {grid_count} levels, range [{lower_bound:.4f}, {upper_bound:.4f}], ATR spacing: {base_spacing:.4f}")

        # 根据当前价格与网格中点关系设置初始持仓方向
        mid_price = (lower_bound + upper_bound) / 2
        # P0-1: long_only 模式下初始持仓方向强制为 buy（只做多，禁止做空）
        self._position_side[symbol] = "buy" if (self._long_only or price < mid_price) else "sell"

        # 生产级：记录生命周期事件
        self._log_coin_lifecycle_event(symbol, "grid_built", {
            "grid_count": grid_count,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "base_spacing": round(base_spacing, 6),
            "current_price": round(price, 4),
            "atr": round(atr, 6) if atr > 0 else None,
            "position_side": self._position_side[symbol],
        })

    async def _rebuild_stale_grid(self, symbol: str, current_price: float = None):
        """P10: 重建失效网格 - 当网格价格持续偏离当前价格超过阈值时触发
        
        清理旧网格状态并基于当前价格重新构建网格。
        P14: 接受current_price参数，避免使用过期缓存ticker导致重建到错误价格。
        """
        try:
            logger.info(f"P10: Rebuilding stale grid for {symbol}")
            
            # 清理旧网格状态
            if symbol in self._grids:
                del self._grids[symbol]
            if symbol in self._grid_pending_at:
                del self._grid_pending_at[symbol]
            if symbol in self._grid_retry_count:
                del self._grid_retry_count[symbol]
            if symbol in self._grid_retry_first_ts:
                del self._grid_retry_first_ts[symbol]
            if symbol in self._grid_rollback_cooldown:
                del self._grid_rollback_cooldown[symbol]
            if symbol in self._grid_stale_count:
                del self._grid_stale_count[symbol]
            if symbol in self._martingale_state:
                del self._martingale_state[symbol]
            if symbol in self._trend_mode:
                del self._trend_mode[symbol]
            
            # P15: 清除过期Volume Profile缓存，防止VP缓存覆盖current_price
            # VP缓存可能包含旧价格范围，导致网格重建到错误价格
            if symbol in self._volume_profile_cache:
                old_vp = self._volume_profile_cache[symbol]
                logger.info(
                    f"P15: Clearing stale VP cache for {symbol} "
                    f"(VP range [{old_vp.get('min_price', '?')}, {old_vp.get('max_price', '?')}], "
                    f"current_price={current_price})"
                )
                del self._volume_profile_cache[symbol]
            
            # 重建网格 - P14: 传入当前价格避免缓存过期
            self._grid_rebuild_cooldown[symbol] = time.time()  # P13: 记录重建时间
            await self._build_grid(symbol, current_price=current_price)
            logger.info(f"P13: Grid rebuilt for {symbol}")
        except Exception as e:
            logger.error(f"P10: Failed to rebuild stale grid for {symbol}: {e}")

    def _calculate_density_factor(self, symbol: str, price: float, vp: Optional[Dict]) -> float:
        if not vp or not vp.get("high_volume_zones"):
            return 1.0
        
        for zone in vp["high_volume_zones"]:
            if zone["start"] <= price <= zone["end"]:
                return 0.75
        
        return 1.25

    async def _detect_grid_trend(self, symbol: str) -> tuple:
        """P32: 检测EMA趋势方向，用于非对称网格分配
        
        基于1H K线的EMA(20)斜率判断短期趋势方向。
        返回 (direction, strength)：
        - direction: "up" / "down" / "sideways"
        - strength: 0.0-1.0 趋势强度
        
        趋势强度 = EMA斜率 / ATR，归一化到0-1
        """
        if not self._validate_symbol(symbol):
            return "sideways", 0.0

        try:
            klines = await self.okx_client.get_kline_async(symbol, "1H", limit=30)
            if not klines or len(klines) < 20:
                return "sideways", 0.0
            
            closes = [float(k[4]) for k in klines[-25:]]  # 最近25根K线收盘价
            
            # 计算 EMA(20)
            ema_period = 20
            multiplier = 2 / (ema_period + 1)
            ema = closes[0]
            ema_values = [ema]
            for price in closes[1:]:
                ema = (price - ema) * multiplier + ema
                ema_values.append(ema)
            
            # 计算EMA斜率（最近5根的变化率）
            if len(ema_values) >= 5:
                slope = (ema_values[-1] - ema_values[-5]) / ema_values[-5]
            else:
                slope = 0.0
            
            # 使用ATR归一化趋势强度
            atr = self._atr_cache.get(symbol, 0)
            current_price = closes[-1]
            if atr > 0 and current_price > 0:
                atr_pct = atr / current_price
                normalized_strength = min(1.0, abs(slope) / (atr_pct * 5)) if atr_pct > 0 else 0
            else:
                normalized_strength = min(1.0, abs(slope) * 100)
            
            # 判断方向
            if normalized_strength < 0.15:
                direction = "sideways"
                normalized_strength = 0.0
            elif slope > 0:
                direction = "up"
            else:
                direction = "down"
            
            logger.debug(
                f"P32: Grid trend {symbol}: direction={direction} "
                f"strength={normalized_strength:.2f} slope={slope:.6f} "
                f"ema[-1]={ema_values[-1]:.4f} ema[-5]={ema_values[-5]:.4f}"
            )
            self._record_metric(
                "grid_trend_direction",
                1.0 if direction == "up" else -1.0 if direction == "down" else 0.0,
                {"symbol": symbol},
            )
            self._record_metric("grid_trend_strength", normalized_strength, {"symbol": symbol, "direction": direction})
            return direction, normalized_strength
            
        except Exception as e:
            self._handle_exception(
                e,
                context={"symbol": symbol},
                module="GridStrategy",
                function="_detect_grid_trend",
                severity="low",
                category="business",
            )
            return "sideways", 0.0

    async def _monitor_ticks(self):
        # 频率控制：WS可用时0.1s，REST降级时1s
        self._last_active_time = time.time()  # 策略活跃时间戳，用于心跳检测
        heartbeat_interval = 30  # 每30秒更新一次活跃信号
        _adaptive_eval_interval = 300  # 每300s评估一次grid自适应利用率（补上evaluate调度）
        _last_adaptive_eval = 0.0
        
        while True:
            ws_used = False

            async def _tick_one(sym):
                # R6: 趋势模式下不再完全停摆，改为顺势网格（_process_tick 内部做方向过滤）
                return await self._process_tick(sym)

            results = await asyncio.gather(
                *[_tick_one(sym) for sym in self._all_symbols],
                return_exceptions=True,
            )
            ws_used = any(r is True for r in results)
            
            # 定期更新活跃时间戳，避免心跳超时误报
            now = time.time()
            if now - self._last_active_time >= heartbeat_interval:
                self._last_active_time = now
                # 更新全局心跳：使用特殊key "_heartbeat" 让调度器感知策略存活
                self._last_signal_time["_heartbeat"] = now
            
            # 周期评估 grid 自适应利用率（修复：此前 evaluate 从未被调用，multiplier 恒为 1.0）
            if self._grid_adaptive_engine and now - _last_adaptive_eval >= _adaptive_eval_interval:
                _last_adaptive_eval = now
                try:
                    self._grid_adaptive_engine.evaluate()
                except Exception as e:
                    logger.debug(f"Grid adaptive evaluate error: {e}")

            # WS数据正常则高频轮询，REST降级则低频
            await asyncio.sleep(0.1 if ws_used else 1.0)

    async def _process_tick(self, symbol: str) -> bool:
        """处理tick数据，返回True表示使用了WS数据，False表示REST降级"""
        # 生产级：逐币暂停检查
        if self._coin_paused.get(symbol, False):
            return False

        # P0-3: 极端波动暂停期检查（到期前禁止触发新网格订单）
        until = self._extreme_vol_until.get(symbol)
        if until:
            if time.time() < until:
                return False
            # 暂停期已过，清理并恢复
            del self._extreme_vol_until[symbol]
            logger.info(f"Extreme volatility pause expired for {symbol}, resuming grid trading")

        # P32: 时段风险分级 - 基于UTC小时的风险分级控制
        # 数据分析：UTC 00:00 57笔交易 0%胜率 → 高风险时段历史胜率极低
        # 修复：高风险时段不再完全禁止开仓，改为大幅提高信号质量门槛（优质信号仍可开仓）
        utc_hour = datetime.utcnow().hour
        HIGH_RISK_HOURS = {0, 4, 8, 11, 13, 17, 20, 22}  # 高风险：提高信号质量门槛
        MEDIUM_RISK_HOURS = {1, 2, 15, 16, 21}            # 中风险：降仓50%
        
        if utc_hour in HIGH_RISK_HOURS:
            # R4: 高风险时段门槛从 0.25 降至 0.10，避免叠加后阈值过高过滤掉大量信号
            self._high_risk_quality_margin = 0.10
            self._medium_risk_reduce = False
        elif utc_hour in MEDIUM_RISK_HOURS:
            self._high_risk_quality_margin = 0.0
            self._medium_risk_reduce = True
        else:
            self._high_risk_quality_margin = 0.0
            self._medium_risk_reduce = False

        tick = self.redis_cache.get_tick(symbol)
        from_ws = True

        # REST API降级：当redis_cache无tick数据时，使用OKX REST API
        if not tick:
            tick = await self._get_tick_rest(symbol)
            from_ws = False
            if not tick:
                return False
        
        price = tick.price
        last_price = self._last_tick_price.get(symbol, 0)
        
        if last_price == 0:
            self._last_tick_price[symbol] = price
            return
        
        grids = self._grids.get(symbol, [])
        if not grids:
            return

        # P1: 日内亏损熔断 —— 当日亏损超过阈值则停摆，不再开新网格
        if not self._check_daily_loss_limit():
            return

        # P0: 最大同时pending层数限制 —— 防止快速行情下多个网格同时触发
        pending_count = len(self._grid_pending_at.get(symbol, {}))
        # R14: 从 //6 放宽到 //3，释放主策略闲置资金（原限制导致 70-85% 网格层闲置）
        max_pending_grids = max(1, len(grids) // 3)
        if pending_count >= max_pending_grids:
            return  # 等待pending确认/回滚后再触发新层
        
        price_change = abs(price - last_price) / last_price if last_price > 0 else 0
        
        for i, grid in enumerate(grids):
            if grid["filled"]:
                continue
            
            # P0: 回滚冷却检查 —— 被回滚的网格在冷却期内禁止再次触发
            cooldown_until = self._grid_rollback_cooldown.get(symbol, {}).get(i)
            if cooldown_until and time.time() < cooldown_until:
                continue
            
            # R15: 最大重试次数 3→6，超过1小时自动重置计数
            retry_count = self._grid_retry_count.get(symbol, {}).get(i, 0)
            first_ts = self._grid_retry_first_ts.get(symbol, {}).get(i, 0)
            if first_ts > 0 and time.time() - first_ts > 3600:
                self._grid_retry_count.get(symbol, {}).pop(i, None)
                self._grid_retry_first_ts.get(symbol, {}).pop(i, None)
                retry_count = 0
            max_retries = 6
            if retry_count >= max_retries:
                continue
            
            grid_price = grid["price"]
            side = grid["side"]
            
            # P32: 高风险时段不再完全禁止开仓，而是提高信号质量门槛（下方阈值叠加 margin）
            if side == "buy" and last_price > grid_price >= price:
                _entry_penalty = await self._confirm_grid_entry(symbol, "buy", price, tick)
                if _entry_penalty >= 0:
                    # 信号质量评分检查
                    try:
                        quality = await self._calculate_grid_signal_quality(symbol, "buy", price, grid)
                        # R7: 扣分制 — 入场确认的过滤器惩罚从质量分中扣除
                        effective_quality = quality - _entry_penalty
                        # P0-4: buy方向阈值接入config min_signal_quality（替代硬编码0.50）
                        # P32: 高风险时段叠加质量门槛
                        _buy_threshold = self._min_signal_quality + self._high_risk_quality_margin
                        # R4: 逆势惩罚从 0.05 降至 0.03，减少信号过滤
                        _bias = self._trend_bias_cache.get(symbol)
                        if _bias and _bias.get("direction") == -1 and _bias.get("strength", 0) > 0.015:
                            _buy_threshold += 0.03
                        if effective_quality < _buy_threshold:
                            logger.debug(f"Grid {symbol} buy signal rejected: quality={quality:.2f} penalty={_entry_penalty:.2f} effective={effective_quality:.2f} < {_buy_threshold:.2f}")
                            continue
                        # 信号验证
                        is_valid, reason = self._validate_grid_signal(symbol, "buy", price, grid)
                        if not is_valid:
                            logger.debug(f"Grid {symbol} buy signal invalid: {reason}")
                            continue
                    except Exception as e:
                        logger.debug(f"Grid {symbol} quality/validation check error: {e}, rejecting")
                        continue
                    await self._trigger_grid_order(symbol, "buy", grid_price, grid["layer"], grid_index=i)
                    grid["filled"] = "pending"
                    if symbol not in self._grid_pending_at:
                        self._grid_pending_at[symbol] = {}
                    self._grid_pending_at[symbol][i] = time.time()
                    break  # P0: 每tick只触发一个网格，避免同时触发多层
            elif (not self._long_only) and side == "sell" and last_price < grid_price <= price:
                # P0-1: long_only 模式下禁止 sell 信号（历史 sell 层也不会触发空单）
                _entry_penalty = await self._confirm_grid_entry(symbol, "sell", price, tick)
                if _entry_penalty >= 0:
                    # 信号质量评分检查
                    try:
                        quality = await self._calculate_grid_signal_quality(symbol, "sell", price, grid)
                        # R7: 扣分制 — 入场确认的过滤器惩罚从质量分中扣除
                        effective_quality = quality - _entry_penalty
                        # R4: 做空溢价从 0.25/0.10 降至 0.10/0.05，减少信号过滤
                        _short_premium = 0.10
                        _bias = self._trend_bias_cache.get(symbol)
                        if _bias and _bias.get("direction") == -1 and _bias.get("strength", 0) > 0.015:
                            _short_premium = 0.05
                        # P32: 高风险时段叠加质量门槛
                        # P2: 叠加信号质量放松量（资本层意图传导到信号门槛）+ 门槛上限
                        _sell_threshold = self._min_signal_quality + _short_premium + self._high_risk_quality_margin
                        _sell_threshold -= self._get_signal_relaxation()
                        _sell_threshold = max(self._min_signal_quality, min(_sell_threshold, self._sell_signal_quality_cap))
                        if effective_quality < _sell_threshold:
                            logger.debug(f"Grid {symbol} sell signal rejected: quality={quality:.2f} penalty={_entry_penalty:.2f} effective={effective_quality:.2f} < {_sell_threshold:.2f}")
                            continue
                        # 信号验证
                        is_valid, reason = self._validate_grid_signal(symbol, "sell", price, grid)
                        if not is_valid:
                            logger.debug(f"Grid {symbol} sell signal invalid: {reason}")
                            continue
                    except Exception as e:
                        logger.debug(f"Grid {symbol} quality/validation check error: {e}, rejecting")
                        continue
                    await self._trigger_grid_order(symbol, "sell", grid_price, grid["layer"], grid_index=i)
                    grid["filled"] = "pending"
                    if symbol not in self._grid_pending_at:
                        self._grid_pending_at[symbol] = {}
                    self._grid_pending_at[symbol][i] = time.time()
                    break  # P0: 每tick只触发一个网格，避免同时触发多层
        
        self._last_tick_price[symbol] = price
        return from_ws

    async def _get_tick_rest(self, symbol: str) -> Optional[TickData]:
        """REST API降级：从OKX REST接口获取ticker数据，构造TickData"""
        now = time.time()
        cache_entry = self._tick_rest_cache.get(symbol)
        if cache_entry and (now - cache_entry[0]) < self._tick_rest_interval:
            return cache_entry[1]
        
        try:
            # P3修复：使用异步版本避免阻塞事件循环
            ticker = await self.okx_client.get_ticker_async(symbol)
            if not ticker:
                return None
            
            tick = TickData(
                symbol=symbol,
                price=float(ticker.get("last", 0)),
                volume=float(ticker.get("vol24h", 0)),
                bid_price=float(ticker.get("bidPx", 0)),
                bid_volume=float(ticker.get("bidSz", 0)),
                ask_price=float(ticker.get("askPx", 0)),
                ask_volume=float(ticker.get("askSz", 0)),
                timestamp=datetime.now()
            )
            
            if tick.price <= 0:
                return None
            
            self._tick_rest_cache[symbol] = (now, tick)
            return tick
        except Exception as e:
            logger.debug(f"REST tick fallback failed for {symbol}: {e}")
            return None

    def _record_side_stop_loss(self, symbol: str, position_direction: str):
        """P34: 记录一次单方向止损事件，用于熔断判定。

        position_direction 为持仓方向（long/short），归一化到开仓方向 buy/sell。
        仅保留观察窗口内的止损时间戳，避免内存无限增长。
        """
        if not self._sl_circuit_breaker_enabled:
            return
        side = "buy" if position_direction == "long" else "sell"
        key = f"{symbol}:{side}"
        now = time.time()
        if key not in self._stop_loss_history:
            self._stop_loss_history[key] = []
        self._stop_loss_history[key].append(now)
        # 清理观察窗口外的旧止损记录
        cutoff = now - self._sl_circuit_breaker_window
        self._stop_loss_history[key] = [t for t in self._stop_loss_history[key] if t >= cutoff]

    def _is_side_circuit_broken(self, symbol: str, side: str) -> bool:
        """P34: 判断 symbol 的 side 方向是否处于熔断冷却期。

        观察窗口内连续止损次数 >= 阈值，且最近一次止损距今 < 冷却时长时，返回 True。
        """
        if not self._sl_circuit_breaker_enabled:
            return False
        key = f"{symbol}:{side}"
        history = self._stop_loss_history.get(key, [])
        if not history:
            return False
        now = time.time()
        cutoff = now - self._sl_circuit_breaker_window
        recent = [t for t in history if t >= cutoff]
        if len(recent) < self._sl_circuit_breaker_threshold:
            return False
        last_sl = max(recent)
        remaining = self._sl_circuit_breaker_cooldown - (now - last_sl)
        if remaining > 0:
            return True
        # 冷却已过期，清空记录重新开始计数
        self._stop_loss_history[key] = []
        return False

    async def _get_minute_rebound_metrics(self, symbol: str) -> Optional[Dict[str, float]]:
        """P36: 计算分钟级反转指标（带30秒缓存）。

        用最近 1m K 线度量短期急拉(rise)/急杀(drop)，区分方向：
        - rise = (最后一根收盘 - 窗口最低) / 窗口最低  → 做空危险（反弹）
        - drop = (窗口最高 - 最后一根收盘) / 窗口最高  → 做多危险（急杀）
        返回 None 表示数据不足/异常（调用方放行，不阻塞主流程）。
        """
        try:
            now = time.time()
            cached = self._minute_rebound_cache.get(symbol)
            if cached and now - cached.get("ts", 0) < self._minute_rebound_cache_ttl:
                return cached.get("metrics")

            window = max(3, int(self._minute_rebound_window))
            klines = await self.okx_client.get_kline_async(symbol, "1m", limit=window + 2)
            metrics: Optional[Dict[str, float]] = None
            if len(klines) >= 3:
                highs, lows, closes = [], [], []
                for k in klines:
                    try:
                        h, l, c = float(k[2]), float(k[3]), float(k[4])
                    except (ValueError, IndexError, TypeError):
                        continue
                    if not (math.isfinite(h) and math.isfinite(l) and math.isfinite(c)):
                        continue
                    if l <= 0 or h <= 0:
                        continue
                    highs.append(h); lows.append(l); closes.append(c)
                if len(closes) >= 3:
                    lo = min(lows)
                    hi = max(highs)
                    last_close = closes[-1]
                    rise = (last_close - lo) / lo if lo > 0 else 0.0
                    drop = (hi - last_close) / hi if hi > 0 else 0.0
                    metrics = {"rise": max(0.0, rise), "drop": max(0.0, drop)}
            self._minute_rebound_cache[symbol] = {"ts": now, "metrics": metrics}
            return metrics
        except Exception as e:
            logger.debug(f"P36: minute rebound metrics error for {symbol}: {e}")
            return None

    async def _check_minute_rebound(self, symbol: str, side: str, price: float) -> bool:
        """P36: 分钟级反转过滤器。返回 True 表示存在危险反转，应拦截开仓。

        - 做空(sell)：短期急拉（反弹）幅度超阈值 → 禁开空（对称于既有做多逆势惩罚）。
        - 做多(buy)：短期急杀幅度超阈值 → 禁开多。
        仅拦截开仓（提前减仓由上层 _check_reversal_take_profit 处理）。
        """
        if self._minute_rebound_threshold <= 0:
            return False
        metrics = await self._get_minute_rebound_metrics(symbol)
        if not metrics:
            return False
        threshold = self._minute_rebound_threshold
        if side == "sell":
            if metrics.get("rise", 0.0) >= threshold:
                logger.debug(
                    f"P36: Grid {symbol} sell blocked — minute rebound rise={metrics['rise']:.2%} "
                    f">= {threshold:.2%} (window={self._minute_rebound_window}m)"
                )
                return True
        else:
            if metrics.get("drop", 0.0) >= threshold:
                logger.debug(
                    f"P36: Grid {symbol} buy blocked — minute drop={metrics['drop']:.2%} "
                    f">= {threshold:.2%} (window={self._minute_rebound_window}m)"
                )
                return True
        return False

    async def _confirm_grid_entry(self, symbol: str, side: str, price: float, tick) -> float:
        """R7: 扣分制入场确认 — 返回质量惩罚值（0.0=无惩罚，>0=扣分）。
        -1.0 表示硬性否决（熔断），调用方应直接拒绝。
        原一票否决过滤器改为扣分，只有最终质量分低于阈值才拒绝。"""
        # P34: 单方向连续止损熔断 — 硬性否决（安全底线，不改为扣分）
        if self._is_side_circuit_broken(symbol, side):
            now = time.time()
            key = f"{symbol}_{side}"
            if now - self._last_grid_skip_log.get(key, 0) > 30:
                logger.warning(
                    f"Grid circuit-breaker: {symbol} {side} 开仓被熔断拦截 "
                    f"(观察窗口 {self._sl_circuit_breaker_window}s 内连续止损 >= "
                    f"{self._sl_circuit_breaker_threshold} 次)"
                )
                self._last_grid_skip_log[key] = now
            return -1.0

        penalty = 0.0

        # R8: 趋势方向过滤 — 从一票否决改为扣分（强趋势逆势扣更多）
        bias = self._trend_bias_cache.get(symbol)
        if bias:
            strength = bias.get("strength", 0)
            trend_dir = bias.get("direction", 0)
            if strength > 0.05:
                is_counter = (trend_dir == 1 and side == "sell") or (trend_dir == -1 and side == "buy")
                if is_counter:
                    # 强趋势逆势：扣分与强度成正比，strength=0.05→0.08, strength=0.2→0.15
                    penalty += min(0.15, 0.08 + strength * 0.35)

        # P36: 分钟级反转 — 从一票否决改为扣分
        if await self._check_minute_rebound(symbol, side, price):
            penalty += 0.08

        # 成交量确认 — 从一票否决改为扣分
        if tick.volume and tick.volume > 0:
            vol_history = self._trade_history.get(symbol, [])
            if len(vol_history) >= 5:
                recent_vols = [t.get("volume", 0) for t in vol_history[-5:]]
                avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 0
                if avg_vol > 0 and tick.volume < avg_vol * 0.1:
                    penalty += 0.05

        # 买卖盘压力确认 — 从一票否决改为扣分
        if side == "buy":
            bid_vol = tick.bid_volume or 0
            ask_vol = tick.ask_volume or 0
            if ask_vol > 0 and bid_vol / ask_vol < 0.2:
                penalty += 0.05
        else:
            bid_vol = tick.bid_volume or 0
            ask_vol = tick.ask_volume or 0
            if bid_vol > 0 and ask_vol / bid_vol < 0.2:
                penalty += 0.05

        return penalty

    async def _check_trend_ready_for_entry(self, symbol: str, side: str, price: float) -> bool:
        """P33: 趋势确认门禁，按 trend_confirmation_mode 分流。

        - strict：ADX < 20 直接拒绝（旧行为，震荡市不做网格）
        - regime_aware：震荡市（ADX < 20）放行，由信号质量门槛 + 趋势方向过滤兜底；
          趋势市（ADX ≥ 20）才做 +DI / -DI 方向对齐（顺势）+ 价格位置过滤（避免追高/追低）
        """
        try:
            klines = await self.okx_client.get_kline_async(symbol, "15m", limit=50)
            if len(klines) < 20:
                return True  # 数据不足时放行，避免阻塞

            highs = np.array([float(k[2]) for k in klines])
            lows = np.array([float(k[3]) for k in klines])
            closes = np.array([float(k[4]) for k in klines])

            # 数据层：NaN/Inf 防护
            if np.any(np.isnan(highs)) or np.any(np.isinf(highs)) or \
               np.any(np.isnan(lows)) or np.any(np.isinf(lows)) or \
               np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
                return True

            adx, plus_di, minus_di = self._calculate_adx(highs, lows, closes, symbol)

            # 运算层：NaN/Inf 防护
            if not (math.isfinite(adx) and math.isfinite(plus_di) and math.isfinite(minus_di)):
                return True

            adx_floor = 20.0

            # P33 趋势确认门禁模式
            if self._trend_confirmation_mode == "strict":
                # strict：ADX < 20 直接拒绝（旧行为，震荡市不做网格）
                if adx < adx_floor:
                    logger.debug(f"P33: Grid {symbol} {side} rejected — ADX={adx:.1f} < {adx_floor} (strict)")
                    return False
            else:
                # regime_aware：震荡市（ADX < 20）是 grid 均值回归主场，放行；
                # 趋势市（ADX >= 20）才做方向对齐 + 价格位置过滤，避免逆势追涨杀跌。
                if adx < adx_floor:
                    return True

            if side == "buy":
                # 趋势市做多条件：+DI > -DI（上涨趋势顺势）
                if plus_di <= minus_di:
                    logger.debug(f"P33: Grid {symbol} buy rejected — +DI={plus_di:.1f} <= -DI={minus_di:.1f}")
                    return False

                # 高位等位做多：价格不应在近期高点（避免追高）
                recent_high = float(np.max(highs[-20:]))
                recent_low = float(np.min(lows[-20:]))
                if recent_high > recent_low:
                    price_position = (price - recent_low) / (recent_high - recent_low)
                    if price_position > 0.75:
                        logger.debug(
                            f"P33: Grid {symbol} buy rejected — price at {price_position:.1%} "
                            f"of recent range (too high, wait for pullback)"
                        )
                        return False
            else:
                # 趋势市做空条件：-DI > +DI（下跌趋势顺势）
                if minus_di <= plus_di:
                    logger.debug(f"P33: Grid {symbol} sell rejected — -DI={minus_di:.1f} <= +DI={plus_di:.1f}")
                    return False

                # 高位等位开空：价格不应在近期低点（避免追低）
                recent_high = float(np.max(highs[-20:]))
                recent_low = float(np.min(lows[-20:]))
                if recent_high > recent_low:
                    price_position = (price - recent_low) / (recent_high - recent_low)
                    if price_position < 0.25:
                        logger.debug(
                            f"P33: Grid {symbol} sell rejected — price at {price_position:.1%} "
                            f"of recent range (too low, wait for rally)"
                        )
                        return False

            return True
        except Exception as e:
            logger.debug(f"P33: Grid trend check error for {symbol}: {e}, allowing entry")
            return True  # 异常时放行，避免阻塞

    def _generate_clordid(self, symbol: str, side: str, signal_type: str = "grid") -> str:
        """生成合规 clOrdId（纯字母+数字，≤32位），用于成交回执关联（ghost_close 专项 Phase 4）。"""
        import re
        ts_ms = int(time.time() * 1000)
        short_symbol = re.sub(r'[^a-zA-Z0-9]', '', symbol.replace("-USDT", ""))[:6]
        short_type = re.sub(r'[^a-zA-Z0-9]', '', str(signal_type or ""))[:6]
        dir_flag = "L" if side in ("long", "buy") else "S"
        clordid = f"grd{short_symbol}{dir_flag}{ts_ms}{short_type}"
        if len(clordid) > 32:
            clordid = clordid[:32]
        return clordid

    async def _trigger_grid_order(self, symbol: str, side: str, price: float, layer: int, grid_index: int = None):
        # 企业级：参数前置校验，非法输入直接拒绝并埋点
        if not self._validate_symbol(symbol) or not self._validate_direction(side) or not self._validate_price(price):
            logger.warning(
                f"Grid order rejected: invalid params "
                f"symbol={symbol!r} side={side!r} price={price!r}"
            )
            self._increment_metric("grid_signal_rejected_total", 1.0, {"reason": "invalid_params", "symbol": symbol})
            return

        # R1: 跨策略持仓冲突检查 — 同 symbol 同方向已被其他策略持有时拒绝开仓
        if self._check_cross_strategy_conflict(symbol, side):
            logger.info(f"R1: Grid {symbol} {side} rejected - cross-strategy position conflict")
            self._increment_metric("grid_signal_rejected_total", 1.0, {"reason": "cross_strategy_conflict", "symbol": symbol})
            return

        # P32: 网格最小间距保护 - 前置检查，避免无效信号消耗资源
        # 数据分析：62.1%交易在-0.01~0 USDT区间，手续费蚕食利润
        # 强制间距 >= 双边手续费 * 4（0.4%），确保每格至少有利润空间
        taker_fee = float(self.config.get("trading", {}).get("taker_fee_rate", 0.0005) or 0.0005)
        actual_spacing = self._actual_grid_spacing.get(symbol, 0)
        min_profitable_spacing = taker_fee * 2 * 4  # 双边手续费 * 4 = 0.4%
        if actual_spacing <= 0 or actual_spacing <= min_profitable_spacing:
            logger.info(
                f"P32: Grid {symbol} {side} rejected - spacing {actual_spacing:.4%} <= "
                f"min_profitable {min_profitable_spacing:.4%} (fee*2*4), "
                f"insufficient profit per grid to cover fees"
            )
            return

        # P1: 磨损型交易过滤器 - 历史平均净利为负的币种停止开新网格（避免手续费蚕食）
        # 仅拦截开仓，不影响平仓；样本不足 5 笔不做判断，防止过早停用
        perf = self._grid_performance.get(symbol, {})
        wear_trades = perf.get("total_trades", 0)
        wear_pnl = perf.get("total_pnl", 0)
        if wear_trades >= 5 and wear_pnl < 0:
            avg_net_profit = wear_pnl / wear_trades
            logger.info(
                f"P1: Grid {symbol} {side} rejected - avg net profit {avg_net_profit:.4f} USDT < 0 "
                f"({wear_trades} trades, {wear_pnl:.4f} USDT), wear-type loss protection"
            )
            return

        # 信号去重和节流检查
        now = time.time()

        # 同币种最小信号间隔检查
        last_signal = self._last_signal_time.get(symbol, 0)
        if now - last_signal < self._min_signal_interval:
            logger.debug(f"Grid {symbol}: signal throttled (interval={now-last_signal:.2f}s < {self._min_signal_interval}s)")
            return

        # P0: 滑动窗口频率限制 - 防止快速tick场景下多个网格层被连续触发
        if symbol not in self._symbol_signal_timestamps:
            self._symbol_signal_timestamps[symbol] = deque()
        window_q = self._symbol_signal_timestamps[symbol]
        # 清理窗口外的旧时间戳
        cutoff = now - self._signal_window_seconds
        while window_q and window_q[0] < cutoff:
            window_q.popleft()
        if len(window_q) >= self._max_signals_per_window:
            logger.debug(
                f"Grid {symbol}: window throttle ({len(window_q)} signals in "
                f"{self._signal_window_seconds}s >= {self._max_signals_per_window}), skipping"
            )
            return

        # 同层同方向信号去重（基于指纹）
        fingerprint = f"{symbol}:{side}:{layer}:{round(price, 4)}"
        last_fingerprint_ts = self._signal_fingerprints.get(fingerprint, 0)
        if now - last_fingerprint_ts < self._signal_cooldown:
            logger.debug(f"Grid {symbol}: signal deduplicated (fingerprint={fingerprint}, cooldown={now-last_fingerprint_ts:.2f}s)")
            return

        # 更新时间戳
        self._last_signal_time[symbol] = now
        self._signal_fingerprints[fingerprint] = now
        window_q.append(now)  # 记录到滑动窗口

        # R6: 趋势模式下不再完全阻断信号，由 _trigger_grid_order 做顺势过滤

        # P0-3: 极端波动暂停期检查
        until = self._extreme_vol_until.get(symbol)
        if until and time.time() < until:
            logger.debug(f"Grid {symbol}: in extreme volatility pause, skip grid entry")
            return

        # P28: 最终价格验证 - 在下单前重新验证网格价格与当前市价的距离
        # 防止网格重建冷却期间、or validation pass但实际价格已偏离的情况下放置无效订单
        try:
            ticker = await self.okx_client.get_ticker_async(symbol)
            if ticker:
                current_price = float(ticker.get("last", 0))
                if current_price > 0 and price > 0:
                    distance = abs(price - current_price) / current_price
                    # P28: 网格价格偏离超过10%直接拒绝并触发重建
                    MAX_ORDER_PRICE_DEVIATION = 0.10
                    if distance > MAX_ORDER_PRICE_DEVIATION:
                        logger.warning(
                            f"P28: Grid order rejected for {symbol}: grid_price={price:.4f} "
                            f"vs market={current_price:.4f} (distance={distance:.4%} > {MAX_ORDER_PRICE_DEVIATION:.0%}), "
                            f"triggering grid rebuild"
                        )
                        # 触发异步重建
                        asyncio.create_task(self._rebuild_stale_grid(symbol, current_price=current_price))
                        return
                    # P28: 偏离5-10%之间记录警告但仍允许（可能是快速波动中的正常情况）
                    elif distance > 0.05:
                        logger.warning(
                            f"P28: Grid order large deviation for {symbol}: grid_price={price:.4f} "
                            f"vs market={current_price:.4f} (distance={distance:.4%}), proceeding with caution"
                        )
        except Exception as e:
            logger.debug(f"P28: Grid price re-validation failed for {symbol}: {e}, proceeding")

        # P33: 退出后冷却检查 — 防止止损/手动平仓后立即追入
        _last_exit = self._last_exit_time.get(symbol, 0)
        # 数据层：NaN/Inf 防护
        if not math.isfinite(_last_exit):
            _last_exit = 0
        # R12: 按波动率差异化冷却期 — 高波动长冷却（避免反复止损），低波动短冷却（加速再部署）
        _effective_cooldown = self._post_exit_cooldown
        atr = self._atr_cache.get(symbol, 0)
        if atr > 0 and price > 0:
            atr_ratio = atr / price
            if atr_ratio > 0.03:
                _effective_cooldown = self._post_exit_cooldown * 2.0
            elif atr_ratio < 0.01:
                _effective_cooldown = self._post_exit_cooldown * 0.5
        _cooldown_remaining = _effective_cooldown - (time.time() - _last_exit)
        if _cooldown_remaining > 0:
            logger.debug(
                f"P33: Grid {symbol} {side} rejected — post-exit cooldown "
                f"{_cooldown_remaining:.0f}s remaining (base={self._post_exit_cooldown:.0f}s effective={_effective_cooldown:.0f}s)"
            )
            return

        # P33: 趋势确认门禁 — ADX > 20 且 DI 方向对齐才允许开仓
        if not await self._check_trend_ready_for_entry(symbol, side, price):
            return

        tier_settings = get_symbol_config(symbol, self.config)
        leverage = tier_settings["leverage_default"]

        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        position_limit = tier_settings["position_limit"]

        # position_limit 是单币种仓位上限（封顶），与 allocation 是独立维度，
        # 不应相乘导致双重折价；取 min 保留封顶语义即可。
        base_position = trading_capital * min(allocation, position_limit)

        # P32: 中风险时段仓位减半
        if self._medium_risk_reduce:
            base_position *= 0.5
            logger.debug(
                f"P32: Grid {symbol} position reduced 50% due to medium risk hour "
                f"(UTC {datetime.utcnow().hour:02d}:00)"
            )

        # 注入空闲资金放大乘数（来自 AdaptiveController）
        if self._adaptive_controller:
            try:
                boost = self._adaptive_controller.get_position_boost()
                if boost > 1.0:
                    base_position *= boost
            except Exception:
                pass

        # P0: 企业级凯利仓位调整（基于逐币种网格绩效）
        base_position = self._calculate_kelly_position(base_position, symbol)

        # 企业级：Grid 自适应利用率乘数（正收益币种逐步提升资金利用率）
        if self._grid_adaptive_engine:
            try:
                multiplier = self._grid_adaptive_engine.get_multiplier(symbol)
                if multiplier != 1.0:
                    base_position *= multiplier
                    logger.debug(f"Grid {symbol}: adaptive multiplier {multiplier:.2f}x, margin={base_position:.2f}")
            except Exception as e:
                logger.debug(f"Grid adaptive multiplier error: {e}")

        instrument_info = await self.okx_client.get_instrument_info_async(symbol)
        if not instrument_info:
            logger.debug(f"Grid {symbol}: instrument info unavailable, skip")
            return
        min_lot_size = self._safe_float(instrument_info.get("lotSz"), 0.0)
        if min_lot_size <= 0 or leverage <= 0 or price <= 0:
            logger.debug(
                f"Grid {symbol}: invalid lot/leverage/price "
                f"(lotSz={min_lot_size}, leverage={leverage}, price={price}), skip"
            )
            return
        margin_needed_for_min_lot = price * min_lot_size / leverage

        strategy_cap = trading_capital * allocation

        # R16: 跨策略资金协调 — 扣除本策略已在其他币种占用的保证金
        if self._position_manager is not None:
            try:
                used_margin = self._position_manager.get_strategy_used_margin("grid")
                remaining_cap = max(0.0, strategy_cap - used_margin)
                if remaining_cap < strategy_cap * 0.1:
                    logger.debug(f"Grid {symbol}: cross-strategy coord — used {used_margin:.2f}/{strategy_cap:.2f}, skip")
                    return
                if base_position > remaining_cap:
                    logger.info(f"Grid {symbol}: cross-strategy coord — clip margin {base_position:.2f} -> {remaining_cap:.2f} (used {used_margin:.2f})")
                    base_position = remaining_cap
            except Exception as e:
                logger.debug(f"Grid {symbol}: cross-strategy margin check failed: {e}")
        if base_position < margin_needed_for_min_lot:
            adjusted = min(margin_needed_for_min_lot, strategy_cap)
            if adjusted >= margin_needed_for_min_lot:
                logger.info(f"Grid {symbol}: adjusted margin {base_position:.2f} -> {adjusted:.2f} for min lot (strategy_cap={strategy_cap:.2f})")
                base_position = adjusted
            else:
                logger.debug(f"Grid {symbol}: margin {base_position:.2f} < min lot margin {margin_needed_for_min_lot:.2f}, skip (strategy_cap {strategy_cap:.2f} too small)")
                return

        # P33: 单币种累计仓位上限 - 防止多层网格叠加突破单币种风控阈值
        # （阈值与 risk/global_risk.py 的 0.25 保持一致，从源头封顶，不增加风险）
        single_symbol_ratio = 0.25
        cumulative_margin_cap = total_capital * single_symbol_ratio
        existing_margin = self._get_existing_symbol_margin(symbol)
        if existing_margin is None:
            logger.warning(f"Grid {symbol}: current position margin unavailable, skip new layer")
            return
        available_margin = max(0.0, cumulative_margin_cap - existing_margin)
        if base_position > available_margin:
            logger.info(
                f"Grid {symbol}: cumulative cap - existing margin {existing_margin:.2f} + "
                f"new {base_position:.2f} > cap {cumulative_margin_cap:.2f}, "
                f"clip new layer to {available_margin:.2f}"
            )
            base_position = available_margin
            if base_position < margin_needed_for_min_lot:
                logger.debug(f"Grid {symbol}: available margin {base_position:.2f} < min lot margin {margin_needed_for_min_lot:.2f}, skip")
                return

        quantity = base_position * leverage / price

        # 数量校验：取整后必须 >= 最小合约面额
        qty_rounded = round(quantity / min_lot_size) * min_lot_size
        if qty_rounded < min_lot_size:
            # R13: 计算量不足一手时，尝试用最小手数开仓（利用零散资金而非闲置）
            qty_rounded = min_lot_size
            min_lot_margin = min_lot_size * price / leverage
            if min_lot_margin > base_position * 1.2:
                logger.debug(f"Grid {symbol}: min lot margin {min_lot_margin:.4f} > budget {base_position:.4f}*1.2, skip")
                return
            logger.info(f"Grid {symbol}: qty {quantity:.6f} < min_lot, using min lot {min_lot_size} (R13 fallback)")
        quantity = qty_rounded

        # P0: 名义价值上限保护 - 防止小资金时 AdaptiveController boost 导致仓位过大
        nominal_value = quantity * price
        # P15: 小账户自适应上限系数 - 资金越少，单仓位占比可越高以提升利用率
        if total_capital < 100:
            cap_factor = 0.95  # 小账户：允许单仓占权益66.5%
        elif total_capital < 500:
            cap_factor = 0.85
        else:
            cap_factor = 0.80
        max_nominal = total_capital * self.config["risk"].get("max_symbol_position_ratio", 1.0) * cap_factor
        min_notional = self.config["trading"].get("min_notional_usd", 1.0)
        if nominal_value > max_nominal:
            capped_qty = round((max_nominal * 0.9) / (price * min_lot_size)) * min_lot_size
            capped_nominal = capped_qty * price
            if capped_qty >= min_lot_size and capped_nominal >= min_notional:
                logger.info(f"Grid {symbol}: capped qty {quantity:.4f}->{capped_qty:.4f} "
                          f"(nominal {nominal_value:.2f}>{max_nominal:.2f})")
                quantity = capped_qty
            elif capped_qty >= min_lot_size:
                # 裁剪后不满足最小名义价值，适当放大到 min_notional
                min_qty = round((min_notional * 1.1) / (price * min_lot_size)) * min_lot_size
                if min_qty * price <= max_nominal * 1.2:
                    logger.info(f"Grid {symbol}: boosted to min_notional {quantity:.4f}->{min_qty:.4f}")
                    quantity = min_qty
                else:
                    logger.debug(f"Grid {symbol}: capped qty {capped_qty:.4f} below min_notional {min_notional}, skip")
                    return
            else:
                logger.debug(f"Grid {symbol}: capped qty {capped_qty:.4f} < min_lot {min_lot_size}, skip")
                return

        # R6: 趋势模式下只允许顺势网格（顺势方向开仓），逆势方向跳过
        if self._trend_mode.get(symbol, False):
            bias = self._trend_bias_cache.get(symbol)
            trend_dir = bias.get("direction", 0) if bias else 0
            if trend_dir == 1 and side == "sell":
                logger.debug(f"Grid {symbol}: trend-up mode, skip counter-trend sell")
                return
            elif trend_dir == -1 and side == "buy":
                logger.debug(f"Grid {symbol}: trend-down mode, skip counter-trend buy")
                return
            # 顺势方向：放行（无 bias 信息时也放行，由后续质量门槛兜底）

        # 移除马丁格尔：小账户使用马丁格尔必然爆仓
        # 固定仓位：每层同样大小，不放大

        # P0-2: 资金费率检查 - 距结算 <5min 或 费率吃掉 >50% 预期利润时跳过开仓
        # 注：okx_client.get_funding_rate 为同步 REST 接口，try/except 包裹保证失败时默认放行
        try:
            funding_data = await self.okx_client.get_funding_rate_async(symbol)
            if funding_data:
                rate_str = funding_data.get("fundingRate")
                next_funding_str = funding_data.get("nextFundingTime")
                if rate_str:
                    funding_rate = float(rate_str)
                    position_value = quantity * price
                    expected_funding_cost = abs(funding_rate) * position_value
                    # 资金费率吃掉一半预期利润（base_position 视为预期利润上限）则跳过
                    if base_position > 0 and expected_funding_cost > base_position * 0.5:
                        logger.info(
                            f"Grid skip {symbol}: funding cost {expected_funding_cost:.4f} > 50% of "
                            f"expected profit {base_position:.4f} (rate={funding_rate:.6f})"
                        )
                        return
                    # 距离 nextFundingTime < 5 分钟则跳过开仓（避免结算前开仓）
                    if next_funding_str:
                        try:
                            next_funding_ts = int(next_funding_str) / 1000.0  # OKX 返回毫秒
                            seconds_to_funding = next_funding_ts - time.time()
                            if 0 < seconds_to_funding < 300:
                                logger.info(
                                    f"Grid skip {symbol}: funding in {seconds_to_funding:.0f}s < 5min, skip entry"
                                )
                                return
                        except (ValueError, TypeError):
                            pass
        except Exception as e:
            logger.debug(f"Funding rate check failed for {symbol}: {e}, proceeding with grid entry")

        # P32: 间距检查已前置到函数头部，此处不再重复检查

        take_profit_levels = self._calculate_multiple_take_profit(symbol, side, price, layer)

        # 计算止损：使用配置的止损百分比
        grid_cfg = self.config.get("strategies", {}).get("grid", {})
        stop_loss_pct = grid_cfg.get("stop_loss_pct", 0.025)
        # 硬性最大止损上限：止损百分比不超过max_stop_loss_pct
        max_stop_loss_pct = grid_cfg.get("max_stop_loss_pct", 0.04)
        stop_loss_pct = min(stop_loss_pct, max_stop_loss_pct)
        slippage = tier_settings.get("slippage", 0.001)
        precision = get_price_precision(symbol)

        if side == "buy":
            stop_loss = price * (1 - stop_loss_pct) - price * slippage
            # 硬性下限：止损不得低于 price * (1 - max_stop_loss_pct)
            hard_floor = price * (1 - max_stop_loss_pct)
            if stop_loss < hard_floor:
                stop_loss = hard_floor
        else:
            stop_loss = price * (1 + stop_loss_pct) + price * slippage
            # 硬性上限：止损不得高于 price * (1 + max_stop_loss_pct)
            hard_cap = price * (1 + max_stop_loss_pct)
            if stop_loss > hard_cap:
                stop_loss = hard_cap
        stop_loss = round(stop_loss, precision)

        signal = Signal(
            symbol=symbol,
            strategy_name="grid",
            signal_type="grid_trade",
            direction=side,
            price=price,
            quantity=quantity,
            leverage=leverage,
            confidence=0.7 + layer * 0.05,
            timestamp=datetime.now()
        )

        # ghost_close 专项 Phase 4: 生成 clOrdId 用于成交回执关联
        clordid = self._generate_clordid(symbol, side, "grid") if self._use_fill_driven_position else ""

        signal_data = {
            "type": "signal",
            "data": {
                "symbol": signal.symbol,
                "strategy_name": signal.strategy_name,
                "signal_type": signal.signal_type,
                "direction": signal.direction,
                "price": signal.price,
                "quantity": signal.quantity,
                "leverage": signal.leverage,
                "clOrdId": clordid,
                "stop_loss": stop_loss,
                "take_profit": take_profit_levels[0] if take_profit_levels else None,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat(),
                "take_profit_levels": take_profit_levels
            }
        }

        self.redis_cache.publish_signal(signal_data)

        # ghost_close 专项 Phase 4: 将 clOrdId 记录到对应网格层，供成交回执精确匹配
        if clordid and grid_index is not None:
            grids = self._grids.get(symbol, [])
            if 0 <= grid_index < len(grids):
                grids[grid_index]["clOrdId"] = clordid

        self._record_metric("grid_signal_generated_total", 1.0, {"symbol": symbol, "direction": side, "layer": str(layer)})
        self._record_metric("grid_signal_confidence", signal.confidence, {"symbol": symbol, "direction": side})

        self._record_trade(symbol, side, price, quantity, layer)
        self._position_side[symbol] = side

        # 记录入场时间和初始化移动止损追踪
        self._position_entry_time[symbol] = time.time()
        self._trailing_state[symbol] = {
            "entry_price": price,
            "highest_price": price if side == "buy" else None,
            "lowest_price": price if side == "sell" else None,
            "trailing_activated": False,
            "trailing_sl": stop_loss,
        }

        logger.info(f"Grid signal: {side} {symbol} @ {price:.4f}, qty: {quantity:.4f}, layer: {layer}, SL: {stop_loss:.4f}, TP levels: {len(take_profit_levels)}")
    
    def _calculate_multiple_take_profit(self, symbol: str, side: str, price: float, layer: int) -> List[float]:
        tier = self._get_tier_from_symbol(symbol)
        
        atr = self._atr_cache.get(symbol, 0)
        if atr > 0:
            grid_spacing = atr * self._atr_multiplier / price
        else:
            grid_spacing = self.config["currencies"].get(f"{tier}_settings", {}).get("grid_spacing_min", 0.005)
        
        take_profit_levels = []
        
        # 使用配置的多级止盈比例（tp1_pct, tp2_pct），回退到基于网格间距的计算
        # 配置值为百分比（如0.8=0.8%），需除以100转为小数
        raw_tp1 = (self._tp1_pct / 100.0) if self._tp1_pct > 0 else grid_spacing
        raw_tp2 = (self._tp2_pct / 100.0) if self._tp2_pct > 0 else grid_spacing * 2.5

        # 波动率自适应止盈上限：低波动市中固定 TP（如 4%）可能超出市场日内振幅，
        # 导致 TradeCostAnalyzer 以「震荡空间不足」拦截全部开单。
        # 将 TP 上限锚定到 ATR 衍生的 grid_spacing，确保止盈在可达范围内。
        tp1_vol_cap = grid_spacing * 1.5
        tp2_vol_cap = grid_spacing * 3.0
        tp1_pct = min(raw_tp1, tp1_vol_cap) if tp1_vol_cap > 0 else raw_tp1
        tp2_pct = min(raw_tp2, tp2_vol_cap) if tp2_vol_cap > 0 else raw_tp2

        # 最小盈亏比保底：波动率上限可能将 TP 压缩到远低于止损距离，
        # 导致风险/收益倒挂（如 TP=0.75% vs SL=2%，R:R 仅 0.375:1）。
        # 确保 TP 至少维持 min_rr_ratio * SL 距离，否则交易期望值为负。
        grid_cfg = self.config.get("strategies", {}).get("grid", {})
        sl_pct = grid_cfg.get("stop_loss_pct", 0.02)
        min_rr_ratio = grid_cfg.get("min_rr_ratio", 0.75)
        tp1_floor = sl_pct * min_rr_ratio
        tp2_floor = sl_pct * min_rr_ratio * 2
        tp1_pct = max(tp1_pct, tp1_floor)
        tp2_pct = max(tp2_pct, tp2_floor)

        if tp1_pct != raw_tp1:
            logger.debug(
                f"[grid] {symbol} TP1 adjusted: {raw_tp1:.4%} -> {tp1_pct:.4%} "
                f"(grid_spacing={grid_spacing:.4%}, sl={sl_pct:.2%}, min_rr={min_rr_ratio})"
            )
        
        if layer == 0:
            take_profit_levels = [
                price * (1 + tp1_pct) if side == "buy" else price * (1 - tp1_pct),
                price * (1 + tp2_pct) if side == "buy" else price * (1 - tp2_pct)
            ]
        elif layer == 1:
            take_profit_levels = [
                price * (1 + tp1_pct) if side == "buy" else price * (1 - tp1_pct),
                price * (1 + tp2_pct) if side == "buy" else price * (1 - tp2_pct),
                price * (1 + grid_spacing * 4) if side == "buy" else price * (1 - grid_spacing * 4)
            ]
        else:
            vp = self._volume_profile_cache.get(symbol)
            if vp:
                target_price = vp["poc_price"] if side == "buy" else vp["vwap_price"]
                if side == "buy" and target_price > price:
                    take_profit_levels.append(target_price)
                elif side == "sell" and target_price < price:
                    take_profit_levels.append(target_price)
            
            take_profit_levels.extend([
                price * (1 + tp1_pct) if side == "buy" else price * (1 - tp1_pct),
                price * (1 + tp2_pct) if side == "buy" else price * (1 - tp2_pct),
                price * (1 + grid_spacing * 5) if side == "buy" else price * (1 - grid_spacing * 5),
                price * (1 + grid_spacing * 8) if side == "buy" else price * (1 - grid_spacing * 8)
            ])
        
        return [round(level, 4) for level in take_profit_levels]
    
    def _get_tier_from_symbol(self, symbol_or_price) -> str:
        if isinstance(symbol_or_price, str):
            symbol = symbol_or_price
            # 同时处理 "BTC-USDT-SWAP"（合约）和 "BTC-USDT"（现货）格式
            if "-USDT-SWAP" in symbol:
                base = symbol.replace("-USDT-SWAP", "")
            elif "-USDT" in symbol:
                base = symbol.replace("-USDT", "")
            else:
                base = symbol.split("-")[0] if "-" in symbol else symbol
            for tier in ["tier1", "tier2", "tier3"]:
                if base in self.config["currencies"].get(f"{tier}_symbols", []):
                    return tier
        return "tier2"

    def _record_trade(self, symbol: str, side: str, price: float, quantity: float, layer: int):
        if symbol not in self._trade_history:
            self._trade_history[symbol] = []
        
        self._trade_history[symbol].append({
            "time": datetime.now(),
            "side": side,
            "price": price,
            "quantity": quantity,
            "layer": layer,
            "martingale": self._martingale_state.get(symbol, 0)
        })
        
        if len(self._trade_history[symbol]) > 50:
            self._trade_history[symbol] = self._trade_history[symbol][-50:]

    async def _switch_to_trend_mode(self, symbol: str):
        now = time.time()
        last_switch = self._trend_mode_last_switch.get(symbol, 0)
        if now - last_switch < self._trend_mode_cooldown:
            logger.debug(f"Grid {symbol}: trend mode switch blocked by cooldown ({now - last_switch:.0f}s < {self._trend_mode_cooldown:.0f}s)")
            return
        self._trend_mode[symbol] = True
        self._trend_mode_last_switch[symbol] = now
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        pos_side = self._position_side.get(symbol)
        if pos_side is None:
            # _position_side为空时用ticker数据推断方向：比较当前价与24h开盘价
            open_24h = float(ticker.get("open24h", 0))
            pos_side = "buy" if (open_24h > 0 and current_price >= open_24h) else "sell"
        # P0-1: long_only 模式下趋势切换也只做多，禁止开空
        if self._long_only:
            pos_side = "buy"
        direction = "long" if pos_side == "buy" else "short"
        
        tier_settings = get_symbol_config(symbol, self.config)
        leverage = tier_settings["leverage_max"]
        
        precision = get_price_precision(symbol)
        slippage = tier_settings.get("slippage", 0.001)
        
        atr = self._atr_cache.get(symbol, 0)
        
        if atr > 0:
            take_profit, stop_loss = calculate_tp_sl_from_atr(current_price, direction, atr, 
                                                               tp_multiplier=3.0, sl_multiplier=2.0,
                                                               slippage_pct=slippage, precision=precision)
        else:
            stop_loss_pct = tier_settings["grid_spacing_max"] * self._martingale_layers
            stop_loss = calculate_stop_loss(current_price, direction, stop_loss_pct, slippage, precision)
            take_profit = calculate_take_profit(current_price, direction, stop_loss_pct * 3, slippage, precision)
        
        validation = validate_tp_sl_prices(current_price, direction, take_profit, stop_loss, current_price)
        if not validation["valid"]:
            logger.error(f"TP/SL validation failed for {symbol}: {validation['errors']}")
            return
        
        min_lot_size = self._safe_float((await self.okx_client.get_instrument_info_async(symbol) or {}).get("lotSz", "1"), 1.0)
        margin_needed_for_min_lot = current_price * min_lot_size / leverage
        
        effective_capital = self._get_effective_capital()
        if margin_needed_for_min_lot > effective_capital * self.config["trading"]["trading_capital_ratio"] * 0.1:
            logger.warning(f"Insufficient capital for trend switch {symbol}: need {margin_needed_for_min_lot:.2f} USDT. Skipping.")
            return
        
        signal = Signal(
            symbol=symbol,
            strategy_name="grid",
            signal_type="grid_trend_switch",
            direction=direction,
            price=current_price,
            quantity=0,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.6,
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
                "stop_loss": signal.stop_loss,
                "take_profit": signal.take_profit,
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat()
            }
        })
        
        logger.info(f"Grid -> Trend mode for {symbol}, direction: {direction}, SL: {stop_loss:.4f}, TP: {take_profit:.4f}")

        # 生产级：记录生命周期事件
        self._log_coin_lifecycle_event(symbol, "trend_mode_entered", {
            "direction": direction,
            "stop_loss": round(stop_loss, 4),
            "take_profit": round(take_profit, 4),
            "leverage": leverage,
        })

    async def _trend_monitor_loop(self):
        while True:
            for symbol in self._all_symbols:
                if self._trend_mode.get(symbol, False):
                    await self._check_trend_exit(symbol)
                    await self._check_reversal_take_profit(symbol)
            await asyncio.sleep(5)

    async def _check_reversal_take_profit(self, symbol: str):
        """趋势模式持仓的行情反转落袋（HMM 为主 + 指标兜底）。

        检测到反转落袋信号（已盈利 + 反转评分达标，含冷却去重）时退出趋势模式，
        回到网格模式，锁住趋势利润。
        """
        slm = self._stop_loss_manager
        if slm is None or not hasattr(slm, 'compute_reversal_take_profit'):
            return

        pos_side = self._position_side.get(symbol)
        if not pos_side:
            return
        direction = "long" if pos_side == "buy" else "short"

        trailing = self._trailing_state.get(symbol, {})
        entry_price = float(trailing.get("entry_price", 0) or 0)
        if entry_price <= 0:
            return

        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        current_price = float(ticker["last"])

        result = await slm.compute_reversal_take_profit(
            symbol=symbol,
            strategy_name="grid",
            direction=direction,
            entry_price=entry_price,
            current_price=current_price,
        )
        if result is None or result.exit_action == "none":
            return

        logger.warning(
            f"[REVERSAL_TAKE_PROFIT] grid {symbol}: action={result.exit_action}, "
            f"score={result.reversal_score:.2f}, source={result.reversal_source}, "
            f"pnl={result.details.get('pnl_pct', 0):.2%}"
        )
        await self._exit_trend_mode(symbol)

    async def _check_trend_exit(self, symbol: str):
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        
        klines = await self.okx_client.get_kline_async(symbol, "1H", limit=12)
        if len(klines) < 6:
            return
        
        closes = np.array([float(kline[4]) for kline in klines])
        highs = np.array([float(kline[2]) for kline in klines])
        lows = np.array([float(kline[3]) for kline in klines])
        
        recent_avg = np.mean(closes[-3:])
        prev_avg = np.mean(closes[-6:-3])
        
        rsi = self._calculate_rsi(closes)
        
        adx, plus_di, minus_di = self._calculate_adx(highs, lows, closes, symbol)
        
        if self._position_side.get(symbol) == "long":
            if recent_avg < prev_avg * 0.99 or rsi > 70 or (minus_di > plus_di and adx > 20):
                await self._exit_trend_mode(symbol)
        else:
            if recent_avg > prev_avg * 1.01 or rsi < 30 or (plus_di > minus_di and adx > 20):
                await self._exit_trend_mode(symbol)

    def _calculate_rsi(self, prices: np.ndarray) -> float:
        if len(prices) < 14:
            return 50
        
        if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-14:])
        avg_loss = np.mean(losses[-14:])
        
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

    def _calculate_adx(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, symbol: str = "") -> tuple:
        if len(highs) < 14:
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
                # P34: True Range 归一化——DI 必须除以 TR，否则随币价刻度漂移
                # （高价币 DM×100 饱和到 100，低价币退化到 0，方向判断失效）
                tr = max(highs[i] - lows[i],
                         abs(highs[i] - closes[i-1]),
                         abs(lows[i] - closes[i-1]))
                tr_values.append(tr)
                
                plus_dm = highs[i] - highs[i-1]
                minus_dm = lows[i-1] - lows[i]
                
                plus_di_values.append(plus_dm if (plus_dm > minus_dm and plus_dm > 0) else 0.0)
                minus_di_values.append(minus_dm if (minus_dm > plus_dm and minus_dm > 0) else 0.0)
            except (ValueError, TypeError):
                continue
        
        if len(tr_values) < 14:
            return 20, 25, 25
        
        tr_smooth = np.mean(tr_values[-14:])
        plus_di_smooth = np.mean(plus_di_values[-14:])
        minus_di_smooth = np.mean(minus_di_values[-14:])
        
        if np.isnan(tr_smooth) or np.isinf(tr_smooth) or tr_smooth == 0:
            plus_di = 0
            minus_di = 0
        else:
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

        if symbol:
            if symbol not in self._dx_history:
                self._dx_history[symbol] = []
            self._dx_history[symbol].append(dx)
            if len(self._dx_history[symbol]) > 14:
                self._dx_history[symbol] = self._dx_history[symbol][-14:]
            adx = np.mean(self._dx_history[symbol])
        else:
            adx = dx
        
        if np.isnan(adx) or np.isinf(adx):
            adx = 20
        
        return max(0, min(100, adx)), max(0, min(100, plus_di)), max(0, min(100, minus_di))

    async def _exit_trend_mode(self, symbol: str):
        now = time.time()
        last_switch = self._trend_mode_last_switch.get(symbol, 0)
        if now - last_switch < self._trend_mode_cooldown:
            logger.debug(f"Grid {symbol}: trend mode exit blocked by cooldown ({now - last_switch:.0f}s < {self._trend_mode_cooldown:.0f}s)")
            return
        self._trend_mode[symbol] = False
        self._trend_mode_last_switch[symbol] = now
        self._martingale_state[symbol] = 0
        self._trailing_state.pop(symbol, None)
        self._position_entry_time.pop(symbol, None)
        
        await self._build_grid(symbol)
        logger.info(f"Exited trend mode for {symbol}, grid rebuilt")

        # 生产级：记录生命周期事件
        self._log_coin_lifecycle_event(symbol, "trend_mode_exited", {
            "reason": "trend_exit_signal",
        })

    async def _dynamic_adjust_loop(self):
        while True:
            for symbol in self._all_symbols:
                if self._trend_mode.get(symbol, False):
                    continue
                await self._check_volatility_adjust(symbol)
            await asyncio.sleep(self._dynamic_adjust_interval)

    async def _check_volatility_adjust(self, symbol: str):
        # P23: 添加冷却时间，防止同一币种频繁调整网格（减少CPU和API消耗）
        now = time.time()
        last_adjust = self._last_vol_adjust_ts.get(symbol, 0)
        if now - last_adjust < self._vol_adjust_cooldown:
            return
        
        # P23: 使用异步kline接口，避免阻塞事件循环
        klines = await self.okx_client.get_kline_async(symbol, "1H", limit=24)
        if len(klines) < 24:
            return
        
        self._last_vol_adjust_ts[symbol] = now
        
        prices = np.array([float(kline[4]) for kline in klines])
        volatility = np.std(prices) / np.mean(prices)
        
        if volatility > self._volatility_threshold * 2.0:
            logger.info(f"Extreme volatility detected for {symbol}: {volatility:.2%}, entering extreme mode")
            await self._enter_extreme_volatility_mode(symbol, volatility)
        elif volatility > self._volatility_threshold:
            logger.debug(f"High volatility detected for {symbol}: {volatility:.2%}, expanding grid")
            await self._expand_grid(symbol, volatility)
        elif volatility < self._volatility_threshold * 0.5:
            logger.debug(f"Low volatility detected for {symbol}: {volatility:.2%}, contracting grid")
            await self._contract_grid(symbol)

    async def _enter_extreme_volatility_mode(self, symbol: str, volatility: float):
        # P0-3: 极端波动改为暂停而非主动开仓
        # 原逻辑会在极端波动时主动开仓追多/追空，小账户极易爆仓
        # 新逻辑：仅记录警告，暂停该 symbol 网格交易 30 分钟
        # 已有网格持仓保持不变，由 _check_stop_loss_orders 与止损单保护
        pause_seconds = 600  # 10 分钟
        self._extreme_vol_until[symbol] = time.time() + pause_seconds
        logger.warning(
            f"Extreme volatility detected for {symbol}: {volatility:.2%}. "
            f"Pausing grid trading for {pause_seconds/60:.0f} minutes. "
            f"Existing positions remain protected by stop-loss orders."
        )

    async def _expand_grid(self, symbol: str, volatility: float = None):
        current_grid = self._grids.get(symbol, [])
        if not current_grid:
            await self._build_grid(symbol)
            return
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        
        avg_spacing_pct = self._safe_avg_spacing(current_grid, current_price)

        # P16: 网格价格全部相同时(零间距)，avg_spacing_pct=0导致范围塌缩为单点
        # 此时应重建网格而非扩展，因为扩展0间距毫无意义
        if avg_spacing_pct < 0.001:
            logger.warning(
                f"P16: Grid {symbol} has zero spacing (all prices rounded to same tick), "
                f"rebuilding instead of expanding (price={current_price:.4f})"
            )
            self._grid_rebuild_cooldown[symbol] = time.time()
            await self._build_grid(symbol, current_price=current_price)
            return
        
        if volatility and volatility > self._volatility_threshold * 2:
            expansion_factor = 1.5
        else:
            expansion_factor = 1.25
        
        # P0-2: 动态扩张必须遵守 min_grid_spacing 下限，防止间距跌破盈利阈值
        expanded_spacing_pct = max(self._min_grid_spacing, avg_spacing_pct * expansion_factor)
        if not math.isfinite(expanded_spacing_pct) or expanded_spacing_pct <= 0:
            expanded_spacing_pct = self._min_grid_spacing
        
        grid_count = len(current_grid)
        half_grids = grid_count // 2
        
        upper_bound = current_price * (1 + expanded_spacing_pct * half_grids)
        lower_bound = current_price * (1 - expanded_spacing_pct * half_grids)
        
        lower_bound = max(lower_bound, current_price * 0.5)
        upper_bound = min(upper_bound, current_price * 1.5)
        
        if lower_bound >= upper_bound:
            lower_bound = current_price * (1 - expanded_spacing_pct * half_grids)
            upper_bound = current_price * (1 + expanded_spacing_pct * half_grids)
        
        prices = np.linspace(lower_bound, upper_bound, grid_count)
        
        # P0: 渐进式调整——保留已存在的网格状态
        existing_by_price = {}
        for g in current_grid:
            existing_by_price[g["price"]] = g
        
        price_tolerance = (prices[-1] - prices[0]) / grid_count * 0.3
        
        new_grid = []
        for i, price in enumerate(prices):
            rounded_price = self.okx_client.round_price_to_tick(symbol, price)
            side = "buy" if (self._long_only or i < half_grids) else "sell"  # P0-1: long_only 禁止 sell 层
            
            # 查找匹配的已有网格（价格在容差范围内）
            matched = None
            for ep, eg in existing_by_price.items():
                if abs(ep - rounded_price) <= price_tolerance:
                    matched = eg
                    break
            
            if matched and matched.get("filled") in (True, "pending"):
                # 保留已成交/pending状态的网格
                matched["layer"] = abs(i - half_grids)
                matched["side"] = side
                if "density_factor" not in matched:
                    matched["density_factor"] = self._calculate_density_factor(symbol, rounded_price, None)
                matched["adjusted_spacing"] = expanded_spacing_pct * matched.get("density_factor", 1.0)
                new_grid.append(matched)
            elif matched:
                # 已有但未成交的网格，合并保留但更新属性
                matched["layer"] = abs(i - half_grids)
                matched["side"] = side
                matched["density_factor"] = self._calculate_density_factor(symbol, rounded_price, None)
                matched["adjusted_spacing"] = expanded_spacing_pct * matched["density_factor"]
                new_grid.append(matched)
            else:
                density_factor = self._calculate_density_factor(symbol, rounded_price, None)
                new_grid.append({
                    "price": rounded_price,
                    "side": side,
                    "filled": False,
                    "layer": abs(i - half_grids),
                    "quantity": 0,
                    "density_factor": density_factor,
                    "adjusted_spacing": expanded_spacing_pct * density_factor
                })
        
        self._grids[symbol] = new_grid
        # P16: 更新实际网格间距，防止_trigger_grid_order使用过期间距导致0.0000%误判
        self._actual_grid_spacing[symbol] = expanded_spacing_pct
        preserved = sum(1 for g in new_grid if g.get("filled") in (True, "pending"))
        logger.debug(f"Grid expanded for {symbol}: spacing x{expansion_factor}, range [{lower_bound:.4f}, {upper_bound:.4f}], preserved {preserved} filled/pending")

        # 生产级：记录生命周期事件
        self._log_coin_lifecycle_event(symbol, "grid_expanded", {
            "expansion_factor": expansion_factor,
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "preserved_filled": preserved,
            "total_grids": len(new_grid),
        })

    async def _contract_grid(self, symbol: str):
        current_grid = self._grids.get(symbol, [])
        if not current_grid:
            await self._build_grid(symbol)
            return
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        
        # P13: 网格重建冷却检查 - 防止高频重建导致无法成交
        last_rebuild = self._grid_rebuild_cooldown.get(symbol, 0)
        if time.time() - last_rebuild < self._grid_rebuild_cooldown_seconds:
            return  # 冷却中，跳过本次调整
        
        # P13: 空网格不收缩，直接重建到当前价格
        filled_count = sum(1 for g in current_grid if g.get("filled") in (True, "pending"))
        if filled_count == 0:
            logger.debug(f"P13: Grid {symbol} empty (0 filled), rebuilding at current price {current_price:.4f}")
            self._grid_rebuild_cooldown[symbol] = time.time()
            # P14: 传入当前价格避免使用过期缓存ticker
            await self._rebuild_stale_grid(symbol, current_price=current_price)
            return
        
        avg_spacing_pct = self._safe_avg_spacing(current_grid, current_price)

        # P16: 网格价格全部相同时(零间距)，avg_spacing_pct=0导致范围塌缩
        # 收缩零间距网格无意义，直接重建
        if avg_spacing_pct < 0.001:
            logger.warning(
                f"P16: Grid {symbol} has zero spacing, "
                f"rebuilding instead of contracting (price={current_price:.4f})"
            )
            self._grid_rebuild_cooldown[symbol] = time.time()
            await self._build_grid(symbol, current_price=current_price)
            return
        
        # P13: 最小网格间距保护 - 防止过度收缩导致价格偏离
        MIN_RANGE_PCT = 0.05  # 最小范围5% (相对于当前价格)
        
        # P13: 基于波动率的网格范围自适应 - 高波动扩大范围，低波动缩小范围
        atr = self._atr_cache.get(symbol, 0)
        if atr > 0 and current_price > 0:
            atr_pct = atr / current_price
            # ATR越大，网格范围越大；但限制在 [MIN_RANGE_PCT, 0.15] 之间
            vol_adjusted_range = max(MIN_RANGE_PCT, min(0.15, atr_pct * 3))
            # P0-2: 收缩必须遵守 min_grid_spacing 下限，防止 *0.8 长期收缩跌破盈利阈值
            contracted_spacing_pct = max(self._min_grid_spacing, avg_spacing_pct * 0.8, vol_adjusted_range / len(current_grid))
        else:
            contracted_spacing_pct = max(self._min_grid_spacing, avg_spacing_pct * 0.8, MIN_RANGE_PCT / len(current_grid))
        if not math.isfinite(contracted_spacing_pct) or contracted_spacing_pct <= 0:
            contracted_spacing_pct = self._min_grid_spacing
        
        grid_count = len(current_grid)
        half_grids = grid_count // 2
        
        upper_bound = current_price * (1 + contracted_spacing_pct * half_grids)
        lower_bound = current_price * (1 - contracted_spacing_pct * half_grids)
        
        # P13: 检查当前价格是否在收缩后的范围内
        # 如果价格已经偏离网格中心太远，直接重建而非收缩
        stale_count = self._grid_stale_count.get(symbol, 0)
        if stale_count >= 5 and current_price > 0:
            logger.warning(
                f"P13: Grid {symbol} stale count {stale_count}, "
                f"rebuilding instead of contracting (price={current_price:.4f})"
            )
            self._grid_stale_count[symbol] = 0
            # P14: 传入当前价格避免使用过期缓存ticker
            await self._rebuild_stale_grid(symbol, current_price=current_price)
            return
        
        # P13: 确保价格在新范围内（至少靠近边界）
        if current_price < lower_bound * 0.95 or current_price > upper_bound * 1.05:
            logger.warning(
                f"P13: Grid {symbol} price {current_price:.4f} outside contracted range "
                f"[{lower_bound:.4f}, {upper_bound:.4f}], rebuilding instead"
            )
            # P14: 传入当前价格避免使用过期缓存ticker
            await self._rebuild_stale_grid(symbol, current_price=current_price)
            return
        
        prices = np.linspace(lower_bound, upper_bound, grid_count)
        
        # P0: 渐进式调整——保留已存在的网格状态
        existing_by_price = {}
        for g in current_grid:
            existing_by_price[g["price"]] = g
        
        price_tolerance = (prices[-1] - prices[0]) / grid_count * 0.3
        
        new_grid = []
        for i, price in enumerate(prices):
            rounded_price = self.okx_client.round_price_to_tick(symbol, price)
            side = "buy" if (self._long_only or i < half_grids) else "sell"  # P0-1: long_only 禁止 sell 层
            
            # 查找匹配的已有网格（价格在容差范围内）
            matched = None
            for ep, eg in existing_by_price.items():
                if abs(ep - rounded_price) <= price_tolerance:
                    matched = eg
                    break
            
            if matched and matched.get("filled") in (True, "pending"):
                # 保留已成交/pending状态的网格
                matched["layer"] = abs(i - half_grids)
                matched["side"] = side
                if "density_factor" not in matched:
                    matched["density_factor"] = self._calculate_density_factor(symbol, rounded_price, None)
                matched["adjusted_spacing"] = contracted_spacing_pct * matched.get("density_factor", 1.0)
                new_grid.append(matched)
            elif matched:
                # 已有但未成交的网格，合并保留但更新属性
                matched["layer"] = abs(i - half_grids)
                matched["side"] = side
                matched["density_factor"] = self._calculate_density_factor(symbol, rounded_price, None)
                matched["adjusted_spacing"] = contracted_spacing_pct * matched["density_factor"]
                new_grid.append(matched)
            else:
                density_factor = self._calculate_density_factor(symbol, rounded_price, None)
                new_grid.append({
                    "price": rounded_price,
                    "side": side,
                    "filled": False,
                    "layer": abs(i - half_grids),
                    "quantity": 0,
                    "density_factor": density_factor,
                    "adjusted_spacing": contracted_spacing_pct * density_factor
                })
        
        self._grids[symbol] = new_grid
        # P16: 更新实际网格间距，防止_trigger_grid_order使用过期间距导致0.0000%误判
        self._actual_grid_spacing[symbol] = contracted_spacing_pct
        preserved = sum(1 for g in new_grid if g.get("filled") in (True, "pending"))
        logger.debug(f"Grid contracted for {symbol}: spacing x0.8, range [{lower_bound:.4f}, {upper_bound:.4f}], preserved {preserved} filled/pending")

        # 生产级：记录生命周期事件
        self._log_coin_lifecycle_event(symbol, "grid_contracted", {
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "preserved_filled": preserved,
            "total_grids": len(new_grid),
        })

    

    def _get_effective_martingale_layers(self, symbol: str) -> int:
        """R11: 按 ATR 自适应马丁格尔层数上限 — 高波动(单边行情)允许多层加仓，低波动用基础值。"""
        atr = self._atr_cache.get(symbol, 0)
        if atr <= 0:
            return self._martingale_layers
        last_price = self._last_tick_price.get(symbol, 0)
        if last_price <= 0:
            return self._martingale_layers
        atr_ratio = atr / last_price
        # atr_ratio > 0.03 (高波动): +2 层; > 0.02 (中波动): +1 层; 其余: 基础值
        if atr_ratio > 0.03:
            return self._martingale_layers + 2
        elif atr_ratio > 0.02:
            return self._martingale_layers + 1
        return self._martingale_layers

    def handle_trade_completion(self, symbol: str, is_profitable: bool):
        if is_profitable:
            self._martingale_state[symbol] = 0
            for grid in self._grids.get(symbol, []):
                grid["filled"] = False
            # 清理移动止损追踪和入场时间
            self._trailing_state.pop(symbol, None)
            self._position_entry_time.pop(symbol, None)
        else:
            # R11: 动态层数上限 — 高波动行情允许多层加仓
            effective_max = self._get_effective_martingale_layers(symbol)
            self._martingale_state[symbol] = min(self._martingale_state.get(symbol, 0) + 1, effective_max)
    
    def on_order_filled(self, fill_payload: Dict[str, Any]):
        """成交回执回调（ghost_close 专项 Phase 4）：匹配网格层 clOrdId 并提前确认成交。

        网格采用逐层 pending（`_grid_pending_at` + `grid["filled"]="pending"`），
        本回调在订单 FILLED 时立即物化该层为 `filled=True`，无需等待 60s 的
        `_reconcile_pending_grids` 对账循环；`_reconcile_pending_grids` 仍作为兜底。
        """
        try:
            if fill_payload.get("strategy_name", "") != "grid":
                return
            if not self._use_fill_driven_position:
                return

            symbol = fill_payload.get("symbol", "")
            clordid = fill_payload.get("clOrdId", "")
            if not symbol or not clordid:
                return

            grids = self._grids.get(symbol, [])
            for idx, grid in enumerate(grids):
                if grid.get("clOrdId") != clordid or grid.get("filled") != "pending":
                    continue

                grid["filled"] = True
                grid.pop("clOrdId", None)
                pending_at = self._grid_pending_at.get(symbol, {})
                pending_at.pop(idx, None)
                if not pending_at:
                    self._grid_pending_at.pop(symbol, None)
                # 成交成功，清除该层重试/冷却记录
                self._grid_retry_count.get(symbol, {}).pop(idx, None)
                self._grid_retry_first_ts.get(symbol, {}).pop(idx, None)
                self._grid_rollback_cooldown.get(symbol, {}).pop(idx, None)
                self._fill_callback_hits += 1
                logger.info(
                    f"[ghost_close-P4] grid layer filled: {symbol}[{idx}] "
                    f"layer={grid.get('layer')} clOrdId={clordid}"
                )
                return
        except Exception as e:
            logger.debug(f"on_order_filled error: {e}")

    async def _order_check_loop(self):
        while True:
            for symbol in self._all_symbols:
                await self._check_pending_orders(symbol)
                await self._check_stop_loss_orders(symbol)
                await self._reconcile_pending_grids(symbol)
            await asyncio.sleep(self._order_check_interval)

    async def _reconcile_pending_grids(self, symbol: str):
        """对账pending网格：基于实际持仓查询确认成交，超时回滚并增加冷却。
        
        P0改进：
        1. 查询OKX实际持仓确认网格是否真实成交（不再仅依赖_position_side）
        2. 回滚后增加冷却期 + 重试计数，防止同一层反复触发
        3. 冷却时间随重试次数递增：30s -> 90s -> 180s -> 永久禁用
        """
        grids = self._grids.get(symbol, [])
        if not grids:
            return

        pending_at = self._grid_pending_at.get(symbol, {})
        if not pending_at:
            return

        now = time.time()
        pending_timeout = 180  # P18-4: 180秒超时回滚（网络不稳定时给予更多确认时间）
        
        # P31: 网络降级时缩短超时时间，避免资源浪费
        if self.okx_client and hasattr(self.okx_client, 'is_network_degraded'):
            if self.okx_client.is_network_degraded:
                pending_timeout = 60
                logger.debug(f"P31: Grid pending timeout reduced to {pending_timeout}s due to network degradation")
        resolved_indices = []

        # P0: 查询OKX实际持仓状态（有OKX client才做）
        actual_positions = {}
        try:
            if self.okx_client:
                all_positions = self.okx_client.get_positions()
                if all_positions:
                    for p in all_positions:
                        if p.get("instId", "") == symbol:
                            side = p.get("posSide", "")
                            qty = float(p.get("pos", 0))
                            if qty > 0:
                                actual_positions[side] = {
                                    "quantity": qty,
                                    "avg_px": float(p.get("avgPx", 0)),
                                    "upl": float(p.get("upl", 0)),
                                }
        except Exception as e:
            logger.debug(f"Failed to query positions for {symbol} during grid reconcile: {e}")

        # 网格方向 -> posSide 映射
        # grid side "buy" 对应合约 posSide "long"
        # grid side "sell" 对应合约 posSide "short"
        has_long_pos = "long" in actual_positions
        has_short_pos = "short" in actual_positions

        for grid_idx, pending_ts in list(pending_at.items()):
            if grid_idx >= len(grids):
                resolved_indices.append(grid_idx)
                continue

            grid = grids[grid_idx]
            if grid["filled"] != "pending":
                resolved_indices.append(grid_idx)
                continue

            grid_side = grid.get("side", "")
            grid_layer = grid.get("layer", 0)

            # P0-1: 基于OKX实际持仓确认成交
            # 网格 buy 层 -> 应有 long 持仓；sell 层 -> 应有 short 持仓
            position_confirmed = False
            if grid_side == "buy" and has_long_pos:
                position_confirmed = True
            elif grid_side == "sell" and has_short_pos:
                position_confirmed = True

            if position_confirmed and (now - pending_ts) > 15:
                # P18-4: 有实际持仓 + 已等15秒(订单成交确认时间) -> 确认成交
                grid["filled"] = True
                resolved_indices.append(grid_idx)
                # 清除该层的重试计数（成功成交）
                if symbol in self._grid_retry_count:
                    self._grid_retry_count[symbol].pop(grid_idx, None)
                if symbol in self._grid_retry_first_ts:
                    self._grid_retry_first_ts[symbol].pop(grid_idx, None)
                if symbol in self._grid_rollback_cooldown:
                    self._grid_rollback_cooldown[symbol].pop(grid_idx, None)
                logger.info(f"Grid {symbol}[{grid_idx}] layer={grid_layer} confirmed filled via actual position check")
                continue

            # P0-2: 检查通过TradeJournal的历史成交
            trade_confirmed = False
            if not position_confirmed and (now - pending_ts) > 20:
                try:
                    if hasattr(self, '_trade_journal') and self._trade_journal:
                        # 查询该symbol最近的成交记录
                        recent_trades = getattr(self._trade_journal, 'get_recent_trades', None)
                        if recent_trades:
                            trades = recent_trades(symbol, limit=10)
                            for t in trades:
                                trade_time = t.get("timestamp", 0)
                                if isinstance(trade_time, datetime):
                                    trade_time = trade_time.timestamp()
                                if abs(now - trade_time) < 120:  # 2分钟内的成交
                                    trade_confirmed = True
                                    break
                except Exception:
                    pass

            if trade_confirmed:
                grid["filled"] = True
                resolved_indices.append(grid_idx)
                if symbol in self._grid_retry_count:
                    self._grid_retry_count[symbol].pop(grid_idx, None)
                if symbol in self._grid_retry_first_ts:
                    self._grid_retry_first_ts[symbol].pop(grid_idx, None)
                if symbol in self._grid_rollback_cooldown:
                    self._grid_rollback_cooldown[symbol].pop(grid_idx, None)
                logger.info(f"Grid {symbol}[{grid_idx}] layer={grid_layer} confirmed filled via trade journal")
                continue

            # 超时回滚：增加冷却 + 重试计数
            if now - pending_ts > pending_timeout:
                grid["filled"] = False
                grid.pop("clOrdId", None)  # ghost_close 专项 Phase 4: 清理关联的 clOrdId
                resolved_indices.append(grid_idx)
                self._pending_timeout_cleaned += 1  # ghost_close 专项 Phase 4: 观测指标

                # 记录重试次数
                if symbol not in self._grid_retry_count:
                    self._grid_retry_count[symbol] = {}
                if symbol not in self._grid_retry_first_ts:
                    self._grid_retry_first_ts[symbol] = {}
                retry_count = self._grid_retry_count[symbol].get(grid_idx, 0) + 1
                self._grid_retry_count[symbol][grid_idx] = retry_count
                if retry_count == 1:
                    self._grid_retry_first_ts[symbol][grid_idx] = now

                # R15: 递增冷却时间：60s → 120s → 300s → 600s → 900s，上限6次
                cooldown_map = {1: 60, 2: 120, 3: 300, 4: 600, 5: 900}
                cooldown = cooldown_map.get(retry_count, 1200)

                if symbol not in self._grid_rollback_cooldown:
                    self._grid_rollback_cooldown[symbol] = {}
                self._grid_rollback_cooldown[symbol][grid_idx] = now + cooldown

                logger.info(
                    f"Grid {symbol}[{grid_idx}] layer={grid_layer} rolled back "
                    f"(timeout {pending_timeout}s, retry={retry_count}, cooldown={cooldown}s)"
                )

        for idx in resolved_indices:
            pending_at.pop(idx, None)

        if not pending_at:
            self._grid_pending_at.pop(symbol, None)
        
        # P0: 清理过期的冷却记录（冷却时间已过的自动清除）
        self._cleanup_expired_cooldowns(symbol, now)

    def cleanup_position(self, symbol: str, strategy_name: str = ""):
        """P6: 清理幽灵仓位 - 清除网格策略对该symbol的所有内部状态
        
        当订单执行器检测到51169错误（交易所无对应持仓）时调用，
        清除grid层、马丁格尔状态、趋势标记等所有相关状态。
        """
        try:
            self._ghost_close_count += 1  # ghost_close 专项 Phase 4: 观测指标
            # 清除网格层
            if symbol in self._grids:
                grid_count = len(self._grids[symbol])
                del self._grids[symbol]
                logger.info(f"Grid cleanup_position: removed {grid_count} grid layers for {symbol}")
            
            # 清除马丁格尔状态
            self._martingale_state.pop(symbol, None)
            
            # 清除趋势标记
            self._trend_mode.pop(symbol, None)
            
            # 清除持仓方向
            self._position_side.pop(symbol, None)
            # P33: 记录退出时间，用于冷却追踪
            self._last_exit_time[symbol] = time.time()
            
            # 清除调整时间
            self._last_adjust_time.pop(symbol, None)
            
            # 清除冷却记录
            self._grid_rollback_cooldown.pop(symbol, None)
            
            # 清除生命周期事件中的pending记录
            if symbol in self._coin_lifecycle:
                events = self._coin_lifecycle[symbol]
                # 标记最后一个事件为cleaned
                if events:
                    events[-1]["status"] = "cleaned_by_51169"
            
            logger.info(f"Grid cleanup_position: fully cleaned state for {symbol}")
        except Exception as e:
            logger.error(f"Grid cleanup_position failed for {symbol}: {e}")

    def _cleanup_expired_cooldowns(self, symbol: str, now: float):
        """清理已过期的冷却记录，防止内存泄漏"""
        cooldowns = self._grid_rollback_cooldown.get(symbol, {})
        if cooldowns:
            expired = [idx for idx, until in cooldowns.items() if now >= until]
            for idx in expired:
                cooldowns.pop(idx, None)
            if not cooldowns:
                self._grid_rollback_cooldown.pop(symbol, None)
        
        retries = self._grid_retry_count.get(symbol, {})
        if retries:
            # 清理冷却已过期且超过1小时的retry计数
            stale = []
            for idx, count in retries.items():
                if count >= 6 and symbol in self._grid_rollback_cooldown:
                    # R15: 高重试次数的如果冷却也过期了，清理掉
                    continue
                if count >= 8:
                    stale.append(idx)
            for idx in stale:
                retries.pop(idx, None)
                self._grid_retry_first_ts.get(symbol, {}).pop(idx, None)
            if not retries:
                self._grid_retry_count.pop(symbol, None)
                self._grid_retry_first_ts.pop(symbol, None)

    async def _check_pending_orders(self, symbol: str):
        pending = self._pending_orders.get(symbol, [])
        if not pending:
            return
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        still_pending = []
        
        for order in pending:
            if order["status"] != "pending":
                continue
            
            should_trigger = False
            if order["type"] == "take_profit":
                if order["direction"] == "long" and current_price >= order["price"]:
                    should_trigger = True
                elif order["direction"] == "short" and current_price <= order["price"]:
                    should_trigger = True
            elif order["type"] == "stop_loss":
                if order["direction"] == "long" and current_price <= order["price"]:
                    should_trigger = True
                elif order["direction"] == "short" and current_price >= order["price"]:
                    should_trigger = True
            
            if should_trigger:
                order["status"] = "triggered"
                await self._execute_pending_order(symbol, order)
            else:
                still_pending.append(order)
        
        self._pending_orders[symbol] = still_pending

    async def _check_stop_loss_orders(self, symbol: str):
        sl_order = self._stop_loss_orders.get(symbol)
        if not sl_order or sl_order["status"] != "active":
            return

        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return

        sl_price = sl_order["price"]
        # P0-1: 优先使用 mark price 作为止损判断价格，fallback 到 last price
        mark_px_str = ticker.get("markPx")
        last_price = float(ticker["last"])
        mark_price = float(mark_px_str) if mark_px_str else last_price

        direction = sl_order["direction"]
        entry_price = sl_order.get("entry_price")
        trailing_state = self._trailing_state.get(symbol, {})

        # === 1. 保本止损（配置驱动） ===
        if entry_price and entry_price > 0:
            if direction == "long":
                pnl_pct = (last_price - entry_price) / entry_price
                if pnl_pct >= self._breakeven_trigger_pct:
                    breakeven_sl = entry_price * (1 + self._breakeven_stop_pct)
                    if breakeven_sl > sl_price:
                        await self._update_stop_loss_order(symbol, breakeven_sl)
                        sl_price = breakeven_sl
                        trailing_state["trailing_activated"] = True
            elif direction == "short":
                pnl_pct = (entry_price - last_price) / entry_price
                if pnl_pct >= self._breakeven_trigger_pct:
                    breakeven_sl = entry_price * (1 - self._breakeven_stop_pct)
                    if breakeven_sl < sl_price:
                        await self._update_stop_loss_order(symbol, breakeven_sl)
                        sl_price = breakeven_sl
                        trailing_state["trailing_activated"] = True

        # === 2. 移动止损 ===
        if self._trailing_stop_enabled and trailing_state:
            if direction == "long":
                highest = trailing_state.get("highest_price")
                if highest is None or last_price > highest:
                    trailing_state["highest_price"] = last_price
                if trailing_state.get("trailing_activated"):
                    trail_sl = trailing_state["highest_price"] * (1 - self._trailing_stop_pct)
                    if trail_sl > sl_price:
                        await self._update_stop_loss_order(symbol, trail_sl)
                        sl_price = trail_sl
                        trailing_state["trailing_sl"] = trail_sl
            elif direction == "short":
                lowest = trailing_state.get("lowest_price")
                if lowest is None or last_price < lowest:
                    trailing_state["lowest_price"] = last_price
                if trailing_state.get("trailing_activated"):
                    trail_sl = trailing_state["lowest_price"] * (1 + self._trailing_stop_pct)
                    if trail_sl < sl_price:
                        await self._update_stop_loss_order(symbol, trail_sl)
                        sl_price = trail_sl
                        trailing_state["trailing_sl"] = trail_sl

        # === 3. 时间止盈（持有超时强制退出） ===
        if self._time_exit_enabled and symbol in self._position_entry_time:
            hold_seconds = time.time() - self._position_entry_time[symbol]
            hold_hours = hold_seconds / 3600.0
            if hold_hours >= self._max_hold_hours:
                logger.info(
                    f"Time exit triggered for {symbol}: held {hold_hours:.1f}h >= "
                    f"max {self._max_hold_hours}h, force closing position"
                )
                await self._trigger_stop_loss(symbol, sl_order)
                return
            elif hold_hours >= self._time_exit_after_hours and self._time_exit_partial_pct > 0:
                # 部分退出：将止盈价收紧到接近当前价，让执行层做部分平仓
                if direction == "long":
                    tight_target = entry_price * (1 + self._breakeven_trigger_pct * 0.5)
                    if tight_target > sl_price:
                        logger.info(
                            f"Time partial exit for {symbol}: held {hold_hours:.1f}h >= "
                            f"{self._time_exit_after_hours}h, tightening stop to breakeven"
                        )
                        await self._update_stop_loss_order(symbol, tight_target)
                        sl_price = tight_target

        # === 4. 波动率止盈（波动率飙升时部分退出） ===
        if self._volatility_stop_enabled and entry_price and entry_price > 0:
            atr = self._atr_cache.get(symbol, 0)
            if atr > 0 and entry_price > 0:
                current_vol = atr / entry_price
                # 计算历史波动率均值作为基准
                avg_vol = self._volatility_threshold  # 使用配置阈值作为基准
                if avg_vol > 0 and current_vol > avg_vol * self._vol_spike_threshold:
                    logger.info(
                        f"Volatility spike detected for {symbol}: "
                        f"current_vol={current_vol:.4%} > {avg_vol * self._vol_spike_threshold:.4%}, "
                        f"tightening stop loss"
                    )
                    if direction == "long":
                        vol_sl = last_price * (1 - self._vol_stop_partial_pct * current_vol)
                        if vol_sl > sl_price:
                            await self._update_stop_loss_order(symbol, vol_sl)
                            sl_price = vol_sl
                    elif direction == "short":
                        vol_sl = last_price * (1 + self._vol_stop_partial_pct * current_vol)
                        if vol_sl < sl_price:
                            await self._update_stop_loss_order(symbol, vol_sl)
                            sl_price = vol_sl

        # === 5. 止损触发判断 ===
        if direction == "long":
            if mark_price <= sl_price or last_price <= sl_price * 0.995:
                logger.info(
                    f"Stop loss trigger (long) {symbol}: mark={mark_price:.4f}, "
                    f"last={last_price:.4f}, sl={sl_price:.4f}"
                )
                await self._trigger_stop_loss(symbol, sl_order)
        elif direction == "short":
            if mark_price >= sl_price or last_price >= sl_price * 1.005:
                logger.info(
                    f"Stop loss trigger (short) {symbol}: mark={mark_price:.4f}, "
                    f"last={last_price:.4f}, sl={sl_price:.4f}"
                )
                await self._trigger_stop_loss(symbol, sl_order)

    async def _execute_pending_order(self, symbol: str, order: Dict[str, Any]):
        tier_settings = get_symbol_config(symbol, self.config)
        
        # exit_reason 全链路补全：止盈/止损挂单必须显式透传 reason，禁止下游兜底 "close"
        exit_reason = "take_profit" if order.get("type") == "take_profit" else "stop_loss"
        signal_data = {
            "type": "signal",
            "data": {
                "symbol": symbol,
                "strategy_name": "grid",
                "signal_type": f"grid_{order['type']}",
                "direction": order["close_direction"],
                "price": order["price"],
                "quantity": order["quantity"],
                "leverage": tier_settings["leverage_default"],
                "confidence": 0.9,
                "exit_reason": exit_reason,
                "reduce_only": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.redis_cache.publish_signal(signal_data)
                logger.info(f"Pending order executed: {order['type']} {symbol} @ {order['price']:.4f}, qty: {order['quantity']:.4f}")
                return
            except Exception as e:
                logger.error(f"Failed to execute pending order (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
        
        logger.error(f"All {max_retries} retries failed for pending order {order['id']}. Re-adding to pending list.")
        order["status"] = "pending"
        order["retries"] = order.get("retries", 0) + 1
        
        if order["retries"] < 5:
            if symbol not in self._pending_orders:
                self._pending_orders[symbol] = []
            self._pending_orders[symbol].append(order)
            logger.warning(f"Pending order re-added for retry: {order['id']}, retry count: {order['retries']}")
        else:
            logger.error(f"Order {order['id']} has exceeded max retries, discarding")

    async def _trigger_stop_loss(self, symbol: str, sl_order: Dict[str, Any]):
        sl_order["status"] = "triggered"
        
        # P34: 记录单方向止损事件，供熔断判定（position_direction = 持仓方向）
        self._record_side_stop_loss(symbol, sl_order.get("direction", ""))
        
        # 清理移动止损追踪和入场时间
        self._trailing_state.pop(symbol, None)
        self._position_entry_time.pop(symbol, None)
        
        tier_settings = get_symbol_config(symbol, self.config)
        
        # exit_reason 全链路补全：止损必须显式透传 reason，禁止下游兜底 "close"
        signal_data = {
            "type": "signal",
            "data": {
                "symbol": symbol,
                "strategy_name": "grid",
                "signal_type": "grid_stop_loss",
                "direction": sl_order["close_direction"],
                "price": sl_order["price"],
                "quantity": sl_order["quantity"],
                "leverage": tier_settings["leverage_max"],
                "confidence": 0.95,
                "exit_reason": "stop_loss",
                "reduce_only": True,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                self.redis_cache.publish_signal(signal_data)
                logger.info(f"Stop loss triggered for {symbol} @ {sl_order['price']:.4f}, switching to trend mode")
                break
            except Exception as e:
                logger.error(f"Failed to trigger stop loss (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2

        # stop_loss_audit 落库：网格用自身止损触发逻辑（不走 StopLossManager.check_and_execute_stop_loss），
        # 必须显式补记审计，否则 stop_loss_audit 表缺失网格止损记录，复盘时止损执行质量不可见。
        self._record_grid_stop_loss_audit(symbol, sl_order)
        
        try:
            await self._switch_to_trend_mode(symbol)
        except Exception as e:
            logger.error(f"Failed to switch to trend mode after stop loss: {e}")

    def _record_grid_stop_loss_audit(self, symbol: str, sl_order: Dict[str, Any]):
        """网格止损审计落库（无副作用，失败不影响主流程）。"""
        slm = getattr(self, "_stop_loss_manager", None)
        if slm is None or not hasattr(slm, "record_stop_loss_audit"):
            return
        try:
            direction = sl_order.get("direction", "")
            entry_price = float(sl_order.get("entry_price") or sl_order.get("price") or 0.0)
            trigger_price = float(sl_order.get("price") or 0.0)
            quantity = float(sl_order.get("quantity") or 0.0)
            if entry_price <= 0 or trigger_price <= 0 or quantity <= 0:
                return
            if direction == "long":
                pnl = (trigger_price - entry_price) * quantity
                pnl_pct = (trigger_price - entry_price) / entry_price
            else:
                pnl = (entry_price - trigger_price) * quantity
                pnl_pct = (entry_price - trigger_price) / entry_price
            slm.record_stop_loss_audit(
                symbol=symbol,
                strategy_name="grid",
                trigger_type="hard_stop",
                entry_price=entry_price,
                trigger_price=trigger_price,
                exit_price=trigger_price,
                quantity=quantity,
                pnl=pnl,
                pnl_percent=pnl_pct,
                exit_reason="stop_loss",
            )
        except Exception as e:
            logger.warning(f"Failed to record grid stop_loss_audit for {symbol}: {e}")

    def _add_pending_order(self, symbol: str, order_type: str, direction: str, 
                          price: float, quantity: float):
        if symbol not in self._pending_orders:
            self._pending_orders[symbol] = []
        
        close_direction = "sell" if direction == "long" else "buy"
        
        order = {
            "id": f"{symbol}_{order_type}_{int(datetime.now().timestamp())}",
            "type": order_type,
            "direction": direction,
            "close_direction": close_direction,
            "price": price,
            "quantity": quantity,
            "status": "pending",
            "created_at": datetime.now()
        }
        
        self._pending_orders[symbol].append(order)
        
        if len(self._pending_orders[symbol]) > 20:
            self._pending_orders[symbol] = self._pending_orders[symbol][-20:]
        
        logger.debug(f"Pending order added: {order_type} {symbol} @ {price:.4f}")

    def _add_stop_loss_order(self, symbol: str, direction: str, price: float, quantity: float, entry_price: float = None):
        close_direction = "sell" if direction == "long" else "buy"
        
        self._stop_loss_orders[symbol] = {
            "type": "stop_loss",
            "direction": direction,
            "close_direction": close_direction,
            "price": price,
            "quantity": quantity,
            "status": "active",
            "created_at": datetime.now(),
            "entry_price": entry_price or price
        }
        
        logger.info(f"Stop loss order added: {symbol} @ {price:.4f}")

    def _cancel_stop_loss_order(self, symbol: str):
        if symbol in self._stop_loss_orders:
            self._stop_loss_orders[symbol]["status"] = "cancelled"
            logger.info(f"Stop loss order cancelled: {symbol}")

    async def _update_stop_loss_order(self, symbol: str, new_price: float):
        if symbol not in self._stop_loss_orders:
            return
        
        sl_order = self._stop_loss_orders[symbol]
        if sl_order["status"] != "active":
            return
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            logger.warning(f"Cannot update SL: no ticker for {symbol}")
            return
        
        current_price = float(ticker["last"])
        direction = sl_order["direction"]
        
        if direction == "long" and new_price >= current_price:
            logger.warning(f"SL reset blocked: new SL ({new_price:.4f}) would immediately trigger above current price ({current_price:.4f})")
            return
        elif direction == "short" and new_price <= current_price:
            logger.warning(f"SL reset blocked: new SL ({new_price:.4f}) would immediately trigger below current price ({current_price:.4f})")
            return
        
        old_price = sl_order["price"]
        sl_order["price"] = new_price
        logger.info(f"Stop loss updated: {symbol} {old_price:.4f} -> {new_price:.4f}")

    # ===================== 多币种调度与绩效跟踪 =====================

    async def _multi_symbol_rebalance_loop(self):
        """多币种重新平衡循环"""
        while True:
            try:
                now = datetime.now()
                if self._last_rebalance_time and \
                   (now - self._last_rebalance_time).total_seconds() < self._rebalance_interval:
                    await asyncio.sleep(60)
                    continue
                
                await self._update_symbol_activity()
                await self._rebalance_grid_resources()
                self._last_rebalance_time = now
            except Exception as e:
                logger.error(f"Multi-symbol rebalance error: {e}")
            await asyncio.sleep(60)

    async def _update_symbol_activity(self):
        """更新币种活跃度"""
        for symbol in self._all_symbols:
            try:
                trade_count = len(self._trade_history.get(symbol, []))
                grid_utilization = 0
                grids = self._grids.get(symbol, [])
                if grids:
                    filled = sum(1 for g in grids if g["filled"])
                    grid_utilization = filled / len(grids)
                
                atr = self._atr_cache.get(symbol, 0)
                ticker = await self.okx_client.get_ticker_async(symbol)
                current_price = float(ticker["last"]) if ticker else 0
                
                volatility = atr / current_price if current_price > 0 and atr > 0 else 0
                
                perf = self._grid_performance.get(symbol, {})
                win_rate = perf.get("win_rate", 0.5)
                
                activity_score = (
                    trade_count * 0.3 +
                    grid_utilization * 0.3 +
                    volatility * 10 * 0.2 +
                    win_rate * 0.2
                )
                
                self._symbol_activity[symbol] = {
                    "trade_count": trade_count,
                    "grid_utilization": grid_utilization,
                    "volatility": volatility,
                    "win_rate": win_rate,
                    "activity_score": activity_score,
                    "last_updated": datetime.now().isoformat()
                }
            except Exception as e:
                logger.debug(f"Error updating activity for {symbol}: {e}")

    async def _rebalance_grid_resources(self):
        """重新平衡网格资源：根据活跃度调整各币种网格密度"""
        if not self._multi_symbol_scheduling or not self._symbol_activity:
            return
        
        try:
            activity_scores = {
                s: a.get("activity_score", 0.5) 
                for s, a in self._symbol_activity.items()
            }
            
            if not activity_scores:
                return
            
            max_score = max(activity_scores.values()) if activity_scores else 1
            if max_score <= 0:
                return
            
            for symbol in self._all_symbols:
                if self._trend_mode.get(symbol, False):
                    continue
                
                score = activity_scores.get(symbol, 0.5)
                normalized_score = score / max_score
                
                current_grids = self._grids.get(symbol, [])
                if not current_grids:
                    continue
                
                current_count = len(current_grids)
                
                if normalized_score > 0.7:
                    target_count = min(self._grid_count_max, current_count + 2)
                elif normalized_score < 0.3:
                    target_count = max(self._grid_count_min, current_count - 2)
                else:
                    target_count = current_count
                
                if target_count != current_count and target_count >= 3:
                    logger.info(f"Rebalancing grids for {symbol}: {current_count} -> {target_count} (score: {normalized_score:.2f})")
                    await self._adjust_grid_count(symbol, target_count)
        except Exception as e:
            logger.error(f"Error rebalancing grid resources: {e}")

    async def _adjust_grid_count(self, symbol: str, target_count: int):
        """调整网格数量"""
        grids = self._grids.get(symbol, [])
        if not grids:
            return
        
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return
        
        current_price = float(ticker["last"])
        
        avg_spacing = self._safe_avg_spacing(grids, current_price)

        # P0-2: 调整网格数量时同样遵守 min_grid_spacing 下限，防止生成过密网格
        avg_spacing = max(self._min_grid_spacing, avg_spacing)
        
        half_grids = target_count // 2
        upper_bound = current_price * (1 + avg_spacing * half_grids)
        lower_bound = current_price * (1 - avg_spacing * half_grids)
        
        lower_bound = max(lower_bound, current_price * 0.5)
        upper_bound = min(upper_bound, current_price * 1.5)
        
        prices = np.linspace(lower_bound, upper_bound, target_count)
        
        new_grids = []
        for i, price in enumerate(prices):
            side = "buy" if (self._long_only or i < half_grids) else "sell"  # P0-1: long_only 禁止 sell 层
            
            density_factor = self._calculate_density_factor(symbol, price, None)
            
            new_grids.append({
                "price": self.okx_client.round_price_to_tick(symbol, price),
                "side": side,
                "filled": False,
                "layer": abs(i - half_grids),
                "quantity": 0,
                "density_factor": density_factor,
                "adjusted_spacing": avg_spacing * density_factor
            })
        
        self._grids[symbol] = new_grids

    def update_grid_performance(self, symbol: str, is_profit: bool, pnl: float):
        """更新网格绩效（由交易引擎回调）"""
        if symbol not in self._grid_performance:
            self._grid_performance[symbol] = {
                "wins": 0,
                "losses": 0,
                "total_pnl": 0,
                "total_trades": 0,
                "total_profit": 0,
                "total_loss": 0,
                "count": 0,
                "pnl_list": [],
                "consecutive_wins": 0,
                "consecutive_losses": 0,
                "peak_equity": 0,
                "max_drawdown": 0,
            }

        perf = self._grid_performance[symbol]
        perf["total_trades"] += 1
        perf["count"] = perf.get("count", 0) + 1
        perf["total_pnl"] = perf.get("total_pnl", 0) + pnl

        # P1: 日内亏损追踪（跨币种累计，用于熔断）
        self._track_daily_pnl(pnl)

        if is_profit:
            perf["wins"] += 1
            perf["total_profit"] = perf.get("total_profit", 0) + pnl
            perf["consecutive_wins"] = perf.get("consecutive_wins", 0) + 1
            perf["consecutive_losses"] = 0
        else:
            perf["losses"] += 1
            perf["total_loss"] = perf.get("total_loss", 0) + pnl
            perf["consecutive_losses"] = perf.get("consecutive_losses", 0) + 1
            perf["consecutive_wins"] = 0

        pnl_list = perf.setdefault("pnl_list", [])
        pnl_list.append(pnl)
        if len(pnl_list) > 200:
            perf["pnl_list"] = pnl_list[-200:]

        perf["peak_equity"] = max(perf.get("peak_equity", 0), perf["total_pnl"])
        drawdown = perf["peak_equity"] - perf["total_pnl"]
        perf["max_drawdown"] = max(perf.get("max_drawdown", 0), drawdown)

        total = perf["wins"] + perf["losses"]
        if total > 0:
            perf["win_rate"] = perf["wins"] / total

        # 企业级：Grid 自适应利用率引擎追踪
        if self._grid_adaptive_engine:
            try:
                self._grid_adaptive_engine.update_trade(symbol, is_profit, pnl)
            except Exception as e:
                logger.debug(f"Grid adaptive update_trade error: {e}")

    def on_close_pnl(self, strategy_name: str, symbol: str, pnl: float):
        """P1: 执行层平仓盈亏回调（由 OrderExecutor 平仓成交后触发）。

        用于日内亏损熔断：仅处理 grid 策略的平仓 PnL，累计到当日盈亏。
        """
        if strategy_name != "grid":
            return
        try:
            self._track_daily_pnl(float(pnl))
        except Exception as e:
            logger.debug(f"Grid on_close_pnl error: {e}")

    def _track_daily_pnl(self, pnl_usdt: float):
        """P1: 累计当日盈亏，跨日自动重置。"""
        today = datetime.now().date()
        if self._daily_pnl_date != today:
            self._daily_pnl = 0.0
            self._daily_pnl_date = today
        self._daily_pnl += pnl_usdt

    def _check_daily_loss_limit(self) -> bool:
        """P1: 当日亏损达到熔断阈值则返回 False（停摆开仓）。"""
        today = datetime.now().date()
        if self._daily_pnl_date != today:
            self._daily_pnl = 0.0
            self._daily_pnl_date = today
        if self._max_daily_loss_usdt > 0 and self._daily_pnl <= -self._max_daily_loss_usdt:
            logger.warning(f"Grid daily loss limit reached ({self._daily_pnl:.2f} <= -{self._max_daily_loss_usdt} USDT), pausing new entries")
            return False
        return True

    def _calculate_kelly_position(self, base_position: float, symbol: str) -> float:
        """P0: 企业级凯利仓位调整（基于逐币种网格绩效叠加在基础仓位之上）。"""
        perf = self._grid_performance.get(symbol, {})
        wins = perf.get("wins", 0)
        losses = perf.get("losses", 0)
        total = wins + losses
        if total < 10:
            return base_position

        avg_win = perf.get("total_profit", 0) / wins if wins > 0 else 0.0
        avg_loss = abs(perf.get("total_loss", 0)) / losses if losses > 0 else 0.0
        if avg_win <= 0 or avg_loss <= 0:
            return base_position

        total_pnl = perf.get("total_pnl", 0)
        peak = perf.get("peak_equity", 0)
        drawdown = max(0.0, peak - total_pnl)
        drawdown_pct = drawdown / max(abs(peak), 0.01)

        # 网格本质是区间策略；趋势模式下切换到趋势 regime 以降低仓位
        regime = "trending_up" if self._trend_mode.get(symbol, False) else "ranging"

        result = self._adaptive_kelly.compute_kelly(
            win_rate=wins / total,
            avg_win=avg_win,
            avg_loss=avg_loss,
            regime=regime,
            drawdown=drawdown_pct,
            consecutive_wins=int(perf.get("consecutive_wins", 0)),
            consecutive_losses=int(perf.get("consecutive_losses", 0)),
            trade_count=total,
        )
        final_kelly = result.get("final_kelly", 0.05)
        if final_kelly <= 0:
            adjusted = base_position * 0.5
        else:
            adjusted = base_position * (1.0 + (final_kelly - 0.05) * 4.0)
            adjusted = max(base_position * 0.3, min(adjusted, base_position * 2.0))
        return adjusted

    def get_stats(self) -> Dict[str, Any]:
        total_trades = sum(len(history) for history in self._trade_history.values())
        trend_mode_count = sum(1 for tm in self._trend_mode.values() if tm)
        
        grid_stats = []
        for symbol in self._all_symbols:
            grids = self._grids.get(symbol, [])
            if grids:
                filled_count = sum(1 for g in grids if g["filled"])
                pending_count = len(self._pending_orders.get(symbol, []))
                has_sl = symbol in self._stop_loss_orders and self._stop_loss_orders[symbol]["status"] == "active"
                activity = self._symbol_activity.get(symbol, {})
                perf = self._grid_performance.get(symbol, {})
                grid_stats.append({
                    "symbol": symbol,
                    "total_grids": len(grids),
                    "filled_grids": filled_count,
                    "martingale": self._martingale_state.get(symbol, 0),
                    "trend_mode": self._trend_mode.get(symbol, False),
                    "atr": self._atr_cache.get(symbol, 0),
                    "pending_orders": pending_count,
                    "active_stop_loss": has_sl,
                    "activity_score": activity.get("activity_score", 0),
                    "win_rate": perf.get("win_rate", 0),
                    "total_trades_symbol": perf.get("total_trades", 0),
                    # 生产级：逐币健康状态
                    "paused": self._coin_paused.get(symbol, False),
                    "health": self._coin_health.get(symbol, {}),
                    "lifecycle_events": len(self._coin_lifecycle.get(symbol, [])),
                })
        
        # 生产级：逐币健康汇总
        coin_health_summary = {}
        for symbol in self._all_symbols:
            health = self._coin_health.get(symbol, {})
            coin_health_summary[symbol] = {
                "status": health.get("status", "unknown"),
                "integrity_score": health.get("grid_integrity_score"),
                "error_count": health.get("error_count", 0),
                "paused": self._coin_paused.get(symbol, False),
                "last_check": health.get("last_check"),
            }
        
        return {
            "strategy": "grid",
            "is_enabled": self._enabled,
            "total_trades": total_trades,
            "trend_mode_count": trend_mode_count,
            "grid_stats": grid_stats,
            "total_pending_orders": sum(len(orders) for orders in self._pending_orders.values()),
            "active_stop_loss_orders": sum(1 for o in self._stop_loss_orders.values() if o["status"] == "active"),
            "dynamic_grid_count": self._dynamic_grid_count,
            "volatility_adaptive_spacing": self._volatility_adaptive_spacing,
            "multi_symbol_scheduling": self._multi_symbol_scheduling,
            "grid_performance_summary": {
                "total_symbols": len(self._grid_performance),
                "total_wins": sum(p.get("wins", 0) for p in self._grid_performance.values()),
                "total_losses": sum(p.get("losses", 0) for p in self._grid_performance.values()),
                "total_pnl": sum(p.get("total_pnl", 0) for p in self._grid_performance.values())
            },
            # 生产级：逐币健康汇总
            "coin_health_summary": coin_health_summary,
            "paused_coins": [s for s, p in self._coin_paused.items() if p],
            "total_lifecycle_events": sum(len(events) for events in self._coin_lifecycle.values()),
            # ghost_close 专项 Phase 4: 观测指标
            "use_fill_driven_position": self._use_fill_driven_position,
            "pending_entry_count": sum(len(v) for v in self._grid_pending_at.values()),
            "fill_callback_hits": self._fill_callback_hits,
            "pending_timeout_cleaned": self._pending_timeout_cleaned,
            "ghost_close_count": self._ghost_close_count,
        }

    # ===================== 生产级各币种网格状态管理 =====================

    def _log_coin_lifecycle_event(self, symbol: str, event_type: str, details: Dict[str, Any]):
        """记录币种生命周期事件"""
        if symbol not in self._coin_lifecycle:
            self._coin_lifecycle[symbol] = []
        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "details": details,
        }
        self._coin_lifecycle[symbol].append(event)
        # 限制事件数量防止内存泄漏
        if len(self._coin_lifecycle[symbol]) > self._coin_max_lifecycle_events:
            self._coin_lifecycle[symbol] = self._coin_lifecycle[symbol][-self._coin_max_lifecycle_events:]

    def get_coin_grid_state(self, symbol: str) -> Optional[Dict[str, Any]]:
        """获取单个币种的完整网格状态快照，用于调试和监控
        
        Returns:
            Dict with keys: grids, martingale, position_side, trend_mode, 
            pending_orders, stop_loss, trailing, performance, health, lifecycle, paused
            如果该币种无网格数据则返回 None
        """
        if symbol not in self._grids and symbol not in self._all_symbols:
            return None
        
        grids = self._grids.get(symbol, [])
        filled = sum(1 for g in grids if g.get("filled") is True)
        pending_grid = sum(1 for g in grids if g.get("filled") == "pending")
        unfilled = sum(1 for g in grids if not g.get("filled"))
        
        sl_order = self._stop_loss_orders.get(symbol)
        sl_info = None
        if sl_order and sl_order.get("status") == "active":
            sl_info = {
                "price": sl_order.get("price"),
                "direction": sl_order.get("direction"),
                "entry_price": sl_order.get("entry_price"),
                "created_at": sl_order.get("created_at").isoformat() if sl_order.get("created_at") else None,
            }
        
        # 计算持仓时长
        entry_time = self._position_entry_time.get(symbol)
        hold_duration = None
        if entry_time:
            hold_duration = time.time() - entry_time
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "grid_summary": {
                "total": len(grids),
                "filled": filled,
                "pending": pending_grid,
                "unfilled": unfilled,
                "fill_rate": round(filled / len(grids), 4) if grids else 0,
            },
            "grids": grids,
            "martingale_layer": self._martingale_state.get(symbol, 0),
            "position_side": self._position_side.get(symbol),
            "trend_mode": self._trend_mode.get(symbol, False),
            "paused": self._coin_paused.get(symbol, False),
            "pending_orders": self._pending_orders.get(symbol, []),
            "stop_loss": sl_info,
            "trailing_state": self._trailing_state.get(symbol),
            "hold_duration_seconds": round(hold_duration, 1) if hold_duration else None,
            "extreme_vol_until": self._extreme_vol_until.get(symbol),
            "grid_retry_count": self._grid_retry_count.get(symbol, {}),
            "grid_rollback_cooldown": self._grid_rollback_cooldown.get(symbol, {}),
            "performance": self._grid_performance.get(symbol, {}),
            "activity": self._symbol_activity.get(symbol, {}),
            "health": self._coin_health.get(symbol, {}),
            "lifecycle_events": self._coin_lifecycle.get(symbol, [])[-10:],  # 最近10条事件
            "lifecycle_event_count": len(self._coin_lifecycle.get(symbol, [])),
        }

    def validate_coin_grid_state(self, symbol: str) -> Dict[str, Any]:
        """校验单个币种网格状态的一致性
        
        检查项：
        1. 网格filled数量与实际持仓方向一致性
        2. pending订单与grid pending标记一致性
        3. 止损单方向与持仓方向一致性
        4. 网格计数与martingale状态一致性
        5. 回滚冷却与重试计数一致性
        
        Returns:
            {valid: bool, errors: [str], warnings: [str], integrity_score: float}
        """
        errors = []
        warnings = []
        grids = self._grids.get(symbol, [])
        
        if not grids:
            return {"valid": True, "errors": [], "warnings": ["no_grids"], "integrity_score": 1.0}
        
        # 1. 检查filled数量
        filled_count = sum(1 for g in grids if g.get("filled") is True)
        pending_count = sum(1 for g in grids if g.get("filled") == "pending")
        total_count = len(grids)
        
        if filled_count > total_count:
            errors.append(f"filled_count({filled_count}) > total_count({total_count})")
        
        # 2. 检查pending一致性
        pending_at = self._grid_pending_at.get(symbol, {})
        grid_pending_set = set()
        for i, g in enumerate(grids):
            if g.get("filled") == "pending":
                grid_pending_set.add(i)
        pending_at_set = set(pending_at.keys())
        
        if grid_pending_set != pending_at_set:
            only_in_grid = grid_pending_set - pending_at_set
            only_in_pending = pending_at_set - grid_pending_set
            # 自动修复不一致：
            # 1. grids标记pending但_pending_at中没有 → 补录到_pending_at
            for idx in only_in_grid:
                if idx < len(grids):
                    self._grid_pending_at.setdefault(symbol, {})[idx] = time.time()
                    logger.info(f"Auto-fixed: added grid[{idx}] to _grid_pending_at for {symbol}")
            # 2. _grid_pending_at中有但grids未标记pending → 清理过期条目（订单已成交/回滚）
            for idx in only_in_pending:
                if symbol in self._grid_pending_at:
                    self._grid_pending_at[symbol].pop(idx, None)
                    logger.info(f"Auto-fixed: removed stale _grid_pending_at[{idx}] for {symbol}")
            if not self._grid_pending_at.get(symbol):
                self._grid_pending_at.pop(symbol, None)
            # 修复后重新检查，仅当仍有不一致时才报警
            pending_at2 = self._grid_pending_at.get(symbol, {})
            pending_at_set2 = set(pending_at2.keys())
            if grid_pending_set != pending_at_set2:
                only_in_grid2 = grid_pending_set - pending_at_set2
                only_in_pending2 = pending_at_set2 - grid_pending_set
                if only_in_grid2:
                    warnings.append(f"Grids marked pending but not in _grid_pending_at (unfixable): {only_in_grid2}")
                if only_in_pending2:
                    warnings.append(f"_grid_pending_at has entries not marked pending in grids (unfixable): {only_in_pending2}")
        
        # 3. 止损单一致性
        sl_order = self._stop_loss_orders.get(symbol)
        if sl_order and sl_order.get("status") == "active":
            sl_direction = sl_order.get("direction")
            pos_side = self._position_side.get(symbol)
            if sl_direction and pos_side:
                if sl_direction == "long" and pos_side != "buy":
                    warnings.append(f"SL direction 'long' but position_side is '{pos_side}'")
                elif sl_direction == "short" and pos_side != "sell":
                    warnings.append(f"SL direction 'short' but position_side is '{pos_side}'")
        
        # 4. 回滚冷却与重试计数一致性
        retry_counts = self._grid_retry_count.get(symbol, {})
        cooldowns = self._grid_rollback_cooldown.get(symbol, {})
        # 有重试计数但无冷却记录的检查
        for idx, count in retry_counts.items():
            if count >= 3 and idx not in cooldowns:
                warnings.append(f"Grid[{idx}] retry={count} but no cooldown record")
        
        # 5. 趋势模式一致性
        if self._trend_mode.get(symbol, False):
            if self._trailing_state.get(symbol):
                warnings.append("Trend mode active but trailing_state exists")
            if self._position_entry_time.get(symbol):
                warnings.append("Trend mode active but position_entry_time exists")
        
        # 计算完整性分数
        base_score = 1.0
        if errors:
            base_score = max(0.1, 1.0 - len(errors) * 0.3)
        if warnings:
            base_score = max(0.1, base_score - len(warnings) * 0.1)
        
        valid = len(errors) == 0
        return {
            "valid": valid,
            "errors": errors,
            "warnings": warnings,
            "integrity_score": round(base_score, 4),
        }

    def reset_coin_grid_state(self, symbol: str) -> Dict[str, Any]:
        """重置单个币种的网格状态（不影响其他币种）
        
        清理该币种的所有状态：网格、持仓方向、马丁格尔、待处理订单、止损单、
        移动止损、入场时间、重试计数、冷却、pending标记、绩效、活跃度、健康状态、生命周期事件
        
        Returns:
            {success: bool, symbol: str, cleared_fields: [str]}
        """
        cleared = []
        
        # 清理网格
        if symbol in self._grids:
            del self._grids[symbol]
            cleared.append("grids")
        
        # 清理马丁格尔状态
        if symbol in self._martingale_state:
            del self._martingale_state[symbol]
            cleared.append("martingale_state")
        
        # 清理持仓方向
        if symbol in self._position_side:
            del self._position_side[symbol]
            cleared.append("position_side")
        
        # 清理趋势模式
        if symbol in self._trend_mode:
            del self._trend_mode[symbol]
            cleared.append("trend_mode")
        
        # 清理待处理订单
        if symbol in self._pending_orders:
            del self._pending_orders[symbol]
            cleared.append("pending_orders")
        
        # 清理止损单
        if symbol in self._stop_loss_orders:
            del self._stop_loss_orders[symbol]
            cleared.append("stop_loss_orders")
        
        # 清理移动止损
        if symbol in self._trailing_state:
            del self._trailing_state[symbol]
            cleared.append("trailing_state")
        
        # 清理入场时间
        if symbol in self._position_entry_time:
            del self._position_entry_time[symbol]
            cleared.append("position_entry_time")
        
        # 清理极端波动暂停
        if symbol in self._extreme_vol_until:
            del self._extreme_vol_until[symbol]
            cleared.append("extreme_vol_until")
        
        # 清理重试计数
        if symbol in self._grid_retry_count:
            del self._grid_retry_count[symbol]
            cleared.append("grid_retry_count")
        if symbol in self._grid_retry_first_ts:
            del self._grid_retry_first_ts[symbol]
            cleared.append("grid_retry_first_ts")
        
        # 清理回滚冷却
        if symbol in self._grid_rollback_cooldown:
            del self._grid_rollback_cooldown[symbol]
            cleared.append("grid_rollback_cooldown")
        
        # 清理pending标记
        if symbol in self._grid_pending_at:
            del self._grid_pending_at[symbol]
            cleared.append("grid_pending_at")
        
        # 清理绩效
        if symbol in self._grid_performance:
            del self._grid_performance[symbol]
            cleared.append("grid_performance")
        
        # 清理活跃度
        if symbol in self._symbol_activity:
            del self._symbol_activity[symbol]
            cleared.append("symbol_activity")
        
        # 清理健康状态
        if symbol in self._coin_health:
            del self._coin_health[symbol]
            cleared.append("coin_health")
        
        # 清理暂停标记
        if symbol in self._coin_paused:
            del self._coin_paused[symbol]
            cleared.append("coin_paused")
        
        # 清理生命周期事件
        if symbol in self._coin_lifecycle:
            del self._coin_lifecycle[symbol]
            cleared.append("coin_lifecycle")
        
        # 清理ATR缓存
        if symbol in self._atr_cache:
            del self._atr_cache[symbol]
            cleared.append("atr_cache")
        
        # 清理趋势偏向缓存
        if symbol in self._trend_bias_cache:
            del self._trend_bias_cache[symbol]
            cleared.append("trend_bias_cache")
        
        # 清理REST tick缓存
        if symbol in self._tick_rest_cache:
            del self._tick_rest_cache[symbol]
            cleared.append("tick_rest_cache")
        
        # 清除状态快照
        if symbol in self._coin_last_state_snapshot:
            del self._coin_last_state_snapshot[symbol]
            cleared.append("coin_last_state_snapshot")
        
        logger.info(f"Grid state reset for {symbol}: cleared {len(cleared)} fields: {cleared}")
        
        return {
            "success": True,
            "symbol": symbol,
            "cleared_fields": cleared,
            "cleared_count": len(cleared),
        }

    def pause_coin(self, symbol: str, reason: str = "manual") -> Dict[str, Any]:
        """暂停单个币种的网格交易（不影响其他币种）
        
        暂停后该币种将：
        - 不触发新网格订单
        - 不调整止损单
        - 现有持仓和止损单保持不变
        
        Returns:
            {success: bool, symbol: str, was_already_paused: bool}
        """
        was_paused = self._coin_paused.get(symbol, False)
        self._coin_paused[symbol] = True
        
        self._log_coin_lifecycle_event(symbol, "coin_paused", {"reason": reason})
        logger.info(f"Coin paused: {symbol}, reason: {reason}, was_already_paused: {was_paused}")
        
        return {
            "success": True,
            "symbol": symbol,
            "was_already_paused": was_paused,
        }

    def resume_coin(self, symbol: str, rebuild_grid: bool = True) -> Dict[str, Any]:
        """恢复单个币种的网格交易
        
        Args:
            symbol: 币种符号
            rebuild_grid: 是否在恢复后重建网格（默认True）
        
        Returns:
            {success: bool, symbol: str, was_paused: bool}
        """
        was_paused = self._coin_paused.pop(symbol, False)
        
        if not was_paused:
            return {"success": True, "symbol": symbol, "was_paused": False, "note": "coin was not paused"}
        
        self._log_coin_lifecycle_event(symbol, "coin_resumed", {"rebuild_grid": rebuild_grid})
        logger.info(f"Coin resumed: {symbol}, rebuild_grid: {rebuild_grid}")
        
        return {
            "success": True,
            "symbol": symbol,
            "was_paused": True,
            "rebuild_grid": rebuild_grid,
        }

    def get_coin_health_report(self, symbol: str) -> Dict[str, Any]:
        """获取单个币种的健康报告
        
        Returns:
            {symbol, status, integrity_score, error_count, warnings, 
             lifecycle_summary, grid_summary, recommendations}
        """
        validation = self.validate_coin_grid_state(symbol)
        health = self._coin_health.get(symbol, {})
        lifecycle = self._coin_lifecycle.get(symbol, [])
        
        # 生命周期事件类型统计
        lifecycle_summary = {}
        for event in lifecycle[-50:]:
            etype = event.get("event_type", "unknown")
            lifecycle_summary[etype] = lifecycle_summary.get(etype, 0) + 1
        
        # 网格摘要
        grids = self._grids.get(symbol, [])
        grid_summary = {
            "total": len(grids),
            "filled": sum(1 for g in grids if g.get("filled") is True),
            "pending": sum(1 for g in grids if g.get("filled") == "pending"),
            "unfilled": sum(1 for g in grids if not g.get("filled")),
        }
        
        # 建议
        recommendations = []
        status = health.get("status", "unknown")
        
        if validation["errors"]:
            recommendations.append("State inconsistency detected: consider reset_coin_grid_state()")
        if status == "error":
            recommendations.append("Coin in error state: check logs and consider reset")
        if status == "degraded" and validation["integrity_score"] < 0.7:
            recommendations.append("Low integrity score: validation recommended")
        if self._coin_paused.get(symbol, False):
            recommendations.append("Coin is paused: call resume_coin() to resume trading")
        if grid_summary["filled"] > 0 and not self._stop_loss_orders.get(symbol):
            recommendations.append("Active positions without stop loss: risk exposure")
        if grid_summary["pending"] > 2:
            recommendations.append(f"High pending grid count ({grid_summary['pending']}): check order status")
        
        if not recommendations:
            recommendations.append("No issues detected")
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "status": status,
            "integrity_score": validation["integrity_score"],
            "error_count": health.get("error_count", 0),
            "warnings": validation["warnings"],
            "errors": validation["errors"],
            "paused": self._coin_paused.get(symbol, False),
            "trend_mode": self._trend_mode.get(symbol, False),
            "lifecycle_summary": lifecycle_summary,
            "grid_summary": grid_summary,
            "recommendations": recommendations,
        }

    def diff_coin_grid_state(self, symbol: str) -> Dict[str, Any]:
        """对比当前状态与上次快照的差异
        
        Returns:
            {symbol, has_changes, changes: [{field, old_value, new_value}], 
             current_snapshot, previous_snapshot}
        """
        current = self.get_coin_grid_state(symbol)
        if not current:
            return {"symbol": symbol, "has_changes": False, "changes": [], "error": "no_grid_data"}
        
        previous = self._coin_last_state_snapshot.get(symbol)
        self._coin_last_state_snapshot[symbol] = current
        
        if not previous:
            return {
                "symbol": symbol,
                "has_changes": True,
                "changes": [{"field": "_initial", "old_value": None, "new_value": "first_snapshot"}],
                "current_snapshot_timestamp": current["timestamp"],
                "previous_snapshot_timestamp": None,
            }
        
        changes = []
        # 对比关键字段
        compare_fields = [
            ("grid_summary.total", lambda s: s.get("grid_summary", {}).get("total")),
            ("grid_summary.filled", lambda s: s.get("grid_summary", {}).get("filled")),
            ("grid_summary.pending", lambda s: s.get("grid_summary", {}).get("pending")),
            ("martingale_layer", lambda s: s.get("martingale_layer")),
            ("position_side", lambda s: s.get("position_side")),
            ("trend_mode", lambda s: s.get("trend_mode")),
            ("paused", lambda s: s.get("paused")),
            ("hold_duration_seconds", lambda s: s.get("hold_duration_seconds")),
            ("performance.win_rate", lambda s: s.get("performance", {}).get("win_rate")),
            ("performance.total_pnl", lambda s: s.get("performance", {}).get("total_pnl")),
            ("health.status", lambda s: s.get("health", {}).get("status")),
        ]
        
        for field_name, extractor in compare_fields:
            old_val = extractor(previous)
            new_val = extractor(current)
            if old_val != new_val:
                changes.append({
                    "field": field_name,
                    "old_value": old_val,
                    "new_value": new_val,
                })
        
        return {
            "symbol": symbol,
            "has_changes": len(changes) > 0,
            "changes": changes,
            "current_snapshot_timestamp": current["timestamp"],
            "previous_snapshot_timestamp": previous["timestamp"],
        }

    # ===================== 信号质量增强 =====================

    async def _calculate_grid_signal_quality(self, symbol: str, side: str, price: float, grid: Dict[str, Any]) -> float:
        """计算网格信号质量评分 0-1
        
        基于以下维度：
        1. 成交量确认 (0-0.25): 当前量 > 24h均值 * 0.5
        2. 价格动量对齐 (0-0.25): 价格在网格方向移动
        3. 价差检查 (0-0.25): spread < 0.1% of price
        4. 市场状态兼容 (0-0.15): 不在极端波动中
        5. 近期成交率 (0-0.10): 最近N个网格实际成交比例
        """
        try:
            score = 0.0

            # 预取 ticker（成交量 + 价差共用，避免重复 REST 调用）
            ticker = None
            try:
                ticker = await self.okx_client.get_ticker_async(symbol)
            except Exception:
                pass

            # 1. 成交量确认 (0-0.25)
            if hasattr(self, '_trade_history') and symbol in self._trade_history:
                vol_history = self._trade_history.get(symbol, [])
                if len(vol_history) >= 5:
                    recent_vols = [t.get("volume", 0) for t in vol_history[-5:]]
                    avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 0
                    if avg_vol > 0:
                        if ticker:
                            vol_24h = float(ticker.get("vol24h", 0))
                            if vol_24h > avg_vol * 0.5:
                                vol_ratio = min(1.0, vol_24h / (avg_vol * 2))
                                score += 0.25 * vol_ratio
                        else:
                            score += 0.125
                    else:
                        score += 0.125
                else:
                    score += 0.125  # 历史数据不足，给中等分数
            else:
                score += 0.125

            # 2. 价格动量对齐 (0-0.25)
            last_price = self._last_tick_price.get(symbol, 0)
            if last_price > 0 and price > 0:
                momentum = (price - last_price) / last_price if last_price > 0 else 0
                if side == "buy" and momentum < 0:
                    # 价格下跌，买入网格方向一致
                    momentum_score = min(0.25, abs(momentum) * 50)
                    score += momentum_score
                elif side == "sell" and momentum > 0:
                    # 价格上涨，卖出网格方向一致
                    momentum_score = min(0.25, momentum * 50)
                    score += momentum_score
                elif abs(momentum) < 0.0001:
                    score += 0.15  # 价格平稳，中性
                else:
                    score += 0.05  # 价格逆向，低分
            else:
                score += 0.125

            # 3. 价差检查 (0-0.25) — 复用预取 ticker
            if ticker:
                bid = float(ticker.get("bidPx", 0))
                ask = float(ticker.get("askPx", 0))
                if bid > 0 and ask > 0 and price > 0:
                    spread = (ask - bid) / price
                    if spread < 0.001:  # < 0.1%
                        score += 0.25
                    elif spread < 0.002:  # < 0.2%
                        score += 0.15
                    elif spread < 0.005:  # < 0.5%
                        score += 0.08
                    else:
                        score += 0.02
                else:
                    score += 0.125
            else:
                score += 0.125

            # 4. 市场状态兼容 (0-0.15)
            if hasattr(self, '_extreme_vol_until'):
                until = self._extreme_vol_until.get(symbol)
                if until and time.time() < until:
                    score += 0.0  # 极端波动中，零分
                else:
                    score += 0.15
            else:
                score += 0.15

            # 5. 近期成交率 (0-0.10)
            grids = self._grids.get(symbol, [])
            if grids:
                total = len(grids)
                filled = sum(1 for g in grids if g.get("filled") is True or g.get("filled") == "pending")
                fill_rate = filled / total if total > 0 else 0
                score += 0.10 * fill_rate
            else:
                score += 0.05

            # 6. 趋势确认（统一 ADX/DI 方向对齐，0 上下浮动 ±0.10）
            # 复用 MarketRegimeEngine.get_symbol_trend_detail 的统一口径（与框架层
            # TrendConfirmationFilter 同源），替代/补充策略内自算 EMA20/50 的单因子趋势判断：
            # 顺势均值回归（上涨趋势买回调 / 下跌趋势卖反弹）加分，逆势（接飞刀）减分。
            # 引擎未注入或明细缺失时保持中性（0），不改变原有评分行为。
            detail = self._get_unified_trend_detail(symbol)
            if detail:
                try:
                    direction = float(detail.get("direction", 0.0))
                    adx_strength = float(detail.get("adx_strength", 0.0))
                    if math.isfinite(direction) and math.isfinite(adx_strength):
                        aligned = (side == "buy" and direction > 0) or (side == "sell" and direction < 0)
                        opposing = (side == "buy" and direction < 0) or (side == "sell" and direction > 0)
                        weight = max(0.0, min(1.0, adx_strength))
                        if aligned:
                            score += 0.10 * weight
                        elif opposing:
                            score -= 0.10 * weight
                except (TypeError, ValueError):
                    pass

            return min(1.0, max(0.0, score))
        except Exception as e:
            logger.debug(f"Grid signal quality calculation error for {symbol}: {e}")
            return 0.5  # 出错时返回中性分数，不阻塞

    def _validate_grid_signal(self, symbol: str, side: str, price: float, grid: Dict[str, Any]) -> tuple:
        """验证网格信号是否合理
        
        Returns:
            (is_valid: bool, reason: str)
        检查项：
        1. 网格价格在合理范围内（不过远偏离当前价格）
        2. 网格层近期未被触发（冷却检查）
        3. 网格方向与当前市场趋势兼容
        """
        try:
            # 1. 网格价格合理范围检查
            atr = self._atr_cache.get(symbol, 0)
            if atr > 0 and price > 0:
                grid_price = grid.get("price", 0)
                if grid_price > 0:
                    distance = abs(grid_price - price) / price
                    max_reasonable = atr * 6 / price  # 6倍ATR为合理范围上限
                    if distance > max_reasonable:
                        # P10: 跟踪连续"too far"拒绝，超过阈值自动重建网格
                        stale_count = self._grid_stale_count.get(symbol, 0) + 1
                        self._grid_stale_count[symbol] = stale_count
                        
                        # P28: 极端偏离(>10%)立即重建，不等待stale_count累积
                        # 防止网格价格严重偏离导致无效交易（如ADA 13%偏离仍触发订单）
                        EXTREME_DEVIATION_THRESHOLD = 0.10  # 10%偏离立即重建
                        if distance > EXTREME_DEVIATION_THRESHOLD:
                            # P6: 添加冷却期检查，防止频繁重建
                            last_rebuild = self._grid_rebuild_cooldown.get(symbol, 0)
                            if time.time() - last_rebuild < self._grid_rebuild_cooldown_seconds:
                                logger.debug(
                                    f"P6 grid rebuild cooldown: {symbol} "
                                    f"(last_rebuild={time.time() - last_rebuild:.0f}s ago, "
                                    f"cooldown={self._grid_rebuild_cooldown_seconds}s)"
                                )
                            else:
                                logger.warning(
                                    f"P19 grid extreme deviation: {symbol} grid_price={grid_price:.4f} "
                                    f"vs current={price:.4f} (distance={distance:.4%}), "
                                    f"triggering immediate rebuild"
                                )
                                self._grid_stale_count[symbol] = 0
                                asyncio.create_task(self._rebuild_stale_grid(symbol, current_price=price))
                        elif stale_count >= 5:
                            logger.warning(
                                f"P28 grid auto-reset: {symbol} grid stale {stale_count} times "
                                f"(grid_price={grid_price:.4f}, current={price:.4f}, distance={distance:.4%}), "
                                f"rebuilding grids..."
                            )
                            self._grid_stale_count[symbol] = 0
                            # 异步重建网格（不阻塞当前tick处理）
                            # P14: 传入当前价格避免使用过期缓存ticker
                            asyncio.create_task(self._rebuild_stale_grid(symbol, current_price=price))
                        return False, f"grid price {grid_price:.4f} too far from current {price:.4f} (distance={distance:.4%} > max={max_reasonable:.4%})"
                    else:
                        # 网格价格在合理范围内，重置stale计数
                        if symbol in self._grid_stale_count:
                            self._grid_stale_count[symbol] = 0

            # 2. 网格层冷却检查
            grids = self._grids.get(symbol, [])
            if grids:
                grid_index = next((i for i, g in enumerate(grids) if g is grid), None)
                if grid_index is not None:
                    # 检查回滚冷却
                    cooldown = self._grid_rollback_cooldown.get(symbol, {}).get(grid_index)
                    if cooldown and time.time() < cooldown:
                        return False, f"grid layer {grid_index} in rollback cooldown"
                    # 检查重试次数
                    retry_count = self._grid_retry_count.get(symbol, {}).get(grid_index, 0)
                    if retry_count >= 6:
                        return False, f"grid layer {grid_index} exceeded max retries ({retry_count})"

            # 3. 网格方向与市场趋势兼容
            bias = self._trend_bias_cache.get(symbol)
            if bias and bias.get("strength", 0) > 0.015:
                trend_dir = bias.get("direction", 0)
                if trend_dir == 1 and side == "sell":
                    return False, f"strong uptrend, sell grid incompatible"
                elif trend_dir == -1 and side == "buy":
                    return False, f"strong downtrend, buy grid incompatible"

            return True, "ok"
        except Exception as e:
            logger.debug(f"Grid signal validation error for {symbol}: {e}")
            return False, f"validation_error: {e}"  # 出错时拒绝（fail-closed）

    async def _check_coin_health(self, symbol: str):
        """执行单币种健康检查，更新健康状态"""
        now = datetime.now()
        
        # 初始化健康状态
        if symbol not in self._coin_health:
            self._coin_health[symbol] = {
                "status": "healthy",
                "last_check": None,
                "error_count": 0,
                "warnings": [],
                "grid_integrity_score": 1.0,
                "last_status_change": None,
            }
        
        health = self._coin_health[symbol]
        health["last_check"] = now.isoformat()
        warnings = []
        
        try:
            # 1. 状态一致性校验
            validation = self.validate_coin_grid_state(symbol)
            health["grid_integrity_score"] = validation["integrity_score"]
            
            if validation["errors"]:
                warnings.extend(validation["errors"])
            if validation["warnings"]:
                warnings.extend(validation["warnings"])
            
            # 2. 检查网格是否存在
            grids = self._grids.get(symbol, [])
            if not grids and symbol not in self._trend_mode:
                warnings.append("No grids and not in trend mode")
            
            # 3. 检查并自动清理pending超时的网格
            pending_at = self._grid_pending_at.get(symbol, {})
            now_ts = time.time()
            stale_pending = []
            for idx, ts in list(pending_at.items()):
                if now_ts - ts > 180:  # 超过3分钟
                    stale_pending.append(idx)
            if stale_pending:
                for idx in stale_pending:
                    pending_at.pop(idx, None)
                    # 同时回滚对应grid的filled状态
                    if idx < len(grids) and grids[idx].get("filled") == "pending":
                        grids[idx]["filled"] = False
                if not pending_at:
                    self._grid_pending_at.pop(symbol, None)
                logger.info(f"Auto-cleaned {len(stale_pending)} stale pending grids for {symbol}: {stale_pending}")
            
            # 4. 检查是否有持仓但无止损
            has_position = False
            for g in grids:
                if g.get("filled") is True:
                    has_position = True
                    break
            if has_position and symbol not in self._stop_loss_orders:
                warnings.append("Active positions without stop loss order")
            
            # 5. 检查止损价是否合理
            sl_order = self._stop_loss_orders.get(symbol)
            if sl_order and sl_order.get("status") == "active":
                try:
                    ticker = await self.okx_client.get_ticker_async(symbol)
                    if ticker:
                        current_price = float(ticker["last"])
                        sl_price = sl_order.get("price", 0)
                        direction = sl_order.get("direction", "")
                        if direction == "long" and sl_price >= current_price:
                            warnings.append(f"Long SL price ({sl_price}) >= current ({current_price}): would trigger immediately")
                        elif direction == "short" and sl_price <= current_price:
                            warnings.append(f"Short SL price ({sl_price}) <= current ({current_price}): would trigger immediately")
                except Exception:
                    pass
            
            # 更新健康状态
            health["warnings"] = warnings[-10:]  # 只保留最近10条警告
            
            if validation["errors"]:
                health["error_count"] = health.get("error_count", 0) + 1
                new_status = "error"
            elif validation["integrity_score"] < self._coin_health_degraded_score:
                new_status = "degraded"
            elif warnings:
                new_status = "degraded"
            else:
                new_status = "healthy"
                # 健康时重置错误计数
                health["error_count"] = 0
            
            # 状态变更记录
            old_status = health.get("status", "healthy")
            if new_status != old_status:
                health["status"] = new_status
                health["last_status_change"] = now.isoformat()
                self._log_coin_lifecycle_event(symbol, "health_status_change", {
                    "old_status": old_status,
                    "new_status": new_status,
                    "warnings": warnings,
                })
                logger.warning(f"Grid health status change for {symbol}: {old_status} -> {new_status}, warnings: {warnings}")
            else:
                health["status"] = new_status
            
            # 自动暂停：连续错误超过阈值
            if self._coin_auto_pause_on_error and health["error_count"] >= self._coin_health_max_errors:
                if not self._coin_paused.get(symbol, False):
                    self.pause_coin(symbol, f"auto_pause: error_count={health['error_count']} >= {self._coin_health_max_errors}")
                    logger.error(f"Auto-paused {symbol}: error_count={health['error_count']} >= threshold={self._coin_health_max_errors}")
        
        except Exception as e:
            logger.error(f"Health check failed for {symbol}: {e}")
            health["error_count"] = health.get("error_count", 0) + 1
            health["status"] = "error"
            health["warnings"] = [f"health_check_exception: {str(e)}"]

    async def _coin_health_check_loop(self):
        """逐币健康检查循环"""
        await asyncio.sleep(30)  # 启动后延迟30秒再开始检查
        
        while True:
            try:
                for symbol in self._all_symbols:
                    await self._check_coin_health(symbol)
            except Exception as e:
                logger.error(f"Coin health check loop error: {e}")
            await asyncio.sleep(self._coin_health_check_interval)