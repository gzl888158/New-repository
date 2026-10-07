"""现货网格交易策略：在设定价格区间内布设网格低买高卖，并根据波动率动态调整网格。"""
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from core.models import Signal, TickData, Position
from configs.settings import get_currency_tier
from utils.helpers import get_price_precision
from utils.state_persistence import PersistentStrategy


class SpotGridStrategy(PersistentStrategy):
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        super().__init__()
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache

        self._enabled = config["strategies"].get("spot_grid", {}).get("enabled", False)
        self._grid_count_min = config["strategies"].get("spot_grid", {}).get("grid_count_min", 5)
        self._grid_count_max = config["strategies"].get("spot_grid", {}).get("grid_count_max", 10)
        self._dynamic_adjust_interval = config["strategies"].get("spot_grid", {}).get("dynamic_adjust_interval", 300)
        self._volatility_threshold = config["strategies"].get("spot_grid", {}).get("volatility_threshold", 0.003)
        self._atr_period = config["strategies"].get("spot_grid", {}).get("atr_period", 14)
        self._atr_multiplier = config["strategies"].get("spot_grid", {}).get("atr_multiplier", 0.3)
        self._volume_profile_period = config["strategies"].get("spot_grid", {}).get("volume_profile_period", 30)

        self._grids: Dict[str, List[Dict[str, Any]]] = {}
        self._last_adjust_time: Dict[str, datetime] = {}
        self._last_tick_price: Dict[str, float] = {}
        self._trade_history: Dict[str, List[Dict[str, Any]]] = {}
        self._current_holdings: Dict[str, float] = {}

        self._tick_rest_cache: Dict[str, tuple] = {}
        self._tick_rest_interval = 1.0

        self._order_check_interval = 60

        self._adaptive_controller = None

        self._dynamic_grid_count = config["strategies"].get("spot_grid", {}).get("dynamic_grid_count", True)
        self._volatility_adaptive_spacing = config["strategies"].get("spot_grid", {}).get("volatility_adaptive_spacing", True)
        self._multi_symbol_scheduling = config["strategies"].get("spot_grid", {}).get("multi_symbol_scheduling", True)
        self._min_grid_spacing = config["strategies"].get("spot_grid", {}).get("min_grid_spacing", 0.002)
        self._max_grid_spacing = config["strategies"].get("spot_grid", {}).get("max_grid_spacing", 0.03)

        self._grid_performance: Dict[str, Dict[str, Any]] = {}
        self._symbol_activity: Dict[str, Dict[str, Any]] = {}
        self._last_rebalance_time: Optional[datetime] = None
        self._rebalance_interval = 3600

        self._all_symbols = []
        for tier in ["tier1", "tier2", "tier3"]:
            for base in config["currencies"][f"{tier}_symbols"]:
                self._all_symbols.append(f"{base}-USDT")

        self._signal_callback = None

        # === 优化：持仓管理 ===
        self._active_positions: Dict[str, Dict[str, Any]] = {}  # symbol -> {avg_price, quantity, unrealized_pnl}
        self._position_check_interval = 30  # 持仓检查间隔（秒）
        self._take_profit_pct = config["strategies"].get("spot_grid", {}).get("take_profit_pct", 0.015)
        self._stop_loss_pct = config["strategies"].get("spot_grid", {}).get("stop_loss_pct", 0.08)

        # === 优化：日志节流 ===
        self._last_log_time: Dict[str, float] = {}  # symbol -> timestamp
        self._log_throttle_interval = 60  # 同一symbol日志间隔（秒）

        # === 优化：趋势过滤 ===
        self._trend_filter_enabled = config["strategies"].get("spot_grid", {}).get("trend_filter_enabled", True)
        self._ema_fast_period = config["strategies"].get("spot_grid", {}).get("ema_fast_period", 12)
        self._ema_slow_period = config["strategies"].get("spot_grid", {}).get("ema_slow_period", 26)
        self._ema_cache: Dict[str, Dict[str, float]] = {}  # symbol -> {fast, slow, trend}

        # === 优化：网格重置 ===
        self._grid_reset_enabled = config["strategies"].get("spot_grid", {}).get("grid_reset_enabled", True)
        self._grid_reset_delay = config["strategies"].get("spot_grid", {}).get("grid_reset_delay", 60)  # 成交后60秒重置

        # === 优化：成交量过滤 ===
        self._volume_filter_enabled = config["strategies"].get("spot_grid", {}).get("volume_filter_enabled", True)
        self._volume_threshold_ratio = config["strategies"].get("spot_grid", {}).get("volume_threshold_ratio", 0.3)  # 成交量低于均值的30%则过滤

        # === P0-1: 入场价对账跟踪 ===
        self._reconciled_symbols: set = set()  # 已对账过入场价的 symbol，避免每轮都调 fills API
        self._entry_price_cache: Dict[str, float] = {}  # symbol -> 本地缓存的真实入场价

        # === P0-4: 状态持久化 ===
        self.init_state_persistence("spot_grid", redis_cache)

        self._capital_cache_value = 0.0
        self._capital_cache_ts = 0.0
        self._capital_cache_ttl = 30.0

    def set_adaptive_controller(self, controller):
        self._adaptive_controller = controller

    def set_coordinator(self, coordinator):
        """注入StrategyCoordinator实例"""
        self._coordinator = coordinator

    async def update_config(self, updates: Dict[str, Any]):
        """运行时热更新策略配置（不重启策略）"""
        strategy_cfg = self.config.get("strategies", {}).get("spot_grid", {})
        strategy_cfg.update(updates)
        self.config.setdefault("strategies", {})["spot_grid"] = strategy_cfg

        attr_map = {
            "grid_count_max": "_grid_count_max",
            "grid_count_min": "_grid_count_min",
            "min_grid_spacing": "_min_grid_spacing",
            "max_grid_spacing": "_max_grid_spacing",
            "atr_multiplier": "_atr_multiplier",
            "stop_loss_pct": "_stop_loss_pct",
            "take_profit_pct": "_take_profit_pct",
            "grid_reset_enabled": "_grid_reset_enabled",
            "grid_reset_delay": "_grid_reset_delay",
        }
        for cfg_key, attr_name in attr_map.items():
            if cfg_key in updates:
                setattr(self, attr_name, updates[cfg_key])
                logger.info(f"SpotGrid config hot-updated: {attr_name}={updates[cfg_key]}")

    def set_signal_callback(self, callback):
        self._signal_callback = callback

    def _get_allocation(self) -> float:
        if self._adaptive_controller:
            try:
                return self._adaptive_controller.get_allocation("spot_grid")
            except Exception as e:
                # fail-closed: 资金分配查询失败时返回 0，拒绝开仓
                logger.warning(f"SpotGrid 资金分配查询失败，返回 0（fail-closed）: {e}")
                return 0.0
        return self.config["trading"].get("spot_grid_allocation", 0.20)

    def _get_effective_capital(self) -> float:
        """获取有效资金：优先使用实际账户权益，失败时 fail-closed 返回 0。30s TTL 缓存。"""
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
            logger.warning(f"[spot_grid] get_account_info failed: {e}")
        # fail-closed: 账户权益查询失败时返回 0，避免用静态 total_capital 兜底导致仓位失真
        logger.warning("[spot_grid] 账户权益查询失败，返回 0（fail-closed）")
        return 0.0

    def apply_param_update(self, params: Dict[str, Any]):
        applied = []
        if "grid_count_min" in params:
            self._grid_count_min = params["grid_count_min"]
            applied.append("grid_count_min")
        if "grid_count_max" in params:
            self._grid_count_max = params["grid_count_max"]
            applied.append("grid_count_max")
        if "volatility_threshold" in params:
            self._volatility_threshold = params["volatility_threshold"]
            applied.append("volatility_threshold")
        if "atr_period" in params:
            self._atr_period = params["atr_period"]
            applied.append("atr_period")
        if "atr_multiplier" in params:
            self._atr_multiplier = params["atr_multiplier"]
            applied.append("atr_multiplier")
        if "take_profit_pct" in params:
            self._take_profit_pct = params["take_profit_pct"]
            applied.append("take_profit_pct")
        if "stop_loss_pct" in params:
            self._stop_loss_pct = params["stop_loss_pct"]
            applied.append("stop_loss_pct")
        if applied:
            logger.info(f"Spot Grid strategy params hot-updated: {applied}")
        return applied

    async def start(self):
        if not self._enabled:
            logger.info("Spot Grid strategy is disabled")
            return

        logger.info("Starting Spot Grid Strategy")
        # P0-4: 启动时恢复持久化状态（_grids / _active_positions / _last_adjust_time / _grid_performance）
        await self.load_state_async()
        await self._initialize_grids()

        asyncio.create_task(self._monitor_ticks())
        asyncio.create_task(self._dynamic_adjust_loop())
        asyncio.create_task(self._update_indicators_loop())
        asyncio.create_task(self._order_check_loop())
        asyncio.create_task(self._multi_symbol_rebalance_loop())
        # 优化：添加持仓监控循环
        asyncio.create_task(self._position_monitor_loop())
        asyncio.create_task(self._update_ema_loop())
        # P0-4: 周期保存策略状态
        asyncio.create_task(self.periodic_save_loop())

    async def _initialize_grids(self):
        for symbol in self._all_symbols:
            await self._build_grid(symbol)

    async def _update_indicators_loop(self):
        while True:
            try:
                for symbol in self._all_symbols:
                    await self._update_atr(symbol)
                    await self._update_volume_profile(symbol)
                await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SpotGrid _update_indicators_loop error: {e}")
                await asyncio.sleep(60)

    async def _update_ema_loop(self):
        """优化：更新EMA趋势指标"""
        while True:
            try:
                for symbol in self._all_symbols:
                    await self._update_ema(symbol)
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SpotGrid _update_ema_loop error: {e}")
                await asyncio.sleep(60)

    async def _update_ema(self, symbol: str):
        """计算EMA趋势"""
        if not self._trend_filter_enabled:
            return

        try:
            klines = await self.okx_client.get_kline_async(symbol, "1H", limit=self._ema_slow_period + 5)
            if len(klines) < self._ema_slow_period:
                return

            closes = np.array([float(kline[4]) for kline in klines])
            if np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
                return

            # 计算EMA
            fast_ema = self._calculate_ema(closes, self._ema_fast_period)
            slow_ema = self._calculate_ema(closes, self._ema_slow_period)

            if fast_ema is None or slow_ema is None:
                return

            # 判断趋势：fast > slow = 上涨趋势，fast < slow = 下跌趋势
            trend = "up" if fast_ema > slow_ema else "down" if fast_ema < slow_ema else "neutral"
            trend_strength = abs(fast_ema - slow_ema) / slow_ema if slow_ema > 0 else 0

            self._ema_cache[symbol] = {
                "fast": fast_ema,
                "slow": slow_ema,
                "trend": trend,
                "trend_strength": trend_strength
            }
        except Exception as e:
            logger.debug(f"Failed to update EMA for {symbol}: {e}")

    def _calculate_ema(self, data: np.ndarray, period: int) -> Optional[float]:
        """计算EMA"""
        if len(data) < period:
            return None

        multiplier = 2.0 / (period + 1)
        ema = np.mean(data[:period])

        for i in range(period, len(data)):
            ema = (data[i] - ema) * multiplier + ema

        return float(ema)

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
                         abs(highs[i] - closes[i - 1]),
                         abs(lows[i] - closes[i - 1]))
                tr_values.append(tr)

            if not tr_values:
                return

            atr = np.mean(tr_values[-self._atr_period:])

            if np.isnan(atr) or np.isinf(atr) or atr <= 0:
                return

            self._atr_cache[symbol] = atr
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

            # 优化：保存平均成交量用于过滤
            avg_volume = np.mean(volumes)

            self._volume_profile_cache[symbol] = {
                "poc_price": poc_price,
                "vwap_price": vwap_price,
                "min_price": min_price,
                "max_price": max_price,
                "avg_volume": avg_volume,
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

    async def _build_grid(self, symbol: str):
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return

        current_price = float(ticker.get("last", 0) or 0)
        if not np.isfinite(current_price) or current_price <= 0:
            logger.warning(f"Spot Grid {symbol}: invalid current_price {current_price!r}, skip grid build")
            return
        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]

        atr = self._atr_cache.get(symbol, 0)
        if atr > 0:
            atr_spacing = atr * self._atr_multiplier / current_price
            if self._volatility_adaptive_spacing:
                base_spacing = max(self._min_grid_spacing, min(self._max_grid_spacing, atr_spacing))
            else:
                base_spacing = max(tier_settings["grid_spacing_min"], min(tier_settings["grid_spacing_max"], atr_spacing))
        else:
            base_spacing = (tier_settings["grid_spacing_min"] + tier_settings["grid_spacing_max"]) / 2

        if self._dynamic_grid_count and atr > 0:
            volatility_ratio = atr / current_price
            if volatility_ratio > 0.02:
                grid_count = self._grid_count_min
            elif volatility_ratio > 0.01:
                grid_count = (self._grid_count_min + self._grid_count_max) // 2
            else:
                grid_count = self._grid_count_max
        else:
            grid_count = int(np.random.uniform(self._grid_count_min, self._grid_count_max))

        grid_count = max(3, min(grid_count, 30))
        half_grids = grid_count // 2

        vp = self._volume_profile_cache.get(symbol)
        if vp and vp.get("min_price") is not None and vp.get("max_price") is not None:
            vp_min = float(vp["min_price"])
            vp_max = float(vp["max_price"])
            if vp_min > 0 and vp_max > vp_min:
                upper_bound = vp_max * 1.01
                lower_bound = vp_min * 0.99
            else:
                upper_bound = current_price * (1 + base_spacing * half_grids)
                lower_bound = current_price * (1 - base_spacing * half_grids)
        else:
            upper_bound = current_price * (1 + base_spacing * half_grids)
            lower_bound = current_price * (1 - base_spacing * half_grids)

        lower_bound = max(lower_bound, current_price * 0.5)
        upper_bound = min(upper_bound, current_price * 1.5)

        if lower_bound >= upper_bound:
            lower_bound = current_price * (1 - base_spacing * half_grids)
            upper_bound = current_price * (1 + base_spacing * half_grids)

        # P1: 使用几何间距在百分比空间上均匀分布网格
        prices = np.geomspace(max(lower_bound, 1e-8), max(upper_bound, 1e-8), grid_count)

        # P1: 找到最接近 current_price 的索引作为买卖分界线
        closest_idx = int(np.argmin(np.abs(prices - current_price)))

        self._grids[symbol] = []
        for i, price in enumerate(prices):
            side = "buy" if i <= closest_idx else "sell"

            density_factor = self._calculate_density_factor(symbol, price, vp)
            adjusted_spacing = base_spacing * density_factor

            self._grids[symbol].append({
                "price": round(price, 4),
                "side": side,
                "filled": False,
                "fill_time": None,  # 优化：记录成交时间
                "layer": abs(i - closest_idx),
                "quantity": 0,
                "density_factor": density_factor,
                "adjusted_spacing": adjusted_spacing
            })

        self._last_adjust_time[symbol] = datetime.now()

        self._log_throttled(symbol, f"Spot Grid built for {symbol}: {grid_count} levels, range [{lower_bound:.4f}, {upper_bound:.4f}], spacing: {base_spacing:.4f}")

    def _log_throttled(self, symbol: str, message: str):
        """优化：日志节流输出"""
        now = time.time()
        last_log = self._last_log_time.get(symbol, 0)

        if now - last_log >= self._log_throttle_interval:
            logger.info(message)
            self._last_log_time[symbol] = now
        else:
            logger.debug(message)

    def _calculate_density_factor(self, symbol: str, price: float, vp: Optional[Dict]) -> float:
        if not vp or not vp.get("high_volume_zones"):
            return 1.0

        for zone in vp["high_volume_zones"]:
            if zone["start"] <= price <= zone["end"]:
                return 0.75

        return 1.25

    async def _monitor_ticks(self):
        while True:
            try:
                results = await asyncio.gather(
                    *[self._process_tick(sym) for sym in self._all_symbols],
                    return_exceptions=True,
                )
                ws_used = any(r is True for r in results)
                await asyncio.sleep(0.1 if ws_used else 1.0)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SpotGrid _monitor_ticks error: {e}")
                await asyncio.sleep(1.0)

    async def _process_tick(self, symbol: str) -> bool:
        tick = self.redis_cache.get_tick(symbol)
        from_ws = True

        if not tick:
            tick = await self._get_tick_rest(symbol)
            from_ws = False
            if not tick:
                return False

        price = tick.price
        last_price = self._last_tick_price.get(symbol, 0)

        if last_price == 0:
            self._last_tick_price[symbol] = price
            return from_ws

        # 优化：趋势过滤
        if self._trend_filter_enabled:
            trend_info = self._ema_cache.get(symbol, {})
            trend = trend_info.get("trend", "neutral")

        grids = self._grids.get(symbol, [])
        if not grids:
            return from_ws

        for grid in grids:
            if grid["filled"]:
                # 优化：网格重置逻辑
                if self._grid_reset_enabled and grid.get("fill_time"):
                    fill_time = grid["fill_time"]
                    if (datetime.now() - fill_time).total_seconds() >= self._grid_reset_delay:
                        grid["filled"] = False
                        grid["fill_time"] = None
                continue

            grid_price = grid["price"]
            side = grid["side"]

            # 优化：趋势过滤 - 下跌趋势禁止买入网格
            if self._trend_filter_enabled and side == "buy":
                trend_info = self._ema_cache.get(symbol, {})
                trend = trend_info.get("trend", "neutral")
                trend_strength = trend_info.get("trend_strength", 0)

                # 强下跌趋势（trend_strength > 0.01）禁止买入
                if trend == "down" and trend_strength > 0.01:
                    continue

            if side == "buy" and last_price > grid_price >= price:
                if await self._confirm_grid_entry(symbol, "buy", price, tick):
                    await self._trigger_grid_order(symbol, "buy", grid_price, grid["layer"])
                    grid["filled"] = True
                    grid["fill_time"] = datetime.now()
            elif side == "sell" and last_price < grid_price <= price:
                if await self._confirm_grid_entry(symbol, "sell", price, tick):
                    await self._trigger_grid_order(symbol, "sell", grid_price, grid["layer"])
                    grid["filled"] = True
                    grid["fill_time"] = datetime.now()

        self._last_tick_price[symbol] = price
        return from_ws

    async def _get_tick_rest(self, symbol: str) -> Optional[TickData]:
        now = time.time()
        cache_entry = self._tick_rest_cache.get(symbol)
        if cache_entry and (now - cache_entry[0]) < self._tick_rest_interval:
            return cache_entry[1]

        try:
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
                timestamp=datetime.now(),
                instrument_type="SPOT"
            )

            if tick.price <= 0:
                return None

            self._tick_rest_cache[symbol] = (now, tick)
            return tick
        except Exception as e:
            logger.debug(f"REST tick fallback failed for {symbol}: {e}")
            return None

    async def _confirm_grid_entry(self, symbol: str, side: str, price: float, tick) -> bool:
        # 优化：成交量过滤
        if self._volume_filter_enabled:
            vp = self._volume_profile_cache.get(symbol)
            if vp and vp.get("avg_volume"):
                avg_volume = vp["avg_volume"]
                current_volume = tick.volume or 0
                # 成交量低于均值的30%则过滤
                if current_volume < avg_volume * self._volume_threshold_ratio:
                    return False

        # 原有订单簿过滤
        if side == "buy":
            bid_vol = tick.bid_volume or 0
            ask_vol = tick.ask_volume or 0
            if ask_vol > 0 and bid_vol / ask_vol < 0.2:
                return False
        else:
            bid_vol = tick.bid_volume or 0
            ask_vol = tick.ask_volume or 0
            if bid_vol > 0 and ask_vol / bid_vol < 0.2:
                return False

        return True

    async def _trigger_grid_order(self, symbol: str, side: str, price: float, layer: int):
        # 企业级：参数前置校验，非法输入直接拒绝并埋点
        if not self._validate_symbol(symbol) or not self._validate_direction(side) or not self._validate_price(price):
            logger.warning(
                f"Spot grid order rejected: invalid params "
                f"symbol={symbol!r} side={side!r} price={price!r}"
            )
            self._increment_metric("spot_grid_signal_rejected_total", 1.0, {"reason": "invalid_params", "symbol": symbol})
            return

        tier = get_currency_tier(symbol, self.config)
        tier_settings = self.config["currencies"][f"{tier}_settings"]

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
                logger.debug(f"[spot_grid] get_position_boost failed: {e}")

        # === 优化：单币种与全策略仓位上限（小账户需要更集中的资金利用）===
        if side == "buy":
            # 单币种最多使用 100% 策略配额，避免小账户因 cap 过低无法交易高价币
            max_symbol_exposure = trading_capital * allocation * 1.0
            current_holdings = self._get_current_holdings(symbol)
            ticker = await self.okx_client.get_ticker_async(symbol)
            ref_price = float(ticker["last"]) if ticker else price
            current_holdings_value = current_holdings * ref_price
            if current_holdings_value + base_position > max_symbol_exposure:
                logger.debug(
                    f"Spot Grid {symbol}: skip buy, symbol exposure {current_holdings_value:.2f} + {base_position:.2f} > cap {max_symbol_exposure:.2f}"
                )
                return

            # 全策略总持仓上限：累计 <= 100% 策略配额
            total_exposure = 0.0
            for sym, pos in self._active_positions.items():
                pos_qty = float(pos.get("quantity", 0))
                if pos_qty <= 0:
                    continue
                if sym == symbol:
                    # 同币种用上面已查的实时价
                    pos_price = ref_price
                else:
                    t = await self.okx_client.get_ticker_async(sym)
                    pos_price = float(t["last"]) if t else float(pos.get("avg_price", 0))
                total_exposure += pos_qty * pos_price
            if total_exposure + base_position > trading_capital * allocation * 1.0:
                logger.debug(
                    f"Spot Grid {symbol}: skip buy, total exposure {total_exposure:.2f} + {base_position:.2f} > strategy cap {trading_capital * allocation * 1.0:.2f}"
                )
                return

        quantity = base_position / price

        instr_info = await self.okx_client.get_instrument_info_async(symbol) or {}
        min_lot_size = self._safe_float(instr_info.get("lotSz", "0.001"), 0.001)
        if quantity < min_lot_size:
            logger.debug(f"Spot Grid {symbol}: quantity {quantity:.6f} < min lot {min_lot_size}, skip")
            return

        if side == "buy":
            available_balance = self._get_available_usdt_balance()
            if available_balance < base_position:
                logger.debug(f"Spot Grid {symbol}: insufficient USDT balance {available_balance:.2f} < required {base_position:.2f}")
                return
        else:
            holdings = self._get_current_holdings(symbol)
            if holdings < quantity:
                logger.debug(f"Spot Grid {symbol}: insufficient holdings {holdings:.6f} < required {quantity:.6f}")
                return

        precision = get_price_precision(symbol)
        quantity_precision = await self._get_quantity_precision(symbol)

        quantity = round(quantity, quantity_precision)
        if quantity <= 0:
            logger.debug(f"Spot Grid {symbol}: skip open, quantity {quantity} <= 0 after rounding")
            return
        price = round(price, precision)

        signal = Signal(
            symbol=symbol,
            strategy_name="spot_grid",
            signal_type="spot_grid_trade",
            direction=side,
            price=price,
            quantity=quantity,
            leverage=1,
            stop_loss=None,
            take_profit=None,
            confidence=0.95 - layer * 0.05,
            timestamp=datetime.now()
        )

        if self._signal_callback:
            await self._signal_callback(signal)

        self._record_metric("spot_grid_signal_generated_total", 1.0, {"symbol": symbol, "direction": side, "layer": str(layer)})
        self._record_metric("spot_grid_signal_confidence", signal.confidence, {"symbol": symbol, "direction": side})

        # === P0-1: buy 成交后把成交价加权合并进 _active_positions[symbol]["avg_price"] ===
        if side == "buy":
            fill_price = price  # 网格限价单成交价即信号价
            if symbol in self._active_positions:
                pos = self._active_positions[symbol]
                old_qty = float(pos.get("quantity", 0))
                old_avg = float(pos.get("avg_price", 0))
                new_qty = quantity
                if old_qty > 0 and old_avg > 0:
                    new_total = old_qty + new_qty
                    new_avg = (old_qty * old_avg + new_qty * fill_price) / new_total
                    pos["avg_price"] = new_avg
                    pos["quantity"] = new_total
                    pos["unreconciled"] = False  # 已用真实成交价更新
                else:
                    pos["avg_price"] = fill_price
                    pos["quantity"] = new_qty
                    pos["unreconciled"] = False
                # 同步本地入场价缓存
                self._entry_price_cache[symbol] = pos["avg_price"]
                self._reconciled_symbols.add(symbol)
            else:
                self._active_positions[symbol] = {
                    "avg_price": fill_price,
                    "quantity": quantity,
                    "start_time": datetime.now(),
                    "unreconciled": False,
                }
                self._entry_price_cache[symbol] = fill_price
                self._reconciled_symbols.add(symbol)

        # P2: 防卖飞底仓 - 卖出成交后保留30%仓位在当前价附近回补
        if side == "sell":
            retain_ratio = 0.3
            rebuy_qty = quantity * retain_ratio
            instr_info = await self.okx_client.get_instrument_info_async(symbol) or {}
            min_lot_size = self._safe_float(instr_info.get("lotSz", "0.001"), 0.001)
            if rebuy_qty >= min_lot_size:
                precision = get_price_precision(symbol)
                qty_prec = await self._get_quantity_precision(symbol)
                rebuy_qty = round(rebuy_qty, qty_prec)
                if rebuy_qty <= 0:
                    logger.debug(f"Spot Grid {symbol}: skip retain rebuy, rebuy_qty {rebuy_qty} <= 0 after rounding")
                else:
                    rebuy_price = round(price, precision)
                    retain_signal = Signal(
                        symbol=symbol,
                        strategy_name="spot_grid",
                        signal_type="spot_grid_retain",
                        direction="buy",
                        price=rebuy_price,
                        quantity=rebuy_qty,
                        leverage=1,
                        stop_loss=None,
                        take_profit=None,
                        confidence=0.85,
                        timestamp=datetime.now()
                    )
                    if self._signal_callback:
                        await self._signal_callback(retain_signal)
                    logger.info(f"Spot Grid retain: {symbol} rebuy {rebuy_qty:.6f} @ {rebuy_price:.4f} (30% of sold qty)")

        self._log_throttled(symbol, f"Spot Grid {side.upper()} signal: {symbol} @ {price:.4f} x {quantity:.6f}")

    def _get_available_usdt_balance(self) -> float:
        try:
            balance = self.okx_client.get_spot_balance("USDT")
            return float(balance.get("available", 0)) if balance else 0
        except Exception as e:
            logger.error(f"Failed to get USDT balance: {e}")
            return 0

    def _get_current_holdings(self, symbol: str) -> float:
        base_asset = symbol.replace("-USDT", "")
        try:
            balance = self.okx_client.get_spot_balance(base_asset)
            return float(balance.get("available", 0)) if balance else 0
        except Exception as e:
            logger.error(f"Failed to get {base_asset} balance: {e}")
            return 0

    async def _get_quantity_precision(self, symbol: str) -> int:
        try:
            info = await self.okx_client.get_instrument_info_async(symbol)
            lot_size = float(info.get("lotSz", "0.001"))
            return len(str(lot_size).split(".")[1]) if "." in str(lot_size) else 0
        except Exception:
            return 3

    async def _dynamic_adjust_loop(self):
        while True:
            try:
                now = datetime.now()
                for symbol in self._all_symbols:
                    last_adjust = self._last_adjust_time.get(symbol)
                    if last_adjust and (now - last_adjust).total_seconds() >= self._dynamic_adjust_interval:
                        await self._adjust_grid(symbol)
                await asyncio.sleep(self._dynamic_adjust_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SpotGrid _dynamic_adjust_loop error: {e}")
                await asyncio.sleep(60)

    async def _adjust_grid(self, symbol: str):
        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return

        current_price = float(ticker["last"])
        grids = self._grids.get(symbol, [])
        if not grids:
            return

        avg_price = sum(g["price"] for g in grids) / len(grids)
        price_diff = current_price - avg_price
        price_diff_ratio = abs(price_diff) / avg_price

        if price_diff_ratio > self._volatility_threshold:
            self._log_throttled(symbol, f"Adjusting spot grid for {symbol}: price moved {price_diff_ratio:.2%} from center")
            await self._build_grid(symbol)

    async def _order_check_loop(self):
        while True:
            try:
                for symbol in self._all_symbols:
                    await self._check_pending_orders(symbol)
                await asyncio.sleep(self._order_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SpotGrid _order_check_loop error: {e}")
                await asyncio.sleep(60)

    async def _check_pending_orders(self, symbol: str):
        pass

    async def _multi_symbol_rebalance_loop(self):
        while True:
            try:
                await asyncio.sleep(self._rebalance_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"SpotGrid _multi_symbol_rebalance_loop error: {e}")
                await asyncio.sleep(60)

    async def _position_monitor_loop(self):
        """优化：持仓监控循环 - 动态止盈止损"""
        while True:
            try:
                for symbol in self._all_symbols:
                    await self._check_position_pnl(symbol)
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
            await asyncio.sleep(self._position_check_interval)

    def _fetch_recent_fills_from_okx(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """P0-1: 从 OKX /api/v5/trade/fills-history 拉最近 N 天成交记录（用于反推真实入场价）"""
        try:
            # OKX fills-history 最多返回最近 3 天，要 7 天需翻页；这里尽量取，取不到就返回空
            path = f"/api/v5/trade/fills-history?instType=SPOT&instId={symbol}&limit=100"
            make_req = getattr(self.okx_client, "_make_request", None)
            if not callable(make_req):
                return []
            data = make_req("GET", path)
            if data and data.get("code") == "0":
                return data.get("data", []) or []
            return []
        except Exception as e:
            logger.debug(f"Failed to fetch fills-history for {symbol}: {e}")
            return []

    def _reconcile_entry_price(self, symbol: str, holdings: float, current_price: float) -> Tuple[float, bool]:
        """P0-1: 反推真实入场价。
        优先级：本地缓存 > OKX fills-history > 当前价（打 unreconciled=True 兜底）。
        返回 (avg_price, unreconciled)
        """
        # 1. 本地缓存（之前 buy 成交时已写入）
        if symbol in self._entry_price_cache and self._entry_price_cache[symbol] > 0:
            return float(self._entry_price_cache[symbol]), False

        # 2. OKX fills-history 反推（每个 symbol 只对账一次，避免每轮调 API）
        if symbol not in self._reconciled_symbols:
            try:
                fills = self._fetch_recent_fills_from_okx(symbol, days=7)
                buy_fills = [f for f in fills if str(f.get("side", "")).lower() == "buy"]
                if buy_fills:
                    # 按时间倒序，累加 qty 直到覆盖当前持仓
                    buy_fills.sort(key=lambda f: f.get("fillTime", "0"), reverse=True)
                    total_qty = 0.0
                    total_value = 0.0
                    for fill in buy_fills:
                        try:
                            qty = float(fill.get("fillSz", 0) or 0)
                            px = float(fill.get("fillPx", 0) or 0)
                            if qty <= 0 or px <= 0:
                                continue
                            total_qty += qty
                            total_value += qty * px
                            if total_qty >= holdings * 0.95:
                                break
                        except (ValueError, TypeError):
                            continue
                    if total_qty > 0 and total_value > 0:
                        avg_px = total_value / total_qty
                        self._entry_price_cache[symbol] = avg_px
                        self._reconciled_symbols.add(symbol)
                        logger.info(f"Spot Grid {symbol}: reconciled entry price from fills: {avg_px:.4f}")
                        return avg_px, False
            except Exception as e:
                logger.debug(f"Spot Grid {symbol}: fills reconciliation failed: {e}")
            self._reconciled_symbols.add(symbol)

        # 3. 兜底：当前价 + unreconciled 标记
        logger.warning(
            f"Spot Grid {symbol}: no fills data for reconciliation, using current_price {current_price:.4f} as avg_price (unreconciled=True)"
        )
        return current_price, True

    def _compute_atr_based_tp_sl(self, symbol: str, current_price: float) -> Tuple[float, float]:
        """P0-2: 基于 ATR 计算动态止盈止损比例。
        tp_pct = max(0.005, min(0.05, atr/price * 1.5))
        sl_pct = max(0.02, min(0.15, atr/price * 2.0))
        ATR 不可用时 fallback 到配置值。
        返回 (tp_pct, sl_pct)
        """
        atr = float(self._atr_cache.get(symbol, 0) or 0)
        if atr <= 0 or current_price <= 0:
            return float(self._take_profit_pct), float(self._stop_loss_pct)
        atr_ratio = atr / current_price
        tp_pct = max(0.005, min(0.05, atr_ratio * 1.5))
        sl_pct = max(0.02, min(0.15, atr_ratio * 2.0))
        return tp_pct, sl_pct

    def _get_net_profit_pct(self, entry_price: float, exit_price: float) -> float:
        taker_fee = self.config["trading"].get("taker_fee_rate", 0.0005)
        gross_pnl = (exit_price - entry_price) / entry_price
        fee_cost = taker_fee * (entry_price + exit_price) / entry_price
        return gross_pnl - fee_cost

    async def _check_position_pnl(self, symbol: str):
        """检查持仓盈亏，触发止盈止损"""
        holdings = self._get_current_holdings(symbol)
        if holdings <= 0:
            # 持仓已平，清理本地状态
            if symbol in self._active_positions:
                self._active_positions.pop(symbol, None)
                self._entry_price_cache.pop(symbol, None)
                self._reconciled_symbols.discard(symbol)
            return

        ticker = await self.okx_client.get_ticker_async(symbol)
        if not ticker:
            return

        current_price = float(ticker["last"])

        # === P0-1: 首次检测到持仓时尝试反推真实入场价 ===
        if symbol not in self._active_positions:
            avg_price, unreconciled = self._reconcile_entry_price(symbol, holdings, current_price)
            self._active_positions[symbol] = {
                "avg_price": avg_price,
                "quantity": holdings,
                "start_time": datetime.now(),
                "unreconciled": unreconciled,
            }
            return

        pos = self._active_positions[symbol]
        # 同步最新持仓量（外部可能加减仓）
        pos["quantity"] = holdings
        avg_price = float(pos.get("avg_price", current_price))
        if avg_price <= 0:
            avg_price = current_price
            pos["avg_price"] = avg_price
            pos["unreconciled"] = True

        # 计算盈亏比例（扣除买卖手续费）
        net_pnl_pct = self._get_net_profit_pct(avg_price, current_price) if avg_price > 0 else 0
        pnl_pct = net_pnl_pct

        # === P0-2: ATR 动态止盈止损 ===
        tp_pct, sl_pct = self._compute_atr_based_tp_sl(symbol, current_price)

        # 止盈检查
        if pnl_pct >= tp_pct:
            logger.info(
                f"Spot Grid take profit triggered: {symbol} profit={pnl_pct:.2%} (tp_pct={tp_pct:.4f})"
            )
            await self._close_position(symbol, current_price, holdings, "take_profit")

        # 止损检查
        elif pnl_pct <= -sl_pct:
            logger.warning(
                f"Spot Grid stop loss triggered: {symbol} loss={pnl_pct:.2%} (sl_pct={sl_pct:.4f})"
            )
            await self._close_position(symbol, current_price, holdings, "stop_loss")

    async def _close_position(self, symbol: str, price: float, quantity: float, reason: str):
        """平仓"""
        precision = get_price_precision(symbol)
        quantity_precision = await self._get_quantity_precision(symbol)

        quantity = round(quantity, quantity_precision)
        if quantity <= 0:
            logger.debug(f"Spot Grid {symbol}: skip close, quantity {quantity} <= 0 after rounding")
            if symbol in self._active_positions:
                del self._active_positions[symbol]
            return
        price = round(price, precision)

        signal = Signal(
            symbol=symbol,
            strategy_name="spot_grid",
            signal_type=f"spot_grid_{reason}",
            direction="sell",
            price=price,
            quantity=quantity,
            leverage=1,
            stop_loss=None,
            take_profit=None,
            confidence=0.9,
            timestamp=datetime.now()
        )

        if self._signal_callback:
            await self._signal_callback(signal)

        logger.info(f"Spot Grid close position ({reason}): {symbol} @ {price:.4f} x {quantity:.6f}")

        if symbol in self._active_positions:
            del self._active_positions[symbol]
        # P0-1: 同步清理对账缓存
        self._entry_price_cache.pop(symbol, None)
        self._reconciled_symbols.discard(symbol)
        # 平仓后立即重置网格，确保后续能重新开仓
        asyncio.create_task(self._build_grid(symbol))

    # === P0-4: 状态持久化 ===
    def collect_persistent_state(self) -> Dict[str, Any]:
        """收集需要持久化的策略状态。"""
        return {
            "grids": self._grids,
            "active_positions": self._active_positions,
            "last_adjust_time": {
                sym: ts.isoformat() if isinstance(ts, datetime) else ts
                for sym, ts in self._last_adjust_time.items()
            },
            "grid_performance": self._grid_performance,
        }

    def restore_persistent_state(self, state: Dict[str, Any]):
        """从持久化数据恢复策略状态。"""
        try:
            grids = state.get("grids")
            if isinstance(grids, dict):
                # P0-4: 反序列化 fill_time（json.dumps(default=str) 会将其转字符串），
                #       否则重启后 _process_tick 的 (datetime.now() - fill_time) 会抛 TypeError。
                for sym, levels in grids.items():
                    if not isinstance(levels, list):
                        continue
                    for g in levels:
                        if isinstance(g, dict) and isinstance(g.get("fill_time"), str):
                            try:
                                g["fill_time"] = datetime.fromisoformat(g["fill_time"])
                            except (ValueError, TypeError):
                                g["fill_time"] = None
                self._grids = grids
            ap = state.get("active_positions")
            if isinstance(ap, dict):
                self._active_positions = ap
            lat = state.get("last_adjust_time")
            if isinstance(lat, dict):
                for sym, ts in lat.items():
                    if isinstance(ts, str):
                        try:
                            self._last_adjust_time[sym] = datetime.fromisoformat(ts)
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(ts, datetime):
                        self._last_adjust_time[sym] = ts
            gp = state.get("grid_performance")
            if isinstance(gp, dict):
                self._grid_performance = gp
            logger.info(
                f"Spot Grid state restored: grids={len(self._grids)}, positions={len(self._active_positions)}, perf={len(self._grid_performance)}"
            )
        except Exception as e:
            logger.error(f"Spot Grid restore_persistent_state failed: {e}")

    @property
    def _atr_cache(self):
        if not hasattr(self, '_atr_cache_dict'):
            self._atr_cache_dict = {}
        return self._atr_cache_dict

    @property
    def _volume_profile_cache(self):
        if not hasattr(self, '_volume_profile_cache_dict'):
            self._volume_profile_cache_dict = {}
        return self._volume_profile_cache_dict