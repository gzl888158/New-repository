"""现货马丁格尔策略：价格下跌时按比例分批加仓摊薄成本，反弹后分批止盈。"""
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger

from core.models import Signal, TickData
from configs.settings import get_currency_tier
from utils.helpers import get_price_precision
from utils.state_persistence import PersistentStrategy


class SpotMartingaleStrategy(PersistentStrategy):
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        super().__init__()
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache

        self._enabled = config["strategies"].get("spot_martingale", {}).get("enabled", False)
        # 提升 base_position_ratio 默认值至0.05（之前0.02导致基础仓位过小）
        self._base_position_ratio = config["strategies"].get("spot_martingale", {}).get("base_position_ratio", 0.05)
        self._martingale_coefficient = config["strategies"].get("spot_martingale", {}).get("martingale_coefficient", 1.5)
        self._max_layers = config["strategies"].get("spot_martingale", {}).get("max_layers", 5)
        self._price_drop_pct = config["strategies"].get("spot_martingale", {}).get("price_drop_pct", 0.02)
        self._take_profit_pct = config["strategies"].get("spot_martingale", {}).get("take_profit_pct", 0.015)
        self._stop_loss_pct = config["strategies"].get("spot_martingale", {}).get("stop_loss_pct", 0.15)
        self._check_interval = config["strategies"].get("spot_martingale", {}).get("check_interval", 5)
        self._min_signal_quality = config["strategies"].get("spot_martingale", {}).get("min_signal_quality", 0.20)

        self._active_positions: Dict[str, Dict[str, Any]] = {}
        self._last_check_time: Dict[str, float] = {}
        self._tick_rest_cache: Dict[str, tuple] = {}
        self._tick_rest_interval = 1.0

        self._adaptive_controller = None
        self._signal_callback = None

        self._all_symbols = []
        currencies = config.get("currencies", {})
        if currencies:
            for tier in ["tier1", "tier2", "tier3"]:
                for base in currencies.get(f"{tier}_symbols", []):
                    self._all_symbols.append(f"{base}-USDT")

        # === 优化：波动率自适应 ===
        self._atr_cache: Dict[str, float] = {}
        self._atr_period = config["strategies"].get("spot_martingale", {}).get("atr_period", 14)
        self._volatility_adaptive = config["strategies"].get("spot_martingale", {}).get("volatility_adaptive", True)
        self._volatility_threshold_high = config["strategies"].get("spot_martingale", {}).get("volatility_threshold_high", 0.05)  # 5%波动率暂停开仓
        self._volatility_threshold_low = config["strategies"].get("spot_martingale", {}).get("volatility_threshold_low", 0.02)  # 2%波动率正常开仓

        # === 优化：动态止盈 ===
        self._dynamic_take_profit = config["strategies"].get("spot_martingale", {}).get("dynamic_take_profit", True)
        self._take_profit_base = config["strategies"].get("spot_martingale", {}).get("take_profit_pct", 0.015)
        self._take_profit_decay = config["strategies"].get("spot_martingale", {}).get("take_profit_decay", 0.002)  # 每层减少0.2%

        # === 优化：市场状态检测 ===
        self._market_state: Dict[str, str] = {}  # symbol -> bull/bear/neutral
        self._ema_fast_period = config["strategies"].get("spot_martingale", {}).get("ema_fast_period", 12)
        self._ema_slow_period = config["strategies"].get("spot_martingale", {}).get("ema_slow_period", 26)
        self._ema_cache: Dict[str, Dict[str, float]] = {}
        self._bear_market_suspend = config["strategies"].get("spot_martingale", {}).get("bear_market_suspend", True)  # 熊市暂停开仓

        # === 优化：最大持仓时间 ===
        self._max_hold_time_hours = config["strategies"].get("spot_martingale", {}).get("max_hold_time_hours", 72)  # 72小时

        # === 优化：分批止盈 ===
        self._partial_close_enabled = config["strategies"].get("spot_martingale", {}).get("partial_close_enabled", True)
        self._partial_close_threshold = config["strategies"].get("spot_martingale", {}).get("partial_close_threshold", 3)  # 第3层以上启用分批止盈
        self._partial_close_ratio = config["strategies"].get("spot_martingale", {}).get("partial_close_ratio", 0.5)  # 先平50%

        # === 优化：日志节流 ===
        self._last_log_time: Dict[str, float] = {}
        self._log_throttle_interval = 60

        # === 优化：动态加仓阈值 ===
        self._dynamic_drop_threshold = config["strategies"].get("spot_martingale", {}).get("dynamic_drop_threshold", True)

        # === 优化：交易次数限制 ===
        self._daily_trades: Dict[str, int] = {}  # symbol -> count
        self._max_daily_trades_per_symbol = config["strategies"].get("spot_martingale", {}).get("max_daily_trades_per_symbol", 5)
        self._daily_reset_date: Optional[str] = datetime.now().strftime("%Y-%m-%d")

        # === P0-3: 止损冷却期 + 持仓期最高价 ===
        self._cooldown_until: Dict[str, datetime] = {}  # symbol -> 冷却到期时间
        self._last_open_time: Dict[str, datetime] = {}  # P0-4: 同币种开仓冷却
        self._same_symbol_cooldown_minutes = config["strategies"].get("spot_martingale", {}).get("same_symbol_cooldown_minutes", 30)
        self._rsi_entry_threshold = config["strategies"].get("spot_martingale", {}).get("rsi_entry_threshold", 40.0)

        # === P0-4: 信号质量 ===
        self._rsi_cache: Dict[str, float] = {}  # symbol -> RSI(1H)

        # === P0-6: 状态持久化 ===
        self.init_state_persistence("spot_martingale", redis_cache)

    def set_adaptive_controller(self, controller):
        self._adaptive_controller = controller

    def set_coordinator(self, coordinator):
        """注入StrategyCoordinator实例"""
        self._coordinator = coordinator

    async def update_config(self, updates: Dict[str, Any]):
        """运行时热更新策略配置（不重启策略）"""
        strategy_cfg = self.config.get("strategies", {}).get("spot_martingale", {})
        strategy_cfg.update(updates)
        self.config.setdefault("strategies", {})["spot_martingale"] = strategy_cfg

        attr_map = {
            "max_layers": "max_layers",
            "price_drop_pct": "price_drop_pct",
            "take_profit_pct": "take_profit_pct",
            "stop_loss_pct": "stop_loss_pct",
            "min_signal_quality": "min_signal_quality",
            "martingale_coefficient": "martingale_coefficient",
            "max_hold_time_hours": "max_hold_time_hours",
            "base_position_ratio": "base_position_ratio",
            "bear_market_suspend": "bear_market_suspend",
            "partial_close_enabled": "partial_close_enabled",
        }
        for cfg_key, attr_name in attr_map.items():
            if cfg_key in updates:
                setattr(self, attr_name, updates[cfg_key])
                logger.info(f"SpotMartingale config hot-updated: {attr_name}={updates[cfg_key]}")

    def set_signal_callback(self, callback):
        self._signal_callback = callback

    def _get_allocation(self) -> float:
        if self._adaptive_controller:
            try:
                return self._adaptive_controller.get_allocation("spot_martingale")
            except Exception:
                pass
        return self.config["trading"].get("spot_martingale_allocation", 0.15)

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
        except Exception:
            pass
        return self.config["trading"].get("total_capital", 100.0)

    def apply_param_update(self, params: Dict[str, Any]):
        applied = []
        if "base_position_ratio" in params:
            self._base_position_ratio = params["base_position_ratio"]
            applied.append("base_position_ratio")
        if "martingale_coefficient" in params:
            self._martingale_coefficient = params["martingale_coefficient"]
            applied.append("martingale_coefficient")
        if "max_layers" in params:
            self._max_layers = params["max_layers"]
            applied.append("max_layers")
        if "price_drop_pct" in params:
            self._price_drop_pct = params["price_drop_pct"]
            applied.append("price_drop_pct")
        if "take_profit_pct" in params:
            self._take_profit_pct = params["take_profit_pct"]
            self._take_profit_base = params["take_profit_pct"]
            applied.append("take_profit_pct")
        if "stop_loss_pct" in params:
            self._stop_loss_pct = params["stop_loss_pct"]
            applied.append("stop_loss_pct")
        if applied:
            logger.info(f"Spot Martingale strategy params hot-updated: {applied}")
        return applied

    async def start(self):
        if not self._enabled:
            logger.info("Spot Martingale strategy is disabled")
            return

        logger.info("Starting Spot Martingale Strategy")
        # P0-2 / P0-6: 启动时优先恢复持久化状态（_active_positions 等）
        state_loaded = await self.load_state_async()
        # P0-2: 加载持久化状态后，再校验实际持仓；未加载到则走 fills 反推
        await self._load_active_positions(skip_if_loaded=state_loaded)

        asyncio.create_task(self._monitor_ticks())
        asyncio.create_task(self._check_positions_loop())
        # 优化：添加指标更新循环
        asyncio.create_task(self._update_indicators_loop())
        asyncio.create_task(self._daily_reset_loop())
        asyncio.create_task(self._update_rsi_loop())
        # P0-6: 周期保存策略状态
        asyncio.create_task(self.periodic_save_loop())

    async def _daily_reset_loop(self):
        """每日0点重置交易次数"""
        try:
            while True:
                now = datetime.now()
                next_reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                wait_seconds = (next_reset - now).total_seconds()
                await asyncio.sleep(wait_seconds)
                self._daily_trades.clear()
                self._daily_reset_date = datetime.now().strftime("%Y-%m-%d")
                logger.info("Spot Martingale daily trade counters reset")
        except asyncio.CancelledError:
            logger.info("Spot Martingale _daily_reset_loop cancelled")

    async def _update_rsi_loop(self):
        """P0-4: 每分钟刷新 RSI(1H) 缓存，供开仓信号质量评估使用"""
        while True:
            try:
                for symbol in self._all_symbols:
                    await self._update_rsi(symbol)
            except Exception as e:
                logger.error(f"RSI update loop error: {e}")
            await asyncio.sleep(60)

    async def _update_rsi(self, symbol: str):
        try:
            klines = self.okx_client.get_kline(symbol, "1H", limit=20)
            if len(klines) < 15:
                return
            closes = np.array([float(k[4]) for k in klines], dtype=float)
            if np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
                return
            rsi = self._calculate_rsi(closes, 14)
            if 0 <= rsi <= 100:
                self._rsi_cache[symbol] = float(rsi)
        except Exception as e:
            logger.debug(f"Failed to update RSI for {symbol}: {e}")

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """计算 RSI 指标（Wilder's EMA 平滑）"""
        if len(prices) < period + 1:
            return 50.0
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)

            # 初始值用简单平均（前 period 根）
            avg_gain = float(np.mean(gains[:period]))
            avg_loss = float(np.mean(losses[:period]))

            # 后续用 Wilder's EMA: alpha = 1/period
            alpha = 1.0 / period
            for i in range(period, len(gains)):
                avg_gain = alpha * float(gains[i]) + (1.0 - alpha) * avg_gain
                avg_loss = alpha * float(losses[i]) + (1.0 - alpha) * avg_loss

            if avg_loss == 0:
                return 100.0
            if avg_gain == 0:
                return 0.0
            rs = avg_gain / avg_loss
            return max(0.0, min(100.0, 100.0 - (100.0 / (1.0 + rs))))
        except Exception:
            return 50.0

    async def _update_indicators_loop(self):
        """更新ATR和EMA指标"""
        while True:
            for symbol in self._all_symbols:
                await self._update_atr(symbol)
                await self._update_ema(symbol)
            await asyncio.sleep(300)  # 5分钟更新一次

    async def _update_atr(self, symbol: str):
        """更新ATR波动率"""
        try:
            klines = self.okx_client.get_kline(symbol, "1H", limit=self._atr_period + 1)
            if len(klines) < self._atr_period + 1:
                return

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

            # 保存ATR和波动率比率
            current_price = closes[-1]
            volatility_ratio = atr / current_price if current_price > 0 else 0

            self._atr_cache[symbol] = volatility_ratio
        except Exception as e:
            logger.debug(f"Failed to update ATR for {symbol}: {e}")

    async def _update_ema(self, symbol: str):
        """更新EMA判断市场状态"""
        try:
            klines = self.okx_client.get_kline(symbol, "1D", limit=self._ema_slow_period + 5)
            if len(klines) < self._ema_slow_period:
                return

            closes = np.array([float(kline[4]) for kline in klines])
            if np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
                return

            fast_ema = self._calculate_ema(closes, self._ema_fast_period)
            slow_ema = self._calculate_ema(closes, self._ema_slow_period)

            if fast_ema is None or slow_ema is None:
                return

            # 判断市场状态
            if fast_ema > slow_ema * 1.02:  # 快线比慢线高2%以上
                self._market_state[symbol] = "bull"
            elif fast_ema < slow_ema * 0.98:  # 快线比慢线低2%以上
                self._market_state[symbol] = "bear"
            else:
                self._market_state[symbol] = "neutral"

            self._ema_cache[symbol] = {
                "fast": fast_ema,
                "slow": slow_ema,
                "trend": self._market_state[symbol]
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

    def _get_dynamic_take_profit(self, layers: int) -> float:
        """动态止盈：层数越多，止盈比例越低"""
        if not self._dynamic_take_profit:
            return self._take_profit_base

        # 基础止盈 - 每层递减
        dynamic_tp = self._take_profit_base - (layers - 1) * self._take_profit_decay

        # 最低不低于0.5%
        return max(0.005, dynamic_tp)

    def _get_dynamic_drop_threshold(self, symbol: str, layers: int) -> float:
        """动态加仓阈值：波动率大时阈值更大"""
        if not self._dynamic_drop_threshold:
            return self._price_drop_pct

        volatility = self._atr_cache.get(symbol, 0)

        # 基础阈值 + 波动率调整
        # 波动率每增加1%，阈值增加0.5%
        adjusted = self._price_drop_pct + max(0, (volatility - self._volatility_threshold_low) * 0.5)

        # 层数越高，阈值越大（避免过度加仓）
        adjusted += (layers - 1) * 0.005

        # 限制范围
        return max(0.015, min(0.05, adjusted))

    def _fetch_recent_fills_from_okx(self, symbol: str, days: int = 7) -> List[Dict[str, Any]]:
        """P0-2: 从 OKX /api/v5/trade/fills-history 拉最近成交记录，用于反推真实入场价"""
        try:
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

    def _reconcile_position_from_fills(self, symbol: str, holdings: float, current_price: float) -> Dict[str, Any]:
        """P0-2: 从 OKX fills-history 反推持仓状态。
        返回包含 avg_entry_price / base_price / last_buy_price / current_layers / total_cost_usdt 的字典。
        无法反推时返回空字典，由调用方走兜底逻辑。
        """
        fills = self._fetch_recent_fills_from_okx(symbol, days=7)
        buy_fills = [f for f in fills if str(f.get("side", "")).lower() == "buy"]
        if not buy_fills:
            return {}
        try:
            # 按时间正序（旧->新）
            buy_fills.sort(key=lambda f: f.get("fillTime", "0"))
            total_qty = 0.0
            total_value = 0.0
            first_price = 0.0
            last_price = 0.0
            for fill in buy_fills:
                qty = float(fill.get("fillSz", 0) or 0)
                px = float(fill.get("fillPx", 0) or 0)
                if qty <= 0 or px <= 0:
                    continue
                if total_qty == 0:
                    first_price = px  # 最早一笔作为 base_price
                total_qty += qty
                total_value += qty * px
                last_price = px
            if total_qty <= 0 or total_value <= 0:
                return {}
            avg_entry = total_value / total_qty
            # 估算层数：用最早一笔作为 base_layer_usdt 估算（粗略）
            base_layer_usdt = first_price * (total_qty / max(1, len(buy_fills)))
            return {
                "avg_entry_price": avg_entry,
                "base_price": first_price if first_price > 0 else avg_entry,
                "last_buy_price": last_price if last_price > 0 else avg_entry,
                "current_layers": max(1, len(buy_fills)),
                "total_cost_usdt": total_value,
                "base_layer_usdt": base_layer_usdt,
                "unreconciled": False,
            }
        except Exception as e:
            logger.debug(f"Spot Martingale {symbol}: fills reconciliation parse failed: {e}")
            return {}

    async def _load_active_positions(self, skip_if_loaded: bool = False):
        """P0-2: 重启状态恢复。
        1) 持久化已恢复 -> 仅校验 total_quantity vs 实际持仓
        2) 未恢复 -> 从 OKX fills-history 反推
        3) 都不可用 -> 当前价兜底 + unreconciled=True warning
        """
        for symbol in self._all_symbols:
            holdings = self._get_current_holdings(symbol)
            ticker = self.okx_client.get_ticker(symbol)
            current_price = float(ticker["last"]) if ticker else 0.0

            # 无持仓：清理任何残留状态
            if holdings <= 0:
                if symbol in self._active_positions:
                    logger.info(f"Spot Martingale {symbol}: no holdings, cleaning stale position state")
                    self._active_positions.pop(symbol, None)
                continue

            if ticker is None:
                logger.warning(f"Spot Martingale {symbol}: holdings={holdings} but no ticker, skip")
                continue

            # 1) 持久化已恢复 + symbol 在 _active_positions 中 -> 校验持仓量
            if skip_if_loaded and symbol in self._active_positions:
                pos = self._active_positions[symbol]
                recorded_qty = float(pos.get("total_quantity", 0) or 0)
                # P0-2: 不一致以实际持仓为准
                if abs(recorded_qty - holdings) / max(holdings, 1e-9) > 0.01:
                    logger.warning(
                        f"Spot Martingale {symbol}: holdings mismatch recorded={recorded_qty:.6f} actual={holdings:.6f}, using actual"
                    )
                    pos["total_quantity"] = holdings
                # 确保 max_price_since_open 字段存在（P0-3）
                if "max_price_since_open" not in pos:
                    pos["max_price_since_open"] = max(current_price, float(pos.get("base_price", current_price)))
                logger.info(
                    f"Spot Martingale {symbol}: restored from persistence qty={holdings:.6f} avg={pos.get('avg_entry_price', 0):.4f} layers={pos.get('current_layers', 1)}"
                )
                continue

            # 2) 未恢复 -> 从 fills-history 反推
            reconciled = self._reconcile_position_from_fills(symbol, holdings, current_price)
            if reconciled:
                self._active_positions[symbol] = {
                    "current_layers": reconciled["current_layers"],
                    "total_quantity": holdings,  # 以实际持仓为准
                    "avg_entry_price": reconciled["avg_entry_price"],
                    "base_price": reconciled["base_price"],
                    "last_buy_price": reconciled["last_buy_price"],
                    "base_layer_usdt": reconciled["base_layer_usdt"],
                    "total_cost_usdt": reconciled["total_cost_usdt"],
                    "max_price_since_open": max(current_price, reconciled["base_price"]),
                    "status": "active",
                    "create_time": datetime.now(),
                    "unreconciled": False,
                }
                logger.info(
                    f"Spot Martingale {symbol}: reconciled from fills qty={holdings:.6f} avg={reconciled['avg_entry_price']:.4f} layers={reconciled['current_layers']}"
                )
                continue

            # 3) 兜底：当前价 + unreconciled=True warning
            logger.warning(
                f"Spot Martingale {symbol}: no persistence/fills data, using current_price {current_price:.4f} as entry (unreconciled=True)"
            )
            self._active_positions[symbol] = {
                "current_layers": 1,
                "total_quantity": holdings,
                "avg_entry_price": current_price,
                "base_price": current_price,
                "last_buy_price": current_price,
                "base_layer_usdt": current_price * holdings,  # 兜底估算
                "total_cost_usdt": current_price * holdings,
                "max_price_since_open": current_price,
                "status": "active",
                "create_time": datetime.now(),
                "unreconciled": True,
            }

    async def _monitor_ticks(self):
        while True:
            for symbol in self._all_symbols:
                await self._process_tick(symbol)
            await asyncio.sleep(0.5)

    async def _process_tick(self, symbol: str):
        tick = self.redis_cache.get_tick(symbol)
        from_ws = True

        if not tick:
            tick = self._get_tick_rest(symbol)
            from_ws = False
            if not tick:
                return

        price = tick.price
        if price <= 0:
            return

        now = time.time()
        if now - self._last_check_time.get(symbol, 0) < self._check_interval:
            return
        self._last_check_time[symbol] = now

        if symbol in self._active_positions:
            await self._check_existing_position(symbol, price)
        else:
            await self._check_new_entry(symbol, price, tick)

    def _get_tick_rest(self, symbol: str) -> Optional[TickData]:
        now = time.time()
        cache_entry = self._tick_rest_cache.get(symbol)
        if cache_entry and (now - cache_entry[0]) < self._tick_rest_interval:
            return cache_entry[1]

        try:
            ticker = self.okx_client.get_ticker(symbol)
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

    def _compute_signal_quality(self, symbol: str) -> float:
        """P0-4: 计算开仓信号质量分数（0~1）。
        综合考虑：RSI 越低越优、市场状态、波动率是否在合理区间。
        """
        score = 0.0
        # 1) RSI 因子：RSI < 30 -> 0.5, RSI < 40 -> 0.35, RSI < 50 -> 0.2, 否则 0
        rsi = self._rsi_cache.get(symbol)
        if rsi is not None:
            if rsi < 30:
                score += 0.5
            elif rsi < 40:
                score += 0.35
            elif rsi < 50:
                score += 0.2
        # 2) 市场状态：neutral > bull > bear（已由 _bear_market_suspend 拦截，这里只做加分）
        market_state = self._market_state.get(symbol, "neutral")
        if market_state == "neutral":
            score += 0.2
        elif market_state == "bull":
            score += 0.3
        # 3) 波动率在 [low, high] 区间加分
        volatility = self._atr_cache.get(symbol, 0)
        if 0 < volatility and self._volatility_threshold_low <= volatility <= self._volatility_threshold_high:
            score += 0.2
        return max(0.0, min(1.0, score))

    async def _check_new_entry(self, symbol: str, price: float, tick: TickData) -> bool:
        # 优化：熊市暂停开仓
        if self._bear_market_suspend:
            market_state = self._market_state.get(symbol, "neutral")
            if market_state == "bear":
                self._log_throttled(symbol, f"Spot Martingale {symbol}: suspended in bear market")
                return False

        # 优化：波动率过高暂停开仓
        if self._volatility_adaptive:
            volatility = self._atr_cache.get(symbol, 0)
            if volatility > self._volatility_threshold_high:
                self._log_throttled(symbol, f"Spot Martingale {symbol}: suspended due to high volatility {volatility:.2%}")
                return False

        # P0-3: 止损冷却期内禁止开仓
        cooldown_until = self._cooldown_until.get(symbol)
        if cooldown_until and datetime.now() < cooldown_until:
            self._log_throttled(symbol, f"Spot Martingale {symbol}: in cooldown until {cooldown_until.isoformat()}")
            return False

        # P0-4: 同币种开仓冷却
        last_open = self._last_open_time.get(symbol)
        if last_open and (datetime.now() - last_open).total_seconds() < self._same_symbol_cooldown_minutes * 60:
            self._log_throttled(symbol, f"Spot Martingale {symbol}: same-symbol cooldown active")
            return False

        # P0-4: RSI(1H) < 阈值 才允许开仓
        rsi = self._rsi_cache.get(symbol)
        if rsi is not None and rsi >= self._rsi_entry_threshold:
            self._log_throttled(symbol, f"Spot Martingale {symbol}: RSI {rsi:.1f} >= {self._rsi_entry_threshold}, skip entry")
            return False

        # 优化：每日交易次数限制
        daily_count = self._daily_trades.get(symbol, 0)
        if daily_count >= self._max_daily_trades_per_symbol:
            return False

        if not await self._confirm_entry(symbol, tick):
            return False

        # P0-4: 信号质量门槛
        signal_quality = self._compute_signal_quality(symbol)
        if signal_quality < self._min_signal_quality:
            self._log_throttled(
                symbol,
                f"Spot Martingale {symbol}: signal quality {signal_quality:.2f} < min {self._min_signal_quality}"
            )
            return False

        position = await self._calculate_base_position(symbol, price)
        if not position:
            return False

        # 优化：全策略总持仓上限动态调整，避免已有大仓位阻塞新加仓
        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        total_exposure = sum(float(p.get("total_cost_usdt", 0) or 0) for p in self._active_positions.values())
        base_cap = trading_capital * allocation * 0.8
        # 若已有持仓超过 base_cap，则 cap 上浮 10%，允许策略继续管理加仓
        effective_cap = max(base_cap, total_exposure * 1.1)
        # 新币种开仓时不检查总 exposure cap，避免单一大仓位阻塞其他币种开仓
        is_new_symbol = symbol not in self._active_positions
        if not is_new_symbol and total_exposure + position["usdt_value"] > effective_cap:
            self._log_throttled(
                symbol,
                f"Spot Martingale {symbol}: total exposure {total_exposure:.2f} + {position['usdt_value']:.2f} > cap {effective_cap:.2f}"
            )
            return False

        await self._place_martingale_order(symbol, "buy", price, position["quantity"], 1)

        # P0-1: 初始化 base_layer_usdt + total_cost_usdt
        self._active_positions[symbol] = {
            "current_layers": 1,
            "total_quantity": position["quantity"],
            "avg_entry_price": price,
            "base_price": price,
            "last_buy_price": price,
            "base_layer_usdt": position["usdt_value"],  # P0-1: 第一层投入金额
            "total_cost_usdt": position["usdt_value"],  # P0-5: 累计投入
            "max_price_since_open": price,  # P0-3: 持仓期最高价
            "status": "active",
            "create_time": datetime.now(),
            "unreconciled": False,
        }

        # 记录交易次数 + 开仓时间
        self._daily_trades[symbol] = daily_count + 1
        self._last_open_time[symbol] = datetime.now()

        self._log_throttled(
            symbol,
            f"Spot Martingale new entry: {symbol} @ {price:.4f} qty={position['quantity']:.6f} quality={signal_quality:.2f}"
        )
        return True

    async def _confirm_entry(self, symbol: str, tick: TickData) -> bool:
        if tick.volume and tick.volume > 0:
            klines = self.okx_client.get_kline(symbol, "1H", limit=24)
            if len(klines) >= 12:
                recent_vols = [float(kline[5]) for kline in klines[-12:]]
                avg_vol = sum(recent_vols) / len(recent_vols) if recent_vols else 0
                if avg_vol > 0 and tick.volume < avg_vol * 24 * 0.3:
                    return False

        return True

    async def _calculate_base_position(self, symbol: str, price: float) -> Optional[Dict[str, float]]:
        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()

        base_usdt = trading_capital * allocation * self._base_position_ratio

        # 小账户适配：放大单笔 base 仓位，确保满足最小手数和加仓需求
        if total_capital < 1500:
            small_cap_multiplier = 3.0 if total_capital < 300 else 2.0 if total_capital < 800 else 1.5
            base_usdt *= small_cap_multiplier

        # 注入空闲资金放大乘数（来自 AdaptiveController）
        if self._adaptive_controller:
            try:
                boost = self._adaptive_controller.get_position_boost()
                if boost > 1.0:
                    base_usdt *= boost
            except Exception:
                pass

        min_margin = self.config["trading"].get("min_margin_per_trade", 0.5)

        if base_usdt < min_margin:
            logger.debug(f"Spot Martingale {symbol}: base_usdt {base_usdt:.4f} < min_margin {min_margin}")
            return None

        # 单币种 exposure 上限：trading_capital * allocation * 1.2（允许加仓）
        max_symbol_exposure_usdt = trading_capital * allocation * 1.2
        existing_exposure = float(self._active_positions.get(symbol, {}).get("total_cost_usdt", 0) or 0)
        if existing_exposure + base_usdt > max_symbol_exposure_usdt:
            logger.debug(
                f"Spot Martingale {symbol}: symbol exposure {existing_exposure:.2f} + {base_usdt:.2f} > cap {max_symbol_exposure_usdt:.2f}"
            )
            return None

        available_balance = self._get_available_usdt_balance()
        if available_balance < base_usdt:
            logger.debug(f"Spot Martingale {symbol}: insufficient USDT {available_balance:.2f} < required {base_usdt:.2f}")
            return None

        quantity = base_usdt / price

        min_lot_size = float(self.okx_client.get_instrument_info(symbol).get("lotSz", "0.001"))
        if quantity < min_lot_size:
            logger.debug(f"Spot Martingale {symbol}: quantity {quantity:.6f} < min lot {min_lot_size}")
            return None

        quantity_precision = self._get_quantity_precision(symbol)
        quantity = round(quantity, quantity_precision)

        return {"quantity": quantity, "usdt_value": quantity * price}

    async def _check_existing_position(self, symbol: str, price: float):
        pos = self._active_positions[symbol]
        if pos["status"] != "active":
            return

        avg_entry = pos["avg_entry_price"]
        base_price = pos["base_price"]
        last_buy_price = pos["last_buy_price"]
        current_layers = pos["current_layers"]
        total_quantity = pos["total_quantity"]

        # === P0-3: 更新持仓期最高价 ===
        max_price_since_open = float(pos.get("max_price_since_open", price) or price)
        if price > max_price_since_open:
            pos["max_price_since_open"] = price
            max_price_since_open = price
        # 同步最新持仓量（外部可能加减仓）
        actual_holdings = self._get_current_holdings(symbol)
        if actual_holdings > 0 and abs(actual_holdings - total_quantity) / max(total_quantity, 1e-9) > 0.01:
            pos["total_quantity"] = actual_holdings
            total_quantity = actual_holdings

        # 优化：动态止盈比例
        take_profit_pct = self._get_dynamic_take_profit(current_layers)

        rise_from_avg = (price - avg_entry) / avg_entry if avg_entry > 0 else 0
        drop_from_last = (last_buy_price - price) / last_buy_price if last_buy_price > 0 else 0
        drop_from_base = (base_price - price) / base_price if base_price > 0 else 0
        # P0-3: 相对均价跌幅 + 从持仓期最高点回撤
        drop_from_avg = (avg_entry - price) / avg_entry if avg_entry > 0 else 0
        drawdown_from_peak = (price - max_price_since_open) / max_price_since_open if max_price_since_open > 0 else 0

        # 优化：最大持仓时间检查
        create_time = pos.get("create_time", datetime.now())
        hold_hours = (datetime.now() - create_time).total_seconds() / 3600
        if hold_hours >= self._max_hold_time_hours:
            logger.warning(f"Spot Martingale max hold time exceeded: {symbol} held {hold_hours:.1f}h")
            await self._close_position(symbol, price, "max_hold_time")
            return

        # 优化：分批止盈
        if self._partial_close_enabled and current_layers >= self._partial_close_threshold:
            if rise_from_avg >= take_profit_pct * 0.7:  # 达到70%止盈目标时先平一半
                partial_qty = total_quantity * self._partial_close_ratio
                await self._partial_close(symbol, price, partial_qty, current_layers)
                return

        # 止盈检查
        if rise_from_avg >= take_profit_pct:
            await self._close_position(symbol, price, "take_profit")
            return

        # === P1: 绝对金额硬止损 ===
        unrealized_pnl = total_quantity * (price - avg_entry)
        if unrealized_pnl < -50.0:
            logger.warning(f"Spot Martingale hard stop: {symbol} unrealized PnL {unrealized_pnl:.2f} USDT")
            await self._close_position(symbol, price, "hard_stop_loss")
            return

        # === P0-3: 多重止损（任一触发即止损） ===
        stop_loss_triggered = False
        stop_loss_reason = ""
        # 1) 原逻辑：drop_from_base >= 0.15
        if drop_from_base >= self._stop_loss_pct:
            stop_loss_triggered = True
            stop_loss_reason = f"drop_from_base {drop_from_base:.2%} >= {self._stop_loss_pct:.2%}"
        # 2) 相对均价跌 10%
        elif drop_from_avg >= 0.10:
            stop_loss_triggered = True
            stop_loss_reason = f"drop_from_avg {drop_from_avg:.2%} >= 10.00%"
        # 3) 从持仓期最高点回撤 8%
        elif drawdown_from_peak <= -0.08:
            stop_loss_triggered = True
            stop_loss_reason = f"drawdown_from_peak {drawdown_from_peak:.2%} <= -8.00%"

        if stop_loss_triggered:
            logger.warning(f"Spot Martingale stop loss triggered: {symbol} {stop_loss_reason}")
            await self._close_position(symbol, price, "stop_loss")
            # P0-3: 止损后加 2 小时冷却期
            self._cooldown_until[symbol] = datetime.now() + timedelta(hours=2)
            return

        # 加仓检查 - 使用动态阈值
        dynamic_drop_threshold = self._get_dynamic_drop_threshold(symbol, current_layers)
        if current_layers < self._max_layers and drop_from_last >= dynamic_drop_threshold:
            await self._add_martingale_layer(symbol, price, current_layers, total_quantity, avg_entry)

    async def _add_martingale_layer(self, symbol: str, price: float, current_layers: int, total_quantity: float, avg_entry: float):
        pos = self._active_positions[symbol]
        # === P0-1: 修复加仓公式数学错误 ===
        # 原代码 base_usdt = total_quantity * avg_entry / current_layers 在加仓后 total_quantity 已变，
        # 且 avg_entry 是加权均价，不能用此反推首层金额。改为记录 base_layer_usdt。
        base_layer_usdt = float(pos.get("base_layer_usdt", 0) or 0)
        if base_layer_usdt <= 0:
            # 兜底：用当前持仓总量 / 层数估算（仅当历史状态丢失时）
            base_layer_usdt = (total_quantity * avg_entry) / max(1, current_layers)
            pos["base_layer_usdt"] = base_layer_usdt

        # current_layers 从 1 起算：layer 2 = base * coeff^1, layer 3 = base * coeff^2 ...
        layer_usdt = base_layer_usdt * (self._martingale_coefficient ** current_layers)

        # 优化：波动率自适应 - 高波动时减少加仓量
        if self._volatility_adaptive:
            volatility = self._atr_cache.get(symbol, 0)
            if volatility > self._volatility_threshold_low:
                # 波动率每超1%，加仓量减少10%
                reduction = min(0.5, (volatility - self._volatility_threshold_low) * 10)
                layer_usdt *= (1 - reduction)

        # === P0-5: 单币种 exposure 上限 ===
        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self.config["trading"]["trading_capital_ratio"]
        allocation = self._get_allocation()
        exposure_ratio = self.config["strategies"].get("spot_martingale", {}).get("max_symbol_exposure_ratio", 0.6)
        max_symbol_exposure_usdt = trading_capital * allocation * exposure_ratio
        existing_cost = float(pos.get("total_cost_usdt", 0) or 0)
        if existing_cost + layer_usdt > max_symbol_exposure_usdt:
            logger.debug(
                f"Spot Martingale {symbol}: layer {current_layers + 1} rejected, exposure {existing_cost:.2f} + {layer_usdt:.2f} > cap {max_symbol_exposure_usdt:.2f}"
            )
            return

        # === P0-1: 累计 USDT 投入上限：sum(layer_usdts) <= trading_capital * allocation * 0.6 ===
        cumulative_cap = trading_capital * allocation * 0.6
        if existing_cost + layer_usdt > cumulative_cap:
            logger.debug(
                f"Spot Martingale {symbol}: layer {current_layers + 1} rejected, cumulative {existing_cost:.2f} + {layer_usdt:.2f} > cap {cumulative_cap:.2f}"
            )
            return

        available_balance = self._get_available_usdt_balance()
        if available_balance < layer_usdt:
            logger.debug(f"Spot Martingale {symbol}: insufficient USDT for layer {current_layers + 1}: {available_balance:.2f} < {layer_usdt:.2f}")
            return

        new_quantity = layer_usdt / price
        min_lot_size = float(self.okx_client.get_instrument_info(symbol).get("lotSz", "0.001"))
        if new_quantity < min_lot_size:
            logger.debug(f"Spot Martingale {symbol}: new quantity {new_quantity:.6f} < min lot {min_lot_size}")
            return

        quantity_precision = self._get_quantity_precision(symbol)
        new_quantity = round(new_quantity, quantity_precision)

        await self._place_martingale_order(symbol, "buy", price, new_quantity, current_layers + 1)

        new_total = total_quantity + new_quantity
        new_avg = (total_quantity * avg_entry + new_quantity * price) / new_total

        # 保留原 pos 的引用，仅更新字段（避免丢失 max_price_since_open / base_layer_usdt 等）
        pos["current_layers"] = current_layers + 1
        pos["total_quantity"] = new_total
        pos["avg_entry_price"] = new_avg
        pos["last_buy_price"] = price
        pos["status"] = "active"
        # P0-5: 累计投入
        pos["total_cost_usdt"] = existing_cost + (new_quantity * price)
        # create_time 保留原值
        if "create_time" not in pos:
            pos["create_time"] = datetime.now()

        self._log_throttled(
            symbol,
            f"Spot Martingale layer {current_layers + 1}: {symbol} @ {price:.4f} qty={new_quantity:.6f}, avg={new_avg:.4f}, layer_usdt={layer_usdt:.2f}, total_cost={pos['total_cost_usdt']:.2f}"
        )

    async def _partial_close(self, symbol: str, price: float, quantity: float, layers: int):
        """分批平仓"""
        min_lot_size = float(self.okx_client.get_instrument_info(symbol).get("lotSz", "0.001"))
        if quantity < min_lot_size:
            return

        quantity_precision = self._get_quantity_precision(symbol)
        quantity = round(quantity, quantity_precision)

        await self._place_martingale_order(symbol, "sell", price, quantity, layers)

        # 更新持仓
        if symbol in self._active_positions:
            pos = self._active_positions[symbol]
            old_qty = float(pos.get("total_quantity", 0) or 0)
            remaining_qty = old_qty - quantity
            if remaining_qty > 0:
                pos["total_quantity"] = remaining_qty
                pos["current_layers"] = max(1, layers - 1)
                # P2: 保持原始avg_entry_price不变，不缩放total_cost
                pos["total_cost_usdt"] = pos.get("avg_entry_price", price) * remaining_qty
            else:
                del self._active_positions[symbol]

        logger.info(f"Spot Martingale partial close: {symbol} @ {price:.4f} x {quantity:.6f}")

    async def _close_position(self, symbol: str, price: float, reason: str):
        pos = self._active_positions[symbol]
        quantity = pos["total_quantity"]

        holdings = self._get_current_holdings(symbol)
        if holdings < quantity:
            quantity = holdings

        if quantity <= 0:
            logger.debug(f"Spot Martingale {symbol}: no holdings to close")
            del self._active_positions[symbol]
            return

        min_lot_size = float(self.okx_client.get_instrument_info(symbol).get("lotSz", "0.001"))
        if quantity < min_lot_size:
            logger.debug(f"Spot Martingale {symbol}: quantity {quantity:.6f} < min lot {min_lot_size}")
            del self._active_positions[symbol]
            return

        quantity_precision = self._get_quantity_precision(symbol)
        quantity = round(quantity, quantity_precision)

        await self._place_martingale_order(symbol, "sell", price, quantity, pos["current_layers"])

        avg_entry = pos["avg_entry_price"]
        profit = (price - avg_entry) * quantity
        profit_pct = (profit / (avg_entry * quantity)) * 100 if avg_entry > 0 else 0

        logger.info(f"Spot Martingale close ({reason}): {symbol} exit={price:.4f} entry={avg_entry:.4f}, profit={profit:.4f} ({profit_pct:.2f}%), layers={pos['current_layers']}")

        del self._active_positions[symbol]
        # 止损冷却由调用方（_check_existing_position）设置；这里清理同币种开仓冷却以便重新建仓后立即可用
        # 注意：不清理 _cooldown_until，它由止损路径显式设置

    async def _place_martingale_order(self, symbol: str, side: str, price: float, quantity: float, layer: int):
        # 企业级：参数前置校验，非法输入直接拒绝并埋点
        if not self._validate_symbol(symbol) or not self._validate_direction(side) \
                or not self._validate_price(price) or not self._validate_quantity(quantity):
            logger.warning(
                f"Martingale order rejected: invalid params "
                f"symbol={symbol!r} side={side!r} price={price!r} quantity={quantity!r}"
            )
            self._increment_metric("spot_martingale_signal_rejected_total", 1.0, {"reason": "invalid_params", "symbol": symbol})
            return

        precision = get_price_precision(symbol)
        quantity_precision = self._get_quantity_precision(symbol)

        quantity = round(quantity, quantity_precision)
        price = round(price, precision)

        confidence = 0.6 + (1 - layer * 0.05)

        signal = Signal(
            symbol=symbol,
            strategy_name="spot_martingale",
            signal_type="spot_martingale_trade",
            direction=side,
            price=price,
            quantity=quantity,
            leverage=1,
            stop_loss=None,
            take_profit=None,
            confidence=confidence,
            timestamp=datetime.now()
        )

        if self._signal_callback:
            await self._signal_callback(signal)

        self._record_metric("spot_martingale_signal_generated_total", 1.0, {"symbol": symbol, "direction": side, "layer": str(layer)})
        self._record_metric("spot_martingale_signal_confidence", confidence, {"symbol": symbol, "direction": side})

    def _log_throttled(self, symbol: str, message: str):
        """日志节流"""
        now = time.time()
        last_log = self._last_log_time.get(symbol, 0)

        if now - last_log >= self._log_throttle_interval:
            logger.info(message)
            self._last_log_time[symbol] = now
        else:
            logger.debug(message)

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

    def _get_quantity_precision(self, symbol: str) -> int:
        try:
            info = self.okx_client.get_instrument_info(symbol)
            lot_size = float(info.get("lotSz", "0.001"))
            return len(str(lot_size).split(".")[1]) if "." in str(lot_size) else 0
        except Exception:
            return 3

    async def _check_positions_loop(self):
        while True:
            await asyncio.sleep(30)
            for symbol in list(self._active_positions.keys()):
                holdings = self._get_current_holdings(symbol)
                if holdings <= 0:
                    logger.info(f"Spot Martingale {symbol}: no holdings detected, cleaning up")
                    del self._active_positions[symbol]

    # === P0-6: 状态持久化 ===
    def collect_persistent_state(self) -> Dict[str, Any]:
        """收集需要持久化的策略状态。"""
        # 将 datetime 序列化为 isoformat 字符串
        def _serialize_pos(p: Dict[str, Any]) -> Dict[str, Any]:
            out = {}
            for k, v in p.items():
                if isinstance(v, datetime):
                    out[k] = v.isoformat()
                else:
                    out[k] = v
            return out

        return {
            "active_positions": {sym: _serialize_pos(p) for sym, p in self._active_positions.items()},
            "daily_trades": dict(self._daily_trades),
            "daily_reset_date": self._daily_reset_date,
            "cooldown_until": {
                sym: ts.isoformat() if isinstance(ts, datetime) else ts
                for sym, ts in self._cooldown_until.items()
            },
        }

    def restore_persistent_state(self, state: Dict[str, Any]):
        """从持久化数据恢复策略状态。"""
        try:
            ap = state.get("active_positions")
            if isinstance(ap, dict):
                # 反序列化 datetime 字段
                datetime_fields = {"create_time", "start_time"}
                for sym, p in ap.items():
                    if not isinstance(p, dict):
                        continue
                    for k in list(p.keys()):
                        if k in datetime_fields and isinstance(p[k], str):
                            try:
                                p[k] = datetime.fromisoformat(p[k])
                            except (ValueError, TypeError):
                                pass
                    self._active_positions[sym] = p
            dt = state.get("daily_trades")
            if isinstance(dt, dict):
                self._daily_trades = {k: int(v) for k, v in dt.items() if v is not None}
            drd = state.get("daily_reset_date")
            if isinstance(drd, str):
                self._daily_reset_date = drd
            cu = state.get("cooldown_until")
            if isinstance(cu, dict):
                for sym, ts in cu.items():
                    if isinstance(ts, str):
                        try:
                            self._cooldown_until[sym] = datetime.fromisoformat(ts)
                        except (ValueError, TypeError):
                            pass
                    elif isinstance(ts, datetime):
                        self._cooldown_until[sym] = ts
            logger.info(
                f"Spot Martingale state restored: positions={len(self._active_positions)}, daily_trades={len(self._daily_trades)}, cooldowns={len(self._cooldown_until)}"
            )
        except Exception as e:
            logger.error(f"Spot Martingale restore_persistent_state failed: {e}")