"""
趋势类策略共享基类（A/B/E 复用）

提供统一的能力：
- 构造函数 (config, okx_client, redis_cache) 契约，匹配 StrategyFactory。
- 标的池构建（A/B/E 默认仅 BTC/ETH）。
- 资金费率增强 / 波动率突破过滤器挂载。
- 统一信号发布、仓位跟踪、状态持久化、资金/仓位计算。

命名约束：本模块以 `_` 开头，且类名 `TrendStrategyBase` 位于 `_trend_base.py`，
不会被 StrategyDiscovery 自动发现为独立策略（发现器会跳过 `_` 前缀模块）。
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from core.models import Signal
from core.event_id import EventIDGenerator
from core.direction_unifier import DirectionUnifier
from strategies.base import StrategyBase
from strategies.funding_rate_enhancer import FundingRateEnhancer
from strategies.volatility_breakout_filter import VolatilityBreakoutFilter


class TrendStrategyBase(StrategyBase):
    """A/B/E 共享基类；子类需设置策略标识并实现信号与持仓管理接口。"""

    STRATEGY_KEY: str = ""
    DISPLAY_NAME: str = ""
    SYMBOL_SCOPE: str = "tier1"  # tier1 | tier1_tier2 | all
    DEFAULT_BAR: str = "15m"
    ENTRY_SIGNAL_TYPE: str = "trend_entry"
    EXIT_SIGNAL_TYPE: str = "trend_exit"

    def __init__(self, config: Dict[str, Any], okx_client, redis_cache):
        super().__init__()
        self.config = config
        self.okx_client = okx_client
        self.redis_cache = redis_cache

        key = self.STRATEGY_KEY
        self._strategy_name = key
        self._cfg = config.get("strategies", {}).get(key, {}) or {}

        self._enabled = bool(self._cfg.get("enabled", False))
        self._bar = str(self._cfg.get("timeframe", self.DEFAULT_BAR))
        self._min_signal_quality = max(0.35, self._safe_float(self._cfg.get("min_signal_quality", 0.35), 0.35))
        self._leverage = self._safe_int(self._cfg.get("leverage", 5), 5)
        self._max_concurrent_positions = self._safe_int(self._cfg.get("max_concurrent_positions", 2), 2)
        self._capital_allocation = self._safe_float(self._cfg.get("capital_allocation", 0.10), 0.10)
        self._max_stop_loss_pct = self._safe_float(self._cfg.get("max_stop_loss_pct", 0.02), 0.02)
        self._take_profit_pct = self._safe_float(self._cfg.get("take_profit_pct", 0.04), 0.04)
        self._loop_interval = self._safe_float(self._cfg.get("loop_interval_seconds", 15.0), 15.0)  # R34: 60s→15s
        self._min_hold_minutes = self._safe_float(config.get("trading", {}).get("min_hold_minutes", 5), 5)
        # R3: 单币种保证金上限比例（与 grid 对齐），防止趋势策略垄断某币种
        self._single_symbol_ratio = self._safe_float(self._cfg.get("single_symbol_ratio", 0.25), 0.25)
        self._taker_fee_rate = self._safe_float(config.get("trading", {}).get("taker_fee_rate", 0.0005), 0.0005)

        self._symbols: List[str] = self._build_symbols()
        self._position_state: Dict[str, Dict[str, Any]] = {}
        self._last_signal_time: Dict[str, datetime] = {}
        # P33: 退出后冷却追踪 {symbol: exit_timestamp}
        self._last_exit_time: Dict[str, float] = {}
        self._post_exit_cooldown = self._safe_float(self._cfg.get("post_exit_cooldown", 300), 300)  # 默认5分钟

        self._adaptive_controller = None
        self._stop_loss_manager = None
        self._coordinator = None
        self._regime_engine = None
        self._position_manager = None
        self._funding_enhancer = FundingRateEnhancer(config, okx_client)
        self._vol_breakout_filter = VolatilityBreakoutFilter(config)

        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._save_task: Optional[asyncio.Task] = None

        self.init_state_persistence(key, redis_cache)

        self._capital_cache_value = 0.0
        self._capital_cache_ts = 0.0
        self._capital_cache_ttl = 30.0

        logger.info(f"[{key}] 初始化完成，标的={self._symbols}, bar={self._bar}, enabled={self._enabled}")

    # ------------------------------------------------------------------
    # 依赖注入
    # ------------------------------------------------------------------
    def set_adaptive_controller(self, controller):
        self._adaptive_controller = controller

    def set_stop_loss_manager(self, manager):
        self._stop_loss_manager = manager

    def set_coordinator(self, coordinator):
        self._coordinator = coordinator

    def set_regime_engine(self, engine):
        """注入 MarketRegimeEngine（供震荡/趋势类策略做状态过滤）。"""
        self._regime_engine = engine

    def set_position_manager(self, manager):
        """注入 PositionManager（供跨策略持仓相关性检查）。"""
        self._position_manager = manager

    def _check_cross_strategy_conflict(self, symbol: str, side: str) -> bool:
        """检查是否与已有跨策略持仓冲突（同 symbol 同 side）。

        返回 True 表示存在冲突，应跳过该信号。
        """
        if self._position_manager is None:
            return False
        try:
            allowed, reason = self._position_manager.would_create_cross_strategy_duplicate(
                symbol, side, self._strategy_name
            )
            if not allowed:
                logger.debug(
                    f"[{self._strategy_name}] 跨策略持仓冲突: {symbol} {side} — {reason}"
                )
                return True
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # 配置热更新
    # ------------------------------------------------------------------
    async def update_config(self, updates: Dict[str, Any]):
        """运行时热更新配置。"""
        strategy_cfg = self.config.get("strategies", {}).get(self._strategy_name, {})
        strategy_cfg.update(updates)
        self.config.setdefault("strategies", {})[self._strategy_name] = strategy_cfg

        attr_map = {
            "min_signal_quality": "_min_signal_quality",
            "leverage": "_leverage",
            "max_concurrent_positions": "_max_concurrent_positions",
            "capital_allocation": "_capital_allocation",
            "max_stop_loss_pct": "_max_stop_loss_pct",
            "take_profit_pct": "_take_profit_pct",
            "loop_interval_seconds": "_loop_interval",
        }
        int_attrs = {"leverage", "max_concurrent_positions"}
        for cfg_key, attr in attr_map.items():
            if cfg_key in updates:
                if cfg_key in int_attrs:
                    setattr(self, attr, self._safe_int(updates[cfg_key], getattr(self, attr)))
                else:
                    setattr(self, attr, self._safe_float(updates[cfg_key], getattr(self, attr)))
                logger.info(f"[{self._strategy_name}] config hot-updated: {attr}={updates[cfg_key]}")

        # 阈值硬下限：min_signal_quality 不得低于 0.35（A/B/E 趋势类信号质量门槛）
        self._min_signal_quality = max(0.35, self._safe_float(self._min_signal_quality, 0.35))

    # ------------------------------------------------------------------
    # 标的池
    # ------------------------------------------------------------------
    def _build_symbols(self) -> List[str]:
        currencies = self.config.get("currencies", {})
        bases: List[str] = []
        if self.SYMBOL_SCOPE == "tier1":
            bases = list(currencies.get("tier1_symbols", ["BTC", "ETH"]))
        elif self.SYMBOL_SCOPE == "tier1_tier2":
            bases = list(currencies.get("tier1_symbols", [])) + list(currencies.get("tier2_symbols", []))
        else:
            for tier in ("tier1", "tier2", "tier3"):
                bases += list(currencies.get(f"{tier}_symbols", []))
        return [f"{b}-USDT-SWAP" for b in bases if b]

    # ------------------------------------------------------------------
    # 资金 / 仓位
    # ------------------------------------------------------------------
    def _get_effective_capital(self) -> float:
        import time
        now = time.time()
        if self._capital_cache_value > 0 and (now - self._capital_cache_ts) < self._capital_cache_ttl:
            return self._capital_cache_value
        try:
            account_info = self.okx_client.get_account_info()
            if account_info:
                for detail in account_info.get("details", []):
                    if detail.get("ccy") == "USDT":
                        eq = self._safe_float(detail.get("eq"), 0.0)
                        if eq > 0:
                            self._capital_cache_value = eq
                            self._capital_cache_ts = now
                            return eq
                total_eq = self._safe_float(account_info.get("totalEq"), 0.0)
                if total_eq > 0:
                    self._capital_cache_value = total_eq
                    self._capital_cache_ts = now
                    return total_eq
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] get_account_info failed: {type(e).__name__}: {e}")
        logger.warning(f"[{self._strategy_name}] 账户权益查询失败，返回 0（fail-closed）")
        return 0.0

    def _get_allocation(self) -> float:
        if self._adaptive_controller:
            try:
                return self._adaptive_controller.get_allocation(self._strategy_name)
            except Exception as e:
                # fail-closed: 自适应资金分配查询失败时拒绝开仓，避免风险收缩失效
                logger.warning(f"[{self._strategy_name}] 资金分配查询失败，返回 0（fail-closed）: {e}")
                return 0.0
        return self._capital_allocation

    def _get_leverage(self, symbol: str) -> int:
        try:
            from configs.settings import get_currency_tier
            tier = get_currency_tier(symbol, self.config)
            tier_settings = self.config.get("currencies", {}).get(f"{tier}_settings", {})
            leverage = self._safe_int(tier_settings.get("leverage_max", self._leverage), self._leverage)
        except Exception:
            leverage = self._leverage
        abs_max = self._safe_int(self.config.get("leverage_tiers", {}).get("absolute_max", 5), 5)
        return min(max(leverage, 1), abs_max)

    def _calculate_quantity(self, symbol: str, price: float) -> Tuple[float, float]:
        """计算开仓数量（合约张数）与名义资金。

        返回 (contracts, base_position_usd)。数量已按 lot size 向下取整。
        """
        total_capital = self._get_effective_capital()
        trading_capital = total_capital * self._safe_float(
            self.config.get("trading", {}).get("trading_capital_ratio", 0.95), 0.95)
        allocation = self._get_allocation()
        base_position = trading_capital * allocation

        # 小资金适配：确保单笔保证金足够
        if total_capital < 500:
            if total_capital >= 200:
                base_position *= 3.0
            elif total_capital >= 100:
                base_position *= 4.0
            elif total_capital >= 50:
                base_position *= 6.0
            else:
                base_position *= 10.0

        # R3: 单币种保证金上限 — 该币种已有持仓保证金 + 新开仓保证金 ≤ total_capital * ratio
        if self._position_manager is not None:
            try:
                existing_positions = self._position_manager.get_positions_by_symbol(symbol)
                existing_margin = sum(
                    getattr(p, "margin", 0.0) or 0.0 for p in existing_positions
                )
                symbol_cap = total_capital * self._single_symbol_ratio
                available = max(0.0, symbol_cap - existing_margin)
                if base_position > available:
                    logger.debug(
                        f"[{self._strategy_name}] R3: {symbol} base_position {base_position:.2f} "
                        f"clipped to {available:.2f} (existing margin {existing_margin:.2f}, "
                        f"cap {symbol_cap:.2f})"
                    )
                    base_position = available
            except Exception:
                pass

        leverage = self._get_leverage(symbol)
        quantity_coin = base_position * leverage / price if price > 0 else 0.0

        contracts = 0.0
        try:
            contracts = self.okx_client.coin_to_contracts(symbol, quantity_coin)
            contracts = self.okx_client.round_quantity_to_lot(symbol, contracts, round_up=False)
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] quantity calc failed for {symbol}: {e}")

        return contracts, base_position

    # ------------------------------------------------------------------
    # K 线 / 价格
    # ------------------------------------------------------------------
    async def _fetch_klines(self, symbol: str, limit: int = 120) -> List[List[float]]:
        """获取正序 K 线（升序），返回 OKX 原始列表格式。"""
        try:
            klines = await self.okx_client.get_kline_async(symbol, self._bar, limit=limit)
            if not klines:
                return []
            return sorted(klines, key=lambda k: int(k[0]))
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] _fetch_klines failed for {symbol}: {e}")
            return []

    @staticmethod
    def _klines_to_arrays(klines: List[List[float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        opens = np.array([float(k[1]) for k in klines])
        highs = np.array([float(k[2]) for k in klines])
        lows = np.array([float(k[3]) for k in klines])
        closes = np.array([float(k[4]) for k in klines])
        volumes = np.array([float(k[5]) for k in klines])
        return opens, highs, lows, closes, volumes

    async def _get_current_price(self, symbol: str) -> float:
        try:
            ticker = await self.okx_client.get_ticker_async(symbol)
            if ticker:
                return self._safe_float(ticker.get("last"), 0.0)
        except Exception:
            pass
        try:
            ticker = await self.okx_client.get_ticker_async(symbol)
            if ticker:
                return self._safe_float(ticker.get("last"), 0.0)
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] get price failed for {symbol}: {e}")
        return 0.0

    def _round_price(self, symbol: str, price: Optional[float]) -> Optional[float]:
        if price is None or price <= 0:
            return price
        try:
            return self.okx_client.round_price_to_tick(symbol, price)
        except Exception:
            return price

    # ------------------------------------------------------------------
    # 信号发布
    # ------------------------------------------------------------------
    def _publish_entry_signal(self, symbol: str, direction: str, price: float,
                              quantity: float, stop_loss: Optional[float],
                              take_profit: Optional[float], confidence: float):
        # P2: 信号源头生成全链路 traceID（贯穿 信号→裁决→订单→记账），开仓/平仓复用同源
        trace_id = EventIDGenerator.get_instance().generate()
        signal = Signal(
            symbol=symbol,
            strategy_name=self._strategy_name,
            signal_type=self.ENTRY_SIGNAL_TYPE,
            direction=direction,
            price=price,
            quantity=quantity,
            leverage=self._get_leverage(symbol),
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            timestamp=datetime.now(),
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
                "timestamp": signal.timestamp.isoformat(),
                "trace_id": trace_id,
            },
        })

        self._position_state[symbol] = {
            "direction": direction,
            "entry_price": price,
            "current_quantity": quantity,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "status": "open",
            "entry_time": datetime.now(),
            "trace_id": trace_id,
        }
        self._last_signal_time[symbol] = datetime.now()
        self._record_metric("strategy_signal_generated_total", 1.0,
                            {"strategy": self._strategy_name, "symbol": symbol, "direction": direction})
        logger.info(f"[{self._strategy_name}] 开仓信号: {direction} {symbol} @ {price:.4f} "
                    f"qty={quantity} conf={confidence:.2f}")

    def _publish_exit_signal(self, symbol: str, direction: str, price: float,
                             quantity: float, reason: str = "exit"):
        signal_type = f"{self._strategy_name}_{reason}"  # 含 exit/close/reduce 关键词由 signal_processor 识别
        # P0: 平仓 direction 语义反转——本方法入参 direction 为「持仓方向」(long/short)，
        #     下游 order_executor 将平仓信号的 direction 解释为「平仓 side 归一化值」(与持仓相反)，
        #     故需反转为 opposite(direction)；同时显式下发 reduce_only/close_position，防止平仓被当开仓。
        close_dir = DirectionUnifier.opposite(direction)
        # P2: 复用开仓 traceID，实现开仓→平仓全链路同源追踪；缺失时生成新 ID
        existing = self._position_state.get(symbol) or {}
        trace_id = existing.get("trace_id") or EventIDGenerator.get_instance().generate()
        signal = Signal(
            symbol=symbol,
            strategy_name=self._strategy_name,
            signal_type=signal_type,
            direction=close_dir,
            price=price,
            quantity=quantity,
            leverage=self._get_leverage(symbol),
            confidence=1.0,
            timestamp=datetime.now(),
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
                "confidence": signal.confidence,
                "timestamp": signal.timestamp.isoformat(),
                "trace_id": trace_id,
                "reduce_only": True,
                "close_position": True,
            },
        })
        self._position_state.pop(symbol, None)
        self._last_signal_time[symbol] = datetime.now()
        # P33: 记录退出时间，用于冷却追踪
        self._last_exit_time[symbol] = datetime.now().timestamp()
        logger.info(f"[{self._strategy_name}] 平仓信号: {direction}→{close_dir} {symbol} @ {price:.4f} "
                    f"qty={quantity} reason={reason}")

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    async def start(self):
        if not self._enabled:
            logger.info(f"[{self._strategy_name}] 策略已禁用，跳过启动")
            return
        logger.info(f"[{self._strategy_name}] 启动中...")
        try:
            await self.load_state_async()
        except Exception as e:
            logger.warning(f"[{self._strategy_name}] 状态加载失败，使用默认状态: {e}")

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        self._save_task = asyncio.create_task(self.periodic_save_loop())

    async def stop(self):
        self._running = False
        for task in (self._monitor_task, self._save_task):
            if task and not task.done():
                task.cancel()
        # 取消后等待任务退出，确保最终状态被保存
        for task in (self._monitor_task, self._save_task):
            if task:
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass
        self._monitor_task = None
        self._save_task = None
        await self.save_state_async()
        logger.info(f"[{self._strategy_name}] 已停止")

    async def pause(self):
        self._running = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
        self._monitor_task = None
        logger.info(f"[{self._strategy_name}] 已暂停")

    async def resume(self):
        if self._enabled:
            self._running = True
            if not self._monitor_task or self._monitor_task.done():
                self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"[{self._strategy_name}] 已恢复")

    def get_health(self) -> Dict[str, Any]:
        return {
            "strategy": self._strategy_name,
            "enabled": self._enabled,
            "running": self._running,
            "symbols": len(self._symbols),
            "open_positions": sum(1 for p in self._position_state.values() if p.get("status") == "open"),
            "bar": self._bar,
        }

    async def _sync_positions_with_exchange(self):
        """同步交易所实际持仓，清理被外部（如减仓/爆仓）平掉的幽灵仓位。

        若缺少此步，_position_state 中残留的 open 状态会让 max_positions_reached
        永远成立，策略无法开新仓。
        """
        if not self._position_state:
            return
        try:
            positions = await self.okx_client.get_positions_async()
            if positions is None:
                return
            exchange_symbols = set()
            for p in positions:
                sym = p.get("instId", "")
                qty = abs(float(p.get("pos", 0) or 0))
                if sym and qty > 0:
                    exchange_symbols.add(sym)
            stale = []
            for sym in list(self._position_state.keys()):
                if self._position_state[sym].get("status") == "open" and sym not in exchange_symbols:
                    stale.append(sym)
                    del self._position_state[sym]
                    self._last_signal_time.pop(sym, None)
            if stale:
                logger.info(f"[{self._strategy_name}] removed {len(stale)} stale positions: {stale}")
        except Exception as e:
            logger.debug(f"[{self._strategy_name}] _sync_positions_with_exchange error: {e}")

    async def _monitor_loop(self):
        while self._running:
            try:
                await self._sync_positions_with_exchange()
                await self._check_signals()
                await self._manage_positions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{self._strategy_name}] 主循环异常: {e}")
            await asyncio.sleep(self._loop_interval)

    # ------------------------------------------------------------------
    # 子类接口
    # ------------------------------------------------------------------
    def _get_risk_params(self) -> Dict[str, Any]:
        """Return the normalized risk settings shared by trend strategies."""
        return {
            "leverage": self._leverage,
            "capital_allocation": self._capital_allocation,
            "max_concurrent_positions": self._max_concurrent_positions,
            "max_stop_loss_pct": self._max_stop_loss_pct,
            "take_profit_pct": self._take_profit_pct,
        }

    # ------------------------------------------------------------------
    # 状态持久化
    # ------------------------------------------------------------------
    def collect_persistent_state(self) -> Dict[str, Any]:
        return {
            "_position_state": self._position_state,
            "_last_signal_time": self._last_signal_time,
            "_last_exit_time": self._last_exit_time,
        }

    def restore_persistent_state(self, state: Dict[str, Any]):
        raw_pos = state.get("_position_state", {}) or {}
        self._position_state = {}
        for sym, ps in raw_pos.items():
            if isinstance(ps, dict):
                entry_time = ps.get("entry_time")
                if isinstance(entry_time, str):
                    try:
                        ps["entry_time"] = datetime.fromisoformat(entry_time)
                    except (ValueError, TypeError):
                        ps["entry_time"] = datetime.now()
                self._position_state[sym] = ps

        raw_ts = state.get("_last_signal_time", {}) or {}
        self._last_signal_time = {}
        for sym, ts in raw_ts.items():
            if isinstance(ts, datetime):
                self._last_signal_time[sym] = ts
            elif isinstance(ts, str):
                try:
                    self._last_signal_time[sym] = datetime.fromisoformat(ts)
                except (ValueError, TypeError):
                    pass

        # P33: 恢复退出冷却时间戳（json.dumps(default=str) 会将其转为字符串/数字）
        raw_exit = state.get("_last_exit_time", {}) or {}
        self._last_exit_time = {}
        for sym, ts in raw_exit.items():
            try:
                self._last_exit_time[sym] = float(ts)
            except (TypeError, ValueError):
                pass
