"""
多币种动量轮动策略（资金分散策略 C）

逻辑：定时周期计算品种涨跌幅，做多强势币种、剔除弱势币种。
用途：多合约分散持仓，降低单一币种风险。

出场：品种掉出强势榜（或由强转弱）时轮动平仓。
标的：默认 tier1+tier2（BTC/ETH 及主流山寨）。
"""
from __future__ import annotations

from typing import Any, Dict, List

from loguru import logger

from strategies._trend_base import TrendStrategyBase
from strategies.trend_sub_strategies import rolling_return, rank_momentum


class MomentumRotationStrategy(TrendStrategyBase):
    STRATEGY_KEY = "momentum_rotation"
    DISPLAY_NAME = "多币种动量轮动"
    SYMBOL_SCOPE = "tier1_tier2"
    DEFAULT_BAR = "1h"
    ENTRY_SIGNAL_TYPE = "momentum_rotation_entry"
    EXIT_SIGNAL_TYPE = "momentum_rotation_exit"

    def __init__(self, config, okx_client, redis_cache):
        super().__init__(config, okx_client, redis_cache)
        self._lookback = self._safe_int(self._cfg.get("lookback", 24), 24)
        self._top_n = self._safe_int(self._cfg.get("top_n", 3), 3)
        self._bottom_n = self._safe_int(self._cfg.get("bottom_n", 2), 2)
        self._min_long_return = self._safe_float(self._cfg.get("min_long_return", 0.0), 0.0)
        self._enable_short = bool(self._cfg.get("enable_short", False))
        self._rotation_count = 1

    # ------------------------------------------------------------------
    # 资金分配：动量轮动按目标持仓数均分单笔资金
    # ------------------------------------------------------------------
    def _get_allocation(self) -> float:
        base = super()._get_allocation()
        if self._rotation_count > 1:
            return base / self._rotation_count
        return base

    async def _compute_returns(self) -> Dict[str, float]:
        returns: Dict[str, float] = {}
        for symbol in self._symbols:
            klines = await self._fetch_klines(symbol, limit=self._lookback + 2)
            if len(klines) < self._lookback + 1:
                continue
            _, _, _, closes, _ = self._klines_to_arrays(klines)
            returns[symbol] = rolling_return(closes, self._lookback)
        return returns

    async def _check_signals(self):
        returns = await self._compute_returns()
        if not returns:
            return

        ranked = rank_momentum(returns, self._top_n, self._bottom_n)
        longs = [(s, r) for s, r in ranked["long"] if r >= self._min_long_return]
        shorts = ranked["short"] if self._enable_short else []

        open_count = sum(1 for p in self._position_state.values() if p.get("status") == "open")

        for symbol, ret in longs:
            if self._position_state.get(symbol, {}).get("status") == "open":
                continue
            if open_count >= self._max_concurrent_positions:
                break
            price = await self._get_current_price(symbol)
            if price <= 0:
                continue
            confidence = min(0.95, 0.5 + abs(ret) * 5.0)
            if confidence < self._min_signal_quality:
                continue
            confidence = await self._funding_enhancer.adjust_entry_confidence(symbol, "long", confidence)
            if confidence < self._min_signal_quality:
                continue
            self._rotation_count = max(len(longs), 1)
            quantity, _ = self._calculate_quantity(symbol, price)
            if quantity <= 0:
                continue
            self._publish_entry_signal(symbol, "long", price, quantity, None, None, confidence)
            open_count += 1

        for symbol, ret in shorts:
            if self._position_state.get(symbol, {}).get("status") == "open":
                continue
            if open_count >= self._max_concurrent_positions:
                break
            price = await self._get_current_price(symbol)
            if price <= 0:
                continue
            confidence = min(0.95, 0.5 + abs(ret) * 5.0)
            if confidence < self._min_signal_quality:
                continue
            confidence = await self._funding_enhancer.adjust_entry_confidence(symbol, "short", confidence)
            if confidence < self._min_signal_quality:
                continue
            self._rotation_count = max(len(shorts), 1)
            quantity, _ = self._calculate_quantity(symbol, price)
            if quantity <= 0:
                continue
            self._publish_entry_signal(symbol, "short", price, quantity, None, None, confidence)
            open_count += 1

    async def _manage_positions(self):
        """剔除掉出强势榜的品种（轮动平仓）。"""
        returns = await self._compute_returns()
        if not returns:
            return

        ranked = rank_momentum(returns, self._top_n, self._bottom_n)
        keep_long = {s for s, r in ranked["long"] if r >= self._min_long_return}
        keep_short = {s for s, _ in ranked["short"]} if self._enable_short else set()

        for symbol, pos in list(self._position_state.items()):
            if pos.get("status") != "open":
                continue
            direction = pos.get("direction")
            should_close = False
            if direction == "long" and symbol not in keep_long:
                should_close = True
            elif direction == "short" and symbol not in keep_short:
                should_close = True
            if not should_close:
                continue
            price = await self._get_current_price(symbol)
            if price <= 0:
                continue
            self._publish_exit_signal(symbol, direction, price,
                                      self._safe_float(pos.get("current_quantity"), 0.0),
                                      reason="rotation")
