"""
资金费率增强模块（通用附属模块，挂载全部策略）

根据当前/预期资金费率调整持仓时长与开平仓时机。
本模块不独立产生交易信号，只优化开仓置信度与出场时机。

设计约束（重要）：
- 类名不含 "Strategy" 后缀，也不继承 PersistentStrategy / BaseStrategy，
  确保不会被 StrategyDiscovery 自动发现为可独立交易的策略。
- 通过 okx_client 获取资金费率（async 优先，sync 兜底）。
- 所有方法均为辅助性质，调用方（各策略）自行决定是否采用其建议。
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from loguru import logger


class FundingRateEnhancer:
    """资金费率增强器 — 通用附属模块，不产生交易信号。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None, okx_client=None):
        self.config = config or {}
        self.okx_client = okx_client

        cfg = self.config.get("strategies", {}).get("funding_rate_enhancer", {})
        self.enabled = bool(cfg.get("enabled", True))

        # 多头：正费率高于阈值视为不利（多头向空头付费）
        self.long_penalty_threshold = float(cfg.get("long_penalty_threshold", 0.0003))
        self.long_penalty_high = float(cfg.get("long_penalty_high", 0.0008))
        # 多头：负费率低于阈值视为有利（多头收取费用）
        self.long_favor_threshold = float(cfg.get("long_favor_threshold", -0.0002))

        # 空头：负费率深于阈值视为不利（空头向多头付费）
        self.short_penalty_threshold = float(cfg.get("short_penalty_threshold", -0.0003))
        self.short_penalty_high = float(cfg.get("short_penalty_high", -0.0008))
        # 空头：正费率高于阈值视为有利（空头收取费用）
        self.short_favor_threshold = float(cfg.get("short_favor_threshold", 0.0002))

        # 置信度最大调整幅度（单方向）
        self.max_confidence_adjust = float(cfg.get("max_confidence_adjust", 0.15))
        # 持仓时长调整系数范围
        self.min_hold_multiplier = float(cfg.get("min_hold_multiplier", 0.5))
        self.max_hold_multiplier = float(cfg.get("max_hold_multiplier", 1.3))

        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = float(cfg.get("cache_ttl_seconds", 300.0))

    # ------------------------------------------------------------------
    # 资金费率获取
    # ------------------------------------------------------------------
    async def get_funding_rate(self, symbol: str) -> Optional[float]:
        """获取品种当前资金费率（带缓存，async 优先，sync 兜底）。"""
        if not self.okx_client:
            return None

        cached = self._cache.get(symbol)
        if cached and (time.time() - cached["ts"]) < self._cache_ttl:
            return cached["rate"]

        rate = None
        try:
            if hasattr(self.okx_client, "get_funding_rate_async"):
                data = await self.okx_client.get_funding_rate_async(symbol)
                if data:
                    rate = self._parse_rate(data)
        except Exception as e:
            logger.debug(f"[FundingRateEnhancer] async fetch failed for {symbol}: {e}")

        if rate is None:
            try:
                if hasattr(self.okx_client, "get_funding_rate_async"):
                    data = await self.okx_client.get_funding_rate_async(symbol)
                    if data:
                        rate = self._parse_rate(data)
            except Exception as e:
                logger.debug(f"[FundingRateEnhancer] sync fetch failed for {symbol}: {e}")

        if rate is not None:
            self._cache[symbol] = {"rate": rate, "ts": time.time()}
        return rate

    @staticmethod
    def _parse_rate(data: Dict[str, Any]) -> Optional[float]:
        raw = data.get("fundingRate")
        if raw is None:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    def get_funding_rate_sync(self, symbol: str) -> Optional[float]:
        """同步获取资金费率（带缓存），供同步代码路径复用统一逻辑。"""
        if not self.okx_client:
            return None

        cached = self._cache.get(symbol)
        if cached and (time.time() - cached["ts"]) < self._cache_ttl:
            return cached["rate"]

        rate = None
        try:
            if hasattr(self.okx_client, "get_funding_rate"):
                data = self.okx_client.get_funding_rate(symbol)
                if data:
                    rate = self._parse_rate(data)
        except Exception as e:
            logger.debug(f"[FundingRateEnhancer] sync fetch failed for {symbol}: {e}")

        if rate is not None:
            self._cache[symbol] = {"rate": rate, "ts": time.time()}
        return rate

    # ------------------------------------------------------------------
    # 费率质量评分（-1..1：负为不利，正为有利）
    # ------------------------------------------------------------------
    def _quality(self, direction: str, rate: float) -> float:
        """根据方向与费率计算质量分，范围 [-1, 1]。"""
        if direction == "long":
            # 负费率有利（多头收钱），正费率不利（多头付钱）
            if rate <= self.long_favor_threshold:
                return 1.0
            if rate >= self.long_penalty_high:
                return -1.0
            if rate >= self.long_penalty_threshold:
                # 线性插值：penalty_threshold → 0，penalty_high → -1
                span = max(self.long_penalty_high - self.long_penalty_threshold, 1e-12)
                return -(rate - self.long_penalty_threshold) / span
            # 中性区间 (favor_threshold, penalty_threshold)
            span = max(self.long_penalty_threshold - self.long_favor_threshold, 1e-12)
            return (self.long_penalty_threshold - rate) / span  # 越接近阈值越低，0~1
        else:  # short
            # 正费率有利（空头收钱），负费率不利（空头付钱）
            if rate >= self.short_favor_threshold:
                return 1.0
            if rate <= self.short_penalty_high:
                return -1.0
            if rate <= self.short_penalty_threshold:
                span = max(self.short_penalty_threshold - self.short_penalty_high, 1e-12)
                return (rate - self.short_penalty_threshold) / span
            # 中性区间 (penalty_threshold, favor_threshold)
            span = max(self.short_favor_threshold - self.short_penalty_threshold, 1e-12)
            return (rate - self.short_penalty_threshold) / span

    def get_quality_sync(self, symbol: str, direction: str) -> Optional[float]:
        """同步计算资金费率质量分（-1..1），获取失败返回 None。

        供同步打分路径（如综合信号质量评分）复用统一费率逻辑。
        """
        if not self.enabled:
            return None
        rate = self.get_funding_rate_sync(symbol)
        if rate is None:
            return None
        return self._quality(direction, rate)

    # ------------------------------------------------------------------
    # 对外辅助接口
    # ------------------------------------------------------------------
    async def adjust_entry_confidence(self, symbol: str, direction: str,
                                      confidence: float) -> float:
        """根据资金费率调整开仓置信度（择时开仓，不产生信号）。

        不利费率方向降低置信度，有利方向提高置信度，幅度受 max_confidence_adjust 限制。
        """
        if not self.enabled:
            return confidence

        rate = await self.get_funding_rate(symbol)
        if rate is None:
            return confidence

        quality = self._quality(direction, rate)
        adjust = quality * self.max_confidence_adjust
        adjusted = max(0.0, min(1.0, confidence + adjust))
        if abs(adjusted - confidence) > 1e-9:
            logger.debug(
                f"[FundingRateEnhancer] {symbol} {direction}: rate={rate:.6f} "
                f"quality={quality:.2f}, confidence {confidence:.2f} -> {adjusted:.2f}"
            )
        return adjusted

    async def get_hold_advice(self, symbol: str, direction: str,
                              holding_minutes: float = 0.0) -> Dict[str, Any]:
        """给出持仓时长建议：返回 {action, multiplier, reason}。

        action ∈ {"shorten", "extend", "normal"}，multiplier 为持仓时长调整系数。
        只优化出场时机，不产生交易信号。
        """
        default = {"action": "normal", "multiplier": 1.0,
                   "rate": None, "reason": "funding_rate_enhancer_disabled"}
        if not self.enabled:
            return default

        rate = await self.get_funding_rate(symbol)
        if rate is None:
            return default

        quality = self._quality(direction, rate)
        if quality <= -0.5:
            # 费率明显不利：缩短持仓，尽早了结避免持续支付资金费
            multiplier = self.min_hold_multiplier
            action = "shorten"
            reason = f"unfavorable_funding_rate({rate:.6f})"
        elif quality >= 0.5:
            # 费率明显有利：可延长持仓，持续收取资金费
            multiplier = self.max_hold_multiplier
            action = "extend"
            reason = f"favorable_funding_rate({rate:.6f})"
        else:
            multiplier = 1.0
            action = "normal"
            reason = f"neutral_funding_rate({rate:.6f})"

        return {"action": action, "multiplier": multiplier, "rate": rate, "reason": reason}

    async def should_favor_exit(self, symbol: str, direction: str,
                                unrealized_pnl: float = 0.0) -> bool:
        """判断是否应优先平仓（优化出场时机）。

        当费率方向对持仓不利、且持仓已无盈利保护时，倾向于提前了结。
        """
        advice = await self.get_hold_advice(symbol, direction)
        return advice.get("action") == "shorten" and unrealized_pnl <= 0.0
