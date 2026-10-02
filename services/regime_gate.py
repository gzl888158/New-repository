"""
L0 市场状态总门控（RegimeGate）。

在策略信号进入审计（audit_signal）之前，按当前市场 regime 硬过滤不匹配的开仓信号，
从而在源头丢弃废信号、降低交易频率与磨损型亏损。

门控规则（对应 docs/enterprise_profit_strategy.md Section 5.1）：
- 趋势 regime（trend_bullish / trend_bearish）：仅放行 trend / arbitrage 趋势跟随类；
  均值回归类（grid/scalping/spot_grid/spot_martingale）开仓直接拒绝。
- 震荡 regime（range_bound）：均值回归类直接放行（震荡市即网格/短线本命行情）；
  趋势跟随类开仓拒绝（除非高置信度突破信号，见 range_bound_trend_allow_high_confidence）。
- 高波动/黑天鹅 regime（extreme_volatility / funding_crush / liquidity_crisis）：
  均值回归类开仓拒绝（只平仓）；趋势跟随类放行（仓位下调由既有 regime 仓位调整兜底）。
- 平仓/降仓/止损/止盈信号始终放行——平仓风险由 RiskGate 管理，而非 regime 门控。
- regime 未知（引擎未就绪）：放行，避免启动阶段误杀全部信号。

仓位下调（高波动降 50% 等）由 SignalProcessor._apply_dynamic_sizing 中既有的
MarketRegimeEngine.get_position_adjustment() 负责，本门控只做硬过滤，不做重复调仓。
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set

from loguru import logger

from utils.helpers import safe_float

# regime 字符串值（与 MarketRegime.value 一致）
TREND_REGIMES = {"trend_bullish", "trend_bearish"}
RANGE_REGIMES = {"range_bound"}
BREAKOUT_REGIMES = {"breakout", "breakdown", "reversal"}
HIGH_RISK_REGIMES = {"extreme_volatility", "funding_crush", "liquidity_crisis"}

# 兼容历史口径：部分模块可能仍然使用 trending_up / ranging / high_vol 等别名。
REGIME_ALIASES = {
    "trending_up": "trend_bullish",
    "trend_up": "trend_bullish",
    "trending_down": "trend_bearish",
    "trend_down": "trend_bearish",
    "ranging": "range_bound",
    "range": "range_bound",
    "high_vol": "extreme_volatility",
    "low_vol": "range_bound",
    "breakout": "breakout",
    "breakdown": "breakdown",
    "reversal": "reversal",
}

# 趋势跟随 / 套利类策略：仅在趋势 regime 放行
TREND_FOLLOWING_STRATEGIES = {"trend", "arbitrage"}
# 均值回归类策略：仅在震荡 regime 且符号在白名单内放行
MEAN_REVERSION_STRATEGIES = {"grid", "scalping", "spot_grid", "spot_martingale"}

# 平仓/降仓信号关键词
_CLOSE_KEYWORDS = (
    "close", "take_profit", "stop_loss", "reduce", "exit", "liquidation", "partial",
)


@dataclass
class GateResult:
    allowed: bool
    reason: str
    regime: str = "unknown"
    action: str = "allow"  # allow / reject


def _is_close_signal(signal_type: str) -> bool:
    t = (signal_type or "").lower()
    return any(k in t for k in _CLOSE_KEYWORDS)


class RegimeGate:
    """L0 总门控：按市场 regime 过滤不匹配的开仓信号。"""

    def __init__(
        self,
        regime_engine,
        whitelist_provider: Optional[Callable[[], Set[str]]] = None,
        config: Optional[Dict[str, Any]] = None,
        regime_arbiter=None,
        detector=None,
    ):
        self._regime_engine = regime_engine
        self._regime_arbiter = regime_arbiter
        self._detector = detector
        self._whitelist_provider = whitelist_provider
        self._config = config or {}
        # 放开 L0：range_bound 下 trend 跟随类凭高置信度突破信号放行（可配置开关）
        self._range_trend_allow_high_confidence = self._config.get(
            "range_bound_trend_allow_high_confidence", True
        )
        self._range_trend_confidence_threshold = self._config.get(
            "range_bound_trend_confidence_threshold", 0.60
        )
        # 趋势行情下均值回归策略（grid/scalping/spot_*）顺趋势方向放行开关：
        # 小账户资金利用率优化——趋势 regime 不再一刀切禁用均值回归，仅拦截逆势方向。
        self._trend_mr_allow_with_trend = self._config.get(
            "trend_regime_mean_reversion_allow_with_trend", True
        )
        # trend 策略逆势开仓拒绝的币种级趋势强度阈值：仅当 |strength| >= 该阈值时，
        # 逆势开仓（bearish 做多 / bullish 做空）才硬拒；弱趋势下不拦截（交给后续
        # 置信度/风控裁决），避免误杀反转初期信号。原为硬编码 0.6。
        self._trend_counter_direction_strength = safe_float(self._config.get(
            "trend_counter_direction_strength", 0.6
        ), 0.6)
        self._reversal_probability_threshold = safe_float(self._config.get(
            "reversal_probability_threshold", 0.65
        ), 0.65)
        self._reversal_signal_confidence_threshold = safe_float(self._config.get(
            "reversal_signal_confidence_threshold", 0.70
        ), 0.70)
        # 震荡磨损事前防护：range_bound 下均值回归策略开仓前，校验币种 24h 振幅
        # 是否足以覆盖开仓成本（双边 taker 0.1% + 滑点 + 点差 ≈ 0.18%）。振幅低于
        # 成本安全倍数时，窄幅震荡开仓必然被手续费磨损，源头拒绝，而非等磨损发生后
        # 由事后 WTI 统计兜底。
        self._range_bound_vol_check_enabled = self._config.get(
            "range_bound_volatility_check_enabled", True
        )
        self._range_bound_min_volatility = safe_float(self._config.get(
            "range_bound_min_volatility", 0.005
        ), 0.005)

    def _get_whitelist(self) -> Set[str]:
        if self._whitelist_provider is None:
            return set()
        try:
            wl = self._whitelist_provider()
            return set(wl) if wl else set()
        except Exception as e:
            logger.debug(f"RegimeGate whitelist provider error: {e}")
            return set()

    @staticmethod
    def _normalize_regime(value: Optional[str]) -> str:
        if value is None:
            return "unknown"
        raw = str(value).strip().lower()
        if not raw:
            return "unknown"
        return REGIME_ALIASES.get(raw, raw)

    def _resolve_regime_info(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """优先取融合后的 regiem/arbiter 输出，再回退主引擎。"""
        if self._regime_arbiter is not None:
            try:
                arbiter_result = self._regime_arbiter.arbitrate(symbol) if symbol else self._regime_arbiter.arbitrate()
                if isinstance(arbiter_result, dict) and arbiter_result.get("regime"):
                    return arbiter_result
            except Exception as e:
                logger.debug(f"RegimeGate arbiter lookup failed: {e}")

        if self._detector is not None:
            try:
                get_regime = getattr(self._detector, "get_regime", None)
                if callable(get_regime):
                    info = get_regime(symbol) if symbol else get_regime()
                    if isinstance(info, dict) and info.get("regime"):
                        return info
            except Exception as e:
                logger.debug(f"RegimeGate detector lookup failed: {e}")

        if self._regime_engine is not None:
            try:
                info = self._regime_engine.get_regime() or {}
                if isinstance(info, dict) and info.get("regime"):
                    return info
            except Exception as e:
                logger.debug(f"RegimeGate get_regime error: {e}")

        return None

    def _get_regime(self) -> str:
        info = self._resolve_regime_info()
        if info is None:
            return "unknown"
        regime = (info or {}).get("regime", "unknown")
        return self._normalize_regime(regime)

    def _get_symbol_regime_info(self, symbol: str):
        """返回 (regime, strength)。优先 symbol 级独立 regime，回退全局 regime。"""
        info = self._resolve_regime_info(symbol)
        if isinstance(info, dict) and info.get("regime"):
            regime = self._normalize_regime(info.get("regime", "unknown"))
            return (regime, safe_float(info.get("strength"), 0.0))

        if not self._regime_engine:
            return "unknown", 0.0
        try:
            data = self._regime_engine.get_symbol_regime(symbol)
            if data:
                regime = self._normalize_regime(data.get("regime", "unknown"))
                return (regime, safe_float(data.get("strength"), 0.0))
        except Exception as e:
            logger.debug(f"RegimeGate get_symbol_regime error: {e}")
        try:
            info = self._regime_engine.get_regime() or {}
            regime = self._normalize_regime(info.get("regime", "unknown"))
            return (regime, safe_float(info.get("strength"), 0.0))
        except Exception:
            return "unknown", 0.0

    def _get_symbol_volatility(self, symbol: str) -> Optional[float]:
        """返回币种真实 24h 振幅（(high24-low24)/last），不可靠时返回 None 放行。

        MarketRegimeEngine.get_symbol_regime 的 volatility 字段在 orderbook 数据可用时
        即为真实 24h 振幅（0~0.5 量级）；币种未被监控或缓存缺失时回退为波动率得分
        （-1~1，量纲完全不同）。此处只信任落在真实振幅合理区间的值，避免量纲错配误杀。
        """
        if not self._regime_engine:
            return None
        try:
            data = self._regime_engine.get_symbol_regime(symbol)
            if not data:
                return None
            vol = safe_float(data.get("volatility"), 0.0)
            # 真实 24h 振幅合理上界 0.5（50%），超出或非正视为不可靠数据
            if 0.0 < vol <= 0.5:
                return vol
            return None
        except Exception as e:
            logger.debug(f"RegimeGate get volatility error for {symbol}: {e}")
            return None

    def evaluate(
        self,
        symbol: str,
        strategy_name: str,
        signal_type: str = "",
        direction: str = "",
        confidence: float = 0.0,
    ) -> GateResult:
        """评估单个开仓信号是否放行。

        修正思路：
        - 统一 normalize regime，兼容旧状态和新状态；
        - 明确处理 breakout / breakdown / reversal；
        - 让 mean-reversion / trend-following 在具体状态下各自使用最对的门控；
        - 避免因为状态口径不统一导致误杀或误放。
        """
        if _is_close_signal(signal_type):
            return GateResult(True, "close signal always allowed", regime=self._get_regime())

        regime_info = self._resolve_regime_info(symbol)
        if isinstance(regime_info, dict) and regime_info.get("data_stale") is True:
            return GateResult(False, "market regime data stale; reject open signal", regime="unknown", action="reject")

        regime = self._normalize_regime((regime_info or {}).get("regime", "unknown"))
        if regime in ("unknown", ""):
            return GateResult(True, "regime unknown, allow by default", regime=regime)

        strategy_key = (strategy_name or "").lower()
        direction_key = (direction or "").lower()

        if regime in TREND_REGIMES:
            if strategy_key in TREND_FOLLOWING_STRATEGIES:
                if strategy_key == "trend" and direction_key in ("long", "short"):
                    sym_regime, sym_strength = self._get_symbol_regime_info(symbol)
                    if sym_strength >= self._trend_counter_direction_strength:
                        if sym_regime == "trend_bearish" and direction_key == "long":
                            return GateResult(
                                False,
                                f"trend_bearish (strength={sym_strength:.2f}) blocks trend long open ({symbol})",
                                regime=sym_regime,
                                action="reject",
                            )
                        if sym_regime == "trend_bullish" and direction_key == "short":
                            return GateResult(
                                False,
                                f"trend_bullish (strength={sym_strength:.2f}) blocks trend short open ({symbol})",
                                regime=sym_regime,
                                action="reject",
                            )
                return GateResult(True, f"trend regime allows {strategy_name}", regime=regime)

            if strategy_key in MEAN_REVERSION_STRATEGIES and self._trend_mr_allow_with_trend:
                if regime == "trend_bullish" and direction_key == "long":
                    return GateResult(True, f"trend_bullish allows mean-reversion {strategy_name} long", regime=regime)
                if regime == "trend_bearish" and direction_key == "short":
                    return GateResult(True, f"trend_bearish allows mean-reversion {strategy_name} short", regime=regime)
                return GateResult(
                    False,
                    f"trend regime blocks counter-trend mean-reversion {strategy_name} {direction}",
                    regime=regime,
                    action="reject",
                )

            return GateResult(False, f"trend regime blocks mean-reversion {strategy_name}", regime=regime, action="reject")

        if regime in BREAKOUT_REGIMES:
            if regime == "reversal":
                probabilities = regime_info.get("probabilities", {})
                if not isinstance(probabilities, dict):
                    probabilities = {}
                reversal_probability = safe_float(
                    regime_info.get(
                        "detector_reversal_prob",
                        probabilities.get("reversal", 0.0),
                    ),
                    0.0,
                )
                if (
                    reversal_probability < self._reversal_probability_threshold
                    or confidence < self._reversal_signal_confidence_threshold
                ):
                    return GateResult(
                        False,
                        "reversal confirmation insufficient "
                        f"(probability={reversal_probability:.2f}, confidence={confidence:.2f})",
                        regime=regime,
                        action="reject",
                    )

                detector_raw = regime_info.get("detector_raw", {})
                if not isinstance(detector_raw, dict):
                    detector_raw = {}
                reversal_direction = str(
                    regime_info.get("reversal_direction")
                    or detector_raw.get("reversal_direction")
                    or ""
                ).strip().lower()
                if reversal_direction in ("bullish", "buy"):
                    reversal_direction = "long"
                elif reversal_direction in ("bearish", "sell"):
                    reversal_direction = "short"

                if strategy_key not in TREND_FOLLOWING_STRATEGIES or direction_key != reversal_direction:
                    return GateResult(
                        False,
                        "reversal open requires explicit matching direction confirmation",
                        regime=regime,
                        action="reject",
                    )

            if strategy_key in TREND_FOLLOWING_STRATEGIES:
                if regime == "breakout" and direction_key == "long":
                    return GateResult(True, f"breakout regime allows trend {strategy_name} long", regime=regime)
                if regime == "breakdown" and direction_key == "short":
                    return GateResult(True, f"breakdown regime allows trend {strategy_name} short", regime=regime)
            if strategy_key in MEAN_REVERSION_STRATEGIES:
                return GateResult(False, f"breakout/breakdown regime blocks {strategy_name} mean-reversion", regime=regime, action="reject")
            return GateResult(True, f"breakout-like regime {regime} allows {strategy_name}", regime=regime)

        if regime in RANGE_REGIMES:
            if strategy_key in TREND_FOLLOWING_STRATEGIES:
                if self._range_trend_allow_high_confidence and confidence >= self._range_trend_confidence_threshold:
                    return GateResult(
                        True,
                        f"range regime: {strategy_name} high-confidence breakout allowed "
                        f"(confidence={confidence:.2f} >= {self._range_trend_confidence_threshold:.2f})",
                        regime=regime,
                    )
                return GateResult(False, f"range regime blocks {strategy_name}", regime=regime, action="reject")

            if strategy_key in MEAN_REVERSION_STRATEGIES:
                if self._range_bound_vol_check_enabled:
                    vol = self._get_symbol_volatility(symbol)
                    if vol is not None and vol < self._range_bound_min_volatility:
                        return GateResult(
                            False,
                            f"range regime: {symbol} 24h amplitude {vol:.4f} < {self._range_bound_min_volatility:.4f}, block mean-reversion {strategy_name}",
                            regime=regime,
                            action="reject",
                        )
                return GateResult(True, f"range regime: allow mean-reversion {strategy_name} ({symbol})", regime=regime)

            return GateResult(True, f"range regime: unknown strategy {strategy_name} allowed", regime=regime)

        if regime in HIGH_RISK_REGIMES:
            if strategy_key in TREND_FOLLOWING_STRATEGIES:
                return GateResult(True, f"high-risk regime: {strategy_name} allowed (sizing reduced elsewhere)", regime=regime)
            return GateResult(False, f"high-risk regime blocks {strategy_name} opens", regime=regime, action="reject")

        return GateResult(True, f"regime {regime} not gated", regime=regime)
