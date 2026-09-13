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

# regime 字符串值（与 MarketRegime.value 一致）
TREND_REGIMES = {"trend_bullish", "trend_bearish"}
RANGE_REGIMES = {"range_bound"}
HIGH_RISK_REGIMES = {"extreme_volatility", "funding_crush", "liquidity_crisis"}

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
    ):
        self._regime_engine = regime_engine
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

    def _get_whitelist(self) -> Set[str]:
        if self._whitelist_provider is None:
            return set()
        try:
            wl = self._whitelist_provider()
            return set(wl) if wl else set()
        except Exception as e:
            logger.debug(f"RegimeGate whitelist provider error: {e}")
            return set()

    def _get_regime(self) -> str:
        if not self._regime_engine:
            return "unknown"
        try:
            info = self._regime_engine.get_regime()
            return (info or {}).get("regime", "unknown")
        except Exception as e:
            logger.debug(f"RegimeGate get_regime error: {e}")
            return "unknown"

    def _get_symbol_regime_info(self, symbol: str):
        """返回 (regime, strength)。优先 symbol 级独立 regime，回退全局 regime。"""
        if not self._regime_engine:
            return "unknown", 0.0
        try:
            data = self._regime_engine.get_symbol_regime(symbol)
            if data:
                return (data.get("regime", "unknown"), float(data.get("strength", 0.0)))
        except Exception as e:
            logger.debug(f"RegimeGate get_symbol_regime error: {e}")
        try:
            info = self._regime_engine.get_regime() or {}
            return (info.get("regime", "unknown"), float(info.get("strength", 0.0)))
        except Exception:
            return "unknown", 0.0

    def evaluate(
        self,
        symbol: str,
        strategy_name: str,
        signal_type: str = "",
        direction: str = "",
        confidence: float = 0.0,
    ) -> GateResult:
        """评估单个开仓信号是否放行。

        Returns:
            GateResult: allowed=False 表示应直接丢弃该信号（不进入审计）。
        """
        # 平仓类信号永远放行（平仓风险由 RiskGate 管理）
        if _is_close_signal(signal_type):
            return GateResult(True, "close signal always allowed", regime=self._get_regime())

        regime = self._get_regime()

        # 引擎未就绪：放行，避免启动阶段误杀全部信号
        if regime in ("unknown", ""):
            return GateResult(True, "regime unknown, allow by default", regime=regime)

        if regime in TREND_REGIMES:
            if strategy_name in TREND_FOLLOWING_STRATEGIES:
                # 趋势方向一致性检查：trend 策略开仓方向须与（币种级）趋势方向一致，
                # 逆势开仓（bearish 中做多 / bullish 中做空）在强趋势下直接拒绝，避免磨损型亏损。
                if strategy_name == "trend" and direction in ("long", "short"):
                    sym_regime, sym_strength = self._get_symbol_regime_info(symbol)
                    if sym_strength >= 0.6:
                        if sym_regime == "trend_bearish" and direction == "long":
                            return GateResult(
                                False,
                                f"trend_bearish (strength={sym_strength:.2f}) blocks trend long open ({symbol})",
                                regime=sym_regime,
                                action="reject",
                            )
                        if sym_regime == "trend_bullish" and direction == "short":
                            return GateResult(
                                False,
                                f"trend_bullish (strength={sym_strength:.2f}) blocks trend short open ({symbol})",
                                regime=sym_regime,
                                action="reject",
                            )
                return GateResult(True, f"trend regime allows {strategy_name}", regime=regime)
            if strategy_name in MEAN_REVERSION_STRATEGIES and self._trend_mr_allow_with_trend:
                # 趋势行情下均值回归策略仅放行顺趋势方向，逆势方向拒绝；
                # 仓位下调由 SignalProcessor._apply_dynamic_sizing 既有 regime 调整兜底。
                if regime == "trend_bullish" and direction == "long":
                    return GateResult(
                        True,
                        f"trend_bullish allows mean-reversion {strategy_name} long",
                        regime=regime,
                    )
                if regime == "trend_bearish" and direction == "short":
                    return GateResult(
                        True,
                        f"trend_bearish allows mean-reversion {strategy_name} short",
                        regime=regime,
                    )
                return GateResult(
                    False,
                    f"trend regime blocks counter-trend mean-reversion "
                    f"{strategy_name} {direction}",
                    regime=regime,
                    action="reject",
                )
            return GateResult(
                False,
                f"trend regime blocks mean-reversion {strategy_name}",
                regime=regime,
                action="reject",
            )

        if regime in RANGE_REGIMES:
            if strategy_name in TREND_FOLLOWING_STRATEGIES:
                # 放开 L0：高置信度突破信号（趋势可能正在脱离震荡）放行，低置信度仍拒绝
                if (self._range_trend_allow_high_confidence
                        and confidence >= self._range_trend_confidence_threshold):
                    return GateResult(
                        True,
                        f"range regime: {strategy_name} high-confidence breakout allowed "
                        f"(confidence={confidence:.2f} >= {self._range_trend_confidence_threshold:.2f})",
                        regime=regime,
                    )
                return GateResult(
                    False,
                    f"range regime blocks {strategy_name}",
                    regime=regime,
                    action="reject",
                )
            if strategy_name in MEAN_REVERSION_STRATEGIES:
                # 震荡市是网格/短线的本命行情：直接放行，不再依赖正期望白名单。
                # （白名单门控曾与策略实际交易标的完全错位，系统性锁死小账户交易。）
                return GateResult(
                    True,
                    f"range regime: allow mean-reversion {strategy_name} ({symbol})",
                    regime=regime,
                )
            # 未知策略在震荡中放行（保守不拦截）
            return GateResult(
                True,
                f"range regime: unknown strategy {strategy_name} allowed",
                regime=regime,
            )

        if regime in HIGH_RISK_REGIMES:
            if strategy_name in TREND_FOLLOWING_STRATEGIES:
                return GateResult(
                    True,
                    f"high-risk regime: {strategy_name} allowed (sizing reduced elsewhere)",
                    regime=regime,
                )
            return GateResult(
                False,
                f"high-risk regime blocks {strategy_name} opens",
                regime=regime,
                action="reject",
            )

        # 未识别 regime：放行
        return GateResult(True, f"regime {regime} not gated", regime=regime)
