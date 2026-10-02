"""
统一自适应仓位管理引擎 - Adaptive Position Sizer
==================================================

职责：把分散在各处的「开仓前仓位计算」收敛为单一口径的企业级自适应引擎，
      融合多因子输出统一 PositionSizingResult，并附带可解释 breakdown。

现状（收敛目标）——仓位逻辑原先分散在：
  1. PositionPlanner.calculate_position_size_for_signal  （简单风险预算）
  2. EquityMonitor.get_position_multiplier                （权益模式）
  3. AdaptiveKelly.compute_kelly                          （企业级 Kelly，被各策略直接实例化）
  4. StrategyRisk.calculate_position_size                 （旧式离散 Kelly）
  5. utils.helpers.calculate_adaptive_position_size       （市场状态乘数）

多因子融合（分层，避免重复计算）：
  ── Kelly 层（复用 AdaptiveKelly，内部已融合贝叶斯胜率收缩 + 赔率收缩 +
     市场状态 regime + 回撤惩罚 + 连续盈亏 + 样本量折扣 + 分数 Kelly）
  ── 引擎层（Kelly 未覆盖的因子）：
       · 权益模式乘数   equity_factor   （来自 EquityMonitor.get_position_multiplier）
       · 信号强度因子   signal_factor   （0.5 + signal_strength * 0.5）
       · 波动率因子     volatility_factor（ATR% 阶梯，对齐 PositionPlanner）
       · 策略资金占比   allocation      （trading.{strategy}_allocation）
  ── 资金层：单笔风险金额 / 名义价值
  ── 裁剪层：币种 tier position_limit、杠杆上限、最小名义价值、最大名义价值

计算模式：
  · 风险预算式（提供 stop_distance 或 atr）：
        risk_amount = balance * risk_per_trade * risk_fraction
        quantity    = risk_amount / stop_distance
  · 名义价值式（未提供止损距离，回退）：
        notional = balance * risk_fraction
        quantity = notional / price

企业级特性：
  · 纯函数计算（同步、无网络/IO），不下单，仅作为口径收敛与计算链
  · 可解释 breakdown：输出每个因子贡献，便于审计与回溯
  · 配置裁剪 + 默认值 + 容错（tier 解析失败回退默认档）
  · 短路拒绝：非法输入 / 紧急模式 / 低于最小名义价值
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from loguru import logger

from configs.settings import get_currency_tier
from risk.dynamic_allocator import AdaptiveKelly
from utils.helpers import map_market_state_to_regime, safe_float, safe_int, safe_finite


# ═══════════════════════════════════════════════════════════════
# 结果模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class PositionSizingResult:
    """统一仓位计算结果（可解释 + 可序列化）"""
    symbol: str
    strategy_name: str
    quantity: float = 0.0            # 最终数量
    notional: float = 0.0            # 名义价值（USDT）= quantity * price
    margin: float = 0.0              # 保证金 = notional / leverage
    leverage: float = 1.0            # 实际杠杆
    risk_amount: float = 0.0         # 单笔风险金额（USDT）
    risk_fraction: float = 0.0       # 综合仓位分数
    kelly_fraction: float = 0.0      # 实际采用的 Kelly 分量
    position_multiplier: float = 1.0  # 引擎层乘数（equity*signal*vol，不含 kelly/allocation）
    allowed: bool = False
    reject_reason: str = ""
    mode: str = "notional"           # "risk_budget" | "notional"
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "strategy_name": self.strategy_name,
            "quantity": safe_finite(self.quantity),
            "notional": round(safe_finite(self.notional), 4),
            "margin": round(safe_finite(self.margin), 4),
            "leverage": safe_finite(self.leverage),
            "risk_amount": round(safe_finite(self.risk_amount), 6),
            "risk_fraction": round(safe_finite(self.risk_fraction), 6),
            "kelly_fraction": round(safe_finite(self.kelly_fraction), 6),
            "position_multiplier": round(safe_finite(self.position_multiplier), 6),
            "allowed": self.allowed,
            "reject_reason": self.reject_reason,
            "mode": self.mode,
            "breakdown": self.breakdown,
        }


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _f(v, default: float = 0.0) -> float:
    """安全转 float，None/空串/NaN/Inf/非法值返回默认值。"""
    return safe_float(v, default)


def _i(v, default: int = 0) -> int:
    """安全转 int。"""
    return safe_int(v, default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _get(d: Dict[str, Any], key: str, default=None):
    """安全取值，兼容 dict 与 pydantic 对象。"""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


# Kelly 合法 regime 值（AdaptiveKelly._regime_multipliers 的 key）
_KELLY_REGIMES = {
    "trending_up", "trending_down", "ranging",
    "high_volatility", "low_volatility", "unknown",
}


def _normalize_regime(regime) -> str:
    """把市场状态/regime 归一化为 AdaptiveKelly 的 regime 值。

    既接受 detect_market_state 的 state（uptrend/downtrend/range/...），
    也接受已经是 Kelly regime 的字符串。
    """
    if regime is None:
        return "unknown"
    value = getattr(regime, "value", regime)
    s = str(value) if value else "unknown"
    if s in _KELLY_REGIMES:
        return s
    return map_market_state_to_regime(s)


# ═══════════════════════════════════════════════════════════════
# 统一引擎
# ═══════════════════════════════════════════════════════════════

class AdaptivePositionSizer:
    """统一自适应仓位管理引擎（纯计算，不下单）。

    融合：AdaptiveKelly（离散+连续） + 权益模式 + 信号强度 + 波动率 +
          策略资金占比 + 币种 tier + 杠杆上限，输出统一 PositionSizingResult。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = self.config.get("adaptive_position_sizing", {})
        trading = self.config.get("trading", {})

        # ── 引擎专属配置 ──
        self._default_risk_fraction = _clamp(_f(cfg.get("default_risk_fraction"), 0.02), 0.0, 1.0)
        self._min_risk_fraction = _clamp(_f(cfg.get("min_risk_fraction"), 0.005), 0.0, 1.0)
        self._max_risk_fraction = _clamp(_f(cfg.get("max_risk_fraction"), 0.30), 0.0, 1.0)
        self._min_notional_usd = _f(cfg.get("min_notional_usd"), 5.0)
        self._signal_neutral = _clamp(_f(cfg.get("signal_strength_neutral"), 0.5), 0.0, 1.0)
        self._volatility_max = _f(cfg.get("volatility_max"), 0.05)
        self._atr_sl_multiplier = _f(cfg.get("atr_sl_multiplier"), 2.0)
        self._round_precision = _i(cfg.get("round_precision"), 8)
        self._use_equity = bool(cfg.get("use_equity_multiplier", True))
        self._use_signal = bool(cfg.get("use_signal_factor", True))
        self._use_volatility = bool(cfg.get("use_volatility_factor", True))

        # ── 交易层默认值（tier 解析失败时回退）──
        self._risk_per_trade = _f(trading.get("risk_per_trade"), 0.02)
        self._max_position_ratio = _f(trading.get("max_position_ratio"), 0.8)
        self._default_leverage = _i(trading.get("default_leverage"), 5)
        self._max_leverage = _i(trading.get("max_leverage"), 20)

        # 复用的企业级 Kelly（内部已融合 regime/回撤/连续盈亏/样本量/分数 Kelly）
        self._kelly = AdaptiveKelly(self.config.get("adaptive_kelly", {}))

        logger.info(
            f"AdaptivePositionSizer initialized: default_risk={self._default_risk_fraction:.1%}, "
            f"max_risk={self._max_risk_fraction:.1%}, min_notional={self._min_notional_usd:.2f}"
        )

    # ─────────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────────

    def compute(
        self,
        symbol: str,
        price: float,
        account_balance: float,
        strategy_name: str = "grid",
        win_rate: float = 0.5,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        trade_count: int = 0,
        consecutive_wins: int = 0,
        consecutive_losses: int = 0,
        regime: str = "unknown",
        drawdown: float = 0.0,
        signal_strength: float = 0.5,
        volatility: float = 0.0,
        equity_multiplier: Optional[float] = None,
        atr: float = 0.0,
        stop_distance: float = 0.0,
        leverage: Optional[int] = None,
        allocation: Optional[float] = None,
        direction: str = "long",
        position_boost: Optional[float] = None,
    ) -> PositionSizingResult:
        """计算统一仓位。

        Args:
            symbol: 交易对（如 BTC-USDT-SWAP）
            price: 当前价格
            account_balance: 账户权益（USDT）
            strategy_name: 策略名（grid/trend/scalping/arbitrage/spot_grid/spot_martingale）
            win_rate/avg_win/avg_loss/trade_count/consecutive_wins/consecutive_losses:
                策略历史表现（用于 Kelly）
            regime: 市场状态（state 或 Kelly regime 均可）
            drawdown: 当前回撤比例（0~1）
            signal_strength: 信号强度（0~1）
            volatility: 波动率（ATR%）
            equity_multiplier: 权益模式乘数（来自 EquityMonitor，None=1.0）
            atr: 平均真实波幅（用于推导止损距离）
            stop_distance: 止损距离（价格单位，优先于 atr）
            leverage: 杠杆（None=按 tier 默认）
            allocation: 动态策略资金占比（None=回退 config.trading.{strategy}_allocation）
            direction: 方向（long/short，用于记录）
        """
        breakdown: Dict[str, Any] = {}

        price = _f(price)
        account_balance = _f(account_balance)

        # ── 短路：非法输入 ──
        if account_balance <= 0:
            return self._reject(symbol, strategy_name, "invalid_balance", breakdown)
        if price <= 0:
            return self._reject(symbol, strategy_name, "invalid_price", breakdown)

        # ── 币种 tier / 杠杆 / 资金占比 ──
        tier = self._resolve_tier(symbol)
        tier_settings = self._resolve_tier_settings(tier)
        if allocation is not None:
            allocation = _clamp(_f(allocation), 0.0, 1.0)
        else:
            allocation = self._resolve_allocation(strategy_name)
        position_limit = _f(_get(tier_settings, "position_limit"), 0.5)
        tier_leverage_default = _i(_get(tier_settings, "leverage_default"), self._default_leverage)
        tier_leverage_max = _i(_get(tier_settings, "leverage_max"), self._max_leverage)

        leverage = _i(leverage, 0) or tier_leverage_default
        leverage = max(1, min(int(leverage), tier_leverage_max))

        # 资金利用率引擎仓位放大（idle-cash position boost），默认 1.0 不放大
        position_boost = _clamp(_f(position_boost, 1.0), 0.0, 6.0)

        breakdown.update({
            "tier": tier,
            "allocation": round(allocation, 4),
            "position_limit": round(position_limit, 4),
            "leverage": leverage,
            "leverage_max": tier_leverage_max,
            "risk_per_trade": round(self._risk_per_trade, 4),
            "position_boost": round(position_boost, 4),
        })

        # ── 权益模式乘数 ──
        equity_factor = 1.0
        if self._use_equity and equity_multiplier is not None:
            equity_factor = max(0.0, _f(equity_multiplier, 1.0))
        # 紧急模式（乘数 <= 0）：禁止开新仓
        if equity_factor <= 0:
            breakdown["equity_factor"] = 0.0
            return self._reject(symbol, strategy_name, "equity_emergency", breakdown)

        # ── 信号强度因子（signal_strength=0.5 中性 → 1.0）──
        signal_factor = 1.0
        if self._use_signal:
            ss = _clamp(_f(signal_strength, self._signal_neutral), 0.0, 1.0)
            signal_factor = 0.5 + ss

        # ── 波动率因子 ──
        volatility_factor = 1.0
        if self._use_volatility:
            volatility_factor = self._volatility_factor(_f(volatility))

        # ── Kelly 分量（复用 AdaptiveKelly，内部融合 regime/回撤/连续盈亏/样本量）──
        kelly_regime = _normalize_regime(regime)
        kelly_result = self._kelly.compute_kelly(
            win_rate=_f(win_rate, 0.5),
            avg_win=_f(avg_win),
            avg_loss=_f(avg_loss),
            regime=kelly_regime,
            drawdown=_f(drawdown),
            consecutive_wins=_i(consecutive_wins),
            consecutive_losses=_i(consecutive_losses),
            trade_count=_i(trade_count),
        )
        kelly_fraction = _f(kelly_result.get("fractional_kelly"), 0.0)

        # 无边际优势 / 数据不足 → 回退保守默认（并标注）
        kelly_fallback = False
        if kelly_fraction <= 0:
            kelly_fallback = True
            kelly_fraction = self._default_risk_fraction

        # ── 引擎层乘数 & 综合仓位分数 ──
        position_multiplier = equity_factor * signal_factor * volatility_factor
        risk_fraction = kelly_fraction * position_multiplier * allocation
        # 仅在有实际资金分配时应用最小风险地板，避免 allocation=0 时被强制开仓
        if allocation > 0 and risk_fraction > 0:
            risk_fraction = max(risk_fraction, self._min_risk_fraction)
        risk_fraction = _clamp(risk_fraction, 0.0, self._max_risk_fraction)

        breakdown.update({
            "kelly": kelly_result,
            "kelly_fraction": round(kelly_fraction, 6),
            "kelly_fallback_default": kelly_fallback,
            "equity_factor": round(equity_factor, 4),
            "signal_factor": round(signal_factor, 4),
            "volatility_factor": round(volatility_factor, 4),
            "position_multiplier": round(position_multiplier, 6),
            "risk_fraction": round(risk_fraction, 6),
            "direction": direction,
        })

        # ── 止损距离解析（优先显式 stop_distance，其次 atr）──
        resolved_stop_distance = _f(stop_distance)
        if resolved_stop_distance <= 0 and _f(atr) > 0:
            resolved_stop_distance = _f(atr) * self._atr_sl_multiplier

        # ── 计算原始名义价值 ──
        if resolved_stop_distance > 0:
            mode = "risk_budget"
            risk_amount = account_balance * self._risk_per_trade * risk_fraction
            quantity_raw = risk_amount / resolved_stop_distance
            notional_raw = quantity_raw * price
        else:
            mode = "notional"
            risk_amount = account_balance * risk_fraction
            notional_raw = account_balance * risk_fraction
            quantity_raw = notional_raw / price

        # ── 资金利用率引擎仓位放大（idle-cash position boost）──
        # boost_aggressive 时放大单笔名义价值，解决小账户占用不足；下方仍受 max_notional 裁剪。
        if position_boost != 1.0:
            notional_raw *= position_boost
            quantity_raw *= position_boost
            risk_amount *= position_boost

        # ── 裁剪层：最大名义价值（position_limit 与杠杆双约束）──
        max_notional_by_ratio = account_balance * self._max_position_ratio
        max_notional_by_leverage = account_balance * leverage
        max_notional = min(max_notional_by_ratio, max_notional_by_leverage)

        notional = min(notional_raw, max_notional)
        quantity = notional / price
        quantity = round(quantity, self._round_precision)
        margin = notional / leverage if leverage > 0 else notional

        breakdown.update({
            "stop_distance": round(resolved_stop_distance, 8),
            "risk_amount": round(risk_amount, 6),
            "notional_raw": round(notional_raw, 4),
            "max_notional": round(max_notional, 4),
            "notional": round(notional, 4),
            "margin": round(margin, 4),
        })

        # ── 最小名义价值短路 ──
        if self._min_notional_usd > 0 and notional < self._min_notional_usd:
            return self._reject(
                symbol, strategy_name,
                f"below_min_notional: {notional:.4f} < {self._min_notional_usd:.2f}",
                breakdown,
            )

        if risk_fraction <= 0:
            return self._reject(symbol, strategy_name, "zero_risk_fraction", breakdown)

        return PositionSizingResult(
            symbol=symbol,
            strategy_name=strategy_name,
            quantity=quantity,
            notional=notional,
            margin=margin,
            leverage=float(leverage),
            risk_amount=risk_amount,
            risk_fraction=risk_fraction,
            kelly_fraction=kelly_fraction,
            position_multiplier=position_multiplier,
            allowed=True,
            mode=mode,
            breakdown=breakdown,
        )

    # ─────────────────────────────────────────────────────────────
    # 相对调整（回退兼容现有策略）
    # ─────────────────────────────────────────────────────────────

    def compute_multiplier(
        self,
        win_rate: float = 0.5,
        avg_win: float = 0.0,
        avg_loss: float = 0.0,
        trade_count: int = 0,
        consecutive_wins: int = 0,
        consecutive_losses: int = 0,
        regime: str = "unknown",
        drawdown: float = 0.0,
        signal_strength: float = 0.5,
        volatility: float = 0.0,
        equity_multiplier: Optional[float] = None,
    ) -> Dict[str, Any]:
        """计算统一的仓位乘数（供现有策略做相对 base_position 调整）。

        返回：
            {
                "multiplier": float,       # 总乘数（>=0）
                "kelly_fraction": float,
                "kelly_fallback_default": bool,
                "position_multiplier": float,
                "kelly": {...},
            }
        """
        kelly_regime = _normalize_regime(regime)
        kelly_result = self._kelly.compute_kelly(
            win_rate=_f(win_rate, 0.5),
            avg_win=_f(avg_win),
            avg_loss=_f(avg_loss),
            regime=kelly_regime,
            drawdown=_f(drawdown),
            consecutive_wins=_i(consecutive_wins),
            consecutive_losses=_i(consecutive_losses),
            trade_count=_i(trade_count),
        )
        kelly_fraction = _f(kelly_result.get("fractional_kelly"), 0.0)
        kelly_fallback = kelly_fraction <= 0

        equity_factor = 1.0
        if self._use_equity and equity_multiplier is not None:
            equity_factor = max(0.0, _f(equity_multiplier, 1.0))

        signal_factor = 1.0
        if self._use_signal:
            ss = _clamp(_f(signal_strength, self._signal_neutral), 0.0, 1.0)
            signal_factor = 0.5 + ss

        volatility_factor = 1.0
        if self._use_volatility:
            volatility_factor = self._volatility_factor(_f(volatility))

        position_multiplier = equity_factor * signal_factor * volatility_factor
        if equity_factor <= 0:
            multiplier = 0.0
        else:
            kelly_base = kelly_fraction if kelly_fraction > 0 else self._default_risk_fraction
            # 以 0.05 为基准做相对映射（与现有策略 _calculate_kelly_position 对齐）
            multiplier = 1.0 + (kelly_base - 0.05) * 4.0
            multiplier *= position_multiplier
            multiplier = max(0.0, multiplier)

        return {
            "multiplier": round(multiplier, 6),
            "kelly_fraction": round(kelly_fraction, 6),
            "kelly_fallback_default": kelly_fallback,
            "position_multiplier": round(position_multiplier, 6),
            "kelly": kelly_result,
        }

    # ─────────────────────────────────────────────────────────────
    # 内部辅助
    # ─────────────────────────────────────────────────────────────

    def _resolve_tier(self, symbol: str) -> str:
        try:
            return get_currency_tier(symbol, self.config)
        except Exception:
            return "tier3"

    def _resolve_tier_settings(self, tier: str) -> Dict[str, Any]:
        try:
            return self.config.get("currencies", {}).get(f"{tier}_settings", {}) or {}
        except Exception:
            return {}

    def _resolve_allocation(self, strategy_name: str) -> float:
        trading = self.config.get("trading", {})
        alloc = trading.get(f"{strategy_name}_allocation", 0.20)
        return _clamp(_f(alloc, 0.20), 0.0, 1.0)

    def _volatility_factor(self, volatility: float) -> float:
        """波动率阶梯因子（对齐 PositionPlanner.adjust_for_volatility）。"""
        if volatility <= 0:
            return 1.0
        if volatility > self._volatility_max:
            return 0.5
        if volatility > self._volatility_max * 0.7:
            return 0.7
        if volatility > self._volatility_max * 0.5:
            return 0.85
        if volatility < self._volatility_max * 0.3:
            return 1.2
        return 1.0

    @staticmethod
    def _reject(symbol: str, strategy_name: str, reason: str,
                breakdown: Dict[str, Any]) -> PositionSizingResult:
        return PositionSizingResult(
            symbol=symbol,
            strategy_name=strategy_name,
            allowed=False,
            reject_reason=reason,
            breakdown=breakdown,
        )

    # ─────────────────────────────────────────────────────────────
    # 摘要
    # ─────────────────────────────────────────────────────────────

    def get_config_summary(self) -> Dict[str, Any]:
        return {
            "default_risk_fraction": self._default_risk_fraction,
            "min_risk_fraction": self._min_risk_fraction,
            "max_risk_fraction": self._max_risk_fraction,
            "min_notional_usd": self._min_notional_usd,
            "risk_per_trade": self._risk_per_trade,
            "max_position_ratio": self._max_position_ratio,
            "volatility_max": self._volatility_max,
            "atr_sl_multiplier": self._atr_sl_multiplier,
            "kelly": self._kelly.get_kelly_summary(),
        }
