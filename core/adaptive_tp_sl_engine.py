"""
统一自适应止损止盈引擎 - Adaptive TP/SL Engine

职责：把分散在各处的止损/止盈逻辑收敛为「单一口径」的企业级自适应计算，
      融合多因子输出 SL / TP / 保本 / 追踪止损 / 分段止盈，并附带可解释 breakdown。

多因子融合：
1. 波动率（ATR / 波动率得分）—— 决定止损/止盈的基础距离
2. 市场状态（regime）—— 趋势放大止盈、震荡收紧、极端波动放宽止损
3. 策略表现（胜率 / 盈亏比）—— 低胜率收紧止损、低盈亏比放大止盈
4. 持仓盈亏/回撤 —— 盈利上移保本→追踪，亏损收紧止损保护资本

企业级特性：
- EMA 平滑：sl/tp 距离跨更新周期平滑，抑制瞬时噪声抖动
- 迟滞（hysteresis）+ 单向棘轮：保本/追踪保护态只升不降，避免频繁翻转
- 可解释 breakdown：输出每个因子对最终 SL/TP 的乘数贡献，便于审计与回溯

纯函数计算（compute 为同步、无网络/无 IO），仅作为口径收敛与计算链，不直接下单；
与 StopLossManager / ConditionalOrderManager 复用同一底层价格函数（utils.helpers）。
"""

from typing import Dict, Any, Optional, List, Tuple

from loguru import logger

from utils.helpers import (
    calculate_stop_loss,
    calculate_take_profit,
    normalize_direction,
    safe_float,
)

# 保护态单向棘轮：none → breakeven → trailing，只升不降（新仓位重置）
_PROTECTION_ORDER = {"none": 0, "breakeven": 1, "trailing": 2}


def _f(v, default: float = 0.0) -> float:
    """安全转 float，None/空串/NaN/Inf/非法值返回默认值。"""
    return safe_float(v, default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _regime_name(regime) -> Optional[str]:
    """把 regime 归一化为字符串（兼容枚举与字符串/None）。"""
    if regime is None:
        return None
    value = getattr(regime, "value", regime)
    return str(value) if value else None


class AdaptiveTpSlEngine:
    """统一自适应止损止盈引擎（纯计算 + 平滑/迟滞状态，不下单）"""

    # 默认策略表现中性值（无历史数据时采用）
    NEUTRAL_WIN_RATE = 0.5
    NEUTRAL_PROFIT_FACTOR = 1.0

    def __init__(self, config: Optional[Dict[str, Any]] = None, regime_engine=None):
        self.config = config or {}
        self._regime_engine = regime_engine
        cfg = self.config.get("adaptive_tp_sl", {})

        # ── 基础距离参数 ──
        self._base_sl_pct = _f(cfg.get("base_sl_pct"), 0.02)
        self._base_tp_pct = _f(cfg.get("base_tp_pct"), 0.06)
        self._atr_sl_multiplier = _f(cfg.get("atr_sl_multiplier"), 1.5)
        self._atr_tp_ratio = _f(cfg.get("atr_tp_ratio"), 3.0)
        self._min_sl_pct = _f(cfg.get("min_sl_pct"), 0.005)
        self._max_sl_pct = _f(cfg.get("max_sl_pct"), 0.08)
        self._min_tp_pct = _f(cfg.get("min_tp_pct"), 0.01)
        self._max_tp_pct = _f(cfg.get("max_tp_pct"), 0.30)

        # ── 保本 / 追踪止损 ──
        self._breakeven_safety_mult = _f(cfg.get("breakeven_safety_mult"), 1.5)
        self._breakeven_min_buffer = _f(cfg.get("breakeven_min_buffer"), 0.001)
        self._trailing_activation_pct = _f(cfg.get("trailing_activation_pct"), 0.015)
        self._trailing_distance_pct = _f(cfg.get("trailing_distance_pct"), 0.005)

        # ── 平滑 / 迟滞 ──
        self._smoothing_alpha = _clamp(_f(cfg.get("smoothing_alpha"), 0.5), 0.0, 1.0)
        self._protection_hysteresis_margin = _f(cfg.get("protection_hysteresis_margin"), 0.002)

        # ── 价格格式化 ──
        self._slippage_pct = _f(cfg.get("slippage_pct"), 0.001)
        self._precision = int(cfg.get("precision", 4))

        # ── 分段止盈（默认 TP1=60%/40%仓、TP2=100%/50%仓、TP3=150%/10%仓）──
        staged = cfg.get("staged_tp", {}).get("levels", None)
        if staged:
            self._staged_levels = [
                {"ratio": _f(lv.get("ratio"), 1.0), "close_ratio": _f(lv.get("close_ratio"), 0.0)}
                for lv in staged
            ]
        else:
            self._staged_levels = [
                {"ratio": 0.6, "close_ratio": 0.4},
                {"ratio": 1.0, "close_ratio": 0.5},
                {"ratio": 1.5, "close_ratio": 0.1},
            ]

        # 单向棘轮：保护态只升不降（key: f"{symbol}:{direction}"）
        self._protection_state: Dict[str, str] = {}
        # 平滑状态（key: f"{symbol}:{direction}" → {"sl_pct", "tp_pct"}）
        self._smoothed: Dict[str, Dict[str, float]] = {}

    # ─────────────────────────────────────────────────────────────
    # 公开 API
    # ─────────────────────────────────────────────────────────────

    def compute(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        *,
        atr: float = 0.0,
        leverage: float = 10.0,
        regime: Optional[Any] = None,
        regime_strength: float = 0.0,
        vol_score: float = 0.0,
        win_rate: Optional[float] = None,
        profit_factor: Optional[float] = None,
        unrealized_pnl: float = 0.0,
        current_price: Optional[float] = None,
        apply_smoothing: bool = True,
    ) -> Dict[str, Any]:
        """计算自适应止损止盈（纯函数，同步）。

        参数：
            symbol          合约标识（如 "BTC-USDT-SWAP"）
            entry_price     开仓均价
            direction       long/short/buy/sell
            atr             绝对 ATR 值（0 表示无数据，回退基础距离）
            leverage        杠杆倍数（用于强平安全距离参考）
            regime          市场状态（枚举或字符串），None 表示未知
            regime_strength 市场状态强度 [0,1]
            vol_score       波动率得分 [-1,1]
            win_rate        策略胜率 [0,1]，None 使用中性值
            profit_factor   策略盈亏比，None 使用中性值
            unrealized_pnl  未实现盈亏（USDT，符号无关，正=盈利）
            current_price   当前标记价（None 则用 entry_price 近似）
            apply_smoothing 是否启用跨周期 EMA 平滑

        返回统一结果字典（含 prices / distances / protection / staged / breakdown）。
        """
        direction = normalize_direction(direction)
        entry_price = _f(entry_price)
        if entry_price <= 0:
            raise ValueError(f"entry_price must be > 0, got {entry_price}")

        regime_name = _regime_name(regime)
        win_rate = self.NEUTRAL_WIN_RATE if win_rate is None else _clamp(_f(win_rate), 0.0, 1.0)
        profit_factor = self.NEUTRAL_PROFIT_FACTOR if profit_factor is None else max(_f(profit_factor), 0.0)

        # 1. 基础距离（ATR 优先，回退基础百分比）
        atr = _f(atr)
        atr_pct = (atr / entry_price) if (atr > 0 and entry_price > 0) else 0.0
        base_sl_pct, base_tp_pct = self._base_distances(atr_pct)

        # 2. 各因子乘数
        vol_sl, vol_tp, vol_meta = self._volatility_factor(vol_score)
        reg_sl, reg_tp, reg_meta = self._regime_factor(regime_name, regime_strength)
        perf_sl, perf_tp, perf_meta = self._performance_factor(win_rate, profit_factor)
        pnl_sl, protection_mode, pnl_meta = self._pnl_protection_factor(
            symbol, direction, entry_price, unrealized_pnl, current_price
        )

        # 3. 合成距离（连乘 + 裁剪）
        raw_sl_pct = base_sl_pct * vol_sl * reg_sl * perf_sl * pnl_sl
        raw_tp_pct = base_tp_pct * vol_tp * reg_tp * perf_tp

        sl_pct = _clamp(raw_sl_pct, self._min_sl_pct, self._max_sl_pct)
        tp_pct = _clamp(raw_tp_pct, self._min_tp_pct, self._max_tp_pct)
        # 止损距离不能大于止盈距离（避免风险回报倒挂）
        if sl_pct > tp_pct:
            tp_pct = max(tp_pct, sl_pct * 1.1)

        smoothed = False
        if apply_smoothing:
            sl_pct, tp_pct, smoothed = self._apply_smoothing(symbol, direction, sl_pct, tp_pct)

        # 4. 价格
        stop_loss = calculate_stop_loss(
            entry_price, direction, sl_pct, self._slippage_pct, self._precision
        )
        take_profit = calculate_take_profit(
            entry_price, direction, tp_pct, self._slippage_pct, self._precision
        )
        breakeven_price = self._breakeven_price(entry_price, direction)

        # 5. 追踪止损 & 保本上移
        trailing = self._trailing_config(entry_price, direction, protection_mode)

        # 6. 分段止盈
        staged = self._staged_take_profit(entry_price, direction, tp_pct)

        # 7. 风险回报比
        rr = self._risk_reward_ratio(entry_price, take_profit, stop_loss, direction)

        breakdown = {
            "atr": round(atr, 6),
            "atr_pct": round(atr_pct, 6),
            "base_sl_pct": round(base_sl_pct, 6),
            "base_tp_pct": round(base_tp_pct, 6),
            "factors": {
                "volatility": vol_meta,
                "regime": reg_meta,
                "performance": perf_meta,
                "pnl_protection": pnl_meta,
            },
            "raw_sl_pct": round(raw_sl_pct, 6),
            "raw_tp_pct": round(raw_tp_pct, 6),
            "final_sl_pct": round(sl_pct, 6),
            "final_tp_pct": round(tp_pct, 6),
            "smoothed": smoothed,
        }

        action = self._resolve_action(protection_mode, unrealized_pnl, entry_price, current_price, direction)

        return {
            "symbol": symbol,
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "breakeven_price": breakeven_price,
            "sl_distance_pct": round(sl_pct, 6),
            "tp_distance_pct": round(tp_pct, 6),
            "risk_reward_ratio": rr,
            "protection_mode": protection_mode,
            "trailing_stop": trailing,
            "staged_take_profit": staged,
            "action": action,
            "breakdown": breakdown,
        }

    def build_context(self, symbol: str) -> Dict[str, Any]:
        """从 regime_engine 提取市场状态上下文，供 compute(**ctx) 展开。"""
        ctx: Dict[str, Any] = {"symbol": symbol}
        if self._regime_engine is None:
            return ctx
        try:
            regime_data = self._regime_engine.get_symbol_regime(symbol)
            if regime_data:
                ctx["regime"] = regime_data.get("regime")
                ctx["regime_strength"] = _f(regime_data.get("strength"), 0.0)
                factor_scores = regime_data.get("factor_scores", {}) or {}
                ctx["vol_score"] = _f(factor_scores.get("volatility"), 0.0)
        except Exception as e:
            logger.debug(f"AdaptiveTpSlEngine.build_context failed for {symbol}: {e}")
        return ctx

    def reset_position(self, symbol: str, direction: str):
        """清除某持仓的平滑/保护态（开新仓或平仓后调用，棘轮随之重置）。"""
        direction = normalize_direction(direction)
        key = f"{symbol}:{direction}"
        self._protection_state.pop(key, None)
        self._smoothed.pop(key, None)

    # ─────────────────────────────────────────────────────────────
    # 基础距离
    # ─────────────────────────────────────────────────────────────

    def _base_distances(self, atr_pct: float) -> Tuple[float, float]:
        if atr_pct > 0:
            sl_pct = _clamp(atr_pct * self._atr_sl_multiplier, self._min_sl_pct, self._max_sl_pct)
            tp_pct = _clamp(sl_pct * self._atr_tp_ratio, self._min_tp_pct, self._max_tp_pct)
        else:
            sl_pct = _clamp(self._base_sl_pct, self._min_sl_pct, self._max_sl_pct)
            tp_pct = _clamp(self._base_tp_pct, self._min_tp_pct, self._max_tp_pct)
        return sl_pct, tp_pct

    # ─────────────────────────────────────────────────────────────
    # 因子：波动率
    # ─────────────────────────────────────────────────────────────

    def _volatility_factor(self, vol_score: float) -> Tuple[float, float, Dict[str, Any]]:
        """波动率得分映射到 ATR 比率，波动越大 SL/TP 越宽（SL 更敏感）。"""
        vol_ratio = _clamp(1.0 + _clamp(vol_score, -1.0, 1.0), 0.5, 2.0)
        sl_mult = vol_ratio
        tp_mult = 1.0 + (vol_ratio - 1.0) * 0.8
        meta = {"vol_score": round(_clamp(vol_score, -1.0, 1.0), 4), "vol_ratio": round(vol_ratio, 4),
                "sl_mult": round(sl_mult, 4), "tp_mult": round(tp_mult, 4)}
        return sl_mult, tp_mult, meta

    # ─────────────────────────────────────────────────────────────
    # 因子：市场状态
    # ─────────────────────────────────────────────────────────────

    def _regime_factor(self, regime_name: Optional[str], strength: float) -> Tuple[float, float, Dict[str, Any]]:
        """按市场状态调整 SL/TP：趋势放大止盈、震荡收紧、极端波动放宽止损。"""
        strength = _clamp(_f(strength), 0.0, 1.0)
        # 强度分量：强度越高，状态调整越充分（0.5 为中性基准，避免无强度时过度反应）
        s = 0.5 + 0.5 * strength

        if regime_name in ("trend_bullish", "trend_bearish"):
            sl_mult = 1.0
            tp_mult = 1.0 + 0.2 * s          # 趋势：让利润奔跑
            reason = f"trend regime (strength={strength:.2f}) widens TP"
        elif regime_name == "range_bound":
            sl_mult = 1.0 - 0.1 * s
            tp_mult = 1.0 - 0.2 * s          # 震荡：均值回归，收紧止盈
            reason = f"range regime (strength={strength:.2f}) tightens TP"
        elif regime_name == "extreme_volatility":
            sl_mult = 1.0 + 0.3 * s
            tp_mult = 1.0 + 0.1 * s          # 极端波动：放宽止损防洗盘
            reason = f"extreme volatility (strength={strength:.2f}) widens SL"
        elif regime_name == "funding_crush":
            sl_mult = 1.0
            tp_mult = 1.0 - 0.2 * s          # 高费率：缩短持仓，收紧止盈
            reason = f"funding crush (strength={strength:.2f}) tightens TP"
        elif regime_name == "liquidity_crisis":
            sl_mult = 1.0 + 0.2 * s          # 流动性差：放宽止损缓冲滑点
            tp_mult = 1.0
            reason = f"liquidity crisis (strength={strength:.2f}) widens SL"
        else:
            sl_mult = 1.0
            tp_mult = 1.0
            reason = "unknown/neutral regime"

        meta = {"regime": regime_name or "unknown", "strength": round(strength, 4),
                "sl_mult": round(sl_mult, 4), "tp_mult": round(tp_mult, 4), "reason": reason}
        return sl_mult, tp_mult, meta

    # ─────────────────────────────────────────────────────────────
    # 因子：策略表现
    # ─────────────────────────────────────────────────────────────

    def _performance_factor(self, win_rate: float, profit_factor: float) -> Tuple[float, float, Dict[str, Any]]:
        """低胜率收紧止损、低盈亏比放大止盈（需要更大的赢家回补）。"""
        # 胜率：0% → sl_mult 0.8（收紧），100% → sl_mult 1.1（略放宽）
        sl_mult = 0.8 + 0.3 * win_rate
        # 盈亏比：pf=1.0 中性；pf 越低越需要放大止盈
        if profit_factor >= 1.5:
            tp_mult = 1.1
        elif profit_factor >= 1.0:
            tp_mult = 1.0
        elif profit_factor >= 0.7:
            tp_mult = 1.2
        else:
            tp_mult = 1.3

        meta = {"win_rate": round(win_rate, 4), "profit_factor": round(profit_factor, 4),
                "sl_mult": round(sl_mult, 4), "tp_mult": round(tp_mult, 4),
                "reason": f"wr={win_rate:.2f}/pf={profit_factor:.2f}"}
        return sl_mult, tp_mult, meta

    # ─────────────────────────────────────────────────────────────
    # 因子：持仓盈亏 / 回撤（保护态 + 止损收紧）
    # ─────────────────────────────────────────────────────────────

    def _pnl_protection_factor(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        unrealized_pnl: float,
        current_price: Optional[float],
    ) -> Tuple[float, str, Dict[str, Any]]:
        """根据持仓盈亏决定保护态（none→breakeven→trailing）与止损收紧。

        盈利时按棘轮上移保护态；亏损时收紧止损（sl_mult < 1）。
        返回 (sl_mult, protection_mode, meta)。
        """
        price = current_price if (current_price and current_price > 0) else entry_price
        profit_pct = (price - entry_price) / entry_price if direction == "long" else (entry_price - price) / entry_price

        breakeven_buffer = self._breakeven_buffer()
        key = f"{symbol}:{direction}"
        current_mode = self._protection_state.get(key, "none")

        target_mode = "none"
        sl_mult = 1.0

        if profit_pct >= self._trailing_activation_pct + self._protection_hysteresis_margin:
            target_mode = "trailing"
        elif profit_pct >= breakeven_buffer + self._protection_hysteresis_margin:
            target_mode = "breakeven"

        # 棘轮：只升不降；但若已实现亏损（profit_pct < 0）则维持当前态不动（保护已损资本）
        if _PROTECTION_ORDER[target_mode] >= _PROTECTION_ORDER[current_mode]:
            new_mode = target_mode
        else:
            new_mode = current_mode

        # 保本/追踪态下，止损至少回撤到保本价之上（sl_mult 放大以覆盖成本）
        if new_mode == "breakeven":
            sl_mult = 1.0 + breakeven_buffer
        elif new_mode == "trailing":
            sl_mult = 1.0 + self._trailing_distance_pct

        # 亏损保护：未实现亏损超过 -0.5% 时收紧止损
        if profit_pct < -0.005:
            sl_mult = 0.7

        self._protection_state[key] = new_mode

        meta = {
            "profit_pct": round(profit_pct, 6),
            "breakeven_buffer": round(breakeven_buffer, 6),
            "protection_mode": new_mode,
            "sl_mult": round(sl_mult, 4),
            "reason": f"profit={profit_pct:.4%} → {new_mode}",
        }
        return sl_mult, new_mode, meta

    # ─────────────────────────────────────────────────────────────
    # 平滑 / 迟滞
    # ─────────────────────────────────────────────────────────────

    def _apply_smoothing(
        self, symbol: str, direction: str, sl_pct: float, tp_pct: float
    ) -> Tuple[float, float, bool]:
        """对 sl/tp 距离做 EMA 平滑，抑制瞬时噪声。首次无历史时直接采用原始值。"""
        key = f"{symbol}:{direction}"
        prev = self._smoothed.get(key)
        if prev is None:
            self._smoothed[key] = {"sl_pct": sl_pct, "tp_pct": tp_pct}
            return sl_pct, tp_pct, False

        alpha = self._smoothing_alpha
        new_sl = alpha * sl_pct + (1.0 - alpha) * prev["sl_pct"]
        new_tp = alpha * tp_pct + (1.0 - alpha) * prev["tp_pct"]
        self._smoothed[key] = {"sl_pct": new_sl, "tp_pct": new_tp}
        return new_sl, new_tp, True

    # ─────────────────────────────────────────────────────────────
    # 保本 / 追踪 / 分段止盈 / 风险回报
    # ─────────────────────────────────────────────────────────────

    def _breakeven_buffer(self) -> float:
        """统一保本缓冲：往返 taker 手续费 × 安全系数，下限 0.1%。"""
        trading_cfg = self.config.get("trading", {})
        taker_fee = _f(trading_cfg.get("taker_fee_rate"), 0.0005)
        round_trip = taker_fee * 2.0
        return max(round_trip * self._breakeven_safety_mult, self._breakeven_min_buffer)

    def _breakeven_price(self, entry_price: float, direction: str) -> float:
        buffer = self._breakeven_buffer()
        if direction == "long":
            return round(entry_price * (1 + buffer), self._precision)
        return round(entry_price * (1 - buffer), self._precision)

    def _trailing_config(self, entry_price: float, direction: str, protection_mode: str) -> Dict[str, Any]:
        enabled = protection_mode in ("breakeven", "trailing")
        if direction == "long":
            activation = entry_price * (1 + self._trailing_activation_pct)
        else:
            activation = entry_price * (1 - self._trailing_activation_pct)
        return {
            "enabled": enabled,
            "activation_price": round(activation, self._precision),
            "distance_pct": round(self._trailing_distance_pct, 6),
            "callback_ratio": round(self._trailing_distance_pct, 6),
        }

    def _staged_take_profit(self, entry_price: float, direction: str, tp_pct: float) -> List[Dict[str, Any]]:
        """分段止盈：基于最终止盈距离按 ratio 拆出多档。"""
        levels = []
        for lv in self._staged_levels:
            ratio = lv["ratio"]
            level_pct = tp_pct * ratio
            if direction == "long":
                price = entry_price * (1 + level_pct)
            else:
                price = entry_price * (1 - level_pct)
            levels.append({
                "level": len(levels) + 1,
                "price": round(price, self._precision),
                "distance_pct": round(level_pct, 6),
                "close_ratio": lv["close_ratio"],
            })
        return levels

    @staticmethod
    def _risk_reward_ratio(entry_price: float, take_profit: float, stop_loss: float, direction: str) -> Optional[float]:
        if direction == "long":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        if risk <= 0:
            # 零/负风险说明止损无效，盈亏比无数学意义，返回 None 避免 Infinity 污染 JSON
            return None if reward > 0 else 0.0
        return round(reward / risk, 4)

    @staticmethod
    def _resolve_action(protection_mode: str, unrealized_pnl: float, entry_price: float,
                        current_price: Optional[float], direction: str) -> str:
        price = current_price if (current_price and current_price > 0) else entry_price
        profit_pct = (price - entry_price) / entry_price if direction == "long" else (entry_price - price) / entry_price
        if protection_mode == "trailing":
            return "trail"
        if protection_mode == "breakeven":
            return "breakeven"
        if profit_pct < -0.05:
            return "close"
        return "hold"
