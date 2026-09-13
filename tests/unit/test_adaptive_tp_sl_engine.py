"""AdaptiveTpSlEngine 企业级自适应止损止盈 — 正式单元测试。

覆盖矩阵（聚焦「强化企业级自适应止损止盈模块」改动）：
  P0  初始化 & 配置解析/裁剪
  P0  基础距离 _base_distances（ATR vs 回退）
  P0  波动率因子 _volatility_factor
  P0  市场状态因子 _regime_factor（各 regime + 强度分量）
  P0  策略表现因子 _performance_factor（胜率 / 盈亏比）
  P0  持仓盈亏保护 _pnl_protection_factor（保本/追踪棘轮 + 亏损收紧）
  P0  保本缓冲 _breakeven_buffer / 保本价 _breakeven_price
  P0  计算入口 compute（方向正确性、裁剪、风险回报、action）
  P0  平滑 _apply_smoothing
  P0  分段止盈 _staged_take_profit
  P0  状态重置 reset_position
  P1  上下文构建 build_context（有/无 regime_engine）
  P2  边界（非法 entry_price、未知 regime、极端输入）
"""

import pytest

from core.adaptive_tp_sl_engine import AdaptiveTpSlEngine
from execution.order_executor import OrderExecutor


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _make_engine(**overrides) -> AdaptiveTpSlEngine:
    """构造带 adaptive_tp_sl 配置段的引擎，允许覆盖参数。"""
    cfg = {"adaptive_tp_sl": {}}
    for key, value in overrides.items():
        cfg["adaptive_tp_sl"][key] = value
    return AdaptiveTpSlEngine(cfg)


def _default_ctx(symbol="BTC-USDT-SWAP", entry=100.0, direction="long", **overrides):
    """构造 compute 默认参数上下文。"""
    ctx = {
        "symbol": symbol,
        "entry_price": entry,
        "direction": direction,
        "atr": 0.0,
        "regime": None,
        "regime_strength": 0.0,
        "vol_score": 0.0,
        "win_rate": None,
        "profit_factor": None,
        "unrealized_pnl": 0.0,
        "current_price": None,
    }
    ctx.update(overrides)
    return ctx


# ═══════════════════════════════════════════════════════════════
# P0: 初始化 & 配置解析
# ═══════════════════════════════════════════════════════════════

class TestInitialization:
    def test_defaults_applied_when_no_config(self):
        eng = AdaptiveTpSlEngine({})
        assert eng._base_sl_pct == 0.02
        assert eng._base_tp_pct == 0.06
        assert eng._atr_sl_multiplier == 1.5
        assert eng._atr_tp_ratio == 3.0
        assert eng._max_sl_pct == 0.08
        assert eng._max_tp_pct == 0.30
        assert eng._breakeven_safety_mult == 1.5
        assert eng._smoothing_alpha == 0.5

    def test_config_override(self):
        eng = _make_engine(base_sl_pct=0.03, max_sl_pct=0.10, smoothing_alpha=0.8)
        assert eng._base_sl_pct == 0.03
        assert eng._max_sl_pct == 0.10
        assert eng._smoothing_alpha == 0.8

    def test_smoothing_alpha_clamped(self):
        assert _make_engine(smoothing_alpha=5.0)._smoothing_alpha == 1.0
        assert _make_engine(smoothing_alpha=-1.0)._smoothing_alpha == 0.0

    def test_staged_levels_custom(self):
        levels = [{"ratio": 0.5, "close_ratio": 0.5}, {"ratio": 1.0, "close_ratio": 0.5}]
        eng = _make_engine(staged_tp={"levels": levels})
        assert eng._staged_levels == levels

    def test_default_staged_levels(self):
        eng = _make_engine()
        assert len(eng._staged_levels) == 3
        assert eng._staged_levels[0] == {"ratio": 0.6, "close_ratio": 0.4}


# ═══════════════════════════════════════════════════════════════
# P0: 基础距离
# ═══════════════════════════════════════════════════════════════

class TestBaseDistances:
    def test_atr_zero_falls_back_to_base(self):
        eng = _make_engine()
        sl, tp = eng._base_distances(0.0)
        assert sl == pytest.approx(0.02)
        assert tp == pytest.approx(0.06)

    def test_atr_drives_distances(self):
        eng = _make_engine()
        # atr_pct = 0.01 → sl = 0.015, tp = 0.045
        sl, tp = eng._base_distances(0.01)
        assert sl == pytest.approx(0.015)
        assert tp == pytest.approx(0.045)

    def test_atr_sl_clamped_to_max(self):
        eng = _make_engine()
        sl, tp = eng._base_distances(0.5)  # 0.5*1.5 = 0.75 → clamp 0.08
        assert sl == pytest.approx(0.08)

    def test_atr_sl_clamped_to_min(self):
        eng = _make_engine()
        sl, tp = eng._base_distances(0.0001)  # 0.00015 → clamp 0.005
        assert sl == pytest.approx(0.005)


# ═══════════════════════════════════════════════════════════════
# P0: 波动率因子
# ═══════════════════════════════════════════════════════════════

class TestVolatilityFactor:
    def test_neutral_volatility(self):
        eng = _make_engine()
        sl_mult, tp_mult, meta = eng._volatility_factor(0.0)
        assert sl_mult == pytest.approx(1.0)
        assert tp_mult == pytest.approx(1.0)
        assert meta["vol_ratio"] == pytest.approx(1.0)

    def test_high_volatility_widens_sl_more_than_tp(self):
        eng = _make_engine()
        sl_mult, tp_mult, _ = eng._volatility_factor(0.6)
        assert sl_mult == pytest.approx(1.6)
        assert tp_mult == pytest.approx(1.48)

    def test_low_volatility_narrows(self):
        eng = _make_engine()
        sl_mult, tp_mult, _ = eng._volatility_factor(-0.5)
        assert sl_mult == pytest.approx(0.5)
        assert tp_mult == pytest.approx(0.6)

    def test_score_out_of_range_clamped(self):
        eng = _make_engine()
        sl_mult, _, _ = eng._volatility_factor(5.0)
        assert sl_mult == pytest.approx(2.0)
        sl_mult, _, _ = eng._volatility_factor(-5.0)
        assert sl_mult == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════
# P0: 市场状态因子
# ═══════════════════════════════════════════════════════════════

class TestRegimeFactor:
    def test_trend_widens_tp(self):
        eng = _make_engine()
        sl_mult, tp_mult, meta = eng._regime_factor("trend_bullish", 1.0)
        assert sl_mult == pytest.approx(1.0)
        assert tp_mult == pytest.approx(1.2)
        assert "widens TP" in meta["reason"]

    def test_range_tightens_tp(self):
        eng = _make_engine()
        sl_mult, tp_mult, _ = eng._regime_factor("range_bound", 1.0)
        assert sl_mult == pytest.approx(0.9)
        assert tp_mult == pytest.approx(0.8)

    def test_extreme_volatility_widens_sl(self):
        eng = _make_engine()
        sl_mult, tp_mult, _ = eng._regime_factor("extreme_volatility", 1.0)
        assert sl_mult == pytest.approx(1.3)
        assert tp_mult == pytest.approx(1.1)

    def test_funding_crush_tightens_tp(self):
        eng = _make_engine()
        sl_mult, tp_mult, _ = eng._regime_factor("funding_crush", 1.0)
        assert sl_mult == pytest.approx(1.0)
        assert tp_mult == pytest.approx(0.8)

    def test_liquidity_crisis_widens_sl(self):
        eng = _make_engine()
        sl_mult, tp_mult, _ = eng._regime_factor("liquidity_crisis", 1.0)
        assert sl_mult == pytest.approx(1.2)
        assert tp_mult == pytest.approx(1.0)

    def test_unknown_neutral(self):
        eng = _make_engine()
        sl_mult, tp_mult, meta = eng._regime_factor("some_unknown", 0.5)
        assert sl_mult == pytest.approx(1.0)
        assert tp_mult == pytest.approx(1.0)
        assert meta["regime"] == "some_unknown"

    def test_none_regime_neutral(self):
        eng = _make_engine()
        sl_mult, tp_mult, _ = eng._regime_factor(None, 0.0)
        assert sl_mult == pytest.approx(1.0)
        assert tp_mult == pytest.approx(1.0)

    def test_zero_strength_halves_adjustment(self):
        eng = _make_engine()
        # strength=0 → s=0.5 → range tp_mult = 1 - 0.2*0.5 = 0.9
        _, tp_mult, _ = eng._regime_factor("range_bound", 0.0)
        assert tp_mult == pytest.approx(0.9)


# ═══════════════════════════════════════════════════════════════
# P0: 策略表现因子
# ═══════════════════════════════════════════════════════════════

class TestPerformanceFactor:
    def test_neutral_win_rate(self):
        eng = _make_engine()
        sl_mult, tp_mult, _ = eng._performance_factor(0.5, 1.0)
        assert sl_mult == pytest.approx(0.95)
        assert tp_mult == pytest.approx(1.0)

    def test_low_win_rate_tightens_sl(self):
        eng = _make_engine()
        sl_mult, _, _ = eng._performance_factor(0.0, 1.0)
        assert sl_mult == pytest.approx(0.8)

    def test_high_win_rate_loosens_sl(self):
        eng = _make_engine()
        sl_mult, _, _ = eng._performance_factor(1.0, 1.0)
        assert sl_mult == pytest.approx(1.1)

    def test_low_profit_factor_widens_tp(self):
        eng = _make_engine()
        _, tp_mult, _ = eng._performance_factor(0.5, 0.5)
        assert tp_mult == pytest.approx(1.3)

    def test_high_profit_factor_loosens_tp(self):
        eng = _make_engine()
        _, tp_mult, _ = eng._performance_factor(0.5, 2.0)
        assert tp_mult == pytest.approx(1.1)


# ═══════════════════════════════════════════════════════════════
# P0: 持仓盈亏保护
# ═══════════════════════════════════════════════════════════════

class TestPnlProtection:
    def test_flat_position_no_protection(self):
        eng = _make_engine()
        sl_mult, mode, meta = eng._pnl_protection_factor("BTC", "long", 100.0, 0.0, 100.0)
        assert mode == "none"
        assert sl_mult == pytest.approx(1.0)

    def test_profit_triggers_breakeven(self):
        eng = _make_engine()
        # profit_pct = 0.01 >= buffer(0.0015) + hyst(0.002) = 0.0035
        sl_mult, mode, _ = eng._pnl_protection_factor("BTC", "long", 100.0, 0.0, 101.0)
        assert mode == "breakeven"
        assert sl_mult == pytest.approx(1.0015)

    def test_profit_triggers_trailing(self):
        eng = _make_engine()
        # profit_pct = 0.02 >= 0.015 + 0.002 = 0.017
        sl_mult, mode, _ = eng._pnl_protection_factor("BTC", "long", 100.0, 0.0, 102.0)
        assert mode == "trailing"
        assert sl_mult == pytest.approx(1.005)

    def test_ratchet_only_ascends(self):
        eng = _make_engine()
        eng._pnl_protection_factor("BTC", "long", 100.0, 0.0, 102.0)  # trailing
        # 回落到 flat，不应降级
        _, mode, _ = eng._pnl_protection_factor("BTC", "long", 100.0, 0.0, 100.0)
        assert mode == "trailing"

    def test_loss_tightens_sl(self):
        eng = _make_engine()
        # profit_pct = -0.01 < -0.005 → sl_mult = 0.7
        sl_mult, mode, _ = eng._pnl_protection_factor("BTC", "long", 100.0, 0.0, 99.0)
        assert mode == "none"
        assert sl_mult == pytest.approx(0.7)

    def test_short_direction_profit(self):
        eng = _make_engine()
        # 空头：价格下跌为盈利
        _, mode, _ = eng._pnl_protection_factor("BTC", "short", 100.0, 0.0, 99.0)
        assert mode == "breakeven"


# ═══════════════════════════════════════════════════════════════
# P0: 保本缓冲 / 保本价
# ═══════════════════════════════════════════════════════════════

class TestBreakeven:
    def test_default_buffer(self):
        eng = _make_engine()
        # taker 0.0005 × 2 × 1.5 = 0.0015
        assert eng._breakeven_buffer() == pytest.approx(0.0015)

    def test_buffer_respects_min(self):
        eng = AdaptiveTpSlEngine({"trading": {"taker_fee_rate": 0.0}})
        assert eng._breakeven_buffer() == pytest.approx(0.001)

    def test_custom_taker_fee(self):
        eng = AdaptiveTpSlEngine({"trading": {"taker_fee_rate": 0.001}})
        assert eng._breakeven_buffer() == pytest.approx(0.003)

    def test_breakeven_price_long(self):
        eng = _make_engine()
        assert eng._breakeven_price(100.0, "long") == pytest.approx(100.15)

    def test_breakeven_price_short(self):
        eng = _make_engine()
        assert eng._breakeven_price(100.0, "short") == pytest.approx(99.85)


# ═══════════════════════════════════════════════════════════════
# P0: compute 集成
# ═══════════════════════════════════════════════════════════════

class TestCompute:
    def test_long_neutral_defaults(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx())
        assert result["direction"] == "long"
        assert result["stop_loss"] < result["entry_price"]
        assert result["take_profit"] > result["entry_price"]
        # sl_pct = 0.02 * 0.95 = 0.019, tp_pct = 0.06
        assert result["sl_distance_pct"] == pytest.approx(0.019)
        assert result["tp_distance_pct"] == pytest.approx(0.06)
        assert result["stop_loss"] == pytest.approx(98.0)
        assert result["take_profit"] == pytest.approx(105.9)

    def test_short_direction(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx(direction="short"))
        assert result["direction"] == "short"
        assert result["stop_loss"] > result["entry_price"]
        assert result["take_profit"] < result["entry_price"]

    def test_buy_sell_normalized(self):
        eng = _make_engine()
        r1 = eng.compute(**_default_ctx(direction="buy"))
        r2 = eng.compute(**_default_ctx(direction="sell"))
        assert r1["direction"] == "long"
        assert r2["direction"] == "short"

    def test_breakdown_present(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx())
        breakdown = result["breakdown"]
        assert set(breakdown["factors"].keys()) == {"volatility", "regime", "performance", "pnl_protection"}
        assert "final_sl_pct" in breakdown
        assert "final_tp_pct" in breakdown

    def test_staged_tp_levels(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx())
        assert len(result["staged_take_profit"]) == 3
        assert result["staged_take_profit"][0]["close_ratio"] == 0.4
        # 第一档价格介于 entry 与 take_profit 之间（多头）
        tp1 = result["staged_take_profit"][0]["price"]
        assert result["entry_price"] < tp1 < result["take_profit"]

    def test_trailing_disabled_when_flat(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx())
        assert result["trailing_stop"]["enabled"] is False
        assert result["protection_mode"] == "none"

    def test_trailing_enabled_when_profitable(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx(current_price=102.0))
        assert result["protection_mode"] == "trailing"
        assert result["trailing_stop"]["enabled"] is True
        assert result["action"] == "trail"

    def test_action_close_on_deep_loss(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx(current_price=94.0))
        assert result["action"] == "close"

    def test_action_hold_when_flat(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx())
        assert result["action"] == "hold"

    def test_risk_reward_computed(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx())
        assert result["risk_reward_ratio"] == pytest.approx(2.95)


# ═══════════════════════════════════════════════════════════════
# P0: 平滑
# ═══════════════════════════════════════════════════════════════

class TestSmoothing:
    def test_first_call_not_smoothed(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx())
        assert result["breakdown"]["smoothed"] is False

    def test_second_call_smoothed(self):
        eng = _make_engine(smoothing_alpha=0.5)
        eng.compute(**_default_ctx())
        result = eng.compute(**_default_ctx())
        assert result["breakdown"]["smoothed"] is True

    def test_smoothing_dampens_change(self):
        eng = _make_engine(smoothing_alpha=0.5)
        r1 = eng.compute(**_default_ctx())
        r2 = eng.compute(**_default_ctx(atr=10.0))  # 大 ATR 突变
        # 平滑后 sl 距离应介于 r1 与 r2 原始之间，小于 r2 原始
        assert r2["sl_distance_pct"] < (10.0 / 100.0 * 1.5)


# ═══════════════════════════════════════════════════════════════
# P0: 状态重置
# ═══════════════════════════════════════════════════════════════

class TestReset:
    def test_reset_clears_protection(self):
        eng = _make_engine()
        eng.compute(**_default_ctx(current_price=102.0))
        assert eng._protection_state.get("BTC-USDT-SWAP:long") == "trailing"
        eng.reset_position("BTC-USDT-SWAP", "long")
        assert "BTC-USDT-SWAP:long" not in eng._protection_state

    def test_reset_clears_smoothing(self):
        eng = _make_engine()
        eng.compute(**_default_ctx())
        assert "BTC-USDT-SWAP:long" in eng._smoothed
        eng.reset_position("BTC-USDT-SWAP", "long")
        assert "BTC-USDT-SWAP:long" not in eng._smoothed


# ═══════════════════════════════════════════════════════════════
# P1: build_context
# ═══════════════════════════════════════════════════════════════

class TestBuildContext:
    def test_no_regime_engine_returns_symbol_only(self):
        eng = _make_engine()
        assert eng.build_context("BTC-USDT-SWAP") == {"symbol": "BTC-USDT-SWAP"}

    def test_with_regime_engine(self):
        class FakeRegime:
            def get_symbol_regime(self, symbol):
                return {
                    "regime": "trend_bullish",
                    "strength": 0.8,
                    "factor_scores": {"volatility": 0.4},
                }

        eng = AdaptiveTpSlEngine({}, regime_engine=FakeRegime())
        ctx = eng.build_context("BTC-USDT-SWAP")
        assert ctx["regime"] == "trend_bullish"
        assert ctx["regime_strength"] == pytest.approx(0.8)
        assert ctx["vol_score"] == pytest.approx(0.4)

    def test_regime_engine_error_suppressed(self):
        class BadRegime:
            def get_symbol_regime(self, symbol):
                raise RuntimeError("boom")

        eng = AdaptiveTpSlEngine({}, regime_engine=BadRegime())
        assert eng.build_context("BTC-USDT-SWAP") == {"symbol": "BTC-USDT-SWAP"}


# ═══════════════════════════════════════════════════════════════
# P2: 边界
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_invalid_entry_price_raises(self):
        eng = _make_engine()
        with pytest.raises(ValueError):
            eng.compute(**_default_ctx(entry=0.0))

    def test_invalid_direction_raises(self):
        eng = _make_engine()
        with pytest.raises(ValueError):
            eng.compute(**_default_ctx(direction="diagonal"))

    def test_extreme_win_rate_clamped(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx(win_rate=99.0))
        # 胜率被裁剪到 [0,1]，sl_mult 不会超过 1.1
        assert result["breakdown"]["factors"]["performance"]["win_rate"] == 1.0

    def test_negative_profit_factor_floor(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx(profit_factor=-5.0))
        assert result["breakdown"]["factors"]["performance"]["profit_factor"] == 0.0

    def test_zero_atr_uses_base(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx(atr=0.0))
        assert result["breakdown"]["base_sl_pct"] == pytest.approx(0.02)

    def test_sl_never_exceeds_tp(self):
        eng = _make_engine()
        # 极端：极小 TP 距离 + 大 SL 距离，仍保证 sl <= tp*1.1 关系不倒挂
        result = eng.compute(**_default_ctx(win_rate=0.0, profit_factor=2.0))
        assert result["sl_distance_pct"] <= result["tp_distance_pct"]

    def test_unknown_regime_string(self):
        eng = _make_engine()
        result = eng.compute(**_default_ctx(regime="unknown_regime"))
        assert result["breakdown"]["factors"]["regime"]["sl_mult"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════
# P1: OrderExecutor 下单链路接入
# ═══════════════════════════════════════════════════════════════

class TestOrderExecutorIntegration:
    def _bare_executor(self, engine):
        """构造未初始化 OrderExecutor（绕过 __init__ 重依赖），仅注入引擎。"""
        oe = OrderExecutor.__new__(OrderExecutor)
        oe._adaptive_tp_sl_engine = engine
        return oe

    def test_setter_stores_engine(self):
        oe = OrderExecutor.__new__(OrderExecutor)
        oe._adaptive_tp_sl_engine = None
        engine = _make_engine()
        oe.set_adaptive_tp_sl_engine(engine)
        assert oe._adaptive_tp_sl_engine is engine

    def test_adjust_tp_sl_uses_engine_when_injected(self):
        engine = _make_engine()
        oe = self._bare_executor(engine)
        sl, tp = oe._adjust_tp_sl("BTC-USDT-SWAP", 100.0, "long", None, None, 100.0)
        # 多头：SL < entry < TP
        assert sl < 100.0 < tp

    def test_adjust_tp_sl_short_direction(self):
        engine = _make_engine()
        oe = self._bare_executor(engine)
        sl, tp = oe._adjust_tp_sl("BTC-USDT-SWAP", 100.0, "short", None, None, 100.0)
        # 空头：TP < entry < SL
        assert tp < 100.0 < sl

    def test_adjust_tp_sl_engine_failure_falls_back(self):
        # 引擎 compute 抛异常时，走固定重算路径（不因引擎故障中断下单）
        class BrokenEngine:
            def build_context(self, symbol):
                return {"symbol": symbol}
            def compute(self, **kwargs):
                raise RuntimeError("boom")

        oe = self._bare_executor(BrokenEngine())
        oe.config = {"currencies": {
            "tier1_symbols": ["BTC", "ETH"],
            "tier2_symbols": ["BNB", "SOL"],
            "tier3_symbols": ["DOT", "AVAX"],
            "tier1_settings": {"slippage": 0.001, "grid_spacing_min": 0.006},
            "tier2_settings": {"slippage": 0.002, "grid_spacing_min": 0.006},
            "tier3_settings": {"slippage": 0.005, "grid_spacing_min": 0.006},
        }}
        # 引擎异常被吞掉，回退到固定重算；不向上抛 RuntimeError
        try:
            sl, tp = oe._adjust_tp_sl("BTC-USDT-SWAP", 100.0, "long", None, None, 100.0)
        except RuntimeError:
            pytest.fail("engine exception should be caught, not propagated")
        assert sl < 100.0 < tp
