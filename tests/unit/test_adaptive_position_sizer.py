"""AdaptivePositionSizer 企业级自适应仓位管理 — 正式单元测试。

覆盖矩阵（聚焦「强化企业级自适应仓位管理模块」改动）：
  P0  初始化 & 配置解析/裁剪
  P0  Kelly 融合（有优势 / 无优势回退默认）
  P0  引擎层因子（权益模式 / 信号强度 / 波动率）
  P0  风险预算式 vs 名义价值式
  P0  裁剪（position_limit / 杠杆上限 / 最小名义价值）
  P0  短路拒绝（非法余额 / 非法价格 / 紧急模式）
  P0  币种 tier 解析
  P1  相对乘数 compute_multiplier
  P1  配置摘要 get_config_summary / to_dict
  P2  边界（regime 归一化、极端输入、缺省配置）
"""

import pytest

from core.adaptive_position_sizer import (
    AdaptivePositionSizer,
    PositionSizingResult,
    _normalize_regime,
)


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _make_config(aps: dict = None, trading: dict = None,
                 adaptive_kelly: dict = None) -> dict:
    """构造包含 currencies/trading/adaptive_kelly 段的完整配置。"""
    cfg = {
        "currencies": {
            "tier1_symbols": ["BTC", "ETH"],
            "tier2_symbols": ["SOL", "XRP"],
            "tier3_symbols": ["DOGE", "POL"],
            "tier1_settings": {
                "leverage_min": 1, "leverage_max": 20, "leverage_default": 5,
                "position_limit": 0.5, "grid_spacing_min": 0.001,
                "grid_spacing_max": 0.05, "slippage": 0.001,
            },
            "tier2_settings": {
                "leverage_min": 1, "leverage_max": 15, "leverage_default": 5,
                "position_limit": 0.3, "grid_spacing_min": 0.001,
                "grid_spacing_max": 0.05, "slippage": 0.001,
            },
            "tier3_settings": {
                "leverage_min": 1, "leverage_max": 10, "leverage_default": 3,
                "position_limit": 0.2, "grid_spacing_min": 0.001,
                "grid_spacing_max": 0.05, "slippage": 0.001,
            },
        },
        "trading": {
            "total_capital": 1000.0,
            "risk_per_trade": 0.02,
            "max_position_ratio": 0.8,
            "default_leverage": 5,
            "max_leverage": 20,
            "grid_allocation": 0.12,
            "trend_allocation": 0.25,
            "scalping_allocation": 0.28,
            "arbitrage_allocation": 0.13,
            "spot_grid_allocation": 0.12,
            "spot_martingale_allocation": 0.10,
        },
        "adaptive_kelly": {},
        "adaptive_position_sizing": {},
    }
    if trading:
        cfg["trading"].update(trading)
    if adaptive_kelly:
        cfg["adaptive_kelly"].update(adaptive_kelly)
    if aps:
        cfg["adaptive_position_sizing"].update(aps)
    return cfg


def _make_engine(aps: dict = None, trading: dict = None) -> AdaptivePositionSizer:
    return AdaptivePositionSizer(_make_config(aps=aps, trading=trading))


# 中性历史表现（无边际优势，触发 Kelly 回退默认）
def _neutral_perf():
    return dict(win_rate=0.5, avg_win=0.0, avg_loss=0.0, trade_count=0)


# ═══════════════════════════════════════════════════════════════
# P0 初始化 & 配置
# ═══════════════════════════════════════════════════════════════

class TestInitialization:
    def test_default_config(self):
        eng = AdaptivePositionSizer({})
        assert eng._default_risk_fraction == pytest.approx(0.02)
        assert eng._min_notional_usd == pytest.approx(5.0)
        assert eng._max_position_ratio == pytest.approx(0.8)

    def test_custom_config_applied(self):
        eng = _make_engine(aps={
            "default_risk_fraction": 0.05,
            "min_notional_usd": 1.0,
            "max_risk_fraction": 0.40,
        })
        assert eng._default_risk_fraction == pytest.approx(0.05)
        assert eng._min_notional_usd == pytest.approx(1.0)
        assert eng._max_risk_fraction == pytest.approx(0.40)

    def test_clamp_default_risk_fraction(self):
        eng = _make_engine(aps={"default_risk_fraction": 5.0})
        assert eng._default_risk_fraction == pytest.approx(1.0)

    def test_clamp_max_risk_fraction(self):
        eng = _make_engine(aps={"max_risk_fraction": -1.0})
        assert eng._max_risk_fraction == pytest.approx(0.0)

    def test_trading_defaults_parsed(self):
        eng = AdaptivePositionSizer(_make_config(trading={"risk_per_trade": 0.03}))
        assert eng._risk_per_trade == pytest.approx(0.03)


# ═══════════════════════════════════════════════════════════════
# P0 Kelly 融合
# ═══════════════════════════════════════════════════════════════

class TestKellyFusion:
    def test_positive_edge_produces_kelly(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", win_rate=0.6, avg_win=1.0, avg_loss=0.5,
            trade_count=100, stop_distance=2.0,
        )
        assert r.allowed
        assert r.kelly_fraction > 0
        # Kelly 上限 = max_kelly_fraction(0.25) * default_fraction(0.5) = 0.125
        assert r.kelly_fraction <= 0.125 + 1e-6

    def test_no_data_falls_back_to_default(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
        )
        assert r.allowed
        assert r.breakdown["kelly_fallback_default"] is True
        assert r.kelly_fraction == pytest.approx(0.02)

    def test_losing_edge_falls_back(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", win_rate=0.2, avg_win=0.3, avg_loss=1.0,
            trade_count=50, stop_distance=2.0,
        )
        assert r.breakdown["kelly_fallback_default"] is True


# ═══════════════════════════════════════════════════════════════
# P0 引擎层因子
# ═══════════════════════════════════════════════════════════════

class TestEngineFactors:
    def test_equity_multiplier_scales(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0, "min_risk_fraction": 0.0})
        base = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
            equity_multiplier=1.0,
        )
        boosted = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
            equity_multiplier=2.0,
        )
        assert boosted.position_multiplier == pytest.approx(base.position_multiplier * 2.0)
        assert boosted.quantity == pytest.approx(base.quantity * 2.0)

    def test_equity_emergency_rejects(self):
        eng = _make_engine()
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=1000.0,
            strategy_name="grid", equity_multiplier=0.0,
        )
        assert not r.allowed
        assert r.reject_reason == "equity_emergency"

    def test_signal_strength_monotonic(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0, "min_risk_fraction": 0.0})
        low = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
            signal_strength=0.2,
        )
        high = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
            signal_strength=0.9,
        )
        assert high.breakdown["signal_factor"] > low.breakdown["signal_factor"]
        assert high.quantity > low.quantity

    def test_high_volatility_reduces(self):
        eng = _make_engine()
        assert eng._volatility_factor(0.10) == pytest.approx(0.5)
        assert eng._volatility_factor(0.04) == pytest.approx(0.7)
        assert eng._volatility_factor(0.03) == pytest.approx(0.85)
        assert eng._volatility_factor(0.01) == pytest.approx(1.2)
        assert eng._volatility_factor(0.0) == pytest.approx(1.0)

    def test_disabled_factors(self):
        eng = _make_engine(aps={
            "min_notional_usd": 0.0,
            "use_signal_factor": False,
            "use_volatility_factor": False,
        })
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
            signal_strength=0.0, volatility=0.10,
        )
        assert r.breakdown["signal_factor"] == pytest.approx(1.0)
        assert r.breakdown["volatility_factor"] == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════
# P0 计算模式
# ═══════════════════════════════════════════════════════════════

class TestComputeModes:
    def test_risk_budget_mode_with_stop_distance(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
        )
        assert r.allowed
        assert r.mode == "risk_budget"
        assert r.risk_amount > 0
        assert r.quantity == pytest.approx(r.risk_amount / 2.0)

    def test_risk_budget_mode_with_atr(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0, "atr_sl_multiplier": 2.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), atr=1.0,
        )
        assert r.mode == "risk_budget"
        assert r.breakdown["stop_distance"] == pytest.approx(2.0)

    def test_notional_mode_without_stop(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(),
        )
        assert r.allowed
        assert r.mode == "notional"
        assert r.notional == pytest.approx(r.risk_amount)

    def test_stop_distance_priority_over_atr(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=3.0, atr=1.0,
        )
        assert r.breakdown["stop_distance"] == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════
# P0 裁剪
# ═══════════════════════════════════════════════════════════════

class TestCapping:
    def test_notional_capped_by_position_ratio(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0}, trading={"max_position_ratio": 0.1})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", win_rate=0.6, avg_win=1.0, avg_loss=0.5,
            trade_count=100,
        )
        assert r.allowed
        # 最大名义价值 = 100000 * 0.1 = 10000
        assert r.notional <= 10000.0 + 1e-6

    def test_leverage_capped_by_tier_max(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), leverage=50,
        )
        assert r.allowed
        assert r.leverage == pytest.approx(20.0)  # tier1 max

    def test_below_min_notional_rejects(self):
        eng = _make_engine(aps={"min_notional_usd": 100.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=10.0,
            strategy_name="grid", **_neutral_perf(),
        )
        assert not r.allowed
        assert "below_min_notional" in r.reject_reason

    def test_zero_risk_fraction_rejects(self):
        eng = _make_engine(aps={
            "min_notional_usd": 0.0,
            "default_risk_fraction": 0.0,
        })
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(),
        )
        assert not r.allowed
        assert r.reject_reason == "zero_risk_fraction"


# ═══════════════════════════════════════════════════════════════
# P0 短路拒绝
# ═══════════════════════════════════════════════════════════════

class TestRejections:
    def test_invalid_balance(self):
        r = _make_engine().compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=0.0,
            strategy_name="grid",
        )
        assert not r.allowed
        assert r.reject_reason == "invalid_balance"

    def test_invalid_price(self):
        r = _make_engine().compute(
            symbol="BTC-USDT-SWAP", price=0.0, account_balance=1000.0,
            strategy_name="grid",
        )
        assert not r.allowed
        assert r.reject_reason == "invalid_price"


# ═══════════════════════════════════════════════════════════════
# P0 币种 tier
# ═══════════════════════════════════════════════════════════════

class TestTierResolution:
    def test_tier1_symbol(self):
        eng = _make_engine()
        assert eng._resolve_tier("BTC-USDT-SWAP") == "tier1"
        assert eng._resolve_tier("ETH-USDT-SWAP") == "tier1"

    def test_tier2_symbol(self):
        eng = _make_engine()
        assert eng._resolve_tier("SOL-USDT-SWAP") == "tier2"

    def test_unknown_symbol_defaults_tier3(self):
        eng = _make_engine()
        assert eng._resolve_tier("ZZZ-USDT-SWAP") == "tier3"

    def test_missing_currencies_falls_back(self):
        eng = AdaptivePositionSizer({"trading": {}})
        assert eng._resolve_tier("BTC-USDT-SWAP") == "tier3"


# ═══════════════════════════════════════════════════════════════
# P1 compute_multiplier
# ═══════════════════════════════════════════════════════════════

class TestComputeMultiplier:
    def test_positive_edge_above_one(self):
        eng = _make_engine()
        m = eng.compute_multiplier(
            win_rate=0.6, avg_win=1.0, avg_loss=0.5, trade_count=100,
        )
        assert m["multiplier"] > 1.0
        assert m["kelly_fallback_default"] is False

    def test_no_edge_fallback(self):
        eng = _make_engine()
        m = eng.compute_multiplier(**_neutral_perf())
        assert m["kelly_fallback_default"] is True

    def test_equity_zero_returns_zero(self):
        eng = _make_engine()
        m = eng.compute_multiplier(
            win_rate=0.6, avg_win=1.0, avg_loss=0.5, trade_count=100,
            equity_multiplier=0.0,
        )
        assert m["multiplier"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════
# P1 摘要 & 序列化
# ═══════════════════════════════════════════════════════════════

class TestSummaryAndSerialization:
    def test_config_summary(self):
        eng = _make_engine()
        s = eng.get_config_summary()
        assert "default_risk_fraction" in s
        assert "kelly" in s

    def test_to_dict_roundtrip(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
        )
        d = r.to_dict()
        assert d["symbol"] == "BTC-USDT-SWAP"
        assert d["allowed"] is True
        assert "breakdown" in d
        assert d["quantity"] == pytest.approx(r.quantity)


# ═══════════════════════════════════════════════════════════════
# P2 边界
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_normalize_regime_from_state(self):
        assert _normalize_regime("uptrend") == "trending_up"
        assert _normalize_regime("downtrend") == "trending_down"
        assert _normalize_regime("range") == "ranging"

    def test_normalize_regime_passthrough(self):
        assert _normalize_regime("trending_up") == "trending_up"
        assert _normalize_regime("high_volatility") == "high_volatility"
        assert _normalize_regime("unknown") == "unknown"

    def test_normalize_regime_none(self):
        assert _normalize_regime(None) == "unknown"

    def test_zero_balance_short_circuit(self):
        r = _make_engine().compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=-1.0,
            strategy_name="grid",
        )
        assert not r.allowed
        assert r.reject_reason == "invalid_balance"

    def test_extreme_signal_clamped(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), stop_distance=2.0,
            signal_strength=5.0,
        )
        assert r.breakdown["signal_factor"] == pytest.approx(1.5)

    def test_negative_leverage_clamped_to_min(self):
        eng = _make_engine(aps={"min_notional_usd": 0.0})
        r = eng.compute(
            symbol="BTC-USDT-SWAP", price=100.0, account_balance=100000.0,
            strategy_name="grid", **_neutral_perf(), leverage=-5,
        )
        assert r.leverage >= 1.0
