"""MarketRegimeEngine 企业级市场状态识别 — 正式单元测试。

覆盖矩阵（聚焦本次「强化企业级市场状态识别」改动）：
  P0  初始化 & 企业级稳定性参数解析/裁剪
  P0  因子得分 EMA 平滑 _apply_smoothing
  P0  状态迟滞 _resolve_regime（进入/退出阈值带）
  P0  结果裁剪 _build_result
  P0  状态稳定性追踪 _track_regime_transition（切换计数/历史/稳定标志）
  P0  状态持续时长 _get_regime_age_seconds
  P1  _update_regime 集成管线（短命状态置信度降级 + 稳定后不降级 + 币种独立状态）
  P2  边界（空因子、未知状态、自定义迟滞幅度）
"""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.market_regime_engine import (
    MarketRegime,
    MarketRegimeEngine,
    MarketSubtype,
)


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _make_engine(**overrides) -> MarketRegimeEngine:
    """构造带 market_regime 配置段的引擎，允许覆盖稳定性参数。"""
    cfg = {"market_regime": {
        "watched_symbols": ["BTC-USDT-SWAP"],
        "trend_adx_period": 14,
        "trend_adx_floor": 15,
        "trend_adx_saturation": 40,
        "trend_mtf_weight": 0.4,
    }}
    for key, value in overrides.items():
        cfg["market_regime"][key] = value
    return MarketRegimeEngine(cfg)


def _neutral_scores(overrides=None) -> dict:
    """构造中性因子得分（除指定项外全为 0，避免触发其它状态）。"""
    scores = {
        "trend": 0.0,
        "volatility": 0.0,
        "funding_rate": 0.0,
        "liquidity": 0.0,
        "momentum": 0.0,
        "sentiment": 0.0,
    }
    if overrides:
        scores.update(overrides)
    return scores


def _apply_transition(eng: MarketRegimeEngine, new_regime: MarketRegime):
    """镜像生产流程：先追踪切换，再把 new_regime 写回 _current_regime。

    _track_regime_transition 以 self._current_regime 作为“上一状态”进行对比，
    真实流程中 _update_regime 在追踪之后才更新 _current_regime。
    """
    eng._track_regime_transition(new_regime)
    eng._current_regime = new_regime


class TestPositionAdjustmentFusion:
    def test_adjustment_uses_fused_reversal_not_native_trend(self):
        engine = _make_engine()
        engine._current_regime = MarketRegime.TREND_BULLISH
        engine._current_strength = 0.8
        engine._current_subtype = MarketSubtype.STRONG
        engine.set_regime_arbiter(SimpleNamespace(arbitrate=lambda symbol=None: {
            "regime": "reversal",
            "strength": 0.9,
            "confidence": 0.8,
        }))

        adjustment = engine.get_position_adjustment()
        recommendation = engine.get_strategy_recommendation("grid")

        assert adjustment["overall"] == pytest.approx(0.6)
        assert adjustment["grid"] == pytest.approx(0.5)
        assert recommendation["regime"] == "reversal"
        assert recommendation["adjustment_factor"] == pytest.approx(0.5)
        assert "reversal" in recommendation["reason"].lower()

    @pytest.mark.parametrize(
        ("regime", "expected"),
        [
            (MarketRegime.BREAKOUT, {"overall": 0.9, "trend": 1.1, "grid": 0.6, "spot_grid": 0.6, "spot_martingale": 0.4}),
            (MarketRegime.BREAKDOWN, {"overall": 0.8, "trend": 1.0, "grid": 0.5, "spot_grid": 0.5, "spot_martingale": 0.3}),
            (MarketRegime.REVERSAL, {"overall": 0.6, "trend": 0.7, "grid": 0.5, "spot_grid": 0.5, "spot_martingale": 0.3}),
        ],
    )
    def test_special_regimes_have_conservative_strategy_adjustments(self, regime, expected):
        engine = _make_engine()
        engine._current_regime = regime

        adjustment = engine.get_position_adjustment()

        for strategy, factor in expected.items():
            assert adjustment[strategy] == pytest.approx(factor)


# ═══════════════════════════════════════════════════════════════
# P0: 初始化 & 企业级稳定性参数
# ═══════════════════════════════════════════════════════════════

class TestInit:
    def test_default_stability_params(self):
        eng = _make_engine()
        assert eng._smoothing_alpha == 0.5
        assert eng._regime_hysteresis_margin == 0.08
        assert eng._min_stable_updates == 3
        assert eng._stability_confidence_discount == 0.7
        assert eng._history_max_len == 100

    def test_initial_stability_state(self):
        eng = _make_engine()
        assert eng._stable is False
        assert eng._regime_switch_count == 0
        assert eng._regime_consecutive_updates == 0
        assert eng._regime_history == []
        assert eng._regime_started_at is None
        assert eng._smoothed_scores == {}
        assert eng._smoothed_symbol_scores == {}

    def test_custom_stability_params(self):
        eng = _make_engine(
            smoothing_alpha=0.3,
            regime_hysteresis_margin=0.15,
            min_stable_updates=5,
            stability_confidence_discount=0.4,
            history_max_len=50,
        )
        assert eng._smoothing_alpha == 0.3
        assert eng._regime_hysteresis_margin == 0.15
        assert eng._min_stable_updates == 5
        assert eng._stability_confidence_discount == 0.4
        assert eng._history_max_len == 50

    def test_smoothing_alpha_clamped(self):
        assert _make_engine(smoothing_alpha=2.0)._smoothing_alpha == 1.0
        assert _make_engine(smoothing_alpha=-1.0)._smoothing_alpha == 0.0

    def test_hysteresis_margin_clamped(self):
        assert _make_engine(regime_hysteresis_margin=-0.5)._regime_hysteresis_margin == 0.0

    def test_min_stable_updates_clamped(self):
        assert _make_engine(min_stable_updates=0)._min_stable_updates == 1

    def test_confidence_discount_clamped(self):
        assert _make_engine(stability_confidence_discount=1.5)._stability_confidence_discount == 1.0
        assert _make_engine(stability_confidence_discount=-0.5)._stability_confidence_discount == 0.0

    def test_history_max_len_clamped(self):
        assert _make_engine(history_max_len=0)._history_max_len == 1


# ═══════════════════════════════════════════════════════════════
# P0: 因子得分 EMA 平滑 _apply_smoothing
# ═══════════════════════════════════════════════════════════════

class TestApplySmoothing:
    def test_first_call_returns_raw(self):
        eng = _make_engine()
        raw = {"trend": 1.0, "volatility": -0.5}
        out = eng._apply_smoothing("BTC-USDT-SWAP", raw)
        assert out == raw
        assert eng._smoothed_scores == raw

    def test_second_call_applies_ema(self):
        eng = _make_engine()
        eng._apply_smoothing("BTC-USDT-SWAP", {"trend": 1.0, "volatility": -0.5})
        out = eng._apply_smoothing("BTC-USDT-SWAP", {"trend": 0.0, "volatility": 0.0})
        # alpha=0.5: 0.5*0 + 0.5*1 = 0.5 ; 0.5*0 + 0.5*(-0.5) = -0.25
        assert out["trend"] == pytest.approx(0.5)
        assert out["volatility"] == pytest.approx(-0.25)

    def test_symbol_separate_state(self):
        eng = _make_engine()
        eng._apply_smoothing("BTC-USDT-SWAP", {"trend": 1.0})
        out = eng._apply_smoothing("SOL-USDT-SWAP", {"trend": 0.2})
        # 不同币种首次调用返回原始值，互不影响
        assert out["trend"] == 0.2
        assert eng._smoothed_scores["trend"] == 1.0
        assert eng._smoothed_symbol_scores["SOL-USDT-SWAP"]["trend"] == 0.2

    def test_alpha_zero_keeps_previous(self):
        eng = _make_engine(smoothing_alpha=0.0)
        eng._apply_smoothing("BTC-USDT-SWAP", {"trend": 1.0})
        out = eng._apply_smoothing("BTC-USDT-SWAP", {"trend": 0.0})
        assert out["trend"] == 1.0

    def test_alpha_one_returns_raw(self):
        eng = _make_engine(smoothing_alpha=1.0)
        eng._apply_smoothing("BTC-USDT-SWAP", {"trend": 1.0})
        out = eng._apply_smoothing("BTC-USDT-SWAP", {"trend": 0.0})
        assert out["trend"] == 0.0


# ═══════════════════════════════════════════════════════════════
# P0: 状态迟滞 _resolve_regime
# ═══════════════════════════════════════════════════════════════

class TestResolveRegimeHysteresis:
    def test_directional_factor_weights_change_regime(self):
        scores = _neutral_scores({"trend": 0.4, "momentum": -0.8})
        default_regime, *_ = _make_engine()._resolve_regime(scores)
        momentum_weighted_regime, *_ = _make_engine(
            factor_weights={"trend": 0.1, "momentum": 0.8, "sentiment": 0.05}
        )._resolve_regime(scores)

        assert default_regime == MarketRegime.RANGE_BOUND
        assert momentum_weighted_regime == MarketRegime.BREAKDOWN

    def test_volatility_enter_requires_higher_threshold(self):
        eng = _make_engine()  # hyst=0.08，进入阈值 0.68
        regime, *_ = eng._resolve_regime(_neutral_scores({"volatility": 0.55}))
        assert regime == MarketRegime.RANGE_BOUND

    def test_volatility_exit_uses_lower_threshold(self):
        eng = _make_engine()  # 退出阈值 0.52
        regime, *_ = eng._resolve_regime(
            _neutral_scores({"volatility": 0.55}),
            previous_regime=MarketRegime.EXTREME_VOLATILITY,
        )
        assert regime == MarketRegime.EXTREME_VOLATILITY

    def test_funding_enter_requires_higher_threshold(self):
        eng = _make_engine()  # 进入阈值 0.58
        regime, *_ = eng._resolve_regime(_neutral_scores({"funding_rate": 0.5}))
        assert regime == MarketRegime.RANGE_BOUND

    def test_funding_exit_uses_lower_threshold(self):
        eng = _make_engine()  # 退出阈值 0.42
        regime, *_ = eng._resolve_regime(
            _neutral_scores({"funding_rate": 0.5}),
            previous_regime=MarketRegime.FUNDING_CRUSH,
        )
        assert regime == MarketRegime.FUNDING_CRUSH

    def test_liquidity_enter_requires_lower_threshold(self):
        eng = _make_engine()  # 进入阈值 -0.58
        regime, *_ = eng._resolve_regime(_neutral_scores({"liquidity": -0.5}))
        assert regime == MarketRegime.RANGE_BOUND

    def test_liquidity_exit_uses_higher_threshold(self):
        eng = _make_engine()  # 退出阈值 -0.42
        regime, *_ = eng._resolve_regime(
            _neutral_scores({"liquidity": -0.5}),
            previous_regime=MarketRegime.LIQUIDITY_CRISIS,
        )
        assert regime == MarketRegime.LIQUIDITY_CRISIS

    def test_bullish_enter_requires_higher_threshold(self):
        eng = _make_engine()  # 进入阈值 0.28
        regime, *_ = eng._resolve_regime(_neutral_scores({"trend": 0.2}))
        assert regime == MarketRegime.RANGE_BOUND

    def test_bullish_exit_uses_lower_threshold(self):
        eng = _make_engine()  # 退出阈值 0.12
        regime, *_ = eng._resolve_regime(
            _neutral_scores({"trend": 0.2}),
            previous_regime=MarketRegime.TREND_BULLISH,
        )
        assert regime == MarketRegime.TREND_BULLISH

    def test_bearish_enter_requires_lower_threshold(self):
        eng = _make_engine()  # 进入阈值 -0.28
        regime, *_ = eng._resolve_regime(_neutral_scores({"trend": -0.2}))
        assert regime == MarketRegime.RANGE_BOUND

    def test_bearish_exit_uses_higher_threshold(self):
        eng = _make_engine()  # 退出阈值 -0.12
        regime, *_ = eng._resolve_regime(
            _neutral_scores({"trend": -0.2}),
            previous_regime=MarketRegime.TREND_BEARISH,
        )
        assert regime == MarketRegime.TREND_BEARISH

    def test_strong_bullish_subtype(self):
        eng = _make_engine()
        regime, subtype, strength, confidence, _ = eng._resolve_regime(
            _neutral_scores({"trend": 0.6})
        )
        assert regime == MarketRegime.TREND_BULLISH
        assert subtype == MarketSubtype.STRONG
        assert strength == pytest.approx(0.9)
        assert confidence == pytest.approx(0.84)

    def test_custom_hysteresis_margin_shifts_threshold(self):
        eng = _make_engine(regime_hysteresis_margin=0.2)  # 进入阈值 0.8
        regime, *_ = eng._resolve_regime(_neutral_scores({"volatility": 0.7}))
        assert regime == MarketRegime.RANGE_BOUND

    def test_no_previous_uses_strict_threshold(self):
        eng = _make_engine()
        regime, *_ = eng._resolve_regime(_neutral_scores({"volatility": 0.7}))
        assert regime == MarketRegime.EXTREME_VOLATILITY

    def test_breakout_detected_when_trend_and_momentum_converge(self):
        eng = _make_engine()
        regime, subtype, strength, confidence, _ = eng._resolve_regime(
            _neutral_scores({"trend": 0.8, "momentum": 0.9})
        )
        assert regime == MarketRegime.BREAKOUT
        assert subtype in {MarketSubtype.STRONG, MarketSubtype.MODERATE}
        assert strength > 0.0
        assert confidence > 0.7

    def test_reversal_detected_when_direction_conflicts_with_momentum(self):
        eng = _make_engine()
        regime, subtype, strength, confidence, _ = eng._resolve_regime(
            _neutral_scores({"trend": 0.2, "momentum": -0.8})
        )
        assert regime == MarketRegime.REVERSAL
        assert subtype == MarketSubtype.REVERSAL
        assert strength > 0.5
        assert confidence > 0.0


# ═══════════════════════════════════════════════════════════════
# P0: 结果裁剪 _build_result
# ═══════════════════════════════════════════════════════════════

class TestBuildResult:
    def test_clips_strength_and_confidence(self):
        regime, subtype, strength, confidence, breakdown = MarketRegimeEngine._build_result(
            MarketRegime.RANGE_BOUND,
            MarketSubtype.MODERATE,
            2.0,
            -0.5,
            {"trend": 0.1},
        )
        assert regime == MarketRegime.RANGE_BOUND
        assert subtype == MarketSubtype.MODERATE
        assert strength == 1.0
        assert confidence == 0.0
        assert breakdown == {"trend": 0.1}


# ═══════════════════════════════════════════════════════════════
# P0: 状态稳定性追踪 _track_regime_transition
# ═══════════════════════════════════════════════════════════════

class TestTrackRegimeTransition:
    def test_first_transition_starts_unstable(self):
        eng = _make_engine()  # min_stable_updates=3
        eng._track_regime_transition(MarketRegime.RANGE_BOUND)
        assert eng._regime_consecutive_updates == 1
        assert eng._regime_started_at is not None
        assert eng._stable is False
        assert eng._regime_switch_count == 0

    def test_repeated_regime_reaches_stable(self):
        eng = _make_engine()
        _apply_transition(eng, MarketRegime.RANGE_BOUND)
        _apply_transition(eng, MarketRegime.RANGE_BOUND)
        _apply_transition(eng, MarketRegime.RANGE_BOUND)
        assert eng._regime_consecutive_updates == 3
        assert eng._stable is True

    def test_switch_increments_count_and_resets(self):
        eng = _make_engine()
        _apply_transition(eng, MarketRegime.RANGE_BOUND)
        _apply_transition(eng, MarketRegime.RANGE_BOUND)
        _apply_transition(eng, MarketRegime.TREND_BULLISH)
        assert eng._regime_switch_count == 1
        assert eng._regime_consecutive_updates == 1
        assert eng._stable is False
        assert len(eng._regime_history) == 1
        assert eng._regime_history[0]["from"] == "range_bound"
        assert eng._regime_history[0]["to"] == "trend_bullish"
        assert eng._regime_history[0]["consecutive_updates"] == 2

    def test_min_stable_updates_config(self):
        eng = _make_engine(min_stable_updates=1)
        eng._track_regime_transition(MarketRegime.RANGE_BOUND)
        assert eng._stable is True

    def test_history_max_len_enforced(self):
        eng = _make_engine(history_max_len=2)
        _apply_transition(eng, MarketRegime.RANGE_BOUND)
        _apply_transition(eng, MarketRegime.TREND_BULLISH)
        _apply_transition(eng, MarketRegime.RANGE_BOUND)
        _apply_transition(eng, MarketRegime.TREND_BULLISH)
        assert len(eng._regime_history) == 2
        assert eng._regime_history[-1]["to"] == "trend_bullish"


# ═══════════════════════════════════════════════════════════════
# P0: 状态持续时长 _get_regime_age_seconds
# ═══════════════════════════════════════════════════════════════

class TestGetRegimeAge:
    def test_age_zero_when_not_started(self):
        eng = _make_engine()
        assert eng._get_regime_age_seconds() == 0.0

    def test_age_non_negative_after_start(self):
        eng = _make_engine()
        eng._track_regime_transition(MarketRegime.RANGE_BOUND)
        assert eng._get_regime_age_seconds() >= 0.0


# ═══════════════════════════════════════════════════════════════
# P1: _update_regime 集成管线
# ═══════════════════════════════════════════════════════════════

class TestUpdateRegimeIntegration:
    async def test_fresh_funding_cache_allows_temporary_empty_response(self):
        eng = _make_engine()
        detector_calls = []

        async def detect_regime(ohlcv_data, symbol, timeframe=None, timeframe_data=None):
            detector_calls.append((ohlcv_data, symbol, timeframe, timeframe_data))
            return {"regime": "ranging"}

        eng.set_detector(SimpleNamespace(detect_regime=detect_regime))
        eng.okx_client = SimpleNamespace(
            get_ticker_async=AsyncMock(return_value={"last": "100", "bidPx": "99", "askPx": "101"}),
            get_funding_rate_async=AsyncMock(return_value=None),
            get_kline_async=AsyncMock(return_value=[{"timestamp": 1, "close": "100"}]),
        )
        eng._funding_cache["BTC-USDT-SWAP"] = {"funding_rate": 0.0}
        eng._funding_cache_time["BTC-USDT-SWAP"] = datetime.now()

        collected = await eng._collect_symbol_data("BTC-USDT-SWAP")

        assert collected is True
        assert len(detector_calls) == 1
        bars, symbol, timeframe, timeframe_data = detector_calls[0]
        assert symbol == "BTC-USDT-SWAP"
        assert timeframe == "medium"
        assert timeframe_data == {"medium": bars, "long": bars}
        eng.okx_client.get_kline_async.assert_any_await("BTC-USDT-SWAP", "4H", limit=60)

    async def test_incomplete_collection_keeps_previous_regime_and_timestamp(self):
        eng = _make_engine()
        eng.okx_client = object()
        eng._current_regime = MarketRegime.TREND_BULLISH
        eng._data_collection_ok = {"BTC-USDT-SWAP": True}

        async def fail_collection():
            eng._data_collection_ok["BTC-USDT-SWAP"] = False

        eng._collect_factor_data = AsyncMock(side_effect=fail_collection)
        eng._calculate_factor_scores = AsyncMock(return_value=_neutral_scores())

        await eng._update_regime()

        assert eng._current_regime == MarketRegime.TREND_BULLISH
        assert eng._last_update is None
        assert eng._data_stale is True
        eng._calculate_factor_scores.assert_not_awaited()

    async def test_first_update_discounts_confidence(self):
        eng = _make_engine()
        eng._collect_factor_data = AsyncMock()
        eng._calculate_factor_scores = AsyncMock(
            return_value=_neutral_scores({"trend": 0.6})
        )
        await eng._update_regime()
        assert eng._current_regime == MarketRegime.TREND_BULLISH
        assert eng._stable is False
        # 置信度 0.84 被短命状态折扣 0.7 → 0.588
        assert eng._regime_confidence == pytest.approx(0.84 * 0.7, abs=1e-6)

    async def test_stable_regime_no_discount(self):
        eng = _make_engine()
        eng._collect_factor_data = AsyncMock()
        eng._calculate_factor_scores = AsyncMock(
            return_value=_neutral_scores({"trend": 0.6})
        )
        for _ in range(3):
            await eng._update_regime()
        assert eng._stable is True
        assert eng._regime_confidence == pytest.approx(0.84, abs=1e-6)

    async def test_symbol_regimes_computed_independently(self):
        eng = _make_engine(watched_symbols=["BTC-USDT-SWAP", "SOL-USDT-SWAP"])
        eng._collect_factor_data = AsyncMock()

        async def fake_scores(symbol=None):
            symbol = symbol or "BTC-USDT-SWAP"
            if symbol == "SOL-USDT-SWAP":
                return _neutral_scores({"trend": -0.6})
            return _neutral_scores({"trend": 0.6})

        eng._calculate_factor_scores = fake_scores
        await eng._update_regime()
        assert eng._symbol_regimes["BTC-USDT-SWAP"]["regime"] == "trend_bullish"
        assert eng._symbol_regimes["SOL-USDT-SWAP"]["regime"] == "trend_bearish"


# ═══════════════════════════════════════════════════════════════
# P2: 边界 & 未知状态
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_scores_returns_range(self):
        eng = _make_engine()
        regime, subtype, strength, confidence, _ = eng._resolve_regime({})
        assert regime == MarketRegime.RANGE_BOUND
        assert subtype == MarketSubtype.MODERATE
        assert 0.0 <= strength <= 1.0
        assert 0.0 <= confidence <= 1.0

    def test_get_regime_unknown_before_update(self):
        eng = _make_engine()
        info = eng.get_regime()
        assert info["regime"] == "unknown"
        assert info["subtype"] == "unknown"
        assert info["last_update"] is None
