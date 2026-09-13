"""ReversalTakeProfitEngine 行情反转智能计算落袋自适应引擎 — 单元测试。

覆盖矩阵：
  P0  初始化 & 配置解析
  P0  方向归一化
  P0  HMM 反转评分（regime / probabilities / early_warnings）
  P0  指标兜底（RSI 超买 / EMA 交叉 / 结构破坏）
  P0  动态止盈收紧（多头 / 空头 / 阈值以下不收紧）
  P0  反转落袋动作（full / partial / 未盈利不触发 / 冷却去重）
  P2  边界（非法方向 / 空数据 / 极端评分裁剪）
"""

import pytest

from core.reversal_take_profit_engine import ReversalTakeProfitEngine


def _make_engine(**overrides) -> ReversalTakeProfitEngine:
    """构造带 reversal_take_profit 配置段的引擎，允许覆盖参数。"""
    cfg = {"reversal_take_profit": {}}
    for key, value in overrides.items():
        cfg["reversal_take_profit"][key] = value
    return ReversalTakeProfitEngine(cfg)


def _hmm(regime="reversal", probs=None, warnings=None):
    return {
        "regime": regime,
        "probabilities": probs or {},
        "early_warnings": warnings or [],
    }


# ═══════════════════════════════════════════════════════════════
# P0: 初始化 & 配置解析
# ═══════════════════════════════════════════════════════════════

class TestInitialization:
    def test_defaults(self):
        eng = _make_engine()
        assert eng._enabled is True
        assert eng._hmm_reversal_regime == "reversal"
        assert eng._tighten_start_score == 0.4
        assert eng._tighten_max_factor == 0.5
        assert eng._partial_exit_score == 0.65
        assert eng._partial_exit_ratio == 0.5
        assert eng._full_exit_score == 0.85
        assert eng._min_profit_lock_pct == 0.002
        assert eng._cooldown_seconds == 300.0

    def test_config_override(self):
        eng = _make_engine(full_exit_score=0.9, partial_exit_ratio=0.6, cooldown_seconds=60)
        assert eng._full_exit_score == 0.9
        assert eng._partial_exit_ratio == 0.6
        assert eng._cooldown_seconds == 60.0


# ═══════════════════════════════════════════════════════════════
# P0: 方向归一化
# ═══════════════════════════════════════════════════════════════

class TestDirectionNormalization:
    def test_buy_long_normalized(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "buy", 100.0, 100.0)
        assert r.details["direction"] == "long"

    def test_sell_short_normalized(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "sell", 100.0, 100.0)
        assert r.details["direction"] == "short"


# ═══════════════════════════════════════════════════════════════
# P0: HMM 反转评分
# ═══════════════════════════════════════════════════════════════

class TestHmmScore:
    def test_reversal_regime_sets_high_score(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=_hmm("reversal"))
        assert r.hmm_reversal is True
        assert r.reversal_score == pytest.approx(0.80)
        assert "hmm" in r.reversal_source

    def test_non_reversal_regime_no_score(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=_hmm("trending"))
        assert r.hmm_reversal is False
        assert r.reversal_score == pytest.approx(0.0)
        assert r.reversal_source == "none"

    def test_reversal_probability_used(self):
        eng = _make_engine()
        hmm = _hmm("trending", probs={"reversal": 0.9})
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=hmm)
        assert r.reversal_score == pytest.approx(0.9)

    def test_early_warnings_boost_score(self):
        eng = _make_engine()
        hmm = _hmm("reversal", warnings=[{"type": "w1"}, {"type": "w2"}])
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=hmm)
        # 0.80 + 2 * 0.05 = 0.90
        assert r.reversal_score == pytest.approx(0.90)

    def test_none_hmm_result_ignored(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=None)
        assert r.hmm_reversal is False


# ═══════════════════════════════════════════════════════════════
# P0: 指标兜底
# ═══════════════════════════════════════════════════════════════

def _candles(closes):
    """把收盘价序列构造为 OKX K 线 list[list]（idx4=close）。"""
    return [[0, 0, 0, 0, c, 0] for c in closes]


class TestIndicatorFallback:
    def test_rsi_overbought_for_long(self):
        eng = _make_engine()
        # 持续上涨 → RSI 超买
        closes = [100 + i * 1.0 for i in range(30)]
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, ohlcv_data=_candles(closes))
        assert r.details["indicator_score"] >= 0.30
        assert "indicator" in r.reversal_source

    def test_rsi_oversold_for_short(self):
        eng = _make_engine()
        closes = [200 - i * 1.0 for i in range(30)]
        r = eng.compute("BTC", "trend", "short", 100.0, 99.0, ohlcv_data=_candles(closes))
        assert r.details["indicator_score"] >= 0.30

    def test_empty_ohlcv_no_indicator(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, ohlcv_data=[])
        assert r.details["indicator_score"] == 0.0

    def test_insufficient_data_no_indicator(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, ohlcv_data=_candles([100, 101, 102]))
        assert r.details["indicator_score"] == 0.0


# ═══════════════════════════════════════════════════════════════
# P0: 动态止盈收紧
# ═══════════════════════════════════════════════════════════════

class TestAdaptiveTp:
    def test_long_tightens_when_reversal(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0,
                        base_tp_price=110.0, hmm_result=_hmm("reversal"))
        # score=0.80 → factor = 1 - (0.8-0.4)/0.6*0.5 = 0.667 → tp=100+10*0.667=106.67
        assert r.adaptive_tp_price < 110.0
        assert r.adaptive_tp_price > 100.0
        assert r.tp_tighten_factor < 1.0

    def test_short_tightens_toward_entry(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "short", 100.0, 99.0,
                        base_tp_price=90.0, hmm_result=_hmm("reversal"))
        assert r.adaptive_tp_price > 90.0
        assert r.adaptive_tp_price < 100.0

    def test_no_tightening_below_threshold(self):
        eng = _make_engine()
        # 无反转 → score=0 → 不收紧
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, base_tp_price=110.0)
        assert r.adaptive_tp_price == pytest.approx(110.0)
        assert r.tp_tighten_factor == pytest.approx(1.0)

    def test_no_base_tp_returns_none(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=_hmm("reversal"))
        assert r.adaptive_tp_price is None


# ═══════════════════════════════════════════════════════════════
# P0: 反转落袋动作
# ═══════════════════════════════════════════════════════════════

class TestExitAction:
    def test_full_exit_on_high_score(self):
        eng = _make_engine()
        # 0.80 + 1 warning = 0.85 >= full_exit_score
        hmm = _hmm("reversal", warnings=[{"type": "w1"}])
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=hmm)
        assert r.exit_action == "close"
        assert r.partial_ratio == 1.0

    def test_partial_exit_on_mid_score(self):
        eng = _make_engine()
        # 0.80 >= partial(0.65) < full(0.85)
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=_hmm("reversal"))
        assert r.exit_action == "partial"
        assert r.partial_ratio == pytest.approx(0.5)

    def test_no_exit_when_not_profitable(self):
        eng = _make_engine()
        # 平价位，pnl=0 < min_profit_lock
        r = eng.compute("BTC", "trend", "long", 100.0, 100.0, hmm_result=_hmm("reversal"))
        assert r.exit_action == "none"

    def test_no_exit_when_score_low(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=_hmm("trending"))
        assert r.exit_action == "none"

    def test_cooldown_suppresses_repeat(self):
        eng = _make_engine(cooldown_seconds=300)
        hmm = _hmm("reversal", warnings=[{"type": "w1"}])
        first = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=hmm)
        assert first.exit_action == "close"
        second = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=hmm)
        assert second.exit_action == "none"


# ═══════════════════════════════════════════════════════════════
# P2: 边界
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_score_clamped_to_one(self):
        eng = _make_engine()
        # 大量 warnings 也不会超过 1.0
        warnings = [{"type": f"w{i}"} for i in range(50)]
        r = eng.compute("BTC", "trend", "long", 100.0, 101.0, hmm_result=_hmm("reversal", warnings=warnings))
        assert 0.0 <= r.reversal_score <= 1.0

    def test_unknown_direction_no_crash(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "diagonal", 100.0, 101.0)
        assert r.details["direction"] == "diagonal"

    def test_negative_entry_no_crash(self):
        eng = _make_engine()
        r = eng.compute("BTC", "trend", "long", 0.0, 101.0, base_tp_price=110.0)
        # entry<=0 时动态止盈保持原样，pnl 归零
        assert r.adaptive_tp_price == pytest.approx(110.0)
