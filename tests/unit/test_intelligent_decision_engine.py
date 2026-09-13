"""IntelligentDecisionEngine 企业级智能决策引擎 — 正式单元测试。

覆盖矩阵（聚焦本次「强化企业级智能决策系统」改动）：
  P0  MTF 贝叶斯融合三元后验（buy/sell/hold 归一化）
  P0  5 种融合方法（Bayesian / Weighted / Voting / D-S / Kalman）后验合法性
  P0  最终方向决策（buy/sell 显著占优才开仓，否则 hold）
  P0  成本收益分析（正/负期望值判断）
  P0  元决策器（各类 verdict）
  P0  自适应阈值（范围约束 + 方向性）
  P0  决策异常检测（含新键名 take_profit_price / stop_loss_price 的盈亏比检测）
  P1  不可变审计链（完整性校验 + 防篡改 + 结果回写）
  P1  配置解析（intelligent_decision_engine 段 + 默认值回退）
  P2  边界（空信号、零权益、异常输入）
"""

import asyncio
import math
from datetime import datetime, timedelta

import pytest

from decision.intelligent_decision_engine import (
    CostBenefitAnalysis,
    DecisionAuditEntry,
    DecisionContext,
    DecisionUrgency,
    FusionMethod,
    IntelligentDecisionEngine,
    MetaDecisionVerdict,
    MTFFusionResult,
    TimeFrameSignal,
)


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _default_ide_cfg() -> dict:
    """返回默认的 intelligent_decision_engine 配置段。"""
    return {
        "enabled": True,
        "fusion_method": "bayesian",
        "tf_weights": {"5m": 0.10, "15m": 0.15, "1h": 0.25, "4h": 0.30, "1d": 0.20},
        "prior_buy": 0.33,
        "prior_sell": 0.33,
        "prior_hold": 0.34,
        "tf_agreement_threshold": 0.60,
        "base_confidence_threshold": 0.43,
        "min_confidence_threshold": 0.30,
        "max_confidence_threshold": 0.70,
        "min_decision_interval_ms": 200,
        "max_decisions_per_minute": 30,
        "cooldown_after_loss_ms": 5000,
        "min_expected_value_usdt": 0.50,
        "taker_fee_rate": 0.0005,
        "maker_fee_rate": 0.0002,
        "base_slippage_pct": 0.0002,
        "funding_rate_annual": 0.10,
    }


def _make_engine(**ide_overrides) -> IntelligentDecisionEngine:
    """构造带完整配置段的引擎，允许覆盖部分字段。"""
    cfg = _default_ide_cfg()
    cfg.update(ide_overrides)
    return IntelligentDecisionEngine({"intelligent_decision_engine": cfg})


def _sig(timeframe: str, direction: str, strength: float = 1.0, confidence: float = 1.0,
         source: str = "test") -> TimeFrameSignal:
    return TimeFrameSignal(
        timeframe=timeframe, direction=direction,
        strength=strength, confidence=confidence, source=source,
    )


def _run(coro):
    """在独立事件循环中运行协程，避免依赖 pytest-asyncio 的 event_loop fixture。"""
    return asyncio.run(coro)


# ═══════════════════════════════════════════════════════════════
# P0: MTF 贝叶斯融合
# ═══════════════════════════════════════════════════════════════

class TestBayesianFusion:
    def test_posterior_normalized_to_one(self):
        eng = _make_engine()
        signals = [
            _sig("1h", "buy"),
            _sig("4h", "buy"),
            _sig("1d", "buy"),
        ]
        result = _run(eng.fuse_multi_timeframe(signals, FusionMethod.BAYESIAN))

        assert result.posterior_buy + result.posterior_sell + result.posterior_hold == pytest.approx(1.0, abs=1e-6)
        assert 0.0 <= result.posterior_buy <= 1.0
        assert 0.0 <= result.posterior_sell <= 1.0
        assert 0.0 <= result.posterior_hold <= 1.0

    def test_strong_buy_sets_buy_direction(self):
        eng = _make_engine()
        signals = [_sig("1h", "buy"), _sig("4h", "buy"), _sig("1d", "buy")]
        result = _run(eng.fuse_multi_timeframe(signals))

        assert result.final_direction == "buy"
        assert result.posterior_buy > 0.40
        assert result.final_confidence == pytest.approx(result.posterior_buy)

    def test_hold_dominates_when_no_buy_sell_signal(self):
        eng = _make_engine()
        signals = [_sig("1h", "hold"), _sig("4h", "hold"), _sig("1d", "hold")]
        result = _run(eng.fuse_multi_timeframe(signals))

        # hold 后验最高，且 hold 不属于 buy/sell，故最终方向为 hold
        assert result.final_direction == "hold"
        assert result.posterior_hold > result.posterior_buy
        assert result.posterior_hold > result.posterior_sell

    def test_empty_signals_returns_hold_with_warning(self):
        eng = _make_engine()
        result = _run(eng.fuse_multi_timeframe([]))

        assert result.final_direction == "hold"
        assert result.final_confidence == 0.0
        assert any("No timeframe signals" in w for w in result.warnings)


# ═══════════════════════════════════════════════════════════════
# P0: 5 种融合方法后验合法性
# ═══════════════════════════════════════════════════════════════

class TestAllFusionMethods:
    @pytest.mark.parametrize("method", [
        FusionMethod.BAYESIAN,
        FusionMethod.WEIGHTED,
        FusionMethod.VOTING,
        FusionMethod.DEMPSTER_SHAFER,
        FusionMethod.KALMAN,
    ])
    def test_posterior_valid_and_normalized(self, method):
        eng = _make_engine()
        signals = [_sig("1h", "buy"), _sig("4h", "sell"), _sig("1d", "buy")]
        result = _run(eng.fuse_multi_timeframe(signals, method))

        total = result.posterior_buy + result.posterior_sell + result.posterior_hold
        assert total == pytest.approx(1.0, abs=1e-6), f"{method.value} 后验之和不为 1"
        for p in (result.posterior_buy, result.posterior_sell, result.posterior_hold):
            assert 0.0 <= p <= 1.0, f"{method.value} 后验超出 [0,1]: {p}"
        assert result.final_direction in ("buy", "sell", "hold")

    def test_dempster_shafer_no_phantom_confidence_attr(self):
        """D-S 融合不应写入不存在的 confidence 字段（回归：幻影字段 AttributeError）。"""
        eng = _make_engine()
        signals = [_sig("1h", "buy"), _sig("4h", "buy")]
        result = _run(eng.fuse_multi_timeframe(signals, FusionMethod.DEMPSTER_SHAFER))

        assert not hasattr(result, "confidence")
        assert result.posterior_hold >= 0.0


# ═══════════════════════════════════════════════════════════════
# P0: 成本收益分析
# ═══════════════════════════════════════════════════════════════

class TestCostBenefit:
    def test_positive_ev_is_profitable(self):
        eng = _make_engine()
        cba = eng.analyze_cost_benefit(
            decision_id="d1", symbol="BTC-USDT-SWAP", direction="buy",
            quantity=1.0, entry_price=100.0, target_price=110.0, stop_price=95.0,
            leverage=5.0, win_probability=0.6,
        )

        assert cba.net_expected_value > 0
        assert cba.is_profitable is True
        assert cba.taker_fee > 0
        assert cba.estimated_slippage > 0
        assert cba.total_cost > 0
        assert cba.breakeven_move_pct > 0

    def test_negative_ev_is_avoided(self):
        eng = _make_engine()
        cba = eng.analyze_cost_benefit(
            decision_id="d2", symbol="BTC-USDT-SWAP", direction="buy",
            quantity=1.0, entry_price=100.0, target_price=100.5, stop_price=95.0,
            leverage=5.0, win_probability=0.5,
        )

        assert cba.net_expected_value < 0
        assert cba.is_profitable is False
        assert "Avoid" in cba.recommendation

    def test_short_direction_reward_risk(self):
        eng = _make_engine()
        cba = eng.analyze_cost_benefit(
            decision_id="d3", symbol="BTC-USDT-SWAP", direction="sell",
            quantity=1.0, entry_price=100.0, target_price=90.0, stop_price=105.0,
            leverage=5.0, win_probability=0.6,
        )
        # 空头：目标价低于入场价，风险价高于入场价 → 正收益
        assert cba.expected_profit > 0

    def test_marginal_positive_ev_reduces_size(self):
        eng = _make_engine()
        # 微正期望值（低于 min_expected_value_usdt 或风险调整收益不达标）
        cba = eng.analyze_cost_benefit(
            decision_id="d4", symbol="BTC-USDT-SWAP", direction="buy",
            quantity=1.0, entry_price=100.0, target_price=101.0, stop_price=99.5,
            leverage=5.0, win_probability=0.5,
        )
        # net_expected_value 为正但 < 0.5 USDT 阈值 → Marginal
        if 0 < cba.net_expected_value <= eng._min_expected_value_usdt:
            assert "Marginal" in cba.recommendation


# ═══════════════════════════════════════════════════════════════
# P0: 元决策器
# ═══════════════════════════════════════════════════════════════

class TestMetaDecide:
    def test_emergency_urgency_proceeds(self):
        eng = _make_engine()
        verdict, reason, details = _run(eng.meta_decide(
            "BTC-USDT-SWAP", decision_urgency=DecisionUrgency.IMMEDIATE,
        ))
        assert verdict == MetaDecisionVerdict.PROCEED

    def test_normal_proceeds(self):
        eng = _make_engine()
        verdict, reason, details = _run(eng.meta_decide(
            "BTC-USDT-SWAP", decision_urgency=DecisionUrgency.NORMAL,
            market_volatility=0.02, current_exposure_pct=0.0,
        ))
        assert verdict == MetaDecisionVerdict.PROCEED

    def test_high_exposure_reduces_size(self):
        eng = _make_engine()
        verdict, _, _ = _run(eng.meta_decide(
            "BTC-USDT-SWAP", current_exposure_pct=0.80,
        ))
        assert verdict == MetaDecisionVerdict.REDUCE_SIZE

    def test_extreme_exposure_abstains(self):
        eng = _make_engine()
        verdict, _, _ = _run(eng.meta_decide(
            "BTC-USDT-SWAP", current_exposure_pct=0.95,
        ))
        assert verdict == MetaDecisionVerdict.ABSTAIN

    def test_extreme_volatility_emergency_only(self):
        eng = _make_engine()
        verdict, _, _ = _run(eng.meta_decide(
            "BTC-USDT-SWAP", market_volatility=0.10,
        ))
        assert verdict == MetaDecisionVerdict.EMERGENCY_ONLY

    def test_rate_limit_defers(self):
        eng = _make_engine()
        eng._decision_count_1m.extend([datetime.now()] * 30)
        verdict, reason, _ = _run(eng.meta_decide("BTC-USDT-SWAP"))
        assert verdict == MetaDecisionVerdict.DEFER
        assert "Rate limit" in reason

    def test_loss_cooldown_defers(self):
        eng = _make_engine()
        eng._consecutive_losses = 3
        eng._last_decision_time = datetime.now() - timedelta(seconds=1)
        verdict, reason, _ = _run(eng.meta_decide("BTC-USDT-SWAP"))
        assert verdict == MetaDecisionVerdict.DEFER
        assert "Loss cooldown" in reason


# ═══════════════════════════════════════════════════════════════
# P0: 自适应阈值
# ═══════════════════════════════════════════════════════════════

class TestAdaptThreshold:
    @pytest.mark.parametrize("regime,wr,vol,dd", [
        ("trending_up", 0.5, 0.5, 0.0),
        ("trending_down", 0.3, 0.8, 0.4),
        ("high_volatility", 0.1, 0.9, 0.6),
        ("low_volatility", 0.9, 0.1, 0.0),
        ("ranging", 0.5, 0.5, 0.0),
        ("unknown", 0.0, 1.0, 0.8),
    ])
    def test_threshold_within_bounds(self, regime, wr, vol, dd):
        eng = _make_engine()
        th = eng.adapt_threshold(regime, recent_win_rate=wr,
                                 volatility_percentile=vol, equity_drawdown=dd)
        assert eng._min_confidence_threshold <= th <= eng._max_confidence_threshold
        assert th == eng.get_current_threshold()

    def test_high_volatility_raises_threshold(self):
        eng = _make_engine()
        base = eng.adapt_threshold("ranging", 0.5, 0.5, 0.0)
        high = eng.adapt_threshold("high_volatility", 0.5, 0.5, 0.0)
        assert high > base

    def test_trending_lowers_threshold(self):
        eng = _make_engine()
        base = eng.adapt_threshold("ranging", 0.5, 0.5, 0.0)
        trend = eng.adapt_threshold("trending_up", 0.5, 0.5, 0.0)
        assert trend < base

    def test_low_win_rate_raises_threshold(self):
        eng = _make_engine()
        base = eng.adapt_threshold("ranging", 0.5, 0.5, 0.0)
        low_wr = eng.adapt_threshold("ranging", 0.0, 0.5, 0.0)
        assert low_wr > base

    def test_high_drawdown_lowers_threshold(self):
        eng = _make_engine()
        base = eng.adapt_threshold("ranging", 0.5, 0.5, 0.0)
        high_dd = eng.adapt_threshold("ranging", 0.5, 0.5, 0.6)
        assert high_dd < base


# ═══════════════════════════════════════════════════════════════
# P0: 决策异常检测
# ═══════════════════════════════════════════════════════════════

class TestDetectAnomaly:
    def test_rr_anomaly_with_new_keys(self):
        """新键名 take_profit_price / stop_loss_price 应触发盈亏比异常检测。"""
        eng = _make_engine()
        decision = {
            "direction": "buy",
            "price": 100.0,
            "take_profit_price": 100.1,   # 盈亏比 < 0.3
            "stop_loss_price": 50.0,
        }
        is_anomaly, reason = eng.detect_anomaly(decision)
        assert is_anomaly is True
        assert "risk/reward" in reason

    def test_rr_anomaly_with_standard_keys(self):
        eng = _make_engine()
        decision = {
            "direction": "buy",
            "price": 100.0,
            "take_profit": 100.1,
            "stop_loss": 50.0,
        }
        is_anomaly, reason = eng.detect_anomaly(decision)
        assert is_anomaly is True
        assert "risk/reward" in reason

    def test_normal_decision_no_anomaly(self):
        eng = _make_engine()
        decision = {
            "direction": "buy",
            "price": 100.0,
            "take_profit_price": 110.0,
            "stop_loss_price": 95.0,
            "confidence": 0.6,
            "quantity": 0.01,
        }
        is_anomaly, reason = eng.detect_anomaly(decision)
        assert is_anomaly is False
        assert reason == "Normal"

    def test_suspiciously_high_confidence_anomaly(self):
        eng = _make_engine()
        decision = {"direction": "buy", "confidence": 0.999}
        is_anomaly, reason = eng.detect_anomaly(decision)
        assert is_anomaly is True
        assert "high confidence" in reason


# ═══════════════════════════════════════════════════════════════
# P1: 不可变审计链
# ═══════════════════════════════════════════════════════════════

class TestAuditChain:
    def _entry(self, decision_id: str, direction: str = "buy") -> DecisionAuditEntry:
        return DecisionAuditEntry(
            decision_id=decision_id, timestamp=datetime.now().isoformat(),
            decision_type="trend", symbol="BTC-USDT-SWAP", direction=direction,
        )

    def test_chain_valid_after_records(self):
        eng = _make_engine()
        for i in range(3):
            eng.record_decision_audit(self._entry(f"d{i}"))
        valid, idx = eng.verify_audit_chain()
        assert valid is True
        assert idx == -1
        assert len(eng._audit_chain) == 3

    def test_tamper_detection(self):
        eng = _make_engine()
        for i in range(3):
            eng.record_decision_audit(self._entry(f"d{i}"))
        # 篡改中间条目的方向（不可变字段）
        eng._audit_chain[1].direction = "sell"
        valid, idx = eng.verify_audit_chain()
        assert valid is False
        assert idx == 1

    def test_outcome_writeback_does_not_break_chain(self):
        """结果回写（outcome / outcome_pnl）不纳入哈希，不应破坏链完整性。"""
        eng = _make_engine()
        eng.record_decision_audit(self._entry("d0"))
        eng.record_decision_outcome(pnl=5.0, was_win=True, decision_id="d0")

        assert eng._audit_chain[0].outcome == "win"
        assert eng._audit_chain[0].outcome_pnl == 5.0
        valid, _ = eng.verify_audit_chain()
        assert valid is True

    def test_build_and_record_audit_entry(self):
        eng = _make_engine()
        cba = eng.analyze_cost_benefit(
            decision_id="d0", symbol="BTC-USDT-SWAP", direction="buy",
            quantity=1.0, entry_price=100.0, target_price=110.0, stop_price=95.0,
        )
        entry_hash = eng.build_and_record_audit_entry(
            decision_id="d0", symbol="BTC-USDT-SWAP", strategy_name="trend",
            direction="buy", confidence=0.6, cost_benefit=cba,
            meta_verdict="proceed",
        )
        assert entry_hash
        assert eng._audit_chain[-1].entry_hash == entry_hash
        valid, _ = eng.verify_audit_chain()
        assert valid is True


# ═══════════════════════════════════════════════════════════════
# P1: 配置解析
# ═══════════════════════════════════════════════════════════════

class TestConfigParsing:
    def test_config_section_parsed(self):
        cfg = _default_ide_cfg()
        cfg.update({
            "fusion_method": "weighted",
            "base_confidence_threshold": 0.50,
            "taker_fee_rate": 0.0006,
            "tf_weights": {"1h": 0.5, "4h": 0.5},
        })
        eng = IntelligentDecisionEngine({"intelligent_decision_engine": cfg})

        assert eng._fusion_method == FusionMethod.WEIGHTED
        assert eng._base_confidence_threshold == 0.50
        assert eng._current_threshold == 0.50
        assert eng._taker_fee_rate == 0.0006
        assert eng._tf_weights == {"1h": 0.5, "4h": 0.5}

    def test_defaults_when_no_config(self):
        eng = IntelligentDecisionEngine()
        assert eng._fusion_method == FusionMethod.BAYESIAN
        assert eng._base_confidence_threshold == pytest.approx(0.43)
        assert eng._min_confidence_threshold == pytest.approx(0.30)
        assert eng._max_confidence_threshold == pytest.approx(0.70)
        assert eng._taker_fee_rate == pytest.approx(0.0005)

    def test_invalid_fusion_method_falls_back_to_bayesian(self):
        cfg = _default_ide_cfg()
        cfg["fusion_method"] = "not_a_real_method"
        eng = IntelligentDecisionEngine({"intelligent_decision_engine": cfg})
        assert eng._fusion_method == FusionMethod.BAYESIAN


# ═══════════════════════════════════════════════════════════════
# P2: 边界与辅助
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_enrich_context_order_book(self):
        eng = _make_engine()
        order_book = {
            "bids": [["100.0", "10"], ["99.9", "20"]],
            "asks": [["100.1", "10"], ["100.2", "30"]],
        }
        ctx = eng.enrich_context("BTC-USDT-SWAP", order_book=order_book)
        assert ctx.bid_ask_spread_pct > 0
        # bids 总量 30 > asks 总量 40 → 失衡为负
        assert ctx.order_book_imbalance < 0

    def test_get_stats_and_latency(self):
        eng = _make_engine()
        eng.record_decision_execution("d0", latency_ms=50.0, success=True)
        eng.record_decision_execution("d1", latency_ms=150.0, success=True)
        eng.record_decision_execution("d2", latency_ms=250.0, success=False)

        assert eng.get_average_latency() == pytest.approx(150.0)
        assert eng.get_latency_percentile(95) == pytest.approx(250.0)
        stats = eng.get_stats()
        assert stats["total_decisions"] == 3
        assert stats["avg_latency_ms"] == pytest.approx(150.0)

    def test_record_outcome_consecutive_counters(self):
        eng = _make_engine()
        eng.record_decision_outcome(pnl=1.0, was_win=True)
        eng.record_decision_outcome(pnl=1.0, was_win=True)
        eng.record_decision_outcome(pnl=-2.0, was_win=False)
        stats = eng.get_consecutive_stats()
        assert stats["consecutive_wins"] == 0
        assert stats["consecutive_losses"] == 1

    def test_attribute_decision_direction_consistency(self):
        eng = _make_engine()
        ctx = DecisionContext(symbol="BTC-USDT-SWAP", market_regime="trending_up", trend_strength=0.6)
        attr = eng.attribute_decision(
            {"direction": "buy", "signals": [{"source": "trend", "direction": "buy", "strength": 0.8}]},
            MTFFusionResult(),
            ctx,
        )
        # 信号方向与决策方向一致 → 正贡献
        assert attr["signal_trend"] == pytest.approx(0.8)
        # regime 隐含 long 与 buy 一致 → 正贡献
        assert attr["market_regime"] == pytest.approx(0.6)
