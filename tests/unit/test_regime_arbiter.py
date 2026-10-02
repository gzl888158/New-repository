"""
RegimeArbiter 单元测试 — 两套 regime 引擎输出融合-统一仲裁
================================================================
覆盖：共识加分、加权仲裁、reversal 强制接管、fail-closed 回退、
枚举映射、stop_loss_manager 集成、orchestrator _perceive 注入、冲突日志。
"""
import asyncio
import os
import tempfile
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from services.regime_arbiter import RegimeArbiter, _DETECTOR_TO_MAIN
from services.market_regime_engine import MarketRegime


# ── 工具 ───────────────────────────────────────────────

def _main_out(regime="trend_bullish", confidence=0.7, strength=0.6):
    return {
        "regime": regime,
        "normalized_regime": regime,
        "state": regime,
        "subtype": "moderate",
        "strength": strength,
        "confidence": confidence,
        "factor_scores": {"trend": 0.5},
        "factor_weights": {"trend": 0.3},
        "last_update": "2026-09-30T00:00:00",
    }


def _det_out(regime="trending_up", reversal_prob=0.1, confidence=0.6):
    probs = {"reversal": reversal_prob, regime: confidence}
    return {
        "symbol": "ETH-USDT-SWAP",
        "regime": regime,
        "probabilities": probs,
        "features": {"adx": 25.0},
        "early_warnings": [],
        "timeframe": "4H",
        "detected_at": "2026-09-30T00:00:00",
    }


def _main_engine(out=None):
    """模拟主引擎：get_regime() 返回固定 out。"""
    e = MagicMock()
    main_out = out or _main_out()
    e.get_regime = MagicMock(return_value=main_out)
    e.get_symbol_regime = MagicMock(return_value={**main_out, "symbol_specific": True})
    return e


def _detector(out=None, exc=None):
    """模拟检测器：get_regime(symbol) 返回固定 out 或抛 exc。"""
    d = MagicMock()
    if exc:
        d.get_regime = MagicMock(side_effect=exc)
    else:
        d.get_regime = MagicMock(return_value=out or _det_out())
    return d


def _arbiter(main_out=None, det_out=None, det_exc=None, **cfg):
    return RegimeArbiter(
        main_engine=_main_engine(main_out),
        detector=_detector(det_out, det_exc),
        config=cfg,
    )


# ── 仲裁算法测试 ───────────────────────────────────────

class TestArbitrateConsensus:
    def test_symbol_arbitration_uses_same_symbol_main_regime(self):
        main = _main_engine(_main_out("trend_bullish", confidence=0.9))
        main.get_symbol_regime.return_value = {
            **_main_out("range_bound", confidence=0.9),
            "symbol_specific": True,
        }
        arb = RegimeArbiter(main, _detector(_det_out("trending_up", confidence=0.6)))

        result = arb.arbitrate("ETH-USDT-SWAP")

        main.get_symbol_regime.assert_called_once_with("ETH-USDT-SWAP", include_fusion=False)
        assert result["regime"] == "range_bound"

    def test_symbol_arbitration_skips_detector_without_symbol_level_main_data(self):
        main = _main_engine(_main_out("trend_bullish", confidence=0.9))
        main.get_symbol_regime.return_value = {"regime": "trend_bullish", "symbol_specific": False}
        detector = _detector(_det_out("ranging", confidence=0.9))
        arb = RegimeArbiter(main, detector)

        result = arb.arbitrate("ETH-USDT-SWAP")

        assert result["arbiter_strategy"] == "main_fallback"
        detector.get_regime.assert_not_called()

    def test_consensus_confidence_is_weighted_not_synthetically_increased(self):
        """一致时置信度按主/检测器来源权重融合，不添加固定加分。"""
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.7),
            det_out=_det_out("trending_up", confidence=0.6),
        )
        r = arb.arbitrate("ETH-USDT-SWAP")
        assert r["regime"] == "trend_bullish"
        assert r["arbiter_strategy"] == "consensus"
        assert r["arbiter_conflict"] is False
        assert r["confidence"] == pytest.approx(0.66)  # 0.7*0.6 + 0.6*0.4

    def test_consensus_confidence_remains_weighted_even_near_one(self):
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.95),
            det_out=_det_out("trending_up", confidence=0.6),
        )
        r = arb.arbitrate("BTC-USDT-SWAP")
        assert r["confidence"] == pytest.approx(0.81)


class TestArbitrateWeighted:
    def test_weighted_disagree_general(self):
        """冲突 + 非 reversal → 加权仲裁，conflict=True"""
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.3),  # 低置信
            det_out=_det_out("ranging", confidence=0.9),  # 高置信 → 胜
        )
        r = arb.arbitrate("ETH-USDT-SWAP")
        assert r["arbiter_conflict"] is True
        assert r["arbiter_strategy"] == "weighted"
        assert r["regime"] == "range_bound"  # 检测器胜

    def test_weighted_main_wins_when_higher(self):
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.9),
            det_out=_det_out("ranging", confidence=0.2),
        )
        r = arb.arbitrate("ETH-USDT-SWAP")
        assert r["regime"] == "trend_bullish"  # 主引擎胜

    def test_unstable_detector_discount_is_used_for_weighting(self):
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.5),
            det_out={
                **_det_out("ranging", confidence=0.9),
                "confidence": 0.3,
                "stable": False,
            },
        )

        result = arb.arbitrate("ETH-USDT-SWAP")

        assert result["regime"] == "trend_bullish"


class TestArbitrateReversalOverride:
    def test_detector_reversal_override(self):
        """REVERSAL 概率超阈值 → 强制 REVERSAL"""
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.7),
            det_out=_det_out("reversal", reversal_prob=0.6, confidence=0.5),
            reversal_threshold=0.45,
        )
        r = arb.arbitrate("ETH-USDT-SWAP")
        assert r["regime"] == "reversal"
        assert r["arbiter_strategy"] == "detector_reversal_override"
        assert r["arbiter_conflict"] is True

    def test_reversal_below_threshold_no_override(self):
        """REVERSAL 概率低于阈值 → 走加权"""
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.9),
            det_out=_det_out("reversal", reversal_prob=0.2, confidence=0.3),
            reversal_threshold=0.45,
        )
        r = arb.arbitrate("ETH-USDT-SWAP")
        # 主引擎置信 0.9*0.6=0.54 > 检测器 0.3*0.4=0.12 → 主引擎胜
        assert r["arbiter_strategy"] == "weighted"
        assert r["regime"] == "trend_bullish"


class TestRegimeIntegration:
    def test_reversal_gate_requires_probability_and_signal_confirmation(self):
        from services.regime_gate import RegimeGate

        engine = MagicMock()
        engine.get_regime.return_value = {
            "regime": "reversal",
            "probabilities": {"reversal": 0.8},
            "reversal_direction": "long",
        }
        gate = RegimeGate(engine)

        weak_signal = gate.evaluate("BTC-USDT-SWAP", "trend", "open", direction="long", confidence=0.69)
        strong_signal = gate.evaluate("BTC-USDT-SWAP", "trend", "open", direction="long", confidence=0.70)
        wrong_direction = gate.evaluate("BTC-USDT-SWAP", "trend", "open", direction="short", confidence=0.9)
        mean_reversion = gate.evaluate("BTC-USDT-SWAP", "grid", "open", direction="long", confidence=0.9)
        engine.get_regime.return_value = {
            "regime": "reversal",
            "probabilities": {"reversal": 0.64},
            "reversal_direction": "long",
        }
        weak_regime = gate.evaluate("BTC-USDT-SWAP", "grid", "open", direction="long", confidence=0.9)
        close_signal = gate.evaluate("BTC-USDT-SWAP", "trend", "close", direction="long", confidence=0.0)

        assert weak_signal.allowed is False
        assert strong_signal.allowed is True
        assert wrong_direction.allowed is False
        assert mean_reversion.allowed is False
        assert weak_regime.allowed is False
        assert close_signal.allowed is True

    def test_detector_reversal_direction_survives_arbiter_into_gate(self):
        from services.regime_gate import RegimeGate

        detector_result = {
            **_det_out("reversal", reversal_prob=0.8, confidence=0.8),
            "reversal_direction": "long",
        }
        arbiter = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.5),
            det_out=detector_result,
            reversal_threshold=0.45,
        )
        gate = RegimeGate(_main_engine(_main_out("trend_bullish")), regime_arbiter=arbiter)

        aligned = gate.evaluate("ETH-USDT-SWAP", "trend", "open", direction="long", confidence=0.8)
        opposed = gate.evaluate("ETH-USDT-SWAP", "trend", "open", direction="short", confidence=0.8)

        assert aligned.allowed is True
        assert opposed.allowed is False

    def test_stale_regime_rejects_open_but_allows_close(self):
        from services.regime_gate import RegimeGate

        engine = MagicMock()
        engine.get_regime.return_value = {
            "regime": "trend_bullish",
            "confidence": 0.8,
            "data_stale": True,
        }
        gate = RegimeGate(engine)

        open_result = gate.evaluate("BTC-USDT-SWAP", "trend", "open", direction="long")
        close_result = gate.evaluate("BTC-USDT-SWAP", "trend", "close", direction="long")

        assert open_result.allowed is False
        assert open_result.action == "reject"
        assert "stale" in open_result.reason
        assert close_result.allowed is True

    def test_engine_and_arbiter_do_not_recurse_when_fused(self):
        from services.market_regime_engine import MarketRegimeEngine

        engine = MarketRegimeEngine({"market_regime": {}}, okx_client=None)
        engine._symbol_regimes["ETH-USDT-SWAP"] = {
            "regime": "trend_bullish",
            "subtype": "moderate",
            "strength": 0.7,
            "confidence": 0.8,
            "factor_scores": {},
        }
        arbiter = RegimeArbiter(engine, _detector(_det_out("trending_up", confidence=0.6)))
        engine.set_regime_arbiter(arbiter)

        result = engine.get_symbol_regime("ETH-USDT-SWAP")

        assert result["regime"] == "trend_bullish"
        assert result["symbol_specific"] is True

    def test_main_engine_prefers_detector_breakout_state(self):
        """主引擎在检测器给出 breakout/reversal 时应优先使用细粒度状态。"""
        from services.market_regime_engine import MarketRegimeEngine
        from services.regime_gate import RegimeGate

        main = MarketRegimeEngine({"market_regime": {}}, okx_client=None)
        main._current_regime = MarketRegime.TREND_BULLISH
        main._current_subtype = "moderate"
        main._current_strength = 0.8
        main._regime_confidence = 0.7

        detector = MagicMock()
        detector.get_regime.return_value = {
            "symbol": "ETH-USDT-SWAP",
            "regime": "breakout",
            "probabilities": {"breakout": 0.83, "trend_up": 0.12},
            "timeframe": "medium",
        }
        main.set_detector(detector)

        result = main.get_regime()
        assert result["regime"] == "breakout"
        assert result["normalized_regime"] == "breakout"

        gate = RegimeGate(main, config={})
        gate_result = gate.evaluate("ETH-USDT-SWAP", "trend", "long", confidence=0.82)
        assert gate_result.allowed is True
        assert gate_result.regime == "breakout"

    def test_main_engine_prefers_detector_reversal_state(self):
        """反转状态应被主引擎与门控消费，避免被硬塞回 trend/range。"""
        from services.market_regime_engine import MarketRegimeEngine
        from services.regime_gate import RegimeGate

        main = MarketRegimeEngine({"market_regime": {}}, okx_client=None)
        main._current_regime = MarketRegime.TREND_BULLISH
        main._current_subtype = "moderate"
        main._current_strength = 0.7
        main._regime_confidence = 0.8

        detector = MagicMock()
        detector.get_regime.return_value = {
            "symbol": "BTC-USDT-SWAP",
            "regime": "reversal",
            "probabilities": {"reversal": 0.76, "ranging": 0.14},
            "reversal_direction": "long",
            "timeframe": "medium",
        }
        main.set_detector(detector)

        result = main.get_regime()
        assert result["regime"] == "reversal"

        gate = RegimeGate(main, config={})
        gate_result = gate.evaluate("BTC-USDT-SWAP", "trend", "open", direction="long", confidence=0.7)
        assert gate_result.allowed is True
        assert gate_result.regime == "reversal"


# ── Fail-closed 测试 ────────────────────────────────────

class TestArbitrateFailClosed:
    def test_fail_closed_when_detector_missing(self):
        """检测器异常 → 回退主引擎 strategy=main_fallback"""
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.7),
            det_exc=RuntimeError("detector boom"),
        )
        r = arb.arbitrate("ETH-USDT-SWAP")
        assert r["arbiter_strategy"] == "main_fallback"
        assert r["arbiter_conflict"] is False
        assert r["regime"] == "trend_bullish"
        assert r.get("arbiter_error") is not None

    def test_unknown_detector_regime_fallback(self):
        """检测器返回 unknown → 不映射，main_fallback"""
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.7),
            det_out=_det_out("unknown", confidence=0.5),
        )
        r = arb.arbitrate("ETH-USDT-SWAP")
        assert r["arbiter_strategy"] == "main_fallback"
        assert r["regime"] == "trend_bullish"

    def test_global_detector_snapshot_resolves_symbol_regime(self):
        """整体市场融合只消费主引擎基准币状态，不借用其他币种。"""
        detector_snapshot = {
            "timestamp": "2026-10-01T00:00:00",
            "symbols": {
                "BTC-USDT-SWAP": {
                    "symbol": "BTC-USDT-SWAP",
                    "regime": "breakout",
                    "probabilities": {"breakout": 0.82, "trending_up": 0.1},
                    "timeframe": "medium",
                    "detected_at": "2026-10-01T00:00:00",
                },
                "ETH-USDT-SWAP": {
                    "symbol": "ETH-USDT-SWAP",
                    "regime": "breakout",
                    "probabilities": {"breakout": 0.99},
                    "timeframe": "medium",
                }
            },
            "count": 2,
        }
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.30),
            det_out=detector_snapshot,
        )
        r = arb.arbitrate()
        assert r["arbiter_strategy"] == "weighted"
        assert r["regime"] == "breakout"
        assert r["detector_regime"] == "breakout"

    def test_global_detector_snapshot_without_base_symbol_falls_back_to_main(self):
        detector_snapshot = {
            "symbols": {
                "ETH-USDT-SWAP": {
                    "symbol": "ETH-USDT-SWAP",
                    "regime": "breakout",
                    "probabilities": {"breakout": 0.99},
                }
            },
            "count": 1,
        }
        arb = _arbiter(
            main_out=_main_out("trend_bullish", confidence=0.7),
            det_out=detector_snapshot,
        )

        result = arb.arbitrate()

        assert result["arbiter_strategy"] == "main_fallback"
        assert result["regime"] == "trend_bullish"


# ── 枚举映射测试 ───────────────────────────────────────

class TestEnumMapping:
    def test_enum_mapping_complete(self):
        """7 个检测器 regime 全部归一到主引擎 enum"""
        cases = [
            ("trending_up", "trend_bullish"),
            ("trending_down", "trend_bearish"),
            ("ranging", "range_bound"),
            ("high_volatility", "extreme_volatility"),
            ("low_volatility", "range_bound"),
            ("breakout", "breakout"),
            ("reversal", "reversal"),
        ]
        for det_reg, expected_main in cases:
            assert _DETECTOR_TO_MAIN[det_reg] == expected_main, f"mapping failed: {det_reg}"


# ── stop_loss_manager 集成测试 ─────────────────────────

class TestStopLossIntegration:
    def test_stop_loss_uses_arbiter_when_set(self):
        """arbiter 存在时 _fetch_reversal_inputs 走 arbiter"""
        from core.stop_loss_manager import StopLossManager
        cfg = {"trading": {"max_stop_loss_pct": 0.05}}
        # 构造函数: (config, okx_client, redis_cache, trade_journal, order_executor)
        mgr = StopLossManager(cfg, MagicMock(), MagicMock(), MagicMock(), MagicMock())
        # 注入 arbiter
        arb = MagicMock()
        arb.arbitrate = MagicMock(return_value={
            "regime": "reversal",
            "arbiter_strategy": "detector_reversal_override",
            "detector_raw": {"symbol": "ETH", "regime": "reversal", "probabilities": {"reversal": 0.7}},
            "detector_regime": "reversal",
            "detector_reversal_prob": 0.7,
            "early_warnings": [],
        })
        mgr.set_regime_arbiter(arb)
        # 调 _fetch_reversal_inputs
        hmm, ohlcv = asyncio.get_event_loop().run_until_complete(
            mgr._fetch_reversal_inputs("ETH-USDT-SWAP", None, ["dummy"])
        )
        assert hmm is not None
        assert hmm.get("regime") == "reversal"
        arb.arbitrate.assert_called_once_with("ETH-USDT-SWAP")

    def test_stop_loss_backward_compat_no_arbiter(self):
        """arbiter=None 时走旧检测器路径"""
        from core.stop_loss_manager import StopLossManager
        cfg = {"trading": {"max_stop_loss_pct": 0.05}}
        mgr = StopLossManager(cfg, MagicMock(), MagicMock(), MagicMock(), MagicMock())
        det = MagicMock()
        det.get_regime = MagicMock(return_value={"symbol": "ETH", "regime": "reversal"})
        mgr.set_market_regime_detector(det)
        # _regime_arbiter 未注入 → None
        hmm, ohlcv = asyncio.get_event_loop().run_until_complete(
            mgr._fetch_reversal_inputs("ETH-USDT-SWAP", None, ["dummy"])
        )
        assert hmm is not None
        det.get_regime.assert_called_once()


# ── orchestrator _perceive 注入测试 ────────────────────

class TestOrchestratorPerceive:
    def test_orchestrator_perceive_uses_arbiter(self):
        """arbiter 存在时 perception["market_regime"] 来自 arbiter"""
        from core.quant_agi_orchestrator import QuantAGIOrchestrator
        arb = MagicMock()
        arb.arbitrate = MagicMock(return_value={
            "regime": "reversal", "confidence": 0.8, "arbiter_strategy": "detector_reversal_override"
        })
        orch = QuantAGIOrchestrator(config={}, regime_arbiter=arb)
        # 调 _perceive（async 方法）
        perception = asyncio.get_event_loop().run_until_complete(orch._perceive())
        assert perception["market_regime"]["regime"] == "reversal"
        arb.arbitrate.assert_called_once()

    def test_orchestrator_perceive_fallback_no_arbiter(self):
        """arbiter=None 时回退 regime_engine"""
        from core.quant_agi_orchestrator import QuantAGIOrchestrator
        eng = MagicMock()
        eng.get_regime = MagicMock(return_value={"regime": "trend_bullish", "confidence": 0.7})
        # MagicMock 会自动创建 arbitrate 属性，需显式置 None 强制走 get_regime
        eng.arbitrate = None
        orch = QuantAGIOrchestrator(config={}, regime_engine=eng, regime_arbiter=None)
        perception = asyncio.get_event_loop().run_until_complete(orch._perceive())
        assert perception["market_regime"]["regime"] == "trend_bullish"
        eng.get_regime.assert_called_once()


# ── 冲突日志与历史测试 ─────────────────────────────────

class TestConflictLog:
    def test_conflict_logged_and_history_persisted(self):
        """冲突时记录 history + 写 conflicts.jsonl"""
        with tempfile.TemporaryDirectory() as td:
            log_path = os.path.join(td, "conflicts.jsonl")
            arb = _arbiter(
                main_out=_main_out("trend_bullish", confidence=0.3),
                det_out=_det_out("ranging", confidence=0.9),
                conflict_log=log_path,
            )
            r = arb.arbitrate("ETH-USDT-SWAP")
            assert r["arbiter_conflict"] is True
            # history 记录
            hist = arb.get_arbiter_history()
            assert len(hist) == 1
            assert hist[0]["conflict"] is True
            # conflicts.jsonl 写入
            assert os.path.exists(log_path)
            # 冲突统计
            stats = arb.get_conflict_stats()
            assert stats["total_arbitrations"] == 1
            assert stats["conflicts"] == 1
            assert stats["conflict_rate"] == 1.0
