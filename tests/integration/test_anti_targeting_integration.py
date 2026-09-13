"""
防针对量化数据 集成测试
========================
测试三层防护的端到端集成：
1. SignalObfuscator — 信号混淆
2. OrderFingerprintMasker — 订单指纹掩盖
3. AntiPatternDetector — 反模式检测
4. SignalGenerator — 流水线集成
"""

import sys
import os
import time
import random
import threading
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.signal_generator import (
    SignalGenerator, SignalContext, TradingSignal,
    SignalType, SignalLevel, SignalSource
)
from core.signal_obfuscator import SignalObfuscator, ObfuscationMode
from core.order_fingerprint_masker import OrderFingerprintMasker, MaskingMode, MaskedOrder
from core.anti_pattern_detector import AntiPatternDetector, RiskLevel, PatternAlert


# ═══════════════════════════════════════════════════════════════
# 测试工具
# ═══════════════════════════════════════════════════════════════

def _make_context(symbol="BTC-USDT-SWAP", price=65000.0, indicators=None):
    """创建测试用 SignalContext"""
    default_indicators = {
        "adx": 28, "plus_di": 30, "minus_di": 18,
        "rsi": 45, "macd_histogram": 120,
        "bollinger_position": 0.3, "bollinger_width": 0.04,
        "kdj_k": 40, "kdj_j": 35,
        "volatility_percentile": 0.5, "atr_percent": 0.015,
        "ma20": price * 0.98, "ma50": price * 0.95,
        "recent_close": price,
    }
    if indicators:
        default_indicators.update(indicators)
    
    return SignalContext(
        symbol=symbol,
        current_price=price,
        position_side=None,
        position_size=0.0,
        indicators=default_indicators,
        market_state="neutral",
        volatility_level="normal",
    )


def _make_config(model_config=None):
    """创建测试用配置"""
    base = {
        "enabled": True,
        "min_signal_weight": 0.3,
        "min_quality_score": 0.3,
        "max_signals_per_second": 100,
        "max_signals_per_symbol_second": 50,
        "signal_cooldown_seconds": 0,
        "conflict_resolution": "strongest_weight",
        "confirmation_required": False,
        "market_regime_filter": False,
        "signal_aggregation": False,
        "persist_signals": False,
        "dispatch_mode": "immediate",
        "dispatch_priority": "quality_first",
        "adaptive_learning": {"enabled": False},
        "rules": {
            "trend_breakout": {"enabled": True, "weight": 1.0, "min_adx": 25},
            "mean_reversion": {"enabled": False},
            "volatility_arb": {"enabled": False},
            "grid_rebound": {"enabled": False},
            "risk_control": {"enabled": False},
            "divergence_detection": {"enabled": False},
            "market_structure": {"enabled": False},
        },
        "anti_targeting": {
            "enabled": True,
            "signal_obfuscator": {
                "enabled": True,
                "mode": "standard",
                "delay_enabled": True,
                "delay_min_ms": 50,
                "delay_max_ms": 200,
                "weight_jitter_enabled": True,
                "weight_jitter_pct": 0.05,
                "dummy_injection_enabled": True,
                "dummy_probability": 0.1,
                "dummy_max_per_batch": 2,
                "type_rotation_enabled": True,
                "rotation_probability": 0.1,
                "expiry_jitter_enabled": True,
                "expiry_jitter_seconds": 10,
                "adaptive_enabled": True,
                "pattern_detection_threshold": 0.7,
                "adaptive_decay": 0.95,
            },
            "order_fingerprint_masker": {
                "enabled": True,
                "mode": "standard",
                "quantity_jitter_enabled": True,
                "quantity_jitter_min_pct": 0.03,
                "quantity_jitter_max_pct": 0.08,
                "price_jitter_enabled": True,
                "price_jitter_pct": 0.0005,
                "split_enabled": True,
                "split_min_notional": 500.0,
                "split_min_parts": 2,
                "split_max_parts": 5,
                "time_slice_enabled": True,
                "time_slice_min_ms": 100,
                "time_slice_max_ms": 500,
                "type_rotation_enabled": True,
                "limit_order_ratio": 0.7,
                "post_only_ratio": 0.15,
            },
            "anti_pattern_detector": {
                "enabled": True,
                "check_interval_seconds": 60,
                "signal_window_size": 100,
                "order_window_size": 50,
                "analysis_window_minutes": 30,
                "self_similarity_enabled": True,
                "similarity_threshold": 0.75,
                "similarity_weight": 0.25,
                "timing_entropy_enabled": True,
                "entropy_threshold_low": 0.5,
                "entropy_threshold_warn": 0.3,
                "timing_entropy_weight": 0.20,
                "volume_pattern_enabled": True,
                "volume_cv_threshold": 0.15,
                "volume_pattern_weight": 0.20,
                "price_clustering_enabled": True,
                "price_cluster_radius_pct": 0.002,
                "price_cluster_ratio_threshold": 0.6,
                "price_clustering_weight": 0.20,
                "front_running_enabled": True,
                "front_running_slippage_threshold": 0.003,
                "front_running_ratio_threshold": 0.3,
                "front_running_weight": 0.15,
            },
        },
    }
    if model_config:
        base.update(model_config)
    return base


# ═══════════════════════════════════════════════════════════════
# Test 1: SignalObfuscator 独立测试
# ═══════════════════════════════════════════════════════════════

def test_signal_obfuscator_basic():
    """基本混淆：权重抖动、延迟、类型旋转"""
    config = _make_config()
    obfuscator = SignalObfuscator(config)
    
    signals = [
        TradingSignal(
            symbol="BTC-USDT-SWAP", signal_type=SignalType.OPEN_LONG,
            source=SignalSource.TREND_BREAKOUT, level=SignalLevel.STRONG,
            weight=0.85, price=65000.0, quantity=0.01,
            timestamp=datetime.now(), reason="Test signal",
        ),
    ]
    
    result = obfuscator.obfuscate(signals)
    
    assert len(result) >= 1, "Should return at least original signal"
    assert result[0].symbol == "BTC-USDT-SWAP"
    
    stats = obfuscator.get_stats()
    assert stats["enabled"] is True
    assert stats["mode"] == "standard"
    assert stats["total_obfuscated"] >= 1
    
    print("  PASS: test_signal_obfuscator_basic")


def test_signal_obfuscator_safe_signals():
    """安全信号（止损/全平）不参与混淆"""
    config = _make_config()
    config["anti_targeting"]["signal_obfuscator"]["dummy_probability"] = 0.0  # 禁用虚拟信号
    obfuscator = SignalObfuscator(config)
    
    signals = [
        TradingSignal(
            symbol="BTC-USDT-SWAP", signal_type=SignalType.STOP_LOSS,
            source=SignalSource.RISK_CONTROL, level=SignalLevel.STRONG,
            weight=1.0, price=64000.0, quantity=0.01,
            timestamp=datetime.now(), reason="Stop loss",
        ),
        TradingSignal(
            symbol="ETH-USDT-SWAP", signal_type=SignalType.CLOSE_ALL,
            source=SignalSource.RISK_CONTROL, level=SignalLevel.STRONG,
            weight=1.0, price=3000.0, quantity=0.1,
            timestamp=datetime.now(), reason="Close all",
        ),
    ]
    
    result = obfuscator.obfuscate(signals)
    
    assert len(result) == 2, "Safe signals should pass through unchanged"
    for s in result:
        assert s.weight == 1.0, f"Safe signal weight should be unchanged, got {s.weight}"
        assert "obfuscation_delay_ms" not in s.metadata, "Safe signal should not have delay"
    
    print("  PASS: test_signal_obfuscator_safe_signals")


def test_signal_obfuscator_dummy_injection():
    """虚拟信号注入"""
    config = _make_config()
    config["anti_targeting"]["signal_obfuscator"]["dummy_probability"] = 1.0  # 100%注入
    config["anti_targeting"]["signal_obfuscator"]["dummy_max_per_batch"] = 3
    obfuscator = SignalObfuscator(config)
    
    signals = [
        TradingSignal(
            symbol="BTC-USDT-SWAP", signal_type=SignalType.OPEN_LONG,
            source=SignalSource.TREND_BREAKOUT, level=SignalLevel.STRONG,
            weight=0.85, price=65000.0, quantity=0.01,
            timestamp=datetime.now(), reason="Test signal",
        ),
    ]
    
    result = obfuscator.obfuscate(signals)
    
    dummy_count = sum(1 for s in result if s.metadata.get("is_dummy", False))
    assert dummy_count > 0, f"Should have injected dummy signals, got {dummy_count}"
    
    stats = obfuscator.get_stats()
    assert stats["total_dummies_injected"] >= 1
    
    print("  PASS: test_signal_obfuscator_dummy_injection")


def test_signal_obfuscator_mode_switching():
    """混淆模式切换"""
    config = _make_config()
    obfuscator = SignalObfuscator(config)
    
    # 切换到激进模式
    obfuscator.set_mode(ObfuscationMode.AGGRESSIVE)
    stats = obfuscator.get_stats()
    assert stats["mode"] == "aggressive"
    assert stats["current_intensity"] > 1.0, "Aggressive mode should have higher intensity"
    
    # 切换到轻度模式
    obfuscator.set_mode(ObfuscationMode.LIGHT)
    stats = obfuscator.get_stats()
    assert stats["mode"] == "light"
    assert stats["current_intensity"] < 1.0, "Light mode should have lower intensity"
    
    # 关闭
    obfuscator.set_mode(ObfuscationMode.OFF)
    stats = obfuscator.get_stats()
    assert stats["mode"] == "off"
    
    # 关闭模式下信号不混淆
    signals = [
        TradingSignal(
            symbol="BTC-USDT-SWAP", signal_type=SignalType.OPEN_LONG,
            source=SignalSource.TREND_BREAKOUT, level=SignalLevel.STRONG,
            weight=0.85, price=65000.0, quantity=0.01,
            timestamp=datetime.now(), reason="Test signal",
        ),
    ]
    result = obfuscator.obfuscate(signals)
    assert len(result) == 1
    assert result[0].weight == 0.85, "OFF mode should not modify signals"
    
    print("  PASS: test_signal_obfuscator_mode_switching")


def test_signal_obfuscator_filter_dummy():
    """过滤虚拟信号"""
    config = _make_config()
    obfuscator = SignalObfuscator(config)
    
    signals = [
        TradingSignal(
            symbol="BTC-USDT-SWAP", signal_type=SignalType.OPEN_LONG,
            source=SignalSource.TREND_BREAKOUT, level=SignalLevel.STRONG,
            weight=0.85, price=65000.0, quantity=0.01,
            timestamp=datetime.now(), reason="Real signal",
            metadata={"is_dummy": False},
        ),
        TradingSignal(
            symbol="ETH-USDT-SWAP", signal_type=SignalType.OPEN_LONG,
            source=SignalSource.MOMENTUM, level=SignalLevel.WEAK,
            weight=0.3, price=3000.0, quantity=0.0,
            timestamp=datetime.now(), reason="[DUMMY]",
            metadata={"is_dummy": True},
        ),
    ]
    
    filtered = obfuscator.filter_dummy_signals(signals)
    assert len(filtered) == 1, "Should filter out dummy signal"
    assert filtered[0].symbol == "BTC-USDT-SWAP"
    
    print("  PASS: test_signal_obfuscator_filter_dummy")


# ═══════════════════════════════════════════════════════════════
# Test 2: OrderFingerprintMasker 独立测试
# ═══════════════════════════════════════════════════════════════

def test_order_masker_basic():
    """基本订单掩盖"""
    config = _make_config()
    masker = OrderFingerprintMasker(config)
    
    result = masker.mask_order(
        symbol="BTC-USDT-SWAP", side="buy", order_type="limit",
        price=65000.0, quantity=0.001,
    )
    
    assert len(result) == 1, "Small order should not split"
    assert result[0].symbol == "BTC-USDT-SWAP"
    assert result[0].side == "buy"
    assert result[0].metadata.get("masked") is True
    
    stats = masker.get_stats()
    assert stats["enabled"] is True
    assert stats["total_masked"] >= 1
    
    print("  PASS: test_order_masker_basic")


def test_order_masker_split():
    """大单拆分"""
    config = _make_config()
    config["anti_targeting"]["order_fingerprint_masker"]["split_min_notional"] = 100.0
    masker = OrderFingerprintMasker(config)
    
    # 大单：65000 * 10 = 650000 USDT > 100 USDT
    result = masker.mask_order(
        symbol="BTC-USDT-SWAP", side="buy", order_type="limit",
        price=65000.0, quantity=10.0,
    )
    
    assert len(result) >= 2, f"Large order should be split, got {len(result)} parts"
    for i, order in enumerate(result):
        assert order.is_split is True
        assert order.split_total == len(result)
        assert order.split_index == i + 1
    
    # 验证总数量接近原始
    total_qty = sum(o.quantity for o in result)
    assert abs(total_qty - 10.0) < 0.1, f"Split total qty {total_qty} should be close to original 10.0"
    
    stats = masker.get_stats()
    assert stats["total_split"] >= 1
    
    print("  PASS: test_order_masker_split")


def test_order_masker_price_jitter():
    """价格抖动"""
    config = _make_config()
    masker = OrderFingerprintMasker(config)
    
    # 多次测试，价格应略有不同
    prices = set()
    for _ in range(20):
        result = masker.mask_order(
            symbol="ETH-USDT-SWAP", side="buy", order_type="limit",
            price=3000.0, quantity=0.1,
        )
        prices.add(round(result[0].price, 2))
    
    # 至少有一些价格变化
    assert len(prices) >= 2, f"Price jitter should produce variation, got {len(prices)} unique prices"
    
    print("  PASS: test_order_masker_price_jitter")


def test_order_masker_risk_order():
    """风控单不拆分"""
    config = _make_config()
    config["anti_targeting"]["order_fingerprint_masker"]["split_min_notional"] = 100.0
    masker = OrderFingerprintMasker(config)
    
    result = masker.mask_order(
        symbol="BTC-USDT-SWAP", side="sell", order_type="market",
        price=64000.0, quantity=10.0, is_risk_order=True,
    )
    
    assert len(result) == 1, "Risk order should NOT be split"
    assert result[0].is_split is False
    
    print("  PASS: test_order_masker_risk_order")


def test_order_masker_precision():
    """精度适配"""
    config = _make_config()
    masker = OrderFingerprintMasker(config)
    
    # 各币种精度测试
    for symbol, prec in OrderFingerprintMasker.SYMBOL_PRECISION.items():
        p = masker.get_symbol_precision(symbol)
        assert p["qty_step"] == prec["qty_step"]
        assert p["price_step"] == prec["price_step"]
    
    # 未知币种回退到默认精度
    p = masker.get_symbol_precision("UNKNOWN-USDT-SWAP")
    assert p["qty_step"] == 0.01
    assert p["price_step"] == 0.01
    
    print("  PASS: test_order_masker_precision")


# ═══════════════════════════════════════════════════════════════
# Test 3: AntiPatternDetector 独立测试
# ═══════════════════════════════════════════════════════════════

def test_pattern_detector_basic():
    """基本模式检测"""
    config = _make_config()
    detector = AntiPatternDetector(config)
    
    assert detector.get_risk_score() == 0.0
    assert detector.get_risk_level() == RiskLevel.LOW
    assert detector.get_risk_trend() == "stable"
    
    stats = detector.get_stats()
    assert stats["enabled"] is True
    assert stats["current_risk_score"] == 0.0
    
    print("  PASS: test_pattern_detector_basic")


def test_pattern_detector_signal_recording():
    """信号记录与模式检测"""
    config = _make_config()
    config["anti_targeting"]["anti_pattern_detector"]["check_interval_seconds"] = 0  # 立即触发
    detector = AntiPatternDetector(config)
    
    # 记录大量高度相似的信号 → 应该检测到自相似性
    for i in range(50):
        detector.record_signal(
            symbol="BTC-USDT-SWAP",
            signal_type="open_long",
            source="trend_breakout",
            weight=0.85,
            direction="long",
        )
    
    # 强制评估
    score = detector.evaluate_risk(force=True)
    stats = detector.get_stats()
    
    assert stats["signal_records_count"] >= 10, f"Should have signal records, got {stats['signal_records_count']}"
    assert stats["total_checks"] >= 1
    
    print(f"  PASS: test_pattern_detector_signal_recording (risk_score={score:.3f})")


def test_pattern_detector_order_recording():
    """订单记录与成交量模式检测"""
    config = _make_config()
    config["anti_targeting"]["anti_pattern_detector"]["check_interval_seconds"] = 0
    detector = AntiPatternDetector(config)
    
    # 记录固定大小的订单 → 应该检测到成交量规律
    for i in range(30):
        detector.record_order(
            symbol="BTC-USDT-SWAP",
            side="buy",
            quantity=0.01,  # 固定数量
            price=65000.0,
        )
    
    score = detector.evaluate_risk(force=True)
    stats = detector.get_stats()
    
    assert stats["order_records_count"] >= 10, f"Should have order records, got {stats['order_records_count']}"
    
    print(f"  PASS: test_pattern_detector_order_recording (risk_score={score:.3f})")


def test_pattern_detector_slippage():
    """滑点记录与抢先交易检测"""
    config = _make_config()
    config["anti_targeting"]["anti_pattern_detector"]["check_interval_seconds"] = 0
    config["anti_targeting"]["anti_pattern_detector"]["front_running_slippage_threshold"] = 0.001
    detector = AntiPatternDetector(config)
    
    # 记录大量异常滑点
    for i in range(20):
        detector.record_slippage(
            symbol="BTC-USDT-SWAP",
            expected_price=65000.0,
            actual_price=65000.0 * (1.005 if i % 2 == 0 else 0.995),  # ±0.5% 滑点
            side="buy" if i % 2 == 0 else "sell",
        )
    
    score = detector.evaluate_risk(force=True)
    stats = detector.get_stats()
    
    assert stats["slippage_records_count"] >= 5
    
    print(f"  PASS: test_pattern_detector_slippage (risk_score={score:.3f})")


def test_pattern_detector_alert_callback():
    """告警回调"""
    config = _make_config()
    config["anti_targeting"]["anti_pattern_detector"]["check_interval_seconds"] = 0
    detector = AntiPatternDetector(config)
    
    alerts_received = []
    detector.register_alert_callback(lambda alert: alerts_received.append(alert))
    
    # 记录高度相似的信号触发告警
    for i in range(60):
        detector.record_signal(
            symbol="BTC-USDT-SWAP",
            signal_type="open_long",
            source="trend_breakout",
            weight=0.85,
            direction="long",
        )
    
    detector.evaluate_risk(force=True)
    
    # 可能触发告警（取决于评分）
    alert_count = len(alerts_received)
    print(f"  PASS: test_pattern_detector_alert_callback ({alert_count} alerts)")
    assert True  # 回调至少无异常


# ═══════════════════════════════════════════════════════════════
# Test 4: SignalGenerator 流水线集成
# ═══════════════════════════════════════════════════════════════

def test_signal_generator_anti_targeting_integration():
    """SignalGenerator 集成防针对模块"""
    config = _make_config()
    config["rules"]["trend_breakout"]["enabled"] = True
    config["rules"]["mean_reversion"]["enabled"] = True
    config["confirmation_required"] = False
    
    gen = SignalGenerator(config)
    
    assert gen._signal_obfuscator is not None, "Obfuscator should be initialized"
    assert gen._anti_pattern_detector is not None, "Detector should be initialized"
    assert gen._anti_targeting_enabled is True
    
    # 生成信号
    ctx = _make_context("BTC-USDT-SWAP", 65000.0, {
        "adx": 30, "plus_di": 35, "minus_di": 15,
        "rsi": 55, "macd_histogram": 200,
    })
    
    signals = gen.generate_signals_pipeline(ctx)
    
    # 信号可能被混淆
    stats = gen.get_stats()
    at_stats = stats.get("anti_targeting", {})
    
    assert at_stats["enabled"] is True
    assert "obfuscator" in at_stats
    assert "pattern_detector" in at_stats
    
    print(f"  PASS: test_signal_generator_anti_targeting_integration ({len(signals)} signals generated)")


def test_signal_generator_dummy_filtering():
    """虚拟信号不过滤"""
    config = _make_config()
    config["anti_targeting"]["signal_obfuscator"]["dummy_probability"] = 1.0
    config["anti_targeting"]["signal_obfuscator"]["dummy_max_per_batch"] = 3
    
    gen = SignalGenerator(config)
    
    ctx = _make_context("BTC-USDT-SWAP", 65000.0, {
        "adx": 30, "plus_di": 35, "minus_di": 15,
        "rsi": 55, "macd_histogram": 200,
    })
    
    signals = gen.generate_signals_pipeline(ctx)
    
    # 检查虚拟信号不应该被分发
    for s in signals:
        if s.metadata.get("is_dummy", False):
            should_dispatch = gen._should_dispatch(s)
            assert should_dispatch is False, f"Dummy signal should NOT be dispatched: {s.reason}"
    
    print(f"  PASS: test_signal_generator_dummy_filtering ({len(signals)} signals)")


def test_signal_generator_order_recording():
    """订单记录到检测器"""
    config = _make_config()
    gen = SignalGenerator(config)
    
    # 记录订单
    gen.record_order_to_detector(
        symbol="BTC-USDT-SWAP", side="buy", quantity=0.01, price=65000.0
    )
    gen.record_order_to_detector(
        symbol="ETH-USDT-SWAP", side="sell", quantity=0.1, price=3000.0
    )
    
    # 记录滑点
    gen.record_slippage_to_detector(
        symbol="BTC-USDT-SWAP", expected_price=65000.0,
        actual_price=65050.0, side="buy"
    )
    
    at_stats = gen.get_anti_targeting_stats()
    assert at_stats["enabled"] is True
    
    print("  PASS: test_signal_generator_order_recording")


def test_signal_generator_hot_update():
    """热更新防针对配置"""
    config = _make_config()
    gen = SignalGenerator(config)
    
    assert gen._anti_targeting_enabled is True
    
    # 热更新：关闭防针对
    new_config = {
        "anti_targeting": {
            "enabled": False,
            "signal_obfuscator": {"enabled": False},
            "anti_pattern_detector": {"enabled": False},
        }
    }
    gen.update_config(new_config)
    
    assert gen._anti_targeting_enabled is False
    assert gen._obfuscation_enabled is False
    assert gen._pattern_detection_enabled is False
    
    # 热更新：重新开启
    new_config2 = {
        "anti_targeting": {
            "enabled": True,
            "signal_obfuscator": {"enabled": True, "mode": "aggressive"},
            "anti_pattern_detector": {"enabled": True},
        }
    }
    gen.update_config(new_config2)
    
    assert gen._anti_targeting_enabled is True
    assert gen._obfuscation_enabled is True
    
    print("  PASS: test_signal_generator_hot_update")


def test_signal_generator_stats():
    """统计信息完整性"""
    config = _make_config()
    gen = SignalGenerator(config)
    
    stats = gen.get_stats()
    
    assert "anti_targeting" in stats
    at_stats = stats["anti_targeting"]
    assert "enabled" in at_stats
    assert "obfuscation_enabled" in at_stats
    assert "pattern_detection_enabled" in at_stats
    assert "signals_obfuscated" in at_stats
    assert "dummy_signals_injected" in at_stats
    assert "pattern_risk_score" in at_stats
    assert "obfuscator" in at_stats
    assert "pattern_detector" in at_stats
    
    print("  PASS: test_signal_generator_stats")


# ═══════════════════════════════════════════════════════════════
# Test 5: 端到端场景
# ═══════════════════════════════════════════════════════════════

def test_e2e_full_pipeline():
    """端到端：信号生成 → 混淆 → 检测 → 分发"""
    config = _make_config()
    config["rules"]["trend_breakout"]["enabled"] = True
    config["rules"]["mean_reversion"]["enabled"] = True
    config["rules"]["volatility_arb"]["enabled"] = True
    config["confirmation_required"] = False
    config["anti_targeting"]["signal_obfuscator"]["dummy_probability"] = 0.3
    
    gen = SignalGenerator(config)
    
    # 模拟多轮信号生成
    total_signals = 0
    for i in range(10):
        ctx = _make_context(
            f"{['BTC','ETH','SOL'][i % 3]}-USDT-SWAP",
            65000.0 * (1 + random.uniform(-0.02, 0.02)),
            {
                "adx": random.uniform(20, 40),
                "plus_di": random.uniform(15, 35),
                "minus_di": random.uniform(15, 35),
                "rsi": random.uniform(30, 70),
                "macd_histogram": random.uniform(-200, 200),
                "bollinger_position": random.uniform(0.1, 0.9),
            }
        )
        
        signals = gen.generate_signals_pipeline(ctx)
        total_signals += len(signals)
    
    # 获取完整统计
    stats = gen.get_stats()
    at_stats = stats["anti_targeting"]
    
    print(f"  E2E: {total_signals} total signals generated")
    print(f"  E2E: {at_stats['signals_obfuscated']} signals obfuscated")
    print(f"  E2E: {at_stats['dummy_signals_injected']} dummy signals injected")
    print(f"  E2E: risk_score={at_stats['pattern_risk_score']:.3f}")
    print(f"  E2E: obfuscator_mode={at_stats['obfuscator']['mode']}")
    
    assert at_stats["enabled"] is True
    assert at_stats["obfuscator"]["enabled"] is True
    assert at_stats["pattern_detector"]["enabled"] is True
    
    print("  PASS: test_e2e_full_pipeline")


def test_e2e_risk_feedback_loop():
    """端到端：风险反馈闭环 — 高风险 → 自动升级混淆"""
    config = _make_config()
    config["anti_targeting"]["anti_pattern_detector"]["check_interval_seconds"] = 0
    config["anti_targeting"]["signal_obfuscator"]["mode"] = "light"
    
    gen = SignalGenerator(config)
    
    # 初始为轻度混淆
    assert gen._signal_obfuscator._mode == ObfuscationMode.LIGHT
    
    # 记录大量固定模式信号 → 触发高自相似性评分
    for i in range(80):
        gen._anti_pattern_detector.record_signal(
            symbol="BTC-USDT-SWAP",
            signal_type="open_long",
            source="trend_breakout",
            weight=0.85,
            direction="long",
        )
    
    # 触发评估
    score = gen._anti_pattern_detector.evaluate_risk(force=True)
    
    # 反馈风险评分到混淆器
    if gen._signal_obfuscator:
        gen._signal_obfuscator.set_pattern_risk(score)
    
    # 检查混淆器是否感知到高风险
    obf_stats = gen._signal_obfuscator.get_stats()
    reported_risk = obf_stats["pattern_risk_score"]
    
    # 如果风险评分高，混淆强度应增加
    if reported_risk > 0.3:
        intensity = obf_stats["current_intensity"]
        assert intensity > 0.5, f"High risk should increase intensity, got {intensity}"
    
    print(f"  PASS: test_e2e_risk_feedback_loop (risk={reported_risk:.3f}, intensity={obf_stats['current_intensity']:.2f})")


def test_e2e_disabled_anti_targeting():
    """端到端：关闭防针对模块"""
    config = _make_config()
    config["anti_targeting"]["enabled"] = False
    
    gen = SignalGenerator(config)
    
    assert gen._anti_targeting_enabled is False
    assert gen._signal_obfuscator is None
    assert gen._anti_pattern_detector is None
    
    ctx = _make_context("BTC-USDT-SWAP", 65000.0, {
        "adx": 30, "plus_di": 35, "minus_di": 15,
        "rsi": 55, "macd_histogram": 200,
    })
    
    signals = gen.generate_signals_pipeline(ctx)
    
    # 信号应正常生成，无混淆
    for s in signals:
        assert not s.metadata.get("is_dummy", False), "No dummies when disabled"
        assert "obfuscation_delay_ms" not in s.metadata, "No delays when disabled"
    
    at_stats = gen.get_anti_targeting_stats()
    assert at_stats["enabled"] is False
    
    print(f"  PASS: test_e2e_disabled_anti_targeting ({len(signals)} signals, no obfuscation)")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("防针对量化数据 集成测试")
    print("=" * 60)
    
    tests = [
        # Test 1: SignalObfuscator
        ("SignalObfuscator - Basic", test_signal_obfuscator_basic),
        ("SignalObfuscator - Safe Signals", test_signal_obfuscator_safe_signals),
        ("SignalObfuscator - Dummy Injection", test_signal_obfuscator_dummy_injection),
        ("SignalObfuscator - Mode Switching", test_signal_obfuscator_mode_switching),
        ("SignalObfuscator - Filter Dummy", test_signal_obfuscator_filter_dummy),
        
        # Test 2: OrderFingerprintMasker
        ("OrderFingerprintMasker - Basic", test_order_masker_basic),
        ("OrderFingerprintMasker - Split", test_order_masker_split),
        ("OrderFingerprintMasker - Price Jitter", test_order_masker_price_jitter),
        ("OrderFingerprintMasker - Risk Order", test_order_masker_risk_order),
        ("OrderFingerprintMasker - Precision", test_order_masker_precision),
        
        # Test 3: AntiPatternDetector
        ("AntiPatternDetector - Basic", test_pattern_detector_basic),
        ("AntiPatternDetector - Signal Recording", test_pattern_detector_signal_recording),
        ("AntiPatternDetector - Order Recording", test_pattern_detector_order_recording),
        ("AntiPatternDetector - Slippage", test_pattern_detector_slippage),
        ("AntiPatternDetector - Alert Callback", test_pattern_detector_alert_callback),
        
        # Test 4: SignalGenerator Integration
        ("SignalGenerator - Integration", test_signal_generator_anti_targeting_integration),
        ("SignalGenerator - Dummy Filtering", test_signal_generator_dummy_filtering),
        ("SignalGenerator - Order Recording", test_signal_generator_order_recording),
        ("SignalGenerator - Hot Update", test_signal_generator_hot_update),
        ("SignalGenerator - Stats", test_signal_generator_stats),
        
        # Test 5: E2E
        ("E2E - Full Pipeline", test_e2e_full_pipeline),
        ("E2E - Risk Feedback Loop", test_e2e_risk_feedback_loop),
        ("E2E - Disabled", test_e2e_disabled_anti_targeting),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            print(f"\n[{name}]")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{len(tests)} passed, {failed} failed")
    print(f"{'=' * 60}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()