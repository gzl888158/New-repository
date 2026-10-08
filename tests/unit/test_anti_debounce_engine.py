"""
AntiDebounceEngine 单元测试
============================
覆盖：初始化、品种级冷却、信号去重、策略级冷却、全局冷却、亏损后冷却、
刷单检测、自适应冷却、管理接口、与 AntiDebounceFilter 集成、边缘场景
"""
import pytest
import time
import threading

from core.anti_debounce_engine import (
    AntiDebounceEngine, DebounceResult, DebounceLayer,
    DEFAULT_SYMBOL_COOLDOWN, DEFAULT_SIGNAL_DEDUP_WINDOW,
    DEFAULT_STRATEGY_COOLDOWN, DEFAULT_GLOBAL_COOLDOWN,
    DEFAULT_LOSS_COOLDOWN, DEFAULT_CHURN_WINDOW,
    DEFAULT_CHURN_MAX_FLIPS, DEFAULT_CHURN_BLOCK_SECONDS,
)
from core.signal_pre_filter import (
    AntiDebounceFilter, SignalContext, FilterDecision,
    SignalPreFilterChain, ACTION_DELAY, ACTION_PASS, ACTION_REJECT,
)


# ============================================================================
# 辅助函数
# ============================================================================

def _make_config(**overrides) -> dict:
    """构造 anti_debounce 配置段"""
    cfg = {
        "anti_debounce": {
            "symbol_cooldown_seconds": DEFAULT_SYMBOL_COOLDOWN,
            "signal_dedup_window_seconds": DEFAULT_SIGNAL_DEDUP_WINDOW,
            "strategy_cooldown_seconds": DEFAULT_STRATEGY_COOLDOWN,
            "global_cooldown_seconds": DEFAULT_GLOBAL_COOLDOWN,
            "loss_cooldown_seconds": DEFAULT_LOSS_COOLDOWN,
            "churn_window_seconds": DEFAULT_CHURN_WINDOW,
            "churn_max_flips": DEFAULT_CHURN_MAX_FLIPS,
            "churn_block_seconds": DEFAULT_CHURN_BLOCK_SECONDS,
            "enable_symbol": True,
            "enable_signal_dedup": True,
            "enable_strategy": True,
            "enable_global": True,
            "enable_loss_cooldown": True,
            "enable_churn": True,
            "enable_adaptive": True,
        }
    }
    cfg["anti_debounce"].update(overrides)
    return cfg


def _make_engine(**overrides) -> AntiDebounceEngine:
    return AntiDebounceEngine(_make_config(**overrides))


def _make_ctx(symbol="BTC-USDT-SWAP", strategy="grid", direction="long",
              signal_type="open", is_close=False, confidence=0.6) -> SignalContext:
    return SignalContext(
        symbol=symbol, strategy_name=strategy, direction=direction,
        signal_type=signal_type, is_close=is_close, confidence=confidence,
    )


def _advance_time(engine: AntiDebounceEngine, seconds: float):
    """模拟时间推进：通过直接修改内部状态的时间戳来模拟时间流逝。"""
    offset = seconds
    with engine._lock:
        for sym_dir in engine._symbol_records.values():
            for rec in sym_dir.values():
                rec.timestamp -= offset
        for rec in engine._signal_dedup.values():
            rec.timestamp -= offset
        for ts_list in engine._strategy_timestamps.values():
            for i in range(len(ts_list)):
                ts_list[i] -= offset
        for i in range(len(engine._global_timestamps)):
            engine._global_timestamps[i] -= offset
        for k in list(engine._loss_timestamps.keys()):
            engine._loss_timestamps[k] -= offset
        for churn_list in engine._churn_history.values():
            for i in range(len(churn_list)):
                ts, d = churn_list[i]
                churn_list[i] = (ts - offset, d)
        for k in list(engine._churn_blocked.keys()):
            engine._churn_blocked[k] -= offset


# 短冷却用于消除其他层干扰，只测试目标层
_TINY = 0.0  # 零冷却值，彻底禁用非目标层拦截


# ============================================================================
# Test: 初始化
# ============================================================================

class TestInitialization:
    """测试引擎初始化与配置读取"""

    def test_default_config(self):
        engine = AntiDebounceEngine()
        assert engine._symbol_cooldown == DEFAULT_SYMBOL_COOLDOWN
        assert engine._signal_dedup_window == DEFAULT_SIGNAL_DEDUP_WINDOW
        assert engine._strategy_cooldown == DEFAULT_STRATEGY_COOLDOWN
        assert engine._global_cooldown == DEFAULT_GLOBAL_COOLDOWN
        assert engine._loss_cooldown == DEFAULT_LOSS_COOLDOWN
        assert engine._enable_symbol is True
        assert engine._enable_adaptive is True

    def test_custom_config(self):
        cfg = _make_config(
            symbol_cooldown_seconds=60,
            strategy_cooldown_seconds=10,
            enable_signal_dedup=False,
        )
        engine = AntiDebounceEngine(cfg)
        assert engine._symbol_cooldown == 60
        assert engine._strategy_cooldown == 10
        assert engine._enable_signal_dedup is False

    def test_all_enabled_by_default(self):
        engine = AntiDebounceEngine()
        assert engine._enable_symbol
        assert engine._enable_signal_dedup
        assert engine._enable_strategy
        assert engine._enable_global
        assert engine._enable_loss_cooldown
        assert engine._enable_churn
        assert engine._enable_adaptive

    def test_all_disabled(self):
        cfg = _make_config(
            enable_symbol=False, enable_signal_dedup=False,
            enable_strategy=False, enable_global=False,
            enable_loss_cooldown=False, enable_churn=False,
            enable_adaptive=False,
        )
        engine = AntiDebounceEngine(cfg)
        result = engine.check(symbol="BTC-USDT-SWAP", strategy_name="grid",
                              direction="long", signal_type="open")
        assert result.allowed is True

    def test_empty_config_no_anti_debounce_key(self):
        """配置中无 anti_debounce 段时使用默认值"""
        engine = AntiDebounceEngine({})
        assert engine._symbol_cooldown == DEFAULT_SYMBOL_COOLDOWN


# ============================================================================
# Test: 品种级冷却
# ============================================================================

class TestSymbolCoolown:
    """测试品种级冷却：同一品种+同一方向，冷却窗口内禁止重复开仓"""

    def test_first_trade_allowed(self):
        engine = _make_engine()
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert result.allowed is True

    def test_same_symbol_direction_blocked(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert result.allowed is False
        assert result.blocked_layer == DebounceLayer.SYMBOL.value
        assert "品种级冷却" in result.blocked_reason

    def test_different_direction_allowed(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="short")
        assert result.allowed is True

    def test_different_symbol_allowed(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="ETH-USDT-SWAP", direction="long")
        assert result.allowed is True

    def test_cooldown_expired(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        _advance_time(engine, 31)
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert result.allowed is True

    def test_remaining_cooldown_reported(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert result.allowed is False
        assert result.remaining_cooldown > 0
        assert result.remaining_cooldown <= 30

    def test_close_always_allowed(self):
        engine = _make_engine(symbol_cooldown_seconds=30)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long", is_close=True)
        assert result.allowed is True


# ============================================================================
# Test: 信号去重
# ============================================================================

class TestSignalDedup:
    """测试信号去重：同一品种+同一信号类型，去重窗口内只放行一次"""

    def test_same_signal_type_blocked(self):
        engine = _make_engine(signal_dedup_window_seconds=10,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", signal_type="trend_entry")
        result = engine.check(symbol="BTC-USDT-SWAP", signal_type="trend_entry")
        assert result.allowed is False
        assert result.blocked_layer == DebounceLayer.SIGNAL_DEDUP.value

    def test_different_signal_type_allowed(self):
        engine = _make_engine(signal_dedup_window_seconds=10,
                              symbol_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", signal_type="trend_entry")
        result = engine.check(symbol="BTC-USDT-SWAP", signal_type="grid_entry")
        assert result.allowed is True

    def test_different_symbol_same_signal_type_allowed(self):
        engine = _make_engine(signal_dedup_window_seconds=10,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", signal_type="trend_entry")
        result = engine.check(symbol="ETH-USDT-SWAP", signal_type="trend_entry")
        assert result.allowed is True

    def test_dedup_window_expired(self):
        engine = _make_engine(signal_dedup_window_seconds=10,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", signal_type="trend_entry")
        _advance_time(engine, 11)
        result = engine.check(symbol="BTC-USDT-SWAP", signal_type="trend_entry")
        assert result.allowed is True

    def test_empty_signal_type_skips_dedup(self):
        engine = _make_engine(signal_dedup_window_seconds=10,
                              symbol_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", signal_type="")
        result = engine.check(symbol="BTC-USDT-SWAP", signal_type="")
        assert result.allowed is True

    def test_dedup_disabled(self):
        engine = _make_engine(enable_signal_dedup=False,
                              symbol_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", signal_type="trend_entry")
        result = engine.check(symbol="BTC-USDT-SWAP", signal_type="trend_entry")
        # dedup 关闭，不会被该层拦截
        assert result.blocked_layer != DebounceLayer.SIGNAL_DEDUP.value


# ============================================================================
# Test: 策略级冷却
# ============================================================================

class TestStrategyCoolown:
    """测试策略级冷却：同一策略所有交易，冷却窗口内限制频率"""

    def test_same_strategy_blocked(self):
        engine = _make_engine(strategy_cooldown_seconds=5,
                              symbol_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", strategy_name="grid")
        result = engine.check(symbol="ETH-USDT-SWAP", strategy_name="grid")
        assert result.blocked_layer == DebounceLayer.STRATEGY.value

    def test_different_strategy_allowed(self):
        engine = _make_engine(strategy_cooldown_seconds=5,
                              symbol_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", strategy_name="grid")
        result = engine.check(symbol="BTC-USDT-SWAP", strategy_name="trend")
        assert result.allowed is True

    def test_strategy_cooldown_expired(self):
        engine = _make_engine(strategy_cooldown_seconds=5,
                              symbol_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", strategy_name="grid")
        _advance_time(engine, 6)
        result = engine.check(symbol="ETH-USDT-SWAP", strategy_name="grid")
        assert result.allowed is True

    def test_empty_strategy_name_skips(self):
        engine = _make_engine(strategy_cooldown_seconds=5,
                              symbol_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", strategy_name="")
        result = engine.check(symbol="BTC-USDT-SWAP", strategy_name="")
        assert result.allowed is True


# ============================================================================
# Test: 全局冷却
# ============================================================================

class TestGlobalCoolown:
    """测试全局冷却：所有交易，冷却窗口内限制频率"""

    def test_global_cooldown_blocked(self):
        engine = _make_engine(global_cooldown_seconds=2,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", strategy_name="grid")
        result = engine.check(symbol="ETH-USDT-SWAP", strategy_name="trend")
        assert result.blocked_layer == DebounceLayer.GLOBAL.value

    def test_global_cooldown_expired(self):
        engine = _make_engine(global_cooldown_seconds=2,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", strategy_name="grid")
        _advance_time(engine, 3)
        result = engine.check(symbol="ETH-USDT-SWAP", strategy_name="trend")
        assert result.allowed is True


# ============================================================================
# Test: 亏损后冷却
# ============================================================================

class TestLossCooldown:
    """测试亏损后冷却：该品种最近一笔亏损后，进入冷却期"""

    def test_loss_cooldown_blocked(self):
        engine = _make_engine(loss_cooldown_seconds=120,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=-5.0)
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=-5.0)
        assert result.blocked_layer == DebounceLayer.LOSS_COOLDOWN.value

    def test_no_loss_no_cooldown(self):
        engine = _make_engine(loss_cooldown_seconds=120,
                              symbol_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=5.0)
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=0.0)
        assert result.allowed is True

    def test_loss_cooldown_expired(self):
        engine = _make_engine(loss_cooldown_seconds=120,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=-5.0)
        _advance_time(engine, 121)
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=-5.0)
        assert result.allowed is True

    def test_different_symbol_not_affected_by_loss(self):
        engine = _make_engine(loss_cooldown_seconds=120,
                              global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=-5.0)
        result = engine.check(symbol="ETH-USDT-SWAP", direction="long", pnl_usdt=0.0)
        assert result.allowed is True


# ============================================================================
# Test: 刷单检测
# ============================================================================

class TestChurnDetection:
    """测试刷单检测：同一品种短时间内多方向频繁交易"""

    def test_below_threshold_allowed(self):
        engine = _make_engine(churn_window_seconds=60, churn_max_flips=3,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        # 记录 2 次翻转：long → short → long
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        engine.record(symbol="BTC-USDT-SWAP", direction="short")
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert result.allowed is True

    def test_above_threshold_blocked(self):
        engine = _make_engine(churn_window_seconds=60, churn_max_flips=3,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        # 记录 3 次翻转：long → short → long → short
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        engine.record(symbol="BTC-USDT-SWAP", direction="short")
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        engine.record(symbol="BTC-USDT-SWAP", direction="short")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert result.allowed is False
        assert result.blocked_layer == DebounceLayer.CHURN.value

    def test_churn_blocked_duration(self):
        engine = _make_engine(churn_window_seconds=60, churn_max_flips=3,
                              churn_block_seconds=300,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        for _ in range(4):
            engine.record(symbol="BTC-USDT-SWAP", direction="long")
            engine.record(symbol="BTC-USDT-SWAP", direction="short")
        engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert engine.is_churn_blocked("BTC-USDT-SWAP") is True

    def test_churn_block_expired(self):
        engine = _make_engine(churn_window_seconds=60, churn_max_flips=3,
                              churn_block_seconds=300,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        for _ in range(4):
            engine.record(symbol="BTC-USDT-SWAP", direction="long")
            engine.record(symbol="BTC-USDT-SWAP", direction="short")
        engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert engine.is_churn_blocked("BTC-USDT-SWAP") is True
        _advance_time(engine, 301)
        assert engine.is_churn_blocked("BTC-USDT-SWAP") is False

    def test_same_direction_no_flip(self):
        engine = _make_engine(churn_window_seconds=60, churn_max_flips=3,
                              symbol_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY,
                              global_cooldown_seconds=_TINY)
        for _ in range(5):
            engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert result.allowed is True


# ============================================================================
# Test: 自适应冷却
# ============================================================================

class TestAdaptiveCoolown:
    """测试自适应冷却：根据波动率、回撤、账户档位动态调整冷却时间"""

    def test_adaptive_multiplier_default(self):
        engine = _make_engine()
        mult = engine._compute_adaptive_multiplier()
        assert mult == 1.0

    def test_high_volatility_increases_multiplier(self):
        engine = _make_engine()
        engine.set_market_state(volatility=0.05, drawdown=0.0, account_tier="small")
        mult = engine._compute_adaptive_multiplier()
        assert mult > 1.0

    def test_high_drawdown_increases_multiplier(self):
        engine = _make_engine()
        engine.set_market_state(volatility=0.0, drawdown=0.15, account_tier="small")
        mult = engine._compute_adaptive_multiplier()
        assert mult > 1.0

    def test_nano_tier_increases_multiplier(self):
        engine = _make_engine()
        engine.set_market_state(volatility=0.0, drawdown=0.0, account_tier="nano")
        mult = engine._compute_adaptive_multiplier()
        assert mult > 1.0

    def test_all_factors_combined(self):
        engine = _make_engine()
        engine.set_market_state(volatility=0.05, drawdown=0.15, account_tier="nano")
        mult = engine._compute_adaptive_multiplier()
        # R115: ADAPTIVE_DRAWDOWN_MULTIPLIER 2.0→1.3, combined result ~2.5
        assert mult >= 2.5

    def test_adaptive_multiplier_extends_cooldown(self):
        engine = _make_engine(symbol_cooldown_seconds=10,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.set_market_state(volatility=0.05, drawdown=0.15, account_tier="nano")
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        _advance_time(engine, 10)
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        # 自适应倍率 > 1.0，所以 10s 后仍被拦截
        assert result.allowed is False
        assert result.adaptive_multiplier > 1.0

    def test_adaptive_disabled(self):
        engine = _make_engine(enable_adaptive=False)
        engine.set_market_state(volatility=0.05, drawdown=0.15, account_tier="nano")
        mult = engine._compute_adaptive_multiplier()
        assert mult == 1.0


# ============================================================================
# Test: 多层级联
# ============================================================================

class TestLayerCascade:
    """测试多层防抖动并行检查时的短路行为"""

    def test_symbol_hits_first(self):
        """品种级冷却最先命中（当所有冷却时间相同时）"""
        engine = _make_engine(
            symbol_cooldown_seconds=30,
            signal_dedup_window_seconds=30,
            strategy_cooldown_seconds=30,
            global_cooldown_seconds=30,
        )
        engine.record(symbol="BTC-USDT-SWAP", direction="long", strategy_name="grid",
                      signal_type="open")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long",
                              strategy_name="grid", signal_type="open")
        assert result.allowed is False
        assert result.blocked_layer == DebounceLayer.SYMBOL.value

    def test_breakdown_contains_all_layers(self):
        engine = _make_engine()
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long",
                              strategy_name="grid", signal_type="open")
        assert result.allowed is True
        assert "layers" in result.breakdown
        layers = result.breakdown["layers"]
        for layer in ["symbol", "signal_dedup", "strategy", "global", "churn"]:
            assert layer in layers


# ============================================================================
# Test: 管理接口
# ============================================================================

class TestManagementAPI:
    """测试 reset_symbol / reset_all / get_stats / get_symbol_status"""

    def test_reset_symbol_clears_all_state(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=-5.0)
        engine.record(symbol="BTC-USDT-SWAP", signal_type="grid_entry")
        engine.reset_symbol("BTC-USDT-SWAP")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long")
        assert result.allowed is True

    def test_reset_symbol_does_not_affect_other(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        engine.record(symbol="ETH-USDT-SWAP", direction="long")
        engine.reset_symbol("BTC-USDT-SWAP")
        # BTC 已重置
        assert engine.check(symbol="BTC-USDT-SWAP", direction="long").allowed is True
        # ETH 仍被拦截
        assert engine.check(symbol="ETH-USDT-SWAP", direction="long").allowed is False

    def test_reset_all(self):
        engine = _make_engine()
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        engine.record(symbol="ETH-USDT-SWAP", direction="long")
        engine.reset_all()
        assert engine.check(symbol="BTC-USDT-SWAP", direction="long").allowed is True
        assert engine.check(symbol="ETH-USDT-SWAP", direction="long").allowed is True

    def test_get_stats(self):
        engine = _make_engine()
        engine.record(symbol="BTC-USDT-SWAP", direction="long", strategy_name="grid")
        stats = engine.get_stats()
        assert stats["symbol_records"] == 1
        assert "grid" in stats["strategy_timestamps"]
        assert stats["global_timestamps"] == 1

    def test_get_symbol_status(self):
        engine = _make_engine()
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        status = engine.get_symbol_status("BTC-USDT-SWAP")
        assert "last_long_ts" in status
        assert "last_long_elapsed" in status


# ============================================================================
# Test: AntiDebounceFilter 集成
# ============================================================================

class TestAntiDebounceFilterIntegration:
    """测试 AntiDebounceFilter 与 SignalPreFilterChain 的集成"""

    def test_filter_blocks_when_engine_blocks(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        flt = AntiDebounceFilter(engine=engine)
        ctx = _make_ctx(symbol="BTC-USDT-SWAP", direction="long")

        decision = flt.check(ctx, {"signal_type": "open", "pnl_usdt": 0.0})
        assert decision is not None
        assert decision.action == ACTION_DELAY
        assert "品种级冷却" in decision.reason

    def test_filter_passes_when_engine_passes(self):
        engine = _make_engine()
        flt = AntiDebounceFilter(engine=engine)
        ctx = _make_ctx(symbol="BTC-USDT-SWAP", direction="long")

        decision = flt.check(ctx, {"signal_type": "open", "pnl_usdt": 0.0})
        assert decision is None

    def test_filter_passes_for_close(self):
        engine = _make_engine(symbol_cooldown_seconds=30)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        flt = AntiDebounceFilter(engine=engine)
        ctx = _make_ctx(symbol="BTC-USDT-SWAP", direction="long", is_close=True)

        decision = flt.check(ctx, {"signal_type": "close", "pnl_usdt": 0.0})
        assert decision is None

    def test_filter_passes_when_engine_none(self):
        flt = AntiDebounceFilter(engine=None)
        ctx = _make_ctx(symbol="BTC-USDT-SWAP", direction="long")
        decision = flt.check(ctx, {})
        assert decision is None

    def test_filter_uses_state_engine(self):
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        flt = AntiDebounceFilter(engine=None)
        ctx = _make_ctx(symbol="BTC-USDT-SWAP", direction="long")
        decision = flt.check(ctx, {"debounce_engine": engine, "signal_type": "open"})
        assert decision is not None
        assert decision.action == ACTION_DELAY

    def test_filter_chain_includes_anti_debounce(self):
        chain = SignalPreFilterChain()
        filter_names = [f.name for f in chain.filters]
        assert "anti_debounce" in filter_names

    def test_filter_chain_anti_debounce_priority(self):
        chain = SignalPreFilterChain()
        debounce_filter = next(f for f in chain._filters if f.name == "anti_debounce")
        confidence_filter = next(f for f in chain._filters if f.name == "confidence")
        frequency_filter = next(f for f in chain._filters if f.name == "frequency")
        # anti_debounce 在 confidence (2) 之后，frequency (6) 之前
        assert confidence_filter.priority < debounce_filter.priority < frequency_filter.priority


# ============================================================================
# Test: 边缘场景
# ============================================================================

class TestEdgeCases:
    """测试边缘场景"""

    def test_empty_symbol_handled(self):
        engine = _make_engine()
        result = engine.check(symbol="", direction="long")
        assert result.allowed is True

    def test_empty_direction_handled(self):
        engine = _make_engine(global_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="")
        assert result.allowed is True

    def test_rapid_same_symbol_different_directions(self):
        """快速切换方向：symbol 级只拦截同方向，方向切换应放行"""
        engine = _make_engine(symbol_cooldown_seconds=30,
                              global_cooldown_seconds=_TINY,
                              strategy_cooldown_seconds=_TINY)
        engine.record(symbol="BTC-USDT-SWAP", direction="long")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="short")
        assert result.allowed is True

    def test_concurrent_record_and_check_safe(self):
        """并发安全：record 和 check 在锁保护下不会冲突"""
        engine = _make_engine()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    engine.record(symbol="BTC-USDT-SWAP", direction="long",
                                  strategy_name="grid", signal_type="open")
                    engine.check(symbol="BTC-USDT-SWAP", direction="long",
                                 strategy_name="grid", signal_type="open")
                    engine.check(symbol="ETH-USDT-SWAP", direction="long",
                                 strategy_name="trend", signal_type="trend_entry")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_to_dict_serialization(self):
        result = DebounceResult(
            allowed=False,
            blocked_layer="symbol",
            blocked_reason="冷却中",
            remaining_cooldown=25.5,
            adaptive_multiplier=1.5,
            breakdown={"layers": {"symbol": {"blocked": True}}},
        )
        d = result.to_dict()
        assert d["allowed"] is False
        assert d["remaining_cooldown"] == 25.5
        assert d["adaptive_multiplier"] == 1.5
        assert "breakdown" in d

    def test_zero_cooldown_config(self):
        """冷却时间为 0 时不应该拦截"""
        engine = _make_engine(
            symbol_cooldown_seconds=0,
            signal_dedup_window_seconds=0,
            strategy_cooldown_seconds=0,
            global_cooldown_seconds=0,
        )
        engine.record(symbol="BTC-USDT-SWAP", direction="long", strategy_name="grid",
                      signal_type="open")
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long",
                              strategy_name="grid", signal_type="open")
        assert result.allowed is True

    def test_negative_pnl_not_blocked_without_record(self):
        """check 中传入负 pnl_usdt 但没有 record 记录亏损，应该不拦截"""
        engine = _make_engine(loss_cooldown_seconds=120)
        result = engine.check(symbol="BTC-USDT-SWAP", direction="long", pnl_usdt=-5.0)
        assert result.allowed is True

    def test_with_very_large_volatility(self):
        engine = _make_engine()
        engine.set_market_state(volatility=0.50, drawdown=0.0, account_tier="small")
        mult = engine._compute_adaptive_multiplier()
        assert mult >= 1.5  # 至少波动率因子生效

    def test_debounce_result_defaults(self):
        result = DebounceResult()
        assert result.allowed is True
        assert result.blocked_layer is None
        assert result.remaining_cooldown == 0.0
        assert result.adaptive_multiplier == 1.0