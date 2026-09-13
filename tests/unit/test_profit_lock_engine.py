"""ProfitLockEngine 统一利润锁定梯度引擎 — 单元测试。

覆盖矩阵：
  P0  初始化 & 配置解析（含阈值范围校验）
  P0  方向归一化 / pnl / 回撤计算
  P0  保本位移：激活 + 回落至成本全平（锁 0 利润）
  P0  部分落袋：浮盈 ≥1.0% 减仓 + 剩余进入紧追踪
  P0  紧追踪：回撤 ≥0.6% 全平
  P0  反转落袋：评分 ≥0.5 且浮盈 → 全平
  P0  动作冷却去重
  P2  边界：非法价格/方向 / 单 tick 跳变 / prune / reset
"""

import pytest

from core.profit_lock_engine import ProfitLockEngine, ProfitLockDecision


def _make_engine(**overrides) -> ProfitLockEngine:
    """构造带 profit_lock 配置段的引擎，允许覆盖参数。"""
    cfg = {"profit_lock": {}}
    for key, value in overrides.items():
        cfg["profit_lock"][key] = value
    return ProfitLockEngine(cfg)


# ═══════════════════════════════════════════════════════════════
# P0: 初始化 & 配置解析
# ═══════════════════════════════════════════════════════════════

class TestInitialization:
    def test_defaults(self):
        eng = _make_engine()
        assert eng.enabled is True
        assert eng._breakeven_pct == 0.005
        assert eng._partial_pct == 0.01
        assert eng._partial_ratio == 0.35
        assert eng._trailing_distance_pct == 0.006
        assert eng._reversal_close_score == 0.5

    def test_partial_ratio_clamped(self):
        # 非法比例回退到 (0.05, 0.95) 区间
        assert _make_engine(partial_ratio=5.0)._partial_ratio == 0.95
        assert _make_engine(partial_ratio=-1.0)._partial_ratio == 0.05

    def test_nan_threshold_fallbacks(self):
        eng = _make_engine(breakeven_pct=float("nan"), trailing_distance_pct=None)
        assert eng._breakeven_pct == 0.005
        assert eng._trailing_distance_pct == 0.006

    def test_disabled(self):
        eng = _make_engine(enabled=False)
        assert eng.enabled is False


# ═══════════════════════════════════════════════════════════════
# P0: 方向归一化 / pnl / 回撤
# ═══════════════════════════════════════════════════════════════

class TestBasics:
    def test_normalize_direction(self):
        assert ProfitLockEngine._normalize_direction("buy") == "long"
        assert ProfitLockEngine._normalize_direction("LONG") == "long"
        assert ProfitLockEngine._normalize_direction("sell") == "short"
        assert ProfitLockEngine._normalize_direction("junk") == "junk"

    def test_pnl_long_short(self):
        assert abs(ProfitLockEngine._pnl_pct("long", 100, 101) - 0.01) < 1e-9
        assert abs(ProfitLockEngine._pnl_pct("short", 100, 99) - 0.01) < 1e-9
        assert ProfitLockEngine._pnl_pct("long", 0, 101) == 0.0

    def test_retrace(self):
        assert abs(ProfitLockEngine._retrace_pct("long", 100, 99.5) - 0.005) < 1e-9
        assert abs(ProfitLockEngine._retrace_pct("short", 100, 100.5) - 0.005) < 1e-9
        assert ProfitLockEngine._retrace_pct("long", 100, 101) == 0.0


# ═══════════════════════════════════════════════════════════════
# P0: 保本位移
# ═══════════════════════════════════════════════════════════════

class TestBreakeven:
    def test_arm_breakeven(self):
        eng = _make_engine()
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 100.6)
        assert d.action == "none"          # 0.6% 达保本激活，但未到部分落袋/回想
        assert d.phase == "breakeven"

    def test_breakeven_protect_full_close(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "long", 100, 100.6)   # arm breakeven
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 99.9)  # 回落至成本
        assert d.action == "full"
        assert d.exit_reason == "profit_lock_breakeven"

    def test_short_breakeven(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "short", 100, 99.4)   # +0.6% arm
        d = eng.compute("BTC-USDT-SWAP", "short", 100, 100.1)  # 反弹至成本上方
        assert d.action == "full"
        assert d.exit_reason == "profit_lock_breakeven"

    def test_no_close_while_still_profitable_above_cost(self):
        eng = _make_engine()
        eng.compute("BTC-USDT-SWAP", "long", 100, 100.6)   # arm
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 100.4)  # 仍在成本上方
        assert d.action == "none"


# ═══════════════════════════════════════════════════════════════
# P0: 部分落袋 + 紧追踪
# ═══════════════════════════════════════════════════════════════

class TestPartialAndTrailing:
    def test_partial_at_1pct(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "long", 100, 100.6)   # breakeven armed
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 101.0)  # 1.0% → partial
        assert d.action == "partial"
        assert d.exit_reason == "profit_lock_partial"
        assert abs(d.partial_ratio - 0.35) < 1e-9
        assert d.phase == "trailing"

    def test_trailing_full_close_after_partial(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "long", 100, 101.0)   # jump to partial
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 100.3)  # retrace (101->100.3) ≈0.69%
        assert d.action == "full"
        assert d.exit_reason == "profit_lock_trailing"

    def test_single_tick_jump_triggers_partial(self):
        eng = _make_engine(cooldown_seconds=0)
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 102.0)  # 单 tick 直接 2%
        assert d.action == "partial"
        assert d.partial_ratio == 0.35

    def test_short_partial_and_trailing(self):
        eng = _make_engine(cooldown_seconds=0)
        d1 = eng.compute("BTC-USDT-SWAP", "short", 100, 98.9)  # +1.1% → partial
        assert d1.action == "partial"
        d2 = eng.compute("BTC-USDT-SWAP", "short", 100, 99.6)  # 从 98.9 反弹 → trailing
        assert d2.action == "full"
        assert d2.exit_reason == "profit_lock_trailing"


# ═══════════════════════════════════════════════════════════════
# P0: 反转落袋
# ═══════════════════════════════════════════════════════════════

class TestReversal:
    def test_reversal_full_close(self):
        eng = _make_engine(cooldown_seconds=0)
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 100.5, reversal_score=0.8)
        assert d.action == "full"
        assert d.exit_reason == "profit_lock_reversal"

    def test_reversal_requires_profit(self):
        eng = _make_engine()
        # 0.1% 浮盈 < reversal_min_profit_pct(0.2%)，不触发反转落袋
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 100.1, reversal_score=0.9)
        assert d.action == "none"

    def test_reversal_below_score_not_trigger(self):
        eng = _make_engine()
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 101.0, reversal_score=0.3)
        assert d.action == "partial"  # 正常梯度：1% 部分落袋，而非反转全平


# ═══════════════════════════════════════════════════════════════
# P0: 冷却去重
# ═══════════════════════════════════════════════════════════════

class TestCooldown:
    def test_partial_cooldown_blocks_trailing(self):
        eng = _make_engine(cooldown_seconds=1000)
        eng.compute("BTC-USDT-SWAP", "long", 100, 101.0)   # partial (写 last_action_ts)
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 100.3)  # 立即尝试 trailing
        assert d.action == "none"   # 冷却期内不重复触发
        assert eng.get_state("BTC-USDT-SWAP", "long")["phase"] == "trailing"


# ═══════════════════════════════════════════════════════════════
# P2: 边界与状态管理
# ═══════════════════════════════════════════════════════════════

class TestEdgeAndState:
    def test_invalid_entry(self):
        eng = _make_engine()
        d = eng.compute("BTC-USDT-SWAP", "long", 0, 101)
        assert d.action == "none"

    def test_nan_price(self):
        eng = _make_engine()
        d = eng.compute("BTC-USDT-SWAP", "long", float("nan"), 101)
        assert d.action == "none"

    def test_invalid_direction(self):
        eng = _make_engine()
        d = eng.compute("BTC-USDT-SWAP", "sideways", 100, 101)
        assert d.action == "none"

    def test_prune_and_reset(self):
        eng = _make_engine()
        eng.compute("BTC-USDT-SWAP", "long", 100, 100.6)
        eng.compute("ETH-USDT-SWAP", "short", 200, 198)
        assert len(eng.get_all_states()) == 2

        eng.reset_position("BTC-USDT-SWAP", "long")
        assert "BTC-USDT-SWAP:long" not in eng.get_all_states()

        eng.prune({"ETH-USDT-SWAP:short"})
        assert list(eng.get_all_states().keys()) == ["ETH-USDT-SWAP:short"]

        eng.reset()
        assert eng.get_all_states() == {}


# ═══════════════════════════════════════════════════════════════
# P0: fee-aware 保本缓冲（企业级统一口径）
# ═══════════════════════════════════════════════════════════════

class TestFeeAwareBreakeven:
    def test_buffer_raises_breakeven_exit_line(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "long", 100, 100.6)  # arm breakeven
        # 回落至 +0.1%（高于 0 但低于 0.15% 手续费缓冲）→ 应平仓锁费
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 100.1, breakeven_buffer=0.0015)
        assert d.action == "full"
        assert d.exit_reason == "profit_lock_breakeven"

    def test_no_close_above_buffer(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "long", 100, 100.6)  # arm
        # 回落至 +0.3%（高于缓冲 0.15%）→ 不触发保本平仓
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 100.3, breakeven_buffer=0.0015)
        assert d.action == "none"

    def test_invalid_buffer_sanitized(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "long", 100, 100.6)
        d = eng.compute("BTC-USDT-SWAP", "long", 100, 99.9, breakeven_buffer=float("nan"))
        assert d.action == "full"  # NaN buffer → 按默认 0 处理，回落至成本触发


# ═══════════════════════════════════════════════════════════════
# P0: 阈值硬下限护栏
# ═══════════════════════════════════════════════════════════════

class TestGuardrails:
    def test_breakeven_pct_floor(self):
        assert _make_engine(breakeven_pct=0.0001)._breakeven_pct == 0.002

    def test_partial_pct_floor_and_order(self):
        # partial_pct 低于下限时回退到 0.5%，且不得低于保本阈值
        assert _make_engine(partial_pct=0.0001)._partial_pct == 0.005
        # partial_pct 低于 breakeven_pct 时，提升到 breakeven_pct
        eng = _make_engine(breakeven_pct=0.01, partial_pct=0.006)
        assert eng._partial_pct == 0.01

    def test_trailing_distance_floor(self):
        assert _make_engine(trailing_distance_pct=0.0)._trailing_distance_pct == 0.002
        assert _make_engine(trailing_distance_pct=-1.0)._trailing_distance_pct == 0.002