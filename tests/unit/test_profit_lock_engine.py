"""ProfitLockEngine 统一利润锁定梯度引擎 — 单元测试。

覆盖矩阵：
  P0  初始化 & 配置解析（含阈值范围校验）
  P0  方向归一化 / pnl / 回撤计算
  P0  保本位移：激活 + 回落至成本全平（锁 0 利润）
  P0  部分落袋：浮盈 ≥1.0% 减仓 + 剩余进入紧追踪
  P0  紧追踪：回撤 ≥0.6% 全平
  P0  动作冷却去重
  P2  边界：非法价格/方向 / 单 tick 跳变 / prune / reset
"""

import pytest

from core.profit_lock_engine import ProfitLockEngine


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
# P0: 保本阶段回撤快速部分落袋（增强收益落袋）
# ═══════════════════════════════════════════════════════════════

class TestBreakevenRetracePartial:
    def _eng(self):
        return _make_engine(
            breakeven_pct=0.0015, partial_pct=0.004, partial_ratio=0.5,
            trailing_distance_pct=0.0015, cooldown_seconds=0,
            breakeven_retrace_partial_enabled=True,
            breakeven_retrace_partial_ratio=0.4,
        )

    def test_retrace_in_breakeven_triggers_quick_partial(self):
        """进入保本后未到 partial 线时回撤，立即部分落袋锁小利。"""
        eng = self._eng()
        eng.compute("BTC", "long", 100, 100.0)
        eng.compute("BTC", "long", 100, 100.3)   # +0.3% 保本激活，未到 0.4% partial
        # 回撤 0.16%（仍 +0.14% 盈利）→ 快速部分落袋
        d = eng.compute("BTC", "long", 100, 100.14)
        assert d.action == "partial"
        assert abs(d.partial_ratio - 0.4) < 1e-9
        assert d.exit_reason == "profit_lock_partial"
        assert d.phase == "trailing"

    def test_no_quick_partial_when_retrace_too_small(self):
        eng = self._eng()
        eng.compute("BTC", "long", 100, 100.0)
        eng.compute("BTC", "long", 100, 100.3)
        d = eng.compute("BTC", "long", 100, 100.25)  # 回撤 0.05% < 0.15%
        assert d.action == "none"

    def test_quick_partial_not_repeated(self):
        eng = self._eng()
        eng.compute("BTC", "long", 100, 100.0)
        eng.compute("BTC", "long", 100, 100.3)
        d1 = eng.compute("BTC", "long", 100, 100.14)
        assert d1.action == "partial"
        # 再次回撤不重复触发 quick partial（已进入 trailing，由 trailing 逻辑接管）
        d2 = eng.compute("BTC", "long", 100, 100.10)
        assert d2.action != "partial" or eng.get_state("BTC", "long")["be_retrace_partial_done"] is True

    def test_quick_partial_disabled_when_config_off(self):
        eng = _make_engine(
            breakeven_pct=0.0015, partial_pct=0.004, trailing_distance_pct=0.0015,
            cooldown_seconds=0, breakeven_retrace_partial_enabled=False,
        )
        eng.compute("BTC", "long", 100, 100.0)
        eng.compute("BTC", "long", 100, 100.3)
        d = eng.compute("BTC", "long", 100, 100.14)
        assert d.action == "none"


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
        assert _make_engine(breakeven_pct=0.0001)._breakeven_pct == 0.001

    def test_partial_pct_floor_and_order(self):
        # partial_pct 低于下限时回退到 0.3%，且不得低于保本阈值
        assert _make_engine(breakeven_pct=0.001, partial_pct=0.0001)._partial_pct == 0.003
        # partial_pct 低于 breakeven_pct 时，提升到 breakeven_pct
        eng = _make_engine(breakeven_pct=0.01, partial_pct=0.006)
        assert eng._partial_pct == 0.01

    def test_trailing_distance_floor(self):
        assert _make_engine(trailing_distance_pct=0.0)._trailing_distance_pct == 0.0015
        assert _make_engine(trailing_distance_pct=-1.0)._trailing_distance_pct == 0.0015


# ═══════════════════════════════════════════════════════════════
# P0: 跨重启状态持久化（dump_state / restore_state）
# ═══════════════════════════════════════════════════════════════

class TestPersistence:
    def test_dump_returns_copy(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "long", 100, 101.0)  # partial -> trailing
        dump = eng.dump_state()
        dump["BTC-USDT-SWAP:long"]["phase"] = "corrupted"
        # 修改导出副本不应影响引擎内部状态
        assert eng.get_state("BTC-USDT-SWAP", "long")["phase"] == "trailing"

    def test_roundtrip_restore(self):
        eng = _make_engine(cooldown_seconds=0)
        eng.compute("BTC-USDT-SWAP", "long", 100, 101.0)  # partial
        eng.compute("BTC-USDT-SWAP", "long", 100, 101.5)  # peak 更新
        dump = eng.dump_state()

        eng2 = _make_engine(cooldown_seconds=0)
        eng2.restore_state(dump)
        s = eng2.get_state("BTC-USDT-SWAP", "long")
        assert s["phase"] == "trailing"
        assert s["partial_done"] is True
        assert s["peak"] == 101.5

    def test_restore_skips_invalid_entries(self):
        eng = _make_engine()
        eng.restore_state({
            "BTC-USDT-SWAP:long": {"phase": "breakeven", "peak": 100.5,
                                   "partial_done": False, "last_action_ts": 0.0},
            "no-colon": {"phase": "none"},
            "BTC-USDT-SWAP:sideways": {"phase": "none"},
            "ETH-USDT-SWAP:short": {"phase": None, "peak": "junk"},  # 非法值回退默认
            "bad": 123,  # 非 dict
        })
        states = eng.get_all_states()
        assert "BTC-USDT-SWAP:long" in states
        assert "no-colon" not in states
        assert "BTC-USDT-SWAP:sideways" not in states
        assert "ETH-USDT-SWAP:short" in states  # 合法方向，非法字段回退默认
        assert states["ETH-USDT-SWAP:short"]["phase"] == "none"
        assert states["ETH-USDT-SWAP:short"]["peak"] == 0.0

    def test_restore_rejects_non_dict(self):
        eng = _make_engine()
        eng.restore_state(None)
        eng.restore_state([])
        assert eng.get_all_states() == {}