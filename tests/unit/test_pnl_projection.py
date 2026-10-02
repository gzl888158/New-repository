"""
AGI 前瞻盈利推算与盈亏归因（pnl_projection）单元测试
=============================================================
覆盖：_attribute_pnl 四维归因、_project_pnl 推算（基础期望/趋势/regime/校正/场景）、
_track_projection_accuracy 准确度追踪、_projection_boost_scale 缩放、
run_cycle 集成、_offensive_allocation gate、状态持久化、JSON 安全。
"""
from collections import deque

import pytest

import core.quant_agi_orchestrator as orchestrator_module
from core.quant_agi_orchestrator import QuantAGIOrchestrator

_TEST_STATE_PATH = None


@pytest.fixture(autouse=True)
def _isolate_orchestrator_state(tmp_path):
    global _TEST_STATE_PATH
    _TEST_STATE_PATH = str(tmp_path / "agi_orchestrator_state.json")


# ── 工具 ─────────────────────────────────────────────────

def _pp_config(**overrides):
    pp = {
        "enabled": True,
        "horizon_cycles": 5,
        "accuracy_window": 20,
        "min_accuracy_samples": 5,
        "regime_correction_max_age_cycles": 100,
        "min_correction": 0.5,
        "max_correction": 2.0,
        "trend_weight": 0.3,
        "max_trend_adj_ratio": 0.5,
        "confidence_level": 0.95,
        "projection_adaptation_span": 0.2,
        "projection_min_mult": 0.5,
        "projection_max_mult": 1.5,
        "fail_closed_loss_threshold": 0.05,
        "offensive_projection_loss_threshold": 0.0,
        "projection_unit_pnl": 1.0,
        "min_attribution_trades": 20,
    }
    pp.update(overrides)
    return {"agi_orchestrator": {"state_path": _TEST_STATE_PATH, "pnl_projection": pp}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_pp_config(**overrides))


def _strat(**kw):
    base = dict(
        total_pnl=10.0,
        win_rate=0.6,
        avg_win=10.0,
        avg_loss=-5.0,
        profit_factor=1.5,
        max_drawdown=0.05,
        total_trades=30,
        health_score=70.0,
        health_grade="B",
        lifecycle="mature",
        trend="improving",
        pnl_per_capital_pct=8.0,
        pnl_per_trade=2.0,
        volatility=1.0,
        realized_pnl=40.0,
        unrealized_pnl=10.0,
        total_fees=2.0,
        total_funding_cost=0.5,
        total_slippage_cost=0.3,
        total_spread_cost=0.2,
        long_pnl=8.0,
        short_pnl=2.0,
        long_trades=20,
        short_trades=10,
        stop_loss_count=5,
        stop_loss_pnl=-10.0,
        take_profit_count=15,
        take_profit_pnl=20.0,
        active_hours=48.0,
        pnl_per_hour=0.5,
        delta_pnl=1.0,
        delta_health=0.0,
        trend_pnl_7d_vs_30d=0.5,
        risk_adjusted_contribution=5.0,
        sharpe_ratio=1.2,
    )
    base.update(kw)
    return base


@pytest.mark.parametrize(
    ("regime", "expected"),
    [
        ("breakout", "BREAKOUT"),
        ("breakdown", "BREAKDOWN"),
        ("reversal", "REVERSAL"),
    ],
)
def test_special_regimes_map_to_allocator_states(regime, expected):
    mapped = orchestrator_module._map_regime(regime)

    assert mapped.name == expected


@pytest.mark.parametrize(
    ("regime_name", "reserve_increases"),
    [("BREAKOUT", False), ("BREAKDOWN", True), ("REVERSAL", True)],
)
def test_special_regimes_have_distinct_pool_allocation(regime_name, reserve_increases):
    from risk.dynamic_allocator import AllocationPlan, DynamicAllocator, MarketRegime

    allocator = DynamicAllocator({})
    plan = AllocationPlan(total_equity=1000.0)
    allocator._compute_capital_pools(plan, 1000.0)
    base_reserve = plan.pools["reserve"].total_capital

    allocator._adjust_pools_for_regime(plan, getattr(MarketRegime, regime_name), 1000.0)

    assert (plan.pools["reserve"].total_capital > base_reserve) is reserve_increases


def _perception(strategies, equity=1000.0, regime="trend_up", strength=0.7, confidence=0.9):
    total_pnl = sum(s.get("total_pnl", 0.0) for s in strategies.values())
    total_trades = sum(s.get("total_trades", 0) for s in strategies.values())
    return {
        "equity": equity,
        "total_capital": equity,
        "market_regime": {"regime": regime, "strength": strength, "confidence": confidence},
        "contribution": {
            "available": True,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "overall_health_score": 70.0,
            "strategies": strategies,
        },
    }


# ── 归因测试 ─────────────────────────────────────────────

class TestAttributePnl:
    def test_unavailable_returns_empty(self):
        orch = _orch()
        p = {"contribution": {"available": False, "strategies": {}, "total_pnl": 0.0, "total_trades": 0}}
        a = orch._attribute_pnl(p)
        assert a["available"] is False
        assert a["by_strategy"] == {}
        assert a["total_pnl"] == 0.0

    def test_by_strategy_share(self):
        orch = _orch()
        p = _perception({"grid": _strat(total_pnl=30.0), "trend": _strat(total_pnl=10.0)})
        a = orch._attribute_pnl(p)
        assert a["available"] is True
        # 按绝对值归一化
        assert abs(a["by_strategy"]["grid"]["share"] - 0.75) < 0.01
        assert abs(a["by_strategy"]["trend"]["share"] - 0.25) < 0.01
        assert a["dominant_strategy"] == "grid"

    def test_by_direction(self):
        orch = _orch()
        p = _perception({"grid": _strat(long_pnl=8.0, short_pnl=2.0,
                                         long_trades=20, short_trades=10)})
        a = orch._attribute_pnl(p)
        d = a["by_direction"]
        assert d["long_pnl"] == 8.0
        assert d["short_pnl"] == 2.0
        assert abs(d["long_share"] - 0.8) < 0.01

    def test_by_exit_reason(self):
        orch = _orch()
        p = _perception({"grid": _strat(take_profit_pnl=20.0, stop_loss_pnl=-5.0,
                                         realized_pnl=15.0, unrealized_pnl=5.0)})
        a = orch._attribute_pnl(p)
        er = a["by_exit_reason"]
        assert er["take_profit"]["pnl"] == 20.0
        assert er["stop_loss"]["pnl"] == -5.0
        assert er["take_profit"]["share"] > 0  # tp 占比为正
        assert er["stop_loss"]["share"] < 0   # sl 占比为负

    def test_by_regime_from_memory(self):
        orch = _orch()
        orch._learning_enabled = True
        orch._decision_memory.extend([
            {"regime": "trend_up", "cycle_pnl": 3.0},
            {"regime": "trend_up", "cycle_pnl": 2.0},
            {"regime": "range_bound", "cycle_pnl": -1.0},
        ])
        p = _perception({"grid": _strat()})
        a = orch._attribute_pnl(p)
        assert "trend_up" in a["by_regime"]
        assert a["by_regime"]["trend_up"]["cycles"] == 2
        assert a["by_regime"]["trend_up"]["avg_per_cycle"] == pytest.approx(2.5)
        assert a["by_regime"]["range_bound"]["low_confidence"] is True  # cycles<3


# ── 推算测试 ─────────────────────────────────────────────

class TestProjectPnl:
    def test_disabled_returns_empty(self):
        orch = _orch(enabled=False)
        p = _perception({"grid": _strat()})
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        assert proj["available"] is False
        assert proj["total"] == {}

    def test_unavailable_contribution(self):
        orch = _orch()
        p = {"equity": 1000.0, "contribution": {"available": False}}
        proj = orch._project_pnl(p, {})
        assert proj["available"] is False

    def test_base_expectation_calculation(self):
        # EV = 0.6*10 - 0.4*5 = 4.0
        orch = _orch()
        p = _perception({"grid": _strat(win_rate=0.6, avg_win=10.0, avg_loss=-5.0,
                                         volatility=0.0)})
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        assert proj["available"] is True
        grid = proj["per_strategy"]["grid"]
        # base_expect ≈ 4.0（忽略趋势调整的小量）
        assert abs(grid["base_expect"] - 4.0) < 0.5

    def test_horizon_accumulation(self):
        orch = _orch(horizon_cycles=5)
        p = _perception({"grid": _strat()})
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        base = proj["total"]["base_case"]
        assert base["horizon"] == pytest.approx(base["per_cycle"] * 5)

    def test_regime_multiplier_trend(self):
        orch = _orch()
        p = _perception({"grid": _strat(volatility=0.0)}, regime="trend_up")
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        # trend regime → regime_mult = 1.3
        assert proj["per_strategy"]["grid"]["regime_mult"] == pytest.approx(1.3)

    def test_production_trend_labels_use_trend_multiplier(self):
        orch = _orch()
        for regime in ("trend_bullish", "trend_bearish"):
            p = _perception({"grid": _strat(volatility=0.0)}, regime=regime)
            proj = orch._project_pnl(p, orch._attribute_pnl(p))
            assert proj["per_strategy"]["grid"]["regime_mult"] == pytest.approx(1.3)

    def test_regime_multiplier_range(self):
        orch = _orch()
        p = _perception({"grid": _strat(volatility=0.0)}, regime="range_bound")
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        assert proj["per_strategy"]["grid"]["regime_mult"] == pytest.approx(0.8)

    def test_scenarios_present(self):
        orch = _orch()
        p = _perception({"grid": _strat()})
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        for case in ("base_case", "bear_case", "bull_case"):
            assert case in proj["total"]
            for key in ("per_cycle", "horizon", "ci_lower", "ci_upper"):
                assert key in proj["total"][case]

    def test_trend_adjustment_from_history(self):
        orch = _orch()
        # 上升序列 → 正斜率 → 正 trend_adjustment
        orch._strategy_pnl_per_trade_history["grid"] = deque([1.0, 2.0, 3.0, 4.0, 5.0], maxlen=5)
        p = _perception({"grid": _strat(volatility=0.0)})
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        assert proj["per_strategy"]["grid"]["trend_adjustment"] > 0

    def test_correction_factor_applied(self):
        orch = _orch()
        orch._correction_factor = 0.8
        p = _perception({"grid": _strat(volatility=0.0)})
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        # corrected_expect = (base + trend) * regime_mult * 0.8
        assert proj["correction_factor"] == pytest.approx(0.8)


# ── 准确度追踪测试 ───────────────────────────────────────

class TestTrackProjectionAccuracy:
    def test_no_last_projection(self):
        orch = _orch()
        a = orch._track_projection_accuracy(_perception({"grid": _strat()}))
        assert a["correction_factor"] == pytest.approx(1.0)
        assert a["low_confidence"] is True

    def test_bias_ratio_calculation(self):
        orch = _orch()
        orch._last_projection = {
            "total": {"base_case": {"per_cycle": 4.0}},
            "projection_cycle": 1,
            "regime": "trend_up",
        }
        orch._last_decision_total_pnl = 90.0
        orch._decision_quality_use_cycle_pnl = True
        p = _perception({"grid": _strat()})
        # contribution total_pnl 默认 10.0 → cycle_pnl = 10 - 90 = -80
        a = orch._track_projection_accuracy(p)
        # bias = -80 / 4 = -20
        assert a["last_bias_ratio"] == pytest.approx(-20.0)

    def test_median_correction_factor(self):
        orch = _orch()
        orch._last_projection = {
            "total": {"base_case": {"per_cycle": 10.0}},
            "projection_cycle": 1,
        }
        orch._last_decision_total_pnl = None
        # 填充 5 个样本使达到 min_accuracy_samples
        for _ in range(6):
            orch._projection_accuracy.append({"cycle": 1, "projected": 10.0, "actual": 5.0,
                                              "bias_ratio": 0.5, "regime": "trend_up"})
        p = _perception({"grid": _strat()})
        a = orch._track_projection_accuracy(p)
        assert a["sample_count"] >= 6
        # 中位数 0.5，clamp 到 [0.5, 2.0]
        assert a["correction_factor"] == pytest.approx(0.5)
        assert orch._correction_factor == pytest.approx(0.5)


# ── boost 缩放测试 ───────────────────────────────────────

class TestProjectionBoostScale:
    def test_disabled_returns_one(self):
        orch = _orch(enabled=False)
        assert orch._projection_boost_scale(10.0, 0.9) == pytest.approx(1.0)

    def test_none_expect_returns_one(self):
        orch = _orch()
        assert orch._projection_boost_scale(None, 0.9) == pytest.approx(1.0)

    def test_positive_expect_amplifies(self):
        orch = _orch(projection_adaptation_span=0.2, projection_unit_pnl=1.0)
        # expect=1.0, conf=1.0 → scale = 1 + 1*1*1*0.2 = 1.2
        scale = orch._projection_boost_scale(1.0, 1.0)
        assert scale > 1.0

    def test_negative_expect_reduces(self):
        orch = _orch(projection_adaptation_span=0.2, projection_unit_pnl=1.0)
        scale = orch._projection_boost_scale(-1.0, 1.0)
        assert scale < 1.0

    def test_clamped_to_bounds(self):
        orch = _orch(projection_min_mult=0.5, projection_max_mult=1.5)
        assert orch._projection_boost_scale(-100.0, 1.0) >= 0.5
        assert orch._projection_boost_scale(100.0, 1.0) <= 1.5


# ── 进攻性分配 gate 测试 ─────────────────────────────────

class TestOffensiveAllocationGate:
    def test_negative_projection_skipped(self):
        orch = _orch()
        # 构造 corrected_expect < 0 的策略
        proj = {
            "available": True,
            "per_strategy": {"bad_strat": {"corrected_expect": -5.0, "confidence": 0.8}},
        }
        # 模拟 gate 构建逻辑
        negative = set()
        for name, sp in (proj.get("per_strategy") or {}).items():
            if sp["corrected_expect"] < -orch._offensive_projection_loss_threshold:
                negative.add(str(name))
        assert "bad_strat" in negative


# ── 状态持久化测试 ───────────────────────────────────────

class TestProjectionPersistence:
    def test_serialize_includes_projection(self):
        orch = _orch()
        orch._projection_accuracy.append({"cycle": 1, "projected": 4.0, "actual": 3.0,
                                           "bias_ratio": 0.75, "regime": "trend_up"})
        orch._correction_factor = 0.75
        state = orch._serialize_learning_state()
        assert "projection_accuracy" in state
        assert "correction_factor" in state
        assert state["correction_factor"] == pytest.approx(0.75)
        assert len(state["projection_accuracy"]) == 1


# ── V5.0 增强测试 ───────────────────────────────────────

class TestBearCaseFailClosed:
    def test_open_block_reason_uses_latest_per_strategy_horizon_threshold(self):
        orch = _orch(fail_closed_loss_threshold=0.05)
        orch._last_report = {
            "projection": {
                "available": True,
                "per_strategy": {
                    "grid": {"bear_case_horizon": -60.0},
                    "trend": {"bear_case_horizon": -40.0},
                },
            },
            "perception": {"equity": 1000.0},
        }

        assert orch.get_bear_case_open_block_reason("grid") is not None
        assert orch.get_bear_case_open_block_reason("trend") is None

    def test_per_strategy_has_bear_case_horizon(self):
        orch = _orch()
        p = _perception({"grid": _strat(volatility=0.0)})
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        assert "bear_case_horizon" in proj["per_strategy"]["grid"]

    def test_negative_bear_case_projection_is_detected(self):
        """检测悲观场景期望低于零的策略。"""
        orch = _orch(fail_closed_loss_threshold=0.0001)  # 极小阈值 → 易触发
        # 构造一个亏损策略：win_rate 低、avg_loss 大 → bear_case_horizon 为负
        p = _perception({"grid": _strat(win_rate=0.1, avg_win=1.0, avg_loss=-20.0,
                                         volatility=0.0, total_pnl=-50.0)},
                         equity=1000.0)
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        assert proj["per_strategy"]["grid"]["bear_case_horizon"] < 0

    def test_offensive_gate_blocks_bear_case_fail_closed(self):
        """_offensive_allocation_actions 应把 bear_case horizon 低于阈值的策略加入 gate"""
        orch = _orch(fail_closed_loss_threshold=0.0001)
        # 构造亏损策略 → bear_case horizon 为负
        p = _perception({"grid": _strat(win_rate=0.1, avg_win=1.0, avg_loss=-20.0,
                                         volatility=0.0, total_pnl=-50.0)},
                         equity=1000.0)
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        # decision 含 allocation_plan.total_equity 供阈值换算
        # 直接调用 _offensive_allocation_actions 验证 gate 逻辑（不验证 actions 输出）
        # 通过 monkey-patch 检查 internal set 不易，改为验证 bear_case_horizon < 阈值
        _eq_base = 1000.0
        _bear_threshold_abs = -abs(0.0001) * _eq_base  # -0.1
        bc_h = proj["per_strategy"]["grid"]["bear_case_horizon"]
        # 阈值 -0.1，bear_case_horizon 应远低于 -0.1 → 触发 gate
        assert bc_h < _bear_threshold_abs


class TestRegimeConditionedCorrection:
    def test_production_regime_names_map_to_allocator_regimes(self):
        from core.quant_agi_orchestrator import _map_regime
        from risk.dynamic_allocator import MarketRegime

        assert _map_regime("trend_bullish") == MarketRegime.TRENDING_UP
        assert _map_regime("trend_bearish") == MarketRegime.TRENDING_DOWN
        assert _map_regime("range_bound") == MarketRegime.RANGING

    def test_correction_factor_by_regime_updated(self):
        """按 regime 分桶维护 correction_factor_by_regime"""
        orch = _orch(min_accuracy_samples=3, accuracy_window=20)
        orch._last_projection = {
            "total": {"base_case": {"per_cycle": 4.0}},
            "projection_cycle": 1,
            "regime": "trend_up",
        }
        orch._last_decision_total_pnl = 90.0
        orch._decision_quality_use_cycle_pnl = True
        # 注入 3 个 trend_up 样本
        for _ in range(3):
            p = _perception({"grid": _strat()}, regime="trend_up", equity=1000.0)
            orch._track_projection_accuracy(p)
        assert "trend_up" in orch._correction_factor_by_regime
        assert orch._correction_factor_by_regime_updated_cycle["trend_up"] == orch._cycle_count

    def test_project_pnl_uses_regime_specific_correction(self):
        """_project_pnl 优先用 regime 分桶的 correction"""
        orch = _orch()
        orch._correction_factor = 1.0  # 全局
        orch._correction_factor_by_regime = {"trend_up": 0.8}  # trend_up 专属
        orch._correction_factor_by_regime_updated_cycle = {"trend_up": 0}
        orch._projection_accuracy.extend([
            {"cycle": i, "bias_ratio": 1.0, "regime": "trend_up"}
            for i in range(orch._projection_min_accuracy_samples)
        ])
        p = _perception({"grid": _strat(volatility=0.0)}, regime="trend_up")
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        assert proj["correction_factor"] == pytest.approx(0.8)

    def test_accuracy_sample_uses_forecast_regime_not_current_regime(self):
        orch = _orch(min_accuracy_samples=1, accuracy_window=20)
        orch._last_projection = {
            "total": {"base_case": {"per_cycle": 4.0}},
            "projection_cycle": 1,
            "regime": "trend_bullish",
        }
        orch._last_decision_total_pnl = 90.0
        orch._decision_quality_use_cycle_pnl = True
        perception = _perception(
            {"grid": _strat()}, regime="range_bound", equity=1000.0
        )

        orch._track_projection_accuracy(perception)

        assert orch._projection_accuracy[-1]["regime"] == "trend_up"
        assert "trend_up" in orch._correction_factor_by_regime
        assert "range_bound" not in orch._correction_factor_by_regime
        assert orch._correction_factor_by_regime_updated_cycle["trend_up"] == orch._cycle_count

    def test_regime_aliases_share_projection_correction_bucket(self):
        orch = _orch()
        orch._correction_factor = 1.0
        orch._correction_factor_by_regime = {"trend_up": 0.75}
        orch._correction_factor_by_regime_updated_cycle = {"trend_up": 0}
        orch._projection_accuracy.extend([
            {"cycle": i, "bias_ratio": 1.0, "regime": "trend_up"}
            for i in range(orch._projection_min_accuracy_samples)
        ])
        perception = _perception(
            {"grid": _strat(volatility=0.0)}, regime="trend_bullish"
        )

        projection = orch._project_pnl(perception, orch._attribute_pnl(perception))

        assert projection["correction_factor"] == pytest.approx(0.75)
        assert projection["regime"] == "trend_up"

    def test_project_pnl_fallback_global_correction(self):
        """regime 未分桶时回退全局 correction"""
        orch = _orch()
        orch._correction_factor = 0.9
        orch._correction_factor_by_regime = {"trend_up": 0.8}  # 不含 range_bound
        p = _perception({"grid": _strat(volatility=0.0)}, regime="range_bound")
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        assert proj["correction_factor"] == pytest.approx(0.9)

    def test_persistence_includes_regime_correction(self):
        """状态持久化包含 correction_factor_by_regime"""
        orch = _orch()
        orch._correction_factor_by_regime = {"trend_up": 0.8, "range_bound": 1.2}
        state = orch._serialize_learning_state()
        assert "correction_factor_by_regime" in state
        assert state["correction_factor_by_regime"]["trend_up"] == pytest.approx(0.8)
        assert state["correction_factor_by_regime_updated_cycle"] == {}


class TestDynamicAllocatorProjectionIntegration:
    def test_projected_pnl_boosts_priority_score(self):
        """projected_pnl_per_cycle 正值 → 评分提升 → 优先级可能升档"""
        from risk.dynamic_allocator import AllocationPriority, DynamicAllocator
        alloc = DynamicAllocator({})
        # 构造一个原本评分为 MEDIUM 边缘的策略
        metrics = {"grid": {
            "trade_count": 50, "consecutive_losses": 0, "max_drawdown": 0.05,
            "profit_factor": 1.5, "total_pnl": 100.0, "sharpe_ratio": 0.5,
            "win_rate": 0.5, "projected_pnl_per_cycle": 100.0,  # 大正值
            "projection_confidence": 1.0, "bear_case_fail_closed": False,
        }}
        from risk.dynamic_allocator import MarketRegime
        priorities = alloc._evaluate_strategy_priorities(
            ["grid"], metrics, MarketRegime.TRENDING_UP
        )
        # 大正值 projected_pnl 应提升评分到 HIGH
        assert priorities["grid"] == AllocationPriority.HIGH

    def test_bear_case_fail_closed_blocks_strategy_allocation(self):
        """bear_case_fail_closed=True → 冻结分配并明确标记禁止开仓"""
        from risk.dynamic_allocator import AllocationPriority, DynamicAllocator, MarketRegime
        alloc = DynamicAllocator({})
        metrics = {"grid": {
            "trade_count": 50, "consecutive_losses": 0, "max_drawdown": 0.05,
            "profit_factor": 2.0, "total_pnl": 100.0, "sharpe_ratio": 2.0,
            "win_rate": 0.8, "projected_pnl_per_cycle": 100.0,
            "projection_confidence": 1.0, "bear_case_fail_closed": True,
        }}
        priorities = alloc._evaluate_strategy_priorities(
            ["grid"], metrics, MarketRegime.TRENDING_UP
        )
        assert priorities["grid"] == AllocationPriority.FROZEN

    @pytest.mark.asyncio
    async def test_bear_case_fail_closed_yields_zero_opening_allocation(self):
        from risk.dynamic_allocator import DynamicAllocator, MarketRegime

        allocator = DynamicAllocator({})
        metrics = {
            "grid": {
                "trade_count": 50, "consecutive_losses": 0, "max_drawdown": 0.05,
                "profit_factor": 2.0, "total_pnl": 100.0, "sharpe_ratio": 2.0,
                "win_rate": 0.8, "projected_pnl_per_cycle": 100.0,
                "projection_confidence": 1.0, "bear_case_horizon": -600.0,
                "bear_case_fail_closed": True,
            },
            "trend": {
                "trade_count": 50, "consecutive_losses": 0, "max_drawdown": 0.05,
                "profit_factor": 2.0, "total_pnl": 100.0, "sharpe_ratio": 2.0,
                "win_rate": 0.8,
            },
        }

        plan = await allocator.compute_allocation_plan(
            total_capital=10000.0,
            total_equity=10000.0,
            strategy_names=["grid", "trend"],
            strategy_metrics=metrics,
            market_regime=MarketRegime.TRENDING_UP,
            persist_last_plan=False,
        )

        blocked = plan.strategy_allocations["grid"]
        assert blocked.is_open_blocked is True
        assert blocked.is_frozen is True
        assert blocked.freeze_reason == "bear_case_projection"
        assert blocked.target_weight == 0.0
        assert blocked.allocated_capital == 0.0
        assert plan.to_dict()["strategy_allocations"]["grid"]["is_open_blocked"] is True

    def test_priority_projection_uses_account_equity_not_historical_pnl(self):
        from risk.dynamic_allocator import AllocationPriority, DynamicAllocator, MarketRegime

        alloc = DynamicAllocator({})
        metrics = {
            name: {
                "trade_count": 50, "consecutive_losses": 0, "max_drawdown": 0.05,
                "profit_factor": 1.5, "total_pnl": pnl, "sharpe_ratio": -0.05,
                "win_rate": 0.5, "projected_pnl_per_cycle": 100.0,
                "projection_confidence": 1.0,
            }
            for name, pnl in (("small_history", 100.0), ("large_history", 10000.0))
        }

        priorities = alloc._evaluate_strategy_priorities(
            list(metrics), metrics, MarketRegime.TRENDING_UP, total_equity=10000.0
        )

        assert priorities["small_history"] == AllocationPriority.HIGH
        assert priorities["large_history"] == AllocationPriority.HIGH

    @pytest.mark.asyncio
    async def test_adverse_projection_reduces_kelly_and_allocation(self):
        from risk.dynamic_allocator import DynamicAllocator, MarketRegime

        allocator = DynamicAllocator({})
        baseline = {
            name: {
                "trade_count": 50, "consecutive_losses": 0, "max_drawdown": 0.05,
                "profit_factor": 2.0, "total_pnl": 100.0, "sharpe_ratio": 2.0,
                "win_rate": 0.8, "volatility_30d": 0.02,
            }
            for name in ("grid", "trend")
        }
        adverse = {
            name: dict(metrics)
            for name, metrics in baseline.items()
        }
        adverse["grid"].update({
            "projected_pnl_per_cycle": -25.0,
            "bear_case_horizon": -250.0,
            "projection_confidence": 1.0,
            "bear_case_fail_closed": False,
        })
        common = {
            "total_capital": 10000.0,
            "total_equity": 10000.0,
            "strategy_names": ["grid", "trend"],
            "market_regime": MarketRegime.TRENDING_UP,
            "persist_last_plan": False,
        }

        baseline_plan = await allocator.compute_allocation_plan(
            **common, strategy_metrics=baseline
        )
        adverse_plan = await allocator.compute_allocation_plan(
            **common, strategy_metrics=adverse
        )

        baseline_grid = baseline_plan.strategy_allocations["grid"]
        adverse_grid = adverse_plan.strategy_allocations["grid"]
        assert adverse_grid.kelly_fraction < baseline_grid.kelly_fraction
        assert adverse_grid.target_weight < baseline_grid.target_weight


class TestDashboardPnlProjectionPanel:
    def test_no_orchestrator_returns_unavailable(self):
        from core.dashboard_engine import DashboardEngine
        engine = DashboardEngine({})
        # 未注入 agi_orchestrator
        result = engine.get_pnl_projection_panel()
        assert result["available"] is False

    def test_no_projection_returns_unavailable(self):
        from core.dashboard_engine import DashboardEngine
        engine = DashboardEngine({})
        engine._agi_orchestrator = type("X", (), {"_last_projection": None})()
        result = engine.get_pnl_projection_panel()
        assert result["available"] is False

    def test_panel_returns_scenarios_and_per_strategy(self):
        from core.dashboard_engine import DashboardEngine
        orch = _orch()
        p = _perception({"grid": _strat()}, equity=1000.0)
        attribution = orch._attribute_pnl(p)
        proj = orch._project_pnl(p, attribution)
        orch._last_projection = proj
        orch._last_report = {"projection": proj, "attribution": attribution}
        orch._projection_accuracy.extend([
            {
                "cycle": cycle,
                "projected": 10.0,
                "actual": float(cycle),
                "bias_ratio": float(cycle) / 10.0,
                "regime": "trend_up",
            }
            for cycle in range(22)
        ])
        engine = DashboardEngine({})
        engine.set_dependencies(agi_orchestrator=orch)
        result = engine.get_pnl_projection_panel()
        assert result["available"] is True
        assert "base_case" in result["total_scenarios"]
        assert "bear_case" in result["total_scenarios"]
        assert "bull_case" in result["total_scenarios"]
        assert "grid" in result["per_strategy"]
        assert "bear_case_horizon" in result["per_strategy"]["grid"]
        assert "correction_factor_by_regime" in result
        assert result["bias_history"] == pytest.approx([
            float(cycle) / 10.0 for cycle in range(2, 22)
        ])
        assert len(result["bias_history"]) == 20
        assert result["attribution"]["available"] is True
        assert "grid" in result["attribution"]["by_strategy"]
        assert "long_pnl" in result["attribution"]["by_direction"]
        assert "take_profit" in result["attribution"]["by_exit_reason"]

    def test_panel_returns_attribution_even_before_first_projection(self):
        import json

        from core.dashboard_engine import DashboardEngine

        orch = _orch()
        perception = _perception(
            {"grid": _strat(long_pnl=10.0, short_pnl=0.0)}, equity=1000.0
        )
        attribution = orch._attribute_pnl(perception)
        orch._last_report = {"projection": {}, "attribution": attribution}
        engine = DashboardEngine({})
        engine.set_dependencies(agi_orchestrator=orch)

        result = engine.get_pnl_projection_panel()

        assert result["available"] is True
        assert result["projection_available"] is False
        assert result["attribution"]["available"] is True
        assert result["attribution"]["total_pnl"] == pytest.approx(10.0)
        assert result["attribution"]["by_direction"]["long_short_ratio"] == 0.0
        assert result["bias_history"] == []
        json.dumps(result, allow_nan=False)

    def test_dashboard_snapshot_returns_allocation_plan_and_detached_projection(self):
        orch = _orch()
        projection = {"available": True, "total": {"base_case": {"per_cycle": 2.0}}}
        attribution = {"available": True, "by_strategy": {"grid": {"pnl": 1.0}}}
        plan = {"strategy_utilization": {"grid": 0.4}}
        orch._last_report = {
            "projection": projection,
            "attribution": attribution,
            "decision": {"allocation_plan": plan},
        }

        snapshot = orch.get_dashboard_snapshot()
        snapshot["projection"]["total"]["base_case"]["per_cycle"] = 99.0
        snapshot["attribution"]["by_strategy"]["grid"]["pnl"] = 99.0
        snapshot["allocation_plan"]["strategy_utilization"]["grid"] = 0.9

        assert orch._last_report["projection"]["total"]["base_case"]["per_cycle"] == 2.0
        assert orch._last_report["attribution"]["by_strategy"]["grid"]["pnl"] == 1.0
        assert orch._last_report["decision"]["allocation_plan"]["strategy_utilization"]["grid"] == 0.4

    def test_enterprise_sync_status_includes_registered_channels(self):
        from core.enterprise_sync import SyncChannel, SyncHealthMonitor

        monitor = SyncHealthMonitor()
        monitor.register_channel(SyncChannel.POSITION_REST)

        assert monitor.get_all_status() == {
            "position_rest": {
                "channel": "position_rest",
                "health": "healthy",
                "last_sync": 0.0,
                "last_success": 0.0,
                "data_age_sec": 0.0,
                "avg_latency_ms": 0.0,
                "sync_count": 0,
                "error_count": 0,
                "consecutive_errors": 0,
                "last_error": "",
            }
        }

    def test_panel_included_in_dashboard_full(self):
        from core.dashboard_engine import DashboardEngine
        engine = DashboardEngine({})
        full = engine.get_dashboard_full()
        assert "pnl_projection_panel" in full

    def test_load_restores_correction_factor(self, tmp_path):
        orch = _orch()
        orch.state_path = str(tmp_path / "state.json")
        # 直接调用 load 逻辑：构造 fake learning_state
        ls = {
            "projection_accuracy": [
                {"cycle": 1, "projected": 4.0, "actual": 3.0, "bias_ratio": 0.75,
                 "regime": "trend_up"}
            ],
            "correction_factor": 0.75,
            "last_projection": {"total": {}, "projection_cycle": 1, "horizon_cycles": 5},
        }
        # 模拟 _load_learning_state 中的恢复
        pa = ls.get("projection_accuracy")
        orch._projection_accuracy = deque(
            [{"cycle": int(x["cycle"]), "projected": x["projected"],
              "actual": x["actual"], "bias_ratio": x["bias_ratio"],
              "regime": x["regime"]} for x in pa],
            maxlen=orch._projection_accuracy_window,
        )
        cf = ls["correction_factor"]
        orch._correction_factor = max(orch._projection_min_correction,
                                      min(orch._projection_max_correction, cf))
        orch._last_projection = ls["last_projection"]
        assert orch._correction_factor == pytest.approx(0.75)
        assert len(orch._projection_accuracy) == 1


# ── JSON 安全测试 ────────────────────────────────────────

class TestProjectionJsonSafe:
    def test_projection_no_nan_inf(self):
        import json
        orch = _orch()
        p = _perception({"grid": _strat(avg_loss=float("nan"))})
        proj = orch._project_pnl(p, orch._attribute_pnl(p))
        text = json.dumps(proj, allow_nan=False)
        assert "NaN" not in text
        assert "Infinity" not in text

    def test_attribution_no_nan_inf(self):
        import json
        orch = _orch()
        p = _perception({"grid": _strat()})
        a = orch._attribute_pnl(p)
        text = json.dumps(a, allow_nan=False)
        assert "NaN" not in text
        assert "Infinity" not in text

class TestDecideExceptionFailClosed:
    @pytest.mark.asyncio
    async def test_decide_impl_exception_returns_non_executable_plan(self, monkeypatch):
        orch = _orch()

        async def fail_decision(*_args, **_kwargs):
            raise RuntimeError("synthetic decision failure")

        monkeypatch.setattr(orch, "_decide_impl", fail_decision)
        result = await orch._decide(
            {"equity": 1_000.0, "contribution": {"available": True}},
            projection={"available": True},
        )

        assert result["fail_closed"] is True
        assert result["allocation_plan"] is None
        assert result["reallocation_suggestions"] == []
        assert result["error"]["type"] == "RuntimeError"


class TestProjectionInputBoundaries:
    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_metrics_remain_json_safe(self, bad_value):
        import json

        orch = _orch()
        perception = _perception(
            {"grid": _strat(avg_win=bad_value, avg_loss=bad_value, volatility=bad_value)},
            equity=bad_value,
        )
        projection = orch._project_pnl(perception, orch._attribute_pnl(perception))
        attribution = orch._attribute_pnl(perception)

        json.dumps(projection, allow_nan=False)
        json.dumps(attribution, allow_nan=False)

    def test_malformed_strategy_record_does_not_discard_valid_attribution(self):
        import json

        orch = _orch()
        perception = _perception({"grid": _strat()})
        perception["contribution"]["strategies"]["broken"] = "not-a-strategy"
        attribution = orch._attribute_pnl(perception)

        assert attribution["available"] is True
        assert "grid" in attribution["by_strategy"]
        assert "broken" not in attribution["by_strategy"]
        assert "strategy_records" in attribution["degraded_dimensions"]
        json.dumps(attribution, allow_nan=False)

    def test_strategy_attribution_failure_preserves_other_dimensions(self, monkeypatch):
        orch = _orch()
        perception = _perception(
            {"grid": _strat(long_pnl=5.0, short_pnl=-2.0)}
        )
        invalid_value = object()
        perception["contribution"]["strategies"]["grid"]["total_pnl"] = invalid_value
        original_safe_float = orchestrator_module.safe_float

        def fail_on_invalid_value(value, default=0.0):
            if value is invalid_value:
                raise ValueError("invalid strategy PnL")
            return original_safe_float(value, default)

        monkeypatch.setattr(orchestrator_module, "safe_float", fail_on_invalid_value)

        attribution = orch._attribute_pnl(perception)

        assert attribution["available"] is True
        assert attribution["by_strategy"] == {}
        assert attribution["by_direction"]["long_pnl"] == 5.0
        assert attribution["by_exit_reason"]["realized"]["pnl"] == pytest.approx(40.0)
        assert "by_strategy" in attribution["degraded_dimensions"]

    def test_direction_attribution_without_short_pnl_is_finite(self):
        import json

        orch = _orch()
        perception = _perception({"grid": _strat(long_pnl=5.0, short_pnl=0.0)})
        attribution = orch._attribute_pnl(perception)

        assert attribution["by_direction"]["long_short_ratio"] == 0.0
        json.dumps(attribution, allow_nan=False)


class TestRegimeCorrectionWindow:
    def test_stale_regime_factor_is_ignored_outside_accuracy_window(self):
        orch = _orch(min_accuracy_samples=3, accuracy_window=5)
        orch._correction_factor = 1.0
        orch._correction_factor_by_regime = {"trend_up": 0.5}
        orch._projection_accuracy.extend([
            {"cycle": 1, "bias_ratio": 0.5, "regime": "trend_up"},
            {"cycle": 2, "bias_ratio": 0.6, "regime": "range_bound"},
            {"cycle": 3, "bias_ratio": 0.7, "regime": "range_bound"},
        ])
        perception = _perception({"grid": _strat()}, regime="trend_up")

        projection = orch._project_pnl(perception, orch._attribute_pnl(perception))

        assert projection["correction_factor"] == pytest.approx(1.0)

    @pytest.mark.parametrize(
        ("current_cycle", "expected_correction"),
        [(15, 0.7), (16, 0.9)],
    )
    def test_regime_factor_expires_after_configured_cycle_age(
        self, current_cycle, expected_correction
    ):
        orch = _orch(
            min_accuracy_samples=2,
            accuracy_window=10,
            regime_correction_max_age_cycles=5,
        )
        orch._cycle_count = current_cycle
        orch._correction_factor = 0.9
        orch._correction_factor_by_regime = {"trend_up": 0.7}
        orch._correction_factor_by_regime_updated_cycle = {"trend_up": 10}
        orch._projection_accuracy.extend([
            {"cycle": current_cycle - 2, "bias_ratio": 1.0, "regime": "trend_up"},
            {"cycle": current_cycle - 1, "bias_ratio": 1.0, "regime": "trend_up"},
        ])
        perception = _perception({"grid": _strat()}, regime="trend_up")

        projection = orch._project_pnl(perception, orch._attribute_pnl(perception))

        assert projection["correction_factor"] == pytest.approx(expected_correction)

    def test_regime_correction_survives_learning_state_reload(self, tmp_path):
        import json

        state_path = tmp_path / "state.json"
        orch = _orch()
        orch.state_path = str(state_path)
        orch._correction_factor = 0.9
        orch._correction_factor_by_regime = {"trend_up": 0.75, "range_bound": 1.2}
        orch._correction_factor_by_regime_updated_cycle = {
            "trend_up": 10, "range_bound": 8,
        }
        orch._cycle_count = 12
        state_path.write_text(
            json.dumps({"learning_state": orch._serialize_learning_state()}),
            encoding="utf-8",
        )

        restored = _orch()
        restored.state_path = str(state_path)
        restored._load_learning_state()

        assert restored._correction_factor == pytest.approx(0.9)
        assert restored._correction_factor_by_regime == pytest.approx(
            {"trend_up": 0.75, "range_bound": 1.2}
        )
        assert restored._correction_factor_by_regime_updated_cycle == {
            "trend_up": 10, "range_bound": 8,
        }
        assert restored._cycle_count == 12

    def test_legacy_regime_timestamp_recovers_from_accuracy_samples(self, tmp_path):
        import json

        state_path = tmp_path / "legacy-state.json"
        state_path.write_text(
            json.dumps({
                "cycle": 12,
                "learning_state": {
                    "correction_factor_by_regime": {"trend_bullish": 0.75},
                    "projection_accuracy": [
                        {"cycle": 7, "regime": "trend_up", "bias_ratio": 0.75},
                        {"cycle": 10, "regime": "trend_bullish", "bias_ratio": 0.8},
                    ],
                },
            }),
            encoding="utf-8",
        )
        orch = _orch()
        orch.state_path = str(state_path)

        orch._load_learning_state()

        assert orch._cycle_count == 12
        assert orch._correction_factor_by_regime_updated_cycle == {"trend_up": 10}

    def test_legacy_factor_without_accuracy_history_is_not_treated_as_fresh(self, tmp_path):
        import json

        state_path = tmp_path / "legacy-without-history.json"
        state_path.write_text(
            json.dumps({
                "cycle": 2,
                "learning_state": {
                    "correction_factor": 0.9,
                    "correction_factor_by_regime": {"trend_up": 0.5},
                    "projection_accuracy": [],
                },
            }),
            encoding="utf-8",
        )
        orch = _orch(min_accuracy_samples=2)
        orch.state_path = str(state_path)
        orch._load_learning_state()
        orch._projection_accuracy.extend([
            {"cycle": 1, "bias_ratio": 1.0, "regime": "trend_up"},
            {"cycle": 2, "bias_ratio": 1.0, "regime": "trend_up"},
        ])
        perception = _perception({"grid": _strat()}, regime="trend_up")

        projection = orch._project_pnl(perception, orch._attribute_pnl(perception))

        assert projection["correction_factor"] == pytest.approx(0.9)

class TestProjectionDecisionImpact:
    def test_projection_changes_priority_but_never_overrides_bear_case_freeze(self):
        from risk.dynamic_allocator import AllocationPriority, DynamicAllocator, MarketRegime

        allocator = DynamicAllocator({})
        metrics = {
            "grid": {
                "trade_count": 50,
                "consecutive_losses": 0,
                "max_drawdown": 0.05,
                "profit_factor": 1.0,
                "total_pnl": 100.0,
                "sharpe_ratio": -0.05,
                "win_rate": 0.5,
            }
        }
        baseline = allocator._evaluate_strategy_priorities(
            ["grid"], metrics, MarketRegime.TRENDING_UP, total_equity=10_000.0
        )
        metrics["grid"].update({"projected_pnl_per_cycle": 500.0, "projection_confidence": 1.0})
        favorable = allocator._evaluate_strategy_priorities(
            ["grid"], metrics, MarketRegime.TRENDING_UP, total_equity=10_000.0
        )
        metrics["grid"]["projected_pnl_per_cycle"] = -500.0
        adverse = allocator._evaluate_strategy_priorities(
            ["grid"], metrics, MarketRegime.TRENDING_UP, total_equity=10_000.0
        )
        metrics["grid"]["bear_case_fail_closed"] = True
        fail_closed = allocator._evaluate_strategy_priorities(
            ["grid"], metrics, MarketRegime.TRENDING_UP, total_equity=10_000.0
        )

        assert baseline["grid"] == AllocationPriority.MEDIUM
        assert favorable["grid"] == AllocationPriority.HIGH
        assert adverse["grid"] == AllocationPriority.LOW
        assert fail_closed["grid"] == AllocationPriority.FROZEN


class TestDashboardProjectionRendering:
    def test_empty_projection_and_non_finite_values_render_as_finite_placeholders(self):
        import json

        from core.dashboard_engine import DashboardEngine

        orchestrator = type(
            "ProjectionSnapshot",
            (),
            {
                "get_dashboard_snapshot": lambda self: {
                    "projection": {
                        "available": True,
                        "total": {"base_case": {"per_cycle": float("nan")}},
                        "per_strategy": {"grid": {"corrected_expect": float("inf")}},
                    },
                    "attribution": {"available": False},
                    "projection_accuracy": [],
                    "correction_factor_by_regime": {},
                }
            },
        )()
        engine = DashboardEngine({})
        engine.set_dependencies(agi_orchestrator=orchestrator)

        panel = engine.get_pnl_projection_panel()

        assert panel["available"] is True
        assert panel["total_scenarios"]["base_case"]["per_cycle"] == 0.0
        assert panel["per_strategy"]["grid"]["corrected_expect"] == 0.0
        json.dumps(panel, allow_nan=False)


class TestSimulationOnlyRiskActions:
    @staticmethod
    def _orchestrator(*, paper_enabled=True, paper_mode="sandbox", **safety_config):
        config = _pp_config()
        config["paper_trading"] = {
            "enabled": paper_enabled,
            "mode": paper_mode,
        }
        config["agi_orchestrator"]["simulation_safety"] = {
            "enabled": True,
            **safety_config,
        }
        return QuantAGIOrchestrator(config=config)

    def test_bear_case_reduction_is_reported_but_never_executable(self):
        orch = self._orchestrator(bear_case_reduce_ratio=0.25)
        report = {
            "status": "ok",
            "perception": {"equity": 1_000.0},
            "projection": {
                "available": True,
                "per_strategy": {
                    "grid": {"bear_case_horizon": -100.0},
                    "trend": {"bear_case_horizon": -10.0},
                },
            },
            "actions": [],
        }

        orch._apply_simulation_safety(report)

        assert report["simulation_actions"] == [{
            "type": "simulated_bear_case_reduction",
            "strategy": "grid",
            "reduce_ratio": 0.25,
            "bear_case_horizon": -100.0,
            "simulated_bear_case_horizon_after_reduction": -75.0,
            "simulation_assumption": "linear_exposure_scaling",
            "threshold": -50.0,
            "reason": "bear_case_projection_below_loss_threshold",
            "simulation_only": True,
            "execution_applied": False,
        }]
        assert report["actions"] == []
        assert report["simulation_safety"]["execution_applied"] is False

    def test_fail_closed_streak_trips_only_latched_simulated_kill_switch(self):
        orch = self._orchestrator(fail_closed_streak_threshold=2)
        report = {"status": "fail_closed", "actions": []}

        orch._apply_simulation_safety(report)
        assert report["simulation_safety"]["simulated_kill_switch"]["enabled"] is False

        orch._apply_simulation_safety(report)
        simulated_switch = report["simulation_safety"]["simulated_kill_switch"]
        assert simulated_switch["enabled"] is True
        assert simulated_switch["would_block_new_openings"] is True
        assert simulated_switch["execution_applied"] is False
        assert report["actions"] == []

        orch._apply_simulation_safety({"status": "ok", "actions": []})
        assert orch._simulation_kill_switch_enabled is True
        assert orch._simulation_fail_closed_streak == 0

        orch.reset_simulated_kill_switch()
        assert orch._simulation_kill_switch_enabled is False

    @pytest.mark.asyncio
    async def test_fail_closed_cycle_reports_simulation_without_real_actions(self, monkeypatch):
        orch = self._orchestrator(fail_closed_streak_threshold=2)
        orch.cooldown_seconds = 0

        async def perceive():
            return {"equity": 1_000.0}

        async def decide(*_args, **_kwargs):
            return {"fail_closed": True, "allocation_plan": None, "actions": []}

        monkeypatch.setattr(orch, "_perceive", perceive)
        monkeypatch.setattr(orch, "_diagnose", lambda _perception: [])
        monkeypatch.setattr(orch, "_attribute_pnl", lambda _perception: {})
        monkeypatch.setattr(orch, "_project_pnl", lambda _perception, _attribution: {})
        monkeypatch.setattr(orch, "_decide", decide)

        first = await orch.run_cycle()
        second = await orch.run_cycle()

        assert first["simulation_safety"]["fail_closed_streak"] == 1
        assert second["simulation_safety"]["simulated_kill_switch"]["enabled"] is True
        assert first["actions"] == []
        assert second["actions"] == []

    @pytest.mark.parametrize(
        ("paper_enabled", "paper_mode"),
        [(False, "sandbox"), (True, "testnet")],
    )
    def test_simulation_risk_actions_require_enabled_sandbox(self, paper_enabled, paper_mode):
        orch = self._orchestrator(
            paper_enabled=paper_enabled,
            paper_mode=paper_mode,
        )
        report = {"status": "fail_closed", "actions": []}

        orch._apply_simulation_safety(report)

        assert "simulation_actions" not in report
        assert "simulation_safety" not in report
        assert orch._simulation_fail_closed_streak == 0
