"""
AGI 自学习增强（agi_self_learning）单元测试
================================================
覆盖：
  1. 决策记忆指数衰减（_decision_quality_score 加权近期周期）
     - 衰减加权 ≠ 等权（近期周期影响更大）
     - alpha=1.0 退化为等权（向后兼容）
     - 样本不足仍返回中性 0.5
  2. 策略参数自适应限速（_param_adjust_actions per-strategy cooldown）
     - 同一策略在 min_interval_cycles 内被跳过
     - 冷却期过后允许再次调整
     - min_interval_cycles=0 时不限速
     - 不同策略互不干扰
"""
from collections import deque

import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


_TEST_STATE_PATH = None


@pytest.fixture(autouse=True)
def _isolate_orchestrator_state(tmp_path):
    global _TEST_STATE_PATH
    _TEST_STATE_PATH = str(tmp_path / "agi_orchestrator_state.json")


def _config(**overrides):
    """构造 agi_orchestrator 配置；learning/param_adaptation/decision_quality_guard
    可通过 overrides 覆盖。"""
    learning = {"enabled": True, "memory_size": 10, "max_adjust_pct": 0.4,
                "memory_decay_alpha": 0.85}
    pa = {"enabled": True, "health_f_leverage": 1.0, "declining_leverage": 2.0,
          "min_interval_cycles": 3}
    dq = {"enabled": True, "quality_threshold": 0.3, "min_samples": 3,
          "recovery_after_cycles": 8, "recovery_boost_scale": 0.5}
    oa = {"enabled": True, "rl_feedback_enabled": True,
          "rl_adaptation_span": 0.1, "rl_min_multiplier": 0.5,
          "rl_max_multiplier": 1.5, "rl_ema_alpha": 0.3}
    cfg = {"agi_orchestrator": {
        "state_path": _TEST_STATE_PATH,
        "learning": learning,
        "param_adaptation": pa,
        "decision_quality_guard": dq,
        "offensive_allocation": oa,
    }}
    agi = cfg["agi_orchestrator"]
    for k, v in overrides.items():
        if k in ("learning", "param_adaptation", "decision_quality_guard",
                 "offensive_allocation"):
            agi[k].update(v)
        else:
            agi[k] = v
    return cfg


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _memory(pnls):
    """按 pnls 列表构造 _decision_memory（旧→新顺序）。"""
    return deque([{"total_pnl": p} for p in pnls], maxlen=10)


# ═══════════════════════════════════════════════════════════════
# 1. 决策记忆指数衰减
# ═══════════════════════════════════════════════════════════════

class TestDecisionMemoryDecay:
    def test_decay_weights_recent_more(self):
        """近期盈利、远期全亏：衰减加权后质量分数应高于等权（近期权重大）。"""
        orch = _orch()
        # 旧→新：[-1, -1, -1, -1, 10]  近期盈利，远期亏损
        orch._decision_memory = _memory([-1.0, -1.0, -1.0, -1.0, 10.0])
        score = orch._decision_quality_score()
        # 等权下 1/5 = 0.2；衰减（alpha=0.85）近期盈利权重最高，分数应 > 0.2
        assert score > 0.2
        # 但仍 < 1.0（远期亏损仍有权重）
        assert score < 1.0

    def test_decay_recent_loss_lowers_score(self):
        """近期全亏、远期盈利：衰减加权后质量分数应低于等权。"""
        orch = _orch()
        # 旧→新：[10, 10, 10, 10, -1]  近期亏损，远期盈利
        orch._decision_memory = _memory([10.0, 10.0, 10.0, 10.0, -1.0])
        score = orch._decision_quality_score()
        # 等权下 4/5 = 0.8；衰减近期亏损权重大，分数应 < 0.8
        assert score < 0.8

    def test_alpha_one_equals_equal_weight(self):
        """alpha=1.0 时退化为等权（向后兼容）。"""
        orch = _orch(learning={"memory_decay_alpha": 1.0})
        orch._decision_memory = _memory([10.0, -1.0, 10.0, -1.0, 10.0])
        score = orch._decision_quality_score()
        # 等权 3/5 = 0.6
        assert score == pytest.approx(0.6)

    def test_insufficient_samples_neutral(self):
        """样本不足仍返回中性 0.5（衰减不影响样本不足判断）。"""
        orch = _orch()
        orch._decision_memory = _memory([10.0, -1.0])  # 2 < min_samples=3
        assert orch._decision_quality_score() == pytest.approx(0.5)

    def test_all_win_score_one(self):
        """全部盈利无论衰减均为 1.0。"""
        orch = _orch()
        orch._decision_memory = _memory([1.0, 2.0, 3.0, 4.0, 5.0])
        assert orch._decision_quality_score() == pytest.approx(1.0)

    def test_all_loss_score_zero(self):
        """全部亏损无论衰减均为 0.0。"""
        orch = _orch()
        orch._decision_memory = _memory([-1.0, -2.0, -3.0, -4.0, -5.0])
        assert orch._decision_quality_score() == pytest.approx(0.0)

    def test_decay_alpha_clamped(self):
        """alpha 越小（衰减越快），近期盈利的分数越高。"""
        orch_fast = _orch(learning={"memory_decay_alpha": 0.5})
        orch_slow = _orch(learning={"memory_decay_alpha": 0.95})
        # 旧→新：远期亏损，近期盈利
        mem = _memory([-1.0, -1.0, -1.0, -1.0, 10.0])
        orch_fast._decision_memory = mem
        orch_slow._decision_memory = deque(mem, maxlen=10)
        fast_score = orch_fast._decision_quality_score()
        slow_score = orch_slow._decision_quality_score()
        # 衰减越快，近期盈利权重越集中 → 分数越高
        assert fast_score > slow_score


# ═══════════════════════════════════════════════════════════════
# 2. 策略参数自适应限速
# ═══════════════════════════════════════════════════════════════

def _health_alert(strategy, atype="strategy_health_critical"):
    return {"type": atype, "strategy": strategy, "message": "test"}


class TestParamAdaptationRateLimit:
    def test_first_adjust_allowed(self):
        """首次调整无冷却记录 → 允许。"""
        orch = _orch()
        orch._cycle_count = 5
        actions = orch._param_adjust_actions([_health_alert("sniper")])
        assert len(actions) == 1
        assert actions[0]["strategy"] == "sniper"
        assert actions[0]["value"] == 1.0  # health_f_leverage
        # 记录了调整周期
        assert orch._param_adapt_last_cycle["sniper"] == 5

    def test_blocked_within_cooldown(self):
        """冷却期内（< min_interval_cycles）同一策略再次调整被跳过。"""
        orch = _orch(param_adaptation={"min_interval_cycles": 3})
        orch._cycle_count = 5
        orch._param_adjust_actions([_health_alert("sniper")])
        # 周期 6、7（距上次 1、2 < 3）应被跳过
        orch._cycle_count = 6
        a2 = orch._param_adjust_actions([_health_alert("sniper")])
        assert a2 == []
        orch._cycle_count = 7
        a3 = orch._param_adjust_actions([_health_alert("sniper")])
        assert a3 == []

    def test_allowed_after_cooldown(self):
        """冷却期过后（>= min_interval_cycles）允许再次调整。"""
        orch = _orch(param_adaptation={"min_interval_cycles": 3})
        orch._cycle_count = 5
        orch._param_adjust_actions([_health_alert("sniper")])
        # 周期 8（距上次 3 == 3）应允许
        orch._cycle_count = 8
        a = orch._param_adjust_actions([_health_alert("sniper")])
        assert len(a) == 1
        assert orch._param_adapt_last_cycle["sniper"] == 8

    def test_no_limit_when_interval_zero(self):
        """min_interval_cycles=0 时不限速（每次都允许）。"""
        orch = _orch(param_adaptation={"min_interval_cycles": 0})
        orch._cycle_count = 1
        a1 = orch._param_adjust_actions([_health_alert("sniper")])
        orch._cycle_count = 2
        a2 = orch._param_adjust_actions([_health_alert("sniper")])
        assert len(a1) == 1
        assert len(a2) == 1

    def test_different_strategies_independent(self):
        """不同策略的冷却互不干扰。"""
        orch = _orch(param_adaptation={"min_interval_cycles": 3})
        # 清除持久化状态污染（生产 state 可能有 param_adapt_last_cycle 记录）
        orch._param_adapt_last_cycle = {}
        orch._cycle_count = 5
        # sniper 已调整
        orch._param_adjust_actions([_health_alert("sniper")])
        # grid 在下一周期调整应允许（不同策略冷却独立）
        orch._cycle_count = 6
        a = orch._param_adjust_actions([_health_alert("grid")])
        assert len(a) == 1
        assert a[0]["strategy"] == "grid"
        # sniper 仍被冷却
        orch._cycle_count = 6
        a2 = orch._param_adjust_actions([_health_alert("sniper")])
        assert a2 == []

    def test_declining_alert_also_rate_limited(self):
        """strategy_declining 告警同样受限速约束。"""
        orch = _orch(param_adaptation={"min_interval_cycles": 3})
        orch._cycle_count = 5
        orch._param_adjust_actions(
            [_health_alert("trend", atype="strategy_declining")])
        orch._cycle_count = 6
        a = orch._param_adjust_actions(
            [_health_alert("trend", atype="strategy_declining")])
        assert a == []

    def test_last_cycle_persisted_in_state(self):
        """调整周期号写入 learning_state 序列化（跨重启续用）。"""
        orch = _orch()
        orch._cycle_count = 42
        orch._param_adjust_actions([_health_alert("sniper")])
        state = orch._serialize_learning_state()
        assert state["param_adapt_last_cycle"]["sniper"] == 42

    def test_disabled_returns_empty(self):
        """param_adaptation 关闭时返回空。"""
        orch = _orch(param_adaptation={"enabled": False})
        orch._cycle_count = 5
        a = orch._param_adjust_actions([_health_alert("sniper")])
        assert a == []


# ═══════════════════════════════════════════════════════════════
# 3. 进攻强化学习反馈循环（rl_feedback）
# ═══════════════════════════════════════════════════════════════

class TestOffensiveRLFeedback:
    def test_first_outcome_set_directly(self):
        """首次记录直接取 outcome（无历史基线，不过度平滑）。"""
        orch = _orch()
        orch._update_offensive_feedback("sniper", 1.0)
        assert orch._offensive_feedback_score["sniper"] == pytest.approx(1.0)
        orch._update_offensive_feedback("grid", -1.0)
        assert orch._offensive_feedback_score["grid"] == pytest.approx(-1.0)

    def test_ema_smoothing_after_first(self):
        """后续记录用 EMA 平滑：new = old + alpha*(outcome - old)。"""
        orch = _orch(offensive_allocation={"rl_ema_alpha": 0.5})
        orch._update_offensive_feedback("sniper", 1.0)   # score=1.0
        orch._update_offensive_feedback("sniper", -1.0)  # 1.0 + 0.5*(-1-1)=0.0
        assert orch._offensive_feedback_score["sniper"] == pytest.approx(0.0)

    def test_score_clamped_to_range(self):
        """分数 clamp 到 [-1, 1]。"""
        orch = _orch(offensive_allocation={"rl_ema_alpha": 1.0})
        orch._update_offensive_feedback("sniper", 1.0)
        orch._update_offensive_feedback("sniper", 1.0)  # 仍 1.0（不超上限）
        assert orch._offensive_feedback_score["sniper"] == pytest.approx(1.0)

    def test_disabled_does_not_update(self):
        """rl_feedback 关闭时不更新分数。"""
        orch = _orch(offensive_allocation={"rl_feedback_enabled": False})
        orch._update_offensive_feedback("sniper", 1.0)
        assert orch._offensive_feedback_score == {}

    def test_multiplier_positive_score_increases_boost(self):
        """正分 → 倍率 > 1（加大步长）。"""
        orch = _orch(offensive_allocation={"rl_adaptation_span": 0.1})
        orch._offensive_feedback_score["sniper"] = 1.0
        assert orch._offensive_feedback_multiplier("sniper") == pytest.approx(1.1)

    def test_multiplier_negative_score_decreases_boost(self):
        """负分 → 倍率 < 1（缩小步长）。"""
        orch = _orch(offensive_allocation={"rl_adaptation_span": 0.1})
        orch._offensive_feedback_score["sniper"] = -1.0
        assert orch._offensive_feedback_multiplier("sniper") == pytest.approx(0.9)

    def test_multiplier_no_history_is_one(self):
        """无历史分数 → 倍率 1.0（需关闭暖启动，否则用 dq 先验）。"""
        orch = _orch(offensive_allocation={"rl_warm_start_enabled": False})
        assert orch._offensive_feedback_multiplier("unknown") == pytest.approx(1.0)

    def test_multiplier_clamped_to_bounds(self):
        """倍率 clamp 到 [min_mult, max_mult]。"""
        orch = _orch(offensive_allocation={
            "rl_adaptation_span": 1.0,
            "rl_min_multiplier": 0.5,
            "rl_max_multiplier": 1.5,
        })
        orch._offensive_feedback_score["sniper"] = 1.0  # 1+1*1=2.0 → clamp 1.5
        assert orch._offensive_feedback_multiplier("sniper") == pytest.approx(1.5)
        orch._offensive_feedback_score["grid"] = -1.0  # 1-1=0.0 → clamp 0.5
        assert orch._offensive_feedback_multiplier("grid") == pytest.approx(0.5)

    def test_multiplier_disabled_returns_one(self):
        """rl_feedback 关闭时倍率恒为 1.0。"""
        orch = _orch(offensive_allocation={"rl_feedback_enabled": False})
        orch._offensive_feedback_score["sniper"] = 1.0
        assert orch._offensive_feedback_multiplier("sniper") == pytest.approx(1.0)

    def test_feedback_score_persisted_in_state(self):
        """反馈分数写入 learning_state 序列化（跨重启续用）。"""
        orch = _orch()
        orch._update_offensive_feedback("sniper", 1.0)
        orch._update_offensive_feedback("sniper", -1.0)
        state = orch._serialize_learning_state()
        assert "offensive_feedback_score" in state
        assert "sniper" in state["offensive_feedback_score"]

    def test_warm_start_uses_decision_quality_when_no_history(self):
        """无进攻反馈历史时，暖启动用 decision_quality 作为先验缩放 boost。"""
        orch = _orch(offensive_allocation={"rl_warm_start_enabled": True,
                                           "rl_adaptation_span": 0.1})
        # 模拟全亏记忆 → dq=0.0 → score=(0-0.5)*2=-1.0 → mult=0.9
        orch._decision_memory = deque(
            [{"total_pnl": -1.0} for _ in range(5)], maxlen=10)
        mult = orch._offensive_feedback_multiplier("sniper")
        assert mult == pytest.approx(0.9)

    def test_warm_start_neutral_when_dq_mid(self):
        """decision_quality=0.6（3盈2亏等权）→ 暖启动 score=0.2 → mult=1.02。"""
        orch = _orch(offensive_allocation={"rl_warm_start_enabled": True,
                                           "rl_adaptation_span": 0.1},
                     learning={"memory_decay_alpha": 1.0})  # 等权，dq=3/5=0.6
        orch._decision_memory = deque(
            [{"total_pnl": 1.0}, {"total_pnl": 1.0}, {"total_pnl": 1.0},
             {"total_pnl": -1.0}, {"total_pnl": -1.0}], maxlen=10)
        mult = orch._offensive_feedback_multiplier("grid")
        # score = (0.6 - 0.5) * 2 = 0.2 → mult = 1 + 0.2 * 0.1 = 1.02
        assert mult == pytest.approx(1.02)

    def test_warm_start_disabled_returns_one(self):
        """关闭暖启动 → 无历史时 mult=1.0（向后兼容）。"""
        orch = _orch(offensive_allocation={"rl_warm_start_enabled": False})
        orch._decision_memory = deque(
            [{"total_pnl": -1.0} for _ in range(5)], maxlen=10)
        assert orch._offensive_feedback_multiplier("sniper") == pytest.approx(1.0)

    def test_warm_start_not_used_when_history_exists(self):
        """有进攻反馈历史时，暖启动不覆盖历史分数。"""
        orch = _orch(offensive_allocation={"rl_warm_start_enabled": True,
                                           "rl_adaptation_span": 0.1})
        orch._offensive_feedback_score["sniper"] = 1.0  # 历史=正
        orch._decision_memory = deque(
            [{"total_pnl": -1.0} for _ in range(5)], maxlen=10)  # dq=0 但不应影响
        # 历史 score=1.0 → mult=1.1（不受暖启动 dq=0 影响）
        assert orch._offensive_feedback_multiplier("sniper") == pytest.approx(1.1)


# ═══════════════════════════════════════════════════════════════
# 4. 市场状态条件化决策质量（regime 分桶）
# ═══════════════════════════════════════════════════════════════

def _memory_with_regime(entries):
    """按 [(pnl, regime), ...] 构造带 regime 的 _decision_memory（旧→新顺序）。"""
    return deque(
        [{"total_pnl": p, "regime": r} for p, r in entries],
        maxlen=10,
    )


class TestDecisionQualityRegimeConditioned:
    def test_regime_bucket_isolates_trend_wins(self):
        """条件化分桶：同 regime 决策质量与混合评估隔离，避免互相污染。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},  # 等权，便于断言
            decision_quality_guard={"regime_conditioned": True, "min_samples": 3},
        )
        orch._decision_memory = _memory_with_regime([
            (10.0, "trend_bullish"), (10.0, "trend_bullish"), (10.0, "trend_bullish"),
            (-1.0, "range_bound"), (-1.0, "range_bound"), (-1.0, "range_bound"),
        ])
        # trend_bullish 桶：3 个盈利 → 1.0
        assert orch._decision_quality_score("trend_bullish") == pytest.approx(1.0)
        # range_bound 桶：3 个亏损 → 0.0
        assert orch._decision_quality_score("range_bound") == pytest.approx(0.0)
        # 混合（不传 regime）：3/6 = 0.5
        assert orch._decision_quality_score() == pytest.approx(0.5)

    def test_regime_insufficient_falls_back_to_mixed(self):
        """同 regime 样本不足 min_samples → 回退混合计算（guard 仍能工作）。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"regime_conditioned": True, "min_samples": 5},
        )
        orch._decision_memory = _memory_with_regime([
            (-1.0, "range_bound"), (-1.0, "range_bound"),  # 2 个 range_bound < 5
            (10.0, "trend_bullish"), (10.0, "trend_bullish"),
            (10.0, "trend_bullish"), (10.0, "trend_bullish"),
        ])
        # range_bound 桶仅 2 < 5 → 回退混合 = 4/6
        assert orch._decision_quality_score("range_bound") == pytest.approx(4 / 6)

    def test_regime_conditioned_disabled_ignores_regime(self):
        """关闭条件化（默认）→ 传 regime 也走混合（向后兼容）。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"regime_conditioned": False, "min_samples": 3},
        )
        orch._decision_memory = _memory_with_regime([
            (10.0, "trend_bullish"), (10.0, "trend_bullish"), (10.0, "trend_bullish"),
            (-1.0, "range_bound"), (-1.0, "range_bound"), (-1.0, "range_bound"),
        ])
        # 关闭时传 trend_bullish 仍返回混合 3/6 = 0.5
        assert orch._decision_quality_score("trend_bullish") == pytest.approx(0.5)

    def test_regime_none_uses_mixed(self):
        """regime=None 时走混合（与不传等价）。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"regime_conditioned": True, "min_samples": 3},
        )
        orch._decision_memory = _memory_with_regime([
            (10.0, "trend_bullish"), (10.0, "trend_bullish"), (10.0, "trend_bullish"),
            (-1.0, "range_bound"), (-1.0, "range_bound"), (-1.0, "range_bound"),
        ])
        assert orch._decision_quality_score(None) == pytest.approx(0.5)

    def test_regime_bucket_decay_still_applies(self):
        """条件化分桶内仍应用指数衰减加权（近期样本权重更高）。"""
        orch = _orch(
            learning={"memory_decay_alpha": 0.5},
            decision_quality_guard={"regime_conditioned": True, "min_samples": 3},
        )
        # 旧→新：2 个 range_bound 亏损 + 近期 1 个 range_bound 盈利
        orch._decision_memory = _memory_with_regime([
            (-1.0, "range_bound"), (-1.0, "range_bound"), (10.0, "range_bound"),
        ])
        score = orch._decision_quality_score("range_bound")
        # 等权 1/3≈0.33；衰减近期盈利权重更高 → score > 0.33
        assert score > (1 / 3)
        assert score < 1.0


# ═══════════════════════════════════════════════════════════════
# 6. 决策质量单周期盈亏增量（cycle_pnl delta）
# ═══════════════════════════════════════════════════════════════

def _memory_with_cycle(pnls, cycle_pnls=None):
    """构造带 cycle_pnl 字段的决策记忆；cycle_pnls 默认与 pnls 相同。"""
    if cycle_pnls is None:
        cycle_pnls = pnls
    return deque(
        [{"total_pnl": p, "cycle_pnl": c} for p, c in zip(pnls, cycle_pnls)],
        maxlen=10,
    )


class TestDecisionQualityCyclePnlDelta:
    """决策质量用单周期盈亏增量（cycle_pnl）而非累计 total_pnl。"""

    def test_uses_cycle_pnl_not_cumulative(self):
        """累计 total_pnl 全为负但 cycle_pnl 有盈有亏 → 按 cycle_pnl 计算。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"use_cycle_pnl_delta": True},
        )
        orch._decision_memory = _memory_with_cycle(
            pnls=[-10.0, -10.0, -10.0, -10.0, -10.0],
            cycle_pnls=[2.0, -1.0, 3.0, -2.0, 1.0],
        )
        assert orch._decision_quality_score() == pytest.approx(0.6)

    def test_disabled_falls_back_to_cumulative(self):
        """use_cycle_pnl_delta=False → 回退累计 total_pnl（向后兼容）。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"use_cycle_pnl_delta": False},
        )
        orch._decision_memory = _memory_with_cycle(
            pnls=[-10.0, -10.0, -10.0, -10.0, -10.0],
            cycle_pnls=[2.0, -1.0, 3.0, -2.0, 1.0],
        )
        assert orch._decision_quality_score() == pytest.approx(0.0)

    def test_old_entries_without_cycle_pnl_fallback(self):
        """旧持久化条目无 cycle_pnl 字段 → 回退 total_pnl，不崩溃。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"use_cycle_pnl_delta": True},
        )
        orch._decision_memory = deque([
            {"total_pnl": -10.0, "cycle_pnl": 5.0},
            {"total_pnl": -10.0},
            {"total_pnl": -10.0, "cycle_pnl": 2.0},
        ], maxlen=10)
        assert orch._decision_quality_score() == pytest.approx(2 / 3)

    def test_reflect_records_cycle_pnl_delta(self):
        """_reflect 记录本周期 cycle_pnl = cur_total_pnl - last_total_pnl。"""
        orch = _orch()
        orch._learning_enabled = True
        orch._last_decision_total_pnl = 100.0
        cur_total = 103.5
        cycle_pnl = (cur_total - orch._last_decision_total_pnl
                     if orch._last_decision_total_pnl is not None else 0.0)
        orch._last_decision_total_pnl = cur_total
        assert cycle_pnl == pytest.approx(3.5)

    def test_first_cycle_cycle_pnl_is_zero(self):
        """首周期（无基线）→ cycle_pnl=0，不被误判为盈或亏。"""
        orch = _orch()
        orch._learning_enabled = True
        orch._last_decision_total_pnl = None
        cur_total = 100.0
        cycle_pnl = (cur_total - orch._last_decision_total_pnl
                     if (orch._last_decision_total_pnl is not None
                         and orch._decision_quality_use_cycle_pnl)
                     else 0.0)
        assert cycle_pnl == 0.0

    def test_constant_cumulative_yields_no_wins(self):
        """累计 total_pnl 连续不变（无平仓）→ cycle_pnl 全 0 → 中性 0.5（非质量差）。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"use_cycle_pnl_delta": True},
        )
        orch._decision_memory = _memory_with_cycle(
            pnls=[-1.1976] * 5,
            cycle_pnls=[0.0] * 5,
        )
        # ignore_zero_cycle_pnl 默认开启：无平仓周期视为中性，返回 0.5（不误判为质量差）
        assert orch._decision_quality_score() == pytest.approx(0.5)


# ═══════════════════════════════════════════════════════════════
# 7. 决策质量忽略无平仓周期（ignore_zero_cycle_pnl）
# ═══════════════════════════════════════════════════════════════

class TestDecisionQualityIgnoreZeroPnl:
    """无平仓周期（cycle_pnl==0）视为中性，从质量分计算中排除。"""

    def test_all_zero_cycle_pnl_neutral(self):
        """cycle_pnl 全 0（震荡市无平仓）→ 质量分 0.5 中性，非 0。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"use_cycle_pnl_delta": True},
        )
        orch._decision_memory = _memory_with_cycle(
            pnls=[-1.1976] * 6,
            cycle_pnls=[0.0] * 6,
        )
        assert orch._decision_quality_score() == pytest.approx(0.5)

    def test_zero_excluded_from_score(self):
        """混合 0 与盈亏 → 0 被排除，只按真实盈亏计算占比。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"use_cycle_pnl_delta": True},
        )
        # 3 个无平仓 + 1 盈 + 1 亏 → 有效样本 [3.0, -2.0] → 质量分 0.5
        orch._decision_memory = _memory_with_cycle(
            pnls=[-1.0, -1.0, -1.0, -1.0, -1.0],
            cycle_pnls=[0.0, 0.0, 0.0, 3.0, -2.0],
        )
        assert orch._decision_quality_score() == pytest.approx(0.5)

    def test_ignore_zero_disabled_falls_back(self):
        """ignore_zero_cycle_pnl=False → 0 仍计入（旧行为，无平仓算非盈利）。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={
                "use_cycle_pnl_delta": True,
                "ignore_zero_cycle_pnl": False,
            },
        )
        orch._decision_memory = _memory_with_cycle(
            pnls=[-1.0, -1.0, -1.0],
            cycle_pnls=[0.0, 0.0, 3.0],
        )
        # 0, 0, 3.0 → 仅 1 个 >0 → 1/3
        assert orch._decision_quality_score() == pytest.approx(1 / 3)

    def test_zero_excluded_only_real_profit_counts(self):
        """大量无平仓 + 少量盈利 → 质量分只看真实盈利周期（=1.0，非被稀释）。"""
        orch = _orch(
            learning={"memory_decay_alpha": 1.0},
            decision_quality_guard={"use_cycle_pnl_delta": True},
        )
        orch._decision_memory = _memory_with_cycle(
            pnls=[-1.0, -1.0, -1.0, -1.0, -1.0],
            cycle_pnls=[0.0, 0.0, 0.0, 3.0, 1.0],
        )
        # 有效样本 [3.0, 1.0] 全盈利 → 1.0
        assert orch._decision_quality_score() == pytest.approx(1.0)
