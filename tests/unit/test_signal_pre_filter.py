"""SignalPreFilterChain 企业级多层信号前置过滤 — 正式单元测试。

覆盖矩阵（聚焦「强化企业级多层信号前置过滤模块」改动）：
  P0  过滤链构造与优先级排序
  P0  统一结果口径 FilterDecision / FilterChainResult（to_dict / reduced_quantity）
  P0  12 个过滤器各自命中逻辑（blacklist / strategy_pause / confidence / regime /
      trend_alignment / consecutive_loss / frequency / hour_risk / volatility /
      min_profit / wear_type / multi_timeframe）
  P0  evaluate 短路与 breakdown 可解释报告
  P0  惰性求值 _resolve（黑名单命中时不触发 wear_type / mtf callable）
  P0  过滤器异常不中断链路（按放行不误杀）
  P0  config 禁用过滤器
  P1  IntelligentTradingAgent 桥接（set_signal_pre_filter_chain / _audit_via_chain）
  P1  FilterChainResult → AgentDecision 映射（reject / approve / reduce / delay）
  P1  链异常回退 legacy audit_signal
  P2  空 state / 空 context / 未知 action 边界
"""

import pytest

from core.signal_pre_filter import (
    SignalPreFilterChain,
    SignalContext,
    FilterDecision,
    FilterChainResult,
    BaseSignalFilter,
    BlacklistFilter,
    StrategyPauseFilter,
    ConfidenceFilter,
    RegimeCompatibilityFilter,
    TrendAlignmentFilter,
    ConsecutiveLossFilter,
    FrequencyFilter,
    HourRiskFilter,
    VolatilityFilter,
    MinProfitFilter,
    WearTypeFilter,
    MultiTimeframeFilter,
    ACTION_PASS,
    ACTION_REJECT,
    ACTION_REDUCE,
    ACTION_DELAY,
    _resolve,
)
from core.intelligent_agent import IntelligentTradingAgent, AgentDecision, DecisionLevel


# ═══════════════════════════════════════════════════════════════
# 辅助
# ═══════════════════════════════════════════════════════════════

def _ctx(**overrides) -> SignalContext:
    base = dict(
        symbol="BTC-USDT-SWAP",
        strategy_name="trend",
        signal_type="entry",
        direction="long",
        price=100.0,
        quantity=0.01,
        confidence=0.9,
        is_close=False,
    )
    base.update(overrides)
    return SignalContext(**base)


def _state(**overrides) -> dict:
    base = {
        "blacklist": {"hit": False, "reason": ""},
        "strategy_paused": {"hit": False, "reason": ""},
        "adaptive_threshold": 0.0,
        "regime_ok": True,
        "regime_reason": "OK",
        "trend_alignment": {"opposing": False, "regime_str": None, "strength": 0.0},
        "consecutive_losses": 0,
        "dynamic_threshold": 5,
        "trades_last_hour": 0,
        "max_trades_per_hour": 10,
        "hour_risk_level": "low",
        "high_volatility": False,
        "min_profit_ok": True,
        "wear_type": {"hit": False, "reason": ""},
        "mtf": {"ok": True, "reason": ""},
    }
    base.update(overrides)
    return base


class _RejectFilter(BaseSignalFilter):
    """测试用：恒驳回过滤器。"""
    name = "_test_reject"
    priority = 0
    source = "_test_reject_source"

    def check(self, ctx, state):
        return FilterDecision(action=ACTION_REJECT, source=self.source, reason="test reject")


class _RaiseFilter(BaseSignalFilter):
    """测试用：恒抛异常过滤器。"""
    name = "_test_raise"
    priority = 0
    source = "_test_raise_source"

    def check(self, ctx, state):
        raise RuntimeError("boom")


# ═══════════════════════════════════════════════════════════════
# P0: 构造与优先级排序
# ═══════════════════════════════════════════════════════════════

class TestChainConstruction:
    def test_default_filters_loaded(self):
        chain = SignalPreFilterChain()
        names = [f.name for f in chain.filters]
        assert names[0] == "blacklist"
        assert names[-1] == "multi_timeframe"
        assert len(names) == 13

    def test_filters_sorted_by_priority(self):
        chain = SignalPreFilterChain()
        priorities = [f.priority for f in chain.filters]
        assert priorities == sorted(priorities)

    def test_custom_filters_preserved(self):
        chain = SignalPreFilterChain(filters=[_RejectFilter()])
        assert [f.name for f in chain.filters] == ["_test_reject"]

    def test_empty_filters_always_pass(self):
        chain = SignalPreFilterChain(filters=[])
        result = chain.evaluate(_ctx(), _state())
        assert result.passed is True

    def test_disabled_config_disables_filter(self):
        chain = SignalPreFilterChain(config={"signal_pre_filter": {"disabled": ["blacklist"]}})
        blacklist = [f for f in chain.filters if f.name == "blacklist"][0]
        assert blacklist.enabled is False
        # 黑名单命中但被禁用 → 放行
        state = _state(blacklist={"hit": True, "reason": "x"})
        result = chain.evaluate(_ctx(), state)
        assert result.passed is True


# ═══════════════════════════════════════════════════════════════
# P0: 统一结果口径
# ═══════════════════════════════════════════════════════════════

class TestResultContracts:
    def test_filter_decision_to_dict(self):
        d = FilterDecision(action=ACTION_REJECT, source="audit_blacklist", reason="r", confidence=0.9)
        assert d.to_dict()["action"] == ACTION_REJECT
        assert d.to_dict()["source"] == "audit_blacklist"

    def test_chain_result_to_dict(self):
        r = FilterChainResult(passed=True, action=ACTION_PASS, blocked_source=None)
        d = r.to_dict()
        assert d["passed"] is True
        assert d["decision"] is None

    def test_reduced_quantity_from_reduce_decision(self):
        d = FilterDecision(action=ACTION_REDUCE, source="s", reason="r", details={"reduced_quantity": 0.004})
        r = FilterChainResult(passed=False, action=ACTION_REDUCE, blocked_source="s", decision=d)
        assert r.reduced_quantity == 0.004

    def test_reduced_quantity_none_for_reject(self):
        d = FilterDecision(action=ACTION_REJECT, source="s", reason="r")
        r = FilterChainResult(passed=False, action=ACTION_REJECT, blocked_source="s", decision=d)
        assert r.reduced_quantity is None

    def test_reduced_quantity_none_when_no_decision(self):
        r = FilterChainResult(passed=True, action=ACTION_PASS, blocked_source=None)
        assert r.reduced_quantity is None


# ═══════════════════════════════════════════════════════════════
# P0: 各过滤器命中逻辑
# ═══════════════════════════════════════════════════════════════

class TestIndividualFilters:
    def test_blacklist_hit(self):
        chain = SignalPreFilterChain(filters=[BlacklistFilter()])
        r = chain.evaluate(_ctx(), _state(blacklist={"hit": True, "reason": "累计亏损"}))
        assert r.action == ACTION_REJECT
        assert r.blocked_source == "audit_blacklist"

    def test_strategy_pause_hit(self):
        chain = SignalPreFilterChain(filters=[StrategyPauseFilter()])
        r = chain.evaluate(_ctx(), _state(strategy_paused={"hit": True, "reason": "5连亏"}))
        assert r.action == ACTION_REJECT
        assert r.blocked_source == "audit_strategy_pause"

    def test_strategy_pause_skips_close(self):
        chain = SignalPreFilterChain(filters=[StrategyPauseFilter()])
        r = chain.evaluate(_ctx(is_close=True), _state(strategy_paused={"hit": True}))
        assert r.passed is True

    def test_confidence_reject(self):
        chain = SignalPreFilterChain(filters=[ConfidenceFilter()])
        r = chain.evaluate(_ctx(confidence=0.3), _state(adaptive_threshold=0.5))
        assert r.action == ACTION_REJECT
        assert r.blocked_source == "audit_confidence"

    def test_confidence_pass(self):
        chain = SignalPreFilterChain(filters=[ConfidenceFilter()])
        r = chain.evaluate(_ctx(confidence=0.6), _state(adaptive_threshold=0.5))
        assert r.passed is True

    def test_regime_reject(self):
        chain = SignalPreFilterChain(filters=[RegimeCompatibilityFilter()])
        r = chain.evaluate(_ctx(), _state(regime_ok=False, regime_reason="网格不适配强趋势"))
        assert r.action == ACTION_REJECT
        assert r.blocked_source == "audit_regime"

    def test_regime_skips_close(self):
        chain = SignalPreFilterChain(filters=[RegimeCompatibilityFilter()])
        r = chain.evaluate(_ctx(is_close=True), _state(regime_ok=False))
        assert r.passed is True

    def test_trend_alignment_reduce(self):
        chain = SignalPreFilterChain(filters=[TrendAlignmentFilter()])
        state = _state(trend_alignment={"opposing": True, "regime_str": "trend_bearish", "strength": 0.6})
        r = chain.evaluate(_ctx(quantity=0.01), state)
        assert r.action == ACTION_REDUCE
        assert r.blocked_source == "audit_trend_alignment"
        assert r.reduced_quantity == pytest.approx(0.006)

    def test_trend_alignment_weak_ignored(self):
        chain = SignalPreFilterChain(filters=[TrendAlignmentFilter()])
        state = _state(trend_alignment={"opposing": True, "regime_str": "trend_bearish", "strength": 0.2})
        r = chain.evaluate(_ctx(), state)
        assert r.passed is True

    def test_consecutive_loss_delay(self):
        chain = SignalPreFilterChain(filters=[ConsecutiveLossFilter()])
        r = chain.evaluate(_ctx(), _state(consecutive_losses=5, dynamic_threshold=5))
        assert r.action == ACTION_DELAY
        assert r.blocked_source == "audit_performance"

    def test_frequency_delay(self):
        chain = SignalPreFilterChain(filters=[FrequencyFilter()])
        r = chain.evaluate(_ctx(), _state(trades_last_hour=11, max_trades_per_hour=10))
        assert r.action == ACTION_DELAY
        assert r.blocked_source == "audit_frequency"

    def test_hour_risk_high_reduce(self):
        chain = SignalPreFilterChain(filters=[HourRiskFilter()])
        r = chain.evaluate(_ctx(confidence=0.9, quantity=0.01), _state(hour_risk_level="high", adaptive_threshold=0.45))
        assert r.action == ACTION_REDUCE
        assert r.reduced_quantity == pytest.approx(0.004)

    def test_hour_risk_high_reject_low_confidence(self):
        chain = SignalPreFilterChain(filters=[HourRiskFilter()])
        r = chain.evaluate(_ctx(confidence=0.5), _state(hour_risk_level="high", adaptive_threshold=0.45))
        assert r.action == ACTION_REJECT

    def test_hour_risk_medium_reduce(self):
        chain = SignalPreFilterChain(filters=[HourRiskFilter()])
        r = chain.evaluate(_ctx(quantity=0.01), _state(hour_risk_level="medium"))
        assert r.action == ACTION_REDUCE
        assert r.reduced_quantity == pytest.approx(0.007)

    def test_volatility_reduce(self):
        chain = SignalPreFilterChain(filters=[VolatilityFilter()])
        r = chain.evaluate(_ctx(quantity=0.01), _state(high_volatility=True))
        assert r.action == ACTION_REDUCE
        assert r.reduced_quantity == pytest.approx(0.005)

    def test_min_profit_reject(self):
        chain = SignalPreFilterChain(filters=[MinProfitFilter()])
        r = chain.evaluate(_ctx(price=100, quantity=0.01), _state(min_profit_ok=False))
        assert r.action == ACTION_REJECT
        assert r.blocked_source == "audit_min_profit"

    def test_wear_type_reject_lazy(self):
        chain = SignalPreFilterChain(filters=[WearTypeFilter()])
        r = chain.evaluate(_ctx(), _state(wear_type={"hit": True, "reason": "磨损"}))
        assert r.action == ACTION_REJECT
        assert r.blocked_source == "audit_wear_type"

    def test_wear_type_callable(self):
        chain = SignalPreFilterChain(filters=[WearTypeFilter()])
        calls = []
        state = _state(wear_type=lambda: calls.append(1) or {"hit": True, "reason": "磨损"})
        r = chain.evaluate(_ctx(), state)
        assert r.action == ACTION_REJECT
        assert len(calls) == 1

    def test_multi_timeframe_reject_lazy(self):
        chain = SignalPreFilterChain(filters=[MultiTimeframeFilter()])
        r = chain.evaluate(_ctx(), _state(mtf={"ok": False, "reason": "1H未确认"}))
        assert r.action == ACTION_REJECT
        assert r.blocked_source == "audit_multi_timeframe"


# ═══════════════════════════════════════════════════════════════
# P0: evaluate 短路 + breakdown
# ═══════════════════════════════════════════════════════════════

class TestEvaluateShortCircuit:
    def test_blacklist_short_circuits_before_confidence(self):
        chain = SignalPreFilterChain()
        state = _state(
            blacklist={"hit": True, "reason": "x"},
            adaptive_threshold=0.5,
        )
        r = chain.evaluate(_ctx(confidence=0.0), state)
        # 黑名单优先级更高，即使置信度极低也先命中黑名单
        assert r.blocked_source == "audit_blacklist"
        assert r.breakdown[-1]["name"] == "blacklist"

    def test_breakdown_records_all_passed(self):
        chain = SignalPreFilterChain()
        r = chain.evaluate(_ctx(), _state())
        assert r.passed is True
        assert len(r.breakdown) == 13
        assert all(b["action"] == ACTION_PASS for b in r.breakdown)
        assert all(b["hit"] is False for b in r.breakdown)

    def test_breakdown_truncated_on_hit(self):
        chain = SignalPreFilterChain()
        state = _state(confidence=0.1, adaptive_threshold=0.5)
        r = chain.evaluate(_ctx(confidence=0.1), state)
        # 命中 confidence（priority 2）即短路，后续 10 层不再执行
        assert len(r.breakdown) == 3
        assert r.breakdown[-1]["hit"] is True

    def test_breakdown_fields_complete(self):
        chain = SignalPreFilterChain(filters=[ConfidenceFilter()])
        r = chain.evaluate(_ctx(confidence=0.0), _state(adaptive_threshold=0.5))
        entry = r.breakdown[-1]
        for key in ("name", "source", "priority", "action", "reason", "elapsed_ms", "hit"):
            assert key in entry

    def test_stats_accumulate(self):
        chain = SignalPreFilterChain(filters=[ConfidenceFilter()])
        chain.evaluate(_ctx(confidence=0.0), _state(adaptive_threshold=0.5))
        chain.evaluate(_ctx(confidence=0.0), _state(adaptive_threshold=0.5))
        assert chain.stats().get("confidence") == 2

    def test_reset_stats(self):
        chain = SignalPreFilterChain(filters=[ConfidenceFilter()])
        chain.evaluate(_ctx(confidence=0.0), _state(adaptive_threshold=0.5))
        chain.reset_stats()
        assert chain.stats() == {}


# ═══════════════════════════════════════════════════════════════
# P0: 惰性求值 + 异常兜底
# ═══════════════════════════════════════════════════════════════

class TestLazyAndResilience:
    def test_blacklist_hit_skips_wear_type_and_mtf(self):
        chain = SignalPreFilterChain()
        wear_calls = []
        mtf_calls = []
        state = _state(
            blacklist={"hit": True, "reason": "x"},
            wear_type=lambda: wear_calls.append(1) or {"hit": False},
            mtf=lambda: mtf_calls.append(1) or {"ok": True},
        )
        chain.evaluate(_ctx(), state)
        assert wear_calls == []
        assert mtf_calls == []

    def test_filter_exception_does_not_break_chain(self):
        chain = SignalPreFilterChain(filters=[_RaiseFilter(), ConfidenceFilter()])
        # _RaiseFilter 抛异常 → 放行，继续 ConfidenceFilter
        r = chain.evaluate(_ctx(confidence=0.0), _state(adaptive_threshold=0.5))
        assert r.action == ACTION_REJECT
        assert r.blocked_source == "audit_confidence"

    def test_exception_only_filter_passes(self):
        chain = SignalPreFilterChain(filters=[_RaiseFilter()])
        r = chain.evaluate(_ctx(), _state())
        assert r.passed is True

    def test_resolve_plain_value(self):
        assert _resolve(42) == 42
        assert _resolve(None) is None

    def test_resolve_callable(self):
        assert _resolve(lambda: "x") == "x"


# ═══════════════════════════════════════════════════════════════
# P1: IntelligentTradingAgent 桥接
# ═══════════════════════════════════════════════════════════════

class TestAgentBridge:
    def _make_agent(self, tmp_path) -> IntelligentTradingAgent:
        return IntelligentTradingAgent({
            "data_dir": str(tmp_path),
            "total_capital": 1000.0,
        })

    def test_set_signal_pre_filter_chain(self, tmp_path):
        agent = self._make_agent(tmp_path)
        chain = SignalPreFilterChain()
        agent.set_signal_pre_filter_chain(chain)
        assert agent._signal_pre_filter_chain is chain

    def test_audit_signal_routes_through_chain_pass(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.set_signal_pre_filter_chain(SignalPreFilterChain(filters=[]))
        decision = agent.audit_signal(
            "BTC-USDT-SWAP", "trend", "entry", "long", 100.0, 0.01, 0.9, False
        )
        assert isinstance(decision, AgentDecision)
        assert decision.action == "approve"
        assert decision.source == "audit_pass"
        assert "chain_breakdown" in decision.details

    def test_audit_signal_routes_through_chain_reject(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.set_signal_pre_filter_chain(SignalPreFilterChain(filters=[_RejectFilter()]))
        decision = agent.audit_signal(
            "BTC-USDT-SWAP", "trend", "entry", "long", 100.0, 0.01, 0.9, False
        )
        assert decision.action == "reject"
        assert decision.source == "_test_reject_source"
        assert decision.level == DecisionLevel.SYMBOL

    def test_chain_reject_maps_level(self, tmp_path):
        agent = self._make_agent(tmp_path)
        # 自定义过滤器 level="global"
        class _GlobalRejectFilter(BaseSignalFilter):
            name = "_global_reject"
            priority = 0
            source = "_global_source"
            def check(self, ctx, state):
                return FilterDecision(action=ACTION_REJECT, source=self.source, reason="r", level="global")

        agent.set_signal_pre_filter_chain(SignalPreFilterChain(filters=[_GlobalRejectFilter()]))
        decision = agent.audit_signal(
            "BTC-USDT-SWAP", "trend", "entry", "long", 100.0, 0.01, 0.9, False
        )
        assert decision.action == "reject"
        assert decision.level == DecisionLevel.GLOBAL

    def test_chain_reduce_maps_action(self, tmp_path):
        agent = self._make_agent(tmp_path)
        class _ReduceFilter(BaseSignalFilter):
            name = "_reduce"
            priority = 0
            source = "_reduce_source"
            def check(self, ctx, state):
                return FilterDecision(action=ACTION_REDUCE, source=self.source, reason="r",
                                      details={"reduced_quantity": 0.004})

        agent.set_signal_pre_filter_chain(SignalPreFilterChain(filters=[_ReduceFilter()]))
        decision = agent.audit_signal(
            "BTC-USDT-SWAP", "trend", "entry", "long", 100.0, 0.01, 0.9, False
        )
        assert decision.action == "reduce"
        assert decision.details.get("reduced_quantity") == pytest.approx(0.004)

    def test_chain_exception_falls_back_to_legacy(self, tmp_path):
        agent = self._make_agent(tmp_path)
        agent.set_signal_pre_filter_chain(SignalPreFilterChain(filters=[_RaiseFilter()]))
        # _RaiseFilter 抛异常 → 框架视为放行，仍正常返回 approve（不抛异常）
        decision = agent.audit_signal(
            "BTC-USDT-SWAP", "trend", "entry", "long", 100.0, 0.01, 0.9, False
        )
        assert isinstance(decision, AgentDecision)

    def test_chain_evaluate_raises_falls_back(self, tmp_path):
        agent = self._make_agent(tmp_path)
        class _BoomChain:
            def evaluate(self, ctx, state):
                raise RuntimeError("framework down")
        agent.set_signal_pre_filter_chain(_BoomChain())
        # 框架异常 → audit_signal 捕获并回退 legacy，不向上抛异常
        decision = agent.audit_signal(
            "BTC-USDT-SWAP", "trend", "entry", "long", 100.0, 0.01, 0.9, False
        )
        assert isinstance(decision, AgentDecision)


# ═══════════════════════════════════════════════════════════════
# P2: 边界
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_state_all_pass(self):
        chain = SignalPreFilterChain()
        r = chain.evaluate(_ctx(), {})
        assert r.passed is True

    def test_none_state_all_pass(self):
        chain = SignalPreFilterChain()
        r = chain.evaluate(_ctx(), None)
        assert r.passed is True

    def test_close_signal_skips_open_only_filters(self):
        # 平仓信号跳过 regime/consecutive_loss/frequency/hour_risk/volatility/min_profit
        chain = SignalPreFilterChain()
        state = _state(
            regime_ok=False,
            consecutive_losses=10,
            trades_last_hour=100,
            hour_risk_level="high",
            high_volatility=True,
            min_profit_ok=False,
        )
        r = chain.evaluate(_ctx(is_close=True), state)
        assert r.passed is True

    def test_direction_not_long_short_skips_trend_mtf(self):
        chain = SignalPreFilterChain(filters=[TrendAlignmentFilter(), MultiTimeframeFilter()])
        state = _state(
            trend_alignment={"opposing": True, "regime_str": "x", "strength": 0.9},
            mtf={"ok": False, "reason": "x"},
        )
        r = chain.evaluate(_ctx(direction="neutral"), state)
        assert r.passed is True
