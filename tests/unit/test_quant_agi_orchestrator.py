"""
QuantAGIOrchestrator 单元测试
==============================
覆盖：全依赖缺失、fake 注入、冷却机制、fail-closed、JSON 安全。
"""
import json
from collections import deque
from types import SimpleNamespace
import asyncio

import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator, build_strategy_metrics


def _make_snapshot():
    strat = SimpleNamespace(
        total_pnl=50.0,
        win_rate=0.6,
        profit_factor=1.5,
        max_drawdown=0.05,
        total_trades=30,
        health_score=70.0,
        health_grade="B",
        lifecycle="mature",
        trend="improving",
        pnl_per_capital_pct=8.0,
        realized_pnl=40.0,
        unrealized_pnl=10.0,
    )
    return SimpleNamespace(
        available=True,
        total_pnl=50.0,
        total_trades=30,
        overall_health_score=70.0,
        strategies={"grid": strat},
    )


class FakeRegimeEngine:
    def get_regime(self):
        return {"regime": "range_bound", "strength": 0.4, "confidence": 0.6}


class FakeContributionAnalyzer:
    def analyze(self, window="24h"):
        return _make_snapshot()

    def get_capital_reallocation_suggestions(self, snapshot=None):
        return [
            {
                "strategy": "grid",
                "action": "increase",
                "target_allocation": 0.3,
                "reason": "A/B级健康度+改善趋势",
            }
        ]


class FakeCapitalAllocator:
    def __init__(self, equity=1000.0):
        self._equity = equity

    def get_equity(self):
        return self._equity

    def get_total_capital(self):
        return self._equity

    def get_strategy_allocation(self, name):
        return 0.25

    def get_summary(self):
        return {"equity": self._equity}


class FakeDynamicAllocator:
    async def compute_allocation_plan(self, total_capital, total_equity, strategy_names,
                                      strategy_metrics=None, market_regime=None,
                                      current_weights=None, used_margin_by_strategy=None,
                                      persist_last_plan=True):
        return {
            "timestamp": "2026-09-17T00:00:00",
            "total_capital": total_capital,
            "total_equity": total_equity,
            "strategy_allocations": {n: {"target_weight": 0.25} for n in (strategy_names or [])},
            "auto_actions": ["AUTO_SWEEP: 10 USDT base→addon"],
            "recommendations": ["Deploy idle cash"],
            "warnings": [],
        }

    def get_strategy_allocations(self):
        return {"grid": {"target_weight": 0.25, "used_margin": 100.0}}

    def check_consecutive_streaks(self, metrics):
        return {"grid": {"action": "hold", "consecutive_losses": 0}}


def _assert_json_safe(report):
    # allow_nan=False：若存在 NaN/Inf 会抛 ValueError
    text = json.dumps(report, allow_nan=False)
    assert "NaN" not in text
    assert "Infinity" not in text
    return text


def _full_orchestrator(tmp_path, capital_equity=1000.0, cooldown=3600):
    return QuantAGIOrchestrator(
        config={
            "agi_orchestrator": {
                "state_path": str(tmp_path / "state.json"),
                "cooldown_seconds": cooldown,
            }
        },
        regime_engine=FakeRegimeEngine(),
        contribution_analyzer=FakeContributionAnalyzer(),
        capital_allocator=FakeCapitalAllocator(equity=capital_equity),
        dynamic_allocator=FakeDynamicAllocator(),
    )


async def test_all_dependencies_none_returns_json_safe_report(tmp_path):
    orch = QuantAGIOrchestrator(
        config={"agi_orchestrator": {"state_path": str(tmp_path / "state.json")}}
    )
    report = await orch.run_cycle()
    assert report["status"] == "fail_closed"
    assert report["actions"] == []
    assert report["decision"]["allocation_plan"]["warnings"]
    _assert_json_safe(report)


async def test_with_fakes_produces_report_and_json_safe(tmp_path):
    orch = _full_orchestrator(tmp_path)
    report = await orch.run_cycle()

    assert report["status"] == "ok"
    assert report["cooldown"] is False
    assert report["decision"]["allocation_plan"] is not None
    assert report["decision"]["allocation_plan"]["strategy_allocations"]
    assert report["decision"]["reallocation_suggestions"]
    assert report["actions"]
    assert report["perception"]["equity"] == 1000.0
    _assert_json_safe(report)


async def test_cooldown_returns_cached_report(tmp_path):
    orch = _full_orchestrator(tmp_path, cooldown=3600)

    first = await orch.run_cycle()
    assert first["cooldown"] is False

    second = await orch.run_cycle()
    assert second["cooldown"] is True
    assert second["status"] == "cooldown"
    assert second["cycle"] == first["cycle"]
    _assert_json_safe(second)


async def test_fail_closed_on_zero_equity(tmp_path):
    orch = _full_orchestrator(tmp_path, capital_equity=0.0)
    report = await orch.run_cycle()
    assert report["status"] == "fail_closed"
    assert report["actions"] == []
    assert report["decision"]["allocation_plan"]["warnings"]
    _assert_json_safe(report)


async def test_fail_closed_on_nan_equity(tmp_path):
    orch = _full_orchestrator(tmp_path, capital_equity=float("nan"))
    report = await orch.run_cycle()
    assert report["status"] == "fail_closed"
    _assert_json_safe(report)


async def test_fail_closed_on_capital_exception(tmp_path):
    class BoomCapital:
        def get_equity(self):
            raise RuntimeError("boom")

        def get_total_capital(self):
            return 1000.0

    orch = QuantAGIOrchestrator(
        config={"agi_orchestrator": {"state_path": str(tmp_path / "state.json")}},
        regime_engine=FakeRegimeEngine(),
        contribution_analyzer=FakeContributionAnalyzer(),
        capital_allocator=BoomCapital(),
        dynamic_allocator=FakeDynamicAllocator(),
    )
    report = await orch.run_cycle()
    assert report["status"] == "fail_closed"
    _assert_json_safe(report)


async def test_degraded_when_partial_dependencies(tmp_path):
    # 仅注入资本与分配器，缺 regime/contribution → 权益可确认，应降级而非崩溃
    orch = QuantAGIOrchestrator(
        config={"agi_orchestrator": {"state_path": str(tmp_path / "state.json")}},
        capital_allocator=FakeCapitalAllocator(equity=1000.0),
        dynamic_allocator=FakeDynamicAllocator(),
    )
    report = await orch.run_cycle()
    assert report["status"] == "degraded"
    assert report["perception"]["equity"] == 1000.0
    _assert_json_safe(report)


# ── 低风险闲置资金自动归集（P0-6） ────────────────────────

def _decision(idle_cash, total_equity, suggestions):
    return {
        "allocation_plan": {"idle_cash": idle_cash, "total_equity": total_equity},
        "reallocation_suggestions": suggestions,
    }


def test_idle_cash_deploy_generated_for_core_strategy():
    orch = QuantAGIOrchestrator(config={})
    decision = _decision(50.0, 100.0, [
        {"strategy": "grid", "health_grade": "B", "target_allocation": 0.3},
    ])
    actions = orch._idle_cash_deploy_actions(decision)
    assert len(actions) == 1
    assert actions[0]["type"] == "idle_cash_deploy"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["health_grade"] == "B"


def test_idle_cash_deploy_no_core_strategy():
    orch = QuantAGIOrchestrator(config={})
    decision = _decision(50.0, 100.0, [
        {"strategy": "scalping", "health_grade": "F"},
        {"strategy": "sync", "health_grade": "C"},
    ])
    assert orch._idle_cash_deploy_actions(decision) == []


def test_idle_cash_deploy_below_threshold():
    orch = QuantAGIOrchestrator(config={})  # 默认 idle_deploy_threshold=0.10
    decision = _decision(5.0, 100.0, [  # 5% 闲置 < 10%
        {"strategy": "grid", "health_grade": "B"},
    ])
    assert orch._idle_cash_deploy_actions(decision) == []


def test_idle_cash_deploy_positive_return_fallback():
    """无 A/B 级时，资金池内正收益策略（profit_factor>1 且 total_pnl>0）兜底归集，打破死锁。"""
    orch = QuantAGIOrchestrator(config={})
    decision = {
        "allocation_plan": {"idle_cash": 50.0, "total_equity": 100.0},
        "reallocation_suggestions": [
            {"strategy": "grid", "health_grade": "D", "target_allocation": 0.2},
        ],
        "strategy_metrics": {
            "grid": {"profit_factor": 1.5, "total_pnl": 3.0},
        },
    }
    actions = orch._idle_cash_deploy_actions(decision)
    assert len(actions) == 1
    assert actions[0]["type"] == "idle_cash_deploy"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["health_grade"] == "positive"


def test_idle_cash_deploy_positive_return_excludes_non_pool_sync():
    """正收益但非资金池标签（sync）不应被归集，避免给占位标签部署资金。"""
    orch = QuantAGIOrchestrator(config={})
    decision = {
        "allocation_plan": {"idle_cash": 50.0, "total_equity": 100.0},
        "reallocation_suggestions": [
            {"strategy": "grid", "health_grade": "D", "target_allocation": 0.2},
        ],
        "strategy_metrics": {
            "sync": {"profit_factor": 2.28, "total_pnl": 0.46},  # 非资金池标签
            "grid": {"profit_factor": 0.43, "total_pnl": -9.05},  # 亏损
        },
    }
    assert orch._idle_cash_deploy_actions(decision) == []


# ── P2-7 负期望早期熔断信号贯通 ────────────────────────────

def test_build_strategy_metrics_includes_consecutive_losses():
    contrib = {"strategies": {"grid": {
        "pnl_per_capital_pct": -3.0, "max_drawdown": 0.1,
        "consecutive_losses": 7, "total_trades": 10, "total_pnl": -1.0,
        "win_rate": 0.0, "profit_factor": 0.5,
    }}}
    metrics = build_strategy_metrics(contrib)
    assert metrics["grid"]["consecutive_losses"] == 7


# ── P2-8 冻结策略观察期自动解冻 / 缩量试探诊断 ─────────────

def test_diagnose_freeze_state_alerts(tmp_path):
    orch = QuantAGIOrchestrator(
        config={"agi_orchestrator": {"state_path": str(tmp_path / "state.json")}}
    )
    perception = {
        "equity": 1000.0,
        "used_margin": 0.0,
        "contribution": None,
        "market_regime": None,
        "freeze_state": {
            "scalping": {
                "reason": "consecutive_losses", "probing": False,
                "probe_attempts": 0, "probe_max_attempts": 3,
                "observe_seconds": 3600, "remaining_seconds": 1800,
                "permanently_frozen": False,
            },
            "sync": {
                "reason": "high_drawdown", "probing": True,
                "probe_attempts": 2, "probe_max_attempts": 3,
                "observe_seconds": 3600, "remaining_seconds": 500,
                "permanently_frozen": False,
            },
            "grid": {
                "reason": "consecutive_losses", "probing": True,
                "probe_attempts": 3, "probe_max_attempts": 3,
                "observe_seconds": 3600, "remaining_seconds": 0,
                "permanently_frozen": True,
            },
        },
    }
    alerts = orch._diagnose(perception)
    by_type = {}
    for a in alerts:
        by_type.setdefault(a["type"], []).append(a.get("strategy"))
    assert by_type.get("strategy_frozen_observing") == ["scalping"]
    assert by_type.get("strategy_probing") == ["sync"]
    assert by_type.get("strategy_permanently_frozen") == ["grid"]


# ── 账户级收益检测自动平仓（profit_take）────────────────────

def _profit_take_orch(**overrides):
    cfg = {
        "agi_orchestrator": {
            "profit_take": {
                "enabled": True,
                "activation_pct": 0.03,
                "max_pct": 0.10,
                "max_close_ratio": 0.5,
            },
        },
    }
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["profit_take"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _perception(upl, equity=1000.0):
    return {
        "equity": equity,
        "unrealized_pnl": upl,
        "used_margin": 0.0,
        "contribution": None,
        "market_regime": None,
        "freeze_state": {},
    }


def test_profit_take_signal_generated_and_acted():
    orch = _profit_take_orch()
    alerts = orch._diagnose(_perception(50.0))  # 浮盈 5% >= 3%
    signals = [a for a in alerts if a["type"] == "profit_take_signal"]
    assert len(signals) == 1
    sig = signals[0]
    # intensity=(0.05-0.03)/(0.10-0.03)=0.2857, close_ratio=0.5*0.2857
    assert sig["close_ratio"] == pytest.approx(0.5 * (0.05 - 0.03) / (0.10 - 0.03))

    actions = orch._act({"allocation_plan": {}, "reallocation_suggestions": []}, alerts)
    closes = [a for a in actions if a["type"] == "profit_take_close"]
    assert len(closes) == 1
    assert closes[0]["close_ratio"] == pytest.approx(sig["close_ratio"])


def test_profit_take_below_activation_no_signal():
    orch = _profit_take_orch()
    alerts = orch._diagnose(_perception(10.0))  # 浮盈 1% < 3%
    assert not [a for a in alerts if a["type"] == "profit_take_signal"]


def test_profit_take_negative_upl_no_signal():
    orch = _profit_take_orch()
    alerts = orch._diagnose(_perception(-50.0))  # 浮亏，不落袋
    assert not [a for a in alerts if a["type"] == "profit_take_signal"]


def test_profit_take_disabled_no_signal():
    orch = _profit_take_orch(enabled=False)
    alerts = orch._diagnose(_perception(50.0))
    assert not [a for a in alerts if a["type"] == "profit_take_signal"]


def test_profit_take_max_close_ratio_capped():
    orch = _profit_take_orch()
    alerts = orch._diagnose(_perception(300.0))  # 浮盈 30% >> max_pct 10%
    signals = [a for a in alerts if a["type"] == "profit_take_signal"]
    assert len(signals) == 1
    assert signals[0]["close_ratio"] == pytest.approx(0.5)  # 封顶 max_close_ratio


# ── 主动风控响应（risk_response）───────────────────────────

def _risk_response_orch(**overrides):
    cfg = {"agi_orchestrator": {"risk_response": {"enabled": True, "reduce_target": 0.1}}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["risk_response"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_risk_reduce_actions_generate_decrease():
    orch = _risk_response_orch()
    alerts = [
        {"type": "strategy_health_critical", "strategy": "grid", "message": "健康度F"},
        {"type": "consecutive_losses", "strategy": "scalping", "message": "连续亏损"},
    ]
    actions = orch._risk_reduce_actions(alerts)
    assert len(actions) == 2
    # 健康度 F → 减配到 0
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == 0.0
    assert actions[0]["action"] == "decrease"
    # 连续亏损 → 减配到 reduce_target
    assert actions[1]["strategy"] == "scalping"
    assert actions[1]["target_allocation"] == pytest.approx(0.1)


def test_risk_reduce_actions_dedup_per_strategy():
    orch = _risk_response_orch()
    alerts = [
        {"type": "strategy_health_critical", "strategy": "grid", "message": "F"},
        {"type": "consecutive_losses", "strategy": "grid", "message": "连续亏损"},
    ]
    actions = orch._risk_reduce_actions(alerts)
    assert len(actions) == 1


def test_risk_reduce_actions_ignores_non_risk_alerts():
    orch = _risk_response_orch()
    alerts = [
        {"type": "strategy_dormant", "strategy": "grid", "message": "休眠"},
        {"type": "regime_shift", "strategy": None, "message": "突变"},
    ]
    assert orch._risk_reduce_actions(alerts) == []


def test_risk_reduce_actions_disabled():
    orch = _risk_response_orch(enabled=False)
    alerts = [{"type": "strategy_health_critical", "strategy": "grid", "message": "F"}]
    assert orch._risk_reduce_actions(alerts) == []


# ── 市场状态自适应（regime_adaptive）───────────────────────

def test_effective_profit_take_activation_regime():
    orch = QuantAGIOrchestrator(config={"agi_orchestrator": {
        "profit_take": {"enabled": True, "activation_pct": 0.03, "max_pct": 0.10,
                        "max_close_ratio": 0.5},
        "regime_adaptive": {"enabled": True, "trend_multiplier": 1.3, "range_multiplier": 0.8},
    }})
    assert orch._effective_profit_take_activation(None) == pytest.approx(0.03)
    assert orch._effective_profit_take_activation("trend_bullish") == pytest.approx(0.03 * 1.3)
    assert orch._effective_profit_take_activation("trend_bearish") == pytest.approx(0.03 * 1.3)
    assert orch._effective_profit_take_activation("range_bound") == pytest.approx(0.03 * 0.8)


def test_effective_profit_take_activation_regime_disabled():
    orch = QuantAGIOrchestrator(config={"agi_orchestrator": {
        "profit_take": {"enabled": True, "activation_pct": 0.03, "max_pct": 0.10,
                        "max_close_ratio": 0.5},
        "regime_adaptive": {"enabled": False, "trend_multiplier": 1.3, "range_multiplier": 0.8},
    }})
    # regime_adaptive 关闭时，regime 不改变激活阈值
    assert orch._effective_profit_take_activation("trend_bullish") == pytest.approx(0.03)
    assert orch._effective_profit_take_activation("range_bound") == pytest.approx(0.03)


# ── 跨周期学习记忆（learning）─────────────────────────────

def _learning_orch(state_path=None, **overrides):
    cfg = {"agi_orchestrator": {"learning": {"enabled": True, "memory_size": 10,
                                             "max_adjust_pct": 0.4}}}
    if state_path:
        cfg["agi_orchestrator"]["state_path"] = state_path
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["learning"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_learning_adjust_insufficient_sample(tmp_path):
    orch = _learning_orch(state_path=str(tmp_path / "state.json"))
    assert orch._learning_profit_take_adjust() == 1.0  # 样本不足，不调整


def test_learning_adjust_improving_widens(tmp_path):
    orch = _learning_orch(state_path=str(tmp_path / "state.json"))
    orch._decision_memory = deque([
        {"health_score": 60.0}, {"health_score": 70.0}, {"health_score": 80.0},
    ], maxlen=10)
    assert orch._learning_profit_take_adjust() > 1.0  # 改善 → 放宽落袋


def test_learning_adjust_declining_tightens(tmp_path):
    orch = _learning_orch(state_path=str(tmp_path / "state.json"))
    orch._decision_memory = deque([
        {"health_score": 80.0}, {"health_score": 70.0}, {"health_score": 60.0},
    ], maxlen=10)
    assert orch._learning_profit_take_adjust() < 1.0  # 恶化 → 收紧落袋


def test_learning_adjust_stable_returns_one(tmp_path):
    orch = _learning_orch(state_path=str(tmp_path / "state.json"))
    orch._decision_memory = deque([
        {"health_score": 70.0}, {"health_score": 70.5}, {"health_score": 70.2},
    ], maxlen=10)
    assert orch._learning_profit_take_adjust() == pytest.approx(1.0)


# ── 目标导向规划（goal_planning）──────────────────────────

def _goal_planning_orch(**overrides):
    cfg = {"agi_orchestrator": {"goal_planning": {
        "enabled": True,
        "daily_target_return_pct": 0.02,
        "max_drawdown_pct": 0.08,
        "near_drawdown_ratio": 0.8,
        "goal_reached_reduce_target": 0.3,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["goal_planning"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_goal_planning_reached_alert_and_action():
    orch = _goal_planning_orch()
    # 初始资本 1000，权益 1030 → 收益 3% >= 2%
    perception = _perception(0.0, equity=1030.0)
    perception["total_capital"] = 1000.0
    alerts = orch._diagnose(perception)
    reached = [a for a in alerts if a["type"] == "goal_reached"]
    assert len(reached) == 1
    assert reached[0]["return_pct"] == pytest.approx(0.03)

    actions = orch._act({"allocation_plan": {}, "strategy_names": ["grid"]}, alerts)
    closes = [a for a in actions if a["type"] == "profit_take_close"]
    assert len(closes) == 1
    assert closes[0]["close_ratio"] == pytest.approx(0.3)


def test_goal_planning_near_drawdown_defensive():
    orch = _goal_planning_orch()
    # 首次权益 1000（峰值 1000），随后权益 940 → 回撤 6% >= 8%*0.8=6.4%? 否。
    # 用权益 930 → 回撤 7% >= 6.4%
    orch._update_goal_planning_state(1000.0, 1000.0)
    perception = _perception(0.0, equity=930.0)
    perception["total_capital"] = 1000.0
    alerts = orch._diagnose(perception)
    near = [a for a in alerts if a["type"] == "near_drawdown_limit"]
    assert len(near) == 1

    actions = orch._act(
        {"allocation_plan": {}, "strategy_names": ["grid", "sniper"]}, alerts
    )
    realloc = [a for a in actions if a["type"] == "reallocate"]
    assert len(realloc) == 2
    assert all(a["action"] == "decrease" for a in realloc)


def test_goal_planning_disabled_no_alerts():
    orch = _goal_planning_orch(enabled=False)
    perception = _perception(0.0, equity=1030.0)
    perception["total_capital"] = 1000.0
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] in ("goal_reached", "near_drawdown_limit")]


# ── 策略参数自适应（param_adaptation）────────────────────

def _param_adjust_orch(**overrides):
    cfg = {"agi_orchestrator": {"param_adaptation": {
        "enabled": True, "health_f_leverage": 1.0, "declining_leverage": 2.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["param_adaptation"][k] = v
    orch = QuantAGIOrchestrator(config=cfg)
    # 重置跨周期冷却状态，避免持久化 state.json 污染测试基线
    orch._param_adapt_last_cycle = {}
    return orch


def test_param_adjust_actions_generate():
    orch = _param_adjust_orch()
    alerts = [
        {"type": "strategy_health_critical", "strategy": "grid", "message": "健康度F"},
        {"type": "strategy_declining", "strategy": "scalping", "message": "趋势恶化"},
        {"type": "strategy_dormant", "strategy": "trend", "message": "休眠"},
    ]
    actions = orch._param_adjust_actions(alerts)
    assert len(actions) == 2
    by_strategy = {a["strategy"]: a for a in actions}
    assert by_strategy["grid"]["value"] == 1.0
    assert by_strategy["grid"]["param"] == "leverage"
    assert by_strategy["scalping"]["value"] == 2.0


def test_param_adjust_actions_dedup():
    orch = _param_adjust_orch()
    alerts = [
        {"type": "strategy_health_critical", "strategy": "grid", "message": "F"},
        {"type": "strategy_declining", "strategy": "grid", "message": "恶化"},
    ]
    assert len(orch._param_adjust_actions(alerts)) == 1


def test_param_adjust_actions_consume_rl_recommendation_with_risk_cap():
    calls = []

    class FakeRLAgent:
        def get_parameter_adjustment(self, param_name, current_value, state):
            calls.append((param_name, current_value, state))
            return 1.5

    orch = QuantAGIOrchestrator(
        config={"agi_orchestrator": {"param_adaptation": {
            "enabled": True, "declining_leverage": 2.0,
        }}},
        rl_agent=FakeRLAgent(),
    )
    orch._param_adapt_last_cycle = {}
    alerts = [{"type": "strategy_declining", "strategy": "grid"}]
    decision = {
        "market_regime": "trend_bearish",
        "market_regime_strength": 0.7,
        "market_regime_confidence": 0.8,
        "strategy_metrics": {"grid": {"max_drawdown": 0.12}},
    }

    actions = orch._param_adjust_actions(alerts, decision)

    assert len(calls) == 1
    assert calls[0][0:2] == ("leverage", 2.0)
    assert calls[0][2].strategy_id == "grid"
    assert calls[0][2].market_regime == "trend_bearish"
    assert actions[0]["value"] == 1.5


def test_param_adjust_actions_never_exceed_rule_risk_cap():
    class FakeRLAgent:
        def get_parameter_adjustment(self, param_name, current_value, state):
            return 4.0

    orch = QuantAGIOrchestrator(
        config={"agi_orchestrator": {"param_adaptation": {
            "enabled": True, "declining_leverage": 2.0,
        }}},
        rl_agent=FakeRLAgent(),
    )
    orch._param_adapt_last_cycle = {}

    actions = orch._param_adjust_actions(
        [{"type": "strategy_declining", "strategy": "grid"}]
    )

    assert actions[0]["value"] == 2.0


def test_param_adjust_disabled():
    orch = _param_adjust_orch(enabled=False)
    alerts = [{"type": "strategy_health_critical", "strategy": "grid", "message": "F"}]
    assert orch._param_adjust_actions(alerts) == []


# ── 参数自适应恢复（param adaptation restore）────────────

def _param_restore_orch(**overrides):
    cfg = {"agi_orchestrator": {"param_adaptation": {
        "enabled": True, "health_f_leverage": 1.0, "declining_leverage": 2.0,
        "restore_enabled": True, "restore_leverage": 3.0,
        "restore_leverage_b": 2.0,
        "restore_min_interval_cycles": 5,
    }, "state_path": "data/__test_isolated_state__.json"}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["param_adaptation"][k] = v
    orch = QuantAGIOrchestrator(config=cfg)
    orch._param_adapt_last_cycle = {}
    return orch


def test_param_restore_generate():
    """strategy_recovered（A 级）→ 生成 param_adjust value=restore_leverage。"""
    orch = _param_restore_orch()
    alerts = [{"type": "strategy_recovered", "strategy": "grid",
               "grade": "A", "message": "健康度恢复A"}]
    actions = orch._param_restore_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 3.0


def test_param_restore_grade_b_more_conservative():
    """B 级恢复 → 只恢复到 restore_leverage_b（更保守），非完整 3.0。"""
    orch = _param_restore_orch()
    alerts = [{"type": "strategy_recovered", "strategy": "grid", "grade": "B"}]
    actions = orch._param_restore_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["value"] == 2.0


def test_param_restore_missing_grade_falls_back_b():
    """无 grade 字段（旧告警兼容）→ 保守回退 restore_leverage_b。"""
    orch = _param_restore_orch()
    alerts = [{"type": "strategy_recovered", "strategy": "grid"}]
    actions = orch._param_restore_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["value"] == 2.0


def test_param_restore_disabled():
    """restore_enabled=False → 无恢复动作。"""
    orch = _param_restore_orch(restore_enabled=False)
    alerts = [{"type": "strategy_recovered", "strategy": "grid"}]
    assert orch._param_restore_actions(alerts) == []


def test_param_restore_skip_paused():
    """策略被 AGI 暂停 → 不升杠杆。"""
    orch = _param_restore_orch()
    orch._paused_strategies = {"grid"}
    alerts = [{"type": "strategy_recovered", "strategy": "grid"}]
    assert orch._param_restore_actions(alerts) == []


def test_param_restore_skip_give_back():
    """同周期有回吐告警 → 回吐中不升杠杆。"""
    orch = _param_restore_orch()
    alerts = [
        {"type": "strategy_recovered", "strategy": "grid"},
        {"type": "pnl_give_back", "strategy": "grid"},
    ]
    assert orch._param_restore_actions(alerts) == []


def test_param_restore_cooldown():
    """距上次 param 调整不足 restore_min_interval_cycles → 跳过（防降了又升振荡）。"""
    orch = _param_restore_orch()
    orch._param_adapt_last_cycle = {"grid": orch._cycle_count - 2}  # 2 < 5 周期
    alerts = [{"type": "strategy_recovered", "strategy": "grid"}]
    assert orch._param_restore_actions(alerts) == []


def test_param_restore_ignore_non_recovered():
    """非 strategy_recovered 告警 → 无恢复动作。"""
    orch = _param_restore_orch()
    alerts = [{"type": "strategy_health_critical", "strategy": "grid"}]
    assert orch._param_restore_actions(alerts) == []


# ── 决策可解释审计（rationale）───────────────────────────

def test_decision_includes_rationale(tmp_path):
    orch = _full_orchestrator(tmp_path)
    perception = {
        "equity": 1000.0,
        "total_capital": 1000.0,
        "contribution": None,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
    }
    decision = asyncio.run(orch._decide(perception))
    assert "rationale" in decision
    assert isinstance(decision["rationale"], list)


async def test_unhandled_decision_failure_returns_fail_closed_report(tmp_path, monkeypatch):
    orch = _full_orchestrator(tmp_path, cooldown=0)

    def raise_on_regime(_regime):
        raise RuntimeError("regime mapping unavailable")

    monkeypatch.setattr("core.quant_agi_orchestrator._map_regime", raise_on_regime)

    report = await orch.run_cycle()

    assert report["status"] == "fail_closed"
    assert report["decision"]["allocation_plan"] is None
    assert report["decision"]["strategy_metrics"] == {}
    assert report["decision"]["strategy_names"] == []
    assert report["decision"]["rationale"]
    assert report["decision"]["fail_closed"] is True
    assert report["decision"]["error"] == {
        "type": "RuntimeError",
        "message": "regime mapping unavailable",
    }
    assert report["decision"]["fail_closed"] is True
    _assert_json_safe(report)


async def test_decision_fail_closed_skips_reflection_and_persistence(tmp_path, monkeypatch):
    orch = _full_orchestrator(tmp_path, cooldown=0)
    calls = []

    async def raise_decision_error(*_args, **_kwargs):
        raise RuntimeError("decision unavailable")

    def unexpected_call(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("fail-closed decision must not run side effects")

    monkeypatch.setattr(orch, "_decide_impl", raise_decision_error)
    for method_name in (
        "_act", "_reflect", "_serialize_learning_state",
        "_persist_state", "_append_decision_lineage",
    ):
        monkeypatch.setattr(orch, method_name, unexpected_call)

    report = await orch.run_cycle()

    assert report["status"] == "fail_closed"
    assert report["decision"]["allocation_plan"] is None
    assert report["decision"]["strategy_metrics"] == {}
    assert report["decision"]["strategy_names"] == []
    assert report["decision"]["rationale"]
    assert report["decision"]["fail_closed"] is True
    assert report["actions"] == []
    assert report["errors"] == [{
        "stage": "decision",
        "type": "RuntimeError",
        "message": "decision unavailable",
    }]
    assert calls == []
    assert not (tmp_path / "state.json").exists()
    _assert_json_safe(report)


@pytest.mark.parametrize(
    ("stage", "method_name"),
    [
        ("attribution", "_attribute_pnl"),
        ("projection", "_project_pnl"),
        ("decision", "_decide"),
    ],
)
async def test_run_cycle_contains_pipeline_stage_failures(
    tmp_path, monkeypatch, stage, method_name
):
    orch = _full_orchestrator(tmp_path, cooldown=0)

    def raise_stage_error(*_args, **_kwargs):
        raise RuntimeError(f"{stage} unavailable")

    monkeypatch.setattr(orch, method_name, raise_stage_error)

    report = await orch.run_cycle()

    assert report["status"] == ("fail_closed" if stage == "decision" else "degraded")
    assert report["errors"] == [{
        "stage": stage,
        "type": "RuntimeError",
        "message": f"{stage} unavailable",
    }]
    if stage == "decision":
        assert report["actions"] == []
        assert report["decision"]["fail_closed"] is True
    _assert_json_safe(report)


async def test_malformed_perception_is_contained_by_decision_guard(tmp_path):
    orch = _full_orchestrator(tmp_path)

    decision = await orch._decide(None)

    assert decision["allocation_plan"] is None
    assert decision["error"]["type"] == "AttributeError"
    assert decision["rationale"]


async def test_decide_exception_logs_traceback_and_returns_fail_closed_shape(
    tmp_path, monkeypatch
):
    from loguru import logger

    orch = _full_orchestrator(tmp_path)
    captured = []

    async def raise_decision_error(*_args, **_kwargs):
        raise RuntimeError("unexpected decision failure")

    handler_id = logger.add(
        lambda message: captured.append(message.record),
        level="ERROR",
    )
    monkeypatch.setattr(orch, "_decide_impl", raise_decision_error)
    try:
        decision = await orch._decide(
            {"equity": 1000.0, "total_capital": 1000.0}
        )
    finally:
        logger.remove(handler_id)

    assert decision["allocation_plan"] is None
    assert decision["strategy_metrics"] == {}
    assert decision["strategy_names"] == []
    assert decision["rationale"]
    assert decision["fail_closed"] is True
    assert decision["error"] == {
        "type": "RuntimeError",
        "message": "unexpected decision failure",
    }
    assert any(
        record["message"].startswith("[AGI-Decide]")
        and record["exception"] is not None
        for record in captured
    )


def test_actions_carry_rationale():
    orch = _param_adjust_orch()
    alerts = [{"type": "strategy_health_critical", "strategy": "grid", "message": "健康度F"}]
    actions = orch._act({"allocation_plan": {}, "strategy_names": ["grid"]}, alerts)
    for a in actions:
        assert "rationale" in a


# ── 决策溯源与可观测性（decision_lineage）──────────────────

def _lineage_orch(tmp_path, **overrides):
    cfg = {
        "agi_orchestrator": {
            "state_path": str(tmp_path / "state.json"),
            "cooldown_seconds": 0,
            "decision_lineage": {
                "enabled": True, "max_entries": 50,
                "path": str(tmp_path / "lineage.json"),
            },
        }
    }
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["decision_lineage"][k] = v
    return QuantAGIOrchestrator(
        config=cfg,
        regime_engine=FakeRegimeEngine(),
        contribution_analyzer=FakeContributionAnalyzer(),
        capital_allocator=FakeCapitalAllocator(equity=1000.0),
        dynamic_allocator=FakeDynamicAllocator(),
    )


async def test_decision_lineage_appends_and_history(tmp_path):
    orch = _lineage_orch(tmp_path)
    report = await orch.run_cycle()
    assert "decision_id" in report
    history = orch.get_decision_history()
    assert len(history) == 1
    assert history[0]["decision_id"] == report["decision_id"]
    assert "rationale" in history[0]
    assert "actions" in history[0]
    _assert_json_safe(history)


async def test_decision_lineage_disabled_by_default(tmp_path):
    orch = _full_orchestrator(tmp_path, cooldown=0)
    await orch.run_cycle()
    assert orch.get_decision_history() == []


async def test_decision_lineage_persisted_and_reloaded(tmp_path):
    orch = _lineage_orch(tmp_path)
    await orch.run_cycle()
    orch2 = _lineage_orch(tmp_path)  # 同一 path 重新加载
    assert len(orch2.get_decision_history()) == 1


# ── 风险自愈闭环（self_heal）──────────────────────────────

def test_self_heal_actions_generate():
    orch = QuantAGIOrchestrator(config={})
    alerts = [
        {"type": "strategy_probing", "strategy": "grid", "message": "缩量试探中"},
        {"type": "strategy_frozen_observing", "strategy": "scalping", "message": "观察期"},
        {"type": "strategy_permanently_frozen", "strategy": "trend", "message": "永久冻结"},
    ]
    actions = orch._self_heal_actions(alerts)
    assert len(actions) == 2
    assert all(a["type"] == "self_heal" for a in actions)
    assert {a["strategy"] for a in actions} == {"grid", "scalping"}


# ── 自主进攻性资金分配（offensive_allocation）────────────

def _offensive_orch(**overrides):
    cfg = {"agi_orchestrator": {"offensive_allocation": {
        "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
        "boost_step": 0.05, "max_target": 0.4,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["offensive_allocation"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _offensive_perception(regime="trend_bullish", strength=0.7, equity=1000.0,
                          healthy="A", strategies=None):
    return {
        "equity": equity,
        "total_capital": 1000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 0.0,
        "market_regime": {"regime": regime, "strength": strength},
        "contribution": {"strategies": strategies or {"grid": {"health_grade": healthy}}},
        "freeze_state": {},
    }


def test_offensive_opportunity_alert_on_trend():
    orch = _offensive_orch()
    alerts = orch._diagnose(_offensive_perception())
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    assert opp[0]["strategies"] == ["grid"]


def test_offensive_skipped_on_range_bound():
    orch = _offensive_orch()
    alerts = orch._diagnose(_offensive_perception(regime="range_bound"))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_skipped_on_weak_strength():
    orch = _offensive_orch()
    alerts = orch._diagnose(_offensive_perception(strength=0.4))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_skipped_on_weak_health():
    orch = _offensive_orch()
    alerts = orch._diagnose(_offensive_perception(healthy="C"))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_skipped_on_drawdown():
    orch = _offensive_orch()
    orch._update_goal_planning_state(1000.0, 1000.0)  # 峰值 1000
    alerts = orch._diagnose(_offensive_perception(equity=930.0))  # 回撤 7%
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_allocation_actions_generate():
    orch = _offensive_orch()
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "increase"
    assert actions[0]["target_allocation"] == pytest.approx(0.25)  # 0.2 + 0.05


def test_offensive_target_capped_at_max():
    orch = _offensive_orch()
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.38}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions[0]["target_allocation"] == pytest.approx(0.4)  # 封顶 max_target


def test_offensive_skip_when_at_max_target():
    """进攻封顶守卫：当前权重已达 max_target 时跳过，不生成无效 increase 动作。"""
    orch = _offensive_orch()
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    # base = 0.4 == max_target=0.4 → 加仓无意义，跳过
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []


def test_offensive_skip_when_above_max_target():
    """进攻封顶守卫：当前权重已超过 max_target 时跳过（避免 target<base 的反向动作）。"""
    orch = _offensive_orch()
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    # base = 0.45 > max_target=0.4 → 若加仓 target=0.4 < base（实际减仓），必须跳过
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.45}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []


def test_offensive_total_allocation_cap_blocks_overflow():
    """账户级总敞口封顶：加仓后组合总权重突破 max_total_allocation 时跳过。"""
    orch = _offensive_orch(max_total_allocation=0.5)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    # 总权重 0.2+0.28=0.48，加仓 0.05 → 0.53 > 0.5 → 跳过
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2}, "trend": {"target_weight": 0.28},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []


def test_offensive_total_allocation_cap_allow_within():
    """账户级总敞口封顶：加仓后组合总权重未突破上限时放行。"""
    orch = _offensive_orch(max_total_allocation=0.5)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    # 总权重 0.2+0.2=0.4，加仓 0.05 → 0.45 ≤ 0.5 → 放行
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2}, "trend": {"target_weight": 0.2},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["target_allocation"] == pytest.approx(0.25)


def test_offensive_total_allocation_cap_cumulative():
    """账户级总敞口封顶：多策略累计加仓，突破上限后后续策略被阻断。"""
    orch = _offensive_orch(max_total_allocation=0.45)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid", "trend"], "boost_step": 0.05}]
    # 初始总 0.4；grid 加 0.05 → 0.45（达上限）；trend 再加 0.05 → 0.5 > 0.45 被阻断
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2}, "trend": {"target_weight": 0.2},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert [a["strategy"] for a in actions] == ["grid"]


def test_offensive_total_allocation_cap_exact_boundary():
    """账户级总敞口封顶：加仓后总权重恰好等于上限时放行（≤ 边界）。"""
    orch = _offensive_orch(max_total_allocation=0.45)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    # 总权重 0.2+0.2=0.4，加仓 0.05 → 0.45 == 上限 → 放行
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2}, "trend": {"target_weight": 0.2},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["target_allocation"] == pytest.approx(0.25)


def test_offensive_disabled():
    orch = _offensive_orch(enabled=False)
    alerts = orch._diagnose(_offensive_perception())
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert orch._offensive_allocation_actions(
        {}, [{"type": "offensive_opportunity", "strategies": ["grid"]}]
    ) == []


# ── 进攻止盈回落（offensive profit_take）──────────────────

def test_offensive_profit_take_triggers_decrease():
    """进攻止盈：策略盈利达阈值 → 生成 decrease 动作，清除归因。"""
    orch = _offensive_orch(profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03})
    # 模拟上次进攻加仓时 entry_pnl=10.0，当前 pnl=15.0（盈利5≥3）
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 15.0}},
    }
    actions = orch._offensive_allocation_actions(decision, [])
    dec = [a for a in actions if a["action"] == "decrease"]
    assert len(dec) == 1
    assert dec[0]["strategy"] == "grid"
    assert dec[0]["target_allocation"] == pytest.approx(0.17)  # 0.2 - 0.03
    # 归因已清除
    assert "grid" not in orch._offensive_attribution


def test_offensive_profit_take_below_threshold_skips():
    """盈利未达阈值 → 不止盈。"""
    orch = _offensive_orch(profit_take={"enabled": True, "pnl_threshold": 5.0, "revert_step": 0.03})
    orch._offensive_attribution["grid"] = 10.0  # entry
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 12.0}},  # 盈利2 < 5
    }
    actions = orch._offensive_allocation_actions(decision, [])
    assert not [a for a in actions if a["action"] == "decrease"]
    assert "grid" in orch._offensive_attribution  # 归因未清除


def test_offensive_profit_take_disabled_skips():
    """profit_take 禁用 → 不止盈。"""
    orch = _offensive_orch(profit_take={"enabled": False})
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 100.0}},  # 大盈利
    }
    actions = orch._offensive_allocation_actions(decision, [])
    assert not [a for a in actions if a["action"] == "decrease"]


def test_offensive_profit_take_excludes_from_re_increase():
    """止盈后同周期不再重新加仓（避免「减了又加」自相矛盾）。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03},
        min_interval_cycles=0,
    )
    orch._offensive_attribution["grid"] = 10.0
    # 同时有进攻告警请求加仓 grid → 但 grid 已止盈，应跳过
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 15.0}},
    }
    actions = orch._offensive_allocation_actions(decision, alerts)
    # 有 decrease（止盈），无 increase（被 profit_taken 排除）
    dec = [a for a in actions if a["action"] == "decrease"]
    inc = [a for a in actions if a["action"] == "increase"]
    assert len(dec) == 1
    assert len(inc) == 0


# ── 进攻止损回落（offensive stop_loss）──────────────────

def test_offensive_stop_loss_triggers_decrease():
    """进攻止损：策略亏损达阈值 → 生成 decrease 动作，清除归因。"""
    orch = _offensive_orch(stop_loss={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03})
    orch._offensive_attribution["grid"] = 10.0  # entry_pnl=10
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 6.0}},  # 亏损4≥3
    }
    actions = orch._offensive_allocation_actions(decision, [])
    dec = [a for a in actions if a["action"] == "decrease"]
    assert len(dec) == 1
    assert dec[0]["strategy"] == "grid"
    assert dec[0]["target_allocation"] == pytest.approx(0.17)  # 0.2 - 0.03
    assert "grid" not in orch._offensive_attribution


def test_offensive_stop_loss_below_threshold_skips():
    """亏损未达阈值 → 不止损。"""
    orch = _offensive_orch(stop_loss={"enabled": True, "pnl_threshold": 5.0, "revert_step": 0.03})
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 7.0}},  # 亏损3 < 5
    }
    actions = orch._offensive_allocation_actions(decision, [])
    assert not [a for a in actions if a["action"] == "decrease"]
    assert "grid" in orch._offensive_attribution


def test_offensive_stop_loss_disabled_skips():
    """stop_loss 禁用 → 不止损。"""
    orch = _offensive_orch(stop_loss={"enabled": False})
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": -100.0}},  # 巨亏
    }
    actions = orch._offensive_allocation_actions(decision, [])
    assert not [a for a in actions if a["action"] == "decrease"]


def test_offensive_stop_loss_and_profit_take_coexist():
    """止盈和止损同时启用：grid 止盈 + trend 止损，互不干扰。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03},
        stop_loss={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03},
        min_interval_cycles=0,
    )
    # grid 盈利5≥3（止盈），trend 亏损4≥3（止损）
    orch._offensive_attribution["grid"] = 10.0
    orch._offensive_attribution["trend"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {
            "grid": {"target_weight": 0.2}, "trend": {"target_weight": 0.3},
        }},
        "strategy_metrics": {
            "grid": {"total_pnl": 15.0},  # 盈利5
            "trend": {"total_pnl": 6.0},  # 亏损4
        },
    }
    actions = orch._offensive_allocation_actions(decision, [])
    dec = [a for a in actions if a["action"] == "decrease"]
    assert len(dec) == 2
    assert {a["strategy"] for a in dec} == {"grid", "trend"}
    assert len(orch._offensive_attribution) == 0  # 两者归因都清除


def test_offensive_stop_loss_cooldown_blocks_re_offend():
    """止损后冷却期内禁止重新进攻该策略。"""
    orch = _offensive_orch(
        stop_loss={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "cooldown_cycles": 10},
        min_interval_cycles=0,
    )
    # 模拟 grid 在 cycle 100 止损
    orch._offensive_stop_loss_cooldown["grid"] = 100
    orch._cycle_count = 105  # 距止损5周期 < 10 → 冷却中
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.1}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    inc = [a for a in actions if a["action"] == "increase"]
    assert len(inc) == 0  # 冷却中，不进攻


def test_offensive_stop_loss_cooldown_expires():
    """止损后超过冷却期可重新进攻。"""
    orch = _offensive_orch(
        stop_loss={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "cooldown_cycles": 10},
        min_interval_cycles=0,
    )
    orch._offensive_stop_loss_cooldown["grid"] = 100
    orch._cycle_count = 111  # 距止损11周期 >= 10 → 冷却结束
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.1}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    inc = [a for a in actions if a["action"] == "increase"]
    assert len(inc) == 1  # 冷却结束，可进攻


def test_offensive_stop_loss_cooldown_records_on_trigger():
    """止损触发时记录冷却周期号。"""
    orch = _offensive_orch(
        stop_loss={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "cooldown_cycles": 10},
    )
    orch._cycle_count = 42
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 6.0}},  # 亏损4≥3
    }
    orch._offensive_allocation_actions(decision, [])
    assert orch._offensive_stop_loss_cooldown.get("grid") == 42


# ── 止盈/止损阶梯式减仓（revert_max_multiplier）──────────

def test_offensive_profit_take_scaled_revert():
    """止盈阶梯减仓：盈利远超阈值 → 减仓幅度放大。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "revert_max_multiplier": 3.0},
    )
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.5}}},
        "strategy_metrics": {"grid": {"total_pnl": 19.0}},  # 盈利9 = 阈值3倍
    }
    actions = orch._offensive_allocation_actions(decision, [])
    dec = [a for a in actions if a["action"] == "decrease"]
    assert len(dec) == 1
    # over_ratio=9/3=3, mult=min(3.0, 3.0)=3.0, revert=0.03*3=0.09
    assert dec[0]["target_allocation"] == pytest.approx(0.5 - 0.09)


def test_offensive_profit_take_scaled_revert_capped():
    """止盈阶梯减仓：超出阈值远超 max_multiplier → 封顶。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "revert_max_multiplier": 2.0},
    )
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.5}}},
        "strategy_metrics": {"grid": {"total_pnl": 25.0}},  # 盈利15 = 阈值5倍
    }
    actions = orch._offensive_allocation_actions(decision, [])
    dec = [a for a in actions if a["action"] == "decrease"]
    assert len(dec) == 1
    # over_ratio=15/3=5, mult=min(2.0, 5.0)=2.0, revert=0.03*2=0.06
    assert dec[0]["target_allocation"] == pytest.approx(0.5 - 0.06)


def test_offensive_stop_loss_scaled_revert():
    """止损阶梯减仓：亏损远超阈值 → 减仓幅度放大（快速止血）。"""
    orch = _offensive_orch(
        stop_loss={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "revert_max_multiplier": 3.0},
    )
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.5}}},
        "strategy_metrics": {"grid": {"total_pnl": 1.0}},  # 亏损9 = 阈值3倍
    }
    actions = orch._offensive_allocation_actions(decision, [])
    dec = [a for a in actions if a["action"] == "decrease"]
    assert len(dec) == 1
    # over_ratio=9/3=3, mult=3.0, revert=0.09
    assert dec[0]["target_allocation"] == pytest.approx(0.5 - 0.09)


def test_offensive_scaled_revert_default_no_scale():
    """未配置 revert_max_multiplier → 默认 1.0，减仓不缩放（向后兼容）。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03},
    )
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.5}}},
        "strategy_metrics": {"grid": {"total_pnl": 19.0}},  # 盈利9
    }
    actions = orch._offensive_allocation_actions(decision, [])
    dec = [a for a in actions if a["action"] == "decrease"]
    assert len(dec) == 1
    # 默认 revert_max_multiplier=1.0 → revert=0.03
    assert dec[0]["target_allocation"] == pytest.approx(0.5 - 0.03)


# ── 止盈后冷却（profit_take cooldown）────────────────────

def test_offensive_profit_take_cooldown_blocks_re_offend():
    """止盈后冷却期内禁止重新进攻该策略。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "cooldown_cycles": 5},
        min_interval_cycles=0,
    )
    orch._offensive_profit_take_cooldown["grid"] = 100
    orch._cycle_count = 103  # 距止盈3周期 < 5 → 冷却中
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.1}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    inc = [a for a in actions if a["action"] == "increase"]
    assert len(inc) == 0  # 冷却中，不进攻


def test_offensive_profit_take_cooldown_expires():
    """止盈后超过冷却期可重新进攻。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "cooldown_cycles": 5},
        min_interval_cycles=0,
    )
    orch._offensive_profit_take_cooldown["grid"] = 100
    orch._cycle_count = 106  # 距止盈6周期 >= 5 → 冷却结束
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.1}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    inc = [a for a in actions if a["action"] == "increase"]
    assert len(inc) == 1  # 冷却结束，可进攻


def test_offensive_profit_take_cooldown_records_on_trigger():
    """止盈触发时记录冷却周期号。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03, "cooldown_cycles": 5},
    )
    orch._cycle_count = 42
    orch._offensive_attribution["grid"] = 10.0
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 15.0}},  # 盈利5≥3
    }
    orch._offensive_allocation_actions(decision, [])
    assert orch._offensive_profit_take_cooldown.get("grid") == 42


def test_offensive_profit_take_cooldown_default_zero():
    """未配置 cooldown_cycles → 默认 0，止盈后不冷却（向后兼容）。"""
    orch = _offensive_orch(
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03},
        min_interval_cycles=0,
    )
    orch._offensive_attribution["grid"] = 10.0
    orch._cycle_count = 50
    # 同时有进攻告警请求加仓 grid，但 grid 本周期止盈 → profit_taken 已排除
    # 这里验证默认 cooldown=0 时，后续周期（51）无冷却记录可重新进攻
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.1}}},
        "strategy_metrics": {"grid": {"total_pnl": 15.0}},
    }
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    orch._offensive_allocation_actions(decision, alerts)
    # 本周期止盈（profit_taken 排除），无 increase
    # 下一周期（51）grid 无 cooldown 记录（默认0），应可重新进攻
    orch._cycle_count = 51
    orch._offensive_attribution["grid"] = 15.0  # 重新建立归因
    decision2 = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.1}}},
        "strategy_metrics": {"grid": {"total_pnl": 15.0}},
    }
    actions2 = orch._offensive_allocation_actions(decision2, alerts)
    inc = [a for a in actions2 if a["action"] == "increase"]
    assert len(inc) == 1


# ── 进攻力度动量缩放（momentum_scaling）──────────────────

def test_offensive_momentum_scaling_strong_trend():
    """强趋势（strength=0.9）→ boost 被放大到接近 max_mult。"""
    orch = _offensive_orch(
        momentum_scaling_enabled=True,
        momentum_scaling_min_multiplier=0.5,
        momentum_scaling_max_multiplier=1.5,
    )
    alerts = orch._diagnose(_offensive_perception(regime="trend_bullish", strength=0.9))
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    # min_regime_strength=0.6, norm=(0.9-0.6)/(1.0-0.6)=0.75
    # boost = 0.05 * (0.5 + (1.5-0.5)*0.75) = 0.05 * 1.25 = 0.0625
    assert opp[0]["boost_step"] == pytest.approx(0.0625)


def test_offensive_momentum_scaling_marginal_trend():
    """边际趋势（strength=0.6）→ boost 被缩小到 min_mult。"""
    orch = _offensive_orch(
        momentum_scaling_enabled=True,
        momentum_scaling_min_multiplier=0.5,
        momentum_scaling_max_multiplier=1.5,
    )
    alerts = orch._diagnose(_offensive_perception(regime="trend_bullish", strength=0.6))
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    # norm=0 → boost = 0.05 * 0.5 = 0.025
    assert opp[0]["boost_step"] == pytest.approx(0.025)


def test_offensive_momentum_scaling_disabled():
    """禁用 → boost 不缩放，保持原始值。"""
    orch = _offensive_orch(
        momentum_scaling_enabled=False,
    )
    alerts = orch._diagnose(_offensive_perception(regime="trend_bullish", strength=0.9))
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    assert opp[0]["boost_step"] == pytest.approx(0.05)  # 原始 boost_step


def test_offensive_momentum_scaling_range_bound():
    """震荡市也按 strength 缩放，用 range_bound_min_strength 归一化。"""
    orch = _offensive_orch(
        range_bound_enabled=True,
        range_bound_boost_step=0.03,
        momentum_scaling_enabled=True,
        momentum_scaling_min_multiplier=0.5,
        momentum_scaling_max_multiplier=1.5,
    )
    alerts = orch._diagnose(_offensive_perception(regime="range_bound", strength=0.9))
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    # min_strength=0.3, norm=(0.9-0.3)/(1.0-0.3)=0.857
    # boost = 0.03 * (0.5 + 1.0*0.857) = 0.03 * 1.357 = 0.0407
    assert opp[0]["boost_step"] == pytest.approx(0.03 * 1.3571, rel=1e-2)


def test_offensive_range_bound_enabled_opportunity():
    orch = _offensive_orch(range_bound_enabled=True)
    alerts = orch._diagnose(_offensive_perception(regime="range_bound", strength=0.5))
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    assert opp[0]["strategies"] == ["grid"]


def test_offensive_range_bound_strategy_filter():
    orch = _offensive_orch(range_bound_enabled=True)
    strategies = {
        "grid": {"health_grade": "A"},
        "oscillation_harvest": {"health_grade": "B"},
        "trend": {"health_grade": "A"},
    }
    alerts = orch._diagnose(
        _offensive_perception(regime="range_bound", strength=0.5, strategies=strategies)
    )
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    assert set(opp[0]["strategies"]) == {"grid", "oscillation_harvest"}


def test_offensive_range_bound_weak_strength():
    orch = _offensive_orch(range_bound_enabled=True, range_bound_min_strength=0.3)
    alerts = orch._diagnose(_offensive_perception(regime="range_bound", strength=0.2))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_range_bound_disabled():
    orch = _offensive_orch(range_bound_enabled=False)
    alerts = orch._diagnose(_offensive_perception(regime="range_bound", strength=0.5))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_range_bound_drawdown_gate():
    orch = _offensive_orch(range_bound_enabled=True)
    orch._update_goal_planning_state(1000.0, 1000.0)  # 峰值 1000
    alerts = orch._diagnose(_offensive_perception(regime="range_bound", strength=0.5, equity=930.0))
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_range_bound_independent_boost_step():
    """震荡市进攻告警用独立的 range_bound_boost_step，不复用趋势市 boost_step。"""
    orch = _offensive_orch(range_bound_enabled=True,
                           boost_step=0.05, range_bound_boost_step=0.03)
    alerts = orch._diagnose(_offensive_perception(regime="range_bound", strength=0.5))
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    assert opp[0]["mode"] == "range_bound"
    assert opp[0]["boost_step"] == pytest.approx(0.03)


def test_offensive_trend_mode_label_and_boost():
    """趋势市进攻告警 mode=trend，用趋势市 boost_step。"""
    orch = _offensive_orch(range_bound_enabled=True,
                           boost_step=0.05, range_bound_boost_step=0.03)
    alerts = orch._diagnose(_offensive_perception(regime="trend_bullish", strength=0.7))
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    assert opp[0]["mode"] == "trend"
    assert opp[0]["boost_step"] == pytest.approx(0.05)


def test_offensive_range_bound_independent_max_target():
    """震荡市进攻加仓封顶用独立的 range_bound_max_target。"""
    orch = _offensive_orch(range_bound_enabled=True,
                           range_bound_boost_step=0.03, range_bound_max_target=0.25)
    alerts = [{"type": "offensive_opportunity", "mode": "range_bound",
               "strategies": ["grid"], "boost_step": 0.03}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.24}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    # 0.24 + 0.03 = 0.27，但封顶 0.25
    assert actions[0]["target_allocation"] == pytest.approx(0.25)


def test_offensive_range_bound_independent_cooldown():
    """震荡市用独立的 range_bound_min_interval_cycles（更短冷却）。"""
    orch = _offensive_orch(range_bound_enabled=True,
                           range_bound_min_interval_cycles=2,
                           min_interval_cycles=5)
    # 模拟上一周期刚进攻过：cycle_count=0，last=-1 → 距离1 < 5（趋势冷却）但 == 2-1=1 < 2
    orch._cycle_count = 10
    orch._last_offensive_cycle["grid"] = 9  # 1 周期前
    alerts = [{"type": "offensive_opportunity", "mode": "range_bound",
               "strategies": ["grid"], "boost_step": 0.03}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.1}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # 距离1 < range_bound_min_interval_cycles=2 → 仍冷却中
    assert len(actions) == 0

    # 2 周期后 → 距离2 >= 2 → 放行
    orch._last_offensive_cycle["grid"] = 8  # 2 周期前
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1


def test_offensive_focus_top_n_by_health_score():
    """进攻火力集中：多健康策略按 health_score 降序取前 max_offensive_strategies 个。"""
    orch = _offensive_orch(max_offensive_strategies=2)
    strategies = {
        "grid": {"health_grade": "A", "health_score": 70},
        "trend": {"health_grade": "A", "health_score": 90},
        "scalping": {"health_grade": "B", "health_score": 60},
        "sniper": {"health_grade": "A", "health_score": 85},
    }
    alerts = orch._diagnose(
        _offensive_perception(regime="trend_bullish", strength=0.7, strategies=strategies)
    )
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    assert opp[0]["strategies"] == ["trend", "sniper"]  # health_score 降序前 2


def test_offensive_focus_no_limit_when_few_strategies():
    """策略数 ≤ max_offensive_strategies 时全部保留（不截断）。"""
    orch = _offensive_orch(max_offensive_strategies=5)
    strategies = {
        "grid": {"health_grade": "A", "health_score": 70},
        "trend": {"health_grade": "A", "health_score": 90},
    }
    alerts = orch._diagnose(
        _offensive_perception(regime="trend_bullish", strength=0.7, strategies=strategies)
    )
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    assert set(opp[0]["strategies"]) == {"grid", "trend"}


def test_offensive_range_bound_independent_max_strategies():
    """震荡市火力集中用独立的 range_bound_max_offensive_strategies（比趋势市更集中）。"""
    orch = _offensive_orch(range_bound_enabled=True,
                           max_offensive_strategies=3,
                           range_bound_max_offensive_strategies=1)
    strategies = {
        "grid": {"health_grade": "A", "health_score": 70},
        "oscillation_harvest": {"health_grade": "A", "health_score": 90},
        "trend": {"health_grade": "A", "health_score": 85},
    }
    alerts = orch._diagnose(
        _offensive_perception(regime="range_bound", strength=0.5, strategies=strategies)
    )
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    # 震荡市过滤到 range_bound_strategies 后按 health_score 取前 1（oscillation_harvest 90）
    assert opp[0]["strategies"] == ["oscillation_harvest"]


def test_offensive_range_bound_max_strategies_fallback():
    """未设置 range_bound_max_offensive_strategies 时回退到 max_offensive_strategies。"""
    orch = _offensive_orch(range_bound_enabled=True, max_offensive_strategies=1)
    strategies = {
        "grid": {"health_grade": "A", "health_score": 70},
        "oscillation_harvest": {"health_grade": "A", "health_score": 90},
    }
    alerts = orch._diagnose(
        _offensive_perception(regime="range_bound", strength=0.5, strategies=strategies)
    )
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    # 未设 range_bound 上限 → 回退 max_offensive_strategies=1 → 只取最优
    assert opp[0]["strategies"] == ["oscillation_harvest"]


def test_offensive_trend_unaffected_by_range_bound_max_strategies():
    """趋势市火力集中仍用 max_offensive_strategies（不受震荡市上限影响）。"""
    orch = _offensive_orch(range_bound_enabled=True,
                           max_offensive_strategies=2,
                           range_bound_max_offensive_strategies=1)
    strategies = {
        "grid": {"health_grade": "A", "health_score": 70},
        "trend": {"health_grade": "A", "health_score": 90},
        "scalping": {"health_grade": "A", "health_score": 85},
    }
    alerts = orch._diagnose(
        _offensive_perception(regime="trend_bullish", strength=0.7, strategies=strategies)
    )
    opp = [a for a in alerts if a["type"] == "offensive_opportunity"]
    assert len(opp) == 1
    # 趋势市用 max_offensive_strategies=2 → 取前 2（trend 90, scalping 85）
    assert opp[0]["strategies"] == ["trend", "scalping"]


# ── 进攻归因周期过期（TTL）──────────────────────────────

def test_offensive_attribution_ttl_allows_reoffense_after_expiry():
    """归因基线超过 TTL 后过期清除，不再被陈旧 entry_pnl 阻塞重新进攻。"""
    orch = _offensive_orch(attribution_ttl_cycles=10)
    orch._cycle_count = 30
    orch._offensive_attribution = {"grid": 100.0}
    orch._offensive_attribution_cycle = {"grid": 5}  # 25 周期前，已过期
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 90.0}},  # 90 < 100，本会被归因阻塞
    }
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["action"] == "increase"  # TTL 过期 → 陈旧归因不再阻塞


def test_offensive_attribution_fresh_still_blocks():
    """归因基线未过期时仍按 entry_pnl 阻塞（current_pnl < entry_pnl）。"""
    orch = _offensive_orch(attribution_ttl_cycles=10)
    orch._cycle_count = 10
    orch._offensive_attribution = {"grid": 100.0}
    orch._offensive_attribution_cycle = {"grid": 5}  # 5 周期前，未过期
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 90.0}},
    }
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []  # 未过期 → 90 < 100 仍阻塞


def test_offensive_attribution_ttl_disabled_never_expires():
    """TTL=0（默认）永不过期，陈旧归因持续阻塞（向后兼容）。"""
    orch = _offensive_orch()  # attribution_ttl_cycles 默认 0
    orch._cycle_count = 100
    orch._offensive_attribution = {"grid": 100.0}
    orch._offensive_attribution_cycle = {"grid": 5}
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 90.0}},
    }
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []  # TTL=0 永不过期 → 90 < 100 仍阻塞


# ── 连续进攻次数上限（max_consecutive_offenses）──────────

def test_offensive_max_consecutive_blocks_after_limit():
    """连续进攻次数达上限后暂停（无了结地一路追高加仓被阻断）。"""
    orch = _offensive_orch(max_consecutive_offenses=2)
    orch._offensive_consecutive = {"grid": 2}
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []


def test_offensive_consecutive_increments_on_offense():
    """进攻加仓后连续进攻计数 +1。"""
    orch = _offensive_orch(max_consecutive_offenses=5)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    orch._offensive_allocation_actions(decision, alerts)
    assert orch._offensive_consecutive["grid"] == 1


def test_offensive_consecutive_resets_on_profit_take():
    """止盈了结后连续进攻计数归零。"""
    orch = _offensive_orch(
        max_consecutive_offenses=5,
        profit_take={"enabled": True, "pnl_threshold": 3.0, "revert_step": 0.03},
    )
    orch._offensive_attribution = {"grid": 10.0}
    orch._offensive_attribution_cycle = {"grid": 0}
    orch._offensive_consecutive = {"grid": 3}
    alerts = []  # 无 offensive_opportunity，仅测试止盈预扫描
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 15.0}},  # 盈利 5 ≥ 3
    }
    orch._offensive_allocation_actions(decision, alerts)
    assert "grid" not in orch._offensive_consecutive  # 止盈了结 → 计数归零


# ── 进攻与账户可用保证金联动检查（available_margin_check）──

def _margin_orch(equity, enabled, ratio=0.1):
    return QuantAGIOrchestrator(
        config={"agi_orchestrator": {"offensive_allocation": {
            "enabled": True, "available_margin_check_enabled": enabled,
            "min_available_margin_ratio": ratio,
        }}},
        capital_allocator=FakeCapitalAllocator(equity=equity),
        dynamic_allocator=FakeDynamicAllocator(),
    )


def test_offensive_margin_check_blocks_insufficient():
    """可用保证金占比不足 → 暂停进攻加仓。"""
    orch = _margin_orch(equity=100.0, enabled=True, ratio=0.1)  # 可用 100-100=0 → 0 < 0.1
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []


def test_offensive_margin_check_allows_sufficient():
    """可用保证金占比充足 → 放行进攻加仓。"""
    orch = _margin_orch(equity=1000.0, enabled=True, ratio=0.1)  # 可用 900/1000=0.9 ≥ 0.1
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["action"] == "increase"


def test_offensive_margin_check_disabled_allows():
    """保证金检查未启用 → 放行（向后兼容）。"""
    orch = _margin_orch(equity=100.0, enabled=False)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1


# ── 进攻最小增幅保护（min_boost_delta）────────────────────

def test_offensive_min_boost_delta_skips_tiny_boost():
    """缩放后加仓幅度低于 min_boost_delta → 跳过（不生成动作、不污染状态）。"""
    orch = _offensive_orch(min_boost_delta=0.01)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.005}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []
    # 未污染状态
    assert orch._offensive_attribution == {}
    assert orch._last_offensive_cycle == {}
    assert orch._offensive_consecutive == {}


def test_offensive_min_boost_delta_allows_above_threshold():
    """加仓幅度 ≥ min_boost_delta → 放行。"""
    orch = _offensive_orch(min_boost_delta=0.01)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["action"] == "increase"


def test_offensive_min_boost_delta_disabled_allows_tiny_boost():
    """min_boost_delta=0（默认）无下限 → 微小加仓放行（向后兼容）。"""
    orch = _offensive_orch()  # min_boost_delta 默认 0
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.001}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1


# ── 自主策略生命周期管理（strategy_lifecycle）────────────

def _lifecycle_orch(**overrides):
    cfg = {"agi_orchestrator": {"strategy_lifecycle": {
        "enabled": True, "pause_permanently_frozen": True, "pause_dormant": True,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["strategy_lifecycle"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_strategy_lifecycle_pause_actions():
    orch = _lifecycle_orch()
    alerts = [
        {"type": "strategy_permanently_frozen", "strategy": "grid", "message": "永久冻结"},
        {"type": "strategy_dormant", "strategy": "scalping", "message": "休眠"},
        {"type": "strategy_probing", "strategy": "trend", "message": "试探中"},
    ]
    actions = orch._strategy_lifecycle_actions(alerts)
    assert len(actions) == 2
    assert all(a["type"] == "strategy_pause" for a in actions)
    assert {a["strategy"] for a in actions} == {"grid", "scalping"}


def test_strategy_lifecycle_pause_dormant_disabled():
    orch = _lifecycle_orch(pause_dormant=False)
    alerts = [
        {"type": "strategy_permanently_frozen", "strategy": "grid", "message": "永久冻结"},
        {"type": "strategy_dormant", "strategy": "scalping", "message": "休眠"},
    ]
    actions = orch._strategy_lifecycle_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["strategy"] == "grid"


def test_strategy_lifecycle_disabled():
    orch = _lifecycle_orch(enabled=False)
    alerts = [{"type": "strategy_permanently_frozen", "strategy": "grid", "message": "永久冻结"}]
    assert orch._strategy_lifecycle_actions(alerts) == []


# ── 自适应风险偏好（adaptive_risk）────────────────────────

def _adaptive_risk_orch(**overrides):
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4,
        },
        "adaptive_risk": {
            "enabled": True, "window": 5, "min_appetite": 0.2, "min_boost_ratio": 0.3,
        },
    }}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["adaptive_risk"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_risk_appetite_neutral_insufficient_sample():
    orch = _adaptive_risk_orch()
    assert orch._risk_appetite() == pytest.approx(0.5)  # 样本不足 → 中性


def test_risk_appetite_rising_equity():
    orch = _adaptive_risk_orch()
    orch._equity_window = deque([100.0, 102.0, 105.0], maxlen=5)
    assert orch._risk_appetite() > 0.5


def test_risk_appetite_falling_equity():
    orch = _adaptive_risk_orch()
    orch._equity_window = deque([100.0, 98.0, 95.0], maxlen=5)
    assert orch._risk_appetite() < 0.5


def test_risk_appetite_disabled_returns_neutral():
    orch = _offensive_orch()  # adaptive_risk 未启用
    orch._equity_window = deque([100.0, 95.0], maxlen=5)
    assert orch._risk_appetite() == pytest.approx(0.5)


# ── 回撤感知风险偏好（drawdown-aware risk appetite）────────

def test_risk_appetite_drawdown_penalizes_deep_drawdown():
    """斜率微正但回撤深 → 惩罚后偏好显著低于纯斜率型（0.6 → 0.125）。"""
    orch = _adaptive_risk_orch()  # drawdown_aware 默认 True, scale 默认 0.2
    orch._equity_window = deque([100.0, 120.0, 101.0], maxlen=5)
    # 纯斜率：slope=0.01 → appetite=0.6；回撤=(120-101)/120≈0.158 → penalty≈0.208
    appetite = orch._risk_appetite()
    assert appetite < 0.3  # 0.125
    assert appetite > 0.0


def test_risk_appetite_drawdown_aware_off_no_penalty():
    """drawdown_aware=False → 回撤深度不惩罚（向后兼容，仅看斜率）。"""
    orch = _adaptive_risk_orch(drawdown_aware=False)
    orch._equity_window = deque([100.0, 120.0, 101.0], maxlen=5)
    assert orch._risk_appetite() == pytest.approx(0.6)  # 纯斜率 0.6 无惩罚


def test_risk_appetite_drawdown_at_scale_zeroes():
    """回撤达 drawdown_scale(20%) → 惩罚因子归零（完全收敛）。"""
    orch = _adaptive_risk_orch()
    orch._equity_window = deque([100.0, 120.0, 96.0], maxlen=5)
    # 回撤=(120-96)/120=0.2 → penalty=0；斜率型 appetite=0.1 → 0.1*0=0
    assert orch._risk_appetite() == pytest.approx(0.0, abs=1e-6)


def test_risk_appetite_no_drawdown_no_penalty():
    """创新高（无回撤）→ 惩罚因子=1，偏好不受影响。"""
    orch = _adaptive_risk_orch()
    orch._equity_window = deque([100.0, 105.0, 110.0], maxlen=5)
    assert orch._risk_appetite() == pytest.approx(1.0)  # slope=0.1 → 1.0，无回撤


def test_risk_appetite_drawdown_scale_custom():
    """drawdown_scale 越大惩罚越温和（回撤更不易触发强收敛）。"""
    orch = _adaptive_risk_orch(drawdown_scale=0.5)
    orch._equity_window = deque([100.0, 120.0, 101.0], maxlen=5)
    # 回撤≈0.158，penalty=1-0.158/0.5=0.683；appetite=0.6*0.683=0.41
    appetite = orch._risk_appetite()
    assert 0.3 < appetite < 0.5  # 0.41，比 scale=0.2 时（0.125）温和


def test_offensive_scaled_by_appetite():
    orch = _adaptive_risk_orch()
    orch._equity_window = deque([100.0, 105.0], maxlen=5)  # 上涨 → appetite 1.0
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # appetite=1.0 → scaled boost = 0.05*(0.3+0.7*1.0) = 0.05
    assert actions[0]["target_allocation"] == pytest.approx(0.25)


def test_offensive_gated_by_low_appetite():
    orch = _adaptive_risk_orch(min_appetite=0.6)
    orch._equity_window = deque([100.0, 95.0], maxlen=5)  # 下跌 → appetite 0.0 < 0.6
    alerts = orch._diagnose(_offensive_perception())
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


# ── 动作冲突消解与优先级仲裁（action_reconciliation）──────

def _reconciliation_orch(enabled=True):
    cfg = {"agi_orchestrator": {"action_reconciliation": {"enabled": enabled}}}
    return QuantAGIOrchestrator(config=cfg)


def test_reconciliation_suppresses_offensive_on_risk_reduce():
    orch = _reconciliation_orch()
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.4},
        {"type": "reallocate", "strategy": "grid", "action": "decrease", "target_allocation": 0.1},
    ]
    kept = orch._reconcile_actions(actions)
    assert len(kept) == 1
    assert kept[0]["action"] == "decrease"
    assert kept[0]["target_allocation"] == 0.1
    assert orch._last_reconciliation["dropped_count"] == 1


def test_reconciliation_suppresses_idle_deploy_on_strategy_pause():
    orch = _reconciliation_orch()
    actions = [
        {"type": "strategy_pause", "strategy": "grid", "reason": "永久冻结"},
        {"type": "idle_cash_deploy", "strategy": "grid", "health_grade": "B"},
    ]
    kept = orch._reconcile_actions(actions)
    types = {a["type"] for a in kept}
    assert types == {"strategy_pause"}
    assert orch._last_reconciliation["dropped"][0]["dropped_reason"] == "risk_reduce_priority"


def test_reconciliation_dedup_reallocate_keeps_min_target():
    orch = _reconciliation_orch()
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "decrease", "target_allocation": 0.3},
        {"type": "reallocate", "strategy": "grid", "action": "decrease", "target_allocation": 0.1},
    ]
    kept = orch._reconcile_actions(actions)
    assert len(kept) == 1
    assert kept[0]["target_allocation"] == 0.1


def test_reconciliation_preserves_neutral_actions():
    orch = _reconciliation_orch()
    actions = [
        {"type": "alert_action", "level": "critical", "detail": "x"},
        {"type": "self_heal", "strategy": "grid", "reason": "probe"},
        {"type": "profit_take_close", "close_ratio": 0.3},
    ]
    kept = orch._reconcile_actions(actions)
    assert len(kept) == 3


def test_reconciliation_disabled_passthrough():
    orch = _reconciliation_orch(enabled=False)
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.4},
        {"type": "reallocate", "strategy": "grid", "action": "decrease", "target_allocation": 0.1},
    ]
    kept = orch._reconcile_actions(actions)
    assert len(kept) == 2
    assert orch._last_reconciliation is None


def test_reconciliation_clears_offensive_attribution_on_defensive_decrease():
    """防守性减仓清除该策略的进攻归因基线（避免陈旧 entry_pnl 误判后续进攻）。"""
    orch = _reconciliation_orch()
    orch._offensive_attribution = {"grid": 100.0}
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "decrease", "target_allocation": 0.1},
    ]
    kept = orch._reconcile_actions(actions)
    assert len(kept) == 1
    assert "grid" not in orch._offensive_attribution  # 归因已清除


def test_reconciliation_clears_attribution_when_offensive_increase_dropped():
    """进攻加仓被防守减仓仲裁丢弃后，其误写入的归因基线被清除。"""
    orch = _reconciliation_orch()
    orch._offensive_attribution = {"grid": 100.0}  # 模拟加仓循环已写入基线
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.4},
        {"type": "reallocate", "strategy": "grid", "action": "decrease", "target_allocation": 0.1},
    ]
    kept = orch._reconcile_actions(actions)
    assert [a["action"] for a in kept] == ["decrease"]  # increase 被丢弃
    assert "grid" not in orch._offensive_attribution  # 归因已清除


def test_reconciliation_preserves_offensive_cycle_on_defensive_decrease():
    """防守性减仓保留观察期冷却 _last_offensive_cycle（避免立即重新进攻振荡）。"""
    orch = _reconciliation_orch()
    orch._offensive_attribution = {"grid": 100.0}
    orch._last_offensive_cycle = {"grid": 5}
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "decrease", "target_allocation": 0.1},
    ]
    orch._reconcile_actions(actions)
    assert "grid" not in orch._offensive_attribution  # 归因清除
    assert orch._last_offensive_cycle == {"grid": 5}  # 冷却保留


def test_reconciliation_resets_consecutive_on_defensive_decrease():
    """防守性减仓「了结」进攻仓位 → 连续进攻计数归零。"""
    orch = _reconciliation_orch()
    orch._offensive_consecutive = {"grid": 3}
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "decrease", "target_allocation": 0.1},
    ]
    orch._reconcile_actions(actions)
    assert "grid" not in orch._offensive_consecutive  # 计数归零


def test_act_integrates_reconciliation_and_reports():
    orch = QuantAGIOrchestrator(config={"agi_orchestrator": {
        "action_reconciliation": {"enabled": True},
        "risk_response": {"enabled": True, "reduce_target": 0.1},
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4,
        },
    }})
    decision = {
        "allocation_plan": {
            "strategy_allocations": {"grid": {"target_weight": 0.2}},
        },
        "reallocation_suggestions": [],
        "strategy_names": ["grid"],
        "strategy_metrics": {},
    }
    alerts = [
        {"type": "strategy_declining", "strategy": "grid", "message": "趋势恶化"},
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
    ]
    actions = orch._act(decision, alerts)
    # 进攻加仓被风控减仓抑制，最终仅保留减仓动作
    realloc = [a for a in actions if a["type"] == "reallocate"]
    assert len(realloc) == 1
    assert realloc[0]["action"] == "decrease"
    assert orch._last_reconciliation["dropped_count"] >= 1


# ── 决策质量仲裁（confidence + priority）──────────────────

def test_action_confidence_high_for_risk_reduce():
    orch = _reconciliation_orch()
    assert orch._action_confidence({"type": "strategy_pause"}, {}) == 0.95
    assert orch._action_confidence({"type": "param_adjust"}, {}) == 0.95
    assert orch._action_confidence({"type": "profit_take_close"}, {}) == 0.95
    assert orch._action_confidence(
        {"type": "reallocate", "action": "decrease"}, {}
    ) == 0.9


def test_action_confidence_offensive_uses_evidence():
    orch = _reconciliation_orch()
    decision = {
        "market_regime_strength": 0.6,
        "strategy_metrics": {"grid": {"profit_factor": 1.5, "sharpe_ratio": 1.0}},
    }
    conf = orch._action_confidence(
        {"type": "reallocate", "action": "increase", "strategy": "grid"}, decision
    )
    # 0.3 + 0.3*0.6 + 0.2*0.5 + 0.2*1.0 = 0.78
    assert conf == pytest.approx(0.78)
    weak = orch._action_confidence(
        {"type": "reallocate", "action": "increase", "strategy": "grid"}, {}
    )
    # 无证据：0.3
    assert weak == pytest.approx(0.3)


def test_apply_decision_quality_gates_low_confidence_offensive():
    orch = _reconciliation_orch()  # min_confidence 默认 0.5
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.4},
    ]
    kept = orch._apply_decision_quality(actions, {})
    assert kept == []
    assert orch._last_confidence_gate["dropped_count"] == 1
    assert orch._last_confidence_gate["dropped"][0]["dropped_reason"] == "low_confidence"


def test_apply_decision_quality_preserves_high_confidence_offensive():
    orch = _reconciliation_orch()
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.4},
    ]
    decision = {
        "market_regime_strength": 0.7,
        "strategy_metrics": {"grid": {"profit_factor": 2.0, "sharpe_ratio": 1.5}},
    }
    kept = orch._apply_decision_quality(actions, decision)
    assert len(kept) == 1
    assert kept[0]["confidence"] == pytest.approx(0.91)


def test_apply_decision_quality_sorts_by_priority():
    orch = _reconciliation_orch()
    actions = [
        {"type": "alert_action", "detail": "x"},
        {"type": "strategy_pause", "strategy": "grid"},
        {"type": "idle_cash_deploy", "strategy": "grid"},
    ]
    decision = {
        "market_regime_strength": 0.7,
        "strategy_metrics": {"grid": {"profit_factor": 2.0, "sharpe_ratio": 1.5}},
    }
    kept = orch._apply_decision_quality(actions, decision)
    priorities = [a["priority"] for a in kept]
    assert priorities == sorted(priorities, reverse=True)
    assert kept[0]["type"] == "strategy_pause"


def test_apply_decision_quality_gate_off_when_reconciliation_disabled():
    orch = _reconciliation_orch(enabled=False)
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.4},
    ]
    kept = orch._apply_decision_quality(actions, {})
    assert len(kept) == 1  # 门控不生效，仅附字段
    assert "confidence" in kept[0]
    assert orch._last_confidence_gate is None


# ── 多时间尺度分层自治（timescale）────────────────────────

def _timescale_orch(interval=10, enabled=True):
    cfg = {"agi_orchestrator": {"timescale": {
        "enabled": enabled, "slow_cycle_interval": interval,
    }}}
    return QuantAGIOrchestrator(config=cfg)


def test_is_slow_cycle_disabled_returns_true():
    orch = _timescale_orch(enabled=False)
    assert orch._is_slow_cycle() is True  # 未启用：每周期全执行（向后兼容）


def test_is_slow_cycle_interval_boundary():
    orch = _timescale_orch(interval=10)
    orch._cycle_count = 10
    assert orch._is_slow_cycle() is True
    orch._cycle_count = 11
    assert orch._is_slow_cycle() is False
    orch._cycle_count = 20
    assert orch._is_slow_cycle() is True


def test_act_timescale_fast_skips_strategic_reallocate():
    orch = _timescale_orch(interval=10)
    orch._cycle_count = 5  # 非慢周期
    decision = {
        "allocation_plan": {},
        "reallocation_suggestions": [
            {"strategy": "grid", "action": "increase", "target_allocation": 0.3, "reason": "x"},
        ],
        "strategy_names": ["grid"],
        "strategy_metrics": {},
    }
    actions = orch._act(decision, [])
    assert actions == []  # 战略重分配（慢尺度）被跳过
    assert orch._last_timescale == "fast"


def test_act_timescale_slow_runs_strategic_reallocate():
    orch = _timescale_orch(interval=10)
    orch._cycle_count = 10  # 慢周期
    decision = {
        "allocation_plan": {},
        "reallocation_suggestions": [
            {"strategy": "grid", "action": "increase", "target_allocation": 0.3, "reason": "x"},
        ],
        "strategy_names": ["grid"],
        "strategy_metrics": {},
    }
    actions = orch._act(decision, [])
    realloc = [a for a in actions if a["type"] == "reallocate"]
    assert len(realloc) == 1
    assert realloc[0]["action"] == "increase"
    assert orch._last_timescale == "slow"


def test_act_timescale_disabled_runs_strategic_every_cycle():
    orch = _timescale_orch(enabled=False)
    orch._cycle_count = 5  # 即使非慢周期，未启用分层时仍全执行
    decision = {
        "allocation_plan": {},
        "reallocation_suggestions": [
            {"strategy": "grid", "action": "increase", "target_allocation": 0.3, "reason": "x"},
        ],
        "strategy_names": ["grid"],
        "strategy_metrics": {},
    }
    actions = orch._act(decision, [])
    realloc = [a for a in actions if a["type"] == "reallocate"]
    assert len(realloc) == 1
    assert orch._last_timescale == "slow"


# ── 自主策略恢复（strategy_resume）────────────────────────

def _resume_orch(**overrides):
    cfg = {"agi_orchestrator": {"strategy_lifecycle": {
        "enabled": True, "pause_permanently_frozen": True,
        "pause_dormant": True, "resume_recovered": True,
    }, "state_path": "data/__test_isolated_state__.json"}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["strategy_lifecycle"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_strategy_resume_actions_generate_for_paused():
    orch = _resume_orch()
    orch._paused_strategies = {"grid"}
    alerts = [{"type": "strategy_recovered", "strategy": "grid", "message": "健康度改善至 B"}]
    actions = orch._strategy_resume_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "strategy_resume"
    assert actions[0]["strategy"] == "grid"
    assert "grid" not in orch._paused_strategies  # 幂等：resume 后从集合移除


def test_strategy_resume_ignores_unpaused():
    orch = _resume_orch()
    alerts = [{"type": "strategy_recovered", "strategy": "grid", "message": "健康度改善"}]
    assert orch._strategy_resume_actions(alerts) == []


def test_strategy_resume_disabled():
    orch = _resume_orch(resume_recovered=False)
    orch._paused_strategies = {"grid"}
    alerts = [{"type": "strategy_recovered", "strategy": "grid"}]
    assert orch._strategy_resume_actions(alerts) == []


def test_strategy_lifecycle_pause_records_paused():
    orch = _resume_orch()
    alerts = [{"type": "strategy_dormant", "strategy": "grid", "message": "休眠"}]
    orch._strategy_lifecycle_actions(alerts)
    assert "grid" in orch._paused_strategies


def test_diagnose_strategy_recovered_alert():
    orch = _resume_orch()
    perception = {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_grade": "B", "trend": "improving",
            "lifecycle": "mature", "total_trades": 10, "total_pnl": 50.0,
        }}},
        "freeze_state": {},
    }
    alerts = orch._diagnose(perception)
    recovered = [a for a in alerts if a["type"] == "strategy_recovered"]
    assert len(recovered) == 1
    assert recovered[0]["strategy"] == "grid"


# ── 决策执行结果反馈（execution_result）────────────────────

def _exec_result_perception():
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
    }


def test_report_execution_result_stores_and_get():
    # 隔离真实 state 文件，避免 execution_memory 残留污染初始状态
    orch = QuantAGIOrchestrator(config={"agi_orchestrator": {"state_path": "data/__test_isolated_state__.json"}})
    assert orch.get_execution_result() is None
    orch.report_execution_result({
        "cycle": 4,
        "decision_id": "decision-4",
        "deployed": 3,
        "queued": 1,
        "rejected": 2,
        "notified": 0,
        "action_results": [{
            "trace_id": "trace-1",
            "type": "param_adjust",
            "status": "deployed",
        }],
    })
    result = orch.get_execution_result()
    assert result["deployed"] == 3
    assert result["rejected"] == 2
    assert result["action_results"][0]["trace_id"] == "trace-1"
    assert orch.get_execution_history()[0]["cycle"] == 4


def test_report_execution_result_persists_memory(tmp_path):
    config = {"agi_orchestrator": {"state_path": str(tmp_path / "state.json")}}
    orch = QuantAGIOrchestrator(config=config)
    orch._last_report = {"cycle": 2, "reflection": {}}
    orch.report_execution_result({
        "cycle": 2,
        "decision_id": "decision-2",
        "deployed": 1,
        "action_results": [{
            "trace_id": "trace-2",
            "type": "strategy_pause",
            "status": "deployed",
            "result": {"deployed": True, "strategy": "grid", "reason": "health guard"},
        }],
    })

    restored = QuantAGIOrchestrator(config=config)
    history = restored.get_execution_history()

    assert len(history) == 1
    assert history[0]["decision_id"] == "decision-2"
    assert history[0]["result"]["action_results"][0]["status"] == "deployed"
    assert history[0]["result"]["action_results"][0]["result"]["reason"] == "health guard"
    assert restored.get_execution_result()["deployed"] == 1


def test_diagnose_execution_rejected_alert():
    orch = QuantAGIOrchestrator(config={})
    orch._last_execution_result = {"deployed": 0, "queued": 0, "rejected": 2, "notified": 0}
    alerts = orch._diagnose(_exec_result_perception())
    rejected = [a for a in alerts if a["type"] == "execution_rejected"]
    assert len(rejected) == 1
    assert rejected[0]["rejected"] == 2


def test_diagnose_no_execution_rejected_alert_when_no_rejection():
    orch = QuantAGIOrchestrator(config={})
    orch._last_execution_result = {"deployed": 1, "queued": 0, "rejected": 0, "notified": 0}
    alerts = orch._diagnose(_exec_result_perception())
    assert not [a for a in alerts if a["type"] == "execution_rejected"]


def test_reflect_records_execution_result():
    orch = QuantAGIOrchestrator(config={})
    orch._last_execution_result = {"deployed": 1, "queued": 0, "rejected": 1, "notified": 0}
    report = {"reflection": {}, "cycle": 1}
    orch._reflect(report, {"equity": 1000.0, "contribution": {"total_trades": 10}}, [])
    assert report["reflection"]["execution_result"]["rejected"] == 1


# ── 账户状态机感知（equity_status）────────────────────────

class FakeEquityMonitor:
    def __init__(self, mode="normal"):
        self._mode = mode

    def get_equity_status(self):
        return {
            "mode": self._mode,
            "current_equity": 1000.0,
            "peak_equity": 1000.0,
            "max_drawdown_pct": 0.05,
        }


def test_perceive_equity_status_with_monitor():
    orch = QuantAGIOrchestrator(config={}, equity_monitor=FakeEquityMonitor("growth"))
    assert orch._perceive_equity_status()["mode"] == "growth"


def test_perceive_equity_status_no_monitor():
    orch = QuantAGIOrchestrator(config={})
    assert orch._perceive_equity_status() == {}


def test_diagnose_equity_emergency_alert():
    orch = QuantAGIOrchestrator(config={})
    perception = {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "equity_status": {"mode": "emergency"},
    }
    alerts = orch._diagnose(perception)
    emergency = [a for a in alerts if a["type"] == "equity_emergency"]
    assert len(emergency) == 1
    assert emergency[0]["level"] == "critical"


def test_diagnose_equity_recovery_alert():
    orch = QuantAGIOrchestrator(config={})
    perception = {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "equity_status": {"mode": "recovery"},
    }
    alerts = orch._diagnose(perception)
    recovery = [a for a in alerts if a["type"] == "equity_recovery"]
    assert len(recovery) == 1


def test_offensive_skipped_on_decline_mode():
    orch = _offensive_orch()
    perception = _offensive_perception()  # trend_bullish + strength 0.7 + healthy A
    perception["equity_status"] = {"mode": "decline"}
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


def test_offensive_allowed_on_normal_mode():
    orch = _offensive_orch()
    perception = _offensive_perception()
    perception["equity_status"] = {"mode": "normal"}
    alerts = orch._diagnose(perception)
    assert [a for a in alerts if a["type"] == "offensive_opportunity"]


# ── 组合级分散化响应（diversification）────────────────────

def _diversification_orch(**overrides):
    cfg = {"agi_orchestrator": {"diversification": {
        "enabled": True, "reduce_target": 0.2,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["diversification"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_diversification_actions_generate():
    orch = _diversification_orch()
    decision = {
        "allocation_plan": {"strategy_allocations": {
            "grid": {"target_weight": 0.6},
            "trend": {"target_weight": 0.1},
        }},
    }
    alerts = [{"type": "high_concentration", "concentration": 0.7}]
    actions = orch._diversification_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["strategy"] == "grid"  # 最高权重策略
    assert actions[0]["action"] == "decrease"
    assert actions[0]["target_allocation"] == 0.2


def test_diversification_skip_when_already_diversified():
    orch = _diversification_orch()
    decision = {
        "allocation_plan": {"strategy_allocations": {
            "grid": {"target_weight": 0.15},
            "trend": {"target_weight": 0.15},
        }},
    }
    alerts = [{"type": "high_concentration", "concentration": 0.7}]
    assert orch._diversification_actions(decision, alerts) == []


def test_diversification_disabled():
    orch = _diversification_orch(enabled=False)
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.6}}}}
    alerts = [{"type": "high_concentration"}]
    assert orch._diversification_actions(decision, alerts) == []


def test_diversification_ignores_non_concentration_alerts():
    orch = _diversification_orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.6}}}}
    alerts = [{"type": "low_diversification", "diversification_score": 0.1}]
    assert orch._diversification_actions(decision, alerts) == []


# ── 进攻观察期冷却（offensive cooldown）───────────────────

def test_offensive_cooldown_skips_within_interval():
    orch = _offensive_orch()
    orch._cycle_count = 5
    orch._last_offensive_cycle = {"grid": 3}  # 2 周期前进攻过，min_interval=5
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []  # 观察期内，跳过


def test_offensive_cooldown_allows_after_interval():
    orch = _offensive_orch()
    orch._cycle_count = 10
    orch._last_offensive_cycle = {"grid": 3}  # 7 周期前进攻过，min_interval=5
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert orch._last_offensive_cycle["grid"] == 10  # 更新为本次 cycle


def test_offensive_cooldown_first_time_allowed():
    orch = _offensive_orch()
    orch._cycle_count = 5
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert orch._last_offensive_cycle["grid"] == 5


# ── 进攻决策效果归因（offensive attribution）─────────────

def test_offensive_attribution_skips_when_pnl_worsened():
    orch = _offensive_orch()
    orch._cycle_count = 10
    orch._offensive_attribution = {"grid": 100.0}  # 上次进攻时累计盈亏 100
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 80.0}},  # 转差（80 < 100）
    }
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert actions == []  # 上次决策错误，不追高加仓


def test_offensive_attribution_allows_when_pnl_improved():
    orch = _offensive_orch()
    orch._cycle_count = 10
    orch._offensive_attribution = {"grid": 100.0}
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 120.0}},  # 改善（120 > 100）
    }
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert orch._offensive_attribution["grid"] == 120.0  # 更新基线


def test_offensive_attribution_first_time_allowed():
    orch = _offensive_orch()
    orch._cycle_count = 10
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}},
        "strategy_metrics": {"grid": {"total_pnl": 50.0}},
    }
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1
    assert orch._offensive_attribution["grid"] == 50.0


# ── 组合级相关性感知与响应（correlation）──────────────────

class FakeStrategyCorrelation:
    def __init__(self, summary):
        self._summary = summary

    def get_summary(self):
        return self._summary


def _correlation_orch(**overrides):
    cfg = {"agi_orchestrator": {"correlation": {
        "enabled": True, "high_corr_threshold": 0.7, "reduce_target": 0.2,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["correlation"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _ready_summary(max_pair=0.8, avg=0.5, erosion=False):
    return {
        "status": "ready",
        "average_correlation": avg,
        "max_pair_correlation": max_pair,
        "correlation_regime": "high",
        "effective_n": 2.5,
        "diversification_erosion": erosion,
    }


def test_perceive_correlation_no_dependency():
    orch = QuantAGIOrchestrator(config={})
    assert orch._perceive_correlation() == {}
    assert orch._last_correlation == {}


def test_perceive_correlation_ready_summary():
    orch = QuantAGIOrchestrator(
        config={}, strategy_correlation=FakeStrategyCorrelation(_ready_summary())
    )
    summary = orch._perceive_correlation()
    assert summary["max_pair_correlation"] == 0.8
    assert summary["average_correlation"] == 0.5
    assert orch._last_correlation["correlation_regime"] == "high"


def test_perceive_correlation_no_data_status():
    orch = QuantAGIOrchestrator(
        config={}, strategy_correlation=FakeStrategyCorrelation({"status": "no_data"})
    )
    assert orch._perceive_correlation() == {}


def test_diagnose_high_correlation_alert():
    orch = _correlation_orch()
    perception = {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "correlation": _ready_summary(max_pair=0.8),
    }
    alerts = orch._diagnose(perception)
    high = [a for a in alerts if a["type"] == "high_correlation"]
    assert len(high) == 1
    assert high[0]["max_pair_correlation"] == 0.8


def test_diagnose_correlation_below_threshold_no_alert():
    orch = _correlation_orch()
    perception = {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "correlation": _ready_summary(max_pair=0.3),
    }
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_correlation"]


def test_diagnose_diversification_eroding_alert():
    orch = _correlation_orch()
    perception = {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "correlation": _ready_summary(max_pair=0.3, erosion=True),
    }
    alerts = orch._diagnose(perception)
    eroding = [a for a in alerts if a["type"] == "diversification_eroding"]
    assert len(eroding) == 1


def test_diagnose_correlation_disabled_no_alert():
    orch = _correlation_orch(enabled=False)
    perception = {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "correlation": _ready_summary(max_pair=0.8),
    }
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "high_correlation"]


def test_correlation_response_actions_generate():
    orch = _correlation_orch()
    decision = {
        "allocation_plan": {"strategy_allocations": {
            "grid": {"target_weight": 0.6},
            "trend": {"target_weight": 0.1},
        }},
    }
    alerts = [{"type": "high_correlation", "max_pair_correlation": 0.8}]
    actions = orch._correlation_response_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["strategy"] == "grid"  # 最高权重策略
    assert actions[0]["action"] == "decrease"
    assert actions[0]["target_allocation"] == 0.2


def test_correlation_response_triggered_by_erosion():
    orch = _correlation_orch()
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.5}}},
    }
    alerts = [{"type": "diversification_eroding", "effective_n": 2.0}]
    actions = orch._correlation_response_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["action"] == "decrease"


def test_correlation_response_skip_when_already_low_weight():
    orch = _correlation_orch()
    decision = {
        "allocation_plan": {"strategy_allocations": {
            "grid": {"target_weight": 0.1},
            "trend": {"target_weight": 0.1},
        }},
    }
    alerts = [{"type": "high_correlation", "max_pair_correlation": 0.8}]
    assert orch._correlation_response_actions(decision, alerts) == []


def test_correlation_response_ignores_unrelated_alerts():
    orch = _correlation_orch()
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.6}}},
    }
    alerts = [{"type": "high_concentration", "concentration": 0.7}]
    assert orch._correlation_response_actions(decision, alerts) == []


def test_correlation_response_disabled():
    orch = _correlation_orch(enabled=False)
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.6}}}}
    alerts = [{"type": "high_correlation", "max_pair_correlation": 0.8}]
    assert orch._correlation_response_actions(decision, alerts) == []


# ── 成本意识调仓门控（cost_guard）────────────────────────

def _cost_guard_orch(**overrides):
    cfg = {"agi_orchestrator": {"cost_guard": {
        "enabled": True, "min_delta": 0.02,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["cost_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _cost_guard_decision(allocations):
    return {"allocation_plan": {"strategy_allocations": allocations}}


def test_cost_guard_drops_tiny_rebalance():
    orch = _cost_guard_orch()
    decision = _cost_guard_decision({"grid": {"target_weight": 0.20}})
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.21},
    ]
    kept = orch._apply_cost_guard(actions, decision)
    assert kept == []  # delta 0.01 < 0.02 → 丢弃
    assert orch._last_cost_guard["dropped_count"] == 1
    assert orch._last_cost_guard["dropped"][0]["delta"] == pytest.approx(0.01)


def test_cost_guard_keeps_significant_rebalance():
    orch = _cost_guard_orch()
    decision = _cost_guard_decision({"grid": {"target_weight": 0.20}})
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.25},
    ]
    kept = orch._apply_cost_guard(actions, decision)
    assert len(kept) == 1  # delta 0.05 >= 0.02 → 保留
    assert orch._last_cost_guard["dropped_count"] == 0


def test_cost_guard_ignores_non_reallocate():
    orch = _cost_guard_orch()
    decision = _cost_guard_decision({})
    actions = [
        {"type": "profit_take_close", "close_ratio": 0.3},
        {"type": "param_adjust", "strategy": "grid", "value": 1.0},
    ]
    kept = orch._apply_cost_guard(actions, decision)
    assert len(kept) == 2  # 非调仓动作不受影响


def test_cost_guard_passthrough_when_current_unknown():
    orch = _cost_guard_orch()
    decision = _cost_guard_decision({})  # grid 不在 allocations
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.3},
    ]
    kept = orch._apply_cost_guard(actions, decision)
    assert len(kept) == 1  # 无法读取当前权重 → 放行（向后兼容）


def test_cost_guard_disabled_passthrough():
    orch = _cost_guard_orch(enabled=False)
    decision = _cost_guard_decision({"grid": {"target_weight": 0.20}})
    actions = [
        {"type": "reallocate", "strategy": "grid", "action": "increase", "target_allocation": 0.21},
    ]
    kept = orch._apply_cost_guard(actions, decision)
    assert len(kept) == 1
    assert orch._last_cost_guard is None


def test_act_integrates_cost_guard():
    orch = QuantAGIOrchestrator(config={"agi_orchestrator": {
        "cost_guard": {"enabled": True, "min_delta": 0.02},
        "risk_response": {"enabled": True, "reduce_target": 0.1},
    }})
    decision = {
        "allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.20}}},
        "reallocation_suggestions": [],
        "strategy_names": ["grid"],
        "strategy_metrics": {},
    }
    # strategy_declining → 减配到 0.1，delta=0.1 显著，保留
    alerts = [{"type": "strategy_declining", "strategy": "grid", "message": "趋势恶化"}]
    actions = orch._act(decision, alerts)
    realloc = [a for a in actions if a["type"] == "reallocate"]
    assert len(realloc) == 1
    assert orch._last_cost_guard["dropped_count"] == 0


# ── 策略健康度趋势外推（health_trend_guard）──────────────

def _health_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"health_trend_guard": {
        "enabled": True, "window": 3, "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["health_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _health_trend_perception(health_score):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": health_score,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": 10,
            "total_pnl": 50.0,
        }}},
        "freeze_state": {},
    }


def test_health_trend_detects_decline():
    orch = _health_trend_orch()
    orch._strategy_health_history["grid"] = deque([70, 65], maxlen=3)
    alerts = orch._diagnose(_health_trend_perception(health_score=60))
    deteriorating = [a for a in alerts if a["type"] == "health_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"


def test_health_trend_no_alert_when_rising():
    orch = _health_trend_orch()
    orch._strategy_health_history["grid"] = deque([60, 65], maxlen=3)
    alerts = orch._diagnose(_health_trend_perception(health_score=70))
    assert not [a for a in alerts if a["type"] == "health_deteriorating"]


def test_health_trend_insufficient_samples():
    orch = _health_trend_orch()
    orch._strategy_health_history["grid"] = deque([65], maxlen=3)
    alerts = orch._diagnose(_health_trend_perception(health_score=60))
    assert not [a for a in alerts if a["type"] == "health_deteriorating"]


def test_health_trend_disabled():
    orch = _health_trend_orch(enabled=False)
    alerts = orch._diagnose(_health_trend_perception(health_score=60))
    assert not [a for a in alerts if a["type"] == "health_deteriorating"]


def test_health_trend_actions_generate():
    orch = _health_trend_orch()
    alerts = [{"type": "health_deteriorating", "strategy": "grid", "message": "连续下降"}]
    actions = orch._health_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_health_trend_actions_ignore_other_alerts():
    orch = _health_trend_orch()
    alerts = [
        {"type": "strategy_declining", "strategy": "grid"},
        {"type": "strategy_health_critical", "strategy": "grid"},
    ]
    assert orch._health_trend_actions(alerts) == []


def test_health_trend_actions_dedup():
    orch = _health_trend_orch()
    alerts = [
        {"type": "health_deteriorating", "strategy": "grid"},
        {"type": "health_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._health_trend_actions(alerts)) == 1


def test_health_trend_actions_disabled():
    orch = _health_trend_orch(enabled=False)
    alerts = [{"type": "health_deteriorating", "strategy": "grid"}]
    assert orch._health_trend_actions(alerts) == []


# ── 市场状态突变进攻冷却（regime_shift_guard）────────────

def _regime_shift_guard_orch(**overrides):
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "regime_shift_guard": {"enabled": True, "cooldown_cycles": 3},
    }}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["regime_shift_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _regime_shift_perception(regime, confidence):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": regime, "strength": 0.7, "confidence": confidence},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "equity_status": {},
        "correlation": {},
    }


def test_regime_shift_records_cycle():
    orch = _regime_shift_guard_orch()
    orch._last_regime = "range_bound"
    orch._cycle_count = 7
    alerts = orch._diagnose(_regime_shift_perception("trend_bullish", 0.8))
    assert [a for a in alerts if a["type"] == "regime_shift"]
    assert orch._last_regime_shift_cycle == 7


def test_regime_shift_low_confidence_no_record():
    orch = _regime_shift_guard_orch()
    orch._last_regime = "range_bound"
    orch._cycle_count = 7
    alerts = orch._diagnose(_regime_shift_perception("trend_bullish", 0.3))
    assert [a for a in alerts if a["type"] == "regime_shift_low_confidence"]
    assert orch._last_regime_shift_cycle is None


def test_offensive_cooled_after_regime_shift():
    orch = _regime_shift_guard_orch()
    orch._cycle_count = 5
    orch._last_regime_shift_cycle = 3  # 2 周期前突变，cooldown=3
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_allowed_after_cooldown():
    orch = _regime_shift_guard_orch()
    orch._cycle_count = 7
    orch._last_regime_shift_cycle = 3  # 4 周期前，冷却期已过
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    assert len(actions) == 1


def test_offensive_not_cooled_when_no_shift():
    orch = _regime_shift_guard_orch()
    orch._cycle_count = 5
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


def test_offensive_not_cooled_when_guard_disabled():
    orch = _regime_shift_guard_orch(enabled=False)
    orch._cycle_count = 5
    orch._last_regime_shift_cycle = 3
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


# ── 策略级利润回吐保护（give_back_guard）──────────────────

def _give_back_orch(**overrides):
    cfg = {"agi_orchestrator": {"give_back_guard": {
        "enabled": True, "give_back_threshold": 0.2, "reduce_target": 0.1,
    }, "state_path": "data/__test_isolated_state__.json"}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["give_back_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _give_back_perception(total_pnl):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "total_pnl": total_pnl,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": 10,
        }}},
        "freeze_state": {},
    }


def test_give_back_detects_drawdown():
    orch = _give_back_orch()
    orch._strategy_pnl_peak["grid"] = 100.0
    alerts = orch._diagnose(_give_back_perception(total_pnl=70.0))  # 回吐 30% > 20%
    give_back = [a for a in alerts if a["type"] == "pnl_give_back"]
    assert len(give_back) == 1
    assert give_back[0]["strategy"] == "grid"
    assert give_back[0]["peak_pnl"] == 100.0


def test_give_back_updates_peak():
    orch = _give_back_orch()
    orch._strategy_pnl_peak["grid"] = 100.0
    alerts = orch._diagnose(_give_back_perception(total_pnl=120.0))  # 创新高
    assert not [a for a in alerts if a["type"] == "pnl_give_back"]
    assert orch._strategy_pnl_peak["grid"] == 120.0


def test_give_back_no_alert_when_no_peak():
    orch = _give_back_orch()
    alerts = orch._diagnose(_give_back_perception(total_pnl=-10.0))  # 首见负值，peak=0
    assert not [a for a in alerts if a["type"] == "pnl_give_back"]
    assert orch._strategy_pnl_peak["grid"] == 0.0


def test_give_back_disabled():
    orch = _give_back_orch(enabled=False)
    orch._strategy_pnl_peak["grid"] = 100.0
    alerts = orch._diagnose(_give_back_perception(total_pnl=70.0))
    assert not [a for a in alerts if a["type"] == "pnl_give_back"]


def test_give_back_actions_generate():
    orch = _give_back_orch()
    alerts = [{"type": "pnl_give_back", "strategy": "grid", "message": "回吐"}]
    actions = orch._give_back_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["target_allocation"] == 0.1


def test_give_back_actions_ignore_other_alerts():
    orch = _give_back_orch()
    alerts = [{"type": "strategy_declining", "strategy": "grid"}]
    assert orch._give_back_actions(alerts) == []


def test_give_back_actions_dedup():
    orch = _give_back_orch()
    alerts = [
        {"type": "pnl_give_back", "strategy": "grid"},
        {"type": "pnl_give_back", "strategy": "grid"},
    ]
    assert len(orch._give_back_actions(alerts)) == 1


def test_give_back_actions_disabled():
    orch = _give_back_orch(enabled=False)
    alerts = [{"type": "pnl_give_back", "strategy": "grid"}]
    assert orch._give_back_actions(alerts) == []


# ── 利润回吐分级保护（severe / critical escalation）────────

def _give_back_escalation_orch(**overrides):
    cfg = {"agi_orchestrator": {"give_back_guard": {
        "enabled": True, "give_back_threshold": 0.2, "reduce_target": 0.1,
        "severe_enabled": True, "severe_threshold": 0.5,
        "severe_reduce_target": 0.02,
        "critical_enabled": True, "critical_threshold": 0.8,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["give_back_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_give_back_severe_alert_triggers():
    """回吐 60%（>50% severe）→ 同时触发 pnl_give_back + pnl_give_back_severe。"""
    orch = _give_back_escalation_orch()
    orch._strategy_pnl_peak["grid"] = 100.0
    alerts = orch._diagnose(_give_back_perception(total_pnl=40.0))  # 回吐 60%
    types = {a["type"] for a in alerts}
    assert "pnl_give_back" in types
    assert "pnl_give_back_severe" in types
    assert "pnl_give_back_critical" not in types  # 60% < 80%


def test_give_back_critical_alert_triggers():
    """回吐 90%（>80% critical）→ 触发全部三级告警。"""
    orch = _give_back_escalation_orch()
    orch._strategy_pnl_peak["grid"] = 100.0
    alerts = orch._diagnose(_give_back_perception(total_pnl=10.0))  # 回吐 90%
    types = {a["type"] for a in alerts}
    assert "pnl_give_back" in types
    assert "pnl_give_back_severe" in types
    assert "pnl_give_back_critical" in types


def test_give_back_severe_action_reduces_to_2pct():
    """严重回吐 → reallocate decrease 到 severe_reduce_target (2%)。"""
    orch = _give_back_escalation_orch()
    alerts = [{"type": "pnl_give_back_severe", "strategy": "grid"}]
    actions = orch._give_back_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["target_allocation"] == 0.02


def test_give_back_critical_action_freeze():
    """危急回吐 → reallocate 到 0.0 + strategy_pause，并加入 paused_strategies。"""
    orch = _give_back_escalation_orch()
    orch._paused_strategies = set()
    alerts = [{"type": "pnl_give_back_critical", "strategy": "grid"}]
    actions = orch._give_back_actions(alerts)
    types = [a["type"] for a in actions]
    assert "reallocate" in types
    assert "strategy_pause" in types
    realloc = [a for a in actions if a["type"] == "reallocate"][0]
    assert realloc["target_allocation"] == 0.0
    assert "grid" in orch._paused_strategies


def test_give_back_escalation_picks_worst_tier():
    """同一策略同时有 basic + severe + critical → 只取最严重（critical）动作。"""
    orch = _give_back_escalation_orch()
    orch._paused_strategies = set()
    alerts = [
        {"type": "pnl_give_back", "strategy": "grid"},
        {"type": "pnl_give_back_severe", "strategy": "grid"},
        {"type": "pnl_give_back_critical", "strategy": "grid"},
    ]
    actions = orch._give_back_actions(alerts)
    # critical: 1 reallocate + 1 strategy_pause = 2 actions（无重复 basic/severe）
    assert len(actions) == 2
    assert "grid" in orch._paused_strategies


def test_give_back_severe_disabled_falls_back():
    """severe_enabled=False 时 pnl_give_back_severe 回退为 basic 减仓。"""
    orch = _give_back_escalation_orch(severe_enabled=False, critical_enabled=False)
    alerts = [{"type": "pnl_give_back_severe", "strategy": "grid"}]
    actions = orch._give_back_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["target_allocation"] == 0.1  # 回退 reduce_target


# ── 利润回吐峰值生命周期管理（peak lifecycle）────────────

def _peak_lifecycle_orch(**overrides):
    cfg = {"agi_orchestrator": {"give_back_guard": {
        "enabled": True, "give_back_threshold": 0.2, "reduce_target": 0.1,
        "severe_enabled": True, "severe_threshold": 0.5,
        "severe_reduce_target": 0.02,
        "critical_enabled": True, "critical_threshold": 0.8,
        "peak_reset_on_resume": True,
        "peak_reset_on_critical": True,
        "peak_decay_cycles": 0,
    }, "strategy_lifecycle": {
        "enabled": True, "resume_recovered": True,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["give_back_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_peak_reset_on_critical_clears_peak():
    """危急回吐清仓后重置 peak=0，避免旧峰值在恢复后立即重新触发回吐。"""
    orch = _peak_lifecycle_orch()
    orch._strategy_pnl_peak["grid"] = 100.0
    orch._paused_strategies = set()
    alerts = [{"type": "pnl_give_back_critical", "strategy": "grid"}]
    orch._give_back_actions(alerts)
    assert orch._strategy_pnl_peak["grid"] == 0.0


def test_peak_reset_on_resume_clears_peak():
    """策略恢复开仓后重置 peak=0。"""
    orch = _peak_lifecycle_orch()
    orch._strategy_pnl_peak["grid"] = 100.0
    orch._paused_strategies = {"grid"}
    alerts = [{"type": "strategy_recovered", "strategy": "grid", "message": "恢复"}]
    orch._strategy_resume_actions(alerts)
    assert orch._strategy_pnl_peak["grid"] == 0.0
    assert "grid" not in orch._paused_strategies


def test_peak_reset_disabled_preserves_peak():
    """peak_reset_on_critical=False 时保留旧峰值。"""
    orch = _peak_lifecycle_orch(peak_reset_on_critical=False)
    orch._strategy_pnl_peak["grid"] = 100.0
    orch._paused_strategies = set()
    alerts = [{"type": "pnl_give_back_critical", "strategy": "grid"}]
    orch._give_back_actions(alerts)
    assert orch._strategy_pnl_peak["grid"] == 100.0


def test_peak_decay_reduces_stale_peak():
    """peak_decay_cycles=3：第 3 周期起陈旧峰值按 0.9 衰减。"""
    orch = _peak_lifecycle_orch(peak_decay_cycles=3)
    orch._strategy_pnl_peak["grid"] = 100.0
    orch._give_back_peak_last_decay_cycle["grid"] = 0
    orch._cycle_count = 3
    # total_pnl=50 低于衰减后峰值 90，避免被 max(peak, total_pnl) 覆盖
    orch._diagnose(_give_back_perception(total_pnl=50.0))
    assert orch._strategy_pnl_peak["grid"] == pytest.approx(90.0, abs=1e-6)
    # 周期内再次调用不重复衰减
    orch._diagnose(_give_back_perception(total_pnl=50.0))
    assert orch._strategy_pnl_peak["grid"] == pytest.approx(90.0, abs=1e-6)


def test_peak_decay_disabled_no_decay():
    """peak_decay_cycles=0 时峰值不衰减。"""
    orch = _peak_lifecycle_orch()  # 默认 peak_decay_cycles=0
    orch._strategy_pnl_peak["grid"] = 100.0
    orch._cycle_count = 100
    orch._diagnose(_give_back_perception(total_pnl=100.0))
    assert orch._strategy_pnl_peak["grid"] == 100.0


# ── 学习型状态持久化（learning_state）────────────────────

def _learning_state_orch(tmp_path):
    return QuantAGIOrchestrator(
        config={"agi_orchestrator": {"state_path": str(tmp_path / "state.json")}}
    )


def test_learning_state_roundtrip(tmp_path):
    orch = _learning_state_orch(tmp_path)
    orch._offensive_attribution = {"grid": 100.0}
    orch._last_offensive_cycle = {"grid": 5}
    orch._offensive_stop_loss_cooldown = {"grid": 20}
    orch._offensive_profit_take_cooldown = {"grid": 15}
    orch._offensive_attribution_cycle = {"grid": 5}
    orch._offensive_consecutive = {"grid": 2}
    orch._strategy_pnl_peak = {"grid": 50.0}
    orch._strategy_health_history = {"grid": deque([70.0, 65.0], maxlen=3)}
    orch._last_regime_shift_cycle = 3

    orch._persist_state({"reflection": {}, "cycle": 1})

    orch2 = _learning_state_orch(tmp_path)  # 同一 state_path 重新加载
    assert orch2._offensive_attribution == {"grid": 100.0}
    assert orch2._last_offensive_cycle == {"grid": 5}
    assert orch2._offensive_stop_loss_cooldown == {"grid": 20}
    assert orch2._offensive_profit_take_cooldown == {"grid": 15}
    assert orch2._offensive_attribution_cycle == {"grid": 5}
    assert orch2._offensive_consecutive == {"grid": 2}
    assert orch2._strategy_pnl_peak == {"grid": 50.0}
    assert list(orch2._strategy_health_history["grid"]) == [70.0, 65.0]
    assert orch2._last_regime_shift_cycle == 3


def test_learning_state_no_file_defaults(tmp_path):
    orch = _learning_state_orch(tmp_path)  # state.json 尚不存在
    assert orch._offensive_attribution == {}
    assert orch._last_offensive_cycle == {}
    assert orch._offensive_stop_loss_cooldown == {}
    assert orch._offensive_profit_take_cooldown == {}
    assert orch._offensive_attribution_cycle == {}
    assert orch._offensive_consecutive == {}
    assert orch._strategy_pnl_peak == {}
    assert orch._strategy_health_history == {}
    assert orch._last_regime_shift_cycle is None


def test_serialize_learning_state_json_safe(tmp_path):
    orch = _learning_state_orch(tmp_path)
    orch._offensive_attribution = {"grid": 100.0}
    orch._strategy_health_history = {"grid": deque([70.0, 65.0], maxlen=3)}
    serialized = orch._serialize_learning_state()
    text = json.dumps(serialized, allow_nan=False)  # NaN/Inf 会抛 ValueError
    assert "NaN" not in text
    assert "Infinity" not in text


def test_learning_state_corrupt_file_fails_soft(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text("{ not valid json", encoding="utf-8")
    orch = QuantAGIOrchestrator(config={"agi_orchestrator": {"state_path": str(state_path)}})
    # 损坏文件应静默失败，状态保持默认，不抛异常
    assert orch._offensive_attribution == {}
    assert orch._last_regime_shift_cycle is None


# ── 交易成本感知与响应（cost_awareness）──────────────────

def _cost_awareness_orch(**overrides):
    cfg = {"agi_orchestrator": {"cost_awareness": {
        "enabled": True, "fee_ratio_threshold": 0.3,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["cost_awareness"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _cost_awareness_offensive_orch(**overrides):
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "cost_awareness": {"enabled": True, "fee_ratio_threshold": 0.3},
    }}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["cost_awareness"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _cost_awareness_perception(total_pnl, total_fees):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {
            "strategies": {},
            "total_pnl": total_pnl,
            "total_fees": total_fees,
        },
        "freeze_state": {},
    }


def test_cost_awareness_detects_high_fee_ratio():
    orch = _cost_awareness_orch()
    # pnl=70, fees=30 → gross=100, fee_ratio=0.3 ≥ 0.3
    alerts = orch._diagnose(_cost_awareness_perception(total_pnl=70.0, total_fees=30.0))
    high = [a for a in alerts if a["type"] == "high_trading_cost"]
    assert len(high) == 1
    assert high[0]["fee_ratio"] == pytest.approx(0.3)


def test_cost_awareness_no_alert_low_fee():
    orch = _cost_awareness_orch()
    # pnl=90, fees=10 → gross=100, fee_ratio=0.1 < 0.3
    alerts = orch._diagnose(_cost_awareness_perception(total_pnl=90.0, total_fees=10.0))
    assert not [a for a in alerts if a["type"] == "high_trading_cost"]


def test_cost_awareness_no_alert_when_loss():
    orch = _cost_awareness_orch()
    # pnl=-50, fees=30 → gross=-20 < 0，无毛利可侵蚀
    alerts = orch._diagnose(_cost_awareness_perception(total_pnl=-50.0, total_fees=30.0))
    assert not [a for a in alerts if a["type"] == "high_trading_cost"]


def test_cost_awareness_disabled():
    orch = _cost_awareness_orch(enabled=False)
    alerts = orch._diagnose(_cost_awareness_perception(total_pnl=70.0, total_fees=30.0))
    assert not [a for a in alerts if a["type"] == "high_trading_cost"]


def test_offensive_suppressed_by_high_trading_cost():
    orch = _cost_awareness_offensive_orch()
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "high_trading_cost", "fee_ratio": 0.3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_without_cost_alert():
    orch = _cost_awareness_offensive_orch()
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


def test_offensive_not_suppressed_when_cost_awareness_disabled():
    orch = _cost_awareness_offensive_orch(enabled=False)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "high_trading_cost", "fee_ratio": 0.3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


# ── 追涨抑制（momentum_guard）────────────────────────────

def _momentum_guard_orch(**overrides):
    cfg = {"agi_orchestrator": {"momentum_guard": {
        "enabled": True, "max_consecutive_up": 3,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["momentum_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _momentum_offensive_orch(**overrides):
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "momentum_guard": {"enabled": True, "max_consecutive_up": 3},
    }}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["momentum_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _momentum_perception(consecutive_up):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "equity_status": {"mode": "normal", "consecutive_up": consecutive_up},
    }


def test_momentum_detects_overheated():
    orch = _momentum_guard_orch()
    alerts = orch._diagnose(_momentum_perception(consecutive_up=3))
    over = [a for a in alerts if a["type"] == "market_overheated"]
    assert len(over) == 1
    assert over[0]["consecutive_up"] == 3


def test_momentum_no_alert_below_threshold():
    orch = _momentum_guard_orch()
    alerts = orch._diagnose(_momentum_perception(consecutive_up=2))
    assert not [a for a in alerts if a["type"] == "market_overheated"]


def test_momentum_disabled():
    orch = _momentum_guard_orch(enabled=False)
    alerts = orch._diagnose(_momentum_perception(consecutive_up=3))
    assert not [a for a in alerts if a["type"] == "market_overheated"]


def test_offensive_suppressed_by_overheated():
    orch = _momentum_offensive_orch()
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "market_overheated", "consecutive_up": 3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_without_overheated():
    orch = _momentum_offensive_orch()
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


def test_offensive_not_suppressed_when_momentum_disabled():
    orch = _momentum_offensive_orch(enabled=False)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "market_overheated", "consecutive_up": 3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


# ── 资金利用率过高收敛（utilization_guard）────────────────

def _utilization_guard_orch(**overrides):
    cfg = {"agi_orchestrator": {"utilization_guard": {
        "enabled": True, "high_utilization_threshold": 0.8,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["utilization_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _utilization_offensive_orch(**overrides):
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "utilization_guard": {"enabled": True, "high_utilization_threshold": 0.8},
    }}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["utilization_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _utilization_perception(utilization):
    return {
        "equity": 1000.0,
        "used_margin": utilization * 1000.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
    }


def test_utilization_detects_high():
    orch = _utilization_guard_orch()
    alerts = orch._diagnose(_utilization_perception(utilization=0.9))
    high = [a for a in alerts if a["type"] == "high_capital_utilization"]
    assert len(high) == 1
    assert high[0]["utilization"] == pytest.approx(0.9)


def test_utilization_no_alert_below_threshold():
    orch = _utilization_guard_orch()
    alerts = orch._diagnose(_utilization_perception(utilization=0.7))
    assert not [a for a in alerts if a["type"] == "high_capital_utilization"]


def test_utilization_disabled():
    orch = _utilization_guard_orch(enabled=False)
    alerts = orch._diagnose(_utilization_perception(utilization=0.9))
    assert not [a for a in alerts if a["type"] == "high_capital_utilization"]


def test_offensive_suppressed_by_high_utilization():
    orch = _utilization_offensive_orch()
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "high_capital_utilization", "utilization": 0.9},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_without_utilization_alert():
    orch = _utilization_offensive_orch()
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


def test_offensive_not_suppressed_when_utilization_disabled():
    orch = _utilization_offensive_orch(enabled=False)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "high_capital_utilization", "utilization": 0.9},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


# ── 资金利用率过高主动降仓（utilization_reduce_actions）────────────────

def test_utilization_reduce_generates_decrease():
    orch = _utilization_guard_orch()
    alerts = [{"type": "high_capital_utilization", "utilization": 0.9}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._utilization_reduce_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_utilization_reduce_no_alert():
    orch = _utilization_guard_orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._utilization_reduce_actions(decision, []) == []


def test_utilization_reduce_disabled():
    orch = _utilization_guard_orch(enabled=False)
    alerts = [{"type": "high_capital_utilization", "utilization": 0.9}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._utilization_reduce_actions(decision, alerts) == []


def test_utilization_reduce_skips_when_already_low():
    orch = _utilization_guard_orch()
    alerts = [{"type": "high_capital_utilization", "utilization": 0.9}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.1}}}}
    assert orch._utilization_reduce_actions(decision, alerts) == []


def test_utilization_reduce_no_allocations():
    orch = _utilization_guard_orch()
    alerts = [{"type": "high_capital_utilization", "utilization": 0.9}]
    decision = {"allocation_plan": {}}
    assert orch._utilization_reduce_actions(decision, alerts) == []


# ── 策略级浮亏止损（unrealized_loss_guard）────────────────

def _unrealized_loss_orch(**overrides):
    cfg = {"agi_orchestrator": {"unrealized_loss_guard": {
        "enabled": True, "loss_threshold": 0.03, "reduce_target": 0.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["unrealized_loss_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _unrealized_loss_perception(strategies):
    return {
        "equity": 1000.0,
        "used_margin": 500.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": strategies},
        "freeze_state": {},
    }


def test_unrealized_loss_detects_high():
    orch = _unrealized_loss_orch()
    perception = _unrealized_loss_perception({"grid": {"unrealized_pnl": -50.0}})
    alerts = orch._diagnose(perception)
    loss = [a for a in alerts if a["type"] == "strategy_floating_loss"]
    assert len(loss) == 1
    assert loss[0]["strategy"] == "grid"
    assert loss[0]["loss_pct"] == pytest.approx(0.05)


def test_unrealized_loss_below_threshold():
    orch = _unrealized_loss_orch()
    perception = _unrealized_loss_perception({"grid": {"unrealized_pnl": -10.0}})
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "strategy_floating_loss"]


def test_unrealized_loss_positive_ignored():
    orch = _unrealized_loss_orch()
    perception = _unrealized_loss_perception({"grid": {"unrealized_pnl": 30.0}})
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "strategy_floating_loss"]


def test_unrealized_loss_disabled():
    orch = _unrealized_loss_orch(enabled=False)
    perception = _unrealized_loss_perception({"grid": {"unrealized_pnl": -50.0}})
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "strategy_floating_loss"]


def test_unrealized_loss_actions_generate_stop():
    orch = _unrealized_loss_orch()
    alerts = [{"type": "strategy_floating_loss", "strategy": "grid", "unrealized_pnl": -50.0}]
    actions = orch._unrealized_loss_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == 0.0


def test_unrealized_loss_actions_dedup():
    orch = _unrealized_loss_orch()
    alerts = [
        {"type": "strategy_floating_loss", "strategy": "grid", "unrealized_pnl": -50.0},
        {"type": "strategy_floating_loss", "strategy": "grid", "unrealized_pnl": -60.0},
    ]
    assert len(orch._unrealized_loss_actions(alerts)) == 1


def test_unrealized_loss_actions_disabled():
    orch = _unrealized_loss_orch(enabled=False)
    alerts = [{"type": "strategy_floating_loss", "strategy": "grid", "unrealized_pnl": -50.0}]
    assert orch._unrealized_loss_actions(alerts) == []


# ── 浮盈占比过高检测（unrealized_profit_ratio_guard）────────────────

def _unrealized_profit_ratio_orch(**overrides):
    cfg = {"agi_orchestrator": {"unrealized_profit_ratio_guard": {
        "enabled": True, "ratio_threshold": 0.8, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["unrealized_profit_ratio_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _unrealized_profit_ratio_perception(strategies):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": strategies},
        "freeze_state": {},
    }


def test_unrealized_profit_ratio_detects_high():
    # 浮盈 90 / 总盈亏 100 = 0.9 > 0.8 → 盈利脆弱性预警
    orch = _unrealized_profit_ratio_orch()
    perception = _unrealized_profit_ratio_perception(
        {"grid": {"total_pnl": 100.0, "unrealized_pnl": 90.0}}
    )
    alerts = orch._diagnose(perception)
    hits = [a for a in alerts if a["type"] == "unrealized_profit_concentration"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"


def test_unrealized_profit_ratio_below_threshold():
    # 浮盈 50 / 100 = 0.5 < 0.8 → 不触发
    orch = _unrealized_profit_ratio_orch()
    perception = _unrealized_profit_ratio_perception(
        {"grid": {"total_pnl": 100.0, "unrealized_pnl": 50.0}}
    )
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "unrealized_profit_concentration"]


def test_unrealized_profit_ratio_negative_pnl_ignored():
    # 总盈亏非正 → 不触发（盈利脆弱性只对盈利策略有意义）
    orch = _unrealized_profit_ratio_orch()
    perception = _unrealized_profit_ratio_perception(
        {"grid": {"total_pnl": -100.0, "unrealized_pnl": 90.0}}
    )
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "unrealized_profit_concentration"]


def test_unrealized_profit_ratio_disabled():
    orch = _unrealized_profit_ratio_orch(enabled=False)
    perception = _unrealized_profit_ratio_perception(
        {"grid": {"total_pnl": 100.0, "unrealized_pnl": 90.0}}
    )
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "unrealized_profit_concentration"]


def test_unrealized_profit_ratio_actions_generate():
    orch = _unrealized_profit_ratio_orch()
    alerts = [{
        "type": "unrealized_profit_concentration", "strategy": "grid",
        "unrealized_pnl": 90.0, "total_pnl": 100.0,
    }]
    actions = orch._unrealized_profit_ratio_actions({}, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_unrealized_profit_ratio_actions_dedup():
    orch = _unrealized_profit_ratio_orch()
    alerts = [
        {"type": "unrealized_profit_concentration", "strategy": "grid"},
        {"type": "unrealized_profit_concentration", "strategy": "grid"},
    ]
    assert len(orch._unrealized_profit_ratio_actions({}, alerts)) == 1


def test_unrealized_profit_ratio_actions_disabled():
    orch = _unrealized_profit_ratio_orch(enabled=False)
    alerts = [{"type": "unrealized_profit_concentration", "strategy": "grid"}]
    assert orch._unrealized_profit_ratio_actions({}, alerts) == []


def test_offensive_suppressed_by_unrealized_profit_ratio():
    # 浮盈占比过高的策略不进攻加仓（per-strategy gate，收益侧盈利质量维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "unrealized_profit_ratio_guard": {
            "enabled": True, "ratio_threshold": 0.8, "reduce_target": 0.1,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "unrealized_profit_concentration", "strategy": "sync"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 策略级浮亏加深趋势外推（unrealized_loss_trend_guard）──────────────────

def _unrealized_loss_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"unrealized_loss_trend_guard": {
        "enabled": True, "window": 3, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["unrealized_loss_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _unrealized_loss_trend_perception(strategies):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": strategies},
        "freeze_state": {},
    }


def test_unrealized_loss_trend_detects_deepening():
    # 浮亏连续 3 周期加深（0→-1→-2）→ unrealized_loss_deteriorating
    orch = _unrealized_loss_trend_orch()
    orch._strategy_unrealized_loss_history["grid"] = deque([0.0, -1.0], maxlen=3)
    alerts = orch._diagnose(_unrealized_loss_trend_perception(
        {"grid": {"unrealized_pnl": -2.0}}
    ))
    hits = [a for a in alerts if a["type"] == "unrealized_loss_deteriorating"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["unrealized_pnl"] == pytest.approx(-2.0)


def test_unrealized_loss_trend_no_alert_when_positive():
    # 当前处于浮盈（unrealized_pnl>0）→ 即使下降也不触发（give_back 负责收益侧）
    orch = _unrealized_loss_trend_orch()
    orch._strategy_unrealized_loss_history["grid"] = deque([3.0, 2.0], maxlen=3)
    alerts = orch._diagnose(_unrealized_loss_trend_perception(
        {"grid": {"unrealized_pnl": 1.0}}
    ))
    assert not [a for a in alerts if a["type"] == "unrealized_loss_deteriorating"]


def test_unrealized_loss_trend_no_alert_when_improving():
    # 浮亏收敛（回升）→ 不触发
    orch = _unrealized_loss_trend_orch()
    orch._strategy_unrealized_loss_history["grid"] = deque([-3.0, -2.0], maxlen=3)
    alerts = orch._diagnose(_unrealized_loss_trend_perception(
        {"grid": {"unrealized_pnl": -1.0}}
    ))
    assert not [a for a in alerts if a["type"] == "unrealized_loss_deteriorating"]


def test_unrealized_loss_trend_disabled():
    orch = _unrealized_loss_trend_orch(enabled=False)
    orch._strategy_unrealized_loss_history["grid"] = deque([0.0, -1.0], maxlen=3)
    alerts = orch._diagnose(_unrealized_loss_trend_perception(
        {"grid": {"unrealized_pnl": -2.0}}
    ))
    assert not [a for a in alerts if a["type"] == "unrealized_loss_deteriorating"]


def test_unrealized_loss_trend_actions_generate():
    orch = _unrealized_loss_trend_orch()
    alerts = [{"type": "unrealized_loss_deteriorating", "strategy": "grid", "message": "连续加深"}]
    actions = orch._unrealized_loss_trend_actions({}, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_unrealized_loss_trend_actions_dedup():
    orch = _unrealized_loss_trend_orch()
    alerts = [
        {"type": "unrealized_loss_deteriorating", "strategy": "grid"},
        {"type": "unrealized_loss_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._unrealized_loss_trend_actions({}, alerts)) == 1


def test_unrealized_loss_trend_actions_disabled():
    orch = _unrealized_loss_trend_orch(enabled=False)
    alerts = [{"type": "unrealized_loss_deteriorating", "strategy": "grid"}]
    assert orch._unrealized_loss_trend_actions({}, alerts) == []


def test_offensive_suppressed_by_unrealized_loss_trend():
    # 浮亏加深的策略不进攻加仓（per-strategy gate，亏损侧趋势维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "unrealized_loss_trend_guard": {"enabled": True, "window": 3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "unrealized_loss_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_unrealized_loss_trend():
    # 浮亏加深的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "unrealized_loss_trend_guard": {"enabled": True, "window": 3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "unrealized_loss_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


def test_offensive_not_suppressed_for_other_strategy_by_unrealized_profit_ratio():
    # 浮盈占比过高的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "unrealized_profit_ratio_guard": {
            "enabled": True, "ratio_threshold": 0.8, "reduce_target": 0.1,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync", "grid"], "boost_step": 0.05},
        {"type": "unrealized_profit_concentration", "strategy": "sync"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.5},
        "grid": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # sync 被抑制，grid 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "grid"


# ── 回撤加速预警（drawdown_acceleration_guard）────────────────

def _drawdown_accel_orch(**overrides):
    cfg = {"agi_orchestrator": {"drawdown_acceleration_guard": {
        "enabled": True, "window": 3,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["drawdown_acceleration_guard"][k] = v
    orch = QuantAGIOrchestrator(config=cfg)
    # 隔离真实 state 文件（data/agi_orchestrator_state.json）残留的运行时数据，
    # 保证测试从干净的 drawdown_history 开始，不依赖磁盘状态。
    orch._drawdown_history.clear()
    return orch


def _drawdown_accel_perception(dd):
    return {
        "equity": 1000.0,
        "used_margin": 500.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "equity_status": {"mode": "normal", "max_drawdown_pct": dd},
    }


def test_drawdown_accel_detects_consecutive_increase():
    orch = _drawdown_accel_orch()
    orch._diagnose(_drawdown_accel_perception(0.01))
    orch._diagnose(_drawdown_accel_perception(0.02))
    alerts = orch._diagnose(_drawdown_accel_perception(0.03))
    accel = [a for a in alerts if a["type"] == "drawdown_accelerating"]
    assert len(accel) == 1
    assert accel[0]["current_drawdown"] == pytest.approx(0.03)


def test_drawdown_accel_no_alert_when_not_increasing():
    orch = _drawdown_accel_orch()
    orch._diagnose(_drawdown_accel_perception(0.03))
    orch._diagnose(_drawdown_accel_perception(0.02))
    alerts = orch._diagnose(_drawdown_accel_perception(0.01))
    assert not [a for a in alerts if a["type"] == "drawdown_accelerating"]


def test_drawdown_accel_no_alert_when_flat():
    orch = _drawdown_accel_orch()
    orch._diagnose(_drawdown_accel_perception(0.02))
    orch._diagnose(_drawdown_accel_perception(0.02))
    alerts = orch._diagnose(_drawdown_accel_perception(0.02))
    assert not [a for a in alerts if a["type"] == "drawdown_accelerating"]


def test_drawdown_accel_no_alert_when_all_zero():
    orch = _drawdown_accel_orch()
    orch._diagnose(_drawdown_accel_perception(0.0))
    orch._diagnose(_drawdown_accel_perception(0.0))
    alerts = orch._diagnose(_drawdown_accel_perception(0.0))
    assert not [a for a in alerts if a["type"] == "drawdown_accelerating"]


def test_drawdown_accel_disabled():
    orch = _drawdown_accel_orch(enabled=False)
    orch._diagnose(_drawdown_accel_perception(0.01))
    orch._diagnose(_drawdown_accel_perception(0.02))
    alerts = orch._diagnose(_drawdown_accel_perception(0.03))
    assert not [a for a in alerts if a["type"] == "drawdown_accelerating"]


def test_offensive_suppressed_by_drawdown_accel():
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "drawdown_acceleration_guard": {"enabled": True, "window": 3},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "drawdown_accelerating", "current_drawdown": 0.03},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_without_drawdown_accel():
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "drawdown_acceleration_guard": {"enabled": True, "window": 3},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


def test_drawdown_accel_persist_restore(tmp_path):
    orch = _drawdown_accel_orch()
    orch._diagnose(_drawdown_accel_perception(0.01))
    orch._diagnose(_drawdown_accel_perception(0.02))
    state_file = tmp_path / "test_state.json"
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _drawdown_accel_orch()
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._drawdown_history) == [pytest.approx(0.01), pytest.approx(0.02)]


# ── 回撤加速收敛动作（drawdown_acceleration_guard 主动侧）────────────────

def test_drawdown_accel_actions_generate_decrease():
    orch = _drawdown_accel_orch()
    alerts = [{"type": "drawdown_accelerating", "current_drawdown": 0.03}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._drawdown_acceleration_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_drawdown_accel_actions_no_alert():
    orch = _drawdown_accel_orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._drawdown_acceleration_actions(decision, []) == []


def test_drawdown_accel_actions_disabled():
    orch = _drawdown_accel_orch(enabled=False)
    alerts = [{"type": "drawdown_accelerating", "current_drawdown": 0.03}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._drawdown_acceleration_actions(decision, alerts) == []


def test_drawdown_accel_actions_below_reduce_target():
    orch = _drawdown_accel_orch(reduce_target=0.3)
    alerts = [{"type": "drawdown_accelerating", "current_drawdown": 0.03}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._drawdown_acceleration_actions(decision, alerts) == []


# ── 交易成本侵蚀主动降仓（cost_awareness 的主动收敛侧）────────────────

def test_high_trading_cost_actions_generate_decrease():
    # 手续费侵蚀过高 → 主动降低 target_weight 最高（保证金占用代理）策略的权重
    orch = _cost_awareness_orch()
    alerts = [{"type": "high_trading_cost", "fee_ratio": 0.3}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._high_trading_cost_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_high_trading_cost_actions_no_alert():
    orch = _cost_awareness_orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._high_trading_cost_actions(decision, []) == []


def test_high_trading_cost_actions_disabled():
    orch = _cost_awareness_orch(enabled=False)
    alerts = [{"type": "high_trading_cost", "fee_ratio": 0.3}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._high_trading_cost_actions(decision, alerts) == []


def test_high_trading_cost_actions_below_reduce_target():
    orch = _cost_awareness_orch(reduce_target=0.3)
    alerts = [{"type": "high_trading_cost", "fee_ratio": 0.3}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._high_trading_cost_actions(decision, alerts) == []


# ── 权益状态机主动收敛（equity_mode_guard）────────────────

def _equity_mode_orch(**overrides):
    cfg = {"agi_orchestrator": {"equity_mode_guard": {
        "enabled": True, "emergency_reduce_target": 0.0, "decline_reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["equity_mode_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_equity_mode_emergency_generate_decrease():
    # EMERGENCY 紧急状态 → 全面降仓到 emergency_reduce_target（默认0）
    orch = _equity_mode_orch()
    alerts = [{"type": "equity_emergency", "mode": "emergency"}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._equity_mode_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.0)


def test_equity_mode_decline_generate_decrease():
    # DECLINE 衰退状态 → 谨慎降仓到 decline_reduce_target（默认0.1）
    orch = _equity_mode_orch()
    alerts = [{"type": "equity_decline", "mode": "decline"}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._equity_mode_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_equity_mode_no_alert():
    orch = _equity_mode_orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._equity_mode_actions(decision, []) == []


def test_equity_mode_disabled():
    orch = _equity_mode_orch(enabled=False)
    alerts = [{"type": "equity_decline", "mode": "decline"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._equity_mode_actions(decision, alerts) == []


def test_equity_mode_below_reduce_target():
    # 当前权重已 ≤ decline_reduce_target → 不再降仓
    orch = _equity_mode_orch(decline_reduce_target=0.3)
    alerts = [{"type": "equity_decline", "mode": "decline"}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._equity_mode_actions(decision, alerts) == []


# ── 连续下跌收敛（downside_momentum_guard）────────────────

def _downside_momentum_orch(**overrides):
    cfg = {"agi_orchestrator": {"downside_momentum_guard": {
        "enabled": True, "max_consecutive_down": 3, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["downside_momentum_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _downside_momentum_perception(consecutive_down):
    return {
        "equity": 1000.0,
        "used_margin": 500.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "equity_status": {"mode": "normal", "consecutive_down": consecutive_down},
    }


def test_downside_momentum_detects_panic():
    orch = _downside_momentum_orch()
    alerts = orch._diagnose(_downside_momentum_perception(consecutive_down=4))
    panic = [a for a in alerts if a["type"] == "market_panicking"]
    assert len(panic) == 1
    assert panic[0]["consecutive_down"] == 4


def test_downside_momentum_below_threshold():
    orch = _downside_momentum_orch()
    alerts = orch._diagnose(_downside_momentum_perception(consecutive_down=2))
    assert not [a for a in alerts if a["type"] == "market_panicking"]


def test_downside_momentum_disabled():
    orch = _downside_momentum_orch(enabled=False)
    alerts = orch._diagnose(_downside_momentum_perception(consecutive_down=4))
    assert not [a for a in alerts if a["type"] == "market_panicking"]


def test_downside_momentum_actions_generate_decrease():
    orch = _downside_momentum_orch()
    alerts = [{"type": "market_panicking", "consecutive_down": 4}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.4},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._downside_momentum_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_downside_momentum_actions_no_alert():
    orch = _downside_momentum_orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._downside_momentum_actions(decision, []) == []


def test_downside_momentum_actions_disabled():
    orch = _downside_momentum_orch(enabled=False)
    alerts = [{"type": "market_panicking", "consecutive_down": 4}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.4}}}}
    assert orch._downside_momentum_actions(decision, alerts) == []


def test_offensive_suppressed_by_downside_momentum():
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "downside_momentum_guard": {"enabled": True, "max_consecutive_down": 3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "market_panicking", "consecutive_down": 4},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_without_downside_momentum():
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "downside_momentum_guard": {"enabled": True, "max_consecutive_down": 3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [{"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert len(orch._offensive_allocation_actions(decision, alerts)) == 1


# ── 策略胜率趋势外推（win_rate_trend_guard）──────────────────

def _win_rate_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"win_rate_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    state_path = overrides.pop("state_path", None)
    if state_path is not None:
        cfg["agi_orchestrator"]["state_path"] = str(state_path)
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["win_rate_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _win_rate_trend_perception(win_rate, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": win_rate,
        }}},
        "freeze_state": {},
    }


def test_win_rate_trend_detects_decline():
    orch = _win_rate_trend_orch()
    orch._strategy_win_rate_history["grid"] = deque([0.6, 0.5], maxlen=3)
    alerts = orch._diagnose(_win_rate_trend_perception(win_rate=0.4))
    deteriorating = [a for a in alerts if a["type"] == "win_rate_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["win_rate"] == pytest.approx(0.4)


def test_win_rate_trend_no_alert_when_rising():
    orch = _win_rate_trend_orch()
    orch._strategy_win_rate_history["grid"] = deque([0.4, 0.5], maxlen=3)
    alerts = orch._diagnose(_win_rate_trend_perception(win_rate=0.6))
    assert not [a for a in alerts if a["type"] == "win_rate_deteriorating"]


def test_win_rate_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免噪声误触发
    orch = _win_rate_trend_orch(min_trades=10)
    orch._strategy_win_rate_history["grid"] = deque([0.6, 0.5], maxlen=3)
    alerts = orch._diagnose(_win_rate_trend_perception(win_rate=0.4, total_trades=5))
    assert not [a for a in alerts if a["type"] == "win_rate_deteriorating"]


def test_win_rate_trend_insufficient_samples():
    orch = _win_rate_trend_orch()
    orch._strategy_win_rate_history["grid"] = deque([0.5], maxlen=3)
    alerts = orch._diagnose(_win_rate_trend_perception(win_rate=0.4))
    assert not [a for a in alerts if a["type"] == "win_rate_deteriorating"]


def test_win_rate_trend_disabled():
    orch = _win_rate_trend_orch(enabled=False)
    orch._strategy_win_rate_history["grid"] = deque([0.6, 0.5], maxlen=3)
    alerts = orch._diagnose(_win_rate_trend_perception(win_rate=0.4))
    assert not [a for a in alerts if a["type"] == "win_rate_deteriorating"]


def test_win_rate_trend_actions_generate():
    orch = _win_rate_trend_orch()
    alerts = [{"type": "win_rate_deteriorating", "strategy": "grid", "message": "连续下降"}]
    actions = orch._win_rate_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_win_rate_trend_actions_ignore_other_alerts():
    orch = _win_rate_trend_orch()
    alerts = [
        {"type": "health_deteriorating", "strategy": "grid"},
        {"type": "strategy_declining", "strategy": "grid"},
    ]
    assert orch._win_rate_trend_actions(alerts) == []


def test_win_rate_trend_actions_dedup():
    orch = _win_rate_trend_orch()
    alerts = [
        {"type": "win_rate_deteriorating", "strategy": "grid"},
        {"type": "win_rate_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._win_rate_trend_actions(alerts)) == 1


def test_win_rate_trend_actions_disabled():
    orch = _win_rate_trend_orch(enabled=False)
    alerts = [{"type": "win_rate_deteriorating", "strategy": "grid"}]
    assert orch._win_rate_trend_actions(alerts) == []


def test_offensive_suppressed_by_win_rate_trend():
    # 胜率恶化的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "win_rate_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "win_rate_deteriorating", "strategy": "grid", "win_rate": 0.3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_healthy_strategy():
    # 胜率恶化的策略不进攻，但同周期其他健康策略仍可进攻加仓（per-strategy 而非全局）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "win_rate_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "trend"], "boost_step": 0.05},
        {"type": "win_rate_deteriorating", "strategy": "grid", "win_rate": 0.3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，trend 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "trend"


def test_win_rate_trend_persist_restore(tmp_path):
    state_file = tmp_path / "test_state.json"
    orch = _win_rate_trend_orch(state_path=state_file)
    orch._diagnose(_win_rate_trend_perception(win_rate=0.6))
    orch._diagnose(_win_rate_trend_perception(win_rate=0.5))
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _win_rate_trend_orch(state_path=state_file)
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._strategy_win_rate_history["grid"]) == [pytest.approx(0.6), pytest.approx(0.5)]


# ── 策略夏普比率趋势外推（sharpe_trend_guard）──────────────────

def _sharpe_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"sharpe_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    state_path = overrides.pop("state_path", None)
    if state_path is not None:
        cfg["agi_orchestrator"]["state_path"] = str(state_path)
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["sharpe_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _sharpe_trend_perception(sharpe_ratio, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "sharpe_ratio": sharpe_ratio,
        }}},
        "freeze_state": {},
    }


def test_sharpe_trend_detects_decline():
    orch = _sharpe_trend_orch()
    orch._strategy_sharpe_history["grid"] = deque([2.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_sharpe_trend_perception(sharpe_ratio=1.0))
    deteriorating = [a for a in alerts if a["type"] == "sharpe_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["sharpe_ratio"] == pytest.approx(1.0)


def test_sharpe_trend_no_alert_when_rising():
    orch = _sharpe_trend_orch()
    orch._strategy_sharpe_history["grid"] = deque([1.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_sharpe_trend_perception(sharpe_ratio=2.0))
    assert not [a for a in alerts if a["type"] == "sharpe_deteriorating"]


def test_sharpe_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免噪声误触发
    orch = _sharpe_trend_orch(min_trades=10)
    orch._strategy_sharpe_history["grid"] = deque([2.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_sharpe_trend_perception(sharpe_ratio=1.0, total_trades=5))
    assert not [a for a in alerts if a["type"] == "sharpe_deteriorating"]


def test_sharpe_trend_insufficient_samples():
    orch = _sharpe_trend_orch()
    orch._strategy_sharpe_history["grid"] = deque([1.5], maxlen=3)
    alerts = orch._diagnose(_sharpe_trend_perception(sharpe_ratio=1.0))
    assert not [a for a in alerts if a["type"] == "sharpe_deteriorating"]


def test_sharpe_trend_disabled():
    orch = _sharpe_trend_orch(enabled=False)
    orch._strategy_sharpe_history["grid"] = deque([2.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_sharpe_trend_perception(sharpe_ratio=1.0))
    assert not [a for a in alerts if a["type"] == "sharpe_deteriorating"]


def test_sharpe_trend_actions_generate():
    orch = _sharpe_trend_orch()
    alerts = [{"type": "sharpe_deteriorating", "strategy": "grid", "message": "连续下降"}]
    actions = orch._sharpe_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_sharpe_trend_actions_ignore_other_alerts():
    orch = _sharpe_trend_orch()
    alerts = [
        {"type": "health_deteriorating", "strategy": "grid"},
        {"type": "win_rate_deteriorating", "strategy": "grid"},
    ]
    assert orch._sharpe_trend_actions(alerts) == []


def test_sharpe_trend_actions_dedup():
    orch = _sharpe_trend_orch()
    alerts = [
        {"type": "sharpe_deteriorating", "strategy": "grid"},
        {"type": "sharpe_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._sharpe_trend_actions(alerts)) == 1


def test_sharpe_trend_actions_disabled():
    orch = _sharpe_trend_orch(enabled=False)
    alerts = [{"type": "sharpe_deteriorating", "strategy": "grid"}]
    assert orch._sharpe_trend_actions(alerts) == []


def test_offensive_suppressed_by_sharpe_trend():
    # 夏普恶化的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "sharpe_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "sharpe_deteriorating", "strategy": "grid", "sharpe_ratio": 0.5},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_healthy_strategy_sharpe():
    # 夏普恶化的策略不进攻，但同周期其他健康策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "sharpe_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "trend"], "boost_step": 0.05},
        {"type": "sharpe_deteriorating", "strategy": "grid", "sharpe_ratio": 0.5},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，trend 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "trend"


def test_sharpe_trend_persist_restore(tmp_path):
    state_file = tmp_path / "test_state.json"
    orch = _sharpe_trend_orch(state_path=state_file)
    orch._diagnose(_sharpe_trend_perception(sharpe_ratio=2.0))
    orch._diagnose(_sharpe_trend_perception(sharpe_ratio=1.5))
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _sharpe_trend_orch(state_path=state_file)
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._strategy_sharpe_history["grid"]) == [pytest.approx(2.0), pytest.approx(1.5)]


# ── 策略级连续亏损收敛（consecutive_losses_guard）──────────────────

def _consecutive_losses_orch(**overrides):
    cfg = {"agi_orchestrator": {"consecutive_losses_guard": {
        "enabled": True, "loss_threshold": 3, "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["consecutive_losses_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _consecutive_losses_perception(consecutive_losses, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "consecutive_losses": consecutive_losses,
        }}},
        "freeze_state": {},
    }


def test_consecutive_losses_detects_threshold():
    orch = _consecutive_losses_orch()
    alerts = orch._diagnose(_consecutive_losses_perception(consecutive_losses=3))
    losing = [a for a in alerts if a["type"] == "strategy_losing_streak"]
    assert len(losing) == 1
    assert losing[0]["strategy"] == "grid"
    assert losing[0]["consecutive_losses"] == 3


def test_consecutive_losses_above_threshold():
    # 超过阈值同样触发（≥ loss_threshold）
    orch = _consecutive_losses_orch()
    alerts = orch._diagnose(_consecutive_losses_perception(consecutive_losses=4))
    assert [a for a in alerts if a["type"] == "strategy_losing_streak"]


def test_consecutive_losses_below_threshold():
    orch = _consecutive_losses_orch()
    alerts = orch._diagnose(_consecutive_losses_perception(consecutive_losses=2))
    assert not [a for a in alerts if a["type"] == "strategy_losing_streak"]


def test_consecutive_losses_disabled():
    orch = _consecutive_losses_orch(enabled=False)
    alerts = orch._diagnose(_consecutive_losses_perception(consecutive_losses=5))
    assert not [a for a in alerts if a["type"] == "strategy_losing_streak"]


def test_consecutive_losses_actions_generate():
    orch = _consecutive_losses_orch()
    alerts = [{"type": "strategy_losing_streak", "strategy": "grid", "message": "连续亏损"}]
    actions = orch._consecutive_losses_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_consecutive_losses_actions_ignore_other_alerts():
    orch = _consecutive_losses_orch()
    alerts = [
        {"type": "health_deteriorating", "strategy": "grid"},
        {"type": "sharpe_deteriorating", "strategy": "grid"},
    ]
    assert orch._consecutive_losses_actions(alerts) == []


def test_consecutive_losses_actions_dedup():
    orch = _consecutive_losses_orch()
    alerts = [
        {"type": "strategy_losing_streak", "strategy": "grid"},
        {"type": "strategy_losing_streak", "strategy": "grid"},
    ]
    assert len(orch._consecutive_losses_actions(alerts)) == 1


def test_consecutive_losses_actions_disabled():
    orch = _consecutive_losses_orch(enabled=False)
    alerts = [{"type": "strategy_losing_streak", "strategy": "grid"}]
    assert orch._consecutive_losses_actions(alerts) == []


def test_offensive_suppressed_by_consecutive_losses():
    # 连续亏损达阈值的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "consecutive_losses_guard": {"enabled": True, "loss_threshold": 3},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "strategy_losing_streak", "strategy": "grid", "consecutive_losses": 3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_healthy_strategy_consecutive_losses():
    # 连续亏损策略不进攻，但同周期其他健康策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "consecutive_losses_guard": {"enabled": True, "loss_threshold": 3},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "trend"], "boost_step": 0.05},
        {"type": "strategy_losing_streak", "strategy": "grid", "consecutive_losses": 3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，trend 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "trend"


# ── 策略盈亏比趋势外推（profit_factor_trend_guard）──────────────────

def _profit_factor_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"profit_factor_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    state_path = overrides.pop("state_path", None)
    if state_path is not None:
        cfg["agi_orchestrator"]["state_path"] = str(state_path)
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["profit_factor_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _profit_factor_trend_perception(profit_factor, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "profit_factor": profit_factor,
        }}},
        "freeze_state": {},
    }


def test_profit_factor_trend_detects_decline():
    orch = _profit_factor_trend_orch()
    orch._strategy_profit_factor_history["grid"] = deque([2.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_profit_factor_trend_perception(profit_factor=1.0))
    deteriorating = [a for a in alerts if a["type"] == "profit_factor_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["profit_factor"] == pytest.approx(1.0)


def test_profit_factor_trend_no_alert_when_rising():
    orch = _profit_factor_trend_orch()
    orch._strategy_profit_factor_history["grid"] = deque([1.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_profit_factor_trend_perception(profit_factor=2.0))
    assert not [a for a in alerts if a["type"] == "profit_factor_deteriorating"]


def test_profit_factor_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免噪声误触发
    orch = _profit_factor_trend_orch(min_trades=10)
    orch._strategy_profit_factor_history["grid"] = deque([2.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_profit_factor_trend_perception(profit_factor=1.0, total_trades=5))
    assert not [a for a in alerts if a["type"] == "profit_factor_deteriorating"]


def test_profit_factor_trend_insufficient_samples():
    orch = _profit_factor_trend_orch()
    orch._strategy_profit_factor_history["grid"] = deque([1.5], maxlen=3)
    alerts = orch._diagnose(_profit_factor_trend_perception(profit_factor=1.0))
    assert not [a for a in alerts if a["type"] == "profit_factor_deteriorating"]


def test_profit_factor_trend_disabled():
    orch = _profit_factor_trend_orch(enabled=False)
    orch._strategy_profit_factor_history["grid"] = deque([2.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_profit_factor_trend_perception(profit_factor=1.0))
    assert not [a for a in alerts if a["type"] == "profit_factor_deteriorating"]


def test_profit_factor_trend_actions_generate():
    orch = _profit_factor_trend_orch()
    alerts = [{"type": "profit_factor_deteriorating", "strategy": "grid", "message": "连续下降"}]
    actions = orch._profit_factor_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_profit_factor_trend_actions_ignore_other_alerts():
    orch = _profit_factor_trend_orch()
    alerts = [
        {"type": "health_deteriorating", "strategy": "grid"},
        {"type": "sharpe_deteriorating", "strategy": "grid"},
    ]
    assert orch._profit_factor_trend_actions(alerts) == []


def test_profit_factor_trend_actions_dedup():
    orch = _profit_factor_trend_orch()
    alerts = [
        {"type": "profit_factor_deteriorating", "strategy": "grid"},
        {"type": "profit_factor_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._profit_factor_trend_actions(alerts)) == 1


def test_profit_factor_trend_actions_disabled():
    orch = _profit_factor_trend_orch(enabled=False)
    alerts = [{"type": "profit_factor_deteriorating", "strategy": "grid"}]
    assert orch._profit_factor_trend_actions(alerts) == []


def test_offensive_suppressed_by_profit_factor_trend():
    # 盈亏比恶化的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "profit_factor_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "profit_factor_deteriorating", "strategy": "grid", "profit_factor": 0.8},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_healthy_strategy_profit_factor():
    # 盈亏比恶化的策略不进攻，但同周期其他健康策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "profit_factor_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "trend"], "boost_step": 0.05},
        {"type": "profit_factor_deteriorating", "strategy": "grid", "profit_factor": 0.8},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，trend 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "trend"


def test_profit_factor_trend_persist_restore(tmp_path):
    state_file = tmp_path / "test_state.json"
    orch = _profit_factor_trend_orch(state_path=state_file)
    orch._diagnose(_profit_factor_trend_perception(profit_factor=2.0))
    orch._diagnose(_profit_factor_trend_perception(profit_factor=1.5))
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _profit_factor_trend_orch(state_path=state_file)
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._strategy_profit_factor_history["grid"]) == [
        pytest.approx(2.0), pytest.approx(1.5),
    ]


# ── 策略最大回撤趋势外推（max_drawdown_trend_guard）──────────────────

def _max_drawdown_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"max_drawdown_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    state_path = overrides.pop("state_path", None)
    if state_path is not None:
        cfg["agi_orchestrator"]["state_path"] = str(state_path)
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["max_drawdown_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _max_drawdown_trend_perception(max_drawdown, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "max_drawdown": max_drawdown,
        }}},
        "freeze_state": {},
    }


def test_max_drawdown_trend_detects_decline():
    # 回撤连续加深（上升）→ 触发告警
    orch = _max_drawdown_trend_orch()
    orch._strategy_max_drawdown_history["grid"] = deque([0.05, 0.08], maxlen=3)
    alerts = orch._diagnose(_max_drawdown_trend_perception(max_drawdown=0.12))
    deteriorating = [a for a in alerts if a["type"] == "max_drawdown_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["max_drawdown"] == pytest.approx(0.12)


def test_max_drawdown_trend_no_alert_when_falling():
    orch = _max_drawdown_trend_orch()
    orch._strategy_max_drawdown_history["grid"] = deque([0.12, 0.08], maxlen=3)
    alerts = orch._diagnose(_max_drawdown_trend_perception(max_drawdown=0.05))
    assert not [a for a in alerts if a["type"] == "max_drawdown_deteriorating"]


def test_max_drawdown_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免噪声误触发
    orch = _max_drawdown_trend_orch(min_trades=10)
    orch._strategy_max_drawdown_history["grid"] = deque([0.05, 0.08], maxlen=3)
    alerts = orch._diagnose(_max_drawdown_trend_perception(max_drawdown=0.12, total_trades=5))
    assert not [a for a in alerts if a["type"] == "max_drawdown_deteriorating"]


def test_max_drawdown_trend_insufficient_samples():
    orch = _max_drawdown_trend_orch()
    orch._strategy_max_drawdown_history["grid"] = deque([0.08], maxlen=3)
    alerts = orch._diagnose(_max_drawdown_trend_perception(max_drawdown=0.12))
    assert not [a for a in alerts if a["type"] == "max_drawdown_deteriorating"]


def test_max_drawdown_trend_disabled():
    orch = _max_drawdown_trend_orch(enabled=False)
    orch._strategy_max_drawdown_history["grid"] = deque([0.05, 0.08], maxlen=3)
    alerts = orch._diagnose(_max_drawdown_trend_perception(max_drawdown=0.12))
    assert not [a for a in alerts if a["type"] == "max_drawdown_deteriorating"]


def test_max_drawdown_trend_actions_generate():
    orch = _max_drawdown_trend_orch()
    alerts = [{"type": "max_drawdown_deteriorating", "strategy": "grid", "message": "连续加深"}]
    actions = orch._max_drawdown_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_max_drawdown_trend_actions_ignore_other_alerts():
    orch = _max_drawdown_trend_orch()
    alerts = [
        {"type": "health_deteriorating", "strategy": "grid"},
        {"type": "profit_factor_deteriorating", "strategy": "grid"},
    ]
    assert orch._max_drawdown_trend_actions(alerts) == []


def test_max_drawdown_trend_actions_dedup():
    orch = _max_drawdown_trend_orch()
    alerts = [
        {"type": "max_drawdown_deteriorating", "strategy": "grid"},
        {"type": "max_drawdown_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._max_drawdown_trend_actions(alerts)) == 1


def test_max_drawdown_trend_actions_disabled():
    orch = _max_drawdown_trend_orch(enabled=False)
    alerts = [{"type": "max_drawdown_deteriorating", "strategy": "grid"}]
    assert orch._max_drawdown_trend_actions(alerts) == []


def test_offensive_suppressed_by_max_drawdown_trend():
    # 最大回撤加深的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "max_drawdown_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "max_drawdown_deteriorating", "strategy": "grid", "max_drawdown": 0.2},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_healthy_strategy_max_drawdown():
    # 最大回撤加深的策略不进攻，但同周期其他健康策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "max_drawdown_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "trend"], "boost_step": 0.05},
        {"type": "max_drawdown_deteriorating", "strategy": "grid", "max_drawdown": 0.2},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，trend 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "trend"


def test_max_drawdown_trend_persist_restore(tmp_path):
    state_file = tmp_path / "test_state.json"
    orch = _max_drawdown_trend_orch(state_path=state_file)
    orch._diagnose(_max_drawdown_trend_perception(max_drawdown=0.05))
    orch._diagnose(_max_drawdown_trend_perception(max_drawdown=0.08))
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _max_drawdown_trend_orch(state_path=state_file)
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._strategy_max_drawdown_history["grid"]) == [
        pytest.approx(0.05), pytest.approx(0.08),
    ]


# ── 多守卫共振收敛（resonance_guard）──────────────────

def _resonance_orch(**overrides):
    cfg = {"agi_orchestrator": {
        "health_trend_guard": {"enabled": True, "window": 3, "deterioration_leverage": 1.0},
        "win_rate_trend_guard": {"enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0},
        "sharpe_trend_guard": {"enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0},
        "resonance_guard": {"enabled": True, "resonance_threshold": 2, "resonance_leverage": 0.5},
    }}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["resonance_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _resonance_perception(health_score=70.0, win_rate=0.4, sharpe_ratio=1.0, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": health_score,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe_ratio,
        }}},
        "freeze_state": {},
    }


def test_resonance_detects_multi_decay():
    # 三个趋势守卫（health/win_rate/sharpe）同时衰退 → 多守卫共振
    orch = _resonance_orch()
    orch._strategy_health_history["grid"] = deque([80.0, 75.0], maxlen=3)
    orch._strategy_win_rate_history["grid"] = deque([0.6, 0.5], maxlen=3)
    orch._strategy_sharpe_history["grid"] = deque([2.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_resonance_perception(health_score=70.0, win_rate=0.4, sharpe_ratio=1.0))
    res = [a for a in alerts if a["type"] == "multi_guard_resonance"]
    assert len(res) == 1
    assert res[0]["strategy"] == "grid"
    assert res[0]["level"] == "critical"
    assert len(res[0]["decay_types"]) >= 2


def test_resonance_no_alert_single_decay():
    # 仅单维度（health）衰退，不足 threshold=2 → 不共振
    orch = _resonance_orch()
    orch._strategy_health_history["grid"] = deque([80.0, 75.0], maxlen=3)
    alerts = orch._diagnose(_resonance_perception(health_score=70.0, win_rate=0.4, sharpe_ratio=1.0))
    assert not [a for a in alerts if a["type"] == "multi_guard_resonance"]


def test_resonance_respects_threshold():
    # threshold=3，仅 2 维衰退 → 不共振
    orch = _resonance_orch(resonance_threshold=3)
    orch._strategy_health_history["grid"] = deque([80.0, 75.0], maxlen=3)
    orch._strategy_win_rate_history["grid"] = deque([0.6, 0.5], maxlen=3)
    alerts = orch._diagnose(_resonance_perception(health_score=70.0, win_rate=0.4, sharpe_ratio=1.0))
    assert not [a for a in alerts if a["type"] == "multi_guard_resonance"]


def test_resonance_disabled():
    orch = _resonance_orch(enabled=False)
    orch._strategy_health_history["grid"] = deque([80.0, 75.0], maxlen=3)
    orch._strategy_win_rate_history["grid"] = deque([0.6, 0.5], maxlen=3)
    orch._strategy_sharpe_history["grid"] = deque([2.0, 1.5], maxlen=3)
    alerts = orch._diagnose(_resonance_perception(health_score=70.0, win_rate=0.4, sharpe_ratio=1.0))
    assert not [a for a in alerts if a["type"] == "multi_guard_resonance"]


def test_resonance_actions_generate():
    orch = _resonance_orch()
    alerts = [{"type": "multi_guard_resonance", "strategy": "grid", "message": "多维度共振衰退"}]
    actions = orch._resonance_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 0.5


def test_resonance_actions_ignore_other_alerts():
    orch = _resonance_orch()
    alerts = [
        {"type": "health_deteriorating", "strategy": "grid"},
        {"type": "sharpe_deteriorating", "strategy": "grid"},
    ]
    assert orch._resonance_actions(alerts) == []


def test_resonance_actions_dedup():
    orch = _resonance_orch()
    alerts = [
        {"type": "multi_guard_resonance", "strategy": "grid"},
        {"type": "multi_guard_resonance", "strategy": "grid"},
    ]
    assert len(orch._resonance_actions(alerts)) == 1


def test_resonance_actions_disabled():
    orch = _resonance_orch(enabled=False)
    alerts = [{"type": "multi_guard_resonance", "strategy": "grid"}]
    assert orch._resonance_actions(alerts) == []


def test_offensive_suppressed_by_resonance():
    # 共振衰退的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "resonance_guard": {"enabled": True, "resonance_threshold": 2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "multi_guard_resonance", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_healthy_strategy_resonance():
    # 共振衰退的策略不进攻，但同周期其他健康策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "resonance_guard": {"enabled": True, "resonance_threshold": 2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "trend"], "boost_step": 0.05},
        {"type": "multi_guard_resonance", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.2},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，trend 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "trend"


# ── 恢复纯度门控（recovery_purity_guard）──────────────────

def _recovery_purity_orch(**overrides):
    cfg = {"agi_orchestrator": {
        "health_trend_guard": {"enabled": True, "window": 3, "deterioration_leverage": 1.0},
        "recovery_purity_guard": {"enabled": True},
    }}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["recovery_purity_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _recovery_purity_perception(health_score=70.0):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": health_score,
            "health_grade": "A",
            "lifecycle": "mature",
            "trend": "improving",
            "total_trades": 20,
            "total_pnl": 50.0,
            "win_rate": 0.6,
        }}},
        "freeze_state": {},
    }


def test_recovery_purity_blocks_when_decaying():
    # 策略 trend=improving+grade=A 但 health_score 连续下降 → 抑制 strategy_recovered
    orch = _recovery_purity_orch()
    orch._strategy_health_history["grid"] = deque([80.0, 75.0], maxlen=3)
    alerts = orch._diagnose(_recovery_purity_perception(health_score=70.0))
    assert [a for a in alerts if a["type"] == "health_deteriorating"]
    assert not [a for a in alerts if a["type"] == "strategy_recovered"]


def test_recovery_purity_allows_when_clean():
    # 无趋势衰退告警时，strategy_recovered 正常生成
    orch = _recovery_purity_orch()
    alerts = orch._diagnose(_recovery_purity_perception(health_score=70.0))
    assert not [a for a in alerts if a["type"] == "health_deteriorating"]
    assert [a for a in alerts if a["type"] == "strategy_recovered"]


def test_recovery_purity_disabled():
    # disabled 时不抑制（即使有衰退告警，recovered 仍生成——向后兼容）
    orch = _recovery_purity_orch(enabled=False)
    orch._strategy_health_history["grid"] = deque([80.0, 75.0], maxlen=3)
    alerts = orch._diagnose(_recovery_purity_perception(health_score=70.0))
    assert [a for a in alerts if a["type"] == "health_deteriorating"]
    assert [a for a in alerts if a["type"] == "strategy_recovered"]


def test_recovery_purity_blocks_on_other_decay_type():
    # 任意趋势型衰退告警（如 sharpe_deteriorating）也抑制 recovered
    cfg = {"agi_orchestrator": {
        "sharpe_trend_guard": {"enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0},
        "recovery_purity_guard": {"enabled": True},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    orch._strategy_sharpe_history["grid"] = deque([2.0, 1.5], maxlen=3)
    perception = {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 80.0,
            "health_grade": "A",
            "lifecycle": "mature",
            "trend": "improving",
            "total_trades": 20,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "sharpe_ratio": 1.0,
        }}},
        "freeze_state": {},
    }
    alerts = orch._diagnose(perception)
    assert [a for a in alerts if a["type"] == "sharpe_deteriorating"]
    assert not [a for a in alerts if a["type"] == "strategy_recovered"]


# ── 组合级盈利集中度收敛（profit_concentration_guard）──────────────────

def _profit_concentration_orch(**overrides):
    cfg = {"agi_orchestrator": {"profit_concentration_guard": {
        "enabled": True, "concentration_threshold": 0.8,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["profit_concentration_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _pc_strategy(pnl):
    return {
        "total_pnl": pnl,
        "total_trades": 20,
        "health_grade": "A",
        "lifecycle": "mature",
        "trend": "stable",
        "health_score": 80.0,
        "win_rate": 0.5,
    }


def _profit_concentration_perception(strategies, total_pnl):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"total_pnl": total_pnl, "strategies": strategies},
        "freeze_state": {},
    }


def test_profit_concentration_detects_single_pillar():
    # 单一策略贡献 90% 盈利 → 告警
    orch = _profit_concentration_orch()
    strategies = {"sync": _pc_strategy(90.0), "grid": _pc_strategy(10.0)}
    alerts = orch._diagnose(_profit_concentration_perception(strategies, total_pnl=100.0))
    pc = [a for a in alerts if a["type"] == "profit_concentration"]
    assert len(pc) == 1
    assert pc[0]["strategy"] == "sync"
    assert pc[0]["concentration"] == pytest.approx(0.9)


def test_profit_concentration_no_alert_when_diversified():
    # 多策略均分盈利，无单一支柱 → 不告警
    orch = _profit_concentration_orch()
    strategies = {"sync": _pc_strategy(50.0), "grid": _pc_strategy(50.0)}
    alerts = orch._diagnose(_profit_concentration_perception(strategies, total_pnl=100.0))
    assert not [a for a in alerts if a["type"] == "profit_concentration"]


def test_profit_concentration_no_alert_when_loss():
    # 组合亏损（total_pnl <= 0）→ 不告警
    orch = _profit_concentration_orch()
    strategies = {"sync": _pc_strategy(-10.0)}
    alerts = orch._diagnose(_profit_concentration_perception(strategies, total_pnl=-10.0))
    assert not [a for a in alerts if a["type"] == "profit_concentration"]


def test_profit_concentration_disabled():
    orch = _profit_concentration_orch(enabled=False)
    strategies = {"sync": _pc_strategy(90.0), "grid": _pc_strategy(10.0)}
    alerts = orch._diagnose(_profit_concentration_perception(strategies, total_pnl=100.0))
    assert not [a for a in alerts if a["type"] == "profit_concentration"]


def test_offensive_suppressed_by_profit_concentration():
    # 盈利支柱策略不进攻加仓（组合级 per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "profit_concentration_guard": {"enabled": True, "concentration_threshold": 0.8},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "profit_concentration", "strategy": "sync"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_profit_concentration():
    # 盈利支柱策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "profit_concentration_guard": {"enabled": True, "concentration_threshold": 0.8},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync", "grid"], "boost_step": 0.05},
        {"type": "profit_concentration", "strategy": "sync"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.5},
        "grid": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # sync 被抑制，grid 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "grid"


# ── 组合级尾部风险收敛（tail_risk_guard）──────────────────

def _tail_risk_orch(**overrides):
    cfg = {"agi_orchestrator": {"tail_risk_guard": {
        "enabled": True, "tail_risk_threshold": 0.15, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["tail_risk_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _tr_strategy(max_dd):
    return {
        "total_pnl": 10.0,
        "total_trades": 20,
        "health_grade": "A",
        "lifecycle": "mature",
        "trend": "stable",
        "health_score": 80.0,
        "win_rate": 0.5,
        "max_drawdown": max_dd,
    }


def _tail_risk_perception(strategies):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": strategies},
        "freeze_state": {},
    }


def test_tail_risk_detects_high_drawdown():
    # 组合中最深回撤策略 > 阈值 → high_tail_risk 告警
    orch = _tail_risk_orch()
    strategies = {"sync": _tr_strategy(0.20), "grid": _tr_strategy(0.05)}
    alerts = orch._diagnose(_tail_risk_perception(strategies))
    tr = [a for a in alerts if a["type"] == "high_tail_risk"]
    assert len(tr) == 1
    assert tr[0]["strategy"] == "sync"
    assert tr[0]["tail_risk"] == pytest.approx(0.20)


def test_tail_risk_no_alert_when_low():
    # 所有策略回撤 < 阈值 → 不告警
    orch = _tail_risk_orch()
    strategies = {"sync": _tr_strategy(0.10), "grid": _tr_strategy(0.05)}
    alerts = orch._diagnose(_tail_risk_perception(strategies))
    assert not [a for a in alerts if a["type"] == "high_tail_risk"]


def test_tail_risk_disabled():
    orch = _tail_risk_orch(enabled=False)
    strategies = {"sync": _tr_strategy(0.20)}
    alerts = orch._diagnose(_tail_risk_perception(strategies))
    assert not [a for a in alerts if a["type"] == "high_tail_risk"]


def test_tail_risk_actions_generate():
    # high_tail_risk → reallocate decrease 到 reduce_target
    orch = _tail_risk_orch()
    alerts = [{"type": "high_tail_risk", "strategy": "sync", "message": "尾部风险偏高"}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.3}}}}
    actions = orch._tail_risk_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_tail_risk_actions_skip_low_weight():
    # 策略当前权重已 <= reduce_target → 不再生成降权动作
    orch = _tail_risk_orch()
    alerts = [{"type": "high_tail_risk", "strategy": "sync", "message": "尾部风险偏高"}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.05}}}}
    assert orch._tail_risk_actions(decision, alerts) == []


def test_offensive_suppressed_by_tail_risk():
    # 尾部风险偏高 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "tail_risk_guard": {"enabled": True, "tail_risk_threshold": 0.15},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "high_tail_risk", "strategy": "sync"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 组合级风险共振收敛（portfolio_risk_resonance_guard）──────────────────

def _portfolio_resonance_orch(**overrides):
    cfg = {"agi_orchestrator": {
        "portfolio_risk_resonance_guard": {
            "enabled": True, "resonance_threshold": 2, "reduce_target": 0.1,
        },
        "profit_concentration_guard": {"enabled": True, "concentration_threshold": 0.8},
        "tail_risk_guard": {"enabled": True, "tail_risk_threshold": 0.15, "reduce_target": 0.1},
    }}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["portfolio_risk_resonance_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _pr_strategy(pnl, max_dd):
    return {
        "total_pnl": pnl,
        "total_trades": 20,
        "health_grade": "A",
        "lifecycle": "mature",
        "trend": "stable",
        "health_score": 80.0,
        "win_rate": 0.5,
        "max_drawdown": max_dd,
    }


def _portfolio_resonance_perception(strategies, total_pnl):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        # concentration/diversification_score 给安全值，避免误触发权重集中维度
        "contribution": {
            "total_pnl": total_pnl,
            "strategies": strategies,
            "concentration": 0.0,
            "diversification_score": 1.0,
        },
        "freeze_state": {},
    }


def test_portfolio_resonance_detects_multi_dimension():
    # 盈利集中（profit_concentration）+ 尾部风险（high_tail_risk）两维同时触发 → 组合级共振
    orch = _portfolio_resonance_orch()
    strategies = {"sync": _pr_strategy(90.0, 0.20), "grid": _pr_strategy(10.0, 0.05)}
    alerts = orch._diagnose(_portfolio_resonance_perception(strategies, total_pnl=100.0))
    pr = [a for a in alerts if a["type"] == "portfolio_risk_resonance"]
    assert len(pr) == 1
    assert set(pr[0]["dimensions"]) == {"profit_concentration", "tail_risk"}


def test_portfolio_resonance_no_alert_single_dimension():
    # 仅尾部风险一维触发（盈利均分不集中）→ 不共振
    orch = _portfolio_resonance_orch()
    strategies = {"sync": _pr_strategy(50.0, 0.20), "grid": _pr_strategy(50.0, 0.05)}
    alerts = orch._diagnose(_portfolio_resonance_perception(strategies, total_pnl=100.0))
    assert not [a for a in alerts if a["type"] == "portfolio_risk_resonance"]


def test_portfolio_resonance_disabled():
    orch = _portfolio_resonance_orch(enabled=False)
    strategies = {"sync": _pr_strategy(90.0, 0.20), "grid": _pr_strategy(10.0, 0.05)}
    alerts = orch._diagnose(_portfolio_resonance_perception(strategies, total_pnl=100.0))
    assert not [a for a in alerts if a["type"] == "portfolio_risk_resonance"]


def test_portfolio_resonance_actions_generate():
    # 组合级共振 → reallocate decrease 到 reduce_target（比单维度更保守的全局收敛）
    orch = _portfolio_resonance_orch()
    alerts = [{"type": "portfolio_risk_resonance", "dimensions": ["profit_concentration", "tail_risk"]}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.3},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._portfolio_resonance_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_portfolio_resonance_actions_skip_low_weight():
    # 最高权重已 <= reduce_target → 不再生成降权动作
    orch = _portfolio_resonance_orch()
    alerts = [{"type": "portfolio_risk_resonance", "dimensions": ["profit_concentration", "tail_risk"]}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.05}}}}
    assert orch._portfolio_resonance_actions(decision, alerts) == []


def test_offensive_suppressed_by_portfolio_resonance():
    # 组合级风险共振 → 暂停进攻性加仓（账户级 gate，全局收敛）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "portfolio_risk_resonance_guard": {"enabled": True, "resonance_threshold": 2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "portfolio_risk_resonance", "dimensions": ["profit_concentration", "tail_risk"]},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 组合级健康度趋势外推（portfolio_health_trend_guard）──────────────────

def _pht_orch(**overrides):
    cfg = {"agi_orchestrator": {"portfolio_health_trend_guard": {
        "enabled": True, "window": 3, "min_sample": 10, "reduce_target": 0.15,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["portfolio_health_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _pht_perception(health_score, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {
            "overall_health_score": health_score,
            "total_trades": total_trades,
            "strategies": {},
        },
        "freeze_state": {},
    }


def test_portfolio_health_trend_detects_decline():
    # 组合整体健康度连续 3 周期严格下降 → portfolio_health_deteriorating
    orch = _pht_orch()
    orch._diagnose(_pht_perception(80.0))
    orch._diagnose(_pht_perception(75.0))
    alerts = orch._diagnose(_pht_perception(70.0))
    pht = [a for a in alerts if a["type"] == "portfolio_health_deteriorating"]
    assert len(pht) == 1
    assert pht[0]["health_score"] == pytest.approx(70.0)


def test_portfolio_health_trend_no_alert_when_stable():
    # 健康度平稳（非严格下降）→ 不告警
    orch = _pht_orch()
    orch._diagnose(_pht_perception(80.0))
    orch._diagnose(_pht_perception(80.0))
    alerts = orch._diagnose(_pht_perception(80.0))
    assert not [a for a in alerts if a["type"] == "portfolio_health_deteriorating"]


def test_portfolio_health_trend_disabled():
    orch = _pht_orch(enabled=False)
    orch._diagnose(_pht_perception(80.0))
    orch._diagnose(_pht_perception(75.0))
    alerts = orch._diagnose(_pht_perception(70.0))
    assert not [a for a in alerts if a["type"] == "portfolio_health_deteriorating"]


def test_portfolio_health_trend_actions_generate():
    # portfolio_health_deteriorating → reallocate decrease 到 reduce_target
    orch = _pht_orch()
    alerts = [{"type": "portfolio_health_deteriorating", "health_score": 70.0}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.3},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._portfolio_health_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_portfolio_health_trend_actions_skip_low_weight():
    # 最高权重已 <= reduce_target → 不再生成降权动作
    orch = _pht_orch()
    alerts = [{"type": "portfolio_health_deteriorating", "health_score": 70.0}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.1}}}}
    assert orch._portfolio_health_trend_actions(decision, alerts) == []


def test_offensive_suppressed_by_portfolio_health_trend():
    # 组合整体健康度连续下降 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "portfolio_health_trend_guard": {"enabled": True, "window": 3, "min_sample": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "portfolio_health_deteriorating", "health_score": 70.0},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 资金利用率趋势外推（utilization_trend_guard）──────────────────

def _utg_orch(**overrides):
    cfg = {"agi_orchestrator": {"utilization_trend_guard": {
        "enabled": True, "window": 3, "reduce_target": 0.15,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["utilization_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _utg_perception(used_margin, equity=1000.0):
    return {
        "equity": equity,
        "used_margin": used_margin,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
    }


def test_utilization_trend_detects_rising():
    # 资金利用率连续 3 周期严格上升（0.3→0.4→0.5）→ utilization_rising
    orch = _utg_orch()
    orch._diagnose(_utg_perception(300.0))
    orch._diagnose(_utg_perception(400.0))
    alerts = orch._diagnose(_utg_perception(500.0))
    ut = [a for a in alerts if a["type"] == "utilization_rising"]
    assert len(ut) == 1
    assert ut[0]["utilization"] == pytest.approx(0.5)


def test_utilization_trend_no_alert_when_stable():
    # 利用率平稳（非严格上升）→ 不告警
    orch = _utg_orch()
    orch._diagnose(_utg_perception(300.0))
    orch._diagnose(_utg_perception(300.0))
    alerts = orch._diagnose(_utg_perception(300.0))
    assert not [a for a in alerts if a["type"] == "utilization_rising"]


def test_utilization_trend_disabled():
    orch = _utg_orch(enabled=False)
    orch._diagnose(_utg_perception(300.0))
    orch._diagnose(_utg_perception(400.0))
    alerts = orch._diagnose(_utg_perception(500.0))
    assert not [a for a in alerts if a["type"] == "utilization_rising"]


def test_utilization_trend_actions_generate():
    # utilization_rising → reallocate decrease 降保证金/权重最高策略到 reduce_target
    orch = _utg_orch()
    alerts = [{"type": "utilization_rising", "utilization": 0.5}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.3},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._utilization_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_utilization_trend_actions_skip_low_weight():
    # 最高权重已 <= reduce_target → 不再生成降权动作
    orch = _utg_orch()
    alerts = [{"type": "utilization_rising", "utilization": 0.5}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.1}}}}
    assert orch._utilization_trend_actions(decision, alerts) == []


def test_offensive_suppressed_by_utilization_trend():
    # 资金利用率持续上升 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "utilization_trend_guard": {"enabled": True, "window": 3},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "utilization_rising", "utilization": 0.5},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 策略资本回报率趋势外推（capital_return_trend_guard，第六子）──────────────────

def _capital_return_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"capital_return_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["capital_return_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _capital_return_trend_perception(pnl_per_capital_pct, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "max_drawdown": 0.1,
            "pnl_per_capital_pct": pnl_per_capital_pct,
        }}},
        "freeze_state": {},
    }


def test_capital_return_trend_detects_decline():
    # 资本回报率连续下降 → 触发告警
    orch = _capital_return_trend_orch()
    orch._strategy_capital_return_history["grid"] = deque([10.0, 8.0], maxlen=3)
    alerts = orch._diagnose(_capital_return_trend_perception(pnl_per_capital_pct=6.0))
    deteriorating = [a for a in alerts if a["type"] == "capital_return_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["pnl_per_capital_pct"] == pytest.approx(6.0)


def test_capital_return_trend_no_alert_when_rising():
    orch = _capital_return_trend_orch()
    orch._strategy_capital_return_history["grid"] = deque([6.0, 8.0], maxlen=3)
    alerts = orch._diagnose(_capital_return_trend_perception(pnl_per_capital_pct=10.0))
    assert not [a for a in alerts if a["type"] == "capital_return_deteriorating"]


def test_capital_return_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免噪声误触发
    orch = _capital_return_trend_orch(min_trades=10)
    orch._strategy_capital_return_history["grid"] = deque([10.0, 8.0], maxlen=3)
    alerts = orch._diagnose(_capital_return_trend_perception(pnl_per_capital_pct=6.0, total_trades=5))
    assert not [a for a in alerts if a["type"] == "capital_return_deteriorating"]


def test_capital_return_trend_disabled():
    orch = _capital_return_trend_orch(enabled=False)
    orch._strategy_capital_return_history["grid"] = deque([10.0, 8.0], maxlen=3)
    alerts = orch._diagnose(_capital_return_trend_perception(pnl_per_capital_pct=6.0))
    assert not [a for a in alerts if a["type"] == "capital_return_deteriorating"]


def test_capital_return_trend_actions_generate():
    orch = _capital_return_trend_orch()
    alerts = [{"type": "capital_return_deteriorating", "strategy": "grid", "message": "连续下降"}]
    actions = orch._capital_return_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_offensive_suppressed_by_capital_return_trend():
    # 资本回报率下降的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "capital_return_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "capital_return_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 策略波动率趋势外推（volatility_trend_guard，第七子）──────────────────

def _volatility_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"volatility_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["volatility_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _volatility_trend_perception(volatility, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "max_drawdown": 0.1,
            "volatility": volatility,
        }}},
        "freeze_state": {},
    }


def test_volatility_trend_detects_rising():
    # 波动率连续 3 周期严格上升（0.1→0.2→0.3）→ volatility_deteriorating
    orch = _volatility_trend_orch()
    orch._strategy_volatility_history["grid"] = deque([0.1, 0.2], maxlen=3)
    alerts = orch._diagnose(_volatility_trend_perception(volatility=0.3))
    deteriorating = [a for a in alerts if a["type"] == "volatility_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["volatility"] == pytest.approx(0.3)


def test_volatility_trend_no_alert_when_falling():
    # 波动率下降（非严格上升）→ 不告警
    orch = _volatility_trend_orch()
    orch._strategy_volatility_history["grid"] = deque([0.3, 0.2], maxlen=3)
    alerts = orch._diagnose(_volatility_trend_perception(volatility=0.1))
    assert not [a for a in alerts if a["type"] == "volatility_deteriorating"]


def test_volatility_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免噪声误触发
    orch = _volatility_trend_orch(min_trades=10)
    orch._strategy_volatility_history["grid"] = deque([0.1, 0.2], maxlen=3)
    alerts = orch._diagnose(_volatility_trend_perception(volatility=0.3, total_trades=5))
    assert not [a for a in alerts if a["type"] == "volatility_deteriorating"]


def test_volatility_trend_disabled():
    orch = _volatility_trend_orch(enabled=False)
    orch._strategy_volatility_history["grid"] = deque([0.1, 0.2], maxlen=3)
    alerts = orch._diagnose(_volatility_trend_perception(volatility=0.3))
    assert not [a for a in alerts if a["type"] == "volatility_deteriorating"]


def test_volatility_trend_actions_generate():
    orch = _volatility_trend_orch()
    alerts = [{"type": "volatility_deteriorating", "strategy": "grid", "message": "连续上升"}]
    actions = orch._volatility_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_volatility_trend_actions_dedup():
    orch = _volatility_trend_orch()
    alerts = [
        {"type": "volatility_deteriorating", "strategy": "grid"},
        {"type": "volatility_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._volatility_trend_actions(alerts)) == 1


def test_volatility_trend_actions_disabled():
    orch = _volatility_trend_orch(enabled=False)
    alerts = [{"type": "volatility_deteriorating", "strategy": "grid"}]
    assert orch._volatility_trend_actions(alerts) == []


def test_offensive_suppressed_by_volatility_trend():
    # 波动率上升的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "volatility_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "volatility_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_volatility_trend():
    # 波动率上升的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "volatility_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "volatility_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 组合级集中度趋势外推（portfolio_concentration_trend_guard）──────────────────

def _portfolio_concentration_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"portfolio_concentration_trend_guard": {
        "enabled": True, "window": 3, "reduce_target": 0.2,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["portfolio_concentration_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _pct_perception(concentration):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {
            "concentration": concentration,
            "overall_health_score": 70.0,
            "total_trades": 20,
            "strategies": {},
        },
        "freeze_state": {},
    }


def test_portfolio_concentration_trend_detects_rising():
    # 组合集中度连续 3 周期严格上升（0.2→0.3→0.4）→ portfolio_concentration_rising
    orch = _portfolio_concentration_trend_orch()
    orch._diagnose(_pct_perception(0.2))
    orch._diagnose(_pct_perception(0.3))
    alerts = orch._diagnose(_pct_perception(0.4))
    rising = [a for a in alerts if a["type"] == "portfolio_concentration_rising"]
    assert len(rising) == 1
    assert rising[0]["concentration"] == pytest.approx(0.4)


def test_portfolio_concentration_trend_no_alert_when_stable():
    # 集中度平稳（非严格上升）→ 不告警
    orch = _portfolio_concentration_trend_orch()
    orch._diagnose(_pct_perception(0.3))
    orch._diagnose(_pct_perception(0.3))
    alerts = orch._diagnose(_pct_perception(0.3))
    assert not [a for a in alerts if a["type"] == "portfolio_concentration_rising"]


def test_portfolio_concentration_trend_disabled():
    orch = _portfolio_concentration_trend_orch(enabled=False)
    orch._diagnose(_pct_perception(0.2))
    orch._diagnose(_pct_perception(0.3))
    alerts = orch._diagnose(_pct_perception(0.4))
    assert not [a for a in alerts if a["type"] == "portfolio_concentration_rising"]


def test_portfolio_concentration_trend_actions_generate():
    # portfolio_concentration_rising → reallocate decrease 降最高权重策略到 reduce_target
    orch = _portfolio_concentration_trend_orch()
    alerts = [{"type": "portfolio_concentration_rising", "concentration": 0.4}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.5},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._portfolio_concentration_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.2)


def test_portfolio_concentration_trend_actions_skip_low_weight():
    # 最高权重已 <= reduce_target → 不再生成降权动作
    orch = _portfolio_concentration_trend_orch()
    alerts = [{"type": "portfolio_concentration_rising", "concentration": 0.4}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.15}}}}
    assert orch._portfolio_concentration_trend_actions(decision, alerts) == []


def test_offensive_suppressed_by_portfolio_concentration_trend():
    # 组合集中度持续上升 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "portfolio_concentration_trend_guard": {"enabled": True, "window": 3, "reduce_target": 0.2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "portfolio_concentration_rising", "concentration": 0.4},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 策略级手续费率守卫（strategy_fee_ratio_guard）──────────────────

def _strategy_fee_ratio_orch(**overrides):
    cfg = {"agi_orchestrator": {"strategy_fee_ratio_guard": {
        "enabled": True, "fee_ratio_threshold": 0.3, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["strategy_fee_ratio_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _strategy_fee_ratio_perception(total_pnl, total_fees):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": 20,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
        }}},
        "freeze_state": {},
    }


def test_strategy_fee_ratio_detects_high():
    # 手续费 40 / 毛利 100 = 0.4 ≥ 0.3 → 过度交易告警
    orch = _strategy_fee_ratio_orch()
    alerts = orch._diagnose(_strategy_fee_ratio_perception(total_pnl=60.0, total_fees=40.0))
    hits = [a for a in alerts if a["type"] == "high_strategy_fee"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["fee_ratio"] == pytest.approx(0.4)


def test_strategy_fee_ratio_below_threshold():
    # 手续费 10 / 毛利 100 = 0.1 < 0.3 → 不触发
    orch = _strategy_fee_ratio_orch()
    alerts = orch._diagnose(_strategy_fee_ratio_perception(total_pnl=90.0, total_fees=10.0))
    assert not [a for a in alerts if a["type"] == "high_strategy_fee"]


def test_strategy_fee_ratio_negative_pnl_ignored():
    # 总盈亏非正 → 不触发（仅对有毛利的策略评估手续费侵蚀）
    orch = _strategy_fee_ratio_orch()
    alerts = orch._diagnose(_strategy_fee_ratio_perception(total_pnl=-50.0, total_fees=40.0))
    assert not [a for a in alerts if a["type"] == "high_strategy_fee"]


def test_strategy_fee_ratio_disabled():
    orch = _strategy_fee_ratio_orch(enabled=False)
    alerts = orch._diagnose(_strategy_fee_ratio_perception(total_pnl=60.0, total_fees=40.0))
    assert not [a for a in alerts if a["type"] == "high_strategy_fee"]


def test_strategy_fee_ratio_actions_generate():
    orch = _strategy_fee_ratio_orch()
    alerts = [{"type": "high_strategy_fee", "strategy": "grid", "message": "手续费侵蚀过高"}]
    actions = orch._strategy_fee_ratio_actions({}, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_strategy_fee_ratio_actions_dedup():
    orch = _strategy_fee_ratio_orch()
    alerts = [
        {"type": "high_strategy_fee", "strategy": "grid"},
        {"type": "high_strategy_fee", "strategy": "grid"},
    ]
    assert len(orch._strategy_fee_ratio_actions({}, alerts)) == 1


def test_strategy_fee_ratio_actions_disabled():
    orch = _strategy_fee_ratio_orch(enabled=False)
    alerts = [{"type": "high_strategy_fee", "strategy": "grid"}]
    assert orch._strategy_fee_ratio_actions({}, alerts) == []


def test_offensive_suppressed_by_strategy_fee_ratio():
    # 手续费侵蚀过高的策略不进攻加仓（per-strategy gate，交易频率维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "strategy_fee_ratio_guard": {"enabled": True, "fee_ratio_threshold": 0.3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "high_strategy_fee", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_strategy_fee_ratio():
    # 手续费侵蚀过高的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "strategy_fee_ratio_guard": {"enabled": True, "fee_ratio_threshold": 0.3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "high_strategy_fee", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级资金费率守卫（funding_cost_guard）──────────────────

def _funding_cost_orch(**overrides):
    cfg = {"agi_orchestrator": {"funding_cost_guard": {
        "enabled": True, "funding_ratio_threshold": 0.3, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["funding_cost_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _funding_cost_perception(total_pnl, total_fees, total_funding_cost):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": 20,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "total_funding_cost": total_funding_cost,
        }}},
        "freeze_state": {},
    }


def test_funding_cost_detects_high():
    # 资金费 40 / 毛利 60 = 0.667 ≥ 0.3 → 持仓时间成本侵蚀告警
    orch = _funding_cost_orch()
    alerts = orch._diagnose(_funding_cost_perception(total_pnl=60.0, total_fees=0.0, total_funding_cost=40.0))
    hits = [a for a in alerts if a["type"] == "high_funding_cost"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["funding_ratio"] == pytest.approx(40.0 / 60.0)


def test_funding_cost_below_threshold():
    # 资金费 10 / 毛利 90 = 0.111 < 0.3 → 不触发
    orch = _funding_cost_orch()
    alerts = orch._diagnose(_funding_cost_perception(total_pnl=90.0, total_fees=0.0, total_funding_cost=10.0))
    assert not [a for a in alerts if a["type"] == "high_funding_cost"]


def test_funding_cost_negative_funding_ignored():
    # 收到资金费（负值）不算成本 → 不触发
    orch = _funding_cost_orch()
    alerts = orch._diagnose(_funding_cost_perception(total_pnl=90.0, total_fees=0.0, total_funding_cost=-10.0))
    assert not [a for a in alerts if a["type"] == "high_funding_cost"]


def test_funding_cost_negative_pnl_ignored():
    # 总盈亏非正 → 不触发（仅对有毛利的策略评估持仓成本侵蚀）
    orch = _funding_cost_orch()
    alerts = orch._diagnose(_funding_cost_perception(total_pnl=-50.0, total_fees=0.0, total_funding_cost=40.0))
    assert not [a for a in alerts if a["type"] == "high_funding_cost"]


def test_funding_cost_disabled():
    orch = _funding_cost_orch(enabled=False)
    alerts = orch._diagnose(_funding_cost_perception(total_pnl=60.0, total_fees=0.0, total_funding_cost=40.0))
    assert not [a for a in alerts if a["type"] == "high_funding_cost"]


def test_funding_cost_actions_generate():
    orch = _funding_cost_orch()
    alerts = [{"type": "high_funding_cost", "strategy": "grid", "message": "资金费侵蚀过高"}]
    actions = orch._funding_cost_actions({}, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_funding_cost_actions_dedup():
    orch = _funding_cost_orch()
    alerts = [
        {"type": "high_funding_cost", "strategy": "grid"},
        {"type": "high_funding_cost", "strategy": "grid"},
    ]
    assert len(orch._funding_cost_actions({}, alerts)) == 1


def test_funding_cost_actions_disabled():
    orch = _funding_cost_orch(enabled=False)
    alerts = [{"type": "high_funding_cost", "strategy": "grid"}]
    assert orch._funding_cost_actions({}, alerts) == []


def test_offensive_suppressed_by_funding_cost():
    # 资金费侵蚀过高的策略不进攻加仓（per-strategy gate，持仓时间成本维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "funding_cost_guard": {"enabled": True, "funding_ratio_threshold": 0.3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "high_funding_cost", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_funding_cost():
    # 资金费侵蚀过高的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "funding_cost_guard": {"enabled": True, "funding_ratio_threshold": 0.3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "high_funding_cost", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级多空方向失衡守卫（long_short_imbalance_guard）──────────────────

def _long_short_imbalance_orch(**overrides):
    cfg = {"agi_orchestrator": {"long_short_imbalance_guard": {
        "enabled": True, "min_trades": 10, "loss_ratio_threshold": 0.3,
        "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["long_short_imbalance_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _long_short_imbalance_perception(long_pnl, short_pnl, long_trades, short_trades):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": long_trades + short_trades,
            "total_pnl": long_pnl + short_pnl,
            "long_pnl": long_pnl,
            "short_pnl": short_pnl,
            "long_trades": long_trades,
            "short_trades": short_trades,
        }}},
        "freeze_state": {},
    }


def test_long_short_imbalance_detects_long_losing():
    # 多头 -30 / 空头 +10，多头亏损占比 30/40 = 0.75 ≥ 0.3 → long_side_losing
    orch = _long_short_imbalance_orch()
    alerts = orch._diagnose(_long_short_imbalance_perception(
        long_pnl=-30.0, short_pnl=10.0, long_trades=20, short_trades=20))
    hits = [a for a in alerts if a["type"] == "long_side_losing"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["loss_ratio"] == pytest.approx(30.0 / 40.0)


def test_long_short_imbalance_detects_short_losing():
    # 多头 +10 / 空头 -30，空头亏损占比 30/40 = 0.75 ≥ 0.3 → short_side_losing
    orch = _long_short_imbalance_orch()
    alerts = orch._diagnose(_long_short_imbalance_perception(
        long_pnl=10.0, short_pnl=-30.0, long_trades=20, short_trades=20))
    hits = [a for a in alerts if a["type"] == "short_side_losing"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["loss_ratio"] == pytest.approx(30.0 / 40.0)


def test_long_short_imbalance_below_threshold():
    # 多头 -10 / 空头 +30，多头亏损占比 10/40 = 0.25 < 0.3 → 不触发
    orch = _long_short_imbalance_orch()
    alerts = orch._diagnose(_long_short_imbalance_perception(
        long_pnl=-10.0, short_pnl=30.0, long_trades=20, short_trades=20))
    assert not [a for a in alerts if a["type"] in ("long_side_losing", "short_side_losing")]


def test_long_short_imbalance_insufficient_trades():
    # 单边交易笔数不足 min_trades → 不触发
    orch = _long_short_imbalance_orch()
    alerts = orch._diagnose(_long_short_imbalance_perception(
        long_pnl=-30.0, short_pnl=10.0, long_trades=20, short_trades=5))
    assert not [a for a in alerts if a["type"] in ("long_side_losing", "short_side_losing")]


def test_long_short_imbalance_disabled():
    orch = _long_short_imbalance_orch(enabled=False)
    alerts = orch._diagnose(_long_short_imbalance_perception(
        long_pnl=-30.0, short_pnl=10.0, long_trades=20, short_trades=20))
    assert not [a for a in alerts if a["type"] in ("long_side_losing", "short_side_losing")]


def test_long_short_imbalance_actions_generate():
    orch = _long_short_imbalance_orch()
    alerts = [{"type": "long_side_losing", "strategy": "grid", "message": "多头方向持续亏损"}]
    actions = orch._long_short_imbalance_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["value"] == pytest.approx(1.0)


def test_long_short_imbalance_actions_dedup():
    orch = _long_short_imbalance_orch()
    alerts = [
        {"type": "long_side_losing", "strategy": "grid"},
        {"type": "short_side_losing", "strategy": "grid"},
    ]
    assert len(orch._long_short_imbalance_actions(alerts)) == 1


def test_long_short_imbalance_actions_disabled():
    orch = _long_short_imbalance_orch(enabled=False)
    alerts = [{"type": "long_side_losing", "strategy": "grid"}]
    assert orch._long_short_imbalance_actions(alerts) == []


def test_offensive_suppressed_by_long_short_imbalance():
    # 方向失衡的策略不进攻加仓（per-strategy gate，方向判断质量维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "long_short_imbalance_guard": {
            "enabled": True, "min_trades": 10, "loss_ratio_threshold": 0.3,
            "deterioration_leverage": 1.0,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "long_side_losing", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_long_short_imbalance():
    # 方向失衡的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "long_short_imbalance_guard": {
            "enabled": True, "min_trades": 10, "loss_ratio_threshold": 0.3,
            "deterioration_leverage": 1.0,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "long_side_losing", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级止盈止损比守卫（take_profit_ratio_guard）──────────────────

def _take_profit_ratio_orch(**overrides):
    cfg = {"agi_orchestrator": {"take_profit_ratio_guard": {
        "enabled": True, "min_trades": 10, "min_take_profit_ratio": 0.5,
        "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["take_profit_ratio_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _take_profit_ratio_perception(total_trades, take_profit_count, stop_loss_count):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 10.0,
            "take_profit_count": take_profit_count,
            "stop_loss_count": stop_loss_count,
        }}},
        "freeze_state": {},
    }


def test_take_profit_ratio_detects_low():
    # 止盈 5 / 止损 20 = 0.25 < 0.5 → low_take_profit_ratio
    orch = _take_profit_ratio_orch()
    alerts = orch._diagnose(_take_profit_ratio_perception(
        total_trades=25, take_profit_count=5, stop_loss_count=20))
    hits = [a for a in alerts if a["type"] == "low_take_profit_ratio"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["take_profit_ratio"] == pytest.approx(5.0 / 20.0)


def test_take_profit_ratio_above_threshold():
    # 止盈 15 / 止损 20 = 0.75 ≥ 0.5 → 不触发
    orch = _take_profit_ratio_orch()
    alerts = orch._diagnose(_take_profit_ratio_perception(
        total_trades=35, take_profit_count=15, stop_loss_count=20))
    assert not [a for a in alerts if a["type"] == "low_take_profit_ratio"]


def test_take_profit_ratio_no_stop_loss_ignored():
    # 无止损单（stop_loss_count=0）→ 不评估止盈止损比，不触发
    orch = _take_profit_ratio_orch()
    alerts = orch._diagnose(_take_profit_ratio_perception(
        total_trades=20, take_profit_count=20, stop_loss_count=0))
    assert not [a for a in alerts if a["type"] == "low_take_profit_ratio"]


def test_take_profit_ratio_insufficient_trades():
    # 交易笔数不足 min_trades → 不触发
    orch = _take_profit_ratio_orch()
    alerts = orch._diagnose(_take_profit_ratio_perception(
        total_trades=8, take_profit_count=2, stop_loss_count=6))
    assert not [a for a in alerts if a["type"] == "low_take_profit_ratio"]


def test_take_profit_ratio_disabled():
    orch = _take_profit_ratio_orch(enabled=False)
    alerts = orch._diagnose(_take_profit_ratio_perception(
        total_trades=25, take_profit_count=5, stop_loss_count=20))
    assert not [a for a in alerts if a["type"] == "low_take_profit_ratio"]


def test_take_profit_ratio_actions_generate():
    orch = _take_profit_ratio_orch()
    alerts = [{"type": "low_take_profit_ratio", "strategy": "grid", "message": "止盈止损比过低"}]
    actions = orch._take_profit_ratio_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["value"] == pytest.approx(1.0)


def test_take_profit_ratio_actions_dedup():
    orch = _take_profit_ratio_orch()
    alerts = [
        {"type": "low_take_profit_ratio", "strategy": "grid"},
        {"type": "low_take_profit_ratio", "strategy": "grid"},
    ]
    assert len(orch._take_profit_ratio_actions(alerts)) == 1


def test_take_profit_ratio_actions_disabled():
    orch = _take_profit_ratio_orch(enabled=False)
    alerts = [{"type": "low_take_profit_ratio", "strategy": "grid"}]
    assert orch._take_profit_ratio_actions(alerts) == []


def test_offensive_suppressed_by_take_profit_ratio():
    # 止盈止损比过低的策略不进攻加仓（per-strategy gate，离场质量维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "take_profit_ratio_guard": {
            "enabled": True, "min_trades": 10, "min_take_profit_ratio": 0.5,
            "deterioration_leverage": 1.0,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "low_take_profit_ratio", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_take_profit_ratio():
    # 止盈止损比过低的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "take_profit_ratio_guard": {
            "enabled": True, "min_trades": 10, "min_take_profit_ratio": 0.5,
            "deterioration_leverage": 1.0,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "low_take_profit_ratio", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级盈亏比守卫（win_loss_ratio_guard）──────────────────

def _win_loss_ratio_orch(**overrides):
    cfg = {"agi_orchestrator": {"win_loss_ratio_guard": {
        "enabled": True, "min_trades": 10, "min_win_loss_ratio": 1.0,
        "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["win_loss_ratio_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _win_loss_ratio_perception(avg_win, avg_loss, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 10.0,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
        }}},
        "freeze_state": {},
    }


def test_win_loss_ratio_detects_low():
    # 平均盈利 1.0 / 平均亏损 4.0 = 0.25 < 1.0 → low_win_loss_ratio
    orch = _win_loss_ratio_orch()
    alerts = orch._diagnose(_win_loss_ratio_perception(avg_win=1.0, avg_loss=4.0))
    hits = [a for a in alerts if a["type"] == "low_win_loss_ratio"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["win_loss_ratio"] == pytest.approx(1.0 / 4.0)


def test_win_loss_ratio_above_threshold():
    # 平均盈利 4.0 / 平均亏损 2.0 = 2.0 ≥ 1.0 → 不触发
    orch = _win_loss_ratio_orch()
    alerts = orch._diagnose(_win_loss_ratio_perception(avg_win=4.0, avg_loss=2.0))
    assert not [a for a in alerts if a["type"] == "low_win_loss_ratio"]


def test_win_loss_ratio_no_loss_ignored():
    # 无亏损单（avg_loss=0）→ 不评估盈亏比，不触发
    orch = _win_loss_ratio_orch()
    alerts = orch._diagnose(_win_loss_ratio_perception(avg_win=1.0, avg_loss=0.0))
    assert not [a for a in alerts if a["type"] == "low_win_loss_ratio"]


def test_win_loss_ratio_no_win_ignored():
    # 无盈利单（avg_win=0）→ 不评估盈亏比，不触发
    orch = _win_loss_ratio_orch()
    alerts = orch._diagnose(_win_loss_ratio_perception(avg_win=0.0, avg_loss=4.0))
    assert not [a for a in alerts if a["type"] == "low_win_loss_ratio"]


def test_win_loss_ratio_insufficient_trades():
    # 交易笔数不足 min_trades → 不触发
    orch = _win_loss_ratio_orch()
    alerts = orch._diagnose(_win_loss_ratio_perception(avg_win=1.0, avg_loss=4.0, total_trades=8))
    assert not [a for a in alerts if a["type"] == "low_win_loss_ratio"]


def test_win_loss_ratio_disabled():
    orch = _win_loss_ratio_orch(enabled=False)
    alerts = orch._diagnose(_win_loss_ratio_perception(avg_win=1.0, avg_loss=4.0))
    assert not [a for a in alerts if a["type"] == "low_win_loss_ratio"]


def test_win_loss_ratio_actions_generate():
    orch = _win_loss_ratio_orch()
    alerts = [{"type": "low_win_loss_ratio", "strategy": "grid", "message": "盈亏比过低"}]
    actions = orch._win_loss_ratio_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["value"] == pytest.approx(1.0)


def test_win_loss_ratio_actions_dedup():
    orch = _win_loss_ratio_orch()
    alerts = [
        {"type": "low_win_loss_ratio", "strategy": "grid"},
        {"type": "low_win_loss_ratio", "strategy": "grid"},
    ]
    assert len(orch._win_loss_ratio_actions(alerts)) == 1


def test_win_loss_ratio_actions_disabled():
    orch = _win_loss_ratio_orch(enabled=False)
    alerts = [{"type": "low_win_loss_ratio", "strategy": "grid"}]
    assert orch._win_loss_ratio_actions(alerts) == []


def test_offensive_suppressed_by_win_loss_ratio():
    # 盈亏比过低的策略不进攻加仓（per-strategy gate，单笔盈亏结构维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "win_loss_ratio_guard": {
            "enabled": True, "min_trades": 10, "min_win_loss_ratio": 1.0,
            "deterioration_leverage": 1.0,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "low_win_loss_ratio", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_win_loss_ratio():
    # 盈亏比过低的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "win_loss_ratio_guard": {
            "enabled": True, "min_trades": 10, "min_win_loss_ratio": 1.0,
            "deterioration_leverage": 1.0,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "low_win_loss_ratio", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级止盈止损盈亏金额比守卫（take_profit_pnl_ratio_guard）──────────────────

def _take_profit_pnl_ratio_orch(**overrides):
    cfg = {"agi_orchestrator": {"take_profit_pnl_ratio_guard": {
        "enabled": True, "min_trades": 10, "min_take_profit_pnl_ratio": 1.0,
        "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["take_profit_pnl_ratio_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _take_profit_pnl_ratio_perception(take_profit_pnl, stop_loss_pnl, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 10.0,
            "take_profit_pnl": take_profit_pnl,
            "stop_loss_pnl": stop_loss_pnl,
        }}},
        "freeze_state": {},
    }


def test_take_profit_pnl_ratio_detects_low():
    # 止盈累计 50 / 止损累计 |−100| = 0.5 < 1.0 → low_take_profit_pnl_ratio
    orch = _take_profit_pnl_ratio_orch()
    alerts = orch._diagnose(_take_profit_pnl_ratio_perception(
        take_profit_pnl=50.0, stop_loss_pnl=-100.0))
    hits = [a for a in alerts if a["type"] == "low_take_profit_pnl_ratio"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["take_profit_pnl_ratio"] == pytest.approx(50.0 / 100.0)


def test_take_profit_pnl_ratio_above_threshold():
    # 止盈累计 200 / 止损累计 |−100| = 2.0 ≥ 1.0 → 不触发
    orch = _take_profit_pnl_ratio_orch()
    alerts = orch._diagnose(_take_profit_pnl_ratio_perception(
        take_profit_pnl=200.0, stop_loss_pnl=-100.0))
    assert not [a for a in alerts if a["type"] == "low_take_profit_pnl_ratio"]


def test_take_profit_pnl_ratio_no_stop_loss_ignored():
    # 无止损亏损（stop_loss_pnl>=0）→ 不评估，不触发
    orch = _take_profit_pnl_ratio_orch()
    alerts = orch._diagnose(_take_profit_pnl_ratio_perception(
        take_profit_pnl=50.0, stop_loss_pnl=0.0))
    assert not [a for a in alerts if a["type"] == "low_take_profit_pnl_ratio"]


def test_take_profit_pnl_ratio_no_take_profit_ignored():
    # 无止盈盈利（take_profit_pnl<=0）→ 不评估，不触发
    orch = _take_profit_pnl_ratio_orch()
    alerts = orch._diagnose(_take_profit_pnl_ratio_perception(
        take_profit_pnl=0.0, stop_loss_pnl=-100.0))
    assert not [a for a in alerts if a["type"] == "low_take_profit_pnl_ratio"]


def test_take_profit_pnl_ratio_insufficient_trades():
    # 交易笔数不足 min_trades → 不触发
    orch = _take_profit_pnl_ratio_orch()
    alerts = orch._diagnose(_take_profit_pnl_ratio_perception(
        take_profit_pnl=50.0, stop_loss_pnl=-100.0, total_trades=8))
    assert not [a for a in alerts if a["type"] == "low_take_profit_pnl_ratio"]


def test_take_profit_pnl_ratio_disabled():
    orch = _take_profit_pnl_ratio_orch(enabled=False)
    alerts = orch._diagnose(_take_profit_pnl_ratio_perception(
        take_profit_pnl=50.0, stop_loss_pnl=-100.0))
    assert not [a for a in alerts if a["type"] == "low_take_profit_pnl_ratio"]


def test_take_profit_pnl_ratio_actions_generate():
    orch = _take_profit_pnl_ratio_orch()
    alerts = [{"type": "low_take_profit_pnl_ratio", "strategy": "grid", "message": "盈亏金额比过低"}]
    actions = orch._take_profit_pnl_ratio_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["value"] == pytest.approx(1.0)


def test_take_profit_pnl_ratio_actions_dedup():
    orch = _take_profit_pnl_ratio_orch()
    alerts = [
        {"type": "low_take_profit_pnl_ratio", "strategy": "grid"},
        {"type": "low_take_profit_pnl_ratio", "strategy": "grid"},
    ]
    assert len(orch._take_profit_pnl_ratio_actions(alerts)) == 1


def test_take_profit_pnl_ratio_actions_disabled():
    orch = _take_profit_pnl_ratio_orch(enabled=False)
    alerts = [{"type": "low_take_profit_pnl_ratio", "strategy": "grid"}]
    assert orch._take_profit_pnl_ratio_actions(alerts) == []


def test_offensive_suppressed_by_take_profit_pnl_ratio():
    # 止盈止损盈亏金额比过低的策略不进攻加仓（per-strategy gate，累计金额仓位结构维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "take_profit_pnl_ratio_guard": {
            "enabled": True, "min_trades": 10, "min_take_profit_pnl_ratio": 1.0,
            "deterioration_leverage": 1.0,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "low_take_profit_pnl_ratio", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_take_profit_pnl_ratio():
    # 止盈止损盈亏金额比过低的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "take_profit_pnl_ratio_guard": {
            "enabled": True, "min_trades": 10, "min_take_profit_pnl_ratio": 1.0,
            "deterioration_leverage": 1.0,
        },
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "low_take_profit_pnl_ratio", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级执行质量成本守卫（execution_cost_guard）──────────────────

def _execution_cost_orch(**overrides):
    cfg = {"agi_orchestrator": {"execution_cost_guard": {
        "enabled": True, "cost_ratio_threshold": 0.3, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["execution_cost_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _execution_cost_perception(total_pnl, total_fees, slippage, spread):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": 20,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "total_slippage_cost": slippage,
            "total_spread_cost": spread,
        }}},
        "freeze_state": {},
    }


def test_execution_cost_detects_high():
    # 执行成本 (20+10)=30 / 毛利 60 = 0.5 ≥ 0.3 → high_execution_cost
    orch = _execution_cost_orch()
    alerts = orch._diagnose(_execution_cost_perception(
        total_pnl=60.0, total_fees=0.0, slippage=20.0, spread=10.0))
    hits = [a for a in alerts if a["type"] == "high_execution_cost"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["execution_cost_ratio"] == pytest.approx(30.0 / 60.0)


def test_execution_cost_below_threshold():
    # 执行成本 (5+5)=10 / 毛利 90 = 0.111 < 0.3 → 不触发
    orch = _execution_cost_orch()
    alerts = orch._diagnose(_execution_cost_perception(
        total_pnl=90.0, total_fees=0.0, slippage=5.0, spread=5.0))
    assert not [a for a in alerts if a["type"] == "high_execution_cost"]


def test_execution_cost_zero_ignored():
    # 无滑点/点差成本（exec_cost=0）→ 不触发
    orch = _execution_cost_orch()
    alerts = orch._diagnose(_execution_cost_perception(
        total_pnl=90.0, total_fees=0.0, slippage=0.0, spread=0.0))
    assert not [a for a in alerts if a["type"] == "high_execution_cost"]


def test_execution_cost_negative_pnl_ignored():
    # 总盈亏非正 → 不触发
    orch = _execution_cost_orch()
    alerts = orch._diagnose(_execution_cost_perception(
        total_pnl=-50.0, total_fees=0.0, slippage=20.0, spread=10.0))
    assert not [a for a in alerts if a["type"] == "high_execution_cost"]


def test_execution_cost_disabled():
    orch = _execution_cost_orch(enabled=False)
    alerts = orch._diagnose(_execution_cost_perception(
        total_pnl=60.0, total_fees=0.0, slippage=20.0, spread=10.0))
    assert not [a for a in alerts if a["type"] == "high_execution_cost"]


def test_execution_cost_actions_generate():
    orch = _execution_cost_orch()
    alerts = [{"type": "high_execution_cost", "strategy": "grid", "message": "执行成本侵蚀过高"}]
    actions = orch._execution_cost_actions({}, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_execution_cost_actions_dedup():
    orch = _execution_cost_orch()
    alerts = [
        {"type": "high_execution_cost", "strategy": "grid"},
        {"type": "high_execution_cost", "strategy": "grid"},
    ]
    assert len(orch._execution_cost_actions({}, alerts)) == 1


def test_execution_cost_actions_disabled():
    orch = _execution_cost_orch(enabled=False)
    alerts = [{"type": "high_execution_cost", "strategy": "grid"}]
    assert orch._execution_cost_actions({}, alerts) == []


def test_offensive_suppressed_by_execution_cost():
    # 执行质量成本过高的策略不进攻加仓（per-strategy gate，执行成本维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "execution_cost_guard": {"enabled": True, "cost_ratio_threshold": 0.3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "high_execution_cost", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_execution_cost():
    # 执行质量成本过高的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "execution_cost_guard": {"enabled": True, "cost_ratio_threshold": 0.3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "high_execution_cost", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级 PnL 动量趋势外推（pnl_momentum_trend_guard）──────────────────

def _pnl_momentum_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"pnl_momentum_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["pnl_momentum_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _pnl_momentum_trend_perception(momentum, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "max_drawdown": 0.1,
            "trend_pnl_7d_vs_30d": momentum,
        }}},
        "freeze_state": {},
    }


def test_pnl_momentum_trend_detects_deteriorating():
    # PnL 动量连续 3 周期严格下降（0.9→0.6→0.3）→ pnl_momentum_deteriorating
    orch = _pnl_momentum_trend_orch()
    orch._strategy_pnl_momentum_history["grid"] = deque([0.9, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_momentum_trend_perception(momentum=0.3))
    deteriorating = [a for a in alerts if a["type"] == "pnl_momentum_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["pnl_momentum"] == pytest.approx(0.3)


def test_pnl_momentum_trend_no_alert_when_rising():
    # PnL 动量上升（非严格下降）→ 不告警
    orch = _pnl_momentum_trend_orch()
    orch._strategy_pnl_momentum_history["grid"] = deque([0.3, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_momentum_trend_perception(momentum=0.9))
    assert not [a for a in alerts if a["type"] == "pnl_momentum_deteriorating"]


def test_pnl_momentum_trend_zero_ignored():
    # PnL 动量为 0（无数据，恒 0）→ 不追踪，避免误触发
    orch = _pnl_momentum_trend_orch()
    orch._strategy_pnl_momentum_history["grid"] = deque([0.9, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_momentum_trend_perception(momentum=0.0))
    assert not [a for a in alerts if a["type"] == "pnl_momentum_deteriorating"]


def test_pnl_momentum_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免样本不足噪声误触发
    orch = _pnl_momentum_trend_orch(min_trades=10)
    orch._strategy_pnl_momentum_history["grid"] = deque([0.9, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_momentum_trend_perception(momentum=0.3, total_trades=5))
    assert not [a for a in alerts if a["type"] == "pnl_momentum_deteriorating"]


def test_pnl_momentum_trend_disabled():
    orch = _pnl_momentum_trend_orch(enabled=False)
    orch._strategy_pnl_momentum_history["grid"] = deque([0.9, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_momentum_trend_perception(momentum=0.3))
    assert not [a for a in alerts if a["type"] == "pnl_momentum_deteriorating"]


def test_pnl_momentum_trend_actions_generate():
    orch = _pnl_momentum_trend_orch()
    alerts = [{"type": "pnl_momentum_deteriorating", "strategy": "grid", "message": "连续下降"}]
    actions = orch._pnl_momentum_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_pnl_momentum_trend_actions_dedup():
    orch = _pnl_momentum_trend_orch()
    alerts = [
        {"type": "pnl_momentum_deteriorating", "strategy": "grid"},
        {"type": "pnl_momentum_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._pnl_momentum_trend_actions(alerts)) == 1


def test_pnl_momentum_trend_actions_disabled():
    orch = _pnl_momentum_trend_orch(enabled=False)
    alerts = [{"type": "pnl_momentum_deteriorating", "strategy": "grid"}]
    assert orch._pnl_momentum_trend_actions(alerts) == []


def test_pnl_momentum_trend_persist_restore(tmp_path):
    orch = _pnl_momentum_trend_orch()
    orch._diagnose(_pnl_momentum_trend_perception(momentum=0.9))
    orch._diagnose(_pnl_momentum_trend_perception(momentum=0.6))
    state_file = tmp_path / "test_state.json"
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _pnl_momentum_trend_orch()
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._strategy_pnl_momentum_history["grid"]) == [
        pytest.approx(0.9), pytest.approx(0.6),
    ]


def test_offensive_suppressed_by_pnl_momentum_trend():
    # PnL 动量趋势恶化的策略不进攻加仓（per-strategy gate，收益趋势维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "pnl_momentum_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "pnl_momentum_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_pnl_momentum_trend():
    # PnL 动量趋势恶化的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "pnl_momentum_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "pnl_momentum_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级单笔期望值趋势外推（pnl_per_trade_trend_guard）──────────────────

def _pnl_per_trade_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"pnl_per_trade_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    state_path = overrides.pop("state_path", None)
    if state_path is not None:
        cfg["agi_orchestrator"]["state_path"] = str(state_path)
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["pnl_per_trade_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _pnl_per_trade_trend_perception(pnl_per_trade, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "max_drawdown": 0.1,
            "pnl_per_trade": pnl_per_trade,
        }}},
        "freeze_state": {},
    }


def test_pnl_per_trade_trend_detects_deteriorating():
    # 单笔期望值连续 3 周期严格下降（0.9→0.6→0.3）→ pnl_per_trade_deteriorating
    orch = _pnl_per_trade_trend_orch()
    orch._strategy_pnl_per_trade_history["grid"] = deque([0.9, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_per_trade_trend_perception(pnl_per_trade=0.3))
    deteriorating = [a for a in alerts if a["type"] == "pnl_per_trade_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["pnl_per_trade"] == pytest.approx(0.3)


def test_pnl_per_trade_trend_no_alert_when_rising():
    # 单笔期望值上升（非严格下降）→ 不告警
    orch = _pnl_per_trade_trend_orch()
    orch._strategy_pnl_per_trade_history["grid"] = deque([0.3, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_per_trade_trend_perception(pnl_per_trade=0.9))
    assert not [a for a in alerts if a["type"] == "pnl_per_trade_deteriorating"]


def test_pnl_per_trade_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免样本不足噪声误触发
    orch = _pnl_per_trade_trend_orch(min_trades=10)
    orch._strategy_pnl_per_trade_history["grid"] = deque([0.9, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_per_trade_trend_perception(pnl_per_trade=0.3, total_trades=5))
    assert not [a for a in alerts if a["type"] == "pnl_per_trade_deteriorating"]


def test_pnl_per_trade_trend_disabled():
    orch = _pnl_per_trade_trend_orch(enabled=False)
    orch._strategy_pnl_per_trade_history["grid"] = deque([0.9, 0.6], maxlen=3)
    alerts = orch._diagnose(_pnl_per_trade_trend_perception(pnl_per_trade=0.3))
    assert not [a for a in alerts if a["type"] == "pnl_per_trade_deteriorating"]


def test_pnl_per_trade_trend_actions_generate():
    orch = _pnl_per_trade_trend_orch()
    alerts = [{"type": "pnl_per_trade_deteriorating", "strategy": "grid", "message": "连续下降"}]
    actions = orch._pnl_per_trade_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_pnl_per_trade_trend_actions_dedup():
    orch = _pnl_per_trade_trend_orch()
    alerts = [
        {"type": "pnl_per_trade_deteriorating", "strategy": "grid"},
        {"type": "pnl_per_trade_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._pnl_per_trade_trend_actions(alerts)) == 1


def test_pnl_per_trade_trend_actions_disabled():
    orch = _pnl_per_trade_trend_orch(enabled=False)
    alerts = [{"type": "pnl_per_trade_deteriorating", "strategy": "grid"}]
    assert orch._pnl_per_trade_trend_actions(alerts) == []


def test_pnl_per_trade_trend_persist_restore(tmp_path):
    state_file = tmp_path / "test_state.json"
    orch = _pnl_per_trade_trend_orch(state_path=state_file)
    orch._diagnose(_pnl_per_trade_trend_perception(pnl_per_trade=0.9))
    orch._diagnose(_pnl_per_trade_trend_perception(pnl_per_trade=0.6))
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _pnl_per_trade_trend_orch(state_path=state_file)
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._strategy_pnl_per_trade_history["grid"]) == [
        pytest.approx(0.9), pytest.approx(0.6),
    ]


def test_offensive_suppressed_by_pnl_per_trade_trend():
    # 单笔期望值持续下降的策略不进攻加仓（per-strategy gate，交易质量维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "pnl_per_trade_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "pnl_per_trade_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_pnl_per_trade_trend():
    # 单笔期望值持续下降的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "pnl_per_trade_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "pnl_per_trade_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 策略级边际盈亏趋势外推（delta_pnl_trend_guard）──────────────────

def _delta_pnl_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"delta_pnl_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    state_path = overrides.pop("state_path", None)
    if state_path is not None:
        cfg["agi_orchestrator"]["state_path"] = str(state_path)
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["delta_pnl_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _delta_pnl_trend_perception(delta_pnl, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "max_drawdown": 0.1,
            "delta_pnl": delta_pnl,
        }}},
        "freeze_state": {},
    }


def test_delta_pnl_trend_detects_negative_streak():
    # 边际盈亏连续 3 周期为负（-0.1→-0.2→-0.3）→ delta_pnl_deteriorating
    orch = _delta_pnl_trend_orch()
    orch._strategy_delta_pnl_history["grid"] = deque([-0.1, -0.2], maxlen=3)
    alerts = orch._diagnose(_delta_pnl_trend_perception(delta_pnl=-0.3))
    deteriorating = [a for a in alerts if a["type"] == "delta_pnl_deteriorating"]
    assert len(deteriorating) == 1
    assert deteriorating[0]["strategy"] == "grid"
    assert deteriorating[0]["delta_pnl"] == pytest.approx(-0.3)


def test_delta_pnl_trend_no_alert_when_positive():
    # 边际盈亏为正（非负）→ 不告警
    orch = _delta_pnl_trend_orch()
    orch._strategy_delta_pnl_history["grid"] = deque([0.1, 0.2], maxlen=3)
    alerts = orch._diagnose(_delta_pnl_trend_perception(delta_pnl=0.3))
    assert not [a for a in alerts if a["type"] == "delta_pnl_deteriorating"]


def test_delta_pnl_trend_no_alert_when_mixed():
    # 边际盈亏正负混合（非连续为负）→ 不告警
    orch = _delta_pnl_trend_orch()
    orch._strategy_delta_pnl_history["grid"] = deque([-0.1, 0.2], maxlen=3)
    alerts = orch._diagnose(_delta_pnl_trend_perception(delta_pnl=-0.3))
    assert not [a for a in alerts if a["type"] == "delta_pnl_deteriorating"]


def test_delta_pnl_trend_insufficient_trades():
    # 交易笔数不足 min_trades 时不追踪，避免样本不足噪声误触发
    orch = _delta_pnl_trend_orch(min_trades=10)
    orch._strategy_delta_pnl_history["grid"] = deque([-0.1, -0.2], maxlen=3)
    alerts = orch._diagnose(_delta_pnl_trend_perception(delta_pnl=-0.3, total_trades=5))
    assert not [a for a in alerts if a["type"] == "delta_pnl_deteriorating"]


def test_delta_pnl_trend_disabled():
    orch = _delta_pnl_trend_orch(enabled=False)
    orch._strategy_delta_pnl_history["grid"] = deque([-0.1, -0.2], maxlen=3)
    alerts = orch._diagnose(_delta_pnl_trend_perception(delta_pnl=-0.3))
    assert not [a for a in alerts if a["type"] == "delta_pnl_deteriorating"]


def test_delta_pnl_trend_actions_generate():
    orch = _delta_pnl_trend_orch()
    alerts = [{"type": "delta_pnl_deteriorating", "strategy": "grid", "message": "连续为负"}]
    actions = orch._delta_pnl_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == 1.0


def test_delta_pnl_trend_actions_dedup():
    orch = _delta_pnl_trend_orch()
    alerts = [
        {"type": "delta_pnl_deteriorating", "strategy": "grid"},
        {"type": "delta_pnl_deteriorating", "strategy": "grid"},
    ]
    assert len(orch._delta_pnl_trend_actions(alerts)) == 1


def test_delta_pnl_trend_actions_disabled():
    orch = _delta_pnl_trend_orch(enabled=False)
    alerts = [{"type": "delta_pnl_deteriorating", "strategy": "grid"}]
    assert orch._delta_pnl_trend_actions(alerts) == []


def test_delta_pnl_trend_persist_restore(tmp_path):
    state_file = tmp_path / "test_state.json"
    orch = _delta_pnl_trend_orch(state_path=state_file)
    orch._diagnose(_delta_pnl_trend_perception(delta_pnl=-0.1))
    orch._diagnose(_delta_pnl_trend_perception(delta_pnl=-0.2))
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _delta_pnl_trend_orch(state_path=state_file)
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._strategy_delta_pnl_history["grid"]) == [
        pytest.approx(-0.1), pytest.approx(-0.2),
    ]


def test_offensive_suppressed_by_delta_pnl_trend():
    # 边际盈亏持续为负的策略不进攻加仓（per-strategy gate，持续失血维度）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "delta_pnl_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "delta_pnl_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_offensive_not_suppressed_for_other_strategy_by_delta_pnl_trend():
    # 边际盈亏持续为负的策略不进攻，但同周期其他策略仍可进攻加仓
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "delta_pnl_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid", "sync"], "boost_step": 0.05},
        {"type": "delta_pnl_deteriorating", "strategy": "grid"},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.5},
        "sync": {"target_weight": 0.1},
    }}}
    actions = orch._offensive_allocation_actions(decision, alerts)
    # grid 被抑制，sync 仍进攻
    assert len(actions) == 1
    assert actions[0]["strategy"] == "sync"


# ── 组合级相关性趋势外推（portfolio_correlation_trend_guard）──────────────────

def _portfolio_correlation_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"portfolio_correlation_trend_guard": {
        "enabled": True, "window": 3, "reduce_target": 0.2,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["portfolio_correlation_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _pcorr_perception(max_pair_correlation):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {}},
        "freeze_state": {},
        "correlation": {"max_pair_correlation": max_pair_correlation},
    }


def test_portfolio_correlation_trend_detects_rising():
    # 相关性连续 3 周期严格上升（0.4→0.5→0.6）→ portfolio_correlation_rising
    orch = _portfolio_correlation_trend_orch()
    orch._diagnose(_pcorr_perception(0.4))
    orch._diagnose(_pcorr_perception(0.5))
    alerts = orch._diagnose(_pcorr_perception(0.6))
    rising = [a for a in alerts if a["type"] == "portfolio_correlation_rising"]
    assert len(rising) == 1
    assert rising[0]["max_pair_correlation"] == pytest.approx(0.6)


def test_portfolio_correlation_trend_no_alert_when_stable():
    # 相关性平稳（非严格上升）→ 不告警
    orch = _portfolio_correlation_trend_orch()
    orch._diagnose(_pcorr_perception(0.5))
    orch._diagnose(_pcorr_perception(0.5))
    alerts = orch._diagnose(_pcorr_perception(0.5))
    assert not [a for a in alerts if a["type"] == "portfolio_correlation_rising"]


def test_portfolio_correlation_trend_disabled():
    orch = _portfolio_correlation_trend_orch(enabled=False)
    orch._diagnose(_pcorr_perception(0.4))
    orch._diagnose(_pcorr_perception(0.5))
    alerts = orch._diagnose(_pcorr_perception(0.6))
    assert not [a for a in alerts if a["type"] == "portfolio_correlation_rising"]


def test_portfolio_correlation_trend_actions_generate():
    # portfolio_correlation_rising → reallocate decrease 降最高权重策略到 reduce_target
    orch = _portfolio_correlation_trend_orch()
    alerts = [{"type": "portfolio_correlation_rising", "max_pair_correlation": 0.6}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.5},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._portfolio_correlation_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.2)


def test_portfolio_correlation_trend_actions_skip_low_weight():
    # 最高权重已 <= reduce_target → 不再生成降权动作
    orch = _portfolio_correlation_trend_orch()
    alerts = [{"type": "portfolio_correlation_rising", "max_pair_correlation": 0.6}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.15}}}}
    assert orch._portfolio_correlation_trend_actions(decision, alerts) == []


def test_offensive_suppressed_by_portfolio_correlation_trend():
    # 组合相关性持续上升 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "portfolio_correlation_trend_guard": {"enabled": True, "window": 3, "reduce_target": 0.2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "portfolio_correlation_rising", "max_pair_correlation": 0.6},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 组合级尾部风险趋势外推（portfolio_tail_risk_trend_guard）──────────────────

def _portfolio_tail_risk_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"portfolio_tail_risk_trend_guard": {
        "enabled": True, "window": 3, "reduce_target": 0.2,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["portfolio_tail_risk_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _ptr_perception(max_drawdown):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {"max_drawdown": max_drawdown}}},
        "freeze_state": {},
    }


def test_portfolio_tail_risk_trend_detects_rising():
    # 组合最深回撤连续 3 周期加深（0.1→0.2→0.3）→ tail_risk_rising
    orch = _portfolio_tail_risk_trend_orch()
    orch._diagnose(_ptr_perception(0.1))
    orch._diagnose(_ptr_perception(0.2))
    alerts = orch._diagnose(_ptr_perception(0.3))
    rising = [a for a in alerts if a["type"] == "tail_risk_rising"]
    assert len(rising) == 1
    assert rising[0]["tail_risk"] == pytest.approx(0.3)


def test_portfolio_tail_risk_trend_no_alert_when_stable():
    # 最深回撤平稳（非严格上升）→ 不告警
    orch = _portfolio_tail_risk_trend_orch()
    orch._diagnose(_ptr_perception(0.2))
    orch._diagnose(_ptr_perception(0.2))
    alerts = orch._diagnose(_ptr_perception(0.2))
    assert not [a for a in alerts if a["type"] == "tail_risk_rising"]


def test_portfolio_tail_risk_trend_disabled():
    orch = _portfolio_tail_risk_trend_orch(enabled=False)
    orch._diagnose(_ptr_perception(0.1))
    orch._diagnose(_ptr_perception(0.2))
    alerts = orch._diagnose(_ptr_perception(0.3))
    assert not [a for a in alerts if a["type"] == "tail_risk_rising"]


def test_portfolio_tail_risk_trend_actions_generate():
    # tail_risk_rising → reallocate decrease 降最高权重策略到 reduce_target
    orch = _portfolio_tail_risk_trend_orch()
    alerts = [{"type": "tail_risk_rising", "tail_risk": 0.3}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.5},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._portfolio_tail_risk_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.2)


def test_portfolio_tail_risk_trend_actions_skip_low_weight():
    # 最高权重已 <= reduce_target → 不再生成降权动作
    orch = _portfolio_tail_risk_trend_orch()
    alerts = [{"type": "tail_risk_rising", "tail_risk": 0.3}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.15}}}}
    assert orch._portfolio_tail_risk_trend_actions(decision, alerts) == []


def test_offensive_suppressed_by_portfolio_tail_risk_trend():
    # 组合尾部风险持续加深 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "portfolio_tail_risk_trend_guard": {"enabled": True, "window": 3, "reduce_target": 0.2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "tail_risk_rising", "tail_risk": 0.3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 组合级盈利集中度趋势外推（profit_concentration_trend_guard）──────────────────

def _profit_concentration_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"profit_concentration_trend_guard": {
        "enabled": True, "window": 3, "reduce_target": 0.2,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["profit_concentration_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_profit_concentration_trend_detects_rising():
    # 盈利来源集中度连续 3 周期上升（0.6→0.75→0.9）→ profit_concentration_rising
    orch = _profit_concentration_trend_orch()
    orch._diagnose(_profit_concentration_perception(
        {"sync": _pc_strategy(60.0), "grid": _pc_strategy(40.0)}, total_pnl=100.0))
    orch._diagnose(_profit_concentration_perception(
        {"sync": _pc_strategy(75.0), "grid": _pc_strategy(25.0)}, total_pnl=100.0))
    alerts = orch._diagnose(_profit_concentration_perception(
        {"sync": _pc_strategy(90.0), "grid": _pc_strategy(10.0)}, total_pnl=100.0))
    rising = [a for a in alerts if a["type"] == "profit_concentration_rising"]
    assert len(rising) == 1
    assert rising[0]["strategy"] == "sync"
    assert rising[0]["concentration"] == pytest.approx(0.9)


def test_profit_concentration_trend_no_alert_when_stable():
    # 集中度平稳（0.5，非严格上升）→ 不告警
    orch = _profit_concentration_trend_orch()
    for _ in range(3):
        orch._diagnose(_profit_concentration_perception(
            {"sync": _pc_strategy(50.0), "grid": _pc_strategy(50.0)}, total_pnl=100.0))
    alerts = orch._diagnose(_profit_concentration_perception(
        {"sync": _pc_strategy(50.0), "grid": _pc_strategy(50.0)}, total_pnl=100.0))
    assert not [a for a in alerts if a["type"] == "profit_concentration_rising"]


def test_profit_concentration_trend_disabled():
    orch = _profit_concentration_trend_orch(enabled=False)
    orch._diagnose(_profit_concentration_perception(
        {"sync": _pc_strategy(60.0), "grid": _pc_strategy(40.0)}, total_pnl=100.0))
    orch._diagnose(_profit_concentration_perception(
        {"sync": _pc_strategy(75.0), "grid": _pc_strategy(25.0)}, total_pnl=100.0))
    alerts = orch._diagnose(_profit_concentration_perception(
        {"sync": _pc_strategy(90.0), "grid": _pc_strategy(10.0)}, total_pnl=100.0))
    assert not [a for a in alerts if a["type"] == "profit_concentration_rising"]


def test_profit_concentration_trend_actions_generate():
    # profit_concentration_rising → reallocate decrease 降主导盈利策略到 reduce_target
    orch = _profit_concentration_trend_orch()
    alerts = [{"type": "profit_concentration_rising", "strategy": "sync", "concentration": 0.9}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.5},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._profit_concentration_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.2)


def test_profit_concentration_trend_actions_skip_low_weight():
    # 主导盈利策略权重已 <= reduce_target → 不再生成降权动作
    orch = _profit_concentration_trend_orch()
    alerts = [{"type": "profit_concentration_rising", "strategy": "sync", "concentration": 0.9}]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.15}}}}
    assert orch._profit_concentration_trend_actions(decision, alerts) == []


def test_offensive_suppressed_by_profit_concentration_trend():
    # 盈利来源集中度连续上升 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "profit_concentration_trend_guard": {"enabled": True, "window": 3, "reduce_target": 0.2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "profit_concentration_rising", "strategy": "sync", "concentration": 0.9},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 组合级健康度阈值收敛（portfolio_health_guard）──────────────────

def _portfolio_health_orch(**overrides):
    cfg = {"agi_orchestrator": {"portfolio_health_guard": {
        "enabled": True, "health_threshold": 45, "min_sample": 10, "reduce_target": 0.15,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["portfolio_health_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_portfolio_health_detects_low():
    # 组合整体健康度 40 < 45（阈值）→ portfolio_health_low
    orch = _portfolio_health_orch()
    alerts = orch._diagnose(_pht_perception(40.0))
    low = [a for a in alerts if a["type"] == "portfolio_health_low"]
    assert len(low) == 1
    assert low[0]["health_score"] == pytest.approx(40.0)


def test_portfolio_health_no_alert_when_healthy():
    # 组合整体健康度 80 ≥ 45 → 不告警
    orch = _portfolio_health_orch()
    alerts = orch._diagnose(_pht_perception(80.0))
    assert not [a for a in alerts if a["type"] == "portfolio_health_low"]


def test_portfolio_health_no_alert_low_sample():
    # 成交数 5 < min_sample 10 → 样本不足不评估，不告警
    orch = _portfolio_health_orch()
    alerts = orch._diagnose(_pht_perception(40.0, total_trades=5))
    assert not [a for a in alerts if a["type"] == "portfolio_health_low"]


def test_portfolio_health_disabled():
    orch = _portfolio_health_orch(enabled=False)
    alerts = orch._diagnose(_pht_perception(40.0))
    assert not [a for a in alerts if a["type"] == "portfolio_health_low"]


def test_portfolio_health_actions_generate():
    # portfolio_health_low → reallocate decrease 到 reduce_target
    orch = _portfolio_health_orch()
    alerts = [{"type": "portfolio_health_low", "health_score": 40.0}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.3},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._portfolio_health_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_offensive_suppressed_by_portfolio_health():
    # 组合整体健康度跌破健康线 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "portfolio_health_guard": {"enabled": True, "health_threshold": 45, "min_sample": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "portfolio_health_low", "health_score": 40.0},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 组合级协同度收敛（synergy_guard）──────────────────

def _synergy_orch(**overrides):
    cfg = {"agi_orchestrator": {"synergy_guard": {
        "enabled": True, "synergy_threshold": 0.3, "min_strategies": 2, "reduce_target": 0.15,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["synergy_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _synergy_perception(synergy_score, n_strategies=2):
    strategies = {f"s{i}": {"total_pnl": 10.0} for i in range(n_strategies)}
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {
            "synergy_score": synergy_score,
            "strategies": strategies,
            "total_trades": 20,
        },
        "freeze_state": {},
    }


def test_synergy_detects_low():
    # 协同度 0.2 < 0.3（阈值）→ low_synergy
    orch = _synergy_orch()
    alerts = orch._diagnose(_synergy_perception(0.2))
    low = [a for a in alerts if a["type"] == "low_synergy"]
    assert len(low) == 1
    assert low[0]["synergy_score"] == pytest.approx(0.2)


def test_synergy_no_alert_when_high():
    # 协同度 0.7 ≥ 0.3 → 不告警
    orch = _synergy_orch()
    alerts = orch._diagnose(_synergy_perception(0.7))
    assert not [a for a in alerts if a["type"] == "low_synergy"]


def test_synergy_no_alert_single_strategy():
    # 仅 1 个策略 < min_strategies 2 → 不评估，不告警
    orch = _synergy_orch()
    alerts = orch._diagnose(_synergy_perception(0.2, n_strategies=1))
    assert not [a for a in alerts if a["type"] == "low_synergy"]


def test_synergy_disabled():
    orch = _synergy_orch(enabled=False)
    alerts = orch._diagnose(_synergy_perception(0.2))
    assert not [a for a in alerts if a["type"] == "low_synergy"]


def test_synergy_actions_generate():
    # low_synergy → reallocate decrease 到 reduce_target
    orch = _synergy_orch()
    alerts = [{"type": "low_synergy", "synergy_score": 0.2}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.3},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._synergy_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_offensive_suppressed_by_synergy():
    # 协同度低于阈值 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "synergy_guard": {"enabled": True, "synergy_threshold": 0.3, "min_strategies": 2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "low_synergy", "synergy_score": 0.2},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 组合级协同度趋势外推（synergy_trend_guard）──────────────────

def _synergy_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"synergy_trend_guard": {
        "enabled": True, "window": 3, "min_strategies": 2, "reduce_target": 0.15,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["synergy_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_synergy_trend_detects_decline():
    # 协同度连续 3 周期下降（0.7→0.5→0.3）→ synergy_deteriorating
    orch = _synergy_trend_orch()
    orch._diagnose(_synergy_perception(0.7))
    orch._diagnose(_synergy_perception(0.5))
    alerts = orch._diagnose(_synergy_perception(0.3))
    det = [a for a in alerts if a["type"] == "synergy_deteriorating"]
    assert len(det) == 1
    assert det[0]["synergy_score"] == pytest.approx(0.3)


def test_synergy_trend_no_alert_when_stable():
    # 协同度平稳（非严格下降）→ 不告警
    orch = _synergy_trend_orch()
    orch._diagnose(_synergy_perception(0.5))
    orch._diagnose(_synergy_perception(0.5))
    alerts = orch._diagnose(_synergy_perception(0.5))
    assert not [a for a in alerts if a["type"] == "synergy_deteriorating"]


def test_synergy_trend_disabled():
    orch = _synergy_trend_orch(enabled=False)
    orch._diagnose(_synergy_perception(0.7))
    orch._diagnose(_synergy_perception(0.5))
    alerts = orch._diagnose(_synergy_perception(0.3))
    assert not [a for a in alerts if a["type"] == "synergy_deteriorating"]


def test_synergy_trend_actions_generate():
    # synergy_deteriorating → reallocate decrease 到 reduce_target
    orch = _synergy_trend_orch()
    alerts = [{"type": "synergy_deteriorating", "synergy_score": 0.3}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "sync": {"target_weight": 0.3},
        "grid": {"target_weight": 0.2},
    }}}
    actions = orch._synergy_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "sync"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_offensive_suppressed_by_synergy_trend():
    # 协同度连续下降 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "synergy_trend_guard": {"enabled": True, "window": 3, "min_strategies": 2},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["sync"], "boost_step": 0.05},
        {"type": "synergy_deteriorating", "synergy_score": 0.3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"sync": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 浮盈占比趋势外推（unrealized_profit_ratio_trend_guard）────────────────

def _unrealized_profit_ratio_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"unrealized_profit_ratio_trend_guard": {
        "enabled": True, "window": 3, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["unrealized_profit_ratio_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_unrealized_profit_ratio_trend_detects_rising():
    # 浮盈占比连续 3 周期上升（0.5→0.7→0.9）→ unrealized_profit_ratio_rising
    orch = _unrealized_profit_ratio_trend_orch()
    orch._strategy_unrealized_profit_ratio_history["grid"] = deque([0.5, 0.7], maxlen=3)
    alerts = orch._diagnose(_unrealized_profit_ratio_perception(
        {"grid": {"total_pnl": 100.0, "unrealized_pnl": 90.0}}
    ))
    hits = [a for a in alerts if a["type"] == "unrealized_profit_ratio_rising"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["unrealized_profit_ratio"] == pytest.approx(0.9)


def test_unrealized_profit_ratio_trend_no_alert_when_stable():
    # 浮盈占比平稳（0.5，非严格上升）→ 不告警
    orch = _unrealized_profit_ratio_trend_orch()
    orch._strategy_unrealized_profit_ratio_history["grid"] = deque([0.5, 0.5], maxlen=3)
    alerts = orch._diagnose(_unrealized_profit_ratio_perception(
        {"grid": {"total_pnl": 100.0, "unrealized_pnl": 50.0}}
    ))
    assert not [a for a in alerts if a["type"] == "unrealized_profit_ratio_rising"]


def test_unrealized_profit_ratio_trend_no_alert_low_sample():
    # 仅 2 个样本（< window 3）→ 不告警
    orch = _unrealized_profit_ratio_trend_orch()
    orch._strategy_unrealized_profit_ratio_history["grid"] = deque([0.5], maxlen=3)
    alerts = orch._diagnose(_unrealized_profit_ratio_perception(
        {"grid": {"total_pnl": 100.0, "unrealized_pnl": 90.0}}
    ))
    assert not [a for a in alerts if a["type"] == "unrealized_profit_ratio_rising"]


def test_unrealized_profit_ratio_trend_disabled():
    orch = _unrealized_profit_ratio_trend_orch(enabled=False)
    orch._strategy_unrealized_profit_ratio_history["grid"] = deque([0.5, 0.7], maxlen=3)
    alerts = orch._diagnose(_unrealized_profit_ratio_perception(
        {"grid": {"total_pnl": 100.0, "unrealized_pnl": 90.0}}
    ))
    assert not [a for a in alerts if a["type"] == "unrealized_profit_ratio_rising"]


def test_unrealized_profit_ratio_trend_actions_generate():
    # unrealized_profit_ratio_rising → reallocate decrease 到 reduce_target
    orch = _unrealized_profit_ratio_trend_orch()
    alerts = [{"type": "unrealized_profit_ratio_rising", "strategy": "grid", "unrealized_profit_ratio": 0.9}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.5}}}}
    actions = orch._unrealized_profit_ratio_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_offensive_suppressed_by_unrealized_profit_ratio_trend():
    # 浮盈占比持续上升 → 该策略不追高加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "unrealized_profit_ratio_trend_guard": {"enabled": True, "window": 3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "unrealized_profit_ratio_rising", "strategy": "grid", "unrealized_profit_ratio": 0.9},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 策略级手续费率趋势外推（strategy_fee_ratio_trend_guard）────────────────

def _strategy_fee_ratio_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"strategy_fee_ratio_trend_guard": {
        "enabled": True, "window": 3, "reduce_target": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["strategy_fee_ratio_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_strategy_fee_ratio_trend_detects_rising():
    # 手续费率连续 3 周期上升（0.1→0.3→0.5）→ strategy_fee_ratio_rising
    orch = _strategy_fee_ratio_trend_orch()
    orch._strategy_fee_ratio_history["grid"] = deque([0.1, 0.3], maxlen=3)
    alerts = orch._diagnose(_strategy_fee_ratio_perception(50.0, 50.0))
    hits = [a for a in alerts if a["type"] == "strategy_fee_ratio_rising"]
    assert len(hits) == 1
    assert hits[0]["strategy"] == "grid"
    assert hits[0]["fee_ratio"] == pytest.approx(0.5)


def test_strategy_fee_ratio_trend_no_alert_when_stable():
    # 手续费率平稳（0.2，非严格上升）→ 不告警
    orch = _strategy_fee_ratio_trend_orch()
    orch._strategy_fee_ratio_history["grid"] = deque([0.2, 0.2], maxlen=3)
    alerts = orch._diagnose(_strategy_fee_ratio_perception(80.0, 20.0))
    assert not [a for a in alerts if a["type"] == "strategy_fee_ratio_rising"]


def test_strategy_fee_ratio_trend_disabled():
    orch = _strategy_fee_ratio_trend_orch(enabled=False)
    orch._strategy_fee_ratio_history["grid"] = deque([0.1, 0.3], maxlen=3)
    alerts = orch._diagnose(_strategy_fee_ratio_perception(50.0, 50.0))
    assert not [a for a in alerts if a["type"] == "strategy_fee_ratio_rising"]


def test_strategy_fee_ratio_trend_actions_generate():
    # strategy_fee_ratio_rising → reallocate decrease 到 reduce_target
    orch = _strategy_fee_ratio_trend_orch()
    alerts = [{"type": "strategy_fee_ratio_rising", "strategy": "grid", "fee_ratio": 0.5}]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.5}}}}
    actions = orch._strategy_fee_ratio_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.1)


def test_offensive_suppressed_by_strategy_fee_ratio_trend():
    # 手续费率持续上升 → 该策略不追高加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "strategy_fee_ratio_trend_guard": {"enabled": True, "window": 3, "reduce_target": 0.1},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "strategy_fee_ratio_rising", "strategy": "grid", "fee_ratio": 0.5},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 组合级资金效率响应（portfolio_efficiency_guard）──────────────────

def _portfolio_efficiency_orch(**overrides):
    cfg = {"agi_orchestrator": {"portfolio_efficiency_guard": {
        "enabled": True, "reduce_target": 0.15,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["portfolio_efficiency_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def test_portfolio_efficiency_actions_generate():
    # low_capital_efficiency → reallocate decrease 到 reduce_target（最高权重策略）
    orch = _portfolio_efficiency_orch()
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.6},
        "trend": {"target_weight": 0.1},
    }}}
    alerts = [{"type": "low_capital_efficiency", "efficiency_score": 0.05}]
    actions = orch._portfolio_efficiency_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_portfolio_efficiency_actions_skip_low_weight():
    # 最高权重已 ≤ reduce_target → 无需收敛
    orch = _portfolio_efficiency_orch()
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.1},
        "trend": {"target_weight": 0.1},
    }}}
    alerts = [{"type": "low_capital_efficiency", "efficiency_score": 0.05}]
    assert orch._portfolio_efficiency_actions(decision, alerts) == []


def test_portfolio_efficiency_actions_disabled():
    orch = _portfolio_efficiency_orch(enabled=False)
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.6}}}}
    alerts = [{"type": "low_capital_efficiency", "efficiency_score": 0.05}]
    assert orch._portfolio_efficiency_actions(decision, alerts) == []


def test_portfolio_efficiency_ignores_non_efficiency_alerts():
    orch = _portfolio_efficiency_orch()
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.6}}}}
    alerts = [{"type": "high_concentration", "concentration": 0.7}]
    assert orch._portfolio_efficiency_actions(decision, alerts) == []


# ── 组合级资金效率趋势外推（portfolio_efficiency_trend_guard）────────────────

def _portfolio_efficiency_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"portfolio_efficiency_trend_guard": {
        "enabled": True, "window": 3, "min_sample": 10, "reduce_target": 0.15,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["portfolio_efficiency_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _portfolio_efficiency_trend_perception(efficiency_score, total_trades=20):
    strategies = {"s0": {"total_pnl": 10.0}, "s1": {"total_pnl": 5.0}}
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {
            "efficiency_score": efficiency_score,
            "strategies": strategies,
            "total_trades": total_trades,
        },
        "freeze_state": {},
    }


def test_portfolio_efficiency_trend_detects_decline():
    # 资金效率连续 3 周期下降（0.7→0.5→0.3）→ efficiency_deteriorating
    orch = _portfolio_efficiency_trend_orch()
    orch._diagnose(_portfolio_efficiency_trend_perception(0.7))
    orch._diagnose(_portfolio_efficiency_trend_perception(0.5))
    alerts = orch._diagnose(_portfolio_efficiency_trend_perception(0.3))
    det = [a for a in alerts if a["type"] == "efficiency_deteriorating"]
    assert len(det) == 1
    assert det[0]["efficiency_score"] == pytest.approx(0.3)


def test_portfolio_efficiency_trend_no_alert_when_stable():
    orch = _portfolio_efficiency_trend_orch()
    orch._diagnose(_portfolio_efficiency_trend_perception(0.5))
    orch._diagnose(_portfolio_efficiency_trend_perception(0.5))
    alerts = orch._diagnose(_portfolio_efficiency_trend_perception(0.5))
    assert not [a for a in alerts if a["type"] == "efficiency_deteriorating"]


def test_portfolio_efficiency_trend_no_alert_below_min_sample():
    # 成交数 < min_sample 10 → 不追踪，不告警
    orch = _portfolio_efficiency_trend_orch()
    orch._diagnose(_portfolio_efficiency_trend_perception(0.7, total_trades=5))
    orch._diagnose(_portfolio_efficiency_trend_perception(0.5, total_trades=5))
    alerts = orch._diagnose(_portfolio_efficiency_trend_perception(0.3, total_trades=5))
    assert not [a for a in alerts if a["type"] == "efficiency_deteriorating"]


def test_portfolio_efficiency_trend_disabled():
    orch = _portfolio_efficiency_trend_orch(enabled=False)
    orch._diagnose(_portfolio_efficiency_trend_perception(0.7))
    orch._diagnose(_portfolio_efficiency_trend_perception(0.5))
    alerts = orch._diagnose(_portfolio_efficiency_trend_perception(0.3))
    assert not [a for a in alerts if a["type"] == "efficiency_deteriorating"]


def test_portfolio_efficiency_trend_actions_generate():
    # efficiency_deteriorating → reallocate decrease 到 reduce_target
    orch = _portfolio_efficiency_trend_orch()
    alerts = [{"type": "efficiency_deteriorating", "efficiency_score": 0.3}]
    decision = {"allocation_plan": {"strategy_allocations": {
        "grid": {"target_weight": 0.6},
        "trend": {"target_weight": 0.1},
    }}}
    actions = orch._portfolio_efficiency_trend_actions(decision, alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "reallocate"
    assert actions[0]["action"] == "decrease"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["target_allocation"] == pytest.approx(0.15)


def test_offensive_suppressed_by_portfolio_efficiency_trend():
    # 资金效率连续下降 → 暂停进攻性加仓（账户级 gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "portfolio_efficiency_trend_guard": {"enabled": True, "window": 3, "min_sample": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "efficiency_deteriorating", "efficiency_score": 0.3},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


# ── 策略回撤持续时间趋势外推（drawdown_duration_trend_guard）────────────────

def _drawdown_duration_trend_orch(**overrides):
    cfg = {"agi_orchestrator": {"drawdown_duration_trend_guard": {
        "enabled": True, "window": 3, "min_trades": 10, "deterioration_leverage": 1.0,
    }}}
    state_path = overrides.pop("state_path", None)
    if state_path is not None:
        cfg["agi_orchestrator"]["state_path"] = str(state_path)
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["drawdown_duration_trend_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _drawdown_duration_trend_perception(duration_hours, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "max_drawdown_duration_hours": duration_hours,
        }}},
        "freeze_state": {},
    }


def test_drawdown_duration_trend_detects_decline():
    # 回撤持续时长连续拉长（上升）→ 触发告警
    orch = _drawdown_duration_trend_orch()
    orch._strategy_drawdown_duration_history["grid"] = deque([2.0, 4.0], maxlen=3)
    alerts = orch._diagnose(_drawdown_duration_trend_perception(duration_hours=6.0))
    det = [a for a in alerts if a["type"] == "drawdown_duration_deteriorating"]
    assert len(det) == 1
    assert det[0]["strategy"] == "grid"
    assert det[0]["max_drawdown_duration_hours"] == pytest.approx(6.0)


def test_drawdown_duration_trend_no_alert_when_falling():
    orch = _drawdown_duration_trend_orch()
    orch._strategy_drawdown_duration_history["grid"] = deque([6.0, 4.0], maxlen=3)
    alerts = orch._diagnose(_drawdown_duration_trend_perception(duration_hours=2.0))
    assert not [a for a in alerts if a["type"] == "drawdown_duration_deteriorating"]


def test_drawdown_duration_trend_insufficient_trades():
    orch = _drawdown_duration_trend_orch(min_trades=10)
    orch._strategy_drawdown_duration_history["grid"] = deque([2.0, 4.0], maxlen=3)
    alerts = orch._diagnose(_drawdown_duration_trend_perception(duration_hours=6.0, total_trades=5))
    assert not [a for a in alerts if a["type"] == "drawdown_duration_deteriorating"]


def test_drawdown_duration_trend_disabled():
    orch = _drawdown_duration_trend_orch(enabled=False)
    orch._strategy_drawdown_duration_history["grid"] = deque([2.0, 4.0], maxlen=3)
    alerts = orch._diagnose(_drawdown_duration_trend_perception(duration_hours=6.0))
    assert not [a for a in alerts if a["type"] == "drawdown_duration_deteriorating"]


def test_drawdown_duration_trend_actions_generate():
    orch = _drawdown_duration_trend_orch()
    alerts = [{"type": "drawdown_duration_deteriorating", "strategy": "grid", "message": "持续拉长"}]
    actions = orch._drawdown_duration_trend_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == pytest.approx(1.0)


def test_offensive_suppressed_by_drawdown_duration_trend():
    # 回撤持续时间拉长的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "drawdown_duration_trend_guard": {"enabled": True, "window": 3, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "drawdown_duration_deteriorating", "strategy": "grid", "max_drawdown_duration_hours": 6.0},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []


def test_drawdown_duration_trend_persist_restore(tmp_path):
    state_file = tmp_path / "test_state.json"
    orch = _drawdown_duration_trend_orch(state_path=state_file)
    orch._diagnose(_drawdown_duration_trend_perception(duration_hours=2.0))
    orch._diagnose(_drawdown_duration_trend_perception(duration_hours=4.0))
    orch.state_path = str(state_file)
    ls = orch._serialize_learning_state()
    import json as _json
    with open(state_file, "w") as f:
        _json.dump({"learning_state": ls}, f)
    orch2 = _drawdown_duration_trend_orch(state_path=state_file)
    orch2.state_path = str(state_file)
    orch2._load_learning_state()
    assert list(orch2._strategy_drawdown_duration_history["grid"]) == [
        pytest.approx(2.0), pytest.approx(4.0),
    ]


# ── 策略级止损频率收敛（stop_loss_frequency_guard）────────────────

def _stop_loss_frequency_orch(**overrides):
    cfg = {"agi_orchestrator": {"stop_loss_frequency_guard": {
        "enabled": True, "loss_ratio_threshold": 0.5, "min_trades": 10,
        "deterioration_leverage": 1.0,
    }}}
    for k, v in overrides.items():
        cfg["agi_orchestrator"]["stop_loss_frequency_guard"][k] = v
    return QuantAGIOrchestrator(config=cfg)


def _stop_loss_frequency_perception(stop_loss_count, total_trades=20):
    return {
        "equity": 1000.0,
        "used_margin": 100.0,
        "market_regime": {"regime": "range_bound", "strength": 0.4, "confidence": 0.6},
        "contribution": {"strategies": {"grid": {
            "health_score": 70.0,
            "health_grade": "B",
            "lifecycle": "mature",
            "trend": "stable",
            "total_trades": total_trades,
            "total_pnl": 50.0,
            "win_rate": 0.6,
            "stop_loss_count": stop_loss_count,
        }}},
        "freeze_state": {},
    }


def test_stop_loss_frequency_detects_threshold():
    # 止损率 10/20 = 0.5 ≥ 0.5 → 触发告警
    orch = _stop_loss_frequency_orch()
    alerts = orch._diagnose(_stop_loss_frequency_perception(stop_loss_count=10))
    det = [a for a in alerts if a["type"] == "high_stop_loss_rate"]
    assert len(det) == 1
    assert det[0]["strategy"] == "grid"
    assert det[0]["stop_loss_count"] == 10
    assert det[0]["stop_loss_ratio"] == pytest.approx(0.5)


def test_stop_loss_frequency_below_threshold():
    # 止损率 5/20 = 0.25 < 0.5 → 不告警
    orch = _stop_loss_frequency_orch()
    alerts = orch._diagnose(_stop_loss_frequency_perception(stop_loss_count=5))
    assert not [a for a in alerts if a["type"] == "high_stop_loss_rate"]


def test_stop_loss_frequency_insufficient_trades():
    # 交易笔数不足 min_trades 时不评估
    orch = _stop_loss_frequency_orch(min_trades=10)
    alerts = orch._diagnose(_stop_loss_frequency_perception(stop_loss_count=4, total_trades=5))
    assert not [a for a in alerts if a["type"] == "high_stop_loss_rate"]


def test_stop_loss_frequency_disabled():
    orch = _stop_loss_frequency_orch(enabled=False)
    alerts = orch._diagnose(_stop_loss_frequency_perception(stop_loss_count=10))
    assert not [a for a in alerts if a["type"] == "high_stop_loss_rate"]


def test_stop_loss_frequency_actions_generate():
    orch = _stop_loss_frequency_orch()
    alerts = [{"type": "high_stop_loss_rate", "strategy": "grid", "message": "止损率过高"}]
    actions = orch._stop_loss_frequency_actions(alerts)
    assert len(actions) == 1
    assert actions[0]["type"] == "param_adjust"
    assert actions[0]["strategy"] == "grid"
    assert actions[0]["param"] == "leverage"
    assert actions[0]["value"] == pytest.approx(1.0)


def test_offensive_suppressed_by_stop_loss_frequency():
    # 止损率过高的策略不进攻加仓（per-strategy gate）
    cfg = {"agi_orchestrator": {
        "offensive_allocation": {
            "enabled": True, "min_regime_strength": 0.6, "max_drawdown_pct": 0.05,
            "boost_step": 0.05, "max_target": 0.4, "min_interval_cycles": 5,
        },
        "stop_loss_frequency_guard": {"enabled": True, "loss_ratio_threshold": 0.5, "min_trades": 10},
    }}
    orch = QuantAGIOrchestrator(config=cfg)
    alerts = [
        {"type": "offensive_opportunity", "strategies": ["grid"], "boost_step": 0.05},
        {"type": "high_stop_loss_rate", "strategy": "grid", "stop_loss_ratio": 0.6},
    ]
    decision = {"allocation_plan": {"strategy_allocations": {"grid": {"target_weight": 0.2}}}}
    assert orch._offensive_allocation_actions(decision, alerts) == []
