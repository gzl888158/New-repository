"""
AGI 失效场景压力测试（resilience stress）
=========================================
针对具体失效场景，验证 QuantAGIOrchestrator 的 fail-closed 韧性：
不崩溃、不放大资金风险、输出 JSON 安全、正确收敛。

与 tests/perf/test_performance.py（吞吐/延迟基准）和
tests/perf/test_stress_local_client.py（高频回放/内存泄漏）区分：
本套聚焦「具体失效场景」下的正确性/韧性，而非吞吐性能。

覆盖失效场景：
  1. 账户权益归零/为负（清算/强平）
  2. 权益数据不可用（账户接口故障 → equity=None）
  3. 感知数值 NaN/Inf 污染（上游数据损坏）
  4. 畸形字段类型 + 缺失关键键（协议不兼容/脏数据）
  5. 尾部极端敞口（毛敞口 >> 权益，闪崩/杠杆飙升）
  6. 快速市场状态振荡（跨周期 thrash）
  7. 全策略同时衰退（组合级共振）
  8. 依赖模块抛异常（regime_engine / contribution_analyzer 故障）
"""
import asyncio
import json

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _orch(**agi_cfg) -> QuantAGIOrchestrator:
    """构造一个最小化配置的 orchestrator（各失效场景按需开启对应守卫）。"""
    return QuantAGIOrchestrator(config={"agi_orchestrator": dict(agi_cfg)})


def _perception(**overrides) -> dict:
    """基线感知快照：健康账户 + 温和敞口，各场景用 overrides 注入失效条件。"""
    p = {
        "equity": 10000.0,
        "total_capital": 10000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 1000.0,
        "market_regime": {"regime": "trend_bullish", "strength": 0.7, "confidence": 0.8},
        "contribution": {
            "strategies": {
                "grid": {
                    "health_grade": "A",
                    "health_score": 85.0,
                    "total_pnl": 100.0,
                    "total_trades": 20,
                    "trend": "improving",
                },
            },
            "total_pnl": 100.0,
            "overall_health_score": 85.0,
        },
        "freeze_state": {},
        "equity_status": {"mode": "normal", "max_drawdown_pct": 0.0},
        "net_exposure": {"long": 1000.0, "short": 1000.0},
        "correlation": {},
        "symbol_pnl": {},
        "spot_holdings": {"currencies": [], "count": 0},
    }
    p.update(overrides)
    return p


class _FakeCapitalAllocator:
    """最小 capital_allocator，仅返回给定权益/总资金（可注入 None/NaN/负数）。"""

    def __init__(self, equity, total_capital=None):
        self._equity = equity
        self._total_capital = total_capital if total_capital is not None else equity

    def get_equity(self):
        return self._equity

    def get_total_capital(self):
        return self._total_capital


# ─────────────────────────────────────────────────────────────
# 1. 账户权益归零 / 为负（清算 / 强平）
# ─────────────────────────────────────────────────────────────
class TestNonPositiveEquity:
    def test_zero_equity_diagnosed_critical(self):
        orch = _orch()
        alerts = orch._diagnose(_perception(equity=0.0))
        assert any(a["type"] == "non_positive_equity" for a in alerts)

    def test_negative_equity_diagnosed_critical(self):
        orch = _orch()
        alerts = orch._diagnose(_perception(equity=-123.45))
        assert any(a["type"] == "non_positive_equity" for a in alerts)

    def test_zero_equity_run_cycle_fail_closed_no_actions(self):
        orch = QuantAGIOrchestrator(
            config={"agi_orchestrator": {}},
            capital_allocator=_FakeCapitalAllocator(0.0),
        )
        report = asyncio.run(orch.run_cycle())
        assert report["status"] == "fail_closed"
        assert report["actions"] == []

    def test_negative_equity_run_cycle_fail_closed_no_actions(self):
        orch = QuantAGIOrchestrator(
            config={"agi_orchestrator": {}},
            capital_allocator=_FakeCapitalAllocator(-50.0),
        )
        report = asyncio.run(orch.run_cycle())
        assert report["status"] == "fail_closed"
        assert report["actions"] == []


# ─────────────────────────────────────────────────────────────
# 2. 权益数据不可用（账户接口故障）
# ─────────────────────────────────────────────────────────────
class TestEquityUnavailable:
    def test_no_capital_allocator_fail_closed(self):
        # 不注入 capital_allocator → _perceive 权益为 None → fail-closed
        orch = QuantAGIOrchestrator(config={"agi_orchestrator": {}})
        report = asyncio.run(orch.run_cycle())
        assert report["status"] == "fail_closed"
        assert report["actions"] == []

    def test_equity_none_fail_closed(self):
        orch = QuantAGIOrchestrator(
            config={"agi_orchestrator": {}},
            capital_allocator=_FakeCapitalAllocator(None),
        )
        report = asyncio.run(orch.run_cycle())
        assert report["status"] == "fail_closed"
        assert report["actions"] == []


# ─────────────────────────────────────────────────────────────
# 3. 感知数值 NaN / Inf 污染
# ─────────────────────────────────────────────────────────────
class TestNaNInfPollution:
    def test_diagnose_no_crash_on_nan_inf(self):
        orch = _orch(portfolio_stress_guard={"enabled": True})
        p = _perception(
            equity=float("nan"),
            unrealized_pnl=float("inf"),
            net_exposure={"long": float("inf"), "short": float("-inf")},
        )
        alerts = orch._diagnose(p)  # 不得抛异常
        assert isinstance(alerts, list)

    def test_sanitize_removes_nan_inf_json_safe(self):
        orch = _orch()
        dirty = {
            "a": float("nan"),
            "b": float("inf"),
            "c": [float("-inf"), {"d": float("nan")}],
            "e": "ok",
        }
        clean = orch._sanitize(dirty)
        json.dumps(clean)  # 不得抛 ValueError（JSON 序列化安全）
        assert clean["a"] == 0.0
        assert clean["b"] == 0.0
        assert clean["c"][0] == 0.0
        assert clean["c"][1]["d"] == 0.0
        assert clean["e"] == "ok"

    def test_nan_equity_run_cycle_fail_closed(self):
        orch = QuantAGIOrchestrator(
            config={"agi_orchestrator": {}},
            capital_allocator=_FakeCapitalAllocator(float("nan")),
        )
        report = asyncio.run(orch.run_cycle())  # safe_finite(NaN)→0.0 → fail_closed
        assert report["status"] == "fail_closed"
        assert report["actions"] == []
        json.dumps(report)  # 整份报告 JSON 安全


# ─────────────────────────────────────────────────────────────
# 4. 畸形字段类型 + 缺失关键键（脏数据 / 协议不兼容）
# ─────────────────────────────────────────────────────────────
class TestMalformedInput:
    def test_sparse_perception_no_crash(self):
        orch = _orch()
        alerts = orch._diagnose({"equity": 1000.0})  # 其余键全部缺失
        assert isinstance(alerts, list)

    def test_string_numeric_pollution_no_crash(self):
        orch = _orch(portfolio_stress_guard={"enabled": True})
        p = _perception(
            equity="not_a_number",
            net_exposure={"long": "huge", "short": "also_huge"},
            used_margin="unknown",
        )
        alerts = orch._diagnose(p)  # safe_float 回退默认值，不得抛异常
        assert isinstance(alerts, list)

    def test_malformed_contribution_no_crash(self):
        orch = _orch()
        p = _perception(contribution={"strategies": "not_a_dict", "total_pnl": "abc"})
        alerts = orch._diagnose(p)
        assert isinstance(alerts, list)

    def test_contribution_as_string_no_crash(self):
        orch = _orch()
        p = _perception(contribution="garbage_not_a_dict")
        alerts = orch._diagnose(p)  # _coerce_dict 回退空 dict，不得抛异常
        assert isinstance(alerts, list)

    def test_net_exposure_as_string_no_crash(self):
        orch = _orch(portfolio_stress_guard={"enabled": True})
        p = _perception(net_exposure="garbage_not_a_dict")
        alerts = orch._diagnose(p)  # _coerce_dict 回退空 dict，不得抛异常
        assert isinstance(alerts, list)


# ─────────────────────────────────────────────────────────────
# 5. 尾部极端敞口（毛敞口 >> 权益，闪崩 / 杠杆飙升）
# ─────────────────────────────────────────────────────────────
class TestExtremeTailExposure:
    def _stress_orch(self):
        return _orch(portfolio_stress_guard={
            "enabled": True,
            "stress_scenario_pct": 0.2,
            "stress_loss_budget": 0.5,
            "severe_scenario_pct": 0.4,
            "severe_loss_budget": 0.8,
            "reduce_target": 0.15,
        })

    def test_extreme_exposure_triggers_both_stress_alerts(self):
        orch = self._stress_orch()
        # 毛敞口 200000 vs 权益 100 → 压力损失远超预算
        p = _perception(equity=100.0, net_exposure={"long": 100000.0, "short": 100000.0})
        alerts = orch._diagnose(p)
        assert any(a["type"] == "stress_test_failed" for a in alerts)
        assert any(a["type"] == "severe_stress_test_failed" for a in alerts)

    def test_extreme_exposure_zero_equity_no_zero_division(self):
        orch = self._stress_orch()
        # 权益为 0 时压力测试分支被跳过（equity>0 才评估），不得 ZeroDivisionError
        p = _perception(equity=0.0, net_exposure={"long": 1e9, "short": 1e9})
        alerts = orch._diagnose(p)
        assert isinstance(alerts, list)

    def test_huge_exposure_huge_equity_no_overflow_crash(self):
        orch = self._stress_orch()
        p = _perception(equity=1e18, net_exposure={"long": 1e18, "short": 1e18})
        alerts = orch._diagnose(p)
        assert isinstance(alerts, list)


# ─────────────────────────────────────────────────────────────
# 6. 快速市场状态振荡（跨周期 thrash）
# ─────────────────────────────────────────────────────────────
class TestRegimeOscillation:
    def _offensive_orch(self):
        return _orch(offensive_allocation={
            "enabled": True,
            "min_regime_strength": 0.6,
            "min_regime_confidence": 0.5,
            "trend_confirmation_cycles": 2,
            "max_drawdown_pct": 0.05,
            "boost_step": 0.05,
            "max_target": 0.4,
        })

    def test_alternating_regime_no_crash_and_no_offense(self):
        orch = self._offensive_orch()
        # 状态每周期翻转 → 趋势永不确认（confirmed_streak 恒 1 < 2），不得进攻
        for i in range(60):
            regime = "trend_bullish" if i % 2 == 0 else "trend_bearish"
            p = _perception(market_regime={"regime": regime, "strength": 0.7, "confidence": 0.8})
            alerts = orch._diagnose(p)
            assert isinstance(alerts, list)
            assert not [a for a in alerts if a["type"] == "offensive_opportunity"]


# ─────────────────────────────────────────────────────────────
# 7. 全策略同时衰退（组合级共振）
# ─────────────────────────────────────────────────────────────
class TestAllStrategiesFailing:
    def test_all_strategies_fail_diagnosed_no_crash(self):
        orch = _orch(risk_response={"enabled": True, "reduce_target": 0.0})
        p = _perception(contribution={
            "strategies": {
                "grid": {"health_grade": "F", "health_score": 10.0, "total_pnl": -500.0,
                         "total_trades": 20, "trend": "declining"},
                "trend": {"health_grade": "F", "health_score": 8.0, "total_pnl": -800.0,
                          "total_trades": 20, "trend": "declining"},
            },
            "total_pnl": -1300.0,
            "overall_health_score": 9.0,
        })
        alerts = orch._diagnose(p)
        assert isinstance(alerts, list)
        # 每个 F 策略都生成 critical 健康告警
        critical = [a for a in alerts if a["type"] == "strategy_health_critical"]
        assert len(critical) == 2


# ─────────────────────────────────────────────────────────────
# 8. 依赖模块抛异常（regime_engine / contribution_analyzer 故障）
# ─────────────────────────────────────────────────────────────
class TestDependencyException:
    def test_perceive_isolates_dependency_exceptions(self):
        class BoomRegime:
            def get_regime(self):
                raise RuntimeError("regime engine down")

        class BoomAnalyzer:
            def analyze(self, **kwargs):
                raise RuntimeError("contribution analyzer down")

        orch = QuantAGIOrchestrator(
            config={"agi_orchestrator": {}},
            regime_engine=BoomRegime(),
            contribution_analyzer=BoomAnalyzer(),
        )
        perception = asyncio.run(orch._perceive())  # 异常被隔离，不得抛出
        assert perception["market_regime"] is None
        assert perception["contribution"] is None

    def test_run_cycle_with_raising_deps_no_crash(self):
        class BoomRegime:
            def get_regime(self):
                raise RuntimeError("regime engine down")

        class BoomAnalyzer:
            def analyze(self, **kwargs):
                raise RuntimeError("contribution analyzer down")

        orch = QuantAGIOrchestrator(
            config={"agi_orchestrator": {}},
            regime_engine=BoomRegime(),
            contribution_analyzer=BoomAnalyzer(),
            capital_allocator=_FakeCapitalAllocator(1000.0),
        )
        report = asyncio.run(orch.run_cycle())  # 不得崩溃，权益有效 → ok 或 degraded
        assert report["status"] in ("ok", "degraded", "fail_closed")
        json.dumps(report)  # JSON 安全
