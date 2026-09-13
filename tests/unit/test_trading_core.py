"""
生产级交易核心集成测试
======================
覆盖：TradingSessionManager + RealTimeRiskMonitor
"""

import os
import sys
import json
import time
import asyncio
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.trading_session import (
    TradingSessionManager, SessionState, SessionEvent,
    StrategySchedule, PreMarketCheckResult, SessionMetrics,
)
from core.risk_monitor import (
    RealTimeRiskMonitor, RiskAlertLevel, RiskDimension,
    MitigationAction, RiskScore, RiskSnapshot, RiskEvent,
)


# ═══════════════════════════════════════════════════════════════
# 测试辅助
# ═══════════════════════════════════════════════════════════════

def load_config():
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MockAccountManager:
    def __init__(self):
        self._account_info = {
            "totalEq": "1000.0",
            "availBal": "800.0",
            "totalMgn": "200.0",
            "mgnRatio": "0.2",
            "upl": "50.0",
        }

    def get_account_info(self):
        return self._account_info


class MockPositionManager:
    def __init__(self):
        self._positions = {}
        self._risk = Mock(drawdown_pct=0.05)

    def get_all_positions(self):
        return list(self._positions.values())

    def get_position_count(self):
        return len(self._positions)

    def get_position(self, symbol, side):
        key = f"{symbol}:{side}"
        return self._positions.get(key)

    def get_account_risk(self):
        return self._risk

    def add_position(self, pos):
        key = f"{pos.symbol}:{pos.side.value}"
        self._positions[key] = pos

    def clear(self):
        self._positions.clear()


class MockRiskGate:
    def get_status(self):
        return {"blocked": False, "total_checks": 0}


class MockCircuitBreaker:
    def get_status(self):
        return {"tripped": False, "reason": ""}


class MockOKXClient:
    def get_ticker(self, symbol):
        return {"last": "100.0"}

    def get_atr(self, symbol):
        return 0.5


class MockStrategy:
    def __init__(self, name):
        self.name = name
        self.started = False
        self.paused = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


# ═══════════════════════════════════════════════════════════════
# TradingSessionManager 测试
# ═══════════════════════════════════════════════════════════════

class TestSessionStateMachine(unittest.TestCase):
    """测试1: 会话状态机"""

    def setUp(self):
        self.config = load_config()
        self.mgr = TradingSessionManager(self.config)

    def test_01_initial_state(self):
        """初始状态"""
        self.assertEqual(self.mgr.state, SessionState.IDLE)
        print(f"  [PASS] test_01: initial state = {self.mgr.state.value}")

    def test_02_valid_transitions(self):
        """有效状态转换"""
        self.assertTrue(self.mgr.can_transition(SessionEvent.INIT_COMPLETE))
        self.assertTrue(self.mgr.can_transition(SessionEvent.MANUAL_START))
        print(f"  [PASS] test_02: valid transitions from IDLE")

    def test_03_invalid_transition(self):
        """无效状态转换"""
        self.assertFalse(self.mgr.can_transition(SessionEvent.PAUSE_REQUESTED))
        self.assertFalse(self.mgr.can_transition(SessionEvent.CLOSE_COMPLETE))
        print(f"  [PASS] test_03: invalid transitions rejected")

    def test_04_state_transition_flow(self):
        """完整状态流转"""
        loop = asyncio.new_event_loop()

        # IDLE -> PRE_MARKET
        result = loop.run_until_complete(self.mgr._transition(SessionEvent.INIT_COMPLETE))
        self.assertTrue(result)
        self.assertEqual(self.mgr.state, SessionState.PRE_MARKET)

        # PRE_MARKET -> STARTING
        result = loop.run_until_complete(self.mgr._transition(SessionEvent.PRE_CHECKS_PASSED))
        self.assertTrue(result)
        self.assertEqual(self.mgr.state, SessionState.STARTING)

        # STARTING -> TRADING
        result = loop.run_until_complete(self.mgr._transition(SessionEvent.START_COMPLETE))
        self.assertTrue(result)
        self.assertEqual(self.mgr.state, SessionState.TRADING)

        # TRADING -> PAUSED
        result = loop.run_until_complete(self.mgr._transition(SessionEvent.PAUSE_REQUESTED))
        self.assertTrue(result)
        self.assertEqual(self.mgr.state, SessionState.PAUSED)

        # PAUSED -> RESUMING
        result = loop.run_until_complete(self.mgr._transition(SessionEvent.RESUME_REQUESTED))
        self.assertTrue(result)
        self.assertEqual(self.mgr.state, SessionState.RESUMING)

        loop.close()
        print(f"  [PASS] test_04: full state flow: IDLE->PRE_MARKET->STARTING->TRADING->PAUSED->RESUMING")

    def test_05_emergency_transition(self):
        """紧急状态转换"""
        loop = asyncio.new_event_loop()

        # IDLE -> PRE_MARKET -> STARTING -> TRADING
        loop.run_until_complete(self.mgr._transition(SessionEvent.INIT_COMPLETE))
        loop.run_until_complete(self.mgr._transition(SessionEvent.PRE_CHECKS_PASSED))
        loop.run_until_complete(self.mgr._transition(SessionEvent.START_COMPLETE))

        # TRADING -> EMERGENCY
        result = loop.run_until_complete(self.mgr._transition(SessionEvent.EMERGENCY_TRIGGERED))
        self.assertTrue(result)
        self.assertEqual(self.mgr.state, SessionState.EMERGENCY)

        # EMERGENCY -> PAUSED
        result = loop.run_until_complete(self.mgr._transition(SessionEvent.EMERGENCY_RESOLVED))
        self.assertTrue(result)
        self.assertEqual(self.mgr.state, SessionState.PAUSED)

        loop.close()
        print(f"  [PASS] test_05: emergency transition: TRADING->EMERGENCY->PAUSED")


class TestSessionStrategyManagement(unittest.TestCase):
    """测试2: 策略管理"""

    def setUp(self):
        self.config = load_config()
        self.mgr = TradingSessionManager(self.config)

    def test_06_register_strategy(self):
        """注册策略"""
        strategy = MockStrategy("grid")
        schedule = StrategySchedule(name="grid", priority=1)
        self.mgr.register_strategy("grid", strategy, schedule)

        status = self.mgr.get_strategy_status("grid")
        self.assertFalse(status["active"])
        self.assertEqual(status["error_count"], 0)
        print(f"  [PASS] test_06: strategy registered: grid (priority=1)")

    def test_07_multiple_strategies(self):
        """注册多个策略"""
        strategies = {
            "trend": StrategySchedule(name="trend", priority=0),
            "grid": StrategySchedule(name="grid", priority=1),
            "scalping": StrategySchedule(name="scalping", priority=2),
        }
        for name, schedule in strategies.items():
            self.mgr.register_strategy(name, MockStrategy(name), schedule)

        self.assertEqual(len(self.mgr.get_strategy_status()), 3)
        self.assertEqual(len(self.mgr.get_active_strategies()), 0)
        print(f"  [PASS] test_07: 3 strategies registered, 0 active")

    def test_08_unregister_strategy(self):
        """注销策略"""
        self.mgr.register_strategy("test", MockStrategy("test"))
        self.mgr.unregister_strategy("test")
        self.assertEqual(self.mgr.get_strategy_status("test"), {})
        print(f"  [PASS] test_08: strategy unregistered")

    def test_09_start_strategies_by_priority(self):
        """按优先级启动策略"""
        trend = MockStrategy("trend")
        grid = MockStrategy("grid")
        self.mgr.register_strategy("trend", trend, StrategySchedule(name="trend", priority=0))
        self.mgr.register_strategy("grid", grid, StrategySchedule(name="grid", priority=1))

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.mgr._start_strategies())
        loop.close()

        self.assertTrue(trend.started)
        self.assertTrue(grid.started)
        status = self.mgr.get_strategy_status()
        self.assertTrue(status["trend"]["active"])
        self.assertTrue(status["grid"]["active"])
        print(f"  [PASS] test_09: strategies started by priority")

    def test_10_stop_strategies(self):
        """停止策略"""
        trend = MockStrategy("trend")
        self.mgr.register_strategy("trend", trend)
        self.mgr._strategy_status["trend"]["active"] = True

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.mgr._stop_strategies())
        loop.close()

        self.assertTrue(trend.stopped)
        self.assertFalse(self.mgr._strategy_status["trend"]["active"])
        print(f"  [PASS] test_10: strategies stopped")


class TestSessionPreMarketChecks(unittest.TestCase):
    """测试3: 盘前检查"""

    def setUp(self):
        self.config = load_config()
        self.mgr = TradingSessionManager(self.config)
        self.mgr.inject_dependencies(
            okx_client=MockOKXClient(),
            account_manager=MockAccountManager(),
            risk_gate=MockRiskGate(),
            circuit_breaker=MockCircuitBreaker(),
        )

    def test_11_pre_market_checks_pass(self):
        """盘前检查通过"""
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.mgr._run_pre_market_checks())
        loop.close()

        self.assertTrue(result.passed)
        self.assertTrue(result.checks.get("connectivity", False))
        self.assertTrue(result.checks.get("balance", False))
        print(f"  [PASS] test_11: pre-market checks passed: {result.checks}")

    def test_12_pre_market_checks_with_warnings(self):
        """盘前检查告警"""
        self.mgr.inject_dependencies(
            okx_client=MockOKXClient(),
            account_manager=None,
            risk_gate=None,
            circuit_breaker=None,
        )
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.mgr._run_pre_market_checks())
        loop.close()

        self.assertTrue(result.passed)
        self.assertTrue(len(result.warnings) > 0)
        print(f"  [PASS] test_12: pre-market checks with warnings: {len(result.warnings)}")

    def test_13_session_start_skip_checks(self):
        """跳过盘前检查启动"""
        self.mgr.register_strategy("test", MockStrategy("test"))
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.mgr.start_session(skip_checks=True))
        loop.close()

        self.assertTrue(result)
        self.assertEqual(self.mgr.state, SessionState.TRADING)
        print(f"  [PASS] test_13: session started with skip_checks=True")


class TestSessionMetrics(unittest.TestCase):
    """测试4: 会话指标"""

    def setUp(self):
        self.config = load_config()
        self.mgr = TradingSessionManager(self.config)

    def test_14_metrics_recording(self):
        """指标记录"""
        self.mgr.record_signal(5)
        self.mgr.record_order(3)
        self.mgr.record_fill(2, pnl=10.0, fee=0.5)

        metrics = self.mgr.get_metrics()
        self.assertEqual(metrics["total_signals"], 5)
        self.assertEqual(metrics["total_orders"], 3)
        self.assertEqual(metrics["total_fills"], 2)
        self.assertEqual(metrics["total_pnl"], 10.0)
        self.assertEqual(metrics["total_fees"], 0.5)
        print(f"  [PASS] test_14: metrics: signals={metrics['total_signals']}, "
              f"orders={metrics['total_orders']}, fills={metrics['total_fills']}")

    def test_15_session_id(self):
        """会话ID"""
        sid = self.mgr.get_session_id()
        self.assertTrue(sid.startswith("session_"))
        self.assertIn(datetime.now().strftime("%Y%m%d"), sid)
        print(f"  [PASS] test_15: session_id={sid}")

    def test_16_event_log(self):
        """事件日志"""
        self.mgr._log_event({"type": "test", "data": "hello"})
        self.mgr._log_event({"type": "test", "data": "world"})
        events = self.mgr.get_event_log()
        self.assertEqual(len(events), 2)
        print(f"  [PASS] test_16: event log: {len(events)} events")


# ═══════════════════════════════════════════════════════════════
# RealTimeRiskMonitor 测试
# ═══════════════════════════════════════════════════════════════

class TestRiskMonitorScoring(unittest.TestCase):
    """测试5: 风险评分"""

    def setUp(self):
        self.config = load_config()
        self.monitor = RealTimeRiskMonitor(self.config)

    def test_17_value_to_level(self):
        """值映射到告警级别"""
        self.assertEqual(
            self.monitor._value_to_level(0.09, RiskDimension.DRAWDOWN),
            RiskAlertLevel.INFO
        )
        self.assertEqual(
            self.monitor._value_to_level(0.15, RiskDimension.DRAWDOWN),
            RiskAlertLevel.WARNING
        )
        self.assertEqual(
            self.monitor._value_to_level(0.25, RiskDimension.DRAWDOWN),
            RiskAlertLevel.CRITICAL
        )
        self.assertEqual(
            self.monitor._value_to_level(0.35, RiskDimension.DRAWDOWN),
            RiskAlertLevel.EMERGENCY
        )
        print(f"  [PASS] test_17: value_to_level works correctly")

    def test_18_overall_score_calculation(self):
        """综合评分计算"""
        scores = {
            RiskDimension.DIRECTION: RiskScore(
                dimension=RiskDimension.DIRECTION, score=0.3,
                level=RiskAlertLevel.WARNING, value=0.3, threshold=0.6,
                timestamp=time.time(),
            ),
            RiskDimension.LEVERAGE: RiskScore(
                dimension=RiskDimension.LEVERAGE, score=0.2,
                level=RiskAlertLevel.INFO, value=0.2, threshold=0.7,
                timestamp=time.time(),
            ),
            RiskDimension.DRAWDOWN: RiskScore(
                dimension=RiskDimension.DRAWDOWN, score=0.05,
                level=RiskAlertLevel.INFO, value=0.05, threshold=0.20,
                timestamp=time.time(),
            ),
        }

        overall = self.monitor._calculate_overall_score(scores)
        self.assertGreater(overall, 0.0)
        self.assertLess(overall, 1.0)
        print(f"  [PASS] test_18: overall score = {overall:.3f}")

    def test_19_empty_scores(self):
        """空评分"""
        overall = self.monitor._calculate_overall_score({})
        self.assertEqual(overall, 0.0)
        print(f"  [PASS] test_19: empty scores = 0.0")


class TestRiskMonitorChecks(unittest.TestCase):
    """测试6: 风险检查"""

    def setUp(self):
        self.config = load_config()
        self.monitor = RealTimeRiskMonitor(self.config)
        self.mock_pm = MockPositionManager()
        self.mock_account = MockAccountManager()
        self.monitor.inject_dependencies(
            okx_client=MockOKXClient(),
            position_manager=self.mock_pm,
            account_manager=self.mock_account,
        )

    def test_20_direction_risk_no_positions(self):
        """无持仓方向风险"""
        score = self.monitor._check_direction_risk()
        self.assertEqual(score.level, RiskAlertLevel.INFO)
        self.assertEqual(score.score, 0.0)
        print(f"  [PASS] test_20: direction risk (no positions) = {score.score:.2f}")

    def test_21_concentration_risk_no_positions(self):
        """无持仓集中度风险"""
        score = self.monitor._check_concentration_risk()
        self.assertEqual(score.level, RiskAlertLevel.INFO)
        print(f"  [PASS] test_21: concentration risk (no positions) = {score.score:.2f}")

    def test_22_drawdown_risk(self):
        """回撤风险"""
        score = self.monitor._check_drawdown_risk()
        self.assertGreaterEqual(score.score, 0.0)
        print(f"  [PASS] test_22: drawdown risk = {score.score:.2f} [{score.level.value}]")

    def test_23_margin_risk(self):
        """保证金风险"""
        score = self.monitor._check_margin_risk()
        self.assertGreaterEqual(score.score, 0.0)
        print(f"  [PASS] test_23: margin risk = {score.score:.2f} [{score.level.value}]")

    def test_24_correlation_risk(self):
        """相关性风险"""
        score = self.monitor._check_correlation_risk()
        self.assertGreaterEqual(score.score, 0.0)
        print(f"  [PASS] test_24: correlation risk = {score.score:.2f} [{score.level.value}]")


class TestRiskMonitorAlerts(unittest.TestCase):
    """测试7: 告警管理"""

    def setUp(self):
        self.config = load_config()
        self.monitor = RealTimeRiskMonitor(self.config)
        self.alerts = []

    def test_25_alert_creation(self):
        """告警创建"""
        score = RiskScore(
            dimension=RiskDimension.DRAWDOWN,
            score=0.25,
            level=RiskAlertLevel.CRITICAL,
            value=0.25,
            threshold=0.20,
            timestamp=time.time(),
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.monitor._handle_risk_alert(RiskDimension.DRAWDOWN, score))
        loop.close()

        active = self.monitor.get_active_alerts()
        self.assertGreaterEqual(len(active), 1)
        self.assertEqual(active[0]["level"], "critical")
        print(f"  [PASS] test_25: alert created: {active[0]['type']} [{active[0]['level']}]")

    def test_26_clear_alert(self):
        """清除告警"""
        score = RiskScore(
            dimension=RiskDimension.MARGIN,
            score=0.55,
            level=RiskAlertLevel.WARNING,
            value=0.55,
            threshold=0.70,
            timestamp=time.time(),
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.monitor._handle_risk_alert(RiskDimension.MARGIN, score))
        loop.close()

        self.assertTrue(self.monitor.clear_alert("margin"))
        self.assertEqual(len(self.monitor.get_active_alerts()), 0)
        print(f"  [PASS] test_26: alert cleared")

    def test_27_alert_callback(self):
        """告警回调"""
        received = []

        def callback(event):
            received.append(event)

        self.monitor.on_risk_alert(callback)

        score = RiskScore(
            dimension=RiskDimension.LEVERAGE,
            score=0.6,
            level=RiskAlertLevel.WARNING,
            value=0.6,
            threshold=0.70,
            timestamp=time.time(),
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.monitor._handle_risk_alert(RiskDimension.LEVERAGE, score))
        loop.close()

        self.assertEqual(len(received), 1)
        self.assertEqual(received[0].dimension, RiskDimension.LEVERAGE)
        print(f"  [PASS] test_27: alert callback received: {received[0].event_type}")

    def test_28_alert_history(self):
        """告警历史"""
        score = RiskScore(
            dimension=RiskDimension.CONCENTRATION,
            score=0.4,
            level=RiskAlertLevel.WARNING,
            value=0.4,
            threshold=0.50,
            timestamp=time.time(),
        )

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.monitor._handle_risk_alert(RiskDimension.CONCENTRATION, score))
        loop.close()

        history = self.monitor.get_alert_history()
        self.assertGreaterEqual(len(history), 1)
        print(f"  [PASS] test_28: alert history: {len(history)} events")


# ═══════════════════════════════════════════════════════════════
# 配置加载测试
# ═══════════════════════════════════════════════════════════════

class TestConfigLoading(unittest.TestCase):
    """测试8: 配置加载"""

    def test_29_session_config(self):
        """会话配置"""
        config = load_config()
        session_cfg = config.get("trading_session", {})
        self.assertTrue(session_cfg.get("enabled", False))
        self.assertIn("strategy_schedules", session_cfg)
        schedules = session_cfg["strategy_schedules"]
        self.assertIn("trend", schedules)
        self.assertIn("grid", schedules)
        self.assertEqual(schedules["trend"]["priority"], 0)
        print(f"  [PASS] test_29: session config loaded: {len(schedules)} strategy schedules")

    def test_30_risk_monitor_config(self):
        """风险监控配置"""
        config = load_config()
        rm_cfg = config.get("risk_monitor", {})
        self.assertTrue(rm_cfg.get("enabled", False))
        self.assertGreater(rm_cfg.get("check_interval_sec", 0), 0)
        self.assertIn("scoring_weights", rm_cfg)
        self.assertIn("thresholds", rm_cfg)
        self.assertIn("auto_mitigation", rm_cfg)
        print(f"  [PASS] test_30: risk monitor config loaded: "
              f"{len(rm_cfg['scoring_weights'])} scoring weights, "
              f"{len(rm_cfg['thresholds'])} threshold dimensions")

    def test_31_full_config_integrity(self):
        """完整配置完整性"""
        config = load_config()
        required = [
            "trading_session", "risk_monitor", "execution",
            "position_manager", "websocket", "strategies",
            "trading", "risk", "dashboard", "capital_pool",
        ]
        for section in required:
            self.assertIn(section, config, f"Missing config section: {section}")
        print(f"  [PASS] test_31: all {len(required)} required config sections present")


# ═══════════════════════════════════════════════════════════════
# 集成测试
# ═══════════════════════════════════════════════════════════════

class TestIntegration(unittest.TestCase):
    """测试9: 集成测试"""

    def setUp(self):
        self.config = load_config()
        self.session_mgr = TradingSessionManager(self.config)
        self.risk_monitor = RealTimeRiskMonitor(self.config)

    def test_32_session_risk_integration(self):
        """会话与风险集成"""
        # 注入依赖
        mock_pm = MockPositionManager()
        self.risk_monitor.inject_dependencies(
            okx_client=MockOKXClient(),
            position_manager=mock_pm,
            account_manager=MockAccountManager(),
            session_manager=self.session_mgr,
        )
        self.session_mgr.inject_dependencies(
            okx_client=MockOKXClient(),
            account_manager=MockAccountManager(),
            position_manager=mock_pm,
        )

        # 注册策略
        self.session_mgr.register_strategy("grid", MockStrategy("grid"))
        self.session_mgr.register_strategy("trend", MockStrategy("trend"))

        # 启动会话（跳过检查）
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(self.session_mgr.start_session(skip_checks=True))
        loop.close()

        self.assertTrue(result)
        self.assertEqual(self.session_mgr.state, SessionState.TRADING)
        self.assertEqual(len(self.session_mgr.get_active_strategies()), 2)

        # 获取风险快照
        snapshot = self.risk_monitor.get_risk_snapshot()
        self.assertIn("overall_score", snapshot)
        self.assertIn("dimensions", snapshot)
        self.assertIn("position_count", snapshot)

        print(f"  [PASS] test_32: session+risk integration: "
              f"state={self.session_mgr.get_state()}, "
              f"risk_score={snapshot['overall_score']:.3f}")

    def test_33_mitigation_actions(self):
        """缓解动作枚举"""
        self.assertEqual(MitigationAction.NOTIFY.value, "notify")
        self.assertEqual(MitigationAction.REDUCE_POSITION.value, "reduce_position")
        self.assertEqual(MitigationAction.EMERGENCY_STOP.value, "emergency_stop")
        print(f"  [PASS] test_33: mitigation actions: {len(MitigationAction)} actions")

    def test_34_risk_dimensions(self):
        """风险维度"""
        dimensions = list(RiskDimension)
        self.assertGreaterEqual(len(dimensions), 7)
        self.assertIn(RiskDimension.DIRECTION, dimensions)
        self.assertIn(RiskDimension.DRAWDOWN, dimensions)
        print(f"  [PASS] test_34: {len(dimensions)} risk dimensions")

    def test_35_session_state_coverage(self):
        """会话状态覆盖"""
        states = list(SessionState)
        self.assertGreaterEqual(len(states), 10)
        self.assertIn(SessionState.IDLE, states)
        self.assertIn(SessionState.TRADING, states)
        self.assertIn(SessionState.EMERGENCY, states)
        print(f"  [PASS] test_35: {len(states)} session states")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 60)
    print("生产级交易核心集成测试")
    print("=" * 60)
    print(f"时间: {datetime.now().isoformat()}")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSessionStateMachine))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionStrategyManagement))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionPreMarketChecks))
    suite.addTests(loader.loadTestsFromTestCase(TestSessionMetrics))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskMonitorScoring))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskMonitorChecks))
    suite.addTests(loader.loadTestsFromTestCase(TestRiskMonitorAlerts))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))

    runner = unittest.TextTestRunner(verbosity=0)
    result = runner.run(suite)

    print()
    print("=" * 60)
    passed = result.testsRun - len(result.failures) - len(result.errors)
    print(f"测试结果: {passed}/{result.testsRun} 通过")
    if result.failures:
        print(f"  失败: {len(result.failures)}")
        for test, traceback in result.failures:
            print(f"    - {test}")
    if result.errors:
        print(f"  错误: {len(result.errors)}")
        for test, traceback in result.errors:
            print(f"    - {test}: {traceback[:200]}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)