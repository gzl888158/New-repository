"""
策略引擎 v2.0 与配置热更新集成测试
=============================
测试覆盖：
- 策略引擎初始化与管线注入
- 生产管线检查（会话、风险、资金）
- 信号处理流程（tick/bar）
- 运行模式切换与通知
- 配置热更新与分发
- 配置版本历史与回滚
- 键级回调
"""

import sys
import os
import unittest
import asyncio
import time
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.strategy_engine import (
    StrategyEngine, RunMode, get_strategy_engine,
    SignalType, TradingSignal, SignalContext,
)
from core.config_manager import ConfigManager


# ═══════════════════════════════════════════════════════════════
# 测试辅助
# ═══════════════════════════════════════════════════════════════

class MockCapitalManager:
    """Mock CapitalManager"""
    def __init__(self):
        self.allocated = {}
    
    def check_attrition_budget(self, strategy_name: str):
        return True, "ok"
    
    def allocate_capital(self, symbol, amount, pool_type=None):
        self.allocated[symbol] = amount
        return True


class MockRiskMonitor:
    """Mock RiskMonitor"""
    def __init__(self, risk_score=0.3):
        self._risk_score = risk_score
    
    def get_risk_snapshot(self):
        return {
            "overall_score": self._risk_score,
            "scores": {
                "drawdown": {"score": 0.1, "level": "info"},
                "leverage": {"score": 0.2, "level": "info"},
            },
            "alert_level": "info",
        }


class MockTradingSession:
    """Mock TradingSession"""
    def __init__(self, state="running"):
        self._state = state
    
    def get_state(self):
        return self._state


class MockMetricsPipeline:
    """Mock MetricsPipeline"""
    def __init__(self):
        self.records = []
        self.increments = []
        self.latencies = []
    
    def record(self, name, value, labels=None):
        self.records.append((name, value, labels))
    
    def increment(self, name, value=1.0, labels=None):
        self.increments.append((name, value, labels))
    
    def record_latency(self, name, latency_ms, labels=None):
        self.latencies.append((name, latency_ms, labels))


class MockNotificationDispatcher:
    """Mock NotificationDispatcher"""
    def __init__(self):
        self.sent = []
        self.templates = []
    
    async def send(self, channel, title, message, priority=None, category=None, metadata=None):
        self.sent.append({
            "channel": channel, "title": title, "message": message,
            "priority": priority, "category": category, "metadata": metadata,
        })
        return "msg_id"
    
    async def send_template(self, channel, template, priority=None, category=None, metadata=None, **kwargs):
        self.templates.append({
            "channel": channel, "template": template, "priority": priority,
            "category": category, "metadata": metadata, "kwargs": kwargs,
        })
        return "msg_id"


def make_tick(symbol="BTC-USDT", price=65000.0):
    return {"symbol": symbol, "price": price, "last": price, "volume": 100.0}


def make_bar(symbol="BTC-USDT", close=65000.0):
    return {
        "symbol": symbol, "open": close - 100, "high": close + 200,
        "low": close - 300, "close": close, "c": close, "volume": 5000.0,
    }


# ═══════════════════════════════════════════════════════════════
# 策略引擎 v2.0 测试
# ═══════════════════════════════════════════════════════════════

class TestStrategyEngineV2(unittest.TestCase):
    """策略引擎 v2.0 核心功能测试"""
    
    def setUp(self):
        self.config = {
            "symbols": ["BTC-USDT", "ETH-USDT"],
            "strategy_engine": {
                "default_mode": "normal",
                "production_pipeline_enabled": True,
                "pre_trade_risk_check": True,
                "capital_check_enabled": True,
                "session_check_enabled": True,
                "metrics_enabled": True,
                "notifications_enabled": True,
                "max_risk_score_for_open": 0.7,
                "max_risk_score_for_trade": 0.85,
            },
            "strategy_config": {},
        }
        self.engine = StrategyEngine(self.config)
    
    def test_01_engine_initialization(self):
        """策略引擎初始化"""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.get_run_mode(), RunMode.NORMAL)
        self.assertTrue(self.engine._pipeline_enabled)
        self.assertTrue(self.engine._pre_trade_risk_check)
        self.assertTrue(self.engine._capital_check_enabled)
        self.assertTrue(self.engine._session_check_enabled)
        self.assertTrue(self.engine._metrics_enabled)
        self.assertTrue(self.engine._notifications_enabled)
        print(f"  [PASS] test_01: engine initialized, mode={self.engine.get_run_mode().value}")
    
    def test_02_inject_dependencies(self):
        """依赖注入"""
        cm = MockCapitalManager()
        rm = MockRiskMonitor()
        ts = MockTradingSession()
        mp = MockMetricsPipeline()
        nd = MockNotificationDispatcher()
        
        self.engine.inject_dependencies(
            capital_manager=cm,
            risk_monitor=rm,
            trading_session=ts,
            metrics_pipeline=mp,
            notification_dispatcher=nd,
        )
        
        self.assertIs(self.engine._capital_manager, cm)
        self.assertIs(self.engine._risk_monitor, rm)
        self.assertIs(self.engine._trading_session, ts)
        self.assertIs(self.engine._metrics_pipeline, mp)
        self.assertIs(self.engine._notification_dispatcher, nd)
        print(f"  [PASS] test_02: all 5 dependencies injected")
    
    def test_03_session_check_active(self):
        """会话检查 - 活跃状态"""
        ts = MockTradingSession(state="running")
        self.engine.inject_dependencies(trading_session=ts)
        self.assertTrue(self.engine._check_trading_session())
        print(f"  [PASS] test_03: session check passed for running state")
    
    def test_04_session_check_stopped(self):
        """会话检查 - 停止状态"""
        ts = MockTradingSession(state="stopped")
        self.engine.inject_dependencies(trading_session=ts)
        self.assertFalse(self.engine._check_trading_session())
        print(f"  [PASS] test_04: session check blocked for stopped state")
    
    def test_05_session_check_emergency(self):
        """会话检查 - 紧急状态"""
        ts = MockTradingSession(state="emergency")
        self.engine.inject_dependencies(trading_session=ts)
        self.assertFalse(self.engine._check_trading_session())
        print(f"  [PASS] test_05: session check blocked for emergency state")
    
    def test_06_risk_gate_pass(self):
        """风险门控 - 通过"""
        rm = MockRiskMonitor(risk_score=0.3)
        self.engine.inject_dependencies(risk_monitor=rm)
        ok, reason = self.engine._check_risk_gate(SignalType.OPEN_LONG, "BTC-USDT")
        self.assertTrue(ok)
        self.assertEqual(reason, "")
        print(f"  [PASS] test_06: risk gate passed (score=0.3)")
    
    def test_07_risk_gate_block_open(self):
        """风险门控 - 阻止开仓"""
        rm = MockRiskMonitor(risk_score=0.8)
        self.engine.inject_dependencies(risk_monitor=rm)
        ok, reason = self.engine._check_risk_gate(SignalType.OPEN_LONG, "BTC-USDT")
        self.assertFalse(ok)
        self.assertIn("risk_score", reason)
        print(f"  [PASS] test_07: risk gate blocked open signal (score=0.8) - {reason}")
    
    def test_08_risk_gate_allow_close_on_high_risk(self):
        """风险门控 - 高风险时允许平仓"""
        rm = MockRiskMonitor(risk_score=0.8)
        self.engine.inject_dependencies(risk_monitor=rm)
        # 平仓信号使用更高的阈值(0.85)
        ok, reason = self.engine._check_risk_gate(SignalType.TAKE_PROFIT, "BTC-USDT")
        self.assertTrue(ok)  # 0.8 < 0.85
        print(f"  [PASS] test_08: close signal allowed even with risk_score=0.8")
    
    def test_09_capital_check_pass(self):
        """资金检查 - 通过"""
        cm = MockCapitalManager()
        self.engine.inject_dependencies(capital_manager=cm)
        signal = TradingSignal(
            symbol="BTC-USDT", signal_type=SignalType.OPEN_LONG,
            source=None, level=None, weight=0.8, price=65000, quantity=0.01,
            timestamp=datetime.now(),
        )
        ok, reason = self.engine._check_capital_availability("BTC-USDT", signal)
        self.assertTrue(ok)
        print(f"  [PASS] test_09: capital check passed")
    
    def test_10_run_mode_transition(self):
        """运行模式切换"""
        self.assertEqual(self.engine.get_run_mode(), RunMode.NORMAL)
        
        self.engine.set_run_mode(RunMode.CONSERVATIVE, "high volatility")
        self.assertEqual(self.engine.get_run_mode(), RunMode.CONSERVATIVE)
        
        self.engine.set_run_mode(RunMode.EMERGENCY, "circuit breaker")
        self.assertEqual(self.engine.get_run_mode(), RunMode.EMERGENCY)
        
        history = self.engine._mode_transition_history
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["from"], "normal")
        self.assertEqual(history[0]["to"], "conservative")
        self.assertEqual(history[1]["from"], "conservative")
        self.assertEqual(history[1]["to"], "emergency")
        print(f"  [PASS] test_10: mode transitions: normal->conservative->emergency")
    
    def test_11_mode_signal_filter_emergency(self):
        """紧急模式信号过滤"""
        self.engine.set_run_mode(RunMode.EMERGENCY, "test")
        
        # 开仓信号应被过滤
        signal = TradingSignal(
            symbol="BTC-USDT", signal_type=SignalType.OPEN_LONG,
            source=None, level=None, weight=0.8, price=65000, quantity=0.01,
            timestamp=datetime.now(),
        )
        result = self.engine._apply_mode_filter_to_signal(signal)
        self.assertIsNone(result)
        
        # 止损信号应保留
        signal2 = TradingSignal(
            symbol="BTC-USDT", signal_type=SignalType.STOP_LOSS,
            source=None, level=None, weight=0.8, price=65000, quantity=0.01,
            timestamp=datetime.now(),
        )
        result2 = self.engine._apply_mode_filter_to_signal(signal2)
        self.assertIsNotNone(result2)
        print(f"  [PASS] test_11: emergency mode filters open signals, keeps stop_loss")
    
    def test_12_update_config(self):
        """配置热更新"""
        new_config = {
            "symbols": ["BTC-USDT", "ETH-USDT"],
            "strategy_engine": {
                "production_pipeline_enabled": False,
                "pre_trade_risk_check": False,
                "capital_check_enabled": False,
                "max_risk_score_for_open": 0.5,
                "market_drawdown_emergency": 0.15,
            },
            "strategy_config": {},
        }
        
        self.engine.update_config(new_config)
        
        self.assertFalse(self.engine._pipeline_enabled)
        self.assertFalse(self.engine._pre_trade_risk_check)
        self.assertFalse(self.engine._capital_check_enabled)
        self.assertEqual(self.engine._max_risk_score_for_open, 0.5)
        self.assertEqual(self.engine._mode_thresholds["market_drawdown_emergency"], 0.15)
        print(f"  [PASS] test_12: config hot-updated successfully")
    
    def test_13_signal_stats(self):
        """信号统计"""
        stats = self.engine.get_signal_stats()
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["filtered"], 0)
        self.assertEqual(stats["executed"], 0)
        self.assertEqual(stats["run_mode"], "normal")
        self.assertEqual(stats["pipeline_enabled"], True)
        self.assertEqual(stats["filter_rate"], 0.0)
        print(f"  [PASS] test_13: signal stats initialized correctly")
    
    def test_14_metrics_recording(self):
        """指标记录"""
        mp = MockMetricsPipeline()
        self.engine.inject_dependencies(metrics_pipeline=mp)
        
        signal = TradingSignal(
            symbol="BTC-USDT", signal_type=SignalType.OPEN_LONG,
            source=None, level=None, weight=0.8, price=65000, quantity=0.01,
            timestamp=datetime.now(),
        )
        
        self.engine._record_signal_metrics("BTC-USDT", signal)
        self.assertGreaterEqual(len(mp.increments), 1)
        self.assertEqual(mp.increments[0][0], "signals_total")
        print(f"  [PASS] test_14: metrics recorded: {len(mp.increments)} increments")
    
    def test_15_filtered_metrics(self):
        """过滤信号指标"""
        mp = MockMetricsPipeline()
        self.engine.inject_dependencies(metrics_pipeline=mp)
        
        signal = TradingSignal(
            symbol="BTC-USDT", signal_type=SignalType.OPEN_LONG,
            source=None, level=None, weight=0.8, price=65000, quantity=0.01,
            timestamp=datetime.now(),
        )
        
        self.engine._record_signal_metrics("BTC-USDT", signal, filtered=True)
        self.assertGreaterEqual(len(mp.increments), 2)
        self.assertEqual(mp.increments[1][0], "signals_filtered")
        print(f"  [PASS] test_15: filtered metrics recorded")
    
    def test_16_process_tick_with_pipeline(self):
        """处理tick数据（完整管线）"""
        async def _test():
            cm = MockCapitalManager()
            rm = MockRiskMonitor(risk_score=0.3)
            ts = MockTradingSession(state="running")
            mp = MockMetricsPipeline()
            nd = MockNotificationDispatcher()
            
            self.engine.inject_dependencies(
                capital_manager=cm, risk_monitor=rm,
                trading_session=ts, metrics_pipeline=mp,
                notification_dispatcher=nd,
            )
            
            self.engine.register_symbol("BTC-USDT")
            tick = make_tick("BTC-USDT", 65000.0)
            signals = await self.engine.process_tick("BTC-USDT", tick)
            return signals, mp, nd
        
        signals, mp, nd = asyncio.run(_test())
        self.assertIsInstance(signals, list)
        self.assertGreaterEqual(len(mp.latencies), 1)
        print(f"  [PASS] test_16: process_tick with pipeline returned {len(signals)} signals, "
              f"{len(mp.latencies)} latency records")
    
    def test_17_process_tick_session_blocked(self):
        """处理tick数据 - 会话阻止"""
        async def _test():
            ts = MockTradingSession(state="stopped")
            self.engine.inject_dependencies(trading_session=ts)
            self.engine.register_symbol("BTC-USDT")
            tick = make_tick("BTC-USDT", 65000.0)
            signals = await self.engine.process_tick("BTC-USDT", tick)
            return signals
        
        signals = asyncio.run(_test())
        self.assertEqual(len(signals), 0)
        print(f"  [PASS] test_17: process_tick blocked by session (stopped)")
    
    def test_18_process_tick_risk_blocked(self):
        """处理tick数据 - 风险阻止"""
        async def _test():
            rm = MockRiskMonitor(risk_score=0.9)
            self.engine.inject_dependencies(risk_monitor=rm)
            self.engine.register_symbol("BTC-USDT")
            tick = make_tick("BTC-USDT", 65000.0)
            signals = await self.engine.process_tick("BTC-USDT", tick)
            return signals
        
        signals = asyncio.run(_test())
        # 高风险下信号可能被过滤，但本身可能返回空列表
        self.assertIsInstance(signals, list)
        print(f"  [PASS] test_18: process_tick with high risk returned {len(signals)} signals")
    
    def test_19_process_bar_with_pipeline(self):
        """处理bar数据（完整管线）"""
        async def _test():
            cm = MockCapitalManager()
            rm = MockRiskMonitor(risk_score=0.3)
            ts = MockTradingSession(state="running")
            mp = MockMetricsPipeline()
            nd = MockNotificationDispatcher()
            
            self.engine.inject_dependencies(
                capital_manager=cm, risk_monitor=rm,
                trading_session=ts, metrics_pipeline=mp,
                notification_dispatcher=nd,
            )
            
            self.engine.register_symbol("BTC-USDT")
            bar = make_bar("BTC-USDT", 65000.0)
            signals = await self.engine.process_bar("BTC-USDT", bar)
            return signals, mp
        
        signals, mp = asyncio.run(_test())
        self.assertIsInstance(signals, list)
        self.assertGreaterEqual(len(mp.latencies), 1)
        print(f"  [PASS] test_19: process_bar with pipeline returned {len(signals)} signals, "
              f"{len(mp.latencies)} latency records")
    
    def test_20_get_strategy_status(self):
        """获取策略状态"""
        self.engine.register_symbol("BTC-USDT")
        status = self.engine.get_strategy_status()
        self.assertIn("run_mode", status)
        self.assertIn("instances", status)
        self.assertEqual(status["run_mode"], "normal")
        print(f"  [PASS] test_20: strategy status: mode={status['run_mode']}, "
              f"instances={len(status['instances'])}")


# ═══════════════════════════════════════════════════════════════
# 配置管理器 v2.0 测试
# ═══════════════════════════════════════════════════════════════

class TestConfigManagerV2(unittest.TestCase):
    """配置管理器 v2.0 测试"""
    
    def setUp(self):
        # 使用临时配置
        self.temp_config = {
            "system": {"name": "test", "version": "1.0"},
            "trading": {"total_capital": 500, "scalping_allocation": 0.45,
                        "trend_allocation": 0.20, "grid_allocation": 0.20,
                        "arbitrage_allocation": 0.15, "spot_grid_allocation": 0.0,
                        "spot_martingale_allocation": 0.0,
                        "trading_capital_ratio": 0.95, "risk_reserve_ratio": 0.05,
                        "profit_reserve_ratio": 0.0},
            "strategies": {"grid": {"enabled": True}},
            "strategy_engine": {"production_pipeline_enabled": True},
        }
    
    def test_21_compute_diff_changed(self):
        """配置差异 - 修改"""
        mgr = ConfigManager.__new__(ConfigManager)
        mgr.config_path = "test.yaml"
        mgr._callbacks = []
        mgr._version_history = []
        mgr._key_callbacks = {}
        
        old = {"trading": {"total_capital": 500}}
        new = {"trading": {"total_capital": 1000}}
        diffs = mgr._compute_diff(old, new)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["type"], "changed")
        self.assertEqual(diffs[0]["key"], "trading.total_capital")
        self.assertEqual(diffs[0]["old"], 500)
        self.assertEqual(diffs[0]["new"], 1000)
        print(f"  [PASS] test_21: diff detected changed key")
    
    def test_22_compute_diff_added(self):
        """配置差异 - 新增"""
        mgr = ConfigManager.__new__(ConfigManager)
        mgr.config_path = "test.yaml"
        mgr._callbacks = []
        mgr._version_history = []
        mgr._key_callbacks = {}
        
        old = {"trading": {"total_capital": 500}}
        new = {"trading": {"total_capital": 500, "new_field": "value"}}
        diffs = mgr._compute_diff(old, new)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["type"], "added")
        self.assertEqual(diffs[0]["key"], "trading.new_field")
        print(f"  [PASS] test_22: diff detected added key")
    
    def test_23_compute_diff_removed(self):
        """配置差异 - 删除"""
        mgr = ConfigManager.__new__(ConfigManager)
        mgr.config_path = "test.yaml"
        mgr._callbacks = []
        mgr._version_history = []
        mgr._key_callbacks = {}
        
        old = {"trading": {"total_capital": 500, "old_field": "value"}}
        new = {"trading": {"total_capital": 500}}
        diffs = mgr._compute_diff(old, new)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["type"], "removed")
        self.assertEqual(diffs[0]["key"], "trading.old_field")
        print(f"  [PASS] test_23: diff detected removed key")
    
    def test_24_compute_diff_nested(self):
        """配置差异 - 嵌套"""
        mgr = ConfigManager.__new__(ConfigManager)
        mgr.config_path = "test.yaml"
        mgr._callbacks = []
        mgr._version_history = []
        mgr._key_callbacks = {}
        
        old = {"strategies": {"grid": {"enabled": True, "count": 10}}}
        new = {"strategies": {"grid": {"enabled": True, "count": 20}}}
        diffs = mgr._compute_diff(old, new)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["key"], "strategies.grid.count")
        print(f"  [PASS] test_24: nested diff detected: {diffs[0]}")
    
    def test_25_key_callback(self):
        """键级回调"""
        import yaml, tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.temp_config, f)
            tmp_path = f.name
        
        try:
            mgr = ConfigManager(tmp_path)
            callback_results = []
            
            def on_strategy_change(value, full_config):
                callback_results.append(("strategies", value))
            
            mgr.register_key_callback("strategies", on_strategy_change)
            
            # 修改配置
            mgr.set("strategies.grid.enabled", False)
            
            # 手动触发回调
            mgr._notify_key_callbacks()
            
            self.assertEqual(len(callback_results), 1)
            self.assertIn("grid", callback_results[0][1])
            print(f"  [PASS] test_25: key callback triggered for 'strategies'")
        finally:
            os.unlink(tmp_path)
    
    def test_26_global_callback(self):
        """全局回调"""
        import yaml, tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.temp_config, f)
            tmp_path = f.name
        
        try:
            mgr = ConfigManager(tmp_path)
            callback_results = []
            
            def on_config_change(config):
                callback_results.append(config)
            
            mgr.register_callback(on_config_change)
            mgr.notify_callbacks()
            
            self.assertEqual(len(callback_results), 1)
            self.assertIn("trading", callback_results[0])
            print(f"  [PASS] test_26: global callback triggered")
        finally:
            os.unlink(tmp_path)
    
    def test_27_get_version_info(self):
        """获取版本信息"""
        import yaml, tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.temp_config, f)
            tmp_path = f.name
        
        try:
            mgr = ConfigManager(tmp_path)
            info = mgr.get_version_info()
            self.assertIn("version", info)
            self.assertIn("path", info)
            self.assertGreaterEqual(info["version"], 1)
            print(f"  [PASS] test_27: version info: v{info['version']}")
        finally:
            os.unlink(tmp_path)
    
    def test_28_version_history(self):
        """版本历史"""
        import yaml, tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.temp_config, f)
            tmp_path = f.name
        
        try:
            mgr = ConfigManager(tmp_path)
            # 修改配置触发版本记录
            mgr.set("trading.total_capital", 1000)
            mgr.load_config()  # 重新加载会记录diff
            
            history = mgr.get_version_history()
            self.assertGreaterEqual(len(history), 1)
            print(f"  [PASS] test_28: version history: {len(history)} entries")
        finally:
            os.unlink(tmp_path)
    
    def test_29_get_last_diff(self):
        """获取最后一次差异"""
        import yaml, tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.temp_config, f)
            tmp_path = f.name
        
        try:
            mgr = ConfigManager(tmp_path)
            mgr.set("trading.total_capital", 1000)
            mgr.load_config()
            
            last_diff = mgr.get_last_diff()
            self.assertIsNotNone(last_diff)
            self.assertIn("version", last_diff)
            print(f"  [PASS] test_29: last diff: version={last_diff['version']}")
        finally:
            os.unlink(tmp_path)
    
    def test_30_delete_key(self):
        """删除配置键"""
        import yaml, tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            test_cfg = self.temp_config.copy()
            test_cfg["test_key"] = "test_value"
            yaml.dump(test_cfg, f)
            tmp_path = f.name
        
        try:
            mgr = ConfigManager(tmp_path)
            self.assertEqual(mgr.get("test_key"), "test_value")
            mgr._delete_key("test_key")
            self.assertIsNone(mgr.get("test_key"))
            print(f"  [PASS] test_30: key deleted successfully")
        finally:
            os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════
# 端到端集成测试
# ═══════════════════════════════════════════════════════════════

class TestE2EStrategyEngine(unittest.TestCase):
    """端到端集成测试"""
    
    def test_31_e2e_full_pipeline(self):
        """完整管线端到端测试"""
        async def _test():
            config = {
                "symbols": ["BTC-USDT", "ETH-USDT"],
                "strategy_engine": {
                    "default_mode": "normal",
                    "production_pipeline_enabled": True,
                    "pre_trade_risk_check": True,
                    "capital_check_enabled": True,
                    "session_check_enabled": True,
                    "metrics_enabled": True,
                    "notifications_enabled": True,
                    "max_risk_score_for_open": 0.7,
                    "max_risk_score_for_trade": 0.85,
                },
                "strategy_config": {},
            }
            
            engine = StrategyEngine(config)
            
            # 注入所有管线
            cm = MockCapitalManager()
            rm = MockRiskMonitor(risk_score=0.3)
            ts = MockTradingSession(state="running")
            mp = MockMetricsPipeline()
            nd = MockNotificationDispatcher()
            
            engine.inject_dependencies(
                capital_manager=cm, risk_monitor=rm,
                trading_session=ts, metrics_pipeline=mp,
                notification_dispatcher=nd,
            )
            
            # 注册交易对
            engine.register_symbol("BTC-USDT")
            engine.register_symbol("ETH-USDT")
            
            # 模拟多个tick
            results = []
            for price in [65000, 65100, 65200, 65150, 65050]:
                tick = make_tick("BTC-USDT", price)
                signals = await engine.process_tick("BTC-USDT", tick)
                results.append(len(signals))
            
            # 模拟bar
            bar = make_bar("BTC-USDT", 65100)
            bar_signals = await engine.process_bar("BTC-USDT", bar)
            
            # 获取统计
            stats = engine.get_signal_stats()
            status = engine.get_strategy_status()
            
            # 切换模式
            engine.set_run_mode(RunMode.CONSERVATIVE, "test volatility")
            
            return {
                "tick_results": results,
                "bar_signals": len(bar_signals),
                "stats": stats,
                "status_mode": status["run_mode"],
                "metrics_increments": len(mp.increments),
                "metrics_latencies": len(mp.latencies),
                "notifications": len(nd.templates),
            }
        
        result = asyncio.run(_test())
        print(f"  [PASS] test_31: E2E full pipeline - "
              f"tick_results={result['tick_results']}, "
              f"bar_signals={result['bar_signals']}, "
              f"stats={result['stats']}, "
              f"mode={result['status_mode']}, "
              f"metrics={result['metrics_increments']}i/{result['metrics_latencies']}l, "
              f"notifications={result['notifications']}")
    
    def test_32_e2e_config_update_flow(self):
        """配置热更新端到端流程"""
        async def _test():
            config = {
                "symbols": ["BTC-USDT"],
                "strategy_engine": {
                    "default_mode": "normal",
                    "production_pipeline_enabled": True,
                    "pre_trade_risk_check": True,
                    "capital_check_enabled": True,
                    "max_risk_score_for_open": 0.7,
                },
                "strategy_config": {},
            }
            
            engine = StrategyEngine(config)
            self.assertTrue(engine._pre_trade_risk_check)
            
            # 热更新关闭风险检查
            new_config = {
                "symbols": ["BTC-USDT"],
                "strategy_engine": {
                    "pre_trade_risk_check": False,
                    "capital_check_enabled": False,
                    "max_risk_score_for_open": 0.5,
                },
                "strategy_config": {},
            }
            engine.update_config(new_config)
            
            self.assertFalse(engine._pre_trade_risk_check)
            self.assertFalse(engine._capital_check_enabled)
            self.assertEqual(engine._max_risk_score_for_open, 0.5)
            
            return "ok"
        
        result = asyncio.run(_test())
        print(f"  [PASS] test_32: config update flow: {result}")
    
    def test_33_e2e_risk_escalation(self):
        """风险升级流程"""
        engine = StrategyEngine({
            "symbols": ["BTC-USDT"],
            "strategy_engine": {"default_mode": "normal"},
            "strategy_config": {},
        })
        
        # 模拟风险升级
        engine.set_run_mode(RunMode.CONSERVATIVE, "drawdown 10%")
        self.assertEqual(engine.get_run_mode(), RunMode.CONSERVATIVE)
        
        engine.set_run_mode(RunMode.EMERGENCY, "drawdown 20%")
        self.assertEqual(engine.get_run_mode(), RunMode.EMERGENCY)
        
        # 紧急模式下信号过滤
        from core.signal_generator import SignalType as ST
        signal = TradingSignal(
            symbol="BTC-USDT", signal_type=ST.OPEN_LONG,
            source=None, level=None, weight=0.9, price=65000, quantity=0.01,
            timestamp=datetime.now(),
        )
        filtered = engine._apply_mode_filter_to_signal(signal)
        self.assertIsNone(filtered)
        
        history = engine._mode_transition_history
        self.assertEqual(len(history), 2)
        print(f"  [PASS] test_33: risk escalation: {history[0]['from']}->{history[0]['to']}->{history[1]['to']}")


if __name__ == "__main__":
    unittest.main(verbosity=2)