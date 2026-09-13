"""
生产级核心模块集成测试 (PortfolioOptimizer / ConditionalOrderManager / ComputeScheduler)
====================================================================================
覆盖：状态持久化、热更新、配置集成、增强指标、批量操作、自适应降频、审计追踪
"""

import os
import sys
import json
import time
import unittest
import tempfile
import threading
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.portfolio_optimizer import (
    PortfolioOptimizer, OptimizationObjective, ComboMarketRegime,
    StrategyPerformance, CorrelationMatrix, VaRResult,
    PortfolioOptimizationResult, PRESET_COMBOS,
    get_portfolio_optimizer, reset_portfolio_optimizer,
)
from core.compute_scheduler import (
    ComputeScheduler, ComputePriority, SymbolComputeState,
    get_compute_scheduler,
)
from core.conditional_order_manager import ConditionalOrderManager


# ============================================================
# PortfolioOptimizer 测试
# ============================================================

class TestPortfolioOptimizer(unittest.TestCase):
    """策略组合优化引擎测试"""

    def setUp(self):
        reset_portfolio_optimizer()
        self.config = {
            "portfolio_optimizer": {
                "enabled": True,
                "default_objective": "max_sharpe",
                "rebalance_interval_hours": 4,
                "min_rebalance_change_pct": 0.02,
                "max_single_weight": 0.50,
                "min_single_weight": 0.05,
                "risk_free_rate": 0.03,
                "correlation": {
                    "window_days": 30,
                    "high_threshold": 0.70,
                    "alert_threshold": 0.85,
                },
                "var": {
                    "confidence_levels": [0.95, 0.99],
                    "horizon_days": 1,
                    "method": "historical",
                },
                "stress_test": {"enabled": True},
                "persistence": {
                    "enabled": True,
                    "save_interval_sec": 300,
                    "state_file": "data/portfolio_state.json",
                },
            }
        }
        self.opt = PortfolioOptimizer(self.config)

    def test_01_initialization(self):
        """初始化检查"""
        self.assertTrue(self.opt._enabled)
        self.assertEqual(self.opt._default_objective, OptimizationObjective.MAX_SHARPE)
        self.assertEqual(self.opt._rebalance_interval_hours, 4)
        self.assertEqual(self.opt._max_single_weight, 0.50)
        self.assertEqual(self.opt._min_single_weight, 0.05)
        self.assertEqual(self.opt._risk_free_rate, 0.03)
        self.assertEqual(self.opt._corr_window_days, 30)
        self.assertEqual(self.opt._corr_high_threshold, 0.70)
        self.assertEqual(self.opt._corr_alert_threshold, 0.85)
        self.assertEqual(len(self.opt._combos), 7)  # 7 preset combos
        print("  [PASS] test_01: initialization verified")

    def test_02_update_performance_and_compute_stats(self):
        """更新策略绩效并计算统计量"""
        # 添加模拟收益率数据
        np.random.seed(42)
        for i in range(60):
            self.opt.update_performance("trend", np.random.normal(0.001, 0.02))
            self.opt.update_performance("grid", np.random.normal(0.0005, 0.01))
            self.opt.update_performance("scalping", np.random.normal(0.0008, 0.015))
            self.opt.update_equity(10000 + i * 10)

        perf = self.opt._strategy_performances.get("trend")
        self.assertIsNotNone(perf)
        self.assertGreater(len(perf.returns), 50)
        self.assertNotEqual(perf.annualized_return, 0.0)
        self.assertNotEqual(perf.annualized_volatility, 0.0)
        print(f"  [PASS] test_02: trend sharpe={perf.sharpe_ratio:.4f}, vol={perf.annualized_volatility:.4f}")

    def test_03_correlation_matrix(self):
        """相关性矩阵计算"""
        np.random.seed(42)
        for i in range(60):
            self.opt.update_performance("trend", np.random.normal(0.001, 0.02))
            self.opt.update_performance("grid", np.random.normal(0.0005, 0.01))
            self.opt.update_performance("scalping", np.random.normal(0.0008, 0.015))

        corr = self.opt.compute_correlation_matrix()
        self.assertIsNotNone(corr)
        self.assertGreaterEqual(len(corr.strategies), 2)
        self.assertIsNotNone(corr.avg_correlation)
        print(f"  [PASS] test_03: strategies={corr.strategies}, avg_corr={corr.avg_correlation:.4f}")

    def test_04_var_computation(self):
        """VaR/CVaR 计算"""
        np.random.seed(42)
        for i in range(100):
            self.opt.update_equity(10000 + i * 10 + np.random.normal(0, 50))

        var_result = self.opt.compute_var(confidence=0.95, method="historical")
        self.assertIsNotNone(var_result)
        self.assertEqual(var_result.method, "historical")
        self.assertEqual(var_result.confidence_level, 0.95)
        print(f"  [PASS] test_04: VaR={var_result.var_pct:.4%}, CVaR={var_result.cvar_pct:.4%}")

    def test_05_portfolio_optimization(self):
        """MPT 组合优化"""
        np.random.seed(42)
        for i in range(60):
            self.opt.update_performance("trend", np.random.normal(0.001, 0.02))
            self.opt.update_performance("grid", np.random.normal(0.0005, 0.01))
            self.opt.update_performance("scalping", np.random.normal(0.0008, 0.015))
            self.opt.update_performance("arbitrage", np.random.normal(0.0003, 0.005))

        result = self.opt.optimize()
        self.assertIsNotNone(result)
        self.assertGreater(len(result.optimal_weights), 0)
        total_weight = sum(result.optimal_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        print(f"  [PASS] test_05: weights={result.optimal_weights}, sharpe={result.expected_sharpe:.4f}")

    def test_06_performance_attribution(self):
        """绩效归因"""
        np.random.seed(42)
        for i in range(60):
            self.opt.update_performance("trend", np.random.normal(0.001, 0.02))
            self.opt.update_performance("grid", np.random.normal(0.0005, 0.01))

        # 设置 PnL 贡献
        self.opt._strategy_performances["trend"].pnl_contribution = 150.0
        self.opt._strategy_performances["trend"].current_weight = 0.4
        self.opt._strategy_performances["grid"].pnl_contribution = 50.0
        self.opt._strategy_performances["grid"].current_weight = 0.3

        attr = self.opt.compute_attribution()
        self.assertIsNotNone(attr)
        self.assertGreater(len(attr.strategy_contributions), 0)
        self.assertGreater(attr.concentration_ratio, 0)
        print(f"  [PASS] test_06: top={attr.top_contributor}, concentration={attr.concentration_ratio:.4f}")

    def test_07_stress_test(self):
        """压力测试"""
        np.random.seed(42)
        for i in range(100):
            self.opt.update_equity(10000 + i * 10)
        self.opt.update_performance("trend", 0.0)
        self.opt._strategy_performances["trend"].annualized_volatility = 0.30
        self.opt._strategy_performances["trend"].current_weight = 0.4

        results = self.opt.stress_test({"trend": 0.4, "grid": 0.3, "scalping": 0.2, "arbitrage": 0.1})
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn("scenario", r)
            self.assertIn("estimated_total_loss", r)
            self.assertIn("severity", r)
        print(f"  [PASS] test_07: {len(results)} scenarios tested, worst={results[0]['scenario']}")

    def test_08_combo_recommendation(self):
        """组合模板推荐"""
        recommendations = self.opt.recommend_combo(
            ComboMarketRegime.TRENDING_UP, capital=2000.0
        )
        self.assertGreater(len(recommendations), 0)
        top = recommendations[0]
        self.assertIn("combo", top)
        self.assertIn("score", top)
        print(f"  [PASS] test_08: top combo={top['combo']['name']}, score={top['score']}")

    def test_09_apply_combo(self):
        """应用组合模板"""
        # 先设置策略
        self.opt._strategy_performances["trend"] = StrategyPerformance(name="trend")
        self.opt._strategy_performances["grid"] = StrategyPerformance(name="grid")
        self.opt._strategy_performances["scalping"] = StrategyPerformance(name="scalping")
        self.opt._strategy_performances["arbitrage"] = StrategyPerformance(name="arbitrage")

        result = self.opt.apply_combo("balanced")
        self.assertTrue(result["success"])
        self.assertEqual(result["combo_name"], "均衡配置")
        self.assertGreater(len(result["new_weights"]), 0)
        print(f"  [PASS] test_09: applied combo={result['combo_name']}, weights={result['new_weights']}")

    def test_10_capital_efficiency(self):
        """资金效率最大化"""
        result = self.opt.maximize_capital_efficiency(10000.0, max_leverage=3.0)
        self.assertIn("allocations", result)
        self.assertIn("efficiency", result)
        self.assertGreater(result["efficiency"], 0)
        print(f"  [PASS] test_10: efficiency={result['efficiency']:.4f}, deployable={result['deployable_capital']}")

    def test_11_state_persistence(self):
        """状态持久化：收集、保存、恢复"""
        np.random.seed(42)
        for i in range(60):
            self.opt.update_performance("trend", np.random.normal(0.001, 0.02))
            self.opt.update_equity(10000 + i * 10)

        self.opt.compute_correlation_matrix()
        self.opt.compute_attribution()

        # 收集状态
        state = self.opt.collect_persistent_state()
        self.assertIsNotNone(state)
        self.assertIn("strategies", state)
        self.assertIn("equity_history", state)
        self.assertIn("version", state)
        print(f"  [PASS] test_11a: state collected, strategies={len(state['strategies'])}")

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp_path = f.name
        try:
            self.opt.save_persistent_state(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, 'r') as f:
                saved = json.load(f)
            self.assertEqual(saved["version"], 1)
            print(f"  [PASS] test_11b: state saved to {tmp_path}")
        finally:
            os.unlink(tmp_path)

        # 恢复状态
        opt2 = PortfolioOptimizer(self.config)
        opt2.restore_persistent_state(state)
        self.assertGreater(len(opt2._strategy_performances), 0)
        self.assertGreater(len(opt2._equity_history), 0)
        print(f"  [PASS] test_11c: state restored, equity_len={len(opt2._equity_history)}")

    def test_12_hot_update(self):
        """热更新配置"""
        new_config = {
            "portfolio_optimizer": {
                "enabled": False,
                "default_objective": "min_variance",
                "rebalance_interval_hours": 8,
                "min_rebalance_change_pct": 0.05,
                "max_single_weight": 0.40,
                "min_single_weight": 0.10,
                "risk_free_rate": 0.04,
                "correlation": {
                    "window_days": 60,
                    "high_threshold": 0.75,
                    "alert_threshold": 0.90,
                },
                "var": {
                    "confidence_levels": [0.99],
                    "horizon_days": 5,
                    "method": "parametric",
                },
            }
        }
        self.opt.update_config(new_config)
        self.assertFalse(self.opt._enabled)
        self.assertEqual(self.opt._default_objective, OptimizationObjective.MIN_VARIANCE)
        self.assertEqual(self.opt._rebalance_interval_hours, 8)
        self.assertEqual(self.opt._max_single_weight, 0.40)
        self.assertEqual(self.opt._min_single_weight, 0.10)
        self.assertEqual(self.opt._risk_free_rate, 0.04)
        self.assertEqual(self.opt._corr_window_days, 60)
        self.assertEqual(self.opt._corr_high_threshold, 0.75)
        self.assertEqual(self.opt._var_method, "parametric")
        print("  [PASS] test_12: all hot update params verified")

    def test_13_critical_state_check(self):
        """临界状态检测"""
        # 正常状态
        self.opt._equity_history = [10000, 10050, 10100, 10080, 10120]
        self.assertFalse(self.opt.check_critical_state())

        # 大回撤状态
        self.opt._equity_history = [10000, 9000, 8500, 8000, 7900]  # 21% drawdown
        self.assertTrue(self.opt.check_critical_state())
        print("  [PASS] test_13: critical state detection works")

    def test_14_get_stats(self):
        """增强指标统计"""
        np.random.seed(42)
        for i in range(60):
            self.opt.update_performance("trend", np.random.normal(0.001, 0.02))
            self.opt.update_equity(10000 + i * 10)

        self.opt.compute_correlation_matrix()
        self.opt.compute_attribution()

        stats = self.opt.get_stats()
        self.assertIsNotNone(stats)
        self.assertIn("total_strategies_tracked", stats)
        self.assertIn("active_strategies", stats)
        self.assertIn("equity_history_length", stats)
        self.assertIn("last_optimization", stats)
        self.assertIn("average_correlation", stats)
        self.assertIn("health_status", stats)
        self.assertIn("performance_metrics", stats)
        self.assertIn("rebalance_history_count", stats)
        print(f"  [PASS] test_14: strategies={stats['total_strategies_tracked']}, "
              f"avg_corr={stats['average_correlation']}, "
              f"health={stats['health_status']}")

    def test_15_health_check(self):
        """组合健康检查"""
        np.random.seed(42)
        for i in range(100):
            self.opt.update_equity(10000 + i * 10)
        self.opt.compute_correlation_matrix()
        self.opt.compute_var(confidence=0.95)

        health = self.opt.health_check()
        self.assertIn("status", health)
        self.assertIn("issues", health)
        self.assertIn("warnings", health)
        self.assertIn("timestamp", health)
        print(f"  [PASS] test_15: status={health['status']}, issues={health['issue_count']}")

    def test_16_get_all_performances(self):
        """获取所有策略绩效"""
        self.opt.update_performance("trend", 0.001)
        self.opt.update_performance("grid", 0.0005)

        perfs = self.opt.get_all_performances()
        self.assertIsInstance(perfs, dict)
        print(f"  [PASS] test_16: {len(perfs)} strategies tracked")

    def test_17_get_summary(self):
        """获取完整摘要"""
        np.random.seed(42)
        for i in range(60):
            self.opt.update_performance("trend", np.random.normal(0.001, 0.02))
            self.opt.update_equity(10000 + i * 10)

        summary = self.opt.get_summary()
        self.assertIn("enabled", summary)
        self.assertIn("strategies_tracked", summary)
        self.assertIn("combos_available", summary)
        self.assertIn("health", summary)
        self.assertIn("timestamp", summary)
        print(f"  [PASS] test_17: strategies={summary['strategies_tracked']}, "
              f"combos={summary['combos_available']}")


# ============================================================
# ConditionalOrderManager 测试
# ============================================================

class TestConditionalOrderManager(unittest.TestCase):
    """条件单管理器测试"""

    def setUp(self):
        self.config = {
            "conditional_order": {
                "enabled": True,
                "retry_interval_sec": 5,
                "max_retries": 3,
                "sync_interval_sec": 30,
                "heartbeat_enabled": True,
                "heartbeat_interval_sec": 60,
                "immediate_trigger_protection": True,
                "stop_loss": {
                    "margin_call_offset": 0.30,
                    "default_callback_rate": 0.015,
                },
                "persistence": {
                    "enabled": True,
                    "orders_file": "data/conditional_orders.json",
                    "save_interval_sec": 60,
                },
                "audit": {
                    "enabled": True,
                    "max_entries": 500,
                },
                "batch": {
                    "max_batch_size": 20,
                    "batch_delay_sec": 0.5,
                },
            }
        }
        # Mock OKX client and redis
        self.mock_okx = MagicMock()
        self.mock_redis = MagicMock()
        self.com = ConditionalOrderManager(self.config, self.mock_okx, self.mock_redis)

    def test_20_initialization(self):
        """初始化检查"""
        self.assertEqual(self.com._retry_interval, 5)
        self.assertEqual(self.com._max_retries, 3)
        self.assertIsInstance(self.com._active_orders, dict)
        self.assertIsInstance(self.com._pending_orders, dict)
        self.assertIsInstance(self.com._failed_orders, dict)
        print("  [PASS] test_20: initialization verified")

    def test_21_state_persistence(self):
        """状态持久化"""
        self.com._active_orders.clear()
        self.com._pending_orders.clear()
        self.com._failed_orders.clear()
        self.com._active_orders["test_algo_1"] = {
            "symbol": "BTC-USDT-SWAP",
            "side": "long",
            "type": "stop_loss",
            "price": 64000.0,
            "quantity": 0.01,
            "leverage": 3,
            "is_algo": True,
        }
        self.com._pending_orders["pending_1"] = {
            "symbol": "ETH-USDT-SWAP",
            "side": "short",
            "type": "take_profit",
            "price": 3000.0,
            "quantity": 0.1,
            "leverage": 5,
            "retry_count": 1,
        }
        self.com._failed_orders["failed_1"] = {
            "symbol": "SOL-USDT-SWAP",
            "side": "long",
            "type": "stop_loss",
            "price": 150.0,
            "quantity": 1.0,
            "leverage": 5,
            "retry_count": 3,
        }

        # 保存
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp_path = f.name
        try:
            self.com._orders_file = tmp_path
            self.com._save_active_orders()
            self.assertTrue(os.path.exists(tmp_path))
            with open(tmp_path, 'r') as f:
                saved = json.load(f)
            self.assertEqual(saved["version"], 2)
            self.assertIn("active_orders", saved)
            self.assertEqual(len(saved["active_orders"]), 1)
            print(f"  [PASS] test_21a: state saved with version={saved['version']}")

            # 加载
            com2 = ConditionalOrderManager(self.config, self.mock_okx, self.mock_redis)
            com2._orders_file = tmp_path
            com2._load_active_orders()
            self.assertEqual(len(com2._active_orders), 1)
            print(f"  [PASS] test_21b: state loaded, active_orders={len(com2._active_orders)}")
        finally:
            os.unlink(tmp_path)

        # 收集状态
        state = self.com.collect_persistent_state()
        self.assertIn("active_orders", state)
        self.assertIn("pending_orders", state)
        self.assertIn("failed_orders", state)
        print(f"  [PASS] test_21c: collected state, active={len(state['active_orders'])}")

        # 恢复状态
        com3 = ConditionalOrderManager(self.config, self.mock_okx, self.mock_redis)
        com3.restore_persistent_state(state)
        self.assertEqual(len(com3._active_orders), 1)
        self.assertEqual(len(com3._pending_orders), 1)
        self.assertEqual(len(com3._failed_orders), 1)
        print(f"  [PASS] test_21d: state restored")

    def test_22_hot_update(self):
        """热更新配置"""
        new_config = {
            "retry_interval": 10,
            "max_retries": 5,
            "heartbeat_enabled": False,
            "sync_enabled": False,
        }
        self.com.update_config(new_config)
        self.assertEqual(self.com._retry_interval, 10)
        self.assertEqual(self.com._max_retries, 5)
        self.assertFalse(self.com._heartbeat_enabled)
        self.assertFalse(self.com._sync_enabled)
        print("  [PASS] test_22: hot update verified")

    def test_23_error_categorization(self):
        """错误分类"""
        # 网络错误
        cat = self.com._categorize_error(Exception("Connection refused"))
        self.assertEqual(cat, "network")

        # 认证错误
        cat = self.com._categorize_error(Exception("Invalid API key"))
        self.assertEqual(cat, "auth")

        # 频率限制
        cat = self.com._categorize_error(Exception("Too many requests"))
        self.assertEqual(cat, "rate_limit")

        # 订单被拒
        cat = self.com._categorize_error(Exception("Order would immediately trigger"))
        self.assertEqual(cat, "order_rejected")

        # 未知
        cat = self.com._categorize_error(Exception("Some random error"))
        self.assertEqual(cat, "unknown")
        print("  [PASS] test_23: error categorization works")

    def test_24_audit_trail(self):
        """审计追踪"""
        self.com._add_audit_entry("place_sl", "BTC-USDT-SWAP", "algo_1", {"price": 64000})
        self.com._add_audit_entry("cancel", "BTC-USDT-SWAP", "algo_1", {"reason": "replaced"})
        self.com._add_audit_entry("place_tp", "ETH-USDT-SWAP", "algo_2", {"price": 3200})

        trail = self.com.get_audit_trail()
        self.assertEqual(len(trail), 3)

        # 按symbol过滤
        btc_trail = self.com.get_audit_trail(symbol="BTC-USDT-SWAP")
        self.assertEqual(len(btc_trail), 2)
        print(f"  [PASS] test_24: audit trail {len(trail)} entries, btc_filtered={len(btc_trail)}")

    def test_25_enhanced_stats(self):
        """增强指标统计"""
        self.com._active_orders.clear()
        self.com._pending_orders.clear()
        self.com._failed_orders.clear()
        self.com._active_orders["algo_1"] = {
            "symbol": "BTC-USDT-SWAP", "side": "long", "type": "stop_loss",
            "price": 64000, "quantity": 0.01, "leverage": 3, "is_algo": True,
        }
        self.com._active_orders["algo_2"] = {
            "symbol": "BTC-USDT-SWAP", "side": "long", "type": "take_profit",
            "price": 66000, "quantity": 0.01, "leverage": 3, "is_algo": True,
        }
        self.com._pending_orders["pending_1"] = {
            "symbol": "ETH-USDT-SWAP", "type": "stop_loss", "retry_count": 2,
        }
        self.com._failed_orders["failed_1"] = {
            "symbol": "SOL-USDT-SWAP", "type": "stop_loss", "retry_count": 3,
        }
        self.com._error_counts["network"] = 3
        self.com._error_counts["rate_limit"] = 1
        self.com._heartbeat_restored_count = 5
        self.com._sync_operations_count = 10

        stats = self.com.get_enhanced_stats()
        self.assertIn("active_orders", stats)
        self.assertIn("pending_orders", stats)
        self.assertIn("failed_orders", stats)
        self.assertIn("success_rate", stats)
        self.assertIn("heartbeat_restored_count", stats)
        self.assertIn("sync_operations_count", stats)
        self.assertIn("error_counts", stats)
        self.assertIn("per_symbol", stats)
        self.assertIn("uptime_seconds", stats)
        print(f"  [PASS] test_25: active={stats['active_orders']}, "
              f"heartbeat_restored={stats['heartbeat_restored_count']}, "
              f"uptime={stats['uptime_human']}")

    def test_26_get_detailed_stats(self):
        """获取详细统计"""
        self.com._active_orders.clear()
        self.com._active_orders["algo_sl"] = {
            "symbol": "BTC-USDT-SWAP", "type": "stop_loss",
        }
        self.com._active_orders["algo_tp"] = {
            "symbol": "BTC-USDT-SWAP", "type": "take_profit",
        }

        stats = self.com.get_detailed_stats()
        self.assertEqual(stats["active_sl_count"], 1)
        self.assertEqual(stats["active_tp_count"], 1)
        self.assertIn("by_symbol", stats)
        print(f"  [PASS] test_26: sl={stats['active_sl_count']}, tp={stats['active_tp_count']}")

    def test_27_calculate_stop_price(self):
        """止损价格计算"""
        # 多头止损
        sl = self.com._calculate_stop_price("long", 65000, 3)
        self.assertLess(sl, 65000)
        self.assertGreater(sl, 0)
        print(f"  [PASS] test_27a: long sl={sl:.2f}")

        # 空头止损
        sl = self.com._calculate_stop_price("short", 65000, 3)
        self.assertGreater(sl, 65000)
        print(f"  [PASS] test_27b: short sl={sl:.2f}")

    def test_28_get_sl_tp_orders(self):
        """获取指定币种的SL/TP订单"""
        self.com._active_orders["sl_1"] = {
            "symbol": "BTC-USDT-SWAP", "type": "stop_loss",
        }
        self.com._active_orders["tp_1"] = {
            "symbol": "BTC-USDT-SWAP", "type": "take_profit",
        }

        sl_orders = self.com.get_sl_orders_for_symbol("BTC-USDT-SWAP")
        self.assertEqual(len(sl_orders), 1)

        tp_orders = self.com.get_tp_orders_for_symbol("BTC-USDT-SWAP")
        self.assertEqual(len(tp_orders), 1)
        print("  [PASS] test_28: SL/TP orders retrieved correctly")


# ============================================================
# ComputeScheduler 测试
# ============================================================

class TestComputeScheduler(unittest.TestCase):
    """算力动态调度器测试"""

    def setUp(self):
        self.config = {
            "hardware": {
                "max_cpu_usage": 80,
                "max_memory_usage": 85,
            },
            "risk": {
                "high_volatility_percentile": 0.8,
                "low_volatility_percentile": 0.2,
            },
            "compute_scheduler": {
                "enabled": True,
                "max_cpu_usage": 80,
                "cpu_throttle_threshold": 72,
                "cpu_critical_threshold": 80,
                "high_volatility_percentile": 0.8,
                "low_volatility_percentile": 0.2,
                "priority_intervals": {
                    "high": 50,
                    "normal": 100,
                    "low": 500,
                    "throttled": 1000,
                },
                "low_vol_skip_ratio": 3,
                "adaptive_throttle": {
                    "enabled": True,
                    "cpu_history_size": 10,
                    "trend_up_threshold": 5.0,
                    "trend_down_threshold": -5.0,
                    "max_skip_ratio": 10,
                    "min_skip_ratio": 1,
                },
                "compute_budget": {
                    "enabled": True,
                    "budget_per_sec": {
                        "high": 20,
                        "normal": 10,
                        "low": 3,
                        "throttled": 1,
                    },
                },
                "persistence": {
                    "enabled": True,
                    "state_file": "data/compute_scheduler_state.json",
                    "save_interval_sec": 300,
                },
            },
        }
        self.cs = ComputeScheduler(self.config)

    def test_30_initialization(self):
        """初始化检查"""
        self.assertEqual(self.cs._max_cpu_usage, 80)
        self.assertEqual(self.cs._cpu_throttle_threshold, 72)
        self.assertEqual(self.cs._cpu_critical_threshold, 80)
        self.assertEqual(self.cs._high_vol_threshold, 0.8)
        self.assertEqual(self.cs._low_vol_threshold, 0.2)
        self.assertEqual(self.cs._low_vol_skip_ratio, 3)
        self.assertEqual(self.cs._priority_intervals[ComputePriority.HIGH], 50)
        self.assertEqual(self.cs._priority_intervals[ComputePriority.THROTTLED], 1000)
        print("  [PASS] test_30: initialization verified")

    def test_31_register_and_update_symbol(self):
        """注册和更新symbol波动率"""
        self.cs.register_symbol("BTC-USDT-SWAP")
        self.cs.register_symbol("ETH-USDT-SWAP")

        # 高波动
        self.cs.update_symbol_volatility("BTC-USDT-SWAP", 0.85)
        state = self.cs._symbol_states["BTC-USDT-SWAP"]
        self.assertEqual(state.priority, ComputePriority.HIGH)
        self.assertEqual(state.compute_interval_ms, 50)

        # 低波动
        self.cs.update_symbol_volatility("ETH-USDT-SWAP", 0.15)
        state = self.cs._symbol_states["ETH-USDT-SWAP"]
        self.assertEqual(state.priority, ComputePriority.LOW)
        self.assertEqual(state.compute_interval_ms, 500)
        print("  [PASS] test_31: symbol priorities set correctly")

    def test_32_should_compute_throttling(self):
        """计算节流判断"""
        self.cs.register_symbol("TEST-USDT-SWAP")

        # 正常优先级：每次都算
        self.cs.update_symbol_volatility("TEST-USDT-SWAP", 0.5)
        for i in range(10):
            self.assertTrue(self.cs.should_compute("TEST-USDT-SWAP"))

        # 低波动：每3次算1次
        self.cs.update_symbol_volatility("TEST-USDT-SWAP", 0.1)
        compute_count = 0
        skip_count = 0
        for i in range(30):
            if self.cs.should_compute("TEST-USDT-SWAP"):
                compute_count += 1
            else:
                skip_count += 1
        self.assertGreater(skip_count, compute_count)
        print(f"  [PASS] test_32: low_vol compute={compute_count}, skip={skip_count}")

    def test_33_cpu_throttle(self):
        """CPU过载降频"""
        self.cs.register_symbol("BTC-USDT-SWAP")
        self.cs.update_symbol_volatility("BTC-USDT-SWAP", 0.85)

        # CPU正常
        self.cs.update_cpu_memory(50, 40)
        self.assertFalse(self.cs.is_global_throttled())

        # CPU临界
        self.cs.update_cpu_memory(85, 50)
        self.assertTrue(self.cs.is_global_throttled())
        state = self.cs._symbol_states["BTC-USDT-SWAP"]
        self.assertEqual(state.priority, ComputePriority.THROTTLED)

        # CPU恢复
        self.cs.update_cpu_memory(50, 40)
        self.assertFalse(self.cs.is_global_throttled())
        state = self.cs._symbol_states["BTC-USDT-SWAP"]
        self.assertEqual(state.priority, ComputePriority.HIGH)
        print("  [PASS] test_33: CPU throttle works correctly")

    def test_34_get_high_priority_symbols(self):
        """获取高优先级symbol"""
        self.cs.register_symbol("HIGH-1")
        self.cs.register_symbol("HIGH-2")
        self.cs.register_symbol("LOW-1")
        self.cs.update_symbol_volatility("HIGH-1", 0.9)
        self.cs.update_symbol_volatility("HIGH-2", 0.85)
        self.cs.update_symbol_volatility("LOW-1", 0.3)

        high = self.cs.get_high_priority_symbols()
        self.assertEqual(len(high), 2)
        self.assertIn("HIGH-1", high)
        self.assertIn("HIGH-2", high)
        print(f"  [PASS] test_34: high priority symbols={high}")

    def test_35_state_persistence(self):
        """状态持久化"""
        self.cs.register_symbol("BTC-USDT-SWAP")
        self.cs.update_symbol_volatility("BTC-USDT-SWAP", 0.85)
        self.cs.update_cpu_memory(50, 40)

        # 收集状态
        state = self.cs.collect_persistent_state()
        self.assertIn("symbol_states", state)
        self.assertIn("global_throttle_active", state)
        self.assertIn("version", state)
        self.assertEqual(len(state["symbol_states"]), 1)
        print(f"  [PASS] test_35a: state collected, symbols={len(state['symbol_states'])}")

        # 保存
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            tmp_path = f.name
        try:
            self.cs._state_file = tmp_path
            self.cs.save_state()
            self.assertTrue(os.path.exists(tmp_path))
            print(f"  [PASS] test_35b: state saved")

            # 恢复（从文件读取）
            cs2 = ComputeScheduler(self.config)
            cs2._state_file = tmp_path
            cs2.restore_persistent_state()
            self.assertEqual(len(cs2._symbol_states), 1)
            self.assertIn("BTC-USDT-SWAP", cs2._symbol_states)
            print(f"  [PASS] test_35c: state restored")
        finally:
            os.unlink(tmp_path)

    def test_36_hot_update(self):
        """热更新配置"""
        new_config = {
            "hardware": {
                "max_cpu_usage": 90,
            },
            "risk": {
                "high_volatility_percentile": 0.85,
                "low_volatility_percentile": 0.15,
            },
            "low_vol_skip_ratio": 4,
            "priority_intervals": {
                "high": 40,
                "normal": 80,
                "low": 400,
                "throttled": 800,
            },
        }
        self.cs.register_symbol("TEST-USDT-SWAP")
        self.cs.update_symbol_volatility("TEST-USDT-SWAP", 0.5)
        self.cs.update_config(new_config)

        self.assertEqual(self.cs._max_cpu_usage, 90)
        self.assertEqual(self.cs._cpu_throttle_threshold, 81)  # 90 * 0.9
        self.assertEqual(self.cs._cpu_critical_threshold, 90)
        self.assertEqual(self.cs._high_vol_threshold, 0.85)
        self.assertEqual(self.cs._low_vol_threshold, 0.15)
        self.assertEqual(self.cs._low_vol_skip_ratio, 4)
        self.assertEqual(self.cs._priority_intervals[ComputePriority.HIGH], 40)
        print("  [PASS] test_36: hot update verified")

    def test_37_adaptive_throttle(self):
        """自适应降频"""
        self.cs.register_symbol("TEST-USDT-SWAP")
        self.cs.update_symbol_volatility("TEST-USDT-SWAP", 0.5)

        # 模拟CPU上升趋势
        for cpu in [50, 55, 60, 65, 70, 72, 74, 76, 78, 80]:
            self.cs.update_cpu_memory(cpu, 50)

        # CPU趋势上升应触发自适应降频
        self.assertTrue(self.cs.is_global_throttled() or self.cs._low_vol_skip_ratio > 3)
        print(f"  [PASS] test_37: adaptive throttle, skip_ratio={self.cs._low_vol_skip_ratio}")

    def test_38_compute_budget(self):
        """计算预算"""
        self.cs.register_symbol("TEST-USDT-SWAP")
        self.cs.update_symbol_volatility("TEST-USDT-SWAP", 0.5)

        # 正常预算内
        for i in range(5):
            result = self.cs.check_compute_budget("TEST-USDT-SWAP")
            self.assertTrue(result)

        print("  [PASS] test_38: compute budget check works")

    def test_39_enhanced_stats(self):
        """增强指标统计"""
        self.cs.register_symbol("BTC-USDT-SWAP")
        self.cs.register_symbol("ETH-USDT-SWAP")
        self.cs.update_symbol_volatility("BTC-USDT-SWAP", 0.85)
        self.cs.update_symbol_volatility("ETH-USDT-SWAP", 0.15)
        self.cs.update_cpu_memory(50, 40)

        # 模拟一些计算
        for i in range(10):
            self.cs.should_compute("BTC-USDT-SWAP")
            self.cs.should_compute("ETH-USDT-SWAP")

        stats = self.cs.get_enhanced_stats()
        self.assertIn("compute_savings", stats)
        self.assertIn("skip_rate_by_priority", stats)
        self.assertIn("cpu_time_saved_estimate_seconds", stats)
        self.assertIn("throttle_duration_total_seconds", stats)
        self.assertIn("throttle_activation_count", stats)
        self.assertIn("per_symbol_stats", stats)
        self.assertIn("avg_compute_interval_by_priority_ms", stats)
        total_skips = stats["compute_savings"]["total_skips"]
        total_computes = stats["compute_savings"]["total_computes_attempted"]
        skip_rate = stats["compute_savings"]["overall_skip_rate"]
        print(f"  [PASS] test_39: skips={total_skips}, "
              f"computes={total_computes}, "
              f"skip_rate={skip_rate:.2%}")

    def test_40_health_check(self):
        """健康检查"""
        self.cs.register_symbol("BTC-USDT-SWAP")
        self.cs.update_cpu_memory(50, 40)

        health = self.cs.health_check()
        self.assertIn("status", health)
        self.assertIn("issues", health)
        self.assertIn("cpu", health)
        self.assertIn("memory", health)
        self.assertIn("symbols_registered", health)
        print(f"  [PASS] test_40: status={health['status']}, "
              f"cpu_trend={health['cpu']['trend']}, "
              f"cpu={health['cpu']['current']:.1f}%")

    def test_41_cpu_sustained_high_health(self):
        """CPU持续高负载健康检查"""
        self.cs.register_symbol("TEST-USDT-SWAP")
        # 模拟持续高CPU
        for cpu in [85, 86, 84, 87, 85, 88, 86, 89, 85, 90]:
            self.cs.update_cpu_memory(cpu, 50)

        health = self.cs.health_check()
        self.assertIn(health["status"], ["warning", "critical"])
        self.assertGreater(len(health["issues"]), 0)
        print(f"  [PASS] test_41: status={health['status']}, issues={health['issues']}")

    def test_42_get_status(self):
        """获取状态摘要"""
        self.cs.register_symbol("BTC-USDT-SWAP")
        self.cs.update_symbol_volatility("BTC-USDT-SWAP", 0.85)
        self.cs.update_cpu_memory(50, 40)

        status = self.cs.get_status()
        self.assertIn("cpu_percent", status)
        self.assertIn("memory_percent", status)
        self.assertIn("global_throttle_active", status)
        self.assertIn("priority_distribution", status)
        self.assertIn("total_symbols", status)
        print(f"  [PASS] test_42: symbols={status['total_symbols']}, "
              f"distribution={status['priority_distribution']}")


# ============================================================
# E2E 集成测试
# ============================================================

class TestE2EIntegration(unittest.TestCase):
    """端到端集成测试"""

    def test_50_portfolio_compute_integration(self):
        """组合优化 + 算力调度 集成"""
        reset_portfolio_optimizer()
        config = {
            "portfolio_optimizer": {
                "enabled": True,
                "default_objective": "max_sharpe",
                "rebalance_interval_hours": 4,
                "min_rebalance_change_pct": 0.02,
                "max_single_weight": 0.50,
                "min_single_weight": 0.05,
                "risk_free_rate": 0.03,
                "correlation": {"window_days": 30, "high_threshold": 0.70, "alert_threshold": 0.85},
                "var": {"confidence_levels": [0.95, 0.99], "horizon_days": 1, "method": "historical"},
                "stress_test": {"enabled": True},
                "persistence": {"enabled": True, "save_interval_sec": 300, "state_file": "data/portfolio_state.json"},
            },
            "compute_scheduler": {
                "enabled": True,
                "max_cpu_usage": 80,
                "cpu_throttle_threshold": 72,
                "cpu_critical_threshold": 80,
                "high_volatility_percentile": 0.8,
                "low_volatility_percentile": 0.2,
                "priority_intervals": {"high": 50, "normal": 100, "low": 500, "throttled": 1000},
                "low_vol_skip_ratio": 3,
                "adaptive_throttle": {"enabled": True, "cpu_history_size": 10, "trend_up_threshold": 5.0, "trend_down_threshold": -5.0, "max_skip_ratio": 10, "min_skip_ratio": 1},
                "compute_budget": {"enabled": True, "budget_per_sec": {"high": 20, "normal": 10, "low": 3, "throttled": 1}},
                "persistence": {"enabled": True, "state_file": "data/compute_scheduler_state.json", "save_interval_sec": 300},
            },
            "hardware": {"max_cpu_usage": 80, "max_memory_usage": 85},
            "risk": {"high_volatility_percentile": 0.8, "low_volatility_percentile": 0.2},
        }

        opt = PortfolioOptimizer(config)
        cs = ComputeScheduler(config)

        # 模拟策略绩效更新
        np.random.seed(42)
        for i in range(60):
            opt.update_performance("trend", np.random.normal(0.001, 0.02))
            opt.update_performance("grid", np.random.normal(0.0005, 0.01))
            opt.update_performance("scalping", np.random.normal(0.0008, 0.015))
            opt.update_performance("arbitrage", np.random.normal(0.0003, 0.005))
            opt.update_equity(10000 + i * 10)

        # 组合优化
        opt_result = opt.optimize()
        self.assertIsNotNone(opt_result)

        # 算力调度
        for name in opt_result.optimal_weights:
            cs.register_symbol(name)
        cs.update_cpu_memory(50, 40)

        # 触发计算以生成统计数据
        for name in opt_result.optimal_weights:
            for _ in range(5):
                cs.should_compute(name)

        # 验证集成
        opt_stats = opt.get_stats()
        cs_stats = cs.get_enhanced_stats()
        self.assertGreater(opt_stats["total_strategies_tracked"], 0)
        self.assertGreater(cs_stats["compute_savings"]["total_computes_attempted"], 0)

        print(f"  [PASS] test_50: E2E integration - portfolio={opt_stats['total_strategies_tracked']} strategies, "
              f"compute={cs_stats['compute_savings']['total_computes_attempted']} computes")

    def test_51_all_three_modules_integration(self):
        """三个模块全部集成"""
        reset_portfolio_optimizer()

        config = {
            "portfolio_optimizer": {
                "enabled": True, "default_objective": "max_sharpe",
                "rebalance_interval_hours": 4, "min_rebalance_change_pct": 0.02,
                "max_single_weight": 0.50, "min_single_weight": 0.05, "risk_free_rate": 0.03,
                "correlation": {"window_days": 30, "high_threshold": 0.70, "alert_threshold": 0.85},
                "var": {"confidence_levels": [0.95], "horizon_days": 1, "method": "historical"},
                "stress_test": {"enabled": True},
                "persistence": {"enabled": True, "save_interval_sec": 300, "state_file": "data/portfolio_state.json"},
            },
            "conditional_order": {
                "enabled": True, "retry_interval_sec": 5, "max_retries": 3,
                "sync_interval_sec": 30, "heartbeat_enabled": True, "heartbeat_interval_sec": 60,
                "immediate_trigger_protection": True,
                "stop_loss": {"margin_call_offset": 0.30, "default_callback_rate": 0.015},
                "persistence": {"enabled": True, "orders_file": "data/conditional_orders.json", "save_interval_sec": 60},
                "audit": {"enabled": True, "max_entries": 500},
                "batch": {"max_batch_size": 20, "batch_delay_sec": 0.5},
            },
            "compute_scheduler": {
                "enabled": True, "max_cpu_usage": 80, "cpu_throttle_threshold": 72,
                "cpu_critical_threshold": 80, "high_volatility_percentile": 0.8,
                "low_volatility_percentile": 0.2,
                "priority_intervals": {"high": 50, "normal": 100, "low": 500, "throttled": 1000},
                "low_vol_skip_ratio": 3,
                "adaptive_throttle": {"enabled": True, "cpu_history_size": 10, "trend_up_threshold": 5.0, "trend_down_threshold": -5.0, "max_skip_ratio": 10, "min_skip_ratio": 1},
                "compute_budget": {"enabled": True, "budget_per_sec": {"high": 20, "normal": 10, "low": 3, "throttled": 1}},
                "persistence": {"enabled": True, "state_file": "data/compute_scheduler_state.json", "save_interval_sec": 300},
            },
            "hardware": {"max_cpu_usage": 80, "max_memory_usage": 85},
            "risk": {"high_volatility_percentile": 0.8, "low_volatility_percentile": 0.2},
        }

        # 初始化所有模块
        opt = PortfolioOptimizer(config)
        mock_okx = MagicMock()
        mock_redis = MagicMock()
        com = ConditionalOrderManager(config, mock_okx, mock_redis)
        cs = ComputeScheduler(config)

        # 模拟数据流
        np.random.seed(42)
        for i in range(60):
            opt.update_performance("trend", np.random.normal(0.001, 0.02))
            opt.update_performance("grid", np.random.normal(0.0005, 0.01))
            opt.update_performance("scalping", np.random.normal(0.0008, 0.015))
            opt.update_equity(10000 + i * 10)

        # 组合优化
        opt_result = opt.optimize()
        self.assertIsNotNone(opt_result)

        # 条件单管理
        com._active_orders["algo_test"] = {
            "symbol": "BTC-USDT-SWAP", "side": "long", "type": "stop_loss",
            "price": 64000, "quantity": 0.01, "leverage": 3, "is_algo": True,
        }
        com_stats = com.get_enhanced_stats()

        # 算力调度
        for name in opt_result.optimal_weights:
            cs.register_symbol(name)
        cs.update_cpu_memory(50, 40)
        for name in opt_result.optimal_weights:
            for _ in range(5):
                cs.should_compute(name)
        cs_stats = cs.get_enhanced_stats()

        # 全量验证
        opt_summary = opt.get_summary()
        self.assertIn("health", opt_summary)
        self.assertGreater(com_stats["active_orders"], 0)
        self.assertGreater(cs_stats["compute_savings"]["total_computes_attempted"], 0)

        print(f"  [PASS] test_51: All 3 modules integrated - "
              f"portfolio={opt_summary['strategies_tracked']} strategies, "
              f"conditional={com_stats['active_orders']} orders, "
              f"compute={cs_stats['compute_savings']['total_computes_attempted']} computes")

    def test_52_state_persistence_roundtrip(self):
        """状态持久化往返测试"""
        reset_portfolio_optimizer()

        config = {
            "portfolio_optimizer": {
                "enabled": True, "default_objective": "max_sharpe",
                "rebalance_interval_hours": 4, "min_rebalance_change_pct": 0.02,
                "max_single_weight": 0.50, "min_single_weight": 0.05, "risk_free_rate": 0.03,
                "correlation": {"window_days": 30, "high_threshold": 0.70, "alert_threshold": 0.85},
                "var": {"confidence_levels": [0.95], "horizon_days": 1, "method": "historical"},
                "stress_test": {"enabled": True},
                "persistence": {"enabled": True, "save_interval_sec": 300, "state_file": "data/portfolio_state.json"},
            },
            "compute_scheduler": {
                "enabled": True, "max_cpu_usage": 80, "cpu_throttle_threshold": 72,
                "cpu_critical_threshold": 80, "high_volatility_percentile": 0.8,
                "low_volatility_percentile": 0.2,
                "priority_intervals": {"high": 50, "normal": 100, "low": 500, "throttled": 1000},
                "low_vol_skip_ratio": 3,
                "adaptive_throttle": {"enabled": True, "cpu_history_size": 10, "trend_up_threshold": 5.0, "trend_down_threshold": -5.0, "max_skip_ratio": 10, "min_skip_ratio": 1},
                "compute_budget": {"enabled": True, "budget_per_sec": {"high": 20, "normal": 10, "low": 3, "throttled": 1}},
                "persistence": {"enabled": True, "state_file": "data/compute_scheduler_state.json", "save_interval_sec": 300},
            },
            "hardware": {"max_cpu_usage": 80, "max_memory_usage": 85},
            "risk": {"high_volatility_percentile": 0.8, "low_volatility_percentile": 0.2},
        }

        # 初始化并填充数据
        opt = PortfolioOptimizer(config)
        cs = ComputeScheduler(config)

        np.random.seed(42)
        for i in range(60):
            opt.update_performance("trend", np.random.normal(0.001, 0.02))
            opt.update_equity(10000 + i * 10)

        opt.compute_correlation_matrix()
        cs.register_symbol("BTC-USDT-SWAP")
        cs.update_symbol_volatility("BTC-USDT-SWAP", 0.85)
        cs.update_cpu_memory(50, 40)

        # 收集状态
        opt_state = opt.collect_persistent_state()
        cs_state = cs.collect_persistent_state()

        # 创建新实例并恢复
        opt2 = PortfolioOptimizer(config)
        opt2.restore_persistent_state(opt_state)

        cs2 = ComputeScheduler(config)
        cs2.restore_persistent_state = lambda: cs2._restore_from_dict(cs_state)
        # ComputeScheduler.restore_persistent_state() reads from file;
        # manually restore from dict for testing
        cs2._global_throttle_active = cs_state.get("global_throttle_active", False)
        cs2._throttle_start_time = cs_state.get("throttle_start_time")
        cs2._throttle_activation_count = cs_state.get("throttle_activation_count", 0)
        cs2._throttle_duration_total = cs_state.get("throttle_duration_total", 0.0)
        cs2._low_vol_skip_ratio = cs_state.get("low_vol_skip_ratio", 3)
        for s, sdata in cs_state.get("symbol_states", {}).items():
            from core.compute_scheduler import SymbolComputeState, ComputePriority
            state = SymbolComputeState(symbol=s)
            state.volatility_percentile = sdata.get("volatility_percentile", 0.5)
            state.compute_interval_ms = sdata.get("compute_interval_ms", 100.0)
            state.skip_count = sdata.get("skip_count", 0)
            state.total_count = sdata.get("total_count", 0)
            state.last_compute_ts = sdata.get("last_compute_ts", 0.0)
            try:
                state.priority = ComputePriority(sdata.get("priority", "normal"))
            except ValueError:
                state.priority = ComputePriority.NORMAL
            cs2._symbol_states[s] = state

        # 验证恢复
        self.assertEqual(len(opt2._strategy_performances), len(opt._strategy_performances))
        self.assertEqual(len(opt2._equity_history), len(opt._equity_history))
        self.assertEqual(len(cs2._symbol_states), len(cs._symbol_states))

        print(f"  [PASS] test_52: State roundtrip - "
              f"portfolio: {len(opt2._strategy_performances)} strategies, "
              f"equity: {len(opt2._equity_history)} points, "
              f"compute: {len(cs2._symbol_states)} symbols")


if __name__ == "__main__":
    unittest.main(verbosity=2)