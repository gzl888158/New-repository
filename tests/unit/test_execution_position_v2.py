"""
执行引擎与仓位管理系统 v2.0 集成测试
===================================
测试覆盖：
- 执行引擎：批量操作、执行质量报告、幂等键、指数退避、部分成交
- 仓位管理：跨策略冲突检测、集中度分析、最优减仓顺序
- 端到端集成：执行引擎 + 仓位管理联动
"""

import sys
import os
import unittest
import time
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from execution.order_executor import (
    OrderExecutor, RetryableError, PartialFillAction,
)
from core.position_manager import (
    PositionManager, PositionSnapshot, PositionSide,
    PositionStatus, AccountRiskSnapshot, RiskLevel, RiskEvent,
)


# ═══════════════════════════════════════════════════════════════
# Mock 工厂
# ═══════════════════════════════════════════════════════════════

def make_mock_config():
    return {
        "execution": {
            "retry_attempts": 3,
            "slippage_tolerance": 0.001,
            "rate_limit": {"rest_requests_per_second": 5},
            "reconciliation_interval": 10,
            "idempotency_enabled": True,
            "idempotency_prefix": "okxqt",
            "clordid_cache_size": 1000,
            "clordid_cache_ttl_sec": 3600,
            "retry_base_delay_sec": 1.0,
            "retry_max_delay_sec": 60.0,
            "retry_jitter_pct": 0.3,
            "retry_multiplier": 2.0,
            "partial_fill_enabled": True,
            "partial_fill_action": "cancel_remaining",
            "partial_fill_min_fill_ratio": 0.5,
            "partial_fill_max_wait_sec": 10.0,
            "no_trade_hours": [21, 22],
        },
        "trading": {
            "min_hold_minutes": 5,
            "max_concurrent_positions": 4,
            "taker_fee_rate": 0.0005,
            "min_notional_usd": 15.0,
            "min_margin_per_trade": 0.5,
            "min_profit_cost_ratio": 2.0,
        },
        "currencies": {
            "tier1_settings": {"slippage": 0.001, "grid_spacing_min": 0.006},
            "tier1_symbols": ["BTC", "ETH"],
            "tier2_settings": {"slippage": 0.0015, "grid_spacing_min": 0.006},
            "tier2_symbols": ["SOL", "XRP"],
            "tier3_settings": {"slippage": 0.002, "grid_spacing_min": 0.006},
            "tier3_symbols": ["DOT", "LINK"],
        },
        "position_manager": {
            "enabled": True,
            "sync_interval_sec": 3,
            "risk_check_interval_sec": 5,
            "max_total_positions": 6,
            "max_positions_per_symbol": 2,
            "max_positions_per_strategy": 3,
            "risk_linkage": {
                "drawdown_trigger": 0.15,
                "drawdown_reduce_pct": 0.3,
                "margin_ratio_warning": 0.5,
                "margin_ratio_critical": 0.3,
                "position_loss_limit": 0.1,
            },
            "dynamic_adjust": {
                "enabled": True,
                "rebalance_interval": 300,
                "max_adjust_per_iter": 0.1,
                "min_position_value": 5.0,
            },
            "persistence": {
                "enabled": True,
                "save_interval": 60,
                "max_snapshots": 100,
            },
        },
        "system": {"data_dir": "data"},
        "strategies": {},
    }


class MockOKXClient:
    """Mock OKX客户端"""
    def __init__(self):
        self._positions = []
        self._orders = {}
        self._account_info = {
            "totalEq": "1000.0",
            "availBal": "800.0",
            "totalMgn": "200.0",
            "mgnRatio": "0.2",
            "upl": "50.0",
            "details": [{"ccy": "USDT", "eq": "1000.0", "availBal": "800.0"}],
        }
    
    def get_account_info(self):
        return self._account_info
    
    def get_positions(self):
        return self._positions
    
    def get_ticker(self, symbol):
        return {"last": "65000.0"}
    
    def place_order(self, **kwargs):
        return {"ordId": f"test_ord_{int(time.time()*1000)}", "sCode": "0"}
    
    def cancel_order(self, symbol, order_id):
        return {"sCode": "0"}
    
    def get_order(self, symbol, order_id):
        return {"state": "filled", "avgPx": "65000.0", "fillSz": "0.01", "fee": "0.0"}
    
    def set_leverage(self, symbol, leverage, pos_side=None):
        return True
    
    def round_quantity_to_lot(self, symbol, qty, round_up=False):
        return qty
    
    def contracts_to_coins(self, symbol, contracts):
        return contracts * 0.01
    
    def get_instrument_info(self, symbol):
        return {"ctVal": "0.01"}
    
    def get_atr(self, symbol):
        return 100.0
    
    def get_bills(self, symbol, bill_type="", limit=5):
        return []
    
    def transfer_to_trading(self, ccy, amount):
        return True
    
    def _parse_position(self, pos_data):
        """位置数据解析"""
        class ParsedPosition:
            pass
        p = ParsedPosition()
        p.symbol = pos_data.get("instId", "")
        p.side = pos_data.get("posSide", "long")
        p.quantity = float(pos_data.get("pos", 0))
        p.avg_cost = float(pos_data.get("avgPx", 0))
        p.mark_price = float(pos_data.get("markPx", 0))
        p.unrealized_pnl = float(pos_data.get("upl", 0))
        p.margin = float(pos_data.get("margin", 0))
        p.leverage = int(float(pos_data.get("lever", 1)))
        p.liq_price = float(pos_data.get("liqPx", 0))
        p.margin_ratio = float(pos_data.get("mgnRatio", 0))
        return p


# ═══════════════════════════════════════════════════════════════
# 订单执行引擎 v2.0 测试
# ═══════════════════════════════════════════════════════════════

class TestExecutionEngineV2(unittest.TestCase):
    """执行引擎 v2.0 增强功能测试"""
    
    def setUp(self):
        self.config = make_mock_config()
        self.mock_client = MockOKXClient()
        self.executor = OrderExecutor(
            self.config, self.mock_client,
            redis_cache=MagicMock(),
            sqlite_storage=MagicMock(),
        )
    
    def test_01_idempotency_key_generation(self):
        """幂等键生成"""
        key = self.executor._generate_idempotency_key(
            "BTC-USDT", "grid", "open_long"
        )
        self.assertIsInstance(key, str)
        self.assertLessEqual(len(key), 32)
        # clOrdId 不允许特殊字符（含下划线），前缀默认为 'okxqt'（无下划线）
        self.assertTrue(key.startswith(self.executor._idempotency_prefix))
        self.assertIn("grid", key)
        self.assertIn("BTC", key)
        print(f"  [PASS] test_01: idempotency key: {key}")
    
    def test_02_idempotency_check(self):
        """幂等键重复检测"""
        key = self.executor._generate_idempotency_key(
            "ETH-USDT", "trend", "open_long"
        )
        # 首次检查
        self.assertIsNone(self.executor._check_idempotency(key))
        # 记录key
        self.executor._record_idempotency_key(key, "ord_12345")
        # 第二次检查
        existing = self.executor._check_idempotency(key)
        self.assertEqual(existing, "ord_12345")
        print(f"  [PASS] test_02: idempotency check works, hit={existing}")
    
    def test_03_idempotency_stats(self):
        """幂等键统计"""
        key = self.executor._generate_idempotency_key(
            "BTC-USDT", "grid", "open_long"
        )
        self.executor._record_idempotency_key(key, "ord_001")
        _ = self.executor._check_idempotency(key)
        
        stats = self.executor.get_execution_stats()
        self.assertEqual(stats["idempotency_hits"], 1)
        self.assertEqual(stats["clordid_cache_size"], 1)
        print(f"  [PASS] test_03: idempotency stats: {stats['idempotency_hits']} hits")
    
    def test_04_clordid_cache_cleanup(self):
        """幂等键缓存清理"""
        # 添加过期key
        old_key = "okxqt_grid_BTC_1000000_abcd"
        self.executor._sent_clordids[old_key] = (time.time() - 7200, "old_ord")
        # 添加新key
        new_key = self.executor._generate_idempotency_key(
            "BTC-USDT", "grid", "open_long"
        )
        self.executor._record_idempotency_key(new_key, "new_ord")
        
        self.executor._cleanup_clordid_cache()
        self.assertNotIn(old_key, self.executor._sent_clordids)
        self.assertIn(new_key, self.executor._sent_clordids)
        print(f"  [PASS] test_04: clordid cache cleanup works")
    
    def test_05_backoff_delays(self):
        """指数退避延迟计算"""
        d1 = self.executor._calculate_backoff_delay(1)
        d3 = self.executor._calculate_backoff_delay(3)
        d10 = self.executor._calculate_backoff_delay(10)
        
        self.assertGreater(d1, 0)
        self.assertLess(d3, d10)
        self.assertLessEqual(d10, 60.0)
        print(f"  [PASS] test_05: backoff delays: 1st={d1:.1f}s, 3rd={d3:.1f}s, 10th={d10:.1f}s")
    
    def test_06_rate_limit_backoff(self):
        """限流错误退避（2倍基础延迟）"""
        d_normal = self.executor._calculate_backoff_delay(1)
        d_rate = self.executor._calculate_backoff_delay(1, RetryableError.RATE_LIMIT)
        self.assertGreater(d_rate, d_normal)
        print(f"  [PASS] test_06: rate_limit backoff ({d_rate:.1f}s) > normal ({d_normal:.1f}s)")
    
    def test_07_network_backoff(self):
        """网络错误退避（0.5倍基础延迟）"""
        d_normal = self.executor._calculate_backoff_delay(1)
        d_net = self.executor._calculate_backoff_delay(1, RetryableError.NETWORK_ERROR)
        self.assertLess(d_net, d_normal)
        print(f"  [PASS] test_07: network backoff ({d_net:.1f}s) < normal ({d_normal:.1f}s)")
    
    def test_08_error_classification(self):
        """错误分类"""
        self.assertEqual(
            self.executor._classify_error("429", ""),
            RetryableError.RATE_LIMIT
        )
        self.assertEqual(
            self.executor._classify_error("500", ""),
            RetryableError.SERVER_ERROR
        )
        self.assertEqual(
            self.executor._classify_error("", "timeout"),
            RetryableError.NETWORK_ERROR
        )
        self.assertEqual(
            self.executor._classify_error("51000", ""),
            RetryableError.RATE_LIMIT
        )
        self.assertIsNone(
            self.executor._classify_error("51169", "")
        )
        print(f"  [PASS] test_08: error classification works")
    
    def test_09_partial_fill_tracker(self):
        """部分成交跟踪"""
        order_info = {
            "symbol": "BTC-USDT",
            "clOrdId": "test_clordid_001",
            "strategy_name": "grid",
            "signal_type": "open_long",
            "direction": "long",
        }
        
        self.executor._partial_fill_tracker["test_clordid_001"] = {
            "symbol": "BTC-USDT",
            "total_qty": 0.1,
            "filled_qty": 0.03,
            "remaining_qty": 0.07,
            "filled_price": 65000.0,
            "action": PartialFillAction.CANCEL_REMAINING,
            "start_time": time.time(),
            "strategy_name": "grid",
        }
        
        self.assertEqual(len(self.executor._partial_fill_tracker), 1)
        self.assertEqual(
            self.executor._partial_fill_tracker["test_clordid_001"]["filled_qty"],
            0.03
        )
        print(f"  [PASS] test_09: partial fill tracker works (fill_ratio=30%)")
    
    def test_10_partial_fill_enum(self):
        """部分成交处理策略枚举"""
        self.assertEqual(PartialFillAction.CANCEL_REMAINING.value, "cancel_remaining")
        self.assertEqual(PartialFillAction.RESUBMIT_REMAINING.value, "resubmit")
        self.assertEqual(PartialFillAction.WAIT.value, "wait")
        self.assertEqual(PartialFillAction.MARKET_CLOSE.value, "market_close")
        print(f"  [PASS] test_10: enum values correct")
    
    def test_11_execution_stats(self):
        """执行统计"""
        stats = self.executor.get_execution_stats()
        self.assertEqual(stats["total_orders"], 0)
        self.assertEqual(stats["active_orders"], 0)
        self.assertEqual(stats["partial_fills_pending"], 0)
        self.assertIn("timestamp", stats)
        print(f"  [PASS] test_11: execution stats: {stats['total_orders']} orders")
    
    def test_12_execution_quality_report(self):
        """执行质量报告"""
        report = self.executor.get_execution_quality_report()
        self.assertIn("summary", report)
        self.assertIn("active", report)
        self.assertIn("quality_score", report)
        self.assertEqual(report["quality_score"], 1.0)  # 无订单 = 满分
        print(f"  [PASS] test_12: quality report score={report['quality_score']}")
    
    def test_13_quality_score_with_retries(self):
        """质量评分 - 有重试"""
        self.executor._execution_stats["total_orders"] = 10
        self.executor._execution_stats["retry_count"] = 3
        
        score = self.executor._calculate_execution_quality_score(
            self.executor.get_execution_stats()
        )
        self.assertLess(score, 1.0)
        print(f"  [PASS] test_13: quality score with retries={score:.2f}")
    
    def test_14_quality_score_with_duplicates(self):
        """质量评分 - 有重复"""
        self.executor._execution_stats["total_orders"] = 10
        self.executor._execution_stats["duplicate_prevented"] = 2
        
        score = self.executor._calculate_execution_quality_score(
            self.executor.get_execution_stats()
        )
        self.assertLess(score, 1.0)
        print(f"  [PASS] test_14: quality score with duplicates={score:.2f}")
    
    def test_15_symbol_fail_state(self):
        """Symbol退避状态"""
        self.assertFalse(self.executor._is_symbol_blocked("BTC-USDT"))
        
        self.executor._record_symbol_fail("BTC-USDT", "test_error")
        self.executor._record_symbol_fail("BTC-USDT", "test_error")
        self.executor._record_symbol_fail("BTC-USDT", "test_error")
        
        self.assertTrue(self.executor._is_symbol_blocked("BTC-USDT"))
        
        self.executor._reset_symbol_fail("BTC-USDT")
        self.assertFalse(self.executor._is_symbol_blocked("BTC-USDT"))
        print(f"  [PASS] test_15: symbol fail state: block/unblock works")


# ═══════════════════════════════════════════════════════════════
# 仓位管理系统 v2.0 测试
# ═══════════════════════════════════════════════════════════════

class TestPositionManagerV2(unittest.TestCase):
    """仓位管理系统 v2.0 增强功能测试"""
    
    def setUp(self):
        self.config = make_mock_config()
        self.mock_client = MockOKXClient()
        self.pm = PositionManager(
            self.config, self.mock_client,
            sqlite_storage=MagicMock(),
            redis_cache=MagicMock(),
        )
    
    def _add_position(self, symbol, side, qty, avg_cost, mark_price, margin,
                      strategy="", unrealized_pnl=0.0, leverage=1,
                      liq_price=0.0, margin_ratio=0.0, health_score=1.0):
        """添加持仓到管理器"""
        key = f"{symbol}:{side}"
        pos = PositionSnapshot(
            symbol=symbol,
            side=PositionSide(side),
            quantity=qty,
            avg_cost=avg_cost,
            mark_price=mark_price,
            unrealized_pnl=unrealized_pnl,
            margin=margin,
            leverage=leverage,
            liquidation_price=liq_price,
            margin_ratio=margin_ratio,
            strategy_name=strategy,
            entry_time=time.time(),
            health_score=health_score,
        )
        self.pm._positions[key] = pos
        if strategy:
            self.pm._positions_by_strategy[strategy].append(key)
        self.pm._positions_by_symbol[symbol].append(key)
    
    def test_21_positions_by_symbol(self):
        """按币种查询持仓"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3010, 150, "grid")
        
        btc_positions = self.pm.get_positions_by_symbol("BTC-USDT")
        self.assertEqual(len(btc_positions), 1)
        self.assertEqual(btc_positions[0].symbol, "BTC-USDT")
        print(f"  [PASS] test_21: positions by symbol: BTC-USDT -> {len(btc_positions)}")
    
    def test_22_positions_by_strategy(self):
        """按策略查询持仓"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3010, 150, "grid")
        self._add_position("SOL-USDT", "long", 1.0, 100, 102, 60, "grid")
        
        grid_positions = self.pm.get_positions_by_strategy("grid")
        self.assertEqual(len(grid_positions), 2)
        symbols = [p.symbol for p in grid_positions]
        self.assertIn("ETH-USDT", symbols)
        self.assertIn("SOL-USDT", symbols)
        print(f"  [PASS] test_22: positions by strategy: grid -> {len(grid_positions)}")
    
    def test_23_stats(self):
        """仓位统计"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, 
                          "trend", unrealized_pnl=100.0, health_score=0.8)
        self._add_position("ETH-USDT", "short", 0.1, 3000, 2900, 150,
                          "grid", unrealized_pnl=-50.0, health_score=0.3)
        
        stats = self.pm.get_stats()
        self.assertEqual(stats["total_positions"], 2)
        self.assertEqual(stats["health"]["healthy"], 1)
        self.assertEqual(stats["health"]["critical"], 1)
        print(f"  [PASS] test_23: stats: {stats['total_positions']} positions, "
              f"healthy={stats['health']['healthy']}, critical={stats['health']['critical']}")
    
    def test_24_strategy_set(self):
        """设置策略关联"""
        self._add_position("SOL-USDT", "long", 1.0, 100, 102, 60)
        self.pm.set_position_strategy("SOL-USDT", "long", "scalping")
        
        pos = self.pm.get_position("SOL-USDT", "long")
        self.assertEqual(pos.strategy_name, "scalping")
        print(f"  [PASS] test_24: strategy set: SOL-USDT -> scalping")
    
    def test_25_account_risk(self):
        """账户风险快照"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self.pm._account_risk = AccountRiskSnapshot(
            total_equity=1000.0,
            available_balance=800.0,
            used_margin=200.0,
            margin_ratio=0.2,
            unrealized_pnl=50.0,
            drawdown_pct=0.05,
            risk_level=RiskLevel.NORMAL,
            position_count=1,
        )
        
        risk = self.pm.get_account_risk()
        self.assertEqual(risk.total_equity, 1000.0)
        self.assertEqual(risk.margin_ratio, 0.2)
        print(f"  [PASS] test_25: account risk: equity={risk.total_equity}, "
              f"margin_ratio={risk.margin_ratio}")
    
    def test_26_position_health_score(self):
        """持仓健康评分"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200,
                          "trend", unrealized_pnl=100.0, margin_ratio=0.1)
        pos = self.pm.get_position("BTC-USDT", "long")
        health = self.pm._calculate_position_health(pos)
        self.assertGreaterEqual(health, 0.7)
        print(f"  [PASS] test_26: position health score: {health:.2f} (profitable)")
    
    def test_27_losing_position_health(self):
        """亏损持仓健康评分"""
        self._add_position("ETH-USDT", "short", 0.1, 3000, 3300, 150,
                          "grid", unrealized_pnl=-300.0, margin_ratio=0.6,
                          liq_price=3500.0)
        pos = self.pm.get_position("ETH-USDT", "short")
        health = self.pm._calculate_position_health(pos)
        self.assertLess(health, 0.5)
        print(f"  [PASS] test_27: losing position health: {health:.2f}")
    
    def test_28_risk_event_callback(self):
        """风险事件回调"""
        events = []
        self.pm.on_risk_event(lambda e: events.append(e))
        
        event = RiskEvent(
            event_type="drawdown",
            severity=RiskLevel.HIGH,
            message="Test drawdown event",
            value=0.15,
            threshold=0.15,
        )
        self.pm._notify_risk_event(event)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, "drawdown")
        print(f"  [PASS] test_28: risk event callback: {len(events)} events")
    
    def test_29_total_exposure(self):
        """总风险敞口"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3010, 150, "grid")
        
        exposure = self.pm.get_total_exposure()
        expected = 0.01 * 65100 + 0.1 * 3010
        self.assertAlmostEqual(exposure, expected, delta=1.0)
        print(f"  [PASS] test_29: total exposure={exposure:.2f}")
    
    def test_30_position_removal(self):
        """持仓移除"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self.assertEqual(self.pm.get_position_count(), 1)
        
        self.pm.remove_position("BTC-USDT", "long")
        self.assertEqual(self.pm.get_position_count(), 0)
        print(f"  [PASS] test_30: position removal: {self.pm.get_position_count()} positions")
    
    def test_31_cross_strategy_conflicts(self):
        """跨策略冲突检测 - 同币种多策略"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self._add_position("BTC-USDT", "long", 0.02, 64500, 65100, 400, "grid")
        
        conflicts = self.pm.detect_cross_strategy_conflicts()
        self.assertGreaterEqual(len(conflicts), 1)
        dup_conflict = [c for c in conflicts if c["type"] == "same_direction_duplicate"]
        self.assertEqual(len(dup_conflict), 1)
        self.assertEqual(dup_conflict[0]["symbol"], "BTC-USDT")
        self.assertEqual(dup_conflict[0]["position_count"], 2)
        print(f"  [PASS] test_31: cross-strategy conflicts: {len(conflicts)} found")
    
    def test_32_hedging_conflict(self):
        """跨策略冲突检测 - 反向对冲"""
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3010, 150, "trend")
        self._add_position("ETH-USDT", "short", 0.05, 3050, 3010, 80, "grid")
        
        conflicts = self.pm.detect_cross_strategy_conflicts()
        hedge = [c for c in conflicts if c["type"] == "hedging_conflict"]
        self.assertEqual(len(hedge), 1)
        self.assertEqual(hedge[0]["severity"], "critical")
        self.assertIn("trend", hedge[0]["long_strategies"])
        self.assertIn("grid", hedge[0]["short_strategies"])
        print(f"  [PASS] test_32: hedging conflict detected: {hedge[0]['severity']}")
    
    def test_33_strategy_limit_conflict(self):
        """跨策略冲突检测 - 单策略超限"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "grid")
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3010, 150, "grid")
        self._add_position("SOL-USDT", "long", 1.0, 100, 102, 60, "grid")
        self._add_position("XRP-USDT", "long", 100, 0.5, 0.51, 30, "grid")
        
        conflicts = self.pm.detect_cross_strategy_conflicts()
        limit = [c for c in conflicts if c["type"] == "strategy_limit_exceeded"]
        self.assertEqual(len(limit), 1)
        self.assertEqual(limit[0]["strategy"], "grid")
        self.assertEqual(limit[0]["position_count"], 4)
        print(f"  [PASS] test_33: strategy limit exceeded: grid has {limit[0]['position_count']} positions")
    
    def test_34_no_conflicts(self):
        """跨策略冲突检测 - 无冲突"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3010, 150, "grid")
        
        conflicts = self.pm.detect_cross_strategy_conflicts()
        self.assertEqual(len(conflicts), 0)
        print(f"  [PASS] test_34: no conflicts: {len(conflicts)} conflicts")
    
    def test_35_position_concentration(self):
        """持仓集中度分析"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3010, 150, "grid")
        
        conc = self.pm.get_position_concentration()
        self.assertIn("concentration", conc)
        self.assertIn("hhi", conc)
        self.assertIn("top_symbols", conc)
        self.assertIn("risk_level", conc)
        self.assertEqual(conc["symbol_count"], 2)
        print(f"  [PASS] test_35: concentration: hhi={conc['hhi']:.4f}, "
              f"risk_level={conc['risk_level']}, symbols={conc['symbol_count']}")
    
    def test_36_concentration_high_hhi(self):
        """持仓集中度 - 高集中度"""
        self._add_position("BTC-USDT", "long", 0.1, 65000, 65100, 5000, "trend")
        self._add_position("ETH-USDT", "long", 0.01, 3000, 3010, 15, "grid")
        
        conc = self.pm.get_position_concentration()
        self.assertGreater(conc["hhi"], 0.5)
        self.assertEqual(conc["risk_level"], "critical")
        print(f"  [PASS] test_36: high concentration: hhi={conc['hhi']:.4f}, "
              f"risk_level={conc['risk_level']}")
    
    def test_37_optimal_close_order(self):
        """最优减仓顺序"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 60000, 200,
                          "trend", unrealized_pnl=-500.0, health_score=0.3)
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3100, 150,
                          "grid", unrealized_pnl=100.0, health_score=0.9)
        self._add_position("SOL-USDT", "short", 1.0, 100, 110, 60,
                          "scalping", unrealized_pnl=-100.0, health_score=0.5)
        
        self.pm._account_risk.total_equity = 1000.0
        
        order = self.pm.get_optimal_position_close_order(500.0)
        self.assertGreater(len(order), 0)
        # BTC亏损最大，应排第一
        self.assertEqual(order[0]["symbol"], "BTC-USDT")
        self.assertGreaterEqual(order[0]["priority"], order[-1]["priority"])
        print(f"  [PASS] test_37: optimal close order: {len(order)} positions, "
              f"first={order[0]['symbol']} priority={order[0]['priority']:.4f}")
    
    def test_38_optimal_close_empty(self):
        """最优减仓顺序 - 空持仓"""
        order = self.pm.get_optimal_position_close_order(500.0)
        self.assertEqual(len(order), 0)
        print(f"  [PASS] test_38: optimal close order empty: {len(order)} positions")
    
    def test_39_ws_position_update(self):
        """WebSocket持仓更新"""
        ws_data = [{
            "instId": "SOL-USDT",
            "posSide": "long",
            "pos": "10",
            "avgPx": "100.0",
            "markPx": "102.0",
            "upl": "20.0",
            "margin": "60.0",
            "lever": "5",
            "liqPx": "80.0",
            "mgnRatio": "0.15",
        }]
        
        self.pm.update_position_from_ws(ws_data)
        self.assertEqual(self.pm.get_position_count(), 1)
        
        pos = self.pm.get_position("SOL-USDT", "long")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.quantity, 10.0)
        print(f"  [PASS] test_39: WS position update: SOL-USDT long x{pos.quantity}")
    
    def test_40_ws_position_removal(self):
        """WebSocket持仓移除"""
        self._add_position("SOL-USDT", "long", 1.0, 100, 102, 60, "scalping")
        
        ws_data = [{
            "instId": "SOL-USDT",
            "posSide": "long",
            "pos": "0",
        }]
        
        self.pm.update_position_from_ws(ws_data)
        self.assertEqual(self.pm.get_position_count(), 0)
        print(f"  [PASS] test_40: WS position removal works")
    
    def test_41_save_restore(self):
        """持久化保存/恢复"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self.pm._peak_equity = 1000.0
        self.pm._account_risk = AccountRiskSnapshot(
            total_equity=1000.0, available_balance=800.0,
            used_margin=200.0, margin_ratio=0.2, unrealized_pnl=50.0,
        )
        
        self.pm._save_state()
        
        # 创建新管理器恢复
        pm2 = PositionManager(
            self.config, self.mock_client,
            sqlite_storage=MagicMock(),
            redis_cache=MagicMock(),
        )
        pm2._restore_state()
        
        self.assertEqual(pm2.get_position_count(), 1)
        self.assertEqual(pm2._peak_equity, 1000.0)
        pos = pm2.get_position("BTC-USDT", "long")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.strategy_name, "trend")
        print(f"  [PASS] test_41: save/restore: {pm2.get_position_count()} positions, "
              f"peak_equity={pm2._peak_equity}")
    
    def test_42_restore_nonexistent(self):
        """恢复不存在的状态文件"""
        pm2 = PositionManager(
            self.config, self.mock_client,
            sqlite_storage=MagicMock(),
            redis_cache=MagicMock(),
        )
        pm2._persist_file = "data/nonexistent_state.json"
        pm2._restore_state()
        self.assertEqual(pm2.get_position_count(), 0)
        print(f"  [PASS] test_42: restore nonexistent: no crash")


# ═══════════════════════════════════════════════════════════════
# 端到端集成测试
# ═══════════════════════════════════════════════════════════════

class TestE2EExecutionPosition(unittest.TestCase):
    """执行引擎 + 仓位管理 端到端集成测试"""
    
    def setUp(self):
        self.config = make_mock_config()
        self.mock_client = MockOKXClient()
        self.executor = OrderExecutor(
            self.config, self.mock_client,
            redis_cache=MagicMock(),
            sqlite_storage=MagicMock(),
        )
        self.pm = PositionManager(
            self.config, self.mock_client,
            sqlite_storage=MagicMock(),
            redis_cache=MagicMock(),
        )
    
    def test_51_e2e_execution_quality_flow(self):
        """E2E: 执行质量完整流程"""
        # 模拟几次订单
        for i in range(5):
            key = self.executor._generate_idempotency_key(
                "BTC-USDT", "grid", "open_long"
            )
            self.executor._record_idempotency_key(key, f"ord_{i:03d}")
            self.executor._execution_stats["total_orders"] += 1
        
        # 模拟一次重复检测
        existing_key = list(self.executor._sent_clordids.keys())[0]
        self.executor._check_idempotency(existing_key)
        
        # 验证质量报告
        report = self.executor.get_execution_quality_report()
        self.assertEqual(report["summary"]["total_orders"], 5)
        self.assertEqual(report["summary"]["idempotency_hits"], 1)
        self.assertGreaterEqual(report["quality_score"], 0.8)
        print(f"  [PASS] test_51: E2E execution quality: orders={report['summary']['total_orders']}, "
              f"score={report['quality_score']:.2f}")
    
    def test_52_e2e_position_conflict_flow(self):
        """E2E: 仓位冲突检测完整流程"""
        # 设置模拟持仓
        self._add_position("BTC-USDT", "long", 0.01, 65000, 65100, 200, "trend")
        self._add_position("BTC-USDT", "long", 0.02, 64500, 65100, 400, "grid")
        self._add_position("ETH-USDT", "long", 0.1, 3000, 3010, 150, "trend")
        self._add_position("ETH-USDT", "short", 0.05, 3050, 3010, 80, "grid")
        
        # 冲突检测
        conflicts = self.pm.detect_cross_strategy_conflicts()
        self.assertEqual(len(conflicts), 2)  # 1个同向 + 1个对冲
        
        # 集中度
        conc = self.pm.get_position_concentration()
        self.assertGreater(conc["hhi"], 0.3)
        
        # 最优减仓
        self.pm._account_risk.total_equity = 1000.0
        order = self.pm.get_optimal_position_close_order(500.0)
        self.assertGreater(len(order), 0)
        
        print(f"  [PASS] test_52: E2E position conflict: {len(conflicts)} conflicts, "
              f"hhi={conc['hhi']:.4f}, close_order={len(order)} positions")
    
    def test_53_e2e_risk_linkage(self):
        """E2E: 风险联动"""
        self._add_position("BTC-USDT", "long", 0.01, 65000, 66000, 200,
                          "trend", unrealized_pnl=-100.0, health_score=0.3,
                          margin_ratio=0.4, liq_price=100000.0)
        
        # 风险事件回调
        events = []
        self.pm.on_risk_event(lambda e: events.append(e))
        
        # 模拟高回撤
        self.pm._peak_equity = 1200.0
        self.pm._account_risk.total_equity = 900.0
        
        # 触发回撤检查
        drawdown = (1200.0 - 900.0) / 1200.0
        self.assertGreater(drawdown, self.pm._drawdown_trigger)
        
        event = RiskEvent(
            event_type="drawdown",
            severity=RiskLevel.HIGH,
            message=f"Drawdown {drawdown:.2%}",
            value=drawdown,
            threshold=self.pm._drawdown_trigger,
        )
        self.pm._notify_risk_event(event)
        
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].severity, RiskLevel.HIGH)
        print(f"  [PASS] test_53: E2E risk linkage: {len(events)} events, "
              f"drawdown={drawdown:.2%}")
    
    def test_54_e2e_concentration_warning(self):
        """E2E: 集中度预警"""
        # 单一币种高集中度
        self._add_position("BTC-USDT", "long", 0.1, 65000, 65100, 5000, "trend")
        self._add_position("ETH-USDT", "long", 0.01, 3000, 3010, 15, "grid")
        
        conc = self.pm.get_position_concentration()
        self.assertEqual(conc["risk_level"], "critical")
        self.assertGreater(conc["hhi"], 0.8)
        
        # 最优减仓应优先减BTC
        self.pm._account_risk.total_equity = 5000.0
        order = self.pm.get_optimal_position_close_order(1000.0)
        self.assertEqual(order[0]["symbol"], "BTC-USDT")
        print(f"  [PASS] test_54: E2E concentration: risk={conc['risk_level']}, "
              f"hhi={conc['hhi']:.4f}, first_close={order[0]['symbol']}")
    
    def _add_position(self, symbol, side, qty, avg_cost, mark_price, margin,
                      strategy="", unrealized_pnl=0.0, leverage=1,
                      liq_price=0.0, margin_ratio=0.0, health_score=1.0):
        """辅助方法"""
        key = f"{symbol}:{side}"
        pos = PositionSnapshot(
            symbol=symbol,
            side=PositionSide(side),
            quantity=qty,
            avg_cost=avg_cost,
            mark_price=mark_price,
            unrealized_pnl=unrealized_pnl,
            margin=margin,
            leverage=leverage,
            liquidation_price=liq_price,
            margin_ratio=margin_ratio,
            strategy_name=strategy,
            entry_time=time.time(),
            health_score=health_score,
        )
        self.pm._positions[key] = pos
        if strategy:
            self.pm._positions_by_strategy[strategy].append(key)
        self.pm._positions_by_symbol[symbol].append(key)


if __name__ == "__main__":
    unittest.main(verbosity=2)