"""
生产级执行核心集成测试
======================
覆盖：订单执行引擎（幂等键/指数退避/部分成交）+ 仓位管理系统（实时追踪/风险联动/动态调整）
"""

import os
import sys
import json
import time
import asyncio
import unittest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from execution.order_executor import (
    OrderExecutor, RetryableError, PartialFillAction
)
from core.position_manager import (
    PositionManager, PositionSnapshot, PositionSide,
    PositionStatus, RiskLevel, AccountRiskSnapshot, RiskEvent
)


# ═══════════════════════════════════════════════════════════════
# 测试辅助
# ═══════════════════════════════════════════════════════════════

def load_config():
    """加载测试配置"""
    import yaml
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class MockOKXClient:
    """模拟OKX客户端"""

    def __init__(self):
        self.api_key = "test_api_key"
        self.secret_key = "test_secret_key"
        self.passphrase = "test_passphrase"
        self._positions = {}
        self._orders = {}
        self._account_info = {
            "totalEq": "1000.0",
            "availBal": "800.0",
            "totalMgn": "200.0",
            "mgnRatio": "0.2",
            "upl": "50.0",
            "details": [
                {"ccy": "USDT", "eq": "1000.0", "availBal": "800.0"}
            ]
        }

    def get_account_info(self):
        return self._account_info

    def get_positions(self):
        return [
            {
                "instId": symbol,
                "posSide": side,
                "pos": str(qty),
                "avgPx": str(avg),
                "markPx": str(mark),
                "upl": str(pnl),
                "margin": str(margin),
                "lever": str(lev),
                "liqPx": str(liq),
                "mgnRatio": str(mgn),
            }
            for (symbol, side), (qty, avg, mark, pnl, margin, lev, liq, mgn) in self._positions.items()
        ]

    def _parse_position(self, pos_data):
        """模拟解析持仓"""
        mock = Mock()
        mock.symbol = pos_data.get("instId", "")
        mock.side = pos_data.get("posSide", "long")
        mock.quantity = float(pos_data.get("pos", 0))
        mock.avg_cost = float(pos_data.get("avgPx", 0))
        mock.mark_price = float(pos_data.get("markPx", 0))
        mock.unrealized_pnl = float(pos_data.get("upl", 0))
        mock.margin = float(pos_data.get("margin", 0))
        mock.leverage = int(float(pos_data.get("lever", 1)))
        mock.liq_price = float(pos_data.get("liqPx", 0))
        mock.margin_ratio = float(pos_data.get("mgnRatio", 0))
        return mock

    def get_ticker(self, symbol):
        return {"last": "100.0"}

    def place_order(self, **kwargs):
        return {"ordId": f"test_ord_{int(time.time()*1000)}", "sCode": "0", "sMsg": ""}

    def get_order(self, symbol, order_id):
        if order_id in self._orders:
            return self._orders[order_id]
        return {"state": "filled", "avgPx": "100.0", "fee": "0.05", "pnl": "1.0", "sz": "1"}

    def set_leverage(self, symbol, leverage, pos_side=None):
        return True

    def round_quantity_to_lot(self, symbol, quantity, round_up=False):
        return quantity

    def cancel_order(self, symbol, order_id):
        return {"sCode": "0"}

    def transfer_to_trading(self, currency, amount):
        return True

    def get_instrument_info(self, symbol):
        return {"ctVal": "1"}

    def contracts_to_coins(self, symbol, contracts):
        return contracts

    def get_atr(self, symbol):
        return 0.5

    def get_bills(self, symbol, bill_type="", limit=5):
        return []


class MockSQLiteStorage:
    def __init__(self):
        self._records = {}
        self._open_records = []

    def get_all_open_records(self, symbol):
        return self._open_records

    def get_trade_records_by_status(self, status, limit=100):
        return [r for r in self._records.values() if r.get("status") == status]

    def save_trade_record(self, record):
        self._records[record.get("id", str(time.time()))] = record
        if record.get("status") == "open":
            self._open_records.append(record)

    def update_trade_record(self, record_id, updates):
        if record_id in self._records:
            self._records[record_id].update(updates)

    def get_latest_open_record(self, symbol, strategy_name):
        for rec in reversed(self._open_records):
            if rec.get("symbol") == symbol and rec.get("strategy_name") == strategy_name:
                return rec
        return None

    def save_position_history(self, data):
        pass

    def save_account_history(self, data):
        pass

    def get_latest_position_history(self, symbol):
        return None


class MockRedisCache:
    def set_position(self, pos):
        pass

    def set_account_info(self, acc):
        pass


class MockAccountManager:
    def notify_order_placed(self, strategy, margin):
        pass

    def notify_order_filled(self, strategy, margin):
        pass

    def notify_order_canceled(self, strategy, margin):
        pass


# ═══════════════════════════════════════════════════════════════
# 订单执行引擎测试
# ═══════════════════════════════════════════════════════════════

class TestOrderExecutorIdempotency(unittest.TestCase):
    """测试1: 幂等键"""

    def setUp(self):
        self.config = load_config()
        self.mock_client = MockOKXClient()
        self.mock_redis = MockRedisCache()
        self.mock_sqlite = MockSQLiteStorage()
        self.executor = OrderExecutor(
            self.config, self.mock_client,
            self.mock_redis, self.mock_sqlite
        )

    def test_01_generate_idempotency_key(self):
        """测试幂等键生成（确定性：同时间戳+同参数生成相同 key，用于幂等去重）"""
        ts = time.time()
        key1 = self.executor._generate_idempotency_key("BTC-USDT", "grid", "open", timestamp=ts)
        key2 = self.executor._generate_idempotency_key("BTC-USDT", "grid", "open", timestamp=ts)
        # 同一时间戳 + 同参数 → 确定性相同 key（网络重试命中同一幂等键）
        self.assertEqual(key1, key2)
        # 不同时间戳 → 不同 key
        key3 = self.executor._generate_idempotency_key("BTC-USDT", "grid", "open", timestamp=ts + 0.001)
        self.assertNotEqual(key1, key3)
        # 长度不超过32
        self.assertLessEqual(len(key1), 32)
        self.assertLessEqual(len(key2), 32)
        # 包含前缀（clOrdId 不允许下划线，前缀默认 'okxqt'）
        self.assertTrue(key1.startswith(self.executor._idempotency_prefix))
        print(f"  [PASS] test_01: idempotency key generated: {key1}")

    def test_02_check_idempotency(self):
        """测试幂等检测"""
        key = self.executor._generate_idempotency_key("ETH-USDT", "trend", "open")
        # 初始无记录
        self.assertIsNone(self.executor._check_idempotency(key))
        # 记录后应检测到
        self.executor._record_idempotency_key(key, "ord_12345")
        existing = self.executor._check_idempotency(key)
        self.assertEqual(existing, "ord_12345")
        print(f"  [PASS] test_02: idempotency check works")

    def test_03_idempotency_stats(self):
        """测试幂等统计"""
        key = self.executor._generate_idempotency_key("SOL-USDT", "scalping", "open")
        self.executor._record_idempotency_key(key, "ord_67890")
        self.executor._check_idempotency(key)  # 触发命中
        stats = self.executor.get_execution_stats()
        self.assertEqual(stats["idempotency_hits"], 1)
        print(f"  [PASS] test_03: idempotency stats: {stats['idempotency_hits']} hits")

    def test_04_clordid_cache_cleanup(self):
        """测试幂等缓存清理"""
        # 添加过期记录
        old_time = time.time() - 7200  # 2小时前
        self.executor._sent_clordids["old_key"] = (old_time, "old_ord")
        self.executor._cleanup_clordid_cache()
        self.assertNotIn("old_key", self.executor._sent_clordids)
        print(f"  [PASS] test_04: clordid cache cleanup works")


class TestOrderExecutorBackoff(unittest.TestCase):
    """测试2: 指数退避重试"""

    def setUp(self):
        self.config = load_config()
        self.mock_client = MockOKXClient()
        self.mock_redis = MockRedisCache()
        self.mock_sqlite = MockSQLiteStorage()
        self.executor = OrderExecutor(
            self.config, self.mock_client,
            self.mock_redis, self.mock_sqlite
        )

    def test_05_backoff_calculation(self):
        """测试退避延迟计算"""
        # 第一次重试：基础延迟
        delay1 = self.executor._calculate_backoff_delay(1)
        self.assertGreater(delay1, 0)
        self.assertLess(delay1, 5.0)

        # 第三次重试：指数增长
        delay3 = self.executor._calculate_backoff_delay(3)
        self.assertGreater(delay3, delay1)

        # 验证上限
        delay10 = self.executor._calculate_backoff_delay(10)
        self.assertLessEqual(delay10, self.executor._retry_max_delay + 1)
        print(f"  [PASS] test_05: backoff delays: 1st={delay1:.1f}s, 3rd={delay3:.1f}s, 10th={delay10:.1f}s")

    def test_06_backoff_rate_limit(self):
        """测试限流错误退避（更长延迟）"""
        normal_delay = self.executor._calculate_backoff_delay(1)
        rate_limit_delay = self.executor._calculate_backoff_delay(1, RetryableError.RATE_LIMIT)
        self.assertGreater(rate_limit_delay, normal_delay)
        print(f"  [PASS] test_06: rate_limit backoff ({rate_limit_delay:.1f}s) > normal ({normal_delay:.1f}s)")

    def test_07_backoff_network_error(self):
        """测试网络错误退避（更短延迟）"""
        normal_delay = self.executor._calculate_backoff_delay(1)
        network_delay = self.executor._calculate_backoff_delay(1, RetryableError.NETWORK_ERROR)
        self.assertLess(network_delay, normal_delay)
        print(f"  [PASS] test_07: network backoff ({network_delay:.1f}s) < normal ({normal_delay:.1f}s)")

    def test_08_error_classification(self):
        """测试错误分类"""
        # 可重试
        self.assertEqual(
            self.executor._classify_error("429", ""),
            RetryableError.RATE_LIMIT
        )
        self.assertEqual(
            self.executor._classify_error("500", ""),
            RetryableError.SERVER_ERROR
        )
        self.assertEqual(
            self.executor._classify_error("503", ""),
            RetryableError.SERVER_ERROR
        )
        self.assertEqual(
            self.executor._classify_error("", "Connection timed out"),
            RetryableError.NETWORK_ERROR
        )
        # 不可重试
        self.assertIsNone(self.executor._classify_error("51008", ""))
        self.assertIsNone(self.executor._classify_error("51121", ""))
        print(f"  [PASS] test_08: error classification works")


class TestOrderExecutorPartialFill(unittest.TestCase):
    """测试3: 部分成交处理"""

    def setUp(self):
        self.config = load_config()
        self.mock_client = MockOKXClient()
        self.mock_redis = MockRedisCache()
        self.mock_sqlite = MockSQLiteStorage()
        self.executor = OrderExecutor(
            self.config, self.mock_client,
            self.mock_redis, self.mock_sqlite
        )

    def test_09_partial_fill_tracker(self):
        """测试部分成交跟踪"""
        order_info = {
            "symbol": "BTC-USDT",
            "strategy_name": "grid",
            "signal_type": "open",
            "direction": "long",
            "quantity": 1.0,
            "leverage": 5,
            "clOrdId": "test_clord_001",
        }
        # 模拟部分成交
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.executor._handle_partial_fill(
                "ord_001", order_info,
                filled_qty=0.3, total_qty=1.0, filled_price=100.0
            )
        )
        loop.close()
        # 成交率30% < 50%: 应取消剩余
        self.assertTrue(result)
        self.assertNotIn("test_clord_001", self.executor._partial_fill_tracker)
        stats = self.executor.get_execution_stats()
        self.assertEqual(stats["partial_fills"], 1)
        self.assertEqual(stats["partial_fill_resolved"], 1)
        print(f"  [PASS] test_09: partial fill tracker works (fill_ratio=30%, cancelled)")

    def test_10_partial_fill_high_ratio(self):
        """测试高成交率部分成交"""
        self.executor._partial_fill_action = PartialFillAction.WAIT
        order_info = {
            "symbol": "ETH-USDT",
            "strategy_name": "trend",
            "signal_type": "open",
            "direction": "long",
            "quantity": 1.0,
            "leverage": 3,
            "clOrdId": "test_clord_002",
        }
        loop = asyncio.new_event_loop()
        result = loop.run_until_complete(
            self.executor._handle_partial_fill(
                "ord_002", order_info,
                filled_qty=0.8, total_qty=1.0, filled_price=200.0
            )
        )
        loop.close()
        # 成交率80% > 50%: 继续等待
        self.assertFalse(result)
        self.assertIn("test_clord_002", self.executor._partial_fill_tracker)
        print(f"  [PASS] test_10: partial fill high ratio (80%) waits for completion")

    def test_11_execution_stats(self):
        """测试执行统计"""
        stats = self.executor.get_execution_stats()
        self.assertIn("total_orders", stats)
        self.assertIn("duplicate_prevented", stats)
        self.assertIn("retry_count", stats)
        self.assertIn("partial_fills", stats)
        self.assertIn("active_orders", stats)
        self.assertIn("clordid_cache_size", stats)
        self.assertIn("timestamp", stats)
        print(f"  [PASS] test_11: execution stats: {stats['total_orders']} orders, {stats['duplicate_prevented']} dups prevented")

    def test_12_enum_values(self):
        """测试枚举值"""
        self.assertEqual(RetryableError.RATE_LIMIT.value, "rate_limit")
        self.assertEqual(RetryableError.SERVER_ERROR.value, "server_error")
        self.assertEqual(PartialFillAction.CANCEL_REMAINING.value, "cancel_remaining")
        self.assertEqual(PartialFillAction.RESUBMIT_REMAINING.value, "resubmit")
        print(f"  [PASS] test_12: enum values correct")


# ═══════════════════════════════════════════════════════════════
# 仓位管理系统测试
# ═══════════════════════════════════════════════════════════════

class TestPositionManagerSync(unittest.TestCase):
    """测试4: 持仓同步"""

    def setUp(self):
        self.config = load_config()
        self.mock_client = MockOKXClient()
        self.mock_sqlite = MockSQLiteStorage()
        self.mock_redis = MockRedisCache()
        self.pm = PositionManager(
            self.config, self.mock_client,
            self.mock_sqlite, self.mock_redis
        )
        # 设置模拟持仓
        self.mock_client._positions = {
            ("BTC-USDT", "long"): (0.1, 90000, 95000, 500, 1800, 10, 45000, 0.02),
            ("ETH-USDT", "short"): (1.0, 3000, 2950, 50, 600, 5, 4500, 0.015),
        }

    def test_13_position_sync(self):
        """测试持仓同步"""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.pm._sync_positions())
        loop.close()

        positions = self.pm.get_all_positions()
        self.assertEqual(len(positions), 2)
        # 检查BTC持仓
        btc_pos = self.pm.get_position("BTC-USDT", "long")
        self.assertIsNotNone(btc_pos)
        self.assertEqual(btc_pos.symbol, "BTC-USDT")
        self.assertEqual(btc_pos.side, PositionSide.LONG)
        self.assertEqual(btc_pos.quantity, 0.1)
        self.assertEqual(btc_pos.leverage, 10)
        # 检查ETH持仓
        eth_pos = self.pm.get_position("ETH-USDT", "short")
        self.assertIsNotNone(eth_pos)
        self.assertEqual(eth_pos.side, PositionSide.SHORT)
        print(f"  [PASS] test_13: position sync: {len(positions)} positions")

    def test_14_position_removal(self):
        """测试持仓移除"""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.pm._sync_positions())
        # 移除持仓
        self.mock_client._positions = {}
        loop.run_until_complete(self.pm._sync_positions())
        loop.close()

        positions = self.pm.get_all_positions()
        self.assertEqual(len(positions), 0)
        print(f"  [PASS] test_14: position removal: {len(positions)} positions")

    def test_15_ws_position_update(self):
        """测试WebSocket仓位更新"""
        ws_data = [
            {
                "instId": "SOL-USDT",
                "posSide": "long",
                "pos": "10",
                "avgPx": "150",
                "markPx": "155",
                "upl": "50",
                "margin": "300",
                "lever": "5",
                "liqPx": "120",
                "mgnRatio": "0.1",
            }
        ]
        self.pm.update_position_from_ws(ws_data)
        pos = self.pm.get_position("SOL-USDT", "long")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.quantity, 10.0)
        self.assertEqual(pos.avg_cost, 150.0)
        self.assertEqual(pos.leverage, 5)
        print(f"  [PASS] test_15: WS position update: SOL-USDT long x10")

    def test_16_ws_position_remove(self):
        """测试WebSocket持仓清零"""
        # 先添加持仓
        ws_data_add = [{"instId": "DOT-USDT", "posSide": "long", "pos": "5", "avgPx": "7", "markPx": "7.5", "upl": "2.5", "margin": "7", "lever": "5", "liqPx": "5", "mgnRatio": "0.05"}]
        self.pm.update_position_from_ws(ws_data_add)
        self.assertEqual(self.pm.get_position_count(), 1)
        # 清零
        ws_data_remove = [{"instId": "DOT-USDT", "posSide": "long", "pos": "0"}]
        self.pm.update_position_from_ws(ws_data_remove)
        self.assertEqual(self.pm.get_position_count(), 0)
        print(f"  [PASS] test_16: WS position removal works")


class TestPositionManagerRisk(unittest.TestCase):
    """测试5: 风险联动"""

    def setUp(self):
        self.config = load_config()
        self.mock_client = MockOKXClient()
        self.mock_sqlite = MockSQLiteStorage()
        self.mock_redis = MockRedisCache()
        self.pm = PositionManager(
            self.config, self.mock_client,
            self.mock_sqlite, self.mock_redis
        )
        self.risk_events = []

    def test_17_account_risk_check(self):
        """测试账户风险检查"""
        self.pm.on_risk_event(lambda e: self.risk_events.append(e))

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.pm._check_account_risk())
        loop.close()

        risk = self.pm.get_account_risk()
        self.assertGreater(risk.total_equity, 0)
        self.assertGreaterEqual(risk.position_count, 0)
        print(f"  [PASS] test_17: account risk: equity={risk.total_equity}, margin_ratio={risk.margin_ratio}")

    def test_17b_account_risk_empty_mgnratio(self):
        """单币种账户 mgnRatio 为空时从 details 计算占用率，不静默失败"""
        self.mock_client._account_info = {
            "totalEq": "215.0",
            "mgnRatio": "",  # 单币种 USDT 账户该字段为空
            "details": [
                {"ccy": "USDT", "eq": "215.0", "availEq": "170.0",
                 "availBal": "170.0", "upl": "-5.0"},
            ],
        }

        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.pm._check_account_risk())
        loop.close()

        risk = self.pm.get_account_risk()
        self.assertAlmostEqual(risk.margin_ratio, 45.0 / 215.0, places=3)
        self.assertGreater(risk.total_equity, 0)
        print(f"  [PASS] test_17b: empty mgnRatio → margin_ratio={risk.margin_ratio:.3f}")

    def test_18_position_health_score(self):
        """测试持仓健康评分"""
        pos = PositionSnapshot(
            symbol="BTC-USDT",
            side=PositionSide.LONG,
            quantity=0.1,
            avg_cost=90000,
            mark_price=95000,
            unrealized_pnl=500,
            margin=1800,
            leverage=10,
            liquidation_price=45000,
            margin_ratio=0.02,
        )
        health = self.pm._calculate_position_health(pos)
        self.assertGreaterEqual(health, 0.0)
        self.assertLessEqual(health, 1.0)
        print(f"  [PASS] test_18: position health score: {health:.2f} (profitable)")

    def test_19_losing_position_health(self):
        """测试亏损持仓健康评分"""
        pos = PositionSnapshot(
            symbol="ETH-USDT",
            side=PositionSide.SHORT,
            quantity=1.0,
            avg_cost=3000,
            mark_price=3300,  # 亏损10%
            unrealized_pnl=-300,
            margin=600,
            leverage=5,
            liquidation_price=4500,
            margin_ratio=0.5,
        )
        health = self.pm._calculate_position_health(pos)
        self.assertLess(health, 1.0)
        print(f"  [PASS] test_19: losing position health: {health:.2f}")

    def test_20_risk_event_callback(self):
        """测试风险事件回调"""
        event = RiskEvent(
            event_type="margin_warning",
            severity=RiskLevel.ELEVATED,
            symbol="BTC-USDT",
            message="Test margin warning",
            value=0.6,
            threshold=0.5,
        )
        self.pm._notify_risk_event(event)
        events = self.pm.get_risk_events()
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[-1].event_type, "margin_warning")
        print(f"  [PASS] test_20: risk event callback: {len(events)} events")


class TestPositionManagerQuery(unittest.TestCase):
    """测试6: 查询接口"""

    def setUp(self):
        self.config = load_config()
        self.mock_client = MockOKXClient()
        self.mock_sqlite = MockSQLiteStorage()
        self.mock_redis = MockRedisCache()
        self.pm = PositionManager(
            self.config, self.mock_client,
            self.mock_sqlite, self.mock_redis
        )

    def test_21_get_position_by_symbol(self):
        """测试按币种查询持仓"""
        # 手动添加持仓
        self.pm._positions["BTC-USDT:long"] = PositionSnapshot(
            symbol="BTC-USDT", side=PositionSide.LONG, quantity=0.1,
            avg_cost=90000, mark_price=95000, unrealized_pnl=500,
            leverage=10, strategy_name="grid",
        )
        self.pm._positions_by_symbol["BTC-USDT"] = ["BTC-USDT:long"]

        btc_positions = self.pm.get_positions_by_symbol("BTC-USDT")
        self.assertEqual(len(btc_positions), 1)
        self.assertEqual(btc_positions[0].strategy_name, "grid")
        print(f"  [PASS] test_21: positions by symbol: BTC-USDT -> {len(btc_positions)}")

        eth_positions = self.pm.get_positions_by_symbol("ETH-USDT")
        self.assertEqual(len(eth_positions), 0)

    def test_22_get_position_by_strategy(self):
        """测试按策略查询持仓"""
        self.pm._positions["ETH-USDT:short"] = PositionSnapshot(
            symbol="ETH-USDT", side=PositionSide.SHORT, quantity=1.0,
            avg_cost=3000, mark_price=2950, unrealized_pnl=50,
            leverage=5, strategy_name="trend",
        )
        self.pm._positions_by_strategy["trend"] = ["ETH-USDT:short"]

        trend_positions = self.pm.get_positions_by_strategy("trend")
        self.assertEqual(len(trend_positions), 1)
        print(f"  [PASS] test_22: positions by strategy: trend -> {len(trend_positions)}")

    def test_23_get_stats(self):
        """测试统计信息"""
        self.pm._positions["BTC-USDT:long"] = PositionSnapshot(
            symbol="BTC-USDT", side=PositionSide.LONG, quantity=0.1,
            avg_cost=90000, mark_price=95000, unrealized_pnl=500,
            margin=1800, leverage=10, health_score=0.9,
        )
        self.pm._positions["ETH-USDT:short"] = PositionSnapshot(
            symbol="ETH-USDT", side=PositionSide.SHORT, quantity=1.0,
            avg_cost=3000, mark_price=2950, unrealized_pnl=50,
            margin=600, leverage=5, health_score=0.3,
        )

        stats = self.pm.get_stats()
        self.assertEqual(stats["total_positions"], 2)
        self.assertGreater(stats["total_exposure"], 0)
        self.assertEqual(stats["health"]["healthy"], 1)
        self.assertEqual(stats["health"]["critical"], 1)
        self.assertIn("account_risk", stats)
        print(f"  [PASS] test_23: stats: {stats['total_positions']} positions, "
              f"healthy={stats['health']['healthy']}, critical={stats['health']['critical']}")

    def test_24_set_position_strategy(self):
        """测试设置持仓策略关联"""
        self.pm._positions["SOL-USDT:long"] = PositionSnapshot(
            symbol="SOL-USDT", side=PositionSide.LONG, quantity=10,
            avg_cost=150, mark_price=155, unrealized_pnl=50,
        )
        self.pm.set_position_strategy("SOL-USDT", "long", "scalping")
        pos = self.pm.get_position("SOL-USDT", "long")
        self.assertEqual(pos.strategy_name, "scalping")
        print(f"  [PASS] test_24: strategy set: SOL-USDT -> scalping")


class TestPositionManagerPersistence(unittest.TestCase):
    """测试7: 状态持久化"""

    def setUp(self):
        self.config = load_config()
        self.mock_client = MockOKXClient()
        self.mock_sqlite = MockSQLiteStorage()
        self.mock_redis = MockRedisCache()
        self.pm = PositionManager(
            self.config, self.mock_client,
            self.mock_sqlite, self.mock_redis
        )
        self.pm._persist_file = "data/test_position_state.json"

    def tearDown(self):
        if os.path.exists(self.pm._persist_file):
            os.remove(self.pm._persist_file)

    def test_25_save_and_restore(self):
        """测试保存和恢复"""
        # 添加持仓
        self.pm._positions["BTC-USDT:long"] = PositionSnapshot(
            symbol="BTC-USDT", side=PositionSide.LONG, quantity=0.1,
            avg_cost=90000, mark_price=95000, unrealized_pnl=500,
            margin=1800, leverage=10, strategy_name="grid",
            health_score=0.85,
        )
        self.pm._peak_equity = 1000.0
        self.pm._account_risk = AccountRiskSnapshot(
            total_equity=950.0,
            drawdown_pct=0.05,
            risk_level=RiskLevel.ELEVATED,
        )

        # 保存
        self.pm._save_state()
        self.assertTrue(os.path.exists(self.pm._persist_file))

        # 创建新的PM实例恢复
        pm2 = PositionManager(
            self.config, self.mock_client,
            self.mock_sqlite, self.mock_redis
        )
        pm2._persist_file = "data/test_position_state.json"
        pm2._restore_state()

        self.assertEqual(pm2.get_position_count(), 1)
        restored_pos = pm2.get_position("BTC-USDT", "long")
        self.assertIsNotNone(restored_pos)
        self.assertEqual(restored_pos.strategy_name, "grid")
        self.assertEqual(pm2._peak_equity, 1000.0)
        print(f"  [PASS] test_25: save/restore: {pm2.get_position_count()} positions, peak_equity={pm2._peak_equity}")

    def test_26_restore_nonexistent(self):
        """测试恢复不存在的文件"""
        pm2 = PositionManager(
            self.config, self.mock_client,
            self.mock_sqlite, self.mock_redis
        )
        pm2._persist_file = "data/nonexistent.json"
        pm2._restore_state()
        self.assertEqual(pm2.get_position_count(), 0)
        print(f"  [PASS] test_26: restore nonexistent: no crash")


# ═══════════════════════════════════════════════════════════════
# 配置加载测试
# ═══════════════════════════════════════════════════════════════

class TestConfigLoading(unittest.TestCase):
    """测试8: 配置加载"""

    def test_27_execution_config(self):
        """测试执行引擎配置"""
        config = load_config()
        exec_cfg = config.get("execution", {})
        self.assertTrue(exec_cfg.get("idempotency_enabled", False))
        self.assertEqual(exec_cfg.get("idempotency_prefix"), "okxqt")
        self.assertGreater(exec_cfg.get("retry_base_delay_sec", 0), 0)
        self.assertGreater(exec_cfg.get("retry_max_delay_sec", 0), 0)
        self.assertTrue(exec_cfg.get("partial_fill_enabled", False))
        print(f"  [PASS] test_27: execution config loaded")

    def test_28_position_manager_config(self):
        """测试仓位管理配置"""
        config = load_config()
        pm_cfg = config.get("position_manager", {})
        self.assertTrue(pm_cfg.get("enabled", False))
        self.assertGreater(pm_cfg.get("max_total_positions", 0), 0)
        self.assertGreater(pm_cfg.get("sync_interval_sec", 0), 0)
        rl = pm_cfg.get("risk_linkage", {})
        self.assertGreater(rl.get("drawdown_trigger", 0), 0)
        self.assertGreater(rl.get("margin_ratio_warning", 0), 0)
        print(f"  [PASS] test_28: position manager config loaded")

    def test_29_websocket_config(self):
        """测试WebSocket配置"""
        config = load_config()
        ws_cfg = config.get("websocket", {})
        self.assertTrue(ws_cfg.get("enabled", False))
        self.assertGreater(ws_cfg.get("ping_interval", 0), 0)
        self.assertGreater(ws_cfg.get("buffer_size", 0), 0)
        pc = ws_cfg.get("public_channels", {})
        self.assertIn("tickers", pc)
        print(f"  [PASS] test_29: websocket config loaded")

    def test_30_full_config_integrity(self):
        """测试完整配置完整性"""
        config = load_config()
        # 核心配置必须存在
        self.assertIn("execution", config)
        self.assertIn("position_manager", config)
        self.assertIn("websocket", config)
        self.assertIn("strategies", config)
        self.assertIn("trading", config)
        self.assertIn("risk", config)
        print(f"  [PASS] test_30: full config integrity: {len(config)} top-level sections")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("生产级执行核心集成测试")
    print("=" * 60)
    print(f"时间: {datetime.now().isoformat()}")
    print()

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 按顺序添加测试
    suite.addTests(loader.loadTestsFromTestCase(TestOrderExecutorIdempotency))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderExecutorBackoff))
    suite.addTests(loader.loadTestsFromTestCase(TestOrderExecutorPartialFill))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionManagerSync))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionManagerRisk))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionManagerQuery))
    suite.addTests(loader.loadTestsFromTestCase(TestPositionManagerPersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestConfigLoading))

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