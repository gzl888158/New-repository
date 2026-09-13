"""
生产级仪表板 V3.0 集成测试
================================
测试覆盖：
1. DashboardEngine 初始化与配置
2. 数据模型创建
3. 账户快照获取（模拟数据）
4. 权益曲线计算
5. 策略对比数据聚合
6. 持仓分布多维分析
7. 风险水位计算
8. 缓存机制
9. 策略推断
10. 快照持久化
11. 依赖注入
12. 一键全量获取
13. 边界条件
14. 热更新配置
15. 历史表现统计 (V3.0)
16. 资金效率指标 (V3.0)
17. 活跃告警检测 (V3.0)
18. 资金费率趋势 (V3.0)
19. 订单簿摘要 (V3.0)
20. 策略热力图 (V3.0)
21. 市场概览 (V3.0)
22. 全量数据包含V3.0模块 (V3.0)
"""

import os
import sys
import json
import time
import sqlite3
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.dashboard_engine import (
    DashboardEngine, AccountSnapshot, EquityCurvePoint,
    StrategyMetric, PositionItem, RiskMetrics,
    HistoricalPerformance, CapitalEfficiencyMetrics,
    ActiveAlert, FundingTrendItem, OrderbookSummary,
    StrategyHeatmapCell, MarketOverview,
)


class MockOKXClient:
    """模拟OKX客户端"""
    def fetch_account(self):
        return {
            "totalEq": "5000.00",
            "details": [{"ccy": "USDT", "availBal": "3000.00", "frozenBal": "100.00", "eq": "5000.00"}]
        }

    def fetch_positions(self):
        return [
            {"instId": "BTC-USDT-SWAP", "posSide": "long", "pos": "0.1", "avgPx": "65000.0", "markPx": "66000.0",
             "liqPx": "55000.0", "margin": "2000.00", "upl": "100.00", "lever": "5", "maintMargin": "50.00",
             "notionalUsd": "6500.00", "fundingRate": "0.0001"},
            {"instId": "ETH-USDT-SWAP", "posSide": "short", "pos": "1.0", "avgPx": "3500.0", "markPx": "3400.0",
             "liqPx": "5000.0", "margin": "500.00", "upl": "100.00", "lever": "3", "maintMargin": "20.00",
             "notionalUsd": "3400.00", "fundingRate": "-0.00005"},
            {"instId": "SOL-USDT-SWAP", "posSide": "long", "pos": "0", "avgPx": "0", "markPx": "0",
             "liqPx": "0", "margin": "0", "upl": "0", "lever": "1", "maintMargin": "0",
             "notionalUsd": "0", "fundingRate": "0"},
        ]

    def fetch_ticker(self, symbol):
        if "BTC" in symbol:
            return {"last": "68000.00", "change24h": "2.5"}
        if "ETH" in symbol:
            return {"last": "3500.00", "change24h": "-1.5"}
        return {}

    def fetch_orderbook(self, symbol, depth=20):
        return {
            "bids": [["67990.0", "1.5"], ["67980.0", "2.0"], ["67970.0", "3.0"]],
            "asks": [["68010.0", "1.0"], ["68020.0", "1.5"], ["68030.0", "2.0"]],
        }


class MockStateManager:
    """模拟状态管理器"""
    def load_all_states(self):
        return {
            "grid_strategy": {"symbols": ["BTC", "ETH"], "coins": ["BTC", "ETH"]},
            "trend_strategy": {"symbol": "SOL"},
            "scalping_strategy": {"symbols": ["BTC"]},
        }


class MockCapitalManager:
    """模拟资金管理器"""
    def __init__(self):
        self.attrition_analyzer = None
        self._last_rebalance_date = "2024-01-01"


def create_test_db(db_path):
    """创建测试数据库"""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS account_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            total_equity REAL DEFAULT 0,
            available_balance REAL DEFAULT 0,
            used_margin REAL DEFAULT 0,
            unrealized_pnl REAL DEFAULT 0,
            realized_pnl_today REAL DEFAULT 0,
            margin_utilization REAL DEFAULT 0,
            health_level TEXT DEFAULT 'normal'
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            strategy_name TEXT DEFAULT '',
            realized_pnl REAL DEFAULT 0
        )
    """)
    # 插入历史数据
    for i in range(30):
        ts = (datetime.now() - timedelta(days=i)).isoformat()
        conn.execute(
            "INSERT INTO account_history (timestamp, total_equity, available_balance, used_margin, unrealized_pnl) VALUES (?, ?, ?, ?, ?)",
            (ts, 5000 - i * 10, 3000 - i * 5, 2000 - i * 3, 100 - i * 2)
        )
    # 插入交易记录
    today = datetime.now().strftime("%Y-%m-%d")
    for i in range(5):
        conn.execute("INSERT INTO trades (timestamp, strategy_name, realized_pnl) VALUES (?, ?, ?)",
                     (today, "grid", 50.0))
        conn.execute("INSERT INTO trades (timestamp, strategy_name, realized_pnl) VALUES (?, ?, ?)",
                     (today, "grid", -20.0))
        conn.execute("INSERT INTO trades (timestamp, strategy_name, realized_pnl) VALUES (?, ?, ?)",
                     (today, "trend", 100.0))
        conn.execute("INSERT INTO trades (timestamp, strategy_name, realized_pnl) VALUES (?, ?, ?)",
                     (today, "trend", -30.0))
        conn.execute("INSERT INTO trades (timestamp, strategy_name, realized_pnl) VALUES (?, ?, ?)",
                     (today, "scalping", 15.0))
        conn.execute("INSERT INTO trades (timestamp, strategy_name, realized_pnl) VALUES (?, ?, ?)",
                     (today, "scalping", -5.0))
    conn.commit()
    conn.close()


# ═══════════════════════════════════════════════════════════════
# 测试用例
# ═══════════════════════════════════════════════════════════════

def test_01_engine_initialization():
    """测试1: DashboardEngine 初始化"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    assert engine is not None
    assert engine._db_path.endswith("data/test_dashboard_v3.db")
    assert engine.CACHE_TTL_SECONDS == 3.0
    print("  [PASS] test_01: DashboardEngine 初始化成功")


def test_02_data_models():
    """测试2: 数据模型创建"""
    # 原有模型
    snapshot = AccountSnapshot(total_equity=5000.0, health_level="normal")
    assert snapshot.total_equity == 5000.0

    point = EquityCurvePoint(equity=5000.0, drawdown=0.02)
    assert point.drawdown == 0.02

    metric = StrategyMetric(name="grid", profit_factor=1.5)
    assert metric.profit_factor == 1.5

    item = PositionItem(symbol_base="BTC", risk_level="caution")
    assert item.risk_level == "caution"

    # V3.0 新模型
    perf = HistoricalPerformance(period="30d", sharpe_ratio=1.5, total_return=500.0)
    assert perf.period == "30d"
    assert perf.sharpe_ratio == 1.5

    eff = CapitalEfficiencyMetrics(total_capital=5000.0, efficiency_ratio=0.6)
    assert eff.total_capital == 5000.0

    alert = ActiveAlert(type="margin", severity="critical", message="保证金不足")
    assert alert.severity == "critical"

    funding = FundingTrendItem(symbol="BTC", daily_cost=0.5, trend="rising")
    assert funding.trend == "rising"

    ob = OrderbookSummary(symbol="BTC", liquidity_score=85.0)
    assert ob.liquidity_score == 85.0

    cell = StrategyHeatmapCell(strategy="grid", status="profitable")
    assert cell.status == "profitable"

    market = MarketOverview(btc_price=68000.0, market_regime="bull")
    assert market.btc_price == 68000.0
    print("  [PASS] test_02: 所有数据模型创建成功")


def test_03_dependency_injection():
    """测试3: 依赖注入"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    mock_capital = MockCapitalManager()
    mock_state = MockStateManager()

    engine.set_dependencies(okx_client=mock_client, capital_manager=mock_capital, state_manager=mock_state)

    assert engine._okx_client is not None
    assert engine._capital_manager is not None
    assert engine._state_manager is not None
    assert len(engine._strategy_symbol_map) > 0
    print("  [PASS] test_03: 依赖注入成功")


def test_04_account_snapshot():
    """测试4: 账户快照获取（模拟数据）"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    snapshot = engine.get_account_snapshot(force_refresh=True)
    assert snapshot is not None
    assert snapshot.total_equity == 5000.0
    assert snapshot.available_balance == 3000.0
    assert snapshot.health_level in ("normal", "caution", "danger")
    print(f"  [PASS] test_04: 账户快照获取成功, equity={snapshot.total_equity}")


def test_05_position_distribution():
    """测试5: 持仓分布多维分析"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    mock_state = MockStateManager()
    engine.set_dependencies(okx_client=mock_client, state_manager=mock_state)

    result = engine.get_position_distribution(force_refresh=True)
    assert result is not None
    assert len(result["positions"]) == 2
    assert "by_symbol" in result["distribution"]
    assert "by_side" in result["distribution"]
    assert "by_leverage" in result["distribution"]
    assert "by_strategy" in result["distribution"]
    assert "by_risk_level" in result["distribution"]
    print(f"  [PASS] test_05: 持仓分布成功, {len(result['positions'])} 个持仓")


def test_06_risk_dashboard():
    """测试6: 风险水位计算"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    result = engine.get_risk_dashboard(force_refresh=True)
    assert "risk_metrics" in result
    assert "gauges" in result
    assert len(result["gauges"]) == 6
    print(f"  [PASS] test_06: 风险水位成功, 风险等级={result['risk_metrics']['risk_level']}")


def test_07_equity_curve():
    """测试7: 权益曲线"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    result = engine.get_equity_curve(days=30, force_refresh=True)
    assert len(result["points"]) > 0
    print(f"  [PASS] test_07: 权益曲线成功, {len(result['points'])} 个数据点")


def test_08_strategy_comparison():
    """测试8: 策略对比数据聚合"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    result = engine.get_strategy_comparison(force_refresh=True)
    assert "strategies" in result
    assert "summary" in result
    print(f"  [PASS] test_08: 策略对比成功, {len(result['strategies'])} 个策略")


def test_09_cache_mechanism():
    """测试9: 缓存机制"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    result1 = engine.get_position_distribution(force_refresh=True)
    t1 = time.time()
    result2 = engine.get_position_distribution()
    t2 = time.time()
    assert result1["summary"]["total_positions"] == result2["summary"]["total_positions"]
    print(f"  [PASS] test_09: 缓存机制成功 (缓存命中)")


def test_10_strategy_inference():
    """测试10: 策略推断"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_state = MockStateManager()
    engine.set_dependencies(state_manager=mock_state)

    name, display = engine._infer_position_strategy("BTC-USDT-SWAP", "BTC")
    assert name == "grid_strategy"
    assert display == "Grid"

    name, display = engine._infer_position_strategy("UNKNOWN-USDT-SWAP", "UNKNOWN")
    assert name == "unknown"
    print("  [PASS] test_10: 策略推断成功")


def test_11_full_dashboard():
    """测试11: 一键全量获取"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    mock_state = MockStateManager()
    engine.set_dependencies(okx_client=mock_client, state_manager=mock_state)

    result = engine.get_dashboard_full()
    assert "account" in result
    assert "equity_curve" in result
    assert "strategy_comparison" in result
    assert "position_distribution" in result
    assert "risk_dashboard" in result
    assert "timestamp" in result
    print("  [PASS] test_11: 一键全量获取成功")


def test_12_edge_cases():
    """测试12: 边界条件"""
    engine = DashboardEngine()
    assert engine.config == {}

    # 无注入 OKX 客户端时，_fetch_okx_account/_fetch_okx_positions 会回退到
    # dashboard_api.fetch_okx_account() 发起真实请求，若真实账户可达会污染
    # "空数据" 断言。这里隔离真实 API，保证边界条件测试的确定性。
    with patch("dashboard_api.fetch_okx_account", return_value={}), \
         patch("dashboard_api.fetch_okx_positions", return_value=[]), \
         patch("dashboard_api._okx_circuit_open", return_value=True):
        engine_no_db = DashboardEngine({"dashboard": {"db_path": "data/nonexistent_test.db"}})
        snapshot = engine_no_db.get_account_snapshot(force_refresh=True)
        assert snapshot.total_equity == 0.0

        result = engine_no_db.get_position_distribution(force_refresh=True)
        assert result["positions"] == []

        curve = engine_no_db.get_equity_curve(days=1, force_refresh=True)
        assert isinstance(curve["points"], list)

        comp = engine_no_db.get_strategy_comparison(force_refresh=True)
        assert comp["strategies"] == []
    print("  [PASS] test_12: 边界条件通过")


def test_13_config_hot_update():
    """测试13: 热更新配置"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    assert engine._db_path.endswith("data/test_dashboard_v3.db")
    engine.update_config({"dashboard": {"db_path": "data/new_test.db"}})
    assert engine._db_path.endswith("data/new_test.db")
    print("  [PASS] test_13: 热更新配置成功")


def test_14_snapshot_persistence():
    """测试14: 快照持久化"""
    db_path = "data/test_dashboard_v3.db"
    engine = DashboardEngine({"dashboard": {"db_path": db_path}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    snapshot = engine.get_account_snapshot(force_refresh=True)
    engine._last_snapshot_persist = 0
    engine._maybe_persist_snapshot(snapshot)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) as cnt FROM account_history")
    row = cursor.fetchone()
    count = row["cnt"]
    conn.close()
    assert count > 0
    print(f"  [PASS] test_14: 快照持久化成功, DB中 {count} 条记录")


# ═══════════════════════════════════════════════════════════════
# V3.0 新增测试
# ═══════════════════════════════════════════════════════════════

def test_15_historical_performance():
    """测试15: 历史表现统计 (V3.0)"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    result = engine.get_historical_performance(days=30, force_refresh=True)
    assert result is not None
    assert "period" in result
    assert "sharpe_ratio" in result
    assert "calmar_ratio" in result
    assert "sortino_ratio" in result
    assert "win_rate" in result
    assert "profit_factor" in result
    assert "total_trades" in result
    assert "volatility_annual" in result
    print(f"  [PASS] test_15: 历史表现统计成功, 夏普={result.get('sharpe_ratio', 'N/A')}")


def test_16_capital_efficiency():
    """测试16: 资金效率指标 (V3.0)"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    result = engine.get_capital_efficiency(force_refresh=True)
    assert result is not None
    assert "total_capital" in result
    assert "deployed_capital" in result
    assert "idle_capital" in result
    assert "efficiency_ratio" in result
    assert "base_pool_usage" in result
    assert "idle_capital_cost" in result
    assert "rebalance_needed" in result
    print(f"  [PASS] test_16: 资金效率指标成功, 效率={result.get('efficiency_ratio', 'N/A')}")


def test_17_active_alerts():
    """测试17: 活跃告警检测 (V3.0)"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    result = engine.get_active_alerts(force_refresh=True)
    assert result is not None
    assert "alerts" in result
    assert "summary" in result
    assert "total" in result["summary"]
    assert "critical" in result["summary"]
    print(f"  [PASS] test_17: 活跃告警检测成功, 告警数={result['summary']['total']}")


def test_18_funding_trend():
    """测试18: 资金费率趋势 (V3.0)"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    result = engine.get_funding_trend(force_refresh=True)
    assert result is not None
    assert "items" in result
    assert "summary" in result
    assert "total_daily_cost" in result["summary"]
    assert "total_weekly_cost" in result["summary"]
    print(f"  [PASS] test_18: 资金费率趋势成功, 日费用={result['summary']['total_daily_cost']}")


def test_19_orderbook_summary():
    """测试19: 订单簿摘要 (V3.0)"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    result = engine.get_orderbook_summary(force_refresh=True)
    assert result is not None
    assert "items" in result
    assert "timestamp" in result
    print(f"  [PASS] test_19: 订单簿摘要成功, {len(result['items'])} 个币种")


def test_20_strategy_heatmap():
    """测试20: 策略热力图 (V3.0)"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    result = engine.get_strategy_heatmap(days=30, force_refresh=True)
    assert result is not None
    assert "cells" in result
    if result["cells"]:
        cell = result["cells"][0]
        assert "strategy" in cell
        assert "total_pnl" in cell
        assert "win_rate" in cell
        assert "status" in cell
    print(f"  [PASS] test_20: 策略热力图成功, {len(result['cells'])} 个单元")


def test_21_market_overview():
    """测试21: 市场概览 (V3.0)"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    engine.set_dependencies(okx_client=mock_client)

    result = engine.get_market_overview(force_refresh=True)
    assert result is not None
    assert "btc_price" in result
    assert "eth_price" in result
    assert "market_regime" in result
    print(f"  [PASS] test_21: 市场概览成功, BTC=${result.get('btc_price', 'N/A')}, 状态={result.get('market_regime', 'N/A')}")


def test_22_full_dashboard_v3():
    """测试22: 全量数据包含V3.0模块"""
    engine = DashboardEngine({"dashboard": {"db_path": "data/test_dashboard_v3.db"}})
    mock_client = MockOKXClient()
    mock_state = MockStateManager()
    engine.set_dependencies(okx_client=mock_client, state_manager=mock_state)

    result = engine.get_dashboard_full()
    # V2.1 模块
    assert "account" in result
    assert "equity_curve" in result
    assert "strategy_comparison" in result
    assert "position_distribution" in result
    assert "risk_dashboard" in result
    # V3.0 模块
    assert "historical_performance" in result
    assert "capital_efficiency" in result
    assert "active_alerts" in result
    assert "funding_trend" in result
    assert "orderbook_summary" in result
    assert "strategy_heatmap" in result
    assert "market_overview" in result
    print("  [PASS] test_22: 全量V3.0数据获取成功")


def test_23_calc_position_risk():
    """测试23: 单仓位风险评分"""
    engine = DashboardEngine()
    score = engine._calc_position_risk(60000, 65000, 500, 5000, 100, 3)
    assert score < 35
    score = engine._calc_position_risk(63000, 65000, 2000, 5000, -500, 15)
    assert score >= 30
    print("  [PASS] test_23: 风险评分计算成功")


def test_24_data_model_to_dict():
    """测试24: 数据模型序列化"""
    engine = DashboardEngine()
    snapshot = AccountSnapshot(total_equity=5000.0, health_level="normal")
    d = engine._snapshot_to_dict(snapshot)
    assert d["total_equity"] == 5000.0
    assert d["health_level"] == "normal"
    print("  [PASS] test_24: 数据模型序列化成功")


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  生产级仪表板 V3.0 集成测试")
    print("=" * 60)

    test_db = "data/test_dashboard_v3.db"
    create_test_db(test_db)
    print(f"\n[INFO] 测试数据库已创建: {test_db}")

    tests = [
        test_01_engine_initialization,
        test_02_data_models,
        test_03_dependency_injection,
        test_04_account_snapshot,
        test_05_position_distribution,
        test_06_risk_dashboard,
        test_07_equity_curve,
        test_08_strategy_comparison,
        test_09_cache_mechanism,
        test_10_strategy_inference,
        test_11_full_dashboard,
        test_12_edge_cases,
        test_13_config_hot_update,
        test_14_snapshot_persistence,
        test_15_historical_performance,
        test_16_capital_efficiency,
        test_17_active_alerts,
        test_18_funding_trend,
        test_19_orderbook_summary,
        test_20_strategy_heatmap,
        test_21_market_overview,
        test_22_full_dashboard_v3,
        test_23_calc_position_risk,
        test_24_data_model_to_dict,
    ]

    passed = 0
    failed = 0

    print()
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  结果: {passed}/{len(tests)} 通过, {failed} 失败")
    print(f"{'='*60}")

    # 清理测试数据库
    try:
        if os.path.exists(test_db):
            os.remove(test_db)
        if os.path.exists(test_db + "-journal"):
            os.remove(test_db + "-journal")
        print("[INFO] 测试数据库已清理")
    except Exception:
        pass

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)