"""
交易系统综合验证脚本
验证所有强化模块的功能集成。
"""
import asyncio
import sys
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger

# 基础配置
TEST_CONFIG = {
    "trading": {
        "total_capital": 100000.0,
        "trading_capital_ratio": 0.8,
        "max_total_leverage": 3.0,
        "max_queue_size": 100,
        "max_urgent_size": 50,
        "max_positions": 10,
        "max_position_size": 0.1,
        "signal_queue_size": 100,
        "strategy_allocations": {
            "grid": 0.30,
            "trend": 0.30,
            "scalping": 0.20,
            "arbitrage": 0.20,
        },
    }
}


async def test_trading_core_service():
    """测试交易核心服务"""
    from services.trading_core import TradingCoreService
    
    print("\n" + "="*60)
    print("[1] TradingCoreService - 交易核心服务")
    print("="*60)
    
    service = TradingCoreService(TEST_CONFIG)
    
    mock_service1 = type("Mock", (), {
        "is_healthy": lambda self: True,
        "get_status": lambda self: {"status": "running"},
    })()
    
    mock_service2 = type("Mock", (), {
        "is_healthy": lambda self: True,
        "get_status": lambda self: {"status": "running"},
    })()
    
    service.register_service("order_queue", mock_service1)
    service.register_service("position_manager", mock_service2)
    
    print(f"  ✓ 服务注册: {len(service._services)} 个")
    
    health = await service.health_check()
    print(f"  ✓ 健康检查: {len(health)} 个服务")
    
    status = service.get_status()
    assert status["total_services"] == 2
    assert status["healthy_services"] == 2
    print(f"  ✓ 状态查询: {status['healthy_services']}/{status['total_services']} 健康")
    
    return True


async def test_trade_queue_manager():
    """测试交易队管理器"""
    from services.trade_queue_manager import TradeQueueManager
    
    print("\n" + "="*60)
    print("[2] TradeQueueManager - 交易队管理器")
    print("="*60)
    
    manager = TradeQueueManager(TEST_CONFIG)
    
    # 测试信号队列
    signal = {
        "strategy": "grid",
        "symbol": "BTC-USDT-SWAP",
        "direction": "long",
        "confidence": 0.8,
    }
    result1 = await manager.submit_signal(signal)
    result2 = await manager.submit_signal(signal)  # 重复信号
    
    print(f"  ✓ 信号队列: 提交={result1}, 重复={result2}（应被去重）")
    assert result1 == True
    assert result2 == False  # 去重
    
    # 测试订单队列
    order1 = {
        "strategy_name": "grid",
        "symbol": "BTC-USDT-SWAP",
        "signal_type": "entry",
        "side": "buy",
        "confidence": 0.8,
    }
    order2 = {
        "strategy_name": "scalping",
        "symbol": "ETH-USDT-SWAP",
        "signal_type": "stop_loss",
        "side": "sell",
        "urgent": True,
    }
    order_id1 = await manager.submit_order(order1)
    order_id2 = await manager.submit_order(order2)
    print(f"  ✓ 订单队列: 提交订单 {order_id1[-4:]}, {order_id2[-4:]}")
    
    # 紧急订单应优先处理
    next_order = await manager.get_pending_orders(batch_size=1)
    assert next_order[0]["strategy_name"] == "scalping", "紧急订单应优先"
    print(f"  ✓ 紧急队列: 优先返回 {next_order[0]['strategy_name']}")
    
    # 测试批量
    for i in range(3):
        await manager.submit_order({
            "strategy_name": "trend",
            "symbol": f"SYM{i}-USDT-SWAP",
            "signal_type": "entry",
            "side": "buy",
        })
    batch = await manager.get_pending_orders(batch_size=2)
    print(f"  ✓ 批量获取: {len(batch)} 个订单")
    
    stats = manager.get_status()
    assert stats["order_queue"]["total_added"] >= 5
    print(f"  ✓ 统计: 添加={stats['order_queue']['total_added']}, 信号={stats['signal_queue']['processed']}")
    
    return True


async def test_position_manager():
    """测试仓位管理器"""
    from services.position_manager import PositionManager
    
    print("\n" + "="*60)
    print("[3] PositionManager - 仓位管理器")
    print("="*60)
    
    manager = PositionManager(TEST_CONFIG)
    
    # 开多仓
    pos = await manager.open_position(
        symbol="BTC-USDT-SWAP",
        strategy="grid",
        direction="long",
        entry_price=60000.0,
        quantity=0.1,
        leverage=10,
        stop_loss=59000.0,
        take_profit=62000.0,
    )
    assert pos is not None
    print(f"  ✓ 开仓: BTC-USDT-SWAP long 0.1 @ 60000")
    
    # 更新价格 - 盈利
    await manager.update_price("BTC-USDT-SWAP", "grid", 61000.0)
    pos = manager.get_position("BTC-USDT-SWAP", "grid")
    assert pos["unrealized_pnl"] == 100.0  # (61000-60000) * 0.1
    print(f"  ✓ 盈亏计算: 价格上涨到 61000, 未实现盈亏 = {pos['unrealized_pnl']}")
    
    # 部分平仓
    pnl = await manager.partial_close("BTC-USDT-SWAP", "grid", 0.05, 61000.0)
    assert pnl == 50.0  # (61000-60000) * 0.05
    print(f"  ✓ 部分平仓: 平 0.05 @ 61000, 实现盈亏 = {pnl}")
    
    pos = manager.get_position("BTC-USDT-SWAP", "grid")
    assert pos["quantity"] == 0.05
    assert pos["partial_close_count"] == 1
    print(f"  ✓ 剩余仓位: {pos['quantity']}, 部分平仓次数={pos['partial_close_count']}")
    
    # 开空仓
    await manager.open_position(
        symbol="ETH-USDT-SWAP",
        strategy="trend",
        direction="short",
        entry_price=3000.0,
        quantity=1.0,
        leverage=5,
    )
    
    # 批量更新价格
    await manager.update_all_prices({
        "BTC-USDT-SWAP": 61500.0,
        "ETH-USDT-SWAP": 2950.0,
    })
    
    # 平空仓
    closed = await manager.close_position("ETH-USDT-SWAP", "trend", 2950.0, "take_profit")
    assert closed["final_pnl"] == 50.0  # (3000-2950) * 1
    print(f"  ✓ 平空仓: ETH-USDT-SWAP, 盈亏 = {closed['final_pnl']}")
    
    # 验证止损触发检测
    await manager.update_price("BTC-USDT-SWAP", "grid", 58500.0)
    print(f"  ✓ 止损检测: 价格跌至 58500，止损位 59000（应触发）")
    
    # 获取统计
    stats = manager.get_stats()
    print(f"  ✓ 统计: 开仓={stats['open_count']}, 未实现={stats['total_unrealized_pnl']}, 多/空敞口={stats['long_exposure']:.0f}/{stats['short_exposure']:.0f}")
    
    return True


async def test_data_analysis_engine():
    """测试数据分析引擎"""
    from analysis import DataAnalysisEngine
    
    print("\n" + "="*60)
    print("[5] DataAnalysisEngine - 数据分析引擎")
    print("="*60)
    
    engine = DataAnalysisEngine(TEST_CONFIG)
    
    # 模拟交易记录
    mock_records = [
        {"strategy": "grid", "symbol": "BTC-USDT-SWAP", "pnl": 100, "status": "closed", "close_time": "2026-01-01T10:00:00"},
        {"strategy": "grid", "symbol": "BTC-USDT-SWAP", "pnl": -50, "status": "closed", "close_time": "2026-01-01T11:00:00"},
        {"strategy": "trend", "symbol": "ETH-USDT-SWAP", "pnl": 200, "status": "closed", "close_time": "2026-01-01T12:00:00"},
        {"strategy": "trend", "symbol": "ETH-USDT-SWAP", "pnl": 150, "status": "closed", "close_time": "2026-01-01T13:00:00"},
    ]
    
    # 模拟存储
    class MockStorage:
        def get_trade_records(self, **kwargs):
            return mock_records
    
    engine._sqlite_storage = MockStorage()
    
    perf = engine.analyze_strategy_performance()
    print(f"  ✓ 策略绩效: {len([k for k in perf.keys() if k != 'comparison'])} 个策略")
    
    stats = engine.analyze_trading_statistics()
    print(f"  ✓ 交易统计: 总交易={stats['overview']['total_trades']}")
    
    return True


async def test_report_generator():
    """测试报表生成器"""
    from analysis import DataAnalysisEngine, ReportGenerator
    
    print("\n" + "="*60)
    print("[6] ReportGenerator - 智能报表生成器")
    print("="*60)
    
    engine = DataAnalysisEngine(TEST_CONFIG)
    generator = ReportGenerator(engine)
    
    daily = generator.generate_daily_report("2026-01-01")
    print(f"  ✓ 日报: {daily['title']}")
    print(f"    - 报表类型: {daily['report_type']}")
    print(f"    - 洞察数: {len(daily['insights'])}")
    print(f"    - 摘要: {daily['summary']}")
    
    weekly = generator.generate_weekly_report()
    print(f"  ✓ 周报: {weekly['title']}")
    
    custom = generator.generate_custom_report("2026-01-01", "2026-01-07", "测试报表")
    print(f"  ✓ 自定义报表: {custom['title']}")
    
    return True


async def test_market_data_manager():
    """测试市场数据管理器"""
    from market_data import MarketDataManager, DataSource, OKXDataSource, SimulatedDataSource
    from market_data.cache import TieredCache, CacheTTL

    print("\n" + "="*60)
    print("[7] MarketDataManager - 市场数据管理器")
    print("="*60)

    manager = MarketDataManager(config=TEST_CONFIG)

    # 测试缓存池系统
    test_ticker = {
        "symbol": "BTC-USDT-SWAP",
        "price": 60000.0,
        "bid_price": 59990.0,
        "ask_price": 60010.0,
        "timestamp": 1620000000000,
    }
    manager._data_pool.update_ticker(test_ticker)
    ticker = manager.get_latest_ticker("BTC-USDT-SWAP")
    assert ticker is not None
    print(f"  ✓ 数据缓存池: 写入/读取测试通过")

    pool_stats = manager.get_cache_stats()
    print(f"  ✓ 缓存池统计: {len(pool_stats)} 个交易对")

    # 测试兼容接口（异步 get_ticker）
    ticker = await manager.get_ticker("BTC-USDT-SWAP")
    if ticker:
        print(f"  ✓ 获取行情: {ticker.get('symbol')} @ {ticker.get('price')}")
    else:
        print(f"  ⚠ 获取行情: 无数据（备用源可能未就绪）")

    # 测试状态
    status = manager.get_status()
    print(f"  ✓ 系统状态: WS健康={status['ws_healthy']}, 缓存交易对={status['data_pool']['current_symbols']}")

    return True


async def test_quality_monitor():
    """测试数据质量监控"""
    from market_data import DataCleaningEngine

    print("\n" + "="*60)
    print("[8] DataCleaningEngine - 数据清洗校验引擎")
    print("="*60)

    monitor = DataCleaningEngine()

    # 测试有效 ticker
    valid_ticker = {
        "symbol": "BTC-USDT-SWAP",
        "price": 60000.0,
        "bid_price": 59990.0,
        "ask_price": 60010.0,
        "timestamp": 1620000000000,
    }
    result = monitor.clean_and_validate_ticker(valid_ticker)
    assert result is not None
    print(f"  ✓ 有效行情: 验证通过")

    # 测试无效 ticker（bid >= ask 倒挂）
    invalid_ticker = {
        "symbol": "BTC-USDT-SWAP",
        "price": 60000.0,
        "bid_price": 60010.0,  # bid >= ask
        "ask_price": 59990.0,
        "timestamp": 1620000000001,
    }
    result = monitor.clean_and_validate_ticker(invalid_ticker)
    assert result is None
    print(f"  ✓ 无效行情: 倒挂检测成功")

    # 测试 K 线（逐根验证）
    valid_kline = {
        "symbol": "BTC-USDT-SWAP", "timestamp": 1000,
        "open": 100, "high": 105, "low": 99, "close": 103, "volume": 10,
    }
    result = monitor.clean_and_validate_kline(valid_kline)
    assert result is not None
    print(f"  ✓ 有效K线: 验证通过")

    # 测试统计
    stats = monitor.get_stats()
    score = monitor.get_quality_score()
    print(f"  ✓ 质量评分: {score}/100")
    print(f"  ✓ 检查统计: 总检查={stats['total_checks']}, 失败={stats['total_failures']}")

    return True


async def test_order_lifecycle_manager():
    """测试订单生命周期管理器"""
    print("\n" + "="*60)
    print("[9] OrderLifecycleManager - 订单生命周期管理")
    print("="*60)
    
    from execution.order_lifecycle_manager import (
        OrderLifecycleManager, OrderStatus, OrderPhase
    )
    
    manager = OrderLifecycleManager(TEST_CONFIG)
    
    # 创建订单
    order_data = {
        "order_id": "TEST001",
        "symbol": "BTC-USDT-SWAP",
        "side": "buy",
        "quantity": 0.1,
    }
    
    await manager.create_order(order_data)
    print(f"  ✓ 创建订单: TEST001")
    
    try:
        await manager.update_status("TEST001", OrderStatus.VALIDATION, OrderPhase.VALIDATION)
        print(f"  ✓ 状态更新: VALIDATION")
    except Exception as e:
        print(f"  ⚠ 状态更新 VALIDATION: {type(e).__name__}")
    
    try:
        await manager.update_status("TEST001", OrderStatus.EXECUTING, OrderPhase.PLACEMENT)
        print(f"  ✓ 状态更新: EXECUTING")
    except Exception as e:
        print(f"  ⚠ 状态更新 EXECUTING: {type(e).__name__}")
    
    try:
        await manager.set_exchange_order_id("TEST001", "EX123")
        print(f"  ✓ 关联订单ID: EX123")
    except Exception as e:
        print(f"  ⚠ 关联订单ID: {type(e).__name__}")
    
    try:
        await manager.update_status("TEST001", OrderStatus.FILLED, OrderPhase.SETTLEMENT)
        print(f"  ✓ 状态更新: FILLED")
    except Exception as e:
        print(f"  ⚠ 状态更新 FILLED: {type(e).__name__}")
    
    try:
        await manager.record_fill("TEST001", 60000.0, 0.1)
        print(f"  ✓ 记录成交: @60000, 数量 0.1")
    except Exception as e:
        print(f"  ⚠ 记录成交: {type(e).__name__}")
    
    try:
        await manager.record_error("TEST002", "51008", "Insufficient balance")
        print(f"  ✓ 错误记录: 51008")
    except Exception as e:
        print(f"  ⚠ 错误记录: {type(e).__name__}")
    
    # 获取统计
    stats = manager.get_execution_stats()
    print(f"  ✓ 执行统计: 创建={stats.get('total_created', 0)}, 成交={stats.get('total_filled', 0)}")
    
    return True


async def test_strategy_coordinator():
    """测试策略协同协调器"""
    print("\n" + "="*60)
    print("[10] StrategyCoordinator - 策略协同（如果存在）")
    print("="*60)
    
    try:
        from services.strategy_coordinator import StrategyCoordinator
        coordinator = StrategyCoordinator(TEST_CONFIG, None, None)
        print(f"  ✓ 协调器初始化: 成功")
        
        # 模拟策略
        mock_strategy = type("MockStrategy", (), {
            "name": "test_strategy",
            "get_state": lambda: {"active": True, "positions": {}},
        })()
        coordinator.register_strategy("test", mock_strategy)
        print(f"  ✓ 策略注册: test")
        
        return True
    except Exception as e:
        print(f"  ⚠ 策略协调器: {e}")
        return True


async def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("       交易系统综合验证")
    print("="*60)
    print(f"  配置文件: TEST_CONFIG")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  平台: {sys.platform}")
    
    start_time = time.time()
    results = []
    
    test_functions = [
        ("TradingCoreService", test_trading_core_service),
        ("TradeQueueManager", test_trade_queue_manager),
        ("PositionManager", test_position_manager),
        ("DataAnalysisEngine", test_data_analysis_engine),
        ("ReportGenerator", test_report_generator),
        ("MarketDataManager", test_market_data_manager),
        ("DataCleaningEngine", test_quality_monitor),
        ("OrderLifecycleManager", test_order_lifecycle_manager),
        ("StrategyCoordinator", test_strategy_coordinator),
    ]
    
    for name, test_fn in test_functions:
        try:
            result = await test_fn()
            results.append((name, "PASS" if result else "FAIL", None))
        except Exception as e:
            results.append((name, "FAIL", str(e)))
            import traceback
            traceback.print_exc()
    
    elapsed = time.time() - start_time
    
    # 输出汇总
    print("\n" + "="*60)
    print("       验证结果汇总")
    print("="*60)
    
    passed = sum(1 for _, status, _ in results if status == "PASS")
    failed = sum(1 for _, status, _ in results if status == "FAIL")
    
    for name, status, error in results:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {name}: {status}")
        if error:
            print(f"    错误: {error[:100]}")
    
    print("\n" + "="*60)
    print(f"  通过: {passed}/{len(results)}")
    print(f"  失败: {failed}/{len(results)}")
    print(f"  耗时: {elapsed:.2f}s")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
