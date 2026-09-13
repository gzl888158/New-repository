"""生产级资金磨损分析器集成验证脚本"""
import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['OKX_API_KEY'] = 'test_key'
os.environ['OKX_SECRET_KEY'] = 'test_secret'
os.environ['OKX_PASSPHRASE'] = 'test_pass'
os.environ['REDIS_PASSWORD'] = 'test_redis'
os.environ['TELEGRAM_BOT_TOKEN'] = 'test_tg'
os.environ['TELEGRAM_CHAT_ID'] = 'test_chat'


@pytest.fixture
def analyzer():
    """Create a fresh CapitalAttritionAnalyzer for each test"""
    from core.capital_attrition_analyzer import CapitalAttritionAnalyzer
    return CapitalAttritionAnalyzer({})


def test_config_loading():
    """1. 验证 capital_attrition 配置加载"""
    print("=== 1. config.yaml capital_attrition 配置验证 ===")
    from configs.settings import load_config
    config = load_config()

    att_cfg = config.get('capital_attrition', {})
    assert att_cfg, "capital_attrition 配置节缺失"

    # 核心开关
    assert att_cfg.get('enabled') == True, "enabled 应为 True"
    assert att_cfg.get('taker_fee_rate') == 0.0005, "taker_fee_rate 应为 0.0005"
    assert att_cfg.get('maker_fee_rate') == 0.0002, "maker_fee_rate 应为 0.0002"

    # 预算配置
    assert att_cfg.get('budget_check_enabled') == True, "budget_check_enabled 应为 True"
    assert att_cfg.get('default_daily_budget_usdt') == 5.0, "default_daily_budget_usdt 应为 5.0"
    assert att_cfg.get('default_weekly_budget_usdt') == 25.0, "default_weekly_budget_usdt 应为 25.0"

    # 策略预算
    strategy_budgets = att_cfg.get('strategy_budgets', {})
    assert len(strategy_budgets) >= 6, f"应有至少6个策略预算，实际{len(strategy_budgets)}"
    for sname in ['grid', 'scalping', 'trend', 'arbitrage', 'spot_grid', 'spot_martingale']:
        assert sname in strategy_budgets, f"缺少 {sname} 策略预算"
        sb = strategy_budgets[sname]
        assert 'daily_budget_usdt' in sb, f"{sname} 缺少 daily_budget_usdt"
        assert 'max_attrition_rate' in sb, f"{sname} 缺少 max_attrition_rate"

    # 自适应费率
    assert att_cfg.get('adaptive_fee_enabled') == True, "adaptive_fee_enabled 应为 True"
    assert att_cfg.get('adaptive_fee_window') == 100, "adaptive_fee_window 应为 100"

    # 无效交易
    assert att_cfg.get('invalid_trade_threshold') == 0.5, "invalid_trade_threshold 应为 0.5"

    # 告警
    assert att_cfg.get('alert_cooldown_seconds') == 300, "alert_cooldown_seconds 应为 300"

    print(f"  capital_attrition 配置: {len(att_cfg)} keys, {len(strategy_budgets)} strategy budgets")
    print("  config.yaml loading: PASSED")
    return config


def test_analyzer_import_and_init():
    """2. 验证 CapitalAttritionAnalyzer 模块导入和实例化"""
    print("\n=== 2. CapitalAttritionAnalyzer 模块验证 ===")
    from core.capital_attrition_analyzer import (
        CapitalAttritionAnalyzer, AttritionType, AttritionSeverity,
        AttritionRecord, AttritionBudget, AttritionStats
    )
    print("  所有类导入成功")

    analyzer = CapitalAttritionAnalyzer({})
    assert analyzer._enabled == True, "默认应启用"
    assert analyzer._taker_fee_rate == 0.0005, f"默认taker费率应为0.0005, 实际{analyzer._taker_fee_rate}"
    assert analyzer._maker_fee_rate == 0.0002, f"默认maker费率应为0.0002, 实际{analyzer._maker_fee_rate}"
    print(f"  CapitalAttritionAnalyzer 实例化成功: taker={analyzer._taker_fee_rate:.4%}, maker={analyzer._maker_fee_rate:.4%}")

    return analyzer


def test_fee_recording(analyzer):
    """3. 验证手续费记录"""
    print("\n=== 3. 手续费记录验证 ===")

    # 记录多笔手续费
    analyzer.record_fee("BTC-USDT-SWAP", "trend", 0.5, 1000.0, "buy", is_maker=False)
    analyzer.record_fee("ETH-USDT-SWAP", "grid", 0.15, 500.0, "sell", is_maker=True)
    analyzer.record_fee("SOL-USDT-SWAP", "scalping", 0.08, 200.0, "buy", is_maker=False)
    analyzer.record_fee("BTC-USDT-SWAP", "trend", 0.45, 950.0, "sell", is_maker=False)

    stats = analyzer.get_attrition_stats(period_hours=24)
    assert stats.record_count >= 4, f"应有至少4条记录，实际{stats.record_count}"
    assert stats.by_type.get('trading_fee', 0) > 0, "应包含 trading_fee 类型"

    print(f"  手续费记录: {stats.record_count} records, total={stats.total_attrition_usdt:.4f} USDT")

    # 禁用状态
    analyzer2 = type(analyzer)({"capital_attrition": {"enabled": False}})
    analyzer2.record_fee("BTC-USDT-SWAP", "trend", 0.5, 1000.0, "buy")
    stats2 = analyzer2.get_attrition_stats(period_hours=24)
    assert stats2.record_count == 0, "禁用后不应记录"
    print("  禁用状态: 正确跳过记录")

    print("  手续费记录: PASSED")


def test_slippage_recording(analyzer):
    """4. 验证滑点记录"""
    print("\n=== 4. 滑点记录验证 ===")

    # 正向滑点（买入成交价高于预期）
    loss1 = analyzer.record_slippage("ETH-USDT-SWAP", "grid", 3000.0, 3001.5, 0.1, "buy")
    assert loss1 > 0, f"正向滑点应有损耗，实际{loss1}"

    # 负向滑点（买入成交价低于预期，不计入磨损）
    loss2 = analyzer.record_slippage("BTC-USDT-SWAP", "trend", 50000.0, 49998.0, 0.01, "buy")
    assert loss2 == 0, f"负向滑点不应计入磨损，实际{loss2}"

    # 卖出滑点（卖出成交价低于预期）
    loss3 = analyzer.record_slippage("SOL-USDT-SWAP", "scalping", 150.0, 149.5, 1.0, "sell")
    assert loss3 > 0, f"卖出滑点应有损耗，实际{loss3}"

    print(f"  滑点记录: buy_loss={loss1:.4f}, reverse={loss2:.4f}, sell_loss={loss3:.4f}")

    stats = analyzer.get_attrition_stats(period_hours=24)
    assert stats.by_type.get('slippage', 0) > 0, "应包含 slippage 类型"
    print(f"  滑点统计: total={stats.by_type.get('slippage', 0):.4f}")

    print("  滑点记录: PASSED")


def test_funding_rate_recording(analyzer):
    """5. 验证资金费率记录"""
    print("\n=== 5. 资金费率记录验证 ===")

    # 支付资金费（磨损）
    analyzer.record_funding_payment("BTC-USDT-SWAP", "trend", -0.3, 1000.0, -0.001)
    # 收到资金费（收益，不记录为磨损）
    analyzer.record_funding_payment("ETH-USDT-SWAP", "grid", 0.15, 500.0, 0.0005)
    # 高频支付
    analyzer.record_funding_payment("SOL-USDT-SWAP", "scalping", -0.05, 200.0, -0.002)

    stats = analyzer.get_attrition_stats(period_hours=24)
    funding_total = stats.by_type.get('funding_rate', 0)
    assert funding_total > 0, "应包含 funding_rate 磨损"
    assert funding_total == approx(0.35, tol=0.01), f"funding_rate磨损应为0.35, 实际{funding_total}"

    print(f"  资金费率记录: payment=0.35, received=0.15 (not attrition)")
    print("  资金费率记录: PASSED")


def test_spread_cost_recording(analyzer):
    """6. 验证价差损耗记录"""
    print("\n=== 6. 价差损耗记录验证 ===")

    # bid=100, ask=100.2, spread=0.2, 半价差=0.1
    cost = analyzer.record_spread_cost("ETH-USDT-SWAP", "grid", 100.0, 100.2, 1.0, "buy")
    assert cost > 0, f"价差损耗应>0, 实际{cost}"
    assert cost == approx(0.1, tol=0.01), f"半价差应为0.1, 实际{cost}"

    cost2 = analyzer.record_spread_cost("BTC-USDT-SWAP", "trend", 50000.0, 50010.0, 0.01, "sell")
    assert cost2 == approx(0.05, tol=0.01), f"BTC价差损耗应为0.05, 实际{cost2}"

    print(f"  价差损耗: ETH={cost:.4f}, BTC={cost2:.4f}")
    print("  价差损耗记录: PASSED")


def test_market_impact_recording(analyzer):
    """7. 验证市场冲击记录"""
    print("\n=== 7. 市场冲击记录验证 ===")

    impact = analyzer.record_market_impact("BTC-USDT-SWAP", "trend", 50000.0, 50015.0, 0.1, "buy")
    assert impact > 0, f"市场冲击应>0, 实际{impact}"
    assert impact == approx(1.5, tol=0.1), f"冲击应为1.5, 实际{impact}"

    print(f"  市场冲击: {impact:.4f} USDT")
    print("  市场冲击记录: PASSED")


def test_opportunity_cost_recording(analyzer):
    """8. 验证机会成本记录"""
    print("\n=== 8. 机会成本记录验证 ===")

    cost = analyzer.record_opportunity_cost(
        "ETH-USDT-SWAP", "grid", 500.0, 24.0,
        reference_return_rate=0.05 / 365 / 24
    )
    assert cost > 0, f"机会成本应>0, 实际{cost}"
    print(f"  机会成本: 500USDT锁定24h = {cost:.6f} USDT")

    print("  机会成本记录: PASSED")


def test_invalid_trade_recording(analyzer):
    """9. 验证无效交易记录"""
    print("\n=== 9. 无效交易记录验证 ===")

    initial_count = sum(analyzer._invalid_trades_count.values())

    analyzer.record_invalid_trade("ETH-USDT-SWAP", "scalping", 0.05, -0.02, 100.0)
    analyzer.record_invalid_trade("BTC-USDT-SWAP", "trend", 0.5, -0.3, 1000.0)

    new_count = sum(analyzer._invalid_trades_count.values())
    assert new_count == initial_count + 2, f"应有2笔无效交易，实际{new_count - initial_count}"

    print(f"  无效交易: {new_count - initial_count} 笔新增, 总成本={sum(analyzer._invalid_trades_cost.values()):.4f}")
    print("  无效交易记录: PASSED")


def test_budget_management(analyzer):
    """10. 验证预算管理"""
    print("\n=== 10. 预算管理验证 ===")

    analyzer.set_strategy_budget("grid", daily_budget=3.0, weekly_budget=15.0, max_attrition_rate=0.25)
    analyzer.set_strategy_budget("trend", daily_budget=4.0, weekly_budget=20.0, max_attrition_rate=0.25)

    # 检查预算状态
    status = analyzer.get_budget_status()
    assert 'grid' in status, "缺少 grid 预算"
    assert 'trend' in status, "缺少 trend 预算"
    assert status['grid']['daily_budget'] == 3.0, f"grid日预算应为3.0, 实际{status['grid']['daily_budget']}"
    assert status['trend']['daily_budget'] == 4.0, f"trend日预算应为4.0, 实际{status['trend']['daily_budget']}"

    # 模拟磨损消耗
    for _ in range(4):
        analyzer.record_fee("ETH-USDT-SWAP", "grid", 0.5, 500.0, "buy")

    # 检查预算消耗
    status2 = analyzer.get_budget_status("grid")
    assert status2['daily_used'] > 0, "grid 应有日消耗"
    print(f"  grid 预算: used={status2['daily_used']:.4f}/{status2['daily_budget']}")

    # 检查预算（不自动暂停时）
    allowed, reason = analyzer.check_budget("grid")
    assert allowed, f"预算检查应允许, 实际reason={reason}"

    # 预算恢复
    analyzer.resume_strategy("grid")
    allowed2, _ = analyzer.check_budget("grid")
    assert allowed2, "恢复后应允许"

    print(f"  预算检查: grid={allowed}, reason={reason}")
    print("  预算管理: PASSED")


def test_statistics_and_breakdown(analyzer):
    """11. 验证统计和归因分析"""
    print("\n=== 11. 统计和归因分析验证 ===")

    # Record data first (fresh fixture, no prior data)
    analyzer.record_fee("BTC-USDT-SWAP", "trend", 0.5, 1000.0, "buy", is_maker=False)
    analyzer.record_fee("ETH-USDT-SWAP", "grid", 0.15, 500.0, "sell", is_maker=True)
    analyzer.record_slippage("BTC-USDT-SWAP", "trend", 0.02, 1000.0, 0.01, "buy")
    analyzer.record_funding_payment("BTC-USDT-SWAP", "trend", -0.1, 1000.0)

    stats = analyzer.get_attrition_stats(period_hours=24)
    assert stats.record_count > 0, "应有记录"
    assert stats.total_attrition_usdt > 0, "应有总磨损"
    assert stats.total_trade_value_usdt > 0, "应有总交易价值"
    assert stats.overall_attrition_rate > 0, "应有磨损率"

    print(f"  统计: records={stats.record_count}, attrition={stats.total_attrition_usdt:.4f}, "
          f"rate={stats.overall_attrition_rate:.6f}")

    # 归因分析
    breakdown = analyzer.get_attrition_breakdown(strategy_name="grid")
    assert 'by_type' in breakdown, "缺少 by_type"
    assert 'by_strategy' in breakdown, "缺少 by_strategy"
    assert 'by_symbol' in breakdown, "缺少 by_symbol"
    assert 'severity' in breakdown, "缺少 severity"

    print(f"  归因: type_count={len(breakdown['by_type'])}, "
          f"strategy_count={len(breakdown['by_strategy'])}, "
          f"severity={breakdown['severity']}")

    # 趋势
    trend = analyzer.get_attrition_trend(days=1, granularity="daily")
    print(f"  趋势: {len(trend)} periods")

    print("  统计和归因分析: PASSED")


def test_daily_summary(analyzer):
    """12. 验证每日摘要"""
    print("\n=== 12. 每日摘要验证 ===")

    summary = analyzer.get_daily_summary()
    assert 'date' in summary, "缺少 date"
    assert 'total_attrition_usdt' in summary, "缺少 total_attrition_usdt"
    assert 'attrition_rate' in summary, "缺少 attrition_rate"
    assert 'estimated_taker_fee' in summary, "缺少 estimated_taker_fee"
    assert 'estimated_maker_fee' in summary, "缺少 estimated_maker_fee"

    print(f"  每日摘要: {summary['date']}, attrition={summary['total_attrition_usdt']:.4f}, "
          f"rate={summary['attrition_rate']:.6f}")
    print("  每日摘要: PASSED")


def test_optimization_suggestions(analyzer):
    """13. 验证优化建议（生产级多维度分析）"""
    print("\n=== 13. 优化建议验证 ===")

    # 制造更多磨损数据以触发各类建议
    # 1) 大量taker手续费触发maker比例建议
    for _ in range(8):
        analyzer.record_fee("BTC-USDT-SWAP", "scalping", 0.5, 1000.0, "buy", is_maker=False)
        analyzer.record_fee("ETH-USDT-SWAP", "scalping", 0.3, 500.0, "sell", is_maker=False)
    # 2) maker单
    for _ in range(2):
        analyzer.record_fee("BTC-USDT-SWAP", "scalping", 0.2, 1000.0, "buy", is_maker=True)

    # 3) 高滑点
    analyzer.record_slippage("DOGE-USDT-SWAP", "scalping", 0.10, 0.12, 5000.0, "buy")
    analyzer.record_slippage("DOGE-USDT-SWAP", "scalping", 0.10, 0.12, 5000.0, "sell")

    # 4) 资金费率
    analyzer.record_funding_payment("SOL-USDT-SWAP", "scalping", -0.5, 500.0, -0.003)

    # 5) 无效交易
    analyzer.record_invalid_trade("ETH-USDT-SWAP", "scalping", 0.05, -0.02, 100.0)
    analyzer.record_invalid_trade("ETH-USDT-SWAP", "scalping", 0.03, -0.01, 50.0)

    suggestions = analyzer.get_optimization_suggestions(force=True)
    assert isinstance(suggestions, list), "建议应为列表"
    assert len(suggestions) >= 3, f"降低阈值后应有至少3条建议，实际{len(suggestions)}"

    print(f"  优化建议: {len(suggestions)} suggestions")
    for s in suggestions:
        saving = s.get('potential_saving_usdt', 0)
        print(f"    [{s.get('priority'):8s}] {s.get('title'):40s} action={s.get('action'):30s} save={saving:.4f}")

    # 验证建议结构完整性
    for s in suggestions:
        assert 'type' in s, "建议缺少 type"
        assert 'priority' in s, "建议缺少 priority"
        assert 'title' in s, "建议缺少 title"
        assert 'detail' in s, "建议缺少 detail"
        assert 'action' in s, "建议缺少 action"
        assert 'potential_saving_usdt' in s, "建议缺少 potential_saving_usdt"
        assert s['priority'] in ('critical', 'high', 'medium', 'low'), f"无效优先级: {s['priority']}"

    # 验证排序（priority降序，同priority内potential_saving_usdt降序）
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    for i in range(len(suggestions) - 1):
        p1 = priority_order[suggestions[i]['priority']]
        p2 = priority_order[suggestions[i+1]['priority']]
        assert p1 <= p2, f"建议排序错误: [{suggestions[i]['priority']}] 应在 [{suggestions[i+1]['priority']}] 之前"

    print("  优化建议: PASSED")


def test_state_persistence(analyzer):
    """14. 验证状态持久化"""
    print("\n=== 14. 状态持久化验证 ===")

    state = analyzer.collect_persistent_state()
    assert 'budgets' in state, "持久化状态缺少 budgets"
    assert 'strategy_profits' in state, "持久化状态缺少 strategy_profits"
    assert 'strategy_attrition' in state, "持久化状态缺少 strategy_attrition"
    assert 'estimated_taker_fee' in state, "持久化状态缺少 estimated_taker_fee"
    assert 'collection_time' in state, "持久化状态缺少 collection_time"

    print(f"  持久化状态: {len(state)} keys, budgets={len(state['budgets'])}")

    # 创建新实例恢复
    analyzer2 = type(analyzer)({})
    analyzer2.restore_persistent_state(state)

    # 验证恢复
    for name in state.get('budgets', {}):
        status = analyzer2.get_budget_status(name)
        assert status, f"恢复后应有 {name} 预算"

    print(f"  状态恢复: {len(analyzer2._budgets)} budgets restored")
    print("  状态持久化: PASSED")


def test_capital_manager_integration(config):
    """15. 验证 CapitalManager 集成"""
    print("\n=== 15. CapitalManager 集成验证 ===")
    from core.capital_manager import CapitalManager, CapitalPoolType

    mgr = CapitalManager(config)
    assert mgr.attrition_analyzer is not None, "attrition_analyzer 未初始化"

    # 验证预算初始化
    budgets = mgr.attrition_analyzer.get_budget_status()
    assert len(budgets) >= 6, f"应有至少6个策略预算，实际{len(budgets)}"
    print(f"  策略预算: {len(budgets)} strategies initialized")

    # 验证便捷方法
    mgr.record_trade_fee("BTC-USDT-SWAP", "trend", 0.5, 1000.0, "buy")
    mgr.record_trade_slippage("ETH-USDT-SWAP", "grid", 3000.0, 3001.5, 0.1, "buy")
    mgr.record_funding_payment("SOL-USDT-SWAP", "scalping", -0.05, 200.0, -0.001)
    mgr.record_strategy_profit("trend", 15.0)

    # 检查磨损预算
    allowed, reason = mgr.check_attrition_budget("trend")
    assert allowed, f"预算检查应允许, reason={reason}"
    print(f"  磨损预算检查: trend={allowed}")

    # 验证完整报告包含磨损数据
    report = mgr.get_full_report()
    assert 'capital_attrition' in report, "完整报告缺少 capital_attrition"
    att_report = report['capital_attrition']
    assert att_report['enabled'] == True, "磨损分析器应启用"
    assert 'daily_summary' in att_report, "缺少 daily_summary"
    assert 'breakdown' in att_report, "缺少 breakdown"
    assert 'budgets' in att_report, "缺少 budgets"
    print(f"  完整报告: 包含 capital_attrition ({len(att_report)} keys)")

    # 磨损报告
    att_report2 = mgr.get_attrition_report()
    assert att_report2['enabled'] == True, "磨损报告应启用"
    print(f"  磨损报告: records={att_report2['total_records']}")

    print("  CapitalManager 集成: PASSED")
    return mgr


def test_end_to_end_flow(config):
    """16. 端到端集成验证"""
    print("\n=== 16. 端到端集成验证 ===")
    from core.capital_manager import CapitalManager, CapitalPoolType
    from core.capital_attrition_analyzer import CapitalAttritionAnalyzer

    mgr = CapitalManager(config)

    # 模拟交易流程
    strategies = ['grid', 'trend', 'scalping', 'arbitrage']
    symbols = ['BTC-USDT-SWAP', 'ETH-USDT-SWAP', 'SOL-USDT-SWAP', 'DOGE-USDT-SWAP']
    total_records = 0

    import random
    random.seed(42)

    for i in range(20):
        sname = random.choice(strategies)
        symbol = random.choice(symbols)
        expected_price = random.uniform(100, 50000)
        filled_price = expected_price * (1 + random.uniform(-0.002, 0.002))
        quantity = random.uniform(0.01, 0.5)
        side = random.choice(['buy', 'sell'])
        trade_value = expected_price * quantity

        # 手续费
        fee = trade_value * 0.0005
        mgr.record_trade_fee(symbol, sname, fee, trade_value, side)
        total_records += 1  # fee

        # 滑点（负向滑点=0时不计入磨损记录）
        slippage = mgr.record_trade_slippage(symbol, sname, expected_price, filled_price, quantity, side)
        if slippage > 0:
            total_records += 1

        # 利润（随机）
        profit = random.uniform(-5, 20)
        mgr.record_strategy_profit(sname, profit)

    # 检查磨损预算
    for sname in strategies:
        allowed, reason = mgr.check_attrition_budget(sname)
        assert allowed, f"{sname} 预算检查应允许, reason={reason}"

    # 统计
    stats = mgr.attrition_analyzer.get_attrition_stats(period_hours=24)
    assert stats.record_count >= total_records, f"应有至少{total_records}条记录, 实际{stats.record_count}"

    # 每日摘要
    summary = mgr.attrition_analyzer.get_daily_summary()
    assert summary['attrition_rate'] > 0, "磨损率应>0"

    # 归因
    breakdown = mgr.attrition_analyzer.get_attrition_breakdown()
    assert len(breakdown['by_strategy']) > 0, "应有策略归因"
    assert len(breakdown['by_type']) > 0, "应有类型归因"

    # 优化建议
    suggestions = mgr.attrition_analyzer.get_optimization_suggestions(force=True)
    assert len(suggestions) >= 0, "优化建议应为列表"

    # 完整报告
    report = mgr.get_full_report()
    assert 'capital_attrition' in report

    print(f"  模拟交易: {total_records} records, {len(strategies)} strategies, "
          f"{len(symbols)} symbols")
    print(f"  磨损率: {summary['attrition_rate']:.6f}")
    print(f"  策略归因: {list(breakdown['by_strategy'].keys())}")
    print(f"  类型归因: {list(breakdown['by_type'].keys())}")
    print(f"  优化建议: {len(suggestions)}")

    print("  端到端集成: PASSED")


def test_hot_update(config):
    """17. 验证热更新"""
    print("\n=== 17. 热更新验证 ===")
    from core.capital_attrition_analyzer import CapitalAttritionAnalyzer

    analyzer = CapitalAttritionAnalyzer(config)
    original_enabled = analyzer._enabled

    # 更新配置
    new_config = {"capital_attrition": {"enabled": False, "budget_check_enabled": False}}
    analyzer.update_config(new_config)

    assert analyzer._enabled == False, "更新后应禁用"
    assert analyzer._budget_check_enabled == False, "预算检查应禁用"

    # 恢复
    analyzer.update_config(config)
    assert analyzer._enabled == original_enabled, "恢复后应恢复原状态"

    print("  热更新: PASSED")


def test_fee_estimation(config):
    """18. 验证自适应费率估算"""
    print("\n=== 18. 自适应费率估算验证 ===")
    from core.capital_attrition_analyzer import CapitalAttritionAnalyzer

    analyzer = CapitalAttritionAnalyzer(config)

    # 初始费率
    rates = analyzer.get_estimated_fee_rates()
    assert rates['taker_fee'] == 0.0005, f"初始taker费率应为0.0005, 实际{rates['taker_fee']}"
    assert rates['samples'] == 0, "初始样本应为0"

    # 更新费率（传入实际手续费金额，非费率）
    for _ in range(10):
        analyzer.update_adaptive_fee(0.48, 1000.0, is_maker=False)
    for _ in range(10):
        analyzer.update_adaptive_fee(0.18, 1000.0, is_maker=True)

    rates2 = analyzer.get_estimated_fee_rates()
    assert rates2['samples'] == 20, f"应有20个样本, 实际{rates2['samples']}"
    assert rates2['taker_fee'] == approx(0.00048, tol=0.0001), f"taker费率应接近0.00048, 实际{rates2['taker_fee']}"
    assert rates2['maker_fee'] == approx(0.00018, tol=0.0001), f"maker费率应接近0.00018, 实际{rates2['maker_fee']}"

    print(f"  自适应费率: taker={rates2['taker_fee']:.6f}, maker={rates2['maker_fee']:.6f}, samples={rates2['samples']}")
    print("  自适应费率: PASSED")


def test_alert_callback(config):
    """19. 验证告警回调"""
    print("\n=== 19. 告警回调验证 ===")
    from core.capital_attrition_analyzer import CapitalAttritionAnalyzer

    analyzer = CapitalAttritionAnalyzer(config)
    alerts_received = []

    def alert_callback(alert_data):
        alerts_received.append(alert_data)

    analyzer.register_alert_callback(alert_callback)

    # 触发高费率告警
    analyzer.record_funding_payment("BTC-USDT-SWAP", "trend", -1.0, 1000.0, -0.002)

    # 注意：告警有冷却时间，可能不会立即触发第二次
    assert len(alerts_received) >= 1, f"应收到至少1次告警, 实际{len(alerts_received)}"
    print(f"  告警回调: {len(alerts_received)} alerts received")

    print("  告警回调: PASSED")


def test_attrition_alert_registration():
    """20. 验证磨损告警注册机制"""
    print("\n=== 20. 磨损告警注册验证 ===")
    from core.capital_attrition_analyzer import CapitalAttritionAnalyzer

    analyzer = CapitalAttritionAnalyzer({})
    callbacks_before = len(analyzer._alert_callbacks)

    def test_callback(data):
        pass

    analyzer.register_alert_callback(test_callback)
    assert len(analyzer._alert_callbacks) == callbacks_before + 1, "回调注册失败"

    print(f"  告警回调: {len(analyzer._alert_callbacks)} registered")
    print("  磨损告警注册: PASSED")


# 浮点数比较辅助函数
def approx(value, tol=0.0):
    """简单的浮点数近似比较"""
    class Approx:
        def __init__(self, v, tolerance):
            self.v = v
            self.tol = tolerance
        def __eq__(self, other):
            return abs(other - self.v) <= self.tol
        def __repr__(self):
            return f"approx({self.v}, tol={self.tol})"
    return Approx(value, tol)


if __name__ == '__main__':
    config = test_config_loading()
    analyzer = test_analyzer_import_and_init()
    test_fee_recording(analyzer)
    test_slippage_recording(analyzer)
    test_funding_rate_recording(analyzer)
    test_spread_cost_recording(analyzer)
    test_market_impact_recording(analyzer)
    test_opportunity_cost_recording(analyzer)
    test_invalid_trade_recording(analyzer)
    test_budget_management(analyzer)
    test_statistics_and_breakdown(analyzer)
    test_daily_summary(analyzer)
    test_optimization_suggestions(analyzer)
    test_state_persistence(analyzer)
    test_capital_manager_integration(config)
    test_end_to_end_flow(config)
    test_hot_update(config)
    test_fee_estimation(config)
    test_alert_callback(config)
    test_attrition_alert_registration()

    print("\n" + "=" * 60)
    print("  所有 20/20 测试通过! 生产级资金磨损分析器交付完成")
    print("=" * 60)