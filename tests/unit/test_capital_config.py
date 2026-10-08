"""生产级资金分布配置验证脚本"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 设置测试环境变量
os.environ['OKX_API_KEY'] = 'test_key'
os.environ['OKX_SECRET_KEY'] = 'test_secret'
os.environ['OKX_PASSPHRASE'] = 'test_pass'
os.environ['REDIS_PASSWORD'] = 'test_redis'
os.environ['TELEGRAM_BOT_TOKEN'] = 'test_tg'
os.environ['TELEGRAM_CHAT_ID'] = 'test_chat'

def test_config_loading():
    """1. 验证 config.yaml 加载"""
    print("=== 1. config.yaml 加载验证 ===")
    from configs.settings import load_config
    config = load_config()
    
    sections = [
        'capital_pool', 'allocation_agent', 'symbol_allocation',
        'leverage_tiers', 'pnl_reallocation', 'hedge_scheduler',
        'volatility_targeting', 'adaptive_kelly', 'capital_efficiency'
    ]
    for s in sections:
        val = config.get(s)
        status = "OK" if val else "MISSING!"
        count = len(val) if val else 0
        print(f"  {s}: {status} ({count} keys)")
    
    # 验证关键值
    cp = config.get('capital_pool', {})
    assert cp.get('base_ratio') == 0.60, f"base_ratio mismatch: {cp.get('base_ratio')}"
    assert cp.get('risk_isolation_ratio') == 0.15, f"risk_isolation_ratio mismatch"
    
    aa = config.get('allocation_agent', {})
    assert aa.get('method') == 'dynamic', f"method mismatch: {aa.get('method')}"
    assert aa.get('enabled') == True, f"enabled mismatch"
    
    lt = config.get('leverage_tiers', {})
    assert lt.get('absolute_max') == 10, f"absolute_max mismatch"
    
    vt = config.get('volatility_targeting', {})
    assert vt.get('target_volatility') == 0.20, f"target_volatility mismatch"
    
    ak = config.get('adaptive_kelly', {})
    assert ak.get('max_kelly_fraction') == 0.35, f"max_kelly_fraction mismatch"  # R53: 0.25→0.35
    
    ce = config.get('capital_efficiency', {})
    assert ce.get('idle_cash_threshold') == 0.15, f"idle_cash_threshold mismatch"
    
    print("  config.yaml loading: PASSED")
    return config

def test_capital_manager_import(config):
    """2. 验证 CapitalManager 模块导入和实例化"""
    print("\n=== 2. CapitalManager 模块验证 ===")
    from core.capital_manager import (
        CapitalManager, CapitalPoolController, CapitalPool, CapitalPoolType,
        SymbolWeightAllocator, LeverageTierManager, LeverageTier, LeverageAssignment,
        PnLReallocationUnit, HedgeScheduler, HedgeType, HedgePosition,
        get_capital_manager
    )
    print("  所有类导入成功")
    
    # 实例化
    mgr = CapitalManager(config)
    print("  CapitalManager 实例化成功")
    
    # 初始化
    symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
    mgr.initialize(symbols, 5000.0)
    print(f"  CapitalManager 初始化成功: {len(symbols)} symbols")
    
    # 验证模块
    assert mgr.capital_pool is not None, "capital_pool 未初始化"
    assert mgr.symbol_allocator is not None, "symbol_allocator 未初始化"
    assert mgr.leverage_manager is not None, "leverage_manager 未初始化"
    assert mgr.pnl_reallocation is not None, "pnl_reallocation 未初始化"
    assert mgr.hedge_scheduler is not None, "hedge_scheduler 未初始化"
    
    # 资金池操作
    mgr.allocate_capital("BTC-USDT-SWAP", 100.0, CapitalPoolType.BASE)
    pool = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    assert pool.used_amount > 0, "资金分配失败"
    print(f"  资金池分配: BASE pool used={pool.used_amount:.2f}")
    
    # 权重分配
    mgr.update_symbol_metrics("BTC-USDT-SWAP", 0.02, 0.01, 0.8, 10.0, 0.6)
    weight = mgr.get_symbol_weight("BTC-USDT-SWAP")
    assert weight > 0, "权重获取失败"
    print(f"  币种权重: BTC={weight:.4f}")
    
    # 杠杆分配
    lev = mgr.assign_leverage("BTC-USDT-SWAP", "initial", 0.02, 0.6, 0.0)
    assert lev.leverage > 0, "杠杆分配失败"
    print(f"  杠杆分配: BTC={lev.leverage}x, tier={lev.tier.value}")
    
    # 对冲评估
    hedge = mgr.evaluate_hedge("BTC-USDT-SWAP", "long", 0.1, 50000, 49000, 0.05, -0.06)
    if hedge:
        print(f"  对冲建议: side={hedge['hedge_side']}, ratio={hedge['hedge_ratio']}")
    else:
        print(f"  对冲评估: 无需对冲")
    
    # 完整报告
    report = mgr.get_full_report()
    assert 'capital_pool' in report, "报告缺少 capital_pool"
    assert 'symbol_weights' in report, "报告缺少 symbol_weights"
    assert 'leverage_tiers' in report, "报告缺少 leverage_tiers"
    assert 'pnl_reallocation' in report, "报告缺少 pnl_reallocation"
    assert 'hedge_scheduler' in report, "报告缺少 hedge_scheduler"
    print("  完整报告生成: 5/5 模块")
    
    print("  CapitalManager 验证: PASSED")
    return mgr

def test_dynamic_allocator_import(config):
    """3. 验证 DynamicAllocator 和辅助模块"""
    print("\n=== 3. DynamicAllocator 模块验证 ===")
    from risk.dynamic_allocator import (
        DynamicAllocator, PoolType, MarketRegime, AllocationPriority,
        CapitalPool, StrategyAllocation, AllocationPlan,
        VolatilityTargeter, AdaptiveKelly, CapitalEfficiencyMonitor,
        get_dynamic_allocator, reset_dynamic_allocator
    )
    print("  所有类导入成功")
    
    # 实例化 DynamicAllocator
    allocator = DynamicAllocator(config)
    print("  DynamicAllocator 实例化成功")
    
    # 实例化辅助模块
    vol_targeter = VolatilityTargeter(config.get('volatility_targeting', {}))
    print(f"  VolatilityTargeter: target={vol_targeter._target_vol:.0%}")
    
    kelly = AdaptiveKelly(config.get('adaptive_kelly', {}))
    print(f"  AdaptiveKelly: max={kelly._max_kelly:.0%}")
    
    eff_monitor = CapitalEfficiencyMonitor(config.get('capital_efficiency', {}))
    print(f"  CapitalEfficiencyMonitor: idle_threshold={eff_monitor._idle_threshold:.0%}")
    
    print("  DynamicAllocator 验证: PASSED")
    return allocator

def test_allocation_agent_import(config):
    """4. 验证 AllocationAgent 模块"""
    print("\n=== 4. AllocationAgent 模块验证 ===")
    # 需要 mock 依赖
    from unittest.mock import MagicMock
    from risk.allocation_agent import AllocationAgent
    
    trade_journal = MagicMock()
    trade_journal.get_trades_by_strategy.return_value = []
    profit_optimizer = MagicMock()
    account_manager = MagicMock()
    account_manager.get_total_equity.return_value = 5000.0
    account_manager.get_total_capital.return_value = 5000.0
    
    agent = AllocationAgent(config, trade_journal, profit_optimizer, account_manager)
    print(f"  AllocationAgent 实例化成功: method={agent._allocation_method}")
    
    # 注入 DynamicAllocator
    from risk.dynamic_allocator import DynamicAllocator
    allocator = DynamicAllocator(config)
    agent.set_dynamic_allocator(allocator)
    assert agent._dynamic_allocator is not None, "DynamicAllocator 注入失败"
    print("  DynamicAllocator 注入成功")
    
    # 验证策略名称从 config 动态加载（不再硬编码）
    assert len(agent._strategy_names) > 0, "策略名称列表为空"
    print(f"  策略名称: {len(agent._strategy_names)} 个 -> {agent._strategy_names}")
    
    print("  AllocationAgent 验证: PASSED")

def test_volatility_targeter():
    """5. 验证 VolatilityTargeter"""
    print("\n=== 5. VolatilityTargeter 功能验证 ===")
    from risk.dynamic_allocator import VolatilityTargeter
    
    vt = VolatilityTargeter({'target_volatility': 0.20, 'max_volatility': 0.50})
    
    # 模拟收益率序列
    import numpy as np
    returns = list(np.random.normal(0.001, 0.02, 100))
    
    est = vt.estimate_volatility(returns)
    assert 'current' in est, "波动率估计缺少 current"
    assert 'ewma' in est, "波动率估计缺少 ewma"
    print(f"  波动率估计: current={est['current']:.4f}, ewma={est['ewma']:.4f}")
    
    # 仓位缩放
    scale = vt.compute_scale_factor(0.25, 0.20, 5.0, 10.0)
    assert 'scale_factor' in scale, "缩放因子缺失"
    print(f"  仓位缩放: scale={scale['scale_factor']:.4f}, target={scale['target_position']:.2f}")
    
    # 波动率突破检查
    breached, reason = vt.check_volatility_breach(0.55)
    assert breached, "波动率突破检测失败"
    print(f"  波动率突破: breached={breached}, reason={reason[:50]}...")
    
    print("  VolatilityTargeter: PASSED")

def test_adaptive_kelly():
    """6. 验证 AdaptiveKelly"""
    print("\n=== 6. AdaptiveKelly 功能验证 ===")
    from risk.dynamic_allocator import AdaptiveKelly
    
    ak = AdaptiveKelly({})
    
    # 基础 Kelly 计算
    result = ak.compute_kelly(
        win_rate=0.55, avg_win=0.02, avg_loss=0.015,
        regime='trending_up', drawdown=0.03,
        consecutive_wins=2, consecutive_losses=0,
        trade_count=50
    )
    assert 'final_kelly' in result, "Kelly 计算缺少 final_kelly"
    assert 'fractional_kelly' in result, "Kelly 计算缺少 fractional_kelly"
    print(f"  Kelly: base={result['base_kelly']:.4f}, final={result['final_kelly']:.4f}, fractional={result['fractional_kelly']:.4f}")
    print(f"  regime_mult={result['regime_multiplier']:.2f}, dd_penalty={result['drawdown_penalty']:.2f}")
    
    # 连续版本 Kelly
    import numpy as np
    returns = list(np.random.normal(0.001, 0.02, 100))
    cont_kelly = ak.compute_continuous_kelly(returns)
    print(f"  连续 Kelly: {cont_kelly:.4f}")
    
    # 回撤惩罚
    penalty = ak.compute_drawdown_penalty(0.15)
    print(f"  回撤惩罚(15%): {penalty:.4f}")
    
    print("  AdaptiveKelly: PASSED")

def test_capital_efficiency_monitor():
    """7. 验证 CapitalEfficiencyMonitor"""
    print("\n=== 7. CapitalEfficiencyMonitor 功能验证 ===")
    import asyncio
    from risk.dynamic_allocator import CapitalEfficiencyMonitor
    
    cem = CapitalEfficiencyMonitor({})
    
    # 更新指标
    cem.update_metrics("grid", 1000.0, 600.0, 50.0, 5000.0, 30.0)
    cem.update_metrics("trend", 800.0, 700.0, -20.0, 3000.0, 15.0)
    cem.update_metrics("scalping", 500.0, 200.0, 30.0, 8000.0, 10.0)
    print("  策略指标更新: 3 strategies")
    
    # 效率计算
    async def run():
        eff = await cem.compute_efficiency("grid")
        assert 'roc_e' in eff, "效率计算缺少 roc_e"
        print(f"  grid效率: ROCE={eff['roc_e']:.4f}, score={eff['efficiency_score']:.4f}, trend={eff['trend']}")
        
        # 排名
        ranked = await cem.rank_strategies()
        print(f"  效率排名: {' > '.join([r['strategy'] + '(' + str(round(r['efficiency_score'], 2)) + ')' for r in ranked])}")
        
        # 闲置检测
        idle = await cem.detect_idle_capital()
        print(f"  闲置资金: {idle['total_idle_amount']:.2f} USDT, {idle['idle_count']} strategies")
        
        # 再部署建议
        suggestions = await cem.suggest_redeployment()
        print(f"  再部署建议: {len(suggestions)} suggestions")
        
        # 摘要
        summary = cem.get_efficiency_summary()
        print(f"  效率摘要: ROCE={summary['overall_roc_e']:.4f}, margin_util={summary['overall_margin_utilization']:.4f}")
    
    asyncio.run(run())
    print("  CapitalEfficiencyMonitor: PASSED")

if __name__ == '__main__':
    config = test_config_loading()
    mgr = test_capital_manager_import(config)
    allocator = test_dynamic_allocator_import(config)
    test_allocation_agent_import(config)
    test_volatility_targeter()
    test_adaptive_kelly()
    test_capital_efficiency_monitor()
    
    print("\n" + "=" * 60)
    print("  所有测试通过! 生产级资金分布交付完成")
    print("=" * 60)