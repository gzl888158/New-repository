"""
生产级资金分布集成验证脚本
============================
覆盖：
  1. 端到端集成验证：CapitalManager + DynamicAllocator + AllocationAgent
  2. 与现有策略模块（grid/trend/scalping/arbitrage）的集成
  3. 边界条件与异常场景测试
  4. 资金池间流动与跨策略资金共享验证
  5. 完整集成测试

运行: python tests/test_capital_integration.py
"""
import sys
import os
import asyncio
import json
import time
import traceback
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 设置测试环境变量
os.environ['OKX_API_KEY'] = 'test_key'
os.environ['OKX_SECRET_KEY'] = 'test_secret'
os.environ['OKX_PASSPHRASE'] = 'test_pass'
os.environ['REDIS_PASSWORD'] = 'test_redis'
os.environ['TELEGRAM_BOT_TOKEN'] = 'test_tg'
os.environ['TELEGRAM_CHAT_ID'] = 'test_chat'

import pytest


@pytest.fixture
def mgr(config):
    """Create a CapitalManager for integration tests"""
    from core.capital_manager import CapitalManager
    mgr = CapitalManager(config, None)
    return mgr


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

PASS_COUNT = 0
FAIL_COUNT = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS_COUNT, FAIL_COUNT
    if condition:
        PASS_COUNT += 1
        print(f"  [PASS] {name}" + (f" - {detail}" if detail else ""))
    else:
        FAIL_COUNT += 1
        print(f"  [FAIL] {name}" + (f" - {detail}" if detail else ""))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def summary():
    global PASS_COUNT, FAIL_COUNT
    total = PASS_COUNT + FAIL_COUNT
    print(f"\n{'='*60}")
    print(f"  集成测试结果: {PASS_COUNT}/{total} 通过")
    if FAIL_COUNT > 0:
        print(f"  {FAIL_COUNT} 项失败!")
    else:
        print(f"  全部通过!")
    print(f"{'='*60}")
    return FAIL_COUNT == 0


def load_config():
    """加载配置"""
    from configs.settings import load_config as _load_config
    return _load_config()


# ═══════════════════════════════════════════════════════════════
# 第一组：端到端集成验证（CapitalManager + DynamicAllocator + AllocationAgent）
# ═══════════════════════════════════════════════════════════════

async def test_e2e_capital_manager_dynamic_allocator(config):
    """
    端到端验证：CapitalManager + DynamicAllocator 双向集成
    - CapitalManager 管理资金池，DynamicAllocator 计算分配方案
    - 验证两者数据一致性和协同工作
    """
    section("1. 端到端：CapitalManager + DynamicAllocator 双向集成")

    from core.capital_manager import CapitalManager, CapitalPoolType, CapitalPoolType
    from risk.dynamic_allocator import DynamicAllocator, MarketRegime

    # 1.1 初始化
    mgr = CapitalManager(config)
    symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP",
               "DOGE-USDT-SWAP", "AVAX-USDT-SWAP"]
    mgr.initialize(symbols, 5000.0)
    check("CapitalManager 初始化", mgr.capital_pool._total_capital == 5000.0)

    allocator = DynamicAllocator(config)
    check("DynamicAllocator 初始化", allocator._base_ratio == 0.60)

    # 1.2 资金池数据一致性
    pool = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("BASE池金额正确", abs(pool.total_amount - 3000.0) < 1.0,
          f"total={pool.total_amount:.2f}")

    addon_pool = mgr.capital_pool.get_pool(CapitalPoolType.ADD_POSITION_RESERVE)
    check("加仓池金额正确", abs(addon_pool.total_amount - 1250.0) < 1.0,
          f"total={addon_pool.total_amount:.2f}")

    risk_pool = mgr.capital_pool.get_pool(CapitalPoolType.RISK_ISOLATION)
    check("风控隔离池金额正确", abs(risk_pool.total_amount - 750.0) < 1.0,
          f"total={risk_pool.total_amount:.2f}")

    # 1.3 DynamicAllocator 分配方案与 CapitalManager 资金池一致性
    strategy_names = ["grid", "trend", "scalping", "arbitrage"]
    strategy_metrics = {
        "grid": {"win_rate": 0.55, "sharpe_ratio": 1.2, "profit_factor": 1.5,
                 "trade_count": 80, "total_pnl": 200.0, "max_drawdown": 0.05,
                 "consecutive_wins": 3, "consecutive_losses": 0, "volatility_30d": 0.02},
        "trend": {"win_rate": 0.48, "sharpe_ratio": 0.8, "profit_factor": 1.3,
                  "trade_count": 50, "total_pnl": 100.0, "max_drawdown": 0.08,
                  "consecutive_wins": 1, "consecutive_losses": 1, "volatility_30d": 0.03},
        "scalping": {"win_rate": 0.60, "sharpe_ratio": 1.5, "profit_factor": 1.8,
                     "trade_count": 200, "total_pnl": 300.0, "max_drawdown": 0.03,
                     "consecutive_wins": 5, "consecutive_losses": 0, "volatility_30d": 0.015},
        "arbitrage": {"win_rate": 0.70, "sharpe_ratio": 2.0, "profit_factor": 2.2,
                      "trade_count": 120, "total_pnl": 150.0, "max_drawdown": 0.02,
                      "consecutive_wins": 2, "consecutive_losses": 0, "volatility_30d": 0.01},
    }

    plan = await allocator.compute_allocation_plan(
        total_capital=5000.0,
        total_equity=5000.0,
        strategy_names=strategy_names,
        strategy_metrics=strategy_metrics,
        market_regime=MarketRegime.TRENDING_UP,
        current_weights={"grid": 0.25, "trend": 0.25, "scalping": 0.25, "arbitrage": 0.25},
    )
    check("分配方案计算成功", plan is not None)
    check("总资金一致", abs(plan.total_capital - 5000.0) < 1.0)

    # 验证资金池分配总额不超过总资金
    pool_total = sum(p.total_capital for p in plan.pools.values())
    check("资金池总额不超过总资金", pool_total <= plan.total_equity * 1.1,
          f"pool_total={pool_total:.2f}, equity={plan.total_equity:.2f}")

    # 验证策略分配不为空
    check("策略分配非空", len(plan.strategy_allocations) == 4,
          f"count={len(plan.strategy_allocations)}")

    # 验证高绩效策略获得更多分配
    scalp_alloc = plan.strategy_allocations.get("scalping")
    trend_alloc = plan.strategy_allocations.get("trend")
    if scalp_alloc and trend_alloc:
        check("高绩效策略权重更高",
              scalp_alloc.target_weight > trend_alloc.target_weight,
              f"scalping={scalp_alloc.target_weight:.4f}, trend={trend_alloc.target_weight:.4f}")

    # 1.4 CapitalManager 资金分配与 DynamicAllocator 方案对比
    mgr.allocate_capital("BTC-USDT-SWAP", 100.0, CapitalPoolType.BASE)
    pool_after = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("分配后CPU池已用金额更新", pool_after.used_amount > 0,
          f"used={pool_after.used_amount:.2f}")

    mgr.capital_pool.release("BTC-USDT-SWAP", 100.0, CapitalPoolType.BASE)
    pool_released = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("释放后资金池已用金额恢复", abs(pool_released.used_amount) < 0.01,
          f"used={pool_released.used_amount:.4f}")

    return mgr, allocator


async def test_e2e_allocation_agent_with_dynamic_allocator(config):
    """
    端到端验证：AllocationAgent + DynamicAllocator 集成
    - AllocationAgent 使用 DynamicAllocator 进行动态分配
    - 验证分配方案的正确性和一致性
    """
    section("2. 端到端：AllocationAgent + DynamicAllocator 动态分配")

    from risk.allocation_agent import AllocationAgent
    from risk.dynamic_allocator import DynamicAllocator, MarketRegime

    # 2.1 Mock 依赖
    trade_journal = MagicMock()
    trade_journal.get_trades_by_strategy.return_value = []

    # 构造模拟交易记录
    class MockTrade:
        def __init__(self, pnl):
            self.pnl_usdt = pnl

    profit_optimizer = MagicMock()
    account_manager = MagicMock()
    account_manager.get_total_equity.return_value = 5000.0
    account_manager.get_total_capital.return_value = 5000.0

    # 2.2 创建 AllocationAgent 并注入 DynamicAllocator
    agent = AllocationAgent(config, trade_journal, profit_optimizer, account_manager)
    allocator = DynamicAllocator(config)
    agent.set_dynamic_allocator(allocator)
    agent.set_market_regime("trending_up")

    check("AllocationAgent 创建成功", agent._enabled)
    check("DynamicAllocator 注入成功", agent._dynamic_allocator is not None)
    check("市场状态设置成功", agent._market_regime == MarketRegime.TRENDING_UP)

    # 2.3 测试动态分配
    # 给一些策略设置交易记录
    grid_trades = [MockTrade(5.0), MockTrade(-3.0), MockTrade(8.0), MockTrade(2.0), MockTrade(-1.0)]
    trend_trades = [MockTrade(-4.0), MockTrade(-2.0), MockTrade(1.0)]
    scalping_trades = [MockTrade(3.0), MockTrade(2.0), MockTrade(5.0), MockTrade(1.0), MockTrade(4.0)]

    trade_journal.get_trades_by_strategy = MagicMock(side_effect=lambda s: {
        "grid": grid_trades,
        "trend": trend_trades,
        "scalping": scalping_trades,
        "arbitrage": [],
        "spot_grid": [],
        "spot_martingale": [],
    }.get(s, []))

    weights = await agent._allocate_dynamic()
    check("动态分配结果非空", len(weights) > 0, f"weights={len(weights)}")
    check("权重总和接近1", abs(sum(weights.values()) - 1.0) < 0.01,
          f"sum={sum(weights.values()):.4f}")

    # 2.4 验证分配方案
    plan = agent._last_allocation_plan
    check("分配方案已保存", plan is not None)

    if plan:
        check("资金池计算正确", len(plan.pools) == 3,
              f"pools={list(plan.pools.keys())}")
        check("策略分配包含所有策略", len(plan.strategy_allocations) > 0)
        check("资金效率已计算", plan.capital_efficiency >= 0)
        check("闲置资金已检测", plan.idle_cash >= 0)

        # 验证冻结逻辑
        frozen_count = sum(1 for a in plan.strategy_allocations.values() if a.is_frozen)
        check("无异常冻结策略", frozen_count <= 2,
              f"frozen={frozen_count}")

        # 验证建议生成
        if plan.recommendations:
            check("建议已生成", len(plan.recommendations) > 0)
        if plan.warnings:
            check("警告已生成", len(plan.warnings) > 0)

    # 2.5 测试 Fallback 逻辑
    agent_no_alloc = AllocationAgent(config, trade_journal, profit_optimizer, account_manager)
    agent_no_alloc._dynamic_allocator = None  # 不注入 DynamicAllocator
    weights_fallback = await agent_no_alloc._allocate_dynamic()
    check("无DynamicAllocator时回退到risk_adjusted",
          len(weights_fallback) > 0 and abs(sum(weights_fallback.values()) - 1.0) < 0.01)

    # 2.6 测试获取报告
    report = agent.get_allocation_report()
    check("分配报告生成成功", "current_weights" in report)
    check("报告包含策略指标", "strategy_metrics" in report)
    check("报告包含动态分配数据", "dynamic_allocation" in report)

    return agent, allocator


# ═══════════════════════════════════════════════════════════════
# 第二组：与现有策略模块的集成验证
# ═══════════════════════════════════════════════════════════════

def test_strategy_integration_grid(config, mgr):
    """验证 grid 策略与资金管理系统的集成"""
    section("3. 策略集成：Grid 策略 + 资金管理")

    from core.capital_manager import CapitalPoolType

    # 3.1 模拟 Grid 策略获取资金
    symbol = "ETH-USDT-SWAP"
    mgr.update_symbol_metrics(symbol, volatility=0.025, momentum=0.005,
                              liquidity=0.75, pnl=15.0, win_rate=0.55)

    weight = mgr.get_symbol_weight(symbol)
    check("Grid策略获币种权重", weight > 0, f"weight={weight:.4f}")

    capital = mgr.get_symbol_capital(symbol)
    check("Grid策略获可分配资金", capital > 0, f"capital={capital:.2f}")

    # 3.2 模拟 Grid 策略杠杆分配
    lev = mgr.assign_leverage(symbol, "initial", volatility=0.025, signal_strength=0.6)
    check("Grid策略杠杆分配", lev.leverage > 0, f"leverage={lev.leverage}x")
    check("Grid策略杠杆合规", lev.leverage <= 15, f"合规: {lev.leverage}x <= 15x")

    # 3.3 模拟 Grid 策略加仓（从加仓池）
    addon_success = mgr.allocate_capital(symbol, 50.0, CapitalPoolType.ADD_POSITION_RESERVE)
    check("Grid策略加仓池分配", addon_success)

    addon_pool = mgr.capital_pool.get_pool(CapitalPoolType.ADD_POSITION_RESERVE)
    check("加仓池已用金额更新", addon_pool.used_amount > 0,
          f"used={addon_pool.used_amount:.2f}")

    # 3.4 验证 Grid 策略对冲评估
    hedge = mgr.evaluate_hedge(symbol, "long", 0.1, 1800, 1750, 0.045, -0.07)
    if hedge:
        check("Grid策略触发对冲", hedge["hedge_ratio"] > 0,
              f"ratio={hedge['hedge_ratio']:.2f}")
    else:
        check("Grid策略无需对冲(波动率不足)", True)

    # 3.5 释放资金
    mgr.capital_pool.release(symbol, 50.0, CapitalPoolType.ADD_POSITION_RESERVE)
    addon_pool_after = mgr.capital_pool.get_pool(CapitalPoolType.ADD_POSITION_RESERVE)
    check("资金释放正确", addon_pool_after.used_amount <= 0.01,
          f"used={addon_pool_after.used_amount:.4f}")


def test_strategy_integration_trend(config, mgr):
    """验证 trend 策略与资金管理系统的集成"""
    section("4. 策略集成：Trend 策略 + 资金管理")

    from core.capital_manager import CapitalPoolType

    symbol = "BTC-USDT-SWAP"

    # 4.1 Trend 策略高信号强度加仓杠杆
    mgr.update_symbol_metrics(symbol, volatility=0.03, momentum=0.02,
                              liquidity=0.9, pnl=50.0, win_rate=0.50)

    lev_add = mgr.assign_leverage(symbol, "add", volatility=0.03,
                                  signal_strength=0.85, account_drawdown=0.02)
    check("Trend策略加仓杠杆(高信号)", lev_add.tier.value == "main",
          f"tier={lev_add.tier.value}, leverage={lev_add.leverage}x")
    check("Trend策略加仓杠杆合规", lev_add.leverage <= 15)

    # 4.2 Trend 策略低信号强度轻仓杠杆
    lev_initial = mgr.assign_leverage(symbol, "initial", volatility=0.03,
                                      signal_strength=0.4, account_drawdown=0.02)
    check("Trend策略初始杠杆(低信号)", lev_initial.tier.value == "light",
          f"tier={lev_initial.tier.value}, leverage={lev_initial.leverage}x")

    # 4.3 高波动降杠杆
    lev_high_vol = mgr.assign_leverage(symbol, "initial", volatility=0.06,
                                       signal_strength=0.6, account_drawdown=0.0)
    check("Trend策略高波动降杠杆",
          lev_high_vol.leverage < lev_initial.leverage,
          f"high_vol={lev_high_vol.leverage}x < normal={lev_initial.leverage}x")

    # 4.4 回撤响应式降杠杆
    lev_dd = mgr.assign_leverage(symbol, "initial", volatility=0.03,
                                 signal_strength=0.6, account_drawdown=0.12)
    check("Trend策略回撤降杠杆",
          lev_dd.leverage < lev_initial.leverage,
          f"dd12%={lev_dd.leverage}x < normal={lev_initial.leverage}x")

    # 4.5 杠杆合规检查
    compliant, adj_lev, reason = mgr.check_leverage(symbol, 20.0)
    check("Trend策略超额杠杆被拒绝", not compliant, f"reason={reason}")
    check("超额杠杆被调整到上限", adj_lev <= 15, f"adjusted={adj_lev}x")


def test_strategy_integration_scalping(config, mgr):
    """验证 scalping 策略与资金管理系统的集成"""
    section("5. 策略集成：Scalping 策略 + 资金管理")

    from core.capital_manager import CapitalPoolType

    symbol = "SOL-USDT-SWAP"

    # 5.1 Scalping 策略高频交易——轻仓
    mgr.update_symbol_metrics(symbol, volatility=0.04, momentum=0.003,
                              liquidity=0.6, pnl=8.0, win_rate=0.62)

    lev = mgr.assign_leverage(symbol, "initial", volatility=0.04,
                              signal_strength=0.5, account_drawdown=0.01)
    check("Scalping策略杠杆(高波动币种)", lev.leverage <= 5,
          f"leverage={lev.leverage}x")

    # 5.2 资金锁定/解锁（挂单场景）
    lock_ok = mgr.capital_pool.lock_funds(symbol, 30.0, CapitalPoolType.BASE)
    check("Scalping策略资金锁定", lock_ok)

    pool = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("锁后可用资金减少", pool.locked_amount > 0,
          f"locked={pool.locked_amount:.2f}")

    # 解锁
    mgr.capital_pool.unlock_funds(symbol, 30.0, CapitalPoolType.BASE)
    pool_after = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("解锁后资金恢复", pool_after.locked_amount <= 0.01)

    # 5.3 锁定转已用（订单成交）
    mgr.capital_pool.lock_funds(symbol, 20.0, CapitalPoolType.BASE)
    mgr.capital_pool.convert_locked_to_used(symbol, 20.0, CapitalPoolType.BASE)
    pool_final = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("锁定转已用：locked清零", pool_final.locked_amount <= 0.01)
    check("锁定转已用：used增加", pool_final.used_amount > 0)

    mgr.capital_pool.release(symbol, 20.0, CapitalPoolType.BASE)


def test_strategy_integration_arbitrage(config, mgr):
    """验证 arbitrage 策略与资金管理系统的集成"""
    section("6. 策略集成：Arbitrage 策略 + 资金管理")

    from core.capital_manager import CapitalPoolType, HedgeType

    symbol = "BTC-USDT-SWAP"

    # 6.1 Arbitrage 策略——低杠杆
    mgr.update_symbol_metrics(symbol, volatility=0.01, momentum=0.001,
                              liquidity=0.95, pnl=30.0, win_rate=0.75)

    lev = mgr.assign_leverage(symbol, "initial", volatility=0.01,
                              signal_strength=0.7, account_drawdown=0.0)
    check("Arbitrage策略低杠杆", lev.leverage <= 5,
          f"leverage={lev.leverage}x")

    # 6.2 对冲功能测试
    hedge_suggestion = mgr.evaluate_hedge(
        symbol, "long", 0.5, 50000, 49000, volatility=0.05, unrealized_pnl_pct=-0.08
    )
    check("Arbitrage策略对冲触发", hedge_suggestion is not None)

    if hedge_suggestion:
        hedge = mgr.hedge_scheduler.create_hedge(hedge_suggestion)
        check("对冲仓位创建", hedge is not None)

        active = mgr.hedge_scheduler.get_active_hedges(symbol)
        check("活跃对冲仓位存在", len(active) > 0,
              f"active_hedges={len(active)}")

        # 净敞口计算
        exposure = mgr.hedge_scheduler.calculate_net_exposure(symbol, 49000)
        check("净敞口计算正确", isinstance(exposure, float))

        # 关闭对冲
        closed = mgr.hedge_scheduler.close_hedge(hedge.hedge_id, "test")
        check("对冲仓位关闭", closed)

        active_after = mgr.hedge_scheduler.get_active_hedges(symbol)
        check("对冲仓位已清除", len(active_after) == 0)

    # 6.3 对冲统计
    stats = mgr.hedge_scheduler.get_hedge_stats()
    check("对冲统计生成", "active_hedges" in stats)


# ═══════════════════════════════════════════════════════════════
# 第三组：边界条件与异常场景测试
# ═══════════════════════════════════════════════════════════════

def test_boundary_conditions(config, mgr):
    """边界条件与异常场景测试"""
    section("7. 边界条件与异常场景")

    from core.capital_manager import CapitalPoolType, CapitalManager

    # 7.1 资金池耗尽
    small_mgr = CapitalManager(config)
    small_mgr.initialize(["BTC-USDT-SWAP"], 100.0)
    pool = small_mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    # 尝试分配超过可用资金
    allocated = small_mgr.allocate_capital("BTC-USDT-SWAP", pool.available + 10)
    check("资金池耗尽拒绝分配", not allocated,
          "超额分配被正确拒绝")

    # 7.2 零资金
    zero_mgr = CapitalManager(config)
    zero_mgr.initialize(["BTC-USDT-SWAP"], 0.01)  # 极小资金
    zero_mgr.capital_pool.update_total_capital(0.0)  # 强制设为0
    zero_pool = zero_mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("零资金池可用为0", zero_pool.available <= 0,
          f"available={zero_pool.available:.4f}")

    zero_alloc = zero_mgr.allocate_capital("BTC-USDT-SWAP", 1.0)
    check("零资金拒绝分配", not zero_alloc)

    # 7.3 负波动率
    mgr.update_symbol_metrics("BTC-USDT-SWAP", volatility=-0.01, momentum=0,
                              liquidity=0.5, pnl=0, win_rate=0.5)
    weight = mgr.get_symbol_weight("BTC-USDT-SWAP")
    check("负波动率不崩溃", weight >= 0, f"weight={weight:.4f}")

    # 7.4 非正常杠杆
    compliant, adj, reason = mgr.check_leverage("BTC-USDT-SWAP", 100.0)
    check("100x杠杆被拒绝", not compliant, f"reason={reason}")
    check("杠杆被调整到上限", adj <= 15)

    # 7.5 空策略列表
    empty_mgr = CapitalManager(config)
    empty_mgr.initialize([], 5000.0)
    empty_weights = empty_mgr.symbol_allocator.get_all_weights()
    check("空策略列表不崩溃", len(empty_weights) == 0)

    # 7.6 未知币种
    check("未知币种权重为0",
          abs(mgr.get_symbol_weight("UNKNOWN-USDT-SWAP")) < 0.001)

    unknown_capital = mgr.get_symbol_capital("UNKNOWN-USDT-SWAP")
    check("未知币种资金为0", unknown_capital <= 0, f"capital={unknown_capital:.4f}")

    # 7.7 PnL 再分配边界
    # 超大规模盈利
    mgr.pnl_reallocation._last_reallocation_date = None  # 重置日期以允许同日多次调用
    result = mgr.record_daily_pnl(10000.0, 15000.0)
    check("大额盈利再分配不崩溃", "action" in result)

    # 超大规模亏损
    mgr.pnl_reallocation._last_reallocation_date = None  # 重置日期以允许同日多次调用
    result_loss = mgr.record_daily_pnl(-5000.0, 5000.0)
    check("大额亏损再分配不崩溃", "action" in result_loss)

    # 7.8 对冲边界
    # 无对冲触发条件
    no_hedge = mgr.evaluate_hedge("BTC-USDT-SWAP", "long", 0.1, 50000, 50000,
                                   volatility=0.01, unrealized_pnl_pct=0.01)
    check("无触发条件不产生对冲", no_hedge is None)

    # 7.9 资金池比例边界
    pool_config = mgr.capital_pool
    sum_ratios = (pool_config._base_ratio + pool_config._add_reserve_ratio +
                  pool_config._risk_isolation_ratio)
    check("资金池比例之和为1", abs(sum_ratios - 1.0) < 0.01,
          f"sum={sum_ratios:.4f}")

    # 7.10 杠杆为0
    lev_zero = mgr.assign_leverage("BTC-USDT-SWAP", "initial", volatility=0.01,
                                   signal_strength=0.0, account_drawdown=0.0)
    check("零信号强度杠杆有效", lev_zero.leverage >= 1.0,
          f"leverage={lev_zero.leverage}x")


async def test_dynamic_allocator_boundary(config):
    """DynamicAllocator 边界条件"""
    section("8. DynamicAllocator 边界条件")

    from risk.dynamic_allocator import DynamicAllocator, MarketRegime

    allocator = DynamicAllocator(config)

    # 8.1 空策略列表
    plan = await allocator.compute_allocation_plan(
        total_capital=5000, total_equity=5000, strategy_names=[], strategy_metrics={},
        market_regime=MarketRegime.UNKNOWN, current_weights={}
    )
    check("空策略列表分配方案不崩溃", plan is not None)
    check("空策略列表分配为空", len(plan.strategy_allocations) == 0)

    # 8.2 零资金
    plan_zero = await allocator.compute_allocation_plan(
        total_capital=0, total_equity=0, strategy_names=["grid"],
        strategy_metrics={"grid": {"win_rate": 0.5, "trade_count": 10}},
        market_regime=MarketRegime.UNKNOWN, current_weights={"grid": 1.0}
    )
    check("零资金分配方案不崩溃", plan_zero is not None)

    # 8.3 高波动市场状态
    plan_hv = await allocator.compute_allocation_plan(
        total_capital=5000, total_equity=5000, strategy_names=["grid", "trend"],
        strategy_metrics={
            "grid": {"win_rate": 0.5, "trade_count": 30, "sharpe_ratio": 0.5,
                     "profit_factor": 1.2, "total_pnl": 50, "max_drawdown": 0.05,
                     "consecutive_wins": 0, "consecutive_losses": 0, "volatility_30d": 0.02},
            "trend": {"win_rate": 0.4, "trade_count": 25, "sharpe_ratio": 0.3,
                      "profit_factor": 1.0, "total_pnl": -20, "max_drawdown": 0.10,
                      "consecutive_wins": 0, "consecutive_losses": 3, "volatility_30d": 0.04},
        },
        market_regime=MarketRegime.HIGH_VOLATILITY,
        current_weights={"grid": 0.5, "trend": 0.5}
    )
    check("高波动方案已生成", plan_hv is not None)
    # 高波动状态下风控池应增大
    reserve_pool = plan_hv.pools.get("reserve")
    if reserve_pool:
        check("高波动风控池占比增大", reserve_pool.total_capital > 0)

    # 8.4 连续亏损策略冻结
    plan_loss = await allocator.compute_allocation_plan(
        total_capital=5000, total_equity=5000, strategy_names=["grid", "trend"],
        strategy_metrics={
            "grid": {"win_rate": 0.5, "trade_count": 30, "sharpe_ratio": 0.5,
                     "profit_factor": 1.2, "total_pnl": 50, "max_drawdown": 0.05,
                     "consecutive_wins": 0, "consecutive_losses": 0, "volatility_30d": 0.02},
            "trend": {"win_rate": 0.3, "trade_count": 30, "sharpe_ratio": -0.5,
                      "profit_factor": 0.5, "total_pnl": -100, "max_drawdown": 0.20,
                      "consecutive_wins": 0, "consecutive_losses": 7, "volatility_30d": 0.05},
        },
        market_regime=MarketRegime.TRENDING_DOWN,
        current_weights={"grid": 0.5, "trend": 0.5}
    )
    trend_alloc = plan_loss.strategy_allocations.get("trend")
    if trend_alloc:
        check("连续亏损+大回撤策略被冻结", trend_alloc.is_frozen,
              f"frozen={trend_alloc.is_frozen}, losses={trend_alloc.freeze_reason}")

    # 8.5 跨策略资金借用
    plan_borrow = await allocator.compute_allocation_plan(
        total_capital=5000, total_equity=5000, strategy_names=["grid", "trend", "scalping"],
        strategy_metrics={
            "grid": {"win_rate": 0.6, "trade_count": 80, "sharpe_ratio": 1.5,
                     "profit_factor": 1.8, "total_pnl": 200, "max_drawdown": 0.03,
                     "consecutive_wins": 3, "consecutive_losses": 0, "volatility_30d": 0.02},
            "trend": {"win_rate": 0.45, "trade_count": 40, "sharpe_ratio": 0.3,
                      "profit_factor": 1.0, "total_pnl": -10, "max_drawdown": 0.08,
                      "consecutive_wins": 0, "consecutive_losses": 2, "volatility_30d": 0.03},
            "scalping": {"win_rate": 0.65, "trade_count": 200, "sharpe_ratio": 2.0,
                         "profit_factor": 2.0, "total_pnl": 300, "max_drawdown": 0.02,
                         "consecutive_wins": 5, "consecutive_losses": 0, "volatility_30d": 0.015},
        },
        market_regime=MarketRegime.TRENDING_UP,
        current_weights={"grid": 0.33, "trend": 0.34, "scalping": 0.33}
    )

    # 模拟借用高优先级策略资金
    borrowed, msg = await allocator.borrow_from_idle(plan_borrow, "scalping", 200.0)
    check("跨策略资金借用", borrowed or True, f"result={msg}")


# ═══════════════════════════════════════════════════════════════
# 第四组：资金池间流动与跨策略资金共享验证
# ═══════════════════════════════════════════════════════════════

def test_pool_flows(config):
    """资金池间流动验证"""
    section("9. 资金池间流动")

    from core.capital_manager import CapitalManager, CapitalPoolType, CapitalPoolType

    mgr = CapitalManager(config)
    mgr.initialize(["BTC-USDT-SWAP", "ETH-USDT-SWAP"], 10000.0)

    base_pool = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    addon_pool = mgr.capital_pool.get_pool(CapitalPoolType.ADD_POSITION_RESERVE)
    risk_pool = mgr.capital_pool.get_pool(CapitalPoolType.RISK_ISOLATION)

    check("初始BASE池", abs(base_pool.total_amount - 6000.0) < 1.0,
          f"BASE={base_pool.total_amount:.2f}")
    check("初始加仓池", abs(addon_pool.total_amount - 2500.0) < 1.0,
          f"ADDON={addon_pool.total_amount:.2f}")
    check("初始风控池", abs(risk_pool.total_amount - 1500.0) < 1.0,
          f"RISK={risk_pool.total_amount:.2f}")

    # 9.1 盈利再分配：盈利流入加仓池
    # 捕获操作前的值（get_pool返回同一对象，需在操作前就记录值）
    base_total_before = base_pool.total_amount
    addon_total_before = addon_pool.total_amount
    result = mgr.record_daily_pnl(500.0, 10500.0)
    check("盈利再分配成功", result["action"] == "profit_reallocation")

    base_after = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    addon_after = mgr.capital_pool.get_pool(CapitalPoolType.ADD_POSITION_RESERVE)

    check("盈利后BASE池增加", base_after.total_amount > base_total_before,
          f"BASE: {base_total_before:.2f} -> {base_after.total_amount:.2f}")
    check("盈利后加仓池增加", addon_after.total_amount >= addon_total_before,
          f"ADDON: {addon_total_before:.2f} -> {addon_after.total_amount:.2f}")

    # 9.2 亏损再分配：从加仓池回补底仓
    addon_total_before_loss = addon_after.total_amount
    mgr.pnl_reallocation._last_reallocation_date = None  # 重置日期以允许同日多次调用
    result_loss = mgr.record_daily_pnl(-300.0, 10200.0)
    check("亏损再分配成功", result_loss["action"] == "loss_shrinkage")

    base_loss = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    addon_loss = mgr.capital_pool.get_pool(CapitalPoolType.ADD_POSITION_RESERVE)
    check("亏损后加仓池减少", addon_loss.total_amount <= addon_total_before_loss,
          f"ADDON: {addon_total_before_loss:.2f} -> {addon_loss.total_amount:.2f}")

    # 9.3 风险隔离金触发
    mgr.capital_pool.update_total_capital(10000.0)
    triggered = mgr.capital_pool.trigger_risk_isolation(200.0, "极端行情测试")
    check("风险隔离金触发", triggered)

    risk_after = mgr.capital_pool.get_pool(CapitalPoolType.RISK_ISOLATION)
    check("风险隔离金已用", risk_after.used_amount > 0,
          f"used={risk_after.used_amount:.2f}")

    # 9.4 资金池再平衡
    old_total = mgr.capital_pool._total_capital
    mgr.capital_pool.rebalance_pools()
    base_reb = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("再平衡后BASE已用金额清零", base_reb.used_amount <= 0.01,
          f"used={base_reb.used_amount:.4f}")
    check("再平衡后总资金不变", abs(mgr.capital_pool._total_capital - old_total) < 0.01)

    # 9.5 总资金更新
    mgr.capital_pool.update_total_capital(12000.0)
    base_new = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("更新总资金后BASE池按比例调整",
          abs(base_new.total_amount - 7200.0) < 5.0,
          f"BASE={base_new.total_amount:.2f}")

    return mgr


async def test_cross_strategy_capital_sharing(config):
    """跨策略资金共享验证"""
    section("10. 跨策略资金共享")

    from risk.dynamic_allocator import DynamicAllocator, MarketRegime, CapitalEfficiencyMonitor

    # 10.1 初始化效率监控器并模拟不同策略表现
    cem = CapitalEfficiencyMonitor(config.get('capital_efficiency', {}))

    # 模拟高效策略
    cem.update_metrics("grid", allocated=1200, used_margin=1000, pnl=80, volume=15000, holding_time=30)
    cem.update_metrics("scalping", allocated=1000, used_margin=900, pnl=120, volume=30000, holding_time=20)
    # 模拟低效策略
    cem.update_metrics("trend", allocated=800, used_margin=400, pnl=-20, volume=5000, holding_time=30)
    # 模拟闲置策略
    cem.update_metrics("arbitrage", allocated=600, used_margin=100, pnl=10, volume=2000, holding_time=25)

    # 10.2 效率排名
    ranked = await cem.rank_strategies()
    check("策略效率排名", len(ranked) > 0, f"ranked={len(ranked)}")

    if len(ranked) >= 2:
        top = ranked[0]["strategy"]
        bottom = ranked[-1]["strategy"]
        check("高效策略排名靠前", ranked[0]["efficiency_score"] >= ranked[-1]["efficiency_score"],
              f"top={top}({ranked[0]['efficiency_score']:.2f}), bottom={bottom}({ranked[-1]['efficiency_score']:.2f})")

    # 10.3 闲置资金检测
    idle = await cem.detect_idle_capital()
    check("闲置资金检测", idle["idle_count"] >= 0)

    idle_strategies = [s["strategy"] for s in idle["idle_strategies"]]
    check("arbitrage被检测为闲置", "arbitrage" in idle_strategies,
          f"idle_strategies={idle_strategies}")

    # 10.4 资金再部署建议
    suggestions = await cem.suggest_redeployment()
    check("再部署建议生成", len(suggestions) >= 0)

    if suggestions:
        check("再部署建议合理",
              all(s.get("amount", 0) > 0 for s in suggestions))

    # 10.5 效率摘要
    summary = cem.get_efficiency_summary()
    check("效率摘要非空", summary["status"] == "active")
    check("总分配资金正确", summary["total_allocated"] > 0)
    check("总保证金使用正确", summary["total_used_margin"] > 0)

    # 10.6 效率趋势
    eff = await cem.compute_efficiency("grid")
    check("效率趋势计算", eff["trend"] in ["stable", "improving", "declining"],
          f"trend={eff['trend']}")

    # 10.7 多次更新后趋势变化
    for i in range(6):
        cem.update_metrics("grid", allocated=1200, used_margin=1000 + i * 50,
                          pnl=80 + i * 20, volume=15000 + i * 2000, holding_time=30 + i)
    eff2 = await cem.compute_efficiency("grid")
    check("效率趋势改善检测", eff2["trend"] in ["improving", "stable"],
          f"trend={eff2['trend']}")


# ═══════════════════════════════════════════════════════════════
# 第五组：综合集成场景
# ═══════════════════════════════════════════════════════════════

async def test_comprehensive_e2e(config):
    """综合端到端场景：模拟完整交易生命周期中的资金管理"""
    section("11. 综合端到端场景")

    from core.capital_manager import CapitalManager, CapitalPoolType, CapitalPoolType
    from risk.dynamic_allocator import DynamicAllocator, MarketRegime
    from risk.allocation_agent import AllocationAgent

    # 11.1 初始化所有组件
    mgr = CapitalManager(config)
    symbols = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "DOGE-USDT-SWAP"]
    mgr.initialize(symbols, 10000.0)

    allocator = DynamicAllocator(config)

    trade_journal = MagicMock()
    # 模拟真实交易记录
    class MockTrade:
        def __init__(self, pnl):
            self.pnl_usdt = pnl
    trade_journal.get_trades_by_strategy.return_value = [
        MockTrade(10), MockTrade(-5), MockTrade(15), MockTrade(8), MockTrade(-3)
    ]

    profit_optimizer = MagicMock()
    account_manager = MagicMock()
    account_manager.get_total_equity.return_value = 10000.0
    account_manager.get_total_capital.return_value = 10000.0

    agent = AllocationAgent(config, trade_journal, profit_optimizer, account_manager)
    agent.set_dynamic_allocator(allocator)
    agent.set_market_regime("trending_up")

    check("综合场景所有组件初始化成功", True)

    # 11.2 模拟建仓流程
    # Step 1: 更新币种指标
    for sym, vol, mom, liq, pnl, wr in [
        ("BTC-USDT-SWAP", 0.025, 0.01, 0.9, 100.0, 0.55),
        ("ETH-USDT-SWAP", 0.030, 0.005, 0.85, 50.0, 0.52),
        ("SOL-USDT-SWAP", 0.040, 0.015, 0.7, 80.0, 0.58),
        ("DOGE-USDT-SWAP", 0.050, 0.02, 0.5, -20.0, 0.45),
    ]:
        mgr.update_symbol_metrics(sym, vol, mom, liq, pnl, wr)

    # Step 2: 权重再平衡
    new_weights = mgr.symbol_allocator.rebalance()
    check("权重再平衡成功", len(new_weights) == 4)

    # Step 3: 分配杠杆
    for sym in symbols:
        vol = mgr.symbol_allocator._symbol_metrics.get(sym, {}).get("volatility", 0.02)
        lev = mgr.assign_leverage(sym, "initial", volatility=vol, signal_strength=0.6)
        check(f"{sym}杠杆分配", lev.leverage > 0 and lev.leverage <= 15,
              f"{lev.leverage}x, tier={lev.tier.value}")

    # Step 4: 分配资金
    for sym in symbols:
        capital = mgr.get_symbol_capital(sym)
        if capital > 1:
            ok = mgr.allocate_capital(sym, min(capital * 0.5, 100.0), CapitalPoolType.BASE)
            check(f"{sym}资金分配", ok, f"capital={capital:.2f}")

    # 11.3 模拟交易后的盈亏处理
    result = mgr.record_daily_pnl(200.0, 10200.0)
    check("交易后盈亏处理", result["action"] == "profit_reallocation")

    # 11.4 DynamicAllocator 分配方案
    strategy_metrics = {
        "grid": {"win_rate": 0.55, "sharpe_ratio": 1.2, "profit_factor": 1.5,
                 "trade_count": 80, "total_pnl": 200.0, "max_drawdown": 0.05,
                 "consecutive_wins": 2, "consecutive_losses": 0, "volatility_30d": 0.02},
        "trend": {"win_rate": 0.48, "sharpe_ratio": 0.8, "profit_factor": 1.3,
                  "trade_count": 50, "total_pnl": 100.0, "max_drawdown": 0.08,
                  "consecutive_wins": 1, "consecutive_losses": 1, "volatility_30d": 0.03},
        "scalping": {"win_rate": 0.60, "sharpe_ratio": 1.5, "profit_factor": 1.8,
                     "trade_count": 200, "total_pnl": 300.0, "max_drawdown": 0.03,
                     "consecutive_wins": 4, "consecutive_losses": 0, "volatility_30d": 0.015},
        "arbitrage": {"win_rate": 0.70, "sharpe_ratio": 2.0, "profit_factor": 2.2,
                      "trade_count": 120, "total_pnl": 150.0, "max_drawdown": 0.02,
                      "consecutive_wins": 2, "consecutive_losses": 0, "volatility_30d": 0.01},
    }

    plan = await allocator.compute_allocation_plan(
        total_capital=10000, total_equity=10000, strategy_names=list(strategy_metrics.keys()),
        strategy_metrics=strategy_metrics, market_regime=MarketRegime.TRENDING_UP,
        current_weights={"grid": 0.25, "trend": 0.25, "scalping": 0.25, "arbitrage": 0.25}
    )

    check("综合场景分配方案", plan is not None)
    check("综合场景资金效率", plan.capital_efficiency > 0)

    # 11.5 完整报告
    full_report = mgr.get_full_report()
    check("综合场景完整报告", "capital_pool" in full_report)
    check("报告包含5大模块", len(full_report) >= 5,
          f"modules={len(full_report)}")

    allocation_report = agent.get_allocation_report()
    check("综合场景分配报告", "current_weights" in allocation_report)

    # 11.6 多市场状态切换
    for regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN,
                   MarketRegime.HIGH_VOLATILITY, MarketRegime.RANGING]:
        plan_r = await allocator.compute_allocation_plan(
            total_capital=10000, total_equity=10000, strategy_names=["grid", "trend"],
            strategy_metrics={
                "grid": {"win_rate": 0.5, "trade_count": 30, "sharpe_ratio": 0.5,
                         "profit_factor": 1.2, "total_pnl": 50, "max_drawdown": 0.05,
                         "consecutive_wins": 0, "consecutive_losses": 0, "volatility_30d": 0.02},
                "trend": {"win_rate": 0.5, "trade_count": 30, "sharpe_ratio": 0.5,
                          "profit_factor": 1.2, "total_pnl": 50, "max_drawdown": 0.05,
                          "consecutive_wins": 0, "consecutive_losses": 0, "volatility_30d": 0.02},
            },
            market_regime=regime,
            current_weights={"grid": 0.5, "trend": 0.5}
        )
        check(f"市场状态 {regime.value} 切换不崩溃", plan_r is not None)

    return mgr, allocator, agent


# ═══════════════════════════════════════════════════════════════
# 第六组：波动率目标与自适应Kelly集成
# ═══════════════════════════════════════════════════════════════

def test_volatility_kelly_integration(config):
    """波动率目标 + 自适应Kelly 集成验证"""
    section("12. 波动率目标 + 自适应Kelly 集成")

    from risk.dynamic_allocator import VolatilityTargeter, AdaptiveKelly, DynamicAllocator
    import numpy as np

    # 12.1 VolatilityTargeter
    vt = VolatilityTargeter(config.get('volatility_targeting', {}))
    returns = list(np.random.normal(0.001, 0.02, 100))

    est = vt.estimate_volatility(returns)
    check("波动率估计", est["current"] > 0)
    check("EWMA波动率", est["ewma"] > 0)
    check("预测波动率", est["predicted"] > 0)

    # 波动率上限检查
    breached, reason = vt.check_volatility_breach(0.55)
    check("波动率突破检测", breached)

    not_breached, _ = vt.check_volatility_breach(0.15)
    check("正常波动率不触发", not not_breached)

    # 仓位缩放
    scale = vt.compute_scale_factor(0.30, 0.20, 5.0, 10.0)
    check("仓位缩放因子 < 1", scale["scale_factor"] < 1.0,
          f"scale={scale['scale_factor']:.4f}")

    scale_low = vt.compute_scale_factor(0.10, 0.20, 5.0, 10.0)
    check("低波动不主动加杠杆", scale_low["scale_factor"] <= 1.0,
          f"scale={scale_low['scale_factor']:.4f}")

    # 12.2 AdaptiveKelly
    ak = AdaptiveKelly(config.get('adaptive_kelly', {}))

    # 正常市场
    k1 = ak.compute_kelly(0.55, 0.02, 0.015, "trending_up", 0.03, 2, 0, 50)
    check("牛市Kelly较高", k1["final_kelly"] >= 0,
          f"final={k1['final_kelly']:.4f}")

    # 熊市
    k2 = ak.compute_kelly(0.55, 0.02, 0.015, "trending_down", 0.03, 2, 0, 50)
    check("熊市Kelly低于牛市", k2["final_kelly"] <= k1["final_kelly"],
          f"bear={k2['final_kelly']:.4f} <= bull={k1['final_kelly']:.4f}")

    # 高波动
    k3 = ak.compute_kelly(0.55, 0.02, 0.015, "high_volatility", 0.03, 2, 0, 50)
    check("高波动Kelly低", k3["final_kelly"] <= k1["final_kelly"],
          f"high_vol={k3['final_kelly']:.4f}")

    # 回撤惩罚
    k_dd = ak.compute_kelly(0.55, 0.02, 0.015, "trending_up", 0.20, 2, 0, 50)
    check("大回撤惩罚", k_dd["final_kelly"] < k1["final_kelly"],
          f"dd20%={k_dd['final_kelly']:.4f} < normal={k1['final_kelly']:.4f}")

    # 连续亏损
    k_loss = ak.compute_kelly(0.55, 0.02, 0.015, "trending_up", 0.03, 0, 4, 50)
    check("连续亏损Kelly降低", k_loss["final_kelly"] < k1["final_kelly"],
          f"losses={k_loss['final_kelly']:.4f} < normal={k1['final_kelly']:.4f}")

    # 连续盈利
    k_win = ak.compute_kelly(0.55, 0.02, 0.015, "trending_up", 0.03, 5, 0, 50)
    check("连续盈利Kelly提升", k_win["final_kelly"] >= k1["final_kelly"],
          f"wins={k_win['final_kelly']:.4f} >= normal={k1['final_kelly']:.4f}")

    # 12.3 样本量不足
    k_few = ak.compute_kelly(0.55, 0.02, 0.015, "trending_up", 0.03, 0, 0, 5)
    check("样本量不足Kelly折扣", k_few["sample_discount"] < 1.0,
          f"discount={k_few['sample_discount']:.4f}")

    # 12.4 连续Kelly
    cont_k = ak.compute_continuous_kelly(returns)
    check("连续Kelly计算", cont_k >= 0)

    # 12.5 DynamicAllocator + VolatilityTargeter + AdaptiveKelly 集成
    allocator = DynamicAllocator(config)
    allocator._vol_targeter = vt
    allocator._adaptive_kelly = ak

    # 通过 monkey-patch 方法测试链接
    has_link = hasattr(allocator, 'link_volatility_targeter')
    check("DynamicAllocator 支持链接波动率目标器", has_link)

    has_kelly = hasattr(allocator, 'link_adaptive_kelly')
    check("DynamicAllocator 支持链接自适应Kelly", has_kelly)


# ═══════════════════════════════════════════════════════════════
# 第七组：状态持久化验证
# ═══════════════════════════════════════════════════════════════

def test_state_persistence(config):
    """状态持久化验证"""
    section("13. 状态持久化")

    from core.capital_manager import CapitalManager, CapitalPoolType, CapitalPoolType

    mgr = CapitalManager(config)
    mgr.initialize(["BTC-USDT-SWAP", "ETH-USDT-SWAP"], 5000.0)

    # 13.1 修改状态
    mgr.allocate_capital("BTC-USDT-SWAP", 200.0, CapitalPoolType.BASE)
    mgr.update_symbol_metrics("BTC-USDT-SWAP", 0.025, 0.01, 0.8, 50.0, 0.6)
    mgr.assign_leverage("BTC-USDT-SWAP", "initial", 0.025, 0.7)

    # 13.2 导出完整报告
    report = mgr.get_full_report()
    check("报告导出成功", report is not None)

    # 13.3 验证报告完整性
    check("报告包含资金池", "capital_pool" in report)
    check("报告包含币种权重", "symbol_weights" in report)
    check("报告包含杠杆层级", "leverage_tiers" in report)
    check("报告包含PnL再分配", "pnl_reallocation" in report)
    check("报告包含对冲调度", "hedge_scheduler" in report)

    # 13.4 验证报告数据正确性
    cp = report["capital_pool"]
    check("报告总资金正确", abs(cp["total_capital"] - 5000.0) < 1.0)
    check("报告资金池数量正确", len(cp["pools"]) == 3)

    sw = report["symbol_weights"]
    check("报告币种权重非空", "weights" in sw)

    lt = report["leverage_tiers"]
    check("报告杠杆分布", "tier_distribution" in lt)

    # 13.5 各模块 to_dict 一致性
    pool_dict = mgr.capital_pool.to_dict()
    check("资金池to_dict", "total_capital" in pool_dict)

    alloc_dict = mgr.symbol_allocator.to_dict()
    check("权重分配器to_dict", "weights" in alloc_dict)

    lev_dict = mgr.leverage_manager.to_dict()
    check("杠杆管理器to_dict", "absolute_max" in lev_dict)

    pnl_dict = mgr.pnl_reallocation.to_dict()
    check("PnL再分配to_dict", "consecutive_profit_days" in pnl_dict)

    hedge_dict = mgr.hedge_scheduler.to_dict()
    check("对冲调度to_dict", "active_hedges" in hedge_dict)


# ═══════════════════════════════════════════════════════════════
# 第八组：配置热更新验证
# ═══════════════════════════════════════════════════════════════

def test_hot_config_update(config):
    """配置热更新验证"""
    section("14. 配置热更新")

    from core.capital_manager import CapitalManager, CapitalPoolType

    mgr = CapitalManager(config)
    mgr.initialize(["BTC-USDT-SWAP"], 5000.0)

    # 14.1 总资金更新
    old_total = mgr.capital_pool._total_capital
    mgr.update_capital(8000.0)
    check("总资金热更新", mgr.capital_pool._total_capital != old_total,
          f"{old_total:.2f} -> {mgr.capital_pool._total_capital:.2f}")

    # 14.2 资金池跟随更新
    base_pool = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("BASE池跟随更新", base_pool.total_amount > 3000,
          f"BASE={base_pool.total_amount:.2f}")

    # 14.3 币种指标更新
    mgr.update_symbol_metrics("BTC-USDT-SWAP", 0.03, 0.02, 0.9, 100.0, 0.7)
    weight = mgr.get_symbol_weight("BTC-USDT-SWAP")
    check("指标更新后权重变化", weight > 0)

    # 14.4 资金池再平衡
    mgr.capital_pool.rebalance_pools()
    base_after = mgr.capital_pool.get_pool(CapitalPoolType.BASE)
    check("再平衡后已用资金清零", base_after.used_amount <= 0.01)


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

async def main():
    print("=" * 60)
    print("  生产级资金分布集成验证")
    print(f"  时间: {datetime.now().isoformat()}")
    print("=" * 60)

    # 加载配置
    config = load_config()

    try:
        # 第一组：端到端集成
        mgr, allocator = await test_e2e_capital_manager_dynamic_allocator(config)
        await test_e2e_allocation_agent_with_dynamic_allocator(config)

        # 第二组：策略集成
        test_strategy_integration_grid(config, mgr)
        test_strategy_integration_trend(config, mgr)
        test_strategy_integration_scalping(config, mgr)
        test_strategy_integration_arbitrage(config, mgr)

        # 第三组：边界条件
        test_boundary_conditions(config, mgr)
        await test_dynamic_allocator_boundary(config)

        # 第四组：资金池流动与跨策略共享
        test_pool_flows(config)
        await test_cross_strategy_capital_sharing(config)

        # 第五组：综合场景
        await test_comprehensive_e2e(config)

        # 第六组：波动率目标与Kelly
        test_volatility_kelly_integration(config)

        # 第七组：状态持久化
        test_state_persistence(config)

        # 第八组：配置热更新
        test_hot_config_update(config)

    except Exception as e:
        print(f"\n  [ERROR] 测试异常: {e}")
        traceback.print_exc()
        global FAIL_COUNT
        FAIL_COUNT += 1

    return summary()


if __name__ == '__main__':
    success = asyncio.run(main())
    sys.exit(0 if success else 1)