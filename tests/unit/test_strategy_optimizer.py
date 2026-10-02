"""StrategyOptimizer 交易分配归一化 — 单元测试。

背景：persist_config 写回 config.yaml 前会 _round_floats 对每个字段独立四舍五入到
6 位小数，浮点误差累积会导致 6 个 allocation 之和变成 0.999999 而非 1.0，进而触发
AppConfig 的「allocations sum must be 1.0」校验失败。_normalize_trading_allocations
把误差集中到值最大的字段，确保总和精确回到 1.0。
"""

from analysis.strategy_optimizer import (
    TRADING_ALLOCATION_FIELDS,
    _normalize_trading_allocations,
)


def _sum(cfg):
    return sum(float(cfg.get(f) or 0.0) for f in TRADING_ALLOCATION_FIELDS)


def test_normalize_fixes_underflow():
    """总和 0.999999（四舍五入下溢）→ 归一化后精确 1.0。"""
    cfg = {
        "grid_allocation": 0.077909,
        "spot_grid_allocation": 0.0,
        "spot_martingale_allocation": 0.0,
        "trend_allocation": 0.652615,
        "scalping_allocation": 0.269475,
        "arbitrage_allocation": 0.0,
    }
    _normalize_trading_allocations(cfg)
    assert abs(_sum(cfg) - 1.0) < 1e-6


def test_normalize_fixes_overflow():
    """总和 1.000001（四舍五入上溢）→ 归一化后精确 1.0。"""
    cfg = {
        "grid_allocation": 0.077910,
        "spot_grid_allocation": 0.0,
        "spot_martingale_allocation": 0.0,
        "trend_allocation": 0.652616,
        "scalping_allocation": 0.269475,
        "arbitrage_allocation": 0.0,
    }
    _normalize_trading_allocations(cfg)
    assert abs(_sum(cfg) - 1.0) < 1e-6


def test_normalize_preserves_zero_disabled_fields():
    """归一化只调整值最大的字段，不改已置 0 的 disabled 字段。"""
    cfg = {
        "grid_allocation": 0.077909,
        "spot_grid_allocation": 0.0,
        "spot_martingale_allocation": 0.0,
        "trend_allocation": 0.652615,
        "scalping_allocation": 0.269475,
        "arbitrage_allocation": 0.0,
    }
    _normalize_trading_allocations(cfg)
    assert cfg["spot_grid_allocation"] == 0.0
    assert cfg["spot_martingale_allocation"] == 0.0
    assert cfg["arbitrage_allocation"] == 0.0
    # 误差被加到值最大的字段（trend）
    assert cfg["trend_allocation"] == 0.652616


def test_normalize_noop_when_already_one():
    """总和已精确 = 1.0 → 保持不变。"""
    cfg = {
        "grid_allocation": 0.1,
        "spot_grid_allocation": 0.0,
        "spot_martingale_allocation": 0.0,
        "trend_allocation": 0.6,
        "scalping_allocation": 0.3,
        "arbitrage_allocation": 0.0,
    }
    before = dict(cfg)
    _normalize_trading_allocations(cfg)
    assert cfg == before


def test_normalize_handles_empty_and_non_dict():
    """空 dict / 非 dict / 全 0 均不报错且不改动。"""
    _normalize_trading_allocations({})
    _normalize_trading_allocations(None)
    _normalize_trading_allocations("not a dict")
    all_zero = {f: 0.0 for f in TRADING_ALLOCATION_FIELDS}
    _normalize_trading_allocations(all_zero)
    assert all_zero == {f: 0.0 for f in TRADING_ALLOCATION_FIELDS}
