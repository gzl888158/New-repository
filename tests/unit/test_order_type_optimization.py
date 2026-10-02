"""
挂单类型优化（recommend_order_type，滑点/波动率分级）单元测试
========================================================
覆盖：按紧急程度 + 市场波动率分级推荐 market/limit、拆分计划集成。
"""
import pytest

from execution.algo_orders.smart_order_router import (
    SmartOrderRouter, OrderUrgency,
)


def _make_router():
    r = SmartOrderRouter({
        "smart_order_router": {
            "anti_gaming": False,
            "random_delay_ms": [0, 0],
            "random_size_pct": 0.0,
        }
    })
    r.setup_default_venues()
    return r


# ── 紧急程度分级 ────────────────────────────────────────

def test_recommend_order_type_immediate_is_market():
    r = _make_router()
    assert r.recommend_order_type("BTC-USDT-SWAP", OrderUrgency.IMMEDIATE) == "market"


def test_recommend_order_type_high_is_market():
    r = _make_router()
    assert r.recommend_order_type("BTC-USDT-SWAP", OrderUrgency.HIGH) == "market"


def test_recommend_order_type_low_is_limit():
    r = _make_router()
    assert r.recommend_order_type("BTC-USDT-SWAP", OrderUrgency.LOW) == "limit"


# ── 波动率分级（滑点分级） ─────────────────────────────

def test_recommend_order_type_normal_market_in_high_volatility():
    r = _make_router()
    r.update_market_state("BTC-USDT-SWAP", volatility_pct=8.0)
    assert r.recommend_order_type("BTC-USDT-SWAP", OrderUrgency.NORMAL) == "market"


def test_recommend_order_type_normal_limit_in_low_volatility():
    r = _make_router()
    r.update_market_state("BTC-USDT-SWAP", volatility_pct=1.0)
    assert r.recommend_order_type("BTC-USDT-SWAP", OrderUrgency.NORMAL) == "limit"


# ── 拆分计划集成 ────────────────────────────────────────

def test_create_splitting_plan_market_for_immediate():
    r = _make_router()
    rankings = r.rank_venues("buy", 60000, urgency=OrderUrgency.IMMEDIATE)
    plan = r.create_splitting_plan(
        "ord_imm", "BTC-USDT-SWAP", "buy", 0.01, 60000, rankings, OrderUrgency.IMMEDIATE)
    assert plan.slices
    assert all(s.order_type == "market" for s in plan.slices)


def test_create_splitting_plan_limit_for_normal():
    r = _make_router()
    rankings = r.rank_venues("buy", 60000, urgency=OrderUrgency.NORMAL)
    plan = r.create_splitting_plan(
        "ord_norm", "BTC-USDT-SWAP", "buy", 0.01, 60000, rankings, OrderUrgency.NORMAL)
    assert plan.slices
    assert all(s.order_type == "limit" for s in plan.slices)


def test_create_splitting_plan_v2_market_for_immediate():
    r = _make_router()
    rankings = r.rank_venues("buy", 60000, urgency=OrderUrgency.IMMEDIATE)
    plan = r.create_splitting_plan_v2(
        "ord_v2_imm", "BTC-USDT-SWAP", "buy", 0.01, 60000, rankings, OrderUrgency.IMMEDIATE)
    assert plan.slices
    assert all(s.order_type == "market" for s in plan.slices)


def test_create_splitting_plan_v2_limit_for_normal():
    r = _make_router()
    rankings = r.rank_venues("buy", 60000, urgency=OrderUrgency.NORMAL)
    plan = r.create_splitting_plan_v2(
        "ord_v2_norm", "BTC-USDT-SWAP", "buy", 0.01, 60000, rankings, OrderUrgency.NORMAL)
    assert plan.slices
    assert all(s.order_type == "limit" for s in plan.slices)
