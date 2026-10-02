"""
ContributionAnalyzer P0/P1 计算 bug 修复回归测试
================================================
覆盖：
  P0-1 集中度 HHI 改用资金配置权重（不再用 PnL 份额，避免无界/负值）
  P0-2 健康度 ra_score/ce_score clamp 下限，composite clamp [0,100]
  P1-3 max_drawdown clamp 到 [0,1]，防权益曲线负值导致量纲越界
  P1-4 重分配建议复用快照，避免重复 analyze 口径不一致
  P1-5 权重归一化 + 排除 manual_override 等非资金池标签
"""
from types import SimpleNamespace

import pytest

from analysis.contribution_analyzer import ContributionAnalyzer, StrategyContribution


def _contrib(strategy, **kw):
    defaults = dict(total_trades=10, win_rate=0.5, profit_factor=1.0,
                    max_drawdown=0.1, risk_adjusted_contribution=0.0,
                    pnl_per_capital_pct=0.0, fee_ratio=0.0, trend="stable")
    defaults.update(kw)
    return StrategyContribution(strategy=strategy, **defaults)


# ── P0-1 集中度 HHI ───────────────────────────────────────

def test_concentration_bounded_with_negative_pnl():
    """负 PnL、总和近 0 时 HHI 仍 ∈ [0,1]，不再出现 300+（30055%）。"""
    analyzer = ContributionAnalyzer()
    analyzer.set_dynamic_allocations({"scalping": 0.2349, "sync": 0.2, "grid": 0.1306})
    contribs = {
        "scalping": _contrib("scalping", total_pnl=-0.5518),
        "sync": _contrib("sync", total_pnl=1.5691),
        "grid": _contrib("grid", total_pnl=-1.1334),
    }
    hhi = analyzer._compute_concentration(contribs)
    assert 0.0 <= hhi <= 1.0


def test_concentration_uses_allocation_not_pnl():
    """HHI 应只依赖注入的权重，与 PnL 数值无关。"""
    analyzer = ContributionAnalyzer()
    analyzer.set_dynamic_allocations({"a": 0.6, "b": 0.4})
    contribs = {
        "a": _contrib("a", total_pnl=-1000.0),
        "b": _contrib("b", total_pnl=1000.0),
    }
    hhi = analyzer._compute_concentration(contribs)
    # 归一化权重 [0.6, 0.4] → 0.36 + 0.16 = 0.52
    assert abs(hhi - 0.52) < 1e-9


def test_concentration_fallback_equal_weight():
    """无注入权重时回退等权（1/N，最分散）。"""
    analyzer = ContributionAnalyzer()
    contribs = {"a": _contrib("a"), "b": _contrib("b"), "c": _contrib("c")}
    hhi = analyzer._compute_concentration(contribs)
    assert abs(hhi - 1.0 / 3) < 1e-9


def test_concentration_overallocated_normalized():
    """权重之和 > 1 时先归一化，HHI 仍 ∈ [0,1]。"""
    analyzer = ContributionAnalyzer()
    analyzer.set_dynamic_allocations({"a": 0.5, "b": 0.5, "c": 0.4})
    contribs = {"a": _contrib("a"), "b": _contrib("b"), "c": _contrib("c")}
    hhi = analyzer._compute_concentration(contribs)
    assert 0.0 <= hhi <= 1.0


# ── P0-2 健康度 clamp ─────────────────────────────────────

def test_health_score_clamped_non_negative():
    """负 pnl_per_capital_pct / 负 risk_adjusted_contribution 不再产出负健康度。"""
    analyzer = ContributionAnalyzer()
    contrib = _contrib(
        "grid", total_trades=10, win_rate=0.0, profit_factor=0.5,
        risk_adjusted_contribution=-23.71, pnl_per_capital_pct=-23.71,
        fee_ratio=10.0, trend="declining",
    )
    analyzer._compute_health_scores({"grid": contrib})
    assert 0.0 <= contrib.health_score <= 100.0
    # 负贡献被 clamp 后至少不为负，等级仍为 F（低健康度）
    assert contrib.health_grade == "F"


def test_health_score_clamped_to_100():
    """极端正输入时 health_score 不越界 100。"""
    analyzer = ContributionAnalyzer()
    contrib = _contrib(
        "a", total_trades=10, win_rate=1.0, profit_factor=10.0,
        risk_adjusted_contribution=1000.0, pnl_per_capital_pct=1000.0,
        fee_ratio=0.0, trend="improving",
    )
    analyzer._compute_health_scores({"a": contrib})
    assert contrib.health_score <= 100.0
    assert contrib.health_grade == "A"


def test_health_score_na_below_sample():
    """低于最小样本量仍标记 N/A，不受 clamp 影响。"""
    analyzer = ContributionAnalyzer()
    contrib = _contrib("x", total_trades=3, pnl_per_capital_pct=-50.0)
    analyzer._compute_health_scores({"x": contrib})
    assert contrib.health_score == 0.0
    assert contrib.health_grade == "N/A"


# ── P1-3 max_drawdown 量纲 ──────────────────────────────────

def test_max_drawdown_clamped_when_curve_goes_negative():
    """权益曲线（累计 PnL）先正后深负时，回撤不再越界为 11948%。"""
    curve = [0.0, 0.5, -1.19]
    dd = ContributionAnalyzer._calc_max_drawdown(curve)
    assert dd == 1.0
    assert 0.0 <= dd <= 1.0


def test_max_drawdown_normal_curve():
    """正常非负曲线回撤计算不变。"""
    curve = [0.0, 1.0, 0.8, 0.9]
    dd = ContributionAnalyzer._calc_max_drawdown(curve)
    assert abs(dd - 0.2) < 1e-9


# ── P1-4 重分配建议复用快照 ────────────────────────────────

def test_reallocation_reuses_passed_snapshot():
    """传入 snapshot 时复用其健康度，不再重复 analyze（口径一致）。"""
    analyzer = ContributionAnalyzer()
    analyzer.set_dynamic_allocations({"grid": 0.3})
    contrib = StrategyContribution(
        strategy="grid", total_trades=10, health_score=70.0, health_grade="B",
        lifecycle="mature", trend="improving", pnl_per_capital_pct=10.0,
        pnl_contribution_pct=30.0,
    )
    fake_snapshot = SimpleNamespace(strategies={"grid": contrib})
    suggestions = analyzer.get_capital_reallocation_suggestions(snapshot=fake_snapshot)
    assert len(suggestions) == 1
    assert suggestions[0]["strategy"] == "grid"
    assert suggestions[0]["health_score"] == 70.0
    assert suggestions[0]["health_grade"] == "B"


# ── P1-5 权重归一化 + 排除非资金池标签 ──────────────────────

def test_normalized_allocation_pool():
    """归一化仅保留正权重，剔除零值/负值，和为 1.0。"""
    analyzer = ContributionAnalyzer()
    analyzer.set_dynamic_allocations({"a": 0.6, "b": 0.6, "c": 0.0, "d": -1.0})
    pool = analyzer._normalized_allocation_pool()
    assert set(pool.keys()) == {"a", "b"}
    assert abs(sum(pool.values()) - 1.0) < 1e-9
    assert abs(pool["a"] - 0.5) < 1e-9
    assert abs(pool["b"] - 0.5) < 1e-9


def test_reallocation_skips_non_pool_labels():
    """manual_override / 未配置权重标签（sync）不参与重配，不混入 0.2 兜底值。"""
    analyzer = ContributionAnalyzer()
    analyzer.set_dynamic_allocations({"grid": 0.5, "trend": 0.5})
    contribs = {
        "grid": _contrib("grid", total_trades=10, health_score=50.0, health_grade="C",
                         lifecycle="newborn", pnl_per_capital_pct=-1.0),
        "trend": _contrib("trend", total_trades=10, health_score=40.0, health_grade="D",
                          lifecycle="newborn", pnl_per_capital_pct=-2.0),
        "manual_override": _contrib("manual_override", total_trades=5,
                                    health_score=0.0, health_grade="N/A"),
        "sync": _contrib("sync", total_trades=14, health_score=47.5, health_grade="C"),
    }
    fake_snapshot = SimpleNamespace(strategies=contribs)
    suggestions = analyzer.get_capital_reallocation_suggestions(snapshot=fake_snapshot)
    names = {s["strategy"] for s in suggestions}
    assert names == {"grid", "trend"}
    assert "manual_override" not in names
    assert "sync" not in names


def test_reallocation_normalizes_weights():
    """资金池权重之和 >1 时，current_allocation 归一化到和为 1.0。"""
    analyzer = ContributionAnalyzer()
    analyzer.set_dynamic_allocations({"a": 0.6, "b": 0.6})  # 和为 1.2
    contribs = {
        "a": _contrib("a", total_trades=10, health_score=70.0, health_grade="B",
                      lifecycle="mature", pnl_per_capital_pct=10.0, trend="improving"),
        "b": _contrib("b", total_trades=10, health_score=40.0, health_grade="D",
                      lifecycle="newborn", pnl_per_capital_pct=-5.0, trend="declining"),
    }
    fake_snapshot = SimpleNamespace(strategies=contribs)
    suggestions = analyzer.get_capital_reallocation_suggestions(snapshot=fake_snapshot)
    total = sum(s["current_allocation"] for s in suggestions)
    assert abs(total - 1.0) < 1e-9
    for s in suggestions:
        assert abs(s["current_allocation"] - 0.5) < 1e-9


# ── P2-7 负期望早期熔断信号 ────────────────────────────────

def test_calc_consecutive_losses():
    assert ContributionAnalyzer._calc_consecutive_losses([]) == 0
    assert ContributionAnalyzer._calc_consecutive_losses([1.0, 2.0, 3.0]) == 0
    assert ContributionAnalyzer._calc_consecutive_losses([-1, -2, -3, -4, -5, 1]) == 5
    assert ContributionAnalyzer._calc_consecutive_losses([-1, 1, -2, -3, -4, 1]) == 3
    assert ContributionAnalyzer._calc_consecutive_losses([0, -1, -2, 0, -3]) == 2



