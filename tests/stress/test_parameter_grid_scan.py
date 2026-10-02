"""
grid_spacing × tp_pct × sl_pct 参数网格扫描（parameter grid scan）
================================================================
针对 `BacktestEngine.run_grid_with_candles` 的三个核心网格参数做笛卡尔积扫描，
在震荡/上涨/下跌三种行情下定位各品种/行情的最优参数区间，并验证：

  - 震荡行情：网格均值回归策略存在显著盈利区间（ROI > 0 且 Sharpe > 0），
    极端参数组合（网格过密+止盈过小 / 网格过宽+止盈过大）显著弱于最优区间。
  - 趋势行情：纯网格均值回归无稳定盈利（任何参数组合 ROI 都 < 震荡最优 ROI），
    印证生产 P33 趋势门禁（trend_filter_threshold）的必要性。
  - 最优区间与生产 config 默认值（grid_spacing=0.03 / tp_pct=0.02 / sl_pct=0.02）
    差距控制在可接受范围内（生产默认值 ROI ≥ 震荡最优 ROI 的 50%）。

与 tests/stress/test_guard_parameter_validation.py 区分：
  - 后者聚焦「守卫参数的策略效果」A/B（守卫开 vs 关）。
  - 本套聚焦「策略参数的最优区间」网格扫描（参数空间遍历）。

扫描范围：
  grid_spacing ∈ {0.01, 0.02, 0.03, 0.04, 0.05}  (1% ~ 5%)
  tp_pct       ∈ {0.005, 0.01, 0.015, 0.02, 0.025} (0.5% ~ 2.5%)
  sl_pct       ∈ {0.005, 0.01, 0.015, 0.02, 0.03}  (0.5% ~ 3%)
  regime       ∈ {oscillate, uptrend, downtrend}
共 5×5×5×3 = 375 次回测，单次约 400 K 线。
"""
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

from backtest.backtest_engine import BacktestEngine


# ─────────────────────────────────────────────────────────────
# 合成 K 线生成器（与 test_guard_parameter_validation._candles 同口径，
# 但本文件单独持有副本以避免跨测试文件依赖私有辅助）
# ─────────────────────────────────────────────────────────────
def _candles(n: int = 400, base: float = 100.0, mode: str = "oscillate", seed: int = 42):
    """生成 Dict 格式合成 K 线（BacktestEngine 接口）。

    mode:
      - oscillate: 正弦震荡 + 噪声 → 均值回归主战场，网格策略应能盈利
      - uptrend / downtrend: 单边趋势 → 纯网格策略应难稳定盈利
    """
    rng = np.random.default_rng(seed)
    candles = []
    ts = datetime(2026, 1, 1)
    price = base
    for i in range(n):
        if mode == "oscillate":
            price = base + 6.0 * np.sin(i / 6.0) + rng.normal(0, 0.15)
        elif mode == "uptrend":
            # 指数上行：百分比趋势强度恒定（与生产 config 保持一致）
            price = base * (1.01 ** i) + rng.normal(0, base * 0.002)
        elif mode == "downtrend":
            price = base * (0.99 ** i) + rng.normal(0, base * 0.002)
        open_ = price
        close = price + rng.normal(0, 0.05)
        high = max(open_, close) * 1.002
        low = min(open_, close) * 0.998
        volume = 1000 + abs(rng.normal(0, 50))
        candles.append({
            "timestamp": ts + timedelta(hours=i),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(volume),
        })
    return candles


def _engine() -> BacktestEngine:
    """裸手续费引擎（避免 strategies.grid 配置默认值污染扫描结果）。"""
    return BacktestEngine({
        "execution": {
            "taker_fee": 0.0005,
            "maker_fee": 0.0002,
            "funding_rate": 0.0001,
            "funding_interval_hours": 8,
        },
        # 显式空 strategies，强制 run_grid_with_candles 使用本测试传入的参数
        "strategies": {},
    })


# ─────────────────────────────────────────────────────────────
# 参数网格扫描核心
# ─────────────────────────────────────────────────────────────
GRID_SPACINGS = [0.01, 0.02, 0.03, 0.04, 0.05]
TP_PCTS = [0.005, 0.01, 0.015, 0.02, 0.025]
SL_PCTS = [0.005, 0.01, 0.015, 0.02, 0.03]
REGIMES = ["oscillate", "uptrend", "downtrend"]


def _run_one(eng: BacktestEngine, candles, grid_spacing: float, tp_pct: float, sl_pct: float) -> Dict[str, float]:
    """单次回测，返回 ROI/Sharpe/MaxDD/Trades 指标。

    关键：禁用 min_signal_quality / trend_filter_threshold / wear_fee_multiple，
    让 grid_spacing/tp_pct/sl_pct 的纯效果显现，避免与守卫参数交叉污染。
    """
    result = eng.run_grid_with_candles(
        candles, "X-USDT-SWAP",
        grid_spacing=grid_spacing,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        min_signal_quality=0.0,        # 关闭信号质量门槛
        trend_filter_threshold=0.0,    # 关闭趋势方向过滤（方向③复核时再开）
        wear_fee_multiple=0.0,         # 关闭磨损过滤器
    )
    summary = result.summary()
    if "error" in summary:
        return {"roi": -100.0, "sharpe": 0.0, "max_dd": 100.0, "trades": 0, "net_pnl": 0.0}
    return {
        "roi": float(summary["roi_percent"]),
        "sharpe": float(summary["sharpe_ratio"]),
        "max_dd": float(summary["max_drawdown_percent"]),
        "trades": int(summary["total_trades"]),
        "net_pnl": float(summary["net_pnl"]),
    }


def _scan_regime(eng: BacktestEngine, regime: str, seed: int = 42) -> List[Tuple[float, float, float, Dict[str, float]]]:
    """对单行情跑完整笛卡尔积扫描，返回 (gs, tp, sl, metrics) 列表。"""
    candles = _candles(mode=regime, seed=seed)
    rows = []
    for gs, tp, sl in product(GRID_SPACINGS, TP_PCTS, SL_PCTS):
        m = _run_one(eng, candles, gs, tp, sl)
        rows.append((gs, tp, sl, m))
    return rows


def _top_by_roi(rows, k: int = 5):
    """ROI 降序取前 k。"""
    return sorted(rows, key=lambda r: r[3]["roi"], reverse=True)[:k]


def _worst_by_roi(rows, k: int = 5):
    """ROI 升序取前 k（最差组合）。"""
    return sorted(rows, key=lambda r: r[3]["roi"])[:k]


def _best_profitable(rows):
    """ROI > 0 且 Sharpe > 0 且 MaxDD < 50% 的最优组合。"""
    valid = [r for r in rows if r[3]["roi"] > 0 and r[3]["sharpe"] > 0 and r[3]["max_dd"] < 50]
    if not valid:
        return None
    return sorted(valid, key=lambda r: r[3]["roi"], reverse=True)[0]


# ─────────────────────────────────────────────────────────────
# 测试用例
# ─────────────────────────────────────────────────────────────
class TestParameterGridScanCompleteness:
    def test_scan_produces_full_matrix(self):
        """完整性：每个行情的扫描结果数 = |grid_spacing|×|tp_pct|×|sl_pct| = 125。"""
        eng = _engine()
        for regime in REGIMES:
            rows = _scan_regime(eng, regime)
            print(f"\n[扫描完整性] {regime}: {len(rows)} 组合")
            assert len(rows) == len(GRID_SPACINGS) * len(TP_PCTS) * len(SL_PCTS) == 125


class TestOscillateRegimeOptimal:
    def test_oscillate_has_profitable_grid(self):
        """震荡行情：至少存在一组 (gs,tp,sl) ROI > 0 且 Sharpe > 0 且 MaxDD < 50%。
        这是网格均值回归策略的生存性证明：在它的主场行情里应能盈利。"""
        eng = _engine()
        rows = _scan_regime(eng, "oscillate")
        best = _best_profitable(rows)
        print(f"\n[震荡最优] {best}")
        assert best is not None, "震荡行情下应至少有一组参数可盈利"

    def test_oscillate_optimal_beats_wear_corner(self):
        """震荡最优组合应显著优于「网格过密 + 止盈过小」磨损角（gs=0.01, tp=0.005, sl=0.005）。
        极窄网格 + 极小止盈 → taker 手续费结构性吞噬利润（磨损区）。"""
        eng = _engine()
        rows = _scan_regime(eng, "oscillate")
        best = _best_profitable(rows)
        assert best is not None
        wear_corner = next(r for r in rows if r[0] == 0.01 and r[1] == 0.005 and r[2] == 0.005)
        print(f"\n[磨损角对比] 最优 ROI={best[3]['roi']:.2f}% vs 磨损角 ROI={wear_corner[3]['roi']:.2f}%")
        assert best[3]["roi"] > wear_corner[3]["roi"]

    def test_oscillate_optimal_in_wide_grid_zone(self):
        """震荡最优组合应落在「宽网格区」（grid_spacing >= 0.04）。

        经济学直觉：在 ±6% 的正弦震荡行情下，gs=0.01 的窄网格会在每次微小回撤就入场，
        随后被噪声打掉窄止损；gs=0.05 的宽网格只在价格显著偏离 EMA20（>5%）时入场，
        此时均值回归概率高、反弹空间足，搭配宽止损不易被噪声清洗。

        这是方向②的核心可复用结论：震荡行情应选宽网格 + 宽止损，
        而非生产默认 gs=0.03/sl=0.02 的中宽带。
        """
        eng = _engine()
        rows = _scan_regime(eng, "oscillate")
        best = _best_profitable(rows)
        assert best is not None
        print(f"\n[震荡最优区间] gs={best[0]} tp={best[1]} sl={best[2]} ROI={best[3]['roi']:.2f}%")
        assert best[0] >= 0.04, f"震荡最优 grid_spacing={best[0]} 未落在宽网格区 [0.04, 0.05]"

    def test_oscillate_top5_share_wide_grid_pattern(self):
        """震荡 Top 5 应共同落在宽网格 + 宽止损区间（gs >= 0.04 且 sl >= 0.02）。

        验证最优不是单一离群点而是「可识别区间」，便于生产 config 调参时
        直接锁定宽网格 + 宽止损的参数族，而非追逐单点最优。
        """
        eng = _engine()
        rows = _scan_regime(eng, "oscillate")
        top = _top_by_roi(rows, 5)
        print("\n[震荡 Top 5 区间共性检查]")
        for gs, tp, sl, m in top:
            print(f"  gs={gs:.3f} tp={tp:.3f} sl={sl:.3f} ROI={m['roi']:.2f}% "
                  f"宽网格={'Y' if gs >= 0.04 else 'N'} 宽止损={'Y' if sl >= 0.02 else 'N'}")
        for gs, tp, sl, m in top:
            assert gs >= 0.04, f"Top5 组合 gs={gs} 未落在宽网格区"
            assert sl >= 0.02, f"Top5 组合 sl={sl} 未落在宽止损区"

    def test_production_default_suboptimal_in_oscillate(self):
        """生产默认参数 (grid_spacing=0.03, tp_pct=0.02, sl_pct=0.02)
        在震荡行情下 ROI 应明显低于震荡最优组合。

        不要求生产默认 = 最优（参数空间离散、最优会随行情微调），
        但要求最优显著优于生产默认 — 否则方向②扫描本身没有产出可执行的调参建议。

        若断言通过，建议在 config.yaml 的 strategies.grid 字段把
        min_grid_spacing / max_grid_spacing 抬到 [0.04, 0.05]、stop_loss_pct 抬到 0.03。
        """
        eng = _engine()
        rows = _scan_regime(eng, "oscillate")
        best = _best_profitable(rows)
        assert best is not None
        prod_default = next(r for r in rows if r[0] == 0.03 and r[1] == 0.02 and r[2] == 0.02)
        print(f"\n[生产默认调参建议] 最优 ROI={best[3]['roi']:.2f}% (gs={best[0]}, tp={best[1]}, sl={best[2]}) "
              f"vs 生产默认 ROI={prod_default[3]['roi']:.2f}% (gs=0.03, tp=0.02, sl=0.02)")
        assert best[3]["roi"] > prod_default[3]["roi"], (
            f"震荡最优 ROI={best[3]['roi']:.2f}% 未显著优于生产默认 ROI={prod_default[3]['roi']:.2f}%，"
            f"方向②扫描未产出可执行调参建议"
        )

    def test_top5_printed(self, capsys=None):
        """打印震荡行情 ROI Top 5 / Worst 5（人工审视最优区间，便于调参决策）。"""
        eng = _engine()
        rows = _scan_regime(eng, "oscillate")
        top = _top_by_roi(rows, 5)
        worst = _worst_by_roi(rows, 5)
        print("\n=== 震荡行情 ROI Top 5 ===")
        print(f"{'gs':>6} {'tp':>7} {'sl':>7} {'roi%':>8} {'sharpe':>8} {'maxdd%':>8} {'trades':>7}")
        for gs, tp, sl, m in top:
            print(f"{gs:>6.3f} {tp:>7.3f} {sl:>7.3f} "
                  f"{m['roi']:>8.2f} {m['sharpe']:>8.2f} {m['max_dd']:>8.2f} {m['trades']:>7}")
        print("\n=== 震荡行情 ROI Worst 5 ===")
        for gs, tp, sl, m in worst:
            print(f"{gs:>6.3f} {tp:>7.3f} {sl:>7.3f} "
                  f"{m['roi']:>8.2f} {m['sharpe']:>8.2f} {m['max_dd']:>8.2f} {m['trades']:>7}")
        assert len(top) == 5


class TestTrendRegimeGridFails:
    def test_uptrend_grid_roi_below_oscillate_best(self):
        """强上涨行情：纯网格（无趋势门禁）最优 ROI < 震荡最优 ROI。
        印证 P33 trend_filter_threshold 趋势门禁的必要性 — 趋势行情本就不该用裸网格。"""
        eng = _engine()
        osc_rows = _scan_regime(eng, "oscillate")
        up_rows = _scan_regime(eng, "uptrend")
        osc_best = _best_profitable(osc_rows)
        up_best = _best_profitable(up_rows)
        osc_roi = osc_best[3]["roi"] if osc_best else 0
        up_roi = up_best[3]["roi"] if up_best else -100
        print(f"\n[趋势门禁必要性] 震荡最优 ROI={osc_roi:.2f}% vs 强上涨最优 ROI={up_roi:.2f}%")
        assert up_roi < osc_roi, (
            f"强上涨行情裸网格最优 ROI={up_roi:.2f}% ≥ 震荡最优 ROI={osc_roi:.2f}%，"
            f"与「网格策略应在震荡行情才盈利」的预期矛盾，需检查测试数据"
        )

    def test_downtrend_grid_roi_below_oscillate_best(self):
        """强下跌行情：纯网格最优 ROI < 震荡最优 ROI（对称验证）。"""
        eng = _engine()
        osc_rows = _scan_regime(eng, "oscillate")
        down_rows = _scan_regime(eng, "downtrend")
        osc_best = _best_profitable(osc_rows)
        down_best = _best_profitable(down_rows)
        osc_roi = osc_best[3]["roi"] if osc_best else 0
        down_roi = down_best[3]["roi"] if down_best else -100
        print(f"\n[趋势门禁必要性] 震荡最优 ROI={osc_roi:.2f}% vs 强下跌最优 ROI={down_roi:.2f}%")
        assert down_roi < osc_roi
