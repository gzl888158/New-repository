"""grid 参数网格扫描真实历史K线复核（direction ③ offline review）。

背景：
  方向②用「合成震荡 K 线」对 grid_spacing × tp_pct × sl_pct 做笛卡尔积扫描，
  得出「震荡行情应选宽网格(gs≥0.04) + 宽止损(sl≥0.02)」的结论。
  但合成震荡行情是 ±6% 正弦波 + 0.15% 噪声的「完美均值回归」，真实 1H K 线
  混杂趋势/噪声，结论未必能直接迁移。

本脚本（direction ③）：
  1. 抓取 OKX 真实历史 1H K 线（近 30 天，与 reports/grid_backtest_report_20260914.md 同口径）
  2. 对每个币种跑同一套 grid_spacing × tp_pct × sl_pct 笛卡尔积扫描
  3. 对比「宽网格 vs 窄网格 vs 生产默认」在真实数据下的 ROI
  4. 复核方向②的「宽网格 + 宽止损」结论是否在真实数据上成立

用法（Windows venv）:
    .venv\\Scripts\\python.exe scripts\\_verify_grid_params_real.py
"""
import os
import sys
from itertools import product

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import yaml
from backtest.backtest_engine import BacktestEngine

with open(os.path.join(BASE, "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

engine = BacktestEngine(cfg)
SYMBOLS = ["ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP", "SUI-USDT-SWAP"]

GRID_SPACINGS = [0.01, 0.02, 0.03, 0.04, 0.05]
TP_PCTS = [0.005, 0.01, 0.015, 0.02, 0.025]
SL_PCTS = [0.005, 0.01, 0.015, 0.02, 0.03]


def _run(eng, candles, gs, tp, sl, min_q):
    """单次回测，返回 ROI/交易数/胜率。守卫参数可开关以便对比纯参数效果 vs 生产门槛。"""
    r = eng.run_grid_with_candles(
        candles, "X-USDT-SWAP",
        grid_spacing=gs, tp_pct=tp, sl_pct=sl,
        min_signal_quality=min_q,
        trend_filter_threshold=0.0,
        wear_fee_multiple=0.0,
    )
    s = r.summary()
    if "error" in s:
        return {"roi": -100.0, "trades": 0, "win": 0.0}
    return {
        "roi": float(s["roi_percent"]),
        "trades": int(s["total_trades"]),
        "win": float(s["win_rate_percent"]),
    }


def _scan(symbol, candles, min_q):
    """对单币种跑完整笛卡尔积，返回 (gs, tp, sl, roi, trades, win) 列表 + 最佳组合。"""
    rows = []
    for gs, tp, sl in product(GRID_SPACINGS, TP_PCTS, SL_PCTS):
        m = _run(engine, candles, gs, tp, sl, min_q)
        rows.append((gs, tp, sl, m["roi"], m["trades"], m["win"]))
    best = max(rows, key=lambda r: r[3])
    return rows, best


def _trend_strength(candles, fast=20, slow=50):
    """用 |EMA20-EMA50|/EMA50 估算趋势强度，判断币种当前偏趋势还是偏震荡。"""
    from backtest.backtest_engine import BacktestEngine
    closes = [float(c["close"]) for c in candles]
    ema_f = BacktestEngine._calc_ema(engine, closes, fast)
    ema_s = BacktestEngine._calc_ema(engine, closes, slow)
    strength = abs(ema_f[-1] - ema_s[-1]) / ema_s[-1] if ema_s[-1] > 0 else 0.0
    return strength


def main():
    print("=" * 100)
    print("grid 参数网格扫描真实历史K线复核（direction ③）")
    print(f"  数据源: OKX history-candles 1H  近30天  代理={cfg.get('okx', {}).get('proxy')}")
    print(f"  参数空间: gs∈{GRID_SPACINGS} tp∈{TP_PCTS} sl∈{SL_PCTS} = {len(GRID_SPACINGS)*len(TP_PCTS)*len(SL_PCTS)} 组合/币")
    print("=" * 100)

    all_rows = []
    print("\n[1] 纯参数扫描（min_signal_quality=0，关闭守卫，与方向②同口径）")
    print(f"{'symbol':16s} {'趋势强度':>8s} {'最优gs':>7s} {'最优tp':>7s} {'最优sl':>7s} "
          f"{'最优ROI%':>9s} {'默认ROI%':>9s} {'宽网格胜?':>8s}")
    for sym in SYMBOLS:
        candles = engine.fetch_historical_klines(sym, bar="1H", days=30)
        if len(candles) < 60:
            print(f"  {sym}: 数据不足({len(candles)})，跳过")
            continue
        rows, best = _scan(sym, candles, min_q=0.0)
        strength = _trend_strength(candles)
        default = next(r for r in rows if r[0] == 0.03 and r[1] == 0.02 and r[2] == 0.02)
        wide_wins = best[0] >= 0.04
        print(f"{sym:16s} {strength:>8.4f} {best[0]:>7.3f} {best[1]:>7.3f} {best[2]:>7.3f} "
              f"{best[3]:>9.2f} {default[3]:>9.2f} {'是' if wide_wins else '否':>8s}")
        all_rows.append((sym, strength, best, default, wide_wins))

    # 汇总：宽网格是否在真实数据上一致胜出
    wide_wins_count = sum(1 for r in all_rows if r[4])
    print(f"\n  >> 宽网格(gs≥0.04)胜出的币种: {wide_wins_count}/{len(all_rows)}")
    print(f"  >> 所有币种纯参数最优 ROI 是否 > 0: {sum(1 for r in all_rows if r[2][3] > 0)}/{len(all_rows)}")

    # [2] 生产信号门槛下的复核（min_signal_quality=0.35，与 reports/grid_backtest_report 同口径）
    print("\n[2] 生产信号门槛扫描（min_signal_quality=0.35，验证「信号质量才是命门」）")
    print(f"{'symbol':16s} {'最优gs':>7s} {'最优tp':>7s} {'最优sl':>7s} {'最优ROI%':>9s} {'交易数':>6s} {'胜率%':>6s}")
    for sym in SYMBOLS:
        candles = engine.fetch_historical_klines(sym, bar="1H", days=30)
        if len(candles) < 60:
            continue
        rows, best = _scan(sym, candles, min_q=0.35)
        print(f"{sym:16s} {best[0]:>7.3f} {best[1]:>7.3f} {best[2]:>7.3f} "
              f"{best[3]:>9.2f} {best[4]:>6d} {best[5]:>6.1f}")

    print("\n[DONE] direction ③ 真实K线复核完成")


if __name__ == "__main__":
    main()
