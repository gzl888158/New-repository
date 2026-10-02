"""grid 策略强化分析：真实 OKX 历史数据回测 + 手续费磨损/TP-SL 不对称/趋势方向过滤量化。

用法（Windows venv）:
    .venv\\Scripts\\python.exe scripts\\_analyze_grid.py
"""
import os, sys
from collections import Counter

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import yaml
from backtest.backtest_engine import BacktestEngine

with open(os.path.join(BASE, "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

engine = BacktestEngine(cfg)
SYMBOLS = ["ETH-USDT-SWAP", "SOL-USDT-SWAP", "XRP-USDT-SWAP", "DOGE-USDT-SWAP", "SUI-USDT-SWAP"]

# 网格核心参数（对齐生产 config strategies.grid）
GRID_CFG = cfg.get("strategies", {}).get("grid", {})
SPACING = float(GRID_CFG.get("min_grid_spacing", 0.03))
TP = float(GRID_CFG.get("tp1_pct", 2.0)) / 100.0
SL_LOCKED = float(GRID_CFG.get("stop_loss_pct", 0.078))
SL_CAP = float(GRID_CFG.get("max_stop_loss_pct", 0.02))
SL_EFFECTIVE = min(SL_LOCKED, SL_CAP)
MIN_Q = float(GRID_CFG.get("min_signal_quality", 0.35))
MAX_HOLD = float(GRID_CFG.get("max_hold_hours", 72.0))
# 趋势方向过滤阈值（对齐生产 _confirm_grid_entry: strength>0.05 禁逆势）
TREND_THRESHOLD = 0.05

print("=" * 78)
print("GRID 强化分析参数（对齐生产 config）")
print(f"  grid_spacing(入场带) = {SPACING:.2%}  (min_grid_spacing)")
print(f"  tp_pct(tp1)           = {TP:.2%}")
print(f"  stop_loss_pct(锁定)   = {SL_LOCKED:.2%}")
print(f"  max_stop_loss_pct     = {SL_CAP:.2%}")
print(f"  >> 有效止损           = {SL_EFFECTIVE:.2%}  (min 取更小值)")
print(f"  min_signal_quality    = {MIN_Q}")
print(f"  trend_filter_threshold= {TREND_THRESHOLD}  (|EMA20-EMA50|/EMA50 禁逆势)")
print(f"  max_hold_hours        = {MAX_HOLD}")
print(f"  taker_fee             = {engine.taker_fee:.4%}  往返 = {engine.taker_fee*2:.4%}")
print(f"  tp/往返手续费比       = {TP/(engine.taker_fee*2):.1f}x")
print("=" * 78)


def analyze(symbol, candles, **kw):
    r = engine.run_grid_with_candles(candles, symbol, **kw)
    s = r.summary()
    closed = [t for t in r.trades if t.status == "closed"]
    reasons = Counter(t.exit_reason for t in closed)
    wins = [t for t in closed if t.pnl > 0]
    gross_profit = sum(t.pnl for t in wins)
    total_fee = sum(t.fee for t in closed)
    return {
        "symbol": symbol,
        "trades": len(closed),
        "net_pnl": round(s.get("net_pnl", 0.0), 3),
        "final_equity": round(s.get("final_equity", 0.0), 2),
        "roi%": round(s.get("roi_percent", 0.0), 2),
        "win_rate%": round(s.get("win_rate_percent", 0.0), 1),
        "total_fee": round(total_fee, 3),
        "gross_profit": round(gross_profit, 3),
        "fee/gross_profit": round(total_fee / gross_profit, 2) if gross_profit > 0 else None,
        "reasons": dict(reasons),
        "avg_hold_h": round(s.get("avg_hold_minutes", 0.0) / 60, 1),
    }


def run_scenario(name, **kw):
    print(f"\n--- 场景: {name} ---")
    rows = []
    for sym in SYMBOLS:
        candles = engine.fetch_historical_klines(sym, bar="1H", days=30)
        if len(candles) < 25:
            print(f"  {sym}: 数据不足({len(candles)})，跳过")
            continue
        row = analyze(sym, candles, **kw)
        rows.append(row)
        print(f"  {row['symbol']:16s} trades={row['trades']:3d} net={row['net_pnl']:>8.3f} "
              f"win={row['win_rate%']:>5.1f}% fee={row['total_fee']:>6.3f} "
              f"fee/GP={row['fee/gross_profit']} reasons={row['reasons']}")
    if rows:
        tot_net = sum(r['net_pnl'] for r in rows)
        tot_fee = sum(r['total_fee'] for r in rows)
        tot_trades = sum(r['trades'] for r in rows)
        tot_gp = sum(r['gross_profit'] for r in rows)
        print(f"  合计: 净盈亏={tot_net:.3f} 手续费={tot_fee:.3f} "
              f"毛盈利={tot_gp:.3f} 手续费/毛盈利={tot_fee/tot_gp:.2f} 交易数={tot_trades}")
    return rows


# 场景 1: 裸网格（无信号门槛 + 无趋势过滤）—— 下限约束基准
run_scenario(
    "裸网格（无信号门槛 + 无趋势过滤，基准）",
    grid_spacing=SPACING, tp_pct=TP, sl_pct=SL_EFFECTIVE,
    max_hold_hours=MAX_HOLD, min_signal_quality=0.0, trend_filter_threshold=0.0,
)

# 场景 2: 仅趋势方向过滤（无信号门槛）—— 量化趋势过滤单独增量
run_scenario(
    "仅趋势方向过滤 strength>0.05 禁逆势（无信号门槛）",
    grid_spacing=SPACING, tp_pct=TP, sl_pct=SL_EFFECTIVE,
    max_hold_hours=MAX_HOLD, min_signal_quality=0.0, trend_filter_threshold=TREND_THRESHOLD,
)

# 场景 3: 仅生产信号门槛 0.35（无趋势过滤）
run_scenario(
    "仅生产信号门槛 0.35（无趋势过滤）",
    grid_spacing=SPACING, tp_pct=TP, sl_pct=SL_EFFECTIVE,
    max_hold_hours=MAX_HOLD, min_signal_quality=MIN_Q, trend_filter_threshold=0.0,
)

# 场景 4: 生产近似 = 信号门槛 0.35 + 趋势方向过滤
run_scenario(
    "生产近似：信号门槛 0.35 + 趋势方向过滤",
    grid_spacing=SPACING, tp_pct=TP, sl_pct=SL_EFFECTIVE,
    max_hold_hours=MAX_HOLD, min_signal_quality=MIN_Q, trend_filter_threshold=TREND_THRESHOLD,
)

print("\n[DONE] grid 强化分析完成（含趋势方向过滤对比）")
