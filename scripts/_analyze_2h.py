"""分析最近两小时的真实交易记录（data/trading.db -> trade_records）。

用法（Windows venv）:
    .venv\\Scripts\\python.exe scripts\\_analyze_2h.py
"""
import sqlite3, datetime, os, sys
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "data", "trading.db")

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

now = datetime.datetime.now()
window_start = now - datetime.timedelta(hours=2)
ws = window_start.strftime("%Y-%m-%d %H:%M:%S")

print("=" * 80)
print(f"最近两小时交易分析  窗口: {ws} ~ {now.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ── 1. 最近 2h 内平仓的交易（有 pnl 结果） ──
cur.execute("""
    SELECT symbol, strategy_name, side, signal_type, quantity, filled_price,
           leverage, pnl, fees, status, exit_reason, create_time, close_time
    FROM trade_records
    WHERE close_time >= ?
    ORDER BY close_time ASC
""", (ws,))
closed = [dict(r) for r in cur.fetchall()]

print(f"\n[1] 最近 2h 平仓交易: {len(closed)} 笔")
if closed:
    tot_pnl = sum(t["pnl"] or 0 for t in closed)
    tot_fee = sum(t["fees"] or 0 for t in closed)
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    losses = [t for t in closed if (t["pnl"] or 0) <= 0]
    print(f"  净盈亏={tot_pnl:.4f} USDT  手续费={tot_fee:.4f} USDT  "
          f"胜率={len(wins)/len(closed)*100:.1f}% ({len(wins)}胜/{len(losses)}负)")
    print(f"  离场原因分布: {dict(Counter(t['exit_reason'] for t in closed))}")
    print()
    # 按策略汇总
    by_strat = defaultdict(lambda: {"pnl": 0.0, "fee": 0.0, "n": 0, "win": 0})
    for t in closed:
        s = by_strat[t["strategy_name"]]
        s["pnl"] += t["pnl"] or 0
        s["fee"] += t["fees"] or 0
        s["n"] += 1
        s["win"] += 1 if (t["pnl"] or 0) > 0 else 0
    print("  按策略汇总:")
    for k, v in sorted(by_strat.items(), key=lambda x: -x[1]["pnl"]):
        print(f"    {k:18s} n={v['n']:3d} 净={v['pnl']:>8.4f} 费={v['fee']:>7.4f} 胜={v['win']/v['n']*100:4.1f}%")
    print()
    # 按币种汇总
    by_sym = defaultdict(lambda: {"pnl": 0.0, "n": 0})
    for t in closed:
        s = by_sym[t["symbol"]]
        s["pnl"] += t["pnl"] or 0
        s["n"] += 1
    print("  按币种汇总:")
    for k, v in sorted(by_sym.items(), key=lambda x: -x[1]["pnl"]):
        print(f"    {k:18s} n={v['n']:3d} 净={v['pnl']:>8.4f}")
    print()
    print("  平仓明细:")
    for t in closed:
        ct = (t["close_time"] or "")[:19]
        cr = (t["create_time"] or "")[:19]
        print(f"    {ct} {t['symbol']:16s} {t['strategy_name']:14s} {t['side']:5s} "
              f"qty={t['quantity']:.4f} px={t['filled_price']} pnl={t['pnl']:>8.4f} "
              f"fee={t['fees']:.4f} {t['exit_reason']}")

# ── 2. 最近 2h 内开仓（仍在持仓） ──
cur.execute("""
    SELECT symbol, strategy_name, side, signal_type, quantity, filled_price,
           leverage, margin, pnl, fees, status, create_time
    FROM trade_records
    WHERE create_time >= ? AND status = 'open'
    ORDER BY create_time ASC
""", (ws,))
open_pos = [dict(r) for r in cur.fetchall()]

print(f"\n[2] 最近 2h 开仓且仍持仓: {len(open_pos)} 笔")
for t in open_pos:
    ct = (t["create_time"] or "")[:19]
    print(f"    {ct} {t['symbol']:16s} {t['strategy_name']:14s} {t['side']:5s} "
          f"qty={t['quantity']:.4f} px={t['filled_price']} margin={t['margin']:.2f}")

# ── 3. 最近 2h 新开的全部记录（含已平仓） ──
cur.execute("""
    SELECT COUNT(*) FROM trade_records WHERE create_time >= ?
""", (ws,))
new_cnt = cur.fetchone()[0]
print(f"\n[3] 最近 2h 新开交易记录总数: {new_cnt} 笔")

conn.close()
print("\n[DONE] 两小时交易分析完成")
