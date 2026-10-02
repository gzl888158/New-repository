"""分析 XRP-USDT-SWAP 的真实交易记录（data/trading.db -> trade_records）。

用法（Windows venv）:
    .venv\\Scripts\\python.exe scripts\\_analyze_xrp.py
"""
import sqlite3, datetime, os
from collections import Counter, defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "data", "trading.db")
SYMBOL = "XRP-USDT-SWAP"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# ── 0. 总览 ──
cur.execute("SELECT COUNT(*), MIN(create_time), MAX(create_time) FROM trade_records WHERE symbol=?", (SYMBOL,))
total, tmin, tmax = cur.fetchone()
cur.execute("SELECT COUNT(*) FROM trade_records WHERE symbol=? AND status='open'", (SYMBOL,))
open_cnt = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM trade_records WHERE symbol=? AND status='closed'", (SYMBOL,))
closed_cnt = cur.fetchone()[0]

print("=" * 80)
print(f"XRP-USDT-SWAP 交易记录分析")
print("=" * 80)
print(f"  总记录数: {total}  平仓: {closed_cnt}  当前持仓: {open_cnt}")
print(f"  时间跨度: {tmin} ~ {tmax}")

# ── 1. 平仓交易整体 ──
cur.execute("""
    SELECT symbol, strategy_name, side, signal_type, quantity, filled_price,
           leverage, pnl, fees, status, exit_reason, create_time, close_time
    FROM trade_records
    WHERE symbol=? AND status='closed'
    ORDER BY close_time ASC
""", (SYMBOL,))
closed = [dict(r) for r in cur.fetchall()]

print(f"\n[1] 平仓交易: {len(closed)} 笔")
if closed:
    tot_pnl = sum(t["pnl"] or 0 for t in closed)
    tot_fee = sum(t["fees"] or 0 for t in closed)
    wins = [t for t in closed if (t["pnl"] or 0) > 0]
    losses = [t for t in closed if (t["pnl"] or 0) <= 0]
    print(f"  净盈亏={tot_pnl:.4f} USDT  手续费={tot_fee:.4f} USDT  "
          f"胜率={len(wins)/len(closed)*100:.1f}% ({len(wins)}胜/{len(losses)}负)")
    print(f"  离场原因分布: {dict(Counter(t['exit_reason'] for t in closed))}")

    # 排除 ghost 污染后
    real = [t for t in closed if t["exit_reason"] not in ("ghost_close", "ghost_cleanup", None)]
    if real:
        r_pnl = sum(t["pnl"] or 0 for t in real)
        r_fee = sum(t["fees"] or 0 for t in real)
        r_win = sum(1 for t in real if (t["pnl"] or 0) > 0)
        print(f"  [排除ghost污染] 有效平仓 {len(real)} 笔: 净={r_pnl:.4f} 费={r_fee:.4f} "
              f"胜率={r_win/len(real)*100:.1f}%")

    # 按策略汇总
    by_strat = defaultdict(lambda: {"pnl": 0.0, "fee": 0.0, "n": 0, "win": 0})
    for t in closed:
        s = by_strat[t["strategy_name"]]
        s["pnl"] += t["pnl"] or 0
        s["fee"] += t["fees"] or 0
        s["n"] += 1
        s["win"] += 1 if (t["pnl"] or 0) > 0 else 0
    print("\n  按策略汇总:")
    for k, v in sorted(by_strat.items(), key=lambda x: -x[1]["pnl"]):
        print(f"    {k:18s} n={v['n']:3d} 净={v['pnl']:>8.4f} 费={v['fee']:>7.4f} 胜={v['win']/v['n']*100:4.1f}%")

    # 按方向汇总
    by_side = defaultdict(lambda: {"pnl": 0.0, "fee": 0.0, "n": 0, "win": 0})
    for t in closed:
        s = by_side[t["side"]]
        s["pnl"] += t["pnl"] or 0
        s["fee"] += t["fees"] or 0
        s["n"] += 1
        s["win"] += 1 if (t["pnl"] or 0) > 0 else 0
    print("\n  按方向汇总:")
    for k, v in sorted(by_side.items(), key=lambda x: -x[1]["pnl"]):
        print(f"    {k:6s} n={v['n']:3d} 净={v['pnl']:>8.4f} 费={v['fee']:>7.4f} 胜={v['win']/v['n']*100:4.1f}%")

    # 按天汇总
    by_day = defaultdict(lambda: {"pnl": 0.0, "fee": 0.0, "n": 0, "win": 0})
    for t in closed:
        day = (t["close_time"] or t["create_time"] or "")[:10]
        s = by_day[day]
        s["pnl"] += t["pnl"] or 0
        s["fee"] += t["fees"] or 0
        s["n"] += 1
        s["win"] += 1 if (t["pnl"] or 0) > 0 else 0
    print("\n  按天汇总:")
    for k in sorted(by_day):
        v = by_day[k]
        wr = f"{v['win']/v['n']*100:4.1f}%" if v['n'] else "  - "
        print(f"    {k}  n={v['n']:3d} 净={v['pnl']:>8.4f} 费={v['fee']:>7.4f} 胜={wr}")

    # 明细（最近 40 笔）
    print("\n  平仓明细（最近 40 笔）:")
    for t in closed[-40:]:
        ct = (t["close_time"] or "")[:19]
        print(f"    {ct} {t['strategy_name']:14s} {t['side']:5s} "
              f"qty={(t['quantity'] or 0):.4f} px={t['filled_price']} pnl={(t['pnl'] or 0):>8.4f} "
              f"fee={(t['fees'] or 0):.4f} {t['exit_reason']}")

# ── 2. 当前持仓 ──
cur.execute("""
    SELECT symbol, strategy_name, side, signal_type, quantity, filled_price,
           leverage, margin, pnl, fees, status, create_time
    FROM trade_records
    WHERE symbol=? AND status='open'
    ORDER BY create_time ASC
""", (SYMBOL,))
open_pos = [dict(r) for r in cur.fetchall()]
print(f"\n[2] 当前持仓: {len(open_pos)} 笔")
for t in open_pos:
    ct = (t["create_time"] or "")[:19]
    print(f"    {ct} {t['strategy_name']:14s} {t['side']:5s} "
          f"qty={t['quantity']:.4f} px={t['filled_price']} margin={t['margin']:.2f}")

conn.close()
print("\n[DONE] XRP-USDT-SWAP 交易分析完成")
