# -*- coding: utf-8 -*-
"""临时分析：最近6小时已平仓交易（权威字段 trades.pnl_usdt）"""
import sqlite3, os, sys

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db")
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

SIX_H = "datetime('now', 'localtime', '-6 hours')"

# 最近6小时平仓交易
rows = cur.execute(f"""
    SELECT * FROM trades
    WHERE replace(exit_time, 'T', ' ') >= datetime('now', 'localtime', '-6 hours')
    ORDER BY exit_time
""").fetchall()

print(f"=== 最近6小时平仓交易: {len(rows)} 笔 ===")
if not rows:
    print("(无交易)")
    sys.exit(0)

tot_pnl = sum(float(r['pnl_usdt'] or 0) for r in rows)
wins = sum(1 for r in rows if float(r['pnl_usdt'] or 0) > 0)
losses = sum(1 for r in rows if float(r['pnl_usdt'] or 0) < 0)
flat = sum(1 for r in rows if float(r['pnl_usdt'] or 0) == 0)
tot_fee = sum(float(r['fees'] or 0) for r in rows)
print(f"总PnL={tot_pnl:.4f} USDT | 胜{wins}/负{losses}/平{flat} | 胜率={wins/len(rows)*100:.1f}% | 总手续费={tot_fee:.4f}")

print("\n--- 按策略 ---")
by_strat = {}
for r in rows:
    k = r['strategy_name']
    by_strat.setdefault(k, []).append(r)
for k, rs in sorted(by_strat.items(), key=lambda x: sum(float(r['pnl_usdt'] or 0) for r in x[1])):
    pnl = sum(float(r['pnl_usdt'] or 0) for r in rs)
    w = sum(1 for r in rs if float(r['pnl_usdt'] or 0) > 0)
    f = sum(float(r['fees'] or 0) for r in rs)
    print(f"  {k:22s} n={len(rs):3d} PnL={pnl:8.4f} 胜率={w/len(rs)*100:5.1f}% 手续费={f:.4f}")

print("\n--- 按币种 ---")
by_sym = {}
for r in rows:
    k = r['symbol']
    by_sym.setdefault(k, []).append(r)
for k, rs in sorted(by_sym.items(), key=lambda x: sum(float(r['pnl_usdt'] or 0) for r in x[1])):
    pnl = sum(float(r['pnl_usdt'] or 0) for r in rs)
    w = sum(1 for r in rs if float(r['pnl_usdt'] or 0) > 0)
    f = sum(float(r['fees'] or 0) for r in rs)
    print(f"  {k:20s} n={len(rs):3d} PnL={pnl:8.4f} 胜率={w/len(rs)*100:5.1f}% 手续费={f:.4f}")

print("\n--- 按方向 ---")
by_dir = {}
for r in rows:
    k = r['direction']
    by_dir.setdefault(k, []).append(r)
for k, rs in sorted(by_dir.items(), key=lambda x: sum(float(r['pnl_usdt'] or 0) for r in x[1])):
    pnl = sum(float(r['pnl_usdt'] or 0) for r in rs)
    w = sum(1 for r in rs if float(r['pnl_usdt'] or 0) > 0)
    print(f"  {k:10s} n={len(rs):3d} PnL={pnl:8.4f} 胜率={w/len(rs)*100:5.1f}%")

print("\n--- 按平仓小时(本地) ---")
by_hour = {}
for r in rows:
    h = r['exit_time'].replace('T', ' ')[11:13]
    by_hour.setdefault(h, []).append(r)
for k in sorted(by_hour):
    rs = by_hour[k]
    pnl = sum(float(r['pnl_usdt'] or 0) for r in rs)
    print(f"  {k}时 n={len(rs):3d} PnL={pnl:8.4f}")

print("\n--- 磨损型交易检测 (|PnL|<=0.01 且 手续费>0) ---")
wear = [r for r in rows if abs(float(r['pnl_usdt'] or 0)) <= 0.01 and float(r['fees'] or 0) > 0]
print(f"  磨损型笔数: {len(wear)} / {len(rows)} ({len(wear)/len(rows)*100:.1f}%)")
for r in wear[:20]:
    print(f"    {r['exit_time']} {r['symbol']:16s} {r['strategy_name']:16s} {r['direction']:5s} pnl={float(r['pnl_usdt'] or 0):+.4f} fee={float(r['fees'] or 0):.4f}")

print("\n--- 亏损明细 (PnL<0) ---")
for r in sorted([x for x in rows if float(x['pnl_usdt'] or 0) < 0], key=lambda x: float(x['pnl_usdt'] or 0)):
    print(f"    {r['exit_time']} {r['symbol']:16s} {r['strategy_name']:16s} {r['direction']:5s} pnl={float(r['pnl_usdt'] or 0):+.4f} fee={float(r['fees'] or 0):.4f} exit_reason={r['exit_reason']}")

print("\n--- 盈利明细 (PnL>0) ---")
for r in sorted([x for x in rows if float(x['pnl_usdt'] or 0) > 0], key=lambda x: -float(x['pnl_usdt'] or 0)):
    print(f"    {r['exit_time']} {r['symbol']:16s} {r['strategy_name']:16s} {r['direction']:5s} pnl={float(r['pnl_usdt'] or 0):+.4f} fee={float(r['fees'] or 0):.4f} exit_reason={r['exit_reason']}")

conn.close()
