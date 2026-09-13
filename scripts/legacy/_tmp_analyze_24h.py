# -*- coding: utf-8 -*-
"""临时分析：最近24小时已平仓交易 + 各策略历史胜率 + 黑名单/暂停状态"""
import sqlite3, os, sys, json

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db")
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== 最近24小时平仓交易 ===")
rows = cur.execute("""
    SELECT * FROM trades
    WHERE replace(exit_time, 'T', ' ') >= datetime('now', 'localtime', '-24 hours')
    ORDER BY exit_time
""").fetchall()
print(f"笔数: {len(rows)}")
if rows:
    tot = sum(float(r['pnl_usdt'] or 0) for r in rows)
    w = sum(1 for r in rows if float(r['pnl_usdt'] or 0) > 0)
    l = sum(1 for r in rows if float(r['pnl_usdt'] or 0) < 0)
    f = sum(float(r['fees'] or 0) for r in rows)
    print(f"总PnL={tot:.4f} 胜率={w/len(rows)*100:.1f}% (胜{w}/负{l}) 手续费={f:.4f}")
    by_strat = {}
    for r in rows:
        by_strat.setdefault(r['strategy_name'], []).append(r)
    for k, rs in sorted(by_strat.items(), key=lambda x: sum(float(r['pnl_usdt'] or 0) for r in x[1])):
        p = sum(float(r['pnl_usdt'] or 0) for r in rs)
        ww = sum(1 for r in rs if float(r['pnl_usdt'] or 0) > 0)
        print(f"  {k:20s} n={len(rs):3d} PnL={p:8.4f} 胜率={ww/len(rs)*100:5.1f}%")

print("\n=== 各策略近7天累计表现 (trades.pnl_usdt) ===")
strat_rows = cur.execute("""
    SELECT strategy_name,
           COUNT(*) as cnt,
           SUM(pnl_usdt) as total_pnl,
           SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins
    FROM trades
    WHERE replace(exit_time, 'T', ' ') >= datetime('now', 'localtime', '-7 days')
    GROUP BY strategy_name
    ORDER BY total_pnl ASC
""").fetchall()
for r in strat_rows:
    cnt = r['cnt'] or 0
    wr = (r['wins'] / cnt * 100) if cnt else 0
    print(f"  {r['strategy_name']:20s} n={cnt:4d} PnL={float(r['total_pnl'] or 0):9.4f} 胜率={wr:5.1f}%")

print("\n=== 近7天币种累计表现 ===")
sym_rows = cur.execute("""
    SELECT symbol,
           COUNT(*) as cnt,
           SUM(pnl_usdt) as total_pnl,
           SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) as wins
    FROM trades
    WHERE replace(exit_time, 'T', ' ') >= datetime('now', 'localtime', '-7 days')
    GROUP BY symbol
    ORDER BY total_pnl ASC
""").fetchall()
for r in sym_rows:
    cnt = r['cnt'] or 0
    wr = (r['wins'] / cnt * 100) if cnt else 0
    print(f"  {r['symbol']:20s} n={cnt:4d} PnL={float(r['total_pnl'] or 0):9.4f} 胜率={wr:5.1f}%")

print("\n=== 24小时磨损型交易 (|PnL|<=0.01 且 手续费>0) ===")
if rows:
    wear = [r for r in rows if abs(float(r['pnl_usdt'] or 0)) <= 0.01 and float(r['fees'] or 0) > 0]
    print(f"  磨损笔数: {len(wear)} / {len(rows)}")
    for r in wear:
        print(f"    {r['exit_time']} {r['symbol']} {r['strategy_name']} {r['direction']} pnl={float(r['pnl_usdt'] or 0):+.4f} fee={float(r['fees'] or 0):.4f}")

conn.close()

# 黑名单/策略暂停状态
print("\n=== 黑名单/暂停状态 ===")
state_files = [
    ("data/blacklist.json", None),
    ("data/strategy_state.json", None),
]
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for rel, _ in state_files:
    p = os.path.join(base, rel)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            s = json.dumps(d, ensure_ascii=False)
            print(f"  [{rel}]: {s[:1500]}")
        except Exception as e:
            print(f"  [{rel}]: 读取失败 {e}")
    else:
        print(f"  [{rel}]: (不存在)")
