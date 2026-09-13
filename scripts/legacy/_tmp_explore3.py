# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

def q(sql, p=()):
    return [dict(r) for r in c.execute(sql, p).fetchall()]

CUT = "2026-08-19 09:15:00"

print("=== account_history trajectory (post-opt, sampled) ===")
rows = q("SELECT timestamp, total_equity, used_margin, unrealized_pnl FROM account_history WHERE timestamp >= ? ORDER BY timestamp ASC", (CUT,))
print("total account_history rows post-opt:", len(rows))
# sample every N
step = max(1, len(rows)//30)
for i, r in enumerate(rows):
    if i % step == 0 or i == len(rows)-1:
        print(f"  {r['timestamp']}  equity={r['total_equity']:.2f}  margin={r['used_margin']:.2f}  upnl={r['unrealized_pnl']}")

print("\n=== all post-opt trades sorted by pnl_usdt asc (worst first) ===")
for r in q("SELECT exit_time, strategy_name, symbol, direction, ROUND(pnl,6) AS pnl, ROUND(pnl_usdt,6) AS pnl_usdt, ROUND(fees,6) AS fees, leverage, quantity, win FROM trades WHERE exit_time >= ? ORDER BY pnl_usdt ASC LIMIT 30", (CUT,)):
    print(r)

print("\n=== scalping trades: pnl vs pnl_usdt all ===")
for r in q("SELECT exit_time, symbol, direction, pnl, pnl_usdt, leverage, quantity, margin, fees FROM trades WHERE strategy_name='scalping' ORDER BY exit_time DESC LIMIT 10"):
    print(r)

print("\n=== post-opt total pnl (sum pnl_usdt) by strategy ===")
for r in q("SELECT strategy_name, COUNT(*) n, ROUND(SUM(pnl_usdt),6) pnl_usdt, ROUND(SUM(fees),6) fees, ROUND(SUM(pnl),6) pnl FROM trades WHERE exit_time >= ? GROUP BY strategy_name", (CUT,)):
    print(r)

print("\n=== biggest pnl_usdt losers in whole trades table (any time, last 5 days) ===")
for r in q("SELECT exit_time, strategy_name, symbol, direction, ROUND(pnl_usdt,6) AS pnl_usdt, win FROM trades WHERE exit_time >= '2026-08-19 00:00:00' ORDER BY pnl_usdt ASC LIMIT 20"):
    print(r)
