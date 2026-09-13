# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

def q(sql, p=()):
    return [dict(r) for r in c.execute(sql, p).fetchall()]

total = q("SELECT COUNT(*) AS n FROM trades")[0]["n"]
print("trades total:", total)

rng = q("SELECT MIN(exit_time) AS mn, MAX(exit_time) AS mx FROM trades")[0]
print("exit_time range:", rng["mn"], "->", rng["mx"])

cut = "2026-08-19 09:15:00"
post = q("SELECT COUNT(*) AS n FROM trades WHERE exit_time >= ?", (cut,))[0]["n"]
print("post-opt (exit_time >= 09:15):", post)

print("\npost-opt by strategy:")
for r in q("SELECT strategy_name, direction, COUNT(*) AS n, SUM(win) AS wins, ROUND(SUM(pnl_usdt),6) AS pnl, ROUND(SUM(fees),6) AS fees FROM trades WHERE exit_time >= ? GROUP BY strategy_name, direction ORDER BY strategy_name, direction", (cut,)):
    print("  ", r)

print("\npost-opt by direction:")
for r in q("SELECT direction, COUNT(*) AS n, SUM(win) AS wins, ROUND(SUM(pnl_usdt),6) AS pnl FROM trades WHERE exit_time >= ? GROUP BY direction", (cut,)):
    print("  ", r)

print("\ngrid shorts post-opt (recent 20):")
for r in q("SELECT exit_time, symbol, direction, ROUND(pnl_usdt,6) AS pnl FROM trades WHERE exit_time >= ? AND strategy_name='grid' AND direction='short' ORDER BY exit_time DESC LIMIT 20", (cut,)):
    print("  ", r)

print("\nlatest 10 trades:")
for r in q("SELECT exit_time, strategy_name, direction, ROUND(pnl_usdt,6) AS pnl, win FROM trades ORDER BY exit_time DESC LIMIT 10"):
    print("  ", r)
