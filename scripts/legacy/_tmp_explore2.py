# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

def q(sql, p=()):
    return [dict(r) for r in c.execute(sql, p).fetchall()]

CUT = "2026-08-19 09:15:00"
BASE = "2026-08-19 08:45:00"

print("=== account_history baseline (<= 08:45) ===")
for r in q("SELECT total_equity, used_margin, timestamp FROM account_history WHERE timestamp <= ? ORDER BY timestamp DESC LIMIT 1", (BASE,)):
    print(r)

print("\n=== account_history post-opt earliest (>= 09:15) ===")
for r in q("SELECT total_equity, used_margin, timestamp FROM account_history WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1", (CUT,)):
    print(r)

print("\n=== account_history post-opt latest (>= 09:15) ===")
for r in q("SELECT total_equity, used_margin, timestamp FROM account_history WHERE timestamp >= ? ORDER BY timestamp DESC LIMIT 1", (CUT,)):
    print(r)

print("\n=== grid long latest exit_time ===")
for r in q("SELECT MAX(exit_time) AS mx FROM trades WHERE strategy_name='grid' AND direction='long'"):
    print(r)

print("\n=== grid short latest exit_time ===")
for r in q("SELECT MAX(exit_time) AS mx FROM trades WHERE strategy_name='grid' AND direction='short'"):
    print(r)

print("\n=== post-opt grid trades full ===")
for r in q("SELECT exit_time, symbol, direction, ROUND(pnl_usdt,6) AS pnl, win FROM trades WHERE exit_time >= ? AND strategy_name='grid' ORDER BY exit_time", (CUT,)):
    print(r)

print("\n=== post-opt scalping trades full ===")
for r in q("SELECT exit_time, symbol, direction, ROUND(pnl_usdt,6) AS pnl, win, exit_reason FROM trades WHERE exit_time >= ? AND strategy_name='scalping' ORDER BY exit_time", (CUT,)):
    print(r)

print("\n=== wear-type trades post-opt by strategy (pnl_usdt between -0.01 and 0) ===")
for r in q("SELECT strategy_name, COUNT(*) AS wear FROM trades WHERE exit_time >= ? AND pnl_usdt >= -0.01 AND pnl_usdt <= 0 GROUP BY strategy_name", (CUT,)):
    print(r)
