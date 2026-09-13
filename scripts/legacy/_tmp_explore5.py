# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

def q(sql, p=()):
    return [dict(r) for r in c.execute(sql, p).fetchall()]

print("=== position_history 08-22 13:00 - 16:10 (large positions) ===")
for r in q("SELECT timestamp, symbol, side, quantity, avg_cost, mark_price, ROUND(unrealized_pnl,4) upnl, ROUND(margin,2) margin, leverage FROM position_history WHERE timestamp >= '2026-08-22 13:00:00' AND timestamp <= '2026-08-22 16:10:00' ORDER BY timestamp ASC LIMIT 200"):
    print(r)

print("\n=== distinct symbols in position_history 08-22 13:00-16:10 ===")
for r in q("SELECT DISTINCT symbol, side FROM position_history WHERE timestamp >= '2026-08-22 13:00:00' AND timestamp <= '2026-08-22 16:10:00'"):
    print(r)

print("\n=== trade_records SOL/USD 08-22 (all) ===")
for r in q("SELECT create_time, close_time, symbol, strategy_name, side, status, ROUND(pnl,6) pnl, ROUND(margin,4) margin, leverage, quantity, exit_reason FROM trade_records WHERE symbol LIKE '%SOL%' AND (create_time >= '2026-08-22 00:00:00' OR close_time >= '2026-08-22 00:00:00') ORDER BY create_time ASC LIMIT 50"):
    print(r)
