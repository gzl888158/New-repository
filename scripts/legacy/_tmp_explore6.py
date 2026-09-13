# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

def q(sql, p=()):
    return [dict(r) for r in c.execute(sql, p).fetchall()]

print("=== trades table TRUMP (any) ===")
for r in q("SELECT entry_time, exit_time, strategy_name, direction, quantity, leverage, ROUND(pnl_usdt,6) pnl_usdt, win, exit_reason FROM trades WHERE symbol LIKE '%TRUMP%' ORDER BY exit_time DESC LIMIT 20"):
    print(r)

print("\n=== trade_records TRUMP ===")
for r in q("SELECT create_time, close_time, symbol, strategy_name, side, status, ROUND(pnl,6) pnl, ROUND(margin,4) margin, leverage, quantity, exit_reason FROM trade_records WHERE symbol LIKE '%TRUMP%' ORDER BY COALESCE(close_time, create_time) DESC LIMIT 20"):
    print(r)

print("\n=== risk_events TRUMP or liquidation (last 3 days) ===")
for r in q("SELECT timestamp, event_type, severity, symbol, message FROM risk_events WHERE timestamp >= '2026-08-20 00:00:00' ORDER BY timestamp DESC LIMIT 40"):
    print(r)

print("\n=== trades table strategy_name distinct values ===")
for r in q("SELECT DISTINCT strategy_name FROM trades ORDER BY strategy_name"):
    print(r)

print("\n=== trades table any position with TRUMP symbol count ===")
for r in q("SELECT COUNT(*) n FROM trades WHERE symbol LIKE '%TRUMP%'"):
    print(r)
