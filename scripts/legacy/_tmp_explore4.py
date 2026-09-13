# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

def q(sql, p=()):
    return [dict(r) for r in c.execute(sql, p).fetchall()]

print("=== position_history 08-22 (around collapse) ===")
for r in q("SELECT timestamp, symbol, side, quantity, avg_cost, mark_price, unrealized_pnl, margin, leverage FROM position_history WHERE timestamp >= '2026-08-22 11:00:00' ORDER BY timestamp ASC LIMIT 40"):
    print(r)

print("\n=== risk_events 08-22 ===")
for r in q("SELECT timestamp, event_type, severity, symbol, message FROM risk_events WHERE timestamp >= '2026-08-22 00:00:00' ORDER BY timestamp ASC LIMIT 60"):
    print(r)

print("\n=== trade_records 08-22 (status/pnl) ===")
for r in q("SELECT create_time, close_time, symbol, strategy_name, side, status, ROUND(pnl,6) pnl, ROUND(fees,6) fees, exit_reason FROM trade_records WHERE create_time >= '2026-08-22 00:00:00' OR close_time >= '2026-08-22 00:00:00' ORDER BY COALESCE(close_time, create_time) ASC LIMIT 60"):
    print(r)

print("\n=== trades table 08-22 all (entry_time/exit_time) ===")
for r in q("SELECT entry_time, exit_time, strategy_name, symbol, direction, quantity, leverage, ROUND(pnl_usdt,6) pnl_usdt, win FROM trades WHERE entry_time >= '2026-08-22 00:00:00' OR exit_time >= '2026-08-22 00:00:00' ORDER BY exit_time ASC LIMIT 60"):
    print(r)

print("\n=== equity_curve 08-22 tail ===")
for r in q("SELECT timestamp, total_equity, used_margin, unrealized_pnl, realized_pnl FROM equity_curve WHERE timestamp >= '2026-08-22 10:00:00' ORDER BY timestamp ASC LIMIT 60"):
    print(r)
