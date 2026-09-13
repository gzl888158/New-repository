import sqlite3
from datetime import datetime

conn = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

def rows(sql, p=()):
    return [dict(r) for r in cur.execute(sql, p).fetchall()]

print("=== trades schema ===")
for r in cur.execute("PRAGMA table_info(trades)").fetchall():
    print(r["cid"], r["name"], r["type"])

print("\n=== trades count ===")
print(rows("SELECT COUNT(*) c FROM trades"))

print("\n=== trades time range ===")
print(rows("SELECT MIN(entry_time) mn_e, MAX(entry_time) mx_e, MIN(exit_time) mn_x, MAX(exit_time) mx_x FROM trades"))

print("\n=== trades per-strategy (win via win col, pnl_usdt) ===")
print(rows("SELECT strategy_name, COUNT(*) c, SUM(pnl) sum_pnl, SUM(pnl_usdt) sum_pnl_usdt, SUM(fees) sum_fees, SUM(win) wins FROM trades GROUP BY strategy_name"))

print("\n=== overall win/loss using win col ===")
print(rows("SELECT COUNT(*) c, SUM(win) wins, SUM(CASE WHEN win=0 THEN 1 ELSE 0 END) losses FROM trades"))

print("\n=== overall win/loss using pnl_usdt>0 ===")
print(rows("SELECT COUNT(*) c, SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) wins, SUM(CASE WHEN pnl_usdt<=0 THEN 1 ELSE 0 END) losses FROM trades"))

print("\n=== direction distribution (all) ===")
print(rows("SELECT direction, COUNT(*) c, SUM(pnl_usdt) pnl_usdt FROM trades GROUP BY direction"))

print("\n=== grid direction distribution ===")
print(rows("SELECT direction, COUNT(*) c, SUM(pnl_usdt) pnl_usdt FROM trades WHERE strategy_name='grid' GROUP BY direction"))

print("\n=== trades exited after 09:15 (post-opt) ===")
print(rows("SELECT COUNT(*) c FROM trades WHERE exit_time >= '2026-08-19 09:15:00'"))
print(rows("SELECT COUNT(*) c FROM trades WHERE entry_time >= '2026-08-19 09:15:00'"))

print("\n=== trades exited after 08:45 ===")
print(rows("SELECT COUNT(*) c FROM trades WHERE exit_time >= '2026-08-19 08:45:00'"))

print("\n=== post-opt (exit>=09:15) per-strategy ===")
print(rows("SELECT strategy_name, COUNT(*) c, SUM(pnl_usdt) pnl_usdt, SUM(fees) fees, SUM(win) wins, SUM(CASE WHEN win=0 THEN 1 ELSE 0 END) losses FROM trades WHERE exit_time >= '2026-08-19 09:15:00' GROUP BY strategy_name"))

print("\n=== post-opt (exit>=09:15) direction ===")
print(rows("SELECT direction, COUNT(*) c, SUM(pnl_usdt) pnl_usdt FROM trades WHERE exit_time >= '2026-08-19 09:15:00' GROUP BY direction"))

print("\n=== latest 10 trades by exit_time ===")
print(rows("SELECT trade_id, symbol, strategy_name, direction, entry_time, exit_time, pnl_usdt, fees, win FROM trades ORDER BY exit_time DESC LIMIT 10"))

print("\n=== grid wear-type: pnl_usdt in [-0.01, 0] (all time) ===")
print(rows("SELECT COUNT(*) c, SUM(CASE WHEN pnl_usdt >= -0.01 AND pnl_usdt <= 0 THEN 1 ELSE 0 END) wear FROM trades WHERE strategy_name='grid'"))

print("\n=== distinct entry_time format sample ===")
print(rows("SELECT entry_time, exit_time FROM trades LIMIT 2"))

conn.close()
