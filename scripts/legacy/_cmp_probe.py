import sqlite3, json
from datetime import datetime

DB = "data/trading.db"
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

def q(sql, params=()):
    return cur.execute(sql, params).fetchall()

print("=== trade_records status counts ===")
for r in q("SELECT status, COUNT(*) c FROM trade_records GROUP BY status"):
    print(dict(r))

print("\n=== time ranges ===")
r = q("SELECT MIN(create_time) mn_c, MAX(create_time) mx_c, MIN(close_time) mn_x, MAX(close_time) mx_x FROM trade_records")[0]
print(dict(r))

print("\n=== trades closed after 2026-08-19 09:15 ===")
r = q("SELECT COUNT(*) c FROM trade_records WHERE status='closed' AND close_time >= '2026-08-19 09:15:00'")[0]
print("closed after 09:15:", dict(r))
r = q("SELECT COUNT(*) c FROM trade_records WHERE create_time >= '2026-08-19 09:15:00'")[0]
print("created after 09:15:", dict(r))
r = q("SELECT COUNT(*) c FROM trade_records WHERE close_time >= '2026-08-19 09:15:00'")[0]
print("any close_time after 09:15:", dict(r))

print("\n=== trades closed after 08:45 (post-baseline) ===")
r = q("SELECT COUNT(*) c FROM trade_records WHERE status='closed' AND close_time >= '2026-08-19 08:45:00'")[0]
print(dict(r))

print("\n=== per-strategy (all closed) ===")
for r in q("SELECT strategy_name, COUNT(*) c, SUM(pnl) pnl, SUM(fees) fees FROM trade_records WHERE status='closed' GROUP BY strategy_name"):
    print(dict(r))

print("\n=== per-strategy trades created after 09:15 (any status) ===")
for r in q("SELECT strategy_name, status, COUNT(*) c FROM trade_records WHERE create_time >= '2026-08-19 09:15:00' GROUP BY strategy_name, status"):
    print(dict(r))

print("\n=== latest 5 closed trades ===")
for r in q("SELECT create_time, close_time, symbol, strategy_name, side, pnl, fees FROM trade_records WHERE status='closed' ORDER BY close_time DESC LIMIT 5"):
    print(dict(r))

print("\n=== account_history latest 5 ===")
for r in q("SELECT timestamp, total_equity, available_balance, used_margin FROM account_history ORDER BY timestamp DESC LIMIT 5"):
    print(dict(r))

print("\n=== account_history count and time range ===")
r = q("SELECT COUNT(*) c, MIN(timestamp) mn, MAX(timestamp) mx FROM account_history")[0]
print(dict(r))

print("\n=== account_history around baseline (before 08:45) latest ===")
r = q("SELECT timestamp, total_equity, used_margin FROM account_history WHERE timestamp <= '2026-08-19 08:45:00' ORDER BY timestamp DESC LIMIT 1")[0]
print(dict(r))
print("=== account_history after 09:15 latest ===")
r = q("SELECT timestamp, total_equity, used_margin FROM account_history WHERE timestamp >= '2026-08-19 09:15:00' ORDER BY timestamp DESC LIMIT 1")[0]
print(dict(r))

print("\n=== table list ===")
for r in q("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
    print(r["name"])

conn.close()
