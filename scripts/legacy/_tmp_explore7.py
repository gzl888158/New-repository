# -*- coding: utf-8 -*-
import sqlite3

c = sqlite3.connect("file:data/trading.db?mode=ro", uri=True)
c.row_factory = sqlite3.Row

def q(sql, p=()):
    return [dict(r) for r in c.execute(sql, p).fetchall()]

print("=== account_history 08-22 12:30 - 14:00 (equity jump check) ===")
rows = q("SELECT timestamp, total_equity, used_margin, available_balance, unrealized_pnl FROM account_history WHERE timestamp >= '2026-08-22 12:30:00' AND timestamp <= '2026-08-22 14:00:00' ORDER BY timestamp ASC")
print("rows:", len(rows))
step = max(1, len(rows)//40)
for i, r in enumerate(rows):
    if i % step == 0 or i == len(rows)-1:
        print(f"  {r['timestamp']}  eq={r['total_equity']:.2f}  margin={r['used_margin']:.2f}  avail={r['available_balance']}  upnl={r['unrealized_pnl']}")

print("\n=== account_history 08-22 15:00 - 16:10 (collapse) ===")
rows2 = q("SELECT timestamp, total_equity, used_margin, available_balance, unrealized_pnl FROM account_history WHERE timestamp >= '2026-08-22 15:00:00' AND timestamp <= '2026-08-22 16:10:00' ORDER BY timestamp ASC")
print("rows:", len(rows2))
step = max(1, len(rows2)//40)
for i, r in enumerate(rows2):
    if i % step == 0 or i == len(rows2)-1:
        print(f"  {r['timestamp']}  eq={r['total_equity']:.2f}  margin={r['used_margin']:.2f}  avail={r['available_balance']}  upnl={r['unrealized_pnl']}")
