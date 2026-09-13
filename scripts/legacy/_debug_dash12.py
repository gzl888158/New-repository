import sqlite3, os
p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 8-4 全天记录总数、权益 min/max 及时间
print("=== 8-4 全天统计 ===")
for r in c.execute(
    "SELECT COUNT(*) n, MIN(total_equity) mn, MAX(total_equity) mx FROM account_history "
    "WHERE timestamp >= '2026-08-04 00:00' AND timestamp <= '2026-08-04 23:59'"
):
    print(dict(r))

print("\n=== 8-4 权益最小值记录 ===")
for r in c.execute(
    "SELECT timestamp, total_equity, available_balance FROM account_history "
    "WHERE timestamp >= '2026-08-04 00:00' AND timestamp <= '2026-08-04 23:59' "
    "AND total_equity < 8.0 ORDER BY total_equity ASC LIMIT 20"
):
    print(f"{r['timestamp']}  eq={r['total_equity']:.4f}  avail={r['available_balance']:.4f}")

print("\n=== 8-4 15:01 前后 3 分钟所有记录 ===")
for r in c.execute(
    "SELECT timestamp, total_equity, available_balance FROM account_history "
    "WHERE timestamp >= '2026-08-04 15:00' AND timestamp <= '2026-08-04 15:03' "
    "ORDER BY timestamp ASC LIMIT 40"
):
    print(f"{r['timestamp']}  eq={r['total_equity']:.4f}  avail={r['available_balance']:.4f}")

c.close()
