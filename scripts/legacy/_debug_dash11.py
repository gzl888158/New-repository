import sqlite3, os
p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 查 8-4 14:50 ~ 15:10 的所有记录
print("=== 8-4 14:50 ~ 15:10 权益记录 ===")
for r in c.execute(
    "SELECT timestamp, total_equity, available_balance, unrealized_pnl "
    "FROM account_history WHERE timestamp >= '2026-08-04 14:50' "
    "AND timestamp <= '2026-08-04 15:10' ORDER BY timestamp ASC"
):
    print(f"{r['timestamp']}  eq={r['total_equity']:.4f}  avail={r['available_balance']:.4f}  upl={r['unrealized_pnl']:.4f}")

# 查 8-4 全天权益的 min/max 及时间
print("\n=== 8-4 全天权益 min/max ===")
for r in c.execute(
    "SELECT MIN(total_equity) as mn, MAX(total_equity) as mx FROM account_history "
    "WHERE timestamp >= '2026-08-04 00:00' AND timestamp <= '2026-08-04 23:59'"
):
    print(dict(r))

# 峰值 8.04 出现在哪个具体时间
print("\n=== 8-4 权益 >= 8.0 的记录 ===")
for r in c.execute(
    "SELECT timestamp, total_equity FROM account_history "
    "WHERE timestamp >= '2026-08-04 00:00' AND timestamp <= '2026-08-04 23:59' "
    "AND total_equity >= 7.8 ORDER BY timestamp ASC"
):
    print(f"{r['timestamp']}  eq={r['total_equity']:.4f}")

c.close()
