import sqlite3, os
p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 8-12 日初/日末权益 + 23:13 前后
print("=== 8-12 日初首条 / 日末末条 ===")
for r in c.execute(
    "SELECT timestamp, total_equity, available_balance FROM account_history "
    "WHERE timestamp >= '2026-08-12 00:00' AND timestamp <= '2026-08-12 23:59' "
    "ORDER BY timestamp ASC LIMIT 1"
):
    print("首条:", dict(r))
for r in c.execute(
    "SELECT timestamp, total_equity, available_balance FROM account_history "
    "WHERE timestamp >= '2026-08-12 00:00' AND timestamp <= '2026-08-12 23:59' "
    "ORDER BY timestamp DESC LIMIT 1"
):
    print("末条:", dict(r))

# 8-12 23:10 ~ 23:16 入金细节
print("\n=== 8-12 23:10 ~ 23:16 ===")
for r in c.execute(
    "SELECT timestamp, total_equity, available_balance, unrealized_pnl FROM account_history "
    "WHERE timestamp >= '2026-08-12 23:10' AND timestamp <= '2026-08-12 23:16' "
    "ORDER BY timestamp ASC"
):
    print(f"{r['timestamp']}  eq={r['total_equity']:.4f}  avail={r['available_balance']:.4f}  upl={r['unrealized_pnl']:.4f}")

# 8-11 日末权益（作为 8-12 的基准）
print("\n=== 8-11 日末末条 ===")
for r in c.execute(
    "SELECT timestamp, total_equity FROM account_history "
    "WHERE timestamp >= '2026-08-11 00:00' AND timestamp <= '2026-08-11 23:59' "
    "ORDER BY timestamp DESC LIMIT 1"
):
    print(dict(r))

c.close()
