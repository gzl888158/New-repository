import sqlite3, os

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 查 8月4日-8月6日 分钟级权益跳变（定位入金/出金时刻）
rows = c.execute(
    "SELECT timestamp, total_equity, available_balance, unrealized_pnl "
    "FROM account_history WHERE timestamp >= '2026-08-04 20:00:00' "
    "AND timestamp <= '2026-08-05 08:00:00' ORDER BY timestamp ASC"
).fetchall()

print("=== 8月4日夜 ~ 8月5日晨 权益跳变（>5 USDT）===")
prev = None
for r in rows:
    ts = r["timestamp"]
    eq = float(r["total_equity"]); avail = float(r["available_balance"]); upl = float(r["unrealized_pnl"])
    if prev is None:
        prev = (ts, eq, avail, upl); continue
    deq = eq - prev[1]
    if abs(deq) > 3.0:
        print(f"{ts}: eq {prev[1]:.2f}→{eq:.2f} (Δ{deq:+.2f}) avail {prev[2]:.2f}→{avail:.2f} upl {prev[3]:.2f}→{upl:.2f}")
    prev = (ts, eq, avail, upl)

c.close()
