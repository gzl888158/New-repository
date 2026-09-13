import sqlite3, os
from datetime import datetime

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 全表扫描，检测 available_balance 单步跳变 > 5 USDT 的事件（出入金特征）
# 注意：equity 跳变 + avail 跳变 + upl 归零 = 出入金
rows = c.execute(
    "SELECT timestamp, total_equity, available_balance, unrealized_pnl "
    "FROM account_history ORDER BY timestamp ASC"
).fetchall()

print("=== 全表出入金事件检测（avail 单步跳变 > 5 USDT 或 equity 跳变 > 5 USDT）===")
prev = None
for r in rows:
    ts = r["timestamp"]
    eq = float(r["total_equity"])
    avail = float(r["available_balance"])
    upl = float(r["unrealized_pnl"])
    if prev is None:
        prev = (ts, eq, avail, upl)
        continue
    davail = avail - prev[2]
    deq = eq - prev[1]
    if abs(davail) > 5.0 or abs(deq) > 5.0:
        print(f"{ts}: eq {prev[1]:.2f}→{eq:.2f} (Δ{deq:+.2f})  avail {prev[2]:.2f}→{avail:.2f} (Δ{davail:+.2f})  upl {prev[3]:.2f}→{upl:.2f}")
    prev = (ts, eq, avail, upl)

c.close()
