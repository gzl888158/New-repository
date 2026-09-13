import sqlite3, os
from datetime import datetime

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 追踪 available_balance 跳变（出入金特征：avail 突变 + equity 突变 + upl 不变）
rows = c.execute(
    "SELECT timestamp, total_equity, available_balance, used_margin, unrealized_pnl "
    "FROM account_history WHERE timestamp >= '2026-08-11 00:00:00' "
    "AND timestamp < '2026-08-13 00:00:00' ORDER BY timestamp ASC"
).fetchall()

print("=== 8月12日 权益/可用资金 跳变追踪（间隔采样）===")
prev = None
for r in rows:
    ts = r["timestamp"]
    eq = float(r["total_equity"])
    avail = float(r["available_balance"])
    upl = float(r["unrealized_pnl"])
    if prev is None:
        prev = (ts, eq, avail, upl)
        print(f"{ts}: eq={eq:.2f} avail={avail:.2f} upl={upl:.3f}  (起点)")
        continue
    deq = eq - prev[1]
    davail = avail - prev[2]
    # 只打印变化较大的点
    if abs(davail) > 1.0 or abs(deq) > 1.0:
        print(f"{ts}: eq={eq:.2f} (Δ{deq:+.2f}) avail={avail:.2f} (Δ{davail:+.2f}) upl={upl:.3f}")
    prev = (ts, eq, avail, upl)

c.close()
