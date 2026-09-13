import sqlite3, os

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 定位所有大额入金/出金时刻（equity 单步 >3 USDT），全表按真实时间排序
rows = c.execute(
    "SELECT timestamp, total_equity, available_balance, unrealized_pnl "
    "FROM account_history ORDER BY timestamp ASC"
).fetchall()

print("=== 全表 equity 单步跳变 >3 USDT 的事件 ===")
prev = None
for r in rows:
    ts = r["timestamp"]
    eq = float(r["total_equity"]); avail = float(r["available_balance"]); upl = float(r["unrealized_pnl"])
    if prev is None:
        prev = (ts, eq, avail, upl); continue
    deq = eq - prev[1]
    davail = avail - prev[2]
    # 入金特征：equity 跳增 且 avail 跳增 且 upl 基本不变（小幅）
    if abs(deq) > 3.0:
        kind = "入金?" if deq > 0 else "出金?"
        print(f"{ts}: eq {prev[1]:.2f}→{eq:.2f} (Δ{deq:+.2f}) avail {prev[2]:.2f}→{avail:.2f} (Δ{davail:+.2f}) upl {prev[3]:.2f}→{upl:.2f}  [{kind}]")
    prev = (ts, eq, avail, upl)

c.close()
