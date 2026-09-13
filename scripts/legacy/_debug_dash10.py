import sqlite3, os
from datetime import datetime, timedelta

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

days = 30
cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
print("cutoff =", cutoff)

rows = c.execute(
    "SELECT timestamp, total_equity FROM account_history "
    "WHERE timestamp >= ? ORDER BY timestamp ASC",
    (cutoff,)
).fetchall()

print("总记录数 =", len(rows))

# 复现 _capital_adjusted_max_drawdown，同时记录每次更新
equities = []
ts_list = []
for r in rows:
    eq = float(r["total_equity"] or 0)
    if eq > 0:
        equities.append(eq)
        ts_list.append(r["timestamp"])

peak = equities[0]
max_dd = 0.0
max_dd_info = None
reset_events = []
for i in range(1, len(equities)):
    prev = equities[i-1]
    curr = equities[i]
    if prev > 0 and abs((curr - prev) / prev) > 0.30:
        # 出入金事件
        reset_events.append((ts_list[i], prev, curr, (curr-prev)/prev))
        peak = curr
    else:
        peak = max(peak, curr)
    if peak > 0:
        dd = (peak - curr) / peak
        if dd > max_dd:
            max_dd = dd
            max_dd_info = (ts_list[i], peak, curr, dd)

print("\nmax_drawdown =", max_dd)
print("峰值/谷值信息:", max_dd_info)

print("\n=== 检测到的出入金重置事件 (>30% 单步跳变) ===")
for ev in reset_events:
    print(f"ts={ev[0]}  prev={ev[1]:.2f}  curr={ev[2]:.2f}  跳变={ev[3]*100:.1f}%")

# 找出峰值出现在哪个时间点（全局绝对峰值）
global_peak_idx = max(range(len(equities)), key=lambda i: equities[i])
print("\n全局绝对峰值:", equities[global_peak_idx], "于", ts_list[global_peak_idx])

c.close()
