import sqlite3, os
from datetime import datetime, timedelta
from collections import defaultdict

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 观察 8月11日 ~ 8月17日 每天的权益细节（含出入金判断）
rows = c.execute(
    "SELECT timestamp, total_equity, available_balance FROM account_history "
    "WHERE timestamp >= '2026-08-10' ORDER BY timestamp ASC"
).fetchall()

daily = defaultdict(list)
for r in rows:
    day = str(r["timestamp"])[:10]
    daily[day].append((r["timestamp"], float(r["total_equity"]), float(r["available_balance"])))

for day in sorted(daily.keys()):
    lst = daily[day]
    first = lst[0]
    last = lst[-1]
    # 当日 max/min
    eqs = [x[1] for x in lst]
    mn = min(eqs); mx = max(eqs)
    print(f"{day}: n={len(lst):3d} 首={first[1]:8.2f} 末={last[1]:8.2f} 日低={mn:8.2f} 日高={mx:8.2f} 振幅={mx-mn:8.2f}")

c.close()
