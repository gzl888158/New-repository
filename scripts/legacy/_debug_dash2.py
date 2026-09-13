import sqlite3, os
from datetime import datetime, timedelta

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 复现 get_historical_performance 的核心逻辑
days = 30
cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

rows = c.execute(
    "SELECT timestamp, total_equity FROM account_history WHERE timestamp >= ? ORDER BY timestamp ASC",
    (cutoff,)
).fetchall()

print(f"rows in {days}d window: {len(rows)}")

# 按日聚合：后写入覆盖 = 当日最后一条
daily_equity = {}
for row in rows:
    ts = row["timestamp"]
    val = row["total_equity"]
    val = float(val) if val else 0
    if val <= 0:
        continue
    day = str(ts)[:10]
    daily_equity[day] = val

days_sorted = sorted(daily_equity.keys())
equities = [daily_equity[k] for k in days_sorted]

print(f"\n=== 每日权益（按日聚合）===")
for d, e in zip(days_sorted, equities):
    print(f"{d}: {e:.4f}")

# 日收益率
daily_returns = []
for i in range(1, len(equities)):
    if equities[i-1] > 0:
        daily_returns.append((equities[i] - equities[i-1]) / equities[i-1])

print(f"\n=== 日收益率 ===")
for i, (d, r) in enumerate(zip(days_sorted[1:], daily_returns)):
    flag = " <-- >30%跳变" if abs(r) > 0.30 else ""
    print(f"{d}: {r*100:.2f}%{flag}")

trading_returns = [r for r in daily_returns if abs(r) <= 0.30]
print(f"\n=== 统计 ===")
print(f"daily_returns: {len(daily_returns)} 个")
print(f"trading_returns(剔除>30%): {len(trading_returns)} 个")
if trading_returns:
    print(f"best_day = {max(trading_returns)*100:.2f}%")
    print(f"worst_day = {min(trading_returns)*100:.2f}%")

c.close()
