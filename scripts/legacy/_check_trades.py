"""检查近2小时交易"""
import sqlite3
from datetime import datetime, timedelta

db = sqlite3.connect('data/trading.db')
cur = db.cursor()

cutoff = (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
cur.execute("SELECT symbol, strategy_name, direction, pnl_usdt, exit_time FROM trades WHERE exit_time > ? ORDER BY exit_time DESC", (cutoff,))
rows = cur.fetchall()
print(f"近2小时交易: {len(rows)}笔")
for r in rows[:20]:
    print(f"  {r[0]:20s} | {r[1]:10s} | {r[2]:5s} | PnL={r[3]:+.6f} | {r[4][:19]}")

# 今日汇总
today = datetime.now().strftime('%Y-%m-%d')
cur.execute("""
    SELECT strategy_name, COUNT(*), SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END), 
           ROUND(SUM(pnl_usdt),6), ROUND(AVG(pnl_usdt),6)
    FROM trades WHERE exit_time > ? GROUP BY strategy_name
""", (today,))
print(f"\n今日({today})策略汇总:")
for r in cur.fetchall():
    wr = r[2]/r[1]*100 if r[1] else 0
    print(f"  {r[0]:12s}: {r[1]:3d}笔 | WR={wr:5.1f}% | 总PnL={r[3]:+.6f} | 均PnL={r[4]:+.6f}")

db.close()