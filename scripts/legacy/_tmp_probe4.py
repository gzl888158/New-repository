import sqlite3
from datetime import datetime

c = sqlite3.connect('data/trading.db')
# 找一个 pnl=0 的 closed 记录，看看 close_time 是什么
rows = c.execute("SELECT symbol, side, close_time, create_time, price, quantity FROM trade_records WHERE status='closed' AND (pnl IS NULL OR pnl=0) ORDER BY close_time DESC LIMIT 10").fetchall()
for r in rows:
    print(r)

print('--- 最近 closed 记录时间范围 ---')
print(c.execute("SELECT MIN(close_time), MAX(close_time) FROM trade_records WHERE status='closed'").fetchone())
