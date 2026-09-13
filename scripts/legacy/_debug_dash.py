import sqlite3, os
from datetime import datetime

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

print("=== account_history schema ===")
for r in c.execute("PRAGMA table_info(account_history)"):
    print(dict(r))

print("\n=== account_history count ===")
print(c.execute("SELECT COUNT(*) FROM account_history").fetchone()[0])

print("\n=== account_history: min/max timestamp ===")
for r in c.execute("SELECT MIN(timestamp), MAX(timestamp) FROM account_history"):
    print(dict(r))

print("\n=== account_history: sample first 5 ===")
for r in c.execute("SELECT * FROM account_history ORDER BY timestamp ASC LIMIT 5"):
    print(dict(r))

print("\n=== account_history: sample last 5 ===")
for r in c.execute("SELECT * FROM account_history ORDER BY timestamp DESC LIMIT 5"):
    print(dict(r))

print("\n=== account_history: distinct timestamp formats (first 10 chars) ===")
for r in c.execute("SELECT DISTINCT substr(timestamp, 11, 1) AS sep, COUNT(*) FROM account_history GROUP BY sep"):
    print(dict(r))

# 时间戳格式检查：是否同时存在 T 和空格
print("\n=== timestamp containing 'T' vs space ===")
for r in c.execute("SELECT SUM(CASE WHEN instr(timestamp,'T')>0 THEN 1 ELSE 0 END) AS with_T, SUM(CASE WHEN instr(timestamp,'T')=0 THEN 1 ELSE 0 END) AS without_T FROM account_history"):
    print(dict(r))

c.close()
