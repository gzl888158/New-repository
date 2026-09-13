import sqlite3

conn = sqlite3.connect('data/trading.db')
c = conn.cursor()

print("=== 表结构 ===")
tables = c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
for table in tables:
    print(f"  {table[0]}")

print("\n=== trades 表结构 ===")
info = c.execute('PRAGMA table_info(trades)').fetchall()
for col in info:
    print(f"  {col[1]} ({col[2]})")

print("\n=== trades 最近记录 ===")
trades = c.execute('SELECT * FROM trades ORDER BY created_at DESC LIMIT 20').fetchall()
if trades:
    for t in trades:
        print(f"  {t[:8]}...")
else:
    print("  无交易记录")

print("\n=== signals 表结构 ===")
info = c.execute('PRAGMA table_info(signals)').fetchall()
for col in info:
    print(f"  {col[1]} ({col[2]})")

print("\n=== signals 最近记录 ===")
signals = c.execute('SELECT * FROM signals ORDER BY created_at DESC LIMIT 20').fetchall()
if signals:
    for s in signals:
        print(f"  {s[:6]}...")
else:
    print("  无信号记录")

conn.close()
