import sqlite3

conn = sqlite3.connect('./data/trading.db')
cursor = conn.cursor()

# 查看所有表
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = cursor.fetchall()
print("表列表:")
for t in tables:
    print(f"  {t[0]}")
    # 查看表结构
    cursor.execute(f"PRAGMA table_info({t[0]})")
    cols = cursor.fetchall()
    for c in cols:
        print(f"    {c[1]} ({c[2]})")
    print()

conn.close()
