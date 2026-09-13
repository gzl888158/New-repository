import sqlite3, os

p = os.path.join('data', 'trading.db')
c = sqlite3.connect(p)
c.row_factory = sqlite3.Row

# 带 T 的记录分布
print("=== 带 T 时间戳记录的日期分布 ===")
for r in c.execute(
    "SELECT substr(timestamp,1,10) AS d, COUNT(*) AS n "
    "FROM account_history WHERE instr(timestamp,'T')>0 GROUP BY d ORDER BY d"
):
    print(dict(r))

# 检查排序错乱：取 8月17日 按字符串排序的前5和后5
print("\n=== 8月17日 记录，按 timestamp 字符串排序 前3/后3 ===")
for r in c.execute(
    "SELECT timestamp, total_equity FROM account_history "
    "WHERE timestamp LIKE '2026-08-17%' ORDER BY timestamp ASC LIMIT 3"
):
    print("前:", dict(r))
for r in c.execute(
    "SELECT timestamp, total_equity FROM account_history "
    "WHERE timestamp LIKE '2026-08-17%' ORDER BY timestamp DESC LIMIT 3"
):
    print("后:", dict(r))

# 检查真正的物理时间最后一条（带T的应该是 isoformat 且包含微秒）
print("\n=== 8月17日 带T记录数 vs 带空格记录数 ===")
for r in c.execute(
    "SELECT SUM(CASE WHEN instr(timestamp,'T')>0 THEN 1 ELSE 0 END) AS withT, "
    "SUM(CASE WHEN instr(timestamp,'T')=0 THEN 1 ELSE 0 END) AS withSpace "
    "FROM account_history WHERE timestamp LIKE '2026-08-17%'"
):
    print(dict(r))

c.close()
