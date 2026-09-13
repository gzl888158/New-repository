"""历史自动开单分析脚本 - 验证开单记录数据质量与一致性"""
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# 1. 表清单
tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print("=== 表清单 ===")
print(tables)

# 2. trade_records 概况
print("\n=== trade_records 概况 ===")
total = c.execute("SELECT COUNT(*) FROM trade_records").fetchone()[0]
print(f"总记录数: {total}")

print("\n--- 按 status ---")
for r in c.execute("SELECT status, COUNT(*) c FROM trade_records GROUP BY status ORDER BY c DESC"):
    print(f"  {r['status']:<10} {r['c']}")

print("\n--- 按 strategy_name ---")
for r in c.execute("SELECT strategy_name, COUNT(*) c FROM trade_records GROUP BY strategy_name ORDER BY c DESC"):
    print(f"  {r['strategy_name'] or '(空)':<20} {r['c']}")

print("\n--- 按 side ---")
for r in c.execute("SELECT side, COUNT(*) c FROM trade_records GROUP BY side ORDER BY c DESC"):
    print(f"  {r['side'] or '(空)':<10} {r['c']}")

print("\n--- 按 order_type ---")
for r in c.execute("SELECT order_type, COUNT(*) c FROM trade_records GROUP BY order_type ORDER BY c DESC"):
    print(f"  {r['order_type'] or '(空)':<20} {r['c']}")

# 3. 数据质量检查
print("\n=== 数据质量检查 ===")
checks = [
    ("quantity <= 0", "SELECT COUNT(*) FROM trade_records WHERE quantity <= 0"),
    ("price <= 0", "SELECT COUNT(*) FROM trade_records WHERE price <= 0"),
    ("filled_price <= 0", "SELECT COUNT(*) FROM trade_records WHERE filled_price <= 0"),
    ("fees IS NULL", "SELECT COUNT(*) FROM trade_records WHERE fees IS NULL"),
    ("fees = 0", "SELECT COUNT(*) FROM trade_records WHERE fees = 0"),
    ("leverage IS NULL or <= 0", "SELECT COUNT(*) FROM trade_records WHERE leverage IS NULL OR leverage <= 0"),
    ("side 空/异常", "SELECT COUNT(*) FROM trade_records WHERE side NOT IN ('buy','sell','long','short')"),
    ("strategy_name 空", "SELECT COUNT(*) FROM trade_records WHERE strategy_name IS NULL OR strategy_name = ''"),
    ("create_time 空", "SELECT COUNT(*) FROM trade_records WHERE create_time IS NULL"),
]
for label, sql in checks:
    n = c.execute(sql).fetchone()[0]
    flag = "  <-- 异常" if n > 0 else ""
    print(f"  {label}: {n}{flag}")

# 4. open 记录（未平仓自动开单）
print("\n=== 未平仓(open)自动开单 ===")
open_rows = c.execute("""
    SELECT symbol, strategy_name, side, order_type, quantity, price, leverage, margin, create_time
    FROM trade_records WHERE status='open' ORDER BY create_time DESC
""").fetchall()
print(f"open 记录数: {len(open_rows)}")
for r in open_rows:
    print(f"  {r['symbol']:<18} {r['strategy_name'] or '(空)':<12} {r['side']:<6} qty={r['quantity']} "
          f"px={r['price']} lev={r['leverage']} margin={r['margin']} t={r['create_time']}")

# 5. closed 记录对账
print("\n=== 已平仓(closed)开单统计 ===")
closed = c.execute("SELECT COUNT(*) FROM trade_records WHERE status='closed'").fetchone()[0]
print(f"closed 记录数: {closed}")
print("\n--- closed 按 exit_reason ---")
for r in c.execute("SELECT exit_reason, COUNT(*) c FROM trade_records WHERE status='closed' GROUP BY exit_reason ORDER BY c DESC"):
    print(f"  {r['exit_reason'] or '(空)':<20} {r['c']}")

# 6. 相同 symbol 的 open 重复检查（同一币种是否有多条 open 记录）
print("\n=== 同一币种多条 open 记录（可能重复开单）===")
dup = c.execute("""
    SELECT symbol, COUNT(*) c FROM trade_records WHERE status='open'
    GROUP BY symbol HAVING c > 1 ORDER BY c DESC
""").fetchall()
if dup:
    for r in dup:
        print(f"  {r['symbol']}: {r['c']} 条 open 记录")
else:
    print("  无重复 open 记录")

conn.close()
