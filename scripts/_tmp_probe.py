import sqlite3, os, json

DB = os.path.join("data", "trading.db")
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print("=== trade_records 表结构 ===")
for r in c.execute("PRAGMA table_info(trade_records)"):
    print(f"  {r['name']} {r['type']}")

print("\n=== trade_records 中 TRUMP 记录（全字段） ===")
rows = list(c.execute("SELECT * FROM trade_records WHERE symbol LIKE '%TRUMP%'"))
for r in rows:
    print(json.dumps({k: r[k] for k in r.keys()}, ensure_ascii=False, default=str))

print("\n=== trades 表结构 ===")
for r in c.execute("PRAGMA table_info(trades)"):
    print(f"  {r['name']} {r['type']}")

print("\n=== trades 中 TRUMP 记录（全字段） ===")
rows = list(c.execute("SELECT * FROM trades WHERE symbol LIKE '%TRUMP%'"))
for r in rows:
    print(json.dumps({k: r[k] for k in r.keys()}, ensure_ascii=False, default=str))

print("\n=== 最近一次 pnl_reconciliation 记录 ===")
for r in c.execute("SELECT * FROM pnl_reconciliation ORDER BY rowid DESC LIMIT 2"):
    print(json.dumps({k: r[k] for k in r.keys()}, ensure_ascii=False, default=str))

c.close()
