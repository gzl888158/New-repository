"""诊断 web 持仓明细不同步：对比 DB position_history 与熔断状态"""
import sqlite3, os, sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "data", "trading.db")

conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

print("=== position_history 最近 10 分钟快照（按 symbol+side 最新） ===")
rows = conn.execute(
    "SELECT symbol, side, quantity, avg_cost, mark_price, margin, leverage, timestamp "
    "FROM position_history "
    "WHERE timestamp > datetime('now', '-10 minutes') "
    "ORDER BY timestamp DESC LIMIT 60"
).fetchall()

seen = {}
for r in rows:
    key = (r["symbol"], r["side"])
    if key not in seen:
        seen[key] = r

for key, r in sorted(seen.items()):
    print(f"  {r['symbol']:18s} {r['side']:6s} qty={r['quantity']} "
          f"mark={r['mark_price']} margin={r['margin']} lev={r['leverage']} ts={r['timestamp']}")

print(f"\n共 {len(seen)} 个 (symbol, side) 组合")

print("\n=== trade_records 当前 open 持仓 ===")
open_rows = conn.execute(
    "SELECT symbol, strategy_name, side, quantity, filled_price, leverage, margin, status, create_time "
    "FROM trade_records WHERE status='open' ORDER BY create_time DESC"
).fetchall()
if not open_rows:
    print("  (无 open 持仓)")
for r in open_rows:
    print(f"  {r['symbol']:18s} {r['strategy_name']:14s} {r['side']:5s} "
          f"qty={r['quantity']} px={r['filled_price']} margin={r['margin']} "
          f"lev={r['leverage']} ts={r['create_time']}")

conn.close()
