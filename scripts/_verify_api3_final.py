"""最终验证：权威已实现盈亏是否计入 API3 manual_override 亏损。"""
import sqlite3, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = os.path.join(BASE, "data", "trading.db")
c = sqlite3.connect(db)
c.row_factory = sqlite3.Row
cur = c.cursor()

r = cur.execute("SELECT COALESCE(SUM(pnl_usdt),0) AS total, COUNT(*) AS n FROM trades").fetchone()
print(f"authoritative_realized_pnl = {r['total']:.4f}  (trades_count={r['n']})")

print("\n-- manual_override rows in trades --")
for x in cur.execute(
    "SELECT strategy_name, symbol, direction, quantity, fees, pnl_usdt, exit_reason "
    "FROM trades WHERE strategy_name='manual_override'").fetchall():
    print(dict(x))

print("\n-- 24h strategy pnl (trades, exit_time window) --")
for x in cur.execute(
    "SELECT strategy_name, ROUND(SUM(pnl_usdt),4) AS pnl "
    "FROM trades WHERE exit_time >= datetime('now','-24 hours') "
    "GROUP BY strategy_name ORDER BY pnl").fetchall():
    print(dict(x))

print("\n-- trade_records manual_override --")
for x in cur.execute(
    "SELECT strategy_name, symbol, pnl, pnl_percent, exit_reason "
    "FROM trade_records WHERE strategy_name='manual_override'").fetchall():
    print(dict(x))

c.close()
print("\nDONE")
