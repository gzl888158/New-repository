import sqlite3
c = sqlite3.connect('data/trading.db')

print("=== trades 表 (TradeJournal) ===")
print("总记录数:", c.execute("SELECT COUNT(*) FROM trades").fetchone()[0])
print("exit_reason 分布:")
for r in c.execute("SELECT exit_reason, COUNT(*), SUM(pnl_usdt) FROM trades GROUP BY exit_reason").fetchall():
    print("  ", r)

print()
print("=== stop_loss_audit 表 ===")
print("总记录数:", c.execute("SELECT COUNT(*) FROM stop_loss_audit").fetchone()[0])
print("exit_reason 分布:")
for r in c.execute("SELECT exit_reason, COUNT(*), SUM(pnl) FROM stop_loss_audit GROUP BY exit_reason").fetchall():
    print("  ", r)

print()
print("=== trade_records 表 exit_reason/pnl 分布 (closed) ===")
for r in c.execute("SELECT exit_reason, status, COUNT(*), SUM(pnl) FROM trade_records GROUP BY exit_reason, status ORDER BY status, COUNT(*) DESC").fetchall():
    print("  ", r)
