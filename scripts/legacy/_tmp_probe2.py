import sqlite3
c = sqlite3.connect('data/trading.db')

print("=== trades 样本 (direction 分布) ===")
for r in c.execute("SELECT direction, COUNT(*) FROM trades GROUP BY direction").fetchall():
    print("  ", r)

print()
print("=== trades 样本 (exit_reason=None 的 trade_records 对应) ===")
print("trade_records closed 且 exit_reason IS NULL 的币种分布:")
for r in c.execute("SELECT symbol, strategy_name, COUNT(*), SUM(pnl) FROM trade_records WHERE status='closed' AND exit_reason IS NULL GROUP BY symbol, strategy_name ORDER BY COUNT(*) DESC LIMIT 30").fetchall():
    print("  ", r)

print()
print("=== trades 样本行 (前10) ===")
for r in c.execute("SELECT symbol, strategy_name, direction, exit_time, pnl_usdt, exit_reason FROM trades ORDER BY exit_time DESC LIMIT 10").fetchall():
    print("  ", r)

print()
print("=== trade_records exit_reason IS NULL 样本 (前10) ===")
for r in c.execute("SELECT symbol, strategy_name, side, close_time, pnl, create_time FROM trade_records WHERE status='closed' AND exit_reason IS NULL ORDER BY close_time DESC LIMIT 10").fetchall():
    print("  ", r)
