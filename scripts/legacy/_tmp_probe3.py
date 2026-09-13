import sqlite3
c = sqlite3.connect('data/trading.db')
print('trade_records.side:', c.execute("SELECT side, COUNT(*) FROM trade_records GROUP BY side").fetchall())
print('trades.direction:', c.execute("SELECT direction, COUNT(*) FROM trades GROUP BY direction").fetchall())
print('trade_records cols:', [r[1] for r in c.execute('PRAGMA table_info(trade_records)').fetchall()])
print('trades cols:', [r[1] for r in c.execute('PRAGMA table_info(trades)').fetchall()])
# sample of NULL exit_reason records
print('--- NULL exit_reason samples ---')
for r in c.execute("SELECT symbol, side, close_time, signal_type, status FROM trade_records WHERE status='closed' AND exit_reason IS NULL LIMIT 10"):
    print(r)
print('--- trades exit_reason samples ---')
for r in c.execute("SELECT symbol, direction, exit_time, exit_reason, pnl_usdt FROM trades WHERE exit_reason IS NOT NULL LIMIT 10"):
    print(r)
