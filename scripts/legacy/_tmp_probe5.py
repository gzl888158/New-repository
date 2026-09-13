import sqlite3
c = sqlite3.connect('data/trading.db')
print('--- pnl=0/null closed 记录按 exit_reason 分布 ---')
for r in c.execute("SELECT COALESCE(exit_reason,'(NULL)'), COUNT(*) FROM trade_records WHERE status='closed' AND (pnl IS NULL OR pnl=0) GROUP BY exit_reason ORDER BY 2 DESC"):
    print(r)
print('--- 非 reconciled 的 pnl=0 记录：symbol/side/时间范围 ---')
for r in c.execute("SELECT symbol, side, MIN(close_time), MAX(close_time), COUNT(*) FROM trade_records WHERE status='closed' AND (pnl IS NULL OR pnl=0) AND (exit_reason IS NULL OR exit_reason!='reconciled') GROUP BY symbol, side ORDER BY 5 DESC LIMIT 30"):
    print(r)
