import sqlite3
c = sqlite3.connect('data/trading.db')
print('exit_reason 分布:')
for r in c.execute("SELECT COALESCE(exit_reason,'(NULL)'), COUNT(*) FROM trade_records WHERE status='closed' GROUP BY exit_reason ORDER BY 2 DESC LIMIT 20"):
    print(r)
print('pnl=0/null closed 数量:', c.execute("SELECT COUNT(*) FROM trade_records WHERE status='closed' AND (pnl IS NULL OR pnl=0)").fetchone()[0])
