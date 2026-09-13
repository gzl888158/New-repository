import sqlite3
import json
from datetime import datetime

conn = sqlite3.connect('data/trading.db')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables:", [t[0] for t in tables])

if ('trades',) in tables:
    cursor.execute('SELECT COUNT(*) FROM trades')
    trades_count = cursor.fetchone()[0]
    print(f"Trades count: {trades_count}")
    
    if trades_count > 0:
        cursor.execute('SELECT * FROM trades ORDER BY entry_time DESC LIMIT 10')
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        print("Recent trades:")
        for row in rows:
            print(dict(zip(columns, row)))
        
        cursor.execute('SELECT strategy_name, COUNT(*), SUM(pnl_usdt), AVG(pnl_usdt) FROM trades GROUP BY strategy_name')
        print("\nStrategy stats:")
        for r in cursor.fetchall():
            print(f"  {r[0]}: trades={r[1]}, total_pnl={r[2]:.2f}, avg_pnl={r[3]:.2f}")
else:
    print("No trades table found")

if ('equity_curve',) in tables:
    cursor.execute('SELECT COUNT(*) FROM equity_curve')
    equity_count = cursor.fetchone()[0]
    print(f"\nEquity curve count: {equity_count}")
    
    if equity_count > 0:
        cursor.execute('SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT 5')
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        print("Recent equity points:")
        for row in rows:
            print(dict(zip(columns, row)))
else:
    print("No equity_curve table found")

conn.close()
