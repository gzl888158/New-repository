import sqlite3
from datetime import datetime

db = sqlite3.connect(r'e:\新建文件夹\okx_quant_trading\data\trading.db')
cur = db.cursor()

# 24h stats
cur.execute("""
    SELECT COUNT(*), ROUND(SUM(pnl_usdt),4), ROUND(AVG(pnl_usdt),4),
    ROUND(SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1)
    FROM trades WHERE exit_time > datetime('now','-24 hours')
""")
r = cur.fetchone()
print(f'=== 24h: {r[0]} trades, PnL={r[1]}, avg={r[2]}, WR={r[3]}% ===')

# By strategy (24h)
cur.execute("""
    SELECT strategy_name, COUNT(*), ROUND(SUM(pnl_usdt),4),
    ROUND(SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1)
    FROM trades WHERE exit_time > datetime('now','-24 hours')
    GROUP BY strategy_name
""")
print('\nBy strategy (24h):')
for r in cur.fetchall():
    print(f'  {r[0]:12s} {r[1]:3d} trades PnL={r[2]:+.4f} WR={r[3]}%')

# By symbol (24h)
cur.execute("""
    SELECT symbol, COUNT(*), ROUND(SUM(pnl_usdt),4),
    ROUND(SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1)
    FROM trades WHERE exit_time > datetime('now','-24 hours')
    GROUP BY symbol ORDER BY SUM(pnl_usdt) DESC LIMIT 15
""")
print('\nBy symbol (24h):')
for r in cur.fetchall():
    print(f'  {r[0]:18s} {r[1]:3d} trades PnL={r[2]:+.4f} WR={r[3]}%')

# Recent trades with reason
cur.execute("""
    SELECT trade_id, symbol, strategy_name, direction, pnl_usdt, win, exit_time, exit_reason
    FROM trades ORDER BY exit_time DESC LIMIT 20
""")
print('\n=== Recent 20 trades ===')
for r in cur.fetchall():
    et = r[6][:19] if r[6] else "N/A"
    print(f'{r[1]:12s} {r[2]:10s} {r[3]:6s} pnl={r[4]:+.4f} win={r[5]} {et} {r[7]}')

# Hourly PnL
cur.execute("""
    SELECT strftime('%Hh', exit_time) as hour, COUNT(*), ROUND(SUM(pnl_usdt),4),
    ROUND(SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1)
    FROM trades WHERE exit_time > datetime('now','-24 hours')
    GROUP BY hour ORDER BY hour
""")
print('\nBy hour (24h):')
for r in cur.fetchall():
    print(f'  {r[0]:5s} {r[1]:3d} trades PnL={r[2]:+.4f} WR={r[3]}%')

# Active positions check - from trades table
cur.execute("""
    SELECT symbol, direction, COUNT(*) as cnt, ROUND(SUM(pnl_usdt),4) as total_pnl
    FROM trades WHERE status = 'open'
    GROUP BY symbol, direction
""")
print('\n=== Open trades in DB ===')
for r in cur.fetchall():
    print(f'  {r[0]:12s} {r[1]:6s} {r[2]:2d} trades PnL={r[3]:+.4f}')

db.close()