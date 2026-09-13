import sqlite3
db = sqlite3.connect(r'e:\新建文件夹\okx_quant_trading\data\trading.db')
cur = db.cursor()
cur.execute("""
    SELECT strategy_name, COUNT(*), ROUND(SUM(pnl_usdt),4),
    ROUND(SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1),
    MAX(exit_time)
    FROM trades GROUP BY strategy_name
""")
print("=== By strategy (all time) ===")
for r in cur.fetchall():
    print(f'{r[0]:12s} {r[1]:3d} trades PnL={r[2]:+.4f} WR={r[3]}% last={r[4]}')

# Check trend and scalping specifically
for s in ['trend', 'scalping']:
    cur.execute("""
        SELECT COUNT(*), ROUND(SUM(pnl_usdt),4),
        ROUND(SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1)
        FROM trades WHERE strategy_name = ? AND status = 'closed'
    """, (s,))
    r = cur.fetchone()
    print(f'\n{s} closed: {r[0]} trades, PnL={r[1]}, WR={r[2]}%')

db.close()