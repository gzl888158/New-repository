import sqlite3

db = sqlite3.connect(r'e:\新建文件夹\okx_quant_trading\data\trading.db')
cur = db.cursor()

# Find losing symbols
cur.execute("""
    SELECT symbol, COUNT(*), ROUND(SUM(pnl_usdt),4),
    ROUND(SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1)
    FROM trades GROUP BY symbol 
    HAVING SUM(pnl_usdt) < -0.03 
    ORDER BY SUM(pnl_usdt)
""")
print("=== Losing symbols ===")
for r in cur.fetchall():
    print(f'{r[0]:18s} {r[1]:3d} trades PnL={r[2]:+.4f} WR={r[3]}%')

# Check POL and AVAX specifically
for sym in ['POL-USDT-SWAP', 'AVAX-USDT-SWAP']:
    cur.execute("""
        SELECT COUNT(*), ROUND(SUM(pnl_usdt),4),
        ROUND(SUM(CASE WHEN win=1 THEN 1 ELSE 0 END)*100.0/MAX(COUNT(*),1),1)
        FROM trades WHERE symbol = ?
    """, (sym,))
    r = cur.fetchone()
    print(f'\n{sym}: {r[0]} trades, PnL={r[1]}, WR={r[2]}%')

db.close()