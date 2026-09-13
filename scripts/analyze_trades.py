"""分析历史交易数据"""
import sqlite3
from datetime import datetime

conn = sqlite3.connect('data/trading.db')
cursor = conn.cursor()

# 总体统计
cursor.execute('SELECT COUNT(*), SUM(pnl), SUM(CASE WHEN pnl!=0 THEN 1 ELSE 0 END) FROM trade_records')
r = cursor.fetchone()
print(f"=== 总体统计 ===")
print(f"总交易数: {r[0]}, 总PnL: {r[1]}, 非零PnL数: {r[2]}")

# 状态分布
cursor.execute('SELECT status, COUNT(*) FROM trade_records GROUP BY status')
print(f"\n=== 状态分布 ===")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

# 策略统计
cursor.execute('SELECT strategy_name, COUNT(*), SUM(margin), SUM(pnl), AVG(pnl) FROM trade_records GROUP BY strategy_name')
print(f"\n=== 策略统计 ===")
columns = [d[0] for d in cursor.description]
for row in cursor.fetchall():
    print(f"  {dict(zip(columns, row))}")

# 币种统计
cursor.execute('SELECT symbol, COUNT(*), SUM(margin), SUM(pnl) FROM trade_records GROUP BY symbol')
print(f"\n=== 币种统计 ===")
columns = [d[0] for d in cursor.description]
for row in cursor.fetchall():
    print(f"  {dict(zip(columns, row))}")

# 价格滑点分析
cursor.execute("SELECT symbol, side, price, filled_price, quantity, margin, leverage, strategy_name FROM trade_records WHERE status='closed' ORDER BY create_time")
rows = cursor.fetchall()
print(f"\n=== 已平仓交易滑点分析 ===")
for r in rows:
    slippage = ((r[3]-r[2])/r[2]*100) if r[2] > 0 else 0
    print(f'{r[7]:6s} {r[1]:5s} {r[0]:15s} qty={r[4]:8.4f} px={r[2]:10.4f} fill={r[3]:10.4f} lev={r[6]} margin={r[5]:.4f} slippage={slippage:.3f}%')

# 交易频率分析
cursor.execute("SELECT symbol, COUNT(*), MIN(create_time), MAX(create_time) FROM trade_records GROUP BY symbol ORDER BY COUNT(*) DESC")
print(f"\n=== 交易频率分析 ===")
columns = [d[0] for d in cursor.description]
for row in cursor.fetchall():
    print(f"  {dict(zip(columns, row))}")

# 最近的交易
cursor.execute('SELECT id, symbol, strategy_name, side, status, pnl, pnl_percent, create_time, close_time FROM trade_records ORDER BY create_time DESC LIMIT 10')
print(f"\n=== 最近10笔交易 ===")
columns = [d[0] for d in cursor.description]
for row in cursor.fetchall():
    print(f"  {dict(zip(columns, row))}")

# 检查close_time不为空的交易
cursor.execute('SELECT COUNT(*) FROM trade_records WHERE close_time IS NOT NULL')
r = cursor.fetchone()
print(f"\n有close_time的记录数: {r[0]}")

# 检查pnl不为空的交易
cursor.execute('SELECT COUNT(*) FROM trade_records WHERE pnl != 0')
r = cursor.fetchone()
print(f"pnl不为0的记录数: {r[0]}")

conn.close()
