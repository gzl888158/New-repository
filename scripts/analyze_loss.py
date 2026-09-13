"""分析资金消耗根因"""
import sqlite3

conn = sqlite3.connect('data/trading.db')
c = conn.cursor()

# 1. 已平仓交易统计
c.execute("SELECT COUNT(*), SUM(margin), AVG(margin), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), SUM(CASE WHEN pnl<0 THEN 1 ELSE 0 END), SUM(pnl) FROM trade_records WHERE status='closed'")
r = c.fetchone()
print("=== 已平仓交易统计 ===")
print(f"笔数={r[0]}, 总保证金={r[1] or 0:.4f}, 平均保证金={r[2] or 0:.4f}")
print(f"盈利单={r[3] or 0}, 亏损单={r[4] or 0}, 总盈亏={r[5] or 0:.4f}")

# 2. 滑点分析
c.execute("SELECT symbol, COUNT(*), AVG(price), AVG(filled_price) FROM trade_records WHERE status='closed' AND filled_price>0 GROUP BY symbol")
print("\n=== 滑点分析 ===")
for r in c.fetchall():
    slip = ((r[3]-r[2])/r[2]*100) if r[2] > 0 else 0
    print(f"  {r[0]:16s}: count={r[1]}, avg_price={r[2]:.4f}, avg_fill={r[3]:.4f}, slippage={slip:.3f}%")

# 3. 策略-方向分布
c.execute("SELECT strategy_name, side, COUNT(*) FROM trade_records GROUP BY strategy_name, side ORDER BY strategy_name, side")
print("\n=== 策略-方向分布 ===")
for r in c.fetchall():
    print(f"  {r[0]:8s} {r[1]:6s}: {r[2]}")

# 4. 手续费估算 (0.05% taker fee)
c.execute("SELECT SUM(filled_price * quantity * 0.0005) FROM trade_records WHERE status='closed' AND filled_price>0")
fee = c.fetchone()[0] or 0
print(f"\n=== 手续费估算 (taker 0.05%) ===")
print(f"  已平仓交易总手续费: {fee:.4f} USDT")

c.execute("SELECT SUM(filled_price * quantity * 0.0005) FROM trade_records WHERE status='open' AND filled_price>0")
fee_open = c.fetchone()[0] or 0
print(f"  持仓中交易手续费: {fee_open:.4f} USDT")
print(f"  总手续费: {fee + fee_open:.4f} USDT")

# 5. 交易频率分析
c.execute("SELECT DATE(create_time) as d, COUNT(*), SUM(margin) FROM trade_records GROUP BY d ORDER BY d")
print("\n=== 每日交易频率 ===")
for r in c.fetchall():
    print(f"  {r[0]}: {r[1]}笔, 保证金={r[2] or 0:.4f}")

# 6. 单笔保证金分布
c.execute("SELECT MIN(margin), MAX(margin), AVG(margin) FROM trade_records WHERE margin>0")
r = c.fetchone()
print(f"\n=== 单笔保证金分布 ===")
print(f"  最小={r[0]:.4f}, 最大={r[1]:.4f}, 平均={r[2]:.4f}")

# 7. 小于最小有效保证金的交易数
c.execute("SELECT COUNT(*) FROM trade_records WHERE margin < 0.5")
small = c.fetchone()[0]
print(f"  保证金<0.5 USDT的交易数: {small} (这些交易手续费占比过高)")

conn.close()
