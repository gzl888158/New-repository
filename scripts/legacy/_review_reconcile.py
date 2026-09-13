# -*- coding: utf-8 -*-
"""数据源核对：表清单 + 各表 trend/scalping 汇总，定位 0%/0.1% 胜率结论的来源"""
import sqlite3

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
c = conn.cursor()

print("=== 表清单 ===")
for r in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
    print(" ", r[0])

print("\n=== strategy_performance 表 ===")
try:
    for r in c.execute("SELECT * FROM strategy_performance ORDER BY strategy_name, symbol"):
        print(" ", dict(r))
except Exception as e:
    print("  无/错误:", e)

print("\n=== trades 表按策略总览(全部状态) ===")
for r in c.execute(
    "SELECT strategy_name, COUNT(*), SUM(CASE WHEN exit_time IS NOT NULL THEN 1 ELSE 0 END), "
    "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(pnl_usdt) FROM trades "
    "GROUP BY strategy_name ORDER BY COUNT(*) DESC"):
    total = r[1] or 0
    closed = r[2] or 0
    wins = r[3] or 0
    print(f"  {str(r[0]):14s} 总{total:5d} 已平{closed:5d} 胜{wins:4d} 胜率{wins/max(closed,1)*100:5.1f}% PnL{r[4] or 0:+9.4f}")

print("\n=== stop_loss_audit 表 trend/scalping ===")
try:
    for r in c.execute(
        "SELECT strategy_name, COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), "
        "SUM(pnl), AVG(pnl), AVG(pnl_percent) FROM stop_loss_audit "
        "WHERE strategy_name IN ('trend','scalping') GROUP BY strategy_name"):
        n = r[1] or 0
        print(f"  {r[0]:12s} {n}笔 胜率{r[2]/max(n,1)*100:5.1f}% PnL{r[3] or 0:+.4f} 均PnL{r[4] or 0:+.4f} 均PnL%{r[5] or 0:+.4%}")
except Exception as e:
    print("  无/错误:", e)

print("\n=== trade_records 表 trend/scalping (SQLAlchemy) ===")
try:
    for r in c.execute(
        "SELECT strategy_name, COUNT(*), SUM(CASE WHEN status='closed' THEN 1 ELSE 0 END), "
        "SUM(pnl) FROM trade_records WHERE strategy_name IN ('trend','scalping') GROUP BY strategy_name"):
        print(f"  {r[0]:12s} 总{r[1]} 已平{r[2]} PnL{r[3] or 0:+.4f}")
except Exception as e:
    print("  无/错误:", e)

conn.close()
