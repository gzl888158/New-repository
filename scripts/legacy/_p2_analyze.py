# -*- coding: utf-8 -*-
import sqlite3

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print("=== grid exit_reason 分布 (closed) ===")
for r in c.execute(
    "SELECT exit_reason, COUNT(*) n, SUM(pnl) pnl, "
    "100.0*SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)/COUNT(*) wr "
    "FROM trade_records WHERE status='closed' AND strategy_name='grid' "
    "GROUP BY exit_reason ORDER BY SUM(pnl)"
):
    print(f"  {str(r['exit_reason']):20s} n={r['n']:5d} PnL={r['pnl'] or 0:+.4f} WR={r['wr'] or 0:.1f}%")

print()
print("=== grid 磨损型交易: |pnl| <= fees (纯手续费损耗) ===")
for r in c.execute(
    "SELECT COUNT(*) n, SUM(pnl) pnl, SUM(fees) fees "
    "FROM trade_records WHERE status='closed' AND strategy_name='grid' "
    "AND ABS(pnl) <= fees AND pnl < 0"
):
    total = c.execute("SELECT COUNT(*) FROM trade_records WHERE status='closed' AND strategy_name='grid'").fetchone()[0]
    print(f"  磨损单 n={r['n']} / {total} = {r['n']/total*100 if total else 0:.1f}%  PnL={r['pnl'] or 0:+.4f} fees={r['fees'] or 0:.4f}")

print()
print("=== trend exit_reason 分布 (closed) ===")
for r in c.execute(
    "SELECT exit_reason, COUNT(*) n, SUM(pnl) pnl, "
    "100.0*SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)/COUNT(*) wr "
    "FROM trade_records WHERE status='closed' AND strategy_name='trend' "
    "GROUP BY exit_reason ORDER BY SUM(pnl)"
):
    print(f"  {str(r['exit_reason']):20s} n={r['n']:5d} PnL={r['pnl'] or 0:+.4f} WR={r['wr'] or 0:.1f}%")

print()
print("=== scalping exit_reason 分布 (closed) ===")
for r in c.execute(
    "SELECT exit_reason, COUNT(*) n, SUM(pnl) pnl, "
    "100.0*SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END)/COUNT(*) wr "
    "FROM trade_records WHERE status='closed' AND strategy_name='scalping' "
    "GROUP BY exit_reason ORDER BY SUM(pnl)"
):
    print(f"  {str(r['exit_reason']):20s} n={r['n']:5d} PnL={r['pnl'] or 0:+.4f} WR={r['wr'] or 0:.1f}%")

print()
print("=== grid 分币种 PnL TOP5 亏损 ===")
for r in c.execute(
    "SELECT symbol, COUNT(*) n, SUM(pnl) pnl, SUM(fees) fees "
    "FROM trade_records WHERE status='closed' AND strategy_name='grid' "
    "GROUP BY symbol ORDER BY SUM(pnl) ASC LIMIT 5"
):
    print(f"  {r['symbol']:16s} n={r['n']:4d} PnL={r['pnl'] or 0:+.4f} fees={r['fees'] or 0:+.4f}")

print()
print("=== 当前未平仓记录 (status='open') 大仓检测 ===")
for r in c.execute(
    "SELECT symbol, strategy_name, quantity, margin, create_time "
    "FROM trade_records WHERE status='open' ORDER BY margin DESC LIMIT 15"
):
    print(f"  {r['symbol']:16s} {r['strategy_name']:10s} qty={r['quantity'] or 0:9.2f} margin={r['margin'] or 0:9.2f} t={r['create_time']}")
