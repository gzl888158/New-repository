# -*- coding: utf-8 -*-
import sqlite3

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print("=== 各策略汇总(closed) ===")
for r in c.execute(
    "SELECT strategy_name, COUNT(*) n, SUM(pnl) pnl, "
    "SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) w "
    "FROM trade_records WHERE status='closed' GROUP BY strategy_name ORDER BY SUM(pnl)"
):
    wr = r["w"] / r["n"] * 100 if r["n"] else 0
    print(f"  {r['strategy_name']:12s} {r['n']:4d}笔 PnL={r['pnl'] or 0:+.4f} WR={wr:.1f}%")

print()
print("=== 单笔亏损 TOP10 ===")
for r in c.execute(
    "SELECT symbol,strategy_name,quantity,margin,pnl,exit_reason "
    "FROM trade_records WHERE status='closed' ORDER BY pnl ASC LIMIT 10"
):
    print(f"  {r['symbol']:16s} {r['strategy_name']:10s} qty={r['quantity'] or 0:8.2f} "
          f"margin={r['margin'] or 0:8.2f} pnl={r['pnl']:+.4f} reason={r['exit_reason']}")

print()
print("=== scalping 分币种 (剔除单笔异常后看剩余) ===")
for r in c.execute(
    "SELECT symbol, COUNT(*) n, SUM(pnl) pnl FROM trade_records "
    "WHERE status='closed' AND strategy_name='scalping' GROUP BY symbol ORDER BY SUM(pnl)"
):
    print(f"  {r['symbol']:16s} {r['n']:4d}笔 PnL={r['pnl'] or 0:+.4f}")

print()
print("=== 大仓检测: 单笔margin超过账户权益20% (>13 USDT) 的已平仓记录 ===")
for r in c.execute(
    "SELECT symbol,strategy_name,quantity,margin,pnl,exit_reason,create_time "
    "FROM trade_records WHERE status='closed' AND margin > 13 ORDER BY margin DESC LIMIT 15"
):
    print(f"  {r['symbol']:16s} {r['strategy_name']:10s} qty={r['quantity'] or 0:8.2f} "
          f"margin={r['margin'] or 0:8.2f} pnl={r['pnl']:+.4f} reason={r['exit_reason']} t={r['create_time']}")
