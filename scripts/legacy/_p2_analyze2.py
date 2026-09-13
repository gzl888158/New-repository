# -*- coding: utf-8 -*-
import sqlite3

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print("=== ghost_close/reconciled 按日期趋势 (grid) ===")
for r in c.execute(
    "SELECT DATE(close_time) d, exit_reason, COUNT(*) n "
    "FROM trade_records WHERE status='closed' AND strategy_name='grid' "
    "AND exit_reason IN ('ghost_close','reconciled') "
    "GROUP BY DATE(close_time), exit_reason ORDER BY d"
):
    print(f"  {r['d']}  {r['exit_reason']:12s} n={r['n']}")

print()
print("=== 全部策略 manual 平仓的 symbol 分布 ===")
for r in c.execute(
    "SELECT symbol, strategy_name, COUNT(*) n, SUM(pnl) pnl "
    "FROM trade_records WHERE status='closed' AND exit_reason='manual' "
    "GROUP BY symbol, strategy_name ORDER BY SUM(pnl) ASC LIMIT 15"
):
    print(f"  {r['symbol']:16s} {r['strategy_name']:10s} n={r['n']:3d} PnL={r['pnl'] or 0:+.4f}")

print()
print("=== manual 平仓中 quantity 异常大 (qty*price>50) 的记录 ===")
for r in c.execute(
    "SELECT symbol, strategy_name, quantity, margin, pnl, create_time, close_time "
    "FROM trade_records WHERE status='closed' AND exit_reason='manual' "
    "AND margin > 13 ORDER BY margin DESC LIMIT 10"
):
    print(f"  {r['symbol']:16s} {r['strategy_name']:10s} qty={r['quantity'] or 0:9.2f} "
          f"margin={r['margin'] or 0:9.2f} pnl={r['pnl']:+.4f} open={r['create_time']} close={r['close_time']}")
