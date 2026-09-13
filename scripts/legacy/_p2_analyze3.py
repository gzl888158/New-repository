# -*- coding: utf-8 -*-
import sqlite3

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print("=== reconciled/ghost_close 记录的 filled_price 分布 ===")
for r in c.execute(
    "SELECT exit_reason, "
    "SUM(CASE WHEN (filled_price IS NULL OR filled_price=0) THEN 1 ELSE 0 END) no_fill, "
    "SUM(CASE WHEN filled_price>0 THEN 1 ELSE 0 END) has_fill, "
    "COUNT(*) total "
    "FROM trade_records WHERE status='closed' AND exit_reason IN ('reconciled','ghost_close') "
    "GROUP BY exit_reason"
):
    print(f"  {r['exit_reason']:12s} no_fill={r['no_fill']} has_fill={r['has_fill']} total={r['total']}")

print()
print("=== 当前 open 记录的 filled_price 分布 ===")
for r in c.execute(
    "SELECT "
    "SUM(CASE WHEN (filled_price IS NULL OR filled_price=0) THEN 1 ELSE 0 END) no_fill, "
    "SUM(CASE WHEN filled_price>0 THEN 1 ELSE 0 END) has_fill, "
    "COUNT(*) total "
    "FROM trade_records WHERE status='open'"
):
    print(f"  no_fill={r['no_fill']} has_fill={r['has_fill']} total={r['total']}")

print()
print("=== reconciled 记录 order_type 分布 ===")
for r in c.execute(
    "SELECT order_type, COUNT(*) n FROM trade_records "
    "WHERE status='closed' AND exit_reason IN ('reconciled','ghost_close') GROUP BY order_type"
):
    print(f"  {str(r['order_type']):15s} n={r['n']}")
