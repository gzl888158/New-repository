# -*- coding: utf-8 -*-
"""临时分析：trend明细 + AVAX明细 + learning_state黑名单"""
import sqlite3, os

DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db")
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()

print("=== trend 最近24小时全部明细 ===")
rows = cur.execute("""
    SELECT * FROM trades
    WHERE strategy_name='trend'
      AND replace(exit_time, 'T', ' ') >= datetime('now', 'localtime', '-24 hours')
    ORDER BY exit_time
""").fetchall()
for r in rows:
    print(f"  {r['exit_time'][11:19]} {r['symbol']:16s} {r['direction']:5s} "
          f"entry={r['entry_price']} exit={r['exit_price']} qty={r['quantity']} "
          f"pnl={float(r['pnl_usdt'] or 0):+.4f} fee={float(r['fees'] or 0):.4f} reason={r['exit_reason']}")

print("\n=== AVAX 近7天全部明细 ===")
rows = cur.execute("""
    SELECT * FROM trades
    WHERE symbol='AVAX-USDT-SWAP'
      AND replace(exit_time, 'T', ' ') >= datetime('now', 'localtime', '-7 days')
    ORDER BY exit_time
""").fetchall()
for r in rows:
    print(f"  {r['exit_time'][:16]} {r['strategy_name']:10s} {r['direction']:5s} "
          f"pnl={float(r['pnl_usdt'] or 0):+.4f} fee={float(r['fees'] or 0):.4f} reason={r['exit_reason']}")

print("\n=== learning_state 表内容 ===")
rows = cur.execute("SELECT key, substr(value,1,400) as v, updated_at FROM learning_state ORDER BY updated_at DESC LIMIT 40").fetchall()
for r in rows:
    print(f"  [{r['key']}] {r['v']}")

conn.close()
