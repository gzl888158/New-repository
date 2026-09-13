import sqlite3, os
DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "trading.db")
c = sqlite3.connect(DB)
c.row_factory = sqlite3.Row

print("=== liquidation+risk_stop 记录详情 ===")
for r in c.execute("""
  SELECT id, symbol, strategy_name, side, quantity, price, filled_price, leverage, margin, pnl, pnl_percent, exit_reason, create_time, close_time
  FROM trade_records WHERE status='closed' AND exit_reason='liquidation+risk_stop'
"""):
    for k in r.keys():
        print(f"  {k}: {r[k]}")

print("\n=== ghost_close 但 pnl 非零的 33 条（应回到 recovered_close，取证） ===")
for r in c.execute("""
  SELECT id, symbol, side, quantity, margin, pnl, create_time, close_time
  FROM trade_records WHERE status='closed' AND exit_reason='ghost_close' AND pnl IS NOT NULL AND pnl != 0
  ORDER BY pnl ASC LIMIT 10
"""):
    print(f"  {r['symbol']} {r['side']} qty={r['quantity']} margin={r['margin']} pnl={round(r['pnl'],4)}")

print("\n=== 最大亏损的 5 笔 trades ===")
for r in c.execute("SELECT symbol, direction, entry_price, exit_price, quantity, pnl_usdt, exit_reason FROM trades ORDER BY pnl_usdt ASC LIMIT 5"):
    print(f"  {r['symbol']} {r['direction']} entry={r['entry_price']} exit={r['exit_price']} qty={r['quantity']} pnl={round(r['pnl_usdt'],4)} {r['exit_reason']}")