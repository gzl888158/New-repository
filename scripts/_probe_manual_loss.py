import sqlite3, datetime
c = sqlite3.connect('data/trading.db')
c.row_factory = sqlite3.Row

print('=== 1. trade_records 里 sync 策略（手动仓占位）全部记录 ===')
rs = c.execute("SELECT * FROM trade_records WHERE strategy_name='sync' ORDER BY create_time").fetchall()
for r in rs:
    print(f"  {r['symbol']:16s} side={r['side']:5s} qty={r['quantity']} "
          f"price={r['price']} filled_price={r['filled_price']} "
          f"pnl={r['pnl']} fees={r['fees']} exit={r['exit_reason']} "
          f"create={r['create_time']} close={r['close_time']}")

print('\n=== 2. trade_records 里 API3 全部记录 ===')
rs = c.execute("SELECT * FROM trade_records WHERE symbol LIKE 'API3%' ORDER BY create_time").fetchall()
for r in rs:
    print(f"  {r['strategy_name']:12s} side={r['side']:5s} qty={r['quantity']} "
          f"price={r['price']} filled_price={r['filled_price']} pnl={r['pnl']} "
          f"exit={r['exit_reason']} create={r['create_time']} close={r['close_time']}")

print('\n=== 3. 24h trades 全部（含 sync/grid/scalping）===')
cut = (datetime.datetime.now() - datetime.timedelta(hours=24)).isoformat()
rs = c.execute("SELECT * FROM trades WHERE exit_time >= ? ORDER BY exit_time", (cut,)).fetchall()
for r in rs:
    print(f"  {r['symbol']:16s} {r['strategy_name']:10s} {r['direction']:5s} "
          f"pnl_usdt={r['pnl_usdt']} fees={r['fees']} exit={r['exit_reason']} "
          f"exit_time={r['exit_time']}")

c.close()
