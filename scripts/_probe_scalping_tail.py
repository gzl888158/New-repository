"""分析剥头皮尾部大亏损：找出 pnl_usdt 最差的成交，看入场/离场原因。"""
import sqlite3, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db = os.path.join(BASE, "data", "trading.db")
c = sqlite3.connect(db)
c.row_factory = sqlite3.Row
cur = c.cursor()

print("== trades: scalping 最差 20 笔 (pnl_usdt 升序) ==")
rows = cur.execute(
    "SELECT trade_id, symbol, direction, entry_price, exit_price, quantity, leverage, "
    "entry_time, exit_time, fees, pnl_usdt, exit_reason, trace_id "
    "FROM trades WHERE strategy_name='scalping' AND pnl_usdt IS NOT NULL "
    "ORDER BY pnl_usdt ASC LIMIT 20"
).fetchall()
for r in rows:
    d = dict(r)
    hold = ""
    try:
        from datetime import datetime
        et = datetime.fromisoformat(d["entry_time"]); xt = datetime.fromisoformat(d["exit_time"])
        hold = f"{(xt-et).total_seconds():.0f}s"
    except Exception:
        pass
    print(f"{d['symbol']:<14} {d['direction']:<6} pnl={d['pnl_usdt']:+.4f} "
          f"qty={d['quantity']} lev={d['leverage']} "
          f"entry={d['entry_price']} exit={d['exit_price']} hold={hold} reason={d['exit_reason']}")

print("\n== exit_reason 分布 (scalping) ==")
for r in cur.execute(
    "SELECT exit_reason, COUNT(*) n, ROUND(SUM(pnl_usdt),4) total_pnl "
    "FROM trades WHERE strategy_name='scalping' GROUP BY exit_reason ORDER BY n DESC").fetchall():
    print(dict(r))

print("\n== 每 symbol 汇总 (scalping) ==")
for r in cur.execute(
    "SELECT symbol, COUNT(*) n, ROUND(SUM(pnl_usdt),4) total_pnl, "
    "ROUND(MIN(pnl_usdt),4) worst "
    "FROM trades WHERE strategy_name='scalping' GROUP BY symbol ORDER BY total_pnl ASC").fetchall():
    print(dict(r))

c.close()
print("\nDONE")