import sqlite3, os
from collections import defaultdict

db = os.path.join("data", "trading.db")
conn = sqlite3.connect(db)
conn.row_factory = sqlite3.Row

# 1. trend 策略 exit_reason 分布
print("=" * 70)
print("trend 策略 exit_reason 分布（全历史）")
print("=" * 70)
rows = conn.execute(
    "SELECT exit_reason, COUNT(*) cnt, SUM(pnl_usdt) pnl, SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) wins, "
    "SUM(fees) fees, AVG(pnl_usdt) avg_pnl, MIN(pnl_usdt) min_pnl, MAX(pnl_usdt) max_pnl "
    "FROM trades WHERE strategy_name='trend' GROUP BY exit_reason ORDER BY cnt DESC"
).fetchall()
for r in rows:
    n = r['cnt'] or 0
    wr = (r['wins'] or 0) / n if n else 0
    print(f"{str(r['exit_reason']):24s} cnt={n:3d} pnl={r['pnl']:+.4f} wr={wr:.0%} avg={r['avg_pnl']:+.5f} "
          f"min={r['min_pnl']:+.5f} max={r['max_pnl']:+.5f} fees={r['fees']:.4f}")

# 2. 盈亏分布
print()
print("=" * 70)
print("trend 策略盈亏分布（win/loss 对比）")
print("=" * 70)
wins = conn.execute("SELECT pnl_usdt FROM trades WHERE strategy_name='trend' AND pnl_usdt>0").fetchall()
losses = conn.execute("SELECT pnl_usdt FROM trades WHERE strategy_name='trend' AND pnl_usdt<=0").fetchall()
w = [r['pnl_usdt'] for r in wins]
l = [r['pnl_usdt'] for r in losses]
if w:
    print(f"盈利单 {len(w)} 笔: 总 +{sum(w):.4f}, 平均 +{sum(w)/len(w):.5f}, 最大 +{max(w):.5f}")
if l:
    print(f"亏损单 {len(l)} 笔: 总 {sum(l):.4f}, 平均 {sum(l)/len(l):.5f}, 最小 {min(l):.5f}")
if w and l:
    avg_win = sum(w)/len(w); avg_loss = abs(sum(l)/len(l))
    print(f"盈亏比 (avg_win/avg_loss) = {avg_win/avg_loss:.3f}  ->  {'健康(>1.5)' if avg_win/avg_loss>=1.5 else '倒挂(小赢大亏)'}")

# 3. 近24h trend 明细
print()
print("=" * 70)
print("trend 策略近24h交易明细")
print("=" * 70)
rows = conn.execute(
    "SELECT symbol, direction, entry_price, exit_price, quantity, pnl_usdt, fees, entry_time, exit_time, exit_reason "
    "FROM trades WHERE strategy_name='trend' AND replace(exit_time,'T',' ') > datetime('now','-24 hours') "
    "ORDER BY exit_time"
).fetchall()
for r in rows:
    print(f"{r['symbol']:16s} {r['direction']:5s} qty={r['quantity']:.4f} entry={r['entry_price']:.5f} exit={r['exit_price']:.5f} "
          f"pnl={r['pnl_usdt']:+.5f} fees={r['fees']:.5f} reason={r['exit_reason']}")

# 4. 近7天 trend 明细（全部）
print()
print("=" * 70)
print("trend 策略近7天交易明细（含手续费后净盈亏）")
print("=" * 70)
rows = conn.execute(
    "SELECT symbol, direction, quantity, entry_price, exit_price, pnl_usdt, fees, exit_reason, exit_time "
    "FROM trades WHERE strategy_name='trend' AND replace(exit_time,'T',' ') > datetime('now','-7 days') "
    "ORDER BY exit_time"
).fetchall()
for r in rows:
    print(f"{r['exit_time'][:16]:16s} {r['symbol']:16s} {r['direction']:5s} qty={r['quantity']:.4f} "
          f"e={r['entry_price']:.5f}->x={r['exit_price']:.5f} pnl={r['pnl_usdt']:+.5f} fees={r['fees']:.5f} {r['exit_reason']}")

conn.close()
