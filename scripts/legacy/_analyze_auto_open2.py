"""历史自动开单深入验证 - 量化数据质量问题的实际影响"""
import sqlite3
from collections import Counter, defaultdict

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
c = conn.cursor()

# 1. fees 缺失按策略分布
print("=== fees=0 按策略分布（手续费缺失的根因）===")
for r in c.execute("""
    SELECT strategy_name, COUNT(*) total, SUM(CASE WHEN fees=0 THEN 1 ELSE 0 END) fee_zero
    FROM trade_records GROUP BY strategy_name ORDER BY total DESC
"""):
    total = r["total"]; zero = r["fee_zero"]
    print(f"  {r['strategy_name'] or '(空)':<12} 总{total:>5}  fee=0 {zero:>5} ({zero/total*100:.0f}%)")

# 2. filled_price 缺失按策略/exit_reason
print("\n=== filled_price=0 按策略分布 ===")
for r in c.execute("""
    SELECT strategy_name, COUNT(*) c FROM trade_records
    WHERE filled_price <= 0 GROUP BY strategy_name ORDER BY c DESC
"""):
    print(f"  {r['strategy_name'] or '(空)':<12} {r['c']}")

print("\n=== filled_price=0 按 exit_reason 分布 ===")
for r in c.execute("""
    SELECT exit_reason, COUNT(*) c FROM trade_records
    WHERE filled_price <= 0 GROUP BY exit_reason ORDER BY c DESC
"""):
    print(f"  {r['exit_reason'] or '(空)':<20} {r['c']}")

# 3. exit_reason 交叉策略（真实平仓 vs 幽灵对账）
print("\n=== exit_reason × strategy 交叉 ===")
for r in c.execute("""
    SELECT strategy_name, exit_reason, COUNT(*) c FROM trade_records
    WHERE status='closed' GROUP BY strategy_name, exit_reason
    ORDER BY strategy_name, c DESC
"""):
    print(f"  {r['strategy_name'] or '(空)':<12} {r['exit_reason'] or '(空)':<20} {r['c']}")

# 4. 真实平仓（非幽灵）vs 幽灵对账 统计
print("\n=== 平仓类型汇总 ===")
ghost_reasons = ("ghost_close", "reconciled", "ghost_cleanup", "orphaned_cleanup")
real = c.execute(
    "SELECT COUNT(*) FROM trade_records WHERE status='closed' AND exit_reason NOT IN ('ghost_close','reconciled','ghost_cleanup','orphaned_cleanup','')"
).fetchone()[0]
ghost = c.execute(
    "SELECT COUNT(*) FROM trade_records WHERE status='closed' AND exit_reason IN ('ghost_close','reconciled','ghost_cleanup','orphaned_cleanup')"
).fetchone()[0]
empty = c.execute(
    "SELECT COUNT(*) FROM trade_records WHERE status='closed' AND (exit_reason IS NULL OR exit_reason='')"
).fetchone()[0]
print(f"  真实平仓: {real}")
print(f"  幽灵对账: {ghost}")
print(f"  exit_reason空: {empty}")

# 5. PnL 分布：有真实 pnl 的记录 vs pnl=0/null
print("\n=== PnL 数据完整性 ===")
pnl_zero = c.execute("SELECT COUNT(*) FROM trade_records WHERE pnl IS NULL OR pnl=0").fetchone()[0]
pnl_nonzero = c.execute("SELECT COUNT(*) FROM trade_records WHERE pnl IS NOT NULL AND pnl != 0").fetchone()[0]
print(f"  pnl=0/null: {pnl_zero}")
print(f"  pnl非0: {pnl_nonzero}")

# 6. 各策略真实 pnl 汇总（仅真实平仓，非幽灵）
print("\n=== 真实平仓 pnl 汇总（排除幽灵对账）===")
for r in c.execute("""
    SELECT strategy_name, COUNT(*) c, SUM(pnl) total_pnl, SUM(fees) total_fee,
           SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) wins
    FROM trade_records WHERE status='closed'
      AND exit_reason NOT IN ('ghost_close','reconciled','ghost_cleanup','orphaned_cleanup','')
      AND pnl IS NOT NULL
    GROUP BY strategy_name ORDER BY total_pnl
"""):
    wr = r["wins"]/r["c"]*100 if r["c"] else 0
    print(f"  {r['strategy_name'] or '(空)':<12} 笔{r['c']:>4} pnl={r['total_pnl'] or 0:>8.4f} "
          f"fee={r['total_fee'] or 0:>8.4f} 胜率{wr:>5.1f}%")

# 7. 手续费估算影响：真实平仓记录按 0.05% taker 补算 vs 已记录 fee
print("\n=== 手续费低估影响（真实平仓，非幽灵）===")
rows = c.execute("""
    SELECT symbol, side, filled_price, quantity, fees FROM trade_records
    WHERE status='closed' AND pnl IS NOT NULL AND pnl != 0
      AND exit_reason NOT IN ('ghost_close','reconciled','ghost_cleanup','orphaned_cleanup','')
""").fetchall()
est_fee = 0.0
rec_fee = 0.0
n = 0
for r in rows:
    px = r["filled_price"] or r["price"] or 0
    qty = r["quantity"] or 0
    if px > 0 and qty > 0:
        est_fee += px * qty * 0.0005 * 2  # 开+平双向
        rec_fee += (r["fees"] or 0)
        n += 1
print(f"  样本数(有真实pnl): {n}")
print(f"  已记录手续费: {rec_fee:.4f} USDT")
print(f"  按0.05% taker双向估算: {est_fee:.4f} USDT")
print(f"  手续费低估: {est_fee - rec_fee:.4f} USDT")

# 8. 总 pnl（真实平仓）扣费前后对比
total_pnl = c.execute("""
    SELECT SUM(pnl) FROM trade_records WHERE status='closed' AND pnl IS NOT NULL
      AND exit_reason NOT IN ('ghost_close','reconciled','ghost_cleanup','orphaned_cleanup','')
""").fetchone()[0] or 0
print(f"\n  真实平仓总PnL(未扣费已按逻辑含): {total_pnl:.4f} USDT")

conn.close()
