"""企业级交易策略全面分析"""
import sqlite3, json, os, statistics
from collections import defaultdict
from datetime import datetime, timedelta

DB = 'e:/新建文件夹/okx_quant_trading/data/trading.db'
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

print("=" * 70)
print("  企业级交易策略全面分析")
print(f"  分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ========== 1. 全量交易概览 ==========
print("\n## 1. 全量交易概览 ##")
total = conn.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL").fetchone()[0]
total_pnl = conn.execute("SELECT SUM(pnl_usdt) FROM trades WHERE exit_time IS NOT NULL").fetchone()[0] or 0
total_fees = conn.execute("SELECT SUM(fees) FROM trades WHERE exit_time IS NOT NULL").fetchone()[0] or 0
wins = conn.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL AND pnl_usdt > 0").fetchone()[0]
losses = conn.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL AND pnl_usdt <= 0").fetchone()[0]
open_positions = conn.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NULL").fetchone()[0]
first_trade = conn.execute("SELECT MIN(exit_time) FROM trades WHERE exit_time IS NOT NULL").fetchone()[0]
last_trade = conn.execute("SELECT MAX(exit_time) FROM trades WHERE exit_time IS NOT NULL").fetchone()[0]

print(f"  总交易数(已平仓): {total}")
print(f"  当前持仓中: {open_positions}")
print(f"  总盈亏: {total_pnl:.4f} USDT")
print(f"  总手续费: {total_fees:.4f} USDT")
print(f"  净收益: {(total_pnl - total_fees):.4f} USDT")
print(f"  盈利/亏损: {wins}/{losses}")
print(f"  胜率: {wins/total*100:.1f}%" if total > 0 else "  无交易数据")
print(f"  首笔交易: {first_trade}")
print(f"  末笔交易: {last_trade}")

# ========== 2. 策略表现深度分析 ==========
print("\n## 2. 策略表现深度分析 ##")
strats = conn.execute("""SELECT strategy_name, 
    COUNT(*) as cnt, 
    SUM(pnl_usdt) as pnl, 
    SUM(fees) as fees,
    SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN pnl_usdt<=0 THEN 1 ELSE 0 END) as losses,
    AVG(CASE WHEN pnl_usdt>0 THEN pnl_usdt END) as avg_win,
    AVG(CASE WHEN pnl_usdt<=0 THEN pnl_usdt END) as avg_loss,
    MAX(pnl_usdt) as max_win,
    MIN(pnl_usdt) as max_loss
    FROM trades WHERE exit_time IS NOT NULL 
    GROUP BY strategy_name ORDER BY pnl ASC""").fetchall()

for s in strats:
    wr = s['wins']/s['cnt']*100 if s['cnt']>0 else 0
    net = (s['pnl'] or 0) - (s['fees'] or 0)
    aw = s['avg_win'] or 0
    al = s['avg_loss'] or 0
    rr = abs(aw/al) if al != 0 else 999
    print(f"  [{s['strategy_name']:20s}] {s['cnt']:4d}笔 | PnL={s['pnl'] or 0:+.4f} | Fees={s['fees'] or 0:.4f} | Net={net:+.4f}")
    print(f"    胜率={wr:.1f}% | 盈亏比={rr:.2f} | 均盈={aw:.6f} | 均亏={al:.6f} | 最大盈={s['max_win']:.6f} | 最大亏={s['max_loss']:.6f}")

# ========== 3. 币种盈亏矩阵 ==========
print("\n## 3. 币种盈亏矩阵 (按净盈亏排序) ##")
symbols = conn.execute("""SELECT symbol, 
    COUNT(*) as cnt, 
    SUM(pnl_usdt) as pnl, 
    SUM(fees) as fees,
    SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) as wins,
    AVG(CASE WHEN pnl_usdt>0 THEN pnl_usdt END) as avg_win,
    AVG(CASE WHEN pnl_usdt<=0 THEN pnl_usdt END) as avg_loss
    FROM trades WHERE exit_time IS NOT NULL 
    GROUP BY symbol ORDER BY (SUM(pnl_usdt) - SUM(fees)) ASC""").fetchall()

for s in symbols:
    wr = s['wins']/s['cnt']*100 if s['cnt']>0 else 0
    net = (s['pnl'] or 0) - (s['fees'] or 0)
    aw = s['avg_win'] or 0
    al = s['avg_loss'] or 0
    rr = abs(aw/al) if al != 0 else 999
    flag = "!!! 拉黑" if net < -0.02 else ("!! 警惕" if net < -0.01 else "")
    print(f"  {s['symbol']:20s} | {s['cnt']:4d}笔 | PnL={s['pnl'] or 0:+.4f} | 净={net:+.4f} | WR={wr:5.1f}% | RR={rr:.2f} {flag}")

# ========== 4. 时段风险分析 ==========
print("\n## 4. 时段风险分析 (UTC) ##")
hours = conn.execute("""SELECT CAST(strftime('%H', exit_time) AS INTEGER) as h, 
    COUNT(*) as cnt, 
    SUM(pnl_usdt) as pnl, 
    SUM(fees) as fees,
    SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN pnl_usdt<=0 THEN 1 ELSE 0 END) as losses
    FROM trades WHERE exit_time IS NOT NULL 
    GROUP BY h ORDER BY h""").fetchall()

high_risk = {4, 8, 11, 13, 17, 20, 22}
med_risk = {0, 1, 2, 15, 16, 21}
for h in hours:
    wr = h['wins']/h['cnt']*100 if h['cnt']>0 else 0
    net = (h['pnl'] or 0) - (h['fees'] or 0)
    risk_tag = "HIGH" if h['h'] in high_risk else ("MED" if h['h'] in med_risk else "LOW")
    print(f"  UTC {h['h']:02d}:00 [{risk_tag:4s}] | {h['cnt']:4d}笔 | PnL={h['pnl'] or 0:+.4f} | 净={net:+.4f} | WR={wr:.1f}%")

# ========== 5. 盈亏分布分析 ==========
print("\n## 5. 盈亏分布分析 ##")
all_pnl = [t['pnl_usdt'] or 0 for t in conn.execute(
    "SELECT pnl_usdt FROM trades WHERE exit_time IS NOT NULL"
).fetchall()]

ranges = [
    (-1.0, "极亏 < -1.0"),
    (-0.1, "-1.0 ~ -0.1"),
    (-0.01, "-0.1 ~ -0.01"),
    (0, "-0.01 ~ 0"),
    (0.01, "0 ~ 0.01"),
    (0.1, "0.01 ~ 0.1"),
    (1.0, "0.1 ~ 1.0"),
    (float('inf'), "> 1.0"),
]
buckets = {r[1]: {'cnt':0, 'pnl':0} for r in ranges}
for pnl in all_pnl:
    for threshold, name in ranges:
        if pnl <= threshold:
            buckets[name]['cnt'] += 1
            buckets[name]['pnl'] += pnl
            break

for name, data in buckets.items():
    pct = data['cnt']/total*100 if total > 0 else 0
    bar = '#' * int(pct * 2)
    print(f"  {name:20s} | {data['cnt']:4d}笔 ({pct:5.1f}%) | PnL={data['pnl']:+.4f} {bar}")

# ========== 6. 连续亏损分析 ==========
print("\n## 6. 连续亏损分析 ##")
all_trades = conn.execute("""SELECT exit_time, pnl_usdt, symbol, strategy_name 
    FROM trades WHERE exit_time IS NOT NULL ORDER BY exit_time""").fetchall()

streak = 0; max_streak = 0; current_streak = 0
streak_details = []
for t in all_trades:
    pnl = t['pnl_usdt'] or 0
    if pnl <= 0:
        streak += 1
        if streak >= 5:
            streak_details.append((t['exit_time'], streak, t['symbol'], pnl))
    else:
        if streak > max_streak:
            max_streak = streak
        streak = 0
if streak > max_streak:
    max_streak = streak
    current_streak = streak

print(f"  历史最大连续亏损: {max_streak} 笔")
print(f"  当前连续亏损: {current_streak} 笔")
if streak_details:
    print(f"  连续亏损片段 (>=5笔):")
    for d in streak_details[-10:]:
        print(f"    {d[0]} | {d[2]} | streak={d[1]} | PnL={d[3]:+.4f}")

# ========== 7. 手续费效率分析 ==========
print("\n## 7. 手续费效率分析 (费/PnL比) ##")
fee_ratio = conn.execute("""SELECT symbol, 
    SUM(fees) as tf, 
    SUM(pnl_usdt) as tp, 
    COUNT(*) as cnt,
    SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) as wins
    FROM trades WHERE exit_time IS NOT NULL 
    GROUP BY symbol 
    HAVING tf > 0 AND tp < 0
    ORDER BY tf/ABS(tp) DESC""").fetchall()

for f in fee_ratio:
    ratio = (f['tf'] or 0) / abs(f['tp'] or 0.0001)
    wr = f['wins']/f['cnt']*100 if f['cnt']>0 else 0
    flag = "!!! 严重" if ratio > 5 else ("!! 警惕" if ratio > 2 else "")
    print(f"  {f['symbol']:20s} | 费/PnL={ratio:6.1f}x | 费={f['tf']:.4f} | PnL={f['tp']:.4f} | {f['cnt']}笔 | WR={wr:.1f}% {flag}")

# ========== 8. 退出原因分析 ==========
print("\n## 8. 退出原因分析 ##")
exits = conn.execute("""SELECT exit_reason, 
    COUNT(*) as cnt, 
    SUM(pnl_usdt) as pnl,
    SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) as wins
    FROM trades WHERE exit_time IS NOT NULL AND exit_reason IS NOT NULL
    GROUP BY exit_reason ORDER BY cnt DESC""").fetchall()

for e in exits:
    wr = e['wins']/e['cnt']*100 if e['cnt']>0 else 0
    avg_pnl = (e['pnl'] or 0) / e['cnt'] if e['cnt'] > 0 else 0
    print(f"  {e['exit_reason']:30s} | {e['cnt']:4d}笔 | PnL={e['pnl'] or 0:+.4f} | 均={avg_pnl:+.4f} | WR={wr:.1f}%")

# ========== 9. 每日盈亏趋势 ==========
print("\n## 9. 每日盈亏趋势 ##")
daily = conn.execute("""SELECT date(exit_time) as d, 
    COUNT(*) as cnt, 
    SUM(pnl_usdt) as pnl, 
    SUM(fees) as fees,
    SUM(CASE WHEN pnl_usdt>0 THEN 1 ELSE 0 END) as wins
    FROM trades WHERE exit_time IS NOT NULL 
    GROUP BY d ORDER BY d""").fetchall()

cumulative = 0
for d in daily:
    net = (d['pnl'] or 0) - (d['fees'] or 0)
    cumulative += net
    wr = d['wins']/d['cnt']*100 if d['cnt']>0 else 0
    bar = '+' * max(0, int(net * 100)) if net > 0 else '-' * max(0, int(abs(net) * 100))
    print(f"  {d['d']} | {d['cnt']:4d}笔 | PnL={d['pnl'] or 0:+.4f} | 净={net:+.4f} | 累计={cumulative:+.4f} | WR={wr:.1f}% {bar}")

# ========== 10. 策略有效性评估 ==========
print("\n## 10. 策略有效性评估 ##")
for s in strats:
    wr = s['wins']/s['cnt']*100 if s['cnt']>0 else 0
    aw = s['avg_win'] or 0
    al = s['avg_loss'] or 0
    rr = abs(aw/al) if al != 0 else 999
    net = (s['pnl'] or 0) - (s['fees'] or 0)
    
    # 评分
    score = 0
    if (s['pnl'] or 0) > 0: score += 30
    if wr > 40: score += 20
    elif wr > 30: score += 10
    if rr > 1.5: score += 20
    elif rr > 1.0: score += 10
    if s['cnt'] > 20: score += 10  # 样本充足
    if net > 0: score += 20
    
    if score >= 70: grade = "A - 优秀"
    elif score >= 50: grade = "B - 可优化"
    elif score >= 30: grade = "C - 需重构"
    else: grade = "D - 建议停用"
    
    print(f"  [{s['strategy_name']:20s}] 评分={score:2d} {grade:15s} | PnL={s['pnl'] or 0:+.4f} | WR={wr:.1f}% | RR={rr:.2f} | {s['cnt']}笔")

# ========== 11. 滑点与价差分析 ==========
print("\n## 11. 滑点与价差分析 (entry_price vs exit_price) ##")
slippage = conn.execute("""SELECT symbol, strategy_name,
    AVG(CASE WHEN ABS(exit_price - entry_price) > 0 AND entry_price > 0 
        THEN ABS(exit_price - entry_price) / entry_price * 100 ELSE NULL END) as avg_slip,
    COUNT(*) as cnt
    FROM trades WHERE exit_time IS NOT NULL AND entry_price > 0
    GROUP BY symbol, strategy_name
    ORDER BY avg_slip DESC LIMIT 15""").fetchall()

for s in slippage:
    if s['avg_slip'] is not None:
        print(f"  {s['symbol']:15s} | {s['strategy_name']:15s} | 均价差={s['avg_slip']:.4f}% | {s['cnt']}笔")

# ========== 12. 持仓时长分析 ==========
print("\n## 12. 持仓时长分析 ##")
durations = conn.execute("""SELECT 
    AVG((julianday(exit_time) - julianday(entry_time)) * 86400) as avg_sec,
    MIN((julianday(exit_time) - julianday(entry_time)) * 86400) as min_sec,
    MAX((julianday(exit_time) - julianday(entry_time)) * 86400) as max_sec,
    COUNT(*) as cnt,
    strategy_name,
    SUM(pnl_usdt) as pnl
    FROM trades WHERE exit_time IS NOT NULL AND entry_time IS NOT NULL
    GROUP BY strategy_name
    ORDER BY avg_sec""").fetchall()

for d in durations:
    avg_min = (d['avg_sec'] or 0) / 60
    print(f"  {d['strategy_name']:20s} | 均={avg_min:.1f}min | 最短={d['min_sec']:.0f}s | 最长={d['max_sec']:.0f}s | {d['cnt']}笔 | PnL={d['pnl']:+.4f}")

# ========== 13. 风险指标 ==========
print("\n## 13. 风险指标 ##")
if len(all_pnl) > 1:
    mean_pnl = statistics.mean(all_pnl)
    std_pnl = statistics.stdev(all_pnl)
    sharpe = mean_pnl / std_pnl * (252**0.5) if std_pnl > 0 else 0
    print(f"  平均每笔PnL: {mean_pnl:.6f} USDT")
    print(f"  PnL标准差: {std_pnl:.6f} USDT")
    print(f"  年化夏普比: {sharpe:.2f}")
    
    # 最大回撤
    cumulative_pnl = []
    cum = 0
    peak = 0
    max_dd = 0
    for pnl in all_pnl:
        cum += pnl
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
    print(f"  累计最大回撤: {max_dd:.4f} USDT (peak={peak:.4f})")
    
    # 盈利因子
    gross_profit = sum(p for p in all_pnl if p > 0)
    gross_loss = abs(sum(p for p in all_pnl if p < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    print(f"  盈利因子: {profit_factor:.2f}")

# ========== 14. 策略与币种交叉分析 ==========
print("\n## 14. 策略-币种交叉矩阵 (净PnL) ##")
cross = conn.execute("""SELECT strategy_name, symbol,
    COUNT(*) as cnt, SUM(pnl_usdt) as pnl, SUM(fees) as fees
    FROM trades WHERE exit_time IS NOT NULL
    GROUP BY strategy_name, symbol
    ORDER BY strategy_name, (SUM(pnl_usdt) - SUM(fees)) ASC""").fetchall()

cross_data = defaultdict(dict)
for c in cross:
    net = (c['pnl'] or 0) - (c['fees'] or 0)
    cross_data[c['strategy_name']][c['symbol']] = (c['cnt'], net)

for strat, symbols_data in cross_data.items():
    print(f"\n  [{strat}]")
    for sym, (cnt, net) in sorted(symbols_data.items(), key=lambda x: x[1][1]):
        flag = "!!! 拉黑" if net < -0.02 else ("!! 警惕" if net < -0.01 else "")
        print(f"    {sym:15s} | {cnt:3d}笔 | 净={net:+.4f} {flag}")

conn.close()

print("\n" + "=" * 70)
print("  分析完成")
print("=" * 70)