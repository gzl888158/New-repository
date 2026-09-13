# -*- coding: utf-8 -*-
"""深度分析 - 交易诊断"""
import sqlite3, os, json
from datetime import datetime, timedelta
from collections import defaultdict

BASE = r'e:\新建文件夹\okx_quant_trading'
db = os.path.join(BASE, 'data', 'trading.db')
conn = sqlite3.connect(db)
c = conn.cursor()

now = datetime.now()
cutoff_24h = (now - timedelta(hours=24)).isoformat()
cutoff_6h = (now - timedelta(hours=6)).isoformat()

# ===== 1. 网格策略磨损分析 =====
print("=" * 60)
print("1. 网格策略磨损型交易分析 (PnL -0.01 ~ 0)")
print("=" * 60)
wear_trades = c.execute("""
    SELECT symbol, COUNT(*), SUM(pnl), SUM(fees) 
    FROM trades 
    WHERE exit_time IS NOT NULL 
    AND strategy_name = 'grid'
    AND pnl <= 0 AND pnl >= -0.01
    GROUP BY symbol ORDER BY COUNT(*) DESC
""").fetchall()
total_wear = 0
total_wear_pnl = 0
for w in wear_trades:
    fee = w[3] if w[3] else 0
    print(f"  {w[0]:15s} | {w[1]:3d}笔 | PnL={w[2]:+.5f} | 手续费={fee:.5f}")
    total_wear += w[1]
    total_wear_pnl += w[2]

grid_total = c.execute("SELECT COUNT(*) FROM trades WHERE exit_time IS NOT NULL AND strategy_name='grid'").fetchone()[0]
print(f"\n  磨损交易占比: {total_wear}/{grid_total} = {total_wear/grid_total*100:.1f}%")
print(f"  磨损交易总PnL: {total_wear_pnl:+.5f}")

# ===== 2. 各币种详细分析 =====
print("\n" + "=" * 60)
print("2. 各币种交易详情 (24h内)")
print("=" * 60)
sym_detail = c.execute("""
    SELECT symbol, strategy_name, COUNT(*), SUM(pnl), AVG(CASE WHEN pnl>0 THEN 1.0 ELSE 0.0 END),
           SUM(fees), AVG(entry_price), AVG(exit_price)
    FROM trades WHERE exit_time IS NOT NULL AND exit_time > ?
    GROUP BY symbol, strategy_name ORDER BY SUM(pnl) ASC
""", (cutoff_24h,)).fetchall()
for s in sym_detail:
    fee = s[5] if s[5] else 0
    pnl = s[3] if s[3] else 0
    fee_ratio = abs(fee / pnl) if pnl != 0 and abs(pnl) > 0.0001 else 0
    print(f"  {s[0]:15s} | {s[1]:10s} | {s[2]:3d}笔 | PnL={pnl:+.5f} | WR={s[4]:.1%} | 费/PnL={fee_ratio:.1f}x | 费={fee:.5f}")

# ===== 3. 未平仓交易分析 =====
print("\n" + "=" * 60)
print("3. 当前未平仓交易")
print("=" * 60)
open_trades = c.execute("""
    SELECT symbol, strategy_name, direction, entry_time, entry_price, quantity, pnl
    FROM trades WHERE exit_time IS NULL
    ORDER BY entry_time DESC
""").fetchall()
if open_trades:
    for t in open_trades:
        print(f"  {t[0]:15s} | {t[1]:10s} | {t[2]:6s} | 入场={t[4]:.6f} | 量={t[5]:.3f} | 持仓PnL={t[6]:+.5f} | {t[3][:19]}")
else:
    print("  无未平仓交易")

# ===== 4. 黑名单候选分析 =====
print("\n" + "=" * 60)
print("4. 黑名单候选 (累计亏损币种)")
print("=" * 60)
sym_cum = c.execute("""
    SELECT symbol, COUNT(*), SUM(pnl), AVG(CASE WHEN pnl>0 THEN 1.0 ELSE 0.0 END),
           SUM(CASE WHEN exit_time > ? THEN pnl ELSE 0 END) as recent_pnl
    FROM trades WHERE exit_time IS NOT NULL
    GROUP BY symbol ORDER BY SUM(pnl) ASC
""", (cutoff_24h,)).fetchall()

# 当前黑名单
state_path = os.path.join(BASE, 'data', 'intelligent_agent_state.json')
blacklist = {}
if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        state = json.load(f)
    blacklist = state.get('blacklist', {})

for s in sym_cum:
    sym = s[0]
    cnt = s[1]
    pnl = s[2] if s[2] else 0
    wr = s[3] if s[3] else 0
    recent_pnl = s[4] if s[4] else 0
    in_bl = sym in blacklist
    
    # 黑名单条件
    reasons = []
    if cnt >= 5 and pnl < -0.02 and wr < 0.2:
        reasons.append(f"低胜率+亏损: {cnt}笔, WR={wr:.0%}, PnL={pnl:+.5f}")
    if cnt >= 10 and pnl < -0.1:
        reasons.append(f"累计亏损>0.1: {cnt}笔, PnL={pnl:+.5f}")
    if cnt >= 5 and pnl < -0.02:
        # 检查费/PnL比
        fee_total = c.execute("SELECT SUM(fees) FROM trades WHERE symbol=? AND exit_time IS NOT NULL", (sym,)).fetchone()[0]
        if fee_total and abs(pnl) > 0.0001 and abs(fee_total / pnl) > 5:
            reasons.append(f"费/PnL={abs(fee_total/pnl):.1f}x")
    
    status = "BLACKLISTED" if in_bl else ""
    if reasons:
        status = "CANDIDATE" if not in_bl else status
        print(f"  {sym:15s} | {cnt:3d}笔 | PnL={pnl:+.5f} | WR={wr:.1%} | 24hPnL={recent_pnl:+.5f} | {status} | {'; '.join(reasons)}")

# ===== 5. 时段分析 =====
print("\n" + "=" * 60)
print("5. 时段风险分析 (24h)")
print("=" * 60)
hour_stats = c.execute("""
    SELECT CAST(strftime('%H', exit_time) AS INTEGER) as hour, 
           COUNT(*), SUM(pnl), AVG(CASE WHEN pnl>0 THEN 1.0 ELSE 0.0 END)
    FROM trades WHERE exit_time IS NOT NULL AND exit_time > ?
    GROUP BY hour ORDER BY hour
""", (cutoff_24h,)).fetchall()
for h in hour_stats:
    risk = "HIGH" if h[0] in [0,4,8,11,13,17,20,22] else ("MED" if h[0] in [1,2,15,16,21] else "LOW")
    print(f"  UTC {h[0]:02d}:00 | {h[1]:3d}笔 | PnL={h[2]:+.5f} | WR={h[3]:.1%} | {risk}")

# ===== 6. 网格策略统计 =====
print("\n" + "=" * 60)
print("6. 网格策略交易方向统计")
print("=" * 60)
dir_stats = c.execute("""
    SELECT direction, COUNT(*), SUM(pnl), AVG(CASE WHEN pnl>0 THEN 1.0 ELSE 0.0 END)
    FROM trades WHERE exit_time IS NOT NULL AND strategy_name='grid'
    GROUP BY direction
""").fetchall()
for d in dir_stats:
    print(f"  {d[0]:6s} | {d[1]:3d}笔 | PnL={d[2]:+.5f} | WR={d[3]:.1%}")

conn.close()