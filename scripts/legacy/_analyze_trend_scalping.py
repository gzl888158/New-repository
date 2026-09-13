# -*- coding: utf-8 -*-
"""趋势/剥头皮 逐笔复盘（基于 trade_records 表，数据更全）"""
import sqlite3
from collections import defaultdict
from datetime import datetime

DB = "data/trading.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
cur = conn.cursor()


def group_report(strategy):
    print(f"\n{'='*70}\n策略: {strategy}\n{'='*70}")
    cur.execute("""SELECT * FROM trade_records
                   WHERE strategy_name=? AND status='closed' ORDER BY create_time""", (strategy,))
    rows = cur.fetchall()
    total = len(rows)
    pnl_sum = sum(r["pnl"] or 0 for r in rows)
    wins = sum(1 for r in rows if (r["pnl"] or 0) > 0)
    losses = sum(1 for r in rows if (r["pnl"] or 0) <= 0)
    fees = sum(r["fees"] or 0 for r in rows)
    print(f"总笔数={total}  总PnL={pnl_sum:.4f} USDT  手续费={fees:.4f}  "
          f"胜={wins} 负={losses} 胜率={wins/total*100:.1f}%")

    def agg(keyfn, label):
        d = defaultdict(lambda: [0, 0.0, 0])  # count, pnl, wins
        for r in rows:
            k = keyfn(r)
            d[k][0] += 1
            d[k][1] += (r["pnl"] or 0)
            if (r["pnl"] or 0) > 0:
                d[k][2] += 1
        print(f"\n--- 按 {label} ---")
        for k, (c, pnl, w) in sorted(d.items(), key=lambda x: x[1][1]):
            print(f"  {str(k):24s} {c:4d}笔  PnL={pnl:8.4f}  胜率={w/c*100:5.1f}%")

    agg(lambda r: r["symbol"], "币种(symbol)")
    agg(lambda r: r["side"], "方向(side)")
    agg(lambda r: r["exit_reason"] or "None", "离场原因(exit_reason)")
    agg(lambda r: r["order_type"], "订单类型(order_type)")

    # 时段：按 create_time 的小时（UTC）
    def hour(r):
        ct = r["create_time"]
        if not ct:
            return "None"
        if isinstance(ct, str):
            ct = datetime.fromisoformat(ct)
        return f"UTC {ct.hour:02d}"
    agg(hour, "开仓时段(create_time hour, UTC)")

    # 持仓时长
    def hold(r):
        ct, cl = r["create_time"], r["close_time"]
        if not ct or not cl:
            return "None"
        if isinstance(ct, str):
            ct = datetime.fromisoformat(ct)
        if isinstance(cl, str):
            cl = datetime.fromisoformat(cl)
        mins = (cl - ct).total_seconds() / 60
        if mins < 5:
            return "<5min"
        if mins < 30:
            return "5-30min"
        if mins < 120:
            return "30min-2h"
        if mins < 480:
            return "2h-8h"
        return ">8h"
    agg(hold, "持仓时长")


group_report("trend")
group_report("scalping")

# 逐笔明细（亏损前20）
print("\n\n" + "="*70)
print("亏损明细 TOP（按 PnL 升序）")
print("="*70)
cur.execute("""SELECT * FROM trade_records
               WHERE strategy_name IN ('trend','scalping') AND status='closed'
               ORDER BY pnl ASC LIMIT 20""")
for r in cur.fetchall():
    print(f"  {r['strategy_name']:9s} {r['symbol']:16s} {r['side']:6s} "
          f"qty={r['quantity']:.4f} pnl={r['pnl']:.4f} exit={r['exit_reason']!r} "
          f"ct={r['create_time']}")

print("\n\n" + "="*70)
print("AVAX scalping 全部记录（含 open）")
print("="*70)
cur.execute("""SELECT id, symbol, side, order_type, quantity, price, filled_price,
               leverage, margin, pnl, pnl_percent, status, exit_reason, create_time, close_time
               FROM trade_records WHERE symbol='AVAX-USDT-SWAP' AND strategy_name='scalping'
               ORDER BY create_time""")
for r in cur.fetchall():
    print(f"  {r['side']:6s} qty={r['quantity']:.4f} px={r['price']} lev={r['leverage']} "
          f"margin={r['margin']:.4f} pnl={r['pnl']:.4f} pnl%={r['pnl_percent']} "
          f"status={r['status']!r} exit={r['exit_reason']!r} ct={r['create_time']}")

conn.close()
