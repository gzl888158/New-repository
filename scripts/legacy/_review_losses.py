# -*- coding: utf-8 -*-
"""趋势/剥头皮 亏损复盘：定位亏损集中在哪个信号类型/时段/币种/方向/离场原因"""
import sqlite3
from collections import defaultdict

DB = r"e:\新建文件夹\okx_quant_trading\data\trading.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row
c = conn.cursor()

for strat in ("trend", "scalping"):
    print("=" * 78)
    print(f"策略: {strat}")
    print("=" * 78)

    # 1. 总体
    row = c.execute(
        "SELECT COUNT(*), SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), "
        "SUM(pnl_usdt), AVG(pnl_usdt), SUM(fees), AVG(fees) "
        "FROM trades WHERE exit_time IS NOT NULL AND strategy_name=?",
        (strat,),
    ).fetchone()
    cnt = row[0] or 0
    wins = row[1] or 0
    print(f"[总体] {cnt}笔 | 胜率 {wins/max(cnt,1)*100:.1f}% | "
          f"PnL {row[2] or 0:+.4f} | 均PnL {row[3] or 0:+.4f} | "
          f"总费 {row[4] or 0:.4f} | 均费 {row[5] or 0:.4f}")

    # 2. 按方向
    print("\n[按方向]")
    for r in c.execute(
        "SELECT direction, COUNT(*), SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), "
        "SUM(pnl_usdt), AVG(pnl_usdt) FROM trades "
        "WHERE exit_time IS NOT NULL AND strategy_name=? "
        "GROUP BY direction ORDER BY SUM(pnl_usdt) ASC", (strat,)):
        n = r[1] or 0
        print(f"  {r[0]:8s} {n:4d}笔 胜率{r[2]/max(n,1)*100:5.1f}% PnL{r[3] or 0:+8.4f} 均PnL{r[4] or 0:+8.4f}")

    # 3. 按离场原因
    print("\n[按离场原因]")
    for r in c.execute(
        "SELECT exit_reason, COUNT(*), SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), "
        "SUM(pnl_usdt), AVG(pnl_usdt) FROM trades "
        "WHERE exit_time IS NOT NULL AND strategy_name=? "
        "GROUP BY exit_reason ORDER BY COUNT(*) DESC", (strat,)):
        n = r[1] or 0
        print(f"  {str(r[0]):24s} {n:4d}笔 胜率{r[2]/max(n,1)*100:5.1f}% PnL{r[3] or 0:+8.4f} 均PnL{r[4] or 0:+8.4f}")

    # 4. 按信号类型
    print("\n[按入场信号类型]")
    for r in c.execute(
        "SELECT entry_signal_type, COUNT(*), SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), "
        "SUM(pnl_usdt), AVG(pnl_usdt) FROM trades "
        "WHERE exit_time IS NOT NULL AND strategy_name=? "
        "GROUP BY entry_signal_type ORDER BY COUNT(*) DESC", (strat,)):
        n = r[1] or 0
        print(f"  {str(r[0]):24s} {n:4d}笔 胜率{r[2]/max(n,1)*100:5.1f}% PnL{r[3] or 0:+8.4f} 均PnL{r[4] or 0:+8.4f}")

    # 5. 按币种（亏损TOP）
    print("\n[按币种 - 亏损TOP12]")
    for r in c.execute(
        "SELECT symbol, COUNT(*), SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), "
        "SUM(pnl_usdt), AVG(pnl_usdt) FROM trades "
        "WHERE exit_time IS NOT NULL AND strategy_name=? "
        "GROUP BY symbol ORDER BY SUM(pnl_usdt) ASC LIMIT 12", (strat,)):
        n = r[1] or 0
        print(f"  {r[0]:20s} {n:4d}笔 胜率{r[2]/max(n,1)*100:5.1f}% PnL{r[3] or 0:+8.4f} 均PnL{r[4] or 0:+8.4f}")

    # 6. 按UTC小时
    print("\n[按UTC小时]")
    for r in c.execute(
        "SELECT CAST(substr(exit_time,12,2) AS INTEGER), COUNT(*), "
        "SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(pnl_usdt) FROM trades "
        "WHERE exit_time IS NOT NULL AND strategy_name=? "
        "GROUP BY CAST(substr(exit_time,12,2) AS INTEGER) ORDER BY 1", (strat,)):
        n = r[1] or 0
        print(f"  UTC {r[0]:02d}:00 {n:4d}笔 胜率{r[2]/max(n,1)*100:5.1f}% PnL{r[3] or 0:+8.4f}")

    # 7. 入场→出场价差 vs 方向（判断是否方向系统性错误）
    print("\n[价差方向一致性 - 最近40笔亏损]")
    for r in c.execute(
        "SELECT symbol, direction, entry_price, exit_price, pnl_usdt, exit_reason, entry_signal_type, entry_time "
        "FROM trades WHERE exit_time IS NOT NULL AND strategy_name=? AND win=0 "
        "ORDER BY exit_time DESC LIMIT 40", (strat,)):
        chg = (r[3] - r[2]) / r[2] if r[2] else 0
        # long 亏损 => 价格下跌; short 亏损 => 价格上涨
        print(f"  {r[0]:18s} {r[1]:6s} {r[7]:19s} 入{r[2]:.6f} 出{r[3]:.6f} 变动{chg:+.4%} "
              f"PnL{r[4]:+.4f} {str(r[5]):16s} {str(r[6]):16s}")

    print()

conn.close()
