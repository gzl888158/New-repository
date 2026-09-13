#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OKX 量化交易系统 - 历史数据分析报告
"""

import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import math
import sqlite3
from datetime import datetime

# ============================================================
# 配置
# ============================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "trading.db")
STRATEGY_STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "strategy_state")

# ============================================================
# 1. 数据库查询
# ============================================================
def query_database():
    """查询 trading.db 中的所有交易记录"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 查询所有已平仓的交易记录
    cursor.execute("""
        SELECT * FROM trade_records WHERE status = 'closed'
        ORDER BY create_time ASC
    """)
    closed_trades = [dict(row) for row in cursor.fetchall()]

    # 查询所有开仓中的记录
    cursor.execute("""
        SELECT * FROM trade_records WHERE status = 'open'
        ORDER BY create_time ASC
    """)
    open_trades = [dict(row) for row in cursor.fetchall()]

    # 查询账户历史
    cursor.execute("""
        SELECT * FROM account_history ORDER BY timestamp ASC
    """)
    account_history = [dict(row) for row in cursor.fetchall()]

    # 查询策略绩效表
    cursor.execute("""
        SELECT * FROM strategy_performance ORDER BY strategy_name, symbol
    """)
    strategy_perf = [dict(row) for row in cursor.fetchall()]

    conn.close()

    return {
        "closed_trades": closed_trades,
        "open_trades": open_trades,
        "account_history": account_history,
        "strategy_performance": strategy_perf,
    }


# ============================================================
# 2. 策略状态文件读取
# ============================================================
def read_strategy_states():
    """读取 strategy_state/ 目录下的所有 JSON 状态文件"""
    states = {}
    for fname in os.listdir(STRATEGY_STATE_DIR):
        if fname.endswith(".json"):
            fpath = os.path.join(STRATEGY_STATE_DIR, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                try:
                    states[fname.replace(".json", "")] = json.load(f)
                except json.JSONDecodeError:
                    states[fname.replace(".json", "")] = {"error": "JSON decode failed"}
    return states


# ============================================================
# 3. 分析函数
# ============================================================
def calc_sharpe_ratio(pnl_list, risk_free_rate=0.02, periods_per_year=365):
    """计算夏普比率"""
    if len(pnl_list) < 2:
        return 0.0
    mean_return = sum(pnl_list) / len(pnl_list)
    variance = sum((r - mean_return) ** 2 for r in pnl_list) / (len(pnl_list) - 1)
    std_dev = math.sqrt(variance) if variance > 0 else 0.0000001
    # 年化
    annualized_return = mean_return * periods_per_year
    annualized_vol = std_dev * math.sqrt(periods_per_year)
    if annualized_vol == 0:
        return 0.0
    return (annualized_return - risk_free_rate) / annualized_vol


def calc_max_drawdown(equity_curve):
    """计算最大回撤"""
    if not equity_curve:
        return 0.0, None, None
    peak = equity_curve[0]
    max_dd = 0.0
    peak_time = None
    trough_time = None
    for i, val in enumerate(equity_curve):
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return max_dd


# ============================================================
# 4. 生成报告
# ============================================================
def generate_report(db_data, strategy_states):
    closed = db_data["closed_trades"]
    open_trades = db_data["open_trades"]
    account_hist = db_data["account_history"]
    strategy_perf = db_data["strategy_performance"]

    print("=" * 80)
    print("   OKX 量化交易系统 - 历史交易数据分析报告")
    print(f"   生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # -------- 数据概览 --------
    print("\n" + "─" * 80)
    print("【一、数据概览】")
    print(f"  已平仓交易笔数: {len(closed)}")
    print(f"  当前持仓笔数:   {len(open_trades)}")
    print(f"  账户历史记录数: {len(account_hist)}")
    print(f"  策略绩效记录数: {len(strategy_perf)}")

    if not closed:
        print("\n  ⚠ 没有已平仓交易记录，无法进行深入分析。")
        # 仍然输出策略状态
        report_strategy_states(strategy_states)
        return

    # -------- 整体盈亏 --------
    total_pnl = sum(t.get("pnl", 0) or 0 for t in closed)
    winning = [t for t in closed if (t.get("pnl") or 0) > 0]
    losing = [t for t in closed if (t.get("pnl") or 0) <= 0]
    total_win_pnl = sum(t["pnl"] for t in winning)
    total_loss_pnl = sum(t["pnl"] for t in losing)

    print("\n" + "─" * 80)
    print("【二、整体业绩指标】")
    print(f"  总盈亏 (PnL):       {total_pnl:+.2f} USDT")
    print(f"  盈利交易笔数:       {len(winning)}")
    print(f"  亏损交易笔数:       {len(losing)}")
    print(f"  总体胜率:           {len(winning)/len(closed)*100:.1f}%")
    print(f"  总盈利金额:         {total_win_pnl:+.2f} USDT")
    print(f"  总亏损金额:         {total_loss_pnl:.2f} USDT")
    profit_factor = abs(total_win_pnl / total_loss_pnl) if total_loss_pnl != 0 else float('inf')
    print(f"  盈亏比 (Profit Factor): {profit_factor:.2f}")
    avg_win = total_win_pnl / len(winning) if winning else 0
    avg_loss = total_loss_pnl / len(losing) if losing else 0
    print(f"  平均盈利:           {avg_win:+.4f} USDT")
    print(f"  平均亏损:           {avg_loss:.4f} USDT")
    print(f"  平均每笔盈亏:       {total_pnl/len(closed):+.4f} USDT")

    # 夏普比率
    daily_pnl = [t.get("pnl", 0) or 0 for t in closed]
    sharpe = calc_sharpe_ratio(daily_pnl)
    print(f"  夏普比率 (近似):    {sharpe:.2f}")

    # 最大回撤 - 用账户历史计算
    if account_hist:
        equity_curve = [h.get("total_equity", 0) or 0 for h in account_hist]
        max_dd = calc_max_drawdown(equity_curve)
        print(f"  最大回撤 (账户):    {max_dd*100:.2f}%")
    else:
        # 用累计 pnl 近似
        cumulative = 0
        equity_curve = []
        for t in closed:
            cumulative += (t.get("pnl") or 0)
            equity_curve.append(cumulative)
        max_dd = calc_max_drawdown(equity_curve)
        print(f"  最大回撤 (PnL累计): {max_dd*100:.2f}%")

    # -------- 按Symbol分组 --------
    print("\n" + "─" * 80)
    print("【三、按交易品种 (Symbol) 分组统计】")
    print(f"  {'Symbol':<22} {'笔数':>5} {'胜率':>7} {'总盈亏':>12} {'平均盈亏':>12}")
    print("  " + "-" * 60)

    symbol_stats = {}
    for t in closed:
        sym = t.get("symbol", "Unknown")
        if sym not in symbol_stats:
            symbol_stats[sym] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        symbol_stats[sym]["count"] += 1
        pnl = t.get("pnl") or 0
        if pnl > 0:
            symbol_stats[sym]["wins"] += 1
        symbol_stats[sym]["total_pnl"] += pnl

    for sym in sorted(symbol_stats.keys(), key=lambda s: symbol_stats[s]["total_pnl"], reverse=True):
        s = symbol_stats[sym]
        wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
        avg = s["total_pnl"] / s["count"] if s["count"] > 0 else 0
        print(f"  {sym:<22} {s['count']:>5} {wr:>6.1f}% {s['total_pnl']:>+11.2f} {avg:>+11.4f}")

    # -------- 按策略分组 --------
    print("\n" + "─" * 80)
    print("【四、按策略类型分组统计】")
    print(f"  {'策略名称':<22} {'笔数':>5} {'胜率':>7} {'总盈亏':>12} {'平均盈亏':>12}")
    print("  " + "-" * 60)

    strategy_stats = {}
    for t in closed:
        st = t.get("strategy_name", "Unknown")
        if st not in strategy_stats:
            strategy_stats[st] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        strategy_stats[st]["count"] += 1
        pnl = t.get("pnl") or 0
        if pnl > 0:
            strategy_stats[st]["wins"] += 1
        strategy_stats[st]["total_pnl"] += pnl

    for st in sorted(strategy_stats.keys(), key=lambda s: strategy_stats[s]["total_pnl"], reverse=True):
        s = strategy_stats[st]
        wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
        avg = s["total_pnl"] / s["count"] if s["count"] > 0 else 0
        print(f"  {st:<22} {s['count']:>5} {wr:>6.1f}% {s['total_pnl']:>+11.2f} {avg:>+11.4f}")

    # -------- Top 10 盈利 --------
    print("\n" + "─" * 80)
    print("【五、盈利最多的前 10 笔交易】")
    top_winners = sorted(closed, key=lambda t: t.get("pnl") or 0, reverse=True)[:10]
    print(f"  {'时间':<22} {'Symbol':<22} {'策略':<18} {'方向':>5} {'PnL':>12} {'出场原因':>14}")
    print("  " + "-" * 95)
    for t in top_winners:
        ct = str(t.get("create_time", ""))[:19]
        er = t.get("exit_reason") or ""
        print(f"  {ct:<22} {t.get('symbol',''):<22} {t.get('strategy_name',''):<18} "
              f"{t.get('side',''):>5} {t.get('pnl',0):>+11.4f} {str(er):<14}")

    # -------- Top 10 亏损 --------
    print("\n" + "─" * 80)
    print("【六、亏损最多的前 10 笔交易】")
    top_losers = sorted(closed, key=lambda t: t.get("pnl") or 0)[:10]
    print(f"  {'时间':<22} {'Symbol':<22} {'策略':<18} {'方向':>5} {'PnL':>12} {'出场原因':>14}")
    print("  " + "-" * 95)
    for t in top_losers:
        ct = str(t.get("create_time", ""))[:19]
        er = t.get("exit_reason") or ""
        print(f"  {ct:<22} {t.get('symbol',''):<22} {t.get('strategy_name',''):<18} "
              f"{t.get('side',''):>5} {t.get('pnl',0):>+11.4f} {str(er):<14}")

    # -------- 出场原因分析 --------
    print("\n" + "─" * 80)
    print("【七、出场原因分析】")
    exit_reasons = {}
    for t in closed:
        reason = t.get("exit_reason") or "unknown"
        pnl = t.get("pnl") or 0
        if reason not in exit_reasons:
            exit_reasons[reason] = {"count": 0, "wins": 0, "total_pnl": 0.0}
        exit_reasons[reason]["count"] += 1
        if pnl > 0:
            exit_reasons[reason]["wins"] += 1
        exit_reasons[reason]["total_pnl"] += pnl

    print(f"  {'出场原因':<20} {'笔数':>5} {'胜率':>7} {'总盈亏':>12} {'平均盈亏':>12}")
    print("  " + "-" * 58)
    for reason in sorted(exit_reasons.keys(), key=lambda r: exit_reasons[r]["total_pnl"], reverse=True):
        s = exit_reasons[reason]
        wr = s["wins"] / s["count"] * 100 if s["count"] > 0 else 0
        avg = s["total_pnl"] / s["count"] if s["count"] > 0 else 0
        print(f"  {reason:<20} {s['count']:>5} {wr:>6.1f}% {s['total_pnl']:>+11.2f} {avg:>+11.4f}")

    # -------- 成功/失败案例特征 --------
    print("\n" + "─" * 80)
    print("【八、交易特征分析】")

    # 盈利交易的共同特征
    if winning:
        win_symbols = {}
        for t in winning:
            sym = t.get("symbol", "")
            win_symbols[sym] = win_symbols.get(sym, 0) + 1
        print("\n  ▶ 盈利交易共同特征:")
        print(f"    - 盈利交易占总交易的 {len(winning)/len(closed)*100:.1f}%")
        print(f"    - 平均盈利额: {avg_win:+.4f} USDT")

        # 按方向统计
        long_wins = sum(1 for t in winning if t.get("side") == "long")
        short_wins = sum(1 for t in winning if t.get("side") == "short")
        long_total = sum(1 for t in closed if t.get("side") == "long")
        short_total = sum(1 for t in closed if t.get("side") == "short")
        long_wr = long_wins / long_total * 100 if long_total > 0 else 0
        short_wr = short_wins / short_total * 100 if short_total > 0 else 0
        print(f"    - 做多胜率: {long_wr:.1f}% ({long_wins}/{long_total})")
        print(f"    - 做空胜率: {short_wr:.1f}% ({short_wins}/{short_total})")

        # 盈利最多的品种
        top_win_sym = sorted(win_symbols.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"    - 盈利最集中的品种: {', '.join(f'{s}({c}笔)' for s, c in top_win_sym)}")

        # 按杠杆分析
        leverage_win = {}
        leverage_total = {}
        for t in closed:
            lev = t.get("leverage", 1)
            leverage_total[lev] = leverage_total.get(lev, 0) + 1
            if (t.get("pnl") or 0) > 0:
                leverage_win[lev] = leverage_win.get(lev, 0) + 1
        print(f"    - 各杠杆胜率: ", end="")
        lev_parts = []
        for lev in sorted(leverage_total.keys()):
            wr = leverage_win.get(lev, 0) / leverage_total[lev] * 100
            lev_parts.append(f"{lev}x={wr:.0f}%")
        print(", ".join(lev_parts))

    # 亏损交易的共同特征
    if losing:
        print("\n  ▶ 亏损交易共同特征:")
        print(f"    - 亏损交易占总交易的 {len(losing)/len(closed)*100:.1f}%")
        print(f"    - 平均亏损额: {avg_loss:.4f} USDT")

        lose_symbols = {}
        for t in losing:
            sym = t.get("symbol", "")
            lose_symbols[sym] = lose_symbols.get(sym, 0) + 1
        top_lose_sym = sorted(lose_symbols.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"    - 亏损最集中的品种: {', '.join(f'{s}({c}笔)' for s, c in top_lose_sym)}")

        # 出场原因
        lose_reasons = {}
        for t in losing:
            reason = t.get("exit_reason") or "unknown"
            lose_reasons[reason] = lose_reasons.get(reason, 0) + 1
        top_lose_reasons = sorted(lose_reasons.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"    - 亏损出场原因分布: {', '.join(f'{r}({c}笔)' for r, c in top_lose_reasons)}")

    # -------- 策略状态汇总 --------
    print("\n" + "─" * 80)
    print("【九、各策略当前状态汇总】")
    report_strategy_states(strategy_states)

    # -------- 策略绩效表 (来自 DB) --------
    if strategy_perf:
        print("\n" + "─" * 80)
        print("【十、策略绩效汇总 (来自 strategy_performance 表)】")
        print(f"  {'策略':<20} {'Symbol':<22} {'总笔数':>6} {'胜率':>7} {'总PnL':>10} {'最大回撤':>9} {'盈亏比':>8}")
        print("  " + "-" * 80)
        for p in strategy_perf:
            print(f"  {p.get('strategy_name',''):<20} {p.get('symbol',''):<22} "
                  f"{p.get('total_trades',0):>6} {p.get('win_rate',0)*100:>6.1f}% "
                  f"{p.get('total_pnl',0):>+9.2f} {p.get('max_drawdown',0)*100:>8.2f}% "
                  f"{p.get('profit_factor',0):>7.2f}")

    # -------- 结论与建议 --------
    print("\n" + "=" * 80)
    print("【十一、综合结论与建议】")
    print("=" * 80)

    # 最佳策略
    if strategy_stats:
        best_strategy = max(strategy_stats.items(), key=lambda x: x[1]["total_pnl"])
        best_wr_strategy = max(
            [(k, v["wins"]/v["count"]) for k, v in strategy_stats.items() if v["count"] > 0],
            key=lambda x: x[1], default=("N/A", 0)
        )
        print(f"  • 盈利最佳策略: {best_strategy[0]} (总PnL: {best_strategy[1]['total_pnl']:+.2f} USDT)")
        print(f"  • 胜率最高策略: {best_wr_strategy[0]} (胜率: {best_wr_strategy[1]*100:.1f}%)")

    # 最佳品种
    if symbol_stats:
        best_sym = max(symbol_stats.items(), key=lambda x: x[1]["total_pnl"])
        print(f"  • 盈利最佳品种: {best_sym[0]} (总PnL: {best_sym[1]['total_pnl']:+.2f} USDT)")

    # 风险提示
    if max_dd > 0.2:
        print(f"  ⚠ 最大回撤 {max_dd*100:.1f}% 较高，建议加强风控")
    if profit_factor < 1.0:
        print(f"  ⚠ 盈亏比 {profit_factor:.2f} < 1，整体策略可能需要优化")
    elif profit_factor < 1.5:
        print(f"  • 盈亏比 {profit_factor:.2f}，策略有盈利空间但需持续监控")

    # 策略状态总结
    active_positions_count = 0
    for sname, sdata in strategy_states.items():
        meta = sdata.get("_meta", {})
        strategy = meta.get("strategy", sname)

        # scalping/trend 有 positions
        if "positions" in sdata:
            for pos_name, pos in sdata["positions"].items():
                if isinstance(pos, dict) and pos.get("status") == "open":
                    active_positions_count += 1

        # spot_grid 有 active_positions
        if "active_positions" in sdata:
            active_positions_count += len(sdata["active_positions"])

        # spot_martingale 有 active_positions
        if "active_positions" in sdata and isinstance(sdata.get("active_positions"), dict):
            active_positions_count += len(sdata["active_positions"])

    print(f"  • 当前活跃持仓总数: {active_positions_count}")

    print("\n" + "=" * 80)
    print("  报告结束")
    print("=" * 80)


def report_strategy_states(states):
    """输出策略状态汇总"""
    for sname in sorted(states.keys()):
        sdata = states[sname]
        if "error" in sdata:
            print(f"\n  [{sname}] 读取失败: {sdata['error']}")
            continue

        meta = sdata.get("_meta", {})
        strategy = meta.get("strategy", sname)
        saved_at = meta.get("saved_at", "N/A")
        print(f"\n  ▸ {strategy} (状态文件: {sname}.json, 保存时间: {saved_at})")

        # ---- Scalping ----
        if "positions" in sdata and "daily_pnl" in sdata:
            positions = sdata["positions"]
            open_pos = {k: v for k, v in positions.items() if isinstance(v, dict) and v.get("status") == "open"}
            closed_pos = {k: v for k, v in positions.items() if isinstance(v, dict) and v.get("status") == "closed"}
            print(f"    日PnL: {sdata.get('daily_pnl',0):+.4f} USDT")
            print(f"    日初始权益: {sdata.get('daily_start_equity',0):.2f} USDT")
            print(f"    当日收益率: {sdata.get('daily_pnl',0)/sdata.get('daily_start_equity',1)*100:.2f}%")
            print(f"    开仓: {len(open_pos)} 笔")
            for pos_name, pos in list(open_pos.items())[:6]:
                print(f"      {pos_name}: {pos.get('direction','')} @{pos.get('entry_price')} "
                      f"x{pos.get('leverage','?')} 盈亏={pos.get('current_profit',0):+.4f}")
            if len(open_pos) > 6:
                print(f"      ... 还有 {len(open_pos)-6} 个开仓")
            if closed_pos:
                print(f"    已平仓: {len(closed_pos)} 笔")
                closed_pnl = sum(p.get("pnl_usdt", 0) for p in closed_pos.values())
                print(f"    已平仓PnL: {closed_pnl:+.4f} USDT")

            # 信号类型表现
            sig_perf = sdata.get("signal_type_performance", {})
            if sig_perf:
                for sig_type, perf in sig_perf.items():
                    total = perf.get("wins", 0) + perf.get("losses", 0)
                    if total > 0:
                        wr = perf["wins"] / total * 100
                        print(f"    信号[{sig_type}]: {perf['wins']}W/{perf['losses']}L "
                              f"胜率={wr:.0f}% PnL={perf.get('total_pnl',0):+.4f}")

        # ---- Arbitrage ----
        elif "arbitrage_performance" in sdata:
            arb_perf = sdata["arbitrage_performance"]
            for arb_type, perf in arb_perf.items():
                total = perf.get("count", perf.get("wins", 0) + perf.get("losses", 0))
                if total > 0:
                    wr = perf["wins"] / total * 100 if total > 0 else 0
                    print(f"    {arb_type}: {perf['wins']}W/{perf['losses']}L "
                          f"胜率={wr:.0f}% PnL={perf.get('total_pnl',0):+.4f}")
            positions = sdata.get("positions", {})
            open_pos = {k: v for k, v in positions.items() if isinstance(v, dict) and v.get("status") == "open"}
            print(f"    当前套利对: {len(open_pos)} 个")
            for pair, pos in list(open_pos.items())[:4]:
                print(f"      {pair}: z-score={pos.get('z_score','?'):.2f} "
                      f"type={pos.get('arbitrage_type','?')}")

        # ---- Trend ----
        elif "_position_state" in sdata:
            pos_state = sdata["_position_state"]
            open_positions = []
            for k, v in pos_state.items():
                if isinstance(v, dict):
                    open_positions.append((k, v))
            print(f"    持仓: {len(open_positions)} 个")
            for name, pos in open_positions[:8]:
                profit = pos.get("current_profit", 0)
                status = pos.get("status", "?")
                atr = pos.get("atr", 0)
                print(f"      {name}: {pos.get('direction','?')} @{pos.get('entry_price')} "
                      f"PnL%={profit*100:+.2f}% 状态={status} ATR={atr:.2f}")
            if len(open_positions) > 8:
                print(f"      ... 还有 {len(open_positions)-8} 个持仓")

        # ---- Grid ----
        elif "_grids" in sdata:
            grids = sdata["_grids"]
            print(f"    网格品种: {len(grids)} 个")
            for sym, grid in list(grids.items())[:5]:
                filled = sum(1 for g in grid if g.get("filled") in (True, "pending"))
                print(f"      {sym}: {len(grid)}层, 已成交{pending}={filled}")
            if len(grids) > 5:
                print(f"      ... 还有 {len(grids)-5} 个品种")

        # ---- Spot Grid ----
        elif "grids" in sdata:
            grids = sdata["grids"]
            active = sdata.get("active_positions", {})
            print(f"    网格品种: {len(grids)} 个, 活跃持仓: {len(active)} 个")
            for sym, pos in active.items():
                print(f"      {sym}: 均价={pos.get('avg_price')} 数量={pos.get('quantity'):.2e}"
                      f" 开仓时间={pos.get('start_time','?')[:10]}")

        # ---- Spot Martingale ----
        elif "active_positions" in sdata:
            active = sdata["active_positions"]
            if isinstance(active, dict):
                print(f"    活跃马丁格尔: {len(active)} 个")
                for sym, pos in list(active.items())[:5]:
                    print(f"      {sym}: {pos.get('current_layers')}层 "
                          f"均价={pos.get('avg_entry_price')} 成本={pos.get('total_cost_usdt',0):+.2f} USDT")
                if len(active) > 5:
                    print(f"      ... 还有 {len(active)-5} 个")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("正在查询数据库...")
    db_data = query_database()
    print("正在读取策略状态文件...")
    states = read_strategy_states()
    print("正在生成报告...\n")
    generate_report(db_data, states)
