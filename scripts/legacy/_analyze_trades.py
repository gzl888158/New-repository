"""
深度交易历史分析脚本
分析 stop_loss_audit, fill_quality, equity_curve_backup 等表
"""
import sqlite3
import json
from collections import defaultdict, Counter
from datetime import datetime

DB_PATH = r"e:\新建文件夹\okx_quant_trading\data\trading.db"

def analyze_stop_loss_audit(conn):
    """分析止损记录 - 找出亏损模式"""
    print("\n" + "="*60)
    print("=== 止损审计分析 ===")
    
    # 按策略统计
    rows = conn.execute("""
        SELECT strategy_name, COUNT(*) as cnt, 
               SUM(pnl) as total_pnl, AVG(pnl) as avg_pnl,
               AVG(pnl_percent) as avg_pnl_pct, 
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END) as losses
        FROM stop_loss_audit 
        GROUP BY strategy_name
        ORDER BY total_pnl ASC
    """).fetchall()
    
    print(f"{'策略':<15} {'笔数':>6} {'总PnL':>10} {'均PnL':>10} {'均PnL%':>10} {'胜':>5} {'负':>5} {'胜率':>7}")
    print("-"*70)
    for r in rows:
        strategy, cnt, total, avg, avg_pct, wins, losses = r
        wr = wins/(wins+losses)*100 if (wins+losses) > 0 else 0
        print(f"{strategy:<15} {cnt:>6} {total:>10.4f} {avg:>10.4f} {avg_pct:>10.4%} {wins:>5} {losses:>5} {wr:>6.1f}%")
    
    # 按币种统计
    print("\n--- 按币种统计 ---")
    rows = conn.execute("""
        SELECT symbol, COUNT(*) as cnt, 
               SUM(pnl) as total_pnl, AVG(pnl) as avg_pnl,
               AVG(pnl_percent) as avg_pnl_pct,
               AVG(execution_latency_ms) as avg_latency,
               AVG(slippage_pct) as avg_slippage
        FROM stop_loss_audit 
        GROUP BY symbol
        ORDER BY total_pnl ASC
        LIMIT 20
    """).fetchall()
    
    for r in rows:
        sym, cnt, total, avg, avg_pct, avg_lat, avg_slip = r
        print(f"  {sym:<20} cnt={cnt:>5} totalPnL={total:>8.4f} avgPnL={avg:>8.4f} avgPnL%={avg_pct:>8.4%} latency={avg_lat:>6.1f}ms slip={avg_slip:>6.4%}")
    
    # 按触发类型统计
    print("\n--- 按触发类型统计 ---")
    rows = conn.execute("""
        SELECT trigger_type, exit_reason, COUNT(*) as cnt,
               SUM(pnl) as total_pnl, AVG(pnl_percent) as avg_pnl_pct
        FROM stop_loss_audit 
        GROUP BY trigger_type, exit_reason
        ORDER BY cnt DESC
    """).fetchall()
    for r in rows:
        print(f"  {r[0]:<15} {r[1]:<15} cnt={r[2]:>5} totalPnL={r[3]:>8.4f} avgPnL%={r[4]:>8.4%}")

def analyze_fill_quality(conn):
    """分析成交质量"""
    print("\n" + "="*60)
    print("=== 成交质量分析 ===")
    
    # 按策略统计滑点
    rows = conn.execute("""
        SELECT strategy_name, COUNT(*) as cnt,
               AVG(abs_slippage) as avg_slip,
               AVG(direction_adjusted_slippage) as avg_dir_slip,
               SUM(CASE WHEN exceeds_tolerance THEN 1 ELSE 0 END) as exceeds
        FROM fill_quality
        GROUP BY strategy_name
        ORDER BY cnt DESC
    """).fetchall()
    
    print(f"{'策略':<15} {'笔数':>6} {'均滑点':>10} {'方向滑点':>10} {'超限':>5}")
    print("-"*55)
    for r in rows:
        print(f"{r[0]:<15} {r[1]:>6} {r[2]:>10.6f} {r[3]:>10.6f} {r[4]:>5}")
    
    # 按币种统计
    rows = conn.execute("""
        SELECT symbol, COUNT(*) as cnt,
               AVG(abs_slippage) as avg_slip,
               AVG(direction_adjusted_slippage) as avg_dir_slip
        FROM fill_quality
        GROUP BY symbol
        ORDER BY avg_dir_slip ASC
        LIMIT 15
    """).fetchall()
    
    print("\n--- 按币种滑点(正=有利，负=不利) ---")
    for r in rows:
        dir_slip = r[3] if r[3] else 0
        tag = "✓" if dir_slip > 0 else "✗" if dir_slip < -0.001 else " "
        print(f"  {tag} {r[0]:<20} cnt={r[1]:>5} avg_slip={r[2]:>8.6f} dir_slip={dir_slip:>8.6f}")

def analyze_equity_curve(conn):
    """分析权益曲线"""
    print("\n" + "="*60)
    print("=== 权益曲线分析 ===")
    
    rows = conn.execute("""
        SELECT timestamp, total_equity, available_balance, 
               used_margin, unrealized_pnl, realized_pnl,
               win_rate, total_trades
        FROM equity_curve_backup
        ORDER BY timestamp
    """).fetchall()
    
    if len(rows) < 2:
        print("  equity_curve_backup 数据不足")
        return
    
    # 找出关键时间点
    first = rows[0]
    last = rows[-1]
    print(f"  起始: {first[0]} equity={first[1]:.2f} margin={first[3]:.2f}")
    print(f"  最新: {last[0]} equity={last[1]:.2f} margin={last[3]:.2f}")
    
    # 计算最大回撤
    peak = first[1]
    max_dd = 0
    max_dd_start = max_dd_end = first[0]
    for r in rows:
        eq = r[1]
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
            max_dd_end = r[0]
    print(f"  最大回撤: {max_dd:.2%}")
    
    # 采样显示
    step = max(1, len(rows) // 10)
    print("\n  采样:")
    for i in range(0, len(rows), step):
        r = rows[i]
        print(f"    {r[0][:19]} equity={r[1]:.2f} margin={r[3]:.2f} upnl={r[4]:.2f} rpnl={r[5]:.2f} wr={r[6]:.1%}")

def analyze_trade_patterns(conn):
    """分析交易模式 - 找出规律"""
    print("\n" + "="*60)
    print("=== 交易模式分析 ===")
    
    # 按小时统计
    rows = conn.execute("""
        SELECT CAST(substr(timestamp, 12, 2) AS INTEGER) as hour,
               COUNT(*) as cnt,
               SUM(pnl) as total_pnl,
               AVG(pnl) as avg_pnl
        FROM stop_loss_audit
        GROUP BY hour
        ORDER BY hour
    """).fetchall()
    
    print("\n--- 按小时统计 ---")
    print(f"{'小时':>6} {'笔数':>6} {'总PnL':>10} {'均PnL':>10}")
    for r in rows:
        print(f"  {r[0]:>4}时 {r[1]:>6} {r[2]:>10.4f} {r[3]:>10.4f}")
    
    # 按星期统计
    rows = conn.execute("""
        SELECT CASE 
            WHEN CAST(strftime('%w', timestamp) AS INTEGER) = 0 THEN '周日'
            WHEN CAST(strftime('%w', timestamp) AS INTEGER) = 1 THEN '周一'
            WHEN CAST(strftime('%w', timestamp) AS INTEGER) = 2 THEN '周二'
            WHEN CAST(strftime('%w', timestamp) AS INTEGER) = 3 THEN '周三'
            WHEN CAST(strftime('%w', timestamp) AS INTEGER) = 4 THEN '周四'
            WHEN CAST(strftime('%w', timestamp) AS INTEGER) = 5 THEN '周五'
            WHEN CAST(strftime('%w', timestamp) AS INTEGER) = 6 THEN '周六'
        END as dow,
               COUNT(*) as cnt,
               SUM(pnl) as total_pnl
        FROM stop_loss_audit
        GROUP BY dow
        ORDER BY MIN(timestamp)
    """).fetchall()
    
    print("\n--- 按星期统计 ---")
    for r in rows:
        print(f"  {r[0]:<6} cnt={r[1]:>5} totalPnL={r[2]:>8.4f}")

def analyze_entry_exit_gap(conn):
    """分析入场到出场价格差距"""
    print("\n" + "="*60)
    print("=== 入场→出场价差分析 ===")
    
    rows = conn.execute("""
        SELECT symbol, strategy_name, trigger_type,
               entry_price, exit_price, 
               (exit_price - entry_price) / entry_price as price_change,
               pnl_percent, quantity, pnl
        FROM stop_loss_audit
        WHERE trigger_type = 'stop_loss' AND pnl < 0
        ORDER BY pnl ASC
        LIMIT 20
    """).fetchall()
    
    print("\n--- 最差止损TOP20 ---")
    for r in rows:
        print(f"  {r[0]:<20} {r[1]:<10} entry={r[3]:.4f} exit={r[4]:.4f} change={r[5]:.4%} pnl={r[8]:.4f}")

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    analyze_stop_loss_audit(conn)
    analyze_fill_quality(conn)
    analyze_equity_curve(conn)
    analyze_trade_patterns(conn)
    analyze_entry_exit_gap(conn)
    
    # 汇总关键指标
    print("\n" + "="*60)
    print("=== 关键指标汇总 ===")
    
    total = conn.execute("SELECT COUNT(*), SUM(pnl), AVG(pnl), SUM(pnl_percent) FROM stop_loss_audit").fetchone()
    print(f"  总交易: {total[0]} | 总PnL: {total[1]:.4f} | 均PnL: {total[2]:.4f} | 累计PnL%: {total[3]:.4%}")
    
    scalping = conn.execute("SELECT COUNT(*), SUM(pnl), AVG(pnl) FROM stop_loss_audit WHERE strategy_name='scalping'").fetchone()
    if scalping[0]:
        print(f"  剥头皮: {scalping[0]}笔 | 总PnL: {scalping[1]:.4f} | 均PnL: {scalping[2]:.4f}")
    
    grid = conn.execute("SELECT COUNT(*), SUM(pnl), AVG(pnl) FROM stop_loss_audit WHERE strategy_name='grid'").fetchone()
    if grid[0]:
        print(f"  网格: {grid[0]}笔 | 总PnL: {grid[1]:.4f} | 均PnL: {grid[2]:.4f}")
    
    trend = conn.execute("SELECT COUNT(*), SUM(pnl), AVG(pnl) FROM stop_loss_audit WHERE strategy_name='trend'").fetchone()
    if trend[0]:
        print(f"  趋势: {trend[0]}笔 | 总PnL: {trend[1]:.4f} | 均PnL: {trend[2]:.4f}")
    
    conn.close()

if __name__ == "__main__":
    main()