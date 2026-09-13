import sqlite3
import numpy as np

conn = sqlite3.connect('./data/trading.db')
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# 获取最近交易
cursor.execute("SELECT * FROM trades ORDER BY created_at DESC LIMIT 200")
trades = [dict(r) for r in cursor.fetchall()]
print(f"总交易记录: {len(trades)}")
print()

# 按策略统计
strategy_stats = {}
for t in trades:
    strategy = t.get('strategy_name', 'unknown')
    if not strategy:
        strategy = 'unknown'
    if strategy not in strategy_stats:
        strategy_stats[strategy] = {'count': 0, 'pnls': [], 'wins': 0, 'losses': 0}
    pnl = float(t.get('pnl_usdt', 0) or 0)
    strategy_stats[strategy]['count'] += 1
    strategy_stats[strategy]['pnls'].append(pnl)
    if pnl > 0:
        strategy_stats[strategy]['wins'] += 1
    elif pnl < 0:
        strategy_stats[strategy]['losses'] += 1

print("=" * 60)
print("各策略表现统计 (按总盈亏排序)")
print("=" * 60)
for strategy, stats in sorted(strategy_stats.items(), key=lambda x: sum(x[1]['pnls']), reverse=True):
    total_pnl = sum(stats['pnls'])
    win_rate = stats['wins'] / stats['count'] * 100 if stats['count'] > 0 else 0
    avg_pnl = np.mean(stats['pnls']) if stats['pnls'] else 0
    print(f"\n{strategy}:")
    print(f"  交易次数: {stats['count']}")
    print(f"  总盈亏: {total_pnl:.4f} USDT")
    print(f"  胜率: {win_rate:.1f}%")
    print(f"  平均盈亏: {avg_pnl:.4f} USDT")

# 按币种统计
symbol_stats = {}
for t in trades:
    symbol = t.get('symbol', 'unknown')
    if not symbol:
        symbol = 'unknown'
    if symbol not in symbol_stats:
        symbol_stats[symbol] = {'count': 0, 'pnls': []}
    pnl = float(t.get('pnl_usdt', 0) or 0)
    symbol_stats[symbol]['count'] += 1
    symbol_stats[symbol]['pnls'].append(pnl)

print()
print("=" * 60)
print("各币种表现 (按总盈亏排序)")
print("=" * 60)
sorted_symbols = sorted(symbol_stats.items(), key=lambda x: sum(x[1]['pnls']), reverse=True)
for symbol, stats in sorted_symbols[:10]:
    total_pnl = sum(stats['pnls'])
    print(f"  {symbol}: {total_pnl:.4f} USDT ({stats['count']}笔)")

# 资金曲线
try:
    cursor.execute("SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT 500")
    equity_points = [dict(r) for r in cursor.fetchall()]
    equity_points.reverse()
    if equity_points:
        print()
        print("=" * 60)
        print("资金曲线摘要")
        print("=" * 60)
        equities = [float(p.get('total_equity', 0) or 0) for p in equity_points]
        if equities:
            print(f"  起始: {equities[0]:.2f} USDT")
            print(f"  最新: {equities[-1]:.2f} USDT")
            print(f"  最高: {max(equities):.2f} USDT")
            print(f"  最低: {min(equities):.2f} USDT")
            change = equities[-1] - equities[0]
            change_pct = (equities[-1]/equities[0]-1)*100 if equities[0] > 0 else 0
            print(f"  变化: {change:.2f} USDT ({change_pct:.2f}%)")
except Exception as e:
    print(f"资金曲线查询失败: {e}")

conn.close()
