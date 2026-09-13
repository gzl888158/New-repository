import sqlite3
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

conn = sqlite3.connect('./data/trading.db')
c = conn.cursor()

print("=" * 60)
print("数据库检查")
print("=" * 60)

c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [r[0] for r in c.fetchall()]
print(f"\n数据表: {tables}")

print("\n" + "=" * 60)
print("1. 交易记录统计")
print("=" * 60)
try:
    c.execute('SELECT COUNT(*) FROM trade_records')
    total = c.fetchone()[0]
    print(f"总交易记录: {total}")
    
    c.execute("SELECT status, COUNT(*) FROM trade_records GROUP BY status")
    print("\n按状态:")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]}")
    
    c.execute("SELECT strategy_name, COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins FROM trade_records GROUP BY strategy_name ORDER BY 2 DESC")
    print("\n按策略:")
    for row in c.fetchall():
        wr = row[2]/row[1]*100 if row[1] > 0 else 0
        print(f"  {row[0]}: {row[1]}笔, 胜{row[2]}, 胜率{wr:.1f}%")
    
    c.execute("SELECT create_time, strategy_name, symbol, side, status, pnl FROM trade_records ORDER BY create_time DESC LIMIT 10")
    print("\n最近10笔交易:")
    for row in c.fetchall():
        print(f"  {row[0][:19]} {row[1]:10s} {row[2]:15s} {row[3]:5s} {row[4]:8s} pnl:{row[5]:.4f}")
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 60)
print("2. 账户历史")
print("=" * 60)
try:
    c.execute('SELECT COUNT(*) FROM account_history')
    print(f"账户记录数: {c.fetchone()[0]}")
    
    c.execute("SELECT total_equity, available_balance, used_margin, unrealized_pnl, realized_pnl, margin_ratio, timestamp FROM account_history ORDER BY timestamp DESC LIMIT 3")
    print("\n最近账户状态:")
    for row in c.fetchall():
        eq = row[0] or 0
        used = row[2] or 0
        util = used/eq*100 if eq > 0 else 0
        print(f"  时间: {row[6]}")
        print(f"  总权益: {eq:.4f} USDT")
        print(f"  可用余额: {row[1]:.4f} USDT")
        print(f"  已用保证金: {used:.4f} USDT")
        print(f"  资金利用率: {util:.2f}%")
        print(f"  未实现盈亏: {row[3]:.4f}")
        print(f"  已实现盈亏: {row[4]:.4f}")
        print(f"  保证金率: {row[5]}")
        print()
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 60)
print("3. 持仓表")
print("=" * 60)
try:
    c.execute("SELECT COUNT(*) FROM positions WHERE quantity != 0")
    print(f"当前持仓数: {c.fetchone()[0]}")
    
    c.execute("SELECT symbol, side, quantity, avg_cost, unrealized_pnl, leverage, margin FROM positions WHERE quantity != 0 ORDER BY symbol")
    print("\n持仓明细:")
    total_margin = 0
    total_notional = 0
    for row in c.fetchall():
        margin = row[6] or 0
        notional = abs(row[2]) * row[3]
        total_margin += margin
        total_notional += notional
        print(f"  {row[0]:15s} {row[1]:5s} qty:{row[2]:.4f} 成本:{row[3]:.4f} 浮盈亏:{row[4]:.4f} 杠杆:{row[5]}x 保证金:{margin:.4f}")
    
    print(f"\n  总保证金: {total_margin:.4f} USDT")
    print(f"  总名义价值: {total_notional:.4f} USDT")
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 60)
print("4. 信号表")
print("=" * 60)
try:
    c.execute("SELECT COUNT(*) FROM signals")
    print(f"总信号数: {c.fetchone()[0]}")
    
    c.execute("SELECT strategy_name, COUNT(*) FROM signals GROUP BY strategy_name ORDER BY 2 DESC")
    print("\n按策略:")
    for row in c.fetchall():
        print(f"  {row[0]}: {row[1]}个信号")
    
    c.execute("SELECT create_time, strategy_name, symbol, direction, confidence, signal_type FROM signals ORDER BY create_time DESC LIMIT 10")
    print("\n最近10个信号:")
    for row in c.fetchall():
        print(f"  {row[0][:19]} {row[1]:10s} {row[2]:15s} {row[3]:5s} conf:{row[4]:.3f} type:{row[5]}")
except Exception as e:
    print(f"错误: {e}")

print("\n" + "=" * 60)
print("5. 策略绩效表")
print("=" * 60)
try:
    c.execute("SELECT * FROM strategy_performance ORDER BY last_update DESC LIMIT 10")
    cols = [d[0] for d in c.description]
    rows = c.fetchall()
    for row in rows:
        d = dict(zip(cols, row))
        print(f"  {d.get('strategy_name','')}: 交易{d.get('total_trades',0)}笔, 胜{d.get('winning_trades',0)}, "
              f"胜率{d.get('win_rate',0):.2%}, 盈亏{d.get('total_pnl',0):.4f}, 最大回撤{d.get('max_drawdown',0):.2%}")
except Exception as e:
    print(f"错误: {e}")

conn.close()

print("\n" + "=" * 60)
print("6. 策略状态文件检查")
print("=" * 60)

strategy_state_dir = "./data/strategy_state"
for f in os.listdir(strategy_state_dir):
    if f.endswith('.json') and f != 'reset_instructions.json':
        path = os.path.join(strategy_state_dir, f)
        try:
            with open(path, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
            
            pos_count = 0
            if '_position_state' in data:
                pos_count = len(data['_position_state'])
            elif 'positions' in data:
                pos_count = len(data['positions'])
            elif '_grids' in data:
                pos_count = len(data['_grids'])
            
            print(f"  {f:25s} 持仓/网格数: {pos_count}")
        except Exception as e:
            print(f"  {f}: 读取错误 - {e}")
