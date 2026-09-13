import sqlite3
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

conn = sqlite3.connect('./data/trading.db')
c = conn.cursor()

print("=" * 60)
print("账户历史详细检查")
print("=" * 60)

c.execute("PRAGMA table_info(account_history)")
cols = c.fetchall()
print("\naccount_history表结构:")
for col in cols:
    print(f"  {col[1]} ({col[2]})")

c.execute("SELECT * FROM account_history ORDER BY timestamp DESC LIMIT 1")
row = c.fetchone()
col_names = [d[0] for d in c.description]
print("\n最新账户数据:")
for i, name in enumerate(col_names):
    print(f"  {name}: {row[i]}")

print("\n" + "=" * 60)
print("交易记录时间分布")
print("=" * 60)

c.execute("SELECT DATE(create_time) as dt, COUNT(*), strategy_name FROM trade_records GROUP BY dt, strategy_name ORDER BY dt DESC LIMIT 10")
print("\n按日期+策略:")
for row in c.fetchall():
    print(f"  {row[0]}  {row[2]:10s}  {row[1]}笔")

c.execute("SELECT MAX(create_time) FROM trade_records")
last_trade = c.fetchone()[0]
print(f"\n最后一笔交易时间: {last_trade}")

print("\n" + "=" * 60)
print("策略绩效表检查")
print("=" * 60)

c.execute("PRAGMA table_info(strategy_performance)")
cols = c.fetchall()
print("\nstrategy_performance表结构:")
for col in cols:
    print(f"  {col[1]} ({col[2]})")

c.execute("SELECT * FROM strategy_performance ORDER BY last_update DESC")
rows = c.fetchall()
col_names = [d[0] for d in c.description]
print(f"\n策略绩效记录数: {len(rows)}")
for row in rows:
    d = dict(zip(col_names, row))
    print(f"  {d.get('strategy_name','')}: trades={d.get('total_trades',0)} wins={d.get('winning_trades',0)} "
          f"wr={d.get('win_rate',0):.2%} pnl={d.get('total_pnl',0):.4f} dd={d.get('max_drawdown',0):.2%} "
          f"updated={d.get('last_update','')}")

print("\n" + "=" * 60)
print("trades表检查")
print("=" * 60)

c.execute("PRAGMA table_info(trades)")
cols = c.fetchall()
print("\ntrades表结构:")
for col in cols:
    print(f"  {col[1]} ({col[2]})")

c.execute("SELECT COUNT(*) FROM trades")
print(f"\ntrades表记录数: {c.fetchone()[0]}")

c.execute("SELECT * FROM trades ORDER BY create_time DESC LIMIT 3")
rows = c.fetchall()
col_names = [d[0] for d in c.description]
for row in rows:
    d = dict(zip(col_names, row))
    print(f"  {d.get('create_time','')} {d.get('strategy_name',''):10s} {d.get('symbol',''):15s} "
          f"{d.get('side',''):5s} {d.get('status',''):8s} pnl={d.get('pnl',0):.4f}")

print("\n" + "=" * 60)
print("risk_events表检查")
print("=" * 60)

c.execute("SELECT COUNT(*) FROM risk_events")
print(f"风险事件总数: {c.fetchone()[0]}")

c.execute("SELECT * FROM risk_events ORDER BY event_time DESC LIMIT 5")
rows = c.fetchall()
col_names = [d[0] for d in c.description]
for row in rows:
    d = dict(zip(col_names, row))
    print(f"  {d.get('event_time','')} {d.get('event_type',''):15s} level={d.get('severity_level','')} "
          f"action={d.get('action_taken','')}")

conn.close()
