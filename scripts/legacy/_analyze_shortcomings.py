"""量化核心短板分析脚本"""
import sys, os, json, sqlite3
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

db_path = 'data/trading.db'

print("=" * 70)
print("  量化核心短板分析")
print("=" * 70)

# ========== 1. 数据库交易记录 ==========
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("\n--- 最近30笔交易 ---")
    cursor.execute('SELECT * FROM trades ORDER BY created_at DESC LIMIT 30')
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    for row in rows:
        d = dict(zip(cols, row))
        print(f"{str(d.get('created_at',''))[:19]} | {d.get('symbol',''):20s} | {str(d.get('direction','')):5s} | qty={d.get('quantity',0):.6f} | px={d.get('exit_price',0) or 0:.6f} | pnl={d.get('pnl_usdt',0) or 0:+.4f} | fee={d.get('fees',0) or 0:.4f} | {d.get('strategy_name','')}")

    print("\n--- 策略盈亏汇总 ---")
    cursor.execute('SELECT strategy_name, COUNT(*), SUM(COALESCE(pnl_usdt,0)), SUM(COALESCE(fees,0)), MIN(created_at), MAX(created_at) FROM trades GROUP BY strategy_name')
    for row in cursor.fetchall():
        print(f"{str(row[0] or 'unknown'):20s}: {row[1]:3d}笔, PnL={row[2]:+.4f}, 手续费={row[3]:.4f}, {str(row[4])[:10]}~{str(row[5])[:10]}")

    cursor.execute('SELECT SUM(COALESCE(pnl_usdt,0)), SUM(COALESCE(fees,0)), COUNT(*) FROM trades')
    t = cursor.fetchone()
    print(f"\n总计: {t[2]}笔, PnL={t[0] or 0:+.4f}, 手续费={t[1] or 0:.4f}, 净盈亏={(t[0] or 0)-(t[1] or 0):+.4f}")

    # 按币种汇总
    print("\n--- 币种盈亏排行 ---")
    cursor.execute('SELECT symbol, COUNT(*), SUM(COALESCE(pnl_usdt,0)), SUM(COALESCE(fees,0)) FROM trades GROUP BY symbol ORDER BY SUM(COALESCE(pnl_usdt,0))')
    for row in cursor.fetchall():
        print(f"{row[0]:20s}: {row[1]:3d}笔, PnL={row[2] or 0:+.4f}, 净={(row[2] or 0)-(row[3] or 0):+.4f}")

    # 买卖比
    print("\n--- 买卖比例 ---")
    cursor.execute("SELECT direction, COUNT(*) FROM trades GROUP BY direction")
    for row in cursor.fetchall():
        print(f"{row[0]}: {row[1]}笔")

    # 胜率
    print("\n--- 胜率分析 ---")
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN win=1 THEN 1 ELSE 0 END) FROM trades")
    t = cursor.fetchone()
    print(f"总交易: {t[0]}, 盈利: {t[1]}, 胜率: {t[1]/t[0]*100:.1f}%" if t[0] > 0 else "无数据")

    # 权益曲线
    print("\n--- 权益曲线(最近15条) ---")
    cursor.execute('SELECT * FROM equity_curve ORDER BY timestamp DESC LIMIT 15')
    rows = cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    for row in rows:
        d = dict(zip(cols, row))
        print(f"{str(d.get('timestamp',''))[:19]} | equity={d.get('equity',0):.2f} | balance={d.get('balance',0):.2f} | margin={d.get('margin',0):.2f} | upnl={d.get('unrealized_pnl',0):+.4f}")

    conn.close()
else:
    print("DB not found")

# ========== 2. 策略状态分析 ==========
print("\n" + "=" * 70)
print("  策略状态分析")
print("=" * 70)

state_dir = 'data/strategy_state'
for fname in sorted(os.listdir(state_dir)):
    if not fname.endswith('.json') or fname == 'reset_instructions.json':
        continue
    fpath = os.path.join(state_dir, fname)
    with open(fpath, 'r') as f:
        data = json.load(f)
    
    strategy = fname.replace('.json', '')
    positions = data.get('positions', data.get('_position_state', data.get('grids', data.get('active_positions', {}))))
    if isinstance(positions, dict):
        open_pos = {k: v for k, v in positions.items() if v.get('status') == 'open' or (isinstance(v, dict) and 'status' not in v)}
        closed_pos = {k: v for k, v in positions.items() if v.get('status') == 'closed'}
        print(f"\n{strategy}: {len(open_pos)} open, {len(closed_pos)} closed")
        for sym, pos in open_pos.items():
            print(f"  {sym}: dir={pos.get('direction','?')}, entry={pos.get('entry_price',0)}, qty={pos.get('current_quantity',pos.get('quantity',0))}, profit={pos.get('current_profit',0):+.4f}")

# ========== 3. 配置分析 ==========
print("\n" + "=" * 70)
print("  资金配置分析")
print("=" * 70)

from configs.settings import load_config
config = load_config()

trading = config.get('trading', {})
cp = config.get('capital_pool', {})
print(f"总资金: {trading.get('total_capital', 'N/A')} USDT")
print(f"交易资金比例: {trading.get('trading_capital_ratio', 'N/A')}")
print(f"资金池: BASE={cp.get('base_ratio',0)}, ADD={cp.get('add_reserve_ratio',0)}, RISK={cp.get('risk_isolation_ratio',0)}")
print(f"风控: 日最大亏损={trading.get('daily_max_loss',0)}, 单笔风险={trading.get('risk_per_trade',0)}, 最大回撤={trading.get('max_drawdown',0)}")
print(f"最大持仓数: {trading.get('max_concurrent_positions',0)}")
print(f"最大杠杆: {trading.get('max_total_leverage',0)}")
print(f"最小余额: {trading.get('min_balance',0)}")
print(f"最小名义价值: {trading.get('min_notional_usd',0)}")

# 策略分配
for s in ['grid', 'scalping', 'trend', 'arbitrage', 'spot_grid', 'spot_martingale']:
    alloc = trading.get(f'{s}_allocation', 0)
    strat_cfg = config.get('strategies', {}).get(s, {})
    enabled = strat_cfg.get('enabled', False)
    print(f"{s}: enabled={enabled}, alloc={alloc:.2%}, leverage={strat_cfg.get('leverage', strat_cfg.get('leverage_default', 'N/A'))}")

# 资金磨损
attr = config.get('capital_attrition', {})
print(f"\n资金磨损预算: 日={attr.get('default_daily_budget_usdt',0)} USDT, 周={attr.get('default_weekly_budget_usdt',0)} USDT")
print(f"费用率: maker={attr.get('maker_fee_rate',0)}, taker={attr.get('taker_fee_rate',0)}")

# ========== 4. 短板诊断 ==========
print("\n" + "=" * 70)
print("  核心短板诊断")
print("=" * 70)

shortcomings = []

# 1. 资金利用率
total_capital = trading.get('total_capital', 0)
trading_capital = total_capital * trading.get('trading_capital_ratio', 0.95)
# 从DB获取最近持仓
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT equity, balance, margin FROM equity_curve ORDER BY timestamp DESC LIMIT 1")
        last = cursor.fetchone()
        if last:
            equity, balance, margin = last
            margin_ratio = (margin or 0) / (equity or 1) if equity else 0
            print(f"\n1. 资金利用率: 保证金={margin:.2f} / 权益={equity:.2f} = {margin_ratio:.1%}")
            if margin_ratio < 0.1:
                shortcomings.append(("CRITICAL", f"资金利用率极低({margin_ratio:.1%})，大量资金闲置"))
                print(f"   -> 短板: 资金利用率仅{margin_ratio:.1%}，超90%资金闲置")
            elif margin_ratio < 0.3:
                shortcomings.append(("HIGH", f"资金利用率偏低({margin_ratio:.1%})，建议提升到30-50%"))
                print(f"   -> 短板: 资金利用率{margin_ratio:.1%}，建议提升")
        else:
            print("\n1. 资金利用率: 无数据")
    except Exception as e:
        print(f"查询出错: {e}")
    conn.close()

# 2. 策略启用状态
enabled_count = 0
for s in ['grid', 'scalping', 'trend', 'arbitrage']:
    if config.get('strategies', {}).get(s, {}).get('enabled', False):
        enabled_count += 1
print(f"\n2. 活跃策略数: {enabled_count}/4 (合约)")
if enabled_count < 4:
    shortcomings.append(("MEDIUM", f"仅{enabled_count}个合约策略活跃，可增加策略多样性"))

# 3. 最小名义价值限制
min_notional = trading.get('min_notional_usd', 3)
print(f"\n3. 最小名义价值: {min_notional} USDT")
if total_capital < 200:
    print(f"   -> 总资金{total_capital:.1f}USDT，单笔{min_notional}USDT，可同时开{total_capital/min_notional:.0f}笔")
    if total_capital / min_notional < 5:
        shortcomings.append(("HIGH", f"小资金({total_capital:.1f}USDT)下可开仓数量有限"))

# 4. 最大回撤与止损
max_dd = trading.get('max_drawdown', 0.25)
daily_max_loss = trading.get('daily_max_loss', 0.04)
print(f"\n4. 风控参数: 最大回撤={max_dd:.0%}, 日最大亏损={daily_max_loss:.0%}")
drawdown_risk = total_capital * max_dd
print(f"   可承受最大亏损: {drawdown_risk:.2f} USDT")
if daily_max_loss * total_capital < 5:
    shortcomings.append(("MEDIUM", f"日亏损上限仅{daily_max_loss*total_capital:.2f}USDT，可能限制交易机会"))

# 5. 资金磨损
daily_budget = attr.get('default_daily_budget_usdt', 5)
print(f"\n5. 资金磨损日预算: {daily_budget} USDT")
if total_capital > 0 and daily_budget / total_capital > 0.05:
    shortcomings.append(("MEDIUM", f"资金磨损预算占比{daily_budget/total_capital:.1%}偏高"))

# 6. 系统健康
health_path = 'data/health_status.json'
if os.path.exists(health_path):
    with open(health_path, 'r') as f:
        health = json.load(f)
    score = health.get('overall_score', 0)
    print(f"\n6. 系统健康度: {score:.1f}/100 ({health.get('overall_level','?')})")
    if score < 70:
        shortcomings.append(("MEDIUM", f"系统健康度{score:.1f}偏低，组件: {health.get('components',{})}"))

# 汇总
print("\n" + "=" * 70)
print("  短板汇总")
print("=" * 70)
for severity, msg in sorted(shortcomings, key=lambda x: {"CRITICAL":0,"HIGH":1,"MEDIUM":2}.get(x[0], 3)):
    print(f"  [{severity}] {msg}")

print(f"\n共发现 {len(shortcomings)} 个短板")
print(f"CRITICAL: {sum(1 for s,_ in shortcomings if s=='CRITICAL')}")
print(f"HIGH: {sum(1 for s,_ in shortcomings if s=='HIGH')}")
print(f"MEDIUM: {sum(1 for s,_ in shortcomings if s=='MEDIUM')}")