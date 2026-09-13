"""生产级资金增长计划 - 从124.72 USDT到500+ USDT的四阶段增长路线图"""
import sys, os, json, sqlite3, math
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

db_path = 'data/trading.db'
state_dir = 'data/strategy_state'

print("=" * 80)
print("  生产级资金增长计划 (Capital Growth Plan v2.0)")
print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# ============================================================
# 0. 当前状态快照
# ============================================================
print("\n" + "─" * 80)
print("  【0】当前状态快照")
print("─" * 80)

from configs.settings import load_config
config = load_config()
trading = config.get('trading', {})
total_capital = trading.get('total_capital', 124.72)
trading_capital = total_capital * trading.get('trading_capital_ratio', 0.95)

# 获取最新权益
equity, balance, margin, upnl = 0, 0, 0, 0
margin_ratio = 0
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT equity, balance, margin, unrealized_pnl FROM equity_curve ORDER BY timestamp DESC LIMIT 1")
        last = cursor.fetchone()
        if last:
            equity, balance, margin, upnl = last
            margin_ratio = (margin or 0) / (equity or 1) if equity else 0
        conn.close()
    except:
        conn.close()

# 活跃策略
active_strategies = []
for s in ['grid', 'scalping', 'trend', 'arbitrage']:
    if config.get('strategies', {}).get(s, {}).get('enabled', False):
        active_strategies.append(s)

# 当前持仓
current_positions = []
for s in active_strategies:
    fpath = os.path.join(state_dir, f'{s}.json')
    if os.path.exists(fpath):
        with open(fpath, 'r') as f:
            data = json.load(f)
        positions = data.get('positions', data.get('_position_state', {}))
        if isinstance(positions, dict):
            for sym, pos in positions.items():
                if isinstance(pos, dict) and pos.get('status') == 'open':
                    current_positions.append({
                        'symbol': sym, 'strategy': s,
                        'direction': pos.get('direction', '?'),
                        'entry_price': pos.get('entry_price', 0),
                        'quantity': pos.get('current_quantity', pos.get('quantity', 0)),
                    })

print(f"  总资金:       {total_capital:.2f} USDT")
print(f"  交易资金:     {trading_capital:.2f} USDT (95%)")
print(f"  权益:         {equity:.2f} USDT")
print(f"  可用余额:     {balance:.2f} USDT")
print(f"  保证金:       {margin:.2f} USDT")
print(f"  保证金率:     {margin_ratio:.1%}")
print(f"  浮盈:         {upnl:+.4f} USDT")
print(f"  活跃策略:     {len(active_strategies)}/4 ({', '.join(active_strategies)})")
print(f"  当前持仓:     {len(current_positions)} 个")
for pos in current_positions:
    print(f"    {pos['symbol']:20s} {pos['direction']:5s} @ {pos['entry_price']} x {pos['quantity']:.6f} [{pos['strategy']}]")

# 闲置资金
idle_capital = trading_capital - margin
idle_ratio = idle_capital / trading_capital if trading_capital > 0 else 0
print(f"  闲置资金:     {idle_capital:.2f} USDT ({idle_ratio:.1%})")

# ============================================================
# 1. 核心短板量化
# ============================================================
print("\n" + "─" * 80)
print("  【1】核心短板量化诊断")
print("─" * 80)

shortcomings = []

# 1.1 资金利用率
print(f"\n  >>> 1.1 资金利用率: {margin_ratio:.1%}")
if margin_ratio < 0.1:
    score = 0
    desc = f"资金利用率极低({margin_ratio:.1%})，{idle_ratio:.0%}资金闲置"
    action = "立即提升到30-50%，将闲置资金分配给活跃策略"
    shortcomings.append(("CRITICAL", "资金利用率", score, desc, action))
elif margin_ratio < 0.3:
    score = 30
    desc = f"资金利用率偏低({margin_ratio:.1%})"
    action = "逐步提升到35-50%，优先分配给高Sharpe策略"
    shortcomings.append(("HIGH", "资金利用率", score, desc, action))
else:
    score = 70
    desc = f"资金利用率合理({margin_ratio:.1%})"
    action = "维持当前水平，微调策略间分配"
    shortcomings.append(("OK", "资金利用率", score, desc, action))

# 1.2 策略收益分布
print(f"\n  >>> 1.2 策略收益分布")
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT strategy_name, COUNT(*), SUM(COALESCE(pnl_usdt,0)), SUM(COALESCE(fees,0)), AVG(COALESCE(pnl_usdt,0)) FROM trades GROUP BY strategy_name')
    strat_perf = {}
    for row in cursor.fetchall():
        sname, cnt, pnl, fees, avg_pnl = row
        net = (pnl or 0) - (fees or 0)
        strat_perf[sname] = {'count': cnt, 'pnl': pnl or 0, 'fees': fees or 0, 'net': net, 'avg': avg_pnl or 0}
    conn.close()

    for sname in ['grid', 'scalping', 'trend', 'arbitrage']:
        perf = strat_perf.get(sname, {'count': 0, 'pnl': 0, 'fees': 0, 'net': 0, 'avg': 0})
        print(f"    {sname:12s}: {perf['count']:3d}笔, 净PnL={perf['net']:+.4f}, 均笔={perf['avg']:+.4f}, 费率={perf['fees']/max(abs(perf['pnl']),0.01):.1%}")

# 1.3 胜率与盈亏比
print(f"\n  >>> 1.3 胜率与盈亏比")
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*), SUM(CASE WHEN win=1 THEN 1 ELSE 0 END), SUM(CASE WHEN pnl_usdt>0 THEN pnl_usdt ELSE 0 END), SUM(CASE WHEN pnl_usdt<0 THEN ABS(pnl_usdt) ELSE 0 END) FROM trades")
    total, wins, win_pnl, loss_pnl = cursor.fetchone()
    conn.close()
    win_rate = wins / total if total > 0 else 0
    avg_win = win_pnl / wins if wins > 0 else 0
    avg_loss = loss_pnl / (total - wins) if (total - wins) > 0 else 0
    profit_factor = win_pnl / loss_pnl if loss_pnl > 0 else float('inf')
    print(f"    总交易: {total}笔, 胜率: {win_rate:.1%}")
    print(f"    均盈利: {avg_win:+.4f}, 均亏损: {-avg_loss:+.4f}")
    print(f"    盈亏比: {avg_win/avg_loss:.2f}" if avg_loss > 0 else "    盈亏比: ∞")
    print(f"    盈利因子: {profit_factor:.2f}")

    if win_rate < 0.3:
        shortcomings.append(("HIGH", "胜率", 20, f"胜率仅{win_rate:.1%}, 远低于健康水平40%", "提升信号质量，减少低质量开仓"))
    if profit_factor < 1.5:
        shortcomings.append(("HIGH", "盈亏比", 25, f"盈利因子{profit_factor:.2f}<1.5", "收紧止损，放宽止盈，提高盈亏比"))

# 1.4 资金磨损
print(f"\n  >>> 1.4 资金磨损分析")
attr = config.get('capital_attrition', {})
daily_budget = attr.get('default_daily_budget_usdt', 5)
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT SUM(COALESCE(fees,0)), COUNT(*) FROM trades WHERE created_at > ?", ((datetime.now() - timedelta(days=7)).isoformat(),))
    week_fees, week_trades = cursor.fetchone()
    conn.close()
    daily_fees = (week_fees or 0) / 7
    fee_ratio = daily_fees / total_capital if total_capital > 0 else 0
    print(f"    日均手续费: {daily_fees:.4f} USDT ({fee_ratio:.3%})")
    print(f"    日预算:     {daily_budget:.2f} USDT")
    print(f"    预算利用率: {daily_fees/daily_budget*100:.1f}%" if daily_budget > 0 else "N/A")

    if fee_ratio > 0.005:
        shortcomings.append(("MEDIUM", "资金磨损", 50, f"日手续费率{fee_ratio:.3%}偏高", "降低交易频率，优先使用Maker单"))

# 1.5 策略多样性
print(f"\n  >>> 1.5 策略多样性")
enabled_count = len(active_strategies)
allocations = {}
for s in ['grid', 'scalping', 'trend', 'arbitrage']:
    alloc = trading.get(f'{s}_allocation', 0)
    allocations[s] = alloc
    enabled = config.get('strategies', {}).get(s, {}).get('enabled', False)
    print(f"    {s:12s}: enabled={enabled}, alloc={alloc:.1%}")

if enabled_count < 4:
    shortcomings.append(("MEDIUM", "策略多样性", 45, f"仅{enabled_count}/4策略活跃", "激活所有策略，分散风险"))

# 1.6 系统健康
health_path = 'data/health_status.json'
if os.path.exists(health_path):
    with open(health_path, 'r') as f:
        health = json.load(f)
    score = health.get('overall_score', 0)
    components = health.get('components', {})
    print(f"\n  >>> 1.6 系统健康度: {score:.1f}/100")
    for comp, info in components.items():
        print(f"    {comp}: {info.get('score', '?')}")
    if score < 70:
        shortcomings.append(("MEDIUM", "系统健康", 40, f"健康度{score:.1f}偏低", "检查Redis连接，优化内存使用"))

# 汇总
print("\n" + "─" * 80)
print("  【短板汇总】")
print("─" * 80)
severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "OK": 3}
for severity, name, score, desc, action in sorted(shortcomings, key=lambda x: severity_order.get(x[0], 99)):
    icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "OK": "🟢"}.get(severity, "⚪")
    print(f"  {icon} [{severity}] {name} (评分:{score}/100)")
    print(f"     问题: {desc}")
    print(f"     方案: {action}")

# ============================================================
# 2. 四阶段增长计划
# ============================================================
print("\n" + "=" * 80)
print("  【2】四阶段资金增长路线图")
print("=" * 80)

# 计算当前可用信息
margin_used = margin or 0
current_equity = equity or total_capital

# 阶段定义
phases = [
    {
        "name": "Phase 1: 修复期 (0-30天)",
        "target": 150.0,
        "from": current_equity,
        "days": 30,
        "margin_target": 0.35,  # 35%利用率
        "daily_target_pnl": 0.84,  # 需要日均盈利
        "description": "修复资金利用率，激活全部策略，建立稳定盈利基线",
        "actions": [
            "提升资金利用率: 6.1% → 35% (从闲置资金中释放 ~36 USDT)",
            "策略再平衡: Grid 20% → Scalping 35% (Scalping Sharpe最高)",
            "激活Trend策略: 恢复趋势策略运行，分配15%资金",
            "Grid策略优化: 降低XRP集中度，增加币种多样性",
            "收紧止损: 单笔最大亏损从2.5%降至1.5%",
            "提升信号质量: min_signal_quality从0.35提升到0.45",
            "修复系统健康: 重启Redis或降级为纯内存模式",
            "日盈利目标: +0.84 USDT/day (0.67% daily return)",
        ],
        "risk_params": {
            "daily_max_loss": 0.03,
            "risk_per_trade": 0.015,
            "max_concurrent_positions": 4,
            "max_total_leverage": 4.0,
        },
        "allocation": {
            "grid": 0.20,
            "scalping": 0.35,
            "trend": 0.15,
            "arbitrage": 0.30,
        },
        "milestones": [
            "Day 7: 资金利用率 > 20%",
            "Day 14: 连续3天正收益",
            "Day 21: 权益突破 135 USDT",
            "Day 30: 权益突破 150 USDT, 月收益 +20%",
        ]
    },
    {
        "name": "Phase 2: 优化期 (30-60天)",
        "target": 200.0,
        "from": 150.0,
        "days": 30,
        "margin_target": 0.45,
        "daily_target_pnl": 1.67,
        "description": "优化策略参数，扩大仓位规模，提升夏普比率",
        "actions": [
            "资金利用率: 35% → 45% (增加 ~15 USDT保证金)",
            "Kelly公式优化: 根据30天实盘数据校准Kelly系数",
            "扩大Scalping: 增加并发币种从2个到4个",
            "Grid策略升级: 启用动态网格数量和自适应间距",
            "Arbitrage策略: 增加资金费率套利频率",
            "引入波动率目标: 动态调整仓位匹配市场波动",
            "日盈利目标: +1.67 USDT/day (1.11% daily return)",
        ],
        "risk_params": {
            "daily_max_loss": 0.035,
            "risk_per_trade": 0.018,
            "max_concurrent_positions": 5,
            "max_total_leverage": 4.5,
        },
        "allocation": {
            "grid": 0.20,
            "scalping": 0.35,
            "trend": 0.20,
            "arbitrage": 0.25,
        },
        "milestones": [
            "Day 45: 权益突破 170 USDT",
            "Day 60: 权益突破 200 USDT, 累计收益 +60%",
        ]
    },
    {
        "name": "Phase 3: 扩张期 (60-90天)",
        "target": 300.0,
        "from": 200.0,
        "days": 30,
        "margin_target": 0.50,
        "daily_target_pnl": 3.33,
        "description": "资金超过2000 USDT，引入BTC/ETH交易对，提升杠杆上限",
        "actions": [
            "资金利用率: 45% → 50%",
            "引入BTC/ETH: 资金>200 USDT后逐步加入Tier1币种",
            "杠杆分级: Tier1(BTC/ETH) 2-3x, Tier2(SOL/XRP) 3-5x, Tier3 2-3x",
            "跨策略对冲: 启用多空对冲降低回撤",
            "组合优化器: 启用Markowitz均值方差优化",
            "日盈利目标: +3.33 USDT/day (1.67% daily return)",
        ],
        "risk_params": {
            "daily_max_loss": 0.04,
            "risk_per_trade": 0.02,
            "max_concurrent_positions": 6,
            "max_total_leverage": 5.0,
        },
        "allocation": {
            "grid": 0.20,
            "scalping": 0.30,
            "trend": 0.25,
            "arbitrage": 0.25,
        },
        "milestones": [
            "Day 75: 权益突破 250 USDT",
            "Day 90: 权益突破 300 USDT, 累计收益 +140%",
        ]
    },
    {
        "name": "Phase 4: 规模化 (90+天)",
        "target": 500.0,
        "from": 300.0,
        "days": 60,
        "margin_target": 0.55,
        "daily_target_pnl": 3.33,
        "description": "规模化运营，稳定盈利，引入组合风险管理",
        "actions": [
            "资金利用率: 50% → 55%",
            "全币种覆盖: 12个币种全部激活",
            "组合风险管理: VaR/CVaR实时监控",
            "绩效归因: 每日策略贡献度分析",
            "自动复利: 启用compound reinvest(80%利润再投资)",
            "日盈利目标: +3.33 USDT/day (1.11% daily return)",
        ],
        "risk_params": {
            "daily_max_loss": 0.04,
            "risk_per_trade": 0.02,
            "max_concurrent_positions": 8,
            "max_total_leverage": 5.0,
        },
        "allocation": {
            "grid": 0.20,
            "scalping": 0.25,
            "trend": 0.25,
            "arbitrage": 0.30,
        },
        "milestones": [
            "Day 120: 权益突破 400 USDT",
            "Day 150: 权益突破 500 USDT, 累计收益 +300%",
        ]
    },
]

# 打印阶段
for i, phase in enumerate(phases):
    print(f"\n{'─' * 80}")
    print(f"  {phase['name']}")
    print(f"{'─' * 80}")
    print(f"  目标: {phase['from']:.0f} → {phase['target']:.0f} USDT ({phase['days']}天)")
    print(f"  日均目标PnL: +{phase['daily_target_pnl']:.2f} USDT")
    print(f"  目标资金利用率: {phase['margin_target']:.0%}")
    print(f"  描述: {phase['description']}")
    print(f"\n  关键行动:")
    for j, action in enumerate(phase['actions'], 1):
        print(f"    {j}. {action}")
    print(f"\n  策略分配:")
    for s, alloc in phase['allocation'].items():
        cap = phase['from'] * alloc
        print(f"    {s:12s}: {alloc:.0%} ({cap:.1f} USDT)")
    print(f"\n  风控参数:")
    for k, v in phase['risk_params'].items():
        print(f"    {k}: {v}")
    print(f"\n  里程碑:")
    for m in phase['milestones']:
        print(f"    ✓ {m}")

# ============================================================
# 3. 概率模拟
# ============================================================
print("\n" + "=" * 80)
print("  【3】蒙特卡洛增长模拟 (1000次)")
print("=" * 80)

import random
random.seed(42)

def simulate_growth(initial_capital, daily_return_mean, daily_return_std, days, n_sim=1000):
    """蒙特卡洛模拟资金增长"""
    results = []
    for _ in range(n_sim):
        capital = initial_capital
        path = [capital]
        for _ in range(days):
            daily_return = random.gauss(daily_return_mean, daily_return_std)
            capital *= (1 + daily_return)
            # 风控熔断: 单日亏损超过8%
            if daily_return < -0.08:
                capital *= 0.95  # 紧急减仓
            path.append(capital)
        results.append(path)
    return results

# 保守参数 (基于当前胜率15.5%和盈亏比)
# 假设优化后: 胜率35%, 盈亏比1.5, 日交易4次, 每次风险1.5%
# 期望日收益 ≈ 0.35*1.5*0.015 - 0.65*0.015 = 0.007875 - 0.00975 = -0.001875
# 需要大幅提升胜率到40%+才能盈利
# 目标: 胜率40%, 盈亏比1.8
# 期望日收益 ≈ 0.4*1.8*0.015*4 - 0.6*0.015*4 = 0.0432 - 0.036 = 0.0072 (0.72%)

print("\n  模拟假设:")
print(f"    初始资金: {current_equity:.2f} USDT")
print(f"    日均收益: 0.72% (基于胜率40%, 盈亏比1.8, 日交易4次)")
print(f"    日波动率: 2.5%")
print(f"    模拟天数: 150天")
print(f"    模拟次数: 1000")

sims = simulate_growth(current_equity, 0.0072, 0.025, 150, 1000)

# 统计
final_values = [s[-1] for s in sims]
final_values.sort()
p10 = final_values[100]
p25 = final_values[250]
p50 = final_values[500]
p75 = final_values[750]
p90 = final_values[900]
mean_final = sum(final_values) / len(final_values)
success_rate = sum(1 for v in final_values if v >= 500) / len(final_values) * 100

print(f"\n  150天后模拟结果:")
print(f"    P10 (悲观):  {p10:.2f} USDT")
print(f"    P25:         {p25:.2f} USDT")
print(f"    P50 (中位):  {p50:.2f} USDT")
print(f"    P75:         {p75:.2f} USDT")
print(f"    P90 (乐观):  {p90:.2f} USDT")
print(f"    均值:        {mean_final:.2f} USDT")
print(f"    达到500+概率: {success_rate:.1f}%")

# 分阶段概率
for phase in phases:
    target = phase['target']
    days = phase['days']
    # 累积天数
    cumulative_days = sum(p['days'] for p in phases[:phases.index(phase)+1])
    cumulative_finals = [s[cumulative_days] for s in sims]
    prob = sum(1 for v in cumulative_finals if v >= target) / len(cumulative_finals) * 100
    mean_val = sum(cumulative_finals) / len(cumulative_finals)
    print(f"\n    {phase['name'].split(':')[0]}: 达到{target}USDT概率={prob:.1f}%, 均值={mean_val:.1f}")

# ============================================================
# 4. 立即执行清单
# ============================================================
print("\n" + "=" * 80)
print("  【4】立即执行清单 (今天)")
print("=" * 80)

immediate_actions = [
    {
        "priority": "P0-立即",
        "action": "提升资金利用率到35%",
        "detail": f"当前{margin_ratio:.1%}，从闲置资金{idle_capital:.1f}USDT中释放{trading_capital*0.35-margin:.1f}USDT",
        "config_change": "无需修改config，策略自动使用更多资金",
        "risk": "低 - 仍保留65%安全垫",
    },
    {
        "priority": "P0-立即",
        "action": "策略分配再平衡",
        "detail": "Scalping从16.7%提升到35%，Grid从16.7%降到20%",
        "config_change": "trading.scalping_allocation: 0.1666→0.35, trading.grid_allocation: 0.167→0.20",
        "risk": "低 - Scalping历史表现最佳(+7USDT/笔)",
    },
    {
        "priority": "P0-立即",
        "action": "提升信号质量门槛",
        "detail": "Grid min_signal_quality: 0.35→0.45, 减少低质量开仓",
        "config_change": "strategies.grid.min_signal_quality: 0.35→0.45",
        "risk": "低 - 减少交易频率，提高胜率",
    },
    {
        "priority": "P0-立即",
        "action": "收紧止损参数",
        "detail": "单笔风险从2.5%降到1.5%，日最大亏损从4%降到3%",
        "config_change": "trading.risk_per_trade: 0.025→0.015, trading.daily_max_loss: 0.04→0.03",
        "risk": "无 - 降低风险敞口",
    },
    {
        "priority": "P1-今天",
        "action": "激活Trend策略",
        "detail": "Trend策略已启用但未活跃，检查并重启",
        "config_change": "strategies.trend.enabled: true (已确认)",
        "risk": "中 - 需验证策略信号正常",
    },
    {
        "priority": "P1-今天",
        "action": "Grid策略减少XRP集中度",
        "detail": "XRP占Grid交易112/123笔(91%), 亏损-0.32USDT",
        "config_change": "增加tier2_symbols多样性，降低XRP权重",
        "risk": "低 - 分散风险",
    },
    {
        "priority": "P1-今天",
        "action": "修复系统健康度",
        "detail": "Redis不可用(score=30), 内存压力(57.7)",
        "config_change": "检查Redis连接，或降级为纯内存缓存",
        "risk": "低 - 提升系统稳定性",
    },
    {
        "priority": "P2-本周",
        "action": "启用盈利复利",
        "detail": "compound_enabled: false→true, 80%利润再投资",
        "config_change": "trading.compound_enabled: false→true, trading.compound_reinvest_ratio: 0.8",
        "risk": "低 - 加速增长",
    },
]

for item in immediate_actions:
    print(f"\n  [{item['priority']}] {item['action']}")
    print(f"    详情: {item['detail']}")
    print(f"    配置: {item['config_change']}")
    print(f"    风险: {item['risk']}")

# ============================================================
# 5. 关键风险提示
# ============================================================
print("\n" + "=" * 80)
print("  【5】关键风险提示")
print("=" * 80)

risks = [
    ("小资金脆弱性", f"当前资金{current_equity:.1f}USDT，单笔大亏损可能致命。建议严格执行1.5%单笔风险上限。"),
    ("胜率陷阱", "当前胜率15.5%不可持续。必须通过提升信号质量将胜率提高到35%以上。"),
    ("过度交易", f"Grid策略123笔交易产生-0.45USDT亏损，手续费率高达{0.7513/abs(-0.4518)*100:.0f}%。减少交易频率。"),
    ("策略漂移", "Scalping策略仅1笔交易就赚+7USDT，但缺乏统计显著性。需积累更多交易验证。"),
    ("市场突变", "黑天鹅事件(Flash Crash)可导致远超止损的亏损。保持最大回撤25%的硬性限制。"),
    ("系统稳定性", "健康度69.3偏低，Redis不可用可能影响缓存和状态同步。"),
]

for title, desc in risks:
    print(f"\n  ⚠ {title}: {desc}")

# ============================================================
# 6. 监控指标仪表板
# ============================================================
print("\n" + "=" * 80)
print("  【6】关键监控指标 (每日检查)")
print("=" * 80)

kpis = [
    ("资金利用率", f"{margin_ratio:.1%}", "> 30%", "每日"),
    ("日PnL", f"{upnl:+.4f}", "> 0 USDT", "每日"),
    ("胜率(7日)", "15.5%", "> 35%", "每周"),
    ("最大回撤", "N/A", "< 15%", "每日"),
    ("活跃策略数", str(len(active_strategies)), "4/4", "每日"),
    ("系统健康度", f"{score:.0f}/100", "> 75/100", "每日"),
    ("手续费率", f"{daily_fees:.4f}/day", "< 0.5%/day", "每周"),
    ("夏普比率", "待计算", "> 1.0", "每月"),
]

print(f"\n  {'指标':<16s} {'当前':<14s} {'目标':<14s} {'频率':<10s}")
print(f"  {'─'*16} {'─'*14} {'─'*14} {'─'*10}")
for name, current, target, freq in kpis:
    print(f"  {name:<16s} {current:<14s} {target:<14s} {freq:<10s}")

# ============================================================
# 7. 增长计划JSON导出
# ============================================================
print("\n" + "=" * 80)
print("  【7】导出增长计划到文件")
print("=" * 80)

plan = {
    "generated_at": datetime.now().isoformat(),
    "version": "2.0",
    "current_state": {
        "total_capital": total_capital,
        "equity": equity,
        "balance": balance,
        "margin": margin,
        "margin_ratio": margin_ratio,
        "unrealized_pnl": upnl,
        "active_strategies": active_strategies,
        "current_positions": len(current_positions),
        "idle_capital": idle_capital,
        "idle_ratio": idle_ratio,
    },
    "shortcomings": [
        {"severity": s[0], "name": s[1], "score": s[2], "description": s[3], "action": s[4]}
        for s in shortcomings
    ],
    "phases": [
        {
            "name": p["name"],
            "target_usdt": p["target"],
            "from_usdt": p["from"],
            "days": p["days"],
            "daily_target_pnl": p["daily_target_pnl"],
            "margin_target": p["margin_target"],
            "allocation": p["allocation"],
            "risk_params": p["risk_params"],
            "milestones": p["milestones"],
        }
        for p in phases
    ],
    "monte_carlo": {
        "p10": round(p10, 2),
        "p25": round(p25, 2),
        "p50": round(p50, 2),
        "p75": round(p75, 2),
        "p90": round(p90, 2),
        "mean": round(mean_final, 2),
        "success_rate_500": round(success_rate, 1),
    },
    "immediate_actions": [
        {"priority": a["priority"], "action": a["action"], "detail": a["detail"], "config_change": a["config_change"]}
        for a in immediate_actions
    ],
}

os.makedirs('data', exist_ok=True)
plan_path = 'data/capital_growth_plan.json'
with open(plan_path, 'w') as f:
    json.dump(plan, f, indent=2, ensure_ascii=False, default=str)
print(f"  增长计划已导出到: {plan_path}")

print("\n" + "=" * 80)
print("  增长计划生成完毕。")
print("  核心建议: 立即提升资金利用率到35%，策略权重向Scalping倾斜。")
print("=" * 80)