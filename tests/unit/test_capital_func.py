"""Quick functional test for capital distribution modules"""
import sys, os, asyncio, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.environ['OKX_API_KEY'] = 'test'
os.environ['OKX_SECRET_KEY'] = 'test'
os.environ['OKX_PASSPHRASE'] = 'test'
os.environ['REDIS_PASSWORD'] = 'test'
os.environ['TELEGRAM_BOT_TOKEN'] = 'test'
os.environ['TELEGRAM_CHAT_ID'] = 'test'

from configs.settings import load_config
config = load_config()

# 1. VolatilityTargeter
from risk.dynamic_allocator import VolatilityTargeter, AdaptiveKelly, CapitalEfficiencyMonitor
vt = VolatilityTargeter(config.get('volatility_targeting', {}))
returns = list(np.random.normal(0.001, 0.02, 100))
est = vt.estimate_volatility(returns)
print(f"VolatilityTargeter: current={est['current']:.4f}, ewma={est['ewma']:.4f}")
scale = vt.compute_scale_factor(0.25, 0.20, 5.0, 10.0)
print(f"  scale={scale['scale_factor']:.4f}, target_pos={scale['target_position']:.2f}")
breached, reason = vt.check_volatility_breach(0.55)
print(f"  breach={breached}")

# 2. AdaptiveKelly
ak = AdaptiveKelly(config.get('adaptive_kelly', {}))
result = ak.compute_kelly(win_rate=0.55, avg_win=0.02, avg_loss=0.015,
                          regime='trending_up', drawdown=0.03,
                          consecutive_wins=2, consecutive_losses=0, trade_count=50)
print(f"AdaptiveKelly: base={result['base_kelly']:.4f}, final={result['final_kelly']:.4f}, frac={result['fractional_kelly']:.4f}")
cont = ak.compute_continuous_kelly(returns)
print(f"  continuous={cont:.4f}")
penalty = ak.compute_drawdown_penalty(0.15)
print(f"  dd_penalty(15%)={penalty:.4f}")

# 3. CapitalEfficiencyMonitor
cem = CapitalEfficiencyMonitor(config.get('capital_efficiency', {}))
cem.update_metrics('grid', 1000, 600, 50, 5000, 30)
cem.update_metrics('trend', 800, 700, -20, 3000, 15)
cem.update_metrics('scalping', 500, 200, 30, 8000, 10)

async def run_eff():
    eff = await cem.compute_efficiency('grid')
    print(f"CapitalEfficiency: grid ROCE={eff['roc_e']:.4f}, score={eff['efficiency_score']:.4f}")
    ranked = await cem.rank_strategies()
    print(f"  ranking: {' > '.join([r['strategy'] + str(round(r['efficiency_score'],2)) for r in ranked])}")
    idle = await cem.detect_idle_capital()
    print(f"  idle={idle['total_idle_amount']:.2f}, strategies={idle['idle_count']}")
    suggestions = await cem.suggest_redeployment()
    print(f"  suggestions={len(suggestions)}")
    summary = cem.get_efficiency_summary()
    print(f"  summary: ROCE={summary['overall_roc_e']:.4f}, margin_util={summary['overall_margin_utilization']:.4f}")

asyncio.run(run_eff())
print("ALL FUNCTIONAL TESTS PASSED")