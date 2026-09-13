import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.dashboard_engine import DashboardEngine

cfg = {"dashboard": {"db_path": "data/trading.db"}}
eng = DashboardEngine(cfg)

print("=== get_historical_performance(30d) ===")
p = eng.get_historical_performance(days=30, force_refresh=True)
for k in ["start_equity","end_equity","total_return","total_return_pct","max_drawdown","max_drawdown_duration","best_day","worst_day","avg_daily_return","total_trades","win_rate"]:
    print(f"  {k}: {p.get(k)}")

print("\n=== get_historical_performance(7d) ===")
p7 = eng.get_historical_performance(days=7, force_refresh=True)
for k in ["start_equity","end_equity","total_return","total_return_pct","max_drawdown","best_day","worst_day","avg_daily_return"]:
    print(f"  {k}: {p7.get(k)}")

print("\n=== get_risk_dashboard ===")
r = eng.get_risk_dashboard(force_refresh=True)
print("  risk_metrics:", r.get("risk_metrics"))
