from core.dashboard_engine import DashboardEngine

eng = DashboardEngine()

daily = eng._fetch_daily_series(days=30)
equities = daily["equities"]
flow_days = daily["flow_days"]

print("=== 日聚合序列（天数=%d）===" % len(equities))
print("首:", equities[0], "末:", equities[-1])
print("出入金日:", sorted(flow_days))

print("\n=== 每日权益 ===")
for d, e in equities:
    mark = " [出入金]" if d in flow_days else ""
    print(f"  {d}  eq={e:.4f}{mark}")

hp = eng.get_historical_performance(days=30, force_refresh=True)
print("\n=== 历史表现 30日 ===")
for k in ("start_equity", "end_equity", "total_return_pct", "max_drawdown",
          "max_drawdown_duration", "best_day", "worst_day", "avg_daily_return",
          "volatility_annual", "sharpe_ratio", "sortino_ratio", "calmar_ratio"):
    print(f"  {k} = {hp.get(k)}")

rd = eng.get_risk_dashboard(force_refresh=True)
rm = rd.get("risk_metrics", {})
print("\n=== 风险仪表盘 ===")
print("  max_drawdown =", rm.get("max_drawdown"))
print("  current_drawdown =", rm.get("current_drawdown"))
