"""诊断 web 持仓不同步：对比 dashboard_api.fetch_okx_positions 与 DashboardEngine.get_position_distribution"""
import os, sys, json

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)
sys.path.insert(0, BASE)

# 直接查询 OKX（dashboard_api.fetch_okx_positions，主账号）
from dashboard_api import fetch_okx_positions, _okx_circuit_open
print("=== fetch_okx_positions() (主账号, 5s缓存) ===")
print("  circuit_open =", _okx_circuit_open())
positions = fetch_okx_positions()
print(f"  count = {len(positions)}")
for p in positions:
    print(f"    instId={p.get('instId')} posSide={p.get('posSide')} pos={p.get('pos')} "
          f"margin={p.get('margin')} imr={p.get('imr')} notionalUsd={p.get('notionalUsd')} "
          f"markPx={p.get('markPx')} avgPx={p.get('avgPx')} lever={p.get('lever')} upl={p.get('upl')}")

# 绕过缓存，直接重新 fetch（读缓存 vs 强制）
print("\n=== DashboardEngine.get_position_distribution(force_refresh=True) ===")
from core.dashboard_engine import DashboardEngine
from configs.settings import load_config
cfg = load_config()
engine = DashboardEngine(cfg)
print("  engine._okx_client =", engine._okx_client)
result = engine.get_position_distribution(force_refresh=True)
print(f"  summary.total_positions = {result.get('summary', {}).get('total_positions')}")
for p in result.get("positions", []):
    print(f"    {p.get('symbol')} {p.get('side')} qty={p.get('quantity')} "
          f"notional={p.get('notional')} mark={p.get('mark_price')} margin={p.get('margin')}")

# DB position_history 快照
print("\n=== position_history 最近 5 分钟（symbol+side 最新） ===")
import sqlite3
conn = sqlite3.connect(os.path.join(BASE, "data", "trading.db"))
conn.row_factory = sqlite3.Row
rows = conn.execute(
    "SELECT symbol, side, quantity, mark_price, margin, leverage, timestamp "
    "FROM position_history WHERE timestamp > datetime('now', '-5 minutes') "
    "ORDER BY timestamp DESC LIMIT 200"
).fetchall()
seen = {}
for r in rows:
    k = (r["symbol"], r["side"])
    if k not in seen:
        seen[k] = r
for k, r in sorted(seen.items()):
    print(f"    {r['symbol']:18s} {r['side']:6s} qty={r['quantity']} mark={r['mark_price']} "
          f"margin={r['margin']} lev={r['leverage']} ts={r['timestamp']}")
print(f"  total (symbol,side) = {len(seen)}")
conn.close()
