"""查询实盘交易状态"""
import sys, os, sqlite3, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import load_config
from core.okx_client import OKXClient

c = load_config()
k = OKXClient(c)

# 查询账户配置（持仓模式）
print("=" * 70)
print("[0] 账户配置")
print("=" * 70)
try:
    path = "/api/v5/account/config"
    headers = k._get_headers("GET", path)
    r = k._session.get(f"{k.rest_url}{path}", headers=headers)
    cfg = r.json()
    if cfg.get("code") == "0" and cfg.get("data"):
        d = cfg["data"][0]
        print(f"  posMode: {d.get('posMode')}")
        print(f"  acctLv: {d.get('acctLv')} (账户等级)")
        print(f"  ctIsoMode: {d.get('ctIsoMode')}")
        print(f"  mgnIsoMode: {d.get('mgnIsoMode')}")
    else:
        print(f"  查询失败: {cfg}")
except Exception as e:
    print(f"  异常: {e}")


print("=" * 70)
print("[1] 当前持仓")
print("=" * 70)
ps = k.get_positions()
if ps:
    for p in ps:
        print(f"  {p.get('instId'):20s} {p.get('posSide'):6s} qty={p.get('pos'):>10s} "
              f"entry={p.get('avgPx'):>10s} upl={p.get('upl'):>8s} "
              f"margin={p.get('margin'):>8s} lever={p.get('lever')}x")
else:
    print("  无持仓")

print()
print("=" * 70)
print("[2] 最近20条订单历史（已成交）")
print("=" * 70)
hs = k.get_order_history(limit=20)
if hs:
    for o in hs[:20]:
        print(f"  {o.get('instId'):20s} side={o.get('side'):4s} posSide={o.get('posSide'):6s} "
              f"state={o.get('state'):10s} qty={o.get('sz'):>10s} "
              f"fillPx={o.get('avgPx'):>10s} pnl={o.get('pnl'):>8s} "
              f"fillTime={o.get('fillTime')}")
else:
    print("  无订单历史")

print()
print("=" * 70)
print("[3] 挂单（pending orders）")
print("=" * 70)
po = k.get_orders("SWAP")
if po:
    for o in po[:20]:
        print(f"  {o.get('instId'):20s} side={o.get('side'):4s} posSide={o.get('posSide'):6s} "
              f"state={o.get('state'):10s} qty={o.get('sz'):>10s} px={o.get('px'):>10s} "
              f"ordType={o.get('ordType')}")
else:
    print("  无挂单")

print()
print("=" * 70)
print("[4] 账户信息")
print("=" * 70)
a = k.get_account_info()
if a:
    print(f"  totalEq={a.get('totalEq')} upl={a.get('upl')} "
          f"availBal={a.get('availBal')} ordFroz={a.get('ordFroz')} "
          f"mgnRatio={a.get('mgnRatio')}")
    details = a.get("details", [])
    for d in details[:5]:
        print(f"  ccy={d.get('ccy')} eq={d.get('eq')} availBal={d.get('availBal')} "
              f"frozenBal={d.get('frozenBal')}")
else:
    print("  无法获取账户信息")

print()
print("=" * 70)
print("[5] SQLite数据库交易记录")
print("=" * 70)
try:
    conn = sqlite3.connect("data/trading.db")
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    print(f"Tables: {tables}")

    if "trades" in tables:
        cur.execute("SELECT COUNT(*) FROM trades")
        print(f"trades记录数: {cur.fetchone()[0]}")
        cur.execute("SELECT * FROM trades ORDER BY rowid DESC LIMIT 10")
        cols = [d[0] for d in cur.description]
        print(f"字段: {cols}")
        for row in cur.fetchall():
            print(f"  {row}")

    if "orders" in tables:
        cur.execute("SELECT COUNT(*) FROM orders")
        print(f"orders记录数: {cur.fetchone()[0]}")

    if "equity_history" in tables:
        cur.execute("SELECT COUNT(*) FROM equity_history")
        print(f"equity_history记录数: {cur.fetchone()[0]}")
        cur.execute("SELECT * FROM equity_history ORDER BY rowid DESC LIMIT 5")
        for row in cur.fetchall():
            print(f"  {row}")

    for tbl in ["trade_records", "position_history", "account_history", "equity_curve", "risk_events", "strategy_performance"]:
        if tbl in tables:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            print(f"{tbl}记录数: {cur.fetchone()[0]}")
            if tbl == "trade_records":
                cur.execute(f"SELECT * FROM {tbl} ORDER BY rowid DESC LIMIT 10")
                cols = [d[0] for d in cur.description]
                print(f"  字段: {cols}")
                for row in cur.fetchall():
                    print(f"  {row}")
    conn.close()
except Exception as e:
    print(f"SQLite查询失败: {e}")

