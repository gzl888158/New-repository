"""分析当前仓位及历史开单"""
import sys, os, sqlite3, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.settings import load_config
from core.okx_client import OKXClient

c = load_config()
k = OKXClient(c)

# ============================================================
# [0] 账户配置
# ============================================================
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

# ============================================================
# [1] 当前持仓
# ============================================================
print()
print("=" * 70)
print("[1] 当前持仓")
print("=" * 70)
ps = k.get_positions()
if ps:
    total_upl = 0.0
    total_margin = 0.0
    for p in ps:
        inst_id = p.get("instId", "")
        pos_side = p.get("posSide", "")
        qty = p.get("pos", "0")
        avg_px = p.get("avgPx", "0")
        upl = p.get("upl", "0")
        margin = p.get("margin", "0")
        lever = p.get("lever", "0")
        total_upl += float(upl) if upl else 0
        total_margin += float(margin) if margin else 0
        print(f"  {inst_id:20s} {pos_side:6s} qty={qty:>10s} "
              f"entry={avg_px:>10s} upl={upl:>8s} "
              f"margin={margin:>8s} lever={lever}x")
    print(f"  ---")
    print(f"  总浮盈: {total_upl:.4f} USDT  总保证金: {total_margin:.4f} USDT")
else:
    print("  无持仓")

# ============================================================
# [2] 最近50条订单历史（已成交）
# ============================================================
print()
print("=" * 70)
print("[2] 最近50条订单历史（已成交）")
print("=" * 70)
hs = k.get_order_history(limit=50)
if hs:
    total_pnl = 0.0
    buy_count = 0
    sell_count = 0
    by_symbol = {}
    for o in hs[:50]:
        inst_id = o.get("instId", "")
        side = o.get("side", "")
        pos_side = o.get("posSide", "")
        state = o.get("state", "")
        sz = o.get("sz", "")
        avg_px = o.get("avgPx", "")
        pnl = o.get("pnl", "0")
        fill_time = o.get("fillTime", "")
        if pnl:
            total_pnl += float(pnl)
        if side == "buy":
            buy_count += 1
        elif side == "sell":
            sell_count += 1
        if inst_id not in by_symbol:
            by_symbol[inst_id] = {"count": 0, "pnl": 0.0}
        by_symbol[inst_id]["count"] += 1
        if pnl:
            by_symbol[inst_id]["pnl"] += float(pnl)
        print(f"  {fill_time} {inst_id:20s} {side:4s} {pos_side:6s} "
              f"{state:10s} qty={sz:>10s} "
              f"fillPx={avg_px:>10s} pnl={pnl:>8s}")
    print(f"  ---")
    print(f"  总盈亏: {total_pnl:.4f} USDT  |  买入: {buy_count}笔  卖出: {sell_count}笔")
    print(f"  按币种统计:")
    for sym, info in sorted(by_symbol.items()):
        print(f"    {sym}: {info['count']}笔, PnL={info['pnl']:.4f}")
else:
    print("  无订单历史")

# ============================================================
# [3] 挂单（pending orders）
# ============================================================
print()
print("=" * 70)
print("[3] 挂单（pending orders）")
print("=" * 70)
po = k.get_orders("SWAP")
if po:
    for o in po[:20]:
        print(f"  {o.get('instId',''):20s} side={o.get('side',''):4s} posSide={o.get('posSide',''):6s} "
              f"state={o.get('state',''):10s} qty={o.get('sz',''):>10s} px={o.get('px',''):>10s} "
              f"ordType={o.get('ordType','')}")
else:
    print("  无挂单")

# ============================================================
# [4] 账户信息
# ============================================================
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

# ============================================================
# [5] SQLite数据库交易记录
# ============================================================
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

    for tbl in ["trade_records", "position_history", "account_history", "equity_curve", "risk_events", "strategy_performance"]:
        if tbl in tables:
            cur.execute(f"SELECT COUNT(*) FROM {tbl}")
            print(f"{tbl}记录数: {cur.fetchone()[0]}")
    conn.close()
except Exception as e:
    print(f"SQLite查询失败: {e}")

# ============================================================
# [6] 策略状态文件
# ============================================================
print()
print("=" * 70)
print("[6] 策略状态 (data/strategy_state/)")
print("=" * 70)
state_dir = "data/strategy_state"
if os.path.exists(state_dir):
    for fname in sorted(os.listdir(state_dir)):
        if fname.endswith(".json"):
            fpath = os.path.join(state_dir, fname)
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                print(f"  [{fname}]")
                if isinstance(data, dict):
                    # 显示关键字段
                    for key in ["virtual_positions", "active_positions", "positions", "pending_orders", "trailing_state"]:
                        if key in data:
                            val = data[key]
                            if isinstance(val, dict):
                                print(f"    {key}: {len(val)} entries")
                                for k, v in list(val.items())[:5]:
                                    print(f"      - {k}: {v}")
                            elif isinstance(val, list):
                                print(f"    {key}: {len(val)} items")
                            else:
                                print(f"    {key}: {val}")
                    # 显示其他关键字段
                    for key in ["last_update", "timestamp", "version", "status"]:
                        if key in data:
                            print(f"    {key}: {data[key]}")
                print()
            except Exception as e:
                print(f"  [{fname}] 读取失败: {e}")
else:
    print("  策略状态目录不存在")

# ============================================================
# [7] global_state.json
# ============================================================
print()
print("=" * 70)
print("[7] 全局状态 (data/global_state.json)")
print("=" * 70)
try:
    with open("data/global_state.json", "r") as f:
        gs = json.load(f)
    state = gs.get("state", {})
    for k, v in state.items():
        print(f"  {k}: {v}")
    print(f"  timestamp: {gs.get('timestamp')}")
except Exception as e:
    print(f"  读取失败: {e}")

# ============================================================
# [8] 资金费率历史
# ============================================================
print()
print("=" * 70)
print("[8] 最近资金费率结算")
print("=" * 70)
try:
    bills = k.get_bills(limit=20)
    if bills:
        funding_bills = [b for b in bills if b.get("type") == "8"]
        if funding_bills:
            total_funding = 0.0
            for b in funding_bills[:10]:
                amt = float(b.get("px", 0))
                total_funding += amt
                print(f"  {b.get('ts','')} {b.get('instId',''):20s} "
                      f"funding={amt:>10.6f} {b.get('ccy','')}")
            print(f"  累计资金费率: {total_funding:.6f} USDT")
        else:
            print("  无资金费率结算记录")
    else:
        print("  无法获取账单")
except Exception as e:
    print(f"  查询失败: {e}")