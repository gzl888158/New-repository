# -*- coding: utf-8 -*-
"""诊断 GPS-USDT-SWAP 51169 根因：账户 posMode + 原始持仓 posSide + 挂单"""
import os, sys, json
sys.path.insert(0, r'e:\新建文件夹\okx_quant_trading')
os.chdir(r'e:\新建文件夹\okx_quant_trading')

from configs.settings import load_config
from core.okx_client import OKXClient

c = load_config()
k = OKXClient(c)

print("=" * 70)
print("[1] 账户配置 posMode")
print("=" * 70)
try:
    path = "/api/v5/account/config"
    headers = k._get_headers("GET", path)
    r = k._session.get(f"{k.rest_url}{path}", headers=headers)
    cfg = r.json()
    if cfg.get("code") == "0" and cfg.get("data"):
        d = cfg["data"][0]
        print(f"  posMode    = {d.get('posMode')}")
        print(f"  acctLv     = {d.get('acctLv')}")
    else:
        print(f"  查询失败: {json.dumps(cfg, ensure_ascii=False)}")
except Exception as e:
    print(f"  异常: {e}")

print()
print("=" * 70)
print("[2] 当前所有持仓 (pos != 0)")
print("=" * 70)
try:
    path = "/api/v5/account/positions"
    headers = k._get_headers("GET", path)
    r = k._session.get(f"{k.rest_url}{path}", headers=headers)
    data = r.json()
    if data.get("code") == "0" and data.get("data"):
        for p in data["data"]:
            try:
                qty = float(p.get("pos", "0") or 0)
            except Exception:
                qty = 0.0
            if qty == 0:
                continue
            inst = p.get("instId", "?")
            if "GPS" in inst:
                print(f"--- {inst} 完整原始字段 ---")
                print(json.dumps(p, ensure_ascii=False, indent=2, default=str))
            else:
                print(f"  {inst:20s} posSide={p.get('posSide'):6s} pos={p.get('pos'):>12s} "
                      f"avgPx={p.get('avgPx'):>10s} lever={p.get('lever'):>4s}")
    else:
        print(f"  查询失败: {json.dumps(data, ensure_ascii=False)[:500]}")
except Exception as e:
    print(f"  异常: {e}")

print()
print("=" * 70)
print("[3] GPS-USDT-SWAP 挂单 (pending)")
print("=" * 70)
try:
    path = "/api/v5/trade/orders-pending"
    headers = k._get_headers("GET", path)
    r = k._session.get(f"{k.rest_url}{path}?instType=SWAP&instId=GPS-USDT-SWAP", headers=headers)
    data = r.json()
    if data.get("code") == "0":
        if data.get("data"):
            for o in data["data"]:
                print(f"  side={o.get('side')} posSide={o.get('posSide')} sz={o.get('sz')} "
                      f"state={o.get('state')} ordType={o.get('ordType')} reduceOnly={o.get('reduceOnly')}")
        else:
            print("  无挂单")
    else:
        print(f"  查询失败: {json.dumps(data, ensure_ascii=False)[:500]}")
except Exception as e:
    print(f"  异常: {e}")

k.close()
