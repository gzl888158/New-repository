# -*- coding: utf-8 -*-
"""诊断：8个API密钥是否映射到不同子账户（各自持仓/余额是否不同）"""
import os, sys, json, time, base64, hashlib, hmac
from datetime import datetime
sys.path.insert(0, r'e:\新建文件夹\okx_quant_trading')
os.chdir(r'e:\新建文件夹\okx_quant_trading')

from configs.settings import load_config
from core.okx_client import OKXClient

c = load_config()
k = OKXClient(c)

keys = c["okx"].get("api_keys", [])
rest_url = c["okx"]["rest_url"]
proxy = c["okx"].get("proxy")

import requests
sess = requests.Session()
if proxy:
    sess.proxies = {"http": proxy, "https": proxy}

def sign(ts, method, path, body, secret):
    msg = f"{ts}{method}{path}{body}"
    return base64.b64encode(hmac.new(secret.encode(), msg.encode(), hashlib.sha256).digest()).decode()

def query(idx, key):
    path = "/api/v5/account/positions"
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    sig = sign(ts, "GET", path, "", key["secret_key"])
    headers = {
        "OK-ACCESS-KEY": key["api_key"],
        "OK-ACCESS-SIGN": sig,
        "OK-ACCESS-TIMESTAMP": ts,
        "OK-ACCESS-PASSPHRASE": key["passphrase"],
        "Content-Type": "application/json",
    }
    r = sess.get(f"{rest_url}{path}", headers=headers)
    d = r.json()
    if d.get("code") != "0":
        return f"[key {idx}] ERROR code={d.get('code')} msg={d.get('msg')}"
    lines = []
    for p in d.get("data", []):
        try:
            qty = float(p.get("pos", "0") or 0)
        except Exception:
            qty = 0.0
        if qty != 0:
            lines.append(f"{p.get('instId')}:{p.get('posSide')}:{p.get('pos')}")
    return f"[key {idx}] {len(lines)} positions: {', '.join(lines) if lines else '(empty)'}"

print("=" * 70)
print(f"共 {len(keys)} 个密钥，逐一查询持仓（判断是否为不同子账户）")
print("=" * 70)
for i, key in enumerate(keys):
    try:
        print(query(i, key))
    except Exception as e:
        print(f"[key {i}] EXCEPTION: {e}")

k.close()
