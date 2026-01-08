import json
import time
import requests
import numpy as np
import yaml
from datetime import datetime
from Crypto.Hash import SHA256
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
import base64
import hmac

# 加载配置
with open("config/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

def get_okx_headers(method: str, path: str, body: str, api_key: str, api_secret: str, passphrase: str, env: str) -> dict:
    """生成欧易API请求头（含签名）"""
    timestamp = datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
    message = f"{timestamp}{method}{path}{body}"
    signature = base64.b64encode(hmac.new(api_secret.encode(), message.encode(), SHA256).digest()).decode()
    base_url = CONFIG["okx"]["real_base_url"] if env == "real" else CONFIG["okx"]["sim_base_url"]
    return {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": signature,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": passphrase,
        "Content-Type": "application/json",
        "x-simulated-trading": "1" if env == "sim" else "0"
    }

def verify_api(api_key: str, api_secret: str, passphrase: str, inst_id: str, env: str) -> str:
    """验证API有效性"""
    base_url = CONFIG["okx"]["real_base_url"] if env == "real" else CONFIG["okx"]["sim_base_url"]
    path = "/api/v5/account/account"
    headers = get_okx_headers("GET", path, "", api_key, api_secret, passphrase, env)
    resp = requests.get(f"{base_url}{path}", headers=headers, timeout=CONFIG["okx"]["timeout"])
    data = resp.json()
    if data["code"] != "0":
        raise Exception(f"API错误：{data['msg']}")
    return data["data"][0]["uid"]

def fetch_ticker(inst_id: str, env: str) -> dict:
    """获取实时行情"""
    base_url = CONFIG["okx"]["real_base_url"] if env == "real" else CONFIG["okx"]["sim_base_url"]
    resp = requests.get(f"{base_url}/api/v5/market/ticker?instId={inst_id}", timeout=CONFIG["okx"]["timeout"])
    data = resp.json()["data"][0]
    return {
        "instId": data["instId"],
        "last": data["last"],
        "change": f"{float(data['sodUtc0']) * 100:.2f}"
    }

def fetch_candles(inst_id: str, bar: str, limit: int, env: str) -> list:
    """获取K线数据"""
    base_url = CONFIG["okx"]["real_base_url"] if env == "real" else CONFIG["okx"]["sim_base_url"]
    resp = requests.get(
        f"{base_url}/api/v5/market/history-candles?instId={inst_id}&bar={bar}&limit={limit}",
        timeout=CONFIG["okx"]["timeout"]
    )
    data = resp.json()["data"]
    return [
        {
            "high": float(c[2]), "low": float(c[3]),
            "close": float(c[4]), "volume": float(c[5])
        } for c in reversed(data)
    ]

def calculate_atr(candles: list, period: int) -> float:
    """计算ATR指标"""
    if len(candles) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(candles)):
        tr = max(
            candles[i]["high"] - candles[i]["low"],
            abs(candles[i]["high"] - candles[i-1]["close"]),
            abs(candles[i]["low"] - candles[i-1]["close"])
        )
        tr_list.append(tr)
    return round(np.mean(tr_list[-period:]), 4)

def place_grid_orders(inst_id: str, buy_levels: list, sell_levels: list, volume: float, api_key: str, api_secret: str, passphrase: str, env: str) -> dict:
    """挂网格买单和卖单"""
    base_url = CONFIG["okx"]["real_base_url"] if env == "real" else CONFIG["okx"]["sim_base_url"]
    path = "/api/v5/trade/order"
    headers = get_okx_headers("POST", path, "", api_key, api_secret, passphrase, env)
    orders = []
    # 挂买单
    for price in buy_levels:
        body = json.dumps({
            "instId": inst_id,
            "tdMode": "isolated",
            "side": "buy",
            "posSide": "long",
            "ordType": "limit",
            "px": str(price),
            "sz": str(round(volume / len(buy_levels), 4))
        })
        resp = requests.post(f"{base_url}{path}", headers=headers, data=body, timeout=CONFIG["okx"]["timeout"])
        data = resp.json()
        if data["code"] != "0":
            raise Exception(f"挂单失败：{data['msg']}")
        orders.append({"side": "buy", "price": price, "ordId": data["data"][0]["ordId"]})
    # 挂卖单
    for price in sell_levels:
        body = json.dumps({
            "instId": inst_id,
            "tdMode": "isolated",
            "side": "sell",
            "posSide": "long",
            "ordType": "limit",
            "px": str(price),
            "sz": str(round(volume / len(sell_levels), 4))
        })
        resp = requests.post(f"{base_url}{path}", headers=headers, data=body, timeout=CONFIG["okx"]["timeout"])
        data = resp.json()
        if data["code"] != "0":
            raise Exception(f"挂单失败：{data['msg']}")
        orders.append({"side": "sell", "price": price, "ordId": data["data"][0]["ordId"]})
    return {"status": "success", "orders": orders}

def cancel_all_orders(inst_id: str, api_key: str, api_secret: str, passphrase: str, env: str) -> dict:
    """取消所有挂单"""
    base_url = CONFIG["okx"]["real_base_url"] if env == "real" else CONFIG["okx"]["sim_base_url"]
    path = "/api/v5/trade/cancel-batch-orders"
    body = json.dumps({"instId": inst_id})
    headers = get_okx_headers("POST", path, body, api_key, api_secret, passphrase, env)
    resp = requests.post(f"{base_url}{path}", headers=headers, data=body, timeout=CONFIG["okx"]["timeout"])
    data = resp.json()
    if data["code"] != "0":
        raise Exception(f"撤单失败：{data['msg']}")
    return {"status": "success"}

def get_account_info(api_key: str, api_secret: str, passphrase: str, env: str) -> dict:
    """获取账户信息"""
    base_url = CONFIG["okx"]["real_base_url"] if env == "real" else CONFIG["okx"]["sim_base_url"]
    path = "/api/v5/account/balance"
    headers = get_okx_headers("GET", path, "", api_key, api_secret, passphrase, env)
    resp = requests.get(f"{base_url}{path}", headers=headers, timeout=CONFIG["okx"]["timeout"])
    data = resp.json()["data"][0]["details"][0]
    return {"available": data["availBal"], "balance": data["bal"]}