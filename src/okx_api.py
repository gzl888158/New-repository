import requests
import time
import hmac
import base64
from urllib.parse import urljoin
import json
from .utils import request_retry

# -------------------------- 基础配置 --------------------------
def get_okx_domain(env):
    if env == "实盘":
        return "https://www.okx.com"
    elif env == "模拟盘":
        return "https://www.okx.cab"
    else:
        raise ValueError(f"无效环境：{env}")

def sign_request(api_secret, method, path, timestamp, body=""):
    message = f"{timestamp}{method.upper()}{path}{body}"
    mac = hmac.new(api_secret.encode("utf-8"), message.encode("utf-8"), digestmod="sha256")
    return base64.b64encode(mac.digest()).decode("utf-8")

# -------------------------- 核心API（集成重试） --------------------------
@request_retry(retry_times=3, retry_delay=3)
def verify_api(api_key, api_secret, api_passphrase, instId, env):
    domain = get_okx_domain(env)
    path = "/api/v5/account/info"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    sign = sign_request(api_secret, "GET", path, timestamp)
    
    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": api_passphrase,
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"API验证失败：{data['msg']}（{data['code']}）")
    return data["data"][0]["uid"]

@request_retry(retry_times=3, retry_delay=3)
def fetch_ticker(instId, env):
    domain = get_okx_domain(env)
    path = f"/api/v5/market/ticker?instId={instId}"
    url = urljoin(domain, path)
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"获取行情失败：{data['msg']}")
    ticker = data["data"][0]
    return {
        "last": float(ticker["last"]),
        "high": float(ticker["high24h"]),
        "low": float(ticker["low24h"]),
        "volume": float(ticker["vol24h"]),
        "timestamp": int(ticker["ts"]) / 1000
    }

@request_retry(retry_times=3, retry_delay=3)
def fetch_candles(instId, bar, limit, env):
    domain = get_okx_domain(env)
    path = f"/api/v5/market/candles?instId={instId}&bar={bar}&limit={limit}"
    url = urljoin(domain, path)
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"获取K线失败：{data['msg']}")
    candles = []
    for item in data["data"][::-1]:
        candles.append({
            "timestamp": int(item[0]) / 1000,
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5])
        })
    return candles

@request_retry(retry_times=3, retry_delay=3)
def get_account_info(api_key, api_secret, api_passphrase, env):
    domain = get_okx_domain(env)
    path = "/api/v5/account/balance"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    sign = sign_request(api_secret, "GET", path, timestamp)
    
    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": api_passphrase,
        "Content-Type": "application/json"
    }
    
    response = requests.get(url, headers=headers, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"获取账户信息失败：{data['msg']}")
    for asset in data["data"][0]["details"]:
        if asset["ccy"] == "USDT":
            return {
                "available": float(asset["availBal"]),
                "frozen": float(asset["frozenBal"]),
                "total": float(asset["availBal"]) + float(asset["frozenBal"])
            }
    raise Exception("账户无USDT资产")

@request_retry(retry_times=3, retry_delay=3)
def place_grid_orders(instId, buy_levels, sell_levels, volume, api_key, api_secret, api_passphrase, env, leverage=10):
    domain = get_okx_domain(env)
    path = "/api/v5/trade/order"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    
    orders = []
    order_ids = []
    # 买单
    for price in buy_levels:
        orders.append({
            "instId": instId,
            "side": "buy",
            "ordType": "limit",
            "px": str(price),
            "sz": str(volume),
            "lever": str(leverage),
            "posSide": "net",
            "timeInForce": "GTC"  # 永久有效订单
        })
    # 卖单
    for price in sell_levels:
        orders.append({
            "instId": instId,
            "side": "sell",
            "ordType": "limit",
            "px": str(price),
            "sz": str(volume),
            "lever": str(leverage),
            "posSide": "net",
            "timeInForce": "GTC"
        })
    
    # 批量下单（每次20个）
    for i in range(0, len(orders), 20):
        batch = orders[i:i+20]
        body = json.dumps(batch)
        sign = sign_request(api_secret, "POST", path, timestamp, body)
        headers = {
            "OK-ACCESS-KEY": api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": api_passphrase,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=body, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["code"] != "0":
            raise Exception(f"挂单失败：{data['msg']}")
        # 收集订单ID
        for ord in data["data"]:
            order_ids.append(ord["ordId"])
    return order_ids

@request_retry(retry_times=3, retry_delay=3)
def cancel_orders(instId, order_ids, api_key, api_secret, api_passphrase, env):
    """取消指定订单"""
    domain = get_okx_domain(env)
    path = "/api/v5/trade/cancel-batch-orders"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    
    # 批量取消（每次20个）
    for i in range(0, len(order_ids), 20):
        batch_ids = order_ids[i:i+20]
        body = json.dumps([{"instId": instId, "ordId": ordId} for ordId in batch_ids])
        sign = sign_request(api_secret, "POST", path, timestamp, body)
        headers = {
            "OK-ACCESS-KEY": api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": api_passphrase,
            "Content-Type": "application/json"
        }
        response = requests.post(url, headers=headers, data=body, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["code"] != "0":
            raise Exception(f"取消订单失败：{data['msg']}")

@request_retry(retry_times=3, retry_delay=3)
def cancel_all_orders(instId, api_key, api_secret, api_passphrase, env):
    """取消所有订单"""
    domain = get_okx_domain(env)
    path = f"/api/v5/trade/cancel-batch-orders?instId={instId}"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    sign = sign_request(api_secret, "POST", path, timestamp)
    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": api_passphrase,
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"取消所有订单失败：{data['msg']}")

@request_retry(retry_times=3, retry_delay=3)
def query_order_status(instId, order_ids, api_key, api_secret, api_passphrase, env):
    """查询订单状态（已成交/未成交/超时）"""
    domain = get_okx_domain(env)
    path = "/api/v5/trade/orders"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    # 拼接订单ID
    ord_ids = ",".join(order_ids)
    params = {"instId": instId, "ordId": ord_ids}
    sign = sign_request(api_secret, "GET", path, timestamp)
    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": api_passphrase,
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers, params=params, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"查询订单状态失败：{data['msg']}")
    # 格式化结果：ordId -> 状态（filled/partially_filled/open/canceled）
    result = {}
    for ord in data["data"]:
        result[ord["ordId"]] = {
            "status": ord["state"],
            "filled_volume": float(ord["accFillSz"]),
            "filled_price": float(ord["avgPx"]) if ord["avgPx"] else 0.0,
            "pnl": float(ord["pnl"]) if ord["pnl"] else 0.0
        }
    return result

@request_retry(retry_times=3, retry_delay=3)
def get_position_pnl(instId, api_key, api_secret, api_passphrase, env):
    """查询当前仓位盈亏"""
    domain = get_okx_domain(env)
    path = "/api/v5/account/positions"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    params = {"instId": instId}
    sign = sign_request(api_secret, "GET", path, timestamp)
    headers = {
        "OK-ACCESS-KEY": api_key,
        "OK-ACCESS-SIGN": sign,
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": api_passphrase,
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers, params=params, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"查询仓位盈亏失败：{data['msg']}")
    for pos in data["data"]:
        if pos["instId"] == instId:
            return {
                "floating_pnl": float(pos["unRealizedPnl"]),  # 浮动盈亏
                "realized_pnl": float(pos["realizedPnl"]),    # 已实现盈亏
                "position_volume": float(pos["pos"]),         # 持仓量
                "entry_price": float(pos["avgPx"])            # 开仓均价
            }
    return {"floating_pnl": 0.0, "realized_pnl": 0.0, "position_volume": 0.0, "entry_price": 0.0}

# -------------------------- ATR计算（无修改） --------------------------
def calculate_atr(candles, period=14):
    if len(candles) < period:
        raise Exception(f"K线不足{period}根，无法计算ATR")
    tr_list = []
    for i in range(1, len(candles)):
        current = candles[i]
        prev = candles[i-1]
        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - prev["close"]),
            abs(current["low"] - prev["close"])
        )
        tr_list.append(tr)
    return round(sum(tr_list[-period:])/period, 4)