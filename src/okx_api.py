import requests
import time
import hmac
import base64
from urllib.parse import urljoin
import json
from .utils import request_retry

# -------------------------- 基础配置（无修改） --------------------------
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

# -------------------------- 核心API（新增盘口、多币种持仓接口） --------------------------
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
        "timestamp": int(ticker["ts"]) / 1000,
        "volatility": (float(ticker["high24h"]) - float(ticker["low24h"])) / float(ticker["low24h"])  # 24h波动率
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
def fetch_order_book(instId, depth=5, env):
    """获取盘口深度数据（买一到买五、卖一到卖五）"""
    domain = get_okx_domain(env)
    path = f"/api/v5/market/books?instId={instId}&sz={depth}"
    url = urljoin(domain, path)
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"获取盘口数据失败：{data['msg']}")
    book = data["data"][0]
    # 格式化买单（买一到买五，价格从高到低）
    bids = [{"price": float(item[0]), "volume": float(item[1])} for item in book["bids"]]
    # 格式化卖单（卖一到卖五，价格从低到高）
    asks = [{"price": float(item[0]), "volume": float(item[1])} for item in book["asks"]]
    return {"bids": bids, "asks": asks, "timestamp": int(book["ts"]) / 1000}

@request_retry(retry_times=3, retry_delay=3)
def fetch_trades(instId, limit=100, env):
    """获取实时成交明细（最近100笔）"""
    domain = get_okx_domain(env)
    path = f"/api/v5/market/trades?instId={instId}&limit={limit}"
    url = urljoin(domain, path)
    response = requests.get(url, timeout=5)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"获取成交明细失败：{data['msg']}")
    trades = []
    for item in data["data"]:
        trades.append({
            "price": float(item[0]),
            "volume": float(item[1]),
            "side": "buy" if item[2] == "buy" else "sell",
            "timestamp": int(item[3]) / 1000
        })
    return trades

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
    usdt_asset = next((asset for asset in data["data"][0]["details"] if asset["ccy"] == "USDT"), None)
    if not usdt_asset:
        raise Exception("账户无USDT资产")
    return {
        "available": float(usdt_asset["availBal"]),
        "frozen": float(usdt_asset["frozenBal"]),
        "total": float(usdt_asset["availBal"]) + float(usdt_asset["frozenBal"])
    }

@request_retry(retry_times=3, retry_delay=3)
def get_all_positions(api_key, api_secret, api_passphrase, env):
    """获取所有币种持仓信息"""
    domain = get_okx_domain(env)
    path = "/api/v5/account/positions"
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
        raise Exception(f"获取持仓信息失败：{data['msg']}")
    positions = {}
    for pos in data["data"]:
        instId = pos["instId"]
        positions[instId] = {
            "floating_pnl": float(pos["unRealizedPnl"]),
            "realized_pnl": float(pos["realizedPnl"]),
            "position_volume": float(pos["pos"]),
            "entry_price": float(pos["avgPx"]),
            "liquidation_price": float(pos["liqPx"]) if pos["liqPx"] else 0.0  # 爆仓价格
        }
    return positions

@request_retry(retry_times=3, retry_delay=3)
def get_position_risk(api_key, api_secret, api_passphrase, instId, env):
    """获取单个币种爆仓风险"""
    domain = get_okx_domain(env)
    path = f"/api/v5/account/position-risk?instId={instId}"
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
        raise Exception(f"获取爆仓风险失败：{data['msg']}")
    risk = data["data"][0]
    return {
        "liquidation_price": float(risk["liqPx"]) if risk["liqPx"] else 0.0,
        "margin_ratio": float(risk["marginRatio"]),  # 保证金比例
        "available_margin": float(risk["availMargin"])  # 可用保证金
    }

# -------------------------- 订单操作API（无修改，适配多币种） --------------------------
@request_retry(retry_times=3, retry_delay=3)
def place_grid_orders(instId, buy_levels, sell_levels, volume, api_key, api_secret, api_passphrase, env, leverage=10):
    domain = get_okx_domain(env)
    path = "/api/v5/trade/order"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    
    orders = []
    order_ids = []
    for price in buy_levels:
        orders.append({
            "instId": instId,
            "side": "buy",
            "ordType": "limit",
            "px": str(price),
            "sz": str(volume),
            "lever": str(leverage),
            "posSide": "net",
            "timeInForce": "GTC"
        })
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
        for ord in data["data"]:
            order_ids.append(ord["ordId"])
    return order_ids

@request_retry(retry_times=3, retry_delay=3)
def cancel_orders(instId, order_ids, api_key, api_secret, api_passphrase, env):
    domain = get_okx_domain(env)
    path = "/api/v5/trade/cancel-batch-orders"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    
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
    domain = get_okx_domain(env)
    path = "/api/v5/trade/orders"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
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
    result = {}
    for ord in data["data"]:
        result[ord["ordId"]] = {
            "status": ord["state"],
            "filled_volume": float(ord["accFillSz"]),
            "filled_price": float(ord["avgPx"]) if ord["avgPx"] else 0.0,
            "pnl": float(ord["pnl"]) if ord["pnl"] else 0.0
        }
    return result