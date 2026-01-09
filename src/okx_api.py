import requests
import time
import hmac
import base64
from urllib.parse import urljoin
import json

# -------------------------- 【新增】域名切换逻辑 --------------------------
def get_okx_domain(env):
    """根据环境返回欧易API域名"""
    if env == "实盘":
        return "https://www.okx.com"
    elif env == "模拟盘":
        return "https://www.okx.cab"
    else:
        raise ValueError(f"无效环境：{env}（仅支持'实盘'或'模拟盘'）")

# -------------------------- 【修改】完善签名逻辑 --------------------------
def sign_request(api_secret, method, path, timestamp, body=""):
    """生成欧易API签名（支持GET/POST请求）"""
    message = f"{timestamp}{method.upper()}{path}{body}"
    mac = hmac.new(api_secret.encode("utf-8"), message.encode("utf-8"), digestmod="sha256")
    return base64.b64encode(mac.digest()).decode("utf-8")

# 验证API并获取UID
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
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"API验证失败：{data['msg']}（错误码：{data['code']}）")
    return data["data"][0]["uid"]

# 获取行情数据
def fetch_ticker(instId, env):
    domain = get_okx_domain(env)
    path = f"/api/v5/market/ticker?instId={instId}"
    url = urljoin(domain, path)
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"获取行情失败：{data['msg']}")
    ticker = data["data"][0]
    return {
        "last": float(ticker["last"]),
        "high": float(ticker["high24h"]),
        "low": float(ticker["low24h"]),
        "volume": float(ticker["vol24h"])
    }

# -------------------------- 【修改】修复K线接口路径 --------------------------
def fetch_candles(instId, bar, limit, env):
    """获取K线数据（路径修正为欧易官方接口）"""
    domain = get_okx_domain(env)
    path = f"/api/v5/market/candles?instId={instId}&bar={bar}&limit={limit}"
    url = urljoin(domain, path)
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"获取K线失败：{data['msg']}")
    # 格式化K线数据（时间戳转datetime，价格转float）
    candles = []
    for item in data["data"][::-1]:  # 倒序排列（最新数据在最后）
        candles.append({
            "timestamp": datetime.fromtimestamp(int(item[0])/1000),
            "open": float(item[1]),
            "high": float(item[2]),
            "low": float(item[3]),
            "close": float(item[4]),
            "volume": float(item[5])
        })
    return candles

# 计算ATR指标
def calculate_atr(candles, period=14):
    if len(candles) < period:
        raise Exception(f"K线数据不足{period}根，无法计算ATR")
    tr_list = []
    for i in range(1, len(candles)):
        current = candles[i]
        previous = candles[i-1]
        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"])
        )
        tr_list.append(tr)
    atr = np.mean(tr_list[-period:])
    return round(atr, 4)

# 获取账户信息
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
    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"获取账户信息失败：{data['msg']}")
    # 返回USDT可用余额
    for asset in data["data"][0]["details"]:
        if asset["ccy"] == "USDT":
            return {"available": asset["availBal"], "frozen": asset["frozenBal"]}
    raise Exception("账户中未找到USDT资产")

# 挂网格订单
def place_grid_orders(instId, buy_levels, sell_levels, volume, api_key, api_secret, api_passphrase, env):
    domain = get_okx_domain(env)
    path = "/api/v5/trade/order"
    url = urljoin(domain, path)
    timestamp = str(int(time.time() * 1000))
    
    # 构造买单和卖单
    orders = []
    # 买单（限价，低于当前价）
    for price in buy_levels:
        orders.append({
            "instId": instId,
            "side": "buy",
            "ordType": "limit",
            "px": str(price),
            "sz": str(volume),
            "lever": "10",  # 杠杆倍数（与config.yaml一致）
            "posSide": "net"  # 净仓位模式
        })
    # 卖单（限价，高于当前价）
    for price in sell_levels:
        orders.append({
            "instId": instId,
            "side": "sell",
            "ordType": "limit",
            "px": str(price),
            "sz": str(volume),
            "lever": "10",
            "posSide": "net"
        })
    
    # 批量下单（每次最多20个订单）
    for i in range(0, len(orders), 20):
        batch_orders = orders[i:i+20]
        body = json.dumps(batch_orders)
        sign = sign_request(api_secret, "POST", path, timestamp, body)
        
        headers = {
            "OK-ACCESS-KEY": api_key,
            "OK-ACCESS-SIGN": sign,
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": api_passphrase,
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, data=body)
        response.raise_for_status()
        data = response.json()
        if data["code"] != "0":
            raise Exception(f"挂单失败：{data['msg']}（订单组：{i//20 + 1}）")

# 取消所有订单
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
    
    response = requests.post(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    if data["code"] != "0":
        raise Exception(f"取消订单失败：{data['msg']}")