import os
import json
import requests
import hmac
import hashlib
import base64
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, List, Any
from .utils import setup_logger

# 加载环境变量
load_dotenv()
logger = setup_logger()

# 欧易API配置
OKX_API_KEY = os.getenv("OKX_API_KEY")
OKX_SECRET_KEY = os.getenv("OKX_SECRET_KEY")
OKX_PASSPHRASE = os.getenv("OKX_PASSPHRASE")
OKX_MARKET_URL = "https://www.okx.com/api/v5/market"
OKX_TRADE_URL = "https://www.okx.com/api/v5/trade"
OKX_ACCOUNT_URL = "https://www.okx.com/api/v5/account"

def get_okx_signature(timestamp: str, method: str, request_path: str, body: str = "") -> str:
    """生成欧易API签名"""
    message = timestamp + method.upper() + request_path + body
    mac = hmac.new(
        bytes(OKX_SECRET_KEY, encoding="utf8"),
        bytes(message, encoding="utf8"),
        hashlib.sha256
    )
    return base64.b64encode(mac.digest()).decode("utf-8")

def get_okx_headers(request_path: str, body: str = "", method: str = "GET") -> Dict[str, str]:
    """生成欧易API请求头"""
    timestamp = datetime.utcnow().isoformat(timespec="milliseconds").replace("+00:00", "Z")
    return {
        "OK-ACCESS-KEY": OKX_API_KEY,
        "OK-ACCESS-SIGN": get_okx_signature(timestamp, method, request_path, body),
        "OK-ACCESS-TIMESTAMP": timestamp,
        "OK-ACCESS-PASSPHRASE": OKX_PASSPHRASE,
        "Content-Type": "application/json"
    }

def fetch_candles(inst_id: str, bar: str, limit: int = 100, end_ts: int = None) -> List[Dict[str, Any]]:
    """拉取欧易K线数据"""
    url = f"{OKX_MARKET_URL}/candles"
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    if end_ts:
        params["after"] = end_ts
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data["code"] != "0":
            logger.error(f"拉取K线失败: {data.get('msg')}")
            return []
        candles = []
        for item in data["data"]:
            candles.append({
                "ts": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "vol": float(item[5])
            })
        return sorted(candles, key=lambda x: x["ts"])
    except Exception as e:
        logger.error(f"拉取K线异常: {str(e)}")
        return []

def get_account_balance(ccy: str = "USDT") -> float:
    """获取账户可用余额"""
    url = f"{OKX_ACCOUNT_URL}/balance"
    params = {"ccy": ccy}
    headers = get_okx_headers("/api/v5/account/balance")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        if data["code"] == "0" and len(data["data"]) > 0:
            for detail in data["data"][0]["details"]:
                if detail["ccy"] == ccy:
                    return float(detail["availBal"])
        logger.error(f"获取余额失败: {data.get('msg')}")
        return 0.0
    except Exception as e:
        logger.error(f"获取余额异常: {str(e)}")
        return 0.0

def get_position(inst_id: str) -> Dict[str, Any]:
    """获取指定合约持仓"""
    url = f"{OKX_ACCOUNT_URL}/positions"
    params = {"instId": inst_id}
    headers = get_okx_headers("/api/v5/account/positions")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        if data["code"] == "0" and len(data["data"]) > 0:
            pos = data["data"][0]
            return {
                "instId": pos["instId"],
                "posSide": pos["posSide"],
                "posAmt": float(pos["posAmt"]),
                "avgPx": float(pos["avgPx"]),
                "upl": float(pos["upl"]),
                "liqPx": float(pos["liqPx"])
            }
        return {"instId": inst_id, "posSide": "net", "posAmt": 0.0}
    except Exception as e:
        logger.error(f"获取持仓异常: {str(e)}")
        return {"instId": inst_id, "posSide": "net", "posAmt": 0.0}

def place_order(inst_id: str, pos_side: str, vol: float, tp_price: float, sl_price: float) -> Dict[str, Any]:
    """下单（市价单+止盈止损）"""
    url = f"{OKX_TRADE_URL}/order"
    method = "POST"
    side = "buy" if pos_side == "long" else "sell"
    body = json.dumps({
        "instId": inst_id,
        "tdMode": "isolated",
        "side": side,
        "posSide": pos_side,
        "ordType": "market",
        "sz": str(round(vol, 4)),
        "tpTriggerPx": str(tp_price),
        "tpOrdPx": str(tp_price),
        "slTriggerPx": str(sl_price),
        "slOrdPx": str(sl_price)
    })
    headers = get_okx_headers("/api/v5/trade/order", body, method)
    try:
        response = requests.post(url, headers=headers, data=body, timeout=10)
        data = response.json()
        if data["code"] == "0":
            logger.info(f"下单成功: {inst_id} {pos_side} | 订单号: {data['data'][0]['ordId']}")
            return {"status": "success", "ordId": data["data"][0]["ordId"]}
        logger.error(f"下单失败: {data.get('msg')}")
        return {"status": "failed", "msg": data.get("msg")}
    except Exception as e:
        logger.error(f"下单异常: {str(e)}")
        return {"status": "failed", "msg": str(e)}