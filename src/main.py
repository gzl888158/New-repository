import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple, Optional
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import yaml
import os
from dotenv import load_dotenv
from .utils import setup_logger
from .okx_api import (
    fetch_candles, get_account_balance, get_position, place_order,
    get_okx_headers, OKX_MARKET_URL, OKX_TRADE_URL, OKX_ACCOUNT_URL
)

# 初始化
load_dotenv()
logger=setup_logger()
app=FastAPI(title="OKX Grid Trading Robot API")

# 加载配置
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config=yaml.safe_load(f)

# 全局状态（控制策略运行）
ROBOT_RUNNING=False
GRID_PARAMS={}
RISK_STATE={
    "daily_loss": 0.0,
    "consecutive_loss": 0,
    "trade_count": 0
}

# Pydantic模型（前端参数校验）
class GridStartRequest(BaseModel):
    api_key: str
    api_secret: str
    api_passphrase: str
    inst_id: str = "BTC-USDT-SWAP"
    grid_levels: int = 5
    atr_multiplier: float = 1.2
    atr_period: int=14
    leverage: int=config["risk"]["leverage"]
    risk_ratio: float=config["risk"]["risk_ratio"]

# 核心指标计算（复用+优化）
def calculate_atr(candles: List[Dict[str, float]], period: int = 14) -> float:
    if len(candles) < period+1:
        return 0.0
    tr_list = []
    for i in range(1, len(candles)):
        tr=max(candles[i]["high"] - candles[i]["low"],
                 abs(candles[i]["high"] - candles[i-1]["close"]),
                 abs(candles[i]["low"] - candles[i-1]["close"]))
        tr_list.append(tr)
    return round(np.mean(tr_list[-period:]), 4)

# 多周期共振信号（开仓过滤）
def get_multi_timeframe_signal(inst_id: str, api_key: str, api_secret: str, api_passphrase: str) -> str:
    timeframes = ["15m", "1h", "4h"]
    signals = []
    for tf in timeframes:
        candles=fetch_candles(inst_id, tf, config["strategy"]["kline_limit"])
        if len(candles) < 30:
            signals.append("hold")
            continue
        close_prices=np.array([c["close"] for c in candles])
        # 简易MACD金叉死叉判断
        ema_fast=calculate_ema(close_prices, 12)
        ema_slow=calculate_ema(close_prices, 26)
        if len(ema_fast) < 1 or len(ema_slow) < 1:
            signals.append("hold")
            continue
        if ema_fast[-1] > ema_slow[-1] and ema_fast[-2] < ema_slow[-2]:
            signals.append("buy")
        elif ema_fast[-1] < ema_slow[-1] and ema_fast[-2] > ema_slow[-2]:
            signals.append("sell")
        else:
            signals.append("hold")
    # 多周期共振才开仓
    if all(s == "buy" for s in signals):
        return "buy"
    elif all(s == "sell" for s in signals):
        return "sell"
    else:
        return "hold"

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    alpha=2/(period+1)
    ema=np.zeros_like(prices, dtype=float)
    ema[period-1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        ema[i] = alpha*prices[i] + (1-alpha)*ema[i-1]
    return ema

# 网格交易核心策略（低买高卖）
def grid_trading_strategy(params: GridStartRequest):
    global RISK_STATE
    logger.info(f"启动网格交易：{params.inst_id}，层数：{params.grid_levels}，ATR乘数：{params.atr_multiplier}")
    # 1. 获取K线+ATR+最新价格
    candles=fetch_candles(params.inst_id, "15m", config["strategy"]["kline_limit"])
    atr=calculate_atr(candles, params.atr_period)
    last_price=candles[-1]["close"] if candles else 0.0
    if atr == 0 or last_price == 0:
        logger.error("数据异常，ATR/价格为0，停止策略")
        return {"status": "failed", "msg": "数据异常"}

    # 2. 设置网格参数（低买高卖）
    grid_spacing=atr*params.atr_multiplier
    base_price=last_price
    grid_buy_prices = [base_price-grid_spacing*i for i in range(1, params.grid_levels+1)]
    grid_sell_prices = [base_price+grid_spacing*i for i in range(1, params.grid_levels+1)]
    GRID_PARAMS.update({
        "inst_id": params.inst_id,
        "base_price": base_price,
        "grid_spacing": grid_spacing,
        "buy_levels": grid_buy_prices,
        "sell_levels": grid_sell_prices,
        "params": params
    })

    # 3. 风控检查（熔断）
    if not check_risk_control(params.inst_id, params.api_key, params.api_secret, params.api_passphrase):
        logger.error("触发风控熔断，停止策略")
        return {"status": "failed", "msg": "触发风控熔断"}

    # 4. 多周期共振开仓+网格挂单
    signal=get_multi_timeframe_signal(params.inst_id, params.api_key, params.api_secret, params.api_passphrase)
    if signal == "buy":
        # 低买：开多+挂多档买单+挂多档卖单（止盈）
        order_result=place_grid_order(
            params.inst_id, "long", params.leverage, params.risk_ratio,
            grid_buy_prices, grid_sell_prices, params.api_key, params.api_secret, params.api_passphrase
        )
    elif signal == "sell":
        # 高卖：开空+挂多档卖单+挂多档买单（止盈）
        order_result=place_grid_order(
            params.inst_id, "short", params.leverage, params.risk_ratio,
            grid_buy_prices, grid_sell_prices, params.api_key, params.api_secret, params.api_passphrase
        )
    else:
        logger.info("无共振信号，只挂单不开仓")
        order_result={"status": "success", "msg": "无共振信号，挂单完成"}

    return {
        "status": "success",
        "base_price": base_price,
        "grid_spacing": grid_spacing,
        "buy_levels": grid_buy_prices,
        "sell_levels": grid_sell_prices,
        "order_result": order_result
    }

# 风控熔断函数
def check_risk_control(inst_id: str, api_key: str, api_secret: str, api_passphrase: str) -> bool:
    global RISK_STATE
    max_daily_loss=config["risk"]["max_daily_loss"]
    max_consecutive_loss=config["risk"]["max_consecutive_loss"]
    daily_trade_limit=config["risk"]["daily_trade_limit"]

    # 1. 今日交易次数检查
    if RISK_STATE["trade_count"] >= daily_trade_limit:
        return False
    # 2. 单日亏损检查
    if RISK_STATE["daily_loss"] >= max_daily_loss:
        return False
    # 3. 连续亏损检查
    if RISK_STATE["consecutive_loss"] >= max_consecutive_loss:
        return False
    return True

# 网格下单函数（核心低买高卖执行）
def place_grid_order(inst_id: str, pos_side: str, leverage: int, risk_ratio: float,
                     buy_levels: List[float], sell_levels: List[float],
                     api_key: str, api_secret: str, api_passphrase: str) -> Dict:
    # 计算下单量（按风险率）
    balance=get_account_balance("USDT", api_key, api_secret, api_passphrase)
    atr=GRID_PARAMS["grid_spacing"] / GRID_PARAMS["params"].atr_multiplier
    order_vol=(balance*risk_ratio) / (atr*leverage)
    order_vol=max(order_vol, config["risk"]["min_order_volume"]) # 最小下单量

    # 开仓（市价）+ 止盈止损（限价）
    last_price=GRID_PARAMS["base_price"]
    tp_price=last_price+atr*config["strategy"]["atr_tp_multiplier"] if pos_side == "long" else last_price-atr*config["strategy"]["atr_tp_multiplier"]
    sl_price=last_price-atr*config["strategy"]["atr_sl_multiplier"] if pos_side == "long" else last_price+atr*config["strategy"]["atr_sl_multiplier"]

    # 主单开仓
    main_order=place_order(inst_id, pos_side, order_vol, tp_price, sl_price, api_key, api_secret, api_passphrase)
    if main_order["status"] != "success":
        return {"status": "failed", "msg": f"主单开仓失败：{main_order['msg']}"}

    # 网格挂单（低买高卖）
    grid_orders = []
    for buy_price in buy_levels:
        # 买单（低买）
        buy_order=place_limit_order(inst_id, "buy" if pos_side == "long" else "sell", buy_price, order_vol/len(buy_levels), api_key, api_secret, api_passphrase)
        grid_orders.append({"type": "buy", "price": buy_price, "order": buy_order})
    for sell_price in sell_levels:
        # 卖单（高卖）
        sell_order=place_limit_order(inst_id, "sell" if pos_side == "long" else "buy", sell_price, order_vol/len(sell_levels), api_key, api_secret, api_passphrase)
        grid_orders.append({"type": "sell", "price": sell_price, "order": sell_order})

    return {"status": "success", "main_order": main_order, "grid_orders": grid_orders}

# 限价单函数（补充okx_api.py的接口）
def place_limit_order(inst_id: str, side: str, price: float, vol: float, api_key: str, api_secret: str, api_passphrase: str) -> Dict:
    url=f"{OKX_TRADE_URL}/order"
    method = "POST"
    body=json.dumps({
        "instId": inst_id,
        "tdMode": "isolated",
        "side": side,
        "posSide": "long" if side == "buy" else "short",
        "ordType": "limit",
        "px": str(round(price, 4)),
        "sz": str(round(vol, 4))
    })
    headers=get_okx_headers("/api/v5/trade/order", body, method, api_key, api_secret, api_passphrase)
    try:
        response=requests.post(url, headers=headers, data=body, timeout=10)
        data=response.json()
        if data["code"] == "0":
            return {"status": "success", "ordId": data["data"][0]["ordId"]}
        else:
            return {"status": "failed", "msg": data["msg"]}
    except Exception as e:
        return {"status": "failed", "msg": str(e)}

# FastAPI接口（前端调用）
@app.post("/start_grid")
async def start_grid(request: GridStartRequest):
    global ROBOT_RUNNING
    if ROBOT_RUNNING:
        raise HTTPException(status_code=400, detail="机器人已在运行中")
    ROBOT_RUNNING=True
    try:
        result=grid_trading_strategy(request)
        if result["status"] != "success":
            ROBOT_RUNNING=False
            raise HTTPException(status_code=400, detail=result["msg"])
        return {
            "status": "success",
            "base_price": result["base_price"],
            "grid_spacing": result["grid_spacing"],
            "buy_levels": result["buy_levels"],
            "sell_levels": result["sell_levels"]
        }
    except Exception as e:
        ROBOT_RUNNING=False
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_grid")
async def stop_grid():
    global ROBOT_RUNNING, GRID_PARAMS, RISK_STATE
    if not ROBOT_RUNNING:
        raise HTTPException(status_code=400, detail="机器人未运行")
    # 1. 取消所有挂单
    cancel_all_orders(GRID_PARAMS.get("inst_id", "BTC-USDT-SWAP"))
    # 2. 重置状态
    ROBOT_RUNNING=False
    GRID_PARAMS={}
    RISK_STATE={
        "daily_loss": 0.0,
        "consecutive_loss": 0,
        "trade_count": 0
    }
    return {"status": "success", "msg": "网格交易机器人已停止，所有挂单已取消"}

# 启动服务（main函数）
if __name__ == "__main__":
    # 启动FastAPI服务，端口8000，允许前端跨域调用
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)