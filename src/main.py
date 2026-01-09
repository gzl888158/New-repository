import os
import json
import time
import numpy as np
from datetime import datetime
from typing import List, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml
from dotenv import load_dotenv

from .okx_api import (
    verify_api, fetch_ticker, fetch_candles, calculate_atr,
    place_grid_orders, cancel_all_orders, get_account_info
)
from .utils import setup_logger, get_logs

# -------------------------- 【新增/修改】内置API和网格参数 --------------------------
BUILTIN_API_INFO = {
    "apiKey": "b9781f6b-08a0-469b-9674-ae3ff3fc9744",
    "apiSecret": "68AA1EAC3B22BEBA32765764D10F163D",
    "apiPassphrase": "Gzl123.@",
    "instId": "BTC-USDT-SWAP",
    "env": "实盘"
}
BUILTIN_GRID_PARAMS = {
    "atrMulti": 1.2,
    "gridLevels": 5
}
# -----------------------------------------------------------------------------------

# 初始化
load_dotenv()
app = FastAPI(title="OKX Grid Trading Robot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger = setup_logger()

# 加载配置
with open("config/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# 全局状态
ROBOT_RUNNING = False
GLOBAL_STATE = {
    "api_info": {},
    "grid_params": {},
    "risk_state": {"daily_loss": 0.0, "consecutive_loss": 0, "trade_count": 0}
}

# -------------------------- 【新增】启动时自动验证API --------------------------
@app.on_event("startup")
async def auto_verify_api():
    try:
        uid = verify_api(
            BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
            BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["instId"],
            BUILTIN_API_INFO["env"]
        )
        ticker = fetch_ticker(BUILTIN_API_INFO["instId"], BUILTIN_API_INFO["env"])
        candles = fetch_candles(
            BUILTIN_API_INFO["instId"], "15m", 
            CONFIG["strategy"]["kline_limit"], BUILTIN_API_INFO["env"]
        )
        atr = calculate_atr(candles, CONFIG["strategy"]["atr_period"])
        ticker["atr"] = atr
        GLOBAL_STATE["api_info"] = BUILTIN_API_INFO
        logger.info(f"API自动验证成功 | 账户UID：{uid}")
    except Exception as e:
        logger.error(f"API自动验证失败：{str(e)}")
        raise SystemExit(f"程序启动失败：API初始化错误")

# -------------------------- 【新增】API状态查询接口 --------------------------
@app.get("/get_api_status")
async def get_api_status():
    if GLOBAL_STATE["api_info"]:
        return {"status": "success", "msg": "API已验证"}
    return {"status": "failed", "msg": "API未验证"}

# -------------------------- 【修改】启动网格策略（无需传参） --------------------------
@app.post("/start_grid")
async def start_grid():
    global ROBOT_RUNNING
    if ROBOT_RUNNING:
        raise HTTPException(status_code=400, detail="机器人已处于运行状态")
    try:
        api_info = GLOBAL_STATE["api_info"]
        params = {**BUILTIN_GRID_PARAMS, "instId": api_info["instId"], "env": api_info["env"]}

        candles = fetch_candles(params["instId"], "15m", CONFIG["strategy"]["kline_limit"], params["env"])
        atr = calculate_atr(candles, CONFIG["strategy"]["atr_period"])
        last_price = candles[-1]["close"]
        if atr == 0:
            raise Exception("ATR指标计算失败（K线数据异常）")
        
        grid_spacing = atr * params["atrMulti"]
        grid_levels = params["gridLevels"]
        buy_levels = [round(last_price - grid_spacing * i, 4) for i in range(1, grid_levels+1)]
        sell_levels = [round(last_price + grid_spacing * i, 4) for i in range(1, grid_levels+1)]
        
        account = get_account_info(api_info["apiKey"], api_info["apiSecret"], api_info["apiPassphrase"], api_info["env"])
        balance = float(account["available"])
        order_volume = (balance * CONFIG["risk"]["risk_ratio"]) / (grid_spacing * CONFIG["risk"]["leverage"])
        order_volume = max(order_volume, CONFIG["risk"]["min_order_volume"])
        
        place_grid_orders(
            params["instId"], buy_levels, sell_levels, order_volume,
            api_info["apiKey"], api_info["apiSecret"], api_info["apiPassphrase"], api_info["env"]
        )
        
        ROBOT_RUNNING = True
        GLOBAL_STATE["grid_params"] = {
            "base_price": last_price, "grid_spacing": grid_spacing,
            "buy_levels": buy_levels, "sell_levels": sell_levels,
            "order_volume": order_volume
        }
        logger.info(f"网格策略启动成功 | 基准价：{last_price} | 间距：{grid_spacing} | 层数：{grid_levels}")
        return {
            "status": "success", "basePrice": last_price, "gridSpacing": grid_spacing,
            "buyLevels": buy_levels, "sellLevels": sell_levels
        }
    except Exception as e:
        logger.error(f"策略启动失败：{str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# 停止机器人（无修改）
@app.post("/stop_grid")
async def stop_grid():
    global ROBOT_RUNNING
    if not ROBOT_RUNNING:
        raise HTTPException(status_code=400, detail="机器人未运行")
    try:
        api_info = GLOBAL_STATE["api_info"]
        cancel_all_orders(api_info["instId"], api_info["apiKey"], api_info["apiSecret"], api_info["apiPassphrase"], api_info["env"])
        ROBOT_RUNNING = False
        logger.info("机器人已停止 | 所有挂单已取消")
        return {"status": "success", "msg": "停止成功"}
    except Exception as e:
        logger.error(f"停止失败：{str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# 获取最新日志（无修改）
@app.get("/get_logs")
async def get_logs_api():
    return {"logs": get_logs()[-20:]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)