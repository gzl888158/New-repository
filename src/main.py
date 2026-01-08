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

# API接口
@app.post("/verify_api")
async def verify_api(params: Dict):
    """验证API并获取基础信息"""
    try:
        uid = verify_api(
            params["apiKey"], params["apiSecret"], params["apiPassphrase"],
            params["instId"], params["env"]
        )
        ticker = fetch_ticker(params["instId"], params["env"])
        candles = fetch_candles(params["instId"], "15m", CONFIG["strategy"]["kline_limit"], params["env"])
        atr = calculate_atr(candles, CONFIG["strategy"]["atr_period"])
        ticker["atr"] = atr
        GLOBAL_STATE["api_info"] = params
        logger.info(f"API验证成功，账户UID：{uid}")
        return {"status": "success", "uid": uid, "ticker": ticker}
    except Exception as e:
        logger.error(f"API验证失败：{str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/start_grid")
async def start_grid(params: Dict):
    """启动网格交易策略"""
    global ROBOT_RUNNING
    if ROBOT_RUNNING:
        raise HTTPException(status_code=400, detail="机器人已运行")
    try:
        # 获取数据
        candles = fetch_candles(params["instId"], "15m", CONFIG["strategy"]["kline_limit"], params["env"])
        atr = calculate_atr(candles, CONFIG["strategy"]["atr_period"])
        last_price = candles[-1]["close"]
        if atr == 0:
            raise Exception("ATR计算失败")
        
        # 计算网格参数
        grid_spacing = atr * params["atrMulti"]
        grid_levels = params["gridLevels"]
        buy_levels = [round(last_price - grid_spacing * i, 4) for i in range(1, grid_levels+1)]
        sell_levels = [round(last_price + grid_spacing * i, 4) for i in range(1, grid_levels+1)]
        
        # 风控检查
        account = get_account_info(params["apiKey"], params["apiSecret"], params["apiPassphrase"], params["env"])
        balance = float(account["available"])
        order_volume = (balance * CONFIG["risk"]["risk_ratio"]) / (grid_spacing * CONFIG["risk"]["leverage"])
        order_volume = max(order_volume, CONFIG["risk"]["min_order_volume"])
        
        # 挂网格单
        place_grid_orders(
            params["instId"], buy_levels, sell_levels, order_volume,
            params["apiKey"], params["apiSecret"], params["apiPassphrase"], params["env"]
        )
        
        # 更新状态
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

@app.post("/stop_grid")
async def stop_grid():
    """停止机器人"""
    global ROBOT_RUNNING
    if not ROBOT_RUNNING:
        raise HTTPException(status_code=400, detail="机器人未运行")
    try:
        api_info = GLOBAL_STATE["api_info"]
        cancel_all_orders(api_info["instId"], api_info["apiKey"], api_info["apiSecret"], api_info["apiPassphrase"], api_info["env"])
        ROBOT_RUNNING = False
        logger.info("机器人已停止，所有挂单已取消")
        return {"status": "success", "msg": "停止成功"}
    except Exception as e:
        logger.error(f"停止失败：{str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_logs")
async def get_logs_api():
    """获取运行日志"""
    return {"logs": get_logs()[-20:]}  # 返回最新20条

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)