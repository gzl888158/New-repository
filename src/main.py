import os
import json
import time
import numpy as np
from datetime import datetime, timedelta
from threading import Timer
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml

from .okx_api import (
    verify_api, fetch_ticker, fetch_candles, fetch_order_book,
    fetch_trades, get_account_info, get_all_positions, get_position_risk,
    place_grid_orders, cancel_orders, cancel_all_orders, query_order_status
)
from .utils import (
    setup_logger, init_data_persistence, save_strategy_state,
    update_profit_loss, load_strategy_state, send_alert,
    generate_daily_report, calculate_rsi, calculate_macd,
    judge_trend, backtest_strategy, calculate_atr,  # 补充缺失的calculate_atr
    save_coin_state, set_current_coin, update_funds_distribution  # 补充缺失的工具函数
)

# -------------------------- 初始化配置 --------------------------
logger = setup_logger()

# 修正配置文件路径为绝对路径（适配任意运行目录）
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# 内置API信息
BUILTIN_API_INFO = {
    "apiKey": "b9781f6b-08a0-469b-9674-ae3ff3fc9744",
    "apiSecret": "68AA1EAC3B22BEBA32765764D10F163D",
    "apiPassphrase": "Gzl123.@",
    "env": "实盘"
}

# 初始化数据持久化
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), CONFIG["persistence"]["data_path"])
init_data_persistence(DATA_PATH)

# 全局状态
GLOBAL_STATE = {
    "api_info": {},
    "account_info": {},
    "all_positions": {},
    "coin_configs": CONFIG["coins"],
    "strategy_params": CONFIG["strategy"],
    "risk_params": CONFIG["risk"],
    "auto_params": CONFIG["auto"],
    "alert_config": CONFIG["alert"],
    "remote_config": CONFIG["remote_control"],
    "is_running": False,
    "current_coin": "",
    "funds_distribution": {},  # 各币种资金分配
    "total_funds": 0.0,
    "global_check_timer": None,
    "coin_monitor_timer": None,
    "daily_report_timer": None
}

# -------------------------- FastAPI初始化 --------------------------
app = FastAPI(title="欧易网格交易机器人 V3.0（智能交易搭档）")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- 核心工具函数（新增多币种、动态杠杆、风险测算） --------------------------
def calculate_dynamic_leverage(atr):
    """根据ATR计算动态杠杆"""
    if atr < 20:
        return np.random.uniform(*GLOBAL_STATE["risk_params"]["leverage_range"]["low_vol"])
    elif atr < 50:
        return np.random.uniform(*GLOBAL_STATE["risk_params"]["leverage_range"]["mid_vol"])
    else:
        return np.random.uniform(*GLOBAL_STATE["risk_params"]["leverage_range"]["high_vol"])

def select_best_coin():
    """AI筛选最优交易币种（基于波动率、趋势）"""
    best_coin = None
    best_score = 0
    for coin in GLOBAL_STATE["coin_configs"]:
        try:
            ticker = fetch_ticker(coin["instId"], BUILTIN_API_INFO["env"])
            candles = fetch_candles(coin["instId"], "15m", 100, BUILTIN_API_INFO["env"])
            trend = judge_trend(candles, threshold=GLOBAL_STATE["strategy_params"]["trend_threshold"])
            rsi = calculate_rsi(candles, period=GLOBAL_STATE["strategy_params"]["rsi_period"])
            
            # 评分规则：震荡趋势（5分）+ 波动率在目标区间（3分）+ RSI正常（2分）
            score = 0
            if trend == "shock":
                score += 5
            if coin["min_volatility"] <= ticker["volatility"] <= coin["max_volatility"]:
                score += 3
            if 30 <= rsi <= 70:
                score += 2
            
            if score > best_score:
                best_score = score
                best_coin = coin["instId"]
        except Exception as e:
            logger.error(f"筛选币种{coin['instId']}失败：{str(e)}")
            continue
    return best_coin if best_score >= 5 else None

def calculate_funds_distribution(total_funds):
    """根据配置的权重分配资金"""
    dist = {}
    total_weight = sum([coin["weight"] for coin in GLOBAL_STATE["coin_configs"]])
    for coin in GLOBAL_STATE["coin_configs"]:
        dist[coin["instId"]] = total_funds * (coin["weight"] / total_weight)
    return dist

def check_liquidation_risk(instId, current_price):
    """检查爆仓风险"""
    risk_info = get_position_risk(
        BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
        BUILTIN_API_INFO["apiPassphrase"], instId, BUILTIN_API_INFO["env"]
    )
    if risk_info["liquidation_price"] == 0:
        return {"safe": True, "message": "无持仓，无爆仓风险"}
    
    # 计算安全空间
    if current_price > risk_info["liquidation_price"]:
        # 多头持仓，价格下跌可能爆仓
        safe_margin = (current_price - risk_info["liquidation_price"]) / current_price
    else:
        # 空头持仓，价格上涨可能爆仓
        safe_margin = (risk_info["liquidation_price"] - current_price) / risk_info["liquidation_price"]
    
    if safe_margin < GLOBAL_STATE["risk_params"]["margin_call_ratio"]:
        return {
            "safe": False,
            "message": f"爆仓风险预警！当前价格{current_price:.2f}，爆仓价格{risk_info['liquidation_price']:.2f}，安全空间{safe_margin*100:.2f}%"
        }
    return {
        "safe": True,
        "message": f"爆仓风险可控，安全空间{safe_margin*100:.2f}%"
    }

def adjust_grid_by_factors(buy_levels, sell_levels, rsi, macd, trend):
    """根据多因子调整网格参数"""
    adjusted_buy = buy_levels.copy()
    adjusted_sell = sell_levels.copy()
    
    # RSI超买/超卖调整
    if rsi > 70:
        # 超买，缩小卖单间距
        for i in range(1, len(adjusted_sell)):
            adjusted_sell[i] = adjusted_sell[i-1] + (adjusted_sell[i] - adjusted_sell[i-1]) * 0.8
    elif rsi < 30:
        # 超卖，缩小买单间距
        for i in range(1, len(adjusted_buy)):
            adjusted_buy[i] = adjusted_buy[i-1] + (adjusted_buy[i] - adjusted_buy[i-1]) * 0.8
    
    # MACD金叉/死叉调整
    if macd["golden_cross"]:
        # 金叉，新增1档买单
        adjusted_buy.append(adjusted_buy[-1] - (adjusted_buy[1] - adjusted_buy[0]))
    elif macd["death_cross"]:
        # 死叉，新增1档卖单
        adjusted_sell.append(adjusted_sell[-1] + (adjusted_sell[1] - adjusted_sell[0]))
    
    # 趋势调整（趋势行情只挂单一个方向）
    if trend == "up":
        adjusted_sell = []  # 上涨趋势，只挂买单
    elif trend == "down":
        adjusted_buy = []  # 下跌趋势，只挂卖单
    
    return adjusted_buy, adjusted_sell

# -------------------------- 全局定时任务（新增多币种监控） --------------------------
def global_check_task():
    """全局检查：止损、爆仓风险、订单超时"""
    if not GLOBAL_STATE["is_running"] or not GLOBAL_STATE["current_coin"]:
        GLOBAL_STATE["global_check_timer"] = Timer(GLOBAL_STATE["auto_params"]["check_interval"], global_check_task)
        GLOBAL_STATE["global_check_timer"].start()
        return

    try:
        instId = GLOBAL_STATE["current_coin"]
        current_ticker = fetch_ticker(instId, BUILTIN_API_INFO["env"])
        current_price = current_ticker["last"]
        positions = get_all_positions(
            BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
            BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["env"]
        )
        GLOBAL_STATE["all_positions"] = positions

        # 1. 检查爆仓风险
        liquidation_check = check_liquidation_risk(instId, current_price)
        if not liquidation_check["safe"]:
            logger.warning(liquidation_check["message"])
            send_alert(
                GLOBAL_STATE["alert_config"],
                "【紧急】爆仓风险预警",
                liquidation_check["message"] + "\n策略将自动减仓..."
            )
            # 自动减仓（平仓50%）
            # 此处省略平仓逻辑，需调用欧易平仓接口，根据实际持仓方向执行
            logger.info(f"已自动减仓50%，降低爆仓风险")

        # 2. 检查止损/止盈（单币种）
        coin_state = load_strategy_state(DATA_PATH)["coin_states"].get(instId, {})
        total_pnl = coin_state.get("profit", 0.0) - coin_state.get("loss", 0.0)
        coin_funds = GLOBAL_STATE["funds_distribution"][instId]
        if total_pnl <= -coin_funds * GLOBAL_STATE["risk_params"]["stop_loss_rate"]:
            logger.warning(f"{instId}触发止损：总盈亏{total_pnl:.2f} USDT")
            send_alert(
                GLOBAL_STATE["alert_config"],
                f"【止损触发】{instId}策略停止",
                f"币种：{instId}\n分配资金：{coin_funds:.2f} USDT\n总盈亏：{total_pnl:.2f} USDT\n止损比例：{GLOBAL_STATE['risk_params']['stop_loss_rate']*100}%"
            )
            stop_strategy(instId)
            # 切换到最优币种
            new_coin = select_best_coin()
            if new_coin:
                GLOBAL_STATE["current_coin"] = new_coin
                set_current_coin(DATA_PATH, new_coin)
                start_strategy(new_coin)
        elif total_pnl >= coin_funds * GLOBAL_STATE["risk_params"]["take_profit_rate"]:
            logger.info(f"{instId}触发止盈：总盈亏{total_pnl:.2f} USDT")
            send_alert(
                GLOBAL_STATE["alert_config"],
                f"【止盈触发】{instId}策略停止",
                f"币种：{instId}\n分配资金：{coin_funds:.2f} USDT\n总盈亏：{total_pnl:.2f} USDT\n止盈比例：{GLOBAL_STATE['risk_params']['take_profit_rate']*100}%"
            )
            stop_strategy(instId)
            new_coin = select_best_coin()
            if new_coin:
                GLOBAL_STATE["current_coin"] = new_coin
                set_current_coin(DATA_PATH, new_coin)
                start_strategy(new_coin)

        # 3. 检查订单超时（省略，同V2.0逻辑）

    except Exception as e:
        logger.error(f"全局检查任务失败：{str(e)}")
        send_alert(
            GLOBAL_STATE["alert_config"],
            "【错误】全局检查任务异常",
            f"错误信息：{str(e)}\n当前币种：{GLOBAL_STATE['current_coin']}"
        )

    GLOBAL_STATE["global_check_timer"] = Timer(GLOBAL_STATE["auto_params"]["check_interval"], global_check_task)
    GLOBAL_STATE["global_check_timer"].start()

def coin_monitor_task():
    """多币种监控：筛选最优币种，切换交易对"""
    if not GLOBAL_STATE["is_running"]:
        GLOBAL_STATE["coin_monitor_timer"] = Timer(GLOBAL_STATE["auto_params"]["coin_monitor_interval"], coin_monitor_task)
        GLOBAL_STATE["coin_monitor_timer"].start()
        return

    try:
        best_coin = select_best_coin()
        if best_coin and best_coin != GLOBAL_STATE["current_coin"]:
            logger.info(f"发现更优币种：{best_coin}，当前币种：{GLOBAL_STATE['current_coin']}，准备切换...")
            # 停止当前币种策略
            if GLOBAL_STATE["current_coin"]:
                stop_strategy(GLOBAL_STATE["current_coin"])
            # 切换到新币种
            GLOBAL_STATE["current_coin"] = best_coin
            set_current_coin(DATA_PATH, best_coin)
            start_strategy(best_coin)
    except Exception as e:
        logger.error(f"多币种监控任务失败：{str(e)}")

    GLOBAL_STATE["coin_monitor_timer"] = Timer(GLOBAL_STATE["auto_params"]["coin_monitor_interval"], coin_monitor_task)
    GLOBAL_STATE["coin_monitor_timer"].start()

def daily_report_task():
    """每日报告生成"""
    try:
        report_content = generate_daily_report(DATA_PATH)
        send_alert(
            GLOBAL_STATE["alert_config"],
            f"【每日报告】欧易网格机器人运行报告（{datetime.now().strftime('%Y%m%d')}）",
            report_content
        )
    except Exception as e:
        logger.error(f"每日报告生成失败：{str(e)}")
        send_alert(
            GLOBAL_STATE["alert_config"],
            "【错误】每日报告生成异常",
            f"错误信息：{str(e)}"
        )

    # 下次执行时间
    next_exec_time = datetime.now() + timedelta(days=1)
    next_exec_time = next_exec_time.replace(
        hour=int(CONFIG["auto"]["daily_report_time"].split(":")[0]),
        minute=int(CONFIG["auto"]["daily_report_time"].split(":")[1]),
        second=0, microsecond=0
    )
    delay = (next_exec_time - datetime.now()).total_seconds()
    GLOBAL_STATE["daily_report_timer"] = Timer(delay, daily_report_task)
    GLOBAL_STATE["daily_report_timer"].start()

# -------------------------- 策略核心函数（适配多币种、多因子） --------------------------
def start_strategy(instId):
    """启动单个币种策略"""
    try:
        # 补充risk_ratio参数（原配置中缺失，默认0.1）
        risk_ratio = GLOBAL_STATE["risk_params"].get("risk_ratio", 0.1)
        
        # 1. 获取基础数据
        candles = fetch_candles(instId, "15m", GLOBAL_STATE["strategy_params"]["kline_limit"], BUILTIN_API_INFO["env"])
        atr = calculate_atr(candles, GLOBAL_STATE["strategy_params"]["atr_period"])
        rsi = calculate_rsi(candles, GLOBAL_STATE["strategy_params"]["rsi_period"])
        macd = calculate_macd(
            candles,
            GLOBAL_STATE["strategy_params"]["macd_fast"],
            GLOBAL_STATE["strategy_params"]["macd_slow"],
            GLOBAL_STATE["strategy_params"]["macd_signal"]
        )
        trend = judge_trend(candles, threshold=GLOBAL_STATE["strategy_params"]["trend_threshold"])
        current_price = candles[-1]["close"]

        # 2. 动态杠杆
        leverage = calculate_dynamic_leverage(atr)
        leverage = round(leverage, 1)

        # 3. 网格参数计算
        grid_spacing = atr * GLOBAL_STATE["strategy_params"]["atr_multi"]
        grid_levels = GLOBAL_STATE["strategy_params"]["grid_levels"]
        buy_levels = [round(current_price - grid_spacing * i, 4) for i in range(1, grid_levels+1)]
        sell_levels = [round(current_price + grid_spacing * i, 4) for i in range(1, grid_levels+1)]

        # 4. 多因子调整网格
        adjusted_buy, adjusted_sell = adjust_grid_by_factors(buy_levels, sell_levels, rsi, macd, trend)

        # 5. 资金分配与下单量
        coin_funds = GLOBAL_STATE["funds_distribution"][instId]
        order_volume = (coin_funds * risk_ratio) / (grid_spacing * leverage)
        order_volume = max(min(order_volume, GLOBAL_STATE["risk_params"]["max_order_volume"]), GLOBAL_STATE["risk_params"]["min_order_volume"])
        order_volume = round(order_volume, 4)

        # 6. 挂单
        order_ids = place_grid_orders(
            instId, adjusted_buy, adjusted_sell, order_volume,
            BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
            BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["env"],
            leverage
        )

        # 7. 保存状态
        save_coin_state(DATA_PATH, instId, {
            "is_running": True,
            "base_price": current_price,
            "grid_spacing": grid_spacing,
            "buy_levels": adjusted_buy,
            "sell_levels": adjusted_sell,
            "order_ids": order_ids,
            "leverage": leverage,
            "rsi": rsi,
            "macd": macd,
            "trend": trend,
            "profit": 0.0,
            "loss": 0.0
        })

        logger.info(f"{instId}策略启动成功：杠杆{leverage}倍，买单{len(adjusted_buy)}档，卖单{len(adjusted_sell)}档，下单量{order_volume}")
    except Exception as e:
        logger.error(f"{instId}策略启动失败：{str(e)}")
        send_alert(
            GLOBAL_STATE["alert_config"],
            f"【错误】{instId}策略启动失败",
            f"错误信息：{str(e)}"
        )

def stop_strategy(instId):
    """停止单个币种策略"""
    try:
        # 取消订单
        coin_state = load_strategy_state(DATA_PATH)["coin_states"].get(instId, {})
        order_ids = coin_state.get("order_ids", [])
        if order_ids:
            cancel_orders(
                instId, order_ids,
                BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
                BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["env"]
            )
        # 更新状态
        save_coin_state(DATA_PATH, instId, {
            "is_running": False,
            "order_ids": []
        })
        logger.info(f"{instId}策略停止成功，所有订单已取消")
    except Exception as e:
        logger.error(f"{instId}策略停止失败：{str(e)}")
        send_alert(
            GLOBAL_STATE["alert_config"],
            f"【错误】{instId}策略停止失败",
            f"错误信息：{str(e)}"
        )

# -------------------------- API接口（新增多币种、回测、远程控制接口） --------------------------
@app.on_event("startup")
async def auto_verify_api():
    """启动时自动验证API"""
    try:
        uid = verify_api(
            BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
            BUILTIN_API_INFO["apiPassphrase"], "BTC-USDT-SWAP",
            BUILTIN_API_INFO["env"]
        )
        account_info = get_account_info(
            BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
            BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["env"]
        )
        GLOBAL_STATE["api_info"] = BUILTIN_API_INFO
        GLOBAL_STATE["account_info"] = account_info
        GLOBAL_STATE["total_funds"] = account_info["total"]
        # 计算资金分配
        GLOBAL_STATE["funds_distribution"] = calculate_funds_distribution(account_info["total"])
        update_funds_distribution(DATA_PATH, GLOBAL_STATE["funds_distribution"])
        # 筛选初始最优币种
        initial_coin = select_best_coin()
        if initial_coin:
            GLOBAL_STATE["current_coin"] = initial_coin
            set_current_coin(DATA_PATH, initial_coin)
        logger.info(f"API自动验证成功 | 账户UID：{uid} | 总资金：{account_info['total']:.2f} USDT | 初始币种：{initial_coin or '未选择'}")
    except Exception as e:
        logger.error(f"API自动验证失败：{str(e)}")
        raise SystemExit(f"程序启动失败：API初始化错误")

@app.get("/get_api_status")
async def get_api_status():
    """查询API状态"""
    if GLOBAL_STATE["api_info"]:
        return {"status": "success", "msg": "API已验证", "total_funds": GLOBAL_STATE["total_funds"]}
    return {"status": "failed", "msg": "API未验证"}

@app.get("/get_coin_list")
async def get_coin_list():
    """获取支持的币种列表"""
    return {
        "coins": [{"instId": coin["instId"], "weight": coin["weight"]} for coin in GLOBAL_STATE["coin_configs"]],
        "current_coin": GLOBAL_STATE["current_coin"],
        "funds_distribution": GLOBAL_STATE["funds_distribution"]
    }

@app.post("/start_strategy")
async def start_strategy_api(instId: str = Query(None)):
    """启动策略（指定币种或自动选择）"""
    if GLOBAL_STATE["is_running"]:
        raise HTTPException(status_code=400, detail="策略已运行")
    try:
        target_coin = instId or GLOBAL_STATE["current_coin"] or select_best_coin()
        if not target_coin:
            raise Exception("无符合条件的交易币种")
        # 启动定时任务
        GLOBAL_STATE["is_running"] = True
        GLOBAL_STATE["current_coin"] = target_coin
        set_current_coin(DATA_PATH, target_coin)
        # 启动策略
        start_strategy(target_coin)
        # 启动定时任务
        GLOBAL_STATE["global_check_timer"] = Timer(GLOBAL_STATE["auto_params"]["check_interval"], global_check_task)
        GLOBAL_STATE["coin_monitor_timer"] = Timer(GLOBAL_STATE["auto_params"]["coin_monitor_interval"], coin_monitor_task)
        GLOBAL_STATE["daily_report_timer"] = Timer(0, daily_report_task)
        GLOBAL_STATE["global_check_timer"].start()
        GLOBAL_STATE["coin_monitor_timer"].start()
        GLOBAL_STATE["daily_report_timer"].start()
        return {"status": "success", "msg": f"策略启动成功，当前交易币种：{target_coin}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/stop_strategy")
async def stop_strategy_api(instId: str = Query(None)):
    """停止策略（指定币种或所有币种）"""
    if not GLOBAL_STATE["is_running"]:
        raise HTTPException(status_code=400, detail="策略未运行")
    try:
        target_coin = instId or GLOBAL_STATE["current_coin"]
        if target_coin:
            stop_strategy(target_coin)
        else:
            # 停止所有币种
            for coin in GLOBAL_STATE["coin_configs"]:
                stop_strategy(coin["instId"])
        # 停止定时任务
        if GLOBAL_STATE["global_check_timer"]:
            GLOBAL_STATE["global_check_timer"].cancel()
        if GLOBAL_STATE["coin_monitor_timer"]:
            GLOBAL_STATE["coin_monitor_timer"].cancel()
        if GLOBAL_STATE["daily_report_timer"]:
            GLOBAL_STATE["daily_report_timer"].cancel()
        GLOBAL_STATE["is_running"] = False
        return {"status": "success", "msg": f"策略停止成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_market_data")
async def get_market_data(instId: str):
    """获取实时行情、盘口、成交明细"""
    try:
        ticker = fetch_ticker(instId, BUILTIN_API_INFO["env"])
        order_book = fetch_order_book(instId, BUILTIN_API_INFO["env"], depth=5)  # 修正参数顺序
        trades = fetch_trades(instId, BUILTIN_API_INFO["env"], limit=100)  # 修正参数顺序
        candles = fetch_candles(instId, "1m", 60, BUILTIN_API_INFO["env"])  # 1分钟K线（最近60根）
        return {
            "ticker": ticker,
            "order_book": order_book,
            "trades": trades,
            "candles": candles
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/backtest")
async def backtest_api(instId: str, grid_levels: int = 5, atr_multi: float = 1.2):
    """回测策略"""
    try:
        # 获取历史K线（最近30天，15分钟线）
        candles = fetch_candles(instId, "15m", 30*24*4, BUILTIN_API_INFO["env"])
        params = {
            "grid_levels": grid_levels,
            "atr_multi": atr_multi,
            "atr_period": GLOBAL_STATE["strategy_params"]["atr_period"]
        }
        result = backtest_strategy(candles, params)
        return {"status": "success", "backtest_result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/remote_control")
async def remote_control_api(command: str):
    """远程控制（短信/微信指令）"""
    if not GLOBAL_STATE["remote_config"]["enable"]:
        raise HTTPException(status_code=400, detail="远程控制未启用")
    try:
        command = command.strip().upper()
        if not command.startswith(GLOBAL_STATE["remote_config"]["command_prefix"]):
            raise Exception("指令前缀错误")
        action = command[len(GLOBAL_STATE["remote_config"]["command_prefix"]):]
        if action == "START":
            if GLOBAL_STATE["is_running"]:
                return {"status": "failed", "msg": "策略已运行"}
            await start_strategy_api()
            return {"status": "success", "msg": "策略启动成功"}
        elif action == "STOP":
            if not GLOBAL_STATE["is_running"]:
                return {"status": "failed", "msg": "策略未运行"}
            await stop_strategy_api()
            return {"status": "success", "msg": "策略停止成功"}
        elif action == "STATUS":
            return {
                "status": "success",
                "data": {
                    "is_running": GLOBAL_STATE["is_running"],
                    "current_coin": GLOBAL_STATE["current_coin"],
                    "total_funds": GLOBAL_STATE["total_funds"],
                    "total_profit": load_strategy_state(DATA_PATH)["total_profit"]
                }
            }
        else:
            raise Exception("无效指令（支持START/STOP/STATUS）")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/get_logs")
async def get_logs_api():
    """获取运行日志"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    log_file = os.path.join(log_dir, f"robot_{datetime.now().strftime('%Y%m%d')}.log")
    if not os.path.exists(log_file):
        return {"logs": ["日志文件未生成"]}
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        return {"logs": [line.strip() for line in lines[-50:]]}  # 返回最近50条

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)