import os
import json
import time
import numpy as np
from datetime import datetime, timedelta
from threading import Timer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml

from .okx_api import (
    verify_api, fetch_ticker, fetch_candles, calculate_atr,
    place_grid_orders, cancel_orders, cancel_all_orders,
    query_order_status, get_account_info, get_position_pnl
)
from .utils import (
    setup_logger, init_data_persistence, save_strategy_state,
    update_profit_loss, load_strategy_state, send_email_alert,
    generate_daily_report
)

# -------------------------- 初始化配置 --------------------------
logger = setup_logger()

# 加载配置文件
with open("config/config.yaml", "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# 内置API信息（从之前配置迁移）
BUILTIN_API_INFO = {
    "apiKey": "b9781f6b-08a0-469b-9674-ae3ff3fc9744",
    "apiSecret": "68AA1EAC3B22BEBA32765764D10F163D",
    "apiPassphrase": "Gzl123.@",
    "instId": "BTC-USDT-SWAP",
    "env": "实盘"
}

# 初始化数据持久化
init_data_persistence(CONFIG["persistence"]["data_path"])

# 全局状态（整合配置+持久化数据）
GLOBAL_STATE = {
    "api_info": {},
    "account_info": {},  # 账户信息
    "strategy_params": CONFIG["strategy"],
    "risk_params": CONFIG["risk"],
    "auto_params": CONFIG["auto"],
    "alert_config": CONFIG["alert"],
    "is_running": False,
    "total_profit": 0.0,
    "total_loss": 0.0,
    "current_orders": [],  # 当前挂单ID
    "floating_pnl": 0.0,   # 浮动盈亏
    "last_price": 0.0,     # 最新价格
    "global_max_loss": 0.0,  # 账户最大亏损阈值
    "daily_report_timer": None  # 每日报告定时器
}

# -------------------------- FastAPI初始化 --------------------------
app = FastAPI(title="欧易网格交易机器人 V2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------- 核心工具函数 --------------------------
def check_market_volatility(current_price, last_price, threshold):
    """检查行情波动率（1分钟内）"""
    if last_price == 0:
        return False
    volatility = abs(current_price - last_price) / last_price
    return volatility > threshold

def adjust_position_by_volatility(atr, base_volume):
    """根据ATR（波动率）调整下单量：ATR越大，下单量越小"""
    max_volume = CONFIG["risk"]["max_order_volume"]
    min_volume = CONFIG["risk"]["min_order_volume"]
    # ATR>50时，下单量减半；ATR<20时，用基础量；中间按比例调整
    if atr > 50:
        return max(min_volume, base_volume * 0.5)
    elif atr < 20:
        return min(max_volume, base_volume)
    else:
        return min(max_volume, max(min_volume, base_volume * (1 - (atr-20)/60)))

def expand_grid_if_needed(current_price, buy_levels, sell_levels, expand_threshold):
    """价格突破网格时，自动新增档位"""
    new_buy = []
    new_sell = []
    grid_spacing = CONFIG["strategy"]["atr_multi"] * calculate_atr(fetch_candles(
        BUILTIN_API_INFO["instId"], "15m", 20, BUILTIN_API_INFO["env"]
    ), CONFIG["strategy"]["atr_period"])
    
    # 跌破所有买单档位，新增买单
    if current_price < min(buy_levels) if buy_levels else True:
        for i in range(1, expand_threshold+1):
            new_buy.append(round(min(buy_levels) - i*grid_spacing, 4) if buy_levels else round(current_price - i*grid_spacing, 4))
    # 突破所有卖单档位，新增卖单
    if current_price > max(sell_levels) if sell_levels else True:
        for i in range(1, expand_threshold+1):
            new_sell.append(round(max(sell_levels) + i*grid_spacing, 4) if sell_levels else round(current_price + i*grid_spacing, 4))
    return new_buy, new_sell

def check_stop_loss_take_profit(realized_pnl, floating_pnl, initial_balance):
    """检查止损/止盈：已实现盈亏+浮动盈亏"""
    total_pnl = realized_pnl + floating_pnl
    stop_loss_threshold = initial_balance * CONFIG["risk"]["stop_loss_rate"]
    take_profit_threshold = initial_balance * CONFIG["risk"]["take_profit_rate"]
    if total_pnl <= -stop_loss_threshold:
        return "stop_loss", total_pnl
    elif total_pnl >= take_profit_threshold:
        return "take_profit", total_pnl
    return None, total_pnl

def check_order_timeout(order_ids, timeout_minutes):
    """检查超时未成交订单"""
    if not order_ids:
        return []
    order_status = query_order_status(
        BUILTIN_API_INFO["instId"], order_ids,
        BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
        BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["env"]
    )
    timeout_orders = []
    for ord_id, status in order_status.items():
        # 未成交且创建时间超过timeout_minutes
        # 欧易订单状态为"open"表示未成交
        if status["status"] == "open":
            # 这里简化：假设订单创建时间超过timeout_minutes即超时（实际需通过订单创建时间计算）
            timeout_orders.append(ord_id)
    return timeout_orders

def global_check_task():
    """全局定时检查任务：止损、订单超时、行情异常、数据备份"""
    if not GLOBAL_STATE["is_running"]:
        # 策略未运行，仅备份数据
        save_strategy_state(CONFIG["persistence"]["data_path"], {
            "is_running": False,
            "floating_pnl": GLOBAL_STATE["floating_pnl"]
        })
        # 重新启动定时器
        Timer(GLOBAL_STATE["auto_params"]["check_interval"], global_check_task).start()
        return

    try:
        # 1. 获取最新数据
        current_ticker = fetch_ticker(BUILTIN_API_INFO["instId"], BUILTIN_API_INFO["env"])
        current_price = current_ticker["last"]
        position_pnl = get_position_pnl(
            BUILTIN_API_INFO["instId"], BUILTIN_API_INFO["apiKey"],
            BUILTIN_API_INFO["apiSecret"], BUILTIN_API_INFO["apiPassphrase"],
            BUILTIN_API_INFO["env"]
        )
        GLOBAL_STATE["floating_pnl"] = position_pnl["floating_pnl"]
        GLOBAL_STATE["last_price"] = current_price

        # 2. 检查行情异常（波动率）
        if check_market_volatility(
            current_price, GLOBAL_STATE.get("prev_price", current_price),
            GLOBAL_STATE["risk_params"]["volatility_limit"]
        ):
            logger.warning(f"行情异常波动：1分钟波动率超过{GLOBAL_STATE['risk_params']['volatility_limit']*100}%")
            send_email_alert(
                GLOBAL_STATE["alert_config"],
                "【紧急】行情异常波动，策略自动停止",
                f"当前价格：{current_price} USDT\n上一分钟价格：{GLOBAL_STATE.get('prev_price', current_price)} USDT\n波动率：{abs(current_price - GLOBAL_STATE.get('prev_price', current_price))/GLOBAL_STATE.get('prev_price', current_price)*100:.2f}%"
            )
            # 停止策略
            stop_strategy()
            GLOBAL_STATE["prev_price"] = current_price
            Timer(GLOBAL_STATE["auto_params"]["check_interval"], global_check_task).start()
            return

        # 3. 检查止损/止盈
        initial_balance = GLOBAL_STATE["account_info"]["total"]
        sl_tp_flag, total_pnl = check_stop_loss_take_profit(
            position_pnl["realized_pnl"], position_pnl["floating_pnl"],
            initial_balance
        )
        if sl_tp_flag == "stop_loss":
            logger.warning(f"触发止损：总盈亏{total_pnl:.2f} USDT，超过阈值{initial_balance*GLOBAL_STATE['risk_params']['stop_loss_rate']:.2f} USDT")
            send_email_alert(
                GLOBAL_STATE["alert_config"],
                "【止损触发】策略自动停止",
                f"账户初始余额：{initial_balance:.2f} USDT\n总盈亏：{total_pnl:.2f} USDT\n止损比例：{GLOBAL_STATE['risk_params']['stop_loss_rate']*100}%"
            )
            stop_strategy()
            # 更新总亏损
            update_profit_loss(CONFIG["persistence"]["data_path"], 0.0, abs(total_pnl))
            GLOBAL_STATE["total_loss"] += abs(total_pnl)
        elif sl_tp_flag == "take_profit":
            logger.info(f"触发止盈：总盈亏{total_pnl:.2f} USDT，达到阈值{initial_balance*GLOBAL_STATE['risk_params']['take_profit_rate']:.2f} USDT")
            send_email_alert(
                GLOBAL_STATE["alert_config"],
                "【止盈触发】策略自动停止",
                f"账户初始余额：{initial_balance:.2f} USDT\n总盈亏：{total_pnl:.2f} USDT\n止盈比例：{GLOBAL_STATE['risk_params']['take_profit_rate']*100}%"
            )
            stop_strategy()
            # 更新总盈利
            update_profit_loss(CONFIG["persistence"]["data_path"], total_pnl, 0.0)
            GLOBAL_STATE["total_profit"] += total_pnl

        # 4. 检查订单超时
        timeout_orders = check_order_timeout(
            GLOBAL_STATE["current_orders"],
            GLOBAL_STATE["strategy_params"]["order_timeout"]
        )
        if timeout_orders:
            logger.info(f"发现{len(timeout_orders)}个超时订单，自动取消")
            cancel_orders(
                BUILTIN_API_INFO["instId"], timeout_orders,
                BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
                BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["env"]
            )
            # 从当前订单列表中移除
            GLOBAL_STATE["current_orders"] = [ord_id for ord_id in GLOBAL_STATE["current_orders"] if ord_id not in timeout_orders]
            # 重新挂单（按当前价格生成新网格）
            new_buy, new_sell = expand_grid_if_needed(
                current_price, [], [], GLOBAL_STATE["strategy_params"]["grid_expand_threshold"]
            )
            base_volume = (initial_balance * GLOBAL_STATE["risk_params"]["risk_ratio"]) / (
                GLOBAL_STATE["strategy_params"]["grid_spacing"] * GLOBAL_STATE["risk_params"]["leverage"]
            )
            atr = calculate_atr(fetch_candles(
                BUILTIN_API_INFO["instId"], "15m", 20, BUILTIN_API_INFO["env"]
            ), GLOBAL_STATE["strategy_params"]["atr_period"])
            adjusted_volume = adjust_position_by_volatility(atr, base_volume)
            new_order_ids = place_grid_orders(
                BUILTIN_API_INFO["instId"], new_buy, new_sell, adjusted_volume,
                BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
                BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["env"],
                GLOBAL_STATE["risk_params"]["leverage"]
            )
            GLOBAL_STATE["current_orders"].extend(new_order_ids)
            logger.info(f"超时订单重新挂单：{len(new_order_ids)}个")

        # 5. 检查网格是否需要扩展
        strategy_state = load_strategy_state(CONFIG["persistence"]["data_path"])
        current_buy = strategy_state["current_strategy"]["buy_levels"]
        current_sell = strategy_state["current_strategy"]["sell_levels"]
        new_buy, new_sell = expand_grid_if_needed(
            current_price, current_buy, current_sell,
            GLOBAL_STATE["strategy_params"]["grid_expand_threshold"]
        )
        if new_buy or new_sell:
            logger.info(f"价格突破网格，新增买单{len(new_buy)}档，卖单{len(new_sell)}档")
            base_volume = (initial_balance * GLOBAL_STATE["risk_params"]["risk_ratio"]) / (
                GLOBAL_STATE["strategy_params"]["grid_spacing"] * GLOBAL_STATE["risk_params"]["leverage"]
            )
            atr = calculate_atr(fetch_candles(
                BUILTIN_API_INFO["instId"], "15m", 20, BUILTIN_API_INFO["env"]
            ), GLOBAL_STATE["strategy_params"]["atr_period"])
            adjusted_volume = adjust_position_by_volatility(atr, base_volume)
            new_order_ids = place_grid_orders(
                BUILTIN_API_INFO["instId"], new_buy, new_sell, adjusted_volume,
                BUILTIN_API_INFO["apiKey"], BUILTIN_API_INFO["apiSecret"],
                BUILTIN_API_INFO["apiPassphrase"], BUILTIN_API_INFO["env"],
                GLOBAL_STATE["risk_params"]["leverage"]
            )
            GLOBAL_STATE["current_orders"].extend(new_order_ids)
            # 更新策略状态
            save_strategy_state(CONFIG["persistence"]["data_path"], {
                "buy_levels": current_buy + new_buy,
                "sell_levels": current_sell + new_sell,
                "order_ids": GLOBAL_STATE["current_orders"]
            })

        # 6. 数据备份
        save_strategy_state(CONFIG["persistence"]["data_path"], {
            "is_running": GLOBAL_STATE["is_running"],
            "base_price": strategy_state["current_strategy"]["base_price"],
            "grid_spacing": GLOBAL_STATE["strategy_params"]["grid_spacing"],
            "buy_levels": current_buy + new_buy,
            "sell_levels": current_sell + new_sell,
            "order_ids": GLOBAL_STATE["current_orders"],
            "floating_pnl": GLOBAL_STATE["floating_pnl"]
        })

        # 7. 更新上一次价格
        GLOBAL_STATE["prev_price"] = current_price

    except Exception as e:
        logger.error(f"全局检查任务失败：{str(e)}")
        send_email_alert(
            GLOBAL_STATE["alert_config"],
            "【错误】全局检查任务异常",
            f"错误信息：{str(e)}\n策略状态：{'运行中' if GLOBAL_STATE['is_running'] else '已停止'}"
        )

    # 重新启动定时器
    Timer(GLOBAL_STATE["auto_params"]["check_interval"], global_check_task).start()

def daily_report_task():
    """每日报告生成任务"""
    try:
        report_content = generate_daily_report(CONFIG["persistence"]["data_path"])
        # 发送报告到邮箱
        send_email_alert(
            GLOBAL_STATE["alert_config"],
            f"【每日报告】欧易网格机器人运行报告（{datetime.now().strftime('%Y%m%d')}）",
            report_content
        )
    except Exception as e:
        logger.error(f"每日报告生成失败：{str(e)}")
        send_email_alert(
            GLOBAL_STATE["alert_config"],
            "【错误】每日报告生成异常",
            f"错误信息：{str(e)}"
        )

    # 计算下次执行时间（明天同一时间）
    next_exec_time = datetime.now() + timedelta(days=1)
    next_exec_time = next_exec_time.replace(
        hour=int(CONFIG["auto"]["daily_report_time"].split(":")[0]),
        minute=int(CONFIG["auto"]["daily_report_time"].split(":")[1]),
        second=0, microsecond=0
    )
    delay = (next_exec_time - datetime.now()).total_seconds()
   