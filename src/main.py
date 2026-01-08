import time
import json
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from .utils import setup_logger
from .okx_api import fetch_candles, get_account_balance, get_position, place_order, get_okx_headers
import requests
import os
from dotenv import load_dotenv

# 加载环境变量与配置
load_dotenv()
logger = setup_logger()
with open("config/config.yaml", "r", encoding="utf-8") as f:
    import yaml
    config = yaml.safe_load(f)

# 配置常量
INST_IDS = config["strategy"]["inst_ids"]
TIMEFRAMES = config["strategy"]["timeframes"]
KLINE_LIMIT = config["strategy"]["kline_limit"]
MAX_DAILY_LOSS = config["risk"]["max_daily_loss"]
MAX_CONSECUTIVE_LOSS = config["risk"]["max_consecutive_loss"]
DAILY_TRADE_LIMIT = config["risk"]["daily_trade_limit"]
RISK_RATIO = config["risk"]["risk_ratio"]
LEVERAGE = config["risk"]["leverage"]
ATR_TP_MULTIPLIER = config["strategy"]["atr_tp_multiplier"]
ATR_SL_MULTIPLIER = config["strategy"]["atr_sl_multiplier"]

# ==================== 指标计算 ====================
def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """计算EMA均线"""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices, dtype=float)
    ema[period - 1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_macd(close_prices: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """计算MACD指标"""
    if len(close_prices) < max(fastperiod, slowperiod, signalperiod) + 1:
        return np.array([]), np.array([])
    ema_fast = calculate_ema(close_prices, fastperiod)
    ema_slow = calculate_ema(close_prices, slowperiod)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signalperiod)
    min_len = min(len(macd_line), len(signal_line))
    return macd_line[-min_len:], signal_line[-min_len:]

def calculate_atr(candles: List[Dict[str, float]], period: int = 14) -> float:
    """计算ATR波动率"""
    if len(candles) < period + 1:
        return 0.0
    tr_list = []
    for i in range(1, len(candles)):
        high = candles[i]["high"]
        low = candles[i]["low"]
        prev_close = candles[i-1]["close"]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)
    return round(np.mean(tr_list[-period:]), 4)

def get_single_timeframe_signal(candles: List[Dict[str, float]]) -> str:
    """单周期信号判断"""
    if len(candles) < 30:
        return "hold"
    close_prices = np.array([c["close"] for c in candles])
    macd_line, signal_line = calculate_macd(close_prices)
    if len(macd_line) < 2 or len(signal_line) < 2:
        return "hold"
    if macd_line[-1] > signal_line[-1] and macd_line[-2] < signal_line[-2]:
        return "buy"
    elif macd_line[-1] < signal_line[-1] and macd_line[-2] > signal_line[-2]:
        return "sell"
    else:
        return "hold"

# ==================== 多周期分析 ====================
def analyze_multi_timeframe(inst_id: str) -> Dict[str, Any]:
    """多周期共振分析"""
    timeframe_signals = {}
    for tf in TIMEFRAMES:
        candles = fetch_candles(inst_id, tf, KLINE_LIMIT)
        timeframe_signals[tf] = get_single_timeframe_signal(candles) if candles else "hold"
    
    # 共振信号判断
    multi_signal = "hold"
    if all(s == "buy" for s in timeframe_signals.values()):
        multi_signal = "buy"
    elif all(s == "sell" for s in timeframe_signals.values()):
        multi_signal = "sell"
    
    # 15m K线补充数据
    main_candles = fetch_candles(inst_id, "15m", KLINE_LIMIT)
    single_signal = get_single_timeframe_signal(main_candles) if main_candles else "hold"
    atr = calculate_atr(main_candles) if main_candles else 0.0
    last_price = main_candles[-1]["close"] if main_candles else 0.0
    
    return {
        "instId": inst_id,
        "signal": single_signal,
        "multiTimeframeSignal": multi_signal,
        "atr": atr,
        "lastPrice": last_price,
        "timeframeSignals": timeframe_signals,
        "analysis": f"多周期信号: {timeframe_signals} | 共振: {multi_signal} | ATR: {atr}"
    }

# ==================== 风控熔断 ====================
def get_today_trades(inst_id: str) -> List[Dict[str, Any]]:
    """获取今日交易记录"""
    url = f"{config['okx']['trade_url']}/orders"
    params = {
        "instId": inst_id,
        "state": "filled",
        "begin": int((datetime.now().replace(hour=0, minute=0, second=0) - timedelta(days=1)).timestamp() * 1000)
    }
    headers = get_okx_headers("/api/v5/trade/orders")
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        return data["data"] if data["code"] == "0" else []
    except Exception as e:
        logger.error(f"获取交易记录异常: {str(e)}")
        return []

def check_risk_control(inst_id: str) -> bool:
    """风控检查"""
    today_trades = get_today_trades(inst_id)
    if len(today_trades) >= DAILY_TRADE_LIMIT:
        logger.warning(f"触发单日交易次数限制: {len(today_trades)}/{DAILY_TRADE_LIMIT}")
        return False
    
    # 计算盈亏与连续亏损
    total_profit = 0.0
    consecutive_loss = 0
    for trade in today_trades:
        profit = float(trade["fillPx"]) * float(trade["fillSz"]) * (1 if trade["side"] == "buy" else -1)
        total_profit += profit
        consecutive_loss = consecutive_loss + 1 if profit < 0 else 0
    
    if consecutive_loss >= MAX_CONSECUTIVE_LOSS:
        logger.warning(f"触发连续亏损限制: {consecutive_loss}/{MAX_CONSECUTIVE_LOSS}")
        return False
    
    balance = get_account_balance()
    if balance > 0 and abs(total_profit) / balance >= MAX_DAILY_LOSS:
        logger.warning(f"触发单日亏损限制: {abs(total_profit)/balance*100:.2f}%/{MAX_DAILY_LOSS*100:.2f}%")
        return False
    
    return True

# ==================== 自动交易 ====================
def auto_trade():
    """全自动交易执行"""
    logger.info("===== 开始全自动交易执行 =====")
    for inst_id in INST_IDS:
        # 1. 获取多周期信号
        analysis = analyze_multi_timeframe(inst_id)
        signal = analysis["multiTimeframeSignal"]
        if signal == "hold":
            continue
        
        # 2. 风控检查
        if not check_risk_control(inst_id):
            logger.info(f"{inst_id} 触发风控，跳过交易")
            continue
        
        # 3. 检查持仓
        position = get_position(inst_id)
        if position["posAmt"] > 0 and signal == "buy":
            logger.info(f"{inst_id} 已有多仓，跳过买入")
            continue
        if position["posAmt"] < 0 and signal == "sell":
            logger.info(f"{inst_id} 已有空仓，跳过卖出")
            continue
        
        # 4. 计算下单参数
        atr = analysis["atr"]
        last_price = analysis["lastPrice"]
        if atr == 0 or last_price == 0:
            logger.error(f"{inst_id} 数据异常，跳过交易")
            continue
        
        tp_price = last_price + atr * ATR_TP_MULTIPLIER if signal == "buy" else last_price - atr * ATR_TP_MULTIPLIER
        sl_price = last_price - atr * ATR_SL_MULTIPLIER if signal == "buy" else last_price + atr * ATR_SL_MULTIPLIER
        balance = get_account_balance()
        vol = (balance * RISK_RATIO) / (abs(last_price - sl_price)) * LEVERAGE
        if vol < config["risk"]["min_order_volume"]:
            logger.info(f"{inst_id} 下单量过小，跳过")
            continue
        
        # 5. 下单
        pos_side = "long" if signal == "buy" else "short"
        order_result = place_order(inst_id, pos_side, vol, tp_price, sl_price)
        if order_result["status"] == "success":
            logger.info(f"{inst_id} 交易执行成功")
        else:
            logger.error(f"{inst_id} 交易执行失败: {order_result['msg']}")

# ==================== 回测模块 ====================
def backtest_strategy(inst_id: str) -> Dict[str, Any]:
    """策略回测"""
    logger.info(f"开始回测{inst_id}近{config['run']['backtest_days']}天数据")
    end_ts = int(time.time() * 1000)
    start_ts = int((datetime.now() - timedelta(days=config['run']['backtest_days'])).timestamp() * 1000)
    
    candles = fetch_candles(inst_id, "4h", 1000)
    if not candles:
        return {"status": "failed", "msg": "回测数据拉取失败"}
    
    backtest_candles = [c for c in candles if start_ts <= c["ts"] <= end_ts]
    if len(backtest_candles) < 10:
        return {"status": "failed", "msg": "有效数据不足"}
    
    # 回测初始化
    capital = config["run"]["init_capital"]
    position = 0
    trades = []
    win_count = 0
    total_trades = 0

    for i in range(14, len(backtest_candles)):
        current_candles = backtest_candles[:i+1]
        signal = get_single_timeframe_signal(current_candles)
        current_price = current_candles[-1]["close"]
        atr = calculate_atr(current_candles)
        
        if atr == 0:
            continue
        
        risk_per_trade = capital * RISK_RATIO
        stop_loss = current_price - atr * 1 if signal == "buy" else current_price + atr * 1
        position_size = risk_per_trade / (abs(current_price - stop_loss)) / LEVERAGE

        # 开仓
        if position == 0:
            if signal == "buy":
                position = position_size
                entry_price = current_price
                total_trades += 1
            elif signal == "sell":
                position = -position_size
                entry_price = current_price
                total_trades += 1
        # 平仓
        else:
            if (position > 0 and current_price <= stop_loss) or (position < 0 and current_price >= stop_loss):
                profit = (current_price - entry_price) * position * LEVERAGE if position > 0 else (entry_price - current_price) * abs(position) * LEVERAGE
                capital += profit
                win_count += 1 if profit > 0 else 0
                trades.append({"profit": profit, "capital": capital})
                position = 0
            elif (position > 0 and signal == "sell") or (position < 0 and signal == "buy"):
                profit = (current_price - entry_price) * position * LEVERAGE if position > 0 else (entry_price - current_price) * abs(position) * LEVERAGE
                capital += profit
                win_count += 1 if profit > 0 else 0
                trades.append({"profit": profit, "capital": capital})
                position = 0

    total_return = (capital - config["run"]["init_capital"]) / config["run"]["init_capital"] * 100
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    return {
        "status": "success",
        "instId": inst_id,
        "init_capital": config["run"]["init_capital"],
        "final_capital": round(capital, 2),
        "total_return": f"{round(total_return, 2)}%",
        "win_rate": f"{round(win_rate, 2)}%",
        "total_trades": total_trades
    }

# ==================== 主函数 ====================
def main():
    # 1. 策略回测
    logger.info("===== 开始策略回测 =====")
    backtest_results = []
    for inst in INST_IDS:
        res = backtest_strategy(inst)
        backtest_results.append(res)
        logger.info(f"{inst}回测结果: {json.dumps(res, ensure_ascii=False)}")
    with open("backtest_result.json", "w", encoding="utf-8") as f:
        json.dump(backtest_results, f, ensure_ascii=False, indent=4)
    
    # 2. 多周期分析
    logger.info("===== 开始多周期分析 =====")
    analysis_results = {
        "updateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updateTimestamp": int(time.time() * 1000),
        "coins": [],
        "backtestSummary": backtest_results
    }
    for inst in INST_IDS:
        res = analyze_multi_timeframe(inst)
        analysis_results["coins"].append(res)
    with open("multi_coin_multi_timeframe_log.json", "w", encoding="utf-8") as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=4)
    
    # 3. 自动交易
    auto_trade()

if __name__ == "__main__":
    logger.info("===== 欧易全自动交易系统启动 =====")
    while True:
        try:
            main()
        except Exception as e:
            logger.error(f"程序运行异常: {str(e)}")
        logger.info(f"===== 休眠{config['run']['interval']/60}分钟 =====")
        time.sleep(config["run"]["interval"])