import time
import json
import requests
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

# ==================== 1. 全局配置（可根据需求调整） ====================
class Config:
    # 欧易行情接口
    OKX_MARKET_URL = "https://www.okx.com/api/v5/market"
    # 分析合约品种
    INST_IDS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP"]
    # 多周期配置（共振周期）
    TIMEFRAMES = ["15m", "1h", "4h"]
    # K线数据量（指标计算最小需求）
    KLINE_LIMIT = 100
    # 输出日志文件名（前端读取）
    OUTPUT_LOG_FILE = "multi_coin_multi_timeframe_log.json"
    # 脚本运行间隔（秒）- 30分钟
    RUN_INTERVAL = 30 * 60
    # 回测配置
    BACKTEST_DAYS = 30  # 回测近30天数据
    INIT_CAPITAL = 1000  # 初始资金（USDT）
    RISK_RATIO = 0.02    # 单笔风险比例（2%）
    LEVERAGE = 10        # 回测杠杆倍数

# ==================== 2. 日志配置（分级输出，便于调试） ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("okx_strategy.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== 3. 核心指标计算 ====================
def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """计算EMA均线（平滑指数移动平均）"""
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices, dtype=float)
    ema[period - 1] = np.mean(prices[:period])
    for i in range(period, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    return ema

def calculate_macd(close_prices: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """计算MACD指标（快线、慢线）"""
    if len(close_prices) < max(fastperiod, slowperiod, signalperiod) + 1:
        return np.array([]), np.array([])
    ema_fast = calculate_ema(close_prices, fastperiod)
    ema_slow = calculate_ema(close_prices, slowperiod)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signalperiod)
    # 对齐长度
    min_len = min(len(macd_line), len(signal_line))
    return macd_line[-min_len:], signal_line[-min_len:]

def calculate_atr(candles: List[Dict[str, float]], period: int = 14) -> float:
    """计算ATR波动率指标"""
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
    """单周期信号判断（MACD金叉买/死叉卖）"""
    if len(candles) < 30:
        return "hold"
    close_prices = np.array([c["close"] for c in candles])
    macd_line, signal_line = calculate_macd(close_prices)
    if len(macd_line) < 2 or len(signal_line) < 2:
        return "hold"
    # 金叉：MACD线上穿信号线
    if macd_line[-1] > signal_line[-1] and macd_line[-2] < signal_line[-2]:
        return "buy"
    # 死叉：MACD线下穿信号线
    elif macd_line[-1] < signal_line[-1] and macd_line[-2] > signal_line[-2]:
        return "sell"
    else:
        return "hold"

# ==================== 4. 欧易行情拉取（限流+异常处理） ====================
def fetch_okx_candles(inst_id: str, bar: str, limit: int = 100, end_ts: int = None) -> List[Dict[str, Any]]:
    """
    拉取欧易K线数据（支持指定结束时间，用于回测）
    :param inst_id: 合约品种
    :param bar: K线周期
    :param limit: 数据量
    :param end_ts: 结束时间戳（毫秒）
    :return: 格式化后的K线列表
    """
    url = f"{Config.OKX_MARKET_URL}/candles"
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": limit
    }
    if end_ts:
        params["after"] = end_ts
    try:
        # 接口请求限流：间隔1秒
        time.sleep(1)
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data["code"] != "0" or len(data["data"]) == 0:
            logger.error(f"拉取{inst_id}({bar})失败: {data.get('msg', '未知错误')}")
            return []
        # 格式化数据
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
        logger.error(f"拉取{inst_id}({bar})异常: {str(e)}")
        return []

# ==================== 5. 多周期共振分析 ====================
def analyze_multi_timeframe(inst_id: str, end_ts: int = None) -> Dict[str, Any]:
    """单合约多周期共振分析（支持回测）"""
    # 1. 拉取所有周期K线
    timeframe_signals = {}
    for tf in Config.TIMEFRAMES:
        candles = fetch_okx_candles(inst_id, tf, Config.KLINE_LIMIT, end_ts)
        if not candles:
            timeframe_signals[tf] = "hold"
            continue
        timeframe_signals[tf] = get_single_timeframe_signal(candles)
    
    # 2. 多周期共振判断
    multi_signal = "hold"
    if all(s == "buy" for s in timeframe_signals.values()):
        multi_signal = "buy"
    elif all(s == "sell" for s in timeframe_signals.values()):
        multi_signal = "sell"
    
    # 3. 拉取15m K线计算ATR和单周期信号
    main_candles = fetch_okx_candles(inst_id, "15m", Config.KLINE_LIMIT, end_ts)
    single_signal = get_single_timeframe_signal(main_candles) if main_candles else "hold"
    atr_value = calculate_atr(main_candles) if main_candles else 0.0
    last_price = main_candles[-1]["close"] if main_candles else 0.0
    
    # 4. 生成分析结论
    analysis_text = f"多周期信号: {timeframe_signals} | 共振结果: {multi_signal} | ATR波动率: {atr_value}"
    return {
        "instId": inst_id,
        "signal": single_signal,
        "multiTimeframeSignal": multi_signal,
        "atr": atr_value,
        "lastPrice": last_price,
        "timeframeSignals": timeframe_signals,
        "analysis": analysis_text
    }

# ==================== 6. 新增：策略回测模块 ====================
def backtest_strategy(inst_id: str) -> Dict[str, Any]:
    """回测多周期共振策略"""
    logger.info(f"开始回测{inst_id}近{Config.BACKTEST_DAYS}天数据")
    # 计算回测结束时间（当前时间）和起始时间（30天前）
    end_ts = int(time.time() * 1000)
    start_ts = int((datetime.now() - timedelta(days=Config.BACKTEST_DAYS)).timestamp() * 1000)
    
    # 拉取4h K线（用于回测，周期更长，数据点更少）
    candles = fetch_okx_candles(inst_id, "4h", 1000)
    if not candles:
        return {"status": "failed", "msg": "回测数据拉取失败"}
    
    # 过滤时间范围内的数据
    backtest_candles = [c for c in candles if start_ts <= c["ts"] <= end_ts]
    if len(backtest_candles) < 10:
        return {"status": "failed", "msg": "有效回测数据不足"}
    
    # 回测初始化
    capital = Config.INIT_CAPITAL
    position = 0  # 持仓量：>0多仓，<0空仓，0无持仓
    trades = []
    win_count = 0
    total_trades = 0

    # 逐根K线回测
    for i in range(14, len(backtest_candles)):
        # 取当前K线及之前的数据，模拟实时分析
        current_candles = backtest_candles[:i+1]
        signal = get_single_timeframe_signal(current_candles)
        current_price = current_candles[-1]["close"]
        atr = calculate_atr(current_candles)
        
        if atr == 0:
            continue
        
        # 计算止损价格和仓位（基于ATR和风险比例）
        risk_per_trade = capital * Config.RISK_RATIO
        stop_loss = current_price - (atr * 1) if signal == "buy" else current_price + (atr * 1)
        position_size = risk_per_trade / (abs(current_price - stop_loss)) / Config.LEVERAGE

        # 开仓逻辑：无持仓时，根据信号开仓
        if position == 0:
            if signal == "buy":
                position = position_size
                entry_price = current_price
                total_trades += 1
                logger.debug(f"开多仓：价格{entry_price}，仓位{position_size}")
            elif signal == "sell":
                position = -position_size
                entry_price = current_price
                total_trades += 1
                logger.debug(f"开空仓：价格{entry_price}，仓位{position_size}")
        # 平仓逻辑：有持仓时，反向信号或止损触发平仓
        else:
            # 止损触发
            if (position > 0 and current_price <= stop_loss) or (position < 0 and current_price >= stop_loss):
                profit = (current_price - entry_price) * position * Config.LEVERAGE if position > 0 else (entry_price - current_price) * abs(position) * Config.LEVERAGE
                capital += profit
                if profit > 0:
                    win_count += 1
                trades.append({
                    "type": "止损平仓",
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "profit": profit,
                    "capital": capital
                })
                position = 0
                logger.debug(f"止损平仓：利润{profit}，剩余资金{capital}")
            # 反向信号触发平仓
            elif (position > 0 and signal == "sell") or (position < 0 and signal == "buy"):
                profit = (current_price - entry_price) * position * Config.LEVERAGE if position > 0 else (entry_price - current_price) * abs(position) * Config.LEVERAGE
                capital += profit
                if profit > 0:
                    win_count += 1
                trades.append({
                    "type": "信号平仓",
                    "entry_price": entry_price,
                    "exit_price": current_price,
                    "profit": profit,
                    "capital": capital
                })
                position = 0
                logger.debug(f"信号平仓：利润{profit}，剩余资金{capital}")

    # 计算回测结果
    total_return = (capital - Config.INIT_CAPITAL) / Config.INIT_CAPITAL * 100
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    return {
        "status": "success",
        "instId": inst_id,
        "init_capital": Config.INIT_CAPITAL,
        "final_capital": round(capital, 2),
        "total_return": f"{round(total_return, 2)}%",
        "total_trades": total_trades,
        "win_trades": win_count,
        "win_rate": f"{round(win_rate, 2)}%",
        "trades": trades[-10:]  # 只返回最近10笔交易
    }

# ==================== 7. 主函数（实时分析+回测） ====================
def main():
    # 1. 执行策略回测
    logger.info("===== 开始策略回测 =====")
    backtest_results = []
    for inst in Config.INST_IDS:
        result = backtest_strategy(inst)
        backtest_results.append(result)
        logger.info(f"{inst}回测结果：{json.dumps(result, ensure_ascii=False, indent=2)}")
    # 保存回测结果
    with open("backtest_result.json", "w", encoding="utf-8") as f:
        json.dump(backtest_results, f, ensure_ascii=False, indent=4)
    
    # 2. 执行实时多周期分析
    logger.info("===== 开始实时多周期分析 =====")
    total_result = {
        "updateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updateTimestamp": int(time.time() * 1000),
        "coins": [],
        "backtestSummary": backtest_results
    }
    for inst in Config.INST_IDS:
        coin_result = analyze_multi_timeframe(inst)
        total_result["coins"].append(coin_result)
        logger.info(f"{inst}分析完成 | 共振信号: {coin_result['multiTimeframeSignal']}")
    
    # 3. 保存分析结果
    with open(Config.OUTPUT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(total_result, f, ensure_ascii=False, indent=4)
    logger.info(f"===== 分析完成，日志已保存至 {Config.OUTPUT_LOG_FILE} =====")

# ==================== 8. 脚本启动入口 ====================
if __name__ == "__main__":
    logger.info("===== 欧易多周期AI分析脚本启动 =====")
    while True:
        try:
            main()
        except Exception as e:
            logger.error(f"脚本运行异常: {str(e)}")
        logger.info(f"===== 休眠{Config.RUN_INTERVAL/60}分钟 =====")
        time.sleep(Config.RUN_INTERVAL)