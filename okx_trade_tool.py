import time
import json
import requests
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# ==================== 核心配置（必填） ====================
# 欧易API基础配置（无需密钥，仅拉取公开行情）
OKX_MARKET_URL = "https://www.okx.com/api/v5/market"
# 需要分析的合约品种
INST_IDS = [
    "BTC-USDT-SWAP",
    "ETH-USDT-SWAP",
    "SOL-USDT-SWAP"
]
# 多周期配置（15m+1h+4h 共振）
TIMEFRAMES = ["15m", "1h", "4h"]
# K线数据量（至少100根，保证指标计算准确性）
KLINE_LIMIT = 100
# 输出日志文件名（前端需读取此文件）
OUTPUT_LOG_FILE = "multi_coin_multi_timeframe_log.json"

# ==================== 指标计算工具函数 ====================
def calculate_macd(close_prices: np.ndarray, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9) -> tuple:
    """
    计算MACD指标
    :param close_prices: 收盘价数组
    :return: macd_line, signal_line
    """
    # 计算EMA
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        alpha = 2 / (period + 1)
        ema_values = np.zeros_like(prices)
        ema_values[period - 1] = np.mean(prices[:period])
        for i in range(period, len(prices)):
            ema_values[i] = alpha * prices[i] + (1 - alpha) * ema_values[i - 1]
        return ema_values[period - 1:]
    
    ema_fast = ema(close_prices, fastperiod)
    ema_slow = ema(close_prices, slowperiod)
    # 对齐长度
    min_len = min(len(ema_fast), len(ema_slow))
    ema_fast = ema_fast[-min_len:]
    ema_slow = ema_slow[-min_len:]
    
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signalperiod)
    # 再次对齐
    final_len = min(len(macd_line), len(signal_line))
    return macd_line[-final_len:], signal_line[-final_len:]

def calculate_atr(candles: List[Dict[str, float]], period: int = 14) -> float:
    """
    计算ATR波动率指标
    :param candles: K线数据列表，包含high/low/close字段
    :return: ATR值
    """
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
    """
    单周期信号判断（MACD金叉买/死叉卖）
    :param candles: K线数据
    :return: buy/sell/hold
    """
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

# ==================== 欧易行情拉取 ====================
def fetch_okx_candles(inst_id: str, bar: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    拉取欧易K线数据
    :param inst_id: 合约品种
    :param bar: K线周期
    :param limit: 数据量
    :return: 格式化后的K线列表
    """
    url = f"{OKX_MARKET_URL}/candles"
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": limit
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data["code"] != "0" or len(data["data"]) == 0:
            print(f"拉取{inst_id}({bar})失败: {data.get('msg', '未知错误')}")
            return []
        # 格式化数据：时间戳转时间，价格转浮点数
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
        # 按时间正序排列
        return sorted(candles, key=lambda x: x["ts"])
    except Exception as e:
        print(f"拉取{inst_id}({bar})异常: {str(e)}")
        return []

# ==================== 多周期共振分析 ====================
def analyze_multi_timeframe(inst_id: str) -> Dict[str, Any]:
    """
    单合约多周期共振分析
    :param inst_id: 合约品种
    :return: 分析结果字典
    """
    # 1. 拉取所有周期K线
    timeframe_signals = {}
    for tf in TIMEFRAMES:
        candles = fetch_okx_candles(inst_id, tf, KLINE_LIMIT)
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
    main_candles = fetch_okx_candles(inst_id, "15m", KLINE_LIMIT)
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

# ==================== 主函数 ====================
def main():
    print(f"===== 欧易多周期AI分析启动 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    # 1. 遍历所有合约分析
    total_result = {
        "updateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updateTimestamp": int(time.time() * 1000),
        "coins": []
    }
    for inst in INST_IDS:
        coin_result = analyze_multi_timeframe(inst)
        total_result["coins"].append(coin_result)
        print(f"{inst} 分析完成 | 单周期: {coin_result['signal']} | 共振: {coin_result['multiTimeframeSignal']}")
    
    # 2. 保存JSON日志
    with open(OUTPUT_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(total_result, f, ensure_ascii=False, indent=4)
    print(f"===== 分析完成，日志已保存至 {OUTPUT_LOG_FILE} =====")

if __name__ == "__main__":
    # 循环运行：每30分钟执行一次（与前端AI分析更新频率一致）
    while True:
        main()
        # 休眠30分钟
        time.sleep(30 * 60)