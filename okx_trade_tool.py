import time
import json
import requests
import numpy as np
from datetime import datetime, timedelta

# ==================== 配置参数 ====================
OKX_API_KEY = ""          # 替换为你的欧易API Key
OKX_SECRET_KEY = ""       # 替换为你的欧易Secret Key
OKX_PASSPHRASE = ""       # 替换为你的欧易Passphrase
INST_ID = "BTC-USDT-SWAP" # 分析的合约品种
BAR = "15m"               # K线周期
LIMIT = 20                # 拉取最近20根K线
LOG_FILE = "ai_analysis_log.json" # 输出日志文件

# ==================== 欧易 API 接口 ====================
OKX_BASE_URL = "https://www.okx.com"
OKX_CANDLES_URL = f"{OKX_BASE_URL}/api/v5/market/candles"
OKX_TICKER_URL = f"{OKX_BASE_URL}/api/v5/market/ticker"

# ==================== 工具函数 ====================
def get_okx_candles(inst_id, bar, limit):
    """拉取欧易K线数据"""
    params = {
        "instId": inst_id,
        "bar": bar,
        "limit": limit
    }
    try:
        response = requests.get(OKX_CANDLES_URL, params=params, timeout=10)
        data = response.json()
        if data["code"] == "0" and len(data["data"]) > 0:
            # 格式化K线数据：时间戳、开、高、低、收、成交量
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
            return sorted(candles, key=lambda x: x["ts"]) # 按时间正序排列
        else:
            print(f"拉取K线失败: {data['msg']}")
            return []
    except Exception as e:
        print(f"拉取K线异常: {str(e)}")
        return []

def calculate_macd(close_prices, fastperiod=12, slowperiod=26, signalperiod=9):
    """计算MACD指标"""
    ema_fast = np.convolve(close_prices, np.ones(fastperiod)/fastperiod, mode='valid')
    ema_slow = np.convolve(close_prices, np.ones(slowperiod)/slowperiod, mode='valid')
    macd_line = ema_fast - ema_slow[-len(ema_fast):]
    signal_line = np.convolve(macd_line, np.ones(signalperiod)/signalperiod, mode='valid')
    histogram = macd_line[-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def analyze_trend(candles):
    """趋势分析：多头/空头/震荡"""
    if len(candles) < 5:
        return "震荡", "K线数据不足"
    
    recent_close = [c["close"] for c in candles[-5:]]
    # 计算涨幅
    max_price = max(recent_close)
    min_price = min(recent_close)
    change_rate = (max_price - min_price) / min_price * 100

    if recent_close[-1] > recent_close[0] and change_rate > 1:
        return "多头", f"近5根K线上涨{change_rate:.2f}%"
    elif recent_close[-1] < recent_close[0] and change_rate > 1:
        return "空头", f"近5根K线下跌{change_rate:.2f}%"
    else:
        return "震荡", f"近5根K线振幅{change_rate:.2f}%"

def analyze_volume(candles):
    """成交量异动检测"""
    if len(candles) < 10:
        return "正常", "成交量数据不足"
    
    recent_vol = [c["vol"] for c in candles[-5:]]
    history_vol = [c["vol"] for c in candles[:-5]]
    avg_recent = np.mean(recent_vol)
    avg_history = np.mean(history_vol)

    if avg_recent > avg_history * 1.5:
        return "放量", f"近5根K线成交量放大{((avg_recent/avg_history)-1)*100:.2f}%"
    elif avg_recent < avg_history * 0.5:
        return "缩量", f"近5根K线成交量缩小{((1-avg_recent/avg_history))*100:.2f}%"
    else:
        return "正常", "成交量无明显异动"

def generate_trade_signal(trend, macd_signal, volume_status):
    """生成交易信号：buy/sell/hold"""
    # 多头+MACD金叉+放量 → 买入
    if trend == "多头" and macd_signal == "金叉" and volume_status == "放量":
        return "buy"
    # 空头+MACD死叉+放量 → 卖出
    elif trend == "空头" and macd_signal == "死叉" and volume_status == "放量":
        return "sell"
    # 其他情况 → 持有
    else:
        return "hold"

# ==================== 主分析函数 ====================
def main():
    print(f"开始AI分析 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    # 1. 拉取K线数据
    candles = get_okx_candles(INST_ID, BAR, LIMIT)
    if not candles:
        return
    
    # 2. 提取收盘价和成交量
    close_prices = np.array([c["close"] for c in candles])
    vol_data = [c["vol"] for c in candles]

    # 3. 趋势分析
    trend, trend_desc = analyze_trend(candles)

    # 4. MACD分析
    macd_line, signal_line, histogram = calculate_macd(close_prices)
    if len(macd_line) < 2 or len(signal_line) < 2:
        macd_desc = "MACD数据不足"
        macd_signal = "无"
    else:
        # 判断金叉/死叉
        if macd_line[-1] > signal_line[-1] and macd_line[-2] < signal_line[-2]:
            macd_signal = "金叉"
            macd_desc = "MACD金叉形成，短期看涨"
        elif macd_line[-1] < signal_line[-1] and macd_line[-2] > signal_line[-2]:
            macd_signal = "死叉"
            macd_desc = "MACD死叉形成，短期看跌"
        else:
            macd_signal = "无"
            macd_desc = "MACD无明显交叉"

    # 5. 成交量分析
    volume_status, volume_desc = analyze_volume(candles)

    # 6. 生成交易信号
    trade_signal = generate_trade_signal(trend, macd_signal, volume_status)

    # 7. 生成分析结论
    analysis_conclusion = f"{trend_desc} | {macd_desc} | {volume_desc}"

    # 8. 生成日志数据
    log_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instId": INST_ID,
        "trend": trend,
        "macdSignal": macd_signal,
        "volumeStatus": volume_status,
        "analysis": analysis_conclusion,
        "signal": trade_signal,
        "lastPrice": candles[-1]["close"],
        "updateTime": int(time.time() * 1000)
    }

    # 9. 保存日志文件
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)
    
    print(f"AI分析完成，信号: {trade_signal}")
    print(f"日志已保存至: {LOG_FILE}")

if __name__ == "__main__":
    main()