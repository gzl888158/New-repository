import time
import json
import requests
import numpy as np
from datetime import datetime

# 配置参数
OKX_API_KEY = ""          # GitHub Secrets会覆盖此值
OKX_SECRET_KEY = ""       # GitHub Secrets会覆盖此值
OKX_PASSPHRASE = ""       # GitHub Secrets会覆盖此值
INST_ID = "BTC-USDT-SWAP"
BAR = "15m"
LIMIT = 20
LOG_FILE = "ai_analysis_log.json"

# 欧易API地址
OKX_BASE_URL = "https://www.okx.com"
OKX_CANDLES_URL = f"{OKX_BASE_URL}/api/v5/market/candles"
OKX_TICKER_URL = f"{OKX_BASE_URL}/api/v5/market/ticker"

def get_okx_candles(inst_id, bar, limit):
    """拉取K线数据"""
    params = {"instId": inst_id, "bar": bar, "limit": limit}
    try:
        response = requests.get(OKX_CANDLES_URL, params=params, timeout=10)
        data = response.json()
        if data["code"] == "0" and len(data["data"]) > 0:
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
        else:
            print(f"K线拉取失败: {data['msg']}")
            return []
    except Exception as e:
        print(f"K线异常: {str(e)}")
        return []

def calculate_macd(close_prices, fastperiod=12, slowperiod=26, signalperiod=9):
    """计算MACD"""
    ema_fast = np.convolve(close_prices, np.ones(fastperiod)/fastperiod, mode='valid')
    ema_slow = np.convolve(close_prices, np.ones(slowperiod)/slowperiod, mode='valid')
    macd_line = ema_fast - ema_slow[-len(ema_fast):]
    signal_line = np.convolve(macd_line, np.ones(signalperiod)/signalperiod, mode='valid')
    histogram = macd_line[-len(signal_line):] - signal_line
    return macd_line, signal_line, histogram

def analyze_trend(candles):
    """趋势分析"""
    if len(candles) < 5:
        return "震荡", "K线数据不足"
    recent = [c["close"] for c in candles[-5:]]
    change = (recent[-1] - recent[0]) / recent[0] * 100
    if change > 0.5:
        return "多头", f"近5根K线上涨{change:.2f}%"
    elif change < -0.5:
        return "空头", f"近5根K线下跌{abs(change):.2f}%"
    else:
        return "震荡", f"近5根K线振幅{abs(change):.2f}%"

def analyze_volume(candles):
    """成交量分析"""
    if len(candles) < 10:
        return "正常", "成交量数据不足"
    recent_vol = [c["vol"] for c in candles[-5:]]
    hist_vol = [c["vol"] for c in candles[:-5]]
    avg_recent = np.mean(recent_vol)
    avg_hist = np.mean(hist_vol)
    if avg_recent > avg_hist * 1.5:
        return "放量", f"成交量放大{((avg_recent/avg_hist)-1)*100:.2f}%"
    elif avg_recent < avg_hist * 0.5:
        return "缩量", f"成交量缩小{((1-avg_recent/avg_hist))*100:.2f}%"
    else:
        return "正常", "成交量无异动"

def generate_signal(trend, macd, volume):
    """生成交易信号"""
    if trend == "多头" and macd == "金叉" and volume == "放量":
        return "buy"
    elif trend == "空头" and macd == "死叉" and volume == "放量":
        return "sell"
    else:
        return "hold"

def main():
    """主函数"""
    print(f"AI分析启动: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    candles = get_okx_candles(INST_ID, BAR, LIMIT)
    if not candles:
        return

    close = np.array([c["close"] for c in candles])
    trend, trend_desc = analyze_trend(candles)
    vol_status, vol_desc = analyze_volume(candles)

    # MACD分析
    macd_line, signal_line, _ = calculate_macd(close)
    macd_signal = "无"
    macd_desc = "MACD无交叉"
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        if macd_line[-1] > signal_line[-1] and macd_line[-2] < signal_line[-2]:
            macd_signal = "金叉"
            macd_desc = "MACD金叉，短期看涨"
        elif macd_line[-1] < signal_line[-1] and macd_line[-2] > signal_line[-2]:
            macd_signal = "死叉"
            macd_desc = "MACD死叉，短期看跌"

    # 生成信号
    signal = generate_signal(trend, macd_signal, vol_status)
    analysis = f"{trend_desc} | {macd_desc} | {vol_desc}"

    # 保存日志
    log_data = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "instId": INST_ID,
        "trend": trend,
        "macdSignal": macd_signal,
        "volumeStatus": vol_status,
        "analysis": analysis,
        "signal": signal,
        "lastPrice": candles[-1]["close"],
        "updateTime": int(time.time() * 1000)
    }

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)
    print(f"分析完成，信号: {signal}，日志已保存")

if __name__ == "__main__":
    main()