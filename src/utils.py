import logging
import os
import json
import time
import smtplib
import requests
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import numpy as np

# -------------------------- 日志配置（无修改） --------------------------
def setup_logger():
    logger = logging.getLogger("okx_grid_robot_v3")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/robot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger

logger = setup_logger()

# -------------------------- API重试装饰器（无修改） --------------------------
def request_retry(retry_times=3, retry_delay=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for i in range(retry_times):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"API请求失败（{i+1}/{retry_times}）：{str(e)}，{retry_delay}秒后重试")
                    time.sleep(retry_delay)
            raise Exception(f"API请求重试{retry_times}次失败")
        return wrapper
    return decorator

# -------------------------- 数据持久化（新增多币种状态保存） --------------------------
def init_data_persistence(data_path):
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_profit": 0.0,
                "total_loss": 0.0,
                "coin_states": {},  # 多币种状态：instId -> 状态
                "current_coin": "", # 当前交易币种
                "strategy组合": {
                    "total_funds": 0.0,  # 总资金
                    "funds_distribution": {}  # 各币种资金分配
                }
            }, f, ensure_ascii=False, indent=2)

def save_coin_state(data_path, instId, state):
    with open(data_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        if instId not in data["coin_states"]:
            data["coin_states"][instId] = {}
        data["coin_states"][instId].update(state)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

def set_current_coin(data_path, instId):
    with open(data_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["current_coin"] = instId
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

def update_funds_distribution(data_path, funds_dist):
    with open(data_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["strategy组合"]["funds_distribution"] = funds_dist
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

# 原有持久化函数保留
def save_strategy_state(data_path, state):
    with open(data_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data.update(state)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

def update_profit_loss(data_path, profit, loss):
    with open(data_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["total_profit"] += profit
        data["total_loss"] += loss
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

def load_strategy_state(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------- 告警功能（新增短信告警） --------------------------
def send_email_alert(config, subject, content):
    if not config["enable_email_alert"]:
        return
    try:
        msg = MIMEText(content, "plain", "utf-8")
        msg["From"] = config["email_sender"]
        msg["To"] = config["email_receiver"]
        msg["Subject"] = subject
        
        server = smtplib.SMTP_SSL(config["smtp_server"], config["smtp_port"])
        server.login(config["email_sender"], config["email_password"])
        server.sendmail(config["email_sender"], config["email_receiver"], msg.as_string())
        server.quit()
        logger.info(f"邮件告警发送成功：{subject}")
    except Exception as e:
        logger.error(f"邮件告警发送失败：{str(e)}")

def send_sms_alert(config, content):
    if not config["enable_sms_alert"] or not config["sms_api_key"]:
        return
    try:
        # 示例：对接阿里云短信API（需替换为实际接口）
        sms_url = "https://dysmsapi.aliyuncs.com/"
        params = {
            "Action": "SendSms",
            "PhoneNumbers": config["sms_phone"],
            "SignName": "你的短信签名",
            "TemplateCode": "你的短信模板CODE",
            "TemplateParam": json.dumps({"content": content}),
            "AccessKeyId": config["sms_api_key"].split(":")[0],
            "AccessKeySecret": config["sms_api_key"].split(":")[1],
            "RegionId": "cn-hangzhou",
            "Timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
        }
        response = requests.get(sms_url, params=params, timeout=10)
        response.raise_for_status()
        logger.info(f"短信告警发送成功：{content}")
    except Exception as e:
        logger.error(f"短信告警发送失败：{str(e)}")

def send_alert(config, subject, content):
    send_email_alert(config, subject, content)
    send_sms_alert(config, content)

# -------------------------- 多因子指标计算（新增） --------------------------
def calculate_rsi(candles, period=14):
    """计算RSI指标"""
    if len(candles) < period + 1:
        raise Exception(f"K线不足{period+1}根，无法计算RSI")
    closes = [candle["close"] for candle in candles]
    deltas = np.diff(closes)
    gains = deltas[deltas > 0]
    losses = -deltas[deltas < 0]
    
    avg_gain = np.mean(gains[:period]) if len(gains) > 0 else 0.0
    avg_loss = np.mean(losses[:period]) if len(losses) > 0 else 0.0
    
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return round(rsi, 2)

def calculate_macd(candles, fast=12, slow=26, signal=9):
    """计算MACD指标（EMA12 - EMA26 = DIF；DIF的EMA9 = DEA；DIF-DEA=BAR）"""
    if len(candles) < slow + signal:
        raise Exception(f"K线不足{slow+signal}根，无法计算MACD")
    closes = [candle["close"] for candle in candles]
    ema_fast = np.zeros_like(closes)
    ema_slow = np.zeros_like(closes)
    
    # 计算EMA
    ema_fast[slow-1] = np.mean(closes[:slow])
    ema_slow[slow-1] = np.mean(closes[:slow])
    for i in range(slow, len(closes)):
        ema_fast[i] = (closes[i] * 2 + ema_fast[i-1] * (fast - 1)) / (fast + 1)
        ema_slow[i] = (closes[i] * 2 + ema_slow[i-1] * (slow - 1)) / (slow + 1)
    
    dif = ema_fast - ema_slow
    dea = np.zeros_like(dif)
    dea[slow+signal-1] = np.mean(dif[slow-1:slow+signal-1])
    for i in range(slow+signal, len(dif)):
        dea[i] = (dif[i] * 2 + dea[i-1] * (signal - 1)) / (signal + 1)
    
    bar = dif - dea
    return {
        "dif": round(dif[-1], 4),
        "dea": round(dea[-1], 4),
        "bar": round(bar[-1], 4),
        "golden_cross": dif[-1] > dea[-1] and dif[-2] < dea[-2],  # 金叉
        "death_cross": dif[-1] < dea[-1] and dif[-2] > dea[-2]   # 死叉
    }

def judge_trend(candles, period=20, threshold=0.02):
    """判断市场趋势：上涨/下跌/震荡（基于EMA20与价格差值）"""
    if len(candles) < period:
        raise Exception(f"K线不足{period}根，无法判断趋势")
    closes = [candle["close"] for candle in candles]
    ema20 = np.mean(closes[-period:])
    price_diff = (closes[-1] - ema20) / ema20
    if price_diff > threshold:
        return "up"  # 上涨趋势
    elif price_diff < -threshold:
        return "down"  # 下跌趋势
    else:
        return "shock"  # 震荡趋势

# -------------------------- 回测工具（新增） --------------------------
def backtest_strategy(candles, params, initial_funds=10000):
    """回测网格策略：基于历史K线数据计算收益"""
    atr = calculate_atr(candles, params["atr_period"])
    grid_spacing = atr * params["atr_multi"]
    grid_levels = params["grid_levels"]
    balance = initial_funds
    position = 0.0  # 持仓量
    trade_logs = []
    
    for i in range(params["atr_period"], len(candles)):
        current_price = candles[i]["close"]
        # 生成网格
        buy_levels = [current_price - grid_spacing * j for j in range(1, grid_levels+1)]
        sell_levels = [current_price + grid_spacing * j for j in range(1, grid_levels+1)]
        
        # 模拟成交（价格触碰网格档位即成交）
        for buy_price in buy_levels:
            if current_price <= buy_price and balance > 0:
                volume = (balance * 0.1) / current_price  # 每次用10%资金买入
                position += volume
                balance -= volume * current_price
                trade_logs.append({
                    "time": candles[i]["timestamp"],
                    "side": "buy",
                    "price": current_price,
                    "volume": volume,
                    "balance": balance
                })
        
        for sell_price in sell_levels:
            if current_price >= sell_price and position > 0:
                volume = position * 0.5  # 每次卖出50%持仓
                balance += volume * current_price
                position -= volume
                trade_logs.append({
                    "time": candles[i]["timestamp"],
                    "side": "sell",
                    "price": current_price,
                    "volume": volume,
                    "balance": balance
                })
    
    # 计算回测结果
    final_funds = balance + position * candles[-1]["close"]
    total_return = (final_funds - initial_funds) / initial_funds * 100
    max_drawdown = calculate_max_drawdown([log["balance"] for log in trade_logs]) if trade_logs else 0.0
    
    return {
        "initial_funds": initial_funds,
        "final_funds": round(final_funds, 2),
        "total_return": round(total_return, 2),
        "max_drawdown": round(max_drawdown, 2),
        "trade_count": len(trade_logs),
        "params": params
    }

def calculate_max_drawdown(balance_list):
    """计算最大回撤"""
    if len(balance_list) < 2:
        return 0.0
    max_balance = balance_list[0]
    max_drawdown = 0.0
    for balance in balance_list[1:]:
        if balance > max_balance:
            max_balance = balance
        drawdown = (max_balance - balance) / max_balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown * 100

# -------------------------- 每日报告生成（优化多币种统计） --------------------------
def generate_daily_report(data_path, report_path="reports/"):
    os.makedirs(report_path, exist_ok=True)
    state = load_strategy_state(data_path)
    date = datetime.now().strftime("%Y%m%d")
    
    # 多币种收益统计
    coin_profits = []
    for instId, coin_state in state["coin_states"].items():
        profit = coin_state.get("profit", 0.0) - coin_state.get("loss", 0.0)
        coin_profits.append(f"- {instId}：{profit:.2f} USDT")
    
    report_content = f"""
欧易网格交易机器人每日报告（{date}）
==================================
1. 总盈利：{state['total_profit']:.2f} USDT
2. 总亏损：{state['total_loss']:.2f} USDT
3. 净收益：{state['total_profit'] - state['total_loss']:.2f} USDT
4. 总收益率：{((state['total_profit'] - state['total_loss']) / state['strategy组合']['total_funds'] * 100):.2f}%
5. 最大回撤：待补充（需历史数据统计）
6. 交易总次数：{sum([len(coin_state.get('order_ids', [])) for coin_state in state['coin_states'].values()])} 次
7. 多币种收益明细：
{"\n".join(coin_profits) if coin_profits else "无交易记录"}
8. 当前交易币种：{state['current_coin'] or '未选择'}
9. 资金分配比例：
{json.dumps(state['strategy组合']['funds_distribution'], indent=2, ensure_ascii=False)}
==================================
风险提示：加密货币合约交易风险极高，请谨慎操作！
"""
    with open(f"{report_path}/report_{date}.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"每日报告生成成功：{report_path}/report_{date}.txt")
    return report_content

# -------------------------- ATR计算（保留） --------------------------
def calculate_atr(candles, period=14):
    if len(candles) < period:
        raise Exception(f"K线不足{period}根，无法计算ATR")
    tr_list = []
    for i in range(1, len(candles)):
        current = candles[i]
        prev = candles[i-1]
        tr = max(
            current["high"] - current["low"],
            abs(current["high"] - prev["close"]),
            abs(current["low"] - prev["close"])
        )
        tr_list.append(tr)
    return round(sum(tr_list[-period:])/period, 4)