import logging
import os
import json
import time
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

# -------------------------- 日志配置 --------------------------
def setup_logger():
    logger = logging.getLogger("okx_grid_robot_v2")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 文件处理器（按日期拆分日志）
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/robot_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)
    
    # 日志格式化
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

# -------------------------- API重试装饰器（核心容错） --------------------------
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

# -------------------------- 数据持久化（状态备份与恢复） --------------------------
def init_data_persistence(data_path):
    """初始化数据持久化文件"""
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    if not os.path.exists(data_path):
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump({
                "total_profit": 0.0,  # 总盈利
                "total_loss": 0.0,    # 总亏损
                "current_strategy": {  # 当前策略状态
                    "is_running": False,
                    "base_price": 0.0,
                    "grid_spacing": 0.0,
                    "buy_levels": [],
                    "sell_levels": [],
                    "order_ids": [],  # 已挂订单ID
                    "start_time": "",
                    "floating_pnl": 0.0  # 浮动盈亏
                }
            }, f, ensure_ascii=False, indent=2)

def save_strategy_state(data_path, state):
    """保存策略状态"""
    with open(data_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["current_strategy"].update(state)
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

def update_profit_loss(data_path, profit, loss):
    """更新总盈利/亏损"""
    with open(data_path, "r+", encoding="utf-8") as f:
        data = json.load(f)
        data["total_profit"] += profit
        data["total_loss"] += loss
        f.seek(0)
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.truncate()

def load_strategy_state(data_path):
    """加载策略状态"""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -------------------------- 告警功能（邮件） --------------------------
def send_email_alert(config, subject, content):
    """发送邮件告警"""
    if not config["enable_email_alert"]:
        return
    try:
        msg = MIMEText(content, "plain", "utf-8")
        msg["From"] = config["email_sender"]
        msg["To"] = config["email_receiver"]
        msg["Subject"] = subject
        
        # 连接SMTP服务器
        server = smtplib.SMTP_SSL(config["smtp_server"], config["smtp_port"])
        server.login(config["email_sender"], config["email_password"])
        server.sendmail(config["email_sender"], config["email_receiver"], msg.as_string())
        server.quit()
        logger.info(f"邮件告警发送成功：{subject}")
    except Exception as e:
        logger.error(f"邮件告警发送失败：{str(e)}")

# -------------------------- 每日报告生成 --------------------------
def generate_daily_report(data_path, report_path="reports/"):
    """生成每日运行报告"""
    os.makedirs(report_path, exist_ok=True)
    state = load_strategy_state(data_path)
    date = datetime.now().strftime("%Y%m%d")
    report_content = f"""
欧易网格交易机器人每日报告（{date}）
==================================
1. 总盈利：{state['total_profit']:.2f} USDT
2. 总亏损：{state['total_loss']:.2f} USDT
3. 净收益：{state['total_profit'] - state['total_loss']:.2f} USDT
4. 当日交易次数：{len(state['current_strategy']['order_ids'])} 次
5. 当前策略状态：{'运行中' if state['current_strategy']['is_running'] else '已停止'}
6. 最大回撤：待补充（需历史数据统计）
7. 最优网格档位：待补充（需回测分析）
==================================
风险提示：加密货币合约交易风险极高，请谨慎操作！
"""
    with open(f"{report_path}/report_{date}.txt", "w", encoding="utf-8") as f:
        f.write(report_content)
    logger.info(f"每日报告生成成功：{report_path}/report_{date}.txt")
    return report_content