import logging
import time
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime

def setup_logger():
    """配置按天轮转的日志"""
    logger = logging.getLogger("okx_trader")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return logger
    
    # 日志格式
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件输出（按天轮转，保留7天）
    file_handler = TimedRotatingFileHandler(
        "logs/okx_trader.log",
        when="D",
        interval=1,
        backupCount=7,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def timestamp_to_datetime(ts: int) -> str:
    """毫秒时间戳转格式化字符串"""
    return datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d %H:%M:%S")

def get_current_timestamp() -> int:
    """获取当前毫秒时间戳"""
    return int(time.time() * 1000)