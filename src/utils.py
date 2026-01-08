import logging
import os
from datetime import datetime

# 日志配置
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
LOG_FILE = os.path.join(LOG_DIR, f"okx_robot_{datetime.now().strftime('%Y%m%d')}.log")

def setup_logger() -> logging.Logger:
    """初始化日志"""
    logger = logging.getLogger("OKX_Grid_Robot")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    
    # 文件处理器
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_logs() -> list:
    """获取最新日志"""
    if not os.path.exists(LOG_FILE):
        return ["日志文件未生成"]
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        return f.readlines()