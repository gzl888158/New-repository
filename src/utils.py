import logging
import os

# -------------------------- 【修改】完善日志配置（添加文件持久化） --------------------------
def setup_logger():
    logger = logging.getLogger("okx_grid_robot")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # 避免重复输出
    
    # 控制台处理器（输出到终端）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 【新增】文件处理器（保存到logs/robot.log）
    os.makedirs("logs", exist_ok=True)  # 自动创建logs目录
    file_handler = logging.FileHandler("logs/robot.log", encoding="utf-8", mode="a")
    file_handler.setLevel(logging.INFO)
    
    # 日志格式化（包含时间、级别、信息）
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# 获取最新日志
def get_logs():
    """读取日志文件，返回最新日志列表"""
    log_file = "logs/robot.log"
    if not os.path.exists(log_file):
        return ["日志文件未生成"]
    
    with open(log_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # 返回最后20条日志（避免日志过多）
        return [line.strip() for line in lines[-20:]]