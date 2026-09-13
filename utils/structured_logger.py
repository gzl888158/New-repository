"""
结构化日志配置
支持日志轮转、日志分级、结构化输出
"""
import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger
import threading


class StructuredLogFormatter:
    """结构化日志格式化器"""
    
    def __init__(self, include_extra: bool = True):
        self.include_extra = include_extra
    
    def format(self, record: Dict[str, Any]) -> str:
        """格式化日志记录"""
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "message": record["message"],
        }
        
        if record.get("extra"):
            log_entry["extra"] = record["extra"]
        
        if record.get("exception"):
            log_entry["exception"] = self._format_exception(record["exception"])
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)
    
    def _format_exception(self, exception: Any) -> str:
        """格式化异常信息"""
        if hasattr(exception, "traceback"):
            return str(exception.traceback)
        return str(exception)


class LogConfig:
    """日志配置"""
    
    def __init__(
        self,
        log_dir: str = "./logs",
        log_level: str = "INFO",
        rotation_size: str = "10 MB",
        retention_days: str = "7 days",
        compression: str = "zip",
        json_format: bool = False,
        console_output: bool = True,
        file_output: bool = True,
        enqueue: bool = True,
        backtrace: bool = True,
        diagnose: bool = False,
    ):
        self.log_dir = log_dir
        self.log_level = log_level
        self.rotation_size = rotation_size
        self.retention_days = retention_days
        self.compression = compression
        self.json_format = json_format
        self.console_output = console_output
        self.file_output = file_output
        self.enqueue = enqueue
        self.backtrace = backtrace
        self.diagnose = diagnose


class StructuredLogger:
    """结构化日志管理器"""
    
    _instance = None
    _lock = threading.Lock()
    _configured = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def configure(self, config: Optional[LogConfig] = None):
        """配置日志"""
        if self._configured:
            return
        
        if config is None:
            config = LogConfig()
        
        logger.remove()
        
        if config.console_output:
            self._add_console_handler(config)
        
        if config.file_output:
            self._add_file_handlers(config)
        
        logger.add(
            lambda _: None,
            level=config.log_level,
            enqueue=config.enqueue,
        )
        
        self._configured = True
    
    def _add_console_handler(self, config: LogConfig):
        """添加控制台处理器"""
        if config.json_format:
            format_str = "{message}"
        else:
            format_str = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        
        logger.add(
            sys.stdout,
            format=format_str,
            level=config.log_level,
            enqueue=config.enqueue,
            colorize=not config.json_format,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
        )
    
    def _add_file_handlers(self, config: LogConfig):
        """添加文件处理器"""
        os.makedirs(config.log_dir, exist_ok=True)
        
        if config.json_format:
            format_str = "{message}"
        else:
            format_str = (
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
                "{level: <8} | "
                "{name}:{function}:{line} | "
                "{message}"
            )
        
        logger.add(
            os.path.join(config.log_dir, "trading_{time:YYYY-MM-DD}.log"),
            format=format_str,
            level=config.log_level,
            rotation=config.rotation_size,
            retention=config.retention_days,
            compression=config.compression,
            enqueue=config.enqueue,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
        )
        
        logger.add(
            os.path.join(config.log_dir, "error_{time:YYYY-MM-DD}.log"),
            format=format_str,
            level="ERROR",
            rotation=config.rotation_size,
            retention=config.retention_days,
            compression=config.compression,
            enqueue=config.enqueue,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
        )
        
        logger.add(
            os.path.join(config.log_dir, "trade_{time:YYYY-MM-DD}.log"),
            format=format_str,
            level="INFO",
            rotation=config.rotation_size,
            retention=config.retention_days,
            compression=config.compression,
            enqueue=config.enqueue,
            backtrace=config.backtrace,
            diagnose=config.diagnose,
            filter=lambda record: record["extra"].get("trade", False),
        )
    
    def bind_trade(self, **kwargs):
        """绑定交易上下文"""
        return logger.bind(trade=True, **kwargs)
    
    def bind_strategy(self, strategy_name: str, **kwargs):
        """绑定策略上下文"""
        return logger.bind(strategy=strategy_name, **kwargs)
    
    def bind_symbol(self, symbol: str, **kwargs):
        """绑定交易对上下文"""
        return logger.bind(symbol=symbol, **kwargs)


class LogAnalyzer:
    """日志分析器"""
    
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
    
    def analyze_errors(self, hours: int = 24) -> Dict[str, Any]:
        """分析错误日志"""
        error_log = os.path.join(self.log_dir, "error.log")
        
        if not os.path.exists(error_log):
            return {"errors": [], "total": 0}
        
        errors = []
        cutoff_time = datetime.now().timestamp() - hours * 3600
        
        try:
            with open(error_log, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        if "ERROR" in line or "CRITICAL" in line:
                            errors.append({
                                "line": line.strip(),
                                "timestamp": self._extract_timestamp(line),
                            })
                    except Exception:
                        continue
            
            errors = [e for e in errors if e.get("timestamp")]
            
            return {
                "errors": errors[-100:],
                "total": len(errors),
                "last_24h": len([e for e in errors if e.get("timestamp", 0) >= cutoff_time]),
            }
            
        except Exception as e:
            return {"errors": [], "total": 0, "error": str(e)}
    
    def count_by_level(self, hours: int = 24) -> Dict[str, int]:
        """按级别统计日志数量"""
        log_file = os.path.join(self.log_dir, f"trading_{datetime.now().strftime('%Y-%m-%d')}.log")
        
        if not os.path.exists(log_file):
            return {}
        
        counts = {
            "DEBUG": 0,
            "INFO": 0,
            "WARNING": 0,
            "ERROR": 0,
            "CRITICAL": 0,
        }
        
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    for level in counts:
                        if f"| {level: <8} |" in line or f"| {level}" in line:
                            counts[level] += 1
                            break
            
            return counts
            
        except Exception:
            return counts
    
    def search(self, keyword: str, hours: int = 24, limit: int = 100) -> List[Dict[str, Any]]:
        """搜索日志"""
        results = []
        cutoff_time = datetime.now().timestamp() - hours * 3600
        
        for filename in os.listdir(self.log_dir):
            if not filename.endswith(".log"):
                continue
            
            filepath = os.path.join(self.log_dir, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f):
                        if keyword.lower() in line.lower():
                            results.append({
                                "file": filename,
                                "line_num": line_num,
                                "content": line.strip(),
                            })
                            
                            if len(results) >= limit:
                                return results
            except Exception:
                continue
        
        return results
    
    def _extract_timestamp(self, line: str) -> Optional[float]:
        """提取时间戳"""
        import re
        
        match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if match:
            try:
                dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                return dt.timestamp()
            except ValueError:
                pass
        
        return None
    
    def get_log_stats(self) -> Dict[str, Any]:
        """获取日志统计"""
        stats = {
            "total_size_mb": 0,
            "files": [],
            "oldest_file": None,
            "newest_file": None,
        }
        
        if not os.path.exists(self.log_dir):
            return stats
        
        for filename in os.listdir(self.log_dir):
            if not filename.endswith(".log"):
                continue
            
            filepath = os.path.join(self.log_dir, filename)
            file_stat = os.stat(filepath)
            
            stats["total_size_mb"] += file_stat.st_size / (1024 * 1024)
            
            file_info = {
                "name": filename,
                "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            }
            
            stats["files"].append(file_info)
        
        stats["total_size_mb"] = round(stats["total_size_mb"], 2)
        
        if stats["files"]:
            stats["files"].sort(key=lambda x: x["modified"], reverse=True)
            stats["newest_file"] = stats["files"][0]["name"]
            stats["oldest_file"] = stats["files"][-1]["name"]
        
        return stats


def configure_logging(
    log_dir: str = "./logs",
    log_level: str = "INFO",
    json_format: bool = False,
    rotation_size: str = "10 MB",
    retention_days: str = "7 days",
) -> StructuredLogger:
    """配置日志系统"""
    config = LogConfig(
        log_dir=log_dir,
        log_level=log_level,
        rotation_size=rotation_size,
        retention_days=retention_days,
        json_format=json_format,
    )
    
    sl = StructuredLogger()
    sl.configure(config)
    
    return sl


def get_logger(name: Optional[str] = None) -> "logger":
    """获取日志器"""
    if name:
        return logger.bind(name=name)
    return logger