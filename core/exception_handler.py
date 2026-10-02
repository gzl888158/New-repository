"""
全局异常处理和监控框架 - Global Exception Handler and Monitoring Framework

功能：
1. 统一异常处理（避免空异常处理）
2. 异常分类和分级
3. 关键路径监控
4. 自动恢复机制
"""

import asyncio
import traceback
import sys
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from loguru import logger
from functools import wraps
import sqlite3


class ExceptionSeverity:
    """异常严重程度"""
    LOW = "low"           # 可忽略，仅记录
    MEDIUM = "medium"     # 需要关注，但不影响运行
    HIGH = "high"         # 影响功能，需要立即处理
    CRITICAL = "critical" # 系统级错误，可能导致崩溃


class ExceptionCategory:
    """异常分类"""
    NETWORK = "network"           # 网络错误
    API = "api"                   # API调用错误
    DATABASE = "database"         # 数据库错误
    VALIDATION = "validation"     # 数据验证错误
    BUSINESS_LOGIC = "business"   # 业务逻辑错误
    SYSTEM = "system"             # 系统级错误


class GlobalExceptionHandler:
    """全局异常处理器"""
    
    def __init__(self, config: Dict[str, Any], alert_manager=None):
        self.config = config
        self._alert_manager = alert_manager
        self._db_path = config.get("sqlite", {}).get("db_path", "./data/trading.db")
        self._init_exception_table()
        
        # 异常统计
        self._exception_stats: Dict[str, int] = {}
        
        # 自动恢复配置
        self._auto_recovery_enabled = config.get("monitoring", {}).get("auto_recovery", True)
        self._recovery_handlers: Dict[str, Callable] = {}
        
        logger.info("GlobalExceptionHandler initialized")
    
    def _init_exception_table(self):
        """初始化异常记录表"""
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exception_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    category VARCHAR(30),
                    severity VARCHAR(20),
                    module VARCHAR(100),
                    function VARCHAR(100),
                    exception_type VARCHAR(100),
                    message TEXT,
                    traceback TEXT,
                    handled BOOLEAN,
                    recovery_attempted BOOLEAN
                )
            """)
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to init exception_log table: {e}")
        finally:
            if conn:
                conn.close()
    
    def register_recovery_handler(self, exception_type: str, handler: Callable):
        """注册异常恢复处理器"""
        self._recovery_handlers[exception_type] = handler
        logger.info(f"Registered recovery handler for {exception_type}")
    
    def handle_exception(
        self,
        exception: Exception,
        context: Dict[str, Any] = None,
        module: str = "",
        function: str = "",
        severity: str = ExceptionSeverity.MEDIUM,
        category: str = ExceptionCategory.SYSTEM,
        reraise: bool = False
    ):
        """
        统一异常处理入口
        
        参数：
        - exception: 异常对象
        - context: 上下文信息
        - module: 模块名
        - function: 函数名
        - severity: 严重程度
        - category: 异常分类
        - reraise: 是否重新抛出异常
        """
        # 记录异常
        self._record_exception(exception, context, module, function, severity, category)
        
        # 更新统计
        exception_key = f"{category}:{type(exception).__name__}"
        self._exception_stats[exception_key] = self._exception_stats.get(exception_key, 0) + 1
        
        # 发送告警（HIGH和CRITICAL级别）
        if severity in [ExceptionSeverity.HIGH, ExceptionSeverity.CRITICAL]:
            self._send_alert(exception, context, module, function, severity, category)
        
        # 尝试自动恢复
        recovery_attempted = False
        if self._auto_recovery_enabled:
            recovery_attempted = self._attempt_recovery(exception, category)
        
        # 记录到数据库
        self._save_exception(exception, context, module, function, severity, category, recovery_attempted)
        
        # 根据严重程度决定是否重新抛出
        if reraise or severity == ExceptionSeverity.CRITICAL:
            raise exception
    
    def _record_exception(self, exception: Exception, context: Dict[str, Any], module: str, function: str, severity: str, category: str):
        """记录异常到日志"""
        log_message = f"[{severity.upper()}] [{category.upper()}] {module}.{function}: {type(exception).__name__}: {str(exception)}"
        
        if context:
            log_message += f" | Context: {context}"
        
        if severity == ExceptionSeverity.CRITICAL:
            logger.critical(log_message)
        elif severity == ExceptionSeverity.HIGH:
            logger.error(log_message)
        elif severity == ExceptionSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _send_alert(self, exception: Exception, context: Dict[str, Any], module: str, function: str, severity: str, category: str):
        """发送告警"""
        try:
            if not self._alert_manager:
                return

            alert_message = f"Exception Alert: {module}.{function} - {type(exception).__name__}: {str(exception)}"
            # 内部严重级别映射为告警级别（大写，兼容 AlertManager._should_filter）
            alert_severity = severity.upper() if isinstance(severity, str) else "ERROR"

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # 无运行中的事件循环，无法异步发送告警
                logger.warning(f"Cannot send alert asynchronously (no running event loop): {alert_message}")
                return

            coro = self._alert_manager.send_alert(
                alert_type=type(exception).__name__,
                message=alert_message,
                severity=alert_severity,
            )
            if asyncio.iscoroutine(coro):
                loop.create_task(coro)
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
    
    def _attempt_recovery(self, exception: Exception, category: str) -> bool:
        """尝试自动恢复"""
        exception_type = type(exception).__name__
        handler = self._recovery_handlers.get(exception_type)
        
        if handler:
            try:
                logger.info(f"Attempting recovery for {exception_type}")
                handler(exception)
                logger.info(f"Recovery successful for {exception_type}")
                return True
            except Exception as e:
                logger.error(f"Recovery failed for {exception_type}: {e}")
        
        return False
    
    def _save_exception(self, exception: Exception, context: Dict[str, Any], module: str, function: str, severity: str, category: str, recovery_attempted: bool):
        """保存异常到数据库"""
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO exception_log 
                (timestamp, category, severity, module, function, exception_type, message, traceback, handled, recovery_attempted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                category,
                severity,
                module,
                function,
                type(exception).__name__,
                str(exception),
                ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__)),
                True,
                recovery_attempted
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save exception to database: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_exception_stats(self) -> Dict[str, int]:
        """获取异常统计"""
        return dict(self._exception_stats)


# 装饰器：自动异常处理
def handle_exceptions(
    module: str = "",
    severity: str = ExceptionSeverity.MEDIUM,
    category: str = ExceptionCategory.SYSTEM,
    reraise: bool = True  # 默认重新抛出异常，避免静默吞噬
):
    """
    异常处理装饰器
    
    用法：
    @handle_exceptions(module="order_executor", severity=ExceptionSeverity.HIGH, category=ExceptionCategory.API)
    async def place_order(self, ...):
        ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                # 从全局获取handler（需要在scheduler中注入）
                # 这里使用简化版本，仅记录日志
                logger.error(f"[{severity.upper()}] [{category.upper()}] {module}.{func.__name__}: {type(e).__name__}: {str(e)}")
                if reraise or severity == ExceptionSeverity.CRITICAL:
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[{severity.upper()}] [{category.upper()}] {module}.{func.__name__}: {type(e).__name__}: {str(e)}")
                if reraise or severity == ExceptionSeverity.CRITICAL:
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# 全局异常钩子
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """全局未捕获异常处理器"""
    logger.critical(f"Uncaught exception: {exc_type.__name__}: {exc_value}")
    logger.critical("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    
    # 保存到文件
    try:
        with open("data/crash_log.txt", "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Crash at {datetime.now().isoformat()}\n")
            f.write(f"Exception: {exc_type.__name__}: {exc_value}\n")
            f.write("Traceback:\n")
            f.write("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
    except Exception:
        pass
    
    # 调用原始钩子
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def install_global_exception_handler():
    """安装全局异常处理器（同步 + asyncio 双兜底）

    P0 修复：原实现仅设置 sys.excepthook，无法捕获 asyncio 任务内抛出的异常
    （asyncio 通过 loop.call_exception_handler 处理，不走 sys.excepthook），
    导致协程异常被静默吞掉。现额外设置 loop.set_exception_handler。
    """
    sys.excepthook = global_exception_handler

    def _async_exception_handler(loop, context):
        exception = context.get("exception")
        message = context.get("message", "")
        if exception is not None:
            logger.critical(
                f"[ASYNC_UNCAUGHT] task={context.get('task')} "
                f"{type(exception).__name__}: {exception}"
            )
            try:
                with open("data/crash_log.txt", "a", encoding="utf-8") as f:
                    f.write(f"\n{'='*80}\n")
                    f.write(f"Async crash at {datetime.now().isoformat()}\n")
                    f.write(f"Exception: {type(exception).__name__}: {exception}\n")
                    f.write("Traceback:\n")
                    f.write("".join(traceback.format_exception(type(exception), exception, exception.__traceback__)))
            except Exception:
                pass
        else:
            logger.critical(f"[ASYNC_UNCAUGHT] {message}")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # 无运行中的事件循环（如未在 async 上下文中调用）
        try:
            loop = asyncio.get_event_loop()
        except Exception:
            loop = None

    if loop is not None:
        try:
            loop.set_exception_handler(_async_exception_handler)
        except Exception as e:
            logger.warning(f"Failed to set asyncio exception handler: {e}")

    logger.info("Global exception handler installed (sync + asyncio)")