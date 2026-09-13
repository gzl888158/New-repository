"""
DEPRECATED（企业化收敛，模块 7）：本模块与 monitoring/performance_monitor.py 同名
（PerformanceMonitor），但当前无生产使用，仅被 core/__init__.py 导出。
权威实现为 monitoring/performance_monitor.py（被 core/scheduler.py 实例化），
本模块保留仅作历史参考，禁止新代码引用。

性能监控和优化建议模块 - Performance Monitor and Optimization Advisor

功能：
1. 关键路径性能监控
2. 性能瓶颈识别
3. 优化建议生成
4. 资源使用统计
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable
from loguru import logger
from functools import wraps
from collections import defaultdict
import sqlite3


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 性能统计
        self._function_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "call_count": 0,
            "total_time": 0.0,
            "max_time": 0.0,
            "min_time": float('inf'),
            "error_count": 0
        })
        
        # 资源统计
        self._resource_stats: Dict[str, List[float]] = defaultdict(list)
        self._resource_lock = threading.Lock()
        
        # 慢查询阈值（秒）
        self._slow_query_threshold = config.get("monitoring", {}).get("slow_query_threshold", 1.0)
        
        # 性能瓶颈列表
        self._performance_issues: List[Dict[str, Any]] = []
        
        # 数据库路径
        self._db_path = config.get("sqlite", {}).get("db_path", "./data/trading.db")
        
        logger.info("PerformanceMonitor initialized")
    
    def record_function_call(self, function_name: str, execution_time: float, error: bool = False):
        """记录函数调用性能"""
        stats = self._function_stats[function_name]
        stats["call_count"] += 1
        stats["total_time"] += execution_time
        stats["max_time"] = max(stats["max_time"], execution_time)
        stats["min_time"] = min(stats["min_time"], execution_time)
        if error:
            stats["error_count"] += 1
        
        # 检测慢调用
        if execution_time > self._slow_query_threshold:
            self._performance_issues.append({
                "timestamp": datetime.now().isoformat(),
                "type": "slow_function",
                "function": function_name,
                "execution_time": execution_time,
                "threshold": self._slow_query_threshold
            })
            logger.warning(f"Slow function detected: {function_name} took {execution_time:.2f}s")
    
    def record_resource_usage(self):
        """记录资源使用情况"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()
            
            with self._resource_lock:
                self._resource_stats["cpu_percent"].append(cpu_percent)
                self._resource_stats["memory_percent"].append(memory_info.percent)
                self._resource_stats["process_memory_mb"].append(process_memory.rss / 1024 / 1024)
                
                # 保持最近100个样本
                for key in self._resource_stats:
                    if len(self._resource_stats[key]) > 100:
                        self._resource_stats[key] = self._resource_stats[key][-100:]
        except Exception as e:
            logger.error(f"Failed to record resource usage: {e}")
    
    def get_function_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取函数性能统计"""
        result = {}
        for func_name, stats in self._function_stats.items():
            if stats["call_count"] > 0:
                avg_time = stats["total_time"] / stats["call_count"]
                result[func_name] = {
                    "call_count": stats["call_count"],
                    "avg_time": round(avg_time, 4),
                    "max_time": round(stats["max_time"], 4),
                    "min_time": round(stats["min_time"], 4) if stats["min_time"] != float('inf') else 0,
                    "error_count": stats["error_count"],
                    "error_rate": round(stats["error_count"] / stats["call_count"] * 100, 2)
                }
        return result
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """获取资源使用统计"""
        with self._resource_lock:
            result = {}
            for key, values in self._resource_stats.items():
                if values:
                    result[key] = {
                        "current": round(values[-1], 2),
                        "avg": round(sum(values) / len(values), 2),
                        "max": round(max(values), 2),
                        "min": round(min(values), 2)
                    }
            return result
    
    def get_performance_issues(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取性能问题列表"""
        return self._performance_issues[-limit:]
    
    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """生成优化建议"""
        recommendations = []
        
        # 分析函数性能
        stats = self.get_function_stats()
        for func_name, func_stats in stats.items():
            # 高频调用但平均耗时较长
            if func_stats["call_count"] > 100 and func_stats["avg_time"] > 0.1:
                recommendations.append({
                    "type": "high_frequency_slow_function",
                    "function": func_name,
                    "issue": f"高频率调用({func_stats['call_count']}次)但平均耗时{func_stats['avg_time']:.3f}s",
                    "recommendation": "考虑添加缓存、批量处理或异步优化",
                    "priority": "high"
                })
            
            # 错误率高
            if func_stats["error_rate"] > 5:
                recommendations.append({
                    "type": "high_error_rate",
                    "function": func_name,
                    "issue": f"错误率{func_stats['error_rate']:.2f}%，共{func_stats['error_count']}次错误",
                    "recommendation": "检查错误原因，增加异常处理和重试机制",
                    "priority": "high"
                })
        
        # 分析资源使用
        resource_stats = self.get_resource_stats()
        if "cpu_percent" in resource_stats:
            if resource_stats["cpu_percent"]["avg"] > 80:
                recommendations.append({
                    "type": "high_cpu_usage",
                    "issue": f"平均CPU使用率{resource_stats['cpu_percent']['avg']:.1f}%过高",
                    "recommendation": "优化计算密集型任务，考虑分布式处理",
                    "priority": "medium"
                })
        
        if "process_memory_mb" in resource_stats:
            if resource_stats["process_memory_mb"]["avg"] > 1000:
                recommendations.append({
                    "type": "high_memory_usage",
                    "issue": f"平均内存使用{resource_stats['process_memory_mb']['avg']:.1f}MB过高",
                    "recommendation": "检查内存泄漏，优化数据结构，减少缓存大小",
                    "priority": "medium"
                })
        
        return recommendations
    
    def save_stats_to_db(self):
        """保存性能统计到数据库"""
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            # 创建性能统计表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME,
                    function_name VARCHAR(100),
                    call_count INTEGER,
                    avg_time FLOAT,
                    max_time FLOAT,
                    error_count INTEGER
                )
            """)
            
            # 插入统计数据
            timestamp = datetime.now().isoformat()
            for func_name, stats in self._function_stats.items():
                if stats["call_count"] > 0:
                    cursor.execute("""
                        INSERT INTO performance_stats 
                        (timestamp, function_name, call_count, avg_time, max_time, error_count)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        timestamp,
                        func_name,
                        stats["call_count"],
                        stats["total_time"] / stats["call_count"],
                        stats["max_time"],
                        stats["error_count"]
                    ))
            
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save performance stats: {e}")
        finally:
            if conn:
                conn.close()


# 装饰器：性能监控
def monitor_performance(function_name: str = None):
    """
    性能监控装饰器
    
    用法：
    @monitor_performance("place_order")
    async def place_order(self, ...):
        ...
    """
    def decorator(func):
        name = function_name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            error = False
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error = True
                raise
            finally:
                execution_time = time.time() - start_time
                # 这里应该调用全局PerformanceMonitor，简化版本直接记录日志
                if execution_time > 1.0:
                    logger.warning(f"Slow function {name}: {execution_time:.2f}s")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            error = False
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = True
                raise
            finally:
                execution_time = time.time() - start_time
                if execution_time > 1.0:
                    logger.warning(f"Slow function {name}: {execution_time:.2f}s")
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class PerformanceReporter:
    """性能报告生成器"""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        self._monitor = performance_monitor
    
    def generate_report(self) -> str:
        """生成性能报告"""
        report = []
        report.append("=" * 80)
        report.append(f"性能报告 - {datetime.now().isoformat()}")
        report.append("=" * 80)
        
        # 函数性能统计
        report.append("\n【函数性能统计】")
        stats = self._monitor.get_function_stats()
        if stats:
            for func_name, func_stats in sorted(stats.items(), key=lambda x: x[1]["call_count"], reverse=True)[:10]:
                report.append(f"  {func_name}:")
                report.append(f"    调用次数: {func_stats['call_count']}")
                report.append(f"    平均耗时: {func_stats['avg_time']:.4f}s")
                report.append(f"    最大耗时: {func_stats['max_time']:.4f}s")
                report.append(f"    错误率: {func_stats['error_rate']:.2f}%")
        else:
            report.append("  无统计数据")
        
        # 资源使用统计
        report.append("\n【资源使用统计】")
        resource_stats = self._monitor.get_resource_stats()
        if resource_stats:
            for key, values in resource_stats.items():
                report.append(f"  {key}:")
                report.append(f"    当前: {values['current']:.2f}")
                report.append(f"    平均: {values['avg']:.2f}")
                report.append(f"    最大: {values['max']:.2f}")
        else:
            report.append("  无统计数据")
        
        # 性能问题
        report.append("\n【性能问题】")
        issues = self._monitor.get_performance_issues(limit=10)
        if issues:
            for issue in issues:
                report.append(f"  [{issue['timestamp']}] {issue['type']}: {issue.get('function', 'N/A')}")
        else:
            report.append("  无性能问题")
        
        # 优化建议
        report.append("\n【优化建议】")
        recommendations = self._monitor.get_optimization_recommendations()
        if recommendations:
            for rec in recommendations:
                report.append(f"  [{rec['priority'].upper()}] {rec['type']}: {rec['issue']}")
                report.append(f"    建议: {rec['recommendation']}")
        else:
            report.append("  无优化建议")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)