"""性能监控模块：采集 CPU、内存等系统指标并生成性能报告。"""
import asyncio
import psutil
import time
import requests as http_requests
from collections import deque
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
from core.atomic_writer import atomic_write_json


class PerformanceMonitor:
    def __init__(self, config: Dict[str, Any], alert_manager):
        self.config = config
        self.alert_manager = alert_manager

        self._max_cpu_usage = config["hardware"]["max_cpu_usage"]
        self._max_memory_usage = 85
        # API 延迟告警阈值（毫秒）
        self._api_latency_warning_ms = config.get("monitoring", {}).get("api_latency_warning_ms", 3000)
        self._api_latency_critical_ms = config.get("monitoring", {}).get("api_latency_critical_ms", 8000)

        # Redis 引用（由 scheduler 注入，可选）
        self._redis_cache = None
        self._okx_client = None

        self._metrics: Dict[str, Any] = {}
        self._start_time = datetime.now()

        # API 延迟历史（最近20次）
        self._api_latency_history = deque(maxlen=20)

        self._trade_metrics: Dict[str, Any] = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_pnl": 0.0,
            "total_winning_pnl": 0.0,
            "total_losing_pnl": 0.0,
            "max_win": 0.0,
            "max_loss": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "strategy_pnl": {},
            "daily_trades": {}
        }

        self._monitor_task = None

    def set_dependencies(self, redis_cache=None, okx_client=None):
        """注入 redis 和 okx_client 引用，用于监控其状态"""
        self._redis_cache = redis_cache
        self._okx_client = okx_client

    async def start(self):
        if self._monitor_task is None:
            self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def stop(self):
        """取消监控任务，优雅停止"""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None

    async def _monitor_loop(self):
        while True:
            try:
                await self._collect_metrics()
                await self._check_thresholds()
            except Exception as e:
                logger.exception(f"Monitor loop error: {e}")
            self._export_monitor_status()
            await asyncio.sleep(5)

    def _export_monitor_status(self):
        """导出监控状态到JSON文件，供dashboard读取"""
        import json
        import os
        try:
            export_data = {
                "timestamp": self._metrics.get("timestamp"),
                "cpu": self._metrics.get("cpu", {}),
                "memory": self._metrics.get("memory", {}),
                "redis": self._metrics.get("redis", {}),
                "api_latency_ms": self._metrics.get("api_latency_ms", -1),
                "api_latency_avg_ms": self._metrics.get("api_latency_avg_ms", 0),
                "uptime": self._metrics.get("uptime", 0),
            }
            atomic_write_json("./data/monitor_status.json", export_data)
        except Exception as e:
            logger.debug(f"Failed to export monitor status: {e}")

    async def _collect_metrics(self):
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_usage = psutil.virtual_memory().percent
        disk_usage = psutil.disk_usage('/').percent

        process = psutil.Process()
        process_cpu = process.cpu_percent(interval=0.1)
        process_memory = process.memory_percent()

        network_io = psutil.net_io_counters()

        # Redis 状态监控
        redis_status = self._collect_redis_status()

        # API 延迟监控（探测 OKX 服务器状态）
        api_latency_ms = await self._measure_api_latency()

        self._metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {
                "system": cpu_usage,
                "process": process_cpu,
                "cores": psutil.cpu_count()
            },
            "memory": {
                "system": memory_usage,
                "process": process_memory,
                "available": psutil.virtual_memory().available / 1024 / 1024 / 1024
            },
            "disk": {
                "usage": disk_usage,
                "free": psutil.disk_usage('/').free / 1024 / 1024 / 1024
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv
            },
            "uptime": (datetime.now() - self._start_time).total_seconds(),
            "trading": self._trade_metrics,
            "redis": redis_status,
            "api_latency_ms": api_latency_ms,
            "api_latency_avg_ms": (
                sum(self._api_latency_history) / len(self._api_latency_history)
                if self._api_latency_history else 0
            ),
        }

    def _collect_redis_status(self) -> Dict[str, Any]:
        """Redis 连接状态监控"""
        if self._redis_cache is None:
            return {"available": False, "reason": "not_configured"}

        try:
            is_available = getattr(self._redis_cache, "_redis_available", False)
            ping_ok = False
            if is_available:
                try:
                    ping_ok = self._redis_cache.health_check()
                except Exception:
                    ping_ok = False

            # 内存模式降级状态
            memory_mode = getattr(self._redis_cache, "_memory_cache", None) is not None and not ping_ok

            return {
                "available": ping_ok,
                "configured": is_available,
                "degraded_to_memory": memory_mode,
                "memory_cache_size": len(getattr(self._redis_cache, "_memory_cache", {})) if memory_mode else 0,
            }
        except Exception as e:
            return {"available": False, "error": str(e)}

    async def _measure_api_latency(self) -> float:
        """测量 OKX API 延迟（毫秒），失败返回 -1"""
        if self._okx_client is None:
            return -1
        try:
            start = time.time()
            # 用最轻量的接口探测
            await asyncio.to_thread(self._okx_client.get_account_info)
            elapsed_ms = (time.time() - start) * 1000
            self._api_latency_history.append(elapsed_ms)
            return round(elapsed_ms, 1)
        except Exception as e:
            logger.debug(f"API latency measurement failed: {e}")
            return -1

    async def _check_thresholds(self):
        cpu_usage = self._metrics["cpu"]["system"]
        memory_usage = self._metrics["memory"]["system"]

        if cpu_usage >= self._max_cpu_usage:
            await self.alert_manager.send_system_alert(
                "HIGH_CPU",
                f"CPU usage {cpu_usage:.1f}% exceeds threshold {self._max_cpu_usage}%"
            )

        if memory_usage >= self._max_memory_usage:
            await self.alert_manager.send_system_alert(
                "HIGH_MEMORY",
                f"Memory usage {memory_usage:.1f}% exceeds threshold {self._max_memory_usage}%"
            )

        # API 延迟告警 - P15: 使用滑动窗口平均值替代瞬时值，减少误报
        api_latency = self._metrics.get("api_latency_ms", -1)
        api_latency_avg = self._metrics.get("api_latency_avg_ms", 0)
        # P15: 优先使用平均值（20次滑动窗口），避免单次尖峰触发误报
        effective_latency = api_latency_avg if api_latency_avg > 0 else api_latency
        if effective_latency > 0:
            if effective_latency >= self._api_latency_critical_ms:
                await self.alert_manager.send_system_alert(
                    "API_LATENCY_CRITICAL",
                    f"OKX API latency avg {effective_latency:.0f}ms exceeds critical threshold {self._api_latency_critical_ms}ms"
                )
            elif effective_latency >= self._api_latency_warning_ms:
                await self.alert_manager.send_system_alert(
                    "API_LATENCY_WARNING",
                    f"OKX API latency avg {effective_latency:.0f}ms exceeds warning threshold {self._api_latency_warning_ms}ms"
                )

        # Redis 降级告警（仅当Redis已配置但不可用时触发，抑制config禁用时的虚假告警）
        redis_status = self._metrics.get("redis", {})
        if redis_status.get("degraded_to_memory") and redis_status.get("configured"):
            await self.alert_manager.send_system_alert(
                "REDIS_DEGRADED",
                f"Redis unavailable, system degraded to memory cache mode (size={redis_status.get('memory_cache_size', 0)})"
            )

    def update_trade_result(self, strategy_name: str, symbol: str, pnl: float):
        self._trade_metrics["total_trades"] += 1

        if pnl >= 0:
            self._trade_metrics["winning_trades"] += 1
            self._trade_metrics["total_winning_pnl"] += pnl
            if pnl > self._trade_metrics["max_win"]:
                self._trade_metrics["max_win"] = pnl
        else:
            self._trade_metrics["losing_trades"] += 1
            self._trade_metrics["total_losing_pnl"] += abs(pnl)
            if abs(pnl) > self._trade_metrics["max_loss"]:
                self._trade_metrics["max_loss"] = abs(pnl)

        self._trade_metrics["total_pnl"] += pnl

        if strategy_name not in self._trade_metrics["strategy_pnl"]:
            self._trade_metrics["strategy_pnl"][strategy_name] = {"total": 0, "count": 0}
        self._trade_metrics["strategy_pnl"][strategy_name]["total"] += pnl
        self._trade_metrics["strategy_pnl"][strategy_name]["count"] += 1

        today = datetime.now().date()
        date_key = str(today)
        if date_key not in self._trade_metrics["daily_trades"]:
            self._trade_metrics["daily_trades"][date_key] = {"total": 0, "winning": 0, "losing": 0, "pnl": 0}
        self._trade_metrics["daily_trades"][date_key]["total"] += 1
        self._trade_metrics["daily_trades"][date_key]["winning" if pnl >= 0 else "losing"] += 1
        self._trade_metrics["daily_trades"][date_key]["pnl"] += pnl

        if self._trade_metrics["total_trades"] > 0:
            self._trade_metrics["win_rate"] = (
                self._trade_metrics["winning_trades"] / self._trade_metrics["total_trades"]
            )

            total_winning = self._trade_metrics["total_winning_pnl"]
            total_losing = self._trade_metrics["total_losing_pnl"]
            if total_losing > 0:
                self._trade_metrics["profit_factor"] = total_winning / total_losing
            elif total_winning > 0:
                self._trade_metrics["profit_factor"] = 999.0

    def get_metrics(self) -> Dict[str, Any]:
        return self._metrics

    def get_uptime(self) -> str:
        uptime_seconds = self._metrics.get("uptime", 0)
        hours = int(uptime_seconds // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_trade_summary(self) -> Dict[str, Any]:
        return {
            "total_trades": self._trade_metrics["total_trades"],
            "winning_trades": self._trade_metrics["winning_trades"],
            "losing_trades": self._trade_metrics["losing_trades"],
            "win_rate": self._trade_metrics["win_rate"],
            "total_pnl": self._trade_metrics["total_pnl"],
            "max_win": self._trade_metrics["max_win"],
            "max_loss": self._trade_metrics["max_loss"],
            "profit_factor": self._trade_metrics["profit_factor"],
            "strategy_pnl": self._trade_metrics["strategy_pnl"]
        }