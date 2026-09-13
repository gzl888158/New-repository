"""健康评分模块：定义健康等级并综合各组件指标计算系统整体健康分数。"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional
from loguru import logger
from enum import Enum
from core.atomic_writer import atomic_write_json


class HealthLevel(Enum):
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"


class HealthComponent(Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    API = "api"
    REDIS = "redis"
    DATABASE = "database"
    STRATEGIES = "strategies"
    POSITIONS = "positions"
    SYSTEM = "system"


class ComponentScore:
    def __init__(self, component: HealthComponent, score: float, level: HealthLevel, 
                 details: Dict[str, Any] = None):
        self.component = component
        self.score = score
        self.level = level
        self.details = details or {}
        self.timestamp = datetime.now()


class HealthScorer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._component_weights: Dict[HealthComponent, float] = {
            HealthComponent.CPU: 0.08,
            HealthComponent.MEMORY: 0.10,
            HealthComponent.DISK: 0.05,
            HealthComponent.NETWORK: 0.05,
            HealthComponent.API: 0.15,
            HealthComponent.REDIS: 0.05,
            HealthComponent.DATABASE: 0.05,
            HealthComponent.STRATEGIES: 0.25,
            HealthComponent.POSITIONS: 0.15,
            HealthComponent.SYSTEM: 0.12,
        }
        
        self._history = deque(maxlen=100)
        self._last_score = None
        self._last_update = None
        
        self._strategy_health_cache: Dict[str, Dict[str, Any]] = {}
        self._position_health_cache: Dict[str, Dict[str, Any]] = {}
        
        self._dependencies = {}
    
    def set_dependencies(self, **kwargs):
        self._dependencies.update(kwargs)
    
    def _get_component_weight(self, component: HealthComponent) -> float:
        return self._component_weights.get(component, 0.1)
    
    def _calculate_cpu_score(self, metrics: Dict[str, Any]) -> ComponentScore:
        cpu_usage = metrics.get("cpu", {}).get("system", 0)
        process_cpu = metrics.get("cpu", {}).get("process", 0)
        
        if cpu_usage < 40:
            score = 100
            level = HealthLevel.EXCELLENT
        elif cpu_usage < 60:
            score = 85
            level = HealthLevel.GOOD
        elif cpu_usage < 80:
            score = 65
            level = HealthLevel.WARNING
        else:
            score = max(0, 100 - (cpu_usage - 80) * 2)
            level = HealthLevel.CRITICAL
        
        return ComponentScore(
            HealthComponent.CPU,
            score,
            level,
            {"system_usage_pct": cpu_usage, "process_usage_pct": process_cpu}
        )
    
    def _calculate_memory_score(self, metrics: Dict[str, Any]) -> ComponentScore:
        memory_usage = metrics.get("memory", {}).get("system", 0)
        available_gb = metrics.get("memory", {}).get("available", 0)
        
        if memory_usage < 50:
            score = 100
            level = HealthLevel.EXCELLENT
        elif memory_usage < 70:
            score = 85
            level = HealthLevel.GOOD
        elif memory_usage < 85:
            score = 60
            level = HealthLevel.WARNING
        else:
            score = max(0, 100 - (memory_usage - 85) * 4)
            level = HealthLevel.CRITICAL
        
        return ComponentScore(
            HealthComponent.MEMORY,
            score,
            level,
            {"system_usage_pct": memory_usage, "available_gb": round(available_gb, 2)}
        )
    
    def _calculate_disk_score(self, metrics: Dict[str, Any]) -> ComponentScore:
        disk_usage = metrics.get("disk", {}).get("usage", 0)
        free_gb = metrics.get("disk", {}).get("free", 0)
        
        if disk_usage < 60:
            score = 100
            level = HealthLevel.EXCELLENT
        elif disk_usage < 75:
            score = 85
            level = HealthLevel.GOOD
        elif disk_usage < 90:
            score = 60
            level = HealthLevel.WARNING
        else:
            score = max(0, 100 - (disk_usage - 90) * 5)
            level = HealthLevel.CRITICAL
        
        return ComponentScore(
            HealthComponent.DISK,
            score,
            level,
            {"usage_pct": disk_usage, "free_gb": round(free_gb, 2)}
        )
    
    def _calculate_network_score(self, metrics: Dict[str, Any]) -> ComponentScore:
        bytes_sent = metrics.get("network", {}).get("bytes_sent", 0)
        bytes_recv = metrics.get("network", {}).get("bytes_recv", 0)
        
        if bytes_sent > 0 or bytes_recv > 0:
            score = 100
            level = HealthLevel.EXCELLENT
        else:
            score = 50
            level = HealthLevel.WARNING
        
        return ComponentScore(
            HealthComponent.NETWORK,
            score,
            level,
            {"bytes_sent": bytes_sent, "bytes_recv": bytes_recv}
        )
    
    def _calculate_api_score(self, metrics: Dict[str, Any]) -> ComponentScore:
        latency_ms = metrics.get("api_latency_ms", 0)
        avg_latency_ms = metrics.get("api_latency_avg_ms", 0)
        
        if latency_ms < 0:
            score = 0
            level = HealthLevel.CRITICAL
        elif avg_latency_ms < 500:
            score = 100
            level = HealthLevel.EXCELLENT
        elif avg_latency_ms < 1500:
            score = 85
            level = HealthLevel.GOOD
        elif avg_latency_ms < 3000:
            score = 60
            level = HealthLevel.WARNING
        else:
            score = max(0, 100 - (avg_latency_ms - 3000) / 50)
            level = HealthLevel.CRITICAL
        
        return ComponentScore(
            HealthComponent.API,
            score,
            level,
            {"latency_ms": round(latency_ms, 1), "avg_latency_ms": round(avg_latency_ms, 1)}
        )
    
    def _calculate_redis_score(self, metrics: Dict[str, Any]) -> ComponentScore:
        redis_status = metrics.get("redis", {})
        available = redis_status.get("available", False)
        degraded = redis_status.get("degraded_to_memory", False)
        
        if available:
            score = 100
            level = HealthLevel.EXCELLENT
        elif degraded:
            score = 70
            level = HealthLevel.WARNING
        else:
            score = 30
            level = HealthLevel.CRITICAL
        
        return ComponentScore(
            HealthComponent.REDIS,
            score,
            level,
            {"available": available, "degraded_to_memory": degraded}
        )
    
    def _calculate_database_score(self, metrics: Dict[str, Any]) -> ComponentScore:
        try:
            import sqlite3
            db_path = self.config.get("sqlite", {}).get("db_path", "./data/trading.db")
            conn = sqlite3.connect(db_path, timeout=2)
            conn.execute("SELECT 1")
            conn.close()
            return ComponentScore(
                HealthComponent.DATABASE,
                100,
                HealthLevel.EXCELLENT,
                {"status": "connected"}
            )
        except Exception as e:
            return ComponentScore(
                HealthComponent.DATABASE,
                30,
                HealthLevel.CRITICAL,
                {"status": "error", "error": str(e)}
            )
    
    def _calculate_strategy_score(self) -> ComponentScore:
        strategy_health = self._strategy_health_cache
        
        if not strategy_health:
            return ComponentScore(
                HealthComponent.STRATEGIES,
                70,
                HealthLevel.WARNING,
                {"status": "no_data", "strategies": 0}
            )
        
        total_strategies = len(strategy_health)
        healthy_strategies = sum(1 for s in strategy_health.values() if s.get("healthy", False))
        
        if total_strategies == 0:
            score = 50
            level = HealthLevel.WARNING
        else:
            health_ratio = healthy_strategies / total_strategies
            if health_ratio >= 0.9:
                score = 100
                level = HealthLevel.EXCELLENT
            elif health_ratio >= 0.7:
                score = 80
                level = HealthLevel.GOOD
            elif health_ratio >= 0.5:
                score = 55
                level = HealthLevel.WARNING
            else:
                score = max(0, health_ratio * 100)
                level = HealthLevel.CRITICAL
        
        return ComponentScore(
            HealthComponent.STRATEGIES,
            score,
            level,
            {
                "total_strategies": total_strategies,
                "healthy_strategies": healthy_strategies,
                "health_ratio": round(health_ratio, 2) if total_strategies > 0 else 0
            }
        )
    
    def _calculate_position_score(self) -> ComponentScore:
        position_health = self._position_health_cache
        
        if not position_health:
            return ComponentScore(
                HealthComponent.POSITIONS,
                100,
                HealthLevel.EXCELLENT,
                {"status": "no_positions", "total_positions": 0}
            )
        
        total_positions = len(position_health)
        risky_positions = sum(1 for p in position_health.values() if p.get("risky", False))
        
        if risky_positions == 0:
            score = 100
            level = HealthLevel.EXCELLENT
        elif risky_positions <= total_positions * 0.2:
            score = 80
            level = HealthLevel.GOOD
        elif risky_positions <= total_positions * 0.5:
            score = 55
            level = HealthLevel.WARNING
        else:
            score = max(0, 100 - (risky_positions / total_positions) * 100)
            level = HealthLevel.CRITICAL
        
        return ComponentScore(
            HealthComponent.POSITIONS,
            score,
            level,
            {
                "total_positions": total_positions,
                "risky_positions": risky_positions,
                "risk_ratio": round(risky_positions / total_positions, 2) if total_positions > 0 else 0
            }
        )
    
    def _calculate_system_score(self, metrics: Dict[str, Any]) -> ComponentScore:
        uptime_seconds = metrics.get("uptime", 0)
        
        if uptime_seconds < 300:
            score = 70
            level = HealthLevel.WARNING
        elif uptime_seconds < 3600:
            score = 90
            level = HealthLevel.GOOD
        else:
            score = 100
            level = HealthLevel.EXCELLENT
        
        return ComponentScore(
            HealthComponent.SYSTEM,
            score,
            level,
            {"uptime_seconds": uptime_seconds}
        )
    
    def calculate_overall_score(self, metrics: Dict[str, Any] = None) -> Dict[str, Any]:
        if metrics is None:
            metrics = {}
        
        scores = []
        
        scores.append(self._calculate_cpu_score(metrics))
        scores.append(self._calculate_memory_score(metrics))
        scores.append(self._calculate_disk_score(metrics))
        scores.append(self._calculate_network_score(metrics))
        scores.append(self._calculate_api_score(metrics))
        scores.append(self._calculate_redis_score(metrics))
        scores.append(self._calculate_database_score(metrics))
        scores.append(self._calculate_strategy_score())
        scores.append(self._calculate_position_score())
        scores.append(self._calculate_system_score(metrics))
        
        total_weighted_score = sum(
            s.score * self._get_component_weight(s.component)
            for s in scores
        )
        
        # 口径归一化：组件权重总和为 1.05（历史遗留），直接加权会使总分上限超 100、
        # 阈值分级失真；按实际权重和归一化回 [0, 100] 尺度。
        weight_sum = sum(self._get_component_weight(s.component) for s in scores)
        if weight_sum > 0:
            total_weighted_score = total_weighted_score / weight_sum

        overall_score = round(total_weighted_score, 1)
        
        if overall_score >= 90:
            overall_level = HealthLevel.EXCELLENT
        elif overall_score >= 75:
            overall_level = HealthLevel.GOOD
        elif overall_score >= 60:
            overall_level = HealthLevel.WARNING
        else:
            overall_level = HealthLevel.CRITICAL
        
        result = {
            "overall_score": overall_score,
            "overall_level": overall_level.value,
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        for s in scores:
            result["components"][s.component.value] = {
                "score": s.score,
                "level": s.level.value,
                "details": s.details
            }
        
        self._last_score = result
        self._last_update = datetime.now()
        self._history.append(result)
        
        return result
    
    def update_strategy_health(self, strategy_name: str, healthy: bool, 
                              details: Dict[str, Any] = None):
        self._strategy_health_cache[strategy_name] = {
            "healthy": healthy,
            "details": details or {},
            "last_update": datetime.now().isoformat()
        }
    
    def update_position_health(self, symbol: str, risky: bool, 
                              details: Dict[str, Any] = None):
        self._position_health_cache[symbol] = {
            "risky": risky,
            "details": details or {},
            "last_update": datetime.now().isoformat()
        }
    
    def get_last_score(self) -> Optional[Dict[str, Any]]:
        return self._last_score
    
    def get_score_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        return list(self._history)[-limit:]
    
    def export_status(self):
        os.makedirs("./data", exist_ok=True)
        
        data = {
            "last_score": self._last_score,
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "strategy_health": self._strategy_health_cache,
            "position_health": self._position_health_cache
        }
        
        atomic_write_json("./data/health_status.json", data)
