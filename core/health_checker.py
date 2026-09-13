"""
生产级健康检查器 - Production Health Checker

实现 Kubernetes 风格的多级健康探针:
  - Liveness Probe:  进程是否存活（最轻量，仅检查自身）
  - Readiness Probe:  是否准备好接收流量（依赖检查通过）
  - Startup Probe:    启动是否完成（初始化阶段专用）

功能:
  1. 多级探针：liveness / readiness / startup
  2. 依赖检查：OKX API、数据库、Redis、文件系统、内存
  3. 健康评分：加权聚合各组件状态
  4. 健康历史：保留最近N次检查结果
  5. 状态回调：状态变更通知
  6. 自动修复建议：检测到问题时提供修复建议
"""
import os
import time
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime

from loguru import logger


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"        # 一切正常
    DEGRADED = "degraded"      # 部分降级，但仍可服务
    UNHEALTHY = "unhealthy"    # 不可用
    STARTING = "starting"      # 启动中
    UNKNOWN = "unknown"        # 未知


class ProbeType(Enum):
    """探针类型"""
    LIVENESS = "liveness"      # 存活探针
    READINESS = "readiness"    # 就绪探针
    STARTUP = "startup"        # 启动探针


@dataclass
class HealthProbeResult:
    """单个探针检查结果"""
    name: str
    status: HealthStatus
    probe_type: ProbeType
    latency_ms: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    checked_at: str = ""

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetime.now().isoformat()


@dataclass
class DependencyStatus:
    """依赖组件状态"""
    name: str
    status: HealthStatus
    latency_ms: float = 0.0
    error: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    checked_at: str = ""

    def __post_init__(self):
        if not self.checked_at:
            self.checked_at = datetime.now().isoformat()


@dataclass
class AggregateHealth:
    """聚合健康状态"""
    overall: HealthStatus
    overall_score: float          # 0-100
    liveness: HealthStatus
    readiness: HealthStatus
    startup: HealthStatus
    dependencies: Dict[str, DependencyStatus]
    probes: List[HealthProbeResult]
    timestamp: str
    uptime_seconds: float = 0.0
    version: str = "unknown"
    suggestions: List[str] = field(default_factory=list)


class HealthChecker:
    """生产级健康检查器

    用法:
        checker = HealthChecker(name="trading-engine", version="2.0.0")

        # 注册依赖检查
        checker.register_dependency("okx_api", check_okx_api)
        checker.register_dependency("database", check_db)
        checker.register_dependency("redis", check_redis)

        # 运行检查
        health = checker.run_all()
        print(health.overall.value)  # "healthy" / "degraded" / "unhealthy"

        # 单探针检查
        liveness = checker.run_liveness()
        readiness = checker.run_readiness()
    """

    # 依赖权重（健康评分计算用）
    DEFAULT_DEPENDENCY_WEIGHTS = {
        "okx_api": 0.35,
        "database": 0.25,
        "redis": 0.20,
        "file_system": 0.10,
        "memory": 0.10,
    }

    # 健康评分阈值
    HEALTHY_THRESHOLD = 80.0
    DEGRADED_THRESHOLD = 50.0

    def __init__(self, name: str = "trading-system", version: str = "2.0.0",
                 max_history: int = 100):
        self.name = name
        self.version = version
        self._start_time = time.time()
        self._startup_complete = False
        self._lock = threading.Lock()

        # 依赖检查函数注册表
        self._dependency_checks: Dict[str, Callable[[], DependencyStatus]] = {}
        self._dependency_weights: Dict[str, float] = dict(self.DEFAULT_DEPENDENCY_WEIGHTS)

        # 自定义探针
        self._liveness_probes: List[Callable[[], HealthProbeResult]] = []
        self._readiness_probes: List[Callable[[], HealthProbeResult]] = []
        self._startup_probes: List[Callable[[], HealthProbeResult]] = []

        # 健康历史
        self._history: deque = deque(maxlen=max_history)

        # 状态变更回调
        self._on_status_change: Optional[Callable[[HealthStatus, HealthStatus], None]] = None

        # 上一次检查结果缓存
        self._last_result: Optional[AggregateHealth] = None
        self._last_check_time: float = 0.0
        self._cache_ttl: float = 5.0  # 缓存5秒

        logger.info(f"HealthChecker[{name}] initialized (version={version})")

    # ── 生命周期 ──

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time

    def mark_startup_complete(self):
        """标记启动完成"""
        self._startup_complete = True
        logger.info(f"HealthChecker[{self.name}]: startup complete (took {self.uptime_seconds:.1f}s)")

    @property
    def is_startup_complete(self) -> bool:
        return self._startup_complete

    def on_status_change(self, callback: Callable[[HealthStatus, HealthStatus], None]):
        """注册状态变更回调"""
        self._on_status_change = callback

    # ── 依赖注册 ──

    def register_dependency(self, name: str, check_fn: Callable[[], DependencyStatus],
                            weight: float = None):
        """注册依赖检查函数"""
        self._dependency_checks[name] = check_fn
        if weight is not None:
            self._dependency_weights[name] = weight
        logger.debug(f"HealthChecker: registered dependency '{name}' (weight={self._dependency_weights.get(name, 'default')})")

    def set_dependency_weight(self, name: str, weight: float):
        """设置依赖权重"""
        self._dependency_weights[name] = weight

    # ── 探针注册 ──

    def add_liveness_probe(self, probe: Callable[[], HealthProbeResult]):
        self._liveness_probes.append(probe)

    def add_readiness_probe(self, probe: Callable[[], HealthProbeResult]):
        self._readiness_probes.append(probe)

    def add_startup_probe(self, probe: Callable[[], HealthProbeResult]):
        self._startup_probes.append(probe)

    # ── 核心检查方法 ──

    def _check_single_dependency(self, name: str,
                                  check_fn: Callable[[], DependencyStatus]) -> DependencyStatus:
        """检查单个依赖（带计时和异常保护）"""
        t0 = time.time()
        try:
            result = check_fn()
            result.latency_ms = (time.time() - t0) * 1000
            result.checked_at = datetime.now().isoformat()
            return result
        except Exception as e:
            elapsed = (time.time() - t0) * 1000
            logger.warning(f"HealthChecker: dependency '{name}' check failed: {e}")
            return DependencyStatus(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=elapsed,
                error=str(e),
                checked_at=datetime.now().isoformat(),
            )

    def _run_probes(self, probes: List[Callable[[], HealthProbeResult]]) -> List[HealthProbeResult]:
        """运行一组探针（带异常保护）"""
        results = []
        for probe in probes:
            try:
                result = probe()
                results.append(result)
            except Exception as e:
                results.append(HealthProbeResult(
                    name="unknown",
                    status=HealthStatus.UNHEALTHY,
                    probe_type=ProbeType.LIVENESS,
                    message=f"Probe error: {e}",
                ))
        return results

    def run_liveness(self) -> HealthProbeResult:
        """
        存活探针: 检查进程是否存活。
        最轻量级，仅验证自身运行状态，不检查外部依赖。
        """
        t0 = time.time()
        try:
            # 基础存活检查：自身状态正常
            if self._liveness_probes:
                probe_results = self._run_probes(self._liveness_probes)
                failed = [r for r in probe_results if r.status == HealthStatus.UNHEALTHY]
                if failed:
                    return HealthProbeResult(
                        name="liveness",
                        status=HealthStatus.UNHEALTHY,
                        probe_type=ProbeType.LIVENESS,
                        latency_ms=(time.time() - t0) * 1000,
                        message=f"Liveness probes failed: {[f.name for f in failed]}",
                    )

            return HealthProbeResult(
                name="liveness",
                status=HealthStatus.HEALTHY,
                probe_type=ProbeType.LIVENESS,
                latency_ms=(time.time() - t0) * 1000,
                message="Process is alive",
                details={"uptime_seconds": self.uptime_seconds, "version": self.version},
            )
        except Exception as e:
            return HealthProbeResult(
                name="liveness",
                status=HealthStatus.UNHEALTHY,
                probe_type=ProbeType.LIVENESS,
                latency_ms=(time.time() - t0) * 1000,
                message=f"Liveness check failed: {e}",
            )

    def run_readiness(self) -> HealthProbeResult:
        """
        就绪探针: 检查是否准备好接收流量。
        验证所有关键依赖可用，依赖全部健康才算就绪。
        """
        t0 = time.time()
        suggestions = []

        try:
            if not self._startup_complete:
                return HealthProbeResult(
                    name="readiness",
                    status=HealthStatus.STARTING,
                    probe_type=ProbeType.READINESS,
                    latency_ms=(time.time() - t0) * 1000,
                    message="Startup not yet complete",
                )

            # 检查所有依赖
            unhealthy_deps = []
            for name, check_fn in self._dependency_checks.items():
                dep = self._check_single_dependency(name, check_fn)
                if dep.status == HealthStatus.UNHEALTHY:
                    unhealthy_deps.append(dep)
                    suggestions.append(f"Check dependency '{name}': {dep.error or 'unknown error'}")

            if unhealthy_deps:
                return HealthProbeResult(
                    name="readiness",
                    status=HealthStatus.UNHEALTHY,
                    probe_type=ProbeType.READINESS,
                    latency_ms=(time.time() - t0) * 1000,
                    message=f"Unhealthy dependencies: {[d.name for d in unhealthy_deps]}",
                    details={"unhealthy_dependencies": [d.name for d in unhealthy_deps]},
                    suggestions=suggestions,
                )

            # 运行自定义就绪探针
            if self._readiness_probes:
                probe_results = self._run_probes(self._readiness_probes)
                failed = [r for r in probe_results if r.status == HealthStatus.UNHEALTHY]
                if failed:
                    return HealthProbeResult(
                        name="readiness",
                        status=HealthStatus.UNHEALTHY,
                        probe_type=ProbeType.READINESS,
                        latency_ms=(time.time() - t0) * 1000,
                        message=f"Readiness probes failed: {[f.name for f in failed]}",
                    )

            return HealthProbeResult(
                name="readiness",
                status=HealthStatus.HEALTHY,
                probe_type=ProbeType.READINESS,
                latency_ms=(time.time() - t0) * 1000,
                message="Ready to serve traffic",
            )
        except Exception as e:
            return HealthProbeResult(
                name="readiness",
                status=HealthStatus.UNHEALTHY,
                probe_type=ProbeType.READINESS,
                latency_ms=(time.time() - t0) * 1000,
                message=f"Readiness check failed: {e}",
            )

    def run_startup(self) -> HealthProbeResult:
        """
        启动探针: 检查初始化是否完成。
        在启动阶段使用，完成后切换到就绪探针。
        """
        t0 = time.time()
        try:
            if self._startup_complete:
                return HealthProbeResult(
                    name="startup",
                    status=HealthStatus.HEALTHY,
                    probe_type=ProbeType.STARTUP,
                    latency_ms=(time.time() - t0) * 1000,
                    message="Startup complete",
                )

            if self._startup_probes:
                probe_results = self._run_probes(self._startup_probes)
                failed = [r for r in probe_results if r.status == HealthStatus.UNHEALTHY]
                if failed:
                    return HealthProbeResult(
                        name="startup",
                        status=HealthStatus.UNHEALTHY,
                        probe_type=ProbeType.STARTUP,
                        latency_ms=(time.time() - t0) * 1000,
                        message=f"Startup probes failed: {[f.name for f in failed]}",
                    )

            return HealthProbeResult(
                name="startup",
                status=HealthStatus.STARTING,
                probe_type=ProbeType.STARTUP,
                latency_ms=(time.time() - t0) * 1000,
                message="Still starting up...",
            )
        except Exception as e:
            return HealthProbeResult(
                name="startup",
                status=HealthStatus.UNHEALTHY,
                probe_type=ProbeType.STARTUP,
                latency_ms=(time.time() - t0) * 1000,
                message=f"Startup check failed: {e}",
            )

    def _compute_health_score(self, dependencies: Dict[str, DependencyStatus]) -> float:
        """计算加权健康评分 (0-100)"""
        # 无任何依赖注册时无法评估健康，返回中性分（不虚报满分 100），
        # 使 overall 落入 DEGRADED 区间，提示依赖监控未就绪。
        if not dependencies:
            return 50.0

        total_weight = 0.0
        weighted_score = 0.0

        for name, dep in dependencies.items():
            weight = self._dependency_weights.get(name, 0.1)
            total_weight += weight

            if dep.status == HealthStatus.HEALTHY:
                score = 100.0
            elif dep.status == HealthStatus.DEGRADED:
                score = 60.0
            elif dep.status == HealthStatus.UNKNOWN:
                score = 50.0
            else:
                score = 0.0

            weighted_score += score * weight

        if total_weight == 0:
            return 50.0

        return weighted_score / total_weight

    def _status_from_score(self, score: float) -> HealthStatus:
        if score >= self.HEALTHY_THRESHOLD:
            return HealthStatus.HEALTHY
        elif score >= self.DEGRADED_THRESHOLD:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.UNHEALTHY

    def run_all(self, force: bool = False) -> AggregateHealth:
        """
        运行全部健康检查（liveness + readiness + 依赖）。

        Args:
            force: 是否强制刷新（忽略缓存）

        Returns:
            AggregateHealth 聚合健康状态
        """
        # 缓存检查
        now = time.time()
        if not force and self._last_result and (now - self._last_check_time) < self._cache_ttl:
            return self._last_result

        with self._lock:
            t0 = time.time()
            suggestions = []

            # 1. Liveness
            liveness = self.run_liveness()

            # 2. Startup
            startup = self.run_startup()

            # 3. 依赖检查
            dependencies = {}
            for name, check_fn in self._dependency_checks.items():
                dep = self._check_single_dependency(name, check_fn)
                dependencies[name] = dep
                if dep.status == HealthStatus.UNHEALTHY:
                    suggestions.append(f"[{name}] {dep.error or 'unhealthy'}")
                if dep.suggestions:
                    suggestions.extend(dep.suggestions)

            # 4. Readiness
            readiness = self.run_readiness()

            # 5. 计算评分
            score = self._compute_health_score(dependencies)

            # 如果 liveness 失败，直接 UNHEALTHY
            if liveness.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
                score = 0.0
            elif readiness.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
            else:
                overall = self._status_from_score(score)

            # 收集探针结果
            all_probes = [liveness, readiness, startup]

            result = AggregateHealth(
                overall=overall,
                overall_score=round(score, 1),
                liveness=liveness.status,
                readiness=readiness.status,
                startup=startup.status,
                dependencies=dependencies,
                probes=all_probes,
                timestamp=datetime.now().isoformat(),
                uptime_seconds=self.uptime_seconds,
                version=self.version,
                suggestions=suggestions,
            )

            # 状态变更通知
            if self._last_result and self._last_result.overall != overall and self._on_status_change:
                try:
                    self._on_status_change(self._last_result.overall, overall)
                except Exception:
                    pass

            # 更新缓存和历史
            self._last_result = result
            self._last_check_time = now
            self._history.append(result)

            elapsed = (time.time() - t0) * 1000
            logger.debug(f"HealthChecker: all checks completed in {elapsed:.1f}ms, "
                        f"overall={overall.value}, score={score:.1f}")

            return result

    def get_last_result(self) -> Optional[AggregateHealth]:
        """获取最近一次检查结果"""
        return self._last_result

    def get_history(self, limit: int = 20) -> List[AggregateHealth]:
        """获取健康检查历史"""
        return list(self._history)[-limit:]

    def to_dict(self) -> Dict[str, Any]:
        """导出为字典（用于 API 响应）"""
        result = self._last_result
        if not result:
            result = self.run_all()

        return {
            "status": result.overall.value,
            "score": result.overall_score,
            "timestamp": result.timestamp,
            "uptime_seconds": result.uptime_seconds,
            "version": result.version,
            "probes": {
                "liveness": result.liveness.value,
                "readiness": result.readiness.value,
                "startup": result.startup.value,
            },
            "dependencies": {
                name: {
                    "status": dep.status.value,
                    "latency_ms": round(dep.latency_ms, 1),
                    "error": dep.error,
                    "checked_at": dep.checked_at,
                }
                for name, dep in result.dependencies.items()
            },
            "suggestions": result.suggestions,
        }

    def write_health_file(self, path: str = "./data/health_status.json"):
        """将健康状态写入文件（供外部监控使用）"""
        try:
            data = self.to_dict()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"HealthChecker: failed to write health file: {e}")


# ═══════════════════════════════════════════════════════════════
# 内置依赖检查工厂函数
# ═══════════════════════════════════════════════════════════════

def create_file_system_check(base_dir: str) -> Callable[[], DependencyStatus]:
    """创建文件系统健康检查（检查关键目录和文件权限）"""
    def check() -> DependencyStatus:
        t0 = time.time()
        try:
            checks = {
                "base_dir_exists": os.path.exists(base_dir),
                "base_dir_writable": os.access(base_dir, os.W_OK),
                "data_dir_exists": os.path.exists(os.path.join(base_dir, "data")),
                "logs_dir_exists": os.path.exists(os.path.join(base_dir, "logs")),
            }
            failed = [k for k, v in checks.items() if not v]
            if failed:
                return DependencyStatus(
                    name="file_system",
                    status=HealthStatus.DEGRADED if len(failed) <= 2 else HealthStatus.UNHEALTHY,
                    latency_ms=(time.time() - t0) * 1000,
                    error=f"Issues: {', '.join(failed)}",
                    details=checks,
                    suggestions=[f"Check path: {base_dir}"],
                )
            return DependencyStatus(
                name="file_system",
                status=HealthStatus.HEALTHY,
                latency_ms=(time.time() - t0) * 1000,
                details=checks,
            )
        except Exception as e:
            return DependencyStatus(
                name="file_system",
                status=HealthStatus.UNHEALTHY,
                error=str(e),
            )
    return check


def create_memory_check(threshold_mb: float = 500) -> Callable[[], DependencyStatus]:
    """创建内存使用检查"""
    def check() -> DependencyStatus:
        t0 = time.time()
        try:
            import psutil
            mem = psutil.virtual_memory()
            used_mb = mem.used / (1024 * 1024)
            available_mb = mem.available / (1024 * 1024)
            pct = mem.percent

            if pct > 90:
                status = HealthStatus.UNHEALTHY
                msg = f"Critical memory usage: {pct:.1f}%"
            elif pct > 75:
                status = HealthStatus.DEGRADED
                msg = f"High memory usage: {pct:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                msg = f"Memory OK: {pct:.1f}%"

            return DependencyStatus(
                name="memory",
                status=status,
                latency_ms=(time.time() - t0) * 1000,
                message=msg,
                details={
                    "used_mb": round(used_mb, 1),
                    "available_mb": round(available_mb, 1),
                    "percent": pct,
                },
            )
        except ImportError:
            return DependencyStatus(
                name="memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )
        except Exception as e:
            return DependencyStatus(
                name="memory",
                status=HealthStatus.UNHEALTHY,
                error=str(e),
            )
    return check


def create_okx_api_check(okx_client=None) -> Callable[[], DependencyStatus]:
    """创建 OKX API 连接检查"""
    def check() -> DependencyStatus:
        t0 = time.time()
        try:
            if okx_client is None:
                return DependencyStatus(
                    name="okx_api",
                    status=HealthStatus.UNKNOWN,
                    message="OKX client not configured",
                )
            # 简单状态检查
            if hasattr(okx_client, 'is_connected') and okx_client.is_connected():
                return DependencyStatus(
                    name="okx_api",
                    status=HealthStatus.HEALTHY,
                    latency_ms=(time.time() - t0) * 1000,
                    message="OKX API connected",
                )
            return DependencyStatus(
                name="okx_api",
                status=HealthStatus.DEGRADED,
                latency_ms=(time.time() - t0) * 1000,
                message="OKX API connection status unknown",
            )
        except Exception as e:
            return DependencyStatus(
                name="okx_api",
                status=HealthStatus.UNHEALTHY,
                error=str(e),
            )
    return check


# 全局健康检查器实例
_global_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """获取全局健康检查器"""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = HealthChecker()
    return _global_health_checker


def set_health_checker(checker: HealthChecker):
    """设置全局健康检查器"""
    global _global_health_checker
    _global_health_checker = checker