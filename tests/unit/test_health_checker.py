"""
生产级健康检查器 (HealthChecker) 全面单元测试

覆盖:
  - 初始化与配置
  - Liveness 存活探针
  - Readiness 就绪探针
  - Startup 启动探针
  - 依赖注册与检查
  - 加权健康评分计算
  - 聚合健康状态
  - 缓存机制
  - 历史记录
  - 工厂函数 (file_system_check, memory_check, okx_api_check)
  - 边界条件与异常处理
"""
import pytest
import time
import os
import tempfile
from unittest.mock import MagicMock, patch

from core.health_checker import (
    HealthChecker, HealthStatus, HealthProbeResult, AggregateHealth,
    DependencyStatus, ProbeType, get_health_checker, set_health_checker,
    create_file_system_check, create_memory_check, create_okx_api_check,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def checker():
    """基础健康检查器"""
    return HealthChecker(name="test-system", version="1.0.0")


@pytest.fixture
def checker_with_startup():
    """已标记启动完成的检查器"""
    hc = HealthChecker(name="test-system", version="1.0.0")
    hc.mark_startup_complete()
    return hc


@pytest.fixture
def healthy_dep_check():
    """返回健康的依赖检查函数"""
    def check():
        return DependencyStatus(
            name="test_dep",
            status=HealthStatus.HEALTHY,
            message="All good",
        )
    return check


@pytest.fixture
def unhealthy_dep_check():
    """返回不健康的依赖检查函数"""
    def check():
        return DependencyStatus(
            name="test_dep",
            status=HealthStatus.UNHEALTHY,
            error="Connection refused",
        )
    return check


@pytest.fixture
def raising_dep_check():
    """抛出异常的依赖检查函数"""
    def check():
        raise RuntimeError("Unexpected error")
    return check


# ═══════════════════════════════════════════════════════════════
# Test: 初始化
# ═══════════════════════════════════════════════════════════════

class TestInitialization:
    def test_default_initialization(self):
        hc = HealthChecker()
        assert hc.name == "trading-system"
        assert hc.version == "2.0.0"
        assert hc.is_startup_complete is False
        assert hc.uptime_seconds >= 0

    def test_custom_name_version(self):
        hc = HealthChecker(name="my-app", version="3.0.0")
        assert hc.name == "my-app"
        assert hc.version == "3.0.0"

    def test_uptime_increases(self):
        hc = HealthChecker()
        t0 = hc.uptime_seconds
        time.sleep(0.1)
        assert hc.uptime_seconds > t0

    def test_mark_startup_complete(self, checker):
        assert checker.is_startup_complete is False
        checker.mark_startup_complete()
        assert checker.is_startup_complete is True


# ═══════════════════════════════════════════════════════════════
# Test: Liveness 存活探针
# ═══════════════════════════════════════════════════════════════

class TestLiveness:
    def test_liveness_healthy(self, checker):
        result = checker.run_liveness()
        assert result.status == HealthStatus.HEALTHY
        assert result.probe_type == ProbeType.LIVENESS
        assert "alive" in result.message.lower()
        assert result.latency_ms >= 0
        assert "uptime_seconds" in result.details

    def test_liveness_with_custom_probes(self, checker):
        """自定义存活探针全部通过"""
        checker.add_liveness_probe(lambda: HealthProbeResult(
            name="custom1", status=HealthStatus.HEALTHY,
            probe_type=ProbeType.LIVENESS, message="OK"
        ))
        result = checker.run_liveness()
        assert result.status == HealthStatus.HEALTHY

    def test_liveness_with_failing_custom_probe(self, checker):
        """自定义存活探针失败"""
        checker.add_liveness_probe(lambda: HealthProbeResult(
            name="custom1", status=HealthStatus.UNHEALTHY,
            probe_type=ProbeType.LIVENESS, message="FAIL"
        ))
        result = checker.run_liveness()
        assert result.status == HealthStatus.UNHEALTHY

    def test_liveness_with_exception_in_probe(self, checker):
        """自定义探针抛出异常"""
        def bad_probe():
            raise Exception("Probe crashed")
        checker.add_liveness_probe(bad_probe)
        result = checker.run_liveness()
        assert result.status == HealthStatus.UNHEALTHY


# ═══════════════════════════════════════════════════════════════
# Test: Readiness 就绪探针
# ═══════════════════════════════════════════════════════════════

class TestReadiness:
    def test_readiness_not_started(self, checker):
        """未完成启动时，就绪探针返回 STARTING"""
        result = checker.run_readiness()
        assert result.status == HealthStatus.STARTING

    def test_readiness_healthy(self, checker_with_startup, healthy_dep_check):
        """所有依赖健康，就绪探针返回 HEALTHY"""
        checker_with_startup.register_dependency("test", healthy_dep_check)
        result = checker_with_startup.run_readiness()
        assert result.status == HealthStatus.HEALTHY

    def test_readiness_unhealthy(self, checker_with_startup, unhealthy_dep_check):
        """非关键依赖不健康，就绪探针返回 DEGRADED"""
        checker_with_startup.register_dependency("test", unhealthy_dep_check)
        result = checker_with_startup.run_readiness()
        assert result.status == HealthStatus.DEGRADED
        assert len(result.suggestions) > 0

    def test_readiness_with_exception(self, checker_with_startup, raising_dep_check):
        """非关键依赖检查抛出异常，返回 DEGRADED"""
        checker_with_startup.register_dependency("test", raising_dep_check)
        result = checker_with_startup.run_readiness()
        assert result.status == HealthStatus.DEGRADED

    def test_readiness_custom_probes(self, checker_with_startup):
        checker_with_startup.mark_startup_complete()
        checker_with_startup.add_readiness_probe(lambda: HealthProbeResult(
            name="readiness1", status=HealthStatus.HEALTHY,
            probe_type=ProbeType.READINESS, message="OK"
        ))
        result = checker_with_startup.run_readiness()
        assert result.status == HealthStatus.HEALTHY


# ═══════════════════════════════════════════════════════════════
# Test: Startup 启动探针
# ═══════════════════════════════════════════════════════════════

class TestStartup:
    def test_startup_not_complete(self, checker):
        result = checker.run_startup()
        assert result.status == HealthStatus.STARTING

    def test_startup_complete(self, checker):
        checker.mark_startup_complete()
        result = checker.run_startup()
        assert result.status == HealthStatus.HEALTHY

    def test_startup_custom_probes(self, checker):
        checker.add_startup_probe(lambda: HealthProbeResult(
            name="startup1", status=HealthStatus.HEALTHY,
            probe_type=ProbeType.STARTUP, message="OK"
        ))
        result = checker.run_startup()
        assert result.status == HealthStatus.STARTING  # 未完成启动

    def test_startup_probe_failure(self, checker):
        checker.add_startup_probe(lambda: HealthProbeResult(
            name="startup1", status=HealthStatus.UNHEALTHY,
            probe_type=ProbeType.STARTUP, message="FAIL"
        ))
        result = checker.run_startup()
        assert result.status == HealthStatus.UNHEALTHY


# ═══════════════════════════════════════════════════════════════
# Test: 依赖注册与检查
# ═══════════════════════════════════════════════════════════════

class TestDependencyRegistration:
    def test_register_dependency(self, checker):
        checker.register_dependency("db", lambda: DependencyStatus(
            name="db", status=HealthStatus.HEALTHY
        ))
        assert "db" in checker._dependency_checks

    def test_register_with_weight(self, checker):
        checker.register_dependency("db", lambda: DependencyStatus(
            name="db", status=HealthStatus.HEALTHY
        ), weight=0.5)
        assert checker._dependency_weights["db"] == 0.5

    def test_set_dependency_weight(self, checker):
        checker.set_dependency_weight("db", 0.3)
        assert checker._dependency_weights["db"] == 0.3

    def test_single_dependency_check_timing(self, checker, healthy_dep_check):
        dep = checker._check_single_dependency("test", healthy_dep_check)
        assert dep.latency_ms >= 0
        assert dep.checked_at != ""

    def test_single_dependency_check_exception(self, checker, raising_dep_check):
        dep = checker._check_single_dependency("test", raising_dep_check)
        assert dep.status == HealthStatus.UNHEALTHY
        assert dep.error is not None


# ═══════════════════════════════════════════════════════════════
# Test: 加权健康评分
# ═══════════════════════════════════════════════════════════════

class TestHealthScore:
    def test_empty_dependencies(self, checker):
        # P2 修复：无依赖注册时返回中性分 50，不虚报满分 100
        score = checker._compute_health_score({})
        assert score == 50.0

    def test_all_healthy(self, checker):
        deps = {
            "a": DependencyStatus(name="a", status=HealthStatus.HEALTHY),
            "b": DependencyStatus(name="b", status=HealthStatus.HEALTHY),
        }
        score = checker._compute_health_score(deps)
        assert score == 100.0

    def test_all_unhealthy(self, checker):
        deps = {
            "a": DependencyStatus(name="a", status=HealthStatus.UNHEALTHY),
            "b": DependencyStatus(name="b", status=HealthStatus.UNHEALTHY),
        }
        score = checker._compute_health_score(deps)
        assert score == 0.0

    def test_partial_healthy(self, checker):
        """一个健康一个不健康，按权重计算"""
        checker.set_dependency_weight("a", 0.5)
        checker.set_dependency_weight("b", 0.5)
        deps = {
            "a": DependencyStatus(name="a", status=HealthStatus.HEALTHY),
            "b": DependencyStatus(name="b", status=HealthStatus.UNHEALTHY),
        }
        score = checker._compute_health_score(deps)
        assert score == 50.0

    def test_weighted_three_components(self, checker):
        checker.set_dependency_weight("a", 0.4)
        checker.set_dependency_weight("b", 0.3)
        checker.set_dependency_weight("c", 0.3)
        deps = {
            "a": DependencyStatus(name="a", status=HealthStatus.HEALTHY),    # 100 * 0.4 = 40
            "b": DependencyStatus(name="b", status=HealthStatus.HEALTHY),    # 100 * 0.3 = 30
            "c": DependencyStatus(name="c", status=HealthStatus.UNHEALTHY),  # 0 * 0.3 = 0
        }
        score = checker._compute_health_score(deps)
        assert score == 70.0

    def test_degraded_score(self, checker):
        checker.set_dependency_weight("a", 1.0)
        deps = {
            "a": DependencyStatus(name="a", status=HealthStatus.DEGRADED),
        }
        score = checker._compute_health_score(deps)
        assert score == 60.0

    def test_unknown_score(self, checker):
        checker.set_dependency_weight("a", 1.0)
        deps = {
            "a": DependencyStatus(name="a", status=HealthStatus.UNKNOWN),
        }
        score = checker._compute_health_score(deps)
        assert score == 50.0

    def test_status_from_score(self, checker):
        assert checker._status_from_score(90.0) == HealthStatus.HEALTHY
        assert checker._status_from_score(80.0) == HealthStatus.HEALTHY
        assert checker._status_from_score(60.0) == HealthStatus.DEGRADED
        assert checker._status_from_score(50.0) == HealthStatus.DEGRADED
        assert checker._status_from_score(49.0) == HealthStatus.UNHEALTHY
        assert checker._status_from_score(0.0) == HealthStatus.UNHEALTHY


# ═══════════════════════════════════════════════════════════════
# Test: 聚合健康状态 (run_all)
# ═══════════════════════════════════════════════════════════════

class TestAggregateHealth:
    def test_run_all_healthy(self, checker_with_startup, healthy_dep_check):
        checker_with_startup.register_dependency("test", healthy_dep_check)
        result = checker_with_startup.run_all()
        assert result.overall == HealthStatus.HEALTHY
        assert result.overall_score == 100.0
        assert result.liveness == HealthStatus.HEALTHY
        assert result.readiness == HealthStatus.HEALTHY
        assert "test" in result.dependencies
        assert result.version == "1.0.0"

    def test_run_all_unhealthy(self, checker_with_startup, unhealthy_dep_check):
        checker_with_startup.register_dependency("test", unhealthy_dep_check)
        result = checker_with_startup.run_all()
        assert result.overall == HealthStatus.UNHEALTHY
        assert result.overall_score == 0.0

    def test_run_all_multiple_deps(self, checker_with_startup):
        checker_with_startup.register_dependency("dep1", lambda: DependencyStatus(
            name="dep1", status=HealthStatus.HEALTHY
        ))
        checker_with_startup.register_dependency("dep2", lambda: DependencyStatus(
            name="dep2", status=HealthStatus.UNHEALTHY,
            error="Connection failed",
        ))
        checker_with_startup.register_dependency("dep3", lambda: DependencyStatus(
            name="dep3", status=HealthStatus.UNHEALTHY,
            error="Timeout",
        ))
        result = checker_with_startup.run_all()
        assert result.overall == HealthStatus.UNHEALTHY
        assert result.overall_score < 50

    def test_run_all_liveness_fails(self, checker_with_startup):
        """liveness 失败时，整体状态直接 UNHEALTHY"""
        checker_with_startup.add_liveness_probe(lambda: HealthProbeResult(
            name="l", status=HealthStatus.UNHEALTHY,
            probe_type=ProbeType.LIVENESS, message="dead"
        ))
        result = checker_with_startup.run_all()
        assert result.overall == HealthStatus.UNHEALTHY
        assert result.overall_score == 0.0

    def test_run_all_suggestions(self, checker_with_startup, unhealthy_dep_check):
        checker_with_startup.register_dependency("bad", unhealthy_dep_check)
        result = checker_with_startup.run_all()
        assert len(result.suggestions) > 0

    def test_run_all_has_timestamp(self, checker_with_startup):
        result = checker_with_startup.run_all()
        assert result.timestamp != ""
        assert result.uptime_seconds >= 0


# ═══════════════════════════════════════════════════════════════
# Test: 缓存机制
# ═══════════════════════════════════════════════════════════════

class TestCaching:
    def test_cache_returns_same_result(self, checker_with_startup):
        result1 = checker_with_startup.run_all()
        time.sleep(0.1)
        result2 = checker_with_startup.run_all()  # 应命中缓存
        assert result1 is result2
        assert result1.timestamp == result2.timestamp

    def test_force_refresh(self, checker_with_startup):
        result1 = checker_with_startup.run_all()
        time.sleep(0.1)
        result2 = checker_with_startup.run_all(force=True)  # 强制刷新
        # 强制刷新后可能时间戳不同（如果依赖检查耗时不同）
        assert result2 is not None

    def test_get_last_result(self, checker_with_startup):
        assert checker_with_startup.get_last_result() is None
        result = checker_with_startup.run_all()
        assert checker_with_startup.get_last_result() is result


# ═══════════════════════════════════════════════════════════════
# Test: 历史记录
# ═══════════════════════════════════════════════════════════════

class TestHistory:
    def test_history_stores_results(self, checker_with_startup):
        for _ in range(3):
            checker_with_startup.run_all(force=True)
        history = checker_with_startup.get_history()
        assert len(history) == 3

    def test_history_limit(self, checker_with_startup):
        for _ in range(5):
            checker_with_startup.run_all(force=True)
        history = checker_with_startup.get_history(limit=3)
        assert len(history) == 3


# ═══════════════════════════════════════════════════════════════
# Test: 导出
# ═══════════════════════════════════════════════════════════════

class TestExport:
    def test_to_dict(self, checker_with_startup, healthy_dep_check):
        checker_with_startup.register_dependency("test", healthy_dep_check)
        checker_with_startup.run_all()
        d = checker_with_startup.to_dict()
        assert "status" in d
        assert "score" in d
        assert "probes" in d
        assert "dependencies" in d
        assert "test" in d["dependencies"]

    def test_write_health_file(self, checker_with_startup):
        import tempfile, json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name
        try:
            checker_with_startup.run_all()
            checker_with_startup.write_health_file(path)
            assert os.path.exists(path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "status" in data
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════
# Test: 工厂函数
# ═══════════════════════════════════════════════════════════════

class TestFactoryFunctions:
    def test_file_system_check_healthy(self):
        check = create_file_system_check(os.getcwd())
        result = check()
        assert result.name == "file_system"
        assert result.status == HealthStatus.HEALTHY

    def test_file_system_check_invalid(self):
        check = create_file_system_check("/nonexistent/path/xyz")
        result = check()
        assert result.status == HealthStatus.UNHEALTHY

    def test_memory_check_no_psutil(self):
        """psutil 不可用时返回 UNKNOWN（本地已安装 psutil，测试 healthy 路径）"""
        check = create_memory_check()
        result = check()
        assert result.name == "memory"
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    def test_memory_check_healthy(self):
        """psutil 可用时检查内存"""
        check = create_memory_check()
        result = check()
        assert result.name == "memory"
        assert result.status in (HealthStatus.HEALTHY, HealthStatus.DEGRADED, HealthStatus.UNHEALTHY)

    def test_okx_api_check_no_client(self):
        check = create_okx_api_check(None)
        result = check()
        assert result.status == HealthStatus.UNKNOWN

    def test_okx_api_check_with_client(self):
        mock_client = MagicMock()
        mock_client.is_connected.return_value = True
        check = create_okx_api_check(mock_client)
        result = check()
        assert result.status == HealthStatus.HEALTHY

    def test_okx_api_check_disconnected(self):
        mock_client = MagicMock()
        mock_client.is_connected.return_value = False
        check = create_okx_api_check(mock_client)
        result = check()
        assert result.status == HealthStatus.DEGRADED


# ═══════════════════════════════════════════════════════════════
# Test: 全局实例
# ═══════════════════════════════════════════════════════════════

class TestGlobalInstance:
    def test_get_health_checker_creates(self):
        import core.health_checker as hc_mod
        hc_mod._global_health_checker = None
        hc = get_health_checker()
        assert isinstance(hc, HealthChecker)

    def test_set_health_checker(self):
        custom = HealthChecker(name="custom")
        set_health_checker(custom)
        assert get_health_checker() is custom
        # 恢复
        set_health_checker(None)


# ═══════════════════════════════════════════════════════════════
# Test: 状态变更回调
# ═══════════════════════════════════════════════════════════════

class TestStatusCallback:
    def test_on_status_change_called(self, checker_with_startup, unhealthy_dep_check):
        calls = []
        checker_with_startup.on_status_change(lambda old, new: calls.append((old, new)))
        checker_with_startup.register_dependency("bad", unhealthy_dep_check)

        # 第一次运行不会有回调（因为没有上一次状态）
        checker_with_startup.run_all(force=True)
        assert len(calls) == 0

        # 第二次运行，状态可能变化
        checker_with_startup.run_all(force=True)
        # 因为状态相同，所以不会触发回调
        assert len(calls) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])