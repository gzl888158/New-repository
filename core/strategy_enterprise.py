"""
企业级策略强化接入层 - Enterprise Strategy Reinforcement Layer

为所有策略提供统一的企业级能力，包括：
1. 容错健壮性：统一异常分级处理（替代散落的裸 except + logger.error）
2. 可观测性监控：指标埋点（接入 MetricsPipeline）
3. 参数校验：符号 / 方向 / 置信度 / 数量 / 价格前置校验（接入 validators 语义）
4. 告警通知：关键事件通知（接入 NotificationDispatcher）

设计原则：
- 惰性初始化：基础设施在首次使用时才初始化，避免启动开销
- 安全降级：基础设施缺失或初始化失败时自动降级为日志，绝不影响策略主逻辑
- 零侵入：策略通过继承 PersistentStrategy 自动获得能力，无需逐方法改造
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Dict, Optional

from loguru import logger


class EnterpriseStrategyMixin:
    """企业级策略强化能力（混入 PersistentStrategy，由全部策略继承）。"""

    # ─────────────────────────────────────────────────────────────
    # 数值防御
    # ─────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """安全 float 转换，抵御 None / NaN / Inf / 非法字符串。"""
        if value is None:
            return default
        try:
            v = float(value)
            if not math.isfinite(v):
                return default
            return v
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Any, default: int = 0) -> int:
        """安全 int 转换。"""
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            try:
                return int(float(value))
            except (TypeError, ValueError):
                return default

    # ─────────────────────────────────────────────────────────────
    # 配置获取（子类可重写以适配 config / _config 命名差异）
    # ─────────────────────────────────────────────────────────────
    def _get_config(self):
        """获取配置字典。策略层用 ``self.config``，业务逻辑层用 ``self._config``。"""
        cfg = getattr(self, "config", None)
        if cfg is None:
            cfg = getattr(self, "_config", None)
        return cfg

    # ─────────────────────────────────────────────────────────────
    # 可观测性：指标埋点
    # ─────────────────────────────────────────────────────────────
    def _get_metrics(self):
        """惰性获取全局 MetricsPipeline，失败返回 None。"""
        cached = getattr(self, "_enterprise_metrics", None)
        if cached is not None:
            return cached
        try:
            from core.metrics_pipeline import get_metrics_pipeline, MetricsPipeline
            config = self._get_config()
            pipeline = get_metrics_pipeline(config) if config else None
            if pipeline is None and config:
                pipeline = MetricsPipeline(config)
            self._enterprise_metrics = pipeline
            return pipeline
        except Exception as e:
            logger.debug(f"Enterprise metrics unavailable: {e}")
            self._enterprise_metrics = None
            return None

    def _record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        self._record_signal_filter_metric(name, labels)
        pipeline = self._get_metrics()
        if pipeline is None:
            return
        try:
            pipeline.record(name, value, labels)
        except Exception as e:
            logger.debug(f"Metric record failed ({name}): {e}")

    def _increment_metric(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        self._record_signal_filter_metric(name, labels)
        pipeline = self._get_metrics()
        if pipeline is None:
            return
        try:
            pipeline.increment(name, value, labels)
        except Exception as e:
            logger.debug(f"Metric increment failed ({name}): {e}")

    @staticmethod
    def _record_signal_filter_metric(name: str, labels: Optional[Dict[str, str]] = None):
        suffixes = (
            ("_filter_total", "strategy_filter"),
            ("_signal_rejected_total", "strategy_filter"),
            ("_gate_rejected_total", "strategy_gate"),
        )
        for suffix, layer in suffixes:
            if name.endswith(suffix):
                strategy = name[:-len(suffix)].removesuffix("_")
                reason = str(
                    (labels or {}).get("reason")
                    or (labels or {}).get("gate")
                    or name
                )
                try:
                    from core.signal_flow_stats import record_signal_flow_event
                    record_signal_flow_event(
                        "source_rejected", strategy=strategy,
                        layer=layer, reason=reason,
                    )
                except Exception as e:
                    logger.debug(f"Signal flow source rejection metric failed: {e}")
                return

    def _record_latency(self, name: str, latency_ms: float, labels: Optional[Dict[str, str]] = None):
        pipeline = self._get_metrics()
        if pipeline is None:
            return
        try:
            pipeline.record_latency(name, latency_ms, labels)
        except Exception as e:
            logger.debug(f"Latency record failed ({name}): {e}")

    # ─────────────────────────────────────────────────────────────
    # 容错健壮性：统一异常分级
    # ─────────────────────────────────────────────────────────────
    _SEVERITY_LEVEL = {
        "critical": "critical",
        "high": "error",
        "medium": "warning",
        "low": "info",
    }

    def _handle_exception(
        self,
        exc: Exception,
        context: Optional[Dict[str, Any]] = None,
        module: str = "",
        function: str = "",
        severity: str = "medium",
        category: str = "business",
        reraise: bool = False,
    ):
        """统一异常分级处理：分级日志 + 指标计数 + 可选重新抛出。

        用于替换策略中的裸 ``except Exception as e: logger.error(...)``，
        使异常可按严重程度 / 分类被统计与追踪。
        """
        exc_name = type(exc).__name__
        level = self._SEVERITY_LEVEL.get(severity, "warning")
        log_fn = getattr(logger, level, logger.warning)
        ctx = f" | Context: {context}" if context else ""
        log_fn(
            f"[{severity.upper()}] [{category.upper()}] "
            f"{module or type(self).__name__}.{function}: {exc_name}: {exc}{ctx}"
        )

        self._increment_metric(
            "strategy_exception_total",
            1.0,
            {
                "module": module or type(self).__name__,
                "function": function,
                "category": category,
                "severity": severity,
                "type": exc_name,
            },
        )

        if reraise or severity == "critical":
            raise exc

    # ─────────────────────────────────────────────────────────────
    # 参数校验（前置防御）
    # ─────────────────────────────────────────────────────────────
    def _validate_symbol(self, symbol: Any) -> bool:
        if not isinstance(symbol, str) or not symbol.strip():
            self._increment_metric("strategy_validation_rejected_total", 1.0, {"field": "symbol"})
            return False
        if not (symbol.endswith("-USDT") or symbol.endswith("-USDT-SWAP")):
            logger.warning(f"Invalid symbol format rejected: {symbol!r}")
            self._increment_metric("strategy_validation_rejected_total", 1.0, {"field": "symbol"})
            return False
        return True

    def _validate_direction(self, direction: Any) -> bool:
        if direction not in ("long", "short", "buy", "sell"):
            self._increment_metric("strategy_validation_rejected_total", 1.0, {"field": "direction"})
            return False
        return True

    def _validate_confidence(self, confidence: Any) -> bool:
        c = self._safe_float(confidence, default=-1.0)
        if c < 0.0 or c > 1.0:
            self._increment_metric("strategy_validation_rejected_total", 1.0, {"field": "confidence"})
            return False
        return True

    def _validate_quantity(self, quantity: Any, min_qty: float = 0.0) -> bool:
        q = self._safe_float(quantity, default=-1.0)
        if q <= min_qty:
            self._increment_metric("strategy_validation_rejected_total", 1.0, {"field": "quantity"})
            return False
        return True

    def _validate_price(self, price: Any) -> bool:
        p = self._safe_float(price, default=-1.0)
        if p <= 0:
            self._increment_metric("strategy_validation_rejected_total", 1.0, {"field": "price"})
            return False
        return True

    # ─────────────────────────────────────────────────────────────
    # 告警通知
    # ─────────────────────────────────────────────────────────────
    def _get_dispatcher(self):
        cached = getattr(self, "_enterprise_dispatcher", None)
        if cached is not None:
            return cached
        try:
            from core.notification_dispatcher import get_notification_dispatcher, NotificationDispatcher
            config = self._get_config()
            dispatcher = get_notification_dispatcher(config) if config else None
            if dispatcher is None and config:
                dispatcher = NotificationDispatcher(config)
            self._enterprise_dispatcher = dispatcher
            return dispatcher
        except Exception as e:
            logger.debug(f"Notification dispatcher unavailable: {e}")
            self._enterprise_dispatcher = None
            return None

    async def _notify(self, title: str, message: str, priority: str = "warning", category: str = "strategy"):
        """异步发送告警（安全降级：无 dispatcher 时仅日志）。"""
        dispatcher = self._get_dispatcher()
        if dispatcher is None:
            logger.warning(f"[NOTIFY:{priority.upper()}] {title}: {message}")
            return None
        try:
            from core.notification_dispatcher import NotificationPriority
            prio = getattr(NotificationPriority, priority.upper(), NotificationPriority.WARNING)
            return await dispatcher.send_to_all(
                title=title, message=message, priority=prio, category=category
            )
        except Exception as e:
            logger.debug(f"Notify failed: {e}")
            return None

    def _notify_sync(self, title: str, message: str, priority: str = "warning", category: str = "strategy"):
        """同步上下文中触发告警（尽力而为，不阻塞主流程）。"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            asyncio.create_task(self._notify(title, message, priority, category))
        else:
            logger.warning(f"[NOTIFY:{priority.upper()}] {title}: {message}")
