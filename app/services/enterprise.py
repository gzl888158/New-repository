"""
企业级业务逻辑层强化基类 - Enterprise Service Layer Mixin

为 ``app/services`` 下的业务逻辑组件（交易流水线 trading_pipeline、自适应学习
adaptive_learning）提供统一的企业级能力，复用
``core.strategy_enterprise.EnterpriseStrategyMixin`` 的：

1. 容错健壮性：统一异常分级处理（_handle_exception）
2. 可观测性：指标埋点（_record_metric / _increment_metric / _record_latency）
3. 参数校验：符号 / 方向 / 置信度 / 数量 / 价格前置校验（_validate_*）
4. 告警通知：关键事件通知（_notify / _notify_sync）

并在此基础上补充业务逻辑层特有的校验：
- 信号数据校验 _validate_signal_data
- 决策数据校验 _validate_decision_data
- 订单数据校验 _validate_order_data
- 配置对象校验 _validate_config

设计原则（与策略层一致）：
- 惰性初始化：基础设施首次使用时才初始化，避免启动开销
- 安全降级：基础设施缺失或初始化失败时自动降级为日志，绝不影响主逻辑
- 零侵入：业务门面类通过继承 EnterpriseServiceMixin 自动获得能力，无需逐方法改造
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.strategy_enterprise import EnterpriseStrategyMixin


class EnterpriseServiceMixin(EnterpriseStrategyMixin):
    """企业级业务逻辑层能力混入（供 app/services 下的门面类继承）。

    业务逻辑层普遍使用 ``self._config`` 保存配置，而策略层使用 ``self.config``；
    父类的 ``_get_config()`` 已同时兼容两种命名，业务类无需额外适配。
    """

    # ─────────────────────────────────────────────────────────────
    # 业务层参数校验（前置防御）
    # ─────────────────────────────────────────────────────────────
    def _validate_signal_data(self, signal_data: Any) -> bool:
        """校验信号数据是否为非空 dict。"""
        if not isinstance(signal_data, dict) or not signal_data:
            self._increment_metric("service_validation_rejected_total", 1.0, {"field": "signal_data"})
            return False
        return True

    def _validate_decision_data(self, decision_data: Any) -> bool:
        """校验决策数据是否为非空 dict。"""
        if not isinstance(decision_data, dict) or not decision_data:
            self._increment_metric("service_validation_rejected_total", 1.0, {"field": "decision_data"})
            return False
        return True

    def _validate_order_data(self, order_data: Any) -> bool:
        """校验订单数据是否为非空 dict。"""
        if not isinstance(order_data, dict) or not order_data:
            self._increment_metric("service_validation_rejected_total", 1.0, {"field": "order_data"})
            return False
        return True

    def _validate_config(self) -> bool:
        """校验当前实例是否持有有效配置（dict）。"""
        cfg = self._get_config()
        if not isinstance(cfg, dict):
            self._increment_metric("service_validation_rejected_total", 1.0, {"field": "config"})
            return False
        return True
