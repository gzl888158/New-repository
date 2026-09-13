"""
策略沙盒系统
============
提供隔离的策略测试环境，不依赖真实交易账户。

核心模块：
- virtual_account: 虚拟账户管理（资金、持仓、盈亏）
- virtual_executor: 虚拟订单执行引擎（滑点、费率、成交）
- sandbox_engine: 沙盒主引擎（整合行情、策略、执行）
- sandbox_manager: 多实例沙盒管理器（生命周期、对比分析）

快速使用：
    from sandbox import SandboxManager, SandboxConfig, get_sandbox_manager

    # 获取管理器
    manager = get_sandbox_manager(system_config)

    # 从模板创建沙盒
    sb_id = manager.create_from_template("aggressive_trend")

    # 启动沙盒
    manager.start(sb_id)

    # 注入行情
    manager.broadcast_bar("BTC-USDT-SWAP", bar_data)

    # 查看绩效
    perf = manager.get_engine(sb_id).get_performance_summary()
"""

from .virtual_account import (
    VirtualAccount, VirtualPosition, VirtualAccountSnapshot,
    create_virtual_account, get_virtual_account,
    remove_virtual_account, list_virtual_accounts,
)

from .virtual_executor import (
    VirtualOrderExecutor, VirtualOrder, FillResult,
    SlippageModel,
    OrderType, OrderSide, OrderStatus,
)

from .sandbox_engine import (
    SandboxEngine, SandboxConfig, SandboxState, SandboxTrade,
)

from .sandbox_manager import (
    SandboxManager, SandboxInstance, SandboxTemplate,
    SandboxComparison, SandboxRunMode, SandboxRankMetric,
    PRESET_TEMPLATES,
    get_sandbox_manager, reset_sandbox_manager,
)

__all__ = [
    # 虚拟账户
    "VirtualAccount", "VirtualPosition", "VirtualAccountSnapshot",
    "create_virtual_account", "get_virtual_account",
    "remove_virtual_account", "list_virtual_accounts",
    # 虚拟执行
    "VirtualOrderExecutor", "VirtualOrder", "FillResult",
    "SlippageModel", "OrderType", "OrderSide", "OrderStatus",
    # 沙盒引擎
    "SandboxEngine", "SandboxConfig", "SandboxState", "SandboxTrade",
    # 沙盒管理器
    "SandboxManager", "SandboxInstance", "SandboxTemplate",
    "SandboxComparison", "SandboxRunMode", "SandboxRankMetric",
    "PRESET_TEMPLATES",
    "get_sandbox_manager", "reset_sandbox_manager",
]
