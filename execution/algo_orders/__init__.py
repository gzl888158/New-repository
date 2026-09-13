"""
算法订单执行系统 (Algorithmic Order Execution System)

提供智能订单路由和高级算法执行能力：
  - SmartOrderRouter: 智能订单路由，多场所最佳执行
  - AlgoExecutionEngine: 算法执行引擎，管理算法订单生命周期
  - TWAPExecutor: 时间加权平均价格算法
  - VWAPExecutor: 成交量加权平均价格算法
  - IcebergOrderExecutor: 冰山订单算法
  - DarkPoolRouter: 暗池路由
  - ExecutionQualityMonitor: 执行质量监控
"""

from execution.algo_orders.smart_order_router import (
    SmartOrderRouter, VenueType, RouteDecision, VenueRanking,
    OrderSplittingPlan, ExecutionVenue, OrderUrgency, OrderSlice,
)
from execution.algo_orders.algo_execution_engine import (
    AlgoExecutionEngine, AlgoOrderState, AlgoOrderType, AlgoOrderConfig,
    AlgoOrderStatus, ExecutionSlice, AlgoExecutionResult,
)
from execution.algo_orders.twap_executor import (
    TWAPExecutor, TWAPConfig, TWAPState, TWAPSlice,
)
from execution.algo_orders.vwap_executor import (
    VWAPExecutor, VWAPConfig, VWAPState, VolumeProfile, VolumeBucket,
)
from execution.algo_orders.iceberg_executor import (
    IcebergOrderExecutor, IcebergConfig, IcebergState, IcebergSlice,
)
from execution.algo_orders.dark_pool_router import (
    DarkPoolRouter, DarkPoolVenue, DarkPoolOrder, DarkPoolFill,
)
from execution.algo_orders.execution_quality_monitor import (
    ExecutionQualityMonitor, ExecutionQualityReport, QualityMetrics,
    ImplementationShortfall, SlippageAnalysis, ExecutionCostBreakdown,
    EnhancedExecutionQualityMonitor,
    MarketImpactDecomposition, IntervalVWAPBenchmark,
    ParticipationRateAnalysis, PostTradeDrift, RealizedSpreadAnalysis,
)

__all__ = [
    # Smart Order Router
    "SmartOrderRouter", "VenueType", "RouteDecision", "VenueRanking",
    "OrderSplittingPlan", "ExecutionVenue", "OrderUrgency", "OrderSlice",
    # Algo Engine
    "AlgoExecutionEngine", "AlgoOrderState", "AlgoOrderType", "AlgoOrderConfig",
    "AlgoOrderStatus", "ExecutionSlice", "AlgoExecutionResult",
    # TWAP
    "TWAPExecutor", "TWAPConfig", "TWAPState", "TWAPSlice",
    # VWAP
    "VWAPExecutor", "VWAPConfig", "VWAPState", "VolumeProfile", "VolumeBucket",
    # Iceberg
    "IcebergOrderExecutor", "IcebergConfig", "IcebergState", "IcebergSlice",
    # Dark Pool
    "DarkPoolRouter", "DarkPoolVenue", "DarkPoolOrder", "DarkPoolFill",
    # Quality Monitor
    "ExecutionQualityMonitor", "ExecutionQualityReport", "QualityMetrics",
    "ImplementationShortfall", "SlippageAnalysis", "ExecutionCostBreakdown",
    "EnhancedExecutionQualityMonitor",
    "MarketImpactDecomposition", "IntervalVWAPBenchmark",
    "ParticipationRateAnalysis", "PostTradeDrift", "RealizedSpreadAnalysis",
]
