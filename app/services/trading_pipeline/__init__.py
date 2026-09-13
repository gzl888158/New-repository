"""
交易流水线包，汇总编排器、信号处理链与各类信号过滤器。
"""
from .pipeline_orchestrator import PipelineOrchestrator, PipelineStage, PipelineStatus, PipelineContext
from .signal_processor_pipeline import (
    SignalProcessingPipeline,
    SignalValidator,
    SignalCooldownFilter,
    SignalQualityFilter,
    MarketRegimeFilter,
    RiskFilter,
    ConflictFilter,
)
from .decision_executor import DecisionExecutor
from .anomaly_detector import (
    AnomalyDetector,
    Anomaly,
    AnomalyType,
    AnomalySeverity,
    PriceSpikeDetector,
    VolumeSurgeDetector,
    OrderFailureRateDetector,
    LatencySpikeDetector,
    SignalFrequencyDetector,
    PNLDropDetector,
)
from .recovery_handler import (
    RecoveryHandler,
    RecoveryAction,
    RecoveryStatus,
    RecoveryTask,
)

__all__ = [
    "PipelineOrchestrator",
    "PipelineStage",
    "PipelineStatus",
    "PipelineContext",
    "SignalProcessingPipeline",
    "SignalValidator",
    "SignalCooldownFilter",
    "SignalQualityFilter",
    "MarketRegimeFilter",
    "RiskFilter",
    "ConflictFilter",
    "DecisionExecutor",
    "AnomalyDetector",
    "Anomaly",
    "AnomalyType",
    "AnomalySeverity",
    "PriceSpikeDetector",
    "VolumeSurgeDetector",
    "OrderFailureRateDetector",
    "LatencySpikeDetector",
    "SignalFrequencyDetector",
    "PNLDropDetector",
    "RecoveryHandler",
    "RecoveryAction",
    "RecoveryStatus",
    "RecoveryTask",
]
