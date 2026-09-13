"""自适应学习包：导出在线学习、参数自适应、策略进化与市场状态检测等组件。"""
from .online_learner import (
    OnlineLearner, LearningMode, LearningStatus,
    DriftSeverity, ModelType, LRSchedule,
    SampleBuffer, ADWIN, DriftDetector,
    IncrementalModel, OnlineCalibrator,
)
from .parameter_adaptor import ParameterAdaptor
from .strategy_evolver import (
    StrategyEvolver, EvolutionStrategy, EvolutionStatus,
    MultiPopulationEvolver, MigrationTopology,
)
from .market_regime_detector import (
    MarketRegimeDetector, MarketRegime,
    HMMRegimeClassifier, MultiTimeframeRegime,
    RegimeFeatureExtractor, RegimeTransitionPredictor,
    RegimeStrategyMapper,
)
from .performance_feedback import (
    PerformanceFeedback,
    PerformanceMetrics,
    PerformanceScorer,
    AttributionAnalyzer,
    FeedbackController,
    StrategyComparator,
    PerformanceAlerter,
    AlertSeverity,
    ScoreDimension,
    compute_metrics,
)
from .knowledge_base import (
    KnowledgeBase, Experience, ExperienceType,
)
from .meta_learner import (
    MetaLearner, LearningTask, TaskSimilarity,
    KnowledgeTransfer, FastAdaptor, StrategyInitializer,
    LearningStrategyOptimizer,
)

__all__ = [
    # OnlineLearner
    "OnlineLearner",
    "LearningMode",
    "LearningStatus",
    "DriftSeverity",
    "ModelType",
    "LRSchedule",
    "SampleBuffer",
    "ADWIN",
    "DriftDetector",
    "IncrementalModel",
    "OnlineCalibrator",
    # ParameterAdaptor
    "ParameterAdaptor",
    # StrategyEvolver
    "StrategyEvolver",
    "EvolutionStrategy",
    "EvolutionStatus",
    "MultiPopulationEvolver",
    "MigrationTopology",
    # MarketRegimeDetector
    "MarketRegimeDetector",
    "MarketRegime",
    "HMMRegimeClassifier",
    "MultiTimeframeRegime",
    "RegimeFeatureExtractor",
    "RegimeTransitionPredictor",
    "RegimeStrategyMapper",
    # PerformanceFeedback
    "PerformanceFeedback",
    "PerformanceMetrics",
    "PerformanceScorer",
    "AttributionAnalyzer",
    "FeedbackController",
    "StrategyComparator",
    "PerformanceAlerter",
    "AlertSeverity",
    "ScoreDimension",
    "compute_metrics",
    # KnowledgeBase
    "KnowledgeBase",
    "Experience",
    "ExperienceType",
    # MetaLearner
    "MetaLearner",
    "LearningTask",
    "TaskSimilarity",
    "KnowledgeTransfer",
    "FastAdaptor",
    "StrategyInitializer",
    "LearningStrategyOptimizer",
]
