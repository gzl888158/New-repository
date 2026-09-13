"""
参数优化系统 (Parameter Optimization System)

提供统一的交易策略参数优化流水线，整合：
  - GeneticOptimizer: 遗传算法全局搜索
  - BayesianOptimizer: 贝叶斯优化局部精调
  - WalkForwardAnalyzer: 前向行走稳健性分析
  - MonteCarloValidator: 蒙特卡洛验证评估
  - ParameterOptimizationOrchestrator: 统一编排器
"""

from analysis.parameter_optimization.genetic_optimizer import (
    GeneticOptimizer, ParameterDef, Individual, GAOptimizationResult,
    GenerationStats, SelectionMethod, CrossoverMethod, ConvergenceReason,
)
from analysis.parameter_optimization.bayesian_optimizer import (
    BayesianOptimizer, BOOptimizationResult, Observation,
    AcquisitionFunction, KernelType, GaussianProcess,
)
from analysis.parameter_optimization.walk_forward_analyzer import (
    WalkForwardAnalyzer, WalkForwardResult, WindowResult, WindowMode,
)
from analysis.parameter_optimization.monte_carlo_validator import (
    MonteCarloValidator, MCValidationResult, PerturbationSample,
    BootstrapSample, ValidationMethod,
)
from analysis.parameter_optimization.orchestrator import (
    ParameterOptimizationOrchestrator, OptimizationPipelineResult,
    OptimizationPhase, OptimizationStrategy,
)

__all__ = [
    # GA
    "GeneticOptimizer", "ParameterDef", "Individual", "GAOptimizationResult",
    "GenerationStats", "SelectionMethod", "CrossoverMethod", "ConvergenceReason",
    # BO
    "BayesianOptimizer", "BOOptimizationResult", "Observation",
    "AcquisitionFunction", "KernelType", "GaussianProcess",
    # WF
    "WalkForwardAnalyzer", "WalkForwardResult", "WindowResult", "WindowMode",
    # MC
    "MonteCarloValidator", "MCValidationResult", "PerturbationSample",
    "BootstrapSample", "ValidationMethod",
    # Orchestrator
    "ParameterOptimizationOrchestrator", "OptimizationPipelineResult",
    "OptimizationPhase", "OptimizationStrategy",
]
