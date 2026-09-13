"""
企业级分析架构 (Enterprise Analysis Architecture)

统一的分析层入口，整合以下能力：
  - DataAnalysisEngine: 多维数据分析引擎（策略绩效 / 交易统计 / 手续费分析）
  - ReportGenerator: 智能报表生成器（日报 / 周报 / 自定义报表）
  - HistoricalAnalyzer: 历史交易数据分析（短板识别 / 成功模式提取）
  - StrategyOptimizer: 策略参数优化器（参数热更新 / 配置版本管理 / 回滚）
  - ContributionAnalyzer: 策略贡献度分析引擎（多窗口贡献 / 健康度 / 生命周期）
  - ABTestingFramework: A/B 测试框架（多参数变体对比 / 统计显著性检验）
  - IntelligentAnalysisAgent: 企业级智能交易记录分析智能体（市场状态 / 策略方向 / 信号质量 / ADX 确认 / 分析与优化方向报告）

子模块：
  - parameter_optimization: 参数优化流水线（遗传 / 贝叶斯 / 前向行走 / 蒙特卡洛）
"""

from .data_analysis_engine import DataAnalysisEngine
from .report_generator import ReportGenerator
from .historical_analyzer import HistoricalAnalyzer
from .strategy_optimizer import StrategyOptimizer
from .contribution_analyzer import (
    ContributionAnalyzer,
    StrategyContribution,
    ContributionSnapshot,
    get_contribution_analyzer,
    reset_contribution_analyzer,
)
from .ab_testing import (
    ABTestingFramework,
    TestVariant,
    TestResult,
    StatisticalSignificance,
)
from .intelligent_analysis_agent import IntelligentAnalysisAgent

__all__ = [
    # 数据分析引擎
    "DataAnalysisEngine",
    # 报表生成器
    "ReportGenerator",
    # 历史分析器
    "HistoricalAnalyzer",
    # 策略优化器
    "StrategyOptimizer",
    # 贡献度分析
    "ContributionAnalyzer",
    "StrategyContribution",
    "ContributionSnapshot",
    "get_contribution_analyzer",
    "reset_contribution_analyzer",
    # A/B 测试
    "ABTestingFramework",
    "TestVariant",
    "TestResult",
    "StatisticalSignificance",
    # 智能分析智能体
    "IntelligentAnalysisAgent",
]
