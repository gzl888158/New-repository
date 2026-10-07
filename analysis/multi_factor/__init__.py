"""
多因子选股模块 (Multi-Factor Stock Selection Module)

基于AKShare数据，实现A股多因子选股策略：
- 6大类因子：价值、质量、动量、规模、量价、事件
- 因子检验：IC分析、共线性检测、稳定性评估
- 因子评估：经济逻辑评估、分层回测、单调性检验、稳健性检验
- 因子优化：独立因子筛选、冗余消除、权重优化
- 因子打分：z-score/rank标准化，等权/IC加权/优化加权
- 组合构建：Top-N、得分加权、均值-方差优化、风险平价、行业中性
- 选股逻辑：Top-N或阈值过滤
"""

from .factor_engine import FactorEngine, FactorCategory
from .factor_analyzer import FactorAnalyzer
from .factor_evaluator import FactorEvaluator
from .factor_optimizer import FactorOptimizer
from .stock_selector import StockSelector
from .portfolio_builder import PortfolioBuilder

__all__ = [
    "FactorEngine",
    "FactorCategory",
    "FactorAnalyzer",
    "FactorEvaluator",
    "FactorOptimizer",
    "StockSelector",
    "PortfolioBuilder",
]
