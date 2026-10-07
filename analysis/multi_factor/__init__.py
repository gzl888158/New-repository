"""
多因子选股模块 (Multi-Factor Stock Selection Module)

基于AKShare数据，实现A股多因子选股策略：
- 6大类因子：价值、质量、动量、规模、量价、事件
- 因子检验：IC分析、共线性检测、稳定性评估
- 因子打分：z-score/rank标准化，等权/IC加权/优化加权
- 选股逻辑：Top-N或阈值过滤
"""

from .factor_engine import FactorEngine, FactorCategory
from .factor_analyzer import FactorAnalyzer
from .stock_selector import StockSelector

__all__ = [
    "FactorEngine",
    "FactorCategory",
    "FactorAnalyzer",
    "StockSelector",
]
