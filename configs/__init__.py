"""
配置模块入口：导出配置加载与币种层级、交易对配置查询接口。
"""
from .settings import (
    load_config,
    get_currency_tier,
    get_symbol_config,
    get_symbol_leverage,
    get_symbol_leverage_range,
    get_symbol_grid_spacing,
    get_all_symbols,
)

__all__ = [
    "load_config",
    "get_currency_tier",
    "get_symbol_config",
    "get_symbol_leverage",
    "get_symbol_leverage_range",
    "get_symbol_grid_spacing",
    "get_all_symbols",
]
