"""
symbol_overrides 逐币种精细参数单元测试
==========================================
覆盖：CurrenciesConfig 保留 symbol_overrides（不再被 Pydantic 丢弃）、
get_symbol_config 合并 tier + override、get_symbol_leverage / _range / grid_spacing 辅助函数。
"""
from configs.settings import (
    CurrenciesConfig,
    get_symbol_config,
    get_symbol_leverage,
    get_symbol_leverage_range,
    get_symbol_grid_spacing,
    get_currency_tier,
)


def _tier(lev_min=3, lev_max=5, lev_def=5, spacing_min=0.0114, spacing_max=0.018, slippage=0.0015):
    return {
        "leverage_min": lev_min,
        "leverage_max": lev_max,
        "leverage_default": lev_def,
        "position_limit": 0.25,
        "grid_spacing_min": spacing_min,
        "grid_spacing_max": spacing_max,
        "slippage": slippage,
    }


def _currencies():
    return {
        "tier1_symbols": ["ETH"],
        "tier1_settings": _tier(2, 5, 3, 0.0114, 0.012, 0.001),
        "tier2_symbols": ["SOL", "XRP", "DOGE", "ARB"],
        "tier2_settings": _tier(3, 5, 5, 0.0114, 0.018, 0.0015),
        "tier3_symbols": ["SUI"],
        "tier3_settings": _tier(2, 5, 3, 0.0114, 0.025, 0.002),
        "symbol_overrides": {
            "ARB": {
                "leverage_min": 3,
                "leverage_max": 5,
                "leverage_default": 4,
                "position_limit": 0.20,
                "grid_spacing_min": 0.009,
                "grid_spacing_max": 0.016,
                "slippage": 0.0015,
            }
        },
    }


def _config():
    return {"currencies": _currencies()}


def test_currencies_config_preserves_symbol_overrides():
    """symbol_overrides 必须通过 Pydantic 校验（extra=allow），不再被 model_dump 丢弃。"""
    cfg = CurrenciesConfig(**_currencies())
    dumped = cfg.model_dump()
    assert "symbol_overrides" in dumped
    assert dumped["symbol_overrides"]["ARB"]["leverage_default"] == 4
    assert dumped["symbol_overrides"]["ARB"]["grid_spacing_min"] == 0.009


def test_get_symbol_config_merges_override_over_tier():
    cfg = get_symbol_config("ARB-USDT-SWAP", _config())
    # tier2 默认 leverage_default=5，被 ARB override 覆盖为 4
    assert cfg["leverage_default"] == 4
    assert cfg["grid_spacing_min"] == 0.009
    assert cfg["grid_spacing_max"] == 0.016
    assert cfg["position_limit"] == 0.20
    assert cfg["slippage"] == 0.0015
    # 未覆盖字段仍继承 tier2
    assert cfg["leverage_min"] == 3
    assert cfg["leverage_max"] == 5


def test_get_symbol_config_no_override_falls_back_to_tier():
    cfg = get_symbol_config("SOL-USDT-SWAP", _config())
    assert cfg["leverage_default"] == 5  # tier2 默认
    assert cfg["grid_spacing_min"] == 0.0114


def test_get_symbol_leverage_returns_override():
    assert get_symbol_leverage("ARB-USDT-SWAP", _config()) == 4.0
    assert get_symbol_leverage("SOL-USDT-SWAP", _config()) == 5.0


def test_get_symbol_leverage_range():
    lo, hi = get_symbol_leverage_range("ARB-USDT-SWAP", _config())
    assert (lo, hi) == (3.0, 5.0)


def test_get_symbol_grid_spacing():
    lo, hi = get_symbol_grid_spacing("ARB-USDT-SWAP", _config())
    assert (lo, hi) == (0.009, 0.016)


def test_get_currency_tier_arB_is_tier2():
    assert get_currency_tier("ARB-USDT-SWAP", _config()) == "tier2"
