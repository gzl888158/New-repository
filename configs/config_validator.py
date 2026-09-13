"""
生产级配置校验 - Production Configuration Validation

使用 Pydantic 进行配置 schema 验证、环境变量注入、配置审计。
所有配置在启动时一次性校验，防止运行时因配置错误导致崩溃。
"""
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from loguru import logger


class CurrencyTierSettings(BaseModel):
    """币种分层设置"""
    leverage_min: int = Field(ge=1, le=125, default=2)
    leverage_max: int = Field(ge=1, le=125, default=5)
    leverage_default: int = Field(ge=1, le=125, default=3)
    position_limit: float = Field(ge=0.01, le=1.0, default=0.25)
    grid_spacing_min: float = Field(ge=0.001, le=0.1, default=0.0114)
    grid_spacing_max: float = Field(ge=0.001, le=0.1, default=0.018)
    slippage: float = Field(ge=0.0001, le=0.05, default=0.0015)

    @field_validator('leverage_default')
    @classmethod
    def leverage_default_in_range(cls, v, info):
        """校验默认杠杆在 min-max 范围内"""
        if 'leverage_min' in info.data and 'leverage_max' in info.data:
            if v < info.data['leverage_min'] or v > info.data['leverage_max']:
                raise ValueError(f'leverage_default={v} not in [{info.data["leverage_min"]}, {info.data["leverage_max"]}]')
        return v

    @field_validator('grid_spacing_max')
    @classmethod
    def spacing_max_ge_min(cls, v, info):
        if 'grid_spacing_min' in info.data and v < info.data['grid_spacing_min']:
            raise ValueError(f'grid_spacing_max={v} < grid_spacing_min={info.data["grid_spacing_min"]}')
        return v


class CurrenciesConfig(BaseModel):
    """币种配置"""
    tier1_settings: CurrencyTierSettings = Field(default_factory=CurrencyTierSettings)
    tier1_symbols: List[str] = Field(default_factory=lambda: ["BTC", "ETH"])
    tier2_settings: CurrencyTierSettings = Field(default_factory=CurrencyTierSettings)
    tier2_symbols: List[str] = Field(default_factory=list)
    tier3_settings: CurrencyTierSettings = Field(default_factory=CurrencyTierSettings)
    tier3_symbols: List[str] = Field(default_factory=list)


class ExecutionConfig(BaseModel):
    """执行配置"""
    asyncio_workers: int = Field(ge=1, le=32, default=4)
    batch_processing_size: int = Field(ge=1, le=100, default=10)
    concurrent_strategies: int = Field(ge=1, le=32, default=8)
    max_parallel_tasks: int = Field(ge=1, le=64, default=16)
    thread_pool_size: int = Field(ge=1, le=32, default=8)
    retry_attempts: int = Field(ge=1, le=10, default=3)
    slippage_tolerance: float = Field(ge=0.0001, le=0.05, default=0.001)
    reconciliation_interval: int = Field(ge=1, le=300, default=10)
    rate_limit: Dict[str, int] = Field(default_factory=lambda: {
        "rest_requests_per_second": 5,
        "websocket_requests_per_second": 10
    })


class OKXConfig(BaseModel):
    """OKX API 配置"""
    api_key: str = ""
    secret_key: str = ""
    passphrase: str = ""
    is_testnet: bool = False
    rest_url: str = "https://www.okx.com"
    websocket_url: str = "wss://ws.okx.com:8443/ws/v5/public"
    websocket_private_url: str = "wss://ws.okx.com:8443/ws/v5/private"
    proxy: Optional[str] = None

    @model_validator(mode='after')
    def validate_credentials(self):
        """校验 API 凭证非空（非测试网模式）"""
        if not self.is_testnet:
            if not self.api_key or self.api_key.startswith('${'):
                raise ValueError('OKX API key not configured (set OKX_API_KEY in .env)')
            if not self.secret_key or self.secret_key.startswith('${'):
                raise ValueError('OKX secret key not configured (set OKX_SECRET_KEY in .env)')
            if not self.passphrase or self.passphrase.startswith('${'):
                raise ValueError('OKX passphrase not configured (set OKX_PASSPHRASE in .env)')
        return self


class RiskCircuitBreaker(BaseModel):
    """风控熔断参数"""
    api_timeout: int = Field(ge=5, le=120, default=30)
    btc_movement_threshold: float = Field(ge=0.01, le=0.3, default=0.05)
    btc_movement_window: int = Field(ge=10, le=600, default=60)
    drawdown_acceleration_threshold: float = Field(ge=0.001, le=0.1, default=0.01)
    liquidation_volume_threshold: int = Field(ge=100000, le=100000000, default=1000000)
    liquidation_window: int = Field(ge=60, le=3600, default=300)
    network_timeout: int = Field(ge=10, le=300, default=60)
    rapid_drawdown_threshold: float = Field(ge=0.005, le=0.2, default=0.03)
    rapid_drawdown_window: int = Field(ge=10, le=600, default=60)


class BlackSwanConfig(BaseModel):
    """黑天鹅保护配置"""
    enabled: bool = True
    btc_flash_crash_pct: float = Field(ge=0.01, le=0.3, default=0.03)
    btc_volatility_spike_pct: float = Field(ge=0.01, le=0.5, default=0.05)
    check_interval_seconds: int = Field(ge=1, le=60, default=10)
    emergency_full_close_pct: float = Field(ge=0.05, le=0.5, default=0.12)
    flash_crash_window_seconds: int = Field(ge=60, le=3600, default=300)
    market_circuit_breaker_pct: float = Field(ge=0.03, le=0.3, default=0.08)
    volatility_window_minutes: int = Field(ge=10, le=1440, default=60)


class CorrelationRiskConfig(BaseModel):
    """相关性风险配置"""
    check_interval_seconds: int = Field(ge=10, le=3600, default=300)
    kline_bar: str = "1H"
    lookback_bars: int = Field(ge=10, le=500, default=50)
    max_concentration: float = Field(ge=0.1, le=1.0, default=0.4)
    reduce_ratio: float = Field(ge=0.1, le=1.0, default=0.5)
    threshold: float = Field(ge=0.3, le=1.0, default=0.7)


class RiskConfig(BaseModel):
    """风控配置"""
    black_swan: BlackSwanConfig = Field(default_factory=BlackSwanConfig)
    circuit_breakers: RiskCircuitBreaker = Field(default_factory=RiskCircuitBreaker)
    correlation: CorrelationRiskConfig = Field(default_factory=CorrelationRiskConfig)
    full_close_threshold: float = Field(ge=0.05, le=0.5, default=0.2)
    margin_call_threshold: float = Field(ge=0.1, le=0.9, default=0.3)
    margin_warning_threshold: float = Field(ge=0.1, le=0.9, default=0.5)
    position_loss_threshold: float = Field(ge=0.01, le=0.5, default=0.1)

    @model_validator(mode='after')
    def validate_thresholds_order(self):
        if self.margin_warning_threshold <= self.margin_call_threshold:
            raise ValueError('margin_warning_threshold must be > margin_call_threshold')
        return self


class TradingConfig(BaseModel):
    """交易配置"""
    total_capital: float = Field(ge=1.0, default=10.0)
    trading_capital_ratio: float = Field(ge=0.1, le=1.0, default=0.95)
    max_drawdown: float = Field(ge=0.05, le=0.5, default=0.25)
    daily_max_loss: float = Field(ge=0.005, le=0.2, default=0.04)
    daily_risk_limit: float = Field(ge=0.005, le=0.1, default=0.03)
    risk_per_trade: float = Field(ge=0.001, le=0.1, default=0.025)
    max_total_leverage: float = Field(ge=1.0, le=125.0, default=5.0)
    max_concurrent_positions: int = Field(ge=1, le=20, default=3)
    min_notional_usd: float = Field(ge=1.0, le=100.0, default=3.0)
    min_balance: float = Field(ge=1.0, default=5.0)
    target_utilization: float = Field(ge=0.1, le=1.0, default=0.9)
    profit_reserve_ratio: float = Field(ge=0.0, le=0.5, default=0.0)
    risk_reserve_ratio: float = Field(ge=0.0, le=0.5, default=0.05)
    max_slippage_pct: float = Field(ge=0.0001, le=0.05, default=0.001)
    maker_fee_rate: float = Field(ge=0.0, le=0.01, default=0.0002)
    taker_fee_rate: float = Field(ge=0.0, le=0.01, default=0.0005)


class SystemConfig(BaseModel):
    """系统配置"""
    name: str = "OKX Quant Trading"
    version: str = "2.0.0"
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    timezone: str = "Asia/Shanghai"


class RedisConfig(BaseModel):
    """Redis 配置"""
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535, default=6379)
    db: int = Field(ge=0, le=15, default=0)
    password: Optional[str] = None


class AppConfig(BaseModel):
    """应用总配置 Schema"""
    system: SystemConfig = Field(default_factory=SystemConfig)
    currencies: CurrenciesConfig = Field(default_factory=CurrenciesConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    okx: OKXConfig = Field(default_factory=OKXConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    monitoring: Dict[str, Any] = Field(default_factory=dict)
    notifications: Dict[str, Any] = Field(default_factory=dict)
    sqlite: Dict[str, Any] = Field(default_factory=dict)
    strategies: Dict[str, Any] = Field(default_factory=dict)
    hardware: Dict[str, Any] = Field(default_factory=dict)


def validate_config(raw_config: Dict[str, Any]) -> AppConfig:
    """
    校验并解析原始配置。

    Args:
        raw_config: 从 config.yaml 加载的原始字典

    Returns:
        校验通过的 AppConfig 对象

    Raises:
        ValueError: 配置校验失败（含详细错误信息）
    """
    try:
        config = AppConfig(**raw_config)
        logger.info("Configuration validation PASSED")
        return config
    except Exception as e:
        logger.critical(f"Configuration validation FAILED: {e}")
        raise ValueError(f"Invalid configuration: {e}") from e


def validate_config_partial(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    部分校验：仅校验关键字段，非关键字段允许缺失。
    用于向后兼容非完整配置。
    """
    warnings = []

    # 校验 OKX 凭证
    okx = raw_config.get("okx", {})
    if okx:
        api_key = okx.get("api_key", "")
        if not api_key or api_key.startswith("${"):
            warnings.append("OKX API key not configured")
        secret_key = okx.get("secret_key", "")
        if not secret_key or secret_key.startswith("${"):
            warnings.append("OKX secret key not configured")

    # 校验风控参数
    risk = raw_config.get("risk", {})
    if risk:
        margin_warn = risk.get("margin_warning_threshold", 0.5)
        margin_call = risk.get("margin_call_threshold", 0.3)
        if margin_warn <= margin_call:
            warnings.append(f"margin_warning_threshold({margin_warn}) <= margin_call_threshold({margin_call})")

    # 校验交易参数
    trading = raw_config.get("trading", {})
    if trading:
        max_leverage = trading.get("max_total_leverage", 5.0)
        if max_leverage > 125:
            warnings.append(f"max_total_leverage({max_leverage}) exceeds OKX maximum (125)")

    if warnings:
        logger.warning(f"Configuration warnings: {len(warnings)} issues found")
        for w in warnings:
            logger.warning(f"  - {w}")

    return raw_config


def audit_config(config: AppConfig) -> Dict[str, Any]:
    """配置审计：生成配置审计报告"""
    audit = {
        "version": config.system.version,
        "okx": {
            "is_testnet": config.okx.is_testnet,
            "has_api_key": bool(config.okx.api_key and not config.okx.api_key.startswith("${")),
            "has_proxy": bool(config.okx.proxy),
        },
        "risk": {
            "black_swan_enabled": config.risk.black_swan.enabled,
            "max_leverage": config.trading.max_total_leverage,
            "max_drawdown": config.trading.max_drawdown,
            "daily_max_loss": config.trading.daily_max_loss,
        },
        "execution": {
            "concurrent_strategies": config.execution.concurrent_strategies,
            "retry_attempts": config.execution.retry_attempts,
            "slippage_tolerance": config.execution.slippage_tolerance,
        },
        "trading": {
            "total_capital": config.trading.total_capital,
            "trading_capital_ratio": config.trading.trading_capital_ratio,
            "effective_capital": round(config.trading.total_capital * config.trading.trading_capital_ratio, 2),
            "max_concurrent_positions": config.trading.max_concurrent_positions,
            "target_utilization": config.trading.target_utilization,
        },
        "warnings": [],
    }

    # 审计检查
    if config.trading.total_capital < 100:
        audit["warnings"].append(f"Low capital: ${config.trading.total_capital:.2f}")

    if config.trading.max_total_leverage > 20:
        audit["warnings"].append(f"High leverage: {config.trading.max_total_leverage}x")

    if config.trading.max_drawdown > 0.35:
        audit["warnings"].append(f"High max drawdown: {config.trading.max_drawdown * 100:.0f}%")

    if config.risk.black_swan.enabled and config.risk.black_swan.emergency_full_close_pct < 0.05:
        audit["warnings"].append(f"Low emergency close threshold: {config.risk.black_swan.emergency_full_close_pct * 100:.0f}%")

    logger.info(f"Configuration audit: {len(audit['warnings'])} warnings, effective capital=${audit['trading']['effective_capital']:.2f}")
    return audit