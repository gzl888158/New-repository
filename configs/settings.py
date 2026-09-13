"""
定义系统各模块配置模型，并支持配置加载与币种层级、交易对查询。
"""
import os
import json
import yaml
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

load_dotenv()


class SystemConfig(BaseModel):
    name: str
    version: str
    timezone: str
    log_level: str

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()


class OKXConfig(BaseModel):
    api_key: str
    secret_key: str
    passphrase: str
    is_testnet: bool
    rest_url: str
    websocket_url: str
    websocket_private_url: str
    proxy: Optional[str] = None
    api_keys: Optional[List[Dict[str, str]]] = None


class RedisConfig(BaseModel):
    host: str
    port: int = Field(ge=1, le=65535)
    db: int = Field(ge=0, le=15)
    password: Optional[str] = None
    enabled: bool = True


class SQLiteConfig(BaseModel):
    db_path: str


class MonitoringConfig(BaseModel):
    alert_webhook: Optional[str] = None
    metrics_port: int = Field(ge=1, le=65535)
    api_latency_warning_ms: int = Field(gt=0)
    api_latency_critical_ms: int = Field(gt=0)
    log_persistence: Optional[Dict[str, Any]] = None
    alerts: Optional[Dict[str, Any]] = None


class HardwareConfig(BaseModel):
    cpu_cores: int = Field(ge=1)
    logical_processors: int = Field(ge=1)
    high_priority_cores: List[int]
    medium_priority_cores: List[int]
    low_priority_cores: List[int]
    max_cpu_usage: int = Field(ge=1, le=100)
    temperature_threshold: float = Field(gt=0)
    max_memory_usage_mb: int = Field(gt=0)
    memory_warning_threshold_mb: int = Field(gt=0)


class ProfitTarget(BaseModel):
    threshold: float = Field(ge=0)
    close_ratio: float = Field(ge=0, le=1)


class GridStrategyConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool
    grid_count_min: int = Field(ge=2)
    grid_count_max: int = Field(ge=2)
    martingale_layers: int = Field(ge=0)
    martingale_coefficient: float = Field(ge=1.0)
    dynamic_adjust_interval: int = Field(gt=0)
    volatility_threshold: float = Field(ge=0)
    atr_period: int = Field(ge=1)
    atr_multiplier: float = Field(ge=0)
    volume_profile_period: int = Field(ge=1)
    price_update_interval_ms: int = Field(gt=0)
    stop_loss_pct: float = Field(ge=0, le=1)
    trailing_stop_enabled: bool
    trailing_stop_pct: float = Field(ge=0, le=1)
    breakeven_trigger_pct: float = Field(ge=0, le=1)
    breakeven_stop_pct: float = Field(ge=0, le=1)
    min_signal_quality: float = Field(ge=0, le=1)
    extreme_volatility_pause_minutes: Optional[int] = None
    max_stop_loss_pct: Optional[float] = None
    min_signal_interval: Optional[int] = None
    signal_cooldown: Optional[float] = None
    take_profit_enabled: Optional[bool] = None
    tp1_ratio: Optional[float] = None
    tp1_pct: Optional[float] = None
    tp2_ratio: Optional[float] = None
    tp2_pct: Optional[float] = None
    tp3_ratio: Optional[float] = None
    tp3_trailing_pct: Optional[float] = None
    time_exit_enabled: Optional[bool] = None
    max_hold_hours: Optional[float] = None
    time_exit_partial_pct: Optional[float] = None
    time_exit_after_hours: Optional[float] = None
    volatility_stop_enabled: Optional[bool] = None
    vol_spike_threshold: Optional[float] = None
    vol_stop_partial_pct: Optional[float] = None
    volatility_lockout_minutes: Optional[int] = None
    # 生产级各币种网格状态管理
    coin_health_check_interval: Optional[int] = None
    coin_max_lifecycle_events: Optional[int] = None
    coin_health_max_errors: Optional[int] = None
    coin_health_degraded_score: Optional[float] = None
    coin_auto_pause_on_error: Optional[bool] = None

    @model_validator(mode="after")
    def check_grid_range(self) -> "GridStrategyConfig":
        if self.grid_count_min > self.grid_count_max:
            raise ValueError("grid_count_min must be <= grid_count_max")
        return self


class TrendStrategyConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool
    timeframe: str
    confirmation_periods: List[str]
    max_additions: int = Field(ge=0)
    initial_position_ratio: float = Field(ge=0, le=1)
    addition_ratio: float = Field(ge=0, le=1)
    profit_targets: List[ProfitTarget]
    trailing_stop_tier1: float = Field(ge=0)
    trailing_stop_tier2: float = Field(ge=0)
    trailing_stop_tier3: float = Field(ge=0)
    false_break_threshold: float = Field(ge=0)
    trailing_stop_initial: float = Field(ge=0)
    trailing_stop_min: float = Field(ge=0)
    volatility_adaptive_sl: bool
    profit_protection_max_drawdown: float = Field(ge=0)
    reversal_confirmation_bars: int = Field(ge=1)
    reversal_adx_threshold: float = Field(ge=0)
    adaptive_enabled: bool
    min_signal_quality: float = Field(ge=0, le=1)
    market_state_lookback: int = Field(ge=5)
    breakeven_trigger_pct: float = Field(ge=0, le=1)
    breakeven_stop_pct: float = Field(ge=0, le=1)
    max_stop_loss_pct: Optional[float] = Field(default=None, ge=0, le=1)
    take_profit_enabled: Optional[bool] = None
    tp1_ratio: Optional[float] = None
    tp1_pct: Optional[float] = None
    tp2_ratio: Optional[float] = None
    tp2_pct: Optional[float] = None
    tp3_ratio: Optional[float] = None
    tp3_trailing_pct: Optional[float] = None
    time_exit_enabled: Optional[bool] = None
    max_hold_hours: Optional[float] = None
    time_exit_partial_pct: Optional[float] = None
    time_exit_after_hours: Optional[float] = None
    volatility_stop_enabled: Optional[bool] = None
    vol_spike_threshold: Optional[float] = None
    vol_stop_partial_pct: Optional[float] = None
    volatility_lockout_minutes: Optional[int] = None


class ScalpingStrategyConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool
    run_hours_start: int = Field(ge=0, le=23)
    run_hours_end: int = Field(ge=0, le=24)
    min_drop: float = Field(ge=0)
    min_rise: float = Field(ge=0)
    price_deviation: float = Field(ge=0)
    max_hold_minutes: int = Field(gt=0)
    profit_target_min: float = Field(ge=0)
    profit_target_max: float = Field(ge=0)
    stop_loss: float = Field(ge=0)
    momentum_periods: int = Field(ge=1)
    reversal_threshold: float = Field(ge=0)
    rsi_period: int = Field(ge=2)
    rsi_oversold: float = Field(ge=0, le=100)
    rsi_overbought: float = Field(ge=0, le=100)
    stoch_period: int = Field(ge=2)
    volume_delta_threshold: float
    trailing_stop_activation: float = Field(ge=0)
    tick_processing_interval_ms: int = Field(gt=0)
    profit_taking_levels: List[ProfitTarget]
    trailing_stop_initial: float = Field(ge=0)
    trailing_stop_min: float = Field(ge=0)
    volatility_adaptive_sl: bool
    profit_protection_max_drawdown: float = Field(ge=0)
    adaptive_enabled: bool
    min_signal_quality: float = Field(ge=0, le=1)
    market_state_lookback: int = Field(ge=5)
    max_positions: Optional[int] = None
    position_sizing_mode: Optional[str] = None
    take_profit_enabled: Optional[bool] = None
    tp1_ratio: Optional[float] = None
    tp1_pct: Optional[float] = None
    tp2_ratio: Optional[float] = None
    tp2_pct: Optional[float] = None
    tp3_ratio: Optional[float] = None
    tp3_trailing_pct: Optional[float] = None
    time_exit_enabled: Optional[bool] = None
    max_hold_hours: Optional[float] = None
    time_exit_partial_pct: Optional[float] = None
    time_exit_after_hours: Optional[float] = None
    volatility_stop_enabled: Optional[bool] = None
    vol_spike_threshold: Optional[float] = None
    vol_stop_partial_pct: Optional[float] = None
    volatility_lockout_minutes: Optional[int] = None

    @model_validator(mode="after")
    def check_rsi_range(self) -> "ScalpingStrategyConfig":
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("rsi_oversold must be < rsi_overbought")
        return self

    @model_validator(mode="after")
    def check_profit_range(self) -> "ScalpingStrategyConfig":
        if self.profit_target_min > self.profit_target_max:
            raise ValueError("profit_target_min must be <= profit_target_max")
        return self


class ArbitrageStrategyConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool
    # ---- 资金费率套利 ----
    funding_rate_threshold: float
    consecutive_periods: int = Field(ge=1)
    close_threshold: float = Field(ge=0)
    leverage: int = Field(ge=1)
    # ---- 基差套利 ----
    basis_threshold: float = Field(ge=0)
    basis_reversal_threshold: float = Field(ge=0)
    # ---- 费率预测 ----
    rate_forecast_window: int = Field(ge=1)
    # ---- 风控参数 ----
    stop_loss_pct: float = Field(ge=0, le=1)
    take_profit_pct: float = Field(ge=0, le=1)
    max_hold_hours: int = Field(gt=0)
    hedge_leverage: int = Field(ge=1)
    min_net_profit_pct: float = Field(ge=0)
    min_signal_quality: float = Field(ge=0, le=1)
    # ---- 套利类型开关 ----
    arbitrage_types: Optional[List[str]] = None  # ["funding", "basis", "correlation"]
    check_interval: int = Field(ge=60, default=900)
    forecast_interval: int = Field(ge=300, default=7200)
    adaptive_params: bool = True
    # ---- v2.0 生产级增强 ----
    # 凯利公式
    kelly_enabled: bool = True
    kelly_fraction: float = Field(ge=0, le=1, default=0.25)
    # ATR动态止损
    atr_stop_enabled: bool = True
    atr_stop_multiplier: float = Field(ge=0.5, le=10, default=2.0)
    # 市场状态自适应
    regime_adaptive: bool = True
    # 分批执行
    batch_execution: bool = True
    batch_min_notional: float = Field(ge=100, default=500.0)
    batch_count: int = Field(ge=2, le=20, default=3)
    batch_interval_sec: int = Field(ge=10, le=300, default=30)
    # 告警集成
    alert_integration: bool = True
    # 最大并发套利数
    max_concurrent: int = Field(ge=1, le=20, default=5)
    # 净敞口硬限制
    max_net_exposure_pct: float = Field(ge=0, le=1, default=0.05)
    # 相关性矩阵缓存
    correlation_cache_ttl: int = Field(ge=300, default=3600)
    correlation_lookback: int = Field(ge=10, default=48)
    correlation_kline_bar: str = "1H"


class SpotGridStrategyConfig(BaseModel):
    enabled: bool
    grid_count_min: int = Field(ge=2)
    grid_count_max: int = Field(ge=2)
    dynamic_adjust_interval: int = Field(gt=0)
    volatility_threshold: float = Field(ge=0)
    atr_period: int = Field(ge=1)
    atr_multiplier: float = Field(ge=0)
    volume_profile_period: int = Field(ge=1)
    dynamic_grid_count: bool
    volatility_adaptive_spacing: bool
    multi_symbol_scheduling: bool
    min_grid_spacing: float = Field(ge=0)
    max_grid_spacing: float = Field(ge=0)
    min_signal_quality: float = Field(ge=0, le=1)
    take_profit_pct: float = Field(ge=0, le=1)
    stop_loss_pct: float = Field(ge=0, le=1)
    trend_filter_enabled: bool
    ema_fast_period: int = Field(ge=1)
    ema_slow_period: int = Field(ge=1)
    grid_reset_enabled: bool
    grid_reset_delay: int = Field(ge=0)
    volume_filter_enabled: bool
    volume_threshold_ratio: float = Field(ge=0)

    @model_validator(mode="after")
    def check_grid_range(self) -> "SpotGridStrategyConfig":
        if self.grid_count_min > self.grid_count_max:
            raise ValueError("grid_count_min must be <= grid_count_max")
        if self.min_grid_spacing > self.max_grid_spacing:
            raise ValueError("min_grid_spacing must be <= max_grid_spacing")
        return self


class SpotMartingaleStrategyConfig(BaseModel):
    model_config = {"extra": "allow"}
    enabled: bool
    base_position_ratio: float = Field(ge=0, le=1)
    martingale_coefficient: float = Field(ge=1.0)
    max_layers: int = Field(ge=1)
    price_drop_pct: float = Field(ge=0, le=1)
    take_profit_pct: float = Field(ge=0, le=1)
    stop_loss_pct: float = Field(ge=0, le=1)
    check_interval: int = Field(gt=0)
    min_signal_quality: float = Field(ge=0, le=1)
    atr_period: int = Field(ge=1)
    volatility_adaptive: bool
    volatility_threshold_high: float = Field(ge=0, le=1)
    volatility_threshold_low: float = Field(ge=0, le=1)
    dynamic_take_profit: bool
    take_profit_decay: float = Field(ge=0)
    ema_fast_period: int = Field(ge=1)
    ema_slow_period: int = Field(ge=1)
    bear_market_suspend: bool
    max_hold_time_hours: int = Field(ge=1)
    partial_close_enabled: bool
    partial_close_threshold: int = Field(ge=1)
    partial_close_ratio: float = Field(ge=0, le=1)
    dynamic_drop_threshold: bool
    max_daily_trades_per_symbol: int = Field(ge=0)


class StrategiesConfig(BaseModel):
    model_config = {"extra": "allow"}
    grid: GridStrategyConfig
    spot_grid: SpotGridStrategyConfig
    spot_martingale: SpotMartingaleStrategyConfig
    trend: TrendStrategyConfig
    scalping: ScalpingStrategyConfig
    arbitrage: ArbitrageStrategyConfig


class TierSettings(BaseModel):
    leverage_min: int = Field(ge=1)
    leverage_max: int = Field(ge=1)
    leverage_default: int = Field(ge=1)
    position_limit: float = Field(ge=0, le=1)
    grid_spacing_min: float = Field(ge=0)
    grid_spacing_max: float = Field(ge=0)
    slippage: float = Field(ge=0)

    @model_validator(mode="after")
    def check_leverage_range(self) -> "TierSettings":
        if self.leverage_min > self.leverage_max:
            raise ValueError("leverage_min must be <= leverage_max")
        if not (self.leverage_min <= self.leverage_default <= self.leverage_max):
            raise ValueError("leverage_default must be between leverage_min and leverage_max")
        return self

    @model_validator(mode="after")
    def check_grid_spacing_range(self) -> "TierSettings":
        if self.grid_spacing_min > self.grid_spacing_max:
            raise ValueError("grid_spacing_min must be <= grid_spacing_max")
        return self


class CurrenciesConfig(BaseModel):
    tier1_symbols: List[str]
    tier1_settings: TierSettings
    tier2_symbols: List[str]
    tier2_settings: TierSettings
    tier3_symbols: List[str]
    tier3_settings: TierSettings


class TradingConfig(BaseModel):
    model_config = {"extra": "allow"}
    total_capital: float = Field(gt=0)
    trading_capital_ratio: float = Field(ge=0, le=1)
    risk_reserve_ratio: float = Field(ge=0, le=1)
    profit_reserve_ratio: float = Field(ge=0, le=1)
    max_drawdown: float = Field(ge=0, le=1)
    daily_max_loss: float = Field(ge=0, le=1)
    hourly_max_loss: float = Field(ge=0, le=1)
    grid_allocation: float = Field(ge=0, le=1)
    spot_grid_allocation: float = Field(ge=0, le=1)
    spot_martingale_allocation: float = Field(ge=0, le=1)
    trend_allocation: float = Field(ge=0, le=1)
    scalping_allocation: float = Field(ge=0, le=1)
    arbitrage_allocation: float = Field(ge=0, le=1)
    max_consecutive_losses: int = Field(ge=0)
    max_daily_trades_per_symbol: int = Field(ge=0)
    max_concurrent_positions: int = Field(ge=1)
    max_queue_size: int = Field(default=200, ge=50)
    max_total_leverage: float = Field(ge=1)
    compound_enabled: bool
    min_notional_usd: float = Field(ge=0, default=5.0)
    idle_cash_allocation: bool = True
    min_utilization: float = Field(ge=0, le=1, default=0.75)
    target_utilization: float = Field(ge=0, le=1, default=0.95)
    compound_reinvest_ratio: float = Field(ge=0, le=1)
    risk_per_trade: float = Field(ge=0, le=1)
    target_return: float = Field(ge=0)
    trailing_profit_lock: float = Field(ge=0)
    daily_risk_limit: float = Field(ge=0, le=1)
    min_margin_per_trade: float = Field(ge=0)
    taker_fee_rate: float = Field(ge=0)
    maker_fee_rate: float = Field(ge=0)
    max_slippage_pct: float = Field(ge=0)
    min_profit_cost_ratio: float = Field(ge=0)
    min_hold_minutes: int = Field(ge=0)
    funding_rate_check: bool
    funding_rate_min_hold: float = Field(ge=0)
    min_balance: Optional[float] = None
    daily_max_loss_hard: Optional[float] = None
    daily_max_trades: Optional[int] = None

    @model_validator(mode="after")
    def check_allocation_sum(self) -> "TradingConfig":
        total = (
            self.grid_allocation
            + self.spot_grid_allocation
            + self.spot_martingale_allocation
            + self.trend_allocation
            + self.scalping_allocation
            + self.arbitrage_allocation
        )
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Strategy allocations sum to {total:.4f}, must be 1.0 "
                f"(grid={self.grid_allocation}, spot_grid={self.spot_grid_allocation}, "
                f"spot_martingale={self.spot_martingale_allocation}, trend={self.trend_allocation}, "
                f"scalping={self.scalping_allocation}, arbitrage={self.arbitrage_allocation})"
            )
        return self

    @model_validator(mode="after")
    def check_capital_ratio_sum(self) -> "TradingConfig":
        total = (
            self.trading_capital_ratio
            + self.risk_reserve_ratio
            + self.profit_reserve_ratio
        )
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(
                f"Capital ratios sum to {total:.4f}, must be 1.0 "
                f"(trading={self.trading_capital_ratio}, "
                f"risk_reserve={self.risk_reserve_ratio}, "
                f"profit_reserve={self.profit_reserve_ratio})"
            )
        return self


class CircuitBreakersConfig(BaseModel):
    btc_movement_threshold: float = Field(ge=0)
    btc_movement_window: int = Field(gt=0)
    liquidation_volume_threshold: float = Field(ge=0)
    liquidation_window: int = Field(gt=0)
    api_timeout: int = Field(gt=0)
    network_timeout: int = Field(gt=0)
    rapid_drawdown_threshold: float = Field(ge=0, le=1)
    rapid_drawdown_window: int = Field(gt=0)
    drawdown_acceleration_threshold: float = Field(ge=0)


class CorrelationRiskConfig(BaseModel):
    threshold: float = Field(ge=0, le=1)
    max_concentration: float = Field(ge=0, le=1)
    lookback_bars: int = Field(ge=5)
    kline_bar: str
    check_interval_seconds: int = Field(gt=0)
    reduce_ratio: float = Field(ge=0, le=1)


class BlackSwanConfig(BaseModel):
    enabled: bool = True
    check_interval_seconds: int = Field(gt=0, default=10)
    btc_flash_crash_pct: float = Field(ge=0, le=1, default=0.03)
    btc_volatility_spike_pct: float = Field(ge=0, le=1, default=0.05)
    flash_crash_window_seconds: int = Field(gt=0, default=300)
    volatility_window_minutes: int = Field(gt=0, default=60)
    market_circuit_breaker_pct: float = Field(ge=0, le=1, default=0.08)
    emergency_full_close_pct: float = Field(ge=0, le=1, default=0.12)
    volume_spike_ratio: float = Field(ge=1.0, default=3.0)
    volume_lookback_bars: int = Field(ge=5, default=20)
    monitor_symbols: List[str] = [
        "BTC-USDT-SWAP",
        "ETH-USDT-SWAP",
        "SOL-USDT-SWAP",
    ]

    @model_validator(mode="after")
    def check_thresholds(self) -> "BlackSwanConfig":
        if self.btc_flash_crash_pct >= self.market_circuit_breaker_pct:
            raise ValueError("btc_flash_crash_pct must be < market_circuit_breaker_pct")
        if self.market_circuit_breaker_pct >= self.emergency_full_close_pct:
            raise ValueError("market_circuit_breaker_pct must be < emergency_full_close_pct")
        return self


class RiskConfig(BaseModel):
    margin_call_threshold: float = Field(gt=0)
    margin_warning_threshold: float = Field(gt=0)
    position_loss_threshold: float = Field(ge=0, le=1)
    full_close_threshold: float = Field(ge=0, le=1)
    circuit_breakers: CircuitBreakersConfig
    correlation: CorrelationRiskConfig
    black_swan: BlackSwanConfig

    # 五层风控拦截层 L1/L2/L3 配置（必须显式声明，否则 model_dump() 会丢弃）
    # max_symbol_position_ratio 限制的是"名义价值/equity"，杠杆交易可达 2x-5x，故上限放宽到 10
    max_symbol_position_ratio: float = Field(ge=0, le=10, default=0.25)
    max_single_order_usd: float = Field(gt=0, default=500.0)
    max_api_rps: int = Field(ge=1, default=10)
    max_slippage_pct: float = Field(ge=0, default=0.003)
    max_spread_pct: float = Field(ge=0, default=0.002)
    max_latency_ms: int = Field(gt=0, default=3000)
    liquidation_warning_pct: float = Field(ge=0, le=1, default=0.10)
    liquidation_critical_pct: float = Field(ge=0, le=1, default=0.05)
    position_loss_tier1_pct: float = Field(ge=0, le=1, default=0.05)
    position_loss_tier2_pct: float = Field(ge=0, le=1, default=0.10)
    position_loss_tier3_pct: float = Field(ge=0, le=1, default=0.15)
    loss_tier1_reduce_ratio: float = Field(ge=0, le=1, default=0.30)
    loss_tier2_reduce_ratio: float = Field(ge=0, le=1, default=0.50)
    loss_tier3_reduce_ratio: float = Field(ge=0, le=1, default=1.00)
    max_funding_loss_pct: float = Field(ge=0, default=0.005)
    network_timeout_threshold: int = Field(gt=0, default=120)
    price_anomaly_threshold: float = Field(ge=0, default=0.10)
    emergency_cooldown_seconds: int = Field(gt=0, default=3600)
    contract_risk: Optional[Dict[str, Any]] = None  # 激进合约专属风险分析框架配置

    @model_validator(mode="after")
    def check_margin_thresholds(self) -> "RiskConfig":
        if self.margin_call_threshold >= self.margin_warning_threshold:
            raise ValueError("margin_call_threshold must be < margin_warning_threshold")
        return self

    @model_validator(mode="after")
    def check_loss_thresholds(self) -> "RiskConfig":
        if self.position_loss_threshold >= self.full_close_threshold:
            raise ValueError("position_loss_threshold must be < full_close_threshold")
        return self

    @model_validator(mode="after")
    def check_liquidation_thresholds(self) -> "RiskConfig":
        if self.liquidation_critical_pct >= self.liquidation_warning_pct:
            raise ValueError("liquidation_critical_pct must be < liquidation_warning_pct")
        return self

    @model_validator(mode="after")
    def check_loss_tier_thresholds(self) -> "RiskConfig":
        if not (self.position_loss_tier1_pct < self.position_loss_tier2_pct < self.position_loss_tier3_pct):
            raise ValueError("position_loss_tier1/2/3_pct must be strictly increasing")
        return self


class RateLimitConfig(BaseModel):
    websocket_requests_per_second: int = Field(ge=1)
    rest_requests_per_second: int = Field(ge=1)


class ExecutionConfig(BaseModel):
    order_priority: List[str]
    rate_limit: RateLimitConfig
    reconciliation_interval: int = Field(gt=0)
    retry_attempts: int = Field(ge=0)
    slippage_tolerance: float = Field(ge=0)
    concurrent_strategies: int = Field(ge=1)
    max_parallel_tasks: int = Field(ge=1)
    thread_pool_size: int = Field(ge=1)
    asyncio_workers: int = Field(ge=1)
    batch_processing_size: int = Field(ge=1)
    stale_order: Optional[Dict[str, Any]] = None
    slippage_optimizer: Optional[Dict[str, Any]] = None
    order_synchronizer: Optional[Dict[str, Any]] = None
    manual_intervention: Optional[Dict[str, Any]] = None
    orphan_position_auto_close: bool = False


class NotificationsConfig(BaseModel):
    enabled: bool
    level: str
    providers: List[str]

    @field_validator("level")
    @classmethod
    def validate_level(cls, v: str) -> str:
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in allowed:
            raise ValueError(f"notification level must be one of {allowed}")
        return v.upper()


class TelegramConfig(BaseModel):
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None


class MarketDataConfig(BaseModel):
    max_data_delay_seconds: int = 30
    max_price_change_pct: float = 0.05
    quality_monitor_interval_seconds: int = 60
    rest_fallback_interval_seconds: int = 5
    ws_fallback_timeout_seconds: int = 10
    redis_tick_ttl_seconds: int = 60


class SignalGenerationConfig(BaseModel):
    """生产级信号生成配置"""
    model_config = {"extra": "allow"}
    enabled: bool = True
    # 信号过滤
    min_signal_weight: float = Field(ge=0, le=1, default=0.35)
    min_quality_score: float = Field(ge=0, le=1, default=0.35)
    max_signals_per_second: int = Field(ge=1, default=10)
    max_signals_per_symbol_second: int = Field(ge=1, default=3)
    signal_cooldown_seconds: float = Field(ge=0, default=5)
    # 信号冲突解决
    conflict_resolution: str = "majority_vote"
    max_conflicting_signals: int = Field(ge=1, default=3)
    opposite_signal_delay_seconds: float = Field(ge=0, default=30)
    # 信号确认
    confirmation_required: bool = True
    min_confirmation_rules: int = Field(ge=1, default=2)
    confirmation_window_seconds: float = Field(ge=1, default=10)
    # 信号增强
    multi_timeframe_confirm: bool = True
    volume_confirmation: bool = True
    order_book_confirmation: bool = False
    market_regime_filter: bool = True
    # 信号聚合
    signal_aggregation: bool = True
    aggregation_window_seconds: float = Field(ge=1, default=5)
    max_aggregated_signals: int = Field(ge=1, default=5)
    # 自适应学习
    adaptive_learning: Optional[Dict[str, Any]] = None
    # 信号规则配置
    rules: Optional[Dict[str, Any]] = None
    # 信号上下文字段
    include_market_context: bool = True
    include_risk_metrics: bool = True
    include_position_context: bool = True
    # 信号持久化
    persist_signals: bool = True
    signal_history_size: int = Field(ge=50, default=500)
    # 信号发布
    dispatch_mode: str = "immediate"
    dispatch_priority: str = "quality_first"


# ═══════════════════════════════════════════════════════════════
# 防针对量化数据配置模型
# ═══════════════════════════════════════════════════════════════

class SignalObfuscatorConfig(BaseModel):
    """信号混淆器配置 — 第一层防护"""
    enabled: bool = True
    mode: str = "standard"  # off / light / standard / aggressive
    delay_enabled: bool = True
    delay_min_ms: int = Field(ge=0, default=50)
    delay_max_ms: int = Field(ge=0, default=500)
    delay_distribution: str = "exponential"  # uniform / exponential / normal
    weight_jitter_enabled: bool = True
    weight_jitter_pct: float = Field(ge=0, le=1, default=0.05)
    dummy_injection_enabled: bool = True
    dummy_probability: float = Field(ge=0, le=1, default=0.03)
    dummy_max_per_batch: int = Field(ge=0, default=2)
    type_rotation_enabled: bool = True
    rotation_probability: float = Field(ge=0, le=1, default=0.08)
    expiry_jitter_enabled: bool = True
    expiry_jitter_seconds: int = Field(ge=0, default=10)
    adaptive_enabled: bool = True
    pattern_detection_threshold: float = Field(ge=0, le=1, default=0.7)
    adaptive_decay: float = Field(ge=0, le=1, default=0.95)


class OrderFingerprintMaskerConfig(BaseModel):
    """订单指纹掩盖器配置 — 第二层防护"""
    enabled: bool = True
    mode: str = "standard"  # off / light / standard / aggressive
    quantity_jitter_enabled: bool = True
    quantity_jitter_min_pct: float = Field(ge=0, le=1, default=0.03)
    quantity_jitter_max_pct: float = Field(ge=0, le=1, default=0.08)
    price_jitter_enabled: bool = True
    price_jitter_pct: float = Field(ge=0, le=1, default=0.0005)
    split_enabled: bool = True
    split_min_notional: float = Field(ge=0, default=500.0)
    split_min_parts: int = Field(ge=1, default=2)
    split_max_parts: int = Field(ge=1, default=5)
    split_size_ratio_min: float = Field(ge=0, le=1, default=0.15)
    split_size_ratio_max: float = Field(ge=0, le=1, default=0.6)
    time_slice_enabled: bool = True
    time_slice_min_ms: int = Field(ge=0, default=100)
    time_slice_max_ms: int = Field(ge=0, default=2000)
    time_slice_distribution: str = "exponential"
    type_rotation_enabled: bool = True
    limit_order_ratio: float = Field(ge=0, le=1, default=0.7)
    post_only_ratio: float = Field(ge=0, le=1, default=0.15)


class AntiPatternDetectorConfig(BaseModel):
    """反模式检测器配置 — 第三层防护"""
    enabled: bool = True
    check_interval_seconds: int = Field(ge=10, default=60)
    signal_window_size: int = Field(ge=10, default=100)
    order_window_size: int = Field(ge=10, default=50)
    analysis_window_minutes: int = Field(ge=5, default=30)
    self_similarity_enabled: bool = True
    similarity_threshold: float = Field(ge=0, le=1, default=0.75)
    similarity_weight: float = Field(ge=0, le=1, default=0.25)
    timing_entropy_enabled: bool = True
    entropy_threshold_low: float = Field(ge=0, le=1, default=0.5)
    entropy_threshold_warn: float = Field(ge=0, le=1, default=0.3)
    timing_entropy_weight: float = Field(ge=0, le=1, default=0.20)
    volume_pattern_enabled: bool = True
    volume_cv_threshold: float = Field(ge=0, default=0.15)
    volume_pattern_weight: float = Field(ge=0, le=1, default=0.20)
    price_clustering_enabled: bool = True
    price_cluster_radius_pct: float = Field(ge=0, default=0.002)
    price_cluster_ratio_threshold: float = Field(ge=0, le=1, default=0.6)
    price_clustering_weight: float = Field(ge=0, le=1, default=0.20)
    front_running_enabled: bool = True
    front_running_slippage_threshold: float = Field(ge=0, default=0.003)
    front_running_ratio_threshold: float = Field(ge=0, le=1, default=0.3)
    front_running_weight: float = Field(ge=0, le=1, default=0.15)


class AntiTargetingConfig(BaseModel):
    """防针对量化数据总配置 — 三层防护"""
    enabled: bool = True
    signal_obfuscator: SignalObfuscatorConfig = SignalObfuscatorConfig()
    order_fingerprint_masker: OrderFingerprintMaskerConfig = OrderFingerprintMaskerConfig()
    anti_pattern_detector: AntiPatternDetectorConfig = AntiPatternDetectorConfig()


# ═══════════════════════════════════════════════════════════════
# 资金分布 / 资金分配配置模型
# ═══════════════════════════════════════════════════════════════

class CapitalPoolConfig(BaseModel):
    """三级资金池配置"""
    base_ratio: float = Field(ge=0, le=1, default=0.60)
    add_reserve_ratio: float = Field(ge=0, le=1, default=0.25)
    risk_isolation_ratio: float = Field(ge=0, le=1, default=0.15)

    @model_validator(mode="after")
    def check_pool_ratios(self) -> "CapitalPoolConfig":
        total = self.base_ratio + self.add_reserve_ratio + self.risk_isolation_ratio
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Pool ratios sum to {total:.4f}, must be 1.0 "
                f"(base={self.base_ratio}, add={self.add_reserve_ratio}, risk={self.risk_isolation_ratio})"
            )
        return self


class AllocationAgentConfig(BaseModel):
    """智能资金分配Agent配置"""
    enabled: bool = True
    rebalance_interval: int = Field(ge=60, default=3600)
    min_trade_count: int = Field(ge=1, default=20)
    max_allocation_change: float = Field(ge=0, le=1, default=0.05)
    method: str = "dynamic"


class SymbolAllocationConfig(BaseModel):
    """币种权重动态分配配置"""
    max_symbol_weight: float = Field(ge=0, le=1, default=0.15)
    min_symbol_weight: float = Field(ge=0, le=1, default=0.02)
    rebalance_interval: int = Field(ge=30, default=300)
    max_weight_change: float = Field(ge=0, le=1, default=0.03)
    volatility_weight: float = Field(ge=0, le=1, default=0.35)
    momentum_weight: float = Field(ge=0, le=1, default=0.25)
    liquidity_weight: float = Field(ge=0, le=1, default=0.20)
    performance_weight: float = Field(ge=0, le=1, default=0.20)

    @model_validator(mode="after")
    def check_min_max_weight(self) -> "SymbolAllocationConfig":
        if self.min_symbol_weight > self.max_symbol_weight:
            raise ValueError("min_symbol_weight must be <= max_symbol_weight")
        return self


class LeverageTiersConfig(BaseModel):
    """杠杆分级管理配置"""
    light_min: int = Field(ge=1, default=1)
    light_max: int = Field(ge=1, default=5)
    main_min: int = Field(ge=1, default=5)
    main_max: int = Field(ge=1, default=15)
    absolute_max: int = Field(ge=1, default=15)

    @model_validator(mode="after")
    def check_leverage_tiers(self) -> "LeverageTiersConfig":
        if self.light_min > self.light_max:
            raise ValueError("light_min must be <= light_max")
        if self.main_min > self.main_max:
            raise ValueError("main_min must be <= main_max")
        if self.light_max > self.main_max:
            raise ValueError("light_max must be <= main_max")
        if self.main_max > self.absolute_max:
            raise ValueError("main_max must be <= absolute_max")
        return self


class PnLReallocationConfig(BaseModel):
    """盈亏浮动再分配配置"""
    profit_to_add_ratio: float = Field(ge=0, le=1, default=0.50)
    loss_shrink_ratio: float = Field(ge=0, le=1, default=0.30)
    consecutive_profit_days: int = Field(ge=1, default=3)
    consecutive_loss_days: int = Field(ge=1, default=2)
    max_add_pool_ratio: float = Field(ge=0, le=1, default=0.40)
    min_base_pool_ratio: float = Field(ge=0, le=1, default=0.50)


class HedgeSchedulerConfig(BaseModel):
    """多仓对冲调度配置"""
    max_hedge_ratio: float = Field(ge=0, le=1, default=0.70)
    min_hedge_ratio: float = Field(ge=0, le=1, default=0.30)
    hedge_trigger_volatility: float = Field(ge=0, default=0.04)
    hedge_trigger_drawdown: float = Field(ge=0, le=1, default=0.05)
    max_hedge_positions: int = Field(ge=1, default=10)
    hedge_check_interval: int = Field(ge=10, default=60)

    @model_validator(mode="after")
    def check_hedge_ratio(self) -> "HedgeSchedulerConfig":
        if self.min_hedge_ratio > self.max_hedge_ratio:
            raise ValueError("min_hedge_ratio must be <= max_hedge_ratio")
        return self


class VolatilityTargetingConfig(BaseModel):
    """波动率目标管理配置"""
    target_volatility: float = Field(ge=0, le=1, default=0.20)
    max_volatility: float = Field(ge=0, le=1, default=0.50)
    min_volatility: float = Field(ge=0, le=1, default=0.05)
    ewma_lambda: float = Field(ge=0, le=1, default=0.94)
    garch_omega: float = Field(ge=0, default=0.00001)
    garch_alpha: float = Field(ge=0, le=1, default=0.05)
    garch_beta: float = Field(ge=0, le=1, default=0.90)
    forecast_horizon: int = Field(ge=1, default=21)
    confidence_z: float = Field(default=1.96)

    @model_validator(mode="after")
    def check_vol_range(self) -> "VolatilityTargetingConfig":
        if self.target_volatility > self.max_volatility:
            raise ValueError("target_volatility must be <= max_volatility")
        if self.min_volatility > self.target_volatility:
            raise ValueError("min_volatility must be <= target_volatility")
        return self


class AdaptiveKellyConfig(BaseModel):
    """自适应Kelly仓位优化配置"""
    max_kelly_fraction: float = Field(ge=0, le=1, default=0.25)
    default_kelly_fraction: float = Field(ge=0, le=1, default=0.50)
    min_trade_count_kelly: int = Field(ge=1, default=20)
    kelly_trending_up: float = Field(default=1.10)
    kelly_trending_down: float = Field(default=0.60)
    kelly_ranging: float = Field(default=0.90)
    kelly_high_vol: float = Field(default=0.50)
    kelly_low_vol: float = Field(default=1.00)
    kelly_unknown: float = Field(default=0.80)
    kelly_dd_start: float = Field(ge=0, le=1, default=0.05)
    kelly_dd_max: float = Field(ge=0, le=1, default=0.30)
    kelly_dd_min: float = Field(ge=0, le=1, default=0.30)
    kelly_streak_win_step: float = Field(ge=0, default=0.05)
    kelly_streak_loss_step: float = Field(ge=0, default=0.08)
    kelly_streak_max_adj: float = Field(ge=0, default=0.30)
    # P31: 企业级估计误差修正（凯利公式对胜率/赔率估计误差敏感，用贝叶斯收缩+赔率收缩降低过度下注）
    kelly_use_wilson_lcb: bool = False                   # 可选：Wilson置信下界（更保守，替代贝叶斯收缩）
    kelly_wilson_z: float = Field(ge=0, default=1.645)   # Wilson置信度 z 值
    kelly_win_rate_prior_strength: float = Field(ge=0, default=40.0)  # 胜率先验强度（等效α+β样本量，先验50%）
    kelly_b_shrinkage_strength: float = Field(ge=0, default=20.0)     # 赔率收缩强度（等效先验样本量）
    kelly_use_semivariance: bool = True                  # 连续Kelly使用下行半方差
    kelly_skew_penalty_strength: float = Field(ge=0, default=0.5)     # 负偏度惩罚强度


class CapitalEfficiencyConfig(BaseModel):
    """资金效率监控配置（企业级增强版）"""
    idle_cash_threshold: float = Field(ge=0, le=1, default=0.15)
    efficiency_min: float = Field(ge=0, default=0.02)
    turnover_min: float = Field(ge=0, default=1.0)
    trend_window: int = Field(ge=1, default=5)
    deployment_min: float = Field(ge=0, le=1, default=0.30)
    # ── 企业级新增参数 ──
    idle_sweep_threshold: float = Field(ge=0.05, le=0.50, default=0.20)       # 闲置>20%触发归集
    idle_sweep_min_duration: int = Field(ge=5, default=30)                      # 闲置持续30分钟才归集
    utilization_reclaim_threshold: float = Field(ge=0.10, le=0.80, default=0.50)  # 利用率<50%回收
    min_capital_efficiency: float = Field(ge=0.05, le=0.50, default=0.15)       # 最低资金效率
    emergency_reserve_threshold: float = Field(ge=0.05, le=0.30, default=0.10)  # 风控池<10%触发补充
    max_concentration_ratio: float = Field(ge=0.30, le=0.80, default=0.50)      # 最大集中度
    volatility_pool_adapt: bool = True                                          # 波动率自适应池比例
    opportunity_cost_daily: float = Field(ge=0, default=0.0005)                 # 闲置资金机会成本日化率（0.05%）
    positive_return_protection: bool = True                                     # 正收益策略保护（低利用率不全额回收）


class CapitalAttritionConfig(BaseModel):
    """资金磨损分析器配置 — 量化交易中的资金磨损追踪与优化"""
    enabled: bool = True
    taker_fee_rate: float = Field(ge=0, default=0.0005)
    maker_fee_rate: float = Field(ge=0, default=0.0002)
    adaptive_fee_enabled: bool = True
    adaptive_fee_window: int = Field(ge=10, default=100)
    max_records: int = Field(ge=100, default=10000)
    budget_check_enabled: bool = True
    budget_auto_pause: bool = False
    default_daily_budget_usdt: float = Field(ge=0, default=5.0)
    default_weekly_budget_usdt: float = Field(ge=0, default=25.0)
    alert_cooldown_seconds: int = Field(ge=10, default=300)
    invalid_trade_threshold: float = Field(ge=0, le=1, default=0.5)
    funding_rate_alert_threshold: float = Field(ge=0, default=0.001)
    daily_reset_hour: int = Field(ge=0, le=23, default=0)
    suggestion_interval_seconds: int = Field(ge=60, default=3600)
    persist_path: str = ""
    strategy_budgets: Dict[str, Any] = Field(default_factory=dict)


class PaperTradingConfig(BaseModel):
    """P22-5: 模拟盘/沙盒交易环境配置"""
    enabled: bool = False
    mode: str = "sandbox"  # sandbox | testnet
    sandbox_id: str = "paper_main"
    initial_capital: float = Field(ge=10, default=10000.0)
    symbols: List[str] = Field(default_factory=list)
    strategies: List[str] = Field(default_factory=list)
    timeframes: List[str] = Field(default_factory=lambda: ["5m", "15m"])
    max_leverage: int = Field(ge=1, le=125, default=5)
    max_positions: int = Field(ge=1, default=8)
    risk_per_trade_pct: float = Field(ge=0, le=1, default=0.02)
    max_daily_loss_pct: float = Field(ge=0, le=1, default=0.10)
    max_drawdown_pct: float = Field(ge=0, le=1, default=0.25)
    enable_auto_trading: bool = True
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    stop_loss_atr_mult: float = Field(ge=0, default=2.0)
    take_profit_atr_mult: float = Field(ge=0, default=3.0)
    min_signal_strength: float = Field(ge=0, le=1, default=0.40)
    snapshot_interval_seconds: int = Field(ge=10, default=60)
    order_timeout_seconds: int = Field(ge=60, default=3600)
    persist_state: bool = True
    persist_dir: str = "./data/paper_trading"
    # P22-6: 延迟告警暂停开仓
    latency_pause_enabled: bool = True
    latency_warning_ms: int = Field(ge=500, default=3000)
    latency_critical_ms: int = Field(ge=500, default=8000)
    latency_pause_duration_sec: int = Field(ge=60, default=300)
    # P22-7: 限价优先
    limit_order_priority: bool = True
    limit_order_max_spread_pct: float = Field(ge=0, default=0.001)
    # P22-8: 合规红线
    compliance_check_enabled: bool = True
    compliance_check_interval_sec: int = Field(ge=300, default=3600)


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    system: SystemConfig
    okx: OKXConfig
    redis: RedisConfig
    sqlite: SQLiteConfig
    monitoring: MonitoringConfig
    hardware: HardwareConfig
    trading: TradingConfig
    currencies: CurrenciesConfig
    strategies: StrategiesConfig
    risk: RiskConfig
    execution: ExecutionConfig
    notifications: NotificationsConfig
    telegram: TelegramConfig
    market_data: MarketDataConfig = MarketDataConfig()
    allocation_agent: AllocationAgentConfig = AllocationAgentConfig()
    capital_pool: CapitalPoolConfig = CapitalPoolConfig()
    symbol_allocation: SymbolAllocationConfig = SymbolAllocationConfig()
    leverage_tiers: LeverageTiersConfig = LeverageTiersConfig()
    pnl_reallocation: PnLReallocationConfig = PnLReallocationConfig()
    hedge_scheduler: HedgeSchedulerConfig = HedgeSchedulerConfig()
    volatility_targeting: VolatilityTargetingConfig = VolatilityTargetingConfig()
    adaptive_kelly: AdaptiveKellyConfig = AdaptiveKellyConfig()
    capital_efficiency: CapitalEfficiencyConfig = CapitalEfficiencyConfig()
    paper_trading: PaperTradingConfig = PaperTradingConfig()
    review: Optional[Dict[str, Any]] = None
    data_cleanup: Optional[Dict[str, Any]] = None
    dashboard: Optional[Dict[str, Any]] = None
    signal_generation: SignalGenerationConfig = SignalGenerationConfig()
    anti_targeting: AntiTargetingConfig = AntiTargetingConfig()
    capital_attrition: CapitalAttritionConfig = CapitalAttritionConfig()
    # ── 企业级自适应学习系统各子模块配置（Dict 透传，由各模块自行解析） ──
    online_learner: Optional[Dict[str, Any]] = None
    parameter_adaptor: Optional[Dict[str, Any]] = None
    performance_feedback: Optional[Dict[str, Any]] = None
    knowledge_base: Optional[Dict[str, Any]] = None
    market_regime_detector: Optional[Dict[str, Any]] = None
    market_regime: Optional[Dict[str, Any]] = None
    meta_learner: Optional[Dict[str, Any]] = None
    strategy_evolver: Optional[Dict[str, Any]] = None


def _get_resource_path(relative_path: str) -> str:
    """PyInstaller兼容：获取打包后资源文件的绝对路径"""
    import sys as _sys
    if getattr(_sys, 'frozen', False):
        return os.path.join(_sys._MEIPASS, relative_path)
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


_ENV_OVERLAY_MAP = {
    "test": "config.test.yaml",
    "testing": "config.test.yaml",
    "dev": "config.test.yaml",
    "development": "config.test.yaml",
}


def _get_env_name() -> str:
    """当前运行环境名，由环境变量 OKX_ENV 指定，默认 prod。"""
    return (os.getenv("OKX_ENV") or "prod").strip().lower() or "prod"


def _resolve_config_path(filename: str) -> str:
    """解析配置文件绝对路径；打包后优先使用 exe 同目录的外部文件。"""
    import sys as _sys
    if getattr(_sys, "frozen", False):
        exe_dir = os.path.dirname(_sys.executable)
        external_path = os.path.join(exe_dir, filename)
        if os.path.exists(external_path):
            return external_path
    return _get_resource_path(filename)


def _load_yaml_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _deep_merge(base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并：overlay 中的字典递归覆盖 base，其余值直接覆盖。"""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> Dict[str, Any]:
    # base：主配置文件（生产环境）
    base_path = _resolve_config_path("config.yaml")
    config = _load_yaml_file(base_path)

    # overlay：测试等环境叠加配置（仅覆盖差异项，其余继承 base）
    env_name = _get_env_name()
    overlay_name = _ENV_OVERLAY_MAP.get(env_name)
    if overlay_name:
        overlay_path = _resolve_config_path(overlay_name)
        if os.path.exists(overlay_path):
            overlay = _load_yaml_file(overlay_path)
            config = _deep_merge(config, overlay)
            logger.info(f"配置环境: {env_name}（已叠加 {overlay_name}）")
        else:
            logger.warning(f"环境 {env_name} 请求叠加 {overlay_name}，但文件不存在，使用 base 配置")

    config = _resolve_env_vars(config)

    # 从环境变量中构建多API密钥池
    api_keys_list = []
    
    # 第一个密钥（默认配置中的）
    default_key = config.get("okx", {}).get("api_key", "")
    default_secret = config.get("okx", {}).get("secret_key", "")
    default_pass = config.get("okx", {}).get("passphrase", "")
    if default_key and default_secret and default_pass and not default_key.startswith("${"):
        api_keys_list.append({
            "api_key": default_key,
            "secret_key": default_secret,
            "passphrase": default_pass
        })
    
    # 从环境变量中读取编号的API密钥 (OKX_API_KEY_1, OKX_API_KEY_2, ...)
    for i in range(1, 20):
        env_key = f"OKX_API_KEY_{i}"
        env_secret = f"OKX_SECRET_KEY_{i}"
        env_pass = f"OKX_PASSPHRASE_{i}"
        
        api_key = os.getenv(env_key, "")
        secret_key = os.getenv(env_secret, "")
        passphrase = os.getenv(env_pass, "")
        
        if api_key and secret_key and passphrase:
            api_keys_list.append({
                "api_key": api_key,
                "secret_key": secret_key,
                "passphrase": passphrase
            })
    
    if api_keys_list:
        if "okx" not in config:
            config["okx"] = {}
        config["okx"]["api_keys"] = api_keys_list
        logger.info(f"Loaded {len(api_keys_list)} API keys from environment")

    try:
        validated = AppConfig(**config)
        validated_config = validated.model_dump(exclude_none=True)
        logger.info("Configuration loaded and validated successfully")
        _run_config_consistency_guard(validated_config)
        return validated_config
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        raise


def _get_config_baseline_path() -> str:
    """防回归基线文件路径；非生产环境使用独立文件，避免污染生产基线。"""
    env_name = _get_env_name()
    if env_name != "prod":
        return os.path.join("data", f".config_guard_baseline.{env_name}.json")
    return os.path.join("data", ".config_guard_baseline.json")


def _load_config_baseline() -> Optional[Dict[str, Any]]:
    """加载上次验证通过的防回归基线快照（精简、无敏感信息）。"""
    baseline_path = _get_config_baseline_path()
    try:
        with open(baseline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except FileNotFoundError:
        return None
    except Exception as e:
        logger.warning(f"Failed to load config guard baseline: {e}")
        return None


def _save_config_baseline(snapshot: Dict[str, Any], current: Optional[Dict[str, Any]] = None) -> None:
    """保存防回归基线快照；仅在内容变化时写入，避免无谓的磁盘 IO 与热更新误触发。"""
    if current is None:
        current = _load_config_baseline()
    if current == snapshot:
        return
    baseline_path = _get_config_baseline_path()
    try:
        os.makedirs(os.path.dirname(baseline_path) or ".", exist_ok=True)
        with open(baseline_path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save config guard baseline: {e}")


def _run_config_consistency_guard(config: Dict[str, Any]) -> None:
    """运行配置一致性 + 防回归校验（不阻断加载，仅记录错误/告警）。"""
    try:
        from core.config_consistency_guard import ConfigConsistencyGuard

        guard = ConfigConsistencyGuard()
        report = guard.check_consistency(config)
        baseline = _load_config_baseline()
        if baseline:
            report.merge(guard.check_regression(config, baseline=baseline))

        for err in report.errors:
            logger.error(f"ConfigConsistencyGuard: {err}")
        for warn in report.warnings:
            logger.warning(f"ConfigConsistencyGuard: {warn}")

        _save_config_baseline(guard.extract_baseline(config), current=baseline)
    except Exception as e:
        # 守卫失败不应阻断配置加载
        logger.error(f"ConfigConsistencyGuard error: {e}")


def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = _resolve_env_vars(value)
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            resolved = os.getenv(env_var)
            if resolved is None:
                logger.warning(f"Environment variable {env_var} not set, using placeholder")
                result[key] = value
            else:
                result[key] = resolved
        else:
            result[key] = value
    return result

def get_currency_tier(symbol: str, config: Dict[str, Any]) -> str:
    tier1 = config["currencies"]["tier1_symbols"]
    tier2 = config["currencies"]["tier2_symbols"]
    tier3 = config["currencies"]["tier3_symbols"]
    
    base_symbol = symbol.replace("-USDT", "").replace("USDT-", "").replace("-SWAP", "")
    
    if base_symbol in tier1:
        return "tier1"
    elif base_symbol in tier2:
        return "tier2"
    elif base_symbol in tier3:
        return "tier3"
    return "tier3"

def get_symbol_config(symbol: str, config: Dict[str, Any]) -> Dict[str, Any]:
    tier = get_currency_tier(symbol, config)
    return config["currencies"][f"{tier}_settings"]

def get_all_symbols(config: Dict[str, Any]) -> List[str]:
    symbols = []
    for tier in ["tier1", "tier2", "tier3"]:
        for base in config["currencies"][f"{tier}_symbols"]:
            symbols.append(f"{base}-USDT-SWAP")
    return symbols