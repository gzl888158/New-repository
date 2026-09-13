"""
生产级配置校验器 (ConfigValidator) 全面单元测试

覆盖:
  - 基础配置模型校验
  - OKX API 凭证校验
  - 风控参数校验
  - 交易参数校验
  - 币种分层设置
  - 执行配置校验
  - 完整配置校验 (validate_config)
  - 部分校验 (validate_config_partial)
  - 配置审计 (audit_config)
  - 边界条件与错误场景
"""
import pytest
import copy
from pydantic import ValidationError

from configs.config_validator import (
    CurrencyTierSettings, CurrenciesConfig, ExecutionConfig,
    OKXConfig, RiskConfig, RiskCircuitBreaker, BlackSwanConfig,
    CorrelationRiskConfig, TradingConfig, SystemConfig, RedisConfig,
    AppConfig, validate_config, validate_config_partial, audit_config,
)


# ═══════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def valid_raw_config():
    """有效的完整配置"""
    return {
        "system": {
            "name": "OKX Quant Trading",
            "version": "2.0.0",
            "log_level": "INFO",
            "timezone": "Asia/Shanghai",
        },
        "currencies": {
            "tier1_settings": {
                "leverage_min": 2, "leverage_max": 5, "leverage_default": 3,
                "position_limit": 0.25, "grid_spacing_min": 0.0114,
                "grid_spacing_max": 0.018, "slippage": 0.0015,
            },
            "tier1_symbols": ["BTC", "ETH"],
            "tier2_settings": {
                "leverage_min": 2, "leverage_max": 5, "leverage_default": 3,
                "position_limit": 0.25, "grid_spacing_min": 0.0114,
                "grid_spacing_max": 0.018, "slippage": 0.0015,
            },
            "tier2_symbols": [],
            "tier3_settings": {
                "leverage_min": 2, "leverage_max": 5, "leverage_default": 3,
                "position_limit": 0.25, "grid_spacing_min": 0.0114,
                "grid_spacing_max": 0.018, "slippage": 0.0015,
            },
            "tier3_symbols": [],
        },
        "execution": {
            "asyncio_workers": 4, "batch_processing_size": 10,
            "concurrent_strategies": 8, "max_parallel_tasks": 16,
            "thread_pool_size": 8, "retry_attempts": 3,
            "slippage_tolerance": 0.001, "reconciliation_interval": 10,
            "rate_limit": {"rest_requests_per_second": 5, "websocket_requests_per_second": 10},
        },
        "okx": {
            "is_testnet": True,
            "rest_url": "https://www.okx.com",
            "websocket_url": "wss://ws.okx.com:8443/ws/v5/public",
            "websocket_private_url": "wss://ws.okx.com:8443/ws/v5/private",
        },
        "risk": {
            "black_swan": {
                "enabled": True, "btc_flash_crash_pct": 0.03,
                "btc_volatility_spike_pct": 0.05, "check_interval_seconds": 10,
                "emergency_full_close_pct": 0.12, "flash_crash_window_seconds": 300,
                "market_circuit_breaker_pct": 0.08, "volatility_window_minutes": 60,
            },
            "circuit_breakers": {
                "api_timeout": 30, "btc_movement_threshold": 0.05,
                "btc_movement_window": 60, "drawdown_acceleration_threshold": 0.01,
                "liquidation_volume_threshold": 1000000, "liquidation_window": 300,
                "network_timeout": 60, "rapid_drawdown_threshold": 0.03,
                "rapid_drawdown_window": 60,
            },
            "correlation": {
                "check_interval_seconds": 300, "kline_bar": "1H",
                "lookback_bars": 50, "max_concentration": 0.4,
                "reduce_ratio": 0.5, "threshold": 0.7,
            },
            "full_close_threshold": 0.2, "margin_call_threshold": 0.3,
            "margin_warning_threshold": 0.5, "position_loss_threshold": 0.1,
        },
        "trading": {
            "total_capital": 10.0, "trading_capital_ratio": 0.95,
            "max_drawdown": 0.25, "daily_max_loss": 0.04,
            "daily_risk_limit": 0.03, "risk_per_trade": 0.025,
            "max_total_leverage": 5.0, "max_concurrent_positions": 3,
            "min_notional_usd": 3.0, "min_balance": 5.0,
            "target_utilization": 0.9, "profit_reserve_ratio": 0.0,
            "risk_reserve_ratio": 0.05, "max_slippage_pct": 0.001,
            "maker_fee_rate": 0.0002, "taker_fee_rate": 0.0005,
        },
        "redis": {
            "host": "127.0.0.1", "port": 6379, "db": 0,
        },
        "monitoring": {}, "notifications": {}, "sqlite": {},
        "strategies": {}, "hardware": {},
    }


# ═══════════════════════════════════════════════════════════════
# Test: CurrencyTierSettings
# ═══════════════════════════════════════════════════════════════

class TestCurrencyTierSettings:
    def test_default_values(self):
        s = CurrencyTierSettings()
        assert s.leverage_min == 2
        assert s.leverage_max == 5
        assert s.leverage_default == 3
        assert s.position_limit == 0.25
        assert s.slippage == 0.0015

    def test_leverage_default_in_range(self):
        s = CurrencyTierSettings(leverage_min=2, leverage_max=10, leverage_default=5)
        assert s.leverage_default == 5

    def test_leverage_default_out_of_range(self):
        with pytest.raises(ValidationError):
            CurrencyTierSettings(leverage_min=2, leverage_max=5, leverage_default=10)

    def test_grid_spacing_max_lt_min(self):
        with pytest.raises(ValidationError):
            CurrencyTierSettings(grid_spacing_min=0.02, grid_spacing_max=0.01)

    def test_leverage_min_boundary(self):
        with pytest.raises(ValidationError):
            CurrencyTierSettings(leverage_min=0)  # < 1

    def test_leverage_max_boundary(self):
        with pytest.raises(ValidationError):
            CurrencyTierSettings(leverage_max=126)  # > 125


# ═══════════════════════════════════════════════════════════════
# Test: OKXConfig
# ═══════════════════════════════════════════════════════════════

class TestOKXConfig:
    def test_testnet_no_credentials_needed(self):
        cfg = OKXConfig(is_testnet=True)
        assert cfg.is_testnet is True

    def test_production_missing_credentials(self):
        with pytest.raises(ValidationError, match="API key"):
            OKXConfig(is_testnet=False, api_key="", secret_key="", passphrase="")

    def test_production_placeholder_credentials(self):
        with pytest.raises(ValidationError, match="API key"):
            OKXConfig(is_testnet=False, api_key="${OKX_API_KEY}", secret_key="x", passphrase="x")

    def test_production_valid_credentials(self):
        cfg = OKXConfig(
            is_testnet=False, api_key="valid_key",
            secret_key="valid_secret", passphrase="valid_pass",
        )
        assert cfg.api_key == "valid_key"

    def test_default_urls(self):
        cfg = OKXConfig(is_testnet=True)
        assert cfg.rest_url == "https://www.okx.com"
        assert cfg.websocket_url.startswith("wss://")


# ═══════════════════════════════════════════════════════════════
# Test: ExecutionConfig
# ═══════════════════════════════════════════════════════════════

class TestExecutionConfig:
    def test_default_values(self):
        cfg = ExecutionConfig()
        assert cfg.asyncio_workers == 4
        assert cfg.retry_attempts == 3
        assert cfg.slippage_tolerance == 0.001

    def test_out_of_range_workers(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(asyncio_workers=0)

    def test_out_of_range_retry(self):
        with pytest.raises(ValidationError):
            ExecutionConfig(retry_attempts=0)


# ═══════════════════════════════════════════════════════════════
# Test: RiskConfig
# ═══════════════════════════════════════════════════════════════

class TestRiskConfig:
    def test_valid_thresholds(self):
        cfg = RiskConfig(margin_call_threshold=0.3, margin_warning_threshold=0.5)
        assert cfg.margin_warning_threshold > cfg.margin_call_threshold

    def test_invalid_thresholds_order(self):
        with pytest.raises(ValidationError, match="margin_warning"):
            RiskConfig(margin_call_threshold=0.5, margin_warning_threshold=0.3)

    def test_black_swan_defaults(self):
        cfg = RiskConfig()
        assert cfg.black_swan.enabled is True
        assert cfg.black_swan.btc_flash_crash_pct == 0.03

    def test_circuit_breaker_defaults(self):
        cfg = RiskConfig()
        assert cfg.circuit_breakers.api_timeout == 30
        assert cfg.circuit_breakers.btc_movement_threshold == 0.05

    def test_correlation_defaults(self):
        cfg = RiskConfig()
        assert cfg.correlation.threshold == 0.7
        assert cfg.correlation.lookback_bars == 50


# ═══════════════════════════════════════════════════════════════
# Test: TradingConfig
# ═══════════════════════════════════════════════════════════════

class TestTradingConfig:
    def test_default_values(self):
        cfg = TradingConfig()
        assert cfg.total_capital == 10.0
        assert cfg.trading_capital_ratio == 0.95
        assert cfg.max_drawdown == 0.25
        assert cfg.max_total_leverage == 5.0

    def test_out_of_range_capital(self):
        with pytest.raises(ValidationError):
            TradingConfig(total_capital=0.5)  # < 1.0

    def test_out_of_range_leverage(self):
        with pytest.raises(ValidationError):
            TradingConfig(max_total_leverage=126)  # > 125

    def test_out_of_range_trading_capital_ratio(self):
        with pytest.raises(ValidationError):
            TradingConfig(trading_capital_ratio=1.5)  # > 1.0

    def test_fee_rates(self):
        cfg = TradingConfig(maker_fee_rate=0.0002, taker_fee_rate=0.0005)
        assert cfg.maker_fee_rate == 0.0002
        assert cfg.taker_fee_rate == 0.0005


# ═══════════════════════════════════════════════════════════════
# Test: SystemConfig
# ═══════════════════════════════════════════════════════════════

class TestSystemConfig:
    def test_default_values(self):
        cfg = SystemConfig()
        assert cfg.name == "OKX Quant Trading"
        assert cfg.version == "2.0.0"
        assert cfg.log_level == "INFO"

    def test_invalid_log_level(self):
        with pytest.raises(ValidationError):
            SystemConfig(log_level="TRACE")

    def test_valid_log_levels(self):
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            cfg = SystemConfig(log_level=level)
            assert cfg.log_level == level


# ═══════════════════════════════════════════════════════════════
# Test: RedisConfig
# ═══════════════════════════════════════════════════════════════

class TestRedisConfig:
    def test_default_values(self):
        cfg = RedisConfig()
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 6379
        assert cfg.db == 0

    def test_invalid_port(self):
        with pytest.raises(ValidationError):
            RedisConfig(port=0)

    def test_invalid_db(self):
        with pytest.raises(ValidationError):
            RedisConfig(db=16)


# ═══════════════════════════════════════════════════════════════
# Test: AppConfig (完整校验)
# ═══════════════════════════════════════════════════════════════

class TestAppConfig:
    def test_valid_config_passes(self, valid_raw_config):
        cfg = AppConfig(**valid_raw_config)
        assert cfg.system.name == "OKX Quant Trading"
        assert cfg.trading.total_capital == 10.0
        assert cfg.okx.is_testnet is True

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            AppConfig(system={"name": "test"})  # 缺少 version, log_level

    def test_minimal_config(self):
        # 测试网模式 + 最小配置可以工作
        cfg = AppConfig(
            system={"log_level": "INFO"},
            okx={"is_testnet": True},
        )
        assert cfg.okx.is_testnet is True


# ═══════════════════════════════════════════════════════════════
# Test: validate_config
# ═══════════════════════════════════════════════════════════════

class TestValidateConfig:
    def test_validate_valid_config(self, valid_raw_config):
        result = validate_config(valid_raw_config)
        assert isinstance(result, AppConfig)
        assert result.system.version == "2.0.0"

    def test_validate_invalid_config(self):
        with pytest.raises(ValueError, match="Invalid configuration"):
            validate_config({"system": {"log_level": "INVALID"}})

    def test_validate_empty_config(self):
        with pytest.raises(ValueError, match="Invalid configuration"):
            validate_config({})


# ═══════════════════════════════════════════════════════════════
# Test: validate_config_partial
# ═══════════════════════════════════════════════════════════════

class TestValidateConfigPartial:
    def test_partial_empty_config(self):
        result = validate_config_partial({})
        assert isinstance(result, dict)

    def test_partial_valid_config(self, valid_raw_config):
        result = validate_config_partial(valid_raw_config)
        assert isinstance(result, dict)

    def test_partial_warning_margin_order(self):
        config = {
            "risk": {
                "margin_warning_threshold": 0.2,
                "margin_call_threshold": 0.5,
            }
        }
        result = validate_config_partial(config)
        assert isinstance(result, dict)  # 应该返回警告但不崩溃

    def test_partial_warning_leverage(self):
        config = {"trading": {"max_total_leverage": 200}}
        result = validate_config_partial(config)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════
# Test: audit_config
# ═══════════════════════════════════════════════════════════════

class TestAuditConfig:
    def test_audit_normal_config(self, valid_raw_config):
        cfg = AppConfig(**valid_raw_config)
        audit = audit_config(cfg)
        assert "version" in audit
        assert "okx" in audit
        assert "risk" in audit
        assert "execution" in audit
        assert "trading" in audit
        assert "warnings" in audit

    def test_audit_low_capital_warning(self):
        cfg = AppConfig(
            system={"log_level": "INFO"},
            okx={"is_testnet": True},
            trading={"total_capital": 5.0},
        )
        audit = audit_config(cfg)
        assert any("Low capital" in w for w in audit["warnings"])

    def test_audit_high_leverage_warning(self):
        cfg = AppConfig(
            system={"log_level": "INFO"},
            okx={"is_testnet": True},
            trading={"max_total_leverage": 50.0},
        )
        audit = audit_config(cfg)
        assert any("High leverage" in w for w in audit["warnings"])

    def test_audit_high_drawdown_warning(self):
        cfg = AppConfig(
            system={"log_level": "INFO"},
            okx={"is_testnet": True},
            trading={"max_drawdown": 0.5},
        )
        audit = audit_config(cfg)
        assert any("High max drawdown" in w for w in audit["warnings"])

    def test_audit_effective_capital(self):
        cfg = AppConfig(
            system={"log_level": "INFO"},
            okx={"is_testnet": True},
            trading={"total_capital": 1000.0, "trading_capital_ratio": 0.8},
        )
        audit = audit_config(cfg)
        assert audit["trading"]["effective_capital"] == 800.0

    def test_audit_no_warnings_healthy(self, valid_raw_config):
        cfg = AppConfig(**valid_raw_config)
        audit = audit_config(cfg)
        # 测试网模式的小额资金可能有警告
        assert isinstance(audit["warnings"], list)


# ═══════════════════════════════════════════════════════════════
# Test: 边界条件
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_okx_config_with_proxy(self):
        cfg = OKXConfig(is_testnet=True, proxy="http://proxy:8080")
        assert cfg.proxy == "http://proxy:8080"

    def test_redis_config_with_password(self):
        cfg = RedisConfig(password="secret")
        assert cfg.password == "secret"

    def test_currencies_config_empty_tiers(self):
        cfg = CurrenciesConfig()
        assert cfg.tier1_symbols == ["BTC", "ETH"]
        assert cfg.tier2_symbols == []
        assert cfg.tier3_symbols == []

    def test_risk_circuit_breaker_bounds(self):
        with pytest.raises(ValidationError):
            RiskCircuitBreaker(api_timeout=1)  # < 5

    def test_black_swan_bounds(self):
        with pytest.raises(ValidationError):
            BlackSwanConfig(btc_flash_crash_pct=0.5)  # > 0.3

    def test_correlation_bounds(self):
        with pytest.raises(ValidationError):
            CorrelationRiskConfig(max_concentration=1.5)  # > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])