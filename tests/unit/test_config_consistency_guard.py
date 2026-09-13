"""ConfigConsistencyGuard 配置一致性 + 防回归校验单元测试。"""
import pytest

from core.config_consistency_guard import ConfigConsistencyGuard, ConfigGuardReport


def _valid_config() -> dict:
    return {
        "trading": {
            "total_capital": 1000.0,
            "trading_capital_ratio": 0.80,
            "risk_reserve_ratio": 0.15,
            "profit_reserve_ratio": 0.05,
            "scalping_allocation": 0.30,
            "trend_allocation": 0.20,
            "grid_allocation": 0.20,
            "arbitrage_allocation": 0.15,
            "spot_grid_allocation": 0.10,
            "spot_martingale_allocation": 0.05,
            "max_drawdown": 0.25,
            "daily_max_loss": 0.10,
            "hourly_max_loss": 0.05,
            "daily_risk_limit": 0.10,
            "risk_per_trade": 0.02,
            "max_total_leverage": 20,
            "max_stop_loss_pct": 0.05,
            "max_slippage_pct": 0.003,
        },
        "monitoring": {
            "api_latency_warning_ms": 1500,
            "api_latency_critical_ms": 5000,
        },
        "currencies": {
            "tier1_settings": {"leverage_min": 1, "leverage_max": 20},
            "tier2_settings": {"leverage_min": 1, "leverage_max": 10},
            "tier3_settings": {"leverage_min": 1, "leverage_max": 5},
        },
        "strategies": {
            "grid": {"min_signal_quality": 0.40, "stop_loss_pct": 0.035, "max_stop_loss_pct": 0.04},
            "trend": {"min_signal_quality": 0.20, "max_stop_loss_pct": 0.025},
            "scalping": {"min_signal_quality": 0.10, "stop_loss": 0.003},
        },
    }


# ============================================================
# 一致性校验
# ============================================================

def test_valid_config_has_no_errors():
    report = ConfigConsistencyGuard().check_consistency(_valid_config())
    assert report.valid is True
    assert report.errors == []


def test_allocation_sum_mismatch_is_error():
    config = _valid_config()
    config["trading"]["scalping_allocation"] = 0.50  # 总和变成 1.20
    report = ConfigConsistencyGuard().check_consistency(config)
    assert report.valid is False
    assert any("策略分配总和" in e for e in report.errors)


def test_capital_ratio_sum_mismatch_is_error():
    config = _valid_config()
    config["trading"]["profit_reserve_ratio"] = 0.50  # 总和变成 1.45
    report = ConfigConsistencyGuard().check_consistency(config)
    assert report.valid is False
    assert any("资本比例总和" in e for e in report.errors)


def test_daily_max_loss_exceeds_drawdown_is_error():
    config = _valid_config()
    config["trading"]["daily_max_loss"] = 0.30
    config["trading"]["max_drawdown"] = 0.20
    report = ConfigConsistencyGuard().check_consistency(config)
    assert report.valid is False
    assert any("daily_max_loss" in e for e in report.errors)


def test_latency_threshold_order_is_error():
    config = _valid_config()
    config["monitoring"]["api_latency_warning_ms"] = 6000
    config["monitoring"]["api_latency_critical_ms"] = 5000
    report = ConfigConsistencyGuard().check_consistency(config)
    assert report.valid is False
    assert any("api_latency_warning_ms" in e for e in report.errors)


def test_tier_min_greater_than_max_is_error():
    config = _valid_config()
    config["currencies"]["tier1_settings"]["leverage_min"] = 30
    config["currencies"]["tier1_settings"]["leverage_max"] = 20
    report = ConfigConsistencyGuard().check_consistency(config)
    assert report.valid is False
    assert any("leverage_min" in e for e in report.errors)


def test_non_dict_config_is_error_not_crash():
    report = ConfigConsistencyGuard().check_consistency(None)
    assert report.valid is False


# ============================================================
# 防回归校验
# ============================================================

def test_max_drawdown_widening_warns():
    baseline = _valid_config()
    new = _valid_config()
    new["trading"]["max_drawdown"] = 0.30  # 相对 0.25 上升 20%
    report = ConfigConsistencyGuard().check_regression(new, baseline=baseline)
    assert any("max_drawdown" in w for w in report.warnings)


def test_risk_per_trade_widening_warns():
    baseline = _valid_config()
    new = _valid_config()
    new["trading"]["risk_per_trade"] = 0.05  # 相对 0.02 上升 150%
    report = ConfigConsistencyGuard().check_regression(new, baseline=baseline)
    assert any("risk_per_trade" in w for w in report.warnings)


def test_min_signal_quality_lowering_warns():
    baseline = _valid_config()
    new = _valid_config()
    new["strategies"]["grid"]["min_signal_quality"] = 0.20  # 相对 0.40 下降 50%
    report = ConfigConsistencyGuard().check_regression(new, baseline=baseline)
    assert any("min_signal_quality" in w for w in report.warnings)


def test_stop_loss_widening_warns():
    baseline = _valid_config()
    new = _valid_config()
    new["strategies"]["scalping"]["stop_loss"] = 0.006  # 相对 0.003 上升 100%
    report = ConfigConsistencyGuard().check_regression(new, baseline=baseline)
    assert any("stop_loss" in w for w in report.warnings)


def test_safe_tightening_does_not_warn():
    baseline = _valid_config()
    new = _valid_config()
    new["trading"]["max_drawdown"] = 0.20  # 收紧
    new["strategies"]["grid"]["min_signal_quality"] = 0.50  # 提高门槛
    report = ConfigConsistencyGuard().check_regression(new, baseline=baseline)
    assert report.warnings == []


def test_empty_baseline_no_crash():
    report = ConfigConsistencyGuard().check_regression(_valid_config(), baseline={})
    assert report.valid is True
    assert report.warnings == []


def test_missing_fields_no_crash():
    report = ConfigConsistencyGuard().check_regression({}, baseline=_valid_config())
    assert report.valid is True


# ============================================================
# 基线快照提取
# ============================================================

def test_extract_baseline_is_secret_free():
    config = _valid_config()
    config["okx"] = {"api_key": "SECRET", "secret_key": "SECRET", "passphrase": "SECRET"}
    snapshot = ConfigConsistencyGuard().extract_baseline(config)
    assert "okx" not in snapshot
    assert "trading" in snapshot
    assert "strategies" in snapshot
    # 只保留安全关键字段，不含无关字段
    assert "total_capital" not in snapshot["trading"]
    assert "max_drawdown" in snapshot["trading"]
    assert "min_signal_quality" in snapshot["strategies"]["grid"]


def test_validate_combines_consistency_and_regression():
    baseline = _valid_config()
    new = _valid_config()
    new["trading"]["max_drawdown"] = 0.30
    new["trading"]["scalping_allocation"] = 0.50  # 触发一致性 error
    report = ConfigConsistencyGuard().validate(new, baseline=baseline)
    assert report.valid is False
    assert any("max_drawdown" in w for w in report.warnings)
    assert any("策略分配总和" in e for e in report.errors)
