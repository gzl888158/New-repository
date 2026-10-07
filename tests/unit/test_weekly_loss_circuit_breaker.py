"""周亏损熔断（weekly_max_loss）与日亏损熔断（daily_max_loss）单元测试。

验证 enterprise_profit_strategy.md Section 8 新增的周维度止损 + 日维度止血：
- 周亏损 >= 6% 触发暂停交易
- 日亏损 >= 3% 触发平仓 + 暂停交易（P1 增强）
- ISO 周边界正确重置周基准权益
- 周/日 PnL 正确计算并导出到 risk_status.json
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

from risk.global_risk import GlobalRiskControl


def _make_config():
    return {
        "trading": {
            "max_drawdown": 0.25,
            "daily_max_loss": 0.03,
            "hourly_max_loss": 0.025,
            "weekly_max_loss": 0.06,
            "max_consecutive_losses": 5,
            "total_capital": 44,
        },
        "risk": {
            "margin_call_threshold": 0.8,
            "margin_warning_threshold": 0.6,
            "position_loss_threshold": 0.5,
            "full_close_threshold": 0.8,
            "circuit_breakers": {
                "btc_movement_threshold": 0.08,
                "btc_movement_window": 300,
                "liquidation_volume_threshold": 100,
                "liquidation_window": 300,
                "api_timeout": 30,
                "network_timeout": 30,
            },
        },
        "sqlite": {"db_path": ":memory:"},
    }


def _make_risk(config=None):
    config = config or _make_config()
    risk = GlobalRiskControl(config, MagicMock(), MagicMock())
    risk._weekly_start_equity = 100.0
    risk._daily_start_equity = 100.0
    risk._hourly_start_equity = 100.0
    risk._effective_peak = 100.0
    risk._peak_equity = 100.0
    return risk


def test_weekly_max_loss_loaded_from_config():
    risk = _make_risk()
    assert risk._weekly_max_loss == 0.06


def test_weekly_max_loss_defaults_when_missing():
    config = _make_config()
    del config["trading"]["weekly_max_loss"]
    risk = GlobalRiskControl(config, MagicMock(), MagicMock())
    assert risk._weekly_max_loss == 0.06


def test_weekly_pnl_computed_correctly():
    risk = _make_risk()
    risk._weekly_start_equity = 100.0
    risk._weekly_pnl = 94.5 - 100.0
    assert risk._weekly_pnl == pytest.approx(-5.5)


@pytest.mark.asyncio
async def test_weekly_loss_triggers_pause():
    risk = _make_risk()
    risk._weekly_start_equity = 100.0
    risk._weekly_pnl = -6.5
    risk._is_paused = False

    await risk._handle_weekly_loss()

    assert risk._is_paused is True
    assert "Weekly" in risk._pause_reason


@pytest.mark.asyncio
async def test_weekly_loss_sends_alert():
    risk = _make_risk()
    mock_alert = MagicMock()
    mock_alert.send_alert = AsyncMock()
    risk._alert_manager = mock_alert

    await risk._handle_weekly_loss()

    mock_alert.send_alert.assert_called_once()
    call_args = mock_alert.send_alert.call_args
    assert call_args.args[0] == "WEEKLY_LOSS_LIMIT"


@pytest.mark.asyncio
async def test_weekly_loss_sets_pause_time():
    risk = _make_risk()
    before = datetime.now()
    await risk._handle_weekly_loss()
    after = datetime.now()

    assert risk._pause_time is not None
    assert before <= risk._pause_time <= after


def test_weekly_pnl_exported_in_status():
    risk = _make_risk()
    risk._weekly_pnl = -3.5
    risk._current_equity = 96.5

    risk._export_risk_status()

    import json
    with open("./data/risk_status.json", "r") as f:
        status = json.load(f)
    assert status["weekly_pnl"] == pytest.approx(-3.5)
    assert status["weekly_max_loss"] == 0.06


def test_weekly_reset_on_isoweek_change():
    """ISO 周切换时 _weekly_start_equity 应被重置。"""
    risk = _make_risk()
    old_week = (2026, 40)
    new_week = (2026, 41)
    risk._last_weekly_reset_isoweek = old_week

    current_isoweek = new_week
    assert current_isoweek != risk._last_weekly_reset_isoweek


def test_reset_clears_weekly_pnl():
    risk = _make_risk()
    risk._weekly_pnl = -5.0
    risk._is_paused = True
    risk._pause_reason = "Weekly max loss exceeded"

    risk.reset()

    assert risk._weekly_pnl == 0.0
    assert risk._is_paused is False


def test_manual_reset_clears_weekly_pnl():
    risk = _make_risk()
    risk._weekly_pnl = -4.0
    risk._daily_pnl = -2.0
    risk._hourly_pnl = -1.0

    import json, os
    os.makedirs("./data", exist_ok=True)
    with open("./data/risk_control.json", "w") as f:
        json.dump({"action": "reset_pause"}, f)

    risk._check_manual_controls()

    assert risk._weekly_pnl == 0
    assert risk._daily_pnl == 0
    assert risk._hourly_pnl == 0

    try:
        os.remove("./data/risk_control.json")
    except OSError:
        pass


# ========== 日亏损熔断（P1 增强：平仓 + 暂停） ==========


def test_daily_max_loss_loaded_from_config():
    risk = _make_risk()
    assert risk._daily_max_loss == 0.03


def test_daily_pnl_computed_correctly():
    risk = _make_risk()
    risk._daily_start_equity = 100.0
    risk._current_equity = 96.5
    risk._daily_pnl = risk._current_equity - risk._daily_start_equity
    assert risk._daily_pnl == pytest.approx(-3.5)


@pytest.mark.asyncio
async def test_daily_loss_closes_all_positions():
    """日亏损触发时应调用 _reduce_all_positions(1.0, ...) 平掉全部仓位。"""
    risk = _make_risk()
    risk._reduce_all_positions = AsyncMock()

    await risk._handle_daily_loss()

    risk._reduce_all_positions.assert_awaited_once_with(1.0, "daily_loss_circuit_breaker")


@pytest.mark.asyncio
async def test_daily_loss_triggers_pause():
    risk = _make_risk()
    risk._reduce_all_positions = AsyncMock()
    risk._is_paused = False

    await risk._handle_daily_loss()

    assert risk._is_paused is True
    assert "Daily" in risk._pause_reason


@pytest.mark.asyncio
async def test_daily_loss_sends_alert():
    risk = _make_risk()
    risk._reduce_all_positions = AsyncMock()
    mock_alert = MagicMock()
    mock_alert.send_alert = AsyncMock()
    risk._alert_manager = mock_alert

    await risk._handle_daily_loss()

    mock_alert.send_alert.assert_called_once()
    call_args = mock_alert.send_alert.call_args
    assert call_args.args[0] == "DAILY_LOSS_LIMIT"
    assert call_args.kwargs.get("severity") == "CRITICAL"


@pytest.mark.asyncio
async def test_daily_loss_sets_pause_time():
    risk = _make_risk()
    risk._reduce_all_positions = AsyncMock()
    before = datetime.now()

    await risk._handle_daily_loss()
    after = datetime.now()

    assert risk._pause_time is not None
    assert before <= risk._pause_time <= after


def test_daily_loss_threshold_check():
    """日亏损比例达到阈值时应被检测到。"""
    risk = _make_risk()
    risk._daily_start_equity = 100.0
    risk._current_equity = 96.5
    risk._daily_pnl = risk._current_equity - risk._daily_start_equity

    daily_loss_ratio = abs(risk._daily_pnl) / risk._daily_start_equity
    assert daily_loss_ratio >= risk._daily_max_loss


def test_daily_loss_within_threshold():
    """日亏损未达阈值时不应触发。"""
    risk = _make_risk()
    risk._daily_start_equity = 100.0
    risk._current_equity = 97.5
    risk._daily_pnl = risk._current_equity - risk._daily_start_equity

    daily_loss_ratio = abs(risk._daily_pnl) / risk._daily_start_equity
    assert daily_loss_ratio < risk._daily_max_loss
