"""EquityMonitor 自适应强化 — 外部资金变动（充值/提现/划转）识别单元测试。

覆盖修复点：
  1. 提现检测后重建回撤基准（peak_equity 重置为提现后值，max_drawdown_pct=0）
  2. 提现不触发 EMERGENCY（外部资金变动与交易回撤分治）
  3. 提现期间恰逢 EMERGENCY，重置后退出 NORMAL 并解除开仓冻结
  4. 外部资金变动事件持久化 + 重启自愈（_load_state 识别资金规模变化而非交易回撤）
  5. equity_monitor 配置节读取 + 未配置时回落默认值
"""

import json
from datetime import datetime

import pytest

from core.equity_monitor import EquityMonitor, EquityEventType, EquityMode


# ═══════════════════════════════════════════════════════════════
# 1. 提现检测后重建回撤基准
# ═══════════════════════════════════════════════════════════════

def test_withdrawal_resets_baseline():
    m = EquityMonitor(None)
    # 建立历史基准：权益 235
    m.feed(equity=235.0, available=200.0, margin=30.0, upl=5.0)
    assert m._peak_equity == pytest.approx(235.0)

    # 提现：权益 235 → 173，未实现盈亏不变（非交易亏损）
    event = m.feed(equity=173.0, available=140.0, margin=30.0, upl=5.0)

    assert event is not None
    assert event.event_type == EquityEventType.WITHDRAWAL
    # 基准重建为提现后值
    assert m._peak_equity == pytest.approx(173.0)
    assert m._max_drawdown_pct == 0.0
    assert m._trough_equity == pytest.approx(173.0)
    # 记录了外部资金变动事件
    assert len(m._recent_external_flows) == 1
    assert m._recent_external_flows[0]["type"] == "withdrawal"
    assert m._baseline_equity == pytest.approx(173.0)
    assert m._baseline_updated_at is not None


def test_deposit_resets_baseline():
    m = EquityMonitor(None)
    m.feed(equity=100.0, available=80.0, margin=20.0, upl=0.0)

    # 充值：权益 100 → 150
    event = m.feed(equity=150.0, available=130.0, margin=20.0, upl=0.0)

    assert event is not None
    assert event.event_type == EquityEventType.DEPOSIT
    assert m._peak_equity == pytest.approx(150.0)
    assert m._max_drawdown_pct == 0.0


# ═══════════════════════════════════════════════════════════════
# 2. 提现不触发 EMERGENCY
# ═══════════════════════════════════════════════════════════════

def test_withdrawal_does_not_trigger_emergency():
    m = EquityMonitor(None)
    m.feed(equity=235.0, available=200.0, margin=30.0, upl=5.0)

    # 提现幅度 26% > emergency_drop_pct(10%)，但应被识别为提现而非交易回撤
    event = m.feed(equity=173.0, available=140.0, margin=30.0, upl=5.0)

    assert event.event_type == EquityEventType.WITHDRAWAL
    assert m._current_mode == EquityMode.NORMAL
    assert m.is_new_position_allowed() is True
    params = m.get_adaptive_params()
    assert params["position_multiplier"] == 1.0
    assert params["max_positions"] == 10


# ═══════════════════════════════════════════════════════════════
# 3. 提现期间恰逢 EMERGENCY，重置后退出 NORMAL 并解除冻结
# ═══════════════════════════════════════════════════════════════

def test_reset_baseline_exits_emergency():
    m = EquityMonitor(None)
    # 模拟已处于紧急冻结状态
    m._current_mode = EquityMode.EMERGENCY
    m._adaptive_params["position_multiplier"] = 0.0
    m._adaptive_params["max_positions"] = 0
    m._adaptive_params["mode"] = "emergency"

    m._reset_baseline(173.0, "withdrawal")

    assert m._current_mode == EquityMode.NORMAL
    assert m._peak_equity == pytest.approx(173.0)
    assert m._max_drawdown_pct == 0.0
    assert m.is_new_position_allowed() is True
    params = m.get_adaptive_params()
    assert params["position_multiplier"] == 1.0
    assert params["max_positions"] == 10
    assert params["mode"] == "normal"


def test_withdrawal_during_emergency_exits_via_feed():
    m = EquityMonitor(None)
    # 建立基准并置为紧急模式（模拟此前真实交易回撤导致的冻结）
    m.feed(equity=235.0, available=200.0, margin=30.0, upl=5.0)
    m._current_mode = EquityMode.EMERGENCY
    m._adaptive_params["position_multiplier"] = 0.0

    event = m.feed(equity=173.0, available=140.0, margin=30.0, upl=5.0)

    assert event.event_type == EquityEventType.WITHDRAWAL
    assert m._current_mode == EquityMode.NORMAL
    assert m.get_adaptive_params()["position_multiplier"] == 1.0


# ═══════════════════════════════════════════════════════════════
# 4. 外部资金变动事件持久化 + 重启自愈
# ═══════════════════════════════════════════════════════════════

def test_persistence_and_restart_self_heal(tmp_path):
    state_path = tmp_path / "equity_monitor_state.json"

    # 构建一份「历史峰值 235、最近权益 173、记录过提现、当前仍为紧急冻结」的旧状态
    src = EquityMonitor(None)
    src._peak_equity = 235.0
    src._last_known_equity = 173.0
    src._current_mode = EquityMode.EMERGENCY
    src._adaptive_params["position_multiplier"] = 0.0
    src._recent_external_flows.append({
        "type": "withdrawal",
        "timestamp": datetime.now().isoformat(),
        "amount": -62.0,
        "equity_after": 173.0,
    })
    src._baseline_updated_at = datetime.now()
    src._state_path = str(state_path)
    src._save_state()

    # 重启后加载：识别外部资金变动而非交易回撤，重建基准
    m = EquityMonitor(None)
    m._state_path = str(state_path)
    m._load_state()

    assert m._peak_equity == pytest.approx(173.0)
    assert m._max_drawdown_pct == 0.0
    assert m._current_mode == EquityMode.NORMAL
    assert m.is_new_position_allowed() is True
    assert m.get_adaptive_params()["position_multiplier"] == 1.0


def test_restart_self_heal_via_baseline_updated_at(tmp_path):
    """无 recent_external_flows 但有 baseline_updated_at 时同样触发自愈。"""
    state_path = tmp_path / "equity_monitor_state.json"

    src = EquityMonitor(None)
    src._peak_equity = 235.0
    src._last_known_equity = 160.0
    src._current_mode = EquityMode.EMERGENCY
    src._baseline_updated_at = datetime.now()  # 说明资金规模已变
    src._state_path = str(state_path)
    src._save_state()

    m = EquityMonitor(None)
    m._state_path = str(state_path)
    m._load_state()

    assert m._peak_equity == pytest.approx(160.0)
    assert m._current_mode == EquityMode.NORMAL


def test_restart_fallback_3x_still_works(tmp_path):
    """3x 兜底逻辑：无任何外部资金变动记录但峰值远高于权益（如爆仓/漏检）。"""
    state_path = tmp_path / "equity_monitor_state.json"

    src = EquityMonitor(None)
    src._peak_equity = 1000.0
    src._last_known_equity = 100.0
    src._current_mode = EquityMode.EMERGENCY
    src._state_path = str(state_path)
    src._save_state()

    m = EquityMonitor(None)
    m._state_path = str(state_path)
    m._load_state()

    assert m._peak_equity == pytest.approx(100.0)
    assert m._current_mode == EquityMode.NORMAL


# ═══════════════════════════════════════════════════════════════
# 5. 配置节读取
# ═══════════════════════════════════════════════════════════════

def test_equity_monitor_config_section_read():
    m = EquityMonitor({
        "equity_monitor": {
            "emergency_drop_pct": 0.20,
            "deposit_detect_pct": 0.08,
            "ema_alpha": 0.10,
            "trend_confirm_bars": 6,
            "save_interval": 120,
        }
    })
    assert m._emergency_drop_threshold == 0.20
    assert m._deposit_threshold == 0.08
    assert m._ema_alpha == 0.10
    assert m._consecutive_up_threshold == 6
    assert m._save_interval == 120


def test_equity_monitor_defaults_when_no_config():
    m = EquityMonitor(None)
    assert m._emergency_drop_threshold == 0.10
    assert m._emergency_recovery_threshold == 0.05
    assert m._deposit_threshold == 0.05
    assert m._ema_alpha == 0.05
    assert m._consecutive_up_threshold == 5
    assert m._save_interval == 60


# ═══════════════════════════════════════════════════════════════
# 6. 分级回撤护栏（tiered_drawdown_guard）
# ═══════════════════════════════════════════════════════════════

def _make_tiered_monitor():
    return EquityMonitor({
        "equity_monitor": {
            "tiered_drawdown_guard": {
                "enabled": True,
                "levels": [
                    {"drawdown_pct": 0.03, "position_multiplier": 0.7, "risk_budget_ratio": 0.7, "max_positions": 7},
                    {"drawdown_pct": 0.05, "position_multiplier": 0.5, "risk_budget_ratio": 0.5, "max_positions": 5},
                    {"drawdown_pct": 0.08, "position_multiplier": 0.25, "risk_budget_ratio": 0.3, "max_positions": 3},
                ],
            },
        },
    })


def test_tiered_guard_no_drawdown():
    m = _make_tiered_monitor()
    m._adaptive_params["position_multiplier"] = 1.0
    m._adaptive_params["max_positions"] = 10
    m._apply_tiered_drawdown_guard(0.0)
    assert m._adaptive_params["position_multiplier"] == 1.0
    assert m._adaptive_params["max_positions"] == 10


def test_tiered_guard_3pct_drawdown():
    m = _make_tiered_monitor()
    m._adaptive_params["position_multiplier"] = 1.0
    m._adaptive_params["risk_budget_ratio"] = 1.0
    m._adaptive_params["max_positions"] = 10
    m._apply_tiered_drawdown_guard(0.03)
    assert m._adaptive_params["position_multiplier"] == pytest.approx(0.7)
    assert m._adaptive_params["risk_budget_ratio"] == pytest.approx(0.7)
    assert m._adaptive_params["max_positions"] == 7


def test_tiered_guard_8pct_drawdown():
    m = _make_tiered_monitor()
    m._adaptive_params["position_multiplier"] = 1.0
    m._adaptive_params["risk_budget_ratio"] = 1.0
    m._adaptive_params["max_positions"] = 10
    m._apply_tiered_drawdown_guard(0.08)
    assert m._adaptive_params["position_multiplier"] == pytest.approx(0.25)
    assert m._adaptive_params["max_positions"] == 3


def test_tiered_guard_disabled_by_default():
    m = EquityMonitor(None)
    assert m._tiered_guard_enabled is False
    m._adaptive_params["position_multiplier"] = 1.0
    m._apply_tiered_drawdown_guard(0.08)
    assert m._adaptive_params["position_multiplier"] == 1.0


def test_tiered_guard_skipped_in_emergency():
    m = _make_tiered_monitor()
    m._current_mode = EquityMode.EMERGENCY
    m._adaptive_params["position_multiplier"] = 0.0
    m._adaptive_params["max_positions"] = 0
    m._apply_tiered_drawdown_guard(0.08)
    assert m._adaptive_params["position_multiplier"] == 0.0
    assert m._adaptive_params["max_positions"] == 0


# ═══════════════════════════════════════════════════════════════
# 7. 账户级利润留存（profit_reserve）
# ═══════════════════════════════════════════════════════════════

def _make_profit_reserve_monitor(**overrides):
    cfg = {
        "trading": {"total_capital": 200.0},
        "equity_monitor": {
            "profit_reserve": {
                "enabled": True,
                "activation_pct": 0.02,
                "max_pct": 0.10,
                "max_position_multiplier_reduction": 0.3,
            },
        },
    }
    for k, v in overrides.items():
        cfg["equity_monitor"]["profit_reserve"][k] = v
    return EquityMonitor(cfg)


def test_profit_reserve_not_triggered_below_activation():
    m = _make_profit_reserve_monitor()
    m._adaptive_params["position_multiplier"] = 1.0
    m._apply_profit_reserve(201.0)  # 增长 0.5% < 2%
    assert m._adaptive_params["position_multiplier"] == pytest.approx(1.0)


def test_profit_reserve_reduces_multiplier_on_profit():
    m = _make_profit_reserve_monitor()
    m._adaptive_params["position_multiplier"] = 1.0
    m._apply_profit_reserve(205.0)  # 增长 2.5%
    expected = 1.0 * (1.0 - 0.01875)
    assert m._adaptive_params["position_multiplier"] == pytest.approx(expected)


def test_profit_reserve_max_reduction():
    m = _make_profit_reserve_monitor()
    m._adaptive_params["position_multiplier"] = 1.0
    m._apply_profit_reserve(230.0)  # 增长 15% → 达 max_pct，满力度留存
    expected = 1.0 * (1.0 - 0.3)
    assert m._adaptive_params["position_multiplier"] == pytest.approx(expected)


def test_profit_reserve_disabled():
    m = _make_profit_reserve_monitor(enabled=False)
    m._adaptive_params["position_multiplier"] = 1.0
    m._apply_profit_reserve(230.0)
    assert m._adaptive_params["position_multiplier"] == pytest.approx(1.0)


def test_profit_reserve_baseline_initialized_on_first_feed():
    m = EquityMonitor({
        "equity_monitor": {
            "profit_reserve": {"enabled": True, "activation_pct": 0.02, "max_pct": 0.10,
                               "max_position_multiplier_reduction": 0.3},
        },
    })
    assert m._profit_reserve_baseline == 0.0
    m.feed(equity=150.0)  # 无 total_capital，首次权益作为基准
    assert m._profit_reserve_baseline == pytest.approx(150.0)


# ═══════════════════════════════════════════════════════════════
# 8. 回撤速度预警（drawdown_velocity）
# ═══════════════════════════════════════════════════════════════

def _make_dv_monitor(**overrides):
    cfg = {"equity_monitor": {"drawdown_velocity": {
        "enabled": True, "window": 5, "drop_pct_per_window": 0.03,
        "position_multiplier": 0.5,
    }}}
    for k, v in overrides.items():
        cfg["equity_monitor"]["drawdown_velocity"][k] = v
    return EquityMonitor(cfg)


def _append_history(m, equities):
    from core.equity_monitor import EquitySnapshot
    for e in equities:
        m._history.append(EquitySnapshot(
            timestamp=datetime.now(), total_equity=e,
            available_balance=0.0, used_margin=0.0, unrealized_pnl=0.0,
        ))


def test_drawdown_velocity_triggered_on_rapid_drop():
    m = _make_dv_monitor()
    m._adaptive_params["position_multiplier"] = 1.0
    _append_history(m, [100.0, 100.0, 100.0, 100.0, 96.0])  # -4% 急跌
    m._apply_drawdown_velocity_guard()
    assert m._adaptive_params["position_multiplier"] == pytest.approx(0.5)


def test_drawdown_velocity_not_triggered_on_small_drop():
    m = _make_dv_monitor()
    m._adaptive_params["position_multiplier"] = 1.0
    _append_history(m, [100.0, 100.0, 100.0, 100.0, 99.0])  # -1% < 3%
    m._apply_drawdown_velocity_guard()
    assert m._adaptive_params["position_multiplier"] == pytest.approx(1.0)


def test_drawdown_velocity_disabled_by_default():
    m = EquityMonitor(None)
    assert m._dv_enabled is False
    m._adaptive_params["position_multiplier"] = 1.0
    _append_history(m, [100.0, 100.0, 100.0, 100.0, 96.0])
    m._apply_drawdown_velocity_guard()
    assert m._adaptive_params["position_multiplier"] == pytest.approx(1.0)


def test_drawdown_velocity_skipped_in_emergency():
    m = _make_dv_monitor()
    m._current_mode = EquityMode.EMERGENCY
    m._adaptive_params["position_multiplier"] = 0.0
    _append_history(m, [100.0, 100.0, 100.0, 100.0, 96.0])
    m._apply_drawdown_velocity_guard()
    assert m._adaptive_params["position_multiplier"] == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════
# 9. 企业级强化：JSON 安全 + 配置边界
# ═══════════════════════════════════════════════════════════════

def test_get_equity_status_trough_equity_json_safe():
    m = EquityMonitor(None)
    # 尚未喂入权益时 trough_equity 为 +inf 哨兵，对外应回退为 0.0（JSON 安全）
    assert m._trough_equity == float("inf")
    status = m.get_equity_status()
    assert status["trough_equity"] == 0.0
    json.dumps(status, allow_nan=False)  # 不应抛 ValueError


def test_emergency_drop_threshold_clamped():
    m = EquityMonitor({"equity_monitor": {"emergency_drop_pct": 0.0}})
    # 阈值 0 会导致 severity/mode_confidence 除零，应钳制到下限 0.01
    assert m._emergency_drop_threshold == pytest.approx(0.01)


def test_emergency_drop_threshold_default():
    m = EquityMonitor(None)
    assert m._emergency_drop_threshold == pytest.approx(0.10)


# ═══════════════════════════════════════════════════════════════
# 10. 恢复期渐进加仓（recovery_ramp）
# ═══════════════════════════════════════════════════════════════

def _make_recovery_ramp_monitor(**overrides):
    cfg = {"equity_monitor": {"recovery_ramp": {
        "enabled": True, "floor": 0.3, "ceiling": 1.0, "step_per_bar": 0.1,
    }}}
    for k, v in overrides.items():
        cfg["equity_monitor"]["recovery_ramp"][k] = v
    return EquityMonitor(cfg)


def _recovery_snapshot(equity=950.0):
    from core.equity_monitor import EquitySnapshot
    return EquitySnapshot(timestamp=datetime.now(), total_equity=equity,
                          available_balance=0.0, used_margin=0.0, unrealized_pnl=0.0)


def test_recovery_ramp_progressive():
    m = _make_recovery_ramp_monitor()
    m._current_mode = EquityMode.RECOVERY
    m._peak_equity = 1000.0
    m._trough_equity = 900.0
    snap = _recovery_snapshot()
    m._update_adaptive_params(datetime.now(), 950.0, snap)
    assert m._adaptive_params["position_multiplier"] == pytest.approx(0.4)  # floor + 1*step
    m._update_adaptive_params(datetime.now(), 950.0, snap)
    assert m._adaptive_params["position_multiplier"] == pytest.approx(0.5)  # floor + 2*step


def test_recovery_ramp_caps_at_ceiling():
    m = _make_recovery_ramp_monitor()
    m._current_mode = EquityMode.RECOVERY
    m._peak_equity = 1000.0
    m._trough_equity = 900.0
    snap = _recovery_snapshot()
    for _ in range(20):
        m._update_adaptive_params(datetime.now(), 950.0, snap)
    assert m._adaptive_params["position_multiplier"] == pytest.approx(1.0)  # 封顶 ceiling


def test_recovery_ramp_disabled_by_default():
    m = EquityMonitor(None)
    assert m._recovery_ramp_enabled is False