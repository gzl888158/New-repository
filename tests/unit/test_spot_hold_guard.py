"""
现货持有守卫（spot_hold_guard）单元测试
==========================================
覆盖：account_manager 现货持币余额提取、AGI 现货持有感知、_diagnose 现货过度分散/
现货止盈告警、_spot_hold_actions 收敛动作、禁用。
"""
import pytest

from core.account_manager import AccountManager
from core.quant_agi_orchestrator import QuantAGIOrchestrator


# ── account_manager 现货持币余额提取 ─────────────────────

def test_extract_spot_holdings_non_usdt():
    info = {"details": [
        {"ccy": "USDT", "cashBal": "1000.0", "availBal": "900.0"},
        {"ccy": "BTC", "cashBal": "0.5", "availBal": "0.5"},
        {"ccy": "ETH", "cashBal": "3.2", "availBal": "3.2"},
        {"ccy": "SOL", "cashBal": "0", "availBal": "0"},
    ]}
    holdings = AccountManager._extract_spot_holdings(info)
    assert holdings == {"BTC": 0.5, "ETH": 3.2}


def test_extract_spot_holdings_falls_back_to_avail_bal():
    info = {"details": [{"ccy": "BTC", "availBal": "0.25"}]}
    holdings = AccountManager._extract_spot_holdings(info)
    assert holdings == {"BTC": 0.25}


def test_extract_spot_holdings_empty_or_none():
    assert AccountManager._extract_spot_holdings(None) == {}
    assert AccountManager._extract_spot_holdings({}) == {}
    assert AccountManager._extract_spot_holdings({"details": []}) == {}


# ── AGI 现货持有感知 ─────────────────────────────────────

class FakeAccountManager:
    def __init__(self, holdings=None):
        self._holdings = holdings or {}

    def get_spot_holdings(self):
        return dict(self._holdings)


def _config(**overrides):
    guard = {
        "enabled": True,
        "max_spot_currencies": 3,
        "profit_take_pct": 0.02,
        "reduce_target": 0.1,
    }
    guard.update(overrides)
    return {"agi_orchestrator": {"spot_hold_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def test_perceive_spot_holdings_from_account_manager():
    orch = _orch()
    orch.account_manager = FakeAccountManager({"BTC": 0.5, "ETH": 3.2, "SOL": 10.0, "XRP": 100.0})
    perceived = orch._perceive_spot_holdings()
    assert perceived["count"] == 4
    assert perceived["currencies"]["BTC"] == 0.5


def test_perceive_spot_holdings_no_account_manager():
    orch = _orch()
    assert orch._perceive_spot_holdings() == {"currencies": {}, "count": 0}


# ── 诊断 ─────────────────────────────────────────────────

def test_diagnose_spot_overweight():
    orch = _orch(max_spot_currencies=3)
    perception = {
        "spot_holdings": {"currencies": {"BTC": 1, "ETH": 1, "SOL": 1, "XRP": 1}, "count": 4},
    }
    alerts = orch._diagnose(perception)
    overweight = [a for a in alerts if a["type"] == "spot_overweight"]
    assert len(overweight) == 1
    assert overweight[0]["spot_count"] == 4


def test_diagnose_no_spot_overweight_within_limit():
    orch = _orch(max_spot_currencies=3)
    perception = {
        "spot_holdings": {"currencies": {"BTC": 1, "ETH": 1}, "count": 2},
    }
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "spot_overweight"]


def test_diagnose_spot_profit():
    orch = _orch(profit_take_pct=0.02)
    perception = {
        "equity": 1000.0,
        "contribution": {"strategies": {
            "spot_grid": {"total_pnl": 25.0},
            "spot_martingale": {"total_pnl": -5.0},
        }},
        "spot_holdings": {"currencies": {}, "count": 0},
    }
    alerts = orch._diagnose(perception)
    profit = [a for a in alerts if a["type"] == "spot_profit"]
    assert len(profit) == 1  # (25 - 5)/1000 = 0.02 >= 0.02


def test_diagnose_no_spot_profit_below_threshold():
    orch = _orch(profit_take_pct=0.02)
    perception = {
        "equity": 1000.0,
        "contribution": {"strategies": {"spot_grid": {"total_pnl": 10.0}}},
        "spot_holdings": {"currencies": {}, "count": 0},
    }
    alerts = orch._diagnose(perception)
    assert not [a for a in alerts if a["type"] == "spot_profit"]


# ── 动作生成 ─────────────────────────────────────────────

def test_spot_hold_actions_generate_decrease():
    orch = _orch()
    alerts = [{"type": "spot_overweight", "spot_count": 4}]
    actions = orch._spot_hold_actions(alerts)
    assert len(actions) == 2
    assert all(a["type"] == "reallocate" for a in actions)
    assert all(a["action"] == "decrease" for a in actions)
    assert {a["strategy"] for a in actions} == {"spot_grid", "spot_martingale"}


def test_spot_hold_actions_no_alert():
    orch = _orch()
    assert orch._spot_hold_actions([{"type": "strategy_health_critical"}]) == []


def test_spot_hold_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "spot_overweight", "spot_count": 4}]
    assert orch._spot_hold_actions(alerts) == []
    perception = {"spot_holdings": {"currencies": {"BTC": 1, "ETH": 1, "SOL": 1, "XRP": 1}, "count": 4}}
    assert not [a for a in orch._diagnose(perception) if a["type"] == "spot_overweight"]
