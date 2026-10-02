"""
逐币种精细杠杆守卫（symbol_param_guard）单元测试
====================================================
覆盖：_diagnose 产生 symbol_losing 告警、_symbol_param_actions 生成 symbol 维度
param_adjust、冷却、杠杆下限、禁用、scheduler 逐币种部署器。
"""
import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _tier(lev_min=3, lev_max=5, lev_def=5):
    return {
        "leverage_min": lev_min,
        "leverage_max": lev_max,
        "leverage_default": lev_def,
        "position_limit": 0.25,
        "grid_spacing_min": 0.0114,
        "grid_spacing_max": 0.018,
        "slippage": 0.0015,
    }


def _config(**overrides):
    guard = {
        "enabled": True,
        "loss_threshold": 5.0,
        "reduce_leverage_step": 1,
        "min_leverage": 1,
        "cooldown_cycles": 5,
    }
    guard.update(overrides)
    return {
        "agi_orchestrator": {"symbol_param_guard": guard},
        "currencies": {
            "tier1_symbols": ["ETH"],
            "tier1_settings": _tier(2, 5, 3),
            "tier2_symbols": ["SOL", "ARB"],
            "tier2_settings": _tier(3, 5, 5),
            "tier3_symbols": [],
            "tier3_settings": _tier(2, 5, 3),
            "symbol_overrides": {
                "ARB": {
                    "leverage_min": 3,
                    "leverage_max": 5,
                    "leverage_default": 4,
                }
            },
        },
    }


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


class FakeAccountManager:
    def __init__(self, symbol_pnl=None):
        self._symbol_pnl = symbol_pnl or {}

    def get_symbol_unrealized_pnl(self):
        return dict(self._symbol_pnl)


def test_symbol_param_actions_generate_symbol_adjust():
    """单币种浮亏超阈值 → 生成 symbol 维度 param_adjust（leverage_default 下调）。"""
    orch = _orch()
    alerts = [{"type": "symbol_losing", "symbol": "ARB-USDT-SWAP", "upl": -8.0}]
    actions = orch._symbol_param_actions(alerts)
    assert len(actions) == 1
    a = actions[0]
    assert a["type"] == "param_adjust"
    assert a["symbol"] == "ARB-USDT-SWAP"
    assert a["param"] == "leverage_default"
    assert a["value"] == pytest.approx(3.0)  # ARB 默认 4 → 下调 1 步到 3


def test_symbol_param_actions_ignore_non_symbol_losing():
    orch = _orch()
    alerts = [{"type": "strategy_health_critical", "strategy": "grid"}]
    assert orch._symbol_param_actions(alerts) == []


def test_symbol_param_actions_respects_min_leverage_floor():
    """当前杠杆已在 min_leverage 下限 → 不再下调。"""
    orch = _orch(min_leverage=4)
    alerts = [{"type": "symbol_losing", "symbol": "ARB-USDT-SWAP", "upl": -8.0}]
    # ARB 当前 4，min_leverage=4 → new_lev = max(4, 3) = 4 == cur → 跳过
    assert orch._symbol_param_actions(alerts) == []


def test_symbol_param_actions_respects_cooldown():
    """冷却期内不重复下调，冷却结束后放行。"""
    orch = _orch(cooldown_cycles=5)
    orch._cycle_count = 10
    orch._symbol_leverage_cooldown["ARB-USDT-SWAP"] = 8  # 2 周期前 → 冷却中
    alerts = [{"type": "symbol_losing", "symbol": "ARB-USDT-SWAP", "upl": -8.0}]
    assert orch._symbol_param_actions(alerts) == []

    orch._symbol_leverage_cooldown["ARB-USDT-SWAP"] = 4  # 6 周期前 → 冷却结束
    actions = orch._symbol_param_actions(alerts)
    assert len(actions) == 1


def test_symbol_param_guard_disabled():
    orch = _orch(enabled=False)
    alerts = [{"type": "symbol_losing", "symbol": "ARB-USDT-SWAP", "upl": -8.0}]
    assert orch._symbol_param_actions(alerts) == []


def test_diagnose_symbol_losing_alert():
    orch = _orch()
    orch.account_manager = FakeAccountManager({"ARB-USDT-SWAP": -8.0, "SOL-USDT-SWAP": -2.0})
    alerts = orch._diagnose({"symbol_pnl": {"ARB-USDT-SWAP": -8.0, "SOL-USDT-SWAP": -2.0}})
    losing = [a for a in alerts if a["type"] == "symbol_losing"]
    assert len(losing) == 1
    assert losing[0]["symbol"] == "ARB-USDT-SWAP"


def test_diagnose_no_symbol_losing_below_threshold():
    orch = _orch()
    alerts = orch._diagnose({"symbol_pnl": {"ARB-USDT-SWAP": -3.0}})
    assert not [a for a in alerts if a["type"] == "symbol_losing"]


def test_perceive_symbol_pnl_from_account_manager():
    orch = _orch()
    orch.account_manager = FakeAccountManager({"ARB-USDT-SWAP": -8.0})
    assert orch._perceive_symbol_pnl() == {"ARB-USDT-SWAP": -8.0}


def test_perceive_symbol_pnl_no_account_manager():
    orch = _orch()
    assert orch._perceive_symbol_pnl() == {}


# ── scheduler 逐币种部署器 ───────────────────────────────

import asyncio
from core.scheduler import TradingScheduler


def _run(coro):
    return asyncio.run(coro)


class _S(TradingScheduler):
    def __init__(self):
        # 不调用父类 __init__（避免重量级依赖），仅设置部署器所需字段
        self.config = {"currencies": {}}
        self.strategy_manager = None


def test_symbol_param_deployer_writes_symbol_overrides():
    s = _S()
    result = _run(TradingScheduler._param_adjust_deployer(s, {
        "symbol": "ARB-USDT-SWAP", "param": "leverage_default", "value": 3.0,
    }))
    assert result["deployed"] is True
    assert s.config["currencies"]["symbol_overrides"]["ARB"]["leverage_default"] == 3.0


def test_symbol_param_deployer_missing_target_fail_closed():
    s = _S()
    result = _run(TradingScheduler._param_adjust_deployer(s, {
        "param": "leverage_default", "value": 3.0,
    }))
    assert result["deployed"] is False
