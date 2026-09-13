"""P0 企业级升级：持久化全局 Kill Switch 单元测试。

覆盖：
1. KillSwitch enable/disable/is_enabled 基本语义
2. KillSwitch 持久化（fail-closed：重启后保持 enabled）
3. KillSwitch 默认 disabled（无状态文件）
4. RiskGate L0 集成：开仓信号被 L0 拦截，且不误伤平仓信号
"""

import pytest

from core.kill_switch import KillSwitch
from core.risk_gate import RiskGate, RiskLayer, RiskAction


class TestKillSwitch:
    def test_default_disabled(self, tmp_path):
        ks = KillSwitch(state_file=str(tmp_path / "ks.json"))
        assert ks.is_enabled() is False

    def test_enable_disable_roundtrip(self, tmp_path):
        ks = KillSwitch(state_file=str(tmp_path / "ks.json"))
        ks.enable(reason="手动暂停", by="ops")
        assert ks.is_enabled() is True
        assert "暂停" in ks.get_reason()
        ks.disable(reason="恢复")
        assert ks.is_enabled() is False

    def test_persistence_fail_closed_restore(self, tmp_path):
        path = str(tmp_path / "ks.json")
        ks1 = KillSwitch(state_file=path)
        ks1.enable(reason="市场异常暂停", by="auto")

        # 模拟进程重启：新实例从磁盘恢复，必须保持 enabled（fail-closed）
        ks2 = KillSwitch(state_file=path)
        assert ks2.is_enabled() is True
        assert "市场异常" in ks2.get_reason()

    def test_disable_persists(self, tmp_path):
        path = str(tmp_path / "ks.json")
        ks1 = KillSwitch(state_file=path)
        ks1.enable(reason="x")
        ks1.disable(reason="恢复")
        ks2 = KillSwitch(state_file=path)
        assert ks2.is_enabled() is False

    def test_history_recorded_on_enable_disable(self, tmp_path):
        ks = KillSwitch(state_file=str(tmp_path / "ks.json"))
        ks.enable(reason="市场异常", by="auto")
        ks.disable(reason="恢复交易", by="ops")

        history = ks.get_history()
        assert len(history) == 2
        assert history[0]["action"] == "enable"
        assert history[0]["by"] == "auto"
        assert "市场异常" in history[0]["reason"]
        assert history[1]["action"] == "disable"
        assert history[1]["by"] == "ops"

    def test_history_persisted(self, tmp_path):
        path = str(tmp_path / "ks.json")
        ks1 = KillSwitch(state_file=path)
        ks1.enable(reason="auto_trip", by="auto")

        ks2 = KillSwitch(state_file=path)
        assert len(ks2.get_history()) == 1
        assert ks2.get_history()[0]["action"] == "enable"

    def test_history_fifo_bounded(self, tmp_path):
        ks = KillSwitch(state_file=str(tmp_path / "ks.json"))
        # 触发 60 次，历史应截断到 _MAX_HISTORY（50）
        for i in range(60):
            ks.enable(reason=f"r{i}")
            ks.disable(reason=f"d{i}")
        history = ks.get_history()
        assert len(history) == 50
        assert history[0]["action"] == "enable"
        # 最早的动作（enable r0）应已被挤出
        assert history[0]["reason"] != "r0"


class TestRiskGateKillSwitchIntegration:
    def _make_risk_gate(self, tmp_path):
        rg = RiskGate(config={})
        rg.set_kill_switch(KillSwitch(state_file=str(tmp_path / "ks.json")))
        return rg

    def test_open_signal_blocked_by_l0(self, tmp_path):
        rg = self._make_risk_gate(tmp_path)
        rg.enable_kill_switch(reason="测试暂停")

        signal = {"symbol": "ETH-USDT-SWAP", "signal_type": "grid_open", "direction": "long"}
        result = rg.validate(signal)

        assert result.passed is False
        assert result.blocked_layer == RiskLayer.L0_KILL_SWITCH
        assert result.action == RiskAction.FREEZE
        assert "Kill" in result.summary or "Kill" in result.summary.lower()

    def test_close_signal_not_blocked_by_l0(self, tmp_path):
        rg = self._make_risk_gate(tmp_path)
        rg.enable_kill_switch(reason="测试暂停")

        # 平仓/减仓信号必须穿透 L0（降低风险路径永不阻断）
        close_signal = {"symbol": "ETH-USDT-SWAP", "signal_type": "stop_loss", "direction": "close", "reduce_only": True}
        result = rg.validate(close_signal, is_close=True)

        assert result.blocked_layer != RiskLayer.L0_KILL_SWITCH

    def test_disabled_does_not_block_l0(self, tmp_path):
        rg = self._make_risk_gate(tmp_path)
        assert rg.is_kill_switch_enabled() is False

        signal = {"symbol": "ETH-USDT-SWAP", "signal_type": "grid_open", "direction": "long"}
        result = rg.validate(signal)

        assert result.blocked_layer != RiskLayer.L0_KILL_SWITCH

    def test_emergency_triggered_default_false(self, tmp_path):
        rg = self._make_risk_gate(tmp_path)
        # 未手动/自动触发 L5 时，紧急熔断未触发（自适应联动信号源为 False）
        assert rg.is_emergency_triggered() is False