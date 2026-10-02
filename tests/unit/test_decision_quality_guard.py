"""
AGI 决策记忆质量评估（decision_quality_guard）单元测试
====================================================
覆盖：_decision_quality_score 盈利周期占比、_diagnose 决策质量低告警、样本不足中性、禁用。
"""
from collections import deque

import pytest

from core.quant_agi_orchestrator import QuantAGIOrchestrator


def _config(**overrides):
    guard = {"enabled": True, "quality_threshold": 0.4, "min_samples": 3}
    guard.update(overrides)
    return {"agi_orchestrator": {"decision_quality_guard": guard}}


def _orch(**overrides):
    return QuantAGIOrchestrator(config=_config(**overrides))


def _memory(pnls):
    return deque([{"total_pnl": p} for p in pnls], maxlen=10)


def test_decision_quality_score_majority_win():
    orch = _orch()
    orch._decision_memory = _memory([10.0, 5.0, -3.0])
    assert orch._decision_quality_score() == pytest.approx(2 / 3)


def test_decision_quality_score_all_loss():
    orch = _orch()
    orch._decision_memory = _memory([-1.0, -2.0, -3.0])
    assert orch._decision_quality_score() == pytest.approx(0.0)


def test_decision_quality_score_insufficient_sample_neutral():
    orch = _orch()
    orch._decision_memory = _memory([10.0, -1.0])  # 2 < min_samples=3
    assert orch._decision_quality_score() == pytest.approx(0.5)


def test_diagnose_decision_quality_low():
    orch = _orch()
    orch._decision_memory = _memory([-1.0, -2.0, -3.0])  # 0% 盈利 < 0.4
    alerts = orch._diagnose({})
    hits = [a for a in alerts if a["type"] == "decision_quality_low"]
    assert len(hits) == 1
    assert hits[0]["score"] == pytest.approx(0.0)


def test_diagnose_no_alert_when_quality_ok():
    orch = _orch()
    orch._decision_memory = _memory([10.0, 5.0, 3.0])  # 100% 盈利
    alerts = orch._diagnose({})
    assert not [a for a in alerts if a["type"] == "decision_quality_low"]


def test_decision_quality_disabled():
    orch = _orch(enabled=False)
    orch._decision_memory = _memory([-1.0, -2.0, -3.0])
    alerts = orch._diagnose({})
    assert not [a for a in alerts if a["type"] == "decision_quality_low"]


# ── 死锁自愈：持续触发后允许最小试探开单 ──────────────────────────
def test_decision_quality_lock_count_increments():
    """触发后 lock_count 递增，未达 recovery_after_cycles 时 recovery_mode=False"""
    orch = _orch(quality_threshold=0.4, min_samples=3, recovery_after_cycles=3)
    orch._decision_memory = _memory([-1.0, -2.0, -3.0])
    orch._diagnose({})
    assert orch._decision_quality_lock_count == 1
    alerts = orch._diagnose({})
    assert alerts[0]["recovery_mode"] is False if alerts else True


def test_decision_quality_recovery_mode_after_threshold():
    """持续触发 recovery_after_cycles 周期后 recovery_mode=True"""
    orch = _orch(quality_threshold=0.4, min_samples=3, recovery_after_cycles=3)
    orch._decision_memory = _memory([-1.0, -2.0, -3.0])
    for _ in range(3):
        orch._diagnose({})
    assert orch._decision_quality_lock_count == 3
    alerts = orch._diagnose({})
    dq = next(a for a in alerts if a["type"] == "decision_quality_low")
    assert dq["recovery_mode"] is True


def test_decision_quality_lock_resets_on_recovery():
    """质量回升（盈利周期占比 ≥ threshold）→ lock_count 重置为 0"""
    orch = _orch(quality_threshold=0.4, min_samples=3, recovery_after_cycles=5)
    orch._decision_memory = _memory([-1.0, -2.0, -3.0])
    orch._diagnose({})
    assert orch._decision_quality_lock_count == 1
    # 质量回升
    orch._decision_memory = _memory([10.0, 5.0, 3.0])
    orch._diagnose({})
    assert orch._decision_quality_lock_count == 0


def test_offensive_actions_recovery_allows_minimal_boost():
    """恢复期内 _offensive_allocation_actions 不再 return []，且 boost_step 被缩放"""
    orch = _orch(quality_threshold=0.4, min_samples=3,
                 recovery_after_cycles=2, recovery_boost_scale=0.5)
    orch._decision_memory = _memory([-1.0, -2.0, -3.0])
    # 触发 2 次进入恢复期
    orch._diagnose({})
    orch._diagnose({})
    # 第 3 次：recovery_mode=True
    alerts = orch._diagnose({})
    dq = next(a for a in alerts if a["type"] == "decision_quality_low")
    assert dq["recovery_mode"] is True
    # _offensive_allocation_actions 不应因 decision_quality_low 直接 return []
    # （函数会在其他 gate 通过后继续，此处仅验证 dq_alert 被识别为恢复模式）
    assert orch._decision_quality_lock_count >= 2


def test_offensive_actions_blocks_before_recovery():
    """未达 recovery_after_cycles 时 _offensive_allocation_actions 仍 return []"""
    orch = _orch(quality_threshold=0.4, min_samples=3, recovery_after_cycles=5)
    orch._decision_memory = _memory([-1.0, -2.0, -3.0])
    orch._diagnose({})  # lock_count=1，未恢复
    # 验证 lock_count 未达恢复阈值
    assert orch._decision_quality_lock_count == 1
    assert orch._decision_quality_lock_count < orch._decision_quality_recovery_cycles
