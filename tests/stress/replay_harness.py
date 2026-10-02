"""感知序列回放仿真 harness
=================================
为 QuantAGIOrchestrator 编排层守卫提供回放验证能力。

核心价值：验证需要跨周期状态累积的守卫参数有效性。这些守卫依赖 orchestrator
内部的时序状态，单次 perception 注入无法触发，必须顺序回放 perception 序列：
  - trend_confirmation_cycles：依赖 _trend_confirmed_streak（在 _reflect 里跨周期更新）
  - drawdown_accelerating：依赖 _drawdown_history（在 _diagnose 里累积）
  - decision_quality_low：依赖 _decision_memory（跨周期累积）
  - health trend：依赖 _last_health_score（跨周期对比）

工作方式：
  构造合成 perception 序列 → 顺序喂给 _diagnose + _reflect →
  收含 alerts → A/B 对比守卫参数开 vs 关，验证守卫是否真实有效。

切入点：直接调 _diagnose + _reflect，绕过 _perceive 的真实数据拉取与
_decide 的 dynamic_allocator 依赖，让验证完全离线、确定性、可复现。
守卫触发主要在 _diagnose（alerts），进攻块也生成 offensive_opportunity alert，
因此 _diagnose + _reflect 足以覆盖编排层守卫验证。
"""
import copy
from datetime import datetime
from typing import Any, Dict, List, Optional

from core.quant_agi_orchestrator import QuantAGIOrchestrator


# ---------------------------------------------------------------------------
# 基线 perception 与序列生成器
# ---------------------------------------------------------------------------

def make_perception(**overrides) -> dict:
    """生成基线 healthy perception：所有守卫都不触发，进攻块前置条件全满足。

    进攻块前置条件（_diagnose 第 3791-3816 行）：
      - offense_allowed: equity_status.mode == "normal"
      - trend_offense: regime in (trend_bullish/trend_bearish), strength >= min
      - confident: confidence >= min_regime_confidence
      - regime_confirmed: streak >= trend_confirmation_cycles（由序列回放累积）
    """
    p = {
        "equity": 10000.0,
        "total_capital": 10000.0,
        "unrealized_pnl": 0.0,
        "used_margin": 1000.0,
        "market_regime": {
            "regime": "trend_bullish",
            "strength": 0.8,
            "confidence": 0.9,
        },
        "contribution": {
            "strategies": {
                "grid": {
                    "health_grade": "A",
                    "health_score": 90,
                    "total_pnl": 100.0,
                    "trend": "improving",
                },
            },
            "overall_health_score": 90.0,
            "total_trades": 10,
            "total_pnl": 100.0,
        },
        "freeze_state": {},
        "equity_status": {
            "mode": "normal",
            "max_drawdown_pct": 0.0,
            "consecutive_up": 0,
            "consecutive_down": 0,
        },
        "net_exposure": {"long": 1000.0, "short": 1000.0},
        "correlation": {},
        "symbol_pnl": {},
        "spot_holdings": {"currencies": [], "count": 0},
    }
    p.update(overrides)
    return p


def trend_confirmation_sequence(
    n: int, regime: str = "trend_bullish"
) -> List[dict]:
    """连续 n 周期相同 regime 的 perception 序列。

    用于验证 trend_confirmation_cycles：市场状态需持续 >= cycles 个周期才开单。
    """
    return [
        make_perception(market_regime={"regime": regime, "strength": 0.8, "confidence": 0.9})
        for _ in range(n)
    ]


def regime_switch_sequence(switch_at: int, n: int = 10) -> List[dict]:
    """前 switch_at 周期 trend_bullish，之后切到 range_bound，验证 streak 重置。"""
    seq = []
    for i in range(n):
        regime = "trend_bullish" if i < switch_at else "range_bound"
        seq.append(make_perception(market_regime={"regime": regime, "strength": 0.8, "confidence": 0.9}))
    return seq


def stress_escalation_sequence(
    equity: float, gross_steps: List[float]
) -> List[dict]:
    """gross 敞口逐步放大的 perception 序列。

    用于验证 portfolio_stress_guard：毛敞口压力损失占权益比例超 budget 时触发。
    gross_steps: 每周期的总敞口（long+short），逐步放大逼近 stress 预算。
    """
    return [
        make_perception(equity=equity, net_exposure={"long": g / 2, "short": g / 2})
        for g in gross_steps
    ]


def drawdown_acceleration_sequence(
    window: int, start_dd: float = 0.02
) -> List[dict]:
    """回撤连续加深 window 周期的 perception 序列。

    用于验证 drawdown_accelerating：max_drawdown_pct 连续 window 周期严格加深。
    """
    return [
        make_perception(equity_status={
            "mode": "decline",
            "max_drawdown_pct": start_dd * (i + 1),
            "consecutive_up": 0,
            "consecutive_down": i + 1,
        })
        for i in range(window)
    ]


def downside_momentum_sequence(n_down: int) -> List[dict]:
    """consecutive_down 逐步累积的 perception 序列。

    用于验证 downside_momentum 守卫：连续下跌周期数达阈值触发 market_panicking。
    """
    return [
        make_perception(equity_status={
            "mode": "decline",
            "max_drawdown_pct": 0.005 * (i + 1),
            "consecutive_up": 0,
            "consecutive_down": i + 1,
        })
        for i in range(n_down)
    ]


def tail_risk_sequence(
    max_dd: float, n: int = 3, threshold: float = 0.15
) -> List[dict]:
    """策略回撤超阈值的 perception 序列。

    用于验证 tail_risk_guard：组合中最深策略回撤 > tail_risk_threshold。
    """
    return [
        make_perception(contribution={
            "strategies": {
                "grid": {
                    "health_grade": "A", "health_score": 90,
                    "total_pnl": 100.0, "trend": "improving",
                    "max_drawdown": max_dd,
                },
                "trend": {
                    "health_grade": "B", "health_score": 70,
                    "total_pnl": 50.0, "trend": "stable",
                    "max_drawdown": max_dd * 0.8,
                },
            },
            "overall_health_score": 80.0,
            "total_trades": 10,
            "total_pnl": 150.0,
        })
        for _ in range(n)
    ]


def health_degradation_sequence(grades: List[str]) -> List[dict]:
    """策略健康度逐步劣化的 perception 序列（A → B → C → D → F）。

    用于验证 risk_response / health_degraded 守卫：策略健康度恶化时收敛。
    """
    score_map = {"A": 90, "B": 75, "C": 55, "D": 35, "F": 15}
    seq = []
    for g in grades:
        score = score_map.get(g, 50)
        seq.append(make_perception(contribution={
            "strategies": {
                "grid": {
                    "health_grade": g,
                    "health_score": score,
                    "total_pnl": max(10.0 * score / 90, -50.0),
                    "trend": "declining" if g in ("D", "F") else "stable",
                },
            },
            "overall_health_score": float(score),
            "total_trades": 10,
            "total_pnl": max(10.0 * score / 90, -50.0),
        }))
    return seq


# ---------------------------------------------------------------------------
# 回放 harness
# ---------------------------------------------------------------------------

class PerceptionReplayHarness:
    """感知序列回放 harness：顺序喂 perception 给 orchestrator，收集守卫触发。

    每个 tick 执行 _diagnose + _reflect 两步（与 run_cycle 的诊断/反馈一致），
    保证跨周期状态（streak/history/memory）正确累积。跳过 _perceive（真实数据）
    与 _decide/_act（依赖 dynamic_allocator），聚焦守卫触发验证。
    """

    def __init__(self, agi_cfg: Optional[dict] = None):
        self.agi_cfg = dict(agi_cfg or {})
        self.orch = QuantAGIOrchestrator(config={"agi_orchestrator": self.agi_cfg})
        self._cycle = 0

    def _reset_temporal_state(self):
        """重置 orchestrator 的跨周期时序状态，避免持久化残留污染 A/B 对比。

        orchestrator 初始化时可能从 data/agi_orchestrator_state.json 加载残留状态
       （drawdown_history / decision_memory / streak 等），导致 A/B 两个分支起点
        不一致。本方法在每次 replay 开头把所有时序状态归零，保证确定性基线。
        """
        o = self.orch
        # deque 类型直接 clear
        for attr in ("_drawdown_history", "_decision_memory", "_equity_window"):
            dq = getattr(o, attr, None)
            if dq is not None and hasattr(dq, "clear"):
                dq.clear()
        # streak / regime 归零
        o._trend_confirmed_streak = 0
        o._trend_confirmed_regime = None
        # 健康度 / 目标规划基线
        o._last_health_score = None
        o._goal_baseline_equity = None
        o._goal_peak_equity = 0.0
        # 策略级回撤历史
        if hasattr(o, "_strategy_max_drawdown_history"):
            o._strategy_max_drawdown_history.clear()
        # 组合健康度历史
        if hasattr(o, "_portfolio_health_history") and hasattr(o._portfolio_health_history, "clear"):
            o._portfolio_health_history.clear()
        # 进攻冷却周期
        if hasattr(o, "_offensive_last_boost_cycle"):
            o._offensive_last_boost_cycle = {}
        # 执行结果反馈
        o._last_execution_result = None

    def replay(self, perceptions: List[dict]) -> dict:
        """顺序回放 perception 序列，返回汇总指标。

        返回 dict 含：
          - alerts_log: 每周期的 alerts 列表
          - streak_log: 每周期结束后的 _trend_confirmed_streak 值
          - alerts_by_type: 按 type 汇总计数
          - alerts_by_level: 按 level 汇总计数
          - offensive_opportunity_cycles: 出现 offensive_opportunity 的周期序号列表
          - total_alerts: 全周期 alert 总数
        """
        self._reset_temporal_state()
        alerts_log: List[List[dict]] = []
        streak_log: List[int] = []
        offensive_cycles: List[int] = []

        for p in perceptions:
            self._cycle += 1
            # 1. 诊断（生成 alerts + 更新 _drawdown_history 等时序状态）
            alerts = self.orch._diagnose(p)
            # 2. 反馈（更新 _trend_confirmed_streak 供下一周期 _diagnose 进攻块用）
            report = {
                "cycle": self._cycle,
                "perception": p,
                "diagnosis": {"alerts": alerts},
                "decision": {},
                "actions": [],
                "reflection": {},
            }
            self.orch._reflect(report, p, alerts)

            alerts_log.append(alerts)
            streak_log.append(self.orch._trend_confirmed_streak)
            for a in alerts:
                if a.get("type") == "offensive_opportunity":
                    offensive_cycles.append(self._cycle)

        return {
            "alerts_log": alerts_log,
            "streak_log": streak_log,
            "alerts_by_type": self._count_by(alerts_log, key="type"),
            "alerts_by_level": self._count_by(alerts_log, key="level"),
            "offensive_opportunity_cycles": offensive_cycles,
            "offensive_opportunity_count": len(offensive_cycles),
            "total_alerts": sum(len(a) for a in alerts_log),
            "n_cycles": len(perceptions),
        }

    def run_ab(
        self,
        perceptions: List[dict],
        guard_cfg_on: dict,
        guard_cfg_off: dict,
    ) -> dict:
        """A/B 对比守卫参数开 vs 关。

        每个分支创建独立的 orchestrator 实例（状态隔离），回放同一 perception 序列。
        返回 {"on": {...}, "off": {...}, "delta": {...}}。
        """
        result_on = PerceptionReplayHarness(agi_cfg=guard_cfg_on).replay(perceptions)
        result_off = PerceptionReplayHarness(agi_cfg=guard_cfg_off).replay(perceptions)
        delta = {
            "alerts_delta": result_on["total_alerts"] - result_off["total_alerts"],
            "offensive_delta": (
                result_on["offensive_opportunity_count"]
                - result_off["offensive_opportunity_count"]
            ),
        }
        return {"on": result_on, "off": result_off, "delta": delta}

    @staticmethod
    def _count_by(alerts_log: List[List[dict]], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for alerts in alerts_log:
            for a in alerts:
                k = str(a.get(key, "unknown"))
                counts[k] = counts.get(k, 0) + 1
        return counts

    @staticmethod
    def alert_types_in(result: dict) -> Dict[str, int]:
        """从 replay 结果提取 alert type 计数（测试断言用）。"""
        return result["alerts_by_type"]

    @staticmethod
    def has_alert(result: dict, alert_type: str) -> bool:
        """判断结果中是否出现指定 alert type。"""
        return alert_type in result["alerts_by_type"]
