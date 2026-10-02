"""
顶层 AGI 编排器 TopLevelAGICoordinator
========================================

把四个自治闭环（资金侧 / 感知侧 / 运维侧 / 寻优侧）串成一个统一的 AGI 大脑，
实现跨闭环联动，而不重写它们内部的算法。

闭环流程（一次 run_cycle）：
    聚合 Gather  → 联动诊断 Diagnose  → 决策 Act  → 反馈 Reflect

设计约束：
  - 纯编排：只读取四个闭环的 get_stats/get_last_report 既有接口，不复制算法。
  - fail-closed：任一闭环未启用/异常时降级跳过，不放大风险；联动仅产出「低风险
    标记」与「高风险建议」，绝不自动执行改参/暂停等危险动作。
  - 幂等 + 冷却：run_cycle 带最小间隔，冷却期内返回缓存报告。
  - JSON 安全：所有统计经 utils.helpers 的 safe_* 清洗。
  - 可观测：维护联动状态与联动计数，供 Dashboard 观察跨闭环联动效果。

跨闭环联动规则（低风险标记 + 建议）：
  1. 寻优侧上线新参数 → 标记资金侧重评估（funding_rebalance_needed）
  2. 运维侧策略异常告警增加 → 建议暂停相关策略优化（optimization_pause_advised）
  3. 资金侧健康度恶化（F/D）→ 全局告警建议
"""
import copy
import time
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger

from utils.helpers import safe_float, safe_int, safe_finite, safe_div


class TopLevelAGICoordinator:
    """顶层 AGI 编排器：聚合四闭环状态并运行跨闭环联动规则。"""

    def __init__(
        self,
        quant_agi=None,
        perception=None,
        ops_self_heal=None,
        auto_optimization=None,
        learning_memory=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self._quant_agi = quant_agi
        self._perception = perception
        self._ops_self_heal = ops_self_heal
        self._auto_optimization = auto_optimization
        self._learning_memory = learning_memory
        self._config = dict(config or {})

        self._cooldown_seconds = safe_float(self._config.get("cooldown_seconds"), 60.0)
        if self._cooldown_seconds < 0:
            self._cooldown_seconds = 60.0

        # 联动规则阈值：资金集中度 HHI / 闲置资金占比（跨阈值触发）
        self._concentration_high_threshold = safe_float(
            self._config.get("concentration_high_threshold"), 0.6
        )
        self._idle_high_threshold = safe_float(self._config.get("idle_high_threshold"), 0.3)
        self._auto_execute_low_risk = bool(self._config.get("auto_execute_low_risk", True))
        self._max_source_age_seconds = max(
            0.0, safe_float(self._config.get("max_source_age_seconds"), 300.0)
        )
        self._idle_action_cooldown_seconds = max(
            0.0, safe_float(self._config.get("idle_action_cooldown_seconds"), 300.0)
        )
        self._last_idle_action_monotonic: Optional[float] = None

        # 联动状态（跨闭环共享标记，供各闭环循环/人工读取）
        self._linkage_state: Dict[str, Any] = {
            "funding_rebalance_needed": False,
            "optimization_pause_advised": False,
            "optimization_advised": False,
            "health_grade": "N/A",
            "last_update": None,
        }
        # 上次聚合快照（用于检测「增加/变化」）
        self._last: Optional[Dict[str, Any]] = None
        self._last_report: Optional[Dict[str, Any]] = None
        self._last_run_ts: Optional[float] = None
        self._stats: Dict[str, Any] = {
            "total_cycles": 0,
            "linkage_actions": 0,
            "by_type": {},
        }

        logger.info(
            f"TopLevelAGICoordinator initialized: cooldown={self._cooldown_seconds:.0f}s, "
            f"quant_agi={'on' if quant_agi else 'off'}, "
            f"perception={'on' if perception else 'off'}, "
            f"ops_self_heal={'on' if ops_self_heal else 'off'}, "
            f"auto_optimization={'on' if auto_optimization else 'off'}"
        )

    # ─────────────────────────────────────────────────────────────
    # 顶层编排入口
    # ─────────────────────────────────────────────────────────────

    def run_cycle(self) -> Dict[str, Any]:
        """执行一次顶层 AGI 编排：聚合四闭环状态 → 联动诊断 → 决策。"""
        now = time.monotonic()
        if (
            self._last_report is not None
            and self._last_run_ts is not None
            and (now - self._last_run_ts) < self._cooldown_seconds
        ):
            cached = copy.deepcopy(self._last_report)
            cached["cooldown"] = True
            return cached

        self._stats["total_cycles"] = safe_int(self._stats.get("total_cycles"), 0) + 1

        # fail-closed：任一聚合/联动诊断环节异常都降级为空计划，不放大风险
        deferred_actions = []
        try:
            # 1. 聚合 Gather
            state = self._gather()

            # 2. 联动诊断 Diagnose
            actions = self._diagnose_linkage(state)
            actions, deferred_actions = self._apply_decision_gates(state, actions)
            for action in actions:
                self._record_action(action["type"])
        except Exception as e:
            logger.warning(f"[TopAGI] cycle failed (fail-closed): {e}")
            state = {
                "quant_agi": None,
                "perception": None,
                "ops_self_heal": None,
                "auto_optimization": None,
                "shared_learning_memory": None,
            }
            actions = []
            deferred_actions = []

        # 3. 决策 Act（更新联动状态 + 统计）
        self._linkage_state["last_update"] = datetime.now().isoformat()
        analysis = self._build_analysis(state, actions, deferred_actions)

        # 4. 反馈 Reflect
        report = {
            "timestamp": datetime.now().isoformat(),
            "cycle": self._stats["total_cycles"],
            "cooldown": False,
            "linkage_state": copy.deepcopy(self._linkage_state),
            "actions": actions,
            "summary": {
                "quant_agi": self._summarize_quant_agi(state.get("quant_agi")),
                "perception": state.get("perception"),
                "ops_self_heal": state.get("ops_self_heal"),
                "auto_optimization": state.get("auto_optimization"),
                "shared_learning_memory": state.get("shared_learning_memory"),
            },
            "analysis": analysis["steps"],
            "decision_plan": analysis["decision_plan"],
        }
        report = self._sanitize(report)
        self._last_report = copy.deepcopy(report)
        self._last_run_ts = time.monotonic()
        return report

    # ─────────────────────────────────────────────────────────────
    # 1. 聚合 Gather
    # ─────────────────────────────────────────────────────────────

    def _gather(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "quant_agi": None,
            "perception": None,
            "ops_self_heal": None,
            "auto_optimization": None,
            "shared_learning_memory": None,
        }
        if self._learning_memory is not None:
            try:
                state["shared_learning_memory"] = self._learning_memory.get_summary()
            except Exception as e:
                logger.debug(f"[TopAGI] shared learning memory unavailable: {e}")
        if self._quant_agi is not None:
            try:
                state["quant_agi"] = self._quant_agi.get_last_report()
            except Exception as e:
                logger.debug(f"[TopAGI] quant_agi gather failed: {e}")
        if self._perception is not None:
            try:
                state["perception"] = self._perception.get_stats()
            except Exception as e:
                logger.debug(f"[TopAGI] perception gather failed: {e}")
        if self._ops_self_heal is not None:
            try:
                state["ops_self_heal"] = self._ops_self_heal.get_stats()
            except Exception as e:
                logger.debug(f"[TopAGI] ops_self_heal gather failed: {e}")
        if self._auto_optimization is not None:
            try:
                state["auto_optimization"] = self._auto_optimization.get_stats()
            except Exception as e:
                logger.debug(f"[TopAGI] auto_optimization gather failed: {e}")
        return state

    @staticmethod
    def _summarize_quant_agi(report) -> Optional[Dict[str, Any]]:
        if not report:
            return None
        reflection = report.get("reflection") or {}
        perception = report.get("perception") or {}
        contribution = perception.get("contribution") or {}
        plan = (report.get("decision") or {}).get("allocation_plan") or {}
        idle_pct = safe_div(safe_float(plan.get("idle_cash"), 0.0),
                            safe_float(plan.get("total_equity"), 0.0), 0.0)
        return {
            "status": report.get("status"),
            "health_grade": reflection.get("health_grade", "N/A"),
            "health_score": safe_float(reflection.get("health_score"), 0.0),
            "alerts_count": safe_int(reflection.get("alerts_count"), 0),
            "concentration": safe_float(contribution.get("concentration"), 0.0),
            "idle_pct": idle_pct,
        }

    @staticmethod
    def _sanitize(obj: Any) -> Any:
        """递归清洗，保证 json.dumps 无 NaN/Infinity，且不污染调用方数据结构。"""
        if obj is None:
            return None
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, int):
            return int(obj)
        if isinstance(obj, float):
            return safe_finite(obj, 0.0)
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return {str(k): TopLevelAGICoordinator._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [TopLevelAGICoordinator._sanitize(v) for v in obj]
        try:
            return safe_finite(float(obj), 0.0)
        except (TypeError, ValueError):
            return str(obj)

    # ─────────────────────────────────────────────────────────────
    # 2. 联动诊断 Diagnose
    # ─────────────────────────────────────────────────────────────

    def _diagnose_linkage(self, state: Dict[str, Any]) -> list:
        actions = []

        # 当前快照关键指标
        ao = self._as_mapping(state.get("auto_optimization"))
        deployed = safe_int(ao.get("deployed"), 0)

        ops = self._as_mapping(state.get("ops_self_heal"))
        by_root_cause = self._as_mapping(ops.get("by_root_cause"))
        strategy_root_cause = self._as_mapping(by_root_cause.get("strategy"))
        strategy_alerts = safe_int(
            strategy_root_cause.get("alert_only"), 0
        )

        qa_summary = self._summarize_quant_agi(state.get("quant_agi")) or {}
        grade = str(qa_summary.get("health_grade", "N/A"))
        concentration = safe_float(qa_summary.get("concentration"), 0.0)
        idle_pct = safe_float(qa_summary.get("idle_pct"), 0.0)

        # 首次运行：仅初始化基线，不触发联动
        if self._last is None:
            self._last = {
                "deployed": deployed,
                "strategy_alerts": strategy_alerts,
                "grade": None,
                "concentration": concentration,
                "idle_pct": idle_pct,
            }
            self._linkage_state["health_grade"] = grade
            return actions

        # 联动 1：寻优上线新参数 → 资金侧重评估
        if deployed > self._last["deployed"]:
            actions.append({
                "type": "funding_rebalance",
                "level": "info",
                "reason": f"寻优侧上线 {deployed} 次参数变更，建议资金侧重评估分配",
            })
            self._linkage_state["funding_rebalance_needed"] = True

        # 联动 2：运维策略异常告警增加 → 寻优暂停建议
        if strategy_alerts > self._last["strategy_alerts"]:
            actions.append({
                "type": "optimization_pause_advised",
                "level": "warning",
                "reason": "运维侧策略异常告警增加，建议暂停相关策略的自动寻优",
            })
            self._linkage_state["optimization_pause_advised"] = True

        # 联动 3：资金侧健康度恶化（F/D）→ 全局告警建议（仅在「进入」F/D 时触发一次，避免每周期刷屏）
        prev_grade = self._last.get("grade")
        if grade in ("F", "D") and prev_grade != grade:
            actions.append({
                "type": "health_degraded",
                "level": "critical" if grade == "F" else "warning",
                "reason": f"资金侧健康度 {grade}，建议全局审查策略分配",
            })
        self._linkage_state["health_grade"] = grade

        # 联动 4：资金集中度过高（跨阈值）→ 资金侧重评估
        if (concentration > self._concentration_high_threshold
                and self._last.get("concentration", 0.0) <= self._concentration_high_threshold):
            actions.append({
                "type": "funding_rebalance",
                "level": "warning",
                "reason": (
                    f"资金集中度 {concentration:.0%} 超过阈值 "
                    f"{self._concentration_high_threshold:.0%}，建议资金侧重评估分散化"
                ),
            })
            self._linkage_state["funding_rebalance_needed"] = True

        # 联动 5：资金利用率过低（闲置资金过高，跨阈值）→ 寻优触发建议
        now_monotonic = time.monotonic()
        idle_action_due = (
            idle_pct > self._idle_high_threshold
            and (
                self._last_idle_action_monotonic is None
                or now_monotonic - self._last_idle_action_monotonic
                >= self._idle_action_cooldown_seconds
            )
        )
        if idle_action_due:
            self._last_idle_action_monotonic = now_monotonic
            quant_report = self._as_mapping(state.get("quant_agi"))
            source_actions = quant_report.get("actions") or []
            source_time = quant_report.get("timestamp")
            source_is_fresh = False
            try:
                source_dt = datetime.fromisoformat(str(source_time).replace("Z", "+00:00"))
                now_dt = datetime.now(source_dt.tzinfo) if source_dt.tzinfo else datetime.now()
                source_is_fresh = (
                    0 <= (now_dt - source_dt).total_seconds() <= self._max_source_age_seconds
                )
            except (TypeError, ValueError):
                pass
            source_cycle = safe_int(quant_report.get("cycle"), 0)
            source_decision_id = str(quant_report.get("decision_id") or "")
            deploy_actions = []
            for index, source_action in enumerate(source_actions):
                if (not isinstance(source_action, dict)
                        or source_action.get("type") != "idle_cash_deploy"
                        or not str(source_action.get("strategy") or "").strip()):
                    continue
                action = copy.deepcopy(source_action)
                identity = source_decision_id or f"cycle-{source_cycle}"
                action.setdefault("trace_id", f"agi-{identity}-{index}")
                deploy_actions.append(action)
            safe_to_auto_deploy = (
                self._auto_execute_low_risk
                and source_is_fresh
                and grade not in ("D", "F")
            )
            if safe_to_auto_deploy and deploy_actions:
                for action in deploy_actions:
                    action["source"] = "quant_agi"
                    action["level"] = "low"
                    action["reason"] = (
                        f"QuantAGI 已审核的闲置资金建议；闲置占比 {idle_pct:.0%}"
                    )
                    actions.append(action)
            else:
                actions.append({
                    "type": "optimization_advised",
                    "level": "info",
                    "reason": (
                        f"闲置资金占比 {idle_pct:.0%} 超过阈值 "
                        f"{self._idle_high_threshold:.0%}；无新鲜且已审核的低风险部署动作，保持建议模式"
                    ),
                })
            self._linkage_state["optimization_advised"] = True

        self._last = {
            "deployed": deployed,
            "strategy_alerts": strategy_alerts,
            "grade": grade,
            "concentration": concentration,
            "idle_pct": idle_pct,
        }
        return actions

    @staticmethod
    def _as_mapping(value: Any) -> Dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _apply_decision_gates(self, state, actions):
        """抑制与高风险诊断冲突的寻优建议，保留延期原因供审核。"""
        summary = self._summarize_quant_agi(state.get("quant_agi")) or {}
        health_grade = str(summary.get("health_grade", "N/A"))
        pause_advised = any(
            action.get("type") == "optimization_pause_advised" for action in actions
        )
        optimization_blocked = health_grade in ("D", "F") or pause_advised
        if not optimization_blocked:
            return actions, []

        blockers = []
        if health_grade in ("D", "F"):
            blockers.append("health_degraded")
        if pause_advised:
            blockers.append("optimization_pause_advised")

        active_actions = []
        deferred_actions = []
        for action in actions:
            if action.get("type") == "optimization_advised":
                deferred_actions.append({
                    **action,
                    "deferred_reason": "Higher-priority risk finding requires review first",
                    "blocked_by": blockers,
                })
            else:
                active_actions.append(action)
        return active_actions, deferred_actions

    @staticmethod
    def _is_low_risk_auto_executable(action_type: str) -> bool:
        """低风险可自动执行的动作类型白名单。

        - funding_rebalance：触发资金侧权重重评估（不直接改参/暂停策略）
        - idle_cash_deploy：QuantAGI 已审核的闲置资金部署
        其余类型（暂停寻优、健康恶化告警等）一律需人工审批。
        """
        return action_type in ("funding_rebalance", "idle_cash_deploy")

    def _build_analysis(self, state, actions, deferred_actions):
        """Create an evidence-backed, ordered plan; proposed actions are never executed here."""
        quant_summary = self._summarize_quant_agi(state.get("quant_agi")) or {}
        auto_optimization = self._as_mapping(state.get("auto_optimization"))
        ops_self_heal = self._as_mapping(state.get("ops_self_heal"))
        by_root_cause = self._as_mapping(ops_self_heal.get("by_root_cause"))
        strategy_root_cause = self._as_mapping(by_root_cause.get("strategy"))

        observations = {
            "sources_available": {
                "quant_agi": state.get("quant_agi") is not None,
                "perception": state.get("perception") is not None,
                "ops_self_heal": state.get("ops_self_heal") is not None,
                "auto_optimization": state.get("auto_optimization") is not None,
            },
            "health_grade": quant_summary.get("health_grade", "N/A"),
            "concentration": safe_float(quant_summary.get("concentration"), 0.0),
            "concentration_threshold": self._concentration_high_threshold,
            "idle_cash_pct": safe_float(quant_summary.get("idle_pct"), 0.0),
            "idle_cash_threshold": self._idle_high_threshold,
            "deployed_parameter_count": safe_int(auto_optimization.get("deployed"), 0),
            "strategy_alert_count": safe_int(strategy_root_cause.get("alert_only"), 0),
        }
        priorities = {
            "health_degraded": 100,
            "optimization_pause_advised": 90,
            "funding_rebalance": 70,
            "optimization_advised": 40,
        }
        ordered_actions = sorted(
            actions,
            key=lambda action: (
                -priorities.get(action.get("type"), 10),
                action.get("type", ""),
            ),
        )
        plan_steps = [
            {
                "step": index,
                "action": action.get("type"),
                "priority": priorities.get(action.get("type"), 10),
                "reason": action.get("reason", ""),
                "evidence": observations,
                "requires_human_approval": not (
                    self._auto_execute_low_risk
                    and self._is_low_risk_auto_executable(action.get("type", ""))
                ),
                "auto_execute": (
                    self._auto_execute_low_risk
                    and self._is_low_risk_auto_executable(action.get("type", ""))
                ),
            }
            for index, action in enumerate(ordered_actions, start=1)
        ]
        sources_available = observations["sources_available"]
        analysis_steps = [
            {
                "step": "observe",
                "status": "complete" if all(sources_available.values()) else "partial",
                "evidence": observations,
            },
            {
                "step": "diagnose",
                "status": "findings" if actions or deferred_actions else "no_findings",
                "findings": [
                    {"action": action.get("type"), "reason": action.get("reason", "")}
                    for action in ordered_actions + deferred_actions
                ],
            },
            {
                "step": "resolve_conflicts",
                "status": "deferred" if deferred_actions else "clear",
                "deferred_count": len(deferred_actions),
                "deferred_actions": deferred_actions,
            },
            {
                "step": "plan",
                "status": "review_required" if plan_steps else "no_action",
                "planned_action_count": len(plan_steps),
            },
        ]
        return {
            "steps": analysis_steps,
            "decision_plan": {
                "status": (
                    "auto_execute_authorized"
                    if any(step["auto_execute"] for step in plan_steps)
                    else "human_review_required" if plan_steps else "no_action"
                ),
                "auto_execute": any(step["auto_execute"] for step in plan_steps),
                "requires_human_approval": any(step["requires_human_approval"] for step in plan_steps),
                "steps": plan_steps,
                "deferred_actions": deferred_actions,
            },
        }

    def _record_action(self, action_type: str):
        try:
            self._stats["linkage_actions"] = safe_int(self._stats.get("linkage_actions"), 0) + 1
            self._stats["by_type"][action_type] = safe_int(
                self._stats["by_type"].get(action_type), 0
            ) + 1
        except Exception as e:
            logger.debug(f"[TopAGI] record action failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # 可观测
    # ─────────────────────────────────────────────────────────────

    def get_linkage_state(self) -> Dict[str, Any]:
        """返回跨闭环联动状态（深拷贝，不暴露内部引用）。"""
        return copy.deepcopy(self._linkage_state)

    def get_stats(self) -> Dict[str, Any]:
        """返回顶层编排统计（深拷贝）。"""
        return copy.deepcopy(self._stats)

    def get_status(self) -> Dict[str, Any]:
        """返回顶层 AGI 聚合状态（四闭环概览 + 联动状态）。"""
        return {
            "linkage_state": copy.deepcopy(self._linkage_state),
            "stats": copy.deepcopy(self._stats),
        }


__all__ = ["TopLevelAGICoordinator"]
