"""
受限执行通道 RestrictedExecutionChannel
========================================

量化 AGI 产出动作指令后的唯一执行入口（fail-closed）。

原则：
  - 高风险动作（资金重配 / 策略调整 / 仓位变更）绝不自动执行，只排队等待人工确认。
  - 低风险降风险动作仅通过显式注入的 deployer 落地；无执行器时只记录并通知。
  - 任一环节异常 → 拒绝（fail-closed），绝不静默放行。
  - traceID 贯穿：每条动作指令生成/复用 trace_id，EventStore 落库审计。
  - 动作先持久化认领，按稳定 trace_id 去重，避免缓存报告或进程重启造成重复执行。

动作风险分级：
  HIGH（需人工确认）: reallocate / allocation_auto_action / allocation_recommendation
  INFO（仅记录+通知）: alert_action / 其他
"""
import asyncio
import copy
import hashlib
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from core.atomic_writer import atomic_write_json
from utils.helpers import safe_int

# 高风险动作：涉及资金/策略/仓位，绝不自动执行
_HIGH_RISK_TYPES = {"reallocate", "allocation_auto_action", "allocation_recommendation"}

# 低风险自动动作：仅流入 A/B 级核心策略的闲置资金归集、账户级收益落袋平仓、
# 策略参数下调（降杠杆）、策略暂停（停开新仓）、策略恢复（健康度改善后恢复开仓），
# 均属降风险/提效方向，无需人工确认（仍受下游 RiskGate 约束）
_LOW_RISK_AUTO_TYPES = {
    "idle_cash_deploy", "profit_take_close", "param_adjust",
    "strategy_pause", "strategy_resume",
}

_TOOL_RISK_LEVELS = {
    "reallocate": "high",
    "allocation_auto_action": "high",
    "allocation_recommendation": "high",
    "idle_cash_deploy": "low",
    "profit_take_close": "low",
    "param_adjust": "low",
    "strategy_pause": "low",
    "strategy_resume": "low",
}


class RestrictedExecutionChannel:
    """受限执行通道：AGI 动作指令的唯一执行入口（fail-closed）。"""

    def __init__(
        self,
        event_store=None,
        alert_manager=None,
        pending_path: str = "data/agi_pending_actions.json",
        max_pending: int = 200,
        idle_cash_deployer=None,
        autonomous: bool = False,
        reallocate_deployer=None,
        kill_switch_check=None,
        close_position_deployer=None,
        param_adjust_deployer=None,
        strategy_pause_deployer=None,
        strategy_resume_deployer=None,
        action_history_path: Optional[str] = None,
        max_actions_per_report: int = 100,
        always_require_confirmation: Optional[List[str]] = None,
    ):
        self._event_store = event_store
        self._alert_manager = alert_manager
        self._pending_path = pending_path
        self._max_pending = max_pending
        self._idle_cash_deployer = idle_cash_deployer
        self._autonomous = bool(autonomous)
        self._reallocate_deployer = reallocate_deployer
        self._kill_switch_check = kill_switch_check
        self._close_position_deployer = close_position_deployer
        self._param_adjust_deployer = param_adjust_deployer
        self._strategy_pause_deployer = strategy_pause_deployer
        self._strategy_resume_deployer = strategy_resume_deployer
        self._action_history_path = action_history_path or f"{pending_path}.history.json"
        self._max_actions_per_report = max(1, safe_int(max_actions_per_report, 100))
        self._action_history: Dict[str, Dict[str, Any]] = {}
        self._action_history_ready = True
        self._load_action_history()
        # P3: 强制人工确认的动作类型（即使 autonomous=True 也需排队等待确认）
        self._always_require_confirmation = set(always_require_confirmation or ["reallocate"])
        self._stats: Dict[str, int] = {
            "routed": 0, "queued": 0, "notified": 0, "rejected": 0,
            "confirmed": 0, "rejected_manual": 0, "deployed": 0, "skipped": 0,
        }

    async def route(self, report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """路由 AGI 报告中的动作指令，返回本轮执行结果统计（JSON 安全）。

        高风险动作 → 排队等待人工确认（写 pending 文件 + EventStore 事件）。
        低风险动作 → 受限 deployer 执行或降级为通知，并持久化逐动作回执。
        全程 fail-closed：任何异常被捕获，不向外抛出，不静默放行资金动作。
        """
        if not report or not isinstance(report, dict):
            return {"status": "no_report", "queued": 0, "notified": 0, "rejected": 0}

        if report.get("status") == "cooldown":
            return {
                "status": "cached_report",
                "cycle": safe_int(report.get("cycle"), 0),
                "decision_id": str(report.get("decision_id") or ""),
                "queued": 0,
                "notified": 0,
                "deployed": 0,
                "rejected": 0,
                "skipped": 0,
                "action_results": [],
            }

        if report.get("status") == "fail_closed":
            return {
                "status": "fail_closed",
                "queued": 0,
                "notified": 0,
                "deployed": 0,
                "rejected": 0,
                "skipped": 0,
                "action_results": [],
            }

        actions = report.get("actions", [])
        if not isinstance(actions, list):
            logger.warning("[RESTRICTED-EXEC] invalid action list; report rejected")
            return {
                "status": "invalid_report",
                "queued": 0,
                "notified": 0,
                "deployed": 0,
                "rejected": 1,
                "skipped": 0,
                "action_results": [],
            }
        if not actions:
            return {"status": "no_actions", "queued": 0, "notified": 0, "rejected": 0}
        if len(actions) > self._max_actions_per_report:
            logger.warning(
                f"[RESTRICTED-EXEC] action count {len(actions)} exceeds "
                f"limit {self._max_actions_per_report}; report rejected"
            )
            return {
                "status": "action_limit_exceeded",
                "queued": 0,
                "notified": 0,
                "deployed": 0,
                "rejected": len(actions),
                "skipped": 0,
                "action_results": [],
            }

        cycle = safe_int(report.get("cycle"), 0)
        decision_id = str(report.get("decision_id") or "")
        ts = datetime.now()
        queued: List[Dict[str, Any]] = []
        queued_records: List[Dict[str, Any]] = []
        action_results: List[Dict[str, Any]] = []
        notified = 0
        deployed = 0
        rejected = 0
        skipped = 0

        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                rejected += 1
                action_results.append({
                    "trace_id": "",
                    "type": "invalid",
                    "status": "rejected",
                    "reason": "action_must_be_an_object",
                })
                continue
            atype = str(action.get("type") or "unknown")
            trace_id = str(
                action.get("trace_id") or self._gen_trace_id(cycle, i, decision_id)
            )

            if not self._action_history_ready:
                rejected += 1
                action_results.append({
                    "trace_id": trace_id,
                    "type": atype,
                    "status": "rejected",
                    "reason": "action_history_unavailable",
                })
                continue
            if trace_id in self._action_history:
                prior = self._action_history[trace_id]
                fingerprint = self._action_fingerprint(action)
                if prior.get("fingerprint") and prior["fingerprint"] != fingerprint:
                    rejected += 1
                    action_results.append({
                        "trace_id": trace_id,
                        "type": atype,
                        "status": "rejected",
                        "reason": "trace_id_conflict",
                    })
                    continue
                skipped += 1
                prior_result = prior.get("result")
                action_results.append({
                    "trace_id": trace_id,
                    "type": atype,
                    "status": "duplicate",
                    "previous_status": str(prior.get("status") or "unknown"),
                    "result": copy.deepcopy(
                        prior_result.get("result", prior_result)
                        if isinstance(prior_result, dict) else prior_result
                    ),
                })
                continue

            validation_error = self._validate_action(
                atype, action, autonomous=self._autonomous
            )
            if validation_error:
                rejected += 1
                result = {
                    "trace_id": trace_id,
                    "type": atype,
                    "status": "rejected",
                    "reason": validation_error,
                }
                action_results.append(result)
                continue

            started = {
                "trace_id": trace_id,
                "type": atype,
                "cycle": cycle,
                "decision_id": decision_id,
                "status": "processing",
                "timestamp": ts.isoformat(),
                "action": copy.deepcopy(action),
                "fingerprint": self._action_fingerprint(action),
            }
            if not self._save_action_record(started):
                rejected += 1
                action_results.append({
                    "trace_id": trace_id,
                    "type": atype,
                    "status": "rejected",
                    "reason": "could_not_persist_action_claim",
                })
                continue

            was_queued = False
            try:
                if atype in _HIGH_RISK_TYPES:
                    # P3: 强制人工确认的动作类型，即使 autonomous=True 也需排队等待确认
                    if atype in self._always_require_confirmation:
                        item = self._build_pending_item(action, atype, trace_id, cycle, ts)
                        queued.append(item)
                        queued_records.append(started)
                        was_queued = True
                        self._publish_event("AGI_ACTION_QUEUED", item, trace_id)
                        outcome = {"status": "queued", "reason": "requires_human_confirmation"}
                        logger.info(
                            f"[RESTRICTED-EXEC] {atype} queued for human confirmation "
                            f"(always_require_confirmation): {action.get('strategy') or action.get('detail') or '?'}"
                        )
                    elif self._autonomous:
                        # 完全自主模式：自动执行，仅受全局 Kill Switch 熔断约束
                        if self._kill_switch_active():
                            rejected += 1
                            outcome = {"status": "rejected", "reason": "kill_switch_active"}
                            logger.warning(
                                f"[RESTRICTED-EXEC] kill_switch active, reject autonomous "
                                f"{atype}: {action.get('strategy') or action.get('detail') or '?'}"
                            )
                        elif atype == "reallocate":
                            result = await self._deploy_reallocate(action)
                            if result.get("deployed"):
                                deployed += 1
                                outcome = {"status": "deployed", "result": result}
                                self._publish_event(
                                    "AGI_ACTION_DEPLOYED", {"action": action},
                                    trace_id, suffix="-autonomous",
                                )
                            else:
                                # 落地失败降级为通知，绝不静默放行
                                await self._notify(action, atype)
                                notified += 1
                                outcome = {
                                    "status": "notified",
                                    "reason": str(result.get("reason") or "deployment_not_completed"),
                                }
                        else:
                            # allocation_auto_action / allocation_recommendation：
                            # dynamic_allocator 已执行动作的「通知」，自动记录即可
                            await self._notify(action, atype)
                            notified += 1
                            outcome = {"status": "notified", "reason": "recommendation"}
                    else:
                        item = self._build_pending_item(action, atype, trace_id, cycle, ts)
                        queued.append(item)
                        queued_records.append(started)
                        was_queued = True
                        self._publish_event("AGI_ACTION_QUEUED", item, trace_id)
                        outcome = {"status": "queued"}
                elif atype in _LOW_RISK_AUTO_TYPES:
                    # 低风险自动动作：按类型分发到对应 deployer 落地（无 deployer 则降级为仅通知），绝不排队
                    if atype == "idle_cash_deploy" and self._kill_switch_active():
                        rejected += 1
                        outcome = {"status": "rejected", "reason": "kill_switch_active"}
                    else:
                        if atype == "profit_take_close":
                            result = await self._deploy_close_position(action)
                        elif atype == "param_adjust":
                            result = await self._deploy_param_adjust(action)
                        elif atype == "strategy_pause":
                            result = await self._deploy_strategy_pause(action)
                        elif atype == "strategy_resume":
                            result = await self._deploy_strategy_resume(action)
                        else:
                            result = await self._deploy_idle_cash(action)
                        if result.get("deployed"):
                            deployed += 1
                            outcome = {"status": "deployed", "result": result}
                            self._publish_event("AGI_ACTION_DEPLOYED", {"action": action},
                                                trace_id, suffix="-deployed")
                        else:
                            notified += 1
                            outcome = {
                                "status": "notified",
                                "reason": str(result.get("reason") or "no_deployer"),
                            }
                else:
                    await self._notify(action, atype)
                    notified += 1
                    outcome = {"status": "notified", "reason": "notification"}
            except Exception as e:
                # fail-closed：单条动作异常只拒绝该条，不影响其它动作，但绝不执行
                rejected += 1
                outcome = {"status": "rejected", "reason": str(e)[:200]}
                logger.warning(f"[RESTRICTED-EXEC] route action failed (rejected): {atype}: {e}")

            if not was_queued:
                result = {
                    "trace_id": trace_id,
                    "type": atype,
                    **outcome,
                }
                action_results.append(result)
                self._finish_action_record(trace_id, result)

        # 持久化待确认队列（有高风险动作时才写）
        if queued:
            if self._persist_pending(queued, ts):
                await self._notify_queue_summary(queued)
            else:
                rejected += len(queued)
                queued_ids = {item["trace_id"] for item in queued}
                queued = []
                for record in queued_records:
                    if record["trace_id"] in queued_ids:
                        result = {
                            "trace_id": record["trace_id"],
                            "type": record["type"],
                            "status": "rejected",
                            "reason": "could_not_persist_pending_action",
                        }
                        action_results.append(result)
                        self._finish_action_record(record["trace_id"], result)

        queued_ids = {item["trace_id"] for item in queued}
        for record in queued_records:
            if record["trace_id"] in queued_ids:
                result = {
                    "trace_id": record["trace_id"],
                    "type": record["type"],
                    "status": "queued",
                }
                action_results.append(result)
                self._finish_action_record(record["trace_id"], result)

        self._stats["routed"] += len(actions)
        self._stats["queued"] += len(queued)
        self._stats["notified"] += notified
        self._stats["deployed"] += deployed
        self._stats["rejected"] += rejected
        self._stats["skipped"] += skipped

        logger.info(
            f"[RESTRICTED-EXEC] cycle={cycle} routed={len(actions)} queued={len(queued)} "
            f"notified={notified} deployed={deployed} rejected={rejected}"
        )
        return {
            "status": "ok",
            "queued": len(queued),
            "notified": notified,
            "deployed": deployed,
            "rejected": rejected,
            "skipped": skipped,
            "cycle": cycle,
            "decision_id": decision_id,
            "action_results": action_results,
        }

    # ── 内部辅助 ──────────────────────────────────────────────

    @staticmethod
    def _gen_trace_id(cycle: int, index: int, decision_id: str = "") -> str:
        identity = decision_id or f"cycle-{cycle}"
        return f"agi-{identity}-{index}"

    @staticmethod
    def _action_fingerprint(action: Dict[str, Any]) -> str:
        payload = json.dumps(action, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_action(
        atype: str, action: Dict[str, Any], autonomous: bool = False
    ) -> str:
        """Validate executable tool arguments before claiming or dispatching them."""
        if atype not in _TOOL_RISK_LEVELS:
            return ""
        if atype in ("reallocate", "idle_cash_deploy", "strategy_pause", "strategy_resume"):
            if not str(action.get("strategy") or "").strip():
                return "missing_strategy"
        if atype == "reallocate" and autonomous:
            target = action.get("target_allocation")
            try:
                target = float(target)
            except (TypeError, ValueError):
                return "invalid_target_allocation"
            if not math.isfinite(target) or not 0.0 <= target <= 1.0:
                return "invalid_target_allocation"
        elif atype == "profit_take_close":
            try:
                close_ratio = float(action.get("close_ratio"))
            except (TypeError, ValueError):
                return "invalid_close_ratio"
            if not math.isfinite(close_ratio) or not 0.0 < close_ratio <= 1.0:
                return "invalid_close_ratio"
        elif atype == "param_adjust":
            if not (
                str(action.get("strategy") or "").strip()
                or str(action.get("symbol") or "").strip()
            ):
                return "missing_strategy_or_symbol"
            try:
                value = float(action.get("value"))
            except (TypeError, ValueError):
                return "invalid_value"
            if not math.isfinite(value):
                return "invalid_value"
        return ""

    @staticmethod
    def _build_pending_item(action: Dict[str, Any], atype: str, trace_id: str,
                            cycle: int, ts: datetime) -> Dict[str, Any]:
        return {
            "trace_id": trace_id,
            "type": atype,
            "cycle": cycle,
            "timestamp": ts.isoformat(),
            "status": "pending",  # pending / confirmed / rejected（人工在 dashboard 或脚本处理）
            "action": dict(action),
        }

    def _publish_event(self, event_type: str, item: Dict[str, Any], trace_id: str,
                       suffix: str = "") -> None:
        if self._event_store is None:
            return
        try:
            self._event_store.append(
                event_type=event_type,
                data={"trace_id": trace_id, "action": item.get("action", {})},
                event_id=f"{trace_id}{suffix}",  # 幂等：同一 trace_id(+suffix) 只写一次
                source="agi_orchestrator",
                symbol=item.get("action", {}).get("symbol", ""),
            )
        except Exception as e:
            logger.debug(f"[RESTRICTED-EXEC] event publish failed: {e}")

    async def _notify(self, action: Dict[str, Any], atype: str) -> None:
        if self._alert_manager is None:
            return
        level = str(action.get("level") or "info").lower()
        severity = "CRITICAL" if level == "critical" else ("WARNING" if level == "warning" else "INFO")
        message = str(action.get("detail") or action.get("message") or atype)
        try:
            await self._alert_manager.send_alert(
                alert_type=f"agi_{atype}",
                message=message,
                severity=severity,
                symbol=str(action.get("symbol") or action.get("strategy") or ""),
                metadata={"action": action},
            )
        except Exception as e:
            logger.debug(f"[RESTRICTED-EXEC] notify failed: {e}")

    async def _deploy_idle_cash(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """落地低风险闲置资金归集动作。

        无 deployer 时返回未部署（调用方降级为仅通知）；deployer 异常向上抛出，
        由 route 的 fail-closed 分支计数为 rejected，绝不静默放行资金动作。
        """
        if self._idle_cash_deployer is None:
            return {"deployed": False, "reason": "no_deployer"}
        result = await self._idle_cash_deployer(action)
        if isinstance(result, dict):
            return result
        return {"deployed": bool(result)}

    async def _deploy_close_position(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """落地账户级收益落袋平仓动作（profit_take_close）。

        无 close_position_deployer 时返回未部署（调用方降级为仅通知）；deployer 异常
        向上抛出，由 route 的 fail-closed 分支计数为 rejected，绝不静默放行平仓动作。
        """
        if self._close_position_deployer is None:
            return {"deployed": False, "reason": "no_close_deployer"}
        result = await self._close_position_deployer(action)
        if isinstance(result, dict):
            return result
        return {"deployed": bool(result)}

    async def _deploy_param_adjust(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """落地策略参数下调动作（param_adjust，如降杠杆）。

        无 param_adjust_deployer 时返回未部署（调用方降级为仅通知）；deployer 异常
        向上抛出，由 route 的 fail-closed 分支计数为 rejected，绝不静默放行参数动作。
        """
        if self._param_adjust_deployer is None:
            return {"deployed": False, "reason": "no_param_adjust_deployer"}
        result = await self._param_adjust_deployer(action)
        if isinstance(result, dict):
            return result
        return {"deployed": bool(result)}

    async def _deploy_strategy_pause(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """落地策略暂停动作（strategy_pause，停开新仓）。

        无 strategy_pause_deployer 时返回未部署（调用方降级为仅通知）；deployer 异常
        向上抛出，由 route 的 fail-closed 分支计数为 rejected，绝不静默放行。
        """
        if self._strategy_pause_deployer is None:
            return {"deployed": False, "reason": "no_strategy_pause_deployer"}
        result = await self._strategy_pause_deployer(action)
        if isinstance(result, dict):
            return result
        return {"deployed": bool(result)}

    async def _deploy_strategy_resume(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """落地策略恢复动作（strategy_resume，健康度改善后恢复开仓）。

        无 strategy_resume_deployer 时返回未部署（调用方降级为仅通知）；deployer 异常
        向上抛出，由 route 的 fail-closed 分支计数为 rejected，绝不静默放行。
        """
        if self._strategy_resume_deployer is None:
            return {"deployed": False, "reason": "no_strategy_resume_deployer"}
        result = await self._strategy_resume_deployer(action)
        if isinstance(result, dict):
            return result
        return {"deployed": bool(result)}

    def _kill_switch_active(self) -> bool:
        """查询全局 Kill Switch 熔断状态。

        完全自主模式下，资金动作落地前必须过这一道关：熔断则拒绝一切资金动作。
        检查回调缺失时视为未熔断（False）；检查本身异常时 fail-closed 视为熔断（True），
        绝不在无法确认安全的情况下放行。
        """
        if self._kill_switch_check is None:
            return False
        try:
            return bool(self._kill_switch_check())
        except Exception as e:
            logger.warning(f"[RESTRICTED-EXEC] kill_switch check failed (fail-closed): {e}")
            return True

    async def _deploy_reallocate(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """落地高风险 reallocate 动作（完全自主模式专用）。

        无 reallocate_deployer 时返回未部署（调用方降级为通知）；deployer 异常向上抛出，
        由 route 的 fail-closed 分支计数为 rejected，绝不静默放行资金动作。
        """
        if self._reallocate_deployer is None:
            return {"deployed": False, "reason": "no_reallocate_deployer"}
        result = await self._reallocate_deployer(action)
        if isinstance(result, dict):
            return result
        return {"deployed": bool(result)}

    async def _notify_queue_summary(self, queued: List[Dict[str, Any]]) -> None:
        if self._alert_manager is None:
            return
        summary = "; ".join(
            f"{q['type']}:{q['action'].get('strategy') or q['action'].get('detail') or '?'}"
            for q in queued[:5]
        )
        try:
            await self._alert_manager.send_alert(
                alert_type="agi_manual_confirmation_required",
                message=f"AGI 产生 {len(queued)} 条高风险动作，等待人工确认: {summary}",
                severity="WARNING",
            )
        except Exception as e:
            logger.debug(f"[RESTRICTED-EXEC] summary notify failed: {e}")

    def _load_pending(self) -> List[Dict[str, Any]]:
        """读取待确认队列（文件缺失/损坏时返回空列表，fail-closed）。"""
        try:
            if not os.path.exists(self._pending_path):
                return []
            with open(self._pending_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            return loaded if isinstance(loaded, list) else []
        except Exception as e:
            logger.warning(f"[RESTRICTED-EXEC] load pending failed (fail-closed): {e}")
            return []

    def _save_pending(self, items: List[Dict[str, Any]]) -> bool:
        """原子覆盖待确认队列；失败时不报告已排队。"""
        if atomic_write_json(self._pending_path, items):
            return True
        logger.error(f"[RESTRICTED-EXEC] save pending failed (fail-closed): {self._pending_path}")
        return False

    def _persist_pending(self, new_items: List[Dict[str, Any]], ts: datetime) -> bool:
        """追加待确认队列到 JSON 文件（有界 FIFO，最多保留 max_pending 条）。"""
        existing = self._load_pending()
        known = {str(item.get("trace_id")) for item in existing if isinstance(item, dict)}
        merged = list(existing)
        for item in new_items:
            trace_id = str(item.get("trace_id"))
            if trace_id not in known:
                merged.append(item)
                known.add(trace_id)
        if len(merged) > self._max_pending:
            merged = merged[-self._max_pending:]
        return self._save_pending(merged)

    def _load_action_history(self) -> None:
        """恢复已路由动作账本；损坏时禁用动作执行，防止重放不确定动作。"""
        if not os.path.exists(self._action_history_path):
            return
        try:
            with open(self._action_history_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("action history must be a list")
            self._action_history = {
                str(item["trace_id"]): item
                for item in data
                if isinstance(item, dict) and item.get("trace_id")
            }
        except Exception as e:
            self._action_history_ready = False
            logger.error(
                f"[RESTRICTED-EXEC] action history unavailable (fail-closed): {e}"
            )

    def _save_action_record(self, record: Dict[str, Any]) -> bool:
        previous = self._action_history.get(record["trace_id"])
        self._action_history[record["trace_id"]] = copy.deepcopy(record)
        records = list(self._action_history.values())[-max(1, self._max_pending * 5):]
        if atomic_write_json(self._action_history_path, records):
            self._action_history = {item["trace_id"]: item for item in records}
            return True
        if previous is None:
            self._action_history.pop(record["trace_id"], None)
        else:
            self._action_history[record["trace_id"]] = previous
        logger.error(
            f"[RESTRICTED-EXEC] action history persist failed (fail-closed): "
            f"{self._action_history_path}"
        )
        return False

    def _finish_action_record(self, trace_id: str, result: Dict[str, Any]) -> bool:
        record = self._action_history.get(trace_id)
        if record is None:
            return False
        updated = copy.deepcopy(record)
        updated.update({
            "status": str(result.get("status") or "unknown"),
            "result": copy.deepcopy(result),
            "updated_at": datetime.now().isoformat(),
        })
        return self._save_action_record(updated)

    def get_action_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent routed action outcomes, newest first."""
        limit = max(1, min(1000, int(limit)))
        records = list(self._action_history.values())[-limit:]
        return [copy.deepcopy(item) for item in reversed(records)]

    def get_tool_catalog(self) -> Dict[str, Any]:
        """Expose available tool contracts and execution policy to planners."""
        deployers = {
            "reallocate": self._reallocate_deployer,
            "idle_cash_deploy": self._idle_cash_deployer,
            "profit_take_close": self._close_position_deployer,
            "param_adjust": self._param_adjust_deployer,
            "strategy_pause": self._strategy_pause_deployer,
            "strategy_resume": self._strategy_resume_deployer,
        }
        tools = {}
        for name, risk_level in _TOOL_RISK_LEVELS.items():
            registered = name in ("allocation_auto_action", "allocation_recommendation")
            if name in deployers:
                registered = deployers[name] is not None
            tools[name] = {
                "risk": risk_level,
                "available": registered or (
                    risk_level == "high" and not self._autonomous
                ),
                "execution": (
                    "manual_confirmation" if risk_level == "high" and not self._autonomous
                    else "kill_switch_gated" if risk_level == "high"
                    else "deployer" if registered
                    else "notification_only"
                ),
            }
        return {
            "autonomous": self._autonomous,
            "action_history_ready": self._action_history_ready,
            "max_actions_per_report": self._max_actions_per_report,
            "tools": tools,
        }

    # ── 人工确认 / 拒绝 ───────────────────────────────────────

    def list_pending(self, status: str = "pending") -> List[Dict[str, Any]]:
        """返回匹配指定状态的动作条目（默认仅 pending），供 dashboard / 脚本人工审查。"""
        items = self._load_pending()
        if not status:
            return items
        return [it for it in items if it.get("status") == status]

    def confirm(self, trace_id: str) -> Dict[str, Any]:
        """人工确认一条高风险动作：status → confirmed，落盘并审计。

        fail-closed：trace_id 为空或未找到对应条目时返回 success=False，绝不伪造确认。
        """
        return self._resolve(trace_id, "confirmed")

    def reject(self, trace_id: str, reason: str = "") -> Dict[str, Any]:
        """人工拒绝一条高风险动作：status → rejected，落盘并审计。

        fail-closed：trace_id 为空或未找到对应条目时返回 success=False。
        """
        return self._resolve(trace_id, "rejected", reason)

    def _resolve(self, trace_id: str, status: str, reason: str = "") -> Dict[str, Any]:
        if not trace_id:
            return {"success": False, "error": "trace_id required"}
        if status not in ("confirmed", "rejected"):
            return {"success": False, "error": "invalid status"}

        items = self._load_pending()
        target = next((it for it in items if it.get("trace_id") == trace_id), None)
        if target is None:
            return {"success": False, "error": "trace_id not found"}

        target["status"] = status
        target[f"{status}_at"] = datetime.now().isoformat()
        if status == "rejected":
            target["rejected_reason"] = str(reason or "")[:500]
        if not self._save_pending(items):
            return {"success": False, "error": "failed to persist action status"}

        event_type = "AGI_ACTION_CONFIRMED" if status == "confirmed" else "AGI_ACTION_REJECTED"
        self._publish_event(event_type, target, trace_id, suffix=f"-{status}")
        stat_key = "confirmed" if status == "confirmed" else "rejected_manual"
        self._stats[stat_key] = self._stats.get(stat_key, 0) + 1

        logger.info(f"[RESTRICTED-EXEC] action {status}: trace_id={trace_id}")
        return {"success": True, "trace_id": trace_id, "status": status}

    def get_stats(self) -> Dict[str, Any]:
        """返回通道统计（深拷贝）。"""
        return dict(self._stats)


__all__ = ["RestrictedExecutionChannel"]
