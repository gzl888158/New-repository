"""自动恢复模块：定义恢复状态机，负责系统异常后的自动降级与恢复调度。"""
import asyncio
import json
import os
from datetime import datetime, timedelta
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Callable
from loguru import logger
from enum import Enum


class RecoveryState(Enum):
    IDLE = "idle"
    DETECTING = "detecting"
    RECOVERING = "recovering"
    RECOVERED = "recovered"
    FAILED = "failed"
    MANUAL_INTERVENTION = "manual_intervention"


class FailureType(Enum):
    NETWORK = "network"
    API = "api"
    REDIS = "redis"
    DATABASE = "database"
    STRATEGY = "strategy"
    WEBSOCKET = "websocket"
    ORDER_EXECUTION = "order_execution"
    POSITION_SYNC = "position_sync"


class RecoveryAction(Enum):
    RETRY = "retry"
    RECONNECT = "reconnect"
    RESET_STATE = "reset_state"
    FALLBACK = "fallback"
    RESTART_SERVICE = "restart_service"
    MANUAL = "manual"


class RecoveryAttempt:
    def __init__(self, failure_type: FailureType, action: RecoveryAction, attempt: int):
        self.failure_type = failure_type
        self.action = action
        self.attempt = attempt
        self.start_time = datetime.now()
        self.end_time = None
        self.success = False
        self.error = None
        self.duration_ms = 0

    def complete(self, success: bool, error: str = None):
        self.end_time = datetime.now()
        self.success = success
        self.error = error
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000


class RecoveryPolicy:
    def __init__(self, failure_type: FailureType, max_attempts: int = 3, 
                 base_delay_seconds: int = 5, max_delay_seconds: int = 60,
                 actions: List[RecoveryAction] = None):
        self.failure_type = failure_type
        self.max_attempts = max_attempts
        self.base_delay_seconds = base_delay_seconds
        self.max_delay_seconds = max_delay_seconds
        self.actions = actions or [RecoveryAction.RETRY]
    
    def get_delay(self, attempt: int) -> int:
        delay = self.base_delay_seconds * (2 ** (attempt - 1))
        return min(delay, self.max_delay_seconds)


class AutoRecovery:
    def __init__(self, config: Dict[str, Any], alert_manager):
        self.config = config
        self.alert_manager = alert_manager
        
        self._state = RecoveryState.IDLE
        self._current_failure: Optional[FailureType] = None
        self._recovery_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._policy_registry: Dict[FailureType, RecoveryPolicy] = {}
        self._action_handlers: Dict[str, Callable] = {}
        self._recovery_lock = asyncio.Lock()
        self._last_recovery_time = None
        
        self._policies = {
            FailureType.NETWORK: RecoveryPolicy(
                FailureType.NETWORK,
                max_attempts=5,
                base_delay_seconds=3,
                max_delay_seconds=30,
                actions=[RecoveryAction.RECONNECT, RecoveryAction.RETRY]
            ),
            FailureType.API: RecoveryPolicy(
                FailureType.API,
                max_attempts=5,
                base_delay_seconds=5,
                max_delay_seconds=60,
                actions=[RecoveryAction.RETRY, RecoveryAction.RECONNECT]
            ),
            FailureType.REDIS: RecoveryPolicy(
                FailureType.REDIS,
                max_attempts=3,
                base_delay_seconds=5,
                max_delay_seconds=30,
                actions=[RecoveryAction.RECONNECT, RecoveryAction.FALLBACK]
            ),
            FailureType.DATABASE: RecoveryPolicy(
                FailureType.DATABASE,
                max_attempts=3,
                base_delay_seconds=10,
                max_delay_seconds=60,
                actions=[RecoveryAction.RETRY, RecoveryAction.RESTART_SERVICE]
            ),
            FailureType.STRATEGY: RecoveryPolicy(
                FailureType.STRATEGY,
                max_attempts=3,
                base_delay_seconds=10,
                max_delay_seconds=60,
                actions=[RecoveryAction.RESET_STATE, RecoveryAction.RESTART_SERVICE]
            ),
            FailureType.WEBSOCKET: RecoveryPolicy(
                FailureType.WEBSOCKET,
                max_attempts=5,
                base_delay_seconds=3,
                max_delay_seconds=30,
                actions=[RecoveryAction.RECONNECT, RecoveryAction.RETRY]
            ),
            FailureType.ORDER_EXECUTION: RecoveryPolicy(
                FailureType.ORDER_EXECUTION,
                max_attempts=3,
                base_delay_seconds=2,
                max_delay_seconds=15,
                actions=[RecoveryAction.RETRY, RecoveryAction.MANUAL]
            ),
            FailureType.POSITION_SYNC: RecoveryPolicy(
                FailureType.POSITION_SYNC,
                max_attempts=3,
                base_delay_seconds=5,
                max_delay_seconds=30,
                actions=[RecoveryAction.RETRY, RecoveryAction.RESET_STATE]
            ),
        }
        
        self._dependencies = {}
    
    def set_dependencies(self, **kwargs):
        self._dependencies.update(kwargs)
    
    def register_action_handler(self, action_type: str, handler: Callable):
        self._action_handlers[action_type] = handler
    
    async def detect_and_recover(self, failure_type: FailureType, 
                                details: Dict[str, Any] = None) -> bool:
        async with self._recovery_lock:
            if self._state == RecoveryState.RECOVERING:
                logger.info(f"Already recovering from {self._current_failure}, skipping {failure_type}")
                return False
            
            self._state = RecoveryState.DETECTING
            self._current_failure = failure_type
            
            await self._notify_failure(failure_type, details)
            
            policy = self._get_policy(failure_type)
            if not policy:
                logger.warning(f"No recovery policy for {failure_type}")
                self._state = RecoveryState.IDLE
                return False
            
            self._state = RecoveryState.RECOVERING
            success = await self._execute_recovery(policy, failure_type, details)
            
            if success:
                self._state = RecoveryState.RECOVERED
                await self._notify_recovery_success(failure_type)
            else:
                self._state = RecoveryState.FAILED
                await self._notify_recovery_failure(failure_type)
            
            self._last_recovery_time = datetime.now()
            
            await asyncio.sleep(5)
            self._state = RecoveryState.IDLE
            
            return success
    
    def _get_policy(self, failure_type: FailureType) -> Optional[RecoveryPolicy]:
        return self._policies.get(failure_type)
    
    async def _execute_recovery(self, policy: RecoveryPolicy, failure_type: FailureType,
                               details: Dict[str, Any]) -> bool:
        for attempt in range(1, policy.max_attempts + 1):
            delay = policy.get_delay(attempt)
            logger.info(f"Recovery attempt {attempt}/{policy.max_attempts} for {failure_type}, "
                        f"delay={delay}s, actions={[a.value for a in policy.actions]}")
            
            await asyncio.sleep(delay)
            
            for action in policy.actions:
                attempt_record = RecoveryAttempt(failure_type, action, attempt)
                
                try:
                    success = await self._execute_action(action, failure_type, details)
                    attempt_record.complete(success)
                    self._record_recovery(attempt_record)
                    
                    if success:
                        logger.info(f"Recovery successful on attempt {attempt} with {action}")
                        return True
                except Exception as e:
                    attempt_record.complete(False, str(e))
                    self._record_recovery(attempt_record)
                    logger.error(f"Recovery action {action} failed: {e}")
        
        logger.error(f"Recovery failed after {policy.max_attempts} attempts for {failure_type}")
        return False
    
    async def _execute_action(self, action: RecoveryAction, failure_type: FailureType,
                             details: Dict[str, Any]) -> bool:
        handler = self._action_handlers.get(action.value)
        
        if handler:
            try:
                return await handler(failure_type, details)
            except Exception as e:
                logger.error(f"Action handler for {action} failed: {e}")
                return False
        
        return await self._default_action(action, failure_type, details)
    
    async def _default_action(self, action: RecoveryAction, failure_type: FailureType,
                             details: Dict[str, Any]) -> bool:
        if action == RecoveryAction.RETRY:
            return await self._action_retry(failure_type, details)
        elif action == RecoveryAction.RECONNECT:
            return await self._action_reconnect(failure_type, details)
        elif action == RecoveryAction.RESET_STATE:
            return await self._action_reset_state(failure_type, details)
        elif action == RecoveryAction.FALLBACK:
            return await self._action_fallback(failure_type, details)
        elif action == RecoveryAction.RESTART_SERVICE:
            return await self._action_restart_service(failure_type, details)
        
        return False
    
    async def _action_retry(self, failure_type: FailureType, details: Dict[str, Any]) -> bool:
        okx_client = self._dependencies.get("okx_client")
        if okx_client:
            try:
                await asyncio.to_thread(okx_client.get_account_info)
                return True
            except Exception:
                return False
        return False
    
    async def _action_reconnect(self, failure_type: FailureType, details: Dict[str, Any]) -> bool:
        okx_client = self._dependencies.get("okx_client")
        if okx_client:
            try:
                await asyncio.to_thread(okx_client.reconnect)
                return True
            except Exception:
                return False
        return False
    
    async def _action_reset_state(self, failure_type: FailureType, details: Dict[str, Any]) -> bool:
        strategy_coordinator = self._dependencies.get("strategy_coordinator")
        if not strategy_coordinator:
            return False
        try:
            from services.strategy_coordinator import StrategyState

            def _is_paused(name: str) -> bool:
                state = strategy_coordinator.get_strategy_state(name)
                return getattr(state, "value", state) == "paused"

            strategy_name = details.get("strategy_name")
            if strategy_name:
                # 防御性检查：人工暂停的策略不应被自动 reset 误恢复
                if _is_paused(strategy_name):
                    logger.warning(f"Skip reset: strategy '{strategy_name}' is paused, not auto-recovering")
                    return False
                # StrategyCoordinator 无 reset_strategy；reset 语义 = 清除错误状态回到 IDLE
                strategy_coordinator.set_strategy_state(strategy_name, StrategyState.IDLE)
                return True
            else:
                reset_count = 0
                for name in list(strategy_coordinator.get_all_strategy_states().keys()):
                    if _is_paused(name):
                        continue
                    strategy_coordinator.set_strategy_state(name, StrategyState.IDLE)
                    reset_count += 1
                return reset_count > 0
        except Exception as e:
            logger.error(f"Reset state action failed: {e}")
            return False
    
    async def _action_fallback(self, failure_type: FailureType, details: Dict[str, Any]) -> bool:
        redis_cache = self._dependencies.get("redis_cache")
        if redis_cache:
            try:
                redis_cache._redis_available = False
                logger.info("Redis fallback to memory cache mode")
                return True
            except Exception:
                return False
        return False
    
    async def _action_restart_service(self, failure_type: FailureType, details: Dict[str, Any]) -> bool:
        logger.warning(f"Manual restart required for {failure_type}")
        return False
    
    def _record_recovery(self, attempt: RecoveryAttempt):
        key = attempt.failure_type.value
        self._recovery_history[key].append({
            "attempt": attempt.attempt,
            "action": attempt.action.value,
            "success": attempt.success,
            "error": attempt.error,
            "duration_ms": round(attempt.duration_ms, 2),
            "timestamp": attempt.start_time.isoformat()
        })
    
    async def _notify_failure(self, failure_type: FailureType, details: Dict[str, Any]):
        message = f"🔴 故障检测：{failure_type.value}\n详情：{json.dumps(details, ensure_ascii=False) if details else '无'}"
        await self.alert_manager.send_system_alert("FAILURE_DETECTED", message)
    
    async def _notify_recovery_success(self, failure_type: FailureType):
        message = f"🟢 自动恢复成功：{failure_type.value}"
        await self.alert_manager.send_system_alert("RECOVERY_SUCCESS", message)
    
    async def _notify_recovery_failure(self, failure_type: FailureType):
        message = f"🔴 自动恢复失败：{failure_type.value}，需要人工介入"
        await self.alert_manager.send_critical_alert(message)
    
    def get_recovery_history(self, failure_type: FailureType = None, 
                            limit: int = 20) -> List[Dict[str, Any]]:
        if failure_type:
            return list(self._recovery_history[failure_type.value])[-limit:]
        
        all_history = []
        for key, records in self._recovery_history.items():
            all_history.extend([{"failure_type": key, **r} for r in records])
        
        return sorted(all_history, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_recovery_stats(self) -> Dict[str, Any]:
        stats = {}
        
        for failure_type, records in self._recovery_history.items():
            total = len(records)
            successful = sum(1 for r in records if r["success"])
            avg_duration = sum(r["duration_ms"] for r in records) / total if total > 0 else 0
            
            stats[failure_type] = {
                "total_attempts": total,
                "successful_attempts": successful,
                "success_rate": round(successful / total * 100, 2) if total > 0 else 0,
                "avg_duration_ms": round(avg_duration, 2)
            }
        
        return stats
    
    def get_state(self) -> Dict[str, Any]:
        return {
            "state": self._state.value,
            "current_failure": self._current_failure.value if self._current_failure else None,
            "last_recovery_time": self._last_recovery_time.isoformat() if self._last_recovery_time else None,
            "policies": {k.value: {"max_attempts": v.max_attempts} for k, v in self._policies.items()}
        }

    # ===================== 强化：仓位一致性自愈 =====================
    
    async def check_and_fix_position_consistency(self) -> Dict[str, Any]:
        """检查并修复仓位一致性：对比OKX实际持仓与本地记录"""
        result = {
            "checked": False,
            "mismatches": 0,
            "fixed": 0,
            "details": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        okx_client = self._dependencies.get("okx_client")
        strategy_coordinator = self._dependencies.get("strategy_coordinator")
        
        if not okx_client or not strategy_coordinator:
            result["error"] = "Missing dependencies"
            return result
        
        try:
            okx_positions = await asyncio.to_thread(okx_client.get_positions)
            okx_pos_map = {}
            
            for pos_data in okx_positions or []:
                pos = okx_client._parse_position(pos_data)
                if pos and abs(pos.quantity) > 0:
                    okx_pos_map[pos.symbol] = {
                        "side": pos.side,
                        "quantity": abs(pos.quantity),
                        "avg_cost": pos.avg_cost,
                    }
            
            unified_pos = strategy_coordinator.get_unified_position_view()
            
            for symbol, okx_pos in okx_pos_map.items():
                local_info = unified_pos.get(symbol, {})
                local_qty = local_info.get("total_long", 0) if okx_pos["side"] == "long" \
                    else local_info.get("total_short", 0)
                
                diff_pct = abs(local_qty - okx_pos["quantity"]) / max(okx_pos["quantity"], 0.0001)
                
                if diff_pct > 0.05:
                    result["mismatches"] += 1
                    result["details"].append({
                        "symbol": symbol,
                        "side": okx_pos["side"],
                        "okx_qty": okx_pos["quantity"],
                        "local_qty": local_qty,
                        "diff_pct": round(diff_pct * 100, 2),
                    })
            
            result["checked"] = True
            
            if result["mismatches"] > 0:
                logger.warning(f"Position consistency check found {result['mismatches']} mismatches")
                
                await strategy_coordinator.reconcile_position_consistency()
                result["fixed"] = result["mismatches"]
            
            return result
        except Exception as e:
            logger.error(f"Position consistency check failed: {e}")
            result["error"] = str(e)
            return result

    # ===================== 强化：WebSocket连接自愈 =====================
    
    async def monitor_websocket_health(self) -> Dict[str, Any]:
        """监控WebSocket健康状态，自动重连"""
        result = {
            "ws_instances": 0,
            "healthy": 0,
            "reconnected": 0,
            "failed": 0,
            "timestamp": datetime.now().isoformat(),
        }
        
        websocket_managers = self._dependencies.get("websocket_managers", [])
        if not websocket_managers:
            result["error"] = "No websocket managers registered"
            return result
        
        for ws_mgr in websocket_managers:
            result["ws_instances"] += 1
            
            try:
                is_connected = getattr(ws_mgr, 'is_connected', None)
                if callable(is_connected):
                    connected = is_connected()
                else:
                    connected = getattr(ws_mgr, '_connected', False)
                
                if connected:
                    result["healthy"] += 1
                else:
                    reconnect_fn = getattr(ws_mgr, 'reconnect', None)
                    if reconnect_fn and callable(reconnect_fn):
                        try:
                            if asyncio.iscoroutinefunction(reconnect_fn):
                                await reconnect_fn()
                            else:
                                reconnect_fn()
                            result["reconnected"] += 1
                            logger.info(f"Auto-reconnected websocket")
                        except Exception as e:
                            result["failed"] += 1
                            logger.error(f"WebSocket reconnect failed: {e}")
                    else:
                        result["failed"] += 1
            except Exception as e:
                result["failed"] += 1
                logger.debug(f"WS health check error: {e}")
        
        return result

    # ===================== 强化：周期性健康巡检 =====================
    
    async def periodic_health_check(self) -> Dict[str, Any]:
        """定期健康检查：综合检查所有系统组件健康状态"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": 1.0,
            "checks": {},
        }
        
        position_check = await self.check_and_fix_position_consistency()
        results["checks"]["position_consistency"] = {
            "mismatches": position_check.get("mismatches", 0),
            "fixed": position_check.get("fixed", 0),
        }
        if position_check.get("mismatches", 0) > 0:
            results["overall_health"] -= 0.2
        
        ws_check = await self.monitor_websocket_health()
        results["checks"]["websocket_health"] = ws_check
        if ws_check.get("failed", 0) > 0:
            results["overall_health"] -= 0.2
        
        strategy_coordinator = self._dependencies.get("strategy_coordinator")
        if strategy_coordinator:
            try:
                if hasattr(strategy_coordinator, 'diagnose_all'):
                    diag = await strategy_coordinator.diagnose_all()
                    results["checks"]["strategies"] = {
                        "overall_health": diag.get("overall_health", 0),
                        "strategy_count": diag.get("strategy_count", 0),
                    }
                    results["overall_health"] = min(results["overall_health"], diag.get("overall_health", 0))
            except Exception as e:
                results["checks"]["strategies"] = {"error": str(e)}
                results["overall_health"] -= 0.1
        
        results["overall_health"] = max(0.0, results["overall_health"])
        return results

    # ===================== 强化：故障记忆与学习 =====================
    
    def get_failure_patterns(self, min_occurrences: int = 2) -> Dict[str, Any]:
        """分析故障模式：识别频繁发生的故障类型
        返回按频率排序的故障模式列表
        """
        patterns = {}
        
        for failure_type, records in self._recovery_history.items():
            if len(records) >= min_occurrences:
                recent_records = records[-20:]
                
                success_rate = sum(1 for r in recent_records if r["success"]) / len(recent_records)
                avg_duration = sum(r["duration_ms"] for r in recent_records) / len(recent_records)
                
                action_stats = {}
                for r in recent_records:
                    action = r["action"]
                    if action not in action_stats:
                        action_stats[action] = {"count": 0, "success": 0}
                    action_stats[action]["count"] += 1
                    if r["success"]:
                        action_stats[action]["success"] += 1
                
                patterns[failure_type] = {
                    "total_occurrences": len(records),
                    "recent_occurrences": len(recent_records),
                    "success_rate": round(success_rate * 100, 2),
                    "avg_duration_ms": round(avg_duration, 2),
                    "best_action": max(
                        action_stats.items(),
                        key=lambda x: x[1]["success"] / max(x[1]["count"], 1)
                    )[0] if action_stats else None,
                    "action_stats": action_stats,
                }
        
        return dict(sorted(
            patterns.items(),
            key=lambda x: x[1]["recent_occurrences"],
            reverse=True
        ))

    def get_recovery_recommendations(self) -> List[str]:
        """基于故障历史生成恢复优化建议"""
        recommendations = []
        patterns = self.get_failure_patterns(min_occurrences=1)
        
        for failure_type, info in patterns.items():
            if info["success_rate"] < 50:
                recommendations.append(
                    f"{failure_type}恢复成功率仅{info['success_rate']}%，建议检查网络配置或增加重试次数"
                )
            
            if info["avg_duration_ms"] > 10000:
                recommendations.append(
                    f"{failure_type}平均恢复时间{info['avg_duration_ms']/1000:.1f}s过长，考虑使用更快的恢复策略"
                )
            
            if info["recent_occurrences"] > 5:
                recommendations.append(
                    f"{failure_type}近期发生{info['recent_occurrences']}次，建议排查根本原因"
                )
        
        if not recommendations:
            recommendations.append("系统恢复状态良好，暂无优化建议")
        
        return recommendations

    # ===================== 强化：多级熔断自愈 =====================
    
    async def apply_circuit_breaker_healing(self, component: str) -> Dict[str, Any]:
        """针对熔断组件的自愈处理：分级降级恢复"""
        result = {
            "component": component,
            "level": 0,
            "actions": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        circuit_breaker = self._dependencies.get("circuit_breaker")
        
        if circuit_breaker and hasattr(circuit_breaker, 'get_state'):
            try:
                cb_state = circuit_breaker.get_state()
                result["circuit_state"] = cb_state
                
                if cb_state.get("state") == "open":
                    result["level"] = 3
                    result["actions"].append("circuit_open_escalation")
                    
                    await self.detect_and_recover(FailureType.API, {"component": component})
                elif cb_state.get("state") == "half_open":
                    result["level"] = 2
                    result["actions"].append("circuit_half_open_monitoring")
                else:
                    result["level"] = 1
                    result["actions"].append("circuit_closed_normal")
            except Exception as e:
                result["error"] = str(e)
        
        return result
