"""
决策协调器（Decision Coordinator）— 完善强化版

核心增强：
  1. 决策依赖图 — 拓扑排序保证执行顺序（如先平仓后开仓）
  2. 决策批处理 — 原子化批量提交，全部成功或全部回滚
  3. 决策超时/过期 — 自动拒绝过期决策，防止过期信号执行
  4. 决策重试机制 — 指数退避重试，避免瞬时故障丢弃信号
  5. 优先级自动升级 — 长时挂起决策自动提升优先级
  6. 决策影响分析 — 执行前模拟PnL、风险、资金占用
  7. 速率限制 — 每策略/每币种独立限速
  8. 决策分组 — 按相关性/币种/策略分组协调
  9. 生命周期回调 — 预检查/后处理钩子
  10. 增强冲突检测 — 时间碰撞、策略不兼容、反方向对冲
"""
import asyncio
import copy
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════════════

class DecisionType(Enum):
    SIGNAL = "signal"
    ORDER = "order"
    RISK_ACTION = "risk_action"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    STRATEGY_CONTROL = "strategy_control"
    BATCH = "batch"
    SEQUENCE = "sequence"


class DecisionPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3


class ConflictType(Enum):
    OPPOSITE_DIRECTION = "opposite_direction"
    OVER_EXPOSURE = "over_exposure"
    TIME_COLLISION = "time_collision"
    INSUFFICIENT_CAPITAL = "insufficient_capital"
    STRATEGY_INCOMPATIBILITY = "strategy_incompatibility"
    HEDGE_OPPOSITE = "hedge_opposite"           # 对冲方向冲突
    SYMBOL_CORRELATION = "symbol_correlation"    # 高相关币种同向重仓


class DecisionStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRYING = "retrying"


class BatchStatus(Enum):
    COLLECTING = "collecting"
    VALIDATING = "validating"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"
    PARTIAL = "partial"


class DependencyType(Enum):
    MUST_BEFORE = "must_before"       # A must execute before B
    MUST_AFTER = "must_after"         # A must execute after B
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # A and B cannot coexist


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class DecisionDependency:
    """决策依赖关系"""
    source_id: str
    target_id: str
    dep_type: DependencyType
    reason: str = ""


@dataclass
class ImpactAnalysis:
    """决策影响分析"""
    decision_id: str = ""
    # PnL 影响
    estimated_pnl: float = 0.0
    estimated_fee: float = 0.0
    estimated_slippage: float = 0.0
    net_impact: float = 0.0
    # 风险影响
    delta_exposure_pct: float = 0.0
    post_execution_exposure: float = 0.0
    risk_budget_consumed: float = 0.0
    # 资金影响
    capital_locked: float = 0.0
    available_after: float = 0.0
    # 组合影响
    portfolio_delta_var: float = 0.0
    correlation_impact: float = 0.0
    # 判断
    is_safe: bool = True
    warnings: List[str] = field(default_factory=list)


@dataclass
class RateLimitState:
    """速率限制状态"""
    max_per_second: float = 30.0
    max_per_minute: float = 180.0
    timestamps: deque = field(default_factory=deque)
    blocked_count: int = 0


@dataclass
class DecisionLifecycleEntry:
    """决策生命周期记录"""
    decision_id: str = ""
    timestamp: str = ""
    from_status: str = ""
    to_status: str = ""
    reason: str = ""
    latency_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════
# Decision 类（增强）
# ═══════════════════════════════════════════════════════════════

class Decision:
    """决策实体"""
    __slots__ = (
        "decision_id", "decision_type", "source", "data",
        "priority", "original_priority", "confidence",
        "status", "created_at", "processed_at", "executed_at",
        "expires_at", "rejection_reason", "conflicts",
        "dependencies", "retry_count", "max_retries",
        "lifecycle", "impact", "batch_id", "group_key",
    )

    def __init__(
        self,
        decision_id: str,
        decision_type: DecisionType,
        source: str,
        data: Dict[str, Any],
        priority: DecisionPriority = DecisionPriority.NORMAL,
        confidence: float = 0.5,
        ttl_seconds: float = 30.0,
        max_retries: int = 3,
        dependencies: List[DecisionDependency] = None,
    ):
        self.decision_id = decision_id
        self.decision_type = decision_type
        self.source = source
        self.data = data
        self.priority = priority
        self.original_priority = priority
        self.confidence = confidence
        self.status = DecisionStatus.PENDING
        self.created_at = datetime.now()
        self.processed_at: Optional[datetime] = None
        self.executed_at: Optional[datetime] = None
        self.expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        self.rejection_reason: Optional[str] = None
        self.conflicts: List[Dict[str, Any]] = []
        self.dependencies: List[DecisionDependency] = dependencies or []
        self.retry_count: int = 0
        self.max_retries: int = max_retries
        self.lifecycle: List[DecisionLifecycleEntry] = []
        self.impact: Optional[ImpactAnalysis] = None
        self.batch_id: Optional[str] = None
        self.group_key: Optional[str] = None

        self._record_lifecycle("", DecisionStatus.PENDING.value, "created")

    def _record_lifecycle(self, from_status: str, to_status: str, reason: str = ""):
        self.lifecycle.append(DecisionLifecycleEntry(
            decision_id=self.decision_id,
            timestamp=datetime.now().isoformat(),
            from_status=from_status,
            to_status=to_status,
            reason=reason,
        ))

    def is_expired(self) -> bool:
        return datetime.now() > self.expires_at

    def can_retry(self) -> bool:
        return self.retry_count < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "decision_type": self.decision_type.value,
            "source": self.source,
            "data": self.data,
            "priority": self.priority.value,
            "original_priority": self.original_priority.value,
            "confidence": self.confidence,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "processed_at": self.processed_at.isoformat() if self.processed_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "expires_at": self.expires_at.isoformat(),
            "rejection_reason": self.rejection_reason,
            "conflicts": self.conflicts,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "batch_id": self.batch_id,
            "group_key": self.group_key,
            "lifecycle": [
                {"from": e.from_status, "to": e.to_status, "at": e.timestamp, "reason": e.reason}
                for e in self.lifecycle
            ],
            "impact": {
                "estimated_pnl": self.impact.estimated_pnl,
                "net_impact": self.impact.net_impact,
                "is_safe": self.impact.is_safe,
            } if self.impact else None,
        }


# ═══════════════════════════════════════════════════════════════
# DecisionBatch — 原子批处理
# ═══════════════════════════════════════════════════════════════

class DecisionBatch:
    """原子决策批次 — 全部成功或全部回滚"""

    def __init__(self, batch_id: str, name: str = "", atomic: bool = True):
        self.batch_id = batch_id
        self.name = name
        self.atomic = atomic
        self.status = BatchStatus.COLLECTING
        self.decisions: List[Decision] = []
        self.created_at = datetime.now()
        self.completed_at: Optional[datetime] = None
        self.error: Optional[str] = None
        self._executed: List[str] = []  # 已执行的决策ID列表（用于回滚）

    def add(self, decision: Decision) -> None:
        decision.batch_id = self.batch_id
        self.decisions.append(decision)

    def mark_executed(self, decision_id: str) -> None:
        self._executed.append(decision_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "name": self.name,
            "atomic": self.atomic,
            "status": self.status.value,
            "decision_count": len(self.decisions),
            "decision_ids": [d.decision_id for d in self.decisions],
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "executed_count": len(self._executed),
        }


# ═══════════════════════════════════════════════════════════════
# DecisionCoordinator — 增强版
# ═══════════════════════════════════════════════════════════════

class DecisionCoordinator:
    """
    决策协调器 — 完善强化版

    新增核心能力：
      - 决策依赖拓扑排序（如先平后开）
      - 原子批处理（全部成功或全部回滚）
      - 超时自动过期
      - 指数退避重试
      - 优先级自动升级
      - 执行前影响分析
      - 速率限制（每策略/每币种）
      - 生命周期完整追踪
      - 对冲检测、相关性冲突检测
      - 决策分组协同管理
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        coord_cfg = config.get("decision_coordinator", {}) if config else {}

        # ── 决策队列与状态 ──
        self._decision_queue: List[Decision] = []
        self._active_decisions: Dict[str, Decision] = {}
        self._processed_decisions: List[Decision] = []
        self._lock = asyncio.Lock()
        self._max_processed = coord_cfg.get("max_processed", 5000)

        # ── 依赖图管理 ──
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._reverse_deps: Dict[str, Set[str]] = defaultdict(set)

        # ── 批处理管理 ──
        self._batches: Dict[str, DecisionBatch] = {}
        self._active_batches: Dict[str, DecisionBatch] = {}

        # ── 冲突解决策略 ──
        self._conflict_resolution_strategies = {
            ConflictType.OPPOSITE_DIRECTION: self._resolve_opposite_direction,
            ConflictType.OVER_EXPOSURE: self._resolve_over_exposure,
            ConflictType.TIME_COLLISION: self._resolve_time_collision,
            ConflictType.INSUFFICIENT_CAPITAL: self._resolve_insufficient_capital,
            ConflictType.STRATEGY_INCOMPATIBILITY: self._resolve_strategy_incompatibility,
            ConflictType.HEDGE_OPPOSITE: self._resolve_hedge_opposite,
            ConflictType.SYMBOL_CORRELATION: self._resolve_symbol_correlation,
        }

        # ── 优先级权重 ──
        self._priority_weights = {
            DecisionPriority.CRITICAL: 10.0,
            DecisionPriority.HIGH: 5.0,
            DecisionPriority.NORMAL: 1.0,
            DecisionPriority.LOW: 0.5,
        }

        # ── 速率限制 ──
        self._rate_limiters: Dict[str, RateLimitState] = {}  # key: "strategy:symbol" 或 "global"
        self._global_rate_limiter = RateLimitState(
            max_per_second=coord_cfg.get("rate_limit_per_second", 30),
            max_per_minute=coord_cfg.get("rate_limit_per_minute", 180),
        )

        # ── 超时配置 ──
        self._default_ttl_seconds = coord_cfg.get("default_ttl_seconds", 30)
        self._max_ttl_seconds = coord_cfg.get("max_ttl_seconds", 300)
        self._priority_escalation_seconds = coord_cfg.get("priority_escalation_seconds", 15)

        # ── 重试配置 ──
        self._max_retries = coord_cfg.get("max_retries", 3)
        self._retry_base_delay_ms = coord_cfg.get("retry_base_delay_ms", 200)
        self._retry_max_delay_ms = coord_cfg.get("retry_max_delay_ms", 10000)
        self._batch_timeout_seconds = coord_cfg.get("batch_timeout_seconds", 30)  # 批次执行超时

        # ── 生命周期回调 ──
        self._pre_submit_hooks: List[Callable] = []
        self._post_process_hooks: List[Callable] = []
        self._on_reject_hooks: List[Callable] = []

        # ── 策略不兼容矩阵 ──
        self._strategy_incompatibilities: Dict[str, Set[str]] = {}

        # ── 注入依赖 ──
        self._account_manager = None
        self._risk_gate = None
        self._portfolio_optimizer = None

        # ── 分组配置 ──
        self._decision_groups: Dict[str, List[str]] = defaultdict(list)

        logger.info(
            f"DecisionCoordinator enhanced: ttl={self._default_ttl_seconds}s, "
            f"max_retries={self._max_retries}, rate_limit={self._global_rate_limiter.max_per_second}/s"
        )

    # ═══════════════════════════════════════════════════════
    # 依赖注入
    # ═══════════════════════════════════════════════════════

    def set_account_manager(self, account_manager) -> None:
        self._account_manager = account_manager

    def set_risk_gate(self, risk_gate) -> None:
        self._risk_gate = risk_gate

    def set_portfolio_optimizer(self, optimizer) -> None:
        self._portfolio_optimizer = optimizer

    def register_strategy_incompatibility(self, strategy_a: str, strategy_b: str) -> None:
        """注册策略不兼容关系"""
        self._strategy_incompatibilities.setdefault(strategy_a, set()).add(strategy_b)
        self._strategy_incompatibilities.setdefault(strategy_b, set()).add(strategy_a)

    def register_pre_submit_hook(self, hook: Callable) -> None:
        self._pre_submit_hooks.append(hook)

    def register_post_process_hook(self, hook: Callable) -> None:
        self._post_process_hooks.append(hook)

    def register_on_reject_hook(self, hook: Callable) -> None:
        self._on_reject_hooks.append(hook)

    # ═══════════════════════════════════════════════════════
    # 核心: 决策提交
    # ═══════════════════════════════════════════════════════

    async def submit_decision(
        self,
        decision: Decision,
        dependencies: List[DecisionDependency] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        提交决策到协调器（带锁）

        步骤：
          1. 预提交钩子检查
          2. 速率限制检查
          3. 超时检查
          4. 依赖关系注册
          5. 冲突检测
          6. 冲突解决
          7. 入队
        """
        async with self._lock:
            return await self._submit_decision_nolock(decision, dependencies)

    async def _submit_decision_nolock(
        self,
        decision: Decision,
        dependencies: List[DecisionDependency] = None,
    ) -> Tuple[bool, List[Dict[str, Any]]]:
        """
        提交决策到协调器（调用方必须持有 _lock）

        submit_batch 等批量操作可在持锁状态下直接调用此方法，
        避免 asyncio.Lock 重入死锁。
        """
        # ── 0. 预提交钩子 ──
        for hook in self._pre_submit_hooks:
            try:
                if not hook(decision):
                    decision.status = DecisionStatus.REJECTED
                    decision.rejection_reason = f"Pre-submit hook rejected: {hook.__name__}"
                    self._store_decision(decision)
                    logger.info(f"Decision rejected by pre-submit hook: {decision.decision_id}")
                    return False, []
            except Exception as e:
                logger.error(f"Pre-submit hook error: {e}")

        # ── 1. 速率限制 ──
        rate_key = f"{decision.source}:{decision.data.get('symbol', 'global')}"
        if not self._check_rate_limit(rate_key):
            decision.status = DecisionStatus.REJECTED
            decision.rejection_reason = f"Rate limit exceeded: {rate_key}"
            self._store_decision(decision)
            return False, [{
                "type": "rate_limit",
                "severity": "high",
                "message": f"Rate limit exceeded for {rate_key}",
            }]

        # ── 2. 超时检查 ──
        if decision.is_expired():
            decision.status = DecisionStatus.EXPIRED
            decision.rejection_reason = "Decision expired before submission"
            self._store_decision(decision)
            logger.info(f"Decision expired: {decision.decision_id}")
            return False, [{
                "type": "expired",
                "severity": "low",
                "message": "Decision TTL expired",
            }]

        # ── 3. 注册依赖关系 ──
        if dependencies:
            decision.dependencies = dependencies
            self._register_dependencies(decision, dependencies)

        # ── 4. 影响分析 ──
        try:
            decision.impact = await self._analyze_impact(decision)
        except Exception as e:
            logger.debug(f"Impact analysis skipped: {e}")

        # ── 5. 增强冲突检测 ──
        conflicts = await self._detect_conflicts_enhanced(decision)
        decision.conflicts = conflicts

        # ── 6. 冲突解决 ──
        if conflicts:
            resolution_result = await self._resolve_conflicts(decision)
            if not resolution_result:
                decision.status = DecisionStatus.REJECTED
                decision.rejection_reason = (
                    f"Conflict resolution failed: "
                    f"{[c['type'] for c in conflicts]}"
                )
                self._store_decision(decision)
                await self._notify_reject(decision)
                logger.warning(
                    f"Decision rejected: {decision.decision_id}, conflicts={conflicts}"
                )
                return False, conflicts

        # ── 7. 入队 ──
        decision.status = DecisionStatus.PENDING
        decision._record_lifecycle("", DecisionStatus.PENDING.value, "submitted")
        self._decision_queue.append(decision)
        self._active_decisions[decision.decision_id] = decision
        self._record_rate_limit(rate_key)

        # 自动分组
        self._auto_group(decision)

        logger.info(
            f"Decision submitted: {decision.decision_id}, "
            f"type={decision.decision_type.value}, "
            f"source={decision.source}"
        )
        return True, conflicts

    # ═══════════════════════════════════════════════════════
    # 核心: 决策批处理（原子化）
    # ═══════════════════════════════════════════════════════

    async def submit_batch(
        self,
        decisions: List[Decision],
        batch_name: str = "",
        atomic: bool = True,
    ) -> Tuple[str, List[Decision]]:
        """
        原子化批量提交决策

        全部通过 → 全部执行
        任一失败 → 全部拒绝（atomic=True）

        Returns:
            (batch_id, accepted_decisions)
        """
        batch_id = f"batch_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        batch = DecisionBatch(batch_id, name=batch_name, atomic=atomic)

        async with self._lock:
            self._batches[batch_id] = batch

            # 第一遍：预检所有决策
            pre_check_results = []
            for decision in decisions:
                batch.add(decision)
                # 只做轻量预检，不做完整冲突检测
                if decision.is_expired():
                    pre_check_results.append((decision, False, "expired"))
                    continue
                rate_key = f"{decision.source}:{decision.data.get('symbol', 'global')}"
                if not self._check_rate_limit(rate_key):
                    pre_check_results.append((decision, False, "rate_limited"))
                    continue
                pre_check_results.append((decision, True, ""))

            # 如果 atomic 且有失败 → 全部拒绝
            if atomic:
                failures = [(d, r) for d, ok, r in pre_check_results if not ok]
                if failures:
                    for decision in decisions:
                        decision.status = DecisionStatus.REJECTED
                        decision.rejection_reason = (
                            f"Atomic batch failed: {failures[0][1]}"
                        )
                        self._store_decision(decision)
                    batch.status = BatchStatus.ROLLED_BACK
                    batch.error = f"Pre-check failed: {failures[0][1]}"
                    logger.warning(f"Batch {batch_id} rolled back: {batch.error}")
                    return batch_id, []

            # 第二遍：提交通过预检的决策（用 _submit_decision_nolock 避免锁重入死锁）
            accepted = []
            for decision, ok, reason in pre_check_results:
                if ok:
                    success, conflicts = await self._submit_decision_nolock(decision)
                    if success:
                        accepted.append(decision)
                    else:
                        if atomic:
                            # 回滚已提交的（cancel_decision 内部也会获取锁，需要单独处理）
                            for d in accepted:
                                self._cancel_decision_nolock(d.decision_id)
                            batch.status = BatchStatus.ROLLED_BACK
                            batch.error = f"Decision {decision.decision_id} failed"
                            logger.warning(f"Batch {batch_id} rolled back")
                            return batch_id, []
                else:
                    decision.status = DecisionStatus.REJECTED
                    decision.rejection_reason = reason
                    self._store_decision(decision)

            batch.status = BatchStatus.COLLECTING if accepted else BatchStatus.ROLLED_BACK
            self._active_batches[batch_id] = batch
            logger.info(
                f"Batch {batch_id} submitted: {len(accepted)}/{len(decisions)} accepted"
            )
            return batch_id, accepted

    # ═══════════════════════════════════════════════════════
    # 核心: 决策处理（含依赖排序）
    # ═══════════════════════════════════════════════════════

    async def process_decisions(self) -> List[Decision]:
        """
        处理待处理决策队列

        增强点：
          - 过期决策自动拒绝
          - 优先级自动升级（长时挂起）
          - 依赖拓扑排序执行
          - 后处理钩子
        """
        async with self._lock:
            now = datetime.now()
            processed = []

            # ── 0. 清理过期决策 ──
            expired = [d for d in self._decision_queue if d.is_expired()]
            for d in expired:
                d.status = DecisionStatus.EXPIRED
                d.rejection_reason = "Expired in queue"
                d._record_lifecycle(
                    DecisionStatus.PENDING.value, DecisionStatus.EXPIRED.value, "ttl_expired"
                )
                self._decision_queue.remove(d)
                self._active_decisions.pop(d.decision_id, None)
                self._store_decision(d)
                processed.append(d)
            if expired:
                logger.info(f"Expired {len(expired)} stale decisions")

            # ── 1. 优先级自动升级 ──
            for d in self._decision_queue:
                age_seconds = (now - d.created_at).total_seconds()
                if age_seconds > self._priority_escalation_seconds:
                    if d.priority == DecisionPriority.LOW:
                        d.priority = DecisionPriority.NORMAL
                        d._record_lifecycle("", d.priority.value, "auto_escalated")
                    elif d.priority == DecisionPriority.NORMAL:
                        d.priority = DecisionPriority.HIGH
                        d._record_lifecycle("", d.priority.value, "auto_escalated")

            # ── 2. 依赖拓扑排序 ──
            sorted_decisions = self._topological_sort(
                [d for d in self._decision_queue if not d.is_expired()]
            )

            # ── 3. 拓扑排序保证依赖顺序（不做二次排序破坏依赖关系）
            # 注：优先级已内置在拓扑BFS队列中（同层级高优先级优先出队）

            # ── 4. 逐个处理 ──
            for decision in sorted_decisions:
                if decision not in self._decision_queue:
                    continue

                self._decision_queue.remove(decision)
                decision.status = DecisionStatus.PROCESSING
                decision.processed_at = now
                decision._record_lifecycle(
                    DecisionStatus.PENDING.value, DecisionStatus.PROCESSING.value, "processing"
                )

                # 增强冲突检测
                conflicts = await self._detect_conflicts_enhanced(decision)

                if conflicts:
                    decision.status = DecisionStatus.REJECTED
                    decision.rejection_reason = (
                        f"Conflicts during processing: "
                        f"{[c['type'] for c in conflicts]}"
                    )
                    decision._record_lifecycle(
                        DecisionStatus.PROCESSING.value,
                        DecisionStatus.REJECTED.value,
                        decision.rejection_reason,
                    )
                    await self._notify_reject(decision)
                    logger.warning(f"Decision rejected in processing: {decision.decision_id}")
                else:
                    decision.status = DecisionStatus.APPROVED
                    decision._record_lifecycle(
                        DecisionStatus.PROCESSING.value,
                        DecisionStatus.APPROVED.value,
                        "approved",
                    )
                    logger.info(f"Decision approved: {decision.decision_id}")

                self._store_decision(decision)
                self._active_decisions.pop(decision.decision_id, None)
                processed.append(decision)

            # ── 5. 后处理钩子 ──
            for hook in self._post_process_hooks:
                for decision in processed:
                    try:
                        hook(decision)
                    except Exception as e:
                        logger.error(f"Post-process hook error: {e}")

            return processed

    # ═══════════════════════════════════════════════════════
    # 决策重试
    # ═══════════════════════════════════════════════════════

    async def retry_decision(self, decision_id: str) -> Tuple[bool, str]:
        """重试失败的决策（指数退避，含竞态保护）"""
        async with self._lock:
            # 查找决策
            decision = None
            for d in self._processed_decisions:
                if d.decision_id == decision_id:
                    decision = d
                    break
            if not decision:
                return False, "Decision not found"

            if not decision.can_retry():
                return False, f"Max retries ({decision.max_retries}) exhausted"

            # 退避延迟
            delay_ms = min(
                self._retry_base_delay_ms * (2 ** decision.retry_count),
                self._retry_max_delay_ms,
            )
            decision.retry_count += 1
            decision.status = DecisionStatus.RETRYING
            decision._record_lifecycle(
                "failed", DecisionStatus.RETRYING.value,
                f"retry #{decision.retry_count}, delay={delay_ms}ms",
            )

            logger.info(
                f"Retrying decision {decision_id} "
                f"(attempt {decision.retry_count}/{decision.max_retries}, "
                f"delay={delay_ms}ms)"
            )

            # 记录重试令牌，供睡眠后验证决策未被其他操作修改
            retry_token = decision.retry_count

        await asyncio.sleep(delay_ms / 1000)

        async with self._lock:
            # P0: 重入检查——决策可能在睡眠期间被处理/取消/完成
            current = None
            for d in self._processed_decisions:
                if d.decision_id == decision_id:
                    current = d
                    break
            
            # 如果已不在processed列表中，可能被删除；从active中查找
            if not current:
                current = self._active_decisions.get(decision_id)
            
            if not current:
                return False, "Decision no longer exists after sleep"
            
            # 状态已变更（被其他操作修改）→ 放弃重试
            if current.status not in (DecisionStatus.RETRYING, DecisionStatus.FAILED):
                return False, f"Decision state changed to {current.status.value} during sleep"
            
            # 重试计数不匹配（被并发重试）→ 放弃
            if current.retry_count != retry_token:
                return False, "Concurrent retry detected"
            
            # 验证决策未过期
            if current.is_expired():
                current.status = DecisionStatus.EXPIRED
                current.rejection_reason = "Expired during retry"
                return False, "Decision expired during retry"
            
            # 重新提交到队列
            current.status = DecisionStatus.PENDING
            current.expires_at = datetime.now() + timedelta(seconds=self._default_ttl_seconds)
            self._decision_queue.append(current)
            self._active_decisions[current.decision_id] = current
            return True, f"Retrying (attempt {current.retry_count})"

    # ═══════════════════════════════════════════════════════
    # 影响分析
    # ═══════════════════════════════════════════════════════

    async def _analyze_impact(self, decision: Decision) -> ImpactAnalysis:
        """执行前影响分析"""
        ia = ImpactAnalysis(decision_id=decision.decision_id)
        data = decision.data

        # ── 基本信息 ──
        price = float(data.get("price") or data.get("entry_price") or 0)
        quantity = float(data.get("quantity") or 0)
        leverage = float(data.get("leverage") or 5.0)
        notional = price * quantity
        direction = data.get("direction", "")

        if notional <= 0:
            return ia

        # ── 手续费 ──
        taker_fee_rate = float(self.config.get("trading", {}).get("taker_fee", 0.0005))
        ia.estimated_fee = notional * taker_fee_rate * 2  # 开平仓

        # ── 滑点 ──
        base_slippage = float(self.config.get("trading", {}).get("base_slippage", 0.0002))
        ia.estimated_slippage = notional * base_slippage

        # ── 净影响 ──
        ia.net_impact = -(ia.estimated_fee + ia.estimated_slippage)

        # ── 暴露影响 ──
        margin = notional / leverage if leverage > 0 else notional
        available = self._get_available_capital()
        if available > 0:
            ia.delta_exposure_pct = margin / available
            ia.post_execution_exposure = ia.delta_exposure_pct
            ia.available_after = max(0, available - margin)
        ia.capital_locked = margin

        # ── 风险预算消耗 ──
        if self._risk_gate:
            try:
                ia.risk_budget_consumed = notional * 0.02 / max(available, 1)
            except Exception:
                pass

        # ── 安全检查 ──
        if notional > available * 0.5:
            ia.warnings.append("Notional > 50% of available capital")
        if notional > available:
            ia.is_safe = False
            ia.warnings.append("Notional exceeds available capital")
        if ia.delta_exposure_pct > 0.3:
            ia.warnings.append(f"Single decision exposure {ia.delta_exposure_pct:.1%} > 30%")

        return ia

    # ═══════════════════════════════════════════════════════
    # 增强冲突检测
    # ═══════════════════════════════════════════════════════

    async def _detect_conflicts_enhanced(self, decision: Decision) -> List[Dict[str, Any]]:
        """增强冲突检测（继承原有逻辑 + 新增检测维度）"""
        conflicts = await self._detect_conflicts(decision)

        symbol = decision.data.get("symbol", "")
        direction = decision.data.get("direction", "").lower()
        source = decision.source

        # ── 新增1: 时间碰撞检测 ──
        for existing in self._active_decisions.values():
            if existing.decision_id == decision.decision_id:
                continue
            time_diff = abs((decision.created_at - existing.created_at).total_seconds())
            if time_diff < 0.2:  # 200ms内
                conflicts.append({
                    "type": ConflictType.TIME_COLLISION.value,
                    "severity": "low",
                    "message": f"Decision collision with {existing.decision_id} ({time_diff*1000:.0f}ms)",
                    "colliding_decision": existing.decision_id,
                    "time_diff_ms": time_diff * 1000,
                })

        # ── 新增2: 策略不兼容 ──
        for existing in self._active_decisions.values():
            if existing.decision_id == decision.decision_id:
                continue
            existing_source = existing.source
            incompatible = self._strategy_incompatibilities.get(source, set())
            if existing_source in incompatible:
                conflicts.append({
                    "type": ConflictType.STRATEGY_INCOMPATIBILITY.value,
                    "severity": "high",
                    "message": f"Strategy {source} incompatible with {existing_source}",
                    "incompatible_strategy": existing_source,
                })

        # ── 新增3: 对冲方向冲突 ──
        for existing in self._active_decisions.values():
            if existing.decision_id == decision.decision_id:
                continue
            existing_symbol = existing.data.get("symbol", "")
            existing_dir = existing.data.get("direction", "").lower()

            # 同币种反方向 → 对冲
            if symbol == existing_symbol and direction and existing_dir:
                if direction != existing_dir:
                    conflicts.append({
                        "type": ConflictType.HEDGE_OPPOSITE.value,
                        "severity": "medium",
                        "message": f"Hedge conflict: {source} {direction} vs {existing.source} {existing_dir} on {symbol}",
                        "opposing_decision": existing.decision_id,
                    })

        # ── 新增4: 高相关币种同向重仓 ──
        correlated_pairs = {
            "BTC-USDT-SWAP": ["ETH-USDT-SWAP"],
            "ETH-USDT-SWAP": ["BTC-USDT-SWAP", "ARB-USDT-SWAP"],
            "SOL-USDT-SWAP": ["ETH-USDT-SWAP"],
        }
        related = correlated_pairs.get(symbol, [])
        for existing in self._active_decisions.values():
            existing_symbol = existing.data.get("symbol", "")
            if existing_symbol in related and direction and existing.data.get("direction"):
                if direction == existing.data.get("direction", "").lower():
                    conflicts.append({
                        "type": ConflictType.SYMBOL_CORRELATION.value,
                        "severity": "medium",
                        "message": f"Correlated symbols same direction: {symbol} + {existing_symbol}",
                        "related_symbol": existing_symbol,
                    })

        return conflicts

    async def _detect_conflicts(self, decision: Decision) -> List[Dict[str, Any]]:
        """原有冲突检测逻辑"""
        conflicts = []

        if decision.decision_type in (DecisionType.SIGNAL, DecisionType.ORDER):
            symbol = decision.data.get("symbol", "")
            direction = decision.data.get("direction", "").lower()

            for existing in self._active_decisions.values():
                if existing.decision_id == decision.decision_id:
                    continue
                if existing.decision_type not in (DecisionType.SIGNAL, DecisionType.ORDER):
                    continue

                existing_symbol = existing.data.get("symbol", "")
                existing_direction = existing.data.get("direction", "").lower()

                # 反向冲突
                if symbol == existing_symbol and direction and existing_direction:
                    if direction != existing_direction:
                        conflicts.append({
                            "type": ConflictType.OPPOSITE_DIRECTION.value,
                            "severity": "high",
                            "message": (
                                f"{decision.source} wants {direction} {symbol} "
                                f"but {existing.source} has {existing_direction}"
                            ),
                            "opposing_decision": existing.decision_id,
                        })

                # 暴露溢出
                if symbol == existing_symbol and direction == existing_direction:
                    d_exp = float(decision.data.get("exposure", 0) or 0)
                    e_exp = float(existing.data.get("exposure", 0) or 0)
                    exposure = d_exp + e_exp
                    if exposure > 1.0:
                        conflicts.append({
                            "type": ConflictType.OVER_EXPOSURE.value,
                            "severity": "medium",
                            "message": f"Combined exposure {exposure:.0%} > 100% on {symbol}",
                            "current_exposure": exposure,
                        })

            # 资金不足
            capital_needed = float(decision.data.get("capital_required", 0) or 0)
            available_capital = self._get_available_capital()
            if capital_needed > available_capital:
                conflicts.append({
                    "type": ConflictType.INSUFFICIENT_CAPITAL.value,
                    "severity": "high",
                    "message": (
                        f"Insufficient capital: need {capital_needed:.2f}, "
                        f"available {available_capital:.2f}"
                    ),
                    "needed": capital_needed,
                    "available": available_capital,
                })

        return conflicts

    # ═══════════════════════════════════════════════════════
    # 冲突解决策略
    # ═══════════════════════════════════════════════════════

    async def _resolve_conflicts(self, decision: Decision) -> bool:
        for conflict in decision.conflicts:
            conflict_type = ConflictType(conflict["type"])
            resolver = self._conflict_resolution_strategies.get(conflict_type)
            if resolver and await resolver(decision, conflict):
                continue
            return False
        return True

    async def _resolve_opposite_direction(self, decision: Decision, conflict: Dict[str, Any]) -> bool:
        opposing_id = conflict.get("opposing_decision")
        if opposing_id not in self._active_decisions:
            return True
        opposing = self._active_decisions[opposing_id]

        # 优先级高的获胜
        if decision.priority.value < opposing.priority.value:
            # 先关闭相反方向再执行
            opposing.status = DecisionStatus.REJECTED
            opposing.rejection_reason = (
                f"Replaced by higher priority decision {decision.decision_id}"
            )
            opposing._record_lifecycle(
                opposing.status.value, DecisionStatus.REJECTED.value,
                opposing.rejection_reason,
            )
            self._active_decisions.pop(opposing_id, None)
            return True
        elif decision.priority.value > opposing.priority.value:
            return False
        else:
            # 同优先级 → 置信度高的获胜
            if decision.confidence >= opposing.confidence:
                opposing.status = DecisionStatus.REJECTED
                opposing.rejection_reason = (
                    f"Replaced by higher confidence decision {decision.decision_id}"
                )
                self._active_decisions.pop(opposing_id, None)
                return True
            return False

    async def _resolve_over_exposure(self, decision: Decision, conflict: Dict[str, Any]) -> bool:
        current_exposure = float(conflict.get("current_exposure", 0))
        if current_exposure <= 0:
            return True
        scaling_factor = 1.0 / current_exposure
        decision.data["exposure"] = float(decision.data.get("exposure", 0)) * scaling_factor
        decision.data["quantity"] = float(decision.data.get("quantity", 0)) * scaling_factor
        logger.info(f"Scaled decision {decision.decision_id} by {scaling_factor:.3f} (over exposure)")
        return True

    async def _resolve_time_collision(self, decision: Decision, conflict: Dict[str, Any]) -> bool:
        # 时间碰撞 → 延迟处理（降优先级）
        logger.debug(f"Time collision for {decision.decision_id}, deferring")
        return True

    async def _resolve_insufficient_capital(self, decision: Decision, conflict: Dict[str, Any]) -> bool:
        needed = float(conflict.get("needed", 0))
        available = float(conflict.get("available", 0))
        if available <= 0:
            return False
        scaling_factor = available / needed
        decision.data["capital_required"] = available
        decision.data["quantity"] = float(decision.data.get("quantity", 0)) * scaling_factor
        decision.data["exposure"] = float(decision.data.get("exposure", 0)) * scaling_factor
        logger.info(
            f"Scaled decision {decision.decision_id} by {scaling_factor:.3f} (insufficient capital)"
        )
        return True

    async def _resolve_strategy_incompatibility(self, decision: Decision, conflict: Dict[str, Any]) -> bool:
        return False

    async def _resolve_hedge_opposite(self, decision: Decision, conflict: Dict[str, Any]) -> bool:
        # 对冲方向 → 高风险操作，降低置信度要求
        logger.info(f"Hedge opposite detected for {decision.decision_id}, allowing with reduced size")
        # 降低50%仓位
        decision.data["quantity"] = float(decision.data.get("quantity", 0)) * 0.5
        decision.data["exposure"] = float(decision.data.get("exposure", 0)) * 0.5
        return True

    async def _resolve_symbol_correlation(self, decision: Decision, conflict: Dict[str, Any]) -> bool:
        # 高相关币种同向 → 合并仓位
        logger.info(f"Correlated symbols for {decision.decision_id}, reducing exposure")
        decision.data["quantity"] = float(decision.data.get("quantity", 0)) * 0.6
        decision.data["exposure"] = float(decision.data.get("exposure", 0)) * 0.6
        return True

    # ═══════════════════════════════════════════════════════
    # 依赖管理
    # ═══════════════════════════════════════════════════════

    def _register_dependencies(
        self, decision: Decision, dependencies: List[DecisionDependency]
    ) -> None:
        for dep in dependencies:
            self._dependency_graph[dep.source_id].add(dep.target_id)
            self._reverse_deps[dep.target_id].add(dep.source_id)

    def _detect_deadlock(
        self, decision_ids: Set[str], id_to_decision: Dict[str, Decision]
    ) -> Optional[List[str]]:
        """DFS检测依赖图中的环（死锁检测）

        返回检测到的第一个环的节点ID列表，无环返回None。
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {did: WHITE for did in decision_ids}
        parent: Dict[str, Optional[str]] = {did: None for did in decision_ids}

        def dfs(node: str) -> Optional[List[str]]:
            color[node] = GRAY
            # 遍历出边：node依赖的（必须在其之前执行的）
            for target in self._dependency_graph.get(node, set()):
                if target not in decision_ids:
                    continue
                if color[target] == GRAY:
                    # 找到环：回溯构建环路径
                    cycle = [target, node]
                    curr = node
                    while parent.get(curr) and curr != target:
                        curr = parent[curr]
                        if curr is None:
                            break
                        cycle.append(curr)
                    return cycle
                elif color[target] == WHITE:
                    parent[target] = node
                    result = dfs(target)
                    if result:
                        return result
            color[node] = BLACK
            return None

        for did in decision_ids:
            if color[did] == WHITE:
                result = dfs(did)
                if result:
                    return result
        return None

    def _topological_sort(self, decisions: List[Decision]) -> List[Decision]:
        """拓扑排序 — 保证依赖顺序执行，同层级内按优先级排序"""
        if not decisions:
            return []

        decision_ids = {d.decision_id for d in decisions}
        id_to_decision = {d.decision_id: d for d in decisions}

        # ── 死锁检测 ──
        cycle = self._detect_deadlock(decision_ids, id_to_decision)
        if cycle:
            cycle_names = [id_to_decision.get(did, did) for did in cycle]
            cycle_str = " → ".join(str(c) for c in cycle_names)
            logger.error(f"Deadlock detected in dependency graph: {cycle_str}")
            # 打破死锁：移除最后一条边的依赖（简单策略）
            last_a, last_b = cycle[-2], cycle[-1] if len(cycle) >= 2 else (None, None)
            if last_a and last_b:
                if last_b in self._reverse_deps and last_a in self._reverse_deps[last_b]:
                    self._reverse_deps[last_b].discard(last_a)
                    logger.warning(f"Broke deadlock: removed dep {last_a} → {last_b}")
        in_degree: Dict[str, int] = {d.decision_id: 0 for d in decisions}

        # 计算入度：仅计算源和目标都在当前决策集内的依赖
        for did in decision_ids:
            if did in self._reverse_deps:
                for src in self._reverse_deps[did]:
                    if src in decision_ids:
                        in_degree[did] = in_degree.get(did, 0) + 1

        # 优先级感知BFS：入度为0的节点按优先级排序入队
        zero_degree = sorted(
            [did for did in decision_ids if in_degree.get(did, 0) == 0],
            key=lambda did: (id_to_decision[did].priority.value, -id_to_decision[did].confidence)
        )
        
        sorted_ids = []
        while zero_degree:
            node = zero_degree.pop(0)
            sorted_ids.append(node)
            for target in self._dependency_graph.get(node, set()):
                if target in decision_ids:
                    in_degree[target] -= 1
                    if in_degree[target] == 0:
                        # 插入时保持优先级顺序
                        target_priority = id_to_decision[target].priority.value
                        insert_pos = 0
                        for i, zd in enumerate(zero_degree):
                            if id_to_decision[zd].priority.value > target_priority:
                                insert_pos = i + 1
                            else:
                                break
                        zero_degree.insert(insert_pos, target)

        # 按拓扑序排列，未排序的追加在后面
        result = [id_to_decision[did] for did in sorted_ids if did in id_to_decision]
        covered = set(sorted_ids)
        result.extend([d for d in decisions if d.decision_id not in covered])

        return result

    # ═══════════════════════════════════════════════════════
    # 速率限制
    # ═══════════════════════════════════════════════════════

    def _check_rate_limit(self, key: str) -> bool:
        """检查速率限制"""
        limiter = self._rate_limiters.get(key, self._global_rate_limiter)
        now = time.time()

        # 清理过期时间戳
        while limiter.timestamps and now - limiter.timestamps[0] > 60:
            limiter.timestamps.popleft()

        # 1秒窗口
        recent_1s = sum(1 for t in limiter.timestamps if now - t <= 1.0)
        if recent_1s >= limiter.max_per_second:
            limiter.blocked_count += 1
            return False

        # 1分钟窗口
        recent_1m = sum(1 for t in limiter.timestamps if now - t <= 60.0)
        if recent_1m >= limiter.max_per_minute:
            limiter.blocked_count += 1
            return False

        return True

    def _record_rate_limit(self, key: str) -> None:
        """记录速率限制时间戳"""
        limiter = self._rate_limiters.get(key)
        if limiter is None:
            limiter = RateLimitState()
            self._rate_limiters[key] = limiter
        limiter.timestamps.append(time.time())

    def set_rate_limit(
        self, key: str, max_per_second: float, max_per_minute: float
    ) -> None:
        """设置速率限制"""
        limiter = self._rate_limiters.get(key)
        if limiter is None:
            limiter = RateLimitState()
            self._rate_limiters[key] = limiter
        limiter.max_per_second = max_per_second
        limiter.max_per_minute = max_per_minute

    # ═══════════════════════════════════════════════════════
    # 决策分组
    # ═══════════════════════════════════════════════════════

    def _auto_group(self, decision: Decision) -> None:
        """自动按币种分组"""
        symbol = decision.data.get("symbol", "unknown")
        self._decision_groups[symbol].append(decision.decision_id)
        decision.group_key = symbol

    def get_group_decisions(self, group_key: str) -> List[Decision]:
        """获取分组内决策"""
        ids = self._decision_groups.get(group_key, [])
        result = []
        for did in ids:
            if did in self._active_decisions:
                result.append(self._active_decisions[did])
            else:
                for d in self._processed_decisions:
                    if d.decision_id == did:
                        result.append(d)
                        break
        return result

    # ═══════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════

    async def get_pending_decisions(self) -> List[Decision]:
        async with self._lock:
            return [d for d in self._decision_queue if d.status == DecisionStatus.PENDING]

    async def get_decision_status(self, decision_id: str) -> Optional[DecisionStatus]:
        async with self._lock:
            decision = self._active_decisions.get(decision_id)
            if decision:
                return decision.status
            for d in self._processed_decisions:
                if d.decision_id == decision_id:
                    return d.status
            return None

    async def get_decision_lifecycle(self, decision_id: str) -> List[Dict[str, Any]]:
        """获取决策完整生命周期"""
        async with self._lock:
            decision = self._active_decisions.get(decision_id)
            if not decision:
                for d in self._processed_decisions:
                    if d.decision_id == decision_id:
                        decision = d
                        break
            if not decision:
                return []
            return [
                {"from": e.from_status, "to": e.to_status, "at": e.timestamp, "reason": e.reason}
                for e in decision.lifecycle
            ]

    async def cancel_decision(self, decision_id: str) -> bool:
        async with self._lock:
            return self._cancel_decision_nolock(decision_id)

    def _cancel_decision_nolock(self, decision_id: str) -> bool:
        """取消决策（调用方必须持有 _lock）"""
        if decision_id in self._active_decisions:
            d = self._active_decisions[decision_id]
            d.status = DecisionStatus.REJECTED
            d.rejection_reason = "Cancelled by user"
            d._record_lifecycle(d.status.value, DecisionStatus.REJECTED.value, "cancelled")
            self._decision_queue = [
                x for x in self._decision_queue if x.decision_id != decision_id
            ]
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取协调器统计"""
        pending = len([d for d in self._decision_queue if d.status == DecisionStatus.PENDING])
        approved = len([d for d in self._processed_decisions if d.status == DecisionStatus.APPROVED])
        rejected = len([d for d in self._processed_decisions if d.status == DecisionStatus.REJECTED])
        executed = len([d for d in self._processed_decisions if d.status == DecisionStatus.EXECUTED])
        expired = len([d for d in self._processed_decisions if d.status == DecisionStatus.EXPIRED])
        retried = len([d for d in self._processed_decisions if d.retry_count > 0])
        total = len(self._processed_decisions)

        # 冲突分类统计
        conflict_types = defaultdict(int)
        for d in self._processed_decisions:
            for c in d.conflicts:
                conflict_types[c.get("type", "unknown")] += 1

        # 速率限制统计
        rate_limit_stats = {
            key: {"blocked": rl.blocked_count, "queue_size": len(rl.timestamps)}
            for key, rl in self._rate_limiters.items()
        }

        # 按来源分组统计
        source_stats = defaultdict(lambda: {"submitted": 0, "approved": 0, "rejected": 0})
        for d in self._processed_decisions:
            source_stats[d.source]["submitted"] += 1
            if d.status == DecisionStatus.APPROVED:
                source_stats[d.source]["approved"] += 1
            elif d.status == DecisionStatus.REJECTED:
                source_stats[d.source]["rejected"] += 1

        return {
            "queue": {
                "pending": pending,
                "active": len(self._active_decisions),
            },
            "decisions": {
                "total": total,
                "approved": approved,
                "rejected": rejected,
                "executed": executed,
                "expired": expired,
                "retried": retried,
            },
            "conflict_rate": sum(1 for d in self._processed_decisions if d.conflicts) / max(total, 1),
            "conflict_types": dict(conflict_types),
            "batches": {
                "active": len(self._active_batches),
                "total": len(self._batches),
                "timeout_seconds": self._batch_timeout_seconds,
            },
            "rate_limits": rate_limit_stats,
            "source_stats": {
                k: dict(v) for k, v in source_stats.items()
            },
            "dependency_graph_size": sum(len(v) for v in self._dependency_graph.values()),
        }

    # ═══════════════════════════════════════════════════════
    # 内部工具
    # ═══════════════════════════════════════════════════════

    def _store_decision(self, decision: Decision) -> None:
        self._processed_decisions.append(decision)
        if len(self._processed_decisions) > self._max_processed:
            cutoff = len(self._processed_decisions) - self._max_processed + 500
            self._processed_decisions = self._processed_decisions[cutoff:]

    async def _notify_reject(self, decision: Decision) -> None:
        for hook in self._on_reject_hooks:
            try:
                hook(decision)
            except Exception as e:
                logger.error(f"On-reject hook error: {e}")

    def _get_available_capital(self) -> float:
        if self._account_manager:
            try:
                account_info = self._account_manager.get_account_info()
                if account_info:
                    details = account_info.get("details", [])
                    for detail in details:
                        if detail.get("ccy") == "USDT":
                            avail = float(detail.get("availBal", 0) or 0)
                            if avail > 0:
                                return avail
                    total_eq = float(account_info.get("totalEq", 0))
                    if total_eq > 0:
                        return total_eq * 0.5
            except Exception:
                pass
        trading_cfg = self.config.get("trading", {})
        return trading_cfg.get("total_capital", 1000.0)


# ═══════════════════════════════════════════════════════════════
# 导出
# ═══════════════════════════════════════════════════════════════

__all__ = [
    "DecisionCoordinator",
    "Decision",
    "DecisionBatch",
    "DecisionDependency",
    "ImpactAnalysis",
    "DecisionType",
    "DecisionPriority",
    "ConflictType",
    "DecisionStatus",
    "BatchStatus",
    "DependencyType",
    "RateLimitState",
    "DecisionLifecycleEntry",
]
