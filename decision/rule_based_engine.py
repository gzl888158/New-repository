"""
基于规则的决策引擎（Rule-Based Decision Engine）— 完善强化版

.. deprecated::
    实验性模块，未接入生产交易链路。保留供架构演进参考。

核心增强：
  1. 规则组/命名空间 — 按类别组织规则（风险、信号、订单、风控等）
  2. 加权评分系统 — 规则不再只是二元匹配，可贡献加权分数
  3. 级联规则触发 — 规则匹配后可触发其他规则（链式推理）
  4. 激活窗口 — 时间/市场条件约束的规则激活（如仅高波动时启用）
  5. 规则性能追踪 — 命中率、误报率、PnL影响、延迟统计
  6. 冲突自动检测 — 检测互斥/矛盾的规则对
  7. 条件缓存 — 带TTL的条件评估缓存，减少重复计算
  8. 规则导出/导入 — JSON/YAML序列化，支持版本管理
  9. 预置规则模板 — 常用交易规则直接引用
  10. 后评估钩子 — 规则评估前后的回调系统
  11. 优先级继承 — 父规则触发子规则时继承优先级
  12. 执行耗时追踪 — 每条规则评估/执行的延迟统计
"""
import copy
import hashlib
import json
import re
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 枚举
# ═══════════════════════════════════════════════════════════════

class RuleOperator(Enum):
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_OR_EQUAL = ">="
    LESS_OR_EQUAL = "<="
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    MATCHES = "matches"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"
    BETWEEN = "between"           # value between [min, max]
    EXISTS = "exists"             # field exists in data
    ALL_OF = "all_of"             # multi-value: field contains ALL values
    ANY_OF = "any_of"             # multi-value: field contains ANY value
    CROSSES = "crosses"           # 上穿/下穿检测（需要 _prev_data）


class RuleActionType(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    DELAY = "delay"
    ESCALATE = "escalate"
    SCORE = "score"               # 新增：加权评分
    CASCADE = "cascade"           # 新增：级联触发其他规则
    FLAG = "flag"                 # 新增：标记但不改变决策状态


class RuleGroupCategory(Enum):
    """规则组分类"""
    RISK = "risk"                       # 风控规则
    SIGNAL = "signal"                   # 信号质量规则
    ORDER = "order"                     # 订单规则
    POSITION = "position"               # 持仓规则
    CAPITAL = "capital"                 # 资金规则
    MARKET = "market"                   # 市场条件规则
    STRATEGY = "strategy"               # 策略特定规则
    COMPLIANCE = "compliance"           # 合规规则
    CUSTOM = "custom"                   # 自定义规则


class ActivationMode(Enum):
    """规则激活模式"""
    ALWAYS = "always"                   # 始终激活
    TIME_WINDOW = "time_window"         # 时间窗口
    MARKET_REGIME = "market_regime"     # 市场状态
    VOLATILITY = "volatility"           # 波动率条件
    ACCOUNT_CONDITION = "account_condition"  # 账户条件
    COMPOSITE = "composite"             # 组合条件


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class RulePerformance:
    """规则性能统计"""
    rule_id: str = ""
    total_evaluations: int = 0
    matches: int = 0
    true_positives: int = 0       # 触发且最终正确执行
    false_positives: int = 0      # 触发但不应执行
    total_pnl_impact: float = 0.0 # 该规则导致的累计PnL影响
    avg_eval_latency_us: float = 0.0  # 平均评估延迟（微秒）
    avg_exec_latency_us: float = 0.0  # 平均执行延迟（微秒）
    last_matched_at: Optional[datetime] = None
    last_evaluated_at: Optional[datetime] = None
    latency_history: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def match_rate(self) -> float:
        return self.matches / max(self.total_evaluations, 1)

    @property
    def precision(self) -> float:
        return self.true_positives / max(self.matches, 1)

    def record_eval(self, matched: bool, latency_us: float):
        self.total_evaluations += 1
        self.latency_history.append(latency_us)
        self.avg_eval_latency_us = sum(self.latency_history) / len(self.latency_history)
        self.last_evaluated_at = datetime.now()
        if matched:
            self.matches += 1
            self.last_matched_at = datetime.now()

    def record_outcome(self, was_correct: bool, pnl_impact: float = 0.0):
        if was_correct:
            self.true_positives += 1
        else:
            self.false_positives += 1
        self.total_pnl_impact += pnl_impact


@dataclass
class ActivationWindow:
    """规则激活窗口"""
    mode: ActivationMode = ActivationMode.ALWAYS
    # 时间窗口
    start_hour: int = 0
    end_hour: int = 24
    days_of_week: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])
    # 市场状态
    allowed_regimes: List[str] = field(default_factory=list)  # ["trending", "ranging", "volatile"]
    blocked_regimes: List[str] = field(default_factory=list)
    # 波动率
    min_volatility_pct: Optional[float] = None
    max_volatility_pct: Optional[float] = None
    # 账户条件
    min_equity: Optional[float] = None
    max_exposure_pct: Optional[float] = None
    # 组合
    sub_conditions: List['ActivationWindow'] = field(default_factory=list)
    require_all: bool = True  # True=AND, False=OR


@dataclass
class ConditionCacheEntry:
    """条件缓存条目"""
    result: bool
    cached_at: float
    ttl_seconds: float
    data_hash: str = ""

    @property
    def is_valid(self) -> bool:
        return (time.time() - self.cached_at) < self.ttl_seconds


# ═══════════════════════════════════════════════════════════════
# Rule — 增强版
# ═══════════════════════════════════════════════════════════════

class Rule:
    """增强版规则"""

    __slots__ = (
        "rule_id", "name", "condition", "action", "priority",
        "enabled", "description", "action_params", "group",
        "weight", "min_confidence", "activation_window",
        "cascade_rules", "cooldown_seconds", "perf",
        "version", "created_at", "updated_at", "tags",
        "_last_triggered_at",
        "matched_count", "executed_count",
    )

    def __init__(
        self,
        rule_id: str,
        name: str,
        condition: Dict[str, Any],
        action: RuleActionType,
        priority: int = 100,
        enabled: bool = True,
        description: str = "",
        action_params: Dict[str, Any] = None,
        # ── 新增参数 ──
        group: Optional[RuleGroupCategory] = None,
        weight: float = 1.0,
        min_confidence: float = 0.0,
        activation_window: Optional[ActivationWindow] = None,
        cascade_rules: List[str] = None,
        cooldown_seconds: float = 0.0,
        tags: List[str] = None,
        version: str = "1.0",
    ):
        self.rule_id = rule_id
        self.name = name
        self.condition = condition
        self.action = action
        self.priority = priority
        self.enabled = enabled
        self.description = description
        self.action_params = action_params or {}
        self.group = group or RuleGroupCategory.CUSTOM
        self.weight = weight
        self.min_confidence = min_confidence
        self.activation_window = activation_window or ActivationWindow()
        self.cascade_rules = cascade_rules or []
        self.cooldown_seconds = cooldown_seconds
        self.tags = tags or []
        self.version = version
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # 性能追踪
        self.perf = RulePerformance(rule_id=rule_id)

        # 冷却状态
        self._last_triggered_at: Optional[float] = None

        # 向前兼容
        self.matched_count: int = 0
        self.executed_count: int = 0

    # ── 条件评估 ──

    def evaluate_condition(
        self,
        data: Dict[str, Any],
        prev_data: Dict[str, Any] = None,
        use_cache: bool = False,
    ) -> Tuple[bool, float]:
        """
        评估条件

        Returns:
            (matched: bool, score: float)  — score 仅在 action=SCORE 时有意义
        """
        if not self.enabled:
            return False, 0.0

        start = time.perf_counter_ns()

        # 冷却检查
        if self._is_in_cooldown():
            return False, 0.0

        try:
            matched = self._evaluate_node(self.condition, data, prev_data or {})
        except Exception as e:
            logger.error(f"Rule {self.rule_id} evaluation failed: {e}")
            matched = False

        latency_us = (time.perf_counter_ns() - start) / 1000
        self.perf.record_eval(matched, latency_us)
        return matched, self.weight if matched else 0.0

    def _is_in_cooldown(self) -> bool:
        if self.cooldown_seconds <= 0 or self._last_triggered_at is None:
            return False
        return (time.time() - self._last_triggered_at) < self.cooldown_seconds

    def _mark_triggered(self):
        self._last_triggered_at = time.time()

    def _evaluate_node(
        self,
        node: Dict[str, Any],
        data: Dict[str, Any],
        prev_data: Dict[str, Any],
    ) -> bool:
        if "operator" in node:
            return self._evaluate_operator(node, data, prev_data)
        elif "and" in node:
            return all(self._evaluate_node(n, data, prev_data) for n in node["and"])
        elif "or" in node:
            return any(self._evaluate_node(n, data, prev_data) for n in node["or"])
        elif "not" in node:
            return not self._evaluate_node(node["not"], data, prev_data)
        return False

    def _evaluate_operator(
        self,
        node: Dict[str, Any],
        data: Dict[str, Any],
        prev_data: Dict[str, Any],
    ) -> bool:
        operator = RuleOperator(node["operator"])
        field = node["field"]
        value = node.get("value")

        field_value = self._get_field_value(data, field)

        try:
            if operator == RuleOperator.EQUALS:
                return field_value == value
            elif operator == RuleOperator.NOT_EQUALS:
                return field_value != value
            elif operator == RuleOperator.GREATER_THAN:
                return float(field_value or 0) > float(value)
            elif operator == RuleOperator.LESS_THAN:
                return float(field_value or 0) < float(value)
            elif operator == RuleOperator.GREATER_OR_EQUAL:
                return float(field_value or 0) >= float(value)
            elif operator == RuleOperator.LESS_OR_EQUAL:
                return float(field_value or 0) <= float(value)
            elif operator == RuleOperator.IN:
                return field_value in (value if isinstance(value, list) else [value])
            elif operator == RuleOperator.NOT_IN:
                return field_value not in (value if isinstance(value, list) else [value])
            elif operator == RuleOperator.CONTAINS:
                return str(value) in str(field_value or "")
            elif operator == RuleOperator.MATCHES:
                return bool(re.match(str(value), str(field_value or "")))
            elif operator == RuleOperator.IS_TRUE:
                return bool(field_value) is True
            elif operator == RuleOperator.IS_FALSE:
                return bool(field_value) is False
            elif operator == RuleOperator.BETWEEN:
                if not isinstance(value, list) or len(value) < 2:
                    return False
                return float(value[0]) <= float(field_value or 0) <= float(value[1])
            elif operator == RuleOperator.EXISTS:
                return field_value is not None
            elif operator == RuleOperator.ALL_OF:
                if not isinstance(value, list) or not isinstance(field_value, list):
                    return False
                return all(v in field_value for v in value)
            elif operator == RuleOperator.ANY_OF:
                if not isinstance(value, list) or not isinstance(field_value, list):
                    return False
                return any(v in field_value for v in value)
            elif operator == RuleOperator.CROSSES:
                prev = self._get_field_value(prev_data, field)
                if prev is None or field_value is None:
                    return False
                cross_up = float(prev) <= float(value) and float(field_value) > float(value)
                cross_down = float(prev) >= float(value) and float(field_value) < float(value)
                direction = node.get("direction", "any")
                if direction == "up":
                    return cross_up
                elif direction == "down":
                    return cross_down
                return cross_up or cross_down
            return False
        except (ValueError, TypeError, AttributeError):
            return False

    def _get_field_value(self, data: Dict[str, Any], field: str) -> Any:
        keys = field.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                index = int(key)
                value = value[index] if index < len(value) else None
            else:
                return None
        return value

    # ── 激活检查 ──

    def is_active(self, context: Dict[str, Any] = None) -> bool:
        """
        检查规则在当前上下文中是否激活

        检查激活窗口（时间、市场状态、波动率、账户条件）
        """
        if not self.enabled:
            return False
        aw = self.activation_window
        ctx = context or {}

        if aw.mode == ActivationMode.ALWAYS:
            return True

        if aw.mode == ActivationMode.TIME_WINDOW:
            now = datetime.now()
            if now.weekday() not in aw.days_of_week:
                return False
            if not (aw.start_hour <= now.hour < aw.end_hour):
                return False
            return True

        if aw.mode == ActivationMode.MARKET_REGIME:
            regime = ctx.get("market_regime", "")
            if aw.blocked_regimes and regime in aw.blocked_regimes:
                return False
            if aw.allowed_regimes and regime not in aw.allowed_regimes:
                return False
            return True

        if aw.mode == ActivationMode.VOLATILITY:
            vol = ctx.get("volatility_pct", 0)
            if aw.min_volatility_pct is not None and vol < aw.min_volatility_pct:
                return False
            if aw.max_volatility_pct is not None and vol > aw.max_volatility_pct:
                return False
            return True

        if aw.mode == ActivationMode.ACCOUNT_CONDITION:
            eq = ctx.get("equity", 0)
            exp = ctx.get("exposure_pct", 0)
            if aw.min_equity is not None and eq < aw.min_equity:
                return False
            if aw.max_exposure_pct is not None and exp > aw.max_exposure_pct:
                return False
            return True

        if aw.mode == ActivationMode.COMPOSITE:
            if aw.require_all:
                return all(
                    self._check_sub_window(sw, ctx) for sw in aw.sub_conditions
                )
            else:
                return any(
                    self._check_sub_window(sw, ctx) for sw in aw.sub_conditions
                )

        return True

    def _check_sub_window(self, aw: ActivationWindow, ctx: Dict[str, Any]) -> bool:
        """检查子激活窗口（简化版）"""
        if aw.mode == ActivationMode.TIME_WINDOW:
            now = datetime.now()
            return now.weekday() in aw.days_of_week and aw.start_hour <= now.hour < aw.end_hour
        if aw.mode == ActivationMode.VOLATILITY:
            vol = ctx.get("volatility_pct", 0)
            lo = aw.min_volatility_pct or 0
            hi = aw.max_volatility_pct or float("inf")
            return lo <= vol <= hi
        if aw.mode == ActivationMode.ACCOUNT_CONDITION:
            eq = ctx.get("equity", 0)
            exp = ctx.get("exposure_pct", 0)
            if aw.min_equity is not None and eq < aw.min_equity:
                return False
            if aw.max_exposure_pct is not None and exp > aw.max_exposure_pct:
                return False
            return True
        return True

    # ── 动作执行 ──

    def execute_action(self, decision: Any) -> Dict[str, Any]:
        """执行规则动作"""
        self.executed_count += 1
        self.matched_count += 1
        self._mark_triggered()

        result = {"rule_id": self.rule_id, "action": self.action.value, "success": True}

        start = time.perf_counter_ns()
        try:
            if self.action == RuleActionType.APPROVE:
                decision.status = "approved"
            elif self.action == RuleActionType.REJECT:
                decision.status = "rejected"
                decision.rejection_reason = self.action_params.get(
                    "reason", f"Rule {self.name} triggered"
                )
            elif self.action == RuleActionType.MODIFY:
                for key, val in self.action_params.get("modifications", {}).items():
                    self._set_field_value(decision.data, key, val)
                result["modifications"] = self.action_params.get("modifications")
            elif self.action == RuleActionType.DELAY:
                result["delay_seconds"] = self.action_params.get("seconds", 60)
            elif self.action == RuleActionType.ESCALATE:
                new_priority = self.action_params.get("priority", 0)
                if hasattr(decision, 'priority'):
                    decision.priority = min(decision.priority, new_priority)
            elif self.action == RuleActionType.SCORE:
                result["score"] = self.weight
                result["reason"] = self.description
            elif self.action == RuleActionType.CASCADE:
                result["cascade_to"] = self.cascade_rules
            elif self.action == RuleActionType.FLAG:
                flags = getattr(decision, '_flags', [])
                flags.append({
                    "rule_id": self.rule_id,
                    "reason": self.action_params.get("reason", self.description),
                    "severity": self.action_params.get("severity", "info"),
                })
                if hasattr(decision, '_flags'):
                    decision._flags = flags
                else:
                    setattr(decision, '_flags', flags)
                result["flags"] = flags

            return result
        except Exception as e:
            logger.error(f"Rule {self.rule_id} action execution failed: {e}")
            result["success"] = False
            result["error"] = str(e)
            return result
        finally:
            latency_us = (time.perf_counter_ns() - start) / 1000
            self.perf.avg_exec_latency_us = (
                self.perf.avg_exec_latency_us * 0.9 + latency_us * 0.1
            )

    def _set_field_value(self, data: Dict[str, Any], field: str, value: Any):
        keys = field.split(".")
        current = data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value

    # ── 序列化 ──

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "condition": self.condition,
            "action": self.action.value,
            "priority": self.priority,
            "enabled": self.enabled,
            "description": self.description,
            "action_params": self.action_params,
            "group": self.group.value,
            "weight": self.weight,
            "min_confidence": self.min_confidence,
            "activation_window": {
                "mode": self.activation_window.mode.value,
                "start_hour": self.activation_window.start_hour,
                "end_hour": self.activation_window.end_hour,
                "days_of_week": self.activation_window.days_of_week,
                "allowed_regimes": self.activation_window.allowed_regimes,
                "blocked_regimes": self.activation_window.blocked_regimes,
                "min_volatility_pct": self.activation_window.min_volatility_pct,
                "max_volatility_pct": self.activation_window.max_volatility_pct,
                "min_equity": self.activation_window.min_equity,
                "max_exposure_pct": self.activation_window.max_exposure_pct,
            },
            "cascade_rules": self.cascade_rules,
            "cooldown_seconds": self.cooldown_seconds,
            "tags": self.tags,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "perf": {
                "total_evaluations": self.perf.total_evaluations,
                "matches": self.perf.matches,
                "match_rate": round(self.perf.match_rate, 4),
                "precision": round(self.perf.precision, 4),
                "true_positives": self.perf.true_positives,
                "false_positives": self.perf.false_positives,
                "total_pnl_impact": round(self.perf.total_pnl_impact, 2),
                "avg_eval_latency_us": round(self.perf.avg_eval_latency_us, 1),
                "avg_exec_latency_us": round(self.perf.avg_exec_latency_us, 1),
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Rule':
        aw_data = data.get("activation_window", {})
        aw = ActivationWindow(
            mode=ActivationMode(aw_data.get("mode", "always")),
            start_hour=aw_data.get("start_hour", 0),
            end_hour=aw_data.get("end_hour", 24),
            days_of_week=aw_data.get("days_of_week", [0, 1, 2, 3, 4, 5, 6]),
            allowed_regimes=aw_data.get("allowed_regimes", []),
            blocked_regimes=aw_data.get("blocked_regimes", []),
            min_volatility_pct=aw_data.get("min_volatility_pct"),
            max_volatility_pct=aw_data.get("max_volatility_pct"),
            min_equity=aw_data.get("min_equity"),
            max_exposure_pct=aw_data.get("max_exposure_pct"),
        )
        return cls(
            rule_id=data["rule_id"],
            name=data["name"],
            condition=data["condition"],
            action=RuleActionType(data["action"]),
            priority=data.get("priority", 100),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            action_params=data.get("action_params", {}),
            group=RuleGroupCategory(data.get("group", "custom")),
            weight=data.get("weight", 1.0),
            min_confidence=data.get("min_confidence", 0.0),
            activation_window=aw,
            cascade_rules=data.get("cascade_rules", []),
            cooldown_seconds=data.get("cooldown_seconds", 0.0),
            tags=data.get("tags", []),
            version=data.get("version", "1.0"),
        )


# ═══════════════════════════════════════════════════════════════
# RuleBasedEngine — 增强版
# ═══════════════════════════════════════════════════════════════

class RuleBasedEngine:
    """
    基于规则的决策引擎 — 完善强化版

    新增核心能力：
      - 规则组/命名空间管理
      - 加权评分系统（score action）
      - 级联规则链式触发
      - 激活窗口（时间/市场/波动率条件）
      - 规则性能追踪
      - 冲突自动检测
      - 条件缓存（带TTL）
      - 规则导出/导入
      - 预置模板
      - 后评估钩子
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        rule_cfg = config.get("rule_engine", {}) if config else {}

        # ── 规则集合 ──
        self._rules: List[Rule] = []
        self._rules_by_id: Dict[str, Rule] = {}
        self._rules_by_group: Dict[RuleGroupCategory, List[Rule]] = defaultdict(list)
        self._rules_by_priority: Dict[int, List[Rule]] = defaultdict(list)

        # ── 条件缓存 ──
        self._condition_cache: Dict[str, ConditionCacheEntry] = {}
        self._cache_ttl = rule_cfg.get("cache_ttl_seconds", 5.0)
        self._cache_enabled = rule_cfg.get("cache_enabled", True)
        self._cache_lock = threading.Lock()

        # ── 钩子系统 ──
        self._pre_eval_hooks: List[Callable] = []
        self._post_eval_hooks: List[Callable] = []
        self._on_rule_match_hooks: List[Callable] = []
        self._on_reject_hooks: List[Callable] = []

        # ── 冲突检测 ──
        self._conflict_pairs: List[Tuple[str, str, str]] = []  # (rule_a, rule_b, reason)

        # ── 规则版本历史 ──
        self._rule_versions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # ── 上下文 ──
        self._eval_context: Dict[str, Any] = {}

        # 向前兼容
        self._rules_by_type: Dict[str, List[Rule]] = defaultdict(list)

        logger.info(
            f"RuleBasedEngine enhanced: cache={self._cache_enabled} (ttl={self._cache_ttl}s)"
        )

    # ═══════════════════════════════════════════════════════
    # 规则生命周期管理
    # ═══════════════════════════════════════════════════════

    def add_rule(self, rule: Rule):
        """添加规则"""
        self._rules.append(rule)
        self._rules_by_id[rule.rule_id] = rule
        self._rules_by_group[rule.group].append(rule)
        self._rules_by_priority[rule.priority].append(rule)
        self._rules_by_type[rule.action.value].append(rule)
        self._save_version(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """移除规则"""
        rule = self._rules_by_id.pop(rule_id, None)
        if not rule:
            return False
        self._rules = [r for r in self._rules if r.rule_id != rule_id]

        if rule.group in self._rules_by_group:
            self._rules_by_group[rule.group] = [
                r for r in self._rules_by_group[rule.group] if r.rule_id != rule_id
            ]
        if rule.priority in self._rules_by_priority:
            self._rules_by_priority[rule.priority] = [
                r for r in self._rules_by_priority[rule.priority] if r.rule_id != rule_id
            ]
        if rule.action.value in self._rules_by_type:
            self._rules_by_type[rule.action.value] = [
                r for r in self._rules_by_type[rule.action.value] if r.rule_id != rule_id
            ]
        return True

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """更新规则（保留版本历史）"""
        rule = self._rules_by_id.get(rule_id)
        if not rule:
            return False
        self._save_version(rule)
        for key, val in updates.items():
            if hasattr(rule, key):
                setattr(rule, key, val)
        rule.updated_at = datetime.now()
        return True

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        return self._rules_by_id.get(rule_id)

    def get_rules(self) -> List[Rule]:
        return list(self._rules)

    def disable_rule(self, rule_id: str):
        rule = self._rules_by_id.get(rule_id)
        if rule:
            rule.enabled = False

    def enable_rule(self, rule_id: str):
        rule = self._rules_by_id.get(rule_id)
        if rule:
            rule.enabled = True

    def _save_version(self, rule: Rule):
        self._rule_versions[rule.rule_id].append({
            "version": rule.version,
            "timestamp": datetime.now().isoformat(),
            "snapshot": rule.to_dict(),
        })
        max_versions = self.config.get("rule_engine", {}).get("max_rule_versions", 20)
        if len(self._rule_versions[rule.rule_id]) > max_versions:
            self._rule_versions[rule.rule_id] = self._rule_versions[rule.rule_id][-max_versions:]

    def get_rule_versions(self, rule_id: str) -> List[Dict[str, Any]]:
        return self._rule_versions.get(rule_id, [])

    # ═══════════════════════════════════════════════════════
    # 规则组管理
    # ═══════════════════════════════════════════════════════

    def get_rules_by_group(self, group: RuleGroupCategory) -> List[Rule]:
        return self._rules_by_group.get(group, [])

    def get_rules_by_type(self, action_type: str) -> List[Rule]:
        return self._rules_by_type.get(action_type, [])

    def get_group_summary(self) -> Dict[str, Any]:
        """获取规则组摘要"""
        summary = {}
        for group, rules in self._rules_by_group.items():
            enabled = sum(1 for r in rules if r.enabled)
            summary[group.value] = {
                "total": len(rules),
                "enabled": enabled,
                "total_matches": sum(r.matched_count for r in rules),
                "total_evaluations": sum(r.perf.total_evaluations for r in rules),
            }
        return summary

    # ═══════════════════════════════════════════════════════
    # 核心：规则评估（增强）
    # ═══════════════════════════════════════════════════════

    def set_eval_context(self, context: Dict[str, Any]):
        """设置评估上下文（市场状态、波动率、账户信息等）"""
        self._eval_context = context

    def evaluate(
        self,
        data: Dict[str, Any],
        prev_data: Dict[str, Any] = None,
        context: Dict[str, Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        评估所有规则

        增强点：
          - 激活窗口检查
          - 条件缓存
          - 加权评分汇总
          - 前/后评估钩子
        """
        ctx = context or self._eval_context

        # 前评估钩子
        for hook in self._pre_eval_hooks:
            try:
                hook(data, ctx)
            except Exception as e:
                logger.debug(f"Pre-eval hook error: {e}")

        results = []
        priorities = sorted(self._rules_by_priority.keys(), reverse=True)

        for priority in priorities:
            for rule in self._rules_by_priority[priority]:
                # 激活窗口检查
                if not rule.is_active(ctx):
                    continue

                # 条件缓存
                cache_key = f"{rule.rule_id}:{self._data_fingerprint(data)}"
                if self._cache_enabled:
                    cached = self._get_cached(cache_key)
                    if cached is not None:
                        if cached:
                            results.append(self._build_result(rule))
                        continue

                matched, score = rule.evaluate_condition(data, prev_data)

                # 写缓存
                if self._cache_enabled:
                    self._set_cache(cache_key, matched)

                if matched:
                    # P0: min_confidence检查——数据置信度低于规则要求的跳过
                    if rule.min_confidence > 0:
                        data_confidence = float(data.get("confidence", 0) or 0)
                        if data_confidence < rule.min_confidence:
                            continue
                    rule.matched_count += 1
                    results.append(self._build_result(rule, score))

                    # 匹配钩子
                    for hook in self._on_rule_match_hooks:
                        try:
                            hook(rule, data, ctx)
                        except Exception as e:
                            logger.debug(f"On-rule-match hook error: {e}")

        # 后评估钩子
        for hook in self._post_eval_hooks:
            try:
                hook(results, data, ctx)
            except Exception as e:
                logger.debug(f"Post-eval hook error: {e}")

        return results

    def evaluate_with_score(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """
        带评分汇总的评估

        Returns:
            (results, scores_by_group)  — 按规则组汇总的加权评分
        """
        results = self.evaluate(data, context=context)
        scores_by_group: Dict[str, float] = defaultdict(float)

        for r in results:
            g = r.get("group", "custom")
            scores_by_group[g] += r.get("score", 0.0)

        return results, dict(scores_by_group)

    def _build_result(self, rule: Rule, score: float = 0.0) -> Dict[str, Any]:
        return {
            "rule_id": rule.rule_id,
            "name": rule.name,
            "priority": rule.priority,
            "action": rule.action.value,
            "group": rule.group.value,
            "score": score,
            "weight": rule.weight,
            "description": rule.description,
            "tags": rule.tags,
            "cascade_rules": rule.cascade_rules,
        }

    # ═══════════════════════════════════════════════════════
    # 核心：规则执行（含级联）
    # ═══════════════════════════════════════════════════════

    def execute_rules(
        self,
        decision: Any,
        context: Dict[str, Any] = None,
        max_cascade_depth: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        执行匹配的规则

        增强点：
          - 级联规则链式触发
          - 激活窗口检查
          - 深度限制防无限循环
          - 正向兼容：按优先级终止
        """
        ctx = context or self._eval_context
        results = []
        executed_rule_ids: Set[str] = set()

        priorities = sorted(self._rules_by_priority.keys(), reverse=True)

        for priority in priorities:
            for rule in self._rules_by_priority[priority]:
                if rule.rule_id in executed_rule_ids:
                    continue
                if not rule.is_active(ctx):
                    continue

                matched, _ = rule.evaluate_condition(decision.data)
                if not matched:
                    continue

                rule.matched_count += 1
                executed_rule_ids.add(rule.rule_id)

                action_result = rule.execute_action(decision)
                results.append(action_result)

                # 级联触发
                if rule.cascade_rules and max_cascade_depth > 0:
                    cascade_results = self._execute_cascade(
                        rule.cascade_rules, decision, ctx,
                        executed_rule_ids, max_cascade_depth - 1,
                    )
                    results.extend(cascade_results)

                # 正向兼容：APPROVE/REJECT 终止
                if rule.action in (RuleActionType.APPROVE, RuleActionType.REJECT):
                    return results

        return results

    def _execute_cascade(
        self,
        cascade_ids: List[str],
        decision: Any,
        ctx: Dict[str, Any],
        executed: Set[str],
        depth: int,
    ) -> List[Dict[str, Any]]:
        """递归执行级联规则"""
        results = []
        for cid in cascade_ids:
            if cid in executed:
                continue
            child_rule = self._rules_by_id.get(cid)
            if not child_rule or not child_rule.enabled:
                continue
            if not child_rule.is_active(ctx):
                continue

            matched, _ = child_rule.evaluate_condition(decision.data)
            if not matched:
                continue

            executed.add(cid)
            child_rule.matched_count += 1
            cr = child_rule.execute_action(decision)
            results.append(cr)

            # 递归级联
            if child_rule.cascade_rules and depth > 0:
                sub = self._execute_cascade(
                    child_rule.cascade_rules, decision, ctx, executed, depth - 1,
                )
                results.extend(sub)

        return results

    # ═══════════════════════════════════════════════════════
    # 冲突检测
    # ═══════════════════════════════════════════════════════

    def detect_conflicts(self) -> List[Tuple[str, str, str]]:
        """
        检测规则冲突

        检测维度：
          1. 同字段反方向规则（如 A: quantity>10 APPROVE, B: quantity>10 REJECT）
          2. 条件重叠但行为相反
          3. 优先级相同的互斥规则
        """
        conflicts = []
        rules = [r for r in self._rules if r.enabled]

        for i, ra in enumerate(rules):
            for j, rb in enumerate(rules):
                if j <= i:
                    continue

                # 维度1: 同一字段矛盾的比较操作符
                ca = ra.condition
                cb = rb.condition
                if self._conditions_contradictory(ca, cb):
                    # 相同动作类型的矛盾条件不构成冲突（如两条REJECT规则指向同一字段是有意的冗余保护）
                    if ra.action == rb.action:
                        continue
                    # 一条APPROVE一条REJECT指向同一矛盾条件 → 冲突
                    if (ra.action == RuleActionType.APPROVE and rb.action == RuleActionType.REJECT) or \
                       (ra.action == RuleActionType.REJECT and rb.action == RuleActionType.APPROVE):
                        conflicts.append((ra.rule_id, rb.rule_id, "approve_vs_reject_same_field"))
                        continue

                # 维度2: 条件完全重叠但行为不同
                if ca == cb:
                    if ra.action != rb.action:
                        conflicts.append((ra.rule_id, rb.rule_id,
                                         f"identical_conditions_different_actions: {ra.action.value} vs {rb.action.value}"))

                # 维度3: 同优先级互斥
                if ra.priority == rb.priority and ra.action == RuleActionType.REJECT and rb.action == RuleActionType.REJECT:
                    if ca.get("field") == cb.get("field") and ca.get("operator") == cb.get("operator"):
                        pass  # 两条拒绝规则指向同一字段可能是有意的

        self._conflict_pairs = conflicts
        return conflicts

    def _conditions_contradictory(self, ca: Dict, cb: Dict) -> bool:
        """检查两个条件是否矛盾（同一字段，相反操作符）
        
        P6: 增加值域重叠检查，避免非重叠范围被误判为冲突
        例如: confidence < 0.15 (REJECT) vs confidence >= 0.85 (APPROVE)
        虽然操作符相反，但值域不重叠，不构成真正冲突
        """
        op_opposites = {
            ">": "<=", "<": ">=", ">=": "<", "<=": ">",
            "==": "!=", "!=": "==",
        }
        fa = ca.get("field", "")
        fb = cb.get("field", "")
        oa = ca.get("operator", "")
        ob = cb.get("operator", "")
        if fa == fb and fa:
            if op_opposites.get(oa) == ob or op_opposites.get(ob) == oa:
                # P6: 操作符相反，进一步检查值域是否重叠
                if not self._value_ranges_overlap(ca, cb):
                    return False  # 值域不重叠，不构成真正冲突
                return True
        return False
    
    @staticmethod
    def _value_range_bounds(condition: Dict) -> tuple:
        """P6: 获取条件的值域边界
        
        返回 (lower, upper, lower_inclusive, upper_inclusive)
        其中 lower/upper 为 None 表示无界
        """
        op = condition.get("operator", "")
        val = condition.get("value")
        if val is None:
            return (None, None, False, False)
        
        try:
            val = float(val)
        except (TypeError, ValueError):
            return (None, None, False, False)
        
        if op == "<":
            return (None, val, False, False)
        elif op == "<=":
            return (None, val, False, True)
        elif op == ">":
            return (val, None, False, False)
        elif op == ">=":
            return (val, None, True, False)
        elif op == "==":
            return (val, val, True, True)
        elif op == "!=":
            return (None, None, False, False)  # != 总是可能重叠
        return (None, None, False, False)
    
    @classmethod
    def _value_ranges_overlap(cls, ca: Dict, cb: Dict) -> bool:
        """P6: 检查两个条件的值域是否有重叠
        
        例如: confidence < 0.15 和 confidence >= 0.85 → 不重叠
              confidence < 0.5 和 confidence >= 0.3 → 重叠
        """
        lo_a, hi_a, lo_inc_a, hi_inc_a = cls._value_range_bounds(ca)
        lo_b, hi_b, lo_inc_b, hi_inc_b = cls._value_range_bounds(cb)
        
        # 如果任一条件无界，则必然重叠
        if lo_a is None and hi_a is None:
            return True
        if lo_b is None and hi_b is None:
            return True
        
        # 计算有效下界
        effective_lo_a = lo_a if lo_a is not None else float('-inf')
        effective_lo_b = lo_b if lo_b is not None else float('-inf')
        effective_hi_a = hi_a if hi_a is not None else float('inf')
        effective_hi_b = hi_b if hi_b is not None else float('inf')
        
        max_lo = max(effective_lo_a, effective_lo_b)
        min_hi = min(effective_hi_a, effective_hi_b)
        
        if max_lo < min_hi:
            return True  # 有重叠区间
        if max_lo == min_hi:
            # 边界相等时需要检查是否都包含边界
            lo_match = (max_lo == effective_lo_a and lo_inc_a) or (max_lo == effective_lo_b and lo_inc_b)
            hi_match = (min_hi == effective_hi_a and hi_inc_a) or (min_hi == effective_hi_b and hi_inc_b)
            return lo_match or hi_match
        
        return False  # 无重叠

    # ═══════════════════════════════════════════════════════
    # 条件缓存
    # ═══════════════════════════════════════════════════════

    def _get_cached(self, key: str) -> Optional[bool]:
        with self._cache_lock:
            entry = self._condition_cache.get(key)
            if entry and entry.is_valid:
                return entry.result
            if entry:
                del self._condition_cache[key]
            return None

    def _set_cache(self, key: str, result: bool):
        with self._cache_lock:
            self._condition_cache[key] = ConditionCacheEntry(
                result=result, cached_at=time.time(), ttl_seconds=self._cache_ttl,
            )
            # 清理过期
            self._clean_expired_cache()

    def _clean_expired_cache(self):
        now = time.time()
        expired = [k for k, v in self._condition_cache.items() if not v.is_valid]
        for k in expired:
            del self._condition_cache[k]

    def _data_fingerprint(self, data: Dict[str, Any]) -> str:
        """数据指纹（用于缓存键）"""
        relevant_keys = ["symbol", "direction", "price", "quantity", "confidence"]
        subset = {k: data.get(k) for k in relevant_keys if k in data}
        return hashlib.md5(json.dumps(subset, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def invalidate_cache(self):
        """清空条件缓存"""
        with self._cache_lock:
            self._condition_cache.clear()

    # ═══════════════════════════════════════════════════════
    # 钩子系统
    # ═══════════════════════════════════════════════════════

    def register_pre_eval_hook(self, hook: Callable):
        self._pre_eval_hooks.append(hook)

    def register_post_eval_hook(self, hook: Callable):
        self._post_eval_hooks.append(hook)

    def register_on_match_hook(self, hook: Callable):
        self._on_rule_match_hooks.append(hook)

    def register_on_reject_hook(self, hook: Callable):
        self._on_reject_hooks.append(hook)

    # ═══════════════════════════════════════════════════════
    # 预置规则模板
    # ═══════════════════════════════════════════════════════

    def load_default_templates(self) -> List[Rule]:
        """加载预置规则模板"""
        templates = [
            # ── 风控规则 ──
            Rule(
                rule_id="risk_max_exposure",
                name="单币种最大暴露",
                condition={"field": "exposure", "operator": ">", "value": 0.3},
                action=RuleActionType.REJECT,
                priority=200,
                group=RuleGroupCategory.RISK,
                description="单币种暴露超过30%时拒绝开仓",
                tags=["risk", "exposure"],
            ),
            Rule(
                rule_id="risk_max_leverage",
                name="最大杠杆限制",
                condition={"field": "leverage", "operator": ">", "value": 20},
                action=RuleActionType.REJECT,
                priority=200,
                group=RuleGroupCategory.RISK,
                description="杠杆超过20x时拒绝开仓",
                tags=["risk", "leverage"],
            ),
            Rule(
                rule_id="risk_min_margin",
                name="最低保证金要求",
                condition={"field": "margin_after", "operator": "<", "value": 50},
                action=RuleActionType.REJECT,
                priority=190,
                group=RuleGroupCategory.RISK,
                description="交易后可用保证金低于50USDT时拒绝",
                tags=["risk", "margin"],
            ),
            Rule(
                rule_id="risk_daily_loss_limit",
                name="日内亏损上限",
                condition={
                    "and": [
                        {"field": "daily_loss_pct", "operator": ">", "value": 0.15},
                        {"field": "direction", "operator": "in", "value": ["long", "short"]},
                    ]
                },
                action=RuleActionType.REJECT,
                priority=250,
                group=RuleGroupCategory.RISK,
                description="日内亏损超过15%暂停新开仓",
                tags=["risk", "daily_loss"],
            ),
            Rule(
                rule_id="risk_consecutive_losses",
                name="连续亏损暂停",
                condition={"field": "consecutive_losses", "operator": ">=", "value": 5},
                action=RuleActionType.REJECT,
                priority=180,
                group=RuleGroupCategory.RISK,
                description="连续亏损5次暂停开仓",
                tags=["risk", "consecutive_losses"],
                cooldown_seconds=300,
            ),
            # ── 信号质量规则 ──
            Rule(
                rule_id="signal_low_confidence",
                name="低置信度过滤",
                condition={"field": "confidence", "operator": "<", "value": 0.15},
                action=RuleActionType.REJECT,
                priority=150,
                group=RuleGroupCategory.SIGNAL,
                description="置信度低于0.15的信号拒绝",
                tags=["signal", "confidence"],
            ),
            Rule(
                rule_id="signal_high_confidence_fast_track",
                name="高置信度快速通道",
                condition={"field": "confidence", "operator": ">=", "value": 0.85},
                action=RuleActionType.APPROVE,
                priority=220,
                group=RuleGroupCategory.SIGNAL,
                description="置信度>=0.85的信号自动批准",
                tags=["signal", "confidence"],
            ),
            Rule(
                rule_id="signal_multi_strategy_agree",
                name="多策略一致增强",
                condition={"field": "agreeing_strategies", "operator": ">=", "value": 3},
                action=RuleActionType.SCORE,
                priority=160,
                group=RuleGroupCategory.SIGNAL,
                weight=0.3,
                description="3个以上策略一致时加分",
                tags=["signal", "multi_strategy"],
            ),
            Rule(
                rule_id="signal_mtf_alignment",
                name="多时间框架对齐",
                condition={"field": "mtf_alignment", "operator": "is_true"},
                action=RuleActionType.SCORE,
                priority=160,
                group=RuleGroupCategory.SIGNAL,
                weight=0.25,
                description="多时间框架方向一致时加分",
                tags=["signal", "mtf"],
            ),
            # ── 订单规则 ──
            Rule(
                rule_id="order_min_quantity",
                name="最小交易量检查",
                condition={"field": "quantity", "operator": "<", "value": 0.001},
                action=RuleActionType.REJECT,
                priority=210,
                group=RuleGroupCategory.ORDER,
                description="交易量低于最小单位时拒绝",
                tags=["order", "quantity"],
            ),
            Rule(
                rule_id="order_price_deviation",
                name="价格偏离保护",
                condition={"field": "price_deviation_pct", "operator": ">", "value": 0.05},
                action=RuleActionType.DELAY,
                priority=140,
                group=RuleGroupCategory.ORDER,
                description="价格偏离超过5%时延迟60秒",
                action_params={"seconds": 60},
                tags=["order", "price_deviation"],
            ),
            # ── 持仓规则 ──
            Rule(
                rule_id="position_max_count",
                name="最大持仓数限制",
                condition={"field": "current_position_count", "operator": ">=", "value": 10},
                action=RuleActionType.REJECT,
                priority=170,
                group=RuleGroupCategory.POSITION,
                description="持仓数超过10个时拒绝新开仓",
                tags=["position", "count"],
            ),
            Rule(
                rule_id="position_reduce_only_close",
                name="减仓模式确认",
                condition={"field": "reduce_only", "operator": "is_true"},
                action=RuleActionType.APPROVE,
                priority=230,
                group=RuleGroupCategory.POSITION,
                description="减仓/平仓操作快速批准",
                tags=["position", "reduce_only"],
            ),
            # ── 市场条件规则 ──
            Rule(
                rule_id="market_high_volatility_reduce",
                name="高波动缩仓",
                condition={"field": "volatility_pct", "operator": ">", "value": 0.08},
                action=RuleActionType.MODIFY,
                priority=155,
                group=RuleGroupCategory.MARKET,
                description="波动率>8%时缩仓至50%",
                action_params={"modifications": {"quantity": 0.5}},
                tags=["market", "volatility"],
                activation_window=ActivationWindow(
                    mode=ActivationMode.VOLATILITY,
                    min_volatility_pct=0.08,
                ),
            ),
            Rule(
                rule_id="market_extreme_volatility_block",
                name="极端波动封禁",
                condition={"field": "volatility_pct", "operator": ">", "value": 0.15},
                action=RuleActionType.REJECT,
                priority=240,
                group=RuleGroupCategory.MARKET,
                description="波动率>15%时禁止开仓",
                tags=["market", "volatility"],
                activation_window=ActivationWindow(
                    mode=ActivationMode.VOLATILITY,
                    min_volatility_pct=0.15,
                ),
            ),
            # ── 资金规则 ──
            Rule(
                rule_id="capital_insufficient",
                name="资金不足拒绝",
                condition={"field": "capital_required", "operator": ">", "value": 0},
                action=RuleActionType.FLAG,
                priority=130,
                group=RuleGroupCategory.CAPITAL,
                description="标记资金需求",
                action_params={"reason": "capital_required", "severity": "info"},
                tags=["capital"],
            ),
            Rule(
                rule_id="capital_allocation_exceeded",
                name="资金分配超限",
                condition={"field": "allocation_used_pct", "operator": ">", "value": 0.95},
                action=RuleActionType.REJECT,
                priority=175,
                group=RuleGroupCategory.CAPITAL,
                description="策略资金分配超过95%时拒绝",
                tags=["capital", "allocation"],
            ),
        ]
        for t in templates:
            if t.rule_id not in self._rules_by_id:
                self.add_rule(t)
        return templates

    # ═══════════════════════════════════════════════════════
    # 规则配置加载 / 导出导入
    # ═══════════════════════════════════════════════════════

    def load_rules_from_config(self, config: List[Dict[str, Any]]):
        """从配置列表加载规则（向前兼容）"""
        for rule_config in config:
            action_type = RuleActionType(rule_config["action"])
            rule = Rule(
                rule_id=rule_config["rule_id"],
                name=rule_config["name"],
                condition=rule_config["condition"],
                action=action_type,
                priority=rule_config.get("priority", 100),
                enabled=rule_config.get("enabled", True),
                description=rule_config.get("description", ""),
                action_params=rule_config.get("action_params", {}),
                group=RuleGroupCategory(rule_config.get("group", "custom")),
                weight=rule_config.get("weight", 1.0),
                cascade_rules=rule_config.get("cascade_rules", []),
                cooldown_seconds=rule_config.get("cooldown_seconds", 0.0),
                tags=rule_config.get("tags", []),
            )
            if rule_config.get("activation_window"):
                aw = rule_config["activation_window"]
                rule.activation_window = ActivationWindow(
                    mode=ActivationMode(aw.get("mode", "always")),
                    start_hour=aw.get("start_hour", 0),
                    end_hour=aw.get("end_hour", 24),
                    days_of_week=aw.get("days_of_week", [0, 1, 2, 3, 4, 5, 6]),
                    allowed_regimes=aw.get("allowed_regimes", []),
                    blocked_regimes=aw.get("blocked_regimes", []),
                    min_volatility_pct=aw.get("min_volatility_pct"),
                    max_volatility_pct=aw.get("max_volatility_pct"),
                    min_equity=aw.get("min_equity"),
                    max_exposure_pct=aw.get("max_exposure_pct"),
                )
            self.add_rule(rule)

    def export_rules(self, include_disabled: bool = False) -> List[Dict[str, Any]]:
        """导出所有规则为JSON"""
        result = []
        for rule in self._rules:
            if not include_disabled and not rule.enabled:
                continue
            result.append(rule.to_dict())
        return result

    def export_rules_json(self, filepath: str, include_disabled: bool = False):
        """导出规则到JSON文件"""
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "rule_count": len(self._rules),
            "rules": self.export_rules(include_disabled=include_disabled),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Rules exported to {filepath} ({len(export_data['rules'])} rules)")

    def import_rules_json(self, filepath: str, replace_existing: bool = False):
        """从JSON文件导入规则"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if replace_existing:
            self._rules.clear()
            self._rules_by_id.clear()
            self._rules_by_group.clear()
            self._rules_by_priority.clear()
            self._rules_by_type.clear()

        for rd in data.get("rules", data if isinstance(data, list) else []):
            if rd["rule_id"] in self._rules_by_id and not replace_existing:
                logger.debug(f"Rule {rd['rule_id']} already exists, skipping")
                continue
            rule = Rule.from_dict(rd)
            self.add_rule(rule)

        logger.info(f"Rules imported from {filepath} ({len(data.get('rules', []))} rules)")

    # ═══════════════════════════════════════════════════════
    # 统计与监控
    # ═══════════════════════════════════════════════════════

    def get_rule_stats(self) -> Dict[str, Any]:
        """获取规则统计（增强版）"""
        stats = {}
        for rule in self._rules:
            stats[rule.rule_id] = {
                "name": rule.name,
                "enabled": rule.enabled,
                "priority": rule.priority,
                "group": rule.group.value,
                "action": rule.action.value,
                "version": rule.version,
                "matched_count": rule.matched_count,
                "executed_count": rule.executed_count,
                "match_rate": round(rule.perf.match_rate, 4),
                "precision": round(rule.perf.precision, 4),
                "total_evaluations": rule.perf.total_evaluations,
                "true_positives": rule.perf.true_positives,
                "false_positives": rule.perf.false_positives,
                "total_pnl_impact": round(rule.perf.total_pnl_impact, 2),
                "avg_eval_latency_us": round(rule.perf.avg_eval_latency_us, 1),
                "tags": rule.tags,
            }
        return stats

    def feed_rule_outcome(self, rule_id: str, was_correct: bool, pnl_impact: float = 0.0):
        """反馈规则执行结果（用于精度追踪）"""
        rule = self._rules_by_id.get(rule_id)
        if rule:
            rule.perf.record_outcome(was_correct, pnl_impact)

    def get_conflict_report(self) -> Dict[str, Any]:
        """获取冲突报告"""
        self.detect_conflicts()
        return {
            "conflict_count": len(self._conflict_pairs),
            "conflicts": [
                {"rule_a": a, "rule_b": b, "reason": r}
                for a, b, r in self._conflict_pairs
            ],
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._cache_lock:
            return {
                "enabled": self._cache_enabled,
                "ttl_seconds": self._cache_ttl,
                "entries": len(self._condition_cache),
                "valid_entries": sum(1 for v in self._condition_cache.values() if v.is_valid),
            }

    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎综合统计"""
        rules_list = self._rules
        enabled = sum(1 for r in rules_list if r.enabled)
        total_matches = sum(r.matched_count for r in rules_list)
        total_evaluations = sum(r.perf.total_evaluations for r in rules_list)

        return {
            "total_rules": len(rules_list),
            "enabled_rules": enabled,
            "disabled_rules": len(rules_list) - enabled,
            "total_matches": total_matches,
            "total_evaluations": total_evaluations,
            "overall_match_rate": round(total_matches / max(total_evaluations, 1), 4),
            "groups": self.get_group_summary(),
            "cache": self.get_cache_stats(),
            "conflicts": self.get_conflict_report(),
            "hooks": {
                "pre_eval": len(self._pre_eval_hooks),
                "post_eval": len(self._post_eval_hooks),
                "on_match": len(self._on_rule_match_hooks),
                "on_reject": len(self._on_reject_hooks),
            },
            "version": "2.0",
        }


# ═══════════════════════════════════════════════════════════════
# 预置规则模板工厂
# ═══════════════════════════════════════════════════════════════

def create_rule_from_template(
    template_name: str,
    overrides: Dict[str, Any] = None,
) -> Optional[Rule]:
    """根据模板名创建规则"""
    overrides = overrides or {}

    template_params = {
        "max_exposure": {
            "name": "最大暴露限制",
            "condition": {"field": "exposure", "operator": ">", "value": 0.3},
            "action": RuleActionType.REJECT,
            "priority": 200,
            "group": RuleGroupCategory.RISK,
        },
        "min_confidence": {
            "name": "最小置信度",
            "condition": {"field": "confidence", "operator": "<", "value": 0.15},
            "action": RuleActionType.REJECT,
            "priority": 150,
            "group": RuleGroupCategory.SIGNAL,
        },
        "high_confidence_approve": {
            "name": "高置信度快速通道",
            "condition": {"field": "confidence", "operator": ">=", "value": 0.85},
            "action": RuleActionType.APPROVE,
            "priority": 220,
            "group": RuleGroupCategory.SIGNAL,
        },
        "max_leverage": {
            "name": "最大杠杆限制",
            "condition": {"field": "leverage", "operator": ">", "value": 20},
            "action": RuleActionType.REJECT,
            "priority": 200,
            "group": RuleGroupCategory.RISK,
        },
        "high_volatility_block": {
            "name": "高波动封禁",
            "condition": {"field": "volatility_pct", "operator": ">", "value": 0.15},
            "action": RuleActionType.REJECT,
            "priority": 240,
            "group": RuleGroupCategory.MARKET,
        },
        "reduce_only_fast_track": {
            "name": "减仓快速通道",
            "condition": {"field": "reduce_only", "operator": "is_true"},
            "action": RuleActionType.APPROVE,
            "priority": 230,
            "group": RuleGroupCategory.POSITION,
        },
    }

    tp = template_params.get(template_name)
    if not tp:
        return None

    rule_id = overrides.pop("rule_id", f"template_{template_name}")
    tp.update(overrides)
    return Rule(rule_id=rule_id, **tp)


# ═══════════════════════════════════════════════════════════════
# 导出
# ═══════════════════════════════════════════════════════════════

__all__ = [
    "RuleBasedEngine",
    "Rule",
    "RuleOperator",
    "RuleActionType",
    "RuleGroupCategory",
    "ActivationMode",
    "ActivationWindow",
    "RulePerformance",
    "create_rule_from_template",
]
