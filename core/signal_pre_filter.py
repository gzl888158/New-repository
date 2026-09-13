"""
多层信号前置过滤链 - Signal Pre-Filter Chain
============================================

把散落在 intelligent_agent.audit_signal（9 个维度硬编码串行）、signal_processor（L0
RegimeGate + 信号冲突）、risk_gate（L1-L5）的前置过滤逻辑，收敛为「单一口径」的
可插拔、可解释、可配置过滤链。

设计要点：
- 统一结果口径：FilterDecision（pass/reject/reduce/delay）取代 AgentDecision / RiskCheckResult /
  Tuple[bool, str] 的混乱，任何一层命中即短路，不可逆驳回下单。
- 声明式优先级：黑名单 priority=0 最先执行，命中即短路，不再依赖 if-else 顺序隐式表达。
- 可解释 breakdown：每次 evaluate 记录各过滤器（含 pass 与 hit）的命中状态、耗时、理由，
  便于审计「信号为何被拦 / 放行」。
- 纯框架、可测：过滤器只依赖 SignalContext + 显式 state（由调用方注入），不直接访问
  数据库 / 网络 / 智能体内部状态，单测可完全脱离智能体运行。

过滤器清单（priority 即执行顺序）：
   0  blacklist         币种×策略黑名单（最高优先级，不可逆）
   1  strategy_pause    策略自动暂停
   2  confidence        自适应置信度阈值
   3  regime            市场状态兼容性
   4  trend_alignment   趋势方向一致性（软降仓）
   5  consecutive_loss  连续亏损暂停
   5  anti_debounce     统一防抖动/防频繁交易（品种冷却+信号去重+策略冷却+全局冷却+亏损冷却+刷单检测+自适应冷却）
   6  frequency         交易频率
   7  hour_risk         时段风险（仓位缩减 + 置信度收紧）
   8  volatility        高波动降仓
   9  min_profit        最低盈利阈值（防手续费蚕食）
  10  wear_type         磨损型交易检测 (WTI)
  11  multi_timeframe   多时间框架确认 (MTF)
"""

from typing import Dict, Any, Optional, List, TYPE_CHECKING
from dataclasses import dataclass, field
import time

from loguru import logger

if TYPE_CHECKING:
    from core.anti_debounce_engine import AntiDebounceEngine

# 统一动作常量
ACTION_PASS = "pass"
ACTION_REJECT = "reject"
ACTION_REDUCE = "reduce"
ACTION_DELAY = "delay"


def _resolve(value):
    """惰性求值：state 字段可为 callable（重逻辑延迟到真正执行该层时再计算），
    保证黑名单等前置过滤器命中短路时，不触发后续较重的磨损/MTF 查询。"""
    return value() if callable(value) else value


@dataclass
class SignalContext:
    """过滤链输入：一笔待审核信号的原始字段。"""
    symbol: str
    strategy_name: str
    signal_type: str = ""
    direction: str = "long"
    price: float = 0.0
    quantity: float = 0.0
    confidence: float = 0.0
    is_close: bool = False


@dataclass
class FilterDecision:
    """单层过滤器命中结果（action 非 pass）。"""
    action: str
    source: str
    reason: str
    confidence: float = 0.0
    level: str = "symbol"  # strategy / symbol / portfolio / global
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "source": self.source,
            "reason": self.reason,
            "confidence": self.confidence,
            "level": self.level,
            "details": self.details,
        }


@dataclass
class FilterChainResult:
    """过滤链总结果。"""
    passed: bool
    action: str
    blocked_source: Optional[str]
    blocked_reason: str = ""
    decision: Optional[FilterDecision] = None
    breakdown: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def reduced_quantity(self) -> Optional[float]:
        if self.decision and self.decision.action == ACTION_REDUCE:
            return self.decision.details.get("reduced_quantity")
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "action": self.action,
            "blocked_source": self.blocked_source,
            "blocked_reason": self.blocked_reason,
            "decision": self.decision.to_dict() if self.decision else None,
            "breakdown": self.breakdown,
        }


class BaseSignalFilter:
    """过滤器基类。子类重写 name / priority / check。"""

    name: str = "base"
    priority: int = 100
    source: str = "audit_filter"

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, ctx: SignalContext, state: Dict[str, Any]) -> Optional[FilterDecision]:
        """返回 None 表示放行（pass），否则返回命中决策（reject/reduce/delay）。"""
        raise NotImplementedError


# ============================================================================
# 纯规则过滤器
# ============================================================================

class BlacklistFilter(BaseSignalFilter):
    """维度0：币种×策略黑名单（最高优先级）。"""
    name = "blacklist"
    priority = 0
    source = "audit_blacklist"

    def check(self, ctx, state):
        bl = state.get("blacklist") or {}
        if bl.get("hit"):
            return FilterDecision(
                action=ACTION_REJECT,
                source=self.source,
                reason=f"Symbol {ctx.symbol} 在黑名单中: {bl.get('reason', '')}",
                confidence=0.95,
                level="symbol",
                details={"blacklist": bl},
            )
        return None


class StrategyPauseFilter(BaseSignalFilter):
    """维度0.5：策略自动暂停。"""
    name = "strategy_pause"
    priority = 1
    source = "audit_strategy_pause"

    def check(self, ctx, state):
        if ctx.is_close:
            return None
        pause = state.get("strategy_paused") or {}
        if pause.get("hit"):
            return FilterDecision(
                action=ACTION_REJECT,
                source=self.source,
                reason=f"策略 {ctx.strategy_name} 已暂停: {pause.get('reason', '')}",
                confidence=0.9,
                level="strategy",
                details={"pause": pause},
            )
        return None


class ConfidenceFilter(BaseSignalFilter):
    """维度1：自适应置信度阈值。"""
    name = "confidence"
    priority = 2
    source = "audit_confidence"

    def check(self, ctx, state):
        threshold = state.get("adaptive_threshold", 0.0)
        if ctx.confidence < threshold:
            return FilterDecision(
                action=ACTION_REJECT,
                source=self.source,
                reason=f"信号置信度过低: {ctx.confidence:.2f} < {threshold:.2f} (adaptive)",
                confidence=ctx.confidence,
                level="strategy",
                details={"adaptive_threshold": threshold},
            )
        return None


class RegimeCompatibilityFilter(BaseSignalFilter):
    """维度2：市场状态兼容性。"""
    name = "regime"
    priority = 3
    source = "audit_regime"

    def check(self, ctx, state):
        if ctx.is_close:
            return None
        if not state.get("regime_ok", True):
            return FilterDecision(
                action=ACTION_REJECT,
                source=self.source,
                reason=state.get("regime_reason", "市场状态不兼容"),
                confidence=0.7,
                level="symbol",
            )
        return None


class TrendAlignmentFilter(BaseSignalFilter):
    """维度2.5：趋势方向一致性（中等强度逆势软降仓，强趋势已在 RegimeGate 硬拒）。"""
    name = "trend_alignment"
    priority = 4
    source = "audit_trend_alignment"

    def check(self, ctx, state):
        if ctx.is_close or ctx.direction not in ("long", "short"):
            return None
        ta = state.get("trend_alignment") or {}
        if ta.get("opposing") and ta.get("strength", 0.0) > 0.35:
            reduced_qty = ctx.quantity * 0.6
            return FilterDecision(
                action=ACTION_REDUCE,
                source=self.source,
                reason=(
                    f"趋势方向与信号相反 (regime={ta.get('regime_str')}, "
                    f"strength={ta.get('strength', 0):.2f}, direction={ctx.direction})，仓位降至60%: {reduced_qty:.4f}"
                ),
                confidence=0.7,
                level="symbol",
                details={"reduced_quantity": reduced_qty, "regime": ta.get("regime_str"), "strength": ta.get("strength")},
            )
        return None


class ConsecutiveLossFilter(BaseSignalFilter):
    """维度3：连续亏损暂停开仓。"""
    name = "consecutive_loss"
    priority = 5
    source = "audit_performance"

    def check(self, ctx, state):
        if ctx.is_close:
            return None
        losses = state.get("consecutive_losses", 0)
        threshold = state.get("dynamic_threshold", 5)
        if losses >= threshold:
            return FilterDecision(
                action=ACTION_DELAY,
                source=self.source,
                reason=f"策略连续亏损 {losses} 次(阈值={threshold})，暂停开仓",
                confidence=0.8,
                level="strategy",
                details={"consecutive_losses": losses, "dynamic_threshold": threshold},
            )
        return None


class FrequencyFilter(BaseSignalFilter):
    """维度4：交易频率。"""
    name = "frequency"
    priority = 6
    source = "audit_frequency"

    def check(self, ctx, state):
        if ctx.is_close:
            return None
        trades = state.get("trades_last_hour", 0)
        max_trades = state.get("max_trades_per_hour", 10)
        if trades > max_trades:
            return FilterDecision(
                action=ACTION_DELAY,
                source=self.source,
                reason=f"{ctx.symbol} 1小时内交易 {trades} 次，频率过高",
                confidence=0.65,
                level="symbol",
                details={"trades_last_hour": trades, "max_trades_per_hour": max_trades},
            )
        return None


class AntiDebounceFilter(BaseSignalFilter):
    """维度4.5：统一防抖动/防频繁交易（收敛品种冷却+信号去重+策略冷却+全局冷却+亏损冷却+刷单检测+自适应冷却）。"""
    name = "anti_debounce"
    priority = 5
    source = "audit_anti_debounce"

    def __init__(self, engine: Optional["AntiDebounceEngine"] = None, enabled: bool = True):
        super().__init__(enabled=enabled)
        self._engine = engine

    def check(self, ctx, state):
        if ctx.is_close:
            return None

        engine = self._engine or state.get("debounce_engine")
        if engine is None:
            return None  # 引擎未注入，静默放行（不阻塞）

        signal_type = state.get("signal_type", "open")
        pnl_usdt = state.get("pnl_usdt", 0.0)
        volatility = state.get("volatility", 0.0)
        drawdown = state.get("drawdown", 0.0)
        account_tier = state.get("account_tier", "small")

        # 更新自适应状态
        engine.set_market_state(
            volatility=float(volatility),
            drawdown=float(drawdown),
            account_tier=str(account_tier),
        )

        result = engine.check(
            symbol=ctx.symbol,
            strategy_name=ctx.strategy_name,
            direction=ctx.direction,
            signal_type=signal_type,
            is_close=ctx.is_close,
            pnl_usdt=float(pnl_usdt),
        )

        if result.allowed:
            return None

        return FilterDecision(
            action=ACTION_DELAY,
            source=self.source,
            reason=result.blocked_reason,
            confidence=0.75,
            level="symbol",
            details={
                "blocked_layer": result.blocked_layer,
                "remaining_cooldown": result.remaining_cooldown,
                "adaptive_multiplier": result.adaptive_multiplier,
                "breakdown": result.breakdown,
            },
        )


class HourRiskFilter(BaseSignalFilter):
    """维度5：时段风险（高风险降仓40%+置信度收紧，中风险降仓70%）。"""
    name = "hour_risk"
    priority = 7
    source = "audit_hour_risk"

    def check(self, ctx, state):
        if ctx.is_close:
            return None
        hour_risk = state.get("hour_risk_level", "low")
        threshold = state.get("adaptive_threshold", 0.0)

        if hour_risk == "high":
            reduced_qty = ctx.quantity * 0.4
            risk_confidence = max(threshold + 0.1, 0.6)
            if ctx.confidence < risk_confidence:
                return FilterDecision(
                    action=ACTION_REJECT,
                    source=self.source,
                    reason=f"高风险时段，置信度{ctx.confidence:.2f}<{risk_confidence:.2f}",
                    confidence=0.85,
                    level="global",
                )
            return FilterDecision(
                action=ACTION_REDUCE,
                source=self.source,
                reason=f"高风险时段，仓位降至40%: {reduced_qty:.4f}",
                confidence=0.8,
                level="global",
                details={"reduced_quantity": reduced_qty, "risk_level": "high"},
            )
        if hour_risk == "medium":
            reduced_qty = ctx.quantity * 0.7
            return FilterDecision(
                action=ACTION_REDUCE,
                source=self.source,
                reason=f"中风险时段，仓位降至70%: {reduced_qty:.4f}",
                confidence=0.7,
                level="global",
                details={"reduced_quantity": reduced_qty, "risk_level": "medium"},
            )
        return None


class VolatilityFilter(BaseSignalFilter):
    """维度6：高波动市场降仓。"""
    name = "volatility"
    priority = 8
    source = "audit_volatility"

    def check(self, ctx, state):
        if ctx.is_close:
            return None
        if state.get("high_volatility", False):
            reduced_qty = ctx.quantity * 0.5
            return FilterDecision(
                action=ACTION_REDUCE,
                source=self.source,
                reason=f"高波动市场，建议减少仓位至 {reduced_qty:.4f}",
                confidence=0.75,
                level="symbol",
                details={"reduced_quantity": reduced_qty},
            )
        return None


class MinProfitFilter(BaseSignalFilter):
    """维度7：最低盈利阈值（防手续费蚕食）。"""
    name = "min_profit"
    priority = 9
    source = "audit_min_profit"

    def check(self, ctx, state):
        if ctx.is_close or ctx.price <= 0 or ctx.quantity <= 0:
            return None
        if not state.get("min_profit_ok", True):
            return FilterDecision(
                action=ACTION_REJECT,
                source=self.source,
                reason="预期利润不足以覆盖手续费，拒绝交易以保护资金",
                confidence=0.85,
                level="symbol",
            )
        return None


class WearTypeFilter(BaseSignalFilter):
    """维度8：磨损型交易检测 (WTI)。"""
    name = "wear_type"
    priority = 10
    source = "audit_wear_type"

    def check(self, ctx, state):
        if ctx.is_close or ctx.price <= 0 or ctx.quantity <= 0:
            return None
        wear = _resolve(state.get("wear_type")) or {}
        if wear.get("hit"):
            return FilterDecision(
                action=ACTION_REJECT,
                source=self.source,
                reason=f"WTI: 检测到磨损型交易 - {wear.get('reason', '')}",
                confidence=0.80,
                level="symbol",
            )
        return None


class MultiTimeframeFilter(BaseSignalFilter):
    """维度9：多时间框架确认 (MTF)。"""
    name = "multi_timeframe"
    priority = 11
    source = "audit_multi_timeframe"

    def check(self, ctx, state):
        if ctx.is_close or ctx.direction not in ("long", "short"):
            return None
        mtf = _resolve(state.get("mtf")) or {}
        if not mtf.get("ok", True):
            return FilterDecision(
                action=ACTION_REJECT,
                source=self.source,
                reason=f"MTF: 多时间框架未确认 - {mtf.get('reason', '')}",
                confidence=0.75,
                level="symbol",
            )
        return None


# ============================================================================
# 过滤链
# ============================================================================

class SignalPreFilterChain:
    """多层信号前置过滤链：按优先级串行执行，命中即短路，输出可解释报告。"""

    # 默认过滤器（企业级全量），可被 config 覆盖 / 禁用
    DEFAULT_FILTERS: List[BaseSignalFilter] = [
        BlacklistFilter(),
        StrategyPauseFilter(),
        ConfidenceFilter(),
        RegimeCompatibilityFilter(),
        TrendAlignmentFilter(),
        ConsecutiveLossFilter(),
        AntiDebounceFilter(),
        FrequencyFilter(),
        HourRiskFilter(),
        VolatilityFilter(),
        MinProfitFilter(),
        WearTypeFilter(),
        MultiTimeframeFilter(),
    ]

    def __init__(self, filters: Optional[List[BaseSignalFilter]] = None, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = self.config.get("signal_pre_filter", {})
        disabled = set(cfg.get("disabled", []) or [])

        # 每个链实例需持有独立过滤器实例：DEFAULT_FILTERS 是类级共享的实例列表，
        # 直接 list() 浅拷贝会导致 config.disabled / enabled 状态跨链实例泄漏（污染后续链）。
        filters = filters if filters is not None else [type(f)() for f in self.DEFAULT_FILTERS]
        # 应用启用/禁用开关
        for f in filters:
            if f.name in disabled:
                f.enabled = False
        # 按优先级稳定排序
        self._filters = sorted(filters, key=lambda f: f.priority)

        # 统计：各过滤器命中次数（用于可解释与监控）
        self._stats: Dict[str, int] = {}

    @property
    def filters(self) -> List[BaseSignalFilter]:
        return list(self._filters)

    def evaluate(self, ctx: SignalContext, state: Dict[str, Any]) -> FilterChainResult:
        """串行执行过滤链，返回统一结果。"""
        breakdown: List[Dict[str, Any]] = []
        state = state or {}

        for f in self._filters:
            if not f.enabled:
                continue

            start = time.perf_counter()
            decision = None
            try:
                decision = f.check(ctx, state)
            except Exception as e:
                # 过滤器异常不中断链路，按放行处理（保守不误杀）
                logger.warning(f"SignalPreFilterChain: filter {f.name} raised: {e}")
            elapsed_ms = round((time.perf_counter() - start) * 1000, 4)

            if decision is None:
                breakdown.append({
                    "name": f.name, "source": f.source, "priority": f.priority,
                    "action": ACTION_PASS, "reason": "", "elapsed_ms": elapsed_ms, "hit": False,
                })
                continue

            # 命中 → 短路
            self._stats[f.name] = self._stats.get(f.name, 0) + 1
            breakdown.append({
                "name": f.name, "source": f.source, "priority": f.priority,
                "action": decision.action, "reason": decision.reason,
                "elapsed_ms": elapsed_ms, "hit": True,
            })
            return FilterChainResult(
                passed=False,
                action=decision.action,
                blocked_source=decision.source,
                blocked_reason=decision.reason,
                decision=decision,
                breakdown=breakdown,
            )

        # 全部放行
        return FilterChainResult(
            passed=True,
            action=ACTION_PASS,
            blocked_source=None,
            blocked_reason="",
            decision=None,
            breakdown=breakdown,
        )

    def stats(self) -> Dict[str, int]:
        return dict(self._stats)

    def reset_stats(self):
        self._stats.clear()
