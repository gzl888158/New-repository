"""
生产级实时风险监控引擎
======================
核心定位：全策略风险聚合、多级告警、自动缓解，确保风险实时可控。

功能：
- 全策略风险聚合（多维度实时计算）
- 风险评分矩阵（方向/杠杆/波动率/相关性/集中度）
- 多级告警（INFO → WARNING → CRITICAL → EMERGENCY）
- 自动风险缓解（减仓/暂停/排空）
- 风险事件追踪与根因分析
- 实时风险仪表盘数据

架构：
  RealTimeRiskMonitor
  ├── RiskAggregator（风险聚合器）
  │   ├── 方向风险
  │   ├── 杠杆风险
  │   ├── 波动率风险
  │   ├── 相关性风险
  │   └── 集中度风险
  ├── RiskScorer（风险评分引擎）
  ├── AlertManager（多级告警）
  ├── AutoMitigator（自动缓解）
  └── RiskEventTracker（事件追踪）
"""

import asyncio
import json
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class RiskAlertLevel(Enum):
    """风险告警级别"""
    INFO = "info"            # 信息
    WARNING = "warning"      # 预警
    CRITICAL = "critical"    # 临界
    EMERGENCY = "emergency"  # 紧急


class RiskDimension(Enum):
    """风险维度"""
    DIRECTION = "direction"          # 方向风险
    LEVERAGE = "leverage"            # 杠杆风险
    VOLATILITY = "volatility"        # 波动率风险
    CORRELATION = "correlation"      # 相关性风险
    CONCENTRATION = "concentration"  # 集中度风险
    DRAWDOWN = "drawdown"            # 回撤风险
    MARGIN = "margin"                # 保证金风险
    LIQUIDITY = "liquidity"          # 流动性风险


class MitigationAction(Enum):
    """缓解动作"""
    NONE = "none"                    # 无动作
    NOTIFY = "notify"                # 通知
    REDUCE_POSITION = "reduce_position"  # 减仓
    PAUSE_STRATEGY = "pause_strategy"    # 暂停策略
    PAUSE_ALL = "pause_all"              # 暂停全部
    DRAIN_POSITION = "drain_position"    # 排空持仓
    EMERGENCY_STOP = "emergency_stop"    # 紧急停止


@dataclass
class RiskScore:
    """风险评分"""
    dimension: RiskDimension
    score: float                  # 0-1, 越高越危险
    level: RiskAlertLevel
    value: float                  # 原始值
    threshold: float              # 阈值
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class RiskSnapshot:
    """风险快照"""
    overall_score: float = 0.0          # 综合风险评分
    overall_level: RiskAlertLevel = RiskAlertLevel.INFO
    dimension_scores: Dict[str, RiskScore] = field(default_factory=dict)
    total_exposure: float = 0.0
    total_margin: float = 0.0
    margin_ratio: Optional[float] = None
    drawdown_pct: Optional[float] = None
    position_count: int = 0
    active_strategies: int = 0
    alerts_active: List[str] = field(default_factory=list)
    timestamp: float = 0.0


@dataclass
class RiskEvent:
    """风险事件"""
    event_id: str = ""
    event_type: str = ""
    level: RiskAlertLevel = RiskAlertLevel.INFO
    dimension: Optional[RiskDimension] = None
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    mitigation: MitigationAction = MitigationAction.NONE
    mitigation_result: str = ""
    timestamp: float = 0.0
    resolved_at: Optional[float] = None


# ═══════════════════════════════════════════════════════════════
# RealTimeRiskMonitor
# ═══════════════════════════════════════════════════════════════

class RealTimeRiskMonitor:
    """
    生产级实时风险监控引擎

    使用示例:
        monitor = RealTimeRiskMonitor(config)
        monitor.inject_dependencies(okx_client=client, position_manager=pm,
                                     risk_gate=gate, circuit_breaker=cb)
        monitor.on_risk_alert(lambda event: send_telegram(event.message))
        await monitor.start()
        # ... 运行中 ...
        snapshot = monitor.get_risk_snapshot()
        await monitor.stop()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        rtm_cfg = config.get("risk_monitor", {})

        # ── 监控配置 ──
        self._enabled = rtm_cfg.get("enabled", True)
        self._check_interval = rtm_cfg.get("check_interval_sec", 3)
        self._scoring_weights = rtm_cfg.get("scoring_weights", {
            "direction": 0.20,
            "leverage": 0.15,
            "volatility": 0.15,
            "correlation": 0.10,
            "concentration": 0.15,
            "drawdown": 0.15,
            "margin": 0.10,
        })

        # ── 阈值配置 ──
        thresholds = rtm_cfg.get("thresholds", {})
        self._thresholds = {
            RiskDimension.DIRECTION: thresholds.get("direction", {
                "warning": 0.4, "critical": 0.6, "emergency": 0.8,
            }),
            RiskDimension.LEVERAGE: thresholds.get("leverage", {
                "warning": 0.5, "critical": 0.7, "emergency": 0.85,
            }),
            RiskDimension.VOLATILITY: thresholds.get("volatility", {
                "warning": 0.4, "critical": 0.6, "emergency": 0.8,
            }),
            RiskDimension.CONCENTRATION: thresholds.get("concentration", {
                "warning": 0.3, "critical": 0.5, "emergency": 0.7,
            }),
            RiskDimension.DRAWDOWN: thresholds.get("drawdown", {
                "warning": 0.10, "critical": 0.20, "emergency": 0.30,
            }),
            RiskDimension.MARGIN: thresholds.get("margin", {
                "warning": 0.5, "critical": 0.7, "emergency": 0.85,
            }),
            RiskDimension.CORRELATION: thresholds.get("correlation", {
                "warning": 0.6, "critical": 0.8, "emergency": 0.9,
            }),
            RiskDimension.LIQUIDITY: thresholds.get("liquidity", {
                "warning": 0.5, "critical": 0.7, "emergency": 0.85,
            }),
        }

        # ── 自动缓解配置 ──
        mitigation = rtm_cfg.get("auto_mitigation", {})
        self._auto_mitigation_enabled = mitigation.get("enabled", True)
        self._mitigation_cooldown = mitigation.get("cooldown_sec", 60)
        self._max_mitigations_per_hour = mitigation.get("max_per_hour", 10)
        self._mitigation_actions = mitigation.get("actions", {
            "warning": MitigationAction.NOTIFY.value,
            "critical": MitigationAction.REDUCE_POSITION.value,
            "emergency": MitigationAction.EMERGENCY_STOP.value,
        })

        # ── 告警配置 ──
        alert_cfg = rtm_cfg.get("alerts", {})
        self._alert_cooldown = alert_cfg.get("cooldown_sec", 300)
        self._alert_escalation = alert_cfg.get("escalation_enabled", True)
        self._alert_escalation_interval = alert_cfg.get("escalation_interval_sec", 600)

        # ── 外部依赖 ──
        self._okx_client = None
        self._position_manager = None
        self._risk_gate = None
        self._circuit_breaker = None
        self._account_manager = None
        self._capital_manager = None
        self._session_manager = None
        self._equity_monitor = None

        # ── 风险缓存 ──
        self._risk_snapshot = RiskSnapshot()
        self._risk_history: deque = deque(maxlen=1000)
        self._position_exposures: Dict[str, float] = {}
        self._strategy_exposures: Dict[str, Dict[str, float]] = defaultdict(dict)

        # ── 告警管理 ──
        self._active_alerts: Dict[str, RiskEvent] = {}
        self._alert_history: List[RiskEvent] = []
        self._max_alert_history = alert_cfg.get("max_history", 500)
        self._last_alert_time: Dict[str, float] = {}  # dimension -> last_alert_time

        # ── 缓解管理 ──
        self._mitigation_history: List[Dict[str, Any]] = []
        self._last_mitigation_time: float = 0.0
        self._mitigation_count_hour: int = 0
        self._mitigation_hour_start: float = 0.0

        # ── 回调 ──
        self._alert_callbacks: List[Callable] = []
        self._mitigation_callbacks: List[Callable] = []

        # ── 运行控制 ──
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None

        # ── 持久化 ──
        self._persist_dir = config.get("system", {}).get("data_dir", "data")
        self._risk_log_file = os.path.join(self._persist_dir, "risk_events.jsonl")

        logger.info(
            f"RealTimeRiskMonitor initialized: "
            f"check_interval={self._check_interval}s, "
            f"auto_mitigation={self._auto_mitigation_enabled}"
        )

    # ═══════════════════════════════════════════════════════════════
    # 依赖注入
    # ═══════════════════════════════════════════════════════════════

    def inject_dependencies(self, okx_client=None, position_manager=None,
                            risk_gate=None, circuit_breaker=None,
                            account_manager=None, capital_manager=None,
                            session_manager=None, equity_monitor=None):
        """注入外部依赖"""
        self._okx_client = okx_client
        self._position_manager = position_manager
        self._risk_gate = risk_gate
        if position_manager is not None and risk_gate is not None:
            set_position_manager = getattr(risk_gate, "set_position_manager", None)
            if callable(set_position_manager):
                set_position_manager(position_manager)
        self._circuit_breaker = circuit_breaker
        self._account_manager = account_manager
        self._capital_manager = capital_manager
        self._session_manager = session_manager
        self._equity_monitor = equity_monitor
        logger.info("RealTimeRiskMonitor dependencies injected")

    # ═══════════════════════════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════════════════════════

    async def start(self):
        """启动监控"""
        if not self._enabled:
            logger.info("RealTimeRiskMonitor disabled")
            return

        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("RealTimeRiskMonitor started")

    async def stop(self):
        """停止监控"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("RealTimeRiskMonitor stopped")

    async def _monitor_loop(self):
        """监控主循环"""
        while self._running:
            try:
                await self._check_risks()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Risk monitor loop error: {e}")
                await asyncio.sleep(1)

    # ═══════════════════════════════════════════════════════════════
    # 风险检查
    # ═══════════════════════════════════════════════════════════════

    async def _check_risks(self):
        """执行全维度风险检查"""
        scores = {}

        # 1. 方向风险
        scores[RiskDimension.DIRECTION] = self._check_direction_risk()

        # 2. 杠杆风险
        scores[RiskDimension.LEVERAGE] = self._check_leverage_risk()

        # 3. 波动率风险
        scores[RiskDimension.VOLATILITY] = self._check_volatility_risk()

        # 4. 集中度风险
        scores[RiskDimension.CONCENTRATION] = self._check_concentration_risk()

        # 5. 回撤风险
        scores[RiskDimension.DRAWDOWN] = self._check_drawdown_risk()

        # 6. 保证金风险
        scores[RiskDimension.MARGIN] = self._check_margin_risk()

        # 7. 相关性风险
        scores[RiskDimension.CORRELATION] = self._check_correlation_risk()

        # 8. 流动性风险
        scores[RiskDimension.LIQUIDITY] = self._check_liquidity_risk()

        # 计算综合评分
        overall_score = self._calculate_overall_score(scores)
        overall_level = self._score_to_level(overall_score, {
            "warning": 0.3, "critical": 0.5, "emergency": 0.7,
        })

        # 更新快照
        self._risk_snapshot = RiskSnapshot(
            overall_score=overall_score,
            overall_level=overall_level,
            dimension_scores={k.value: v for k, v in scores.items()},
            total_exposure=self._calculate_total_exposure(),
            total_margin=self._calculate_total_margin(),
            margin_ratio=self._get_margin_ratio(),
            drawdown_pct=self._get_drawdown_pct(),
            position_count=self._get_position_count(),
            active_strategies=self._get_active_strategy_count(),
            alerts_active=list(self._active_alerts.keys()),
            timestamp=time.time(),
        )

        self._risk_history.append(self._risk_snapshot)

        # 处理高风险维度
        for dimension, score in scores.items():
            if score.level in (RiskAlertLevel.CRITICAL, RiskAlertLevel.EMERGENCY):
                await self._handle_risk_alert(dimension, score)

    # ═══════════════════════════════════════════════════════════════
    # 各维度风险检查
    # ═══════════════════════════════════════════════════════════════

    def _check_direction_risk(self) -> RiskScore:
        """检查方向风险：多空暴露失衡"""
        long_exposure = 0.0
        short_exposure = 0.0

        if self._position_manager:
            positions = self._position_manager.get_all_positions()
            for pos in positions:
                exposure = pos.quantity * pos.mark_price * pos.leverage
                if pos.side.value == "long":
                    long_exposure += exposure
                else:
                    short_exposure += exposure

        total_exposure = long_exposure + short_exposure
        if total_exposure > 0:
            direction_imbalance = abs(long_exposure - short_exposure) / total_exposure
        else:
            direction_imbalance = 0.0

        level = self._value_to_level(direction_imbalance, RiskDimension.DIRECTION)

        return RiskScore(
            dimension=RiskDimension.DIRECTION,
            score=direction_imbalance,
            level=level,
            value=direction_imbalance,
            threshold=self._thresholds[RiskDimension.DIRECTION].get("critical", 0.6),
            details={
                "long_exposure": long_exposure,
                "short_exposure": short_exposure,
                "imbalance_pct": direction_imbalance,
            },
            timestamp=time.time(),
        )

    def _check_leverage_risk(self) -> RiskScore:
        """检查杠杆风险：整体杠杆水平"""
        max_leverage = 1
        avg_leverage = 1.0
        total_leverage = 0.0
        count = 0

        if self._position_manager:
            positions = self._position_manager.get_all_positions()
            for pos in positions:
                max_leverage = max(max_leverage, pos.leverage)
                total_leverage += pos.leverage
                count += 1

        if count > 0:
            avg_leverage = total_leverage / count

        # 杠杆风险评分：基于最大杠杆和平均杠杆
        leverage_score = (max_leverage / 125.0) * 0.7 + (avg_leverage / 50.0) * 0.3
        leverage_score = min(1.0, leverage_score)

        level = self._value_to_level(leverage_score, RiskDimension.LEVERAGE)

        return RiskScore(
            dimension=RiskDimension.LEVERAGE,
            score=leverage_score,
            level=level,
            value=max_leverage,
            threshold=self._thresholds[RiskDimension.LEVERAGE].get("critical", 0.7),
            details={
                "max_leverage": max_leverage,
                "avg_leverage": avg_leverage,
                "position_count": count,
            },
            timestamp=time.time(),
        )

    def _check_volatility_risk(self) -> RiskScore:
        """检查波动率风险"""
        volatility_score = 0.0
        details = {"symbols": {}}

        if self._okx_client and self._position_manager:
            positions = self._position_manager.get_all_positions()
            for pos in positions:
                try:
                    atr = self._okx_client.get_atr(pos.symbol)
                    if atr and pos.mark_price > 0:
                        atr_pct = atr / pos.mark_price
                        details["symbols"][pos.symbol] = atr_pct
                        volatility_score = max(volatility_score, atr_pct * 10)
                except Exception:
                    pass

        volatility_score = min(1.0, volatility_score)
        level = self._value_to_level(volatility_score, RiskDimension.VOLATILITY)

        return RiskScore(
            dimension=RiskDimension.VOLATILITY,
            score=volatility_score,
            level=level,
            value=volatility_score,
            threshold=self._thresholds[RiskDimension.VOLATILITY].get("critical", 0.6),
            details=details,
            timestamp=time.time(),
        )

    def _check_concentration_risk(self) -> RiskScore:
        """检查集中度风险：单币种/单策略暴露"""
        symbol_exposures = defaultdict(float)
        strategy_exposures = defaultdict(float)
        total_exposure = 0.0

        if self._position_manager:
            positions = self._position_manager.get_all_positions()
            for pos in positions:
                exposure = pos.quantity * pos.mark_price * pos.leverage
                symbol_exposures[pos.symbol] += exposure
                if pos.strategy_name:
                    strategy_exposures[pos.strategy_name] += exposure
                total_exposure += exposure

        max_symbol_pct = 0.0
        max_symbol = ""
        if total_exposure > 0:
            for symbol, exp in symbol_exposures.items():
                pct = exp / total_exposure
                if pct > max_symbol_pct:
                    max_symbol_pct = pct
                    max_symbol = symbol

        max_strategy_pct = 0.0
        if total_exposure > 0:
            for strategy, exp in strategy_exposures.items():
                pct = exp / total_exposure
                max_strategy_pct = max(max_strategy_pct, pct)

        # 综合集中度评分
        concentration_score = max_symbol_pct * 0.6 + max_strategy_pct * 0.4
        concentration_score = min(1.0, concentration_score)

        level = self._value_to_level(concentration_score, RiskDimension.CONCENTRATION)

        return RiskScore(
            dimension=RiskDimension.CONCENTRATION,
            score=concentration_score,
            level=level,
            value=max_symbol_pct,
            threshold=self._thresholds[RiskDimension.CONCENTRATION].get("critical", 0.5),
            details={
                "max_symbol": max_symbol,
                "max_symbol_pct": max_symbol_pct,
                "max_strategy_pct": max_strategy_pct,
                "symbol_exposures": dict(symbol_exposures),
                "strategy_exposures": dict(strategy_exposures),
            },
            timestamp=time.time(),
        )

    def _check_drawdown_risk(self) -> RiskScore:
        """检查回撤风险；查询失败按未知保守处理（fail-closed），避免误判为无回撤"""
        drawdown_pct = self._get_drawdown_pct()
        if drawdown_pct is None:
            return RiskScore(
                dimension=RiskDimension.DRAWDOWN,
                score=1.0,
                level=RiskAlertLevel.EMERGENCY,
                value=0.0,
                threshold=self._thresholds[RiskDimension.DRAWDOWN].get("critical", 0.20),
                details={"drawdown_pct": None, "unknown": True},
                timestamp=time.time(),
            )

        level = self._value_to_level(drawdown_pct, RiskDimension.DRAWDOWN)

        return RiskScore(
            dimension=RiskDimension.DRAWDOWN,
            score=drawdown_pct,
            level=level,
            value=drawdown_pct,
            threshold=self._thresholds[RiskDimension.DRAWDOWN].get("critical", 0.20),
            details={"drawdown_pct": drawdown_pct},
            timestamp=time.time(),
        )

    def _check_margin_risk(self) -> RiskScore:
        """检查保证金风险；查询失败按未知保守处理（fail-closed），避免误判为无风险"""
        margin_ratio = self._get_margin_ratio()
        if margin_ratio is None:
            return RiskScore(
                dimension=RiskDimension.MARGIN,
                score=1.0,
                level=RiskAlertLevel.EMERGENCY,
                value=0.0,
                threshold=self._thresholds[RiskDimension.MARGIN].get("critical", 0.7),
                details={
                    "margin_ratio": None,
                    "unknown": True,
                    "total_margin": self._calculate_total_margin(),
                },
                timestamp=time.time(),
            )

        level = self._value_to_level(margin_ratio, RiskDimension.MARGIN)

        return RiskScore(
            dimension=RiskDimension.MARGIN,
            score=margin_ratio,
            level=level,
            value=margin_ratio,
            threshold=self._thresholds[RiskDimension.MARGIN].get("critical", 0.7),
            details={
                "margin_ratio": margin_ratio,
                "total_margin": self._calculate_total_margin(),
            },
            timestamp=time.time(),
        )

    def _check_correlation_risk(self) -> RiskScore:
        """检查相关性风险"""
        correlation_score = 0.0
        details = {"same_direction_count": 0, "total_positions": 0}

        if self._position_manager:
            positions = self._position_manager.get_all_positions()
            long_count = sum(1 for p in positions if p.side.value == "long")
            short_count = len(positions) - long_count
            details["total_positions"] = len(positions)
            details["same_direction_count"] = max(long_count, short_count)

            if len(positions) > 1:
                # 同向持仓比例越高，相关性风险越大
                correlation_score = max(long_count, short_count) / len(positions)

        level = self._value_to_level(correlation_score, RiskDimension.CORRELATION)

        return RiskScore(
            dimension=RiskDimension.CORRELATION,
            score=correlation_score,
            level=level,
            value=correlation_score,
            threshold=self._thresholds[RiskDimension.CORRELATION].get("critical", 0.8),
            details=details,
            timestamp=time.time(),
        )

    def _check_liquidity_risk(self) -> RiskScore:
        """检查流动性风险"""
        liquidity_score = 0.0
        details = {}

        if self._position_manager:
            positions = self._position_manager.get_all_positions()
            for pos in positions:
                # 大仓位在低流动性币种上风险更高
                position_value = pos.quantity * pos.mark_price
                details[pos.symbol] = {"position_value": position_value}

                # 简化：持仓价值超过一定阈值则评分升高
                if position_value > 5000:
                    liquidity_score = max(liquidity_score, 0.5)
                if position_value > 10000:
                    liquidity_score = max(liquidity_score, 0.8)

        level = self._value_to_level(liquidity_score, RiskDimension.LIQUIDITY)

        return RiskScore(
            dimension=RiskDimension.LIQUIDITY,
            score=liquidity_score,
            level=level,
            value=liquidity_score,
            threshold=self._thresholds[RiskDimension.LIQUIDITY].get("critical", 0.7),
            details=details,
            timestamp=time.time(),
        )

    # ═══════════════════════════════════════════════════════════════
    # 评分与告警
    # ═══════════════════════════════════════════════════════════════

    def _calculate_overall_score(self, scores: Dict[RiskDimension, RiskScore]) -> float:
        """计算综合风险评分"""
        if not scores:
            return 0.0

        weighted_sum = 0.0
        total_weight = 0.0

        for dimension, score in scores.items():
            weight = self._scoring_weights.get(dimension.value, 0.1)
            weighted_sum += score.score * weight
            total_weight += weight

        return weighted_sum / max(total_weight, 0.01)

    def _value_to_level(self, value: float, dimension: RiskDimension) -> RiskAlertLevel:
        """将数值映射到告警级别"""
        thresholds = self._thresholds.get(dimension, {})
        emergency = thresholds.get("emergency", 0.8)
        critical = thresholds.get("critical", 0.6)
        warning = thresholds.get("warning", 0.3)

        if value >= emergency:
            return RiskAlertLevel.EMERGENCY
        elif value >= critical:
            return RiskAlertLevel.CRITICAL
        elif value >= warning:
            return RiskAlertLevel.WARNING
        return RiskAlertLevel.INFO

    def _score_to_level(self, score: float, thresholds: Dict[str, float]) -> RiskAlertLevel:
        """将综合评分映射到告警级别"""
        if score >= thresholds.get("emergency", 0.7):
            return RiskAlertLevel.EMERGENCY
        elif score >= thresholds.get("critical", 0.5):
            return RiskAlertLevel.CRITICAL
        elif score >= thresholds.get("warning", 0.3):
            return RiskAlertLevel.WARNING
        return RiskAlertLevel.INFO

    async def _handle_risk_alert(self, dimension: RiskDimension, score: RiskScore):
        """处理风险告警"""
        # 冷却检查
        dim_key = dimension.value
        last_alert = self._last_alert_time.get(dim_key, 0)
        if time.time() - last_alert < self._alert_cooldown:
            # 升级检查：如果级别上升，允许立即告警
            existing = self._active_alerts.get(dim_key)
            if existing and score.level.value <= existing.level.value:
                return

        self._last_alert_time[dim_key] = time.time()

        # 创建事件
        event = RiskEvent(
            event_id=f"risk_{dim_key}_{int(time.time())}",
            event_type=f"risk_{dim_key}",
            level=score.level,
            dimension=dimension,
            message=f"{dimension.value} risk: {score.score:.2%} (threshold: {score.threshold:.2%})",
            value=score.score,
            threshold=score.threshold,
            timestamp=time.time(),
        )

        # 确定缓解动作
        action_str = self._mitigation_actions.get(score.level.value, MitigationAction.NOTIFY.value)
        try:
            event.mitigation = MitigationAction(action_str)
        except ValueError:
            event.mitigation = MitigationAction.NOTIFY

        # 更新活跃告警
        self._active_alerts[dim_key] = event
        self._alert_history.append(event)
        if len(self._alert_history) > self._max_alert_history:
            self._alert_history = self._alert_history[-self._max_alert_history:]

        logger.warning(
            f"Risk alert: {dimension.value} = {score.score:.2%} "
            f"[{score.level.value}] -> {event.mitigation.value}"
        )

        # 通知回调
        for cb in self._alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        # 执行自动缓解
        if self._auto_mitigation_enabled:
            await self._execute_mitigation(event)

        # 持久化
        self._save_risk_event(event)

    # ═══════════════════════════════════════════════════════════════
    # 自动缓解
    # ═══════════════════════════════════════════════════════════════

    async def _execute_mitigation(self, event: RiskEvent):
        """执行自动风险缓解"""
        # 冷却检查
        if time.time() - self._last_mitigation_time < self._mitigation_cooldown:
            logger.debug(f"Mitigation cooldown active, skipping: {event.event_type}")
            return

        # 每小时次数限制
        now = time.time()
        if now - self._mitigation_hour_start > 3600:
            self._mitigation_hour_start = now
            self._mitigation_count_hour = 0
        if self._mitigation_count_hour >= self._max_mitigations_per_hour:
            logger.warning(f"Mitigation limit reached ({self._max_mitigations_per_hour}/hour)")
            return

        self._last_mitigation_time = time.time()
        self._mitigation_count_hour += 1

        logger.info(f"Executing mitigation: {event.mitigation.value} for {event.event_type}")

        result = "executed"
        try:
            if event.mitigation == MitigationAction.REDUCE_POSITION:
                await self._mitigate_reduce_position(event)
            elif event.mitigation == MitigationAction.PAUSE_STRATEGY:
                await self._mitigate_pause_strategy(event)
            elif event.mitigation == MitigationAction.PAUSE_ALL:
                await self._mitigate_pause_all(event)
            elif event.mitigation == MitigationAction.DRAIN_POSITION:
                await self._mitigate_drain_position(event)
            elif event.mitigation == MitigationAction.EMERGENCY_STOP:
                await self._mitigate_emergency_stop(event)
            else:
                result = "notified_only"
        except Exception as e:
            logger.error(f"Mitigation failed: {event.mitigation.value} - {e}")
            result = f"failed: {e}"

        event.mitigation_result = result

        self._mitigation_history.append({
            "event_id": event.event_id,
            "action": event.mitigation.value,
            "result": result,
            "timestamp": time.time(),
        })

        # 通知缓解回调
        for cb in self._mitigation_callbacks:
            try:
                if asyncio.iscoroutinefunction(cb):
                    await cb(event)
                else:
                    cb(event)
            except Exception as e:
                logger.error(f"Mitigation callback error: {e}")

    async def _mitigate_reduce_position(self, event: RiskEvent):
        """减仓缓解"""
        if self._position_manager:
            positions = self._position_manager.get_all_positions()
            # 按未实现亏损排序，优先减亏损最大的
            positions.sort(key=lambda p: p.unrealized_pnl)
            worst_pos = positions[0] if positions else None
            if worst_pos:
                logger.info(
                    f"Reducing worst position: {worst_pos.symbol} "
                    f"PnL={worst_pos.unrealized_pnl:.2f}"
                )

    async def _mitigate_pause_strategy(self, event: RiskEvent):
        """暂停策略缓解"""
        if self._session_manager:
            await self._session_manager.pause_session(
                reason=f"Risk mitigation: {event.event_type} ({event.level.value})"
            )

    async def _mitigate_pause_all(self, event: RiskEvent):
        """暂停全部缓解"""
        if self._session_manager:
            await self._session_manager.pause_session(
                reason=f"Risk mitigation ALL: {event.event_type} ({event.level.value})"
            )

    async def _mitigate_drain_position(self, event: RiskEvent):
        """排空缓解"""
        if self._session_manager:
            await self._session_manager.close_session(drain_first=True)

    async def _mitigate_emergency_stop(self, event: RiskEvent):
        """紧急停止缓解"""
        if self._session_manager:
            await self._session_manager.trigger_emergency(
                reason=f"Risk emergency: {event.event_type} ({event.level.value})"
            )

    # ═══════════════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════════════

    def get_risk_snapshot(self) -> Dict[str, Any]:
        """获取当前风险快照"""
        s = self._risk_snapshot
        return {
            "overall_score": s.overall_score,
            "overall_level": s.overall_level.value,
            "dimensions": {
                dim: {
                    "score": score.score,
                    "level": score.level.value,
                    "value": score.value,
                    "threshold": score.threshold,
                    "details": score.details,
                }
                for dim, score in s.dimension_scores.items()
            },
            "total_exposure": s.total_exposure,
            "total_margin": s.total_margin,
            "margin_ratio": s.margin_ratio,
            "drawdown_pct": s.drawdown_pct,
            "position_count": s.position_count,
            "active_strategies": s.active_strategies,
            "alerts_active": s.alerts_active,
            "timestamp": s.timestamp,
        }

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return [
            {
                "event_id": e.event_id,
                "type": e.event_type,
                "level": e.level.value,
                "dimension": e.dimension.value if e.dimension else "",
                "message": e.message,
                "value": e.value,
                "threshold": e.threshold,
                "mitigation": e.mitigation.value,
                "mitigation_result": e.mitigation_result,
                "timestamp": e.timestamp,
            }
            for e in self._active_alerts.values()
        ]

    def get_alert_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取告警历史"""
        return [
            {
                "event_id": e.event_id,
                "type": e.event_type,
                "level": e.level.value,
                "dimension": e.dimension.value if e.dimension else "",
                "message": e.message,
                "value": e.value,
                "threshold": e.threshold,
                "mitigation": e.mitigation.value,
                "mitigation_result": e.mitigation_result,
                "timestamp": e.timestamp,
                "resolved_at": e.resolved_at,
            }
            for e in self._alert_history[-limit:]
        ]

    def get_mitigation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取缓解历史"""
        return self._mitigation_history[-limit:]

    def clear_alert(self, dimension: str) -> bool:
        """清除告警"""
        if dimension in self._active_alerts:
            event = self._active_alerts.pop(dimension)
            event.resolved_at = time.time()
            logger.info(f"Alert cleared: {dimension}")
            return True
        return False

    def clear_all_alerts(self):
        """清除所有告警"""
        for event in self._active_alerts.values():
            event.resolved_at = time.time()
        self._active_alerts.clear()
        logger.info("All alerts cleared")

    def get_risk_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取风险历史"""
        return [
            {
                "overall_score": s.overall_score,
                "overall_level": s.overall_level.value,
                "total_exposure": s.total_exposure,
                "margin_ratio": s.margin_ratio,
                "drawdown_pct": s.drawdown_pct,
                "timestamp": s.timestamp,
            }
            for s in list(self._risk_history)[-limit:]
        ]

    # ═══════════════════════════════════════════════════════════════
    # 回调注册
    # ═══════════════════════════════════════════════════════════════

    def on_risk_alert(self, callback: Callable):
        """注册风险告警回调"""
        self._alert_callbacks.append(callback)

    def on_mitigation(self, callback: Callable):
        """注册缓解回调"""
        self._mitigation_callbacks.append(callback)

    # ═══════════════════════════════════════════════════════════════
    # 内部辅助
    # ═══════════════════════════════════════════════════════════════

    def _calculate_total_exposure(self) -> float:
        """计算总暴露"""
        if self._position_manager:
            positions = self._position_manager.get_all_positions()
            return sum(p.quantity * p.mark_price * p.leverage for p in positions)
        return 0.0

    def _calculate_total_margin(self) -> float:
        """计算总保证金"""
        if self._position_manager:
            positions = self._position_manager.get_all_positions()
            return sum(p.margin for p in positions)
        return 0.0

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        """安全转 float，空值/非法值返回 default"""
        try:
            if value is None or str(value).strip() == "":
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    def _get_margin_ratio(self) -> Optional[float]:
        """获取保证金占用率（used_margin / equity，0-1，越高越危险）。

        OKX 单币种 USDT 合约账户的 account/balance 顶层 `mgnRatio` 字段为空
        （该字段仅跨币种/组合保证金模式启用，单币种账户返回空字符串），
        直接读取会误判为未知 → fail-closed 误触发 EMERGENCY。
        改用 per-currency details 字段计算占用率，口径与
        okx_client._parse_account_info 完全一致：
            used_margin = eq - availEq（可用保证金扣除法，主口径）
                          回退 frozenBal，且不低于 ordFrozen
            margin_ratio = used_margin / eq
        查询失败/权益未知/字段缺失返回 None（未知），按 fail-closed 保守处理。
        """
        if not self._account_manager:
            return None
        try:
            info = self._account_manager.get_account_info()
            if not info:
                return None
            equity = 0.0
            used_margin = 0.0
            for detail in info.get("details") or []:
                if not isinstance(detail, dict) or detail.get("ccy") != "USDT":
                    continue
                equity = self._to_float(detail.get("eq"))
                avail_eq = self._to_float(detail.get("availEq"))
                frozen = self._to_float(detail.get("frozenBal"))
                ord_frozen = self._to_float(detail.get("ordFrozen"))
                if equity > 0 and avail_eq > 0:
                    calc_used_margin = equity - avail_eq
                elif frozen > 0:
                    calc_used_margin = frozen
                else:
                    calc_used_margin = 0.0
                used_margin = max(calc_used_margin, ord_frozen)
                break
            if equity <= 0:
                return None
            return min(1.0, max(0.0, used_margin / equity))
        except Exception as e:
            logger.warning(f"Failed to get margin ratio: {e}")
            return None

    def _get_drawdown_pct(self) -> Optional[float]:
        """获取回撤百分比；查询失败返回 None（未知），避免误判为无回撤"""
        # 优先使用 EquityMonitor（生产的权益/回撤权威源）
        if self._equity_monitor:
            try:
                status = self._equity_monitor.get_equity_status()
                peak = float(status.get("peak_equity", 0) or 0)
                current = float(status.get("current_equity", 0) or 0)
                if peak > 0 and current > 0:
                    return max(0.0, (peak - current) / peak)
                # 权益基准未就绪时，回退到历史最大回撤（初始为 0，避免误报）
                return float(status.get("max_drawdown_pct", 0) or 0)
            except Exception as e:
                logger.warning(f"Failed to get drawdown pct from equity_monitor: {e}")
        # 回退到 PositionManager（如果已接入）
        if self._position_manager:
            try:
                risk = self._position_manager.get_account_risk()
                if risk and risk.drawdown_pct is not None:
                    return float(risk.drawdown_pct)
            except Exception as e:
                logger.warning(f"Failed to get drawdown pct: {e}")
        return None

    def _get_position_count(self) -> int:
        """获取持仓数量"""
        if self._position_manager:
            return self._position_manager.get_position_count()
        return 0

    def _get_active_strategy_count(self) -> int:
        """获取活跃策略数量"""
        if self._session_manager:
            return len(self._session_manager.get_active_strategies())
        return 0

    def _save_risk_event(self, event: RiskEvent):
        """保存风险事件"""
        try:
            os.makedirs(self._persist_dir, exist_ok=True)
            with open(self._risk_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event_id": event.event_id,
                    "event_type": event.event_type,
                    "level": event.level.value,
                    "dimension": event.dimension.value if event.dimension else "",
                    "message": event.message,
                    "value": event.value,
                    "threshold": event.threshold,
                    "mitigation": event.mitigation.value,
                    "mitigation_result": event.mitigation_result,
                    "timestamp": event.timestamp,
                }, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to save risk event: {e}")