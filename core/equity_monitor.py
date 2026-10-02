"""企业级资金变动自适应检测模块。

= 功能概览 =
1. 权益曲线追踪：SMA/EMA 平滑、峰值记录、回撤计算
2. 资金变动事件检测：充值/提现、增长/衰退趋势、恢复检测、里程碑跨越
3. 自适应响应：增长模式、衰退模式、恢复模式、紧急模式
4. 风险参数动态输出：仓位乘数、信号质量阈值、策略权重

= 设计原则 =
- 所有检测基于统计显著性，避免单次波动误触发
- 小账户（<100 USDT）与大账户使用不同灵敏度
- 充值/提现检测需排除 PnL 干扰
- 紧急模式优先级最高，覆盖所有其他模式
"""

import os
import json
import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from loguru import logger


def _safe_float(value: Any, default: float = 0.0) -> float:
    """安全数值转换：None/非法字符串/NaN/Inf 统一回退到 default。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):  # NaN/Inf
        return default
    return f


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

class EquityMode(Enum):
    """资金变动模式"""
    NORMAL = "normal"           # 正常
    GROWTH = "growth"           # 增长趋势
    DECLINE = "decline"         # 衰退趋势
    RECOVERY = "recovery"       # 从回撤中恢复
    EMERGENCY = "emergency"     # 紧急（大幅下跌）
    MILESTONE = "milestone"     # 里程碑跨越


class EquityEventType(Enum):
    """资金变动事件类型"""
    DEPOSIT = "deposit"                 # 充值
    WITHDRAWAL = "withdrawal"           # 提现
    GROWTH_TREND_START = "growth_start"
    GROWTH_TREND_END = "growth_end"
    DECLINE_TREND_START = "decline_start"
    DECLINE_TREND_END = "decline_end"
    RECOVERY_START = "recovery_start"
    RECOVERY_COMPLETE = "recovery_complete"
    EMERGENCY_DROP = "emergency_drop"   # 紧急下跌
    EMERGENCY_RECOVERED = "emergency_recovered"
    MILESTONE_CROSS = "milestone_cross" # 里程碑跨越
    PEAK_EQUITY = "peak_equity"         # 创新高
    NEW_DRAWDOWN = "new_drawdown"       # 新回撤低点


@dataclass
class EquitySnapshot:
    """权益快照"""
    timestamp: datetime
    total_equity: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float
    # 衍生指标
    sma_short: float = 0.0    # 短期 SMA（约 30 分钟）
    sma_long: float = 0.0     # 长期 SMA（约 2 小时）
    ema: float = 0.0          # EMA
    drawdown_pct: float = 0.0 # 从峰值回撤百分比
    mode: str = "normal"


@dataclass
class EquityEvent:
    """资金变动事件"""
    event_type: EquityEventType
    timestamp: datetime
    equity_before: float
    equity_after: float
    change_pct: float
    details: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# 核心模块
# ═══════════════════════════════════════════════════════════════

class EquityMonitor:
    """企业级资金变动自适应检测器。

    用法：
        monitor = EquityMonitor(config)
        await monitor.start()
        # 每 30 秒喂入一次权益数据
        monitor.feed(equity=56.67, available=42.0, margin=14.0, upl=0.02)
        # 获取自适应参数
        params = monitor.get_adaptive_params()
    """

    # ── 里程碑阈值（USDT）──
    MILESTONES = [50, 100, 200, 500, 1000, 2000, 5000, 10000]

    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        # 从 config.yaml 的 equity_monitor 节读取参数，回落默认值（未配置时行为不变）
        self._eq_cfg: Dict[str, Any] = self._config.get("equity_monitor", {}) or {}

        # ── 权益历史 ──
        self._history: Deque[EquitySnapshot] = deque(maxlen=500)
        self._events: Deque[EquityEvent] = deque(maxlen=200)
        self._peak_equity: float = 0.0
        self._peak_equity_time: Optional[datetime] = None
        self._trough_equity: float = float("inf")
        self._max_drawdown_pct: float = 0.0

        # ── 趋势追踪 ──
        self._sma_short: float = 0.0
        self._sma_long: float = 0.0
        self._ema: float = 0.0
        self._ema_alpha: float = self._eq_cfg.get("ema_alpha", 0.05)  # EMA 平滑系数

        # ── 模式状态 ──
        self._current_mode: EquityMode = EquityMode.NORMAL
        self._mode_start_time: datetime = datetime.now()
        self._consecutive_up: int = 0
        self._consecutive_down: int = 0
        self._consecutive_up_threshold: int = self._eq_cfg.get("trend_confirm_bars", 5)
        self._consecutive_down_threshold: int = self._eq_cfg.get("trend_confirm_bars", 5)

        # ── 紧急模式 ──
        # 硬下限 0.01 防止阈值被配成 0 导致后续 severity/mode_confidence 除零
        self._emergency_drop_threshold: float = max(
            _safe_float(self._eq_cfg.get("emergency_drop_pct"), 0.10), 0.01
        )  # 10%
        self._emergency_recovery_threshold: float = self._eq_cfg.get("emergency_recovery_pct", 0.05)  # 从低点恢复 5%
        self._emergency_low_watermark: float = 0.0

        # ── 分级回撤护栏（在 EMERGENCY 冻结前渐进降风险，避免回撤一次性累积到硬冻结）──
        tg_cfg = self._eq_cfg.get("tiered_drawdown_guard", {}) or {}
        self._tiered_guard_enabled = bool(tg_cfg.get("enabled", False))
        self._tiered_levels: List[Tuple[float, float, float, int]] = []
        for lv in tg_cfg.get("levels", []) or []:
            if not isinstance(lv, dict):
                continue
            self._tiered_levels.append((
                _safe_float(lv.get("drawdown_pct"), 0.0),
                _safe_float(lv.get("position_multiplier"), 1.0),
                _safe_float(lv.get("risk_budget_ratio"), 1.0),
                int(_safe_float(lv.get("max_positions"), 10)),
            ))
        # 升序排列，保证按回撤递增匹配最高档位
        self._tiered_levels.sort(key=lambda x: x[0])

        # ── 账户级利润留存（账户盈利达阈值后自动降仓位锁利，防止账面利润整体回吐）──
        pr_cfg = self._eq_cfg.get("profit_reserve", {}) or {}
        self._profit_reserve_enabled = bool(pr_cfg.get("enabled", False))
        self._profit_reserve_activation_pct = _safe_float(pr_cfg.get("activation_pct"), 0.02)
        self._profit_reserve_max_pct = _safe_float(pr_cfg.get("max_pct"), 0.10)
        self._profit_reserve_max_reduction = _safe_float(pr_cfg.get("max_position_multiplier_reduction"), 0.30)
        # 留存基准：优先用初始资本（trading.total_capital），否则运行时以首次权益初始化
        self._profit_reserve_baseline = _safe_float(
            self._config.get("trading", {}).get("total_capital"), 0.0
        )

        # ── 回撤速度预警（drawdown_velocity）：短窗口内急跌 → 快速降仓，
        #     在 EMERGENCY 硬冻结前抢先收敛敞口，避免回撤一次性累积到硬冻结 ──
        dv_cfg = self._eq_cfg.get("drawdown_velocity", {}) or {}
        self._dv_enabled = bool(dv_cfg.get("enabled", False))
        self._dv_window = max(3, int(_safe_float(dv_cfg.get("window"), 10)))
        self._dv_drop_pct = _safe_float(dv_cfg.get("drop_pct_per_window"), 0.03)
        self._dv_position_multiplier = _safe_float(dv_cfg.get("position_multiplier"), 0.5)

        # ── 恢复期渐进加仓（recovery_ramp）：紧急解除后仓位乘数按 bar 逐步回升，
        #     避免从冻结的 0 一次性跳回高位（自愈闭环，默认关闭保持原行为）──
        rr_cfg = self._eq_cfg.get("recovery_ramp", {}) or {}
        self._recovery_ramp_enabled = bool(rr_cfg.get("enabled", False))
        self._recovery_ramp_floor = max(0.1, _safe_float(rr_cfg.get("floor"), 0.3))
        self._recovery_ramp_ceiling = max(self._recovery_ramp_floor, _safe_float(rr_cfg.get("ceiling"), 1.0))
        self._recovery_ramp_step = max(0.0, _safe_float(rr_cfg.get("step_per_bar"), 0.05))
        self._recovery_bars = 0

        # ── 充值/提现检测 ──
        self._deposit_threshold: float = self._eq_cfg.get("deposit_detect_pct", 0.05)  # 5%
        self._last_known_equity: float = 0.0
        self._last_known_upl: float = 0.0

        # ── 外部资金变动追踪（充值/提现/划转）──
        self._recent_external_flows: Deque[Dict[str, Any]] = deque(maxlen=20)
        self._baseline_equity: float = 0.0
        self._baseline_updated_at: Optional[datetime] = None

        # ── 里程碑追踪 ──
        self._crossed_milestones: set = set()
        self._last_milestone: float = 0.0

        # ── 自适应参数 ──
        self._adaptive_params: Dict[str, Any] = {
            "position_multiplier": 1.0,
            "signal_quality_offset": 0.0,      # 加到 min_signal_quality 上
            "risk_budget_ratio": 1.0,           # 风险预算比率
            "max_leverage": 5,
            "max_positions": 10,
            "strategy_allocation_modifier": {},  # 策略权重修正
            "mode": "normal",
            "mode_confidence": 0.0,
        }

        # ── 账户规模分级 ──
        self._account_tier = self._detect_account_tier()

        # ── 持久化 ──
        self._state_path = os.path.join("data", "equity_monitor_state.json")
        self._save_interval = self._eq_cfg.get("save_interval", 60)

        # ── 运行时 ──
        self._running = False
        self._monitor_task = None
        self._save_counter = 0
        self._start_time = datetime.now()

        # ── 告警回调 ──
        self._alert_callback: Optional[callable] = None

    # ═══════════════════════════════════════════════════════════════
    # 生命周期
    # ═══════════════════════════════════════════════════════════════

    async def start(self):
        """启动监控"""
        self._load_state()
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info(f"EquityMonitor started (tier={self._account_tier}, peak={self._peak_equity:.2f})")

    async def stop(self):
        """停止监控"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._save_state()
        logger.info("EquityMonitor stopped")

    async def _monitor_loop(self):
        """后台监控循环：定期保存状态、检测趋势变化"""
        while self._running:
            try:
                self._save_counter += 1
                if self._save_counter >= self._save_interval // 5:
                    self._save_state()
                    self._save_counter = 0
            except Exception as e:
                logger.debug(f"EquityMonitor loop error: {e}")
            await asyncio.sleep(5)

    # ═══════════════════════════════════════════════════════════════
    # 核心：数据喂入 + 事件检测
    # ═══════════════════════════════════════════════════════════════

    def feed(self, equity: float, available: float = 0.0,
             margin: float = 0.0, upl: float = 0.0) -> Optional[EquityEvent]:
        """喂入一次权益数据，返回检测到的事件（如有）。

        Args:
            equity: 总权益 (USDT)
            available: 可用余额
            margin: 已用保证金
            upl: 未实现盈亏
        """
        if equity <= 0:
            return None

        if self._profit_reserve_enabled and self._profit_reserve_baseline <= 0:
            self._profit_reserve_baseline = equity

        now = datetime.now()
        event: Optional[EquityEvent] = None

        # ── 1. 充值/提现检测 ──
        event = self._detect_deposit_withdrawal(now, equity, upl)
        if event:
            self._events.append(event)
            # 记录外部资金变动（用于重启自愈：识别资金规模变化而非交易回撤）
            self._recent_external_flows.append({
                "type": event.event_type.value,
                "timestamp": now.isoformat(),
                "amount": event.details.get("amount", 0.0),
                "equity_after": equity,
            })
            # 外部资金变动（充值/提现/划转）与交易盈亏分治：
            # 重建回撤基准，且不触发 EMERGENCY，避免把提现误判为交易回撤
            self._reset_baseline(equity, event.event_type.value)
            self._update_smas(equity)
            self._save_snapshot(now, equity, available, margin, upl)
            # 构建临时 snapshot 用于自适应参数更新
            temp_snapshot = EquitySnapshot(
                timestamp=now, total_equity=equity,
                available_balance=available, used_margin=margin, unrealized_pnl=upl,
                drawdown_pct=(self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0,
            )
            self._update_adaptive_params(now, equity, temp_snapshot)
            self._last_known_equity = equity
            self._last_known_upl = upl
            return event

        # ── 2. 更新峰值和回撤 ──
        self._update_peaks(equity, now)

        # ── 3. 更新 SMA/EMA ──
        self._update_smas(equity)

        # ── 4. 保存快照 ──
        snapshot = self._save_snapshot(now, equity, available, margin, upl)

        # ── 5. 趋势检测 ──
        event = self._detect_trend_change(now, equity)
        if event:
            self._events.append(event)

        # ── 6. 紧急模式检测 ──
        if not event or event.event_type != EquityEventType.EMERGENCY_DROP:
            event = self._detect_emergency(now, equity) or event

        # ── 7. 里程碑检测 ──
        milestone_event = self._detect_milestone(now, equity)
        if milestone_event:
            self._events.append(milestone_event)
            event = event or milestone_event

        # ── 8. 更新自适应参数 ──
        self._update_adaptive_params(now, equity, snapshot)

        # ── 9. 更新历史记录 ──
        self._last_known_equity = equity
        self._last_known_upl = upl

        return event

    # ═══════════════════════════════════════════════════════════════
    # 充值/提现检测
    # ═══════════════════════════════════════════════════════════════

    def _detect_deposit_withdrawal(self, now: datetime, equity: float,
                                   upl: float) -> Optional[EquityEvent]:
        """检测充值/提现事件。

        判断逻辑：权益变化幅度 > 阈值，且变化量与 PnL 变化不匹配。
        """
        if self._last_known_equity <= 0:
            return None

        equity_change = equity - self._last_known_equity
        upl_change = upl - self._last_known_upl
        change_pct = abs(equity_change) / self._last_known_equity

        if change_pct < self._deposit_threshold:
            return None

        # 排除 PnL 导致的权益变化
        non_pnl_change = equity_change - upl_change
        non_pnl_pct = abs(non_pnl_change) / self._last_known_equity

        if non_pnl_pct < self._deposit_threshold * 0.5:
            return None  # 主要是 PnL 变化，不是充值/提现

        if non_pnl_change > 0:
            logger.warning(
                f"DEPOSIT DETECTED: equity {self._last_known_equity:.2f} → {equity:.2f} "
                f"(+{non_pnl_change:.2f} USDT, +{non_pnl_pct:.1%})"
            )
            self._account_tier = self._detect_account_tier()
            return EquityEvent(
                event_type=EquityEventType.DEPOSIT,
                timestamp=now,
                equity_before=self._last_known_equity,
                equity_after=equity,
                change_pct=non_pnl_pct,
                details={"amount": non_pnl_change, "upl_change": upl_change},
            )
        else:
            logger.warning(
                f"WITHDRAWAL DETECTED: equity {self._last_known_equity:.2f} → {equity:.2f} "
                f"({non_pnl_change:.2f} USDT, {non_pnl_pct:.1%})"
            )
            self._account_tier = self._detect_account_tier()
            return EquityEvent(
                event_type=EquityEventType.WITHDRAWAL,
                timestamp=now,
                equity_before=self._last_known_equity,
                equity_after=equity,
                change_pct=non_pnl_pct,
                details={"amount": non_pnl_change, "upl_change": upl_change},
            )

    # ═══════════════════════════════════════════════════════════════
    # 外部资金变动基准重建
    # ═══════════════════════════════════════════════════════════════

    def _reset_baseline(self, equity: float, reason: str):
        """外部资金变动（充值/提现/划转）时重建回撤基准。

        外部资金流动改变的是资金规模，而非交易表现。为把「提现」与
        「交易回撤」分治：
        - 重置权益峰值/谷值/最大回撤为当前权益
        - 重置 SMA/EMA 平滑基准（资金规模已变，历史平滑值失效）
        - 退出 EMERGENCY 模式（提现不是交易回撤，不应冻结开仓）
        """
        now = datetime.now()
        self._peak_equity = equity
        self._peak_equity_time = now
        self._trough_equity = equity
        self._max_drawdown_pct = 0.0
        self._emergency_low_watermark = 0.0
        self._sma_short = equity
        self._sma_long = equity
        self._ema = equity
        self._baseline_equity = equity
        self._baseline_updated_at = now
        # 就近重设里程碑基准，避免提现后里程碑跨越判定错位
        self._last_milestone = max(
            [m for m in self.MILESTONES if m <= equity] or [0]
        )
        if self._current_mode == EquityMode.EMERGENCY:
            self._current_mode = EquityMode.NORMAL
            self._mode_start_time = now
            # 退出紧急模式时同步解除开仓冻结（position_multiplier/max_positions 不得残留为 0）
            self._adaptive_params.update({
                "position_multiplier": 1.0,
                "signal_quality_offset": 0.0,
                "risk_budget_ratio": 1.0,
                "max_positions": 10,
                "mode": "normal",
                "mode_confidence": 0.0,
            })
        logger.warning(
            f"BASELINE RESET ({reason}): equity→{equity:.2f} — "
            f"peak/trough/drawdown/sma/ema re-anchored to external flow, "
            f"not treated as trade drawdown"
        )

    # ═══════════════════════════════════════════════════════════════
    # 峰值/回撤
    # ═══════════════════════════════════════════════════════════════

    def _update_peaks(self, equity: float, now: datetime):
        """更新峰值和回撤数据"""
        if equity > self._peak_equity:
            old_peak = self._peak_equity
            self._peak_equity = equity
            self._peak_equity_time = now
            if old_peak > 0:
                logger.info(f"NEW PEAK EQUITY: {old_peak:.2f} → {equity:.2f} (+{equity - old_peak:.2f})")
                self._events.append(EquityEvent(
                    event_type=EquityEventType.PEAK_EQUITY,
                    timestamp=now,
                    equity_before=old_peak,
                    equity_after=equity,
                    change_pct=(equity - old_peak) / old_peak,
                    details={"previous_peak": old_peak},
                ))

        if equity < self._trough_equity:
            self._trough_equity = equity

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - equity) / self._peak_equity
            if drawdown > self._max_drawdown_pct:
                self._max_drawdown_pct = drawdown
                if drawdown > 0.05:
                    logger.info(f"NEW MAX DRAWDOWN: {drawdown:.1%} from peak {self._peak_equity:.2f}")

    # ═══════════════════════════════════════════════════════════════
    # SMA/EMA
    # ═══════════════════════════════════════════════════════════════

    def _update_smas(self, equity: float):
        """更新短期/长期 SMA 和 EMA"""
        n = len(self._history)
        if n == 0:
            self._sma_short = equity
            self._sma_long = equity
            self._ema = equity
            return

        # EMA
        self._ema = self._ema_alpha * equity + (1 - self._ema_alpha) * self._ema

        # SMA short (最近 30 个点 ≈ 15 分钟)
        short_window = min(30, n + 1)
        recent = [s.total_equity for s in list(self._history)[-short_window + 1:]] + [equity]
        self._sma_short = sum(recent) / len(recent)

        # SMA long (最近 120 个点 ≈ 60 分钟)
        long_window = min(120, n + 1)
        older = [s.total_equity for s in list(self._history)[-long_window + 1:]] + [equity]
        self._sma_long = sum(older) / len(older)

    def _save_snapshot(self, now: datetime, equity: float, available: float,
                       margin: float, upl: float) -> EquitySnapshot:
        """保存权益快照"""
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0
        snapshot = EquitySnapshot(
            timestamp=now,
            total_equity=equity,
            available_balance=available,
            used_margin=margin,
            unrealized_pnl=upl,
            sma_short=self._sma_short,
            sma_long=self._sma_long,
            ema=self._ema,
            drawdown_pct=drawdown,
            mode=self._current_mode.value,
        )
        self._history.append(snapshot)
        return snapshot

    # ═══════════════════════════════════════════════════════════════
    # 趋势检测
    # ═══════════════════════════════════════════════════════════════

    def _detect_trend_change(self, now: datetime, equity: float) -> Optional[EquityEvent]:
        """检测趋势变化：增长/衰退/恢复"""
        if len(self._history) < 10:
            return None

        prev_equity = self._history[-1].total_equity if self._history else equity

        # 连续涨跌计数
        if equity > prev_equity * 1.001:
            self._consecutive_up += 1
            self._consecutive_down = 0
        elif equity < prev_equity * 0.999:
            self._consecutive_down += 1
            self._consecutive_up = 0
        else:
            self._consecutive_up = max(0, self._consecutive_up - 1)
            self._consecutive_down = max(0, self._consecutive_down - 1)

        # ── 增长趋势确认 ──
        if (self._consecutive_up >= self._consecutive_up_threshold
                and self._current_mode != EquityMode.GROWTH
                and equity > self._sma_short
                and self._sma_short > self._sma_long):
            self._current_mode = EquityMode.GROWTH
            self._mode_start_time = now
            logger.info(f"GROWTH TREND START: equity={equity:.2f}, consecutive_up={self._consecutive_up}")
            return EquityEvent(
                event_type=EquityEventType.GROWTH_TREND_START,
                timestamp=now, equity_before=prev_equity, equity_after=equity,
                change_pct=(equity - prev_equity) / prev_equity if prev_equity > 0 else 0,
                details={"consecutive_up": self._consecutive_up},
            )

        # ── 衰退趋势确认 ──
        if (self._consecutive_down >= self._consecutive_down_threshold
                and self._current_mode != EquityMode.DECLINE
                and equity < self._sma_short
                and self._sma_short < self._sma_long):
            self._current_mode = EquityMode.DECLINE
            self._mode_start_time = now
            logger.warning(f"DECLINE TREND START: equity={equity:.2f}, consecutive_down={self._consecutive_down}")
            return EquityEvent(
                event_type=EquityEventType.DECLINE_TREND_START,
                timestamp=now, equity_before=prev_equity, equity_after=equity,
                change_pct=(equity - prev_equity) / prev_equity if prev_equity > 0 else 0,
                details={"consecutive_down": self._consecutive_down},
            )

        # ── 恢复检测 ──
        if (self._current_mode == EquityMode.DECLINE
                and self._consecutive_up >= 3
                and equity > self._sma_short):
            self._current_mode = EquityMode.RECOVERY
            self._mode_start_time = now
            logger.info(f"RECOVERY START: equity={equity:.2f}, from decline mode")
            return EquityEvent(
                event_type=EquityEventType.RECOVERY_START,
                timestamp=now, equity_before=prev_equity, equity_after=equity,
                change_pct=(equity - prev_equity) / prev_equity if prev_equity > 0 else 0,
            )

        # ── 恢复完成 ──
        if (self._current_mode == EquityMode.RECOVERY
                and equity >= self._peak_equity * 0.98):
            self._current_mode = EquityMode.NORMAL
            self._mode_start_time = now
            logger.info(f"RECOVERY COMPLETE: equity={equity:.2f}, near peak {self._peak_equity:.2f}")
            return EquityEvent(
                event_type=EquityEventType.RECOVERY_COMPLETE,
                timestamp=now, equity_before=prev_equity, equity_after=equity,
                change_pct=(equity - self._trough_equity) / self._trough_equity if self._trough_equity > 0 else 0,
                details={"peak_equity": self._peak_equity, "trough_equity": self._trough_equity},
            )

        # ── 趋势结束 ──
        if (self._current_mode in (EquityMode.GROWTH, EquityMode.DECLINE)
                and self._consecutive_up < 3 and self._consecutive_down < 3
                and abs(equity - self._sma_short) / self._sma_short < 0.01):
            old_mode = self._current_mode
            self._current_mode = EquityMode.NORMAL
            self._mode_start_time = now
            logger.info(f"TREND END: {old_mode.value} → NORMAL, equity={equity:.2f}")
            return None  # 趋势结束不算事件

        return None

    # ═══════════════════════════════════════════════════════════════
    # 紧急模式
    # ═══════════════════════════════════════════════════════════════

    def _detect_emergency(self, now: datetime, equity: float) -> Optional[EquityEvent]:
        """检测紧急下跌和恢复"""
        if self._peak_equity <= 0:
            return None

        drawdown = (self._peak_equity - equity) / self._peak_equity

        # ── 紧急下跌 ──
        if (drawdown >= self._emergency_drop_threshold
                and self._current_mode != EquityMode.EMERGENCY):
            self._current_mode = EquityMode.EMERGENCY
            self._mode_start_time = now
            self._emergency_low_watermark = equity
            logger.error(
                f"EMERGENCY DROP: equity={equity:.2f}, drawdown={drawdown:.1%} "
                f"from peak {self._peak_equity:.2f} — ALL NEW POSITIONS FROZEN"
            )
            if self._alert_callback:
                self._alert_callback(
                    "EMERGENCY_DROP",
                    f"Equity dropped {drawdown:.1%} to {equity:.2f} USDT. "
                    f"All new positions frozen. Emergency mode active."
                )
            return EquityEvent(
                event_type=EquityEventType.EMERGENCY_DROP,
                timestamp=now, equity_before=self._peak_equity, equity_after=equity,
                change_pct=drawdown,
                details={"peak_equity": self._peak_equity, "drawdown_pct": drawdown},
            )

        # ── 紧急恢复 ──
        if self._current_mode == EquityMode.EMERGENCY:
            if equity < self._emergency_low_watermark:
                self._emergency_low_watermark = equity

            if self._emergency_low_watermark > 0:
                recovery_from_low = (equity - self._emergency_low_watermark) / self._emergency_low_watermark
            else:
                recovery_from_low = 0.0
            if recovery_from_low >= self._emergency_recovery_threshold:
                self._current_mode = EquityMode.RECOVERY
                self._mode_start_time = now
                logger.info(
                    f"EMERGENCY RECOVERED: equity={equity:.2f}, "
                    f"+{recovery_from_low:.1%} from low {self._emergency_low_watermark:.2f}"
                )
                return EquityEvent(
                    event_type=EquityEventType.EMERGENCY_RECOVERED,
                    timestamp=now,
                    equity_before=self._emergency_low_watermark,
                    equity_after=equity,
                    change_pct=recovery_from_low,
                    details={"low_watermark": self._emergency_low_watermark},
                )

        return None

    # ═══════════════════════════════════════════════════════════════
    # 里程碑
    # ═══════════════════════════════════════════════════════════════

    def _detect_milestone(self, now: datetime, equity: float) -> Optional[EquityEvent]:
        """检测里程碑跨越"""
        # 向上跨越
        for m in self.MILESTONES:
            if self._last_milestone < m <= equity and m not in self._crossed_milestones:
                self._crossed_milestones.add(m)
                self._last_milestone = m
                self._account_tier = self._detect_account_tier()
                logger.info(f"MILESTONE: equity crossed {m} USDT (tier now {self._account_tier})")
                return EquityEvent(
                    event_type=EquityEventType.MILESTONE_CROSS,
                    timestamp=now, equity_before=self._last_milestone * 0.9, equity_after=equity,
                    change_pct=(equity - m) / m,
                    details={"milestone": m, "direction": "up", "new_tier": self._account_tier},
                )

        # 向下跨越（跌破里程碑）
        for m in reversed(self.MILESTONES):
            if self._last_milestone >= m > equity and m in self._crossed_milestones:
                logger.warning(f"MILESTONE LOST: equity dropped below {m} USDT")
                # 不删除里程碑记录，但更新 last_milestone
                self._last_milestone = max(
                    [x for x in self._crossed_milestones if x <= equity] or [0]
                )
                self._account_tier = self._detect_account_tier()
                return None  # 下跌里程碑不触发事件，避免干扰

        return None

    # ═══════════════════════════════════════════════════════════════
    # 自适应参数
    # ═══════════════════════════════════════════════════════════════

    def _update_adaptive_params(self, now: datetime, equity: float,
                                snapshot: EquitySnapshot):
        """根据当前模式更新自适应参数"""
        drawdown = snapshot.drawdown_pct
        tier = self._account_tier

        # 恢复期 bar 计数：仅在 RECOVERY 模式累计，退出即归零（渐进加仓用）
        if self._current_mode == EquityMode.RECOVERY:
            self._recovery_bars += 1
        else:
            self._recovery_bars = 0

        # ── 模式权重 ──
        if self._current_mode == EquityMode.EMERGENCY:
            self._adaptive_params.update({
                "position_multiplier": 0.0,          # 禁止新开仓
                "signal_quality_offset": 0.30,       # 极高阈值
                "risk_budget_ratio": 0.0,            # 零风险预算
                "max_positions": 0,                  # 不开新仓
                "mode": "emergency",
                "mode_confidence": min(1.0, drawdown / self._emergency_drop_threshold),
            })
        elif self._current_mode == EquityMode.DECLINE:
            severity = min(1.0, drawdown / self._emergency_drop_threshold)
            self._adaptive_params.update({
                "position_multiplier": max(0.15, 0.5 * (1 - severity)),
                "signal_quality_offset": 0.05 + 0.15 * severity,
                "risk_budget_ratio": max(0.2, 1.0 - severity),
                "max_positions": max(3, int(10 * (1 - severity))),
                "mode": "decline",
                "mode_confidence": severity,
            })
        elif self._current_mode == EquityMode.GROWTH:
            # 增长模式：渐进式增加仓位
            growth_pct = (equity - self._sma_long) / self._sma_long if self._sma_long > 0 else 0
            growth_factor = min(1.0, growth_pct * 5)  # 5%增长 = 满倍率
            self._adaptive_params.update({
                "position_multiplier": 1.0 + growth_factor * 0.5,  # 最多 1.5x
                "signal_quality_offset": -0.05 * growth_factor,     # 降低阈值
                "risk_budget_ratio": 1.0 + growth_factor * 0.3,    # 放宽风险预算
                "max_positions": 15,
                "mode": "growth",
                "mode_confidence": growth_factor,
            })
        elif self._current_mode == EquityMode.RECOVERY:
            # 恢复模式：谨慎加仓
            recovery_pct = (equity - self._trough_equity) / self._peak_equity if self._peak_equity > 0 else 0
            if self._recovery_ramp_enabled:
                # 渐进加仓：按 bar 逐步从 floor 回升到 ceiling，避免一次性跳回高位
                pm = min(
                    self._recovery_ramp_ceiling,
                    self._recovery_ramp_floor + self._recovery_bars * self._recovery_ramp_step,
                )
                pm = max(0.1, pm)
            else:
                pm = 0.3 + 0.4 * recovery_pct
            self._adaptive_params.update({
                "position_multiplier": pm,
                "signal_quality_offset": 0.10 - 0.05 * recovery_pct,
                "risk_budget_ratio": 0.3 + 0.5 * recovery_pct,
                "max_positions": max(3, int(5 + 10 * recovery_pct)),
                "mode": "recovery",
                "mode_confidence": recovery_pct,
            })
        else:  # NORMAL
            self._adaptive_params.update({
                "position_multiplier": 1.0,
                "signal_quality_offset": 0.0,
                "risk_budget_ratio": 1.0,
                "max_positions": 10,
                "mode": "normal",
                "mode_confidence": 0.0,
            })

        # ── 账户规模上限修正 ──
        tier_multiplier_caps = {
            "nano": 1.5,     # < 50 USDT
            "micro": 2.0,    # 50-100 USDT
            "small": 3.0,    # 100-500 USDT
            "medium": 4.0,   # 500-2000 USDT
            "large": 5.0,    # 2000-10000 USDT
            "xlarge": 10.0,  # > 10000 USDT
        }
        cap = tier_multiplier_caps.get(tier, 2.0)
        self._adaptive_params["position_multiplier"] = min(
            self._adaptive_params["position_multiplier"], cap
        )
        self._adaptive_params["account_tier"] = tier

        # ── 分级回撤护栏：在非 EMERGENCY 模式下按回撤程度渐进收紧 ──
        self._apply_tiered_drawdown_guard(drawdown)

        # ── 账户级利润留存：盈利达阈值后自动降仓位锁利 ──
        self._apply_profit_reserve(equity)

        # ── 回撤速度预警：短窗口急跌 → 快速降仓 ──
        self._apply_drawdown_velocity_guard()

    def _apply_tiered_drawdown_guard(self, drawdown: float) -> None:
        """分级回撤护栏：在 EMERGENCY 硬冻结之前，按回撤程度渐进收紧风险。

        仅在非 EMERGENCY 模式生效（EMERGENCY 已把仓位乘数冻结为 0，无需叠加）。
        与模式参数取更保守值（min），确保回撤护栏不因模式切换而放宽。
        """
        if not self._tiered_guard_enabled or not self._tiered_levels:
            return
        if self._current_mode == EquityMode.EMERGENCY:
            return

        applied = None
        for dd, pm, rb, mp in self._tiered_levels:
            if drawdown >= dd:
                applied = (pm, rb, mp)
        if applied is None:
            return

        pm, rb, mp = applied
        self._adaptive_params["position_multiplier"] = min(
            _safe_float(self._adaptive_params.get("position_multiplier"), 1.0), pm
        )
        self._adaptive_params["risk_budget_ratio"] = min(
            _safe_float(self._adaptive_params.get("risk_budget_ratio"), 1.0), rb
        )
        self._adaptive_params["max_positions"] = min(
            int(_safe_float(self._adaptive_params.get("max_positions"), 10)), mp
        )

    def _apply_profit_reserve(self, equity: float) -> None:
        """账户级利润留存：账户较基准盈利达阈值后，渐进降低仓位乘数以锁定利润。

        当 equity 相对基准增长超过 activation_pct 时启动，增长到 max_pct 时达到
        最大留存力度（仓位乘数最多下调 max_position_multiplier_reduction 比例）。
        仅在盈利（growth > 0）时生效，且不把仓位乘数降到 0（留存而非冻结）。
        """
        if not self._profit_reserve_enabled:
            return
        baseline = self._profit_reserve_baseline
        if baseline <= 0:
            return
        growth_pct = (equity - baseline) / baseline
        if growth_pct <= self._profit_reserve_activation_pct:
            return

        span = max(self._profit_reserve_max_pct - self._profit_reserve_activation_pct, 1e-6)
        intensity = min(1.0, (growth_pct - self._profit_reserve_activation_pct) / span)
        reduction = self._profit_reserve_max_reduction * intensity

        current_multiplier = _safe_float(self._adaptive_params.get("position_multiplier"), 1.0)
        self._adaptive_params["position_multiplier"] = max(
            current_multiplier * (1.0 - reduction), 0.1
        )

    def _apply_drawdown_velocity_guard(self) -> None:
        """回撤速度预警：最近 window 个快照内权益跌幅超过阈值 → 快速降仓。

        与分级回撤护栏互补：分级护栏看「距峰值回撤幅度」，本护栏看「回撤速度」，
        能在急跌初期（尚未触及深度回撤档位）就抢先收敛仓位乘数，避免一次性砸到
        EMERGENCY 硬冻结。仅在非 EMERGENCY 模式生效，且只下调（取 min）不上调。
        """
        if not self._dv_enabled or self._current_mode == EquityMode.EMERGENCY:
            return
        if len(self._history) < self._dv_window:
            return
        window_snaps = list(self._history)[-self._dv_window:]
        start_equity = window_snaps[0].total_equity
        end_equity = window_snaps[-1].total_equity
        if start_equity <= 0:
            return
        drop_pct = (start_equity - end_equity) / start_equity
        if drop_pct < self._dv_drop_pct:
            return
        self._adaptive_params["position_multiplier"] = min(
            _safe_float(self._adaptive_params.get("position_multiplier"), 1.0),
            self._dv_position_multiplier,
        )
        logger.warning(
            f"DRAWDOWN VELOCITY: -{drop_pct:.1%} over {self._dv_window} bars → "
            f"position_multiplier <= {self._dv_position_multiplier}"
        )

    def _detect_account_tier(self) -> str:
        """检测账户规模等级"""
        if self._last_known_equity > 0:
            equity = self._last_known_equity
        else:
            return "nano"
        if equity < 50:
            return "nano"
        if equity < 100:
            return "micro"
        if equity < 500:
            return "small"
        if equity < 2000:
            return "medium"
        if equity < 10000:
            return "large"
        return "xlarge"

    # ═══════════════════════════════════════════════════════════════
    # 公共接口
    # ═══════════════════════════════════════════════════════════════

    def get_adaptive_params(self) -> Dict[str, Any]:
        """获取当前自适应参数，供外部模块（AdaptiveController 等）使用"""
        return dict(self._adaptive_params)

    def get_equity_status(self) -> Dict[str, Any]:
        """获取完整权益状态"""
        return {
            "current_equity": self._last_known_equity,
            "peak_equity": self._peak_equity,
            "trough_equity": _safe_float(self._trough_equity, 0.0),  # inf 哨兵 → 0.0，JSON 安全
            "max_drawdown_pct": self._max_drawdown_pct,
            "sma_short": self._sma_short,
            "sma_long": self._sma_long,
            "ema": self._ema,
            "mode": self._current_mode.value,
            "mode_since": self._mode_start_time.isoformat() if self._mode_start_time else None,
            "position_multiplier": self.get_position_multiplier(),
            "recovery_bars": self._recovery_bars,
            "account_tier": self._account_tier,
            "crossed_milestones": sorted(list(self._crossed_milestones)),
            "consecutive_up": self._consecutive_up,
            "consecutive_down": self._consecutive_down,
            "snapshot_count": len(self._history),
            "event_count": len(self._events),
        }

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取最近事件"""
        events = list(self._events)[-limit:]
        return [
            {
                "type": e.event_type.value,
                "timestamp": e.timestamp.isoformat(),
                "equity_before": e.equity_before,
                "equity_after": e.equity_after,
                "change_pct": e.change_pct,
                "details": e.details,
            }
            for e in events
        ]

    def get_equity_curve(self, limit: int = 200) -> List[Dict[str, Any]]:
        """获取权益曲线数据"""
        snapshots = list(self._history)[-limit:]
        return [
            {
                "timestamp": s.timestamp.isoformat(),
                "equity": s.total_equity,
                "sma_short": s.sma_short,
                "sma_long": s.sma_long,
                "ema": s.ema,
                "drawdown_pct": s.drawdown_pct,
                "mode": s.mode,
            }
            for s in snapshots
        ]

    def is_new_position_allowed(self) -> bool:
        """检查是否允许开新仓（紧急模式禁止）"""
        return self._current_mode != EquityMode.EMERGENCY

    def get_position_multiplier(self) -> float:
        """获取当前仓位乘数"""
        return self._adaptive_params.get("position_multiplier", 1.0)

    def get_signal_quality_offset(self) -> float:
        """获取信号质量阈值偏移量"""
        return self._adaptive_params.get("signal_quality_offset", 0.0)

    def set_alert_callback(self, callback: callable):
        """设置告警回调"""
        self._alert_callback = callback

    # ═══════════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════════

    def _save_state(self):
        """保存状态到文件"""
        try:
            state = {
                "peak_equity": self._peak_equity,
                "peak_equity_time": self._peak_equity_time.isoformat() if self._peak_equity_time else None,
                "trough_equity": _safe_float(self._trough_equity, 0.0),
                "max_drawdown_pct": self._max_drawdown_pct,
                "last_known_equity": self._last_known_equity,
                "last_known_upl": self._last_known_upl,
                "sma_short": self._sma_short,
                "sma_long": self._sma_long,
                "ema": self._ema,
                "current_mode": self._current_mode.value,
                "mode_start_time": self._mode_start_time.isoformat() if self._mode_start_time else None,
                "emergency_low_watermark": self._emergency_low_watermark,
                "account_tier": self._account_tier,
                "crossed_milestones": sorted(list(self._crossed_milestones)),
                "last_milestone": self._last_milestone,
                "consecutive_up": self._consecutive_up,
                "consecutive_down": self._consecutive_down,
                "adaptive_params": self._adaptive_params,
                "recent_external_flows": list(self._recent_external_flows),
                "baseline_equity": self._baseline_equity,
                "baseline_updated_at": self._baseline_updated_at.isoformat() if self._baseline_updated_at else None,
                "profit_reserve_baseline": self._profit_reserve_baseline,
                "saved_at": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        except Exception as e:
            logger.debug(f"EquityMonitor save error: {e}")

    def _load_state(self):
        """从文件加载状态"""
        try:
            if not os.path.exists(self._state_path):
                return
            with open(self._state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._peak_equity = state.get("peak_equity", 0.0)
            self._trough_equity = state.get("trough_equity", float("inf"))
            self._max_drawdown_pct = state.get("max_drawdown_pct", 0.0)
            self._last_known_equity = state.get("last_known_equity", 0.0)
            self._last_known_upl = state.get("last_known_upl", 0.0)
            self._sma_short = state.get("sma_short", 0.0)
            self._sma_long = state.get("sma_long", 0.0)
            self._ema = state.get("ema", 0.0)
            self._account_tier = state.get("account_tier", "nano")
            self._crossed_milestones = set(state.get("crossed_milestones", []))
            self._last_milestone = state.get("last_milestone", 0.0)
            self._consecutive_up = state.get("consecutive_up", 0)
            self._consecutive_down = state.get("consecutive_down", 0)
            self._emergency_low_watermark = state.get("emergency_low_watermark", 0.0)

            # 峰值时间
            peak_time = state.get("peak_equity_time")
            if peak_time:
                try:
                    self._peak_equity_time = datetime.fromisoformat(peak_time)
                except ValueError:
                    pass

            # 外部资金变动追踪恢复（充值/提现/划转）
            flows = state.get("recent_external_flows", [])
            if isinstance(flows, list):
                self._recent_external_flows = deque(flows[-20:], maxlen=20)
            self._baseline_equity = state.get("baseline_equity", 0.0)
            self._profit_reserve_baseline = _safe_float(
                state.get("profit_reserve_baseline"), self._profit_reserve_baseline
            )
            baseline_updated = state.get("baseline_updated_at")
            if baseline_updated:
                try:
                    self._baseline_updated_at = datetime.fromisoformat(baseline_updated)
                except ValueError:
                    pass

            # 恢复模式
            mode_str = state.get("current_mode", "normal")
            try:
                self._current_mode = EquityMode(mode_str)
            except ValueError:
                self._current_mode = EquityMode.NORMAL
            mode_start = state.get("mode_start_time")
            if mode_start:
                try:
                    self._mode_start_time = datetime.fromisoformat(mode_start)
                except ValueError:
                    pass

            # 恢复自适应参数
            saved_params = state.get("adaptive_params", {})
            if saved_params:
                self._adaptive_params.update(saved_params)

            # ── 外部资金变动重启自愈 ──
            # 历史峰值相对最近权益回撤 ≥ 紧急阈值，且存在近期外部资金变动记录（充值/提现/划转）
            # 或已有基准更新时间，说明资金规模被人为改变，而非交易回撤：重建基准，
            # 避免误判紧急冻结并永久卡死。
            if self._peak_equity > 0 and self._last_known_equity > 0:
                drawdown = (self._peak_equity - self._last_known_equity) / self._peak_equity
                has_external_flow = bool(self._recent_external_flows) or self._baseline_updated_at is not None
                if drawdown >= self._emergency_drop_threshold and has_external_flow:
                    logger.warning(
                        f"Restart self-heal: peak {self._peak_equity:.2f} → equity "
                        f"{self._last_known_equity:.2f} (drawdown {drawdown:.1%}) with external "
                        f"flow detected — re-anchoring baseline, not a trade drawdown"
                    )
                    self._reset_baseline(self._last_known_equity, "restart_reconcile")

            # 兜底：历史峰值远高于最近权益（如爆仓/大额提现漏检），历史峰值不可比，
            # 重置基准并退出紧急模式，避免误判深度回撤并永久冻结开仓
            if (self._peak_equity > 0 and self._last_known_equity > 0
                    and self._peak_equity > self._last_known_equity * 3):
                logger.warning(
                    f"Historical peak {self._peak_equity:.2f} far exceeds last equity "
                    f"{self._last_known_equity:.2f}, resetting baseline (different account state detected)"
                )
                self._reset_baseline(self._last_known_equity, "restart_fallback_3x")

            logger.info(
                f"EquityMonitor state loaded: equity={self._last_known_equity:.2f}, "
                f"peak={self._peak_equity:.2f}, mode={self._current_mode.value}, "
                f"tier={self._account_tier}"
            )
        except Exception as e:
            logger.debug(f"EquityMonitor load error: {e}")