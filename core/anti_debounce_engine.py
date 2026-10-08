"""
企业级统一防抖动/防频繁交易引擎 - Anti-Debounce Engine
======================================================

把散落在 FrequencyFilter（signal_pre_filter）、DecisionCoordinator._check_rate_limit、
order_executor 撤单频率保护 等位置的防频繁/防抖动/防刷单逻辑，收敛为「单一口径」的
可插拔、可解释、可配置统一引擎。

核心能力：
  1. 品种级冷却：同一品种+同一方向，在冷却窗口内禁止重复开仓
  2. 信号去重：同一品种的同一信号类型，在去重窗口内只放行一次
  3. 策略级冷却：同一策略的所有交易，在冷却窗口内限制频率
  4. 全局冷却：所有策略的所有交易，在冷却窗口内限制频率
  5. 亏损后冷却：任意品种亏损后，进入冷却期
  6. 刷单检测：同一品种短时间内多方向频繁交易检测
  7. 自适应冷却：根据市场波动率 / 账户档位 / 回撤动态调整冷却时间

设计原则：
  - 纯内存状态（无 DB 依赖），单测可独立运行
  - 线程安全（RLock 保护）
  - 可解释：每次 check 返回 DeounceResult，含 breakdown 字段
  - 旧 FrequencyFilter 保留回退，本引擎作为新主路径
"""

from __future__ import annotations

import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

from loguru import logger


# ============================================================================
# 常量
# ============================================================================

class DebounceLayer(Enum):
    """防抖动层级"""
    SYMBOL = "symbol"               # 品种级冷却
    SIGNAL_DEDUP = "signal_dedup"   # 信号去重
    STRATEGY = "strategy"           # 策略级冷却
    GLOBAL = "global"               # 全局冷却
    LOSS_COOLDOWN = "loss_cooldown" # 亏损后冷却
    CHURN = "churn"                 # 刷单检测
    ADAPTIVE = "adaptive"           # 自适应（叠加层）


# 默认冷却时间（秒）
DEFAULT_SYMBOL_COOLDOWN = 30.0          # 同一品种+方向 30s
DEFAULT_SIGNAL_DEDUP_WINDOW = 10.0      # 信号去重 10s
DEFAULT_STRATEGY_COOLDOWN = 5.0         # 策略级 5s
DEFAULT_GLOBAL_COOLDOWN = 2.0           # 全局 2s
DEFAULT_LOSS_COOLDOWN = 120.0           # 亏损后 120s
DEFAULT_CHURN_WINDOW = 60.0             # 刷单检测窗口 60s
DEFAULT_CHURN_MAX_FLIPS = 3             # 刷单检测：窗口内最多 3 次方向翻转
DEFAULT_CHURN_BLOCK_SECONDS = 300.0     # 刷单触发后封锁 300s
DEFAULT_ADAPTIVE_VOLATILITY_THRESHOLD = 0.03  # 波动率阈值（3%），高于此值延长冷却

# 自适应冷却倍率
ADAPTIVE_VOLATILITY_MULTIPLIER = 1.5    # 高波动时冷却×1.5
ADAPTIVE_DRAWDOWN_MULTIPLIER = 1.3      # R115: 2.0→1.3 回撤冷却倍率降低
ADAPTIVE_NANO_TIER_MULTIPLIER = 1.3     # nano 账户冷却×1.3
ADAPTIVE_MICRO_TIER_MULTIPLIER = 1.1    # micro 账户冷却×1.1


# ============================================================================
# 数据模型
# ============================================================================

@dataclass
class DebounceResult:
    """防抖动引擎检查结果"""
    allowed: bool = True
    blocked_layer: Optional[str] = None
    blocked_reason: str = ""
    remaining_cooldown: float = 0.0       # 剩余冷却时间（秒）
    adaptive_multiplier: float = 1.0      # 当前自适应冷却倍率
    breakdown: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "blocked_layer": self.blocked_layer,
            "blocked_reason": self.blocked_reason,
            "remaining_cooldown": self.remaining_cooldown,
            "adaptive_multiplier": self.adaptive_multiplier,
            "breakdown": self.breakdown,
        }


@dataclass
class _SymbolRecord:
    """品种级交易记录"""
    symbol: str
    direction: str
    timestamp: float
    is_close: bool = False
    pnl_usdt: float = 0.0

    @property
    def is_open(self) -> bool:
        return not self.is_close


@dataclass
class _SignalDedupRecord:
    """信号去重记录"""
    symbol: str
    signal_type: str
    timestamp: float


# ============================================================================
# 防抖动引擎
# ============================================================================

class AntiDebounceEngine:
    """
    企业级统一防抖动/防频繁交易引擎。

    用法:
        engine = AntiDebounceEngine(config)
        result = engine.check(
            symbol="BTC-USDT-SWAP",
            strategy_name="grid",
            direction="long",
            signal_type="open",
            is_close=False,
        )
        if not result.allowed:
            logger.warning(f"Debounce blocked: {result.blocked_reason}")
        engine.record(symbol="BTC-USDT-SWAP", strategy_name="grid", direction="long",
                      signal_type="open", pnl_usdt=0.0)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = self.config.get("anti_debounce", {})

        # ── 冷却时间 ──
        self._symbol_cooldown = float(cfg.get("symbol_cooldown_seconds", DEFAULT_SYMBOL_COOLDOWN))
        self._signal_dedup_window = float(cfg.get("signal_dedup_window_seconds", DEFAULT_SIGNAL_DEDUP_WINDOW))
        self._strategy_cooldown = float(cfg.get("strategy_cooldown_seconds", DEFAULT_STRATEGY_COOLDOWN))
        self._global_cooldown = float(cfg.get("global_cooldown_seconds", DEFAULT_GLOBAL_COOLDOWN))
        self._loss_cooldown = float(cfg.get("loss_cooldown_seconds", DEFAULT_LOSS_COOLDOWN))
        self._churn_window = float(cfg.get("churn_window_seconds", DEFAULT_CHURN_WINDOW))
        self._churn_max_flips = int(cfg.get("churn_max_flips", DEFAULT_CHURN_MAX_FLIPS))
        self._churn_block_seconds = float(cfg.get("churn_block_seconds", DEFAULT_CHURN_BLOCK_SECONDS))
        self._adaptive_volatility_threshold = float(
            cfg.get("adaptive_volatility_threshold", DEFAULT_ADAPTIVE_VOLATILITY_THRESHOLD)
        )

        # ── 功能开关 ──
        self._enable_symbol = bool(cfg.get("enable_symbol", True))
        self._enable_signal_dedup = bool(cfg.get("enable_signal_dedup", True))
        self._enable_strategy = bool(cfg.get("enable_strategy", True))
        self._enable_global = bool(cfg.get("enable_global", True))
        self._enable_loss_cooldown = bool(cfg.get("enable_loss_cooldown", True))
        self._enable_churn = bool(cfg.get("enable_churn", True))
        self._enable_adaptive = bool(cfg.get("enable_adaptive", True))

        # ── 状态 ──
        self._lock = threading.RLock()

        # 品种级：{symbol: {direction: _SymbolRecord}}
        self._symbol_records: Dict[str, Dict[str, _SymbolRecord]] = {}

        # 信号去重：{dedup_key: _SignalDedupRecord}
        self._signal_dedup: Dict[str, _SignalDedupRecord] = {}

        # 策略级：{strategy_name: deque[timestamp]}
        self._strategy_timestamps: Dict[str, deque] = {}

        # 全局：deque[timestamp]
        self._global_timestamps: deque = deque(maxlen=200)

        # 亏损后冷却：{symbol: timestamp}
        self._loss_timestamps: Dict[str, float] = {}

        # 刷单检测：{symbol: deque[(timestamp, direction)]}
        self._churn_history: Dict[str, deque] = {}

        # 刷单封锁：{symbol: block_until_ts}
        self._churn_blocked: Dict[str, float] = {}

        # 自适应状态
        self._current_volatility: float = 0.0
        self._current_drawdown: float = 0.0
        self._account_tier: str = "small"

        # ── 统计计数器（供 dashboard 展示拦截命中情况）──
        self._total_checks: int = 0
        self._total_allowed: int = 0
        self._total_blocked: int = 0
        self._block_counters: Dict[str, int] = {}

        logger.info(
            f"AntiDebounceEngine initialized: symbol_cooldown={self._symbol_cooldown}s, "
            f"signal_dedup={self._signal_dedup_window}s, strategy_cooldown={self._strategy_cooldown}s, "
            f"global_cooldown={self._global_cooldown}s, loss_cooldown={self._loss_cooldown}s, "
            f"churn_window={self._churn_window}s, churn_max_flips={self._churn_max_flips}, "
            f"adaptive_vol_threshold={self._adaptive_volatility_threshold}"
        )

    # ════════════════════════════════════════════════════════════════════
    # 自适应倍率
    # ════════════════════════════════════════════════════════════════════

    def _compute_adaptive_multiplier(self) -> float:
        """计算自适应冷却倍率：波动率 × 回撤 × 账户档位"""
        if not self._enable_adaptive:
            return 1.0

        multiplier = 1.0

        # 波动率因子
        if self._current_volatility > self._adaptive_volatility_threshold:
            multiplier *= ADAPTIVE_VOLATILITY_MULTIPLIER
            logger.debug(f"AntiDebounce: volatility={self._current_volatility:.4f} > threshold, "
                         f"multiplier={multiplier:.2f}")

        # 回撤因子
        if self._current_drawdown > 0.10:
            multiplier *= ADAPTIVE_DRAWDOWN_MULTIPLIER
            logger.debug(f"AntiDebounce: drawdown={self._current_drawdown:.2%} > 10%, "
                         f"multiplier={multiplier:.2f}")

        # 账户档位因子
        tier_multiplier = {
            "nano": ADAPTIVE_NANO_TIER_MULTIPLIER,
            "micro": ADAPTIVE_MICRO_TIER_MULTIPLIER,
        }.get(self._account_tier, 1.0)
        if tier_multiplier != 1.0:
            multiplier *= tier_multiplier
            logger.debug(f"AntiDebounce: tier={self._account_tier}, multiplier={multiplier:.2f}")

        return round(multiplier, 3)

    def set_market_state(self, volatility: float = 0.0, drawdown: float = 0.0,
                         account_tier: str = "small"):
        """更新市场状态（供自适应冷却使用）"""
        self._current_volatility = volatility
        self._current_drawdown = drawdown
        self._account_tier = account_tier

    # ════════════════════════════════════════════════════════════════════
    # 核心检查
    # ════════════════════════════════════════════════════════════════════

    def check(
        self,
        symbol: str,
        strategy_name: str = "",
        direction: str = "long",
        signal_type: str = "",
        is_close: bool = False,
        pnl_usdt: float = 0.0,
    ) -> DebounceResult:
        """检查信号是否可以通过防抖动引擎。

        返回 DebounceResult，allowed=False 表示被拦截。
        """
        now = time.time()
        adaptive_mult = self._compute_adaptive_multiplier()
        breakdown: Dict[str, Any] = {
            "adaptive_multiplier": adaptive_mult,
            "layers": {},
        }

        with self._lock:
            self._total_checks += 1

            # ── Layer 0: 平仓信号不拦截（保证风控退路）──
            if is_close:
                return DebounceResult(
                    allowed=True,
                    adaptive_multiplier=adaptive_mult,
                    breakdown={"info": "close 信号不受防抖动拦截"},
                )

            # ── Layer 1: 品种级冷却 ──
            if self._enable_symbol:
                r = self._check_symbol(symbol, direction, now, adaptive_mult)
                breakdown["layers"][DebounceLayer.SYMBOL.value] = {
                    "checked": True, "blocked": not r[0],
                    "remaining": r[2] if not r[0] else 0.0,
                }
                if not r[0]:
                    self._record_block(DebounceLayer.SYMBOL.value)
                    return DebounceResult(
                        allowed=False,
                        blocked_layer=DebounceLayer.SYMBOL.value,
                        blocked_reason=r[1],
                        remaining_cooldown=r[2],
                        adaptive_multiplier=adaptive_mult,
                        breakdown=breakdown,
                    )

            # ── Layer 2: 信号去重 ──
            if self._enable_signal_dedup and signal_type:
                r = self._check_signal_dedup(symbol, signal_type, now, adaptive_mult)
                breakdown["layers"][DebounceLayer.SIGNAL_DEDUP.value] = {
                    "checked": True, "blocked": not r[0],
                    "remaining": r[2] if not r[0] else 0.0,
                }
                if not r[0]:
                    self._record_block(DebounceLayer.SIGNAL_DEDUP.value)
                    return DebounceResult(
                        allowed=False,
                        blocked_layer=DebounceLayer.SIGNAL_DEDUP.value,
                        blocked_reason=r[1],
                        remaining_cooldown=r[2],
                        adaptive_multiplier=adaptive_mult,
                        breakdown=breakdown,
                    )

            # ── Layer 3: 策略级冷却 ──
            if self._enable_strategy and strategy_name:
                r = self._check_strategy(strategy_name, now, adaptive_mult)
                breakdown["layers"][DebounceLayer.STRATEGY.value] = {
                    "checked": True, "blocked": not r[0],
                    "remaining": r[2] if not r[0] else 0.0,
                }
                if not r[0]:
                    self._record_block(DebounceLayer.STRATEGY.value)
                    return DebounceResult(
                        allowed=False,
                        blocked_layer=DebounceLayer.STRATEGY.value,
                        blocked_reason=r[1],
                        remaining_cooldown=r[2],
                        adaptive_multiplier=adaptive_mult,
                        breakdown=breakdown,
                    )

            # ── Layer 4: 全局冷却 ──
            if self._enable_global:
                r = self._check_global(now, adaptive_mult)
                breakdown["layers"][DebounceLayer.GLOBAL.value] = {
                    "checked": True, "blocked": not r[0],
                    "remaining": r[2] if not r[0] else 0.0,
                }
                if not r[0]:
                    self._record_block(DebounceLayer.GLOBAL.value)
                    return DebounceResult(
                        allowed=False,
                        blocked_layer=DebounceLayer.GLOBAL.value,
                        blocked_reason=r[1],
                        remaining_cooldown=r[2],
                        adaptive_multiplier=adaptive_mult,
                        breakdown=breakdown,
                    )

            # ── Layer 5: 亏损后冷却 ──
            if self._enable_loss_cooldown and pnl_usdt < 0:
                r = self._check_loss_cooldown(symbol, now, adaptive_mult)
                breakdown["layers"][DebounceLayer.LOSS_COOLDOWN.value] = {
                    "checked": True, "blocked": not r[0],
                    "remaining": r[2] if not r[0] else 0.0,
                }
                if not r[0]:
                    self._record_block(DebounceLayer.LOSS_COOLDOWN.value)
                    return DebounceResult(
                        allowed=False,
                        blocked_layer=DebounceLayer.LOSS_COOLDOWN.value,
                        blocked_reason=r[1],
                        remaining_cooldown=r[2],
                        adaptive_multiplier=adaptive_mult,
                        breakdown=breakdown,
                    )

            # ── Layer 6: 刷单检测 ──
            if self._enable_churn:
                r = self._check_churn(symbol, direction, now, adaptive_mult)
                breakdown["layers"][DebounceLayer.CHURN.value] = {
                    "checked": True, "blocked": not r[0],
                    "remaining": r[2] if not r[0] else 0.0,
                }
                if not r[0]:
                    self._record_block(DebounceLayer.CHURN.value)
                    return DebounceResult(
                        allowed=False,
                        blocked_layer=DebounceLayer.CHURN.value,
                        blocked_reason=r[1],
                        remaining_cooldown=r[2],
                        adaptive_multiplier=adaptive_mult,
                        breakdown=breakdown,
                    )

        with self._lock:
            self._total_allowed += 1
        return DebounceResult(
            allowed=True,
            adaptive_multiplier=adaptive_mult,
            breakdown=breakdown,
        )

    def record(
        self,
        symbol: str,
        strategy_name: str = "",
        direction: str = "long",
        signal_type: str = "",
        is_close: bool = False,
        pnl_usdt: float = 0.0,
    ):
        """记录一笔交易执行，更新所有防抖动状态。

        应在信号被执行（下单成功）后调用，而非 check 时调用。
        """
        now = time.time()

        with self._lock:
            # 品种级记录
            if symbol not in self._symbol_records:
                self._symbol_records[symbol] = {}
            self._symbol_records[symbol][direction] = _SymbolRecord(
                symbol=symbol, direction=direction, timestamp=now,
                is_close=is_close, pnl_usdt=pnl_usdt,
            )

            # 信号去重记录
            if signal_type:
                dedup_key = f"{symbol}:{signal_type}"
                self._signal_dedup[dedup_key] = _SignalDedupRecord(
                    symbol=symbol, signal_type=signal_type, timestamp=now,
                )

            # 策略级时间戳
            if strategy_name:
                if strategy_name not in self._strategy_timestamps:
                    self._strategy_timestamps[strategy_name] = deque(maxlen=100)
                self._strategy_timestamps[strategy_name].append(now)

            # 全局时间戳
            self._global_timestamps.append(now)

            # 亏损后冷却记录
            if pnl_usdt < 0:
                self._loss_timestamps[symbol] = now
                logger.info(f"AntiDebounce: loss recorded for {symbol}, pnl={pnl_usdt:.4f} USDT, "
                            f"cooldown={self._loss_cooldown * self._compute_adaptive_multiplier():.0f}s")

            # 刷单检测记录
            if symbol not in self._churn_history:
                self._churn_history[symbol] = deque(maxlen=50)
            self._churn_history[symbol].append((now, direction))

    # ════════════════════════════════════════════════════════════════════
    # 各层检查实现
    # ════════════════════════════════════════════════════════════════════

    def _record_block(self, layer: str):
        """记录一次拦截命中（调用时需持有 _lock）。"""
        self._block_counters[layer] = self._block_counters.get(layer, 0) + 1
        self._total_blocked += 1

    def _cooldown(self, base: float, adaptive_mult: float) -> float:
        return base * adaptive_mult

    def _check_symbol(self, symbol: str, direction: str, now: float,
                      adaptive_mult: float) -> Tuple[bool, str, float]:
        """品种级冷却：同一品种+同一方向，在冷却窗口内禁止重复开仓。"""
        cooldown = self._cooldown(self._symbol_cooldown, adaptive_mult)
        symbol_dir = self._symbol_records.get(symbol, {})
        record = symbol_dir.get(direction)
        if record is None:
            return (True, "", 0.0)

        elapsed = now - record.timestamp
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return (False, f"{symbol} {direction} 品种级冷却中: 距上次开仓{elapsed:.1f}s < {cooldown:.1f}s", remaining)

        return (True, "", 0.0)

    def _check_signal_dedup(self, symbol: str, signal_type: str, now: float,
                            adaptive_mult: float) -> Tuple[bool, str, float]:
        """信号去重：同一品种+同一信号类型，在去重窗口内只放行一次。"""
        dedup_key = f"{symbol}:{signal_type}"
        cooldown = self._cooldown(self._signal_dedup_window, adaptive_mult)
        record = self._signal_dedup.get(dedup_key)
        if record is None:
            return (True, "", 0.0)

        elapsed = now - record.timestamp
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return (False, f"{symbol} 信号去重: {signal_type} 距上次{elapsed:.1f}s < {cooldown:.1f}s", remaining)

        return (True, "", 0.0)

    def _check_strategy(self, strategy_name: str, now: float,
                        adaptive_mult: float) -> Tuple[bool, str, float]:
        """策略级冷却：同一策略所有交易，在冷却窗口内限制频率。"""
        cooldown = self._cooldown(self._strategy_cooldown, adaptive_mult)
        timestamps = self._strategy_timestamps.get(strategy_name)
        if timestamps is None or len(timestamps) == 0:
            return (True, "", 0.0)

        last_ts = timestamps[-1]
        elapsed = now - last_ts
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return (False, f"策略 {strategy_name} 冷却中: 距上次交易{elapsed:.1f}s < {cooldown:.1f}s", remaining)

        return (True, "", 0.0)

    def _check_global(self, now: float, adaptive_mult: float) -> Tuple[bool, str, float]:
        """全局冷却：所有交易，在冷却窗口内限制频率。"""
        cooldown = self._cooldown(self._global_cooldown, adaptive_mult)
        if len(self._global_timestamps) == 0:
            return (True, "", 0.0)

        last_ts = self._global_timestamps[-1]
        elapsed = now - last_ts
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return (False, f"全局冷却中: 距上次交易{elapsed:.1f}s < {cooldown:.1f}s", remaining)

        return (True, "", 0.0)

    def _check_loss_cooldown(self, symbol: str, now: float,
                             adaptive_mult: float) -> Tuple[bool, str, float]:
        """亏损后冷却：该品种最近一笔亏损后，进入冷却期。"""
        cooldown = self._cooldown(self._loss_cooldown, adaptive_mult)
        loss_ts = self._loss_timestamps.get(symbol)
        if loss_ts is None:
            return (True, "", 0.0)

        elapsed = now - loss_ts
        if elapsed < cooldown:
            remaining = cooldown - elapsed
            return (False, f"{symbol} 亏损后冷却中: 距上次亏损{elapsed:.1f}s < {cooldown:.1f}s", remaining)

        return (True, "", 0.0)

    def _check_churn(self, symbol: str, direction: str, now: float,
                     adaptive_mult: float) -> Tuple[bool, str, float]:
        """刷单检测：同一品种在窗口内方向翻转次数超过阈值。

        检测逻辑：
        1. 优先检查是否已被封锁（刷单触发后封锁 churn_block_seconds）
        2. 统计窗口内方向翻转次数
        3. 超过阈值则封锁并返回拦截
        """
        # 检查是否在封锁期
        block_until = self._churn_blocked.get(symbol)
        if block_until is not None and now < block_until:
            remaining = block_until - now
            return (False, f"{symbol} 刷单封锁中: 剩余{remaining:.0f}s", remaining)

        history = self._churn_history.get(symbol)
        if history is None or len(history) < 2:
            return (True, "", 0.0)

        # 统计窗口内方向翻转次数
        flip_count = 0
        prev_dir = None
        window_start = now - self._churn_window

        for ts, d in history:
            if ts < window_start:
                continue
            if prev_dir is not None and d != prev_dir:
                flip_count += 1
            prev_dir = d

        if flip_count >= self._churn_max_flips:
            # 封锁
            self._churn_blocked[symbol] = now + self._churn_block_seconds
            logger.warning(
                f"AntiDebounce: CHURN detected for {symbol}, "
                f"flips={flip_count} in {self._churn_window}s, "
                f"blocked for {self._churn_block_seconds}s"
            )
            return (False, f"{symbol} 刷单检测: {flip_count}次方向翻转/{self._churn_window:.0f}s, "
                          f"封锁{self._churn_block_seconds:.0f}s", self._churn_block_seconds)

        return (True, "", 0.0)

    # ════════════════════════════════════════════════════════════════════
    # 管理接口
    # ════════════════════════════════════════════════════════════════════

    def reset_symbol(self, symbol: str):
        """重置某个品种的所有防抖动状态"""
        with self._lock:
            self._symbol_records.pop(symbol, None)
            self._loss_timestamps.pop(symbol, None)
            self._churn_history.pop(symbol, None)
            self._churn_blocked.pop(symbol, None)
            # 清理信号去重记录
            keys_to_remove = [k for k in self._signal_dedup if k.startswith(f"{symbol}:")]
            for k in keys_to_remove:
                self._signal_dedup.pop(k, None)

    def reset_all(self):
        """重置所有防抖动状态"""
        with self._lock:
            self._symbol_records.clear()
            self._signal_dedup.clear()
            self._strategy_timestamps.clear()
            self._global_timestamps.clear()
            self._loss_timestamps.clear()
            self._churn_history.clear()
            self._churn_blocked.clear()

    def get_symbol_status(self, symbol: str) -> Dict[str, Any]:
        """获取某个品种的防抖动状态快照"""
        with self._lock:
            now = time.time()
            status: Dict[str, Any] = {"symbol": symbol}

            # 品种级
            symbol_dir = self._symbol_records.get(symbol, {})
            for direction, record in symbol_dir.items():
                status[f"last_{direction}_ts"] = record.timestamp
                status[f"last_{direction}_elapsed"] = now - record.timestamp

            # 亏损冷却
            loss_ts = self._loss_timestamps.get(symbol)
            if loss_ts is not None:
                status["last_loss_ts"] = loss_ts
                status["last_loss_elapsed"] = now - loss_ts

            # 刷单封锁
            block_until = self._churn_blocked.get(symbol)
            if block_until is not None and now < block_until:
                status["churn_blocked"] = True
                status["churn_block_remaining"] = block_until - now

            # 刷单历史
            history = self._churn_history.get(symbol)
            if history:
                status["churn_history_count"] = len(history)

            return status

    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        with self._lock:
            now = time.time()
            return {
                "symbol_records": sum(len(d) for d in self._symbol_records.values()),
                "signal_dedup_keys": len(self._signal_dedup),
                "strategy_timestamps": {k: len(v) for k, v in self._strategy_timestamps.items()},
                "global_timestamps": len(self._global_timestamps),
                "loss_timestamps": len(self._loss_timestamps),
                "churn_blocked_symbols": [s for s, t in self._churn_blocked.items() if now < t],
                "churn_history_symbols": list(self._churn_history.keys()),
                "adaptive_state": {
                    "volatility": self._current_volatility,
                    "drawdown": self._current_drawdown,
                    "tier": self._account_tier,
                    "multiplier": self._compute_adaptive_multiplier(),
                },
            }

    def is_churn_blocked(self, symbol: str) -> bool:
        """检查某个品种是否处于刷单封锁状态"""
        with self._lock:
            block_until = self._churn_blocked.get(symbol)
            if block_until is None:
                return False
            return time.time() < block_until

    # ════════════════════════════════════════════════════════════════════
    # 完整状态快照（供 dashboard 跨进程读取）
    # ════════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """获取完整状态快照：配置 + 拦截计数 + 当前冷却状态 + 自适应状态。"""
        with self._lock:
            now = time.time()
            layer_hits = {
                DebounceLayer.SYMBOL.value: self._block_counters.get(DebounceLayer.SYMBOL.value, 0),
                DebounceLayer.SIGNAL_DEDUP.value: self._block_counters.get(DebounceLayer.SIGNAL_DEDUP.value, 0),
                DebounceLayer.STRATEGY.value: self._block_counters.get(DebounceLayer.STRATEGY.value, 0),
                DebounceLayer.GLOBAL.value: self._block_counters.get(DebounceLayer.GLOBAL.value, 0),
                DebounceLayer.LOSS_COOLDOWN.value: self._block_counters.get(DebounceLayer.LOSS_COOLDOWN.value, 0),
                DebounceLayer.CHURN.value: self._block_counters.get(DebounceLayer.CHURN.value, 0),
            }
            return {
                "timestamp": now,
                "enabled": True,
                "config": {
                    "symbol_cooldown_seconds": self._symbol_cooldown,
                    "signal_dedup_window_seconds": self._signal_dedup_window,
                    "strategy_cooldown_seconds": self._strategy_cooldown,
                    "global_cooldown_seconds": self._global_cooldown,
                    "loss_cooldown_seconds": self._loss_cooldown,
                    "churn_window_seconds": self._churn_window,
                    "churn_max_flips": self._churn_max_flips,
                    "churn_block_seconds": self._churn_block_seconds,
                    "enable_symbol": self._enable_symbol,
                    "enable_signal_dedup": self._enable_signal_dedup,
                    "enable_strategy": self._enable_strategy,
                    "enable_global": self._enable_global,
                    "enable_loss_cooldown": self._enable_loss_cooldown,
                    "enable_churn": self._enable_churn,
                    "enable_adaptive": self._enable_adaptive,
                },
                "counters": {
                    "total_checks": self._total_checks,
                    "total_allowed": self._total_allowed,
                    "total_blocked": self._total_blocked,
                    "block_rate": round(self._total_blocked / self._total_checks, 4) if self._total_checks > 0 else 0.0,
                    "layer_hits": layer_hits,
                },
                "current": {
                    "symbol_records": sum(len(d) for d in self._symbol_records.values()),
                    "signal_dedup_keys": len(self._signal_dedup),
                    "strategy_timestamps": {k: len(v) for k, v in self._strategy_timestamps.items()},
                    "loss_timestamps": len(self._loss_timestamps),
                    "churn_blocked_symbols": [s for s, t in self._churn_blocked.items() if now < t],
                    "churn_history_symbols": list(self._churn_history.keys()),
                },
                "adaptive": {
                    "volatility": self._current_volatility,
                    "drawdown": self._current_drawdown,
                    "tier": self._account_tier,
                    "multiplier": self._compute_adaptive_multiplier(),
                },
            }

    def export_status(self, filepath: str) -> bool:
        """把状态快照写入 JSON 文件（供 dashboard 独立进程读取）。"""
        import json
        import os
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.get_status(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.warning(f"AntiDebounceEngine export_status failed: {e}")
            return False

    @staticmethod
    def load_status(filepath: str) -> Optional[Dict[str, Any]]:
        """从 JSON 文件读取状态快照（dashboard 侧使用；文件不存在返回 None）。"""
        import json
        import os
        try:
            if not os.path.exists(filepath):
                return None
            with open(filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"AntiDebounceEngine load_status failed: {e}")
            return None