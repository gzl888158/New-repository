"""
Grid 自适应资金利用率引擎（企业级）

核心功能：
  1. 逐币种追踪 grid 策略绩效（PnL / 胜率 / 盈亏比 / 连续盈亏）
  2. 计算「盈利置信度」评分（0-1），综合多维度绩效指标
  3. 将置信度映射为「仓位乘数」，逐步渐进提升有效正收益币种的资金利用率
  4. 内置安全约束：最低样本量、熔断、回撤保护、单步变化上限
  5. 与 CapitalUtilizationEngine 和 DynamicAllocator 协同

设计原则：
  - 渐进性：仓位乘数每次变化 ≤ ±0.05，防止剧烈波动
  - 非对称性：提升慢（需持续盈利），降仓快（亏损立即响应）
  - 安全优先：任一安全条件触发即冻结乘数，不回退到安全线以下
"""

import math
import os
import json
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from loguru import logger


# ═══════════════════════════════════════════════════════════════
# 数据模型
# ═══════════════════════════════════════════════════════════════

@dataclass
class GridSymbolPerformance:
    """单个币种的 grid 绩效快照"""
    symbol: str
    # 基础统计
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    # 连续统计
    consecutive_wins: int = 0
    consecutive_losses: int = 0
    # 衍生指标
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_profit_per_trade: float = 0.0
    avg_loss_per_trade: float = 0.0
    # 近期趋势
    recent_pnl: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))  # [(is_profit, pnl), ...]
    # 时间戳
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    last_update_time: Optional[datetime] = None


@dataclass
class GridUtilizationReport:
    """Grid 利用率报告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    # 每币种详情
    symbols: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # 全局汇总
    avg_confidence: float = 0.0
    avg_multiplier: float = 1.0
    active_symbols: int = 0
    frozen_symbols: int = 0
    # 动作日志
    actions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 引擎主体
# ═══════════════════════════════════════════════════════════════

class GridAdaptiveUtilizationEngine:
    """Grid 自适应资金利用率引擎。

    在每个 rebalance 周期被调用，根据各币种 grid 绩效计算仓位乘数，
    逐步提升有效正收益币种的资金利用率。

    用法：
        engine = GridAdaptiveUtilizationEngine(config)
        engine.set_performance_provider(grid_strategy)  # 注入 grid 策略实例

        # 每周期调用
        report = engine.evaluate()
        multipliers = engine.get_position_multipliers()
        # 在 grid 下单时：position_size = base_size * multiplier
    """

    # ── 默认配置 ──
    DEFAULTS = {
        "min_trades_for_activation": 20,       # 最少交易笔数才激活自适应
        "min_trades_for_confidence": 50,       # 最少交易笔数才信任置信度
        "confidence_lookback_trades": 30,      # 置信度只看最近N笔
        "max_position_multiplier": 2.50,       # 最大仓位乘数（grid 高利用率基准）
        "min_position_multiplier": 0.50,       # 最小仓位乘数（亏损时降仓下限）
        "step_up_limit": 0.05,                 # 单步提升上限
        "step_down_limit": 0.10,               # 单步下降上限（更快）
        "win_rate_floor": 0.35,                # 胜率下限：低于此值冻结乘数
        "profit_factor_floor": 0.80,           # 盈亏比下限
        "consecutive_loss_freeze": 5,          # 连续亏损冻结阈值
        "consecutive_loss_reduce": 3,          # 连续亏损开始降仓
        "confidence_recent_weight": 0.60,      # 近期绩效在置信度中的权重
        "confidence_historical_weight": 0.40,  # 历史绩效在置信度中的权重
        "freeze_after_reduce_count": 3,        # 连续降仓N次后冻结
        "recovery_required_trades": 10,         # 冻结后恢复所需交易数
        "stale_minutes": 120,                  # 超过此时间未交易视为过期
        "persist_path": "data/grid_adaptive_state.json",
        "save_interval_seconds": 300,          # 持久化间隔
    }

    def __init__(self, config: Dict[str, Any] = None):
        cfg = (config or {}).get("grid_adaptive_utilization", {})
        for key, default in self.DEFAULTS.items():
            setattr(self, f"_{key}", cfg.get(key, default))

        # ── 运行时状态 ──
        self._lock = threading.Lock()
        self._symbols: Dict[str, GridSymbolPerformance] = {}
        self._position_multipliers: Dict[str, float] = {}  # symbol → multiplier
        self._last_report: Optional[GridUtilizationReport] = None
        self._frozen: Dict[str, str] = {}  # symbol → freeze_reason
        self._reduce_count: Dict[str, int] = {}  # symbol → 连续降仓次数
        self._recovery_trades: Dict[str, int] = {}  # symbol → 冻结后累计交易数

        # ── 外部依赖 ──
        self._grid_strategy = None  # 延迟注入
        self._capital_utilization_engine = None

        # ── 持久化 ──
        self._last_save_time = 0.0
        self._load_state()

        logger.info(
            f"GridAdaptiveUtilizationEngine initialized: "
            f"max_mult={self._max_position_multiplier:.1f}x, "
            f"step_up={self._step_up_limit:.2f}, step_down={self._step_down_limit:.2f}"
        )

    # ═══════════════════════════════════════════════════════════════
    # 依赖注入
    # ═══════════════════════════════════════════════════════════════

    def set_grid_strategy(self, strategy):
        """注入 grid 策略实例，用于读取绩效数据"""
        self._grid_strategy = strategy

    def set_capital_utilization_engine(self, engine):
        """注入 CapitalUtilizationEngine，用于获取全局利用率上下文"""
        self._capital_utilization_engine = engine

    # ═══════════════════════════════════════════════════════════════
    # 绩效数据输入
    # ═══════════════════════════════════════════════════════════════

    def update_trade(self, symbol: str, is_profit: bool, pnl: float):
        """记录单笔交易结果（由 grid 策略在平仓时回调）"""
        with self._lock:
            if symbol not in self._symbols:
                self._symbols[symbol] = GridSymbolPerformance(symbol=symbol)

            perf = self._symbols[symbol]
            now = datetime.now()

            perf.total_trades += 1
            perf.total_pnl += pnl
            perf.last_update_time = now
            if perf.first_trade_time is None:
                perf.first_trade_time = now
            perf.last_trade_time = now

            if is_profit:
                perf.wins += 1
                perf.total_profit += pnl
                perf.consecutive_wins += 1
                perf.consecutive_losses = 0
            else:
                perf.losses += 1
                perf.total_loss += abs(pnl)
                perf.consecutive_losses += 1
                perf.consecutive_wins = 0

            total = perf.wins + perf.losses
            if total > 0:
                perf.win_rate = perf.wins / total
                perf.avg_profit_per_trade = perf.total_profit / max(perf.wins, 1)
                perf.avg_loss_per_trade = perf.total_loss / max(perf.losses, 1)
                perf.profit_factor = (perf.total_profit / max(perf.total_loss, 0.0001))

            perf.recent_pnl.append(pnl)
            perf.recent_trades.append((is_profit, pnl))

            # 冻结后恢复追踪
            if symbol in self._frozen:
                self._recovery_trades[symbol] = self._recovery_trades.get(symbol, 0) + 1

    # ═══════════════════════════════════════════════════════════════
    # 核心：评估与乘数计算
    # ═══════════════════════════════════════════════════════════════

    def evaluate(self) -> GridUtilizationReport:
        """执行完整评估周期，返回报告。

        每周期调用一次（通常与 rebalance 同频），计算各币种仓位乘数。
        """
        with self._lock:
            report = GridUtilizationReport()
            now = datetime.now()

            active_count = 0
            frozen_count = 0
            total_confidence = 0.0
            total_multiplier = 0.0

            for symbol, perf in list(self._symbols.items()):
                # 检查是否过期
                if perf.last_update_time and (now - perf.last_update_time).total_seconds() > self._stale_minutes * 60:
                    report.warnings.append(f"{symbol}: stale (last update {perf.last_update_time})")
                    continue

                # 初始化乘数
                if symbol not in self._position_multipliers:
                    self._position_multipliers[symbol] = 1.0

                # ── 安全检查 ──
                freeze_reason = self._check_safety(symbol, perf)
                if freeze_reason:
                    self._frozen[symbol] = freeze_reason
                    frozen_count += 1
                    report.warnings.append(f"{symbol}: FROZEN — {freeze_reason}")
                    report.symbols[symbol] = self._build_symbol_report(symbol, perf, 0.0, 1.0)
                    continue

                # 检查是否可以解冻
                if symbol in self._frozen:
                    if self._can_unfreeze(symbol, perf):
                        del self._frozen[symbol]
                        self._recovery_trades.pop(symbol, None)
                        self._reduce_count.pop(symbol, None)
                        report.actions.append(f"{symbol}: UNFROZEN — recovery confirmed")
                    else:
                        frozen_count += 1
                        report.symbols[symbol] = self._build_symbol_report(symbol, perf, 0.0, 1.0)
                        continue

                # ── 计算置信度 ──
                confidence = self._compute_confidence(symbol, perf)
                total_confidence += confidence
                active_count += 1

                # ── 计算目标乘数 ──
                target_multiplier = self._confidence_to_multiplier(confidence)
                current = self._position_multipliers.get(symbol, 1.0)

                # ── 渐进式调整 ──
                new_multiplier = self._gradual_adjust(symbol, current, target_multiplier, perf)
                self._position_multipliers[symbol] = new_multiplier
                total_multiplier += new_multiplier

                report.symbols[symbol] = self._build_symbol_report(symbol, perf, confidence, new_multiplier)

                if new_multiplier != current:
                    direction = "↑" if new_multiplier > current else "↓"
                    report.actions.append(
                        f"{symbol}: {direction} {current:.2f}x → {new_multiplier:.2f}x "
                        f"(confidence={confidence:.1%}, wr={perf.win_rate:.1%})"
                    )

            report.active_symbols = active_count
            report.frozen_symbols = frozen_count
            if active_count > 0:
                report.avg_confidence = total_confidence / active_count
                report.avg_multiplier = total_multiplier / active_count

            self._last_report = report
            self._maybe_save()

            return report

    def _check_safety(self, symbol: str, perf: GridSymbolPerformance) -> str:
        """安全检查：返回冻结原因，空字符串表示安全"""
        if perf.total_trades < self._min_trades_for_activation:
            return f"insufficient trades ({perf.total_trades}/{self._min_trades_for_activation})"

        if perf.win_rate < self._win_rate_floor and perf.total_trades >= self._min_trades_for_confidence:
            return f"win_rate={perf.win_rate:.1%} < floor={self._win_rate_floor:.0%}"

        if perf.profit_factor < self._profit_factor_floor and perf.total_trades >= self._min_trades_for_confidence:
            return f"profit_factor={perf.profit_factor:.2f} < floor={self._profit_factor_floor:.2f}"

        if perf.consecutive_losses >= self._consecutive_loss_freeze:
            return f"consecutive_losses={perf.consecutive_losses} >= {self._consecutive_loss_freeze}"

        if self._reduce_count.get(symbol, 0) >= self._freeze_after_reduce_count:
            return f"reduce_count={self._reduce_count[symbol]} >= {self._freeze_after_reduce_count}"

        return ""

    def _can_unfreeze(self, symbol: str, perf: GridSymbolPerformance) -> bool:
        """检查冻结的币种是否可以解冻"""
        recovery_trades = self._recovery_trades.get(symbol, 0)
        if recovery_trades < self._recovery_required_trades:
            return False
        if perf.win_rate < self._win_rate_floor:
            return False
        if perf.profit_factor < self._profit_factor_floor:
            return False
        if perf.consecutive_losses >= self._consecutive_loss_freeze:
            return False
        return True

    def _compute_confidence(self, symbol: str, perf: GridSymbolPerformance) -> float:
        """计算盈利能力置信度（0-1）。

        综合指标：
          - 胜率 (40%)
          - 盈亏比/利润因子 (30%)
          - 近期PnL趋势 (20%)
          - 连续盈利奖励 (10%)
        """
        if perf.total_trades < self._min_trades_for_activation:
            return 0.5  # 样本不足，中性

        # 1. 胜率评分 (0-1)
        wr_score = min(1.0, max(0.0, (perf.win_rate - 0.30) / 0.40))

        # 2. 盈亏比评分 (0-1)
        pf_score = min(1.0, max(0.0, (perf.profit_factor - 0.50) / 1.50))

        # 3. 近期PnL趋势 (0-1)
        recent = list(perf.recent_pnl)
        if len(recent) >= 5:
            recent_avg = sum(recent) / len(recent)
            # 归一化：-0.5~0.5 USDT → 0, 0~1.0 USDT → 1
            trend_score = min(1.0, max(0.0, (recent_avg + 0.5) / 1.5))
        else:
            trend_score = 0.5

        # 4. 连续盈利奖励
        streak_bonus = min(0.10, perf.consecutive_wins * 0.02)

        confidence = (
            wr_score * 0.40 +
            pf_score * 0.30 +
            trend_score * 0.20 +
            streak_bonus
        )

        return min(1.0, max(0.0, confidence))

    def _confidence_to_multiplier(self, confidence: float) -> float:
        """将置信度映射为仓位乘数。

        映射规则：
          - confidence < 0.40  → multiplier = 0.50 ~ 0.80（降仓区）
          - confidence 0.40-0.60 → multiplier = 0.80 ~ 1.20（中性区）
          - confidence 0.60-0.80 → multiplier = 1.20 ~ 1.80（加仓区）
          - confidence > 0.80  → multiplier = 1.80 ~ 2.50（激进区）

        使用 sigmoid 函数实现平滑过渡。
        """
        if confidence < 0.35:
            # 线性降仓：0.50 ~ 0.80
            return 0.50 + (confidence / 0.35) * 0.30
        elif confidence < 0.60:
            # 中性区：0.80 ~ 1.20
            return 0.80 + ((confidence - 0.35) / 0.25) * 0.40
        elif confidence < 0.80:
            # 加仓区：1.20 ~ 1.80
            return 1.20 + ((confidence - 0.60) / 0.20) * 0.60
        else:
            # 激进区：1.80 ~ 2.50
            t = min(1.0, (confidence - 0.80) / 0.20)
            return 1.80 + t * 0.70

    def _gradual_adjust(self, symbol: str, current: float, target: float,
                        perf: GridSymbolPerformance) -> float:
        """渐进式调节乘数，防止剧烈波动。

        规则：
          - 提升：每次最多 +step_up_limit（0.05），需连续盈利 ≥ 2 才提升
          - 降仓：每次最多 -step_down_limit（0.10），连续亏损 ≥ 3 加速降仓
          - 底线：不超过 [min_position_multiplier, max_position_multiplier]
        """
        diff = target - current

        if diff > 0:
            # 提升需满足条件
            if perf.consecutive_wins < 2:
                return current  # 等待更多盈利确认
            step = min(diff, self._step_up_limit)
        elif diff < 0:
            # 降仓更快响应
            if perf.consecutive_losses >= self._consecutive_loss_reduce:
                step = max(diff, -self._step_down_limit * 2)  # 加速降仓
            else:
                step = max(diff, -self._step_down_limit)
        else:
            return current

        new_multiplier = current + step
        new_multiplier = max(self._min_position_multiplier,
                            min(self._max_position_multiplier, new_multiplier))

        # 追踪降仓次数
        if new_multiplier < current:
            self._reduce_count[symbol] = self._reduce_count.get(symbol, 0) + 1
        else:
            self._reduce_count[symbol] = 0

        return round(new_multiplier, 2)

    def _build_symbol_report(self, symbol: str, perf: GridSymbolPerformance,
                             confidence: float, multiplier: float) -> Dict[str, Any]:
        """构建单个币种的报告"""
        return {
            "symbol": symbol,
            "total_trades": perf.total_trades,
            "wins": perf.wins,
            "losses": perf.losses,
            "total_pnl": round(perf.total_pnl, 4),
            "win_rate": round(perf.win_rate, 4),
            "profit_factor": round(perf.profit_factor, 4),
            "consecutive_wins": perf.consecutive_wins,
            "consecutive_losses": perf.consecutive_losses,
            "confidence": round(confidence, 4),
            "position_multiplier": multiplier,
            "is_frozen": symbol in self._frozen,
            "freeze_reason": self._frozen.get(symbol, ""),
            "last_trade": perf.last_trade_time.isoformat() if perf.last_trade_time else None,
        }

    # ═══════════════════════════════════════════════════════════════
    # 查询接口
    # ═══════════════════════════════════════════════════════════════

    def get_position_multipliers(self) -> Dict[str, float]:
        """获取所有币种的仓位乘数"""
        with self._lock:
            return dict(self._position_multipliers)

    def get_multiplier(self, symbol: str) -> float:
        """获取单个币种的仓位乘数"""
        with self._lock:
            return self._position_multipliers.get(symbol, 1.0)

    def get_last_report(self) -> Optional[GridUtilizationReport]:
        return self._last_report

    def get_active_symbols(self) -> List[str]:
        with self._lock:
            return [s for s in self._symbols if s not in self._frozen]

    def get_frozen_symbols(self) -> Dict[str, str]:
        with self._lock:
            return dict(self._frozen)

    # ═══════════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════════

    def _maybe_save(self):
        now = time.time()
        if now - self._last_save_time > self._save_interval_seconds:
            self._save_state()
            self._last_save_time = now

    def _save_state(self):
        try:
            state = {
                "position_multipliers": self._position_multipliers,
                "frozen": self._frozen,
                "reduce_count": self._reduce_count,
                "recovery_trades": self._recovery_trades,
                "symbols": {
                    s: {
                        "total_trades": p.total_trades,
                        "wins": p.wins, "losses": p.losses,
                        "total_pnl": p.total_pnl,
                        "total_profit": p.total_profit,
                        "total_loss": p.total_loss,
                        "consecutive_wins": p.consecutive_wins,
                        "consecutive_losses": p.consecutive_losses,
                        "win_rate": p.win_rate,
                        "profit_factor": p.profit_factor,
                        "recent_pnl": list(p.recent_pnl),
                        "first_trade_time": p.first_trade_time.isoformat() if p.first_trade_time else None,
                        "last_trade_time": p.last_trade_time.isoformat() if p.last_trade_time else None,
                    }
                    for s, p in self._symbols.items()
                },
                "saved_at": datetime.now().isoformat(),
            }
            os.makedirs(os.path.dirname(self._persist_path), exist_ok=True)
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug(f"GridAdaptiveUtilization: save failed: {e}")

    def _load_state(self):
        try:
            if not os.path.exists(self._persist_path):
                return
            with open(self._persist_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self._position_multipliers = state.get("position_multipliers", {})
            self._frozen = state.get("frozen", {})
            self._reduce_count = state.get("reduce_count", {})
            self._recovery_trades = state.get("recovery_trades", {})
            for s, data in state.get("symbols", {}).items():
                perf = GridSymbolPerformance(symbol=s)
                perf.total_trades = data.get("total_trades", 0)
                perf.wins = data.get("wins", 0)
                perf.losses = data.get("losses", 0)
                perf.total_pnl = data.get("total_pnl", 0)
                perf.total_profit = data.get("total_profit", 0)
                perf.total_loss = data.get("total_loss", 0)
                perf.consecutive_wins = data.get("consecutive_wins", 0)
                perf.consecutive_losses = data.get("consecutive_losses", 0)
                perf.win_rate = data.get("win_rate", 0)
                perf.profit_factor = data.get("profit_factor", 0)
                perf.recent_pnl = deque(data.get("recent_pnl", []), maxlen=20)
                if data.get("first_trade_time"):
                    perf.first_trade_time = datetime.fromisoformat(data["first_trade_time"])
                if data.get("last_trade_time"):
                    perf.last_trade_time = datetime.fromisoformat(data["last_trade_time"])
                self._symbols[s] = perf
            logger.info(
                f"GridAdaptiveUtilization state loaded: "
                f"{len(self._symbols)} symbols, {len(self._frozen)} frozen"
            )
        except Exception as e:
            logger.debug(f"GridAdaptiveUtilization: load failed: {e}")


# ═══════════════════════════════════════════════════════════════
# 单例工厂
# ═══════════════════════════════════════════════════════════════

_grid_adaptive_engine: Optional[GridAdaptiveUtilizationEngine] = None


def get_grid_adaptive_engine(config: Dict[str, Any] = None) -> GridAdaptiveUtilizationEngine:
    """获取全局单例"""
    global _grid_adaptive_engine
    if _grid_adaptive_engine is None:
        _grid_adaptive_engine = GridAdaptiveUtilizationEngine(config)
    return _grid_adaptive_engine


def reset_grid_adaptive_engine():
    """重置单例（测试用）"""
    global _grid_adaptive_engine
    _grid_adaptive_engine = None