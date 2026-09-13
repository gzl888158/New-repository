"""
统一利润锁定梯度引擎 - Profit Lock Engine

职责：解决「盈利时没有落袋、回落时被平仓」导致的资金磨损问题。
原移动止盈激活阈值（浮盈 3%）远高于实际盈利幅度（0.2%~1.5%），导致：
  1. 浮盈永远够不到激活线，止损/反向回落直接吞噬利润；
  2. 行情从峰值回落后，仅靠固定止损被动离场，利润回吐。

本引擎将锁利逻辑收敛为「单一口径」的四级梯度（单向棘轮，只升不降）：

  阶段 0 none        —— 新仓，仅记录价格极值
  阶段 1 breakeven   —— 浮盈 ≥ breakeven_pct：激活保本，回落至成本即平仓锁 0 利润
  阶段 2 partial     —— 浮盈 ≥ partial_pct：部分落袋 partial_ratio，剩余进入紧追踪
  阶段 3 trailing    —— 剩余仓位按紧追踪（回撤 trailing_distance_pct）全平
  旁路   reversal    —— 反转评分 ≥ reversal_close_score 且浮盈时，直接全平锁利

设计原则（与 ReversalTakeProfitEngine / AdaptiveTpSlEngine 一致）：
- 纯计算引擎，不直接下单、不读写数据库（I/O 由调用方 Scheduler 负责）；
- 每个持仓方向维护独立状态（key = symbol:pos_side），含峰价极值、阶段、部分落袋标记、冷却时间戳；
- 单次 compute 可跨越多个梯度（单 tick 浮盈跳变也可正确推进），动作冷却防重复下单；
- 所有配置阈值带 NaN/Inf/范围校验，非法值回退安全默认。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from loguru import logger


@dataclass
class ProfitLockDecision:
    """利润锁定引擎的单次计算输出"""
    symbol: str
    position_side: str            # long / short
    action: str                   # none / partial / full
    partial_ratio: float          # partial 时有效，其余为 0.0
    exit_reason: str              # profit_lock_breakeven / profit_lock_partial / profit_lock_trailing / profit_lock_reversal / ""
    pnl_pct: float
    peak_price: float
    phase: str                    # 决策后的阶段（供审计/调试）
    reversal_score: float
    retrace_pct: float            # 当前从峰值回撤比例（trailing 判定用）
    details: Dict[str, Any] = field(default_factory=dict)


def _f(v, default: float = 0.0) -> float:
    """安全转 float，None/空串/非法值返回默认值。"""
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


def _finite(v: float, default: float) -> float:
    """NaN/Inf 防护。"""
    return v if math.isfinite(v) else default


# ── 阈值硬下限（企业级护栏，防止阈值被误配/学习下调到无效或危险区间）──
_BREAKEVEN_PCT_FLOOR = 0.002      # 保本激活最低 0.2%
_PARTIAL_PCT_FLOOR = 0.005        # 部分落袋最低 0.5%
_TRAILING_DIST_FLOOR = 0.002      # 紧追踪回撤最低 0.2%


class ProfitLockEngine:
    """统一利润锁定梯度引擎（纯计算 + 状态管理，不下单）"""

    # 动作 → 退出原因映射（供下游平仓信号与 trade_records.exit_reason 使用）
    ACTION_REASON = {
        "breakeven": "profit_lock_breakeven",
        "partial": "profit_lock_partial",
        "trailing": "profit_lock_trailing",
        "reversal": "profit_lock_reversal",
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cfg = config.get("profit_lock", {}) or {}

        self._enabled = bool(cfg.get("enabled", True))

        # ── 梯度阈值（稳健风格默认，见 config.yaml profit_lock 段）──
        self._breakeven_pct = _finite(_f(cfg.get("breakeven_pct"), 0.005), 0.005)
        self._breakeven_exit_pct = _finite(_f(cfg.get("breakeven_exit_pct"), 0.0), 0.0)
        self._partial_pct = _finite(_f(cfg.get("partial_pct"), 0.01), 0.01)
        self._partial_ratio = _finite(_f(cfg.get("partial_ratio"), 0.35), 0.35)
        self._trailing_distance_pct = _finite(_f(cfg.get("trailing_distance_pct"), 0.006), 0.006)
        self._reversal_close_score = _finite(_f(cfg.get("reversal_close_score"), 0.5), 0.5)
        self._reversal_min_profit_pct = _finite(_f(cfg.get("reversal_min_profit_pct"), 0.002), 0.002)
        self._cooldown_seconds = _finite(_f(cfg.get("cooldown_seconds"), 120.0), 120.0)

        # ── 阈值范围/硬下限校验（企业级护栏：防止阈值被误配/学习下调到无效或危险区间）──
        # 部分落袋比例必须在 (0.05, 0.95) 之间
        self._partial_ratio = max(0.05, min(0.95, self._partial_ratio))
        # 保本激活阈值硬下限 0.2%（过低会导致无意义高频保本平仓）
        self._breakeven_pct = max(_BREAKEVEN_PCT_FLOOR, self._breakeven_pct)
        # 部分落袋阈值不得低于保本激活阈值（保证梯度有序：先保本、后落袋）
        self._partial_pct = max(self._breakeven_pct, max(_PARTIAL_PCT_FLOOR, self._partial_pct))
        # 紧追踪回撤硬下限 0.2%（低于此值会在正常噪声下误触全平）
        self._trailing_distance_pct = max(_TRAILING_DIST_FLOOR, self._trailing_distance_pct)
        self._reversal_close_score = max(0.0, min(1.0, self._reversal_close_score))
        self._reversal_min_profit_pct = max(0.0, self._reversal_min_profit_pct)
        self._breakeven_exit_pct = max(0.0, self._breakeven_exit_pct)
        self._cooldown_seconds = max(0.0, self._cooldown_seconds)

        # key: f"{symbol}:{pos_side}" -> 阶段状态
        self._states: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"ProfitLockEngine initialized: enabled={self._enabled}, "
            f"breakeven={self._breakeven_pct:.2%}(exit@={self._breakeven_exit_pct:.2%}), "
            f"partial={self._partial_pct:.2%}({self._partial_ratio:.0%}), "
            f"trailing={self._trailing_distance_pct:.2%}, "
            f"reversal={self._reversal_close_score:.2f}(min_pnl={self._reversal_min_profit_pct:.2%}), "
            f"cooldown={self._cooldown_seconds:.0f}s"
        )

    # ==================== 主入口 ====================

    def compute(
        self,
        symbol: str,
        pos_side: str,
        entry_price: float,
        current_price: float,
        reversal_score: float = 0.0,
        breakeven_buffer: float = 0.0,
    ) -> ProfitLockDecision:
        """计算利润锁定动作与阶段推进。

        Args:
            symbol: 交易对
            pos_side: long/short
            entry_price: 持仓均价
            current_price: 当前标记价
            reversal_score: 反转评分 0~1（由调用方 HMM/指标计算，缺省 0）
            breakeven_buffer: 保本缓冲（往返手续费折算），把「回落至成本」的平仓线
                             上移到至少覆盖手续费，实现 fee-aware 保本。缺省 0。
        """
        direction = self._normalize_direction(pos_side)

        # 数据层防护：非法价格/方向 → 直接 none，不推进状态
        try:
            entry_price = float(entry_price)
            current_price = float(current_price)
            reversal_score = float(reversal_score)
            breakeven_buffer = float(breakeven_buffer)
        except (TypeError, ValueError):
            return self._none(symbol, direction, 0.0, 0.0, 0.0, 0.0, "unknown")
        if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(current_price) or current_price <= 0:
            return self._none(symbol, direction, 0.0, 0.0, 0.0, 0.0, "unknown")
        if direction not in ("long", "short"):
            return self._none(symbol, direction, 0.0, 0.0, reversal_score, 0.0, "unknown")
        reversal_score = max(0.0, min(1.0, reversal_score))
        breakeven_buffer = 0.0 if not math.isfinite(breakeven_buffer) or breakeven_buffer < 0 else breakeven_buffer

        pnl_pct = self._pnl_pct(direction, entry_price, current_price)

        key = f"{symbol}:{direction}"
        st = self._states.get(key)
        if st is None:
            st = {"phase": "none", "peak": current_price, "partial_done": False, "last_action_ts": 0.0}
            self._states[key] = st

        # 更新价格极值（long 记最高，short 记最低）
        if direction == "long":
            st["peak"] = max(float(st["peak"]), current_price)
        else:
            st["peak"] = min(float(st["peak"]), current_price)

        peak = float(st["peak"])
        retrace_pct = self._retrace_pct(direction, peak, current_price)
        phase = str(st["phase"])

        # ── 旁路：反转落袋（最高优先级，盈利且反转信号强）──
        if pnl_pct >= self._reversal_min_profit_pct and reversal_score >= self._reversal_close_score:
            if self._cooldown_ok(st):
                st["last_action_ts"] = time.time()
                st["phase"] = "trailing"  # 全平后仓位将消失，此处标记仅作审计
                return self._decision(
                    symbol, direction, "full", 0.0, "reversal",
                    pnl_pct, peak, retrace_pct, reversal_score, st
                )
            # 冷却中不重复触发，继续走下方逻辑（可能落入 none）

        # ── 阶段 1：保本激活 ──
        if phase == "none" and pnl_pct >= self._breakeven_pct:
            st["phase"] = "breakeven"
            phase = "breakeven"
            logger.info(
                f"[PROFIT_LOCK] breakeven armed: {symbol} {direction} pnl={pnl_pct:.2%}"
            )

        # ── 阶段 2：部分落袋 ──
        if phase == "breakeven" and not st["partial_done"] and pnl_pct >= self._partial_pct:
            if self._cooldown_ok(st):
                st["partial_done"] = True
                st["phase"] = "trailing"
                st["peak"] = current_price  # 部分落袋后重置紧追踪峰值
                st["last_action_ts"] = time.time()
                return self._decision(
                    symbol, direction, "partial", self._partial_ratio, "partial",
                    pnl_pct, current_price, 0.0, reversal_score, st
                )

        # ── 保本保护：激活后回落至「成本 + 手续费缓冲」→ 平仓，fee-aware 锁利 ──
        exit_pct = max(self._breakeven_exit_pct, breakeven_buffer)
        if phase == "breakeven" and pnl_pct <= exit_pct:
            if self._cooldown_ok(st):
                st["last_action_ts"] = time.time()
                return self._decision(
                    symbol, direction, "full", 0.0, "breakeven",
                    pnl_pct, peak, retrace_pct, reversal_score, st
                )

        # ── 阶段 3：紧追踪 ──
        if phase == "trailing" and retrace_pct >= self._trailing_distance_pct:
            if self._cooldown_ok(st):
                st["last_action_ts"] = time.time()
                return self._decision(
                    symbol, direction, "full", 0.0, "trailing",
                    pnl_pct, peak, retrace_pct, reversal_score, st
                )

        return self._decision(
            symbol, direction, "none", 0.0, "",
            pnl_pct, peak, retrace_pct, reversal_score, st
        )

    # ==================== 状态管理 ====================

    def prune(self, active_keys: set) -> None:
        """清理已平仓/消失持仓的状态，防止残留导致误判。"""
        for key in list(self._states.keys()):
            if key not in active_keys:
                self._states.pop(key, None)

    def reset(self) -> None:
        """清空全部状态（重启/对账重置用）。"""
        self._states.clear()

    def reset_position(self, symbol: str, pos_side: str) -> None:
        """重置某个持仓方向的锁利状态（例如交易所已无该方向仓位时的 51169 防护）。"""
        direction = self._normalize_direction(pos_side)
        self._states.pop(f"{symbol}:{direction}", None)

    def get_state(self, symbol: str, pos_side: str) -> Optional[Dict[str, Any]]:
        """读取某持仓方向的锁利阶段状态（供 Dashboard/调试）。"""
        direction = self._normalize_direction(pos_side)
        st = self._states.get(f"{symbol}:{direction}")
        if st is None:
            return None
        return {
            "phase": st.get("phase"),
            "peak": st.get("peak"),
            "partial_done": st.get("partial_done", False),
            "last_action_ts": st.get("last_action_ts", 0.0),
        }

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """返回全部持仓方向的锁利状态快照。"""
        return {k: dict(v) for k, v in self._states.items()}

    # ==================== 工具方法 ====================

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _cooldown_ok(self, st: Dict[str, Any]) -> bool:
        last = float(st.get("last_action_ts", 0.0) or 0.0)
        return (time.time() - last) >= self._cooldown_seconds

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        d = str(direction or "").lower()
        if d in ("buy", "long"):
            return "long"
        if d in ("sell", "short"):
            return "short"
        return d

    @staticmethod
    def _pnl_pct(direction: str, entry_price: float, current_price: float) -> float:
        if entry_price <= 0:
            return 0.0
        if direction == "long":
            return (current_price - entry_price) / entry_price
        return (entry_price - current_price) / entry_price

    @staticmethod
    def _retrace_pct(direction: str, peak: float, current_price: float) -> float:
        """从峰值的回撤比例（非负）。"""
        if peak <= 0:
            return 0.0
        if direction == "long":
            return max(0.0, (peak - current_price) / peak)
        return max(0.0, (current_price - peak) / peak)

    def _decision(
        self,
        symbol: str,
        direction: str,
        action: str,
        partial_ratio: float,
        reason_key: str,
        pnl_pct: float,
        peak: float,
        retrace_pct: float,
        reversal_score: float,
        st: Dict[str, Any],
    ) -> ProfitLockDecision:
        return ProfitLockDecision(
            symbol=symbol,
            position_side=direction,
            action=action,
            partial_ratio=round(partial_ratio, 4),
            exit_reason=self.ACTION_REASON.get(reason_key, ""),
            pnl_pct=round(pnl_pct, 6),
            peak_price=round(peak, 10),
            phase=str(st["phase"]),
            reversal_score=round(reversal_score, 4),
            retrace_pct=round(retrace_pct, 6),
            details={
                "breakeven_pct": self._breakeven_pct,
                "partial_pct": self._partial_pct,
                "trailing_distance_pct": self._trailing_distance_pct,
                "reversal_close_score": self._reversal_close_score,
            },
        )

    def _none(
        self,
        symbol: str,
        direction: str,
        pnl_pct: float,
        peak: float,
        retrace_pct: float,
        reversal_score: float,
        phase: str,
    ) -> ProfitLockDecision:
        return ProfitLockDecision(
            symbol=symbol,
            position_side=direction,
            action="none",
            partial_ratio=0.0,
            exit_reason="",
            pnl_pct=round(pnl_pct, 6),
            peak_price=round(peak, 10),
            phase=phase,
            reversal_score=round(reversal_score, 4),
            retrace_pct=round(retrace_pct, 6),
            details={},
        )