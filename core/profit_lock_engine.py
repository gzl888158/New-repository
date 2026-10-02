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

反转落袋（HMM/指标反转评分）不在本引擎内处理，统一由 ReversalTakeProfitEngine 负责，
避免同一反转评分被两个引擎以不同阈值重复/过早平仓（S1 收敛）。

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
    exit_reason: str              # profit_lock_breakeven / profit_lock_partial / profit_lock_trailing / ""
    pnl_pct: float
    peak_price: float
    phase: str                    # 决策后的阶段（供审计/调试）
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
# 下调以适配微利行情：保本激活 0.1%（覆盖往返手续费约 0.1%~0.15%），
# 部分落袋 0.3%，紧追踪回撤 0.15%。仍高于手续费缓冲，避免无意义锁利。
_BREAKEVEN_PCT_FLOOR = 0.001      # 保本激活最低 0.1%
_PARTIAL_PCT_FLOOR = 0.003        # 部分落袋最低 0.3%
_TRAILING_DIST_FLOOR = 0.0015     # 紧追踪回撤最低 0.15%


class ProfitLockEngine:
    """统一利润锁定梯度引擎（纯计算 + 状态管理，不下单）"""

    # 动作 → 退出原因映射（供下游平仓信号与 trade_records.exit_reason 使用）
    ACTION_REASON = {
        "breakeven": "profit_lock_breakeven",
        "partial": "profit_lock_partial",
        "trailing": "profit_lock_trailing",
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
        self._cooldown_seconds = _finite(_f(cfg.get("cooldown_seconds"), 120.0), 120.0)

        # ── 保本阶段快速回撤落袋（增强收益落袋：捕捉 0.1%~0.4% 的小利润，
        #    避免其在到达 partial 线前回吐为亏损）──
        # 进入 breakeven 阶段后，若价格从峰值回撤 >= trailing_distance_pct，
        # 且当前仍处于「扣除手续费后仍盈利」区间，则立即平掉 breakeven_retrace_partial_ratio 仓位锁利。
        self._breakeven_retrace_partial_enabled = bool(cfg.get("breakeven_retrace_partial_enabled", True))
        self._breakeven_retrace_partial_ratio = _finite(_f(cfg.get("breakeven_retrace_partial_ratio"), 0.4), 0.4)
        self._breakeven_retrace_partial_ratio = max(0.05, min(0.8, self._breakeven_retrace_partial_ratio))

        # ── 阈值范围/硬下限校验（企业级护栏：防止阈值被误配/学习下调到无效或危险区间）──
        # 部分落袋比例必须在 (0.05, 0.95) 之间
        self._partial_ratio = max(0.05, min(0.95, self._partial_ratio))
        # 保本激活阈值硬下限 0.2%（过低会导致无意义高频保本平仓）
        self._breakeven_pct = max(_BREAKEVEN_PCT_FLOOR, self._breakeven_pct)
        # 部分落袋阈值不得低于保本激活阈值（保证梯度有序：先保本、后落袋）
        self._partial_pct = max(self._breakeven_pct, max(_PARTIAL_PCT_FLOOR, self._partial_pct))
        # 紧追踪回撤硬下限 0.2%（低于此值会在正常噪声下误触全平）
        self._trailing_distance_pct = max(_TRAILING_DIST_FLOOR, self._trailing_distance_pct)
        self._breakeven_exit_pct = max(0.0, self._breakeven_exit_pct)
        self._cooldown_seconds = max(0.0, self._cooldown_seconds)

        # key: f"{symbol}:{pos_side}" -> 阶段状态
        self._states: Dict[str, Dict[str, Any]] = {}

        logger.info(
            f"ProfitLockEngine initialized: enabled={self._enabled}, "
            f"breakeven={self._breakeven_pct:.2%}(exit@={self._breakeven_exit_pct:.2%}), "
            f"partial={self._partial_pct:.2%}({self._partial_ratio:.0%}), "
            f"trailing={self._trailing_distance_pct:.2%}, "
            f"be_retrace_partial={'on' if self._breakeven_retrace_partial_enabled else 'off'}"
            f"({self._breakeven_retrace_partial_ratio:.0%}), "
            f"cooldown={self._cooldown_seconds:.0f}s"
        )

    # ==================== 主入口 ====================

    def compute(
        self,
        symbol: str,
        pos_side: str,
        entry_price: float,
        current_price: float,
        breakeven_buffer: float = 0.0,
        sl_trailing_active: bool = False,
    ) -> ProfitLockDecision:
        """计算利润锁定动作与阶段推进。

        Args:
            symbol: 交易对
            pos_side: long/short
            entry_price: 持仓均价
            current_price: 当前标记价
            breakeven_buffer: 保本缓冲（往返手续费折算），把「回落至成本」的平仓线
                             上移到至少覆盖手续费，实现 fee-aware 保本。缺省 0。
            sl_trailing_active: 止损侧（保命移动止损）是否已进入 trailing 保护态。
                               为 True 时，本引擎的紧追踪全平（阶段3 落袋）让位，
                               由止损侧统一保护，避免双信号重复平仓（S4 止损优先仲裁）。
        """
        direction = self._normalize_direction(pos_side)

        # 数据层防护：非法价格/方向 → 直接 none，不推进状态
        try:
            entry_price = float(entry_price)
            current_price = float(current_price)
            breakeven_buffer = float(breakeven_buffer)
        except (TypeError, ValueError):
            return self._none(symbol, direction, 0.0, 0.0, 0.0, "unknown")
        if not math.isfinite(entry_price) or entry_price <= 0 or not math.isfinite(current_price) or current_price <= 0:
            return self._none(symbol, direction, 0.0, 0.0, 0.0, "unknown")
        if direction not in ("long", "short"):
            return self._none(symbol, direction, 0.0, 0.0, 0.0, "unknown")
        breakeven_buffer = 0.0 if not math.isfinite(breakeven_buffer) or breakeven_buffer < 0 else breakeven_buffer

        pnl_pct = self._pnl_pct(direction, entry_price, current_price)

        key = f"{symbol}:{direction}"
        st = self._states.get(key)
        if st is None:
            st = {
                "phase": "none", "peak": current_price, "partial_done": False,
                "be_retrace_partial_done": False, "last_action_ts": 0.0,
            }
            self._states[key] = st

        # 更新价格极值（long 记最高，short 记最低）
        if direction == "long":
            st["peak"] = max(float(st["peak"]), current_price)
        else:
            st["peak"] = min(float(st["peak"]), current_price)

        peak = float(st["peak"])
        retrace_pct = self._retrace_pct(direction, peak, current_price)
        phase = str(st["phase"])

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
                    pnl_pct, current_price, 0.0, st
                )

        # ── 保本保护：激活后回落至「成本 + 手续费缓冲」→ 平仓，fee-aware 锁利 ──
        exit_pct = max(self._breakeven_exit_pct, breakeven_buffer)
        if phase == "breakeven" and pnl_pct <= exit_pct:
            if self._cooldown_ok(st):
                st["last_action_ts"] = time.time()
                return self._decision(
                    symbol, direction, "full", 0.0, "breakeven",
                    pnl_pct, peak, retrace_pct, st
                )

        # ── 保本阶段回撤快速部分落袋（增强收益落袋）──
        # 已进入 breakeven 但尚未到 partial 线时，若从峰值回撤 >= trailing_distance_pct，
        # 且扣除手续费后仍盈利（pnl_pct > exit_pct），立即平掉部分仓位锁利，
        # 剩余仓位进入 trailing 保护，防止小利润回吐为亏损。
        if (
            phase == "breakeven"
            and not st.get("be_retrace_partial_done", False)
            and self._breakeven_retrace_partial_enabled
            and pnl_pct > exit_pct
            and retrace_pct >= self._trailing_distance_pct
        ):
            if self._cooldown_ok(st):
                st["be_retrace_partial_done"] = True
                st["phase"] = "trailing"
                st["last_action_ts"] = time.time()
                logger.info(
                    f"[PROFIT_LOCK] breakeven retrace partial: {symbol} {direction} "
                    f"pnl={pnl_pct:.2%} retrace={retrace_pct:.2%} "
                    f"ratio={self._breakeven_retrace_partial_ratio:.0%}"
                )
                return self._decision(
                    symbol, direction, "partial", self._breakeven_retrace_partial_ratio,
                    "partial", pnl_pct, peak, retrace_pct, st
                )

        # ── 阶段 3：紧追踪 ──
        if phase == "trailing" and retrace_pct >= self._trailing_distance_pct:
            # S4: 止损优先仲裁 — 止损侧（保命移动止损）已进入 trailing 保护时，
            # 让位给止损侧统一保护，避免同一持仓被两条 trailing 路径先后平仓。
            if sl_trailing_active:
                logger.debug(
                    f"[PROFIT_LOCK] trailing skipped (SL trailing active): {symbol} {direction} "
                    f"retrace={retrace_pct:.2%}"
                )
            elif self._cooldown_ok(st):
                st["last_action_ts"] = time.time()
                return self._decision(
                    symbol, direction, "full", 0.0, "trailing",
                    pnl_pct, peak, retrace_pct, st
                )

        return self._decision(
            symbol, direction, "none", 0.0, "",
            pnl_pct, peak, retrace_pct, st
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
            "be_retrace_partial_done": st.get("be_retrace_partial_done", False),
            "last_action_ts": st.get("last_action_ts", 0.0),
        }

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        """返回全部持仓方向的锁利状态快照。"""
        return {k: dict(v) for k, v in self._states.items()}

    def dump_state(self) -> Dict[str, Dict[str, Any]]:
        """导出全部锁利梯度状态（JSON 可序列化），供跨重启持久化。"""
        return {k: dict(v) for k, v in self._states.items()}

    def restore_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """从持久化快照恢复锁利梯度状态（跨重启续用，含峰价/阶段/冷却时间戳）。

        只恢复 key 形如 `symbol:long/short` 的合法条目；字段缺失/非法值回退安全默认，
        避免脏数据把引擎带入危险状态（例如 peak=NaN 导致回撤计算失效）。
        """
        if not isinstance(state, dict):
            return
        valid_phases = {"none", "breakeven", "trailing"}
        restored = 0
        for key, value in state.items():
            if not isinstance(value, dict) or ":" not in key:
                continue
            symbol, direction = key.rsplit(":", 1)
            if not symbol or direction not in ("long", "short"):
                continue
            phase = value.get("phase")
            if phase not in valid_phases:
                phase = "none"
            self._states[key] = {
                "phase": phase,
                "peak": _finite(_f(value.get("peak"), 0.0), 0.0),
                "partial_done": bool(value.get("partial_done", False)),
                "be_retrace_partial_done": bool(value.get("be_retrace_partial_done", False)),
                "last_action_ts": _finite(_f(value.get("last_action_ts"), 0.0), 0.0),
            }
            restored += 1
        if restored > 0:
            logger.info(f"ProfitLockEngine restored {restored} persisted lock state(s)")

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
            retrace_pct=round(retrace_pct, 6),
            details={
                "breakeven_pct": self._breakeven_pct,
                "partial_pct": self._partial_pct,
                "trailing_distance_pct": self._trailing_distance_pct,
            },
        )

    def _none(
        self,
        symbol: str,
        direction: str,
        pnl_pct: float,
        peak: float,
        retrace_pct: float,
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
            retrace_pct=round(retrace_pct, 6),
            details={},
        )