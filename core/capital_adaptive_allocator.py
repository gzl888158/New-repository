"""
资金自适应自动分配引擎 - Capital Adaptive Allocator
====================================================

针对极小额账户（nano tier，如 <50 USDT）的「专一分配」引擎，把资金做三层收敛：

    总资金 → 单一策略 → 单一标的 → 单一持仓

背景：手动开仓导致爆仓、账户资金所剩无几（如仅 36 USDT）时，若继续沿用
配置里的静态 total_capital（如 667.5）并分散到多策略/多标的，会产生两个致命问题：
  1. 权益失真 → 名义价值/保证金按错误基数计算，单笔即可能超杠杆爆仓；
  2. 资金碎片化 → 36 USDT 被 4 策略 × 5 标的切碎，触发大量 min_notional/保证金摩擦。

本引擎职责（纯决策，不下单）：
  · 接入真实账户权益（来自 AccountManager，替代过时 config total_capital）
  · 检测账户档位（nano < 50 USDT），低于阈值自动进入「专一模式」
  · 专一模式：全资金收敛到 单一策略 + 单一标的 + 单一持仓
  · 非专一模式（资金充足）：回退到 config 的原多策略分配，不影响既有行为

风控强度遵循「跟随现有配置」：本引擎不额外收紧杠杆/风险比例，
复用 EquityMonitor 的 nano tier 1.5x boost 与 tier 5x 杠杆上限，
仅新增「专一」结构收敛，防止资金碎片化带来的被动爆仓。
"""

from typing import Dict, Any, Optional, Tuple

from loguru import logger


def _f(v, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (ValueError, TypeError):
        return default


def _base_symbol(symbol: str) -> str:
    """归一化交易对为裸币名：ETH-USDT-SWAP / ETH-USDT / ETH → ETH。"""
    if not symbol:
        return ""
    return symbol.strip().split("-")[0].upper()


class CapitalAdaptiveAllocator:
    """资金自适应自动分配引擎（单一标的 + 单一策略 + 单持仓）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 account_manager=None):
        self.config = config or {}
        cfg = self.config.get("focused_allocation", {}) or {}

        self._enabled = bool(cfg.get("enabled", True))
        self._nano_equity_threshold = _f(cfg.get("nano_equity_threshold"), 50.0)
        self._focus_strategy = (cfg.get("focus_strategy") or "auto").strip().lower()
        self._focus_symbol = (cfg.get("focus_symbol") or "auto").strip().upper()
        self._single_position = bool(cfg.get("single_position", True))

        # auto 时的优先级（确定性兜底，避免随机选择）
        self._strategy_priority = [str(s).strip().lower() for s in cfg.get("strategy_priority") or []]
        if not self._strategy_priority:
            self._strategy_priority = ["scalping", "grid", "trend", "arbitrage"]
        self._symbol_priority = [str(s).strip().upper() for s in cfg.get("symbol_priority") or []]
        if not self._symbol_priority:
            self._symbol_priority = ["ETH", "SOL", "XRP", "DOGE", "SUI"]

        self._account_manager = account_manager

        # 缓存：聚焦标的/策略按需懒计算，权益变化时自然刷新
        self._cached_focus: Optional[Tuple[str, str]] = None

        logger.info(
            f"CapitalAdaptiveAllocator initialized: enabled={self._enabled}, "
            f"nano_threshold={self._nano_equity_threshold:.0f}, "
            f"focus_strategy={self._focus_strategy}, focus_symbol={self._focus_symbol}, "
            f"single_position={self._single_position}"
        )

    # ─────────────────────────────────────────────────────────────
    # 权益接入（真实权益替代 config total_capital）
    # ─────────────────────────────────────────────────────────────

    def get_equity(self) -> float:
        """返回真实账户权益（USDT），优先 AccountManager，回退 config total_capital。"""
        if self._account_manager is not None:
            try:
                eq = self._account_manager.get_total_equity()
                if eq and eq > 0:
                    return float(eq)
            except Exception:
                pass
        return _f(self.config.get("trading", {}).get("total_capital"), 0.0)

    def get_total_capital(self) -> float:
        return self.get_equity()

    # ─────────────────────────────────────────────────────────────
    # 专一模式判定
    # ─────────────────────────────────────────────────────────────

    def is_focused_mode(self) -> bool:
        """低于 nano 阈值进入专一模式。"""
        if not self._enabled:
            return False
        equity = self.get_equity()
        return 0 < equity <= self._nano_equity_threshold

    # ─────────────────────────────────────────────────────────────
    # 聚焦标的 / 策略选择（确定性）
    # ─────────────────────────────────────────────────────────────

    def _enabled_strategies(self):
        strategies = self.config.get("strategies", {}) or {}
        result = []
        for s in self._strategy_priority:
            sec = strategies.get(s) or {}
            if sec.get("enabled", True):
                result.append(s)
        return result or list(self._strategy_priority)

    def _configured_symbols(self):
        currencies = self.config.get("currencies", {}) or {}
        symbols = []
        for tier in ("tier1", "tier2", "tier3"):
            symbols.extend(currencies.get(f"{tier}_symbols", []) or [])
        return [str(s).strip().upper() for s in symbols if s]

    def get_focused_strategy(self) -> str:
        if self._focus_strategy and self._focus_strategy != "auto":
            return self._focus_strategy
        enabled = self._enabled_strategies()
        return enabled[0] if enabled else "grid"

    def get_focused_symbol(self) -> str:
        if self._focus_symbol and self._focus_symbol != "AUTO":
            return self._focus_symbol
        configured = self._configured_symbols()
        # 按优先级取第一个已配置的标的（确定性，默认 tier1 优先）
        for s in self._symbol_priority:
            if s in configured or not configured:
                return s
        return configured[0] if configured else "ETH"

    def get_focus(self) -> Tuple[str, str]:
        """返回 (focused_strategy, focused_symbol)。"""
        if self._cached_focus is None:
            self._cached_focus = (self.get_focused_strategy(), self.get_focused_symbol())
        return self._cached_focus

    # ─────────────────────────────────────────────────────────────
    # 分配与门控（下单链路接入点）
    # ─────────────────────────────────────────────────────────────

    def get_strategy_allocation(self, strategy_name: str) -> float:
        """专一模式下：聚焦策略=1.0，其余=0.0；非专一模式回退 config 原分配。"""
        strategy_name = (strategy_name or "").strip().lower()
        if self.is_focused_mode():
            return 1.0 if strategy_name == self.get_focused_strategy() else 0.0
        alloc = self.config.get("trading", {}).get(f"{strategy_name}_allocation", 0.0)
        return max(0.0, min(1.0, _f(alloc, 0.0)))

    def evaluate_open(self, symbol: str, strategy_name: str,
                      open_position_count: int = 0) -> Tuple[bool, str]:
        """专一模式硬门控：判断是否允许开新仓。

        返回 (allowed, reason)。平仓/减仓不受此门控限制（调用方自行分流）。
        非专一模式恒放行，保持既有行为不变。
        """
        if not self.is_focused_mode():
            return True, ""

        base = _base_symbol(symbol)
        strategy_name = (strategy_name or "").strip().lower()
        focused_symbol = self.get_focused_symbol()
        focused_strategy = self.get_focused_strategy()

        if base and focused_symbol and base != focused_symbol:
            return False, "symbol_not_focused"
        if strategy_name and focused_strategy and strategy_name != focused_strategy:
            return False, "strategy_not_focused"
        if self._single_position and open_position_count >= 1:
            return False, "single_position_limit"

        return True, ""

    # ─────────────────────────────────────────────────────────────
    # 摘要
    # ─────────────────────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        return {
            "enabled": self._enabled,
            "focused_mode": self.is_focused_mode(),
            "equity": self.get_equity(),
            "nano_equity_threshold": self._nano_equity_threshold,
            "focused_strategy": self.get_focused_strategy(),
            "focused_symbol": self.get_focused_symbol(),
            "single_position": self._single_position,
        }
