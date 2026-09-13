"""
行情反转智能计算落袋自适应引擎 - Reversal Take Profit Engine

功能：
1. 反转风险评估（HMM MarketRegimeDetector 为主 + 轻量技术指标兜底）
2. 动态止盈价（反转风险升高时自适应收紧止盈，提前落袋）
3. 反转落袋动作（检测到明确反转时输出部分/全部平仓建议）

设计原则：
- 纯计算引擎，不直接执行订单、不读写数据库（I/O 由调用方 StopLossManager 负责）
- 输入行情/持仓数据，输出结构化 ReversalTakeProfitResult
- HMM 检测器缺失或数据不足时，自动回退到 EMA/RSI 轻量指标
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class ReversalTakeProfitResult:
    """反转落袋引擎计算输出"""
    symbol: str
    strategy_name: str
    # 反转风险评分 0~1，越高越接近反转
    reversal_score: float
    # 反转信号来源：hmm / indicator / hmm+indicator / none
    reversal_source: str
    # 是否识别为 HMM REVERSAL 主状态
    hmm_reversal: bool
    # 动态止盈价（已按反转风险收紧；无 base_tp 时为 None）
    adaptive_tp_price: Optional[float]
    # 原始止盈价（未收紧）
    base_tp_price: Optional[float]
    # 收紧系数：1.0 = 不收紧，越小越收紧
    tp_tighten_factor: float
    # 落袋动作：none / partial / close
    exit_action: str
    # 部分平仓比例（partial 时有效）
    partial_ratio: float
    # 落袋原因
    exit_reason: str
    # 早期预警/指标触发信号文本
    warnings: List[str] = field(default_factory=list)
    # 额外明细
    details: Dict[str, Any] = field(default_factory=dict)


class ReversalTakeProfitEngine:
    """行情反转智能计算落袋自适应引擎"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        cfg = config.get("reversal_take_profit", {}) or {}

        self._enabled = bool(cfg.get("enabled", True))

        # ── HMM 反转检测 ──
        self._hmm_reversal_regime = str(cfg.get("hmm_reversal_regime", "reversal")).lower()

        # ── 指标兜底 ──
        self._fallback_indicators = bool(cfg.get("fallback_indicators", True))
        self._ema_fast = int(cfg.get("ema_fast", 9))
        self._ema_slow = int(cfg.get("ema_slow", 21))
        self._rsi_period = int(cfg.get("rsi_period", 14))
        self._rsi_overbought = float(cfg.get("rsi_overbought", 70.0))
        self._rsi_oversold = float(cfg.get("rsi_oversold", 30.0))

        # ── 动态止盈收紧 ──
        self._tighten_start_score = float(cfg.get("tighten_start_score", 0.40))
        self._tighten_max_factor = float(cfg.get("tighten_max_factor", 0.50))

        # ── 反转落袋 ──
        self._partial_exit_score = float(cfg.get("partial_exit_score", 0.65))
        self._partial_exit_ratio = float(cfg.get("partial_exit_ratio", 0.50))
        self._full_exit_score = float(cfg.get("full_exit_score", 0.85))
        self._min_profit_lock_pct = float(cfg.get("min_profit_lock_pct", 0.002))

        # ── 防重复落袋冷却 ──
        self._cooldown_seconds = float(cfg.get("cooldown_seconds", 300.0))
        self._last_action_ts: Dict[str, float] = {}

        self._precision = int(cfg.get("price_precision", 4))

        logger.info(
            f"ReversalTakeProfitEngine initialized: enabled={self._enabled}, "
            f"tighten=[{self._tighten_start_score:.2f}->{self._tighten_max_factor:.2f}], "
            f"partial={self._partial_exit_score:.2f}({self._partial_exit_ratio:.0%}), "
            f"full={self._full_exit_score:.2f}, cooldown={self._cooldown_seconds:.0f}s"
        )

    # ==================== 主入口 ====================

    def compute(
        self,
        symbol: str,
        strategy_name: str,
        direction: str,
        entry_price: float,
        current_price: float,
        base_tp_price: Optional[float] = None,
        hmm_result: Optional[Dict[str, Any]] = None,
        ohlcv_data: Optional[List[Any]] = None,
        atr: Optional[float] = None,
    ) -> ReversalTakeProfitResult:
        """计算反转风险、动态止盈价与落袋动作。

        Args:
            symbol: 交易对
            strategy_name: 策略名
            direction: long/short（自动归一化）
            entry_price: 开仓价
            current_price: 当前价
            base_tp_price: 策略原始止盈价（可选）
            hmm_result: MarketRegimeDetector.get_regime(symbol) 或 detect_regime() 的结果
            ohlcv_data: K线数据（指标兜底用），支持 list[list] 或 list[dict]
            atr: ATR 值（可选，用于兜底止盈参考）
        """
        direction = self._normalize_direction(direction)
        pnl_pct = self._pnl_pct(direction, entry_price, current_price)

        details: Dict[str, Any] = {
            "direction": direction,
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
        }

        # 1. 反转评分（HMM 为主 + 指标兜底）
        hmm_score, hmm_warnings, hmm_reversal = self._reversal_score_from_hmm(hmm_result)
        ind_score, ind_warnings = self._reversal_score_from_indicators(ohlcv_data, direction)

        score = hmm_score
        source_parts: List[str] = []
        if hmm_reversal or hmm_score > 0:
            source_parts.append("hmm")
        if self._fallback_indicators and ind_score > 0:
            # 兜底：HMM 未给出有效信号时以指标为主，否则取两者较大值
            score = max(score, ind_score)
            if ind_score >= hmm_score:
                source_parts.append("indicator")
        if not source_parts:
            source_parts.append("none")

        score = max(0.0, min(1.0, score))
        source = "+".join(sorted(set(source_parts)))
        warnings = hmm_warnings + ind_warnings
        details["hmm_score"] = hmm_score
        details["indicator_score"] = ind_score

        # 2. 动态止盈价
        adaptive_tp, tighten_factor = self._compute_adaptive_tp(
            direction, entry_price, base_tp_price, score
        )

        # 3. 落袋动作（含冷却去重）
        exit_action, partial_ratio, exit_reason = self._compute_exit_action(
            symbol, strategy_name, score, pnl_pct
        )

        return ReversalTakeProfitResult(
            symbol=symbol,
            strategy_name=strategy_name,
            reversal_score=round(score, 4),
            reversal_source=source,
            hmm_reversal=hmm_reversal,
            adaptive_tp_price=adaptive_tp,
            base_tp_price=base_tp_price,
            tp_tighten_factor=round(tighten_factor, 4),
            exit_action=exit_action,
            partial_ratio=partial_ratio,
            exit_reason=exit_reason,
            warnings=warnings,
            details=details,
        )

    def reversal_score(
        self,
        direction: str,
        hmm_result: Optional[Dict[str, Any]] = None,
        ohlcv_data: Optional[List[Any]] = None,
    ) -> float:
        """仅计算反转评分 0~1（不触发落袋动作/冷却，供 ProfitLockEngine 等旁路复用）。

        与 compute() 复用同一套 HMM + 指标兜底评分逻辑，但不去写 _last_action_ts，
        避免污染 check_reversal_take_profit 的落袋冷却状态。
        """
        try:
            direction = self._normalize_direction(direction)
            hmm_score, _, _ = self._reversal_score_from_hmm(hmm_result)
            ind_score, _ = self._reversal_score_from_indicators(ohlcv_data, direction)

            score = hmm_score
            if self._fallback_indicators and ind_score > 0:
                score = max(score, ind_score)
                if ind_score >= hmm_score:
                    score = ind_score
            return max(0.0, min(1.0, score))
        except Exception as e:
            logger.debug(f"reversal_score compute error: {e}")
            return 0.0

    # ==================== 反转评分：HMM ====================

    def _reversal_score_from_hmm(self, hmm_result: Optional[Dict[str, Any]]) -> Tuple[float, List[str], bool]:
        """从 HMM MarketRegimeDetector 结果计算反转评分。

        Returns: (score, warnings, is_reversal_regime)
        """
        if not hmm_result or not isinstance(hmm_result, dict):
            return 0.0, [], False

        regime = str(hmm_result.get("regime", "")).lower()
        probs = hmm_result.get("probabilities", {}) or {}
        early_warnings = hmm_result.get("early_warnings", []) or []

        is_reversal = (regime == self._hmm_reversal_regime)

        # 反转概率：主状态为 reversal 给基础分，再从概率分布中提取 reversal 概率
        score = 0.80 if is_reversal else 0.0
        reversal_prob = 0.0
        if isinstance(probs, dict):
            for k, v in probs.items():
                key = str(k).lower().replace("marketregime.", "")
                if key == "reversal":
                    try:
                        reversal_prob = float(v)
                    except (TypeError, ValueError):
                        reversal_prob = 0.0
                    break
        if reversal_prob > 0:
            score = max(score, min(1.0, reversal_prob))

        warnings: List[str] = []
        if isinstance(early_warnings, list):
            for w in early_warnings:
                if isinstance(w, dict):
                    wtype = w.get("type") or w.get("warning") or str(w)
                else:
                    wtype = str(w)
                warnings.append(f"hmm:{wtype}")
                score = min(1.0, score + 0.05)

        return score, warnings, is_reversal

    # ==================== 反转评分：指标兜底 ====================

    def _reversal_score_from_indicators(self, ohlcv_data: Optional[List[Any]], direction: str) -> Tuple[float, List[str]]:
        """轻量技术指标兜底：EMA 交叉 + RSI 超买超卖 + 价格结构破坏。

        Returns: (score, warnings)
        """
        if not self._fallback_indicators or not ohlcv_data:
            return 0.0, []

        closes = self._extract_closes(ohlcv_data)
        need = max(self._ema_slow, self._rsi_period) + 2
        if len(closes) < need:
            return 0.0, []

        score = 0.0
        warnings: List[str] = []

        # EMA 交叉
        ema_fast = self._ema(closes, self._ema_fast)
        ema_slow = self._ema(closes, self._ema_slow)
        prev_fast, prev_slow = ema_fast[-2], ema_slow[-2]
        last_fast, last_slow = ema_fast[-1], ema_slow[-1]
        if direction == "long" and last_fast < last_slow and prev_fast >= prev_slow:
            score += 0.35
            warnings.append("ema_bearish_cross")
        elif direction == "short" and last_fast > last_slow and prev_fast <= prev_slow:
            score += 0.35
            warnings.append("ema_bullish_cross")

        # RSI 超买/超卖
        rsi = self._rsi(closes, self._rsi_period)
        if direction == "long" and rsi[-1] > self._rsi_overbought:
            score += 0.30
            warnings.append("rsi_overbought")
        elif direction == "short" and rsi[-1] < self._rsi_oversold:
            score += 0.30
            warnings.append("rsi_oversold")

        # 价格结构破坏：最近收盘跌破/涨破短期结构
        if len(closes) >= self._ema_slow:
            recent_high = max(closes[-self._ema_slow:])
            recent_low = min(closes[-self._ema_slow:])
            if direction == "long" and closes[-1] < recent_high * 0.995:
                score += 0.20
                warnings.append("structure_break_bearish")
            elif direction == "short" and closes[-1] > recent_low * 1.005:
                score += 0.20
                warnings.append("structure_break_bullish")

        return min(1.0, score), warnings

    # ==================== 动态止盈 ====================

    def _compute_adaptive_tp(
        self,
        direction: str,
        entry_price: float,
        base_tp_price: Optional[float],
        reversal_score: float,
    ) -> Tuple[Optional[float], float]:
        """根据反转评分收紧止盈价位。

        Returns: (adaptive_tp_price, tighten_factor)
        """
        if base_tp_price is None or base_tp_price <= 0 or entry_price <= 0:
            return base_tp_price, 1.0

        start = self._tighten_start_score
        max_factor = self._tighten_max_factor
        if reversal_score <= start:
            return base_tp_price, 1.0

        span = max(1e-6, 1.0 - start)
        factor = 1.0 - (reversal_score - start) / span * (1.0 - max_factor)
        factor = max(max_factor, min(1.0, factor))

        if direction == "long":
            distance = base_tp_price - entry_price
            adaptive_tp = entry_price + distance * factor
        else:
            distance = entry_price - base_tp_price
            adaptive_tp = entry_price - distance * factor

        return round(adaptive_tp, self._precision), factor

    # ==================== 落袋动作 ====================

    def _compute_exit_action(
        self,
        symbol: str,
        strategy_name: str,
        reversal_score: float,
        pnl_pct: float,
    ) -> Tuple[str, float, str]:
        """根据反转评分与盈利状态计算落袋动作（含冷却去重）。"""
        if reversal_score < self._partial_exit_score:
            return "none", 0.0, ""

        # 仅在已盈利时才落袋，避免在亏损/平价位被反转信号误触发减仓
        if pnl_pct < self._min_profit_lock_pct:
            return "none", 0.0, ""

        key = f"{symbol}:{strategy_name}"
        now = datetime.now().timestamp()
        last = self._last_action_ts.get(key, 0.0)
        if now - last < self._cooldown_seconds:
            return "none", 0.0, ""

        if reversal_score >= self._full_exit_score:
            self._last_action_ts[key] = now
            return "close", 1.0, "reversal_full_exit"
        if reversal_score >= self._partial_exit_score:
            self._last_action_ts[key] = now
            return "partial", self._partial_exit_ratio, "reversal_partial_exit"

        return "none", 0.0, ""

    # ==================== 工具方法 ====================

    @staticmethod
    def _normalize_direction(direction: str) -> str:
        d = str(direction).lower()
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
    def _extract_closes(ohlcv_data: List[Any]) -> List[float]:
        """从 K 线数据提取收盘价序列，兼容 list[list]（OKX: idx4=close）与 list[dict]。"""
        closes: List[float] = []
        for k in ohlcv_data:
            try:
                if isinstance(k, dict):
                    v = k.get("close") or k.get("c") or k.get("Close")
                elif isinstance(k, (list, tuple)):
                    v = k[4] if len(k) > 4 else None
                else:
                    v = None
                if v is not None:
                    closes.append(float(v))
            except (TypeError, ValueError, IndexError):
                continue
        return closes

    @staticmethod
    def _ema(values: List[float], period: int) -> List[float]:
        """简单 EMA 序列。"""
        if not values or period <= 0:
            return []
        k = 2.0 / (period + 1.0)
        ema = [values[0]]
        for i in range(1, len(values)):
            ema.append(values[i] * k + ema[-1] * (1.0 - k))
        return ema

    @staticmethod
    def _rsi(values: List[float], period: int = 14) -> List[float]:
        """Wilder RSI 序列。"""
        if len(values) < period + 1 or period <= 0:
            return [50.0] * len(values)
        deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]

        rsi: List[float] = [50.0] * period
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi.append(100.0)
            else:
                rs = avg_gain / avg_loss
                rsi.append(100.0 - 100.0 / (1.0 + rs))
        return rsi
