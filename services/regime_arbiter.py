"""
RegimeArbiter — 两套 regime 引擎输出融合-统一仲裁器
================================================================

背景：
- 主引擎 MarketRegimeEngine（services/market_regime_engine.py）服务于开仓/分配链路
- 细粒度检测器 MarketRegimeDetector（app/services/adaptive_learning/market_regime_detector.py）
  仅服务于止损管理器
- 两者输出未交叉融合，同一时刻可能矛盾

仲裁策略：
1. 一致 → 共识加分（confidence+0.1, cap 1.0）
2. 冲突 + 检测器 REVERSAL 概率超阈值 → 检测器 reversal 强制接管（止损 fail-closed）
3. 冲突 + 其他 → 按置信度加权仲裁
4. 检测器异常/UNKNOWN → fail-closed 回退主引擎
"""
import json
import os
from collections import deque
from datetime import datetime
from typing import Any, Dict, Optional

from services.market_regime_engine import MarketRegime


# 检测器 regime 字符串 → 主引擎 MarketRegime 枚举值
# 主引擎 L34-38 已声明 TRENDING_UP/TRENDING_DOWN/RANGING/HIGH_VOL/LOW_VOL 为别名，
# 可直接用 MarketRegime(det_str).value 归一；这里显式列出便于审查与兜底。
_DETECTOR_TO_MAIN = {
    "trending_up": MarketRegime.TREND_BULLISH.value,
    "trending_down": MarketRegime.TREND_BEARISH.value,
    "ranging": MarketRegime.RANGE_BOUND.value,
    "high_volatility": MarketRegime.EXTREME_VOLATILITY.value,
    "low_volatility": MarketRegime.RANGE_BOUND.value,
    "breakout": MarketRegime.BREAKOUT.value,
    "reversal": MarketRegime.REVERSAL.value,
    # 主引擎别名（检测器可能直接传主引擎口径）
    "trend_bullish": MarketRegime.TREND_BULLISH.value,
    "trend_bearish": MarketRegime.TREND_BEARISH.value,
    "range_bound": MarketRegime.RANGE_BOUND.value,
    "extreme_volatility": MarketRegime.EXTREME_VOLATILITY.value,
    "funding_crush": MarketRegime.FUNDING_CRUSH.value,
    "liquidity_crisis": MarketRegime.LIQUIDITY_CRISIS.value,
}


class RegimeArbiter:
    """两套 regime 引擎输出融合-统一仲裁器。

    仅消费两引擎的 get_regime() 输出，不改其内部实现。
    arbiter=None 时所有下游透明回退旧路径。
    """

    def __init__(self, main_engine, detector, config: Dict = None):
        self._main = main_engine
        self._detector = detector
        cfg = config or {}
        self._reversal_threshold = float(cfg.get("reversal_threshold", 0.45))
        raw_w_main = max(0.0, _safe_float(cfg.get("w_main", 0.6), 0.6))
        raw_w_detector = max(0.0, _safe_float(cfg.get("w_detector", 0.4), 0.4))
        weight_total = raw_w_main + raw_w_detector
        if weight_total <= 0.0:
            raw_w_main, raw_w_detector, weight_total = 0.6, 0.4, 1.0
        self._w_main = raw_w_main / weight_total
        self._w_detector = raw_w_detector / weight_total
        self._conflict_log = str(cfg.get("conflict_log", "data/regime_arbiter/conflicts.jsonl"))
        self._history_len = int(cfg.get("history_len", 200))
        self._history: deque = deque(maxlen=self._history_len)

    # ── 对外主接口 ───────────────────────────────────────

    def _coerce_detector_result(self, det_out: Dict[str, Any]) -> Dict[str, Any]:
        """将全局 detector 快照归一为主引擎基准币状态；不借用其他币种冒充全局态。"""
        if not isinstance(det_out, dict) or not det_out:
            return det_out or {}

        if det_out.get("regime"):
            return det_out

        symbols = det_out.get("symbols")
        if not isinstance(symbols, dict):
            return det_out

        base_symbol = getattr(self._main, "_base_symbol", None)
        if not isinstance(base_symbol, str) or not base_symbol:
            base_symbol = "BTC-USDT-SWAP"
        chosen = symbols.get(base_symbol)
        if not isinstance(chosen, dict) or not chosen.get("regime"):
            return det_out

        chosen = dict(chosen)
        chosen.setdefault("symbol", base_symbol)
        return chosen

    def arbitrate(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """融合两引擎输出，返回统一 regime dict（主引擎 schema + 扩展字段）。

        Args:
            symbol: 交易对符号；None 时取整体市场 regime

        Returns:
            dict: 兼容主引擎 get_regime() 输出，附加 detector_regime/
                  detector_reversal_prob/early_warnings/arbiter_conflict/
                  arbiter_strategy/source_weights/detector_raw
        """
        # 1. 优先获取同一币种的主引擎状态，避免和检测器的 symbol 级输出错位仲裁。
        symbol_scope_error = None
        try:
            main_out = None
            if self._main and symbol:
                get_symbol_regime = getattr(self._main, "get_symbol_regime", None)
                symbol_out = self._call_main_regime(get_symbol_regime, symbol) if callable(get_symbol_regime) else None
                if isinstance(symbol_out, dict) and symbol_out and symbol_out.get("symbol_specific", True) is not False:
                    main_out = symbol_out
                else:
                    symbol_scope_error = f"symbol-level main regime unavailable for '{symbol}'"
                    main_out = self._call_main_regime(getattr(self._main, "get_regime", None))
            else:
                main_out = self._call_main_regime(getattr(self._main, "get_regime", None)) if self._main else None
        except Exception as exc:
            main_out = None
            _log_safe(f"[RegimeArbiter] main_engine.get_regime failed: {exc}")

        # 主引擎也失败 → 返回最小兜底
        if not isinstance(main_out, dict) or not main_out:
            return self._fallback_minimal(symbol)

        if symbol_scope_error:
            return self._assemble_main_fallback(main_out, symbol, symbol_scope_error)

        # 2. 取检测器输出（逐币种 / 全量市场快照）
        det_out = None
        det_err = None
        if self._detector is not None:
            try:
                raw_det_out = self._detector.get_regime(symbol) if symbol else self._detector.get_regime()
                det_out = self._coerce_detector_result(raw_det_out)
            except Exception as exc:
                det_err = str(exc)
                det_out = None

        # 检测器缺失/异常 → fail-closed 回退主引擎
        if not isinstance(det_out, dict) or not det_out:
            return self._assemble_main_fallback(main_out, symbol, det_err)

        # 3. 状态枚举映射
        det_regime_raw = str(det_out.get("regime", "unknown")).lower()
        mapped = _DETECTOR_TO_MAIN.get(det_regime_raw)
        if mapped is None:
            # UNKNOWN 或未识别 → fail-closed 回退主引擎
            return self._assemble_main_fallback(
                main_out, symbol, f"detector regime '{det_regime_raw}' unmapped"
            )

        main_regime = str(main_out.get("regime", "unknown"))

        # 4. 一致 → 共识加分
        if mapped == main_regime:
            return self._assemble_consensus(main_out, det_out, det_regime_raw, mapped, symbol)

        # 5. 冲突 → 仲裁
        return self._resolve_conflict(main_out, det_out, det_regime_raw, mapped, symbol)

    def get_regime(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """兼容主引擎接口：允许外部直接通过 arbiter 获取统一 regime。"""
        return self.arbitrate(symbol)

    def get_symbol_regime(self, symbol: str) -> Dict[str, Any]:
        """兼容主引擎方法：逐币种获取统一 regime。"""
        return self.arbitrate(symbol)

    def get_position_adjustment(self) -> Dict[str, float]:
        """兼容主引擎方法：回退为主引擎的仓位调节策略。"""
        if self._main is not None and hasattr(self._main, "get_position_adjustment"):
            try:
                return self._main.get_position_adjustment()
            except Exception:
                pass
        return {"overall": 1.0, "trend": 1.0, "scalping": 1.0, "grid": 1.0, "arbitrage": 1.0}

    # ── 仲裁分支 ─────────────────────────────────────────

    def _resolve_conflict(
        self,
        main_out: Dict,
        det_out: Dict,
        det_regime_raw: str,
        mapped: str,
        symbol: Optional[str],
    ) -> Dict:
        """冲突消解：reversal 强制接管 / 加权仲裁。"""
        main_conf = _clip(_safe_float(main_out.get("confidence"), 0.5), 0.0, 1.0)
        main_strength = _safe_float(main_out.get("strength"), 0.5)
        det_probs = det_out.get("probabilities", {}) or {}
        det_reversal_prob = _safe_float(det_probs.get("reversal"), 0.0)
        det_conf = _clip(
            _safe_float(
                det_out.get("confidence"),
                _safe_float(det_probs.get(det_regime_raw), max([_safe_float(v) for v in det_probs.values()] + [0.0])),
            ),
            0.0,
            1.0,
        )

        resolved = dict(main_out)  # 复制主引擎输出保持兼容
        strategy = "weighted"
        conflict = True

        # 分支 A：检测器 REVERSAL 概率超阈值 → 强制接管（止损 fail-closed）
        if mapped == MarketRegime.REVERSAL.value and det_reversal_prob >= self._reversal_threshold:
            resolved["regime"] = MarketRegime.REVERSAL.value
            resolved["normalized_regime"] = MarketRegime.REVERSAL.value
            resolved["state"] = MarketRegime.REVERSAL.value
            resolved["subtype"] = "reversal"
            strategy = "detector_reversal_override"

        # 分支 B：加权仲裁
        else:
            s_main = main_conf * self._w_main
            s_det = det_conf * self._w_detector
            if s_det > s_main:
                resolved["regime"] = mapped
                resolved["normalized_regime"] = mapped
                resolved["state"] = mapped
            # s_main >= s_det 时保持主引擎 regime 不变

        # 两路置信度均按归一化来源权重融合，避免固定共识加分。
        blended_conf = self._blend_confidence(main_conf, det_conf)
        if strategy == "detector_reversal_override":
            blended_conf = self._blend_confidence(main_conf, det_reversal_prob)
        resolved["confidence"] = blended_conf
        resolved["strength"] = _clip(
            main_strength * self._w_main + det_conf * self._w_detector, 0.0, 1.0
        )

        # 扩展字段
        resolved.update(self._build_extension(
            det_out, det_regime_raw, det_reversal_prob, conflict, strategy,
            main_conf, det_conf, blended_conf, symbol
        ))

        # 记录历史 + 冲突日志
        self._record(resolved, symbol, main_out.get("regime"), det_regime_raw,
                     strategy, main_conf, det_conf, blended_conf, conflict=True)
        return resolved

    def _assemble_consensus(
        self, main_out: Dict, det_out: Dict,
        det_regime_raw: str, mapped: str, symbol: Optional[str],
    ) -> Dict:
        """一致时按两路状态置信度及归一化来源权重融合。"""
        resolved = dict(main_out)
        main_conf = _clip(_safe_float(main_out.get("confidence"), 0.5), 0.0, 1.0)
        det_probs = det_out.get("probabilities", {}) or {}
        det_reversal_prob = _safe_float(det_probs.get("reversal"), 0.0)
        det_conf = _clip(
            _safe_float(
                det_out.get("confidence"),
                _safe_float(det_probs.get(det_regime_raw), max([_safe_float(v) for v in det_probs.values()] + [0.0])),
            ),
            0.0,
            1.0,
        )
        blended_conf = self._blend_confidence(main_conf, det_conf)
        resolved["confidence"] = blended_conf

        resolved.update(self._build_extension(
            det_out, det_regime_raw, det_reversal_prob, False, "consensus",
            main_conf, det_conf, blended_conf, symbol
        ))
        self._record(resolved, symbol, main_out.get("regime"), det_regime_raw,
                     "consensus", main_conf, det_conf, blended_conf, conflict=False)
        return resolved

    def _assemble_main_fallback(
        self, main_out: Dict, symbol: Optional[str], det_err: Optional[str],
    ) -> Dict:
        """fail-closed 回退主引擎。"""
        resolved = dict(main_out)
        resolved.update({
            "detector_regime": None,
            "detector_reversal_prob": 0.0,
            "early_warnings": [],
            "arbiter_conflict": False,
            "arbiter_strategy": "main_fallback",
            "source_weights": {"main": 1.0, "detector": 0.0},
            "detector_raw": None,
            "arbiter_error": det_err,
        })
        self._record(resolved, symbol, main_out.get("regime"), None,
                     "main_fallback", _safe_float(main_out.get("confidence"), 0.5),
                     0.0, _safe_float(main_out.get("confidence"), 0.5), conflict=False)
        return resolved

    def _fallback_minimal(self, symbol: Optional[str]) -> Dict:
        """主引擎也失败时的最小兜底。"""
        return {
            "regime": "unknown",
            "normalized_regime": "unknown",
            "state": "unknown",
            "subtype": "unknown",
            "strength": 0.0,
            "confidence": 0.0,
            "factor_scores": {},
            "factor_weights": {},
            "last_update": datetime.now().isoformat(),
            "detector_regime": None,
            "detector_reversal_prob": 0.0,
            "early_warnings": [],
            "arbiter_conflict": False,
            "arbiter_strategy": "main_fallback",
            "source_weights": {"main": 0.0, "detector": 0.0},
            "detector_raw": None,
            "arbiter_error": "main_engine unavailable",
        }

    # ── 辅助 ─────────────────────────────────────────────

    @staticmethod
    def _call_main_regime(getter, symbol: Optional[str] = None):
        if not callable(getter):
            return None
        try:
            if symbol is not None:
                return getter(symbol, include_fusion=False)
            return getter(include_fusion=False)
        except TypeError:
            return getter(symbol) if symbol is not None else getter()

    def _blend_confidence(self, main_conf: float, detector_conf: float) -> float:
        return _clip(main_conf * self._w_main + detector_conf * self._w_detector, 0.0, 1.0)

    def _build_extension(
        self, det_out: Dict, det_regime_raw: str, det_reversal_prob: float,
        conflict: bool, strategy: str, main_conf: float, det_conf: float,
        blended_conf: float, symbol: Optional[str],
    ) -> Dict:
        return {
            "detector_regime": det_regime_raw,
            "detector_reversal_prob": det_reversal_prob,
            "early_warnings": det_out.get("early_warnings", []) or [],
            "arbiter_conflict": conflict,
            "arbiter_strategy": strategy,
            "source_weights": {"main": self._w_main, "detector": self._w_detector},
            "detector_raw": det_out,
        }

    def _record(
        self, resolved: Dict, symbol: Optional[str], main_regime: Optional[str],
        det_regime: Optional[str], strategy: str, main_conf: float,
        det_conf: float, blended_conf: float, conflict: bool,
    ) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "symbol": symbol,
            "main_regime": main_regime,
            "detector_regime": det_regime,
            "resolved": resolved.get("regime"),
            "strategy": strategy,
            "main_conf": round(main_conf, 4),
            "det_conf": round(det_conf, 4),
            "blended_conf": round(blended_conf, 4),
            "conflict": conflict,
        }
        self._history.append(entry)
        if conflict:
            _log_safe(
                f"[RegimeArbiter] conflict symbol={symbol} main={main_regime} "
                f"detector={det_regime} strategy={strategy} resolved={resolved.get('regime')}"
            )
            self._write_conflict_log(entry)

    def _write_conflict_log(self, entry: Dict) -> None:
        """追加写冲突日志 JSONL。"""
        try:
            os.makedirs(os.path.dirname(self._conflict_log), exist_ok=True)
            with open(self._conflict_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            _log_safe(f"[RegimeArbiter] conflict log write failed: {exc}")

    def get_arbiter_history(self, limit: int = 200) -> list:
        """返回最近 N 条仲裁记录。"""
        return list(self._history)[-limit:]

    def get_conflict_stats(self) -> Dict[str, Any]:
        """返回冲突统计（供 dashboard 调用）。"""
        total = len(self._history)
        conflicts = sum(1 for e in self._history if e.get("conflict"))
        return {
            "total_arbitrations": total,
            "conflicts": conflicts,
            "conflict_rate": (conflicts / total) if total > 0 else 0.0,
            "strategy_distribution": _count_strategies(self._history),
        }


# ── 模块级工具函数 ────────────────────────────────────────

def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        if f != f or f in (float("inf"), float("-inf")):  # NaN/Inf
            return default
        return f
    except (TypeError, ValueError):
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _count_strategies(history) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in history:
        s = str(e.get("strategy", "unknown"))
        counts[s] = counts.get(s, 0) + 1
    return counts


def _log_safe(msg: str) -> None:
    """安全日志（避免循环导入 logger）。"""
    try:
        import logging
        logging.getLogger("regime_arbiter").warning(msg)
    except Exception:
        pass
