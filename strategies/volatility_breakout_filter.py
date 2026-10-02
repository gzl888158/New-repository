"""
波动率突破过滤器（副策略 D，不可单独运行）

作为 A/B 主策略的信号增强过滤器：
- 检测短期振幅持续缩小（挤压/盘整）。
- 检测到放量突破再顺势确认开仓。

设计约束：本模块为过滤器，类名不含 "Strategy" 后缀、不继承 PersistentStrategy，
不会被 StrategyDiscovery 自动发现为可独立交易策略，也不产生独立交易信号。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from strategies.trend_sub_strategies import _atr, _safe_int, _safe_float


def evaluate_volatility_breakout(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    volumes: np.ndarray,
    direction: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """波动率突破过滤器（纯函数）。

    逻辑：
      - 挤压：短期 ATR / 长期 ATR < squeeze_ratio_threshold → 振幅持续缩小。
      - 放量：当前成交量 > volume_ratio_threshold × 近期均量。
      - 突破：价格突破盘整区间（前 N 根高低点，不含当前 K 线）。

    三者同时满足且方向一致时 confirmed=True，作为 A/B 信号增强。
    """
    params = params or {}
    squeeze_period = _safe_int(params.get("squeeze_period", 20), 20)
    squeeze_compare = _safe_int(params.get("squeeze_compare", 60), 60)
    squeeze_ratio = _safe_float(params.get("squeeze_ratio_threshold", 0.6), 0.6)
    volume_ratio = _safe_float(params.get("volume_ratio_threshold", 1.5), 1.5)

    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    result = {
        "confirmed": False,
        "squeeze": False,
        "volume_breakout": False,
        "direction_breakout": False,
        "score": 0.0,
        "atr_short": 0.0,
        "atr_long": 0.0,
        "volume_ratio": 0.0,
    }

    if len(closes) < squeeze_compare + 1 or len(highs) < squeeze_period + 1:
        return result

    atr_short = _atr(highs, lows, closes, squeeze_period)
    atr_long = _atr(highs, lows, closes, squeeze_compare)
    squeeze = atr_long > 0 and (atr_short / atr_long) < squeeze_ratio

    vol_ma = float(np.mean(volumes[-squeeze_period:])) if len(volumes) >= squeeze_period else 0.0
    cur_vol = float(volumes[-1]) if len(volumes) > 0 else 0.0
    volume_breakout = vol_ma > 0 and cur_vol > volume_ratio * vol_ma

    close = float(closes[-1])
    breakout_high = float(np.max(highs[-squeeze_period - 1:-1]))
    breakout_low = float(np.min(lows[-squeeze_period - 1:-1]))

    direction_breakout = False
    if direction == "long" and close > breakout_high:
        direction_breakout = True
    elif direction == "short" and close < breakout_low:
        direction_breakout = True

    score = 0.0
    if squeeze:
        score += 0.3
    if volume_breakout:
        score += 0.3
    if direction_breakout:
        score += 0.4

    result.update({
        "confirmed": squeeze and volume_breakout and direction_breakout,
        "squeeze": squeeze,
        "volume_breakout": volume_breakout,
        "direction_breakout": direction_breakout,
        "score": score,
        "atr_short": atr_short,
        "atr_long": atr_long,
        "volume_ratio": (cur_vol / vol_ma) if vol_ma > 0 else 0.0,
    })
    return result


class VolatilityBreakoutFilter:
    """波动率突破过滤器（A/B 信号增强用，非独立策略）。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        cfg = self.config.get("strategies", {}).get("volatility_breakout_filter", {})
        self.enabled = bool(cfg.get("enabled", True))
        self.require_confirmation = bool(cfg.get("require_confirmation", False))
        # safe coerce：配置值可能为 None/字符串，避免 int()/float() 在 init 时崩溃
        self.params = {
            "squeeze_period": _safe_int(cfg.get("squeeze_period", 20), 20),
            "squeeze_compare": _safe_int(cfg.get("squeeze_compare", 60), 60),
            "squeeze_ratio_threshold": _safe_float(cfg.get("squeeze_ratio_threshold", 0.6), 0.6),
            "volume_ratio_threshold": _safe_float(cfg.get("volume_ratio_threshold", 1.5), 1.5),
        }

    def evaluate(self, closes, highs, lows, volumes, direction: str) -> Dict[str, Any]:
        """评估过滤器。禁用时放行（confirmed=True，不影响主策略）。"""
        if not self.enabled:
            return {"confirmed": True, "enabled": False, "score": 0.0}
        return evaluate_volatility_breakout(closes, highs, lows, volumes, direction, self.params)
