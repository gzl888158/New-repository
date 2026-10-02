"""
趋势类子策略模块（企业级 v2.0）

包含四类趋势子策略的纯函数实现，供 TrendStrategy 调用：
1. 均线趋势（MA/EMA 金叉死叉 + 多均线排列 + ATR 震荡过滤）
2. 唐奇安通道突破（Donchian + ATR 动态止盈止损）
3. MACD 趋势过滤器（零轴过滤，仅辅助确认，非独立主策略）
4. 动量策略（滚动 2h/4h 涨跌幅，多品种多空对冲）

所有函数均为纯函数：输入 numpy 数组与参数，输出信号字典，无 I/O、无状态，便于单测与复用。
"""
from typing import Dict, List, Optional, Any

import numpy as np


# ============================================================
# 内部指标原语
# ============================================================
def _safe_int(v: Any, default: int) -> int:
    """安全 int 转换：None/非法值回退到 default，避免纯函数被畸形参数打崩。"""
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _safe_float(v: Any, default: float) -> float:
    """安全 float 转换：None/非法值回退到 default。"""
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _sma(prices: np.ndarray, period: int) -> Optional[float]:
    """简单移动平均。数据不足返回 None。"""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period or period <= 0:
        return None
    window = prices[-period:]
    if np.any(np.isnan(window)) or np.any(np.isinf(window)):
        return None
    return float(np.mean(window))


def _ema(prices: np.ndarray, period: int) -> Optional[float]:
    """指数移动平均（最新值）。数据不足返回 None。"""
    prices = np.asarray(prices, dtype=float)
    if len(prices) < period or period <= 0:
        return None
    if np.any(np.isnan(prices)) or np.any(np.isinf(prices)):
        return None
    multiplier = 2.0 / (period + 1)
    ema = float(np.mean(prices[:period]))
    for p in prices[period:]:
        ema = float(p) * multiplier + ema * (1.0 - multiplier)
    return ema


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    """平均真实波幅（ATR）。数据不足返回 0。"""
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)
    if len(highs) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(highs)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return 0.0
    atr = float(np.mean(trs[-period:]))
    if np.isnan(atr) or np.isinf(atr) or atr <= 0:
        return 0.0
    return atr


def _rolling_return(closes: np.ndarray, lookback: int) -> float:
    """滚动区间收益率：closes[-1] / closes[-lookback] - 1。"""
    closes = np.asarray(closes, dtype=float)
    if len(closes) < lookback + 1 or lookback <= 0:
        return 0.0
    base = closes[-lookback - 1]
    if base <= 0 or np.isnan(base) or np.isinf(base):
        return 0.0
    last = closes[-1]
    if np.isnan(last) or np.isinf(last):
        return 0.0
    return float(last / base - 1.0)


# ============================================================
# 1. 均线趋势（MA/EMA 金叉死叉 + 多均线排列 + ATR 震荡过滤）
# ============================================================
def evaluate_ma_trend(
    closes: np.ndarray,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """均线趋势子策略。

    逻辑：
      - 短期均线(MA5)上穿中期均线(MA20) → 金叉做多；下穿 → 死叉做空。
      - 多均线多头排列 MA5 > MA20 > MA60 增强多头；空头排列增强空头。
      - EMA20 方向确认趋势方向。
      - ATR 波动率阈值过滤：ATR% 低于阈值判定为震荡，不产生信号（score=0）。

    返回：
      {"signal": 'long'|'short'|None, "score": 0~1, "confidence": 0~1,
       "ma5": ..., "ma20": ..., "ma60": ..., "ema20": ..., "atr_pct": ..., "range_market": bool}
    """
    params = params or {}
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float) if highs is not None else np.array([])
    lows = np.asarray(lows, dtype=float) if lows is not None else np.array([])

    atr_pct_threshold = _safe_float(params.get("atr_pct_threshold", 0.003), 0.003)  # ATR% 低于 0.3% 视为震荡

    ma5 = _sma(closes, 5)
    ma20 = _sma(closes, 20)
    ma60 = _sma(closes, 60)
    ema20 = _ema(closes, 20)

    price = float(closes[-1]) if len(closes) > 0 else 0.0
    atr_val = _atr(highs, lows, closes, 14) if len(highs) > 0 else 0.0
    atr_pct = (atr_val / price) if price > 0 and atr_val > 0 else 0.0

    detail = {
        "ma5": ma5, "ma20": ma20, "ma60": ma60, "ema20": ema20,
        "atr_pct": atr_pct, "range_market": False,
    }

    # 数据不足无法判断
    if ma5 is None or ma20 is None:
        detail["signal"] = None
        detail["score"] = 0.0
        detail["confidence"] = 0.0
        return detail

    # ATR 震荡过滤：波动率过低，趋势难形成，直接否决
    if atr_pct > 0 and atr_pct < atr_pct_threshold:
        detail["range_market"] = True
        detail["signal"] = None
        detail["score"] = 0.0
        detail["confidence"] = 0.0
        return detail

    # 金叉/死叉判定（MA5 与 MA20 的穿越）
    golden_cross = False
    death_cross = False
    if len(closes) >= 21:
        prev_ma5 = _sma(closes[:-1], 5)
        prev_ma20 = _sma(closes[:-1], 20)
        if prev_ma5 is not None and prev_ma20 is not None:
            golden_cross = prev_ma5 <= prev_ma20 and ma5 > ma20
            death_cross = prev_ma5 >= prev_ma20 and ma5 < ma20

    # 多均线排列
    bull_alignment = ma60 is not None and ma5 > ma20 > ma60
    bear_alignment = ma60 is not None and ma5 < ma20 < ma60

    signal = None
    score = 0.0
    confidence = 0.0

    # 多头信号
    if golden_cross or bull_alignment or (ma5 > ma20 and ema20 is not None and price > ema20):
        signal = "long"
        score = 0.6
        if golden_cross:
            score += 0.2
        if bull_alignment:
            score += 0.2
        confidence = min(0.95, 0.45 + score * 0.5)

    # 空头信号
    elif death_cross or bear_alignment or (ma5 < ma20 and ema20 is not None and price < ema20):
        signal = "short"
        score = 0.6
        if death_cross:
            score += 0.2
        if bear_alignment:
            score += 0.2
        confidence = min(0.95, 0.45 + score * 0.5)

    detail["signal"] = signal
    detail["score"] = score
    detail["confidence"] = confidence
    detail["golden_cross"] = golden_cross
    detail["death_cross"] = death_cross
    detail["bull_alignment"] = bull_alignment
    detail["bear_alignment"] = bear_alignment
    return detail


# ============================================================
# 1b. 多周期 EMA 趋势跟踪（主策略 A）
# ============================================================
def evaluate_ema_trend(
    closes: np.ndarray,
    highs: Optional[np.ndarray] = None,
    lows: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """多周期 EMA 趋势跟踪（主策略 A）。

    逻辑：
      - 短 EMA 上穿长 EMA → 金叉做多；下穿 → 死叉做空。
      - 波动率阈值（ATR%）过滤震荡行情。
      - ATR 动态止盈止损参考（atr_sl_mult / atr_tp_mult）。

    返回：
      {"signal": 'long'|'short'|None, "score": 0~1, "confidence": 0~1,
       "ema_fast": ..., "ema_slow": ..., "atr_pct": ..., "range_market": bool,
       "golden_cross": bool, "death_cross": bool, "stop_loss": ..., "take_profit": ...}
    """
    params = params or {}
    fast = _safe_int(params.get("ema_fast", 9), 9)
    slow = _safe_int(params.get("ema_slow", 21), 21)
    atr_pct_threshold = _safe_float(params.get("atr_pct_threshold", 0.003), 0.003)
    atr_sl_mult = _safe_float(params.get("atr_sl_mult", 2.0), 2.0)
    atr_tp_mult = _safe_float(params.get("atr_tp_mult", 3.0), 3.0)

    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float) if highs is not None else np.array([])
    lows = np.asarray(lows, dtype=float) if lows is not None else np.array([])

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    price = float(closes[-1]) if len(closes) > 0 else 0.0
    atr_val = _atr(highs, lows, closes, 14) if len(highs) > 0 else 0.0
    atr_pct = (atr_val / price) if price > 0 and atr_val > 0 else 0.0

    detail = {
        "ema_fast": ema_fast, "ema_slow": ema_slow,
        "atr_pct": atr_pct, "range_market": False,
    }

    if ema_fast is None or ema_slow is None or price <= 0:
        detail["signal"] = None
        detail["score"] = 0.0
        detail["confidence"] = 0.0
        return detail

    # ATR 震荡过滤：波动率过低，趋势难形成
    if atr_pct > 0 and atr_pct < atr_pct_threshold:
        detail["range_market"] = True
        detail["signal"] = None
        detail["score"] = 0.0
        detail["confidence"] = 0.0
        return detail

    # 穿越判定（前一根与当前 EMA 对比）
    golden_cross = False
    death_cross = False
    if len(closes) >= slow + 1:
        prev_fast = _ema(closes[:-1], fast)
        prev_slow = _ema(closes[:-1], slow)
        if prev_fast is not None and prev_slow is not None:
            golden_cross = prev_fast <= prev_slow and ema_fast > ema_slow
            death_cross = prev_fast >= prev_slow and ema_fast < ema_slow

    signal = None
    score = 0.0
    if golden_cross or ema_fast > ema_slow:
        signal = "long"
        score = 0.6 + (0.2 if golden_cross else 0.0)
    elif death_cross or ema_fast < ema_slow:
        signal = "short"
        score = 0.6 + (0.2 if death_cross else 0.0)

    confidence = min(0.95, 0.45 + score * 0.5) if signal else 0.0

    # ATR 动态止盈止损
    sl_offset = atr_val * atr_sl_mult if atr_val > 0 else 0.0
    tp_offset = atr_val * atr_tp_mult if atr_val > 0 else 0.0
    stop_loss = (price - sl_offset) if signal == "long" else (price + sl_offset) if signal == "short" else None
    take_profit = (price + tp_offset) if signal == "long" else (price - tp_offset) if signal == "short" else None

    detail.update({
        "signal": signal, "score": score, "confidence": confidence,
        "golden_cross": golden_cross, "death_cross": death_cross,
        "stop_loss": stop_loss, "take_profit": take_profit,
    })
    return detail


# ============================================================
# 2. 唐奇安通道突破（Donchian）
# ============================================================
def evaluate_donchian(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """唐奇安通道突破子策略。

    逻辑：
      - 上轨 = 过去 N 周期最高价，下轨 = 过去 N 周期最低价。
      - 收盘价突破上轨 → 做多；跌破下轨 → 做空。
      - 提供 ATR 动态止盈止损参考（atr_sl / atr_tp 为价格偏移）。

    返回：
      {"signal": ..., "score": ..., "confidence": ...,
       "upper": ..., "lower": ..., "mid": ..., "breakout": bool, "atr": ...}
    """
    params = params or {}
    n = _safe_int(params.get("donchian_period", 20), 20)
    atr_multiplier_sl = _safe_float(params.get("atr_sl_multiplier", 2.0), 2.0)
    atr_multiplier_tp = _safe_float(params.get("atr_tp_multiplier", 3.0), 3.0)

    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    closes = np.asarray(closes, dtype=float)

    if len(highs) < n + 1 or len(lows) < n + 1 or len(closes) < 1:
        return {"signal": None, "score": 0.0, "confidence": 0.0,
                "upper": None, "lower": None, "mid": None, "breakout": False, "atr": 0.0}

    # 前 N 根高低点（不含当前 K 线），避免当前 high/low 使突破恒不成立
    upper = float(np.max(highs[-n - 1:-1]))
    lower = float(np.min(lows[-n - 1:-1]))
    mid = (upper + lower) / 2.0
    close = float(closes[-1])
    atr_val = _atr(highs, lows, closes, 14)

    signal = None
    score = 0.0
    confidence = 0.0
    breakout = False

    if upper > 0 and close > upper:
        signal = "long"
        breakout = True
        # 突破幅度越大，信号越强（但限制上限避免追高）
        penetration = min(1.0, (close - upper) / upper)
        score = 0.6 + min(0.4, penetration * 10.0)
    elif lower > 0 and close < lower:
        signal = "short"
        breakout = True
        penetration = min(1.0, (lower - close) / lower)
        score = 0.6 + min(0.4, penetration * 10.0)

    if signal is not None:
        confidence = min(0.95, 0.45 + score * 0.5)

    # ATR 动态止盈止损（价格偏移）
    sl_offset = atr_val * atr_multiplier_sl if atr_val > 0 else 0.0
    tp_offset = atr_val * atr_multiplier_tp if atr_val > 0 else 0.0
    stop_loss = (close - sl_offset) if signal == "long" else (close + sl_offset) if signal == "short" else None
    take_profit = (close + tp_offset) if signal == "long" else (close - tp_offset) if signal == "short" else None

    return {
        "signal": signal, "score": score, "confidence": confidence,
        "upper": upper, "lower": lower, "mid": mid,
        "breakout": breakout, "atr": atr_val,
        "stop_loss": stop_loss, "take_profit": take_profit,
    }


# ============================================================
# 3. MACD 趋势过滤器（零轴过滤，仅辅助）
# ============================================================
def macd_zero_axis(macd: float, signal_line: float) -> str:
    """MACD 零轴过滤。

    零轴上方只做多、零轴下方只做空。返回 'bull' / 'bear' / 'neutral'。
    仅作为辅助过滤器，不单独产生交易信号。
    """
    if macd > 0 and signal_line > 0:
        return "bull"
    if macd < 0 and signal_line < 0:
        return "bear"
    return "neutral"


# ============================================================
# 4. 动量策略（滚动涨跌幅，多品种多空对冲）
# ============================================================
def rolling_return(closes: np.ndarray, lookback: int) -> float:
    """滚动区间收益率。供动量策略计算 2h/4h 涨跌幅。"""
    return _rolling_return(np.asarray(closes, dtype=float), lookback)


def rank_momentum(
    returns: Dict[str, float],
    top_n: int = 3,
    bottom_n: int = 3,
) -> Dict[str, Any]:
    """对品种按动量收益率排序，返回多空对冲候选。

    强者恒强：收益率靠前的做多，靠后的做空（多品种对冲）。

    参数：
      returns: {symbol: return_float}
      top_n / bottom_n: 做多 / 做空的品种数量

    返回：
      {"long": [(symbol, ret), ...], "short": [(symbol, ret), ...]}
    """
    if not returns:
        return {"long": [], "short": []}

    items = sorted(returns.items(), key=lambda kv: kv[1], reverse=True)
    longs = items[:top_n]
    # 只对收益率为负（弱势）的品种做空，避免逆势做空强势品种
    shorts = [it for it in items[-bottom_n:] if it[1] < 0]
    return {"long": longs, "short": shorts}
