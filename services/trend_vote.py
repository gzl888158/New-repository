"""企业级趋势行情判断（共享模块）

提供统一的 ADX 趋势强度 + 价格结构 + EMA 多周期排列 + DI 方向的多因子投票，
供 MarketRegimeEngine（状态层）与 TrendStrategy（策略层）共享，
避免两套趋势判断逻辑漂移、参数不一致。

核心输出 `direction`（-1..1）由四因子投票决定，`adx_strength`（0..1）由 ADX 归一化，
调用方再用各自的组合方式（如多周期确认权重）计算最终趋势得分。
"""
from typing import Dict, List, Union
import math
import numpy as np


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """指数移动平均（alpha = 2/(period+1)）"""
    if len(data) == 0:
        return np.array([], dtype=float)
    alpha = 2.0 / (period + 1)
    out = np.empty_like(data, dtype=float)
    out[0] = data[0]
    for i in range(1, len(data)):
        out[i] = data[i] * alpha + out[i - 1] * (1 - alpha)
    return out


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14):
    """计算 ADX 及 +DI/-DI（简化 Wilder 平滑，用于趋势强度）"""
    n = len(closes)
    if n < period + 1:
        return 20.0, 50.0, 50.0

    # 数据层：NaN/Inf 清洗，避免污染 TR/DM 计算
    if not (np.all(np.isfinite(highs)) and np.all(np.isfinite(lows)) and np.all(np.isfinite(closes))):
        return 20.0, 50.0, 50.0

    tr: List[float] = []
    plus_dm: List[float] = []
    minus_dm: List[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
        plus_dm.append(up if (up > down and up > 0) else 0.0)
        minus_dm.append(down if (down > up and down > 0) else 0.0)

    tr_s = _ema(np.asarray(tr, dtype=float), period)
    pdi_s = _ema(np.asarray(plus_dm, dtype=float), period)
    mdi_s = _ema(np.asarray(minus_dm, dtype=float), period)

    last_tr = tr_s[-1]
    if last_tr <= 0:
        return 20.0, 50.0, 50.0

    pdi = 100.0 * pdi_s[-1] / last_tr
    mdi = 100.0 * mdi_s[-1] / last_tr
    di_sum = pdi + mdi
    dx = 100.0 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0.0
    # 单点 DX 与中性值混合，降低噪声
    adx = 0.6 * dx + 0.4 * 20.0
    return adx, pdi, mdi


def compute_trend_vote(
    closes: Union[List[float], np.ndarray],
    highs: Union[List[float], np.ndarray],
    lows: Union[List[float], np.ndarray],
    adx_period: int = 14,
    adx_floor: float = 15.0,
    adx_saturation: float = 40.0,
) -> Dict[str, float]:
    """趋势方向投票 + ADX 强度

    返回:
      direction:      -1..1（四因子投票：EMA排列、均线斜率、价格结构、DI方向）
      adx:            原始 ADX
      adx_strength:   0..1（ADX 归一化，低于 floor 归零）
      votes:          各因子方向（诊断用）
    """
    # ── 参数层：adx_floor/adx_saturation 合法性校验，非法时回退安全默认 ──
    # 必须满足 0 <= adx_floor < adx_saturation，否则归一化分母为 0 或为负、
    # adx_strength 反向映射，趋势强度输出失真。
    try:
        adx_floor = float(adx_floor)
        adx_saturation = float(adx_saturation)
    except (TypeError, ValueError):
        adx_floor, adx_saturation = 15.0, 40.0
    if not (math.isfinite(adx_floor) and math.isfinite(adx_saturation)) \
            or adx_floor < 0.0 or adx_floor >= adx_saturation:
        adx_floor, adx_saturation = 15.0, 40.0

    # ── 数据层：NaN/Inf 清洗（含 NaN 的序列会污染 EMA/结构/ADX 全链路） ──
    closes = np.asarray(closes, dtype=float)
    highs = np.asarray(highs, dtype=float)
    lows = np.asarray(lows, dtype=float)
    if not (np.all(np.isfinite(closes)) and np.all(np.isfinite(highs)) and np.all(np.isfinite(lows))):
        return {
            "direction": 0.0,
            "adx": 20.0,
            "adx_strength": 0.0,
            "votes": {"ema": 0.0, "slope": 0.0, "structure": 0.0, "di": 0.0},
        }
    n = len(closes)
    if n < 20:
        return {
            "direction": 0.0,
            "adx": 20.0,
            "adx_strength": 0.0,
            "votes": {"ema": 0.0, "slope": 0.0, "structure": 0.0, "di": 0.0},
        }

    ema5 = _ema(closes, 5)
    ema10 = _ema(closes, 10)
    ema20 = _ema(closes, 20)
    e5, e10, e20 = ema5[-1], ema10[-1], ema20[-1]
    ema_dir = 1.0 if e5 > e10 > e20 else (-1.0 if e5 < e10 < e20 else 0.0)

    slope = ema20[-1] - ema20[-4]
    slope_dir = 1.0 if slope > 0 else (-1.0 if slope < 0 else 0.0)

    rh, rl = float(np.max(highs[-5:])), float(np.min(lows[-5:]))
    ph, pl = float(np.max(highs[-10:-5])), float(np.min(lows[-10:-5]))
    if rh > ph and rl > pl:
        structure_dir = 1.0
    elif rh < ph and rl < pl:
        structure_dir = -1.0
    else:
        structure_dir = 0.0

    adx, plus_di, minus_di = _adx(highs, lows, closes, adx_period)
    # ── 运算层：ADX 结果 NaN/Inf 防护（_adx 内部已兜底，此处二次保险） ──
    if not (math.isfinite(adx) and math.isfinite(plus_di) and math.isfinite(minus_di)):
        adx, plus_di, minus_di = 20.0, 50.0, 50.0
    di_dir = 1.0 if plus_di > minus_di else (-1.0 if minus_di > plus_di else 0.0)

    direction = (ema_dir + slope_dir + structure_dir + di_dir) / 4.0
    # ── 运算层：adx_strength 归一化除法除零/反向防护（参数层已保证 floor<saturation） ──
    span = adx_saturation - adx_floor
    if span <= 0 or not math.isfinite(span):
        adx_strength = 0.0
    else:
        adx_strength = (adx - adx_floor) / span
        adx_strength = 0.0 if not math.isfinite(adx_strength) else max(0.0, min(1.0, adx_strength))

    return {
        "direction": direction,
        "adx": float(adx),
        "adx_strength": adx_strength,
        "votes": {"ema": ema_dir, "slope": slope_dir, "structure": structure_dir, "di": di_dir},
    }
