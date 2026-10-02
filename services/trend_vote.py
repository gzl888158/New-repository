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


def _wilder_smooth(values: Union[List[float], np.ndarray], period: int) -> np.ndarray:
    """Wilder 平滑：首值为前 period 项均值，之后按 1/period 指数衰减。

    返回与输入等长的平滑序列，前 period-1 项为 0（未收敛），供 ADX 链路使用。
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    out = np.zeros(n, dtype=float)
    if n < period or period <= 0:
        return out
    out[period - 1] = float(np.mean(arr[:period]))
    for i in range(period, n):
        out[i] = out[i - 1] - out[i - 1] / period + arr[i] / period
    return out


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14):
    """计算 ADX 及 +DI/-DI（真 Wilder 平滑，用于趋势强度）。

    原实现将单点 DX 与中性值混合（0.6*dx + 0.4*20），严重压缩 ADX 动态范围且
    丢失趋势强度的持续性。此处改为标准 Wilder 平滑 TR/DM → DI → DX → ADX 链，
    使 ADX 反映趋势强度的平滑持续性，方向识别更抗噪。
    """
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

    tr_s = _wilder_smooth(tr, period)
    pdi_s = _wilder_smooth(plus_dm, period)
    mdi_s = _wilder_smooth(minus_dm, period)

    # DX 序列（平滑未收敛的索引置 0），同时记录最新 +DI/-DI
    dx: List[float] = []
    last_pdi = 50.0
    last_mdi = 50.0
    for i in range(len(tr_s)):
        trv = tr_s[i]
        if trv <= 0:
            dx.append(0.0)
            continue
        pdi = 100.0 * pdi_s[i] / trv
        mdi = 100.0 * mdi_s[i] / trv
        last_pdi, last_mdi = pdi, mdi
        di_sum = pdi + mdi
        dx.append(100.0 * abs(pdi - mdi) / di_sum if di_sum > 0 else 0.0)

    adx_s = _wilder_smooth(dx, period)
    adx = float(adx_s[-1]) if len(adx_s) > 0 else 20.0

    # 数值防护：ADX/DI 出现非法值或非正值时回退中性
    if not (math.isfinite(adx) and math.isfinite(last_pdi) and math.isfinite(last_mdi)):
        return 20.0, 50.0, 50.0
    if adx <= 0:
        adx = 20.0
    return adx, last_pdi, last_mdi


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

    # ── 强化：因子可靠性加权 + DI 方向梯度化 ──
    # DI 方向用连续梯度（(plus_di-minus_di)/di_sum，-1..1）替代二值，捕捉方向强度差异；
    # DMI 方向因子最可靠（权重最高），结构次之，EMA/斜率对噪声敏感权重较低。
    di_sum = plus_di + minus_di
    di_grad = (plus_di - minus_di) / di_sum if di_sum > 0 else 0.0
    if not math.isfinite(di_grad):
        di_grad = 0.0
    direction = (
        0.20 * ema_dir
        + 0.20 * slope_dir
        + 0.25 * structure_dir
        + 0.35 * di_grad
    )
    direction = max(-1.0, min(1.0, direction))
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
