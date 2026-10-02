"""
eval 模块共享工具：类型安全转换与 JSON 序列化安全辅助。

企业级约束：
- 所有外部输入（config / API 返回 / 持久化数据）必须经 safe_* 转换，禁止直接 int()/float()
- 禁止 float('inf') / float('nan') 出现在可序列化输出中，统一以 None 或有限值替代
"""
from __future__ import annotations

from typing import Any


def safe_int(value: Any, default: int = 0) -> int:
    """安全 int 转换：None/非法值回退到 default（含 bool 与 float 降级）。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全 float 转换：None/NaN/Inf/非法值回退到 default。"""
    try:
        if value is None:
            return default
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def safe_finite(value: Any, default: float = 0.0) -> float:
    """安全有限浮点数：NaN/Inf 回退到 default，确保 JSON 可序列化。"""
    try:
        v = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):  # NaN check: v != v
        return default
    return v


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法：分母为 0 或非有限时返回 default。"""
    try:
        d = float(denominator)
        if d != d or d in (float("inf"), float("-inf")) or d == 0:
            return default
        n = float(numerator)
        if n != n or n in (float("inf"), float("-inf")):
            return default
        return n / d
    except (TypeError, ValueError):
        return default
