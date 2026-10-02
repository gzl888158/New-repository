"""live 包内部共享工具：类型安全 + 数值清洗 + JSON 安全（与 eval/visualize/utils 保持一致）。

不依赖 eval/visualize，避免循环导入。
"""
from __future__ import annotations

import json
import math
from typing import Any


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全转换为 float，None / 非法字符串 / NaN / Inf 一律回退 default。"""
    try:
        if value is None:
            return float(default)
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    """安全转换为 int，非法输入回退 default。"""
    try:
        if isinstance(value, bool):
            return int(value)
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(safe_float(value, float(default)))
        except (TypeError, ValueError):
            return int(default)


def safe_div(numerator: Any, denominator: Any, default: float = 0.0) -> float:
    """安全除法：除零、NaN、Inf 回退 default。"""
    n = safe_float(numerator, default)
    d = safe_float(denominator, default)
    if d == 0.0:
        return float(default)
    result = n / d
    if math.isnan(result) or math.isinf(result):
        return float(default)
    return result


def safe_finite(value: float, default: float = 0.0) -> float:
    """返回有限值，否则 default。"""
    try:
        v = float(value)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except (TypeError, ValueError):
        return float(default)


def _to_finite(value: Any, default: float = 0.0) -> float:
    return safe_finite(value, default)


def _sanitize_for_json(obj: Any) -> Any:
    """递归清洗对象，确保可被 json.dumps 序列化且不含 NaN/Inf。"""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(v) for v in obj]
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()
        except Exception:
            return str(obj)
    try:
        s = str(obj)
        if s.startswith("<") and s.endswith(">"):
            return None
        return s
    except Exception:
        return None


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """json.dumps 的安全包装：先清洗再序列化，任何异常返回 "{}"。"""
    try:
        cleaned = _sanitize_for_json(obj)
        return json.dumps(cleaned, **kwargs)
    except Exception:
        return "{}"


__all__ = [
    "safe_float", "safe_int", "safe_div", "safe_finite",
    "_to_finite", "_sanitize_for_json", "safe_json_dumps",
]
