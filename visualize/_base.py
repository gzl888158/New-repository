"""visualize 共享工具：类型安全、调色板、matplotlib 后端与图像编码。

企业级约束：
- 强制 Agg 后端，无 GUI 依赖
- 中文字体自动探测（SimHei / Microsoft YaHei / Noto Sans CJK / WenQuanYi）
- 所有数值经 safe_* 转换，NaN/Inf 不进入坐标与输出
- 图像输出统一走 fig_to_base64_png / fig_to_svg / save_figure，
  并保证 Figure 被 close 防止内存泄漏
"""
from __future__ import annotations

import base64
import io
import os
from typing import Any, Optional

from loguru import logger

# ── matplotlib 后端与中文字体 ──────────────────────────────────────
import matplotlib

# 强制 Agg 后端（无显示环境也能渲染）；在 pyplot import 前设置
try:
    matplotlib.use("Agg", force=True)
except Exception as e:  # pragma: no cover
    logger.debug(f"visualize: set Agg backend failed: {type(e).__name__}: {e}")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402


def _setup_chinese_font() -> None:
    """探测并启用系统中文字体，避免中文显示为方块。"""
    candidates = [
        "SimHei",
        "Microsoft YaHei",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
        "PingFang SC",
    ]
    try:
        plt.rcParams["font.sans-serif"] = candidates + plt.rcParams.get("font.sans-serif", [])
        plt.rcParams["axes.unicode_minus"] = False
    except Exception as e:  # pragma: no cover
        logger.debug(f"visualize: chinese font setup failed: {type(e).__name__}: {e}")


_setup_chinese_font()


# ── 调色板（色盲友好 + 交易语义） ───────────────────────────────────
CHART_PALETTE = {
    "profit": "#22c55e",      # 盈利绿
    "loss": "#ef4444",        # 亏损红
    "equity": "#3b82f6",      # 权益蓝
    "drawdown": "#f59e0b",    # 回撤橙
    "win": "#22c55e",
    "loss_bar": "#ef4444",
    "neutral": "#6b7280",
    "accent": "#8b5cf6",
    "grid": "#e5e7eb",
    "text": "#111827",
    "strategies": [
        "#3b82f6", "#22c55e", "#f59e0b", "#8b5cf6",
        "#ec4899", "#14b8a6", "#f97316", "#6366f1",
    ],
}


# ── 类型安全转换（与 eval._base 保持一致，解耦独立） ───────────────
def safe_int(value: Any, default: int = 0) -> int:
    """安全 int 转换：None/非法值回退 default（含 bool 与 float 降级）。"""
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
    """安全 float 转换：None/NaN/Inf/非法值回退 default。"""
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
    """NaN/Inf 回退 default，确保可序列化与可绘图。"""
    try:
        v = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return v


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
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


# ── 图像输出 ───────────────────────────────────────────────────────
def fig_to_base64_png(fig: Figure, dpi: int = 100) -> str:
    """将 Figure 编码为 base64 PNG 字符串（用于 HTML 内嵌 / JSON 传输）。

    失败时返回空字符串，不抛异常。调用后 Figure 会被 close。
    """
    try:
        dpi = safe_int(dpi, 100)
        if dpi <= 0:
            dpi = 100
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        return encoded
    except Exception as e:
        logger.error(f"visualize: fig_to_base64_png failed: {type(e).__name__}: {e}")
        return ""
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


def fig_to_svg(fig: Figure) -> str:
    """将 Figure 编码为 SVG 字符串（矢量、可缩放）。失败返回空字符串。"""
    try:
        buf = io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight", facecolor="white")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"visualize: fig_to_svg failed: {type(e).__name__}: {e}")
        return ""
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


def save_figure(fig: Figure, path: str, fmt: str = "png", dpi: int = 100) -> str:
    """将 Figure 保存到文件，返回绝对路径；失败返回空字符串。

    自动创建父目录。调用后 Figure 会被 close。
    """
    try:
        fmt = (fmt or "png").lower()
        if fmt not in ("png", "svg", "jpg", "jpeg", "pdf"):
            fmt = "png"
        dpi = safe_int(dpi, 100)
        if dpi <= 0:
            dpi = 100
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight", facecolor="white")
        return os.path.abspath(path)
    except Exception as e:
        logger.error(f"visualize: save_figure failed ({path}): {type(e).__name__}: {e}")
        return ""
    finally:
        try:
            plt.close(fig)
        except Exception:
            pass


def new_figure(figsize: tuple = (10, 6), dpi: int = 100) -> Figure:
    """创建新 Figure，尺寸与分辨率经安全转换。"""
    try:
        w = safe_float(figsize[0], 10.0) if len(figsize) > 0 else 10.0
        h = safe_float(figsize[1], 6.0) if len(figsize) > 1 else 6.0
        if w <= 0:
            w = 10.0
        if h <= 0:
            h = 6.0
        dpi = safe_int(dpi, 100)
        if dpi <= 0:
            dpi = 100
        fig, _ = plt.subplots(figsize=(w, h), dpi=dpi)
        return fig
    except Exception as e:
        logger.error(f"visualize: new_figure failed: {type(e).__name__}: {e}")
        # 兜底：返回最小可用 figure
        fig, _ = plt.subplots(figsize=(10, 6))
        return fig
