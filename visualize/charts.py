"""visualize 单图生成器。

每个图表函数返回统一结构：
    {
        "chart_type": "...",
        "title": "...",
        "image_base64": "<base64 png>",   # 失败时为空字符串
        "image_svg": "<svg string>",       # 可选，调用方按需生成
        "stats": {...},                    # 图表衍生统计（JSON 安全）
        "error": None,                     # 失败时为错误描述字符串
    }

企业级约束：
- 所有数值经 safe_* 转换；NaN/Inf 不进入绘图数据
- 空数据 / 畸形数据不崩溃，返回带 error 的结果
- Figure 用完即 close（由 _base 的输出函数保证）
- 输出无 NaN/Inf，保证 JSON 可序列化
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from visualize._base import (
    CHART_PALETTE,
    new_figure,
    safe_div,
    safe_finite,
    safe_float,
    safe_int,
)


def _err(chart_type: str, msg: str) -> Dict[str, Any]:
    """构造错误结果。"""
    return {
        "chart_type": chart_type,
        "title": "",
        "image_base64": "",
        "image_svg": "",
        "stats": {},
        "error": msg,
    }


def _wrap(chart_type: str, title: str, fig, stats: Dict[str, Any]) -> Dict[str, Any]:
    """将 Figure 编码为 base64 并包装结果（fig 会被 close）。"""
    import io
    import base64
    from visualize._base import fig_to_base64_png

    b64 = fig_to_base64_png(fig)
    return {
        "chart_type": chart_type,
        "title": title,
        "image_base64": b64,
        "image_svg": "",
        "stats": stats,
        "error": None if b64 else "failed to encode figure",
    }


# ── 1. 权益曲线 + 回撤（双轴） ─────────────────────────────────────
def equity_curve_chart(
    equity_points: List[Dict[str, Any]],
    title: str = "权益曲线与回撤",
    figsize: Tuple[float, float] = (12, 6),
) -> Dict[str, Any]:
    """绘制权益曲线（左轴 USDT）+ 回撤（右轴 %）。

    Args:
        equity_points: 元素含 timestamp(iso) 与 total_equity；
            兼容 TradeJournal.get_equity_curve() 输出。
    """
    chart_type = "equity_curve"
    if not isinstance(equity_points, list) or not equity_points:
        return _err(chart_type, "equity_points is empty")

    try:
        timestamps: List[datetime] = []
        equities: List[float] = []
        for p in equity_points:
            if not isinstance(p, dict):
                continue
            ts = p.get("timestamp")
            eq = safe_finite(p.get("total_equity"), None)
            if eq is None:
                continue
            try:
                t = datetime.fromisoformat(str(ts)) if ts else datetime.now()
            except Exception:
                t = datetime.now()
            timestamps.append(t)
            equities.append(eq)

        if len(equities) < 2:
            return _err(chart_type, "fewer than 2 valid equity points")

        # 计算回撤序列（%）
        peak = equities[0]
        drawdowns: List[float] = []
        for eq in equities:
            peak = max(peak, eq)
            dd = safe_div(peak - eq, peak, 0.0) * 100.0
            drawdowns.append(dd)

        max_dd = max(drawdowns) if drawdowns else 0.0
        total_return = safe_div(equities[-1] - equities[0], equities[0], 0.0) * 100.0

        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=figsize, dpi=100)
        ax1.plot(timestamps, equities, color=CHART_PALETTE["equity"], linewidth=2, label="权益")
        ax1.set_ylabel("权益 (USDT)", color=CHART_PALETTE["equity"])
        ax1.tick_params(axis="y", labelcolor=CHART_PALETTE["equity"])
        ax1.grid(True, alpha=0.3, color=CHART_PALETTE["grid"])
        ax1.set_title(title)

        ax2 = ax1.twinx()
        ax2.fill_between(timestamps, drawdowns, 0, color=CHART_PALETTE["drawdown"], alpha=0.25)
        ax2.plot(timestamps, drawdowns, color=CHART_PALETTE["drawdown"], linewidth=1, label="回撤")
        ax2.set_ylabel("回撤 (%)", color=CHART_PALETTE["drawdown"])
        ax2.tick_params(axis="y", labelcolor=CHART_PALETTE["drawdown"])
        ax2.invert_yaxis()

        fig.autofmt_xdate()
        fig.tight_layout()

        stats = {
            "points": len(equities),
            "start_equity": round(equities[0], 2),
            "end_equity": round(equities[-1], 2),
            "max_equity": round(max(equities), 2),
            "min_equity": round(min(equities), 2),
            "max_drawdown_pct": round(max_dd, 2),
            "total_return_pct": round(total_return, 2),
        }
        return _wrap(chart_type, title, fig, stats)
    except Exception as e:
        logger.error(f"visualize: equity_curve_chart failed: {type(e).__name__}: {e}")
        return _err(chart_type, f"{type(e).__name__}: {e}")


# ── 2. PnL 分布直方图 ──────────────────────────────────────────────
def pnl_distribution_chart(
    pnl_list: List[float],
    title: str = "单笔盈亏分布",
    figsize: Tuple[float, float] = (10, 6),
    bins: int = 30,
) -> Dict[str, Any]:
    """绘制单笔交易 PnL 分布直方图，标注均值/中位数。"""
    chart_type = "pnl_distribution"
    if not isinstance(pnl_list, list) or not pnl_list:
        return _err(chart_type, "pnl_list is empty")

    try:
        pnls = [safe_finite(v, None) for v in pnl_list]
        pnls = [v for v in pnls if v is not None]
        if not pnls:
            return _err(chart_type, "no finite pnl values")

        bins = safe_int(bins, 30)
        if bins < 2:
            bins = 2
        if bins > 200:
            bins = 200

        arr = np.array(pnls, dtype=float)
        mean_val = float(np.mean(arr))
        median_val = float(np.median(arr))
        std_val = float(np.std(arr)) if len(arr) > 1 else 0.0
        wins = int(np.sum(arr > 0))
        losses = int(np.sum(arr < 0))
        win_rate = safe_div(wins, len(arr), 0.0)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        colors = [CHART_PALETTE["profit"] if v >= 0 else CHART_PALETTE["loss"] for v in arr]
        ax.hist(arr, bins=bins, color=CHART_PALETTE["neutral"], alpha=0.7, edgecolor="white")
        ax.axvline(mean_val, color=CHART_PALETTE["accent"], linestyle="--", linewidth=2,
                   label=f"均值 {mean_val:.2f}")
        ax.axvline(median_val, color=CHART_PALETTE["drawdown"], linestyle=":", linewidth=2,
                   label=f"中位数 {median_val:.2f}")
        ax.axvline(0, color=CHART_PALETTE["text"], linestyle="-", linewidth=1, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("单笔盈亏 (USDT)")
        ax.set_ylabel("频次")
        ax.legend()
        ax.grid(True, alpha=0.3, color=CHART_PALETTE["grid"])
        fig.tight_layout()

        stats = {
            "count": len(arr),
            "wins": wins,
            "losses": losses,
            "win_rate": round(win_rate, 4),
            "mean": round(mean_val, 4),
            "median": round(median_val, 4),
            "std": round(std_val, 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }
        return _wrap(chart_type, title, fig, stats)
    except Exception as e:
        logger.error(f"visualize: pnl_distribution_chart failed: {type(e).__name__}: {e}")
        return _err(chart_type, f"{type(e).__name__}: {e}")


# ── 3. 策略对比柱状图 ───────────────────────────────────────────────
def strategy_comparison_chart(
    strategy_stats: Dict[str, Dict[str, Any]],
    metric: str = "total_pnl",
    title: str = "策略表现对比",
    figsize: Tuple[float, float] = (10, 6),
) -> Dict[str, Any]:
    """按指定 metric 对比各策略的柱状图。

    Args:
        strategy_stats: {strategy_name: {metric: value, ...}}
        metric: 对比字段，如 total_pnl / win_rate / total_trades / sharpe_ratio
    """
    chart_type = "strategy_comparison"
    if not isinstance(strategy_stats, dict) or not strategy_stats:
        return _err(chart_type, "strategy_stats is empty")

    try:
        names: List[str] = []
        values: List[float] = []
        for name, stats in strategy_stats.items():
            if not isinstance(stats, dict):
                continue
            v = safe_finite(stats.get(metric), None)
            if v is None:
                continue
            names.append(str(name))
            values.append(v)

        if not names:
            return _err(chart_type, f"no valid '{metric}' values")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        bar_colors = [CHART_PALETTE["strategies"][i % len(CHART_PALETTE["strategies"])]
                      for i in range(len(names))]
        bars = ax.bar(names, values, color=bar_colors, edgecolor="white")
        ax.set_title(f"{title}（{metric}）")
        ax.set_ylabel(metric)
        ax.grid(True, axis="y", alpha=0.3, color=CHART_PALETTE["grid"])

        # 数值标签
        for bar, v in zip(bars, values):
            y = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, y, f"{v:.2f}",
                    ha="center", va="bottom" if y >= 0 else "top", fontsize=8)
        fig.tight_layout()

        stats = {
            "metric": metric,
            "strategies": names,
            "best": names[int(np.argmax(values))] if values else "",
            "worst": names[int(np.argmin(values))] if values else "",
            "mean": round(float(np.mean(values)), 4),
        }
        return _wrap(chart_type, title, fig, stats)
    except Exception as e:
        logger.error(f"visualize: strategy_comparison_chart failed: {type(e).__name__}: {e}")
        return _err(chart_type, f"{type(e).__name__}: {e}")


# ── 4. 胜负饼图 ─────────────────────────────────────────────────────
def win_loss_chart(
    wins: int,
    losses: int,
    title: str = "胜负分布",
    figsize: Tuple[float, float] = (7, 7),
) -> Dict[str, Any]:
    """绘制胜负占比饼图。"""
    chart_type = "win_loss"
    wins = safe_int(wins, 0)
    losses = safe_int(losses, 0)
    if wins < 0:
        wins = 0
    if losses < 0:
        losses = 0
    total = wins + losses
    if total <= 0:
        return _err(chart_type, "no trades (wins+losses <= 0)")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        sizes = [wins, losses]
        labels = [f"盈利 {wins}", f"亏损 {losses}"]
        colors = [CHART_PALETTE["win"], CHART_PALETTE["loss_bar"]]
        ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
               startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title(f"{title}（胜率 {safe_div(wins, total, 0.0):.1%}）")
        fig.tight_layout()

        stats = {
            "wins": wins,
            "losses": losses,
            "total": total,
            "win_rate": round(safe_div(wins, total, 0.0), 4),
        }
        return _wrap(chart_type, title, fig, stats)
    except Exception as e:
        logger.error(f"visualize: win_loss_chart failed: {type(e).__name__}: {e}")
        return _err(chart_type, f"{type(e).__name__}: {e}")


# ── 5. 时段盈亏柱状图 ───────────────────────────────────────────────
def hourly_pnl_chart(
    trades: List[Dict[str, Any]],
    title: str = "各时段盈亏分布",
    figsize: Tuple[float, float] = (12, 6),
) -> Dict[str, Any]:
    """按小时（0-23）聚合 PnL 柱状图。

    Args:
        trades: 元素含 exit_time(iso) 与 pnl_usdt（或 pnl）。
    """
    chart_type = "hourly_pnl"
    if not isinstance(trades, list) or not trades:
        return _err(chart_type, "trades is empty")

    try:
        hourly: Dict[int, float] = {h: 0.0 for h in range(24)}
        hourly_count: Dict[int, int] = {h: 0 for h in range(24)}
        for t in trades:
            if not isinstance(t, dict):
                continue
            pnl = safe_finite(t.get("pnl_usdt", t.get("pnl")), None)
            if pnl is None:
                continue
            ts = t.get("exit_time") or t.get("timestamp")
            try:
                hour = datetime.fromisoformat(str(ts)).hour if ts else 0
            except Exception:
                hour = 0
            hourly[hour] += pnl
            hourly_count[hour] += 1

        hours = list(range(24))
        values = [hourly[h] for h in hours]
        counts = [hourly_count[h] for h in hours]

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        bar_colors = [CHART_PALETTE["profit"] if v >= 0 else CHART_PALETTE["loss"] for v in values]
        ax.bar(hours, values, color=bar_colors, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("小时 (0-23)")
        ax.set_ylabel("盈亏 (USDT)")
        ax.set_xticks(hours)
        ax.axhline(0, color=CHART_PALETTE["text"], linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3, color=CHART_PALETTE["grid"])
        fig.tight_layout()

        nonzero = [v for v in values if v != 0]
        stats = {
            "best_hour": int(np.argmax(values)),
            "best_hour_pnl": round(float(np.max(values)), 2),
            "worst_hour": int(np.argmin(values)),
            "worst_hour_pnl": round(float(np.min(values)), 2),
            "total_pnl": round(sum(values), 2),
            "active_hours": sum(1 for c in counts if c > 0),
        }
        return _wrap(chart_type, title, fig, stats)
    except Exception as e:
        logger.error(f"visualize: hourly_pnl_chart failed: {type(e).__name__}: {e}")
        return _err(chart_type, f"{type(e).__name__}: {e}")


# ── 6. 月度盈亏热力图 ───────────────────────────────────────────────
def monthly_pnl_chart(
    trades: List[Dict[str, Any]],
    title: str = "月度盈亏",
    figsize: Tuple[float, float] = (12, 5),
) -> Dict[str, Any]:
    """按月份聚合 PnL 柱状图（近 12 个月）。"""
    chart_type = "monthly_pnl"
    if not isinstance(trades, list) or not trades:
        return _err(chart_type, "trades is empty")

    try:
        from collections import defaultdict
        monthly: Dict[str, float] = defaultdict(float)
        for t in trades:
            if not isinstance(t, dict):
                continue
            pnl = safe_finite(t.get("pnl_usdt", t.get("pnl")), None)
            if pnl is None:
                continue
            ts = t.get("exit_time") or t.get("timestamp")
            try:
                key = datetime.fromisoformat(str(ts)).strftime("%Y-%m") if ts else "unknown"
            except Exception:
                key = "unknown"
            monthly[key] += pnl

        months = sorted(monthly.keys())[-12:]
        values = [safe_finite(monthly[m], 0.0) for m in months]

        if not months:
            return _err(chart_type, "no valid monthly data")

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=figsize, dpi=100)
        bar_colors = [CHART_PALETTE["profit"] if v >= 0 else CHART_PALETTE["loss"] for v in values]
        ax.bar(months, values, color=bar_colors, edgecolor="white")
        ax.set_title(title)
        ax.set_ylabel("盈亏 (USDT)")
        ax.axhline(0, color=CHART_PALETTE["text"], linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3, color=CHART_PALETTE["grid"])
        fig.autofmt_xdate()
        fig.tight_layout()

        stats = {
            "months": months,
            "total_pnl": round(sum(values), 2),
            "positive_months": sum(1 for v in values if v > 0),
            "negative_months": sum(1 for v in values if v < 0),
        }
        return _wrap(chart_type, title, fig, stats)
    except Exception as e:
        logger.error(f"visualize: monthly_pnl_chart failed: {type(e).__name__}: {e}")
        return _err(chart_type, f"{type(e).__name__}: {e}")
