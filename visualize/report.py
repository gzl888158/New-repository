"""visualize 组合报表：将多个单图组装为一份可视化报告。

VisualizationReport 接收交易数据（权益曲线 / 交易列表 / 策略统计），
批量生成图表并返回 JSON 安全的报告结构，可直接嵌入日报、复盘报告或 Dashboard。

企业级约束：
- 单图失败不影响其他图，失败项以 error 标记保留
- 整份报告 fail-closed：数据收集异常时返回带 error 的最小报告
- 所有数值经 safe_* 转换，禁止 NaN/Inf 进入 JSON
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger

from visualize.charts import (
    equity_curve_chart,
    hourly_pnl_chart,
    monthly_pnl_chart,
    pnl_distribution_chart,
    strategy_comparison_chart,
    win_loss_chart,
)


class VisualizationReport:
    """交易可视化组合报表。"""

    def __init__(
        self,
        equity_curve: Optional[List[Dict[str, Any]]] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        strategy_stats: Optional[Dict[str, Dict[str, Any]]] = None,
        title: str = "交易可视化报告",
    ):
        self.title = title
        self.equity_curve = equity_curve or []
        self.trades = trades or []
        self.strategy_stats = strategy_stats or {}
        self._charts: Dict[str, Dict[str, Any]] = {}

    def generate(self, include: Optional[List[str]] = None) -> Dict[str, Any]:
        """生成全部图表并返回报告结构。

        Args:
            include: 要包含的图表类型列表；None 表示全部。
                     可选: equity_curve, pnl_distribution, strategy_comparison,
                           win_loss, hourly_pnl, monthly_pnl
        """
        all_types = [
            "equity_curve", "pnl_distribution", "strategy_comparison",
            "win_loss", "hourly_pnl", "monthly_pnl",
        ]
        selected = include if include else all_types

        self._charts = {}
        for ctype in selected:
            try:
                self._charts[ctype] = self._generate_one(ctype)
            except Exception as e:
                logger.error(f"visualize: chart {ctype} generation crashed: "
                             f"{type(e).__name__}: {e}")
                self._charts[ctype] = {
                    "chart_type": ctype,
                    "title": "",
                    "image_base64": "",
                    "image_svg": "",
                    "stats": {},
                    "error": f"{type(e).__name__}: {e}",
                }

        summary = self._build_summary()
        return {
            "title": self.title,
            "generated_at": datetime.now().isoformat(),
            "charts": self._charts,
            "summary": summary,
        }

    def _generate_one(self, ctype: str) -> Dict[str, Any]:
        if ctype == "equity_curve":
            return equity_curve_chart(self.equity_curve)
        if ctype == "pnl_distribution":
            pnls = self._extract_pnl_list()
            return pnl_distribution_chart(pnls)
        if ctype == "strategy_comparison":
            return strategy_comparison_chart(self.strategy_stats, metric="total_pnl")
        if ctype == "win_loss":
            wins, losses = self._count_wins_losses()
            return win_loss_chart(wins, losses)
        if ctype == "hourly_pnl":
            return hourly_pnl_chart(self.trades)
        if ctype == "monthly_pnl":
            return monthly_pnl_chart(self.trades)
        return {
            "chart_type": ctype,
            "title": "",
            "image_base64": "",
            "image_svg": "",
            "stats": {},
            "error": f"unknown chart type: {ctype}",
        }

    def _extract_pnl_list(self) -> List[float]:
        pnls: List[float] = []
        for t in self.trades:
            if not isinstance(t, dict):
                continue
            v = t.get("pnl_usdt", t.get("pnl"))
            try:
                f = float(v) if v is not None else None
            except (TypeError, ValueError):
                f = None
            if f is not None and f == f and f not in (float("inf"), float("-inf")):
                pnls.append(f)
        return pnls

    def _count_wins_losses(self) -> tuple:
        wins = 0
        losses = 0
        for t in self.trades:
            if not isinstance(t, dict):
                continue
            win = t.get("win")
            if win is True:
                wins += 1
            elif win is False:
                losses += 1
            else:
                v = t.get("pnl_usdt", t.get("pnl"))
                try:
                    f = float(v) if v is not None else None
                except (TypeError, ValueError):
                    f = None
                if f is None or f != f or f in (float("inf"), float("-inf")):
                    continue
                if f > 0:
                    wins += 1
                elif f < 0:
                    losses += 1
        return wins, losses

    def _build_summary(self) -> Dict[str, Any]:
        total_charts = len(self._charts)
        successful = sum(1 for c in self._charts.values() if c.get("error") is None and c.get("image_base64"))
        failed = total_charts - successful

        wins, losses = self._count_wins_losses()
        total = wins + losses
        pnls = self._extract_pnl_list()
        total_pnl = round(sum(pnls), 2) if pnls else 0.0

        return {
            "total_charts": total_charts,
            "successful_charts": successful,
            "failed_charts": failed,
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": round(safe_div_local(wins, total), 4),
            "total_pnl": total_pnl,
        }


def safe_div_local(numerator: float, denominator: float, default: float = 0.0) -> float:
    """本地安全除法，避免循环导入 visualize._base。"""
    try:
        d = float(denominator)
        if d == 0 or d != d or d in (float("inf"), float("-inf")):
            return default
        n = float(numerator)
        if n != n or n in (float("inf"), float("-inf")):
            return default
        return n / d
    except (TypeError, ValueError):
        return default


def generate_report(
    equity_curve: Optional[List[Dict[str, Any]]] = None,
    trades: Optional[List[Dict[str, Any]]] = None,
    strategy_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    title: str = "交易可视化报告",
    include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """便捷函数：生成可视化报告。

    失败时返回 {"error": ..., "charts": {}, "summary": {...}}。
    """
    try:
        report = VisualizationReport(
            equity_curve=equity_curve,
            trades=trades,
            strategy_stats=strategy_stats,
            title=title,
        )
        return report.generate(include=include)
    except Exception as e:
        logger.error(f"visualize: generate_report failed: {type(e).__name__}: {e}")
        return {
            "title": title,
            "generated_at": datetime.now().isoformat(),
            "charts": {},
            "summary": {
                "total_charts": 0,
                "successful_charts": 0,
                "failed_charts": 0,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
            },
            "error": f"{type(e).__name__}: {e}",
        }
