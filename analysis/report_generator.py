"""
智能报表生成系统
支持日报、周报、自定义报表的自动生成和导出。
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from loguru import logger
import json


class ReportGenerator:
    """智能报表生成器"""

    REPORT_TYPES = ["daily", "weekly", "custom"]
    
    def __init__(self, analysis_engine):
        self._analysis_engine = analysis_engine
        logger.info("ReportGenerator initialized")

    def generate_daily_report(self, date: str = None) -> Dict[str, Any]:
        """生成日报"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        start_date = date
        end_date = date
        
        return self._generate_report(
            report_type="daily",
            start_date=start_date,
            end_date=end_date,
            title=f"交易日报 - {date}"
        )

    def generate_weekly_report(self, week_offset: int = 0) -> Dict[str, Any]:
        """生成周报"""
        today = datetime.now()
        current_week_start = today - timedelta(days=today.weekday())
        
        if week_offset != 0:
            current_week_start -= timedelta(weeks=week_offset)
        
        start_date = current_week_start.strftime("%Y-%m-%d")
        end_date = (current_week_start + timedelta(days=6)).strftime("%Y-%m-%d")
        
        return self._generate_report(
            report_type="weekly",
            start_date=start_date,
            end_date=end_date,
            title=f"交易周报 - {start_date} 至 {end_date}"
        )

    def generate_custom_report(self, start_date: str, end_date: str, 
                               title: str = "自定义报表") -> Dict[str, Any]:
        """生成自定义报表"""
        return self._generate_report(
            report_type="custom",
            start_date=start_date,
            end_date=end_date,
            title=title
        )

    def _generate_report(self, report_type: str, start_date: str, 
                         end_date: str, title: str) -> Dict[str, Any]:
        """生成报表核心逻辑"""
        strategy_performance = self._analysis_engine.analyze_strategy_performance(
            start_date=start_date,
            end_date=end_date
        )
        
        trading_stats = self._analysis_engine.analyze_trading_statistics(
            start_date=start_date,
            end_date=end_date
        )
        
        insights = self._generate_insights(strategy_performance, trading_stats)
        
        report = {
            "title": title,
            "report_type": report_type,
            "start_date": start_date,
            "end_date": end_date,
            "generated_at": datetime.now().isoformat(),
            "strategy_performance": strategy_performance,
            "trading_statistics": trading_stats,
            "insights": insights,
            "summary": self._generate_summary(strategy_performance, trading_stats, insights),
        }
        
        return report

    def _generate_insights(self, strategy_performance: Dict[str, Any], 
                           trading_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成洞察分析"""
        insights = []
        
        overview = trading_stats.get("overview", {})
        daily_stats = trading_stats.get("daily_stats", {})
        hourly_stats = trading_stats.get("hourly_stats", {})
        symbol_stats = trading_stats.get("symbol_stats", {})
        
        total_pnl = overview.get("total_pnl", 0)
        win_rate = overview.get("win_rate", 0)
        total_fees = overview.get("total_fees", 0)
        net_pnl = overview.get("net_pnl", 0)
        
        if total_pnl > 0:
            insights.append({
                "type": "positive",
                "title": "盈利表现",
                "message": f"期间总盈利 {total_pnl:.2f} USDT，净利润 {net_pnl:.2f} USDT",
                "metric": total_pnl,
            })
        else:
            insights.append({
                "type": "warning",
                "title": "亏损预警",
                "message": f"期间总亏损 {abs(total_pnl):.2f} USDT，需要关注策略表现",
                "metric": total_pnl,
            })
        
        if win_rate > 0.6:
            insights.append({
                "type": "positive",
                "title": "胜率优秀",
                "message": f"胜率 {win_rate:.1%}，表现优于平均水平",
                "metric": win_rate,
            })
        elif win_rate < 0.4:
            insights.append({
                "type": "warning",
                "title": "胜率偏低",
                "message": f"胜率 {win_rate:.1%}，建议优化策略",
                "metric": win_rate,
            })
        
        if total_fees > abs(total_pnl) * 0.3:
            insights.append({
                "type": "info",
                "title": "手续费占比较高",
                "message": f"手续费 {total_fees:.2f} USDT，占盈亏比 {total_fees/max(abs(total_pnl), 0.01):.1%}",
                "metric": total_fees / max(abs(total_pnl), 0.01),
            })
        
        if "comparison" in strategy_performance:
            comparison = strategy_performance["comparison"]
            best_pnl = comparison.get("best_pnl", {})
            
            insights.append({
                "type": "positive",
                "title": "最佳策略",
                "message": f"{best_pnl.get('strategy', '')} 表现最佳，盈利 {best_pnl.get('value', 0):.2f} USDT",
                "metric": best_pnl.get("value", 0),
            })
        
        if daily_stats.get("best_day"):
            best_day, best_day_data = daily_stats["best_day"]
            if best_day:
                insights.append({
                    "type": "positive",
                    "title": "最佳交易日",
                    "message": f"{best_day} 盈利 {best_day_data.get('pnl', 0):.2f} USDT",
                    "metric": best_day_data.get("pnl", 0),
                })
        
        if daily_stats.get("worst_day"):
            worst_day, worst_day_data = daily_stats["worst_day"]
            if worst_day and worst_day_data.get("pnl", 0) < -10:
                insights.append({
                    "type": "warning",
                    "title": "最差交易日",
                    "message": f"{worst_day} 亏损 {abs(worst_day_data.get('pnl', 0)):.2f} USDT",
                    "metric": worst_day_data.get("pnl", 0),
                })
        
        if hourly_stats.get("best_hour"):
            best_hour, best_hour_data = hourly_stats["best_hour"]
            if best_hour is not None:
                insights.append({
                    "type": "info",
                    "title": "最佳交易时段",
                    "message": f"UTC {best_hour}:00 时段表现最佳，盈利 {best_hour_data.get('pnl', 0):.2f} USDT",
                    "metric": best_hour_data.get("pnl", 0),
                })
        
        if symbol_stats.get("top_performer"):
            top_symbol, top_data = symbol_stats["top_performer"]
            if top_symbol:
                insights.append({
                    "type": "positive",
                    "title": "最佳交易标的",
                    "message": f"{top_symbol} 盈利 {top_data.get('pnl', 0):.2f} USDT",
                    "metric": top_data.get("pnl", 0),
                })
        
        if symbol_stats.get("worst_performer"):
            worst_symbol, worst_data = symbol_stats["worst_performer"]
            if worst_symbol and worst_data.get("pnl", 0) < -10:
                insights.append({
                    "type": "warning",
                    "title": "最差交易标的",
                    "message": f"{worst_symbol} 亏损 {abs(worst_data.get('pnl', 0)):.2f} USDT",
                    "metric": worst_data.get("pnl", 0),
                })
        
        strategies = {k: v for k, v in strategy_performance.items() if k != "comparison"}
        for strategy, perf in strategies.items():
            max_dd = perf.get("max_drawdown", 0)
            if max_dd > 0.15:
                insights.append({
                    "type": "warning",
                    "title": f"{strategy} 回撤预警",
                    "message": f"{strategy} 最大回撤 {max_dd:.1%}，超过预警阈值",
                    "metric": max_dd,
                })
            
            sharpe = perf.get("sharpe_ratio", 0)
            if sharpe > 1.0:
                insights.append({
                    "type": "positive",
                    "title": f"{strategy} 风险调整收益优秀",
                    "message": f"{strategy} 夏普比率 {sharpe:.2f}，风险调整收益良好",
                    "metric": sharpe,
                })
        
        return insights

    def _generate_summary(self, strategy_performance: Dict[str, Any], 
                          trading_stats: Dict[str, Any], 
                          insights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成报表摘要"""
        overview = trading_stats.get("overview", {})
        
        positive_insights = [i for i in insights if i["type"] == "positive"]
        warning_insights = [i for i in insights if i["type"] == "warning"]
        info_insights = [i for i in insights if i["type"] == "info"]
        
        return {
            "total_trades": overview.get("total_trades", 0),
            "total_pnl": round(overview.get("total_pnl", 0), 2),
            "net_pnl": round(overview.get("net_pnl", 0), 2),
            "win_rate": round(overview.get("win_rate", 0), 4),
            "total_fees": round(overview.get("total_fees", 0), 2),
            "positive_insights": len(positive_insights),
            "warning_insights": len(warning_insights),
            "info_insights": len(info_insights),
            "strategy_count": len([k for k in strategy_performance.keys() if k != "comparison"]),
        }

    def export_report(self, report: Dict[str, Any], format: str = "json", 
                      file_path: str = None) -> str:
        """导出报表"""
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"reports/{report['report_type']}_{timestamp}.{format}"
        
        try:
            if format == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            elif format == "txt":
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self._format_report_as_text(report))
            
            logger.info(f"Report exported to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return ""

    def _format_report_as_text(self, report: Dict[str, Any]) -> str:
        """格式化报表为文本"""
        lines = []
        lines.append("=" * 60)
        lines.append(report["title"])
        lines.append("=" * 60)
        lines.append(f"生成时间: {report['generated_at']}")
        lines.append(f"统计周期: {report['start_date']} 至 {report['end_date']}")
        lines.append("")
        
        summary = report.get("summary", {})
        lines.append("【摘要】")
        lines.append(f"  总交易数: {summary.get('total_trades', 0)}")
        lines.append(f"  总盈亏: {summary.get('total_pnl', 0):.2f} USDT")
        lines.append(f"  净利润: {summary.get('net_pnl', 0):.2f} USDT")
        lines.append(f"  胜率: {summary.get('win_rate', 0):.1%}")
        lines.append(f"  手续费: {summary.get('total_fees', 0):.2f} USDT")
        lines.append("")
        
        lines.append("【策略表现】")
        strategies = {k: v for k, v in report.get("strategy_performance", {}).items() if k != "comparison"}
        for strategy, perf in strategies.items():
            lines.append(f"  {strategy}:")
            lines.append(f"    交易数: {perf.get('total_trades', 0)}")
            lines.append(f"    盈亏: {perf.get('total_pnl', 0):.2f} USDT")
            lines.append(f"    胜率: {perf.get('win_rate', 0):.1%}")
            lines.append(f"    最大回撤: {perf.get('max_drawdown', 0):.1%}")
            lines.append(f"    夏普比率: {perf.get('sharpe_ratio', 0):.2f}")
        lines.append("")
        
        lines.append("【洞察分析】")
        for insight in report.get("insights", []):
            prefix = {"positive": "+", "warning": "!", "info": "*"}.get(insight["type"], "?")
            lines.append(f"  {prefix} {insight['title']}: {insight['message']}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)
