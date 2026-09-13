"""
交易审核器 (Trade Auditor)
生产级模块 - 完整审核链路

功能：
1. 开仓前审核：成本、风险、信号质量、市场状态
2. 平仓时审核：盈亏验证、止损合理性
3. 持仓中审核：风险暴露、资金费率、保证金
4. 交易后审计：历史分析、账单生成
5. 计划性优化：基于历史数据自动调整参数
"""

import logging
import json
import os
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)


class AuditResult(Enum):
    PASS = "pass"
    WARN = "warn"
    BLOCK = "block"
    REDUCE = "reduce"


@dataclass
class AuditRecord:
    """审核记录"""
    audit_id: str
    timestamp: datetime
    symbol: str
    strategy_name: str
    signal_type: str
    direction: str
    result: AuditResult
    reason: str
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)


class TradeAuditor:
    """交易审核器 - 全链路审核"""

    def __init__(self, config: Optional[Dict] = None, data_dir: str = "./data", sqlite_storage=None):
        self._config = config or {}
        self._data_dir = data_dir
        self._sqlite_storage = sqlite_storage  # P14: 注入SQLite存储用于日报生成
        self._audit_log: List[AuditRecord] = []
        self._daily_stats: Dict[str, Dict] = defaultdict(lambda: {
            "total_audited": 0, "passed": 0, "blocked": 0, "warned": 0, "reduced": 0,
        })
        self._audit_counter = 0

        # 风险参数
        self.max_daily_loss_pct = self._config.get("max_daily_loss_pct", 0.05)  # 日最大亏损5%
        self.max_consecutive_losses = self._config.get("max_consecutive_losses", 5)
        self.min_risk_reward_ratio = self._config.get("min_risk_reward_ratio", 1.5)  # 最小盈亏比
        self.max_positions_per_symbol = self._config.get("max_positions_per_symbol", 2)

        # 统计
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._consecutive_losses: int = 0
        self._today = datetime.now().date()

        # 交易记录（用于账单生成）
        self._trade_ledger: List[Dict] = []

    def _check_daily_reset(self):
        """日重置"""
        today = datetime.now().date()
        if today != self._today:
            self._daily_pnl = 0.0
            self._daily_trades = 0
            self._consecutive_losses = 0
            self._today = today

    # ========== 开仓前审核 ==========

    def pre_open_audit(
        self,
        symbol: str,
        strategy_name: str,
        signal_type: str,
        direction: str,
        price: float,
        quantity: float,
        leverage: float,
        confidence: float,
        cost_analysis: Optional[Any] = None,  # TradeCostBreakdown
        market_regime: Optional[str] = None,
    ) -> Tuple[AuditResult, str, Dict]:
        """开仓前全面审核

        Returns:
            (审核结果, 原因, 审核详情)
        """
        self._check_daily_reset()
        self._audit_counter += 1
        audit_id = f"AUDIT-{self._audit_counter:06d}"
        checks = {}
        details = {}

        # 检查1: 日亏损限制
        if self._daily_pnl < -self.max_daily_loss_pct * 100:
            details["daily_loss"] = self._daily_pnl
            self._log_audit(audit_id, symbol, strategy_name, signal_type, direction,
                          AuditResult.BLOCK, "日亏损超过限制", checks)
            return AuditResult.BLOCK, f"日亏损 {self._daily_pnl:.2f} 超过限制", details

        # 检查2: 连续亏损限制
        if self._consecutive_losses >= self.max_consecutive_losses:
            self._log_audit(audit_id, symbol, strategy_name, signal_type, direction,
                          AuditResult.BLOCK, f"连续亏损 {self._consecutive_losses} 次", checks)
            return AuditResult.BLOCK, f"连续亏损 {self._consecutive_losses} 次，暂停开仓", details

        checks["daily_loss"] = True
        checks["consecutive_losses"] = True

        # 检查3: 成本分析
        if cost_analysis and hasattr(cost_analysis, 'is_profitable'):
            checks["cost_analysis"] = cost_analysis.is_profitable
            if not cost_analysis.is_profitable:
                details["cost"] = {
                    "total_cost": cost_analysis.total_cost,
                    "total_cost_pct": cost_analysis.total_cost_pct,
                    "min_required_move": cost_analysis.min_required_move_pct,
                }
                self._log_audit(audit_id, symbol, strategy_name, signal_type, direction,
                              AuditResult.BLOCK, "成本分析显示无法盈利", checks)
                return AuditResult.BLOCK, "交易成本过高，无法盈利", details
        else:
            checks["cost_analysis"] = True

        # 检查4: 风险收益比
        if confidence > 0:
            estimated_rr = confidence / (1 - confidence) if confidence < 1 else 10
            checks["risk_reward"] = estimated_rr >= self.min_risk_reward_ratio
            if not checks["risk_reward"]:
                details["risk_reward"] = estimated_rr
                self._log_audit(audit_id, symbol, strategy_name, signal_type, direction,
                              AuditResult.WARN, f"风险收益比过低 {estimated_rr:.1f}", checks)
                return AuditResult.WARN, f"风险收益比 {estimated_rr:.1f} < {self.min_risk_reward_ratio}", details
        else:
            checks["risk_reward"] = True

        # 检查5: 信号质量
        checks["signal_quality"] = confidence >= 0.35
        if not checks["signal_quality"]:
            self._log_audit(audit_id, symbol, strategy_name, signal_type, direction,
                          AuditResult.BLOCK, f"信号质量过低 {confidence:.2f}", checks)
            return AuditResult.BLOCK, f"信号质量 {confidence:.2f} < 0.35", details

        # 检查6: 市场状态
        if market_regime:
            high_risk_regimes = ["high_volatility", "extreme"]
            checks["market_regime"] = market_regime not in high_risk_regimes
            if not checks["market_regime"]:
                details["market_regime"] = market_regime
                self._log_audit(audit_id, symbol, strategy_name, signal_type, direction,
                              AuditResult.REDUCE, f"高风险市场状态: {market_regime}", checks)
                return AuditResult.REDUCE, f"高风险市场 {market_regime}，建议减仓", details
        else:
            checks["market_regime"] = True

        # 全部通过
        self._log_audit(audit_id, symbol, strategy_name, signal_type, direction,
                      AuditResult.PASS, "审核通过", checks)
        return AuditResult.PASS, "审核通过", details

    # ========== 平仓时审核 ==========

    def pre_close_audit(
        self,
        symbol: str,
        strategy_name: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        pos_side: str,
        close_reason: str,
        expected_pnl: float,
    ) -> Tuple[AuditResult, str, Dict]:
        """平仓前审核"""
        self._audit_counter += 1
        audit_id = f"AUDIT-C{self._audit_counter:06d}"
        checks = {}
        details = {"expected_pnl": expected_pnl}

        # 止损平仓无条件通过
        if "stop_loss" in close_reason.lower() or "liquidation" in close_reason.lower():
            self._log_audit(audit_id, symbol, strategy_name, close_reason, pos_side,
                          AuditResult.PASS, "止损平仓无条件通过", checks)
            return AuditResult.PASS, "止损平仓", details

        # 止盈平仓检查是否真的盈利
        if "take_profit" in close_reason.lower() or "tp" in close_reason.lower():
            if expected_pnl <= 0:
                self._log_audit(audit_id, symbol, strategy_name, close_reason, pos_side,
                              AuditResult.WARN, f"止盈但实际亏损 {expected_pnl:.4f}", checks)
                return AuditResult.WARN, f"止盈信号但预期亏损 {expected_pnl:.4f}", details

        self._log_audit(audit_id, symbol, strategy_name, close_reason, pos_side,
                      AuditResult.PASS, "平仓审核通过", checks)
        return AuditResult.PASS, "平仓审核通过", details

    # ========== 交易后记录 ==========

    def record_trade(
        self,
        symbol: str,
        strategy_name: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        fee: float,
        entry_time: datetime,
        exit_time: datetime,
        details: Optional[Dict] = None,
    ):
        """记录已完成的交易"""
        self._check_daily_reset()
        self._daily_trades += 1
        self._daily_pnl += pnl

        if pnl <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        trade_record = {
            "trade_id": len(self._trade_ledger) + 1,
            "symbol": symbol,
            "strategy": strategy_name,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": round(pnl, 6),
            "pnl_pct": round(pnl_pct, 6),
            "fee": round(fee, 6),
            "entry_time": entry_time.isoformat(),
            "exit_time": exit_time.isoformat(),
            "hold_duration_hours": round((exit_time - entry_time).total_seconds() / 3600, 2),
            "details": details or {},
        }
        self._trade_ledger.append(trade_record)

    # ========== 账单生成 ==========

    def generate_daily_report(self) -> Dict[str, Any]:
        """生成日账单
        
        P14: 优先从SQLite数据库读取交易记录，解决日报0交易问题。
        如果数据库不可用，fallback到内存中的_trade_ledger。
        """
        self._check_daily_reset()
        today = datetime.now().date()

        # P14: 从SQLite数据库读取今天的交易记录
        today_trades = []
        if self._sqlite_storage:
            try:
                from sqlalchemy import text
                conn = self._sqlite_storage.get_connection()
                today_str = today.isoformat()
                result = conn.execute(text(
                    "SELECT * FROM trade_records WHERE status = 'closed' "
                    "AND DATE(close_time) = :today ORDER BY close_time DESC"
                ), {"today": today_str})
                rows = result.fetchall()
                if rows:
                    columns = result.keys()
                    for row in rows:
                        trade = dict(zip(columns, row))
                        today_trades.append({
                            "trade_id": trade.get("id", ""),
                            "symbol": trade.get("symbol", ""),
                            "strategy": trade.get("strategy_name", ""),
                            "direction": trade.get("side", ""),
                            "entry_price": float(trade.get("price", 0) or 0),
                            "exit_price": float(trade.get("filled_price", 0) or 0),
                            "quantity": float(trade.get("quantity", 0) or 0),
                            "pnl": float(trade.get("pnl", 0) or 0),
                            "pnl_pct": float(trade.get("pnl_percent", 0) or 0),
                            "fee": float(trade.get("fees", 0) or 0),
                            "entry_time": str(trade.get("create_time", "")),
                            "exit_time": str(trade.get("close_time", "")),
                            "hold_duration_hours": 0,
                            "details": {},
                        })
                    logger.info(
                        f"P14: Daily report loaded {len(today_trades)} trades from SQLite for {today_str}"
                    )
            except Exception as e:
                logger.warning(f"P14: Failed to read trades from SQLite: {e}, falling back to memory ledger")
        
        # Fallback: 从内存_trade_ledger读取
        if not today_trades:
            today_trades = [
                t for t in self._trade_ledger
                if datetime.fromisoformat(t["entry_time"]).date() == today
            ]

        wins = [t for t in today_trades if t["pnl"] > 0]
        losses = [t for t in today_trades if t["pnl"] <= 0]

        total_pnl = sum(t["pnl"] for t in today_trades)
        total_fee = sum(t["fee"] for t in today_trades)

        # 按策略分组
        by_strategy = defaultdict(lambda: {"count": 0, "pnl": 0.0, "fee": 0.0, "wins": 0, "losses": 0})
        for t in today_trades:
            s = by_strategy[t["strategy"]]
            s["count"] += 1
            s["pnl"] += t["pnl"]
            s["fee"] += t["fee"]
            if t["pnl"] > 0:
                s["wins"] += 1
            else:
                s["losses"] += 1

        # 按币种分组
        by_symbol = defaultdict(lambda: {"count": 0, "pnl": 0.0})
        for t in today_trades:
            s = by_symbol[t["symbol"]]
            s["count"] += 1
            s["pnl"] += t["pnl"]

        return {
            "date": today.isoformat(),
            "summary": {
                "total_trades": len(today_trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(today_trades) if today_trades else 0,
                "total_pnl": round(total_pnl, 6),
                "net_pnl": round(total_pnl, 6),
                "total_fee": round(total_fee, 6),
                "best_trade": max(today_trades, key=lambda t: t["pnl"]) if today_trades else None,
                "worst_trade": min(today_trades, key=lambda t: t["pnl"]) if today_trades else None,
            },
            "by_strategy": {k: dict(v) for k, v in by_strategy.items()},
            "by_symbol": {k: dict(v) for k, v in by_symbol.items()},
            "trades": today_trades,
        }

    def generate_history_analysis(self, days: int = 30) -> Dict[str, Any]:
        """生成历史分析报告
        
        P14: 优先从SQLite数据库读取交易记录。
        """
        # P14: 从SQLite数据库读取历史交易记录
        recent_trades = []
        cutoff = datetime.now() - timedelta(days=days)  # 提前定义，避免后续引用错误
        if self._sqlite_storage:
            try:
                from sqlalchemy import text
                conn = self._sqlite_storage.get_connection()
                cutoff_str = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
                result = conn.execute(text(
                    "SELECT * FROM trade_records WHERE status = 'closed' "
                    "AND close_time >= :cutoff ORDER BY close_time DESC"
                ), {"cutoff": cutoff_str})
                rows = result.fetchall()
                if rows:
                    columns = result.keys()
                    for row in rows:
                        trade = dict(zip(columns, row))
                        recent_trades.append({
                            "trade_id": trade.get("id", ""),
                            "symbol": trade.get("symbol", ""),
                            "strategy": trade.get("strategy_name", ""),
                            "direction": trade.get("side", ""),
                            "entry_price": float(trade.get("price", 0) or 0),
                            "exit_price": float(trade.get("filled_price", 0) or 0),
                            "quantity": float(trade.get("quantity", 0) or 0),
                            "pnl": float(trade.get("pnl", 0) or 0),
                            "pnl_pct": float(trade.get("pnl_percent", 0) or 0),
                            "fee": float(trade.get("fees", 0) or 0),
                            "entry_time": str(trade.get("create_time", "")),
                            "exit_time": str(trade.get("close_time", "")),
                            "hold_duration_hours": 0,
                            "details": {},
                        })
            except Exception as e:
                logger.warning(f"P14: Failed to read history from SQLite: {e}")
        
        # Fallback: 从内存_trade_ledger读取
        if not recent_trades:
            cutoff = datetime.now() - timedelta(days=days)
            recent_trades = [
                t for t in self._trade_ledger
                if datetime.fromisoformat(t["entry_time"]) >= cutoff
            ]

        if not recent_trades:
            return {"error": f"最近 {days} 天无交易记录"}

        # 日统计
        daily_pnl = defaultdict(float)
        for t in recent_trades:
            day = datetime.fromisoformat(t["entry_time"]).date().isoformat()
            daily_pnl[day] += t["pnl"]

        # 盈亏分析
        total_pnl = sum(t["pnl"] for t in recent_trades)
        wins = [t for t in recent_trades if t["pnl"] > 0]
        losses = [t for t in recent_trades if t["pnl"] <= 0]
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0

        # 最大回撤
        cumulative = 0
        peak = 0
        max_drawdown = 0
        for t in sorted(recent_trades, key=lambda x: x["entry_time"]):
            cumulative += t["pnl"]
            peak = max(peak, cumulative)
            max_drawdown = min(max_drawdown, cumulative - peak)

        # 持续改进建议
        suggestions = self._generate_optimization_suggestions(recent_trades)

        return {
            "period": f"{days} days",
            "from": cutoff.isoformat(),
            "to": datetime.now().isoformat(),
            "summary": {
                "total_trades": len(recent_trades),
                "total_pnl": round(total_pnl, 6),
                "win_rate": len(wins) / len(recent_trades) if recent_trades else 0,
                "avg_win": round(avg_win, 6),
                "avg_loss": round(avg_loss, 6),
                "profit_factor": abs(sum(t["pnl"] for t in wins) / sum(t["pnl"] for t in losses)) if losses else float("inf"),
                "max_drawdown": round(max_drawdown, 6),
                "total_fee": round(sum(t["fee"] for t in recent_trades), 6),
            },
            "daily_pnl": dict(daily_pnl),
            "suggestions": suggestions,
        }

    def _generate_optimization_suggestions(self, trades: List[Dict]) -> List[str]:
        """生成优化建议"""
        suggestions = []

        if not trades:
            return suggestions

        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / len(trades) if trades else 0

        # 建议1: 胜率过低
        if win_rate < 0.35:
            suggestions.append(f"胜率过低 ({win_rate:.1%})，建议提高信号质量阈值至0.4+")

        # 建议2: 手续费占比过高
        total_fee = sum(t["fee"] for t in trades)
        total_pnl = sum(t["pnl"] for t in wins)
        if total_pnl > 0 and total_fee / total_pnl > 0.3:
            suggestions.append(f"手续费占比过高 ({total_fee/total_pnl:.1%})，建议减少交易频率或增加单笔交易量")

        # 建议3: 平均亏损过大
        avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0
        avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0
        if avg_win > 0 and abs(avg_loss) > avg_win * 1.5:
            suggestions.append(f"平均亏损 ({avg_loss:.4f}) 大于平均盈利 ({avg_win:.4f})，建议收紧止损")

        # 建议4: 持仓时间过短
        short_trades = [t for t in trades if t.get("hold_duration_hours", 0) < 0.5]
        if len(short_trades) > len(trades) * 0.3:
            suggestions.append(f"短期交易占比过高 ({len(short_trades)/len(trades):.0%})，可能是过度交易")

        # 建议5: 特定时段亏损
        hour_losses = defaultdict(lambda: {"count": 0, "pnl": 0.0})
        for t in losses:
            hour = datetime.fromisoformat(t["entry_time"]).hour
            hour_losses[hour]["count"] += 1
            hour_losses[hour]["pnl"] += t["pnl"]
        worst_hours = sorted(hour_losses.items(), key=lambda x: x[1]["pnl"])[:3]
        for hour, stats in worst_hours:
            if stats["count"] >= 3 and stats["pnl"] < -0.1:
                suggestions.append(f"时段 {hour}:00 亏损严重 ({stats['pnl']:.2f}, {stats['count']}笔)，建议避免该时段交易")

        return suggestions

    # ========== 内部方法 ==========

    def _log_audit(self, audit_id: str, symbol: str, strategy: str, signal_type: str,
                   direction: str, result: AuditResult, reason: str, checks: Dict):
        """记录审核日志"""
        record = AuditRecord(
            audit_id=audit_id,
            timestamp=datetime.now(),
            symbol=symbol,
            strategy_name=strategy,
            signal_type=signal_type,
            direction=direction,
            result=result,
            reason=reason,
            checks=checks,
        )
        self._audit_log.append(record)

        day_key = datetime.now().date().isoformat()
        self._daily_stats[day_key]["total_audited"] += 1
        if result == AuditResult.PASS:
            self._daily_stats[day_key]["passed"] += 1
        elif result == AuditResult.BLOCK:
            self._daily_stats[day_key]["blocked"] += 1
        elif result == AuditResult.WARN:
            self._daily_stats[day_key]["warned"] += 1
        elif result == AuditResult.REDUCE:
            self._daily_stats[day_key]["reduced"] += 1

    def get_audit_stats(self) -> Dict[str, Any]:
        """获取审核统计"""
        return {
            "daily": dict(self._daily_stats),
            "total_audits": len(self._audit_log),
            "daily_pnl": self._daily_pnl,
            "daily_trades": self._daily_trades,
            "consecutive_losses": self._consecutive_losses,
        }

    def save_state(self, filepath: Optional[str] = None):
        """保存审核状态"""
        if filepath is None:
            filepath = os.path.join(self._data_dir, "auditor_state.json")
        try:
            state = {
                "daily_pnl": self._daily_pnl,
                "daily_trades": self._daily_trades,
                "consecutive_losses": self._consecutive_losses,
                "today": self._today.isoformat(),
                "trade_ledger": self._trade_ledger[-1000:],  # 只保留最近1000条
            }
            with open(filepath, "w") as f:
                json.dump(state, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save auditor state: {e}")

    def load_state(self, filepath: Optional[str] = None):
        """加载审核状态"""
        if filepath is None:
            filepath = os.path.join(self._data_dir, "auditor_state.json")
        try:
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    state = json.load(f)
                self._daily_pnl = state.get("daily_pnl", 0)
                self._daily_trades = state.get("daily_trades", 0)
                self._consecutive_losses = state.get("consecutive_losses", 0)
                self._today = datetime.fromisoformat(state.get("today", datetime.now().date().isoformat())).date()
                self._trade_ledger = state.get("trade_ledger", [])
                logger.info(f"Auditor state loaded: {len(self._trade_ledger)} trades, pnl={self._daily_pnl}")
        except Exception as e:
            logger.error(f"Failed to load auditor state: {e}")