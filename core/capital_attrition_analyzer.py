"""
生产级资金磨损分析器 (Capital Attrition Analyzer)
==================================================
量化交易中，资金磨损是持续侵蚀利润的隐形杀手。本模块全面追踪、
分析、预警和优化所有类型的资金磨损，确保每一分钱都花在刀刃上。

磨损类型：
1. 交易手续费 (Trading Fees)     — 开平仓 taker/maker 费
2. 滑点损耗 (Slippage Loss)       — 预期 vs 实际成交价差
3. 资金费率 (Funding Rate)        — 永续合约多空资金费率
4. 价差损耗 (Spread Cost)         — 买卖价差隐性成本
5. 市场冲击 (Market Impact)       — 大单对盘口的影响
6. 机会成本 (Opportunity Cost)    — 资金占用在亏损仓位的损失
7. 无效交易 (Invalid Trades)      — 手续费超过利润的微利交易

核心功能：
- 实时磨损追踪：每笔交易记录完整的磨损明细
- 磨损率监控：磨损/利润比率超标自动告警
- 磨损预算管理：每日/每策略磨损上限，超限自动暂停
- 磨损归因分析：按策略、币种、类型多维度分解
- 磨损优化建议：自动推荐降低磨损的操作（切换maker、避开高费率时段）
- 历史磨损趋势：日/周/月多维趋势分析
- 自适应费率：根据实际成交统计真实费率，检测费率异常
"""

import json
import math
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from loguru import logger


class AttritionType(Enum):
    """磨损类型"""
    TRADING_FEE = "trading_fee"           # 交易手续费
    SLIPPAGE = "slippage"                 # 滑点损耗
    FUNDING_RATE = "funding_rate"         # 资金费率
    SPREAD_COST = "spread_cost"           # 价差损耗
    MARKET_IMPACT = "market_impact"       # 市场冲击
    OPPORTUNITY_COST = "opportunity_cost" # 机会成本
    INVALID_TRADE = "invalid_trade"       # 无效交易（手续费>利润）
    OTHER = "other"                       # 其他


class AttritionSeverity(Enum):
    """磨损严重程度"""
    NORMAL = "normal"         # 正常范围
    ELEVATED = "elevated"     # 偏高
    HIGH = "high"             # 高
    CRITICAL = "critical"     # 严重超标


@dataclass
class AttritionRecord:
    """单笔磨损记录"""
    record_id: str
    timestamp: datetime
    symbol: str
    strategy_name: str
    attrition_type: AttritionType
    amount_usdt: float              # 磨损金额（USDT）
    trade_value_usdt: float         # 交易名义价值
    attrition_rate: float           # 磨损率 = amount / trade_value
    reference_price: float = 0.0    # 参考价格
    filled_price: float = 0.0       # 实际成交价
    quantity: float = 0.0           # 数量
    side: str = ""                  # buy/sell
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttritionBudget:
    """磨损预算"""
    strategy_name: str
    daily_budget_usdt: float        # 日预算
    daily_used_usdt: float = 0.0    # 当日已用
    weekly_budget_usdt: float = 0.0 # 周预算
    weekly_used_usdt: float = 0.0   # 本周已用
    max_attrition_rate: float = 0.3 # 最大磨损率（磨损/利润）
    last_reset_date: str = ""       # 上次重置日期
    is_paused: bool = False         # 是否因超预算暂停
    paused_at: Optional[datetime] = None
    pause_reason: str = ""


@dataclass
class AttritionStats:
    """磨损统计快照"""
    total_attrition_usdt: float = 0.0
    total_trade_value_usdt: float = 0.0
    total_profit_usdt: float = 0.0
    overall_attrition_rate: float = 0.0
    by_type: Dict[str, float] = field(default_factory=dict)
    by_strategy: Dict[str, float] = field(default_factory=dict)
    by_symbol: Dict[str, float] = field(default_factory=dict)
    record_count: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None


class CapitalAttritionAnalyzer:
    """
    生产级资金磨损分析器

    使用示例:
        analyzer = CapitalAttritionAnalyzer(config)
        analyzer.record_fee("BTC-USDT-SWAP", "trend", 0.5, 1000.0, "buy")
        analyzer.record_slippage("ETH-USDT-SWAP", "grid", 0.3, 500.0, "sell", 3000.0, 2998.5)
        stats = analyzer.get_attrition_stats()
        analyzer.check_budget_alerts()
    """

    # ─── 默认费率配置 ───
    DEFAULT_TAKER_FEE = 0.0005       # 0.05% taker
    DEFAULT_MAKER_FEE = 0.0002       # 0.02% maker
    DEFAULT_FUNDING_RATE = 0.0001    # 0.01% 默认资金费率

    # ─── 磨损率阈值 ───
    ATTRITION_RATE_NORMAL = 0.15     # < 15% 正常
    ATTRITION_RATE_ELEVATED = 0.25   # < 25% 偏高
    ATTRITION_RATE_HIGH = 0.40       # < 40% 高
    # >= 40% critical

    # ─── 预算告警阈值 ───
    BUDGET_WARN_PCT = 0.70           # 70% 预算告警
    BUDGET_PAUSE_PCT = 0.95          # 95% 暂停策略

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        att_cfg = self.config.get("capital_attrition", {})
        
        # ─── 基本开关 ───
        self._enabled = att_cfg.get("enabled", True)
        
        # ─── 费率配置 ───
        self._taker_fee_rate = att_cfg.get("taker_fee_rate", self.DEFAULT_TAKER_FEE)
        self._maker_fee_rate = att_cfg.get("maker_fee_rate", self.DEFAULT_MAKER_FEE)
        
        # ─── 自适应费率 ───
        self._adaptive_fee_enabled = att_cfg.get("adaptive_fee_enabled", True)
        self._adaptive_fee_window = att_cfg.get("adaptive_fee_window", 100)  # 滑动窗口
        self._fee_history: deque = deque(maxlen=self._adaptive_fee_window)
        self._estimated_taker_fee = self._taker_fee_rate
        self._estimated_maker_fee = self._maker_fee_rate
        
        # ─── 磨损记录 ───
        self._records: deque = deque(maxlen=att_cfg.get("max_records", 10000))
        self._lock = threading.Lock()
        
        # ─── 预算管理 ───
        self._budgets: Dict[str, AttritionBudget] = {}
        self._default_daily_budget = att_cfg.get("default_daily_budget_usdt", 5.0)
        self._default_weekly_budget = att_cfg.get("default_weekly_budget_usdt", 25.0)
        self._budget_check_enabled = att_cfg.get("budget_check_enabled", True)
        self._budget_auto_pause = att_cfg.get("budget_auto_pause", False)
        
        # ─── 告警回调 ───
        self._alert_callbacks: List[Callable] = []
        self._alert_cooldown_seconds = att_cfg.get("alert_cooldown_seconds", 300)
        self._last_alert_time: Dict[str, float] = {}  # alert_type -> last_alert_time
        
        # ─── 策略利润率跟踪 ───
        self._strategy_profits: Dict[str, float] = defaultdict(float)
        self._strategy_attrition: Dict[str, float] = defaultdict(float)
        self._strategy_trade_count: Dict[str, int] = defaultdict(int)
        
        # ─── 无效交易跟踪 ───
        self._invalid_trade_threshold = att_cfg.get("invalid_trade_threshold", 0.5)
        # 手续费超过利润50%视为无效交易
        self._invalid_trades_count: Dict[str, int] = defaultdict(int)
        self._invalid_trades_cost: Dict[str, float] = defaultdict(float)
        
        # ─── 资金费率跟踪 ───
        self._funding_payments: deque = deque(maxlen=500)
        self._funding_rate_threshold = att_cfg.get("funding_rate_alert_threshold", 0.001)
        
        # ─── 每日统计重置 ───
        self._daily_reset_hour = att_cfg.get("daily_reset_hour", 0)  # UTC+8 零点
        self._last_daily_reset_date = ""
        self._daily_stats = self._create_empty_stats()
        
        # ─── 优化建议 ───
        self._optimization_suggestions: List[Dict[str, Any]] = []
        self._last_suggestion_time: float = 0
        self._suggestion_interval = att_cfg.get("suggestion_interval_seconds", 3600)
        
        # ─── 持久化 ───
        self._persist_path = att_cfg.get("persist_path", "")
        
        logger.info(
            f"CapitalAttritionAnalyzer initialized: "
            f"taker_fee={self._taker_fee_rate:.4%}, maker_fee={self._maker_fee_rate:.4%}, "
            f"daily_budget={self._default_daily_budget}USDT, max_records={self._records.maxlen}"
        )

    # ═══════════════════════════════════════════════════════════════
    # 磨损记录 API
    # ═══════════════════════════════════════════════════════════════

    def record_fee(
        self,
        symbol: str,
        strategy_name: str,
        fee_usdt: float,
        trade_value_usdt: float,
        side: str = "",
        is_maker: bool = False,
    ) -> None:
        """记录交易手续费"""
        if not self._enabled or fee_usdt <= 0:
            return
        
        record = self._create_record(
            symbol, strategy_name, AttritionType.TRADING_FEE,
            fee_usdt, trade_value_usdt, side=side,
            metadata={"is_maker": is_maker}
        )
        self._commit_record(record, strategy_name)

    def record_slippage(
        self,
        symbol: str,
        strategy_name: str,
        expected_price: float,
        filled_price: float,
        quantity: float,
        side: str = "",
    ) -> float:
        """记录滑点损耗，返回损耗金额"""
        if not self._enabled or expected_price <= 0 or filled_price <= 0:
            return 0.0
        
        trade_value = expected_price * quantity
        if side == "sell":
            slippage_usdt = (expected_price - filled_price) * quantity
        elif side == "buy":
            slippage_usdt = (filled_price - expected_price) * quantity
        else:
            slippage_usdt = abs(filled_price - expected_price) * quantity
        
        slippage_usdt = max(0, slippage_usdt)  # 正向滑点不算磨损
        
        if slippage_usdt > 0:
            record = self._create_record(
                symbol, strategy_name, AttritionType.SLIPPAGE,
                slippage_usdt, trade_value,
                reference_price=expected_price, filled_price=filled_price,
                quantity=quantity, side=side,
            )
            self._commit_record(record, strategy_name)
        
        return slippage_usdt

    def record_funding_payment(
        self,
        symbol: str,
        strategy_name: str,
        payment_usdt: float,
        position_value_usdt: float,
        funding_rate: float = 0,
    ) -> None:
        """记录资金费率支付"""
        if not self._enabled:
            return
        
        if payment_usdt > 0:
            # 收到资金费 = 正收益，不记录为磨损
            self._funding_payments.append({
                "timestamp": datetime.now(),
                "symbol": symbol,
                "strategy": strategy_name,
                "payment": payment_usdt,
                "rate": funding_rate,
            })
            return
        
        cost = abs(payment_usdt)
        record = self._create_record(
            symbol, strategy_name, AttritionType.FUNDING_RATE,
            cost, position_value_usdt,
            metadata={"funding_rate": funding_rate}
        )
        self._commit_record(record, strategy_name)
        
        # 高费率告警
        if abs(funding_rate) > self._funding_rate_threshold:
            self._maybe_alert(
                "high_funding_rate",
                f"High funding rate: {symbol} rate={funding_rate:.4%} "
                f"payment={cost:.4f}USDT strategy={strategy_name}",
                severity=AttritionSeverity.HIGH,
            )

    def record_spread_cost(
        self,
        symbol: str,
        strategy_name: str,
        bid: float,
        ask: float,
        quantity: float,
        side: str = "",
    ) -> float:
        """记录价差损耗"""
        if not self._enabled or bid <= 0 or ask <= 0:
            return 0.0
        
        spread = ask - bid
        mid_price = (bid + ask) / 2
        trade_value = mid_price * quantity
        
        # 价差损耗 = 半价差 * 数量（买入时吃ask，卖出时吃bid）
        spread_cost = (spread / 2) * quantity
        
        if spread_cost > 0:
            record = self._create_record(
                symbol, strategy_name, AttritionType.SPREAD_COST,
                spread_cost, trade_value, side=side,
                metadata={"bid": bid, "ask": ask, "spread_pct": spread / mid_price}
            )
            self._commit_record(record, strategy_name)
        
        return spread_cost

    def record_market_impact(
        self,
        symbol: str,
        strategy_name: str,
        order_price: float,
        avg_fill_price: float,
        quantity: float,
        side: str = "",
    ) -> float:
        """记录市场冲击成本（大单对盘口的影响）"""
        if not self._enabled or order_price <= 0:
            return 0.0
        
        trade_value = order_price * quantity
        if side == "buy":
            impact = (avg_fill_price - order_price) * quantity
        elif side == "sell":
            impact = (order_price - avg_fill_price) * quantity
        else:
            impact = abs(avg_fill_price - order_price) * quantity
        
        impact = max(0, impact)
        
        if impact > 0:
            record = self._create_record(
                symbol, strategy_name, AttritionType.MARKET_IMPACT,
                impact, trade_value, side=side,
                metadata={"order_price": order_price, "avg_fill_price": avg_fill_price}
            )
            self._commit_record(record, strategy_name)
        
        return impact

    def record_opportunity_cost(
        self,
        symbol: str,
        strategy_name: str,
        locked_capital_usdt: float,
        holding_duration_hours: float,
        reference_return_rate: float = 0.05 / 365 / 24,  # 年化5% / 365 / 24 = 每小时
    ) -> float:
        """记录机会成本（资金占用在亏损仓位的时间成本）"""
        if not self._enabled or locked_capital_usdt <= 0:
            return 0.0
        
        opportunity_cost = locked_capital_usdt * reference_return_rate * holding_duration_hours
        
        record = self._create_record(
            symbol, strategy_name, AttritionType.OPPORTUNITY_COST,
            opportunity_cost, locked_capital_usdt,
            metadata={
                "holding_hours": holding_duration_hours,
                "reference_rate": reference_return_rate,
            }
        )
        self._commit_record(record, strategy_name)
        
        return opportunity_cost

    def record_invalid_trade(
        self,
        symbol: str,
        strategy_name: str,
        fee_usdt: float,
        profit_usdt: float,
        trade_value_usdt: float,
    ) -> None:
        """记录无效交易（手续费占比过高）"""
        if not self._enabled:
            return
        
        if profit_usdt <= 0 and fee_usdt > 0:
            # 亏损交易 + 手续费，记录为无效交易
            record = self._create_record(
                symbol, strategy_name, AttritionType.INVALID_TRADE,
                fee_usdt, trade_value_usdt,
                metadata={"profit_usdt": profit_usdt, "fee_usdt": fee_usdt}
            )
            self._commit_record(record, strategy_name)
            
            self._invalid_trades_count[strategy_name] += 1
            self._invalid_trades_cost[strategy_name] += fee_usdt

    # ═══════════════════════════════════════════════════════════════
    # 预算管理
    # ═══════════════════════════════════════════════════════════════

    def set_strategy_budget(
        self,
        strategy_name: str,
        daily_budget: float = None,
        weekly_budget: float = None,
        max_attrition_rate: float = None,
    ) -> None:
        """设置策略磨损预算"""
        with self._lock:
            if strategy_name not in self._budgets:
                self._budgets[strategy_name] = AttritionBudget(
                    strategy_name=strategy_name,
                    daily_budget_usdt=daily_budget or self._default_daily_budget,
                    weekly_budget_usdt=weekly_budget or self._default_weekly_budget,
                    max_attrition_rate=max_attrition_rate or 0.3,
                )
            else:
                budget = self._budgets[strategy_name]
                if daily_budget is not None:
                    budget.daily_budget_usdt = daily_budget
                if weekly_budget is not None:
                    budget.weekly_budget_usdt = weekly_budget
                if max_attrition_rate is not None:
                    budget.max_attrition_rate = max_attrition_rate

    def check_budget(self, strategy_name: str) -> Tuple[bool, str]:
        """
        检查策略磨损预算
        
        Returns:
            (allowed, reason) - 是否允许继续交易，原因
        """
        if not self._budget_check_enabled:
            return True, "budget_check_disabled"
        
        self._check_daily_reset()
        
        budget = self._budgets.get(strategy_name)
        if not budget:
            return True, "no_budget_set"
        
        # 如果因超预算暂停
        if budget.is_paused:
            return False, f"budget_paused: {budget.pause_reason}"
        
        # 检查日预算
        daily_usage_pct = budget.daily_used_usdt / max(budget.daily_budget_usdt, 0.01)
        if daily_usage_pct >= self.BUDGET_PAUSE_PCT:
            if self._budget_auto_pause:
                budget.is_paused = True
                budget.paused_at = datetime.now()
                budget.pause_reason = f"daily_budget_exceeded: {daily_usage_pct:.0%}"
                self._maybe_alert(
                    "budget_exceeded",
                    f"Strategy {strategy_name} daily budget exceeded: "
                    f"{budget.daily_used_usdt:.2f}/{budget.daily_budget_usdt:.2f} USDT "
                    f"({daily_usage_pct:.0%}) — AUTO PAUSED",
                    severity=AttritionSeverity.CRITICAL,
                )
                return False, budget.pause_reason
            else:
                self._maybe_alert(
                    "budget_warning",
                    f"Strategy {strategy_name} daily budget critical: "
                    f"{budget.daily_used_usdt:.2f}/{budget.daily_budget_usdt:.2f} USDT "
                    f"({daily_usage_pct:.0%})",
                    severity=AttritionSeverity.HIGH,
                )
        elif daily_usage_pct >= self.BUDGET_WARN_PCT:
            self._maybe_alert(
                "budget_warning",
                f"Strategy {strategy_name} daily budget warning: "
                f"{budget.daily_used_usdt:.2f}/{budget.daily_budget_usdt:.2f} USDT "
                f"({daily_usage_pct:.0%})",
                severity=AttritionSeverity.ELEVATED,
            )
        
        return True, "ok"

    def resume_strategy(self, strategy_name: str) -> bool:
        """恢复被暂停的策略"""
        budget = self._budgets.get(strategy_name)
        if budget and budget.is_paused:
            budget.is_paused = False
            budget.pause_reason = ""
            logger.info(f"Strategy {strategy_name} budget resumed")
            return True
        return False

    def get_budget_status(self, strategy_name: str = None) -> Dict[str, Any]:
        """获取预算状态"""
        self._check_daily_reset()
        
        if strategy_name:
            budget = self._budgets.get(strategy_name)
            if not budget:
                return {}
            return self._budget_to_dict(budget)
        
        return {
            name: self._budget_to_dict(b) for name, b in self._budgets.items()
        }

    # ═══════════════════════════════════════════════════════════════
    # 统计分析
    # ═══════════════════════════════════════════════════════════════

    def get_attrition_stats(
        self,
        period_hours: float = 24,
        strategy_name: str = None,
    ) -> AttritionStats:
        """获取磨损统计"""
        now = datetime.now()
        cutoff = now - timedelta(hours=period_hours)
        
        stats = AttritionStats(period_start=cutoff, period_end=now)
        
        with self._lock:
            for record in self._records:
                if record.timestamp < cutoff:
                    continue
                if strategy_name and record.strategy_name != strategy_name:
                    continue
                
                stats.total_attrition_usdt += record.amount_usdt
                stats.total_trade_value_usdt += record.trade_value_usdt
                stats.record_count += 1
                
                # 按类型
                type_key = record.attrition_type.value
                stats.by_type[type_key] = stats.by_type.get(type_key, 0) + record.amount_usdt
                
                # 按策略
                stats.by_strategy[record.strategy_name] = \
                    stats.by_strategy.get(record.strategy_name, 0) + record.amount_usdt
                
                # 按币种
                stats.by_symbol[record.symbol] = \
                    stats.by_symbol.get(record.symbol, 0) + record.amount_usdt
            
            # 计算总利润（用于磨损率）
            if strategy_name:
                stats.total_profit_usdt = self._strategy_profits.get(strategy_name, 0)
            else:
                stats.total_profit_usdt = sum(self._strategy_profits.values())
            
            if stats.total_trade_value_usdt > 0:
                stats.overall_attrition_rate = stats.total_attrition_usdt / stats.total_trade_value_usdt
        
        return stats

    def get_attrition_trend(
        self,
        days: int = 7,
        granularity: str = "daily",
    ) -> List[Dict[str, Any]]:
        """获取磨损趋势数据"""
        now = datetime.now()
        cutoff = now - timedelta(days=days)
        
        if granularity == "hourly":
            buckets: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
            fmt = "%Y-%m-%d %H:00"
        else:
            buckets: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
            fmt = "%Y-%m-%d"
        
        with self._lock:
            for record in self._records:
                if record.timestamp < cutoff:
                    continue
                key = record.timestamp.strftime(fmt)
                buckets[key][record.attrition_type.value] += record.amount_usdt
                buckets[key]["total"] += record.amount_usdt
        
        trend = []
        for key in sorted(buckets.keys()):
            data = dict(buckets[key])
            data["period"] = key
            trend.append(data)
        
        return trend

    def get_attrition_breakdown(
        self,
        strategy_name: str = None,
        period_hours: float = 24,
    ) -> Dict[str, Any]:
        """获取磨损归因分析"""
        stats = self.get_attrition_stats(period_hours, strategy_name)
        
        breakdown = {
            "total_attrition_usdt": round(stats.total_attrition_usdt, 4),
            "total_trade_value_usdt": round(stats.total_trade_value_usdt, 2),
            "overall_attrition_rate": round(stats.overall_attrition_rate, 6),
            "record_count": stats.record_count,
            "by_type": {k: round(v, 4) for k, v in sorted(
                stats.by_type.items(), key=lambda x: x[1], reverse=True
            )},
            "by_strategy": {k: round(v, 4) for k, v in sorted(
                stats.by_strategy.items(), key=lambda x: x[1], reverse=True
            )},
            "by_symbol": {k: round(v, 4) for k, v in sorted(
                stats.by_symbol.items(), key=lambda x: x[1], reverse=True
            )},
            "severity": self._rate_to_severity(stats.overall_attrition_rate).value,
        }
        
        # 无效交易统计
        if strategy_name:
            breakdown["invalid_trades"] = {
                "count": self._invalid_trades_count.get(strategy_name, 0),
                "cost_usdt": round(self._invalid_trades_cost.get(strategy_name, 0), 4),
            }
        else:
            breakdown["invalid_trades"] = {
                "count": sum(self._invalid_trades_count.values()),
                "cost_usdt": round(sum(self._invalid_trades_cost.values()), 4),
            }
        
        return breakdown

    def get_daily_summary(self) -> Dict[str, Any]:
        """获取当日磨损摘要"""
        self._check_daily_reset()
        stats = self.get_attrition_stats(period_hours=24)
        
        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_attrition_usdt": round(stats.total_attrition_usdt, 4),
            "total_trade_value_usdt": round(stats.total_trade_value_usdt, 2),
            "attrition_rate": round(stats.overall_attrition_rate, 6),
            "trade_count": stats.record_count,
            "top_attrition_type": max(stats.by_type, key=stats.by_type.get) if stats.by_type else "none",
            "top_attrition_strategy": max(stats.by_strategy, key=stats.by_strategy.get) if stats.by_strategy else "none",
            "budgets": self.get_budget_status(),
            "estimated_taker_fee": round(self._estimated_taker_fee, 6),
            "estimated_maker_fee": round(self._estimated_maker_fee, 6),
        }
        
        return summary

    # ═══════════════════════════════════════════════════════════════
    # 优化建议（生产级多维度分析引擎）
    # ═══════════════════════════════════════════════════════════════

    # ─── 建议阈值（生产调优） ───
    SUGGESTION_FEE_PCT_HIGH = 0.40       # 手续费占比 > 40% → 高优先级
    SUGGESTION_FEE_PCT_MEDIUM = 0.25     # 手续费占比 > 25% → 中优先级
    SUGGESTION_SLIPPAGE_RATE = 0.0005    # 滑点率 > 0.05% → 建议
    SUGGESTION_FUNDING_PCT = 0.10        # 资金费率占比 > 10% → 建议
    SUGGESTION_SPREAD_PCT = 0.10         # 价差占比 > 10% → 建议
    SUGGESTION_INVALID_TRADES = 2        # 无效交易 > 2笔 → 建议
    SUGGESTION_MAKER_RATIO_LOW = 0.30    # maker占比 < 30% → 建议
    SUGGESTION_BUDGET_WARN = 0.60        # 预算使用 > 60% → 提醒
    SUGGESTION_ATTRITION_TREND_UP = 1.3  # 磨损趋势上升 > 30% → 警告

    def get_optimization_suggestions(self, force: bool = False) -> List[Dict[str, Any]]:
        """
        获取磨损优化建议 — 生产级多维度分析

        分析维度：
        1. 手续费结构分析（maker/taker比例 + 费率优化）
        2. 滑点损耗分析（按币种 + 时段）
        3. 资金费率优化（高费率时段规避）
        4. 价差损耗分析
        5. 策略对比排名（找出磨损最大的策略）
        6. 币种对比排名（找出磨损最大的交易对）
        7. 预算利用率预警
        8. 趋势分析（磨损是否在恶化）
        9. 无效交易/信号质量分析
        10. 整体健康度综合评估
        """
        now = time.time()
        if not force and now - self._last_suggestion_time < self._suggestion_interval:
            return self._optimization_suggestions

        self._last_suggestion_time = now
        self._optimization_suggestions = []
        stats = self.get_attrition_stats(period_hours=24)

        if stats.record_count == 0:
            return self._optimization_suggestions

        by_type = stats.by_type
        total = stats.total_attrition_usdt
        if total <= 0:
            return self._optimization_suggestions

        # ─── 收集 maker/taker 统计 ───
        maker_count, taker_count, maker_fee_total, taker_fee_total = 0, 0, 0.0, 0.0
        with self._lock:
            for r in self._records:
                if r.attrition_type == AttritionType.TRADING_FEE:
                    if r.metadata.get("is_maker"):
                        maker_count += 1
                        maker_fee_total += r.amount_usdt
                    else:
                        taker_count += 1
                        taker_fee_total += r.amount_usdt
        total_fee_trades = maker_count + taker_count

        # ═══════════════════════════════════════════════════════
        # 1. 手续费结构分析
        # ═══════════════════════════════════════════════════════
        fee_total = by_type.get("trading_fee", 0)
        fee_pct = fee_total / total

        if total_fee_trades > 0:
            maker_ratio = maker_count / total_fee_trades
            if maker_ratio < self.SUGGESTION_MAKER_RATIO_LOW:
                potential_saving = taker_fee_total * 0.6  # 60% taker转maker可节省约60%
                self._optimization_suggestions.append({
                    "type": "maker_ratio",
                    "priority": "high" if maker_ratio < 0.15 else "medium",
                    "title": f"Maker单占比过低 ({maker_ratio:.0%})",
                    "detail": (
                        f"近24h共{total_fee_trades}笔手续费记录，其中maker仅{maker_count}笔({maker_ratio:.0%})。"
                        f"Maker费率({self._maker_fee_rate:.2%})比Taker({self._taker_fee_rate:.2%})低60%，"
                        f"将taker单转为maker可节省约{potential_saving:.4f} USDT。"
                        f"建议：挂限价单替代市价单，适当放宽成交时效要求。"
                    ),
                    "action": "increase_maker_ratio",
                    "potential_saving_usdt": round(potential_saving, 4),
                    "metrics": {
                        "maker_count": maker_count, "taker_count": taker_count,
                        "maker_ratio": round(maker_ratio, 3),
                        "maker_fee_rate": self._maker_fee_rate,
                        "taker_fee_rate": self._taker_fee_rate,
                    }
                })

        if fee_pct > self.SUGGESTION_FEE_PCT_MEDIUM:
            priority = "high" if fee_pct > self.SUGGESTION_FEE_PCT_HIGH else "medium"
            self._optimization_suggestions.append({
                "type": "fee_optimization",
                "priority": priority,
                "title": f"手续费占比偏高 ({fee_pct:.0%})",
                "detail": (
                    f"手续费{round(fee_total, 4)} USDT占总磨损{round(total, 4)} USDT的{fee_pct:.0%}。"
                    f"建议：1) 优先使用限价单(maker)降低费率 2) 检查是否有高频无效交易 3) 关注OKX手续费等级，"
                    f"提升30天交易量可降低费率档位。"
                ),
                "action": "reduce_trading_fees",
                "potential_saving_usdt": round(fee_total * 0.4, 4),
                "metrics": {"fee_pct": round(fee_pct, 3), "fee_total": round(fee_total, 4)},
            })

        # ═══════════════════════════════════════════════════════
        # 2. 滑点损耗分析（按币种）
        # ═══════════════════════════════════════════════════════
        slippage_total = by_type.get("slippage", 0)
        if slippage_total > 0 and stats.total_trade_value_usdt > 0:
            slippage_rate = slippage_total / stats.total_trade_value_usdt
            if slippage_rate > self.SUGGESTION_SLIPPAGE_RATE:
                # 按币种分解滑点
                symbol_slippage = self._get_symbol_slippage_breakdown()
                worst_symbols = sorted(symbol_slippage.items(), key=lambda x: x[1], reverse=True)[:3]
                symbol_detail = ", ".join(f"{s.split('-')[0]}={v:.4f}" for s, v in worst_symbols)

                self._optimization_suggestions.append({
                    "type": "slippage_optimization",
                    "priority": "high" if slippage_rate > 0.002 else "medium",
                    "title": f"滑点损耗偏高 ({slippage_rate:.4%})",
                    "detail": (
                        f"滑点损耗{slippage_total:.4f} USDT，滑点率{slippage_rate:.4%}。"
                        f"滑点最大的币种: {symbol_detail}。"
                        f"建议：1) 高滑点币种增加限价偏移 2) 大单使用TWAP/VWAP分批执行 3) 避开流动性低谷时段。"
                    ),
                    "action": "optimize_slippage",
                    "potential_saving_usdt": round(slippage_total * 0.5, 4),
                    "metrics": {
                        "slippage_rate": round(slippage_rate, 4),
                        "slippage_total": round(slippage_total, 4),
                        "worst_symbols": dict(worst_symbols),
                    },
                })

        # ═══════════════════════════════════════════════════════
        # 3. 资金费率优化
        # ═══════════════════════════════════════════════════════
        funding_total = by_type.get("funding_rate", 0)
        if funding_total > 0:
            funding_pct = funding_total / total
            if funding_pct > self.SUGGESTION_FUNDING_PCT:
                self._optimization_suggestions.append({
                    "type": "funding_optimization",
                    "priority": "medium",
                    "title": f"资金费率损耗偏高 ({funding_pct:.0%})",
                    "detail": (
                        f"资金费率支付{funding_total:.4f} USDT，占总磨损{funding_pct:.0%}。"
                        f"建议：1) 在资金费率结算前15分钟评估是否需要平仓 2) 高费率时段(>0.1%)避免新开仓 "
                        f"3) 考虑在费率结算后立即回补仓位。"
                    ),
                    "action": "optimize_funding_timing",
                    "potential_saving_usdt": round(funding_total * 0.6, 4),
                    "metrics": {"funding_pct": round(funding_pct, 3), "funding_total": round(funding_total, 4)},
                })

        # ═══════════════════════════════════════════════════════
        # 4. 价差损耗分析
        # ═══════════════════════════════════════════════════════
        spread_total = by_type.get("spread_cost", 0)
        if spread_total > 0:
            spread_pct = spread_total / total
            if spread_pct > self.SUGGESTION_SPREAD_PCT:
                self._optimization_suggestions.append({
                    "type": "spread_optimization",
                    "priority": "low",
                    "title": f"价差损耗需关注 ({spread_pct:.0%})",
                    "detail": (
                        f"买卖价差损耗{spread_total:.4f} USDT。"
                        f"建议：1) 优先交易流动性好的币种(BTC/ETH) 2) 避免在盘口稀薄时交易小币种。"
                    ),
                    "action": "trade_high_liquidity_pairs",
                    "potential_saving_usdt": round(spread_total * 0.3, 4),
                    "metrics": {"spread_pct": round(spread_pct, 3)},
                })

        # ═══════════════════════════════════════════════════════
        # 5. 策略对比排名
        # ═══════════════════════════════════════════════════════
        strategy_ranking = self._get_strategy_attrition_ranking(stats)
        if len(strategy_ranking) >= 2:
            worst = strategy_ranking[0]
            best = strategy_ranking[-1]
            if worst["attrition"] > best["attrition"] * 2:
                self._optimization_suggestions.append({
                    "type": "strategy_comparison",
                    "priority": "medium",
                    "title": f"策略磨损差异显著",
                    "detail": (
                        f"磨损最高策略: {worst['name']}({worst['attrition']:.4f} USDT)，"
                        f"最低策略: {best['name']}({best['attrition']:.4f} USDT)。"
                        f"差距{worst['attrition'] - best['attrition']:.4f} USDT。"
                        f"建议：分析{worst['name']}策略的高磨损原因，参考{best['name']}的执行模式。"
                    ),
                    "action": "analyze_high_attrition_strategy",
                    "potential_saving_usdt": round((worst["attrition"] - best["attrition"]) * 0.5, 4),
                    "metrics": {
                        "worst_strategy": worst["name"],
                        "best_strategy": best["name"],
                        "ranking": strategy_ranking,
                    },
                })

        # ═══════════════════════════════════════════════════════
        # 6. 无效交易/信号质量分析
        # ═══════════════════════════════════════════════════════
        total_invalid = sum(self._invalid_trades_count.values())
        total_invalid_cost = sum(self._invalid_trades_cost.values())
        if total_invalid >= self.SUGGESTION_INVALID_TRADES:
            self._optimization_suggestions.append({
                "type": "signal_quality",
                "priority": "high" if total_invalid > 5 else "medium",
                "title": f"无效交易需关注 ({total_invalid}笔)",
                "detail": (
                    f"近24h有{total_invalid}笔无效交易（手续费>利润），累计浪费{total_invalid_cost:.4f} USDT。"
                    f"建议：1) 提高信号质量阈值 2) 增加最小预期利润过滤 3) 降低震荡市中的交易频率。"
                ),
                "action": "improve_signal_quality",
                "potential_saving_usdt": round(total_invalid_cost, 4),
                "metrics": {
                    "invalid_count": total_invalid,
                    "invalid_cost": round(total_invalid_cost, 4),
                    "by_strategy": dict(self._invalid_trades_count),
                },
            })

        # ═══════════════════════════════════════════════════════
        # 7. 预算利用率预警
        # ═══════════════════════════════════════════════════════
        budget_warnings = self._get_budget_warnings()
        self._optimization_suggestions.extend(budget_warnings)

        # ═══════════════════════════════════════════════════════
        # 8. 趋势分析（磨损是否在恶化）
        # ═══════════════════════════════════════════════════════
        trend_warning = self._get_trend_warning()
        if trend_warning:
            self._optimization_suggestions.append(trend_warning)

        # ═══════════════════════════════════════════════════════
        # 9. 整体健康度综合评估
        # ═══════════════════════════════════════════════════════
        health = self._assess_overall_health(stats, total, fee_pct)
        if health:
            self._optimization_suggestions.append(health)

        # ═══════════════════════════════════════════════════════
        # 10. 排序：priority > potential_saving_usdt
        # ═══════════════════════════════════════════════════════
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        self._optimization_suggestions.sort(
            key=lambda s: (priority_order.get(s["priority"], 99), -s.get("potential_saving_usdt", 0))
        )

        return self._optimization_suggestions

    # ─── 优化建议辅助方法 ───

    def _get_symbol_slippage_breakdown(self) -> Dict[str, float]:
        """按币种分解滑点损耗"""
        result: Dict[str, float] = defaultdict(float)
        with self._lock:
            for r in self._records:
                if r.attrition_type == AttritionType.SLIPPAGE:
                    result[r.symbol] += r.amount_usdt
        return dict(result)

    def _get_strategy_attrition_ranking(self, stats: AttritionStats) -> List[Dict[str, Any]]:
        """策略磨损排名"""
        ranking = []
        for name, att in stats.by_strategy.items():
            profit = self._strategy_profits.get(name, 0)
            trade_count = self._strategy_trade_count.get(name, 0)
            ranking.append({
                "name": name,
                "attrition": round(att, 4),
                "profit": round(profit, 4),
                "trade_count": trade_count,
                "attrition_rate": round(att / max(abs(profit), 0.01), 4) if profit != 0 else 0,
            })
        ranking.sort(key=lambda x: x["attrition"], reverse=True)
        return ranking

    def _get_budget_warnings(self) -> List[Dict[str, Any]]:
        """预算利用率预警"""
        warnings = []
        for name, budget in self._budgets.items():
            if budget.daily_budget_usdt > 0:
                usage_pct = budget.daily_used_usdt / budget.daily_budget_usdt
                if usage_pct > self.SUGGESTION_BUDGET_WARN:
                    warnings.append({
                        "type": "budget_warning",
                        "priority": "high" if usage_pct > 0.85 else "medium",
                        "title": f"[{name}] 磨损预算使用{usage_pct:.0%}",
                        "detail": (
                            f"{name}策略当日磨损{budget.daily_used_usdt:.4f} USDT，"
                            f"已达日预算{budget.daily_budget_usdt:.1f} USDT的{usage_pct:.0%}。"
                            f"建议：检查交易频率和单笔磨损是否合理，必要时降低交易频率或提高信号门槛。"
                        ),
                        "action": "review_strategy_budget",
                        "potential_saving_usdt": round(max(0, budget.daily_used_usdt - budget.daily_budget_usdt * 0.5), 4),
                        "metrics": {
                            "strategy": name,
                            "usage_pct": round(usage_pct, 3),
                            "daily_used": round(budget.daily_used_usdt, 4),
                            "daily_budget": budget.daily_budget_usdt,
                        },
                    })
        return warnings

    def _get_trend_warning(self) -> Optional[Dict[str, Any]]:
        """磨损趋势恶化检测"""
        trend = self.get_attrition_trend(days=2, granularity="daily")
        if len(trend) < 2:
            return None

        # 比较最近两天的磨损
        totals = [t.get("total", 0) for t in trend[-2:]]
        if totals[0] > 0 and totals[1] > totals[0] * self.SUGGESTION_ATTRITION_TREND_UP:
            increase_pct = (totals[1] - totals[0]) / totals[0]
            return {
                "type": "trend_warning",
                "priority": "high" if increase_pct > 0.5 else "medium",
                "title": f"磨损趋势恶化 (+{increase_pct:.0%})",
                "detail": (
                    f"磨损从{totals[0]:.4f} USDT上升至{totals[1]:.4f} USDT，增长{increase_pct:.0%}。"
                    f"建议：1) 排查是否有策略参数漂移 2) 检查市场波动率是否异常升高 "
                    f"3) 考虑临时降低交易频率或收紧信号阈值。"
                ),
                "action": "investigate_attrition_trend",
                "potential_saving_usdt": round(totals[1] - totals[0], 4),
                "metrics": {
                    "previous": round(totals[0], 4),
                    "current": round(totals[1], 4),
                    "increase_pct": round(increase_pct, 3),
                },
            }
        return None

    def _assess_overall_health(
        self, stats: AttritionStats, total: float, fee_pct: float
    ) -> Optional[Dict[str, Any]]:
        """整体健康度综合评估"""
        # 综合评分因素
        issues = []
        if stats.overall_attrition_rate > self.ATTRITION_RATE_ELEVATED:
            issues.append(f"磨损率{stats.overall_attrition_rate:.2%}偏高")
        if fee_pct > self.SUGGESTION_FEE_PCT_MEDIUM:
            issues.append(f"手续费占比{fee_pct:.0%}偏高")
        if sum(self._invalid_trades_count.values()) >= self.SUGGESTION_INVALID_TRADES:
            issues.append(f"存在{sum(self._invalid_trades_count.values())}笔无效交易")

        if not issues:
            # 健康状态
            if stats.record_count >= 5:
                return {
                    "type": "health_check",
                    "priority": "low",
                    "title": "磨损健康度: 正常",
                    "detail": (
                        f"近24h磨损{total:.4f} USDT，磨损率{stats.overall_attrition_rate:.4%}，"
                        f"处于健康范围。继续保持当前执行策略。"
                    ),
                    "action": "maintain_current",
                    "potential_saving_usdt": 0,
                    "metrics": {
                        "attrition_rate": round(stats.overall_attrition_rate, 4),
                        "record_count": stats.record_count,
                    },
                }
            return None

        # 有问题
        return {
            "type": "health_check",
            "priority": "critical" if len(issues) >= 3 else "high",
            "title": f"磨损健康度: 需关注 ({len(issues)}项问题)",
            "detail": "；".join(issues) + "。建议逐项排查并执行优化建议。",
            "action": "review_all_issues",
            "potential_saving_usdt": round(total * 0.3, 4),
            "metrics": {"issues": issues, "overall_rate": round(stats.overall_attrition_rate, 4)},
        }

    # ═══════════════════════════════════════════════════════════════
    # 自适应费率
    # ═══════════════════════════════════════════════════════════════

    def update_adaptive_fee(self, fee_usdt: float, trade_value_usdt: float, is_maker: bool) -> None:
        """更新自适应费率估算"""
        if not self._adaptive_fee_enabled or trade_value_usdt <= 0:
            return
        
        actual_rate = fee_usdt / trade_value_usdt
        self._fee_history.append({
            "rate": actual_rate,
            "is_maker": is_maker,
            "timestamp": time.time(),
        })
        
        # 重新计算估算费率
        maker_rates = [r["rate"] for r in self._fee_history if r["is_maker"]]
        taker_rates = [r["rate"] for r in self._fee_history if not r["is_maker"]]
        
        if maker_rates:
            self._estimated_maker_fee = sum(maker_rates) / len(maker_rates)
        if taker_rates:
            self._estimated_taker_fee = sum(taker_rates) / len(taker_rates)
        
        # 检测费率异常（实际费率与预期差异过大）
        expected = self._estimated_maker_fee if is_maker else self._estimated_taker_fee
        if expected > 0 and abs(actual_rate - expected) / expected > 0.5:
            logger.warning(
                f"Fee rate anomaly: actual={actual_rate:.6%} vs expected={expected:.6%} "
                f"(is_maker={is_maker})"
            )

    def get_estimated_fee_rates(self) -> Dict[str, float]:
        """获取估算费率"""
        return {
            "taker_fee": round(self._estimated_taker_fee, 6),
            "maker_fee": round(self._estimated_maker_fee, 6),
            "configured_taker_fee": self._taker_fee_rate,
            "configured_maker_fee": self._maker_fee_rate,
            "samples": len(self._fee_history),
        }

    # ═══════════════════════════════════════════════════════════════
    # 告警
    # ═══════════════════════════════════════════════════════════════

    def register_alert_callback(self, callback: Callable) -> None:
        """注册告警回调"""
        self._alert_callbacks.append(callback)

    def _maybe_alert(
        self,
        alert_type: str,
        message: str,
        severity: AttritionSeverity = AttritionSeverity.NORMAL,
    ) -> None:
        """带冷却的告警发送"""
        now = time.time()
        last = self._last_alert_time.get(alert_type, 0)
        if now - last < self._alert_cooldown_seconds:
            return
        
        self._last_alert_time[alert_type] = now
        
        if severity == AttritionSeverity.CRITICAL:
            logger.error(f"[ATTRITION_CRITICAL] {message}")
        elif severity == AttritionSeverity.HIGH:
            logger.warning(f"[ATTRITION_HIGH] {message}")
        else:
            logger.info(f"[ATTRITION] {message}")
        
        for cb in self._alert_callbacks:
            try:
                cb({
                    "type": alert_type,
                    "message": message,
                    "severity": severity.value,
                    "timestamp": datetime.now().isoformat(),
                })
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 利润追踪
    # ═══════════════════════════════════════════════════════════════

    def record_profit(self, strategy_name: str, profit_usdt: float) -> None:
        """记录策略利润（用于磨损率计算）"""
        if not self._enabled:
            return
        
        with self._lock:
            self._strategy_profits[strategy_name] += profit_usdt
            self._strategy_trade_count[strategy_name] += 1
            
            # 检测无效交易
            attrition = self._strategy_attrition.get(strategy_name, 0)
            if attrition > 0 and profit_usdt > 0:
                attrition_rate = attrition / (attrition + profit_usdt)
                if attrition_rate > self._invalid_trade_threshold:
                    self._maybe_alert(
                        "high_attrition_strategy",
                        f"Strategy {strategy_name} attrition rate {attrition_rate:.0%} "
                        f"exceeds threshold {self._invalid_trade_threshold:.0%}",
                        severity=AttritionSeverity.HIGH,
                    )

    # ═══════════════════════════════════════════════════════════════
    # 持久化
    # ═══════════════════════════════════════════════════════════════

    def collect_persistent_state(self) -> Dict[str, Any]:
        """收集持久化状态"""
        self._check_daily_reset()
        
        return {
            "budgets": {
                name: {
                    "daily_used_usdt": b.daily_used_usdt,
                    "weekly_used_usdt": b.weekly_used_usdt,
                    "last_reset_date": b.last_reset_date,
                    "is_paused": b.is_paused,
                    "pause_reason": b.pause_reason,
                }
                for name, b in self._budgets.items()
            },
            "strategy_profits": dict(self._strategy_profits),
            "strategy_attrition": dict(self._strategy_attrition),
            "strategy_trade_count": dict(self._strategy_trade_count),
            "invalid_trades_count": dict(self._invalid_trades_count),
            "invalid_trades_cost": dict(self._invalid_trades_cost),
            "estimated_taker_fee": self._estimated_taker_fee,
            "estimated_maker_fee": self._estimated_maker_fee,
            "last_daily_reset_date": self._last_daily_reset_date,
            "collection_time": datetime.now().isoformat(),
        }

    def restore_persistent_state(self, state: Dict[str, Any]) -> None:
        """恢复持久化状态"""
        if not state:
            return
        
        try:
            for name, budget_data in state.get("budgets", {}).items():
                if name not in self._budgets:
                    self._budgets[name] = AttritionBudget(
                        strategy_name=name,
                        daily_budget_usdt=self._default_daily_budget,
                        weekly_budget_usdt=self._default_weekly_budget,
                    )
                budget = self._budgets[name]
                budget.daily_used_usdt = budget_data.get("daily_used_usdt", 0)
                budget.weekly_used_usdt = budget_data.get("weekly_used_usdt", 0)
                budget.last_reset_date = budget_data.get("last_reset_date", "")
                budget.is_paused = budget_data.get("is_paused", False)
                budget.pause_reason = budget_data.get("pause_reason", "")
            
            self._strategy_profits = defaultdict(float, state.get("strategy_profits", {}))
            self._strategy_attrition = defaultdict(float, state.get("strategy_attrition", {}))
            self._strategy_trade_count = defaultdict(int, state.get("strategy_trade_count", {}))
            self._invalid_trades_count = defaultdict(int, state.get("invalid_trades_count", {}))
            self._invalid_trades_cost = defaultdict(float, state.get("invalid_trades_cost", {}))
            self._estimated_taker_fee = state.get("estimated_taker_fee", self._taker_fee_rate)
            self._estimated_maker_fee = state.get("estimated_maker_fee", self._maker_fee_rate)
            self._last_daily_reset_date = state.get("last_daily_reset_date", "")
            
            logger.info("CapitalAttritionAnalyzer state restored")
        except Exception as e:
            logger.error(f"Failed to restore CapitalAttritionAnalyzer state: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 内部方法
    # ═══════════════════════════════════════════════════════════════

    def _create_record(
        self,
        symbol: str,
        strategy_name: str,
        attrition_type: AttritionType,
        amount_usdt: float,
        trade_value_usdt: float,
        **kwargs,
    ) -> AttritionRecord:
        """创建磨损记录"""
        record_id = f"{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{attrition_type.value}"
        attrition_rate = amount_usdt / max(trade_value_usdt, 0.01)
        
        return AttritionRecord(
            record_id=record_id,
            timestamp=datetime.now(),
            symbol=symbol,
            strategy_name=strategy_name,
            attrition_type=attrition_type,
            amount_usdt=amount_usdt,
            trade_value_usdt=trade_value_usdt,
            attrition_rate=attrition_rate,
            reference_price=kwargs.get("reference_price", 0),
            filled_price=kwargs.get("filled_price", 0),
            quantity=kwargs.get("quantity", 0),
            side=kwargs.get("side", ""),
            metadata=kwargs.get("metadata", {}),
        )

    def _commit_record(self, record: AttritionRecord, strategy_name: str) -> None:
        """提交磨损记录并更新统计"""
        with self._lock:
            self._records.append(record)
            self._strategy_attrition[strategy_name] += record.amount_usdt
            
            # 更新预算
            if strategy_name in self._budgets:
                budget = self._budgets[strategy_name]
                budget.daily_used_usdt += record.amount_usdt
                budget.weekly_used_usdt += record.amount_usdt

    def _check_daily_reset(self) -> None:
        """检查并执行每日重置"""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._last_daily_reset_date == today:
            return
        
        self._last_daily_reset_date = today
        
        for budget in self._budgets.values():
            budget.daily_used_usdt = 0.0
            budget.last_reset_date = today
            
            # 每周重置（周一）
            if datetime.now().weekday() == 0:
                budget.weekly_used_usdt = 0.0
            
            # 自动恢复被暂停的策略
            if budget.is_paused and budget.pause_reason.startswith("daily_budget"):
                budget.is_paused = False
                budget.pause_reason = ""
                logger.info(f"Strategy {budget.strategy_name} auto-resumed on daily reset")

    def _budget_to_dict(self, budget: AttritionBudget) -> Dict[str, Any]:
        """预算转字典"""
        return {
            "strategy": budget.strategy_name,
            "daily_budget": budget.daily_budget_usdt,
            "daily_used": round(budget.daily_used_usdt, 4),
            "daily_usage_pct": round(budget.daily_used_usdt / max(budget.daily_budget_usdt, 0.01), 4),
            "weekly_budget": budget.weekly_budget_usdt,
            "weekly_used": round(budget.weekly_used_usdt, 4),
            "is_paused": budget.is_paused,
            "pause_reason": budget.pause_reason,
        }

    def _rate_to_severity(self, rate: float) -> AttritionSeverity:
        """磨损率转严重程度"""
        if rate < self.ATTRITION_RATE_NORMAL:
            return AttritionSeverity.NORMAL
        elif rate < self.ATTRITION_RATE_ELEVATED:
            return AttritionSeverity.ELEVATED
        elif rate < self.ATTRITION_RATE_HIGH:
            return AttritionSeverity.HIGH
        else:
            return AttritionSeverity.CRITICAL

    @staticmethod
    def _create_empty_stats() -> AttritionStats:
        """创建空统计"""
        return AttritionStats()

    def get_stats(self) -> Dict[str, Any]:
        """获取综合统计（供外部调用）"""
        daily = self.get_daily_summary()
        breakdown = self.get_attrition_breakdown()
        fee_rates = self.get_estimated_fee_rates()
        
        return {
            "enabled": self._enabled,
            "daily_summary": daily,
            "breakdown": breakdown,
            "fee_rates": fee_rates,
            "total_records": len(self._records),
            "budgets": self.get_budget_status(),
            "optimization_suggestions": self.get_optimization_suggestions(),
        }

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """热更新配置"""
        att_cfg = new_config.get("capital_attrition", {})
        if not att_cfg:
            return
        
        self._enabled = att_cfg.get("enabled", self._enabled)
        self._budget_check_enabled = att_cfg.get("budget_check_enabled", self._budget_check_enabled)
        self._budget_auto_pause = att_cfg.get("budget_auto_pause", self._budget_auto_pause)
        self._adaptive_fee_enabled = att_cfg.get("adaptive_fee_enabled", self._adaptive_fee_enabled)
        
        logger.info("CapitalAttritionAnalyzer config updated")