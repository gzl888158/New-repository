"""
生产级仪表板数据引擎 (Dashboard Engine)
==========================================
实时掌握账户权益、可用资金与浮盈浮亏。汇集权益曲线、策略对比、
持仓分布与风险水位四视图，构成交易决策的核心画像。

架构：
- DashboardEngine: 数据聚合核心，从各模块采集实时数据
- 四视图数据模型：AccountSnapshot / EquityCurve / StrategyComparison /
  PositionDistribution / RiskDashboard
- 支持缓存策略，减少重复计算
- 自动持久化账户快照到DB，供权益曲线回溯
- 策略推断：从策略管理器获取持仓归属，无需手动标注

v2.1 新增：
- 24h权益变化、当日已实现盈亏追踪
- 策略与持仓自动关联
- 账户快照自动写入DB（定期）
- 综合风险评分细化
- 多维度分布（按币种/方向/杠杆/策略/风险等级）

v3.0 新增：
- 历史表现统计（多周期 Sharpe/Sortino/Calmar）
- 资金效率指标（三级池使用率、闲置检测）
- 活跃告警系统（7类告警自动检测）
- 资金费率趋势分析（日/周费用预测）
- 订单簿深度分析（流动性评分、买卖比）
- 策略热力图（按策略聚合表现）
- 市场概览（BTC/ETH价格、市场状态、波动率）
"""

import json
import math
import os
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

# ─── 数据模型 ───


@dataclass
class AccountSnapshot:
    """账户快照 — 权益、资金、浮盈浮亏"""
    timestamp: str = ""
    total_equity: float = 0.0          # 总权益
    available_balance: float = 0.0     # 可用余额
    used_margin: float = 0.0           # 已用保证金
    frozen_balance: float = 0.0        # 冻结资金
    unrealized_pnl: float = 0.0        # 未实现盈亏
    realized_pnl_today: float = 0.0    # 今日已实现盈亏
    total_pnl_pct: float = 0.0         # 总收益率
    equity_change_24h: float = 0.0     # 24h权益变化
    equity_change_24h_pct: float = 0.0 # 24h权益变化率
    margin_utilization: float = 0.0    # 保证金使用率
    margin_ratio: float = 0.0          # 保证金率（越高越安全）
    maintenance_margin: float = 0.0    # 维持保证金
    health_level: str = "normal"       # normal/caution/danger
    is_stale: bool = False             # 是否为DB回退的陈旧快照（实时API失败时）


@dataclass
class EquityCurvePoint:
    """权益曲线数据点"""
    timestamp: str = ""
    equity: float = 0.0
    available: float = 0.0
    margin: float = 0.0
    unrealized_pnl: float = 0.0
    drawdown: float = 0.0              # 当前回撤比例
    drawdown_from_peak: float = 0.0    # 从历史最高回撤


@dataclass
class StrategyMetric:
    """策略指标"""
    name: str = ""
    total_pnl: float = 0.0
    pnl_pct: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_profit_per_trade: float = 0.0
    profit_factor: float = 0.0
    attribution_rate: float = 0.0       # 磨损率
    active_positions: int = 0
    position_margin: float = 0.0        # 持仓保证金
    status: str = "idle"               # running/idle/paused


@dataclass
class PositionItem:
    """持仓项"""
    symbol: str = ""
    symbol_base: str = ""               # 基础币种（去掉后缀）
    side: str = "long"                  # long/short
    quantity: float = 0.0
    entry_price: float = 0.0
    mark_price: float = 0.0
    liq_price: float = 0.0
    margin: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_pct: float = 0.0
    leverage: int = 1
    strategy: str = ""
    strategy_display: str = ""          # 策略显示名
    risk_score: float = 0.0
    risk_level: str = "safe"
    notional: float = 0.0               # 名义价值


@dataclass
class RiskMetrics:
    """风险指标集合"""
    var_95_daily: float = 0.0           # 95%置信度日VaR
    var_99_daily: float = 0.0
    cvar_95_daily: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    current_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    volatility_annual: float = 0.0
    beta: float = 0.0
    leverage_ratio: float = 0.0
    concentration_ratio: float = 0.0    # 最大单币种占比
    margin_utilization: float = 0.0
    liquidation_risk: float = 0.0       # 清算风险评分 0-100
    total_risk_score: float = 0.0       # 综合风险评分 0-100
    risk_level: str = "normal"          # normal/elevated/high/critical
    # 新增
    open_interest_ratio: float = 0.0    # 总持仓/总权益
    funding_rate_exposure: float = 0.0  # 资金费率敞口
    daily_pnl_volatility: float = 0.0   # 日盈亏波动率


@dataclass
class HistoricalPerformance:
    """历史表现 — 多周期统计"""
    period: str = ""                     # 7d/30d/90d/180d/all
    start_equity: float = 0.0
    end_equity: float = 0.0
    total_return: float = 0.0
    total_return_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    best_day: float = 0.0
    worst_day: float = 0.0
    avg_daily_return: float = 0.0
    volatility_annual: float = 0.0
    sortino_ratio: float = 0.0
    recovery_factor: float = 0.0


@dataclass
class CapitalEfficiencyMetrics:
    """资金效率指标"""
    total_capital: float = 0.0
    deployed_capital: float = 0.0       # 已部署资金
    idle_capital: float = 0.0           # 闲置资金
    efficiency_ratio: float = 0.0       # 资金使用效率 (deployed/total)
    base_pool_usage: float = 0.0        # 底仓池使用率
    add_pool_usage: float = 0.0         # 加仓池使用率
    risk_pool_usage: float = 0.0        # 风控池使用率
    idle_capital_cost: float = 0.0      # 闲置资金机会成本（日）
    optimal_deployment: float = 0.0     # 建议最优部署量
    rebalance_needed: bool = False      # 是否需要再平衡
    last_rebalance: str = ""


@dataclass
class ActiveAlert:
    """活跃告警"""
    id: str = ""
    type: str = ""                      # risk/margin/attrition/liquidation/system/performance
    severity: str = "info"             # info/warning/critical
    message: str = ""
    source: str = ""                    # 来源模块
    timestamp: str = ""
    value: float = 0.0
    threshold: float = 0.0
    acknowledged: bool = False


@dataclass
class FundingTrendItem:
    """资金费率趋势项"""
    symbol: str = ""
    current_rate: float = 0.0
    avg_rate_24h: float = 0.0
    predicted_rate: float = 0.0         # 预测下次费率
    next_settle_time: str = ""
    position_side: str = ""            # long/short/none
    daily_cost: float = 0.0            # 日资金费用
    weekly_cost: float = 0.0           # 周资金费用
    trend: str = "stable"              # rising/falling/stable


@dataclass
class OrderbookSummary:
    """订单簿摘要"""
    symbol: str = ""
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    spread_pct: float = 0.0
    bid_depth_1pct: float = 0.0        # 1%深度内买盘量
    ask_depth_1pct: float = 0.0        # 1%深度内卖盘量
    bid_ask_ratio: float = 0.0         # 买卖比
    orderbook_imbalance: float = 0.0   # 订单簿不平衡度 (-1到1)
    liquidity_score: float = 0.0       # 流动性评分 0-100


@dataclass
class StrategyHeatmapCell:
    """策略热力图单元"""
    strategy: str = ""
    symbol: str = ""
    period: str = ""                   # 7d/30d
    total_pnl: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    avg_pnl_per_trade: float = 0.0
    profit_factor: float = 0.0
    attribution_rate: float = 0.0
    margin_used: float = 0.0
    status: str = "idle"               # running/profitable/losing/paused


@dataclass
class MarketOverview:
    """市场概览"""
    btc_price: float = 0.0
    btc_change_24h: float = 0.0
    eth_price: float = 0.0
    eth_change_24h: float = 0.0
    total_market_cap: float = 0.0
    btc_dominance: float = 0.0
    fear_greed_index: int = 50
    market_regime: str = "neutral"     # bull/bear/neutral/volatile
    top_movers: List[Dict[str, Any]] = field(default_factory=list)
    vix_like: float = 0.0              # 加密市场波动率指数


class DashboardEngine:
    """
    生产级仪表板数据引擎

    使用示例:
        engine = DashboardEngine(config)
        engine.set_dependencies(client, capital_mgr, state_mgr)
        snapshot = engine.get_account_snapshot()
        full = engine.get_dashboard_full()

    v3.0: 新增历史表现、资金效率、告警、费率、订单簿、热力图、市场概览
    """

    # ─── 缓存配置 ───
    CACHE_TTL_SECONDS = 3.0             # 数据缓存时间
    EQUITY_CACHE_TTL = 60.0             # 权益曲线缓存时间（从DB读取，更新较慢）
    SNAPSHOT_PERSIST_INTERVAL = 300     # 快照持久化间隔（秒），默认5分钟

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        dash_cfg = self.config.get("dashboard", {})

        # ─── 依赖注入（延迟绑定） ───
        self._okx_client = None
        self._capital_manager = None
        self._state_manager = None
        self._account_manager = None
        self._strategy_manager = None
        self._attrition_analyzer = None
        self._agi_orchestrator = None

        # ─── 缓存 ───
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._cache_lock = threading.Lock()

        # ─── 历史权益峰值追踪 ───
        self._equity_peak: float = 0.0
        self._equity_peak_date: str = ""
        self._max_drawdown: float = 0.0
        self._max_drawdown_duration: int = 0
        self._drawdown_start_date: Optional[str] = None

        # ─── 24h权益变化追踪 ───
        self._equity_24h_ago: float = 0.0
        self._equity_24h_ago_ts: float = 0.0
        self._last_snapshot_persist: float = 0.0

        # ─── DB路径（解析为绝对路径，避免后台进程 cwd 不同导致读取错误 DB）───
        self._db_path = dash_cfg.get("db_path", "data/trading.db")
        if self._db_path and not os.path.isabs(self._db_path):
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._db_path = os.path.join(_project_root, self._db_path)

        # ─── 策略→币种映射（从 state_manager 动态加载） ───
        self._strategy_symbol_map: Dict[str, List[str]] = {}

        # ─── 确保DB表存在 ───
        self._ensure_db_tables()

        logger.info("DashboardEngine v3.0 initialized")

    # ═══════════════════════════════════════════════════════════════
    # 依赖注入
    # ═══════════════════════════════════════════════════════════════

    def set_dependencies(
        self,
        okx_client=None,
        capital_manager=None,
        state_manager=None,
        account_manager=None,
        strategy_manager=None,
        agi_orchestrator=None,
    ):
        """注入外部依赖"""
        self._okx_client = okx_client
        self._capital_manager = capital_manager
        self._state_manager = state_manager
        self._account_manager = account_manager
        self._strategy_manager = strategy_manager
        self._agi_orchestrator = agi_orchestrator

        # 从 capital_manager 获取 attrition_analyzer
        if capital_manager and hasattr(capital_manager, 'attrition_analyzer'):
            self._attrition_analyzer = capital_manager.attrition_analyzer

        # 从 state_manager 加载策略→币种映射
        self._load_strategy_symbol_map()

        logger.info(
            f"DashboardEngine dependencies set: "
            f"okx={'Y' if okx_client else 'N'}, "
            f"capital={'Y' if capital_manager else 'N'}, "
            f"state={'Y' if state_manager else 'N'}, "
            f"strategy={'Y' if strategy_manager else 'N'}"
        )

    def _load_strategy_symbol_map(self):
        """从 state_manager 加载策略→币种映射"""
        self._strategy_symbol_map = {}
        if self._state_manager:
            try:
                state_data = self._state_manager.load_all_states() if hasattr(self._state_manager, 'load_all_states') else {}
                for strategy_name, state in state_data.items():
                    if isinstance(state, dict):
                        symbols = state.get("symbols", []) or state.get("coins", [])
                        if not symbols and "symbol" in state:
                            symbols = [state["symbol"]]
                        if symbols:
                            self._strategy_symbol_map[strategy_name] = symbols
                if self._strategy_symbol_map:
                    logger.info(f"Strategy-symbol map loaded: {self._strategy_symbol_map}")
            except Exception as e:
                logger.debug(f"Could not load strategy-symbol map: {e}")

    def _ensure_db_tables(self):
        """确保数据库表存在（与主进程 ORM account_history schema 对齐）"""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_history (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    total_equity REAL DEFAULT 0,
                    available_balance REAL DEFAULT 0,
                    used_margin REAL DEFAULT 0,
                    unrealized_pnl REAL DEFAULT 0,
                    margin_rate REAL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_account_history_ts
                ON account_history(timestamp)
            """)
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"DashboardEngine: could not ensure DB tables: {e}")

    # ═══════════════════════════════════════════════════════════════
    # 视图一：账户全景快照
    # ═══════════════════════════════════════════════════════════════

    def get_account_snapshot(self, force_refresh: bool = False) -> AccountSnapshot:
        """获取账户实时快照 — 权益、可用资金、浮盈浮亏"""
        cache_key = "account_snapshot"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            # 从 OKX API 获取实时数据
            account_data = self._fetch_okx_account()
            positions_data = self._fetch_okx_positions()

            snapshot = AccountSnapshot(timestamp=datetime.now().isoformat())

            if account_data:
                # 解析账户数据
                snapshot.total_equity = float(account_data.get("totalEq", 0) or 0)
                snapshot.frozen_balance = 0.0

                for detail in account_data.get("details", []):
                    if detail.get("ccy") == "USDT":
                        snapshot.available_balance = float(detail.get("availBal", 0) or 0)
                        snapshot.frozen_balance = float(detail.get("frozenBal", 0) or 0)
                        if snapshot.total_equity <= 0:
                            snapshot.total_equity = float(detail.get("eq", 0) or 0)
                        break

                # 汇总持仓数据
                total_margin = 0.0
                total_upl = 0.0
                total_maint_margin = 0.0
                for pos in positions_data:
                    total_margin += self._extract_position_margin(pos)
                    total_upl += float(pos.get("upl", 0) or 0)
                    total_maint_margin += float(pos.get("mmr", 0) or 0)

                snapshot.used_margin = total_margin
                snapshot.unrealized_pnl = total_upl
                snapshot.maintenance_margin = total_maint_margin

                # 计算比率
                if snapshot.total_equity > 0:
                    snapshot.margin_utilization = total_margin / snapshot.total_equity
                    snapshot.margin_ratio = (snapshot.total_equity - total_margin) / snapshot.total_equity
                    snapshot.total_pnl_pct = total_upl / (snapshot.total_equity - total_upl + 0.01)

                # 健康等级
                snapshot.health_level = self._assess_account_health(snapshot)

            # 实时 API 失败/未连接时，回退到 DB 最近一次快照（主交易进程持续写入）
            if snapshot.total_equity <= 0:
                fallback = self._load_latest_snapshot_from_db()
                if fallback:
                    snapshot = fallback

            # ─── 24h权益变化 ───
            snapshot.equity_change_24h, snapshot.equity_change_24h_pct = self._calc_24h_change(snapshot)

            # ─── 今日已实现盈亏 ───
            snapshot.realized_pnl_today = self._get_today_realized_pnl()

            # 追踪历史最高权益
            if snapshot.total_equity > self._equity_peak:
                self._equity_peak = snapshot.total_equity
                self._equity_peak_date = snapshot.timestamp[:10]
                self._drawdown_start_date = None
            elif self._equity_peak > 0:
                current_dd = (self._equity_peak - snapshot.total_equity) / self._equity_peak
                if current_dd > self._max_drawdown:
                    self._max_drawdown = current_dd

            # ─── 自动持久化快照到DB ───
            self._maybe_persist_snapshot(snapshot)

            self._set_cache(cache_key, snapshot, self.CACHE_TTL_SECONDS)
            return snapshot

        except Exception as e:
            logger.error(f"DashboardEngine: account snapshot error: {e}")
            return AccountSnapshot(timestamp=datetime.now().isoformat())

    # ═══════════════════════════════════════════════════════════════
    # 视图二：权益曲线
    # ═══════════════════════════════════════════════════════════════

    def get_equity_curve(
        self, days: int = 30, force_refresh: bool = False
    ) -> Dict[str, Any]:
        """获取权益曲线 — 历史权益、回撤、浮盈浮亏走势"""
        cache_key = f"equity_curve_{days}"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            points: List[EquityCurvePoint] = []
            peak = 0.0
            max_dd = 0.0

            # 从DB读取历史数据
            conn = self._get_db()
            if conn:
                cursor = conn.cursor()
                cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
                cursor.execute(
                    "SELECT timestamp, total_equity, available_balance, used_margin, unrealized_pnl "
                    "FROM account_history WHERE timestamp >= ? ORDER BY timestamp ASC",
                    (cutoff,)
                )
                for row in cursor.fetchall():
                    # sqlite3.Row 支持键访问（类似dict），但 isinstance(row, dict) 为 False
                    is_dict_like = isinstance(row, (dict, sqlite3.Row))
                    if is_dict_like:
                        equity = row["total_equity"]
                        ts = row["timestamp"]
                        avail = row["available_balance"] if "available_balance" in row.keys() else 0
                        margin_val = row["used_margin"] if "used_margin" in row.keys() else 0
                        upl_val = row["unrealized_pnl"] if "unrealized_pnl" in row.keys() else 0
                    elif hasattr(row, '__getitem__'):
                        # 回退到索引访问：col0=timestamp, col1=total_equity, col2=available_balance, ...
                        equity = row[1]
                        ts = str(row[0]) if row else ""
                        avail = row[2] if len(row) > 2 else 0
                        margin_val = row[3] if len(row) > 3 else 0
                        upl_val = row[4] if len(row) > 4 else 0
                    else:
                        continue

                    equity = float(equity) if equity else 0
                    if equity > 0:
                        peak = max(peak, equity)
                        dd = (peak - equity) / peak if peak > 0 else 0
                        max_dd = max(max_dd, dd)
                        points.append(EquityCurvePoint(
                            timestamp=str(ts) if ts else "",
                            equity=equity,
                            available=float(avail) if avail else 0,
                            margin=float(margin_val) if margin_val else 0,
                            unrealized_pnl=float(upl_val) if upl_val else 0,
                            drawdown=dd,
                            drawdown_from_peak=dd,
                        ))
                conn.close()

            # 如果DB无数据，补充当前快照
            if not points:
                snapshot = self.get_account_snapshot(force_refresh=True)
                if snapshot.total_equity > 0:
                    points.append(EquityCurvePoint(
                        timestamp=snapshot.timestamp,
                        equity=snapshot.total_equity,
                        available=snapshot.available_balance,
                        margin=snapshot.used_margin,
                        unrealized_pnl=snapshot.unrealized_pnl,
                        drawdown=0,
                        drawdown_from_peak=0,
                    ))

            # 降采样：权益曲线点数过多时均匀抽样，保留首尾与最大回撤点，
            # 避免超大 JSON 载荷（近 10 万点约 4.7MB）导致前端渲染卡顿、SSE 连接超时/重置
            MAX_CURVE_POINTS = 500
            if len(points) > MAX_CURVE_POINTS:
                step = len(points) / MAX_CURVE_POINTS
                sampled = [points[int(i * step)] for i in range(MAX_CURVE_POINTS)]
                sampled.append(max(points, key=lambda p: p.drawdown))  # 保留最大回撤点
                sampled.append(points[-1])  # 保留最新点
                seen = set()
                deduped = []
                for p in sampled:
                    if p.timestamp not in seen:
                        seen.add(p.timestamp)
                        deduped.append(p)
                deduped.sort(key=lambda p: p.timestamp)
                points = deduped

            # 统一回撤口径：summary.max_drawdown 改用按日聚合 + 出入金检测口径，
            # 与历史表现 / 风险仪表盘一致；分钟级原始回撤保留在 intraday_max_drawdown。
            daily_data = self._fetch_daily_series(days)
            stat_max_dd = self._capital_adjusted_max_drawdown(
                daily_data["equities"], daily_data["flow_days"]
            )

            result = {
                "points": [
                    {
                        "t": p.timestamp,
                        "e": round(p.equity, 2),
                        "a": round(p.available, 2),
                        "m": round(p.margin, 2),
                        "u": round(p.unrealized_pnl, 2),
                        "dd": round(p.drawdown, 4),
                    }
                    for p in points
                ],
                "summary": {
                    "current_equity": round(points[-1].equity, 2) if points else 0,
                    "peak_equity": round(peak, 2),
                    "max_drawdown": round(stat_max_dd, 4),
                    "intraday_max_drawdown": round(max_dd, 4),
                    "data_points": len(points),
                    "period_days": days,
                },
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.EQUITY_CACHE_TTL)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: equity curve error: {e}")
            return {"points": [], "summary": {}, "timestamp": datetime.now().isoformat()}

    # ═══════════════════════════════════════════════════════════════
    # 视图三：策略对比
    # ═══════════════════════════════════════════════════════════════

    def get_strategy_comparison(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取策略对比 — 收益、回撤、胜率、夏普比率、磨损率"""
        cache_key = "strategy_comparison"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            strategies: List[StrategyMetric] = []

            # 从DB获取策略表现（容错：trades 表可能不存在）
            conn = self._get_db()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT strategy_name, SUM(pnl) as total_pnl, "
                        "COUNT(*) as trade_count, "
                        "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins "
                        "FROM trade_records WHERE status = 'closed' AND create_time >= ? "
                        "GROUP BY strategy_name",
                        ((datetime.now() - timedelta(days=30)).isoformat(),)
                    )
                    for row in cursor.fetchall():
                        is_dict_like = isinstance(row, (dict, sqlite3.Row))
                        name = row["strategy_name"] if is_dict_like else row[0]
                        # 排除持仓同步伪策略（sync 非真实交易策略，不应参与策略排名）
                        if not name or str(name).strip().lower() == "sync":
                            continue
                        total_pnl = float(row["total_pnl"] or 0) if is_dict_like else float(row[1] or 0)
                        trade_count = int(row["trade_count"] or 0) if is_dict_like else int(row[2] or 0)
                        wins = int(row["wins"] or 0) if is_dict_like else int(row[3] or 0)

                        metric = StrategyMetric(
                            name=name,
                            total_pnl=round(total_pnl, 4),
                            trade_count=trade_count,
                            win_rate=round(wins / max(trade_count, 1), 4),
                            avg_profit_per_trade=round(total_pnl / max(trade_count, 1), 4),
                            profit_factor=self._calc_profit_factor(conn, name),
                            status="running",
                        )
                        strategies.append(metric)
                except (sqlite3.OperationalError, Exception):
                    pass  # trades 表不存在，跳过策略统计
                finally:
                    conn.close()

            # 补充磨损数据
            if self._attrition_analyzer:
                att_stats = self._attrition_analyzer.get_attrition_stats(period_hours=24 * 30)
                strategy_ranking = self._attrition_analyzer._get_strategy_attrition_ranking(att_stats)
                ranking_map = {s["name"]: s for s in strategy_ranking}
                for s in strategies:
                    if s.name in ranking_map:
                        s.attribution_rate = ranking_map[s.name].get("attrition_rate", 0)

            # 补充当前活跃持仓（按策略映射）
            positions = self._fetch_okx_positions()
            strategy_positions: Dict[str, int] = defaultdict(int)
            strategy_margins: Dict[str, float] = defaultdict(float)
            for pos in positions:
                pnl = float(pos.get("upl", 0) or 0)
                if pnl != 0 or float(pos.get("pos", 0) or 0) != 0:
                    symbol = pos.get("instId", "")
                    symbol_base = symbol.replace("-USDT-SWAP", "").replace("-USDC-SWAP", "")
                    strat_name, _ = self._infer_position_strategy(symbol, symbol_base)
                    strategy_positions[strat_name] += 1
                    strategy_margins[strat_name] += self._extract_position_margin(pos)
            for s in strategies:
                s.active_positions = strategy_positions.get(s.name, 0)
                s.position_margin = round(strategy_margins.get(s.name, 0), 2)

            result = {
                "strategies": [
                    {
                        "name": s.name,
                        "total_pnl": s.total_pnl,
                        "win_rate": s.win_rate,
                        "trade_count": s.trade_count,
                        "avg_profit_per_trade": s.avg_profit_per_trade,
                        "profit_factor": s.profit_factor,
                        "attribution_rate": s.attribution_rate,
                        "active_positions": s.active_positions,
                        "position_margin": s.position_margin,
                        "status": s.status,
                    }
                    for s in strategies
                ],
                "summary": {
                    "total_strategies": len(strategies),
                    "total_pnl": round(sum(s.total_pnl for s in strategies), 4),
                    "avg_win_rate": round(
                        sum(s.win_rate for s in strategies) / max(len(strategies), 1), 4
                    ),
                    "best_strategy": max(strategies, key=lambda s: s.total_pnl).name if strategies else "",
                    "worst_strategy": min(strategies, key=lambda s: s.total_pnl).name if strategies else "",
                },
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: strategy comparison error: {e}")
            return {"strategies": [], "summary": {}, "timestamp": datetime.now().isoformat()}

    # ═══════════════════════════════════════════════════════════════
    # 视图四：持仓分布 + 风险水位
    # ═══════════════════════════════════════════════════════════════

    def get_position_distribution(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取持仓分布 — 按币种/方向/杠杆/策略/风险等级的多维分布"""
        cache_key = "position_distribution"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            positions_data = self._fetch_okx_positions()
            account_data = self._fetch_okx_account()
            total_equity = float(account_data.get("totalEq", 1)) if account_data else 1

            items: List[PositionItem] = []
            # 多维聚合
            by_symbol: Dict[str, Dict[str, float]] = defaultdict(lambda: {"margin": 0, "pnl": 0, "count": 0, "notional": 0})
            by_side: Dict[str, float] = defaultdict(float)
            by_leverage: Dict[str, Dict[str, float]] = defaultdict(lambda: {"margin": 0, "count": 0})
            by_strategy: Dict[str, Dict[str, float]] = defaultdict(lambda: {"margin": 0, "pnl": 0, "count": 0})
            by_risk_level: Dict[str, Dict[str, float]] = defaultdict(lambda: {"margin": 0, "count": 0})

            for pos in positions_data:
                qty = float(pos.get("pos", 0) or 0)
                if qty == 0:
                    continue

                symbol = pos.get("instId", "")
                side = pos.get("posSide", "long")
                margin = self._extract_position_margin(pos)
                upl = float(pos.get("upl", 0) or 0)
                mark_px = float(pos.get("markPx", 0) or 0)
                liq_px = float(pos.get("liqPx", 0) or 0)
                avg_px = float(pos.get("avgPx", 0) or 0)
                lever = int(pos.get("lever", 1) or 1)
                notional = float(pos.get("notionalUsd", 0) or 0)

                # pos 字段是合约张数；前端"数量"需展示币数（与 trade_records 口径一致）。
                # 线性U本位：币数 = 名义价值 / 标记价，等价于 pos × ctVal，且无需逐币查询 ctVal。
                coin_qty = abs(notional) / mark_px if mark_px > 0 else abs(qty)

                # 币种基础名
                symbol_base = symbol.replace("-USDT-SWAP", "").replace("-USDC-SWAP", "")

                # ─── 策略推断 ───
                strategy_name, strategy_display = self._infer_position_strategy(symbol, symbol_base)

                # 风险评分
                risk_score = self._calc_position_risk(liq_px, mark_px, margin, total_equity, upl, lever)
                risk_level = "danger" if risk_score >= 60 else "warning" if risk_score >= 35 else "caution" if risk_score >= 15 else "safe"

                item = PositionItem(
                    symbol=symbol, symbol_base=symbol_base,
                    side=side, quantity=round(coin_qty, 8),
                    entry_price=avg_px, mark_price=mark_px, liq_price=liq_px,
                    margin=margin, unrealized_pnl=upl,
                    pnl_pct=round(upl / (margin if margin > 0 else (notional if notional > 0 else 1.0)) * 100, 2),
                    leverage=lever,
                    strategy=strategy_name, strategy_display=strategy_display,
                    risk_score=risk_score, risk_level=risk_level,
                    notional=notional,
                )
                items.append(item)

                # 按币种聚合
                key_symbol = symbol_base
                by_symbol[key_symbol]["margin"] += margin
                by_symbol[key_symbol]["pnl"] += upl
                by_symbol[key_symbol]["count"] += 1
                by_symbol[key_symbol]["notional"] += notional

                # 按方向聚合
                by_side[side] += margin

                # 按杠杆聚合
                lev_key = f"{lever}x"
                by_leverage[lev_key]["margin"] += margin
                by_leverage[lev_key]["count"] += 1

                # 按策略聚合
                by_strategy[strategy_display]["margin"] += margin
                by_strategy[strategy_display]["pnl"] += upl
                by_strategy[strategy_display]["count"] += 1

                # 按风险等级聚合
                by_risk_level[risk_level]["margin"] += margin
                by_risk_level[risk_level]["count"] += 1

            # 计算集中度
            max_symbol_margin = max((v["margin"] for v in by_symbol.values()), default=0)
            concentration = max_symbol_margin / total_equity if total_equity > 0 else 0

            result = {
                "positions": [
                    {
                        "symbol": p.symbol,
                        "symbol_base": p.symbol_base,
                        "side": p.side,
                        "quantity": p.quantity,
                        "entry_price": p.entry_price,
                        "mark_price": p.mark_price,
                        "liq_price": p.liq_price,
                        "margin": round(p.margin, 2),
                        "unrealized_pnl": round(p.unrealized_pnl, 4),
                        "pnl_pct": p.pnl_pct,
                        "leverage": p.leverage,
                        "strategy": p.strategy,
                        "strategy_display": p.strategy_display,
                        "risk_score": p.risk_score,
                        "risk_level": p.risk_level,
                        "notional": round(p.notional, 2),
                    }
                    for p in sorted(items, key=lambda x: x.risk_score, reverse=True)
                ],
                "distribution": {
                    "by_symbol": {
                        k: {"margin": round(v["margin"], 2), "pnl": round(v["pnl"], 4),
                            "count": int(v["count"]), "notional": round(v["notional"], 2)}
                        for k, v in sorted(by_symbol.items(), key=lambda x: x[1]["margin"], reverse=True)
                    },
                    "by_side": {k: round(v, 2) for k, v in by_side.items()},
                    "by_leverage": {
                        k: {"margin": round(v["margin"], 2), "count": int(v["count"])}
                        for k, v in sorted(by_leverage.items(), key=lambda x: float(x[0].replace("x", "")), reverse=True)
                    },
                    "by_strategy": {
                        k: {"margin": round(v["margin"], 2), "pnl": round(v["pnl"], 4), "count": int(v["count"])}
                        for k, v in sorted(by_strategy.items(), key=lambda x: x[1]["margin"], reverse=True)
                    },
                    "by_risk_level": {
                        k: {"margin": round(v["margin"], 2), "count": int(v["count"])}
                        for k, v in by_risk_level.items()
                    },
                },
                "summary": {
                    "total_positions": len(items),
                    "total_margin": round(sum(p.margin for p in items), 2),
                    "total_notional": round(sum(p.notional for p in items), 2),
                    "total_unrealized_pnl": round(sum(p.unrealized_pnl for p in items), 4),
                    "concentration_ratio": round(concentration, 4),
                    "danger_positions": sum(1 for p in items if p.risk_level == "danger"),
                    "warning_positions": sum(1 for p in items if p.risk_level == "warning"),
                    "long_count": sum(1 for p in items if p.side == "long"),
                    "short_count": sum(1 for p in items if p.side == "short"),
                    "avg_leverage": round(sum(p.leverage for p in items) / max(len(items), 1), 1),
                },
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: position distribution error: {e}")
            return {"positions": [], "distribution": {}, "summary": {}, "timestamp": datetime.now().isoformat()}

    @staticmethod
    def _parse_history_timestamp(ts: Any) -> Optional[datetime]:
        """解析 account_history 时间戳，兼容空格与 T 两种格式。

        历史数据中同时存在 "2026-08-15 12:00:00" 与 "2026-08-15T12:00:00"
        两种格式，SQL 字符串排序会把 T 格式排在空格格式之后，导致同日内
        快照顺序错乱。统一解析为真实 datetime 后再排序。
        """
        if not ts:
            return None
        # fromisoformat 同时兼容 "T"/空格分隔与可选微秒，比逐格式 strptime
        # 快约 20 倍，避免对近 10 万行快照逐个 strptime 造成 CPU 峰值。
        try:
            return datetime.fromisoformat(str(ts))
        except ValueError:
            return None

    @staticmethod
    def _is_capital_flow_event(
        prev_eq: float, curr_eq: float,
        prev_avail: float, curr_avail: float,
        prev_upl: float = 0.0, curr_upl: float = 0.0,
    ) -> bool:
        """判断相邻两个分钟级快照之间是否发生出入金（资金实际注入/提取）。

        出入金必须同时满足三个特征（区别于正常开平仓与行情波动）：
        1. 权益相对上一快照变化 >20%（|Δequity| / prev_equity > 0.20）——
           正常开平仓与行情波动在单个快照间隔内无法产生如此大的权益跳变，
           只有真实的资金注入/提取才会。实际出入金均 >90%（165%/412%/99%/1038%），
           而交易最大单步仅 ~10%，20% 阈值留出足够安全边际；
        2. 可用余额与权益同向、近似同幅度变化（Δavail ≈ Δequity，入金/提现
           金额同时体现在可用余额与权益上），容差允许同区间内的小额保证金变动；
        3. 未实现盈亏基本不变（|Δupl| <= 1 USDT，排除行情波动与平仓）。

        采用相对阈值而非时间窗口：跨期出入金（如 7-20 +412% 间隔 4 小时、
        7-27 -99% 跨午夜）依然能被捕获；而 7-21/7-24 的正常交易累计（<5%）
        不会被误判为出入金。正常开平仓会在可用余额与已用保证金之间转移资金，
        使 Δavail 与 Δequity 显著背离，也不会被误判。
        """
        d_eq = curr_eq - prev_eq
        d_av = curr_avail - prev_avail
        d_up = curr_upl - prev_upl
        if prev_eq <= 0 or abs(d_eq) < 1.0:
            return False
        # 相对权益变化必须 >20%，否则视为正常交易/行情波动
        if abs(d_eq) / prev_eq <= 0.20:
            return False
        # 未实现盈亏基本不变（紧的绝对容差）
        if abs(d_up) > 1.0:
            return False
        # 可用余额与权益同向、近似同幅度变化（宽松相对容差）
        tol_av = max(1.0, 0.30 * abs(d_eq))
        if abs(d_av - d_eq) > tol_av:
            return False
        return True

    @classmethod
    def _capital_adjusted_peak(cls, equities: List[Tuple[str, float]], flow_days: set) -> float:
        """计算剔除出入金影响后的当前权益峰值基准。

        equities 为按日升序的 (日期, 权益) 序列；flow_days 为发生出入金的
        日期集合。出入金日会重置峰值基准，使回撤仅衡量真实交易盈亏。
        """
        if not equities:
            return 0.0
        peak = equities[0][1]
        for i in range(1, len(equities)):
            day, eq = equities[i]
            if day in flow_days:
                peak = eq
            else:
                peak = max(peak, eq)
        return peak

    @classmethod
    def _capital_adjusted_max_drawdown(cls, equities: List[Tuple[str, float]], flow_days: set) -> float:
        """计算剔除出入金影响后的历史最大回撤（按日聚合口径）。

        equities 为按日升序的 (日期, 权益) 序列；出入金日重置峰值基准。
        """
        if not equities:
            return 0.0
        peak = equities[0][1]
        max_dd = 0.0
        for i in range(1, len(equities)):
            day, eq = equities[i]
            if day in flow_days:
                peak = eq
            else:
                peak = max(peak, eq)
            if peak > 0:
                max_dd = max(max_dd, (peak - eq) / peak)
        return max_dd

    @classmethod
    def _capital_adjusted_drawdown_duration(cls, equities: List[Tuple[str, float]], flow_days: set) -> int:
        """计算剔除出入金影响后的最大回撤持续期（按日聚合，单位：天）。

        与 _capital_adjusted_max_drawdown 采用相同的出入金重置逻辑，
        额外追踪权益持续低于峰值的最长区间长度。
        """
        if len(equities) < 2:
            return 0
        peak = equities[0][1]
        max_duration = 0
        dd_start = None
        for i in range(1, len(equities)):
            day, eq = equities[i]
            if day in flow_days:
                peak = eq
                dd_start = None
            else:
                peak = max(peak, eq)
            if peak > 0 and eq < peak:
                if dd_start is None:
                    dd_start = i
                max_duration = max(max_duration, i - dd_start)
            else:
                dd_start = None
        return max_duration

    def _fetch_daily_series(self, days: int) -> Dict[str, Any]:
        """读取按日聚合的权益序列，并在分钟级检测出入金。

        统一口径核心：先按真实时间（兼容 T/空格格式）排序分钟级快照，
        再逐对检测出入金事件（基于相对权益跳变的分钟级签名），最后按日
        取最后一条快照得到当日权益。返回:
        {
            "equities": [(日期, 权益), ...] 按日升序,
            "flow_days": set(日期)  # 当日发生出入金的日期集合,
        }
        """
        # 实例级缓存：account_history 为 append-only，聚合结果短时间内稳定，
        # full 请求内多个模块（equity_curve/risk/historical）会重复调用，
        # 加缓存避免反复读取近 10 万行快照导致 /api/dashboard/full 超时。
        cache = getattr(self, "_daily_series_cache", None)
        if cache is None:
            cache = {}
            self._daily_series_cache = cache
        entry = cache.get(days)
        if entry and time.time() - entry[0] < 30.0:
            return entry[1]

        conn = self._get_db()
        if not conn:
            result = {"equities": [], "flow_days": set()}
            cache[days] = (time.time(), result)
            return result
        try:
            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT timestamp, total_equity, available_balance, unrealized_pnl "
                "FROM account_history WHERE timestamp >= ? ORDER BY timestamp ASC",
                (cutoff,)
            )
            rows = cursor.fetchall()
        finally:
            conn.close()

        # 解析并按真实时间排序（避免 T 格式记录因字符串排序被排到末尾）
        parsed: List[Tuple[datetime, float, float, float]] = []
        for r in rows:
            is_dict_like = isinstance(r, (dict, sqlite3.Row))
            ts_raw = r["timestamp"] if is_dict_like else r[0]
            eq = r["total_equity"] if is_dict_like else r[1]
            av = r["available_balance"] if is_dict_like else r[2]
            up = r["unrealized_pnl"] if is_dict_like else r[3]
            t = self._parse_history_timestamp(ts_raw)
            if t is None:
                continue
            eq = float(eq) if eq else 0.0
            av = float(av) if av else 0.0
            up = float(up) if up else 0.0
            if eq <= 0:
                continue
            parsed.append((t, eq, av, up))
        parsed.sort(key=lambda x: x[0])

        # 分钟级出入金检测：归入事件发生当日
        flow_days: set = set()
        for i in range(1, len(parsed)):
            pt, pe, pa, pu = parsed[i - 1]
            ct, ce, ca, cu = parsed[i]
            if self._is_capital_flow_event(pe, ce, pa, ca, pu, cu):
                flow_days.add(ct.strftime("%Y-%m-%d"))

        # 每日取最后一条快照作为当日权益
        daily: Dict[str, float] = {}
        for t, eq, _av, _up in parsed:
            daily[t.strftime("%Y-%m-%d")] = eq

        equities = [(d, daily[d]) for d in sorted(daily.keys())]
        result = {"equities": equities, "flow_days": flow_days}
        cache[days] = (time.time(), result)
        return result

    def get_risk_dashboard(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取风险水位 — 综合风险评估仪表盘"""
        cache_key = "risk_dashboard"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            # 复用各子模块内置缓存，避免每次冷算重读近 10 万行快照 / 重复拉取 OKX，
            # 从而消除 SSE 与 /full 冷启动时的 10s+ 超时。
            snapshot = self.get_account_snapshot()
            equity_curve = self.get_equity_curve(days=30)
            pos_dist = self.get_position_distribution()

            metrics = RiskMetrics()

            # 1. 当前回撤 / 最大回撤（统一按日聚合口径，剔除出入金影响）
            points = equity_curve.get("points", [])
            daily_data = self._fetch_daily_series(days=30)
            daily_equities = daily_data["equities"]
            flow_days = daily_data["flow_days"]
            if daily_equities:
                current = daily_equities[-1][1]
                peak = self._capital_adjusted_peak(daily_equities, flow_days)
                metrics.current_drawdown = (peak - current) / peak if peak > 0 else 0.0
                metrics.max_drawdown = self._capital_adjusted_max_drawdown(daily_equities, flow_days)

            # 2. 保证金使用率
            metrics.margin_utilization = snapshot.margin_utilization

            # 3. 杠杆率（总保证金/总权益）
            metrics.leverage_ratio = snapshot.margin_utilization

            # 4. 集中度
            metrics.concentration_ratio = pos_dist.get("summary", {}).get("concentration_ratio", 0)

            # 5. 清算风险评分
            metrics.liquidation_risk = self._calc_liquidation_risk(pos_dist)

            # 6. 波动率（从权益曲线估算）
            if len(points) >= 5:
                returns = []
                for i in range(1, len(points)):
                    if points[i-1]["e"] > 0:
                        r = (points[i]["e"] - points[i-1]["e"]) / points[i-1]["e"]
                        returns.append(r)
                if returns:
                    import statistics
                    daily_vol = statistics.stdev(returns) if len(returns) > 1 else 0
                    metrics.volatility_annual = daily_vol * math.sqrt(365)

            # 7. VaR估算（简化：波动率法）
            if metrics.volatility_annual > 0 and snapshot.total_equity > 0:
                metrics.var_95_daily = snapshot.total_equity * metrics.volatility_annual * 1.645 / math.sqrt(365)
                metrics.var_99_daily = snapshot.total_equity * metrics.volatility_annual * 2.326 / math.sqrt(365)
                metrics.cvar_95_daily = metrics.var_95_daily * 1.4  # CVaR ≈ 1.4x VaR

            # 8. 综合风险评分（0-100）
            metrics.total_risk_score = self._calc_total_risk_score(metrics, snapshot)
            metrics.risk_level = (
                "critical" if metrics.total_risk_score >= 70 else
                "high" if metrics.total_risk_score >= 45 else
                "elevated" if metrics.total_risk_score >= 25 else
                "normal"
            )

            result = {
                "risk_metrics": {
                    "var_95_daily": round(metrics.var_95_daily, 2),
                    "var_99_daily": round(metrics.var_99_daily, 2),
                    "cvar_95_daily": round(metrics.cvar_95_daily, 2),
                    "max_drawdown": round(metrics.max_drawdown, 4),
                    "current_drawdown": round(metrics.current_drawdown, 4),
                    "volatility_annual": round(metrics.volatility_annual, 4),
                    "leverage_ratio": round(metrics.leverage_ratio, 4),
                    "concentration_ratio": round(metrics.concentration_ratio, 4),
                    "margin_utilization": round(metrics.margin_utilization, 4),
                    "liquidation_risk": round(metrics.liquidation_risk, 1),
                    "total_risk_score": round(metrics.total_risk_score, 1),
                    "risk_level": metrics.risk_level,
                },
                "gauges": {
                    "margin": {
                        "value": round(metrics.margin_utilization * 100, 1),
                        "label": "保证金使用率",
                        "max": 100,
                        "warning": 70,
                        "danger": 85,
                    },
                    "drawdown": {
                        "value": round(metrics.current_drawdown * 100, 1),
                        "label": "当前回撤",
                        "max": 100,
                        "warning": 15,
                        "danger": 25,
                    },
                    "concentration": {
                        "value": round(metrics.concentration_ratio * 100, 1),
                        "label": "持仓集中度",
                        "max": 100,
                        "warning": 30,
                        "danger": 50,
                    },
                    "liquidation": {
                        "value": round(metrics.liquidation_risk, 1),
                        "label": "清算风险",
                        "max": 100,
                        "warning": 40,
                        "danger": 60,
                    },
                    "volatility": {
                        "value": round(metrics.volatility_annual * 100, 1),
                        "label": "年化波动率",
                        "max": 200,
                        "warning": 80,
                        "danger": 120,
                    },
                    "risk_score": {
                        "value": round(metrics.total_risk_score, 1),
                        "label": "综合风险评分",
                        "max": 100,
                        "warning": 45,
                        "danger": 70,
                    },
                },
                "account_health": {
                    "level": snapshot.health_level,
                    "total_equity": round(snapshot.total_equity, 2),
                    "margin_ratio": round(snapshot.margin_ratio, 4),
                    "available_balance": round(snapshot.available_balance, 2),
                },
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS * 5)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: risk dashboard error: {e}")
            return {"risk_metrics": {}, "gauges": {}, "timestamp": datetime.now().isoformat()}

    # ═══════════════════════════════════════════════════════════════
    # 一站式全量获取
    # ═══════════════════════════════════════════════════════════════

    def get_pnl_projection_panel(self) -> Dict[str, Any]:
        """AGI 前瞻推算与四维归因面板（V5.0 新增）。

        从注入的 agi_orchestrator 拉取最近一次推算/归因/准确度数据，
        整理为 JSON 安全的字典供前端渲染。
        fail-closed：orchestrator 未注入或尚无推算数据 → {"available": False}
        """
        orch = self._agi_orchestrator
        if orch is None:
            return {"available": False, "reason": "agi_orchestrator not injected"}

        snapshot_getter = getattr(orch, "get_dashboard_snapshot", None)
        if not callable(snapshot_getter):
            logger.error("DashboardEngine: injected AGI orchestrator lacks get_dashboard_snapshot")
            return {"available": False, "reason": "incompatible agi_orchestrator interface"}
        snapshot = snapshot_getter()
        if not isinstance(snapshot, dict):
            logger.error("DashboardEngine: AGI dashboard snapshot is not a dictionary")
            return {"available": False, "reason": "invalid agi_orchestrator snapshot"}

        last_proj = snapshot.get("projection")
        last_proj = last_proj if isinstance(last_proj, dict) else {}
        projection_available = bool(last_proj.get("available"))
        raw_attribution = snapshot.get("attribution")
        raw_attribution = raw_attribution if isinstance(raw_attribution, dict) else {}
        attribution_available = bool(raw_attribution.get("available"))
        if not projection_available and not attribution_available:
            return {"available": False, "reason": "no projection or attribution yet"}

        def _safe_float(x, default=0.0):
            try:
                v = float(x)
                return v if v == v and abs(v) < 1e15 else default  # NaN/Inf 守护
            except (TypeError, ValueError):
                return default

        # 三场景汇总
        total_scenarios = {}
        for case in ("base_case", "bear_case", "bull_case"):
            total = last_proj.get("total")
            c = total.get(case) if isinstance(total, dict) else {}
            c = c if isinstance(c, dict) else {}
            total_scenarios[case] = {
                "per_cycle": _safe_float(c.get("per_cycle")),
                "horizon": _safe_float(c.get("horizon")),
                "ci_lower": _safe_float(c.get("ci_lower")),
                "ci_upper": _safe_float(c.get("ci_upper")),
            }

        # per_strategy 摘要
        per_strategy = {}
        raw_per_strategy = last_proj.get("per_strategy")
        for name, sp in (raw_per_strategy.items() if isinstance(raw_per_strategy, dict) else []):
            if not isinstance(sp, dict):
                continue
            per_strategy[str(name)] = {
                "base_expect": _safe_float(sp.get("base_expect")),
                "trend_adjustment": _safe_float(sp.get("trend_adjustment")),
                "regime_mult": _safe_float(sp.get("regime_mult")),
                "corrected_expect": _safe_float(sp.get("corrected_expect")),
                "horizon": _safe_float(sp.get("horizon")),
                "bear_case_horizon": _safe_float(sp.get("bear_case_horizon")),
                "confidence": _safe_float(sp.get("confidence")),
            }

        # regime 条件化校正因子
        cfbr = snapshot.get("correction_factor_by_regime")
        correction_by_regime = {
            str(k): _safe_float(v, 1.0)
            for k, v in cfbr.items()
        } if isinstance(cfbr, dict) else {}

        # 准确度历史
        accuracy_hist = []
        pa = snapshot.get("projection_accuracy")
        if isinstance(pa, list):
            for x in pa[-20:]:
                if isinstance(x, dict):
                    accuracy_hist.append({
                        "cycle": int(_safe_float(x.get("cycle"), 0)),
                        "projected": _safe_float(x.get("projected")),
                        "actual": _safe_float(x.get("actual")),
                        "bias_ratio": _safe_float(x.get("bias_ratio"), 1.0),
                        "regime": str(x.get("regime", "unknown")),
                    })
        bias_history = [item["bias_ratio"] for item in accuracy_hist]

        # PnL attribution by strategy, regime, direction and exit reason.
        attribution_by_strategy = {}
        raw_attribution_by_strategy = raw_attribution.get("by_strategy")
        if isinstance(raw_attribution_by_strategy, dict):
            for name, item in raw_attribution_by_strategy.items():
                if not isinstance(item, dict):
                    continue
                attribution_by_strategy[str(name)] = {
                    "pnl": _safe_float(item.get("pnl")),
                    "share": _safe_float(item.get("share")),
                    "risk_adjusted_share": _safe_float(item.get("risk_adjusted_share")),
                    "health_score": _safe_float(item.get("health_score")),
                    "lifecycle": str(item.get("lifecycle", "unknown")),
                }

        attribution_by_regime = {}
        raw_attribution_by_regime = raw_attribution.get("by_regime")
        if isinstance(raw_attribution_by_regime, dict):
            for name, item in raw_attribution_by_regime.items():
                if not isinstance(item, dict):
                    continue
                attribution_by_regime[str(name)] = {
                    "pnl": _safe_float(item.get("pnl")),
                    "cycles": int(_safe_float(item.get("cycles"), 0)),
                    "avg_per_cycle": _safe_float(item.get("avg_per_cycle")),
                    "low_confidence": bool(item.get("low_confidence", False)),
                }

        direction_data = raw_attribution.get("by_direction")
        direction_data = direction_data if isinstance(direction_data, dict) else {}
        attribution_by_direction = {
            "long_pnl": _safe_float(direction_data.get("long_pnl")),
            "long_share": _safe_float(direction_data.get("long_share")),
            "long_trades": int(_safe_float(direction_data.get("long_trades"), 0)),
            "short_pnl": _safe_float(direction_data.get("short_pnl")),
            "short_share": _safe_float(direction_data.get("short_share")),
            "short_trades": int(_safe_float(direction_data.get("short_trades"), 0)),
            "long_short_ratio": _safe_float(direction_data.get("long_short_ratio"), 0.0),
        }

        attribution_by_exit_reason = {}
        raw_attribution_by_exit_reason = raw_attribution.get("by_exit_reason")
        if isinstance(raw_attribution_by_exit_reason, dict):
            for reason, item in raw_attribution_by_exit_reason.items():
                if not isinstance(item, dict):
                    continue
                attribution_by_exit_reason[str(reason)] = {
                    "pnl": _safe_float(item.get("pnl")),
                    "count": int(_safe_float(item.get("count"), 0)),
                    "share": _safe_float(item.get("share")),
                }

        pnl_attribution = {
            "available": attribution_available,
            "total_pnl": _safe_float(raw_attribution.get("total_pnl")),
            "attribution_quality": _safe_float(raw_attribution.get("attribution_quality")),
            "dominant_strategy": raw_attribution.get("dominant_strategy"),
            "worst_strategy": raw_attribution.get("worst_strategy"),
            "by_strategy": attribution_by_strategy,
            "by_regime": attribution_by_regime,
            "by_direction": attribution_by_direction,
            "by_exit_reason": attribution_by_exit_reason,
            "timestamp": raw_attribution.get("timestamp", ""),
        }

        return {
            "available": True,
            "projection_available": projection_available,
            "attribution": pnl_attribution,
            "timestamp": last_proj.get("timestamp", ""),
            "horizon_cycles": int(_safe_float(last_proj.get("horizon_cycles"), 0)),
            "regime": str(last_proj.get("regime", "unknown")),
            "correction_factor": _safe_float(last_proj.get("correction_factor"), 1.0),
            "correction_factor_by_regime": correction_by_regime,
            "total_scenarios": total_scenarios,
            "per_strategy": per_strategy,
            "accuracy_history": accuracy_hist,
            "bias_history": bias_history,
            "projection_cycle": int(_safe_float(last_proj.get("projection_cycle"), 0)),
        }

    def get_dashboard_full(self) -> Dict[str, Any]:
        """一站式获取仪表板全量数据（V3.0 增强版）"""
        return {
            "account": self._snapshot_to_dict(self.get_account_snapshot()),
            "equity_curve": self.get_equity_curve(days=30),
            "strategy_comparison": self.get_strategy_comparison(),
            "position_distribution": self.get_position_distribution(),
            "risk_dashboard": self.get_risk_dashboard(),
            # V3.0 新增模块
            "historical_performance": self.get_historical_performance(days=30),
            "capital_efficiency": self.get_capital_efficiency(),
            "capital_utilization": self.get_capital_utilization(),
            "active_alerts": self.get_active_alerts(),
            "funding_trend": self.get_funding_trend(),
            "orderbook_summary": self.get_orderbook_summary(),
            "strategy_heatmap": self.get_strategy_heatmap(days=30),
            "market_overview": self.get_market_overview(),
            # V4.0 新增：Grid 自适应资金利用率
            "grid_adaptive_utilization": self.get_grid_adaptive_utilization(),
            # V4.0 新增：企业级同步健康状态
            "enterprise_sync": self.get_enterprise_sync(),
            # V5.0 新增：AGI 前瞻推算与四维归因面板
            "pnl_projection_panel": self.get_pnl_projection_panel(),
            "timestamp": datetime.now().isoformat(),
        }

    # ═══════════════════════════════════════════════════════════════
    # 内部辅助方法
    # ═══════════════════════════════════════════════════════════════

    def _get_cache(self, key: str) -> Optional[Any]:
        """获取缓存数据"""
        with self._cache_lock:
            if key in self._cache:
                entry = self._cache[key]
                ts, data = entry[0], entry[1]
                ttl = entry[2] if len(entry) > 2 else self.CACHE_TTL_SECONDS
                if time.time() - ts < ttl:
                    return data
        return None

    def _set_cache(self, key: str, data: Any, ttl: float = None) -> None:
        """设置缓存（ttl 优先，缺省使用 CACHE_TTL_SECONDS）"""
        with self._cache_lock:
            self._cache[key] = (time.time(), data, ttl or self.CACHE_TTL_SECONDS)

    def _get_db(self):
        """获取数据库连接"""
        try:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            return conn
        except Exception:
            return None

    def _fetch_okx_account(self) -> Dict[str, Any]:
        """获取OKX账户数据"""
        if self._okx_client:
            try:
                return self._okx_client.fetch_account() or {}
            except Exception:
                pass
        # 回退：从dashboard_api复用函数
        try:
            from dashboard_api import fetch_okx_account
            return fetch_okx_account() or {}
        except Exception:
            return {}

    def _fetch_okx_positions(self) -> List[Dict[str, Any]]:
        """获取OKX持仓数据"""
        if self._okx_client:
            try:
                return self._okx_client.fetch_positions() or []
            except Exception:
                pass
        try:
            from dashboard_api import _okx_circuit_open, fetch_okx_positions
            positions = fetch_okx_positions() or []
            # 仅当 API 调用本身失败（熔断打开）时才回退到 DB；
            # 如果 API 正常返回空列表（交易所确实无持仓），直接返回空列表，
            # 防止已平仓的过期 DB 数据被展示为活跃持仓。
            if positions:
                return positions
            if not _okx_circuit_open():
                return []
        except Exception:
            pass
        # 回退：从 position_history 读取最近一次快照（主交易进程持续写入）
        return self._load_positions_from_db()

    @staticmethod
    def _extract_position_margin(pos: Dict[str, Any]) -> float:
        """提取持仓保证金。

        OKX 全仓（cross）模式持仓的 margin 字段为空字符串，保证金占用体现在
        imr（初始保证金）字段；逐仓（isolated）模式才返回 margin。这里按
        margin → imr → notional/lever 的顺序回退，避免全仓持仓保证金被算成 0，
        导致面板"已用保证金/保证金率/资金利用率"显示与实际不符。
        """
        margin = float(pos.get("margin") or 0)
        if margin > 0:
            return margin
        imr = float(pos.get("imr") or 0)
        if imr > 0:
            return imr
        notional = float(pos.get("notionalUsd") or 0)
        lever = float(pos.get("lever") or 0)
        if notional > 0 and lever > 0:
            return notional / lever
        return 0.0

    def _load_positions_from_db(self) -> List[Dict[str, Any]]:
        """从 position_history 读取最近一次持仓快照（实时 API 失败时的回退数据源）

        映射到 OKX 持仓接口字段（instId/posSide/pos/margin/upl/markPx/avgPx/lever 等），
        供 get_position_distribution / get_account_snapshot 复用。
        position_history 为 append-only，同一批快照的 timestamp 有毫秒级差异，
        故按 (symbol, side) 去重取各自最新记录。

        重要：只取最近 5 分钟的快照，防止已平仓头寸因未写入 quantity=0 行而被展示为活跃持仓。
        """
        try:
            conn = self._get_db()
            if not conn:
                return []
            rows = conn.execute(
                "SELECT symbol, side, quantity, avg_cost, mark_price, "
                "unrealized_pnl, margin, leverage "
                "FROM position_history "
                "WHERE timestamp > datetime('now', 'localtime', '-5 minutes') "
                "ORDER BY timestamp DESC LIMIT 1000"
            ).fetchall()
            conn.close()

            result = []
            seen = set()
            for r in rows:
                key = (r["symbol"], r["side"])
                if key in seen:
                    continue
                seen.add(key)
                side = r["side"] or "long"
                qty = float(r["quantity"] or 0)
                if qty == 0:
                    continue
                mark_px = float(r["mark_price"] or 0)
                margin_val = float(r["margin"] or 0)
                lever_val = int(r["leverage"] or 1)
                # P1 修复：position_history.quantity 是合约张数（OKX pos 原样），
                # 直接 qty × mark_px 会漏乘 ctVal（如 XRP ctVal=100，名义价值差 100 倍）。
                # 改用 OKX 真实保证金 margin × leverage 反推名义价值（margin 已是准确 USDT 口径）。
                notional = margin_val * lever_val if margin_val > 0 else abs(qty) * mark_px
                result.append({
                    "instId": r["symbol"],
                    "posSide": side,
                    "pos": -qty if side == "short" else qty,
                    "margin": margin_val,
                    "upl": float(r["unrealized_pnl"] or 0),
                    "markPx": mark_px,
                    "avgPx": float(r["avg_cost"] or 0),
                    "lever": lever_val,
                    "liqPx": 0.0,
                    "notionalUsd": notional,
                })
            return result
        except Exception as e:
            logger.debug(f"Load positions from DB failed: {e}")
            return []

    def _assess_account_health(self, snapshot: AccountSnapshot) -> str:
        """评估账户健康等级"""
        if snapshot.margin_utilization > 0.85:
            return "danger"
        if snapshot.margin_utilization > 0.70:
            return "caution"
        if snapshot.unrealized_pnl < 0 and abs(snapshot.unrealized_pnl) / max(snapshot.total_equity, 0.01) > 0.15:
            return "caution"
        return "normal"

    def _calc_position_risk(
        self, liq_px: float, mark_px: float, margin: float,
        total_equity: float, upl: float, lever: int,
    ) -> float:
        """计算单仓位风险评分"""
        score = 0.0

        # 清算距离
        if liq_px > 0 and mark_px > 0:
            liq_dist = abs(mark_px - liq_px) / mark_px
            if liq_dist < 0.03:
                score += 40
            elif liq_dist < 0.05:
                score += 30
            elif liq_dist < 0.08:
                score += 20
            elif liq_dist < 0.12:
                score += 10

        # 集中度
        if total_equity > 0:
            conc = margin / total_equity
            if conc > 0.25:
                score += 30
            elif conc > 0.15:
                score += 20
            elif conc > 0.10:
                score += 10

        # 浮亏
        if margin > 0:
            pnl_pct = upl / margin
            if pnl_pct < -0.15:
                score += 20
            elif pnl_pct < -0.08:
                score += 15
            elif pnl_pct < -0.03:
                score += 8
            elif pnl_pct < 0:
                score += 3

        # 杠杆
        if lever >= 15:
            score += 10
        elif lever >= 10:
            score += 6
        elif lever >= 5:
            score += 3

        return score

    def _calc_liquidation_risk(self, pos_dist: Dict[str, Any]) -> float:
        """计算整体清算风险"""
        positions = pos_dist.get("positions", [])
        if not positions:
            return 0.0

        risk_scores = [p.get("risk_score", 0) for p in positions]
        avg_risk = sum(risk_scores) / len(risk_scores)
        max_risk = max(risk_scores)
        return max_risk * 0.6 + avg_risk * 0.4

    def _calc_total_risk_score(self, metrics: RiskMetrics, snapshot: AccountSnapshot) -> float:
        """计算综合风险评分（0-100）"""
        score = 0.0

        # 保证金使用率（权重30%）
        score += min(metrics.margin_utilization * 100, 100) * 0.30

        # 回撤（权重25%）
        score += min(metrics.current_drawdown * 100 * 2, 100) * 0.25

        # 集中度（权重20%）
        score += min(metrics.concentration_ratio * 100 * 2, 100) * 0.20

        # 清算风险（权重15%）
        score += metrics.liquidation_risk * 0.15

        # 浮亏占比（权重10%）
        if snapshot.total_equity > 0:
            loss_pct = abs(min(0, snapshot.unrealized_pnl)) / snapshot.total_equity
            score += min(loss_pct * 100 * 3, 100) * 0.10

        return score

    def _calc_profit_factor(self, conn, strategy_name: str) -> float:
        """计算盈亏比"""
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit, "
                "SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss "
                "FROM trade_records WHERE status = 'closed' AND strategy_name = ?",
                (strategy_name,)
            )
            row = cursor.fetchone()
            if row:
                is_dict_like = isinstance(row, (dict, sqlite3.Row))
                gp = float(row["gross_profit"] or 0) if is_dict_like else float(row[0] or 0)
                gl = float(row["gross_loss"] or 0) if is_dict_like else float(row[1] or 0)
                return round(gp / max(gl, 0.01), 2)
        except Exception:
            pass
        return 0.0

    def _snapshot_to_dict(self, s: AccountSnapshot) -> Dict[str, Any]:
        """快照转字典"""
        return {
            "timestamp": s.timestamp,
            "total_equity": round(s.total_equity, 2),
            "available_balance": round(s.available_balance, 2),
            "used_margin": round(s.used_margin, 2),
            "frozen_balance": round(s.frozen_balance, 2),
            "unrealized_pnl": round(s.unrealized_pnl, 4),
            "realized_pnl_today": round(s.realized_pnl_today, 4),
            "total_pnl_pct": round(s.total_pnl_pct, 4),
            "equity_change_24h": round(s.equity_change_24h, 2),
            "equity_change_24h_pct": round(s.equity_change_24h_pct, 4),
            "margin_utilization": round(s.margin_utilization, 4),
            "margin_ratio": round(s.margin_ratio, 4),
            "maintenance_margin": round(s.maintenance_margin, 2),
            "health_level": s.health_level,
            "is_stale": s.is_stale,
            "data_age_seconds": self._compute_data_age_seconds(s.timestamp),
        }

    def _compute_data_age_seconds(self, timestamp: str) -> Optional[float]:
        """计算快照时间戳距今的秒数（用于前端标记数据陈旧度）。

        返回 None 表示无法解析时间戳；0 表示刚生成。兼容 "T"/空格两种分隔符。
        """
        parsed = self._parse_history_timestamp(timestamp)
        if parsed is None:
            return None
        return max(0.0, (datetime.now() - parsed).total_seconds())

    # ═══════════════════════════════════════════════════════════════
    # 新增辅助方法 v2.1
    # ═══════════════════════════════════════════════════════════════

    def _calc_24h_change(self, snapshot: AccountSnapshot) -> Tuple[float, float]:
        """计算24h权益变化"""
        try:
            now = time.time()
            equity = snapshot.total_equity

            # 尝试从DB获取24h前的权益
            if self._equity_24h_ago_ts == 0 or now - self._equity_24h_ago_ts > 3600:
                conn = self._get_db()
                if conn:
                    cutoff = (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT total_equity FROM account_history "
                        "WHERE timestamp >= ? ORDER BY timestamp ASC LIMIT 1",
                        (cutoff,)
                    )
                    row = cursor.fetchone()
                    if row:
                        is_dict_like = isinstance(row, (dict, sqlite3.Row))
                        self._equity_24h_ago = float(row["total_equity"] or 0) if is_dict_like else float(row[0] or 0)
                        self._equity_24h_ago_ts = now
                    conn.close()

            if self._equity_24h_ago > 0 and equity > 0:
                change = equity - self._equity_24h_ago
                change_pct = change / self._equity_24h_ago
                return round(change, 2), round(change_pct, 4)

            return 0.0, 0.0
        except Exception as e:
            logger.debug(f"_calc_24h_change error: {e}")
            return 0.0, 0.0

    def _get_today_realized_pnl(self) -> float:
        """获取今日已实现盈亏"""
        try:
            conn = self._get_db()
            if conn:
                cursor = conn.cursor()
                today = datetime.now().strftime("%Y-%m-%d")
                cursor.execute(
                    "SELECT SUM(pnl) FROM trade_records WHERE status = 'closed' AND create_time >= ?",
                    (today,)
                )
                row = cursor.fetchone()
                val = float(row[0] or 0) if row else 0
                conn.close()
                return round(float(val or 0), 4)
        except Exception:
            pass
        return 0.0

    def _maybe_persist_snapshot(self, snapshot: AccountSnapshot) -> None:
        """定期持久化账户快照到DB"""
        try:
            now = time.time()
            if now - self._last_snapshot_persist < self.SNAPSHOT_PERSIST_INTERVAL:
                return
            self._last_snapshot_persist = now

            conn = self._get_db()
            if conn:
                # 统一时间戳格式为空格分隔（对齐主进程 SQLAlchemy DateTime 存储），
                # 避免 T/空格 两种格式混存导致 ORDER BY timestamp 字符串排序错乱
                ts = (snapshot.timestamp or "").replace("T", " ")
                if not ts:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                conn.execute(
                    "INSERT INTO account_history (id, timestamp, total_equity, available_balance, "
                    "used_margin, unrealized_pnl, margin_rate) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (ts, ts, snapshot.total_equity, snapshot.available_balance,
                     snapshot.used_margin, snapshot.unrealized_pnl, snapshot.margin_ratio)
                )
                conn.commit()
                conn.close()
                logger.debug(f"Snapshot persisted: equity={snapshot.total_equity:.2f}")
        except Exception as e:
            logger.debug(f"Snapshot persist skipped: {e}")

    def _load_latest_snapshot_from_db(self) -> Optional[AccountSnapshot]:
        """从 account_history 读取最近一次快照（实时 API 失败时的回退数据源）

        注意：account_history 表由主交易进程（sqlite_storage.AccountHistory）维护，
        实际列为 id/total_equity/available_balance/used_margin/unrealized_pnl/margin_rate/timestamp，
        与 dashboard_engine 新建表定义不同，故此处只查询确定存在的列。
        """
        try:
            conn = self._get_db()
            if not conn:
                return None
            row = conn.execute(
                "SELECT timestamp, total_equity, available_balance, used_margin, "
                "unrealized_pnl FROM account_history ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if not row:
                return None
            s = AccountSnapshot(timestamp=str(row["timestamp"]))
            s.total_equity = float(row["total_equity"] or 0)
            s.available_balance = float(row["available_balance"] or 0)
            s.used_margin = float(row["used_margin"] or 0)
            s.unrealized_pnl = float(row["unrealized_pnl"] or 0)
            s.health_level = "normal"
            s.is_stale = True
            if s.total_equity > 0:
                s.margin_utilization = s.used_margin / s.total_equity
                s.margin_ratio = (s.total_equity - s.used_margin) / s.total_equity
                s.total_pnl_pct = s.unrealized_pnl / (s.total_equity - s.unrealized_pnl + 0.01)
            return s
        except Exception as e:
            logger.debug(f"Load latest snapshot from DB failed: {e}")
            return None

    def _infer_position_strategy(self, symbol: str, symbol_base: str) -> Tuple[str, str]:
        """推断持仓归属策略

        策略推断优先级：
        1. 从 strategy_manager 查询（如果注入）
        2. 从 _strategy_symbol_map 匹配
        3. 从 symbol 名称模式匹配
        4. 兜底为 "unknown"
        """
        # 优先级1: 从 strategy_manager 查询
        if self._strategy_manager:
            try:
                if hasattr(self._strategy_manager, 'get_position_strategy'):
                    strat = self._strategy_manager.get_position_strategy(symbol)
                    if strat:
                        return strat, strat
            except Exception:
                pass

        # 优先级2: 从 _strategy_symbol_map 匹配
        for strategy_name, symbols in self._strategy_symbol_map.items():
            if symbol_base in symbols or symbol in symbols:
                display = strategy_name.replace("_strategy", "").replace("_", " ").title()
                return strategy_name, display

        # 优先级3: 名称模式匹配
        symbol_lower = symbol_base.lower()
        if "grid" in symbol_lower:
            return "grid", "Grid"
        if "trend" in symbol_lower:
            return "trend", "Trend"

        # 优先级4: 从 trade_records 最近交易推断（dashboard 独立进程兜底）
        inferred = self._infer_strategy_from_trades(symbol)
        if inferred and inferred != "unknown":
            display = inferred.replace("_strategy", "").replace("_", " ").title()
            return inferred, display

        # 优先级5: 兜底
        return "unknown", "Unknown"

    def _infer_strategy_from_trades(self, symbol: str) -> Optional[str]:
        """从 trade_records 最近交易推断持仓归属策略。

        dashboard 是独立进程，无法访问主进程的 strategy_manager / state_manager，
        导致 _strategy_symbol_map 恒为空、持仓被判为 unknown。这里改为从真实
        交易记录反查：取该 symbol 最近一次非 sync 交易的 strategy_name。
        """
        conn = self._get_db()
        if not conn:
            return None
        try:
            row = conn.execute(
                "SELECT strategy_name FROM trade_records "
                "WHERE symbol = ? AND strategy_name IS NOT NULL "
                "AND LOWER(strategy_name) != 'sync' "
                "ORDER BY COALESCE(close_time, create_time) DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if row and row["strategy_name"]:
                return str(row["strategy_name"]).strip().lower()
        except Exception as e:
            logger.debug(f"Could not infer strategy from trades for {symbol}: {e}")
        finally:
            conn.close()
        return None

    def _get_risk_dashboard_enhanced(self, force_refresh: bool = False) -> Dict[str, Any]:
        """增强版风险水位 — 包含新指标"""
        result = self.get_risk_dashboard(force_refresh=force_refresh)

        try:
            # 补充资金费率敞口
            positions = self._fetch_okx_positions()
            funding_exposure = 0.0
            for pos in positions:
                funding_rate = float(pos.get("fundingRate", 0) or 0)
                notional = float(pos.get("notionalUsd", 0) or 0)
                if funding_rate != 0:
                    funding_exposure += abs(funding_rate * notional)
            result["risk_metrics"]["funding_rate_exposure"] = round(funding_exposure, 4)

            # 补充总持仓/总权益比率
            total_notional = sum(float(p.get("notionalUsd", 0) or 0) for p in positions)
            equity = result.get("account_health", {}).get("total_equity", 0)
            if equity > 0:
                result["risk_metrics"]["open_interest_ratio"] = round(total_notional / equity, 4)
        except Exception:
            pass

        return result

    # ═══════════════════════════════════════════════════════════════
    # 新增 V3.0 模块 — 历史表现 / 资金效率 / 告警 / 费率 / 订单簿 / 热力图 / 市场概览
    # ═══════════════════════════════════════════════════════════════

    def get_historical_performance(self, days: int = 30, force_refresh: bool = False) -> Dict[str, Any]:
        """获取多周期历史表现统计"""
        cache_key = f"historical_perf_{days}"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            perf = HistoricalPerformance(period=f"{days}d")

            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

            # 权益数据（按日聚合 + 分钟级出入金检测，统一口径）
            daily_data = self._fetch_daily_series(days)
            daily_equities = daily_data["equities"]
            flow_days = daily_data["flow_days"]
            equities = [eq for _day, eq in daily_equities]

            if equities:
                perf.start_equity = equities[0]
                perf.end_equity = equities[-1]

                # 交易性日收益：出入金日的日收益不计入（避免把入金/提现误判为交易盈亏）
                trading_returns = []
                for i in range(1, len(daily_equities)):
                    day, eq = daily_equities[i]
                    prev_eq = daily_equities[i - 1][1]
                    if day in flow_days:
                        continue
                    if prev_eq > 0:
                        trading_returns.append((eq - prev_eq) / prev_eq)

                # 资本调整后总收益 = 交易性日收益的复利（剔除出入金）
                adjusted_total_pct = 1.0
                for r in trading_returns:
                    adjusted_total_pct *= (1 + r)
                adjusted_total_pct -= 1.0
                perf.total_return_pct = adjusted_total_pct
                perf.total_return = perf.start_equity * adjusted_total_pct

                # 最大回撤 / 回撤持续期（剔除出入金，按日聚合后以天计）
                perf.max_drawdown = self._capital_adjusted_max_drawdown(daily_equities, flow_days)
                perf.max_drawdown_duration = self._capital_adjusted_drawdown_duration(daily_equities, flow_days)

                if trading_returns:
                    import statistics
                    avg_dr = statistics.mean(trading_returns)
                    vol_daily = statistics.stdev(trading_returns) if len(trading_returns) > 1 else 0
                    perf.best_day = max(trading_returns)
                    perf.worst_day = min(trading_returns)
                    perf.avg_daily_return = avg_dr
                    perf.volatility_annual = vol_daily * math.sqrt(365)

                    risk_free = 0.03  # 3%年化无风险利率

                    # Sharpe
                    if perf.volatility_annual > 0:
                        perf.sharpe_ratio = (perf.total_return_pct - risk_free) / perf.volatility_annual

                    # Sortino
                    downside_returns = [r for r in trading_returns if r < 0]
                    if len(downside_returns) > 1:
                        downside_vol = statistics.stdev(downside_returns) * math.sqrt(365)
                        if downside_vol > 0:
                            perf.sortino_ratio = (perf.total_return_pct - risk_free) / downside_vol

                    # Calmar
                    if perf.max_drawdown > 0:
                        perf.calmar_ratio = perf.total_return_pct / perf.max_drawdown

                    # Recovery factor
                    if perf.max_drawdown > 0 and perf.total_return > 0:
                        perf.recovery_factor = perf.total_return / (perf.max_drawdown * perf.start_equity)

            # 交易统计（容错：trades 表可能不存在）
            conn = self._get_db()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT COUNT(*) as total, "
                        "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
                        "SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses, "
                        "AVG(CASE WHEN pnl > 0 THEN pnl ELSE NULL END) as avg_win, "
                        "AVG(CASE WHEN pnl < 0 THEN pnl ELSE NULL END) as avg_loss "
                        "FROM trade_records WHERE status = 'closed' AND create_time >= ?",
                        (cutoff,)
                    )
                    row = cursor.fetchone()
                    if row:
                        is_dict_like = isinstance(row, (dict, sqlite3.Row))
                        perf.total_trades = int(row["total"] or 0) if is_dict_like else int(row[0] or 0)
                        perf.winning_trades = int(row["wins"] or 0) if is_dict_like else int(row[1] or 0)
                        perf.losing_trades = int(row["losses"] or 0) if is_dict_like else int(row[2] or 0)
                        perf.win_rate = perf.winning_trades / max(perf.total_trades, 1)
                        perf.avg_win = float(row["avg_win"] or 0) if is_dict_like else float(row[3] or 0)
                        perf.avg_loss = abs(float(row["avg_loss"] or 0)) if is_dict_like else abs(float(row[4] or 0))
                        if perf.avg_loss > 0:
                            perf.profit_factor = perf.avg_win / perf.avg_loss
                except (sqlite3.OperationalError, Exception):
                    pass  # trades 表不存在或查询失败，跳过交易统计
                finally:
                    conn.close()

            result = {
                "period": perf.period,
                "start_equity": round(perf.start_equity, 2),
                "end_equity": round(perf.end_equity, 2),
                "total_return": round(perf.total_return, 2),
                "total_return_pct": round(perf.total_return_pct, 4),
                "max_drawdown": round(perf.max_drawdown, 4),
                "max_drawdown_duration": perf.max_drawdown_duration,
                "sharpe_ratio": round(perf.sharpe_ratio, 3),
                "calmar_ratio": round(perf.calmar_ratio, 3),
                "sortino_ratio": round(perf.sortino_ratio, 3),
                "win_rate": round(perf.win_rate, 4),
                "avg_win": round(perf.avg_win, 4),
                "avg_loss": round(perf.avg_loss, 4),
                "profit_factor": round(perf.profit_factor, 2),
                "total_trades": perf.total_trades,
                "winning_trades": perf.winning_trades,
                "losing_trades": perf.losing_trades,
                "best_day": round(perf.best_day, 4),
                "worst_day": round(perf.worst_day, 4),
                "avg_daily_return": round(perf.avg_daily_return, 4),
                "volatility_annual": round(perf.volatility_annual, 4),
                "recovery_factor": round(perf.recovery_factor, 4),
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.EQUITY_CACHE_TTL)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: historical performance error: {e}")
            return {"period": f"{days}d", "timestamp": datetime.now().isoformat()}

    def get_capital_efficiency(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取资金效率指标（企业级增强版）"""
        cache_key = "capital_efficiency"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            metrics = CapitalEfficiencyMetrics()
            snapshot = self.get_account_snapshot()

            metrics.total_capital = snapshot.total_equity
            metrics.deployed_capital = snapshot.used_margin
            metrics.idle_capital = snapshot.available_balance
            if metrics.total_capital > 0:
                metrics.efficiency_ratio = metrics.deployed_capital / metrics.total_capital

            # 从 capital_manager 获取资金池状态
            if self._capital_manager:
                try:
                    cp = self._capital_manager
                    if hasattr(cp, 'capital_pool'):
                        pool = cp.capital_pool
                        total_pool = pool.total_capital if hasattr(pool, 'total_capital') else metrics.total_capital
                        base = pool.base_pool if hasattr(pool, 'base_pool') else 0
                        add = pool.add_pool if hasattr(pool, 'add_pool') else 0
                        risk = pool.risk_pool if hasattr(pool, 'risk_pool') else 0
                        if total_pool > 0:
                            metrics.base_pool_usage = (total_pool * 0.6 - base) / (total_pool * 0.6) if base >= 0 else 0
                            metrics.add_pool_usage = (total_pool * 0.25 - add) / (total_pool * 0.25) if add >= 0 else 0
                            metrics.risk_pool_usage = (total_pool * 0.15 - risk) / (total_pool * 0.15) if risk >= 0 else 0
                    if hasattr(cp, '_last_rebalance_date'):
                        metrics.last_rebalance = cp._last_rebalance or ""
                except Exception:
                    pass

            # 闲置资金机会成本（企业级：日化 0.05%）
            if metrics.idle_capital > 0:
                metrics.idle_capital_cost = metrics.idle_capital * 0.0005  # 日化0.05%
                metrics.optimal_deployment = metrics.idle_capital * 0.7  # 建议部署70%闲置资金
                metrics.rebalance_needed = metrics.idle_capital > metrics.total_capital * 0.3

            # 企业级：每策略资金利用率
            strategy_utilization = {}
            try:
                plan_data = {}
                if self._agi_orchestrator is not None:
                    snapshot_getter = getattr(
                        self._agi_orchestrator, "get_dashboard_snapshot", None
                    )
                    if callable(snapshot_getter):
                        agi_snapshot = snapshot_getter()
                        if isinstance(agi_snapshot, dict):
                            plan_data = agi_snapshot.get("allocation_plan") or {}

                if not plan_data:
                    from risk.dynamic_allocator import get_dynamic_allocator
                    allocator = get_dynamic_allocator()
                    get_last_plan = getattr(allocator, "get_last_plan", None)
                    if callable(get_last_plan):
                        plan_data = get_last_plan() or {}
                if isinstance(plan_data, dict):
                    utilization = plan_data.get("strategy_utilization")
                    if isinstance(utilization, dict):
                        strategy_utilization = dict(utilization)
            except Exception as e:
                logger.warning(f"DashboardEngine: strategy utilization unavailable: {e}")

            result = {
                "total_capital": round(metrics.total_capital, 2),
                "deployed_capital": round(metrics.deployed_capital, 2),
                "idle_capital": round(metrics.idle_capital, 2),
                "efficiency_ratio": round(metrics.efficiency_ratio, 4),
                "base_pool_usage": round(metrics.base_pool_usage, 4),
                "add_pool_usage": round(metrics.add_pool_usage, 4),
                "risk_pool_usage": round(metrics.risk_pool_usage, 4),
                "idle_capital_cost": round(metrics.idle_capital_cost, 4),
                "optimal_deployment": round(metrics.optimal_deployment, 2),
                "rebalance_needed": metrics.rebalance_needed,
                "last_rebalance": metrics.last_rebalance,
                "strategy_utilization": strategy_utilization,
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS * 2)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: capital efficiency error: {e}")
            return {"timestamp": datetime.now().isoformat()}

    def get_capital_utilization(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取企业级资金利用率状态（含 force_rebalance / emergency_blocked 等引擎状态）。

        数据源为 AdaptiveController 持久化的 data/capital_utilization_state.json，
        由后台 _capital_utilization_loop 周期性刷新，前端据此展示状态卡与告警横幅。
        """
        cache_key = "capital_utilization"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            state: Dict[str, Any] = {}
            state_path = os.path.join("data", "capital_utilization_state.json")
            if os.path.exists(state_path):
                try:
                    with open(state_path, "r", encoding="utf-8") as f:
                        state = json.load(f) or {}
                except Exception:
                    state = {}

            result = {
                "utilization_rate": float(state.get("utilization_rate", 0) or 0),
                "used_margin": float(state.get("total_used", 0) or 0),
                "available_balance": float(state.get("total_available", 0) or 0),
                "total_equity": float(state.get("total_equity", 0) or 0),
                "target_utilization": float(state.get("target_utilization", 0.85) or 0.85),
                "status": state.get("status", "normal"),
                "position_boost": float(state.get("position_boost", 1.0) or 1.0),
                "idle_ratio": max(0.0, 1.0 - float(state.get("utilization_rate", 0) or 0)),
                "recommended_action": state.get("recommended_action", "none"),
                "capital_efficiency": float(state.get("capital_efficiency", 0.0) or 0),
                "utilization_tier": state.get("utilization_tier"),
                "utilization_trend": float(state.get("utilization_trend", 0.0) or 0),
                "equity_mode": state.get("equity_mode", "normal"),
                "timestamp": state.get("timestamp", datetime.now().isoformat()),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS * 2)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: capital utilization error: {e}")
            return {"timestamp": datetime.now().isoformat(), "status": "unknown", "error": str(e)}

    def get_active_alerts(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取活跃告警列表"""
        cache_key = "active_alerts"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            alerts: List[ActiveAlert] = []
            snapshot = self.get_account_snapshot()
            risk = self.get_risk_dashboard()
            pos_dist = self.get_position_distribution()

            # 1. 保证金告警
            if snapshot.margin_utilization > 0.85:
                alerts.append(ActiveAlert(
                    id="margin_critical", type="margin", severity="critical",
                    message=f"保证金使用率 {snapshot.margin_utilization*100:.1f}% 超过危急线85%",
                    source="risk_gate", value=snapshot.margin_utilization, threshold=0.85,
                    timestamp=datetime.now().isoformat(),
                ))
            elif snapshot.margin_utilization > 0.70:
                alerts.append(ActiveAlert(
                    id="margin_warning", type="margin", severity="warning",
                    message=f"保证金使用率 {snapshot.margin_utilization*100:.1f}% 超过预警线70%",
                    source="risk_gate", value=snapshot.margin_utilization, threshold=0.70,
                    timestamp=datetime.now().isoformat(),
                ))

            # 2. 回撤告警
            current_dd = risk.get("risk_metrics", {}).get("current_drawdown", 0)
            if current_dd > 0.25:
                alerts.append(ActiveAlert(
                    id="drawdown_critical", type="risk", severity="critical",
                    message=f"当前回撤 {current_dd*100:.1f}% 超过危急线25%",
                    source="risk_gate", value=current_dd, threshold=0.25,
                    timestamp=datetime.now().isoformat(),
                ))
            elif current_dd > 0.15:
                alerts.append(ActiveAlert(
                    id="drawdown_warning", type="risk", severity="warning",
                    message=f"当前回撤 {current_dd*100:.1f}% 超过预警线15%",
                    source="risk_gate", value=current_dd, threshold=0.15,
                    timestamp=datetime.now().isoformat(),
                ))

            # 3. 清算风险告警
            liq_risk = risk.get("risk_metrics", {}).get("liquidation_risk", 0)
            if liq_risk > 60:
                alerts.append(ActiveAlert(
                    id="liquidation_critical", type="liquidation", severity="critical",
                    message=f"清算风险评分 {liq_risk:.0f} 超过危急线60",
                    source="risk_gate", value=liq_risk, threshold=60,
                    timestamp=datetime.now().isoformat(),
                ))

            # 4. 危险持仓告警
            danger_count = pos_dist.get("summary", {}).get("danger_positions", 0)
            if danger_count > 0:
                alerts.append(ActiveAlert(
                    id="danger_positions", type="risk", severity="warning",
                    message=f"存在 {danger_count} 个危险持仓，清算风险较高",
                    source="position_monitor", value=danger_count, threshold=0,
                    timestamp=datetime.now().isoformat(),
                ))

            # 5. 磨损预算告警
            if self._attrition_analyzer:
                try:
                    budget_status = self._attrition_analyzer.get_budget_status()
                    for strategy_name, status in budget_status.items():
                        if isinstance(status, dict) and status.get("usage_pct", 0) > 0.7:
                            alerts.append(ActiveAlert(
                                id=f"attrition_{strategy_name}", type="attrition",
                                severity="warning" if status.get("usage_pct", 0) < 0.95 else "critical",
                                message=f"{strategy_name} 磨损预算使用率 {status.get('usage_pct', 0)*100:.0f}%",
                                source="attrition_analyzer", value=status.get("usage_pct", 0), threshold=0.7,
                                timestamp=datetime.now().isoformat(),
                            ))
                except Exception:
                    pass

            # 6. 资金费率告警
            positions = self._fetch_okx_positions()
            for pos in positions:
                funding_rate = float(pos.get("fundingRate", 0) or 0)
                if abs(funding_rate) > 0.001:  # 0.1%以上
                    notional = float(pos.get("notionalUsd", 0) or 0)
                    daily_cost = abs(funding_rate * notional * 3)  # 每天3次结算
                    if daily_cost > 1.0:
                        alerts.append(ActiveAlert(
                            id=f"funding_{pos.get('instId', '')}", type="risk", severity="warning",
                            message=f"{pos.get('instId', '')} 资金费率 {funding_rate*100:.3f}%，日费用约${daily_cost:.2f}",
                            source="funding_monitor", value=funding_rate, threshold=0.001,
                            timestamp=datetime.now().isoformat(),
                        ))

            # 7. 系统健康告警
            # 注意：dashboard 作为独立进程运行，_okx_client 恒为 None，
            # 实时数据通过 dashboard_api 的直接 OKX HTTP 调用获取，因此不再告警"未连接"。

            result = {
                "alerts": [
                    {
                        "id": a.id, "type": a.type, "severity": a.severity,
                        "message": a.message, "source": a.source,
                        "timestamp": a.timestamp, "value": a.value, "threshold": a.threshold,
                    }
                    for a in sorted(alerts, key=lambda x: {"critical": 0, "warning": 1, "info": 2}[x.severity])
                ],
                "summary": {
                    "total": len(alerts),
                    "critical": sum(1 for a in alerts if a.severity == "critical"),
                    "warning": sum(1 for a in alerts if a.severity == "warning"),
                    "info": sum(1 for a in alerts if a.severity == "info"),
                },
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: active alerts error: {e}")
            return {"alerts": [], "summary": {}, "timestamp": datetime.now().isoformat()}

    def get_funding_trend(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取资金费率趋势分析"""
        cache_key = "funding_trend"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            items: List[FundingTrendItem] = []
            positions = self._fetch_okx_positions()

            # 并发拉取资金费率：逐币种串行请求在冷缓存时累计耗时过长，
            # 改为线程池并发后统一回填，避免 /api/dashboard/full 超时。
            funding_cache: Dict[str, Optional[float]] = {}
            symbols_need_rate = set()
            for pos in positions:
                if float(pos.get("fundingRate", 0) or 0) == 0:
                    symbols_need_rate.add(pos.get("instId", ""))
            if symbols_need_rate:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                syms = sorted(symbols_need_rate)
                with ThreadPoolExecutor(max_workers=min(8, len(syms))) as ex:
                    futs = {ex.submit(self._fetch_current_funding_rate, s): s for s in syms}
                    for fut in as_completed(futs):
                        s = futs[fut]
                        try:
                            funding_cache[s] = fut.result()
                        except Exception:
                            funding_cache[s] = None

            for pos in positions:
                symbol = pos.get("instId", "")
                pos_side = pos.get("posSide", "long")
                notional = float(pos.get("notionalUsd", 0) or 0)
                funding_rate = float(pos.get("fundingRate", 0) or 0)
                # OKX 持仓接口不返回 fundingRate 字段，需从资金费率接口单独获取
                if funding_rate == 0:
                    fetched = funding_cache.get(symbol)
                    if fetched:
                        funding_rate = fetched

                if notional == 0:
                    continue

                symbol_base = symbol.replace("-USDT-SWAP", "").replace("-USDC-SWAP", "")

                # 资金费用计算
                daily_cost = abs(funding_rate * notional * 3)
                weekly_cost = daily_cost * 7

                # 趋势判断（基于费率符号和大小）
                if funding_rate > 0.0005:
                    trend = "rising"
                elif funding_rate < -0.0005:
                    trend = "falling"
                elif abs(funding_rate) < 0.0001:
                    trend = "stable"
                else:
                    trend = "stable"

                items.append(FundingTrendItem(
                    symbol=symbol_base,
                    current_rate=funding_rate,
                    avg_rate_24h=funding_rate,
                    predicted_rate=funding_rate * 1.1 if abs(funding_rate) > 0.0001 else funding_rate,
                    position_side=pos_side,
                    daily_cost=round(daily_cost, 4),
                    weekly_cost=round(weekly_cost, 4),
                    trend=trend,
                ))

            result = {
                "items": [
                    {
                        "symbol": i.symbol,
                        "current_rate": i.current_rate,
                        "avg_rate_24h": i.avg_rate_24h,
                        "predicted_rate": i.predicted_rate,
                        "position_side": i.position_side,
                        "daily_cost": i.daily_cost,
                        "weekly_cost": i.weekly_cost,
                        "trend": i.trend,
                    }
                    for i in items
                ],
                "summary": {
                    "total_daily_cost": round(sum(i.daily_cost for i in items), 4),
                    "total_weekly_cost": round(sum(i.weekly_cost for i in items), 4),
                    "most_expensive": max(items, key=lambda x: x.daily_cost).symbol if items else "",
                    "funding_count": len(items),
                },
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, 60.0)  # 资金费率属慢变行情，延长缓存避免 SSE 高频冷请求
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: funding trend error: {e}")
            return {"items": [], "summary": {}, "timestamp": datetime.now().isoformat()}

    def get_orderbook_summary(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取订单簿摘要 — 深度、流动性、买卖比"""
        cache_key = "orderbook_summary"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            summaries: List[OrderbookSummary] = []
            positions = self._fetch_okx_positions()

            symbols_to_check = set()
            for pos in positions:
                qty = float(pos.get("pos", 0) or 0)
                if qty != 0:
                    symbols_to_check.add(pos.get("instId", ""))

            # 并发拉取订单簿：逐币种串行请求在冷缓存时累计耗时可达数十秒，
            # 导致 /api/dashboard/full 超时、SSE 连接被重置。改为线程池并发。
            symbols = sorted(symbols_to_check)
            orderbooks: Dict[str, Dict[str, Any]] = {}
            if symbols:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as ex:
                    futures = {ex.submit(self._fetch_orderbook, s): s for s in symbols}
                    for fut in as_completed(futures):
                        s = futures[fut]
                        try:
                            orderbooks[s] = fut.result() or {}
                        except Exception:
                            orderbooks[s] = {}

            for symbol in symbols:
                ob = orderbooks.get(symbol) or {}
                if not ob:
                    continue

                symbol_base = symbol.replace("-USDT-SWAP", "").replace("-USDC-SWAP", "")
                bids = ob.get("bids", [])
                asks = ob.get("asks", [])
                if not bids or not asks:
                    continue

                best_bid = float(bids[0][0]) if bids else 0
                best_ask = float(asks[0][0]) if asks else 0
                spread = best_ask - best_bid
                spread_pct = spread / best_ask if best_ask > 0 else 0

                # 1%深度
                bid_depth = sum(float(b[1]) * float(b[0]) for b in bids if float(b[0]) >= best_bid * 0.99)
                ask_depth = sum(float(a[1]) * float(a[0]) for a in asks if float(a[0]) <= best_ask * 1.01)

                bid_ask_ratio = bid_depth / ask_depth if ask_depth > 0 else 0
                imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth) if (bid_depth + ask_depth) > 0 else 0

                # 流动性评分
                depth_score = min((bid_depth + ask_depth) / 100000, 1.0) * 50
                spread_score = max(0, (1 - spread_pct * 100)) * 50
                liquidity_score = depth_score + spread_score

                summaries.append(OrderbookSummary(
                    symbol=symbol_base,
                    best_bid=best_bid, best_ask=best_ask,
                    spread=spread, spread_pct=spread_pct,
                    bid_depth_1pct=bid_depth, ask_depth_1pct=ask_depth,
                    bid_ask_ratio=bid_ask_ratio,
                    orderbook_imbalance=imbalance,
                    liquidity_score=round(liquidity_score, 1),
                ))

            result = {
                "items": [
                    {
                        "symbol": s.symbol, "best_bid": s.best_bid, "best_ask": s.best_ask,
                        "spread": round(s.spread, 6), "spread_pct": round(s.spread_pct, 6),
                        "bid_depth_1pct": round(s.bid_depth_1pct, 2),
                        "ask_depth_1pct": round(s.ask_depth_1pct, 2),
                        "bid_ask_ratio": round(s.bid_ask_ratio, 4),
                        "orderbook_imbalance": round(s.orderbook_imbalance, 4),
                        "liquidity_score": s.liquidity_score,
                    }
                    for s in summaries
                ],
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, 60.0)  # 订单簿属慢变行情，延长缓存避免 SSE 高频冷请求
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: orderbook summary error: {e}")
            return {"items": [], "timestamp": datetime.now().isoformat()}

    def _fetch_orderbook(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
        """获取订单簿数据，返回 {'bids': [[price,size]...], 'asks': [[price,size]...]}"""
        if self._okx_client:
            try:
                if hasattr(self._okx_client, 'fetch_orderbook'):
                    ob = self._okx_client.fetch_orderbook(symbol, depth)
                    if ob:
                        return ob
            except Exception:
                pass
        try:
            from dashboard_api import _make_okx_request
            raw = _make_okx_request("GET", f"/api/v5/market/books?instId={symbol}&sz={depth}") or {}
            data = raw.get("data") or []
            if isinstance(data, list) and data:
                return data[0]
            return {}
        except Exception:
            return {}

    def _fetch_current_funding_rate(self, symbol: str) -> Optional[float]:
        """获取指定合约的当前资金费率"""
        try:
            from dashboard_api import _make_okx_request
            raw = _make_okx_request("GET", f"/api/v5/public/funding-rate?instId={symbol}") or {}
            data = raw.get("data") or []
            if isinstance(data, list) and data:
                return float(data[0].get("fundingRate", 0) or 0)
        except Exception:
            pass
        return None

    def get_strategy_heatmap(self, days: int = 30, force_refresh: bool = False) -> Dict[str, Any]:
        """获取策略热力图数据"""
        cache_key = f"strategy_heatmap_{days}"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            cells: List[StrategyHeatmapCell] = []
            conn = self._get_db()
            if not conn:
                self._set_cache(cache_key, {}, self.CACHE_TTL_SECONDS * 2)
                return {}

            cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")

            # 按策略+币种聚合（容错：trades 表可能不存在）
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT strategy_name, "
                    "SUM(pnl) as total_pnl, "
                    "COUNT(*) as trade_count, "
                    "SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins, "
                    "SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as gross_profit, "
                    "SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as gross_loss "
                    "FROM trade_records WHERE status = 'closed' AND create_time >= ? "
                    "GROUP BY strategy_name",
                    (cutoff,)
                )

                for row in cursor.fetchall():
                    is_dict_like = isinstance(row, (dict, sqlite3.Row))
                    strategy_name = row["strategy_name"] if is_dict_like else row[0]
                    total_pnl = float(row["total_pnl"] or 0) if is_dict_like else float(row[1] or 0)
                    trade_count = int(row["trade_count"] or 0) if is_dict_like else int(row[2] or 0)
                    wins = int(row["wins"] or 0) if is_dict_like else int(row[3] or 0)
                    gp = float(row["gross_profit"] or 0) if is_dict_like else float(row[4] or 0)
                    gl = float(row["gross_loss"] or 0) if is_dict_like else float(row[5] or 0)

                    win_rate = wins / max(trade_count, 1)
                    avg_pnl = total_pnl / max(trade_count, 1)
                    profit_factor = gp / max(gl, 0.01)

                    status = "running"
                    if trade_count == 0:
                        status = "idle"
                    elif total_pnl > 0:
                        status = "profitable"
                    elif total_pnl < 0:
                        status = "losing"

                    cells.append(StrategyHeatmapCell(
                        strategy=strategy_name,
                        symbol="ALL",
                        period=f"{days}d",
                        total_pnl=round(total_pnl, 4),
                        win_rate=round(win_rate, 4),
                        trade_count=trade_count,
                        avg_pnl_per_trade=round(avg_pnl, 4),
                        profit_factor=round(profit_factor, 2),
                        status=status,
                    ))
            except (sqlite3.OperationalError, Exception):
                pass  # trades 表不存在，跳过热力图数据

            conn.close()

            result = {
                "cells": [
                    {
                        "strategy": c.strategy, "symbol": c.symbol, "period": c.period,
                        "total_pnl": c.total_pnl, "win_rate": c.win_rate,
                        "trade_count": c.trade_count, "avg_pnl_per_trade": c.avg_pnl_per_trade,
                        "profit_factor": c.profit_factor, "status": c.status,
                    }
                    for c in sorted(cells, key=lambda x: x.total_pnl, reverse=True)
                ],
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.EQUITY_CACHE_TTL)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: strategy heatmap error: {e}")
            return {"cells": [], "timestamp": datetime.now().isoformat()}

    def get_market_overview(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取市场概览 — BTC/ETH价格、市场情绪、波动率"""
        cache_key = "market_overview"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            overview = MarketOverview()

            # 获取BTC/ETH价格
            if self._okx_client:
                try:
                    if hasattr(self._okx_client, 'fetch_ticker'):
                        btc_ticker = self._okx_client.fetch_ticker("BTC-USDT")
                        eth_ticker = self._okx_client.fetch_ticker("ETH-USDT")
                        if btc_ticker:
                            overview.btc_price = float(btc_ticker.get("last", 0) or 0)
                            overview.btc_change_24h = float(btc_ticker.get("change24h", 0) or 0) / 100 if btc_ticker.get("change24h") else 0
                        if eth_ticker:
                            overview.eth_price = float(eth_ticker.get("last", 0) or 0)
                            overview.eth_change_24h = float(eth_ticker.get("change24h", 0) or 0) / 100 if eth_ticker.get("change24h") else 0
                except Exception:
                    pass

            # 回退：从dashboard_api获取
            if overview.btc_price == 0:
                try:
                    from dashboard_api import _make_okx_request
                    btc_data = _make_okx_request("GET", "/api/v5/market/ticker?instId=BTC-USDT")
                    eth_data = _make_okx_request("GET", "/api/v5/market/ticker?instId=ETH-USDT")
                    if btc_data and btc_data.get("data"):
                        d = btc_data["data"][0]
                        overview.btc_price = float(d.get("last", 0) or 0)
                        open24h = float(d.get("open24h", 0) or 0)
                        if overview.btc_price > 0 and open24h > 0:
                            overview.btc_change_24h = (overview.btc_price - open24h) / open24h
                    if eth_data and eth_data.get("data"):
                        d = eth_data["data"][0]
                        overview.eth_price = float(d.get("last", 0) or 0)
                        open24h = float(d.get("open24h", 0) or 0)
                        if overview.eth_price > 0 and open24h > 0:
                            overview.eth_change_24h = (overview.eth_price - open24h) / open24h
                except Exception:
                    pass

            # 市场状态判断
            if overview.btc_change_24h > 0.05:
                overview.market_regime = "bull"
            elif overview.btc_change_24h < -0.05:
                overview.market_regime = "bear"
            elif abs(overview.btc_change_24h) > 0.02:
                overview.market_regime = "volatile"
            else:
                overview.market_regime = "neutral"

            # 波动率指数（简化：基于BTC 24h振幅）
            try:
                from dashboard_api import _make_okx_request
                btc_candle = _make_okx_request("GET", "/api/v5/market/candles?instId=BTC-USDT&bar=1D&limit=1")
                if btc_candle and btc_candle.get("data"):
                    d = btc_candle["data"][0]
                    high = float(d[2])
                    low = float(d[3])
                    if high > 0:
                        overview.vix_like = (high - low) / high
            except Exception:
                pass

            result = {
                "btc_price": round(overview.btc_price, 2),
                "btc_change_24h": round(overview.btc_change_24h, 4),
                "eth_price": round(overview.eth_price, 2),
                "eth_change_24h": round(overview.eth_change_24h, 4),
                "market_regime": overview.market_regime,
                "vix_like": round(overview.vix_like, 4),
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS * 3)
            return result

        except Exception as e:
            logger.error(f"DashboardEngine: market overview error: {e}")
            return {"timestamp": datetime.now().isoformat()}

    def get_grid_adaptive_utilization(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取 Grid 自适应资金利用率数据（企业级 v4.0 新增）"""
        cache_key = "grid_adaptive_utilization"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            from core.grid_adaptive_utilization import get_grid_adaptive_engine
            engine = get_grid_adaptive_engine(self.config)
            report = engine.get_last_report()

            symbols = {}
            multipliers = engine.get_position_multipliers()
            frozen = engine.get_frozen_symbols()

            # 汇总所有币种数据
            all_symbols = set(list(multipliers.keys()) + list(frozen.keys()))
            total_multiplier = 0.0
            total_confidence = 0.0
            active_count = 0
            frozen_count = 0

            for symbol in sorted(all_symbols):
                is_frozen = symbol in frozen
                multiplier = multipliers.get(symbol, 1.0)

                sym_data = {
                    "symbol": symbol,
                    "position_multiplier": multiplier,
                    "is_frozen": is_frozen,
                    "freeze_reason": frozen.get(symbol, ""),
                }

                if report and symbol in report.symbols:
                    rs = report.symbols[symbol]
                    sym_data.update({
                        "total_trades": rs.get("total_trades", 0),
                        "wins": rs.get("wins", 0),
                        "losses": rs.get("losses", 0),
                        "total_pnl": rs.get("total_pnl", 0),
                        "win_rate": rs.get("win_rate", 0),
                        "profit_factor": rs.get("profit_factor", 0),
                        "consecutive_wins": rs.get("consecutive_wins", 0),
                        "consecutive_losses": rs.get("consecutive_losses", 0),
                        "confidence": rs.get("confidence", 0),
                        "last_trade": rs.get("last_trade"),
                    })
                    if not is_frozen:
                        total_confidence += rs.get("confidence", 0)
                        active_count += 1
                else:
                    sym_data.update({
                        "total_trades": 0, "wins": 0, "losses": 0,
                        "total_pnl": 0, "win_rate": 0, "profit_factor": 0,
                        "consecutive_wins": 0, "consecutive_losses": 0,
                        "confidence": 0, "last_trade": None,
                    })
                    if is_frozen:
                        frozen_count += 1

                if not is_frozen:
                    total_multiplier += multiplier
                symbols[symbol] = sym_data

            if active_count > 0:
                avg_confidence = total_confidence / active_count
                avg_multiplier = total_multiplier / active_count
            else:
                avg_confidence = 0.0
                avg_multiplier = 1.0

            result = {
                "symbols": symbols,
                "avg_confidence": round(avg_confidence, 4),
                "avg_multiplier": round(avg_multiplier, 2),
                "active_symbols": active_count,
                "frozen_symbols": frozen_count + (len(frozen) - sum(1 for s in frozen if s in symbols)),
                "actions": report.actions[-20:] if report else [],
                "warnings": report.warnings[-10:] if report else [],
                "config": {
                    "max_position_multiplier": engine._max_position_multiplier,
                    "min_position_multiplier": engine._min_position_multiplier,
                    "step_up_limit": engine._step_up_limit,
                    "step_down_limit": engine._step_down_limit,
                    "win_rate_floor": engine._win_rate_floor,
                    "profit_factor_floor": engine._profit_factor_floor,
                    "min_trades_for_activation": engine._min_trades_for_activation,
                },
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS * 2)
            return result

        except Exception as e:
            logger.debug(f"DashboardEngine: grid adaptive utilization error: {e}")
            return {"timestamp": datetime.now().isoformat()}

    def get_enterprise_sync(self, force_refresh: bool = False) -> Dict[str, Any]:
        """获取企业级同步健康状态（v4.0 新增）"""
        cache_key = "enterprise_sync"
        if not force_refresh:
            cached = self._get_cache(cache_key)
            if cached:
                return cached

        try:
            from core.enterprise_sync import get_sync_engine
            engine = get_sync_engine(self.config)

            health_summary = engine.get_health_summary()
            report = engine.get_last_report()
            stats = engine.get_stats()

            channels = engine.health_monitor.get_all_status()
            active_gaps = engine.recovery_manager.get_active_gaps()
            stale_entities = engine.stale_detector.get_alert_entities()

            result = {
                "overall_health": health_summary.get("overall_health", "unknown"),
                "channel_count": health_summary.get("channel_count", 0),
                "healthy_channels": health_summary.get("healthy_channels", 0),
                "degraded_channels": health_summary.get("degraded_channels", 0),
                "disconnected_channels": health_summary.get("disconnected_channels", 0),
                "stale_entities_count": health_summary.get("stale_entities", 0),
                "active_gaps_count": health_summary.get("active_gaps", 0),
                "version_count": health_summary.get("version_count", 0),
                "total_syncs": stats.get("total_syncs", 0),
                "total_recoveries": stats.get("total_recoveries", 0),
                "total_gaps_detected": stats.get("total_gaps_detected", 0),
                "total_stale_invalidations": stats.get("total_stale_invalidations", 0),
                "channels": channels,
                "active_gaps": active_gaps,
                "stale_entities": stale_entities[:10],
                "alerts": report.alerts if report else [],
                "timestamp": datetime.now().isoformat(),
            }

            self._set_cache(cache_key, result, self.CACHE_TTL_SECONDS * 2)
            return result

        except Exception as e:
            logger.debug(f"DashboardEngine: enterprise sync error: {e}")
            return {"timestamp": datetime.now().isoformat()}

    def update_config(self, new_config: Dict[str, Any]) -> None:
        """热更新配置"""
        self.config = new_config or {}
        dash_cfg = self.config.get("dashboard", {})
        self._db_path = dash_cfg.get("db_path", self._db_path)
        if self._db_path and not os.path.isabs(self._db_path):
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self._db_path = os.path.join(_project_root, self._db_path)
        logger.info("DashboardEngine config updated")
