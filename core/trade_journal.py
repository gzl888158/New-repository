"""交易日志模块，追踪成交记录、持仓快照、资金曲线与日度统计。"""
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from loguru import logger
from sqlalchemy import text

from data.sqlite_storage import SQLiteStorage
from data.redis_cache import RedisCache
from core.unified_layer import Event, EventType

@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    strategy_name: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: int
    entry_time: datetime
    exit_time: datetime
    fees: float
    # 注意口径：pnl_pct 是收益率（百分比），pnl_usdt 才是 USDT 净额。
    # 切勿与 data/sqlite_storage.py 的 ORM TradeRecord.pnl（USDT 净额）混淆。
    pnl_pct: float
    pnl_usdt: float
    win: bool
    entry_signal_type: str
    exit_reason: str
    slippage_cost: float = 0.0   # 滑点成本（预估入账）
    funding_cost: float = 0.0    # 资金费率成本（预估入账）
    spread_cost: float = 0.0     # 点差成本（预估入账）
    trace_id: str = ""           # 全链路 traceID：信号→裁决→订单→记账同源追踪
    confidence: float = 0.0      # 开仓信号置信度（0~1），用于准确率回测分桶

@dataclass
class PositionSnapshot:
    timestamp: datetime
    symbol: str
    strategy_name: str
    direction: str
    quantity: float
    avg_cost: float
    mark_price: float
    unrealized_pnl: float
    margin: float
    leverage: int
    signal_type: str
    allow_sl_reset: bool = True
    closed_quantity: float = 0.0
    realized_pnl_usdt: float = 0.0
    total_fees: float = 0.0
    total_slippage_cost: float = 0.0
    total_funding_cost: float = 0.0
    total_spread_cost: float = 0.0
    trace_id: str = ""
    confidence: float = 0.0

@dataclass
class EquityPoint:
    timestamp: datetime
    total_equity: float
    available_balance: float
    used_margin: float
    unrealized_pnl: float
    realized_pnl: float
    win_rate: float
    total_trades: int

class TradeJournal:
    def __init__(self, config: Dict[str, Any], sqlite_storage: SQLiteStorage, redis_cache: RedisCache, okx_client=None):
        self.config = config
        self.sqlite_storage = sqlite_storage
        self.redis_cache = redis_cache
        self._okx_client = okx_client
        self._event_bus = None
        self._confidence_calibrator = None

        self._trades: Dict[str, TradeRecord] = {}
        self._open_positions: Dict[str, PositionSnapshot] = {}
        self._equity_curve: List[EquityPoint] = []
        self._daily_stats: Dict[str, Dict[str, Any]] = {}
        self._signal_history: List[Dict[str, Any]] = []

        self._total_starting_capital = config["trading"]["total_capital"]
        self._current_equity = self._total_starting_capital
        self._realized_pnl = 0.0
        self._total_fees_paid = 0.0
        self._wins = 0
        self._losses = 0

        self._compound_growth = 1.0
        self._compound_history: List[Dict[str, float]] = []
        self._max_equity = self._total_starting_capital
        self._drawdown_start = datetime.now()
        self._max_drawdown = 0.0

        self._position_sizing_factor = 1.0
        self._risk_per_trade = config["trading"].get("risk_per_trade", 0.01)
        self._equity_target = self._total_starting_capital * config["trading"].get("target_return", 1.1)
        self._trailing_profit_lock = config["trading"].get("trailing_profit_lock", 0.05)

        self._strategy_allocations: Dict[str, float] = {}
        self._strategy_used_capital: Dict[str, float] = {}
        self._daily_risk_limit = self._total_starting_capital * config["trading"].get("daily_risk_limit", 0.03)
        self._daily_risk_used = 0.0
        self._daily_risk_reset_time = datetime.now()

        self._position_limits: Dict[str, float] = {}
        self._max_open_positions = config["trading"].get(
            "max_concurrent_positions", config["trading"].get("max_open_positions", 10)
        )
        self._min_hold_minutes = config.get("trading", {}).get("min_hold_minutes", 5)

        self._initialize_tables()
        self._load_history()
        self._initialize_strategy_allocations()
    
    async def start(self):
        asyncio.create_task(self._mark_price_update_loop())
        # 定期同步权益曲线（即使无交易也更新，确保 equity_curve 反映 OKX 真实权益）
        asyncio.create_task(self._equity_sync_loop())

    def set_trade_cost_analyzer(self, analyzer):
        """注入交易成本分析器，用于将资金费率/滑点/点差成本入账（平仓时回填）。"""
        self._trade_cost_analyzer = analyzer

    def set_event_bus(self, event_bus):
        """P0-2 记账事件化：注入事件总线，开仓/平仓落账时发布 POSITION_OPENED/TRADE_RECORDED。"""
        self._event_bus = event_bus

    def set_confidence_calibrator(self, calibrator):
        """注入置信度校准器，平仓时自动喂样 (predicted_confidence, actual_outcome)。"""
        self._confidence_calibrator = calibrator

    def _publish_event(self, event_type: EventType, data: Dict[str, Any]):
        """发布记账事件（带 trace_id）。失败不影响记账主流程（仅 DEBUG 记录）。"""
        try:
            event_bus = getattr(self, "_event_bus", None)
            if event_bus is None:
                return
            event_bus.publish_sync(Event(event_type, data))
        except Exception as e:
            logger.debug(f"Journal event publish skipped ({getattr(event_type, 'value', event_type)}): {e}")

    def _compute_extra_costs(self, symbol, direction, entry_price, quantity, leverage,
                             entry_time, exit_time):
        """用 TradeCostAnalyzer 计算滑点/资金费率/点差成本（预估入账）。

        返回 (slippage_cost, funding_cost, spread_cost)；未注入分析器或计算失败时返回 (0, 0, 0)。
        """
        analyzer = getattr(self, "_trade_cost_analyzer", None)
        if analyzer is None:
            return 0.0, 0.0, 0.0
        try:
            side = "buy" if direction == "long" else "sell"
            hold_hours = max(0.0, (exit_time - entry_time).total_seconds() / 3600.0)
            cost = analyzer.calculate_full_cost(
                symbol=symbol,
                side=side,
                price=entry_price,
                quantity=quantity,
                leverage=leverage,
                pos_side=direction,
                estimated_hold_hours=hold_hours,
            )
            return cost.slippage_cost, cost.funding_cost, cost.spread_cost
        except Exception as e:
            logger.debug(f"compute extra costs failed for {symbol}: {e}")
            return 0.0, 0.0, 0.0

    async def _equity_sync_loop(self):
        """每60秒同步一次权益曲线，确保 equity_curve 反映 OKX 真实权益而非本地累积值"""
        while True:
            try:
                await asyncio.sleep(60)
                await self._update_equity_curve()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in equity sync loop: {e}")
                await asyncio.sleep(60)

    def _initialize_tables(self):
        conn = self.sqlite_storage.get_connection()
        
        # 注意口径：trades 表与 trade_records 表（data/sqlite_storage.py）的 pnl 列语义不同。
        #   trades.pnl       = 收益率百分比（本模块 dataclass TradeRecord.pnl_pct），
        #   trades.pnl_usdt  = USDT 净额（本模块 dataclass TradeRecord.pnl_usdt）；
        #   trade_records.pnl = USDT 净额（ORM TradeRecord.pnl）。
        # 切勿把 trades.pnl（百分比）当作 USDT 净额使用。
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS trades (
                trade_id TEXT PRIMARY KEY,
                symbol TEXT,
                strategy_name TEXT,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                quantity REAL,
                leverage INTEGER,
                entry_time TEXT,
                exit_time TEXT,
                fees REAL,
                pnl REAL,
                pnl_usdt REAL,
                win INTEGER,
                entry_signal_type TEXT,
                exit_reason TEXT,
                slippage_cost REAL DEFAULT 0,
                funding_cost REAL DEFAULT 0,
                spread_cost REAL DEFAULT 0,
                trace_id TEXT DEFAULT '',
                created_at TEXT
            )
        '''))
        
        # 轻量迁移：为已存在的 trades 表补充成本列（资金费率/滑点/点差入账）
        for col, col_type in (
            ("slippage_cost", "REAL DEFAULT 0"),
            ("funding_cost", "REAL DEFAULT 0"),
            ("spread_cost", "REAL DEFAULT 0"),
            ("trace_id", "TEXT DEFAULT ''"),
            ("confidence", "REAL DEFAULT 0"),
        ):
            try:
                conn.execute(text(f"ALTER TABLE trades ADD COLUMN {col} {col_type}"))
            except Exception:
                pass  # 列已存在
        
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS equity_curve (
                timestamp TEXT PRIMARY KEY,
                total_equity REAL,
                available_balance REAL,
                used_margin REAL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                win_rate REAL,
                total_trades INTEGER
            )
        '''))
        
        conn.execute(text('''
            CREATE TABLE IF NOT EXISTS learning_state (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TEXT
            )
        '''))
        
        conn.commit()
        conn.close()

    def _load_history(self):
        conn = self.sqlite_storage.get_connection()
        
        result = conn.execute(text('SELECT * FROM trades ORDER BY entry_time'))
        rows = result.fetchall()
        
        for row in rows:
            m = row._mapping
            trade = TradeRecord(
                trade_id=m["trade_id"],
                symbol=m["symbol"],
                strategy_name=m["strategy_name"],
                direction=m["direction"],
                entry_price=m["entry_price"],
                exit_price=m["exit_price"],
                quantity=m["quantity"],
                leverage=m["leverage"],
                entry_time=datetime.fromisoformat(m["entry_time"]),
                exit_time=datetime.fromisoformat(m["exit_time"]),
                fees=m["fees"],
                # trades.pnl 存的是百分比（dataclass 的 pnl_pct），trades.pnl_usdt 才是 USDT 净额
                pnl_pct=m["pnl"],
                pnl_usdt=m["pnl_usdt"],
                win=bool(m["win"]),
                entry_signal_type=m["entry_signal_type"],
                exit_reason=m["exit_reason"],
                slippage_cost=m.get("slippage_cost", 0.0) or 0.0,
                funding_cost=m.get("funding_cost", 0.0) or 0.0,
                spread_cost=m.get("spread_cost", 0.0) or 0.0,
                trace_id=m.get("trace_id", "") or "",
            )
            self._trades[trade.trade_id] = trade
            if trade.win:
                self._wins += 1
            else:
                self._losses += 1
            self._realized_pnl += trade.pnl_usdt
        
        result = conn.execute(text('SELECT * FROM equity_curve ORDER BY timestamp'))
        rows = result.fetchall()
        
        for row in rows:
            point = EquityPoint(
                timestamp=datetime.fromisoformat(row[0]),
                total_equity=row[1],
                available_balance=row[2],
                used_margin=row[3],
                unrealized_pnl=row[4],
                realized_pnl=row[5],
                win_rate=row[6],
                total_trades=row[7]
            )
            self._equity_curve.append(point)
        
        if self._equity_curve:
            last_point = self._equity_curve[-1]
            self._current_equity = last_point.total_equity

        # 恢复 open_positions 状态：从 trade_records 表加载 status='open' 的记录
        # 避免系统重启后 _open_positions 为空，导致平仓时走 _open_position 分支而非 _reduce_position
        # 从而 trades 表无法记录完整生命周期（历史数据显示 88 笔 trade_records 但仅 5 笔 trades）
        try:
            open_records = self.sqlite_storage.get_all_open_records()
            for rec in open_records:
                symbol = rec.get("symbol")
                if not symbol or symbol in self._open_positions:
                    continue
                side = str(rec.get("side", "")).lower()
                direction = "long" if side in ("long", "buy") else ("short" if side in ("short", "sell") else "long")
                qty = float(rec.get("quantity", 0) or 0)
                entry_price = float(rec.get("price", 0) or rec.get("filled_price", 0) or 0)
                leverage = int(rec.get("leverage", 1) or 1)
                if qty <= 0 or entry_price <= 0:
                    continue
                snapshot = PositionSnapshot(
                    timestamp=rec.get("create_time") or datetime.now(),
                    symbol=symbol,
                    strategy_name=rec.get("strategy_name", ""),
                    direction=direction,
                    quantity=qty,
                    avg_cost=entry_price,
                    mark_price=entry_price,
                    unrealized_pnl=0.0,
                    margin=qty * entry_price / leverage if leverage > 0 else 0.0,
                    leverage=leverage,
                    signal_type=rec.get("signal_type", "") or "unknown"
                )
                self._open_positions[symbol] = snapshot
            if self._open_positions:
                logger.info(f"Restored {len(self._open_positions)} open positions from trade_records")
        except Exception as e:
            logger.error(f"Failed to restore open_positions from trade_records: {e}")

        conn.close()
        logger.info(f"Loaded {len(self._trades)} historical trades, {len(self._equity_curve)} equity points")

    def _initialize_strategy_allocations(self):
        trading_capital = self._total_starting_capital * self.config["trading"].get("trading_capital_ratio", 0.8)

        # 从 trading 配置读取分配比例（grid_allocation, trend_allocation 等）
        allocation_map = {
            "grid": "grid_allocation",
            "trend": "trend_allocation",
            "scalping": "scalping_allocation",
            "arbitrage": "arbitrage_allocation",
        }

        for strategy_name, allocation_key in allocation_map.items():
            allocation = self.config["trading"].get(allocation_key, 0)
            if allocation > 0:
                self._strategy_allocations[strategy_name] = trading_capital * allocation
                self._strategy_used_capital[strategy_name] = 0.0

        logger.info(f"Initialized strategy allocations: {self._strategy_allocations}")

    def get_available_capital_for_strategy(self, strategy_name: str) -> float:
        allocated = self._strategy_allocations.get(strategy_name, 0.0)
        used = self._strategy_used_capital.get(strategy_name, 0.0)
        return allocated - used

    def reserve_capital_for_strategy(self, strategy_name: str, amount: float) -> bool:
        available = self.get_available_capital_for_strategy(strategy_name)
        if available >= amount:
            self._strategy_used_capital[strategy_name] += amount
            return True
        return False

    def release_capital_for_strategy(self, strategy_name: str, amount: float):
        if strategy_name in self._strategy_used_capital:
            self._strategy_used_capital[strategy_name] = max(0, self._strategy_used_capital[strategy_name] - amount)

    def check_daily_risk_limit(self, risk_amount: float) -> bool:
        now = datetime.now()
        if now - self._daily_risk_reset_time >= timedelta(days=1):
            self._daily_risk_used = 0.0
            self._daily_risk_reset_time = now
        
        return (self._daily_risk_used + risk_amount) <= self._daily_risk_limit

    def record_daily_risk(self, risk_amount: float):
        now = datetime.now()
        if now - self._daily_risk_reset_time >= timedelta(days=1):
            self._daily_risk_used = 0.0
            self._daily_risk_reset_time = now
        
        self._daily_risk_used += risk_amount

    def check_max_positions(self) -> bool:
        return len(self._open_positions) < self._max_open_positions

    def get_account_summary(self) -> Dict[str, Any]:
        used_margin = sum(p.margin for p in self._open_positions.values())
        unrealized_pnl = sum(self._calculate_unrealized_pnl(p) for p in self._open_positions.values())
        available_balance = self._current_equity - used_margin + unrealized_pnl
        total_equity = self._current_equity + unrealized_pnl
        
        total_trades = self._wins + self._losses
        win_rate = self._wins / total_trades if total_trades > 0 else 0
        
        now = datetime.now()
        daily_risk_remaining = max(0, self._daily_risk_limit - self._daily_risk_used)
        
        return {
            "total_equity": total_equity,
            "available_balance": available_balance,
            "used_margin": used_margin,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": self._realized_pnl,
            "compound_growth": self._compound_growth,
            "max_drawdown": self._max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "open_positions": len(self._open_positions),
            "max_open_positions": self._max_open_positions,
            "daily_risk_used": self._daily_risk_used,
            "daily_risk_limit": self._daily_risk_limit,
            "daily_risk_remaining": daily_risk_remaining,
            "position_sizing_factor": self._position_sizing_factor,
            "strategy_allocations": self._strategy_allocations,
            "strategy_used_capital": self._strategy_used_capital,
            "last_updated": now.isoformat()
        }

    def get_strategy_performance(self) -> Dict[str, Any]:
        strategy_stats = {}
        
        for trade in self._trades.values():
            if trade.strategy_name not in strategy_stats:
                strategy_stats[trade.strategy_name] = {
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                    "avg_pnl": 0.0,
                    "win_rate": 0.0,
                    "max_win": 0.0,
                    "max_loss": 0.0
                }
            
            stats = strategy_stats[trade.strategy_name]
            stats["total_trades"] += 1
            stats["total_pnl"] += trade.pnl_usdt
            
            if trade.win:
                stats["wins"] += 1
                stats["max_win"] = max(stats["max_win"], trade.pnl_usdt)
            else:
                stats["losses"] += 1
                stats["max_loss"] = min(stats["max_loss"], trade.pnl_usdt)
            
            stats["win_rate"] = stats["wins"] / stats["total_trades"]
            stats["avg_pnl"] = stats["total_pnl"] / stats["total_trades"]
        
        return strategy_stats

    def get_confidence_bucketed_win_rate(self, n_buckets: int = 5) -> Dict[str, Any]:
        """按信号置信度分桶统计胜率，用于校准曲线和信号准确率回测。"""
        trades_with_conf = [t for t in self._trades.values() if t.confidence > 0]
        if not trades_with_conf:
            return {"buckets": [], "total_samples": 0, "message": "no trades with confidence data"}

        bucket_size = 1.0 / n_buckets
        buckets = []
        for i in range(n_buckets):
            lo = i * bucket_size
            hi = (i + 1) * bucket_size
            bucket_trades = [t for t in trades_with_conf if lo <= t.confidence < hi or (i == n_buckets - 1 and t.confidence == hi)]
            if not bucket_trades:
                buckets.append({"range": f"{lo:.2f}-{hi:.2f}", "count": 0, "wins": 0, "win_rate": 0.0, "avg_pnl": 0.0})
                continue
            wins = sum(1 for t in bucket_trades if t.win)
            buckets.append({
                "range": f"{lo:.2f}-{hi:.2f}",
                "count": len(bucket_trades),
                "wins": wins,
                "win_rate": wins / len(bucket_trades),
                "avg_pnl": sum(t.pnl_usdt for t in bucket_trades) / len(bucket_trades),
                "avg_confidence": sum(t.confidence for t in bucket_trades) / len(bucket_trades),
            })

        # Brier score: 置信度预测 vs 实际结果的 MSE
        brier = sum((t.confidence - (1.0 if t.win else 0.0)) ** 2 for t in trades_with_conf) / len(trades_with_conf)

        return {
            "buckets": buckets,
            "total_samples": len(trades_with_conf),
            "brier_score": round(brier, 4),
            "overall_win_rate": sum(1 for t in trades_with_conf if t.win) / len(trades_with_conf),
        }

    def check_sl_reset_allowed(self, symbol: str) -> bool:
        if symbol not in self._open_positions:
            return True
        
        snapshot = self._open_positions[symbol]
        
        if not snapshot.allow_sl_reset:
            return False
        
        # P0: margin可能为0（刚恢复的持仓），防止除零
        if not snapshot.margin or snapshot.margin <= 0:
            return False
        
        unrealized_pnl_pct = snapshot.unrealized_pnl / snapshot.margin
        
        if unrealized_pnl_pct < -0.5:
            logger.warning(f"SL reset blocked: {symbol} has deep unrealized loss ({unrealized_pnl_pct:.2%})")
            return False
        
        return True

    def set_sl_reset_allowed(self, symbol: str, allowed: bool):
        if symbol in self._open_positions:
            self._open_positions[symbol].allow_sl_reset = allowed

    def _normalize_direction(self, direction: str) -> str:
        direction_map = {
            "buy": "long",
            "sell": "short",
            "long": "long",
            "short": "short"
        }
        return direction_map.get(direction.lower(), "long")

    @staticmethod
    def _infer_exit_reason(signal_type: str) -> str:
        """从 signal_type 推断标准化退出原因（exit_reason 全链路补全）。

        优先级：止损 > 止盈 > 反转 > 利润锁定 > 移动止损 > 强平 > 退出。
        无任何线索时返回 "close"（仅在 truly unknown 平仓时兜底）。
        """
        sig = (signal_type or "").lower()
        if "stop_loss" in sig or "stop" in sig:
            return "stop_loss"
        if "take_profit" in sig or "tp" in sig:
            return "take_profit"
        if "reversal" in sig:
            return "reversal_partial_exit"
        if "profit_lock" in sig or "breakeven" in sig:
            return "profit_lock_breakeven"
        if "trailing" in sig:
            return "trailing_stop"
        if "liquidation" in sig or "margin_call" in sig:
            return "liquidation"
        if "exit" in sig:
            return "exit_signal"
        return "close"

    def record_signal(self, signal_record: Dict[str, Any]):
        """记录策略信号（用于信号审计与去重）。

        P0 修复：scheduler._on_strategy_signal 曾调用不存在的 record_signal 导致
        信号在写入记账前 AttributeError 中断。现提供内存信号历史记录。
        """
        try:
            self._signal_history.append(signal_record)
            # 防内存泄漏：最多保留最近 1000 条
            if len(self._signal_history) > 1000:
                self._signal_history = self._signal_history[-1000:]
        except Exception as e:
            logger.debug(f"record_signal error: {e}")

    async def record_fill(self, fill_data: Dict[str, Any]):
        trade_id = fill_data["trade_id"]
        symbol = fill_data["symbol"]
        strategy_name = fill_data["strategy_name"]
        direction = self._normalize_direction(fill_data["direction"])
        price = fill_data["price"]
        quantity = fill_data["quantity"]
        leverage = fill_data["leverage"]
        fees = fill_data.get("fees", 0)
        signal_type = fill_data.get("signal_type", "unknown")
        # 缺省值从 "manual" 改为 ""，避免平仓 reason 未透传时被误记为"手工平仓"，
        # 下游 _reduce_position/_try_close_from_trade_records 会统一兜底为 "close"
        exit_reason = fill_data.get("exit_reason", "")
        # 全链路补全：exit_reason 为空时从 signal_type 推断标准化退出原因，禁止静默兜底 "close"
        if not exit_reason:
            exit_reason = self._infer_exit_reason(signal_type)
        trace_id = fill_data.get("trace_id", "")
        confidence = fill_data.get("confidence", 0.0)
        
        if symbol in self._open_positions:
            existing_position = self._open_positions[symbol]
            existing_direction = self._normalize_direction(existing_position.direction)
            
            if existing_direction == direction:
                await self._add_to_position(symbol, price, quantity, leverage, fees, signal_type, trace_id)
            else:
                await self._reduce_position(symbol, price, quantity, leverage, fees, exit_reason, trace_id)
        else:
            # 检测是否为平仓（_open_positions 中无记录但 trade_records 中有开仓记录）
            close_detected = await self._try_close_from_trade_records(
                trade_id, symbol, direction, price, quantity, leverage, fees, signal_type, exit_reason, trace_id
            )
            if not close_detected:
                await self._open_position(trade_id, symbol, strategy_name, direction, price, quantity, leverage, fees, signal_type, trace_id, confidence=confidence)
        
        await self._update_equity_curve()

    async def _open_position(self, trade_id: str, symbol: str, strategy_name: str, direction: str,
                            price: float, quantity: float, leverage: int, fees: float, signal_type: str,
                            trace_id: str = "", confidence: float = 0.0):
        margin = quantity * price / leverage
        position_value = quantity * price

        if not self.check_max_positions():
            logger.warning(f"Cannot open position: max open positions reached ({self._max_open_positions})")
            return

        risk_amount = margin * self._risk_per_trade
        if not self.check_daily_risk_limit(risk_amount):
            logger.warning(f"Cannot open position: daily risk limit exceeded")
            return

        if not self.reserve_capital_for_strategy(strategy_name, margin):
            logger.warning(f"Cannot open position: insufficient capital for {strategy_name}")
            return

        snapshot = PositionSnapshot(
            timestamp=datetime.now(),
            symbol=symbol,
            strategy_name=strategy_name,
            direction=direction,
            quantity=quantity,
            avg_cost=price,
            mark_price=price,
            unrealized_pnl=0,
            margin=margin,
            leverage=leverage,
            signal_type=signal_type,
            trace_id=trace_id,
            confidence=confidence,
        )

        self._open_positions[symbol] = snapshot

        # 精确扣除手续费
        self._current_equity -= fees
        self._total_fees_paid += fees
        self.record_daily_risk(risk_amount)

        # P0-2 记账事件化：开仓落账发布 POSITION_OPENED（带 traceID 贯穿）
        self._publish_event(EventType.POSITION_OPENED, {
            "trade_id": trade_id,
            "symbol": symbol,
            "strategy_name": strategy_name,
            "direction": direction,
            "price": price,
            "quantity": quantity,
            "leverage": leverage,
            "fees": fees,
            "signal_type": signal_type,
            "trace_id": trace_id,
        })

        logger.info(f"Position opened: {direction} {symbol} @ {price:.4f}, qty: {quantity:.4f}, "
                    f"margin: {margin:.2f}, fee: {fees:.4f}, strategy: {strategy_name}")

    async def _add_to_position(self, symbol: str, price: float, quantity: float, leverage: int, fees: float, signal_type: str, trace_id: str = ""):
        snapshot = self._open_positions[symbol]
        
        # traceID 贯穿：加仓若携带新 traceID 则更新快照（平仓落账时聚合到最新链路）
        if trace_id:
            snapshot.trace_id = trace_id

        new_total_qty = snapshot.quantity + quantity
        new_avg_cost = (snapshot.avg_cost * snapshot.quantity + price * quantity) / new_total_qty
        new_margin = new_total_qty * new_avg_cost / leverage

        # 账本闭环：加仓增加持仓保证金，必须同步占用策略已用资金，
        # 否则 strategy_used_capital 与实际持仓保证金脱节
        added_margin = new_margin - snapshot.margin
        if added_margin > 0 and not self.reserve_capital_for_strategy(snapshot.strategy_name, added_margin):
            logger.warning(f"Cannot add position: insufficient capital for {snapshot.strategy_name} "
                           f"(need {added_margin:.4f})")
            return
        
        snapshot.quantity = new_total_qty
        snapshot.avg_cost = new_avg_cost
        snapshot.margin = new_margin
        
        self._current_equity -= fees
        
        logger.info(f"Position added: {snapshot.direction} {symbol} @ {price:.4f}, qty: {quantity:.4f}, new avg_cost: {new_avg_cost:.4f}, total qty: {new_total_qty:.4f}")

    async def _try_close_from_trade_records(self, trade_id: str, symbol: str, direction: str,
                                             price: float, quantity: float, leverage: int, fees: float,
                                             signal_type: str, exit_reason: str, trace_id: str = "") -> bool:
        """检测是否为平仓：_open_positions 中无记录，但 trade_records 表中有该 symbol 的开仓记录。
        若方向相反则视为平仓，构建 TradeRecord 并保存到 trades 表。"""
        # 企业级修复：不再过滤 manual/空 exit_reason。
        # 方向相反本身就是平仓的强信号，manual 平仓（如 AVAX 巨亏单）也必须进入 trades 表，
        # 否则平仓记录不完整，导致 trade_records 与 trades 数量严重背离。
        if not exit_reason:
            exit_reason = "close"

        try:
            open_records = self.sqlite_storage.get_all_open_records()
            matching_open = None
            for rec in open_records:
                if rec.get("symbol") == symbol:
                    matching_open = rec
                    break

            if not matching_open:
                return False

            rec_side = str(matching_open.get("side", "")).lower()
            rec_direction = "long" if rec_side in ("long", "buy") else ("short" if rec_side in ("short", "sell") else "long")

            # 方向相同则不是平仓
            if rec_direction == direction:
                return False

            entry_price_val = float(matching_open.get("price", 0) or matching_open.get("filled_price", 0) or 0)
            rec_qty = float(matching_open.get("quantity", 0) or 0)
            rec_leverage = int(matching_open.get("leverage", 1) or 1)
            entry_time_raw = matching_open.get("create_time")
            if entry_time_raw:
                if isinstance(entry_time_raw, datetime):
                    entry_dt = entry_time_raw
                else:
                    entry_dt = datetime.fromisoformat(str(entry_time_raw))
            else:
                entry_dt = datetime.now()

            if rec_qty <= 0 or entry_price_val <= 0:
                return False

            pnl_per_unit = (price - entry_price_val) / entry_price_val if rec_direction == "long" else (entry_price_val - price) / entry_price_val
            gross_pnl = pnl_per_unit * quantity * entry_price_val
            exit_dt = datetime.now()
            slippage_cost, funding_cost, spread_cost = self._compute_extra_costs(
                symbol, rec_direction, entry_price_val, quantity, rec_leverage, entry_dt, exit_dt
            )
            pnl_usdt = gross_pnl - fees - slippage_cost - funding_cost - spread_cost
            win = pnl_usdt > 0

            trade_record = TradeRecord(
                trade_id=trade_id,
                symbol=symbol,
                strategy_name=matching_open.get("strategy_name", ""),
                direction=rec_direction,
                entry_price=entry_price_val,
                exit_price=price,
                quantity=quantity,
                leverage=rec_leverage,
                entry_time=entry_dt,
                exit_time=exit_dt,
                fees=fees,
                slippage_cost=slippage_cost,
                funding_cost=funding_cost,
                spread_cost=spread_cost,
                pnl_pct=pnl_per_unit,
                pnl_usdt=pnl_usdt,
                win=win,
                entry_signal_type=matching_open.get("signal_type") or signal_type or "unknown",
                exit_reason=exit_reason,
                trace_id=trace_id or matching_open.get("trace_id", "") or ""
            )

            self._trades[trade_record.trade_id] = trade_record
            if win:
                self._wins += 1
            else:
                self._losses += 1
            self._realized_pnl += pnl_usdt
            self._current_equity += pnl_usdt
            self._total_fees_paid += fees

            await self._save_trade(trade_record)
            # 账本闭环：平仓后必须关闭 trade_records 中的 open 记录，
            # 否则该记录仍停留 open 状态，导致 get_all_open_records 重复匹配、幽灵持仓累积
            matching_open_id = matching_open.get("id")
            if matching_open_id:
                # 数据完整性的源缺陷修复：此处之前只写 status/close_time/exit_reason，
                # 却不回写 pnl 与 filled_price，导致 trade_records 中所有经 TradeJournal
                # 平仓的记录 pnl 恒为 0、filled_price 恒为 0（虽然后续 PnL 对账可回填，
                # 但会留下大量 pnl=0 假数据，污染 strategy_performance 与 PnL 对账口径）。
                # 现将真实 pnl 与平仓成交价同步回写，保持 trade_records 与 trades 双账本一致。
                self.sqlite_storage.update_trade_record(
                    matching_open_id,
                    {
                        "status": "closed",
                        "close_time": exit_dt.isoformat(),
                        "exit_reason": exit_reason,
                        "pnl": pnl_usdt,
                        "filled_price": price,
                    }
                )
            logger.info(f"Trade closed from trade_records: {rec_direction} {symbol} qty={quantity:.4f} "
                        f"entry={entry_price_val:.4f} exit={price:.4f} pnl={pnl_usdt:.4f} reason={exit_reason}")
            return True

        except Exception as e:
            logger.error(f"Error detecting close from trade_records for {symbol}: {e}")
            return False

    async def _reduce_position(self, symbol: str, exit_price: float, quantity: float, leverage: int, fees: float, exit_reason: str, trace_id: str = ""):
        snapshot = self._open_positions[symbol]

        # traceID 贯穿：平仓若未携带新 traceID，回退到快照开仓时的 traceID
        if not trace_id:
            trace_id = snapshot.trace_id or ""

        # A3: exit_reason 兜底，禁止空字符串写入（历史事故：ETH 平仓 reason 为空）
        if not exit_reason:
            exit_reason = "close"

        # 最小持仓周期检查（除非是止损/强平信号）
        allow_early_close = any(kw in exit_reason.lower() for kw in ["stop_loss", "take_profit", "liquidation", "margin_call", "trailing", "breakeven", "dynamic"])
        if not allow_early_close:
            hold_minutes = (datetime.now() - snapshot.timestamp).total_seconds() / 60
            if hold_minutes < self._min_hold_minutes:
                logger.debug(f"Skip reduce: {symbol} held {hold_minutes:.1f}min < min {self._min_hold_minutes}min (reason: {exit_reason})")
                return

        reduce_qty = min(quantity, snapshot.quantity)
        remaining_qty = snapshot.quantity - reduce_qty

        pnl_per_unit = (exit_price - snapshot.avg_cost) / snapshot.avg_cost if snapshot.direction == "long" else (snapshot.avg_cost - exit_price) / snapshot.avg_cost
        gross_pnl = pnl_per_unit * reduce_qty * snapshot.avg_cost
        slippage_cost, funding_cost, spread_cost = self._compute_extra_costs(
            snapshot.symbol, snapshot.direction, snapshot.avg_cost, reduce_qty,
            snapshot.leverage, snapshot.timestamp, datetime.now()
        )
        pnl_usdt = gross_pnl - fees - slippage_cost - funding_cost - spread_cost
        win = pnl_usdt > 0

        # P0: 爆仓检测 —— 亏损超过保证金90%且exit_reason不是明确平仓信号时自动标记
        if not win and exit_reason != "liquidation":
            margin = snapshot.margin * (reduce_qty / snapshot.quantity)
            if margin > 0 and abs(pnl_usdt) >= margin * 0.9:
                leveraged_loss_pct = abs(pnl_usdt) / margin if margin > 0 else 0
                if leveraged_loss_pct >= 0.9:
                    logger.warning(f"LIQUIDATION DETECTED: {symbol} PnL={pnl_usdt:.2f} exceeds "
                                   f"{leveraged_loss_pct:.0%} of margin={margin:.2f}, "
                                   f"original reason={exit_reason}, overriding to liquidation")
                    exit_reason = "liquidation"
                    # 爆仓时手续费由交易所承担，PNL修正
                    pnl_usdt = gross_pnl
                    win = False  # 爆仓永远是亏损

        # A2: 累计部分平仓（同一 open 持仓多次减仓聚合为一条 trades 记录）
        snapshot.closed_quantity = (snapshot.closed_quantity or 0.0) + reduce_qty
        snapshot.realized_pnl_usdt = (snapshot.realized_pnl_usdt or 0.0) + pnl_usdt
        snapshot.total_fees = (snapshot.total_fees or 0.0) + fees
        snapshot.total_slippage_cost = (snapshot.total_slippage_cost or 0.0) + slippage_cost
        snapshot.total_funding_cost = (snapshot.total_funding_cost or 0.0) + funding_cost
        snapshot.total_spread_cost = (snapshot.total_spread_cost or 0.0) + spread_cost

        # A1: 从 open 记录回填权威 strategy_name / signal_type（消除 trend/grid 错乱）
        open_rec = self.sqlite_storage.get_latest_open_record(symbol, snapshot.strategy_name)
        if not open_rec:
            open_rec = self.sqlite_storage.get_latest_open_record(symbol)

        if open_rec:
            strategy_name = open_rec.get("strategy_name") or snapshot.strategy_name
            entry_signal_type = open_rec.get("signal_type") or snapshot.signal_type or "unknown"
            # A2: 稳定 trade_id 优先复用 open 记录 id，聚合同一持仓的部分平仓
            stable_trade_id = str(open_rec.get("id") or "")
        else:
            strategy_name = snapshot.strategy_name
            entry_signal_type = snapshot.signal_type
            stable_trade_id = ""

        if not stable_trade_id:
            # 回退：无 open 记录时用 symbol+direction+entry_time 派生稳定聚合键
            stable_trade_id = f"{snapshot.symbol}_{snapshot.direction}_{snapshot.timestamp.strftime('%Y%m%d%H%M%S')}"

        aggregated_win = snapshot.realized_pnl_usdt > 0

        trade_record = TradeRecord(
            trade_id=stable_trade_id,
            symbol=snapshot.symbol,
            strategy_name=strategy_name,
            direction=snapshot.direction,
            entry_price=snapshot.avg_cost,
            exit_price=exit_price,
            quantity=snapshot.closed_quantity,
            leverage=snapshot.leverage,
            entry_time=snapshot.timestamp,
            exit_time=datetime.now(),
            fees=snapshot.total_fees,
            slippage_cost=snapshot.total_slippage_cost,
            funding_cost=snapshot.total_funding_cost,
            spread_cost=snapshot.total_spread_cost,
            pnl_pct=pnl_per_unit,
            pnl_usdt=snapshot.realized_pnl_usdt,
            win=aggregated_win,
            entry_signal_type=entry_signal_type,
            exit_reason=exit_reason,
            trace_id=trace_id,
            confidence=snapshot.confidence,
        )

        self._trades[stable_trade_id] = trade_record

        # 现金流水：每次部分平仓都实现 pnl 与手续费
        self._realized_pnl += pnl_usdt
        self._current_equity += pnl_usdt
        self._total_fees_paid += fees

        await self._save_trade(trade_record)

        if remaining_qty <= 0:
            # 胜率统计：整个持仓平完时按聚合结果计一次（避免部分平仓被计为多笔盈亏）
            if aggregated_win:
                self._wins += 1
            else:
                self._losses += 1

            # 置信度校准器喂样：平仓时用信号置信度 + 实际胜负训练校准模型
            calibrator = getattr(self, "_confidence_calibrator", None)
            if calibrator and snapshot.confidence > 0:
                try:
                    await calibrator.add_sample(snapshot.confidence, aggregated_win)
                except Exception as e:
                    logger.debug(f"Confidence calibrator feed failed: {e}")

            self.release_capital_for_strategy(snapshot.strategy_name, snapshot.margin)
            del self._open_positions[symbol]
            logger.info(f"Position fully closed: {snapshot.symbol}, gross: {gross_pnl:.4f}, fee: {fees:.4f}, "
                        f"net PnL: {pnl_usdt:.4f} USDT ({pnl_per_unit:.2%}), win: {aggregated_win}, reason: {exit_reason}")
        else:
            released_margin = (snapshot.quantity - remaining_qty) * snapshot.avg_cost / snapshot.leverage
            self.release_capital_for_strategy(snapshot.strategy_name, released_margin)

            snapshot.quantity = remaining_qty
            snapshot.margin = remaining_qty * snapshot.avg_cost / snapshot.leverage
            logger.info(f"Position partially closed: {snapshot.symbol}, reduced qty: {reduce_qty:.4f}, "
                        f"remaining: {remaining_qty:.4f}, net PnL: {pnl_usdt:.4f} USDT (fee: {fees:.4f})")

    def mark_position_closed(self, symbol: str, strategy_name: str = None, reason: str = "ghost_close"):
        """将 journal 内存持仓标记为已关闭（幽灵持仓清理）。

        与 sqlite_storage.close_open_record 配合，同步清理内存状态，避免
        状态漂移（历史事故：cleanup_position 调用本方法但方法缺失，异常被静默吞掉）。
        """
        snapshot = self._open_positions.get(symbol)
        if not snapshot:
            return
        if strategy_name and snapshot.strategy_name != strategy_name:
            return
        self.release_capital_for_strategy(snapshot.strategy_name, snapshot.margin)
        del self._open_positions[symbol]
        logger.info(f"Position marked closed in journal: {symbol} (reason={reason})")

    async def _save_trade(self, trade: TradeRecord):
        conn = self.sqlite_storage.get_connection()
        
        conn.execute(text('''
            INSERT OR REPLACE INTO trades 
            (trade_id, symbol, strategy_name, direction, entry_price, exit_price, quantity, leverage,
             entry_time, exit_time, fees, pnl, pnl_usdt, win, entry_signal_type, exit_reason,
             slippage_cost, funding_cost, spread_cost, trace_id, confidence, created_at)
            VALUES (:trade_id, :symbol, :strategy_name, :direction, :entry_price, :exit_price, :quantity, :leverage,
             :entry_time, :exit_time, :fees, :pnl, :pnl_usdt, :win, :entry_signal_type, :exit_reason,
             :slippage_cost, :funding_cost, :spread_cost, :trace_id, :confidence, :created_at)
        '''), {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "strategy_name": trade.strategy_name,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "leverage": trade.leverage,
            "entry_time": trade.entry_time.isoformat(),
            "exit_time": trade.exit_time.isoformat(),
            "fees": trade.fees,
            "pnl": trade.pnl_pct,
            "pnl_usdt": trade.pnl_usdt,
            "win": int(trade.win),
            "entry_signal_type": trade.entry_signal_type,
            "exit_reason": trade.exit_reason,
            "slippage_cost": trade.slippage_cost,
            "funding_cost": trade.funding_cost,
            "spread_cost": trade.spread_cost,
            "trace_id": trade.trace_id,
            "confidence": trade.confidence,
            "created_at": datetime.now().isoformat()
        })
        
        conn.commit()
        conn.close()

        # P0-2 记账事件化：平仓落账发布 TRADE_RECORDED（带 traceID 贯穿）
        self._publish_event(EventType.TRADE_RECORDED, {
            "trade_id": trade.trade_id,
            "symbol": trade.symbol,
            "strategy_name": trade.strategy_name,
            "direction": trade.direction,
            "entry_price": trade.entry_price,
            "exit_price": trade.exit_price,
            "quantity": trade.quantity,
            "pnl_usdt": trade.pnl_usdt,
            "win": trade.win,
            "entry_signal_type": trade.entry_signal_type,
            "exit_reason": trade.exit_reason,
            "trace_id": trade.trace_id,
        })

    async def _update_equity_curve(self):
        # 优先使用 OKX 真实仓位数据计算 used_margin 和 unrealized_pnl
        # 本地 _open_positions 可能与 OKX 真实持仓不一致，导致 equity_curve 偏差
        okx_used_margin = 0.0
        okx_unrealized_pnl = 0.0

        # 优先用 OKX 真实账户权益（避免本地累积偏差越来越大）
        okx_total_equity = None
        okx_avail_balance = None
        if self._okx_client:
            try:
                from datetime import datetime as _dt
                cache_key = "_okx_equity_cache"
                cache_ts_key = "_okx_equity_cache_ts"
                now_ts = _dt.now().timestamp()
                if not hasattr(self, cache_ts_key) or now_ts - getattr(self, cache_ts_key, 0) > 5:
                    account_info = self._okx_client.get_account_info()
                    if account_info:
                        okx_total_equity = float(account_info.get("totalEq", 0) or 0)
                        details = account_info.get("details", [])
                        for detail in details:
                            if detail.get("ccy") == "USDT":
                                okx_avail_balance = float(detail.get("availBal", 0) or detail.get("cashBal", 0) or 0)
                                # frozenBal 是冻结保证金（已开仓占用的保证金）
                                frozen_bal = float(detail.get("frozenBal", 0) or 0)
                                okx_used_margin = frozen_bal
                                break
                    setattr(self, cache_key, (okx_total_equity, okx_avail_balance, okx_used_margin))
                    setattr(self, cache_ts_key, now_ts)
                else:
                    okx_total_equity, okx_avail_balance, okx_used_margin = getattr(self, cache_key, (None, None, 0.0))
            except Exception as e:
                logger.debug(f"Failed to fetch OKX account info for equity sync: {e}")

        # 如果 OKX 数据不可用，回退到本地计算
        if okx_total_equity and okx_total_equity > 0:
            # 从 OKX 实际持仓获取 unrealized_pnl
            try:
                positions = self._okx_client.get_positions() if self._okx_client else []
                okx_unrealized_pnl = sum(
                    float(p.get("upl", 0) or 0) for p in positions
                    if abs(float(p.get("pos", 0) or p.get("position", 0) or 0)) > 0
                )
            except Exception:
                okx_unrealized_pnl = 0.0

            self._current_equity = okx_total_equity - okx_unrealized_pnl
            total_equity = okx_total_equity
            available_balance = okx_avail_balance if okx_avail_balance is not None else (self._current_equity - okx_used_margin + okx_unrealized_pnl)
            used_margin = okx_used_margin
            unrealized_pnl = okx_unrealized_pnl
        else:
            # 回退：使用本地数据
            used_margin = sum(p.margin for p in self._open_positions.values())
            unrealized_pnl = sum(self._calculate_unrealized_pnl(p) for p in self._open_positions.values())
            available_balance = self._current_equity - used_margin + unrealized_pnl
            total_equity = self._current_equity + unrealized_pnl

        total_trades = self._wins + self._losses
        win_rate = self._wins / total_trades if total_trades > 0 else 0

        self._update_compound_growth(total_equity)
        self._update_drawdown(total_equity)
        self._update_position_sizing_factor(total_equity)

        point = EquityPoint(
            timestamp=datetime.now(),
            total_equity=total_equity,
            available_balance=available_balance,
            used_margin=used_margin,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            win_rate=win_rate,
            total_trades=total_trades
        )

        self._equity_curve.append(point)

        if len(self._equity_curve) > 10000:
            self._equity_curve = self._equity_curve[-10000:]

        await self._save_equity_point(point)

    def _update_compound_growth(self, total_equity: float):
        self._compound_growth = total_equity / self._total_starting_capital
        
        self._compound_history.append({
            "timestamp": datetime.now(),
            "growth": self._compound_growth,
            "equity": total_equity
        })
        
        if len(self._compound_history) > 365:
            self._compound_history = self._compound_history[-365:]

    def _update_drawdown(self, total_equity: float):
        if total_equity > self._max_equity:
            self._max_equity = total_equity
            self._drawdown_start = datetime.now()
        
        drawdown = (self._max_equity - total_equity) / self._max_equity
        if drawdown > self._max_drawdown:
            self._max_drawdown = drawdown

    def _update_position_sizing_factor(self, total_equity: float):
        distance_to_target = (self._equity_target - total_equity) / self._equity_target

        if distance_to_target < 0:
            self._position_sizing_factor = 0.5
        elif distance_to_target > 0.1:
            self._position_sizing_factor = 1.2
        else:
            self._position_sizing_factor = 1.0

        drawdown_factor = max(0.5, 1 - self._max_drawdown * 5)
        self._position_sizing_factor *= drawdown_factor

        # 复利因子：盈利后按比例增加仓位
        compound_enabled = self.config.get("trading", {}).get("compound_enabled", True)
        if compound_enabled and self._total_starting_capital > 0:
            compound_ratio = self.config.get("trading", {}).get("compound_reinvest_ratio", 0.6)
            growth = total_equity / self._total_starting_capital
            compound_factor = max(0.5, min(2.5, growth ** compound_ratio))
            self._position_sizing_factor *= compound_factor

        self._position_sizing_factor = max(0.3, min(2.5, self._position_sizing_factor))

    def get_position_size(self, symbol: str, price: float, leverage: int) -> float:
        risk_amount = self._current_equity * self._risk_per_trade * self._position_sizing_factor
        position_value = risk_amount * leverage
        return position_value / price

    def get_active_positions(self) -> List[Dict[str, Any]]:
        """返回当前活跃持仓列表（供 ContributionAnalyzer 查询未实现盈亏）"""
        result = []
        for symbol, snapshot in self._open_positions.items():
            upl = self._calculate_unrealized_pnl(snapshot)
            result.append({
                "symbol": symbol,
                "strategy_name": snapshot.strategy_name,
                "unrealized_pnl": upl,
                "quantity": snapshot.quantity,
                "entry_price": snapshot.avg_cost,
                "direction": snapshot.direction,
            })
        return result

    def get_compound_stats(self) -> Dict[str, Any]:
        if not self._compound_history:
            return {
                "compound_growth": 1.0,
                "total_return": 0,
                "max_drawdown": 0,
                "current_equity": self._current_equity,
                "position_sizing_factor": 1.0,
                "distance_to_target": 0
            }
        
        growth_values = [h["growth"] for h in self._compound_history]
        
        return {
            "compound_growth": self._compound_growth,
            "total_return": (self._current_equity - self._total_starting_capital) / self._total_starting_capital,
            "max_drawdown": self._max_drawdown,
            "current_equity": self._current_equity,
            "position_sizing_factor": self._position_sizing_factor,
            "distance_to_target": (self._equity_target - self._current_equity) / self._equity_target,
            "avg_daily_growth": np.mean(np.diff(growth_values)) if len(growth_values) > 1 else 0,
            "growth_volatility": np.std(growth_values) if len(growth_values) > 1 else 0
        }

    def _calculate_unrealized_pnl(self, snapshot: PositionSnapshot) -> float:
        if snapshot.direction == "long":
            return (snapshot.mark_price - snapshot.avg_cost) * snapshot.quantity
        else:
            return (snapshot.avg_cost - snapshot.mark_price) * snapshot.quantity

    async def _mark_price_update_loop(self):
        while True:
            try:
                if self._okx_client and self._open_positions:
                    for symbol, snapshot in list(self._open_positions.items()):
                        ticker = await asyncio.to_thread(self._okx_client.get_ticker, symbol)
                        if ticker:
                            snapshot.mark_price = float(ticker["last"])
                            snapshot.unrealized_pnl = self._calculate_unrealized_pnl(snapshot)
            except Exception as e:
                logger.error(f"Error updating mark prices: {e}")
            
            await asyncio.sleep(5)

    async def _save_equity_point(self, point: EquityPoint):
        conn = self.sqlite_storage.get_connection()
        
        conn.execute(text('''
            INSERT OR REPLACE INTO equity_curve 
            (timestamp, total_equity, available_balance, used_margin, unrealized_pnl, realized_pnl, win_rate, total_trades)
            VALUES (:timestamp, :total_equity, :available_balance, :used_margin, :unrealized_pnl, :realized_pnl, :win_rate, :total_trades)
        '''), {
            "timestamp": point.timestamp.isoformat(),
            "total_equity": point.total_equity,
            "available_balance": point.available_balance,
            "used_margin": point.used_margin,
            "unrealized_pnl": point.unrealized_pnl,
            "realized_pnl": point.realized_pnl,
            "win_rate": point.win_rate,
            "total_trades": point.total_trades
        })
        
        conn.commit()
        conn.close()

    def get_trade_stats(self) -> Dict[str, Any]:
        total_trades = self._wins + self._losses
        win_rate = self._wins / total_trades if total_trades > 0 else 0

        all_pnl = [t.pnl_usdt for t in self._trades.values()]
        avg_win = np.mean([t.pnl_usdt for t in self._trades.values() if t.win]) if self._wins > 0 else 0
        avg_loss = np.mean([t.pnl_usdt for t in self._trades.values() if not t.win]) if self._losses > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        max_drawdown = self._calculate_max_drawdown()
        sharpe_ratio = self._calculate_sharpe_ratio()

        return {
            "total_trades": total_trades,
            "wins": self._wins,
            "losses": self._losses,
            "win_rate": win_rate,
            "total_pnl": self._realized_pnl,
            "total_fees": self._total_fees_paid,
            "net_pnl_after_fees": self._realized_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "current_equity": self._current_equity,
            "starting_capital": self._total_starting_capital,
            "total_return": (self._current_equity - self._total_starting_capital) / self._total_starting_capital,
            "compound_factor": self._position_sizing_factor,
            "compound_growth": self._compound_growth
        }

    def _calculate_max_drawdown(self) -> float:
        if not self._equity_curve:
            return 0
        
        equity_values = np.array([p.total_equity for p in self._equity_curve])
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (running_max - equity_values) / running_max
        
        return np.max(drawdowns)

    def _calculate_sharpe_ratio(self) -> float:
        if len(self._equity_curve) < 2:
            return 0
        
        equity_values = np.array([p.total_equity for p in self._equity_curve])
        returns = np.diff(equity_values) / equity_values[:-1]
        
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        daily_returns = returns * (24 * 60 / 5)
        annualized_return = np.mean(daily_returns) * 365
        annualized_volatility = np.std(daily_returns) * np.sqrt(365)
        
        if annualized_volatility == 0:
            return 0
        
        return annualized_return / annualized_volatility

    def get_trades_by_strategy(self, strategy_name: str) -> List[TradeRecord]:
        """返回指定策略的 TradeRecord 列表，供 AllocationAgent 读取真实绩效。"""
        return [t for t in self._trades.values() if t.strategy_name == strategy_name]

    def get_strategy_stats(self, strategy_name: str) -> Dict[str, Any]:
        strategy_trades = [t for t in self._trades.values() if t.strategy_name == strategy_name]
        
        if not strategy_trades:
            return {
                "strategy_name": strategy_name,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0
            }
        
        wins = sum(1 for t in strategy_trades if t.win)
        losses = len(strategy_trades) - wins
        win_rate = wins / len(strategy_trades)
        
        total_pnl = sum(t.pnl_usdt for t in strategy_trades)
        avg_win = np.mean([t.pnl_usdt for t in strategy_trades if t.win]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl_usdt for t in strategy_trades if not t.win]) if losses > 0 else 0
        
        return {
            "strategy_name": strategy_name,
            "total_trades": len(strategy_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss
        }

    def get_daily_stats(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        
        if date_str in self._daily_stats:
            return self._daily_stats[date_str]
        
        start_time = datetime(date.year, date.month, date.day)
        end_time = start_time + timedelta(days=1)
        
        daily_trades = [t for t in self._trades.values() 
                       if start_time <= t.exit_time < end_time]
        
        if not daily_trades:
            return {
                "date": date_str,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "equity_change": 0
            }
        
        wins = sum(1 for t in daily_trades if t.win)
        losses = len(daily_trades) - wins
        win_rate = wins / len(daily_trades)
        total_pnl = sum(t.pnl_usdt for t in daily_trades)
        
        return {
            "date": date_str,
            "total_trades": len(daily_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl
        }

    def get_twr(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> float:
        """计算时间加权收益 TWR = ∏(1 + 自然日收益率) - 1。

        按自然日链式复合权益曲线日收益率，消除资金规模/存取款对收益率的扭曲，
        避免简单累计盈亏受本金规模变化影响。
        """
        if len(self._equity_curve) < 2:
            return 0.0

        points = [p for p in self._equity_curve if p.total_equity and p.total_equity > 0]
        if start_date:
            points = [p for p in points if p.timestamp >= start_date]
        if end_date:
            points = [p for p in points if p.timestamp <= end_date]
        if len(points) < 2:
            return 0.0

        # 按自然日取每日末权益（保留当日最后一点）
        daily_last: Dict[str, float] = {}
        for p in points:
            daily_last[p.timestamp.strftime("%Y-%m-%d")] = p.total_equity

        days = sorted(daily_last.keys())
        if len(days) < 2:
            return 0.0

        twr = 1.0
        prev_eq = daily_last[days[0]]
        for day in days[1:]:
            cur_eq = daily_last[day]
            if prev_eq > 0:
                twr *= (cur_eq / prev_eq)
            prev_eq = cur_eq
        return twr - 1.0

    def get_daily_aggregation(self, days: int = 30, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """自然日聚合：返回最近 days 个自然日的逐日绩效（盈亏/交易/胜率/权益变化）。

        按自然日（而非滚动窗口）聚合，供日报/复盘使用。
        """
        if end_date is None:
            end_date = datetime.now()
        end_day = datetime(end_date.year, end_date.month, end_date.day)
        start_day = end_day - timedelta(days=days - 1)

        # 按自然日统计首末权益（来自权益曲线）
        daily_equity: Dict[str, Dict[str, float]] = {}
        for p in self._equity_curve:
            day = p.timestamp.strftime("%Y-%m-%d")
            if day not in daily_equity:
                daily_equity[day] = {"first": p.total_equity, "last": p.total_equity}
            else:
                daily_equity[day]["last"] = p.total_equity

        result = []
        for i in range(days):
            day = start_day + timedelta(days=i)
            day_str = day.strftime("%Y-%m-%d")
            next_day = day + timedelta(days=1)

            daily_trades = [t for t in self._trades.values() if day <= t.exit_time < next_day]
            wins = sum(1 for t in daily_trades if t.win)
            losses = len(daily_trades) - wins
            total_pnl = sum(t.pnl_usdt for t in daily_trades)
            eq = daily_equity.get(day_str)

            result.append({
                "date": day_str,
                "total_trades": len(daily_trades),
                "wins": wins,
                "losses": losses,
                "win_rate": wins / len(daily_trades) if daily_trades else 0,
                "total_pnl": total_pnl,
                "equity_start": eq["first"] if eq else None,
                "equity_end": eq["last"] if eq else None,
                "equity_change": (eq["last"] - eq["first"]) if eq else 0,
            })
        return result

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        return [{
            "timestamp": point.timestamp.isoformat(),
            "total_equity": point.total_equity,
            "available_balance": point.available_balance,
            "used_margin": point.used_margin,
            "unrealized_pnl": point.unrealized_pnl,
            "realized_pnl": point.realized_pnl,
            "win_rate": point.win_rate,
            "total_trades": point.total_trades
        } for point in self._equity_curve]

    def _trade_to_dict(self, t: TradeRecord) -> Dict[str, Any]:
        return {
            "trade_id": t.trade_id,
            "symbol": t.symbol,
            "strategy_name": t.strategy_name,
            "direction": t.direction,
            "entry_price": t.entry_price,
            "exit_price": t.exit_price,
            "quantity": t.quantity,
            "leverage": t.leverage,
            "entry_time": t.entry_time.isoformat(),
            "exit_time": t.exit_time.isoformat(),
            "fees": t.fees,
            "slippage_cost": t.slippage_cost,
            "funding_cost": t.funding_cost,
            "spread_cost": t.spread_cost,
            "pnl_pct": t.pnl_pct,
            "pnl_usdt": t.pnl_usdt,
            "win": t.win,
            "exit_reason": t.exit_reason
        }

    def get_recent_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        sorted_trades = sorted(self._trades.values(), key=lambda t: t.exit_time, reverse=True)[:limit]
        return [self._trade_to_dict(t) for t in sorted_trades]

    def get_trades_since(self, since: datetime, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """返回平仓时间（exit_time）不早于 since 的交易记录（时间线过滤）。

        用于智能分析报告：只统计时间线之后的交易，之前的交易不计入。
        """
        filtered = [t for t in self._trades.values() if t.exit_time >= since]
        filtered.sort(key=lambda t: t.exit_time, reverse=True)
        if limit is not None:
            filtered = filtered[:limit]
        return [self._trade_to_dict(t) for t in filtered]

    def get_trade_stats_since(self, since: datetime) -> Dict[str, Any]:
        """基于时间线（exit_time >= since）之后的交易重算整体统计。

        与 get_trade_stats 的区别：交易维度（笔数/胜率/盈亏/盈亏比）按时间线过滤，
        max_drawdown/sharpe 基于过滤后交易的 PnL 序列近似；账户维度（当前权益、
        起始资金、总收益率）保持全局真实值不变。
        """
        trades = [t for t in self._trades.values() if t.exit_time >= since]
        trades.sort(key=lambda t: t.exit_time)

        total_trades = len(trades)
        wins = sum(1 for t in trades if t.win)
        losses = total_trades - wins
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        total_pnl = sum(t.pnl_usdt for t in trades)
        total_fees = sum(t.fees for t in trades)

        win_pnls = [t.pnl_usdt for t in trades if t.win]
        loss_pnls = [t.pnl_usdt for t in trades if not t.win]
        avg_win = np.mean(win_pnls) if win_pnls else 0.0
        avg_loss = np.mean(loss_pnls) if loss_pnls else 0.0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

        # 基于过滤后交易 PnL 序列计算回撤与夏普
        pnl_series = [t.pnl_usdt for t in trades]
        equity = np.cumsum(pnl_series) if pnl_series else np.array([])
        max_drawdown = 0.0
        sharpe_ratio = 0.0
        if len(equity) > 0:
            running_max = np.maximum.accumulate(equity)
            denom = np.where(running_max != 0, running_max, 1.0)
            drawdowns = (running_max - equity) / denom
            max_drawdown = float(np.max(drawdowns))
        if len(equity) > 1:
            returns = np.diff(equity)
            std = np.std(returns)
            if std > 0:
                sharpe_ratio = float(np.mean(returns) / std)

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "total_fees": total_fees,
            "net_pnl_after_fees": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "current_equity": self._current_equity,
            "starting_capital": self._total_starting_capital,
            "total_return": (self._current_equity - self._total_starting_capital) / self._total_starting_capital,
            "compound_factor": self._position_sizing_factor,
            "compound_growth": self._compound_growth
        }

    def get_strategy_stats_since(self, since: datetime, strategy_name: str) -> Dict[str, Any]:
        """基于时间线（exit_time >= since）之后的交易重算单策略统计。"""
        strategy_trades = [t for t in self._trades.values()
                           if t.strategy_name == strategy_name and t.exit_time >= since]

        if not strategy_trades:
            return {
                "strategy_name": strategy_name,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "avg_win": 0,
                "avg_loss": 0
            }

        wins = sum(1 for t in strategy_trades if t.win)
        losses = len(strategy_trades) - wins
        win_rate = wins / len(strategy_trades)

        total_pnl = sum(t.pnl_usdt for t in strategy_trades)
        avg_win = np.mean([t.pnl_usdt for t in strategy_trades if t.win]) if wins > 0 else 0
        avg_loss = np.mean([t.pnl_usdt for t in strategy_trades if not t.win]) if losses > 0 else 0

        return {
            "strategy_name": strategy_name,
            "total_trades": len(strategy_trades),
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss
        }

    async def persist_learning_state(self, key: str, value: Dict[str, Any]):
        import json
        
        conn = self.sqlite_storage.get_connection()
        
        try:
            value_str = json.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize learning state: {e}")
            return
        
        conn.execute(text('''
            INSERT OR REPLACE INTO learning_state (key, value, updated_at)
            VALUES (:key, :value, :updated_at)
        '''), {"key": key, "value": value_str, "updated_at": datetime.now().isoformat()})
        
        conn.commit()
        conn.close()

    async def load_learning_state(self, key: str) -> Optional[Dict[str, Any]]:
        import json
        
        conn = self.sqlite_storage.get_connection()
        
        result = conn.execute(text('SELECT value FROM learning_state WHERE key = :key'), {"key": key})
        row = result.fetchone()
        
        conn.close()
        
        if not row:
            return None
        
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            logger.error(f"Failed to deserialize learning state for key: {key}")
            return None