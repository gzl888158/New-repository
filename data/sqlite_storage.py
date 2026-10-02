"""
基于 SQLite 的持久化存储，负责交易、持仓、账户与风控等记录的落库与查询。
"""
import os
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import create_engine, Column, String, Float, DateTime, Integer, Boolean, text, Index, func, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from loguru import logger

from data.sharding import ShardRouter
from utils.helpers import safe_float, safe_int

Base = declarative_base()


class TradeRecord(Base):
    __tablename__ = "trade_records"
    id = Column(String, primary_key=True)
    symbol = Column(String, index=True)
    strategy_name = Column(String, index=True)
    side = Column(String)
    order_type = Column(String)
    signal_type = Column(String)  # 信号子类型：scalping=momentum/mean_reversion/breakout/range；trend=trend_entry等
    quantity = Column(Float)
    price = Column(Float)
    filled_price = Column(Float)
    leverage = Column(Integer)
    margin = Column(Float)
    # 注意口径：pnl 是 USDT 净额（不是百分比！），与 core/trade_journal.py 的
    # dataclass TradeRecord.pnl_pct（百分比收益率）语义不同，命名勿混淆。
    pnl = Column(Float)
    pnl_percent = Column(Float)  # 保证金收益率（百分比，= pnl/margin*100）
    fees = Column(Float, default=0.0)
    slippage_cost = Column(Float, default=0.0)
    funding_cost = Column(Float, default=0.0)
    spread_cost = Column(Float, default=0.0)
    status = Column(String, index=True)
    exit_reason = Column(String, default="manual")  # 止盈止损原因：stop_loss, take_profit, trailing_stop, manual
    create_time = Column(DateTime, index=True)
    close_time = Column(DateTime)
    trace_id = Column(String)  # 全链路 traceID：信号→裁决→订单→成交→记账 同源追踪

    __table_args__ = (
        Index('idx_trade_symbol_status', 'symbol', 'status'),
        Index('idx_trade_strategy_time', 'strategy_name', 'create_time'),
    )


# 订单分库分表：trade_records 分片表的列清单，与 TradeRecord ORM 字段一一对应。
# 归档/分片查询显式使用列名，避免 SELECT * 因历史 ALTER 迁移导致的列序漂移。
TRADE_COLS = [
    "id", "symbol", "strategy_name", "side", "order_type", "signal_type",
    "quantity", "price", "filled_price", "leverage", "margin", "pnl",
    "pnl_percent", "fees", "slippage_cost", "funding_cost", "spread_cost",
    "status", "exit_reason", "create_time", "close_time", "trace_id",
]


class PositionHistory(Base):
    __tablename__ = "position_history"
    id = Column(String, primary_key=True)
    symbol = Column(String, index=True)
    side = Column(String)
    quantity = Column(Float)
    avg_cost = Column(Float)
    mark_price = Column(Float)
    unrealized_pnl = Column(Float)
    margin = Column(Float)
    leverage = Column(Integer)
    timestamp = Column(DateTime, index=True)

    __table_args__ = (
        Index('idx_position_symbol_time', 'symbol', 'timestamp'),
    )


class AccountHistory(Base):
    __tablename__ = "account_history"
    id = Column(String, primary_key=True)
    total_equity = Column(Float)
    available_balance = Column(Float)
    used_margin = Column(Float)
    unrealized_pnl = Column(Float)
    margin_rate = Column(Float)
    timestamp = Column(DateTime, index=True)


class RiskEvent(Base):
    __tablename__ = "risk_events"
    id = Column(String, primary_key=True)
    event_type = Column(String, index=True)
    severity = Column(String)
    message = Column(String)
    symbol = Column(String)
    timestamp = Column(DateTime, index=True)


class AlertRecord(Base):
    __tablename__ = "alert_records"
    id = Column(String, primary_key=True)
    alert_type = Column(String, index=True)
    severity = Column(String, index=True)
    message = Column(String)
    symbol = Column(String)
    category = Column(String)
    rule_name = Column(String)
    metric = Column(String)
    value = Column(Float)
    threshold = Column(Float)
    state = Column(String)
    actions = Column(String)
    alert_metadata = Column(String)
    create_time = Column(DateTime, index=True)
    ack_time = Column(DateTime)
    resolve_time = Column(DateTime)

    __table_args__ = (
        Index('idx_alert_severity_time', 'severity', 'create_time'),
        Index('idx_alert_type_time', 'alert_type', 'create_time'),
    )


class RecoveryRecord(Base):
    __tablename__ = "recovery_records"
    id = Column(String, primary_key=True)
    failure_type = Column(String, index=True)
    action = Column(String)
    attempt = Column(Integer)
    success = Column(Boolean)
    error = Column(String)
    duration_ms = Column(Float)
    details = Column(String)
    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime)

    __table_args__ = (
        Index('idx_recovery_failure_time', 'failure_type', 'start_time'),
        Index('idx_recovery_success', 'success'),
    )


class StrategyPerformance(Base):
    __tablename__ = "strategy_performance"
    id = Column(String, primary_key=True)
    strategy_name = Column(String, index=True)
    symbol = Column(String, index=True)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    total_pnl = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    last_update = Column(DateTime, index=True)

    __table_args__ = (
        Index('idx_strategy_perf', 'strategy_name', 'symbol'),
    )


class PnLReconciliation(Base):
    """账户级盈亏对账快照：数据库累计已实现盈亏 vs OKX账户权益变化。

    用于持久化 discrepancy 及其分解，替代原先只记日志、不回写的实现，
    使资金费率/滑点/点差等未入账成本与未实现盈亏可被追溯。
    """
    __tablename__ = "pnl_reconciliation"
    id = Column(String, primary_key=True)
    initial_capital = Column(Float, default=0.0)
    okx_equity = Column(Float, default=0.0)
    okx_total_pnl = Column(Float, default=0.0)
    db_realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    discrepancy = Column(Float, default=0.0)
    unattributed = Column(Float, default=0.0)
    # 残差入账：unattributed 拆分为「资金费」与「滑点+点差+记录噪声」两部分。
    # funding_fee 从 OKX 资金费账单（type=7）精确入账（负=支付成本，正=收取收益）；
    # slippage_spread 为剩余残差（OKX 无独立滑点/点差账单，无法再细分）。
    funding_fee = Column(Float, default=0.0)
    slippage_spread = Column(Float, default=0.0)
    external_flow = Column(Float, default=0.0)  # 检测到的外部资金流动（出入金），非交易盈亏
    timestamp = Column(DateTime, index=True)

    __table_args__ = (
        Index('idx_pnl_reconcile_time', 'timestamp'),
    )


class SQLiteStorage:
    def __init__(self, config: Dict[str, Any]):
        db_path = config["sqlite"]["db_path"]
        self._db_path = db_path
        self._trade_router = ShardRouter("trade_records")
        dir_path = os.path.dirname(db_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        # SQLite 使用 SingletonThreadPool，不支持 pool_size/max_overflow 参数
        is_memory = db_path == ":memory:"
        if is_memory:
            # 内存数据库必须用 StaticPool 共享同一连接，否则跨 session 数据不可见
            self._engine = create_engine(
                "sqlite://",
                poolclass=StaticPool,
                connect_args={"check_same_thread": False}
            )
        else:
            self._engine = create_engine(
                f"sqlite:///{db_path}",
                pool_pre_ping=True,
                pool_recycle=3600,
                connect_args={"timeout": 30, "check_same_thread": False}
            )
        # 为每个新连接设置 WAL 模式 + busy_timeout，防止数据库锁定
        @event.listens_for(self._engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA cache_size=-10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()
        from sqlalchemy import text as sa_text
        with self._engine.connect() as conn:
            conn.execute(sa_text("PRAGMA journal_mode=WAL"))
            conn.execute(sa_text("PRAGMA synchronous=NORMAL"))
            conn.execute(sa_text("PRAGMA busy_timeout=30000"))
            conn.execute(sa_text("PRAGMA cache_size=-10000"))
            conn.execute(sa_text("PRAGMA temp_store=MEMORY"))
        Base.metadata.create_all(self._engine)
        self._migrate_schema()
        self._Session = sessionmaker(bind=self._engine)

    def _migrate_schema(self):
        """轻量级schema迁移：为已存在但缺少的列执行ALTER TABLE ADD COLUMN"""
        from sqlalchemy import text as sa_text
        # 活跃订单落盘表（fill 回执丢失根因修复方案 A）：
        # 持久化 pending 订单供重启恢复，根治「重启后 _active_orders 内存态清空导致回执永久丢失」。
        try:
            with self._engine.connect() as conn:
                conn.execute(sa_text(
                    "CREATE TABLE IF NOT EXISTS active_orders ("
                    "exchange_order_id TEXT PRIMARY KEY, "
                    "order_info TEXT NOT NULL, "
                    "updated_at TEXT"
                    ")"
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"active_orders table creation failed: {e}")
        expected_columns = {
            "trade_records": ["exit_reason", "fees", "signal_type", "slippage_cost", "funding_cost", "spread_cost", "trace_id"],
            "signals": [],
            "events": [],
            "alerts": [],
            "recovery_actions": [],
            "strategy_performance": [],
            "pnl_reconciliation": ["external_flow", "funding_fee", "slippage_spread"],
        }
        # 指定非字符串列的 SQL 类型，避免数值列被声明为 VARCHAR 导致类型亲和性问题
        column_types = {
            "fees": "REAL DEFAULT 0",
            "slippage_cost": "REAL DEFAULT 0",
            "funding_cost": "REAL DEFAULT 0",
            "spread_cost": "REAL DEFAULT 0",
            "external_flow": "REAL DEFAULT 0",
            "funding_fee": "REAL DEFAULT 0",
            "slippage_spread": "REAL DEFAULT 0",
        }
        try:
            with self._engine.connect() as conn:
                for table, columns in expected_columns.items():
                    try:
                        existing = conn.execute(
                            sa_text(f"PRAGMA table_info({table})")
                        ).fetchall()
                        existing_names = {row[1] for row in existing}
                    except Exception as e:
                        logger.debug(f"PRAGMA table_info({table}) failed: {e}")
                        continue
                    for col in columns:
                        if col not in existing_names:
                            try:
                                col_type = column_types.get(col, "VARCHAR")
                                conn.execute(
                                    sa_text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}")
                                )
                                logger.info(f"Schema migration: added column {col} to {table}")
                            except Exception as e:
                                logger.warning(f"Schema migration failed for {table}.{col}: {e}")
                conn.commit()
        except Exception as e:
            logger.warning(f"Schema migration check failed: {e}")

    def _normalize_trade_payload(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """企业级数据口径统一（单一事实来源）：
        1. pnl_percent 统一为保证金收益率 pnl/margin*100（仅 closed 且 margin>0 时覆盖，
           避免调用方口径漂移；历史事故：AVAX 单笔 pnl/margin=-12.45% 但库内 -7.02）。
        2. signal_type 空值兜底为 'unknown'（区分「未记录」与 NULL，便于复盘溯源）。
        """
        status = trade.get("status", "")
        pnl = trade.get("pnl")
        margin = trade.get("margin")

        if status == "closed" and margin:
            margin_val = safe_float(margin)
            pnl_val = safe_float(pnl)
            if margin_val > 0:
                trade["pnl_percent"] = (pnl_val / margin_val * 100) if pnl_val != 0 else 0.0

        if not trade.get("signal_type"):
            trade["signal_type"] = "unknown"

        return trade

    def save_trade_record(self, trade: Dict[str, Any]):
        trade = self._normalize_trade_payload(trade)
        session = self._Session()
        try:
            record = TradeRecord(
                id=trade.get("id") or str(uuid.uuid4()),
                symbol=trade.get("symbol", ""),
                strategy_name=trade.get("strategy_name", ""),
                side=trade.get("side", ""),
                order_type=trade.get("order_type", ""),
                signal_type=trade.get("signal_type", ""),
                quantity=trade.get("quantity", 0.0),
                price=trade.get("price", 0.0),
                filled_price=trade.get("filled_price", 0.0),
                leverage=trade.get("leverage", 1),
                margin=trade.get("margin", 0.0),
                pnl=trade.get("pnl", 0.0),
                pnl_percent=trade.get("pnl_percent", 0.0),
                fees=trade.get("fees", 0.0),
                slippage_cost=trade.get("slippage_cost", 0.0),
                funding_cost=trade.get("funding_cost", 0.0),
                spread_cost=trade.get("spread_cost", 0.0),
                status=trade.get("status", ""),
                exit_reason=trade.get("exit_reason", "manual"),
                create_time=trade.get("create_time", datetime.now()),
                close_time=trade.get("close_time"),
                trace_id=trade.get("trace_id", "")
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save trade record: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def save_trade_records_batch(self, trades: List[Dict[str, Any]]):
        if not trades:
            return True
        
        session = self._Session()
        try:
            records = []
            for trade in trades:
                trade = self._normalize_trade_payload(trade)
                record = TradeRecord(
                    id=trade.get("id") or str(uuid.uuid4()),
                    symbol=trade.get("symbol", ""),
                    strategy_name=trade.get("strategy_name", ""),
                    side=trade.get("side", ""),
                    order_type=trade.get("order_type", ""),
                    signal_type=trade.get("signal_type", ""),
                    quantity=trade.get("quantity", 0.0),
                    price=trade.get("price", 0.0),
                    filled_price=trade.get("filled_price", 0.0),
                    leverage=trade.get("leverage", 1),
                    margin=trade.get("margin", 0.0),
                    pnl=trade.get("pnl", 0.0),
                    pnl_percent=trade.get("pnl_percent", 0.0),
                    fees=trade.get("fees", 0.0),
                    slippage_cost=trade.get("slippage_cost", 0.0),
                    funding_cost=trade.get("funding_cost", 0.0),
                    spread_cost=trade.get("spread_cost", 0.0),
                    status=trade.get("status", ""),
                    exit_reason=trade.get("exit_reason", "manual"),
                    create_time=trade.get("create_time", datetime.now()),
                    close_time=trade.get("close_time"),
                    trace_id=trade.get("trace_id", "")
                )
                records.append(record)
            session.add_all(records)
            session.commit()
            logger.debug(f"Saved {len(records)} trade records in batch")
            return True
        except Exception as e:
            logger.error(f"Failed to save trade records batch: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def save_active_order(self, exchange_order_id: str, order_info: Dict[str, Any]) -> bool:
        """活跃订单落盘（fill 回执丢失根因修复方案 A）：持久化 pending 订单完整信息，
        供系统重启后恢复 `_active_orders`，根治「重启内存态清空导致成交回执永久丢失」。

        纯增量、幂等（INSERT OR REPLACE），不阻塞下单热路径（异常仅告警）。
        """
        try:
            payload = json.dumps(order_info, ensure_ascii=False, default=str)
            with self._engine.connect() as conn:
                conn.execute(text(
                    "INSERT OR REPLACE INTO active_orders "
                    "(exchange_order_id, order_info, updated_at) VALUES (:id, :info, :ts)"
                ), {"id": exchange_order_id, "info": payload, "ts": datetime.now().isoformat()})
                conn.commit()
            return True
        except Exception as e:
            logger.warning(f"save_active_order failed for {exchange_order_id}: {e}")
            return False

    def delete_active_order(self, exchange_order_id: str) -> bool:
        """删除已处理（成交/撤单/失败）的活跃订单落盘记录。"""
        try:
            with self._engine.connect() as conn:
                conn.execute(text(
                    "DELETE FROM active_orders WHERE exchange_order_id = :id"
                ), {"id": exchange_order_id})
                conn.commit()
            return True
        except Exception as e:
            logger.warning(f"delete_active_order failed for {exchange_order_id}: {e}")
            return False

    def load_active_orders(self) -> Dict[str, Dict[str, Any]]:
        """加载全部落盘的活跃订单（用于重启恢复 `_active_orders`）。"""
        result: Dict[str, Dict[str, Any]] = {}
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT exchange_order_id, order_info FROM active_orders"
                )).fetchall()
            for exchange_order_id, order_info_json in rows:
                try:
                    result[exchange_order_id] = json.loads(order_info_json)
                except Exception as e:
                    logger.warning(f"load_active_orders: skip corrupt row {exchange_order_id}: {e}")
        except Exception as e:
            logger.warning(f"load_active_orders failed: {e}")
        return result

    def update_trade_record(self, trade_id: str, updates: Dict[str, Any]):
        session = self._Session()
        try:
            record = session.query(TradeRecord).filter(TradeRecord.id == trade_id).first()
            if record:
                # 企业级口径统一：closed 记录若本次更新含 pnl，则用保证金收益率重算 pnl_percent
                # （update 的 updates 通常不含 margin，需回读 record.margin）
                status_after = updates.get("status", record.status)
                if status_after == "closed" and "pnl" in updates:
                    margin = record.margin
                    if margin:
                        margin_val = safe_float(margin)
                        pnl_val = safe_float(updates.get("pnl") or 0.0)
                        if margin_val > 0:
                            updates["pnl_percent"] = (pnl_val / margin_val * 100) if pnl_val != 0 else 0.0
                # signal_type 空值兜底
                if not updates.get("signal_type") and not record.signal_type:
                    updates["signal_type"] = "unknown"

                for key, value in updates.items():
                    if hasattr(record, key):
                        if key in ("create_time", "close_time") and isinstance(value, str):
                            try:
                                from datetime import datetime as _dt
                                value = _dt.fromisoformat(value)
                            except (ValueError, TypeError):
                                value = _dt.now()
                        setattr(record, key, value)
                session.commit()
                return True
            else:
                logger.warning(f"update_trade_record: trade_id={trade_id} not found")
                return False
        except Exception as e:
            logger.error(f"Failed to update trade record: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_trade_records_checked(self, symbol: str = None, strategy_name: str = None,
                                  limit: int = 100) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """带错误状态的交易记录查询：返回 (records, error)。

        error 为 None 表示查询成功（结果可为空列表，即「确实无数据」）；
        error 非空表示查询失败（区别于空数据，供上层 fail-closed 感知）。
        """
        session = self._Session()
        try:
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            if strategy_name:
                query = query.filter(TradeRecord.strategy_name == strategy_name)
            records = query.order_by(TradeRecord.create_time.desc()).limit(limit).all()
            return [self._trade_to_dict(r) for r in records], None
        except Exception as e:
            logger.error(f"Failed to get trade records: {e}")
            return [], str(e)
        finally:
            session.close()

    def get_trade_records(self, symbol: str = None, strategy_name: str = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """查询交易记录（向后兼容：异常时返回空列表，调用方无法区分失败与空数据，
        需要区分时改用 get_trade_records_checked）。"""
        records, _ = self.get_trade_records_checked(symbol=symbol, strategy_name=strategy_name, limit=limit)
        return records

    def get_trade_records_by_status(self, status: str, limit: int = 200) -> List[Dict[str, Any]]:
        session = self._Session()
        try:
            records = session.query(TradeRecord).filter(
                TradeRecord.status == status
            ).order_by(TradeRecord.create_time.desc()).limit(limit).all()
            return [self._trade_to_dict(r) for r in records]
        except Exception as e:
            logger.error(f"Failed to get trade records by status: {e}")
            return []
        finally:
            session.close()

    def get_latest_open_record(self, symbol: str, strategy_name: str = None) -> Optional[Dict[str, Any]]:
        session = self._Session()
        try:
            query = session.query(TradeRecord).filter(
                TradeRecord.status == "open",
                TradeRecord.symbol == symbol
            )
            if strategy_name:
                query = query.filter(TradeRecord.strategy_name == strategy_name)
            record = query.order_by(TradeRecord.create_time.desc()).first()
            return self._trade_to_dict(record) if record else None
        except Exception as e:
            logger.error(f"Failed to get latest open record for {symbol}: {e}")
            return None
        finally:
            session.close()

    def get_all_open_records(self, symbol: str = None) -> List[Dict[str, Any]]:
        session = self._Session()
        try:
            query = session.query(TradeRecord).filter(TradeRecord.status == "open")
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
            records = query.order_by(TradeRecord.create_time.desc()).all()
            return [self._trade_to_dict(r) for r in records]
        except Exception as e:
            logger.error(f"Failed to get all open records: {e}")
            return []
        finally:
            session.close()

    def close_open_record(self, symbol: str, strategy_name: str = None,
                          exit_reason: str = "ghost_close") -> bool:
        """关闭指定 symbol（可选 strategy）的未平仓记录。

        用于订单执行失败 / 幽灵持仓时同步数据库状态，防止 trade_records 与
        实际持仓漂移（历史事故：cleanup_position 调用本方法但方法缺失，异常被静默吞掉）。
        """
        session = self._Session()
        try:
            query = session.query(TradeRecord).filter(
                TradeRecord.status == "open",
                TradeRecord.symbol == symbol
            )
            if strategy_name:
                query = query.filter(TradeRecord.strategy_name == strategy_name)
            records = query.all()
            now = datetime.now()
            for record in records:
                record.status = "closed"
                record.exit_reason = exit_reason
                record.close_time = now
            session.commit()
            return len(records) > 0
        except Exception as e:
            logger.error(f"Failed to close open record for {symbol}: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_latest_position_history(self, symbol: str, side: str = None) -> Optional[Dict[str, Any]]:
        """查询指定 symbol（可选 side）最近一条持仓快照，用于幽灵持仓对账时回填 unrealized_pnl。"""
        session = self._Session()
        try:
            query = session.query(PositionHistory).filter(PositionHistory.symbol == symbol)
            if side:
                query = query.filter(PositionHistory.side == side)
            record = query.order_by(PositionHistory.timestamp.desc()).first()
            if not record:
                return None
            return {
                "symbol": record.symbol,
                "side": record.side,
                "quantity": record.quantity,
                "avg_cost": record.avg_cost,
                "mark_price": record.mark_price,
                "unrealized_pnl": record.unrealized_pnl,
                "margin": record.margin,
                "leverage": record.leverage,
                "timestamp": record.timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to get latest position history for {symbol}: {e}")
            return None
        finally:
            session.close()

    def get_trade_summary(self, start_time: datetime = None, end_time: datetime = None) -> Dict[str, Any]:
        session = self._Session()
        try:
            query = session.query(TradeRecord).filter(TradeRecord.status == "closed")
            if start_time:
                query = query.filter(TradeRecord.create_time >= start_time)
            if end_time:
                query = query.filter(TradeRecord.create_time <= end_time)
            
            records = query.all()
            
            total_trades = len(records)
            winning_trades = sum(1 for r in records if (r.pnl or 0) > 0)
            losing_trades = sum(1 for r in records if (r.pnl or 0) <= 0)
            total_pnl = sum(r.pnl or 0 for r in records)
            
            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
                "total_pnl": total_pnl,
                "average_pnl": total_pnl / total_trades if total_trades > 0 else 0,
            }
        except Exception as e:
            logger.error(f"Failed to get trade summary: {e}")
            return {}
        finally:
            session.close()

    def get_strategy_pnl_summary(self) -> Dict[str, float]:
        session = self._Session()
        try:
            result = session.query(
                TradeRecord.strategy_name,
                func.sum(TradeRecord.pnl).label('total_pnl')
            ).filter(TradeRecord.status == "closed").group_by(TradeRecord.strategy_name).all()
            
            return {row.strategy_name: (row.total_pnl or 0.0) for row in result}
        except Exception as e:
            logger.error(f"Failed to get strategy PnL summary: {e}")
            return {}
        finally:
            session.close()

    def _trade_to_dict(self, record: TradeRecord) -> Dict[str, Any]:
        return {
            "id": record.id,
            "symbol": record.symbol,
            "strategy_name": record.strategy_name,
            "side": record.side,
            "order_type": record.order_type,
            "signal_type": record.signal_type,
            "quantity": record.quantity,
            "price": record.price,
            "filled_price": record.filled_price,
            "leverage": record.leverage,
            "margin": record.margin,
            "pnl": record.pnl,
            "pnl_percent": record.pnl_percent,
            "fees": record.fees,
            "slippage_cost": record.slippage_cost,
            "funding_cost": record.funding_cost,
            "spread_cost": record.spread_cost,
            "status": record.status,
            "exit_reason": record.exit_reason,
            "create_time": record.create_time,
            "close_time": record.close_time,
            "trace_id": record.trace_id
        }

    def save_position_history(self, position: Dict[str, Any]):
        session = self._Session()
        try:
            record = PositionHistory(
                id=position.get("id") or str(uuid.uuid4()),
                symbol=position.get("symbol", ""),
                side=position.get("side", ""),
                quantity=position.get("quantity", 0.0),
                avg_cost=position.get("avg_cost", 0.0),
                mark_price=position.get("mark_price", 0.0),
                unrealized_pnl=position.get("unrealized_pnl", 0.0),
                margin=position.get("margin", 0.0),
                leverage=position.get("leverage", 1),
                timestamp=position.get("timestamp", datetime.now())
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save position history: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def save_position_history_batch(self, positions: List[Dict[str, Any]]):
        if not positions:
            return True
        
        session = self._Session()
        try:
            records = []
            for position in positions:
                record = PositionHistory(
                    id=position.get("id") or str(uuid.uuid4()),
                    symbol=position.get("symbol", ""),
                    side=position.get("side", ""),
                    quantity=position.get("quantity", 0.0),
                    avg_cost=position.get("avg_cost", 0.0),
                    mark_price=position.get("mark_price", 0.0),
                    unrealized_pnl=position.get("unrealized_pnl", 0.0),
                    margin=position.get("margin", 0.0),
                    leverage=position.get("leverage", 1),
                    timestamp=position.get("timestamp", datetime.now())
                )
                records.append(record)
            session.add_all(records)
            session.commit()
            logger.debug(f"Saved {len(records)} position history records in batch")
            return True
        except Exception as e:
            logger.error(f"Failed to save position history batch: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def save_account_history(self, account: Dict[str, Any]):
        session = self._Session()
        try:
            record = AccountHistory(
                id=account.get("id") or str(uuid.uuid4()),
                total_equity=account.get("total_equity", 0.0),
                available_balance=account.get("available_balance", 0.0),
                used_margin=account.get("used_margin", 0.0),
                unrealized_pnl=account.get("unrealized_pnl", 0.0),
                margin_rate=account.get("margin_rate", 0.0),
                timestamp=account.get("timestamp", datetime.now())
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save account history: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def save_pnl_reconciliation(self, recon: Dict[str, Any]):
        """持久化账户级盈亏对账快照（discrepancy 分解）。"""
        session = self._Session()
        try:
            record = PnLReconciliation(
                id=recon.get("id", str(uuid.uuid4())),
                initial_capital=recon.get("initial_capital", 0.0),
                okx_equity=recon.get("okx_equity", 0.0),
                okx_total_pnl=recon.get("okx_total_pnl", 0.0),
                db_realized_pnl=recon.get("db_realized_pnl", 0.0),
                unrealized_pnl=recon.get("unrealized_pnl", 0.0),
                discrepancy=recon.get("discrepancy", 0.0),
                unattributed=recon.get("unattributed", 0.0),
                funding_fee=recon.get("funding_fee", 0.0),
                slippage_spread=recon.get("slippage_spread", 0.0),
                external_flow=recon.get("external_flow", 0.0),
                timestamp=recon.get("timestamp", datetime.now())
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save PnL reconciliation: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_latest_pnl_reconciliation(self) -> Optional[Dict[str, Any]]:
        """查询最近一条账户级盈亏对账快照。"""
        session = self._Session()
        try:
            record = session.query(PnLReconciliation).order_by(
                PnLReconciliation.timestamp.desc()
            ).first()
            if not record:
                return None
            return {
                "initial_capital": record.initial_capital,
                "okx_equity": record.okx_equity,
                "okx_total_pnl": record.okx_total_pnl,
                "db_realized_pnl": record.db_realized_pnl,
                "unrealized_pnl": record.unrealized_pnl,
                "discrepancy": record.discrepancy,
                "unattributed": record.unattributed,
                "funding_fee": record.funding_fee,
                "slippage_spread": record.slippage_spread,
                "external_flow": record.external_flow,
                "timestamp": record.timestamp,
            }
        except Exception as e:
            logger.error(f"Failed to get latest PnL reconciliation: {e}")
            return None
        finally:
            session.close()

    def save_risk_event(self, event: Dict[str, Any]):
        session = self._Session()
        try:
            record = RiskEvent(
                id=event.get("id") or str(uuid.uuid4()),
                event_type=event.get("event_type", ""),
                severity=event.get("severity", "INFO"),
                message=event.get("message", ""),
                symbol=event.get("symbol", ""),
                timestamp=event.get("timestamp", datetime.now())
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save risk event: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def update_strategy_performance(self, strategy_name: str, symbol: str, 
                                   performance: Dict[str, Any]):
        session = self._Session()
        try:
            record = session.query(StrategyPerformance).filter(
                StrategyPerformance.strategy_name == strategy_name,
                StrategyPerformance.symbol == symbol
            ).first()
            
            if record:
                record.total_trades = performance.get("total_trades", record.total_trades)
                record.winning_trades = performance.get("winning_trades", record.winning_trades)
                record.losing_trades = performance.get("losing_trades", record.losing_trades)
                record.total_pnl = performance.get("total_pnl", record.total_pnl)
                record.max_drawdown = performance.get("max_drawdown", record.max_drawdown)
                record.win_rate = performance.get("win_rate", record.win_rate)
                record.profit_factor = performance.get("profit_factor", record.profit_factor)
                record.last_update = datetime.now()
            else:
                record = StrategyPerformance(
                    id=f"{strategy_name}:{symbol}",
                    strategy_name=strategy_name,
                    symbol=symbol,
                    **performance,
                    last_update=datetime.now()
                )
                session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update strategy performance: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_strategy_performance(self, strategy_name: str = None, 
                                symbol: str = None) -> List[Dict[str, Any]]:
        session = self._Session()
        try:
            query = session.query(StrategyPerformance)
            if strategy_name:
                query = query.filter(StrategyPerformance.strategy_name == strategy_name)
            if symbol:
                query = query.filter(StrategyPerformance.symbol == symbol)
            records = query.all()
            return [self._performance_to_dict(r) for r in records]
        except Exception as e:
            logger.error(f"Failed to get strategy performance: {e}")
            return []
        finally:
            session.close()

    def _performance_to_dict(self, record: StrategyPerformance) -> Dict[str, Any]:
        return {
            "strategy_name": record.strategy_name,
            "symbol": record.symbol,
            "total_trades": record.total_trades,
            "winning_trades": record.winning_trades,
            "losing_trades": record.losing_trades,
            "total_pnl": record.total_pnl,
            "max_drawdown": record.max_drawdown,
            "win_rate": record.win_rate,
            "profit_factor": record.profit_factor,
            "last_update": record.last_update
        }

    async def cleanup_old_data(self, days_to_keep: int = 30, max_retries: int = 3) -> Dict[str, Any]:
        """清理旧数据，带异步重试机制防止数据库锁定。

        返回 {"success": bool, "deleted_positions": int, "deleted_account": int}，
        fail-closed：任何重试失败都不静默吞掉，成功路径必须明确返回。
        """
        import asyncio
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for attempt in range(max_retries):
            session = self._Session()
            try:
                # 设置短超时避免长时间等待
                session.execute(text("PRAGMA busy_timeout = 3000"))
                
                deleted_positions = session.query(PositionHistory).filter(
                    PositionHistory.timestamp < cutoff_date
                ).delete(synchronize_session=False)
                
                deleted_account = session.query(AccountHistory).filter(
                    AccountHistory.timestamp < cutoff_date
                ).delete(synchronize_session=False)
                
                session.commit()
                
                if deleted_positions > 0:
                    logger.info(f"Cleaned up {deleted_positions} old position history records")
                if deleted_account > 0:
                    logger.info(f"Cleaned up {deleted_account} old account history records")
                return {"success": True, "deleted_positions": deleted_positions,
                        "deleted_account": deleted_account}
            except Exception as e:
                session.rollback()
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    logger.warning(f"Database locked on cleanup attempt {attempt + 1}/{max_retries}, retrying...")
                    await asyncio.sleep(1.0 * (attempt + 1))  # P3修复：异步退避，避免阻塞事件循环
                else:
                    logger.error(f"Failed to cleanup old data: {e}")
                    return {"success": False, "deleted_positions": 0,
                            "deleted_account": 0}
            finally:
                session.close()
        return {"success": False, "deleted_positions": 0, "deleted_account": 0}

    def get_connection(self):
        return self._engine.connect()

    def get_session(self):
        return self._Session()

    def health_check(self) -> bool:
        try:
            session = self._Session()
            session.execute(text("SELECT 1"))
            session.close()
            return True
        except Exception as e:
            logger.error(f"SQLite health check failed: {e}")
            return False

    # ==================== WAL 预写日志 ====================

    def get_wal_status(self) -> Dict[str, Any]:
        """读取 WAL 状态：journal_mode 是否为 wal + -wal 文件大小（用于健康上报）。"""
        status = {"journal_mode": None, "wal_enabled": False, "wal_file_bytes": 0}
        try:
            with self._engine.connect() as conn:
                mode = conn.execute(text("PRAGMA journal_mode")).scalar()
                status["journal_mode"] = mode
                status["wal_enabled"] = (str(mode).lower() == "wal")
        except Exception as e:
            status["error"] = str(e)
        if self._db_path and self._db_path != ":memory:":
            wal_path = self._db_path + "-wal"
            if os.path.exists(wal_path):
                try:
                    status["wal_file_bytes"] = os.path.getsize(wal_path)
                except OSError:
                    pass
        return status

    def wal_checkpoint(self, mode: str = "TRUNCATE") -> List[Any]:
        """执行 WAL checkpoint，把预写日志合并回主库并回收磁盘空间。

        mode: PASSIVE / FULL / RESTART / TRUNCATE（默认 TRUNCATE 截断 -wal 文件）。
        """
        result: List[Any] = []
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(f"PRAGMA wal_checkpoint({mode})")).fetchall()
                result = [tuple(r) for r in rows]
        except Exception as e:
            logger.warning(f"WAL checkpoint failed: {e}")
        return result

    # ==================== 订单冷热分库分表 ====================

    def _ensure_trade_shard(self, conn, table_name: str) -> None:
        """惰性创建月份分片表（schema 与 trade_records 一致 + 常用索引）。"""
        ddl = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            symbol TEXT NOT NULL,
            strategy_name TEXT,
            side TEXT,
            order_type TEXT,
            signal_type TEXT,
            quantity REAL,
            price REAL,
            filled_price REAL,
            leverage INTEGER,
            margin REAL,
            pnl REAL,
            pnl_percent REAL,
            fees REAL DEFAULT 0,
            slippage_cost REAL DEFAULT 0,
            funding_cost REAL DEFAULT 0,
            spread_cost REAL DEFAULT 0,
            status TEXT,
            exit_reason TEXT,
            create_time DATETIME,
            close_time DATETIME,
            trace_id TEXT
        )
        """
        conn.execute(text(ddl))
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_symbol ON {table_name}(symbol)"
        ))
        conn.execute(text(
            f"CREATE INDEX IF NOT EXISTS idx_{table_name}_close_time ON {table_name}(close_time)"
        ))

    def archive_closed_trades(self, before_days: int = 60) -> Dict[str, Any]:
        """订单冷热分表：将热表 trade_records 中已平仓且早于 before_days 的记录
        按月迁入分片表 trade_records_YYYYMM（先插后删，同一事务，杜绝数据丢失）。"""
        cutoff = datetime.now() - timedelta(days=before_days)
        col_csv = ", ".join(TRADE_COLS)
        archived_total = 0
        migrated_months: List[str] = []

        with self._engine.begin() as conn:
            rows = conn.execute(text(
                "SELECT DISTINCT strftime('%Y%m', close_time) AS ym FROM trade_records "
                "WHERE status='closed' AND close_time IS NOT NULL AND close_time < :cutoff"
            ), {"cutoff": cutoff}).fetchall()
            months = [r[0] for r in rows if r[0]]

            for ym in months:
                shard = f"trade_records_{ym}"
                self._ensure_trade_shard(conn, shard)
                moved = conn.execute(text(
                    f"INSERT OR IGNORE INTO {shard} ({col_csv}) "
                    f"SELECT {col_csv} FROM trade_records WHERE status='closed' "
                    f"AND close_time IS NOT NULL AND close_time < :cutoff "
                    f"AND strftime('%Y%m', close_time) = :ym"
                ), {"cutoff": cutoff, "ym": ym}).rowcount
                conn.execute(text(
                    f"DELETE FROM trade_records WHERE status='closed' "
                    f"AND close_time IS NOT NULL AND close_time < :cutoff "
                    f"AND strftime('%Y%m', close_time) = :ym"
                ), {"cutoff": cutoff, "ym": ym})
                archived_total += moved
                migrated_months.append(ym)

        if archived_total:
            logger.info(f"Order sharding: archived {archived_total} closed trades "
                        f"into months {migrated_months}")
        return {"archived": archived_total, "months": migrated_months}

    def _iter_trade_shards(self, conn) -> List[str]:
        return self._trade_router.iter_shards(conn)

    def get_db_realized_pnl(self) -> float:
        """跨「热表 + 全部月份分片」汇总已实现盈亏（closed 且 pnl 非空）。"""
        total = 0.0
        try:
            with self._engine.connect() as conn:
                for shard in self._iter_trade_shards(conn):
                    val = conn.execute(text(
                        f"SELECT COALESCE(SUM(pnl), 0) FROM {shard} "
                        f"WHERE status='closed' AND pnl IS NOT NULL"
                    )).scalar()
                    total += safe_float(val)
        except Exception as e:
            logger.error(f"get_db_realized_pnl failed: {e}")
        return total

    def get_authoritative_realized_pnl(self) -> float:
        """以 TradeJournal 权威账本 `trades` 表为准的已实现盈亏（USDT 净额）。

        口径差异：`get_db_realized_pnl` 汇总 trade_records（持仓账本）的 pnl，
        但 trade_records 长期被 ghost_close 记录污染（约 93% closed 记录 pnl=NULL/0），
        导致 realized pnl 被严重低估。`trades.pnl_usdt` 是 journal 在真实平仓时写入的
        权威净额（含手续费/滑点/资金费），是 PnL 对账更准确的基准。
        当 trades 为空时回退到 trade_records 口径，避免首次运行对账得到 0。
        """
        try:
            with self._engine.connect() as conn:
                val = conn.execute(text(
                    "SELECT COALESCE(SUM(pnl_usdt), 0) FROM trades"
                )).scalar()
                authoritative = safe_float(val)
            if authoritative != 0.0:
                return authoritative
        except Exception as e:
            logger.warning(f"get_authoritative_realized_pnl fallback to trade_records: {e}")
        return self.get_db_realized_pnl()

    def get_recent_strategy_pnl_authoritative(self, hours: int = 24) -> Dict[str, float]:
        """按策略汇总最近 N 小时的权威已实现盈亏（trades.pnl_usdt 口径）。

        与 get_authoritative_realized_pnl 一致，以 TradeJournal `trades` 表为准，
        避免用 trade_records.pnl（长期被 ghost_close 污染 pnl=NULL/0 导致低估）
        计算 capital_efficiency 时口径失真。当 trades 表无窗口内数据时返回空 dict，
        交由调用方回退到 trade_records 口径。
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        result: Dict[str, float] = {}
        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(
                    "SELECT COALESCE(strategy_name, 'unknown') AS strategy, "
                    "COALESCE(SUM(pnl_usdt), 0) AS pnl "
                    "FROM trades WHERE exit_time >= :cutoff "
                    "GROUP BY strategy_name"
                ), {"cutoff": cutoff}).fetchall()
                for row in rows:
                    result[str(row[0])] = safe_float(row[1])
        except Exception as e:
            logger.warning(f"get_recent_strategy_pnl_authoritative failed: {e}")
        return result

    @staticmethod
    def _parse_dt(value):
        """把分片查询返回的 DATETIME 字符串/整数转回 datetime（供对账比较）。"""
        if value is None or value == "":
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value)
        try:
            return datetime.fromisoformat(str(value))
        except (ValueError, TypeError):
            return None

    def get_closed_records_missing_pnl(self, limit: int = 500) -> List[Dict[str, Any]]:
        """跨分片获取 pnl 缺失（null/0）的已平仓记录，供 PnL 对账回填。

        每条记录附带 `_shard` 指明所在分片（含热表），便于对账后原地回写。
        """
        col_csv = ", ".join(TRADE_COLS)
        records: List[Dict[str, Any]] = []
        try:
            with self._engine.connect() as conn:
                for shard in self._iter_trade_shards(conn):
                    rows = conn.execute(text(
                        f"SELECT {col_csv} FROM {shard} WHERE status='closed' "
                        f"AND (pnl IS NULL OR pnl = 0) "
                        f"ORDER BY close_time DESC LIMIT :limit"
                    ), {"limit": limit}).fetchall()
                    for row in rows:
                        rec = dict(zip(TRADE_COLS, row))
                        rec["_shard"] = shard
                        rec["create_time"] = self._parse_dt(rec.get("create_time"))
                        rec["close_time"] = self._parse_dt(rec.get("close_time"))
                        records.append(rec)
                    if len(records) >= limit:
                        break
        except Exception as e:
            logger.error(f"get_closed_records_missing_pnl failed: {e}")
        return records[:limit]

    def update_trade_in_shard(self, shard: str, trade_id: str, updates: Dict[str, Any]) -> bool:
        """对指定分片（或热表）中的订单记录执行白名单列更新（供对账原地回写）。

        镜像 ORM update_trade_record 的企业口径：closed 记录更新 pnl 时按保证金
        重算 pnl_percent，避免热/冷两套口径漂移。
        """
        allowed = {
            "pnl", "pnl_percent", "filled_price", "fees",
            "slippage_cost", "funding_cost", "spread_cost",
            "status", "exit_reason", "close_time", "margin",
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}
        if not filtered:
            return False
        try:
            with self._engine.begin() as conn:
                if "pnl" in filtered:
                    # 优先用本次回填的 margin，否则回读 DB 旧值；避免 margin=0 导致 pnl_percent 漏算
                    new_margin = filtered.get("margin")
                    if new_margin is None:
                        row = conn.execute(text(
                            f"SELECT margin, status FROM {shard} WHERE id = :tid"
                        ), {"tid": trade_id}).fetchone()
                        if row and row[1] == "closed":
                            new_margin = row[0]
                    margin = safe_float(new_margin)
                    pnl_val = safe_float(filtered.get("pnl"))
                    if margin > 0:
                        filtered["pnl_percent"] = (pnl_val / margin * 100) if pnl_val != 0 else 0.0
                assignments = ", ".join(f"{k} = :{k}" for k in filtered)
                params = dict(filtered)
                params["trade_id"] = trade_id
                result = conn.execute(text(
                    f"UPDATE {shard} SET {assignments} WHERE id = :trade_id"
                ), params)
                return result.rowcount > 0
        except Exception as e:
            logger.error(f"update_trade_in_shard({shard}) failed: {e}")
            return False

    def save_alert_record(self, alert: Dict[str, Any]):
        import json as _json
        session = self._Session()
        try:
            record = AlertRecord(
                id=alert.get("id") or str(uuid.uuid4()),
                alert_type=alert.get("alert_type", ""),
                severity=alert.get("severity", "INFO"),
                message=alert.get("message", ""),
                symbol=alert.get("symbol", ""),
                category=alert.get("category", ""),
                rule_name=alert.get("rule_name", ""),
                metric=alert.get("metric", ""),
                value=alert.get("value", 0.0),
                threshold=alert.get("threshold", 0.0),
                state=alert.get("state", "triggered"),
                actions=_json.dumps(alert.get("actions", []), ensure_ascii=False),
                alert_metadata=_json.dumps(alert.get("metadata", {}), ensure_ascii=False),
                create_time=alert.get("create_time", datetime.now()),
                ack_time=alert.get("ack_time"),
                resolve_time=alert.get("resolve_time")
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save alert record: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def save_alert_records_batch(self, alerts: List[Dict[str, Any]]):
        import json as _json
        if not alerts:
            return True
        
        session = self._Session()
        try:
            records = []
            for alert in alerts:
                record = AlertRecord(
                    id=alert.get("id") or str(uuid.uuid4()),
                    alert_type=alert.get("alert_type", ""),
                    severity=alert.get("severity", "INFO"),
                    message=alert.get("message", ""),
                    symbol=alert.get("symbol", ""),
                    category=alert.get("category", ""),
                    rule_name=alert.get("rule_name", ""),
                    metric=alert.get("metric", ""),
                    value=alert.get("value", 0.0),
                    threshold=alert.get("threshold", 0.0),
                    state=alert.get("state", "triggered"),
                    actions=_json.dumps(alert.get("actions", []), ensure_ascii=False),
                    alert_metadata=_json.dumps(alert.get("metadata", {}), ensure_ascii=False),
                    create_time=alert.get("create_time", datetime.now()),
                    ack_time=alert.get("ack_time"),
                    resolve_time=alert.get("resolve_time")
                )
                records.append(record)
            session.add_all(records)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save alert records batch: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_alert_records(self, severity: str = None, alert_type: str = None,
                          symbol: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        import json as _json
        session = self._Session()
        try:
            query = session.query(AlertRecord)
            if severity:
                query = query.filter(AlertRecord.severity == severity.upper())
            if alert_type:
                query = query.filter(AlertRecord.alert_type == alert_type)
            if symbol:
                query = query.filter(AlertRecord.symbol == symbol)
            records = query.order_by(AlertRecord.create_time.desc()).limit(limit).all()
            
            result = []
            for r in records:
                try:
                    actions = _json.loads(r.actions) if r.actions else []
                except Exception as e:
                    logger.debug(f"Failed to parse alert actions for {r.id}: {e}")
                    actions = []
                try:
                    metadata = _json.loads(r.alert_metadata) if r.alert_metadata else {}
                except Exception as e:
                    logger.debug(f"Failed to parse alert metadata for {r.id}: {e}")
                    metadata = {}
                
                result.append({
                    "id": r.id,
                    "alert_type": r.alert_type,
                    "severity": r.severity,
                    "message": r.message,
                    "symbol": r.symbol,
                    "category": r.category,
                    "rule_name": r.rule_name,
                    "metric": r.metric,
                    "value": r.value,
                    "threshold": r.threshold,
                    "state": r.state,
                    "actions": actions,
                    "metadata": metadata,
                    "create_time": r.create_time,
                    "ack_time": r.ack_time,
                    "resolve_time": r.resolve_time
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get alert records: {e}")
            return []
        finally:
            session.close()

    def get_alert_count(self, severity: str = None) -> int:
        session = self._Session()
        try:
            query = session.query(func.count(AlertRecord.id))
            if severity:
                query = query.filter(AlertRecord.severity == severity.upper())
            return query.scalar() or 0
        except Exception as e:
            logger.error(f"Failed to get alert count: {e}")
            return 0
        finally:
            session.close()

    def update_alert_state(self, alert_id: str, state: str, ack_time=None, resolve_time=None):
        session = self._Session()
        try:
            record = session.query(AlertRecord).filter(AlertRecord.id == alert_id).first()
            if record:
                record.state = state
                if ack_time:
                    record.ack_time = ack_time
                if resolve_time:
                    record.resolve_time = resolve_time
                session.commit()
                return True
            else:
                logger.warning(f"update_alert_state: alert_id={alert_id} not found")
                return False
        except Exception as e:
            logger.error(f"Failed to update alert state: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def save_recovery_record(self, recovery: Dict[str, Any]):
        import json as _json
        session = self._Session()
        try:
            record = RecoveryRecord(
                id=recovery.get("id") or str(uuid.uuid4()),
                failure_type=recovery.get("failure_type", ""),
                action=recovery.get("action", ""),
                attempt=recovery.get("attempt", 1),
                success=recovery.get("success", False),
                error=recovery.get("error", ""),
                duration_ms=recovery.get("duration_ms", 0.0),
                details=_json.dumps(recovery.get("details", {}), ensure_ascii=False),
                start_time=recovery.get("start_time", datetime.now()),
                end_time=recovery.get("end_time")
            )
            session.add(record)
            session.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to save recovery record: {e}")
            session.rollback()
            return False
        finally:
            session.close()

    def get_recovery_records(self, failure_type: str = None, success: bool = None,
                            limit: int = 100) -> List[Dict[str, Any]]:
        import json as _json
        session = self._Session()
        try:
            query = session.query(RecoveryRecord)
            if failure_type:
                query = query.filter(RecoveryRecord.failure_type == failure_type)
            if success is not None:
                query = query.filter(RecoveryRecord.success == success)
            records = query.order_by(RecoveryRecord.start_time.desc()).limit(limit).all()
            
            result = []
            for r in records:
                try:
                    details = _json.loads(r.details) if r.details else {}
                except Exception as e:
                    logger.debug(f"Failed to parse recovery details for {r.id}: {e}")
                    details = {}
                
                result.append({
                    "id": r.id,
                    "failure_type": r.failure_type,
                    "action": r.action,
                    "attempt": r.attempt,
                    "success": r.success,
                    "error": r.error,
                    "duration_ms": r.duration_ms,
                    "details": details,
                    "start_time": r.start_time,
                    "end_time": r.end_time
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get recovery records: {e}")
            return []
        finally:
            session.close()

    def get_recovery_stats(self) -> Dict[str, Any]:
        session = self._Session()
        try:
            total = session.query(func.count(RecoveryRecord.id)).scalar() or 0
            success_count = session.query(func.count(RecoveryRecord.id)).filter(
                RecoveryRecord.success == True
            ).scalar() or 0
            
            results = session.query(
                RecoveryRecord.failure_type,
                func.count(RecoveryRecord.id),
                func.sum(RecoveryRecord.success.cast(Integer)),
                func.avg(RecoveryRecord.duration_ms)
            ).group_by(RecoveryRecord.failure_type).all()
            
            stats = {
                "total_recovery_attempts": total,
                "successful_attempts": success_count,
                "success_rate": round(success_count / total * 100, 2) if total > 0 else 0,
                "by_failure_type": {}
            }
            
            for failure_type, count, success, avg_duration in results:
                stats["by_failure_type"][failure_type] = {
                    "total_attempts": count,
                    "successful_attempts": success or 0,
                    "success_rate": round((success or 0) / count * 100, 2),
                    "avg_duration_ms": round(avg_duration or 0, 2)
                }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get recovery stats: {e}")
            return {}
        finally:
            session.close()