"""
统一止损管理器 - Stop Loss Manager

功能：
1. 硬性止损上限（max_stop_loss_pct）
2. ATR自适应止损
3. Trailing Stop（多档位）
4. 强制执行机制（确保止损信号直达OrderExecutor）
5. 审计日志和数据库记录
"""

import asyncio
import math
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger
from dataclasses import dataclass
import sqlite3


@dataclass
class StopLossEvent:
    """止损事件"""
    symbol: str
    strategy_name: str
    trigger_type: str  # "hard_stop", "atr_stop", "trailing_stop", "manual"
    entry_price: float
    trigger_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_percent: float
    exit_reason: str
    timestamp: datetime
    execution_latency_ms: float = 0.0
    slippage_pct: float = 0.0


class StopLossManager:
    """统一止损管理器"""
    
    def __init__(self, config: Dict[str, Any], okx_client, redis_cache, trade_journal, order_executor):
        self.config = config
        self._okx_client = okx_client
        self._redis_cache = redis_cache
        self._trade_journal = trade_journal
        self._order_executor = order_executor
        self._conditional_manager = None  # P0: 由Scheduler在初始化后注入
        self._reversal_tp_engine = None   # 行情反转落袋引擎，由 Scheduler 注入
        self._market_regime_detector = None  # HMM 检测器，用于自动获取反转信号
        
        # 止损配置（从config读取）
        trading_cfg = config.get("trading", {})
        self._max_stop_loss_pct = trading_cfg.get("max_stop_loss_pct", 0.05)  # 全局硬性止损上限5%
        # ── 参数层：max_stop_loss_pct 类型/范围/NaN/Inf 校验，非法回退安全默认 ──
        # 止损上限必须是 (0, 1] 之间的有限正数；0/负数会导致止损价等于或高于开仓价、
        # NaN/Inf 会污染下游止损价与 pnl 计算，>1 会让止损超出本金（无限亏损）。
        _default_max_sl = 0.05
        try:
            self._max_stop_loss_pct = float(self._max_stop_loss_pct)
        except (TypeError, ValueError):
            self._max_stop_loss_pct = _default_max_sl
        if not math.isfinite(self._max_stop_loss_pct) or self._max_stop_loss_pct <= 0 or self._max_stop_loss_pct > 1:
            self._max_stop_loss_pct = _default_max_sl
        self._trailing_activation_threshold = trading_cfg.get("trailing_activation_threshold", 0.015)  # 1.5%盈利激活
        self._trailing_min_distance = trading_cfg.get("trailing_min_distance", 0.005)  # 最小追踪距离0.5%
        self._force_execution_enabled = trading_cfg.get("force_stop_loss_execution", True)
        
        # 止损事件历史
        self._stop_loss_events: List[StopLossEvent] = []
        self._pending_stop_losses: Dict[str, Dict[str, Any]] = {}  # key(symbol:strategy) -> stop_loss_order
        self._triggered_recently: Dict[str, float] = {}  # key(symbol:strategy) -> trigger_time, 防重复触发
        self._sl_cooldown_seconds = 120  # 同一position的止损冷却时间（秒）
        
        # 止损审计数据库
        self._db_path = config.get("sqlite", {}).get("db_path", "./data/trading.db")
        self._init_audit_table()
        
        # 回调函数（策略注入）
        self._strategy_callbacks: Dict[str, callable] = {}
        
        logger.info(f"StopLossManager initialized: max_sl={self._max_stop_loss_pct:.2%}, "
                   f"trailing_activation={self._trailing_activation_threshold:.2%}, "
                   f"force_execution={self._force_execution_enabled}")
    
    def _init_audit_table(self):
        """初始化止损审计表"""
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stop_loss_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol VARCHAR(50),
                    strategy_name VARCHAR(50),
                    trigger_type VARCHAR(30),
                    entry_price FLOAT,
                    trigger_price FLOAT,
                    exit_price FLOAT,
                    quantity FLOAT,
                    pnl FLOAT,
                    pnl_percent FLOAT,
                    exit_reason VARCHAR(50),
                    timestamp DATETIME,
                    execution_latency_ms FLOAT,
                    slippage_pct FLOAT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to init stop_loss_audit table: {e}")
        finally:
            if conn:
                conn.close()
    
    def register_strategy_callback(self, strategy_name: str, callback: callable):
        """注册策略回调函数（用于强制执行止损）"""
        self._strategy_callbacks[strategy_name] = callback
        logger.debug(f"StopLossManager registered callback for {strategy_name}")

    def set_conditional_manager(self, conditional_manager):
        """注入条件单管理器，以便止损触发时取消相关条件单（避免孤儿订单）"""
        self._conditional_manager = conditional_manager

    def set_reversal_take_profit_engine(self, engine):
        """注入行情反转落袋引擎（由 Scheduler 调用）"""
        self._reversal_tp_engine = engine

    def set_market_regime_detector(self, detector):
        """注入 HMM 检测器，用于自动获取反转信号（由 Scheduler 调用）"""
        self._market_regime_detector = detector
    
    def calculate_stop_loss(
        self,
        symbol: str,
        entry_price: float,
        direction: str,
        atr: float = 0,
        strategy_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        计算止损价格（统一逻辑）
        
        返回：
        {
            "hard_stop": float,  # 硬性止损价
            "atr_stop": float,   # ATR自适应止损价
            "final_stop": float, # 最终止损价（取更严格的）
            "max_stop": float,   # 最大止损价（硬性上限）
            "trailing_activation": float,  # trailing stop激活价格
        }
        """
        if strategy_config is None:
            strategy_config = {}

        # ── 数据层：entry_price 类型/NaN/Inf/非正数校验 ──
        # entry_price<=0 或 NaN/Inf 会让止损价变成 0/NaN/负价，进而触发错误的止损平仓。
        try:
            entry_price = float(entry_price)
        except (TypeError, ValueError):
            logger.warning(f"[STOP_LOSS] invalid entry_price={entry_price!r} for {symbol}, abort")
            return {
                "hard_stop": 0.0, "atr_stop": 0.0, "final_stop": 0.0,
                "max_stop": 0.0, "trailing_activation": 0.0,
            }
        if not math.isfinite(entry_price) or entry_price <= 0:
            logger.warning(f"[STOP_LOSS] non-finite/<=0 entry_price={entry_price!r} for {symbol}, abort")
            return {
                "hard_stop": 0.0, "atr_stop": 0.0, "final_stop": 0.0,
                "max_stop": 0.0, "trailing_activation": 0.0,
            }

        # ── 数据层：direction 归一化，非法回退中性 ──
        direction = (direction or "").lower()
        if direction not in ("long", "short"):
            logger.warning(f"[STOP_LOSS] invalid direction={direction!r} for {symbol}, abort")
            return {
                "hard_stop": 0.0, "atr_stop": 0.0, "final_stop": 0.0,
                "max_stop": 0.0, "trailing_activation": 0.0,
            }

        precision = 4
        try:
            # 获取价格精度
            inst_info = self._okx_client.get_instrument_info(symbol)
            if inst_info:
                tick_sz = inst_info.get("tickSz", "0.0001")
                precision = len(tick_sz.split(".")[-1]) if "." in tick_sz else 0
        except Exception:
            pass

        # ── 数据层：strategy_sl_pct 清洗（NaN/Inf/类型/越界 → 安全默认） ──
        strategy_sl_pct = strategy_config.get("stop_loss_pct", 0.03)
        try:
            strategy_sl_pct = float(strategy_sl_pct)
        except (TypeError, ValueError):
            strategy_sl_pct = 0.03
        if not math.isfinite(strategy_sl_pct) or strategy_sl_pct <= 0 or strategy_sl_pct > 1:
            strategy_sl_pct = 0.03

        # ── 数据层：max_sl_pct 清洗（None/NaN/Inf/类型/越界 → 回退全局上限） ──
        raw_max = strategy_config.get("max_stop_loss_pct", self._max_stop_loss_pct)
        try:
            max_sl_pct = float(raw_max)
        except (TypeError, ValueError):
            max_sl_pct = self._max_stop_loss_pct
        if not math.isfinite(max_sl_pct) or max_sl_pct <= 0 or max_sl_pct > 1:
            max_sl_pct = self._max_stop_loss_pct
        # 策略止损上限不得超过全局硬上限（取更严格者）
        max_sl_pct = min(max_sl_pct, self._max_stop_loss_pct)

        # 硬性止损
        if direction == "long":
            hard_stop = entry_price * (1 - strategy_sl_pct)
            max_stop = entry_price * (1 - max_sl_pct)
        else:
            hard_stop = entry_price * (1 + strategy_sl_pct)
            max_stop = entry_price * (1 + max_sl_pct)

        # ── 运算层：止损价 NaN/Inf/非正数防护 ──
        if not math.isfinite(hard_stop) or hard_stop <= 0:
            hard_stop = entry_price
        if not math.isfinite(max_stop) or max_stop <= 0:
            max_stop = entry_price
        
        # ATR自适应止损
        atr_stop = hard_stop
        if atr > 0:
            atr_multiplier = strategy_config.get("atr_multiplier", 2.0)
            if direction == "long":
                atr_stop = entry_price - atr * atr_multiplier
                # 确保不超过硬性止损
                atr_stop = max(atr_stop, hard_stop)
            else:
                atr_stop = entry_price + atr * atr_multiplier
                atr_stop = min(atr_stop, hard_stop)
        
        # 最终止损：取更严格的（对于多头，取更高的止损价；对于空头，取更低的止损价）
        if direction == "long":
            final_stop = max(hard_stop, atr_stop)
            # 确保不超过硬性上限
            final_stop = max(final_stop, max_stop)
        else:
            final_stop = min(hard_stop, atr_stop)
            final_stop = min(final_stop, max_stop)
        
        # Trailing stop激活价格
        if direction == "long":
            trailing_activation = entry_price * (1 + self._trailing_activation_threshold)
        else:
            trailing_activation = entry_price * (1 - self._trailing_activation_threshold)
        
        return {
            "hard_stop": round(hard_stop, precision),
            "atr_stop": round(atr_stop, precision),
            "final_stop": round(final_stop, precision),
            "max_stop": round(max_stop, precision),
            "trailing_activation": round(trailing_activation, precision),
        }
    
    def update_trailing_stop(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        direction: str,
        current_stop: float,
        peak_price: float = None
    ) -> Optional[float]:
        """
        更新Trailing Stop
        
        返回：新的止损价（如果需要更新），否则返回None
        """
        if peak_price is None:
            peak_price = current_price
        
        precision = 4
        
        # 检查是否已激活trailing stop
        if direction == "long":
            if current_price < entry_price * (1 + self._trailing_activation_threshold):
                return None
            
            # 盈利超过阈值，开始追踪
            new_stop = current_price * (1 - self._trailing_min_distance)
            # 只向有利方向移动
            if new_stop > current_stop:
                return round(new_stop, precision)
        else:
            if current_price > entry_price * (1 - self._trailing_activation_threshold):
                return None
            
            new_stop = current_price * (1 + self._trailing_min_distance)
            if new_stop < current_stop:
                return round(new_stop, precision)
        
        return None
    
    async def check_and_execute_stop_loss(
        self,
        symbol: str,
        strategy_name: str,
        current_price: float,
        position_state: Dict[str, Any]
    ) -> bool:
        """
        检查并执行止损（核心方法）
        
        返回：是否触发了止损
        """
        direction = position_state.get("direction", "")
        entry_price = position_state.get("entry_price", 0)
        stop_loss_price = position_state.get("stop_loss") or position_state.get("final_stop") or position_state.get("adjusted_stop_loss")
        
        if not direction or entry_price <= 0 or stop_loss_price <= 0:
            return False
        
        triggered = False
        trigger_type = ""
        
        # 检查是否触发止损
        if direction == "long" and current_price <= stop_loss_price:
            triggered = True
            trigger_type = position_state.get("exit_reason", "hard_stop")
        elif direction == "short" and current_price >= stop_loss_price:
            triggered = True
            trigger_type = position_state.get("exit_reason", "hard_stop")
        
        if not triggered:
            return False
        
        # 防重复触发：同一symbol+strategy在冷却期内不重复触发
        sl_key = f"{symbol}:{strategy_name}"
        now_ts = datetime.now().timestamp()
        last_trigger = self._triggered_recently.get(sl_key, 0)
        if now_ts - last_trigger < self._sl_cooldown_seconds:
            logger.debug(f"[STOP_LOSS_COOLDOWN] {sl_key} skipped, last triggered {now_ts - last_trigger:.0f}s ago")
            return False
        # P0 修复：冷却写入延后到执行成功之后，避免止损失败被 120s 冷却阻塞补救。
        # 若执行失败，不设冷却，下一个 tick 会再次进入本方法并重试止损平仓。
        
        # 触发止损，记录事件
        start_time = datetime.now()
        logger.warning(f"[STOP_LOSS_TRIGGER] {symbol} {strategy_name}: {trigger_type}, "
                      f"entry={entry_price:.4f}, current={current_price:.4f}, stop={stop_loss_price:.4f}")
        
        # 强制执行止损平仓
        # 兼容不同策略的字段命名：trend用"current_quantity"，其他用"quantity"
        quantity = position_state.get("current_quantity", 0) or position_state.get("quantity", 0)
        
        # qty=0 表示持仓已不存在（已平仓但状态未同步），跳过执行
        if quantity <= 0:
            logger.info(f"[STOP_LOSS_SKIP] {symbol} {strategy_name}: qty=0, position already closed")
            return False
        
        exit_reason = trigger_type if trigger_type in ["stop_loss", "trailing_stop", "atr_stop"] else "stop_loss"
        
        # 调用OrderExecutor执行止损平仓（最高优先级）
        success = await self._execute_stop_loss_order(
            symbol=symbol,
            strategy_name=strategy_name,
            direction=direction,
            quantity=quantity,
            current_price=current_price,
            exit_reason=exit_reason
        )

        # P0-2 修复：止损平仓成功后，同步关闭 trade_records 的 open 记录并记录 exit_reason。
        # 历史事故：止损平仓因 qty/direction 与 open 记录不匹配，落入 "no open record" 分支，
        # open 记录最终被 reconcile 兜底误标为 ghost_close，且 pnl 被整仓 unrealized_pnl 错误分摊
        # （XRP trend 真实止损仅 2.14%，却被记成 -74%）。
        # 此处主动关闭 open 记录：pnl 保持 None，交由 PnLReconciler 用 OKX 账单精确对账，
        # 确保 exit_reason 正确、不写估算假数据。若 open 记录已被平仓流程正常关闭，本调用幂等无副作用。
        if success:
            # P0 修复：仅在止损平仓成功后才写入冷却，并清理过期冷却记录（> 5 min）
            self._triggered_recently[sl_key] = now_ts
            self._triggered_recently = {
                k: v for k, v in self._triggered_recently.items()
                if now_ts - v < 300
            }
            # 移除已成功处理的挂起止损单
            self._pending_stop_losses.pop(sl_key, None)
            try:
                if self._order_executor is not None and hasattr(self._order_executor, "sqlite_storage"):
                    self._order_executor.sqlite_storage.close_open_record(symbol, strategy_name, exit_reason)
            except Exception as e:
                logger.debug(f"Failed to close open record after stop loss {symbol}/{strategy_name}: {e}")
        else:
            # P0 修复：止损失败，明确告警并挂起待重试，且不设冷却（下一个 tick 会再次进入本方法重试）
            logger.error(
                f"[STOP_LOSS_EXECUTION_FAILED] {symbol} {strategy_name}: stop loss order failed after retries, "
                f"position NOT closed. entry={entry_price:.4f}, stop={stop_loss_price:.4f}, current={current_price:.4f}"
            )
            self._pending_stop_losses[sl_key] = {
                "symbol": symbol,
                "strategy_name": strategy_name,
                "direction": direction,
                "quantity": quantity,
                "current_price": current_price,
                "exit_reason": exit_reason,
                "first_attempt_at": start_time.isoformat(),
            }

        end_time = datetime.now()
        execution_latency_ms = (end_time - start_time).total_seconds() * 1000
        
        # 记录止损事件
        pnl = (current_price - entry_price) * quantity if direction == "long" else (entry_price - current_price) * quantity
        # ── 运算层：pnl_percent 除零/NaN 防护 ──
        if not math.isfinite(entry_price) or entry_price == 0:
            pnl_percent = 0.0
        else:
            pnl_percent = (current_price - entry_price) / entry_price if direction == "long" else (entry_price - current_price) / entry_price
            if not math.isfinite(pnl_percent):
                pnl_percent = 0.0
        
        event = StopLossEvent(
            symbol=symbol,
            strategy_name=strategy_name,
            trigger_type=trigger_type,
            entry_price=entry_price,
            trigger_price=stop_loss_price,
            exit_price=current_price,
            quantity=quantity,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=exit_reason,
            timestamp=end_time,
            execution_latency_ms=execution_latency_ms,
            slippage_pct=abs(current_price - stop_loss_price) / stop_loss_price if stop_loss_price > 0 else 0
        )
        
        self._stop_loss_events.append(event)
        # 防止内存泄漏：最多保留最近500条事件
        if len(self._stop_loss_events) > 500:
            self._stop_loss_events = self._stop_loss_events[-500:]
        self._save_stop_loss_audit(event)
        
        return True

    async def compute_reversal_take_profit(
        self,
        symbol: str,
        strategy_name: str,
        direction: str,
        entry_price: float,
        current_price: float,
        base_tp_price: Optional[float] = None,
        hmm_result: Dict[str, Any] = None,
        ohlcv_data: List = None,
        atr: Optional[float] = None,
    ):
        """纯计算行情反转落袋结果（自动获取 HMM 与 K 线），不执行平仓。

        供 check_reversal_take_profit 与各策略（如网格趋势模式）复用。
        返回 ReversalTakeProfitResult；引擎未注入或计算异常时返回 None。
        """
        engine = getattr(self, '_reversal_tp_engine', None)
        if engine is None:
            return None

        try:
            hmm_result, ohlcv_data = await self._fetch_reversal_inputs(symbol, hmm_result, ohlcv_data)
            return engine.compute(
                symbol=symbol,
                strategy_name=strategy_name,
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                base_tp_price=base_tp_price,
                hmm_result=hmm_result,
                ohlcv_data=ohlcv_data,
                atr=atr,
            )
        except Exception as e:
            logger.error(f"[REVERSAL_TP_COMPUTE_ERROR] {symbol} {strategy_name}: {e}")
            return None

    async def _fetch_reversal_inputs(self, symbol: str, hmm_result: Dict[str, Any], ohlcv_data: List):
        """自动获取反转信号（未显式传入时）：HMM 检测器 + K 线兜底。"""
        if hmm_result is None:
            detector = getattr(self, '_market_regime_detector', None)
            if detector is not None and hasattr(detector, 'get_regime'):
                try:
                    hmm_result = detector.get_regime(symbol)
                except Exception as e:
                    logger.debug(f"get_regime failed for {symbol}: {e}")

        if ohlcv_data is None:
            client = getattr(self, '_okx_client', None)
            if client is not None and hasattr(client, 'get_kline_async'):
                try:
                    ohlcv_data = await client.get_kline_async(symbol, "1H", limit=60)
                except Exception as e:
                    logger.debug(f"get_kline_async failed for {symbol}: {e}")

        return hmm_result, ohlcv_data

    async def check_reversal_take_profit(
        self,
        symbol: str,
        strategy_name: str,
        current_price: float,
        position_state: Dict[str, Any],
        hmm_result: Dict[str, Any] = None,
        ohlcv_data: List = None
    ) -> bool:
        """检查行情反转落袋（动态止盈 + 反转减仓）。

        返回：是否执行了落袋平仓（部分或全部）。
        """
        engine = getattr(self, '_reversal_tp_engine', None)
        if engine is None:
            return False

        try:
            direction = position_state.get("direction", "")
            entry_price = float(position_state.get("entry_price", 0) or 0)
            if not direction or entry_price <= 0:
                return False

            base_tp = (position_state.get("take_profit")
                       or position_state.get("take_profit_price")
                       or position_state.get("target_price"))
            atr = position_state.get("atr") or position_state.get("atr_value")
            quantity = float(position_state.get("current_quantity", 0)
                             or position_state.get("quantity", 0) or 0)
            if quantity <= 0:
                return False

            result = await self.compute_reversal_take_profit(
                symbol=symbol,
                strategy_name=strategy_name,
                direction=direction,
                entry_price=entry_price,
                current_price=current_price,
                base_tp_price=float(base_tp) if base_tp else None,
                hmm_result=hmm_result,
                ohlcv_data=ohlcv_data,
                atr=float(atr) if atr else None,
            )
            if result is None:
                return False

            # 更新动态止盈到持仓状态（供策略/条件单使用）
            if result.adaptive_tp_price is not None:
                position_state["take_profit"] = result.adaptive_tp_price
                position_state["take_profit_price"] = result.adaptive_tp_price
            position_state["reversal_score"] = result.reversal_score
            position_state["reversal_source"] = result.reversal_source

            if result.exit_action == "none":
                return False

            is_full = result.exit_action == "close"
            exit_qty = quantity if is_full else quantity * result.partial_ratio
            if exit_qty <= 0:
                return False

            logger.warning(
                f"[REVERSAL_TAKE_PROFIT] {symbol} {strategy_name}: action={result.exit_action}, "
                f"score={result.reversal_score:.2f}, source={result.reversal_source}, "
                f"pnl={result.details.get('pnl_pct', 0):.2%}, exit_qty={exit_qty:.4f}"
            )

            return await self._execute_reversal_take_profit_order(
                symbol=symbol,
                strategy_name=strategy_name,
                direction=direction,
                quantity=exit_qty,
                current_price=current_price,
                exit_reason=result.exit_reason,
                is_full_close=is_full,
            )
        except Exception as e:
            logger.error(f"[REVERSAL_TAKE_PROFIT_ERROR] {symbol} {strategy_name}: {e}")
            return False

    async def _execute_reversal_take_profit_order(
        self,
        symbol: str,
        strategy_name: str,
        direction: str,
        quantity: float,
        current_price: float,
        exit_reason: str,
        is_full_close: bool = True
    ) -> bool:
        """执行反转落袋平仓订单（reduce-only）。

        - 全平：取消该 symbol 的 SL/TP 条件单，避免孤儿订单。
        - 部分落袋：保留剩余仓位的条件单，只减仓锁定部分利润。
        """
        try:
            # 全平时取消条件单，部分落袋保留剩余仓位条件单
            if is_full_close and self._conditional_manager:
                try:
                    sl_orders = self._conditional_manager.get_sl_orders_for_symbol(symbol)
                    tp_orders = self._conditional_manager.get_tp_orders_for_symbol(symbol)
                    for oid, _ in sl_orders + tp_orders:
                        self._conditional_manager.cancel_conditional_order(symbol, oid)
                except Exception as e:
                    logger.warning(f"Failed to cancel conditional orders for {symbol}: {e}")

            signal_data = {
                "type": "signal",
                "priority": "take_profit",
                "data": {
                    "symbol": symbol,
                    "strategy_name": strategy_name,
                    "signal_type": "take_profit",
                    "direction": "sell" if direction == "long" else "buy",
                    "pos_side": direction,
                    "price": current_price,
                    "quantity": quantity,
                    "leverage": 1,
                    "stop_loss": None,
                    "take_profit": None,
                    "confidence": 1.0,
                    "exit_reason": exit_reason,
                    "reduce_only": True,
                    "timestamp": datetime.now().isoformat(),
                },
            }

            if self._order_executor:
                await self._order_executor.handle_signal(signal_data["data"])
                logger.info(
                    f"[REVERSAL_TAKE_PROFIT_EXECUTED] {symbol} {strategy_name}: "
                    f"{exit_reason}, qty={quantity:.4f}, full={is_full_close}"
                )
                return True
            logger.error(f"[REVERSAL_TAKE_PROFIT_FAILED] No OrderExecutor for {symbol}")
            return False
        except Exception as e:
            logger.error(f"[REVERSAL_TAKE_PROFIT_EXECUTION_ERROR] {symbol} {strategy_name}: {e}")
            return False

    async def _execute_stop_loss_order(
        self,
        symbol: str,
        strategy_name: str,
        direction: str,
        quantity: float,
        current_price: float,
        exit_reason: str,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> bool:
        """
        执行止损平仓订单（最高优先级，带重试）

        P0 修复：原实现单次 try/except，止损失败即静默 return False，无重试、无告警，
        且冷却已在前置写入，失败后 120s 内无法补救，导致仓位裸奔。
        现改为：取消条件单（一次）→ 构造信号 → 最多重试 max_retries 次。
        """
        if not self._order_executor:
            logger.error(f"[STOP_LOSS_FAILED] No OrderExecutor available for {symbol}")
            return False

        # P0: 止损触发前先取消该symbol的所有条件单（避免孤儿订单残留在交易所）
        if self._conditional_manager:
            try:
                sl_orders = self._conditional_manager.get_sl_orders_for_symbol(symbol)
                tp_orders = self._conditional_manager.get_tp_orders_for_symbol(symbol)
                for oid, _ in sl_orders + tp_orders:
                    self._conditional_manager.cancel_conditional_order(symbol, oid)
                if sl_orders or tp_orders:
                    logger.info(f"[STOP_LOSS_CANCEL] Cancelled {len(sl_orders)} SL + {len(tp_orders)} TP conditional orders for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to cancel conditional orders for {symbol}: {e}")

        # 构造止损平仓信号
        signal_data = {
            "type": "signal",
            "priority": "stop_loss",  # 最高优先级
            "data": {
                "symbol": symbol,
                "strategy_name": strategy_name,
                "signal_type": "stop_loss",
                "direction": "sell" if direction == "long" else "buy",
                "pos_side": direction,
                "price": current_price,
                "quantity": quantity,
                "leverage": 1,  # 止损平仓使用实际杠杆
                "stop_loss": None,
                "take_profit": None,
                "confidence": 1.0,  # 止损信号置信度100%
                "exit_reason": exit_reason,
                "reduce_only": True,
                "timestamp": datetime.now().isoformat()
            }
        }

        # 直接发送给OrderExecutor（绕过SignalProcessor优先级检查），失败自动重试
        for attempt in range(1, max_retries + 1):
            try:
                await self._order_executor.handle_signal(signal_data["data"])
                _suffix = f" (attempt {attempt}/{max_retries})" if attempt > 1 else ""
                logger.info(f"[STOP_LOSS_EXECUTED] {symbol} {strategy_name}: {exit_reason}, qty={quantity:.4f}{_suffix}")
                return True
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f"[STOP_LOSS_RETRY] {symbol} {strategy_name} attempt {attempt}/{max_retries} failed: {e}, "
                        f"retrying in {retry_delay:.0f}s"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"[STOP_LOSS_EXECUTION_ERROR] {symbol} {strategy_name} exhausted {max_retries} retries: {e}")
        return False
    
    def _save_stop_loss_audit(self, event: StopLossEvent):
        """保存止损审计记录到数据库"""
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO stop_loss_audit 
                (symbol, strategy_name, trigger_type, entry_price, trigger_price, exit_price,
                 quantity, pnl, pnl_percent, exit_reason, timestamp, execution_latency_ms, slippage_pct)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.symbol, event.strategy_name, event.trigger_type, event.entry_price,
                event.trigger_price, event.exit_price, event.quantity, event.pnl,
                event.pnl_percent, event.exit_reason, event.timestamp.isoformat(),
                event.execution_latency_ms, event.slippage_pct
            ))
            conn.commit()
        except Exception as e:
            logger.error(f"Failed to save stop_loss_audit: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_stop_loss_statistics(self, strategy_name: str = None, days: int = 30) -> Dict[str, Any]:
        """获取止损统计"""
        conn = None
        try:
            conn = sqlite3.connect(self._db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            cursor = conn.cursor()
            
            query = """
                SELECT 
                    trigger_type,
                    COUNT(*) as count,
                    AVG(pnl_percent) as avg_pnl_pct,
                    AVG(execution_latency_ms) as avg_latency_ms,
                    AVG(slippage_pct) as avg_slippage_pct
                FROM stop_loss_audit
                WHERE timestamp >= datetime('now', 'localtime', ?)
            """
            params = [f'-{days} days']
            
            if strategy_name:
                query += " AND strategy_name = ?"
                params.append(strategy_name)
            
            query += " GROUP BY trigger_type"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            stats = {
                "trigger_types": {},
                "total_count": 0,
                "avg_execution_latency_ms": 0,
                "avg_slippage_pct": 0
            }
            
            total_latency = 0
            total_slippage = 0
            
            for row in rows:
                trigger_type, count, avg_pnl_pct, avg_latency_ms, avg_slippage_pct = row
                stats["trigger_types"][trigger_type] = {
                    "count": count,
                    "avg_pnl_pct": avg_pnl_pct or 0,
                    "avg_latency_ms": avg_latency_ms or 0,
                    "avg_slippage_pct": avg_slippage_pct or 0
                }
                stats["total_count"] += count
                total_latency += (avg_latency_ms or 0) * count
                total_slippage += (avg_slippage_pct or 0) * count
            
            if stats["total_count"] > 0:
                stats["avg_execution_latency_ms"] = total_latency / stats["total_count"]
                stats["avg_slippage_pct"] = total_slippage / stats["total_count"]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get stop_loss statistics: {e}")
            return {}
        finally:
            if conn:
                conn.close()