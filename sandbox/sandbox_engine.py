"""
策略沙盒引擎
============
整合虚拟账户、虚拟执行器、策略引擎和实时行情的沙盒运行环境

核心定位：
- 隔离的策略测试环境，不影响真实交易账户
- 使用OKX WebSocket获取实时行情数据进行模拟交易
- 支持策略并行运行和对比测试
- 虚拟资金管理 + 模拟订单执行 + 绩效评估
- 支持从沙盒到实盘的策略迁移

运行流程：
  实时行情 → 指标计算 → 信号生成 → 信号过滤 → 订单模拟 → 虚拟账户更新 → 绩效统计
"""

import asyncio
import threading
import time
from typing import Dict, Any, Optional, List, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from .virtual_account import VirtualAccount, create_virtual_account, get_virtual_account
from .virtual_executor import (
    VirtualOrderExecutor, FillResult, VirtualOrder,
    OrderType, OrderSide, OrderStatus
)

# 延迟导入，避免循环依赖
_indicator_engine = None
_signal_generator = None


def _get_indicator_engine():
    global _indicator_engine
    if _indicator_engine is None:
        from core.indicator_engine import IndicatorEngine, get_indicator_engine
        _indicator_engine = get_indicator_engine
    return _indicator_engine


def _get_signal_generator():
    global _signal_generator
    if _signal_generator is None:
        from core.signal_generator import SignalGenerator, get_signal_generator
        _signal_generator = get_signal_generator
    return _signal_generator


class SandboxState(Enum):
    """沙盒状态"""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    FROZEN = "frozen"  # 爆仓/风控冻结
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SandboxTrade:
    """沙盒交易记录"""
    trade_id: str
    symbol: str
    strategy: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    entry_fee: float
    exit_fee: float
    pnl: float
    pnl_pct: float
    hold_seconds: float
    exit_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "symbol": self.symbol,
            "strategy": self.strategy,
            "side": self.side,
            "entry_price": round(self.entry_price, 6),
            "exit_price": round(self.exit_price, 6),
            "quantity": round(self.quantity, 8),
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 4),
            "hold_seconds": round(self.hold_seconds, 1),
            "exit_reason": self.exit_reason,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
        }


@dataclass
class SandboxConfig:
    """沙盒配置"""
    sandbox_id: str
    name: str = ""
    initial_capital: float = 10000.0
    symbols: List[str] = field(default_factory=list)
    strategies: List[str] = field(default_factory=list)  # ["trend", "grid", "scalping"]
    timeframes: List[str] = field(default_factory=lambda: ["5m", "15m"])
    max_leverage: int = 15
    max_positions: int = 4
    risk_per_trade_pct: float = 0.02
    max_daily_loss_pct: float = 0.10
    max_drawdown_pct: float = 0.25
    snapshot_interval_seconds: int = 60
    order_timeout_seconds: int = 3600
    enable_auto_trading: bool = True
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 3.0
    min_signal_strength: float = 0.4
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sandbox_id": self.sandbox_id,
            "name": self.name,
            "initial_capital": self.initial_capital,
            "symbols": self.symbols,
            "strategies": self.strategies,
            "timeframes": self.timeframes,
            "max_leverage": self.max_leverage,
            "max_positions": self.max_positions,
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "enable_auto_trading": self.enable_auto_trading,
            "min_signal_strength": self.min_signal_strength,
            "description": self.description,
        }


class SandboxEngine:
    """
    策略沙盒引擎

    完整的隔离交易模拟环境：
    1. 创建独立虚拟账户
    2. 接入实时行情数据
    3. 运行策略生成信号
    4. 虚拟订单执行（含滑点/费率）
    5. 持仓管理和风控
    6. 绩效统计和报告
    """

    def __init__(self, config: SandboxConfig, system_config: Dict[str, Any] = None):
        self.config = config
        self.system_config = system_config or {}

        # 沙盒标识
        self.sandbox_id = config.sandbox_id
        self.state = SandboxState.CREATED

        # 创建虚拟账户
        account_config = {
            "maker_fee_rate": self.system_config.get("trading", {}).get("maker_fee_rate", 0.0002),
            "taker_fee_rate": self.system_config.get("trading", {}).get("taker_fee_rate", 0.0005),
            "max_leverage": config.max_leverage,
            "max_position_ratio": self.system_config.get("risk", {}).get("max_symbol_position_ratio", 2.0),
            "max_daily_loss_pct": config.max_daily_loss_pct,
            "max_drawdown_pct": config.max_drawdown_pct,
            "max_concurrent_positions": config.max_positions,
        }
        self.account = create_virtual_account(
            config.sandbox_id, config.initial_capital, account_config
        )

        # 创建虚拟订单执行器
        executor_config = {
            "maker_fee_rate": account_config["maker_fee_rate"],
            "taker_fee_rate": account_config["taker_fee_rate"],
            "slippage": self.system_config.get("execution", {}).get("slippage_optimizer") or {},
            "limit_order_ttl": config.order_timeout_seconds,
        }
        self.executor = VirtualOrderExecutor(executor_config)

        # 策略引擎组件
        self._indicator_engine = _get_indicator_engine()(system_config)
        self._signal_generator = _get_signal_generator()(system_config)

        # 注册symbols
        for symbol in config.symbols:
            self._indicator_engine.register_symbol(symbol)

        # 市场数据缓存
        self._prices: Dict[str, float] = {}
        self._klines: Dict[str, List[Dict[str, Any]]] = {}
        self._volatility_pcts: Dict[str, float] = {}
        self._last_bar_time: Dict[str, datetime] = {}

        # 持仓→交易映射（用于追踪交易对的入场信息）
        self._position_entries: Dict[str, Dict[str, Any]] = {}

        # 交易历史
        self._trade_history: List[SandboxTrade] = []
        self._trade_counter = 0
        self._lock = threading.RLock()

        # 运行控制
        self._running = False
        self._paused = False
        self._main_loop_task: Optional[asyncio.Task] = None
        self._snapshot_task: Optional[asyncio.Task] = None

        # 回调
        self._trade_callback: Optional[Callable] = None
        self._state_callback: Optional[Callable] = None
        self._log_callback: Optional[Callable] = None

        # 统计
        self._start_time: Optional[datetime] = None
        self._signals_generated = 0
        self._signals_executed = 0
        self._signals_filtered = 0

        logger.info(f"SandboxEngine '{config.sandbox_id}' created: "
                    f"capital={config.initial_capital} symbols={config.symbols} "
                    f"strategies={config.strategies}")

    # ── 生命周期 ─────────────────────────────────────────────────

    def initialize(self) -> bool:
        """初始化沙盒"""
        try:
            self.state = SandboxState.INITIALIZING

            # 初始化指标引擎
            for symbol in self.config.symbols:
                self._indicator_engine.register_symbol(symbol)

            self.state = SandboxState.CREATED
            self._log("沙盒初始化完成")
            return True
        except Exception as e:
            logger.error(f"Sandbox '{self.sandbox_id}' initialization failed: {e}")
            self.state = SandboxState.ERROR
            return False

    async def start(self) -> bool:
        """启动沙盒"""
        if self.state == SandboxState.RUNNING:
            return True

        try:
            self.state = SandboxState.RUNNING
            self._running = True
            self._paused = False
            self._start_time = datetime.now()

            # 启动快照定时任务
            self._snapshot_task = asyncio.create_task(self._snapshot_loop())

            self._log(f"沙盒启动 - {self.config.initial_capital} USDT")
            return True
        except Exception as e:
            logger.error(f"Sandbox '{self.sandbox_id}' start failed: {e}")
            self.state = SandboxState.ERROR
            return False

    async def stop(self) -> str:
        """停止沙盒"""
        self._running = False
        self.state = SandboxState.STOPPED

        if self._snapshot_task:
            self._snapshot_task.cancel()
            self._snapshot_task = None

        summary = self.get_performance_summary()
        self._log(f"沙盒停止 - ROI: {summary.get('roi_pct', 0)}%")
        return summary

    def pause(self) -> None:
        """暂停"""
        self._paused = True
        self.state = SandboxState.PAUSED
        self._log("沙盒已暂停")

    def resume(self) -> None:
        """恢复"""
        self._paused = False
        self.state = SandboxState.RUNNING
        self._log("沙盒已恢复")

    # ── 行情处理 ─────────────────────────────────────────────────

    async def on_bar(self, symbol: str, bar: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        处理K线数据

        核心流程：行情 → 指标 → 信号 → 订单 → 账户更新

        Returns:
            产生的信号列表（含执行结果）
        """
        if not self._running or self._paused:
            return []
        if self.account.is_frozen:
            return []

        results = []

        try:
            close = bar.get("close", bar.get("c", 0))
            if close <= 0:
                return []

            # 1. 更新价格缓存
            self._prices[symbol] = close

            # 2. 更新指标引擎
            self._indicator_engine.update_market_data(
                symbol=symbol,
                open_price=bar.get("open", bar.get("o", 0)),
                high=bar.get("high", bar.get("h", 0)),
                low=bar.get("low", bar.get("l", 0)),
                close=close,
                volume=bar.get("volume", bar.get("v", 0)),
            )

            # 3. 计算指标
            indicator_set = self._indicator_engine.calculate_all(symbol)
            atr_pct = indicator_set.get("atr_percent", 0.01)
            self._volatility_pcts[symbol] = atr_pct

            # 4. 处理挂单匹配
            fills = self.executor.process_orders(
                self._prices, self._volatility_pcts
            )
            for fill in fills:
                self._apply_fill(fill)

            # 5. 自动交易
            if self.config.enable_auto_trading:
                signals = await self._generate_and_execute_signals(symbol, indicator_set)
                results.extend(signals)

            # 6. 风控检查
            liq_check = self.account.check_liquidation(self._prices)
            if liq_check["liquidation_risk"]:
                self.state = SandboxState.FROZEN
                self._log(f"风控冻结: {self.account.freeze_reason}", level="WARNING")

            # 7. 更新资金曲线
            self.account.take_snapshot(self._prices)

        except Exception as e:
            logger.error(f"Sandbox '{self.sandbox_id}' on_bar error for {symbol}: {e}")

        return results

    async def on_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """处理tick数据（更新实时价格）"""
        price = tick.get("price", tick.get("last", 0))
        if price > 0:
            self._prices[symbol] = price

        # 处理挂单（高频匹配）
        fills = self.executor.process_orders(
            self._prices, self._volatility_pcts
        )
        for fill in fills:
            self._apply_fill(fill)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """批量更新价格"""
        self._prices.update(prices)

    # ── 信号生成与执行 ───────────────────────────────────────────

    async def _generate_and_execute_signals(self, symbol: str,
                                             indicator_set) -> List[Dict[str, Any]]:
        """生成信号并模拟执行"""
        from core.signal_generator import SignalContext, SignalType

        results = []
        price = self._prices.get(symbol, 0)
        if price <= 0:
            return results

        # 构建信号上下文
        indicators_dict = {
            name: result.value
            for name, result in indicator_set.indicators.items()
        }

        pos = self.account.get_position(symbol)
        context = SignalContext(
            symbol=symbol,
            current_price=price,
            position_side=pos.side if pos else None,
            position_size=pos.quantity if pos else 0.0,
            entry_price=pos.avg_entry_price if pos else 0.0,
            unrealized_pnl=pos.unrealized_pnl(price) if pos else 0.0,
            unrealized_pnl_pct=pos.unrealized_pnl_pct(price) if pos else 0.0,
            indicators=indicators_dict,
        )

        # 生成信号
        signals = self._signal_generator.generate_signals(context)
        self._signals_generated += len(signals)

        for signal in signals:
            # 过滤弱信号
            if signal.weight < self.config.min_signal_strength:
                self._signals_filtered += 1
                continue

            result = await self._execute_signal(symbol, signal, pos)
            if result:
                results.append(result)
                self._signals_executed += 1

        return results

    async def _execute_signal(self, symbol: str, signal,
                               pos: Optional[VirtualAccount] = None) -> Optional[Dict[str, Any]]:
        """执行交易信号"""
        from core.signal_generator import SignalType

        price = self._prices.get(symbol, signal.price)
        vol_pct = self._volatility_pcts.get(symbol, 0.01)
        sig_type = signal.signal_type

        result = {
            "symbol": symbol,
            "type": sig_type.value,
            "weight": signal.weight,
            "reason": signal.reason,
            "executed": False,
        }

        try:
            # ── 开仓信号 ──
            if sig_type in (SignalType.OPEN_LONG, SignalType.OPEN_SHORT):
                if pos and pos.quantity > 0:
                    return result  # 已有持仓

                side = "long" if sig_type == SignalType.OPEN_LONG else "short"
                quantity = self._calculate_order_quantity(symbol, price, vol_pct)

                fill = self.executor.place_market_order(
                    sandbox_id=self.sandbox_id,
                    symbol=symbol,
                    side=OrderSide.BUY if side == "long" else OrderSide.SELL,
                    quantity=quantity,
                    leverage=self.config.max_leverage,
                    pos_side=side,
                    strategy_name=signal.strategy_id or "sandbox",
                    current_price=price,
                    volatility_pct=vol_pct,
                )

                ok, msg, vpos = self.account.open_position(
                    symbol, side, quantity, fill.filled_price, self.config.max_leverage
                )

                if ok:
                    self._position_entries[symbol] = {
                        "entry_price": fill.filled_price,
                        "entry_time": datetime.now(),
                        "strategy": signal.strategy_id or "sandbox",
                        "side": side,
                    }
                    result["executed"] = True
                    result["action"] = f"开{side}"
                    result["price"] = fill.filled_price
                    result["quantity"] = quantity

            # ── 加仓信号 ──
            elif sig_type == SignalType.ADD_POSITION:
                if not pos:
                    return result

                add_qty = min(pos.quantity * 0.5,
                             self._calculate_order_quantity(symbol, price, vol_pct) * 0.5)

                fill = self.executor.place_market_order(
                    sandbox_id=self.sandbox_id,
                    symbol=symbol,
                    side=OrderSide.BUY if pos.side == "long" else OrderSide.SELL,
                    quantity=add_qty,
                    leverage=pos.leverage,
                    pos_side=pos.side,
                    current_price=price,
                    volatility_pct=vol_pct,
                )

                ok, msg = self.account.add_position(symbol, add_qty, fill.filled_price)
                if ok:
                    result["executed"] = True
                    result["action"] = "加仓"
                    result["quantity"] = add_qty

            # ── 减仓信号 ──
            elif sig_type == SignalType.REDUCE_POSITION:
                if not pos:
                    return result

                reduce_qty = pos.quantity * 0.3
                fill = self.executor.place_market_order(
                    sandbox_id=self.sandbox_id,
                    symbol=symbol,
                    side=OrderSide.SELL if pos.side == "long" else OrderSide.BUY,
                    quantity=reduce_qty,
                    leverage=pos.leverage,
                    pos_side=pos.side,
                    current_price=price,
                    volatility_pct=vol_pct,
                )

                ok, msg, pnl = self.account.reduce_position(
                    symbol, reduce_qty, fill.filled_price
                )
                if ok:
                    result["executed"] = True
                    result["action"] = "减仓"
                    result["pnl"] = round(pnl, 4)

            # ── 平仓信号 ──
            elif sig_type in (SignalType.TAKE_PROFIT, SignalType.STOP_LOSS, SignalType.CLOSE_ALL):
                if not pos:
                    return result

                fill = self.executor.place_market_order(
                    sandbox_id=self.sandbox_id,
                    symbol=symbol,
                    side=OrderSide.SELL if pos.side == "long" else OrderSide.BUY,
                    quantity=pos.quantity,
                    leverage=pos.leverage,
                    pos_side=pos.side,
                    current_price=price,
                    volatility_pct=vol_pct,
                )

                ok, msg, pnl = self.account.close_position(
                    symbol, fill.filled_price
                )
                if ok and symbol in self._position_entries:
                    entry = self._position_entries.pop(symbol)
                    self._trade_counter += 1
                    trade = SandboxTrade(
                        trade_id=f"SB_{self._trade_counter:06d}",
                        symbol=symbol,
                        strategy=entry.get("strategy", "sandbox"),
                        side=entry.get("side", "long"),
                        entry_price=entry["entry_price"],
                        exit_price=fill.filled_price,
                        quantity=abs(pnl / (fill.filled_price - entry["entry_price"] + 0.0001)),
                        entry_time=entry["entry_time"],
                        exit_time=datetime.now(),
                        entry_fee=0,
                        exit_fee=fill.fee,
                        pnl=pnl,
                        pnl_pct=pnl / self.account.initial_capital * 100,
                        hold_seconds=(datetime.now() - entry["entry_time"]).total_seconds(),
                        exit_reason=sig_type.value,
                    )
                    self._trade_history.append(trade)
                    self.account.update_daily_pnl(pnl)

                    result["executed"] = True
                    result["action"] = "平仓"
                    result["pnl"] = round(pnl, 4)

                    if self._trade_callback:
                        try:
                            self._trade_callback(trade)
                        except Exception:
                            pass

            # ── 对冲信号 ──
            elif sig_type == SignalType.HEDGE:
                if pos:
                    # 已有持仓时部分减仓作为对冲
                    hedge_qty = pos.quantity * 0.5
                    fill = self.executor.place_market_order(
                        sandbox_id=self.sandbox_id,
                        symbol=symbol,
                        side=OrderSide.SELL if pos.side == "long" else OrderSide.BUY,
                        quantity=hedge_qty,
                        leverage=pos.leverage,
                        pos_side=pos.side,
                        current_price=price,
                        volatility_pct=vol_pct,
                    )
                    ok, msg, pnl = self.account.reduce_position(
                        symbol, hedge_qty, fill.filled_price
                    )
                    if ok:
                        result["executed"] = True
                        result["action"] = "对冲减仓"

        except Exception as e:
            logger.error(f"Sandbox '{self.sandbox_id}' execute signal error: {e}")
            result["error"] = str(e)

        return result

    def _apply_fill(self, fill: FillResult) -> None:
        """将成交应用到账户"""
        # 成交记录已在 virtual_executor 中处理
        # 这里主要处理账户层面的更新
        pass

    def _calculate_order_quantity(self, symbol: str, price: float,
                                   volatility_pct: float) -> float:
        """根据风险参数计算订单数量"""
        equity = self.account.total_equity(self._prices)
        risk_amount = equity * self.config.risk_per_trade_pct

        # 基于ATR的止损距离
        sl_distance = price * volatility_pct * self.config.stop_loss_atr_mult
        if sl_distance <= 0:
            sl_distance = price * 0.01

        quantity = risk_amount / sl_distance

        # 限制单币种仓位
        max_pos_value = equity * 0.8 / max(len(self.config.symbols), 1)
        max_qty = max_pos_value / price
        quantity = min(quantity, max_qty)

        return round(quantity, 8)

    # ── 快照循环 ─────────────────────────────────────────────────

    async def _snapshot_loop(self) -> None:
        """定期快照循环"""
        while self._running:
            try:
                await asyncio.sleep(self.config.snapshot_interval_seconds)
                if not self._paused:
                    self.account.take_snapshot(self._prices)
                    self.account.reset_daily_if_needed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Snapshot loop error: {e}")
                await asyncio.sleep(5)

    # ── 绩效统计 ─────────────────────────────────────────────────

    def get_performance_summary(self) -> Dict[str, Any]:
        """获取绩效摘要"""
        stats = self.account.get_stats(self._prices)

        # 策略信号统计
        total_signals = self._signals_generated
        execution_rate = self._signals_executed / max(total_signals, 1)

        # 交易统计
        trades = self._trade_history
        closed_trades = [t for t in trades if t.pnl != 0]
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl < 0]

        # 胜率
        win_rate = len(wins) / len(closed_trades) if closed_trades else 0

        # 盈亏比
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = abs(sum(t.pnl for t in losses)) / len(losses) if losses else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        # 运行时长
        run_seconds = 0
        if self._start_time:
            run_seconds = (datetime.now() - self._start_time).total_seconds()

        # 合并
        stats.update({
            "sandbox_id": self.sandbox_id,
            "name": self.config.name,
            "state": self.state.value,
            "symbols": self.config.symbols,
            "strategies": self.config.strategies,
            "run_seconds": round(run_seconds, 0),
            "signals_generated": total_signals,
            "signals_executed": self._signals_executed,
            "signals_filtered": self._signals_filtered,
            "execution_rate_pct": round(execution_rate * 100, 2),
            "total_trades_count": len(trades),
            "closed_trades_count": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 4),
            "avg_loss": round(avg_loss, 4),
            "profit_factor": round(profit_factor, 4),
            "avg_hold_minutes": round(
                sum(t.hold_seconds for t in closed_trades) / max(len(closed_trades), 1) / 60, 1
            ),
            "recent_trades": [t.to_dict() for t in trades[-20:]],
        })

        return stats

    def get_equity_curve(self) -> List[Dict[str, Any]]:
        """获取资金曲线"""
        return self.account.get_equity_curve()

    def get_positions(self) -> List[Dict[str, Any]]:
        """获取当前持仓"""
        return self.account.get_positions_summary(self._prices)

    def get_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        """获取订单列表"""
        orders = self.executor.get_orders(sandbox_id=self.sandbox_id, limit=limit)
        return [o.to_dict() for o in orders]

    def get_status(self) -> Dict[str, Any]:
        """获取沙盒状态"""
        return {
            "sandbox_id": self.sandbox_id,
            "name": self.config.name,
            "state": self.state.value,
            "created_at": self._start_time.isoformat() if self._start_time else None,
            "prices": {s: round(p, 6) for s, p in self._prices.items()},
            "volatility_pcts": {s: round(v, 6) for s, v in self._volatility_pcts.items()},
            "account": self.account.get_stats(self._prices),
            "executor": self.executor.get_stats(),
            "positions": self.get_positions(),
        }

    # ── 回调注册 ─────────────────────────────────────────────────

    def register_trade_callback(self, callback: Callable) -> None:
        self._trade_callback = callback

    def register_state_callback(self, callback: Callable) -> None:
        self._state_callback = callback

    def register_log_callback(self, callback: Callable) -> None:
        self._log_callback = callback

    def _log(self, message: str, level: str = "INFO") -> None:
        """日志输出"""
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{self.sandbox_id}] {message}")
        if self._log_callback:
            try:
                self._log_callback(self.sandbox_id, message, level)
            except Exception:
                pass
