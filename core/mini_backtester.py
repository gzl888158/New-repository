"""
实时迷你回测单元
================
核心定位：盘中滚动回测，动态校验当前策略有效性，失效自动冻结策略

功能：
- 滚动窗口回测（最近N根K线）
- 策略有效性评估（胜率、盈亏比、最大回撤）
- 失效检测（连续亏损、胜率过低、回撤过大）
- 自动冻结失效策略
"""

import asyncio
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from loguru import logger
import threading
import time


class StrategyHealth(Enum):
    """策略健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FROZEN = "frozen"


@dataclass
class TradeRecord:
    """交易记录"""
    timestamp: datetime
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    hold_bars: int
    signal_source: str
    entry_reason: str
    exit_reason: str


@dataclass
class BacktestResult:
    """回测结果"""
    strategy_id: str
    symbol: str
    start_time: datetime
    end_time: datetime
    bars_count: int
    
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    
    health: StrategyHealth = StrategyHealth.HEALTHY
    freeze_reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "bars_count": self.bars_count,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": round(self.total_pnl, 4),
            "max_pnl": round(self.max_pnl, 4),
            "min_pnl": round(self.min_pnl, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "avg_trade_pnl": round(self.avg_trade_pnl, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "health": self.health.value,
            "freeze_reason": self.freeze_reason
        }


@dataclass
class HealthThresholds:
    """健康阈值配置"""
    min_win_rate: float = 0.35
    min_profit_factor: float = 0.8
    max_drawdown_pct: float = 0.15
    max_consecutive_losses: int = 5
    min_trades_for_evaluation: int = 5
    
    warning_win_rate: float = 0.40
    warning_profit_factor: float = 1.0
    warning_drawdown_pct: float = 0.10


class MiniBacktester:
    """
    实时迷你回测器
    
    特性：
    - 滚动窗口回测：基于最近N根K线
    - 快速评估：秒级完成回测计算
    - 自动冻结：策略失效时自动冻结
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        backtest_config = self.config.get("mini_backtest", {})
        self._window_size = backtest_config.get("window_size", 100)
        self._min_bars = backtest_config.get("min_bars", 20)
        self._retest_interval = backtest_config.get("retest_interval_seconds", 60)
        
        self._thresholds = HealthThresholds(
            min_win_rate=backtest_config.get("min_win_rate", 0.35),
            min_profit_factor=backtest_config.get("min_profit_factor", 0.8),
            max_drawdown_pct=backtest_config.get("max_drawdown_pct", 0.15),
            max_consecutive_losses=backtest_config.get("max_consecutive_losses", 5),
            min_trades_for_evaluation=backtest_config.get("min_trades_for_evaluation", 5),
            warning_win_rate=backtest_config.get("warning_win_rate", 0.40),
            warning_profit_factor=backtest_config.get("warning_profit_factor", 1.0),
            warning_drawdown_pct=backtest_config.get("warning_drawdown_pct", 0.10)
        )
        
        self._price_history: Dict[str, deque] = {}
        self._trade_history: Dict[str, deque] = {}
        self._backtest_results: Dict[str, BacktestResult] = {}
        self._consecutive_losses: Dict[str, int] = {}
        
        self._lock = threading.RLock()
        
        self._freeze_callback: Optional[callable] = None
        
        self._test_count = 0
        self._total_test_time = 0.0
    
    def set_freeze_callback(self, callback: callable) -> None:
        """设置策略冻结回调"""
        self._freeze_callback = callback
    
    def register_strategy(self, strategy_id: str, symbol: str) -> None:
        """注册策略进行回测监控"""
        with self._lock:
            if strategy_id not in self._price_history:
                self._price_history[strategy_id] = deque(maxlen=self._window_size)
            if strategy_id not in self._trade_history:
                self._trade_history[strategy_id] = deque(maxlen=self._window_size)
            if strategy_id not in self._consecutive_losses:
                self._consecutive_losses[strategy_id] = 0
            
            logger.debug(f"Registered strategy for mini backtest: {strategy_id}")
    
    def add_bar(self, strategy_id: str, bar: Dict[str, Any]) -> None:
        """添加K线数据"""
        with self._lock:
            if strategy_id in self._price_history:
                self._price_history[strategy_id].append({
                    "timestamp": bar.get("timestamp", datetime.now()),
                    "open": bar.get("open", 0),
                    "high": bar.get("high", 0),
                    "low": bar.get("low", 0),
                    "close": bar.get("close", 0),
                    "volume": bar.get("volume", 0)
                })
    
    def add_trade(self, strategy_id: str, trade: TradeRecord) -> None:
        """添加交易记录"""
        with self._lock:
            if strategy_id in self._trade_history:
                self._trade_history[strategy_id].append(trade)
                
                if trade.pnl < 0:
                    self._consecutive_losses[strategy_id] = self._consecutive_losses.get(strategy_id, 0) + 1
                else:
                    self._consecutive_losses[strategy_id] = 0
    
    def run_backtest(self, strategy_id: str) -> Optional[BacktestResult]:
        """
        运行迷你回测
        
        Args:
            strategy_id: 策略ID
        
        Returns:
            BacktestResult: 回测结果
        """
        start_time = time.time()
        
        with self._lock:
            if strategy_id not in self._trade_history:
                return None
            
            trades = list(self._trade_history[strategy_id])
            bars = list(self._price_history[strategy_id])
        
        if len(trades) < self._thresholds.min_trades_for_evaluation:
            return None
        
        result = self._calculate_backtest_metrics(strategy_id, trades)
        
        if result:
            self._evaluate_strategy_health(result, strategy_id)
            
            with self._lock:
                self._backtest_results[strategy_id] = result
        
        elapsed = time.time() - start_time
        self._test_count += 1
        self._total_test_time += elapsed
        
        return result
    
    def _calculate_backtest_metrics(self, strategy_id: str, 
                                     trades: List[TradeRecord]) -> BacktestResult:
        """计算回测指标"""
        if not trades:
            return None
        
        pnls = [t.pnl for t in trades]
        pnl_pcts = [t.pnl_pct for t in trades]
        
        total_pnl = sum(pnls)
        winning_trades = sum(1 for p in pnls if p > 0)
        losing_trades = sum(1 for p in pnls if p < 0)
        total_trades = len(trades)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        cumulative_pnl = np.cumsum(pnls)
        max_pnl = max(cumulative_pnl) if len(cumulative_pnl) > 0 else 0
        
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdowns = running_max - cumulative_pnl
        max_drawdown = max(drawdowns) if len(drawdowns) > 0 else 0
        
        avg_trade_pnl = np.mean(pnls) if pnls else 0
        
        if len(pnls) > 1 and np.std(pnls) > 0:
            sharpe_ratio = np.mean(pnls) / np.std(pnls) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        first_trade = trades[0]
        last_trade = trades[-1]
        
        return BacktestResult(
            strategy_id=strategy_id,
            symbol=first_trade.symbol,
            start_time=first_trade.timestamp,
            end_time=last_trade.timestamp,
            bars_count=len(self._price_history.get(strategy_id, [])),
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            max_pnl=max_pnl,
            min_pnl=min(cumulative_pnl) if len(cumulative_pnl) > 0 else 0,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_pnl=avg_trade_pnl,
            sharpe_ratio=sharpe_ratio
        )
    
    def _evaluate_strategy_health(self, result: BacktestResult, strategy_id: str) -> None:
        """评估策略健康状态"""
        health = StrategyHealth.HEALTHY
        freeze_reason = ""
        
        t = self._thresholds
        
        consecutive_losses = self._consecutive_losses.get(strategy_id, 0)
        
        warnings = []
        
        if result.win_rate < t.min_win_rate:
            health = StrategyHealth.CRITICAL
            freeze_reason = f"胜率过低 ({result.win_rate*100:.1f}% < {t.min_win_rate*100:.1f}%)"
        elif result.win_rate < t.warning_win_rate:
            warnings.append(f"胜率较低 ({result.win_rate*100:.1f}%)")
            health = StrategyHealth.WARNING
        
        if result.profit_factor < t.min_profit_factor:
            health = StrategyHealth.CRITICAL
            freeze_reason = f"盈亏比过低 ({result.profit_factor:.2f} < {t.min_profit_factor:.2f})"
        elif result.profit_factor < t.warning_profit_factor:
            warnings.append(f"盈亏比较低 ({result.profit_factor:.2f})")
            if health == StrategyHealth.HEALTHY:
                health = StrategyHealth.WARNING
        
        if result.max_drawdown > t.max_drawdown_pct:
            health = StrategyHealth.CRITICAL
            freeze_reason = f"回撤过大 ({result.max_drawdown*100:.1f}% > {t.max_drawdown_pct*100:.1f}%)"
        elif result.max_drawdown > t.warning_drawdown_pct:
            warnings.append(f"回撤较大 ({result.max_drawdown*100:.1f}%)")
            if health == StrategyHealth.HEALTHY:
                health = StrategyHealth.WARNING
        
        if consecutive_losses >= t.max_consecutive_losses:
            health = StrategyHealth.CRITICAL
            freeze_reason = f"连续亏损次数过多 ({consecutive_losses}次)"
        
        result.health = health
        result.freeze_reason = freeze_reason
        
        if health == StrategyHealth.CRITICAL and self._freeze_callback:
            logger.warning(f"Strategy {strategy_id} is CRITICAL: {freeze_reason}")
            # P2-12: 冻结审计留痕
            try:
                from core.strategy_audit import get_strategy_audit_logger
                get_strategy_audit_logger().log("freeze", strategy_id, reason=freeze_reason)
            except Exception as e:
                logger.debug(f"MiniBacktester freeze audit error: {e}")
            try:
                self._freeze_callback(strategy_id, freeze_reason)
            except Exception as e:
                logger.error(f"Error in freeze callback: {e}")
        
        if warnings:
            logger.info(f"Strategy {strategy_id} warnings: {'; '.join(warnings)}")
    
    def get_backtest_result(self, strategy_id: str) -> Optional[BacktestResult]:
        """获取回测结果"""
        with self._lock:
            return self._backtest_results.get(strategy_id)
    
    def get_all_results(self) -> Dict[str, BacktestResult]:
        """获取所有回测结果"""
        with self._lock:
            return self._backtest_results.copy()
    
    def get_strategies_by_health(self, health: StrategyHealth) -> List[str]:
        """获取指定健康状态的策略"""
        with self._lock:
            return [
                sid for sid, result in self._backtest_results.items()
                if result.health == health
            ]
    
    def run_periodic_backtest(self) -> Dict[str, BacktestResult]:
        """运行所有策略的回测"""
        results = {}
        
        with self._lock:
            strategy_ids = list(self._trade_history.keys())
        
        for strategy_id in strategy_ids:
            try:
                result = self.run_backtest(strategy_id)
                if result:
                    results[strategy_id] = result
            except Exception as e:
                logger.error(f"Error running backtest for {strategy_id}: {e}")
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "strategies_registered": len(self._price_history),
                "backtests_run": self._test_count,
                "avg_backtest_time_ms": round(self._total_test_time / max(self._test_count, 1) * 1000, 2),
                "healthy_count": len(self.get_strategies_by_health(StrategyHealth.HEALTHY)),
                "warning_count": len(self.get_strategies_by_health(StrategyHealth.WARNING)),
                "critical_count": len(self.get_strategies_by_health(StrategyHealth.CRITICAL)),
                "frozen_count": len(self.get_strategies_by_health(StrategyHealth.FROZEN))
            }
    
    def clear_strategy(self, strategy_id: str) -> None:
        """清除策略数据"""
        with self._lock:
            self._price_history.pop(strategy_id, None)
            self._trade_history.pop(strategy_id, None)
            self._backtest_results.pop(strategy_id, None)
            self._consecutive_losses.pop(strategy_id, None)
    
    def clear_all(self) -> None:
        """清除所有数据"""
        with self._lock:
            self._price_history.clear()
            self._trade_history.clear()
            self._backtest_results.clear()
            self._consecutive_losses.clear()


class BacktestAdapter:
    """P22: 回测适配器 - 严格隔离回测与实盘代码
    
    生产级要求：
    1. 回测模式下策略禁止调用实盘API（下单、查询持仓、查询账户）
    2. 提供历史数据源替代实时行情
    3. 信号捕获不执行真实订单
    4. 明文标注回测模式，防止误操作
    
    使用方式：
        adapter = BacktestAdapter(strategy, historical_data)
        adapter.set_backtest_mode(True)
        signal = await adapter.run_on_bar(bar_data)
    """
    
    def __init__(self, strategy: Any, historical_data: List[Dict[str, Any]] = None):
        self._strategy = strategy
        self._historical_data = historical_data or []
        self._captured_signals: List[Dict[str, Any]] = []
        self._backtest_results: Dict[str, Any] = {}
        self._mode = "backtest"  # "backtest" | "paper" | "live"
        
        # 标记策略为回测模式，禁用实盘API
        if hasattr(strategy, 'set_backtest_mode'):
            strategy.set_backtest_mode(True)
    
    def set_mode(self, mode: str) -> None:
        """设置运行模式: backtest / paper / live"""
        valid_modes = {"backtest", "paper", "live"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        self._mode = mode
        if hasattr(self._strategy, 'set_backtest_mode'):
            self._strategy.set_backtest_mode(mode == "backtest")
        if hasattr(self._strategy, 'set_paper_trading_mode'):
            self._strategy.set_paper_trading_mode(mode == "paper")
    
    async def run_on_bar(self, bar_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """P22: 在历史K线上运行策略，捕获信号但不执行"""
        if self._mode == "live":
            raise RuntimeError(
                "P22: BacktestAdapter is for backtest/paper mode only. "
                "Use StrategyContainer for live trading."
            )
        
        try:
            signal = await self._strategy.on_bar(bar_data)
            if signal:
                self._captured_signals.append(signal)
            return signal
        except RuntimeError as e:
            if "BACKTEST SAFETY BLOCK" in str(e):
                logger.error(f"P22: Strategy attempted live API call in backtest mode: {e}")
            raise
    
    async def run_on_tick(self, tick_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """P22: 在历史tick上运行策略"""
        if self._mode == "live":
            raise RuntimeError("P22: BacktestAdapter is for backtest/paper mode only.")
        
        try:
            signal = await self._strategy.on_tick(tick_data)
            if signal:
                self._captured_signals.append(signal)
            return signal
        except RuntimeError as e:
            if "BACKTEST SAFETY BLOCK" in str(e):
                logger.error(f"P22: Strategy attempted live API call in backtest mode: {e}")
            raise
    
    async def run_backtest(self) -> Dict[str, Any]:
        """P22: 运行完整历史回测"""
        if not self._historical_data:
            return {"error": "No historical data provided"}
        
        self._captured_signals.clear()
        
        for bar in self._historical_data:
            try:
                await self.run_on_bar(bar)
            except Exception as e:
                logger.warning(f"P22: Backtest error on bar {bar.get('timestamp', '')}: {e}")
        
        self._backtest_results = {
            "mode": self._mode,
            "bars_processed": len(self._historical_data),
            "signals_generated": len(self._captured_signals),
            "signals": self._captured_signals,
        }
        return self._backtest_results
    
    def get_signals(self) -> List[Dict[str, Any]]:
        """获取回测中捕获的所有信号"""
        return self._captured_signals
    
    def reset(self) -> None:
        """重置回测状态"""
        self._captured_signals.clear()
        self._backtest_results.clear()


_backtester_instance: Optional[MiniBacktester] = None

def get_mini_backtester(config: Dict[str, Any] = None) -> MiniBacktester:
    """获取迷你回测器单例"""
    global _backtester_instance
    if _backtester_instance is None:
        _backtester_instance = MiniBacktester(config)
    return _backtester_instance