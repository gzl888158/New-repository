"""
策略计算引擎核心入口 (生产级 v2.0)
================================
整合指标计算、信号生成、仓位规划、策略容器、迷你回测的统一接口，
并集成完整生产管线：资金管理、风险监控、交易会话、指标流水线、通知分发。

核心定位：高波动捕捉、多策略并行运算，独立算力隔离，策略互不干扰

运行模式管控：
- NORMAL: 正常模式，所有策略全开
- CONSERVATIVE: 保守模式，高风险策略降权，低杠杆优先
- AGGRESSIVE: 激进模式，提高信号灵敏度，增大仓位
- EMERGENCY: 紧急模式，只允许平仓/止损，禁止开新仓

生产管线集成：
- CapitalManager: 开仓前资金校验、杠杆合规检查、磨损预算控制
- RiskMonitor: 实时风险评分、信号级风险过滤
- TradingSessionManager: 会话状态感知、非交易时段信号抑制
- MetricsPipeline: 信号计数、延迟记录、策略性能指标
- NotificationDispatcher: 重要信号通知、模式切换告警
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple, Callable, TYPE_CHECKING
from datetime import datetime
from enum import Enum
from loguru import logger
import threading

from .indicator_engine import IndicatorEngine, IndicatorSet, get_indicator_engine
from .signal_generator import SignalGenerator, TradingSignal, SignalContext, SignalType, get_signal_generator
from .position_planner import PositionPlanner, PositionPlan, get_position_planner
from .strategy_container import (
    StrategyContainer, StrategyInstance, StrategyType, StrategyState,
    get_strategy_container
)
from .mini_backtester import MiniBacktester, BacktestResult, get_mini_backtester

if TYPE_CHECKING:
    from .capital_manager import CapitalManager
    from .risk_monitor import RealTimeRiskMonitor
    from .trading_session import TradingSessionManager
    from .metrics_pipeline import MetricsPipeline
    from .notification_dispatcher import NotificationDispatcher


class RunMode(Enum):
    """策略运行模式"""
    NORMAL = "normal"           # 正常模式：所有策略全开
    CONSERVATIVE = "conservative"  # 保守模式：高风险策略降权，低杠杆优先
    AGGRESSIVE = "aggressive"   # 激进模式：提高信号灵敏度，增大仓位
    EMERGENCY = "emergency"     # 紧急模式：只允许平仓/止损，禁止开新仓


class StrategyEngine:
    """
    策略计算引擎
    
    核心功能：
    1. 多策略并行计算（趋势突破、网格震荡、波动率套利、波段反转）
    2. 指标高速计算（RSI、MACD、布林带、ATR等）
    3. 信号分级输出（开多/开空/加仓/减仓/止盈/止损/对冲）
    4. 仓位动态规划（梯度加仓、杠杆控制）
    5. 盘中滚动回测（失效自动冻结）
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self._indicator_engine = get_indicator_engine(config)
        self._signal_generator = get_signal_generator(config)
        self._position_planner = get_position_planner(config)
        self._strategy_container = get_strategy_container(config)
        self._mini_backtester = get_mini_backtester(config)
        
        self._setup_backtest_callback()
        
        self._market_data: Dict[str, Dict[str, Any]] = {}
        self._positions: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        
        self._trade_callback: Optional[Callable] = None
        self._signal_callback: Optional[Callable] = None
        
        self._running = False
        self._backtest_task: Optional[asyncio.Task] = None
        
        # ── 生产管线集成 (v2.0) ──
        self._capital_manager: Optional['CapitalManager'] = None
        self._risk_monitor: Optional['RealTimeRiskMonitor'] = None
        self._trading_session: Optional['TradingSessionManager'] = None
        self._metrics_pipeline: Optional['MetricsPipeline'] = None
        self._notification_dispatcher: Optional['NotificationDispatcher'] = None
        
        # 管线启用开关
        engine_config = config.get("strategy_engine", {})
        self._pipeline_enabled = engine_config.get("production_pipeline_enabled", True)
        self._pre_trade_risk_check = engine_config.get("pre_trade_risk_check", True)
        self._capital_check_enabled = engine_config.get("capital_check_enabled", True)
        self._session_check_enabled = engine_config.get("session_check_enabled", True)
        self._metrics_enabled = engine_config.get("metrics_enabled", True)
        self._notifications_enabled = engine_config.get("notifications_enabled", True)
        
        # 风险阈值
        self._max_risk_score_for_open = engine_config.get("max_risk_score_for_open", 0.7)
        self._max_risk_score_for_trade = engine_config.get("max_risk_score_for_trade", 0.85)
        
        # 指标统计
        self._signal_stats: Dict[str, int] = {"total": 0, "filtered": 0, "executed": 0}
        self._last_metrics_report_time = 0.0
        
        # 运行模式管控
        engine_config = self.config.get("strategy_engine", {})
        try:
            self._run_mode = RunMode(engine_config.get("default_mode", "normal"))
        except ValueError:
            logger.warning(f"Invalid default_mode '{engine_config.get('default_mode')}', falling back to NORMAL")
            self._run_mode = RunMode.NORMAL
        self._mode_transition_history: List[Dict[str, Any]] = []
        
        # 模式切换阈值
        self._mode_thresholds = {
            "market_drawdown_aggressive": engine_config.get("market_drawdown_aggressive", 0.05),
            "market_drawdown_conservative": engine_config.get("market_drawdown_conservative", 0.10),
            "market_drawdown_emergency": engine_config.get("market_drawdown_emergency", 0.18),
            "vol_spike_to_conservative": engine_config.get("vol_spike_to_conservative", 0.85),
            "vol_calm_to_aggressive": engine_config.get("vol_calm_to_aggressive", 0.25),
            "daily_loss_to_conservative_pct": engine_config.get("daily_loss_to_conservative_pct", 0.03),
            "daily_loss_to_emergency_pct": engine_config.get("daily_loss_to_emergency_pct", 0.08),
        }
        
        # 模式对应的信号权重乘数
        self._mode_signal_multipliers = {
            RunMode.NORMAL: 1.0,
            RunMode.CONSERVATIVE: 0.7,
            RunMode.AGGRESSIVE: 1.2,
            RunMode.EMERGENCY: 0.0,  # 紧急模式不产生开仓信号
        }
        
        # 模式对应的仓位乘数
        self._mode_position_multipliers = {
            RunMode.NORMAL: 1.0,
            RunMode.CONSERVATIVE: 0.6,
            RunMode.AGGRESSIVE: 1.3,
            RunMode.EMERGENCY: 0.0,
        }
        
        logger.info(f"StrategyEngine initialized (mode={self._run_mode.value}, pipeline={self._pipeline_enabled})")
    
    # ═══════════════════════════════════════════════════════════════
    # 生产管线依赖注入 (v2.0)
    # ═══════════════════════════════════════════════════════════════
    
    def inject_dependencies(self,
                            capital_manager: Optional['CapitalManager'] = None,
                            risk_monitor: Optional['RealTimeRiskMonitor'] = None,
                            trading_session: Optional['TradingSessionManager'] = None,
                            metrics_pipeline: Optional['MetricsPipeline'] = None,
                            notification_dispatcher: Optional['NotificationDispatcher'] = None) -> None:
        """注入生产管线依赖（可选，支持渐进式集成）"""
        if capital_manager:
            self._capital_manager = capital_manager
            logger.info("StrategyEngine: CapitalManager injected")
        if risk_monitor:
            self._risk_monitor = risk_monitor
            logger.info("StrategyEngine: RiskMonitor injected")
        if trading_session:
            self._trading_session = trading_session
            logger.info("StrategyEngine: TradingSessionManager injected")
        if metrics_pipeline:
            self._metrics_pipeline = metrics_pipeline
            logger.info("StrategyEngine: MetricsPipeline injected")
        if notification_dispatcher:
            self._notification_dispatcher = notification_dispatcher
            logger.info("StrategyEngine: NotificationDispatcher injected")
    
    def _is_pipeline_ready(self) -> bool:
        """检查生产管线是否就绪"""
        return self._pipeline_enabled
    
    # ═══════════════════════════════════════════════════════════════
    # 生产管线检查方法 (v2.0)
    # ═══════════════════════════════════════════════════════════════
    
    def _check_trading_session(self) -> bool:
        """检查交易会话是否允许交易"""
        if not self._session_check_enabled or not self._trading_session:
            return True
        try:
            state = self._trading_session.get_state()
            # 非交易状态（STOPPED/PAUSED/EMERGENCY）禁止开仓
            if state in ("stopped", "emergency"):
                return False
            return True
        except Exception as e:
            logger.debug(f"Session check error: {e}")
            return True  # 容错：检查失败时允许交易
    
    def _check_risk_gate(self, signal_type: 'SignalType', symbol: str) -> Tuple[bool, str]:
        """
        风险门控检查
        
        Returns:
            (是否通过, 拒绝原因)
        """
        if not self._pre_trade_risk_check or not self._risk_monitor:
            return True, ""
        
        try:
            snapshot = self._risk_monitor.get_risk_snapshot()
            overall_score = snapshot.get("overall_score", 0)
            
            # 开仓信号需要更严格的风险检查
            from .signal_generator import SignalType as ST
            is_open_signal = signal_type in (
                ST.OPEN_LONG, ST.OPEN_SHORT, ST.ADD_POSITION
            )
            
            threshold = self._max_risk_score_for_open if is_open_signal else self._max_risk_score_for_trade
            
            if overall_score >= threshold:
                return False, f"risk_score={overall_score:.2f} >= {threshold}"
            
            # 检查特定风险维度（get_risk_snapshot 返回的键为 "dimensions"，而非 "scores"）
            scores = snapshot.get("dimensions", snapshot.get("scores", {}))
            for dim, score_data in scores.items():
                if isinstance(score_data, dict) and score_data.get("score", 0) >= 0.85:
                    return False, f"{dim}_risk={score_data['score']:.2f}"
            
            return True, ""
        except Exception as e:
            logger.debug(f"Risk gate check error: {e}")
            return True, ""  # 容错
    
    def _check_capital_availability(self, symbol: str, signal: 'TradingSignal') -> Tuple[bool, str]:
        """检查资金是否充足"""
        if not self._capital_check_enabled or not self._capital_manager:
            return True, ""
        
        try:
            from .signal_generator import SignalType as ST
            
            # 仅开仓信号需要资金检查
            if signal.signal_type not in (ST.OPEN_LONG, ST.OPEN_SHORT, ST.ADD_POSITION):
                return True, ""
            
            # 检查磨损预算
            strategy_name = signal.metadata.get("strategy_type", "unknown") if signal.metadata else "unknown"
            ok, reason = self._capital_manager.check_attrition_budget(strategy_name)
            if not ok:
                return False, f"attrition_budget_exceeded: {reason}"
            
            return True, ""
        except Exception as e:
            logger.debug(f"Capital check error: {e}")
            return True, ""  # 容错
    
    def _record_signal_metrics(self, symbol: str, signal: 'TradingSignal', 
                                filtered: bool = False, latency_ms: float = 0) -> None:
        """记录信号指标"""
        if not self._metrics_enabled or not self._metrics_pipeline:
            return
        
        try:
            labels = {
                "symbol": symbol,
                "signal_type": signal.signal_type.value if hasattr(signal.signal_type, 'value') else str(signal.signal_type),
                "strategy": signal.metadata.get("strategy_type", "unknown") if signal.metadata else "unknown",
            }
            
            self._metrics_pipeline.increment("signals_total", labels=labels)
            
            if filtered:
                self._metrics_pipeline.increment("signals_filtered", labels=labels)
            
            if latency_ms > 0:
                self._metrics_pipeline.record_latency("signal_process_ms", latency_ms, labels=labels)
        
        except Exception as e:
            logger.debug(f"Metrics recording error: {e}")
    
    async def _send_signal_notification(self, symbol: str, signal: 'TradingSignal') -> None:
        """发送重要信号通知"""
        if not self._notifications_enabled or not self._notification_dispatcher:
            return
        
        try:
            from .notification_dispatcher import NotificationChannel, NotificationPriority
            
            # 仅对高置信度信号发送通知
            if signal.weight < 0.7:
                return
            
            strategy = signal.metadata.get("strategy_type", "unknown") if signal.metadata else "unknown"
            direction = "LONG" if "long" in signal.signal_type.value.lower() else (
                "SHORT" if "short" in signal.signal_type.value.lower() else signal.signal_type.value.upper()
            )
            
            await self._notification_dispatcher.send_template(
                channel=NotificationChannel.CONSOLE,
                template="trade_signal",
                priority=NotificationPriority.INFO,
                category="trading",
                symbol=symbol,
                direction=direction,
                signal_type=f"{strategy}_{signal.signal_type.value}",
                price=signal.price,
                quantity=signal.quantity,
                confidence=signal.weight,
            )
        except Exception as e:
            logger.debug(f"Notification error: {e}")
    
    async def _notify_mode_change(self, old_mode: str, new_mode: str, reason: str) -> None:
        """通知运行模式变更"""
        if not self._notifications_enabled or not self._notification_dispatcher:
            return
        
        try:
            from .notification_dispatcher import NotificationChannel, NotificationPriority
            
            priority = NotificationPriority.WARNING
            if new_mode == "emergency":
                priority = NotificationPriority.EMERGENCY
            elif new_mode == "conservative":
                priority = NotificationPriority.CRITICAL
            
            await self._notification_dispatcher.send(
                channel=NotificationChannel.CONSOLE,
                title=f"RunMode: {old_mode} → {new_mode}",
                message=f"Strategy engine mode changed from {old_mode} to {new_mode}\nReason: {reason}",
                priority=priority,
                category="risk",
                metadata={"old_mode": old_mode, "new_mode": new_mode, "reason": reason},
            )
        except Exception as e:
            logger.debug(f"Mode notification error: {e}")
    
    # ═══════════════════════════════════════════════════════════════
    # 配置热更新 (v2.0)
    # ═══════════════════════════════════════════════════════════════
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        热更新配置 —— 重新加载引擎参数并分发到子模块
        
        支持运行时更新：
        - 运行模式阈值
        - 管线开关
        - 风险阈值
        - 信号生成器参数
        - 仓位规划器参数
        - 策略容器内各策略参数
        """
        old_config = self.config
        self.config = new_config
        
        # 更新引擎级配置
        engine_config = new_config.get("strategy_engine", {})
        self._pipeline_enabled = engine_config.get("production_pipeline_enabled", self._pipeline_enabled)
        self._pre_trade_risk_check = engine_config.get("pre_trade_risk_check", self._pre_trade_risk_check)
        self._capital_check_enabled = engine_config.get("capital_check_enabled", self._capital_check_enabled)
        self._session_check_enabled = engine_config.get("session_check_enabled", self._session_check_enabled)
        self._metrics_enabled = engine_config.get("metrics_enabled", self._metrics_enabled)
        self._notifications_enabled = engine_config.get("notifications_enabled", self._notifications_enabled)
        self._max_risk_score_for_open = engine_config.get("max_risk_score_for_open", self._max_risk_score_for_open)
        self._max_risk_score_for_trade = engine_config.get("max_risk_score_for_trade", self._max_risk_score_for_trade)
        
        # 更新模式阈值
        self._mode_thresholds.update({
            "market_drawdown_aggressive": engine_config.get("market_drawdown_aggressive", self._mode_thresholds["market_drawdown_aggressive"]),
            "market_drawdown_conservative": engine_config.get("market_drawdown_conservative", self._mode_thresholds["market_drawdown_conservative"]),
            "market_drawdown_emergency": engine_config.get("market_drawdown_emergency", self._mode_thresholds["market_drawdown_emergency"]),
            "vol_spike_to_conservative": engine_config.get("vol_spike_to_conservative", self._mode_thresholds["vol_spike_to_conservative"]),
            "vol_calm_to_aggressive": engine_config.get("vol_calm_to_aggressive", self._mode_thresholds["vol_calm_to_aggressive"]),
            "daily_loss_to_conservative_pct": engine_config.get("daily_loss_to_conservative_pct", self._mode_thresholds["daily_loss_to_conservative_pct"]),
            "daily_loss_to_emergency_pct": engine_config.get("daily_loss_to_emergency_pct", self._mode_thresholds["daily_loss_to_emergency_pct"]),
        })
        
        # 分发到信号生成器
        try:
            self._signal_generator.update_config(new_config)
        except Exception as e:
            logger.warning(f"Failed to update signal_generator config: {e}")
        
        # 分发到仓位规划器
        try:
            self._position_planner.update_config(new_config)
        except Exception as e:
            logger.warning(f"Failed to update position_planner config: {e}")
        
        # 分发到策略容器（各策略实例）
        try:
            self._strategy_container.update_config(new_config)
        except Exception as e:
            logger.warning(f"Failed to update strategy_container config: {e}")
        
        logger.info(
            f"StrategyEngine config hot-updated: "
            f"pipeline={self._pipeline_enabled}, "
            f"risk_check={self._pre_trade_risk_check}, "
            f"capital_check={self._capital_check_enabled}"
        )
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """获取信号统计"""
        return {
            **self._signal_stats,
            "filter_rate": self._signal_stats["filtered"] / max(self._signal_stats["total"], 1),
            "run_mode": self._run_mode.value,
            "pipeline_enabled": self._pipeline_enabled,
        }
    
    def _setup_backtest_callback(self) -> None:
        """设置回测冻结回调"""
        def on_strategy_freeze(strategy_id: str, reason: str) -> None:
            self._strategy_container.freeze_instance(strategy_id, reason)
            logger.warning(f"Strategy {strategy_id} frozen by backtester: {reason}")

        self._mini_backtester.set_freeze_callback(on_strategy_freeze)

    def set_compute_scheduler(self, scheduler) -> None:
        """注入算力调度器到 IndicatorEngine + StrategyContainer（统一入口）"""
        try:
            self._indicator_engine.set_compute_scheduler(scheduler)
        except Exception as e:
            logger.warning(f"Failed to inject ComputeScheduler into IndicatorEngine: {e}")
        try:
            self._strategy_container.set_compute_scheduler(scheduler)
        except Exception as e:
            logger.warning(f"Failed to inject ComputeScheduler into StrategyContainer: {e}")
        # 注入指标引擎到策略容器（使子策略能使用实时指标）
        try:
            self._strategy_container.set_indicator_engine(self._indicator_engine)
        except Exception as e:
            logger.warning(f"Failed to inject IndicatorEngine into StrategyContainer: {e}")
        # 为所有已注册symbol登记到调度器
        try:
            for symbol in self._indicator_engine._symbol_caches.keys():
                scheduler.register_symbol(symbol)
        except Exception as e:
            logger.debug(f"ComputeScheduler symbol registration error: {e}")
        logger.info("ComputeScheduler injected into StrategyEngine (IndicatorEngine + StrategyContainer)")
    
    def set_run_mode(self, mode: RunMode, reason: str = "") -> None:
        """手动设置运行模式（带通知）"""
        if mode == self._run_mode:
            return
        old_mode = self._run_mode
        self._run_mode = mode
        transition = {
            "from": old_mode.value,
            "to": mode.value,
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }
        self._mode_transition_history.append(transition)
        if len(self._mode_transition_history) > 50:
            self._mode_transition_history = self._mode_transition_history[-50:]
        logger.warning(f"RunMode changed: {old_mode.value} -> {mode.value} (reason: {reason})")
        
        # 发送通知
        try:
            asyncio.create_task(self._notify_mode_change(old_mode.value, mode.value, reason))
        except Exception:
            pass

    def get_run_mode(self) -> RunMode:
        """获取当前运行模式"""
        return self._run_mode

    def evaluate_mode_transition(self, market_conditions: Dict[str, Any]) -> Optional[RunMode]:
        """
        根据市场状况自动评估是否需要切换运行模式
        
        Args:
            market_conditions: 包含 market_drawdown, volatility_percentile, daily_loss_pct 等
        
        Returns:
            建议的新模式，如果无需切换则返回None
        """
        drawdown = market_conditions.get("market_drawdown", 0)
        vol_pct = market_conditions.get("volatility_percentile", 0.5)
        daily_loss = market_conditions.get("daily_loss_pct", 0)
        is_circuit_breaker = market_conditions.get("circuit_breaker_triggered", False)
        is_price_anomaly = market_conditions.get("price_anomaly", False)
        
        t = self._mode_thresholds
        
        # EMERGENCY: 熔断触发 / 价格异常 / 回撤超过紧急阈值 / 日亏损超过紧急阈值
        if is_circuit_breaker or is_price_anomaly:
            return RunMode.EMERGENCY
        if drawdown > t["market_drawdown_emergency"]:
            return RunMode.EMERGENCY
        if daily_loss > t["daily_loss_to_emergency_pct"]:
            return RunMode.EMERGENCY
        
        # CONSERVATIVE: 高波动 / 回撤超过保守阈值 / 日亏损超保守阈值
        if vol_pct > t["vol_spike_to_conservative"]:
            return RunMode.CONSERVATIVE
        if drawdown > t["market_drawdown_conservative"]:
            return RunMode.CONSERVATIVE
        if daily_loss > t["daily_loss_to_conservative_pct"]:
            return RunMode.CONSERVATIVE
        
        # AGGRESSIVE: 低波动环境（趋势/震荡有利可图）
        if self._run_mode in (RunMode.CONSERVATIVE, RunMode.EMERGENCY):
            return None  # 不从紧急/保守自动切换为激进，需手动恢复
        if vol_pct < t["vol_calm_to_aggressive"] and drawdown < t["market_drawdown_aggressive"]:
            return RunMode.AGGRESSIVE
        
        # 回到正常
        if self._run_mode == RunMode.AGGRESSIVE and (vol_pct > 0.3 or drawdown > t["market_drawdown_aggressive"]):
            return RunMode.NORMAL
        
        return None

    def auto_adjust_mode(self, market_conditions: Dict[str, Any]) -> bool:
        """自动评估并切换运行模式，返回是否发生了切换"""
        suggested = self.evaluate_mode_transition(market_conditions)
        if suggested and suggested != self._run_mode:
            self.set_run_mode(suggested, f"auto: market_conditions={market_conditions}")
            return True
        return False

    def _get_mode_signal_multiplier(self) -> float:
        """获取当前模式的信号权重乘数"""
        return self._mode_signal_multipliers.get(self._run_mode, 1.0)

    def _get_mode_position_multiplier(self) -> float:
        """获取当前模式的仓位乘数"""
        return self._mode_position_multipliers.get(self._run_mode, 1.0)

    def _is_open_signal_allowed(self) -> bool:
        """当前模式是否允许开仓信号"""
        return self._run_mode != RunMode.EMERGENCY

    def _apply_mode_filter_to_signal(self, signal: 'TradingSignal') -> Optional['TradingSignal']:
        """
        根据当前运行模式过滤/调整信号
        
        Returns:
            调整后的信号，如果信号被完全过滤则返回None
        """
        from .signal_generator import SignalType
        
        mode = self._run_mode
        
        # EMERGENCY模式：只允许平仓/止损类信号
        if mode == RunMode.EMERGENCY:
            allowed_types = {
                SignalType.CLOSE_ALL, SignalType.STOP_LOSS,
                SignalType.TAKE_PROFIT, SignalType.REDUCE_POSITION
            }
            if signal.signal_type not in allowed_types:
                logger.debug(f"Signal filtered by EMERGENCY mode: {signal.symbol} {signal.signal_type.value}")
                return None
        
        # CONSERVATIVE模式：降低风险信号权重，过滤高风险对冲信号
        if mode == RunMode.CONSERVATIVE:
            if signal.signal_type == SignalType.HEDGE:
                signal.weight *= 0.5
            
        # 应用模式信号乘数
        multiplier = self._mode_signal_multipliers.get(mode, 1.0)
        signal.weight *= multiplier
        
        # 弱信号在保守模式下过滤
        if mode == RunMode.CONSERVATIVE and signal.weight < 0.35:
            return None
        
        return signal

    def initialize(self) -> None:
        """初始化引擎"""
        symbols = self.config.get("symbols", [])
        for symbol in symbols:
            self._indicator_engine.register_symbol(symbol)
        
        self._setup_default_strategies()
        
        logger.info(f"StrategyEngine initialized for {len(symbols)} symbols")
    
    def _setup_default_strategies(self) -> None:
        """设置默认策略实例"""
        symbols = self.config.get("symbols", [])
        strategy_types = [
            StrategyType.TREND_BREAKOUT,
            StrategyType.GRID_OSCILLATION,
            StrategyType.VOLATILITY_ARB,
            StrategyType.BAND_REVERSAL
        ]
        
        for symbol in symbols:
            for strategy_type in strategy_types:
                try:
                    instance = self._strategy_container.create_instance(
                        strategy_type, symbol, self.config.get("strategy_config", {})
                    )
                    self._strategy_container.start_instance(instance.id)
                    self._mini_backtester.register_strategy(instance.id, symbol)
                except Exception as e:
                    logger.error(f"Failed to create {strategy_type.value} for {symbol}: {e}")
    
    def register_symbol(self, symbol: str) -> None:
        """注册交易对"""
        self._indicator_engine.register_symbol(symbol)
        
        if symbol not in self._market_data:
            self._market_data[symbol] = {}
    
    def update_market_data(self, symbol: str, data: Dict[str, Any]) -> None:
        """更新市场数据（带异常保护）"""
        if not symbol or not data:
            return
        try:
            with self._lock:
                self._market_data[symbol] = data
            
            self._indicator_engine.update_market_data(
                symbol=symbol,
                open_price=data.get("open", data.get("o", 0)),
                high=data.get("high", data.get("h", 0)),
                low=data.get("low", data.get("l", 0)),
                close=data.get("close", data.get("c", 0)),
                volume=data.get("volume", data.get("v", 0))
            )
        except Exception as e:
            logger.error(f"StrategyEngine update_market_data error for {symbol}: {e}")
    
    async def process_tick(self, symbol: str, tick: Dict[str, Any]) -> List[TradingSignal]:
        """
        处理tick数据 (生产级 v2.0)
        
        流程：会话检查 → 指标计算 → 信号生成 → 风险门控 → 资金检查 → 模式过滤 → 指标记录 → 通知 → 回调分发
        
        Returns:
            信号列表
        """
        t_start = time.time()
        
        if not symbol or not tick:
            logger.debug(f"StrategyEngine process_tick: invalid input symbol={symbol}, tick={bool(tick)}")
            return []

        try:
            # P0: 会话状态检查
            if not self._check_trading_session():
                logger.debug(f"StrategyEngine process_tick: trading session not active, skipping {symbol}")
                return []
            
            price = tick.get("price", tick.get("last", 0))
            if not isinstance(price, (int, float)) or price <= 0:
                logger.debug(f"StrategyEngine process_tick: invalid price={price} for {symbol}")
                return []

            # P0: 将tick数据推入IndicatorEngine，确保指标计算使用最新数据
            try:
                self.update_market_data(symbol, tick)
            except Exception as e:
                logger.error(f"StrategyEngine process_tick: update_market_data failed for {symbol}: {e}")

            indicator_set = self._indicator_engine.calculate_all(symbol)

            context = self._build_signal_context(symbol, price, indicator_set)

            # 自动评估模式切换
            try:
                self._evaluate_and_adjust_mode(context)
            except Exception as e:
                logger.debug(f"StrategyEngine process_tick: mode evaluation error for {symbol}: {e}")

            signals = []
            try:
                primary_signal = self._signal_generator.generate_primary_signal(context)
            except Exception as e:
                logger.error(f"StrategyEngine process_tick: signal generation failed for {symbol}: {e}")
                primary_signal = None

            if primary_signal:
                self._signal_stats["total"] += 1
                
                # 风险门控检查
                risk_ok, risk_reason = self._check_risk_gate(primary_signal.signal_type, symbol)
                if not risk_ok:
                    self._signal_stats["filtered"] += 1
                    self._record_signal_metrics(symbol, primary_signal, filtered=True)
                    logger.debug(f"Signal filtered by risk gate: {symbol} {primary_signal.signal_type.value} - {risk_reason}")
                else:
                    # 资金检查
                    capital_ok, capital_reason = self._check_capital_availability(symbol, primary_signal)
                    if not capital_ok:
                        self._signal_stats["filtered"] += 1
                        self._record_signal_metrics(symbol, primary_signal, filtered=True)
                        logger.debug(f"Signal filtered by capital check: {symbol} {primary_signal.signal_type.value} - {capital_reason}")
                    else:
                        filtered = self._apply_mode_filter_to_signal(primary_signal)
                        if filtered:
                            signals.append(filtered)
                            self._signal_stats["executed"] += 1
                            self._record_signal_metrics(symbol, filtered)
                        else:
                            self._signal_stats["filtered"] += 1
                            self._record_signal_metrics(symbol, primary_signal, filtered=True)

            try:
                container_signals = await self._strategy_container.process_tick(symbol, tick)
            except Exception as e:
                logger.error(f"StrategyEngine process_tick: strategy_container failed for {symbol}: {e}")
                container_signals = None

            if container_signals:
                for sig in container_signals:
                    try:
                        trading_signal = self._convert_to_trading_signal(symbol, sig, indicator_set)
                        if trading_signal:
                            self._signal_stats["total"] += 1
                            
                            # 风险门控检查
                            risk_ok, risk_reason = self._check_risk_gate(trading_signal.signal_type, symbol)
                            if not risk_ok:
                                self._signal_stats["filtered"] += 1
                                self._record_signal_metrics(symbol, trading_signal, filtered=True)
                                continue
                            
                            # 资金检查
                            capital_ok, capital_reason = self._check_capital_availability(symbol, trading_signal)
                            if not capital_ok:
                                self._signal_stats["filtered"] += 1
                                self._record_signal_metrics(symbol, trading_signal, filtered=True)
                                continue
                            
                            filtered = self._apply_mode_filter_to_signal(trading_signal)
                            if filtered:
                                signals.append(filtered)
                                self._signal_stats["executed"] += 1
                                self._record_signal_metrics(symbol, filtered)
                            else:
                                self._signal_stats["filtered"] += 1
                                self._record_signal_metrics(symbol, trading_signal, filtered=True)
                    except Exception as e:
                        logger.debug(f"StrategyEngine process_tick: signal conversion error for {symbol}: {e}")

            if signals and self._signal_callback:
                for signal in signals:
                    try:
                        self._signal_callback(signal)
                    except Exception as e:
                        logger.error(f"Error in signal callback: {e}")

            # 发送重要信号通知
            for signal in signals:
                try:
                    await self._send_signal_notification(symbol, signal)
                except Exception:
                    pass

            # 记录处理延迟
            latency_ms = (time.time() - t_start) * 1000
            if self._metrics_enabled and self._metrics_pipeline:
                try:
                    self._metrics_pipeline.record_latency("process_tick_ms", latency_ms, {"symbol": symbol})
                except Exception:
                    pass

            return signals

        except Exception as e:
            logger.error(f"StrategyEngine process_tick: unhandled error for {symbol}: {e}", exc_info=True)
            return []
    
    async def process_bar(self, symbol: str, bar: Dict[str, Any]) -> List[TradingSignal]:
        """
        处理K线数据 (生产级 v2.0)
        
        流程：会话检查 → 指标计算 → 组合信号 → 风险门控 → 资金检查 → 模式过滤 → 回测推送 → 指标记录 → 通知 → 回调分发
        
        Returns:
            信号列表
        """
        t_start = time.time()
        
        if not symbol or not bar:
            logger.debug(f"StrategyEngine process_bar: invalid input symbol={symbol}, bar={bool(bar)}")
            return []

        try:
            # P0: 会话状态检查
            if not self._check_trading_session():
                logger.debug(f"StrategyEngine process_bar: trading session not active, skipping {symbol}")
                return []
            
            price = bar.get("close", bar.get("c", 0))
            if not isinstance(price, (int, float)) or price <= 0:
                logger.debug(f"StrategyEngine process_bar: invalid close price={price} for {symbol}")
                return []

            try:
                self.update_market_data(symbol, bar)
            except Exception as e:
                logger.error(f"StrategyEngine process_bar: update_market_data failed for {symbol}: {e}")

            indicator_set = self._indicator_engine.calculate_all(symbol)

            context = self._build_signal_context(symbol, price, indicator_set)

            # 自动评估模式切换
            try:
                self._evaluate_and_adjust_mode(context)
            except Exception as e:
                logger.debug(f"StrategyEngine process_bar: mode evaluation error for {symbol}: {e}")

            signals = []
            try:
                primary_signal = self._signal_generator.generate_combined_signal(context)
            except Exception as e:
                logger.error(f"StrategyEngine process_bar: combined signal generation failed for {symbol}: {e}")
                primary_signal = None

            if primary_signal:
                self._signal_stats["total"] += 1
                
                # 风险门控检查
                risk_ok, risk_reason = self._check_risk_gate(primary_signal.signal_type, symbol)
                if not risk_ok:
                    self._signal_stats["filtered"] += 1
                    self._record_signal_metrics(symbol, primary_signal, filtered=True)
                    logger.debug(f"Signal filtered by risk gate: {symbol} {primary_signal.signal_type.value} - {risk_reason}")
                else:
                    # 资金检查
                    capital_ok, capital_reason = self._check_capital_availability(symbol, primary_signal)
                    if not capital_ok:
                        self._signal_stats["filtered"] += 1
                        self._record_signal_metrics(symbol, primary_signal, filtered=True)
                        logger.debug(f"Signal filtered by capital check: {symbol} {primary_signal.signal_type.value} - {capital_reason}")
                    else:
                        filtered = self._apply_mode_filter_to_signal(primary_signal)
                        if filtered:
                            signals.append(filtered)
                            self._signal_stats["executed"] += 1
                            self._record_signal_metrics(symbol, filtered)
                        else:
                            self._signal_stats["filtered"] += 1
                            self._record_signal_metrics(symbol, primary_signal, filtered=True)

            try:
                container_signals = await self._strategy_container.process_bar(symbol, bar)
            except Exception as e:
                logger.error(f"StrategyEngine process_bar: strategy_container failed for {symbol}: {e}")
                container_signals = None

            if container_signals:
                for sig in container_signals:
                    try:
                        trading_signal = self._convert_to_trading_signal(symbol, sig, indicator_set)
                        if trading_signal:
                            self._signal_stats["total"] += 1
                            
                            # 风险门控检查
                            risk_ok, risk_reason = self._check_risk_gate(trading_signal.signal_type, symbol)
                            if not risk_ok:
                                self._signal_stats["filtered"] += 1
                                self._record_signal_metrics(symbol, trading_signal, filtered=True)
                                continue
                            
                            # 资金检查
                            capital_ok, capital_reason = self._check_capital_availability(symbol, trading_signal)
                            if not capital_ok:
                                self._signal_stats["filtered"] += 1
                                self._record_signal_metrics(symbol, trading_signal, filtered=True)
                                continue
                            
                            filtered = self._apply_mode_filter_to_signal(trading_signal)
                            if filtered:
                                signals.append(filtered)
                                self._signal_stats["executed"] += 1
                                self._record_signal_metrics(symbol, filtered)
                            else:
                                self._signal_stats["filtered"] += 1
                                self._record_signal_metrics(symbol, trading_signal, filtered=True)
                    except Exception as e:
                        logger.debug(f"StrategyEngine process_bar: signal conversion error for {symbol}: {e}")

            # 回测数据推送
            try:
                for instance in self._strategy_container.get_instances_by_symbol(symbol):
                    self._mini_backtester.add_bar(instance.id, bar)
            except Exception as e:
                logger.debug(f"StrategyEngine process_bar: backtest update error for {symbol}: {e}")

            if signals and self._signal_callback:
                for signal in signals:
                    try:
                        self._signal_callback(signal)
                    except Exception as e:
                        logger.error(f"Error in signal callback: {e}")

            # 发送重要信号通知
            for signal in signals:
                try:
                    await self._send_signal_notification(symbol, signal)
                except Exception:
                    pass

            # 记录处理延迟
            latency_ms = (time.time() - t_start) * 1000
            if self._metrics_enabled and self._metrics_pipeline:
                try:
                    self._metrics_pipeline.record_latency("process_bar_ms", latency_ms, {"symbol": symbol})
                except Exception:
                    pass

            return signals

        except Exception as e:
            logger.error(f"StrategyEngine process_bar: unhandled error for {symbol}: {e}", exc_info=True)
            return []
    
    def update_position(self, symbol: str, position: Dict[str, Any]) -> None:
        """更新仓位信息"""
        with self._lock:
            self._positions[symbol] = position
        
        self._strategy_container.update_position(symbol, position)
    
    def _build_signal_context(self, symbol: str, price: float, 
                                indicators: IndicatorSet) -> SignalContext:
        """构建信号上下文（带安全回退）"""
        position = self._positions.get(symbol, {}) if hasattr(self, '_positions') else {}
        
        indicators_dict = {}
        if indicators and hasattr(indicators, 'indicators'):
            try:
                indicators_dict = {
                    name: result.value 
                    for name, result in indicators.indicators.items()
                }
            except Exception as e:
                logger.debug(f"StrategyEngine _build_signal_context: indicator extraction error {e}")
        
        return SignalContext(
            symbol=symbol,
            current_price=price,
            position_side=position.get("side"),
            position_size=abs(position.get("size", 0)),
            entry_price=position.get("entry_price", 0),
            unrealized_pnl=position.get("unrealized_pnl", 0),
            unrealized_pnl_pct=position.get("unrealized_pnl_pct", 0),
            margin_used=position.get("margin", 0),
            leverage=position.get("leverage", 1),
            indicators=indicators_dict,
            market_state=self._determine_market_state(indicators),
            volatility_level=self._determine_volatility_level(indicators)
        )
    
    def _determine_market_state(self, indicators: IndicatorSet) -> str:
        """判断市场状态"""
        adx = indicators.get("adx")
        rsi = indicators.get("rsi")
        
        if adx and adx > 25:
            return "trending"
        elif rsi and (rsi < 30 or rsi > 70):
            return "extreme"
        else:
            return "ranging"
    
    def _determine_volatility_level(self, indicators: IndicatorSet) -> str:
        """判断波动率水平"""
        vol_pct = indicators.get("volatility_percentile")
        
        if vol_pct:
            if vol_pct > 0.8:
                return "high"
            elif vol_pct < 0.2:
                return "low"
        return "normal"

    def _evaluate_and_adjust_mode(self, context: SignalContext) -> None:
        """根据信号上下文自动评估并调整运行模式"""
        try:
            vol_pct = context.indicators.get("volatility_percentile", 0.5)
            market_conditions = {
                "volatility_percentile": vol_pct,
                "market_drawdown": abs(context.unrealized_pnl_pct) if context.position_side else 0,
                "daily_loss_pct": 0,
                "circuit_breaker_triggered": False,
                "price_anomaly": False,
            }
            self.auto_adjust_mode(market_conditions)
        except Exception as e:
            logger.debug(f"Mode auto-adjust evaluation error: {e}")
    
    def _convert_to_trading_signal(self, symbol: str, signal: Dict[str, Any],
                                     indicators: IndicatorSet) -> Optional[TradingSignal]:
        """转换信号格式（带异常保护）"""
        if not signal or not isinstance(signal, dict):
            return None
        try:
            from .signal_generator import SignalType, SignalSource, SignalLevel
            
            signal_type_map = {
                "open_long": SignalType.OPEN_LONG,
                "open_short": SignalType.OPEN_SHORT,
                "add_position": SignalType.ADD_POSITION,
                "reduce_position": SignalType.REDUCE_POSITION,
                "take_profit": SignalType.TAKE_PROFIT,
                "stop_loss": SignalType.STOP_LOSS,
                "hedge": SignalType.HEDGE,
            }
            
            signal_type_str = signal.get("type", "hold")
            if signal_type_str not in signal_type_map:
                return None
            
            signal_type = signal_type_map[signal_type_str]
            strength = signal.get("strength", 0.5)
            
            level = SignalLevel.STRONG if strength >= 0.8 else (
                SignalLevel.MEDIUM if strength >= 0.5 else SignalLevel.WEAK
            )
            
            return TradingSignal(
                symbol=symbol,
                signal_type=signal_type,
                source=SignalSource.MOMENTUM,
                level=level,
                weight=strength,
                price=signal.get("price", 0),
                quantity=signal.get("quantity", 0),
                timestamp=datetime.now(),
                reason=signal.get("reason", ""),
                metadata={"strategy_type": signal.get("strategy_type", "unknown")}
            )
        except Exception as e:
            logger.debug(f"StrategyEngine _convert_to_trading_signal error for {symbol}: {e}")
            return None
    
    def create_position_plan(self, symbol: str, side: str,
                              risk_level: str = "balanced") -> Optional[PositionPlan]:
        """创建仓位规划（带异常保护）"""
        try:
            from .position_planner import RiskLevel
            
            risk_map = {
                "conservative": RiskLevel.CONSERVATIVE,
                "balanced": RiskLevel.BALANCED,
                "aggressive": RiskLevel.AGGRESSIVE,
            }
            
            risk = risk_map.get(risk_level, RiskLevel.BALANCED)
            
            market_data = self._market_data.get(symbol, {})
            price = market_data.get("close", market_data.get("c", 0))
            
            if price <= 0:
                logger.debug(f"StrategyEngine create_position_plan: invalid price={price} for {symbol}")
                return None
            
            indicators = self._indicator_engine.calculate_all(symbol)
            atr = indicators.get("atr", price * 0.02)
            
            if atr <= 0:
                atr = price * 0.02
            
            plan = self._position_planner.calculate_initial_position(
                symbol=symbol,
                price=price,
                atr=atr,
                side=side,
                risk_level=risk
            )
            
            # 应用运行模式仓位乘数
            if plan:
                pos_mult = self._get_mode_position_multiplier()
                if pos_mult != 1.0:
                    plan.initial_size *= pos_mult
                    plan.total_size = plan.initial_size + sum(
                        lvl.get("size", 0) * pos_mult for lvl in plan.add_levels
                    )
                    for lvl in plan.add_levels:
                        lvl["size"] *= pos_mult
            
            return plan
        except Exception as e:
            logger.error(f"StrategyEngine create_position_plan error for {symbol}: {e}")
            return None
    
    def should_add_position(self, symbol: str, current_price: float,
                            current_size: float, pnl_pct: float) -> tuple:
        """判断是否应该加仓"""
        indicators = self._indicator_engine.calculate_all(symbol)
        atr = indicators.get("atr", 0)
        
        return self._position_planner.should_add_position(
            symbol=symbol,
            current_price=current_price,
            current_size=current_size,
            pnl_pct=pnl_pct,
            atr=atr
        )
    
    def start_periodic_backtest(self, interval_seconds: int = 60) -> None:
        """启动定期回测"""
        async def run_periodic_backtest():
            while self._running:
                try:
                    results = self._mini_backtester.run_periodic_backtest()
                    
                    frozen = [
                        sid for sid, r in results.items() 
                        if r.health.value == "critical"
                    ]
                    if frozen:
                        logger.warning(f"Strategies need attention: {frozen}")
                    
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in periodic backtest: {e}")
                    await asyncio.sleep(10)
        
        self._running = True
        self._backtest_task = asyncio.create_task(run_periodic_backtest())
        logger.info("Started periodic backtest")
    
    def stop_periodic_backtest(self) -> None:
        """停止定期回测"""
        self._running = False
        if self._backtest_task:
            self._backtest_task.cancel()
            self._backtest_task = None
        logger.info("Stopped periodic backtest")
    
    def register_signal_callback(self, callback: Callable[[TradingSignal], None]) -> None:
        """注册信号回调"""
        self._signal_callback = callback
    
    def register_trade_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """注册交易回调"""
        self._trade_callback = callback
    
    def get_strategy_status(self, symbol: str = None) -> Dict[str, Any]:
        """获取策略状态"""
        instances = self._strategy_container.get_all_instances()
        
        if symbol:
            instances = [i for i in instances if i.symbol == symbol]
        
        return {
            "run_mode": self._run_mode.value,
            "mode_transitions": self._mode_transition_history[-5:],
            "instances": [inst.to_dict() for inst in instances],
            "aggregate_metrics": self._strategy_container.get_aggregate_metrics(),
            "backtest_stats": self._mini_backtester.get_stats(),
            "indicator_stats": self._indicator_engine.get_stats()
        }
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        from .mini_backtester import StrategyHealth
        
        return {
            "healthy_strategies": self._mini_backtester.get_strategies_by_health(StrategyHealth.HEALTHY),
            "warning_strategies": self._mini_backtester.get_strategies_by_health(StrategyHealth.WARNING),
            "critical_strategies": self._mini_backtester.get_strategies_by_health(StrategyHealth.CRITICAL),
            "frozen_strategies": self._mini_backtester.get_strategies_by_health(StrategyHealth.FROZEN),
            "active_instances": len(self._strategy_container.get_active_instances()),
            "total_instances": len(self._strategy_container.get_all_instances())
        }
    
    def shutdown(self) -> None:
        """关闭引擎"""
        self.stop_periodic_backtest()
        self._strategy_container.clear_all()
        self._mini_backtester.clear_all()
        logger.info("StrategyEngine shutdown complete")


_engine_instance: Optional[StrategyEngine] = None

def get_strategy_engine(config: Dict[str, Any] = None) -> StrategyEngine:
    """获取策略引擎单例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = StrategyEngine(config)
    return _engine_instance


__all__ = [
    "StrategyEngine",
    "IndicatorEngine",
    "SignalGenerator",
    "PositionPlanner",
    "StrategyContainer",
    "MiniBacktester",
    "get_strategy_engine",
    "get_indicator_engine",
    "get_signal_generator",
    "get_position_planner",
    "get_strategy_container",
    "get_mini_backtester",
    "TradingSignal",
    "PositionPlan",
    "StrategyInstance",
    "BacktestResult",
]