"""
策略协同协调器
统一管理中心：策略间通信、状态同步、冲突检测、资金协调、组合风险评估。
"""
import asyncio
import time
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional, List, Set, Tuple
from loguru import logger

from utils.helpers import safe_float, safe_finite, safe_div


class StrategyState(Enum):
    """策略运行状态"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class ConflictType(Enum):
    """冲突类型"""
    OPPOSITE_DIRECTION = "opposite_direction"      # 同标的反方向
    SAME_DIRECTION_OVERLAP = "same_direction_overlap"  # 同标的同方向重叠（不同策略）
    OVERLAPPING_GRID = "overlapping_grid"          # 网格区间重叠
    EXCESSIVE_EXPOSURE = "excessive_exposure"      # 同一标的过度暴露
    TIMING_COLLISION = "timing_collision"          # 同一标的短时间内多信号
    REGIME_INCOMPATIBLE = "regime_incompatible"    # 与市场状态不兼容


class StrategyCoordinator:
    """策略协同协调器：策略间的中央通信与协调枢纽"""

    def __init__(self, config: Dict[str, Any], trade_journal=None, regime_engine=None,
                 adaptive_controller=None):
        self.config = config
        self._trade_journal = trade_journal
        self._regime_engine = regime_engine
        self._adaptive_controller = adaptive_controller
        
        # 注册的策略实例
        self._strategies: Dict[str, Any] = {}
        
        # 策略状态
        self._strategy_states: Dict[str, StrategyState] = {}
        self._strategy_last_heartbeat: Dict[str, datetime] = {}
        
        # 策略持仓快照 {strategy_name: {symbol: position_info}}
        self._strategy_positions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # 策略信号历史 {strategy_name: [signal_records]}
        self._strategy_signals: Dict[str, List[Dict[str, Any]]] = {}
        
        # 冲突规则配置
        self._conflict_rules = config.get("coordination", {}).get("conflict_rules", {})
        self._opposite_direction_timeout = self._conflict_rules.get("opposite_direction_timeout", 120)
        self._same_symbol_max_strategies = self._conflict_rules.get("same_symbol_max_strategies", 3)
        self._min_signal_interval = self._conflict_rules.get("min_signal_interval", 30)
        
        # 协同触发规则
        self._trigger_rules = config.get("coordination", {}).get("trigger_rules", {})
        
        # 资金池
        self._capital_pool: Dict[str, float] = {}
        self._capital_locks: Dict[str, asyncio.Lock] = {}
        
        # 组合风险
        self._portfolio_risk: Dict[str, Any] = {
            "total_exposure": 0.0,
            "symbol_exposure": {},
            "direction_exposure": {"long": 0.0, "short": 0.0},
            "max_drawdown_estimate": 0.0,
            "correlation_risk": 0.0,
            "last_update": None,
        }
        
        # 消息总线
        self._message_bus: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._subscribers: Dict[str, List[callable]] = {}
        
        # 监控循环
        self._running = False
        self._heartbeat_interval = 30
        
        logger.info("StrategyCoordinator initialized")

    # ===================== 策略注册与管理 =====================

    def register_strategy(self, name: str, strategy_instance: Any) -> bool:
        """注册策略实例"""
        if name in self._strategies:
            logger.warning(f"Strategy {name} already registered, replacing")
        
        self._strategies[name] = strategy_instance
        self._strategy_states[name] = StrategyState.IDLE
        self._strategy_positions[name] = {}
        self._strategy_signals[name] = []
        self._capital_locks[name] = asyncio.Lock()
        
        # 注入协调器到策略实例
        if hasattr(strategy_instance, "set_coordinator"):
            strategy_instance.set_coordinator(self)
        
        logger.info(f"Strategy registered: {name}")
        return True

    def unregister_strategy(self, name: str) -> bool:
        """注销策略实例"""
        if name not in self._strategies:
            return False
        
        del self._strategies[name]
        del self._strategy_states[name]
        del self._strategy_positions[name]
        del self._strategy_signals[name]
        del self._capital_locks[name]
        
        logger.info(f"Strategy unregistered: {name}")
        return True

    def set_strategy_state(self, name: str, state: StrategyState):
        """设置策略状态"""
        old_state = self._strategy_states.get(name)
        self._strategy_states[name] = state
        self._strategy_last_heartbeat[name] = datetime.now()
        
        if old_state != state:
            logger.info(f"Strategy {name} state: {old_state.value if old_state else 'none'} -> {state.value}")
            self._broadcast_message("state_change", {
                "strategy": name,
                "old_state": old_state.value if old_state else None,
                "new_state": state.value,
                "timestamp": datetime.now().isoformat(),
            })

    def get_strategy_state(self, name: str) -> StrategyState:
        """获取策略状态"""
        return self._strategy_states.get(name, StrategyState.STOPPED)

    def get_all_strategy_states(self) -> Dict[str, str]:
        """获取所有策略状态"""
        return {k: v.value for k, v in self._strategy_states.items()}

    # ===================== 状态同步 =====================

    def update_strategy_positions(self, strategy_name: str, positions: Dict[str, Dict[str, Any]]):
        """更新策略持仓快照"""
        self._strategy_positions[strategy_name] = positions
        self._strategy_last_heartbeat[strategy_name] = datetime.now()
        self._recalculate_portfolio_risk()

    def record_strategy_signal(self, strategy_name: str, signal_data: Dict[str, Any]):
        """记录策略信号"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "symbol": signal_data.get("symbol", ""),
            "direction": signal_data.get("direction", ""),
            "quantity": signal_data.get("quantity", 0),
            "confidence": signal_data.get("confidence", 0.5),
            "signal_type": signal_data.get("signal_type", ""),
        }
        
        self._strategy_signals[strategy_name].append(record)
        
        if len(self._strategy_signals[strategy_name]) > 200:
            self._strategy_signals[strategy_name] = self._strategy_signals[strategy_name][-200:]
        
        self._broadcast_message("signal", {
            "strategy": strategy_name,
            "signal": record,
        })

    def get_unified_position_view(self) -> Dict[str, Dict[str, Any]]:
        """获取统一持仓视图：按标的聚合所有策略持仓"""
        unified = {}
        
        for strategy_name, positions in self._strategy_positions.items():
            for symbol, pos in positions.items():
                if symbol not in unified:
                    unified[symbol] = {
                        "total_long": 0.0,
                        "total_short": 0.0,
                        "net_exposure": 0.0,
                        "strategies": [],
                        "last_update": datetime.now().isoformat(),
                    }
                
                side = pos.get("side", "long")
                quantity = abs(safe_float(pos.get("quantity"), 0.0))
                
                if side == "long":
                    unified[symbol]["total_long"] += quantity
                    unified[symbol]["net_exposure"] += quantity
                else:
                    unified[symbol]["total_short"] += quantity
                    unified[symbol]["net_exposure"] -= quantity
                
                unified[symbol]["strategies"].append({
                    "strategy": strategy_name,
                    "side": side,
                    "quantity": quantity,
                    "entry_price": pos.get("entry_price", 0),
                })
        
        return unified

    # ===================== 冲突检测与解决 =====================

    def check_signal_conflicts(self, strategy_name: str, signal_data: Dict[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
        """检查信号是否与其他策略产生冲突"""
        conflicts = []
        symbol = signal_data.get("symbol", "")
        direction = signal_data.get("direction", "").lower()
        
        if not symbol or not direction:
            # P0: 空信号不应通过，返回冲突阻止执行
            return False, [{"type": "invalid_signal", "severity": "high", "message": "Signal missing symbol or direction"}]
        
        unified = self.get_unified_position_view()
        
        # 冲突1：同标的反方向持仓
        if symbol in unified:
            pos_info = unified[symbol]
            
            if direction in ("buy", "long") and pos_info["total_short"] > 0:
                # 检查是否由其他策略持有空仓
                for strat_info in pos_info["strategies"]:
                    if strat_info["side"] == "short" and strat_info["strategy"] != strategy_name:
                        conflicts.append({
                            "type": ConflictType.OPPOSITE_DIRECTION.value,
                            "severity": "high",
                            "message": f"{strategy_name} wants {direction} {symbol} but {strat_info['strategy']} holds short",
                            "related_strategy": strat_info["strategy"],
                            "symbol": symbol,
                        })
            
            elif direction in ("sell", "short") and pos_info["total_long"] > 0:
                for strat_info in pos_info["strategies"]:
                    if strat_info["side"] == "long" and strat_info["strategy"] != strategy_name:
                        conflicts.append({
                            "type": ConflictType.OPPOSITE_DIRECTION.value,
                            "severity": "high",
                            "message": f"{strategy_name} wants {direction} {symbol} but {strat_info['strategy']} holds long",
                            "related_strategy": strat_info["strategy"],
                            "symbol": symbol,
                        })
        
        # 冲突1.5：同标的同方向重叠——两个不同策略在同一币种已有同向持仓时，拒绝新信号
        if symbol in unified:
            pos_info = unified[symbol]
            for strat_info in pos_info["strategies"]:
                if strat_info["strategy"] != strategy_name:
                    strat_direction = strat_info["side"]
                    new_direction = "long" if direction in ("buy", "long") else "short"
                    if strat_direction == new_direction and strat_info["quantity"] > 0:
                        conflicts.append({
                            "type": ConflictType.SAME_DIRECTION_OVERLAP.value,
                            "severity": "high",
                            "message": f"{strategy_name} wants {new_direction} {symbol} but {strat_info['strategy']} already holds {strat_direction}",
                            "related_strategy": strat_info["strategy"],
                            "symbol": symbol,
                        })
        
        # 冲突2：同一标的策略数量过多
        if symbol in unified:
            active_strategies = {s["strategy"] for s in unified[symbol]["strategies"]}
            if len(active_strategies) >= self._same_symbol_max_strategies:
                if strategy_name not in active_strategies:
                    conflicts.append({
                        "type": ConflictType.EXCESSIVE_EXPOSURE.value,
                        "severity": "medium",
                        "message": f"{symbol} already managed by {len(active_strategies)} strategies, max={self._same_symbol_max_strategies}",
                        "symbol": symbol,
                    })
        
        # 冲突3：短时间内同一标的多信号
        recent_signals = self._get_recent_signals_for_symbol(symbol, seconds=self._min_signal_interval)
        if recent_signals:
            for sig in recent_signals:
                if sig["strategy"] != strategy_name:
                    conflicts.append({
                        "type": ConflictType.TIMING_COLLISION.value,
                        "severity": "low",
                        "message": f"{symbol} had signal from {sig['strategy']} {self._min_signal_interval}s ago",
                        "related_strategy": sig["strategy"],
                        "symbol": symbol,
                    })
        
        # 冲突4：市场状态不兼容
        # P0: get_regime可能返回None，需要防御性检查
        if self._regime_engine:
            regime = self._regime_engine.get_regime()
            if regime is None:
                regime = {}
            regime_type = regime.get("regime", "unknown")
            
            if regime_type in ("extreme_volatility", "liquidity_crisis"):
                # 极端市场状态下，限制网格和马丁格尔策略
                if strategy_name in ("grid", "spot_grid", "spot_martingale"):
                    conflicts.append({
                        "type": ConflictType.REGIME_INCOMPATIBLE.value,
                        "severity": "medium",
                        "message": f"{strategy_name} paused due to {regime_type}",
                        "symbol": symbol,
                    })
        
        has_high = any(c["severity"] == "high" for c in conflicts)
        return not has_high, conflicts

    def _get_recent_signals_for_symbol(self, symbol: str, seconds: int = 30) -> List[Dict[str, Any]]:
        """获取某标的最近信号"""
        now = datetime.now()
        recent = []
        
        for strategy_name, signals in self._strategy_signals.items():
            for sig in reversed(signals):
                sig_time = datetime.fromisoformat(sig["timestamp"]) if isinstance(sig["timestamp"], str) else sig["timestamp"]
                if (now - sig_time).total_seconds() <= seconds and sig.get("symbol") == symbol:
                    recent.append({**sig, "strategy": strategy_name})
        
        return recent

    def resolve_conflict(self, strategy_name: str, signal_data: Dict[str, Any], 
                         conflicts: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """尝试解决冲突，返回是否允许执行和调整后的信号"""
        resolved_signal = dict(signal_data)
        
        for conflict in conflicts:
            ctype = conflict.get("type", "")
            severity = conflict.get("severity", "low")
            
            if ctype == ConflictType.OPPOSITE_DIRECTION.value and severity == "high":
                # 高严重度反方向冲突：拒绝
                logger.warning(f"Conflict rejected: {conflict.get('message', '')}")
                return False, resolved_signal
            
            elif ctype == ConflictType.SAME_DIRECTION_OVERLAP.value and severity == "high":
                # 同向重叠冲突：拒绝，防止同一币种同一方向双重暴露
                logger.warning(f"Same-direction overlap rejected: {conflict.get('message', '')}")
                return False, resolved_signal
            
            elif ctype == ConflictType.EXCESSIVE_EXPOSURE.value:
                # 过度暴露：降低仓位
                current_qty = safe_float(resolved_signal.get("quantity"), 0.0)
                if current_qty > 0:
                    resolved_signal["quantity"] = current_qty * 0.5
                    logger.info(f"Conflict resolved: reduced quantity by 50% for {strategy_name} {resolved_signal.get('symbol')}")
            
            elif ctype == ConflictType.TIMING_COLLISION.value:
                # 时间碰撞：轻微降低仓位
                current_qty = safe_float(resolved_signal.get("quantity"), 0.0)
                if current_qty > 0:
                    resolved_signal["quantity"] = current_qty * 0.8
            
            elif ctype == ConflictType.REGIME_INCOMPATIBLE.value:
                # 状态不兼容：拒绝
                logger.warning(f"Regime conflict rejected: {conflict.get('message', '')}")
                return False, resolved_signal
        
        return True, resolved_signal

    # ===================== 协同触发机制 =====================

    def check_collaborative_trigger(self, strategy_name: str, signal_data: Dict[str, Any]) -> Tuple[bool, str]:
        """检查协同触发条件
        
        P4-1: 趋势策略必须有实际持仓才能阻挡网格信号。
        仅生成信号但未执行（被拒绝）的趋势信号不应阻挡网格。
        """
        symbol = signal_data.get("symbol", "")
        
        # 规则1：趋势策略确认后，网格/马丁策略才能在该标的上开新仓
        if strategy_name in ("grid", "spot_grid", "spot_martingale"):
            # P4-1: 先检查趋势策略是否实际持有该币种仓位
            trend_positions = self._strategy_positions.get("trend", {})
            trend_has_position = symbol in trend_positions and trend_positions[symbol].get("quantity", 0) > 0
            
            if not trend_has_position:
                # 趋势策略无实际持仓，不阻挡网格信号
                return True, "Trend strategy has no position, grid allowed"
            
            trend_signals = self._get_strategy_recent_signals("trend", symbol, minutes=60)
            
            if trend_signals:
                # 趋势策略近期在该标的上有信号且有实际持仓
                latest_trend = trend_signals[-1]
                trend_direction = latest_trend.get("direction", "").lower()
                signal_direction = signal_data.get("direction", "").lower()
                
                # 如果趋势方向与信号方向一致，允许
                if trend_direction and signal_direction:
                    if (trend_direction in ("buy", "long") and signal_direction in ("buy", "long")) or \
                       (trend_direction in ("sell", "short") and signal_direction in ("sell", "short")):
                        return True, f"Aligned with trend strategy ({trend_direction})"
                    else:
                        return False, f"Opposes trend strategy ({trend_direction})"
        
        # 规则2：套利策略需要多个策略都活跃时才启动
        if strategy_name == "arbitrage":
            active_count = sum(1 for s in self._strategy_states.values() if s == StrategyState.RUNNING)
            if active_count < 2:
                return False, f"Arbitrage requires at least 2 active strategies, found {active_count}"
        
        return True, "No collaborative restrictions"

    def _get_strategy_recent_signals(self, strategy_name: str, symbol: str, minutes: int = 60) -> List[Dict[str, Any]]:
        """获取策略在某标的上的近期信号"""
        now = datetime.now()
        recent = []
        
        signals = self._strategy_signals.get(strategy_name, [])
        for sig in reversed(signals):
            sig_time = datetime.fromisoformat(sig["timestamp"]) if isinstance(sig["timestamp"], str) else sig["timestamp"]
            if (now - sig_time).total_seconds() <= minutes * 60 and sig.get("symbol") == symbol:
                recent.append(sig)
        
        return recent

    # ===================== 组合风险评估 =====================

    def _recalculate_portfolio_risk(self):
        """重新计算组合风险（输出全部 finite，JSON 安全）。"""
        unified = self.get_unified_position_view()
        
        total_long = 0.0
        total_short = 0.0
        symbol_exposure = {}
        
        for symbol, info in unified.items():
            long_qty = safe_float(info.get("total_long"), 0.0)
            short_qty = safe_float(info.get("total_short"), 0.0)
            net_exp = safe_float(info.get("net_exposure"), 0.0)
            total_long += long_qty
            total_short += short_qty
            
            exposure = abs(net_exp)
            symbol_exposure[symbol] = {
                "net": safe_finite(net_exp, 0.0),
                "long": safe_finite(long_qty, 0.0),
                "short": safe_finite(short_qty, 0.0),
                "gross": safe_finite(long_qty + short_qty, 0.0),
            }
        
        total_exposure = safe_finite(total_long + total_short, 0.0)
        net_exposure = safe_finite(total_long - total_short, 0.0)
        
        # 计算方向集中度
        direction_exposure = {"long": safe_finite(total_long, 0.0), "short": safe_finite(total_short, 0.0)}
        
        # 简单估算最大回撤（基于净暴露）
        max_dd_estimate = safe_div(abs(net_exposure), total_exposure, 0.0) if total_exposure > 0 else 0.0
        
        # 相关性风险：标的相关性高的组合风险更大
        correlation_risk = safe_finite(self._estimate_correlation_risk(unified), 0.0)
        
        self._portfolio_risk = {
            "total_exposure": safe_finite(total_exposure, 0.0),
            "net_exposure": safe_finite(net_exposure, 0.0),
            "symbol_exposure": symbol_exposure,
            "direction_exposure": direction_exposure,
            "max_drawdown_estimate": safe_finite(max_dd_estimate, 0.0),
            "correlation_risk": correlation_risk,
            "last_update": datetime.now().isoformat(),
        }

    def _estimate_correlation_risk(self, unified: Dict[str, Dict[str, Any]]) -> float:
        """估算相关性风险（简化版：基于标的后缀分组）"""
        groups = {}
        
        for symbol in unified.keys():
            base = symbol.split("-")[0] if "-" in symbol else symbol
            if base not in groups:
                groups[base] = []
            groups[base].append(symbol)
        
        # 同一基础资产有多个合约 = 高相关性风险
        risk = 0.0
        for base, symbols in groups.items():
            if len(symbols) > 1:
                risk += (len(symbols) - 1) * 0.1
        
        return min(1.0, risk)

    def get_portfolio_risk(self) -> Dict[str, Any]:
        """获取组合风险报告"""
        return dict(self._portfolio_risk)

    def is_portfolio_risk_acceptable(self) -> Tuple[bool, str]:
        """检查组合风险是否在可接受范围内"""
        risk = self._portfolio_risk
        
        max_dd_threshold = self.config.get("trading", {}).get("max_drawdown", 0.12)
        current_dd_estimate = risk.get("max_drawdown_estimate", 0)
        
        if current_dd_estimate > max_dd_threshold * 0.8:
            return False, f"Portfolio drawdown estimate {current_dd_estimate:.1%}接近阈值 {max_dd_threshold:.1%}"
        
        correlation_risk = risk.get("correlation_risk", 0)
        if correlation_risk > 0.5:
            return False, f"Correlation risk {correlation_risk:.1%}过高"
        
        return True, "Portfolio risk acceptable"

    # ===================== 消息总线 =====================

    def _broadcast_message(self, message_type: str, payload: Dict[str, Any]):
        """广播消息到所有订阅者"""
        message = {
            "type": message_type,
            "payload": payload,
            "timestamp": datetime.now().isoformat(),
        }
        
        try:
            self._message_bus.put_nowait(message)
        except asyncio.QueueFull:
            logger.warning("Message bus full, dropping message")
        
        # 直接调用订阅者回调
        callbacks = self._subscribers.get(message_type, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback(message))
                    task.add_done_callback(
                        lambda completed_task: self._log_callback_task_error(
                            completed_task, message_type
                        )
                    )
                else:
                    callback(message)
            except Exception:
                logger.exception(f"Message callback error for {message_type}")

    @staticmethod
    def _log_callback_task_error(task: asyncio.Task, message_type: str):
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            logger.exception(f"Message callback error for {message_type}")

    def subscribe(self, message_type: str, callback: callable):
        """订阅消息"""
        if message_type not in self._subscribers:
            self._subscribers[message_type] = []
        self._subscribers[message_type].append(callback)

    def unsubscribe(self, message_type: str, callback: callable):
        """取消订阅"""
        if message_type in self._subscribers:
            if callback in self._subscribers[message_type]:
                self._subscribers[message_type].remove(callback)

    # ===================== 监控循环 =====================

    async def start(self):
        """启动协调器"""
        self._running = True
        # P0: 保存Task引用，防止被GC回收导致后台循环静默停止
        if not hasattr(self, '_bg_tasks'):
            self._bg_tasks = []
        self._bg_tasks.append(asyncio.create_task(self._heartbeat_loop()))
        self._bg_tasks.append(asyncio.create_task(self._message_dispatch_loop()))
        logger.info("StrategyCoordinator started")

    async def stop(self):
        """停止协调器"""
        self._running = False
        logger.info("StrategyCoordinator stopped")

    async def _heartbeat_loop(self):
        """心跳监控循环"""
        while self._running:
            now = datetime.now()
            
            for name, last_beat in list(self._strategy_last_heartbeat.items()):
                if (now - last_beat).total_seconds() > self._heartbeat_interval * 2:
                    if self._strategy_states.get(name) == StrategyState.RUNNING:
                        logger.warning(f"Strategy {name} heartbeat timeout, marking as error")
                        self.set_strategy_state(name, StrategyState.ERROR)
            
            await asyncio.sleep(self._heartbeat_interval)

    async def _message_dispatch_loop(self):
        """消息分发循环"""
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_bus.get(), timeout=1.0)
                # 消息已通过广播发送，这里只是消费队列
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.debug(f"Message dispatch error: {e}")

    # ===================== 状态查询 =====================

    def get_status(self) -> Dict[str, Any]:
        """获取协调器完整状态"""
        return {
            "registered_strategies": list(self._strategies.keys()),
            "strategy_states": self.get_all_strategy_states(),
            "portfolio_risk": self.get_portfolio_risk(),
            "unified_positions": self.get_unified_position_view(),
            "total_signals_recorded": sum(len(s) for s in self._strategy_signals.values()),
        }

    # ===================== 强化：统一状态总线 =====================

    def publish_state(self, strategy_name: str, state_type: str, state_data: Dict[str, Any]):
        """策略发布状态到统一总线"""
        message = {
            "strategy": strategy_name,
            "state_type": state_type,
            "data": state_data,
            "timestamp": datetime.now().isoformat(),
        }
        self._broadcast_message(f"state.{state_type}", message)

        if not hasattr(self, '_state_store'):
            self._state_store = {}
        if strategy_name not in self._state_store:
            self._state_store[strategy_name] = {}
        self._state_store[strategy_name][state_type] = state_data

    def get_aggregated_state(self) -> Dict[str, Any]:
        """获取聚合后的全局状态"""
        if not hasattr(self, '_state_store'):
            return {}
        return {
            "by_strategy": dict(self._state_store),
            "last_update": datetime.now().isoformat(),
            "total_strategies": len(self._state_store),
        }

    def query_state(self, strategy_name: str = None, state_type: str = None) -> Dict[str, Any]:
        """查询状态：可按策略名和状态类型过滤"""
        if not hasattr(self, '_state_store'):
            return {}

        if strategy_name and state_type:
            return self._state_store.get(strategy_name, {}).get(state_type, {})
        elif strategy_name:
            return self._state_store.get(strategy_name, {})
        elif state_type:
            result = {}
            for sname, states in self._state_store.items():
                if state_type in states:
                    result[sname] = states[state_type]
            return result
        return dict(self._state_store)

    # ===================== 强化：策略联动机制 =====================

    def check_cross_strategy_confirmation(self, symbol: str, direction: str,
                                           min_strategies: int = 2) -> Dict[str, Any]:
        """跨策略信号确认：检查多个策略是否在同一标的上给出同向信号
        返回: {"confirmed": bool, "agreeing_strategies": list, "total_strategies": int, "agree_ratio": float}
        """
        try:
            agreeing = []
            total_active = 0

            for strategy_name, signals in self._strategy_signals.items():
                state = self._strategy_states.get(strategy_name)
                if state != StrategyState.RUNNING:
                    continue
                total_active += 1

                recent = [s for s in signals[-20:] if s.get("symbol") == symbol]
                if not recent:
                    continue

                latest = recent[-1]
                sig_dir = latest.get("direction", "").lower()

                if direction in ("long", "buy") and sig_dir in ("long", "buy"):
                    agreeing.append({
                        "strategy": strategy_name,
                        "confidence": latest.get("confidence", 0),
                        "timestamp": latest.get("timestamp"),
                    })
                elif direction in ("short", "sell") and sig_dir in ("short", "sell"):
                    agreeing.append({
                        "strategy": strategy_name,
                        "confidence": latest.get("confidence", 0),
                        "timestamp": latest.get("timestamp"),
                    })

            confirmed = len(agreeing) >= min_strategies
            agree_ratio = len(agreeing) / total_active if total_active > 0 else 0

            return {
                "confirmed": confirmed,
                "agreeing_strategies": agreeing,
                "total_active_strategies": total_active,
                "agree_count": len(agreeing),
                "agree_ratio": agree_ratio,
            }
        except Exception as e:
            logger.debug(f"Cross-strategy confirmation error: {e}")
            return {"confirmed": False, "agreeing_strategies": [], "total_active_strategies": 0, "agree_count": 0, "agree_ratio": 0}

    def get_strategy_consensus_score(self, symbol: str, direction: str) -> float:
        """计算策略共识分数：0-1，越高代表越多策略同向"""
        result = self.check_cross_strategy_confirmation(symbol, direction)
        agreeing = result.get("agreeing_strategies", [])
        if not agreeing:
            return 0.0

        avg_conf = safe_finite(
            sum(safe_float(s.get("confidence"), 0.0) for s in agreeing) / max(len(agreeing), 1),
            0.0,
        )
        agree_ratio = safe_finite(result.get("agree_ratio"), 0.0)
        consensus = agree_ratio * 0.6 + avg_conf * 0.4

        return min(1.0, max(0.0, safe_finite(consensus, 0.0)))

    # ===================== 强化：策略健康自愈 =====================

    async def diagnose_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """诊断策略健康状态，返回诊断结果和建议"""
        try:
            diagnosis = {
                "strategy": strategy_name,
                "timestamp": datetime.now().isoformat(),
                "issues": [],
                "recommendations": [],
                "health_score": 1.0,
            }

            state = self._strategy_states.get(strategy_name)
            if state == StrategyState.ERROR:
                diagnosis["issues"].append("strategy_in_error_state")
                diagnosis["recommendations"].append("restart_strategy")
                diagnosis["health_score"] = 0.2
            elif state == StrategyState.PAUSED:
                diagnosis["issues"].append("strategy_paused")
                diagnosis["health_score"] = 0.5
            elif state == StrategyState.STOPPED:
                diagnosis["issues"].append("strategy_stopped")
                diagnosis["health_score"] = 0.1

            last_beat = self._strategy_last_heartbeat.get(strategy_name)
            if last_beat:
                elapsed = (datetime.now() - last_beat).total_seconds()
                if elapsed > self._heartbeat_interval * 3:
                    diagnosis["issues"].append(f"heartbeat_timeout_{int(elapsed)}s")
                    diagnosis["recommendations"].append("check_strategy_liveness")
                    diagnosis["health_score"] = min(diagnosis["health_score"], 0.3)

            signals = self._strategy_signals.get(strategy_name, [])
            if len(signals) == 0 and state == StrategyState.RUNNING:
                diagnosis["issues"].append("no_signals_generated")
                diagnosis["recommendations"].append("check_signal_generation")
                diagnosis["health_score"] = min(diagnosis["health_score"], 0.6)

            positions = self._strategy_positions.get(strategy_name, {})
            if positions and state == StrategyState.RUNNING:
                diagnosis["positions_count"] = len(positions)

            return diagnosis
        except Exception as e:
            logger.error(f"Strategy diagnosis error: {e}")
            return {"strategy": strategy_name, "issues": ["diagnosis_failed"], "health_score": 0.0}

    async def auto_heal_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """策略自愈：尝试自动修复策略问题"""
        try:
            diagnosis = await self.diagnose_strategy(strategy_name)
            result = {"strategy": strategy_name, "actions_taken": [], "success": False}

            if not diagnosis["issues"]:
                result["success"] = True
                result["actions_taken"].append("no_issues_found")
                return result

            for issue in diagnosis["issues"]:
                if issue == "strategy_in_error_state":
                    healed = await self._heal_error_state(strategy_name)
                    if healed:
                        result["actions_taken"].append("reset_error_state")
                elif issue.startswith("heartbeat_timeout"):
                    healed = await self._heal_heartbeat_timeout(strategy_name)
                    if healed:
                        result["actions_taken"].append("heartbeat_recovered")

            result["success"] = len(result["actions_taken"]) > 0
            return result
        except Exception as e:
            logger.error(f"Auto-heal error for {strategy_name}: {e}")
            return {"strategy": strategy_name, "actions_taken": [], "success": False, "error": str(e)}

    async def _heal_error_state(self, strategy_name: str) -> bool:
        """修复错误状态：重置策略状态"""
        try:
            strategy = self._strategies.get(strategy_name)
            if strategy and hasattr(strategy, 'start'):
                self.set_strategy_state(strategy_name, StrategyState.IDLE)
                logger.info(f"Healed error state for {strategy_name}")
                return True
            return False
        except Exception as e:
            logger.debug(f"Heal error state failed: {e}")
            return False

    async def _heal_heartbeat_timeout(self, strategy_name: str) -> bool:
        """修复心跳超时：更新心跳时间"""
        try:
            self._strategy_last_heartbeat[strategy_name] = datetime.now()
            if self._strategy_states.get(strategy_name) == StrategyState.ERROR:
                self.set_strategy_state(strategy_name, StrategyState.RUNNING)
            return True
        except Exception:
            return False

    async def diagnose_all(self) -> Dict[str, Any]:
        """诊断所有策略"""
        results = {}
        total_health = 0.0
        count = 0

        for name in self._strategies:
            diag = await self.diagnose_strategy(name)
            results[name] = diag
            total_health += diag.get("health_score", 0)
            count += 1

        return {
            "overall_health": total_health / count if count > 0 else 0,
            "strategy_count": count,
            "details": results,
            "timestamp": datetime.now().isoformat(),
        }

    # ===================== 强化：仓位一致性自愈 =====================

    async def reconcile_position_consistency(self) -> Dict[str, Any]:
        """仓位一致性对账与自愈：对比各策略持仓与交易所实际持仓
        返回 reconciliation 结果
        """
        try:
            result = {
                "timestamp": datetime.now().isoformat(),
                "strategy_positions": {},
                "issues": [],
                "fixed": [],
                "total_mismatches": 0,
            }

            okx_client = None
            for strat in self._strategies.values():
                if hasattr(strat, 'okx_client'):
                    okx_client = strat.okx_client
                    break

            if not okx_client:
                result["issues"].append("okx_client_unavailable")
                return result

            try:
                positions = okx_client.get_positions()
                okx_positions = {}
                for pos_data in positions or []:
                    try:
                        pos = okx_client._parse_position(pos_data)
                        if pos and abs(pos.quantity) > 0:
                            okx_positions[pos.symbol] = {
                                "side": pos.side,
                                "quantity": abs(pos.quantity),
                                "avg_cost": pos.avg_cost,
                            }
                    except Exception as parse_err:
                        logger.debug(f"StrategyCoordinator: failed to parse position: {parse_err}")
                        continue
            except Exception as e:
                result["issues"].append(f"okx_fetch_error: {e}")
                return result

            strategy_total = {}
            for sname, positions_dict in self._strategy_positions.items():
                for symbol, pos in positions_dict.items():
                    if symbol not in strategy_total:
                        strategy_total[symbol] = {"long": 0.0, "short": 0.0}
                    side = pos.get("side", "long")
                    qty = abs(float(pos.get("quantity", 0)))
                    strategy_total[symbol][side] += qty

            result["strategy_positions"] = strategy_total
            result["okx_positions"] = okx_positions

            for symbol, okx_pos in okx_positions.items():
                strat_pos = strategy_total.get(symbol, {"long": 0.0, "short": 0.0})
                strat_side_qty = strat_pos.get(okx_pos["side"], 0)
                diff = abs(strat_side_qty - okx_pos["quantity"])

                if diff > okx_pos["quantity"] * 0.01 and diff > 0.0001:
                    result["total_mismatches"] += 1
                    result["issues"].append({
                        "symbol": symbol,
                        "side": okx_pos["side"],
                        "okx_qty": okx_pos["quantity"],
                        "strategy_qty": strat_side_qty,
                        "diff": diff,
                    })

            result["reconciled"] = result["total_mismatches"] == 0
            return result
        except Exception as e:
            logger.error(f"Position reconciliation error: {e}")
            return {"issues": [f"reconciliation_error: {e}"], "total_mismatches": -1}

    # ===================== 强化：系统级自愈 =====================

    async def system_auto_heal(self) -> Dict[str, Any]:
        """系统级自愈：检查并修复整个系统的问题"""
        try:
            actions = []

            strategy_diag = await self.diagnose_all()
            if strategy_diag["overall_health"] < 0.5:
                for sname, diag in strategy_diag["details"].items():
                    if diag.get("health_score", 1) < 0.5:
                        heal_result = await self.auto_heal_strategy(sname)
                        if heal_result.get("success"):
                            actions.append(f"healed_{sname}")

            pos_recon = await self.reconcile_position_consistency()
            if pos_recon.get("total_mismatches", 0) > 0:
                actions.append(f"position_mismatch_{pos_recon['total_mismatches']}")

            return {
                "timestamp": datetime.now().isoformat(),
                "actions_performed": actions,
                "overall_health_before": strategy_diag.get("overall_health", 0),
                "position_mismatches": pos_recon.get("total_mismatches", 0),
            }
        except Exception as e:
            logger.error(f"System auto-heal error: {e}")
            return {"actions_performed": [], "error": str(e)}

    # ===================== 强化：动态策略权重协调 =====================

    def calculate_dynamic_strategy_weights(self) -> Dict[str, float]:
        """动态计算策略权重：基于近期表现+健康状态+市场适配度（归一化，JSON 安全）。"""
        try:
            weights = {}
            n = len(self._strategies)
            base_weight = safe_div(1.0, n, 0.25) if n > 0 else 0.25

            for name in self._strategies:
                weight = safe_finite(base_weight, 0.0)

                state = self._strategy_states.get(name, StrategyState.STOPPED)
                if state == StrategyState.RUNNING:
                    weight *= 1.0
                elif state == StrategyState.PAUSED:
                    weight *= 0.3
                else:
                    weight *= 0.1

                signals = self._strategy_signals.get(name, [])
                if signals:
                    recent = signals[-20:]
                    avg_conf = safe_finite(
                        sum(safe_float(s.get("confidence"), 0.0) for s in recent) / max(len(recent), 1),
                        0.0,
                    )
                    weight *= (0.7 + avg_conf * 0.6)

                weights[name] = safe_finite(weight, 0.0)

            total = sum(weights.values())
            if total > 0:
                for k in weights:
                    weights[k] = safe_div(weights[k], total, 0.0)

            return weights
        except Exception as e:
            logger.debug(f"Dynamic weight calculation error: {e}")
            n = len(self._strategies)
            if not n:
                return {}
            return {name: safe_div(1.0, n, 0.0) for name in self._strategies}

    def get_strategy_summary(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """获取策略摘要"""
        if strategy_name not in self._strategies:
            return None
        
        return {
            "name": strategy_name,
            "state": self._strategy_states.get(strategy_name, StrategyState.STOPPED).value,
            "position_count": len(self._strategy_positions.get(strategy_name, {})),
            "signal_count": len(self._strategy_signals.get(strategy_name, [])),
            "last_heartbeat": self._strategy_last_heartbeat.get(strategy_name),
        }

    # ===================== 协调循环：问题检测与自动修复 =====================

    async def start_coordination_loop(self):
        """启动协调循环（问题检测与自动修复）"""
        self._running = True
        asyncio.create_task(self._coordination_loop())
        logger.info("StrategyCoordinator coordination loop started")

    async def _coordination_loop(self):
        """协调循环：每5分钟检查一次策略状态、资金、冲突"""
        while self._running:
            try:
                await asyncio.sleep(300)  # 5分钟间隔

                # 1. 检查策略健康状态
                await self._check_strategy_health()

                # 2. 检查资金分配是否合理
                await self._check_capital_allocation()

                # 3. 检查是否有长期无信号的策略
                await self._check_signal_activity()

                # 4. 保存协调状态
                self._save_coordination_state()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(60)

    async def _check_strategy_health(self):
        """检查策略健康状态"""
        for name, state in self._strategy_states.items():
            if state == StrategyState.ERROR:
                logger.warning(f"Strategy {name} in ERROR state, attempting recovery")
                # 尝试重启策略
                self.set_strategy_state(name, StrategyState.IDLE)

            # 检查策略回撤
            try:
                strategy = self._strategies.get(name)
                if strategy and hasattr(strategy, "get_drawdown"):
                    drawdown = strategy.get_drawdown()
                    if drawdown and drawdown > 0.5:  # 回撤超过50%
                        logger.warning(f"Strategy {name} drawdown {drawdown:.1%} critical, pausing")
                        self.set_strategy_state(name, StrategyState.PAUSED)
            except Exception:
                pass

    async def _check_capital_allocation(self):
        """检查资金分配与风险预算，动态调整"""
        try:
            # 获取各策略近期表现
            performance = {}
            for name in self._strategies:
                signals = self._strategy_signals.get(name, [])
                if len(signals) >= 5:
                    # 简单计算胜率
                    wins = sum(1 for s in signals[-20:] if s.get("pnl", 0) > 0)
                    total = len([s for s in signals[-20:] if s.get("pnl") is not None])
                    if total > 0:
                        performance[name] = wins / total

            # 如果某些策略表现很差，降低其资金分配
            for name, win_rate in performance.items():
                if win_rate < 0.3:  # 胜率低于30%
                    logger.warning(f"Strategy {name} win rate {win_rate:.1%} low, reducing allocation")
                    # 通知策略减少仓位
                    strategy = self._strategies.get(name)
                    if strategy and hasattr(strategy, "reduce_position_size"):
                        strategy.reduce_position_size(ratio=0.5)

            # 风险预算检查：查询AdaptiveController的风险预算状态
            if self._adaptive_controller:
                try:
                    risk_status = self._adaptive_controller.get_risk_budget_status()
                    # 检查是否有策略风险预算耗尽
                    for sname, sstatus in risk_status.get("strategy_status", {}).items():
                        if not sstatus.get("healthy", True):
                            logger.warning(
                                f"[RISK_BUDGET] {sname} risk budget low: "
                                f"used={sstatus.get('usage_pct', 0):.0f}%, "
                                f"remaining={sstatus.get('remaining_usdt', 0):.4f}"
                            )
                            # 策略风险预算不足时，暂停其开仓
                            strategy = self._strategies.get(sname)
                            if strategy and hasattr(strategy, "pause_opening"):
                                strategy.pause_opening()
                    # 熔断锁激活时暂停所有策略
                    if risk_status.get("streak_lock_active"):
                        logger.warning("[RISK_BUDGET] Streak lock active, pausing all strategies")
                        for sname in self._strategies:
                            if self._strategy_states.get(sname) == StrategyState.RUNNING:
                                self.set_strategy_state(sname, StrategyState.PAUSED)
                except Exception as re:
                    logger.debug(f"Risk budget check in coordinator error: {re}")

        except Exception as e:
            logger.debug(f"Capital allocation check error: {e}")

    async def _check_signal_activity(self):
        """检查策略信号活跃度"""
        now = datetime.now()
        for name, signals in self._strategy_signals.items():
            if not signals:
                continue

            last_signal_time = datetime.fromisoformat(signals[-1]["timestamp"]) if isinstance(signals[-1]["timestamp"], str) else signals[-1]["timestamp"]
            minutes_since = (now - last_signal_time).total_seconds() / 60

            if minutes_since > 120:  # 2小时无信号
                state = self._strategy_states.get(name)
                if state == StrategyState.RUNNING:
                    logger.warning(f"Strategy {name} no signals for {minutes_since:.0f} minutes")
                    # 只是记录，不暂停，因为可能是市场原因

    def _save_coordination_state(self):
        """保存协调状态到文件"""
        try:
            import json as _json
            import os

            state = {
                "timestamp": datetime.now().isoformat(),
                "strategy_states": self.get_all_strategy_states(),
                "portfolio_risk": self.get_portfolio_risk(),
                "unified_positions": self.get_unified_position_view(),
            }

            os.makedirs("./data/coordination", exist_ok=True)
            with open("./data/coordination/state.json", "w", encoding="utf-8") as f:
                _json.dump(state, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.debug(f"Save coordination state error: {e}")

# ═══════════════════════════════════════════════════════════════
# 信号聚合中心
# ═══════════════════════════════════════════════════════════════

class SignalAggregationHub:
    """信号聚合中心：汇总所有策略信号，统一评估与路由"""
    
    def __init__(self, config: Dict[str, Any], coordinator: Optional[StrategyCoordinator] = None):
        self._config = config
        self._coordinator = coordinator
        self._lock = asyncio.Lock()
        self._signal_buffer: Dict[str, List[Dict[str, Any]]] = {}  # symbol -> signals
        self._max_buffer_size = config.get("coordination", {}).get("signal_buffer_size", 100)
        self._consensus_threshold = config.get("coordination", {}).get("consensus_threshold", 2)
        self._window_default = config.get("coordination", {}).get("signal_window_seconds", 60)
        self._stats = {"total_received": 0, "total_aggregated": 0, "total_conflicts": 0}
        logger.info("SignalAggregationHub initialized")

    async def receive_signal(self, strategy_name: str, symbol: str, direction: str,
                              confidence: float, quantity: float = 0,
                              metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        async with self._lock:
            signal = {
                "strategy": strategy_name, "symbol": symbol,
                "direction": direction, "confidence": confidence,
                "quantity": quantity, "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            }
            if symbol not in self._signal_buffer:
                self._signal_buffer[symbol] = []
            self._signal_buffer[symbol].append(signal)
            if len(self._signal_buffer[symbol]) > self._max_buffer_size:
                self._signal_buffer[symbol] = self._signal_buffer[symbol][-self._max_buffer_size:]
            self._stats["total_received"] += 1
            return {"status": "received", "symbol": symbol}

    async def aggregate_window(self, window_seconds: float = None) -> Dict[str, Any]:
        seconds = window_seconds or self._window_default
        now = datetime.now()
        async with self._lock:
            result = {}
            for symbol, signals in self._signal_buffer.items():
                recent = [s for s in signals if (now - datetime.fromisoformat(s["timestamp"])).total_seconds() <= seconds]
                if not recent:
                    continue
                long_signals = [s for s in recent if s["direction"] in ("buy", "long")]
                short_signals = [s for s in recent if s["direction"] in ("sell", "short")]
                long_conf = sum(s["confidence"] for s in long_signals) / max(len(long_signals), 1)
                short_conf = sum(s["confidence"] for s in short_signals) / max(len(short_signals), 1)
                result[symbol] = {
                    "total_signals": len(recent),
                    "long_count": len(long_signals), "short_count": len(short_signals),
                    "avg_long_confidence": round(long_conf, 4),
                    "avg_short_confidence": round(short_conf, 4),
                    "consensus": "long" if len(long_signals) > len(short_signals) else ("short" if len(short_signals) > len(long_signals) else "neutral"),
                }
            self._stats["total_aggregated"] += 1
            return {"status": "aggregated", "window_seconds": seconds, "symbols": result, "timestamp": now.isoformat()}

    async def compute_consensus(self, symbol: str) -> Dict[str, Any]:
        async with self._lock:
            signals = self._signal_buffer.get(symbol, [])
            if not signals:
                return {"symbol": symbol, "consensus": "none", "agree_ratio": 0, "confidence": 0}
            long_count = sum(1 for s in signals[-20:] if s["direction"] in ("buy", "long"))
            short_count = sum(1 for s in signals[-20:] if s["direction"] in ("sell", "short"))
            total = long_count + short_count
            if total == 0:
                return {"symbol": symbol, "consensus": "none", "agree_ratio": 0, "confidence": 0}
            if long_count >= short_count:
                consensus = "long"
                agree_ratio = long_count / total
            else:
                consensus = "short"
                agree_ratio = short_count / total
            avg_conf = sum(s["confidence"] for s in signals[-20:]) / max(len(signals[-20:]), 1)
            return {"symbol": symbol, "consensus": consensus, "agree_ratio": round(agree_ratio, 4),
                    "confidence": round(avg_conf * agree_ratio, 4), "total_signals": total}

    async def detect_divergence(self, symbol: str) -> Dict[str, Any]:
        async with self._lock:
            signals = self._signal_buffer.get(symbol, [])
            if len(signals) < 2:
                return {"symbol": symbol, "diverged": False, "reason": "insufficient signals"}
            long_count = sum(1 for s in signals[-20:] if s["direction"] in ("buy", "long"))
            short_count = sum(1 for s in signals[-20:] if s["direction"] in ("sell", "short"))
            total = long_count + short_count
            if total == 0:
                return {"symbol": symbol, "diverged": False}
            min_pct = min(long_count, short_count) / total
            diverged = min_pct > 0.25 and total >= 3
            if diverged:
                self._stats["total_conflicts"] += 1
            return {"symbol": symbol, "diverged": diverged, "long_ratio": round(long_count/total, 4),
                    "short_ratio": round(short_count/total, 4), "total": total}

    async def prioritize_signals(self) -> List[Dict[str, Any]]:
        async with self._lock:
            scored = []
            for symbol, signals in self._signal_buffer.items():
                if not signals:
                    continue
                recent = signals[-10:]
                avg_conf = sum(s["confidence"] for s in recent) / len(recent)
                score = avg_conf * (1.0 / (len(recent) + 1)) * min(len(recent), 5)
                scored.append({"symbol": symbol, "score": round(score, 4), "signal_count": len(recent),
                               "avg_confidence": round(avg_conf, 4)})
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored

    def get_signal_buffer(self, symbol: str = None) -> Dict[str, Any]:
        if symbol:
            return {"symbol": symbol, "signals": self._signal_buffer.get(symbol, [])}
        return {s: len(sigs) for s, sigs in self._signal_buffer.items()}

    def flush_stale_signals(self, max_age_seconds: float = 300) -> int:
        now = datetime.now()
        flushed = 0
        for symbol in list(self._signal_buffer.keys()):
            before = len(self._signal_buffer[symbol])
            self._signal_buffer[symbol] = [
                s for s in self._signal_buffer[symbol]
                if (now - datetime.fromisoformat(s["timestamp"])).total_seconds() <= max_age_seconds
            ]
            flushed += before - len(self._signal_buffer[symbol])
        return flushed

    def get_summary(self) -> Dict[str, Any]:
        return {"buffer_symbols": len(self._signal_buffer), "stats": dict(self._stats)}


# ═══════════════════════════════════════════════════════════════
# 优先级调度器
# ═══════════════════════════════════════════════════════════════

class PriorityScheduler:
    """优先级调度器：按策略重要性排序执行"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._priorities: Dict[str, int] = {}
        self._default_priority = config.get("coordination", {}).get("default_priority", 3)
        self._quota_per_strategy = config.get("coordination", {}).get("api_quota_per_strategy", 10)
        self._lock = asyncio.Lock()

    def set_priority(self, strategy_name: str, priority: int):
        self._priorities[strategy_name] = max(1, min(5, priority))

    def get_execution_order(self, strategy_names: List[str]) -> List[str]:
        return sorted(strategy_names, key=lambda n: self._priorities.get(n, self._default_priority))

    def allocate_quota(self, strategy_count: int, total_quota: int) -> Dict[str, int]:
        return {f"strategy_{i}": max(1, total_quota // max(strategy_count, 1)) for i in range(strategy_count)}

    def check_preempt(self, new_priority: int, current_priority: int) -> bool:
        return new_priority < current_priority

    def get_priority_summary(self) -> Dict[str, Any]:
        return {"priorities": dict(self._priorities), "default_priority": self._default_priority}


# ═══════════════════════════════════════════════════════════════
# 熔断联动器
# ═══════════════════════════════════════════════════════════════

class CircuitBreakerLinker:
    """熔断联动器：策略协调与全局熔断的桥梁"""
    
    def __init__(self, config: Dict[str, Any], coordinator: Optional[StrategyCoordinator] = None):
        self._config = config
        self._coordinator = coordinator
        self._lock = asyncio.Lock()
        self._blocked_strategies: Dict[str, str] = {}  # strategy -> reason
        self._recovery_limits: Dict[str, float] = {}    # strategy -> position_limit_pct
        self._breaker_level = 0
        self._global_blocked = False
        self._log: List[Dict[str, Any]] = []

    async def on_circuit_breaker(self, level: int, reason: str,
                                  affected_symbols: List[str] = None) -> Dict[str, Any]:
        async with self._lock:
            self._breaker_level = level
            actions = []
            if level >= 3:
                self._global_blocked = True
                actions.append("global_block")
            if level >= 4:
                for sname in (list(self._blocked_strategies.keys()) if self._coordinator is None
                             else self._coordinator._strategies.keys()):
                    self._blocked_strategies[sname] = f"L{level} breaker: {reason}"
                actions.append(f"blocked_all_strategies")
            self._log.append({"timestamp": datetime.now().isoformat(), "level": level,
                              "reason": reason, "actions": actions,
                              "affected_symbols": affected_symbols or []})
            if len(self._log) > 100:
                self._log = self._log[-100:]
            return {"status": "processed", "level": level, "actions": actions}

    async def apply_strategy_breaker(self, strategy_name: str, reason: str) -> Dict[str, Any]:
        async with self._lock:
            self._blocked_strategies[strategy_name] = reason
            self._log.append({"timestamp": datetime.now().isoformat(), "strategy": strategy_name,
                              "action": "block", "reason": reason})
            return {"status": "blocked", "strategy": strategy_name, "reason": reason}

    async def gradual_recovery(self, strategy_name: str) -> Dict[str, Any]:
        async with self._lock:
            current = self._recovery_limits.get(strategy_name, 0.1)
            new_limit = min(1.0, current + 0.15)
            self._recovery_limits[strategy_name] = new_limit
            if new_limit >= 1.0:
                self._blocked_strategies.pop(strategy_name, None)
            return {"strategy": strategy_name, "new_limit": new_limit, "fully_recovered": new_limit >= 1.0}

    async def get_breaker_state(self) -> Dict[str, Any]:
        return {"level": self._breaker_level, "global_blocked": self._global_blocked,
                "blocked_strategies": dict(self._blocked_strategies),
                "recovery_limits": dict(self._recovery_limits)}

    async def reset(self) -> Dict[str, Any]:
        """P1-⑥：熔断恢复时清空阻断状态（策略协调层写路径统一）。"""
        async with self._lock:
            self._breaker_level = 0
            self._global_blocked = False
            self._blocked_strategies.clear()
            self._recovery_limits.clear()
            self._log.append({"timestamp": datetime.now().isoformat(),
                              "action": "reset", "reason": "breaker recovery"})
            if len(self._log) > 100:
                self._log = self._log[-100:]
            return {"status": "reset"}

    def is_strategy_blocked(self, strategy_name: str) -> Tuple[bool, str]:
        if self._global_blocked:
            return True, "Global breaker active"
        if strategy_name in self._blocked_strategies:
            return True, self._blocked_strategies[strategy_name]
        return False, ""


# ═══════════════════════════════════════════════════════════════
# 策略健康监控器
# ═══════════════════════════════════════════════════════════════

class StrategyHealthMonitor:
    """策略健康监控器"""
    
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._lock = asyncio.Lock()
        self._metrics: Dict[str, Dict[str, List[float]]] = {}  # strategy -> {metric: [values]}
        self._scores: Dict[str, List[float]] = {}  # strategy -> [scores]
        self._max_history = config.get("coordination", {}).get("health_history_max", 100)

    def update_metrics(self, strategy_name: str, metrics: Dict[str, float]):
        if strategy_name not in self._metrics:
            self._metrics[strategy_name] = {}
        for k, v in metrics.items():
            if k not in self._metrics[strategy_name]:
                self._metrics[strategy_name][k] = []
            self._metrics[strategy_name][k].append(v)
            if len(self._metrics[strategy_name][k]) > self._max_history:
                self._metrics[strategy_name][k] = self._metrics[strategy_name][k][-self._max_history:]

    async def compute_health_score(self, strategy_name: str) -> float:
        if strategy_name not in self._metrics or not self._metrics[strategy_name]:
            return 50.0
        m = self._metrics[strategy_name]
        scores = {}
        if "activity" in m:
            recent = m["activity"][-10:]
            scores["activity"] = min(100, sum(recent) / max(len(recent), 1) * 100)
        if "signal_quality" in m:
            recent = m["signal_quality"][-10:]
            scores["signal_quality"] = sum(recent) / max(len(recent), 1) * 100
        if "fill_rate" in m:
            recent = m["fill_rate"][-10:]
            scores["fill_rate"] = sum(recent) / max(len(recent), 1) * 100
        if "slippage" in m:
            recent = m["slippage"][-10:]
            avg_slip = sum(recent) / max(len(recent), 1)
            scores["slippage"] = max(0, 100 - avg_slip * 100)
        if not scores:
            return 50.0
        total = sum(scores.values()) / len(scores)
        if strategy_name not in self._scores:
            self._scores[strategy_name] = []
        self._scores[strategy_name].append(total)
        if len(self._scores[strategy_name]) > self._max_history:
            self._scores[strategy_name] = self._scores[strategy_name][-self._max_history:]
        return round(total, 1)

    async def detect_health_trend(self, strategy_name: str) -> Dict[str, Any]:
        scores = self._scores.get(strategy_name, [])
        if len(scores) < 5:
            return {"trend": "insufficient_data", "scores": scores}
        recent = scores[-10:]
        if len(recent) < 3:
            return {"trend": "stable", "current": recent[-1], "scores": recent}
        slope = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        if slope > 1:
            trend = "improving"
        elif slope < -1:
            trend = "declining"
        else:
            trend = "stable"
        return {"trend": trend, "current": recent[-1], "slope": round(slope, 2), "scores": recent}

    async def diagnose(self, strategy_name: str) -> Dict[str, Any]:
        issues = []
        m = self._metrics.get(strategy_name, {})
        if "signal_quality" in m and m["signal_quality"]:
            recent = m["signal_quality"][-10:]
            if sum(recent) / len(recent) < 0.3:
                issues.append("low_signal_quality")
        if "fill_rate" in m and m["fill_rate"]:
            recent = m["fill_rate"][-10:]
            if sum(recent) / len(recent) < 0.5:
                issues.append("low_fill_rate")
        if "activity" in m and m["activity"]:
            recent = m["activity"][-20:]
            if sum(recent) / len(recent) < 0.1:
                issues.append("inactive_strategy")
        return {"strategy": strategy_name, "issues": issues, "issue_count": len(issues),
                "healthy": len(issues) == 0}

    async def get_health_report(self) -> Dict[str, Any]:
        report = {}
        for sname in self._metrics:
            score = await self.compute_health_score(sname)
            trend = await self.detect_health_trend(sname)
            diag = await self.diagnose(sname)
            report[sname] = {"health_score": score, "trend": trend.get("trend"), "diagnosis": diag}
        return report

    def get_summary(self) -> Dict[str, Any]:
        return {"monitored_strategies": len(self._metrics), "scores": {
            k: v[-1] if v else 50 for k, v in self._scores.items()}}


# ═══════════════════════════════════════════════════════════════
# 扩展 StrategyCoordinator（动态方法挂载）
# ═══════════════════════════════════════════════════════════════

def _extend_coordinator():
    """给 StrategyCoordinator 添加链接方法"""
    def link_signal_hub(self, hub: SignalAggregationHub):
        self._signal_hub = hub

    def link_priority_scheduler(self, scheduler: PriorityScheduler):
        self._priority_scheduler = scheduler

    def link_breaker(self, linker: CircuitBreakerLinker):
        self._breaker_linker = linker

    def link_health_monitor(self, monitor: StrategyHealthMonitor):
        self._health_monitor = monitor

    async def get_coordinated_signal(self, symbol: str) -> Dict[str, Any]:
        if hasattr(self, '_signal_hub') and self._signal_hub:
            return await self._signal_hub.compute_consensus(symbol)
        return {"consensus": "no_hub"}

    async def check_breaker_status(self) -> Dict[str, Any]:
        if hasattr(self, '_breaker_linker') and self._breaker_linker:
            return await self._breaker_linker.get_breaker_state()
        return {"status": "no_linker"}

    async def get_health_summary(self) -> Dict[str, Any]:
        if hasattr(self, '_health_monitor') and self._health_monitor:
            return await self._health_monitor.get_health_report()
        return {"status": "no_monitor"}

    StrategyCoordinator.link_signal_hub = link_signal_hub
    StrategyCoordinator.link_priority_scheduler = link_priority_scheduler
    StrategyCoordinator.link_breaker = link_breaker
    StrategyCoordinator.link_health_monitor = link_health_monitor
    StrategyCoordinator.get_coordinated_signal = get_coordinated_signal
    StrategyCoordinator.check_breaker_status = check_breaker_status
    StrategyCoordinator.get_health_summary = get_health_summary

_extend_coordinator()
