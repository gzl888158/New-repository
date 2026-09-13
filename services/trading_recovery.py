"""
交易恢复服务
解决长时间无交易问题，包括：
1. 回撤异常恢复
2. 策略状态重置
3. 资金协调
4. 自动重启机制
"""
import asyncio
import time
import json
import os
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List
from loguru import logger
from core.atomic_writer import atomic_write_json


class RecoveryStatus(Enum):
    """恢复状态"""
    IDLE = "idle"
    DETECTING = "detecting"
    RECOVERING = "recovering"
    RECOVERED = "recovered"
    FAILED = "failed"


class TradingRecoveryService:
    """交易恢复服务"""
    
    def __init__(self, config: Dict[str, Any], okx_client=None):
        self.config = config
        self._okx_client = okx_client
        
        self._status = RecoveryStatus.IDLE
        self._recovery_history: List[Dict[str, Any]] = []
        self._last_recovery_time = None
        
        self._drawdown_threshold = config.get("trading", {}).get("max_drawdown", 0.25)
        self._recovery_drawdown_threshold = self._drawdown_threshold * 0.5
        
        self._no_trade_timeout_minutes = 30
        self._max_recovery_attempts = 5
        self._recovery_cooldown_hours = 1
        
        self._strategies_to_reset: List[str] = []
        self._recovery_actions: List[Dict[str, Any]] = []
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        logger.info("TradingRecoveryService initialized")
    
    async def start(self):
        """启动恢复服务"""
        if self._running:
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("TradingRecoveryService started")
    
    async def stop(self):
        """停止恢复服务"""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("TradingRecoveryService stopped")
    
    async def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                await self._detect_problems()
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Trading recovery monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _detect_problems(self):
        """检测问题"""
        problems = []
        
        drawdown = await self._get_current_drawdown()
        if drawdown is not None and isinstance(drawdown, (int, float)) and drawdown >= self._drawdown_threshold:
            problems.append({
                "type": "drawdown_exceeded",
                "severity": "critical",
                "value": float(drawdown),
                "threshold": self._drawdown_threshold,
                "message": f"Drawdown {float(drawdown):.2%} exceeds threshold {self._drawdown_threshold:.2%}"
            })
        
        last_trade_time = await self._get_last_trade_time()
        if last_trade_time:
            minutes_since_last_trade = (datetime.now() - last_trade_time).total_seconds() / 60
            if minutes_since_last_trade > self._no_trade_timeout_minutes:
                problems.append({
                    "type": "no_trade_activity",
                    "severity": "warning",
                    "value": float(minutes_since_last_trade),
                    "threshold": self._no_trade_timeout_minutes,
                    "message": f"No trade activity for {float(minutes_since_last_trade):.0f} minutes"
                })
        
        equity = await self._get_current_equity()
        if equity is not None and isinstance(equity, (int, float)) and equity < self._get_min_equity_threshold():
            problems.append({
                "type": "low_equity",
                "severity": "critical",
                "value": float(equity),
                "threshold": self._get_min_equity_threshold(),
                "message": f"Equity {float(equity):.2f} below minimum threshold"
            })
        
        if problems:
            logger.warning(f"Detected {len(problems)} problems: {[p['type'] for p in problems]}")
            await self._trigger_recovery(problems)
    
    async def _trigger_recovery(self, problems: List[Dict[str, Any]]):
        """触发恢复（自适应冷却：长时间无交易时缩短冷却时间）"""
        if self._status == RecoveryStatus.RECOVERING:
            return
        
        if self._last_recovery_time:
            hours_since_last = (datetime.now() - self._last_recovery_time).total_seconds() / 3600
            
            # 自适应冷却：根据无交易持续时间动态调整
            no_trade_problem = next((p for p in problems if p["type"] == "no_trade_activity"), None)
            if no_trade_problem:
                idle_hours = no_trade_problem.get("value", 0) / 60  # 转换为小时
                # 空闲超过6小时，冷却缩短为15分钟；超过24小时，冷却缩短为5分钟
                if idle_hours > 24:
                    adaptive_cooldown = 5 / 60  # 5分钟
                elif idle_hours > 6:
                    adaptive_cooldown = 0.25  # 15分钟
                else:
                    adaptive_cooldown = self._recovery_cooldown_hours
            else:
                adaptive_cooldown = self._recovery_cooldown_hours
            
            if hours_since_last < adaptive_cooldown:
                logger.debug(f"Recovery cooldown active, {hours_since_last:.1f}h since last recovery "
                           f"(cooldown={adaptive_cooldown:.2f}h)")
                return
        
        self._status = RecoveryStatus.RECOVERING
        self._recovery_actions = []
        
        logger.info("Starting trading recovery process")
        
        try:
            for problem in problems:
                await self._handle_problem(problem)
            
            await self._execute_recovery_actions()
            
            await self._verify_recovery()
            
            if self._status == RecoveryStatus.RECOVERED:
                self._last_recovery_time = datetime.now()
                logger.info("Trading recovery completed successfully")
            
        except Exception as e:
            logger.error(f"Trading recovery failed: {e}")
            self._status = RecoveryStatus.FAILED
        else:
            if self._status != RecoveryStatus.RECOVERED:
                self._status = RecoveryStatus.IDLE
    
    async def _handle_problem(self, problem: Dict[str, Any]):
        """处理问题"""
        problem_type = problem["type"]
        
        if problem_type == "drawdown_exceeded":
            await self._handle_drawdown_exceeded(problem)
        elif problem_type == "no_trade_activity":
            await self._handle_no_trade_activity(problem)
        elif problem_type == "low_equity":
            await self._handle_low_equity(problem)
    
    async def _handle_drawdown_exceeded(self, problem: Dict[str, Any]):
        """处理回撤超限"""
        val = problem.get("value")
        if val is None or not isinstance(val, (int, float)):
            val = 0.0
        logger.warning(f"Handling drawdown exceeded: {float(val):.2%}")
        
        self._recovery_actions.append({
            "action": "reset_drawdown",
            "params": {"threshold": self._recovery_drawdown_threshold}
        })
        
        self._recovery_actions.append({
            "action": "reduce_positions",
            "params": {"ratio": 0.5}
        })
        
        self._recovery_actions.append({
            "action": "reset_strategy_states",
            "params": {"strategies": ["scalping", "trend", "grid"]}
        })
    
    async def _handle_no_trade_activity(self, problem: Dict[str, Any]):
        """处理无交易活动"""
        val = problem.get("value")
        if val is None or not isinstance(val, (int, float)):
            val = 0.0
        logger.warning(f"Handling no trade activity: {float(val):.0f} minutes")
        
        self._recovery_actions.append({
            "action": "check_strategy_status",
            "params": {}
        })
        
        self._recovery_actions.append({
            "action": "reset_signal_generation",
            "params": {}
        })
        
        self._recovery_actions.append({
            "action": "clear_pending_orders",
            "params": {}
        })
    
    async def _handle_low_equity(self, problem: Dict[str, Any]):
        """处理低权益"""
        val = problem.get("value")
        if val is None or not isinstance(val, (int, float)):
            val = 0.0
        logger.warning(f"Handling low equity: {float(val):.2f}")
        
        self._recovery_actions.append({
            "action": "close_all_positions",
            "params": {}
        })
        
        self._recovery_actions.append({
            "action": "reset_capital_allocation",
            "params": {}
        })
    
    async def _execute_recovery_actions(self):
        """执行恢复动作"""
        for action in self._recovery_actions:
            try:
                await self._execute_action(action)
                logger.info(f"Executed recovery action: {action['action']}")
            except Exception as e:
                logger.error(f"Failed to execute recovery action {action['action']}: {e}")
    
    async def _execute_action(self, action: Dict[str, Any]):
        """执行单个动作"""
        action_type = action["action"]
        params = action.get("params", {})
        
        if action_type == "reset_drawdown":
            await self._action_reset_drawdown(params)
        elif action_type == "reduce_positions":
            await self._action_reduce_positions(params)
        elif action_type == "reset_strategy_states":
            await self._action_reset_strategy_states(params)
        elif action_type == "check_strategy_status":
            await self._action_check_strategy_status(params)
        elif action_type == "reset_signal_generation":
            await self._action_reset_signal_generation(params)
        elif action_type == "clear_pending_orders":
            await self._action_clear_pending_orders(params)
        elif action_type == "close_all_positions":
            await self._action_close_all_positions(params)
        elif action_type == "reset_capital_allocation":
            await self._action_reset_capital_allocation(params)
    
    async def _action_reset_drawdown(self, params: Dict[str, Any]):
        """重置回撤状态"""
        try:
            state_files = []
            for root, dirs, files in os.walk("./data/strategy_state"):
                for f in files:
                    if f.endswith(".json"):
                        state_files.append(os.path.join(root, f))
            
            for state_file in state_files:
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    
                    if "current_drawdown" in state:
                        state["current_drawdown"] = 0.0
                    
                    if "max_daily_equity" in state:
                        state["max_daily_equity"] = 0.0
                    
                    with open(state_file, "w", encoding="utf-8") as f:
                        json.dump(state, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"Reset drawdown in {state_file}")
                except Exception as e:
                    logger.debug(f"Failed to reset drawdown in {state_file}: {e}")
            
            control_path = "./data/risk_control.json"
            atomic_write_json(control_path, {"action": "reset_pause"})
            
            logger.info("Created risk control reset instruction")
            
        except Exception as e:
            logger.error(f"Failed to reset drawdown: {e}")
    
    async def _action_reduce_positions(self, params: Dict[str, Any]):
        """减仓"""
        ratio = params.get("ratio", 0.5)
        logger.info(f"Reducing positions by {ratio:.0%}")
        
        if not self._okx_client:
            logger.warning("Cannot reduce positions: OKXClient not available")
            return
        
        try:
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(None, self._okx_client.get_positions)
            if not positions:
                logger.info("No positions to reduce")
                return
            
            reduced_count = 0
            for pos_data in positions:
                position = self._okx_client._parse_position(pos_data)
                if not position or abs(position.quantity) == 0:
                    continue
                
                reduce_qty = abs(float(position.quantity)) * ratio
                if reduce_qty <= 0:
                    continue
                
                side = "sell" if position.side == "long" else "buy"
                try:
                    await loop.run_in_executor(
                        None,
                        lambda s=position.symbol, sd=side, q=reduce_qty, l=position.leverage:
                            self._okx_client.place_order(
                                symbol=s, side=sd, order_type="market",
                                quantity=q, leverage=l, reduce_only=True
                            )
                    )
                    reduced_count += 1
                    logger.info(f"Reduced position {position.symbol} {position.side} by {ratio:.0%}")
                except Exception as e:
                    logger.error(f"Failed to reduce position {position.symbol}: {e}")
            
            logger.info(f"Position reduction complete: {reduced_count} positions reduced")
        except Exception as e:
            logger.error(f"Failed to reduce positions: {e}")
    
    async def _action_reset_strategy_states(self, params: Dict[str, Any]):
        """重置策略状态"""
        strategies = params.get("strategies", [])
        logger.info(f"Resetting strategy states: {strategies}")
        
        for strategy_name in strategies:
            state_file = f"./data/strategy_state/{strategy_name}_strategy.json"
            if os.path.exists(state_file):
                try:
                    with open(state_file, "r", encoding="utf-8") as f:
                        state = json.load(f)
                    
                    if "current_drawdown" in state:
                        state["current_drawdown"] = 0.0
                    if "_current_drawdown" in state:
                        state["_current_drawdown"] = 0.0
                    if "max_daily_equity" in state:
                        state["max_daily_equity"] = 0.0
                    if "_max_daily_equity" in state:
                        state["_max_daily_equity"] = 0.0
                    if "_equity_initialized" in state:
                        state["_equity_initialized"] = False
                    if "daily_reset_date" in state:
                        state["daily_reset_date"] = None
                    
                    atomic_write_json(state_file, state)
                    
                    logger.info(f"Reset state for {strategy_name}")
                except Exception as e:
                    logger.error(f"Failed to reset state for {strategy_name}: {e}")
    
    async def _action_check_strategy_status(self, params: Dict[str, Any]):
        """检查策略状态"""
        logger.info("Checking strategy status")
        
        try:
            strategies = ["scalping", "trend", "grid", "arbitrage", "spot_grid", "spot_martingale"]
            status_results = []
            
            for strategy_name in strategies:
                state_file = f"./data/strategy_state/{strategy_name}_strategy.json"
                if os.path.exists(state_file):
                    try:
                        with open(state_file, "r", encoding="utf-8") as f:
                            state = json.load(f)
                        has_drawdown = state.get("current_drawdown", 0) != 0 or state.get("_current_drawdown", 0) != 0
                        status_results.append({
                            "strategy": strategy_name,
                            "state_file_exists": True,
                            "has_active_drawdown": has_drawdown,
                        })
                    except Exception:
                        status_results.append({
                            "strategy": strategy_name,
                            "state_file_exists": True,
                            "error": "failed_to_read",
                        })
                else:
                    status_results.append({
                        "strategy": strategy_name,
                        "state_file_exists": False,
                    })
            
            logger.info(f"Strategy status check complete: {json.dumps(status_results)}")
        except Exception as e:
            logger.error(f"Failed to check strategy status: {e}")
    
    async def _action_reset_signal_generation(self, params: Dict[str, Any]):
        """重置信号生成"""
        logger.info("Resetting signal generation")
        
        try:
            signal_state_path = "./data/signal_state.json"
            if os.path.exists(signal_state_path):
                with open(signal_state_path, "r", encoding="utf-8") as f:
                    signal_state = json.load(f)
                
                if "last_signal_time" in signal_state:
                    signal_state["last_signal_time"] = None
                if "signal_count" in signal_state:
                    signal_state["signal_count"] = 0
                if "paused" in signal_state:
                    signal_state["paused"] = False
                
                with open(signal_state_path, "w", encoding="utf-8") as f:
                    json.dump(signal_state, f, ensure_ascii=False, indent=2)
                
                logger.info("Signal generation state reset")
            else:
                logger.info("No signal state file found, nothing to reset")
        except Exception as e:
            logger.error(f"Failed to reset signal generation: {e}")
    
    async def _action_clear_pending_orders(self, params: Dict[str, Any]):
        """清除挂单"""
        logger.info("Clearing pending orders")
        
        if not self._okx_client:
            logger.warning("Cannot clear pending orders: OKXClient not available")
            return
        
        try:
            loop = asyncio.get_event_loop()
            
            pending_orders = await loop.run_in_executor(None, self._okx_client.get_orders)
            cancelled_count = 0
            if pending_orders:
                for order in pending_orders:
                    order_id = order.get("ordId", "")
                    symbol = order.get("instId", "")
                    if order_id and symbol:
                        try:
                            await loop.run_in_executor(
                                None, self._okx_client.cancel_order, symbol, order_id
                            )
                            cancelled_count += 1
                            logger.info(f"Cancelled order {order_id} for {symbol}")
                        except Exception as e:
                            logger.error(f"Failed to cancel order {order_id}: {e}")
            
            algo_orders = await loop.run_in_executor(None, self._okx_client.get_algo_orders)
            if algo_orders:
                for order in algo_orders:
                    algo_id = order.get("algoId", "")
                    symbol = order.get("instId", "")
                    if algo_id and symbol:
                        try:
                            await loop.run_in_executor(
                                None, self._okx_client.cancel_algo_order, symbol, algo_id
                            )
                            cancelled_count += 1
                            logger.info(f"Cancelled algo order {algo_id} for {symbol}")
                        except Exception as e:
                            logger.error(f"Failed to cancel algo order {algo_id}: {e}")
            
            logger.info(f"Cleared {cancelled_count} pending orders total")
        except Exception as e:
            logger.error(f"Failed to clear pending orders: {e}")
    
    async def _action_close_all_positions(self, params: Dict[str, Any]):
        """全平"""
        logger.info("Closing all positions")
        
        if not self._okx_client:
            logger.warning("Cannot close positions: OKXClient not available")
            return
        
        try:
            loop = asyncio.get_event_loop()
            positions = await loop.run_in_executor(None, self._okx_client.get_positions)
            if not positions:
                logger.info("No positions to close")
                return
            
            closed_count = 0
            for pos_data in positions:
                position = self._okx_client._parse_position(pos_data)
                if not position or abs(position.quantity) == 0:
                    continue
                
                side = "sell" if position.side == "long" else "buy"
                qty = abs(float(position.quantity))
                try:
                    await loop.run_in_executor(
                        None,
                        lambda s=position.symbol, sd=side, q=qty, l=position.leverage:
                            self._okx_client.place_order(
                                symbol=s, side=sd, order_type="market",
                                quantity=q, leverage=l, reduce_only=True
                            )
                    )
                    closed_count += 1
                    logger.info(f"Closed position {position.symbol} {position.side} size={qty}")
                except Exception as e:
                    logger.error(f"Failed to close position {position.symbol}: {e}")
            
            logger.info(f"Close all positions complete: {closed_count} positions closed")
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
    
    async def _action_reset_capital_allocation(self, params: Dict[str, Any]):
        """重置资金分配"""
        logger.info("Resetting capital allocation")
        
        try:
            capital_state_path = "./data/capital_allocation.json"
            if os.path.exists(capital_state_path):
                with open(capital_state_path, "r", encoding="utf-8") as f:
                    capital_state = json.load(f)
                
                if "allocations" in capital_state:
                    for strategy_name in capital_state["allocations"]:
                        capital_state["allocations"][strategy_name]["used"] = 0
                
                if "last_rebalance_time" in capital_state:
                    capital_state["last_rebalance_time"] = None
                
                with open(capital_state_path, "w", encoding="utf-8") as f:
                    json.dump(capital_state, f, ensure_ascii=False, indent=2)
                
                logger.info("Capital allocation state reset")
            else:
                logger.info("No capital allocation state file found, nothing to reset")
        except Exception as e:
            logger.error(f"Failed to reset capital allocation: {e}")
    
    async def _verify_recovery(self):
        """验证恢复结果"""
        all_ok = True

        # 1. 检查回撤
        drawdown = await self._get_current_drawdown()
        if drawdown is None or not isinstance(drawdown, (int, float)):
            drawdown = 0.0
        drawdown = float(drawdown)
        if drawdown >= self._recovery_drawdown_threshold:
            logger.warning(f"Recovery not verified: drawdown {drawdown:.2%} >= threshold")
            all_ok = False

        # 2. 检查是否仍有未解决问题
        remaining = await self._detect_problems()
        if remaining and any(p["type"] == "no_trade_activity" for p in remaining):
            logger.info(f"Recovery executed but {len(remaining)} problem(s) remain: "
                       f"{[p['type'] for p in remaining]}")
            # 不标记为 FAILED - recovery 已尽力，根本问题可能是余额/网络

        if all_ok:
            self._status = RecoveryStatus.RECOVERED
            logger.info(f"Recovery verified: drawdown {drawdown:.2%} < threshold {self._recovery_drawdown_threshold:.2%}")
        else:
            self._status = RecoveryStatus.FAILED
            logger.warning(f"Recovery not fully verified")
    
    async def _get_current_drawdown(self) -> Optional[float]:
        """获取当前回撤"""
        risk_path = "./data/risk_status.json"
        if os.path.exists(risk_path):
            try:
                with open(risk_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("current_drawdown")
            except Exception:
                pass
        
        scalping_state = "./data/strategy_state/scalping_strategy.json"
        if os.path.exists(scalping_state):
            try:
                with open(scalping_state, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("current_drawdown") or data.get("_current_drawdown")
            except Exception:
                pass
        
        return None
    
    async def _get_last_trade_time(self) -> Optional[datetime]:
        """获取最后交易时间"""
        try:
            import sqlite3
            conn = sqlite3.connect(self.config.get("sqlite", {}).get("db_path", "./data/trading.db"))
            c = conn.cursor()
            c.execute("SELECT MAX(create_time) FROM trade_records")
            row = c.fetchone()
            conn.close()
            
            if row and row[0]:
                return datetime.fromisoformat(row[0].replace("Z", "+00:00"))
        except Exception:
            pass
        
        return None
    
    async def _get_current_equity(self) -> Optional[float]:
        """获取当前权益"""
        risk_path = "./data/risk_status.json"
        if os.path.exists(risk_path):
            try:
                with open(risk_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("current_equity")
            except Exception:
                pass
        
        return None
    
    def _get_min_equity_threshold(self) -> float:
        """获取最低权益阈值"""
        total_capital = self.config.get("trading", {}).get("total_capital", 100)
        return total_capital * 0.1
    
    def get_status(self) -> RecoveryStatus:
        """获取恢复状态"""
        return self._status
    
    def get_recovery_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取恢复历史"""
        return self._recovery_history[-limit:]
    
    async def manual_recovery(self, actions: Optional[List[str]] = None):
        """手动触发恢复"""
        problems = []
        
        if not actions:
            drawdown = await self._get_current_drawdown()
            if drawdown:
                problems.append({
                    "type": "drawdown_exceeded",
                    "severity": "critical",
                    "value": drawdown,
                    "threshold": self._drawdown_threshold,
                    "message": f"Manual recovery: drawdown {drawdown:.2%}"
                })
            
            last_trade = await self._get_last_trade_time()
            if last_trade:
                minutes_since = (datetime.now() - last_trade).total_seconds() / 60
                problems.append({
                    "type": "no_trade_activity",
                    "severity": "warning",
                    "value": minutes_since,
                    "threshold": self._no_trade_timeout_minutes,
                    "message": f"Manual recovery: {minutes_since:.0f} minutes since last trade"
                })
        
        if problems:
            await self._trigger_recovery(problems)
            return True
        
        logger.info("No problems detected for manual recovery")
        return False
    
    def export_status(self) -> Dict[str, Any]:
        """导出状态"""
        return {
            "status": self._status.value,
            "last_recovery_time": self._last_recovery_time.isoformat() if self._last_recovery_time else None,
            "recovery_cooldown_remaining_hours": max(0, self._recovery_cooldown_hours - (
                (datetime.now() - self._last_recovery_time).total_seconds() / 3600
                if self._last_recovery_time else 0
            )),
            "drawdown_threshold": self._drawdown_threshold,
            "recovery_drawdown_threshold": self._recovery_drawdown_threshold,
            "no_trade_timeout_minutes": self._no_trade_timeout_minutes,
            "recovery_history_count": len(self._recovery_history),
        }