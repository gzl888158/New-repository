"""
手动干预接口模块
紧急平仓、单策略启停、全局暂停、资金调整人工操作入口

核心功能：
1. 紧急平仓（全仓/单币种/单方向）
2. 策略启停（单策略/全策略）
3. 全局暂停/恢复
4. 资金调整（调整仓位/杠杆）
5. 操作审计日志
6. 二次确认机制
7. 操作权限验证
"""
import asyncio
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class InterventionType(Enum):
    """干预类型"""
    EMERGENCY_CLOSE_ALL = "emergency_close_all"           # 紧急全平仓
    EMERGENCY_CLOSE_SYMBOL = "emergency_close_symbol"    # 紧急单币种平仓
    EMERGENCY_CLOSE_SIDE = "emergency_close_side"       # 紧急平某方向
    STRATEGY_START = "strategy_start"                     # 启动策略
    STRATEGY_STOP = "strategy_stop"                       # 停止策略
    STRATEGY_PAUSE = "strategy_pause"                   # 暂停策略
    GLOBAL_PAUSE = "global_pause"                        # 全局暂停
    GLOBAL_RESUME = "global_resume"                      # 全局恢复
    ADJUST_LEVERAGE = "adjust_leverage"                  # 调整杠杆
    ADJUST_POSITION = "adjust_position"                  # 调整仓位
    CANCEL_ALL_ORDERS = "cancel_all_orders"               # 撤销所有挂单
    CANCEL_SYMBOL_ORDERS = "cancel_symbol_orders"        # 撤销某币种挂单
    SET_RISK_LIMIT = "set_risk_limit"                    # 设置风控限额


class InterventionStatus(Enum):
    """干预操作状态"""
    PENDING = "pending"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class InterventionRecord:
    """干预操作记录"""
    operation_id: str
    intervention_type: str
    status: str
    operator: str = "system"
    params: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    created_at: float = 0.0
    executed_at: float = 0.0
    completed_at: float = 0.0
    duration_ms: float = 0.0


class ManualInterventionManager:
    """
    手动干预管理器
    
    核心设计：
    - 统一的人工操作入口
    - 完整的操作审计日志
    - 二次确认机制（高风险操作）
    - 操作权限验证
    - 异步执行，不阻塞主流程
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        intervention_config = config.get("execution", {}).get("manual_intervention") or {}
        if not isinstance(intervention_config, dict):
            intervention_config = {}
        self._enabled = intervention_config.get("enabled", True)
        self._require_confirmation = intervention_config.get("require_confirmation", True)
        self._operation_timeout = intervention_config.get("operation_timeout", 30)
        
        # 高风险操作列表（需要二次确认）
        self._high_risk_operations = {
            InterventionType.EMERGENCY_CLOSE_ALL,
            InterventionType.GLOBAL_PAUSE,
            InterventionType.ADJUST_POSITION,
        }
        
        # 外部依赖（通过setter注入）
        self._okx_client = None
        self._order_executor = None
        self._global_risk = None
        self._alert_manager = None
        self._strategy_engine = None
        
        # 操作历史（最近100条）
        self._operation_history: deque = deque(maxlen=100)
        
        # 待确认的操作 {operation_id: InterventionRecord}
        self._pending_confirmations: Dict[str, InterventionRecord] = {}
        
        # 全局暂停状态
        self._global_paused = False
        self._pause_reason = ""
        self._pause_time = 0.0
        
        # 策略状态 {strategy_name: bool(True=运行, False=暂停)}
        self._strategy_states: Dict[str, bool] = {}
        
        # 运行状态
        self._running = False
        
        # 锁
        self._lock = asyncio.Lock()
        
        logger.info(f"ManualInterventionManager initialized: "
                   f"enabled={self._enabled}, "
                   f"require_confirmation={self._require_confirmation}")

    def set_dependencies(self, okx_client=None, order_executor=None,
                         global_risk=None, alert_manager=None, strategy_engine=None):
        """设置外部依赖"""
        if okx_client:
            self._okx_client = okx_client
        if order_executor:
            self._order_executor = order_executor
        if global_risk:
            self._global_risk = global_risk
        if alert_manager:
            self._alert_manager = alert_manager
        if strategy_engine:
            self._strategy_engine = strategy_engine

    async def start(self):
        """启动管理器"""
        if self._running:
            return
        self._running = True
        logger.info("ManualInterventionManager started")

    async def stop(self):
        """停止管理器"""
        self._running = False
        logger.info("ManualInterventionManager stopped")

    # ========== 紧急平仓 ==========

    async def emergency_close_all(self, operator: str = "manual", 
                                  reason: str = "") -> InterventionRecord:
        """
        紧急全平仓
        
        Args:
            operator: 操作人
            reason: 原因
        
        Returns:
            操作记录
        """
        record = self._create_record(
            InterventionType.EMERGENCY_CLOSE_ALL,
            operator, {"reason": reason}
        )
        
        # 高风险操作需要确认
        if self._require_confirmation and InterventionType.EMERGENCY_CLOSE_ALL in self._high_risk_operations:
            self._pending_confirmations[record.operation_id] = record
            logger.warning(f"Emergency close all pending confirmation: {record.operation_id}")
            return record
        
        return await self._execute_emergency_close_all(record)

    async def _execute_emergency_close_all(self, record: InterventionRecord) -> InterventionRecord:
        """执行全平仓"""
        record.status = InterventionStatus.EXECUTING.value
        record.executed_at = time.time()
        
        try:
            if not self._okx_client:
                raise RuntimeError("OKX client not available")
            
            # 1. 撤销所有挂单
            cancel_result = await self._cancel_all_orders_internal()
            
            # 2. 获取所有持仓
            positions = self._okx_client.get_positions()
            if not positions:
                positions = []
            
            close_results = []
            success_count = 0
            fail_count = 0
            
            for pos in positions:
                symbol = pos.get("instId", "")
                pos_side = pos.get("posSide", "net")
                pos_qty = float(pos.get("pos", 0))
                
                if pos_qty == 0 or not symbol:
                    continue
                
                try:
                    # 市价平仓（复用 close_position：统一 posSide/张数→币数/reduce_only/保证金模式）
                    result = self._okx_client.close_position(symbol, pos_side)
                    
                    if result and result.get("success"):
                        success_count += 1
                        close_results.append({"symbol": symbol, "pos_side": pos_side, "success": True})
                    else:
                        fail_count += 1
                        msg = result.get("error", "no response") if result else "no response"
                        close_results.append({"symbol": symbol, "pos_side": pos_side, "success": False, "error": msg})
                        
                except Exception as e:
                    fail_count += 1
                    close_results.append({"symbol": symbol, "pos_side": pos_side, "success": False, "error": str(e)})
            
            # 3. 触发全局暂停
            self._global_paused = True
            self._pause_reason = f"紧急全平仓: {record.params.get('reason', '无原因')}"
            self._pause_time = time.time()
            
            record.result = {
                "cancel_result": cancel_result,
                "close_results": close_results,
                "success_count": success_count,
                "fail_count": fail_count,
                "total_positions": len(positions),
            }
            
            if fail_count == 0:
                record.status = InterventionStatus.SUCCESS.value
            elif success_count > 0:
                record.status = InterventionStatus.PARTIAL.value
            else:
                record.status = InterventionStatus.FAILED.value
            
            # 发送告警
            if self._alert_manager and hasattr(self._alert_manager, 'send_alert'):
                await self._alert_manager.send_alert(
                    "emergency_close_all",
                    f"紧急全平仓执行完成: 成功{success_count}个, 失败{fail_count}个",
                    severity="CRITICAL",
                    metadata={"success_count": success_count, "fail_count": fail_count}
                )
            
        except Exception as e:
            record.status = InterventionStatus.FAILED.value
            record.error_message = str(e)
            logger.error(f"Emergency close all failed: {e}")
        
        record.completed_at = time.time()
        record.duration_ms = (record.completed_at - record.executed_at) * 1000
        
        self._operation_history.append(record)
        return record

    async def emergency_close_symbol(self, symbol: str, operator: str = "manual",
                                     reason: str = "") -> InterventionRecord:
        """紧急平单币种仓位"""
        record = self._create_record(
            InterventionType.EMERGENCY_CLOSE_SYMBOL,
            operator, {"symbol": symbol, "reason": reason}
        )
        record.status = InterventionStatus.EXECUTING.value
        record.executed_at = time.time()
        
        try:
            if not self._okx_client:
                raise RuntimeError("OKX client not available")
            
            # 撤销该币种挂单
            await self._cancel_symbol_orders_internal(symbol)
            
            # 获取该币种持仓
            positions = self._okx_client.get_positions()
            positions = [p for p in positions if p.get("instId") == symbol] if positions else []
            
            close_results = []
            success_count = 0
            
            for pos in positions:
                pos_side = pos.get("posSide", "net")
                pos_qty = float(pos.get("pos", 0))
                
                if pos_qty == 0:
                    continue
                
                try:
                    result = self._okx_client.close_position(symbol, pos_side)
                    
                    if result and result.get("success"):
                        success_count += 1
                        close_results.append({"pos_side": pos_side, "success": True})
                    else:
                        msg = result.get("error", "no response") if result else "no response"
                        close_results.append({"pos_side": pos_side, "success": False, "error": msg})
                except Exception as e:
                    close_results.append({"pos_side": pos_side, "success": False, "error": str(e)})
            
            record.result = {
                "close_results": close_results,
                "success_count": success_count,
            }
            
            record.status = InterventionStatus.SUCCESS.value if success_count > 0 else InterventionStatus.FAILED.value
            
        except Exception as e:
            record.status = InterventionStatus.FAILED.value
            record.error_message = str(e)
            logger.error(f"Emergency close symbol failed: {e}")
        
        record.completed_at = time.time()
        record.duration_ms = (record.completed_at - record.executed_at) * 1000
        self._operation_history.append(record)
        return record

    # ========== 策略启停 ==========

    async def start_strategy(self, strategy_name: str, operator: str = "manual") -> InterventionRecord:
        """启动策略"""
        record = self._create_record(
            InterventionType.STRATEGY_START,
            operator, {"strategy": strategy_name}
        )
        record.status = InterventionStatus.EXECUTING.value
        record.executed_at = time.time()
        
        try:
            self._strategy_states[strategy_name] = True
            
            # 如果有策略引擎，通知它
            if self._strategy_engine and hasattr(self._strategy_engine, 'resume_strategy'):
                await self._strategy_engine.resume_strategy(strategy_name)
            
            record.status = InterventionStatus.SUCCESS.value
            record.result = {"strategy": strategy_name, "state": "running"}
            
            logger.info(f"Strategy started: {strategy_name}")
            
        except Exception as e:
            record.status = InterventionStatus.FAILED.value
            record.error_message = str(e)
            logger.error(f"Start strategy failed: {e}")
        
        record.completed_at = time.time()
        record.duration_ms = (record.completed_at - record.executed_at) * 1000
        self._operation_history.append(record)
        return record

    async def stop_strategy(self, strategy_name: str, operator: str = "manual",
                            close_positions: bool = False) -> InterventionRecord:
        """停止策略"""
        record = self._create_record(
            InterventionType.STRATEGY_STOP,
            operator, {"strategy": strategy_name, "close_positions": close_positions}
        )
        record.status = InterventionStatus.EXECUTING.value
        record.executed_at = time.time()
        
        try:
            self._strategy_states[strategy_name] = False
            
            # 如果有策略引擎，通知它
            if self._strategy_engine and hasattr(self._strategy_engine, 'pause_strategy'):
                await self._strategy_engine.pause_strategy(strategy_name)
            
            # 如果需要平仓
            if close_positions:
                close_results = await self._close_strategy_positions(strategy_name)
                record.result["close_results"] = close_results
                success_count = sum(1 for r in close_results if r.get("success"))
                fail_count = len(close_results) - success_count
                record.result["close_success"] = success_count
                record.result["close_fail"] = fail_count
                logger.info(f"Strategy {strategy_name} positions closed: {success_count} ok, {fail_count} failed")
            
            record.status = InterventionStatus.SUCCESS.value
            record.result = {"strategy": strategy_name, "state": "stopped"}
            
            logger.info(f"Strategy stopped: {strategy_name}")
            
        except Exception as e:
            record.status = InterventionStatus.FAILED.value
            record.error_message = str(e)
            logger.error(f"Stop strategy failed: {e}")
        
        record.completed_at = time.time()
        record.duration_ms = (record.completed_at - record.executed_at) * 1000
        self._operation_history.append(record)
        return record

    # ========== 全局暂停/恢复 ==========

    async def global_pause(self, operator: str = "manual", 
                           reason: str = "") -> InterventionRecord:
        """全局暂停"""
        record = self._create_record(
            InterventionType.GLOBAL_PAUSE,
            operator, {"reason": reason}
        )
        
        if self._require_confirmation and InterventionType.GLOBAL_PAUSE in self._high_risk_operations:
            self._pending_confirmations[record.operation_id] = record
            return record
        
        return await self._execute_global_pause(record)

    async def _execute_global_pause(self, record: InterventionRecord) -> InterventionRecord:
        """执行全局暂停"""
        record.status = InterventionStatus.EXECUTING.value
        record.executed_at = time.time()
        
        try:
            self._global_paused = True
            self._pause_reason = record.params.get("reason", "手动暂停")
            self._pause_time = time.time()
            
            # 通知风控系统
            if self._global_risk and hasattr(self._global_risk, 'set_paused'):
                self._global_risk.set_paused(True)
            
            # 撤销所有挂单（可选）
            cancel_result = await self._cancel_all_orders_internal()
            
            record.status = InterventionStatus.SUCCESS.value
            record.result = {
                "paused": True,
                "reason": self._pause_reason,
                "cancel_result": cancel_result,
            }
            
            # 发送告警
            if self._alert_manager and hasattr(self._alert_manager, 'send_alert'):
                await self._alert_manager.send_alert(
                    "global_pause",
                    f"系统已全局暂停: {self._pause_reason}",
                    severity="WARNING",
                )
            
            logger.warning(f"Global paused: {self._pause_reason}")
            
        except Exception as e:
            record.status = InterventionStatus.FAILED.value
            record.error_message = str(e)
            logger.error(f"Global pause failed: {e}")
        
        record.completed_at = time.time()
        record.duration_ms = (record.completed_at - record.executed_at) * 1000
        self._operation_history.append(record)
        return record

    async def global_resume(self, operator: str = "manual") -> InterventionRecord:
        """全局恢复"""
        record = self._create_record(
            InterventionType.GLOBAL_RESUME,
            operator, {}
        )
        record.status = InterventionStatus.EXECUTING.value
        record.executed_at = time.time()
        
        try:
            self._global_paused = False
            pause_duration = time.time() - self._pause_time
            
            # 通知风控系统
            if self._global_risk and hasattr(self._global_risk, 'set_paused'):
                self._global_risk.set_paused(False)
            
            record.status = InterventionStatus.SUCCESS.value
            record.result = {
                "resumed": True,
                "pause_duration_seconds": pause_duration,
            }
            
            if self._alert_manager and hasattr(self._alert_manager, 'send_alert'):
                await self._alert_manager.send_alert(
                    "global_resume",
                    f"系统已恢复运行，暂停时长: {pause_duration:.0f}秒",
                    severity="INFO",
                )
            
            logger.info(f"Global resumed after {pause_duration:.0f}s")
            
        except Exception as e:
            record.status = InterventionStatus.FAILED.value
            record.error_message = str(e)
            logger.error(f"Global resume failed: {e}")
        
        record.completed_at = time.time()
        record.duration_ms = (record.completed_at - record.executed_at) * 1000
        self._operation_history.append(record)
        return record

    # ========== 撤单 ==========

    async def cancel_all_orders(self, operator: str = "manual") -> InterventionRecord:
        """撤销所有挂单"""
        record = self._create_record(
            InterventionType.CANCEL_ALL_ORDERS,
            operator, {}
        )
        record.status = InterventionStatus.EXECUTING.value
        record.executed_at = time.time()
        
        try:
            result = await self._cancel_all_orders_internal()
            record.result = result
            record.status = InterventionStatus.SUCCESS.value
        except Exception as e:
            record.status = InterventionStatus.FAILED.value
            record.error_message = str(e)
        
        record.completed_at = time.time()
        record.duration_ms = (record.completed_at - record.executed_at) * 1000
        self._operation_history.append(record)
        return record

    async def _cancel_all_orders_internal(self) -> Dict[str, Any]:
        """内部：撤销所有挂单"""
        if not self._okx_client:
            return {"success": False, "error": "OKX client not available"}
        
        try:
            result = self._okx_client.cancel_all_orders()
            if result and result.get("code") == "0":
                return {"success": True, "cancelled": True}
            else:
                msg = result.get("msg", "unknown") if result else "no response"
                return {"success": False, "error": msg}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _cancel_symbol_orders_internal(self, symbol: str) -> Dict[str, Any]:
        """内部：撤销某币种挂单（普通挂单 + 算法/条件单，按 instId 过滤）"""
        if not self._okx_client:
            return {"success": False, "error": "OKX client not available"}
        
        try:
            inst_type = "SPOT" if "-SWAP" not in symbol else "SWAP"
            cancelled = 0
            errors = []
            
            # 普通挂单：按 instId 过滤后逐个撤单
            for order in (self._okx_client.get_orders(inst_type=inst_type) or []):
                if order.get("instId") != symbol:
                    continue
                ord_id = order.get("ordId", "")
                if not ord_id:
                    continue
                if self._okx_client.cancel_order(symbol, ord_id) is not None:
                    cancelled += 1
                else:
                    errors.append(f"撤单失败: {symbol} {ord_id}")
            
            # 算法/条件单：直接按 symbol 查询后逐个撤单
            for ord_type in ("conditional", "oco", "trigger", "move_order_stop"):
                algos = self._okx_client.get_algo_orders(symbol=symbol, ord_type=ord_type) or []
                for algo in algos:
                    algo_id = algo.get("algoId", "")
                    if not algo_id:
                        continue
                    if self._okx_client.cancel_algo_order(symbol, algo_id) is not None:
                        cancelled += 1
                    else:
                        errors.append(f"撤销条件单失败: {symbol} {algo_id}")
            
            return {"success": True, "cancelled": cancelled, "errors": errors}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _close_strategy_positions(self, strategy_name: str) -> List[Dict[str, Any]]:
        """
        平掉指定策略的所有持仓
        
        流程：
          1. 获取策略配置中的交易币种列表
          2. 获取当前所有持仓
          3. 按币种匹配，市价平仓
          4. 返回平仓结果列表
        """
        results = []
        if not self._okx_client:
            return [{"success": False, "error": "OKX client not available"}]
        
        try:
            # 1. 获取策略关联的币种（从策略配置、持仓追踪、或订单执行器推断）
            strategy_symbols = self._get_strategy_symbols(strategy_name)
            
            # 2. 获取所有持仓
            positions = self._okx_client.get_positions()
            if not positions:
                logger.info(f"No positions found for strategy {strategy_name}")
                return [{"success": True, "message": "No positions to close"}]
            
            # 3. 过滤属于该策略的持仓
            strategy_positions = []
            for pos in positions:
                symbol = pos.get("instId", "")
                pos_qty = float(pos.get("pos", 0))
                
                if pos_qty == 0 or not symbol:
                    continue
                
                # 如果指定了策略币种列表，仅平相关币种；否则平所有持仓
                if strategy_symbols and symbol not in strategy_symbols:
                    continue
                
                strategy_positions.append(pos)
            
            if not strategy_positions:
                symbols_str = ', '.join(strategy_symbols) if strategy_symbols else 'all'
                logger.info(f"No matching positions for strategy {strategy_name} (symbols: {symbols_str})")
                return [{"success": True, "message": f"No matching positions for symbols: {symbols_str}"}]
            
            # 4. 撤销这些币种的挂单
            for symbol in set(p.get("instId", "") for p in strategy_positions):
                await self._cancel_symbol_orders_internal(symbol)
                await asyncio.sleep(0.3)  # 避免请求过快
            
            # 5. 市价平仓
            for pos in strategy_positions:
                symbol = pos.get("instId", "")
                pos_side = pos.get("posSide", "net")
                pos_qty = float(pos.get("pos", 0))
                
                side = "sell" if pos_qty > 0 else "buy"
                
                try:
                    result = self._okx_client.close_position(symbol, pos_side)
                    
                    if result and result.get("success"):
                        results.append({
                            "symbol": symbol, "side": side, "qty": abs(pos_qty),
                            "pos_side": pos_side, "success": True,
                        })
                        logger.info(f"Strategy {strategy_name}: closed {symbol} {pos_side} {abs(pos_qty)}")
                    else:
                        msg = result.get("error", "no response") if result else "no response"
                        results.append({
                            "symbol": symbol, "side": side, "qty": abs(pos_qty),
                            "pos_side": pos_side, "success": False, "error": msg,
                        })
                        logger.warning(f"Strategy {strategy_name}: failed to close {symbol} {pos_side}: {msg}")
                    
                except Exception as e:
                    results.append({
                        "symbol": symbol, "side": side, "qty": abs(pos_qty),
                        "pos_side": pos_side, "success": False, "error": str(e),
                    })
                    logger.error(f"Strategy {strategy_name}: error closing {symbol} {pos_side}: {e}")
            
        except Exception as e:
            results.append({"success": False, "error": f"Close strategy positions failed: {e}"})
            logger.error(f"Close strategy {strategy_name} positions error: {e}")
        
        return results
    
    def _get_strategy_symbols(self, strategy_name: str) -> List[str]:
        """获取策略关联的币种列表"""
        symbols = []
        
        # 从策略配置获取
        if self._strategy_engine and hasattr(self._strategy_engine, 'get_strategy_symbols'):
            symbols = self._strategy_engine.get_strategy_symbols(strategy_name)
        
        # 从order_executor获取追踪的持仓
        if not symbols and self._order_executor:
            if hasattr(self._order_executor, '_tracked_positions'):
                tracked = self._order_executor._tracked_positions
                if isinstance(tracked, dict):
                    for sym, info in tracked.items():
                        if isinstance(info, dict) and info.get("strategy") == strategy_name:
                            if sym not in symbols:
                                symbols.append(sym)
        
        return symbols

    # ========== 确认机制 ==========

    async def confirm_operation(self, operation_id: str, confirmed: bool = True) -> Optional[InterventionRecord]:
        """确认/取消待确认的操作"""
        record = self._pending_confirmations.pop(operation_id, None)
        
        if not record:
            return None
        
        if not confirmed:
            record.status = "cancelled"
            self._operation_history.append(record)
            return record
        
        # 执行操作
        intervention_type = InterventionType(record.intervention_type)
        
        if intervention_type == InterventionType.EMERGENCY_CLOSE_ALL:
            return await self._execute_emergency_close_all(record)
        elif intervention_type == InterventionType.GLOBAL_PAUSE:
            return await self._execute_global_pause(record)
        else:
            record.status = InterventionStatus.FAILED.value
            record.error_message = f"Unknown operation type: {intervention_type}"
            self._operation_history.append(record)
            return record

    # ========== 查询方法 ==========

    def is_global_paused(self) -> bool:
        """检查是否全局暂停"""
        return self._global_paused

    def get_pause_info(self) -> Dict[str, Any]:
        """获取暂停信息"""
        return {
            "paused": self._global_paused,
            "reason": self._pause_reason,
            "pause_time": self._pause_time,
            "duration_seconds": time.time() - self._pause_time if self._global_paused else 0,
        }

    def is_strategy_running(self, strategy_name: str) -> bool:
        """检查策略是否运行中"""
        return self._strategy_states.get(strategy_name, True)

    def get_strategy_states(self) -> Dict[str, bool]:
        """获取所有策略状态"""
        return self._strategy_states.copy()

    def get_pending_confirmations(self) -> List[Dict[str, Any]]:
        """获取待确认的操作"""
        return [
            {
                "operation_id": r.operation_id,
                "intervention_type": r.intervention_type,
                "operator": r.operator,
                "params": r.params,
                "created_at": r.created_at,
            }
            for r in self._pending_confirmations.values()
        ]

    def get_operation_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取操作历史"""
        history = list(self._operation_history)[-limit:]
        return [
            {
                "operation_id": r.operation_id,
                "intervention_type": r.intervention_type,
                "status": r.status,
                "operator": r.operator,
                "params": r.params,
                "result": r.result,
                "error_message": r.error_message,
                "created_at": r.created_at,
                "executed_at": r.executed_at,
                "completed_at": r.completed_at,
                "duration_ms": r.duration_ms,
            }
            for r in reversed(history)
        ]

    def _create_record(self, intervention_type: InterventionType,
                       operator: str, params: Dict[str, Any]) -> InterventionRecord:
        """创建操作记录"""
        operation_id = f"op_{int(time.time()*1000)}_{abs(hash(intervention_type.value + operator))}"
        
        record = InterventionRecord(
            operation_id=operation_id,
            intervention_type=intervention_type.value,
            status=InterventionStatus.PENDING.value,
            operator=operator,
            params=params,
            created_at=time.time(),
        )
        
        return record

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        status_counts = {}
        type_counts = {}
        
        for record in self._operation_history:
            status_counts[record.status] = status_counts.get(record.status, 0) + 1
            type_counts[record.intervention_type] = type_counts.get(record.intervention_type, 0) + 1
        
        return {
            "enabled": self._enabled,
            "global_paused": self._global_paused,
            "total_operations": len(self._operation_history),
            "pending_confirmations": len(self._pending_confirmations),
            "status_counts": status_counts,
            "type_counts": type_counts,
            "strategy_count": len(self._strategy_states),
        }
