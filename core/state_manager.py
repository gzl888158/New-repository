"""
全局状态管理器 - Global State Manager

功能：
1. 集中管理系统状态
2. 状态变更追踪
3. 状态持久化
4. 状态同步机制
"""

import asyncio
import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from loguru import logger


class StateCategory(Enum):
    """状态分类"""
    SYSTEM = "system"           # 系统状态
    TRADING = "trading"         # 交易状态
    RISK = "risk"               # 风控状态
    ACCOUNT = "account"         # 账户状态
    STRATEGY = "strategy"       # 策略状态
    EXECUTION = "execution"     # 执行状态


class StateKey:
    """状态键定义"""
    # 系统状态
    SYSTEM_HEALTH = "system.health"
    SYSTEM_UPTIME = "system.uptime"
    SYSTEM_VERSION = "system.version"
    
    # 交易状态
    TRADING_ENABLED = "trading.enabled"
    TRADING_PAUSED = "trading.paused"
    TOTAL_POSITIONS = "trading.total_positions"
    ACTIVE_SYMBOLS = "trading.active_symbols"
    
    # 风控状态
    RISK_LEVEL = "risk.level"
    DRAWDOWN_PCT = "risk.drawdown_pct"
    CIRCUIT_BREAKER_ACTIVE = "risk.circuit_breaker_active"
    LEVERAGE_LIMIT = "risk.leverage_limit"
    
    # 账户状态
    ACCOUNT_EQUITY = "account.equity"
    AVAILABLE_BALANCE = "account.available_balance"
    USED_MARGIN = "account.used_margin"
    MARGIN_LEVEL = "account.margin_level"
    
    # 策略状态
    STRATEGY_STATUS = "strategy.{name}.status"
    STRATEGY_PNL = "strategy.{name}.pnl"
    STRATEGY_POSITIONS = "strategy.{name}.positions"
    
    # 执行状态
    EXECUTION_QUEUE_SIZE = "execution.queue_size"
    EXECUTION_LATENCY = "execution.latency"
    ORDER_COUNT_TODAY = "execution.order_count_today"


class StateChange:
    """状态变更记录"""
    
    def __init__(self, key: str, old_value: Any, new_value: Any, timestamp: datetime = None):
        self.key = key
        self.old_value = old_value
        self.new_value = new_value
        self.timestamp = timestamp or datetime.now()
        self.diff = self._calculate_diff()
    
    def _calculate_diff(self) -> Any:
        """计算变更差异"""
        if isinstance(self.old_value, (int, float)) and isinstance(self.new_value, (int, float)):
            return self.new_value - self.old_value
        return None
    
    def is_significant(self, threshold: float = 0.01) -> bool:
        """判断是否为显著变更"""
        if self.diff is None:
            return True
        
        if self.old_value == 0:
            return abs(self.new_value) > 0
        
        return abs(self.diff / self.old_value) > threshold


class GlobalStateManager:
    """全局状态管理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self._config = config or {}
        
        # 状态存储
        self._state: Dict[str, Any] = {}
        self._state_lock = threading.RLock()
        
        # 状态变更记录
        self._change_history: List[StateChange] = []
        self._max_history_size = 1000
        
        # 状态监听器
        self._listeners: Dict[str, List[Callable]] = {}
        
        # 状态持久化
        self._persistence_path = "data/global_state.json"
        self._persistence_interval = 30  # 秒
        self._persistence_task = None
        
        # 状态同步
        self._sync_enabled = True
        self._sync_lock = asyncio.Lock()
        
        # 初始化默认状态
        self._initialize_default_state()
        
        logger.info("GlobalStateManager initialized")
    
    def _initialize_default_state(self):
        """初始化默认状态"""
        now = datetime.now()
        
        # 系统状态
        self.set(StateKey.SYSTEM_HEALTH, "healthy")
        self.set(StateKey.SYSTEM_UPTIME, now.isoformat())
        
        # 交易状态
        self.set(StateKey.TRADING_ENABLED, True)
        self.set(StateKey.TRADING_PAUSED, False)
        self.set(StateKey.TOTAL_POSITIONS, 0)
        self.set(StateKey.ACTIVE_SYMBOLS, [])
        
        # 风控状态
        self.set(StateKey.RISK_LEVEL, "NORMAL")
        self.set(StateKey.DRAWDOWN_PCT, 0.0)
        self.set(StateKey.CIRCUIT_BREAKER_ACTIVE, False)
        
        # 账户状态
        self.set(StateKey.ACCOUNT_EQUITY, 0.0)
        self.set(StateKey.AVAILABLE_BALANCE, 0.0)
        self.set(StateKey.USED_MARGIN, 0.0)
        self.set(StateKey.MARGIN_LEVEL, 0.0)
        
        # 执行状态
        self.set(StateKey.EXECUTION_QUEUE_SIZE, 0)
        self.set(StateKey.EXECUTION_LATENCY, 0.0)
        self.set(StateKey.ORDER_COUNT_TODAY, 0)
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取状态值"""
        with self._state_lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any, notify: bool = True):
        """设置状态值"""
        with self._state_lock:
            old_value = self._state.get(key)
            
            if old_value == value:
                return
            
            # 更新状态
            self._state[key] = value
            
            # 记录变更
            change = StateChange(key, old_value, value)
            self._change_history.append(change)
            
            # 清理历史记录
            if len(self._change_history) > self._max_history_size:
                self._change_history = self._change_history[-self._max_history_size:]
        
        # 通知监听器
        if notify:
            self._notify_listeners(key, change)
    
    def batch_set(self, updates: Dict[str, Any], notify: bool = True):
        """批量设置状态值"""
        changes = []
        
        with self._state_lock:
            for key, value in updates.items():
                old_value = self._state.get(key)
                
                if old_value != value:
                    self._state[key] = value
                    change = StateChange(key, old_value, value)
                    self._change_history.append(change)
                    changes.append(change)
            
            # 清理历史记录
            if len(self._change_history) > self._max_history_size:
                self._change_history = self._change_history[-self._max_history_size:]
        
        # 通知监听器
        if notify:
            for change in changes:
                self._notify_listeners(change.key, change)
    
    def _notify_listeners(self, key: str, change: StateChange):
        """通知状态监听器"""
        # 精确匹配
        if key in self._listeners:
            for listener in self._listeners[key]:
                try:
                    listener(change)
                except Exception as e:
                    logger.error(f"State listener error for {key}: {e}")
        
        # 模式匹配（支持通配符）
        for pattern, listeners in self._listeners.items():
            if '*' in pattern:
                if self._matches_pattern(key, pattern):
                    for listener in listeners:
                        try:
                            listener(change)
                        except Exception as e:
                            logger.error(f"State listener error for {pattern}: {e}")
    
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """检查key是否匹配模式"""
        # 简单的通配符匹配
        parts = pattern.split('.')
        key_parts = key.split('.')
        
        if len(parts) != len(key_parts):
            return False
        
        for p, k in zip(parts, key_parts):
            if p != '*' and p != k:
                return False
        
        return True
    
    def subscribe(self, key: str, listener: Callable[[StateChange], None]):
        """订阅状态变更"""
        if key not in self._listeners:
            self._listeners[key] = []
        
        if listener not in self._listeners[key]:
            self._listeners[key].append(listener)
            logger.info(f"Subscribed to state changes: {key}")
    
    def unsubscribe(self, key: str, listener: Callable[[StateChange], None]):
        """取消订阅"""
        if key in self._listeners and listener in self._listeners[key]:
            self._listeners[key].remove(listener)
            logger.info(f"Unsubscribed from state changes: {key}")
    
    def get_state(self, category: StateCategory = None) -> Dict[str, Any]:
        """获取状态（可选按分类过滤）"""
        with self._state_lock:
            if category:
                prefix = category.value + "."
                return {k: v for k, v in self._state.items() if k.startswith(prefix)}
            
            return dict(self._state)
    
    def get_change_history(self, limit: int = 100) -> List[StateChange]:
        """获取状态变更历史"""
        return self._change_history[-limit:]
    
    def get_significant_changes(self, threshold: float = 0.01) -> List[StateChange]:
        """获取显著变更"""
        return [c for c in self._change_history if c.is_significant(threshold)]
    
    def reset(self, category: StateCategory = None):
        """重置状态"""
        with self._state_lock:
            if category:
                prefix = category.value + "."
                keys_to_remove = [k for k in self._state if k.startswith(prefix)]
                for key in keys_to_remove:
                    del self._state[key]
            else:
                self._state.clear()
        
        logger.info(f"State reset for category: {category or 'all'}")
    
    def save_state(self):
        """保存状态到文件"""
        try:
            import os
            os.makedirs(os.path.dirname(self._persistence_path), exist_ok=True)
            
            state_data = {
                "state": self.get_state(),
                "timestamp": datetime.now().isoformat(),
                "version": "1.0"
            }
            
            with open(self._persistence_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            logger.debug("Global state saved")
        except Exception as e:
            logger.error(f"Failed to save global state: {e}")
    
    def load_state(self):
        """从文件加载状态"""
        try:
            import os
            if not os.path.exists(self._persistence_path):
                logger.debug("Global state file not found, using defaults")
                return
            
            with open(self._persistence_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            if "state" in state_data:
                self.batch_set(state_data["state"], notify=False)
                logger.info("Global state loaded")
        except Exception as e:
            logger.error(f"Failed to load global state: {e}")
    
    async def start_persistence(self):
        """启动状态持久化任务"""
        if self._persistence_task is not None:
            return
        
        async def persist_loop():
            while True:
                self.save_state()
                await asyncio.sleep(self._persistence_interval)
        
        self._persistence_task = asyncio.create_task(persist_loop())
        logger.info("State persistence started")
    
    async def stop_persistence(self):
        """停止状态持久化任务"""
        if self._persistence_task is not None:
            self._persistence_task.cancel()
            try:
                await self._persistence_task
            except asyncio.CancelledError:
                pass
            self._persistence_task = None
            logger.info("State persistence stopped")
    
    async def sync_state(self, external_state: Dict[str, Any]):
        """同步外部状态"""
        async with self._sync_lock:
            if not self._sync_enabled:
                return
            
            changes = {}
            for key, value in external_state.items():
                if self._state.get(key) != value:
                    changes[key] = value
            
            if changes:
                self.batch_set(changes)
                logger.debug(f"Synchronized {len(changes)} state changes")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """获取系统状态摘要"""
        return {
            "health": self.get(StateKey.SYSTEM_HEALTH),
            "uptime": self.get(StateKey.SYSTEM_UPTIME),
            "trading_enabled": self.get(StateKey.TRADING_ENABLED),
            "trading_paused": self.get(StateKey.TRADING_PAUSED),
            "risk_level": self.get(StateKey.RISK_LEVEL),
            "circuit_breaker": self.get(StateKey.CIRCUIT_BREAKER_ACTIVE),
            "total_positions": self.get(StateKey.TOTAL_POSITIONS),
            "equity": self.get(StateKey.ACCOUNT_EQUITY),
            "available_balance": self.get(StateKey.AVAILABLE_BALANCE),
            "used_margin": self.get(StateKey.USED_MARGIN),
            "margin_level": self.get(StateKey.MARGIN_LEVEL),
            "order_count_today": self.get(StateKey.ORDER_COUNT_TODAY)
        }


# 全局单例
_global_state: Optional[GlobalStateManager] = None
_global_state_lock = threading.Lock()


def get_global_state(config: Dict[str, Any] = None) -> GlobalStateManager:
    """获取全局状态管理器单例（线程安全）"""
    global _global_state
    
    if _global_state is None:
        with _global_state_lock:
            if _global_state is None:  # 双重检查锁定
                _global_state = GlobalStateManager(config)
    
    return _global_state


# 状态装饰器
def track_state(key: str):
    """
    状态追踪装饰器
    
    用法：
    @track_state("strategy.grid.pnl")
    def update_pnl(self, new_pnl):
        ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # 更新状态
            try:
                state_manager = get_global_state()
                if result is not None:
                    state_manager.set(key, result)
            except Exception as e:
                logger.error(f"Failed to track state {key}: {e}")
            
            return result
        return wrapper
    return decorator