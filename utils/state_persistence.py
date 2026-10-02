"""
策略状态持久化模块
所有策略继承 PersistentStrategy 基类，获得状态持久化与重启恢复能力。
支持 Redis（首选）和本地 JSON 文件（兜底）两种后端。
"""
import json
import os
import time
import asyncio
from datetime import datetime
from typing import Any, Dict, Optional
from loguru import logger

from core.strategy_enterprise import EnterpriseStrategyMixin


class StatePersistence:
    """策略状态持久化：Redis 优先，JSON 文件兜底。"""

    def __init__(self, strategy_name: str, redis_cache=None, state_dir: str = "./data/strategy_state"):
        self.strategy_name = strategy_name
        self.redis_cache = redis_cache
        self.state_dir = state_dir
        self._redis_prefix = f"strategy:{strategy_name}"
        self._json_path = os.path.join(state_dir, f"{strategy_name}.json")
        self._lock = asyncio.Lock()
        # 确保目录存在
        try:
            os.makedirs(state_dir, exist_ok=True)
        except Exception:
            pass

    def _is_redis_available(self) -> bool:
        """检查 Redis 是否可用（非 None 且连接正常）"""
        if not self.redis_cache:
            return False
        # 检查是否有 redis client 属性
        client = getattr(self.redis_cache, 'redis_client', None) or getattr(self.redis_cache, 'client', None)
        return client is not None

    async def save_state(self, state: Dict[str, Any]) -> bool:
        """保存状态到 Redis（首选）和 JSON 文件（兜底），双写保证至少一处成功。"""
        # 深拷贝调用方 dict，避免 _meta 字段污染调用方数据
        import copy
        state_copy = copy.deepcopy(state) if isinstance(state, dict) else {}
        # 加入元数据
        state_copy['_meta'] = {
            'strategy': self.strategy_name,
            'saved_at': datetime.now().isoformat(),
            'timestamp': time.time(),
            'version': 1,
        }
        # JSON 安全序列化：过滤不可序列化值，default=str 兜底
        try:
            serialized = json.dumps(state_copy, default=str, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            logger.error(f"[{self.strategy_name}] 状态序列化失败: {e}")
            return False

        success = False
        # 1. 尝试 Redis
        if self._is_redis_available():
            try:
                async with self._lock:
                    await self.redis_cache.set(f"{self._redis_prefix}:state", serialized, ttl=86400)
                success = True
            except Exception as e:
                logger.debug(f"[{self.strategy_name}] Redis 保存失败，降级到 JSON: {e}")

        # 2. JSON 文件兜底（无论如何都写一份，保证重启可恢复）
        try:
            async with self._lock:
                # 原子写入：先写临时文件再 rename
                tmp_path = self._json_path + '.tmp'
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    f.write(serialized)
                os.replace(tmp_path, self._json_path)
            success = True
        except Exception as e:
            logger.error(f"[{self.strategy_name}] JSON 状态保存失败: {e}")

        return success

    async def load_state(self, max_age_seconds: int = 86400) -> Optional[Dict[str, Any]]:
        """加载状态：优先 Redis，其次 JSON 文件。
        max_age_seconds: 状态最大有效期（秒），超过则视为过期返回 None。
        """
        state = None
        # 1. 优先 Redis
        if self._is_redis_available():
            try:
                data = await self.redis_cache.get(f"{self._redis_prefix}:state")
                if data:
                    state = json.loads(data) if isinstance(data, str) else data
            except Exception as e:
                logger.debug(f"[{self.strategy_name}] Redis 加载失败: {e}")

        # 2. JSON 文件兜底
        if state is None and os.path.exists(self._json_path):
            try:
                with open(self._json_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            except Exception as e:
                logger.error(f"[{self.strategy_name}] JSON 状态加载失败: {e}")

        # 3. 检查时效性
        if state:
            meta = state.get('_meta', {})
            saved_ts = meta.get('timestamp', 0)
            if saved_ts and (time.time() - saved_ts) > max_age_seconds:
                logger.warning(f"[{self.strategy_name}] 状态已过期 (saved_at={meta.get('saved_at')})")
                return None

        return state

    async def clear_state(self):
        """清除状态（用于手动重置）"""
        try:
            if self._is_redis_available():
                await self.redis_cache.delete(f"{self._redis_prefix}:state")
        except Exception:
            pass
        try:
            if os.path.exists(self._json_path):
                os.remove(self._json_path)
        except Exception as e:
            logger.error(f"[{self.strategy_name}] 清除状态失败: {e}")


class PersistentStrategy(EnterpriseStrategyMixin):
    """策略基类 mixin：提供状态持久化能力和策略协同支持。

    同时继承 EnterpriseStrategyMixin，为所有策略提供统一的企业级能力
    （异常分级、指标埋点、参数校验、告警通知）。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state_persistence: Optional[StatePersistence] = None
        self._state_save_interval = 60  # 默认每 60 秒自动保存一次
        self._last_state_save = 0.0
        self._state_persistence_enabled = True
        self._coordinator = None  # 策略协同协调器（由scheduler注入）

    def set_coordinator(self, coordinator):
        """注入StrategyCoordinator实例，用于策略间协同"""
        self._coordinator = coordinator

    def init_state_persistence(self, strategy_name: str, redis_cache=None):
        """初始化状态持久化（在策略 __init__ 或 start() 中调用）"""
        self._state_persistence = StatePersistence(strategy_name, redis_cache)
        logger.info(f"[{strategy_name}] 状态持久化已启用 (redis={redis_cache is not None})")

    def collect_persistent_state(self) -> Dict[str, Any]:
        """子类重写：收集需要持久化的状态。"""
        return {}

    def restore_persistent_state(self, state: Dict[str, Any]):
        """子类重写：从持久化数据恢复状态。"""
        pass

    async def save_state_async(self):
        """异步保存状态"""
        if not self._state_persistence or not self._state_persistence_enabled:
            return
        try:
            state = self.collect_persistent_state()
            await self._state_persistence.save_state(state)
            self._last_state_save = time.time()
        except Exception as e:
            logger.error(f"状态保存失败: {e}")

    async def load_state_async(self, max_age_seconds: int = 86400) -> bool:
        """异步加载状态，返回是否成功恢复"""
        if not self._state_persistence:
            return False
        try:
            state = await self._state_persistence.load_state(max_age_seconds=max_age_seconds)
            if state:
                # 移除元数据后传给子类
                state.pop('_meta', None)
                self.restore_persistent_state(state)
                logger.info(f"状态恢复成功 (字段数={len(state)})")
                return True
            logger.info("无可恢复的状态，使用默认状态")
            return False
        except Exception as e:
            logger.error(f"状态恢复失败: {e}")
            return False

    async def periodic_save_loop(self):
        """周期性保存状态的协程（在策略 start() 中作为 task 启动）"""
        while self._state_persistence_enabled:
            try:
                await asyncio.sleep(self._state_save_interval)
                await self.save_state_async()
            except asyncio.CancelledError:
                # 退出前最后保存一次
                await self.save_state_async()
                break
            except Exception as e:
                logger.error(f"周期性状态保存异常: {e}")
                await asyncio.sleep(10)


def safe_float(value: Any, default: float = 0.0) -> float:
    """安全的 float 转换，处理 None/NaN/Inf/字符串"""
    if value is None:
        return default
    try:
        v = float(value)
        if v != v or v in (float('inf'), float('-inf')):  # NaN or Inf
            return default
        return v
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """安全的 int 转换"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default
