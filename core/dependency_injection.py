"""
依赖注入容器 - Dependency Injection Container

功能：
1. 组件生命周期管理（单例、瞬态）
2. 依赖自动解析
3. 延迟初始化
4. 配置注入
"""

import asyncio
import re
from functools import wraps
from typing import Dict, Any, Type, Callable, Optional, List
from loguru import logger
from enum import Enum


class Lifecycle(Enum):
    """组件生命周期"""
    SINGLETON = "singleton"  # 单例：整个应用生命周期内只创建一次
    TRANSIENT = "transient"  # 瞬态：每次请求都创建新实例


class ServiceDescriptor:
    """服务描述符"""
    
    def __init__(
        self,
        service_type: Type,
        implementation: Any,
        lifecycle: Lifecycle,
        factory: Optional[Callable] = None
    ):
        self.service_type = service_type
        self.implementation = implementation
        self.lifecycle = lifecycle
        self.factory = factory
        self.instance = None


class DependencyContainer:
    """依赖注入容器"""
    
    def __init__(self):
        self._services: Dict[Type, ServiceDescriptor] = {}
        self._instances: Dict[Type, Any] = {}
        self._initialization_order: List[Type] = []
        
        logger.info("DependencyContainer initialized")
    
    def register_singleton(self, service_type: Type, implementation: Any = None, factory: Callable = None):
        """注册单例服务"""
        descriptor = ServiceDescriptor(service_type, implementation, Lifecycle.SINGLETON, factory)
        self._services[service_type] = descriptor
        self._initialization_order.append(service_type)
        logger.debug(f"Registered singleton: {service_type.__name__}")
        return self
    
    def register_transient(self, service_type: Type, implementation: Any = None, factory: Callable = None):
        """注册瞬态服务"""
        descriptor = ServiceDescriptor(service_type, implementation, Lifecycle.TRANSIENT, factory)
        self._services[service_type] = descriptor
        logger.debug(f"Registered transient: {service_type.__name__}")
        return self
    
    def register_instance(self, service_type: Type, instance: Any):
        """注册已存在的实例"""
        self._instances[service_type] = instance
        descriptor = ServiceDescriptor(service_type, instance, Lifecycle.SINGLETON)
        self._services[service_type] = descriptor
        logger.debug(f"Registered instance: {service_type.__name__}")
        return self
    
    def get(self, service_type: Type) -> Any:
        """获取服务实例"""
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")
        
        descriptor = self._services[service_type]
        
        # 单例模式：返回已存在的实例
        if descriptor.lifecycle == Lifecycle.SINGLETON:
            if service_type in self._instances:
                return self._instances[service_type]
            
            # 创建新实例
            instance = self._create_instance(descriptor)
            self._instances[service_type] = instance
            return instance
        
        # 瞬态模式：每次创建新实例
        return self._create_instance(descriptor)
    
    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """创建服务实例"""
        try:
            if descriptor.factory:
                return descriptor.factory()
            elif descriptor.implementation:
                if isinstance(descriptor.implementation, type):
                    return descriptor.implementation()
                else:
                    return descriptor.implementation
            else:
                raise ValueError(f"No implementation or factory for {descriptor.service_type.__name__}")
        except Exception as e:
            logger.error(f"Failed to create instance for {descriptor.service_type.__name__}: {e}")
            raise
    
    async def initialize_all(self):
        """初始化所有单例服务"""
        for service_type in self._initialization_order:
            descriptor = self._services[service_type]
            if descriptor.lifecycle == Lifecycle.SINGLETON and service_type not in self._instances:
                try:
                    instance = self._create_instance(descriptor)
                    self._instances[service_type] = instance
                    
                    # 如果有异步start方法，调用它
                    if hasattr(instance, 'start') and asyncio.iscoroutinefunction(instance.start):
                        await instance.start()
                        logger.info(f"Initialized and started: {service_type.__name__}")
                    else:
                        logger.info(f"Initialized: {service_type.__name__}")
                except Exception as e:
                    logger.error(f"Failed to initialize {service_type.__name__}: {e}")
                    raise
    
    async def shutdown_all(self):
        """关闭所有服务"""
        # 反向关闭
        for service_type in reversed(self._initialization_order):
            if service_type in self._instances:
                instance = self._instances[service_type]
                try:
                    if hasattr(instance, 'shutdown'):
                        if asyncio.iscoroutinefunction(instance.shutdown):
                            await instance.shutdown()
                        else:
                            instance.shutdown()
                    elif hasattr(instance, 'close'):
                        if asyncio.iscoroutinefunction(instance.close):
                            await instance.close()
                        else:
                            instance.close()
                    logger.info(f"Shutdown: {service_type.__name__}")
                except Exception as e:
                    logger.error(f"Failed to shutdown {service_type.__name__}: {e}")
        
        self._instances.clear()
    
    def get_all_instances(self) -> Dict[Type, Any]:
        """获取所有已初始化的实例"""
        return dict(self._instances)


class ServiceLocator:
    """服务定位器（简化访问）"""
    
    _container: Optional[DependencyContainer] = None
    
    @classmethod
    def set_container(cls, container: DependencyContainer):
        """设置容器"""
        cls._container = container
    
    @classmethod
    def get(cls, service_type: Type) -> Any:
        """获取服务"""
        if cls._container is None:
            raise ValueError("Container not initialized")
        return cls._container.get(service_type)
    
    @classmethod
    def get_container(cls) -> DependencyContainer:
        """获取容器"""
        if cls._container is None:
            raise ValueError("Container not initialized")
        return cls._container


def _to_snake_case(name: str) -> str:
    """PascalCase/CamelCase 转 snake_case（用于生成注入属性名）。"""
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def inject(service_type: Type):
    """依赖注入装饰器（用于属性注入）

    将 service_type 对应的服务实例注入到 self.<snake_case_name> 属性，
    供被装饰方法使用，避免手动调用 ServiceLocator.get。
    """
    attr_name = _to_snake_case(service_type.__name__)
    
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not hasattr(self, attr_name):
                setattr(self, attr_name, ServiceLocator.get(service_type))
            return func(self, *args, **kwargs)
        return wrapper
    return decorator


# 示例：如何使用依赖注入重构TradingScheduler
"""
# 在scheduler.py中使用依赖注入

from core.dependency_injection import DependencyContainer, Lifecycle, ServiceLocator

class TradingScheduler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.container = DependencyContainer()
        
        # 注册核心服务
        self.container.register_instance(Dict, config)  # 配置
        
        self.container.register_singleton(OKXClient, OKXClient)
        self.container.register_singleton(RedisCache, RedisCache)
        self.container.register_singleton(SQLiteStorage, SQLiteStorage)
        self.container.register_singleton(TradeJournal, TradeJournal)
        self.container.register_singleton(GlobalRisk, GlobalRisk)
        self.container.register_singleton(OrderExecutor, OrderExecutor)
        self.container.register_singleton(StopLossManager, StopLossManager)
        self.container.register_singleton(ConditionalOrderManager, ConditionalOrderManager)
        
        # 注册策略
        self.container.register_singleton(GridStrategy, GridStrategy)
        self.container.register_singleton(TrendStrategy, TrendStrategy)
        self.container.register_singleton(ScalpingStrategy, ScalpingStrategy)
        
        # 设置服务定位器
        ServiceLocator.set_container(self.container)
    
    async def start(self):
        # 初始化所有服务
        await self.container.initialize_all()
        
        # 获取需要的实例
        self.okx_client = self.container.get(OKXClient)
        self.order_executor = self.container.get(OrderExecutor)
        # ...
    
    async def shutdown(self):
        await self.container.shutdown_all()
"""