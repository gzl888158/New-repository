"""
配置管理器
支持配置热更新、配置验证、配置变更通知
"""
import os
import json
import yaml
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger
import hashlib
import re


@dataclass
class ConfigChange:
    """配置变更记录"""
    key: str
    old_value: Any
    new_value: Any
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "unknown"


@dataclass
class ConfigValidationResult:
    """配置验证结果"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, key: str, message: str):
        self.errors.append(f"[{key}] {message}")
        self.valid = False
    
    def add_warning(self, key: str, message: str):
        self.warnings.append(f"[{key}] {message}")


class ConfigValidator:
    """配置验证器"""
    
    VALIDATORS = {
        "trading.total_capital": lambda v: (True, None) if v >= 10 else (False, "total_capital must be >= 10"),
        "trading.max_drawdown": lambda v: (True, None) if 0 < v <= 0.5 else (False, "max_drawdown must be in (0, 0.5]"),
        "trading.daily_max_loss": lambda v: (True, None) if 0 < v <= 0.1 else (False, "daily_max_loss must be in (0, 0.1]"),
        "trading.risk_per_trade": lambda v: (True, None) if 0.005 <= v <= 0.05 else (False, "risk_per_trade must be in [0.005, 0.05]"),
        "trading.max_concurrent_positions": lambda v: (True, None) if 1 <= v <= 15 else (False, "max_concurrent_positions must be in [1, 15]"),
        "trading.max_total_leverage": lambda v: (True, None) if 1 <= v <= 20 else (False, "max_total_leverage must be in [1, 20]"),
        "trading.taker_fee_rate": lambda v: (True, None) if 0 <= v <= 0.01 else (False, "taker_fee_rate must be in [0, 0.01]"),
        "monitoring.api_latency_warning_ms": lambda v: (True, None) if v > 0 else (False, "api_latency_warning_ms must be > 0"),
        "monitoring.api_latency_critical_ms": lambda v: (True, None) if v > 0 else (False, "api_latency_critical_ms must be > 0"),
        "hardware.max_cpu_usage": lambda v: (True, None) if 50 <= v <= 100 else (False, "max_cpu_usage must be in [50, 100]"),
        "hardware.max_memory_usage_mb": lambda v: (True, None) if v >= 1024 else (False, "max_memory_usage_mb must be >= 1024"),
        "redis.port": lambda v: (True, None) if 1 <= v <= 65535 else (False, "redis port must be in [1, 65535]"),
    }
    
    def __init__(self):
        self._custom_validators: Dict[str, Callable] = {}
    
    def register_validator(self, key: str, validator: Callable[[Any], tuple]):
        """注册自定义验证器"""
        self._custom_validators[key] = validator
    
    def validate(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """验证配置"""
        result = ConfigValidationResult(valid=True)
        
        self._validate_recursive(config, "", result)
        
        for key, validator in self._custom_validators.items():
            value = self._get_nested_value(config, key)
            if value is not None:
                try:
                    valid, msg = validator(value)
                    if not valid:
                        result.add_error(key, msg)
                except Exception as e:
                    result.add_error(key, f"Validation error: {e}")
        
        self._validate_cross_fields(config, result)
        
        return result
    
    def _validate_recursive(self, config: Dict[str, Any], prefix: str, result: ConfigValidationResult):
        """递归验证配置"""
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._validate_recursive(value, full_key, result)
            else:
                if full_key in self.VALIDATORS:
                    try:
                        valid, msg = self.VALIDATORS[full_key](value)
                        if not valid:
                            result.add_error(full_key, msg)
                    except Exception as e:
                        result.add_error(full_key, f"Validation error: {e}")
                
                if key.endswith("_ratio") or key.endswith("_allocation"):
                    if not (0 <= value <= 1):
                        result.add_warning(full_key, f"Value {value} is outside typical range [0, 1]")
    
    def _validate_cross_fields(self, config: Dict[str, Any], result: ConfigValidationResult):
        """验证跨字段约束"""
        try:
            allocations = config.get("trading", {})
            total_allocation = sum([
                allocations.get("grid_allocation", 0),
                allocations.get("spot_grid_allocation", 0),
                allocations.get("spot_martingale_allocation", 0),
                allocations.get("trend_allocation", 0),
                allocations.get("scalping_allocation", 0),
                allocations.get("arbitrage_allocation", 0),
            ])
            
            if total_allocation > 1.0:
                result.add_error("trading.allocations", f"Total allocation {total_allocation:.2%} exceeds 100%")
            elif total_allocation < 0.9:
                result.add_warning("trading.allocations", f"Total allocation {total_allocation:.2%} is below 90%, capital may be underutilized")
            
            latency_warning = config.get("monitoring", {}).get("api_latency_warning_ms", 1500)
            latency_critical = config.get("monitoring", {}).get("api_latency_critical_ms", 5000)
            if latency_warning >= latency_critical:
                result.add_error("monitoring.latency", "api_latency_warning_ms must be less than api_latency_critical_ms")
            
            warning_threshold = config.get("hardware", {}).get("memory_warning_threshold_mb", 6144)
            max_memory = config.get("hardware", {}).get("max_memory_usage_mb", 8192)
            if warning_threshold >= max_memory:
                result.add_warning("hardware.memory", "memory_warning_threshold_mb should be less than max_memory_usage_mb")
                
        except Exception as e:
            result.add_warning("cross_field", f"Cross-field validation error: {e}")
    
    def _get_nested_value(self, config: Dict[str, Any], key: str) -> Any:
        """获取嵌套配置值"""
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value


class ConfigManager:
    """配置管理器"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        config_path: str = "./config.yaml",
        env_prefix: str = "OKX_",
        watch_interval: float = 5.0,
    ):
        if not hasattr(self, '_initialized'):
            self.config_path = config_path
            self.env_prefix = env_prefix
            self.watch_interval = watch_interval
            
            self._config: Dict[str, Any] = {}
            self._original_config: Dict[str, Any] = {}
            self._validator = ConfigValidator()
            self._change_history: List[ConfigChange] = []
            self._listeners: Dict[str, List[Callable]] = {}
            self._file_hash: Optional[str] = None
            self._last_modified: Optional[float] = None
            self._watch_task: Optional[threading.Thread] = None
            self._running = False
            self._config_lock = threading.RLock()
            
            self._initialized = True
            
            self.load()
    
    def load(self) -> bool:
        """加载配置文件"""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                raw_content = f.read()
            
            content = self._substitute_env_vars(raw_content)
            
            config = yaml.safe_load(content)
            
            if config is None:
                config = {}
            
            result = self._validator.validate(config)
            if not result.valid:
                logger.error(f"Config validation failed: {result.errors}")
                return False
            
            if result.warnings:
                for warning in result.warnings:
                    logger.warning(f"Config validation warning: {warning}")
            
            with self._config_lock:
                self._original_config = self._deep_copy(config)
                self._config = self._deep_copy(config)
                self._file_hash = self._compute_hash(raw_content)
                self._last_modified = os.path.getmtime(self.config_path)
            
            logger.info(f"Config loaded from {self.config_path}")
            return True
            
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return False
        except yaml.YAMLError as e:
            logger.error(f"YAML parse error: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return False
    
    def _substitute_env_vars(self, content: str) -> str:
        """替换环境变量"""
        pattern = r'\$\{([^}]+)\}'
        
        def replace(match):
            var_name = match.group(1)
            
            if self.env_prefix:
                prefixed_name = f"{self.env_prefix}{var_name}"
                value = os.environ.get(prefixed_name) or os.environ.get(var_name, "")
            else:
                value = os.environ.get(var_name, "")
            
            return value
        
        return re.sub(pattern, replace, content)
    
    def _compute_hash(self, content: str) -> str:
        """计算文件哈希"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _deep_copy(self, obj: Any) -> Any:
        """深拷贝"""
        if isinstance(obj, dict):
            return {k: self._deep_copy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._deep_copy(item) for item in obj]
        else:
            return obj
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        with self._config_lock:
            keys = key.split(".")
            value = self._config
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            
            return value
    
    def set(self, key: str, value: Any, notify: bool = True) -> bool:
        """设置配置值"""
        with self._config_lock:
            keys = key.split(".")
            target = self._config
            
            for k in keys[:-1]:
                if k not in target:
                    target[k] = {}
                target = target[k]
            
            old_value = target.get(keys[-1])
            
            if old_value != value:
                target[keys[-1]] = value
                
                change = ConfigChange(
                    key=key,
                    old_value=old_value,
                    new_value=value,
                    source="manual"
                )
                self._change_history.append(change)
                
                if notify:
                    self._notify_listeners(key, old_value, value)
                
                logger.info(f"Config updated: {key} = {value}")
                return True
            
            return False
    
    def validate_current(self) -> ConfigValidationResult:
        """验证当前配置"""
        with self._config_lock:
            return self._validator.validate(self._config)
    
    def reset(self, key: Optional[str] = None):
        """重置配置到原始值"""
        with self._config_lock:
            if key is None:
                self._config = self._deep_copy(self._original_config)
                logger.info("Config reset to original")
            else:
                keys = key.split(".")
                source = self._original_config
                target = self._config
                
                for k in keys[:-1]:
                    if k in source:
                        source = source[k]
                        target = target[k]
                    else:
                        return
                
                if keys[-1] in source:
                    target[keys[-1]] = self._deep_copy(source[keys[-1]])
                    logger.info(f"Config key '{key}' reset to original")
    
    def save(self, path: Optional[str] = None) -> bool:
        """保存配置到文件"""
        save_path = path or self.config_path
        
        try:
            with self._config_lock:
                content = yaml.dump(self._config, default_flow_style=False, allow_unicode=True)
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            logger.info(f"Config saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False
    
    def register_listener(self, key_pattern: str, callback: Callable[[ConfigChange], None]):
        """注册配置变更监听器"""
        if key_pattern not in self._listeners:
            self._listeners[key_pattern] = []
        self._listeners[key_pattern].append(callback)
    
    def _notify_listeners(self, key: str, old_value: Any, new_value: Any):
        """通知监听器"""
        change = ConfigChange(key=key, old_value=old_value, new_value=new_value)
        
        for pattern, callbacks in self._listeners.items():
            if self._match_pattern(key, pattern):
                for callback in callbacks:
                    try:
                        callback(change)
                    except Exception as e:
                        logger.error(f"Config listener error: {e}")
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """匹配键模式"""
        if pattern == "*":
            return True
        
        regex_pattern = pattern.replace(".", r"\.").replace("*", ".*")
        return bool(re.match(f"^{regex_pattern}$", key))
    
    def start_watching(self):
        """开始监视配置文件变化"""
        if self._running:
            return
        
        self._running = True
        self._watch_task = threading.Thread(target=self._watch_loop, daemon=True)
        self._watch_task.start()
        logger.info("Config file watching started")
    
    def stop_watching(self):
        """停止监视配置文件"""
        self._running = False
        if self._watch_task:
            self._watch_task.join(timeout=2)
        logger.info("Config file watching stopped")
    
    def _watch_loop(self):
        """监视循环"""
        while self._running:
            try:
                if os.path.exists(self.config_path):
                    current_modified = os.path.getmtime(self.config_path)
                    
                    if self._last_modified is None or current_modified > self._last_modified:
                        with open(self.config_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        
                        current_hash = self._compute_hash(content)
                        
                        if current_hash != self._file_hash:
                            logger.info("Config file changed, reloading...")
                            self.load()
                            self._notify_reload()
                
                time.sleep(self.watch_interval)
                
            except Exception as e:
                logger.error(f"Config watch error: {e}")
                time.sleep(self.watch_interval * 2)
    
    def _notify_reload(self):
        """通知配置重新加载"""
        change = ConfigChange(
            key="__reload__",
            old_value=None,
            new_value=None,
            source="file_reload"
        )
        
        for pattern, callbacks in self._listeners.items():
            if pattern == "*" or pattern == "__reload__":
                for callback in callbacks:
                    try:
                        callback(change)
                    except Exception as e:
                        logger.error(f"Config reload listener error: {e}")
    
    def get_change_history(self, limit: int = 100) -> List[ConfigChange]:
        """获取配置变更历史"""
        return self._change_history[-limit:]
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        with self._config_lock:
            return self._deep_copy(self._config)
    
    def export_json(self) -> str:
        """导出配置为JSON"""
        with self._config_lock:
            return json.dumps(self._config, indent=2, ensure_ascii=False, default=str)
    
    def import_json(self, json_str: str) -> bool:
        """从JSON导入配置"""
        try:
            config = json.loads(json_str)
            
            result = self._validator.validate(config)
            if not result.valid:
                logger.error(f"Import validation failed: {result.errors}")
                return False
            
            with self._config_lock:
                self._config = config
            
            logger.info("Config imported from JSON")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import config from JSON: {e}")
            return False