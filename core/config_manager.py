"""
配置管理器 - Configuration Manager

功能：
1. 移除硬编码配置
2. 配置验证
3. 热更新支持
4. 配置版本管理
"""

import yaml
import json
import os
import copy
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    Observer = None
    FileSystemEventHandler = None


class ConfigManager:
    """配置管理器（生产级 v2.0）"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self._config_version = 0
        self._last_modified = None
        self._callbacks: List[callable] = []
        
        # ── 版本历史 (v2.0) ──
        self._version_history: List[Dict[str, Any]] = []
        self._max_history = 50
        
        # ── 键级回调 (v2.0) ──
        self._key_callbacks: Dict[str, List[callable]] = {}
        
        # 加载配置
        self.load_config()
        
        logger.info(f"ConfigManager initialized: {config_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件（带版本历史记录）"""
        try:
            old_config = self.config.copy() if self.config else {}
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # 从环境变量覆盖敏感信息
            self._override_from_env()
            
            # 验证配置
            self._validate_config()
            
            self._config_version += 1
            self._last_modified = datetime.now()
            
            # 记录版本历史和差异（含完整快照，供一键回滚）
            if old_config:
                diff = self._compute_diff(old_config, self.config)
                self._version_history.append({
                    "version": self._config_version,
                    "timestamp": self._last_modified.isoformat(),
                    "diff": diff,
                    "snapshot": copy.deepcopy(self.config),
                })
                if len(self._version_history) > self._max_history:
                    self._version_history = self._version_history[-self._max_history:]
            
            logger.info(f"Config loaded successfully (version {self._config_version})")
            return self.config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise
    
    def _override_from_env(self):
        """从环境变量覆盖配置"""
        # OKX配置
        if "OKX_API_KEY" in os.environ:
            self.config.setdefault("okx", {})["api_key"] = os.getenv("OKX_API_KEY")
        if "OKX_SECRET_KEY" in os.environ:
            self.config.setdefault("okx", {})["secret_key"] = os.getenv("OKX_SECRET_KEY")
        if "OKX_PASSPHRASE" in os.environ:
            self.config.setdefault("okx", {})["passphrase"] = os.getenv("OKX_PASSPHRASE")
        if "OKX_PROXY" in os.environ:
            self.config.setdefault("okx", {})["proxy"] = os.getenv("OKX_PROXY")
        
        # Redis配置
        if "REDIS_PASSWORD" in os.environ:
            self.config.setdefault("redis", {})["password"] = os.getenv("REDIS_PASSWORD")
        
        # Telegram配置
        if "TELEGRAM_BOT_TOKEN" in os.environ:
            self.config.setdefault("telegram", {})["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN")
        if "TELEGRAM_CHAT_ID" in os.environ:
            self.config.setdefault("telegram", {})["chat_id"] = os.getenv("TELEGRAM_CHAT_ID")
        
        # 告警Webhook
        if "ALERT_WEBHOOK" in os.environ:
            self.config.setdefault("monitoring", {})["alert_webhook"] = os.getenv("ALERT_WEBHOOK")
    
    def _validate_config(self):
        """验证配置"""
        errors = []
        
        # 验证交易配置
        trading = self.config.get("trading", {})
        
        # 检查资金分配总和
        allocation_fields = [
            "scalping_allocation", "trend_allocation", "grid_allocation",
            "arbitrage_allocation", "spot_grid_allocation", "spot_martingale_allocation"
        ]
        total_allocation = sum(trading.get(f, 0) for f in allocation_fields)
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Allocation sum {total_allocation:.4f} != 1.0")
        
        # 检查资本比例总和
        trading_ratio = trading.get("trading_capital_ratio", 0)
        risk_reserve = trading.get("risk_reserve_ratio", 0)
        profit_reserve = trading.get("profit_reserve_ratio", 0)
        capital_sum = trading_ratio + risk_reserve + profit_reserve
        if abs(capital_sum - 1.0) > 0.01:
            errors.append(f"Capital ratio sum {capital_sum:.4f} != 1.0")
        
        # 检查必要字段
        if "total_capital" not in trading:
            errors.append("Missing required field: trading.total_capital")
        
        if errors:
            logger.warning(f"Config validation warnings: {errors}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值（支持点号分隔的路径）"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.info(f"Config updated: {key} = {value}")
    
    def _write_file(self):
        """仅将当前内存配置写回文件（不改变版本号，供回滚/落盘复用）。"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

    def save_config(self):
        """保存配置到文件（并记录版本快照）"""
        try:
            old_config = copy.deepcopy(self.config)
            self._write_file()

            self._config_version += 1
            self._last_modified = datetime.now()

            # 保存时也记录版本历史与完整快照，保证「任一变更可一键回滚」
            diff = self._compute_diff(old_config, self.config) if old_config else []
            self._version_history.append({
                "version": self._config_version,
                "timestamp": self._last_modified.isoformat(),
                "diff": diff,
                "snapshot": copy.deepcopy(self.config),
            })
            if len(self._version_history) > self._max_history:
                self._version_history = self._version_history[-self._max_history:]

            logger.info(f"Config saved (version {self._config_version})")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
    
    def register_callback(self, callback: callable):
        """注册配置变更回调"""
        self._callbacks.append(callback)
    
    def notify_callbacks(self):
        """通知所有回调（全局 + 键级）"""
        for callback in self._callbacks:
            try:
                callback(self.config)
            except Exception as e:
                logger.error(f"Config callback error: {e}")
        
        # 通知键级回调
        self._notify_key_callbacks()
    
    def register_key_callback(self, key_prefix: str, callback: callable) -> None:
        """
        注册键级回调 —— 仅当配置中指定键变化时触发
        
        Args:
            key_prefix: 配置键前缀（如 "strategies.grid"）
            callback: 回调函数，接收 (new_value, full_config)
        """
        if key_prefix not in self._key_callbacks:
            self._key_callbacks[key_prefix] = []
        self._key_callbacks[key_prefix].append(callback)
        logger.debug(f"Registered key callback for '{key_prefix}'")
    
    def unregister_key_callback(self, key_prefix: str, callback: callable) -> None:
        """注销键级回调"""
        if key_prefix in self._key_callbacks:
            self._key_callbacks[key_prefix] = [
                cb for cb in self._key_callbacks[key_prefix] if cb != callback
            ]
    
    def _notify_key_callbacks(self) -> None:
        """触发键级回调"""
        for key_prefix, callbacks in self._key_callbacks.items():
            value = self.get(key_prefix)
            if value is not None:
                for callback in callbacks:
                    try:
                        callback(value, self.config)
                    except Exception as e:
                        logger.error(f"Key callback error for '{key_prefix}': {e}")
    
    def _compute_diff(self, old: Dict[str, Any], new: Dict[str, Any], 
                      prefix: str = "") -> List[Dict[str, Any]]:
        """计算配置差异（递归）"""
        diffs = []
        all_keys = set(old.keys()) | set(new.keys())
        
        for key in sorted(all_keys):
            full_key = f"{prefix}.{key}" if prefix else key
            old_val = old.get(key)
            new_val = new.get(key)
            
            if key not in old:
                diffs.append({"key": full_key, "type": "added", "new": new_val})
            elif key not in new:
                diffs.append({"key": full_key, "type": "removed", "old": old_val})
            elif isinstance(old_val, dict) and isinstance(new_val, dict):
                diffs.extend(self._compute_diff(old_val, new_val, full_key))
            elif old_val != new_val:
                diffs.append({
                    "key": full_key, "type": "changed",
                    "old": old_val, "new": new_val
                })
        
        return diffs
    
    def get_version_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """获取配置版本历史"""
        return self._version_history[-limit:]
    
    def get_last_diff(self) -> Optional[Dict[str, Any]]:
        """获取最后一次配置变更差异"""
        if self._version_history:
            return self._version_history[-1]
        return None
    
    def rollback(self, target_version: int = None) -> bool:
        """
        回滚到指定版本（需要版本历史中有对应的配置快照）
        
        Args:
            target_version: 目标版本号，默认为上一版本
        
        Returns:
            是否成功
        """
        if not self._version_history:
            logger.warning("No version history available for rollback")
            return False
        
        if target_version is None:
            target_version = self._config_version - 1
        
        if target_version < 1:
            logger.warning(f"Invalid target version: {target_version}")
            return False
        
        # 优先使用完整快照恢复（历史按时间升序，取版本号 <= 目标的最新一条）
        target_entry = None
        for entry in reversed(self._version_history):
            if entry["version"] <= target_version:
                target_entry = entry
                break

        if target_entry is None:
            logger.warning(f"No snapshot found for version <= {target_version}")
            return False

        snapshot = target_entry.get("snapshot")
        if snapshot is not None:
            self.config = copy.deepcopy(snapshot)
        else:
            # 兼容旧历史（仅 diff 无快照）：反向应用 diff
            for entry in reversed(self._version_history):
                if entry["version"] <= target_version:
                    break
                for change in reversed(entry.get("diff", [])):
                    if change["type"] == "changed":
                        self.set(change["key"], change["old"])
                    elif change["type"] == "added":
                        self._delete_key(change["key"])
                    elif change["type"] == "removed":
                        self.set(change["key"], change["old"])
        
        self._config_version = target_version
        # 写回文件，使回滚真正落盘生效（重启后仍保持回滚后的配置）
        try:
            self._write_file()
        except Exception as e:
            logger.error(f"Failed to write rolled-back config to file: {e}")
        logger.warning(f"Config rolled back to version {target_version}")
        self.notify_callbacks()
        return True
    
    def _delete_key(self, key: str) -> None:
        """删除配置键"""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config:
                return
            config = config[k]
        if keys[-1] in config:
            del config[keys[-1]]
    
    def get_version_info(self) -> Dict[str, Any]:
        """获取配置版本信息"""
        return {
            "version": self._config_version,
            "last_modified": self._last_modified.isoformat() if self._last_modified else None,
            "path": self.config_path
        }


class ConfigWatcher:
    """配置文件监视器（支持热更新）"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.observer = Observer() if Observer else None
        self._running = False
        
    def start(self):
        """启动监视器"""
        if self._running:
            return
        
        if self.observer is None:
            logger.warning("ConfigWatcher requires watchdog package, hot reload disabled")
            return
        
        class ConfigHandler(FileSystemEventHandler):
            def __init__(self, manager):
                self.manager = manager
            
            def on_modified(self, event):
                if event.src_path.endswith('config.yaml'):
                    logger.info("Config file modified, reloading...")
                    self.manager.load_config()
                    self.manager.notify_callbacks()
        
        handler = ConfigHandler(self.config_manager)
        self.observer.schedule(handler, path='.', recursive=False)
        self.observer.start()
        self._running = True
        
        logger.info("Config watcher started")
    
    def stop(self):
        """停止监视器"""
        if self._running and self.observer:
            self.observer.stop()
            self.observer.join()
            self._running = False
            logger.info("Config watcher stopped")


# 配置硬编码修复
def fix_hardcoded_config():
    """
    修复硬编码配置
    
    将硬编码值移到config.yaml或环境变量
    """
    fixes = []
    
    # 1. 禁止交易时段（应从配置读取）
    forbidden_hours = """
    # 在config.yaml中添加：
    trading:
      forbidden_hours:
        - start: 23
          end: 0
          reason: "Daily settlement"
        - start: 16
          end: 17
          reason: "Funding rate settlement"
    """
    fixes.append({
        "file": "execution/order_executor.py",
        "line": 76,
        "issue": "硬编码禁止交易时段",
        "fix": forbidden_hours
    })
    
    # 2. 阶梯式风控阈值（应从配置读取）
    risk_tiers = """
    # 在config.yaml中添加：
    risk:
      tier_thresholds:
        tier1:
          drawdown_pct: 0.10
          position_reduction: 0.2
        tier2:
          drawdown_pct: 0.15
          position_reduction: 0.4
        tier3:
          drawdown_pct: 0.20
          position_reduction: 0.6
    """
    fixes.append({
        "file": "risk/global_risk.py",
        "line": 55,
        "issue": "硬编码阶梯式风控阈值",
        "fix": risk_tiers
    })
    
    # 3. API密钥（应从环境变量读取）
    api_keys = """
    # 在.env文件中设置：
    OKX_API_KEY=your_api_key
    OKX_SECRET_KEY=your_secret_key
    OKX_PASSPHRASE=your_passphrase
    
    # 或在启动时设置环境变量：
    export OKX_API_KEY=xxx
    export OKX_SECRET_KEY=xxx
    export OKX_PASSPHRASE=xxx
    """
    fixes.append({
        "file": "configs/settings.py",
        "line": 26,
        "issue": "硬编码API密钥",
        "fix": api_keys
    })
    
    return fixes


# 配置模板生成
def generate_config_template() -> str:
    """生成完整的配置模板"""
    template = """
# OKX量化交易系统配置模板

# 系统配置
system:
  name: OKX_Quant_Trading_System
  version: 1.0.0
  timezone: Asia/Shanghai
  log_level: INFO

# OKX API配置（从环境变量读取）
okx:
  api_key: ${OKX_API_KEY}
  secret_key: ${OKX_SECRET_KEY}
  passphrase: ${OKX_PASSPHRASE}
  rest_url: https://www.okx.com
  websocket_url: wss://ws.okx.com:8443/ws/v5/public
  websocket_private_url: wss://ws.okx.com:8443/ws/v5/private
  proxy: ${OKX_PROXY:http://127.0.0.1:7897}
  is_testnet: false

# Redis配置
redis:
  host: localhost
  port: 6379
  db: 0
  password: ${REDIS_PASSWORD}

# SQLite配置
sqlite:
  db_path: ./data/trading.db

# 交易配置
trading:
  # 总资本（USDT）
  total_capital: 500
  
  # 资金分配
  trading_capital_ratio: 0.95
  risk_reserve_ratio: 0.05
  profit_reserve_ratio: 0.0
  
  # 策略分配（总和必须为1.0）
  scalping_allocation: 0.45
  trend_allocation: 0.20
  grid_allocation: 0.20
  arbitrage_allocation: 0.15
  spot_grid_allocation: 0.0
  spot_martingale_allocation: 0.0
  
  # 风控参数
  risk_per_trade: 0.025
  max_total_leverage: 20
  max_drawdown: 0.25
  daily_max_loss: 0.04
  hourly_max_loss: 0.025
  
  # 止盈止损参数
  max_stop_loss_pct: 0.05
  trailing_activation_threshold: 0.015
  trailing_min_distance: 0.005
  
  # 禁止交易时段
  forbidden_hours:
    - start: 23
      end: 0
      reason: "Daily settlement"
    - start: 16
      end: 17
      reason: "Funding rate settlement"

# 风控配置
risk:
  # 阶梯式风控
  tier_thresholds:
    tier1:
      drawdown_pct: 0.10
      position_reduction: 0.2
      description: "Tier 1: Reduce position by 20%"
    tier2:
      drawdown_pct: 0.15
      position_reduction: 0.4
      description: "Tier 2: Reduce position by 40%"
    tier3:
      drawdown_pct: 0.20
      position_reduction: 0.6
      description: "Tier 3: Reduce position by 60%"
  
  # 熔断器配置
  circuit_breakers:
    api_timeout: 600
    btc_movement_threshold: 0.06
    btc_movement_window: 300

# 策略配置
strategies:
  scalping:
    enabled: true
    min_signal_quality: 0.10
    stop_loss: 0.003
    
  trend:
    enabled: true
    min_signal_quality: 0.20
    max_stop_loss_pct: 0.025
    
  grid:
    enabled: true
    min_signal_quality: 0.40
    stop_loss_pct: 0.035
    max_stop_loss_pct: 0.04

# 监控配置
monitoring:
  metrics_port: 8000
  alert_webhook: ${ALERT_WEBHOOK}
  api_latency_warning_ms: 1500
  api_latency_critical_ms: 5000

# Telegram配置
telegram:
  bot_token: ${TELEGRAM_BOT_TOKEN}
  chat_id: ${TELEGRAM_CHAT_ID}
"""
    return template


if __name__ == "__main__":
    # 生成配置模板
    template = generate_config_template()
    with open("config.template.yaml", 'w', encoding='utf-8') as f:
        f.write(template)
    
    print("Generated config.template.yaml")
    
    # 打印硬编码修复建议
    fixes = fix_hardcoded_config()
    for fix in fixes:
        print(f"\n{fix['file']}:{fix['line']} - {fix['issue']}")
        print(fix['fix'])