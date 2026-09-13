"""
API密钥安全管理 - Secure API Key Management

功能：
1. 密钥加密存储
2. 环境变量注入
3. 密钥轮换支持
4. 访问审计日志
"""

import os
import json
import base64
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from loguru import logger
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


# 环境变量编号密钥的最大数量（与 configs/settings.py 保持一致，避免多密钥池边界不一致）
MAX_API_KEYS = 20


class APIKeyManager:
    """API密钥管理器"""
    
    def __init__(self, config: Dict[str, Any], encryption_key: str = None):
        self.config = config
        
        # 加密密钥（从环境变量或参数获取）
        self._encryption_key = encryption_key or os.getenv("TRAE_ENCRYPTION_KEY")
        
        # 加密器
        self._fernet = None
        if self._encryption_key:
            self._init_encryption()
        
        # 密钥存储文件
        self._keys_file = "data/api_keys.enc"
        
        # 密钥访问审计日志
        self._audit_log_file = "data/api_key_audit.log"
        
        # 当前活跃密钥索引
        self._current_key_index = 0
        
        # 已失效密钥集合（失效自检后标记，轮换时自动跳过）
        self._failed_keys: set = set()
        
        logger.info("APIKeyManager initialized")
    
    def _init_encryption(self):
        """初始化加密器"""
        try:
            # 从密码派生密钥
            password = self._encryption_key.encode()
            salt = b'trae_okx_salt_2024'  # 固定盐值（生产环境应使用随机盐）
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password))
            self._fernet = Fernet(key)
            
            logger.info("Encryption initialized")
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            self._fernet = None
    
    def encrypt_value(self, value: str) -> str:
        """加密值"""
        if not self._fernet:
            logger.warning("Encryption not available, storing value in plain text")
            return value
        
        try:
            encrypted = self._fernet.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            logger.error(f"Failed to encrypt value: {e}")
            return value
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """解密值"""
        if not self._fernet:
            logger.warning("Encryption not available, returning value as-is")
            return encrypted_value
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted = self._fernet.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt value: {e}")
            return encrypted_value
    
    def load_keys_from_env(self) -> Dict[str, str]:
        """从环境变量加载密钥（推荐方式）"""
        keys = {}
        
        # 单密钥模式
        api_key = os.getenv("OKX_API_KEY")
        secret_key = os.getenv("OKX_SECRET_KEY")
        passphrase = os.getenv("OKX_PASSPHRASE")
        
        if api_key and secret_key and passphrase:
            keys["default"] = {
                "api_key": api_key,
                "secret_key": secret_key,
                "passphrase": passphrase
            }
            logger.info("Loaded single API key from environment")
        
        # 多密钥模式（用于密钥轮换）
        for i in range(1, MAX_API_KEYS + 1):  # 与 configs/settings.py 保持一致
            api_key = os.getenv(f"OKX_API_KEY_{i}")
            secret_key = os.getenv(f"OKX_SECRET_KEY_{i}")
            passphrase = os.getenv(f"OKX_PASSPHRASE_{i}")
            
            if api_key and secret_key and passphrase:
                keys[f"key_{i}"] = {
                    "api_key": api_key,
                    "secret_key": secret_key,
                    "passphrase": passphrase
                }
                logger.info(f"Loaded API key {i} from environment")
        
        return keys
    
    def save_keys_to_file(self, keys: Dict[str, Dict[str, str]]):
        """保存密钥到加密文件"""
        try:
            encrypted_keys = {}
            
            for key_name, key_data in keys.items():
                encrypted_keys[key_name] = {
                    "api_key": self.encrypt_value(key_data["api_key"]),
                    "secret_key": self.encrypt_value(key_data["secret_key"]),
                    "passphrase": self.encrypt_value(key_data["passphrase"])
                }
            
            # 写入文件
            import os
            os.makedirs(os.path.dirname(self._keys_file), exist_ok=True)
            
            with open(self._keys_file, 'w', encoding='utf-8') as f:
                json.dump(encrypted_keys, f, indent=2)
            
            logger.info(f"Saved {len(keys)} encrypted keys to {self._keys_file}")
        except Exception as e:
            logger.error(f"Failed to save keys: {e}")
    
    def load_keys_from_file(self) -> Dict[str, Dict[str, str]]:
        """从加密文件加载密钥"""
        try:
            if not os.path.exists(self._keys_file):
                logger.warning(f"Keys file not found: {self._keys_file}")
                return {}
            
            with open(self._keys_file, 'r', encoding='utf-8') as f:
                encrypted_keys = json.load(f)
            
            decrypted_keys = {}
            for key_name, key_data in encrypted_keys.items():
                decrypted_keys[key_name] = {
                    "api_key": self.decrypt_value(key_data["api_key"]),
                    "secret_key": self.decrypt_value(key_data["secret_key"]),
                    "passphrase": self.decrypt_value(key_data["passphrase"])
                }
            
            logger.info(f"Loaded {len(decrypted_keys)} keys from {self._keys_file}")
            return decrypted_keys
        except Exception as e:
            logger.error(f"Failed to load keys: {e}")
            return {}
    
    def get_current_key(self, keys: Dict[str, Dict[str, str]]) -> Dict[str, str]:
        """获取当前活跃密钥（跳过已失效密钥）"""
        if not keys:
            raise ValueError("No API keys available")
        
        healthy_names = self.get_healthy_key_names(keys)
        if not healthy_names:
            raise ValueError("No healthy API keys available")
        
        current_key_name = healthy_names[self._current_key_index % len(healthy_names)]
        
        # 记录访问审计
        self._log_key_access(current_key_name)
        
        return keys[current_key_name]
    
    def rotate_key(self, keys: Dict[str, Dict[str, str]] = None):
        """轮换密钥（切换到下一个健康密钥，跳过已失效密钥）。

        Args:
            keys: 密钥池。传入时按健康密钥池计算并返回新密钥名；不传则仅推进索引。

        Returns:
            新密钥名；若无健康密钥返回 None。
        """
        self._current_key_index += 1
        new_name = None
        if keys:
            healthy_names = self.get_healthy_key_names(keys)
            if not healthy_names:
                logger.error("No healthy API keys available for rotation")
                return None
            new_name = healthy_names[self._current_key_index % len(healthy_names)]
        suffix = f" ({new_name})" if new_name else ""
        logger.info(f"Rotated to key index {self._current_key_index}{suffix}")
        return new_name
    
    def get_healthy_key_names(self, keys: Dict[str, Dict[str, str]]) -> List[str]:
        """返回未标记失效的密钥名列表"""
        return [name for name in keys if name not in self._failed_keys]
    
    def mark_key_failed(self, key_name: str) -> None:
        """标记密钥失效，轮换时自动跳过"""
        self._failed_keys.add(key_name)
        logger.warning(f"API key marked as failed: {key_name}")
    
    def mark_key_healthy(self, key_name: str) -> None:
        """解除密钥失效标记"""
        self._failed_keys.discard(key_name)
    
    def validate_key_health(self, keys: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
        """多密钥池失效自检：检测占位符/长度异常，标记失效密钥并返回健康/失效清单。

        与模块级 check_key_health() 互补：本方法在密钥池轮换前调用，
        直接驱动 mark_key_failed，实现「失效自检 → 轮换跳过」闭环。
        """
        placeholder_markers = ("your_", "xxx", "example", "placeholder", "replace", "changeme")
        healthy: List[str] = []
        failed: Dict[str, List[str]] = {}

        for name, kd in keys.items():
            api_key = kd.get("api_key", "")
            secret_key = kd.get("secret_key", "")
            passphrase = kd.get("passphrase", "")
            issues: List[str] = []

            if not (api_key and secret_key and passphrase):
                issues.append("密钥三要素不完整")
            else:
                for field, val in (("api_key", api_key), ("secret_key", secret_key), ("passphrase", passphrase)):
                    low = val.lower()
                    if any(m in low for m in placeholder_markers):
                        issues.append(f"{field}:疑似占位符")
                    elif len(val) < 8:
                        issues.append(f"{field}:长度异常({len(val)})")

            if issues:
                failed[name] = issues
                self.mark_key_failed(name)
            else:
                healthy.append(name)
                self.mark_key_healthy(name)

        return {"total": len(keys), "healthy": healthy, "failed": failed}
    
    def _log_key_access(self, key_name: str):
        """记录密钥访问审计日志"""
        try:
            import os
            os.makedirs(os.path.dirname(self._audit_log_file), exist_ok=True)
            
            audit_entry = {
                "timestamp": datetime.now().isoformat(),
                "key_name": key_name,
                "action": "access"
            }
            
            with open(self._audit_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(audit_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to log key access: {e}")
    
    def validate_key_permissions(self) -> Dict[str, Any]:
        """验证密钥文件权限"""
        result = {
            "keys_file_exists": os.path.exists(self._keys_file),
            "keys_file_readable": False,
            "keys_file_writable": False,
            "encryption_enabled": self._fernet is not None,
            "env_keys_loaded": len(self.load_keys_from_env()) > 0
        }
        
        if result["keys_file_exists"]:
            try:
                with open(self._keys_file, 'r'):
                    result["keys_file_readable"] = True
            except Exception:
                pass
            
            try:
                with open(self._keys_file, 'a'):
                    result["keys_file_writable"] = True
            except Exception:
                pass
        
        return result


class SecureConfigLoader:
    """安全配置加载器"""
    
    @staticmethod
    def load_config_with_secrets(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        加载配置并注入密钥（优先从环境变量）
        """
        # 加载OKX配置
        okx_config = config.get("okx", {})
        
        # 从环境变量覆盖
        if "OKX_API_KEY" in os.environ:
            okx_config["api_key"] = os.getenv("OKX_API_KEY")
        if "OKX_SECRET_KEY" in os.environ:
            okx_config["secret_key"] = os.getenv("OKX_SECRET_KEY")
        if "OKX_PASSPHRASE" in os.environ:
            okx_config["passphrase"] = os.getenv("OKX_PASSPHRASE")
        if "OKX_REST_URL" in os.environ:
            okx_config["rest_url"] = os.getenv("OKX_REST_URL")
        if "OKX_PROXY" in os.environ:
            okx_config["proxy"] = os.getenv("OKX_PROXY")
        
        config["okx"] = okx_config
        
        # 加载Redis配置
        redis_config = config.get("redis", {})
        if "REDIS_PASSWORD" in os.environ:
            redis_config["password"] = os.getenv("REDIS_PASSWORD")
        if "REDIS_HOST" in os.environ:
            redis_config["host"] = os.getenv("REDIS_HOST")
        
        config["redis"] = redis_config
        
        # 加载Telegram配置
        telegram_config = config.get("telegram", {})
        if "TELEGRAM_BOT_TOKEN" in os.environ:
            telegram_config["bot_token"] = os.getenv("TELEGRAM_BOT_TOKEN")
        if "TELEGRAM_CHAT_ID" in os.environ:
            telegram_config["chat_id"] = os.getenv("TELEGRAM_CHAT_ID")
        
        config["telegram"] = telegram_config
        
        # 加载告警Webhook
        if "ALERT_WEBHOOK" in os.environ:
            if "monitoring" not in config:
                config["monitoring"] = {}
            config["monitoring"]["alert_webhook"] = os.getenv("ALERT_WEBHOOK")
        
        return config
    
    @staticmethod
    def validate_required_env_vars() -> List[str]:
        """验证必需的环境变量"""
        missing = []
        
        required_vars = [
            "OKX_API_KEY",
            "OKX_SECRET_KEY",
            "OKX_PASSPHRASE"
        ]
        
        for var in required_vars:
            if var not in os.environ:
                missing.append(var)
        
        return missing
    
    @staticmethod
    def create_env_template():
        """创建环境变量模板文件"""
        template = """# OKX API配置
OKX_API_KEY=your_api_key_here
OKX_SECRET_KEY=your_secret_key_here
OKX_PASSPHRASE=your_passphrase_here

# 可选：多个API密钥（用于密钥轮换）
# OKX_API_KEY_1=
# OKX_SECRET_KEY_1=
# OKX_PASSPHRASE_1=

# Redis配置
REDIS_HOST=localhost
REDIS_PASSWORD=your_redis_password

# Telegram配置
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# 告警Webhook
ALERT_WEBHOOK=https://your-webhook-url

# 加密密钥（用于API密钥加密存储）
TRAE_ENCRYPTION_KEY=your_encryption_key_here
"""
        
        with open(".env.template", 'w', encoding='utf-8') as f:
            f.write(template)
        
        logger.info("Created .env.template file")


def check_security_risks(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """检查配置中的安全风险"""
    risks = []
    
    # 检查硬编码密钥
    okx_config = config.get("okx", {})
    if okx_config.get("api_key") and not okx_config["api_key"].startswith("${"):
        risks.append({
            "type": "hardcoded_api_key",
            "severity": "HIGH",
            "location": "config.yaml -> okx.api_key",
            "recommendation": "使用环境变量覆盖：OKX_API_KEY"
        })
    
    if okx_config.get("secret_key") and not okx_config["secret_key"].startswith("${"):
        risks.append({
            "type": "hardcoded_secret_key",
            "severity": "HIGH",
            "location": "config.yaml -> okx.secret_key",
            "recommendation": "使用环境变量覆盖：OKX_SECRET_KEY"
        })
    
    if okx_config.get("passphrase") and not okx_config["passphrase"].startswith("${"):
        risks.append({
            "type": "hardcoded_passphrase",
            "severity": "HIGH",
            "location": "config.yaml -> okx.passphrase",
            "recommendation": "使用环境变量覆盖：OKX_PASSPHRASE"
        })
    
    # 检查Redis密码
    redis_config = config.get("redis", {})
    if redis_config.get("password") and not redis_config["password"].startswith("${"):
        risks.append({
            "type": "hardcoded_redis_password",
            "severity": "MEDIUM",
            "location": "config.yaml -> redis.password",
            "recommendation": "使用环境变量覆盖：REDIS_PASSWORD"
        })
    
    return risks


def check_key_health() -> Dict[str, Any]:
    """密钥失效自检（模块9）。

    启动时调用：检测密钥三要素是否齐全、是否为占位符、长度是否异常，
    避免密钥失效时静默下单失败。返回结构化报告供启动告警复用。
    """
    manager = APIKeyManager({})
    keys = manager.load_keys_from_env()

    issues: List[str] = []
    placeholder_markers = ("your_", "xxx", "example", "placeholder", "replace", "changeme")

    for name, kd in keys.items():
        api_key = kd.get("api_key", "")
        secret_key = kd.get("secret_key", "")
        passphrase = kd.get("passphrase", "")
        if not (api_key and secret_key and passphrase):
            issues.append(f"{name}: 密钥三要素不完整")
            continue
        for field, val in (("api_key", api_key), ("secret_key", secret_key), ("passphrase", passphrase)):
            low = val.lower()
            if any(m in low for m in placeholder_markers):
                issues.append(f"{name}.{field}: 疑似占位符")
            elif len(val) < 8:
                issues.append(f"{name}.{field}: 长度异常({len(val)})")

    healthy = bool(keys) and not issues
    report = {"healthy": healthy, "key_count": len(keys), "issues": issues}
    if healthy:
        logger.info(f"API key health check passed ({len(keys)} keys)")
    else:
        logger.warning(f"API key health check FAILED: {issues}")
    return report


def check_key_rotation_reminder(max_age_days: int = None,
                                keys_file: str = "data/api_keys.enc") -> Dict[str, Any]:
    """密钥轮换提醒（模块9）。

    以加密密钥文件的最后修改时间作为最近一次轮换的近似时间，超过阈值
    （默认 90 天，可用环境变量 OKX_KEY_ROTATION_DAYS 覆盖）则告警提醒轮换。
    """
    max_age_days = max_age_days or int(os.getenv("OKX_KEY_ROTATION_DAYS", "90"))
    now = datetime.now()

    rotation_ts = None
    if os.path.exists(keys_file):
        try:
            rotation_ts = datetime.fromtimestamp(os.path.getmtime(keys_file))
        except OSError:
            rotation_ts = None

    age_days = (now - rotation_ts).days if rotation_ts else None
    need_rotation = age_days is not None and age_days >= max_age_days

    if need_rotation:
        logger.warning(f"API keys 未轮换超过 {age_days} 天（阈值 {max_age_days} 天），建议尽快轮换")

    return {
        "last_rotation": rotation_ts.isoformat() if rotation_ts else None,
        "age_days": age_days,
        "max_age_days": max_age_days,
        "need_rotation": need_rotation,
    }