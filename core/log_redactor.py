"""
全局日志脱敏器 (Log Redactor)
==============================
生产级敏感信息防护 — 模块 9 子项 (2)：全局日志脱敏

核心功能：
1. 已知密钥值脱敏 — 从环境变量/.env 收集 API Key/Secret/Passphrase 真实值，
   在日志中精确替换为掩码，防止密钥泄露。
2. 模式匹配脱敏 — 识别 api_key=/secret=/token=/authorization= 等键值模式，
   即使密钥值未注册也能脱敏。
3. clOrdId 脱敏 — 保留前后段，中间掩码，防止订单指纹被完整采集。
4. loguru patcher 集成 — 作为 logger.add(patcher=...) 使用，全链路生效。

设计原则：
- 幂等：脱敏操作可重复执行，不改变已脱敏内容。
- 零误伤：精确值替换 + 严格键名匹配，避免误伤正常日志。
- 自包含：初始化时主动加载 .env，不依赖外部加载顺序。
- 永不失败：脱敏过程任何异常均静默吞掉，绝不阻断日志输出。
"""

import os
import re
import threading
from typing import List, Optional, Tuple

# 敏感环境变量键名（从 .env / os.environ 收集真实值做精确替换）
_SENSITIVE_ENV_KEYS = [
    "OKX_API_KEY",
    "OKX_SECRET_KEY",
    "OKX_PASSPHRASE",
    "TRAE_ENCRYPTION_KEY",
]

# 敏感环境变量关键字：扫描 os.environ 中名字含这些关键字的变量，
# 其值也纳入精确替换集合，避免遗漏非标准命名的密钥（如自定义 token）。
_SENSITIVE_ENV_KEYWORDS = ("KEY", "SECRET", "PASSPHRASE", "PASSWORD", "TOKEN", "ENCRYPTION")

# 敏感键名模式（key=value / key:value 形式，键名严格白名单，避免误伤）
_KEY_VALUE_PATTERN = re.compile(
    r'(?i)(api[_-]?key|secret([_-]?key)?|passphrase|password|'
    r'access[_-]?token|refresh[_-]?token|token)'
    r'(\s*[:=]\s*)([^\s,;]+)'
)

# Authorization 头专用（Bearer token 需把 Bearer 后的 token 一并脱敏）
_AUTH_PATTERN = re.compile(
    r'(?i)(authorization\s*[:=]\s*)(bearer\s+)?([^\s,;]+)'
)

# clOrdId / client_order_id 模式
_CLORDID_PATTERN = re.compile(
    r'(?i)(clordid|cl_ord_id|client_order_id)(\s*[:=]\s*)([A-Za-z0-9]{8,})'
)


class LogRedactor:
    """敏感信息脱敏器（线程安全）"""

    def __init__(self):
        # (value, label) 列表，按长度降序，长值优先替换避免短值破坏长值
        self._sensitive_values: List[Tuple[str, str]] = []
        self._lock = threading.RLock()
        self._load_dotenv()
        self._collect_sensitive_values()

    def _load_dotenv(self) -> None:
        """主动加载 .env（幂等，确保环境变量已就绪，不依赖外部加载顺序）"""
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

    def _collect_sensitive_values(self) -> None:
        """从环境变量收集已知敏感值"""
        for key in _SENSITIVE_ENV_KEYS:
            value = os.getenv(key)
            if value:
                self.register(value, key.lower())
        # 兜底扫描：任何名字含敏感关键字的变量（如自定义 token）都纳入精确替换
        for name, value in os.environ.items():
            if not value or len(value) < 6:
                continue
            if any(kw in name.upper() for kw in _SENSITIVE_ENV_KEYWORDS):
                self.register(value, name.lower())

    def register(self, value: str, label: str = "secret") -> None:
        """注册一个已知敏感值（如动态获取的密钥），后续日志命中即脱敏。"""
        if not value or not isinstance(value, str) or len(value) < 6:
            return
        with self._lock:
            for v, _ in self._sensitive_values:
                if v == value:
                    return
            self._sensitive_values.append((value, label))
            self._sensitive_values.sort(key=lambda x: len(x[0]), reverse=True)

    def redact(self, text: str) -> str:
        """脱敏文本，返回脱敏后的字符串。"""
        if not text or not isinstance(text, str):
            return text
        with self._lock:
            values = list(self._sensitive_values)
        # 1. 精确替换已知敏感值
        for value, label in values:
            text = text.replace(value, f"***{label}***")
        # 2. 模式匹配脱敏（key=value 形式，键名白名单）
        text = _KEY_VALUE_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(3)}***", text)
        # 3. Authorization 头脱敏（Bearer token 一并掩码）
        text = _AUTH_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2) or ''}***", text)
        # 4. clOrdId 脱敏
        text = _CLORDID_PATTERN.sub(self._mask_clordid, text)
        return text

    @staticmethod
    def _mask_clordid(m) -> str:
        """clOrdId 脱敏：保留前 6 后 4，中间掩码；过短则仅保留前 4。"""
        value = m.group(3)
        if len(value) <= 12:
            return f"{m.group(1)}{m.group(2)}{value[:4]}***"
        return f"{m.group(1)}{m.group(2)}{value[:6]}***{value[-4:]}"


# 全局单例
_redactor_instance: Optional[LogRedactor] = None


def get_log_redactor() -> LogRedactor:
    """获取全局日志脱敏器单例"""
    global _redactor_instance
    if _redactor_instance is None:
        _redactor_instance = LogRedactor()
    return _redactor_instance


def redact_text(text: str) -> str:
    """便捷函数：脱敏文本"""
    return get_log_redactor().redact(text)


def redact_log_record(record: dict) -> None:
    """loguru patcher：对日志记录的消息与 extra 做脱敏（原地修改 record）。

    作为 logger.configure(patcher=redact_log_record) 使用，作用于所有 sink。
    幂等：已脱敏内容不再变化；任何异常静默吞掉，绝不阻断日志输出。
    """
    try:
        message = record.get("message")
        if isinstance(message, str):
            record["message"] = redact_text(message)

        extra = record.get("extra")
        if isinstance(extra, dict):
            for key, value in list(extra.items()):
                if isinstance(value, str):
                    extra[key] = redact_text(value)
    except Exception:
        pass
