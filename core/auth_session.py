"""
手机验证码登录的会话管理与验证码存储
====================================
- VerificationCodeStore : 内存态验证码，含发送冷却、频率限制、校验次数限制
- AuthSessionManager   : 长期会话令牌（默认 30 天），持久化到 data/auth_sessions.json
"""
import os
import secrets
import threading
import time
from typing import Dict, Optional, Tuple

from loguru import logger


class VerificationCodeStore:
    """验证码存储（纯内存，线程安全）。

    限制策略（可配置）：
    - 发送冷却（默认 60s）
    - 每小时最大发送次数（默认 5 次）
    - 验证码有效期（默认 300s）
    - 最大校验尝试次数（默认 5 次）
    """

    def __init__(self, cooldown_seconds: int = 60, max_per_hour: int = 5,
                 expire_seconds: int = 300, max_attempts: int = 5):
        self.cooldown_seconds = cooldown_seconds
        self.max_per_hour = max_per_hour
        self.expire_seconds = expire_seconds
        self.max_attempts = max_attempts
        self._store: Dict[str, dict] = {}
        self._lock = threading.RLock()

    def can_send(self, phone: str) -> Tuple[bool, str]:
        """检查能否发送验证码，返回 (是否可发送, 拒绝原因)。"""
        now = time.time()
        with self._lock:
            rec = self._store.get(phone)
            if rec:
                if now - rec.get("last_send_at", 0) < self.cooldown_seconds:
                    remain = int(self.cooldown_seconds - (now - rec["last_send_at"]))
                    return False, f"发送过于频繁，请 {remain}s 后再试"
                sends = [t for t in rec.get("send_times", []) if now - t < 3600]
                if len(sends) >= self.max_per_hour:
                    return False, "该号码 1 小时内发送次数已达上限"
            return True, ""

    def generate(self, phone: str) -> str:
        """生成并保存验证码，返回 6 位验证码。"""
        code = f"{secrets.randbelow(1000000):06d}"
        now = time.time()
        with self._lock:
            rec = self._store.get(phone)
            if rec is None:
                rec = {"send_times": [], "attempts": 0}
                self._store[phone] = rec
            rec["code"] = code
            rec["expires_at"] = now + self.expire_seconds
            rec["last_send_at"] = now
            rec["attempts"] = 0
            rec["send_times"] = [t for t in rec.get("send_times", []) if now - t < 3600]
            rec["send_times"].append(now)
        return code

    def verify(self, phone: str, code: str) -> Tuple[bool, str]:
        """校验验证码，返回 (是否通过, 说明)。"""
        now = time.time()
        with self._lock:
            rec = self._store.get(phone)
            if rec is None:
                return False, "验证码未发送或已失效，请重新获取"
            if now > rec.get("expires_at", 0):
                return False, "验证码已过期，请重新获取"
            if rec.get("attempts", 0) >= self.max_attempts:
                return False, "验证码错误次数过多，请重新获取"
            rec["attempts"] = rec.get("attempts", 0) + 1
            if not hmac_compare(rec.get("code", ""), code):
                return False, "验证码错误"
            # 校验成功后立即作废，防止重放
            self._store.pop(phone, None)
            return True, "ok"

    def clear(self, phone: str) -> None:
        with self._lock:
            self._store.pop(phone, None)


def hmac_compare(a: str, b: str) -> bool:
    """常量时间比较，避免时序侧信道。"""
    import hmac as _hmac
    return _hmac.compare_digest(str(a), str(b))


class AuthSessionManager:
    """长期会话令牌管理器（持久化到 JSON，支持重启后仍有效）。"""

    def __init__(self, ttl_days: int = 30, filepath: str = None):
        self.ttl_seconds = ttl_days * 86400
        self.filepath = filepath or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "auth_sessions.json"
        )
        self.filepath = os.path.normpath(self.filepath)
        self._sessions: Dict[str, dict] = {}
        self._lock = threading.RLock()
        self._load()

    def _load(self):
        try:
            if os.path.exists(self.filepath):
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = __import__("json").load(f)
                self._sessions = data.get("sessions", {})
                logger.info(f"AuthSessionManager loaded {len(self._sessions)} session(s)")
        except Exception as e:
            logger.warning(f"AuthSessionManager load failed: {e}")

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            tmp = self.filepath + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                __import__("json").dump({"sessions": self._sessions}, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.filepath)
        except Exception as e:
            logger.warning(f"AuthSessionManager save failed: {e}")

    def create(self, phone: str, extra: dict = None) -> str:
        """创建会话令牌，返回 token。"""
        token = secrets.token_urlsafe(32)
        now = time.time()
        with self._lock:
            # 清理同号码旧会话，避免累积
            for t in [t for t, s in self._sessions.items() if s.get("phone") == phone]:
                self._sessions.pop(t, None)
            self._sessions[token] = {
                "phone": phone,
                "created_at": now,
                "expires_at": now + self.ttl_seconds,
                "last_seen_at": now,
            }
            if extra:
                self._sessions[token].update(extra)
            self._cleanup_locked()
            self._save()
        return token

    def validate(self, token: str) -> bool:
        """校验令牌是否有效（存在且未过期）。"""
        if not token:
            return False
        now = time.time()
        with self._lock:
            sess = self._sessions.get(token)
            if sess is None:
                return False
            if now > sess.get("expires_at", 0):
                self._sessions.pop(token, None)
                self._save()
                return False
            sess["last_seen_at"] = now
            return True

    def revoke(self, token: str) -> bool:
        with self._lock:
            if token in self._sessions:
                self._sessions.pop(token, None)
                self._save()
                return True
            return False

    def get_phone(self, token: str) -> Optional[str]:
        with self._lock:
            sess = self._sessions.get(token)
            return sess.get("phone") if sess else None

    def _cleanup_locked(self):
        now = time.time()
        expired = [t for t, s in self._sessions.items() if now > s.get("expires_at", 0)]
        for t in expired:
            self._sessions.pop(t, None)
