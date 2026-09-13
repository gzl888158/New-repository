"""
策略生命周期审计日志 (StrategyAuditLogger)
============================================
哈希链防篡改 + JSONL 持久化的策略冻结/退出/恢复/上下线审计日志。

企业化要求（模块12 策略与风控工程化）：
- 策略冻结、永久退出、人工恢复、上线门禁拦截/放行、参数自动回滚均留痕。
- 每条记录带 event_id / timestamp / hash_prev / hash_current，形成不可篡改哈希链。
- 持久化到 data/strategy_manager/audit.jsonl，可离线校验完整性。
"""

import os
import json
import time
import hashlib
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from loguru import logger


_VALID_EVENT_TYPES = {
    "freeze",           # 迷你回测判定策略失效自动冻结
    "exit",             # 策略永久退出
    "reinstate",        # 人工恢复已退出策略
    "pause",            # 策略自动暂停
    "resume",           # 策略恢复
    "prelaunch_deny",   # 上线门禁拦截
    "prelaunch_allow",  # 上线门禁放行
    "rollback",         # 参数自动回滚
}


class StrategyAuditLogger:
    """策略生命周期审计日志（线程安全 + 哈希链防篡改）。"""

    def __init__(self, log_path: str = "data/strategy_manager/audit.jsonl"):
        self._log_path = log_path
        self._lock = threading.RLock()
        self._entry_counter = 0
        self._prev_hash = ""
        self._entries: List[Dict[str, Any]] = []
        # 恢复历史链状态：进程重启后沿用已有 event_id 计数与末条哈希，
        # 避免从 strat_audit_000001 重新计数导致 event_id 重复、哈希链断裂。
        self.load_from_file()

    def log(self, event_type: str, strategy: str, reason: str = "", **fields: Any) -> Optional[str]:
        """追加一条审计记录，返回 event_id（失败返回 None）。"""
        if event_type not in _VALID_EVENT_TYPES:
            logger.warning(f"StrategyAuditLogger: invalid event_type '{event_type}'")
            return None

        with self._lock:
            self._entry_counter += 1
            event_id = f"strat_audit_{self._entry_counter:06d}"

            entry: Dict[str, Any] = {
                "event_id": event_id,
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "event_type": event_type,
                "strategy": strategy,
                "reason": reason,
            }
            # 追加附加字段（排序保证哈希稳定）
            for k in sorted(fields.keys()):
                entry[k] = fields[k]

            entry["hash_prev"] = self._prev_hash
            content_str = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
            entry["hash_current"] = hashlib.sha256(content_str.encode("utf-8")).hexdigest()

            self._entries.append(entry)
            self._prev_hash = entry["hash_current"]

            try:
                log_dir = os.path.dirname(os.path.abspath(self._log_path))
                os.makedirs(log_dir, exist_ok=True)
                with open(self._log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            except Exception as e:
                logger.error(f"StrategyAuditLogger: failed to persist entry {event_id}: {e}")

            logger.info(
                f"AUDIT [{event_type}] strategy={strategy} reason={reason or '-'} "
                f"(hash={entry['hash_current'][:12]}...)"
            )
            return event_id

    def verify(self) -> Dict[str, Any]:
        """校验内存哈希链完整性。"""
        with self._lock:
            broken_at = None
            invalid: List[str] = []
            for i in range(1, len(self._entries)):
                prev = self._entries[i - 1]
                curr = self._entries[i]
                if curr.get("hash_prev") != prev.get("hash_current"):
                    if broken_at is None:
                        broken_at = i
                    invalid.append(curr.get("event_id", "?"))
                    continue
                recompute = dict(curr)
                recompute.pop("hash_current", None)
                content_str = json.dumps(recompute, sort_keys=True, ensure_ascii=False, default=str)
                if hashlib.sha256(content_str.encode("utf-8")).hexdigest() != curr.get("hash_current"):
                    if broken_at is None:
                        broken_at = i
                    invalid.append(curr.get("event_id", "?"))
            return {
                "valid": len(invalid) == 0,
                "total_entries": len(self._entries),
                "broken_at": broken_at,
                "invalid_entries": invalid,
            }

    def get_entries(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """按事件类型筛选审计记录。"""
        with self._lock:
            if event_type is None:
                return list(self._entries)
            return [e for e in self._entries if e.get("event_type") == event_type]

    def get_stats(self) -> Dict[str, Any]:
        """获取审计统计。"""
        with self._lock:
            counts: Dict[str, int] = {}
            for e in self._entries:
                counts[e["event_type"]] = counts.get(e["event_type"], 0) + 1
            return {"total_entries": len(self._entries), "by_event_type": counts}

    def load_from_file(self) -> int:
        """从 JSONL 文件加载历史审计记录并重建哈希链。

        用于跨进程（如 dashboard 进程）读取主交易进程写入的审计日志，
        以便 verify()/get_stats() 能反映真实历史而非仅内存状态。

        返回加载的条数（文件不存在返回 0）。
        """
        entries: List[Dict[str, Any]] = []
        try:
            with open(self._log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"StrategyAuditLogger: skip malformed line: {e}")
        except FileNotFoundError:
            return 0
        except Exception as e:
            logger.error(f"StrategyAuditLogger: load_from_file error: {e}")
            return 0

        with self._lock:
            self._entries = entries
            self._entry_counter = len(entries)
            self._prev_hash = entries[-1].get("hash_current", "") if entries else ""
        return len(entries)


_audit_logger_instance: Optional[StrategyAuditLogger] = None


def get_strategy_audit_logger() -> StrategyAuditLogger:
    """获取策略审计日志单例。"""
    global _audit_logger_instance
    if _audit_logger_instance is None:
        _audit_logger_instance = StrategyAuditLogger()
    return _audit_logger_instance