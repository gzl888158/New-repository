"""Shared, auditable learning memories across trading agents."""

import copy
import hashlib
import threading
import uuid
from collections import Counter, deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from loguru import logger

from utils.helpers import safe_finite


class AgentLearningMemory:
    """Store compact agent decisions and outcomes in the shared EventStore."""

    EVENT_TYPE = "AGENT_LEARNING_MEMORY"
    _CONTEXT_KEYS = frozenset({
        "symbol", "strategy", "regime", "direction", "signal_type",
        "confidence", "factor_score", "regime_confidence",
        "utilization_rate", "avg_correlation", "health_grade",
    })

    def __init__(self, event_store=None, max_entries: int = 5000, enabled: bool = True):
        self._event_store = event_store
        self._enabled = bool(enabled)
        self._entries = deque(maxlen=max(1, int(max_entries)))
        self._seen_memory_ids = set()
        self._lock = threading.RLock()
        if self._enabled and event_store is not None:
            self._restore()

    @staticmethod
    def _clean_value(value: Any, depth: int = 0) -> Any:
        if depth > 3:
            return None
        if value is None or isinstance(value, (bool, int, str)):
            return value[:200] if isinstance(value, str) else value
        if isinstance(value, float):
            return safe_finite(value, 0.0)
        if isinstance(value, dict):
            return {
                str(key)[:80]: AgentLearningMemory._clean_value(item, depth + 1)
                for key, item in list(value.items())[:50]
            }
        if isinstance(value, (list, tuple)):
            return [AgentLearningMemory._clean_value(item, depth + 1) for item in value[:50]]
        return str(value)[:200]

    @classmethod
    def _clean_context(cls, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(context, dict):
            return {}
        clean = {}
        for key, value in context.items():
            if key not in cls._CONTEXT_KEYS:
                continue
            item = cls._clean_value(value)
            if item is not None:
                if isinstance(item, str):
                    item = item.strip().lower()
                    if key == "regime":
                        item = {
                            "trend_bullish": "trending_up",
                            "trend_up": "trending_up",
                            "trend_bearish": "trending_down",
                            "trend_down": "trending_down",
                            "range_bound": "ranging",
                            "range": "ranging",
                        }.get(item, item)
                clean[key] = item
        return clean

    def _restore(self) -> None:
        try:
            records = self._event_store.replay(event_type=self.EVENT_TYPE)
            for record in records[-self._entries.maxlen:]:
                entry = record.get("data")
                if isinstance(entry, dict):
                    self._append_local(entry)
        except Exception as exc:
            logger.warning(f"AgentLearningMemory restore failed: {exc}")

    def _append_local(self, entry: Dict[str, Any]) -> None:
        memory_id = str(entry.get("memory_id") or "")
        if not memory_id or memory_id in self._seen_memory_ids:
            return
        self._entries.append(copy.deepcopy(entry))
        self._seen_memory_ids.add(memory_id)
        if len(self._seen_memory_ids) > self._entries.maxlen * 2:
            self._seen_memory_ids = {
                str(item.get("memory_id")) for item in self._entries if item.get("memory_id")
            }

    def record(
        self,
        agent: str,
        kind: str,
        *,
        context: Optional[Dict[str, Any]] = None,
        decision: Any = None,
        outcome: Any = None,
        veto: bool = False,
        trace_id: Optional[str] = None,
    ) -> Optional[str]:
        """Persist one sanitized memory; repeated trace/agent/kind writes are idempotent."""
        if not self._enabled:
            return None
        agent_name = str(agent or "unknown")[:80]
        memory_kind = str(kind or "observation")[:80]
        trace = str(trace_id or "")[:200]
        identity = f"{agent_name}|{memory_kind}|{trace}" if trace else uuid.uuid4().hex
        memory_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        entry = {
            "memory_id": memory_id,
            "agent": agent_name,
            "kind": memory_kind,
            "timestamp": datetime.now().isoformat(),
            "trace_id": trace,
            "context": self._clean_context(context),
            "decision": self._clean_value(decision),
            "outcome": self._clean_value(outcome),
            "veto": bool(veto),
        }
        with self._lock:
            if memory_id in self._seen_memory_ids:
                return memory_id
            if self._event_store is not None:
                try:
                    self._event_store.append(
                        self.EVENT_TYPE,
                        entry,
                        event_id=f"agent-memory:{memory_id}",
                        source=f"agent:{agent_name}",
                        symbol=entry["context"].get("symbol", ""),
                    )
                except Exception as exc:
                    logger.warning(f"AgentLearningMemory append failed: {exc}")
            self._append_local(entry)
        return memory_id

    def recall(
        self,
        context: Optional[Dict[str, Any]] = None,
        *,
        limit: int = 20,
        lookback_seconds: Optional[float] = None,
        agent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return newest memories matching every supplied whitelisted context field."""
        if not self._enabled:
            return []
        filters = self._clean_context(context)
        cutoff = None
        if lookback_seconds is not None:
            cutoff = datetime.now() - timedelta(seconds=max(0.0, float(lookback_seconds)))
        matches = []
        with self._lock:
            entries = list(self._entries)
        for entry in reversed(entries):
            if agent and entry.get("agent") != agent:
                continue
            if cutoff is not None:
                try:
                    if datetime.fromisoformat(entry.get("timestamp", "")) < cutoff:
                        continue
                except ValueError:
                    continue
            memory_context = entry.get("context") or {}
            if any(memory_context.get(key) != value for key, value in filters.items()):
                continue
            matches.append(copy.deepcopy(entry))
            if len(matches) >= max(1, int(limit)):
                break
        return matches

    def has_veto(
        self,
        context: Optional[Dict[str, Any]] = None,
        *,
        trace_id: Optional[str] = None,
        lookback_seconds: float = 900.0,
    ) -> bool:
        """Check an exact trace veto or a recent matching symbol/strategy/regime rejection."""
        filters = self._clean_context(context)
        if not trace_id and not all(filters.get(key) for key in ("symbol", "strategy")):
            return False
        identity_filters = {
            key: filters[key]
            for key in ("symbol", "strategy", "regime", "direction")
            if key in filters
        }
        memories = self.recall(
            identity_filters,
            limit=self._entries.maxlen,
            lookback_seconds=lookback_seconds,
        )
        return any(
            entry.get("veto")
            and (not trace_id or entry.get("trace_id") == str(trace_id))
            for entry in memories
        )

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            entries = list(self._entries)
        by_agent = Counter(str(entry.get("agent", "unknown")) for entry in entries)
        return {
            "enabled": self._enabled,
            "entries": len(entries),
            "vetoes": sum(bool(entry.get("veto")) for entry in entries),
            "by_agent": dict(by_agent),
            "last_update": entries[-1].get("timestamp") if entries else None,
        }
