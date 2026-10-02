"""Thread-safe process-local counters for the strategy-to-exchange signal path."""

from collections import Counter, defaultdict
from datetime import datetime
import threading
from typing import Any, Dict


class SignalFlowStats:
    def __init__(self):
        self._lock = threading.RLock()
        self._started_at = datetime.now().isoformat()
        self._totals = Counter()
        self._by_strategy = defaultdict(Counter)
        self._rejects_by_layer = Counter()
        self._rejects_by_reason = Counter()
        self._rejects_by_strategy = defaultdict(Counter)

    def record(
        self,
        stage: str,
        strategy: str = "",
        layer: str = "",
        reason: str = "",
    ) -> None:
        with self._lock:
            self._totals[stage] += 1
            if strategy:
                self._by_strategy[strategy][stage] += 1
            if layer:
                self._rejects_by_layer[layer] += 1
                if strategy:
                    self._rejects_by_strategy[strategy][layer] += 1
            if reason:
                self._rejects_by_reason[reason[:120]] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "scope": "current_process",
                "started_at": self._started_at,
                "updated_at": datetime.now().isoformat(),
                "totals": dict(self._totals),
                "by_strategy": {
                    strategy: dict(counts)
                    for strategy, counts in self._by_strategy.items()
                },
                "rejections": {
                    "by_layer": dict(self._rejects_by_layer),
                    "by_reason": dict(self._rejects_by_reason),
                    "by_strategy": {
                        strategy: dict(counts)
                        for strategy, counts in self._rejects_by_strategy.items()
                    },
                },
            }


_stats = SignalFlowStats()


def record_signal_flow_event(
    stage: str,
    strategy: str = "",
    layer: str = "",
    reason: str = "",
) -> None:
    _stats.record(stage, strategy, layer, reason)


def get_signal_flow_stats() -> Dict[str, Any]:
    return _stats.snapshot()