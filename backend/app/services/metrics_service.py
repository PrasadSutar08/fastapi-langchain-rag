from __future__ import annotations

from collections import Counter
from threading import Lock


class MetricsService:
    """Small process-local metrics collector; safe for basic diagnostics."""

    def __init__(self):
        self._lock = Lock()
        self._counters = Counter()

    def increment(self, name: str, amount: int = 1):
        with self._lock:
            self._counters[name] += amount

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._counters)


metrics_service = MetricsService()
