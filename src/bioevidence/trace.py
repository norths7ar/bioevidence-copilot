from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from uuid import uuid4


@dataclass(slots=True)
class TraceRecorder:
    run_id: str = field(default_factory=lambda: uuid4().hex)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _events: list[dict[str, object]] = field(default_factory=list, init=False, repr=False)

    def emit(self, event: str, **details: object) -> dict[str, object]:
        record = {
            "sequence": len(self._events) + 1,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "run_id": self.run_id,
            "event": event,
            **details,
        }
        self._events.append(record)
        return dict(record)

    def events(self) -> tuple[dict[str, object], ...]:
        return tuple(dict(event) for event in self._events)

    @staticmethod
    def start_timer() -> float:
        return perf_counter()

    @staticmethod
    def elapsed_ms(started_at: float) -> int:
        return round((perf_counter() - started_at) * 1000)
