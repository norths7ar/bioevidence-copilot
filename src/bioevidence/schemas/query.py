from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class Query:
    text: str
    rewritten_text: str | None = None
    top_k: int = 10
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
