"""Same-direction duplicate suppression for gate events."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DirectionDeduper:
    """
    Ignore repeated same-direction reads for the same plate within a cooldown window.

    Keyed by (normalized_plate, direction) where direction is a stable string
    (e.g. 'EXIT' / 'ENTRY').
    """

    cooldown_seconds: float = 90.0
    _last_emit: dict[tuple[str, str], datetime] = field(default_factory=dict)

    def should_emit(self, plate_norm: str, direction: str, at: datetime) -> bool:
        """Return False if this read should be suppressed as a duplicate."""
        key = (plate_norm, direction.upper())
        prev = self._last_emit.get(key)
        if prev is not None:
            delta = (at - prev).total_seconds()
            if 0 <= delta < self.cooldown_seconds:
                return False
        self._last_emit[key] = at
        return True

    def last_emit_at(self, plate_norm: str, direction: str) -> Optional[datetime]:
        return self._last_emit.get((plate_norm, direction.upper()))
