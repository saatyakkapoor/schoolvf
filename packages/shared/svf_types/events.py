"""Event-related shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4


class Direction(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


class GateEventStatus(str, Enum):
    PENDING = "pending"
    MATCHED = "matched"
    UNMATCHED = "unmatched"
    ERROR = "error"


@dataclass
class GateEventCreate:
    """Payload for creating a gate event in the API backend."""

    plate_text: str
    direction: Direction
    camera_id: str
    timestamp: datetime
    confidence: float
    snapshot_path: Optional[str] = None
    candidate_count: int = 0
    event_id: str = field(default_factory=lambda: uuid4().hex)
