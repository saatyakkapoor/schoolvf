"""Camera-related shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

import numpy as np


class CameraConnectionState(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


@dataclass(frozen=True)
class ROIConfig:
    """Region of interest for gate detection zone."""

    x1: int
    y1: int
    x2: int
    y2: int

    def contains_point(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def contains_bbox(self, bx1: int, by1: int, bx2: int, by2: int) -> bool:
        """Check if a bounding box overlaps with this ROI."""
        return not (bx2 < self.x1 or bx1 > self.x2 or by2 < self.y1 or by1 > self.y2)

    def as_slice(self) -> tuple[slice, slice]:
        """Return numpy-compatible slices for cropping."""
        return (slice(self.y1, self.y2), slice(self.x1, self.x2))


@dataclass
class CameraConfig:
    """Configuration for a single camera source."""

    camera_id: str
    stream_url: str
    sampling_fps: float = 5.0
    roi: Optional[ROIConfig] = None
    direction: str = "entry"
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = self.camera_id


@dataclass
class FrameEnvelope:
    """Wrapper around a captured frame with metadata."""

    frame: np.ndarray
    camera_id: str
    timestamp: datetime
    sequence_number: int
    frame_id: str = field(default_factory=lambda: uuid4().hex[:12])

    @property
    def height(self) -> int:
        return self.frame.shape[0]

    @property
    def width(self) -> int:
        return self.frame.shape[1]


@dataclass
class CameraStatus:
    """Health status report for a camera source."""

    camera_id: str
    state: CameraConnectionState
    last_frame_time: Optional[datetime] = None
    reconnect_count: int = 0
    fps_actual: float = 0.0
    error_message: Optional[str] = None
    uptime_seconds: float = 0.0
