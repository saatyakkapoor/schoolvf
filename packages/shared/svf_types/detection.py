"""Detection-related shared types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class PlateBox:
    """A detected license plate bounding box with associated data."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    crop: np.ndarray

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)


@dataclass
class PlateCandidate:
    """A plate candidate combining detection and OCR results."""

    plate_text: str
    ocr_confidence: float
    detector_confidence: float
    crop: np.ndarray
    camera_id: str
    timestamp: datetime
    frame_id: str
    bbox: tuple[int, int, int, int]
    direction: str = "entry"
    snapshot_path: Optional[str] = None
