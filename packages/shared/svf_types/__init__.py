"""Shared type definitions for the SchoolVF monorepo."""

from packages.shared.svf_types.camera import (
    CameraConfig,
    CameraStatus,
    FrameEnvelope,
    ROIConfig,
)
from packages.shared.svf_types.detection import PlateBox, PlateCandidate
from packages.shared.svf_types.enums import (
    AlertSeverity,
    AlertType,
    AnomalyCode,
    CorrectionTargetType,
    GateType,
    ReviewStatus,
    TripStatus,
)
from packages.shared.svf_types.enums import CameraStatus as CameraDBStatus
from packages.shared.svf_types.enums import Direction as DirectionEnum
from packages.shared.svf_types.events import (
    Direction,
    GateEventCreate,
    GateEventStatus,
)
from packages.shared.svf_types.ocr import OCRResult

__all__ = [
    "AlertSeverity",
    "AlertType",
    "AnomalyCode",
    "CameraConfig",
    "CameraDBStatus",
    "CameraStatus",
    "CorrectionTargetType",
    "Direction",
    "DirectionEnum",
    "FrameEnvelope",
    "GateEventCreate",
    "GateEventStatus",
    "GateType",
    "OCRResult",
    "PlateBox",
    "PlateCandidate",
    "ROIConfig",
    "ReviewStatus",
    "TripStatus",
]
