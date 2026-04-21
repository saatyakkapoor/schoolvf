"""Enumerations shared across all services."""

from enum import Enum


class GateType(str, Enum):
    EXIT = "EXIT"
    ENTRY = "ENTRY"


class Direction(str, Enum):
    EXIT = "EXIT"
    ENTRY = "ENTRY"


class CameraStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"


class ReviewStatus(str, Enum):
    AUTO = "AUTO"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REVIEWED = "REVIEWED"


class TripStatus(str, Enum):
    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    ABNORMAL_OPEN_REPLACED = "ABNORMAL_OPEN_REPLACED"
    ENTRY_WITHOUT_EXIT = "ENTRY_WITHOUT_EXIT"
    OVERDUE = "OVERDUE"


class AnomalyCode(str, Enum):
    OVERDUE = "OVERDUE"
    ENTRY_WITHOUT_EXIT = "ENTRY_WITHOUT_EXIT"
    DUPLICATE_EXIT = "DUPLICATE_EXIT"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


class AlertType(str, Enum):
    OVERDUE_TRIP = "OVERDUE_TRIP"
    ENTRY_WITHOUT_EXIT = "ENTRY_WITHOUT_EXIT"
    LOW_CONFIDENCE_READ = "LOW_CONFIDENCE_READ"
    CAMERA_OFFLINE = "CAMERA_OFFLINE"
    ABNORMAL_DURATION = "ABNORMAL_DURATION"


class AlertSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class CorrectionTargetType(str, Enum):
    GATE_EVENT = "GATE_EVENT"
    TRIP = "TRIP"
    RAW_PLATE_READ = "RAW_PLATE_READ"
