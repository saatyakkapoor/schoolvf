"""Response models aligned with the dashboard TypeScript types."""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

GateType = Literal["entry", "exit"]
TripStatus = Literal["open", "closed", "overdue"]
AnomalyCode = Literal[
    "none",
    "low_confidence",
    "plate_mismatch",
    "duplicate_event",
    "orphan_entry",
    "rapid_re_entry",
]
AlertSeverity = Literal["info", "warning", "critical"]
ReviewStatus = Literal["pending", "approved", "corrected", "rejected"]
CameraStatus = Literal["online", "offline", "error"]
BusStatus = Literal["inside", "outside", "unknown"]

T = TypeVar("T")


class PaginatedItems(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int = 1
    page_size: int = 20
    pages: int = 0


class CameraOut(BaseModel):
    id: str
    name: str
    gate_type: GateType
    stream_url: str
    status: CameraStatus
    is_active: bool = True
    last_heartbeat: str | None
    created_at: str
    updated_at: str


class GateEventOut(BaseModel):
    id: str
    camera_id: str
    camera_name: str
    gate_type: GateType
    plate_number: str
    confidence: float
    snapshot_url: str | None
    raw_candidates: list[str]
    anomaly_code: AnomalyCode
    review_status: ReviewStatus
    trip_id: str | None
    route_number: str | None = None
    timestamp: str
    created_at: str


class TripOut(BaseModel):
    id: str
    plate_number: str
    exit_event_id: str
    entry_event_id: str | None
    exit_time: str
    entry_time: str | None
    duration_seconds: int | None
    status: TripStatus
    anomaly_code: AnomalyCode
    created_at: str
    updated_at: str


class AlertOut(BaseModel):
    id: str
    trip_id: str | None
    event_id: str | None
    plate_number: str
    severity: AlertSeverity
    alert_type: str
    message: str
    resolved: bool
    resolved_at: str | None
    resolved_by: str | None
    resolution_note: str | None
    created_at: str


class PlateDetailOut(BaseModel):
    plate_number: str
    current_status: BusStatus
    last_seen: str | None
    last_camera: str | None
    total_trips: int
    recent_trips: list[TripOut]
    recent_events: list[GateEventOut]


class ManualCorrectionOut(BaseModel):
    id: str
    event_id: str
    original_plate: str
    corrected_plate: str
    reason: str
    corrected_by: str
    created_at: str


class DashboardSummaryOut(BaseModel):
    total_buses_known: int
    buses_outside_now: int
    buses_overdue: int
    alerts_today: int
    events_today: int
    trips_today: int
    recent_events: list[GateEventOut]
    active_alerts: list[AlertOut]


class CurrentBusStatusItem(BaseModel):
    plate_number: str
    status: BusStatus
    last_event_time: str | None
    last_camera: str | None
    current_trip_id: str | None
    duration_outside_seconds: int | None


class CreateCameraIn(BaseModel):
    name: str
    gate_type: GateType
    stream_url: str


class UpdateCameraIn(BaseModel):
    name: str | None = None
    gate_type: GateType | None = None
    stream_url: str | None = None
    status: CameraStatus | None = None
    is_active: bool | None = None


class CameraProbeOut(BaseModel):
    camera_id: str
    tcp_reachable: bool
    status: CameraStatus
    hint: str | None = None


class ResolveAlertIn(BaseModel):
    resolved_by: str
    resolution_note: str


class CreateCorrectionIn(BaseModel):
    event_id: str
    corrected_plate: str
    reason: str
    corrected_by: str
