"""In-memory trip state: EXIT opens, ENTRY closes; anomalies and overdue."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from uuid import uuid4

from packages.shared.svf_types.enums import AnomalyCode, Direction, TripStatus


@dataclass
class GateEventForMatching:
    """Minimal gate event input for trip matching."""

    event_id: str
    plate_norm: str
    direction: Direction
    detected_at: datetime
    confidence: float


@dataclass
class TripRecord:
    id: str
    plate_number: str
    exit_event_id: str
    entry_event_id: Optional[str]
    exit_time: datetime
    entry_time: Optional[datetime]
    status: TripStatus
    anomaly_code: Optional[AnomalyCode] = None


@dataclass
class TripUpdateResult:
    """Outcome of processing one gate event."""

    affected_trips: list[TripRecord] = field(default_factory=list)
    alerts: list[tuple[AnomalyCode, str]] = field(default_factory=list)


class TripMatcher:
    """
    EXIT starts a trip (PENDING); ENTRY completes it (COMPLETED).

    - ENTRY without open EXIT => ENTRY_WITHOUT_EXIT (no trip row; alert only).
    - EXIT while plate already has open trip => previous ABNORMAL_OPEN_REPLACED, new PENDING.
    - Overdue: use scan_overdue() with SLA threshold.
    """

    def __init__(self, overdue_threshold: timedelta | None = None) -> None:
        self._overdue_threshold = overdue_threshold or timedelta(hours=2)
        # plate -> open trip (only PENDING)
        self._open: dict[str, TripRecord] = {}
        self._all: dict[str, TripRecord] = {}

    def handle_gate_event(self, event: GateEventForMatching) -> TripUpdateResult:
        result = TripUpdateResult()
        plate = event.plate_norm
        if event.direction == Direction.EXIT:
            self._handle_exit(event, plate, result)
        else:
            self._handle_entry(event, plate, result)
        return result

    def _handle_exit(self, event: GateEventForMatching, plate: str, result: TripUpdateResult) -> None:
        prev = self._open.get(plate)
        if prev is not None:
            prev.status = TripStatus.ABNORMAL_OPEN_REPLACED
            prev.anomaly_code = AnomalyCode.DUPLICATE_EXIT
            result.affected_trips.append(prev)
            result.alerts.append((AnomalyCode.DUPLICATE_EXIT, prev.id))

        trip = TripRecord(
            id=uuid4().hex,
            plate_number=plate,
            exit_event_id=event.event_id,
            entry_event_id=None,
            exit_time=event.detected_at,
            entry_time=None,
            status=TripStatus.PENDING,
        )
        self._all[trip.id] = trip
        self._open[plate] = trip
        result.affected_trips.append(trip)

    def _handle_entry(self, event: GateEventForMatching, plate: str, result: TripUpdateResult) -> None:
        open_trip = self._open.get(plate)
        if open_trip is None:
            result.alerts.append((AnomalyCode.ENTRY_WITHOUT_EXIT, event.event_id))
            return

        open_trip.entry_event_id = event.event_id
        open_trip.entry_time = event.detected_at
        open_trip.status = TripStatus.COMPLETED
        open_trip.anomaly_code = None
        del self._open[plate]
        result.affected_trips.append(open_trip)

    def scan_overdue(self, now: datetime) -> TripUpdateResult:
        """Mark open PENDING trips exceeding SLA as OVERDUE."""
        result = TripUpdateResult()
        for _plate, trip in list(self._open.items()):
            if trip.status != TripStatus.PENDING:
                continue
            if now - trip.exit_time <= self._overdue_threshold:
                continue
            trip.status = TripStatus.OVERDUE
            trip.anomaly_code = AnomalyCode.OVERDUE
            result.affected_trips.append(trip)
            result.alerts.append((AnomalyCode.OVERDUE, trip.id))
        return result

    def get_open_trip(self, plate: str) -> Optional[TripRecord]:
        return self._open.get(plate)
