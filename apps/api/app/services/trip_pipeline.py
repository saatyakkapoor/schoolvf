"""Entry/exit trip logic: EXIT opens a trip; ENTRY closes it (bus returns)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from apps.api.app.db.models import AppGateEvent, AppTrip

log = logging.getLogger("schoolvf.trips")


def infer_direction_from_camera_id(camera_id: str) -> tuple[str, str]:
    """
    Returns (gate_type, direction) for the crossing.
    Demo cameras: cam-entry-1 → entry, cam-exit-1 → exit.
    """
    c = camera_id.lower()
    if "entry" in c:
        return "entry", "entry"
    if "exit" in c:
        return "exit", "exit"
    return "exit", "exit"


def process_detection_for_trips(
    db: Session,
    *,
    camera_id: str,
    camera_name: str,
    plate_number: str,
    confidence: float,
    snapshot_base64: str | None,
    detected_at: datetime | None = None,
) -> None:
    """Insert gate event and update app_trips. Caller commits the session."""
    ts = detected_at or datetime.now(timezone.utc)
    plate = plate_number.upper().replace(" ", "")
    gate_type, direction = infer_direction_from_camera_id(camera_id)

    ev = AppGateEvent(
        camera_id=camera_id,
        camera_name=camera_name or "",
        gate_type=gate_type,
        direction=direction,
        plate_number=plate,
        confidence=confidence,
        snapshot_base64=snapshot_base64,
        detected_at=ts,
        created_at=ts,
    )
    db.add(ev)
    db.flush()

    if direction == "exit":
        prev_open = (
            db.query(AppTrip)
            .filter(AppTrip.plate_number == plate, AppTrip.status == "open")
            .order_by(AppTrip.exit_time.desc())
            .first()
        )
        if prev_open:
            prev_open.status = "closed"
            prev_open.anomaly_code = "duplicate_event"
            prev_open.updated_at = ts

        trip = AppTrip(
            id=uuid4().hex[:12],
            plate_number=plate,
            exit_event_id=ev.id,
            entry_event_id=None,
            exit_time=ts,
            entry_time=None,
            duration_seconds=None,
            status="open",
            anomaly_code="none",
            created_at=ts,
            updated_at=ts,
        )
        db.add(trip)
    else:
        open_trip = (
            db.query(AppTrip)
            .filter(AppTrip.plate_number == plate, AppTrip.status == "open")
            .order_by(AppTrip.exit_time.desc())
            .first()
        )
        if open_trip:
            open_trip.entry_event_id = ev.id
            open_trip.entry_time = ts
            if open_trip.exit_time:
                delta = ts - open_trip.exit_time
                open_trip.duration_seconds = int(delta.total_seconds())
            open_trip.status = "closed"
            open_trip.updated_at = ts
        else:
            log.debug("Entry without open trip for plate %s (orphan entry)", plate)
