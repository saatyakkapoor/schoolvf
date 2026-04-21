"""REST API routes backed by PostgreSQL (app_* tables). Cameras: routes_cameras."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc
from sqlalchemy.orm import Session

from apps.api.app.db.models import AppGateEvent, AppTrip, AppVehicle
from apps.api.app.db.session import get_db
from apps.api.app.deps import get_current_username
from apps.api.app.routes_vehicles import count_active_vehicles

from . import schemas as s

router = APIRouter(dependencies=[Depends(get_current_username)])

_NOW = lambda: datetime.now(timezone.utc).isoformat()


def _start_of_today_utc() -> datetime:
    now = datetime.now(timezone.utc)
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _trip_to_out(t: AppTrip) -> s.TripOut:
    ac = t.anomaly_code or "none"
    if ac not in ("none", "low_confidence", "plate_mismatch", "duplicate_event", "orphan_entry", "rapid_re_entry"):
        ac = "none"
    st = t.status
    if st not in ("open", "closed", "overdue"):
        st = "closed"
    return s.TripOut(
        id=t.id,
        plate_number=t.plate_number,
        exit_event_id=str(t.exit_event_id),
        entry_event_id=str(t.entry_event_id) if t.entry_event_id else None,
        exit_time=t.exit_time.isoformat(),
        entry_time=t.entry_time.isoformat() if t.entry_time else None,
        duration_seconds=t.duration_seconds,
        status=st,  # type: ignore[arg-type]
        anomaly_code=ac,  # type: ignore[arg-type]
        created_at=t.created_at.isoformat(),
        updated_at=t.updated_at.isoformat(),
    )


def _event_to_out(ev: AppGateEvent, route_number: str | None = None) -> s.GateEventOut:
    return s.GateEventOut(
        id=str(ev.id),
        camera_id=ev.camera_id,
        camera_name=ev.camera_name,
        gate_type=ev.gate_type,  # type: ignore[arg-type]
        plate_number=ev.plate_number,
        confidence=ev.confidence,
        snapshot_url=None,
        raw_candidates=[],
        anomaly_code="none",
        review_status="approved",
        trip_id=None,
        route_number=route_number,
        timestamp=ev.detected_at.isoformat(),
        created_at=ev.created_at.isoformat(),
    )


def _attach_routes(events: list[AppGateEvent], db: Session) -> list[s.GateEventOut]:
    if not events:
        return []
    plates = {e.plate_number for e in events}
    vehicles = db.query(AppVehicle).filter(AppVehicle.plate_number.in_(plates), AppVehicle.is_active.is_(True)).all()
    route_map = {v.plate_number: v.route_number for v in vehicles}
    return [_event_to_out(e, route_map.get(e.plate_number)) for e in events]


@router.get("/dashboard/summary", response_model=s.DashboardSummaryOut)
def dashboard_summary(db: Session = Depends(get_db)) -> s.DashboardSummaryOut:
    start = _start_of_today_utc()
    buses_known = count_active_vehicles(db)
    open_trips = db.query(AppTrip).filter(AppTrip.status == "open").count()
    trips_today = db.query(AppTrip).filter(AppTrip.exit_time >= start).count()
    events_today = db.query(AppGateEvent).filter(AppGateEvent.detected_at >= start).count()
    recent_ev = (
        db.query(AppGateEvent).order_by(desc(AppGateEvent.detected_at)).limit(8).all()
    )
    return s.DashboardSummaryOut(
        total_buses_known=buses_known,
        buses_outside_now=open_trips,
        buses_overdue=0,
        alerts_today=0,
        events_today=events_today,
        trips_today=trips_today,
        recent_events=_attach_routes(recent_ev, db),
        active_alerts=[],
    )


def _empty_page(page: int, page_size: int) -> tuple[int, int]:
    return max(1, page), max(1, min(100, page_size))


@router.get("/trips", response_model=s.PaginatedItems[s.TripOut])
def list_trips(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> s.PaginatedItems[s.TripOut]:
    p, ps = _empty_page(page, page_size)
    q = db.query(AppTrip).order_by(desc(AppTrip.exit_time))
    total = q.count()
    rows = q.offset((p - 1) * ps).limit(ps).all()
    pages = (total + ps - 1) // ps if total else 0
    return s.PaginatedItems(
        items=[_trip_to_out(t) for t in rows],
        total=total,
        page=p,
        page_size=ps,
        pages=pages,
    )


@router.get("/trips/{trip_id}", response_model=s.TripOut)
def get_trip(trip_id: str, db: Session = Depends(get_db)) -> s.TripOut:
    t = db.query(AppTrip).filter(AppTrip.id == trip_id).first()
    if not t:
        raise HTTPException(status_code=404, detail="Trip not found")
    return _trip_to_out(t)


@router.get("/trips/bus-status", response_model=list[s.CurrentBusStatusItem])
def bus_status(db: Session = Depends(get_db)) -> list[s.CurrentBusStatusItem]:
    """Per active fleet plate: outside if an open trip exists, else inside."""
    vehicles = db.query(AppVehicle).filter(AppVehicle.is_active.is_(True)).all()
    out: list[s.CurrentBusStatusItem] = []
    for v in vehicles:
        open_t = (
            db.query(AppTrip)
            .filter(AppTrip.plate_number == v.plate_number, AppTrip.status == "open")
            .order_by(desc(AppTrip.exit_time))
            .first()
        )
        last_ev = (
            db.query(AppGateEvent)
            .filter(AppGateEvent.plate_number == v.plate_number)
            .order_by(desc(AppGateEvent.detected_at))
            .first()
        )
        st: s.BusStatus = "outside" if open_t else "inside"
        out.append(
            s.CurrentBusStatusItem(
                plate_number=v.plate_number,
                status=st,
                last_event_time=last_ev.detected_at.isoformat() if last_ev else None,
                last_camera=last_ev.camera_name if last_ev else None,
                current_trip_id=open_t.id if open_t else None,
                duration_outside_seconds=None,
            )
        )
    return out


@router.get("/events", response_model=s.PaginatedItems[s.GateEventOut])
def list_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
) -> s.PaginatedItems[s.GateEventOut]:
    p, ps = _empty_page(page, page_size)
    q = db.query(AppGateEvent).order_by(desc(AppGateEvent.detected_at))
    total = q.count()
    rows = q.offset((p - 1) * ps).limit(ps).all()
    pages = (total + ps - 1) // ps if total else 0
    return s.PaginatedItems(
        items=_attach_routes(rows, db),
        total=total,
        page=p,
        page_size=ps,
        pages=pages,
    )


@router.get("/events/{event_id}", response_model=s.GateEventOut)
def get_event(event_id: str, db: Session = Depends(get_db)) -> s.GateEventOut:
    import uuid as uuid_mod

    try:
        uid = uuid_mod.UUID(event_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Event not found")
    ev = db.query(AppGateEvent).filter(AppGateEvent.id == uid).first()
    if not ev:
        raise HTTPException(status_code=404, detail="Event not found")
    return _attach_routes([ev], db)[0]


@router.get("/alerts", response_model=s.PaginatedItems[s.AlertOut])
def list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> s.PaginatedItems[s.AlertOut]:
    p, ps = _empty_page(page, page_size)
    return s.PaginatedItems(items=[], total=0, page=p, page_size=ps, pages=0)


@router.post("/alerts/{alert_id}/resolve", response_model=s.AlertOut)
def resolve_alert(alert_id: str, body: s.ResolveAlertIn) -> s.AlertOut:
    raise HTTPException(status_code=404, detail="Alert not found")


@router.get("/plates/{plate_number}", response_model=s.PlateDetailOut)
def plate_detail(plate_number: str, db: Session = Depends(get_db)) -> s.PlateDetailOut:
    plate = plate_number.upper().replace(" ", "")
    open_t = (
        db.query(AppTrip)
        .filter(AppTrip.plate_number == plate, AppTrip.status == "open")
        .order_by(desc(AppTrip.exit_time))
        .first()
    )
    current: s.BusStatus = "outside" if open_t else "inside"
    last_ev = (
        db.query(AppGateEvent)
        .filter(AppGateEvent.plate_number == plate)
        .order_by(desc(AppGateEvent.detected_at))
        .first()
    )
    recent_trips = (
        db.query(AppTrip)
        .filter(AppTrip.plate_number == plate)
        .order_by(desc(AppTrip.exit_time))
        .limit(10)
        .all()
    )
    recent_events = (
        db.query(AppGateEvent)
        .filter(AppGateEvent.plate_number == plate)
        .order_by(desc(AppGateEvent.detected_at))
        .limit(10)
        .all()
    )
    return s.PlateDetailOut(
        plate_number=plate,
        current_status=current,
        last_seen=last_ev.detected_at.isoformat() if last_ev else None,
        last_camera=last_ev.camera_name if last_ev else None,
        total_trips=db.query(AppTrip).filter(AppTrip.plate_number == plate).count(),
        recent_trips=[_trip_to_out(t) for t in recent_trips],
        recent_events=_attach_routes(recent_events, db),
    )


@router.get("/corrections", response_model=s.PaginatedItems[s.ManualCorrectionOut])
def list_corrections(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
) -> s.PaginatedItems[s.ManualCorrectionOut]:
    p, ps = _empty_page(page, page_size)
    return s.PaginatedItems(items=[], total=0, page=p, page_size=ps, pages=0)


@router.post("/corrections", response_model=s.ManualCorrectionOut)
def create_correction(body: s.CreateCorrectionIn) -> s.ManualCorrectionOut:
    return s.ManualCorrectionOut(
        id=uuid4().hex,
        event_id=body.event_id,
        original_plate="",
        corrected_plate=body.corrected_plate,
        reason=body.reason,
        corrected_by=body.corrected_by,
        created_at=_NOW(),
    )
