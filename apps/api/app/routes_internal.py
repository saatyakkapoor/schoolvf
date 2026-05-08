"""Internal endpoints for vision worker (no JWT — use X-Internal-Token)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.api.app.db.models import AppCamera, AppVehicle
from apps.api.app.db.session import get_db
from apps.api.app.settings import get_api_settings


def _normalize_route(raw: str) -> str:
    cleaned = (raw or "").strip().upper().replace(" ", "").replace("_", "")
    if not cleaned:
        return ""
    if cleaned.startswith("AR-") and cleaned[3:].isdigit():
        return f"AR-{int(cleaned[3:]):02d}"
    if cleaned.startswith("AR") and cleaned[2:].isdigit():
        return f"AR-{int(cleaned[2:]):02d}"
    if cleaned.isdigit():
        return f"AR-{int(cleaned):02d}"
    return cleaned

router = APIRouter()


class VisionCameraItem(BaseModel):
    id: str
    name: str
    gate_type: str
    stream_url: str
    is_active: bool


@router.get("/internal/vision-cameras", response_model=list[VisionCameraItem])
def list_vision_cameras(
    db: Session = Depends(get_db),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> list[VisionCameraItem]:
    """Vision worker polls this to run one RTSP pipeline per active camera (no dashboard required)."""
    settings = get_api_settings()
    if not x_internal_token or x_internal_token != settings.INTERNAL_INGEST_SECRET:
        raise HTTPException(status_code=403, detail="Invalid internal token")
    rows = (
        db.query(AppCamera)
        .filter(AppCamera.is_active.is_(True))
        .order_by(AppCamera.id)
        .all()
    )
    out: list[VisionCameraItem] = []
    for r in rows:
        url = (r.stream_url or "").strip()
        url_l = url.lower()
        # Accept RTSP streams and USB webcam sources (webcam:N); skip unconfigured cameras
        if not url_l.startswith("rtsp") and not url_l.startswith("webcam:"):
            continue
        out.append(
            VisionCameraItem(
                id=r.id,
                name=r.name or r.id,
                gate_type=r.gate_type,
                stream_url=url,
                is_active=bool(r.is_active),
            )
        )
    return out


class VehicleByRouteItem(BaseModel):
    plate_number: str
    route_number: str
    route_name: str
    driver_name: str


@router.get("/internal/vehicles-by-route", response_model=VehicleByRouteItem | None)
def get_vehicle_by_route(
    route: str,
    db: Session = Depends(get_db),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> VehicleByRouteItem | None:
    """Look up the registered vehicle for a route number. Used by the vision
    worker's deep-OCR enhancer to verify a re-OCR'd plate against the
    registry. Returns null when no active vehicle is registered for the route."""
    settings = get_api_settings()
    if not x_internal_token or x_internal_token != settings.INTERNAL_INGEST_SECRET:
        raise HTTPException(status_code=403, detail="Invalid internal token")
    canon = _normalize_route(route)
    if not canon:
        return None
    candidates = [route.strip().upper(), canon]
    if canon.startswith("AR-"):
        bare = canon[3:].lstrip("0") or canon[3:]
        candidates.append(bare)
    veh = (
        db.query(AppVehicle)
        .filter(AppVehicle.is_active.is_(True), AppVehicle.route_number.in_(candidates))
        .order_by(AppVehicle.created_at.asc())
        .first()
    )
    if veh is None:
        return None
    return VehicleByRouteItem(
        plate_number=(veh.plate_number or "").strip().upper(),
        route_number=(veh.route_number or "").strip().upper(),
        route_name=veh.route_name or "",
        driver_name=veh.driver_name or "",
    )
