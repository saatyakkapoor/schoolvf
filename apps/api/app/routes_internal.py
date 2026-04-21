"""Internal endpoints for vision worker (no JWT — use X-Internal-Token)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from apps.api.app.db.models import AppCamera
from apps.api.app.db.session import get_db
from apps.api.app.settings import get_api_settings

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
