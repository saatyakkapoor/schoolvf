"""Cameras CRUD — persisted in app_cameras (vision worker + MJPEG read from DB)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.app.db.models import AppCamera
from apps.api.app.db.session import get_db
from apps.api.app.deps import get_current_username
from apps.api.app.rtsp_probe import probe_rtsp_tcp

from . import schemas as s

router = APIRouter(dependencies=[Depends(get_current_username)])


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _rewrite_rtsp_localhost_for_container(url: str) -> str:
    raw = (url or "").strip()
    if not raw.lower().startswith("rtsp://"):
        return raw
    if not os.path.exists("/.dockerenv"):
        return raw
    p = urlparse(raw)
    host = (p.hostname or "").lower()
    if host not in {"localhost", "127.0.0.1", "::1"}:
        return raw
    if p.port is not None:
        netloc = f"host.docker.internal:{p.port}"
    else:
        netloc = "host.docker.internal"
    if p.username:
        auth = p.username
        if p.password:
            auth += f":{p.password}"
        netloc = f"{auth}@{netloc}"
    return p._replace(netloc=netloc).geturl()


def _camera_to_out(row: AppCamera) -> s.CameraOut:
    st = row.status if row.status in ("online", "offline", "error") else "offline"
    gt = row.gate_type if row.gate_type in ("entry", "exit") else "exit"
    return s.CameraOut(
        id=row.id,
        name=row.name,
        gate_type=gt,  # type: ignore[arg-type]
        stream_url=row.stream_url,
        status=st,  # type: ignore[arg-type]
        is_active=bool(row.is_active),
        last_heartbeat=row.last_heartbeat.isoformat() if row.last_heartbeat else None,
        created_at=row.created_at.isoformat(),
        updated_at=row.updated_at.isoformat(),
    )


@router.get("/cameras", response_model=list[s.CameraOut])
def list_cameras(db: Session = Depends(get_db)) -> list[s.CameraOut]:
    rows = db.query(AppCamera).order_by(AppCamera.id).all()
    return [_camera_to_out(r) for r in rows]


@router.get("/cameras/{camera_id}", response_model=s.CameraOut)
def get_camera(camera_id: str, db: Session = Depends(get_db)) -> s.CameraOut:
    row = db.query(AppCamera).filter(AppCamera.id == camera_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Camera not found")
    return _camera_to_out(row)


@router.post("/cameras", response_model=s.CameraOut)
def create_camera(body: s.CreateCameraIn, db: Session = Depends(get_db)) -> s.CameraOut:
    now = _now()
    cid = uuid4().hex[:12]
    row = AppCamera(
        id=cid,
        name=body.name.strip(),
        gate_type=body.gate_type,
        stream_url=body.stream_url.strip(),
        status="offline",
        is_active=True,
        last_heartbeat=None,
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return _camera_to_out(row)


@router.patch("/cameras/{camera_id}", response_model=s.CameraOut)
def update_camera(
    camera_id: str,
    body: s.UpdateCameraIn,
    db: Session = Depends(get_db),
) -> s.CameraOut:
    row = db.query(AppCamera).filter(AppCamera.id == camera_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Camera not found")
    if body.name is not None:
        row.name = body.name.strip()
    if body.gate_type is not None:
        row.gate_type = body.gate_type
    if body.stream_url is not None:
        row.stream_url = body.stream_url.strip()
    if body.status is not None:
        row.status = body.status
    if body.is_active is not None:
        row.is_active = body.is_active
    row.updated_at = _now()
    db.commit()
    db.refresh(row)
    return _camera_to_out(row)


@router.post("/cameras/{camera_id}/probe", response_model=s.CameraProbeOut)
def probe_camera(camera_id: str, db: Session = Depends(get_db)) -> s.CameraProbeOut:
    row = db.query(AppCamera).filter(AppCamera.id == camera_id).first()
    if not row:
        raise HTTPException(status_code=404, detail="Camera not found")
    url = (row.stream_url or "").strip()
    # USB webcam — no TCP probe possible from API server; vision worker owns the device
    if url.lower().startswith("webcam:"):
        row.updated_at = _now()
        db.commit()
        return s.CameraProbeOut(
            camera_id=camera_id,
            tcp_reachable=True,
            status=row.status or "offline",  # type: ignore[arg-type]
            hint="USB Webcam — the vision worker will open the device directly. Status updates when the worker connects.",
        )
    ok = probe_rtsp_tcp(_rewrite_rtsp_localhost_for_container(url))
    new_status: s.CameraStatus = "online" if ok else "offline"
    row.status = new_status
    if ok:
        row.last_heartbeat = _now()
    row.updated_at = _now()
    db.commit()
    db.refresh(row)
    hint = None
    if not ok:
        hint = (
            "Could not open TCP to the camera host:port from this server. "
            "Check Docker networking to the camera LAN."
        )
    return s.CameraProbeOut(
        camera_id=camera_id,
        tcp_reachable=ok,
        status=new_status,
        hint=hint,
    )
