"""Live plate detections: ingest from worker, WebSocket broadcast, MJPEG stream proxy."""

from __future__ import annotations

import asyncio
import os
import uuid
from collections import deque
from urllib.parse import urlparse
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Depends, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from apps.api.app.db.models import AppBusSwap, AppCamera, AppVehicle
from apps.api.app.db.session import get_db
from apps.api.app.deps import get_current_username
from apps.api.app.services.trip_pipeline import process_detection_for_trips
from apps.api.app.settings import get_api_settings
from sqlalchemy.orm import Session

router = APIRouter()

_recent: deque[dict[str, Any]] = deque(maxlen=300)
_debug_logs: deque[dict[str, Any]] = deque(maxlen=400)

# Per-camera stream health: camera_id -> {"connected": bool, "fps": float, "last_frame": float}
_stream_health: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class DetectionIn(BaseModel):
    plate_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    camera_id: str = "default"
    camera_name: str = "Camera"
    snapshot_base64: str | None = None
    detected_route: str | None = None   # route OCR'd from bus placard / LED display


class AdjustIn(BaseModel):
    swap_type: str   # "temporary" | "permanent" | "glitch" | "other"
    notes: str | None = None


class DebugIn(BaseModel):
    """Vision worker (or tools) push structured debug; shown on Live Monitor."""

    message: str = Field(..., max_length=240)
    detail: dict[str, Any] | None = None
    source: str = Field(default="vision-worker", max_length=64)


def _truncate_debug_detail(d: Any, *, max_depth: int = 4, _depth: int = 0) -> Any:
    if _depth > max_depth:
        return "…"
    if isinstance(d, dict):
        return {
            str(k)[:80]: _truncate_debug_detail(v, max_depth=max_depth, _depth=_depth + 1)
            for k, v in list(d.items())[:48]
        }
    if isinstance(d, list):
        return [_truncate_debug_detail(x, max_depth=max_depth, _depth=_depth + 1) for x in d[:40]]
    if isinstance(d, str) and len(d) > 600:
        return d[:600] + "…"
    return d


async def _push_debug_event(message: str, detail: dict[str, Any] | None, source: str) -> None:
    entry = {
        "id": uuid.uuid4().hex[:12],
        "at": datetime.now(timezone.utc).isoformat(),
        "message": message,
        "detail": _truncate_debug_detail(detail) if detail else {},
        "source": source,
    }
    _debug_logs.appendleft(entry)
    await manager.broadcast({"type": "debug", "payload": entry})


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)

    async def broadcast(self, payload: dict[str, Any]) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------


def _decode_ws_token(token: str) -> str | None:
    settings = get_api_settings()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=["HS256"])
        sub = payload.get("sub")
        if isinstance(sub, str) and sub:
            return sub
    except JWTError:
        pass
    return None


# ---------------------------------------------------------------------------
# MJPEG stream proxy
# ---------------------------------------------------------------------------

_BOUNDARY = b"--frame"
_FRAME_HEADER = b"Content-Type: image/jpeg\r\n\r\n"

# Low-latency preview settings
_MJPEG_TARGET_FPS = 20
_MJPEG_MAX_WIDTH = 1280
_MJPEG_JPEG_QUALITY = 75
# How many frames to grab()-and-discard to drain the RTSP jitter buffer before retrieve().
# grab() only reads the compressed header — much cheaper than cap.read() which decodes pixels.
_MJPEG_DRAIN_GRABS = 2
# Consecutive frame failures before triggering a reconnect
_MJPEG_FAIL_THRESHOLD = 4
# Seconds between keepalive frames sent while the camera is stalled / reconnecting.
# Prevents browsers and nginx from closing the idle connection.
_MJPEG_KEEPALIVE_SEC = 1.5


def _ffmpeg_opts_low_latency() -> str:
    return "rtsp_transport;tcp|fflags|nobuffer|flags|low_delay|max_delay|0|analyzeduration|100000|probesize|100000"


def _rtsp_url_has_credentials(url: str) -> bool:
    try:
        return bool(urlparse(url.strip()).username)
    except Exception:
        return False


def _same_rtsp_endpoint(a: str, b: str) -> bool:
    """True when RTSP host+port match (used for safe credential fallback)."""
    try:
        pa = urlparse(a.strip())
        pb = urlparse(b.strip())
        if pa.scheme.lower() != "rtsp" or pb.scheme.lower() != "rtsp":
            return False
        ah = (pa.hostname or "").lower()
        bh = (pb.hostname or "").lower()
        ap = pa.port or 554
        bp = pb.port or 554
        return bool(ah) and ah == bh and ap == bp
    except Exception:
        return False


def _resolve_stream_url(stored: str) -> str:
    """
    Resolve stream URL for preview.
    Use CAMERA_RTSP_URL credentials only when stored/fallback point to the same RTSP endpoint.
    """
    settings = get_api_settings()
    fallback = (settings.CAMERA_RTSP_URL or "").strip()
    s = (stored or "").strip()
    use_fallback = (
        bool(fallback)
        and bool(s)
        and s.lower().startswith("rtsp://")
        and fallback.lower().startswith("rtsp://")
        and not _rtsp_url_has_credentials(s)
        and _rtsp_url_has_credentials(fallback)
        and _same_rtsp_endpoint(s, fallback)
    )
    resolved = fallback if use_fallback else s
    return _rewrite_rtsp_localhost_for_container(resolved)


def _rewrite_rtsp_localhost_for_container(url: str) -> str:
    """
    In Docker, localhost points to the container itself.
    Rewrite rtsp://localhost / 127.0.0.1 / ::1 to host.docker.internal.
    """
    raw = (url or "").strip()
    if not raw.lower().startswith("rtsp://"):
        return raw
    # Only rewrite when inside a containerized runtime.
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


def _find_vehicle_route(db: Session, plate_text: str) -> dict[str, str] | None:
    """Return {route_number, route_name, driver_name} for a registered plate, or None."""
    norm = plate_text.strip().upper()
    row = (
        db.query(AppVehicle)
        .filter(AppVehicle.plate_number == norm, AppVehicle.is_active.is_(True))
        .first()
    )
    if not row:
        return None
    return {
        "route_number": row.route_number or "",
        "route_name": row.route_name or "",
        "driver_name": row.driver_name or "",
    }


def _find_camera_url(db: Session, camera_id: str) -> str | None:
    row = db.query(AppCamera).filter(AppCamera.id == camera_id).first()
    if not row:
        return None
    return _resolve_stream_url(row.stream_url)


def _make_overlay_frame(w: int, h: int, message: str, *, dark: bool = True) -> bytes:
    """Return MJPEG-ready bytes for a status overlay frame (reconnecting / error)."""
    import cv2
    import numpy as np

    bg = (14, 16, 20) if dark else (30, 30, 30)
    img = np.full((h, w, 3), bg, dtype=np.uint8)
    # Subtle gradient strip at top
    img[:4, :] = (50, 55, 65)
    cv2.putText(
        img, message, (16, h // 2 + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (140, 150, 165), 1, cv2.LINE_AA,
    )
    cv2.putText(
        img, "TSRS Bus Monitor", (16, h - 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60, 65, 75), 1, cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return b""
    return _BOUNDARY + b"\r\n" + _FRAME_HEADER + buf.tobytes() + b"\r\n"


async def _mjpeg_generator(stream_url: str, camera_id: str) -> AsyncGenerator[bytes, None]:
    """
    Persistent MJPEG stream generator — never exits on stream errors.

    Architecture:
    - Outer loop: reconnect with exponential backoff. Runs forever until client disconnects.
    - Inner loop: drain RTSP buffer with cheap grab() calls, retrieve one decoded frame.
    - Keepalive: sends a status frame every _MJPEG_KEEPALIVE_SEC so the browser/nginx
      connection never idles out during camera hiccups or reconnect delays.
    - Frame drain: cap.grab() discards compressed packets without decoding pixels (fast).
      Only cap.retrieve() decodes — called once per pipeline cycle.
    """
    import os
    import time

    import cv2

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _ffmpeg_opts_low_latency()
    loop = asyncio.get_running_loop()

    def _open_cap() -> Any:
        c = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        try:
            c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return c

    def _grab_and_retrieve(c: Any) -> tuple[bool, Any]:
        """Drain the RTSP jitter buffer with cheap grab() calls, then decode one frame."""
        for _ in range(_MJPEG_DRAIN_GRABS):
            if not c.grab():
                return False, None
        return c.retrieve()

    def _encode(frame: Any) -> bytes | None:
        h, w = frame.shape[:2]
        if w > _MJPEG_MAX_WIDTH:
            scale = _MJPEG_MAX_WIDTH / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, _MJPEG_JPEG_QUALITY])
        if not ok:
            return None
        return _BOUNDARY + b"\r\n" + _FRAME_HEADER + buf.tobytes() + b"\r\n"

    _stream_health[camera_id] = {"connected": False, "fps": 0.0, "last_frame": None}
    frame_times: deque[float] = deque(maxlen=40)
    reconnect_delay = 1.0
    last_yield_time = time.time()

    try:
        while True:  # ── outer reconnect loop ─────────────────────────────────────────
            # Send a connecting/reconnecting overlay immediately so the browser doesn't go blank
            overlay = _make_overlay_frame(640, 200, "Connecting to camera…")
            if overlay:
                yield overlay
                last_yield_time = time.time()

            cap = await loop.run_in_executor(None, _open_cap)

            if not cap.isOpened():
                _stream_health[camera_id] = {"connected": False, "fps": 0.0, "last_frame": None}
                # Exponential backoff: 1s, 2s, 4s, max 8s
                reconnect_delay = min(reconnect_delay * 2, 8.0)

                # Keep sending keepalive frames during the wait so the connection stays open
                deadline = time.time() + reconnect_delay
                while time.time() < deadline:
                    elapsed_since_yield = time.time() - last_yield_time
                    if elapsed_since_yield >= _MJPEG_KEEPALIVE_SEC:
                        msg = f"Reconnecting… (retry in {int(deadline - time.time() + 0.5)}s)"
                        ka = _make_overlay_frame(640, 200, msg)
                        if ka:
                            yield ka
                            last_yield_time = time.time()
                    await asyncio.sleep(0.2)
                continue

            # Connected successfully
            reconnect_delay = 1.0
            _stream_health[camera_id] = {"connected": True, "fps": 0.0, "last_frame": time.time()}
            consecutive_failures = 0

            try:
                while True:  # ── inner frame loop ──────────────────────────────────────
                    t0 = time.time()

                    ok, frame = await loop.run_in_executor(None, _grab_and_retrieve, cap)

                    if not ok or frame is None:
                        consecutive_failures += 1
                        # Send keepalive if we're stalling so connection doesn't idle out
                        if time.time() - last_yield_time >= _MJPEG_KEEPALIVE_SEC:
                            ka = _make_overlay_frame(640, 200, "Stream stalled, buffering…")
                            if ka:
                                yield ka
                                last_yield_time = time.time()
                        if consecutive_failures >= _MJPEG_FAIL_THRESHOLD:
                            break  # Exit inner loop → reconnect
                        await asyncio.sleep(0.08)
                        continue

                    consecutive_failures = 0
                    jpg_bytes = await loop.run_in_executor(None, _encode, frame)
                    if jpg_bytes:
                        yield jpg_bytes
                        last_yield_time = time.time()

                    t1 = time.time()
                    frame_times.append(t1 - t0)
                    elapsed_sum = sum(frame_times)
                    _stream_health[camera_id]["fps"] = (
                        round(len(frame_times) / elapsed_sum, 1) if elapsed_sum > 0 else 0.0
                    )
                    _stream_health[camera_id]["last_frame"] = t1

                    # Target FPS pacing
                    sleep = max(0.0, (1.0 / _MJPEG_TARGET_FPS) - (t1 - t0))
                    if sleep > 0:
                        await asyncio.sleep(sleep)

            finally:
                cap.release()
                _stream_health[camera_id]["connected"] = False

            # Brief pause before reconnect attempt
            await asyncio.sleep(0.5)

    except asyncio.CancelledError:
        pass
    finally:
        _stream_health[camera_id] = {"connected": False, "fps": 0.0, "last_frame": None}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.get("/cameras/{camera_id}/stream")
async def stream_camera(
    camera_id: str,
    token: str = Query(..., description="JWT auth token"),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """
    MJPEG stream proxy for a configured camera's RTSP stream.
    Browsers display this via <img src='/api/cameras/{id}/stream?token=...' />.
    """
    if _decode_ws_token(token) is None:
        raise HTTPException(status_code=401, detail="Unauthorized")

    stream_url = _find_camera_url(db, camera_id)
    if stream_url is None:
        raise HTTPException(status_code=404, detail="Camera not found")

    if not stream_url or not stream_url.startswith("rtsp"):
        raise HTTPException(
            status_code=400,
            detail=f"Camera '{camera_id}' has no valid RTSP URL configured (got: {stream_url!r}). "
                   "Set a real rtsp:// URL via PATCH /api/cameras/{id}.",
        )

    return StreamingResponse(
        _mjpeg_generator(stream_url, camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.get("/cameras/{camera_id}/stream-health")
def stream_health(
    camera_id: str,
    _user: str = Depends(get_current_username),
) -> dict[str, Any]:
    """Return current MJPEG stream health for a camera."""
    return _stream_health.get(camera_id, {"connected": False, "fps": 0.0, "last_frame": None})


@router.post("/live/detections")
async def ingest_detection(
    body: DetectionIn,
    db: Session = Depends(get_db),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, str]:
    """Vision worker calls this to ingest a plate detection."""
    settings = get_api_settings()
    if not x_internal_token or x_internal_token != settings.INTERNAL_INGEST_SECRET:
        raise HTTPException(status_code=403, detail="Invalid internal token")
    event_id = uuid.uuid4().hex[:16]
    ts_dt = datetime.now(timezone.utc)
    ts = ts_dt.isoformat()
    plate_clean = body.plate_text.strip().upper()
    detected_route = (body.detected_route or "").strip().upper() or None
    vehicle_info = _find_vehicle_route(db, plate_clean)
    registered_route = vehicle_info["route_number"].strip().upper() if vehicle_info else ""

    # Mismatch conditions:
    #   1. Bus shows a route that conflicts with its DB registration
    #   2. Bus shows a route but its plate is not registered at all (unknown bus on a route)
    is_mismatch = bool(
        detected_route and (
            (registered_route and detected_route != registered_route)
            or (not registered_route)  # unknown plate displaying a school route
        )
    )

    row = {
        "id": event_id,
        "type": "plate",
        "plate_text": plate_clean,
        "confidence": body.confidence,
        "camera_id": body.camera_id,
        "camera_name": body.camera_name,
        "snapshot_base64": body.snapshot_base64,
        "detected_at": ts,
        "route_number": registered_route,
        "route_name": vehicle_info["route_name"] if vehicle_info else "",
        "driver_name": vehicle_info["driver_name"] if vehicle_info else "",
        "is_registered": vehicle_info is not None,
        "detected_route": detected_route,
        "is_mismatch": is_mismatch,
    }
    try:
        process_detection_for_trips(
            db,
            camera_id=body.camera_id,
            camera_name=body.camera_name,
            plate_number=plate_clean,
            confidence=body.confidence,
            snapshot_base64=body.snapshot_base64,
            detected_at=ts_dt,
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    _recent.appendleft(row)
    await manager.broadcast({"type": "detection", "payload": row})
    await _push_debug_event(
        "plate_ingest",
        {
            "plate_text": row["plate_text"],
            "confidence": row["confidence"],
            "camera_id": row["camera_id"],
            "camera_name": row["camera_name"],
            "has_snapshot": bool(body.snapshot_base64),
            "event_id": event_id,
        },
        source="api",
    )
    return {"status": "ok", "id": event_id}


@router.post("/live/debug")
async def ingest_debug(
    body: DebugIn,
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, str]:
    """Internal: vision worker pushes per-frame / pipeline debug (same secret as ingest)."""
    settings = get_api_settings()
    if not x_internal_token or x_internal_token != settings.INTERNAL_INGEST_SECRET:
        raise HTTPException(status_code=403, detail="Invalid internal token")
    await _push_debug_event(
        body.message,
        dict(body.detail or {}),
        source=body.source.strip()[:64] or "vision-worker",
    )
    return {"status": "ok"}


@router.get("/live/debug")
def list_debug_logs(
    _user: str = Depends(get_current_username),
    limit: int = Query(120, ge=1, le=300),
) -> list[dict[str, Any]]:
    """Recent debug events (newest first) for the Live Monitor panel."""
    return list(_debug_logs)[:limit]


@router.get("/live/recent")
def list_recent_detections(
    _user: str = Depends(get_current_username),
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Return recent plate detections (most recent first)."""
    return list(_recent)[:limit]


@router.post("/live/detections/{event_id}/adjust")
async def adjust_detection(
    event_id: str,
    body: AdjustIn,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_username),
) -> dict[str, Any]:
    """
    Staff acknowledge a bus mismatch and record the reason.
    Updates the in-memory detection record so all connected dashboards see it immediately.
    """
    valid_types = {"temporary", "permanent", "glitch", "other"}
    if body.swap_type not in valid_types:
        raise HTTPException(status_code=422, detail=f"swap_type must be one of {valid_types}")

    # Find the detection in memory and update it
    target: dict[str, Any] | None = None
    for rec in _recent:
        if rec.get("id") == event_id:
            target = rec
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Detection not found (may have expired)")

    # Persist to DB
    try:
        swap = AppBusSwap(
            id=uuid.uuid4().hex[:16],
            event_id=event_id,
            plate_number=str(target.get("plate_text", "")),
            registered_route=str(target.get("route_number", "")),
            detected_route=str(target.get("detected_route", "")),
            swap_type=body.swap_type,
            notes=body.notes,
            resolved_by=username,
            created_at=datetime.now(timezone.utc),
        )
        db.add(swap)
        db.commit()
    except Exception:
        db.rollback()
        # Non-fatal — still update memory and broadcast

    # Mutate in-memory record so broadcast reflects resolved state
    target["swap_type"] = body.swap_type
    target["swap_notes"] = body.notes
    target["swap_resolved"] = True
    target["swap_resolved_by"] = username

    # Broadcast update to all connected dashboards
    await manager.broadcast({
        "type": "detection_adjusted",
        "event_id": event_id,
        "swap_type": body.swap_type,
        "swap_notes": body.notes,
        "swap_resolved": True,
        "swap_resolved_by": username,
    })

    return {"status": "ok", "event_id": event_id}


_WS_PING_INTERVAL = 20.0  # seconds between server→client ping frames


@router.websocket("/ws/live")
async def websocket_live(
    websocket: WebSocket,
    token: str = Query(..., description="JWT from login"),
) -> None:
    """WebSocket for real-time plate detection events with ping/pong keepalive."""
    if _decode_ws_token(token) is None:
        await websocket.close(code=4401)
        return
    await manager.connect(websocket)
    try:
        # Send initial snapshot so the UI populates immediately on connect
        await websocket.send_json(
            {
                "type": "snapshot",
                "recent": list(_recent)[:50],
                "debug": list(_debug_logs)[:60],
            },
        )

        # Keep connection alive with periodic ping; receive any client messages.
        # asyncio.wait_for raises TimeoutError after _WS_PING_INTERVAL if no message
        # arrives — we then send a ping frame and loop. This prevents idle TCP drops
        # from nginx / load balancers (typically 60s timeout, we ping at 20s).
        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=_WS_PING_INTERVAL,
                )
                # Client can send {"type":"pong"} in response to our ping
                if data.strip() == '{"type":"pong"}' or data.strip() == "pong":
                    continue
            except asyncio.TimeoutError:
                # Send a lightweight ping so the client knows we're alive
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break  # client gone
    except WebSocketDisconnect:
        pass
    finally:
        manager.disconnect(websocket)
