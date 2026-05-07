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
    """
    Vision worker → API ingest payload.

    `plate_text` is OPTIONAL: when the bus is too far / too blurry for plate OCR
    but the route placard ("AR-29") is readable, the worker can post just the
    route. The API then looks up the plate from the vehicle registry and flags
    the row with `plate_from_storage=true` so the dashboard shows a yellow
    triangle next to the auto-filled plate.

    Symmetrically, `detected_route` is optional: when only a plate is read,
    we still post — the dashboard simply shows the registered route from the
    vehicle table (if the plate is registered).
    """
    plate_text: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    camera_id: str = "default"
    camera_name: str = "Camera"
    snapshot_base64: str | None = None
    detected_route: str | None = None   # route OCR'd from bus placard / LED display


class ManualEntryIn(BaseModel):
    """
    Manual log entry from the dashboard.
    Provide either `plate_text` OR `route_number` (or both):
      - plate_text only       → registered route is auto-filled from vehicle registry
      - route_number only     → plate is auto-filled from vehicle registry
                                 (if the route maps to multiple vehicles, the first
                                  active one is used and `plate_from_storage=true`).
    """
    plate_text: str | None = None
    route_number: str | None = None
    camera_id: str = "manual"
    camera_name: str = "Manual entry"
    notes: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class AdjustIn(BaseModel):
    swap_type: str   # "temporary" | "permanent" | "glitch" | "other"
    notes: str | None = None


class EditDetectionIn(BaseModel):
    """Operator-driven correction of a recent detection (pencil icon on the dashboard)."""
    plate_text: str | None = None
    detected_route: str | None = None
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
# Bumped 75 -> 88. At 75 the JPEG encoder smudges 3-4 px wide plate
# strokes which kills downstream OCR debugging; the user explicitly
# noted "the api -> mjpeg is somehow ruining the quality of the image".
_MJPEG_JPEG_QUALITY = 88
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


def _normalize_route_label(raw: str) -> str:
    """
    Accept any of: 'AR-29', 'ar 29', '29', 'AR29', 'AR-7' → return canonical 'AR-29'.
    Returns original (uppercased, stripped) if not parseable.
    """
    if not raw:
        return ""
    cleaned = raw.strip().upper().replace(" ", "").replace("_", "")
    # already canonical AR-NN
    if cleaned.startswith("AR-") and cleaned[3:].isdigit():
        return f"AR-{int(cleaned[3:]):02d}"
    if cleaned.startswith("AR") and cleaned[2:].isdigit():
        return f"AR-{int(cleaned[2:]):02d}"
    if cleaned.isdigit():
        return f"AR-{int(cleaned):02d}"
    return cleaned


def _find_vehicle_by_route(db: Session, route_number: str) -> AppVehicle | None:
    """
    Look up an active vehicle by route number. The vehicle table can have multiple
    plates per route (relief buses); prefer an exact, active match and fall back
    to a case-insensitive match on the canonicalised route label.
    """
    if not route_number:
        return None
    canon = _normalize_route_label(route_number)
    candidates = [route_number.strip().upper()]
    if canon and canon not in candidates:
        candidates.append(canon)
    # Strip 'AR-' prefix as another fallback (some users register routes as just '29')
    if canon.startswith("AR-"):
        bare = canon[3:].lstrip("0") or canon[3:]
        if bare not in candidates:
            candidates.append(bare)
    q = db.query(AppVehicle).filter(
        AppVehicle.is_active.is_(True),
        AppVehicle.route_number.in_(candidates),
    )
    return q.order_by(AppVehicle.created_at.asc()).first()


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


def _build_detection_row(
    db: Session,
    *,
    event_id: str,
    plate_text: str | None,
    confidence: float,
    camera_id: str,
    camera_name: str,
    snapshot_base64: str | None,
    detected_route: str | None,
    detected_at: datetime,
    source: str = "vision",
    notes: str | None = None,
) -> dict[str, Any]:
    """
    Compose the in-memory detection row. Handles all four input cases:
      1. plate + route   → standard read
      2. plate only       → route filled from registry (if registered)
      3. route only       → plate auto-filled from registry → plate_from_storage=true
      4. neither          → caller should reject (we still emit a debug-only row)
    """
    plate_clean = (plate_text or "").strip().upper() or None
    detected_route_norm = _normalize_route_label(detected_route or "") or None

    # Distinguish "OCR actually read this plate" (`plate_clean` set by the
    # caller) from "we know the route placard, so the registry can SUGGEST
    # what plate it should be" (`suggested_plate` derived below). The user
    # explicitly does not want a registry suggestion to masquerade as a real
    # OCR read — `plate_text` stays empty in that case and the dashboard
    # surfaces the suggestion in a separate field.
    suggested_plate: str | None = None
    plate_from_storage = False
    vehicle: AppVehicle | None = None

    if plate_clean:
        # Look up by plate first (paths 1 + 2 — real OCR read)
        vehicle = (
            db.query(AppVehicle)
            .filter(AppVehicle.plate_number == plate_clean, AppVehicle.is_active.is_(True))
            .first()
        )
    elif detected_route_norm:
        # Path 3: route placard visible but plate not readable. Look up the
        # registry to surface a suggestion, but DO NOT promote it into
        # plate_text — the camera did not actually see those characters.
        vehicle = _find_vehicle_by_route(db, detected_route_norm)
        if vehicle is not None:
            suggested_plate = vehicle.plate_number
            plate_from_storage = True  # still set so the dashboard knows this
                                       # row is "route-only with a suggestion"

    registered_route = ((vehicle.route_number or "").strip().upper()) if vehicle else ""
    route_name = (vehicle.route_name or "") if vehicle else ""
    driver_name = (vehicle.driver_name or "") if vehicle else ""
    is_registered = vehicle is not None

    # Mismatch logic (only meaningful when *both* sides are known and the
    # plate was actually OCR'd, not just suggested from storage).
    is_mismatch = bool(
        not plate_from_storage
        and detected_route_norm
        and (
            (registered_route and detected_route_norm != registered_route)
            or (not registered_route)
        )
    )

    return {
        "id": event_id,
        "type": "plate",
        # plate_text now ONLY contains plates the OCR actually read.
        # Routes-with-suggestion rows leave this empty and put the registry
        # plate into `suggested_plate` instead.
        "plate_text": plate_clean or "",
        "suggested_plate": suggested_plate or "",
        "confidence": confidence,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "snapshot_base64": snapshot_base64,
        "detected_at": detected_at.isoformat(),
        "route_number": registered_route,
        "route_name": route_name,
        "driver_name": driver_name,
        "is_registered": is_registered,
        "detected_route": detected_route_norm,
        "is_mismatch": is_mismatch,
        "plate_from_storage": plate_from_storage,
        "has_plate": bool(plate_clean),
        "has_route": bool(detected_route_norm),
        "source": source,
        "notes": notes,
    }


@router.post("/live/detections")
async def ingest_detection(
    body: DetectionIn,
    db: Session = Depends(get_db),
    x_internal_token: str | None = Header(default=None, alias="X-Internal-Token"),
) -> dict[str, str]:
    """Vision worker calls this to ingest a plate detection (or a route-only sighting)."""
    settings = get_api_settings()
    if not x_internal_token or x_internal_token != settings.INTERNAL_INGEST_SECRET:
        raise HTTPException(status_code=403, detail="Invalid internal token")

    plate_in = (body.plate_text or "").strip().upper() or None
    route_in = (body.detected_route or "").strip().upper() or None
    if not plate_in and not route_in:
        raise HTTPException(status_code=422, detail="Either plate_text or detected_route is required")

    event_id = uuid.uuid4().hex[:16]
    ts_dt = datetime.now(timezone.utc)

    row = _build_detection_row(
        db,
        event_id=event_id,
        plate_text=plate_in,
        confidence=body.confidence,
        camera_id=body.camera_id,
        camera_name=body.camera_name,
        snapshot_base64=body.snapshot_base64,
        detected_route=route_in,
        detected_at=ts_dt,
        source="vision",
    )

    plate_clean = row["plate_text"]
    try:
        # Only run the trip pipeline when we have a real plate to anchor on
        # (route-only sightings without a known vehicle don't represent a gate event).
        if plate_clean:
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
            "plate_from_storage": row["plate_from_storage"],
            "detected_route": row["detected_route"],
        },
        source="api",
    )
    return {"status": "ok", "id": event_id}


@router.post("/live/manual-detection")
async def manual_detection(
    body: ManualEntryIn,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_username),
) -> dict[str, Any]:
    """
    Operator-driven manual log entry.

    Use cases:
      - Camera missed a bus that just rolled in → enter the route number,
        we look up the registered plate and post a detection row.
      - Bus has temporary plates → enter just the plate, route is auto-filled
        from the registry if the plate is recognised.
    """
    plate_in = (body.plate_text or "").strip().upper() or None
    route_in = _normalize_route_label(body.route_number or "") or None
    if not plate_in and not route_in:
        raise HTTPException(
            status_code=422,
            detail="Provide at least one of plate_text or route_number",
        )

    event_id = uuid.uuid4().hex[:16]
    ts_dt = datetime.now(timezone.utc)

    row = _build_detection_row(
        db,
        event_id=event_id,
        plate_text=plate_in,
        confidence=body.confidence,
        camera_id=body.camera_id or "manual",
        camera_name=body.camera_name or f"Manual entry · {username}",
        snapshot_base64=None,
        detected_route=route_in,
        detected_at=ts_dt,
        source="manual",
        notes=body.notes,
    )

    if not row["plate_text"] and route_in:
        # Route given but no vehicle is registered for it — keep the row but flag clearly.
        row["plate_text"] = ""
        row["plate_from_storage"] = False

    plate_clean = row["plate_text"]
    try:
        if plate_clean:
            process_detection_for_trips(
                db,
                camera_id=row["camera_id"],
                camera_name=row["camera_name"],
                plate_number=plate_clean,
                confidence=body.confidence,
                snapshot_base64=None,
                detected_at=ts_dt,
            )
            db.commit()
    except Exception:
        db.rollback()
        raise

    _recent.appendleft(row)
    await manager.broadcast({"type": "detection", "payload": row})
    await _push_debug_event(
        "manual_entry",
        {
            "plate_text": row["plate_text"],
            "detected_route": row["detected_route"],
            "by": username,
            "plate_from_storage": row["plate_from_storage"],
            "notes": body.notes,
        },
        source="api",
    )
    return {"status": "ok", "id": event_id, "row": row}


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


@router.patch("/live/detections/{event_id}")
async def edit_detection(
    event_id: str,
    body: EditDetectionIn,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_username),
) -> dict[str, Any]:
    """
    Edit a detection in-place: correct the plate text, the detected route,
    or attach operator notes. Updates the in-memory record and re-resolves
    the registered route from the vehicle registry.
    """
    target: dict[str, Any] | None = None
    for rec in _recent:
        if rec.get("id") == event_id:
            target = rec
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Detection not found (may have expired from buffer)")

    new_plate = (body.plate_text or "").strip().upper() or None
    new_route = _normalize_route_label(body.detected_route or "") or None
    if new_plate is not None:
        target["plate_text"] = new_plate
        target["has_plate"] = True
        # Re-resolve the registry mapping for the corrected plate.
        veh = (
            db.query(AppVehicle)
            .filter(AppVehicle.plate_number == new_plate, AppVehicle.is_active.is_(True))
            .first()
        )
        target["is_registered"] = veh is not None
        target["route_number"] = (veh.route_number or "").strip().upper() if veh else ""
        target["route_name"] = (veh.route_name or "") if veh else ""
        target["driver_name"] = (veh.driver_name or "") if veh else ""
        target["plate_from_storage"] = False
        target["suggested_plate"] = ""
    if new_route is not None:
        target["detected_route"] = new_route
        target["has_route"] = True
    if body.notes is not None:
        target["notes"] = body.notes.strip() or None

    target["edited_by"] = username
    target["edited_at"] = datetime.now(timezone.utc).isoformat()
    # Mismatch flag may need recomputation now that plate/route may have changed.
    reg = (target.get("route_number") or "").strip().upper()
    det = (target.get("detected_route") or "").strip().upper()
    target["is_mismatch"] = bool(det and reg and det != reg)

    await manager.broadcast({"type": "detection_edited", "event_id": event_id, "row": target})
    return {"status": "ok", "event_id": event_id, "row": target}


# Lower interval = quicker dead-connection detection; client always responds with pong.
_WS_PING_INTERVAL = 12.0  # seconds between server→client ping frames


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
