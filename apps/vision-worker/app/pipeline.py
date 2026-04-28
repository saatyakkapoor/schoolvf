"""RTSP capture + plate OCR + POST to API."""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import cv2
import httpx
import numpy as np

from apps.vision_worker.app.plate_engine import (
    frame_to_jpeg_b64,
    mock_demo_snapshot_b64,
    read_plates_from_frame,
)
from apps.vision_worker.app.settings import VisionSettings, get_settings
from apps.vision_worker.app.webcam_capture import open_webcam

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Route-number OCR — reads the Aravali bus placard / LED display from frame
# ---------------------------------------------------------------------------
# The yellow AR-XX placard sits in the windshield (mid-upper frame).
# The LED destination display shows "29 THE SHRI RAM" (top of bus front).
_PLACARD_RE = re.compile(r"\bAR[-\s]?0*(\d{1,2})\b", re.IGNORECASE)
_LED_RE = re.compile(r"\b(\d{1,2})\s+THE\s+SHRI\b", re.IGNORECASE)
# Partial reads: "AR7", "AR 07", "AR-7" without word boundaries
_PLACARD_LOOSE_RE = re.compile(r"AR[-\s]?0*(\d{1,2})", re.IGNORECASE)
# LED/digital display that shows just a route number: "51", "29", "05" etc.
# Only matched on the placard-zone crop, not full-frame, to avoid false positives
_ROUTE_NUM_RE = re.compile(r"^\s*0*(\d{1,2})\s*$")

# Yellow placard HSV range (Aravali buses use a bright chrome-yellow card)
# Tuned to cover both sunlit and shaded conditions
_YELLOW_HSV_LO = np.array([10, 70, 70],  dtype=np.uint8)
_YELLOW_HSV_HI = np.array([40, 255, 255], dtype=np.uint8)


def _sharpen_roi(roi: "np.ndarray", strength: float = 1.5) -> "np.ndarray":
    """Unsharp mask — improves OCR accuracy on motion-blurred frames."""
    blur = cv2.GaussianBlur(roi, (0, 0), 2)
    return cv2.addWeighted(roi, 1.0 + strength, blur, -strength, 0)


_ROUTE_OCR_MIN_CONF: float = 0.50
"""Minimum RapidOCR confidence for a route candidate to even be considered.
Below this we get spurious hits like 'AR-02' / 'AR-06' / 'AR-51' from
random bus body text. The user wants nothing under 50% to surface."""


def _ocr_scan_for_route(
    ocr_result: list, label: str, *, allow_bare_number: bool = False
) -> tuple[str | None, float]:
    """
    Search one RapidOCR result list for any AR-XX / LED pattern.
    allow_bare_number=True: also accept a lone 1-2 digit number (yellow_mask crop only,
    where spatial context guarantees we're looking at the placard area).
    Returns (route_str, score) where score = bbox_area * ocr_conf so larger
    *and* more confident reads win. Sub-_ROUTE_OCR_MIN_CONF reads are
    rejected outright — the user does not want low-confidence noise.
    """
    best: str | None = None
    best_score: float = 0.0
    for line in ocr_result:
        if len(line) < 2:
            continue
        raw = str(line[1]).strip().upper()
        # RapidOCR returns (bbox, text, conf); EasyOCR returns (bbox, text, conf).
        ocr_conf = 0.0
        if len(line) >= 3:
            try:
                ocr_conf = float(line[2])
            except (TypeError, ValueError):
                ocr_conf = 0.0
        if ocr_conf and ocr_conf < _ROUTE_OCR_MIN_CONF:
            log.debug(
                "route OCR [%s] reject low conf raw='%s' conf=%.2f < %.2f",
                label, raw, ocr_conf, _ROUTE_OCR_MIN_CONF,
            )
            continue
        m = (_PLACARD_RE.search(raw)
             or _LED_RE.search(raw)
             or _PLACARD_LOOSE_RE.search(raw))
        if not m and allow_bare_number:
            m = _ROUTE_NUM_RE.match(raw)
        if not m:
            continue
        route_num = int(m.group(1))
        if route_num < 1 or route_num > 99:
            continue
        route_str = f"AR-{route_num:02d}"
        area = 1.0
        if len(line) >= 1 and hasattr(line[0], "__len__") and len(line[0]) == 4:
            try:
                pts = line[0]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            except Exception:
                area = 1.0
        score = area * max(ocr_conf, 0.5)
        log.info(
            "route OCR [%s] raw='%s' → %s conf=%.2f area=%.0f score=%.0f",
            label, raw, route_str, ocr_conf, area, score,
        )
        if score > best_score:
            best_score = score
            best = route_str
    return best, best_score


def _read_route_number(frame: "np.ndarray") -> str | None:
    """
    Route-number reader optimised for yellow AR-XX placards on a moving bus.

    Strategy: ONE fast OCR pass on the yellow placard blob. If that misses,
    we fall back to a SINGLE wide windshield-band crop. The previous
    multi-zone scan was running 4 OCR passes per frame and adding 1-2 s
    of latency for every frame that didn't have a clear placard.
    """
    try:
        from apps.vision_worker.app.plate_engine import _get_ocr
        ocr = _get_ocr()  # RapidOCR — CPU ONNX, thread-safe
        h, w = frame.shape[:2]

        best_route: str | None = None
        best_area: float = 0.0

        # ------------------------------------------------------------------ #
        # Pass 1: yellow colour mask → placard blob → OCR                    #
        # ------------------------------------------------------------------ #
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.inRange(hsv, _YELLOW_HSV_LO, _YELLOW_HSV_HI)

        # Close small gaps (plate lettering breaks the mask), remove tiny speckles
        k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
        k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (5,  3))
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, k_close)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN,  k_open)

        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            best_cnt: tuple[int, int, int, int] | None = None
            best_cnt_score = 0.0
            frame_area = h * w
            for cnt in contours:
                cx, cy_box, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                ar = cw / max(ch, 1)
                # Must be placard-sized: between 0.05% and 8% of frame area.
                # The bus BODY is yellow but fills >20% of frame — exclude it.
                # The placard is a small card inside the windshield.
                area_frac = area / frame_area
                if area_frac < 0.0005 or area_frac > 0.08:
                    continue
                if ar < 1.2 or ar > 10.0:
                    continue
                # Skip OSD strips and bottom of frame (plate zone — not where placard is)
                centre_y = cy_box + ch / 2
                if centre_y < h * 0.08 or centre_y > h * 0.75:
                    continue
                # Score: prefer larger area (up to the 8% cap) and placard aspect ratio
                aspect_score = 1.0 - min(abs(ar - 3.5) / 3.5, 1.0)
                score = area_frac * 10 + aspect_score
                if score > best_cnt_score:
                    best_cnt_score = score
                    best_cnt = (cx, cy_box, cw, ch)

            if best_cnt is not None:
                bx, by, bw, bh = best_cnt
                # Add generous padding so letters at the edge aren't clipped
                pad_x = max(6, int(bw * 0.10))
                pad_y = max(4, int(bh * 0.20))
                x1 = max(0, bx - pad_x)
                y1 = max(0, by - pad_y)
                x2 = min(w, bx + bw + pad_x)
                y2 = min(h, by + bh + pad_y)
                roi = frame[y1:y2, x1:x2]
                # 3× upscale (LANCZOS) + strong unsharp mask for motion frames
                roi = cv2.resize(roi, (roi.shape[1] * 3, roi.shape[0] * 3),
                                 interpolation=cv2.INTER_LANCZOS4)
                roi = _sharpen_roi(roi, strength=1.8)
                result, _ = ocr(roi)
                if result:
                    # allow_bare_number=True: spatial context confirms this is the placard
                    r, a = _ocr_scan_for_route(result, "yellow_mask", allow_bare_number=True)
                    if r and a > best_area:
                        best_area = a
                        best_route = r
                    # Always log so we can tune the regex
                    texts = [(str(l[1]).strip(), round(float(l[2]), 3))
                             for l in result if len(l) >= 3]
                    log.info("route yellow_mask OCR: match=%s texts=%s", r, texts[:10])

        # ------------------------------------------------------------------ #
        # Pass 2: ONE windshield-band crop (only if yellow mask missed)      #
        # ------------------------------------------------------------------ #
        if not best_route:
            top = int(h * 0.10)
            bot = int(h * 0.55)
            roi = frame[top:bot, :].copy()
            if roi.size > 0:
                roi = cv2.resize(roi, (roi.shape[1] * 2, roi.shape[0] * 2),
                                 interpolation=cv2.INTER_LINEAR)
                roi = _sharpen_roi(roi, strength=1.2)
                result, _ = ocr(roi)
                if result:
                    r, a = _ocr_scan_for_route(result, "windshield")
                    if r and a > best_area:
                        best_area = a
                        best_route = r

        if best_route:
            log.info("route OCR winner: %s  area=%.0f", best_route, best_area)
        return best_route
    except Exception as exc:
        log.warning("route OCR exception: %s", exc)
    return None

log = logging.getLogger("vision.pipeline")

_RTSP_OPTS = "rtsp_transport;tcp|fflags|nobuffer|flags|low_delay|max_delay|0"


def _frame_sharpness(frame: "np.ndarray") -> float:
    """Laplacian variance — higher = sharper / less motion blur.

    Computed on a downsampled copy (320 wide) so picking the sharpest of
    4 candidates costs ~3 ms total instead of ~80 ms on full HD.
    """
    h, w = frame.shape[:2]
    if w > 320:
        sc = 320 / w
        frame = cv2.resize(frame, (320, int(h * sc)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


class _LatestFrameBuffer:
    """Background thread: pulls frames from `cap` as fast as the camera produces
    them and keeps only the most recent `history` decoded frames.

    Why this exists: when OCR takes longer than 1/fps, the underlying ffmpeg
    queue piles up and `cap.read()` returns 2-5 second-old footage. Solution
    is the standard one used by every real-time vision app — a dedicated
    grabber thread that drops everything older than the most recent frame,
    so the OCR loop always sees *now*.

    The buffer also holds a tiny ring of the latest frames (default 4 ≈ 130ms
    at 30fps) so we can pick the sharpest of those for OCR — defence against
    motion blur, but only across genuinely current frames.
    """

    def __init__(self, cap: Any, *, history: int = 4) -> None:
        self._cap = cap
        self._history = max(1, int(history))
        self._frames: list[tuple[float, Any]] = []  # oldest → newest
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._consec_grab_fails = 0
        self._thread = threading.Thread(
            target=self._run, name="rtsp-grabber", daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                ok = self._cap.grab()
            except Exception as e:
                log.debug("grabber: cap.grab() raised %s", e)
                ok = False
            if not ok:
                self._consec_grab_fails += 1
                if self._consec_grab_fails > 50:
                    # Surface persistent grab failures so the outer loop reconnects
                    log.warning("grabber: %d consecutive grab failures", self._consec_grab_fails)
                    self._consec_grab_fails = 0
                # Tiny sleep so we don't burn CPU when the stream is dead
                time.sleep(0.02)
                continue
            self._consec_grab_fails = 0
            try:
                ok2, frame = self._cap.retrieve()
            except Exception:
                continue
            if not ok2 or frame is None:
                continue
            with self._lock:
                self._frames.append((time.time(), frame))
                if len(self._frames) > self._history:
                    # Drop the oldest; we only ever care about the latest
                    self._frames = self._frames[-self._history:]

    def latest_sharpest(self, *, max_age_sec: float = 1.5) -> tuple[bool, Any, float]:
        """Return (ok, frame, captured_at_ts).

        Picks the sharpest frame from the in-memory ring (which only ever
        contains the most recent ~history frames). Anything older than
        `max_age_sec` is discarded so a stalled stream can't return ancient
        footage.
        """
        with self._lock:
            snapshot = list(self._frames)
        if not snapshot:
            return False, None, 0.0
        now = time.time()
        fresh = [(ts, f) for ts, f in snapshot if (now - ts) <= max_age_sec]
        if not fresh:
            # Stream is producing frames but they're all old → caller should
            # treat this as a failure and reconnect.
            return False, None, snapshot[-1][0]
        best_frame = None
        best_score = -1.0
        best_ts = 0.0
        for ts, frame in fresh:
            score = _frame_sharpness(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
                best_ts = ts
        return True, best_frame, best_ts

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
_LAST_LIVE_DEBUG_POST = 0.0
_LIVE_DEBUG_INTERVAL_SEC = 3.0

# Thread pool for non-blocking HTTP POSTs (detection ingest + debug); sized from settings on first use
_http_pool: ThreadPoolExecutor | None = None
# Reusable httpx client with connection pooling
_http_client: httpx.Client | None = None

# Persistent OCR pool — running plate OCR (GPU) and route OCR (CPU) in parallel.
# Created once and reused across every frame to avoid spawning two OS threads
# per frame at ~30 fps. Max-workers=2 because we always have exactly 2 jobs:
# plate OCR + route OCR. Per-camera loops share this pool safely; the underlying
# OCR singletons own their own thread-safety contracts (EasyOCR/RapidOCR).
_ocr_pool: ThreadPoolExecutor | None = None


def _get_ocr_pool() -> ThreadPoolExecutor:
    """Persistent OCR worker pool. Sized for the double-buffered pipeline:
    each cycle uses 2 slots (plate + route OCR), and we keep up to 2 cycles
    in flight (current consuming results, next prefetched). 6 leaves headroom
    for an edge case where the current cycle is still finishing as the next
    one starts."""
    global _ocr_pool
    if _ocr_pool is None:
        _ocr_pool = ThreadPoolExecutor(max_workers=6, thread_name_prefix="ocr")
    return _ocr_pool


def _http_pool_submit(s: VisionSettings, fn, *args, **kwargs) -> None:
    global _http_pool
    if _http_pool is None:
        w = max(1, int(s.HTTP_POST_WORKERS))
        _http_pool = ThreadPoolExecutor(max_workers=w, thread_name_prefix="http-post")
    _http_pool.submit(fn, *args, **kwargs)


def _configure_compute_threads() -> None:
    """Use all allocated CPU: PyTorch + OpenCV thread pools from env (set in compose)."""
    import os

    try:
        import cv2

        oc = (os.environ.get("OPENCV_NUM_THREADS") or "").strip()
        if oc.isdigit() and int(oc) > 0:
            cv2.setNumThreads(int(oc))
    except Exception:
        pass
    try:
        import torch

        tn = (os.environ.get("TORCH_NUM_THREADS") or "").strip()
        if tn.isdigit() and int(tn) > 0:
            t = int(tn)
            torch.set_num_threads(t)
            torch.set_num_interop_threads(max(1, min(4, t // 4)))
    except Exception:
        pass


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None:
        _http_client = httpx.Client(
            timeout=10.0,
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )
    return _http_client


def _grab_latest_frame(cap: Any, *, drain: int) -> tuple[bool, Any]:
    """Use grab() to skip buffered frames (much faster than read()), then retrieve only the last."""
    n = max(1, int(drain))
    for _ in range(n):
        if not cap.grab():
            return False, None
    ok, frame = cap.retrieve()
    if not ok or frame is None:
        return False, None
    return ok, frame


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


def _dedupe_allow(plate: str, last: dict[str, float], window: float,
                  route: str | None = None) -> bool:
    """Allow if (plate, route) combo hasn't been seen within window seconds."""
    now = time.time()
    key = f"{plate}|{route or ''}"
    prev = last.get(key)
    if prev is not None and now - prev < window:
        return False
    last[key] = now
    return True


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein distance — O(len(a)*len(b))."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1,
                            prev[j - 1] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


def _camera_cooldown_allow(
    plate: str,
    camera_id: str,
    recent: dict[str, tuple[float, str]],   # cam_id → (timestamp, plate_text)
    window: float,
    similarity_threshold: int = 3,
) -> bool:
    """
    Allow this plate to post if:
      a) No plate has been posted from this camera in the last `window` seconds, OR
      b) A plate WAS posted recently but the new text is edit-distance > threshold away
         from it (= genuinely different vehicle, not OCR garble of the same plate).

    Example: "NR68N15" then "IR98N262" → edit dist=5 → same car misread → BLOCK
             "NR68N15" then "DL5SAB1234" → edit dist=8 → new car → ALLOW
    """
    now = time.time()
    entry = recent.get(camera_id)
    if entry is None:
        return True
    ts, last_plate = entry
    age = now - ts
    if age >= window:
        return True  # window expired — always allow
    # Within window: allow only if text is sufficiently different
    dist = _edit_distance(plate, last_plate)
    if dist <= similarity_threshold:
        log.info("camera %s suppressed '%s' (edit_dist=%d from '%s', age=%.1fs)",
                 camera_id, plate, dist, last_plate, age)
        return False
    return True  # different vehicle


def _camera_cooldown_mark(
    plate: str,
    camera_id: str,
    recent: dict[str, tuple[float, str]],
) -> None:
    recent[camera_id] = (time.time(), plate)


def _route_only_dedupe_allow(
    route: str,
    camera_id: str,
    last: dict[str, float],
    window: float,
) -> bool:
    """
    Allow posting a route-only sighting (no plate text read) at most once per
    `window` seconds per (camera, route). Prevents flooding the dashboard
    when a bus sits in front of the camera and the placard reads cleanly
    every frame but the plate is too blurry to OCR.
    """
    key = f"route-only::{camera_id}::{route}"
    now = time.time()
    prev = last.get(key)
    if prev is not None and now - prev < window:
        return False
    last[key] = now
    return True


def _post_live_debug_sync(
    api_base: str,
    secret: str,
    message: str,
    detail: dict[str, Any],
) -> None:
    """Actual HTTP POST — runs in thread pool."""
    url = f"{api_base}/api/live/debug"
    try:
        client = _get_http_client()
        r = client.post(
            url,
            json={"message": message, "detail": detail, "source": "vision-worker"},
            headers={"X-Internal-Token": secret},
        )
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        body = (e.response.text or "")[:400]
        log.warning("POST %s HTTP %s: %s", url, e.response.status_code, body)
    except Exception as e:
        log.warning("POST %s failed: %s", url, e)


def _post_live_debug(
    s: VisionSettings,
    message: str,
    detail: dict[str, Any],
    *,
    bypass_interval: bool = False,
) -> None:
    global _LAST_LIVE_DEBUG_POST
    if not getattr(s, "LIVE_DEBUG_PUSH", True):
        return
    now = time.time()
    if not bypass_interval and now - _LAST_LIVE_DEBUG_POST < _LIVE_DEBUG_INTERVAL_SEC:
        return
    if not bypass_interval:
        _LAST_LIVE_DEBUG_POST = now
    # Fire-and-forget in thread pool — don't block the vision loop
    _http_pool_submit(
        s,
        _post_live_debug_sync,
        s.API_BASE_URL.rstrip("/"),
        s.INTERNAL_INGEST_SECRET,
        message,
        detail,
    )


def _announce_worker_online(s: VisionSettings) -> None:
    """One-shot so the Live debug panel proves the worker can reach the API."""
    if not getattr(s, "LIVE_DEBUG_PUSH", True):
        log.info("LIVE_DEBUG_PUSH disabled — skipping /api/live/debug startup ping")
        return
    rtsp = s.CAMERA_RTSP_URL.strip()
    tail = rtsp.split("@")[-1] if "@" in rtsp else rtsp[:120]
    _post_live_debug(
        s,
        "vision_worker_online",
        {
            "camera_id": s.CAMERA_ID,
            "camera_name": s.CAMERA_NAME,
            "api_base_url": s.API_BASE_URL.rstrip("/"),
            "rtsp_target_tail": tail,
            "plate_engine": s.PLATE_ENGINE,
            "vision_stack": s.VISION_STACK,
            "plate_stage": s.PLATE_STAGE,
            "plate_filter": s.PLATE_FILTER,
            "hint": "If Live shows no plates, CAMERA_ID must match a camera id from GET /api/cameras.",
        },
        bypass_interval=True,
    )


def _post_detection_sync(
    api_base: str,
    secret: str,
    plate: str | None,
    confidence: float,
    camera_id: str,
    camera_name: str,
    snapshot_b64: str | None,
    detected_route: str | None = None,
) -> None:
    """Actual HTTP POST — runs in thread pool. plate may be None for route-only sightings."""
    url = f"{api_base}/api/live/detections"
    body: dict[str, Any] = {
        "confidence": confidence,
        "camera_id": camera_id,
        "camera_name": camera_name,
        "snapshot_base64": snapshot_b64,
    }
    if plate:
        body["plate_text"] = plate
    if detected_route:
        body["detected_route"] = detected_route
    try:
        client = _get_http_client()
        r = client.post(url, json=body, headers={"X-Internal-Token": secret})
        r.raise_for_status()
        log.info("Posted plate %s conf=%.2f route=%s", plate, confidence, detected_route or "?")
    except httpx.HTTPStatusError as e:
        log.warning("Ingest HTTP %s: %s", e.response.status_code, (e.response.text or "")[:300])
    except Exception as e:
        log.warning("Ingest failed: %s", e)


def _post_detection(
    s: VisionSettings,
    plate: str | None,
    confidence: float,
    snapshot_b64: str | None,
    *,
    camera_id: str | None = None,
    camera_name: str | None = None,
    detected_route: str | None = None,
) -> None:
    # Fire-and-forget — don't block OCR loop waiting for HTTP response
    _http_pool_submit(
        s,
        _post_detection_sync,
        s.API_BASE_URL.rstrip("/"),
        s.INTERNAL_INGEST_SECRET,
        plate,
        confidence,
        camera_id or s.CAMERA_ID,
        camera_name or s.CAMERA_NAME,
        snapshot_b64,
        detected_route,
    )


def run_mock_loop(s: VisionSettings) -> None:
    last: dict[str, float] = {}
    demo = ["KA01AB1234", "DL03CB9012", "MH12DE3456", "TS09XY1122"]
    log.info(
        "Mock plate engine — synthetic snapshots only; set CAMERA_RTSP_URL for real video + OCR",
    )
    while True:
        time.sleep(4.0)
        plate = random.choice(demo)
        if not _dedupe_allow(plate, last, 5.0):
            continue
        snap = mock_demo_snapshot_b64(plate, s.SNAPSHOT_MAX_WIDTH)
        # Low confidence so it is obvious this is demo data, not a camera read
        _post_detection(s, plate, 0.35, snap or None)


def run_rtsp_loop(
    s: VisionSettings,
    *,
    camera_id: str | None = None,
    camera_name: str | None = None,
    rtsp_url: str | None = None,
    stop_event: threading.Event | None = None,
) -> None:
    import os

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _RTSP_OPTS
    last: dict[str, float] = {}
    cid = (camera_id or s.CAMERA_ID).strip()
    cname = (camera_name or s.CAMERA_NAME).strip()
    url = (rtsp_url or s.CAMERA_RTSP_URL or "").strip()
    url = _rewrite_rtsp_localhost_for_container(url)
    if not url:
        log.error("No RTSP URL for camera %s — exiting thread", cid)
        return

    while True:
        if stop_event is not None and stop_event.is_set():
            log.info("Stop requested for camera %s — exiting RTSP loop", cid)
            return
        log.info("Opening RTSP [%s]: %s", cid, url.split("@")[-1] if "@" in url else url)
        cap = None
        for attempt in range(1, 31):
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if cap.isOpened():
                break
            cap.release()
            cap = None
            log.warning("RTSP open attempt %s/30 failed; retry in 5s", attempt)
            time.sleep(5.0)
        if cap is None or not cap.isOpened():
            log.error(
                "RTSP unavailable after 30 tries — waiting 30s and retrying (no mock fallback)",
            )
            time.sleep(30.0)
            continue

        # Background grabber thread → OCR loop only ever sees fresh frames.
        # This is the *only* defence against the "OCR is showing a 5-second old
        # plate" backlog problem; cv2's BUFFERSIZE=1 + nobuffer flags help but
        # don't fully prevent the queue from filling when OCR is slower than
        # the camera FPS.
        grabber = _LatestFrameBuffer(cap, history=4)
        try:
            frame_n = 0
            last_heartbeat = time.time()
            recent_cam: dict[str, tuple[float, str]] = {}  # per-camera cooldown tracker
            startup_deadline = time.time() + 8.0
            stale_count = 0

            # ── Prefetch / double-buffered OCR pipeline ──────────────────────
            # While we post / jpeg-encode the *current* frame's results, the
            # *next* frame's plate-OCR + route-OCR is already running on the
            # GPU. This is what keeps the T1000 saturated; without it the GPU
            # sits idle for ~150 ms per cycle (HTTP post + snapshot encode).
            # `pending` holds the OCR work submitted for an upcoming frame.
            pending: dict | None = None
            strict_indian = s.PLATE_FILTER.strip().lower() == "indian"
            _pool = _get_ocr_pool()

            def _submit_frame(_f, _captured_at: float) -> dict:
                """Kick off plate + route OCR for one frame, return the futures."""
                _f_copy = _f.copy()
                t0 = time.time()
                return {
                    "frame": _f,
                    "frame_copy": _f_copy,
                    "captured_at": _captured_at,
                    "submitted_at": t0,
                    "plate_fut": _pool.submit(
                        read_plates_from_frame,
                        _f,
                        min_confidence=s.MIN_CONFIDENCE,
                        strict_indian=strict_indian,
                        detection_mode=s.PLATE_DETECTION_MODE.strip().lower(),
                        vision_stack=s.VISION_STACK,
                    ),
                    "route_fut": _pool.submit(_read_route_number, _f_copy),
                }

            while True:
                if stop_event is not None and stop_event.is_set():
                    log.info("Stop requested for camera %s — closing capture", cid)
                    return

                # Step 1: make sure we have an OCR job in flight for the
                # current sharpest frame. If `pending` is None it means we
                # just consumed the previous result; submit a fresh one.
                if pending is None:
                    ok, frame, captured_at = grabber.latest_sharpest(max_age_sec=1.5)
                    if not ok or frame is None:
                        if time.time() > startup_deadline and stale_count > 25:
                            log.warning("Frames went stale (>1.5s old) — reconnecting RTSP")
                            break
                        stale_count += 1
                        time.sleep(0.02)
                        continue
                    stale_count = 0
                    pending = _submit_frame(frame, captured_at)

                # Step 2: wait for the in-flight plate OCR to finish.
                current = pending
                pending = None  # we're consuming it now
                frame = current["frame"]
                frame_copy = current["frame_copy"]
                captured_at = current["captured_at"]
                t_ocr = current["submitted_at"]
                _plate_fut = current["plate_fut"]
                _route_fut = current["route_fut"]

                plates = _plate_fut.result()

                # Step 3: AS SOON AS plate-OCR is done, prefetch the next
                # sharpest frame and submit its OCR. This means the GPU keeps
                # working while we (a) wait for route OCR, (b) jpeg-encode
                # the snapshot, (c) HTTP-post the detection.
                ok_n, frame_n2, captured_at_n = grabber.latest_sharpest(max_age_sec=1.5)
                if ok_n and frame_n2 is not None:
                    pending = _submit_frame(frame_n2, captured_at_n)

                frame_age_ms = int((time.time() - captured_at) * 1000)
                frame_n += 1

                now = time.time()
                if now - last_heartbeat >= 30.0:
                    last_heartbeat = now
                    log.info("camera %s alive — frame #%d", cid, frame_n)

                route: str | None = None
                try:
                    # Tight timeout: route OCR is already running in parallel.
                    # If it's slower than 1s we skip the route for this frame.
                    route = _route_fut.result(timeout=1.0)
                except Exception as _re:
                    log.debug("route OCR timeout/error: %s", _re)
                ocr_ms = int((time.time() - t_ocr) * 1000)
                # ─────────────────────────────────────────────────────────────

                # Always log OCR timing every 10 frames so we can debug slow performance
                if frame_n % 10 == 0:
                    log.info(
                        "camera %s frame#%d age=%dms ocr=%dms plates=%s route=%s",
                        cid, frame_n, frame_age_ms, ocr_ms,
                        [p for p, _ in plates] if plates else "none",
                        route or "none",
                    )
                # Loud warning if frames are getting stale — exposes backlog
                # problems immediately in the docker logs.
                if frame_age_ms > 800:
                    log.warning(
                        "camera %s STALE frame age=%dms (OCR is slower than camera fps; "
                        "consider OCR_GPU=true / lowering YOLO_IMGSZ)",
                        cid, frame_age_ms,
                    )
                # Save latest frame to disk every 30 frames for visual debugging.
                # Was every 5 — turned out to be a 50-100 ms blocking write that
                # held up the loop on slower disks. 30 keeps the debug image
                # fresh-ish without starving the OCR cycle.
                if frame_n % 30 == 1:
                    try:
                        cv2.imwrite(f"/tmp/debug_frame_{cid}.jpg", frame)
                    except Exception:
                        pass

                # Push frame snapshot to debug panel every ~1 s (was every 5
                # frames ≈ every 150 ms — wasteful when nothing's happening).
                # Always push when we have plates or a route hit.
                if plates or route or frame_n % 30 == 1:
                    snap_b64 = frame_to_jpeg_b64(frame, 320)  # small thumbnail for debug panel
                    _post_live_debug(
                        s,
                        "vision_frame",
                        {
                            "camera_id": cid,
                            "n_reads": len(plates),
                            "reads": [{"plate": p, "confidence": round(c, 4)} for p, c in plates[:16]],
                            "frame_hw": [int(frame.shape[0]), int(frame.shape[1])],
                            "ocr_ms": ocr_ms,
                            "engine": s.PLATE_ENGINE,
                            "stack": s.VISION_STACK,
                            "route": route,
                            "snapshot_b64": snap_b64,
                        },
                        bypass_interval=bool(plates),
                    )

                snap: str | None = None
                posted_any = False
                # Confidence floor: anything below this is treated as noise and
                # never posted to the dashboard. The user's hard rule:
                # "atleast 50% confidence for anything to show up on the log".
                INGEST_CONF_FLOOR = float(s.INGEST_MIN_CONFIDENCE)
                for plate, conf in plates:
                    if conf < INGEST_CONF_FLOOR:
                        log.info(
                            "plate suppress: %s conf=%.2f < %.2f (INGEST_MIN_CONFIDENCE)",
                            plate, conf, INGEST_CONF_FLOOR,
                        )
                        continue
                    if not _dedupe_allow(plate, last, s.DEDUPE_SECONDS, route):
                        log.debug("plate dedupe skip: %s", plate)
                        continue
                    if not _camera_cooldown_allow(plate, cid, recent_cam, s.CAMERA_COOLDOWN_SEC):
                        continue
                    if snap is None:
                        snap = frame_to_jpeg_b64(frame, s.SNAPSHOT_MAX_WIDTH)
                    _post_detection(
                        s, plate, conf, snap or None,
                        camera_id=cid, camera_name=cname, detected_route=route,
                    )
                    _camera_cooldown_mark(plate, cid, recent_cam)
                    posted_any = True

                # Route-only fallback: bus visible (placard readable) but plate
                # OCR returned nothing. Post a route-only sighting so the API
                # can fill the plate from the registry — dashboard renders a
                # yellow triangle to flag the auto-fill.
                if not posted_any and route:
                    if _route_only_dedupe_allow(route, cid, last, max(s.DEDUPE_SECONDS, 6.0)):
                        if snap is None:
                            snap = frame_to_jpeg_b64(frame, s.SNAPSHOT_MAX_WIDTH)
                        _post_detection(
                            s, None, 0.0, snap or None,
                            camera_id=cid, camera_name=cname, detected_route=route,
                        )
                        log.info("route-only post: route=%s camera=%s (no plate this frame)", route, cid)
                if s.PROCESS_INTERVAL_SEC > 0:
                    time.sleep(s.PROCESS_INTERVAL_SEC)
        finally:
            try:
                grabber.stop()
            except Exception:
                pass
            cap.release()
            log.warning("RTSP capture ended; reconnecting…")
            time.sleep(2.0)


def run_webcam_loop(
    s: VisionSettings,
    *,
    camera_id: str | None = None,
    camera_name: str | None = None,
    device_index: int = 0,
    stop_event: threading.Event | None = None,
) -> None:
    """USB webcam pipeline — uses cv2.VideoCapture(device_index), works on Mac/Windows/Linux."""
    last: dict[str, float] = {}
    cid = (camera_id or s.CAMERA_ID).strip()
    cname = (camera_name or s.CAMERA_NAME).strip()
    log.info("Opening webcam [%s]: device index %d", cid, device_index)

    while True:
        if stop_event is not None and stop_event.is_set():
            log.info("Stop requested for camera %s — exiting webcam loop", cid)
            return

        cap = open_webcam(device_index)
        if not cap.isOpened():
            log.warning(
                "Webcam device %d not accessible; retry in 10s. "
                "If running in Docker on Mac/Windows, run vision worker natively on host.",
                device_index,
            )
            if stop_event is not None:
                stop_event.wait(timeout=10.0)
            else:
                time.sleep(10.0)
            continue

        log.info("Webcam device %d opened for camera %s", device_index, cid)
        # Same trick as RTSP: dedicated grabber so OCR loop only sees fresh frames.
        grabber = _LatestFrameBuffer(cap, history=4)
        try:
            frame_n = 0
            last_heartbeat = time.time()
            recent_cam: dict[str, tuple[float, str]] = {}
            stale_count = 0
            startup_deadline = time.time() + 8.0
            strict_indian = s.PLATE_FILTER.strip().lower() == "indian"
            _pool = _get_ocr_pool()
            pending: dict | None = None

            def _submit_frame_wc(_f, _captured_at: float) -> dict:
                t0 = time.time()
                _f_copy = _f.copy()
                return {
                    "frame": _f,
                    "frame_copy": _f_copy,
                    "captured_at": _captured_at,
                    "submitted_at": t0,
                    "plate_fut": _pool.submit(
                        read_plates_from_frame,
                        _f,
                        min_confidence=s.MIN_CONFIDENCE,
                        strict_indian=strict_indian,
                        detection_mode=s.PLATE_DETECTION_MODE.strip().lower(),
                        vision_stack=s.VISION_STACK,
                    ),
                    "route_fut": _pool.submit(_read_route_number, _f_copy),
                }

            while True:
                if stop_event is not None and stop_event.is_set():
                    log.info("Stop requested for camera %s — closing webcam", cid)
                    return

                if pending is None:
                    ok, frame, captured_at = grabber.latest_sharpest(max_age_sec=1.5)
                    if not ok or frame is None:
                        if time.time() > startup_deadline and stale_count > 25:
                            log.warning("Webcam frames went stale — reconnecting")
                            break
                        stale_count += 1
                        time.sleep(0.02)
                        continue
                    stale_count = 0
                    pending = _submit_frame_wc(frame, captured_at)

                current = pending
                pending = None
                frame = current["frame"]
                frame_copy = current["frame_copy"]
                captured_at = current["captured_at"]
                t_ocr = current["submitted_at"]
                _plate_fut = current["plate_fut"]
                _route_fut = current["route_fut"]

                plates = _plate_fut.result()

                # Prefetch the next sharpest frame's OCR while we post.
                ok_n, frame_n2, captured_at_n = grabber.latest_sharpest(max_age_sec=1.5)
                if ok_n and frame_n2 is not None:
                    pending = _submit_frame_wc(frame_n2, captured_at_n)

                frame_age_ms = int((time.time() - captured_at) * 1000)
                frame_n += 1
                now = time.time()
                if now - last_heartbeat >= 30.0:
                    last_heartbeat = now
                    log.info("camera %s webcam alive — frame #%d", cid, frame_n)
                route: str | None = None
                try:
                    route = _route_fut.result(timeout=1.0)
                except Exception as _re:
                    log.debug("route OCR timeout/error: %s", _re)
                ocr_ms = int((time.time() - t_ocr) * 1000)
                # ─────────────────────────────────────────────────────────────

                if frame_n % 10 == 0:
                    log.info(
                        "camera %s webcam frame#%d age=%dms ocr=%dms plates=%s route=%s",
                        cid, frame_n, frame_age_ms, ocr_ms,
                        [p for p, _ in plates] if plates else "none",
                        route or "none",
                    )
                if frame_age_ms > 800:
                    log.warning(
                        "camera %s STALE webcam frame age=%dms — OCR is slower than fps",
                        cid, frame_age_ms,
                    )
                if plates or frame_n % 30 == 0:
                    _post_live_debug(
                        s,
                        "vision_frame_webcam",
                        {
                            "camera_id": cid,
                            "device_index": device_index,
                            "n_reads": len(plates),
                            "reads": [{"plate": p, "confidence": round(c, 4)} for p, c in plates[:16]],
                            "frame_hw": [int(frame.shape[0]), int(frame.shape[1])],
                            "ocr_ms": ocr_ms,
                            "engine": s.PLATE_ENGINE,
                            "stack": s.VISION_STACK,
                            "route": route,
                        },
                    )
                snap: str | None = None
                posted_any = False
                INGEST_CONF_FLOOR = float(s.INGEST_MIN_CONFIDENCE)
                for plate, conf in plates:
                    if conf < INGEST_CONF_FLOOR:
                        log.info(
                            "plate suppress: %s conf=%.2f < %.2f (INGEST_MIN_CONFIDENCE)",
                            plate, conf, INGEST_CONF_FLOOR,
                        )
                        continue
                    if not _dedupe_allow(plate, last, s.DEDUPE_SECONDS, route):
                        continue
                    if not _camera_cooldown_allow(plate, cid, recent_cam, s.CAMERA_COOLDOWN_SEC):
                        continue
                    _camera_cooldown_mark(plate, cid, recent_cam)
                    if snap is None:
                        snap = frame_to_jpeg_b64(frame, s.SNAPSHOT_MAX_WIDTH)
                    _post_detection(s, plate, conf, snap or None, camera_id=cid, camera_name=cname,
                                    detected_route=route)
                    posted_any = True

                # Route-only fallback (webcam path) — see RTSP loop for the rationale.
                if not posted_any and route:
                    if _route_only_dedupe_allow(route, cid, last, max(s.DEDUPE_SECONDS, 6.0)):
                        if snap is None:
                            snap = frame_to_jpeg_b64(frame, s.SNAPSHOT_MAX_WIDTH)
                        _post_detection(
                            s, None, 0.0, snap or None,
                            camera_id=cid, camera_name=cname, detected_route=route,
                        )
                        log.info("route-only post (webcam): route=%s camera=%s", route, cid)
                if s.PROCESS_INTERVAL_SEC > 0:
                    time.sleep(s.PROCESS_INTERVAL_SEC)
        finally:
            try:
                grabber.stop()
            except Exception:
                pass
            cap.release()
            log.warning("Webcam capture ended for device %d; reconnecting…", device_index)
            time.sleep(2.0)


def main_loop() -> None:
    logging.basicConfig(level=logging.DEBUG)
    # Keep noisy libs at INFO
    for _lib in ("httpx", "httpcore", "urllib3", "PIL", "onnxruntime"):
        logging.getLogger(_lib).setLevel(logging.WARNING)
    s = get_settings()
    _configure_compute_threads()
    _announce_worker_online(s)
    engine = s.PLATE_ENGINE.strip().lower()

    if engine == "mock":
        log.warning("PLATE_ENGINE=mock — demo plates only (not for production)")
        run_mock_loop(s)
        return

    mode = (s.VISION_CAMERA_SOURCE or "api").strip().lower() or "api"
    log.info(
        "Vision worker starting: mode=%s API_BASE_URL=%s VISION_STACK=%s PLATE_ENGINE=%s",
        mode,
        s.API_BASE_URL.rstrip("/"),
        s.VISION_STACK,
        s.PLATE_ENGINE,
    )
    if mode == "api":
        from apps.vision_worker.app.multi_camera import run_multi_camera_loop

        log.info(
            "VISION_CAMERA_SOURCE=api — polling /api/internal/vision-cameras; "
            "if API returns 0 cameras, env CAMERA_RTSP_URL fallback is used when set.",
        )
        run_multi_camera_loop(s)
        return

    src_url = (s.CAMERA_RTSP_URL or "").strip()
    if not src_url:
        log.error(
            "VISION_CAMERA_SOURCE=single requires CAMERA_RTSP_URL. "
            "Or use VISION_CAMERA_SOURCE=api and configure cameras in the dashboard API/DB.",
        )
        sys.exit(1)

    if src_url.lower().startswith("webcam:"):
        try:
            dev = int(src_url.split(":", 1)[1])
        except (ValueError, IndexError):
            dev = 0
        log.info("VISION_CAMERA_SOURCE=single using webcam device %d", dev)
        run_webcam_loop(s, device_index=dev)
        return

    run_rtsp_loop(s)
